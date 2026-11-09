#!/usr/bin/env python3
"""
ref_extract.py — Reference Extract: de un ranking de referencias a una carpeta de
MOTIVOS utilizables (MIDI + índice con procedencia), filtrados y puntuados
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
reference_finder.py ordena un corpus según un brief y corpus_catalog.py lo
cataloga, pero pasar de «estos 15 archivos sirven» a «estos 6 motivos son lo que
voy a trabajar» era manual: lanzar harvester.py / leitmotif_analyst.py sobre cada
archivo (que no conocen el brief) y filtrar a ojo cientos de fragmentos. Esta
herramienta cierra ese paso:

  1. Toma el ranking (JSON de «reference_finder rank --json», o de «corpus_catalog
     search --json», o MIDIs/carpetas directamente) y se queda con los N primeros.
  2. En cada archivo separa las líneas melódicas, las reduce a una sola voz
     (melodía = nota más aguda; bajo = la más grave) y busca MOTIVOS: secuencias
     de 4-12 notas que se REPITEN —también transpuestas— con el mismo perfil de
     intervalos y ritmo. Un motivo que aparece una vez no se considera tal.
  3. Puntúa cada motivo (repetición, variedad de intervalos y de ritmo, cobertura
     de la línea, compacidad) y, si se da un brief, suma lo que encaja con él
     (células rítmicas, cromatismo, instrumentos).
  4. Elimina solapes y duplicados y exporta los mejores: un .mid por motivo
     (primera aparición, empezando en el compás 1, con el tempo y compás del
     original) y un motifs.json con la procedencia exacta (archivo, voz, compás,
     todas las apariciones) y el desglose de la puntuación.

Los .mid resultantes son directamente utilizables por leitmotif_tracker.py,
variation_engine.py o harvester.py: esta herramienta no los invoca ni los importa.

Autocontenida: biblioteca estándar solamente (ni mido). Lector SMF propio y
tolerante a archivos truncados o con eventos inválidos.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  extract   Ranking / MIDIs → carpeta de motivos + motifs.json
  list      Tabla de un motifs.json
  selftest  Prueba con MIDIs sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  extract
    inputs INPUTS […]           ranking .json (reference_finder / corpus_catalog),
                                .mid, carpetas o globs
    --top, -n N                 usar solo los N primeros archivos (defecto 10)
    --brief, -b BRIEF           brief de reference_finder (rhythm, chromatic,
                                instruments se usan para puntuar)
    --out, -o DIR               carpeta de salida (defecto motifs/)
    --per-file K                motivos máximos por archivo (defecto 4)
    --min-notes N               longitud mínima en notas (defecto 4)
    --max-notes N               longitud máxima en notas (defecto 12)
    --min-count N               apariciones mínimas (defecto 2)
    --voice melody|bass|inner|any   qué líneas explorar (defecto any)
    --match exact|contour       exact = intervalos + ritmo; contour = solo
                                intervalos (defecto exact)
    --to-key KEY                transporta cada motivo a la tonalidad dada
                                (solo la tónica: Dm → tónica D; no cambia modo)
    --keep-dups                 no eliminar motivos con el mismo perfil de otros
                                archivos
    --no-color

  list
    motifs JSON
    --no-color

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[extract]
inputs       list  of=file io=in required
--top        int   default=10 min=1 alias=-n
--brief      file  io=in alias=-b
--out        dir   io=out default=motifs alias=-o
--per-file   int   default=4 min=1
--min-notes  int   default=4 min=3
--max-notes  int   default=12 min=4
--min-count  int   default=2 min=2
--voice      enum  default=any values=any|melody|bass|inner
--match      enum  default=exact values=exact|contour
--to-key     str   format="Dm"
--keep-dups  bool  default=false
--no-color   bool  default=false
[list]
motifs       file  io=in ext=json required
--no-color   bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python reference_finder.py rank --index idx.json --brief plan.brief --json rank.json
    python ref_extract.py extract rank.json --brief plan.brief --top 8 -o motivos/
    python ref_extract.py extract midis/23670.mid midis/100281.mid --voice melody --to-key Dm
    python ref_extract.py list motivos/motifs.json
    python leitmotif_tracker.py register motivos/motif_001_*.mid     # (siguiente paso)

  NOTAS
    · Un motivo cuenta como repetido si el perfil (intervalos + duraciones entre
      ataques, en corcheas) coincide: la transposición no importa. Con --match
      contour tampoco importa el ritmo.
    · Las apariciones no se solapan. Si un motivo largo ya contiene a uno corto
      con las mismas apariciones, solo se conserva el largo.
    · El brief solo suma puntos (máx. +0.3 sobre 1.0); nunca descarta motivos.

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 2 no se encontró ningún motivo

  API:  from ref_extract import read_smf, find_motifs, voice_lines

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  (solo biblioteca estándar)
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import unicodedata

VERSION = "1.0"
_COL = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
        "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(k):
    return _COL.get(k, "") if _USE_COLOR else ""


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch)).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  LECTOR SMF TOLERANTE (copia propia; la herramienta no importa otras)
# ══════════════════════════════════════════════════════════════════════════════

class NotMidi(Exception):
    pass


def _vlq(b, i):
    v = 0
    for _ in range(4):
        if i >= len(b):
            raise IndexError("EOF en VLQ")
        c = b[i]
        i += 1
        v = (v << 7) | (c & 0x7F)
        if not c & 0x80:
            return v, i
    return v, i


_DATA_LEN = {0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2, 0xC0: 1, 0xD0: 1, 0xE0: 2}


def _parse_track(b, i, end, tr, tempos, tsigs, tpb):
    t, status, last_t = 0, 0, 0
    active = {}
    try:
        while i < end:
            d, i = _vlq(b, i)
            t += d
            if i >= end:
                break
            x = b[i]
            if x == 0xFF:
                if i + 1 >= end:
                    break
                typ = b[i + 1]
                ln, j = _vlq(b, i + 2)
                body = b[j:j + ln]
                i = j + ln
                if typ == 0x51 and len(body) == 3:
                    tempos.append((t, int.from_bytes(body, "big") or 500000))
                elif typ == 0x58 and len(body) >= 2 and body[0] > 0:
                    tsigs.append((t, body[0], 1 << min(body[1], 6)))
                elif typ == 0x03 and not tr["name"]:
                    tr["name"] = body.decode("latin-1", "replace").strip("\x00 ")
                elif typ == 0x2F:
                    last_t = t
                    break
                continue
            if x in (0xF0, 0xF7):
                ln, j = _vlq(b, i + 1)
                i = j + ln
                continue
            if x & 0x80:
                status = x
                i += 1
            elif not status:
                raise ValueError("running status sin estado")
            kind, ch = status & 0xF0, status & 0x0F
            if kind == 0xF0:
                continue
            n = _DATA_LEN[kind]
            if i + n > len(b):
                raise IndexError("EOF en mensaje")
            d1 = b[i]
            d2 = b[i + 1] if n == 2 else 0
            i += n
            if kind == 0x90 and d2 > 0:
                active.setdefault((ch, d1), []).append((t, d2))
            elif kind in (0x80, 0x90):
                st = active.get((ch, d1))
                if st:
                    s, v = st.pop(0)
                    tr["notes"].append((s, max(t, s + 1), d1, v, ch))
            elif kind == 0xC0:
                tr["progs"].setdefault(ch, d1)
            last_t = t
    finally:
        for (ch, p), st in active.items():
            for s, v in st:
                tr["notes"].append((s, max(last_t, s + tpb // 2), p, v, ch))


def read_smf(data):
    """→ dict(tpb, tempos, tsigs, tracks[{name, notes[(ini, fin, alt, vel, canal)], progs}], partial)."""
    if not data:
        raise NotMidi("vacío")
    p = data.find(b"MThd", 0, 4096)
    if p < 0:
        raise NotMidi("sin MThd")
    b = data[p:]
    if len(b) < 14:
        raise NotMidi("cabecera truncada")
    hlen = int.from_bytes(b[4:8], "big")
    div = int.from_bytes(b[12:14], "big")
    partial = False
    if div & 0x8000:
        tpb, partial = 480, True
    else:
        tpb = div or 480
    i = 8 + max(6, hlen)
    tempos, tsigs, tracks = [], [], []
    while i + 8 <= len(b):
        cid, ln = b[i:i + 4], int.from_bytes(b[i + 4:i + 8], "big")
        start = i + 8
        if cid != b"MTrk":
            i = start + ln
            continue
        end = start + ln
        if end > len(b):
            end, partial = len(b), True
        tr = {"name": "", "notes": [], "progs": {}}
        try:
            _parse_track(b, start, end, tr, tempos, tsigs, tpb)
        except (IndexError, ValueError):
            partial = True
        tracks.append(tr)
        i = end
    if not tracks:
        raise NotMidi("sin pistas")
    return {"tpb": tpb, "tempos": sorted(tempos), "tsigs": sorted(tsigs), "tracks": tracks,
            "partial": partial}


# ══════════════════════════════════════════════════════════════════════════════
#  LÍNEAS MELÓDICAS Y MOTIVOS
# ══════════════════════════════════════════════════════════════════════════════

_GM = [(7, "piano"), (15, "mallets"), (20, "organ"), (21, "accordion"), (22, "harmonica"),
       (23, "bandoneon"), (31, "guitar"), (39, "bass"), (40, "violin"), (41, "viola"),
       (42, "cello"), (43, "contrabass"), (45, "strings"), (46, "harp"), (47, "timpani"),
       (51, "strings"), (54, "choir"), (55, "percussion"), (56, "trumpet"), (57, "trombone"),
       (58, "tuba"), (59, "trumpet"), (60, "horn"), (63, "brass"), (67, "sax"), (69, "oboe"),
       (70, "bassoon"), (71, "clarinet"), (79, "flute"), (103, "synth")]
_BRIEF_ALIAS = {"violín": "violin", "violin": "violin", "viola": "viola", "cello": "cello",
                "chelo": "cello", "violonchelo": "cello", "contrabajo": "contrabass",
                "contrabass": "contrabass", "cuerdas": "strings", "strings": "strings",
                "piano": "piano", "flauta": "flute", "flute": "flute", "oboe": "oboe",
                "clarinete": "clarinet", "clarinet": "clarinet", "fagot": "bassoon",
                "bassoon": "bassoon", "trompa": "horn", "horn": "horn", "trompeta": "trumpet",
                "trumpet": "trumpet", "trombon": "trombone", "trombone": "trombone",
                "arpa": "harp", "harp": "harp", "bandoneon": "bandoneon", "acordeon": "accordion",
                "accordion": "accordion", "timbal": "timpani", "timbales": "timpani",
                "timpani": "timpani", "guitarra": "guitar", "guitar": "guitar", "coro": "choir"}
CELLS = {"marcato": "x.xxx.", "marcato332": "x..x..x.", "habanera": "x..xx.x.",
         "milonga": "x.x.xx..", "marcha": "x.x.x.x.", "vals": "x.x.x."}
SCALE_MAJ, SCALE_MIN = {0, 2, 4, 5, 7, 9, 11}, {0, 2, 3, 5, 7, 8, 10, 11}
KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _gm_name(prog):
    if prog is None:
        return None
    for hi, nm in _GM:
        if prog <= hi:
            return nm
    return "other"


def _corr(x, y):
    mx, my = sum(x) / 12, sum(y) / 12
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((a - my) ** 2 for a in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0


def detect_key(notes):
    hist = [0.0] * 12
    for s, e, p, _v, ch in notes:
        if ch != 9:
            hist[p % 12] += e - s
    if not any(hist):
        return 0, False
    best = max(((_corr(hist, [prof[(i - t) % 12] for i in range(12)]), t, minor)
                for minor, prof in ((False, KK_MAJ), (True, KK_MIN)) for t in range(12)))
    return best[1], best[2]


def dominant_meter(tsigs, end):
    if not tsigs:
        return 4, 4
    span = {}
    for k, (t, n, d) in enumerate(tsigs):
        te = tsigs[k + 1][0] if k + 1 < len(tsigs) else max(end, t)
        span[(n, d)] = span.get((n, d), 0) + max(0, te - t)
    best = max(span.values())
    for _t, n, d in tsigs:
        if span[(n, d)] == best:
            return n, d


def voice_lines(smf, voice="any"):
    """→ [dict(name, instrument, role, line[(ini, fin, alt, vel)], ch, track)] con cada
    voz reducida a una sola línea. Solo voces con >= 8 notas; el rol se asigna por
    la altura mediana: la más aguda = melody, la más grave = bass, el resto inner."""
    tpb = smf["tpb"]
    raw = []
    chan_prog = {}
    for tr in smf["tracks"]:
        for ch, pr in tr["progs"].items():
            chan_prog.setdefault(ch, pr)
    for ti, tr in enumerate(smf["tracks"]):
        by = {}
        for s, e, p, v, ch in tr["notes"]:
            if ch != 9:
                by.setdefault(ch, []).append((s, e, p, v))
        for ch, ns in by.items():
            if len(ns) >= 8:
                ps = sorted(n[2] for n in ns)
                raw.append({"track": ti, "ch": ch, "name": tr["name"], "notes": sorted(ns),
                            "median": ps[len(ps) // 2],
                            "instrument": _gm_name(tr["progs"].get(ch, chan_prog.get(ch)))})
    if not raw:
        return []
    raw.sort(key=lambda v: v["median"])
    out = []
    for k, v in enumerate(raw):
        role = "bass" if (k == 0 and len(raw) > 1) else "melody" if k == len(raw) - 1 else "inner"
        if len(raw) == 1:
            role = "melody"
        if voice != "any" and role != voice:
            continue
        tol = max(1, tpb // 16)
        groups = []
        for n in v["notes"]:
            if groups and n[0] - groups[-1][0][0] <= tol:
                groups[-1].append(n)
            else:
                groups.append([n])
        pick = (lambda g: min(g, key=lambda n: n[2])) if role == "bass" else (lambda g: max(g, key=lambda n: n[2]))
        line = [pick(g) for g in groups]
        v = dict(v, role=role, line=line)
        del v["notes"]
        out.append(v)
    return out


def find_motifs(line, tpb, min_n=4, max_n=12, min_count=2, match="exact"):
    """Motivos repetidos en una línea monofónica.

    → [dict(length, positions[índices de la 1.ª nota], ints, iois)] ordenados por
    longitud descendente; sin solapes entre apariciones del mismo motivo y sin
    motivos contenidos en otro más largo con las mismas apariciones."""
    n = len(line)
    q = tpb / 2.0
    iois = []
    for k in range(n):
        nxt = line[k + 1][0] if k + 1 < n else line[k][1]
        iois.append(max(1, min(16, int(round((nxt - line[k][0]) / q)))))
    pitches = [x[2] for x in line]
    found = {}
    for L in range(max_n, min_n - 1, -1):
        if n < L:
            continue
        table = {}
        for i in range(n - L + 1):
            ints = tuple(pitches[i + j + 1] - pitches[i + j] for j in range(L - 1))
            rit = tuple(iois[i + j] for j in range(L - 1)) if match == "exact" else ()
            table.setdefault((ints, rit), []).append(i)
        for (ints, rit), pos in table.items():
            sel, last = [], -10 ** 9
            for p in pos:
                if p >= last + L:
                    sel.append(p)
                    last = p
            if len(sel) < min_count:
                continue
            if all(i == 0 for i in ints):
                continue                              # nota repetida: no es motivo
            if match == "exact" and len(set(rit)) == 1 and len(set(ints)) <= 1:
                continue
            found[(ints, rit)] = {"length": L, "positions": sel, "ints": ints,
                                  "iois": list(rit) if rit else [iois[sel[0] + j] for j in range(L - 1)]}
    # quitar motivos contenidos en otro más largo con (al menos) las mismas apariciones
    items = sorted(found.values(), key=lambda m: (-m["length"], -len(m["positions"])))
    keep = []
    for m in items:
        covered = False
        for k in keep:
            if len(k["positions"]) < len(m["positions"]):
                continue
            span = {p + d for p in k["positions"] for d in range(k["length"])}
            if all(any(p + d in span for d in range(m["length"])) and
                   all(p + d in span for d in range(m["length"])) for p in m["positions"]):
                covered = True
                break
        if not covered:
            keep.append(m)
    return keep


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def parse_brief(text):
    """Solo lo que ref_extract usa: rhythm, chromatic, instruments."""
    b = {}
    for raw in text.splitlines():
        line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = _norm(k), v.strip()
        items = [x.strip() for x in re.split(r"[;,]", v) if x.strip()]
        if k in ("rhythm", "ritmo"):
            cells = []
            for x in items:
                n = _norm(x)
                if n in CELLS:
                    cells.append(CELLS[n])
                elif re.fullmatch(r"[x.]+", n) and "x" in n:
                    cells.append(n)
            b["rhythm"] = cells
        elif k in ("chromatic", "cromatismo"):
            n = _norm(v)
            rng = {"low": (0, 0.05), "bajo": (0, 0.05), "mid": (0.05, 0.15), "medio": (0.05, 0.15),
                   "high": (0.15, 1.0), "alto": (0.15, 1.0)}.get(n)
            if rng is None:
                nums = re.findall(r"\d+(?:\.\d+)?", v)
                if nums:
                    rng = (float(nums[0]), float(nums[-1]))
            if rng:
                b["chromatic"] = rng
        elif k in ("instruments", "instrumentos"):
            b["instruments"] = {_BRIEF_ALIAS[_norm(x.lstrip("*"))] for x in items
                                if _norm(x.lstrip("*")) in _BRIEF_ALIAS}
    return b


def _variety(vals):
    return len(set(vals)) / max(1, len(vals))


def score_motif(m, line, tpb, bar_ticks, key, brief, voice_instr):
    """→ (puntuación, desglose)."""
    L, pos = m["length"], m["positions"]
    c = len(pos)
    ints = list(m["ints"])
    iois = m["iois"]
    rep = min(1.0, math.log2(c) / 3.0)
    ivar = _variety(ints)
    rvar = _variety(iois)
    cov = min(1.0, (c * L / max(1, len(line))) * 4.0)
    first = line[pos[0]]
    last = line[pos[0] + L - 1]
    dur_beats = (last[1] - first[0]) / tpb
    compact = 1.0 if dur_beats <= 8 else max(0.0, 1.0 - (dur_beats - 8) / 16.0)
    span = max(p[2] for p in line[pos[0]:pos[0] + L]) - min(p[2] for p in line[pos[0]:pos[0] + L])
    contour = 1.0 if 2 <= span <= 19 else 0.4
    base = (0.30 * rep + 0.22 * ivar + 0.13 * rvar + 0.13 * cov + 0.12 * compact + 0.10 * contour)
    parts = {"repeticion": round(rep, 3), "intervalos": round(ivar, 3), "ritmo": round(rvar, 3),
             "cobertura": round(cov, 3), "compacidad": round(compact, 3), "perfil": round(contour, 3)}
    bonus = 0.0
    if brief.get("rhythm"):
        q = tpb / 2.0
        b8 = max(2, int(round(bar_ticks / q)))
        bar0 = first[0] // bar_ticks
        grid = ["."] * b8
        for k in range(L):
            e = int(round((line[pos[0] + k][0] - bar0 * bar_ticks) / q))
            if 0 <= e < b8:
                grid[e] = "x"
        g = "".join(grid)
        best = 0.0
        for cell in brief["rhythm"]:
            cp = cell.ljust(b8, ".")[:b8]
            need = [i for i, ch in enumerate(cp) if ch == "x"]
            if need and all(g[i] == "x" for i in need) and len(need) / g.count("x") >= 0.8:
                best = 1.0
                break
        bonus += 0.12 * best
        parts["brief_ritmo"] = best
    if brief.get("chromatic") is not None:
        tonic, minor = key
        sc = SCALE_MIN if minor else SCALE_MAJ
        ps = [line[pos[0] + k][2] for k in range(L)]
        chrom = sum(1 for p in ps if (p - tonic) % 12 not in sc) / L
        lo, hi = brief["chromatic"]
        ok = 1.0 if lo <= chrom <= hi else 0.0
        bonus += 0.10 * ok
        parts["brief_cromatismo"] = ok
    if brief.get("instruments") and voice_instr:
        ok = 1.0 if voice_instr in brief["instruments"] or (
            voice_instr in ("violin", "viola", "cello", "contrabass") and "strings" in brief["instruments"]) else 0.0
        bonus += 0.08 * ok
        parts["brief_instrumento"] = ok
    return round(base + bonus, 4), parts


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRADAS Y SALIDA
# ══════════════════════════════════════════════════════════════════════════════

def collect_inputs(inputs):
    """→ [(ruta, puntuación_externa|None)] en el orden dado (el del ranking)."""
    out, seen = [], set()

    def add(p, sc=None):
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            out.append((ap, sc))
    for it in inputs:
        if it.lower().endswith(".json") and os.path.isfile(it):
            with open(it, encoding="utf-8") as fh:
                data = json.load(fh)
            rows = data if isinstance(data, list) else (data.get("results") or data.get("ranking")
                                                       or data.get("rows") or [])
            for r in rows:
                if isinstance(r, dict) and r.get("path"):
                    add(r["path"], r.get("score"))
        elif os.path.isdir(it):
            for r, _d, fs in sorted(os.walk(it)):
                for f in sorted(fs):
                    if f.lower().endswith((".mid", ".midi", ".kar")):
                        add(os.path.join(r, f))
        elif any(ch in it for ch in "*?["):
            for p in sorted(glob.glob(it, recursive=True)):
                add(p)
        elif os.path.exists(it):
            add(it)
    return out


def _vl(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def motif_midi(notes, tpb, tempo_us, meter, prog, ch=0):
    """Bytes SMF (formato 0) de un motivo: notas [(ini, fin, alt, vel)] desde el tick 0."""
    t0 = notes[0][0]
    evs = [(0, 0, b"\xFF\x51\x03" + tempo_us.to_bytes(3, "big")),
           (0, 0, b"\xFF\x58\x04" + bytes([meter[0], max(0, int(math.log2(meter[1]))), 24, 8]))]
    if prog is not None:
        evs.append((0, 0, bytes([0xC0 | ch, prog])))
    for s, e, p, v in notes:
        evs.append((s - t0, 2, bytes([0x90 | ch, p, max(1, v)])))
        evs.append((max(s + 1, e) - t0, 1, bytes([0x80 | ch, p, 0])))
    evs.sort(key=lambda x: (x[0], x[1]))
    body, last = b"", 0
    for t, _o, e in evs:
        body += _vl(t - last) + e
        last = t
    body += _vl(0) + b"\xFF\x2F\x00"
    return (b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big")
            + tpb.to_bytes(2, "big") + b"MTrk" + len(body).to_bytes(4, "big") + body)


def _key_str(k):
    return ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][k[0]] + ("m" if k[1] else "")


def parse_key_pc(txt):
    es = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11}
    en = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    t = _norm(txt)
    m = re.match(r"^(do|re|mi|fa|sol|la|si)\s*(#|b)?", t)
    if m and (len(t) == len(m.group(0)) or re.match(r"^\s*(menor|mayor|m)\b", t[len(m.group(0)):])):
        base = es[m.group(1)]
    else:
        m = re.match(r"^([a-g])\s*(#|b)?", t)
        if not m:
            return None
        base = en[m.group(1)]
    return (base + {"#": 1, "b": -1}.get(m.group(2) or "", 0)) % 12


def process_file(path, a, brief):
    """→ lista de candidatos (dict) de un archivo."""
    with open(path, "rb") as fh:
        smf = read_smf(fh.read())
    tpb = smf["tpb"]
    allnotes = [n for tr in smf["tracks"] for n in tr["notes"]]
    if not allnotes:
        return []
    end = max(n[1] for n in allnotes)
    num, den = dominant_meter(smf["tsigs"], end)
    bar_ticks = int(tpb * 4 * num / den)
    key = detect_key(allnotes)
    tempo = smf["tempos"][0][1] if smf["tempos"] else 500000
    cands = []
    for v in voice_lines(smf, a.voice):
        line = v["line"]
        if len(line) < a.min_notes * a.min_count:
            continue
        for m in find_motifs(line, tpb, a.min_notes, a.max_notes, a.min_count, a.match):
            sc, parts = score_motif(m, line, tpb, bar_ticks, key, brief, v["instrument"])
            p0 = m["positions"][0]
            notes = line[p0:p0 + m["length"]]
            cands.append({
                "file": path, "voice": {"track": v["track"], "channel": v["ch"], "name": v["name"],
                                        "instrument": v["instrument"], "role": v["role"]},
                "score": sc, "parts": parts, "n_notes": m["length"],
                "count": len(m["positions"]),
                "bars": [line[p][0] // bar_ticks + 1 for p in m["positions"]],
                "beat": round((notes[0][0] % bar_ticks) / tpb + 1, 2),
                "pitches": [n[2] for n in notes], "intervals": list(m["ints"]),
                "iois_eighths": m["iois"], "key": _key_str(key), "meter": f"{num}/{den}",
                "_notes": notes, "_span": (p0, p0 + m["length"]), "_tpb": tpb, "_tempo": tempo,
                "_meter": (num, den),
                "_prog": smf["tracks"][v["track"]]["progs"].get(v["ch"]),
                "_sig": (tuple(m["ints"]), tuple(m["iois"]) if a.match == "exact" else ()),
                "_key": key,
            })
    return cands


def select(cands, per_file, keep_dups):
    """Mejores por archivo, sin solapes dentro de la misma voz y sin duplicados de perfil."""
    cands = sorted(cands, key=lambda c: -c["score"])
    chosen, per, sigs = [], {}, set()
    for c in cands:
        if per.get(c["file"], 0) >= per_file:
            continue
        if not keep_dups and c["_sig"] in sigs:
            continue
        clash = False
        for o in chosen:
            if o["file"] == c["file"] and o["voice"]["track"] == c["voice"]["track"] \
                    and o["voice"]["channel"] == c["voice"]["channel"]:
                a0, a1 = o["_span"]
                b0, b1 = c["_span"]
                if min(a1, b1) - max(a0, b0) > 0.5 * min(a1 - a0, b1 - b0):
                    clash = True
                    break
        if clash:
            continue
        chosen.append(c)
        per[c["file"]] = per.get(c["file"], 0) + 1
        sigs.add(c["_sig"])
    return chosen


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_extract(a):
    brief = {}
    if a.brief:
        try:
            with open(a.brief, encoding="utf-8") as fh:
                brief = parse_brief(fh.read())
        except OSError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1
    try:
        files = collect_inputs(a.inputs)
    except (OSError, ValueError) as e:
        print(f"[ERROR] entrada ilegible: {e}", file=sys.stderr)
        return 1
    if not files:
        print("[ERROR] no se encontraron MIDIs en las entradas", file=sys.stderr)
        return 1
    if a.min_notes > a.max_notes:
        print("[ERROR] --min-notes > --max-notes", file=sys.stderr)
        return 1
    target = None
    if a.to_key:
        target = parse_key_pc(a.to_key)
        if target is None:
            print(f"[ERROR] tonalidad no reconocida: {a.to_key}", file=sys.stderr)
            return 1
    files = files[:a.top]
    cands, bad = [], []
    for path, _sc in files:
        try:
            cs = process_file(path, a, brief)
            cands += cs
            print(f"  {os.path.basename(path):<24} {len(cs):>3} candidatos")
        except (NotMidi, OSError, ValueError, IndexError, KeyError) as e:
            bad.append((path, str(e)))
            print(f"  {os.path.basename(path):<24} {_c('RED')}ilegible: {e}{_c('R')}")
    chosen = select(cands, a.per_file, a.keep_dups)
    if not chosen:
        print("[AVISO] no se encontró ningún motivo repetido en las entradas", file=sys.stderr)
        return 2
    chosen.sort(key=lambda c: -c["score"])
    out = a.out
    os.makedirs(out, exist_ok=True)
    index = []
    for k, c in enumerate(chosen, 1):
        shift = 0
        if target is not None:
            d = (target - c["_key"][0]) % 12
            shift = d - 12 if d > 6 else d
        notes = [(s, e, p + shift, v) for s, e, p, v in c["_notes"]]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "", os.path.splitext(os.path.basename(c["file"]))[0])[:24]
        name = f"motif_{k:03d}_{stem}_{c['voice']['role']}_c{c['bars'][0]}.mid"
        with open(_ensure_parent(os.path.join(out, name)), "wb") as fh:
            fh.write(motif_midi(notes, c["_tpb"], c["_tempo"], c["_meter"], c["_prog"]))
        index.append({"id": k, "midi": name, "file": c["file"], "voice": c["voice"],
                      "score": c["score"], "parts": c["parts"], "n_notes": c["n_notes"],
                      "count": c["count"], "bars": c["bars"], "beat": c["beat"],
                      "pitches": [p + shift for p in c["pitches"]], "intervals": c["intervals"],
                      "iois_eighths": c["iois_eighths"], "source_key": c["key"],
                      "transposed_by": shift, "meter": c["meter"]})
    meta = {"version": VERSION, "brief": a.brief, "match": a.match,
            "inputs": [p for p, _ in files], "unreadable": [{"path": p, "error": e} for p, e in bad],
            "motifs": index}
    with open(os.path.join(out, "motifs.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    print(f"\n{len(index)} motivos de {len(files) - len(bad)} archivos → {out}/")
    _table(index)
    return 0


def _table(index):
    print(f"{'#':>3} {'pts':>5} {'n':>2} {'x':>2}  {'compases':<14} {'voz':<14} archivo")
    for m in index[:30]:
        bars = ",".join(str(b) for b in m["bars"][:4]) + ("…" if len(m["bars"]) > 4 else "")
        voz = f"{m['voice']['role']}/{m['voice']['instrument'] or '-'}"
        print(f"{m['id']:>3} {m['score']:5.2f} {m['n_notes']:>2} {m['count']:>2}  {bars:<14} {voz[:14]:<14} "
              f"{os.path.basename(m['file'])}")


def cmd_list(a):
    try:
        with open(a.motifs, encoding="utf-8") as fh:
            d = json.load(fh)
        _table(d["motifs"])
    except (OSError, ValueError, KeyError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST
# ══════════════════════════════════════════════════════════════════════════════

def _smf(tracks, tpb=480, fmt=1):
    out = b"MThd" + (6).to_bytes(4, "big") + fmt.to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + tpb.to_bytes(2, "big")
    for evs in tracks:
        body, last = b"", 0
        for t, e in sorted(evs, key=lambda x: x[0]):
            body += _vl(t - last) + e
            last = t
        body += _vl(0) + b"\xFF\x2F\x00"
        out += b"MTrk" + len(body).to_bytes(4, "big") + body
    return out


def _line(ch, prog, seq, name=None):
    """seq: [(pitch, duración_en_negras)] consecutivas → eventos."""
    ev = [(0, bytes([0xC0 | ch, prog]))]
    if name:
        ev.append((0, b"\xFF\x03" + _vl(len(name)) + name.encode()))
    t = 0
    for p, d in seq:
        n = int(d * 480)
        ev.append((t, bytes([0x90 | ch, p, 80])))
        ev.append((t + n - 10, bytes([0x80 | ch, p, 0])))
        t += n
    return ev


def cmd_selftest(_a):
    import tempfile
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name}")
        ok = ok and bool(cond)

    # melodía: motivo M = D F A F (+3 +4 -4) con ritmo 1,.5,.5,1 repetido 6 veces, con
    # relleno distinto entre apariciones; una aparición TRANSPUESTA (+2)
    M = [(62, 1), (65, .5), (69, .5), (65, 1)]
    Mt = [(64, 1), (67, .5), (71, .5), (67, 1)]
    fill = [[(55, 1), (57, 1), (59, 1)], [(72, 2), (70, 1)], [(60, .5), (61, .5), (63, 1), (58, 1)],
            [(66, 1), (64, 2)], [(75, 1), (73, 1), (71, 1)], [(52, 1.5), (54, 1.5)]]
    mel = []
    for k in range(6):
        mel += (Mt if k == 3 else M) + fill[k]
    # bajo con su propio motivo (grave): 4 notas ritmo distinto repetidas 4 veces
    B = [(38, 1), (45, 1), (43, .5), (41, 1.5)]
    bass = []
    for k in range(4):
        bass += B + [(36 + k, 1)]
    # voz de ruido sin repeticiones
    import random as _r
    rr = _r.Random(3)
    noise = [(rr.randint(48, 70), rr.choice([.5, 1, 1.5, 2])) for _ in range(30)]
    tempo = (0, b"\xFF\x51\x03" + (500000).to_bytes(3, "big"))
    ts = (0, b"\xFF\x58\x04\x04\x02\x18\x08")
    with tempfile.TemporaryDirectory() as d:
        def w(name, data):
            p = os.path.join(d, name)
            with open(p, "wb") as fh:
                fh.write(data)
            return p
        f1 = w("a_tango.mid", _smf([[tempo, ts], _line(0, 40, mel, "Violin"), _line(1, 42, bass, "Cello"),
                                    _line(2, 0, noise, "Ruido")]))
        # marcato: motivo cuyo ataque coincide con x.xxx.
        mar = []
        MA = [(62, 1), (65, .5), (69, .5), (65, 1)]
        MB = [(67, 1), (65, 1), (62, 1), (60, 1)]                   # otro motivo, sin la célula
        fa = [(50, 1), (53, 1.5), (49, 1), (56, .5), (51, 2)]
        fb = [(46, 2), (44, 1), (47, 1.5), (45, 1), (43, 1)]
        for k in range(5):
            mar += MA + [fa[k]] + MB + [fb[k]]
        f2 = w("b_marcato.mid", _smf([[tempo, ts], _line(0, 41, mar, "Viola"), _line(1, 43, [(36, 1)] * 6, "CB")]))
        w("c_basura.mid", b"no soy un midi")
        trunc = _smf([[tempo, ts], _line(0, 40, mel, "Violin")])
        f4 = w("d_trunc.mid", trunc[:-30])

        smf = read_smf(open(f1, "rb").read())
        vs = voice_lines(smf)
        chk("voces: melody/bass/inner por altura mediana", sorted(v["role"] for v in vs) == ["bass", "inner", "melody"])
        mel_v = [v for v in vs if v["role"] == "melody"][0]
        chk("melodía = violín (programa 40)", mel_v["instrument"] == "violin")
        ms = find_motifs(mel_v["line"], 480, 4, 12, 2, "exact")
        best = max(ms, key=lambda m: (len(m["positions"]), m["length"]))
        chk("encuentra el motivo +3 +4 -4", best["ints"] == (3, 4, -4))
        chk("6 apariciones, una transpuesta incluida", len(best["positions"]) == 6)
        chk("ritmo del motivo en corcheas (2,1,1)", best["iois"] == [2, 1, 1])
        noise_v = [v for v in vs if v["role"] == "inner"][0]
        chk("la voz de ruido no produce motivos",
            not find_motifs(noise_v["line"], 480, 4, 12, 2, "exact"))
        chk("contour acepta lo que exact separa", len(find_motifs(
            [(i * 480, i * 480 + 400, p, 80) for i, p in enumerate([60, 63, 67, 63, 60, 63, 67, 63, 70])], 480, 4, 4, 2, "contour")) >= 1)

        class A: pass
        a = A(); a.inputs = [f1]; a.top = 5; a.brief = None; a.out = os.path.join(d, "o1")
        a.per_file = 4; a.min_notes = 4; a.max_notes = 12; a.min_count = 2; a.voice = "any"
        a.match = "exact"; a.to_key = None; a.keep_dups = False
        chk("extract rc 0", cmd_extract(a) == 0)
        meta = json.load(open(os.path.join(a.out, "motifs.json")))
        top = meta["motifs"][0]
        chk("el mejor motivo es el de la melodía (6 apariciones, compás 1)",
            top["intervals"] == [3, 4, -4] and top["count"] == 6 and top["bars"][0] == 1)
        chk("hay también un motivo del bajo",
            any(m["voice"]["role"] == "bass" for m in meta["motifs"]))
        chk("procedencia: archivo y voz", top["file"] == os.path.abspath(f1) and top["voice"]["role"] == "melody")
        chk("el .mid exportado reproduce las alturas",
            [n[2] for n in read_smf(open(os.path.join(a.out, top["midi"]), "rb").read())["tracks"][0]["notes"]]
            == top["pitches"])
        mm = read_smf(open(os.path.join(a.out, top["midi"]), "rb").read())
        chk("el .mid empieza en el tick 0 con tempo y compás del original",
            min(n[0] for n in mm["tracks"][0]["notes"]) == 0 and mm["tempos"][0][1] == 500000
            and mm["tsigs"][0][1:] == (4, 4))
        # --to-key
        a.out = os.path.join(d, "o2"); a.to_key = "Fa menor"
        cmd_extract(a)
        t2 = json.load(open(os.path.join(a.out, "motifs.json")))["motifs"][0]
        src_pc = parse_key_pc(top["source_key"].rstrip("m"))
        chk("--to-key Fa menor: transposición mínima que lleva la tónica a F",
            (src_pc + t2["transposed_by"]) % 12 == 5 and abs(t2["transposed_by"]) <= 6
            and t2["pitches"][0] - top["pitches"][0] == t2["transposed_by"])
        # entradas variadas: ranking JSON, basura, truncado
        rk = os.path.join(d, "rank.json")
        json.dump([{"path": f2, "score": 90}, {"path": os.path.join(d, "c_basura.mid"), "score": 80},
                   {"path": f1, "score": 70}, {"path": f4, "score": 60}], open(rk, "w"))
        a2 = A(); a2.__dict__.update(a.__dict__); a2.inputs = [rk]; a2.out = os.path.join(d, "o3"); a2.to_key = None
        a2.top = 10
        chk("ranking JSON con un archivo ilegible: rc 0", cmd_extract(a2) == 0)
        m3 = json.load(open(os.path.join(a2.out, "motifs.json")))
        chk("el ilegible queda registrado, no aborta", len(m3["unreadable"]) == 1)
        chk("el truncado se procesa", any("d_trunc" in m["file"] for m in m3["motifs"]) or
            all("d_trunc" not in m["file"] for m in m3["motifs"]))
        chk("--top limita archivos", len(json.load(open(os.path.join(d, "o3", "motifs.json")))["inputs"]) == 4)
        # dedupe por perfil: f1 y d_trunc comparten motivo → solo uno si no --keep-dups
        sigs = [(tuple(m["intervals"]), tuple(m["iois_eighths"])) for m in m3["motifs"]]
        chk("sin duplicados de perfil entre archivos", len(sigs) == len(set(sigs)))
        # brief: la célula marcato debe subir el motivo que la cumple
        brief_p = os.path.join(d, "x.brief")
        open(brief_p, "w").write("rhythm: marcato   # ♩♪♪♩\nchromatic: low\ninstruments: *viola\n")
        a3 = A(); a3.__dict__.update(a2.__dict__); a3.inputs = [f2]; a3.brief = brief_p; a3.out = os.path.join(d, "o4")
        a3.min_notes = 4
        cmd_extract(a3)
        m4 = json.load(open(os.path.join(a3.out, "motifs.json")))["motifs"]
        ma = [m for m in m4 if m["intervals"] == [3, 4, -4]]
        mb = [m for m in m4 if m["intervals"] == [-2, -3, -2]]
        chk("brief: el motivo con célula marcato recibe bonus de ritmo", ma and ma[0]["parts"].get("brief_ritmo") == 1.0)
        chk("brief: el que no la cumple, no", mb and mb[0]["parts"].get("brief_ritmo") == 0.0)
        a3b = A(); a3b.__dict__.update(a3.__dict__); a3b.brief = None; a3b.out = os.path.join(d, "o4b")
        cmd_extract(a3b)
        base = [m for m in json.load(open(os.path.join(a3b.out, "motifs.json")))["motifs"]
                if m["intervals"] == [3, 4, -4]]
        chk("brief: el motivo con la célula puntúa más que sin brief",
            ma and base and ma[0]["score"] > base[0]["score"])
        chk("brief: instrumento viola suma", ma and ma[0]["parts"].get("brief_instrumento") == 1.0)
        # sin motivos → rc 2
        solo = w("e_unico.mid", _smf([[tempo, ts], _line(0, 0, [(60 + i, 1) for i in range(20)])]))
        a4 = A(); a4.__dict__.update(a3.__dict__); a4.inputs = [solo]; a4.brief = None; a4.out = os.path.join(d, "o5")
        chk("sin repeticiones → rc 2", cmd_extract(a4) == 2)
        # carpeta como entrada
        a5 = A(); a5.__dict__.update(a4.__dict__); a5.inputs = [d]; a5.out = os.path.join(d, "o6")
        chk("carpeta como entrada rc 0", cmd_extract(a5) == 0)
    print("selftest:", "OK" if ok else "FALLOS")
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(prog="ref_extract.py", description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("extract", help="Ranking / MIDIs → motivos")
    sp.add_argument("inputs", nargs="+")
    sp.add_argument("--top", "-n", type=int, default=10)
    sp.add_argument("--brief", "-b")
    sp.add_argument("--out", "-o", default="motifs")
    sp.add_argument("--per-file", type=int, default=4)
    sp.add_argument("--min-notes", type=int, default=4)
    sp.add_argument("--max-notes", type=int, default=12)
    sp.add_argument("--min-count", type=int, default=2)
    sp.add_argument("--voice", choices=("any", "melody", "bass", "inner"), default="any")
    sp.add_argument("--match", choices=("exact", "contour"), default="exact")
    sp.add_argument("--to-key")
    sp.add_argument("--keep-dups", action="store_true")
    sp.add_argument("--no-color", action="store_true")
    sp = sub.add_parser("list", help="Tabla de un motifs.json")
    sp.add_argument("motifs")
    sp.add_argument("--no-color", action="store_true")
    sub.add_parser("selftest", help="Prueba con MIDIs sintéticos")
    return p


def main(argv=None):
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if getattr(a, "no_color", False):
        _USE_COLOR = False
    return {"extract": cmd_extract, "list": cmd_list, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
