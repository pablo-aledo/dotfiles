#!/usr/bin/env python3
"""
corpus_catalog.py — Corpus Catalog: catálogo legible, incremental y consultable de
un corpus MIDI (título, instrumentos, tonalidad, compás, tempo, familia, duplicados
y causa exacta de cada archivo ilegible)
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Un corpus de miles de MIDIs con nombres numéricos no dice qué contiene cada
archivo. corpus_explorer.py vectoriza el corpus para buscar por intención y
reference_finder.py puntúa contra un brief, pero ninguno deja un CATÁLOGO que se
pueda leer, filtrar, comparar entre ejecuciones o auditar. Esta herramienta:

  · Lee cada MIDI con un lector SMF propio y tolerante: salva los archivos con
    cabecera envuelta (RIFF/RMID), truncados, con firma de tonalidad inválida o
    con eventos corruptos al final (se queda con las notas leídas y lo marca como
    «parcial»). Solo se rechaza lo que de verdad no es MIDI o no tiene notas, y se
    registra la CLASE del fallo.
  · Extrae por archivo: título, instrumentos (programas GM + nombres de pista),
    tonalidad (Krumhansl-Schmuckler), compás dominante, tempo, duración, nº de
    notas y voces, extensión, batería, familia (tango, vals, fuga, orquesta,
    cámara, piano solo…) y dos huellas: del fichero y de las notas (esta última
    detecta el mismo contenido con otro empaquetado).
  · Es incremental: --update solo reprocesa lo que cambió (tamaño + fecha).
  · Se consulta con filtros (tonalidad, instrumento, tempo, duración, familia…) y
    texto libre; compara dos catálogos (diff) y localiza duplicados (dups).

Autocontenida: biblioteca estándar solamente (ni mido). No importa ninguna otra
herramienta de la suite; el catálogo es un JSON que otras pueden leer.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  build     Cataloga un corpus (incremental con --update) → JSON (+ CSV opcional)
  search    Filtra y busca en un catálogo
  show      Ficha completa de un archivo del catálogo
  stats     Resumen: familias, tonalidades, compases, instrumentos, errores
  diff      Qué se añadió, quitó o cambió entre dos catálogos
  dups      Duplicados exactos (fichero) y de contenido (notas)
  selftest  Comprueba lector, clasificación y búsqueda con MIDIs sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  build
    paths PATHS […]             ficheros, globs o carpetas (recursivo)
    --out, -o OUT               catálogo JSON (defecto corpus_catalog.json)
    --update                    reutiliza las fichas de OUT si tamaño y fecha
                                no han cambiado
    --csv CSV                   exporta además una tabla CSV
    --jobs, -j N                procesos en paralelo (defecto 4)
    --no-color

  search
    catalog CATALOG
    query […]                   palabras libres (todas deben aparecer en título,
                                nombre, instrumentos, familia o tonalidad)
    --key KEYS                  tonalidades: «Dm,F», «Re menor», «F#m»
    --instrument NAMES          «viola,cello» (todos); prefijo ~ = alguno de ellos
    --family FAMILIES           «tango,vals»
    --tempo LO-HI               BPM
    --duration LO-HI            segundos
    --meter M                   «4/4,3/4»
    --no-drums                  excluye archivos con batería
    --min-notes N
    --top TOP                   filas (defecto 25)
    --json JSON                 guarda los resultados
    --no-color

  show
    catalog CATALOG
    target                      id, ruta o trozo de nombre
    --no-color

  stats
    catalog CATALOG
    --no-color

  diff
    old OLD
    new NEW
    --no-color

  dups
    catalog CATALOG
    --by file|notes             huella a usar (defecto notes)
    --no-color

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[build]
paths        list  of=file io=in required
--out        file  io=out ext=json default=corpus_catalog.json alias=-o
--update     bool  default=false
--csv        file  io=out ext=csv
--jobs       int   default=4 min=1 alias=-j
--no-color   bool  default=false
[search]
catalog      file  io=in ext=json required
query        list  of=str
--key        str   format="Dm,F"
--instrument str   format="viola,cello"
--family     str   format="tango,vals"
--tempo      str   format="LO-HI" unit=BPM
--duration   str   format="LO-HI" unit=s
--meter      str   format="4/4,3/4"
--no-drums   bool  default=false
--min-notes  int   default=0 min=0
--top        int   default=25 min=1
--json       file  io=out ext=json
--no-color   bool  default=false
[show]
catalog      file  io=in ext=json required
target       str   required
--no-color   bool  default=false
[stats]
catalog      file  io=in ext=json required
--no-color   bool  default=false
[diff]
old          file  io=in ext=json required
new          file  io=in ext=json required
--no-color   bool  default=false
[dups]
catalog      file  io=in ext=json required
--by         enum  default=notes values=file|notes
--no-color   bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python corpus_catalog.py build midis/ -o catalogo.json --csv catalogo.csv -j 8
    python corpus_catalog.py build midis/ -o catalogo.json --update
    python corpus_catalog.py search catalogo.json tango --instrument viola --key "Re menor"
    python corpus_catalog.py search catalogo.json --family orquesta --tempo 50-80 --no-drums
    python corpus_catalog.py stats catalogo.json
    python corpus_catalog.py dups catalogo.json --by notes
    python corpus_catalog.py diff catalogo_viejo.json catalogo.json

  CLASES DE FALLO (campo «errors» del catálogo)
    no-midi (sin MThd), sin-notas, vacío, ilegible. Los archivos truncados o con
    eventos inválidos NO fallan: se catalogan con «partial»: true.

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 2 sin archivos

  API:  from corpus_catalog import read_smf, extract_record

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  (solo biblioteca estándar)
"""
import argparse
import bisect
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ProcessPoolExecutor

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
#  LECTOR SMF TOLERANTE
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


def read_smf(data):
    """Parsea un SMF tolerando cabecera envuelta, truncados y eventos inválidos.

    → dict(tpb, tempos[(tick, µs)], tsigs[(tick, n, d)], tracks[{name, notes
    [(ini, fin, altura, canal)], progs{canal: programa}, texts[str]}], partial,
    fmt, ntrk). Lanza NotMidi si no hay cabecera MThd."""
    if not data:
        raise NotMidi("vacío")
    p = data.find(b"MThd", 0, 4096)
    if p < 0:
        raise NotMidi("sin MThd")
    b = data[p:]
    if len(b) < 14:
        raise NotMidi("cabecera truncada")
    hlen = int.from_bytes(b[4:8], "big")
    fmt = int.from_bytes(b[8:10], "big")
    ntrk = int.from_bytes(b[10:12], "big")
    div = int.from_bytes(b[12:14], "big")
    partial = False
    if div & 0x8000:                         # SMPTE: se aproxima con 480 tpb
        tpb, partial = 480, True
    else:
        tpb = div or 480
    i = 8 + max(6, hlen)
    tempos, tsigs, tracks = [], [], []
    while i + 8 <= len(b):
        cid = b[i:i + 4]
        ln = int.from_bytes(b[i + 4:i + 8], "big")
        body_start = i + 8
        if cid != b"MTrk":                   # chunk ajeno: se salta
            i = body_start + ln
            continue
        end = body_start + ln
        if end > len(b):
            end, partial = len(b), True
        tr = {"name": "", "notes": [], "progs": {}, "texts": []}
        try:
            _parse_track(b, body_start, end, tr, tempos, tsigs, tpb)
        except (IndexError, ValueError):
            partial = True
        tracks.append(tr)
        i = end
    if not tracks:
        raise NotMidi("sin pistas")
    return {"tpb": tpb, "tempos": sorted(tempos), "tsigs": sorted(tsigs), "tracks": tracks,
            "partial": partial, "fmt": fmt, "ntrk": ntrk}


_DATA_LEN = {0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2, 0xC0: 1, 0xD0: 1, 0xE0: 2}


def _parse_track(b, i, end, tr, tempos, tsigs, tpb):
    t, status = 0, 0
    active = {}
    last_t = 0
    try:
        while i < end:
            d, i = _vlq(b, i)
            t += d
            if i >= end:
                break
            x = b[i]
            if x == 0xFF:                                    # meta
                if i + 1 >= end:
                    break
                typ = b[i + 1]
                ln, j = _vlq(b, i + 2)
                body = b[j:j + ln]
                i = j + ln
                if typ == 0x51 and len(body) == 3:
                    tempos.append((t, int.from_bytes(body, "big") or 500000))
                elif typ == 0x58 and len(body) >= 2:
                    n, dd = body[0], 1 << min(body[1], 6)
                    if n > 0:
                        tsigs.append((t, n, dd))
                elif typ == 0x03 and not tr["name"]:
                    tr["name"] = body.decode("latin-1", "replace").strip("\x00 ")
                elif typ in (0x01, 0x04) and len(tr["texts"]) < 6:
                    tr["texts"].append(body.decode("latin-1", "replace").strip("\x00 "))
                elif typ == 0x2F:
                    last_t = t
                    break
                continue
            if x in (0xF0, 0xF7):                            # sysex
                ln, j = _vlq(b, i + 1)
                i = j + ln
                continue
            if x & 0x80:
                status = x
                i += 1
            elif not status:
                raise ValueError("running status sin estado")
            kind, ch = status & 0xF0, status & 0x0F
            if kind == 0xF0:                                  # mensajes de sistema sueltos
                continue
            n = _DATA_LEN[kind]
            if i + n > len(b):
                raise IndexError("EOF en mensaje")
            d1 = b[i]
            d2 = b[i + 1] if n == 2 else 0
            i += n
            if kind == 0x90 and d2 > 0:
                active.setdefault((ch, d1), []).append(t)
            elif kind == 0x80 or kind == 0x90:
                st = active.get((ch, d1))
                if st:
                    s = st.pop(0)
                    tr["notes"].append((s, max(t, s + 1), d1, ch))
            elif kind == 0xC0:
                tr["progs"].setdefault(ch, d1)
            last_t = t
    finally:
        for (ch, p), st in active.items():                    # notas sin cerrar
            for s in st:
                tr["notes"].append((s, max(last_t, s + tpb // 2), p, ch))


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE CARACTERÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _corr(x, y):
    mx, my = sum(x) / 12, sum(y) / 12
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((a - my) ** 2 for a in y))
    if sx == 0 or sy == 0:
        return 0.0
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def detect_key(hist):
    best = (-2.0, 0, "maj")
    for mode, prof in (("maj", KK_MAJ), ("min", KK_MIN)):
        for t in range(12):
            r = _corr(hist, [prof[(i - t) % 12] for i in range(12)])
            if r > best[0]:
                best = (r, t, mode)
    r, t, mode = best
    flats = (t in (0, 2, 3, 5, 7, 10)) if mode == "min" else (t in (1, 3, 5, 8, 10))
    return (FLAT if flats else SHARP)[t] + ("m" if mode == "min" else ""), round(r, 3)


def gm_name(prog, ch):
    if ch == 9:
        return "drums"
    if prog is None:
        return None
    p = prog
    if p in (6,):
        return "harpsichord"
    if p <= 7:
        return "piano"
    if p <= 15:
        return "mallets"
    if p <= 20:
        return "organ"
    if p == 21:
        return "accordion"
    if p == 22:
        return "harmonica"
    if p == 23:
        return "bandoneon"
    if p <= 31:
        return "guitar"
    if p <= 39:
        return "bass"
    if p in (40, 41, 42, 43, 46, 47):
        return {40: "violin", 41: "viola", 42: "cello", 43: "contrabass", 46: "harp", 47: "timpani"}[p]
    if p <= 51:
        return "strings"
    if p <= 54:
        return "choir"
    if p == 55:
        return "percussion"
    if p in (56, 59):
        return "trumpet"
    if p == 57:
        return "trombone"
    if p == 58:
        return "tuba"
    if p == 60:
        return "horn"
    if p <= 63:
        return "brass"
    if p <= 67:
        return "sax"
    if p in (68, 69):
        return "oboe"
    if p == 70:
        return "bassoon"
    if p == 71:
        return "clarinet"
    if p <= 79:
        return "flute"
    if p <= 103:
        return "synth"
    if p <= 111:
        return "ethnic"
    if p <= 119:
        return "percussion"
    return "sfx"


_NAME_ALIASES = {
    "violin": "violin", "violín": "violin", "viola": "viola", "cello": "cello", "chelo": "cello",
    "violonchelo": "cello", "violoncello": "cello", "contrabass": "contrabass",
    "contrabajo": "contrabass", "bandoneon": "bandoneon", "bandoneón": "bandoneon",
    "accordion": "accordion", "acordeon": "accordion", "acordeón": "accordion",
    "flute": "flute", "flauta": "flute", "oboe": "oboe", "clarinet": "clarinet",
    "clarinete": "clarinet", "bassoon": "bassoon", "fagot": "bassoon", "horn": "horn",
    "trompa": "horn", "trumpet": "trumpet", "trompeta": "trumpet", "trombone": "trombone",
    "tuba": "tuba", "harp": "harp", "arpa": "harp", "timpani": "timpani", "timbal": "timpani",
    "timbales": "timpani", "piano": "piano", "organ": "organ", "guitar": "guitar",
    "guitarra": "guitar", "choir": "choir", "coro": "choir", "strings": "strings",
    "cuerdas": "strings",
}
STRING_SET = {"violin", "viola", "cello", "contrabass", "strings", "harp"}
WIND_SET = {"flute", "oboe", "clarinet", "bassoon", "sax"}
BRASS_SET = {"horn", "trumpet", "trombone", "tuba", "brass"}
PERC_SET = {"timpani", "percussion", "mallets"}

_TITLE_FAMILIES = [
    (r"tango|piazzolla|gardel", "tango"), (r"milonga", "milonga"),
    (r"vals\b|waltz|valse", "vals"), (r"fug(?:a|ue)\b", "fuga"), (r"sonat", "sonata"),
    (r"concert|concierto", "concierto"), (r"symph|sinfon", "sinfonia"),
    (r"quartet|cuarteto|quatuor", "cuarteto"), (r"\btrio\b|\btrío\b", "trio"),
    (r"\bmass\b|misa|requiem", "misa"), (r"chorale|choral|coral|himno|hymn", "coral"),
    (r"carol|villancico|noel|christmas", "villancico"), (r"jazz", "jazz"), (r"blues", "blues"),
    (r"\brock\b", "rock"), (r"march|marcha", "marcha"), (r"nocturn", "nocturno"),
    (r"prelud|preludio", "preludio"), (r"etude|estudio|study", "estudio"),
    (r"overture|obertura", "obertura"), (r"minuet|minueto", "minueto"),
]


def classify(rec, title_hint):
    fam = set()
    t = _norm(title_hint)
    for rx, name in _TITLE_FAMILIES:
        if re.search(rx, t):
            fam.add(name)
    ins = set(rec["instruments"]) - {"drums"}
    if ins & {"bandoneon"} or (ins & {"accordion"} and "tango" in fam):
        fam.add("tango") if "tango" in fam or "bandoneon" in ins else None
    if rec["meter"] in ("3/4", "6/8", "3/8") and "vals" in fam:
        fam.add("vals")
    groups = sum(bool(ins & s) for s in (STRING_SET, WIND_SET, BRASS_SET, PERC_SET))
    if groups >= 3 and len(ins) >= 6:
        fam.add("orquesta")
    elif ins and ins <= STRING_SET and len(ins) >= 2:
        fam.add("cuerdas")
    if ins == {"piano"}:
        fam.add("piano_solo")
    if 2 <= rec["n_voices"] <= 6 and not rec["drums"] and "orquesta" not in fam:
        fam.add("camara")
    if rec["drums"] and ins & {"bass", "guitar"}:
        fam.add("banda")
    if ins and ins <= {"choir"}:
        fam.add("coral")
    return sorted(fam)


def _dominant_meter(tsigs, end):
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


def _seconds(tick, tpb, tempos):
    sec, last_t, tempo = 0.0, 0, 500000
    for t, tp in tempos:
        if t >= tick:
            break
        sec += (t - last_t) * tempo / 1e6 / tpb
        last_t, tempo = t, tp
    return sec + (tick - last_t) * tempo / 1e6 / tpb


def extract_record(path, data=None):
    """→ ficha del archivo. Lanza NotMidi/ValueError con la clase del fallo."""
    if data is None:
        with open(path, "rb") as fh:
            data = fh.read()
    smf = read_smf(data)
    tpb, tracks = smf["tpb"], smf["tracks"]
    notes = [(s, e, p, ch, ti) for ti, tr in enumerate(tracks) for (s, e, p, ch) in tr["notes"]]
    if not notes:
        raise ValueError("sin-notas")
    pitched = [n for n in notes if n[3] != 9]
    drums = [n for n in notes if n[3] == 9]
    last = max(n[1] for n in notes)
    dur = _seconds(last, tpb, smf["tempos"])
    bpm = 60.0 * (last / tpb) / dur if dur > 0 else 120.0
    num, den = _dominant_meter(smf["tsigs"], last)
    hist = [0.0] * 12
    for s, e, p, ch, ti in pitched:
        hist[p % 12] += e - s
    key, key_r = detect_key(hist) if sum(hist) > 0 else (None, 0.0)

    # instrumentos: programa por (pista, canal) con respaldo por canal en cualquier pista
    chan_prog = {}
    for tr in tracks:
        for ch, pr in tr["progs"].items():
            chan_prog.setdefault(ch, pr)
    ins, voices = set(), {}
    for s, e, p, ch, ti in notes:
        voices.setdefault((ti, ch), []).append(p)
    nv = 0
    for (ti, ch), ps in voices.items():
        pr = tracks[ti]["progs"].get(ch, chan_prog.get(ch))
        nm = gm_name(pr, ch)
        tname = _norm(tracks[ti]["name"])
        for w in re.findall(r"[a-zñ]+", tname):
            if w in _NAME_ALIASES and ch != 9:
                ins.add(_NAME_ALIASES[w])
        if nm:
            ins.add(nm)
        if ch != 9 and len(ps) >= 8:
            nv += 1
    title = ""
    for tr in tracks:
        if tr["name"] and not re.fullmatch(r"(track|pista)\s*\d*", tr["name"].strip(), re.I):
            title = tr["name"]
            break
    if not title:
        title = next((tx for tr in tracks for tx in tr["texts"] if len(tx) > 2), "")
    pp = [n[2] for n in pitched]
    rec = {
        "path": os.path.abspath(path), "name": os.path.basename(path), "size": len(data),
        "title": title[:80], "duration_s": round(dur, 1), "tempo_bpm": round(bpm, 1),
        "meter": f"{num}/{den}", "key": key, "key_r": key_r,
        "bars": int(math.ceil(last / max(1, int(tpb * 4 * num / den)))),
        "n_notes": len(pitched), "n_voices": nv, "instruments": sorted(ins),
        "drums": bool(drums), "low": min(pp) if pp else None, "high": max(pp) if pp else None,
        "partial": smf["partial"],
        "fhash": hashlib.sha1(data).hexdigest()[:16],
        "nhash": hashlib.sha1(repr(sorted(((s * 480) // tpb, p) for s, _e, p, _c, _t in pitched)).encode())
                 .hexdigest()[:16],
    }
    rec["family"] = classify(rec, title + " " + os.path.splitext(os.path.basename(path))[0])
    return rec


def _work(path):
    """→ (path, ficha | None, clase_de_fallo, mensaje). Nunca lanza."""
    try:
        st = os.stat(path)
        rec = extract_record(path)
        rec["mtime"] = int(st.st_mtime)
        return path, rec, None, ""
    except NotMidi as e:
        cls = "vacio" if "vac" in str(e) else "no-midi"
        return path, None, cls, str(e)
    except ValueError as e:
        return path, None, "sin-notas" if "sin-notas" in str(e) else "ilegible", str(e)
    except Exception as e:                        # noqa: BLE001 — nunca abortar un corpus
        return path, None, "ilegible", f"{type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRADAS, CONSULTAS
# ══════════════════════════════════════════════════════════════════════════════

def _collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for r, _d, fs in os.walk(p):
                out += [os.path.join(r, f) for f in fs if f.lower().endswith((".mid", ".midi", ".kar", ".rmi"))]
        elif any(ch in p for ch in "*?["):
            out += glob.glob(p, recursive=True)
        elif os.path.exists(p):
            out.append(p)
    return sorted(set(os.path.abspath(x) for x in out))


def _load(path):
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if not isinstance(d, dict) or "files" not in d:
        raise ValueError("no es un catálogo de corpus_catalog.py")
    return d


_ES = {"do": "C", "re": "D", "mi": "E", "fa": "F", "sol": "G", "la": "A", "si": "B"}


def parse_key(txt):
    """«Dm», «F#m», «Re menor», «D minor», «Bb» → nombre canónico como el catálogo."""
    t = _norm(txt)
    m = re.match(r"^(do|re|mi|fa|sol|la|si)\s*(#|b|sostenido|bemol)?\s*(menor|mayor|m|min|maj)?$", t)
    if m:
        letter, acc, mode = _ES[m.group(1)], m.group(2) or "", m.group(3) or ""
    else:
        m = re.match(r"^([a-g])\s*(#|b)?\s*(minor|major|min|maj|m)?$", t)
        if not m:
            return None
        letter, acc, mode = m.group(1).upper(), m.group(2) or "", m.group(3) or ""
    acc = {"sostenido": "#", "bemol": "b"}.get(acc, acc)
    pc = ("C D EF G A B".replace(" ", "")).find(letter)
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    pc = (base + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12
    minor = mode in ("menor", "m", "min", "minor")
    flats = (pc in (0, 2, 3, 5, 7, 10)) if minor else (pc in (1, 3, 5, 8, 10))
    return (FLAT if flats else SHARP)[pc] + ("m" if minor else "")


def _rng(s):
    nums = re.findall(r"\d+(?:\.\d+)?", s or "")
    if not nums:
        return None
    a = float(nums[0]); b = float(nums[1]) if len(nums) > 1 else a
    return (min(a, b), max(a, b))


def search_records(files, query=(), key=None, instrument=None, family=None, tempo=None,
                   duration=None, meter=None, no_drums=False, min_notes=0):
    keys = [parse_key(k) for k in re.split(r"[;,]", key)] if key else []
    keys = {k for k in keys if k}
    ins_any = instrument.startswith("~") if instrument else False
    ins = [_norm(x).lstrip("~") for x in re.split(r"[;,]", instrument or "") if x.strip()]
    fam = {_norm(x) for x in re.split(r"[;,]", family or "") if x.strip()}
    meters = {x.strip() for x in re.split(r"[;,]", meter or "") if x.strip()}
    tr, dr = _rng(tempo), _rng(duration)
    toks = [_norm(q) for q in query]
    out = []
    for r in files.values():
        if keys and r.get("key") not in keys:
            continue
        have = set(r["instruments"])
        if ins and not (any(i in have for i in ins) if ins_any else all(i in have for i in ins)):
            continue
        if fam and not (fam & set(r["family"])):
            continue
        if tr and not (tr[0] <= r["tempo_bpm"] <= tr[1]):
            continue
        if dr and not (dr[0] <= r["duration_s"] <= dr[1]):
            continue
        if meters and r["meter"] not in meters:
            continue
        if no_drums and r["drums"]:
            continue
        if r["n_notes"] < min_notes:
            continue
        hay = _norm(" ".join([r["title"], r["name"], " ".join(r["instruments"]),
                              " ".join(r["family"]), r.get("key") or ""]))
        if any(t not in hay for t in toks):
            continue
        out.append(r)
    out.sort(key=lambda r: (-r["n_voices"], -r["n_notes"], r["name"]))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_build(a):
    paths = _collect(a.paths)
    if not paths:
        print("[ERROR] no se encontraron archivos MIDI", file=sys.stderr)
        return 2
    prev = {}
    if a.update and os.path.exists(a.out):
        try:
            prev = _load(a.out)
        except (ValueError, OSError):
            print(f"[AVISO] {a.out} no es un catálogo válido; se rehace desde cero", file=sys.stderr)
    files, errors, todo, reused = {}, {}, [], 0
    for p in paths:
        old = prev.get("files", {}).get(p)
        olde = prev.get("errors", {}).get(p)
        try:
            st = os.stat(p)
        except OSError:
            continue
        if old and old.get("size") == st.st_size and old.get("mtime") == int(st.st_mtime):
            files[p] = old
            reused += 1
        elif olde and olde.get("size") == st.st_size and olde.get("mtime") == int(st.st_mtime):
            errors[p] = olde
            reused += 1
        else:
            todo.append(p)
    t0 = time.time()
    if todo:
        if a.jobs > 1 and len(todo) > 8:
            with ProcessPoolExecutor(max_workers=a.jobs) as ex:
                results = ex.map(_work, todo, chunksize=max(1, len(todo) // (a.jobs * 8)))
                results = list(results)
        else:
            results = [_work(p) for p in todo]
        for p, rec, cls, msg in results:
            if rec:
                files[p] = rec
            else:
                try:
                    st = os.stat(p)
                    errors[p] = {"class": cls, "msg": msg, "size": st.st_size, "mtime": int(st.st_mtime)}
                except OSError:
                    errors[p] = {"class": cls, "msg": msg}
    cat = {"version": VERSION, "created": int(time.time()), "n_files": len(files),
           "files": files, "errors": errors}
    with open(_ensure_parent(a.out), "w", encoding="utf-8") as fh:
        json.dump(cat, fh, ensure_ascii=False)
    part = sum(1 for r in files.values() if r.get("partial"))
    print(f"Catalogados {len(files)} archivos ({reused} reutilizados, {len(todo)} procesados en "
          f"{time.time() - t0:.1f}s); {len(errors)} ilegibles; {part} parciales → {a.out}")
    if errors:
        by = {}
        for e in errors.values():
            by[e["class"]] = by.get(e["class"], 0) + 1
        print("  fallos: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    if a.csv:
        cols = ["name", "title", "key", "meter", "tempo_bpm", "duration_s", "n_notes", "n_voices",
                "instruments", "family", "drums", "partial", "path"]
        with open(_ensure_parent(a.csv), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for r in files.values():
                w.writerow([";".join(r[c]) if isinstance(r[c], list) else r[c] for c in cols])
        print(f"  → {a.csv}")
    return 0 if files else 2


def _row(r):
    return (f"{r['name'][:22]:<22} {r['title'][:24]:<24} {r.get('key') or '-':<4} {r['meter']:<5} "
            f"{r['tempo_bpm']:>5.0f} {r['duration_s']:>6.0f}s {r['n_voices']:>2}v  "
            f"{','.join(r['instruments'])[:34]:<34} {','.join(r['family'])[:18]}")


def cmd_search(a):
    try:
        cat = _load(a.catalog)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    res = search_records(cat["files"], a.query, a.key, a.instrument, a.family, a.tempo,
                         a.duration, a.meter, a.no_drums, a.min_notes)
    print(f"{len(res)} coincidencias de {len(cat['files'])}")
    for r in res[:a.top]:
        print(_row(r))
    if a.json:
        with open(_ensure_parent(a.json), "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print(f"  → {a.json}")
    return 0 if res else 2


def _find(cat, target):
    t = _norm(target)
    hits = [r for p, r in cat["files"].items() if t == _norm(r["name"]) or t == _norm(os.path.splitext(r["name"])[0])
            or t == _norm(p)]
    return hits or [r for p, r in cat["files"].items() if t in _norm(p)]


def cmd_show(a):
    try:
        cat = _load(a.catalog)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    hits = _find(cat, a.target)
    if not hits:
        err = [p for p in cat["errors"] if _norm(a.target) in _norm(p)]
        if err:
            e = cat["errors"][err[0]]
            print(f"{err[0]}\n  ilegible: {e['class']} — {e['msg']}")
            return 0
        print("[ERROR] no encontrado en el catálogo", file=sys.stderr)
        return 1
    for r in hits[:5]:
        print(f"{_c('B')}{r['name']}{_c('R')}  {r['path']}")
        for k in ("title", "key", "meter", "tempo_bpm", "duration_s", "bars", "n_notes", "n_voices",
                  "instruments", "family", "drums", "low", "high", "partial", "fhash", "nhash"):
            v = r.get(k)
            print(f"  {k:<11} {', '.join(v) if isinstance(v, list) else v}")
    if len(hits) > 5:
        print(f"  … y {len(hits) - 5} más")
    return 0


def cmd_stats(a):
    try:
        cat = _load(a.catalog)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    fs = list(cat["files"].values())
    n = len(fs)
    print(f"{_c('B')}{n}{_c('R')} archivos · {len(cat['errors'])} ilegibles · "
          f"{sum(1 for r in fs if r.get('partial'))} parciales · "
          f"{sum(r['duration_s'] for r in fs) / 3600:.1f} h de música")

    def top(title, counter, k=10):
        items = sorted(counter.items(), key=lambda kv: -kv[1])[:k]
        print(f"\n{_c('CYA')}{title}{_c('R')}")
        for name, c in items:
            print(f"  {name:<16} {c:>6}  {_c('G')}{'█' * int(30 * c / max(1, items[0][1]))}{_c('R')}")

    def count(f):
        d = {}
        for r in fs:
            for v in (f(r) if isinstance(f(r), list) else [f(r)]):
                if v:
                    d[v] = d.get(v, 0) + 1
        return d
    top("Familias", count(lambda r: r["family"]))
    top("Instrumentos", count(lambda r: r["instruments"]))
    top("Tonalidades", count(lambda r: r["key"]))
    top("Compases", count(lambda r: r["meter"]), 6)
    if cat["errors"]:
        e = {}
        for v in cat["errors"].values():
            e[v["class"]] = e.get(v["class"], 0) + 1
        top("Archivos ilegibles por causa", e)
    return 0


def cmd_diff(a):
    try:
        old, new = _load(a.old), _load(a.new)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    o, n = old["files"], new["files"]
    added = sorted(set(n) - set(o))
    removed = sorted(set(o) - set(n))
    changed = sorted(p for p in set(o) & set(n) if o[p].get("fhash") != n[p].get("fhash"))
    print(f"+{len(added)} añadidos · -{len(removed)} quitados · ~{len(changed)} modificados")
    for tag, lst in (("+", added), ("-", removed), ("~", changed)):
        for p in lst[:30]:
            print(f"  {tag} {os.path.basename(p)}")
        if len(lst) > 30:
            print(f"  {tag} … {len(lst) - 30} más")
    return 0


def cmd_dups(a):
    try:
        cat = _load(a.catalog)
    except (OSError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    field = "fhash" if a.by == "file" else "nhash"
    groups = {}
    for r in cat["files"].values():
        groups.setdefault(r[field], []).append(r)
    dup = [g for g in groups.values() if len(g) > 1]
    extra = sum(len(g) - 1 for g in dup)
    print(f"{len(dup)} grupos de duplicados ({a.by}); {extra} archivos sobran de {len(cat['files'])}")
    for g in sorted(dup, key=lambda g: -len(g))[:20]:
        print(f"  × {len(g)}: " + ", ".join(r["name"] for r in sorted(g, key=lambda r: r["name"])[:6]))
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST (codificador SMF mínimo propio)
# ══════════════════════════════════════════════════════════════════════════════

def _vl(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def _smf(tracks, tpb=480, fmt=1):
    """tracks: [[(abs_tick, bytes_evento)]] → bytes SMF."""
    out = b"MThd" + (6).to_bytes(4, "big") + fmt.to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + tpb.to_bytes(2, "big")
    for evs in tracks:
        body, last = b"", 0
        for t, e in sorted(evs, key=lambda x: x[0]):
            body += _vl(t - last) + e
            last = t
        body += _vl(0) + b"\xFF\x2F\x00"
        out += b"MTrk" + len(body).to_bytes(4, "big") + body
    return out


def _mk_notes(ch, prog, pitches, step=480, vel=80, name=None, start=0):
    ev = [(0, bytes([0xC0 | ch, prog]))]
    if name:
        ev.append((0, b"\xFF\x03" + _vl(len(name)) + name.encode()))
    for k, p in enumerate(pitches):
        s = start + k * step
        ev.append((s, bytes([0x90 | ch, p, vel])))
        ev.append((s + step - 10, bytes([0x80 | ch, p, 0])))
    return ev


def cmd_selftest(_a):
    import tempfile
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name}")
        ok = ok and bool(cond)

    tempo = (0, b"\xFF\x51\x03" + (1000000).to_bytes(3, "big"))      # 60 bpm
    ts44 = (0, b"\xFF\x58\x04\x04\x02\x18\x08")
    dm = [62, 65, 69, 62, 65, 69, 74, 65] * 12                           # Re menor
    with tempfile.TemporaryDirectory() as d:
        def w(name, data):
            p = os.path.join(d, name)
            with open(p, "wb") as fh:
                fh.write(data)
            return p
        tango = _smf([[tempo, ts44, (0, b"\xFF\x03\x0cTango Abismo")],
                      _mk_notes(0, 23, dm, name="Bandoneon"), _mk_notes(1, 42, [38, 45] * 48, name="Cello"),
                      _mk_notes(2, 41, [57, 62] * 48, name="Viola")])
        pa = w("a_tango.mid", tango)
        w("b_piano.mid", _smf([[tempo, ts44], _mk_notes(0, 0, [60, 64, 67, 72] * 24)]))
        w("c_copia.mid", _smf([[ts44, tempo], _mk_notes(0, 0, [60, 64, 67, 72] * 24)]))   # mismas notas, otro empaquetado
        w("d_truncado.mid", tango[:-40])
        badks = _smf([[tempo, ts44, (0, b"\xFF\x59\x02\x7F\x05")], _mk_notes(0, 40, [69, 72, 76] * 20)])
        w("e_ksinvalida.mid", badks)
        w("f_riff.mid", b"RIFF\x00\x00\x00\x00RMIDdata\x00\x00\x00\x00" + _smf([[tempo, ts44], _mk_notes(0, 71, [62, 64, 65] * 20)]))
        w("g_basura.mid", b"esto no es un midi" * 20)
        w("h_vacio.mid", b"")
        w("i_sinnotas.mid", _smf([[tempo, ts44]]))
        w("j_vals.mid", _smf([[tempo, (0, b"\xFF\x58\x04\x03\x02\x18\x08"), (0, b"\xFF\x03\x0cVals criollo")],
                              _mk_notes(0, 21, [57, 60, 64] * 30)]))
        os.makedirs(os.path.join(d, "sub"))
        w("sub/k_orq.mid", _smf([[tempo, ts44],
                                 _mk_notes(0, 40, [72, 74] * 20), _mk_notes(1, 41, [64, 62] * 20),
                                 _mk_notes(2, 42, [48, 43] * 20), _mk_notes(3, 73, [79, 77] * 20),
                                 _mk_notes(4, 71, [67, 69] * 20), _mk_notes(5, 60, [55, 52] * 20),
                                 _mk_notes(6, 56, [60, 62] * 20), _mk_notes(7, 47, [38, 38] * 20)]))
        cat = os.path.join(d, "out", "cat.json")

        class A: pass
        a = A(); a.paths = [d]; a.out = cat; a.update = False; a.csv = os.path.join(d, "out", "c.csv"); a.jobs = 2
        chk("build rc 0", cmd_build(a) == 0)
        c = _load(cat)
        chk("cataloga 8 archivos legibles", len(c["files"]) == 8)
        chk("errores clasificados: no-midi, vacio, sin-notas",
            sorted(e["class"] for e in c["errors"].values()) == ["no-midi", "sin-notas", "vacio"])
        t = c["files"][os.path.abspath(pa)]
        chk("título de la pista", t["title"] == "Tango Abismo")
        chk("tonalidad Dm (bandoneón en Re menor)", t["key"] == "Dm")
        chk("instrumentos bandoneon, cello, viola", {"bandoneon", "cello", "viola"} <= set(t["instruments"]))
        chk("familia tango", "tango" in t["family"])
        chk("tempo 60 y compás 4/4", abs(t["tempo_bpm"] - 60) < 1 and t["meter"] == "4/4")
        chk("truncado → parcial, no error", c["files"][os.path.abspath(os.path.join(d, "d_truncado.mid"))]["partial"])
        chk("firma de tonalidad inválida se salva",
            os.path.abspath(os.path.join(d, "e_ksinvalida.mid")) in c["files"])
        chk("RIFF/RMID se salva",
            c["files"][os.path.abspath(os.path.join(d, "f_riff.mid"))]["instruments"] == ["clarinet"])
        chk("vals en 3/4 con acordeón",
            c["files"][os.path.abspath(os.path.join(d, "j_vals.mid"))]["meter"] == "3/4")
        chk("orquesta detectada", "orquesta" in c["files"][os.path.abspath(os.path.join(d, "sub", "k_orq.mid"))]["family"])
        chk("piano solo", "piano_solo" in c["files"][os.path.abspath(os.path.join(d, "b_piano.mid"))]["family"])
        chk("CSV escrito", os.path.exists(a.csv))
        r = search_records(c["files"], ["tango"], key="Re menor", instrument="viola,cello")
        chk("search: tango + Re menor + viola,cello → el original y su copia truncada",
            sorted(x["name"] for x in r) == ["a_tango.mid", "d_truncado.mid"])
        chk("search: ~ = alguno de los instrumentos",
            len(search_records(c["files"], instrument="~viola,clarinet")) >= 3)
        chk("search: --key «D minor» ≡ «Dm»", parse_key("D minor") == "Dm" == parse_key("Re menor"))
        chk("search: F# sostenido", parse_key("Fa sostenido menor") == "F#m")
        chk("search: orquesta sin batería", len(search_records(c["files"], family="orquesta", no_drums=True)) == 1)
        chk("search: tempo fuera de rango → 0", not search_records(c["files"], tempo="100-120"))
        # incremental
        r1 = os.stat(cat).st_mtime
        a.update = True
        cmd_build(a)
        c2 = _load(cat)
        chk("update reutiliza fichas", c2["files"].keys() == c["files"].keys())
        w("l_nuevo.mid", _smf([[tempo, ts44], _mk_notes(0, 0, [60, 62, 64] * 10)]))
        cmd_build(a)
        c3 = _load(cat)
        chk("update añade solo el nuevo", len(c3["files"]) == len(c["files"]) + 1)
        # duplicados de contenido
        g = {}
        for rr in c3["files"].values():
            g.setdefault(rr["nhash"], []).append(rr["name"])
        chk("dups por notas: b_piano y c_copia", any(sorted(v) == ["b_piano.mid", "c_copia.mid"] for v in g.values()))
        chk("dups por fichero: distintos", len({rr["fhash"] for rr in c3["files"].values()}) == len(c3["files"]))
    print("selftest:", "OK" if ok else "FALLOS")
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(prog="corpus_catalog.py", description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("build", help="Cataloga un corpus")
    sp.add_argument("--no-color", action="store_true")
    sp.add_argument("paths", nargs="+")
    sp.add_argument("--out", "-o", default="corpus_catalog.json")
    sp.add_argument("--update", action="store_true")
    sp.add_argument("--csv")
    sp.add_argument("--jobs", "-j", type=int, default=4)
    sp = sub.add_parser("search", help="Filtra y busca")
    sp.add_argument("--no-color", action="store_true")
    sp.add_argument("catalog")
    sp.add_argument("query", nargs="*")
    sp.add_argument("--key")
    sp.add_argument("--instrument")
    sp.add_argument("--family")
    sp.add_argument("--tempo")
    sp.add_argument("--duration")
    sp.add_argument("--meter")
    sp.add_argument("--no-drums", action="store_true")
    sp.add_argument("--min-notes", type=int, default=0)
    sp.add_argument("--top", type=int, default=25)
    sp.add_argument("--json")
    sp = sub.add_parser("show", help="Ficha de un archivo")
    sp.add_argument("--no-color", action="store_true")
    sp.add_argument("catalog")
    sp.add_argument("target")
    sp = sub.add_parser("stats", help="Resumen")
    sp.add_argument("--no-color", action="store_true")
    sp.add_argument("catalog")
    sp = sub.add_parser("diff", help="Diferencias entre catálogos")
    sp.add_argument("--no-color", action="store_true")
    sp.add_argument("old")
    sp.add_argument("new")
    sp = sub.add_parser("dups", help="Duplicados")
    sp.add_argument("--no-color", action="store_true")
    sp.add_argument("catalog")
    sp.add_argument("--by", choices=("file", "notes"), default="notes")
    sub.add_parser("selftest", help="Prueba con MIDIs sintéticos")
    return p


def main(argv=None):
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if getattr(a, "no_color", False):
        _USE_COLOR = False
    return {"build": cmd_build, "search": cmd_search, "show": cmd_show, "stats": cmd_stats,
            "diff": cmd_diff, "dups": cmd_dups, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
