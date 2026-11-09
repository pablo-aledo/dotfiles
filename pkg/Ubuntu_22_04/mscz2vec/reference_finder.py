#!/usr/bin/env python3
"""
reference_finder.py — Reference Finder: ordena los MIDIs de un corpus por su
utilidad como REFERENCIA para una obra descrita en un «brief» de texto
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Buscar referencias en un corpus a mano es lento, y las búsquedas semánticas
genéricas no entienden géneros ni instrumentación («tango», «viola»,
«ostinato»). Aquí el brief enumera criterios MEDIBLES —tonalidad, tempo,
duración, compás, instrumentos, células rítmicas, repetición, cromatismo,
acumulación de capas— y cada MIDI recibe una puntuación 0-100 con el
desglose por criterio, de modo que cada posición del ranking es explicable.
Solo puntúa los criterios que aparecen en el brief. Una fase de indexado
(opcional pero recomendada) extrae las características una sola vez.
Autocontenida: solo mido; no lee .docx (el brief es un fichero de texto).

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  index     Extrae y guarda las características de todo el corpus (JSON)
  rank      Ordena el corpus según un brief (con índice o al vuelo)
  explain   Desglose de la puntuación de UN archivo frente al brief
  profile   Ficha de un MIDI: plantilla instrumental, tipo, tonalidad, ritmo

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  index
    paths PATHS […]             ficheros, globs o carpetas (recursivo)
    --out, -o OUT               índice JSON (defecto reference_index.json)
    --no-color

  rank
    paths PATHS […]             corpus (si no se da --index)
    --brief, -b BRIEF           fichero de brief (obligatorio)
    --index INDEX               usa un índice ya creado
    --top TOP                   filas a mostrar (defecto 20)
    --min-score MIN_SCORE       oculta por debajo de esta puntuación (defecto 0)
    --json JSON
    --no-color

  explain
    midi MIDI
    --brief, -b BRIEF
    --no-color

  profile
    midis MIDIS […]
    --no-color

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[index]
paths      list  of=file io=in required
--out      file  io=out ext=json default=reference_index.json alias=-o
--no-color bool  default=false
[rank]
paths      list  of=file io=in
--brief    file  io=in required alias=-b
--index    file  io=in ext=json
--top      int   default=20 min=1
--min-score float default=0 min=0 max=100
--json     file  io=out ext=json
--no-color bool  default=false
[explain]
midi       file  io=in ext=mid required
--brief    file  io=in required alias=-b
--no-color bool  default=false
[profile]
midis      list  of=file io=in required
--no-color bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python reference_finder.py index midis/ -o idx.json
    python reference_finder.py rank --index idx.json --brief plan.brief --top 15
    python reference_finder.py rank midis/ --brief plan.brief --json ranking.json
    python reference_finder.py explain midis/obra.mid --brief plan.brief
    python reference_finder.py profile midis/obra.mid

  FORMATO DEL BRIEF (texto plano, «clave: valor», # comentarios):
      key: Dm, F, A                 # tonalidades por orden de interés (m = menor)
      tempo: 60-120                 # BPM
      duration: 150-420             # segundos
      meter: 4/4, 3/4
      instruments: *viola, *violin, clarinet, flauta, timpani, arpa
                                    # * = doble peso; admite español/inglés y
                                    # grupos: cuerdas, maderas, metales, percusion
      avoid: drums                  # penaliza lo que NO debe haber
      rhythm: marcato, habanera     # células del catálogo o patrón x.xx.x. en corcheas
      repetition: 0.4-1             # fracción de compases que repiten material
      chromatic: high               # low | mid | high | rango 0.05-0.2
      accumulation: yes             # entran capas progresivamente
      weights: instruments=3, key=2 # peso por criterio (defecto 1)

  CÉLULAS RÍTMICAS (rejilla de corcheas): marcato «x.xxx.» (negra-corchea-
  corchea-negra), marcato332 «x..x..x.», habanera «x..xx.x.», milonga
  «x.x.xx..», marcha «x.x.x.x.», vals «x.x.x.».
  La presencia se mide en el bajo y en el acompañamiento: % de periodos que
  contienen la célula completa con precisión ≥ 80 % (tolera algún golpe de más).

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 2 sin archivos legibles

  API:  from reference_finder import extract_features, score_features, parse_brief

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  mido
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import unicodedata

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def _ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARIO: instrumentos, tonalidades, células
# ══════════════════════════════════════════════════════════════════════════════

GROUPS = {
    "strings": {"violin", "viola", "cello", "contrabass", "strings"},
    "woodwinds": {"flute", "oboe", "clarinet", "bassoon", "recorder", "sax"},
    "brass": {"horn", "trumpet", "trombone", "tuba", "brass"},
    "percussion": {"timpani", "percussion", "mallets", "drums"},
    "keys": {"piano", "harpsichord", "organ", "accordion", "bandoneon"},
    "plucked": {"harp", "guitar"},
    "voice": {"choir", "voice"},
}
CANON = set().union(*GROUPS.values()) | {"bass", "synth"}

ALIASES = {
    "violin": "violin", "violín": "violin", "vln": "violin", "violino": "violin", "fiddle": "violin",
    "viola": "viola", "vla": "viola", "cello": "cello", "chelo": "cello", "violonchelo": "cello",
    "violoncello": "cello", "vc": "cello", "contrabass": "contrabass", "contrabajo": "contrabass",
    "doublebass": "contrabass", "double bass": "contrabass", "cb": "contrabass",
    "strings": "strings", "cuerdas": "strings", "cuerda": "strings", "string": "strings",
    "harp": "harp", "arpa": "harp", "piano": "piano", "harpsichord": "harpsichord",
    "clavecin": "harpsichord", "clave": "harpsichord", "organ": "organ", "organo": "organ",
    "accordion": "accordion", "acordeon": "accordion", "bandoneon": "bandoneon",
    "bandoneón": "bandoneon", "guitar": "guitar", "guitarra": "guitar", "bass": "bass", "bajo": "bass",
    "flute": "flute", "flauta": "flute", "piccolo": "flute", "oboe": "oboe",
    "clarinet": "clarinet", "clarinete": "clarinet", "bassoon": "bassoon", "fagot": "bassoon",
    "recorder": "recorder", "sax": "sax", "saxo": "sax", "saxophone": "sax", "saxofon": "sax",
    "horn": "horn", "trompa": "horn", "trumpet": "trumpet", "trompeta": "trumpet",
    "trombone": "trombone", "trombon": "trombone", "tuba": "tuba", "brass": "brass",
    "timpani": "timpani", "timp": "timpani", "timbales": "timpani", "timbal": "timpani",
    "percussion": "percussion", "percusion": "percussion", "drums": "drums", "drum": "drums",
    "bateria": "drums", "glockenspiel": "mallets", "vibraphone": "mallets", "marimba": "mallets",
    "xylophone": "mallets", "choir": "choir", "coro": "choir", "voice": "voice", "voz": "voice",
    "vocals": "voice", "vox": "voice", "synth": "synth",
    "woodwinds": "woodwinds", "maderas": "woodwinds", "madera": "woodwinds", "winds": "woodwinds",
    "metales": "brass", "metal": "brass", "keys": "keys", "teclas": "keys", "plucked": "plucked",
}


def _norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch)).strip()


def _tags_from_text(text):
    """Etiquetas canónicas a partir de un nombre de pista/instrumento."""
    t = _norm(text)
    tags = set()
    for word in re.findall(r"[a-z]+", t):
        if word in ALIASES:
            tags.add(ALIASES[word])
    if "double bass" in t or "contra bass" in t:
        tags.add("contrabass")
    return tags


def gm_tags(prog, channel):
    if channel == 9:
        return {"percussion", "drums"}
    p = prog
    if p is None:
        return set()
    if 0 <= p <= 7:
        return {"harpsichord"} if p == 6 else {"piano"}
    if 8 <= p <= 15:
        return {"mallets"}
    if 16 <= p <= 20:
        return {"organ"}
    if p == 21:
        return {"accordion"}
    if p == 23:
        return {"accordion", "bandoneon"}
    if 24 <= p <= 31:
        return {"guitar"}
    if 32 <= p <= 39:
        return {"bass"}
    return {40: {"violin"}, 41: {"viola"}, 42: {"cello"}, 43: {"contrabass"},
            44: {"strings"}, 45: {"strings"}, 46: {"harp"}, 47: {"timpani"},
            56: {"trumpet"}, 57: {"trombone"}, 58: {"tuba"}, 59: {"trumpet"}, 60: {"horn"},
            68: {"oboe"}, 69: {"oboe"}, 70: {"bassoon"}, 71: {"clarinet"},
            72: {"flute"}, 73: {"flute"}, 74: {"recorder"}}.get(p) or (
        {"strings"} if 48 <= p <= 51 else {"choir"} if 52 <= p <= 54 else
        {"brass"} if 61 <= p <= 63 else {"sax"} if 64 <= p <= 67 else
        {"flute"} if 75 <= p <= 79 else {"synth"} if 80 <= p <= 103 else
        {"percussion"} if 112 <= p <= 119 else set())


def with_groups(tags):
    out = set(tags)
    for g, members in GROUPS.items():
        if tags & members:
            out.add(g)
    return out


SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

CELLS = {
    "marcato": "x.xxx.",        # negra + 2 corcheas + negra (3 pulsos), ADN del tango nuevo
    "marcato332": "x..x..x.",   # 3+3+2
    "habanera": "x..xx.x.",
    "milonga": "x.x.xx..",
    "marcha": "x.x.x.x.",
    "vals": "x.x.x.",
}


def key_name(tonic, mode):
    flats = (tonic in (0, 2, 3, 5, 7, 10)) if mode == "min" else (tonic in (1, 3, 5, 8, 10))
    return (FLAT if flats else SHARP)[tonic] + ("m" if mode == "min" else "")


def parse_key(txt):
    t = txt.strip()
    m = re.match(r"^([A-Ga-g])([#b♯♭]?)\s*(.*)$", t)
    if not m:
        return None
    pc = _LETTER[m.group(1).upper()] + (1 if m.group(2) in ("#", "♯") else -1 if m.group(2) in ("b", "♭") else 0)
    rest = _norm(m.group(3))
    mode = "min" if rest in ("m", "min", "minor", "menor") else "maj"
    return pc % 12, mode


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE CARACTERÍSTICAS (una vez por archivo)
# ══════════════════════════════════════════════════════════════════════════════

def _corr(a, b):
    ma, mb = sum(a) / 12.0, sum(b) / 12.0
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def _ks(hist):
    out = []
    for t in range(12):
        for mode, prof in (("maj", KK_MAJ), ("min", KK_MIN)):
            out.append((_corr(hist, [prof[(i - t) % 12] for i in range(12)]), t, mode))
    out.sort(reverse=True)
    return out


def _read(path):
    mid = mido.MidiFile(path, clip=True)
    tpb = mid.ticks_per_beat
    tempos, tsigs, voices = [], [], {}
    chan_prog = {}   # primer program_change visto por canal (en cualquier pista)
    for ti, tr in enumerate(mid.tracks):
        t = 0
        name = ""
        progs = {}
        active = {}
        for m in tr:
            t += m.time
            if m.type == "track_name":
                name = m.name
            elif m.type == "instrument_name" and not name:
                name = m.name
            elif m.type == "set_tempo":
                tempos.append((t, m.tempo))
            elif m.type == "time_signature":
                tsigs.append((t, m.numerator, m.denominator))
            elif m.type == "program_change":
                progs.setdefault(m.channel, m.program)
                chan_prog.setdefault(m.channel, m.program)
            elif m.type == "note_on" and m.velocity > 0:
                active.setdefault((m.channel, m.note), []).append(t)
            elif m.type in ("note_off", "note_on"):
                st = active.get((m.channel, m.note))
                if st:
                    s = st.pop(0)
                    v = voices.setdefault((ti, m.channel), {"track": ti, "ch": m.channel, "name": name,
                                                            "prog": progs.get(m.channel), "notes": []})
                    v["notes"].append((s, max(t, s + 1), m.note))
        for (ch, p), st in active.items():
            for s in st:
                v = voices.setdefault((ti, ch), {"track": ti, "ch": ch, "name": name,
                                                 "prog": progs.get(ch), "notes": []})
                v["notes"].append((s, s + tpb, p))
    for v in voices.values():
        v["notes"].sort()
        v["name"] = v["name"] or ""
        if v["prog"] is None:
            v["prog"] = chan_prog.get(v["ch"])
    return tpb, sorted(tempos), sorted(tsigs), [v for _, v in sorted(voices.items())]


def _dominant_meter(tsigs, end_tick):
    """Compás que cubre más ticks hasta end_tick (en vez de «el primero»).
    Con firmas repetidas o cambios (p. ej. 4/4 → 2/4) elige el dominante;
    en empate gana el más temprano. Sin firmas: 4/4."""
    if not tsigs:
        return 4, 4
    ts = sorted(tsigs)
    span = {}
    for i, (t, n, d) in enumerate(ts):
        t_end = ts[i + 1][0] if i + 1 < len(ts) else max(end_tick, t)
        span[(n, d)] = span.get((n, d), 0) + max(0, t_end - t)
    best = max(span.values())
    for _t, n, d in ts:
        if span[(n, d)] == best:
            return n, d


def _bar8(meter):
    """Corcheas por compás a partir de «n/d» (4/4 → 8, 3/4 → 6, 6/8 → 6)."""
    try:
        n, d = (int(x) for x in str(meter).split("/"))
        return max(1, int(round(8.0 * n / d)))
    except Exception:
        return 8


def _seconds(tick, tpb, tempos):
    sec, last_t, tempo = 0.0, 0, 500000
    for t, tp in tempos:
        if t >= tick:
            break
        sec += (t - last_t) * tempo / 1e6 / tpb
        last_t, tempo = t, tp
    return sec + (tick - last_t) * tempo / 1e6 / tpb


def _grid8(notes, tpb, cap=6000):
    """Rejilla de corcheas: 'x' donde hay un ataque."""
    if not notes:
        return ""
    q = tpb / 2.0
    slots = set(int(round(s / q)) for s, _e, _p in notes)
    n = min(cap, max(slots) + 1)
    return "".join("x" if i in slots else "." for i in range(n))


def extract_features(path):
    tpb, tempos, tsigs, voices = _read(path)
    pitched = [v for v in voices if v["ch"] != 9 and v["notes"]]
    drums = [v for v in voices if v["ch"] == 9 and v["notes"]]
    allnotes = [n for v in pitched for n in v["notes"]]
    if not allnotes:
        raise ValueError("sin notas")
    last = max(e for _, e, _ in allnotes)
    num, den = _dominant_meter(tsigs, last)
    bar = max(1, int(tpb * 4 * num / den))
    dur = _seconds(last, tpb, tempos)
    bpm = 60.0 * (last / tpb) / dur if dur > 0 else 120.0     # tempo medio ponderado

    # Instrumentos por voz
    vinfo, alltags = [], set()
    for v in voices:
        tags = gm_tags(v["prog"], v["ch"]) | _tags_from_text(v["name"])
        alltags |= tags
        ps = sorted(n[2] for n in v["notes"])
        vinfo.append({"name": v["name"], "prog": v["prog"], "ch": v["ch"], "notes": len(v["notes"]),
                      "tags": sorted(tags), "median": ps[len(ps) // 2],
                      "entry_bar": v["notes"][0][0] // bar + 1})
    # Tonalidad (KS global ponderado por duración)
    hist = [0.0] * 12
    for s, e, p in allnotes:
        hist[p % 12] += e - s
    ks = _ks(hist)
    r1, t1, m1 = ks[0]
    r2, t2, m2 = ks[1]
    # Cromatismo: fracción de duración fuera de la escala de la tonalidad
    scale = {0, 2, 3, 5, 7, 8, 10, 11} if m1 == "min" else {0, 2, 4, 5, 7, 9, 11}
    tot = sum(hist) or 1.0
    chrom = sum(hist[(t1 + d) % 12] for d in range(12) if d not in scale) / tot

    # Voces melódicas para rejillas rítmicas
    main = [v for v in pitched if len(v["notes"]) >= 8] or pitched
    med = lambda v: sorted(n[2] for n in v["notes"])[len(v["notes"]) // 2]  # noqa: E731
    bass_v = min(main, key=med)
    top_v = max(main, key=med)
    rest = [n for v in main if v is not top_v or len(main) < 3 for n in v["notes"]]
    grid_bass = _grid8(bass_v["notes"], tpb)
    grid_rest = _grid8(rest, tpb)

    # Repetición por compases
    sigs = {}
    for s, e, p in allnotes:
        b = s // bar
        sigs.setdefault(b, set()).add((int(round((s - b * bar) / (tpb / 2.0))), p % 12))
    seq = [sigs[b] for b in sorted(sigs)]
    rep = 0
    for i, sg in enumerate(seq):
        for prev in seq[max(0, i - 64):i]:
            u = len(sg | prev)
            if u and len(sg & prev) / u >= 0.85:
                rep += 1
                break
    repetition = rep / len(seq) if seq else 0.0

    # Acumulación: compases de entrada de voces con suficiente material
    entries = sorted(set(vi["entry_bar"] for vi in vinfo if vi["notes"] >= 8))
    n_voices = len([vi for vi in vinfo if vi["notes"] >= 8])

    st = os.stat(path)
    return {
        "path": path, "size": st.st_size, "mtime": int(st.st_mtime),
        "duration_s": round(dur, 1), "tempo_bpm": round(bpm, 1), "meter": f"{num}/{den}",
        "n_notes": len(allnotes), "n_voices": n_voices, "bars": int(math.ceil(last / bar)),
        "tags": sorted(alltags), "voices": vinfo, "drums": bool(drums),
        "key": key_name(t1, m1), "key_r": round(r1, 3), "key2": key_name(t2, m2), "key2_r": round(r2, 3),
        "key_tonic": t1, "key_mode": m1, "key2_tonic": t2, "key2_mode": m2,
        "chromatic": round(chrom, 3), "repetition": round(repetition, 3),
        "entries": entries, "grid_bass": grid_bass, "grid_rest": grid_rest,
    }


def _presence_period(grid, pat, L):
    """Presencia de `pat` (rellena con silencios hasta L corcheas) en ventanas de L."""
    pat = pat.ljust(L, ".")
    need = [i for i, ch in enumerate(pat) if ch == "x"]
    nx = len(need)
    first, last = grid.find("x"), grid.rfind("x")
    if first < 0 or not need:
        return 0.0
    best = 0.0
    for ph in range(L):
        st = ph + ((first - ph) // L) * L
        tot = car = 0
        for w in range(st, last + 1, L):
            seg = grid[max(0, w):w + L].ljust(L, ".")
            if "x" not in seg:
                continue
            tot += 1
            if all(seg[i] == "x" for i in need) and nx / seg.count("x") >= 0.8:
                car += 1
        if tot:
            best = max(best, (car / tot) * min(1.0, tot / 6.0))
    return best


def cell_presence(grid, pat, bar8=None):
    """Fracción de periodos con ataque que contienen la célula (mejor fase).

    Se mide con el periodo propio de la célula (len(pat)) y, si se da `bar8`
    (corcheas por compás) y la célula es más corta que el compás, también con
    el compás como periodo (célula + silencios hasta completarlo). Así una
    célula de 3 pulsos como «marcato» (x.xxx.) se detecta tanto repetida cada
    3 pulsos como una vez por compás de 4/4 (x.xxx...). Devuelve el máximo.
    """
    L = len(pat)
    best = _presence_period(grid, pat, L)
    if bar8 and bar8 > L:
        best = max(best, _presence_period(grid, pat, bar8))
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  BRIEF Y PUNTUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _split(v):
    return [x.strip() for x in re.split(r"[;,]", v) if x.strip()]


def _range(v):
    nums = re.findall(r"\d+(?:\.\d+)?", v)
    if not nums:
        return None
    lo = float(nums[0])
    hi = float(nums[1]) if len(nums) > 1 else lo
    return (min(lo, hi), max(lo, hi))


def parse_brief(text):
    b, warns = {}, []
    for raw in text.splitlines():
        line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()   # «F#m» no es comentario
        if not line:
            continue
        if ":" not in line:
            warns.append(f"línea ignorada: {raw.strip()}")
            continue
        k, v = line.split(":", 1)
        k, v = _norm(k), v.strip()
        if k in ("key", "tonalidad", "tonalidades"):
            ks = [parse_key(x) for x in _split(v)]
            b["key"] = [x for x in ks if x]
        elif k in ("tempo", "bpm"):
            b["tempo"] = _range(v)
        elif k in ("duration", "duracion"):
            b["duration"] = _range(v)
        elif k in ("meter", "compas"):
            b["meter"] = [x.replace(" ", "") for x in _split(v)]
        elif k in ("instruments", "instrumentos"):
            items = []
            for x in _split(v):
                w = 2.0 if x.startswith("*") else 1.0
                tag = ALIASES.get(_norm(x.lstrip("*")))
                if tag:
                    items.append((tag, w))
                else:
                    warns.append(f"instrumento desconocido: {x}")
            b["instruments"] = items
        elif k in ("avoid", "evitar"):
            b["avoid"] = [ALIASES.get(_norm(x)) for x in _split(v) if ALIASES.get(_norm(x))]
        elif k in ("rhythm", "ritmo"):
            items = []
            for x in _split(v):
                n = _norm(x)
                if re.fullmatch(r"[x.]+", n) and "x" in n:
                    items.append((n, n))
                elif n in CELLS:
                    items.append((n, CELLS[n]))
                else:
                    warns.append(f"célula rítmica desconocida: {x}")
            b["rhythm"] = items
        elif k in ("repetition", "repeticion"):
            b["repetition"] = _range(v)
        elif k in ("chromatic", "cromatismo"):
            n = _norm(v)
            b["chromatic"] = {"low": (0, 0.05), "bajo": (0, 0.05), "mid": (0.05, 0.15), "medio": (0.05, 0.15),
                              "high": (0.15, 1.0), "alto": (0.15, 1.0)}.get(n) or _range(v)
        elif k in ("accumulation", "acumulacion"):
            b["accumulation"] = _norm(v) in ("yes", "si", "true", "1", "sí")
        elif k in ("percussion", "percusion"):
            b["percussion"] = _norm(v) in ("yes", "si", "true", "1")
        elif k in ("weights", "pesos"):
            w = {}
            for x in _split(v):
                if "=" in x:
                    a, c = x.split("=", 1)
                    try:
                        w[_norm(a)] = float(c)
                    except ValueError:
                        warns.append(f"peso inválido: {x}")
            b["weights"] = w
        else:
            warns.append(f"clave desconocida: {k}")
    return b, warns


def _range_score(x, rng):
    lo, hi = rng
    if lo <= x <= hi:
        return 1.0
    d = lo - x if x < lo else x - hi
    width = max(0.3 * max(abs(lo), abs(hi)), 1e-9)
    return max(0.0, 1.0 - d / width)


def score_features(f, brief):
    """Devuelve (score0-100, [(criterio, puntuación 0-1, peso, evidencia)])."""
    W = brief.get("weights", {})
    rows = []

    def add(name, s, ev):
        rows.append((name, max(0.0, min(1.0, s)), W.get(name, 1.0), ev))

    if brief.get("key"):
        keys = brief["key"]
        fk = [(f["key_tonic"], f["key_mode"], 1.0), (f["key2_tonic"], f["key2_mode"], 0.6)]
        best, why = 0.0, "ninguna coincide"
        for i, (bt, bm) in enumerate(keys):
            w = 1.0 if i == 0 else max(0.6, 0.85 - 0.05 * (i - 1))
            for ft, fm, fw in fk:
                if (ft, fm) == (bt, bm):
                    sc, tx = w * fw, "coincide"
                elif fm != bm and ((bt - ft) % 12 == 3 if fm == "min" else (ft - bt) % 12 == 3) and fw == 1.0:
                    sc, tx = 0.4 * w, "relativa"
                elif ft == bt and fm != bm and fw == 1.0:
                    sc, tx = 0.3 * w, "paralela"
                else:
                    continue
                if sc > best:
                    best, why = sc, f"{key_name(bt, bm)} vs {f['key']} ({tx})"
        add("key", best, why if best else f"detectada {f['key']} (r={f['key_r']}), no coincide")

    if brief.get("tempo"):
        add("tempo", _range_score(f["tempo_bpm"], brief["tempo"]), f"{f['tempo_bpm']} bpm")
    if brief.get("duration"):
        add("duration", _range_score(f["duration_s"], brief["duration"]), f"{f['duration_s']} s")
    if brief.get("meter"):
        add("meter", 1.0 if f["meter"] in brief["meter"] else 0.0, f["meter"])

    have = with_groups(set(f["tags"]))
    if brief.get("instruments"):
        tot = acc = 0.0
        found, partial, missing = [], [], []
        for tag, w in brief["instruments"]:
            tot += w
            if tag in have:
                acc += w
                found.append(tag)
            else:
                grp = [g for g, mem in GROUPS.items() if tag in mem]
                if grp and grp[0] in have:
                    acc += 0.5 * w
                    partial.append(tag)
                else:
                    missing.append(tag)
        ev = f"tiene {','.join(found) or '—'}"
        if partial:
            ev += f"; misma familia: {','.join(partial)}"
        if missing:
            ev += f"; faltan: {','.join(missing)}"
        add("instruments", acc / tot if tot else 0.0, ev)

    if brief.get("avoid"):
        bad = [a for a in brief["avoid"] if a in have or (a == "drums" and f["drums"])]
        add("avoid", 1.0 - len(bad) / len(brief["avoid"]), f"presentes: {','.join(bad)}" if bad else "ninguno presente")
    if brief.get("percussion") is not None:
        add("percussion", 1.0 if f["drums"] == brief["percussion"] or ("percussion" in have) == brief["percussion"] else 0.0,
            "con percusión" if f["drums"] or "percussion" in have else "sin percusión")

    if brief.get("rhythm"):
        parts, best_ev, ss = [], "", []
        for name, pat in brief["rhythm"]:
            b8 = _bar8(f["meter"])
            sc = max(cell_presence(f["grid_bass"], pat, b8), cell_presence(f["grid_rest"], pat, b8))
            ss.append(sc)
            parts.append(f"{name} {sc:.0%}")
        add("rhythm", sum(ss) / len(ss), ", ".join(parts))
    if brief.get("repetition"):
        add("repetition", _range_score(f["repetition"], brief["repetition"]), f"{f['repetition']:.0%} de compases repiten")
    if brief.get("chromatic"):
        add("chromatic", _range_score(f["chromatic"], brief["chromatic"]), f"{f['chromatic']:.1%} fuera de escala")
    if brief.get("accumulation") is not None:
        n = max(0, len(f["entries"]) - 1)
        s = min(1.0, n / 3.0) if f["n_voices"] >= 3 else 0.0
        add("accumulation", s if brief["accumulation"] else 1.0 - s,
            f"{len(f['entries'])} entradas distintas (c. {','.join(map(str, f['entries'][:6]))})")

    if not rows:
        return 0.0, rows
    tw = sum(w for _, _, w, _ in rows)
    return 100.0 * sum(s * w for _, s, w, _ in rows) / tw, rows


# ══════════════════════════════════════════════════════════════════════════════
#  CORPUS E ÍNDICE
# ══════════════════════════════════════════════════════════════════════════════

def _expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _d, files in os.walk(p):
                for fn in sorted(files):
                    if fn.lower().endswith((".mid", ".midi", ".kar")):
                        out.append(os.path.join(root, fn))
        else:
            g = sorted(glob.glob(p))
            out.extend(g if g else [p])
    return sorted(dict.fromkeys(out))


def build_index(paths, prev=None):
    prev = {e["path"]: e for e in (prev or {}).get("files", [])}
    files, errors = [], []
    for p in paths:
        try:
            st = os.stat(p)
            old = prev.get(p)
            if old and old.get("size") == st.st_size and old.get("mtime") == int(st.st_mtime):
                files.append(old)
                continue
            files.append(extract_features(p))
        except Exception as e:  # noqa: BLE001
            errors.append({"path": p, "error": f"{type(e).__name__}: {e}"[:120]})
    return {"version": VERSION, "files": files, "errors": errors}


def _load_brief(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"No se puede leer el brief: {e}", file=sys.stderr)
        return None
    brief, warns = parse_brief(text)
    for w in warns:
        print(f"  {_c('YEL')}aviso brief:{_c('R')} {w}", file=sys.stderr)
    if not any(k in brief for k in ("key", "tempo", "duration", "meter", "instruments", "avoid",
                                    "rhythm", "repetition", "chromatic", "accumulation", "percussion")):
        print("El brief no contiene ningún criterio reconocible.", file=sys.stderr)
        return None
    return brief


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_index(args):
    files = _expand(args.paths)
    if not files:
        print("No se encontraron MIDIs.", file=sys.stderr)
        return 2
    prev = None
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, ValueError):
            prev = None
    idx = build_index(files, prev)
    _ensure_parent(args.out)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    print(f"Indexados {len(idx['files'])} archivos; {len(idx['errors'])} ilegibles → {args.out}")
    for e in idx["errors"][:10]:
        print(f"  ✗ {e['path']}: {e['error']}")
    return 0


def cmd_rank(args):
    brief = _load_brief(args.brief)
    if brief is None:
        return 1
    if args.index:
        try:
            with open(args.index, encoding="utf-8") as f:
                idx = json.load(f)
        except (OSError, ValueError) as e:
            print(f"Índice ilegible: {e}", file=sys.stderr)
            return 1
    elif args.paths:
        idx = build_index(_expand(args.paths))
    else:
        print("Indica un corpus (paths) o --index.", file=sys.stderr)
        return 1
    if not idx["files"]:
        print("Sin archivos legibles.", file=sys.stderr)
        return 2
    res = []
    for f in idx["files"]:
        sc, rows = score_features(f, brief)
        res.append({"path": f["path"], "score": round(sc, 1),
                    "criteria": {n: round(s, 2) for n, s, _w, _e in rows},
                    "key": f["key"], "tempo": f["tempo_bpm"], "duration_s": f["duration_s"],
                    "tags": f["tags"]})
    res.sort(key=lambda r: -r["score"])
    names = list(res[0]["criteria"].keys()) if res else []
    print(f"\n{_c('B')}RANKING — {len(res)} archivos · criterios: {', '.join(names)}{_c('R')}\n")
    hdr = f"  {'#':>3} {'pts':>5}  {'archivo':<42}" + "".join(f"{n[:5]:>6}" for n in names)
    print(_c("B") + hdr + _c("R"))
    shown = 0
    for i, r in enumerate(res, 1):
        if r["score"] < args.min_score or shown >= args.top:
            continue
        shown += 1
        col = _c("GRN") if r["score"] >= 70 else (_c("YEL") if r["score"] >= 45 else "")
        print(f"  {i:>3} {col}{r['score']:>5.1f}{_c('R')}  {os.path.basename(r['path'])[:41]:<42}"
              + "".join(f"{r['criteria'][n]:>6.2f}" for n in names))
    if idx.get("errors"):
        print(f"\n  ({len(idx['errors'])} archivos ilegibles omitidos; usa corpus_doctor scan para detalles)")
    print()
    if args.json:
        _ensure_parent(args.json)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


def cmd_explain(args):
    brief = _load_brief(args.brief)
    if brief is None:
        return 1
    try:
        f = extract_features(args.midi)
    except Exception as e:  # noqa: BLE001
        print(f"Ilegible: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    sc, rows = score_features(f, brief)
    print(f"\n{_c('B')}{os.path.basename(args.midi)}{_c('R')}  →  {_c('B')}{sc:.1f}/100{_c('R')}\n")
    print(f"  {'criterio':<13}{'punt.':>6}{'peso':>6}   evidencia")
    for n, s, w, ev in rows:
        col = _c("GRN") if s >= 0.7 else (_c("YEL") if s >= 0.4 else _c("RED"))
        print(f"  {n:<13}{col}{s:>6.2f}{_c('R')}{w:>6.1f}   {ev}")
    print()
    return 0


def _kind(f):
    have = with_groups(set(f["tags"]))
    nv = f["n_voices"]
    fam = [g for g in ("strings", "woodwinds", "brass", "percussion", "keys", "plucked", "voice") if g in have]
    if f["drums"] and ("guitar" in have or "bass" in have):
        return "pop/rock"
    if nv <= 1:
        return "solo"
    if "brass" in have and ("strings" in have or "woodwinds" in have) and len(fam) >= 3:
        return "orquestal"
    if "strings" in have or "woodwinds" in have:
        return "cámara/cuerdas" if not f["drums"] else "mixto con batería"
    if "accordion" in have or "bandoneon" in have:
        return "tango/folk (acordeón)"
    return "otros"


def cmd_profile(args):
    rc = 0
    for p in args.midis:
        try:
            f = extract_features(p)
        except Exception as e:  # noqa: BLE001
            print(f"{p}: ilegible ({type(e).__name__}: {e})", file=sys.stderr)
            rc = 2
            continue
        print(f"\n{_c('B')}{os.path.basename(p)}{_c('R')}")
        print(f"  tipo: {_kind(f)}   duración {f['duration_s']}s ({f['bars']} comp.)   {f['meter']}   {f['tempo_bpm']} bpm")
        print(f"  tonalidad {f['key']} (r={f['key_r']}), 2.ª {f['key2']}   cromatismo {f['chromatic']:.1%}   "
              f"repetición {f['repetition']:.0%}")
        b8 = _bar8(f['meter'])
        cells = {n: max(cell_presence(f['grid_bass'], pt, b8), cell_presence(f['grid_rest'], pt, b8)) for n, pt in CELLS.items()}
        top = sorted(cells.items(), key=lambda kv: -kv[1])[:3]
        print("  células: " + ", ".join(f"{n} {s:.0%}" for n, s in top))
        print(f"  entradas de voz en compases: {', '.join(map(str, f['entries'][:10]))}")
        print("  voces:")
        for v in sorted(f["voices"], key=lambda v: v["median"]):
            print(f"    {(v['name'] or '(sin nombre)')[:26]:<27} prog={v['prog']!s:<4} ch{v['ch'] + 1:<3} "
                  f"{v['notes']:>5} notas  {','.join(v['tags']) or '—'}")
    print()
    return rc


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="reference_finder.py",
                                 description="Ranking de MIDIs de un corpus según un brief compositivo.")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    i = sub.add_parser("index", help="Indexa las características del corpus")
    i.add_argument("paths", nargs="+")
    i.add_argument("--out", "-o", default="reference_index.json")
    i.add_argument("--no-color", action="store_true")
    i.set_defaults(fn=cmd_index)

    r = sub.add_parser("rank", help="Ordena el corpus según un brief")
    r.add_argument("paths", nargs="*")
    r.add_argument("--brief", "-b", required=True)
    r.add_argument("--index")
    r.add_argument("--top", type=int, default=20)
    r.add_argument("--min-score", type=float, default=0.0)
    r.add_argument("--json")
    r.add_argument("--no-color", action="store_true")
    r.set_defaults(fn=cmd_rank)

    e = sub.add_parser("explain", help="Desglose de la puntuación de un archivo")
    e.add_argument("midi")
    e.add_argument("--brief", "-b", required=True)
    e.add_argument("--no-color", action="store_true")
    e.set_defaults(fn=cmd_explain)

    p = sub.add_parser("profile", help="Ficha de un MIDI")
    p.add_argument("midis", nargs="+")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(fn=cmd_profile)

    args = ap.parse_args(argv)
    if getattr(args, "no_color", False):
        _USE_COLOR = False
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
