#!/usr/bin/env python3
"""
plan2brief.py — Plan to Brief: convierte el documento de plan de una obra (.docx,
.md, .txt) en «briefs» de texto plano para reference_finder, uno global y uno por
sección
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
reference_finder.py puntúa un corpus MIDI contra un brief de criterios medibles,
pero solo lee texto plano «clave: valor». Un plan de obra real es un .docx con
varias secciones de carácter opuesto (p. ej. Re m pp → marcato → politonalidad →
tutti fff), y un único brief promedia todo. Hasta ahora había que escribir a mano
un brief global y uno por sección, y fundir los rankings a mano.

Esta herramienta lo automatiza con heurísticas de lectura (NO es comprensión de
lenguaje natural: extrae lo que el texto dice con patrones reconocibles) y deja
siempre la EVIDENCIA a la vista —el fragmento del plan del que sale cada valor—
para que el brief sea revisable antes de usarlo.

  · Lee .docx sin dependencias (zip + XML), .md y .txt.
  · Detecta secciones por sus encabezados («Sección I», «Sec. II», «Movimiento 3»,
    «Parte IV»…; patrón configurable con --section-pattern).
  · Extrae tonalidades (Re menor, Dm, F# minor…), tempo (♩=60, 52-72 BPM),
    duración (5 minutos), compás (4/4), instrumentos (es/en), células rítmicas
    (marcato, habanera, milonga…), cromatismo, repetición/ostinato, acumulación
    de capas, percusión y lo que debe EVITARSE (solo si el plan lo niega).
  · Escribe un brief por sección y uno global con el formato de reference_finder,
    más un JSON con todo lo extraído y su evidencia.
  · merge funde los rankings por sección (reference_finder rank --json) en una
    sola tabla que indica en qué sección destaca cada archivo.

Autocontenida: no importa ninguna otra herramienta de la suite; solo comparte el
FORMATO del brief y del JSON de ranking con reference_finder.py.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  extract   Plan → briefs (global + una por sección) + JSON de evidencias
  show      Lo que se extraería, con evidencias, sin escribir ficheros
  check     Valida un brief (claves, tonalidades, instrumentos, células)
  merge     Funde rankings por sección en una tabla única
  selftest  Prueba con un .docx sintético

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  extract
    plan PLAN                   .docx / .md / .txt
    --out, -o PREFIJO           prefijo de salida (defecto: <plan>; genera
                                PREFIJO.brief, PREFIJO_<sección>.brief y
                                PREFIJO.plan.json)
    --section-pattern RE        regex de encabezado de sección (grupo 1 = id)
    --no-sections               solo brief global
    --no-weights                no escribir la línea «weights:»
    --no-color

  show
    plan PLAN
    --section-pattern RE
    --no-color

  check
    briefs BRIEF […]
    --no-color

  merge
    rankings RANK […]           JSON de «reference_finder rank --json»; cada uno
                                como RUTA o ETIQUETA=RUTA (la etiqueta nombra la
                                sección; la primera «global» si se llama así)
    --top TOP                   filas (defecto 20)
    --json JSON                 guarda la tabla fundida
    --no-color

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[extract]
plan             file  io=in required  # .docx, .md o .txt
--out            str   alias=-o  # Prefijo de salida
--section-pattern str  # Regex de encabezado de sección (grupo 1 = id)
--no-sections    bool  default=false
--no-weights     bool  default=false
--no-color       bool  default=false
[show]
plan             file  io=in required
--section-pattern str
--no-color       bool  default=false
[check]
briefs           list  of=file io=in required
--no-color       bool  default=false
[merge]
rankings         list  of=str required  # RUTA o ETIQUETA=RUTA
--top            int   default=20 min=1
--json           file  io=out ext=json
--no-color       bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python plan2brief.py show plan_tango.docx
    python plan2brief.py extract plan_tango.docx -o briefs/tango
    python reference_finder.py rank --index idx.json --brief briefs/tango_i.brief --json r1.json
    python plan2brief.py merge global=rg.json I=r1.json II=r2.json --top 15
    python plan2brief.py check briefs/tango.brief

  NOTAS
    · Las alteraciones se escriben con ♯/♭ en el brief («F♯m»): reference_finder
      trata «#» como inicio de comentario.
    · Un valor mal extraído se corrige editando el .brief: el JSON conserva la
      evidencia para saber de dónde salió.

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 2 nada extraíble

  API:  from plan2brief import read_plan, split_sections, analyze, brief_text

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  (solo biblioteca estándar)
"""
import argparse
import json
import os
import re
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

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
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch)).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DEL PLAN
# ══════════════════════════════════════════════════════════════════════════════

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _read_docx(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    out = []
    for p in root.iter(_W + "p"):
        ppr = p.find(_W + "pPr")
        style = ""
        if ppr is not None:
            ps = ppr.find(_W + "pStyle")
            if ps is not None:
                style = ps.get(_W + "val", "")
        parts = []
        for el in p.iter():
            if el.tag == _W + "t" and el.text:
                parts.append(el.text)
            elif el.tag in (_W + "tab", _W + "br"):
                parts.append(" ")
        text = "".join(parts).strip()
        if text:
            out.append((style, text))
    return out


def read_plan(path):
    """→ lista de (estilo, texto) por párrafo. estilo = 'H' si es encabezado."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        rows = _read_docx(path)
        return [("H" if re.match(r"(heading|ttulo|titulo|title)", st.lower()) else "", tx)
                for st, tx in rows]
    with open(path, encoding="utf-8", errors="replace") as fh:
        rows = []
        for ln in fh:
            ln = ln.rstrip("\n")
            if not ln.strip():
                continue
            m = re.match(r"^\s{0,3}#{1,6}\s+(.*)$", ln)
            rows.append(("H", m.group(1).strip()) if m else ("", ln.strip()))
        return rows


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIONES
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_SECTION_RE = (r"^\s*(?:secci[oó]n|sec\.?|section|movimiento|mov\.?|parte|part)"
                      r"\s*([IVXLC]+|\d+)\b")
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _slug(s):
    s = _norm(s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s or "x"


def split_sections(rows, pattern=DEFAULT_SECTION_RE):
    """→ [(id, título, texto)]; el preámbulo (antes de la 1.ª sección) no es sección.
    Un encabezado es un párrafo de estilo título, o una línea corta (<=100) sin punto
    final, que encaja con `pattern`. Una misma sección repetida se concatena."""
    rx = re.compile(pattern, re.I)
    secs, order = {}, []
    cur = None
    for style, text in rows:
        short = len(text) <= 100 and not text.endswith(".")
        m = rx.match(text) if (style == "H" or short) else None
        if m:
            sid = m.group(1).upper() if m.groups() else str(len(order) + 1)
            cur = sid
            if sid not in secs:
                secs[sid] = [text, []]
                order.append(sid)
            continue
        if cur is not None:
            secs[cur][1].append(text)
    return [(sid, secs[sid][0], "\n".join(secs[sid][1])) for sid in order]


# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARIO (copia propia: la herramienta no importa reference_finder)
# ══════════════════════════════════════════════════════════════════════════════

# alias → nombre canónico del brief. Se excluyen palabras españolas ambiguas
# (bajo, clave, voz, metal…): el plan se lee en prosa.
INSTR = {
    "violin": "violin", "violín": "violin", "violines": "violin", "violino": "violin",
    "viola": "viola", "violas": "viola",
    "cello": "cello", "chelo": "cello", "violonchelo": "cello", "violoncello": "cello",
    "violonchelos": "cello", "contrabass": "contrabass", "contrabajo": "contrabass",
    "contrabajos": "contrabass",
    "cuerdas": "strings", "cuerda": "strings", "strings": "strings",
    "arpa": "harp", "harp": "harp", "piano": "piano", "clavecín": "harpsichord",
    "clavecin": "harpsichord", "harpsichord": "harpsichord",
    "órgano": "organ", "organo": "organ", "organ": "organ",
    "acordeón": "accordion", "acordeon": "accordion", "accordion": "accordion",
    "bandoneón": "bandoneon", "bandoneon": "bandoneon", "guitarra": "guitar",
    "guitar": "guitar",
    "flauta": "flute", "flutas": "flute", "flute": "flute", "oboe": "oboe",
    "oboes": "oboe", "clarinete": "clarinet", "clarinetes": "clarinet",
    "clarinet": "clarinet", "fagot": "bassoon", "fagots": "bassoon",
    "bassoon": "bassoon", "maderas": "woodwinds", "woodwinds": "woodwinds",
    "trompa": "horn", "trompas": "horn", "horn": "horn", "trompeta": "trumpet",
    "trompetas": "trumpet", "trumpet": "trumpet", "trombón": "trombone",
    "trombon": "trombone", "trombones": "trombone", "trombone": "trombone",
    "tuba": "tuba", "metales": "brass", "brass": "brass",
    "timbal": "timpani", "timbales": "timpani", "timpani": "timpani",
    "percusión": "percussion", "percusion": "percussion", "percussion": "percussion",
    "bombo": "percussion", "platillo": "percussion", "platillos": "percussion",
    "tam-tam": "percussion", "caja": "percussion",
    "coro": "choir", "choir": "choir",
}
# «caja» (instrumento) es ambigua en prosa: solo se acepta junto a bombo/platillo/timbal
_AMBIG = {"caja"}

CELLS = ("marcato", "marcato332", "habanera", "milonga", "marcha", "vals")
CELL_WORDS = [
    (r"\bmarcato\b", "marcato"), (r"\b3\s*\+\s*3\s*\+\s*2\b", "marcato332"),
    (r"\bhabanera\b", "habanera"), (r"\bmilonga\b", "milonga"),
    (r"\bmarcha\b", "marcha"), (r"\bvals\b|\bwaltz\b", "vals"),
]

_ES = {"do": "C", "re": "D", "mi": "E", "fa": "F", "sol": "G", "la": "A", "si": "B"}
_CAP_ONLY = {"la", "si", "mi", "do"}      # palabras comunes: exigir mayúscula inicial


def _key_name(letter, acc, minor):
    a = {"#": "♯", "♯": "♯", "b": "♭", "♭": "♭", "sostenido": "♯", "bemol": "♭"}.get(acc or "", "")
    return f"{letter}{a}{'m' if minor else ''}"


def _find_keys(text):
    """→ [(nombre, posición)] de tonalidades mencionadas."""
    out = []
    rx_es = re.compile(r"\b(Do|Re|Mi|Fa|Sol|La|Si)\s?(#|♯|♭|b\b|sostenido|bemol)?\s*"
                       r"(menor|mayor|m\b|M\b)(?!\s*(?:parte|medida|cantidad|vez|veces|"
                       r"manera|forma|tensi))", re.I)
    for m in rx_es.finditer(text):
        w = m.group(1)
        if w.lower() in _CAP_ONLY and not w[0].isupper():
            continue
        mode = m.group(3)
        if mode in ("m", "menor") or mode.lower() == "menor":
            minor = True
        else:
            minor = False
        if mode == "M" or mode.lower() == "mayor":
            minor = False
        out.append((_key_name(_ES[w.lower()], (m.group(2) or "").lower(), minor), m.start()))
    rx_en = re.compile(r"\b([A-G])([#♯♭b])?(?:\s(minor|major|min|maj)\b|(m)\b(?!\w))")
    for m in rx_en.finditer(text):
        acc = m.group(2)
        if acc == "b" and not (m.group(3) or m.group(4)):
            continue
        minor = bool(m.group(4)) or (m.group(3) or "").lower() in ("minor", "min")
        out.append((_key_name(m.group(1), acc, minor), m.start()))
    return out


def _snip(text, pos, w=48):
    a, b = max(0, pos - w), min(len(text), pos + w)
    return re.sub(r"\s+", " ", text[a:b]).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def analyze(text):
    """→ (campos, evidencia). campos usa las claves del brief de reference_finder."""
    F, E = {}, {}

    def ev(k, pos, txt=None):
        E.setdefault(k, [])
        if len(E[k]) < 3:
            E[k].append(_snip(txt if txt is not None else text, pos))

    # tonalidades: orden por frecuencia, desempate por primera aparición
    keys = {}
    for name, pos in _find_keys(text):
        c = keys.setdefault(name, [0, pos])
        c[0] += 1
        ev("key", pos)
    if keys:
        F["key"] = [k for k, _ in sorted(keys.items(), key=lambda kv: (-kv[1][0], kv[1][1]))][:4]

    # tempo
    nums, lo_hi = [], []
    for m in re.finditer(r"(\d{2,3})\s*(?:[-–—]|a)\s*(\d{2,3})\s*(?:bpm|ppm|♩|negras?\b)", text, re.I):
        lo_hi.append((int(m.group(1)), int(m.group(2)))); ev("tempo", m.start())
    for m in re.finditer(r"(?:bpm|♩|negra)\s*[=:]\s*(\d{2,3})(?:\s*[-–—]\s*(\d{2,3}))?", text, re.I):
        a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
        lo_hi.append((a, b)); ev("tempo", m.start())
    for m in re.finditer(r"\b(\d{2,3})\s*bpm\b", text, re.I):
        a = int(m.group(1)); lo_hi.append((a, a)); ev("tempo", m.start())
    for a, b in lo_hi:
        nums += [a, b]
    nums = [n for n in nums if 30 <= n <= 250]
    if nums:
        lo, hi = min(nums), max(nums)
        if lo == hi:
            lo, hi = int(round(lo * 0.9)), int(round(hi * 1.1))
        F["tempo"] = f"{lo}-{hi}"

    # duración (minutos siempre; segundos solo junto a «duración»)
    secs = []
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)(?:\s*[-–—]\s*(\d+(?:[.,]\d+)?))?\s*(?:minutos?|min\b)", text, re.I):
        for g in (m.group(1), m.group(2)):
            if g:
                secs.append(float(g.replace(",", ".")) * 60)
        ev("duration", m.start())
    for m in re.finditer(r"dura(?:ci[oó]n|n|r)?[^.\n]{0,25}?(\d+)\s*(?:s\b|seg|segundos)", text, re.I):
        secs.append(float(m.group(1))); ev("duration", m.start())
    if secs:
        F["duration"] = f"{int(min(secs))}-{int(max(secs))}"

    # compás
    meters = {}
    for m in re.finditer(r"(?<![\d/])([2-9]|1[0-2])/(2|4|8|16)(?![\d/])", text):
        meters[f"{m.group(1)}/{m.group(2)}"] = meters.get(f"{m.group(1)}/{m.group(2)}", 0) + 1
        ev("meter", m.start())
    if meters:
        F["meter"] = [k for k, _ in sorted(meters.items(), key=lambda kv: -kv[1])][:2]

    # instrumentos
    cnt, first = {}, {}
    for m in re.finditer(r"[\wáéíóúñü\-]+", text, re.I):
        w = m.group(0).lower()
        if w in INSTR and w not in _AMBIG:
            c = INSTR[w]
            cnt[c] = cnt.get(c, 0) + 1
            first.setdefault(c, m.start())
            ev("instruments", m.start())
    if re.search(r"\b(bombo|platillos?|tam-tam|timbal(?:es)?)\b", text, re.I) and "caja" in text.lower():
        cnt["percussion"] = cnt.get("percussion", 0) + 1
    if cnt:
        order = sorted(cnt, key=lambda c: (-cnt[c], first.get(c, 0)))
        top2 = [c for c in order if cnt[c] >= 3][:2]
        F["instruments"] = [("*" + c if c in top2 else c) for c in order][:12]

    # células rítmicas
    cells = []
    for rx, name in CELL_WORDS:
        m = re.search(rx, text, re.I)
        if m:
            cells.append(name); ev("rhythm", m.start())
    for m in re.finditer(r"(?<![\w.])([x.]{4,16})(?![\w.])", text):
        g = m.group(1)
        if "x" in g and "." in g:
            cells.append(g); ev("rhythm", m.start())
    if cells:
        F["rhythm"] = list(dict.fromkeys(cells))

    # cromatismo
    chrom = re.search(r"crom[aá]t(?:ic|ism)", text, re.I)
    diat = re.search(r"diat[oó]nic|pentat[oó]nic|modal\b", text, re.I)
    if chrom and diat:
        F["chromatic"] = "mid"; ev("chromatic", chrom.start())
    elif chrom:
        F["chromatic"] = "high"; ev("chromatic", chrom.start())
    elif diat:
        F["chromatic"] = "low"; ev("chromatic", diat.start())

    # repetición / ostinato
    m = re.search(r"ostinato|repetici[oó]n|repite|bucle|loop", text, re.I)
    if m and not re.search(r"sin\s+repetici|no\s+se\s+repite", text, re.I):
        F["repetition"] = "0.4-1"; ev("repetition", m.start())

    # acumulación de capas
    m = re.search(r"acumul|progresiv|entradas?\s+(?:sucesiv|escalonad|progresiv)|capas?\b|"
                  r"entra(?:n)?\s+(?:uno|cada)|layer", text, re.I)
    if m:
        F["accumulation"] = "yes"; ev("accumulation", m.start())

    # percusión
    m = re.search(r"timbal|percusi[oó]n|bombo|platillo|tam-tam|timpani|percussion", text, re.I)
    if m:
        F["percussion"] = "yes"; ev("percussion", m.start())

    # evitar: solo si el plan lo niega explícitamente
    m = re.search(r"\b(?:sin|no\s+(?:hay|usar|incluir|lleva)|evitar|ni)\s+(?:la\s+)?bater[ií]a|"
                  r"\bno\s+drums\b|\bwithout\s+drums\b", text, re.I)
    if m:
        F["avoid"] = ["drums"]; ev("avoid", m.start())
    return F, E


_BASE_WEIGHTS = {"instruments": 3, "key": 2, "rhythm": 3, "accumulation": 2,
                 "chromatic": 1.5, "repetition": 1.5}


def brief_text(fields, evidence=None, title="", weights=True, inherit=None):
    """Formato «clave: valor» de reference_finder. `inherit` (campos del brief
    global) rellena solo meter cuando la sección no lo menciona."""
    f = dict(fields)
    if inherit and "meter" not in f and "meter" in inherit:
        f["meter"] = inherit["meter"]
    evidence = evidence or {}
    L = [f"# Brief generado por plan2brief.py {VERSION}" + (f" — {title}" if title else ""),
         "# Revisa los valores: la extracción es heurística. La evidencia va tras «#».", ""]

    def line(k, v, ek=None):
        e = (evidence.get(ek or k) or [""])[0]
        L.append(f"{k}: {v}" + (f"      # «{e}»" if e else ""))

    order = ["key", "tempo", "duration", "meter", "instruments", "avoid", "rhythm",
             "repetition", "chromatic", "accumulation", "percussion"]
    for k in order:
        if k not in f:
            continue
        v = f[k]
        line(k, ", ".join(v) if isinstance(v, list) else v)
    if weights:
        w = [f"{k}={_BASE_WEIGHTS[k]}" for k in _BASE_WEIGHTS if k in f]
        if w:
            L.append("weights: " + ", ".join(w))
    return "\n".join(L) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN DE UN BRIEF (misma gramática que reference_finder)
# ══════════════════════════════════════════════════════════════════════════════

_KNOWN = {"key", "tonalidad", "tonalidades", "tempo", "bpm", "duration", "duracion", "meter",
          "compas", "instruments", "instrumentos", "avoid", "evitar", "rhythm", "ritmo",
          "repetition", "repeticion", "chromatic", "cromatismo", "accumulation",
          "acumulacion", "percussion", "percusion", "weights", "pesos"}
_INSTR_OK = set(INSTR) | set(INSTR.values()) | {"bass", "bajo", "voz", "voice", "drums", "bateria",
                                                 "clave", "synth", "keys", "plucked"}


def check_brief(text):
    probs = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = re.split(r"(?:^|\s)#", raw, 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            probs.append((n, f"línea sin «clave: valor»: {raw.strip()}")); continue
        k, v = line.split(":", 1)
        k, v = _norm(k), v.strip()
        if k not in _KNOWN:
            probs.append((n, f"clave desconocida: {k}")); continue
        if k in ("key", "tonalidad", "tonalidades"):
            for x in [y.strip() for y in re.split(r"[;,]", v) if y.strip()]:
                if not re.match(r"^[A-Ga-g][#b♯♭]?\s*(m|min|minor|menor|maj|major|mayor)?$", x):
                    probs.append((n, f"tonalidad no reconocida: {x}"))
        elif k in ("instruments", "instrumentos", "avoid", "evitar"):
            for x in [y.strip() for y in re.split(r"[;,]", v) if y.strip()]:
                if _norm(x.lstrip("*")) not in {_norm(a) for a in _INSTR_OK}:
                    probs.append((n, f"instrumento desconocido: {x}"))
        elif k in ("rhythm", "ritmo"):
            for x in [y.strip() for y in re.split(r"[;,]", v) if y.strip()]:
                if not (_norm(x) in CELLS or re.fullmatch(r"[x.]+", _norm(x))):
                    probs.append((n, f"célula rítmica desconocida: {x}"))
        elif k in ("tempo", "bpm", "duration", "duracion", "repetition", "repeticion"):
            if not re.search(r"\d", v):
                probs.append((n, f"{k}: se esperaba un número o rango"))
    return probs


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE DE RANKINGS
# ══════════════════════════════════════════════════════════════════════════════

def merge_rankings(items):
    """items: [(etiqueta, lista_de_filas)] → filas fundidas por archivo."""
    by = {}
    labels = []
    for lab, rows in items:
        labels.append(lab)
        for r in rows:
            d = by.setdefault(r["path"], {"path": r["path"], "scores": {},
                                          "key": r.get("key"), "tempo": r.get("tempo"),
                                          "duration_s": r.get("duration_s")})
            d["scores"][lab] = float(r.get("score", 0))
    out = []
    for d in by.values():
        sc = d["scores"]
        best = max(sc, key=sc.get)
        d["best_section"] = best
        d["best_score"] = sc[best]
        d["mean"] = sum(sc.get(l, 0.0) for l in labels) / max(1, len(labels))
        d["covers"] = len(sc)
        out.append(d)
    out.sort(key=lambda d: (-d["best_score"], -d["mean"], d["path"]))
    return labels, out


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def _load(args):
    rows = read_plan(args.plan)
    full = "\n".join(t for _s, t in rows)
    secs = [] if getattr(args, "no_sections", False) else \
        split_sections(rows, args.section_pattern or DEFAULT_SECTION_RE)
    return rows, full, secs


def _print_fields(F, E, indent="  "):
    for k, v in F.items():
        vs = ", ".join(v) if isinstance(v, list) else v
        print(f"{indent}{_c('B')}{k:<12}{_c('R')} {vs}")
        for e in (E.get(k) or [])[:2]:
            print(f"{indent}{'':<12} {_c('G')}«{e}»{_c('R')}")


def cmd_show(a):
    if not os.path.exists(a.plan):
        print(f"[ERROR] no existe: {a.plan}", file=sys.stderr); return 1
    _rows, full, secs = _load(a)
    F, E = analyze(full)
    print(f"{_c('CYA')}GLOBAL{_c('R')}  ({len(full)} caracteres, {len(secs)} secciones)")
    _print_fields(F, E)
    for sid, title, text in secs:
        Fs, Es = analyze(text)
        print(f"\n{_c('CYA')}SECCIÓN {sid}{_c('R')}  {title[:70]}")
        _print_fields(Fs, Es)
    if not F and not secs:
        print("[AVISO] no se extrajo nada reconocible", file=sys.stderr); return 2
    return 0


def cmd_extract(a):
    if not os.path.exists(a.plan):
        print(f"[ERROR] no existe: {a.plan}", file=sys.stderr); return 1
    _rows, full, secs = _load(a)
    prefix = a.out or os.path.splitext(a.plan)[0]
    F, E = analyze(full)
    if not F:
        print("[AVISO] no se extrajo nada reconocible del plan", file=sys.stderr); return 2
    written = []
    p = _ensure_parent(prefix + ".brief")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(brief_text(F, E, "global", not a.no_weights))
    written.append(p)
    data = {"plan": os.path.basename(a.plan), "version": VERSION,
            "global": {"fields": F, "evidence": E, "brief": os.path.basename(p)},
            "sections": []}
    for sid, title, text in secs:
        Fs, Es = analyze(text)
        if not Fs:
            print(f"[AVISO] sección {sid}: sin criterios reconocibles; no se escribe brief",
                  file=sys.stderr)
            continue
        sp = _ensure_parent(f"{prefix}_{_slug(sid)}.brief")
        with open(sp, "w", encoding="utf-8") as fh:
            fh.write(brief_text(Fs, Es, f"sección {sid}: {title[:60]}", not a.no_weights, inherit=F))
        written.append(sp)
        data["sections"].append({"id": sid, "title": title, "fields": Fs, "evidence": Es,
                                 "brief": os.path.basename(sp)})
    jp = _ensure_parent(prefix + ".plan.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
    written.append(jp)
    for w in written:
        print(f"  → {w}")
    print(f"\n{len(written) - 1} brief(s) · revisa los valores y lanza reference_finder rank "
          f"--brief <brief> --json <ranking>.json")
    return 0


def cmd_check(a):
    bad = 0
    for p in a.briefs:
        if not os.path.exists(p):
            print(f"[ERROR] no existe: {p}", file=sys.stderr); bad += 1; continue
        probs = check_brief(open(p, encoding="utf-8").read())
        if probs:
            bad += 1
            print(f"{_c('YEL')}{p}{_c('R')}")
            for n, m in probs:
                print(f"  línea {n}: {m}")
        else:
            print(f"{_c('GRN')}✓{_c('R')} {p}")
    return 1 if bad else 0


def cmd_merge(a):
    items = []
    for spec in a.rankings:
        lab, _, path = spec.partition("=") if "=" in spec else (None, "", spec)
        if lab is None:
            lab = os.path.splitext(os.path.basename(path))[0]
        if not os.path.exists(path):
            print(f"[ERROR] no existe: {path}", file=sys.stderr); return 1
        try:
            rows = json.load(open(path, encoding="utf-8"))
        except ValueError as e:
            print(f"[ERROR] JSON inválido en {path}: {e}", file=sys.stderr); return 1
        if isinstance(rows, dict):
            rows = rows.get("results") or rows.get("ranking") or []
        items.append((lab, rows))
    labels, out = merge_rankings(items)
    if not out:
        print("[AVISO] los rankings están vacíos", file=sys.stderr); return 2
    head = f"{'#':>3}  {'archivo':<28} {'mejor':<10} {'pts':>5} {'media':>6}  " + \
           " ".join(f"{l[:7]:>7}" for l in labels)
    print(head)
    for i, d in enumerate(out[:a.top], 1):
        cols = " ".join(f"{d['scores'].get(l, 0):7.1f}" if l in d["scores"] else f"{'·':>7}" for l in labels)
        print(f"{i:>3}  {os.path.basename(d['path'])[:28]:<28} {d['best_section'][:10]:<10} "
              f"{d['best_score']:5.1f} {d['mean']:6.1f}  {cols}")
    if a.json:
        with open(_ensure_parent(a.json), "w", encoding="utf-8") as fh:
            json.dump({"labels": labels, "rows": out}, fh, ensure_ascii=False, indent=1)
        print(f"  → {a.json}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST
# ══════════════════════════════════════════════════════════════════════════════

def _make_docx(path, paragraphs):
    """paragraphs: [(estilo|None, texto)] → .docx mínimo válido."""
    W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = []
    for st, tx in paragraphs:
        tx = tx.replace("&", "&amp;").replace("<", "&lt;")
        ppr = f'<w:pPr><w:pStyle w:val="{st}"/></w:pPr>' if st else ""
        body.append(f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{tx}</w:t></w:r></w:p>')
    doc = f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{W}"><w:body>' + \
          "".join(body) + "</w:body></w:document>"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="xml" ContentType="application/xml"/></Types>')
        z.writestr("word/document.xml", doc)


def cmd_selftest(_a):
    import tempfile
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name}")
        ok = ok and bool(cond)

    paras = [
        ("Heading1", "Entre el Tango y el Épico"),
        (None, "Obra de unos 6 minutos en 4/4. Instrumentos: viola, violín, violonchelo, viola, viola y piano."),
        ("Heading2", "Sección I — La Memoria"),
        (None, "Re menor natural, ♩ = 52, pp. La viola sola y el violonchelo. Textura diatónica, sin batería."),
        ("Heading2", "Sección II — El Marcato"),
        (None, "Re menor armónico, 70-100 bpm. Marcato (x.xxx...) en contrabajo y piano; cromatismo "
               "creciente. Entradas progresivas de capas. Timbales pp."),
        ("Heading2", "Sección IV — Tutti"),
        (None, "La mayor, 125-150 BPM, tutti fff con trompas, trompetas y timbales. Ostinato de marcha. F# minor al final."),
    ]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.docx")
        _make_docx(p, paras)
        rows = read_plan(p)
        chk("lee .docx (8 párrafos)", len(rows) == 8)
        full = "\n".join(t for _s, t in rows)
        secs = split_sections(rows)
        chk("detecta 3 secciones I, II, IV", [s[0] for s in secs] == ["I", "II", "IV"])
        F, _E = analyze(full)
        chk("global: meter 4/4", F.get("meter") == ["4/4"])
        chk("global: duración 360 s", F.get("duration") == "360-360")
        F1, _ = analyze(secs[0][2])
        chk("I: Dm", F1.get("key") == ["Dm"])
        chk("I: tempo 52 ±10 %", F1.get("tempo") == "47-57")
        chk("I: cromatismo low y avoid drums", F1.get("chromatic") == "low" and F1.get("avoid") == ["drums"])
        F2, _ = analyze(secs[1][2])
        chk("II: marcato + patrón explícito", F2.get("rhythm") == ["marcato", "x.xxx..."])
        chk("II: acumulación, percusión, cromatismo high",
            F2.get("accumulation") == "yes" and F2.get("percussion") == "yes" and F2.get("chromatic") == "high")
        chk("II: tempo 70-100", F2.get("tempo") == "70-100")
        F4, _ = analyze(secs[2][2])
        chk("IV: La mayor → A, y F♯m", "A" in F4.get("key", []) and "F♯m" in F4.get("key", []))
        chk("IV: ostinato → repetición; marcha", F4.get("repetition") == "0.4-1" and "marcha" in F4.get("rhythm", []))
        chk("no confunde «La viola» con La mayor", "A" not in F1.get("key", []))
        out = os.path.join(d, "o", "t")
        class A: pass
        a = A(); a.plan = p; a.out = out; a.section_pattern = None; a.no_sections = False; a.no_weights = False
        rc = cmd_extract(a)
        chk("extract: rc 0 y 5 ficheros", rc == 0 and len(os.listdir(os.path.join(d, "o"))) == 5)
        chk("briefs generados pasan check",
            all(not check_brief(open(os.path.join(d, "o", f), encoding="utf-8").read())
                for f in os.listdir(os.path.join(d, "o")) if f.endswith(".brief")))
        rows_a = [{"path": "a.mid", "score": 80}, {"path": "b.mid", "score": 40}]
        rows_b = [{"path": "a.mid", "score": 30}, {"path": "b.mid", "score": 90}, {"path": "c.mid", "score": 10}]
        labels, merged = merge_rankings([("I", rows_a), ("II", rows_b)])
        chk("merge: b.mid destaca en II", merged[0]["path"] == "b.mid" and merged[0]["best_section"] == "II")
        chk("merge: c.mid cubre 1 sección", [m for m in merged if m["path"] == "c.mid"][0]["covers"] == 1)
    print("selftest:", "OK" if ok else "FALLOS")
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(prog="plan2brief.py", description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("extract", help="Plan → briefs (global + por sección)")
    sp.add_argument("plan")
    sp.add_argument("--out", "-o")
    sp.add_argument("--section-pattern")
    sp.add_argument("--no-sections", action="store_true")
    sp.add_argument("--no-weights", action="store_true")
    sp.add_argument("--no-color", action="store_true")
    sp = sub.add_parser("show", help="Qué se extraería, con evidencias")
    sp.add_argument("plan")
    sp.add_argument("--section-pattern")
    sp.add_argument("--no-color", action="store_true")
    sp = sub.add_parser("check", help="Validar briefs")
    sp.add_argument("briefs", nargs="+")
    sp.add_argument("--no-color", action="store_true")
    sp = sub.add_parser("merge", help="Fundir rankings por sección")
    sp.add_argument("rankings", nargs="+")
    sp.add_argument("--top", type=int, default=20)
    sp.add_argument("--json")
    sp.add_argument("--no-color", action="store_true")
    sub.add_parser("selftest", help="Prueba con un .docx sintético")
    return p


def main(argv=None):
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if getattr(a, "no_color", False):
        _USE_COLOR = False
    return {"extract": cmd_extract, "show": cmd_show, "check": cmd_check,
            "merge": cmd_merge, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
