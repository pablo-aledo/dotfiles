#!/usr/bin/env python3
"""
orchestral_percussion.py — Orchestral Percussion: percusión de orquesta (timbales
afinados, bombo, caja, platos, plato suspendido, tam-tam) escrita por SECCIONES con
dinámica y patrón, o derivada de la energía de un MIDI existente
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
drummer.py es un motor de batería/grooves (rock, funk, jazz, tango como kit) y
orchestrator.py genera una percusión orquestal mínima como subproducto de la
orquestación (4/4 fijo, guiada por huellas de sección). Faltaba una herramienta
para ESCRIBIR percusión orquestal con intención: «compases 1-8 pp latido,
9-16 mf marcato, 17-24 f>ff ostinato, 25-32 fff tutti», con los timbales afinados
a la tónica y la dominante de cada pasaje y cualquier compás.

Se trabaja con una ESPECIFICACIÓN de texto —legible, editable, versionable—:

    1-8 pp heartbeat; 9-16 mf marcato; 17-28 mf>ff surge cycle=4; 29-36 fff hits key=A

y dos formas de obtenerla: escribirla a mano, o pedir a `analyze` que la proponga
a partir de la energía de un MIDI (densidad de ataques, velocidad y polifonía por
compás) para luego retocarla y aplicarla con `add`.

PATRONES   tacet · pulse · heartbeat · marcato · habanera · milonga · march ·
           ostinato · roll · hits · swell · surge
           (marcato = ♩♪♪♩ de 3 pulsos con bombo en las negras; surge = «oleaje»:
           ciclos de N compases con redoble creciente, pico abrupto con platos y
           tam-tam y cola larga, sin caer nunca a silencio entre ciclos)
DINÁMICAS  ppp pp p mp mf f ff fff, o rampa «pp>ff» a lo largo de la sección
OPCIONES   key=Dm  every=N  cycle=N  div=N  vel=NN  (por sección)

Autocontenida: solo mido. No importa ninguna otra herramienta de la suite; las
notas GM del kit coinciden con las de orchestrator.py (bombo 36, caja 38,
platos 49, plato suspendido 51, tam-tam 54) y se pueden cambiar con --map.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  generate  Escribe un MIDI solo de percusión a partir de una especificación
  add       Añade la percusión a un MIDI existente (spec manual o --auto)
  analyze   Curva de energía por compás y especificación propuesta
  list      Patrones, dinámicas e instrumentos
  selftest  Prueba con MIDIs sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  generate
    --spec, -s SPEC             especificación (secciones separadas por «;»)
    --spec-file FILE            especificación en un fichero (una por línea)
    --out, -o OUT               MIDI de salida (defecto percusion.mid)
    --key KEY                   tonalidad por defecto de los timbales (Dm)
    --meter M                   compás (4/4)
    --bpm BPM                   tempo (72)
    --split                     una pista por instrumento
    --map NOMBRE=NOTA […]       notas GM del kit (bombo, caja, platos, plato_susp, tamtam)
    --seed N                    humanización reproducible (defecto 7)
    --no-humanize

  add
    midi MIDI                   obra a la que añadir percusión
    --spec / --spec-file        especificación; o bien
    --auto                      la deriva de la energía de la obra
    --key KEY                   tonalidad (defecto: detectada en la obra)
    --out, -o OUT               defecto <obra>_perc.mid
    --split · --map · --seed · --no-humanize
    --replace                   quita las pistas de percusión (canal 10) existentes

  analyze
    midi MIDI
    --key KEY
    --no-color

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[generate]
--spec       str   alias=-s format="1-8 pp pulse; 9-16 mf marcato"
--spec-file  file  io=in
--out        file  io=out ext=mid default=percusion.mid alias=-o
--key        str   default=Dm
--meter      str   default=4/4
--bpm        float default=72 min=20 max=300 unit=BPM
--split      bool  default=false
--map        list  of=str format="bombo=36"
--seed       int   default=7
--no-humanize bool default=false
[add]
midi         file  io=in ext=mid required
--spec       str   alias=-s
--spec-file  file  io=in
--auto       bool  default=false
--key        str
--out        file  io=out ext=mid alias=-o
--split      bool  default=false
--map        list  of=str
--seed       int   default=7
--no-humanize bool default=false
--replace    bool  default=false
[analyze]
midi         file  io=in ext=mid required
--key        str
--no-color   bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python orchestral_percussion.py list
    python orchestral_percussion.py generate -s "1-8 pp heartbeat; 9-16 mf marcato; 17-32 mf>ff surge cycle=4; 33-40 fff hits key=A" --key Dm --bpm 66 -o perc.mid
    python orchestral_percussion.py analyze obra.mid
    python orchestral_percussion.py add obra.mid --auto -o obra_perc.mid
    python orchestral_percussion.py add obra.mid --spec-file percusion.txt --split

  NOTAS
    · Los compases se numeran desde 1. La rejilla es de corcheas; en 6/8 y 3/4 el
      compás tiene 6 corcheas, en 2/4 tiene 4, en 4/4 tiene 8.
    · Con cambios de compás en la obra se usa el dominante (aviso en stderr).
    · Los timbales van en el canal 16 (programa 47, afinados a tónica y dominante
      dentro de F2-C4) y el kit en el canal 10, como orchestrator.py.

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso o de especificación · 2 nada que escribir

  API:  from orchestral_percussion import parse_spec, render, auto_spec

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  mido
"""
import argparse
import math
import os
import random
import re
import sys

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "1.0"
_COL = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
        "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
SPARK = "▁▂▃▄▅▆▇█"


def _c(k):
    return _COL.get(k, "") if _USE_COLOR else ""


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGOS
# ══════════════════════════════════════════════════════════════════════════════

DYN = {"ppp": 26, "pp": 38, "p": 52, "mp": 64, "mf": 78, "f": 94, "ff": 110, "fff": 124}
KIT = {"bombo": 36, "caja": 38, "platos": 49, "plato_susp": 51, "tamtam": 54}
INSTR_ALIAS = {"kick": "bombo", "snare": "caja", "crash": "platos", "sus_cym": "plato_susp",
               "plato": "plato_susp", "tam-tam": "tamtam", "tam_tam": "tamtam"}
TIMP_CH, KIT_CH, TIMP_PROG = 15, 9, 47
TIMP_LO, TIMP_HI = 41, 60            # F2-C4: par de timbales estándar

PATTERNS = {
    "tacet": "silencio",
    "pulse": "timbal en el 1.º tiempo, largo; el pulso mínimo",
    "heartbeat": "latido: timbal en el 1 y un segundo golpe más suave a mitad de compás",
    "marcato": "♩♪♪♩ (x.xxx.): bombo en las negras, timbal tónica/dominante en las corcheas",
    "habanera": "x..xx.x. — bombo + timbal sincopados",
    "milonga": "x.x.xx.. — timbal con bombo en 1 y 3",
    "march": "bombo en 1 y 3, caja en 2 y 4, timbal en el 1",
    "ostinato": "timbal continuo (div=1..4 por tiempo), acento en el 1, alterna tónica/dominante",
    "roll": "redoble de timbal con densidad creciente a lo largo de la sección",
    "hits": "golpes de tutti cada every=N compases: bombo+timbal+platos (+tam-tam al empezar)",
    "swell": "plato suspendido largo creciente cada every=N compases",
    "surge": "oleaje: ciclos de cycle=N compases; redoble creciente, pico con platos y tam-tam, cola larga",
}


def parse_key(txt):
    """«Dm», «F#m», «Re menor», «A» → (pc, menor?). None si no se reconoce."""
    t = (txt or "").strip()
    es = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11}
    en = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    m = re.match(r"^(do|re|mi|fa|sol|la|si)\s*(#|♯|b|♭)?\s*(menor|mayor|m|min|maj)?$", t, re.I)
    if m:
        base = es[m.group(1).lower()]
    else:
        m = re.match(r"^([A-Ga-g])\s*(#|♯|b|♭)?\s*(minor|major|min|maj|m|M)?$", t)
        if not m:
            return None
        base = en[m.group(1).lower()]
    acc = {"#": 1, "♯": 1, "b": -1, "♭": -1}.get(m.group(2) or "", 0)
    mode = m.group(3) or ""
    return (base + acc) % 12, mode.lower() in ("menor", "m", "min", "minor") and mode != "M"


def timpani_pair(key):
    """(tónica, dominante) de los timbales dentro de F2-C4; la dominante, por debajo."""
    pc, _minor = key
    tonic = next(p for p in range(TIMP_LO + 5, TIMP_HI + 6) if p % 12 == pc and p <= TIMP_HI)
    dom = tonic - 5
    if dom < TIMP_LO:
        dom = tonic + 7 if tonic + 7 <= TIMP_HI + 7 else tonic - 5 + 12
    return tonic, dom


# ══════════════════════════════════════════════════════════════════════════════
#  ESPECIFICACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class SpecError(ValueError):
    pass


def parse_spec(text, default_key="Dm"):
    """«1-8 pp pulse; 9-16 mf>ff roll key=A» → [dict(start, end, v0, v1, pattern, opts, key)]."""
    secs = []
    dk = parse_key(default_key)
    if dk is None:
        raise SpecError(f"tonalidad no reconocida: {default_key}")
    for raw in re.split(r"[;\n]", text or ""):
        raw = raw.split("//", 1)[0].strip()
        if not raw:
            continue
        tok = raw.split()
        if len(tok) < 3:
            raise SpecError(f"sección incompleta (se espera «compases dinámica patrón»): {raw!r}")
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", tok[0])
        if not m:
            raise SpecError(f"compases inválidos: {tok[0]!r}")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if a < 1 or b < a:
            raise SpecError(f"rango de compases inválido: {tok[0]!r}")
        dyn = tok[1].lower().split(">")
        if len(dyn) > 2 or any(d not in DYN for d in dyn):
            raise SpecError(f"dinámica inválida: {tok[1]!r} (usa {', '.join(DYN)} o «pp>ff»)")
        pat = tok[2].lower()
        if pat not in PATTERNS:
            raise SpecError(f"patrón desconocido: {tok[2]!r} (usa {', '.join(PATTERNS)})")
        opts = {}
        for kv in tok[3:]:
            if "=" not in kv:
                raise SpecError(f"opción inválida (se espera clave=valor): {kv!r}")
            k, v = kv.split("=", 1)
            opts[k.lower()] = v
        key = dk
        if "key" in opts:
            key = parse_key(opts["key"])
            if key is None:
                raise SpecError(f"tonalidad no reconocida: {opts['key']}")
        for k in ("every", "cycle", "div", "vel"):
            if k in opts:
                try:
                    opts[k] = int(opts[k])
                except ValueError:
                    raise SpecError(f"{k} debe ser un entero: {opts[k]!r}")
        v0 = DYN[dyn[0]]
        v1 = DYN[dyn[-1]]
        if "vel" in opts:
            v0 = v1 = max(1, min(127, opts["vel"]))
        secs.append({"start": a, "end": b, "v0": v0, "v1": v1, "pattern": pat,
                     "opts": opts, "key": key, "text": raw})
    secs.sort(key=lambda s: s["start"])
    for x, y in zip(secs, secs[1:]):
        if y["start"] <= x["end"]:
            raise SpecError(f"secciones solapadas: «{x['text']}» y «{y['text']}»")
    return secs


# ══════════════════════════════════════════════════════════════════════════════
#  PATRONES  →  eventos (pos en corcheas desde el inicio del compás, instr, long, acento)
# ══════════════════════════════════════════════════════════════════════════════

def _grid(pat, bar8):
    """Cadena x. recortada/ampliada a bar8 corcheas."""
    return (pat * 4)[:bar8] if len(pat) < bar8 else pat[:bar8]


def _from_cell(cell, bar8, inst, acc=1.0, ln=1.5):
    return [(i, inst, ln, acc) for i, ch in enumerate(_grid(cell, bar8)) if ch == "x"]


def _pat_events(name, ctx):
    bar8, bi, nb = ctx["bar8"], ctx["bi"], ctx["nb"]
    o = ctx["opts"]
    T, D = ctx["tonic"], ctx["dom"]
    ev = []                                     # (pos8, inst, len8, acc, pitch|None)
    half = bar8 // 2
    if name == "tacet":
        return ev
    if name == "pulse":
        ev.append((0, "timpani", bar8 - 0.5, 1.0, T))
    elif name == "heartbeat":
        ev.append((0, "timpani", 2, 1.0, T))
        ev.append((max(2, half - 1) if bar8 > 4 else 2, "timpani", 2, 0.62, D))
    elif name == "marcato":
        cell = "x.xxx." if bar8 >= 6 else "x.xx"
        for i, ch in enumerate(cell):
            if ch != "x":
                continue
            if i in (0, 4):
                ev.append((i, "bombo", 1.5, 1.0, None))
            ev.append((i, "timpani", 1.2, 1.0 if i in (0, 4) else 0.8, T if i in (0, 4) else D))
    elif name == "habanera":
        cell = _grid("x..xx.x.", bar8)
        for i, ch in enumerate(cell):
            if ch == "x":
                ev.append((i, "timpani", 1.2, 1.0 if i == 0 else 0.8, T if i in (0, 4) else D))
        ev.append((0, "bombo", 1.5, 0.9, None))
    elif name == "milonga":
        cell = _grid("x.x.xx..", bar8)
        for i, ch in enumerate(cell):
            if ch == "x":
                ev.append((i, "timpani", 1.2, 1.0 if i == 0 else 0.75, T if i % 4 == 0 else D))
        ev.append((0, "bombo", 1.5, 0.9, None))
        if bar8 >= 8:
            ev.append((4, "bombo", 1.5, 0.8, None))
    elif name == "march":
        beats = max(2, bar8 // 2)
        for b in range(beats):
            pos = b * 2
            if b % 2 == 0:
                ev.append((pos, "bombo", 1.5, 1.0, None))
            else:
                ev.append((pos, "caja", 1.0, 0.7, None))
        ev.append((0, "timpani", 2, 0.9, T))
    elif name == "ostinato":
        div = max(1, min(4, o.get("div", 2)))
        step = 2.0 / div
        n = int(bar8 / step)
        for k in range(n):
            pos = k * step
            ev.append((pos, "timpani", step * 0.9, 1.0 if k == 0 else 0.7 if (pos % 2 == 0) else 0.55,
                       T if (k // div) % 2 == 0 else D))
    elif name == "roll":
        prog = ctx["prog"]
        div = (1, 2, 4, 8)[min(3, int(prog * 4))]
        step = 2.0 / div
        for k in range(int(bar8 / step)):
            pos = k * step
            ev.append((pos, "timpani", step * 0.95, 1.0 if k == 0 else 0.85, T))
    elif name == "hits":
        every = max(1, o.get("every", 2))
        if bi % every == 0:
            ev.append((0, "bombo", 2, 1.0, None))
            ev.append((0, "timpani", 3, 1.0, T))
            ev.append((0, "platos", 4, 0.9, None))
            if bi == 0:
                ev.append((0, "tamtam", bar8 * 2, 0.9, None))
            if bar8 >= 8:
                ev.append((4, "timpani", 2, 0.8, D))
    elif name == "swell":
        every = max(1, o.get("every", 4))
        if bi % every == 0:
            ev.append((0, "plato_susp", min(bar8 * every, bar8 * 4), 0.4 + 0.6 * ctx["prog"], None))
    elif name == "surge":
        cyc = max(2, o.get("cycle", 4))
        ci = bi % cyc
        if ci < cyc - 1:                              # subida: redoble cada vez más denso
            p = (ci + 1) / cyc
            div = (1, 2, 4, 8)[min(3, int(p * 4))]
            step = 2.0 / div
            for k in range(int(bar8 / step)):
                ev.append((k * step, "timpani", step * 0.95, (0.45 + 0.55 * p) * (1.0 if k else 1.1), T))
        else:                                         # pico abrupto + cola larga
            ev.append((0, "bombo", 3, 1.0, None))
            ev.append((0, "timpani", 4, 1.0, T))
            ev.append((0, "platos", 6, 1.0, None))
            ev.append((0, "tamtam", bar8 * 2, 0.95, None))
            ev.append((bar8 // 2, "timpani", 2, 0.5, D))     # la cola no cae a silencio
    return ev


def render(secs, bar8, tpb, bar_ticks, humanize=True, seed=7):
    """→ {instr: [(tick, dur_ticks, pitch|None, vel)]}"""
    rnd = random.Random(seed)
    out = {k: [] for k in ("timpani", "bombo", "caja", "platos", "plato_susp", "tamtam")}
    eighth = tpb / 2.0
    for s in secs:
        nb = s["end"] - s["start"] + 1
        T, D = timpani_pair(s["key"])
        for bi in range(nb):
            prog = bi / max(1, nb - 1)
            ctx = {"bar8": bar8, "bi": bi, "nb": nb, "prog": prog, "opts": s["opts"],
                   "tonic": T, "dom": D}
            base_tick = (s["start"] - 1 + bi) * bar_ticks
            for pos, inst, ln, acc, pitch in _pat_events(s["pattern"], ctx):
                gp = (bi + pos / bar8) / nb                  # progreso dentro de la sección
                v = s["v0"] + (s["v1"] - s["v0"]) * min(1.0, gp)
                v = v * acc
                if humanize:
                    v += rnd.uniform(-3, 3)
                v = int(max(1, min(127, round(v))))
                t = int(round(base_tick + pos * eighth))
                if humanize and pos != 0:
                    t += rnd.randint(-int(eighth * 0.04), int(eighth * 0.04))
                    t = max(base_tick, t)
                d = max(1, int(ln * eighth) - 4)
                out[inst].append((t, d, pitch, v))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  MIDI: lectura de la obra, energía, escritura
# ══════════════════════════════════════════════════════════════════════════════

def _dominant_meter(tsigs, end):
    if not tsigs:
        return 4, 4
    ts = sorted(tsigs)
    span = {}
    for k, (t, n, d) in enumerate(ts):
        te = ts[k + 1][0] if k + 1 < len(ts) else max(end, t)
        span[(n, d)] = span.get((n, d), 0) + max(0, te - t)
    best = max(span.values())
    for _t, n, d in ts:
        if span[(n, d)] == best:
            return n, d


def read_work(path):
    """→ dict(tpb, meter, notes[(ini, fin, alt, vel, ch)], end, tsigs_distinct)."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    notes, tsigs, end = [], [], 0
    for tr in mid.tracks:
        t = 0
        active = {}
        for m in tr:
            t += m.time
            if m.type == "time_signature":
                tsigs.append((t, m.numerator, m.denominator))
            elif m.type == "note_on" and m.velocity > 0:
                active.setdefault((m.channel, m.note), []).append((t, m.velocity))
            elif m.type in ("note_off", "note_on"):
                st = active.get((m.channel, m.note))
                if st:
                    s, v = st.pop(0)
                    notes.append((s, max(t, s + 1), m.note, v, m.channel))
        end = max(end, t)
    end = max([end] + [n[1] for n in notes])
    num, den = _dominant_meter(tsigs, end)
    return {"tpb": tpb, "meter": (num, den), "notes": notes, "end": end,
            "n_meters": len({(n, d) for _t, n, d in tsigs}), "mid": mid}


def energy_curve(work):
    """Energía normalizada 0-1 por compás (ataques, velocidad media y voces simultáneas)."""
    tpb = work["tpb"]
    num, den = work["meter"]
    bar_ticks = int(tpb * 4 * num / den)
    n_bars = max(1, int(math.ceil(work["end"] / bar_ticks)))
    onsets = [0] * n_bars
    vel = [[] for _ in range(n_bars)]
    poly = [0.0] * n_bars
    for s, e, p, v, ch in work["notes"]:
        if ch == KIT_CH:
            continue
        b = min(n_bars - 1, s // bar_ticks)
        onsets[b] += 1
        vel[b].append(v)
        for bb in range(b, min(n_bars, e // bar_ticks + 1)):
            lo, hi = max(s, bb * bar_ticks), min(e, (bb + 1) * bar_ticks)
            poly[bb] += max(0, hi - lo) / bar_ticks
    mv = [sum(x) / len(x) if x else 0.0 for x in vel]

    def norm(a):
        s = sorted(a)
        hi = s[min(len(s) - 1, int(0.95 * len(s)))] or max(s) or 1.0
        return [min(1.0, x / hi) if hi else 0.0 for x in a]
    o, m, p = norm(onsets), norm(mv), norm(poly)
    e = [0.5 * a + 0.2 * b + 0.3 * c for a, b, c in zip(o, m, p)]
    sm = []
    for i in range(n_bars):                       # media móvil de 3 compases
        w = e[max(0, i - 1):i + 2]
        sm.append(sum(w) / len(w))
    return sm, bar_ticks, n_bars


_LEVELS = [(0.10, "pp", "tacet"), (0.30, "pp", "pulse"), (0.48, "p", "heartbeat"),
           (0.64, "mf", "marcato"), (0.82, "f", "ostinato"), (9.0, "ff", "hits")]


def _level(x):
    for i, (th, _d, _p) in enumerate(_LEVELS):
        if x < th:
            return i
    return len(_LEVELS) - 1


def auto_spec(curve, min_bars=2):
    """Curva de energía → especificación de secciones (texto). Antes de cada salto de
    ≥2 niveles se inserta un redoble que sube; el primer compás de nivel máximo lleva
    tam-tam (patrón hits)."""
    lv = [_level(x) for x in curve]
    # mediana de 3 para quitar picos de 1 compás
    sm = [sorted(lv[max(0, i - 1):i + 2])[len(lv[max(0, i - 1):i + 2]) // 2] for i in range(len(lv))]
    runs = []
    for i, l in enumerate(sm):
        if runs and runs[-1][2] == l:
            runs[-1][1] = i + 1
        else:
            runs.append([i + 1, i + 1, l])
    merged = []
    for r in runs:                                 # secciones cortas se funden con la anterior
        if merged and (r[1] - r[0] + 1) < min_bars:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    secs = []
    for k, (a, b, l) in enumerate(merged):
        th, dyn, pat = _LEVELS[l]
        prev = merged[k - 1][2] if k else 0
        if k and l - prev >= 2 and (b - a + 1) >= 3 and l >= 3:
            secs.append(f"{a}-{a} {_LEVELS[prev][1]}>{dyn} roll")
            a += 1
        if l == len(_LEVELS) - 1:
            secs.append(f"{a}-{b} fff hits every=2" if (b - a + 1) >= 1 else f"{a} ff hits")
        elif l >= 3 and k + 1 < len(merged) and merged[k + 1][2] > l:
            secs.append(f"{a}-{b} {dyn}>{_LEVELS[min(l + 1, 5)][1]} {pat}")
        else:
            secs.append(f"{a}-{b} {dyn} {pat}")
    return "; ".join(s for s in secs if not s.endswith(" tacet"))


def _detect_key(work):
    hist = [0.0] * 12
    for s, e, p, v, ch in work["notes"]:
        if ch != KIT_CH:
            hist[p % 12] += e - s
    if not any(hist):
        return (2, True)
    maj = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    mn = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    def corr(x, y):
        mx, my = sum(x) / 12, sum(y) / 12
        sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((a - my) ** 2 for a in y))
        return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0
    best = max(((corr(hist, [prof[(i - t) % 12] for i in range(12)]), t, minor)
                for minor, prof in ((False, maj), (True, mn)) for t in range(12)))
    return best[1], best[2]


def _key_str(k):
    return ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][k[0]] + ("m" if k[1] else "")


def build_tracks(events, tpb, split=False, kit_map=None):
    """events de render() → pistas mido (timbales ch16 + kit ch10, o una por instrumento)."""
    km = dict(KIT)
    km.update(kit_map or {})
    names = {"timpani": "Timbal", "bombo": "Bombo", "caja": "Caja", "platos": "Platos",
             "plato_susp": "Plato susp.", "tamtam": "Tam-tam"}

    def mk(name, items, ch, prog=None):
        evs = []
        for t, d, pitch, v in items:
            evs.append((t, 1, "on", pitch, v))
            evs.append((t + d, 0, "off", pitch, 0))
        evs.sort(key=lambda x: (x[0], x[1]))
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if prog is not None:
            tr.append(mido.Message("program_change", channel=ch, program=prog, time=0))
        last = 0
        for t, _o, kind, pitch, v in evs:
            tr.append(mido.Message("note_on" if kind == "on" else "note_off", channel=ch,
                                   note=pitch, velocity=v, time=t - last))
            last = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        return tr
    tracks = []
    if events["timpani"]:
        tracks.append(mk("Timbal", events["timpani"], TIMP_CH, TIMP_PROG))
    kit_items = []
    for inst in ("bombo", "caja", "platos", "plato_susp", "tamtam"):
        items = [(t, d, km[inst], v) for t, d, _p, v in events[inst]]
        if split and items:
            tracks.append(mk(names[inst], items, KIT_CH))
        kit_items += items
    if kit_items and not split:
        tracks.append(mk("Percusión", kit_items, KIT_CH))
    return tracks


def _fix_overlaps(events):
    """Un timbal no puede sonar dos veces a la vez: recorta la nota anterior de igual altura."""
    for inst, lst in events.items():
        lst.sort(key=lambda x: (x[0], x[2] or 0))
        last = {}
        for i, (t, d, p, v) in enumerate(lst):
            k = p
            if k in last:
                j = last[k]
                tj, dj, pj, vj = lst[j]
                if tj + dj > t:
                    lst[j] = (tj, max(1, t - tj - 1), pj, vj)
            last[k] = i


def parse_map(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SpecError(f"--map espera NOMBRE=NOTA: {it!r}")
        k, v = it.split("=", 1)
        k = INSTR_ALIAS.get(k.strip().lower(), k.strip().lower())
        if k not in KIT:
            raise SpecError(f"instrumento desconocido en --map: {k!r} (usa {', '.join(KIT)})")
        try:
            n = int(v)
        except ValueError:
            raise SpecError(f"nota GM inválida: {v!r}")
        if not 0 <= n <= 127:
            raise SpecError(f"nota GM fuera de rango: {n}")
        out[k] = n
    return out


def _spec_text(a):
    if getattr(a, "spec_file", None):
        with open(a.spec_file, encoding="utf-8") as fh:
            return fh.read()
    return a.spec or ""


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_list(_a):
    print(f"{_c('CYA')}PATRONES{_c('R')}")
    for k, v in PATTERNS.items():
        print(f"  {k:<10} {v}")
    print(f"\n{_c('CYA')}DINÁMICAS{_c('R')}  " + "  ".join(f"{k}={v}" for k, v in DYN.items()) + "   (rampa: pp>ff)")
    print(f"{_c('CYA')}INSTRUMENTOS{_c('R')}  timbal (canal 16, afinado) · " + " · ".join(f"{k}={v}" for k, v in KIT.items()))
    return 0


def _bar8(num, den):
    return max(2, int(round(8.0 * num / den)))


def cmd_generate(a):
    try:
        mm = re.fullmatch(r"(\d+)/(\d+)", a.meter)
        if not mm:
            raise SpecError(f"compás inválido: {a.meter!r}")
        num, den = int(mm.group(1)), int(mm.group(2))
        secs = parse_spec(_spec_text(a), a.key)
        km = parse_map(a.map)
    except (SpecError, OSError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    if not secs:
        print("[ERROR] especificación vacía (usa --spec o --spec-file)", file=sys.stderr)
        return 1
    tpb = 480
    bar_ticks = int(tpb * 4 * num / den)
    ev = render(secs, _bar8(num, den), tpb, bar_ticks, not a.no_humanize, a.seed)
    _fix_overlaps(ev)
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    cond = mido.MidiTrack()
    cond.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(a.bpm), time=0))
    cond.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=0))
    last_bar = secs[-1]["end"]
    cond.append(mido.MetaMessage("end_of_track", time=last_bar * bar_ticks))
    mid.tracks.append(cond)
    mid.tracks += build_tracks(ev, tpb, a.split, km)
    mid.save(_ensure_parent(a.out))
    n = sum(len(v) for v in ev.values())
    print(f"{n} golpes en {last_bar} compases ({num}/{den}, {a.bpm:g} BPM) → {a.out}")
    for s in secs:
        T, D = timpani_pair(s["key"])
        print(f"  c.{s['start']}-{s['end']}  {s['pattern']:<9} {s['v0']}→{s['v1']}  timbales {T}/{D}")
    return 0 if n else 2


def cmd_analyze(a):
    try:
        work = read_work(a.midi)
    except (OSError, ValueError, EOFError, IndexError, KeyError) as e:
        print(f"[ERROR] no se puede leer {a.midi}: {e}", file=sys.stderr)
        return 1
    curve, bar_ticks, n = energy_curve(work)
    num, den = work["meter"]
    key = parse_key(a.key) if a.key else _detect_key(work)
    print(f"{n} compases de {num}/{den} · tonalidad {_key_str(key)}")
    print("energía  " + "".join(SPARK[min(7, int(x * 8))] for x in curve))
    print("nivel    " + "".join(str(_level(x)) for x in curve))
    spec = auto_spec(curve)
    print(f"\nEspecificación propuesta (--key {_key_str(key)}):\n  {spec}" if spec
          else "\nLa obra queda por debajo del umbral: no se propone percusión.")
    if work["n_meters"] > 1:
        print(f"\n[AVISO] la obra cambia de compás {work['n_meters']} veces; se usa el dominante {num}/{den}",
              file=sys.stderr)
    return 0 if spec else 2


def cmd_add(a):
    try:
        work = read_work(a.midi)
        km = parse_map(a.map)
    except (SpecError, OSError, ValueError, EOFError, IndexError, KeyError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    curve, bar_ticks, n = energy_curve(work)
    key = parse_key(a.key) if a.key else _detect_key(work)
    if key is None:
        print(f"[ERROR] tonalidad no reconocida: {a.key}", file=sys.stderr)
        return 1
    text = _spec_text(a)
    if a.auto:
        if text.strip():
            print("[ERROR] usa --auto o --spec/--spec-file, no ambos", file=sys.stderr)
            return 1
        text = auto_spec(curve)
        print(f"Especificación derivada: {text}")
    try:
        secs = parse_spec(text, _key_str(key))
    except SpecError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    if not secs:
        print("[AVISO] nada que escribir (especificación vacía)", file=sys.stderr)
        return 2
    if work["n_meters"] > 1:
        print(f"[AVISO] cambios de compás: se usa el dominante {work['meter'][0]}/{work['meter'][1]}",
              file=sys.stderr)
    if secs[-1]["end"] > n + 1:
        print(f"[AVISO] la especificación llega al compás {secs[-1]['end']} y la obra tiene {n}",
              file=sys.stderr)
    num, den = work["meter"]
    ev = render(secs, _bar8(num, den), work["tpb"], bar_ticks, not a.no_humanize, a.seed)
    _fix_overlaps(ev)
    mid = work["mid"]
    if a.replace:
        keep = []
        for tr in mid.tracks:
            chans = {m.channel for m in tr if hasattr(m, "channel")}
            if chans == {KIT_CH}:
                continue
            keep.append(tr)
        mid.tracks[:] = keep
    mid.tracks += build_tracks(ev, work["tpb"], a.split, km)
    out = a.out or os.path.splitext(a.midi)[0] + "_perc.mid"
    mid.save(_ensure_parent(out))
    print(f"{sum(len(v) for v in ev.values())} golpes añadidos → {out}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_work(path, bars=24, num=4, den=4):
    """Obra con 3 tramos de energía creciente: 8 compases ralos, 8 medios, 8 densos."""
    mid = mido.MidiFile(type=1, ticks_per_beat=480)
    c = mido.MidiTrack()
    c.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(80)))
    c.append(mido.MetaMessage("time_signature", numerator=num, denominator=den))
    mid.tracks.append(c)
    bar = int(480 * 4 * num / den)
    evs = []
    for b in range(bars):
        level = 0 if b < 8 else 1 if b < 16 else 2
        step = (bar, bar // 2, bar // 8)[level]
        vel = (40, 70, 105)[level]
        for k in range(bar // step):
            for p in ([50] if level == 0 else [50, 57] if level == 1 else [50, 57, 62, 66]):
                s = b * bar + k * step
                evs.append((s, "on", p, vel)); evs.append((s + step - 5, "off", p, 0))
    evs.sort(key=lambda x: (x[0], x[1] == "on"))
    t = mido.MidiTrack(); last = 0
    t.append(mido.Message("program_change", program=0, channel=0))
    for tick, k, p, v in evs:
        t.append(mido.Message("note_on" if k == "on" else "note_off", note=p, velocity=v, channel=0, time=tick - last))
        last = tick
    mid.tracks.append(t)
    mid.save(path)
    return path


def cmd_selftest(_a):
    import tempfile
    ok = True

    def chk(name, cond):
        nonlocal ok
        print(f"  {'✓' if cond else '✗'} {name}")
        ok = ok and bool(cond)

    chk("parse_key: Dm, Re menor, F#, Bb", [parse_key(x) for x in ("Dm", "Re menor", "F#", "Bb")] ==
        [(2, True), (2, True), (6, False), (10, False)])
    T, D = timpani_pair((2, True))
    chk("timbales Re menor: D3=50 y A2=45", (T, D) == (50, 45))
    ok_range = all(TIMP_LO <= x <= TIMP_HI for pc in range(12) for x in timpani_pair((pc, False)))
    chk("timbales dentro de F2-C4 en las 12 tonalidades", ok_range)
    bad = 0
    for s in ("1-8 xx pulse", "1-8 pp nada", "0-3 pp pulse", "5-2 pp pulse", "1-4 pp pulse; 3-6 mf marcato", "1-4 pp"):
        try:
            parse_spec(s); bad += 1
        except SpecError:
            pass
    chk("rechaza 6 especificaciones inválidas", bad == 0)
    sp = parse_spec("1-4 pp>ff roll; 5-8 mf marcato key=A")
    chk("rampa y key por sección", sp[0]["v0"] == DYN["pp"] and sp[0]["v1"] == DYN["ff"] and sp[1]["key"] == (9, False))

    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "x", "p.mid")

        class A: pass
        a = A(); a.spec = "1-4 pp heartbeat; 5-8 mf marcato; 9-16 mf>ff surge cycle=4; 17-18 fff hits"
        a.spec_file = None; a.out = out; a.key = "Dm"; a.meter = "4/4"; a.bpm = 66; a.split = False
        a.map = []; a.seed = 7; a.no_humanize = True
        chk("generate rc 0", cmd_generate(a) == 0)
        m = mido.MidiFile(out)
        names = [t.name for t in m.tracks if t.name]
        chk("pistas Timbal y Percusión", "Timbal" in names and "Percusión" in names)
        notes = []
        for tr in m.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    notes.append((t, msg.channel, msg.note, msg.velocity))
        bar = 480 * 4
        mar = [n for n in notes if bar * 4 <= n[0] < bar * 5]
        chk("marcato: bombo en 1.er y 3.er tiempo del compás 5",
            sorted(n[0] - bar * 4 for n in mar if n[1] == 9 and n[2] == 36) == [0, 960])
        chk("marcato: timbal en corcheas 0,2,3,4",
            sorted(n[0] - bar * 4 for n in mar if n[1] == 15) == [0, 480, 720, 960])
        chk("timbales solo tónica/dominante de Re m (50/45)", {n[2] for n in notes if n[1] == 15} <= {50, 45})
        chk("sin notas fuera de los compases pedidos", max(n[0] for n in notes) < 18 * bar)
        sur = [n for n in notes if 8 * bar <= n[0] < 16 * bar]
        peaks = sorted({(n[0] - 8 * bar) // bar for n in sur if n[1] == 9 and n[2] == 49})
        chk("surge: pico de platos cada 4 compases (compases 4 y 8 de la sección)", peaks == [3, 7])
        chk("surge: el 1.er compás de cada ciclo NO queda en silencio",
            all(any((n[0] - 8 * bar) // bar == c for n in sur) for c in range(8)))
        v_early = max(n[3] for n in sur if (n[0] - 8 * bar) // bar == 0 and n[1] == 15)
        v_peak = max(n[3] for n in sur if (n[0] - 8 * bar) // bar == 3 and n[1] == 15)
        chk("surge: el pico es más fuerte que el inicio del ciclo", v_peak > v_early)
        hits = [n for n in notes if n[0] >= 16 * bar and n[1] == 9 and n[2] == 54]
        chk("hits: tam-tam al empezar la sección fff", len(hits) == 1 and hits[0][3] >= 110)
        # compás ternario
        a.meter = "3/4"; a.spec = "1-4 mf marcato"; a.out = os.path.join(d, "v.mid")
        cmd_generate(a)
        m3 = mido.MidiFile(a.out)
        ts = [x for t in m3.tracks for x in t if x.type == "time_signature"]
        chk("3/4: señal de compás escrita", ts and (ts[0].numerator, ts[0].denominator) == (3, 4))
        t3 = []
        for tr in m3.tracks:
            t = 0
            for msg in tr:
                t += msg.time
                if msg.type == "note_on" and msg.velocity > 0 and msg.channel == 9:
                    t3.append(t)
        chk("3/4: ningún golpe cruza la barra (compás de 1440 ticks)", all((x % 1440) < 1440 for x in t3) and max(t3) < 4 * 1440)
        # --split y --map
        a.meter = "4/4"; a.spec = "1-2 f hits"; a.out = os.path.join(d, "s.mid"); a.split = True; a.map = ["bombo=35", "tam-tam=52"]
        cmd_generate(a)
        ms = mido.MidiFile(a.out)
        chk("--split: una pista por instrumento", {"Bombo", "Platos", "Tam-tam", "Timbal"} <= {t.name for t in ms.tracks})
        notes_s = {msg.note for t in ms.tracks for msg in t if msg.type == "note_on" and msg.channel == 9}
        chk("--map: bombo=35 y tam-tam=52", 35 in notes_s and 52 in notes_s and 36 not in notes_s)

        # analyze + add --auto
        src = _synthetic_work(os.path.join(d, "obra.mid"))
        w = read_work(src)
        curve, _bt, n = energy_curve(w)
        lv = [_level(x) for x in curve]
        chk("energía: tercer tramo > primero", sum(lv[16:]) / 8 > sum(lv[:8]) / 8 + 1.5)
        spec = auto_spec(curve)
        secs = parse_spec(spec)
        chk("auto_spec produce secciones válidas y ordenadas", len(secs) >= 2 and secs[0]["start"] == 1 or len(secs) >= 2)
        b = A(); b.midi = src; b.spec = None; b.spec_file = None; b.auto = True; b.key = None
        b.out = os.path.join(d, "o.mid"); b.split = False; b.map = []; b.seed = 7; b.no_humanize = False; b.replace = False
        chk("add --auto rc 0", cmd_add(b) == 0)
        o = mido.MidiFile(b.out)
        orig_notes = sorted((s, e, p) for s, e, p, v, ch in read_work(src)["notes"] if ch == 0)
        new_notes = sorted((s, e, p) for s, e, p, v, ch in read_work(b.out)["notes"] if ch == 0)
        chk("add conserva intactas las notas de la obra", orig_notes == new_notes)
        perc = [n2 for n2 in read_work(b.out)["notes"] if n2[4] in (9, 15)]
        chk("add: percusión añadida y más densa al final que al principio",
            perc and sum(1 for p2 in perc if p2[0] > 16 * 1920) > sum(1 for p2 in perc if p2[0] < 8 * 1920) - 1)
        # compás ternario en la obra
        src3 = _synthetic_work(os.path.join(d, "o3.mid"), bars=12, num=3, den=4)
        b.midi = src3; b.out = os.path.join(d, "o3p.mid")
        chk("add sobre 3/4 rc 0", cmd_add(b) in (0, 2))
        # --replace
        b.midi = b.out = os.path.join(d, "o.mid"); b.out = os.path.join(d, "o_rep.mid"); b.auto = False
        b.spec = "1-4 p pulse"; b.replace = True
        cmd_add(b)
        chk("--replace quita el kit previo", sum(1 for x in read_work(b.out)["notes"] if x[4] == 9) <= 4)
        # sin solapes de timbal
        ev = render(parse_spec("1-4 mf ostinato div=4"), 8, 480, 1920, False)
        _fix_overlaps(ev)
        ts_ = sorted(ev["timpani"])
        chk("timbal: sin notas solapadas de igual altura",
            all(not (a1[0] + a1[1] > b1[0] and a1[2] == b1[2]) for a1, b1 in zip(ts_, ts_[1:])))
    print("selftest:", "OK" if ok else "FALLOS")
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(prog="orchestral_percussion.py", description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Patrones, dinámicas e instrumentos")
    sp = sub.add_parser("generate", help="MIDI de percusión desde una especificación")
    sp.add_argument("--spec", "-s")
    sp.add_argument("--spec-file")
    sp.add_argument("--out", "-o", default="percusion.mid")
    sp.add_argument("--key", default="Dm")
    sp.add_argument("--meter", default="4/4")
    sp.add_argument("--bpm", type=float, default=72)
    sp.add_argument("--split", action="store_true")
    sp.add_argument("--map", nargs="*", default=[])
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--no-humanize", action="store_true")
    sp = sub.add_parser("add", help="Añade percusión a un MIDI")
    sp.add_argument("midi")
    sp.add_argument("--spec", "-s")
    sp.add_argument("--spec-file")
    sp.add_argument("--auto", action="store_true")
    sp.add_argument("--key")
    sp.add_argument("--out", "-o")
    sp.add_argument("--split", action="store_true")
    sp.add_argument("--map", nargs="*", default=[])
    sp.add_argument("--seed", type=int, default=7)
    sp.add_argument("--no-humanize", action="store_true")
    sp.add_argument("--replace", action="store_true")
    sp = sub.add_parser("analyze", help="Energía por compás y spec propuesta")
    sp.add_argument("midi")
    sp.add_argument("--key")
    sp.add_argument("--no-color", action="store_true")
    sub.add_parser("selftest", help="Prueba con MIDIs sintéticos")
    return p


def main(argv=None):
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if getattr(a, "no_color", False):
        _USE_COLOR = False
    return {"list": cmd_list, "generate": cmd_generate, "add": cmd_add, "analyze": cmd_analyze,
            "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
