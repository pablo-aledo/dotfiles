#!/usr/bin/env python3
"""
orchestra_anatomy_v2.py — Orchestra Anatomy: fotografía de una obra orquestal
(quién toca qué en cada compás, con rol, registro y densidad)
(versión 2.0: añade representación visual en HTML, --html).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Para aprender a orquestar hay que mirar cómo lo hacen otros. Esta herramienta
toma un MIDI orquestal (una pista = un instrumento o sección) y dibuja su
anatomía: una fila por instrumento, una columna por compás, y en cada celda
el ROL que cumple y el REGISTRO en que toca respecto a la tesitura de su
instrumento. Debajo, la densidad compás a compás y observaciones neutras
(compases sin bajo, relevos de melodía, tesituras forzadas).

Describe, no juzga ni escribe música por ti. Para criticar TU orquestación
con consejos, ese es otro trabajo (orchestra_critic).

AUTOCONTENIDA: no importa ni llama a ninguna otra herramienta de la suite.
Lectura MIDI propia (solo biblioteca estándar). Dos piezas se COPIAN y adaptan:
  · lógica de roles de role_detector.py (melody/countermelody/bass/harmony/
    percussion), aplicada primero a la obra entera y luego compás a compás,
    con el rol global como a priori. Admite el mismo fichero
    <midi>.roles.json ({"roles": {"<índice de pista>": "<rol>"}}).
  · el catálogo de rangos por instrumento es una tabla APROXIMADA propia
    (RANGES). Ajústala a la de playability_auditor si quieres que coincidan.

CELDA = rol + registro
  rol:      M melodía · C contramelodía · B bajo · A armonía · P percusión
  registro: g grave (bajo su zona cómoda) · m medio · a agudo (sobre su zona
            cómoda) · ! fuera de la tesitura del instrumento · ? instrumento
            sin identificar
  ·  = la pista calla en ese compás

LÍMITES
  Es una estimación sobre MIDI (registro, densidad, velocity). No dice cómo
  suenan tus patches. Los roles son heurísticos: con melodía repartida entre
  instrumentos o texturas muy mezcladas pueden equivocarse; fíjalos con
  --roles-file si lo ves claro. Divisiones SMPTE no soportadas.

COMO MÓDULO:
  from orchestra_anatomy_v2 import analyze, render_text, render_html
  res = analyze("obra.mid", bars=(9, 16))
  open("anatomia.html", "w", encoding="utf-8").write(render_html(res))

DEPENDENCIAS: ninguna (biblioteca estándar)

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  midi              MIDI orquestal de entrada
  --bars A-B        Solo esos compases (1-based, inclusivo; B opcional)
  --map K=INSTR     Fija el instrumento de una pista; K = índice de pista o
                    nombre exacto. Repetible. Ej.: --map 3=oboe
  --roles-file F    Roles fijados a mano (default: <midi>.roles.json si existe)
  --html FILE       Informe visual autocontenido (HTML, sin dependencias externas;
                    claro/oscuro automático, se abre en cualquier navegador)
  --json FILE       Sidecar JSON con la anatomía completa ('-' = a stdout, sin texto)
  --no-color        Desactivar colores ANSI

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[*]
midi          file  io=in ext=mid required
--bars        str
--map         list  of=str
--roles-file  str
--html        file  io=out ext=html
--json        file  io=out ext=json
--no-color    bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python orchestra_anatomy_v2.py obra.mid
  python orchestra_anatomy_v2.py obra.mid --bars 9-24
  python orchestra_anatomy_v2.py obra.mid --map 4=oboe --map "Pista X=fagot"
  python orchestra_anatomy_v2.py obra.mid --html anatomia.html
  python orchestra_anatomy_v2.py obra.mid --json anatomia.json
"""

import os
import re
import sys
import json
import struct
import argparse
from collections import Counter

VERSION = "2.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
_SPARK = "▁▂▃▄▅▆▇█"
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DRUM_CH = 9


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    if isinstance(path, (str, bytes, os.PathLike)):
        d = os.path.dirname(os.path.abspath(os.fsdecode(path)))
        if d:
            os.makedirs(d, exist_ok=True)
    return path


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def pitch_name(p):
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


def sparkline(v):
    v = list(v)
    if not v:
        return ""
    lo, hi = min(v), max(v)
    if hi <= lo:
        return _SPARK[3] * len(v)
    return "".join(_SPARK[min(7, int(round((x - lo) / (hi - lo) * 7)))] for x in v)


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO DE INSTRUMENTOS (aproximado, propio)
# ══════════════════════════════════════════════════════════════════════════════

# nombre: (min, max, cómodo_min, cómodo_max) en notas MIDI (C4 = 60)
RANGES = {
    "flauta": (60, 96, 67, 88), "piccolo": (74, 108, 84, 100),
    "oboe": (58, 91, 64, 84), "corno inglés": (52, 81, 57, 76),
    "clarinete": (50, 94, 55, 86), "fagot": (34, 75, 38, 67),
    "saxofón": (49, 87, 53, 80),
    "trompa": (34, 77, 41, 72), "trompeta": (55, 82, 60, 79),
    "trombón": (40, 72, 43, 67), "tuba": (28, 58, 31, 53),
    "violín": (55, 103, 60, 91), "viola": (48, 91, 53, 81),
    "violonchelo": (36, 76, 40, 69), "contrabajo": (28, 67, 31, 60),
    "arpa": (24, 103, 31, 96), "piano": (21, 108, 28, 100),
    "coro": (48, 84, 52, 79), "timbales": (40, 55, 40, 55),
}

# (instrumento, alias). Orden = prioridad. Alias de varias palabras: frase;
# de una: palabra exacta, o prefijo si tiene ≥4 letras ("violas", "flutes").
ALIASES = [
    ("corno inglés", ["english horn", "cor anglais", "corno ingles", "corno inglés"]),
    ("piccolo", ["piccolo", "picc"]),
    ("flauta", ["flute", "flauta", "fl"]),
    ("oboe", ["oboe", "ob"]),
    ("clarinete", ["clarinet", "clarinete", "cl"]),
    ("fagot", ["bassoon", "fagot", "fg"]),
    ("saxofón", ["sax", "saxophone", "saxofon", "saxofón"]),
    ("trompa", ["french horn", "horn", "trompa", "hn"]),
    ("trompeta", ["trumpet", "trompeta", "tpt"]),
    ("trombón", ["trombone", "trombon", "trombón", "tbn"]),
    ("tuba", ["tuba"]),
    ("timbales", ["timpani", "timbal", "timbales"]),
    ("arpa", ["harp", "arpa"]),
    ("piano", ["piano"]),
    ("coro", ["choir", "coro", "voices", "voces"]),
    ("violín", ["violin", "violín", "violines", "vln"]),
    ("viola", ["viola", "vla"]),
    ("violonchelo", ["cello", "violonchelo", "vc"]),
    ("contrabajo", ["double bass", "contrabass", "contrabajo", "cb", "bass", "bajo"]),
]

GM_PROGRAMS = {
    0: "piano", 46: "arpa", 47: "timbales", 40: "violín", 41: "viola",
    42: "violonchelo", 43: "contrabajo", 44: "violín", 45: "violín",
    48: "violín", 49: "violín", 50: "violín", 51: "violín",
    52: "coro", 53: "coro", 54: "coro", 56: "trompeta", 57: "trombón",
    58: "tuba", 59: "trompeta", 60: "trompa", 61: "trompa", 62: "trompa",
    63: "trompa", 64: "saxofón", 65: "saxofón", 66: "saxofón", 67: "saxofón",
    68: "oboe", 69: "corno inglés", 70: "fagot", 71: "clarinete",
    72: "piccolo", 73: "flauta", 74: "flauta",
}
GM_PROGRAMS.update({p: "contrabajo" for p in range(32, 40)})

FAMILIES = {
    "flauta": "maderas", "piccolo": "maderas", "oboe": "maderas",
    "corno inglés": "maderas", "clarinete": "maderas", "fagot": "maderas",
    "saxofón": "maderas", "trompa": "metales", "trompeta": "metales",
    "trombón": "metales", "tuba": "metales", "timbales": "percusión",
    "percusión": "percusión", "arpa": "teclas/arpa", "piano": "teclas/arpa",
    "coro": "coro", "violín": "cuerdas", "viola": "cuerdas",
    "violonchelo": "cuerdas", "contrabajo": "cuerdas",
}
FAMILY_ORDER = ["maderas", "metales", "percusión", "teclas/arpa", "coro",
                "cuerdas", "otros"]


def identify(track, name_map):
    """→ instrumento (str) o None. Orden: --map, nombre, canal 10, programa GM."""
    for key in (str(track["index"]), (track["name"] or "").lower()):
        for k, v in name_map.items():
            if str(k).lower() == key:
                return v
    if track["channels"] and set(track["channels"]) == {DRUM_CH}:
        return "percusión"
    norm = " ".join(re.findall(r"[a-záéíóúñü]+", (track["name"] or "").lower()))
    toks = norm.split()
    for inst, aliases in ALIASES:
        for al in aliases:
            if " " in al:
                if f" {al} " in f" {norm} ":
                    return inst
            elif al in toks or (len(al) >= 4 and any(t.startswith(al) for t in toks)):
                return inst
    return GM_PROGRAMS.get(track["program"])


def zone(pitch, rng):
    if rng is None:
        return "?"
    lo, hi, clo, chi = rng
    if pitch < lo or pitch > hi:
        return "fuera"
    if pitch < clo:
        return "grave"
    if pitch > chi:
        return "agudo"
    return "medio"


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA MIDI (biblioteca estándar)
# ══════════════════════════════════════════════════════════════════════════════

def _vlq(data, i):
    v = 0
    while True:
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def parse_midi(data):
    """→ {"tpb", "timesigs": [(tick, num, den)], "tracks": [...]}.
    El índice de pista es la posición en el fichero (0-based), igual que en
    role_detector. Un MIDI truncado se lee hasta donde llegue."""
    if data[:4] != b"MThd":
        raise ValueError("no es un fichero MIDI (falta MThd)")
    _, ntr, div = struct.unpack(">HHH", data[8:14])
    if div & 0x8000:
        raise ValueError("división SMPTE no soportada")
    pos = 8 + struct.unpack(">I", data[4:8])[0]
    tracks, timesigs = [], []
    for idx in range(ntr):
        if data[pos:pos + 4] != b"MTrk":
            break
        ln = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        i, end = pos + 8, min(len(data), pos + 8 + ln)
        pos = pos + 8 + ln
        t, status = 0, 0
        name, prog, chans = "", None, set()
        open_, notes = {}, []
        try:
            while i < end:
                d, i = _vlq(data, i)
                t += d
                b = data[i]
                if b == 0xFF:
                    mt = data[i + 1]
                    ml, j = _vlq(data, i + 2)
                    body = data[j:j + ml]
                    i = j + ml
                    if mt == 0x03 and not name:
                        name = body.decode("utf-8", "replace").strip()
                    elif mt == 0x58 and len(body) >= 2:
                        timesigs.append((t, body[0], 2 ** body[1]))
                    continue
                if b in (0xF0, 0xF7):
                    ml, j = _vlq(data, i + 1)
                    i = j + ml
                    continue
                if b & 0x80:
                    status = b
                    i += 1
                kind, ch = status & 0xF0, status & 0x0F
                if kind in (0xC0, 0xD0):
                    p1 = data[i]
                    i += 1
                    if kind == 0xC0 and prog is None:
                        prog = p1
                    continue
                p1, p2 = data[i], data[i + 1]
                i += 2
                if kind == 0x90 and p2 > 0:
                    open_.setdefault((ch, p1), []).append((t, p2))
                    chans.add(ch)
                elif kind == 0x80 or (kind == 0x90 and p2 == 0):
                    q = open_.get((ch, p1))
                    if q:
                        s, v = q.pop(0)
                        notes.append((s, max(t, s + 1), p1, v))
        except IndexError:
            pass
        for (ch, p), q in open_.items():          # notas sin cerrar
            for s, v in q:
                notes.append((s, max(t, s + 1), p, v))
        notes.sort()
        tracks.append({"index": idx, "name": name, "program": prog,
                       "channels": sorted(chans), "notes": notes})
    return {"tpb": div, "timesigs": sorted(timesigs) or [(0, 4, 4)],
            "tracks": tracks}


def bar_grid(parsed):
    """→ lista de (inicio, fin, ticks_por_pulso, pulsos) por compás."""
    last = max((n[1] for t in parsed["tracks"] for n in t["notes"]), default=0)
    ts, tpb = parsed["timesigs"], parsed["tpb"]
    bars, t, k = [], 0, 0
    while t < last:
        while k + 1 < len(ts) and ts[k + 1][0] <= t:
            k += 1
        _, num, den = ts[k]
        beat = max(1, tpb * 4 // den)
        bars.append((t, t + beat * num, beat, num))
        t += beat * num
    return bars


# ══════════════════════════════════════════════════════════════════════════════
#  ROLES (lógica copiada y adaptada de role_detector.py)
# ══════════════════════════════════════════════════════════════════════════════

ROLES = ("melody", "countermelody", "bass", "harmony", "percussion")
ROLE_ES = {"melody": "melodía", "countermelody": "contramelodía", "bass": "bajo",
           "harmony": "armonía", "percussion": "percusión"}
ROLE_CH = {"melody": "M", "countermelody": "C", "bass": "B", "harmony": "A",
           "percussion": "P"}
ROLE_COL = {"M": "CYA", "C": "GRN", "B": "YEL", "A": "R", "P": "G"}
BASS_PROGRAMS = set(range(32, 40))
ENSEMBLE_PROGRAMS = set(range(48, 56)) | set(range(88, 96))
NAME_BASS = ("bajo", "bass", "contrabajo", "double bass", "tuba", "cb")
NAME_HARM = ("pad", "cuerdas", "strings", "ensemble", "sección", "seccion", "section",
             "coro", "choir", "tutti", "arpa", "harp", "piano", "organ", "órgano")
NAME_MEL = ("solista", "solo", "melod", "lead", "voz", "voice")


def concurrency(notes):
    """(fracción de ataques con otra nota ya sonando, máximo simultáneo)."""
    ev = []
    for s, e, p, v in notes:
        ev.append((s, 1))
        ev.append((e, -1))
    ev.sort()
    act = multi = ons = kmax = 0
    for t, k in ev:
        if k == 1:
            act += 1
            ons += 1
            kmax = max(kmax, act)
            if act > 1:
                multi += 1
        else:
            act -= 1
    return (multi / ons if ons else 0.0), kmax


def features(notes, tpb, tpbar, total_bars, cover=None):
    """Rasgos de un conjunto de notas (pista entera o recorte de un compás).
    `cover` fija la cobertura (para compases se usa la de la pista entera)."""
    f = {"n_notes": len(notes)}
    if not notes:
        return f
    pitches = [x[2] for x in notes]
    chord, kmax = concurrency(notes)
    top = {}
    for s, e, p, v in notes:
        top[s] = max(top.get(s, 0), p)
    seq = [top[k] for k in sorted(top)]
    leaps = [abs(b - a) for a, b in zip(seq, seq[1:])] or [0]
    bars = {s // tpbar for s, _, _, _ in notes}
    f.update({
        "mean_pitch": sum(pitches) / len(pitches),
        "pitch_min": min(pitches), "pitch_max": max(pitches),
        "chordiness": chord, "max_poly": kmax, "linearity": 1.0 - chord,
        "smoothness": sum(1 for x in leaps if x <= 5) / len(leaps),
        "coverage": cover if cover is not None else len(bars) / max(1, total_bars),
        "mean_dur_beats": sum((e - s) / tpb for s, e, _, _ in notes) / len(notes),
    })
    return f


def detect_roles(metas, feats, forced=None, prior=None):
    """Misma lógica que role_detector.detect(), sobre rasgos ya calculados.
    metas: [{index,name,program,channels}] · feats: {index: rasgos}
    forced: {index: rol} (confianza 1.0) · prior: {index: rol} (sesgo suave,
    usado compás a compás con el rol global de la pista).
    Diferencia con el original: si ya hay una melodía fijada a mano entre las
    pistas activas, no se elige otra; la mejor candidata pasa a contramelodía/
    armonía por las mismas reglas."""
    forced, prior = forced or {}, prior or {}
    live = [m for m in metas if feats[m["index"]]["n_notes"]]
    out = {}

    def setrole(i, role, conf, why):
        out[i] = {"role": role, "confidence": round(float(conf), 2), "reasons": why}

    for i, r in forced.items():
        if i in feats and feats[i]["n_notes"]:
            setrole(i, r, 1.0, ["rol fijado manualmente"])
    for m in live:
        i = m["index"]
        if i not in out and m["channels"] and set(m["channels"]) == {DRUM_CH}:
            setrole(i, "percussion", 0.98, ["canal 10 (GM percusión)"])

    pool = [m for m in live if m["index"] not in out]
    if pool:
        max_n = max(feats[m["index"]]["n_notes"] for m in pool)
        pm = [feats[m["index"]]["mean_pitch"] for m in pool]
        lo, hi = min(pm), max(pm)

        def rel_reg(i):
            return (feats[i]["mean_pitch"] - lo) / (hi - lo) if hi > lo else 0.5

        bass_score = {}
        for m in pool:
            i, f, low = m["index"], feats[m["index"]], m["name"].lower()
            s, why = 0.0, []
            if f["mean_pitch"] <= 52:
                s += 0.5 * min(1.0, (52 - f["mean_pitch"]) / 16 + 0.4)
                why.append(f"registro grave (pitch medio {f['mean_pitch']:.0f})")
            if m["program"] in BASS_PROGRAMS:
                s += 0.3
                why.append("programa GM de bajo")
            if any(k in low.split() or k == low for k in NAME_BASS) \
                    or "contrabajo" in low or "bass" in low:
                s += 0.2
                why.append("nombre de bajo")
            s += 0.2 * f["linearity"]
            if prior.get(i) == "bass":
                s += 0.1
            elif prior.get(i) in ("melody", "countermelody"):
                s -= 0.1
            bass_score[i] = (s, why)
        for i, (s, why) in bass_score.items():
            if s >= 0.55 and feats[i]["mean_pitch"] <= 55:
                setrole(i, "bass", min(0.95, s), why)

        rest = [m for m in pool if m["index"] not in out]
        has_forced_mel = any(out[j]["role"] == "melody" for j in out)
        mel_score = {}
        for m in rest:
            i, f, low = m["index"], feats[m["index"]], m["name"].lower()
            reg = rel_reg(i)
            act = f["n_notes"] / max_n
            s = (0.25 * reg + 0.25 * min(1.0, act * 2) + 0.20 * f["linearity"]
                 + 0.15 * f["coverage"] + 0.15 * f["smoothness"])
            why = [f"registro relativo {reg:.2f}", f"actividad {act:.2f}",
                   f"linealidad {f['linearity']:.2f}", f"cobertura {f['coverage']:.2f}"]
            if f["n_notes"] < 0.08 * max_n:
                s *= 0.4
                why.append("muy pocas notas (<8 % de la pista más activa)")
            if any(k in low for k in NAME_MEL):
                s += 0.08
                why.append("nombre de solista/melodía")
            if any(k in low for k in NAME_HARM) or m["program"] in ENSEMBLE_PROGRAMS:
                s -= 0.12
                why.append("nombre/programa de sección-armonía")
            if f["chordiness"] > 0.5:
                s -= 0.1
                why.append(f"muy acórdica ({f['chordiness']:.0%})")
            if prior.get(i) == "melody":
                s += 0.08
            mel_score[i] = (s, why)
        if mel_score:
            ranked = sorted(mel_score, key=lambda i: -mel_score[i][0])
            best = ranked[0]
            best_s = mel_score[best][0]
            second_s = mel_score[ranked[1]][0] if len(ranked) > 1 else 0.0
            if not has_forced_mel:
                conf = max(0.2, min(0.95, 0.5 + (best_s - second_s) * 2))
                setrole(best, "melody", conf, mel_score[best][1])
                others = ranked[1:]
            else:
                others = ranked
            for i in others:
                s, why = mel_score[i]
                f = feats[i]
                if s >= 0.70 * best_s and f["linearity"] >= 0.7 \
                        and f["n_notes"] >= 0.2 * max_n:
                    setrole(i, "countermelody", 0.5 + min(0.3, s - 0.5 * best_s), why)
                else:
                    hs = 0.5 + 0.3 * f["chordiness"] + 0.2 * min(1, f["mean_dur_beats"] / 3)
                    setrole(i, "harmony", min(0.9, hs),
                            [f"acórdica {f['chordiness']:.0%}",
                             f"dur. media {f['mean_dur_beats']:.1f} pulsos",
                             f"puntuación de melodía baja ({s:.2f} < {best_s:.2f})"])
    return out


def read_forced(path):
    """Mismo formato que role_detector: {"roles": {"<índice>": "<rol>"}}."""
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return {int(k): v for k, v in d.get("roles", {}).items() if v in ROLES}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS POR COMPÁS
# ══════════════════════════════════════════════════════════════════════════════

def density_label(per_beat):
    return "baja" if per_beat < 1 else ("media" if per_beat < 3 else "alta")


def _clip(notes, b0, b1):
    return [(max(s, b0), min(e, b1), p, v) for s, e, p, v in notes if s < b1 and e > b0]


def parse_bars(txt):
    if not txt:
        return None
    a, _, b = txt.partition("-")
    try:
        lo, hi = int(a), int(b or a)
    except ValueError:
        raise ValueError(f"--bars debe ser A-B (recibido {txt!r})")
    if lo < 1 or hi < lo:
        raise ValueError(f"--bars fuera de rango ({txt!r})")
    return lo, hi


def analyze(source, bars=None, name_map=None, forced=None):
    """API pública. `source`: ruta o bytes. bars=(A,B) 1-based inclusivo.
    name_map: {índice|nombre: instrumento}. forced: {índice: rol}."""
    if isinstance(source, (bytes, bytearray)):
        data, fname = bytes(source), "<bytes>"
    else:
        with open(source, "rb") as fh:
            data = fh.read()
        fname = os.path.basename(str(source))
    parsed = parse_midi(data)
    tpb = parsed["tpb"]
    grid = bar_grid(parsed)
    if not grid:
        raise ValueError("el MIDI no contiene notas")
    total = len(grid)
    tpbar0 = grid[0][1] - grid[0][0]
    name_map, forced = name_map or {}, forced or {}
    metas = parsed["tracks"]
    live_metas = [m for m in metas if m["notes"]]

    # instrumento, familia y rol global (obra entera)
    info = {}
    for m in live_metas:
        inst = identify(m, name_map)
        info[m["index"]] = {"instrument": inst, "family": FAMILIES.get(inst, "otros"),
                            "range": get_range(inst)}
    gfeats = {m["index"]: features(m["notes"], tpb, tpbar0, total) for m in live_metas}
    groles = detect_roles(live_metas, gfeats, forced)
    for i, r in groles.items():
        # instrumento/percusión por rol si no se identificó por nombre
        if r["role"] == "percussion" and not info[i]["instrument"]:
            info[i].update(instrument="percusión", family="percusión", range=None)
    perc = {i: "percussion" for i, r in groles.items() if r["role"] == "percussion"}
    prior = {i: r["role"] for i, r in groles.items()}
    gcover = {i: gfeats[i]["coverage"] for i in gfeats}

    lo_b, hi_b = bars if bars else (1, total)
    if lo_b > total:
        raise ValueError(f"la obra tiene {total} compases; --bars empieza en {lo_b}")
    hi_b = min(hi_b, total)
    timeline = []
    for k in range(lo_b - 1, hi_b):
        b0, b1, beat, beats = grid[k]
        bfeats, lines = {}, {}
        for m in live_metas:
            cl = _clip(m["notes"], b0, b1)
            if cl:
                bfeats[m["index"]] = features(cl, tpb, tpbar0, total, gcover[m["index"]])
                lines[m["index"]] = cl
        bmetas = [m for m in live_metas if m["index"] in bfeats]
        f_bar = {i: r for i, r in {**forced, **perc}.items() if i in bfeats}
        roles = detect_roles(bmetas, bfeats, f_bar, prior)
        row, onsets = {}, 0
        for i, cl in lines.items():
            ons = [n for n in cl if b0 <= n[0] < b1]
            onsets += len(ons)
            role = roles[i]["role"]
            rng = info[i]["range"]
            if role == "percussion":
                z, mean_p, oor = "-", None, 0
                lo_m = hi_m = None
            else:
                mean_p = bfeats[i]["mean_pitch"]
                z = zone(int(round(mean_p)), rng)
                oor = sum(1 for n in cl if rng and not rng[0] <= n[2] <= rng[1])
                lo_m, hi_m = bfeats[i]["pitch_min"], bfeats[i]["pitch_max"]
            vel_src = ons or cl
            row[i] = {
                "role": role, "zone": z,
                "mean_pitch": None if mean_p is None else round(mean_p, 1),
                "low": pitch_name(bfeats[i]["pitch_min"]),
                "high": pitch_name(bfeats[i]["pitch_max"]),
                "low_midi": lo_m, "high_midi": hi_m,
                "onsets": len(ons),
                "density": round(len(ons) / beats, 2),
                "poly": bfeats[i]["max_poly"],
                "velocity": round(sum(n[3] for n in vel_src) / len(vel_src)),
                "out_of_range": oor,
            }
        tot = onsets / beats
        voices = [i for i, r in row.items() if r["role"] != "percussion"]
        timeline.append({
            "bar": k + 1, "voices": len(voices),
            "onsets_per_beat": round(tot, 2), "density": density_label(tot),
            "has_bass": any(r["role"] == "bass" for r in row.values()),
            "has_melody": any(r["role"] == "melody" for r in row.values()),
            "lines": row,
        })

    tracks_out, warnings = [], []
    for m in live_metas:
        i = m["index"]
        rows = [t["lines"][i] for t in timeline if i in t["lines"]]
        gi = groles[i]
        entry = {
            "index": i, "name": m["name"] or f"pista {i}",
            "instrument": info[i]["instrument"], "family": info[i]["family"],
            "range": list(info[i]["range"]) if info[i]["range"] else None,
            "global_role": gi["role"], "global_confidence": gi["confidence"],
            "bars_active": len(rows), "bars_total": len(timeline),
        }
        if rows:
            pm = [r["mean_pitch"] for r in rows if r["mean_pitch"] is not None]
            entry.update({
                "roles": dict(Counter(r["role"] for r in rows)),
                "zones": dict(Counter(r["zone"] for r in rows)),
                "mean_pitch": round(sum(pm) / len(pm), 1) if pm else None,
                "mean_density": round(sum(r["density"] for r in rows) / len(rows), 2),
                "out_of_range_notes": sum(r["out_of_range"] for r in rows),
            })
        tracks_out.append(entry)
        if info[i]["instrument"] is None:
            warnings.append(f"Pista {i} «{m['name'] or 'sin nombre'}»: instrumento no identificado; "
                            f"no se evalúa su registro. Usa --map {i}=oboe (por ejemplo).")
        if gi["confidence"] < 0.45 and gi["role"] != "percussion":
            warnings.append(f"Pista {i} «{m['name'] or 'sin nombre'}»: rol global poco seguro "
                            f"({ROLE_ES[gi['role']]}, conf. {gi['confidence']:.2f}); "
                            "puedes fijarlo con --roles-file.")

    return {
        "tool": "orchestra_anatomy", "version": VERSION, "file": fname,
        "total_bars": total, "bars": [lo_b, hi_b], "ticks_per_beat": tpb,
        "tracks": tracks_out, "timeline": timeline,
        "observations": observations(timeline, tracks_out),
        "warnings": warnings,
        "note": "Estimación sobre MIDI (registro, densidad, velocity); "
                "no refleja cómo suenan tus patches.",
    }


def get_range(inst):
    return RANGES.get(inst)


def _runs(nums):
    if not nums:
        return ""
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}–{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)


def observations(timeline, tracks):
    obs = []
    active = [t for t in timeline if t["voices"]]
    if not active:
        return obs
    nobass = [t["bar"] for t in active if not t["has_bass"]]
    nomel = [t["bar"] for t in active if not t["has_melody"]]
    if nobass:
        obs.append(f"Compases sin bajo: {_runs(nobass)}.")
    if nomel:
        obs.append(f"Compases sin melodía clara: {_runs(nomel)}.")
    names = {t["index"]: t["name"] or f"pista {t['index']}" for t in tracks}
    holders, last = [], None
    for t in active:
        mel = next((i for i, r in t["lines"].items() if r["role"] == "melody"), None)
        if mel is None:
            continue
        if last is not None and mel != last:
            holders.append(f"c.{t['bar']} {names[last]} → {names[mel]}")
        last = mel
    if holders:
        shown = "; ".join(holders[:8]) + (f"; … (+{len(holders) - 8})" if len(holders) > 8 else "")
        obs.append(f"Relevos de melodía ({len(holders)}): {shown}.")
    elif last is not None:
        obs.append(f"La melodía la lleva siempre {names[last]}.")
    mx = max(t["voices"] for t in active)
    mn = min(t["voices"] for t in active)
    obs.append(f"Textura: de {mn} a {mx} voces simultáneas; más llenos: "
               f"c.{_runs([t['bar'] for t in active if t['voices'] == mx][:6])}"
               + ("; más ligeros: c." + _runs([t['bar'] for t in active if t['voices'] == mn][:6])
                  if mn != mx else "") + ".")
    for tr in tracks:
        if tr.get("zones") is None or tr["global_role"] == "percussion":
            continue
        n = tr["bars_active"]
        label = f"{names[tr['index']]} ({tr['instrument']})"
        if tr["global_role"] != "bass" and tr["zones"].get("grave", 0) >= 0.5 * n:
            obs.append(f"{label} toca en su registro grave (zona más tapada) "
                       f"{tr['zones']['grave']}/{n} compases.")
        if tr["zones"].get("agudo", 0) >= 0.5 * n:
            obs.append(f"{label} pasa {tr['zones']['agudo']}/{n} compases en su zona aguda.")
        if tr.get("out_of_range_notes"):
            bars_oor = [t["bar"] for t in timeline
                        if t["lines"].get(tr["index"], {}).get("out_of_range")]
            obs.append(f"{label}: {tr['out_of_range_notes']} nota(s) fuera de su tesitura "
                       f"(c.{_runs(bars_oor[:8])}).")
    return obs


# ══════════════════════════════════════════════════════════════════════════════
#  SALIDA
# ══════════════════════════════════════════════════════════════════════════════

ZONE_CH = {"grave": "g", "medio": "m", "agudo": "a", "fuera": "!", "?": "?", "-": "-"}


def _cells_line(cells):
    s = ""
    for n, c in enumerate(cells):
        if n and n % 4 == 0:
            s += " "
        s += c + " "
    return s.rstrip()


def render_text(res):
    B, R, G = _c("B"), _c("R"), _c("G")
    tl, tracks = res["timeline"], res["tracks"]
    names = {t["index"]: (t["name"] or f"pista {t['index']}") for t in tracks}
    shown = [t for t in tracks if t.get("bars_active")]
    shown.sort(key=lambda t: (FAMILY_ORDER.index(t["family"]),
                              -(t["mean_pitch"] or 0)))
    w = min(22, max((len(names[t["index"]]) for t in shown), default=8))
    out = [f"  {B}Compases {res['bars'][0]}–{res['bars'][1]}{R} de {res['total_bars']}"
           f"   ·   {len(shown)} pistas activas",
           f"  {G}celda = rol (M melodía · C contramelodía · B bajo · A armonía · "
           f"P percusión) + registro (g grave · m medio · a agudo · ! fuera de "
           f"tesitura · ? sin identificar) · «·» = calla{R}", ""]
    ons = [t["onsets_per_beat"] for t in tl]
    smax = max(ons) if ons and max(ons) > 0 else 1
    for s in range(0, len(tl), 16):
        chunk = tl[s:s + 16]
        out.append(f"  {G}{'compás':<{w}} " + _cells_line([f"{t['bar'] % 100:>2}" for t in chunk]) + R)
        prev_fam = None
        for tr in shown:
            if prev_fam and tr["family"] != prev_fam:
                out.append("")
            prev_fam = tr["family"]
            cells = []
            for t in chunk:
                r = t["lines"].get(tr["index"])
                if not r:
                    cells.append(f"{G}·{R} ")
                else:
                    ch = ROLE_CH[r["role"]]
                    z = ZONE_CH[r["zone"]]
                    col = _c("RED") if z == "!" else _c(ROLE_COL[ch])
                    cells.append(f"{col}{ch}{z}{R}")
            out.append(f"  {names[tr['index']][:w]:<{w}} " + _cells_line(cells))
        out.append("")
        out.append(f"  {G}{'voces':<{w}} " + _cells_line([f"{t['voices']:>2}" for t in chunk]) + R)
        out.append(f"  {G}{'densidad':<{w}} " + _cells_line(
            [_SPARK[min(7, int(round(t['onsets_per_beat'] / smax * 7)))] + " " for t in chunk]) + R)
        out.append("")
    if res["observations"]:
        out.append(f"  {_c('CYA')}Observaciones:{R}")
        out += [f"    · {o}" for o in res["observations"]]
        out.append("")
    out.append(f"  {B}Por instrumento{R}  {G}(rol global de la obra entre paréntesis){R}")
    for tr in shown:
        roles = ", ".join(f"{ROLE_ES[r]} {n}" for r, n in
                          sorted(tr["roles"].items(), key=lambda x: -x[1]))
        zones = ", ".join(f"{ZONE_CH.get(z, z)}={n}" for z, n in
                          sorted(tr["zones"].items(), key=lambda x: -x[1]))
        out.append(f"    {names[tr['index']]} [{tr['instrument'] or 'sin identificar'}] "
                   f"({ROLE_ES[tr['global_role']]}): activo {tr['bars_active']}/"
                   f"{tr['bars_total']} · roles: {roles} · registro: {zones} · "
                   f"densidad {tr['mean_density']}/pulso")
    for wmsg in res["warnings"]:
        out.append(f"\n  {_c('YEL')}⚠ {wmsg}{R}")
    out.append(f"\n  {G}{res['note']}{R}")
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME VISUAL (HTML autocontenido)
# ══════════════════════════════════════════════════════════════════════════════

_HTML_CSS = """
:root{--bg:#fbfaf7;--fg:#22211e;--mut:#6b6860;--line:#dedbd2;--panel:#f1eee6;--comf:#e6e2d6;
--melody:#2a78d6;--countermelody:#1f9e8a;--bass:#d98a00;--harmony:#8b7fc0;--percussion:#8a8a8a;--bad:#d03a3a}
@media(prefers-color-scheme:dark){:root{--bg:#1b1a18;--fg:#ecebe6;--mut:#a09d93;--line:#3a3833;--panel:#262420;--comf:#34322d;
--melody:#5aa0f0;--countermelody:#3cc2ac;--bass:#f0a826;--harmony:#a99ee0;--percussion:#9a9a9a;--bad:#f06a6a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:20px 16px 40px;max-width:1200px;margin:auto}
h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:26px 0 8px}
.sub{color:var(--mut);margin:0 0 14px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 12px;font-size:13px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--bg)}
.grid{min-width:max-content;padding:6px 0 8px}
.row{display:flex;align-items:center;height:32px}
.row.h{height:22px}.row.s{height:34px}
.lab{position:sticky;left:0;z-index:2;background:var(--bg);width:150px;min-width:150px;padding:0 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;border-right:1px solid var(--line)}
.lab small{color:var(--mut);margin-left:4px}
.fam{padding:8px 8px 2px;color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.06em;position:sticky;left:0}
.c{position:relative;width:22px;min-width:22px;height:28px;margin:0 1px}
.c.g4{margin-left:7px}
.c .cf{position:absolute;left:0;right:0;background:var(--comf);border-radius:2px}
.c .cb{position:absolute;left:3px;right:3px;min-height:4px;border-radius:2px}
.c .cb.full{top:6px;bottom:6px;opacity:.75}
.c.oor{outline:2px solid var(--bad);outline-offset:-1px;border-radius:2px}
.c.e::after{content:"";position:absolute;left:9px;top:13px;width:4px;height:4px;border-radius:50%;background:var(--line)}
.n{width:22px;min-width:22px;margin:0 1px;text-align:center;font-size:10px;color:var(--mut)}
.n.g4{margin-left:7px}.n.on{color:var(--fg);font-weight:600}
.bar{position:relative;width:22px;min-width:22px;height:26px;margin:0 1px}
.bar.g4{margin-left:7px}
.bar i{position:absolute;left:4px;right:4px;bottom:0;background:var(--mut);opacity:.55;border-radius:2px 2px 0 0}
.r-melody{background:var(--melody)}.r-countermelody{background:var(--countermelody)}
.r-bass{background:var(--bass)}.r-harmony{background:var(--harmony)}.r-percussion{background:var(--percussion)}
ul{margin:4px 0 0;padding-left:20px}li{margin:3px 0}
.warn{border-left:3px solid var(--bass);background:var(--panel);padding:8px 12px;margin:8px 0;border-radius:0 6px 6px 0}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:middle}
th{color:var(--mut);font-weight:500}
.chip{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-size:12px}
.stack{display:flex;height:10px;width:140px;border-radius:5px;overflow:hidden;background:var(--panel)}
.note{color:var(--mut);font-size:12px;margin-top:20px}
.how{background:var(--panel);border-radius:8px;padding:10px 12px;font-size:13px;color:var(--mut)}
"""


def render_html(res):
    """Informe visual autocontenido. Cada celda es un minipiano-roll: la barra
    muestra el ámbito de notas del compás sobre la tesitura del instrumento y
    la banda clara su zona cómoda. Color = rol."""
    import html as _h
    e = _h.escape
    tl, tracks = res["timeline"], res["tracks"]
    names = {t["index"]: (t["name"] or f"pista {t['index']}") for t in tracks}
    shown = [t for t in tracks if t.get("bars_active")]
    shown.sort(key=lambda t: (FAMILY_ORDER.index(t["family"]), -(t["mean_pitch"] or 0)))

    observed = {}
    for t in tl:
        for i, r in t["lines"].items():
            if r["low_midi"] is not None:
                lo, hi = observed.get(i, (999, -1))
                observed[i] = (min(lo, r["low_midi"]), max(hi, r["high_midi"]))

    def scale(tr):
        if tr["range"]:
            lo, hi, clo, chi = tr["range"]
            return lo, hi, clo, chi
        lo, hi = observed.get(tr["index"], (48, 72))
        return lo - 2, hi + 2, None, None

    def cell(tr, t):
        g4 = " g4" if (t["bar"] - res["bars"][0]) % 4 == 0 and t["bar"] != res["bars"][0] else ""
        r = t["lines"].get(tr["index"])
        if not r:
            return f'<div class="c e{g4}"></div>'
        tip = (f"c.{t['bar']} · {ROLE_ES[r['role']]}"
               + (f" · registro {r['zone']} · {r['low']}–{r['high']}" if r["role"] != "percussion" else "")
               + f" · {r['onsets']} ataques ({r['density']}/pulso) · vel. {r['velocity']}")
        oor = " oor" if r["out_of_range"] else ""
        if r["role"] == "percussion" or r["low_midi"] is None:
            body = f'<i class="cb full r-{r["role"]}"></i>'
        else:
            lo, hi, clo, chi = scale(tr)
            pc = lambda x: max(0.0, min(100.0, (x - lo) / (hi - lo) * 100))
            band = (f'<u class="cf" style="bottom:{pc(clo):.1f}%;height:{pc(chi) - pc(clo):.1f}%"></u>'
                    if clo is not None else "")
            b0, b1 = pc(r["low_midi"]), pc(r["high_midi"])
            body = band + (f'<i class="cb r-{r["role"]}" style="bottom:{b0:.1f}%;'
                           f'height:{max(b1 - b0, 6):.1f}%"></i>')
        return f'<div class="c{g4}{oor}" title="{e(tip)}">{body}</div>'

    def nums(cls):
        out = []
        for t in tl:
            g4 = " g4" if (t["bar"] - res["bars"][0]) % 4 == 0 and t["bar"] != res["bars"][0] else ""
            on = " on" if t["bar"] % 4 == 1 else ""
            out.append(f'<div class="n{g4}{on}">{t["bar"]}</div>')
        return "".join(out)

    def meter(key, tip):
        mx = max((t[key] for t in tl), default=0) or 1
        out = []
        for t in tl:
            g4 = " g4" if (t["bar"] - res["bars"][0]) % 4 == 0 and t["bar"] != res["bars"][0] else ""
            out.append(f'<div class="bar{g4}" title="c.{t["bar"]} · {tip}: {t[key]}">'
                       f'<i style="height:{t[key] / mx * 100:.0f}%"></i></div>')
        return "".join(out)

    rows, prev = [], None
    for tr in shown:
        if tr["family"] != prev:
            rows.append(f'<div class="fam">{e(tr["family"])}</div>')
            prev = tr["family"]
        inst = tr["instrument"] or "sin identificar"
        rows.append(f'<div class="row"><div class="lab" title="{e(names[tr["index"]])} · {e(inst)}">'
                    f'{e(names[tr["index"]])}<small>{e(inst)}</small></div>'
                    + "".join(cell(tr, t) for t in tl) + "</div>")

    legend = "".join(f'<span><i class="sw r-{r}"></i>{ROLE_ES[r]}</span>' for r in ROLES)
    summ = []
    for tr in shown:
        roles = tr["roles"]
        tot = sum(roles.values()) or 1
        stack = "".join(f'<i class="r-{r}" style="width:{n / tot * 100:.1f}%" '
                        f'title="{ROLE_ES[r]}: {n}"></i>' for r, n in
                        sorted(roles.items(), key=lambda x: -x[1]))
        zones = ", ".join(f"{z} {n}" for z, n in sorted(tr["zones"].items(), key=lambda x: -x[1])
                          if z != "-")
        summ.append(
            f'<tr><td>{e(names[tr["index"]])}<br><small style="color:var(--mut)">'
            f'{e(tr["instrument"] or "sin identificar")}</small></td>'
            f'<td><span class="chip r-{tr["global_role"]}">{ROLE_ES[tr["global_role"]]}</span>'
            f' <small style="color:var(--mut)">{tr["global_confidence"]:.2f}</small></td>'
            f'<td>{tr["bars_active"]}/{tr["bars_total"]}</td>'
            f'<td><div class="stack">{stack}</div></td>'
            f'<td>{e(zones) or "—"}</td><td>{tr["mean_density"]}</td></tr>')

    obs = "".join(f"<li>{e(o)}</li>" for o in res["observations"])
    warns = "".join(f'<div class="warn">{e(w)}</div>' for w in res["warnings"])
    return f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Anatomía orquestal · {e(res["file"])}</title>
<style>{_HTML_CSS}</style></head><body>
<h1>Anatomía orquestal</h1>
<p class="sub">{e(res["file"])} · compases {res["bars"][0]}–{res["bars"][1]} de {res["total_bars"]} · {len(shown)} pistas activas · orchestra_anatomy v{VERSION}</p>
<div class="legend">{legend}</div>
<div class="how">Cada celda es un compás. <b>La barra</b> muestra el ámbito de notas que toca ese compás dentro de la tesitura del instrumento (abajo = grave, arriba = agudo); <b>la banda clara</b> es su zona cómoda; el <b>color</b> es el rol. Contorno rojo = notas fuera de tesitura. Pasa el cursor sobre una celda para ver el detalle.</div>
<h2>Línea de tiempo</h2>
<div class="scroll"><div class="grid">
<div class="row h"><div class="lab">compás</div>{nums("n")}</div>
{"".join(rows)}
<div class="fam">textura</div>
<div class="row s"><div class="lab">voces</div>{meter("voices", "voces")}</div>
<div class="row s"><div class="lab">densidad</div>{meter("onsets_per_beat", "ataques/pulso")}</div>
</div></div>
<h2>Observaciones</h2>
{"<ul>" + obs + "</ul>" if obs else "<p class='sub'>Sin observaciones.</p>"}
{warns}
<h2>Por instrumento</h2>
<div class="scroll"><table><thead><tr><th>Pista</th><th>Rol global</th><th>Activo</th><th>Roles por compás</th><th>Registro (compases)</th><th>Densidad/pulso</th></tr></thead>
<tbody>{"".join(summ)}</tbody></table></div>
<p class="note">{e(res["note"])}</p>
</body></html>
"""


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="orchestra_anatomy_v2.py",
        description="Fotografía de una obra orquestal: rol, registro y densidad por compás.")
    ap.add_argument("midi")
    ap.add_argument("--bars", help="rango A-B (1-based, inclusivo)")
    ap.add_argument("--map", action="append", default=[], metavar="K=INSTR",
                    help="fija el instrumento de una pista (índice o nombre). Repetible")
    ap.add_argument("--roles-file", dest="roles_file")
    ap.add_argument("--html")
    ap.add_argument("--json")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False
    try:
        nmap = {}
        for item in args.map:
            if "=" not in item:
                raise ValueError(f"--map espera K=INSTRUMENTO (recibido {item!r})")
            k, v = item.split("=", 1)
            nmap[k.strip()] = v.strip().lower()
        rp = args.roles_file or args.midi + ".roles.json"
        res = analyze(args.midi, bars=parse_bars(args.bars), name_map=nmap,
                      forced=read_forced(rp))
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if args.json == "-":
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    B, R, C = _c("B"), _c("R"), _c("CYA")
    print(f"\n{'═' * 78}")
    print(f"  {B}ORCHESTRA ANATOMY v{VERSION}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")
    print(render_text(res))
    if args.html:
        with open(_ensure_parent(args.html), "w", encoding="utf-8") as fh:
            fh.write(render_html(res))
        print(f"\n  {C}HTML: {args.html}{R}")
    if args.json:
        with open(_ensure_parent(args.json), "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        print(f"\n  {C}JSON: {args.json}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
