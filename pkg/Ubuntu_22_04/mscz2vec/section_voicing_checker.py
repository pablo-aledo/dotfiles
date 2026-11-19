#!/usr/bin/env python3
"""
section_voicing_checker.py — Section Voicing Checker: conducción de voces entre
los instrumentos de una misma sección y entre secciones de una orquestación
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
harmony_checker.py corrige armonizaciones SATB (cuatro voces, una pista).
En una orquestación las voces están repartidas en PISTAS de instrumentos
(violines I y II, violas, chelos; flauta, oboe, clarinete, fagot; trompas,
trombones...). Esta herramienta aplica el mismo criterio de profesor de
armonía a cada SECCIÓN de la orquesta y a la unión de todas: lee el MIDI,
agrupa las pistas por familia, ordena las voces por altura y las muestrea en
una rejilla. No reescribe nada: señala y explica.

COMPROBACIONES
  5ª paralelas     — dos voces de la sección se mueven en quintas justas
  8ª paralelas     — solo se avisa si NO es un doblaje intencionado (dos
                     instrumentos a la octava durante la mayor parte de la obra
                     son un doblaje de orquestación, no un error)
  directas         — llegada a 5ª/8ª por movimiento directo entre voces extremas
  cruce            — una voz inferior pasa por encima de la superior
  solapamiento     — una voz invade el registro que acaba de dejar su vecina
  espaciado        — hueco mayor de una 8ª entre voces superiores adyacentes
  saltos interiores— saltos mayores de una 5ª en voces interiores [inner_voices]
  terceras graves  — 2as/3as entre las dos voces más graves bajo C3 [low_chords]
  sensible         — con --key: la sensible que no resuelve a la tónica (info)

COMO MÓDULO:
  from section_voicing_checker import check_sections

DEPENDENCIAS: playability_auditor.py (E/S MIDI y catálogo de instrumentos) ·
orchestral_atlas.py (reglas y explicaciones)

LIMITACIONES: muestrea una nota por pista y por punto de la rejilla (la más
aguda si la pista toca acordes), así que las divisi dentro de una pista no se
analizan; las reglas son las de la armonía a cuatro voces, útiles como guía de
estudio, no como ley en orquestación; las secciones se deducen del instrumento
(nombre de pista o programa GM).

CÓDIGOS DE SALIDA:  0 ok · 1 error de ejecución · 2 uso incorrecto (fichero
inexistente, compases o tonalidad inválidos) · 3 selftest fallido o, con
--strict, hallazgos de severidad aviso/error

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  check     Analiza un MIDI multipista y explica los errores de conducción
  selftest  Comprueba la herramienta con MIDIs sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  check     midi  [--section strings|woodwinds|brass|all] [--grid BEATS]
            [--bars A-B] [--key KEY] [--strict] [--json FILE|-] [--no-color]
  selftest  [--no-color]

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
Una línea por parámetro: <nombre> <tipo> [clave=valor …] [# descripción].
Tipos: int float str bool enum file dir list kv. Claves: default min max typical
unit values of io ext format required advanced when excludes alias. Sin min/max =
sin límite conocido. Se verifica contra el argparse con «suite_index.py lint
--check spec-vs-code».

@params v1
[*]
--no-color bool  default=false
[check]
midi      file  io=in ext=mid required  # MIDI multipista
--section enum  default=all values=strings|woodwinds|brass|all  # Sección a analizar
--grid    float default=1.0 min=0.25 unit=tiempos typical=0.5..2  # Paso de muestreo
--bars    str   format="A-B"  # Rango de compases (1-based)
--key     str   format="Cmaj|Am"  # Tonalidad para comprobar la sensible
--strict  bool  default=false  # rc=3 si hay avisos o errores
--json    file  io=out ext=json  # Sidecar JSON ('-' = stdout)
[selftest]
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python section_voicing_checker.py check obra_orquestada.mid
  python section_voicing_checker.py check obra.mid --section strings --grid 0.5
  python section_voicing_checker.py check obra.mid --bars 5-12 --key Cmaj
  python section_voicing_checker.py check obra.mid --json voces.json --strict
  python section_voicing_checker.py selftest
"""

import os
import re
import sys
import json
import bisect
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from playability_auditor import (read_midi, TimeContext, extract_notes, pitch_name,
                                     INSTRUMENT_DB, resolve_instrument)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)
try:
    from orchestral_atlas import RULES, ATLAS
except ImportError:
    print("ERROR: requiere orchestral_atlas.py en el mismo directorio (catálogo)")
    sys.exit(1)

VERSION = "1.0"
TOOL = "section_voicing_checker"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_SEV_W = {"error": 0.15, "aviso": 0.07, "info": 0.02}
_SEV_MARK = {"error": "✖", "aviso": "⚑", "info": "ℹ"}
_SEV_ORDER = {"error": 0, "aviso": 1, "info": 2}
LOW_LIMIT = 48
SECTIONS = {"strings": "cuerda_frotada", "woodwinds": "viento_madera", "brass": "viento_metal"}
SECTION_ES = {"strings": "cuerdas", "woodwinds": "maderas", "brass": "metales"}
RULE_TEXT = {r["id"]: r for r in RULES}


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre."""
    if isinstance(path, (str, bytes, os.PathLike)):
        d = os.path.dirname(os.path.abspath(os.fsdecode(path)))
        if d:
            os.makedirs(d, exist_ok=True)
    return path


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def _wrap(text, indent, width=78):
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _ranges(bars) -> str:
    b = sorted(set(int(x) for x in bars))
    if not b:
        return ""
    out, s, p = [], b[0], b[0]
    for x in b[1:] + [None]:
        if x is not None and x == p + 1:
            p = x
            continue
        out.append(f"{s}" if s == p else f"{s}–{p}")
        if x is not None:
            s = p = x
    return "c." + ", ".join(out)


def _parse_bars(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if s is None or s == "":
        return None
    m = re.fullmatch(r"\s*(\d+)\s*(?:[-–:]\s*(\d+))?\s*", str(s))
    if not m:
        raise ValueError(f"--bars inválido: {s!r} (usa A-B, p.ej. 5-12)")
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    if a < 1 or b < a:
        raise ValueError(f"--bars inválido: {s!r} (A ≥ 1 y B ≥ A)")
    return a, b


_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_key(s: Optional[str]) -> Optional[Tuple[int, bool]]:
    """'Cmaj', 'Am', 'F#min', 'Bb' → (tónica pc, es_menor)."""
    if not s:
        return None
    m = re.fullmatch(r"\s*([A-Ga-g])([#b]?)\s*(maj|major|mayor|min|minor|menor|m)?\s*", s)
    if not m:
        raise ValueError(f"--key inválida: {s!r} (usa Cmaj, Am, F#min, Bb…)")
    pc = _PC[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    minor = (m.group(3) or "").lower() in ("min", "minor", "menor", "m") or m.group(1).islower() and not m.group(3)
    return pc % 12, bool(minor)


def _bar_grid(mid, last_tick: int):
    tsm = sorted(mid.timesig_map) or [(0, 4, 4)]
    bars, tick, ti = [], 0, 0
    num, den = tsm[0][1], tsm[0][2]
    while tick <= last_tick and len(bars) < 100000:
        while ti + 1 < len(tsm) and tsm[ti + 1][0] <= tick:
            ti += 1
            num, den = tsm[ti][1], tsm[ti][2]
        tpc = max(1, mid.tpb * 4 * num // den)
        bars.append((tick, tick + tpc))
        tick += tpc
    return bars


def _top_at(starts, notes, t):
    """Nota más aguda sonando en el tick t (o None)."""
    i = bisect.bisect_right(starts, t) - 1
    best = None
    j = i
    while j >= 0 and i - j < 24:
        n = notes[j]
        if n.end > t and (best is None or n.pitch > best):
            best = n.pitch
        j -= 1
    return best


def _mk(code, sev, bars, tracks, message, why, suggestion, rule=None):
    return dict(id=code, severity=sev, bars=sorted(set(bars)), bars_text=_ranges(bars),
                tracks=list(tracks), message=message, why=why, suggestion=suggestion, rule=rule)


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════

def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    mid = read_midi(path)
    tracks = []
    for i, trk in enumerate(mid.tracks):
        ns = [n for n in extract_notes(trk) if not (n.pitch < 28 and n.end - n.start <= max(4, mid.tpb // 32))]
        if not ns:
            continue
        key, _ = resolve_instrument(trk, ns)
        prof = INSTRUMENT_DB.get(key, INSTRUMENT_DB["sintetizador"])
        if prof["family"].startswith("percusion") or key == "percusion":
            continue
        es = ATLAS[key]["es"] if key in ATLAS else prof["es"]
        tracks.append(dict(idx=i, name=trk.name or f"Pista {i}", key=key, es=es,
                           family=prof["family"], notes=ns, starts=[n.start for n in ns],
                           mean=sum(n.pitch for n in ns) / len(ns)))
    if not tracks:
        raise ValueError("el MIDI no contiene pistas con notas melódicas")
    last = max(n.end for t in tracks for n in t["notes"])
    return mid, tracks, _bar_grid(mid, max(0, last - 1))


def _label(t):
    return f"{t['name']} ({t['es']})" if t["name"].lower() != t["es"].lower() else t["name"]


def check_sections(path: str, section: str = "all", grid: float = 1.0,
                   bars: Optional[Tuple[int, int]] = None, key: Optional[str] = None) -> dict:
    """API pública. Devuelve hallazgos de conducción de voces + elogios + overall."""
    if section not in SECTIONS and section != "all":
        raise ValueError(f"sección desconocida: {section}")
    if grid <= 0:
        raise ValueError("--grid debe ser positivo")
    tonic = parse_key(key) if isinstance(key, str) else key
    mid, tracks, bar_grid = _load(path)
    n_bars = len(bar_grid)
    b0, b1 = (1, n_bars) if bars is None else bars
    if b0 > n_bars:
        raise ValueError(f"--bars fuera de la obra (la obra tiene {n_bars} compases)")
    b1 = min(b1, n_bars)

    # puntos de muestreo (tick, compás)
    step = max(1, int(round(grid * mid.tpb)))
    samples = []
    for bi in range(b0 - 1, b1):
        s, e = bar_grid[bi]
        t = s
        while t < e:
            samples.append((t, bi + 1))
            t += step
    for t in tracks:
        t["seq"] = [_top_at(t["starts"], t["notes"], tk) for tk, _ in samples]

    findings: List[dict] = []
    praise: List[str] = []
    groups = {}
    for name, fam in SECTIONS.items():
        if section in ("all", name):
            vs = sorted([t for t in tracks if t["family"] == fam], key=lambda t: -t["mean"])
            if len(vs) >= 2:
                groups[name] = vs

    for name, vs in groups.items():
        sec = SECTION_ES[name]
        par5, par8, direct, cross, overl, spac, leaps, lead = {}, {}, {}, {}, {}, {}, {}, {}
        def _oct_ratio(A, B):
            n_both = n_oct = 0
            for k in range(len(samples)):
                if A["seq"][k] is not None and B["seq"][k] is not None:
                    n_both += 1
                    if (A["seq"][k] - B["seq"][k]) % 12 == 0:
                        n_oct += 1
            return n_oct / n_both if n_both else 0.0

        # un instrumento que dobla a otro a la octava (≥75 % del tiempo) no es una voz
        # independiente: se analiza solo la voz «cabeza» del doblaje
        ind, doubled_by = [], {}
        for V in vs:
            head = next((U for U in ind if _oct_ratio(U, V) >= 0.75), None)
            if head is None:
                ind.append(V)
            else:
                doubled_by.setdefault(head["idx"], []).append(V["name"])
        vs = ind

        def L(V):
            extra = doubled_by.get(V["idx"])
            return _label(V) + (f" (+ {', '.join(extra)} a la octava)" if extra else "")
        if len(vs) < 2:
            continue
        same_oct = {}
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                same_oct[(a, b)] = _oct_ratio(vs[a], vs[b])
        for k in range(1, len(samples)):
            bar = samples[k][1]
            for a in range(len(vs)):
                for b in range(a + 1, len(vs)):
                    A, B = vs[a], vs[b]
                    a0, a1, b0_, b1_ = A["seq"][k - 1], A["seq"][k], B["seq"][k - 1], B["seq"][k]
                    if None in (a0, a1, b0_, b1_):
                        continue
                    ma, mb = a1 - a0, b1_ - b0_
                    i0, i1 = a0 - b0_, a1 - b1_
                    both_move = ma != 0 and mb != 0
                    similar = (ma > 0) == (mb > 0)
                    if both_move and similar and i0 > 0 and i1 > 0:
                        if i0 % 12 == 7 and i1 % 12 == 7:
                            par5.setdefault((a, b), []).append(bar)
                        elif i0 % 12 == 0 and i1 % 12 == 0 and same_oct[(a, b)] < 0.75:
                            par8.setdefault((a, b), []).append(bar)
                        elif (a, b) == (0, len(vs) - 1) and i1 % 12 in (0, 7) and i0 % 12 != i1 % 12 \
                                and abs(ma) > 2 and abs(mb) > 2:
                            direct.setdefault((a, b), []).append(bar)
                    if a1 < b1_ and a0 >= b0_:
                        cross.setdefault((a, b), []).append(bar)
                    elif b == a + 1 and a1 < b0_ - 0 and a1 >= b1_ and ma != 0:
                        overl.setdefault((a, b), []).append(bar)
        for a in range(len(vs) - 1):
            A, B = vs[a], vs[a + 1]
            limit = 19 if a + 1 == len(vs) - 1 else 12
            for k, bar in ((k, samples[k][1]) for k in range(len(samples))):
                if A["seq"][k] is not None and B["seq"][k] is not None:
                    if A["seq"][k] - B["seq"][k] > limit and A["seq"][k] - B["seq"][k] > 0:
                        spac.setdefault((a, a + 1), []).append(bar)
        for a in range(1, len(vs) - 1):
            V = vs[a]
            for k in range(1, len(samples)):
                p0, p1 = V["seq"][k - 1], V["seq"][k]
                if p0 is not None and p1 is not None and abs(p1 - p0) > 7:
                    leaps.setdefault(a, []).append(samples[k][1])
        if tonic:
            tpc, minor = tonic
            lt = (tpc - 1) % 12
            for a, V in enumerate(vs):
                for k in range(1, len(samples)):
                    p0, p1 = V["seq"][k - 1], V["seq"][k]
                    if p0 is not None and p1 is not None and p0 % 12 == lt and p1 % 12 != lt \
                            and p1 % 12 != tpc and a in (0, len(vs) - 1):
                        lead.setdefault(a, []).append(samples[k][1])

        for (a, b), bs in par5.items():
            findings.append(_mk(
                "parallel_5ths", "aviso", bs, [L(vs[a]), L(vs[b])],
                f"Quintas paralelas en {sec} entre {L(vs[a])} y {L(vs[b])} ({len(bs)} movimientos).",
                "Dos voces que se mueven a la vez en quinta justa suenan como una sola voz «engrosada» y pierden independencia; es el error más clásico de la armonía a voces.",
                "Mueve una de las dos voces en movimiento contrario u oblicuo, o cambia el acorde de uno de los dos puntos para que el intervalo sea 3ª o 6ª.",
                "inner_voices"))
        for (a, b), bs in par8.items():
            findings.append(_mk(
                "parallel_8ves", "aviso", bs, [L(vs[a]), L(vs[b])],
                f"Octavas paralelas puntuales entre {L(vs[a])} y {L(vs[b])} en {sec}.",
                "Si no es un doblaje mantenido, una octava paralela suelta hace que una voz «desaparezca» dentro de la otra.",
                "O dobla de forma consecuente (todo el pasaje a la octava) o separa las voces con movimiento contrario.",
                "octave_doubling"))
        for (a, b), bs in direct.items():
            findings.append(_mk(
                "direct_5th_8ve", "info", bs, [L(vs[a]), L(vs[b])],
                f"Quinta u octava directa entre las voces extremas de {sec}.",
                "Llegar a 5ª/8ª por movimiento directo con saltos en las voces extremas deja un hueco armónico muy audible.",
                "Haz que la voz superior llegue por grado conjunto, o mueve el bajo en sentido contrario.",
                "spacing"))
        for (a, b), bs in cross.items():
            findings.append(_mk(
                "voice_crossing", "aviso", bs, [L(vs[a]), L(vs[b])],
                f"Cruce de voces: {L(vs[a])} queda por debajo de {L(vs[b])} en {sec}.",
                "Cuando dos instrumentos intercambian su posición el oído pierde la línea de cada uno (y el balance de la sección).",
                "Mantén el orden de registro, o hazlo a propósito dándole a uno un timbre distinto para que se siga su línea.",
                "masking_same_register"))
        for (a, b), bs in overl.items():
            findings.append(_mk(
                "voice_overlap", "info", bs, [L(vs[a]), L(vs[b])],
                f"Solapamiento: {L(vs[a])} invade el registro que {L(vs[b])} acababa de dejar.",
                "El solapamiento entre voces vecinas hace dudar de a qué instrumento pertenece cada nota.",
                "Reduce el salto o reparte la nota al instrumento que ya está en esa zona.",
                "masking_same_register"))
        for (a, b), bs in spac.items():
            findings.append(_mk(
                "wide_spacing", "aviso", bs, [L(vs[a]), L(vs[b])],
                f"Espaciado excesivo entre {L(vs[a])} y {L(vs[b])} (más de una 8ª) en {sec}.",
                "Entre voces superiores adyacentes los huecos grandes suenan vacíos y rompen la fusión de la sección; solo el grave admite huecos amplios.",
                "Acerca las voces o añade una voz intermedia (otro instrumento o divisi).",
                "spacing"))
        for a, bs in leaps.items():
            findings.append(_mk(
                "inner_leaps", "info", bs, [L(vs[a])],
                f"{L(vs[a])} da saltos mayores de una 5ª siendo voz interior de {sec}.",
                "Las voces interiores deben ir por el camino más corto: un salto grande las saca del papel de relleno y llama la atención.",
                "Redistribuye la nota a una voz vecina o resuelve por grado conjunto.",
                "inner_voices"))
        for a, bs in lead.items():
            findings.append(_mk(
                "leading_tone", "info", bs, [L(vs[a])],
                f"La sensible en {L(vs[a])} no resuelve a la tónica.",
                "La sensible en una voz extrema pide subir un semitono a la tónica; si no lo hace el oído lo nota.",
                "Resuelve la sensible a la tónica, salvo que la armonía lo justifique (voz interior, cadencia evitada).",
                None))

    # terceras graves entre las dos voces más graves de TODA la orquesta
    if section == "all" and len(tracks) >= 2:
        bad = []
        for k in range(len(samples)):
            snd = sorted([(t["seq"][k], t) for t in tracks if t["seq"][k] is not None], key=lambda x: x[0])
            if len(snd) >= 2:
                lo, hi = snd[0][0], snd[1][0]
                if lo < LOW_LIMIT and 1 <= hi - lo <= 4:
                    bad.append(samples[k][1])
        if bad:
            findings.append(_mk(
                "low_thirds", "aviso", bad, ["voces más graves"],
                f"Las dos voces más graves forman 2ª o 3ª por debajo de C3 ({len(set(bad))} compases).",
                "Los intervalos cerrados en el registro grave producen batidos y enturbian el bajo [low_chords].",
                "Abre el espaciado: quinta u octava abajo y las terceras por encima de C3.",
                "low_chords"))

    findings.sort(key=lambda f: (_SEV_ORDER[f["severity"]], f["bars"][0] if f["bars"] else 0))
    n_sec = len(groups)
    if n_sec and not any(f["id"] in ("parallel_5ths", "parallel_8ves") for f in findings):
        praise.append("Sin quintas ni octavas paralelas entre las voces de las secciones analizadas.")
    if n_sec and not any(f["id"] in ("voice_crossing", "voice_overlap") for f in findings):
        praise.append("Las voces de cada sección mantienen su orden de registro (sin cruces).")
    if n_sec and not any(f["id"] == "wide_spacing" for f in findings):
        praise.append("Espaciado equilibrado entre voces adyacentes.")
    if section == "all" and n_sec and not any(f["id"] == "low_thirds" for f in findings):
        praise.append("El grave está abierto: sin terceras cerradas bajo C3.")
    if n_sec == 0:
        findings.append(_mk("no_sections", "info", [], [], "No hay ninguna sección con al menos dos instrumentos "
                            "para comprobar la conducción de voces.",
                            "La conducción entre instrumentos solo se puede medir si hay dos o más pistas de la misma familia.",
                            "Reparte tu armonía entre varios instrumentos de la sección (p. ej. violines II, violas y chelos).", None))
    pen = sum(_SEV_W[f["severity"]] for f in findings)
    overall = None if n_sec == 0 else round(max(0.0, 1.0 - pen), 2)
    cnt = {s: sum(1 for f in findings if f["severity"] == s) for s in _SEV_W}
    return dict(tool=TOOL, version=VERSION, file=path, overall=overall,
                summary=dict(sections=n_sec, bars=b1 - b0 + 1, samples=len(samples),
                             errors=cnt["error"], warnings=cnt["aviso"], infos=cnt["info"]),
                section=section, grid=grid, bars_range=[b0, b1],
                sections_found={SECTION_ES[k]: [_label(t) for t in v] for k, v in groups.items()},
                findings=findings, elogios=praise)


def print_report(res: dict):
    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}\n  {B}SECTION VOICING CHECKER v{VERSION}  ·  {res['file']}{R}\n{'═' * 78}")
    s = res["summary"]
    print(f"  compases {res['bars_range'][0]}–{res['bars_range'][1]} · rejilla {res['grid']} t · "
          f"secciones: {', '.join(res['sections_found']) or '—'}")
    for sec, voces in res["sections_found"].items():
        print(f"    {sec:<8} (de agudo a grave): {' > '.join(voces)}")
    print(f"  valoración: {C}{res['overall'] if res['overall'] is not None else '—'}{R}   "
          f"errores {s['errors']} · avisos {s['warnings']} · info {s['infos']}")
    if res["elogios"]:
        print(f"\n  {_c('GRN')}Lo que funciona:{R}")
        for e in res["elogios"]:
            print(_wrap("✓ " + e, "    "))
    if res["findings"]:
        print(f"\n  {_c('YEL')}Para mejorar:{R}")
        for f in res["findings"]:
            col = _c("RED") if f["severity"] == "error" else (_c("YEL") if f["severity"] == "aviso" else G)
            where = f" [{f['bars_text']}]" if f["bars_text"] else ""
            print(_wrap(f"{col}{_SEV_MARK[f['severity']]}{R} {f['message']}{where}", "    "))
            print(_wrap("¿Por qué? " + f["why"], "        "))
            print(_wrap("Prueba: " + f["suggestion"], "        "))
            if f.get("rule"):
                print(f"        {G}regla del atlas: python orchestral_atlas.py rules --grep {f['rule']}{R}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST (MIDI sintético)
# ══════════════════════════════════════════════════════════════════════════════

def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _trk(name, prog, ch, notes, tempo=False):
    ev = [(0, 0, b"\xFF\x03" + _vlq(len(name)) + name.encode("latin-1"))]
    if tempo:
        ev.append((0, 0, b"\xFF\x51\x03" + (600000).to_bytes(3, "big")))
        ev.append((0, 0, b"\xFF\x58\x04\x04\x02\x18\x08"))
    ev.append((0, 1, bytes([0xC0 | ch, prog])))
    for st, dur, p, v in notes:
        ev.append((st, 3, bytes([0x90 | ch, p, v])))
        ev.append((st + dur - 2, 2, bytes([0x80 | ch, p, 0])))
    ev.sort(key=lambda e: (e[0], e[1]))
    body, last = b"", 0
    for t, _, b in ev:
        body += _vlq(t - last) + b
        last = t
    return b"MTrk" + (len(body) + 4).to_bytes(4, "big") + body + _vlq(0) + b"\xFF\x2F\x00"


def _synthetic(path, bad: bool):
    tpb = 480
    bar = tpb * 4
    roots = [48, 45, 41, 43] * 2
    vln = [(i * bar, bar, r + 24 - (0 if i % 2 else 0), 80) for i, r in enumerate(roots)]
    vla = [(i * bar, bar, r + 12 + (4 if (r % 12) in (0, 5, 7) else 3), 80) for i, r in enumerate(roots)]
    cel = [(i * bar, bar, r, 80) for i, r in enumerate(roots)]
    if bad:
        vla = [(s, d, p + 7, v) for (s, d, p, v) in cel]          # quintas paralelas con el chelo
        vla = [(s, d, p + 12, v) if False else (s, d, p, v) for (s, d, p, v) in vla]
    tracks = [_trk("Conductor", 0, 0, [], tempo=True),
              _trk("Violin I", 40, 0, vln), _trk("Viola", 41, 1, vla), _trk("Violonchelo", 42, 2, cel)]
    if bad:
        tracks.append(_trk("Contrabajo", 43, 3, [(s, d, p - 12 + 5, 80) for (s, d, p, v) in cel[:4]]))
    hdr = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + tpb.to_bytes(2, "big")
    _ensure_parent(path)
    Path(path).write_bytes(hdr + b"".join(tracks))


def cmd_selftest(a):
    errs = []
    with tempfile.TemporaryDirectory() as d:
        bad, good = os.path.join(d, "bad.mid"), os.path.join(d, "good.mid")
        _synthetic(bad, True)
        _synthetic(good, False)
        rb = check_sections(bad, "strings")
        rg = check_sections(good, "strings")
        ids = {f["id"] for f in rb["findings"]}
        if "parallel_5ths" not in ids:
            errs.append("no detecta las quintas paralelas viola/chelo")
        if "parallel_5ths" in {f["id"] for f in rg["findings"]}:
            errs.append("detecta quintas paralelas donde no las hay")
        if rb["overall"] is None or rg["overall"] is None or not rb["overall"] < rg["overall"]:
            errs.append("el MIDI defectuoso debería puntuar peor que el bueno")
        if check_sections(bad, "strings", bars=(2, 3))["summary"]["bars"] != 2:
            errs.append("--bars no limita el análisis")
        try:
            check_sections(bad, "strings", bars=(99, 100))
            errs.append("--bars fuera de la obra debería fallar")
        except ValueError:
            pass
        if _parse_bars("5-8") != (5, 8) or _ranges({2, 3, 4, 7}) != "c.2–4, 7" or parse_key("Am") != (9, True):
            errs.append("parse de compases/tonalidad incorrecto")
    for e in errs:
        print(f"  ✗ {e}")
    print(f"  {'OK' if not errs else 'FALLO'} · selftest de {TOOL}")
    return 0 if not errs else 3


def cmd_check(a):
    bars = _parse_bars(a.bars)
    parse_key(a.key)
    res = check_sections(a.midi, a.section, a.grid, bars, a.key)
    if a.json == "-":
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_report(res)
        if a.json:
            _ensure_parent(a.json)
            Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {_c('CYA')}JSON: {a.json}{_c('R')}\n")
    if a.strict and (res["summary"]["errors"] or res["summary"]["warnings"]):
        return 3
    return 0


def main():
    global _USE_COLOR
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true")
    ap = argparse.ArgumentParser(
        prog="section_voicing_checker.py",
        description="Conducción de voces entre instrumentos de una misma sección y entre secciones.")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("check", parents=[common], help="Analiza un MIDI multipista")
    p.add_argument("midi")
    p.add_argument("--section", choices=["strings", "woodwinds", "brass", "all"], default="all")
    p.add_argument("--grid", type=float, default=1.0, help="Paso de muestreo en tiempos")
    p.add_argument("--bars", help="Rango de compases A-B (1-based)")
    p.add_argument("--key", help="Tonalidad (Cmaj, Am…) para comprobar la sensible")
    p.add_argument("--strict", action="store_true", help="rc=3 si hay avisos o errores")
    p.add_argument("--json", help="Sidecar JSON ('-' = stdout)")
    p.set_defaults(fn=cmd_check)

    p = sp.add_parser("selftest", parents=[common], help="Verifica con MIDIs sintéticos")
    p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args()
    if a.no_color:
        _USE_COLOR = False
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    except Exception as e:                                  # pragma: no cover
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
