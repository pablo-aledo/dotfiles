#!/usr/bin/env python3
"""
orchestra_compare.py — Orchestra Compare: compara TU orquestación con una
referencia y explica las diferencias, sin declarar ganador
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Aprender a orquestar es tomar decisiones y ver qué otras decisiones son
posibles. Generar la orquestación con piano_to_orchestra.py u orchestrator.py
y copiarla no enseña nada; en cambio, ORQUESTAR TÚ PRIMERO y comparar después
con una referencia (una orquestación generada, una obra de estudio, la
versión de un compañero...) sí. Esta herramienta lee dos MIDIs multipista y,
tramo a tramo, cuenta qué hace cada versión y qué efecto tímbrico tiene cada
decisión (apoyándose en orchestral_atlas), con una pregunta de estudio por
diferencia. No puntúa quién gana: no hay una orquestación «correcta».

POR TRAMO (compases consecutivos con las mismas diferencias)
  línea principal — qué instrumento lleva la melodía en cada versión, en qué
                    zona de registro y qué carácter tiene esa zona
  instrumentos    — quién aparece solo en una de las dos versiones
  registro        — cuántos semitonos más grave/agudo canta la línea principal
  densidad        — nº de instrumentos simultáneos
  dinámica        — velocity media (diferencias notables)
  bajo            — quién lleva el bajo en cada versión
  doblajes        — pares a la octava/unísono que solo existen en una versión
                    (y si son una combinación recomendada del atlas)

Las líneas se emparejan por ROL (role_detector) y no por nombre de pista, y se
indica si la línea principal de ambas es el mismo material melódico (contorno)
o material distinto.

COMO MÓDULO:
  from orchestra_compare import compare_midis

DEPENDENCIAS: playability_auditor.py (E/S MIDI) · orchestral_atlas.py (zonas,
combinaciones, reglas) · role_detector.py (opcional, por subprocess; si no
está disponible se usa una heurística interna)

LIMITACIONES: compara MIDI, no el sonido de tus librerías. Los roles son
heurísticos y se asignan por pista completa (no por compás). Si las dos
versiones tienen plantillas muy distintas, algunas diferencias serán
inevitables y lo interesante es la pregunta, no la diferencia.

CÓDIGOS DE SALIDA:  0 ok · 1 error de ejecución · 2 uso incorrecto (fichero
inexistente, compases inválidos) · 3 selftest fallido o, con --strict, si hay
diferencias de línea principal

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  compare   Compara dos MIDIs (tu versión y la referencia)
  selftest  Comprueba la herramienta con MIDIs sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  compare   MINE REF  [--bars A-B] [--label-ref TXT] [--strict]
            [--json FILE|-] [--no-color]
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
[compare]
mine        file  io=in ext=mid required  # Tu orquestación
ref         file  io=in ext=mid required  # Orquestación de referencia
--bars      str   format="A-B"  # Rango de compases (1-based)
--label-ref str   default="la referencia"  # Cómo llamar a la referencia en el texto
--strict    bool  default=false  # rc=3 si la línea principal difiere en algún tramo
--json      file  io=out ext=json  # Sidecar JSON ('-' = stdout)
[selftest]
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python orchestra_compare.py compare mi_version.mid referencia.mid
  python orchestra_compare.py compare mia.mid generada.mid --bars 1-8 --label-ref "piano_to_orchestra"
  python orchestra_compare.py compare mia.mid ref.mid --json comparacion.json
  python orchestra_compare.py selftest
"""

import os
import re
import sys
import json
import bisect
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from playability_auditor import (read_midi, extract_notes, pitch_name,
                                     INSTRUMENT_DB, resolve_instrument)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)
try:
    from orchestral_atlas import ATLAS, zone_of, projection, find_combos, combos_for, ROLE_ES
except ImportError:
    print("ERROR: requiere orchestral_atlas.py en el mismo directorio (catálogo)")
    sys.exit(1)

VERSION = "1.0"
TOOL = "orchestra_compare"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


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


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA Y ROLES
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
        perc = prof["family"].startswith("percusion") or key == "percusion"
        tracks.append(dict(idx=i, name=trk.name or f"Pista {i}", key=key,
                           es=ATLAS[key]["es"] if key in ATLAS else prof["es"],
                           family=prof["family"], perc=perc, notes=ns,
                           starts=[n.start for n in ns],
                           mean=sum(n.pitch for n in ns) / len(ns)))
    if not tracks:
        raise ValueError(f"{path}: el MIDI no contiene pistas con notas")
    last = max(n.end for t in tracks for n in t["notes"])
    return mid, tracks, _bar_grid(mid, max(0, last - 1))


def _roles_detector(path) -> Optional[Dict[int, str]]:
    script = Path(__file__).resolve().parent / "role_detector.py"
    if not script.is_file():
        return None
    try:
        r = subprocess.run([sys.executable, str(script), "detect", path, "--json", "-"],
                           capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout[r.stdout.index("{"):])
        return {int(t["index"]): t["role"] for t in data.get("tracks", []) if t.get("n_notes", 0) > 0}
    except Exception:
        return None


def _roles_fallback(tracks) -> Dict[int, str]:
    live = [t for t in tracks if not t["perc"]]
    roles = {t["idx"]: "percussion" for t in tracks if t["perc"]}
    if not live:
        return roles
    mx = max(len(t["notes"]) for t in live)
    lo, hi = min(t["mean"] for t in live), max(t["mean"] for t in live)

    def score(t):
        reg = (t["mean"] - lo) / (hi - lo) if hi > lo else 0.5
        return 0.5 * reg + 0.5 * len(t["notes"]) / mx
    best = max(live, key=score)
    for t in live:
        roles[t["idx"]] = "melody" if t is best else ("bass" if t["mean"] <= 52 else "harmony")
    return roles


def _roles(path, tracks) -> Dict[int, str]:
    r = _roles_detector(path) or {}
    fb = _roles_fallback(tracks)
    return {t["idx"]: r.get(t["idx"], fb.get(t["idx"], "harmony")) for t in tracks}


# ══════════════════════════════════════════════════════════════════════════════
#  DATOS POR COMPÁS
# ══════════════════════════════════════════════════════════════════════════════

def _bar_notes(t, s, e):
    i = bisect.bisect_left(t["starts"], e)
    return [n for n in t["notes"][max(0, i - 400):i] if n.end > s and n.start < e]


def _dens_bucket(n):
    return 0 if n == 0 else (1 if n <= 2 else (2 if n <= 5 else 3))


def _contour(notes):
    ps = [n.pitch for n in sorted(notes, key=lambda n: n.start)]
    return [(b > a) - (b < a) for a, b in zip(ps, ps[1:])]


def _analyse(mid, tracks, roles, bar_grid) -> List[Optional[dict]]:
    out = []
    for (s, e) in bar_grid:
        act = {}
        for t in tracks:
            ns = _bar_notes(t, s, e)
            if ns:
                act[t["idx"]] = ns
        if not act:
            out.append(None)
            continue
        by_idx = {t["idx"]: t for t in tracks}
        cand = [i for i in act if roles[i] == "melody"] or \
               [i for i in act if roles[i] == "countermelody"] or \
               [i for i in act if not by_idx[i]["perc"]]
        lead_i = max(cand, key=lambda i: sum(n.pitch for n in act[i]) / len(act[i])) if cand else None
        lead = None
        if lead_i is not None:
            lt = by_idx[lead_i]
            mp = round(sum(n.pitch for n in act[lead_i]) / len(act[lead_i]))
            z = zone_of(lt["key"], mp)
            lead = dict(key=lt["key"], es=lt["es"], pitch=mp, zone_id=z["id"], zone=z["label"],
                        character=z["character"], power=z["power"],
                        proj=projection(lt["key"], mp, int(sum(n.vel for n in act[lead_i]) / len(act[lead_i]))),
                        contour=_contour(act[lead_i]))
        bass_i = [i for i in act if roles[i] == "bass"]
        vels = [n.vel for ns in act.values() for n in ns]
        keys = sorted({by_idx[i]["key"] for i in act})
        # dobles: pistas con ataques coincidentes a unísono/octava
        dbl = set()
        ids = [i for i in act if not by_idx[i]["perc"]]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                A = {n.start: n.pitch for n in act[ids[x]]}
                B = {n.start: n.pitch for n in act[ids[y]]}
                common = [k for k in A if k in B]
                if len(common) >= 2 and all((A[k] - B[k]) % 12 == 0 for k in common) \
                        and by_idx[ids[x]]["key"] != by_idx[ids[y]]["key"]:
                    dbl.add(tuple(sorted((by_idx[ids[x]]["key"], by_idx[ids[y]]["key"]))))
        out.append(dict(lead=lead, bass=sorted({by_idx[i]["key"] for i in bass_i}),
                        keys=keys, n_inst=len(act), vel=sum(vels) / len(vels),
                        doubles=sorted(dbl), roles={ROLE_ES.get(roles[i], roles[i]): by_idx[i]["key"] for i in act}))
    return out


def _name(k):
    return ATLAS[k]["es"] if k in ATLAS else k


def _combo_note(pair):
    for c in combos_for(pair[0]):
        if pair[1] in c["insts"]:
            return f"es una combinación recomendada del atlas («{c['name']}»: {c['effect']})"
    return "no figura entre las combinaciones del atlas: escúchala y decide si funde"


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _sig(m, r):
    ml = m["lead"]["key"] if m and m["lead"] else None
    rl = r["lead"]["key"] if r and r["lead"] else None
    mz = m["lead"]["zone_id"] if m and m["lead"] else None
    rz = r["lead"]["zone_id"] if r and r["lead"] else None
    return (ml, rl, mz, rz, tuple(m["keys"]) if m else (), tuple(r["keys"]) if r else (),
            _dens_bucket(m["n_inst"]) if m else 0, _dens_bucket(r["n_inst"]) if r else 0,
            tuple(m["bass"]) if m else (), tuple(r["bass"]) if r else (),
            tuple(m["doubles"]) if m else (), tuple(r["doubles"]) if r else (),
            round(m["vel"] / 16) if m else -1, round(r["vel"] / 16) if r else -1)


def _describe(bars, M, R, label):
    """Diferencias de un tramo (usa el primer compás como representante)."""
    m, r = M[0], R[0]
    diffs = []
    if m is None or r is None:
        who, other = ("tu versión", label) if m is None else (label, "tu versión")
        diffs.append(dict(kind="silencio", severity="info",
                          text=f"Solo suena {other} (en {who} este tramo está vacío).",
                          effect="Un tramo en silencio también es una decisión orquestal: respiro, contraste o pausa dramática.",
                          question="¿Es intencionado el silencio? ¿Qué aportaría (o restaría) llenarlo?"))
        return diffs
    ml, rl = m["lead"], r["lead"]
    if ml and rl:
        same_line = bool(ml["contour"] and rl["contour"] and
                         sum(1 for a, b in zip(ml["contour"], rl["contour"]) if a == b) /
                         max(len(ml["contour"]), len(rl["contour"])) >= 0.6)
        line_note = ("(misma línea melódica)" if same_line else
                     "(material melódico distinto: compara los timbres, no las notas)")
        if ml["key"] != rl["key"]:
            stronger = "tu elección proyecta más" if ml["proj"] > rl["proj"] else \
                       ("la de " + label + " proyecta más" if rl["proj"] > ml["proj"] else "proyectan parecido")
            diffs.append(dict(
                kind="lead", severity="aviso",
                text=(f"La línea principal la lleva {_name(ml['key'])} en {pitch_name(ml['pitch'])} ({ml['zone']}: "
                      f"{ml['character']}) en tu versión y {_name(rl['key'])} en {pitch_name(rl['pitch'])} "
                      f"({rl['zone']}: {rl['character']}) en {label} {line_note}."),
                effect=f"Cambia el color y el peso de la melodía; {stronger}.",
                question=f"¿Qué emoción gana la melodía con {_name(rl['key'])} que no tiene con {_name(ml['key'])}, y al revés?"))
        elif ml["zone_id"] != rl["zone_id"]:
            diffs.append(dict(
                kind="zone", severity="info",
                text=(f"Los dos usan {_name(ml['key'])} para la línea principal, pero tú en {ml['zone']} "
                      f"({pitch_name(ml['pitch'])}: {ml['character']}) y {label} en {rl['zone']} "
                      f"({pitch_name(rl['pitch'])}: {rl['character']}) {line_note}."),
                effect="El mismo instrumento cambia de carácter con el registro.",
                question=f"¿Por qué suena distinto {_name(ml['key'])} en esas dos zonas? ¿Cuál se oye mejor sobre el acompañamiento?"))
        dp = ml["pitch"] - rl["pitch"]
        if abs(dp) >= 7 and ml["key"] == rl["key"] and ml["zone_id"] == rl["zone_id"]:
            diffs.append(dict(kind="register", severity="info",
                              text=f"Tu línea principal está {abs(dp)} semitonos más {'aguda' if dp > 0 else 'grave'} que la de {label}.",
                              effect="Más agudo = más brillo y tensión; más grave = más calidez y menos proyección.",
                              question="¿Qué efecto buscas en esa altura?"))
    only_m = sorted(set(m["keys"]) - set(r["keys"]))
    only_r = sorted(set(r["keys"]) - set(m["keys"]))
    if only_m or only_r:
        parts = []
        if only_m:
            parts.append("solo en tu versión: " + ", ".join(_name(k) for k in only_m))
        if only_r:
            parts.append(f"solo en {label}: " + ", ".join(_name(k) for k in only_r))
        diffs.append(dict(kind="instruments", severity="info", text="Instrumentos distintos — " + "; ".join(parts) + ".",
                          effect="Cada familia aporta un color: los instrumentos que faltan o sobran cambian la paleta del tramo.",
                          question="¿Qué papel cumple en ese tramo cada instrumento que no coincide? ¿Qué pierdes sin él?"))
    if abs(m["n_inst"] - r["n_inst"]) >= 2:
        mas = "tu versión" if m["n_inst"] > r["n_inst"] else label
        diffs.append(dict(kind="density", severity="info",
                          text=f"Densidad: tú {m['n_inst']} instrumentos a la vez, {label} {r['n_inst']} ({mas} es más densa).",
                          effect="Menos instrumentos con roles claros suelen sonar más claros [density].",
                          question="¿Quitar un instrumento haría la melodía más clara? ¿Cuál sobra?"))
    if abs(m["vel"] - r["vel"]) >= 16:
        diffs.append(dict(kind="dynamics", severity="info",
                          text=f"Dinámica media: tú {m['vel']:.0f}, {label} {r['vel']:.0f} (velocity).",
                          effect="La dinámica global marca la intensidad del tramo.",
                          question="¿Qué intensidad quieres en este punto de la forma?"))
    if m["bass"] != r["bass"]:
        diffs.append(dict(kind="bass", severity="info",
                          text=(f"Bajo: tú {', '.join(_name(k) for k in m['bass']) or '—ninguno—'}; "
                                f"{label} {', '.join(_name(k) for k in r['bass']) or '—ninguno—'}."),
                          effect="El instrumento del bajo define el peso y la claridad del cimiento [bass_clarity].",
                          question="¿Qué ataque y qué peso necesita el bajo en este tramo?"))
    for pair in sorted(set(m["doubles"]) - set(r["doubles"])):
        diffs.append(dict(kind="double_mine", severity="info",
                          text=f"Solo tú doblas {_name(pair[0])} + {_name(pair[1])}: {_combo_note(pair)}.",
                          effect="El doblaje funde o realza el timbre [octave_doubling].",
                          question="¿Lo necesitas o la línea ya se oye sola?"))
    for pair in sorted(set(r["doubles"]) - set(m["doubles"])):
        diffs.append(dict(kind="double_ref", severity="info",
                          text=f"Solo {label} dobla {_name(pair[0])} + {_name(pair[1])}: {_combo_note(pair)}.",
                          effect="El doblaje funde o realza el timbre [octave_doubling].",
                          question="¿Qué cambia en la melodía con ese doblaje? Pruébalo con doubling_lab.py."))
    return diffs


def compare_midis(mine: str, ref: str, bars: Optional[Tuple[int, int]] = None,
                  label_ref: str = "la referencia") -> dict:
    """API pública. Devuelve tramos con diferencias, coincidencias y preguntas."""
    mm, mt, mg = _load(mine)
    rm, rt, rg = _load(ref)
    MD = _analyse(mm, mt, _roles(mine, mt), mg)
    RD = _analyse(rm, rt, _roles(ref, rt), rg)
    n = max(len(MD), len(RD))
    b0, b1 = (1, n) if bars is None else bars
    if b0 > n:
        raise ValueError(f"--bars fuera de la obra (la más larga tiene {n} compases)")
    b1 = min(b1, n)
    get = lambda L, i: L[i] if i < len(L) else None
    segs, cur = [], None
    for bi in range(b0 - 1, b1):
        m, r = get(MD, bi), get(RD, bi)
        sg = _sig(m, r)
        if cur and cur["sig"] == sg:
            cur["bars"].append(bi + 1)
        else:
            cur = dict(sig=sg, bars=[bi + 1], M=[m], R=[r])
            segs.append(cur)
    differences, coincide = [], []
    for sgm in segs:
        ds = _describe(sgm["bars"], sgm["M"], sgm["R"], label_ref)
        if ds:
            differences.append(dict(bars=sgm["bars"], bars_text=_ranges(sgm["bars"]), differences=ds))
        else:
            m = sgm["M"][0]
            if m:
                coincide.append(dict(bars=sgm["bars"], bars_text=_ranges(sgm["bars"]),
                                     text=("misma línea principal" + (f" ({_name(m['lead']['key'])})" if m["lead"] else "")
                                           + " y misma plantilla")))
    # similitud estructural
    tot = same_lead = jac = same_bass = 0
    for bi in range(b0 - 1, b1):
        m, r = get(MD, bi), get(RD, bi)
        if not m or not r:
            continue
        tot += 1
        if m["lead"] and r["lead"] and m["lead"]["key"] == r["lead"]["key"]:
            same_lead += 1
        a, b = set(m["keys"]), set(r["keys"])
        jac += len(a & b) / len(a | b) if a | b else 1
        same_bass += 1 if m["bass"] == r["bass"] else 0
    sim = round((0.4 * same_lead + 0.4 * jac + 0.2 * same_bass) / tot, 2) if tot else None
    lead_diffs = sum(1 for d in differences for x in d["differences"] if x["kind"] in ("lead", "zone"))
    qs = []
    for d in differences:
        for x in d["differences"]:
            if x["question"] not in qs:
                qs.append(x["question"])
    generic = ["¿Qué decisión de tu versión defenderías ante un director y por qué?",
               "¿Qué tomarías de la otra versión para probarlo en tu mockup?",
               "¿Qué escucharías en FL Studio, con tus librerías, para decidir entre las dos?"]
    qs = (qs[:3] + generic)[:3] if len(qs) < 3 else qs[:3]
    return dict(tool=TOOL, version=VERSION, file=mine, ref=ref, overall=None,
                summary=dict(bars_compared=b1 - b0 + 1, segments=len(segs),
                             segments_with_differences=len(differences), lead_differences=lead_diffs,
                             structural_similarity=sim,
                             similarity_note="índice de similitud estructural 0-1 (no es una puntuación de calidad)"),
                label_ref=label_ref, bars_range=[b0, b1], differences=differences,
                coincidences=coincide, questions=qs)


def print_report(res: dict):
    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    s = res["summary"]
    print(f"\n{'═' * 78}\n  {B}ORCHESTRA COMPARE v{VERSION}{R}\n  tuya: {res['file']}\n  {res['label_ref']}: {res['ref']}\n{'═' * 78}")
    print(f"  compases {res['bars_range'][0]}–{res['bars_range'][1]} · tramos: {s['segments']} "
          f"({s['segments_with_differences']} con diferencias)")
    print(f"  similitud estructural: {C}{s['structural_similarity']}{R} {G}(no puntúa quién orquesta mejor){R}")
    if res["coincidences"]:
        print(f"\n  {_c('GRN')}En lo siguiente coincidís:{R}")
        for c in res["coincidences"]:
            print(_wrap(f"✓ {c['bars_text']}: {c['text']}", "    "))
    for d in res["differences"]:
        print(f"\n  {B}{d['bars_text']}{R}")
        for x in d["differences"]:
            print(_wrap("→ " + x["text"], "    "))
            print(_wrap("Efecto: " + x["effect"], "        "))
            print(_wrap("Para pensar: " + x["question"], "        "))
    print(f"\n  {B}Para reflexionar{R}")
    for i, q in enumerate(res["questions"], 1):
        print(_wrap(f"{i}. {q}", "    "))
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST
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


def _synthetic(path, lead_name, lead_prog, shift):
    tpb, bar = 480, 480 * 4
    mel = [72, 74, 76, 79]
    notes = [(b * bar + i * tpb, tpb, mel[i] + shift, 80) for b in range(8) for i in range(4)]
    pad = [(b * bar, bar, 48 + (b % 2) * 3 + 12, 70) for b in range(8)]
    bass = [(b * bar, bar, 36, 80) for b in range(8)]
    tracks = [_trk("Conductor", 0, 0, [], tempo=True), _trk(lead_name, lead_prog, 0, notes),
              _trk("Violonchelo", 42, 1, pad), _trk("Contrabajo", 43, 2, bass)]
    hdr = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + tpb.to_bytes(2, "big")
    _ensure_parent(path)
    Path(path).write_bytes(hdr + b"".join(tracks))


def cmd_selftest(a):
    errs = []
    with tempfile.TemporaryDirectory() as d:
        mine, ref = os.path.join(d, "mine.mid"), os.path.join(d, "ref.mid")
        _synthetic(mine, "Flauta", 73, -9)
        _synthetic(ref, "Violin I", 40, 0)
        r = compare_midis(mine, ref)
        if r["summary"]["lead_differences"] < 1:
            errs.append("no detecta que la línea principal cambia de instrumento")
        if not any(x["kind"] == "lead" for dd in r["differences"] for x in dd["differences"]):
            errs.append("falta la diferencia «lead»")
        same = compare_midis(ref, ref)
        if same["summary"]["segments_with_differences"] != 0 or same["summary"]["structural_similarity"] != 1.0:
            errs.append("comparar un MIDI consigo mismo debería dar 0 diferencias y similitud 1.0")
        if compare_midis(mine, ref, bars=(2, 3))["summary"]["bars_compared"] != 2:
            errs.append("--bars no limita la comparación")
        try:
            compare_midis(mine, ref, bars=(99, 100))
            errs.append("--bars fuera de la obra debería fallar")
        except ValueError:
            pass
        if len(r["questions"]) != 3:
            errs.append("deben salir 3 preguntas de reflexión")
        if _parse_bars("5-8") != (5, 8) or _ranges({2, 3, 4, 7}) != "c.2–4, 7":
            errs.append("parse/agrupación de compases incorrectos")
    for e in errs:
        print(f"  ✗ {e}")
    print(f"  {'OK' if not errs else 'FALLO'} · selftest de {TOOL}")
    return 0 if not errs else 3


def cmd_compare(a):
    bars = _parse_bars(a.bars)
    res = compare_midis(a.mine, a.ref, bars, a.label_ref)
    if a.json == "-":
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_report(res)
        if a.json:
            _ensure_parent(a.json)
            Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {_c('CYA')}JSON: {a.json}{_c('R')}\n")
    if a.strict and res["summary"]["lead_differences"]:
        return 3
    return 0


def main():
    global _USE_COLOR
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true")
    ap = argparse.ArgumentParser(
        prog="orchestra_compare.py",
        description="Compara tu orquestación con una referencia explicando las diferencias.")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("compare", parents=[common], help="Compara dos MIDIs")
    p.add_argument("mine")
    p.add_argument("ref")
    p.add_argument("--bars", help="Rango de compases A-B (1-based)")
    p.add_argument("--label-ref", default="la referencia", help="Cómo llamar a la referencia")
    p.add_argument("--strict", action="store_true", help="rc=3 si la línea principal difiere")
    p.add_argument("--json", help="Sidecar JSON ('-' = stdout)")
    p.set_defaults(fn=cmd_compare)

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
