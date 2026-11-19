#!/usr/bin/env python3
"""
orchestra_critic.py — Orchestra Critic: crítica didáctica de una orquestación
propia (por instrumento y por compás)
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
melody_critic enseña a mejorar UNA melodía; quality_scorer y
playability_auditor puntúan o verifican orquestaciones, pero ninguna
herramienta EXPLICA por qué un registro o un reparto de instrumentos
funciona o no. Esta herramienta es el equivalente de melody_critic para la
orquesta: lee tu MIDI multipista y devuelve comentario pedagógico, apoyado en
el catálogo de orchestral_atlas (zonas de registro, proyección, combinaciones
y reglas), con tono de profesor: qué funciona, qué mejorar y POR QUÉ.

POR PISTA
  zonas       — % de notas en cada zona de registro del instrumento y qué
                carácter tiene («flauta: 80 % en registro grave: soplado y
                débil; queda tapada»)
  rango       — notas fuera del rango físico
  dinámica    — dinámica fuerte en zonas débiles (flauta grave a ff)
  respiración — frases de viento que no dejan respirar
  proyección  — potencia estimada media (heurística de equilibrio)

POR COMPÁS / GLOBAL
  melodía enterrada   — proyección de la melodía frente al mejor acompañamiento
  enmascaramiento     — melodía y acompañamiento del mismo timbre a < 5 st
  dobles              — unísonos/octavas entre pistas y si son una combinación
                        recomendada del atlas
  graves turbios      — terceras/segundas bajo C3 entre voces adyacentes
  espaciado           — huecos > octava entre voces adyacentes superiores
  bajo                — compases con varios instrumentos y sin bajo
  densidad            — nº de instrumentos y de notas por compás

Cada hallazgo lleva severidad (error / aviso / info), compases afectados
agrupados («c.2–15»), la explicación, una sugerencia concreta con alternativas
del atlas y, cuando procede, el id de regla entre corchetes ([low_chords]).

LIMITACIONES: trabaja sobre MIDI, no sobre el audio de tus librerías: la
«proyección» es una estimación por familia, registro y velocity. No juzga
cómo suena de verdad un patch concreto (calibra siempre por oído). La
detección de la melodía usa role_detector.py (vía subprocess) o, si no está
disponible (p. ej. falta mido), una heurística interna menos fiable: puedes
forzar instrumentos con --instrument N=CLAVE. Los instrumentos sintetizados
no identificados (clave `sintetizador`) no se analizan por registro.

COMO MÓDULO:
  from orchestra_critic import critique_orchestration

DEPENDENCIAS: stdlib · playability_auditor.py (E/S MIDI e instrumentos) ·
orchestral_atlas.py (catálogo) · role_detector.py (opcional, por subprocess)

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  critique  Crítica de una orquestación (MIDI multipista)
  selftest  Verifica la herramienta con MIDIs sintéticos (uno defectuoso)

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  critique  midi [--bars A-B] [--track N] [--instrument N=CLAVE ...]
                 [--strict] [--verbose] [--json FILE|-] [--no-color]
  selftest  [--no-color]

  --bars A-B          Limita el análisis a un rango de compases (1-based)
  --track N           Solo hallazgos que afectan a la pista N (índice 0-based)
  --instrument N=K    Fuerza el instrumento de la pista N (clave de
                      playability_auditor: flauta, oboe, violin, trompa...)
  --strict            rc=3 si hay avisos o errores
  --verbose           Tabla por compás (densidad y equilibrio melodía/acomp.)
  --json FILE         Sidecar JSON ('-' = stdout)

CÓDIGOS DE SALIDA:  0 ok · 1 error de ejecución · 2 uso incorrecto (fichero
                    inexistente, --bars/--track/--instrument inválidos, MIDI
                    sin notas) · 3 selftest fallido o, con --strict, hay
                    avisos o errores

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
[critique]
midi         file  io=in ext=mid required  # MIDI multipista a criticar
--bars       str   format="A-B"  # Rango de compases (1-based)
--track      int   min=0  # Solo hallazgos de esta pista
--instrument list  of=str format="N=CLAVE"  # Forzar instrumento de una pista
--strict     bool  default=false  # rc=3 si hay avisos o errores
--verbose    bool  default=false  # Tabla por compás
--json       file  io=out ext=json  # Sidecar JSON ('-' = stdout)
[selftest]
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python orchestra_critic.py critique obra_orquestada.mid
  python orchestra_critic.py critique obra.mid --bars 9-16 --verbose
  python orchestra_critic.py critique obra.mid --track 1
  python orchestra_critic.py critique obra.mid --instrument 3=oboe --json critica.json
  python orchestra_critic.py selftest
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
    from playability_auditor import (read_midi, TimeContext, extract_notes, pitch_name,
                                     INSTRUMENT_DB, resolve_instrument)
    try:
        from playability_auditor import _split_keyswitches
    except ImportError:                                   # pragma: no cover
        def _split_keyswitches(notes, tpb):
            max_dur = max(4, tpb // 32)
            real, ks = [], []
            for n in notes:
                (ks if (n.pitch < 28 and (n.end - n.start) <= max_dur) else real).append(n)
            return real, ks
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)
try:
    from orchestral_atlas import (ATLAS, zone_of, projection, best_for_role, find_combos,
                                  RULES, register_of, ROLE_ES)
except ImportError:
    print("ERROR: requiere orchestral_atlas.py en el mismo directorio (catálogo)")
    sys.exit(1)

VERSION = "1.0"
TOOL = "orchestra_critic"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_SPARK = "▁▂▃▄▅▆▇█"
_SEV_ORDER = {"error": 0, "aviso": 1, "info": 2}
_SEV_W = {"error": 0.25, "aviso": 0.10, "info": 0.02}
_SEV_MARK = {"error": "✖", "aviso": "⚑", "info": "ℹ"}
LOW_LIMIT = 48            # C3: por debajo, terceras y segundas suenan turbias
RULE_TEXT = {r["id"]: r for r in RULES}


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


def sparkline(v) -> str:
    v = [float(x) for x in v]
    if not v:
        return ""
    lo, hi = min(v), max(v)
    if hi <= lo:
        return _SPARK[3] * len(v)
    return "".join(_SPARK[min(7, int(round((x - lo) / (hi - lo) * 7)))] for x in v)


def _ranges(bars) -> str:
    """{2,3,4,7} → 'c.2–4, 7'."""
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


def _wrap(text, indent, width=78):
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA: pistas, compases, roles
# ══════════════════════════════════════════════════════════════════════════════

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


def parse_instrument_overrides(items) -> Dict[int, str]:
    """«N=CLAVE» (o N=nombre: «corno inglés», «chelo»…, resuelto con el atlas)."""
    out = {}
    for it in items or []:
        m = re.fullmatch(r"\s*(\d+)\s*=\s*(.+?)\s*", str(it))
        if not m:
            raise ValueError(f"--instrument inválido: {it!r} (usa N=CLAVE, p.ej. 3=oboe)")
        key = m.group(2)
        if key not in INSTRUMENT_DB:
            try:
                from orchestral_atlas import instrument_key as _ik
                key = _ik(key) or key
            except ImportError:
                pass
        if key not in INSTRUMENT_DB:
            raise ValueError(f"instrumento desconocido: {m.group(2)!r} "
                             f"(claves: {', '.join(sorted(INSTRUMENT_DB))})")
        out[int(m.group(1))] = key
    return out


def _bar_grid(mid, last_tick: int):
    """Lista [(inicio, fin)] de compases a partir del mapa de compás."""
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


def _load(path: str, overrides: Dict[int, str]):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    mid = read_midi(path)
    tc = TimeContext(mid)
    tracks = []
    for i, trk in enumerate(mid.tracks):
        real, _ks = _split_keyswitches(extract_notes(trk), mid.tpb)
        if not real:
            continue
        key, _note = resolve_instrument(trk, real, overrides.get(i))
        prof = INSTRUMENT_DB.get(key, INSTRUMENT_DB["sintetizador"])
        es = ATLAS[key]["es"] if key in ATLAS else prof["es"]
        tracks.append(dict(idx=i, name=trk.name or f"Pista {i}", key=key, es=es,
                           family=prof["family"], prof=prof, notes=real,
                           perc=prof["family"].startswith("percusion") or key == "percusion",
                           generic=(key == "sintetizador")))
    if not tracks:
        raise ValueError("el MIDI no contiene pistas con notas")
    for oi in overrides:
        if oi >= len(mid.tracks):
            raise ValueError(f"--instrument: la pista {oi} no existe (hay {len(mid.tracks)})")
    last = max(n.end for t in tracks for n in t["notes"])
    return mid, tc, tracks, _bar_grid(mid, max(0, last - 1))


def _roles_detector(path: str) -> Optional[Dict[int, Tuple[str, float]]]:
    """Roles por pista vía role_detector.py (subprocess). None si no está disponible."""
    script = Path(__file__).resolve().parent / "role_detector.py"
    if not script.is_file():
        return None
    try:
        r = subprocess.run([sys.executable, str(script), "detect", path, "--json", "-"],
                           capture_output=True, text=True, timeout=90)
        if r.returncode != 0:
            return None
        txt = r.stdout
        data = json.loads(txt[txt.index("{"):])
        return {int(t["index"]): (t["role"], float(t.get("confidence", 0.5)))
                for t in data.get("tracks", []) if t.get("n_notes", 0) > 0}
    except Exception:
        return None


def _roles_fallback(tracks) -> Dict[int, Tuple[str, float]]:
    """Heurística interna sencilla: melodía = registro alto + actividad + monofonía."""
    live = [t for t in tracks if not t["perc"]]
    if not live:
        return {}
    means = {t["idx"]: sum(n.pitch for n in t["notes"]) / len(t["notes"]) for t in live}
    lo, hi = min(means.values()), max(means.values())
    mx = max(len(t["notes"]) for t in live)

    def mono(t):
        by = {}
        for n in t["notes"]:
            by[n.start] = by.get(n.start, 0) + 1
        return sum(1 for v in by.values() if v == 1) / max(1, len(by))

    def score(t):
        reg = (means[t["idx"]] - lo) / (hi - lo) if hi > lo else 0.5
        return 0.45 * reg + 0.35 * (len(t["notes"]) / mx) + 0.2 * mono(t)

    best = max(live, key=score)
    roles = {best["idx"]: ("melody", 0.4)}
    for t in live:
        if t["idx"] == best["idx"]:
            continue
        roles[t["idx"]] = ("bass", 0.4) if (means[t["idx"]] <= 52 and mono(t) > 0.6) \
            else ("harmony", 0.3)
    return roles


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS POR COMPÁS
# ══════════════════════════════════════════════════════════════════════════════

class _Score:
    """Notas de cada pista indexadas por compás y por rejilla de medias negras."""

    def __init__(self, mid, tracks, bars, lo_bar, hi_bar):
        self.tpb = mid.tpb
        self.starts = [b[0] for b in bars]
        self.bars = bars
        self.lo, self.hi = lo_bar, hi_bar
        self.ovl = {t["idx"]: {} for t in tracks}      # compás → notas que lo solapan
        self.ons = {t["idx"]: {} for t in tracks}      # compás → notas que empiezan en él
        step = max(1, mid.tpb // 2)
        self.step = step
        self.samples = {}                              # k → [(pista, pitch, vel)]
        for t in tracks:
            for n in t["notes"]:
                b0, b1 = self.bar_of(n.start), self.bar_of(max(n.start, n.end - 1))
                if b1 < lo_bar or b0 > hi_bar:
                    continue
                for b in range(max(b0, lo_bar), min(b1, hi_bar) + 1):
                    self.ovl[t["idx"]].setdefault(b, []).append(n)
                if lo_bar <= b0 <= hi_bar:
                    self.ons[t["idx"]].setdefault(b0, []).append(n)
                k0 = -(-n.start // step)
                k1 = (n.end - 1) // step
                for k in range(k0, k1 + 1):
                    if lo_bar <= self.bar_of(k * step) <= hi_bar:
                        self.samples.setdefault(k, []).append((t["idx"], n.pitch, n.vel))

    def bar_of(self, tick: int) -> int:
        return max(1, bisect.bisect_right(self.starts, tick))


def _mk(code, sev, bars, tracks, message, why, suggestion, rule=None, atlas_key=None):
    return dict(id=code, severity=sev, bars=sorted(set(bars)), bars_text=_ranges(bars),
                tracks=list(tracks), message=message, why=why, suggestion=suggestion,
                rule=rule, more=(f"python orchestral_atlas.py show {atlas_key}" if atlas_key else None))


def _shift_suggestion(t) -> Optional[str]:
    """Desplazamiento de octava que lleva ≥70 % de las notas a zonas potentes."""
    ns = t["notes"]
    for s in (12, -12, 24, -24):
        ok = 0
        for n in ns:
            z = zone_of(t["key"], n.pitch + s)
            if z["in_range"] and z["power"] >= 2:
                ok += 1
        if ok / len(ns) >= 0.7:
            return f"{'+' if s > 0 else ''}{s} semitonos ({'subirla' if s > 0 else 'bajarla'} {abs(s) // 12} octava{'s' if abs(s) > 12 else ''})"
    return None


def _alts(role, mean_pitch, exclude, top=3):
    try:
        res = best_for_role(role, pitch=int(round(mean_pitch)), top=10)
    except ValueError:
        return []
    return [r for r in res if r["key"] not in exclude and r["key"] not in ("percusion", "timbal", "laminas")][:top]


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def critique_orchestration(path: str, bars: Optional[Tuple[int, int]] = None,
                           track: Optional[int] = None,
                           overrides: Optional[Dict[int, str]] = None) -> dict:
    """API pública. Devuelve dict con pistas, hallazgos, elogios, curva de densidad y overall."""
    overrides = overrides or {}
    mid, tc, tracks, bar_list = _load(path, overrides)
    total = len(bar_list)
    lo_bar, hi_bar = (1, total) if not bars else bars
    if lo_bar > total:
        raise ValueError(f"--bars fuera de la obra: la obra tiene {total} compases")
    hi_bar = min(hi_bar, total)
    if track is not None and track not in [t["idx"] for t in tracks]:
        raise ValueError(f"--track {track}: la pista no existe o no tiene notas "
                         f"(pistas con notas: {', '.join(str(t['idx']) for t in tracks)})")
    sc = _Score(mid, tracks, bar_list, lo_bar, hi_bar)
    by_idx = {t["idx"]: t for t in tracks}
    live = [t for t in tracks if not t["perc"]]
    blist = list(range(lo_bar, hi_bar + 1))

    # ── roles ──
    roles = _roles_detector(path)
    role_src = "role_detector"
    if roles is None:
        roles, role_src = _roles_fallback(tracks), "heurística interna"
    mel_idx = None
    mels = [(c, i) for i, (r, c) in roles.items() if r == "melody" and i in by_idx and not by_idx[i]["perc"]]
    if mels:
        mel_idx = max(mels)[1]
    elif live:
        mel_idx = _roles_fallback(tracks).get and max(
            (i for i, (r, _c) in _roles_fallback(tracks).items() if r == "melody"), default=None)

    findings, elogios = [], []

    # ── por pista: zonas, rango, dinámica, proyección ──
    tinfo = []
    for t in tracks:
        ns = [n for n in t["notes"] if lo_bar <= sc.bar_of(n.start) <= hi_bar]
        if not ns:
            continue
        info = dict(idx=t["idx"], name=t["name"], instrument=t["key"], es=t["es"],
                    family=t["family"], n_notes=len(ns), role=roles.get(t["idx"], ("?", 0))[0],
                    bars=_ranges({sc.bar_of(n.start) for n in ns}))
        if t["perc"] or t["generic"]:
            info.update(zones=[], proj_mean=None,
                        comment=("instrumento no identificado: no se analiza el registro "
                                 f"(fuerza uno con --instrument {t['idx']}=CLAVE)" if t["generic"]
                                 else "percusión: sin análisis de registro"))
            if t["generic"]:
                findings.append(_mk("instrument_unknown", "info", {sc.bar_of(n.start) for n in ns}, [t["idx"]],
                                    f"{t['name']}: no se ha podido identificar el instrumento.",
                                    "Sin instrumento no se puede juzgar el registro ni el equilibrio.",
                                    f"Pon el nombre del instrumento en la pista o fuerza uno con --instrument {t['idx']}=CLAVE."))
            tinfo.append(info)
            continue
        zc, zbars = {}, {}
        weak_bars, oor_bars, dyn_bars, oor_n = set(), set(), set(), 0
        for n in ns:
            z = zone_of(t["key"], n.pitch)
            b = sc.bar_of(n.start)
            zc.setdefault(z["label"], dict(z, count=0))["count"] += 1
            if not z["in_range"]:
                oor_n += 1
                oor_bars.add(b)
            if z["power"] <= 1 and z["in_range"]:
                weak_bars.add(b)
                if n.vel >= 96:
                    dyn_bars.add(b)
        zones = sorted(zc.values(), key=lambda z: -z["count"])
        for z in zones:
            z["pct"] = round(100.0 * z["count"] / len(ns), 1)
        proj = [projection(t["key"], n.pitch, n.vel) for n in ns]
        info["proj_mean"] = round(sum(proj) / len(proj), 3)
        info["zones"] = [dict(label=z["label"], pct=z["pct"], lo=z["lo"], hi=z["hi"],
                              power=z["power"], character=z["character"]) for z in zones]
        dz = zones[0]
        info["comment"] = (f"{t['es']}: {dz['pct']:.0f} % de las notas en «{dz['label']}» "
                           f"({pitch_name(dz['lo'])}–{pitch_name(dz['hi'])}): {dz['character']}.")
        tinfo.append(info)
        mean_pitch = sum(n.pitch for n in ns) / len(ns)
        weak_share = sum(z["count"] for z in zones if z["power"] <= 1 and z["in_range"]) / len(ns)
        is_mel = (t["idx"] == mel_idx)
        if weak_share >= 0.5:
            wz = max((z for z in zones if z["power"] <= 1), key=lambda z: z["count"])
            shift = _shift_suggestion(t)
            alts = _alts("melody" if is_mel else "countermelody", mean_pitch, {t["key"]})
            sug = []
            if shift:
                sug.append(f"Prueba a moverla {shift} hasta una zona potente")
            if alts:
                sug.append("o confía la línea a " + ", ".join(a["es"] for a in alts)
                           + f" (python orchestral_atlas.py role {'melody' if is_mel else 'countermelody'} --pitch {pitch_name(int(round(mean_pitch)))})")
            wrule = "winds_low" if t["family"] == "viento_madera" else "natural_dynamics"
            findings.append(_mk(
                "register_weak", "aviso", weak_bars, [t["idx"]],
                f"{t['es']} ({t['name']}): {weak_share:.0%} de las notas en «{wz['label']}» "
                f"({pitch_name(wz['lo'])}–{pitch_name(wz['hi'])})"
                + (" y es la línea melódica" if is_mel else "") + ".",
                f"{wz['character'].capitalize()}. Esa zona proyecta poco, así que la línea queda tapada "
                f"por el resto de la orquesta [{wrule}].",
                (" ".join(s if s.endswith(".") else s for s in sug) + ".") if sug else
                "Escríbela en una zona más potente o refuérzala con otro instrumento.",
                rule=wrule, atlas_key=t["key"] if t["key"] in ATLAS else None))
        if oor_n:
            findings.append(_mk(
                "out_of_range", "error", oor_bars, [t["idx"]],
                f"{t['es']} ({t['name']}): {oor_n} nota(s) fuera del rango físico "
                f"({pitch_name(t['prof']['range'][0])}–{pitch_name(t['prof']['range'][1])}).",
                "Un intérprete real no puede tocarlas y las librerías las suenan mal (o no las tienen).",
                "Transpón esas notas por octavas al rango del instrumento o reasigna la línea (python playability_auditor.py --fix).",
                atlas_key=t["key"] if t["key"] in ATLAS else None))
        if dyn_bars:
            findings.append(_mk(
                "dynamics_vs_register", "aviso", dyn_bars, [t["idx"]],
                f"{t['es']} ({t['name']}): dinámica fuerte (velocity ≥ 96) en una zona débil del instrumento.",
                "Cada registro tiene su dinámica natural: en zonas débiles el instrumento no puede sonar fuerte "
                "sin deformarse [natural_dynamics].",
                "Baja la velocity o sube/baja la línea hasta una zona potente; si quieres fuerza, usa otro instrumento.",
                rule="natural_dynamics", atlas_key=t["key"] if t["key"] in ATLAS else None))
        strong = sum(z["count"] for z in zones if z["power"] >= 3 and z["in_range"]) / len(ns)
        if strong >= 0.7:
            sz = max((z for z in zones if z["power"] >= 3), key=lambda z: z["count"])
            elogios.append(dict(text=f"{t['es']} ({t['name']}): {strong:.0%} de las notas en «{sz['label']}», "
                                     f"su registro más potente ({sz['character']}).",
                                tracks=[t["idx"]], rule=None))

    # ── respiración de vientos ──
    for t in live:
        prof = t["prof"]
        if not prof.get("breath") or prof.get("max_breath", 0) <= 0:
            continue
        ns = sorted((n for n in t["notes"] if lo_bar <= sc.bar_of(n.start) <= hi_bar),
                    key=lambda n: n.start)
        if not ns:
            continue
        longest, lb = 0.0, set()
        cs, ce = ns[0].start, ns[0].end
        bar_set = {sc.bar_of(ns[0].start)}
        for n in ns[1:] + [None]:
            if n is not None and tc.sec(n.start) - tc.sec(ce) < 0.35:
                ce = max(ce, n.end)
                bar_set.add(sc.bar_of(n.start))
                continue
            dur = tc.sec(ce) - tc.sec(cs)
            if dur > prof["max_breath"] and dur > longest:
                longest, lb = dur, set(bar_set)
            if n is not None:
                cs, ce, bar_set = n.start, n.end, {sc.bar_of(n.start)}
        if longest:
            others = _alts("melody", sum(n.pitch for n in ns) / len(ns), {t["key"]}, top=2)
            findings.append(_mk(
                "no_breathing", "aviso", lb, [t["idx"]],
                f"{t['es']} ({t['name']}): frase continua de {longest:.0f} s sin respirar "
                f"(límite ≈ {prof['max_breath']} s).",
                "Un instrumentista de viento necesita respirar; un mockup que no lo respeta suena irreal [breathing].",
                "Deja silencios de respiración o reparte la frase con otro instrumento"
                + (f" ({', '.join(a['es'] for a in others)})." if others else "."),
                rule="breathing", atlas_key=t["key"] if t["key"] in ATLAS else None))

    # ── dobles entre pistas (unísono / octava) ──
    pair_bars: Dict[Tuple[int, int], Dict[str, set]] = {}
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            for bar in blist:
                oa, ob = sc.ons[a["idx"]].get(bar, []), sc.ons[b["idx"]].get(bar, [])
                if not oa or not ob:
                    continue
                small, other_t = (oa, sc.ovl[b["idx"]].get(bar, [])) if len(oa) <= len(ob) \
                    else (ob, sc.ovl[a["idx"]].get(bar, []))
                uni = octv = 0
                for n in small:
                    for m in other_t:
                        if m.start <= n.start < m.end:
                            if m.pitch == n.pitch:
                                uni += 1
                                break
                            if (m.pitch - n.pitch) % 12 == 0:
                                octv += 1
                                break
                if (uni + octv) / len(small) >= 0.6:
                    d = pair_bars.setdefault((a["idx"], b["idx"]), dict(uni=set(), oct=set()))
                    (d["uni"] if uni >= octv else d["oct"]).add(bar)

    def _in_combo(t, c):
        k = t["key"]
        if k in c["insts"]:
            return True
        return t["family"] == "cuerda_frotada" and "cuerdas" in c["insts"] and k in (
            "violin", "viola", "violonchelo", "contrabajo", "cuerdas")

    doubled_with_mel: Dict[int, set] = {}
    for (ia, ib), d in pair_bars.items():
        a, b = by_idx[ia], by_idx[ib]
        bs = d["uni"] | d["oct"]
        if mel_idx in (ia, ib):
            other = ib if ia == mel_idx else ia
            doubled_with_mel.setdefault(other, set()).update(bs)
        rel = "unísono" if len(d["uni"]) >= len(d["oct"]) else "octava"
        match = [c for c in find_combos() if _in_combo(a, c) and _in_combo(b, c)
                 and ("octava" in c["relation"] or "unísono" in c["relation"] or "capas" in c["relation"])]
        if match and len(bs) >= 1:
            c = match[0]
            elogios.append(dict(text=f"Doblaje recomendado {a['es']} + {b['es']} en {rel} ({_ranges(bs)}): "
                                     f"«{c['name']}» — {c['effect']}", tracks=[ia, ib], rule="octave_doubling"))
        elif len(bs) >= 4 and a["family"] != b["family"]:
            findings.append(_mk(
                "double_uncatalogued", "info", bs, [ia, ib],
                f"{a['es']} y {b['es']} se doblan en {rel} ({_ranges(bs)}) y no es una combinación catalogada.",
                "Doblar a la octava refuerza sin turbiar; al unísono los timbres se funden en uno nuevo [octave_doubling]. "
                "Comprueba el equilibrio por oído: el instrumento más potente domina.",
                "Si buscas esa mezcla, valora las combinaciones del atlas: python orchestral_atlas.py combos --with " + a["key"],
                rule="octave_doubling"))

    # ── melodía enterrada / enmascaramiento ──
    dens, mel_ratio, rows = [], {}, []
    buried_bars, buried_err, mask_bars = set(), set(), {}
    mel_ok = mel_n = 0
    mel_t = by_idx.get(mel_idx) if mel_idx is not None else None
    worst = (9.0, None, None, None)
    for bar in blist:
        act = [t for t in live if sc.ovl[t["idx"]].get(bar)]
        dens.append(len(act))
        onsets = sum(len(sc.ons[t["idx"]].get(bar, [])) for t in act)
        row = dict(bar=bar, instruments=len(act), onsets=onsets, melody_proj=None, acc_proj=None, ratio=None)
        if mel_t is not None and sc.ovl[mel_t["idx"]].get(bar):
            mp = [projection(mel_t["key"], n.pitch, n.vel) for n in sc.ovl[mel_t["idx"]][bar]]
            mel_proj = sum(mp) / len(mp)
            partners = [t for t in act if t["idx"] != mel_t["idx"] and bar in doubled_with_mel.get(t["idx"], set())]
            for p in partners:
                pp = [projection(p["key"], n.pitch, n.vel) for n in sc.ovl[p["idx"]][bar]]
                mel_proj += 0.5 * sum(pp) / len(pp)
            mel_proj = min(1.0, mel_proj)
            comp = []
            for t in act:
                if t["idx"] == mel_t["idx"] or t in partners:
                    continue
                pp = [projection(t["key"], n.pitch, n.vel) for n in sc.ovl[t["idx"]][bar]]
                comp.append((sum(pp) / len(pp), t))
            row["melody_proj"] = round(mel_proj, 3)
            if comp:
                acc_max, acc_t = max(comp, key=lambda x: x[0])
                ratio = mel_proj / acc_max if acc_max > 0 else 9.0
                row["acc_proj"], row["ratio"] = round(acc_max, 3), round(ratio, 2)
                mel_n += 1
                if ratio < 0.7:
                    buried_bars.add(bar)
                    if ratio < 0.4:
                        buried_err.add(bar)
                else:
                    mel_ok += 1
                if ratio < worst[0]:
                    worst = (ratio, acc_t, mel_proj, acc_max)
                # enmascaramiento: mismo timbre y registro cercano, sin doblar
                for ap, t in comp:
                    if t["family"] != mel_t["family"] or ap < mel_proj:
                        continue
                    close = tot = 0
                    for n in sc.ons[mel_t["idx"]].get(bar, []):
                        near = [abs(m.pitch - n.pitch) for m in sc.ovl[t["idx"]].get(bar, [])
                                if m.start <= n.start < m.end and m.pitch != n.pitch]
                        if near:
                            tot += 1
                            if min(near) < 5:
                                close += 1
                    if tot and close / tot >= 0.5:
                        mask_bars.setdefault(t["idx"], set()).add(bar)
            else:
                mel_n += 1
                mel_ok += 1
        rows.append(row)

    if mel_t is not None and buried_bars:
        ratio, acc_t, mp, ap = worst
        mean_pitch = sum(n.pitch for n in mel_t["notes"]) / len(mel_t["notes"])
        dyn = "loud" if (sum(n.vel for n in mel_t["notes"]) / len(mel_t["notes"])) >= 96 else \
              ("soft" if (sum(n.vel for n in mel_t["notes"]) / len(mel_t["notes"])) < 64 else "mid")
        cs = find_combos(role="melody", register=register_of(int(round(mean_pitch))), dynamic=dyn)[:2]
        sev = "error" if len(buried_err) >= max(2, len(buried_bars) // 2) else "aviso"
        sug = [f"Sube la dinámica de la melodía o baja la de {acc_t['es']} ({acc_t['name']})"]
        if cs:
            sug.append("dobla la melodía con una combinación del atlas: " + "; ".join(c["name"] for c in cs))
        sug.append("o ponla en una zona más potente de su instrumento")
        findings.append(_mk(
            "melody_buried", sev, buried_bars, [mel_t["idx"], acc_t["idx"]],
            f"La melodía ({mel_t['es']}, pista {mel_t['idx']}) queda enterrada: proyección estimada "
            f"{mp:.2f} frente a {ap:.2f} de {acc_t['es']} en el peor compás (relación {ratio:.2f}).",
            "El oído sigue la línea más potente y más aguda o de timbre distinto; si el acompañamiento proyecta más "
            "que la melodía, esta se pierde [melody_register]"
            + (" y, a igual dinámica, los metales cubren a cuerdas y maderas [brass_vs_strings]"
               if acc_t["family"] == "viento_metal" else "") + ".",
            "; ".join(sug) + ".", rule="melody_register",
            atlas_key=mel_t["key"] if mel_t["key"] in ATLAS else None))
    elif mel_t is not None and mel_n and mel_ok / mel_n >= 0.9 and len(live) > 1:
        elogios.append(dict(text=f"La melodía ({mel_t['es']}, pista {mel_t['idx']}) se oye con claridad en "
                                 f"{mel_ok} de {mel_n} compases: proyecta al menos tanto como el acompañamiento.",
                            tracks=[mel_t["idx"]], rule="melody_register"))
    for ti, bs in mask_bars.items():
        t = by_idx[ti]
        findings.append(_mk(
            "masking_same_register", "aviso", bs, [mel_t["idx"], ti],
            f"{mel_t['es']} (melodía) y {t['es']} ({t['name']}) son del mismo timbre y suenan a menos de 5 semitonos.",
            "Si dos líneas del mismo timbre se acercan tanto, los armónicos se solapan y el oído no las separa "
            "[masking_same_register].",
            f"Separa ambas líneas al menos una 5ª, o cambia el acompañamiento a otra familia"
            f" (python orchestral_atlas.py role countermelody --pitch {pitch_name(int(round(sum(n.pitch for n in t['notes']) / len(t['notes']))))}).",
            rule="masking_same_register", atlas_key=mel_t["key"] if mel_t["key"] in ATLAS else None))

    # ── armonía grave y espaciado por muestras ──
    low_pairs: Dict[Tuple[int, int], set] = {}
    gap_bars, gap_max = set(), 0
    nobass_bars = set()
    for k, items in sorted(sc.samples.items()):
        bar = sc.bar_of(k * sc.step)
        voices = {}
        for ti, p, _v in items:
            if by_idx[ti]["perc"]:
                continue
            voices.setdefault(p, set()).add(ti)
        ps = sorted(voices)
        for lo_p, hi_p in zip(ps, ps[1:]):
            if lo_p < LOW_LIMIT and hi_p - lo_p in (1, 2, 3, 4):
                for ta in voices[lo_p]:
                    for tb in voices[hi_p]:
                        if ta != tb:
                            low_pairs.setdefault(tuple(sorted((ta, tb))), set()).add(bar)
        if len(ps) >= 5:
            up = ps[2:]
            for lo_p, hi_p in zip(up, up[1:]):
                if hi_p - lo_p > 12:
                    gap_bars.add(bar)
                    gap_max = max(gap_max, hi_p - lo_p)
    for bar in blist:
        act = [t for t in live if sc.ovl[t["idx"]].get(bar)]
        if len(act) >= 3 and min(n.pitch for t in act for n in sc.ovl[t["idx"]][bar]) >= LOW_LIMIT:
            nobass_bars.add(bar)
    for (ia, ib), bs in sorted(low_pairs.items(), key=lambda kv: min(kv[1])):
        a, b = by_idx[ia], by_idx[ib]
        findings.append(_mk(
            "low_chords", "aviso", bs, [ia, ib],
            f"{a['es']} y {b['es']} forman segundas/terceras por debajo de C3 (MIDI {LOW_LIMIT}).",
            "En el grave los armónicos de dos notas próximas se solapan y producen batidos: el acorde suena turbio "
            "[low_chords].",
            "Abre el espaciado: quinta u octava abajo y terceras arriba. Un instrumento por línea de bajo "
            "[bass_clarity].", rule="low_chords"))
    if low_pairs == {} and any(p < LOW_LIMIT for items in sc.samples.values() for _t, p, _v in items):
        elogios.append(dict(text="Los graves están bien espaciados: no hay terceras ni segundas por debajo de C3.",
                            tracks=[], rule="low_chords"))
    if gap_bars:
        findings.append(_mk(
            "spacing", "aviso" if gap_max > 19 else "info", gap_bars, [],
            f"Hueco de hasta {gap_max} semitonos entre voces adyacentes superiores.",
            "Entre las voces superiores conviene no pasar de una octava; el hueco rompe la cohesión del acorde "
            "[spacing].",
            "Rellena el hueco con una voz intermedia (violas, trompas, maderas) o acerca las voces extremas.",
            rule="spacing"))
    if nobass_bars:
        findings.append(_mk(
            "no_bass", "aviso", nobass_bars, [],
            "Compases con varios instrumentos y ninguno por debajo de C3: falta el bajo.",
            "Sin una línea grave la armonía queda sin cimiento y el conjunto suena hueco [bass_clarity].",
            "Añade violonchelos y contrabajos (python orchestral_atlas.py role bass --register low).",
            rule="bass_clarity"))
    elif any(any(n.pitch < LOW_LIMIT for n in sc.ovl[t['idx']].get(b, [])) for t in live for b in blist):
        elogios.append(dict(text="Hay una línea de bajo presente: los compases con varios instrumentos tienen cimiento.",
                            tracks=[], rule="bass_clarity"))

    # ── densidad ──
    if len(blist) >= 8 and dens and max(dens) - min(dens) <= 1 and max(dens) >= 3:
        findings.append(_mk(
            "density_flat", "info", blist, [],
            f"La densidad es plana: siempre {max(dens)} instrumentos.",
            "Sin entradas ni salidas la orquestación no respira ni crece; menos instrumentos suenan más claros "
            "[density].", "Haz entrar y salir capas (p. ej. empezar con cuerdas y sumar maderas y metales).",
            rule="density"))
    elif dens and max(dens) - min(dens) >= 2:
        elogios.append(dict(text=f"La densidad varía de {min(dens)} a {max(dens)} instrumentos: hay entradas y "
                                 "salidas de capas que dan forma a la textura.", tracks=[], rule="density"))

    # ── filtrar por pista ──
    if track is not None:
        findings = [f for f in findings if track in f["tracks"]]
        elogios = [e for e in elogios if track in e["tracks"]]
        tinfo = [t for t in tinfo if t["idx"] == track]
    findings.sort(key=lambda f: (_SEV_ORDER[f["severity"]], f["bars"][0] if f["bars"] else 0))

    n_b = max(1, len(blist))
    pen = sum(_SEV_W[f["severity"]] * (0.4 + 0.6 * (len(f["bars"]) / n_b if f["bars"] else 1.0))
              for f in findings)
    overall = round(max(0.0, min(1.0, 1.0 - pen)), 2)
    cnt = {s: sum(1 for f in findings if f["severity"] == s) for s in _SEV_ORDER}
    return {
        "tool": TOOL, "version": VERSION, "file": path, "overall": overall,
        "summary": {"tracks": len(tracks), "bars": len(blist), "bar_from": lo_bar, "bar_to": hi_bar,
                    "errors": cnt["error"], "warnings": cnt["aviso"], "infos": cnt["info"],
                    "praise": len(elogios), "melody_track": mel_idx if mel_idx is not None else -1,
                    "melody_clear_pct": round(100.0 * mel_ok / mel_n, 1) if mel_n else None,
                    "density_mean": round(sum(dens) / len(dens), 2) if dens else 0.0,
                    "role_source": role_src},
        "tracks": tinfo, "findings": findings, "elogios": elogios,
        "density": dens, "bars_table": rows,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def print_report(res: dict, verbose: bool = False):
    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    s = res["summary"]
    print(f"\n{'═' * 78}")
    print(f"  {B}ORCHESTRA CRITIC v{VERSION}  ·  {res['file']}{R}")
    print(f"  compases {s['bar_from']}–{s['bar_to']} · {s['tracks']} pistas con notas · "
          f"roles: {s['role_source']}")
    print(f"{'═' * 78}")
    print(f"\n  {B}Instrumentación{R}")
    for t in res["tracks"]:
        proj = f"proy {t['proj_mean']:.2f}" if t.get("proj_mean") is not None else "proy  — "
        print(f"  [{t['idx']:>2}] {t['name'][:18]:<18} → {t['es'][:22]:<22} {t['n_notes']:>4} notas  "
              f"{t['bars']:<10} {proj}")
        if t.get("comment"):
            print(_wrap(t["comment"], "        "))
    if res["density"]:
        print(f"\n  densidad (instrumentos por compás): {sparkline(res['density'])}  "
              f"[{min(res['density'])}–{max(res['density'])}]")
    if verbose and res["bars_table"]:
        print(f"\n  {B}Por compás{R}   comp  instr  notas  mel   acomp  relación")
        for r in res["bars_table"]:
            mp = f"{r['melody_proj']:.2f}" if r["melody_proj"] is not None else " —  "
            ap = f"{r['acc_proj']:.2f}" if r["acc_proj"] is not None else " —  "
            rt = f"{r['ratio']:.2f}" if r["ratio"] is not None else " —  "
            flag = " ⚑" if r["ratio"] is not None and r["ratio"] < 0.7 else ""
            print(f"              {r['bar']:>4}  {r['instruments']:>4}  {r['onsets']:>5}  {mp}  {ap}  {rt}{flag}")

    if res["elogios"]:
        print(f"\n  {_c('GRN')}Lo que funciona:{R}")
        for e in res["elogios"]:
            print(_wrap("✓ " + e["text"], "    "))
    if res["findings"]:
        print(f"\n  {_c('YEL')}Para mejorar:{R}")
        for f in res["findings"]:
            col = {"error": _c("RED"), "aviso": _c("YEL"), "info": _c("CYA")}[f["severity"]]
            where = f"  ({f['bars_text']})" if f["bars_text"] else ""
            print(_wrap(f"{col}{_SEV_MARK[f['severity']]} [{f['severity']}]{R} {f['message']}{where}", "    "))
            print(_wrap("¿Por qué? " + f["why"], "        "))
            print(_wrap("→ " + f["suggestion"], "        "))
            if f.get("more"):
                print(f"        {G}más información: {f['more']}{R}")
    else:
        print(f"\n  {G}(sin sugerencias: la orquestación está muy bien equilibrada){R}")
    print(f"\n  valoración global: {C}{res['overall']:.2f}{R}   "
          f"({s['errors']} errores · {s['warnings']} avisos · {s['infos']} info)")
    print(f"  {G}reglas y fichas: python orchestral_atlas.py rules · python orchestral_atlas.py show INSTRUMENTO{R}")
    print(f"{'═' * 78}\n")


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
        ev += [(0, 0, b"\xFF\x51\x03" + (600000).to_bytes(3, "big")), (0, 0, b"\xFF\x58\x04\x04\x02\x18\x08")]
    if prog is not None:
        ev.append((0, 1, bytes([0xC0 | ch, prog])))
    for st, dur, p, v in notes:
        ev += [(st, 3, bytes([0x90 | ch, p, v])), (st + dur - 2, 2, bytes([0x80 | ch, p, 0]))]
    ev.sort(key=lambda e: (e[0], e[1]))
    body, last = b"", 0
    for t, _o, b in ev:
        body += _vlq(t - last) + b
        last = t
    return b"MTrk" + (len(body) + 4).to_bytes(4, "big") + body + _vlq(0) + b"\xFF\x2F\x00"


def _synthetic(path, bad: bool):
    tpb, bar = 480, 1920
    chords = [(48, 52, 55), (45, 48, 52), (41, 45, 48), (43, 47, 50)]
    mel = [[72, 74, 76, 79], [76, 74, 72, 69], [69, 72, 77, 76], [74, 76, 79, 74]]
    nb = 8

    def line(shift, vel, b0=0, b1=nb):
        return [(b * bar + i * tpb, tpb, p + shift, vel) for b in range(b0, b1) for i, p in enumerate(mel[b % 4])]

    def pad(idx, octv, vel):
        return [(b * bar, bar, chords[b % 4][idx] + 12 * octv, vel) for b in range(nb)]

    tr = [_trk("Conductor", None, 0, [], tempo=True)]
    if bad:     # melodía de flauta grave bajo cuerdas fuertes; fagot en terceras graves sobre el chelo
        tr += [_trk("Flauta", 73, 0, line(-9, 80)), _trk("Violin I", 40, 1, pad(2, 1, 100)),
               _trk("Violin II", 40, 2, pad(1, 1, 100)), _trk("Viola", 41, 3, pad(2, 0, 96)),
               _trk("Violonchelo", 42, 4, pad(0, 0, 90)),
               _trk("Fagot", 70, 5, [(s, d, p + 4, 70) for s, d, p, v in pad(0, 0, 90)])]
    else:       # melodía en violines, flauta a la octava, cuerdas suaves, bajo claro
        tr += [_trk("Violin I", 40, 0, line(0, 88)), _trk("Flauta", 73, 1, line(12, 80, 4)),
               _trk("Violin II", 40, 2, pad(2, 1, 70)), _trk("Viola", 41, 3, pad(1, 1, 68)),
               _trk("Violonchelo", 42, 4, pad(0, 0, 76)), _trk("Contrabajo", 43, 5, pad(0, -1, 76))]
    hdr = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tr).to_bytes(2, "big") + tpb.to_bytes(2, "big")
    _ensure_parent(path)
    Path(path).write_bytes(hdr + b"".join(tr))


def cmd_selftest(a):
    errs = []
    with tempfile.TemporaryDirectory() as d:
        bad, good = os.path.join(d, "bad.mid"), os.path.join(d, "good.mid")
        _synthetic(bad, True)
        _synthetic(good, False)
        rb = critique_orchestration(bad)
        rg = critique_orchestration(good)
        ids = {f["id"] for f in rb["findings"]}
        if "melody_buried" not in ids:
            errs.append("no detecta la melodía enterrada del MIDI defectuoso")
        if "register_weak" not in ids:
            errs.append("no detecta el registro débil de la flauta")
        if "low_chords" not in ids:
            errs.append("no detecta las terceras graves")
        if not rb["overall"] < rg["overall"]:
            errs.append(f"el MIDI defectuoso ({rb['overall']}) debería puntuar peor que el bueno ({rg['overall']})")
        if not rg["elogios"]:
            errs.append("el MIDI bueno no recibe elogios")
        if critique_orchestration(bad, bars=(2, 3))["summary"]["bars"] != 2:
            errs.append("--bars no limita el análisis")
        try:
            critique_orchestration(bad, bars=(99, 100))
            errs.append("--bars fuera de la obra debería fallar")
        except ValueError:
            pass
        if _parse_bars("5-8") != (5, 8) or _ranges({2, 3, 4, 7}) != "c.2–4, 7":
            errs.append("parse/agrupación de compases incorrectos")
    for e in errs:
        print(f"  ✗ {e}")
    print(f"  {'OK' if not errs else 'FALLO'} · selftest de {TOOL}")
    return 0 if not errs else 3


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def cmd_critique(a):
    bars = _parse_bars(a.bars)
    ov = parse_instrument_overrides(a.instrument)
    res = critique_orchestration(a.midi, bars=bars, track=a.track, overrides=ov)
    if a.json == "-":
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print_report(res, verbose=a.verbose)
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
        prog="orchestra_critic.py",
        description="Crítica didáctica de una orquestación (por instrumento y compás).")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("critique", parents=[common], help="Crítica de una orquestación MIDI")
    p.add_argument("midi")
    p.add_argument("--bars", help="Rango de compases A-B (1-based)")
    p.add_argument("--track", type=int, help="Solo hallazgos de esta pista (índice 0-based)")
    p.add_argument("--instrument", nargs="+", action="extend", metavar="N=CLAVE", help="Forzar instrumento de una pista")
    p.add_argument("--strict", action="store_true", help="rc=3 si hay avisos o errores")
    p.add_argument("--verbose", action="store_true", help="Tabla por compás")
    p.add_argument("--json", help="Sidecar JSON ('-' = stdout)")
    p.set_defaults(fn=cmd_critique)

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
