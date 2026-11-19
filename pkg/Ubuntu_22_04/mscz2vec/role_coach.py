#!/usr/bin/env python3
"""
role_coach.py — Role Coach: comprueba que TU plan de roles se cumple en el MIDI
(lista de comprobación del hito «roles claros»)
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
role_detector DECIDE qué papel tiene cada pista; esta herramienta hace lo
contrario: tú declaras en un plan quién hace de melodía, armonía, bajo, ritmo y
textura en cada tramo de compases, y role_coach MIDE el MIDI y te dice si lo
declarado se cumple, como un profesor que revisa el ejercicio de orquestación:

  · ¿el instrumento declarado suena de verdad en esos compases?
  · ¿su papel medido es coherente (melodía = línea más prominente; bajo = lo
    más grave; armonía = acordes sostenidos; ritmo = ataques regulares)?
  · ¿queda algún rol sin cubrir (compases sin bajo, sin armonía, sin melodía)?
  · ¿hay roles duplicados (dos pistas haciendo lo mismo), la misma pista de
    melodía y bajo a la vez, o instrumentos que suenan y no están en el plan?
  · ¿el instrumento está en una zona de registro débil para ese rol
    (orchestral_atlas)? Si es así propone alternativas.

Es la lista de comprobación del hito «demostrar roles claros de melodía,
armonía, bajo, ritmo y textura» de las semanas 4–6 del plan.

FORMATO DEL PLAN (texto; un tramo por línea; «#» abre un comentario):

    1-8   melody=oboe  harmony=cuerdas,trompa  bass=fagot  rhythm=timbal
    9-16  melody=violin  countermelody=flauta  harmony=cuerdas  bass=violonchelo
    5     texture=arpa                    # un solo compás

  Roles: melody · countermelody · harmony · bass · rhythm · texture (también
  en español: melodia, contramelodia, armonia, bajo, ritmo, textura).
  Valores: claves o nombres de instrumento (oboe, violonchelo, «corno inglés»),
  nombres de pista del MIDI, o grupos (cuerdas, maderas, metales, percusion).
  Equivalente en JSON:  {"sections":[{"bars":"1-8","roles":{"melody":["oboe"],
  "bass":"fagot"}}]}

CUMPLIMIENTO: en cada tramo se esperan los roles declarados más los tres
básicos (melodía, armonía, bajo). Un rol está cumplido (✓) si está declarado,
suena (≥75 % de los compases en los básicos) y es coherente con lo medido; no
cumplido (✗) en caso contrario; «·» = opcional no declarado (contramelodía,
ritmo, textura). overall = roles cumplidos / roles esperados. Los avisos de
calidad (registro débil, duplicados…) no cambian el estado ✓/✗.

COMO MÓDULO:
  from role_coach import parse_plan, check_plan, template_plan

DEPENDENCIAS: playability_auditor.py (E/S MIDI y catálogo de instrumentos) ·
orchestral_atlas.py (registros y roles) · role_detector.py (opcional, por
subproceso; si falla se usa una heurística interna)

LIMITACIONES: heurístico y sobre MIDI: «la línea más prominente» se estima por
registro, actividad y continuidad, no por el timbre real de tus librerías. Un
grupo (cuerdas…) se valida pista a pista solo en cobertura y coherencia
conjunta. No distingue un ostinato de un patrón repetitivo cualquiera. Los
compases se numeran desde 1 según el mapa de compás del MIDI.

CÓDIGOS DE SALIDA:  0 ok · 1 error de ejecución · 2 uso incorrecto (MIDI o plan
inexistente, plan mal formado) · 3 selftest fallido o, con --strict, errores o
avisos en el plan

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  template  Genera un plan editable a partir de lo que se detecta en el MIDI
  check     Comprueba el plan contra el MIDI (tabla ✓/✗/· y explicaciones)
  selftest  Verifica la herramienta con MIDIs y planes sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  template  midi [--out FILE]
  check     midi --plan FILE [--strict] [--json FILE|-] [--no-color]
  selftest

OPCIONES COMUNES:
  --no-color    Desactivar colores ANSI

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
[template]
midi  file  io=in ext=mid required  # MIDI orquestado
--out file  io=out ext=txt  # Escribe el plan (por defecto, a stdout)
[check]
midi     file  io=in ext=mid required  # MIDI orquestado
--plan   file  io=in ext=txt,json required  # Plan de roles (texto o JSON)
--strict bool  default=false  # rc=3 si hay errores o avisos
--json   file  io=out ext=json  # Sidecar JSON ('-' = stdout)
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python role_coach.py template obra.mid --out plan.txt
  python role_coach.py check obra.mid --plan plan.txt
  python role_coach.py check obra.mid --plan plan.txt --json roles.json --strict
  python role_coach.py selftest
"""

import re
import sys
import json
import argparse
import subprocess
import tempfile
import shutil
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from playability_auditor import (read_midi, write_midi, TimeContext, extract_notes,
                                     resolve_instrument, INSTRUMENT_DB, pitch_name,
                                     MidiData, MidiTrackData, MidiEvent)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)
try:
    from orchestral_atlas import (ATLAS, RULES, ROLES, ROLE_ES, zone_of, best_for_role,
                                  instrument_key)
except ImportError:
    print("ERROR: requiere orchestral_atlas.py en el mismo directorio (registros y roles)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()

ROLE_ORDER = ["melody", "countermelody", "harmony", "bass", "rhythm", "texture"]
CORE_ROLES = ["melody", "harmony", "bass"]
_ROLE_ALIASES = {
    "melody": "melody", "melodia": "melody", "melodía": "melody",
    "countermelody": "countermelody", "contramelodia": "countermelody",
    "contramelodía": "countermelody",
    "harmony": "harmony", "armonia": "harmony", "armonía": "harmony",
    "bass": "bass", "bajo": "bass",
    "rhythm": "rhythm", "ritmo": "rhythm",
    "texture": "texture", "textura": "texture", "color": "texture",
}
# grupo → (familias de INSTRUMENT_DB, clave del atlas para el grupo)
_GROUPS = {
    "cuerdas": (("cuerda_frotada",), "cuerdas"), "strings": (("cuerda_frotada",), "cuerdas"),
    "metales": (("viento_metal",), "viento_metal_gen"), "brass": (("viento_metal",), "viento_metal_gen"),
    "maderas": (("viento_madera",), None), "woodwinds": (("viento_madera",), None),
    "percusion": (("percusion", "percusion_afinada"), None),
}
_UNPITCHED = {"percusion"}


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn").strip()


def _rule_ok(rid: str) -> bool:
    return any(r["id"] == rid for r in RULES)


def _rt(rid: Optional[str]) -> str:
    return f" [{rid}]" if rid and _rule_ok(rid) else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN
# ══════════════════════════════════════════════════════════════════════════════

def _parse_bars(s: str, where: str) -> Tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*(?:[-–]\s*(\d+))?\s*", str(s))
    if not m:
        raise ValueError(f"{where}: compases no válidos «{s}» (usa 1-8 o un número)")
    a = int(m.group(1))
    b = int(m.group(2)) if m.group(2) else a
    if a < 1 or b < a:
        raise ValueError(f"{where}: rango de compases incoherente «{s}»")
    return a, b


def _norm_role(r: str, where: str) -> str:
    k = _ROLE_ALIASES.get(_fold(r)) or _ROLE_ALIASES.get(str(r).lower())
    if not k:
        raise ValueError(f"{where}: rol desconocido «{r}» (válidos: "
                         f"{', '.join(ROLE_ORDER)})")
    return k


def parse_plan(text: str) -> List[dict]:
    """Texto o JSON → [{'bars': (a, b), 'roles': {rol: [nombres]}, 'line': n}]."""
    if not text or not text.strip():
        raise ValueError("el plan está vacío")
    secs: List[dict] = []
    if text.lstrip()[0] in "{[":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"plan JSON mal formado: {e}")
        items = data.get("sections") if isinstance(data, dict) else data
        if not isinstance(items, list) or not items:
            raise ValueError('el plan JSON debe ser {"sections": [...]}')
        for i, it in enumerate(items, 1):
            where = f"sección {i}"
            if not isinstance(it, dict) or "bars" not in it or "roles" not in it:
                raise ValueError(f"{where}: faltan «bars» y/o «roles»")
            roles: Dict[str, List[str]] = {}
            for r, v in it["roles"].items():
                names = v if isinstance(v, list) else str(v).split(",")
                names = [str(x).strip() for x in names if str(x).strip()]
                if not names:
                    raise ValueError(f"{where}: el rol «{r}» no tiene instrumentos")
                roles.setdefault(_norm_role(r, where), []).extend(names)
            if not roles:
                raise ValueError(f"{where}: tramo sin roles")
            secs.append(dict(bars=_parse_bars(it["bars"], where), roles=roles, line=i))
    else:
        for n, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            where = f"línea {n}"
            m = re.match(r"^(\d+(?:\s*[-–]\s*\d+)?)\s+(.*)$", line)
            if not m:
                raise ValueError(f"{where}: se esperaba «A-B rol=instrumento …» y hay «{line}»")
            bars = _parse_bars(m.group(1), where)
            rest = m.group(2)
            pos = list(re.finditer(r"(?<![\w=])([A-Za-zÁÉÍÓÚáéíóúñÑ]+)=", rest))
            if not pos:
                raise ValueError(f"{where}: tramo sin roles (usa melody=oboe …)")
            if rest[:pos[0].start()].strip():
                raise ValueError(f"{where}: texto inesperado «{rest[:pos[0].start()].strip()}»")
            roles = {}
            for i, mt in enumerate(pos):
                end = pos[i + 1].start() if i + 1 < len(pos) else len(rest)
                names = [x.strip() for x in rest[mt.end():end].split(",") if x.strip()]
                if not names:
                    raise ValueError(f"{where}: el rol «{mt.group(1)}» no tiene instrumentos")
                roles.setdefault(_norm_role(mt.group(1), where), []).extend(names)
            secs.append(dict(bars=bars, roles=roles, line=n))
    secs.sort(key=lambda s: s["bars"])
    return secs


def _fmt_bars(a, b):
    return f"{a}" if a == b else f"{a}–{b}"


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DEL MIDI Y RASGOS
# ══════════════════════════════════════════════════════════════════════════════

class N:
    __slots__ = ("start", "end", "pitch", "vel", "b0", "b1")

    def __init__(self, start, end, pitch, vel, b0, b1):
        self.start, self.end, self.pitch, self.vel, self.b0, self.b1 = start, end, pitch, vel, b0, b1


def load_tracks(path: str):
    """→ (tracks, total_bars, tpb). Solo pistas con notas."""
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    mid = read_midi(path)
    tc = TimeContext(mid)
    min_ks = max(4, mid.tpb // 32)
    tracks = []
    for i, trk in enumerate(mid.tracks):
        raw = [n for n in extract_notes(trk)
               if not (n.pitch < 28 and (n.end - n.start) <= min_ks)]      # sin keyswitches
        if not raw:
            continue
        key, _ = resolve_instrument(trk, raw)
        fam = INSTRUMENT_DB.get(key, {}).get("family", "")
        drum = 9 in trk.channels
        notes = [N(n.start, n.end, n.pitch, n.vel, tc.bar(n.start), tc.bar(max(n.start, n.end - 1)))
                 for n in raw]
        tracks.append(dict(index=i, name=trk.name or f"Pista {i}", key=key, family=fam,
                           unpitched=bool(drum or key in _UNPITCHED), notes=notes))
    if not tracks:
        raise ValueError("el MIDI no contiene notas")
    total = max(n.b1 for t in tracks for n in t["notes"])
    return tracks, total, mid.tpb


def _feats(t, a, b, tpb):
    """Rasgos de la pista en los compases a–b (None si no suena)."""
    snd = [n for n in t["notes"] if n.b1 >= a and n.b0 <= b]
    if not snd:
        return None
    att = sorted((n for n in snd if a <= n.b0 <= b), key=lambda n: n.start) or sorted(snd, key=lambda n: n.start)
    cov_bars = set()
    for n in snd:
        cov_bars.update(range(max(a, n.b0), min(b, n.b1) + 1))
    # linealidad: onsets sin otra nota sonando
    mono, prev_end, groups = 0, -1, {}
    for n in att:
        groups.setdefault(n.start, []).append(n)
    for st in sorted(groups):
        g = groups[st]
        if len(g) == 1 and prev_end <= st + 2:
            mono += 1
        prev_end = max(prev_end, max(x.end for x in g))
    lin = mono / len(groups)
    ons = sorted(groups)
    iois = [ons[i + 1] - ons[i] for i in range(len(ons) - 1)]
    share = 0.0
    if iois:
        from collections import Counter
        share = Counter(iois).most_common(1)[0][1] / len(iois)
    pitched = [n for n in att if True]
    mp = sum(n.pitch for n in pitched) / len(pitched)
    weak = 0
    if t["key"] in INSTRUMENT_DB and not t["unpitched"]:
        weak = sum(1 for n in pitched if zone_of(t["key"], n.pitch)["power"] <= 1) / len(pitched)
    return dict(n_att=len(att), n_onsets=len(ons), mean_pitch=mp,
                mean_vel=sum(n.vel for n in pitched) / len(pitched),
                coverage=len(cov_bars) / max(1, b - a + 1), cov_bars=cov_bars,
                linearity=lin, chordiness=1 - lin,
                mean_dur_beats=sum((n.end - n.start) for n in att) / len(att) / tpb,
                ioi_share=share, weak_frac=weak)


def _melody_score(f, lo, hi, maxact):
    reg = (f["mean_pitch"] - lo) / (hi - lo) if hi > lo else 0.5
    act = f["n_att"] / max(1, maxact)
    s = 0.4 * reg + 0.3 * act + 0.2 * f["linearity"] + 0.1 * f["coverage"]
    if f["mean_dur_beats"] >= 3:
        s -= 0.15                                   # notas largas: más pad que melodía
    return s


def measure_roles(items):
    """items: [(pista, rasgos)] → {rol: [pistas]} medido sin plan (para template y sugerencias)."""
    out = {r: [] for r in ROLE_ORDER}
    pitched = []
    for t, f in items:
        if t["unpitched"]:
            out["rhythm"].append(t)
        elif t["family"] == "percusion_afinada":
            (out["rhythm"] if f["ioi_share"] >= 0.6 and f["n_onsets"] >= 3 else out["texture"]).append(t)
        else:
            pitched.append((t, f))
    rest = list(pitched)
    if rest:
        low = min(rest, key=lambda x: x[1]["mean_pitch"])
        if low[1]["mean_pitch"] <= 55:
            bass = [x for x in rest if x is low or (x[1]["mean_pitch"] <= low[1]["mean_pitch"] + 12
                                                    and x[1]["mean_pitch"] <= 55 and x[1]["linearity"] >= 0.6)]
            out["bass"] = [x[0] for x in bass]
            rest = [x for x in rest if x not in bass]
    if rest:
        lo = min(x[1]["mean_pitch"] for x in pitched)
        hi = max(x[1]["mean_pitch"] for x in pitched)
        mx = max(x[1]["n_att"] for x in pitched)
        best = max(rest, key=lambda x: _melody_score(x[1], lo, hi, mx))
        out["melody"] = [best[0]]
        rest = [x for x in rest if x is not best]
        for t, f in rest:
            if f["mean_dur_beats"] >= 3 or f["chordiness"] >= 0.3:
                out["harmony"].append(t)
            elif f["linearity"] >= 0.7 and f["n_att"] >= 0.3 * mx:
                out["countermelody"].append(t)
            elif f["ioi_share"] >= 0.6 and f["n_onsets"] >= 3:
                out["rhythm"].append(t)
            else:
                out["texture"].append(t)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTOR GLOBAL (subproceso con degradación)
# ══════════════════════════════════════════════════════════════════════════════

def detect_global(path: str, tracks, total, tpb) -> Tuple[Dict[int, Tuple[str, float]], str]:
    """{índice de pista: (rol, confianza)}, fuente. role_detector por subproceso o interno."""
    rd = Path(__file__).resolve().parent / "role_detector.py"
    if rd.is_file():
        try:
            p = subprocess.run([sys.executable, str(rd), "detect", path, "--json", "-"],
                               capture_output=True, text=True, timeout=60)
            if p.returncode == 0:
                txt = p.stdout[p.stdout.find("{"):]
                data = json.loads(txt)
                return ({d["index"]: (d["role"], float(d.get("confidence", 0.5)))
                         for d in data["tracks"] if d.get("n_notes", 0) > 0}, "role_detector")
        except Exception:
            pass
    items = [(t, f) for t in tracks for f in [_feats(t, 1, total, tpb)] if f]
    m = measure_roles(items)
    res = {}
    for role, ts in m.items():
        for t in ts:
            res[t["index"]] = ({"rhythm": "percussion", "texture": "harmony"}.get(role, role), 0.5)
    return res, "heurística interna"


# ══════════════════════════════════════════════════════════════════════════════
#  RESOLUCIÓN DE NOMBRES DEL PLAN → PISTAS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_name(name: str, tracks) -> Tuple[list, str, Optional[str]]:
    """→ (pistas, vía 'group'|'track'|'key', clave de atlas del grupo)."""
    f = _fold(name)
    if f in _GROUPS:
        fams, akey = _GROUPS[f]
        return [t for t in tracks if t["family"] in fams], "group", akey
    exact = [t for t in tracks if _fold(t["name"]) == f]
    if exact:
        return exact, "track", None
    sub = [t for t in tracks if f and f in _fold(t["name"])]
    if sub:
        return sub, "track", None
    key = instrument_key(name)
    if key:
        return [t for t in tracks if t["key"] == key], "key", None
    return [], "track", None


# ══════════════════════════════════════════════════════════════════════════════
#  COMPROBACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _fnd(sev, sec, role, msg, why="", fix="", rule=None, track=None):
    return dict(severity=sev, section=sec, role=role, message=msg, why=why, fix=fix,
                rule=rule if (rule and _rule_ok(rule)) else None, track=track)


def _names(ts):
    return ", ".join(t["name"] for t in ts)


def check_plan(midi: str, plan, detect: bool = True) -> dict:
    """API pública. `plan`: texto (o JSON) o lista de parse_plan. Devuelve el informe."""
    secs = parse_plan(plan) if isinstance(plan, str) else plan
    tracks, total, tpb = load_tracks(midi)
    det, det_src = detect_global(midi, tracks, total, tpb) if detect else ({}, "desactivado")
    findings: List[dict] = []
    out_secs = []
    plan_roles_used = set()

    for si, sec in enumerate(secs):
        a, b = sec["bars"]
        label = _fmt_bars(a, b)
        sec_out = dict(bars=[a, b], line=sec.get("line"), roles={}, expected=[], ok=0, total=0)
        if a > total:
            findings.append(_fnd("error", si, None,
                                 f"Tramo {label}: la obra solo tiene {total} compases.",
                                 "El plan describe compases que no existen en el MIDI.",
                                 f"Corrige el rango (máximo {total}) o exporta la obra completa."))
            sec_out["error"] = "fuera de la obra"
            out_secs.append(sec_out)
            continue
        ab = min(b, total)
        if b > total:
            findings.append(_fnd("info", si, None,
                                 f"Tramo {label}: la obra termina en el compás {total}; se evalúa {a}–{total}.",
                                 "", "Ajusta el final del tramo para que coincida con la obra."))
        nb = ab - a + 1
        active = []
        for t in tracks:
            f = _feats(t, a, ab, tpb)
            if f:
                active.append((t, f))
        by_idx = {t["index"]: f for t, f in active}
        pitched = [(t, f) for t, f in active if not t["unpitched"]]
        measured = measure_roles(active)
        if pitched:
            lo = min(f["mean_pitch"] for _, f in pitched)
            hi = max(f["mean_pitch"] for _, f in pitched)
            mx = max(f["n_att"] for _, f in pitched)
        resolved: Dict[str, list] = {}               # rol → [(pista, vía, nombre, clave grupo)]
        for role in ROLE_ORDER:
            for nm in sec["roles"].get(role, []):
                ts, via, gk = resolve_name(nm, tracks)
                if not ts:
                    findings.append(_fnd(
                        "error", si, role,
                        f"Tramo {label}: «{nm}» ({ROLE_ES[role]}) no existe en el MIDI.",
                        "El nombre no coincide con ninguna pista ni con un instrumento del MIDI.",
                        f"Pistas disponibles: {', '.join(t['name'] for t in tracks)}."))
                    continue
                resolved.setdefault(role, []).extend((t, via, nm, gk) for t in ts)
        declared_idx = {t["index"] for items in resolved.values() for t, _, _, _ in items}

        # ── comprobación rol a rol ──
        statuses: Dict[str, dict] = {}
        for role in ROLE_ORDER:
            declared = sec["roles"].get(role)
            core = role in CORE_ROLES
            if not declared:
                if core:
                    sug = measured.get(role) or []
                    findings.append(_fnd(
                        "error", si, role,
                        f"Tramo {label}: no declaras {ROLE_ES[role]} (rol básico sin cubrir).",
                        {"melody": "Sin una melodía clara el oyente no sabe a qué línea seguir.",
                         "harmony": "Sin armonía la melodía queda sin soporte ni dirección.",
                         "bass": "Sin bajo definido la armonía flota y se pierde la base."}[role]
                        + _rt("clear_roles"),
                        (f"Con lo medido, lo más parecido es: {_names(sug)}. "
                         f"Añade {role}=… al tramo." if sug else
                         f"Escribe una línea de {ROLE_ES[role]} o declara quién la hace."),
                        "clear_roles"))
                    statuses[role] = dict(status="✗", declared=[], tracks=[],
                                          reason="rol básico sin declarar")
                    sec_out["expected"].append(role)
                else:
                    statuses[role] = dict(status="·", declared=[], tracks=[], reason="")
                continue
            plan_roles_used.add(role)
            sec_out["expected"].append(role)
            items = resolved.get(role, [])
            ts = [t for t, _, _, _ in items]
            snd = [t for t in ts if t["index"] in by_idx]
            reason = ""
            ok = True
            if not ts:
                ok, reason = False, "ningún instrumento declarado existe"
            elif not snd:
                ok, reason = False, "ninguno suena en estos compases"
                findings.append(_fnd(
                    "error", si, role,
                    f"Tramo {label}: {_names(ts)} ({ROLE_ES[role]}) no suena en estos compases.",
                    "El rol declarado está vacío: el plan y la partitura no coinciden.",
                    "Escribe notas para ese instrumento o cambia el plan."))
            else:
                cov = set()
                for t in snd:
                    cov |= by_idx[t["index"]]["cov_bars"]
                cov_frac = len(cov) / nb
                if core and cov_frac < 0.75:
                    miss = [x for x in range(a, ab + 1) if x not in cov]
                    ok, reason = False, f"solo suena en {len(cov)} de {nb} compases"
                    findings.append(_fnd(
                        "error", si, role,
                        f"Tramo {label}: {ROLE_ES[role]} ({_names(snd)}) deja sin cubrir los "
                        f"compases {_fmt_ranges(miss)}.",
                        "Un rol básico con huecos deja la textura coja en esos compases.",
                        "Rellena esos compases o pasa el rol a otro instrumento."))
                elif cov_frac < 0.5:
                    findings.append(_fnd(
                        "info", si, role,
                        f"Tramo {label}: {ROLE_ES[role]} ({_names(snd)}) solo suena en "
                        f"{len(cov)} de {nb} compases.", "", ""))
                if ok:
                    ok, reason = _coherence(role, snd, by_idx, pitched, lo if pitched else 0,
                                            hi if pitched else 0, mx if pitched else 1,
                                            label, si, findings, tpb)
            # calidad: registro débil e idoneidad (no cambia ✓/✗)
            for t, via, nm, gk in items:
                if t["index"] not in by_idx:
                    continue
                _quality(role, t, via, gk, by_idx[t["index"]], label, si, findings)
            # detector global (informativo)
            for t in snd:
                d = det.get(t["index"])
                if d and d[1] >= 0.7 and _det_conflict(role, d[0]):
                    findings.append(_fnd(
                        "info", si, role,
                        f"Tramo {label}: {t['name']} figura como {ROLE_ES[role]}, pero el "
                        f"detector global ({det_src}) lo ve como «{d[0]}» (conf. {d[1]:.2f}).",
                        "El detector mira toda la obra; puede cambiar de papel por tramos.",
                        "Si es intencionado ignóralo; si no, revisa su registro y ritmo."))
            statuses[role] = dict(status="✓" if ok else "✗",
                                  declared=list(declared),
                                  tracks=[dict(index=t["index"], name=t["name"]) for t in ts],
                                  reason=reason)

        # ── duplicados, misma pista melodía+bajo, instrumentos fuera del plan ──
        _structure_checks(si, label, a, ab, active, resolved, findings, tpb)
        missing_core = [r for r in CORE_ROLES if statuses[r]["status"] != "✓"]
        sec_out["roles"] = statuses
        sec_out["ok"] = sum(1 for r in sec_out["expected"] if statuses[r]["status"] == "✓")
        sec_out["total"] = len(sec_out["expected"])
        out_secs.append(sec_out)
        del missing_core

    if secs and not ({"rhythm", "texture"} & plan_roles_used):
        findings.append(_fnd(
            "info", None, None,
            "El plan no declara ritmo ni textura en ningún tramo.",
            "El hito pide roles claros de melodía, armonía, bajo, ritmo y textura.",
            "Añade rhythm=… (ostinato, percusión) y texture=… (color, adornos) donde proceda."))
    tot = sum(s["total"] for s in out_secs)
    ok = sum(s["ok"] for s in out_secs)
    overall = round(ok / tot, 3) if tot else 0.0
    nerr = sum(1 for f in findings if f["severity"] == "error")
    nwar = sum(1 for f in findings if f["severity"] == "aviso")
    return dict(tool="role_coach", version=VERSION, file=str(midi), bars_total=total,
                detector=det_src, sections=out_secs, findings=findings, overall=overall,
                summary=dict(sections=len(out_secs), roles_expected=tot, roles_ok=ok,
                             errors=nerr, warnings=nwar,
                             infos=sum(1 for f in findings if f["severity"] == "info")))


def _fmt_ranges(bars: List[int]) -> str:
    if not bars:
        return ""
    bars = sorted(bars)
    out, s, p = [], bars[0], bars[0]
    for x in bars[1:]:
        if x == p + 1:
            p = x
            continue
        out.append(_fmt_bars(s, p))
        s = p = x
    out.append(_fmt_bars(s, p))
    return ", ".join(out)


def _det_conflict(role, drole):
    if role == "melody":
        return drole in ("bass", "percussion", "harmony")
    if role == "bass":
        return drole in ("melody", "percussion", "countermelody")
    if role == "harmony":
        return drole in ("melody", "bass", "percussion")
    return False


def _coherence(role, snd, by_idx, pitched, lo, hi, mx, label, si, findings, tpb):
    fs = [(t, by_idx[t["index"]]) for t in snd]
    if role == "melody":
        pit = [(t, f) for t, f in fs if not t["unpitched"]]
        if not pit or not pitched:
            return False, "la melodía no puede ser percusión sin altura"
        best_decl = max(_melody_score(f, lo, hi, mx) for _, f in pit)
        top = max(pitched, key=lambda x: _melody_score(x[1], lo, hi, mx))
        if best_decl < 0.85 * _melody_score(top[1], lo, hi, mx) and top[0]["index"] not in {t["index"] for t, _ in pit}:
            d = max(pit, key=lambda x: _melody_score(x[1], lo, hi, mx))
            findings.append(_fnd(
                "aviso", si, role,
                f"Tramo {label}: la melodía declarada ({d[0]['name']}, registro medio "
                f"{pitch_name(int(d[1]['mean_pitch']))}) no es la línea más prominente: "
                f"destaca {top[0]['name']} ({pitch_name(int(top[1]['mean_pitch']))}, "
                f"{top[1]['n_att']} ataques).",
                "El oído sigue la línea más aguda, activa y continua; si otra la supera, "
                "la melodía se pierde." + _rt("melody_register"),
                f"Sube la melodía de registro o dale timbre distinto, o cambia el plan a "
                f"melody={top[0]['name']}.", "melody_register"))
            return False, f"destaca más {top[0]['name']}"
        return True, ""
    if role == "bass":
        pit = [(t, f) for t, f in fs if not t["unpitched"]]
        if not pit:
            return False, "el bajo no puede ser percusión sin altura"
        order = sorted(pitched, key=lambda x: x[1]["mean_pitch"])
        lowest2 = {x[0]["index"] for x in order[:2]}
        d = min(pit, key=lambda x: x[1]["mean_pitch"])
        if d[1]["mean_pitch"] <= 52 or d[0]["index"] in lowest2:
            return True, ""
        findings.append(_fnd(
            "aviso", si, role,
            f"Tramo {label}: el bajo declarado ({d[0]['name']}) está en "
            f"{pitch_name(int(d[1]['mean_pitch']))} y suena más grave {order[0][0]['name']} "
            f"({pitch_name(int(order[0][1]['mean_pitch']))}).",
            "El bajo debe ser lo más grave de la textura: es la base de la armonía."
            + _rt("bass_clarity"),
            f"Baja el bajo una octava o declara bass={order[0][0]['name']}.", "bass_clarity"))
        return False, "no es la voz más grave"
    if role == "harmony":
        ev = []
        for t in snd:
            for n in t["notes"]:
                ev.append((n.start, n.end, n.pitch))
        starts = sorted({s for s, _, _ in ev})
        if not starts:
            return False, "sin notas"
        poly = sum(1 for s in starts if len({p for st, en, p in ev if st <= s < en}) >= 2)
        poly_frac = poly / len(starts)
        meandur = sum(by_idx[t["index"]]["mean_dur_beats"] for t in snd) / len(snd)
        if poly_frac >= 0.6 or meandur >= 2:
            return True, ""
        findings.append(_fnd(
            "aviso", si, role,
            f"Tramo {label}: la armonía declarada ({_names(snd)}) no forma acordes: "
            f"solo suena una nota a la vez en el {100 - poly_frac * 100:.0f} % de los ataques.",
            "La armonía necesita al menos dos alturas simultáneas o notas largas que sostengan "
            "el acorde.", "Añade voces (divisi, otro instrumento) o alarga las notas.", "clear_roles"))
        return False, "no forma acordes"
    if role == "rhythm":
        if any(f["ioi_share"] >= 0.6 and f["n_onsets"] >= 3 for _, f in fs):
            return True, ""
        findings.append(_fnd(
            "aviso", si, role,
            f"Tramo {label}: el ritmo declarado ({_names(snd)}) no muestra un patrón regular "
            f"de ataques.",
            "El rol rítmico se reconoce por un pulso u ostinato estable.",
            "Escribe un patrón repetido (ostinato, pulso) o cámbialo a texture.", "clear_roles"))
        return False, "sin patrón regular"
    if role == "countermelody":
        if any(f["linearity"] >= 0.6 for _, f in fs):
            return True, ""
        findings.append(_fnd(
            "aviso", si, role,
            f"Tramo {label}: la contramelodía declarada ({_names(snd)}) no es una línea "
            f"única: toca varias notas a la vez.",
            "Una contramelodía es una voz independiente; los acordes son armonía.",
            "Pásala a harmony o simplifícala a una sola línea.", "clear_roles"))
        return False, "no es una línea"
    return True, ""


def _quality(role, t, via, gk, f, label, si, findings):
    key = gk if via == "group" else t["key"]
    inst = ATLAS.get(key) if key else None
    if not inst or t["unpitched"] and role != "rhythm":
        return
    sc = inst["roles"][role]["score"]
    p = int(round(f["mean_pitch"]))
    if sc <= 1:
        alt = [x for x in best_for_role(role, pitch=p, top=5) if x["key"] != key][:3]
        findings.append(_fnd(
            "aviso", si, role,
            f"Tramo {label}: {inst['es']} apenas encaja como {ROLE_ES[role]} "
            f"(idoneidad {sc}/3: {inst['roles'][role]['note']}).",
            "Cada instrumento tiene funciones donde rinde y otras en las que no."
            + _rt("clear_roles"),
            ("Alternativas en ese registro: " + ", ".join(x["es"] for x in alt) + "."
             if alt else "Reasigna el rol."), "clear_roles", t["name"]))
        return
    if via != "group" and role in ("melody", "countermelody", "bass") and f["weak_frac"] >= 0.4:
        z = zone_of(key, p)
        alt = [x for x in best_for_role(role, pitch=p, top=5) if x["key"] != key][:3]
        rid = "winds_low" if inst["family"] == "viento_madera" else "natural_dynamics"
        findings.append(_fnd(
            "aviso", si, role,
            f"Tramo {label}: {t['name']} toca el {f['weak_frac'] * 100:.0f} % de sus notas en "
            f"una zona de registro débil ({z['label']}: {z['character']}).",
            "En esa zona el instrumento proyecta poco y queda tapado por el acompañamiento."
            + _rt(rid),
            "Sube/baja la línea a la zona media o cámbiala a "
            + (", ".join(x["es"] for x in alt) if alt else "otro instrumento") + ".", rid, t["name"]))


def _structure_checks(si, label, a, b, active, resolved, findings, tpb):
    roles_of: Dict[int, List[Tuple[str, str]]] = {}
    for role, items in resolved.items():
        for t, via, nm, gk in items:
            roles_of.setdefault(t["index"], []).append((role, via))
    # misma pista en melodía y bajo (declaración explícita)
    for idx, rv in roles_of.items():
        ex = {r for r, v in rv if v != "group"}
        if {"melody", "bass"} <= {r for r, _ in rv} and ex & {"melody", "bass"}:
            t = next(x for x, _ in active if x["index"] == idx)
            findings.append(_fnd(
                "aviso", si, "bass",
                f"Tramo {label}: {t['name']} figura a la vez como melodía y como bajo.",
                "Una misma pista no puede estar en los dos extremos de la textura." + _rt("clear_roles"),
                "Reparte los roles entre dos instrumentos o dobla en octavas con otra pista.",
                "clear_roles", t["name"]))
    # instrumentos que suenan y no están en el plan
    for t, f in active:
        if t["index"] not in roles_of:
            findings.append(_fnd(
                "aviso", si, None,
                f"Tramo {label}: {t['name']} suena ({f['n_att']} ataques) y no tiene rol en el plan.",
                "Un instrumento sin papel declarado suele ser relleno accidental que enturbia."
                + _rt("clear_roles"),
                "Asígnalo a un rol (p. ej. texture) o silencia la pista en esos compases.",
                "clear_roles", t["name"]))
    # roles duplicados: pistas que tocan casi lo mismo
    sets = {}
    for t, f in active:
        if t["unpitched"]:
            continue
        sets[t["index"]] = {(n.start, n.pitch % 12) for n in t["notes"] if a <= n.b0 <= b}
    idxs = sorted(sets)
    byid = {t["index"]: t for t, _ in active}
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            A, B = sets[idxs[i]], sets[idxs[j]]
            if len(A) < 4 or len(B) < 4:
                continue
            jac = len(A & B) / len(A | B)
            if jac < 0.8:
                continue
            ra = {r for r, _ in roles_of.get(idxs[i], [])}
            rb = {r for r, _ in roles_of.get(idxs[j], [])}
            ta, tb = byid[idxs[i]], byid[idxs[j]]
            if ra & rb:
                findings.append(_fnd(
                    "info", si, next(iter(ra & rb)),
                    f"Tramo {label}: {ta['name']} y {tb['name']} doblan la misma línea "
                    f"({jac * 100:.0f} % de coincidencia).",
                    "Es un doblaje: refuerza o cambia el color, pero resta independencia.",
                    "Si no es intencionado, elimina uno; si lo es, pruébalo con doubling_lab."))
            elif ra and rb:
                findings.append(_fnd(
                    "aviso", si, None,
                    f"Tramo {label}: {ta['name']} ({'/'.join(ROLE_ES[r] for r in sorted(ra))}) y "
                    f"{tb['name']} ({'/'.join(ROLE_ES[r] for r in sorted(rb))}) tocan casi lo mismo "
                    f"({jac * 100:.0f} % de coincidencia).",
                    "Dos pistas con roles distintos que hacen lo mismo duplican la función."
                    + _rt("clear_roles"),
                    "Dale a una de las dos material propio o quítala.", "clear_roles"))


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def template_plan(midi: str) -> str:
    """Plan editable a partir de lo que se detecta: tramos por cambio de instrumentación."""
    tracks, total, tpb = load_tracks(midi)
    sets = []
    for bar in range(1, total + 1):
        sets.append(frozenset(t["index"] for t in tracks
                              if any(n.b0 <= bar <= n.b1 for n in t["notes"])))
    segs = []
    for bar, s in enumerate(sets, 1):
        if segs and segs[-1][2] == s:
            segs[-1][1] = bar
        else:
            segs.append([bar, bar, s])
    merged: List[list] = []
    for sg in segs:                                       # tramos de <2 compases se funden
        if merged and (sg[1] - sg[0] + 1 < 2 or not sg[2]):
            merged[-1][1] = sg[1]
        else:
            merged.append(sg)
    if len(merged) > 1 and merged[0][1] - merged[0][0] + 1 < 2:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    lines = [f"# Plan de roles generado por role_coach.py a partir de {Path(midi).name}",
             "# Edítalo: cambia los instrumentos y comprueba con  role_coach.py check",
             "# Roles: melody countermelody harmony bass rhythm texture", ""]
    for a, b, _ in merged:
        items = [(t, f) for t in tracks for f in [_feats(t, a, b, tpb)] if f]
        if not items:
            lines.append(f"# {_fmt_bars(a, b).replace('–', '-')}  (sin notas)")
            continue
        m = measure_roles(items)
        parts = [f"{r}={','.join(t['name'] for t in m[r])}" for r in ROLE_ORDER if m[r]]
        lines.append(f"{a}-{b}  " + "  ".join(parts))
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

_SEV_ICON = {"error": ("✖", "RED"), "aviso": ("⚑", "YEL"), "info": ("ℹ", "CYA")}


def _wrap(text, indent, width=78):
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def print_report(rep: dict):
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}")
    print(f"  {B}ROLE COACH v{VERSION}  ·  {rep['file']}{R}   ({rep['bars_total']} compases)")
    print(f"  detector de roles: {rep['detector']}")
    print(f"{'═' * 78}")
    for si, s in enumerate(rep["sections"]):
        a, b = s["bars"]
        print(f"\n  {B}Tramo {_fmt_bars(a, b)}{R}   cumplidos {s['ok']}/{s['total']}")
        if s.get("error"):
            print(f"    {_c('RED')}✖ {s['error']}{R}")
            continue
        for role in ROLE_ORDER:
            st = s["roles"][role]
            mark = {"✓": _c("GRN") + "✓", "✗": _c("RED") + "✗", "·": G + "·"}[st["status"]] + R
            if st["declared"]:
                trk = ", ".join(t["name"] for t in st["tracks"])
                tail = f"{', '.join(st['declared'])} → {trk or '—'}"
                if st["reason"]:
                    tail += f"  {_c('YEL')}({st['reason']}){R}"
            else:
                tail = f"{G}{st['reason'] or '(no declarado)'}{R}"
            print(f"    {ROLE_ES[role]:<14} {mark}  {tail}")
    good = [s for s in rep["sections"] if s.get("total") and s["ok"] == s["total"]]
    if good:
        print(f"\n  {_c('GRN')}Lo que funciona:{R}")
        for s in good:
            print(f"    ✓ Tramo {_fmt_bars(*s['bars'])}: todos los roles esperados están "
                  f"cubiertos y son coherentes con lo medido.")
    elif rep["overall"] > 0:
        print(f"\n  {_c('GRN')}Lo que funciona:{R}")
        for s in rep["sections"]:
            for r in ROLE_ORDER:
                if s.get("roles") and s["roles"][r]["status"] == "✓":
                    print(f"    ✓ Tramo {_fmt_bars(*s['bars'])}: {ROLE_ES[r]} "
                          f"({', '.join(s['roles'][r]['declared'])}) cumple.")
    fs = [f for f in rep["findings"]]
    order = {"error": 0, "aviso": 1, "info": 2}
    fs.sort(key=lambda f: (order[f["severity"]], f["section"] if f["section"] is not None else -1))
    if fs:
        print(f"\n  {_c('YEL')}Para mejorar:{R}")
        for f in fs:
            ic, col = _SEV_ICON[f["severity"]]
            print(_wrap(f"{_c(col)}{ic}{R} {f['message']}", "    "))
            if f["why"]:
                print(_wrap("¿Por qué? " + f["why"], "        "))
            if f["fix"]:
                print(_wrap("→ " + f["fix"], "        "))
    else:
        print(f"\n  {G}(sin sugerencias: el plan se cumple){R}")
    sm = rep["summary"]
    print(f"\n  Roles cumplidos: {sm['roles_ok']}/{sm['roles_expected']}  "
          f"({rep['overall'] * 100:.0f} %)   errores {sm['errors']} · avisos {sm['warnings']}")
    print(f"{'═' * 78}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST
# ══════════════════════════════════════════════════════════════════════════════

def _synth(path, specs, tpb=480):
    """specs: [(nombre, programa, canal, [(inicio, duración, altura, vel)])] → MIDI 4/4."""
    mid = MidiData(fmt=1, tpb=tpb)
    mid.tempo_map = [(0, 600000)]
    mid.timesig_map = [(0, 4, 4)]
    c = MidiTrackData(name="Conductor")
    c.events.append(MidiEvent(abs=0, kind="meta", meta_type=0x51,
                              data=bytes([0xFF, 0x51, 0x03]) + (600000).to_bytes(3, "big")))
    c.events.append(MidiEvent(abs=0, kind="meta", meta_type=0x58,
                              data=bytes([0xFF, 0x58, 0x04, 4, 2, 24, 8])))
    mid.tracks.append(c)
    for name, prog, ch, notes in specs:
        t = MidiTrackData(name=name, programs=[prog], channels=[ch])
        nb = name.encode("latin-1")
        t.events.append(MidiEvent(abs=0, kind="meta", meta_type=0x03,
                                  data=bytes([0xFF, 0x03, len(nb)]) + nb))
        t.events.append(MidiEvent(abs=0, kind="channel", channel=ch, data=bytes([0xC0 | ch, prog])))
        for st, du, p, v in notes:
            t.events.append(MidiEvent(abs=st, kind="note_on", channel=ch, pitch=p, vel=v))
            t.events.append(MidiEvent(abs=st + du - 2, kind="note_off", channel=ch, pitch=p, vel=0))
        mid.tracks.append(t)
    write_midi(mid, path)


def _selftest_specs(bars=8, bad=False):
    BAR = 480 * 4
    chord = [(48, 52, 55), (45, 48, 52), (41, 45, 48), (43, 47, 50)]
    mel = [[76, 74, 72, 79], [79, 77, 76, 74], [77, 76, 74, 72], [74, 76, 79, 76]]
    flute, v1, v2, vla, vc, tim = [], [], [], [], [], []
    for b in range(bars):
        c = chord[b % 4]
        for i, p in enumerate(mel[b % 4]):
            flute.append((b * BAR + i * 480, 480, p, 80))
        v1.append((b * BAR, BAR, c[2] + 12, 70))
        v2.append((b * BAR, BAR, c[1] + 12, 70))
        vla.append((b * BAR, BAR, c[0] + 12, 70))
        vc.append((b * BAR, BAR, c[0] - (12 if not bad else 0), 78))
        for i in range(4):
            tim.append((b * BAR + i * 480, 480, 48 if i % 2 == 0 else 43, 90))
    return [("Flauta", 73, 0, flute), ("Violin I", 40, 1, v1), ("Violin II", 40, 2, v2),
            ("Viola", 41, 3, vla), ("Violonchelo", 42, 4, vc), ("Timbales", 47, 5, tim)]


def selftest():
    errs = []
    tmp = tempfile.mkdtemp(prefix="role_coach_")
    try:
        p = str(Path(tmp) / "t.mid")
        _synth(p, _selftest_specs())
        # parseo
        s = parse_plan("1-8 melody=flauta harmony=cuerdas,violin bass=violonchelo rhythm=timbales # x\n9 texture=arpa")
        if len(s) != 2 or s[0]["bars"] != (1, 8) or s[0]["roles"]["harmony"] != ["cuerdas", "violin"]:
            errs.append("parse_plan texto")
        j = parse_plan('{"sections":[{"bars":"1-4","roles":{"melodia":"oboe","bass":["fagot"]}}]}')
        if j[0]["roles"] != {"melody": ["oboe"], "bass": ["fagot"]}:
            errs.append("parse_plan JSON")
        for bad in ["", "hola", "1-8 foo=oboe", "8-1 melody=oboe", "1-8 melody=", "1-8 oboe"]:
            try:
                parse_plan(bad)
                errs.append(f"plan malformado aceptado: {bad!r}")
            except ValueError:
                pass
        good = "1-8 melody=flauta harmony=cuerdas bass=violonchelo rhythm=timbales"
        r = check_plan(p, good, detect=False)
        if r["overall"] != 1.0 or r["summary"]["errors"]:
            errs.append(f"plan correcto: overall {r['overall']} errores {r['summary']['errors']}")
        r = check_plan(p, "1-8 melody=flauta harmony=cuerdas bass=flauta", detect=False)
        if not any("bass" == f["role"] and f["severity"] in ("aviso", "error") for f in r["findings"]) \
                or r["sections"][0]["roles"]["bass"]["status"] != "✗":
            errs.append("bajo incoherente no detectado")
        r = check_plan(p, "1-8 melody=flauta harmony=cuerdas", detect=False)
        if not any(f["role"] == "bass" and f["severity"] == "error" for f in r["findings"]):
            errs.append("rol básico sin declarar no detectado")
        r = check_plan(p, "1-8 melody=flauta harmony=violin bass=violonchelo", detect=False)
        if not any("no tiene rol" in f["message"] for f in r["findings"]):
            errs.append("instrumento fuera del plan no detectado")
        r = check_plan(p, "1-8 melody=arpa harmony=cuerdas bass=violonchelo", detect=False)
        if not any("no existe" in f["message"] for f in r["findings"]):
            errs.append("instrumento inexistente no detectado")
        r = check_plan(p, "20-24 melody=flauta", detect=False)
        if not any("solo tiene" in f["message"] for f in r["findings"]):
            errs.append("tramo fuera de la obra no detectado")
        r = check_plan(p, "1-8 melody=violin harmony=cuerdas bass=violonchelo", detect=False)
        if r["sections"][0]["roles"]["melody"]["status"] != "✗":
            errs.append("melodía poco prominente no detectada")
        r = check_plan(p, "1-8 melody=flauta harmony=cuerdas bass=violonchelo", detect=True)
        if r["overall"] != 1.0:
            errs.append("check con detector global")
        # template
        tp = template_plan(p)
        if "melody=Flauta" not in tp or "bass=" not in tp:
            errs.append("template sin melodía/bajo")
        r = check_plan(p, tp, detect=False)
        if r["overall"] < 0.99:
            errs.append(f"el template no pasa su propio check ({r['overall']})")
        # una sola pista
        p1 = str(Path(tmp) / "uno.mid")
        _synth(p1, _selftest_specs()[:1])
        r = check_plan(p1, "1-8 melody=flauta", detect=False)
        if "sections" not in r:
            errs.append("MIDI de una pista")
        # pistas vacías y sin nombre
        p2 = str(Path(tmp) / "vacia.mid")
        specs = _selftest_specs()[:2] + [("", 0, 6, [])]
        _synth(p2, specs)
        r = check_plan(p2, "1-8 melody=flauta", detect=False)
        if r["bars_total"] != 8:
            errs.append("pista vacía altera los compases")
        # fichero inexistente
        try:
            check_plan(str(Path(tmp) / "no.mid"), good)
            errs.append("fichero inexistente aceptado")
        except FileNotFoundError:
            pass
        # reglas citadas existen
        for rid in ("clear_roles", "melody_register", "bass_clarity", "winds_low", "natural_dynamics"):
            if not _rule_ok(rid):
                errs.append(f"regla {rid} no existe en el atlas")
    except Exception as e:                                      # pragma: no cover
        errs.append(f"excepción: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for e in errs:
        print(f"  ✗ {e}")
    print(f"  {'OK' if not errs else 'FALLO'} · role_coach v{VERSION} selftest")
    return 0 if not errs else 3


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def cmd_template(a):
    txt = template_plan(a.midi)
    if a.out:
        _ensure_parent(Path(a.out)).write_text(txt, encoding="utf-8")
        print(f"Plan escrito en {a.out}")
    else:
        sys.stdout.write(txt)
    return 0


def cmd_check(a):
    plan_path = Path(a.plan)
    if not plan_path.is_file():
        raise FileNotFoundError(a.plan)
    rep = check_plan(a.midi, plan_path.read_text(encoding="utf-8-sig"))
    if a.json == "-":
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        print_report(rep)
        if a.json:
            _ensure_parent(Path(a.json)).write_text(
                json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  JSON: {a.json}\n")
    if a.strict and (rep["summary"]["errors"] or rep["summary"]["warnings"]):
        return 3
    return 0


def main():
    global _USE_COLOR
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true")
    ap = argparse.ArgumentParser(
        prog="role_coach.py",
        description="Comprueba que tu plan de roles orquestales se cumple en el MIDI.")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("template", parents=[common], help="Plan editable a partir del MIDI")
    p.add_argument("midi")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_template)

    p = sp.add_parser("check", parents=[common], help="Comprueba el plan contra el MIDI")
    p.add_argument("midi")
    p.add_argument("--plan", required=True)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--json")
    p.set_defaults(fn=cmd_check)

    p = sp.add_parser("selftest", parents=[common], help="Verifica con MIDIs sintéticos")
    p.set_defaults(fn=lambda a: selftest())

    a = ap.parse_args()
    if a.no_color:
        _USE_COLOR = False
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e.filename or e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    except Exception as e:                                      # pragma: no cover
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
