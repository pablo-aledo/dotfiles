#!/usr/bin/env python3
"""
doubling_lab.py — Doubling Lab: laboratorio de doblajes para ESCUCHAR y decidir
en FL Studio con tus propias librerías
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Doblar una línea (dársela a dos o más instrumentos a la vez) es la herramienta
de orquestación con más efecto por nota escrita: un oboe sobre las cuerdas da
foco, una flauta a la octava da brillo, las trompas con los chelos dan calidez.
Pero el resultado depende de TUS librerías y de tus oídos: ninguna tabla
sustituye escuchar. Esta herramienta no decide por ti; te prepara el
experimento.

  1. `suggest` analiza una línea de tu MIDI (registro, dinámica, rango, rol
     probable) y, con el catálogo de orchestral_atlas, lista las combinaciones
     de instrumentos que encajan. Para cada instrumento indica si las notas
     caben en su rango físico y qué transposición de octava haría falta.
  2. `variants` escribe un MIDI por variante (multipista, una pista por
     instrumento con su nombre y su program change General MIDI, cada una en su
     canal y nunca en el 10), con la línea original transpuesta por octavas a
     la zona cómoda de cada instrumento y las velocidades equilibradas con la
     «proyección» del atlas. Añade `00_comparar_todas.mid` (todas las variantes
     seguidas, con un compás de silencio entre ellas, para compararlas de un
     tirón) y `ficha_variantes.txt` (efecto, precaución, rol y qué escuchar en
     cada una). Incluye además la variante `solo` (la línea sin doblar) como
     punto de comparación.

Se conservan el tempo y el compás del original. Con --bars el fragmento se
traslada al compás 1 de cada variante, pero se mantienen los cambios de tempo
y compás que ocurren dentro de él.

QUÉ HACE CON LA LÍNEA
  octava       — cada instrumento recibe la línea desplazada por octavas
                 completas (nunca cambia la tonalidad) hasta su zona cómoda;
                 en las combinaciones de «octava» el atlas fija la relación
                 (p. ej. la flauta una octava sobre el violín).
  plegado      — si el ámbito de la línea es mayor que el del instrumento, las
                 notas que sobresalen se pliegan a otra octava: la ficha lo
                 avisa, porque el contorno cambia.
  monofonía    — los instrumentos monofónicos reciben solo la voz superior;
                 el resto se limita a su polifonía máxima.
  velocidades  — la dinámica original se traduce a proyección (atlas) y se
                 reparte: el primer instrumento de la combinación lidera y
                 los demás quedan algo por debajo; cada uno recibe la velocity
                 que iguala ese nivel en SU registro.

LIMITACIONES: el equilibrio es ORIENTATIVO: sale de una heurística por familia,
registro y velocity, no del audio de tus librerías; ajústalo por oído (y con
los faders/expresión de cada patch). Las variantes usan la línea tal cual está
escrita: no reescriben ritmos ni articulaciones para cada instrumento (un
fagot no hará el mismo picado que un violín). El timbal solo recibe las notas
que caen en pulso, plegadas a su ámbito. La estimación del rol (melodía,
bajo…) es una heurística: fuérzalo con --role. Las pistas de percusión no
sirven como línea. No hay audio ni red.

COMO MÓDULO:
  from doubling_lab import suggest_doublings, make_variants

DEPENDENCIAS: stdlib · playability_auditor.py (E/S MIDI e instrumentos) ·
orchestral_atlas.py (catálogo de combinaciones y proyección)

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  suggest   Analiza una línea y lista las combinaciones de doblaje que encajan
  variants  Escribe un MIDI por variante + comparador + ficha para escuchar
  selftest  Verifica la herramienta con MIDIs sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  suggest   midi --track N [--bars A-B] [--role R] [--json FILE|-] [--no-color]
  variants  midi --track N [--bars A-B] [--out-dir DIR] [--combos id1,id2,…]
                 [--strict] [--json FILE|-] [--no-color]
  selftest  [--no-color]

  --track N           Pista (índice 0-based) cuya línea se dobla (obligatorio)
  --bars A-B          Limita la línea a un rango de compases (1-based)
  --role R            Rol de la línea: melody, countermelody, harmony, bass,
                      rhythm, texture (por defecto se estima del registro y
                      la polifonía)
  --out-dir DIR       Carpeta de salida (por defecto ./doubling_variants)
  --combos id1,id2    Ids de combinación del atlas separados por comas (ver
                      `python orchestral_atlas.py combos`); por defecto, las 6
                      mejores de `suggest` cuyas notas caben sin plegar
  --strict            rc=3 si alguna variante necesita plegar notas
  --json FILE         Sidecar JSON ('-' = stdout)

CÓDIGOS DE SALIDA:  0 ok · 1 error de ejecución · 2 uso incorrecto (fichero
                    inexistente, pista inexistente o vacía, --bars/--role/
                    --combos inválidos) · 3 selftest fallido o, con --strict,
                    alguna variante necesita plegar notas

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
[suggest]
midi    file  io=in ext=mid required  # MIDI multipista con la línea a doblar
--track int   min=0 required  # Pista de la línea (0-based)
--bars  str   format="A-B"  # Rango de compases (1-based)
--role  enum  values=melody,countermelody,harmony,bass,rhythm,texture  # Rol de la línea
--json  file  io=out ext=json  # Sidecar JSON ('-' = stdout)
[variants]
midi      file  io=in ext=mid required  # MIDI multipista con la línea a doblar
--track   int   min=0 required  # Pista de la línea (0-based)
--bars    str   format="A-B"  # Rango de compases (1-based)
--out-dir dir   default=./doubling_variants io=out  # Carpeta de salida
--combos  str   format="id1,id2"  # Ids de combinación del atlas
--strict  bool  default=false  # rc=3 si alguna variante necesita plegar notas
--json    file  io=out ext=json  # Sidecar JSON ('-' = stdout)
[selftest]
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python doubling_lab.py suggest obra.mid --track 1
  python doubling_lab.py suggest obra.mid --track 1 --bars 5-12 --role melody
  python doubling_lab.py variants obra.mid --track 1
  python doubling_lab.py variants obra.mid --track 1 --bars 5-12 --out-dir pruebas_doblaje
  python doubling_lab.py variants obra.mid --track 1 --combos flauta_oboe,cuerdas_oboe --json doblajes.json
  python doubling_lab.py selftest
"""

import os
import re
import sys
import json
import math
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from playability_auditor import (read_midi, write_midi, TimeContext, extract_notes,
                                     pitch_name, INSTRUMENT_DB, resolve_instrument,
                                     detect_instrument, MidiData, MidiTrackData, MidiEvent)
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
    from orchestral_atlas import (ATLAS, COMBOS, ROLES, ROLE_ES, REGISTER_ES, zone_of,
                                  projection, find_combos, register_of)
except ImportError:
    print("ERROR: requiere orchestral_atlas.py en el mismo directorio (catálogo)")
    sys.exit(1)

VERSION = "1.0"
TOOL = "doubling_lab"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()

MAX_DEFAULT = 6                        # variantes por defecto (sin contar `solo`)
COMBO_BY_ID = {c["id"]: c for c in COMBOS}

# Program change General MIDI (0-based) por clave de instrumento
GM_PROGRAM = {"violin": 40, "viola": 41, "violonchelo": 42, "contrabajo": 43, "cuerdas": 48,
              "arpa": 46, "timbal": 47, "trompeta": 56, "trombon": 57, "tuba": 58,
              "trompa": 60, "oboe": 68, "corno_ingles": 69, "fagot": 70, "clarinete": 71,
              "flautin": 72, "flauta": 73, "celesta": 8,
              # sin ficha en la lista pedida pero posibles como pista origen o en combos
              "laminas": 12, "viento_metal_gen": 61, "voz": 52, "piano": 0,
              "organo": 19, "guitarra": 24, "bajo": 32, "saxofon": 65, "clave": 6,
              "sintetizador": 80}
# Canales disponibles: nunca el 10 (índice 9, reservado a percusión GM)
CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15]
# Nombre de pista (reconocible por playability_auditor)
TRACK_NAME = {"cuerdas": "Cuerdas", "viento_metal_gen": "Metales", "laminas": "Laminas",
              "corno_ingles": "Corno inglés", "timbal": "Timbales", "voz": "Voz"}

# Relación de octavas respecto al PRIMER instrumento de la combinación (en octavas)
LAYOUT = {
    "flauta_clarinete": {"clarinete": -1},
    "oboe_fagot_2oct": {"fagot": -2},
    "violines_flauta_oct": {"flauta": 1},
    "chelos_contrabajos_oct": {"contrabajo": -1},
    "fagot_chelos_bajo": {"violonchelo": 0, "contrabajo": -1},
    "flautin_violines_climax": {"violin": -1},
    "cuerdas_melodia_octavas": {"viola": 0, "violonchelo": -1},
    "trompeta_trombon": {"trombon": -1},
}

LISTEN = {
    "dos octavas": "si el hueco de las dos octavas suena abierto y pastoral o falta una voz "
                   "intermedia; si el grave sostiene al agudo sin tapar su melodía.",
    "octava": "si la octava superior aporta brillo sin separarse como una voz aparte; "
              "la afinación entre octavas en las notas largas; que el agudo no empuje al grave.",
    "unísono": "si los timbres se funden en uno nuevo o se oye cada uno por separado; "
               "cuál ataca primero en cada nota y si uno tapa al otro en las notas largas.",
    "acordes": "que las voces se mezclen en un bloque homogéneo y ninguna sobresalga; "
               "el color de los acordes largos y el equilibrio de las voces extremas.",
    "capas": "que cada capa conserve su función (la principal al frente, la otra de "
             "color o de apoyo) y que ninguna enmascare a la principal.",
}
LISTEN_DEFAULT = "si la combinación suena como un instrumento nuevo y si la línea gana claridad."


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

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


def _wrap(text, indent, width=78):
    import textwrap
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _parse_bars(s) -> Optional[Tuple[int, int]]:
    if s is None or s == "":
        return None
    if isinstance(s, (tuple, list)):
        a, b = int(s[0]), int(s[1])
    else:
        m = re.fullmatch(r"\s*(\d+)\s*(?:[-–:]\s*(\d+))?\s*", str(s))
        if not m:
            raise ValueError(f"--bars inválido: {s!r} (usa A-B, p.ej. 5-12)")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
    if a < 1 or b < a:
        raise ValueError(f"--bars inválido: {s!r} (A ≥ 1 y B ≥ A)")
    return a, b


def _oct_txt(n: int) -> str:
    if n == 0:
        return "sin transponer"
    k = abs(n)
    return f"{'+' if n > 0 else '−'}{k} {'octava' if k == 1 else 'octavas'}"


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


def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA Y ANÁLISIS DE LA LÍNEA
# ══════════════════════════════════════════════════════════════════════════════

def load_line(path: str, track: int, bars=None) -> dict:
    """Extrae la línea de una pista (y rango de compases) y la analiza.
    Las notas se devuelven en ticks relativos al inicio del fragmento."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    bars = _parse_bars(bars)
    mid = read_midi(path)
    if not isinstance(track, int) or track < 0 or track >= len(mid.tracks):
        raise ValueError(f"la pista {track} no existe (el MIDI tiene {len(mid.tracks)} pistas, 0-{len(mid.tracks) - 1})")
    trk = mid.tracks[track]
    real, _ks = _split_keyswitches(extract_notes(trk), mid.tpb)
    if not real:
        raise ValueError(f"la pista {track} ({trk.name or 'sin nombre'}) no tiene notas")
    key, _why = resolve_instrument(trk, real, None)
    if key == "percusion":
        raise ValueError(f"la pista {track} es de percusión: no tiene línea melódica que doblar")
    alln = [n.end for t in mid.tracks for n in extract_notes(t)]
    grid = _bar_grid(mid, max(0, max(alln) - 1))
    nbars = len(grid)
    if bars is None:
        a = 1
        last_note = max(n.end for n in real) - 1
        b = next((i + 1 for i, g in enumerate(grid) if g[0] <= last_note < g[1]), nbars)
    else:
        a, b = bars
        if a > nbars:
            raise ValueError(f"--bars {a}-{b}: fuera de la obra (tiene {nbars} compases)")
        b = min(b, nbars)
    t0, t1 = grid[a - 1][0], grid[b - 1][1]
    sel = []
    for n in real:
        if t0 <= n.start < t1:
            sel.append((n.start - t0, min(n.end, t1) - t0, n.pitch, max(1, n.vel)))
    if not sel:
        raise ValueError(f"la pista {track} no tiene notas en los compases {a}-{b}")
    sel.sort(key=lambda x: (x[0], x[2]))
    pitches = [x[2] for x in sel]
    med = _median(pitches)
    mean_vel = sum(x[3] for x in sel) / len(sel)
    onsets = {}
    for x in sel:
        onsets[x[0]] = onsets.get(x[0], 0) + 1
    poly_frac = sum(1 for v in onsets.values() if v > 1) / len(onsets)
    dyn = "soft" if mean_vel <= 64 else ("mid" if mean_vel <= 95 else "loud")
    if med < 52 and poly_frac < 0.3:
        role = "bass"
    elif poly_frac >= 0.3:
        role = "harmony"
    else:
        role = "melody"
    # tempo y compás vigentes al inicio del fragmento y cambios posteriores
    tempos = _shift_map(mid.tempo_map, t0, t1)
    sigs = _shift_map([(t, (n, d)) for t, n, d in mid.timesig_map], t0, t1)
    return dict(path=path, track=track, track_name=trk.name or f"Pista {track}",
                src_key=key, src_es=(ATLAS[key]["es"] if key in ATLAS else INSTRUMENT_DB[key]["es"]),
                notes=sel, pitches=pitches, tpb=mid.tpb, bars=(a, b), nbars=nbars,
                len_ticks=t1 - t0, bar_ticks=grid[a - 1][1] - grid[a - 1][0],
                tempos=tempos, sigs=sigs,
                lo=min(pitches), hi=max(pitches), median=med, mean_vel=round(mean_vel, 1),
                register=register_of(int(round(med))), dynamic=dyn, poly_frac=round(poly_frac, 2),
                role_guess=role)


def _shift_map(entries, t0, t1):
    """Entradas (tick, valor) → [(tick_rel, valor)]: la vigente en t0 se coloca en 0."""
    ents = sorted(entries, key=lambda e: e[0])
    cur = None
    out = []
    for t, v in ents:
        if t <= t0:
            cur = v
        elif t < t1:
            out.append((t - t0, v))
    if cur is None:
        cur = ents[0][1]
    return [(0, cur)] + out


# ══════════════════════════════════════════════════════════════════════════════
#  AJUSTE DE LA LÍNEA A CADA INSTRUMENTO
# ══════════════════════════════════════════════════════════════════════════════

def _fit_cost(key: str, pitches, shift_oct: int):
    prof = INSTRUMENT_DB[key]
    lo, hi = prof["range"]
    slo, shi = prof["sweet"]
    sh = 12 * shift_oct
    out = sum(1 for p in pitches if not lo <= p + sh <= hi)
    outsw = sum(1 for p in pitches if not slo <= p + sh <= shi)
    centre = abs(sum(pitches) / len(pitches) + sh - (slo + shi) / 2.0)
    return (out, outsw, round(centre, 1), abs(shift_oct))


def _best_shift(key: str, pitches) -> int:
    return min(range(-6, 7), key=lambda s: _fit_cost(key, pitches, s))


def _fold_pitch(p: int, lo: int, hi: int) -> int:
    while p < lo:
        p += 12
    while p > hi:
        p -= 12
    return max(lo, min(hi, p)) if not lo <= p <= hi else p


def _plan_shifts(combo: dict, pitches) -> Dict[str, int]:
    """Desplazamiento en octavas por instrumento, respetando la relación del atlas."""
    insts = combo["insts"]
    lay = LAYOUT.get(combo["id"])
    if lay:
        first = insts[0]
        rel = dict(lay)
        rel[first] = 0
        best, bcost = None, None
        for anchor in range(-6, 7):
            shifts = {k: anchor + rel[k] for k in insts if k in rel}
            if len(shifts) != len(insts):
                continue
            cost = tuple(sum(_fit_cost(k, pitches, shifts[k])[i] for k in insts) for i in range(3))
            if bcost is None or cost < bcost:
                best, bcost = shifts, cost
        # si la relación fijada obliga a plegar más que el ajuste libre, se respeta igualmente:
        # la relación de octava ES el efecto de la combinación
        if best is not None:
            return best
    return {k: _best_shift(k, pitches) for k in insts}


def _reduce(notes, key: str, tpb: int):
    """Limita la polifonía al máximo del instrumento (voces superiores) y recorta
    solapes; el timbal solo conserva las notas en pulso."""
    prof = INSTRUMENT_DB[key]
    maxp = max(1, prof["max_poly"])
    notes = sorted(notes, key=lambda x: (x[0], -x[2]))
    if key == "timbal":
        on_beat = [n for n in notes if n[0] % tpb == 0]
        notes = on_beat or notes
    groups: Dict[int, list] = {}
    for n in notes:
        groups.setdefault(n[0], []).append(n)
    out = []
    for s in sorted(groups):
        grp = sorted(groups[s], key=lambda x: -x[2])[:maxp]
        active = [i for i, o in enumerate(out) if o[1] > s]
        extra = len(active) + len(grp) - maxp
        for i in sorted(active, key=lambda i: out[i][0])[:max(0, extra)]:
            o = out[i]
            out[i] = (o[0], s, o[2], o[3])
        out.extend(grp)
    return [o for o in out if o[1] > o[0]]


def _vel_for(key: str, pitch: int, target: float) -> int:
    """Velocity que iguala la proyección `target` del instrumento en esa nota."""
    f = (ATLAS[key]["base"] if key in ATLAS else 0.6) * (zone_of(key, pitch)["power"] / 3.0) * 1.35
    if f <= 0:
        return 100
    x = (target / f - 0.4) / 0.6
    return int(max(25, min(127, round(x * 127))))


def _render(line: dict, key: str, shift_oct: int, weight: float, balance: bool):
    """Notas de la línea para un instrumento: (notas, nº plegadas, nº recortadas)."""
    prof = INSTRUMENT_DB[key]
    lo, hi = prof["range"]
    src = line["src_key"]
    notes, folded = [], 0
    for st, en, p, v in line["notes"]:
        q = p + 12 * shift_oct
        if not lo <= q <= hi:
            q = _fold_pitch(q, lo, hi)
            folded += 1
        if balance:
            tgt = min(1.0, projection(src, p, v) * weight)
            v = _vel_for(key, q, tgt)
        notes.append((st, en, q, v))
    red = _reduce(notes, key, line["tpb"])
    return red, folded, len(notes) - len(red)


def _plan_combo(line: dict, combo: dict) -> dict:
    """Plan de una combinación: instrumentos, desplazamientos, ajuste y avisos."""
    pitches = line["pitches"]
    shifts = _plan_shifts(combo, pitches)
    n = len(combo["insts"])
    gain = 1.0 / (1.0 + 0.15 * (n - 1))
    insts = []
    for i, key in enumerate(combo["insts"]):
        if key not in INSTRUMENT_DB:
            raise ValueError(f"la combinación {combo['id']} usa un instrumento sin perfil: {key}")
        w = gain * (1.0 if i == 0 else (0.8 if n < 4 else 0.65))
        s = shifts[key]
        lo, hi = INSTRUMENT_DB[key]["range"]
        raw_out = sum(1 for p in pitches if not lo <= p + 12 * s <= hi)
        med = int(round(line["median"] + 12 * s))
        med = max(lo, min(hi, med))
        z = zone_of(key, med)
        insts.append(dict(key=key, es=(ATLAS[key]["es"] if key in ATLAS else INSTRUMENT_DB[key]["es"]),
                          shift_oct=s, weight=round(w, 2), fits=(raw_out == 0), n_out=raw_out,
                          pct_fit=round(100.0 * (len(pitches) - raw_out) / len(pitches), 1),
                          range=[lo, hi], zone=z["label"], zone_power=z["power"],
                          zone_character=z["character"],
                          sounding=[pitch_name(min(pitches) + 12 * s), pitch_name(max(pitches) + 12 * s)]))
    return dict(combo=combo, insts=insts)


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA: SUGERENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def suggest_doublings(path: str, track: int, bars=None, role: Optional[str] = None) -> list:
    """Combinaciones del atlas que encajan con la línea de `track`, de mejor a peor.
    Cada elemento: id, name, insts, relation, effect, use, caution, roles, register,
    dynamic, score, fits_all, n_folded_insts y, por instrumento, ajuste de rango y
    transposición de octava. Lanza ValueError / FileNotFoundError si la entrada no vale."""
    if role is not None and role not in ROLES:
        raise ValueError(f"rol desconocido: {role} (válidos: {', '.join(ROLES)})")
    line = load_line(path, track, bars)
    return _suggest(line, role)


def _suggest(line: dict, role: Optional[str]) -> list:
    r = role or line["role_guess"]
    cands = find_combos(role=r) or list(COMBOS)
    res = []
    for c in cands:
        plan = _plan_combo(line, c)
        insts = plan["insts"]
        fit = sum(i["pct_fit"] for i in insts) / (100.0 * len(insts))
        score = (3.0 + (2.0 if line["register"] in c["register"] else 0.0)
                 + (1.5 if line["dynamic"] in c["dynamic"] else 0.0)
                 + 2.0 * fit + (1.0 if line["src_key"] in c["insts"] else 0.0)
                 + 0.3 * c["blend"])
        why = []
        why.append("registro " + ("encaja" if line["register"] in c["register"] else "no es el ideal")
                   + f" ({REGISTER_ES[line['register']]})")
        why.append("dinámica " + ("encaja" if line["dynamic"] in c["dynamic"] else "no es la ideal"))
        res.append(dict(id=c["id"], name=c["name"], insts=insts, relation=c["relation"],
                        effect=c["effect"], use=c["use"], caution=c["caution"],
                        roles=list(c["roles"]), register=list(c["register"]),
                        dynamic=list(c["dynamic"]), blend=c["blend"],
                        score=round(score, 2), fits_all=all(i["fits"] for i in insts),
                        n_folded_insts=sum(1 for i in insts if not i["fits"]),
                        role_used=r, why="; ".join(why)))
    res.sort(key=lambda d: (-d["score"], d["id"]))
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _meta(abs_t: int, mtype: int, payload: bytes) -> MidiEvent:
    payload = payload[:120]
    return MidiEvent(abs=abs_t, kind="meta", meta_type=mtype,
                     data=bytes([0xFF, mtype, len(payload)]) + payload)


def _text(s: str) -> bytes:
    return s.encode("latin-1", errors="replace")


def _conductor(line: dict, slots) -> MidiTrackData:
    trk = MidiTrackData(name="Conductor")
    trk.events.append(_meta(0, 0x03, _text("Conductor")))
    for label, off in slots:
        for t, us in line["tempos"]:
            trk.events.append(_meta(off + t, 0x51, int(us).to_bytes(3, "big")))
        for t, (num, den) in line["sigs"]:
            trk.events.append(_meta(off + t, 0x58, bytes([num, int(math.log2(den)), 24, 8])))
        if label:
            trk.events.append(_meta(off, 0x06, _text(label)))
    return trk


def _build_midi(line: dict, slots_plan) -> Tuple[MidiData, dict]:
    """slots_plan: [(etiqueta, offset, [(key, notas)])]. Una pista por instrumento."""
    mid = MidiData(fmt=1, tpb=line["tpb"])
    mid.tracks.append(_conductor(line, [(lab if len(slots_plan) > 1 else "", off)
                                        for lab, off, _ in slots_plan]))
    order, chan = [], {}
    for _lab, _off, parts in slots_plan:
        for key, _n in parts:
            if key not in chan:
                chan[key] = CHANNELS[len(order) % len(CHANNELS)]
                order.append(key)
    tracks = {}
    for key in order:
        t = MidiTrackData(name=TRACK_NAME.get(key, ATLAS[key]["es"] if key in ATLAS else INSTRUMENT_DB[key]["es"]))
        t.events.append(_meta(0, 0x03, _text(t.name)))
        t.programs, t.channels = [], [chan[key]]
        tracks[key] = t
    for _lab, off, parts in slots_plan:
        for key, notes in parts:
            t, ch = tracks[key], chan[key]
            pc = bytes([0xC0 | ch, GM_PROGRAM.get(key, 0)])
            t.events.append(MidiEvent(abs=off, kind="channel", channel=ch, data=pc))
            t.programs.append(GM_PROGRAM.get(key, 0))
            for st, en, p, v in notes:
                t.events.append(MidiEvent(abs=off + st, kind="note_on", channel=ch, pitch=p, vel=v))
                t.events.append(MidiEvent(abs=off + en, kind="note_off", channel=ch, pitch=p, vel=0))
    mid.tracks.extend(tracks[k] for k in order)
    mid.tempo_map = list(line["tempos"])
    mid.timesig_map = [(t, n, d) for t, (n, d) in line["sigs"]]
    return mid, chan


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA: VARIANTES
# ══════════════════════════════════════════════════════════════════════════════

def _variant_defs(line: dict, combos: Optional[List[str]], role: Optional[str] = None):
    """Lista de variantes: la `solo` y las combinaciones elegidas (o las mejores)."""
    if combos:
        ids = []
        for cid in combos:
            if cid not in COMBO_BY_ID:
                raise ValueError(f"combinación desconocida: {cid!r} "
                                 f"(ver `python orchestral_atlas.py combos`; p.ej. {', '.join(list(COMBO_BY_ID)[:4])}…)")
            if cid not in ids:
                ids.append(cid)
        return [_plan_combo(line, COMBO_BY_ID[c]) for c in ids]
    sug = _suggest(line, role)
    good = [s for s in sug if s["fits_all"]] or sug
    return [_plan_combo(line, COMBO_BY_ID[s["id"]]) for s in good[:MAX_DEFAULT]]


def make_variants(path: str, track: int, out_dir: str, combos=None, bars=None) -> dict:
    """Escribe un MIDI por variante, `00_comparar_todas.mid` y `ficha_variantes.txt`
    en `out_dir`. `combos`: lista de ids (o 'id1,id2'); None = las mejores del atlas.
    Devuelve un dict (tool, version, file, overall, summary, variants, …)."""
    if isinstance(combos, str):
        combos = [c.strip() for c in combos.split(",") if c.strip()]
    line = load_line(path, track, bars)
    plans = _variant_defs(line, combos or None)
    os.makedirs(out_dir, exist_ok=True)
    gap = line["bar_ticks"]
    variants, slots = [], []
    offset = 0
    # variante `solo`: la línea sin doblar, con su dinámica original
    solo_plans = [dict(combo=dict(id="solo", name=f"Solo {line['src_es']} (sin doblar)",
                                  insts=[line["src_key"]], relation="sin doblar",
                                  effect="La línea tal como está escrita, sin doblar: tu punto de referencia.",
                                  use="Compara cada doblaje con esta versión para ver qué añade.",
                                  caution="Es la línea original; no incluye ningún cambio.",
                                  roles=[line["role_guess"]], register=[line["register"]],
                                  dynamic=[line["dynamic"]], blend=3),
                       insts=None)]
    entries = [("solo", solo_plans[0]["combo"], None)] + [(p["combo"]["id"], p["combo"], p) for p in plans]
    for idx, (cid, combo, plan) in enumerate(entries, 1):
        parts, info = [], []
        if cid == "solo":
            key = line["src_key"]
            prof = INSTRUMENT_DB[key]
            lo, hi = prof["range"]
            folded = 0
            notes = []
            for st, en, p, v in line["notes"]:
                q = p
                if not lo <= q <= hi:
                    q = _fold_pitch(q, lo, hi)
                    folded += 1
                notes.append((st, en, q, v))
            notes = _reduce(notes, key, line["tpb"])
            parts.append((key, notes))
            info.append(dict(key=key, es=line["src_es"], shift_oct=0, weight=1.0, n_folded=folded,
                             n_notes=len(notes), fits=(folded == 0), program=GM_PROGRAM.get(key, 0),
                             zone=zone_of(key, int(round(line["median"])))["label"],
                             zone_power=zone_of(key, int(round(line["median"])))["zone_power"]
                             if False else zone_of(key, int(round(line["median"])))["power"],
                             zone_character=zone_of(key, int(round(line["median"])))["character"],
                             vel_mean=round(sum(n[3] for n in notes) / len(notes), 1)))
        else:
            for pi in plan["insts"]:
                notes, folded, trimmed = _render(line, pi["key"], pi["shift_oct"], pi["weight"], True)
                if not notes:
                    continue
                parts.append((pi["key"], notes))
                info.append(dict(key=pi["key"], es=pi["es"], shift_oct=pi["shift_oct"],
                                 weight=pi["weight"], n_folded=folded, n_notes=len(notes),
                                 n_trimmed=trimmed, fits=(folded == 0),
                                 program=GM_PROGRAM.get(pi["key"], 0), zone=pi["zone"],
                                 zone_power=pi["zone_power"], zone_character=pi["zone_character"],
                                 vel_mean=round(sum(n[3] for n in notes) / len(notes), 1)))
        fname = f"{idx:02d}_{cid}.mid"
        mid, chan = _build_midi(line, [(combo["name"], 0, parts)])
        write_midi(mid, os.path.join(out_dir, fname))
        for it in info:
            it["channel"] = chan[it["key"]] + 1          # 1-16 como lo ve el usuario
        variants.append(dict(id=cid, name=combo["name"], file=fname, relation=combo["relation"],
                             effect=combo["effect"], use=combo["use"], caution=combo["caution"],
                             roles=list(combo["roles"]), instruments=info,
                             folded_total=sum(i["n_folded"] for i in info)))
        slots.append((combo["name"], offset, parts))
        offset += line["len_ticks"] + gap
    cmid, _ch = _build_midi(line, slots)
    cfile = "00_comparar_todas.mid"
    write_midi(cmid, os.path.join(out_dir, cfile))
    ffile = "ficha_variantes.txt"
    _ensure_parent(os.path.join(out_dir, ffile))
    Path(out_dir, ffile).write_text(_ficha(line, variants), encoding="utf-8")
    n_notes = sum(i["n_notes"] for v in variants for i in v["instruments"])
    n_fold = sum(i["n_folded"] for v in variants for i in v["instruments"])
    overall = round(1.0 - n_fold / n_notes, 3) if n_notes else None
    return dict(tool=TOOL, version=VERSION, file=path, overall=overall,
                summary=dict(track=track, track_name=line["track_name"], instrument=line["src_key"],
                             bars=f"{line['bars'][0]}-{line['bars'][1]}", n_variants=len(variants),
                             n_notes_line=len(line["notes"]), n_notes_written=n_notes,
                             n_notes_folded=n_fold, variants_with_folding=sum(1 for v in variants if v["folded_total"]),
                             out_dir=out_dir),
                line=dict(lo=line["lo"], hi=line["hi"], median=line["median"], register=line["register"],
                          dynamic=line["dynamic"], role_guess=line["role_guess"],
                          lowest=pitch_name(line["lo"]), highest=pitch_name(line["hi"])),
                variants=variants, compare_file=cfile, ficha_file=ffile, out_dir=out_dir)


def _listen(relation: str) -> str:
    for k in ("dos octavas", "octava", "unísono", "acordes", "capas"):
        if k in relation:
            return LISTEN[k]
    return LISTEN_DEFAULT


def _ficha(line: dict, variants: list) -> str:
    L = []
    L.append(f"FICHA DE VARIANTES DE DOBLAJE — {TOOL} v{VERSION}")
    L.append("=" * 64)
    L.append(f"Línea: pista {line['track']} «{line['track_name']}» ({line['src_es']}), "
             f"compases {line['bars'][0]}-{line['bars'][1]}.")
    L.append(f"Notas: {len(line['notes'])} · ámbito {pitch_name(line['lo'])}–{pitch_name(line['hi'])} · "
             f"registro {REGISTER_ES[line['register']]} · velocity media {line['mean_vel']:.0f}.")
    L.append("")
    L.append(_wrap("AVISO IMPORTANTE: el balance de velocidades es ORIENTATIVO. Sale de una "
                   "estimación por familia, registro y velocity (la «proyección» de "
                   "orchestral_atlas), no del sonido de tus librerías. Ajústalo por oído "
                   "con las librerías reales: baja o sube el volumen/expresión de cada pista "
                   "hasta que la mezcla suene como quieres.", ""))
    L.append("")
    L.append("CÓMO USARLO EN FL STUDIO")
    L.append(_wrap("1) Arrastra 00_comparar_todas.mid al Piano roll/Channel rack: cada pista "
                   "es un instrumento (el nombre de la pista te dice cuál) y las variantes "
                   "suenan una tras otra con un compás de silencio entre ellas. Asigna a "
                   "cada canal el patch de tu librería y escucha de seguido.", "  "))
    L.append(_wrap("2) Cuando una variante te convenza, abre su propio .mid (un fichero por "
                   "variante) y llévalo a tu proyecto. Todo está en canales distintos y "
                   "ninguno usa el canal 10 (percusión GM).", "  "))
    L.append(_wrap("3) Compara siempre con la variante «solo»: te dice qué aporta cada doblaje.", "  "))
    L.append("")
    for n, v in enumerate(variants, 1):
        L.append("-" * 64)
        L.append(f"{n:02d}  {v['name']}   [{v['id']}]   → {v['file']}")
        L.append("-" * 64)
        for i in v["instruments"]:
            extra = (f" · {i['n_folded']} notas plegadas a otra octava (cambia el contorno)"
                     if i["n_folded"] else "")
            L.append(f"  · {i['es']:<18} canal {i['channel']:>2} · GM {i['program']:>3} · "
                     f"{_oct_txt(i['shift_oct'])} · zona {i['zone']} · vel. media {i['vel_mean']:.0f}{extra}")
        L.append(_wrap(f"Efecto: {v['effect']}", ""))
        cautions = [v["caution"]]
        for i in v["instruments"]:
            if i["zone_power"] <= 1 and v["id"] != "solo":
                cautions.append(f"{i['es']} queda en su zona más débil ({i['zone']}: "
                                f"{i['zone_character']}); puede perderse.")
            if i["n_folded"]:
                cautions.append(f"{i['es']}: {i['n_folded']} notas no caben en su rango y se "
                                f"han plegado a otra octava.")
        L.append(_wrap("Precaución: " + " ".join(cautions), ""))
        L.append(_wrap("Rol: " + ", ".join(ROLE_ES.get(r, r) for r in v["roles"]) + ".", ""))
        L.append(_wrap("Qué escuchar: " + _listen(v["relation"]), ""))
        L.append("")
    L.append(_wrap("Recuerda: la ficha describe tendencias del catálogo, no tu sonido concreto. "
                   "Si una variante suena distinta a lo anotado, fíate de tu oído.", ""))
    return "\n".join(L) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def print_suggest(line: dict, sug: list, role: str):
    print()
    print(f"  {_c('B')}Doubling Lab · sugerencias{_c('R')}")
    print(f"  Línea: pista {line['track']} «{line['track_name']}» ({line['src_es']}), "
          f"compases {line['bars'][0]}-{line['bars'][1]}")
    print(f"  {len(line['notes'])} notas · ámbito {pitch_name(line['lo'])}–{pitch_name(line['hi'])} · "
          f"registro {REGISTER_ES[line['register']]} · dinámica {line['dynamic']} "
          f"(vel. media {line['mean_vel']:.0f}) · rol: {ROLE_ES[role]}")
    print()
    if not sug:
        print("  (ninguna combinación del atlas encaja)")
    for k, s in enumerate(sug, 1):
        mark = _c("GRN") + "✔" if s["fits_all"] else _c("YEL") + "⚑"
        print(f"  {_c('B')}{k:>2}. {s['name']}{_c('R')}  [{s['id']}]  {mark}{_c('R')}  "
              f"afinidad {s['score']:.1f}")
        print(_wrap(s["effect"], "      "))
        for i in s["insts"]:
            ok = (f"{_c('GRN')}caben todas{_c('R')}" if i["fits"] else
                  f"{_c('YEL')}{i['n_out']} notas fuera de rango{_c('R')} (se plegarían)")
            print(f"      {i['es']:<18} {_oct_txt(i['shift_oct']):<15} "
                  f"{i['sounding'][0]}–{i['sounding'][1]}  {ok} · zona {i['zone']}")
        print(_wrap(f"Ojo: {s['caution']}", "      "))
        print(f"      {_c('G')}{s['why']}{_c('R')}")
        print()
    print(_wrap("Siguiente paso: `variants` genera un MIDI por variante para escucharlas con tus "
                "librerías; el balance es orientativo y se ajusta por oído.", "  "))
    print()


def print_variants(res: dict):
    s = res["summary"]
    print()
    print(f"  {_c('B')}Doubling Lab · variantes{_c('R')}")
    print(f"  Línea: pista {s['track']} «{s['track_name']}» ({s['instrument']}), compases {s['bars']}")
    print(f"  Carpeta: {res['out_dir']}")
    print()
    for v in res["variants"]:
        warn = f"  {_c('YEL')}⚑ {v['folded_total']} notas plegadas{_c('R')}" if v["folded_total"] else ""
        print(f"  {v['file']:<40} {', '.join(i['es'] for i in v['instruments'])}{warn}")
    print(f"  {res['compare_file']:<40} todas seguidas, un compás de silencio entre ellas")
    print(f"  {res['ficha_file']:<40} efecto, precaución, rol y qué escuchar")
    print()
    print(_wrap("El balance de velocidades es orientativo: ajústalo por oído con las "
                "librerías reales.", "  "))
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
        ev += [(0, 0, b"\xFF\x51\x03" + (500000).to_bytes(3, "big")),
               (0, 0, b"\xFF\x58\x04\x04\x02\x18\x08"),
               (3840, 0, b"\xFF\x51\x03" + (400000).to_bytes(3, "big"))]
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


def _synthetic(path):
    tpb, bar, nb = 480, 1920, 6
    mel = [[74, 76, 79, 77], [76, 74, 72, 76], [79, 77, 76, 74], [72, 74, 76, 79]]
    bass = [[40, 43, 47, 45], [38, 41, 45, 43]]
    wide = [36, 52, 68, 84, 100, 84, 68, 52]
    tr = [_trk("Conductor", None, 0, [], tempo=True),
          _trk("Flauta", 73, 0, [(b * bar + i * tpb, tpb, p, 70 + 4 * i)
                                 for b in range(nb) for i, p in enumerate(mel[b % 4])]),
          _trk("Violonchelo", 42, 1, [(b * bar + i * tpb, tpb, p, 80)
                                      for b in range(nb) for i, p in enumerate(bass[b % 2])]),
          _trk("Arpa", 46, 2, [(b * bar + i * (tpb // 2), tpb // 2, p, 64)
                               for b in range(nb) for i, p in enumerate(wide)]),
          _trk("Vacia", 71, 3, [])]
    hdr = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tr).to_bytes(2, "big") + tpb.to_bytes(2, "big")
    _ensure_parent(path)
    Path(path).write_bytes(hdr + b"".join(tr))


def _check_dir(d, errs, src_tpb, tag):
    """Relee con playability_auditor: rango, programa, canal, nombre, tempo."""
    for f in sorted(os.listdir(d)):
        if not f.endswith(".mid"):
            continue
        m = read_midi(os.path.join(d, f))
        if m.tpb != src_tpb:
            errs.append(f"{tag}/{f}: tpb distinto")
        if m.tempo_map[0][1] != 500000 or m.timesig_map[0][1:] != (4, 4):
            errs.append(f"{tag}/{f}: tempo/compás iniciales no conservados")
        for t in m.tracks[1:]:
            notes = extract_notes(t)
            if not notes:
                errs.append(f"{tag}/{f}: pista {t.name!r} sin notas")
                continue
            key = detect_instrument(t)
            prof = INSTRUMENT_DB[key]
            lo, hi = prof["range"]
            bad = [n.pitch for n in notes if not lo <= n.pitch <= hi]
            if bad:
                errs.append(f"{tag}/{f}: {t.name} con {len(bad)} notas fuera de rango ({key})")
            if 9 in t.channels or any(ev.channel == 9 for ev in t.events if ev.kind != "meta"):
                errs.append(f"{tag}/{f}: {t.name} usa el canal 10")
            if len(t.programs) < 1 or any(GM_PROGRAM.get(key) != p for p in t.programs if key in GM_PROGRAM
                                          and key not in ("cuerdas",) and False):
                errs.append(f"{tag}/{f}: {t.name} sin program change")
            if key in GM_PROGRAM and t.programs and key != "viento_metal_gen" and t.programs[0] != GM_PROGRAM[key] \
                    and not (key == "cuerdas" and t.programs[0] == 48):
                errs.append(f"{tag}/{f}: {t.name} program {t.programs[0]} ≠ GM {GM_PROGRAM[key]}")


def cmd_selftest(a):
    errs = []
    tmp = tempfile.mkdtemp(prefix="doublab_")
    try:
        src = os.path.join(tmp, "src.mid")
        _synthetic(src)
        # 1) suggest sobre la flauta (melodía)
        sug = suggest_doublings(src, 1)
        if not sug or not all(k in sug[0] for k in ("id", "insts", "score", "fits_all")):
            errs.append("suggest no devuelve combinaciones con la estructura esperada")
        elif "melody" not in sug[0]["roles"]:
            errs.append("suggest para melodía propone una combinación que no es de melodía")
        elif not all("shift_oct" in i and "fits" in i for i in sug[0]["insts"]):
            errs.append("suggest no informa de transposición y rango por instrumento")
        if suggest_doublings(src, 2)[0]["role_used"] != "bass":
            errs.append("el rol estimado del violonchelo debería ser bajo")
        # 2) variantes por defecto sobre la flauta
        out1 = os.path.join(tmp, "v1")
        r = make_variants(src, 1, out1)
        names = set(os.listdir(out1))
        if not {"00_comparar_todas.mid", "ficha_variantes.txt", "01_solo.mid"} <= names:
            errs.append(f"faltan ficheros de salida ({sorted(names)})")
        if len(r["variants"]) < 2:
            errs.append("make_variants debería generar solo + al menos una combinación")
        ficha = Path(out1, "ficha_variantes.txt").read_text(encoding="utf-8") if "ficha_variantes.txt" in names else ""
        if "ORIENTATIVO" not in ficha or "oído" not in ficha:
            errs.append("la ficha no avisa de que el balance es orientativo y debe ajustarse por oído")
        for k in ("Efecto:", "Precaución:", "Rol:", "Qué escuchar:"):
            if k not in ficha:
                errs.append(f"la ficha no contiene «{k}»")
        _check_dir(out1, errs, 480, "flauta")
        # tempo cambiante conservado en el comparador y compás de silencio
        cm = read_midi(os.path.join(out1, "00_comparar_todas.mid"))
        if len(cm.tempo_map) < 2 or (3840, 400000) not in cm.tempo_map:
            errs.append("el comparador no conserva el cambio de tempo del original")
        n_var = len(r["variants"])
        line = load_line(src, 1)
        last_end = max(n.end for t in cm.tracks[1:] for n in extract_notes(t))
        if last_end < (n_var - 1) * (line["len_ticks"] + line["bar_ticks"]):
            errs.append("el comparador no deja un compás entre variantes")
        # 3) línea grave: la flauta debe subir octavas y todo caber en el rango
        out2 = os.path.join(tmp, "v2")
        r2 = make_variants(src, 2, out2, combos="flauta_clarinete,chelos_contrabajos_oct,fagot_chelos_bajo")
        _check_dir(out2, errs, 480, "chelo")
        fl = [i for v in r2["variants"] if v["id"] == "flauta_clarinete" for i in v["instruments"] if i["key"] == "flauta"]
        if not fl or fl[0]["shift_oct"] < 1:
            errs.append("la línea grave debería subir octavas para la flauta")
        # 4) línea de ámbito enorme: se pliega y sigue dentro del rango
        out3 = os.path.join(tmp, "v3")
        r3 = make_variants(src, 3, out3, combos=["flauta_clarinete", "arpa_pizz", "celesta_flauta"])
        _check_dir(out3, errs, 480, "arpa")
        if r3["summary"]["n_notes_folded"] == 0:
            errs.append("una línea de 5 octavas debería requerir plegado en alguna variante")
        # 5) subconjunto de compases
        out4 = os.path.join(tmp, "v4")
        r4 = make_variants(src, 1, out4, combos=["flauta_oboe"], bars=(2, 3))
        c4 = read_midi(os.path.join(out4, "02_flauta_oboe.mid"))
        st = [n.start for t in c4.tracks[1:] for n in extract_notes(t)]
        if min(st) != 0 or max(st) >= 2 * 1920:
            errs.append("--bars no traslada el fragmento al compás 1")
        if r4["summary"]["bars"] != "2-3":
            errs.append("el resumen no recoge los compases")
        # 6) casos límite
        for label, fn in (("pista vacía", lambda: make_variants(src, 4, os.path.join(tmp, "x"))),
                          ("pista inexistente", lambda: suggest_doublings(src, 9)),
                          ("pista negativa", lambda: suggest_doublings(src, -1)),
                          ("compases fuera de la obra", lambda: suggest_doublings(src, 1, bars=(50, 60))),
                          ("bars sin notas", lambda: suggest_doublings(src, 2, bars=(1, 1)) and
                           make_variants(src, 2, os.path.join(tmp, "y"), bars="7-8")),
                          ("combo desconocida", lambda: make_variants(src, 1, os.path.join(tmp, "z"), combos=["no_existe"])),
                          ("rol desconocido", lambda: suggest_doublings(src, 1, role="solista")),
                          ("bars inválido", lambda: suggest_doublings(src, 1, bars="9-3"))):
            try:
                fn()
                errs.append(f"{label}: debería fallar con ValueError")
            except ValueError:
                pass
        try:
            suggest_doublings(os.path.join(tmp, "nada.mid"), 0)
            errs.append("fichero inexistente: debería fallar")
        except FileNotFoundError:
            pass
        if os.path.exists(os.path.join(tmp, "x")) and os.listdir(os.path.join(tmp, "x")):
            errs.append("una pista vacía no debería dejar ficheros")
        # 7) equilibrio: la velocity invertida iguala la proyección pedida
        for key, p in (("flauta", 80), ("violin", 70), ("trompa", 55)):
            tgt = projection(key, p, 90)
            v = _vel_for(key, p, tgt)
            if abs(projection(key, p, v) - tgt) > 0.04:
                errs.append(f"el equilibrio de velocidades no se aproxima a la proyección ({key})")
        if not all(k in GM_PROGRAM for c in COMBOS for k in c["insts"]):
            errs.append("hay instrumentos del atlas sin program GM")
        if _parse_bars("5-8") != (5, 8):
            errs.append("parse de --bars incorrecto")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for e in errs:
        print(f"  ✗ {e}")
    print(f"  {'OK' if not errs else 'FALLO'} · selftest de {TOOL}")
    return 0 if not errs else 3


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _emit_json(res: dict, dest: Optional[str]) -> bool:
    """Escribe el sidecar. Devuelve True si fue a stdout ('-')."""
    if not dest:
        return False
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if dest == "-":
        print(txt)
        return True
    _ensure_parent(dest)
    Path(dest).write_text(txt, encoding="utf-8")
    print(f"  {_c('CYA')}JSON: {dest}{_c('R')}\n")
    return False


def cmd_suggest(a):
    line = load_line(a.midi, a.track, a.bars)
    if a.role is not None and a.role not in ROLES:
        raise ValueError(f"rol desconocido: {a.role} (válidos: {', '.join(ROLES)})")
    sug = _suggest(line, a.role)
    role = a.role or line["role_guess"]
    n_fit = sum(1 for s in sug if s["fits_all"])
    res = dict(tool=TOOL, version=VERSION, file=a.midi,
               overall=round(n_fit / len(sug), 3) if sug else None,
               summary=dict(track=a.track, track_name=line["track_name"], instrument=line["src_key"],
                            bars=f"{line['bars'][0]}-{line['bars'][1]}", role=role,
                            n_suggestions=len(sug), n_fit_all=n_fit),
               line=dict(lo=line["lo"], hi=line["hi"], median=line["median"], register=line["register"],
                         dynamic=line["dynamic"], mean_vel=line["mean_vel"], role_guess=line["role_guess"]),
               suggestions=sug)
    if a.json != "-":
        print_suggest(line, sug, role)
    _emit_json(res, a.json)
    return 0


def cmd_variants(a):
    combos = [c.strip() for c in a.combos.split(",") if c.strip()] if a.combos else None
    if a.combos is not None and not combos:
        raise ValueError("--combos vacío (usa ids separados por comas)")
    res = make_variants(a.midi, a.track, a.out_dir, combos=combos, bars=a.bars)
    if a.json != "-":
        print_variants(res)
    _emit_json(res, a.json)
    if a.strict and res["summary"]["variants_with_folding"]:
        return 3
    return 0


def main():
    global _USE_COLOR
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true")
    ap = argparse.ArgumentParser(
        prog="doubling_lab.py",
        description="Laboratorio de doblajes: sugiere combinaciones y genera variantes MIDI para escuchar.")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("suggest", parents=[common], help="Lista las combinaciones de doblaje que encajan")
    p.add_argument("midi")
    p.add_argument("--track", type=int, required=True, help="Pista de la línea (0-based)")
    p.add_argument("--bars", help="Rango de compases A-B (1-based)")
    p.add_argument("--role", choices=ROLES, help="Rol de la línea (por defecto, estimado)")
    p.add_argument("--json", help="Sidecar JSON ('-' = stdout)")
    p.set_defaults(fn=cmd_suggest)

    p = sp.add_parser("variants", parents=[common], help="Escribe un MIDI por variante + comparador + ficha")
    p.add_argument("midi")
    p.add_argument("--track", type=int, required=True, help="Pista de la línea (0-based)")
    p.add_argument("--bars", help="Rango de compases A-B (1-based)")
    p.add_argument("--out-dir", default="./doubling_variants", help="Carpeta de salida")
    p.add_argument("--combos", help="Ids de combinación del atlas separados por comas")
    p.add_argument("--strict", action="store_true", help="rc=3 si alguna variante necesita plegar notas")
    p.add_argument("--json", help="Sidecar JSON ('-' = stdout)")
    p.set_defaults(fn=cmd_variants)

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
