#!/usr/bin/env python3
"""
gesture_lab.py — Gesture Lab: trayectorias, gestos y generación a partir de
CÓMO EVOLUCIONA una pieza MIDI (registro, bajo, armonía, tonalidad, células)
(versión 1.0).

──────────────────────────────────────────────────
Fundamento
──────────────────────────────────────────────────
Los detectores de repetición (ostinato_detector, harvester, form_analyzer)
buscan material que VUELVE. Muchas sensaciones musicales —estar «subiendo sin
llegar», desorientación, estancamiento— no dependen de que algo se repita
nota a nota sino de cómo cambian en el tiempo unos pocos rasgos: el registro
de la melodía, la altura del bajo, la estabilidad tonal, cuánto choca la
melodía con el bajo, si hay terceras en los acordes… Esta herramienta:

  1. Extrae TRAYECTORIAS: una serie por compás para cada rasgo.
  2. Busca FORMAS sobre esas series (rampa, diente de sierra, meseta,
     constante, periodicidad) y gestos estructurales (bucle armónico, pedal
     que choca con el bajo, relanzamiento por pulso, célula bloqueada,
     ráfaga, deriva tonal, acordes sin tercera).
  3. Describe lo hallado como un JSON de «gesto» editable (spec) que el
     generador consume: analizar → editar el spec → generar otra pieza.
  4. Descubre formas SIN plantilla (discover): agrupa ventanas de las series
     de uno o muchos MIDIs y devuelve las formas recurrentes, que se guardan
     en una biblioteca reutilizable por analyze.

Mide ESTRUCTURA, no percepción: que una forma produzca una sensación
concreta es una hipótesis que hay que contrastar con el oído. Trabaja sobre
la transcripción: sin timbre, dinámica expresiva ni letra.

──────────────────────────────────────────────────
Modos de uso
──────────────────────────────────────────────────
  analyze    Trayectorias + gestos de un MIDI (informe y JSON)
  discover   Formas recurrentes en uno o varios MIDIs (+ biblioteca)
  spec       Spec de gesto desde un MIDI, un análisis .json o una plantilla
  generate   MIDI nuevo desde un spec (con --verify: vuelve a analizarlo)
  selftest   Pruebas internas: detectores, ida y vuelta, discover

──────────────────────────────────────────────────
Opciones
──────────────────────────────────────────────────
  analyze
    midi MIDI
    --melody N / --bass N       índice de voz (ver «Voces» del informe)
    --bar-beats F               negras por compás (defecto: del compás MIDI)
    --key-window N              compases de la ventana tonal (defecto 2)
    --min-ramp N                compases mínimos de una rampa (defecto 4)
    --min-lock N                repeticiones mínimas de célula (defecto 3)
    --min-score F               descarta gestos por debajo de esta puntuación
                                (defecto 0.55)
    --features LISTA            rasgos sobre los que buscar formas (por defecto
                                solo registro, bajo, densidad y nitidez tonal;
                                los de tipo proporción son ruidosos)
    --library FILE              biblioteca de formas (de discover)
    --json FILE                 guarda el análisis completo
    --no-color

  discover
    paths PATHS…                ficheros o carpetas con .mid
    --mode auto|corpus|within|both
    --features LISTA   --lengths 4,8   --k N   --stride N
    --thr F                     correlación mínima de recurrencia (0.92)
    --top N   --null N          nº de barajados de control (defecto 1)
    --emit-library FILE         guarda las formas como biblioteca
    --json FILE   --seed N

  spec
    [FUENTE]                    .mid o análisis .json (o usa --template)
    --template NOMBRE           treadmill | rise_collapse | static_vamp
    --set CLAVE=VALOR           sobrescribe (repetible; claves con puntos)
    --bars N   --key TONICA   --name TXT   --out FILE

  generate
    [SPEC]                      spec .json (o usa --template)
    --out FILE   --seed N   --variations N
    --bars N   --key TONICA   --tempo F
    --verify                    reanaliza la salida y compara con el spec

──────────────────────────────────────────────────
Parámetros (spec)
──────────────────────────────────────────────────
@params v1
[analyze]
midi         file  io=in ext=mid required
--melody     int   min=0  # índice de la voz melódica (defecto: la más aguda)
--bass       int   min=0  # índice de la voz de bajo (defecto: la más grave)
--bar-beats  float min=0.5 unit=tiempos  # Negras por compás (defecto: del compás MIDI)
--key-window int   default=2 min=1 unit=compases
--min-ramp   int   default=4 min=3 unit=compases
--min-lock   int   default=3 min=2
--min-score  float default=0.55 min=0 max=1
--features   str   # Rasgos sobre los que buscar formas, separados por comas
--library    file  io=in ext=json  # Biblioteca de formas generada por discover
--json       file  io=out ext=json
--no-color   bool  default=false
[discover]
paths        str   required  # Ficheros o carpetas con MIDIs
--mode       enum  default=auto values=auto|corpus|within|both
--features   str
--lengths    str   default=4,8  # Longitudes de ventana en compases
--k          int   default=8 min=2  # Clusters por rasgo y longitud (corpus)
--stride     int   default=1 min=1
--thr        float default=0.92 min=0.5 max=1
--top        int   default=10 min=1
--null       int   default=1 min=0
--seed       int   default=0
--emit-library file io=out ext=json
--json       file  io=out ext=json
--no-color   bool  default=false
[spec]
source       str   # MIDI o análisis .json (opcional con --template)
--template   enum  values=treadmill|rise_collapse|static_vamp
--list-templates bool default=false
--out        file  default=gesture.spec.json io=out ext=json
--set        str   # clave=valor (repetible), p. ej. register.slope=3
--bars       int   min=1 unit=compases
--key        str   # Tónica de referencia del spec
--name       str
--melody     int   min=0
--bass       int   min=0
--no-color   bool  default=false
[generate]
spec         file  io=in ext=json  # Spec de gesto (opcional con --template)
--template   enum  values=treadmill|rise_collapse|static_vamp
--out        file  default=gesture.mid io=out ext=mid
--seed       int   default=0
--variations int   default=1 min=1
--bars       int   min=1 unit=compases
--key        str   # Transpone el spec a esta tónica
--tempo      float min=20 max=300
--verify     bool  default=false
--no-color   bool  default=false
[selftest]
--verbose    bool  default=false
--no-color   bool  default=false
@end

──────────────────────────────────────────────────
Ejemplos de uso
──────────────────────────────────────────────────
    python gesture_lab.py analyze obra.mid
    python gesture_lab.py analyze obra.mid --json obra.gestos.json
    python gesture_lab.py spec obra.mid --out obra.spec.json
    python gesture_lab.py spec --template rise_collapse --set register.slope=3
    python gesture_lab.py generate obra.spec.json --out nueva.mid --key G --verify
    python gesture_lab.py discover carpeta_midis/ --emit-library formas.json
    python gesture_lab.py analyze otra.mid --library formas.json
    python gesture_lab.py selftest

  RASGOS (una serie por compás; ver FEATURES en el código):
    reg_mean reg_max reg_min reg_range   registro de la melodía (semitonos)
    bass_low                             nota más grave del bajo
    density rest_ratio                   notas por compás · proporción de silencio
    leap_ratio desc_ratio turn_ratio     saltos · descensos · cambios de dirección
    cell_fall                            pulsos que empiezan arriba y acaban abajo
    clash                                melodía a 2ª menor/7ª mayor/tritono del bajo
    third_ratio                          el acorde del compás tiene tercera
    key_r                                nitidez tonal (Krumhansl) en la ventana
    vel_mean                             velocidad media

  GESTOS DETECTADOS (campo «shape»):
    ramp / fall / ramp_reset / fall_reset / plateau / constant / periodic
                                         formas sobre cualquier rasgo
    loop            bucle armónico (periodo de las raíces del bajo)
    pedal           nota(s) grave(s) que se repiten mientras el bajo cambia
    micro_cycle     el pulso se relanza: ataque agudo y descenso, y salto arriba
    cell_lock       la misma célula repetida ≥ N pulsos seguidos
    burst           pulso con más notas que el resto (aceleración puntual)
    tonal_wander    la tonalidad estimada cambia de ventana a ventana
    open_fifths     acordes sin tercera
    lib:NOMBRE      forma de la biblioteca (--library)

  ESQUEMA DEL SPEC (campos editables; --set usa claves con puntos):
    bars, time_signature, tempo, velocity, key{tonic_pc,tonic,mode}, scale_pcs,
    harmony{loop_bars,roots_pc,third_prob,third_bars{bar:semitonos},voicing},
    bass{floor,low_override{compás:altura}},
    register{base,slope,period,lift,ceiling,floor_mean,jitter},
    melody_range{lo,hi}, cell{notes_per_pulse,fall_prob,spread,dur_frac,
    chord_bias}, pedal{pitches,prob,bars}, lock{bars,reps,cell_pc_offsets,
    cell_steps}, burst{bars,pulse,n}. pedal y lock admiten una LISTA de
    bloques (varios pedales/bloqueos). Un bloque null (o ausente) lo desactiva.
    register: la media por compás es base + lift·ciclo + slope·posición,
    recortada a [floor_mean, ceiling]; period reinicia la posición.

  LIMITACIONES: usa el primer compás MIDI (--bar-beats para forzarlo); la
  voz melódica es la línea superior (skyline) y el bajo la más grave; con
  una sola voz polifónica separa por altura. Las series cortas (< 8 compases)
  dan formas poco fiables.

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso o prueba fallida · 2 fichero ilegible

  API:  from gesture_lab import (load_piece, analyze_piece, build_spec,
                                 generate_piece, write_midi)

──────────────────────────────────────────────────
Dependencias
──────────────────────────────────────────────────
  mido · numpy (solo discover)
"""

import argparse
import bisect
import glob
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "1.0"
NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_SPARK = "▁▂▃▄▅▆▇█"


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def _ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _pcname(pc):
    return NAMES[int(pc) % 12]


def _pitchname(p):
    return "%s%d" % (NAMES[int(p) % 12], int(p) // 12 - 1)


def _name_to_pc(s):
    s = str(s).strip()
    if s.lstrip("-").isdigit():
        return int(s) % 12
    t = s[0].upper() + s[1:].replace("♭", "b").replace("♯", "#")
    alias = {"C#": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10, "Cb": 11, "E#": 5,
             "Fb": 4, "B#": 0}
    if t in alias:
        return alias[t]
    if t in NAMES:
        return NAMES.index(t)
    raise ValueError("tónica no reconocida: %r" % s)


def _mean(v):
    return sum(v) / len(v) if v else 0.0


def _median(v):
    s = sorted(v)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _std(v):
    if len(v) < 2:
        return 0.0
    m = _mean(v)
    return math.sqrt(sum((a - m) ** 2 for a in v) / len(v))


def _clamp(x, a, b):
    return max(a, min(b, x))


def _corr(a, b):
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    ma, mb = _mean(a), _mean(b)
    sa = sum((x - ma) ** 2 for x in a)
    sb = sum((y - mb) ** 2 for y in b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sa * sb)


def _linfit(y):
    n = len(y)
    if n < 2:
        return 0.0, (y[0] if y else 0.0), 0.0
    mx, my = (n - 1) / 2.0, _mean(y)
    sxx = sum((i - mx) ** 2 for i in range(n))
    sxy = sum((i - mx) * (y[i] - my) for i in range(n))
    slope = sxy / sxx
    icpt = my - slope * mx
    ss_tot = sum((v - my) ** 2 for v in y)
    ss_res = sum((y[i] - (icpt + slope * i)) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return slope, icpt, max(0.0, r2)


def _spark(x, lo=None, hi=None):
    if not x:
        return ""
    lo = min(x) if lo is None else lo
    hi = max(x) if hi is None else hi
    if hi - lo < 1e-12:
        return _SPARK[0] * len(x)
    return "".join(_SPARK[int(_clamp((v - lo) / (hi - lo), 0, 1) * 7 + 0.5)]
                   for v in x)


def _round(v, nd=3):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, (list, tuple)):
        return [_round(a, nd) for a in v]
    if isinstance(v, dict):
        return {k: _round(a, nd) for k, a in v.items()}
    return v


# ══════════════════════════════════════════════════
#  TONALIDAD (Krumhansl-Kessler)
# ══════════════════════════════════════════════════
_KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _key_from_hist(h):
    """((r, tónica, modo), segunda) o None."""
    if sum(h) <= 0:
        return None
    res = []
    for t in range(12):
        for mode, prof in (("major", _KK_MAJ), ("minor", _KK_MIN)):
            rot = [prof[(i - t) % 12] for i in range(12)]
            res.append((_corr(h, rot), t, mode))
    res.sort(reverse=True)
    return res[0], res[1]


def _key_col(tonic, mode):
    """Colección diatónica: tónica de la mayor relativa."""
    return tonic if mode == "major" else (tonic + 3) % 12


# ══════════════════════════════════════════════════
#  CARGA DEL MIDI
# ══════════════════════════════════════════════════
class Piece(object):
    """Pieza cargada. Notas = (onset_en_negras, duración, altura, velocidad)."""


def _skyline(notes, mode="max"):
    groups, order = {}, []
    for n in sorted(notes):
        k = int(round(n[0] * 48))
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(n)
    pick = max if mode == "max" else min
    return [pick(groups[k], key=lambda n: n[2]) for k in order]


def load_piece(path, melody=None, bass=None, bar_beats=None, min_notes=8,
               drums=False):
    try:
        mid = mido.MidiFile(path)
    except Exception as e:  # noqa
        raise IOError("MIDI ilegible (%s): %s" % (path, e))
    tpb = mid.ticks_per_beat
    ts, tempo, ts_found, tempo_found = (4, 4), 500000, False, False
    voices = {}
    for ti, tr in enumerate(mid.tracks):
        tick, on = 0, {}
        for msg in tr:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "time_signature" and not ts_found:
                    ts, ts_found = (msg.numerator, msg.denominator), True
                elif msg.type == "set_tempo" and not tempo_found:
                    tempo, tempo_found = msg.tempo, True
            elif msg.type == "note_on" and msg.velocity > 0:
                if msg.channel == 9 and not drums:
                    continue
                on.setdefault((msg.channel, msg.note), []).append(
                    (tick, msg.velocity))
            elif msg.type in ("note_off", "note_on"):
                st = on.get((msg.channel, msg.note))
                if st:
                    s, v = st.pop(0)
                    voices.setdefault((ti, msg.channel), []).append(
                        (s / float(tpb), (tick - s) / float(tpb), msg.note, v))
    vs = []
    for i, (k, nt) in enumerate(sorted(voices.items())):
        nt = sorted(nt)
        vs.append(dict(idx=i, track=k[0], channel=k[1], notes=nt,
                       mean=_mean([n[2] for n in nt])))
    if not vs:
        raise ValueError("el MIDI no contiene notas")
    big = [v for v in vs if len(v["notes"]) >= min_notes] or vs
    if melody is not None:
        mel_v = vs[melody]
    elif len(big) >= 2:
        mel_v = max(big, key=lambda v: v["mean"])
    else:
        mel_v = big[0]
    others = [v for v in big if v is not mel_v]
    if bass is not None:
        bass_v = vs[bass]
    elif others:
        bass_v = min(others, key=lambda v: v["mean"])
    else:
        bass_v = None

    p = Piece()
    p.path = path
    p.tpb = tpb
    p.time_signature = ts
    p.tempo_bpm = 60000000.0 / tempo
    p.bar_beats = float(bar_beats) if bar_beats else ts[0] * 4.0 / ts[1]
    if ts[1] == 8 and ts[0] in (6, 9, 12):
        p.pulse = 1.5
    else:
        p.pulse = 4.0 / ts[1]
    if p.bar_beats / p.pulse < 1.0:
        p.pulse = p.bar_beats
    p.voices = vs
    p.melody_voice = mel_v["idx"]
    p.bass_voice = bass_v["idx"] if bass_v else None
    p.mel = _skyline(mel_v["notes"], "max")
    if bass_v is not None and bass_v is not mel_v:
        p.bass_all = bass_v["notes"]
        p.harm_all = sorted(n for v in vs if v is not mel_v for n in v["notes"])
    else:
        notes = mel_v["notes"]
        groups = Counter(int(round(n[0] * 48)) for n in notes)
        poly = (sum(1 for c in groups.values() if c >= 2) / float(len(groups))
                if groups else 0.0)
        if poly >= 0.2:
            p.bass_all = _skyline(notes, "min")
            top = set(p.mel)
            p.harm_all = [n for n in notes if n not in top]
        else:
            p.bass_all = []
            p.harm_all = []
    p._bass_on = [n[0] for n in p.bass_all]
    p._bass_maxdur = max([n[1] for n in p.bass_all] or [0.0])
    last = max([n[0] for n in p.mel] + [n[0] for n in p.harm_all])
    p.nbars = int(last / p.bar_beats + 1e-9) + 1
    hist = [0.0] * 12
    for n in p.mel + p.harm_all:
        hist[n[2] % 12] += max(n[1], 0.05)
    ke = _key_from_hist(hist)
    if ke:
        (r, t, m), (r2, t2, m2) = ke
        p.key = dict(tonic_pc=t, tonic=_pcname(t), mode=m, r=r,
                     second=dict(tonic=_pcname(t2), mode=m2, r=r2))
    else:
        p.key = dict(tonic_pc=0, tonic="C", mode="major", r=0.0, second=None)
    return p


def sounding_lowest(p, t):
    """Nota más grave sonando del bajo en el instante t (o la última atacada)."""
    if not p.bass_all:
        return None
    i = bisect.bisect_right(p._bass_on, t + 1e-6)
    best, j = None, i - 1
    while j >= 0 and p._bass_on[j] > t - p._bass_maxdur - 1e-6:
        n = p.bass_all[j]
        if n[0] - 1e-6 <= t < n[0] + n[1] - 1e-6:
            if best is None or n[2] < best:
                best = n[2]
        j -= 1
    if best is None and i - 1 >= 0:
        best = p.bass_all[i - 1][2]
    return best


# ══════════════════════════════════════════════════
#  TRAYECTORIAS (series por compás)
# ══════════════════════════════════════════════════
FEATURES = {
    "reg_mean": dict(unit="st", delta=2.0, label="registro medio"),
    "reg_max": dict(unit="st", delta=2.0, label="nota más aguda"),
    "reg_min": dict(unit="st", delta=2.0, label="nota más grave (melodía)"),
    "reg_range": dict(unit="st", delta=3.0, label="amplitud de registro"),
    "bass_low": dict(unit="st", delta=2.0, label="nota más grave del bajo"),
    "density": dict(unit="notas", delta=2.0, label="notas por compás"),
    "rest_ratio": dict(unit="", delta=0.15, label="proporción de silencio"),
    "leap_ratio": dict(unit="", delta=0.15, label="proporción de saltos"),
    "desc_ratio": dict(unit="", delta=0.15, label="proporción de descensos"),
    "turn_ratio": dict(unit="", delta=0.15, label="cambios de dirección"),
    "cell_fall": dict(unit="", delta=0.15, label="pulsos que descienden"),
    "clash": dict(unit="", delta=0.15, label="choque melodía-bajo"),
    "third_ratio": dict(unit="", delta=0.5, label="acorde con tercera"),
    "key_r": dict(unit="", delta=0.1, label="nitidez tonal"),
    "vel_mean": dict(unit="vel", delta=8.0, label="velocidad media"),
}
FEATURE_ORDER = list(FEATURES.keys())


def _fill(x):
    """Rellena None con el valor previo (o el siguiente). None si todo vacío."""
    if all(v is None for v in x):
        return None
    out, prev = [], None
    for v in x:
        prev = v if v is not None else prev
        out.append(prev)
    nxt = None
    for i in range(len(out) - 1, -1, -1):
        if out[i] is None:
            out[i] = nxt
        else:
            nxt = out[i]
    return [float(v) for v in out]


def _bar_of(p, t):
    return min(p.nbars - 1, max(0, int(t / p.bar_beats + 1e-9)))


def _covered(notes, lo, hi):
    iv = sorted((max(lo, n[0]), min(hi, n[0] + n[1])) for n in notes)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if e <= s:
            continue
        if ce is None:
            cs, ce = s, e
        elif s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        tot += ce - cs
    return tot


def pulse_cells(p):
    """{índice_de_pulso: [notas]} de la melodía."""
    cells = {}
    for n in p.mel:
        cells.setdefault(int(n[0] / p.pulse + 1e-6), []).append(n)
    return cells


def extract_series(p, key_window=2):
    nb, bb = p.nbars, p.bar_beats
    npulse = max(1, int(round(bb / p.pulse)))
    mel_b = [[] for _ in range(nb)]
    iv_b = [[] for _ in range(nb)]
    bass_b = [[] for _ in range(nb)]
    harm_b = [[] for _ in range(nb)]
    for n in p.mel:
        mel_b[_bar_of(p, n[0])].append(n)
    for a, b in zip(p.mel, p.mel[1:]):
        iv_b[_bar_of(p, b[0])].append(b[2] - a[2])
    for n in p.bass_all:
        bass_b[_bar_of(p, n[0])].append(n)
    for n in p.harm_all:
        harm_b[_bar_of(p, n[0])].append(n)

    S = {f: [] for f in FEATURE_ORDER}
    cats = {"bass_pc": [], "key_col": [], "key_tonic": [], "key_mode": []}
    for b in range(nb):
        m = mel_b[b]
        pit = [n[2] for n in m]
        S["reg_mean"].append(_mean(pit) if pit else None)
        S["reg_max"].append(max(pit) if pit else None)
        S["reg_min"].append(min(pit) if pit else None)
        S["reg_range"].append(max(pit) - min(pit) if pit else None)
        S["density"].append(float(len(m)))
        S["vel_mean"].append(_mean([n[3] for n in m]) if m else None)
        lo, hi = b * bb, (b + 1) * bb
        S["rest_ratio"].append(1.0 - _covered(m, lo, hi) / bb)
        iv = iv_b[b]
        nz = [v for v in iv if v != 0]
        S["leap_ratio"].append(sum(1 for v in iv if abs(v) >= 5) / float(len(iv))
                               if iv else None)
        S["desc_ratio"].append(sum(1 for v in nz if v < 0) / float(len(nz))
                               if nz else None)
        sg = [1 if v > 0 else -1 for v in nz]
        S["turn_ratio"].append(
            sum(1 for a, c in zip(sg, sg[1:]) if a != c) / float(len(sg) - 1)
            if len(sg) > 1 else None)
        falls = cells = 0
        for j in range(npulse):
            plo = lo + j * p.pulse
            cell = [n for n in m if plo - 1e-6 <= n[0] < plo + p.pulse - 1e-6]
            if len(cell) >= 2:
                cells += 1
                falls += 1 if cell[0][2] > cell[-1][2] else 0
        S["cell_fall"].append(falls / float(cells) if cells else None)
        bl = min([n[2] for n in bass_b[b]]) if bass_b[b] else None
        S["bass_low"].append(bl)
        cats["bass_pc"].append(bl % 12 if bl is not None else None)
        if p.bass_all and m:
            cl = 0
            tot = 0
            for n in m:
                r = sounding_lowest(p, n[0])
                if r is None:
                    continue
                tot += 1
                cl += 1 if (n[2] - r) % 12 in (1, 6, 11) else 0
            S["clash"].append(cl / float(tot) if tot else None)
        else:
            S["clash"].append(None)
        hb = harm_b[b]
        if hb:
            pcs = set(n[2] % 12 for n in hb)
            root = bl if bl is not None else min(n[2] for n in hb)
            has = ((root + 3) % 12 in pcs) or ((root + 4) % 12 in pcs)
            S["third_ratio"].append(1.0 if has else 0.0)
        else:
            S["third_ratio"].append(None)
        lo_w, hi_w = b * bb, (b + key_window) * bb
        hist = [0.0] * 12
        for n in p.mel + p.harm_all:
            if lo_w - 1e-9 <= n[0] < hi_w - 1e-9:
                hist[n[2] % 12] += max(n[1], 0.05)
        ke = _key_from_hist(hist)
        if ke:
            (r, t, md), _ = ke
            S["key_r"].append(r)
            cats["key_col"].append(_key_col(t, md))
            cats["key_tonic"].append(t)
            cats["key_mode"].append(md)
        else:
            S["key_r"].append(None)
            cats["key_col"].append(None)
            cats["key_tonic"].append(None)
            cats["key_mode"].append(None)
    out = {}
    for f in FEATURE_ORDER:
        v = _fill(S[f])
        if v is not None:
            out[f] = v
    for k in list(cats):
        v = _fill(cats[k]) if k != "key_mode" else None
        if k == "key_mode":
            raw = cats[k]
            cats[k] = [m if m is not None else "major" for m in raw]
        elif v is None:
            del cats[k]
        else:
            cats[k] = [int(a) for a in v]
    return out, cats


# ══════════════════════════════════════════════════
#  DETECTORES DE FORMA SOBRE SERIES
# ══════════════════════════════════════════════════
def _fdelta(f):
    return FEATURES.get(f, {}).get("delta", 1.0)


def _flabel(f):
    return FEATURES.get(f, {}).get("label", f)


def _funit(f):
    return FEATURES.get(f, {}).get("unit", "")


def _val(f, v):
    """Valor con su unidad, sin espacios sobrantes."""
    s = "%.1f" % v if abs(v) >= 10 else "%.2f" % v
    return ("%s %s" % (s, _funit(f))).strip()


# Rasgos sobre los que se buscan formas (rampa, meseta, periodicidad) por
# defecto. Los de tipo proporción son ruidosos: solo con --features.
SHAPE_FEATURES = ["reg_mean", "reg_max", "reg_min", "reg_range", "bass_low",
                  "density", "key_r"]
CONST_FEATURES = SHAPE_FEATURES + ["rest_ratio", "vel_mean"]


def _gesture(feature, shape, bars, score, params, text):
    return dict(feature=feature, shape=shape, bars=bars,
                score=round(float(_clamp(score, 0.0, 1.0)), 3),
                params=_round(params), text=text)


def detect_constant(x, f):
    """(proporción, nivel) si la serie es prácticamente constante."""
    med = _median(x)
    tol = max(0.5 * _fdelta(f), 0.05 * abs(med))
    share = sum(1 for v in x if abs(v - med) <= tol) / float(len(x))
    return (share, med) if share >= 0.85 else None


def detect_ramps(x, f, min_len=4, sign=1):
    """Rampas (sign=+1 sube, -1 baja) y cadenas rampa→reinicio→rampa."""
    delta = _fdelta(f)
    y = [sign * v for v in x]
    n = len(y)
    rng = max(y) - min(y)
    if n < min_len or rng < delta:
        return []
    tol = 0.05 * rng
    ramps, s = [], 0
    for i in range(1, n + 1):
        if i < n and y[i] >= y[i - 1] - tol:
            continue
        e = i - 1
        if e - s + 1 >= min_len:
            rise = y[e] - y[s]
            if rise >= max(delta, 0.25 * rng):
                slope, _, r2 = _linfit(y[s:e + 1])
                ramps.append(dict(start=s, end=e, rise=rise, slope=slope, r2=r2))
        s = i
    if not ramps:
        return []
    chains = [[ramps[0]]]
    for r in ramps[1:]:
        prev = chains[-1][-1]
        gap = r["start"] - prev["end"] - 1
        low = min(y[prev["end"] + 1:r["start"] + 1])
        drop = y[prev["end"]] - low
        if gap <= 2 and drop >= max(0.5 * delta, 0.25 * prev["rise"]):
            chains[-1].append(r)
        else:
            chains.append([r])
    lab = _flabel(f)
    verb = "sube" if sign > 0 else "baja"
    out = []
    for ch in chains:
        scores = [_clamp(r["r2"] * min(1.0, r["rise"] / (0.5 * rng)), 0, 1)
                  for r in ch]
        score = _mean(scores)
        starts = [ch[0]["start"]]
        drops = []
        for k, r in enumerate(ch):
            e = r["end"]
            nxt = y[e + 1:e + 3]
            d = (y[e] - min(nxt)) if nxt else 0.0
            if d >= max(0.5 * delta, 0.25 * r["rise"]):
                starts.append(e + 1)
                drops.append(d)
        slope = sign * _mean([r["slope"] for r in ch])
        rise = _mean([r["rise"] for r in ch])
        a, b = ch[0]["start"] + 1, ch[-1]["end"] + 1
        if len(ch) >= 2:
            diffs = [q - w for w, q in zip(starts, starts[1:])]
            period = _median(diffs) if diffs else None
            shape = "ramp_reset" if sign > 0 else "fall_reset"
            reset = _mean(drops) if drops else 0.0
            text = ("%s: diente de sierra ×%d — %s %s/compás y reinicia "
                    "(c.%d–%d%s, reinicio ≈ %s)"
                    % (lab, len(ch), verb, _val(f, abs(slope)), a, b,
                       ", periodo ≈ %g compases" % period if period else "",
                       _val(f, reset)))
            out.append(_gesture(f, shape, [a, b], score, dict(
                n_ramps=len(ch), slope=slope, rise=rise, reset_depth=reset,
                period_bars=period, cycle_starts=starts, sign=sign,
                r2=_mean([r["r2"] for r in ch])), text))
        elif score >= 0.6:
            shape = "ramp" if sign > 0 else "fall"
            drop = drops[0] if drops else 0.0
            text = ("%s: %s de forma sostenida c.%d–%d (%s%s/compás, %s "
                    "en total)%s" % (lab, "ascenso" if sign > 0 else "descenso",
                                      a, b, "+" if slope >= 0 else "-",
                                      _val(f, abs(slope)), _val(f, rise),
                                      "; después cae %s" % _val(f, drop)
                                      if drop else ""))
            out.append(_gesture(f, shape, [a, b], score, dict(
                slope=slope, rise=rise, drop_after=drop, cycle_starts=starts,
                sign=sign, r2=ch[0]["r2"]), text))
    return out


def detect_plateaus(x, f, min_len=4, frac=0.2):
    delta = _fdelta(f)
    n = len(x)
    rng = max(x) - min(x)
    if n < min_len or rng < 1.5 * delta:
        return []
    thr = max(frac * rng, 0.25 * delta)
    out, i = [], 0
    while i <= n - min_len:
        lo = hi = x[i]
        j = i + 1
        while j < n and max(hi, x[j]) - min(lo, x[j]) <= thr:
            lo, hi = min(lo, x[j]), max(hi, x[j])
            j += 1
        if j - i >= min_len:
            lvl = _mean(x[i:j])
            rel = (lvl - min(x)) / rng
            pos = "alta" if rel >= 0.66 else ("baja" if rel <= 0.33 else "media")
            out.append(_gesture(
                f, "plateau", [i + 1, j], 1.0 - 0.5 * (hi - lo) / thr,
                dict(level=lvl, rel_level=rel, spread=hi - lo),
                "%s: meseta %s en c.%d–%d (≈ %s)"
                % (_flabel(f), pos, i + 1, j, _val(f, lvl))))
            i = j
        else:
            i += 1
    return out


def detect_periodic(x, f, categorical=False, min_overlap=4):
    n = len(x)
    rng = max(x) - min(x)
    if rng < 1e-9 or n < 8:
        return None
    tol = 0.0 if categorical else max(0.1 * rng, 0.25 * _fdelta(f))
    pairs = hit = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            hit += 1 if abs(x[i] - x[j]) <= tol else 0
    chance = hit / float(pairs)
    if chance > 0.97:
        return None
    res = []
    for P in range(2, n // 2 + 1):
        m = n - P
        if m < min_overlap:
            break
        match = sum(1 for i in range(m) if abs(x[i] - x[i + P]) <= tol) / float(m)
        res.append(((match - chance) / (1.0 - chance), P, match))
    if not res:
        return None
    best = max(r[0] for r in res)
    for sc, P, match in res:
        if sc >= best - 0.03:
            return dict(period=P, score=sc, match=match, chance=chance)
    return None


# ══════════════════════════════════════════════════
#  GESTOS ESTRUCTURALES (sobre notas)
# ══════════════════════════════════════════════════
def compute_stats(p, S, cats):
    cells = pulse_cells(p)
    sizes = [len(c) for c in cells.values()]
    modal = Counter(sizes).most_common(1)[0][0] if sizes else 1
    pit = sorted(n[2] for n in p.mel)
    w = [0.0] * 12
    for n in p.mel + p.harm_all:
        w[n[2] % 12] += max(n[1], 0.05)
    tot = sum(w) or 1.0
    scale = [i for i in range(12) if w[i] / tot >= 0.025]
    offs = {}
    hb = {}
    for n in p.harm_all:
        hb.setdefault(_bar_of(p, n[0]), []).append(n)
    for b, ns in hb.items():
        pcs = set(n[2] % 12 for n in ns)
        root = min(n[2] for n in ns)
        bass_b = [n[2] for n in p.bass_all if _bar_of(p, n[0]) == b]
        if bass_b:
            root = min(bass_b)
        if (root + 4) % 12 in pcs:
            offs[b + 1] = 4
        elif (root + 3) % 12 in pcs:
            offs[b + 1] = 3
    spreads = []
    for c in cells.values():
        if len(c) >= 2:
            ps = [n[2] for n in c]
            spreads.append(max(ps) - min(ps))
    return dict(
        notes_per_pulse=modal,
        cell_spread=_mean(spreads) if spreads else 0.0,
        pitch_lo=pit[int(0.02 * (len(pit) - 1))] if pit else 48,
        pitch_hi=pit[int(0.98 * (len(pit) - 1))] if pit else 84,
        scale_pcs=scale,
        third_offsets=offs,
        velocity=int(round(_mean([n[3] for n in p.mel]))) if p.mel else 80,
        pulses_per_bar=max(1, int(round(p.bar_beats / p.pulse))))


def detect_micro_cycle(p):
    cells = pulse_cells(p)
    keys = sorted(cells)
    sizes = [len(cells[k]) for k in keys]
    if not sizes:
        return None
    modal = Counter(sizes).most_common(1)[0][0]
    n2 = fall = top = 0
    drops = []
    for k in keys:
        c = cells[k]
        if len(c) >= 2:
            n2 += 1
            fall += 1 if c[0][2] > c[-1][2] else 0
            top += 1 if c[0][2] >= max(n[2] for n in c) else 0
            drops.append(c[0][2] - c[-1][2])
    up = bnd = 0
    leaps = []
    for a, b in zip(keys, keys[1:]):
        if b == a + 1:
            d = cells[b][0][2] - cells[a][-1][2]
            bnd += 1
            up += 1 if d >= 3 else 0
            leaps.append(d)
    if modal < 2 or n2 < 4 or bnd < 4:
        return None
    fs, ru = fall / float(n2), up / float(bnd)
    if fs < 0.6 or ru < 0.4:
        return None
    return _gesture(
        "cell", "micro_cycle", None, (fs + ru) / 2.0,
        dict(notes_per_pulse=modal, fall_share=fs, restart_up_share=ru,
             top_first_share=top / float(n2), mean_fall=_mean(drops),
             mean_restart=_mean(leaps)),
        "relanzamiento por pulso: %d notas/pulso, %d%% de los pulsos empiezan "
        "arriba y acaban abajo (≈ %.1f st) y %d%% de las fronteras saltan hacia "
        "arriba (≈ %+.1f st de media)"
        % (modal, round(fs * 100), _mean(drops), round(ru * 100), _mean(leaps)))


def detect_locks(p, min_lock=3):
    cells = pulse_cells(p)
    keys = sorted(cells)
    out, i = [], 0
    npb = max(1, int(round(p.bar_beats / p.pulse)))
    while i < len(keys):
        c = tuple(n[2] for n in cells[keys[i]])
        j = i + 1
        if len(c) >= 2:
            while (j < len(keys) and keys[j] == keys[j - 1] + 1 and
                   tuple(n[2] for n in cells[keys[j]]) == c):
                j += 1
            if j - i >= min_lock:
                t0 = cells[keys[i]][0][0]
                root = sounding_lowest(p, t0)
                root_pc = (root if root is not None else min(c)) % 12
                offs = [(x - root_pc) % 12 for x in c]
                b0 = keys[i] // npb + 1
                b1 = keys[j - 1] // npb + 1
                out.append(_gesture(
                    "cell", "cell_lock", [b0, b1], min(1.0, (j - i) / 4.0),
                    dict(reps=j - i, cell=list(c), cell_pc_offsets=offs,
                         first_pulse=keys[i]),
                    "célula bloqueada: %s repetida ×%d seguidas (c.%d–%d)"
                    % (" ".join(_pitchname(x) for x in c), j - i, b0, b1)))
        i = max(j, i + 1)
    return out


def detect_bursts(p):
    cells = pulse_cells(p)
    sizes = [len(c) for c in cells.values()]
    if not sizes:
        return []
    modal = Counter(sizes).most_common(1)[0][0]
    npb = max(1, int(round(p.bar_beats / p.pulse)))
    out = []
    if modal < 2:
        return out
    for k in sorted(cells):
        c = cells[k]
        if len(c) >= modal + 1 and len(c) >= 1.4 * modal:
            b = k // npb + 1
            net = c[-1][2] - c[0][2]
            out.append(_gesture(
                "cell", "burst", [b, b], min(1.0, len(c) / (2.0 * modal)),
                dict(n=len(c), pulse=k % npb, modal=modal,
                     direction="up" if net > 0 else "down", span=abs(net)),
                "ráfaga: %d notas en el pulso %d del compás %d (lo normal: %d), "
                "%s %d st" % (len(c), k % npb + 1, b, modal,
                              "ascendente" if net > 0 else "descendente",
                              abs(net))))
    return out


def detect_pedal(p, min_roots=3, min_bottom=0.5, low_q=0.5):
    """Nota(s) que ANCLAN el fondo de las células mientras el bajo cambia.

    Un candidato es una altura (o dos adyacentes) que se repite (≥ 5 % de las
    notas de melodía y ≥ 6 veces), es la nota más grave de su pulso en la
    mitad o más de las ocasiones y acompaña a ≥ 3 raíces de bajo distintas.
    Devuelve una lista de gestos (mejor primero)."""
    pit = [n[2] for n in p.mel]
    if len(pit) < 16 or not p.bass_all:
        return []
    min_total = max(6, int(0.05 * len(pit)))
    sp = sorted(pit)
    thr = sp[int(low_q * (len(sp) - 1))]
    counts = Counter(pit)
    cells = pulse_cells(p)
    npb = max(1, int(round(p.bar_beats / p.pulse)))
    cand = sorted(q for q, c in counts.items() if c >= 2 and q <= thr)
    groups = []
    for q in cand:
        if groups and q - groups[-1][-1] <= 1:
            groups[-1].append(q)
        else:
            groups.append([q])
    out = []
    for g in groups:
        occ = [n for n in p.mel if n[2] in g]
        roots, iv, bars, bottom, pulses_with = set(), Counter(), set(), 0, set()
        for n in occ:
            r = sounding_lowest(p, n[0])
            bars.add(_bar_of(p, n[0]) + 1)
            k = int(n[0] / p.pulse + 1e-6)
            pulses_with.add(k)
            c = cells.get(k, [])
            if len(c) >= 2 and n[2] == min(x[2] for x in c):
                bottom += 1
            if r is None:
                continue
            roots.add(r % 12)
            iv[(n[2] - r) % 12] += 1
        tot = len(occ)
        if (len(roots) < min_roots or tot < min_total or len(bars) < 3 or
                bottom / float(tot) < min_bottom):
            continue
        bshare = bottom / float(tot)
        clash = sum(c for k, c in iv.items() if k in (1, 6, 11)) / float(tot)
        score = (0.4 * min(1.0, len(roots) / 4.0) +
                 0.3 * min(1.0, tot / (0.06 * len(pit))) + 0.3 * bshare)
        span = (min(bars), max(bars))
        n_pulses = (span[1] - span[0] + 1) * npb
        prob = min(0.9, len(pulses_with) / float(n_pulses))
        names = "/".join(_pitchname(q) for q in g)
        out.append(_gesture(
            "melody", "pedal", [span[0], span[1]], score,
            dict(pitches=list(g), count=tot, distinct_bass_roots=len(roots),
                 clash_share=clash, bottom_share=bshare, prob=prob),
            "pedal %s: ancla el fondo de la célula (%d%% de %d apariciones) "
            "sobre %d raíces de bajo distintas; %d%% a 2ª m/7ª M/tritono del "
            "bajo, c.%d–%d" % (names, round(bshare * 100), tot, len(roots),
                               round(clash * 100), span[0], span[1])))
    out.sort(key=lambda g: -g["score"])
    return out


def detect_wander(cats):
    col = cats.get("key_col")
    if not col or len(col) < 6:
        return None
    changes = sum(1 for a, b in zip(col, col[1:]) if a != b)
    rate = changes / float(len(col) - 1)
    distinct = len(set(col))
    if distinct < 3 or rate < 0.4:
        return None
    return _gesture(
        "key", "tonal_wander", [1, len(col)],
        min(1.0, rate) * min(1.0, distinct / 5.0),
        dict(distinct=distinct, changes=changes, change_rate=rate,
             collections=col),
        "deriva tonal: %d colecciones distintas y %d cambios en %d ventanas"
        % (distinct, changes, len(col)))


def detect_loop(cats, min_score=0.8):
    x = cats.get("bass_pc")
    if not x or len(x) < 8:
        return None
    r = detect_periodic([float(v) for v in x], "bass_pc", categorical=True)
    if not r or r["score"] < min_score:
        return None
    P = r["period"]
    dev = [i + 1 for i in range(P, len(x)) if x[i] != x[i - P]]
    return dict(period_bars=P, roots_pc=list(x[:P]), score=r["score"],
                repeats=len(x) / float(P), deviations=dev)


# ══════════════════════════════════════════════════
#  BIBLIOTECA DE FORMAS (generada por discover)
# ══════════════════════════════════════════════════
SHAPE_K = 8


def shape_vector(win, K=SHAPE_K):
    """Remuestrea a K puntos, resta la media y escala por el rango."""
    n = len(win)
    if n == 1:
        return [0.0] * K, 0.0
    pts = []
    for i in range(K):
        t = i * (n - 1) / float(K - 1)
        a = int(t)
        b = min(a + 1, n - 1)
        pts.append(win[a] + (win[b] - win[a]) * (t - a))
    rng = max(pts) - min(pts)
    if rng < 1e-9:
        return [0.0] * K, 0.0
    m = _mean(pts)
    return [(v - m) / rng for v in pts], max(win) - min(win)


def match_library(S, lib, thr=0.9):
    out = []
    for tpl in lib.get("templates", []):
        f, L, pts = tpl["feature"], int(tpl["length"]), tpl["points"]
        x = S.get(f)
        if not x or len(x) < L:
            continue
        hits = []
        for i in range(len(x) - L + 1):
            vec, rng = shape_vector(x[i:i + L])
            if rng < _fdelta(f):
                continue
            c = _corr(vec, pts)
            if c >= thr:
                hits.append((c, i, rng))
        hits.sort(reverse=True)
        used = []
        for c, i, rng in hits:
            if any(abs(i - j) < L for j in used):
                continue
            used.append(i)
            out.append(_gesture(
                f, "lib:" + tpl["name"], [i + 1, i + L], c,
                dict(template=tpl["name"], corr=c, range=rng),
                "%s: coincide con la forma «%s» de la biblioteca (r=%.2f) en "
                "c.%d–%d" % (_flabel(f), tpl["name"], c, i + 1, i + L)))
    return out


# ══════════════════════════════════════════════════
#  ANÁLISIS COMPLETO
# ══════════════════════════════════════════════════
def analyze_piece(p, key_window=2, min_ramp=4, min_lock=3, features=None,
                  library=None, lib_thr=0.9, min_score=0.55):
    S_all, cats = extract_series(p, key_window)
    stats = compute_stats(p, S_all, cats)
    S = S_all
    if features:
        S = {k: v for k, v in S_all.items() if k in features}
        shape_feats = const_feats = set(features)
    else:
        shape_feats, const_feats = set(SHAPE_FEATURES), set(CONST_FEATURES)
    G = []
    for f, x in S.items():
        if len(x) < 2:
            continue
        if f not in const_feats:
            continue
        const = detect_constant(x, f)
        if const:
            G.append(_gesture(f, "constant", None, const[0], dict(
                level=const[1], share=const[0]),
                "%s: prácticamente constante (≈ %s en el %d%% de los compases)"
                % (_flabel(f), _val(f, const[1]), round(const[0] * 100))))
            continue
        if f not in shape_feats:
            continue
        for sg in (1, -1):
            G += detect_ramps(x, f, min_ramp, sg)
        G += detect_plateaus(x, f, min_len=min_ramp)
        per = detect_periodic(x, f)
        if per and per["score"] >= 0.6:
            G.append(_gesture(
                f, "periodic", None, per["score"],
                dict(period_bars=per["period"], match=per["match"]),
                "%s: se repite cada %d compases (coincidencia %d%%)"
                % (_flabel(f), per["period"], round(per["match"] * 100))))
    loop = detect_loop(cats)
    if loop:
        names = " ".join(_pcname(r) for r in loop["roots_pc"])
        G.append(_gesture(
            "bass_pc", "loop", [1, len(cats["bass_pc"])], loop["score"],
            dict(period_bars=loop["period_bars"], roots_pc=loop["roots_pc"],
                 repeats=loop["repeats"], deviations=loop["deviations"]),
            "bucle armónico de %d compases (raíces: %s) ×%.1f"
            % (loop["period_bars"], names, loop["repeats"])))
    w = detect_wander(cats)
    if w:
        G.append(w)
    if "third_ratio" in S_all and len(S_all["third_ratio"]) >= 4:
        tr = S_all["third_ratio"]
        if _mean(tr) <= 0.25:
            bars = sorted(stats["third_offsets"])
            G.append(_gesture(
                "harmony", "open_fifths", None, 1.0 - _mean(tr),
                dict(third_ratio=_mean(tr), third_bars=bars),
                "acordes sin tercera en el %d%% de los compases%s"
                % (round((1 - _mean(tr)) * 100),
                   " (con tercera: c.%s)" % ",".join(str(b) for b in bars)
                   if bars else "")))
    g = detect_micro_cycle(p)
    if g:
        G.append(g)
    G += detect_pedal(p)
    G += detect_locks(p, min_lock)
    G += detect_bursts(p)
    if library:
        G += match_library(S, library, lib_thr)
    G = [g for g in G if g["score"] >= min_score]
    G.sort(key=lambda g: (-g["score"], g["shape"]))
    kk = p.key
    an = dict(
        tool="gesture_lab", version=VERSION,
        source=dict(file=os.path.basename(p.path), bars=p.nbars,
                    time_signature=list(p.time_signature),
                    bar_beats=p.bar_beats, pulse=p.pulse,
                    tempo=round(p.tempo_bpm, 2), key=_round(kk),
                    voices=[dict(idx=v["idx"], track=v["track"],
                                 channel=v["channel"], notes=len(v["notes"]),
                                 mean_pitch=round(v["mean"], 1))
                            for v in p.voices],
                    melody_voice=p.melody_voice, bass_voice=p.bass_voice,
                    melody_notes=len(p.mel)),
        series=_round(S_all), categorical=cats, stats=_round(stats),
        loop=_round(loop) if loop else None, gestures=G)
    return an


# ══════════════════════════════════════════════════
#  INFORME
# ══════════════════════════════════════════════════
def print_analysis(an, show_series=True):
    src = an["source"]
    bar = "═" * 66
    print("\n" + _c("CYA") + bar)
    print("  GESTURE LAB v%s  ·  analyze  ·  %s" % (VERSION, src["file"]))
    print(bar + _c("R"))
    k = src["key"]
    print("  Compás %d/%d · %d compases · %.0f bpm · tonalidad estimada %s %s "
          "(r=%.2f)" % (src["time_signature"][0], src["time_signature"][1],
                        src["bars"], src["tempo"], k["tonic"], k["mode"], k["r"]))
    vs = ", ".join("#%d pista %d/ch%d (%d notas, media %.0f)%s" % (
        v["idx"], v["track"], v["channel"] + 1, v["notes"], v["mean_pitch"],
        " ← melodía" if v["idx"] == src["melody_voice"] else
        (" ← bajo" if v["idx"] == src["bass_voice"] else ""))
        for v in src["voices"])
    print("  Voces: " + vs)
    if show_series:
        print("\n" + _c("B") + "  TRAYECTORIAS (una columna por compás)" + _c("R"))
        for f in FEATURE_ORDER:
            x = an["series"].get(f)
            if not x:
                continue
            print("   %-11s %s  %7.2f → %-7.2f  %s" % (
                f, _spark(x), x[0], x[-1], _c("G") + _flabel(f) + _c("R")))
        cats = an["categorical"]
        if "bass_pc" in cats:
            print("   %-11s %s" % ("raíz bajo",
                                   " ".join(_pcname(v) for v in cats["bass_pc"])))
    G = an["gestures"]
    print("\n" + _c("B") + "  GESTOS (%d)" % len(G) + _c("R"))
    if not G:
        print("   (ninguno por encima de los umbrales)")
    for g in G:
        rng = "c.%d–%d" % tuple(g["bars"]) if g["bars"] else "global"
        col = "GRN" if g["score"] >= 0.75 else ("YEL" if g["score"] >= 0.55 else "G")
        print("   %s%-12s%s %-9s %s%.2f%s  %s" % (
            _c("B"), g["shape"], _c("R"), rng, _c(col), g["score"], _c("R"),
            g["text"]))
    if an["loop"]:
        l = an["loop"]
        print("\n  " + _c("B") + "Bucle:" + _c("R") + " %d compases × %.1f · raíces %s%s"
              % (l["period_bars"], l["repeats"],
                 " ".join(_pcname(r) for r in l["roots_pc"]),
                 " · compases distintos al repetir: %s" % l["deviations"]
                 if l["deviations"] else ""))
    print(_c("CYA") + bar + _c("R") + "\n")


# ══════════════════════════════════════════════════
#  SPEC DE GESTO
# ══════════════════════════════════════════════════
def build_spec(an, name=None):
    """Convierte un análisis (dict o .json) en un spec de generación."""
    src, st, S = an["source"], an["stats"], an["series"]
    G, loop, cats = an["gestures"], an.get("loop"), an.get("categorical", {})
    nb = int(src["bars"])
    key = src["key"]
    xs = S.get("reg_mean") or [60.0] * nb

    reg = dict(base=round(xs[0], 2), slope=0.0, period=None, lift=0.0,
               ceiling=round(max(xs), 2), floor_mean=round(min(xs), 2),
               jitter=1.0)
    cand = [g for g in G if g["feature"] == "reg_mean" and
            g["shape"] in ("ramp_reset", "ramp", "fall_reset", "fall")]
    if cand:
        g = max(cand, key=lambda q: q["score"])
        pr = g["params"]
        cs = pr["cycle_starts"]
        reg["base"] = round(xs[cs[0]] if cs[0] == 0 else xs[0], 2)
        reg["slope"] = round(pr["slope"], 3)
        if len(cs) >= 2 and cs[1] - cs[0] >= 3:
            reg["slope"] = round(_linfit(xs[cs[0]:cs[1]])[0], 3)
        if len(cs) >= 2:
            per = cs[1] - cs[0]
            if loop and abs(loop["period_bars"] - per) <= 1:
                per = loop["period_bars"]
            reg["period"] = int(per)
            if cs[1] < len(xs):
                reg["lift"] = round(xs[cs[1]] - xs[cs[0]], 2)
    bl = S.get("bass_low")
    if bl:
        seg = bl[:loop["period_bars"]] if loop else bl
        floor = int(min(seg))
    else:
        floor = 36
    if loop:
        roots = [int(r) for r in loop["roots_pc"]]
    elif "bass_pc" in cats:
        roots = [int(r) for r in cats["bass_pc"]]
    else:
        roots = [int(key["tonic_pc"])]
    tr = S.get("third_ratio") or [0.0]
    openf = [g for g in G if g["shape"] == "open_fifths"]
    offs = {str(int(k)): v for k, v in (st.get("third_offsets") or {}).items()}
    third_bars = {}
    if openf:
        for b in openf[0]["params"].get("third_bars", []):
            if str(int(b)) in offs:
                third_bars[str(int(b))] = offs[str(int(b))]
    low_over = {}
    if bl:
        for b, v in enumerate(bl):
            pred = floor + ((roots[b % len(roots)] - floor) % 12)
            if abs(v - pred) >= 5:
                low_over[str(b + 1)] = int(v)
    harmony = dict(
        loop_bars=len(roots), roots_pc=roots,
        third_prob=0.0 if openf else round(_mean(tr), 2),
        third_bars=third_bars, voicing="power" if openf else "triad")
    micro = [g for g in G if g["shape"] == "micro_cycle"]
    cell = dict(
        notes_per_pulse=int(st["notes_per_pulse"]),
        fall_prob=round(micro[0]["params"]["fall_share"], 2) if micro else 0.5,
        spread=round(st["cell_spread"], 1), dur_frac=0.95, chord_bias=2.0)
    pedals = []
    for g in sorted([g for g in G if g["shape"] == "pedal"],
                    key=lambda q: q["bars"][0]):
        pp = g["params"]
        pedals.append(dict(pitches=[int(q) for q in pp["pitches"]],
                           prob=round(pp["prob"], 2), bars=list(g["bars"])))
    pedal = (pedals[0] if len(pedals) == 1 else pedals) or None
    locks = []
    for g in sorted([g for g in G if g["shape"] == "cell_lock"],
                    key=lambda q: q["bars"][0]):
        pp = g["params"]
        c = pp["cell"]
        locks.append(dict(bars=list(g["bars"]), reps=int(pp["reps"]),
                          cell_pc_offsets=[int(v) for v in pp["cell_pc_offsets"]],
                          cell_steps=[int(b - a) for a, b in zip(c, c[1:])]))
    lock = (locks[0] if len(locks) == 1 else locks) or None
    bu = [g for g in G if g["shape"] == "burst"]
    burst = None
    if bu:
        burst = dict(bars=sorted(set(g["bars"][0] for g in bu)),
                     pulse=int(bu[0]["params"]["pulse"]),
                     n=int(bu[0]["params"]["n"]))
    return dict(
        gesture_spec="1",
        name=name or os.path.splitext(src["file"])[0],
        source=dict(file=src["file"], detected=[
            "%s:%s" % (g["feature"], g["shape"]) for g in G]),
        bars=nb, time_signature=list(src["time_signature"]),
        tempo=src["tempo"], velocity=int(st["velocity"]),
        key=dict(tonic_pc=int(key["tonic_pc"]), tonic=key["tonic"],
                 mode=key["mode"]),
        scale_pcs=[int(v) for v in st["scale_pcs"]],
        harmony=harmony, bass=dict(floor=floor, low_override=low_over),
        register=reg,
        melody_range=dict(lo=int(st["pitch_lo"]), hi=int(st["pitch_hi"])),
        cell=cell, pedal=pedal, lock=lock, burst=burst)


def _template_base(name, bars, ts, tempo, tonic_pc, mode, scale):
    return dict(
        gesture_spec="1", name=name, source=dict(file="template", detected=[]),
        bars=bars, time_signature=ts, tempo=tempo, velocity=80,
        key=dict(tonic_pc=tonic_pc, tonic=_pcname(tonic_pc), mode=mode),
        scale_pcs=scale)


TEMPLATES = {
    "treadmill": "cinta de correr: subida de registro con reinicio cada 8 "
                 "compases, bucle armónico, acordes sin tercera, pedal que "
                 "choca y final bloqueado",
    "rise_collapse": "subida sostenida de registro durante toda la pieza y "
                     "desplome en el último compás",
    "static_vamp": "vamp estático: bucle de 4 compases, registro constante y "
                   "relanzamiento por pulso",
}


def template_spec(name):
    if name == "treadmill":
        s = _template_base(name, 16, [3, 4], 120.0, 5, "minor",
                           [0, 1, 3, 5, 7, 8, 10])
        s.update(
            harmony=dict(loop_bars=8, roots_pc=[0, 1, 3, 5, 10, 0, 5, 5],
                         third_prob=0.0, third_bars={"16": 4}, voicing="power"),
            bass=dict(floor=34, low_override={"15": 29}),
            register=dict(base=62.0, slope=2.6, period=8, lift=12.0,
                          ceiling=80.7, floor_mean=62.0, jitter=1.0),
            melody_range=dict(lo=55, hi=84),
            cell=dict(notes_per_pulse=3, fall_prob=0.8, spread=11.0,
                      dur_frac=0.95, chord_bias=2.0),
            pedal=dict(pitches=[60, 61], prob=0.35, bars=[1, 10]),
            lock=dict(bars=[16, 16], reps=3, cell_pc_offsets=[7, 4, 0],
                      cell_steps=[-3, -4]),
            burst=dict(bars=[13], pulse=2, n=5))
    elif name == "rise_collapse":
        s = _template_base(name, 12, [4, 4], 96.0, 9, "minor",
                           [9, 11, 0, 2, 4, 5, 7])
        s.update(
            harmony=dict(loop_bars=12, roots_pc=[9, 11, 0, 2, 4, 5, 7, 9, 11, 0, 2, 4],
                         third_prob=0.6, third_bars={}, voicing="triad"),
            bass=dict(floor=33),
            register=dict(base=55.0, slope=2.6, period=11, lift=0.0,
                          ceiling=88.0, floor_mean=55.0, jitter=1.0),
            melody_range=dict(lo=50, hi=96),
            cell=dict(notes_per_pulse=4, fall_prob=0.3, spread=7.0,
                      dur_frac=0.9, chord_bias=2.0),
            pedal=None, lock=None, burst=None)
    elif name == "static_vamp":
        s = _template_base(name, 16, [4, 4], 100.0, 0, "minor",
                           [0, 2, 3, 5, 7, 8, 10])
        s.update(
            harmony=dict(loop_bars=4, roots_pc=[0, 8, 3, 10], third_prob=0.0,
                         third_bars={}, voicing="power"),
            bass=dict(floor=36),
            register=dict(base=68.0, slope=0.0, period=None, lift=0.0,
                          ceiling=70.0, floor_mean=66.0, jitter=0.8),
            melody_range=dict(lo=60, hi=76),
            cell=dict(notes_per_pulse=2, fall_prob=0.8, spread=6.0,
                      dur_frac=0.9, chord_bias=2.0),
            pedal=dict(pitches=[67], prob=0.4, bars=[1, 16]),
            lock=None, burst=None)
    else:
        raise ValueError("plantilla desconocida: %s (usa: %s)"
                         % (name, ", ".join(sorted(TEMPLATES))))
    return s


def apply_overrides(spec, sets):
    for item in sets or []:
        if "=" not in item:
            raise ValueError("--set espera clave=valor, recibido %r" % item)
        k, v = item.split("=", 1)
        try:
            val = json.loads(v)
        except ValueError:
            val = v
        d = spec
        parts = k.split(".")
        for part in parts[:-1]:
            if not isinstance(d.get(part), dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = val
    return spec


def _as_list(v):
    """Un bloque opcional del spec (None, dict o lista de dicts) como lista."""
    if not v:
        return []
    return list(v) if isinstance(v, list) else [v]


def _shift_spec(spec, key):
    """Copia del spec transportada a la tónica indicada."""
    s = json.loads(json.dumps(spec))
    if key is None:
        return s, 0
    tgt = _name_to_pc(key)
    src = int(s["key"]["tonic_pc"])
    sh = ((tgt - src + 6) % 12) - 6
    s["key"]["tonic_pc"], s["key"]["tonic"] = tgt, _pcname(tgt)
    if sh == 0:
        return s, 0
    s["scale_pcs"] = sorted((q + sh) % 12 for q in s["scale_pcs"])
    s["harmony"]["roots_pc"] = [(q + sh) % 12 for q in s["harmony"]["roots_pc"]]
    r = s["register"]
    for f in ("base", "ceiling", "floor_mean"):
        if r.get(f) is not None:
            r[f] = r[f] + sh
    s["melody_range"]["lo"] += sh
    s["melody_range"]["hi"] += sh
    s["bass"]["floor"] += sh
    s["bass"]["low_override"] = {k: v + sh for k, v in
                                 (s["bass"].get("low_override") or {}).items()}
    for pd in _as_list(s.get("pedal")):
        pd["pitches"] = [q + sh for q in pd["pitches"]]
    return s, sh


def print_spec(spec):
    h, r, c = spec["harmony"], spec["register"], spec["cell"]
    print("  %s — %d compases, %d/%d, %.0f bpm, tónica %s %s"
          % (spec["name"], spec["bars"], spec["time_signature"][0],
             spec["time_signature"][1], spec["tempo"], spec["key"]["tonic"],
             spec["key"]["mode"]))
    print("  escala: %s" % " ".join(_pcname(q) for q in spec["scale_pcs"]))
    print("  armonía: bucle de %d compases · raíces %s · %s%s"
          % (len(h["roots_pc"]), " ".join(_pcname(q) for q in h["roots_pc"]),
             h["voicing"],
             " · terceras en c.%s" % ",".join(h["third_bars"]) if h.get("third_bars")
             else ""))
    print("  registro: base %.1f · pendiente %+.2f/compás · periodo %s · salto "
          "entre ciclos %+.1f · techo %.1f"
          % (r["base"], r["slope"], r.get("period") or "—", r.get("lift", 0.0),
             r.get("ceiling") if r.get("ceiling") is not None else float("nan")))
    print("  célula: %d notas/pulso · descenso %.0f%% · amplitud %.1f st"
          % (c["notes_per_pulse"], c["fall_prob"] * 100, c["spread"]))
    for k in ("pedal", "lock", "burst"):
        items = _as_list(spec.get(k))
        if not items:
            print("  %s: —" % k)
        for it in items:
            print("  %s: %s" % (k, json.dumps(it)))


# ══════════════════════════════════════════════════
#  GENERADOR
# ══════════════════════════════════════════════════
def _target_mean(reg, b):
    per = reg.get("period")
    slope, base, lift = reg.get("slope", 0.0), reg["base"], reg.get("lift", 0.0)
    if per:
        cyc, pos = divmod(b, int(per))
        m = base + lift * cyc + slope * pos
    else:
        m = base + slope * b
    if reg.get("ceiling") is not None:
        m = min(m, reg["ceiling"])
    if reg.get("floor_mean") is not None:
        m = max(m, reg["floor_mean"])
    return m


def _pick(rng, pool, target, chord_pcs, bias, lo, hi):
    c = [q for q in pool if lo <= q <= hi and abs(q - target) <= 4.5]
    if not c:
        c = sorted(pool, key=lambda q: abs(q - target))[:2]
    w = [(1.0 / (1.0 + abs(q - target))) * (bias if q % 12 in chord_pcs else 1.0)
         for q in c]
    return rng.choices(c, weights=w)[0]


def _lock_cell(lock, root_pc, top_target, lo, hi):
    offs, steps = lock["cell_pc_offsets"], lock.get("cell_steps")
    pc0 = (root_pc + offs[0]) % 12
    cands = [q for q in range(lo - 12, hi + 13) if q % 12 == pc0]
    q0 = min(cands, key=lambda q: abs(q - top_target))
    out = [q0]
    for k in range(1, len(offs)):
        if steps and k - 1 < len(steps):
            out.append(out[-1] + steps[k - 1])
        else:
            pc = (root_pc + offs[k]) % 12
            q = out[-1] - ((out[-1] - pc) % 12)
            out.append(q - 12 if q == out[-1] else q)
    return out


def generate_piece(spec, seed=0, bars=None, key=None, tempo=None):
    """Devuelve dict(mel, harm, bar_beats, time_signature, tempo, spec)."""
    s, sh = _shift_spec(spec, key)
    if bars:
        s["bars"] = int(bars)
    if tempo:
        s["tempo"] = float(tempo)
    rng = random.Random(seed)
    nb = int(s["bars"])
    num, den = s["time_signature"]
    bb = num * 4.0 / den
    pulse = 1.5 if (den == 8 and num in (6, 9, 12)) else 4.0 / den
    if bb / pulse < 1.0:
        pulse = bb
    npb = max(1, int(round(bb / pulse)))
    H, R, C = s["harmony"], s["register"], s["cell"]
    roots = H["roots_pc"] or [s["key"]["tonic_pc"]]
    scale = set(s["scale_pcs"])
    lo, hi = s["melody_range"]["lo"], s["melody_range"]["hi"]
    pool = [q for q in range(lo - 3, hi + 4) if q % 12 in scale]
    if len(pool) < 3:
        pool = list(range(lo, hi + 1))
    vel = int(s.get("velocity", 80))
    n = max(1, int(C["notes_per_pulse"]))
    dur_frac = C.get("dur_frac", 0.95)
    bias = C.get("chord_bias", 2.0)
    floor = int(s["bass"]["floor"])
    peds, locks, burst = (_as_list(s.get("pedal")), _as_list(s.get("lock")),
                          s.get("burst"))
    mel, harm = [], []
    for b in range(nb):
        root_pc = roots[b % len(roots)]
        t_off = (H.get("third_bars") or {}).get(str(b + 1))
        chord = {root_pc, (root_pc + 7) % 12}
        if t_off:
            chord.add((root_pc + t_off) % 12)
        M = _target_mean(R, b) + rng.gauss(0.0, R.get("jitter", 0.0) * 0.3)
        lock = next((lk for lk in locks
                     if lk["bars"][0] <= b + 1 <= lk["bars"][1]), None)
        lock_cell = (_lock_cell(lock, root_pc, M + C["spread"] / 2.0, lo, hi)
                     if lock else None)
        for j in range(npb):
            t0 = b * bb + j * pulse
            if lock_cell is not None:
                ps = list(lock_cell)
            elif (burst and (b + 1) in burst["bars"] and
                  j == min(npb - 1, int(burst.get("pulse", npb - 1)))):
                nb_ = int(burst["n"])
                start = _pick(rng, pool, M - C["spread"] / 2.0, chord, bias, lo, hi)
                idx = min(range(len(pool)), key=lambda i: abs(pool[i] - start))
                idx = max(0, min(idx, len(pool) - nb_))
                ps = [pool[idx + k] for k in range(nb_)]
            else:
                tops = M + C["spread"] / 2.0
                lows = M - C["spread"] / 2.0
                ts = ([tops - (tops - lows) * k / float(n - 1) for k in range(n)]
                      if n > 1 else [M])
                ps = [_pick(rng, pool, t, chord, bias, lo, hi) for t in ts]
                for ped in peds:
                    if (n >= 2 and ped["bars"][0] <= b + 1 <= ped["bars"][1]
                            and rng.random() < ped["prob"]):
                        ps[-1] = rng.choice(ped["pitches"])
                        break
                ps_desc = sorted(ps[:-1], reverse=True) + [ps[-1]]
                if rng.random() < C["fall_prob"]:
                    ps = ps_desc
                else:
                    for _ in range(4):
                        rng.shuffle(ps)
                        if ps != ps_desc:
                            break
            k = len(ps)
            step = pulse / float(k)
            for i, q in enumerate(ps):
                mel.append((t0 + i * step, step * dur_frac, int(q), vel))
        over = (s["bass"].get("low_override") or {}).get(str(b + 1))
        root_p = int(over) if over is not None else floor + ((root_pc - floor) % 12)
        if t_off:
            base = root_p + (24 if over is not None else 12)
            if over is not None:
                harm.append((b * bb, pulse * 0.95, root_p, vel))
            for j in range(npb):
                t0 = b * bb + j * pulse
                for q in (base, base + t_off, base + 7):
                    harm.append((t0, pulse * 0.95, q, vel))
        else:
            harm.append((b * bb, bb - 0.15 * pulse, root_p, vel))
            for j in range(npb):
                t0 = b * bb + j * pulse
                harm.append((t0, pulse * 0.95, root_p + 12, vel))
                if j == 0:
                    harm.append((t0, pulse * 0.95, root_p + 7, vel))
                    if H.get("third_prob", 0.0) > 0 and rng.random() < H["third_prob"]:
                        for off in (4, 3):
                            if (root_pc + off) % 12 in scale:
                                harm.append((t0, pulse * 0.95,
                                             root_p + 12 + off, vel))
                                break
    return dict(mel=mel, harm=harm, bar_beats=bb, time_signature=(num, den),
                tempo=float(s["tempo"]), spec=s, shift=sh, bars=nb)


def _track(notes, tpb, channel=0, name=None):
    ev = []
    for on, dur, q, v in notes:
        a = int(round(on * tpb))
        b = max(a + 1, int(round((on + dur) * tpb)))
        ev.append((a, 1, mido.Message("note_on", channel=channel, note=int(q),
                                      velocity=int(v))))
        ev.append((b, 0, mido.Message("note_off", channel=channel, note=int(q),
                                      velocity=0)))
    ev.sort(key=lambda e: (e[0], e[1]))
    tr = mido.MidiTrack()
    if name:
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
    tr.append(mido.Message("program_change", channel=channel, program=0, time=0))
    last = 0
    for tick, _, msg in ev:
        tr.append(msg.copy(time=tick - last))
        last = tick
    tr.append(mido.MetaMessage("end_of_track", time=1))
    return tr


def write_midi(path, gen, tpb=480):
    _ensure_parent(path)
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    t0 = mido.MidiTrack()
    num, den = gen["time_signature"]
    t0.append(mido.MetaMessage("track_name", name=gen["spec"].get("name", "gesture"),
                               time=0))
    t0.append(mido.MetaMessage("time_signature", numerator=num, denominator=den,
                               clocks_per_click=24, notated_32nd_notes_per_beat=8,
                               time=0))
    t0.append(mido.MetaMessage("set_tempo", tempo=int(60000000 / gen["tempo"]),
                               time=0))
    t0.append(mido.MetaMessage("end_of_track", time=1))
    mid.tracks.append(t0)
    mid.tracks.append(_track(gen["mel"], tpb, 0, "melodia"))
    mid.tracks.append(_track(gen["harm"], tpb, 0, "armonia"))
    mid.save(path)
    return path


def generate_piece_to_file(spec, path, seed=0, bars=None, key=None, tempo=None):
    gen = generate_piece(spec, seed, bars, key, tempo)
    write_midi(path, gen)
    return gen


def verify_generated(gen, path):
    """Reanaliza la salida y compara con lo que pedía el spec."""
    s = gen["spec"]
    p = load_piece(path)
    an = analyze_piece(p)
    shapes = {}
    for g in an["gestures"]:
        shapes.setdefault((g["feature"], g["shape"]), []).append(g)
    has = lambda f, *sh: any((f, x) in shapes for x in sh)  # noqa: E731
    any_shape = lambda x: any(k[1] == x for k in shapes)    # noqa: E731
    checks = []
    nb = gen["bars"]
    lb = len(s["harmony"]["roots_pc"])
    if 2 <= lb and nb >= 2 * lb and len(set(s["harmony"]["roots_pc"])) > 1:
        lp = an["loop"]
        ok = bool(lp) and lp["period_bars"] in (lb, lb * 2) or \
            bool(lp) and lb % lp["period_bars"] == 0
        checks.append(("bucle de %d compases" % lb, ok,
                       "detectado: %s" % (lp["period_bars"] if lp else "no")))
    r = s["register"]
    if abs(r.get("slope", 0)) >= 0.5 and nb >= 6:
        ok = has("reg_mean", "ramp", "ramp_reset", "fall", "fall_reset")
        checks.append(("rampa de registro (%+.1f st/compás)" % r["slope"], ok,
                       "reg_mean: %s" % (", ".join(k[1] for k in shapes
                                                   if k[0] == "reg_mean") or "—")))
    elif abs(r.get("slope", 0)) < 0.2:
        ok = has("reg_mean", "constant", "plateau") or not any(
            k[0] == "reg_mean" and k[1] in ("ramp", "ramp_reset") for k in shapes)
        checks.append(("registro sin rampa", ok, ""))
    if s["harmony"].get("third_prob", 0) < 0.1 and s["harmony"].get("voicing") == "power":
        checks.append(("acordes sin tercera", any_shape("open_fifths"), ""))
    for pd in _as_list(s.get("pedal")):
        names = "/".join(_pitchname(q) for q in pd["pitches"])
        got = [g for g in an["gestures"] if g["shape"] == "pedal" and
               set(g["params"]["pitches"]) & set(pd["pitches"])]
        checks.append(("pedal %s" % names, bool(got), ""))
    for lk in _as_list(s.get("lock")):
        got = [g for g in an["gestures"] if g["shape"] == "cell_lock" and
               g["bars"][0] <= lk["bars"][1] and g["bars"][1] >= lk["bars"][0]]
        checks.append(("célula bloqueada c.%d–%d" % tuple(lk["bars"]),
                       bool(got), ""))
    if s.get("burst"):
        checks.append(("ráfaga", any_shape("burst"), ""))
    if s["cell"]["notes_per_pulse"] >= 2 and s["cell"]["fall_prob"] >= 0.6:
        checks.append(("relanzamiento por pulso", any_shape("micro_cycle"), ""))
    return checks, an


# ══════════════════════════════════════════════════
#  DISCOVER: formas recurrentes sin plantilla
# ══════════════════════════════════════════════════
DEFAULT_DISCOVER_FEATURES = ["reg_mean", "reg_max", "reg_min", "bass_low",
                             "density", "desc_ratio", "leap_ratio", "clash",
                             "key_r"]

_CANON_RAW = {
    "rampa ascendente": [0, 1, 2, 3, 4, 5, 6, 7],
    "rampa descendente": [7, 6, 5, 4, 3, 2, 1, 0],
    "arco": [0, 3, 5, 6, 6, 5, 3, 0],
    "valle": [6, 3, 1, 0, 0, 1, 3, 6],
    "diente de sierra ascendente": [0, 1, 2, 3, 4, 5, 6, 0],
    "diente de sierra descendente": [6, 5, 4, 3, 2, 1, 0, 6],
    "escalón ascendente": [0, 0, 0, 0, 1, 1, 1, 1],
    "escalón descendente": [1, 1, 1, 1, 0, 0, 0, 0],
    "zigzag": [0, 1, 0, 1, 0, 1, 0, 1],
}
_CANON = {k: shape_vector([float(a) for a in v])[0] for k, v in _CANON_RAW.items()}


def name_shape(vec):
    best, bn = -2.0, "forma libre"
    for k, v in _CANON.items():
        c = _corr(vec, v)
        if c > best:
            best, bn = c, k
    return (bn, best) if best >= 0.85 else ("forma libre", best)


def _np():
    try:
        import numpy as np
    except ImportError:
        print("ERROR: discover requiere numpy (pip install numpy)")
        sys.exit(1)
    return np


def collect_midis(paths):
    out = []
    for pth in paths:
        if os.path.isdir(pth):
            for root, _, fs in os.walk(pth):
                for f in sorted(fs):
                    if f.lower().endswith((".mid", ".midi")):
                        out.append(os.path.join(root, f))
        elif any(ch in pth for ch in "*?["):
            out += sorted(glob.glob(pth))
        else:
            out.append(pth)
    return out


def _windows(x, f, L, stride):
    out = []
    for i in range(0, len(x) - L + 1, stride):
        vec, rng = shape_vector(x[i:i + L])
        if rng >= _fdelta(f):
            out.append((i, vec, rng))
    return out


def _kmeans(X, k, seed, iters=40):
    np = _np()
    rs = np.random.RandomState(seed)
    k = min(k, len(X))
    idx = [int(rs.randint(len(X)))]
    for _ in range(1, k):
        d = np.min(((X[:, None, :] - X[idx][None, :, :]) ** 2).sum(2), axis=1)
        if d.sum() <= 1e-12:
            idx.append(int(rs.randint(len(X))))
        else:
            idx.append(int(rs.choice(len(X), p=d / d.sum())))
    C = X[idx].copy()
    lab = None
    for _ in range(iters):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = D.argmin(1)
        new = C.copy()
        for c in range(k):
            m = X[lab == c]
            if len(m):
                new[c] = m.mean(0)
        if np.allclose(new, C):
            break
        C = new
    return lab, C


def _cluster_stats(X, lab, C):
    np = _np()
    res = []
    for c in range(len(C)):
        m = X[lab == c]
        if not len(m):
            continue
        cen = C[c] - C[c].mean()
        coh = float(np.mean([_corr(list(v), list(cen)) for v in m]))
        res.append((c, int((lab == c).sum()), coh))
    return res


def _distinct_examples(members, n):
    """Hasta n miembros, priorizando ficheros distintos."""
    out, seen = [], set()
    for m in members:
        if m[0] not in seen:
            seen.add(m[0])
            out.append(m)
        if len(out) == n:
            return out
    for m in members:
        if m not in out:
            out.append(m)
        if len(out) == n:
            break
    return out


def discover_corpus(series_by_file, feats, lengths, k=8, stride=1, nnull=1,
                    seed=0):
    np = _np()
    clusters = []
    for f in feats:
        for L in lengths:
            items = []
            for fi, (name, S) in enumerate(series_by_file):
                x = S.get(f)
                if not x or len(x) < L:
                    continue
                for i, vec, rng in _windows(x, f, L, stride):
                    items.append((fi, i, vec, rng))
            if len(items) < max(2 * k, 6):
                continue
            X = np.array([it[2] for it in items])
            lab, C = _kmeans(X, k, seed)
            st = _cluster_stats(X, lab, C)
            null_coh = []
            for r in range(nnull):
                rng_ = random.Random(seed * 1000 + r + 1)
                nit = []
                for fi, (name, S) in enumerate(series_by_file):
                    x = S.get(f)
                    if not x or len(x) < L:
                        continue
                    y = list(x)
                    rng_.shuffle(y)
                    for i, vec, rg in _windows(y, f, L, stride):
                        nit.append(vec)
                if len(nit) >= k:
                    Xn = np.array(nit)
                    ln, Cn = _kmeans(Xn, k, seed + r + 1)
                    null_coh.append(sorted((s[2] for s in
                                            _cluster_stats(Xn, ln, Cn)),
                                           reverse=True))
            ranked = sorted(st, key=lambda s: -s[2])
            for rank, (c, n, coh) in enumerate(ranked):
                nl = (_mean([nc[rank] for nc in null_coh if rank < len(nc)])
                      if null_coh else 0.0)
                members = [items[i] for i in range(len(items)) if lab[i] == c]
                files = sorted(set(m[0] for m in members))
                cen = [float(v) for v in (C[c] - C[c].mean())]
                nm, nc_ = name_shape(cen)
                clusters.append(dict(
                    feature=f, length=L, n=n, files=len(files),
                    cohesion=round(coh, 3), null_cohesion=round(nl, 3),
                    lift=round(coh - nl, 3), name=nm, name_corr=round(nc_, 3),
                    mean_range=round(_mean([m[3] for m in members]), 2),
                    points=[round(v, 4) for v in cen],
                    examples=[dict(file=series_by_file[m[0]][0], bar=m[1] + 1)
                              for m in _distinct_examples(members, 3)]))
    for cl in clusters:
        cl["interest"] = round(max(cl["lift"], 0.0) * math.log(1 + cl["n"]) *
                               (cl["files"] / float(max(1, len(series_by_file)))
                                ) ** 0.5, 4)
    clusters.sort(key=lambda c: -c["interest"])
    return clusters


def discover_within(name, S, feats, lengths, thr=0.92):
    np = _np()
    groups = []
    for f in feats:
        x = S.get(f)
        if not x:
            continue
        for L in lengths:
            if len(x) < 2 * L:
                continue
            wins = _windows(x, f, L, 1)
            if len(wins) < 2:
                continue
            pos = [w[0] for w in wins]
            V = np.array([w[1] for w in wins])
            Vc = V - V.mean(axis=1, keepdims=True)
            nrm = np.linalg.norm(Vc, axis=1, keepdims=True)
            nrm[nrm < 1e-9] = 1.0
            Z = Vc / nrm
            corr = Z @ Z.T
            used = set()
            order = sorted(range(len(wins)), key=lambda i: -wins[i][2])
            for i in order:
                if i in used:
                    continue
                chosen = [i]
                for j in range(len(wins)):
                    if j != i and j not in used and corr[i, j] >= thr and \
                            all(abs(pos[j] - pos[c]) >= L for c in chosen):
                        chosen.append(j)
                if len(chosen) < 2:
                    continue
                for c in chosen:
                    for j in range(len(wins)):
                        if abs(pos[j] - pos[c]) < L:
                            used.add(j)
                chosen.sort(key=lambda c: pos[c])
                cen = [float(v) for v in V[chosen].mean(axis=0)]
                nm, nc_ = name_shape(cen)
                ps = [pos[c] for c in chosen]
                diffs = [b - a for a, b in zip(ps, ps[1:])]
                groups.append(dict(
                    file=name, feature=f, length=L, n=len(chosen),
                    bars=[p_ + 1 for p_ in ps],
                    period=_median(diffs) if diffs else None,
                    corr=round(float(_mean([corr[i, c] for c in chosen[1:]
                                            if c != i] or [1.0])), 3),
                    mean_range=round(_mean([wins[c][2] for c in chosen]), 2),
                    name=nm, points=[round(v, 4) for v in cen]))
    groups.sort(key=lambda g: (-g["n"] * g["length"], -g["mean_range"]))
    return groups


def build_library(clusters, groups, min_lift=0.05):
    tpls, seen = [], set()
    for cl in clusters:
        if cl["lift"] >= min_lift and cl["name"] != "forma libre":
            nm = "%s|%s|L%d" % (cl["name"], cl["feature"], cl["length"])
            if nm not in seen:
                seen.add(nm)
                tpls.append(dict(name=nm, feature=cl["feature"],
                                 length=cl["length"], points=cl["points"],
                                 support=cl["n"], origin="corpus"))
    for g in groups:
        nm = "%s|%s|L%d" % (g["name"], g["feature"], g["length"])
        if nm not in seen:
            seen.add(nm)
            tpls.append(dict(name=nm, feature=g["feature"], length=g["length"],
                             points=g["points"], support=g["n"],
                             origin="within:" + g["file"]))
    return dict(tool="gesture_lab", version=VERSION, templates=tpls)


# ══════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════
def _feat_list(s, default=None):
    if not s:
        return default
    fs = [a.strip() for a in s.split(",") if a.strip()]
    bad = [a for a in fs if a not in FEATURES]
    if bad:
        raise ValueError("rasgo desconocido: %s (válidos: %s)"
                         % (", ".join(bad), ", ".join(FEATURE_ORDER)))
    return fs


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path, obj):
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def _analysis_from_midi(path, args):
    p = load_piece(path, melody=getattr(args, "melody", None),
                   bass=getattr(args, "bass", None),
                   bar_beats=getattr(args, "bar_beats", None))
    return analyze_piece(
        p, key_window=getattr(args, "key_window", 2),
        min_ramp=getattr(args, "min_ramp", 4),
        min_lock=getattr(args, "min_lock", 3),
        min_score=getattr(args, "min_score", 0.55),
        features=_feat_list(getattr(args, "features", None)),
        library=_load_json(args.library) if getattr(args, "library", None) else None)


def cmd_analyze(args):
    try:
        an = _analysis_from_midi(args.midi, args)
    except (IOError, ValueError) as e:
        print("ERROR: %s" % e)
        return 2
    print_analysis(an)
    if args.json:
        _save_json(args.json, an)
        print("  → JSON: %s" % args.json)
    return 0


def cmd_spec(args):
    if args.list_templates:
        for k, v in TEMPLATES.items():
            print("  %-14s %s" % (k, v))
        return 0
    try:
        if args.template:
            spec = template_spec(args.template)
        elif args.source:
            if args.source.lower().endswith(".json"):
                obj = _load_json(args.source)
                spec = obj if "gesture_spec" in obj else build_spec(obj, args.name)
            else:
                spec = build_spec(_analysis_from_midi(args.source, args), args.name)
        else:
            print("ERROR: indica una fuente (.mid/.json) o --template")
            return 1
        if args.name:
            spec["name"] = args.name
        if args.bars:
            spec["bars"] = int(args.bars)
        if args.key:
            spec, _ = _shift_spec(spec, args.key)
        apply_overrides(spec, args.set)
    except (IOError, ValueError, KeyError) as e:
        print("ERROR: %s" % e)
        return 2
    _save_json(args.out, spec)
    print("\n" + _c("B") + "  SPEC" + _c("R") + " → %s" % args.out)
    print_spec(spec)
    print()
    return 0


def cmd_generate(args):
    try:
        if args.template:
            spec = template_spec(args.template)
        elif args.spec:
            spec = _load_json(args.spec)
        else:
            print("ERROR: indica un spec .json o --template")
            return 1
    except (IOError, ValueError) as e:
        print("ERROR: %s" % e)
        return 2
    base, ext = os.path.splitext(args.out)
    ext = ext or ".mid"
    bad = 0
    for v in range(args.variations):
        out = args.out if args.variations == 1 else "%s_%d%s" % (base, v + 1, ext)
        gen = generate_piece_to_file(spec, out, args.seed + v, args.bars,
                                     args.key, args.tempo)
        print("  → %s  (%d compases, %d notas de melodía, %d de armonía, "
              "semilla %d%s)" % (out, gen["bars"], len(gen["mel"]),
                                 len(gen["harm"]), args.seed + v,
                                 ", transpuesta %+d" % gen["shift"] if gen["shift"]
                                 else ""))
        if args.verify:
            checks, an = verify_generated(gen, out)
            for name, ok, detail in checks:
                print("    %s %s %s" % (_c("GRN") + "✔" + _c("R") if ok
                                         else _c("RED") + "✘" + _c("R"), name,
                                         _c("G") + detail + _c("R") if detail else ""))
                bad += 0 if ok else 1
    if args.verify:
        print("  verificación: %s" % ("todo coincide con el spec" if not bad
                                      else "%d comprobación(es) no se cumplen" % bad))
    return 0


def cmd_discover(args):
    files = collect_midis(args.paths)
    if not files:
        print("ERROR: no se encontraron MIDIs")
        return 1
    try:
        feats = _feat_list(args.features, DEFAULT_DISCOVER_FEATURES)
        lengths = [int(a) for a in args.lengths.split(",") if a.strip()]
    except ValueError as e:
        print("ERROR: %s" % e)
        return 1
    series = []
    for f in files:
        try:
            p = load_piece(f)
            S, _ = extract_series(p)
            series.append((os.path.basename(f), S))
        except (IOError, ValueError) as e:
            print("  aviso: %s se omite (%s)" % (f, e))
    if not series:
        print("ERROR: ningún MIDI legible")
        return 2
    mode = args.mode
    if mode == "auto":
        mode = "within" if len(series) == 1 else "corpus"
    clusters, groups = [], []
    bar = "═" * 66
    print("\n" + _c("CYA") + bar)
    print("  GESTURE LAB v%s  ·  discover  ·  %d MIDI(s)  ·  modo %s"
          % (VERSION, len(series), mode))
    print(bar + _c("R"))
    if mode in ("corpus", "both"):
        clusters = discover_corpus(series, feats, lengths, args.k, args.stride,
                                   args.null, args.seed)
        print("\n" + _c("B") + "  FORMAS RECURRENTES EN EL CORPUS" + _c("R") +
              _c("G") + "  (lift = cohesión − cohesión con series barajadas)" + _c("R"))
        if not clusters:
            print("   (pocas ventanas: usa más MIDIs, --k menor o --lengths menores)")
        for cl in clusters[:args.top]:
            print("   %-9s L=%-2d %-30s n=%-4d MIDIs=%-3d coh=%.2f lift=%+.2f "
                  "rango≈%.1f" % (cl["feature"], cl["length"], cl["name"], cl["n"],
                                  cl["files"], cl["cohesion"], cl["lift"],
                                  cl["mean_range"]))
            print("             " + _spark(cl["points"]) + "   ej.: " +
                  ", ".join("%s c.%d" % (e["file"], e["bar"])
                            for e in cl["examples"]))
    if mode in ("within", "both"):
        for name, S in series:
            gs = discover_within(name, S, feats, lengths, args.thr)
            groups += gs
            print("\n" + _c("B") + "  RECURRENCIAS DENTRO DE %s" % name + _c("R"))
            if not gs:
                print("   (ninguna con r ≥ %.2f)" % args.thr)
            for g in gs[:args.top]:
                print("   %-9s L=%-2d %-30s ×%d en c.%s%s  r=%.2f rango≈%.1f"
                      % (g["feature"], g["length"], g["name"], g["n"],
                         ",".join(str(b) for b in g["bars"]),
                         " (cada %g)" % g["period"] if g["period"] else "",
                         g["corr"], g["mean_range"]))
                print("             " + _spark(g["points"]))
    print(_c("CYA") + bar + _c("R"))
    if args.emit_library:
        lib = build_library(clusters, groups)
        _save_json(args.emit_library, lib)
        print("  → biblioteca: %s (%d formas)" % (args.emit_library,
                                                   len(lib["templates"])))
    if args.json:
        _save_json(args.json, dict(tool="gesture_lab", version=VERSION,
                                   mode=mode, clusters=clusters, recurrences=groups))
        print("  → JSON: %s" % args.json)
    print()
    return 0


# ══════════════════════════════════════════════════
#  SELFTEST
# ══════════════════════════════════════════════════
def cmd_selftest(args):
    res = []

    def check(name, ok, detail=""):
        res.append(bool(ok))
        mark = _c("GRN") + "PASS" + _c("R") if ok else _c("RED") + "FAIL" + _c("R")
        if args.verbose or not ok:
            print("  %s  %s  %s" % (mark, name, _c("G") + str(detail) + _c("R")))
        else:
            print("  %s  %s" % (mark, name))

    print("\n" + _c("B") + "  SELFTEST gesture_lab v%s" % VERSION + _c("R"))
    # 1. detectores sobre series sintéticas
    saw = [50.0 + 2 * (i % 8) for i in range(24)]
    g = [q for q in detect_ramps(saw, "reg_mean", 4, 1) if q["shape"] == "ramp_reset"]
    check("diente de sierra: ramp_reset con periodo 8 y 3 rampas",
          g and g[0]["params"]["period_bars"] == 8 and g[0]["params"]["n_ramps"] == 3,
          g[0]["params"] if g else None)
    fall = [90.0 - 2 * (i % 8) for i in range(16)]
    g = [q for q in detect_ramps(fall, "reg_mean", 4, -1) if q["shape"] == "fall_reset"]
    check("caída con reinicio (fall_reset)", bool(g))
    pl = [50.0] * 3 + [70.0] * 6 + [50.0] * 3
    g = detect_plateaus(pl, "reg_mean")
    check("meseta alta c.4–9", g and g[0]["bars"] == [4, 9] and
          g[0]["params"]["rel_level"] > 0.9, g[0]["bars"] if g else None)
    check("serie constante", detect_constant([80.0] * 10, "vel_mean") is not None)
    check("serie no constante no es constante",
          detect_constant([50.0 + i for i in range(10)], "reg_mean") is None)
    lp = detect_loop(dict(bass_pc=[0, 5, 7, 5] * 4))
    check("bucle de 4 compases", lp and lp["period_bars"] == 4, lp)
    rs = random.Random(3)
    noise = [rs.random() * 12 for _ in range(16)]
    check("ruido: sin diente de sierra ni periodicidad fuerte",
          not [q for q in detect_ramps(noise, "reg_mean", 4, 1)
               if q["shape"] == "ramp_reset"] and
          (detect_periodic(noise, "reg_mean") or dict(score=0))["score"] < 0.6)
    # 2. ida y vuelta con las plantillas
    tmp = tempfile.mkdtemp(prefix="gesture_lab_")
    paths = []
    for tname in TEMPLATES:
        path = os.path.join(tmp, tname + ".mid")
        gen = generate_piece_to_file(template_spec(tname), path, seed=1)
        paths.append(path)
        checks, an = verify_generated(gen, path)
        for name, ok, detail in checks:
            check("%s → %s" % (tname, name), ok, detail)
        sp = build_spec(an)
        path2 = os.path.join(tmp, tname + "_rt.mid")
        gen2 = generate_piece_to_file(sp, path2, seed=2)
        ok = len(gen2["mel"]) > 0 and load_piece(path2).nbars == sp["bars"]
        check("%s → analyze → spec → generate produce un MIDI legible" % tname, ok)
    # 3. transposición
    gen = generate_piece_to_file(template_spec("treadmill"),
                                 os.path.join(tmp, "g.mid"), 1, key="G")
    an = analyze_piece(load_piece(os.path.join(tmp, "g.mid")))
    check("transposición a G: primera raíz del bucle = D (desde C, +2)",
          an["loop"] and an["loop"]["roots_pc"][0] == 2,
          an["loop"]["roots_pc"] if an["loop"] else None)
    # 4. discover + biblioteca
    for sd in (2, 3):
        generate_piece_to_file(template_spec("treadmill"),
                               os.path.join(tmp, "t%d.mid" % sd), sd)
        generate_piece_to_file(template_spec("rise_collapse"),
                               os.path.join(tmp, "r%d.mid" % sd), sd)
    files = collect_midis([tmp])
    ser = []
    for f in files:
        if f.endswith("_rt.mid"):
            continue
        ser.append((os.path.basename(f), extract_series(load_piece(f))[0]))
    cl = discover_corpus(ser, ["reg_mean", "bass_low"], [4], k=4, nnull=1)
    check("discover (corpus) devuelve formas con soporte", len(cl) > 0 and cl[0]["n"] >= 3,
          "%d clusters" % len(cl))
    tp = os.path.join(tmp, "treadmill.mid")
    S = extract_series(load_piece(tp))[0]
    gr = discover_within("treadmill.mid", S, ["bass_low", "reg_mean"], [4], 0.92)
    check("discover (within): bass_low se repite en bloques de 4 compases",
          any(g["feature"] == "bass_low" and g["n"] >= 2 for g in gr),
          [(g["feature"], g["n"]) for g in gr[:3]])
    lib = build_library(cl, gr)
    check("biblioteca con plantillas", len(lib["templates"]) > 0)
    an = analyze_piece(load_piece(tp), library=lib)
    check("analyze --library produce coincidencias lib:*",
          any(g["shape"].startswith("lib:") for g in an["gestures"]),
          len([g for g in an["gestures"] if g["shape"].startswith("lib:")]))
    # 4b. fidelidad y robustez sobre la pieza generada
    p0 = load_piece(os.path.join(tmp, "treadmill.mid"))
    S0, c0 = extract_series(p0)
    an0 = analyze_piece(p0)
    sp0 = build_spec(an0)
    generate_piece_to_file(sp0, os.path.join(tmp, "tm_rt.mid"), seed=7)
    S1, c1 = extract_series(load_piece(os.path.join(tmp, "tm_rt.mid")))
    check("fidelidad: raíces del bajo idénticas tras analyze→spec→generate",
          c0["bass_pc"] == c1["bass_pc"])
    check("fidelidad: registro medio r > 0.9",
          _corr(S0["reg_mean"], S1["reg_mean"]) > 0.9,
          "r=%.2f" % _corr(S0["reg_mean"], S1["reg_mean"]))
    pedals = [g for g in an0["gestures"] if g["shape"] == "pedal"]
    check("el spec conserva todos los pedales detectados",
          len(_as_list(sp0["pedal"])) == len(pedals), "%d pedales" % len(pedals))
    mt = mido.MidiFile(os.path.join(tmp, "treadmill.mid"))
    one = mido.MidiTrack()
    one.extend(mido.merge_tracks(mt.tracks))
    mo = mido.MidiFile(ticks_per_beat=mt.ticks_per_beat)
    mo.tracks.append(one)
    mo.save(os.path.join(tmp, "onetrack.mid"))
    ao = analyze_piece(load_piece(os.path.join(tmp, "onetrack.mid")))
    check("una sola pista polifónica: mismos bucle y acordes sin tercera",
          ao["loop"] and any(g["shape"] == "open_fifths" for g in ao["gestures"]))
    # 5. casos límite
    mono = os.path.join(tmp, "mono.mid")
    m = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    m.tracks.append(tr)
    for i, q in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 74, 76]):
        tr.append(mido.Message("note_on", note=q, velocity=70, time=0 if i == 0 else 0))
        tr.append(mido.Message("note_off", note=q, velocity=0, time=480))
    m.save(mono)
    try:
        a = analyze_piece(load_piece(mono))
        ok = a["source"]["bars"] >= 2 and isinstance(a["gestures"], list)
    except Exception as e:  # noqa
        ok, a = False, repr(e)
    check("monofónico y corto (10 notas) no falla", ok)
    bad = os.path.join(tmp, "bad.mid")
    with open(bad, "wb") as fh:
        fh.write(b"no soy un midi")
    try:
        load_piece(bad)
        ok = False
    except IOError:
        ok = True
    check("fichero ilegible → IOError", ok)
    n_ok = sum(res)
    print("\n  %d/%d pruebas correctas%s\n" % (n_ok, len(res),
                                             "" if n_ok == len(res)
                                             else "  ← hay fallos"))
    return 0 if n_ok == len(res) else 1


# ══════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════
def _add_analysis_opts(sp):
    sp.add_argument("--melody", type=int)
    sp.add_argument("--bass", type=int)
    sp.add_argument("--bar-beats", type=float)
    sp.add_argument("--key-window", type=int, default=2)
    sp.add_argument("--min-ramp", type=int, default=4)
    sp.add_argument("--min-lock", type=int, default=3)
    sp.add_argument("--min-score", type=float, default=0.55)
    sp.add_argument("--features")
    sp.add_argument("--library")


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="gesture_lab.py",
        description="Trayectorias, gestos y generación a partir de la evolución "
                    "de una pieza MIDI.")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    a = sub.add_parser("analyze", help="Trayectorias y gestos de un MIDI")
    a.add_argument("midi")
    _add_analysis_opts(a)
    a.add_argument("--json")
    a.add_argument("--no-color", action="store_true")
    a.set_defaults(fn=cmd_analyze)

    d = sub.add_parser("discover", help="Formas recurrentes en uno o varios MIDIs")
    d.add_argument("paths", nargs="+")
    d.add_argument("--mode", choices=["auto", "corpus", "within", "both"],
                   default="auto")
    d.add_argument("--features")
    d.add_argument("--lengths", default="4,8")
    d.add_argument("--k", type=int, default=8)
    d.add_argument("--stride", type=int, default=1)
    d.add_argument("--thr", type=float, default=0.92)
    d.add_argument("--top", type=int, default=10)
    d.add_argument("--null", type=int, default=1)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--emit-library")
    d.add_argument("--json")
    d.add_argument("--no-color", action="store_true")
    d.set_defaults(fn=cmd_discover)

    s = sub.add_parser("spec", help="Spec de gesto desde MIDI, análisis o plantilla")
    s.add_argument("source", nargs="?")
    s.add_argument("--template", choices=sorted(TEMPLATES))
    s.add_argument("--list-templates", action="store_true")
    s.add_argument("--out", default="gesture.spec.json")
    s.add_argument("--set", action="append")
    s.add_argument("--bars", type=int)
    s.add_argument("--key")
    s.add_argument("--name")
    _add_analysis_opts(s)
    s.add_argument("--no-color", action="store_true")
    s.set_defaults(fn=cmd_spec)

    g = sub.add_parser("generate", help="Genera un MIDI desde un spec")
    g.add_argument("spec", nargs="?")
    g.add_argument("--template", choices=sorted(TEMPLATES))
    g.add_argument("--out", default="gesture.mid")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--variations", type=int, default=1)
    g.add_argument("--bars", type=int)
    g.add_argument("--key")
    g.add_argument("--tempo", type=float)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--no-color", action="store_true")
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("selftest", help="Pruebas internas")
    t.add_argument("--verbose", action="store_true")
    t.add_argument("--no-color", action="store_true")
    t.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    if getattr(args, "no_color", False):
        _USE_COLOR = False
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
