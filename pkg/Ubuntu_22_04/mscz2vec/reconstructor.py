#!/usr/bin/env python3
r"""
╔════════════════════════════════════════════════════════════════════════════╗
║                            RECONSTRUCTOR  v2.0                             ║
║     De un MIDI a su reconstrucción: análisis, plan, ejecución y receta     ║
║                                                                            ║
║ Fusiona en una sola herramienta autocontenida tres versiones previas:      ║
║   reconstruct.py       RECOMPONE una pieza (analiza, captura, regenera,    ║
║                        critica) y procesa catálogos por lotes.             ║
║   reconstruct_fiel.py  CONSERVA el material: variantes fieles, métricas de ║
║                        retención y bucle de selección por calidad.         ║
║   reconstructor_2.py   plan explícito (plan.json), ejecutor con caché y    ║
║                        reanudación, gate de melodía y subcomandos.         ║
║                                                                            ║
║ Analiza el original, construye el pipeline de herramientas del ecosistema  ║
║ mutopia, lo ejecuta, elige la mejor versión y deja una receta .sh          ║
║ reproducible y un informe .md. Es un único fichero: solo necesita `mido` y ║
║ las herramientas mutopia; midi_doctor (saneado/metadatos) y fidelity_meter ║
║ (fidelidad) tienen sustituto interno si no están.                          ║
║                                                                            ║
║ DOS FILOSOFÍAS (--mode):                                                   ║
║   faithful   CONSERVA el material: parte del original y solo aplica        ║
║              transformaciones que lo respetan (performer, reharmonizer,    ║
║              piano_expander). Genera hasta 3 variantes y elige la de mayor ║
║              calidad que conserve la melodía (gate de fidelidad).          ║
║   recompose  RECOMPONE: captura el ADN (tonalidad, compás, progresión,     ║
║              motivo, clímax) y regenera bajo, acompañamiento y melodía.    ║
║              Es una pieza nueva sembrada por la antigua (melodía ≈ 25 %).  ║
║   both       Ejecuta las dos (obra_fiel.mid y obra_final.mid).             ║
║                                                                            ║
║ SUBCOMANDOS:                                                               ║
║   analyze   Muestra el ADN del original (tonalidad, compás, voces, motivo…)║
║   plan      Escribe el plan (plan.json; --yaml para runner.py) sin ejecutar║
║   run       Plan + ejecución + selección + receta.sh + RECONSTRUCCION.md   ║
║   batch     `run` sobre varios MIDIs, una carpeta o un catálogo JSON       ║
║   choose    Elige entre candidatos: gate de melodía y, luego, calidad      ║
║   compare   Compara recomposición vs variantes fieles (COMPARACION.md)     ║
║   report    Regenera el informe .md de una carpeta ya ejecutada            ║
║                                                                            ║
║ USO:                                                                       ║
║   python reconstructor.py analyze original.mid                             ║
║   python reconstructor.py plan original.mid --mode faithful --out-dir rec/ ║
║   python reconstructor.py run original.mid --out-dir rec/                  ║
║   python reconstructor.py run original.mid --mode both --out-dir rec/      ║
║   python reconstructor.py run original.mid --dry-run      # solo receta    ║
║   python reconstructor.py run original.mid --out-dir rec/ --resume         ║
║   python reconstructor.py run original.mid --mode recompose \              ║
║       --harmony captured --melody-source captured --critique full          ║
║   python reconstructor.py run original.mid --meta-engine builtin \         ║
║       --fidelity-engine builtin --clean none    # sin midi_doctor ni meter ║
║   python reconstructor.py batch midis/ --out-dir recs/ --mode both         ║
║   python reconstructor.py batch --catalog seleccion.json \                 ║
║       --midis-dir midis/ --out-dir reconstrucciones/ --start 1 --limit 5 \ ║
║       --style-source catalog                                               ║
║   python reconstructor.py batch --catalog seleccion.json --id 003 --force  ║
║   python reconstructor.py choose original.mid \                            ║
║       --candidates a.mid b.mid c.mid --min-melody 0.8 --output mejor.mid   ║
║   python reconstructor.py compare recs/                                    ║
║   python reconstructor.py report rec/                                      ║
║                                                                            ║
║ DECISIONES DE DISEÑO SELECCIONABLES (la 1.ª opción es la predeterminada):  ║
║   --clean auto|doctor|none                                                 ║
║       Saneado del original con midi_doctor, o copia literal.               ║
║   --meta-engine auto|doctor|builtin                                        ║
║       Quién reinyecta compás/armadura/tempo: midi_doctor o código interno. ║
║   --meta-policy copy|flatten                                               ║
║       copy conserva las modulaciones del original; flatten deja una sola   ║
║       armadura y compás (el comportamiento de fix_meta.py).                ║
║   --fidelity-engine auto|meter|builtin                                     ║
║       fidelity_meter.py o métricas internas (alturas, onsets, chroma y     ║
║       bigramas del skyline).                                               ║
║   --select quality|fidelity                                                ║
║       Criterio entre las variantes que pasan el gate: quality_scorer o     ║
║       la puntuación de fidelidad.                                          ║
║   --harmony tonicized|captured|generated                                   ║
║       Base armónica preferida del modo recompose; las otras, de reserva.   ║
║   --melody-source motif|guided|captured                                    ║
║       Melodía del modo recompose: sembrada con el motivo real, guiada por  ║
║       acordes, o la melodía capturada por el HMM.                          ║
║   --bars-from ticks|metric                                                 ║
║       Nº de compases: por duración del MIDI o con metric_analyzer.py.      ║
║   --style-source programs|catalog                                          ║
║       Estilo de performer y bajo: según los programas GM o según la        ║
║       familia del catálogo (program_family, solo en batch --catalog).      ║
║   --critique basic|full                                                    ║
║       Critica mínima, o la suite completa (armonía, métrica, melodía,      ║
║       tocabilidad, arco y comparación vectorial).                          ║
║   --recipe-paths absolute|relative                                         ║
║       Rutas de receta.sh: absolutas o relativas a su carpeta.              ║
║   --report-style compact|full                                              ║
║       Informe breve o informe pedagógico completo.                         ║
║                                                                            ║
║ OPCIONES DE plan / run / batch:                                            ║
║   --mode M             faithful (default) | recompose | both               ║
║   --out-dir DIR        Carpeta de salida (default: <original>_rec)         ║
║   --tools DIR          Carpeta con las herramientas mutopia (default: junto║
║                        a este script o el directorio actual)               ║
║   --variants LISTA     faithful: base,rehar,orq (default: las tres)        ║
║   --min-melody F       Umbral de melodía del gate de selección (0.8)       ║
║   --tol F              Tolerancia temporal de fidelidad, en pulsos (0.25)  ║
║   --no-original-candidate                                                  ║
║                        faithful: el original saneado no compite en la      ║
║                        selección (por defecto sí, y garantiza resultado)   ║
║   --emotion E          recompose: emoción del colorista (default: deducida)║
║   --seed N             Semilla base de los generadores (default: 5)        ║
║   --tonic-density F    recompose: densidad de dominantes secundarias (0.15)║
║   --bars N             Fuerza el número de compases                        ║
║   --pattern P          recompose: auto | alberti | arpeggio-up | block     ║
║   --perf-style S       recompose: auto | clasico | romantico | mecanico    ║
║   --bass-style S       recompose: auto | classical | pedal                 ║
║   --skip LISTA         Etapas a omitir: climax,tempo,color,critique        ║
║   --yaml FILE          (plan) además escribe el plan en formato runner.py  ║
║                                                                            ║
║ OPCIONES DE run / batch:                                                   ║
║   --plan FILE          Ejecutar un plan.json ya generado (y editado)       ║
║   --dry-run            No ejecuta: escribe plan, receta.sh e informe       ║
║   --resume             Salta los pasos cuya salida ya existe               ║
║   --force              Borra la carpeta de trabajo y empieza de cero       ║
║   --timeout S          Timeout por paso (default 300)                      ║
║   --strict             Aborta al primer paso obligatorio que falle         ║
║                                                                            ║
║ OPCIONES DE batch:                                                         ║
║   ENTRADAS             MIDIs, carpetas o patrones glob (o --catalog)       ║
║   --catalog FILE       Catálogo JSON [{id, file, path?, program_family?}]  ║
║   --midis-dir DIR      Carpeta de los MIDIs del catálogo                   ║
║   --id ID              Procesa solo la pieza con ese id                    ║
║   --start N --limit N  Rango (1-based) de piezas a procesar                ║
║                                                                            ║
║ OPCIONES DE choose:                                                        ║
║   --candidates M...    MIDIs candidatos (los que no existan se ignoran)    ║
║   --output FILE        MIDI ganador                                        ║
║   --no-quality         Atajo de --select fidelity (sin quality_scorer)     ║
║   --report FILE        JSON con la puntuación de todos los candidatos      ║
║                                                                            ║
║ SALIDAS (en --out-dir; en batch, una subcarpeta NNN_slug por pieza):       ║
║   obra_fiel.mid / obra_final.mid   resultado por modo                      ║
║   plan.json · receta.sh · RECONSTRUCCION.md · run_log.json                 ║
║   work/                            intermedios, logs y JSON de métricas    ║
║   batch_log.json · COMPARACION.md  (batch / compare)                       ║
║                                                                            ║
║ GARANTÍA: en modo faithful siempre hay resultado: si todo falla se entrega ║
║ el original saneado y se avisa.                                            ║
║                                                                            ║
║ CÓDIGOS DE SALIDA: 0 ok · 1 error de uso/entorno · 2 sin resultado ·       ║
║                    3 resultado por debajo del gate de melodía              ║
║                                                                            ║
║ DEPENDENCIAS: mido · herramientas mutopia (performer, reharmonizer,        ║
║ piano_expander, quality_scorer… según el modo)                             ║
╚════════════════════════════════════════════════════════════════════════════╝
"""


import argparse
import copy
import glob
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, OrderedDict
from datetime import datetime
from struct import error as struct_error
from typing import Dict, List, Optional, Tuple

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "2.0"
HERE = os.path.dirname(os.path.abspath(__file__))
SELF = "reconstructor.py"   # nombre con el que los planes se refieren a este script
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SKIP_STAGES = ("climax", "tempo", "color", "critique")
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _c(k: str) -> str:
    return _COLORS.get(k, "") if _USE_COLOR else ""


def read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        return None


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DEL ORIGINAL (ADN)
# ══════════════════════════════════════════════════════════════════════════════

def _load(path: str) -> "mido.MidiFile":
    try:
        return mido.MidiFile(path)
    except Exception:  # noqa: BLE001
        return mido.MidiFile(path, clip=True)


def notes_of(mid, track_index: Optional[int] = None) -> List[Tuple[float, int, int]]:
    tpb = mid.ticks_per_beat or 480
    out = []
    for i, tr in enumerate(mid.tracks):
        if track_index is not None and i != track_index:
            continue
        t = 0
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0 and m.channel != 9:
                out.append((t / tpb, m.note, m.velocity))
    out.sort()
    return out


def extract_motif(mid, track_index: Optional[int], max_notes: int = 8) -> Optional[str]:
    """Skyline de la melodía → 'C4:0.5 E4:0.5 …' (formato de --motif)."""
    notes = notes_of(mid, track_index)
    if len(notes) < 4:
        notes = notes_of(mid, None)
    if len(notes) < 4:
        return None
    groups: "OrderedDict[float, int]" = OrderedDict()
    for t, p, _ in notes:
        slot = round(t * 2) / 2.0
        if slot not in groups or p > groups[slot]:
            groups[slot] = p
    items = list(groups.items())[:max_notes]
    parts = []
    for j, (t, p) in enumerate(items):
        nxt = items[j + 1][0] if j + 1 < len(items) else t + 1.0
        dur = max(0.25, min(nxt - t, 4.0))
        parts.append(f"{NOTE_NAMES[p % 12]}{p // 12 - 1}:{round(dur, 2)}")
    return " ".join(parts)


def climax_position(mid, track_index: Optional[int]) -> float:
    notes = notes_of(mid, track_index) or notes_of(mid, None)
    if not notes:
        return 0.75
    hi = max(p for _, p, _ in notes)
    for i, (_, p, _) in enumerate(notes):
        if p == hi:
            return round(i / max(1, len(notes) - 1), 2)
    return 0.75


def styles_from_programs(progs) -> Tuple[str, str]:
    """(estilo de performer, estilo de bajo) según los programas GM del MIDI."""
    progs = set(progs)
    if progs and all((16 <= p <= 23) or (48 <= p <= 55) for p in progs):
        return "mecanico", "pedal"
    if progs & set(range(40, 48)) or progs & set(range(56, 80)):
        return "romantico", "classical"
    return "clasico", "classical"


def styles_from_family(fam: Optional[str]) -> Tuple[str, str]:
    """(performer, bajo) según la familia instrumental del catálogo."""
    if fam in ("organo", "ensemble"):
        return "mecanico", "pedal"
    if fam in ("cuerdas", "viento_madera", "viento", "metal"):
        return "romantico", "classical"
    return "clasico", "classical"


def metric_bars(path: str, tools: Optional[str]) -> Optional[int]:
    """Nº de compases según metric_analyzer.py (None si no está o falla)."""
    mt = os.path.join(tools or "", "metric_analyzer.py")
    if not tools or not os.path.isfile(mt):
        return None
    fd, tmp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        subprocess.run([sys.executable, mt, path, "--json", tmp], cwd=tools,
                       capture_output=True, text=True, timeout=300)
        d = read_json(tmp) or {}
        n = int(d.get("n_bars") or 0)
        return n if n > 0 else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def analyze_midi(path: str, opts: Optional[Dict] = None) -> Dict:
    """ADN del original. opts: bars, bars_from, tools, style_source, family,
    perf_style, bass_style (todas opcionales)."""
    o = opts or {}
    mid = _load(path)
    tpb = mid.ticks_per_beat or 480
    info: Dict = {"file": path, "tpb": tpb, "tracks": [], "tempo": None, "ts": [], "ks": []}
    for i, tr in enumerate(mid.tracks):
        ns, chans, progs = [], set(), []
        for m in tr:
            if m.type == "note_on" and m.velocity > 0:
                ns.append(m.note)
                chans.add(m.channel)
            elif m.type == "program_change":
                progs.append(m.program)
            elif m.type == "set_tempo" and info["tempo"] is None:
                info["tempo"] = round(mido.tempo2bpm(m.tempo), 2)
            elif m.type == "time_signature":
                v = f"{m.numerator}/{m.denominator}"
                if not info["ts"] or info["ts"][-1] != v:
                    info["ts"].append(v)
            elif m.type == "key_signature":
                if not info["ks"] or info["ks"][-1] != m.key:
                    info["ks"].append(m.key)
        info["tracks"].append({"index": i, "name": tr.name or f"track{i}", "n_notes": len(ns),
                               "median": statistics.median(ns) if ns else None,
                               "lo": min(ns) if ns else None, "hi": max(ns) if ns else None,
                               "channels": sorted(chans), "programs": progs})
    info["tempo"] = info["tempo"] or 120.0
    ts = info["ts"][0] if info["ts"] else "4/4"
    try:
        n, d = (int(x) for x in ts.split("/"))
        if not (0 < n <= 12 and d in (1, 2, 4, 8, 16)):
            ts = "4/4"
    except Exception:  # noqa: BLE001
        ts = "4/4"
    info["time_sig"] = ts
    n, d = (int(x) for x in ts.split("/"))
    info["beats_per_bar"] = n * 4.0 / d
    # melodía = pista con más notas entre las agudas; el resto, acompañamiento
    played = [t for t in info["tracks"] if t["n_notes"] >= 8 and t["median"] is not None]
    played.sort(key=lambda t: (t["median"], t["n_notes"]))
    melody = played[-1] if played else None
    info["melody_track"] = melody["index"] if melody else 0
    info["melody_channel"] = melody["channels"][0] if melody and melody["channels"] else 0
    ks = info["ks"][0] if info["ks"] else "C"
    minor = ks.endswith("m")
    tonic = ks[:-1] if minor else ks
    info["tonic"], info["mode"] = tonic or "C", "minor" if minor else "major"
    info["key_short"] = f"{info['tonic']}{'m' if minor else ''}"
    info["key_long"] = f"{info['tonic']} {info['mode']}"
    info["key_from_signature"] = bool(info["ks"])
    end_tick = 0
    for tr in mid.tracks:
        t = 0
        for m in tr:
            t += m.time
        end_tick = max(end_tick, t)
    beats = end_tick / tpb
    nb = max(1, int(round(beats / info["beats_per_bar"])))
    info["bars_source"] = "ticks"
    if o.get("bars"):
        nb, info["bars_source"] = int(o["bars"]), "manual"
    elif o.get("bars_from") == "metric":
        mb = metric_bars(path, o.get("tools"))
        if mb:
            nb, info["bars_source"] = mb, "metric"
    info["n_bars"] = max(1, min(200, nb))
    info["motif"] = extract_motif(mid, info["melody_track"])
    info["climax"] = climax_position(mid, info["melody_track"])
    progs_all = {p for t in info["tracks"] for p in t["programs"]}
    info["programs"] = sorted(progs_all)
    tempo = float(max(30, min(300, info["tempo"])))
    info["tempo_gen"] = tempo
    slow = tempo < 90
    info["emotion"] = ("melancolia" if minor and slow else "tension" if minor else
                       "serenidad" if slow else "triunfo")
    fam = o.get("family")
    if o.get("style_source") == "catalog" and fam:
        info["perf_style"], info["bass_style"] = styles_from_family(fam)
        info["style_source"] = f"catálogo ({fam})"
    else:
        info["perf_style"], info["bass_style"] = styles_from_programs(progs_all)
        info["style_source"] = "programas GM"
    if o.get("perf_style") not in (None, "auto"):
        info["perf_style"], info["style_source"] = o["perf_style"], "manual"
    if o.get("bass_style") not in (None, "auto"):
        info["bass_style"] = o["bass_style"]
    info["n_notes"] = sum(t["n_notes"] for t in info["tracks"])
    return info


def print_analysis(i: Dict) -> None:
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}\n  {B}RECONSTRUCTOR · ADN de{R} {i['file']}\n{'═' * 78}")
    print(f"  Tonalidad   {i['key_long']}  {G}({'armadura' if i['key_from_signature'] else 'sin armadura: se asume C'}){R}")
    print(f"  Compás      {', '.join(i['ts']) or '4/4 (asumido)'}   Tempo {i['tempo']} BPM   "
          f"{i['n_bars']} compases {G}({i['bars_source']}){R}   {i['n_notes']} notas")
    print(f"  Programas   {i['programs'] or '—'}")
    print(f"  Melodía     pista {i['melody_track']} (canal {i['melody_channel']})   "
          f"clímax al {int(i['climax'] * 100)} %")
    print(f"  Motivo      {i['motif'] or '—'}")
    print(f"  Sugerido    emoción={i['emotion']} · interpretación={i['perf_style']} · "
          f"bajo={i['bass_style']}  {G}[{i['style_source']}]{R}")
    print(f"\n  {G}{'pista':>5} {'notas':>6} {'mediana':>8} {'rango':>9}  nombre / canales / programas{R}")
    for t in i["tracks"]:
        rng = f"{t['lo']}-{t['hi']}" if t["lo"] is not None else "—"
        mark = " ◀ melodía" if t["index"] == i["melody_track"] and t["n_notes"] else ""
        print(f"  {t['index']:>5} {t['n_notes']:>6} {t['median'] if t['median'] is not None else '—':>8} "
              f"{rng:>9}  {t['name']} {t['channels']} {t['programs']}{mark}")
    print(f"{'═' * 78}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS DE FIDELIDAD INTERNAS (alternativa a fidelity_meter.py)
# ══════════════════════════════════════════════════════════════════════════════

def note_list(path: str, tol: float = 0.25) -> List[Tuple[int, int]]:
    """[(onset cuantizado a `tol` pulsos, altura)] de todas las pistas."""
    mf = _load(path)
    tpb = mf.ticks_per_beat or 480
    out = []
    for tr in mf.tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0:
                out.append((round(t / tpb / tol), m.note))
    return out


def skyline_seq(path: str, tol: float = 0.5) -> List[int]:
    mf = _load(path)
    tpb = mf.ticks_per_beat or 480
    d: Dict[int, int] = {}
    for tr in mf.tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0:
                k = round(t / tpb / tol)
                d[k] = max(d.get(k, -1), m.note)
    return [d[k] for k in sorted(d)]


def midi_meta(path: str) -> Dict:
    mf = _load(path)
    ts = sorted({"%d/%d" % (m.numerator, m.denominator) for tr in mf.tracks
                 for m in tr if m.type == "time_signature"})
    ks = sorted({m.key for tr in mf.tracks for m in tr if m.type == "key_signature"})
    progs = sorted({m.program for tr in mf.tracks for m in tr if m.type == "program_change"})
    n = sum(1 for tr in mf.tracks for m in tr if m.type == "note_on" and m.velocity > 0)
    try:
        dur = round(mf.length, 1)
    except Exception:  # noqa: BLE001
        dur = 0.0
    return {"ts": ts, "ks": ks, "progs": progs, "notes": n, "tracks": len(mf.tracks), "dur": dur}


def fidelity_builtin(orig: str, cand: str, tol: float = 0.25) -> Dict:
    """Retención de contenido sin herramientas externas. `melody` = coincidencia
    de bigramas del skyline (la métrica del gate); `score` = media de melody,
    pitch, onset y chroma."""
    a, b = note_list(orig, tol), note_list(cand, tol)
    pitch = sum((Counter(n for _, n in a) & Counter(n for _, n in b)).values()) / max(1, len(a))
    onset = len(set(a) & set(b)) / max(1, len(set(a)))
    ca, cb = [0.0] * 12, [0.0] * 12
    for _, p in a:
        ca[p % 12] += 1
    for _, p in b:
        cb[p % 12] += 1
    na = math.sqrt(sum(v * v for v in ca)) or 1.0
    nb = math.sqrt(sum(v * v for v in cb)) or 1.0
    chroma = sum(x * y for x, y in zip(ca, cb)) / (na * nb)
    sa, sb = skyline_seq(orig, tol * 2), skyline_seq(cand, tol * 2)
    A, B = Counter(zip(sa, sa[1:])), Counter(zip(sb, sb[1:]))
    melody = sum((A & B).values()) / max(1, sum(A.values()))
    mo, mc = midi_meta(orig), midi_meta(cand)
    progs_kept = bool(set(mo["progs"]) & set(mc["progs"])) if mo["progs"] else (mc["progs"] == [])
    ts_kept, ks_kept = set(mo["ts"]) == set(mc["ts"]), set(mo["ks"]) == set(mc["ks"])
    score = (melody + pitch + onset + chroma) / 4.0
    verdict = ("conserva la melodía" if melody >= 0.8 else
               "melodía parcialmente conservada" if melody >= 0.5 else
               "recomposición: la melodía cambia sustancialmente")
    return {"engine": "builtin", "tol": tol, "melody": round(melody, 4), "pitch": round(pitch, 4),
            "onset": round(onset, 4), "chroma": round(chroma, 4),
            "instr": 1.0 if progs_kept else 0.0, "meta": 1.0 if (ts_kept and ks_kept) else 0.0,
            "length": round(mc["dur"] / mo["dur"], 3) if mo["dur"] else None,
            "score": round(score, 4), "verdict": verdict,
            "progs_kept": progs_kept, "ts_kept": ts_kept, "ks_kept": ks_kept,
            "notes": mc["notes"], "tracks": mc["tracks"]}


# ══════════════════════════════════════════════════════════════════════════════
#  METADATOS INTERNOS (alternativa a midi_doctor.py meta / fix_meta.py)
# ══════════════════════════════════════════════════════════════════════════════

_META_KINDS = {"ts": "time_signature", "key": "key_signature", "tempo": "set_tempo"}


def _meta_value(m):
    if m.type == "time_signature":
        return (m.numerator, m.denominator)
    if m.type == "key_signature":
        return m.key
    return m.tempo


def apply_meta(src: str, ref: str, out: str, only=("ts", "key", "tempo"), policy: str = "copy",
               fallback_ts: Optional[str] = None, fallback_key: Optional[str] = None) -> List[str]:
    """Reinyecta en `src` el compás / armadura / tempo de `ref` y lo guarda en `out`.
    policy=copy    copia todos los eventos (conserva las modulaciones);
    policy=flatten deja un único evento inicial por tipo (comportamiento de fix_meta.py).
    Un tipo ya idéntico en `src` no se toca (no se pierde lo que ya conservaba).
    Si `ref` no tiene compás/armadura se usan fallback_ts / fallback_key.
    Devuelve los tipos reescritos."""
    ms, mr = _load(src), _load(ref)
    scale = (ms.ticks_per_beat or 480) / (mr.ticks_per_beat or 480)
    ref_ev: Dict[str, list] = {k: [] for k in only}
    for tr in mr.tracks:
        t = 0
        for m in tr:
            t += m.time
            for k in only:
                if m.type == _META_KINDS[k]:
                    ref_ev[k].append((int(round(t * scale)), m))
    for k in only:
        ref_ev[k].sort(key=lambda x: x[0])
    try:
        if "ts" in only and not ref_ev["ts"] and fallback_ts:
            n, d = (int(x) for x in fallback_ts.split("/"))
            ref_ev["ts"] = [(0, mido.MetaMessage("time_signature", numerator=n, denominator=d))]
        if "key" in only and not ref_ev["key"] and fallback_key:
            ref_ev["key"] = [(0, mido.MetaMessage("key_signature", key=fallback_key))]
    except ValueError:
        pass
    if policy == "flatten":
        for k in only:
            if ref_ev[k]:
                ref_ev[k] = [(0, ref_ev[k][0][1])]
    redo = []
    for k in only:
        if not ref_ev[k]:
            continue
        cur = {_meta_value(m) for tr in ms.tracks for m in tr if m.type == _META_KINDS[k]}
        want = {_meta_value(m) for _, m in ref_ev[k]}
        if cur != want:
            redo.append(k)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if not redo:
        if os.path.abspath(src) != os.path.abspath(out):
            shutil.copyfile(src, out)
        return []
    kinds = {_META_KINDS[k] for k in redo}
    tracks_abs: List[list] = []
    for tr in ms.tracks:
        evs, t = [], 0
        for m in tr:
            t += m.time
            if m.is_meta and m.type == "end_of_track":
                continue
            if m.type in kinds:
                continue
            evs.append((t, m))
        tracks_abs.append(evs)
    if not tracks_abs:
        tracks_abs = [[]]
    for k in redo:
        tracks_abs[0] += [(t, m.copy()) for t, m in ref_ev[k]]
    new_tracks = []
    for evs in tracks_abs:
        evs.sort(key=lambda x: (x[0], 0 if x[1].is_meta else 1))
        tr, last = mido.MidiTrack(), 0
        for t, m in evs:
            tr.append(m.copy(time=t - last))
            last = t
        tr.append(mido.MetaMessage("end_of_track", time=0))
        new_tracks.append(tr)
    ms.tracks = new_tracks
    ms.save(out)
    return redo


# ══════════════════════════════════════════════════════════════════════════════
#  OPCIONES Y PLAN
# ══════════════════════════════════════════════════════════════════════════════
# Los pasos usan marcadores que el ejecutor resuelve:
#   $OUT / $TOOLS / $ORIG        rutas
#   @track0:F  @chan0:F          pista / canal con más notas de F
#   @count:A|B|C                 nº de ficheros existentes entre A, B y C (mín. 1)
#   @quarter:F|N                 cuarta parte del nº de compases de F (JSON de
#                                metric_analyzer); N = valor si no se puede leer
#   when: ["exists:RUTA", "missing:RUTA"]   condiciones sobre ficheros

W = "$OUT/work"
TOKEN = re.compile(r"^@(track0|chan0|count|quarter):(.+)$")


def _variants(s: str) -> List[str]:
    v = [x.strip() for x in s.split(",") if x.strip()]
    bad = [x for x in v if x not in ("base", "rehar", "orq")]
    if bad:
        raise ValueError(f"--variants: valores no válidos {bad} (usa base,rehar,orq)")
    return v or ["base"]


def _skips(s: str) -> List[str]:
    v = [x.strip() for x in (s or "").split(",") if x.strip()]
    bad = [x for x in v if x not in SKIP_STAGES]
    if bad:
        raise ValueError(f"--skip: valores no válidos {bad} (usa {','.join(SKIP_STAGES)})")
    return v


def find_tools_dir(explicit: Optional[str]) -> str:
    cands = [explicit] if explicit else [HERE, os.getcwd()]
    for p in cands:
        if p and any(os.path.isfile(os.path.join(p, f)) for f in ("midi_doctor.py", "performer.py")):
            return os.path.abspath(p)
    return os.path.abspath(explicit or HERE)


def plan_options(a, tools: str) -> Dict:
    """Opciones de diseño del plan, con los 'auto' ya resueltos según las
    herramientas disponibles (así el plan.json es reproducible)."""
    def g(k, d=None):
        v = getattr(a, k, None)
        return d if v is None else v

    def has(f):
        return os.path.isfile(os.path.join(tools, f))

    doctor = has("midi_doctor.py")
    clean = g("clean", "auto")
    clean = ("doctor" if doctor else "none") if clean == "auto" else clean
    meta_engine = g("meta_engine", "auto")
    meta_engine = ("doctor" if doctor else "builtin") if meta_engine == "auto" else meta_engine
    meta_policy = g("meta_policy", "copy")
    if meta_policy == "flatten":
        meta_engine = "builtin"      # midi_doctor meta solo copia: flatten es interno
    fid = g("fidelity_engine", "auto")
    fid = ("meter" if has("fidelity_meter.py") else "builtin") if fid == "auto" else fid
    return {
        "mode": g("mode", "faithful"), "variants": _variants(g("variants", "base,rehar,orq")),
        "min_melody": float(g("min_melody", 0.8)), "tol": float(g("tol", 0.25)),
        "emotion": g("emotion"), "seed": int(g("seed", 5)),
        "clean": clean, "doctor_available": doctor,
        "meta_engine": meta_engine, "meta_policy": meta_policy,
        "fidelity_engine": fid, "select": g("select", "quality"),
        "original_candidate": not g("no_original_candidate", False),
        "harmony": g("harmony", "tonicized"), "tonic_density": float(g("tonic_density", 0.15)),
        "melody_source": g("melody_source", "motif"),
        "bars": g("bars"), "bars_from": g("bars_from", "ticks"),
        "style_source": g("style_source", "programs"), "family": g("family"),
        "perf_style": g("perf_style", "auto"), "bass_style": g("bass_style", "auto"),
        "pattern": g("pattern", "auto"),
        "skip": _skips(g("skip", "")), "critique": g("critique", "basic"),
        "recipe_paths": g("recipe_paths", "absolute"), "report_style": g("report_style", "compact"),
    }


def step(name, tool, args, note="", stage="", optional=False, when=None, output=None) -> Dict:
    return {"name": name, "tool": tool, "args": [str(a) for a in args], "note": note,
            "stage": stage, "optional": optional, "when": when or [], "output": output}


def _prep_steps(i: Dict, O: Dict) -> List[Dict]:
    clean = f"{W}/orig.mid"
    if O["clean"] == "doctor":
        return [
            step("f00_clean", "midi_doctor.py",
                 ["fix", "$ORIG", "-o", clean, "--ensure-meta", "--time-sig", i["time_sig"]]
                 + (["--key", i["key_short"]] if i["key_from_signature"] else []),
                 "Saneado del original (notas sin cerrar, duplicadas…); no altera el contenido",
                 "preparación", True, output=clean),
            step("f00b_clean_copia", SELF, ["_first", "--files", "$ORIG", "-o", clean],
                 "(fallback) copia literal del original", "preparación", False,
                 [f"missing:{clean}"], clean),
        ]
    return [step("f00_clean", SELF, ["_first", "--files", "$ORIG", "-o", clean],
                 "Copia de trabajo del original (sin saneado)", "preparación", False, output=clean)]


def _meta_step(name, src, dst, i, O, only=None, note="", when=None) -> Dict:
    """Reinyecta compás/armadura/tempo del original: midi_doctor o código interno."""
    clean = f"{W}/orig.mid"
    if O["meta_engine"] == "doctor":
        args = ["meta", src, "--copy-from", clean, "--replace", "-o", dst] + (["--only", only] if only else [])
        return step(name, "midi_doctor.py", args, note, "metadatos", True, when, dst)
    args = (["_meta", src, "--copy-from", clean, "-o", dst, "--policy", O["meta_policy"],
             "--time-sig", i["time_sig"]]
            + (["--key", i["key_short"]] if i["key_from_signature"] else [])
            + (["--only", only] if only else []))
    return step(name, SELF, args, note, "metadatos", True, when, dst)


def _fidelity_step(name, target, json_out, O, note) -> Dict:
    clean = f"{W}/orig.mid"
    if O["fidelity_engine"] == "meter":
        return step(name, "fidelity_meter.py",
                    ["compare", clean, target, "--tol", O["tol"], "--timeline", "--json", json_out,
                     "--no-color"], note, "crítica", True)
    return step(name, SELF, ["_fidelity", clean, target, "--tol", O["tol"], "--json", json_out],
                note + " (métricas internas)", "crítica", True)


def _doctor_check(name, target, json_out, O) -> List[Dict]:
    if not O["doctor_available"]:
        return []
    return [step(name, "midi_doctor.py", ["check", target, "--json", json_out, "--no-color"],
                 "Comprobación final del MIDI", "crítica", True)]


def critique_steps(i: Dict, O: Dict, kind: str) -> List[Dict]:
    """Crítica del resultado. basic: calidad, color, fidelidad y comprobación.
    full: añade armonía, métrica, melodía, tocabilidad, arco y comparación."""
    if "critique" in O["skip"]:
        return []
    rec = kind == "recompose"
    tgt = "$OUT/obra_final.mid" if rec else "$OUT/obra_fiel.mid"
    jn = "final" if rec else "fiel"
    p = "r" if rec else "f"
    n0 = 19 if rec else 91
    S: List[Dict] = []
    if rec:
        S.append(step("r19_calidad", "quality_scorer.py", [tgt, "--fast", "--no-color"],
                      "Calidad en 5 dimensiones", "crítica", True))
        if "color" not in O["skip"]:
            emo = O["emotion"] or i["emotion"]
            S.append(step("r20_color", "orchestral_colorist.py", [tgt, "--emotion", emo, "--report"],
                          f"Reasignación tímbrica propuesta para «{emo}»", "color", True))
    S.append(_fidelity_step(f"{p}{n0 + (2 if rec else 0)}_fidelity", tgt, f"{W}/fidelidad_{jn}.json", O,
                            "Fidelidad de la recomposición (esperable: baja)" if rec
                            else "Fidelidad del resultado frente al original"))
    S += _doctor_check(f"{p}{n0 + (3 if rec else 1)}_doctor", tgt, f"{W}/doctor_{jn}.json", O)
    if O["critique"] != "full":
        return S
    mt = (f"@count:{W}/bajo.mid|{W}/acomp.mid|{W}/melodia.mid" if rec else i["melody_track"])
    q = f"@quarter:{W}/metric_{jn}.json|{i['n_bars']}"
    base = f"{W}/obra_base.mid" if rec else tgt
    k = 23 if rec else 93
    S += [
        step(f"{p}{k}_armonia_final", "harmonic_analyzer.py",
             [tgt, "--key", i["key_short"], "--json", f"{W}/chords_{jn}.json", "--max-bars", "24"],
             "Re-análisis armónico del resultado", "crítica", True),
        step(f"{p}{k + 1}_metrica_final", "metric_analyzer.py", [tgt, "--json", f"{W}/metric_{jn}.json"],
             "Métrica del resultado", "crítica", True),
        step(f"{p}{k + 2}_melodia_final", "melody_critic.py", [tgt, "--track", mt],
             "Crítica de la melodía final", "crítica", True),
        step(f"{p}{k + 3}_tocabilidad", "playability_auditor.py", [base, "--quiet"],
             "Auditoría de tocabilidad", "crítica", True),
        step(f"{p}{k + 4}_arco", "arc_supervisor.py",
             [tgt, "--sections", q, q, q, q, "--tension", "0.3,0.5,0.7,0.9"],
             "¿Coincide el arco real con el planificado?", "crítica", True,
             [f"exists:{W}/metric_{jn}.json"]),
    ]
    nxt = k + 5
    if rec:
        S.append(step(f"r{nxt}_export_yaml", "midi2yaml.py",
                      [f"{W}/melodia.mid", "--compas", i["time_sig"], "--bpm", i["tempo_gen"], "--track", 1],
                      "Serialización didáctica de la melodía a partitura.yaml", "exportación", True,
                      [f"exists:{W}/melodia.mid"]))
        nxt += 1
    S.append(step(f"{p}{nxt}_compare", "analyzer.py", ["compare", f"{W}/orig.mid", tgt],
                  "Comparación vectorial original ↔ resultado", "crítica", True))
    return S


def faithful_steps(i: Dict, O: Dict) -> List[Dict]:
    S: List[Dict] = _prep_steps(i, O)
    clean = f"{W}/orig.mid"
    key_long = i["key_long"]
    cand = []
    if "base" in O["variants"]:
        S += [
            step("f10_base_perf", "performer.py",
                 [clean, "--output", f"{W}/base_perf.mid", "--style", "clasico", "--intensity", "0.4"],
                 "Expresión conservando notas e instrumentos", "interpretación", True,
                 output=f"{W}/base_perf.mid"),
            _meta_step("f11_base_meta", f"{W}/base_perf.mid", f"{W}/base.mid", i, O,
                       note="Compás/armadura/tempo del original (sin aplanar modulaciones)"
                       if O["meta_policy"] == "copy" else "Compás/armadura/tempo del original (aplanados)",
                       when=[f"exists:{W}/base_perf.mid"]),
        ]
        cand.append(f"{W}/base.mid")
    if "rehar" in O["variants"]:
        S += [
            step("f20_rehar", "reharmonizer.py",
                 [clean, "--key", key_long, "--out-dir", f"{W}/rehar", "--report"],
                 "Rearmoniza conservando la melodía", "armonía", True, output=f"{W}/rehar"),
            step("f21_rehar_pick", SELF, ["_first", "--glob", f"{W}/rehar/*.mid", "-o", f"{W}/rehar_raw.mid"],
                 "Toma la primera propuesta de rearmonización", "armonía", True,
                 [f"exists:{W}/rehar"], f"{W}/rehar_raw.mid"),
            step("f22_rehar_perf", "performer.py",
                 [f"{W}/rehar_raw.mid", "--output", f"{W}/rehar_perf.mid", "--style", "clasico",
                  "--intensity", "0.3"], "Expresión de la versión rearmonizada", "interpretación", True,
                 [f"exists:{W}/rehar_raw.mid"], f"{W}/rehar_perf.mid"),
            _meta_step("f23_rehar_meta", f"{W}/rehar_perf.mid", f"{W}/rehar.mid", i, O,
                       note="Metadatos del original", when=[f"exists:{W}/rehar_perf.mid"]),
        ]
        cand.append(f"{W}/rehar.mid")
    if "orq" in O["variants"]:
        S += [
            step("f30_orq", "piano_expander.py",
                 [clean, "--output", f"{W}/orq_raw.mid", "--template", "chamber"],
                 "Reorquesta conservando las notas", "orquestación", True, output=f"{W}/orq_raw.mid"),
            step("f31_orq_perf", "performer.py",
                 [f"{W}/orq_raw.mid", "--output", f"{W}/orq_perf.mid", "--style", "clasico",
                  "--intensity", "0.3"], "Expresión de la versión orquestada", "interpretación", True,
                 [f"exists:{W}/orq_raw.mid"], f"{W}/orq_perf.mid"),
            _meta_step("f32_orq_meta", f"{W}/orq_perf.mid", f"{W}/orq.mid", i, O,
                       note="Metadatos del original", when=[f"exists:{W}/orq_perf.mid"]),
        ]
        cand.append(f"{W}/orq.mid")
    S.append(step("f90_choose", SELF,
                  ["choose", clean, "--candidates"] + cand + ([clean] if O["original_candidate"] else [])
                  + ["--min-melody", O["min_melody"], "--tol", O["tol"],
                     "--fidelity-engine", O["fidelity_engine"], "--select", O["select"],
                     "--tools", "$TOOLS", "--output", "$OUT/obra_fiel.mid",
                     "--report", f"{W}/seleccion.json", "--no-color"],
                  "Gate de melodía y, entre las que lo superan, "
                  + ("la de mayor calidad" if O["select"] == "quality" else "la de mayor fidelidad"),
                  "selección", output="$OUT/obra_fiel.mid"))
    S += critique_steps(i, O, "faithful")
    return S


def _harmony_order(O: Dict) -> List[str]:
    ton, cap, cpg = f"{W}/chords_ton.mid", f"{W}/orig_chords.mid", f"{W}/cpg.mid"
    return {"tonicized": [ton, cap, cpg], "captured": [cap, ton, cpg],
            "generated": [cpg, cap, ton]}[O["harmony"]]


def recompose_steps(i: Dict, O: Dict) -> List[Dict]:
    S: List[Dict] = _prep_steps(i, O)
    clean = f"{W}/orig.mid"
    tonic, mode, ks, kl = i["tonic"], i["mode"], i["key_short"], i["key_long"]
    ts, bpb, nb = i["time_sig"], int(round(i["beats_per_bar"])), i["n_bars"]
    tempo_i = int(round(i["tempo_gen"]))
    m_idx, seed = i["melody_track"], O["seed"]
    emo = O["emotion"] or i["emotion"]
    chords, bajo, acomp, mel = f"{W}/chords_src.mid", f"{W}/bajo.mid", f"{W}/acomp.mid", f"{W}/melodia.mid"
    S += [
        step("r01_armonia", "harmonic_analyzer.py",
             [clean, "--key", ks, "--json", f"{W}/chords_orig.json", "--max-bars", "24"],
             "Armonía funcional del original", "análisis", True),
        step("r02_metrica", "metric_analyzer.py", [clean, "--json", f"{W}/metric_orig.json"],
             "Compás, síncopa, nPVI e hipermetro", "análisis", True),
        step("r03_melodia", "melody_critic.py", [clean, "--track", m_idx],
             f"Crítica melódica de la pista {m_idx}", "análisis", True),
        step("r04_dna", "midi_dna_unified.py",
             [clean, "--fingerprint-only", "--output", f"{W}/orig.fingerprint.json"],
             "Huella vectorial (ADN)", "análisis", True),
        step("r05_split", "midi_split_harmony.py",
             [clean, "--out-dir", W, "--split-files", "--key", kl, "--beats-per-bar", bpb,
              "--tempo", tempo_i, "--report"],
             "Separación melodía/armonía (HMM)", "captura", True, output=f"{W}/orig_chords.mid"),
        step("r06_cpg", "chord_progression_generator.py",
             [f"pieza en {kl} de carácter {emo}, compás {ts}, con arco hacia un clímax tardío",
              "--key", tonic, "--mode", mode, "--style", "auto", "--bars", nb, "--beats", bpb,
              "--tempo", tempo_i, "--tension-profile", "arch", "--seed", seed + 2,
              "--export-midi", f"{W}/cpg.mid", "--export-text", f"{W}/cpg.chords.txt",
              "--export-json", f"{W}/cpg.json"],
             "Alternativa generativa: progresión desde la intención", "armonía", True,
             output=f"{W}/cpg.mid"),
        step("r07_tonic", "tonicization_engine.py",
             [f"{W}/orig_chords.mid", "--key", ks, "--density", O["tonic_density"], "--min-gap", "4",
              "--seventh", "--seed", "3", "--output", f"{W}/chords_ton.mid"],
             "Inserta V7/x sobre la armonía capturada", "armonía", True,
             [f"exists:{W}/orig_chords.mid"], f"{W}/chords_ton.mid"),
        step("r08_chords", SELF, ["_first", "--files"] + _harmony_order(O) + ["-o", chords],
             {"tonicized": "Base armónica: tonicizada › capturada › generada",
              "captured": "Base armónica: capturada › tonicizada › generada",
              "generated": "Base armónica: generada › capturada › tonicizada"}[O["harmony"]],
             "armonía", output=chords),
        step("r09_bajo", "bass_line_composer.py",
             ["", "--key", tonic, "--mode", mode, "--bars", nb, "--tempo", tempo_i, "--time-sig", ts,
              "--style", i["bass_style"], "--chords-midi", chords, "--seed", seed + 6, "--output", bajo],
             f"Línea de bajo ({i['bass_style']}) sobre la armonía", "bajo", True,
             [f"exists:{chords}"], bajo),
        step("r09b_bajo_pedal", "bass_line_composer.py",
             ["", "--key", tonic, "--mode", mode, "--bars", nb, "--tempo", tempo_i, "--time-sig", ts,
              "--style", i["bass_style"], "--chords", " ".join(["I:4"] * nb), "--seed", seed + 6,
              "--output", bajo],
             "(sin acordes) pedal de tónica", "bajo", True, [f"missing:{bajo}"], bajo),
    ]
    pattern = O["pattern"] if O["pattern"] != "auto" else (
        "alberti" if bpb >= 4 else ("arpeggio-up" if bpb >= 3 else "block"))
    S += [
        step("r10_patron", "chord_pattern_voicer.py",
             [chords, "--track", f"@track0:{chords}", "--pattern", pattern, "--out", acomp, "--report"],
             f"Refiguración del patrón armónico ({pattern})", "textura", True,
             [f"exists:{chords}"], acomp),
        step("r10b_acomp_copia", SELF, ["_first", "--files", chords, "-o", acomp],
             "(fallback) acordes en bloque", "textura", True,
             [f"missing:{acomp}", f"exists:{chords}"], acomp),
    ]
    # ── melodía: el orden de preferencia lo fija --melody-source ──────────────
    base = ["--key", tonic, "--mode", mode, "--bars", nb, "--time-sig", ts, "--tempo", tempo_i,
            "--profile", "pastoral" if mode == "major" else "elegiac", "--rhythm", "flowing",
            "--contour", "arch", "--range", "60", "84"]
    chordsargs = ["--chords-midi", chords, "--chords-midi-track", f"@track0:{chords}",
                  "--chords-midi-channel", f"@chan0:{chords}", "--chords-midi-window", max(1, bpb)]
    tail = ["--candidates", 3, "--seed", seed, "--output", mel]
    gens = {
        "motif": ("r11_melodia", "melody_conditioned_v5.py",
                  ["--engine", "motif", "--motif", i["motif"] or ""] + base + chordsargs + tail,
                  "Melodía sembrada con el motivo real", [f"exists:{chords}"], bool(i["motif"])),
        "guided": ("r11b_melodia_guiada", "melody_conditioned_v5.py",
                   ["--engine", "chord_guided"] + base + chordsargs + tail,
                   "Melodía guiada por acordes", [f"exists:{chords}"], True),
        "free": ("r11c_melodia_libre", "melody_conditioned_v5.py",
                 ["--engine", "chord_guided"] + base + tail, "Melodía sin acordes", [], True),
        "captured": ("r11d_melodia_capturada", SELF, ["_first", "--files", f"{W}/orig_melody.mid", "-o", mel],
                     "Melodía capturada por el HMM (puede venir invertida)",
                     [f"exists:{W}/orig_melody.mid"], True),
    }
    order = {"motif": ["motif", "guided", "free", "captured"],
             "guided": ["guided", "free", "captured"],
             "captured": ["captured", "motif", "guided", "free"]}[O["melody_source"]]
    first = True
    for g_ in order:
        name, tool, args, note, conds, usable = gens[g_]
        if not usable:
            continue
        S.append(step(name, tool, args, note if first else "(fallback) " + note.lower(), "melodía", True,
                      conds + ([] if first else [f"missing:{mel}"]), mel))
        first = False
    # ── ensamblado: con los stems que haya (como reconstruct.py) ──────────────
    stems = [("bajo", bajo, 0), ("acomp", acomp, 1), ("mel", mel, 2)]
    obra_base = f"{W}/obra_base.mid"
    subsets = [("r12_merge", [0, 1, 2]), ("r12a_merge_bajo_mel", [0, 2]),
               ("r12b_merge_acomp_mel", [1, 2]), ("r12c_merge_bajo_acomp", [0, 1])]
    for name, idx in subsets:
        conds = [f"{'exists' if k in idx else 'missing'}:{stems[k][1]}" for k in range(3)]
        S.append(step(name, "midi_merge.py",
                      [stems[k][1] for k in idx] + ["--output", obra_base, "--channels"]
                      + [f"{stems[k][1]}:{stems[k][2]}" for k in idx],
                      "Stems (" + " · ".join(stems[k][0] for k in idx) + ") → MIDI multicanal",
                      "ensamblado", True, conds, obra_base))
    S.append(step("r12d_merge_unico", SELF, ["_first", "--files", mel, acomp, bajo, "-o", obra_base],
                  "(fallback) un único stem disponible", "ensamblado", True,
                  [f"missing:{obra_base}"], obra_base))
    mt = f"@count:{bajo}|{acomp}|{mel}"      # pista de la melodía (1-based): bajo=1, acomp=2, mel=3
    perf_files = [f"{W}/obra_climax.mid", obra_base]
    if "climax" not in O["skip"]:
        S.append(step("r13_climax", "climax_engineer.py",
                      [obra_base, "--output", f"{W}/obra_climax.mid", "--position", i["climax"],
                       "--width", 1, "--hollow-bars", 1, "--intensity", 0.5, "--melody-track", mt],
                      f"Clímax en el {int(i['climax'] * 100)} % (posición real del original)", "forma", True,
                      [f"exists:{obra_base}"], f"{W}/obra_climax.mid"))
    S += [
        step("r14a_perf_in", SELF, ["_first", "--files"] + perf_files + ["-o", f"{W}/perf_in.mid"],
             "Entrada del performer", "interpretación", True, output=f"{W}/perf_in.mid"),
        step("r14_performer", "performer.py",
             [f"{W}/perf_in.mid", "--output", f"{W}/obra_perf.mid", "--style", i["perf_style"],
              "--intensity", 0.5, "--melody-track", mt],
             f"Dinámicas, pedal y fraseo ({i['perf_style']})", "interpretación", True,
             [f"exists:{W}/perf_in.mid"], f"{W}/obra_perf.mid"),
    ]
    if "tempo" not in O["skip"]:
        tg = i["tempo_gen"]
        S.append(step("r15_tempo", "tempo_designer.py",
                      [f"{W}/obra_perf.mid", "--output", f"{W}/obra_tempo.mid", "--base", tg,
                       "--section", f"1:{round(tg * 0.96)}", "--section", f"{max(1, nb // 2)}:{round(tg * 1.04)}",
                       "--section", f"{max(1, int(nb * 0.8))}:{round(tg * 0.96)}",
                       "--final-rit", 2, "--final-rit-factor", 1.2, "--shape", "smooth"],
                      "Arco agógico con ritardando final", "interpretación", True,
                      [f"exists:{W}/obra_perf.mid"], f"{W}/obra_tempo.mid"))
    S += [
        step("r16_pick", SELF,
             ["_first", "--files", f"{W}/obra_tempo.mid", f"{W}/obra_perf.mid", f"{W}/obra_climax.mid",
              obra_base, "-o", f"{W}/final_raw.mid"],
             "Última etapa disponible del pipeline", "ensamblado", output=f"{W}/final_raw.mid"),
        _meta_step("r17_meta", f"{W}/final_raw.mid", f"{W}/final_meta.mid", i, O, only="ts,key",
                   note="Compás/armadura del original (el tempo lo fija el arco agógico)",
                   when=[f"exists:{W}/final_raw.mid"]),
    ]
    if O["doctor_available"]:
        S.append(step("r18_final", "midi_doctor.py", ["fix", f"{W}/final_meta.mid", "-o", "$OUT/obra_final.mid"],
                      "Saneado final", "metadatos", True, [f"exists:{W}/final_meta.mid"], "$OUT/obra_final.mid"))
    S.append(step("r18b_final_copia", SELF,
                  ["_first", "--files", f"{W}/final_meta.mid", f"{W}/final_raw.mid", "-o", "$OUT/obra_final.mid"],
                  "(fallback) entrega de la última etapa" if O["doctor_available"] else "Entrega de la última etapa",
                  "metadatos", False, ["missing:$OUT/obra_final.mid"], "$OUT/obra_final.mid"))
    S += critique_steps(i, O, "recompose")
    return S


def build_plan(orig: str, out_dir: str, O: Dict, tools: str) -> Dict:
    info = analyze_midi(orig, {**O, "tools": tools})
    steps: List[Dict] = []
    if O["mode"] in ("faithful", "both"):
        steps += faithful_steps(info, O)
    if O["mode"] in ("recompose", "both"):
        have = {s["name"] for s in steps}
        steps += [s for s in recompose_steps(info, O) if s["name"] not in have]
    return {"version": VERSION, "mode": O["mode"], "original": os.path.abspath(orig),
            "out_dir": os.path.abspath(out_dir), "min_melody": O["min_melody"], "tol": O["tol"],
            "options": O, "dna": info, "steps": steps}


def plan_to_runner_yaml(plan: Dict) -> str:
    """Plan en formato 'manual' de runner.py (ignora when/optional y no resuelve
    los marcadores @track0/@chan0/@count/@quarter: solo este script los entiende)."""
    L = [f"# Plan de reconstrucción ({plan['mode']}) — generado por reconstructor.py",
         "# runner.py no evalúa 'when': los pasos fallback se ejecutarán igualmente.",
         "# Los marcadores @track0/@chan0/@count/@quarter solo los resuelve reconstructor.py.",
         f"original: {plan['original']}", "steps:"]
    for s in plan["steps"]:
        cmd = " ".join(["python", shlex.quote(s["tool"])] +
                       [shlex.quote(a.replace("$OUT", plan["out_dir"]).replace("$ORIG", plan["original"]))
                        for a in s["args"]])
        L += [f"  - name: {s['name']}", f"    label: \"{s['note']}\"", f"    cmd: {json.dumps(cmd)}"]
    return "\n".join(L) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _first_track(path: str) -> Tuple[int, int]:
    mid = _load(path)
    best = (0, 0, -1)
    for i, tr in enumerate(mid.tracks):
        n = sum(1 for m in tr if m.type == "note_on" and m.velocity > 0)
        ch = next((m.channel for m in tr if m.type == "note_on"), 0)
        if n > best[2]:
            best = (i, ch, n)
    return best[0], best[1]


def count_existing(paths: List[str]) -> int:
    return max(1, sum(1 for p in paths if os.path.isfile(p)))


def quarter_bars(json_path: str, default: int) -> int:
    d = read_json(json_path) or {}
    try:
        n = int(d.get("n_bars") or default)
    except (TypeError, ValueError):
        n = default
    return max(1, n // 4)


class Ctx:
    def __init__(self, plan: Dict, tools: str):
        self.out = plan["out_dir"]
        self.orig = plan["original"]
        self.tools = tools

    def sub(self, s: str) -> str:
        return s.replace("$OUT", self.out).replace("$TOOLS", self.tools).replace("$ORIG", self.orig)

    def resolve(self, tok: str) -> str:
        tok = self.sub(tok)
        m = TOKEN.match(tok)
        if m:
            kind, arg = m.groups()
            if kind in ("track0", "chan0"):
                idx, ch = _first_track(arg)
                return str(idx if kind == "track0" else ch)
            if kind == "count":
                return str(count_existing(arg.split("|")))
            f, _, d = arg.rpartition("|")
            return str(quarter_bars(f, int(d or 4)))
        return tok

    def cond_ok(self, cond: str) -> bool:
        kind, _, path = cond.partition(":")
        ex = os.path.exists(self.sub(path))
        return ex if kind == "exists" else (not ex)


def tool_path(ctx: Ctx, tool: str) -> str:
    if tool == SELF:
        return os.path.abspath(__file__)
    return os.path.join(ctx.tools, tool)


def render_cmd(ctx: Ctx, s: Dict) -> List[str]:
    return [sys.executable, tool_path(ctx, s["tool"])] + [ctx.resolve(a) for a in s["args"]]


def execute(plan: Dict, tools: str, timeout: int, resume: bool, strict: bool, dry: bool) -> List[Dict]:
    ctx = Ctx(plan, tools)
    work = os.path.join(ctx.out, "work")
    logs = os.path.join(work, "logs")
    os.makedirs(logs, exist_ok=True)
    results: List[Dict] = []
    total = len(plan["steps"])
    print(f"\n{'═' * 78}\n  {_c('B')}RECONSTRUCTOR · run{_c('R')}  modo={plan['mode']}  "
          f"{total} pasos  →  {ctx.out}\n{'═' * 78}")
    for n, s in enumerate(plan["steps"], 1):
        rec = {"name": s["name"], "tool": s["tool"], "note": s["note"], "stage": s["stage"],
               "optional": s["optional"], "status": "pending", "rc": None, "seconds": 0.0,
               "args": s["args"], "when": s["when"]}
        results.append(rec)
        tag = f"[{n:>2}/{total}] {s['name']:<24}"
        if not all(ctx.cond_ok(c) for c in s["when"]):
            rec["status"] = "skipped"
            print(f"  {_c('G')}{tag} · omitido (condición){_c('R')}")
            continue
        out = ctx.sub(s["output"]) if s.get("output") else None
        if resume and out and os.path.exists(out):
            rec["status"] = "cached"
            print(f"  {_c('G')}{tag} · en caché{_c('R')}")
            continue
        if dry:
            rec["status"] = "dry"
            print(f"  {_c('G')}{tag} · (dry-run){_c('R')}")
            continue
        try:
            cmd = render_cmd(ctx, s)
        except Exception as e:  # noqa: BLE001
            rec["status"], rec["error"] = "failed", f"no se pudo resolver el comando: {e}"
            print(f"  {_c('RED')}{tag} ✗ {rec['error']}{_c('R')}")
            if strict and not s["optional"]:
                break
            continue
        if not os.path.isfile(cmd[1]):
            rec["status"], rec["error"] = "missing_tool", f"no existe {os.path.basename(cmd[1])}"
            print(f"  {_c('YEL') if s['optional'] else _c('RED')}{tag} ✗ herramienta ausente: "
                  f"{s['tool']}{_c('R')}")
            if strict and not s["optional"]:
                break
            continue
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=ctx.tools, capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, "PYTHONWARNINGS": "ignore",
                                    "PYGAME_HIDE_SUPPORT_PROMPT": "1"})
            text = ANSI.sub("", (p.stdout or "") + (p.stderr or ""))
            rc = p.returncode
        except subprocess.TimeoutExpired:
            text, rc = f"(timeout {timeout}s)", 124
        except Exception as e:  # noqa: BLE001
            text, rc = str(e), 1
        rec["seconds"] = round(time.time() - t0, 2)
        rec["rc"] = rc
        out_ok = (out is None) or os.path.exists(out)
        if rc == 0 and out_ok:
            rec["status"] = "ok"
        else:
            rec["status"] = "warn" if s["optional"] else "failed"
            rec["error"] = (text.strip().splitlines() or ["sin salida"])[-1][:200] if rc != 0 \
                else f"no produjo {os.path.basename(out)}"
        rec["tail"] = "\n".join(text.strip().splitlines()[-3:])
        logp = os.path.join(logs, f"{s['name']}.log")
        with open(logp, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n" + text)
        rec["log"] = os.path.relpath(logp, ctx.out)
        col = {"ok": _c("GRN"), "warn": _c("YEL"), "failed": _c("RED")}[rec["status"]]
        icon = {"ok": "✓", "warn": "⚠", "failed": "✗"}[rec["status"]]
        extra = f" {_c('G')}{rec.get('error', '')}{_c('R')}" if rec["status"] != "ok" else ""
        print(f"  {col}{tag} {icon}{_c('R')} {_c('G')}{rec['seconds']:>5.1f}s  {s['tool']}{_c('R')}{extra}")
        if rec["status"] == "failed" and strict:
            break
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  RECETA E INFORME
# ══════════════════════════════════════════════════════════════════════════════

def write_recipe(plan: Dict, results: List[Dict], tools: str) -> str:
    out = plan["out_dir"]
    O = plan.get("options", {})
    rel = O.get("recipe_paths") == "relative"
    L = ["#!/usr/bin/env bash", "# " + "═" * 70,
         f"# RECETA — reconstructor.py ({plan['mode']}) sobre {os.path.basename(plan['original'])}",
         "# " + "═" * 70, "set -u", 'OUT="$(cd "$(dirname "$0")" && pwd)"']
    if rel:
        def r(p):
            return os.path.relpath(p, out)
        L += [f'TOOLS="${{TOOLS:-$(cd "$OUT/{r(tools)}" && pwd)}}"',
              f'ORIG="${{ORIG:-$OUT/{r(plan["original"])}}}"',
              f'RECONSTRUCTOR="${{RECONSTRUCTOR:-$OUT/{r(os.path.abspath(__file__))}}}"']
    else:
        L += [f'TOOLS="${{TOOLS:-{tools}}}"', f'ORIG="${{ORIG:-{plan["original"]}}}"',
              f'RECONSTRUCTOR="${{RECONSTRUCTOR:-{os.path.abspath(__file__)}}}"']
    L += ['mkdir -p "$OUT/work"', 'cd "$TOOLS"', ""]

    helpers = {"track0": "_track0", "chan0": "_chan0", "count": "_count", "quarter": "_quarter"}

    def q(tok: str) -> str:
        m = TOKEN.match(tok)
        if m:
            kind, arg = m.groups()
            parts = arg.split("|") if kind in ("count", "quarter") else [arg]
            inner = " ".join('"' + p.replace('"', '\\"') + '"' for p in parts)
            return f'"$(python3 "$RECONSTRUCTOR" {helpers[kind]} {inner})"'
        if tok == "" or re.search(r"[$ ]", tok):
            return '"' + tok.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`") + '"'
        return shlex.quote(tok) if re.search(r"[^A-Za-z0-9_@%+=:,./-]", tok) else tok

    by_name = {r_["name"]: r_ for r_ in results}
    cur = None
    for s in plan["steps"]:
        rr = by_name.get(s["name"], {})
        stat = rr.get("status", "?")
        if s["stage"] != cur:
            cur = s["stage"]
            L.append(f"# ── {cur.upper()} " + "─" * max(0, 60 - len(cur)))
        tool = '"$RECONSTRUCTOR"' if s["tool"] == SELF else f'"$TOOLS/{s["tool"]}"'
        cmd = f"python3 {tool} " + " ".join(q(a) for a in s["args"])
        L.append(f"# {s['name']} — {s['note']}" + (f"   [{stat}]" if stat not in ("ok", "?") else ""))
        guards = []
        for c in s["when"]:
            kind, _, path = c.partition(":")
            guards.append(f'[ -e "{path}" ]' if kind == "exists" else f'[ ! -e "{path}" ]')
        line = (" && ".join(guards) + " && " if guards else "") + cmd
        if s["optional"] or guards:
            line += " || true"
        L.append(line)
    L.append('echo "✔ receta completada"')
    path = os.path.join(out, "receta.sh")
    with open(_ensure_parent(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    os.chmod(path, 0o755)
    return path


def write_report(plan: Dict, results: List[Dict], summary: Dict) -> str:
    out = plan["out_dir"]
    i = plan["dna"]
    O = plan.get("options", {})
    full = O.get("report_style") == "full"
    L = [f"# Reconstrucción de `{os.path.basename(plan['original'])}`", "",
         f"> Generado por `reconstructor.py` · modo **{plan['mode']}**", ""]
    if plan["mode"] in ("recompose", "both"):
        L += ["> **Aviso:** el modo `recompose` *recompone*: conserva tonalidad, compás, progresión y un",
              "> motivo, pero regenera el resto. Es una pieza nueva sembrada por el original (la melodía",
              "> suele conservarse en torno al 25 %). Para conservar el material usa `--mode faithful`.", ""]
    L += ["## ADN del original", "",
          f"- Tonalidad: **{i['key_long']}** · compás **{', '.join(i['ts']) or '4/4'}** · "
          f"tempo **{i['tempo']}** BPM · **{i['n_bars']}** compases ({i.get('bars_source', 'ticks')}) · "
          f"{i['n_notes']} notas",
          f"- Programas GM: {i['programs'] or '—'} · melodía en la pista {i['melody_track']} · "
          f"clímax al {int(i['climax'] * 100)} %",
          f"- Motivo semilla: `{i['motif'] or '—'}`",
          f"- Estilos sugeridos: interpretación `{i['perf_style']}` · bajo `{i['bass_style']}` "
          f"({i.get('style_source', '—')})", ""]
    if O:
        L += ["## Decisiones de diseño", "", "| Opción | Valor |", "|---|---|"]
        for k in ("mode", "clean", "meta_engine", "meta_policy", "fidelity_engine", "select",
                  "original_candidate", "harmony", "melody_source", "bars_from", "style_source",
                  "critique", "skip", "min_melody", "tol", "seed"):
            if k in O:
                v = O[k]
                v = ", ".join(v) if isinstance(v, list) and v else ("—" if isinstance(v, list) else v)
                if k == "original_candidate":
                    L.append(f"| `--no-original-candidate` | {'no (el original compite en la selección)' if v else 'sí (el original no compite)'} |")
                else:
                    L.append(f"| `--{k.replace('_', '-')}` | {v} |")
        L.append("")
    L += ["## Pasos ejecutados", "",
          "| Paso | Etapa | Herramienta | Estado | s | Qué hace |" if full else
          "| Paso | Herramienta | Estado | s | Qué hace |",
          "|---|---|---|---|---:|---|" if full else "|---|---|---|---:|---|"]
    icon = {"ok": "✓", "warn": "⚠ aviso", "failed": "✗ fallo", "skipped": "· omitido",
            "cached": "caché", "dry": "dry-run", "missing_tool": "✗ sin herramienta"}
    for r in results:
        st = icon.get(r["status"], r["status"])
        L.append((f"| `{r['name']}` | {r['stage']} | `{r['tool']}` | {st} | {r['seconds']} | {r['note']} |")
                 if full else f"| `{r['name']}` | `{r['tool']}` | {st} | {r['seconds']} | {r['note']} |")
    for fn, title in (("fidelidad_fiel.json", "obra_fiel.mid"), ("fidelidad_final.json", "obra_final.mid")):
        f = read_json(os.path.join(out, "work", fn))
        if f:
            L += ["", f"## Fidelidad de `{title}` (tolerancia ±{f.get('tol', plan.get('tol'))} pulso · "
                      f"motor {f.get('engine', 'fidelity_meter')})", "", "| Métrica | Valor |", "|---|---:|"]
            for k in ("melody", "pitch", "rhythm", "onset", "chroma", "instr", "meta", "length", "score"):
                v = f.get(k)
                if v is not None:
                    L.append(f"| {k} | {v * 100:.1f} % |" if k != "length" else f"| {k} | {v:.2f}x |")
            if f.get("verdict"):
                L += ["", f"**Veredicto:** {f['verdict']}"]
    sel = read_json(os.path.join(out, "work", "seleccion.json"))
    if sel:
        L += ["", "## Selección entre variantes", "",
              f"Gate de melodía ≥ {sel['min_melody']} · criterio `{sel.get('select', 'quality')}`. "
              f"Ganadora: `{os.path.basename(sel['winner'] or '—')}` ({sel['reason']}).", "",
              "| Candidato | melody | calidad | pitch | onset | chroma | elegible |",
              "|---|---:|---:|---:|---:|---:|:-:|"]
        pc = lambda v: "—" if v is None else f"{v * 100:.1f} %"  # noqa: E731
        for c in sel["candidates"]:
            ql = "—" if c.get("quality") is None else f"{c['quality']:.3f}"
            mt = c.get("metrics") or {}
            L.append(f"| `{os.path.basename(c['file'])}` | {pc(c.get('melody'))} | {ql} | {pc(mt.get('pitch'))} | "
                     f"{pc(mt.get('onset'))} | {pc(mt.get('chroma'))} | {'sí' if c['eligible'] else 'no'} |")
    if full:
        bad = [r for r in results if r["status"] in ("failed", "warn", "missing_tool")]
        L += ["", "## Incidencias de las herramientas en esta pieza", ""]
        if not bad:
            L.append("Ninguna: todos los pasos terminaron correctamente.")
        else:
            L += ["| Paso | Estado | Salida (cola) |", "|---|---|---|"]
            for r in bad:
                tail = (r.get("tail") or r.get("error") or "").replace("|", "/").replace("\n", " ")[:180]
                L.append(f"| `{r['name']}` | {icon.get(r['status'], r['status'])} | {tail} |")
        L += ["", "## Ficheros generados", "", "```", os.path.basename(out)]
        for fn, desc in (("original (work/orig.mid)", "copia de trabajo del original"),
                         ("obra_fiel.mid", "← resultado del modo faithful"),
                         ("obra_final.mid", "← resultado del modo recompose"),
                         ("plan.json", "plan ejecutado"), ("receta.sh", "todas las invocaciones"),
                         ("RECONSTRUCCION.md", "este documento"),
                         ("run_log.json", "registro estructurado"),
                         ("work/", "intermedios, logs y JSON de métricas")):
            if fn.startswith("original") or fn.endswith("/") or os.path.exists(os.path.join(out, fn)):
                L.append(f"├── {fn:<26} {desc}")
        L.append("```")
    L += ["", "## Resultado", ""]
    for k, v in summary.items():
        L.append(f"- {k}: {v}")
    L += ["", "Reproducible con `receta.sh` (variables `TOOLS`, `ORIG` y `RECONSTRUCTOR` sobrescribibles)."]
    path = os.path.join(out, "RECONSTRUCCION.md")
    with open(_ensure_parent(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  SELECCIÓN (gate de melodía + calidad / fidelidad)
# ══════════════════════════════════════════════════════════════════════════════

def fidelity_of(tools: str, orig: str, cand: str, tol: float, engine: str = "auto") -> Dict:
    """Fidelidad de `cand` frente a `orig`. Con engine auto/meter usa fidelity_meter.py
    y, si no está o falla, las métricas internas (así el gate siempre funciona)."""
    if engine in ("auto", "meter"):
        fm = os.path.join(tools, "fidelity_meter.py")
        if not os.path.isfile(fm):
            fm = os.path.join(HERE, "fidelity_meter.py")
        if os.path.isfile(fm):
            fd, tmp = tempfile.mkstemp(suffix=".json")
            os.close(fd)
            try:
                p = subprocess.run([sys.executable, fm, "compare", orig, cand, "--tol", str(tol),
                                    "--json", tmp, "--no-color"], capture_output=True, text=True, timeout=300)
                d = read_json(tmp) if p.returncode == 0 else None
                if d and d.get("melody") is not None:
                    d["engine"] = "meter"
                    return d
            except Exception:  # noqa: BLE001
                pass
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
    d = fidelity_builtin(orig, cand, tol)
    if engine == "meter":
        d["fallback"] = True
    return d


def quality_of(tools: str, cand: str) -> Optional[float]:
    qs = os.path.join(tools, "quality_scorer.py")
    if not os.path.isfile(qs):
        return None
    try:
        p = subprocess.run([sys.executable, qs, cand, "--fast", "--no-color"], capture_output=True,
                           text=True, timeout=300, cwd=tools)
        txt = ANSI.sub("", (p.stdout or "") + (p.stderr or ""))
        m = re.search(r"(\d\.\d{3})\s+\[", txt)
        return float(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


def choose(orig: str, cands: List[str], min_melody: float, tools: str, tol: float,
           select: str = "quality", engine: str = "auto") -> Dict:
    """Gate de melodía y, entre las que lo superan, mayor calidad (select=quality)
    o mayor puntuación de fidelidad (select=fidelity)."""
    files = [c for c in OrderedDict.fromkeys(cands) if os.path.isfile(c)]
    rows = []
    for f in files:
        fid = fidelity_of(tools, orig, f, tol, engine)
        met = fid if fid.get("engine") == "builtin" else fidelity_builtin(orig, f, tol)
        mel = fid.get("melody")
        rows.append({"file": f, "melody": mel, "score": fid.get("score"), "engine": fid.get("engine"),
                     "quality": quality_of(tools, f) if select == "quality" else None,
                     "metrics": {k: met.get(k) for k in ("melody", "pitch", "onset", "chroma", "instr",
                                                         "meta", "progs_kept", "ts_kept", "ks_kept")},
                     "eligible": mel is None or mel >= min_melody})
    for r in rows:
        # el propio original es siempre perfectamente fiel; se marca como elegible
        if os.path.abspath(r["file"]) == os.path.abspath(orig):
            r["eligible"] = True
    pool = [r for r in rows if r["eligible"]]
    winner, reason = None, ""
    if pool:
        if select == "quality":
            winner = max(pool, key=lambda r: ((r["quality"] is not None), r["quality"] or -1, r["melody"] or 0))
            reason = "mayor calidad entre las que superan el gate"
        else:
            winner = max(pool, key=lambda r: (r["score"] or 0, r["melody"] or 0))
            reason = "mayor fidelidad entre las que superan el gate"
    elif rows:
        winner = max(rows, key=lambda r: (r["melody"] or 0))
        reason = "ninguna supera el gate: se elige la de mayor melodía"
    return {"version": VERSION, "original": orig, "min_melody": min_melody, "tol": tol,
            "select": select, "engine": engine,
            "winner": winner["file"] if winner else None, "reason": reason, "candidates": rows}


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARACIÓN: recomposición vs variantes fieles
# ══════════════════════════════════════════════════════════════════════════════

VARIANT_NAMES = {"base": "conservador", "rehar": "rearmonizado", "orq": "orquestado", "orig": "original"}


def _piece_dirs(root: str) -> List[str]:
    if os.path.isfile(os.path.join(root, "plan.json")):
        return [root]
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isfile(os.path.join(root, d, "plan.json"))]


def build_comparison(root: str) -> Optional[str]:
    rows = []
    for d in _piece_dirs(root):
        plan = read_json(os.path.join(d, "plan.json")) or {}
        orig = os.path.join(d, "work", "orig.mid")
        if not os.path.isfile(orig):
            orig = plan.get("original", "")
        if not os.path.isfile(orig):
            continue
        tol = float(plan.get("tol", 0.25))
        sel = read_json(os.path.join(d, "work", "seleccion.json")) or {}
        row = {"id": os.path.basename(d)[:3] if os.path.basename(d)[:3].isdigit() else os.path.basename(d),
               "dir": os.path.basename(d), "variants": {}, "recomposicion": None, "chosen": "—"}
        for c in sel.get("candidates", []):
            nm = VARIANT_NAMES.get(os.path.splitext(os.path.basename(c["file"]))[0])
            if nm and nm != "original":
                row["variants"][nm] = c.get("metrics") or {}
        if sel.get("winner"):
            row["chosen"] = VARIANT_NAMES.get(os.path.splitext(os.path.basename(sel["winner"]))[0], "—")
        fin = os.path.join(d, "obra_final.mid")
        if os.path.isfile(fin):
            row["recomposicion"] = fidelity_builtin(orig, fin, tol)
        if row["variants"] or row["recomposicion"]:
            rows.append(row)
    if not rows:
        return None
    write_json(os.path.join(root, "comparacion.json"), rows)

    def agg(vals):
        v = [x for x in vals if x is not None]
        return (statistics.mean(v), statistics.median(v)) if v else (0.0, 0.0)

    names = ["recomposicion", "conservador", "rearmonizado", "orquestado"]

    def metrics_of(r, n):
        return r["recomposicion"] if n == "recomposicion" else r["variants"].get(n)

    L = ["# Comparación: recomposición vs. reconstrucción fiel", "",
         f"Medición sobre **{len(rows)}** piezas con las métricas internas de reconstructor.py. "
         "La *recomposición* es `obra_final.mid` (`--mode recompose`); las variantes fieles parten del "
         "original y conservan el material (`--mode faithful`).", "",
         "## Retención de contenido", "",
         "| Pipeline | n | Alturas (media) | mediana | Onset+altura | Chroma | Melodía (skyline) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        ms = [m for m in (metrics_of(r, n) for r in rows) if m]
        if not ms:
            continue
        p, o_, c_, s_ = (agg([m.get(k) for m in ms]) for k in ("pitch", "onset", "chroma", "melody"))
        L.append(f"| {n} | {len(ms)} | {p[0] * 100:.1f} % | {p[1] * 100:.1f} % | {o_[0] * 100:.1f} % | "
                 f"{c_[0] * 100:.1f} % | {s_[0] * 100:.1f} % |")
    L += ["", "## Conservación de metadatos e instrumentación", "",
          "| Variante | Instrumentación | Compás | Tonalidad |", "|---|---:|---:|---:|"]
    for n in names[1:]:
        ms = [r["variants"][n] for r in rows if n in r["variants"]]
        if ms:
            L.append(f"| {n} | {sum(1 for m in ms if m.get('progs_kept'))}/{len(ms)} | "
                     f"{sum(1 for m in ms if m.get('ts_kept'))}/{len(ms)} | "
                     f"{sum(1 for m in ms if m.get('ks_kept'))}/{len(ms)} |")
    L += ["", "## Detalle por pieza (retención de alturas)", "",
          "| pieza | recomposición | conservador | rearmonizado | orquestado | elegida |",
          "|---|---:|---:|---:|---:|---|"]

    def cell(m):
        return "—" if not m or m.get("pitch") is None else f"{m['pitch'] * 100:.0f}%"
    for r in rows:
        L.append(f"| {r['id']} | {cell(r['recomposicion'])} | {cell(r['variants'].get('conservador'))} | "
                 f"{cell(r['variants'].get('rearmonizado'))} | {cell(r['variants'].get('orquestado'))} | "
                 f"{r['chosen']} |")
    path = os.path.join(root, "COMPARACION.md")
    with open(_ensure_parent(path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS PÚBLICOS
# ══════════════════════════════════════════════════════════════════════════════

def _outdir(a) -> str:
    return a.out_dir or (os.path.splitext(os.path.abspath(a.original))[0] + "_rec")


def _need_file(path: str) -> bool:
    if not os.path.isfile(path):
        print(f"[ERROR] no existe {path}", file=sys.stderr)
        return False
    return True


def cmd_analyze(a) -> int:
    if not _need_file(a.original):
        return 1
    tools = find_tools_dir(a.tools)
    info = analyze_midi(a.original, {"bars": a.bars, "bars_from": a.bars_from, "tools": tools,
                                     "style_source": "programs"})
    print_analysis(info)
    if a.json:
        write_json(a.json, info)
    return 0


def cmd_plan(a) -> int:
    if not _need_file(a.original):
        return 1
    out = _outdir(a)
    tools = find_tools_dir(a.tools)
    plan = build_plan(a.original, out, plan_options(a, tools), tools)
    pj = os.path.join(out, "plan.json")
    write_json(pj, plan)
    print(f"\n  {_c('B')}Plan{_c('R')} ({plan['mode']}, {len(plan['steps'])} pasos) → {pj}")
    for s in plan["steps"]:
        flag = "opt" if s["optional"] else "   "
        cond = f"  {_c('G')}si {' y '.join(s['when'])}{_c('R')}" if s["when"] else ""
        print(f"   {flag} {s['name']:<24} {s['tool']:<30}{cond}")
    if a.yaml:
        with open(_ensure_parent(a.yaml), "w", encoding="utf-8") as fh:
            fh.write(plan_to_runner_yaml(plan))
        print(f"  {_c('CYA')}YAML para runner.py: {a.yaml}{_c('R')}")
    print()
    return 0


def run_one(a) -> int:
    """Plan + ejecución + receta + informe de un MIDI (lo usan run y batch)."""
    out = os.path.abspath(_outdir(a))
    tools = find_tools_dir(a.tools)
    if a.plan:
        plan = read_json(a.plan)
        if not plan:
            print(f"[ERROR] no se puede leer el plan {a.plan}", file=sys.stderr)
            return 1
        plan["out_dir"], plan["original"] = out, os.path.abspath(a.original)
    else:
        plan = build_plan(a.original, out, plan_options(a, tools), tools)
    work = os.path.join(out, "work")
    if not a.dry_run:
        if a.force and os.path.isdir(work):
            shutil.rmtree(work)
        if a.force or not a.resume:
            for fn in ("obra_fiel.mid", "obra_final.mid"):
                if os.path.exists(os.path.join(out, fn)):
                    os.unlink(os.path.join(out, fn))      # que un resultado viejo no enmascare un fallo
    os.makedirs(work, exist_ok=True)
    write_json(os.path.join(out, "plan.json"), plan)
    results = execute(plan, tools, a.timeout, a.resume and not a.force, a.strict, a.dry_run)

    summary: Dict[str, str] = {}
    rc = 0
    if not a.dry_run:
        by = Counter(r["status"] for r in results)
        summary["pasos"] = ", ".join(f"{v} {k}" for k, v in by.most_common())
        if plan["mode"] in ("faithful", "both"):
            fiel = os.path.join(out, "obra_fiel.mid")
            if not os.path.exists(fiel):
                clean = os.path.join(work, "orig.mid")
                shutil.copyfile(clean if os.path.exists(clean) else plan["original"], fiel)
                summary["obra_fiel.mid"] = "⚠ el pipeline no produjo resultado: se entrega el original saneado"
            else:
                summary["obra_fiel.mid"] = "generado"
        if plan["mode"] in ("recompose", "both"):
            if os.path.exists(os.path.join(out, "obra_final.mid")):
                summary["obra_final.mid"] = "generado"
            else:
                summary["obra_final.mid"] = "✗ no se generó"
                rc = 2
        if plan["mode"] in ("faithful", "both"):
            f = read_json(os.path.join(work, "fidelidad_fiel.json"))
            if f and f.get("melody") is not None and f["melody"] < plan["min_melody"]:
                summary["gate"] = f"melodía {f['melody'] * 100:.1f} % < {plan['min_melody'] * 100:.0f} %"
                rc = rc or 3
    write_json(os.path.join(out, "run_log.json"),
               {"version": VERSION, "mode": plan["mode"], "generated": datetime.now().isoformat(timespec="seconds"),
                "steps": results, "summary": summary})
    recipe = write_recipe(plan, results, tools)
    md = write_report(plan, results, summary)
    print(f"\n{'─' * 78}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"  {_c('CYA')}receta: {recipe}\n  informe: {md}{_c('R')}\n")
    return rc


def cmd_run(a) -> int:
    if not _need_file(a.original):
        return 1
    return run_one(a)


def _entry_src(e: Dict, midis_dir: Optional[str]) -> Optional[str]:
    cands = ([os.path.join(midis_dir, e["file"])] if midis_dir else []) + \
            ([e["path"]] if e.get("path") else []) + [e["file"]]
    return next((c for c in cands if os.path.isfile(c)), None)


def _batch_entries(a) -> List[Tuple[str, str, Optional[str]]]:
    """[(id, ruta del MIDI, familia instrumental)] según --catalog o las entradas."""
    items: List[Tuple[str, str, Optional[str]]] = []
    if a.catalog:
        sel = read_json(a.catalog)
        if not isinstance(sel, list):
            raise ValueError(f"--catalog: {a.catalog} no es una lista JSON")
        for n, e in enumerate(sel, 1):
            src = _entry_src(e, a.midis_dir)
            pid = str(e.get("id", n)).zfill(3)
            if not src:
                print(f"[{pid}] aviso: no se encuentra {e.get('file')}", file=sys.stderr)
                continue
            items.append((pid, src, e.get("program_family")))
    else:
        files: List[str] = []
        for p in a.inputs:
            if os.path.isdir(p):
                files += sorted(os.path.join(p, f) for f in os.listdir(p)
                                if f.lower().endswith((".mid", ".midi")))
            else:
                files += sorted(glob.glob(p)) or [p]
        items = [(f"{n:03d}", f, None) for n, f in enumerate(files, 1) if os.path.isfile(f)]
    if a.id:
        return [x for x in items if x[0] == str(a.id).zfill(3)]
    s = max(1, a.start) - 1
    return items[s: (s + a.limit) if a.limit else None]


def cmd_batch(a) -> int:
    if not a.catalog and not a.inputs:
        print("[ERROR] batch necesita MIDIs/carpetas de entrada o --catalog", file=sys.stderr)
        return 1
    entries = _batch_entries(a)
    if not entries:
        print("[ERROR] ninguna pieza que procesar", file=sys.stderr)
        return 1
    root = os.path.abspath(a.out_dir or "reconstrucciones")
    os.makedirs(root, exist_ok=True)
    need = {"faithful": ["obra_fiel.mid"], "recompose": ["obra_final.mid"],
            "both": ["obra_fiel.mid", "obra_final.mid"]}[a.mode]
    t0 = time.time()
    log, worst = [], 0
    for pid, src, fam in entries:
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(src))[0])[:48]
        piece = os.path.join(root, f"{pid}_{slug}")
        if not a.force and not a.resume and all(os.path.exists(os.path.join(piece, f)) for f in need):
            print(f"[{pid}] ya existe, saltando (usa --resume o --force)")
            log.append({"id": pid, "file": src, "rc": 0, "status": "existente"})
            continue
        print(f"\n{'=' * 78}\n[{pid}] {os.path.basename(src)}")
        b = copy.copy(a)
        b.original, b.out_dir, b.family, b.plan = src, piece, fam, None
        try:
            rc = run_one(b)
            status = "ok" if rc == 0 else f"rc={rc}"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rc, status = 2, f"fatal: {e}"
            write_json(os.path.join(piece, "fatal.json"), {"file": src, "fatal": str(e)})
        worst = max(worst, rc)
        log.append({"id": pid, "file": src, "dir": piece, "rc": rc, "status": status})
    write_json(os.path.join(root, "batch_log.json"),
               {"version": VERSION, "mode": a.mode, "seconds": round(time.time() - t0, 1), "pieces": log})
    if a.mode == "both" and not a.dry_run:
        md = build_comparison(root)
        if md:
            print(f"  {_c('CYA')}comparación: {md}{_c('R')}")
    ok = sum(1 for x in log if x["rc"] == 0)
    print(f"\nProcesadas {ok}/{len(log)} piezas sin incidencias en {round(time.time() - t0, 1)} s → {root}")
    return worst


def cmd_choose(a) -> int:
    if not _need_file(a.original):
        return 1
    tools = find_tools_dir(a.tools)
    select = "fidelity" if a.no_quality else a.select
    res = choose(a.original, a.candidates, a.min_melody, tools, a.tol, select, a.fidelity_engine)
    if not res["winner"]:
        print("[ERROR] ningún candidato existente", file=sys.stderr)
        return 2
    B, R = _c("B"), _c("R")
    print(f"\n  {B}CHOOSE{R}  gate melodía ≥ {a.min_melody} · criterio {select}")
    for r in res["candidates"]:
        ml = "  —  " if r["melody"] is None else f"{r['melody'] * 100:5.1f}%"
        ql = "  —  " if r["quality"] is None else f"{r['quality']:.3f}"
        mark = "★" if r["file"] == res["winner"] else " "
        ok = f"{_c('GRN')}elegible{R}" if r["eligible"] else f"{_c('RED')}bajo el gate{R}"
        print(f"   {mark} {os.path.basename(r['file']):<24} melodía {ml}  calidad {ql}  {ok}")
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    shutil.copyfile(res["winner"], a.output)
    print(f"  {_c('GRN')}✔ {os.path.basename(res['winner'])} → {a.output}{R}  ({res['reason']})\n")
    if a.report:
        write_json(a.report, res)
    return 0


def cmd_compare(a) -> int:
    d = os.path.abspath(a.dir)
    if not os.path.isdir(d):
        print(f"[ERROR] no existe la carpeta {d}", file=sys.stderr)
        return 1
    md = build_comparison(d)
    if not md:
        print(f"[ERROR] {d} no contiene piezas con plan.json y resultados "
              "(¿se ejecutó `run`/`batch` con --mode both o faithful?)", file=sys.stderr)
        return 1
    print(f"  {_c('CYA')}comparación escrita: {md}{_c('R')}")
    return 0


def cmd_report(a) -> int:
    d = os.path.abspath(a.dir)
    plan = read_json(os.path.join(d, "plan.json"))
    log = read_json(os.path.join(d, "run_log.json"))
    if not plan or not log:
        print(f"[ERROR] {d} no contiene plan.json y run_log.json (¿se ejecutó `run`?)", file=sys.stderr)
        return 1
    if a.report_style:
        plan.setdefault("options", {})["report_style"] = a.report_style
    md = write_report(plan, log["steps"], log.get("summary", {}))
    print(f"  {_c('CYA')}informe regenerado: {md}{_c('R')}")
    return 0


# Ayudantes internos (los usan los planes y las recetas)
def cmd__first(a) -> int:
    files = list(a.files or [])
    if a.glob:
        files += sorted(glob.glob(a.glob))
    for f in files:
        if os.path.isfile(f):
            os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
            shutil.copyfile(f, a.output)
            return 0
    print("[ERROR] ninguno de los ficheros existe", file=sys.stderr)
    return 1


def cmd__track0(a) -> int:
    print(_first_track(a.file)[0])
    return 0


def cmd__chan0(a) -> int:
    print(_first_track(a.file)[1])
    return 0


def cmd__count(a) -> int:
    print(count_existing(a.files))
    return 0


def cmd__quarter(a) -> int:
    print(quarter_bars(a.file, a.default))
    return 0


def cmd__meta(a) -> int:
    only = tuple(x.strip() for x in a.only.split(",") if x.strip())
    bad = [x for x in only if x not in _META_KINDS]
    if bad or not only:
        raise ValueError(f"--only: valores no válidos {bad} (usa ts,key,tempo)")
    redo = apply_meta(a.src, a.copy_from, a.output, only, a.policy, a.time_sig, a.key)
    print(f"metadatos reescritos: {', '.join(redo) if redo else 'ninguno (ya coincidían)'} → {a.output}")
    return 0


def cmd__fidelity(a) -> int:
    d = fidelity_builtin(a.original, a.candidate, a.tol)
    if a.json:
        write_json(a.json, d)
    print("  ".join(f"{k}={d[k] * 100:.1f}%" for k in ("melody", "pitch", "onset", "chroma", "score")))
    print(d["verdict"])
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="reconstructor.py",
        description="Reconstruye un MIDI con las herramientas mutopia "
                    "(analyze / plan / run / batch / choose / compare / report).")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    def planopts(p):
        g = p.add_argument_group("general")
        g.add_argument("--mode", choices=["faithful", "recompose", "both"], default="faithful")
        g.add_argument("--out-dir", default=None)
        g.add_argument("--tools", default=None)
        g.add_argument("--min-melody", type=float, default=0.8)
        g.add_argument("--tol", type=float, default=0.25)
        g.add_argument("--seed", type=int, default=5)
        g.add_argument("--skip", default="", help=f"etapas a omitir: {','.join(SKIP_STAGES)}")
        g.add_argument("--no-color", action="store_true")
        f = p.add_argument_group("faithful")
        f.add_argument("--variants", default="base,rehar,orq")
        f.add_argument("--no-original-candidate", action="store_true")
        r = p.add_argument_group("recompose")
        r.add_argument("--emotion", default=None)
        r.add_argument("--tonic-density", type=float, default=0.15)
        r.add_argument("--pattern", default="auto", help="auto | alberti | arpeggio-up | block")
        r.add_argument("--perf-style", default="auto", help="auto | clasico | romantico | mecanico")
        r.add_argument("--bass-style", default="auto", help="auto | classical | pedal")
        r.add_argument("--bars", type=int, default=None)
        d = p.add_argument_group("decisiones de diseño (la 1.ª opción es la predeterminada)")
        d.add_argument("--clean", choices=["auto", "doctor", "none"], default="auto")
        d.add_argument("--meta-engine", choices=["auto", "doctor", "builtin"], default="auto")
        d.add_argument("--meta-policy", choices=["copy", "flatten"], default="copy")
        d.add_argument("--fidelity-engine", choices=["auto", "meter", "builtin"], default="auto")
        d.add_argument("--select", choices=["quality", "fidelity"], default="quality")
        d.add_argument("--harmony", choices=["tonicized", "captured", "generated"], default="tonicized")
        d.add_argument("--melody-source", choices=["motif", "guided", "captured"], default="motif")
        d.add_argument("--bars-from", choices=["ticks", "metric"], default="ticks")
        d.add_argument("--style-source", choices=["programs", "catalog"], default="programs")
        d.add_argument("--critique", choices=["basic", "full"], default="basic")
        d.add_argument("--recipe-paths", choices=["absolute", "relative"], default="absolute")
        d.add_argument("--report-style", choices=["compact", "full"], default="compact")

    def runopts(p, with_plan=True):
        g = p.add_argument_group("ejecución")
        if with_plan:
            g.add_argument("--plan", default=None)
        g.add_argument("--dry-run", action="store_true")
        g.add_argument("--resume", action="store_true")
        g.add_argument("--force", action="store_true")
        g.add_argument("--timeout", type=int, default=300)
        g.add_argument("--strict", action="store_true")

    an = sub.add_parser("analyze", help="ADN del original")
    an.add_argument("original")
    an.add_argument("--json", default=None)
    an.add_argument("--tools", default=None)
    an.add_argument("--bars", type=int, default=None)
    an.add_argument("--bars-from", choices=["ticks", "metric"], default="ticks")
    an.add_argument("--no-color", action="store_true")
    an.set_defaults(fn=cmd_analyze)

    pl = sub.add_parser("plan", help="escribe el plan sin ejecutar")
    pl.add_argument("original")
    planopts(pl)
    pl.add_argument("--yaml", default=None)
    pl.set_defaults(fn=cmd_plan)

    ru = sub.add_parser("run", help="plan + ejecución + selección + receta + informe")
    ru.add_argument("original")
    planopts(ru)
    runopts(ru)
    ru.set_defaults(fn=cmd_run)

    ba = sub.add_parser("batch", help="run sobre varios MIDIs, carpetas o un catálogo JSON")
    ba.add_argument("inputs", nargs="*", help="MIDIs, carpetas o patrones glob")
    ba.add_argument("--catalog", default=None, help="JSON [{id, file, path?, program_family?}]")
    ba.add_argument("--midis-dir", default=None)
    ba.add_argument("--id", default=None)
    ba.add_argument("--start", type=int, default=1)
    ba.add_argument("--limit", type=int, default=None)
    planopts(ba)
    runopts(ba, with_plan=False)
    ba.set_defaults(fn=cmd_batch)

    ch = sub.add_parser("choose", help="elige entre candidatos (gate de melodía, luego calidad)")
    ch.add_argument("original")
    ch.add_argument("--candidates", nargs="+", required=True)
    ch.add_argument("--output", "-o", required=True)
    ch.add_argument("--min-melody", type=float, default=0.8)
    ch.add_argument("--tol", type=float, default=0.25)
    ch.add_argument("--tools", default=None)
    ch.add_argument("--fidelity-engine", choices=["auto", "meter", "builtin"], default="auto")
    ch.add_argument("--select", choices=["quality", "fidelity"], default="quality")
    ch.add_argument("--no-quality", action="store_true", help="atajo de --select fidelity")
    ch.add_argument("--report", default=None)
    ch.add_argument("--no-color", action="store_true")
    ch.set_defaults(fn=cmd_choose)

    cp = sub.add_parser("compare", help="compara recomposición vs variantes fieles")
    cp.add_argument("dir")
    cp.add_argument("--no-color", action="store_true")
    cp.set_defaults(fn=cmd_compare)

    rp = sub.add_parser("report", help="regenera RECONSTRUCCION.md")
    rp.add_argument("dir")
    rp.add_argument("--report-style", choices=["compact", "full"], default=None)
    rp.add_argument("--no-color", action="store_true")
    rp.set_defaults(fn=cmd_report)

    f = sub.add_parser("_first")
    f.add_argument("--files", nargs="*", default=[])
    f.add_argument("--glob", default=None)
    f.add_argument("-o", "--output", required=True)
    f.set_defaults(fn=cmd__first)
    t = sub.add_parser("_track0")
    t.add_argument("file")
    t.set_defaults(fn=cmd__track0)
    h = sub.add_parser("_chan0")
    h.add_argument("file")
    h.set_defaults(fn=cmd__chan0)
    n = sub.add_parser("_count")
    n.add_argument("files", nargs="+")
    n.set_defaults(fn=cmd__count)
    qu = sub.add_parser("_quarter")
    qu.add_argument("file")
    qu.add_argument("default", type=int)
    qu.set_defaults(fn=cmd__quarter)
    me = sub.add_parser("_meta")
    me.add_argument("src")
    me.add_argument("--copy-from", required=True)
    me.add_argument("-o", "--output", required=True)
    me.add_argument("--only", default="ts,key,tempo")
    me.add_argument("--policy", choices=["copy", "flatten"], default="copy")
    me.add_argument("--time-sig", default=None)
    me.add_argument("--key", default=None)
    me.set_defaults(fn=cmd__meta)
    fi = sub.add_parser("_fidelity")
    fi.add_argument("original")
    fi.add_argument("candidate")
    fi.add_argument("--tol", type=float, default=0.25)
    fi.add_argument("--json", default=None)
    fi.set_defaults(fn=cmd__fidelity)
    return ap


def main(argv=None) -> int:
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if getattr(a, "no_color", False):
        _USE_COLOR = False
    try:
        return a.fn(a)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 141
    except (EOFError, OSError, KeyError, IndexError, struct_error) as e:
        print(f"[ERROR] MIDI ilegible o truncado ({str(e) or type(e).__name__}). "
              "Prueba: midi_doctor.py check <fichero>", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[interrumpido]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
