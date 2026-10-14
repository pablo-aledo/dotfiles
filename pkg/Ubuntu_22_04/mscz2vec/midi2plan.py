#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MIDI2PLAN  v1.0                                     ║
║        De un MIDI existente a un plan ejecutable por runner.py               ║
║                                                                              ║
║  runner.py ejecuta planes que salen de theorist.py o narrator.py (desde      ║
║  lenguaje natural). Falta el camino inverso: PARTIR DE UN MIDI, extraer su   ║
║  "brief" (tonalidad, tempo, compás real, forma, estilo) y convertirlo en un  ║
║  pipeline listo para ejecutar. Esta herramienta lo hace, y centraliza los    ║
║  cálculos que cada script de lote acababa reimplementando (y a veces mal):   ║
║  compases compuestos (6/8, 12/8, 3/8), pulsos por compás en negras, tempo,   ║
║  estilo inferido DEL CONTENIDO (no del nombre del fichero).                  ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    brief     Extrae el brief de un MIDI (tonalidad, compás, tempo, forma,    ║
║              estilo sugerido y parámetros recomendados por herramienta)      ║
║    plan      Genera un plan .yaml para runner.py (formato theorist)          ║
║    batch     Aplica brief+plan a una carpeta de MIDIs (y opcionalmente       ║
║              los ejecuta con runner.py); escribe una tabla resumen           ║
║    selftest  Comprueba la aritmética de compases y que el plan es válido     ║
║                                                                              ║
║  USO:                                                                        ║
║    python midi2plan.py brief obra.mid                                        ║
║    python midi2plan.py brief obra.mid --json brief.json                      ║
║    python midi2plan.py plan obra.mid --out-dir plan_obra/                    ║
║    python midi2plan.py plan obra.mid --profile analyze                       ║
║    python midi2plan.py plan obra.mid --bars 16 --variant 3 --style jazz      ║
║    python runner.py plan_obra/plan.yaml                                      ║
║    python midi2plan.py batch midis/ --out-dir lote/ --limit 10               ║
║    python midi2plan.py batch midis/ --out-dir lote/ --run --md resumen.md    ║
║    python midi2plan.py selftest                                              ║
║                                                                              ║
║  PERFILES DE PLAN (--profile):                                               ║
║    reconstruct  (default) analiza el original, regenera armonía, melodía,    ║
║                 bajo, batería y coral con las herramientas de la suite,      ║
║                 interpreta, puntúa y mide fidelidad (incl. forma)            ║
║    analyze      solo análisis del original (armonía, métrica, forma,         ║
║                 catálogos, leitmotivs, calidad)                              ║
║                                                                              ║
║  OPCIONES DE PLAN (plan y batch):                                            ║
║    --out-dir DIR     Carpeta de salida (orig.mid, brief.json, plan.yaml)     ║
║    --bars N|auto     Compases a generar (default: auto = real, acotado)      ║
║    --max-bars N      Tope de compases en modo auto (default: 32)             ║
║    --variant N       Rota motor de melodía, bajo y batería (default: 0)      ║
║    --seed N          Semilla base (default: 100)                             ║
║    --style S         Fuerza el estilo en lugar de inferirlo                  ║
║    --key K           Fuerza la tonalidad (p. ej. "A minor", "F# major")      ║
║    --tempo T         Fuerza el tempo BPM                                     ║
║    --chords-from M   generator (default) | template: cómo fijar la           ║
║                      progresión escrita en el plan                           ║
║    --no-form         No analiza la forma (más rápido; sin pasos de forma)    ║
║    --suite-dir DIR   Directorio de las herramientas (default: este script)   ║
║                                                                              ║
║  NOTAS:                                                                      ║
║    · Los pasos que devuelven rc≠0 al "encontrar hallazgos" (arc_supervisor)  ║
║      se marcan con optional: true; runner.py sin --strict continúa.          ║
║    · midi2yaml.py solo escribe por stdout y no puede ser un paso de runner;  ║
║      el plan usa form_analyzer/harmonic_analyzer en su lugar.                ║
║    · El estilo inferido es HEURÍSTICO (ver "razones" en el brief).           ║
║                                                                              ║
║  DEPENDENCIAS: ninguna (solo stdlib). Usa form_analyzer.py y                 ║
║  chord_progression_generator.py si están junto a este script (opcional).     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import re
import json
import math
import shutil
import argparse
import subprocess
import tempfile
import importlib.util
from pathlib import Path
from collections import defaultdict
from bisect import bisect_right

VERSION = "1.0"
_USE_COLOR = sys.stdout.isatty()


def _c(code):
    if not _USE_COLOR:
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "CYA": "\033[96m",
            "YEL": "\033[93m", "RED": "\033[91m", "GRN": "\033[92m"}.get(code, "")


HERE = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
#  Lectura de MIDI (autocontenida)
# ══════════════════════════════════════════════════════════════════════════════

class Note:
    __slots__ = ("start", "end", "pitch", "vel", "ch", "trk")

    def __init__(self, start, end, pitch, vel, ch, trk):
        self.start, self.end, self.pitch, self.vel, self.ch, self.trk = start, end, pitch, vel, ch, trk


def _vlq(buf, i):
    v = 0
    while True:
        b = buf[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def read_midi(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b"MThd":
        raise ValueError(f"{path}: no es un archivo MIDI")
    n_tracks = int.from_bytes(raw[10:12], "big")
    div = int.from_bytes(raw[12:14], "big")
    if div & 0x8000:
        raise ValueError("división SMPTE no soportada")
    tempos, sigs, notes, names, programs = [], [], [], {}, {}
    i = 14
    for ti in range(n_tracks):
        if raw[i:i + 4] != b"MTrk":
            break
        ln = int.from_bytes(raw[i + 4:i + 8], "big")
        ch = raw[i + 8:i + 8 + ln]
        i += 8 + ln
        t, j, status = 0, 0, 0
        open_n = {}
        while j < len(ch):
            d, j = _vlq(ch, j)
            t += d
            if j >= len(ch):
                break
            b0 = ch[j]
            if b0 & 0x80:
                status = b0
                j += 1
            if status == 0xFF:
                mt = ch[j]
                j += 1
                ml, j = _vlq(ch, j)
                md = ch[j:j + ml]
                j += ml
                if mt == 0x51 and ml == 3:
                    tempos.append((t, int.from_bytes(md, "big")))
                elif mt == 0x58 and ml >= 2:
                    sigs.append((t, md[0], 2 ** md[1]))
                elif mt == 0x03 and ti not in names:
                    names[ti] = md.decode("latin-1", "replace").strip()
            elif status in (0xF0, 0xF7):
                ml, j = _vlq(ch, j)
                j += ml
            else:
                hi, c = status & 0xF0, status & 0x0F
                if hi == 0xC0:
                    programs.setdefault(ti, ch[j])
                    j += 1
                elif hi == 0xD0:
                    j += 1
                else:
                    a, b = ch[j], ch[j + 1]
                    j += 2
                    if hi == 0x90 and b > 0:
                        k = (c, a)
                        if k in open_n:
                            s, v = open_n.pop(k)
                            if t > s:
                                notes.append(Note(s, t, a, v, c, ti))
                        open_n[k] = (t, b)
                    elif hi == 0x80 or (hi == 0x90 and b == 0):
                        k = (c, a)
                        if k in open_n:
                            s, v = open_n.pop(k)
                            if t > s:
                                notes.append(Note(s, t, a, v, c, ti))
        for (c, a), (s, v) in open_n.items():
            if t > s:
                notes.append(Note(s, t, a, v, c, ti))
    notes.sort(key=lambda n: (n.start, n.pitch))
    tempos.sort()
    sigs.sort()
    return {"tpb": div, "tempos": tempos, "sigs": sigs, "notes": notes, "names": names,
            "programs": programs, "n_tracks": n_tracks}


# ══════════════════════════════════════════════════════════════════════════════
#  Aritmética de compás (el origen del bug de los compases compuestos)
# ══════════════════════════════════════════════════════════════════════════════

def beats_per_bar_quarters(num, den):
    """Pulsos de NEGRA por compás. 4/4→4, 3/4→3, 6/8→3, 12/8→6, 3/8→1.5, 2/2→4.
    OJO: la división entera (num * (4 // den)) da 0 para den=8 y no sirve."""
    return num * 4.0 / den


def meter_class(num, den):
    if den in (8, 16) and num in (6, 9, 12) or (den == 4 and False):
        return "compound"
    if den in (8, 16) and num in (3,):
        return "simple-fast"
    if num in (5, 7, 10, 11) or (den == 8 and num in (5, 7, 11, 13)):
        return "irregular"
    return "simple"


def build_bars(mid, notes):
    tpb = mid["tpb"]
    end = max(n.end for n in notes)
    sigs = list(mid["sigs"])
    if not sigs or sigs[0][0] > 0:
        sigs.insert(0, (0, 4, 4))
    bars, t = [], 0
    for si, (st, num, den) in enumerate(sigs):
        nxt = sigs[si + 1][0] if si + 1 < len(sigs) else None
        tpc = max(1, tpb * 4 * num // den)
        t = max(t, st)
        while (t < end if nxt is None else t < nxt):
            bars.append((t, t + tpc, num, den))
            t += tpc
    return bars or [(0, tpb * 4, 4, 4)]


def seconds_at(mid, tick):
    """Segundos hasta 'tick' respetando el mapa de tempo."""
    tpb = mid["tpb"]
    tempos = list(mid["tempos"]) or [(0, 500000)]
    if tempos[0][0] > 0:
        tempos.insert(0, (0, 500000))
    sec, prev_t, prev_us = 0.0, 0, tempos[0][1]
    for t, us in tempos:
        if t >= tick:
            break
        sec += (t - prev_t) * prev_us / 1e6 / tpb
        prev_t, prev_us = t, us
    return sec + (tick - prev_t) * prev_us / 1e6 / tpb


# ══════════════════════════════════════════════════════════════════════════════
#  Tonalidad y rasgos
# ══════════════════════════════════════════════════════════════════════════════

_KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
_MAJ_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MIN_SCALE = {0, 2, 3, 5, 7, 8, 9, 10, 11}        # natural + 7ª sensible


def _corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 1e-12 and db > 1e-12 else 0.0


def estimate_key(pc):
    scores = []
    for tonic in range(12):
        for mode, prof in (("major", _KS_MAJ), ("minor", _KS_MIN)):
            rot = [prof[(i - tonic) % 12] for i in range(12)]
            scores.append((_corr(pc, rot), tonic, mode))
    scores.sort(reverse=True)
    return scores


def musical_features(mid, notes, bars):
    tpb = mid["tpb"]
    mel = [n for n in notes if n.ch != 9]
    pc = [0.0] * 12
    for n in mel:
        pc[n.pitch % 12] += n.end - n.start
    ks = estimate_key(pc) if sum(pc) > 0 else [(0.0, 0, "major")]
    r, tonic, mode = ks[0]
    scale = _MAJ_SCALE if mode == "major" else _MIN_SCALE
    out_key = sum(1 for n in mel if (n.pitch - tonic) % 12 not in scale)
    chrom = out_key / max(1, len(mel))
    # ventanas de un pulso: variedad de clases de altura
    end = max(n.end for n in mel) if mel else 0
    nwin = max(1, int(end // tpb))
    wins = [set() for _ in range(nwin + 1)]
    for n in mel:
        w0, w1 = int(n.start // tpb), int(max(n.start, n.end - 1) // tpb)
        for w in range(w0, min(w1, nwin) + 1):
            wins[w].add(n.pitch % 12)
    sizes = [len(w) for w in wins if w]
    ge4 = sum(1 for s in sizes if s >= 4) / max(1, len(sizes))
    eq3 = sum(1 for s in sizes if s == 3) / max(1, len(sizes))
    poly = sum(sizes) / max(1, len(sizes))
    # ritmo
    off8, sx = 0, 0
    for n in mel:
        fr = (n.start % tpb) / tpb
        if abs(fr - 0.5) < 0.08:
            off8 += 1
        elif abs(fr - 0.25) < 0.06 or abs(fr - 0.75) < 0.06:
            sx += 1
    total = max(1, len(mel))
    n_bars = len(bars)
    vels = [n.vel for n in mel]
    vmean = sum(vels) / len(vels) if vels else 0
    vstd = math.sqrt(sum((v - vmean) ** 2 for v in vels) / len(vels)) if vels else 0
    pitches = [n.pitch for n in mel]
    drums = sum(1 for n in notes if n.ch == 9)
    durs = [(n.end - n.start) / tpb for n in mel]
    return {
        "key_corr": round(r, 3), "tonic_pc": tonic, "mode": mode,
        "alt_key": f"{_NAMES[ks[1][1]]} {ks[1][2]}" if len(ks) > 1 else None,
        "chromaticism": round(chrom, 3), "dense_chord_windows": round(ge4, 3),
        "triad_windows": round(eq3, 3), "avg_pitch_classes_per_beat": round(poly, 2),
        "offbeat8_ratio": round(off8 / total, 3), "sixteenth_ratio": round(sx / total, 3),
        "notes_per_bar": round(len(mel) / max(1, n_bars), 1),
        "velocity_mean": round(vmean, 1), "velocity_std": round(vstd, 1),
        "pitch_min": min(pitches) if pitches else None, "pitch_max": max(pitches) if pitches else None,
        "pitch_range": (max(pitches) - min(pitches)) if pitches else 0,
        "has_drums": drums > 20, "drum_notes": drums,
        "mean_note_beats": round(sum(durs) / len(durs), 3) if durs else 0,
    }


# ── estilo inferido del contenido ─────────────────────────────────────────────

STYLES = ["baroque", "romantic", "jazz", "pop", "modal", "impressionist", "diatonic"]


def infer_style(f, tempo, num, den):
    """Puntúa estilos con reglas simples sobre rasgos del contenido. Devuelve
    (ranking, razones). Heurístico: no sustituye al criterio del usuario."""
    sc = {s: 0.0 for s in STYLES}
    why = defaultdict(list)

    def add(style, v, msg):
        sc[style] += v
        why[style].append(msg)

    if not f["has_drums"]:
        add("baroque", 0.6, "sin percusión")
        add("romantic", 0.6, "sin percusión")
    else:
        add("pop", 1.2, "hay percusión")
        add("jazz", 0.4, "hay percusión")
    if f["chromaticism"] < 0.05:
        add("diatonic", 1.2, f"casi sin notas fuera de tonalidad ({f['chromaticism']:.2f})")
        add("pop", 0.6, "diatónico")
        add("baroque", 0.4, "diatónico")
    elif f["chromaticism"] < 0.12:
        add("baroque", 0.5, "cromatismo moderado")
        add("pop", 0.3, "cromatismo moderado")
        add("romantic", 0.4, "cromatismo moderado")
    else:
        add("romantic", 0.8, f"cromatismo alto ({f['chromaticism']:.2f})")
        add("jazz", 0.8, f"cromatismo alto ({f['chromaticism']:.2f})")
        add("impressionist", 0.6, "cromatismo alto")
    if f["dense_chord_windows"] > 0.35:
        add("jazz", 1.0, f"muchas ventanas con ≥4 clases de altura ({f['dense_chord_windows']:.2f})")
        add("impressionist", 0.5, "armonía densa")
        add("romantic", 0.3, "armonía densa")
    elif f["triad_windows"] > 0.35:
        add("pop", 0.6, "predominio de tríadas")
        add("diatonic", 0.4, "predominio de tríadas")
    if f["offbeat8_ratio"] > 0.18:
        add("jazz", 0.6, "muchas corcheas a contratiempo")
        add("pop", 0.5, "muchas corcheas a contratiempo")
    if f["sixteenth_ratio"] > 0.25 and not f["has_drums"]:
        add("baroque", 1.0, f"movimiento continuo de semicorcheas ({f['sixteenth_ratio']:.2f})")
    if f["velocity_std"] < 4 and not f["has_drums"]:
        add("baroque", 0.5, "dinámica plana")
    if f["velocity_std"] > 12:
        add("romantic", 0.3, "dinámica variada (puede ser solo una interpretación humana)")
    if f["pitch_range"] >= 48:
        add("romantic", 0.5, "tesitura amplia")
    if f["key_corr"] < 0.6:
        add("modal", 1.0, f"perfil de tonalidad ambiguo (r={f['key_corr']:.2f})")
        add("impressionist", 0.5, "tonalidad ambigua")
    if den == 8 and num in (6, 9, 12, 3) and not f["has_drums"]:
        add("baroque", 0.4, "compás compuesto sin percusión (giga/siciliana)")
    if num == 4 and den == 4 and f["has_drums"] and 80 <= tempo <= 150:
        add("pop", 0.4, "4/4 con percusión y tempo medio")
    ranked = sorted(sc.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1]
    margin = top[1] - second[1]
    conf = "alta" if margin > 1.0 else ("media" if margin > 0.4 else "baja")
    return {"style": top[0], "confidence": conf, "ranking": [(k, round(v, 2)) for k, v in ranked[:3]],
            "reasons": why[top[0]]}


# ══════════════════════════════════════════════════════════════════════════════
#  Parámetros recomendados por herramienta
# ══════════════════════════════════════════════════════════════════════════════

# Vocabularios válidos (verificados con suite_index.py show <tool>)
MELODY_ENGINES = ["grammar", "chord_guided", "search", "motif", "tension_arc", "bayes", "interpolate"]
BASS_STYLES = ["classical", "walking", "pop", "ostinato", "blues", "baroque", "latin", "pedal"]
DRUM_44 = ["brushes", "rock_straight", "jazz_ride", "bossa_nova", "half_time", "shuffle", "funk_16th", "reggae_one_drop"]
PERF_STYLES = ["clasico", "romantico", "mecanico"]
LEADSHEET = ["block", "arpeggio", "alberti", "ballad", "pop"]
GEN_STYLES = {"diatonic", "baroque", "jazz", "modal", "romantic", "impressionist", "pop", "flamenco", "auto"}

_BY_STYLE = {
    #            engine         bass         drums(4/4)       leadsheet   perf        song_architect
    "baroque":   ("grammar",    "baroque",   None,            "alberti",  "clasico",   "folk"),
    "romantic":  ("tension_arc", "classical", None,           "arpeggio", "romantico", "ballad"),
    "jazz":      ("chord_guided", "walking",  "jazz_ride",    "ballad",   "romantico", "jazz"),
    "pop":       ("motif",      "pop",       "rock_straight", "pop",      "mecanico",  "pop"),
    "modal":     ("search",     "pedal",     "half_time",     "block",    "clasico",   "rock"),
    "impressionist": ("bayes",  "pedal",     "brushes",       "arpeggio", "romantico", "ballad"),
    "diatonic":  ("chord_guided", "classical", None,          "block",    "clasico",   "folk"),
}


def recommend(style, num, den, has_drums, variant):
    eng, bass, drum, ls, perf, sa = _BY_STYLE.get(style, _BY_STYLE["diatonic"])
    if variant:
        eng = MELODY_ENGINES[(MELODY_ENGINES.index(eng) + variant) % len(MELODY_ENGINES)]
        bass = BASS_STYLES[(BASS_STYLES.index(bass) + variant) % len(BASS_STYLES)]
        perf = PERF_STYLES[(PERF_STYLES.index(perf) + variant) % len(PERF_STYLES)]
        if drum:
            drum = DRUM_44[(DRUM_44.index(drum) + variant) % len(DRUM_44)]
        ls = LEADSHEET[(LEADSHEET.index(ls) + variant) % len(LEADSHEET)]
    # la batería depende del compás; en compuestos no hay arquetipo propio: shuffle (tresillo)
    cls = meter_class(num, den)
    if cls == "compound":
        drum_style = "shuffle"
        drum_note = "6/8, 9/8, 12/8: no hay arquetipo propio; shuffle es el más cercano (pulso ternario)"
    elif (num, den) == (3, 4):
        drum_style, drum_note = "waltz", "3/4 → waltz"
    elif (num, den) == (3, 8):
        drum_style, drum_note = "waltz", "3/8 tratado como vals rápido"
    else:
        drum_style, drum_note = drum, None
    if has_drums and drum_style is None:
        drum_style = "rock_straight"
    if drum_style is None:
        drum_note = "estilo sin percusión característica: no se añade batería"
    leadsheet = "waltz" if (num, den) in ((3, 4), (3, 8)) else ls
    return {"generator_style": style if style in GEN_STYLES else "auto", "melody_engine": eng,
            "bass_style": bass, "drum_style": drum_style, "drum_note": drum_note,
            "leadsheet_pattern": leadsheet, "performer_style": perf, "song_architect_style": sa}


# ══════════════════════════════════════════════════════════════════════════════
#  Brief
# ══════════════════════════════════════════════════════════════════════════════

def _load_form_analyzer(suite_dir):
    p = Path(suite_dir) / "form_analyzer.py"
    if not p.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("form_analyzer", str(p))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def compute_brief(path, suite_dir=HERE, with_form=True, force_key=None, force_tempo=None, force_style=None, variant=0):
    mid = read_midi(path)
    notes = mid["notes"]
    mel = [n for n in notes if n.ch != 9]
    if not mel:
        raise ValueError("el MIDI no contiene notas melódicas")
    bars = build_bars(mid, mel)
    num, den = bars[0][2], bars[0][3]
    changes = sorted({(b[2], b[3]) for b in bars})
    tempos = mid["tempos"]
    tempo = round(60e6 / tempos[0][1]) if tempos else 120
    if force_tempo:
        tempo = int(force_tempo)
    feats = musical_features(mid, notes, bars)
    if force_key:
        m = re.match(r"^\s*([A-Ga-g][#b]?)\s*(major|maj|minor|min|m)?\s*$", force_key)
        if not m:
            raise ValueError("--key debe ser como 'A minor' o 'F# major'")
        root = m.group(1)[0].upper() + m.group(1)[1:]
        mode = "minor" if (m.group(2) or "major").lower() in ("minor", "min", "m") else "major"
    else:
        root, mode = _NAMES[feats["tonic_pc"]], feats["mode"]
    st = infer_style(feats, tempo, num, den)
    style = force_style or st["style"]
    dur = seconds_at(mid, max(n.end for n in mel))
    tracks = []
    for ti in range(mid["n_tracks"]):
        tn = [n for n in notes if n.trk == ti]
        if not tn:
            continue
        drum = sum(1 for n in tn if n.ch == 9) > len(tn) / 2
        mp = sum(n.pitch for n in tn) / len(tn)
        tracks.append({"index": ti, "name": mid["names"].get(ti, ""), "notes": len(tn),
                       "mean_pitch": round(mp, 1), "drums": drum})
    melodic = [t for t in tracks if not t["drums"]]
    for t in melodic:
        t["role"] = "melody" if t is max(melodic, key=lambda x: x["mean_pitch"]) else (
            "bass" if t is min(melodic, key=lambda x: x["mean_pitch"]) and len(melodic) > 1 else "harmony")
    brief = {
        "file": str(path), "tool": "midi2plan", "version": VERSION,
        "duration_s": round(dur, 1), "tpb": mid["tpb"],
        "tempo_bpm": tempo, "tempo_changes": len(tempos) > 1,
        "time_signature": f"{num}/{den}", "time_signature_changes": len(changes) > 1,
        "meter_class": meter_class(num, den),
        "beats_per_bar_quarters": beats_per_bar_quarters(num, den),
        "n_bars": len(bars),
        "key": f"{root} {mode}", "root": root, "mode": mode, "key_correlation": feats["key_corr"],
        "alt_key": feats["alt_key"],
        "style": style, "style_inferred": st["style"], "style_forced": bool(force_style),
        "style_confidence": st["confidence"], "style_ranking": st["ranking"], "style_reasons": st["reasons"],
        "features": {k: feats[k] for k in ("chromaticism", "dense_chord_windows", "offbeat8_ratio",
                                           "sixteenth_ratio", "notes_per_bar", "velocity_std",
                                           "pitch_range", "has_drums")},
        "tracks": tracks,
        "recommend": recommend(style, num, den, feats["has_drums"], variant),
        "form": None,
        "warnings": [],
    }
    if feats["key_corr"] < 0.6:
        brief["warnings"].append(f"tonalidad poco clara (r={feats['key_corr']:.2f}); alternativa: {feats['alt_key']}")
    if len(changes) > 1:
        brief["warnings"].append("el MIDI cambia de compás: el plan usa el compás inicial")
    if den >= 16 or (num, den) == (1, 4):
        brief["warnings"].append(f"compás declarado atípico ({num}/{den}): posible metadato erróneo del MIDI; "
                                 "revisa los compases con form_analyzer detect --timesig")
    if meter_class(num, den) in ("irregular",):
        brief["warnings"].append(f"compás irregular {num}/{den}: batería y plantillas pueden no encajar")
    if with_form:
        fa = _load_form_analyzer(suite_dir)
        if fa is not None:
            try:
                r = fa.analyze(str(path))
                brief["form"] = {"form": r["form"], "form_name": r["form_name"],
                                 "sections": [{"label": s["label"], "start_bar": s["start_bar"], "n_bars": s["n_bars"]}
                                              for s in r["sections"]],
                                 "repetition_coverage": r["repetition_coverage"]}
            except Exception as e:                       # la forma es opcional
                brief["warnings"].append(f"form_analyzer falló: {e}")
        else:
            brief["warnings"].append("form_analyzer.py no encontrado: sin forma")
    return brief


def print_brief(b):
    print(f"\n{'═' * 78}")
    print(f"  {_c('B')}MIDI2PLAN v{VERSION} · brief · {b['file']}{_c('R')}")
    print(f"{'═' * 78}")
    print(f"  Tonalidad : {b['key']}  (r={b['key_correlation']}; alternativa {b['alt_key']})")
    print(f"  Tempo     : {b['tempo_bpm']} BPM" + (" (varía)" if b["tempo_changes"] else ""))
    print(f"  Compás    : {b['time_signature']} [{b['meter_class']}] → {b['beats_per_bar_quarters']:g} negras/compás"
          + (" (cambia)" if b["time_signature_changes"] else ""))
    print(f"  Duración  : {b['duration_s']} s · {b['n_bars']} compases")
    print(f"  Estilo    : {_c('GRN')}{b['style']}{_c('R')}"
          + (" (forzado)" if b["style_forced"] else f" — confianza {b['style_confidence']}; "
             + ", ".join(f"{k} {v}" for k, v in b["style_ranking"])))
    if b["style_reasons"] and not b["style_forced"]:
        print(f"              razones: {'; '.join(b['style_reasons'][:4])}")
    if b["form"]:
        print(f"  Forma     : {b['form']['form']}  ({b['form']['form_name']})")
    print(f"  Pistas    : " + ", ".join(f"{t['index']}:{t.get('role', 'drums')}({t['notes']})" for t in b["tracks"]))
    r = b["recommend"]
    print(f"\n  {_c('B')}Parámetros recomendados{_c('R')}")
    print(f"    chord_progression_generator --style {r['generator_style']}")
    print(f"    melody_conditioned_v5 --engine {r['melody_engine']}   bass_line_composer --style {r['bass_style']}")
    print(f"    drummer --style {r['drum_style']}" + (f"   ({r['drum_note']})" if r["drum_note"] else ""))
    print(f"    leadsheet_realizer --pattern {r['leadsheet_pattern']}   performer --style {r['performer_style']}"
          f"   song_architect --style {r['song_architect_style']}")
    for w in b["warnings"]:
        print(f"  {_c('YEL')}⚠ {w}{_c('R')}")
    print(f"{'═' * 78}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Progresión (bake) y plan
# ══════════════════════════════════════════════════════════════════════════════

def trim_chords(chord_string, target_beats):
    """Recorta 'G:2 D:1 ...' a ≤ target_beats negras. target_beats puede ser fraccionario."""
    out, acc = [], 0.0
    for tok in chord_string.split():
        name, dur = (tok.rsplit(":", 1) + ["1"])[:2] if ":" in tok else (tok, "1")
        try:
            d = float(dur)
        except ValueError:
            d = 1.0
        if acc + d > target_beats + 1e-6 and out:
            break
        out.append(tok)
        acc += d
    return " ".join(out)


def template_chords(root, mode, bars, bpb):
    base = _NAMES.index(root) if root in _NAMES else {"Db": 1, "Gb": 6, "A#": 10, "D#": 3, "G#": 8}.get(root, 0)
    degs = [(0, "m"), (8, ""), (5, "m"), (7, "")] if mode == "minor" else [(0, ""), (9, "m"), (5, ""), (7, "")]
    dur = int(bpb) if float(bpb).is_integer() else 2
    toks = []
    for i in range(bars):
        off, q = degs[i % 4]
        toks.append(f"{_NAMES[(base + off) % 12]}{q}:{dur}")
    return " ".join(toks)


def bake_chords(brief, bars, seed, suite_dir, method):
    """Devuelve (chord_string, origen). Ejecuta chord_progression_generator si está."""
    bpb = brief["beats_per_bar_quarters"]
    target = bars * bpb
    gen = Path(suite_dir) / "chord_progression_generator.py"
    if method == "generator" and gen.exists():
        r = brief["recommend"]
        with tempfile.TemporaryDirectory() as td:
            cmd = [sys.executable, str(gen), "--key", brief["root"], "--mode", brief["mode"],
                   "--style", r["generator_style"], "--bars", str(bars), "--tempo", str(brief["tempo_bpm"]),
                   "--export-json", "x.json", "--export-midi", "x.mid", "--output", "x", "--seed", str(seed)]
            if float(bpb).is_integer() and int(bpb) != 4 and int(bpb) >= 2:
                cmd += ["--beats", str(int(bpb))]
            try:
                subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=60)
                cs = json.load(open(os.path.join(td, "x.json"), encoding="utf-8")).get("chord_string", "")
                cs = trim_chords(cs, target)
                if cs.strip():
                    return cs, "chord_progression_generator"
            except Exception:
                pass
    return template_chords(brief["root"], brief["mode"], bars, bpb), "plantilla I-vi-IV-V (fallback)"


def _q(s):
    return json.dumps(str(s), ensure_ascii=False)


def _cmd(*parts):
    out = []
    for p in parts:
        p = str(p)
        out.append(f'"{p}"' if re.search(r"\s", p) else p)
    return " ".join(out)


def make_steps(brief, a, chord_string, bars, variant, seed):
    r = brief["recommend"]
    tempo, root, mode = brief["tempo_bpm"], brief["root"], brief["mode"]
    key_h = f"{root}{'min' if mode == 'minor' else 'maj'}"
    S = []

    def step(name, label, cmd, optional=False, note=None, output=None):
        S.append({"name": name, "label": label, "cmd": cmd, "optional": optional, "note": note, "output": output})

    step("harmonic_analysis", "Análisis armónico del original",
         _cmd("python", "harmonic_analyzer.py", "orig.mid", "--no-color", "--json", "harm.json"))
    step("metric_analysis", "Análisis métrico del original",
         _cmd("python", "metric_analyzer.py", "orig.mid", "--json", "metric.json"))
    form_ok = (not a.no_form) and (Path(a.suite_dir) / "form_analyzer.py").exists()
    if form_ok:
        step("form_analysis", "Forma musical del original",
             _cmd("python", "form_analyzer.py", "detect", "orig.mid", "--no-color", "--json", "form.json"))
    if a.profile == "analyze":
        for tool, flag in (("chord_table.py", "chordtable"), ("motif_table.py", "motif"), ("rhythm_table.py", "rhythm")):
            extra = ["--style", r["generator_style"] if r["generator_style"] != "auto" else "diatonic",
                     "--mode", mode, "--key", root] if tool != "rhythm_table.py" else []
            step(flag, f"Catálogo ({tool[:-3]})",
                 _cmd("python", tool, "--analyze-midi", "orig.mid", *extra, "--export-json", f"{flag}.json"))
        step("leitmotifs", "Leitmotivs del original", _cmd("python", "leitmotif_analyst.py", "orig.mid", "--report"))
        step("quality", "Calidad del original", _cmd("python", "quality_scorer.py", "orig.mid", "--out-json", "calidad.json"))
        return S

    gen_style = r["generator_style"]
    cat_style = gen_style if gen_style != "auto" else "diatonic"
    step("chords_generate", "Progresión de referencia (generador)",
         _cmd("python", "chord_progression_generator.py", "--key", root, "--mode", mode, "--style", gen_style,
              "--bars", bars, "--tempo", tempo, "--export-midi", "acordes.mid", "--export-json", "acordes.json",
              "--output", "acordes", "--seed", seed), output="acordes.json",
         note="La cadena de acordes de los pasos siguientes está fijada en el plan (ver brief) para que sea reproducible.")
    for tool, flag in (("chord_table.py", "chordtable"), ("motif_table.py", "motif")):
        step(flag, f"Catálogo ({tool[:-3]})",
             _cmd("python", tool, "--analyze-midi", "orig.mid", "--style", cat_style, "--mode", mode,
                  "--key", root, "--export-midi", f"{flag}.mid", "--export-json", f"{flag}.json"))
    step("rhythm", "Catálogo rítmico (rhythm_table)",
         _cmd("python", "rhythm_table.py", "--analyze-midi", "orig.mid", "--export-midi", "rhythm.mid",
              "--export-json", "rhythm.json"))
    step("leadsheet", f"Acompañamiento ({r['leadsheet_pattern']})",
         _cmd("python", "leadsheet_realizer.py", "--chords", chord_string, "--pattern", r["leadsheet_pattern"],
              "--tempo", tempo, "--output", "armonia.mid"))
    step("melody", f"Melodía condicionada (motor {r['melody_engine']})",
         _cmd("python", "melody_conditioned_v5.py", "--engine", r["melody_engine"], "--key", root, "--mode", mode,
              "--bars", bars, "--tempo", tempo, "--chords", chord_string, "--include-chords",
              "--output", "melodia.mid", "--seed", seed + 100))
    step("bass", f"Línea de bajo ({r['bass_style']})",
         _cmd("python", "bass_line_composer.py", "--key", root, "--mode", mode, "--style", r["bass_style"],
              "--bars", bars, "--tempo", tempo, "--chords", chord_string, "--output", "bajo.mid",
              "--seed", seed + 200))
    step("merge", "Ensamblaje de stems",
         _cmd("python", "midi_merge.py", "armonia.mid", "melodia.mid", "bajo.mid", "--output", "tema_base.mid"))
    if r["drum_style"]:
        step("drums", f"Percusión ({r['drum_style']})",
             _cmd("python", "drummer.py", "add", "tema_base.mid", "--style", r["drum_style"], "--output", "tema.mid"),
             note=r["drum_note"])
        base_mid = "tema.mid"
    else:
        base_mid = "tema_base.mid"
    step("chorale", "Coral SATB (voice_leader)",
         _cmd("python", "voice_leader.py", "--chords", chord_string, "--soprano", "melodia.mid",
              "--key", f"{root} {mode}", "--beats", max(2, int(round(brief["beats_per_bar_quarters"]))),
              "--output", "coral.mid"))
    step("harmony_check", "Auditoría del coral",
         _cmd("python", "harmony_checker.py", "coral.mid", "--key", key_h, "--json", "coral_check.json"),
         optional=True, note="Falla (rc=1) si el coral tiene menos de 2 acordes.")
    step("climax", "Ingeniería del clímax",
         _cmd("python", "climax_engineer.py", base_mid, "--output", "tema_climax.mid"))
    step("perform", f"Interpretación ({r['performer_style']})",
         _cmd("python", "performer.py", "tema_climax.mid", "--style", r["performer_style"], "--output", "final.mid"))
    step("quality", "Calidad de la reconstrucción",
         _cmd("python", "quality_scorer.py", "final.mid", "--out-json", "calidad.json"))
    step("arc", "Supervisión del arco emocional",
         _cmd("python", "arc_supervisor.py", "final.mid", "--tension", "0.2,0.6,0.95,0.3"),
         optional=True, note="Devuelve rc=1 cuando encuentra desviaciones: es un hallazgo, no un fallo.")
    if (Path(a.suite_dir) / "fidelity_meter.py").exists():
        step("fidelity", "Fidelidad frente al original",
             _cmd("python", "fidelity_meter.py", "compare", "orig.mid", "final.mid", "--json", "fidelity.json",
                  "--no-color"), optional=True)
    if form_ok:
        step("form_compare", "Fidelidad estructural (forma)",
             _cmd("python", "form_analyzer.py", "compare", "orig.mid", "final.mid", "--json", "form_compare.json",
                  "--no-color"), optional=True,
             note="Las reconstrucciones cortas (16-32 compases) darán poca fidelidad de forma por diseño.")
    return S


def write_plan_yaml(path, brief, steps, meta):
    L = ["# Plan generado por midi2plan.py — ejecutar con: python runner.py plan.yaml",
         f"name: {_q(meta['name'])}", f"tool: midi2plan", f"version: {_q(VERSION)}",
         f"profile: {meta['profile']}", "source: orig.mid", f"bars_generated: {meta['bars']}",
         f"chords_origin: {_q(meta['chords_origin'])}", f"chord_string: {_q(meta['chord_string'])}",
         f"brief: {json.dumps({k: brief[k] for k in ('key', 'tempo_bpm', 'time_signature', 'meter_class', 'beats_per_bar_quarters', 'n_bars', 'style', 'style_confidence')}, ensure_ascii=False)}",
         "steps:"]
    for s in steps:
        L.append(f"  - name: {s['name']}")
        L.append(f"    label: {_q(s['label'])}")
        L.append(f"    cmd: {_q(s['cmd'])}")
        if s.get("output"):
            L.append(f"    output: {_q(s['output'])}")
        if s["optional"]:
            L.append("    optional: true")
        if s["note"]:
            L.append(f"    note: {_q(s['note'])}")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def slugify(name):
    s = re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem).strip("_")
    return re.sub(r"_+", "_", s)[:60] or "pieza"


def make_plan(midi, out_dir, a):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(midi, out / "orig.mid")
    brief = compute_brief(midi, a.suite_dir, with_form=not a.no_form, force_key=a.key, force_tempo=a.tempo,
                          force_style=a.style, variant=a.variant)
    real = brief["n_bars"]
    if a.bars == "auto":
        bars = max(4, min(real, a.max_bars))
    else:
        bars = max(1, int(a.bars))
    seed = a.seed + a.variant
    cs, origin = bake_chords(brief, bars, seed, a.suite_dir, a.chords_from)
    steps = make_steps(brief, a, cs, bars, a.variant, seed)
    brief["plan"] = {"bars_generated": bars, "chord_string": cs, "chords_origin": origin,
                     "profile": a.profile, "variant": a.variant}
    (out / "brief.json").write_text(json.dumps(brief, ensure_ascii=False, indent=1), encoding="utf-8")
    write_plan_yaml(out / "plan.yaml", brief, steps,
                    {"name": slugify(Path(midi).name), "profile": a.profile, "bars": bars,
                     "chords_origin": origin, "chord_string": cs})
    return brief, out / "plan.yaml", len(steps)


# ══════════════════════════════════════════════════════════════════════════════
#  batch
# ══════════════════════════════════════════════════════════════════════════════

def cmd_batch(a):
    src = Path(a.dir)
    files = sorted(p for p in src.glob(a.glob) if p.is_file())
    if a.limit:
        files = files[:a.limit]
    if not files:
        print(f"[ERROR] sin MIDIs en {src} con el patrón {a.glob}", file=sys.stderr)
        return 1
    out_root = Path(a.out_dir or "midi2plan_lote")
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    runner = Path(a.suite_dir) / "runner.py"
    for i, f in enumerate(files, 1):
        slug = slugify(f.name)
        try:
            brief, plan_path, n = make_plan(f, out_root / slug, a)
            row = {"slug": slug, "key": brief["key"], "tempo": brief["tempo_bpm"], "meter": brief["time_signature"],
                   "bars": brief["n_bars"], "style": brief["style"], "conf": brief["style_confidence"],
                   "form": brief["form"]["form"] if brief["form"] else "—", "steps": n, "run": "—", "error": ""}
            if a.run:
                if not runner.exists():
                    row["run"] = "sin runner.py"
                else:
                    try:
                        p = subprocess.run([sys.executable, str(runner), str(plan_path), "--timeout", str(a.timeout)],
                                           cwd=str(plan_path.parent), capture_output=True, text=True,
                                           timeout=a.timeout * (n + 2))
                        row["run"] = "ok" if (plan_path.parent / "final.mid").exists() else f"rc={p.returncode}"
                    except subprocess.TimeoutExpired:
                        row["run"] = "timeout"
            print(f"  [{i}/{len(files)}] {slug:<40} {row['key']:<10} {row['meter']:<5} {row['bars']:>4}c  "
                  f"{row['style']:<13} {row['form'][:24]:<24} {row['run']}")
        except Exception as e:
            row = {"slug": slug, "key": "—", "tempo": "—", "meter": "—", "bars": "—", "style": "—", "conf": "—",
                   "form": "—", "steps": 0, "run": "—", "error": str(e)}
            print(f"  [{i}/{len(files)}] {slug:<40} ERROR: {e}")
        rows.append(row)
    if a.md:
        L = ["| pieza | tonalidad | tempo | compás | compases | estilo (conf.) | forma | pasos | ejecución |",
             "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['slug']} | {r['key']} | {r['tempo']} | {r['meter']} | {r['bars']} | "
                     f"{r['style']} ({r['conf']}) | {r['form']} | {r['steps']} | {r['run'] or r['error']} |")
        Path(a.md).write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\n  Resumen: {a.md}")
    (out_root / "batch_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  {len(rows)} piezas · planes en {out_root}/<pieza>/plan.yaml")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  selftest
# ══════════════════════════════════════════════════════════════════════════════

def cmd_selftest(a):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'✔' if cond else '✘'} {name} {detail}")

    for (n, d, want) in [(4, 4, 4.0), (3, 4, 3.0), (6, 8, 3.0), (12, 8, 6.0), (3, 8, 1.5), (9, 8, 4.5), (2, 2, 4.0), (5, 4, 5.0)]:
        got = beats_per_bar_quarters(n, d)
        check(f"negras por compás {n}/{d}", abs(got - want) < 1e-9, f"→ {got:g} (esperado {want:g})")
    check("trim_chords con compás de 1.5 negras", trim_chords("A:2 B:2 C:2 D:2", 4.5) == "A:2 B:2", "")
    check("clase de compás 6/8 = compound", meter_class(6, 8) == "compound")
    check("clase de compás 4/4 = simple", meter_class(4, 4) == "simple")
    check("batería en 12/8 → shuffle", recommend("pop", 12, 8, True, 0)["drum_style"] == "shuffle")
    # plan sobre un MIDI sintético 6/8 y validación de YAML
    with tempfile.TemporaryDirectory() as td:
        fa = _load_form_analyzer(a.suite_dir)
        mp = os.path.join(td, "s.mid")
        if fa is not None:
            ev = []
            bt = 480 * 3
            for si, ch in enumerate("ABA"):
                ev += fa._section_events(ch, si * 8, 8, bt)
            fa._write_midi(mp, ev, 480, (6, 8))
        else:
            check("form_analyzer disponible para el selftest", False, "(se omite la parte de plan)")
            return 1
        ns = argparse.Namespace(suite_dir=a.suite_dir, no_form=False, key=None, tempo=None, style=None, variant=0,
                                bars="auto", max_bars=32, seed=100, chords_from="template", profile="reconstruct")
        brief, plan, n = make_plan(mp, os.path.join(td, "o"), ns)
        check("brief 6/8 detecta 24 compases", brief["n_bars"] == 24, f"→ {brief['n_bars']}")
        check("brief 6/8 pulsos de negra = 3", brief["beats_per_bar_quarters"] == 3.0)
        check("forma ABA en el brief", brief["form"] and brief["form"]["form"].replace("'", "") == "A B A",
              f"→ {brief['form'] and brief['form']['form']}")
        try:
            import yaml
            d = yaml.safe_load(Path(plan).read_text(encoding="utf-8"))
            check("plan.yaml es YAML válido", isinstance(d.get("steps"), list) and len(d["steps"]) == n, f"({n} pasos)")
            cs = d["chord_string"]
            tot = sum(float(t.rsplit(":", 1)[1]) for t in cs.split())
            check("la progresión cubre los compases pedidos", abs(tot - d["bars_generated"] * 3.0) < 1e-6,
                  f"→ {tot:g} negras para {d['bars_generated']} compases de 3")
        except ImportError:
            check("pyyaml no instalado: se omite la validación del YAML", True)
    print(f"\n  {'SELFTEST OK' if ok else 'SELFTEST CON FALLOS'}")
    return 0 if ok else 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _plan_args(p):
    p.add_argument("--profile", choices=["reconstruct", "analyze"], default="reconstruct")
    p.add_argument("--bars", default="auto", help="N o 'auto'")
    p.add_argument("--max-bars", type=int, default=32)
    p.add_argument("--variant", type=int, default=0)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--style", default=None, choices=STYLES + ["flamenco", "auto"])
    p.add_argument("--key", default=None)
    p.add_argument("--tempo", type=int, default=None)
    p.add_argument("--chords-from", choices=["generator", "template"], default="generator")
    p.add_argument("--no-form", action="store_true")
    p.add_argument("--suite-dir", default=str(HERE))
    p.add_argument("--no-color", action="store_true")


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="midi2plan.py", description="Del MIDI al plan ejecutable por runner.py")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("brief", help="Extrae el brief de un MIDI")
    p.add_argument("midi")
    p.add_argument("--json", default=None, help="Escribe el brief ('-' = stdout)")
    p.add_argument("--no-form", action="store_true")
    p.add_argument("--suite-dir", default=str(HERE))
    p.add_argument("--variant", type=int, default=0)
    p.add_argument("--no-color", action="store_true")
    p = sp.add_parser("plan", help="Genera un plan .yaml para runner.py")
    p.add_argument("midi")
    p.add_argument("--out-dir", default=None)
    _plan_args(p)
    p = sp.add_parser("batch", help="Planes para una carpeta de MIDIs")
    p.add_argument("dir")
    p.add_argument("--glob", default="*.mid")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--run", action="store_true", help="Ejecuta cada plan con runner.py")
    p.add_argument("--timeout", type=int, default=120, help="Timeout por paso (default 120 s)")
    p.add_argument("--md", default=None, help="Escribe una tabla resumen en Markdown")
    _plan_args(p)
    p = sp.add_parser("selftest", help="Comprobaciones internas")
    p.add_argument("--suite-dir", default=str(HERE))
    p.add_argument("--no-color", action="store_true")
    a = ap.parse_args(argv)
    if getattr(a, "no_color", False):
        _USE_COLOR = False
    try:
        if a.cmd == "selftest":
            return cmd_selftest(a)
        if a.cmd == "brief":
            b = compute_brief(a.midi, a.suite_dir, with_form=not a.no_form, variant=a.variant)
            if a.json == "-":
                print(json.dumps(b, ensure_ascii=False, indent=1))
                return 0
            print_brief(b)
            if a.json:
                Path(a.json).write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  JSON: {a.json}")
            return 0
        if a.cmd == "plan":
            out = a.out_dir or f"plan_{slugify(Path(a.midi).name)}"
            b, plan, n = make_plan(a.midi, out, a)
            print_brief(b)
            print(f"  Plan: {plan}  ({n} pasos, {b['plan']['bars_generated']} compases, progresión: "
                  f"{b['plan']['chords_origin']})")
            print(f"  Ejecutar: python runner.py {plan}\n")
            return 0
        if a.cmd == "batch":
            return cmd_batch(a)
    except (ValueError, OSError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
