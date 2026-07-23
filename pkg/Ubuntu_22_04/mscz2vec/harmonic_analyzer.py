#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     HARMONIC ANALYZER  v1.1                                 ║
║        Análisis armónico funcional (números romanos) con explicación        ║
║                                                                              ║
║  Es el compañero armónico de metric_analyzer (que cubre el eje rítmico):    ║
║  toma TU MIDI y devuelve, compás a compás, el acorde detectado, su cifrado, ║
║  su número romano, su función (T/S/D) y las cadencias, con una explicación  ║
║  en lenguaje llano de POR QUÉ. La misma información existe enterrada en el   ║
║  HTML de piano_duo_analyzer; aquí es una herramienta didáctica standalone.  ║
║                                                                              ║
║  A diferencia de reharmonizer o voice_leader (que TRANSFORMAN la armonía),  ║
║  esta herramienta solo ANALIZA y EXPLICA lo que ya escribiste.              ║
║                                                                              ║
║  NOVEDAD v1.1 — DOMINANTES SECUNDARIOS:                                     ║
║  Detecta tríadas mayores / 7ª de dominante "sorprendentes" (que no          ║
║  encajarían diatónicamente en ese grado) y comprueba si el acorde           ║
║  siguiente resuelve una 5ª justa por debajo. Si resuelve, se etiqueta       ║
║  como V/x o V7/x (dominante secundario / tonicización de x) con función    ║
║  D, en vez de "diatonizarlo" a un numeral engañoso. Si no resuelve, se      ║
║  mantiene el etiquetado cromático anterior (mediante, préstamo modal…).    ║
║                                                                              ║
║  PIPELINE:                                                                   ║
║  [1] SEGMENTACIÓN — agrupa las notas por compás (o por --window beats)      ║
║  [2] ACORDE       — detecta acorde y bajo por plantilla de intervalos        ║
║  [3] TONALIDAD    — estima la tonalidad global (Krumhansl-Schmuckler)       ║
║  [4] ROMANOS      — numeral + función + inversión respecto a la tonalidad   ║
║  [5] SECUNDARIOS  — confirma dominantes secundarios por resolución  [NUEVO] ║
║  [6] CADENCIAS    — auténtica / plagal / rota / semicadencia                ║
║  [7] INFORME      — tabla por compás + explicación + sidecar JSON opcional  ║
║                                                                              ║
║  USO:                                                                        ║
║    python harmonic_analyzer.py obra.mid                                      ║
║    python harmonic_analyzer.py obra.mid --key Cmaj                          ║
║    python harmonic_analyzer.py obra.mid --window 2                          ║
║    python harmonic_analyzer.py obra.mid --explain --json obra.harm.json     ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi              MIDI de entrada                                         ║
║    --key KEY         Fuerza la tonalidad (p.ej. Cmaj, Am, F#min, Bbmaj)     ║
║    --window BEATS    Ventana de análisis en tiempos (default: 1 compás)     ║
║    --track N         Analiza solo la pista N (default: todas)               ║
║    --explain         Añade una explicación por compás                        ║
║    --json FILE       Escribe un sidecar JSON con el análisis completo        ║
║    --max-bars N      Limita la tabla impresa (default: 48)                   ║
║    --no-color        Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from harmonic_analyzer import analyze_harmony                             ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np

try:
    from playability_auditor import read_midi, TimeContext, extract_notes
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.1"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def pc_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


# ══════════════════════════════════════════════════════════════════════════════
#  PLANTILLAS DE ACORDES Y TONALIDAD
# ══════════════════════════════════════════════════════════════════════════════

CHORD_TEMPLATES = {
    "": {0, 4, 7}, "m": {0, 3, 7}, "dim": {0, 3, 6}, "aug": {0, 4, 8},
    "maj7": {0, 4, 7, 11}, "m7": {0, 3, 7, 10}, "7": {0, 4, 7, 10},
    "m7b5": {0, 3, 6, 10}, "dim7": {0, 3, 6, 9}, "6": {0, 4, 7, 9},
    "m6": {0, 3, 7, 9}, "sus4": {0, 5, 7}, "sus2": {0, 2, 7},
}
_QUAL_TRIAD = {"": "maj", "6": "maj", "maj7": "maj", "7": "maj",
               "m": "min", "m7": "min", "m6": "min", "aug": "aug",
               "dim": "dim", "dim7": "dim", "m7b5": "dim",
               "sus4": "sus", "sus2": "sus"}

# Perfiles Krumhansl-Schmuckler
_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52,
                    5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54,
                    4.75, 3.98, 2.69, 3.34, 3.17])

# Grados de la escala mayor y menor natural para numerales base
_MAJ_DEGREES = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii"}
_MIN_DEGREES = {0: "i", 2: "ii", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}


def detect_chord(pcs_weight: Dict[int, float], bass_pc: Optional[int]
                 ) -> Tuple[Optional[int], str, float]:
    """Devuelve (root_pc, quality_suffix, score) por mejor ajuste de plantilla."""
    present = {pc for pc, w in pcs_weight.items() if w > 0}
    if not present:
        return None, "", 0.0
    best = (None, "", -1e9)
    for root in range(12):
        rel = {(pc - root) % 12 for pc in present}
        for suffix, templ in CHORD_TEMPLATES.items():
            inter = len(rel & templ)
            extra = len(rel - templ)
            missing = len(templ - rel)
            score = 2.0 * inter - 1.0 * extra - 0.7 * missing
            if bass_pc is not None and (bass_pc - root) % 12 == 0:
                score += 0.6                      # bonus fundamental en el bajo
            if score > best[2]:
                best = (root, suffix, score)
    return best


def detect_key(pc_hist: np.ndarray) -> Tuple[int, str]:
    """Krumhansl-Schmuckler → (tonic_pc, 'maj'|'min')."""
    h = pc_hist - pc_hist.mean() if pc_hist.sum() else pc_hist
    best = (0, "maj", -1e9)
    for tonic in range(12):
        for mode, prof in (("maj", _KS_MAJ), ("min", _KS_MIN)):
            p = np.roll(prof - prof.mean(), tonic)
            denom = np.linalg.norm(h) * np.linalg.norm(p)
            corr = float(np.dot(h, p) / denom) if denom > 1e-9 else 0.0
            if corr > best[2]:
                best = (tonic, mode, corr)
    return best[0], best[1]


def roman_of(root_pc: int, suffix: str, tonic_pc: int, mode: str) -> Tuple[str, str]:
    """Devuelve (numeral, funcion) del acorde respecto a la tonalidad."""
    deg = (root_pc - tonic_pc) % 12
    table = _MAJ_DEGREES if mode == "maj" else _MIN_DEGREES
    base = table.get(deg)
    triad = _QUAL_TRIAD.get(suffix, "maj")
    if base is None:
        # acorde cromático: numeral con alteración sobre el grado más cercano
        approx = {1: "bII", 3: "bIII", 6: "#IV", 8: "bVI", 10: "bVII"}.get(deg, "?")
        num = approx if triad in ("maj", "aug", "sus") else approx.lower()
    else:
        upper = base if base[0].isupper() else base
        # ajusta mayúscula/minúscula a la calidad detectada
        num = upper.upper() if triad in ("maj", "aug", "sus") else upper.lower()
        if triad == "dim":
            num = upper.lower() + "°"
    # séptimas / extensiones visibles
    if suffix in ("7", "maj7", "m7", "m7b5", "dim7"):
        num += "7" if suffix != "maj7" else "maj7"
    # función armónica
    if deg in (0, 9) or (mode == "min" and deg in (0, 8)):
        func = "T"
    elif deg in (7, 11) or (mode == "min" and deg in (7, 10)):
        func = "D"
    elif deg in (2, 5):
        func = "S"
    else:
        func = "–"
    return num, func


# Calidad de tríada diatónica NATURAL esperada en cada grado (para detectar
# dominantes secundarios: una tríada mayor o una 7ª de dominante sobre un
# grado que naturalmente sería menor/disminuido es "sorprendente" y candidata
# a tonicizar el grado que queda una 5ª justa por debajo).
_MAJ_DIATONIC_QUALITY = {0: "maj", 2: "min", 4: "min", 5: "maj",
                         7: "maj", 9: "min", 11: "dim"}
_MIN_DIATONIC_QUALITY = {0: "min", 2: "dim", 3: "maj", 5: "min",
                         7: "min", 8: "maj", 10: "maj"}


def secondary_dominant_candidate(root_pc: int, suffix: str,
                                 tonic_pc: int, mode: str) -> Optional[int]:
    """
    Evalúa si (root_pc, suffix) tiene pinta de dominante secundario:
    tríada mayor simple o 7ª de dominante sobre un grado que diatónicamente
    NO sería mayor (o incluso fuera de la escala).
    Si es candidato, devuelve el pitch-class del grado que tonicizaría
    (una 5ª justa por debajo de root_pc). Si no, devuelve None.
    No confirma resolución real: eso se comprueba después contra el
    acorde siguiente en analyze_harmony().
    """
    if suffix not in ("", "7"):
        return None
    deg = (root_pc - tonic_pc) % 12
    if deg == 7:
        return None  # es la propia V de la tonalidad, no un secundario
    table = _MAJ_DEGREES if mode == "maj" else _MIN_DEGREES
    qual_table = _MAJ_DIATONIC_QUALITY if mode == "maj" else _MIN_DIATONIC_QUALITY
    expected = qual_table.get(deg)
    is_plain_diatonic_major = suffix == "" and deg in table and expected == "maj"
    if is_plain_diatonic_major:
        return None  # tríada mayor diatónica normal (I/IV en mayor, III/VI/VII en menor)
    target_pc = (root_pc + 5) % 12          # resolución: baja una 5ª justa
    target_deg = (target_pc - tonic_pc) % 12
    if target_deg in table:
        return target_pc
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def _parse_key(s: str) -> Tuple[int, str]:
    s = s.strip()
    letter = s[0].upper()
    i = 1
    pc = _PC[letter]
    while i < len(s) and s[i] in "#b♯♭sf":
        if s[i] in "#♯s":
            pc += 1
        else:
            pc -= 1
        i += 1
    rest = s[i:].lower()
    mode = "min" if rest in ("m", "min", "minor", "menor") else "maj"
    return pc % 12, mode


def analyze_harmony(path: str, key: Optional[str] = None,
                    window_beats: Optional[float] = None,
                    track: Optional[int] = None) -> dict:
    """API pública. Devuelve el análisis armónico completo."""
    mid = read_midi(path)
    tc = TimeContext(mid)
    src = [mid.tracks[track]] if track is not None else mid.tracks
    if track is not None and not (0 <= track < len(mid.tracks)):
        raise ValueError(f"pista {track} fuera de rango")

    notes = [n for trk in src for n in extract_notes(trk)]
    if not notes:
        raise ValueError("el MIDI (o la pista) no contiene notas")

    last = max(n.end for n in notes)
    n_bars = max(1, tc.bar(last - 1))

    # histograma global de pitch-class ponderado por duración
    pc_hist = np.zeros(12)
    for n in notes:
        pc_hist[n.pitch % 12] += (n.end - n.start)
    if key:
        tonic_pc, mode = _parse_key(key)
    else:
        tonic_pc, mode = detect_key(pc_hist)

    # segmentación por ventana
    tpb = tc.tpb
    segments = []
    if window_beats:
        step = int(round(window_beats * tpb))
        t = 0
        while t < last:
            segments.append((t, t + step))
            t += step
    else:
        for bar in range(1, n_bars + 1):
            t0, t1 = tc.bar_range_ticks(bar, bar)
            segments.append((t0, t1))

    chords = []
    for si, (t0, t1) in enumerate(segments):
        seg_notes = [n for n in notes if n.start < t1 and n.end > t0]
        if not seg_notes:
            chords.append({"bar": tc.bar(t0), "chord": None})
            continue
        pcs_weight: Dict[int, float] = {}
        for n in seg_notes:
            dur = min(n.end, t1) - max(n.start, t0)
            pcs_weight[n.pitch % 12] = pcs_weight.get(n.pitch % 12, 0) + dur
        bass_pc = min(seg_notes, key=lambda n: n.pitch).pitch % 12
        root, suffix, score = detect_chord(pcs_weight, bass_pc)
        if root is None:
            chords.append({"bar": tc.bar(t0), "chord": None})
            continue
        num, func = roman_of(root, suffix, tonic_pc, mode)
        inv = ""
        if bass_pc != root:
            rel = (bass_pc - root) % 12
            inv = {4: "6", 3: "6", 7: "6/4", 10: "7", 11: "7"}.get(rel, "/" + pc_name(bass_pc))
        chords.append({
            "bar": tc.bar(t0),
            "chord": pc_name(root) + suffix,
            "root_pc": root, "suffix": suffix,
            "roman": num, "function": func,
            "bass": pc_name(bass_pc), "inversion": inv,
            "score": round(float(score), 2),
        })

    # ── SEGUNDA PASADA: confirmar dominantes secundarios por resolución ──────
    # Un acorde candidato (tríada mayor / 7ª de dominante "sorprendente") solo
    # se reetiqueta como V/x si el SIGUIENTE acorde real resuelve una 5ª justa
    # por debajo, tal como se espera de una tonicización de verdad. Si no
    # resuelve, se conserva el numeral cromático que ya calculó roman_of().
    seq_all = [c for c in chords if c["chord"]]
    table = _MAJ_DEGREES if mode == "maj" else _MIN_DEGREES
    for pos, c in enumerate(seq_all):
        cand_target = secondary_dominant_candidate(c["root_pc"], c["suffix"],
                                                    tonic_pc, mode)
        if cand_target is None or pos + 1 >= len(seq_all):
            continue
        nxt = seq_all[pos + 1]
        if nxt["root_pc"] != cand_target:
            continue
        target_deg = (cand_target - tonic_pc) % 12
        target_base = table.get(target_deg, "?")
        c["roman"] = f"V7/{target_base}" if c["suffix"] == "7" else f"V/{target_base}"
        c["function"] = "D"
        c["secondary_dominant_of"] = target_base

    secondary_dominants = [
        {"bar": c["bar"], "chord": c["chord"], "roman": c["roman"],
         "of": c["secondary_dominant_of"]}
        for c in seq_all if c.get("secondary_dominant_of")
    ]

    # cadencias: pares consecutivos con acorde
    seq = [c for c in chords if c["chord"]]
    cadences = []
    for a, b in zip(seq[:-1], seq[1:]):
        fa, fb = a["function"], b["function"]
        label = None
        if fa == "D" and fb == "T" and b["roman"].upper().startswith("I"):
            label = "auténtica (V→I)"
        elif fa == "S" and fb == "T":
            label = "plagal (IV→I)"
        elif fa == "D" and b["roman"].lower().startswith("vi"):
            label = "rota (V→vi)"
        elif fb == "D":
            label = "semicadencia (→V)"
        if label:
            cadences.append({"bar": b["bar"], "type": label,
                             "from": a["chord"], "to": b["chord"]})

    return {
        "file": path,
        "key": f"{pc_name(tonic_pc)}{'m' if mode == 'min' else 'maj'}",
        "key_source": "manual" if key else "detectada",
        "n_bars": int(n_bars),
        "chords": chords,
        "cadences": cadences,
        "secondary_dominants": secondary_dominants,
    }


def _explain(c: dict, keyname: str) -> str:
    if not c["chord"]:
        return "silencio / textura no acórdica"
    if c.get("secondary_dominant_of"):
        return (f"{c['chord']} = {c['roman']} en {keyname}: dominante secundario, "
                f"toniciza {c['secondary_dominant_of']} (tensión que resuelve "
                f"una 5ª justa por debajo, hacia {c['secondary_dominant_of']})")
    func = {"T": "tónica (reposo)", "S": "subdominante (alejamiento)",
            "D": "dominante (tensión que pide resolver)", "–": "función ambigua"}[c["function"]]
    inv = ""
    if c["inversion"]:
        inv = f", en inversión ({c['inversion']}, bajo {c['bass']})"
    return f"{c['chord']} = {c['roman']} en {keyname}: {func}{inv}"


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="harmonic_analyzer.py",
        description="Análisis armónico funcional con números romanos.")
    ap.add_argument("midi")
    ap.add_argument("--key")
    ap.add_argument("--window", type=float)
    ap.add_argument("--track", type=int)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--max-bars", type=int, default=48)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    print(f"  {B}HARMONIC ANALYZER v{VERSION}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")
    try:
        st = analyze_harmony(args.midi, key=args.key, window_beats=args.window,
                             track=args.track)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Tonalidad: {C}{st['key']}{R} ({st['key_source']})   "
          f"compases: {st['n_bars']}   acordes: {len([c for c in st['chords'] if c['chord']])}")
    print(f"\n  {G}compás  acorde       romano     func   bajo{R}")
    shown = [c for c in st["chords"] if c["chord"]][:args.max_bars]
    for c in shown:
        fcol = {"T": _c("GRN"), "D": _c("RED"), "S": _c("YEL"), "–": ""}[c["function"]]
        inv = f" ({c['inversion']})" if c["inversion"] else ""
        print(f"  {c['bar']:>5}   {c['chord']:<10}  {c['roman']:<9}  "
              f"{fcol}{c['function']:^4}{R}   {c['bass']}{inv}")
        if args.explain:
            print(f"          {G}{_explain(c, st['key'])}{R}")

    if st["secondary_dominants"]:
        print(f"\n  {B}Dominantes secundarios (tonicizaciones confirmadas):{R}")
        for sd in st["secondary_dominants"]:
            print(f"    c.{sd['bar']}: {sd['chord']} = {_c('CYA')}{sd['roman']}{R}  "
                  f"→ toniciza {sd['of']}")

    if st["cadences"]:
        print(f"\n  {B}Cadencias:{R}")
        for cad in st["cadences"]:
            print(f"    c.{cad['bar']}: {cad['type']}  ({cad['from']} → {cad['to']})")
    else:
        print(f"\n  (no se detectaron cadencias claras)")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"version": VERSION, **st}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n  {C}JSON: {args.json}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
