#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       PRACTICE TUTOR  v1.0                                  ║
║        Graduador de dificultad y generador de plan de estudio (piano)       ║
║                                                                              ║
║  playability_auditor dice SI una parte es tocable y fingering la digita,    ║
║  pero nadie GRADÚA la dificultad ni PLANIFICA la práctica. Esta herramienta ║
║  estima el nivel de una pieza de piano, localiza los compases difíciles y   ║
║  genera material de estudio: manos separadas, versión lenta (usando         ║
║  tempo_designer) y bucles de los compases más duros. Es la pieza que más    ║
║  echa en falta quien está aprendiendo a tocar.                              ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    grade   — estima nivel (≈1–8), factores de dificultad y compases duros   ║
║    plan    — genera material de práctica (manos separadas, lento, bucles)   ║
║                                                                              ║
║  FACTORES DE DIFICULTAD:                                                     ║
║    velocidad     — notas por segundo exigidas (tempo × densidad)            ║
║    extensión     — acordes/intervalos amplios dentro de una mano            ║
║    polifonía     — nº de notas simultáneas por mano                         ║
║    saltos        — saltos melódicos grandes dentro de una mano              ║
║    independencia — divergencia rítmica entre las dos manos                  ║
║                                                                              ║
║  USO:                                                                        ║
║    python practice_tutor.py grade pieza.mid                                 ║
║    python practice_tutor.py grade pieza.mid --split 60                      ║
║    python practice_tutor.py plan pieza.mid --slow-factor 0.6 --hard 3       ║
║    python practice_tutor.py plan pieza.mid --outdir estudio/                ║
║                                                                              ║
║  OPCIONES (grade):                                                           ║
║    midi              MIDI de piano de entrada                               ║
║    --split N         Nota MIDI de corte MD/MI si hay 1 sola pista (def: 60) ║
║    --json FILE       Escribe un sidecar JSON con el diagnóstico              ║
║  OPCIONES (plan):                                                            ║
║    midi              MIDI de piano de entrada                               ║
║    --outdir DIR      Carpeta de salida (default: junto al MIDI)             ║
║    --slow-factor F   Fracción del tempo original para la versión lenta      ║
║                      (default: 0.6)                                          ║
║    --hard N          Nº de compases difíciles a extraer en bucle (def: 3)   ║
║    --loops N         Repeticiones por compás difícil (default: 4)           ║
║    --split N         Corte MD/MI si hay 1 sola pista (default: 60)          ║
║    --no-color        Desactivar colores ANSI                               ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from practice_tutor import grade_piece, make_plan                         ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py · tempo_designer.py            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

try:
    from playability_auditor import (read_midi, write_midi, MidiData,
                                     MidiTrackData, MidiEvent, TimeContext,
                                     extract_notes, Note, _write_vlq)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

try:
    from tempo_designer import design_tempo
    _HAS_TEMPO = True
except ImportError:
    _HAS_TEMPO = False

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_SPARK = "▁▂▃▄▅▆▇█"

_LEVELS = ["principiante", "elemental", "elemental-medio", "intermedio",
           "intermedio-alto", "avanzado", "avanzado-alto", "virtuoso"]


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def sparkline(v) -> str:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return ""
    if v.max() <= v.min():
        return _SPARK[0] * len(v)
    x = (v - v.min()) / (v.max() - v.min())
    return "".join(_SPARK[min(7, int(round(t * 7)))] for t in x)


# ══════════════════════════════════════════════════════════════════════════════
#  SEPARACIÓN DE MANOS
# ══════════════════════════════════════════════════════════════════════════════

def separate_hands(mid, split: int = 60) -> Tuple[List[Note], List[Note]]:
    """Devuelve (notas_MD, notas_MI). Usa pistas si hay ≥2, si no divide por altura."""
    note_tracks = [(i, extract_notes(t)) for i, t in enumerate(mid.tracks)]
    note_tracks = [(i, ns) for i, ns in note_tracks if ns]
    if len(note_tracks) >= 2:
        # la pista de mayor altura media = MD; el resto = MI
        avg = [(i, sum(n.pitch for n in ns) / len(ns), ns) for i, ns in note_tracks]
        avg.sort(key=lambda x: -x[1])
        rh = list(avg[0][2])
        lh = [n for _, _, ns in avg[1:] for n in ns]
        return rh, lh
    all_notes = [n for _, ns in note_tracks for n in ns]
    rh = [n for n in all_notes if n.pitch >= split]
    lh = [n for n in all_notes if n.pitch < split]
    return rh, lh


def _chords_at_onsets(notes: List[Note]) -> List[List[Note]]:
    by_start: Dict[int, List[Note]] = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)
    return [by_start[s] for s in sorted(by_start)]


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS DE DIFICULTAD
# ══════════════════════════════════════════════════════════════════════════════

def _hand_metrics(notes: List[Note], tc: TimeContext) -> dict:
    if not notes:
        return {"peak_nps": 0.0, "wide_ratio": 0.0, "max_span": 0,
                "max_poly": 0, "big_leaps": 0}
    # nps pico en ventana de 2 s
    starts = sorted(tc.sec(n.start) for n in notes)
    peak = 0
    j = 0
    for i in range(len(starts)):
        while starts[i] - starts[j] > 2.0:
            j += 1
        peak = max(peak, i - j + 1)
    peak_nps = peak / 2.0
    # extensión y polifonía por acorde
    chords = _chords_at_onsets(notes)
    spans = [max(c, key=lambda n: n.pitch).pitch - min(c, key=lambda n: n.pitch).pitch
             for c in chords if len(c) > 1]
    max_span = max(spans) if spans else 0
    wide_ratio = (sum(1 for s in spans if s > 9) / len(chords)) if chords else 0.0
    max_poly = max(len(c) for c in chords) if chords else 0
    # saltos melódicos (voz superior)
    top = [max(c, key=lambda n: n.pitch).pitch for c in chords]
    leaps = sum(1 for a, b in zip(top[:-1], top[1:]) if abs(b - a) > 9)
    return {"peak_nps": round(peak_nps, 2), "wide_ratio": round(wide_ratio, 3),
            "max_span": int(max_span), "max_poly": int(max_poly),
            "big_leaps": int(leaps)}


def _independence(rh: List[Note], lh: List[Note], tc: TimeContext) -> float:
    """0 = manos rítmicamente idénticas (fácil), 1 = totalmente divergentes."""
    if not rh or not lh:
        return 0.0
    grid = max(1, tc.tpb // 4)
    sr = {n.start // grid for n in rh}
    sl = {n.start // grid for n in lh}
    if not sr or not sl:
        return 0.0
    inter = len(sr & sl)
    union = len(sr | sl)
    return round(1.0 - inter / union, 3) if union else 0.0


def grade_piece(path: str, split: int = 60) -> dict:
    """API pública. Estima nivel, factores y compases difíciles."""
    mid = read_midi(path)
    tc = TimeContext(mid)
    rh, lh = separate_hands(mid, split)
    all_notes = rh + lh
    if not all_notes:
        raise ValueError("el MIDI no contiene notas")

    last = max(n.end for n in all_notes)
    n_bars = max(1, tc.bar(last - 1))
    dur_s = max(0.5, tc.sec(last))

    m_rh = _hand_metrics(rh, tc)
    m_lh = _hand_metrics(lh, tc)
    indep = _independence(rh, lh, tc)

    # factores 0..1
    speed = np.clip(max(m_rh["peak_nps"], m_lh["peak_nps"]) / 12.0, 0, 1)
    extension = np.clip(max(m_rh["max_span"], m_lh["max_span"]) / 14.0, 0, 1)
    poly = np.clip((max(m_rh["max_poly"], m_lh["max_poly"]) - 1) / 4.0, 0, 1)
    leaps = np.clip((m_rh["big_leaps"] + m_lh["big_leaps"]) / max(1, n_bars) / 2.0, 0, 1)
    independence = np.clip(indep / 0.7, 0, 1)

    factors = {"velocidad": round(float(speed), 3),
               "extensión": round(float(extension), 3),
               "polifonía": round(float(poly), 3),
               "saltos": round(float(leaps), 3),
               "independencia": round(float(independence), 3)}
    weights = {"velocidad": 0.30, "extensión": 0.20, "polifonía": 0.20,
               "saltos": 0.15, "independencia": 0.15}
    diff = sum(factors[k] * weights[k] for k in factors)
    grade_idx = int(np.clip(round(diff * 7), 0, 7))

    # dificultad por compás
    per_bar = np.zeros(n_bars)
    for n in all_notes:
        b = min(tc.bar(n.start) - 1, n_bars - 1)
        per_bar[b] += 1
    # normaliza densidad por compás y añade spans locales
    bar_span = np.zeros(n_bars)
    for hand in (rh, lh):
        for c in _chords_at_onsets(hand):
            if len(c) > 1:
                b = min(tc.bar(c[0].start) - 1, n_bars - 1)
                hi = max(c, key=lambda n: n.pitch).pitch
                lo = min(c, key=lambda n: n.pitch).pitch
                bar_span[b] = max(bar_span[b], hi - lo)
    dens_n = per_bar / (per_bar.max() or 1)
    span_n = bar_span / 14.0
    bar_diff = 0.6 * dens_n + 0.4 * np.clip(span_n, 0, 1)
    hardest = sorted(range(n_bars), key=lambda b: -bar_diff[b])

    return {
        "file": path, "n_bars": int(n_bars), "duration_s": round(dur_s, 1),
        "difficulty": round(float(diff), 3),
        "grade": grade_idx + 1, "level": _LEVELS[grade_idx],
        "factors": factors,
        "rh_metrics": m_rh, "lh_metrics": m_lh,
        "bar_difficulty": [round(float(x), 3) for x in bar_diff],
        "hardest_bars": [b + 1 for b in hardest[:8]],
        "rh_notes": len(rh), "lh_notes": len(lh),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE MATERIAL DE ESTUDIO
# ══════════════════════════════════════════════════════════════════════════════

def _meta_name(name):
    b = name.encode("latin-1", "replace")
    return MidiEvent(abs=0, kind="meta", meta_type=0x03,
                     data=bytes([0xFF, 0x03]) + _write_vlq(len(b)) + b)


def _clone_conductor(mid) -> MidiTrackData:
    import math
    trk = MidiTrackData(name="conductor")
    trk.events.append(_meta_name("conductor"))
    for tick, num, den in mid.timesig_map:
        dd = int(round(math.log2(max(1, den))))
        trk.events.append(MidiEvent(abs=tick, kind="meta", meta_type=0x58,
                                    data=bytes([0xFF, 0x58]) + _write_vlq(4)
                                    + bytes([num, dd, 24, 8])))
    for tick, us in mid.tempo_map:
        trk.events.append(MidiEvent(abs=tick, kind="meta", meta_type=0x51,
                                    data=bytes([0xFF, 0x51]) + _write_vlq(3)
                                    + us.to_bytes(3, "big")))
    trk.events.sort(key=lambda e: e.abs)
    return trk


def _notes_track(name: str, notes: List[Note], program: int = 0) -> MidiTrackData:
    trk = MidiTrackData(name=name)
    trk.events.append(_meta_name(name))
    trk.events.append(MidiEvent(abs=0, kind="channel", channel=0,
                                data=bytes([0xC0, program])))
    for n in notes:
        trk.events.append(MidiEvent(abs=n.start, kind="note_on", channel=0,
                                    pitch=n.pitch, vel=n.vel))
        trk.events.append(MidiEvent(abs=n.end, kind="note_off", channel=0,
                                    pitch=n.pitch, vel=0))
    return trk


def _write_hand(mid, notes, path):
    out = MidiData(fmt=1, tpb=mid.tpb, tempo_map=list(mid.tempo_map),
                   timesig_map=list(mid.timesig_map))
    out.tracks.append(_clone_conductor(mid))
    out.tracks.append(_notes_track("mano", notes))
    write_midi(out, path)


def _write_bar_loop(mid, tc, notes, bar, loops, path):
    t0, t1 = tc.bar_range_ticks(bar, bar)
    seg = [n for n in notes if n.start < t1 and n.end > t0]
    if not seg:
        return False
    length = t1 - t0
    looped = []
    for k in range(loops):
        for n in seg:
            s = max(0, n.start - t0) + k * length
            e = min(n.end, t1) - t0 + k * length
            if e > s:
                looped.append(Note(n.pitch, n.vel, s, e, n.channel))
    out = MidiData(fmt=1, tpb=mid.tpb, tempo_map=list(mid.tempo_map),
                   timesig_map=list(mid.timesig_map))
    out.tracks.append(_clone_conductor(mid))
    out.tracks.append(_notes_track(f"compas_{bar}_x{loops}", looped))
    write_midi(out, path)
    return True


def make_plan(path: str, outdir: Optional[str] = None, slow_factor: float = 0.6,
              hard: int = 3, loops: int = 4, split: int = 60) -> dict:
    """API pública. Genera el material de estudio. Devuelve el listado de ficheros."""
    mid = read_midi(path)
    tc = TimeContext(mid)
    rh, lh = separate_hands(mid, split)
    if not (rh or lh):
        raise ValueError("el MIDI no contiene notas")

    stem = Path(path).stem
    base = Path(outdir) if outdir else Path(path).parent
    base.mkdir(parents=True, exist_ok=True)
    outputs = []

    # manos separadas
    if rh:
        p = str(base / f"{stem}_MD.mid"); _write_hand(mid, rh, p); outputs.append(p)
    if lh:
        p = str(base / f"{stem}_MI.mid"); _write_hand(mid, lh, p); outputs.append(p)

    # versión lenta (vía tempo_designer)
    orig_bpm = 60_000_000 / mid.tempo_map[0][1]
    slow_bpm = max(20.0, orig_bpm * slow_factor)
    slow_path = str(base / f"{stem}_lento.mid")
    if _HAS_TEMPO:
        design_tempo(path, out_path=slow_path, points=[(1, slow_bpm)],
                     shape="step", dry_run=False, write_json=False)
    else:                                            # fallback: reescala tempo_map
        out = read_midi(path)
        us = int(round(60_000_000 / slow_bpm))
        for trk in out.tracks:
            trk.events = [e for e in trk.events
                          if not (e.kind == "meta" and e.meta_type == 0x51)]
        out.tracks[0].events.insert(0, MidiEvent(abs=0, kind="meta", meta_type=0x51,
                                    data=bytes([0xFF, 0x51]) + _write_vlq(3)
                                    + us.to_bytes(3, "big")))
        write_midi(out, slow_path)
    outputs.append(slow_path)

    # bucles de compases difíciles
    g = grade_piece(path, split)
    all_notes = rh + lh
    loop_files = []
    for bar in g["hardest_bars"][:hard]:
        p = str(base / f"{stem}_dificil_c{bar:02d}_x{loops}.mid")
        if _write_bar_loop(mid, tc, all_notes, bar, loops, p):
            loop_files.append(p); outputs.append(p)

    return {"file": path, "grade": g["grade"], "level": g["level"],
            "orig_bpm": round(orig_bpm, 1), "slow_bpm": round(slow_bpm, 1),
            "hard_bars": g["hardest_bars"][:hard],
            "hands_separated": [f for f in outputs if f.endswith(("_MD.mid", "_MI.mid"))],
            "slow": slow_path, "loops": loop_files, "outputs": outputs}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _print_grade(st):
    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    gcol = (_c("GRN") if st["grade"] <= 3 else _c("YEL") if st["grade"] <= 5
            else _c("RED"))
    print(f"  Compases: {st['n_bars']}   duración: {st['duration_s']}s   "
          f"MD: {st['rh_notes']} notas · MI: {st['lh_notes']} notas")
    print(f"  Nivel estimado: {gcol}grado {st['grade']}/8 — {st['level']}{R}  "
          f"(índice {st['difficulty']:.2f})")
    print(f"\n  {G}factor          valor{R}")
    for k, v in st["factors"].items():
        bar = _SPARK[min(7, int(v * 7))] * max(1, int(v * 20))
        print(f"  {k:<14} {v:.2f}  {bar}")
    print(f"\n  dificultad/compás: {sparkline(st['bar_difficulty'])}")
    print(f"  compases más difíciles: "
          f"{', '.join('c.' + str(b) for b in st['hardest_bars'][:5])}")


def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="practice_tutor.py",
        description="Graduador de dificultad y plan de estudio para piano.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("grade", help="estima nivel y compases difíciles")
    pg.add_argument("midi")
    pg.add_argument("--split", type=int, default=60)
    pg.add_argument("--json")
    pg.add_argument("--no-color", action="store_true")

    pp = sub.add_parser("plan", help="genera material de práctica")
    pp.add_argument("midi")
    pp.add_argument("--outdir")
    pp.add_argument("--slow-factor", type=float, default=0.6)
    pp.add_argument("--hard", type=int, default=3)
    pp.add_argument("--loops", type=int, default=4)
    pp.add_argument("--split", type=int, default=60)
    pp.add_argument("--no-color", action="store_true")

    args = ap.parse_args()
    if getattr(args, "no_color", False):
        _USE_COLOR = False
    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")

    print(f"\n{'═' * 78}")
    print(f"  {B}PRACTICE TUTOR v{VERSION}  ·  {args.cmd}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")

    try:
        if args.cmd == "grade":
            st = grade_piece(args.midi, split=args.split)
            _print_grade(st)
            if args.json:
                Path(args.json).write_text(
                    json.dumps({"version": VERSION, **st}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                print(f"\n  {C}JSON: {args.json}{R}")
        else:
            st = make_plan(args.midi, outdir=args.outdir,
                           slow_factor=args.slow_factor, hard=args.hard,
                           loops=args.loops, split=args.split)
            print(f"  Nivel: grado {st['grade']}/8 ({st['level']})   "
                  f"tempo {st['orig_bpm']:g} → {st['slow_bpm']:g} bpm (lento)")
            print(f"  compases difíciles en bucle: "
                  f"{', '.join('c.' + str(b) for b in st['hard_bars'])}")
            print(f"\n  {_c('GRN')}Material de estudio generado:{R}")
            for f in st["outputs"]:
                print(f"    · {f}")
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
