#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       TEMPO DESIGNER  v1.0                                  ║
║          Diseño de la arquitectura de tempo (agógica) de una obra MIDI      ║
║                                                                              ║
║  performer aplica rubato LOCAL por frase con reglas automáticas; esta        ║
║  herramienta DISEÑA la agógica GLOBAL de la obra y la hornea en el mapa de   ║
║  tempo del MIDI (meta 0x51): terrazas de tempo por sección, rampas de        ║
║  accelerando / ritardando entre compases, calderones (fermate) puntuales    ║
║  y ritardando final. Es el equivalente temporal de tension_designer         ║
║  (que diseña la curva de tensión) y de climax_engineer (que diseña la        ║
║  arquitectura de energía): aquí se diseña la arquitectura del TIEMPO.        ║
║                                                                              ║
║  MODELO DE TEMPO:                                                            ║
║    puntos de control  — pares compás:BPM que definen la curva               ║
║    forma (--shape)    — step (terrazas) | linear (rampas) | smooth (coseno) ║
║    calderón (--fermata) — alarga un instante concreto (tempo local lento)   ║
║    ritardando final   — desaceleración progresiva en los últimos compases   ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] LECTURA    — lee el MIDI y su compás/tempo base                        ║
║  [2] CURVA      — construye BPM por compás desde los puntos de control      ║
║  [3] CALDERONES — inserta ralentizaciones locales en los compases pedidos   ║
║  [4] HORNEADO   — reemplaza los eventos meta 0x51 por la curva diseñada      ║
║  [5] INFORME    — sparkline de BPM + duración antes/después + sidecar JSON   ║
║                                                                              ║
║  USO:                                                                        ║
║    python tempo_designer.py obra.mid --point 1:60 --point 8:96 --shape linear║
║    python tempo_designer.py obra.mid --section 1:72 --section 9:120          ║
║    python tempo_designer.py obra.mid --fermata 16 --final-rit 4              ║
║    python tempo_designer.py obra.mid --plan agogica.tempo.json               ║
║    python tempo_designer.py obra.mid --point 1:120 --point 16:60 --dry-run   ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi                MIDI de entrada                                        ║
║    --output FILE       MIDI de salida (default: <obra>_tempo.mid)           ║
║    --point BAR:BPM     Punto de control (repetible). BAR es 1-based.        ║
║    --section BAR:BPM   Alias de --point con forma 'step' implícita          ║
║    --shape S           step | linear | smooth  (default: linear)            ║
║    --base BPM          Tempo base si no hay puntos (default: el del MIDI)   ║
║    --fermata SPEC      BAR[:beat][:factor]  (repetible). factor def. 2.5    ║
║    --final-rit BARS    Ritardando en los últimos BARS compases              ║
║    --final-rit-factor F  Factor de frenado final (default: 1.6)             ║
║    --min-bpm F         Límite inferior de seguridad (default: 20)           ║
║    --max-bpm F         Límite superior de seguridad (default: 300)          ║
║    --plan FILE         Carga un plan .tempo.json (anula --point/--section)  ║
║    --dry-run           Solo informe y sidecar, sin escribir MIDI             ║
║    --no-json           No escribir el sidecar .tempo.json                    ║
║    --no-color          Desactivar colores ANSI                              ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from tempo_designer import design_tempo                                   ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

try:
    from playability_auditor import (read_midi, write_midi, MidiEvent,
                                     TimeContext, _write_vlq)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_SPARK = "▁▂▃▄▅▆▇█"


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def sparkline(v: np.ndarray) -> str:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return ""
    if v.max() <= v.min():
        return _SPARK[3] * len(v)
    x = (v - v.min()) / (v.max() - v.min())
    return "".join(_SPARK[min(7, int(round(t * 7)))] for t in x)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE TEMPO
# ══════════════════════════════════════════════════════════════════════════════

def bpm_to_us(bpm: float) -> int:
    return int(round(60_000_000 / max(1e-6, bpm)))


def us_to_bpm(us: int) -> float:
    return 60_000_000 / max(1, us)


def _n_bars(mid, tc: TimeContext) -> int:
    last = 0
    for trk in mid.tracks:
        for ev in trk.events:
            if ev.kind in ("note_on", "note_off"):
                last = max(last, ev.abs)
    return max(1, tc.bar(last))


def _bar_start_tick(tc: TimeContext, bar: int) -> int:
    """Tick de inicio del compás 'bar' (1-based)."""
    t0, _ = tc.bar_range_ticks(bar, bar)
    return t0


def _tempo_seconds(tempo_map: List[Tuple[int, int]], end_tick: int, tpb: int) -> float:
    """Duración en segundos de [0, end_tick) dado un tempo_map (tick, us/negra)."""
    tm = sorted(tempo_map)
    if not tm or tm[0][0] > 0:
        tm = [(0, tm[0][1] if tm else 500000)] + tm
    sec = 0.0
    for i, (tk, us) in enumerate(tm):
        nxt = tm[i + 1][0] if i + 1 < len(tm) else end_tick
        nxt = min(nxt, end_tick)
        if nxt > tk:
            sec += (nxt - tk) * us / 1e6 / tpb
        if nxt >= end_tick:
            break
    return sec


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE LA CURVA DE TEMPO POR COMPÁS
# ══════════════════════════════════════════════════════════════════════════════

def build_bpm_curve(points: List[Tuple[int, float]], n_bars: int,
                    base: float, shape: str) -> np.ndarray:
    """Devuelve un array de BPM por compás (índice 0 = compás 1)."""
    curve = np.full(n_bars, float(base))
    pts = sorted((b, v) for b, v in points if 1 <= b <= n_bars)
    if not pts:
        return curve
    # extremos anclados
    if pts[0][0] > 1:
        pts = [(1, pts[0][1])] + pts
    if pts[-1][0] < n_bars:
        pts = pts + [(n_bars, pts[-1][1])]

    if shape == "step":
        for i, (b, v) in enumerate(pts):
            nxt = pts[i + 1][0] if i + 1 < len(pts) else n_bars + 1
            for bar in range(b, min(nxt, n_bars + 1)):
                curve[bar - 1] = v
    else:
        for i in range(len(pts) - 1):
            b0, v0 = pts[i]
            b1, v1 = pts[i + 1]
            span = max(1, b1 - b0)
            for bar in range(b0, b1 + 1):
                t = (bar - b0) / span
                if shape == "smooth":
                    t = 0.5 - 0.5 * np.cos(np.pi * t)     # ease in-out coseno
                curve[bar - 1] = v0 + (v1 - v0) * t
    return curve


def apply_final_rit(curve: np.ndarray, bars: int, factor: float) -> np.ndarray:
    """Ritardando progresivo en los últimos 'bars' compases."""
    if bars <= 0:
        return curve
    n = len(curve)
    bars = min(bars, n)
    start_bpm = curve[n - bars]
    end_bpm = start_bpm / max(1.0, factor)
    for k in range(bars):
        bar = n - bars + k
        t = (k + 1) / bars
        t = 0.5 - 0.5 * np.cos(np.pi * t)                 # frenado suave
        curve[bar] = start_bpm + (end_bpm - start_bpm) * t
    return curve


# ══════════════════════════════════════════════════════════════════════════════
#  HORNEADO EN EL MAPA DE TEMPO
# ══════════════════════════════════════════════════════════════════════════════

def _strip_tempo_events(mid):
    for trk in mid.tracks:
        trk.events = [e for e in trk.events
                      if not (e.kind == "meta" and e.meta_type == 0x51)]


def _tempo_meta(tick: int, us: int) -> MidiEvent:
    data = bytes([0xFF, 0x51]) + _write_vlq(3) + us.to_bytes(3, "big")
    return MidiEvent(abs=tick, kind="meta", meta_type=0x51, data=data)


def bake_tempo(mid, tc: TimeContext, bpm_curve: np.ndarray,
               fermatas: List[Tuple[int, float, float]],
               min_bpm: float, max_bpm: float) -> List[Tuple[int, int]]:
    """Construye la lista de eventos (tick, us) y la inyecta en la pista 0.
    Devuelve el nuevo tempo_map."""
    n_bars = len(bpm_curve)
    events: List[Tuple[int, int]] = []
    last_us = None
    for bar in range(1, n_bars + 1):
        bpm = float(np.clip(bpm_curve[bar - 1], min_bpm, max_bpm))
        us = bpm_to_us(bpm)
        tick = _bar_start_tick(tc, bar)
        if us != last_us:
            events.append((tick, us))
            last_us = us

    # calderones: ralentización local en un instante y restauración inmediata
    for (bar, beat, factor) in fermatas:
        if not (1 <= bar <= n_bars):
            continue
        bar_t0 = _bar_start_tick(tc, bar)
        # ancho del beat en ticks según compás vigente
        beat_ticks = tc.tpb
        pos = bar_t0 + int(round(beat * beat_ticks))
        # BPM vigente en ese compás
        base_bpm = float(np.clip(bpm_curve[bar - 1], min_bpm, max_bpm))
        slow_us = bpm_to_us(max(min_bpm, base_bpm / max(1.01, factor)))
        restore_us = bpm_to_us(base_bpm)
        events.append((pos, slow_us))
        events.append((pos + beat_ticks, restore_us))

    # ordenar y deduplicar por tick (último gana), y quitar redundancias
    events.sort()
    dedup: Dict[int, int] = {}
    for tk, us in events:
        dedup[tk] = us
    clean: List[Tuple[int, int]] = []
    prev_us = None
    for tk in sorted(dedup):
        us = dedup[tk]
        if us != prev_us:
            clean.append((tk, us))
            prev_us = us
    if not clean or clean[0][0] > 0:
        first_bpm = float(np.clip(bpm_curve[0], min_bpm, max_bpm))
        clean = [(0, bpm_to_us(first_bpm))] + clean

    _strip_tempo_events(mid)
    for tk, us in clean:
        mid.tracks[0].events.append(_tempo_meta(tk, us))
    mid.tracks[0].events.sort(key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))
    mid.tempo_map = clean
    return clean


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def design_tempo(path: str, out_path: Optional[str] = None,
                 points: Optional[List[Tuple[int, float]]] = None,
                 shape: str = "linear", base: Optional[float] = None,
                 fermatas: Optional[List[Tuple[int, float, float]]] = None,
                 final_rit_bars: int = 0, final_rit_factor: float = 1.6,
                 min_bpm: float = 20.0, max_bpm: float = 300.0,
                 dry_run: bool = False, write_json: bool = True) -> dict:
    """API pública. Diseña y hornea la arquitectura de tempo. Devuelve stats."""
    mid = read_midi(path)
    tc = TimeContext(mid)
    n_bars = _n_bars(mid, tc)

    base_bpm = base if base is not None else us_to_bpm(mid.tempo_map[0][1])
    points = list(points or [])
    fermatas = list(fermatas or [])

    before_map = list(mid.tempo_map)
    end_tick = max((ev.abs for trk in mid.tracks for ev in trk.events
                    if ev.kind == "note_off"), default=0)
    dur_before = _tempo_seconds(before_map, end_tick, mid.tpb)

    curve = build_bpm_curve(points, n_bars, base_bpm, shape)
    if final_rit_bars > 0:
        curve = apply_final_rit(curve, final_rit_bars, final_rit_factor)

    new_map = bake_tempo(mid, tc, curve, fermatas, min_bpm, max_bpm)
    dur_after = _tempo_seconds(new_map, end_tick, mid.tpb)

    stats = {
        "file": path, "n_bars": int(n_bars),
        "base_bpm": round(base_bpm, 2),
        "bpm_curve": [round(float(x), 2) for x in curve],
        "bpm_min": round(float(curve.min()), 2),
        "bpm_max": round(float(curve.max()), 2),
        "tempo_events": len(new_map),
        "fermatas": [{"bar": b, "beat": bt, "factor": f} for b, bt, f in fermatas],
        "duration_before_s": round(dur_before, 3),
        "duration_after_s": round(dur_after, 3),
        "shape": shape,
    }

    if write_json:
        sidecar = str(Path(path).with_suffix("")) + ".tempo.json"
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump({"version": VERSION, "n_bars": int(n_bars),
                       "shape": shape, "base_bpm": round(base_bpm, 2),
                       "points": [[b, v] for b, v in points],
                       "final_rit_bars": final_rit_bars,
                       "final_rit_factor": final_rit_factor,
                       "fermatas": stats["fermatas"],
                       "bpm_curve": stats["bpm_curve"]}, fh,
                      ensure_ascii=False, indent=2)
        stats["sidecar"] = sidecar

    if not dry_run:
        out = out_path or str(Path(path).with_name(Path(path).stem + "_tempo.mid"))
        write_midi(mid, out)
        stats["output"] = out
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO DE ESPECIFICACIONES CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_point(s: str) -> Tuple[int, float]:
    bar, bpm = s.split(":")
    return int(bar), float(bpm)


def _parse_fermata(s: str) -> Tuple[int, float, float]:
    parts = s.split(":")
    bar = int(parts[0])
    beat = float(parts[1]) if len(parts) > 1 and parts[1] != "" else 0.0
    factor = float(parts[2]) if len(parts) > 2 and parts[2] != "" else 2.5
    return bar, beat, factor


def _load_plan(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="tempo_designer.py",
        description="Diseño de la arquitectura de tempo (agógica) de una obra MIDI.")
    ap.add_argument("midi")
    ap.add_argument("--output")
    ap.add_argument("--point", action="append", default=[], metavar="BAR:BPM")
    ap.add_argument("--section", action="append", default=[], metavar="BAR:BPM")
    ap.add_argument("--shape", choices=["step", "linear", "smooth"], default="linear")
    ap.add_argument("--base", type=float)
    ap.add_argument("--fermata", action="append", default=[], metavar="BAR[:beat][:factor]")
    ap.add_argument("--final-rit", type=int, default=0)
    ap.add_argument("--final-rit-factor", type=float, default=1.6)
    ap.add_argument("--min-bpm", type=float, default=20.0)
    ap.add_argument("--max-bpm", type=float, default=300.0)
    ap.add_argument("--plan")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    shape = args.shape
    points: List[Tuple[int, float]] = []
    fermatas: List[Tuple[int, float, float]] = []
    base = args.base
    final_rit = args.final_rit
    final_rit_factor = args.final_rit_factor

    if args.plan:
        try:
            plan = _load_plan(args.plan)
        except Exception as e:
            print(f"[ERROR] no se pudo leer el plan: {e}", file=sys.stderr)
            return 1
        points = [(int(b), float(v)) for b, v in plan.get("points", [])]
        shape = plan.get("shape", shape)
        base = plan.get("base_bpm", base)
        final_rit = plan.get("final_rit_bars", final_rit)
        final_rit_factor = plan.get("final_rit_factor", final_rit_factor)
        fermatas = [(int(f["bar"]), float(f.get("beat", 0.0)),
                     float(f.get("factor", 2.5))) for f in plan.get("fermatas", [])]
    else:
        try:
            points = [_parse_point(p) for p in args.point]
            if args.section:
                points += [_parse_point(p) for p in args.section]
                shape = "step"
            fermatas = [_parse_fermata(f) for f in args.fermata]
        except ValueError as e:
            print(f"[ERROR] especificación inválida: {e}", file=sys.stderr)
            return 1

    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}")
    print(f"  {B}TEMPO DESIGNER v{VERSION}  ·  {args.midi}{R}")
    if points:
        print(f"  Puntos: {', '.join(f'c.{b}={v:g}' for b, v in sorted(points))}"
              f"   forma: {shape}")
    if fermatas:
        print(f"  Calderones: {', '.join(f'c.{b}' for b, _, _ in fermatas)}")
    if final_rit:
        print(f"  Ritardando final: {final_rit} compases  (×{final_rit_factor:g})")
    print(f"{'═' * 78}")

    try:
        st = design_tempo(
            args.midi, args.output, points=points, shape=shape, base=base,
            fermatas=fermatas, final_rit_bars=final_rit,
            final_rit_factor=final_rit_factor, min_bpm=args.min_bpm,
            max_bpm=args.max_bpm, dry_run=args.dry_run, write_json=not args.no_json)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    curve = np.array(st["bpm_curve"])
    print(f"  Compases: {st['n_bars']}   BPM base: {st['base_bpm']:g}")
    print(f"  curva BPM: {sparkline(curve)}  "
          f"[{st['bpm_min']:g} – {st['bpm_max']:g}]")
    print(f"  eventos de tempo escritos: {st['tempo_events']}")
    print(f"  duración: {st['duration_before_s']:g}s → "
          f"{_c('CYA')}{st['duration_after_s']:g}s{R}")
    if "sidecar" in st:
        print(f"  sidecar: {st['sidecar']}")
    if args.dry_run:
        print(f"  {G}(dry-run: no se escribió MIDI){R}")
    elif "output" in st:
        print(f"  {_c('GRN')}MIDI con tempo diseñado: {st['output']}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
