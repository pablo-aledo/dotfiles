#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CLIMAX ENGINEER  v1.0                                  ║
║          Ingeniería deliberada del punto culminante de una obra MIDI        ║
║                                                                              ║
║  arc_supervisor DETECTA si el arco emocional se cumple; esta herramienta    ║
║  lo CONSTRUYE: localiza el clímax (por defecto en la proporción áurea),     ║
║  coordina todas las dimensiones hacia él y elimina falsos clímax que        ║
║  roben energía al principal.                                                 ║
║                                                                              ║
║  OPERACIONES:                                                                ║
║    remodelado   — la curva de energía (densidad + dinámica + registro +     ║
║                   polifonía) se reescala compás a compás hacia una curva    ║
║                   objetivo con un único pico asimétrico                      ║
║    vaciado      — los compases previos al pico bajan en dinámica y se       ║
║                   adelgaza la textura, para que el golpe destaque            ║
║    refuerzo     — duplicación a la octava de la melodía en la zona de pico  ║
║    anti-falsos  — picos secundarios > umbral relativo se atenúan            ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] CURVA      — energía por compás, suavizada                             ║
║  [2] OBJETIVO   — curva diana con pico en --position y anchura --width      ║
║  [3] ESCALADO   — velocidades por compás hacia la diana (limitado)          ║
║  [4] VACIADO    — hueco dramático en los --hollow-bars previos              ║
║  [5] REFUERZO   — octavas de melodía en ±width/2 compases del pico          ║
║  [6] INFORME    — sparkline antes/después + escritura                       ║
║                                                                              ║
║  USO:                                                                        ║
║    python climax_engineer.py obra.mid                                        ║
║    python climax_engineer.py obra.mid --position 0.618 --width 4            ║
║    python climax_engineer.py obra.mid --hollow-bars 2 --intensity 0.8       ║
║    python climax_engineer.py obra.mid --no-double --no-hollow               ║
║    python climax_engineer.py obra.mid --dry-run                             ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi               MIDI de entrada                                        ║
║    --output FILE      MIDI de salida (default: <obra>_climax.mid)           ║
║    --position F       Posición del clímax 0–1 (default: 0.618, áurea)      ║
║    --width N          Anchura de la zona de pico en compases (default: 4)   ║
║    --hollow-bars N    Compases de vaciado previo (default: 2)               ║
║    --intensity F      Cuánto acercarse a la diana 0–1 (default: 0.7)        ║
║    --melody-track N   Pista a duplicar en octavas (default: autodetección)  ║
║    --no-double        Sin duplicación de octavas                             ║
║    --no-hollow        Sin vaciado previo                                     ║
║    --no-antifalse     No atenuar falsos clímax                               ║
║    --dry-run          Solo informe y curvas, sin escribir MIDI               ║
║    --no-color         Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from climax_engineer import engineer_climax                               ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from playability_auditor import (read_midi, write_midi, MidiEvent,
                                     extract_notes, Note)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m"}
_USE_COLOR = sys.stdout.isatty()
_SPARK = "▁▂▃▄▅▆▇█"


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def sparkline(v: np.ndarray) -> str:
    if v.max() <= v.min():
        return _SPARK[0] * len(v)
    x = (v - v.min()) / (v.max() - v.min())
    return "".join(_SPARK[min(7, int(round(t * 7)))] for t in x)


# ══════════════════════════════════════════════════════════════════════════════
#  CURVA DE ENERGÍA
# ══════════════════════════════════════════════════════════════════════════════

def energy_curve(tracks_notes, bar_ticks, n_bars) -> np.ndarray:
    dens = np.zeros(n_bars)
    vel = np.zeros(n_bars)
    reg = np.zeros(n_bars)
    poly = np.zeros(n_bars)
    cnt = np.zeros(n_bars)
    for ns in tracks_notes.values():
        for n in ns:
            b = min(n.start // bar_ticks, n_bars - 1)
            dens[b] += 1
            vel[b] += n.vel
            reg[b] += n.pitch
            poly[b] += (n.end - n.start)
            cnt[b] += 1
    mask = cnt > 0
    vel[mask] /= cnt[mask]
    reg[mask] /= cnt[mask]
    poly /= bar_ticks                                  # voces medias sonando

    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    e = 0.4 * norm(dens) + 0.3 * norm(vel) + 0.2 * norm(reg) + 0.1 * norm(poly)
    if len(e) >= 3:                                    # suavizado (media móvil 3)
        e = np.convolve(e, np.ones(3) / 3, mode="same")
    return e


def target_curve(n_bars: int, position: float, width: int) -> np.ndarray:
    """Pico asimétrico: subida progresiva, caída más rápida."""
    peak = max(1, min(n_bars - 2, int(round(position * (n_bars - 1)))))
    t = np.zeros(n_bars)
    for b in range(n_bars):
        if b <= peak:
            t[b] = (b / peak) ** 1.6 if peak else 1.0
        else:
            t[b] = max(0.0, 1 - ((b - peak) / max(1, (n_bars - 1 - peak))) ** 0.8)
    lo = 0.18                                          # suelo de energía
    return lo + (1 - lo) * t


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild(trk, notes):
    others = [e for e in trk.events if e.kind not in ("note_on", "note_off")]
    evs = list(others)
    for n in notes:
        evs.append(MidiEvent(abs=n.start, kind="note_on", channel=n.channel,
                             pitch=n.pitch, vel=n.vel))
        evs.append(MidiEvent(abs=n.end, kind="note_off", channel=n.channel,
                             pitch=n.pitch, vel=0))
    trk.events = sorted(evs, key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))


def engineer_climax(path: str, out_path: Optional[str] = None,
                    position: float = 0.618, width: int = 4,
                    hollow_bars: int = 2, intensity: float = 0.7,
                    melody_track: Optional[int] = None,
                    double_octaves: bool = True, hollow: bool = True,
                    antifalse: bool = True, dry_run: bool = False) -> dict:
    """API pública. Devuelve curvas y estadísticas."""
    mid = read_midi(path)
    tpb = mid.tpb
    num, den = mid.timesig_map[0][1], mid.timesig_map[0][2]
    bar_ticks = tpb * 4 * num // den

    tracks_notes = {i: extract_notes(t) for i, t in enumerate(mid.tracks)}
    tracks_notes = {i: ns for i, ns in tracks_notes.items() if ns}
    if not tracks_notes:
        raise ValueError("el MIDI no contiene notas")
    n_bars = max(max(n.end for n in ns) for ns in tracks_notes.values()) \
        // bar_ticks + 1

    before = energy_curve(tracks_notes, bar_ticks, n_bars)
    target = target_curve(n_bars, position, width)
    peak_bar = int(np.argmax(target))
    mel_idx = melody_track if melody_track is not None else \
        max(tracks_notes, key=lambda i: sum(n.pitch for n in tracks_notes[i])
            / len(tracks_notes[i]))

    # [3] escalado de velocidades por compás hacia la diana
    eps = 0.05
    ratios = np.clip(target / np.maximum(before, eps), 0.55, 1.6)
    ratios = 1 + (ratios - 1) * intensity
    scaled = 0
    for ns in tracks_notes.values():
        for n in ns:
            b = min(n.start // bar_ticks, n_bars - 1)
            nv = int(round(n.vel * ratios[b]))
            if nv != n.vel:
                scaled += 1
            n.vel = max(1, min(127, nv))

    # [4] vaciado previo al pico (dinámica + adelgazamiento de textura)
    removed = 0
    if hollow and hollow_bars > 0:
        h0 = max(0, peak_bar - hollow_bars)
        for idx, ns in list(tracks_notes.items()):
            in_zone = [n for n in ns if h0 <= n.start // bar_ticks < peak_bar]
            if not in_zone:
                continue
            for k, n in enumerate(in_zone):
                pos = (n.start // bar_ticks - h0 + 1) / max(1, hollow_bars)
                n.vel = max(1, int(n.vel * (0.75 - 0.25 * pos)))
            if idx != mel_idx:                          # adelgazar acompañamiento
                med = float(np.median([n.vel for n in in_zone]))
                drop = {id(n) for n in in_zone if n.vel < med}
                kept = [n for n in ns if id(n) not in drop]
                removed += len(ns) - len(kept)
                tracks_notes[idx] = kept

    # [5] refuerzo: duplicación de octavas de la melodía en la zona de pico
    doubled = 0
    if double_octaves:
        z0, z1 = peak_bar - width // 2, peak_bar + (width + 1) // 2
        extra = []
        for n in tracks_notes[mel_idx]:
            b = n.start // bar_ticks
            if z0 <= b < z1 and n.pitch + 12 <= 124:
                extra.append(Note(pitch=n.pitch + 12, vel=max(1, n.vel - 8),
                                  start=n.start, end=n.end, channel=n.channel))
                doubled += 1
        tracks_notes[mel_idx].extend(extra)
        tracks_notes[mel_idx].sort(key=lambda n: (n.start, n.pitch))

    # [6] anti-falsos clímax: picos secundarios atenuados
    attenuated = 0
    if antifalse:
        cur = energy_curve(tracks_notes, bar_ticks, n_bars)
        thr = 0.92 * cur[peak_bar] if cur[peak_bar] > 0 else 1.0
        for b in range(1, n_bars - 1):
            if abs(b - peak_bar) > width and cur[b] >= thr \
                    and cur[b] >= cur[b - 1] and cur[b] >= cur[b + 1]:
                for ns in tracks_notes.values():
                    for n in ns:
                        if n.start // bar_ticks == b:
                            n.vel = max(1, int(n.vel * 0.85))
                            attenuated += 1

    after = energy_curve(tracks_notes, bar_ticks, n_bars)
    stats = {"n_bars": n_bars, "peak_bar": peak_bar + 1,
             "before": before, "target": target, "after": after,
             "vel_scaled": scaled, "notes_removed": removed,
             "octaves_doubled": doubled, "false_attenuated": attenuated,
             "peak_before_bar": int(np.argmax(before)) + 1,
             "peak_after_bar": int(np.argmax(after)) + 1}
    if not dry_run:
        for idx, trk in enumerate(mid.tracks):
            if idx in tracks_notes:
                _rebuild(trk, tracks_notes[idx])
        out = out_path or str(Path(path).with_name(Path(path).stem + "_climax.mid"))
        write_midi(mid, out)
        stats["output"] = out
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="climax_engineer.py",
                                 description="Ingeniería del punto culminante.")
    ap.add_argument("midi")
    ap.add_argument("--output")
    ap.add_argument("--position", type=float, default=0.618)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--hollow-bars", type=int, default=2)
    ap.add_argument("--intensity", type=float, default=0.7)
    ap.add_argument("--melody-track", type=int)
    ap.add_argument("--no-double", action="store_true")
    ap.add_argument("--no-hollow", action="store_true")
    ap.add_argument("--no-antifalse", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}")
    print(f"  {B}CLIMAX ENGINEER v{VERSION}  ·  {args.midi}{R}")
    print(f"  Posición diana: {args.position:.3f}   anchura: {args.width}   "
          f"intensidad: {args.intensity:.2f}")
    print(f"{'═' * 78}")
    try:
        st = engineer_climax(
            args.midi, args.output, position=args.position, width=args.width,
            hollow_bars=args.hollow_bars, intensity=args.intensity,
            melody_track=args.melody_track, double_octaves=not args.no_double,
            hollow=not args.no_hollow, antifalse=not args.no_antifalse,
            dry_run=args.dry_run)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Compases: {st['n_bars']}   clímax diana: c.{st['peak_bar']}")
    print(f"  energía antes  : {sparkline(st['before'])}  (pico en c."
          f"{st['peak_before_bar']})")
    print(f"  diana          : {sparkline(st['target'])}")
    print(f"  energía después: {sparkline(st['after'])}  (pico en c."
          f"{st['peak_after_bar']})")
    print(f"  Velocidades reescaladas: {st['vel_scaled']}   "
          f"notas retiradas (vaciado): {st['notes_removed']}")
    print(f"  Octavas añadidas: {st['octaves_doubled']}   "
          f"falsos clímax atenuados: {st['false_attenuated']}")
    if args.dry_run:
        print(f"  {G}(dry-run: no se escribió MIDI){R}")
    else:
        print(f"  {_c('GRN')}MIDI con clímax: {st['output']}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
