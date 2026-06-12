#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      SILENCE DESIGNER  v1.0                                  ║
║            Diseño del silencio como recurso dramático en MIDIs              ║
║                                                                              ║
║  Los compositores automáticos del ecosistema tienden a rellenar: textura    ║
║  continua, sin aire. El recurso emotivo más barato y menos usado es el      ║
║  silencio. Esta herramienta AUDITA la saturación de la obra y ABRE espacio  ║
║  donde la narrativa lo pide, sin alterar la métrica (los huecos se crean    ║
║  acortando notas, nunca insertando tiempo).                                  ║
║                                                                              ║
║  OPERACIONES:                                                                ║
║    luftpause    — respiraciones en las fronteras de sección (detectadas     ║
║                   por novedad armónica + cambio de densidad)                 ║
║    pausa        — silencio dramático general antes de un compás clave       ║
║                   (--pause-before N, o automático antes del pico de        ║
║                   energía con --auto)                                        ║
║    reduccion    — adelgazamiento a una sola voz (melodía) en los compases   ║
║                   previos al pico: intimidad antes del golpe                ║
║    fermata      — calderón final: el último acorde se prolonga y se deja    ║
║                   resonar (única operación que extiende la duración)        ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] AUDITORÍA  — saturación por compás (fracción de tiempo con sonido)     ║
║  [2] FRONTERAS  — novedad por croma + densidad → candidatas a luftpause     ║
║  [3] APERTURA   — truncado de notas para crear los huecos                   ║
║  [4] INFORME    — sparkline de saturación antes/después + escritura         ║
║                                                                              ║
║  USO:                                                                        ║
║    python silence_designer.py obra.mid --report                              ║
║    python silence_designer.py obra.mid                                       ║
║    python silence_designer.py obra.mid --pause-before 21 --gap 1.0          ║
║    python silence_designer.py obra.mid --auto --thin --fermata              ║
║    python silence_designer.py obra.mid --luftpauses 3 --gap 0.5             ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi               MIDI de entrada                                        ║
║    --output FILE      MIDI de salida (default: <obra>_silence.mid)          ║
║    --luftpauses N     Nº de respiraciones en fronteras (default: 3)         ║
║    --gap F            Tamaño del hueco en pulsos (default: 0.5)             ║
║    --pause-before N   Pausa dramática antes del compás N (gap doble)        ║
║    --auto             Situar la pausa dramática antes del pico de energía   ║
║    --thin             Reducción a una voz en los 2 compases pre-pausa       ║
║    --fermata          Calderón final (×1.8 el último acorde)                ║
║    --melody-track N   Pista a conservar en la reducción (default: auto)     ║
║    --report           Solo auditoría de saturación, sin escribir MIDI        ║
║    --no-color         Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from silence_designer import design_silence, saturation_curve            ║
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
    from playability_auditor import read_midi, write_midi, MidiEvent, extract_notes
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
    x = np.clip(v, 0, 1)
    return "".join(_SPARK[min(7, int(round(t * 7)))] for t in x)


# ══════════════════════════════════════════════════════════════════════════════
#  AUDITORÍA DE SATURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def saturation_curve(all_notes, bar_ticks, n_bars) -> np.ndarray:
    """Fracción de cada compás cubierta por >=1 nota sonando (global)."""
    sat = np.zeros(n_bars)
    res = 16                                          # subdivisiones por compás
    grid = np.zeros(n_bars * res, dtype=bool)
    step = bar_ticks / res
    for n in all_notes:
        i0 = int(n.start / step)
        i1 = max(i0 + 1, int(np.ceil(n.end / step)))
        grid[i0:min(i1, len(grid))] = True
    for b in range(n_bars):
        sat[b] = grid[b * res:(b + 1) * res].mean()
    return sat


def novelty_boundaries(tracks_notes, bar_ticks, n_bars):
    """Fronteras candidatas: salto de croma + cambio de densidad entre compases."""
    chroma = np.zeros((n_bars, 12))
    dens = np.zeros(n_bars)
    for ns in tracks_notes.values():
        for n in ns:
            b = min(n.start // bar_ticks, n_bars - 1)
            chroma[b, n.pitch % 12] += n.end - n.start
            dens[b] += 1
    nov = np.zeros(n_bars)
    for b in range(1, n_bars):
        a, c = chroma[b - 1], chroma[b]
        if a.sum() and c.sum():
            cos = float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-9))
            nov[b] = (1 - cos)
        nov[b] += abs(dens[b] - dens[b - 1]) / (dens.max() + 1e-9) * 0.5
    return nov                                         # nov[b] = novedad al ENTRAR en b


# ══════════════════════════════════════════════════════════════════════════════
#  APERTURA DE HUECOS (métrica intacta: solo se acortan notas)
# ══════════════════════════════════════════════════════════════════════════════

def open_gap(tracks_notes, at_tick: int, gap_ticks: int,
             only_track: Optional[int] = None) -> int:
    """Trunca toda nota que suene en [at_tick - gap, at_tick)."""
    g0 = at_tick - gap_ticks
    cut = 0
    for idx, ns in tracks_notes.items():
        if only_track is not None and idx != only_track:
            continue
        for n in ns:
            if n.start < at_tick and n.end > g0:
                if n.start >= g0:                      # nota íntegra dentro del hueco
                    n.end = n.start + 1
                else:
                    n.end = g0
                cut += 1
    return cut


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild(trk, notes):
    others = [e for e in trk.events if e.kind not in ("note_on", "note_off")]
    evs = list(others)
    for n in notes:
        if n.end - n.start < 2:                        # notas reducidas a nada
            continue
        evs.append(MidiEvent(abs=n.start, kind="note_on", channel=n.channel,
                             pitch=n.pitch, vel=n.vel))
        evs.append(MidiEvent(abs=n.end, kind="note_off", channel=n.channel,
                             pitch=n.pitch, vel=0))
    trk.events = sorted(evs, key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))


def design_silence(path: str, out_path: Optional[str] = None,
                   luftpauses: int = 3, gap_beats: float = 0.5,
                   pause_before: Optional[int] = None, auto_pause: bool = False,
                   thin: bool = False, fermata: bool = False,
                   melody_track: Optional[int] = None,
                   report_only: bool = False) -> dict:
    """API pública. Devuelve curvas de saturación y estadísticas."""
    mid = read_midi(path)
    tpb = mid.tpb
    num, den = mid.timesig_map[0][1], mid.timesig_map[0][2]
    bar_ticks = tpb * 4 * num // den
    gap_ticks = max(1, int(gap_beats * tpb))

    tracks_notes = {i: extract_notes(t) for i, t in enumerate(mid.tracks)}
    tracks_notes = {i: ns for i, ns in tracks_notes.items() if ns}
    if not tracks_notes:
        raise ValueError("el MIDI no contiene notas")
    all_notes = [n for ns in tracks_notes.values() for n in ns]
    n_bars = max(n.end for n in all_notes) // bar_ticks + 1
    mel_idx = melody_track if melody_track is not None else \
        max(tracks_notes, key=lambda i: sum(n.pitch for n in tracks_notes[i])
            / len(tracks_notes[i]))

    before = saturation_curve(all_notes, bar_ticks, n_bars)
    # tramo continuo más largo de saturación >= 0.98
    longest, cur = 0, 0
    for s in before:
        cur = cur + 1 if s >= 0.98 else 0
        longest = max(longest, cur)

    stats = {"n_bars": n_bars, "before": before, "longest_saturated": longest,
             "luftpauses": [], "pause_bar": None, "cuts": 0,
             "thinned": 0, "fermata": False}

    if report_only:
        stats["after"] = before
        return stats

    # [2-3] luftpauses en las fronteras más novedosas
    nov = novelty_boundaries(tracks_notes, bar_ticks, n_bars)
    order = [int(b) for b in np.argsort(nov)[::-1] if nov[b] > 0.05 and b > 0]
    chosen = sorted(order[:max(0, luftpauses)])
    for b in chosen:
        stats["cuts"] += open_gap(tracks_notes, b * bar_ticks, gap_ticks)
        stats["luftpauses"].append(b + 1)              # 1-based

    # pausa dramática
    pb = pause_before
    if auto_pause and pb is None:
        vel = np.zeros(n_bars)
        cnt = np.zeros(n_bars)
        for ns in tracks_notes.values():
            for n in ns:
                b = min(n.start // bar_ticks, n_bars - 1)
                vel[b] += n.vel
                cnt[b] += 1
        vel[cnt > 0] /= cnt[cnt > 0]
        energy = before * 0.5 + (vel / 127) * 0.5
        pb = int(np.argmax(energy)) + 1
        if pb <= 2 or energy.max() - energy.min() < 0.05:
            pb = max(2, int(round(0.618 * n_bars)))   # curva plana → áurea
    if pb is not None and 1 < pb <= n_bars:
        stats["cuts"] += open_gap(tracks_notes, (pb - 1) * bar_ticks, gap_ticks * 2)
        stats["pause_bar"] = pb
        if thin:                                       # reducción a una voz
            z0 = max(0, pb - 3) * bar_ticks
            z1 = (pb - 1) * bar_ticks
            for idx, ns in list(tracks_notes.items()):
                if idx == mel_idx:
                    continue
                kept = [n for n in ns if not (z0 <= n.start < z1)]
                stats["thinned"] += len(ns) - len(kept)
                tracks_notes[idx] = kept

    # calderón final
    if fermata:
        last_t = max(n.start for n in
                     (n for ns in tracks_notes.values() for n in ns))
        for ns in tracks_notes.values():
            for n in ns:
                if n.start >= last_t - tpb // 8:
                    n.end = n.start + int((n.end - n.start) * 1.8) + tpb
        stats["fermata"] = True

    all_after = [n for ns in tracks_notes.values() for n in ns
                 if n.end - n.start >= 2]
    n_bars_after = max(n_bars, max(n.end for n in all_after) // bar_ticks + 1)
    stats["after"] = saturation_curve(all_after, bar_ticks, n_bars_after)[:n_bars]

    for idx, trk in enumerate(mid.tracks):
        if idx in tracks_notes:
            _rebuild(trk, tracks_notes[idx])
    out = out_path or str(Path(path).with_name(Path(path).stem + "_silence.mid"))
    write_midi(mid, out)
    stats["output"] = out
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="silence_designer.py",
                                 description="Diseño del silencio como recurso dramático.")
    ap.add_argument("midi")
    ap.add_argument("--output")
    ap.add_argument("--luftpauses", type=int, default=3)
    ap.add_argument("--gap", type=float, default=0.5, metavar="PULSOS")
    ap.add_argument("--pause-before", type=int, metavar="COMPAS")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--thin", action="store_true")
    ap.add_argument("--fermata", action="store_true")
    ap.add_argument("--melody-track", type=int)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}")
    print(f"  {B}SILENCE DESIGNER v{VERSION}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")
    try:
        st = design_silence(args.midi, args.output, luftpauses=args.luftpauses,
                            gap_beats=args.gap, pause_before=args.pause_before,
                            auto_pause=args.auto, thin=args.thin,
                            fermata=args.fermata, melody_track=args.melody_track,
                            report_only=args.report)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Compases: {st['n_bars']}   tramo saturado más largo: "
          f"{st['longest_saturated']} compases")
    print(f"  saturación antes  : {sparkline(st['before'])}")
    print(f"  saturación después: {sparkline(st['after'])}")
    if st["luftpauses"]:
        print(f"  Luftpausen al entrar en compases: "
              f"{', '.join(str(b) for b in st['luftpauses'])}")
    if st["pause_bar"]:
        print(f"  Pausa dramática antes de c.{st['pause_bar']}"
              + ("   (con reducción a una voz)" if args.thin else ""))
    if st["fermata"]:
        print("  Calderón final aplicado")
    print(f"  Notas truncadas: {st['cuts']}   retiradas (reducción): {st['thinned']}")
    if args.report:
        print(f"  {G}(solo auditoría){R}")
    else:
        print(f"  {_c('GRN')}MIDI con silencio: {st['output']}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
