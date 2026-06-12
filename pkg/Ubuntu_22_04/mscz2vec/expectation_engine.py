#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      EXPECTATION ENGINE  v1.0                                ║
║       Manipulación de expectativas armónicas para intensificar emoción      ║
║                                                                              ║
║  Según Meyer/Huron, la emoción musical nace de generar expectativas y       ║
║  retrasarlas, negarlas o cumplirlas en el momento justo. Esta herramienta   ║
║  localiza los puntos de resolución previstos (cadencias V→I) e inyecta      ║
║  dispositivos emotivos, reservando la resolución plena para el final.       ║
║  Complementa a reharmonizer y voice_leader, que optimizan corrección,       ║
║  no sorpresa.                                                                ║
║                                                                              ║
║  DISPOSITIVOS:                                                               ║
║    cadencia_rota — V→I se convierte en V→vi (la dominante del acorde de     ║
║                    llegada sube a la submediante; el bajo tónica baja a     ║
║                    la sexta) — la resolución esperada se niega              ║
║    suspension    — la nota melódica previa se prolonga sobre el cambio      ║
║                    de acorde y la resolución llega tarde (retardo 4–3)      ║
║    appoggiatura  — nota extraña acentuada sobre la meta melódica, que       ║
║                    resuelve descendiendo (anhelo)                            ║
║    evasion       — el bajo de la resolución se mueve a la tercera           ║
║                    (I en primera inversión): llegada debilitada              ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] ANÁLISIS   — tonalidad (Krumhansl) + acorde por compás (plantillas)    ║
║  [2] CADENCIAS  — detección de movimientos V→I                              ║
║  [3] SELECCIÓN  — qué cadencias tratar (densidad + semilla); la última      ║
║                   se preserva intacta (o reforzada) con --climax-resolve    ║
║  [4] INYECCIÓN  — aplicación de dispositivos sobre las notas                ║
║  [5] ESCRITURA  — MIDI transformado + informe de cadencias                  ║
║                                                                              ║
║  USO:                                                                        ║
║    python expectation_engine.py obra.mid                                     ║
║    python expectation_engine.py obra.mid --devices cadencia_rota suspension ║
║    python expectation_engine.py obra.mid --density 0.5 --seed 7             ║
║    python expectation_engine.py obra.mid --climax-resolve                   ║
║    python expectation_engine.py obra.mid --report                           ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi               MIDI de entrada                                        ║
║    --output FILE      MIDI de salida (default: <obra>_expect.mid)           ║
║    --devices D...     Dispositivos a usar (default: todos)                  ║
║    --density F        Fracción de cadencias tratadas 0–1 (default: 0.7)    ║
║    --climax-resolve   Negar todas menos la última, que se refuerza          ║
║    --seed N           Semilla aleatoria (default: 42)                       ║
║    --report           Solo informe de cadencias, sin escribir MIDI          ║
║    --no-color         Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from expectation_engine import process_midi, find_cadences               ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from playability_auditor import (read_midi, write_midi, MidiEvent,
                                     extract_notes, pitch_name)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


ALL_DEVICES = ["cadencia_rota", "suspension", "appoggiatura", "evasion"]
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS: TONALIDAD, ACORDES, CADENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def detect_key(all_notes) -> tuple:
    """(pc_tonica, 'major'|'minor') por correlación de Krumhansl."""
    hist = np.zeros(12)
    for n in all_notes:
        hist[n.pitch % 12] += n.end - n.start
    if hist.sum() == 0:
        return 0, "major"
    best = (0, "major", -2.0)
    for k in range(12):
        for mode, prof in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
            r = float(np.corrcoef(np.roll(prof, k), hist)[0, 1])
            if r > best[2]:
                best = (k, mode, r)
    return best[0], best[1]


def bar_chord(notes, bar, bar_ticks) -> Optional[tuple]:
    """Acorde dominante del compás: (pc_raiz, 'maj'|'min') o None."""
    t0, t1 = bar * bar_ticks, (bar + 1) * bar_ticks
    hist = np.zeros(12)
    for n in notes:
        ov = min(n.end, t1) - max(n.start, t0)
        if ov > 0:
            hist[n.pitch % 12] += ov
    if hist.sum() == 0:
        return None
    hist /= hist.sum()
    best = (None, -1.0)
    for root in range(12):
        for qual, third in (("maj", 4), ("min", 3)):
            score = hist[root] * 1.2 + hist[(root + third) % 12] + hist[(root + 7) % 12]
            if score > best[1]:
                best = ((root, qual), score)
    return best[0]


@dataclass
class Cadence:
    bar_v: int            # compás de la dominante
    bar_i: int            # compás de resolución
    device: Optional[str] = None


def find_cadences(all_notes, bar_ticks, n_bars, key_pc) -> List[Cadence]:
    dom_pc, ton_pc = (key_pc + 7) % 12, key_pc
    chords = [bar_chord(all_notes, b, bar_ticks) for b in range(n_bars)]
    out = []
    for b in range(n_bars - 1):
        c0, c1 = chords[b], chords[b + 1]
        if c0 and c1 and c0[0] == dom_pc and c1[0] == ton_pc:
            out.append(Cadence(bar_v=b + 1, bar_i=b + 2))     # 1-based
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DISPOSITIVOS
# ══════════════════════════════════════════════════════════════════════════════

def _notes_in_bar(notes, bar0, bar_ticks):
    t0, t1 = bar0 * bar_ticks, (bar0 + 1) * bar_ticks
    return [n for n in notes if t0 <= n.start < t1]


def apply_cadencia_rota(tracks_notes, cad, bar_ticks, key_pc, bass_idx):
    """V→I pasa a V→vi: la dominante de la llegada sube +2; el bajo tónica −3."""
    changed = 0
    dom_pc, ton_pc = (key_pc + 7) % 12, key_pc
    for idx, notes in tracks_notes.items():
        for n in _notes_in_bar(notes, cad.bar_i - 1, bar_ticks):
            if idx == bass_idx and n.pitch % 12 == ton_pc:
                n.pitch -= 3                                   # tónica → submediante
                changed += 1
            elif n.pitch % 12 == dom_pc:
                n.pitch += 2                                   # 5ª → 6ª grado
                changed += 1
    return changed


def apply_suspension(tracks_notes, cad, bar_ticks, key_pc, mel_idx, tpb):
    """La melodía previa se prolonga sobre el downbeat; la meta llega tarde."""
    notes = tracks_notes[mel_idx]
    t_db = (cad.bar_i - 1) * bar_ticks
    target = next((n for n in sorted(notes, key=lambda x: x.start)
                   if n.start >= t_db), None)
    prev = max((n for n in notes if n.start < t_db), default=None,
               key=lambda n: n.start)
    if not target or not prev:
        return 0
    delay = tpb // 2
    prev.end = t_db + delay                                    # retardo
    target.start = min(target.start + delay, target.end - 1)   # resolución tardía
    target.vel = max(1, target.vel - 6)                        # resolución suave
    return 2


def apply_appoggiatura(tracks_notes, cad, bar_ticks, key_pc, mel_idx, tpb):
    """Nota extraña acentuada un tono sobre la meta, resolviendo en ella."""
    from playability_auditor import Note
    notes = tracks_notes[mel_idx]
    t_db = (cad.bar_i - 1) * bar_ticks
    target = next((n for n in sorted(notes, key=lambda x: x.start)
                   if n.start >= t_db), None)
    if not target:
        return 0
    dur = max(tpb // 2, (target.end - target.start) // 2)
    app = Note(pitch=target.pitch + 2, vel=min(127, target.vel + 10),
               start=target.start, end=target.start + dur, channel=target.channel)
    target.start = app.end                                     # la meta se retrasa
    if target.end <= target.start:
        target.end = target.start + tpb // 2
    target.vel = max(1, target.vel - 4)
    notes.append(app)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return 1


def apply_evasion(tracks_notes, cad, bar_ticks, key_pc, bass_idx):
    """I en primera inversión: el bajo tónica se mueve a la tercera."""
    changed = 0
    for n in _notes_in_bar(tracks_notes[bass_idx], cad.bar_i - 1, bar_ticks):
        if n.pitch % 12 == key_pc:
            n.pitch += 4
            changed += 1
    return changed


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


def process_midi(path: str, out_path: Optional[str] = None,
                 devices: Optional[List[str]] = None, density: float = 0.7,
                 climax_resolve: bool = False, seed: int = 42,
                 report_only: bool = False) -> dict:
    """API pública. Devuelve {'key','cadences':[Cadence], 'changed': int}."""
    rng = random.Random(seed)
    devices = devices or ALL_DEVICES
    mid = read_midi(path)
    tpb = mid.tpb
    num, den = mid.timesig_map[0][1], mid.timesig_map[0][2]
    bar_ticks = tpb * 4 * num // den

    tracks_notes = {i: extract_notes(t) for i, t in enumerate(mid.tracks)}
    tracks_notes = {i: ns for i, ns in tracks_notes.items() if ns}
    if not tracks_notes:
        raise ValueError("el MIDI no contiene notas")
    all_notes = [n for ns in tracks_notes.values() for n in ns]
    n_bars = max(n.end for n in all_notes) // bar_ticks + 1

    key_pc, mode = detect_key(all_notes)
    mel_idx = max(tracks_notes, key=lambda i: sum(n.pitch for n in tracks_notes[i])
                  / len(tracks_notes[i]))
    bass_idx = min(tracks_notes, key=lambda i: sum(n.pitch for n in tracks_notes[i])
                   / len(tracks_notes[i]))

    cadences = find_cadences(all_notes, bar_ticks, n_bars, key_pc)
    changed = 0
    if not report_only and cadences:
        treatable = cadences[:-1] if climax_resolve and len(cadences) > 1 else cadences
        k = max(1, round(len(treatable) * density)) if treatable else 0
        chosen = rng.sample(treatable, min(k, len(treatable)))
        chosen.sort(key=lambda c: c.bar_v)
        for j, cad in enumerate(chosen):
            dev = devices[j % len(devices)]
            cad.device = dev
            if dev == "cadencia_rota":
                changed += apply_cadencia_rota(tracks_notes, cad, bar_ticks,
                                               key_pc, bass_idx)
            elif dev == "suspension":
                changed += apply_suspension(tracks_notes, cad, bar_ticks,
                                            key_pc, mel_idx, tpb)
            elif dev == "appoggiatura":
                changed += apply_appoggiatura(tracks_notes, cad, bar_ticks,
                                              key_pc, mel_idx, tpb)
            elif dev == "evasion":
                changed += apply_evasion(tracks_notes, cad, bar_ticks,
                                         key_pc, bass_idx)
        if climax_resolve and len(cadences) > 1:
            last = cadences[-1]
            last.device = "resolucion_plena"
            for ns in tracks_notes.values():                    # refuerzo final
                for n in _notes_in_bar(ns, last.bar_i - 1, bar_ticks):
                    n.vel = min(127, n.vel + 10)
                    changed += 1
        for idx, trk in enumerate(mid.tracks):
            if idx in tracks_notes:
                _rebuild(trk, tracks_notes[idx])
        out = out_path or str(Path(path).with_name(Path(path).stem + "_expect.mid"))
        write_midi(mid, out)
    else:
        out = None

    return {"key": f"{NOTE_NAMES[key_pc]} {mode}", "key_pc": key_pc,
            "melody_track": mel_idx, "bass_track": bass_idx,
            "cadences": cadences, "changed": changed, "output": out}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="expectation_engine.py",
                                 description="Manipulación de expectativas armónicas.")
    ap.add_argument("midi")
    ap.add_argument("--output")
    ap.add_argument("--devices", nargs="+", choices=ALL_DEVICES, default=ALL_DEVICES)
    ap.add_argument("--density", type=float, default=0.7)
    ap.add_argument("--climax-resolve", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G, Y = _c("B"), _c("R"), _c("GRN"), _c("YEL")
    print(f"\n{'═' * 78}")
    print(f"  {B}EXPECTATION ENGINE v{VERSION}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")
    try:
        res = process_midi(args.midi, args.output, devices=args.devices,
                           density=max(0.0, min(1.0, args.density)),
                           climax_resolve=args.climax_resolve,
                           seed=args.seed, report_only=args.report)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Tonalidad: {res['key']}   melodía: pista {res['melody_track']}   "
          f"bajo: pista {res['bass_track']}")
    print(f"  Cadencias V→I detectadas: {len(res['cadences'])}")
    for cad in res["cadences"]:
        dev = cad.device or "—"
        mark = Y + "⚑" if cad.device and cad.device != "resolucion_plena" \
            else G + "✔"
        print(f"    {mark} c.{cad.bar_v}→c.{cad.bar_i}: {dev}{R}")
    if res["output"]:
        print(f"  Notas modificadas/añadidas: {res['changed']}")
        print(f"  {G}MIDI transformado: {res['output']}{R}")
    else:
        print(f"  {_c('G')}(solo informe){R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
