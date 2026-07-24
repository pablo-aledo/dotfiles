#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     TONICIZATION ENGINE  v1.0                               ║
║   Inserción decidida de dominantes secundarios en una progresión existente  ║
║                                                                              ║
║  Toma una progresión de acordes ya escrita (MIDI de acordes en bloque, del  ║
║  tipo que exportan chord_progression_generator.py / reharmonizer.py) y      ║
║  decide EN QUÉ compases y DE QUÉ MANERA introducir dominantes secundarios   ║
║  (V/x) que tonicen sus acordes diatónicos, sin tocar el resto de la obra.   ║
║                                                                              ║
║  Es el análogo de expectation_engine.py (que manipula expectativas en       ║
║  cadencias V→I) pero para tonicización: en vez de negar/retrasar una        ║
║  resolución esperada, INTRODUCE una tensión nueva y localizada — la 5ª de   ║
║  un grado diatónico — que resuelve de inmediato dentro del mismo compás.    ║
║  Complementa a chord_progression_generator (que ya puede generar V/x desde  ║
║  cero) y a reharmonizer (que sustituye la progresión entera): esta          ║
║  herramienta opera quirúrgicamente sobre UNA progresión ya fijada.          ║
║                                                                              ║
║  MECÁNICA:                                                                   ║
║  Un compás objetivo (acorde diatónico X: ii, IV, V, vi…) se parte en dos:   ║
║  la 1ª mitad se sustituye por una tríada mayor (o 7ª de dominante) sobre    ║
║  la raíz de V/x, en el registro más cercano al original —NUNCA se          ║
║  transporta literalmente el voicing de X, porque conservaría su calidad    ║
║  (p.ej. menor, si X es ii o vi) y un dominante secundario es siempre       ║
║  mayor/dominante—. La 2ª mitad conserva el acorde X original intacto.      ║
║  Solo tonicíza grados donde ya hay sitio (nunca inserta compases nuevos).  ║
║                                                                              ║
║  ALCANCE: pensada para progresiones de UN acorde por compás (la salida      ║
║  por defecto de chord_progression_generator/reharmonizer). Si tu MIDI       ║
║  cambia de acorde varias veces por compás, cuantízalo antes a 1/compás.    ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] ANÁLISIS    — reutiliza harmonic_analyzer.py (tonalidad + numeral +    ║
║                    función por compás; ya reconoce tonicizaciones           ║
║                    existentes para no duplicarlas sobre sí mismas)          ║
║  [2] CANDIDATOS  — compases con acorde diatónico tonicizable (--targets)    ║
║  [3] SELECCIÓN   — --density (fracción) + --curve (sigue una curva de       ║
║                    tensión de tension_designer) + --min-gap + --seed        ║
║  [4] INYECCIÓN   — parte cada nota del compás, escribe V/x transportando    ║
║                    el voicing original; añade la 7ª si --seventh           ║
║  [5] ESCRITURA   — MIDI transformado + informe de qué se tonicizó y por qué ║
║                                                                              ║
║  USO:                                                                        ║
║    python tonicization_engine.py progresion.mid                              ║
║    python tonicization_engine.py progresion.mid --targets ii IV V vi        ║
║    python tonicization_engine.py progresion.mid --density 0.6 --seed 7      ║
║    python tonicization_engine.py progresion.mid --curve obra.curves.json    ║
║    python tonicization_engine.py progresion.mid --seventh                    ║
║    python tonicization_engine.py progresion.mid --report                     ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi               MIDI de entrada (progresión de acordes)                ║
║    --output FILE      MIDI de salida (default: <obra>_tonic.mid)            ║
║    --key KEY          Fuerza la tonalidad (p.ej. Cmaj, Am). Si no,          ║
║                       autodetecta vía harmonic_analyzer                     ║
║    --targets T [T…]   Grados a considerar tonicizables (default: según el   ║
║                       modo — mayor: ii iii IV V vi · menor: III iv VI VII)  ║
║    --density F        Fracción de candidatos tratados 0–1 (default: 0.5)   ║
║    --min-gap N        Compases mínimos entre tonicizaciones (default: 1)    ║
║    --seventh          Usa 7ª de dominante en vez de tríada simple            ║
║    --curve FILE       .curves.json de tension_designer: prioriza los        ║
║                       compases de mayor tensión en vez de elegir a la par   ║
║    --seed N           Semilla aleatoria (default: 42)                       ║
║    --report           Solo informe, sin escribir MIDI                        ║
║    --no-color         Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from tonicization_engine import process_midi, find_tonicizable_bars      ║
║                                                                              ║
║  DEPENDENCIAS: mido · harmonic_analyzer.py (mismo directorio, análisis)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path

import mido

try:
    from harmonic_analyzer import analyze_harmony
except ImportError:
    print("ERROR: requiere harmonic_analyzer.py en el mismo directorio (análisis armónico)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _parse_key_string(key_str: str):
    """
    Traduce el 'key' que devuelve harmonic_analyzer ('Cmaj', 'Am', 'F#maj'…)
    a (tonic_pc, mode). harmonic_analyzer solo emite nombres con sostenidos,
    pero se acepta 'b' también por si --key se reutiliza de otra fuente.
    """
    if key_str.endswith("maj"):
        name, mode = key_str[:-3], "major"
    else:
        name, mode = key_str[:-1], "minor"
    pc = _NOTE_PC[name[0]]
    if len(name) > 1:
        if name[1] == "#":
            pc += 1
        elif name[1] == "b":
            pc -= 1
    return pc % 12, mode


# Grados diatónicos tonicizables. Deben coincidir con las tablas privadas
# _MAJ_DEGREES / _MIN_DEGREES de harmonic_analyzer.py; se duplican aquí (solo
# 7 entradas cada una) para que esta herramienta no dependa de nombres
# internos de otro módulo y siga siendo mayoritariamente autocontenida.
_MAJ_DEGREES = {0: "I", 2: "ii", 4: "iii", 5: "IV", 7: "V", 9: "vi", 11: "vii"}
_MIN_DEGREES = {0: "i", 2: "ii", 3: "III", 5: "iv", 7: "v", 8: "VI", 10: "VII"}

DEFAULT_TARGETS_MAJOR = ["ii", "iii", "IV", "V", "vi"]
DEFAULT_TARGETS_MINOR = ["III", "iv", "VI", "VII"]


# ══════════════════════════════════════════════════════════════════════════════
#  E/S MIDI (autocontenida vía mido; no depende de playability_auditor.py)
# ══════════════════════════════════════════════════════════════════════════════

class Note:
    __slots__ = ("pitch", "start", "end", "velocity", "track", "channel")

    def __init__(self, pitch, start, end, velocity, track, channel):
        self.pitch, self.start, self.end = pitch, start, end
        self.velocity, self.track, self.channel = velocity, track, channel


class SimpleTimeContext:
    """
    Conversión compás→ticks asumiendo compás constante (sin cambios de
    compás a mitad de obra: usa el primer time_signature que encuentre, o
    4/4 por defecto). Suficiente para progresiones de acordes, que rara vez
    cambian de compás.
    """
    def __init__(self, mid: mido.MidiFile):
        self.tpb = mid.ticks_per_beat
        self.numerator, self.denominator = 4, 4
        for trk in mid.tracks:
            for msg in trk:
                if msg.type == "time_signature":
                    self.numerator, self.denominator = msg.numerator, msg.denominator
                    break
            else:
                continue
            break
        self.ticks_per_bar = int(self.tpb * self.numerator * 4 / self.denominator)

    def bar_range_ticks(self, bar: int):
        t0 = (bar - 1) * self.ticks_per_bar
        return t0, t0 + self.ticks_per_bar


def _extract_notes(mid: mido.MidiFile):
    """Notas por pista con tick absoluto de inicio/fin (formato 1, un reloj compartido)."""
    notes = []
    for ti, trk in enumerate(mid.tracks):
        t = 0
        active = {}
        for msg in trk:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start, vel = active.pop(key)
                    notes.append(Note(msg.note, start, t, vel, ti, msg.channel))
    return notes


def _rebuild_midi(orig_mid: mido.MidiFile, notes, tpb: int) -> mido.MidiFile:
    """
    Reconstruye un MIDI conservando la meta-información original de cada
    pista (tempo, compás, nombre, program_change…) en su tick absoluto, y
    sustituyendo únicamente los eventos de nota por los de `notes`.
    """
    new_mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    notes_by_track = {i: [] for i in range(len(orig_mid.tracks))}
    for n in notes:
        notes_by_track.setdefault(n.track, []).append(n)

    for ti, orig_trk in enumerate(orig_mid.tracks):
        meta_events = []
        t = 0
        for msg in orig_trk:
            t += msg.time
            if msg.type not in ("note_on", "note_off", "end_of_track"):
                meta_events.append((t, msg.copy(time=0)))

        note_events = []
        for n in notes_by_track.get(ti, []):
            if n.end <= n.start:
                continue
            note_events.append((n.start, mido.Message(
                "note_on", channel=n.channel, note=n.pitch, velocity=n.velocity, time=0)))
            note_events.append((n.end, mido.Message(
                "note_off", channel=n.channel, note=n.pitch, velocity=0, time=0)))

        # orden a igualdad de tick: meta(0) < note_off(1) < note_on(2)
        combined = [(tk, 0, m) for tk, m in meta_events]
        combined += [(tk, 1, m) for tk, m in note_events if m.type == "note_off"]
        combined += [(tk, 2, m) for tk, m in note_events if m.type == "note_on"]
        combined.sort(key=lambda e: (e[0], e[1]))

        new_trk = mido.MidiTrack()
        last_t = 0
        for tick, _, m in combined:
            new_trk.append(m.copy(time=max(0, tick - last_t)))
            last_t = tick
        new_trk.append(mido.MetaMessage("end_of_track", time=0))
        new_mid.tracks.append(new_trk)

    return new_mid


# ══════════════════════════════════════════════════════════════════════════════
#  [2] CANDIDATOS
# ══════════════════════════════════════════════════════════════════════════════

def find_tonicizable_bars(analysis: dict, tonic_pc: int, mode: str, targets):
    """
    A partir del análisis de harmonic_analyzer.analyze_harmony(), devuelve
    los compases candidatos a tonicizar: acordes diatónicos (no ya
    dominantes secundarios) cuyo grado está en `targets`.
    """
    table = _MAJ_DEGREES if mode == "major" else _MIN_DEGREES
    candidates = []
    for c in analysis["chords"]:
        if not c["chord"] or c.get("secondary_dominant_of"):
            continue
        deg = (c["root_pc"] - tonic_pc) % 12
        label = table.get(deg)
        if label is None or label not in targets:
            continue
        candidates.append({
            "bar": c["bar"],
            "target_root_pc": c["root_pc"],
            "target_label": label,
            "dominant_root_pc": (c["root_pc"] + 7) % 12,
        })
    return candidates


# ══════════════════════════════════════════════════════════════════════════════
#  [3] SELECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def select_tonicizations(candidates, density=0.5, min_gap=1, curve=None, seed=42):
    """
    Elige qué candidatos tonicizar. Con --curve, prioriza los compases de
    mayor tensión; si no, todos parten con el mismo peso y desempata la
    semilla. Respeta --min-gap compases entre tonicizaciones elegidas.
    """
    if not candidates or density <= 0:
        return []
    rnd = random.Random(seed)
    weighted = []
    for cand in candidates:
        w = 1.0
        if curve:
            idx = cand["bar"] - 1
            if 0 <= idx < len(curve):
                w = float(curve[idx])
        w += rnd.uniform(-0.01, 0.01)  # desempate estable con la semilla
        weighted.append((w, cand))
    weighted.sort(key=lambda x: -x[0])

    n_select = max(1, round(density * len(candidates)))
    selected, used_bars = [], []
    for _, cand in weighted:
        if len(selected) >= n_select:
            break
        if any(abs(cand["bar"] - b) < 1 + min_gap for b in used_bars):
            continue
        selected.append(cand)
        used_bars.append(cand["bar"])

    selected.sort(key=lambda c: c["bar"])
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  [4] INYECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def inject_secondary_dominants(notes, selections, tc: SimpleTimeContext, add_seventh=False):
    """
    Para cada compás seleccionado: parte en dos el acorde que empieza en ese
    compás. La 1ª mitad se sustituye por una tríada mayor (o 7ª de dominante
    con --seventh) construida sobre la raíz de V/x, en el registro más
    cercano al acorde original — NO se transporta literalmente el voicing
    original, porque eso conservaría su calidad (p.ej. menor si x es ii o
    vi), y un dominante secundario SIEMPRE es de calidad mayor/dominante,
    tonicíze lo que tonicíze. La 2ª mitad conserva el acorde original
    intacto. Devuelve (nuevas_notas, informe).
    """
    result = [Note(n.pitch, n.start, n.end, n.velocity, n.track, n.channel) for n in notes]
    report = []

    for sel in selections:
        t0, t1 = tc.bar_range_ticks(sel["bar"])
        chord_notes = [n for n in result if t0 <= n.start < t1]
        if not chord_notes:
            continue

        ref = chord_notes[0]
        split = ref.start + max(1, (ref.end - ref.start) // 2)

        # Construye la tríada mayor (o 7ª de dominante) de V/x en el
        # registro más cercano posible al acorde original.
        intervals = [0, 4, 7, 10] if add_seventh else [0, 4, 7]
        n_voices = max(len(chord_notes), len(intervals))
        lowest = min(n.pitch for n in chord_notes)
        root_pitch = min((sel["dominant_root_pc"] + 12 * o for o in range(11)),
                         key=lambda p: abs(p - lowest))
        voicing = []
        octv = 0
        while len(voicing) < n_voices:
            voicing.extend(root_pitch + iv + 12 * octv for iv in intervals)
            octv += 1
        voicing = sorted(voicing)[:n_voices]

        # Cada voz nueva hereda velocity/track/channel de la nota original
        # más próxima en altura (para no perder instrumentación/dinámica).
        template_pool = sorted(chord_notes, key=lambda n: n.pitch)
        new_chord = []
        for i, new_pitch in enumerate(voicing):
            tmpl = template_pool[min(i, len(template_pool) - 1)]
            new_chord.append(Note(new_pitch, ref.start, split, tmpl.velocity,
                                  tmpl.track, tmpl.channel))

        for n in chord_notes:
            n.start = split  # el acorde original ahora solo suena en la 2ª mitad

        result.extend(new_chord)
        label = f"{'V7' if add_seventh else 'V'}/{sel['target_label']}"
        report.append({"bar": sel["bar"], "insertion": label, "resolves_to": sel["target_label"]})

    result.sort(key=lambda n: (n.start, n.pitch))
    return result, report


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def process_midi(path, output=None, key=None, targets=None, density=0.5,
                 min_gap=1, seventh=False, curve_path=None, seed=42,
                 report_only=False) -> dict:
    """API pública. Analiza, selecciona e inyecta dominantes secundarios. Devuelve un informe."""
    analysis = analyze_harmony(path, key=key)
    tonic_pc, mode = _parse_key_string(analysis["key"])

    if targets is None:
        targets = DEFAULT_TARGETS_MAJOR if mode == "major" else DEFAULT_TARGETS_MINOR

    candidates = find_tonicizable_bars(analysis, tonic_pc, mode, targets)

    curve = None
    if curve_path:
        with open(curve_path) as f:
            curve = json.load(f).get("tension", [])

    selections = select_tonicizations(candidates, density=density, min_gap=min_gap,
                                      curve=curve, seed=seed)

    result = {
        "file": path, "key": analysis["key"], "n_bars": analysis["n_bars"],
        "targets": targets, "n_candidates": len(candidates), "n_selected": len(selections),
        "injected": [],
    }

    if selections:
        mid = mido.MidiFile(path)
        tc = SimpleTimeContext(mid)
        notes = _extract_notes(mid)
        new_notes, injected = inject_secondary_dominants(notes, selections, tc, add_seventh=seventh)
        result["injected"] = injected

        if not report_only and injected:
            out_path = output or f"{Path(path).with_suffix('')}_tonic.mid"
            new_mid = _rebuild_midi(mid, new_notes, mid.ticks_per_beat)
            new_mid.save(out_path)
            result["output"] = out_path

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="tonicization_engine.py",
        description="Decide dónde y cómo insertar dominantes secundarios en una progresión existente.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("midi", help="MIDI de entrada (progresión de acordes)")
    ap.add_argument("--output", default=None, help="MIDI de salida (default: <obra>_tonic.mid)")
    ap.add_argument("--key", default=None, help="Fuerza la tonalidad (p.ej. Cmaj, Am)")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="Grados tonicizables a considerar (default: según el modo)")
    ap.add_argument("--density", type=float, default=0.5,
                    help="Fracción de candidatos tratados 0-1 (default: 0.5)")
    ap.add_argument("--min-gap", type=int, default=1,
                    help="Compases mínimos entre tonicizaciones (default: 1)")
    ap.add_argument("--seventh", action="store_true",
                    help="Usa 7ª de dominante en vez de tríada simple")
    ap.add_argument("--curve", default=None, metavar="FILE",
                    help=".curves.json de tension_designer: prioriza compases de mayor tensión")
    ap.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (default: 42)")
    ap.add_argument("--report", action="store_true", help="Solo informe, sin escribir MIDI")
    ap.add_argument("--no-color", action="store_true", help="Desactiva colores ANSI")
    args = ap.parse_args()

    if args.no_color:
        _USE_COLOR = False
    if not os.path.exists(args.midi):
        print(f"[ERROR] No se encuentra: {args.midi}")
        sys.exit(1)

    try:
        result = process_midi(
            args.midi, output=args.output, key=args.key, targets=args.targets,
            density=args.density, min_gap=args.min_gap, seventh=args.seventh,
            curve_path=args.curve, seed=args.seed, report_only=args.report,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    B, R, GRN, YEL, CYA = _c("B"), _c("R"), _c("GRN"), _c("YEL"), _c("CYA")
    print(f"\n{CYA}{'═'*64}{R}")
    print(f"  {B}TONICIZATION ENGINE{R} v{VERSION} — {result['file']}")
    print(f"  Tonalidad: {result['key']}  ·  {result['n_bars']} compases  ·  "
          f"objetivos: {', '.join(result['targets'])}")
    print(f"  Candidatos tonicizables: {result['n_candidates']}  ·  "
          f"seleccionados: {result['n_selected']}")
    print(f"{CYA}{'═'*64}{R}")

    if result["injected"]:
        print(f"\n  {B}Tonicizaciones {'(propuestas, sin escribir)' if args.report else 'insertadas'}:{R}")
        for it in result["injected"]:
            print(f"    c.{it['bar']}: {GRN}{it['insertion']}{R}  → resuelve en {it['resolves_to']}")
    else:
        print(f"\n  {YEL}(ninguna tonicización seleccionada con estos parámetros){R}")

    if "output" in result:
        print(f"\n  → MIDI: {result['output']}")
    print()


if __name__ == "__main__":
    main()
