#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LEADSHEET REALIZER  v1.0                                 ║
║        Cifrado de acordes → acompañamiento pianístico tocable               ║
║                                                                              ║
║  chord_progression_generator y reharmonizer trabajan la ARMONÍA en          ║
║  abstracto; voice_leader produce el coral SATB. Falta el puente práctico    ║
║  para quien toca: convertir un cifrado (C  Am7  F  G) en un acompañamiento  ║
║  de piano que una persona pueda LEER Y TOCAR, con patrones de comping        ║
║  idiomáticos y un voicing cómodo para las manos.                             ║
║                                                                              ║
║  ENTRADA: cifrados separados por espacios (un acorde por compás), con        ║
║  duración opcional 'C:2' (compases). Símbolos admitidos:                     ║
║    C  Cm  C7  Cmaj7  Cm7  Cdim  Caug  Csus2  Csus4  Cm7b5  Cdim7             ║
║    C6  Cm6  C9  Cadd9  y bajo invertido con barra:  C/E  G/B                 ║
║                                                                              ║
║  PATRONES (--pattern):                                                       ║
║    block     — acorde sostenido + bajo (sencillo, para empezar)             ║
║    ballad    — acorde en tiempos 1 y 3, bajo raíz-quinta                     ║
║    arpeggio  — arpegio ascendente de las notas del acorde                    ║
║    alberti   — bajo Alberti (grave-agudo-medio-agudo)                        ║
║    waltz     — vals 3/4: bajo en 1, acorde en 2 y 3                          ║
║    pop       — raíz en la mano izq., acordes a contratiempo en la derecha    ║
║                                                                              ║
║  USO:                                                                        ║
║    python leadsheet_realizer.py --chords \"C Am F G\"                          ║
║    python leadsheet_realizer.py --chords \"Cmaj7 A7 Dm7 G7\" --pattern arpeggio║
║    python leadsheet_realizer.py --chords \"C G/B Am F\" --pattern ballad       ║
║    python leadsheet_realizer.py --file cancion.txt --pattern pop --tempo 100 ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --chords TEXT     Cifrados separados por espacios (un acorde/compás)      ║
║    --file FILE       Lee los cifrados de un fichero de texto                 ║
║    --pattern P       block|ballad|arpeggio|alberti|waltz|pop (def: block)   ║
║    --tempo BPM       Tempo del MIDI (default: 96)                           ║
║    --octave N        Octava central del voicing MD (default: 4)             ║
║    --output FILE     MIDI de salida (default: leadsheet.mid)                ║
║    --no-color        Desactivar colores ANSI                               ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from leadsheet_realizer import realize_leadsheet                          ║
║                                                                              ║
║  DEPENDENCIAS: playability_auditor.py (mismo directorio, E/S MIDI)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import re
import math
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

try:
    from playability_auditor import (MidiData, MidiTrackData, MidiEvent,
                                     write_midi, pitch_name, _write_vlq)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
TPB = 480

# sufijo → intervalos (semitonos desde la fundamental)
QUALITIES = {
    "": [0, 4, 7], "maj": [0, 4, 7], "M": [0, 4, 7],
    "m": [0, 3, 7], "min": [0, 3, 7], "-": [0, 3, 7],
    "dim": [0, 3, 6], "o": [0, 3, 6], "aug": [0, 4, 8], "+": [0, 4, 8],
    "sus2": [0, 2, 7], "sus4": [0, 5, 7], "sus": [0, 5, 7],
    "7": [0, 4, 7, 10], "maj7": [0, 4, 7, 11], "M7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10], "min7": [0, 3, 7, 10], "-7": [0, 3, 7, 10],
    "m7b5": [0, 3, 6, 10], "ø": [0, 3, 6, 10], "dim7": [0, 3, 6, 9], "o7": [0, 3, 6, 9],
    "6": [0, 4, 7, 9], "m6": [0, 3, 7, 9],
    "9": [0, 4, 7, 10, 14], "maj9": [0, 4, 7, 11, 14], "m9": [0, 3, 7, 10, 14],
    "add9": [0, 4, 7, 14], "madd9": [0, 3, 7, 14],
}


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO DE CIFRADOS
# ══════════════════════════════════════════════════════════════════════════════

_CHORD_RE = re.compile(r"^([A-G])([#b♯♭]?)([^/]*)(?:/([A-G][#b♯♭]?))?$")


def parse_chord(sym: str) -> dict:
    m = _CHORD_RE.match(sym.strip())
    if not m:
        raise ValueError(f"cifrado no reconocido: '{sym}'")
    letter, acc, qual, bass = m.groups()
    root = _PC[letter] + (1 if acc in "#♯" else -1 if acc in "b♭" else 0)
    root %= 12
    if qual not in QUALITIES:
        raise ValueError(f"calidad no reconocida en '{sym}' (sufijo '{qual}')")
    intervals = QUALITIES[qual]
    bass_pc = None
    if bass:
        bl = bass[0]
        ba = bass[1:] if len(bass) > 1 else ""
        bass_pc = (_PC[bl] + (1 if ba in "#♯" else -1 if ba in "b♭" else 0)) % 12
    return {"symbol": sym, "root": root, "intervals": intervals, "bass_pc": bass_pc}


def parse_progression(text: str) -> List[Tuple[dict, int]]:
    out = []
    for tok in text.split():
        if ":" in tok:
            sym, dur = tok.rsplit(":", 1)
            bars = max(1, int(dur))
        else:
            sym, bars = tok, 1
        out.append((parse_chord(sym), bars))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  VOICING
# ══════════════════════════════════════════════════════════════════════════════

def voicing(chord: dict, octave: int) -> Tuple[List[int], int]:
    """Devuelve (notas_MD, bajo_MI) en un registro cómodo."""
    base = (octave + 1) * 12
    root = chord["root"]
    rh = [base + root + iv for iv in chord["intervals"]]
    # comprime a menos de una octava y media
    while rh[-1] - rh[0] > 16 and len(rh) > 3:
        rh = rh[:-1]
    bass_pc = chord["bass_pc"] if chord["bass_pc"] is not None else root
    lh = (octave - 1) * 12 + 12 + bass_pc          # bajo ~2 octavas abajo
    return rh, lh


# ══════════════════════════════════════════════════════════════════════════════
#  PATRONES DE ACOMPAÑAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def _ev(seq, start, dur, pitch, vel):
    seq.append((start, dur, pitch, vel))


def render_bar(pattern, rh, lh, fifth, bar_start, bar_ticks, rh_seq, lh_seq):
    q = bar_ticks // 4
    if pattern == "block":
        for p in rh:
            _ev(rh_seq, bar_start, bar_ticks, p, 66)
        _ev(lh_seq, bar_start, bar_ticks, lh, 70)
    elif pattern == "ballad":
        for beat in (0, 2):
            for p in rh:
                _ev(rh_seq, bar_start + beat * q, 2 * q, p, 64)
        _ev(lh_seq, bar_start, 2 * q, lh, 70)
        _ev(lh_seq, bar_start + 2 * q, 2 * q, fifth, 64)
    elif pattern == "arpeggio":
        notes = rh + [rh[0] + 12]
        step = bar_ticks // len(notes)
        for i, p in enumerate(notes):
            _ev(rh_seq, bar_start + i * step, step, p, 68)
        _ev(lh_seq, bar_start, bar_ticks, lh, 68)
    elif pattern == "alberti":
        low, hi, mid = rh[0], rh[-1], rh[1] if len(rh) > 2 else rh[0]
        pat = [low, hi, mid, hi]
        step = bar_ticks // 8
        for i in range(8):
            _ev(rh_seq, bar_start + i * step, step, pat[i % 4], 62)
        _ev(lh_seq, bar_start, bar_ticks, lh, 66)
    elif pattern == "waltz":
        _ev(lh_seq, bar_start, q, lh, 72)
        for beat in (1, 2):
            for p in rh:
                _ev(rh_seq, bar_start + beat * q, q, p, 60)
    elif pattern == "pop":
        _ev(lh_seq, bar_start, 2 * q, lh, 72)
        _ev(lh_seq, bar_start + 2 * q, 2 * q, fifth, 66)
        for beat_off in (q + q // 2, 3 * q):        # a contratiempo
            for p in rh:
                _ev(rh_seq, bar_start + beat_off, q, p, 64)


# ══════════════════════════════════════════════════════════════════════════════
#  META HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _meta_name(name):
    b = name.encode("latin-1", "replace")
    return MidiEvent(abs=0, kind="meta", meta_type=0x03,
                     data=bytes([0xFF, 0x03]) + _write_vlq(len(b)) + b)


def _meta_tempo(bpm):
    us = int(round(60_000_000 / bpm))
    return MidiEvent(abs=0, kind="meta", meta_type=0x51,
                     data=bytes([0xFF, 0x51]) + _write_vlq(3) + us.to_bytes(3, "big"))


def _meta_timesig(num, den):
    dd = int(round(math.log2(den)))
    return MidiEvent(abs=0, kind="meta", meta_type=0x58,
                     data=bytes([0xFF, 0x58]) + _write_vlq(4) + bytes([num, dd, 24, 8]))


def _prog(ch, program):
    return MidiEvent(abs=0, kind="channel", channel=ch, data=bytes([0xC0 | ch, program]))


def _to_events(seq, ch):
    evs = []
    for start, dur, pitch, vel in seq:
        evs.append(MidiEvent(abs=start, kind="note_on", channel=ch,
                             pitch=max(0, min(127, pitch)), vel=vel))
        evs.append(MidiEvent(abs=start + dur, kind="note_off", channel=ch,
                             pitch=max(0, min(127, pitch)), vel=0))
    return evs


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def realize_leadsheet(chords_text: str, pattern: str = "block", tempo: int = 96,
                      octave: int = 4, out_path: Optional[str] = None) -> dict:
    """API pública. Realiza el acompañamiento y escribe el MIDI. Devuelve stats."""
    if pattern not in ("block", "ballad", "arpeggio", "alberti", "waltz", "pop"):
        raise ValueError(f"patrón desconocido: {pattern}")
    prog = parse_progression(chords_text)
    if not prog:
        raise ValueError("no se proporcionaron acordes")

    ts_num = 3 if pattern == "waltz" else 4
    bar_ticks = TPB * 4 * ts_num // 4

    mid = MidiData(fmt=1, tpb=TPB)
    cond = MidiTrackData(name="leadsheet")
    cond.events += [_meta_name(f"leadsheet {pattern}"),
                    _meta_timesig(ts_num, 4), _meta_tempo(tempo)]
    mid.tracks.append(cond)

    rh_seq, lh_seq = [], []
    t = 0
    used = []
    for chord, bars in prog:
        rh, lh = voicing(chord, octave)
        fifth = lh + 7
        for _ in range(bars):
            render_bar(pattern, rh, lh, fifth, t, bar_ticks, rh_seq, lh_seq)
            t += bar_ticks
        used.append(chord["symbol"])

    rh_trk = MidiTrackData(name="Piano MD")
    rh_trk.events += [_meta_name("Piano MD"), _prog(0, 0)] + _to_events(rh_seq, 0)
    lh_trk = MidiTrackData(name="Piano MI")
    lh_trk.events += [_meta_name("Piano MI"), _prog(1, 0)] + _to_events(lh_seq, 1)
    mid.tracks += [rh_trk, lh_trk]

    out = out_path or "leadsheet.mid"
    write_midi(mid, out)
    return {"pattern": pattern, "tempo": tempo, "time_signature": f"{ts_num}/4",
            "n_chords": len(prog), "n_bars": t // bar_ticks,
            "chords": used, "rh_notes": len(rh_seq), "lh_notes": len(lh_seq),
            "output": out}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="leadsheet_realizer.py",
        description="Cifrado de acordes → acompañamiento pianístico tocable.")
    ap.add_argument("--chords")
    ap.add_argument("--file")
    ap.add_argument("--pattern", default="block",
                    choices=["block", "ballad", "arpeggio", "alberti", "waltz", "pop"])
    ap.add_argument("--tempo", type=int, default=96)
    ap.add_argument("--octave", type=int, default=4)
    ap.add_argument("--output")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    if not args.chords and not args.file:
        print("[ERROR] indica --chords \"C Am F G\" o --file cancion.txt", file=sys.stderr)
        return 1
    text = args.chords or Path(args.file).read_text(encoding="utf-8")

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    print(f"  {B}LEADSHEET REALIZER v{VERSION}  ·  patrón: {args.pattern}{R}")
    print(f"{'═' * 78}")
    try:
        st = realize_leadsheet(text, pattern=args.pattern, tempo=args.tempo,
                               octave=args.octave, out_path=args.output)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  compás: {st['time_signature']}   tempo: {st['tempo']} bpm   "
          f"acordes: {st['n_chords']}   compases: {st['n_bars']}")
    print(f"  progresión: {C}{' '.join(st['chords'])}{R}")
    print(f"  notas MD: {st['rh_notes']}   notas MI: {st['lh_notes']}")
    print(f"  {_c('GRN')}MIDI: {st['output']}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
