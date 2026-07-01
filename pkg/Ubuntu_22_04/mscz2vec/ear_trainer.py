#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        EAR TRAINER  v1.0                                    ║
║        Generador de dictados de entrenamiento auditivo (con solucionario)   ║
║                                                                              ║
║  El oído es la mitad de aprender a componer y a tocar, y el ecosistema no   ║
║  tenía nada para entrenarlo. Esta herramienta genera ejercicios de          ║
║  dictado como MIDI reproducible + un solucionario aparte: intervalos,        ║
║  calidades de acorde y progresiones cortas. Reproduce el MIDI, apunta tu    ║
║  respuesta y comprueba con el solucionario.                                  ║
║                                                                              ║
║  TIPOS (--type):                                                             ║
║    intervals      — dos notas (melódicas u armónicas); nombra el intervalo  ║
║    chords         — un acorde; nombra la calidad (M, m, dim, aug, 7…)       ║
║    progressions   — 2–4 acordes en una tonalidad; nombra los grados          ║
║                                                                              ║
║  USO:                                                                        ║
║    python ear_trainer.py --type intervals --count 12                        ║
║    python ear_trainer.py --type intervals --harmonic --seed 7               ║
║    python ear_trainer.py --type chords --count 10                           ║
║    python ear_trainer.py --type progressions --key G --count 8              ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --type T          intervals | chords | progressions                      ║
║    --count N         Número de ejercicios (default: 10)                     ║
║    --key K           Tonalidad de las progresiones (default: C)             ║
║    --harmonic        Intervalos/acordes simultáneos (default: melódicos)    ║
║    --level L         easy | medium | hard  (amplía el repertorio)           ║
║    --seed N          Semilla aleatoria para reproducir el mismo dictado      ║
║    --tempo BPM       Tempo del MIDI (default: 80)                           ║
║    --output FILE     MIDI de salida (default: eartrain_<tipo>.mid)          ║
║    --answers FILE    Guarda el solucionario en un fichero de texto           ║
║    --json FILE       Guarda el solucionario en JSON                          ║
║    --no-color        Desactivar colores ANSI                               ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from ear_trainer import generate_dictation                                ║
║                                                                              ║
║  DEPENDENCIAS: playability_auditor.py (mismo directorio, E/S MIDI)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import math
import random
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

INTERVAL_NAMES = {0: "unísono", 1: "2ª menor", 2: "2ª mayor", 3: "3ª menor",
                  4: "3ª mayor", 5: "4ª justa", 6: "tritono", 7: "5ª justa",
                  8: "6ª menor", 9: "6ª mayor", 10: "7ª menor", 11: "7ª mayor",
                  12: "8ª justa"}
INTERVALS_BY_LEVEL = {
    "easy": [2, 4, 5, 7, 12],
    "medium": [1, 2, 3, 4, 5, 7, 9, 12],
    "hard": list(range(1, 13)),
}
CHORD_QUALITIES = {
    "easy": [("mayor", [0, 4, 7]), ("menor", [0, 3, 7])],
    "medium": [("mayor", [0, 4, 7]), ("menor", [0, 3, 7]),
               ("disminuido", [0, 3, 6]), ("aumentado", [0, 4, 8])],
    "hard": [("mayor", [0, 4, 7]), ("menor", [0, 3, 7]),
             ("disminuido", [0, 3, 6]), ("aumentado", [0, 4, 8]),
             ("mayor 7", [0, 4, 7, 11]), ("dominante 7", [0, 4, 7, 10]),
             ("menor 7", [0, 3, 7, 10]), ("semidism. m7b5", [0, 3, 6, 10])],
}
# progresiones por grados (numeral, offset_semitonos, calidad)
PROG_BANK = {
    "easy": [
        [("I", 0, "maj"), ("IV", 5, "maj"), ("V", 7, "maj"), ("I", 0, "maj")],
        [("I", 0, "maj"), ("V", 7, "maj"), ("I", 0, "maj")],
        [("I", 0, "maj"), ("vi", 9, "min"), ("IV", 5, "maj"), ("V", 7, "maj")],
    ],
    "medium": [
        [("ii", 2, "min"), ("V", 7, "maj"), ("I", 0, "maj")],
        [("I", 0, "maj"), ("V", 7, "maj"), ("vi", 9, "min"), ("IV", 5, "maj")],
        [("vi", 9, "min"), ("IV", 5, "maj"), ("I", 0, "maj"), ("V", 7, "maj")],
    ],
    "hard": [
        [("I", 0, "maj"), ("vi", 9, "min"), ("ii", 2, "min"), ("V", 7, "maj")],
        [("iii", 4, "min"), ("vi", 9, "min"), ("ii", 2, "min"), ("V", 7, "maj")],
        [("I", 0, "maj"), ("V/vi", 4, "maj"), ("vi", 9, "min"), ("IV", 5, "maj")],
    ],
}


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def parse_key(s: str) -> int:
    pc = _PC[s[0].upper()]
    for ch in s[1:]:
        pc += 1 if ch in "#♯s" else -1 if ch in "b♭f" else 0
    return pc % 12


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


def _prog_ch(ch, program):
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

def generate_dictation(etype: str, count: int = 10, key: str = "C",
                       harmonic: bool = False, level: str = "medium",
                       seed: Optional[int] = None, tempo: int = 80,
                       out_path: Optional[str] = None) -> dict:
    """API pública. Genera el MIDI de dictado + el solucionario. Devuelve stats."""
    if etype not in ("intervals", "chords", "progressions"):
        raise ValueError(f"tipo desconocido: {etype}")
    if level not in ("easy", "medium", "hard"):
        raise ValueError(f"nivel desconocido: {level}")
    rng = random.Random(seed)

    mid = MidiData(fmt=1, tpb=TPB)
    cond = MidiTrackData(name=f"eartrain {etype}")
    cond.events += [_meta_name(f"eartrain {etype} {level}"), _meta_tempo(tempo)]
    mid.tracks.append(cond)
    trk = MidiTrackData(name="dictado")
    trk.events.append(_prog_ch(0, 0))

    seq: List[Tuple[int, int, int, int]] = []
    answers = []
    q = TPB
    gap = TPB * 2                                   # silencio entre ejercicios
    t = 0
    base = 60                                        # C4 de referencia

    if etype == "intervals":
        pool = INTERVALS_BY_LEVEL[level]
        for i in range(count):
            iv = rng.choice(pool)
            low = base + rng.randint(-5, 5)
            high = low + iv
            if harmonic:
                seq += [(t, 2 * q, low, 76), (t, 2 * q, high, 76)]
            else:
                seq += [(t, q, low, 76), (t + q, q, high, 76)]
            answers.append({"n": i + 1, "answer": INTERVAL_NAMES[iv],
                            "detail": f"{pitch_name(low)}→{pitch_name(high)}",
                            "semitones": iv})
            t += 2 * q + gap
    elif etype == "chords":
        pool = CHORD_QUALITIES[level]
        for i in range(count):
            name, intervals = rng.choice(pool)
            root = base + rng.randint(-4, 4)
            notes = [root + iv for iv in intervals]
            if harmonic:
                for p in notes:
                    seq.append((t, 2 * q, p, 74))
            else:
                for k, p in enumerate(notes):
                    seq.append((t + k * (q // 2), q // 2, p, 74))
                # y sonando junto al final
                for p in notes:
                    seq.append((t + len(notes) * (q // 2), q, p, 70))
            answers.append({"n": i + 1, "answer": name,
                            "detail": pitch_name(root) + " " + name})
            t += 2 * q + gap
    else:  # progressions
        tonic = parse_key(key)
        bank = PROG_BANK[level]
        for i in range(count):
            prog = rng.choice(bank)
            romans = []
            for (num, off, qual) in prog:
                root = base + (tonic + off) % 12
                iv = [0, 4, 7] if qual == "maj" else [0, 3, 7]
                for p in [root + x for x in iv]:
                    seq.append((t, q, p, 72))
                # bajo
                seq.append((t, q, root - 12, 66))
                t += q
                romans.append(num)
            answers.append({"n": i + 1, "answer": " – ".join(romans),
                            "detail": f"en {key}"})
            t += gap

    trk.events += _to_events(seq, 0)
    mid.tracks.append(trk)
    out = out_path or f"eartrain_{etype}.mid"
    write_midi(mid, out)
    return {"type": etype, "level": level, "count": count, "key": key,
            "harmonic": harmonic, "tempo": tempo, "seed": seed,
            "n_notes": len(seq), "answers": answers, "output": out}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="ear_trainer.py",
        description="Generador de dictados de entrenamiento auditivo.")
    ap.add_argument("--type", required=True,
                    choices=["intervals", "chords", "progressions"])
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--key", default="C")
    ap.add_argument("--harmonic", action="store_true")
    ap.add_argument("--level", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--tempo", type=int, default=80)
    ap.add_argument("--output")
    ap.add_argument("--answers")
    ap.add_argument("--json")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    print(f"  {B}EAR TRAINER v{VERSION}  ·  {args.type} · nivel {args.level}{R}")
    print(f"{'═' * 78}")
    try:
        st = generate_dictation(
            args.type, count=args.count, key=args.key, harmonic=args.harmonic,
            level=args.level, seed=args.seed, tempo=args.tempo, out_path=args.output)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    mode = "armónico" if st["harmonic"] else "melódico"
    print(f"  ejercicios: {st['count']}   modo: {mode}   "
          f"tempo: {st['tempo']} bpm" + (f"   semilla: {st['seed']}"
          if st["seed"] is not None else ""))
    print(f"  {_c('GRN')}MIDI de dictado: {st['output']}{R}")
    print(f"  {G}(reprodúcelo, apunta tus respuestas y compara abajo){R}")

    # solucionario
    lines = [f"SOLUCIONARIO — {st['type']} (nivel {st['level']})"]
    for a in st["answers"]:
        lines.append(f"  {a['n']:>2}. {a['answer']}"
                     + (f"   [{a['detail']}]" if a.get("detail") else ""))
    key_txt = "\n".join(lines)

    print(f"\n  {B}Solucionario:{R}")
    for a in st["answers"]:
        print(f"    {a['n']:>2}. {C}{a['answer']}{R}"
              + (f"  {G}[{a['detail']}]{R}" if a.get("detail") else ""))

    if args.answers:
        Path(args.answers).write_text(key_txt + "\n", encoding="utf-8")
        print(f"\n  solucionario: {args.answers}")
    if args.json:
        Path(args.json).write_text(
            json.dumps({"version": VERSION, **st}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"  JSON: {args.json}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
