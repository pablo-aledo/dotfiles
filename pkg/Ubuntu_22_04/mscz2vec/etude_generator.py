#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ETUDE GENERATOR  v1.0                                  ║
║        Generador de ejercicios técnicos de piano (escalas, arpegios…)       ║
║                                                                              ║
║  El ecosistema compone obras, pero un estudiante necesita material TÉCNICO  ║
║  diario. Esta herramienta genera escalas, arpegios, patrones tipo Hanon y   ║
║  cadencias en cualquier tonalidad, a manos juntas o separadas, con la        ║
║  digitación estándar anotada. Complementa a fingering.py (que digita una    ║
║  pieza dada) generando el material desde cero.                               ║
║                                                                              ║
║  TIPOS (--type):                                                             ║
║    scale      — escala ascendente y descendente en N octavas                ║
║    arpeggio   — arpegio de tríada (fundamental) en N octavas                ║
║    broken     — acorde quebrado / patrón Alberti                            ║
║    hanon      — figura de 5 dedos que sube y baja por la escala (Hanon nº1) ║
║    cadence    — cadencia I–IV–I6/4–V–I con enlace de voces                   ║
║                                                                              ║
║  MODOS (--mode): major · minor · harmonic_minor · melodic_minor ·           ║
║                  dorian · mixolydian · phrygian · lydian                     ║
║                                                                              ║
║  USO:                                                                        ║
║    python etude_generator.py --type scale --key C --octaves 2               ║
║    python etude_generator.py --type scale --key A --mode harmonic_minor     ║
║    python etude_generator.py --type arpeggio --key G --hands right          ║
║    python etude_generator.py --type hanon --key C --tempo 100               ║
║    python etude_generator.py --type cadence --key F --output cadencia.mid   ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --type T          scale | arpeggio | broken | hanon | cadence            ║
║    --key K           Tónica (C, G, F#, Bb…)  (default: C)                    ║
║    --mode M          Modo/escala (default: major)                           ║
║    --octaves N       Octavas (scale/arpeggio/broken)  (default: 2)          ║
║    --hands H         both | right | left  (default: both)                   ║
║    --tempo BPM       Tempo del MIDI  (default: 90)                          ║
║    --note-value V    Valor por nota en negras (default: 0.25 = semicorchea) ║
║    --start-octave N  Octava de inicio de la mano derecha (default: 4)       ║
║    --output FILE     MIDI de salida (default: <tipo>_<tónica><modo>.mid)     ║
║    --no-color        Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from etude_generator import generate_etude                                ║
║                                                                              ║
║  DEPENDENCIAS: playability_auditor.py (mismo directorio, E/S MIDI)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
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

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor": [0, 2, 3, 5, 7, 9, 11],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}
_BLACK_PC = {1, 3, 6, 8, 10}


def _is_black(pc: int) -> bool:
    return pc % 12 in _BLACK_PC


def _octave_fingering(deg_pcs: List[int], hand: str) -> List[int]:
    """Digitación de una escala de 7 grados (pitch-classes) para una mano.

    Regla clásica general (no memoriza una tabla de 12 tonalidades, que
    solo cubriría escalas mayores): el pulgar (1) y el meñique (5) NUNCA
    se colocan sobre una tecla negra; los dedos 2/3/4 sí pueden. La mano
    se divide en dos grupos de dedos consecutivos con un paso de pulgar
    entre ellos; el punto de paso se desplaza para que el pulgar caiga
    siempre en tecla blanca. Reproduce exactamente las tablas estándar
    de Do mayor (1 2 3 1 2 3 4) y Fa mayor (1 2 3 4 1 2 3), y generaliza
    correctamente a cualquier tonalidad o modo, incluida una tónica en
    tecla negra (p.ej. Solb mayor: 2 3 4 1 2 3 4).
    """
    n = len(deg_pcs)
    if hand == "right":
        start = 2 if _is_black(deg_pcs[0]) else 1
        group1_len = 3
        for cand in (3, 4, 2):
            boundary = cand
            if boundary >= n or not _is_black(deg_pcs[boundary]):
                group1_len = min(cand, n)
                break
        fingers = [start + i for i in range(group1_len)]
        group2_len = n - group1_len
        fingers += [1 + i for i in range(group2_len)]
    else:
        start = 4 if _is_black(deg_pcs[0]) else 5
        group1_len = 4
        for cand in (4, 3, 5):
            boundary = cand
            if boundary >= n or not _is_black(deg_pcs[boundary]):
                group1_len = min(cand, n)
                break
        fingers = [start - i for i in range(group1_len)]
        group2_len = n - group1_len
        if group2_len == 3:
            fingers += [1, 3, 2]                    # convención LH verificada (Do/Fa mayor)
        else:
            fingers += [1 + i for i in range(group2_len)]
    return fingers


def _scale_fingering(tonic_pc: int, mode: str, octaves: int, hand: str) -> List[int]:
    """Digitación completa (ida y vuelta) para generate_etude tipo 'scale'."""
    degs = SCALES[mode]
    deg_pcs = [(tonic_pc + d) % 12 for d in degs]
    one_octave = _octave_fingering(deg_pcs, hand)
    up = list(one_octave) * octaves
    up.append(one_octave[0])                        # tónica superior final
    down = up[::-1][1:]
    return up + down


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def parse_key(s: str) -> int:
    pc = _PC[s[0].upper()]
    for ch in s[1:]:
        if ch in "#♯s":
            pc += 1
        elif ch in "b♭f":
            pc -= 1
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


def _meta_timesig(num, den):
    dd = int(round(math.log2(den)))
    return MidiEvent(abs=0, kind="meta", meta_type=0x58,
                     data=bytes([0xFF, 0x58]) + _write_vlq(4) + bytes([num, dd, 24, 8]))


def _prog(ch, program):
    return MidiEvent(abs=0, kind="channel", channel=ch, data=bytes([0xC0 | ch, program]))


def _notes_to_events(seq: List[Tuple[int, int, int, int]], ch: int) -> List[MidiEvent]:
    """seq: (start_tick, dur_tick, pitch, vel)."""
    evs = []
    for start, dur, pitch, vel in seq:
        evs.append(MidiEvent(abs=start, kind="note_on", channel=ch,
                             pitch=pitch, vel=vel))
        evs.append(MidiEvent(abs=start + dur, kind="note_off", channel=ch,
                             pitch=pitch, vel=0))
    return evs


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES DE SECUENCIA (devuelven listas de pitch y digitación)
# ══════════════════════════════════════════════════════════════════════════════

def _scale_pitches(tonic_pc, mode, octaves, base_oct):
    degs = SCALES[mode]
    root = (base_oct + 1) * 12 + tonic_pc
    up = []
    for o in range(octaves):
        for d in degs:
            up.append(root + 12 * o + d)
    up.append(root + 12 * octaves)                 # tónica superior
    down = up[::-1][1:]
    return up + down


def _arpeggio_pitches(tonic_pc, mode, octaves, base_oct):
    third = 3 if mode in ("minor", "harmonic_minor", "melodic_minor", "dorian",
                          "phrygian") else 4
    triad = [0, third, 7]
    root = (base_oct + 1) * 12 + tonic_pc
    up = []
    for o in range(octaves):
        for d in triad:
            up.append(root + 12 * o + d)
    up.append(root + 12 * octaves)
    down = up[::-1][1:]
    return up + down


def _broken_pitches(tonic_pc, mode, octaves, base_oct):
    third = 3 if mode in ("minor", "harmonic_minor", "melodic_minor", "dorian",
                          "phrygian") else 4
    root = (base_oct + 1) * 12 + tonic_pc
    triad = [0, third, 7, 12]
    # patrón Alberti: bajo-alto-medio-alto por octava
    pat = [triad[0], triad[2], triad[1], triad[2]]
    seq = []
    for o in range(octaves):
        for p in pat * 2:
            seq.append(root + 12 * o + p)
    return seq


def _hanon_pitches(tonic_pc, mode, base_oct):
    """Hanon nº1 simplificado: figura de 5 notas que sube por grados de la escala."""
    degs = SCALES[mode]
    root = (base_oct + 1) * 12 + tonic_pc
    scale = [root + 12 * o + d for o in range(2) for d in degs] + [root + 24]
    fig = [0, 2, 3, 4, 3, 2, 1, 2]                 # patrón relativo de índices
    seq = []
    for start in range(0, 8):                       # 8 posiciones ascendentes
        for k in fig:
            idx = min(start + k, len(scale) - 1)
            seq.append(scale[idx])
    # descenso simétrico
    seq += seq[::-1]
    return seq


def _cadence(tonic_pc, base_oct):
    """I – IV – I6/4 – V – I a cuatro voces (RH tríada, LH fundamental)."""
    root = (base_oct + 1) * 12 + tonic_pc
    def triad(r, quality="maj"):
        third = 4 if quality == "maj" else 3
        return [r, r + third, r + 7]
    chords_rh = [triad(root), triad(root + 5), triad(root),
                 triad(root + 7), triad(root)]
    bass = [root - 12, root - 7, root - 12, root - 5, root - 12]
    return chords_rh, bass


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def generate_etude(etype: str, key: str = "C", mode: str = "major",
                   octaves: int = 2, hands: str = "both", tempo: int = 90,
                   note_value: float = 0.25, start_octave: int = 4,
                   out_path: Optional[str] = None) -> dict:
    """API pública. Genera el ejercicio y escribe el MIDI. Devuelve stats."""
    if mode not in SCALES:
        raise ValueError(f"modo desconocido: {mode} (usa uno de {list(SCALES)})")
    if etype not in ("scale", "arpeggio", "broken", "hanon", "cadence"):
        raise ValueError(f"tipo desconocido: {etype}")
    tonic_pc = parse_key(key)
    dur = max(1, int(round(note_value * TPB)))

    mid = MidiData(fmt=1, tpb=TPB)
    cond = MidiTrackData(name=f"{etype} {key} {mode}")
    cond.events += [_meta_name(f"{etype} {key} {mode}"),
                    _meta_timesig(4, 4), _meta_tempo(tempo)]
    mid.tracks.append(cond)

    fingering = {"right": [], "left": []}
    n_notes = 0

    if etype == "cadence":
        chords_rh, bass = _cadence(tonic_pc, start_octave)
        rh = MidiTrackData(name="RH")
        rh.events += [_meta_name("RH"), _prog(0, 0)]
        lh = MidiTrackData(name="LH")
        lh.events += [_meta_name("LH"), _prog(1, 0)]
        cd = TPB * 2
        rh_seq, lh_seq = [], []
        for i, (ch, b) in enumerate(zip(chords_rh, bass)):
            t = i * cd
            for p in ch:
                rh_seq.append((t, cd, p, 72))
            lh_seq.append((t, cd, b, 68))
        rh.events += _notes_to_events(rh_seq, 0)
        lh.events += _notes_to_events(lh_seq, 1)
        if hands in ("both", "right"):
            mid.tracks.append(rh)
        if hands in ("both", "left"):
            mid.tracks.append(lh)
        n_notes = len(rh_seq) + len(lh_seq)
        fingering["right"] = ["1-3-5"] * len(chords_rh)
        fingering["left"] = ["5"] * len(bass)
    else:
        if etype == "scale":
            rh_p = _scale_pitches(tonic_pc, mode, octaves, start_octave)
            n_up = octaves * 7 + 1
            fingering["right"] = _scale_fingering(tonic_pc, mode, octaves, "right")
            fingering["left"] = _scale_fingering(tonic_pc, mode, octaves, "left")
        elif etype == "arpeggio":
            rh_p = _arpeggio_pitches(tonic_pc, mode, octaves, start_octave)
            fingering["right"] = [[1, 2, 3, 5][i % 4] for i in range(len(rh_p))]
            fingering["left"] = [[5, 3, 2, 1][i % 4] for i in range(len(rh_p))]
        elif etype == "broken":
            rh_p = _broken_pitches(tonic_pc, mode, octaves, start_octave)
        else:  # hanon
            rh_p = _hanon_pitches(tonic_pc, mode, start_octave)

        rh = MidiTrackData(name="RH")
        rh.events += [_meta_name("RH"), _prog(0, 0)]
        lh = MidiTrackData(name="LH")
        lh.events += [_meta_name("LH"), _prog(1, 0)]
        rh_seq = [(i * dur, dur, p, 74) for i, p in enumerate(rh_p)]
        lh_seq = [(i * dur, dur, p - 12, 66) for i, p in enumerate(rh_p)]
        rh.events += _notes_to_events(rh_seq, 0)
        lh.events += _notes_to_events(lh_seq, 1)
        if hands in ("both", "right"):
            mid.tracks.append(rh)
        if hands in ("both", "left"):
            mid.tracks.append(lh)
        n_notes = len(rh_seq) * (2 if hands == "both" else 1)

    out = out_path or f"{etype}_{key}{'_' + mode if mode != 'major' else ''}.mid"
    write_midi(mid, out)
    return {"type": etype, "key": key, "mode": mode, "octaves": octaves,
            "hands": hands, "tempo": tempo, "n_notes": n_notes,
            "fingering": fingering, "output": out}


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="etude_generator.py",
        description="Generador de ejercicios técnicos de piano.")
    ap.add_argument("--type", required=True,
                    choices=["scale", "arpeggio", "broken", "hanon", "cadence"])
    ap.add_argument("--key", default="C")
    ap.add_argument("--mode", default="major")
    ap.add_argument("--octaves", type=int, default=2)
    ap.add_argument("--hands", choices=["both", "right", "left"], default="both")
    ap.add_argument("--tempo", type=int, default=90)
    ap.add_argument("--note-value", type=float, default=0.25)
    ap.add_argument("--start-octave", type=int, default=4)
    ap.add_argument("--output")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    print(f"  {B}ETUDE GENERATOR v{VERSION}  ·  {args.type} · {args.key} {args.mode}{R}")
    print(f"{'═' * 78}")
    try:
        st = generate_etude(
            args.type, key=args.key, mode=args.mode, octaves=args.octaves,
            hands=args.hands, tempo=args.tempo, note_value=args.note_value,
            start_octave=args.start_octave, out_path=args.output)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  manos: {st['hands']}   octavas: {st['octaves']}   "
          f"tempo: {st['tempo']} bpm   notas: {st['n_notes']}")
    fr = st["fingering"]["right"]
    if fr and st["hands"] in ("both", "right"):
        show = fr[:16]
        txt = " ".join(str(x) for x in show)
        print(f"  {G}digitación MD (inicio): {txt}"
              f"{' …' if len(fr) > 16 else ''}{R}")
    fl = st["fingering"]["left"]
    if fl and st["hands"] in ("both", "left"):
        show = fl[:16]
        txt = " ".join(str(x) for x in show)
        print(f"  {G}digitación MI (inicio): {txt}"
              f"{' …' if len(fl) > 16 else ''}{R}")
    print(f"  {_c('GRN')}MIDI: {st['output']}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
