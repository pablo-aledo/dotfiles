#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       PART EXTRACTOR  v1.0                                  ║
║        Extracción de partes instrumentales individuales (con transposición) ║
║                                                                              ║
║  El ecosistema genera MIDIs orquestales multipista (orchestrator,           ║
║  piano_to_orchestra, piano_expander, ml_expander) y audita su tocabilidad   ║
║  (playability_auditor), pero ninguna herramienta produce las PARTES         ║
║  individuales que necesita cada intérprete. Esta cubre ese hueco: separa    ║
║  cada pista en su propio MIDI autónomo (con el tempo/compás copiados),      ║
║  aplica la transposición del instrumento transpositor (clarinete/trompeta   ║
║  en Sib, trompa/corno inglés en Fa, flautín 8ª, contrabajo 8ª…) para que    ║
║  el intérprete lea su parte escrita, y reporta rango y compases de tacet.   ║
║                                                                              ║
║  A diferencia de piano_reducer (que fusiona voces en dos pentagramas) y de  ║
║  midi_merge (que combina stems en uno), esta herramienta hace lo inverso:   ║
║  DIVIDE una partitura en partes y las prepara para lectura instrumental.    ║
║                                                                              ║
║  TRANSPOSICIONES POR DEFECTO (escrito = sonante + semitonos):               ║
║    trompeta +2  ·  clarinete +2  ·  saxofón +2  (Sib)                        ║
║    trompa +7  ·  corno inglés +7  (Fa)                                       ║
║    flautín −12  (escrito 8ª baja)  ·  contrabajo +12  (escrito 8ª alta)     ║
║    resto (flauta, oboe, cuerdas, piano, trombón, tuba…) → 0 (en Do)         ║
║  Se pueden anular con --transpose o forzar partitura en Do con --concert.   ║
║                                                                              ║
║  USO:                                                                        ║
║    python part_extractor.py orquesta.mid                                     ║
║    python part_extractor.py orquesta.mid --list                             ║
║    python part_extractor.py orquesta.mid --concert                          ║
║    python part_extractor.py orquesta.mid --transpose 3:9 --as 5:trompeta     ║
║    python part_extractor.py orquesta.mid --outdir partes/ --json p.json      ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi              MIDI multipista de entrada                             ║
║    --outdir DIR      Carpeta de salida (default: junto al MIDI de entrada)  ║
║    --list            Solo detecta y lista partes, sin escribir MIDI         ║
║    --concert         No transponer (partitura en Do, sonante = escrito)     ║
║    --transpose T:S   Anula la transposición de la pista T a S semitonos     ║
║                      (repetible; escrito = sonante + S)                     ║
║    --as T:INSTR      Fuerza el instrumento de la pista T (clave del catálogo)║
║    --min-notes N     Ignora pistas con menos de N notas (default: 1)        ║
║    --json FILE       Escribe un sidecar JSON con el resumen de partes        ║
║    --no-color        Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from part_extractor import extract_parts                                  ║
║                                                                              ║
║  DEPENDENCIAS: playability_auditor.py (mismo directorio, E/S + catálogo)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

try:
    from playability_auditor import (read_midi, write_midi, MidiData,
                                     MidiTrackData, MidiEvent, TimeContext,
                                     extract_notes, detect_instrument,
                                     pitch_name, INSTRUMENT_DB, _write_vlq)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


# Transposición por defecto: escrito = sonante + offset (semitonos)
DEFAULT_TRANSPOSE = {
    "trompeta": 2, "clarinete": 2, "saxofon": 2,     # instrumentos en Sib
    "trompa": 7, "corno_ingles": 7,                  # en Fa
    "flautin": -12,                                  # escrito una 8ª baja
    "contrabajo": 12,                                # escrito una 8ª alta
}
_TRANSPOSE_LABEL = {2: "Sib", 7: "Fa", 9: "Mib", 3: "La", -12: "8ª↓",
                    12: "8ª↑", 14: "Sib (9ª)", 0: "Do"}


# ══════════════════════════════════════════════════════════════════════════════
#  META HELPERS (tempo / compás / nombre) — para partes autónomas
# ══════════════════════════════════════════════════════════════════════════════

def _meta_name(name: str) -> MidiEvent:
    b = name.encode("latin-1", "replace")
    return MidiEvent(abs=0, kind="meta", meta_type=0x03,
                     data=bytes([0xFF, 0x03]) + _write_vlq(len(b)) + b)


def _meta_tempo(tick: int, us: int) -> MidiEvent:
    return MidiEvent(abs=tick, kind="meta", meta_type=0x51,
                     data=bytes([0xFF, 0x51]) + _write_vlq(3) + us.to_bytes(3, "big"))


def _meta_timesig(tick: int, num: int, den: int) -> MidiEvent:
    import math
    dd = int(round(math.log2(max(1, den))))
    return MidiEvent(abs=tick, kind="meta", meta_type=0x58,
                     data=bytes([0xFF, 0x58]) + _write_vlq(4) + bytes([num, dd, 24, 8]))


def _conductor_track(mid: MidiData) -> MidiTrackData:
    """Pista de dirección con tempo y compás copiados, para partes autónomas."""
    trk = MidiTrackData(name="conductor")
    trk.events.append(_meta_name("conductor"))
    for tick, num, den in mid.timesig_map:
        trk.events.append(_meta_timesig(tick, num, den))
    for tick, us in mid.tempo_map:
        trk.events.append(_meta_tempo(tick, us))
    trk.events.sort(key=lambda e: e.abs)
    return trk


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSPOSICIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _transpose_track(src: MidiTrackData, offset: int) -> MidiTrackData:
    """Copia una pista transponiendo pitch de note_on/off (clamp 0..127)."""
    out = MidiTrackData(name=src.name, programs=list(src.programs),
                        channels=list(src.channels))
    for ev in src.events:
        if ev.kind in ("note_on", "note_off"):
            out.events.append(MidiEvent(
                abs=ev.abs, kind=ev.kind, channel=ev.channel,
                pitch=max(0, min(127, ev.pitch + offset)), vel=ev.vel))
        else:
            out.events.append(MidiEvent(
                abs=ev.abs, kind=ev.kind, channel=ev.channel,
                data=ev.data, pitch=ev.pitch, vel=ev.vel, meta_type=ev.meta_type))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE UNA PARTE
# ══════════════════════════════════════════════════════════════════════════════

def _tacet_bars(notes, tc: TimeContext, n_bars: int) -> Tuple[int, int]:
    """Compases sin ninguna nota (tacet) y la carrera de silencio más larga."""
    active = set()
    for n in notes:
        b0 = tc.bar(n.start)
        b1 = tc.bar(max(n.start, n.end - 1))
        for b in range(b0, b1 + 1):
            active.add(b)
    tacet, longest, run = 0, 0, 0
    for b in range(1, n_bars + 1):
        if b not in active:
            tacet += 1
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return tacet, longest


def _range_status(sounding_lo: int, sounding_hi: int, inst: str) -> str:
    prof = INSTRUMENT_DB.get(inst)
    if not prof:
        return "desconocido"
    lo, hi = prof["range"]
    slo, shi = prof["sweet"]
    if sounding_lo < lo or sounding_hi > hi:
        return "FUERA DE RANGO"
    if sounding_lo < slo or sounding_hi > shi:
        return "fuera de tesitura"
    return "ok"


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def extract_parts(path: str, outdir: Optional[str] = None,
                  concert: bool = False,
                  transpose_overrides: Optional[Dict[int, int]] = None,
                  instrument_overrides: Optional[Dict[int, str]] = None,
                  min_notes: int = 1, list_only: bool = False,
                  write_json: bool = True) -> dict:
    """API pública. Extrae las partes. Devuelve un resumen con una entrada por parte."""
    mid = read_midi(path)
    tc = TimeContext(mid)
    transpose_overrides = transpose_overrides or {}
    instrument_overrides = instrument_overrides or {}

    last = max((ev.abs for trk in mid.tracks for ev in trk.events
                if ev.kind == "note_off"), default=0)
    n_bars = max(1, tc.bar(max(0, last - 1)))

    stem = Path(path).stem
    base_dir = Path(outdir) if outdir else Path(path).parent
    if not list_only:
        base_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    part_no = 0
    for tidx, trk in enumerate(mid.tracks):
        notes = extract_notes(trk)
        if len(notes) < min_notes:
            continue
        part_no += 1
        override_inst = instrument_overrides.get(tidx)
        inst = detect_instrument(trk, override=override_inst)
        inst_es = INSTRUMENT_DB.get(inst, {}).get("es", inst)

        if concert:
            offset = 0
        elif tidx in transpose_overrides:
            offset = transpose_overrides[tidx]
        else:
            offset = DEFAULT_TRANSPOSE.get(inst, 0)

        sounding_lo = min(n.pitch for n in notes)
        sounding_hi = max(n.pitch for n in notes)
        written_lo = max(0, min(127, sounding_lo + offset))
        written_hi = max(0, min(127, sounding_hi + offset))
        status = _range_status(sounding_lo, sounding_hi, inst)
        tacet, longest = _tacet_bars(notes, tc, n_bars)

        part_name = trk.name or f"pista_{tidx}"
        safe = "".join(ch if ch.isalnum() else "_" for ch in part_name).strip("_")
        fname = f"{stem}.part_{part_no:02d}_{safe or inst}.mid"
        fpath = str(base_dir / fname)

        entry = {
            "part": part_no, "source_track": tidx, "name": part_name,
            "instrument": inst, "instrument_es": inst_es,
            "notes": len(notes),
            "transpose": offset,
            "transpose_label": _TRANSPOSE_LABEL.get(offset, f"{offset:+d}st"),
            "sounding_range": [pitch_name(sounding_lo), pitch_name(sounding_hi)],
            "written_range": [pitch_name(written_lo), pitch_name(written_hi)],
            "range_status": status,
            "tacet_bars": tacet, "longest_rest_bars": longest,
            "file": None if list_only else fpath,
        }

        if not list_only:
            part_mid = MidiData(fmt=1, tpb=mid.tpb,
                                tempo_map=list(mid.tempo_map),
                                timesig_map=list(mid.timesig_map))
            part_mid.tracks.append(_conductor_track(mid))
            part_mid.tracks.append(_transpose_track(trk, offset))
            write_midi(part_mid, fpath)

        parts.append(entry)

    summary = {"file": path, "n_bars": int(n_bars),
               "concert": concert, "parts": parts,
               "n_parts": len(parts)}

    if write_json and not list_only:
        sidecar = str(base_dir / f"{stem}.parts.json")
        Path(sidecar).write_text(
            json.dumps({"version": VERSION, **summary}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        summary["sidecar"] = sidecar
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_kv_int(items: List[str]) -> Dict[int, int]:
    out = {}
    for s in items:
        k, v = s.split(":")
        out[int(k)] = int(v)
    return out


def _parse_kv_str(items: List[str]) -> Dict[int, str]:
    out = {}
    for s in items:
        k, v = s.split(":")
        if v not in INSTRUMENT_DB:
            raise ValueError(f"instrumento desconocido: {v}")
        out[int(k)] = v
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="part_extractor.py",
        description="Extracción de partes instrumentales con transposición.")
    ap.add_argument("midi")
    ap.add_argument("--outdir")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--concert", action="store_true")
    ap.add_argument("--transpose", action="append", default=[], metavar="T:S")
    ap.add_argument("--as", dest="as_inst", action="append", default=[], metavar="T:INSTR")
    ap.add_argument("--min-notes", type=int, default=1)
    ap.add_argument("--json")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    try:
        toffs = _parse_kv_int(args.transpose)
        insts = _parse_kv_str(args.as_inst)
    except ValueError as e:
        print(f"[ERROR] especificación inválida: {e}", file=sys.stderr)
        return 1

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    mode = "LISTA" if args.list else ("CONCERT (en Do)" if args.concert else "partes")
    print(f"  {B}PART EXTRACTOR v{VERSION}  ·  {args.midi}  [{mode}]{R}")
    print(f"{'═' * 78}")

    try:
        summary = extract_parts(
            args.midi, outdir=args.outdir, concert=args.concert,
            transpose_overrides=toffs, instrument_overrides=insts,
            min_notes=args.min_notes, list_only=args.list,
            write_json=bool(args.json) or not args.list)
        if args.json and not args.list:
            # reescribe el sidecar en la ruta pedida por el usuario
            Path(args.json).write_text(
                json.dumps({"version": VERSION, **summary}, ensure_ascii=False, indent=2),
                encoding="utf-8")
            summary["sidecar"] = args.json
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    if not summary["parts"]:
        print("  (no se encontraron pistas con notas)")
        print(f"{'═' * 78}\n")
        return 0

    print(f"  Compases: {summary['n_bars']}   partes detectadas: {summary['n_parts']}\n")
    print(f"  {G}# pista  instrumento         transp.  escrito         "
          f"sonante         rango  tacet{R}")
    for p in summary["parts"]:
        st = p["range_status"]
        col = (_c("RED") if st == "FUERA DE RANGO" else
               _c("YEL") if st == "fuera de tesitura" else _c("GRN"))
        wr = f"{p['written_range'][0]}–{p['written_range'][1]}"
        sr = f"{p['sounding_range'][0]}–{p['sounding_range'][1]}"
        print(f"  {p['part']:>1} {p['source_track']:>4}  "
              f"{p['instrument_es'][:18]:<18} {p['transpose_label']:>6}  "
              f"{wr:<14} {sr:<14} {col}{st[:6]:>6}{R}  "
              f"{p['tacet_bars']:>2}c")
    if args.list:
        print(f"\n  {G}(--list: no se escribió ningún MIDI){R}")
    else:
        print(f"\n  {_c('GRN')}{summary['n_parts']} partes escritas en "
              f"{args.outdir or Path(args.midi).parent}/{R}")
        if "sidecar" in summary:
            print(f"  sidecar: {summary['sidecar']}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
