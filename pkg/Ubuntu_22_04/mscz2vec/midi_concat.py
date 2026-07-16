#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║                          MIDI CONCAT  v1.0                                       ║
║         Concatena varios MIDIs uno detrás de otro (secuencialmente)              ║
║                                                                                  ║
║  Toma N MIDIs de entrada y genera un único MIDI en el que el contenido           ║
║  de cada uno se reproduce a continuación del anterior (no simultáneo,            ║
║  como hace midi_merge.py, sino en serie). Útil para encadenar frases,            ║
║  variaciones o secciones generadas por separado en una sola pieza.               ║
║                                                                                  ║
║  ORDEN:                                                                          ║
║    · Los MIDIs se concatenan en el orden en que se pasan por línea de            ║
║      comandos                                                                    ║
║                                                                                  ║
║  GESTIÓN DE TIEMPO:                                                              ║
║    · Por defecto: usa el TPB máximo de todos los MIDIs de entrada                ║
║    · Con --tpb N: usa el valor indicado                                          ║
║    · Todos los MIDIs se reescalan al TPB de salida                               ║
║    · Con --gap N: inserta N pulsos (beats) de silencio entre cada MIDI           ║
║      (default: 0)                                                                ║
║                                                                                  ║
║  METADATOS:                                                                      ║
║    · Cada MIDI conserva sus propios cambios de tempo, compás y                   ║
║      program_change en el punto en que ocurren dentro de la secuencia            ║
║    · Si ningún MIDI define tempo al principio, se añade 120 BPM por              ║
║      defecto                                                                     ║
║                                                                                  ║
║  USO:                                                                            ║
║    python midi_concat.py frase1.mid frase2.mid frase3.mid                        ║
║    python midi_concat.py partes/*.mid --output cancion.mid                       ║
║    python midi_concat.py partes/*.mid --gap 2                                    ║
║    python midi_concat.py partes/*.mid --tpb 480                                  ║
║    python midi_concat.py partes/*.mid --preview                                  ║
║    python midi_concat.py partes/*.mid --verbose                                  ║
║                                                                                  ║
║  OPCIONES:                                                                       ║
║    --output FILE     MIDI de salida (default: concat.mid)                        ║
║    --tpb N           Ticks per beat de salida (default: máximo de                ║
║                      los MIDIs de entrada)                                       ║
║    --gap N           Beats de silencio entre cada MIDI concatenado               ║
║                      (default: 0)                                                ║
║    --preview         Mostrar plan de concatenación sin generar MIDI              ║
║    --verbose         Informe detallado                                           ║
║                                                                                  ║
║  SALIDA:                                                                         ║
║    concat.mid  — MIDI de un único track con todos los MIDIs de entrada           ║
║                  reproducidos en serie, uno detrás de otro                       ║
║                                                                                  ║
║  DEPENDENCIAS: mido                                                              ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse

try:
    import mido
    from mido import MidiFile, MidiTrack, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

DEFAULT_TEMPO = 500000   # 120 BPM, en microsegundos por negra


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def stem_name(path: str) -> str:
    """Extrae el nombre base de un MIDI de entrada (sin extensión)."""
    return os.path.splitext(os.path.basename(path))[0]


def get_tpb(mid: MidiFile) -> int:
    return mid.ticks_per_beat


def get_midi_duration_ticks(mid: MidiFile) -> int:
    """Duración total del MIDI en ticks (el track más largo)."""
    max_tick = 0
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
        max_tick = max(max_tick, abs_tick)
    return max_tick


# ══════════════════════════════════════════════════════════════════════════════
#  REESCALADO DE TPB
# ══════════════════════════════════════════════════════════════════════════════

def rescale_ticks(mid: MidiFile, target_tpb: int) -> MidiFile:
    """Reescala todos los tiempos del MIDI al TPB de destino."""
    src_tpb = mid.ticks_per_beat
    if src_tpb == target_tpb:
        return mid

    ratio = target_tpb / src_tpb
    out   = MidiFile(ticks_per_beat=target_tpb)

    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            new_time = int(round(msg.time * ratio))
            new_track.append(msg.copy(time=new_time))
        out.tracks.append(new_track)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  APLANADO DE EVENTOS
# ══════════════════════════════════════════════════════════════════════════════

def flatten_events(mid: MidiFile) -> list:
    """Aplana todos los tracks de un MIDI en una única lista de eventos
    (tick_absoluto, mensaje), ordenada por tiempo.

    Se descartan los metamensajes puramente estructurales (track_name,
    end_of_track), ya que no tienen sentido fuera de su track original.
    El resto de metadatos (tempo, compás, program_change, notas, etc.)
    se conservan tal cual, en el punto exacto en que ocurren.
    """
    events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.is_meta and msg.type in ('track_name', 'end_of_track',
                                             'instrument_name'):
                continue
            events.append((abs_tick, msg))

    events.sort(key=lambda x: (x[0], 0 if x[1].is_meta else 1))
    return events


# ══════════════════════════════════════════════════════════════════════════════
#  CONCATENACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def build_concat(stems: list, target_tpb: int, gap_beats: float,
                  verbose: bool):
    """Concatena los MIDIs de entrada uno detrás de otro.

    Cada MIDI de entrada se aplana y se desplaza en el tiempo para que
    comience justo donde terminó el anterior (más el gap opcional).
    El resultado es un único track con toda la secuencia.

    Devuelve (MidiFile, timeline) donde timeline es una lista de
    (nombre, tick_inicio, tick_fin) útil para el preview.
    """
    gap_ticks = int(round(gap_beats * target_tpb))

    out = MidiFile(ticks_per_beat=target_tpb)
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Concat', time=0))

    all_events = []   # (tick_abs, msg)
    timeline   = []   # (nombre, inicio, fin)
    offset = 0

    for path, mid in stems:
        scaled   = rescale_ticks(mid, target_tpb)
        events   = flatten_events(scaled)
        duration = get_midi_duration_ticks(scaled)
        name     = stem_name(path)

        for tick, msg in events:
            all_events.append((tick + offset, msg))

        timeline.append((name, offset, offset + duration))

        if verbose:
            print(f"  {name:30s}  inicio={offset:8d} ticks  "
                  f"duración={duration:8d} ticks  "
                  f"({duration / target_tpb:.2f} beats)")

        offset += duration + gap_ticks

    # Asegurar tempo por defecto si ningún MIDI define uno al principio
    has_start_tempo = any(msg.type == 'set_tempo' and tick == 0
                          for tick, msg in all_events)
    if not has_start_tempo:
        all_events.insert(0, (0, MetaMessage('set_tempo',
                                              tempo=DEFAULT_TEMPO, time=0)))

    all_events.sort(key=lambda x: (x[0], 0 if x[1].is_meta else 1))

    prev_tick = 0
    for tick, msg in all_events:
        delta = max(0, tick - prev_tick)
        track.append(msg.copy(time=delta))
        prev_tick = tick

    track.append(MetaMessage('end_of_track', time=0))
    out.tracks.append(track)

    return out, timeline


# ══════════════════════════════════════════════════════════════════════════════
#  PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

def show_preview(stems: list, target_tpb: int, gap_beats: float,
                  timeline: list, out_path: str):
    """Muestra el plan de concatenación sin generar el MIDI."""
    total_ticks = timeline[-1][2] if timeline else 0

    print(f"\n── PREVIEW ──────────────────────────────────────────────────")
    print(f"  MIDIs de entrada : {len(stems)}")
    print(f"  TPB de salida    : {target_tpb}")
    print(f"  Gap entre MIDIs  : {gap_beats} beats")
    print(f"  Duración total   : {total_ticks} ticks "
          f"({total_ticks / target_tpb:.2f} beats)")
    print(f"  Fichero de salida: {out_path}")
    print()
    print(f"  {'#':3s}  {'MIDI':30s}  {'Inicio (beats)':16s}  {'Fin (beats)'}")
    print(f"  {'─'*3}  {'─'*30}  {'─'*16}  {'─'*12}")
    for i, (name, start, end) in enumerate(timeline, 1):
        print(f"  {i:3d}  {name:30s}  {start / target_tpb:16.2f}  "
              f"{end / target_tpb:.2f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Concatena varios MIDIs uno detrás de otro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("stems", nargs="+",
                   help="MIDIs de entrada, en el orden en que se concatenan")
    p.add_argument("--output", default="concat.mid",
                   help="MIDI de salida (default: concat.mid)")
    p.add_argument("--tpb", type=int, default=None, metavar="N",
                   help="Ticks per beat de salida (default: máximo de "
                        "los MIDIs de entrada)")
    p.add_argument("--gap", type=float, default=0.0, metavar="N",
                   help="Beats de silencio entre cada MIDI concatenado "
                        "(default: 0)")
    p.add_argument("--preview", action="store_true",
                   help="Mostrar plan de concatenación sin generar MIDI")
    p.add_argument("--verbose", action="store_true",
                   help="Informe detallado")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Cargar MIDIs de entrada ─────────────────────────────────────────────
    stems = []
    for path in args.stems:
        if not os.path.isfile(path):
            print(f"ERROR: no se encuentra el archivo: {path}")
            sys.exit(1)
        try:
            mid = MidiFile(path)
            stems.append((path, mid))
        except Exception as e:
            print(f"ERROR al leer {path}: {e}")
            sys.exit(1)

    if not stems:
        print("ERROR: no se encontraron MIDIs válidos")
        sys.exit(1)

    # ── TPB de salida ─────────────────────────────────────────────────────────
    if args.tpb:
        target_tpb = args.tpb
    else:
        target_tpb = max(get_tpb(mid) for _, mid in stems)

    if args.gap < 0:
        print("ERROR: --gap no puede ser negativo")
        sys.exit(1)

    if args.verbose:
        print(f"\n[INFO] MIDIs de entrada : {len(stems)}")
        print(f"[INFO] TPB salida       : {target_tpb}")
        print(f"[INFO] Gap              : {args.gap} beats")
        print(f"[INFO] Fichero          : {args.output}")
        print()

    # ── Concatenar ───────────────────────────────────────────────────────────
    out_mid, timeline = build_concat(stems, target_tpb, args.gap, args.verbose)

    # ── Preview ───────────────────────────────────────────────────────────────
    if args.preview:
        show_preview(stems, target_tpb, args.gap, timeline, args.output)
        return

    if args.verbose:
        show_preview(stems, target_tpb, args.gap, timeline, args.output)

    # ── Guardar ──────────────────────────────────────────────────────────────
    try:
        out_mid.save(args.output)
        total_beats = timeline[-1][2] / target_tpb if timeline else 0
        print(f"MIDI concatenado guardado en: {args.output}  "
              f"({len(stems)} MIDIs, TPB={target_tpb}, "
              f"duración={total_beats:.2f} beats)")
    except Exception as e:
        print(f"ERROR al guardar el MIDI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
