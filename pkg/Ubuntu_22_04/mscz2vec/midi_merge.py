#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MIDI MERGE  v1.0                                    ║
║         Combina varios MIDIs en un único MIDI multitracks                   ║
║                                                                              ║
║  Toma N MIDIs (stems) y los combina en un único archivo MIDI donde cada     ║
║  stem de entrada se convierte en un track separado. Útil para combinar      ║
║  stems exportados y compararlos entre sí o con un MIDI de                 ║
║  referencia usando analyzer.py.                                              ║
║                                                                              ║
║  GESTIÓN DE CANALES:                                                         ║
║    · Por defecto: reasigna canales automáticamente para evitar colisiones   ║
║    · Canal 9 reservado para percusión (no se reasigna)                     ║
║    · Los program_change originales de cada stem se preservan               ║
║    · Con --channels: fuerza canales concretos para stems específicos        ║
║                                                                              ║
║  GESTIÓN DE TPB:                                                             ║
║    · Por defecto: usa el TPB máximo de todos los stems de entrada           ║
║    · Con --tpb N: usa el valor indicado                                     ║
║    · Todos los stems se reescalan al TPB de salida                         ║
║                                                                              ║
║  USO:                                                                        ║
║    python midi_merge.py stem_violin.mid stem_viola.mid stem_cello.mid      ║
║    python midi_merge.py stems/*.mid --output merged.mid                     ║
║    python midi_merge.py stems/*.mid --tpb 480                               ║
║    python midi_merge.py stems/*.mid --channels violin:0 viola:1 cello:2    ║
║    python midi_merge.py stems/*.mid --preview                               ║
║    python midi_merge.py stems/*.mid --verbose                               ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --output FILE        MIDI de salida (default: merged.mid)               ║
║    --tpb N              Ticks per beat de salida (default: máximo de       ║
║                         los stems de entrada)                               ║
║    --channels S:N [...] Mapeo manual de canal: nombre_stem:canal           ║
║                         El nombre puede ser el fichero completo o solo     ║
║                         la parte antes del primer punto                    ║
║                         Ejemplo: violin:0 viola:1 cello:2                  ║
║    --preview            Mostrar plan de merge sin generar MIDI             ║
║    --verbose            Informe detallado                                   ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    merged.mid  — MIDI multitracks con un track por stem de entrada         ║
║                                                                              ║
║  DEPENDENCIAS: mido                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import math

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

PERCUSSION_CHANNEL = 9


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE STEMS
# ══════════════════════════════════════════════════════════════════════════════

def stem_name(path: str) -> str:
    """Extrae el nombre base de un stem (sin extensión)."""
    return os.path.splitext(os.path.basename(path))[0]


def stem_key(path: str) -> str:
    """Clave de búsqueda para mapeo de canales: nombre antes del primer punto."""
    name = os.path.basename(path)
    return name.split('.')[0]


def get_channels_used(mid: MidiFile) -> set:
    """Devuelve el conjunto de canales usados en el MIDI."""
    channels = set()
    for track in mid.tracks:
        for msg in track:
            if hasattr(msg, 'channel'):
                channels.add(msg.channel)
    return channels


def get_program_changes(mid: MidiFile) -> dict:
    """Devuelve dict de canal → programa para el MIDI."""
    programs = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'program_change':
                programs[msg.channel] = msg.program
    return programs


def get_tpb(mid: MidiFile) -> int:
    return mid.ticks_per_beat


def get_track_name(track: MidiTrack) -> str:
    """Extrae el nombre del track de sus metadatos."""
    for msg in track:
        if msg.type == 'track_name':
            return msg.name
    return ''


def get_midi_duration_ticks(mid: MidiFile) -> int:
    """Duración total del MIDI en ticks."""
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
#  ASIGNACIÓN DE CANALES
# ══════════════════════════════════════════════════════════════════════════════

def build_channel_map(stems: list, manual_map: dict) -> dict:
    """Construye el mapa de (path, canal_original) → canal_destino.

    Reglas:
    - Canal 9 (percusión) siempre se mantiene en canal 9
    - Si hay mapeo manual para el stem, se usa ese canal
    - El resto se asigna secuencialmente evitando colisiones y canal 9
    """
    # Recopilar todos los canales usados por cada stem
    stem_channels = {}   # path → set de canales usados
    for path, mid in stems:
        stem_channels[path] = get_channels_used(mid)

    channel_map = {}   # (path, canal_src) → canal_dst
    used_dst    = {PERCUSSION_CHANNEL}   # canal 9 siempre reservado

    # Aplicar mapeos manuales primero
    for path, mid in stems:
        key  = stem_key(path)
        name = stem_name(path)
        if key in manual_map or name in manual_map:
            forced = manual_map.get(key, manual_map.get(name))
            for ch in stem_channels[path]:
                if ch == PERCUSSION_CHANNEL:
                    channel_map[(path, ch)] = PERCUSSION_CHANNEL
                else:
                    channel_map[(path, ch)] = forced
                    used_dst.add(forced)

    # Asignar el resto automáticamente
    next_ch = 0
    for path, mid in stems:
        key  = stem_key(path)
        name = stem_name(path)
        if key in manual_map or name in manual_map:
            continue   # ya asignado

        for ch in sorted(stem_channels[path]):
            if ch == PERCUSSION_CHANNEL:
                channel_map[(path, ch)] = PERCUSSION_CHANNEL
                continue
            # Buscar siguiente canal libre
            while next_ch in used_dst:
                next_ch += 1
                if next_ch > 15:
                    next_ch = 0   # wrap — en caso extremo de > 15 canales no-perc
            channel_map[(path, ch)] = next_ch
            used_dst.add(next_ch)
            next_ch += 1

    return channel_map


# ══════════════════════════════════════════════════════════════════════════════
#  MERGE
# ══════════════════════════════════════════════════════════════════════════════

def merge_stems(stems: list, target_tpb: int, channel_map: dict,
                verbose: bool) -> MidiFile:
    """Combina los stems en un único MIDI multitracks.

    Cada stem de entrada → un track en el MIDI de salida.
    Los tracks de tempo del primer stem se incluyen en el track 0.
    """
    out = MidiFile(ticks_per_beat=target_tpb)

    # Track 0: tempo y metadatos globales del primer stem
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage('track_name', name='Master', time=0))

    first_path, first_mid = stems[0]
    first_scaled = rescale_ticks(first_mid, target_tpb)
    for track in first_scaled.tracks:
        for msg in track:
            if msg.is_meta and msg.type in ('set_tempo', 'time_signature',
                                             'key_signature'):
                tempo_track.append(msg.copy())
                break   # solo el primero de cada tipo

    # Asegurar tempo por defecto si no hay ninguno
    has_tempo = any(msg.type == 'set_tempo'
                    for msg in tempo_track if msg.is_meta)
    if not has_tempo:
        tempo_track.append(MetaMessage('set_tempo', tempo=500000, time=0))

    tempo_track.append(MetaMessage('end_of_track', time=0))
    out.tracks.append(tempo_track)

    # Un track por stem
    for path, mid in stems:
        scaled = rescale_ticks(mid, target_tpb)
        name   = stem_name(path)

        # Fusionar todos los tracks del stem en uno solo
        all_events = []   # (tick_abs, msg)
        for track in scaled.tracks:
            abs_tick = 0
            for msg in track:
                abs_tick += msg.time
                if msg.is_meta:
                    if msg.type == 'track_name':
                        continue   # usaremos el nombre del fichero
                    if msg.type == 'end_of_track':
                        continue
                    if msg.type in ('set_tempo', 'time_signature', 'key_signature'):
                        continue   # ya en track 0
                all_events.append((abs_tick, msg))

        # Reasignar canales
        remapped = []
        for (tick, msg) in all_events:
            if hasattr(msg, 'channel'):
                dst_ch = channel_map.get((path, msg.channel), msg.channel)
                remapped.append((tick, msg.copy(channel=dst_ch)))
            else:
                remapped.append((tick, msg))

        # Ordenar por tick
        remapped.sort(key=lambda x: (x[0], 0 if x[1].is_meta else 1))

        # Construir track con deltas
        new_track = MidiTrack()
        new_track.append(MetaMessage('track_name', name=name, time=0))

        prev_tick = 0
        for (tick, msg) in remapped:
            delta = max(0, tick - prev_tick)
            new_track.append(msg.copy(time=delta))
            prev_tick = tick

        new_track.append(MetaMessage('end_of_track', time=0))
        out.tracks.append(new_track)

        if verbose:
            ch_used = sorted({channel_map.get((path, ch), ch)
                               for ch in get_channels_used(mid)})
            programs = get_program_changes(mid)
            print(f"  {name:30s}  canales={ch_used}  "
                  f"programas={dict(sorted(programs.items()))}")

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

def show_preview(stems: list, target_tpb: int, channel_map: dict,
                 out_path: str):
    """Muestra el plan de merge sin generar el MIDI."""
    print(f"\n── PREVIEW ──────────────────────────────────────────────────")
    print(f"  Stems de entrada : {len(stems)}")
    print(f"  TPB de salida    : {target_tpb}")
    print(f"  Fichero de salida: {out_path}")
    print()
    print(f"  {'Stem':32s}  {'TPB src':8s}  {'Canales src':14s}  "
          f"{'Canales dst':14s}  {'Programas'}")
    print(f"  {'─'*32}  {'─'*8}  {'─'*14}  {'─'*14}  {'─'*20}")
    for path, mid in stems:
        name     = stem_name(path)
        src_tpb  = get_tpb(mid)
        channels = sorted(get_channels_used(mid))
        programs = get_program_changes(mid)
        dst_chs  = sorted({channel_map.get((path, ch), ch) for ch in channels})
        prog_str = ' '.join(f"ch{k}→p{v}" for k, v in sorted(programs.items()))
        print(f"  {name:32s}  {src_tpb:8d}  "
              f"{str(channels):14s}  {str(dst_chs):14s}  {prog_str}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_channel_map(args_channels: list) -> dict:
    """Parsea lista de 'nombre:canal' a dict."""
    result = {}
    if not args_channels:
        return result
    for item in args_channels:
        if ':' not in item:
            print(f"AVISO: formato de canal no válido '{item}' "
                  f"(usar nombre:N, ej: violin:0) — ignorado")
            continue
        parts = item.rsplit(':', 1)
        name  = parts[0].strip()
        try:
            ch = int(parts[1].strip())
            if not 0 <= ch <= 15:
                print(f"AVISO: canal {ch} fuera de rango 0-15 — ignorado")
                continue
            result[name] = ch
        except ValueError:
            print(f"AVISO: canal no válido en '{item}' — ignorado")
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combina varios MIDIs en un único MIDI multitracks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("stems", nargs="+",
                   help="MIDIs de entrada (stems)")
    p.add_argument("--output", default="merged.mid",
                   help="MIDI de salida (default: merged.mid)")
    p.add_argument("--tpb", type=int, default=None, metavar="N",
                   help="Ticks per beat de salida (default: máximo de los stems)")
    p.add_argument("--channels", nargs="+", metavar="S:N",
                   help="Mapeo manual de canal: nombre_stem:canal "
                        "(ej: violin:0 viola:1 cello:2)")
    p.add_argument("--preview", action="store_true",
                   help="Mostrar plan de merge sin generar MIDI")
    p.add_argument("--verbose", action="store_true",
                   help="Informe detallado")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Cargar stems ─────────────────────────────────────────────────────────
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
        print("ERROR: no se encontraron stems válidos")
        sys.exit(1)

    # ── TPB de salida ─────────────────────────────────────────────────────────
    if args.tpb:
        target_tpb = args.tpb
    else:
        target_tpb = max(get_tpb(mid) for _, mid in stems)

    if args.verbose:
        print(f"\n[INFO] Stems       : {len(stems)}")
        print(f"[INFO] TPB salida  : {target_tpb}")
        print(f"[INFO] Fichero     : {args.output}")

    # ── Mapeo de canales ──────────────────────────────────────────────────────
    manual_map  = parse_channel_map(args.channels)
    channel_map = build_channel_map(stems, manual_map)

    # ── Preview ───────────────────────────────────────────────────────────────
    if args.preview:
        show_preview(stems, target_tpb, channel_map, args.output)
        return

    if args.verbose:
        show_preview(stems, target_tpb, channel_map, args.output)

    # ── Merge ────────────────────────────────────────────────────────────────
    out_mid = merge_stems(stems, target_tpb, channel_map, args.verbose)

    # ── Guardar ──────────────────────────────────────────────────────────────
    try:
        out_mid.save(args.output)
        n_tracks = len(out_mid.tracks) - 1   # sin contar el track de tempo
        print(f"MIDI merged guardado en: {args.output}  "
              f"({n_tracks} stems, TPB={target_tpb})")
    except Exception as e:
        print(f"ERROR al guardar el MIDI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
