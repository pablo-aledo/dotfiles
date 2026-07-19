#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║                          MIDI CONCAT  v2.0                                       ║
║         Concatena varios MIDIs (o secciones de ellos) uno detrás de otro         ║
║                                                                                  ║
║  Toma N MIDIs de entrada y genera un único MIDI en el que el contenido           ║
║  de cada uno se reproduce a continuación del anterior (no simultáneo,            ║
║  como hace midi_merge.py, sino en serie). Útil para encadenar frases,            ║
║  variaciones o secciones generadas por separado en una sola pieza.               ║
║                                                                                  ║
║  SELECCIÓN DE PISTAS:                                                           ║
║    · Cada MIDI de entrada puede llevar, opcionalmente, un selector de            ║
║      pistas tras '@' → archivo.mid@ESPEC                                         ║
║    · ESPEC admite pista suelta "0", rango "0-2", lista "0,2,4" o                 ║
║      combinaciones "0-2,5", o la palabra "all" (todas, valor por defecto)        ║
║    · Las pistas se numeran desde 0, en el orden en que aparecen en el            ║
║      MIDI; usa --list-tracks para ver índice, nombre y contenido de cada         ║
║      pista de un archivo                                                        ║
║    · Sin '@', se usan todas las pistas del MIDI, como antes                      ║
║                                                                                  ║
║  SELECCIÓN DE COMPASES:                                                         ║
║    · Cada MIDI de entrada puede llevar, opcionalmente, un selector de            ║
║      compases tras ':' → archivo.mid:ESPEC (combinable con @ESPEC de            ║
║      pistas: archivo.mid@0,2:1-3)                                                ║
║    · ESPEC admite compás suelto "3", rango "1-3", lista "5,6,7" o                ║
║      combinaciones "1-3,7,9-10" (compases 1-indexados)                           ║
║    · Un rango invertido "7-5" toma los compases 7,6,5 en ese orden               ║
║      (útil para invertir una frase)                                              ║
║    · Los compases se detectan a partir de los eventos time_signature de          ║
║      las pistas seleccionadas (4/4 por defecto si no hay ninguno); los           ║
║      cambios de compás dentro de un mismo MIDI se respetan al calcular           ║
║      los límites                                                                 ║
║    · Sin ':', se usa el MIDI completo (o la pista completa), como antes          ║
║                                                                                  ║
║  ORDEN:                                                                          ║
║    · Los MIDIs (o compases seleccionados) se concatenan en el orden en           ║
║      que se pasan por línea de comandos                                         ║
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
║    python midi_concat.py a.mid:1-3 b.mid:5,6,7                                   ║
║    python midi_concat.py a.mid:1-3 b.mid:5-7 --output frase.mid                  ║
║    python midi_concat.py a.mid:8-1 --output retro.mid                            ║
║    python midi_concat.py a.mid@0 b.mid@1,2 --output solo_pistas.mid              ║
║    python midi_concat.py a.mid@0-1:1-3 b.mid@2:5-7 --output mix.mid              ║
║    python midi_concat.py --list-bars a.mid                                       ║
║    python midi_concat.py --list-tracks a.mid                                     ║
║                                                                                  ║
║  OPCIONES:                                                                       ║
║    --output FILE     MIDI de salida (default: concat.mid)                        ║
║    --tpb N           Ticks per beat de salida (default: máximo de                ║
║                      los MIDIs de entrada)                                       ║
║    --gap N           Beats de silencio entre cada MIDI concatenado               ║
║                      (default: 0)                                                ║
║    --preview         Mostrar plan de concatenación sin generar MIDI              ║
║    --list-bars       Para cada MIDI dado (sin generar nada), listar sus          ║
║                      compases detectados con su posición en beats, para          ║
║                      saber qué números usar en archivo.mid:N-M (si se            ║
║                      combina con @ESPEC, calcula los compases solo sobre         ║
║                      las pistas seleccionadas)                                   ║
║    --list-tracks     Para cada MIDI dado (sin generar nada), listar sus          ║
║                      pistas con índice, nombre, nº de notas, canal y             ║
║                      programa, para saber qué números usar en                    ║
║                      archivo.mid@N                                               ║
║    --verbose         Informe detallado                                           ║
║                                                                                  ║
║  SALIDA:                                                                         ║
║    concat.mid  — MIDI de un único track con todos los MIDIs (o                   ║
║                  fragmentos de compases) de entrada reproducidos en              ║
║                  serie, uno detrás de otro                                       ║
║                                                                                  ║
║  DEPENDENCIAS: mido                                                              ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
import sys
import argparse

try:
    import mido
    from mido import MidiFile, MidiTrack, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

DEFAULT_TEMPO = 500000   # 120 BPM, en microsegundos por negra

# archivo.mid:ESPEC  →  ESPEC es una lista de compases/rango separados por
# comas, p.ej. "3", "1-3", "5,6,7", "1-3,7,9-10", "7-5" (rango invertido)
BAR_SPEC_RE = re.compile(r'^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$')

# archivo.mid@ESPEC  →  ESPEC es una lista de pistas/rango separados por
# comas (0-indexadas), p.ej. "0", "0-2", "0,2,4", "0-2,5", o la palabra
# "all" para todas las pistas (equivalente a omitir el selector)
TRACK_SPEC_RE = re.compile(r'^(?i:all)$|^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$')


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
#  SELECCIÓN DE COMPASES
# ══════════════════════════════════════════════════════════════════════════════

def parse_stem_arg(arg: str):
    """Separa 'archivo.mid[@TRACKSPEC][:BARSPEC]' en (ruta, trackspec,
    barspec). Cada selector es opcional e independiente:
        archivo.mid                 → (archivo.mid, None,  None)
        archivo.mid:1-3             → (archivo.mid, None,  '1-3')
        archivo.mid@0,2             → (archivo.mid, '0,2', None)
        archivo.mid@0,2:1-3         → (archivo.mid, '0,2', '1-3')
    Solo se interpreta como selector si lo que sigue al separador final
    encaja con el patrón correspondiente (para no romper rutas que
    pudieran contener ':' o '@')."""
    path_part, barspec = arg, None
    if ':' in arg:
        p, spec = arg.rsplit(':', 1)
        if p and BAR_SPEC_RE.match(spec):
            path_part, barspec = p, spec

    path, trackspec = path_part, None
    if '@' in path_part:
        p, spec = path_part.rsplit('@', 1)
        if p and TRACK_SPEC_RE.match(spec):
            path, trackspec = p, spec

    return path, trackspec, barspec


def parse_track_spec(spec: str, n_tracks: int) -> list:
    """Convierte un ESPEC de pistas ('all', '0', '0,2', '0-2,5') en una
    lista de índices de pista 0-indexados, en el orden dado. 'all'
    devuelve todas las pistas del MIDI en su orden original."""
    if spec.strip().lower() == 'all':
        return list(range(n_tracks))

    indices = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a_str, b_str = token.split('-')
            a, b = int(a_str), int(b_str)
            step = 1 if a <= b else -1
            indices.extend(range(a, b + step, step))
        else:
            indices.append(int(token))

    for i in indices:
        if i < 0 or i >= n_tracks:
            raise ValueError(
                f"pista {i} fuera de rango (el MIDI tiene {n_tracks} "
                f"pistas; usa --list-tracks para verlas)")
    return indices


def select_tracks(mid: MidiFile, track_indices: list) -> MidiFile:
    """Devuelve una copia del MIDI conteniendo solo las pistas indicadas
    (0-indexadas), en el orden dado. El resto del pipeline (reescalado,
    detección de compases, aplanado de eventos) trabaja después sobre
    este MIDI ya filtrado, sin necesidad de saber que hubo selección."""
    out = MidiFile(ticks_per_beat=mid.ticks_per_beat)
    for i in track_indices:
        out.tracks.append(mid.tracks[i])
    return out


def show_track_list(path: str, mid: MidiFile):
    """Imprime, para un MIDI dado, la lista de pistas detectadas con su
    índice, nombre, nº de notas, canales y programas usados, para saber
    qué números usar en archivo.mid@N."""
    print(f"\n── PISTAS: {stem_name(path)} ──────────────────────────────")
    print(f"  Pistas detectadas: {len(mid.tracks)}")
    print()
    print(f"  {'#':3s}  {'Nombre':24s}  {'Eventos':8s}  {'Notas':6s}  "
          f"{'Canal':6s}  {'Programa'}")
    print(f"  {'─'*3}  {'─'*24}  {'─'*8}  {'─'*6}  {'─'*6}  {'─'*10}")
    for i, track in enumerate(mid.tracks):
        name = ''
        n_notes = 0
        channels = set()
        programs = set()
        for msg in track:
            if msg.is_meta:
                if msg.type == 'track_name' and not name:
                    name = msg.name
            else:
                if msg.type == 'note_on' and msg.velocity > 0:
                    n_notes += 1
                if hasattr(msg, 'channel'):
                    channels.add(msg.channel)
                if msg.type == 'program_change':
                    programs.add(msg.program)
        chan_str = ','.join(str(c) for c in sorted(channels)) if channels else '-'
        prog_str = ','.join(str(p) for p in sorted(programs)) if programs else '-'
        print(f"  {i:3d}  {name[:24]:24s}  {len(track):8d}  {n_notes:6d}  "
              f"{chan_str:6s}  {prog_str}")
    print()


def parse_bar_spec(spec: str) -> list:
    """Convierte un ESPEC ('1-3,7,9-10') en una lista de números de compás
    1-indexados, en el orden dado. Los rangos invertidos ('7-5') se expanden
    en orden descendente."""
    bars = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a_str, b_str = token.split('-')
            a, b = int(a_str), int(b_str)
            step = 1 if a <= b else -1
            bars.extend(range(a, b + step, step))
        else:
            bars.append(int(token))
    return bars


def compute_bar_boundaries(mid: MidiFile) -> list:
    """Calcula los límites (tick_inicio, tick_fin) de cada compás del MIDI,
    a partir de sus eventos time_signature (4/4 por defecto si no hay
    ninguno). Genera compases de sobra más allá de la duración real del
    MIDI, por si se pide un compás que solo contiene silencio final."""
    tpb = mid.ticks_per_beat

    ts_events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'time_signature':
                ts_events.append((abs_tick, msg.numerator, msg.denominator))
    ts_events.sort(key=lambda x: x[0])
    if not ts_events or ts_events[0][0] != 0:
        ts_events.insert(0, (0, 4, 4))

    duration = get_midi_duration_ticks(mid)
    margin   = tpb * 4 * 16   # 16 compases de 4/4 de margen de seguridad
    limit    = duration + margin

    bars = []
    idx = 1
    num, den = ts_events[0][1], ts_events[0][2]
    cur_tick = 0
    while cur_tick < limit:
        while idx < len(ts_events) and ts_events[idx][0] <= cur_tick:
            num, den = ts_events[idx][1], ts_events[idx][2]
            idx += 1
        bar_ticks = int(round(num * 4 / den * tpb))
        if bar_ticks <= 0:
            bar_ticks = tpb * 4
        bars.append((cur_tick, cur_tick + bar_ticks))
        cur_tick += bar_ticks

    return bars


def select_bars(mid: MidiFile, bar_numbers: list):
    """Extrae de un MIDI (ya reescalado) los eventos correspondientes a los
    compases pedidos, concatenándolos entre sí de forma contigua (sin los
    huecos de los compases no seleccionados).

    Devuelve (eventos, duracion_ticks), en el mismo formato que
    flatten_events()/get_midi_duration_ticks(), listo para insertar en la
    línea de tiempo de salida.
    """
    bars = compute_bar_boundaries(mid)
    events_full = flatten_events(mid)

    output = []
    cursor = 0
    for n in bar_numbers:
        if n < 1 or n > len(bars):
            raise ValueError(
                f"compás {n} fuera de rango (se detectaron {len(bars)} "
                f"compases; usa --list-bars para verlos)")
        b_start, b_end = bars[n - 1]
        for tick, msg in events_full:
            if b_start <= tick < b_end:
                output.append((cursor + (tick - b_start), msg))
        cursor += (b_end - b_start)

    output.sort(key=lambda x: (x[0], 0 if x[1].is_meta else 1))
    return output, cursor


def show_bar_list(path: str, mid: MidiFile):
    """Imprime, para un MIDI dado, la lista de compases detectados con su
    posición en beats, para saber qué números usar en archivo.mid:N-M."""
    tpb = mid.ticks_per_beat
    duration = get_midi_duration_ticks(mid)
    bars = compute_bar_boundaries(mid)

    # Solo mostramos compases que caen (al menos parcialmente) dentro de
    # la duración real del MIDI, más uno de cortesía si termina justo
    # en un límite de compás.
    shown = [b for b in bars if b[0] < duration] or bars[:1]

    print(f"\n── COMPASES: {stem_name(path)} ────────────────────────────────")
    print(f"  TPB origen : {tpb}")
    print(f"  Duración   : {duration} ticks ({duration / tpb:.2f} beats)")
    print(f"  Compases detectados: {len(shown)}")
    print()
    print(f"  {'#':4s}  {'Inicio (beats)':16s}  {'Fin (beats)'}")
    print(f"  {'─'*4}  {'─'*16}  {'─'*12}")
    for i, (start, end) in enumerate(shown, 1):
        print(f"  {i:4d}  {start / tpb:16.2f}  {end / tpb:.2f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CONCATENACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def build_concat(stems: list, target_tpb: int, gap_beats: float,
                  verbose: bool):
    """Concatena los MIDIs de entrada (o los compases seleccionados de
    cada uno) uno detrás de otro.

    Cada MIDI de entrada se aplana y se desplaza en el tiempo para que
    comience justo donde terminó el anterior (más el gap opcional). Si el
    stem trae un selector de compases, primero se extraen y se compactan
    solo esos compases. El resultado es un único track con toda la
    secuencia.

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

    for path, mid, barspec, trackspec in stems:
        # 'mid' llega ya filtrado a las pistas seleccionadas (si las hubo);
        # a partir de aquí el resto del pipeline no necesita saberlo.
        scaled = rescale_ticks(mid, target_tpb)

        base_name = stem_name(path)
        if trackspec:
            base_name += f"@{trackspec}"

        if barspec:
            bar_numbers = parse_bar_spec(barspec)
            try:
                events, duration = select_bars(scaled, bar_numbers)
            except ValueError as e:
                print(f"ERROR en {path}: {e}")
                sys.exit(1)
            name = f"{base_name}:{barspec}"
        else:
            events   = flatten_events(scaled)
            duration = get_midi_duration_ticks(scaled)
            name     = base_name

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
                   help="MIDIs de entrada, en el orden en que se concatenan. "
                        "Admite selector de pistas: archivo.mid@0, "
                        "archivo.mid@0,2, archivo.mid@0-2, archivo.mid@all. "
                        "Admite selector de compases: archivo.mid:1-3, "
                        "archivo.mid:5,6,7, archivo.mid:1-3,7. Combinables: "
                        "archivo.mid@0,2:1-3")
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
    p.add_argument("--list-bars", action="store_true",
                   help="Para cada MIDI dado, listar sus compases "
                        "detectados (posición en beats) y salir, sin "
                        "generar ningún MIDI. Si el MIDI trae selector de "
                        "pistas (@ESPEC), los compases se calculan solo "
                        "sobre esas pistas")
    p.add_argument("--list-tracks", action="store_true",
                   help="Para cada MIDI dado, listar sus pistas (índice, "
                        "nombre, nº de notas, canal, programa) y salir, "
                        "sin generar ningún MIDI")
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
    for raw_arg in args.stems:
        path, trackspec, barspec = parse_stem_arg(raw_arg)
        if not os.path.isfile(path):
            print(f"ERROR: no se encuentra el archivo: {path}")
            sys.exit(1)
        try:
            mid = MidiFile(path)
        except Exception as e:
            print(f"ERROR al leer {path}: {e}")
            sys.exit(1)

        # --list-tracks se resuelve sobre el MIDI completo, sin filtrar,
        # para que se puedan ver todas las pistas disponibles
        if args.list_tracks:
            show_track_list(path, mid)
            continue

        if trackspec:
            try:
                indices = parse_track_spec(trackspec, len(mid.tracks))
            except ValueError as e:
                print(f"ERROR: selector de pistas inválido en "
                      f"'{raw_arg}': {e}")
                sys.exit(1)
            mid = select_tracks(mid, indices)

        if barspec:
            try:
                nums = parse_bar_spec(barspec)
                if any(n < 1 for n in nums):
                    raise ValueError("los compases se numeran desde 1")
            except ValueError as e:
                print(f"ERROR: selector de compases inválido en "
                      f"'{raw_arg}': {e}")
                sys.exit(1)
        stems.append((path, mid, barspec, trackspec))

    if args.list_tracks:
        return

    if not stems:
        print("ERROR: no se encontraron MIDIs válidos")
        sys.exit(1)

    # ── --list-bars: solo inspeccionar, no generar nada ────────────────────
    # (si el MIDI trae selector de pistas, los compases se calculan solo
    # sobre las pistas ya filtradas)
    if args.list_bars:
        for path, mid, _, _ in stems:
            show_bar_list(path, mid)
        return

    # ── TPB de salida ─────────────────────────────────────────────────────────
    if args.tpb:
        target_tpb = args.tpb
    else:
        target_tpb = max(get_tpb(mid) for _, mid, _, _ in stems)

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
