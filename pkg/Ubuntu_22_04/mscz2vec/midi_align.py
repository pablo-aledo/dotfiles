#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MIDI ALIGN  v1.0                                    ║
║         Alineación temporal de MIDIs mediante puntos de anclaje             ║
║                                                                              ║
║  Recibe un MIDI y una lista de pares de tiempos (origen → destino) y        ║
║  reescala cada segmento para que sus bordes temporales coincidan con los    ║
║  tiempos de destino. Útil para alinear stems exportados de Suno con el     ║
║  MIDI original de referencia.                                                ║
║                                                                              ║
║  FORMATO DE ANCLAJES:                                                        ║
║    Cada anclaje es un par: ORIGEN DESTINO                                   ║
║    Formato de tiempo: M:SS.mmm  (minutos:segundos.milisegundos)             ║
║    El primer anclaje debe ser 0:00.000 0:00.000                             ║
║    El último segmento se estira hasta el final del MIDI de referencia       ║
║                                                                              ║
║  MODOS DE ALINEACIÓN (--mode):                                              ║
║    tempo   — cambia el BPM de cada segmento (default)                       ║
║              el ritmo interno se preserva, solo cambia la velocidad         ║
║    linear  — redistribuye las notas proporcionalmente en el tiempo          ║
║              útil cuando los desajustes son irregulares dentro del segmento ║
║                                                                              ║
║  USO:                                                                        ║
║    python midi_align.py stem.mid --anchors "0:00.000 0:00.000"             ║
║                                            "0:15.200 0:14.800"             ║
║                                            "0:32.400 0:31.000"             ║
║                                                                              ║
║    python midi_align.py stem.mid --ref original.mid                         ║
║                          --anchors "0:00.000 0:00.000"                     ║
║                                    "0:15.200 0:14.800"                     ║
║                                                                              ║
║    python midi_align.py stem.mid --anchors-file anclajes.txt               ║
║    python midi_align.py stem.mid --anchors ... --mode linear               ║
║    python midi_align.py stem.mid --anchors ... --preview                   ║
║                                                                              ║
║  ARCHIVO DE ANCLAJES (--anchors-file):                                      ║
║    Una línea por anclaje: ORIGEN DESTINO                                    ║
║    Las líneas que empiezan por # se ignoran.                               ║
║    Ejemplo:                                                                  ║
║      # Alineación stem_violin.mid                                           ║
║      0:00.000  0:00.000                                                     ║
║      0:15.200  0:14.800                                                     ║
║      0:32.400  0:31.000                                                     ║
║      1:05.000  1:04.200                                                     ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --anchors T T [T T ...]  Pares de tiempos origen→destino                ║
║    --anchors-file FILE      Leer anclajes desde archivo de texto            ║
║    --ref MIDI               MIDI de referencia (define la duración total)   ║
║    --mode MODE              tempo | linear (default: tempo)                 ║
║    --output FILE            MIDI de salida (default: <input>.aligned.mid)  ║
║    --preview                Mostrar segmentos sin generar MIDI              ║
║    --verbose                Informe detallado                               ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    <input>.aligned.mid  — MIDI alineado temporalmente                       ║
║                                                                              ║
║  DEPENDENCIAS: mido                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import re

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

TPB_REF = 480  # ticks per beat de referencia para modo linear


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO DE TIEMPOS
# ══════════════════════════════════════════════════════════════════════════════

def parse_time(s: str) -> float:
    """Convierte 'M:SS.mmm' a segundos float.

    Ejemplos: '0:00.000'→0.0, '0:15.200'→15.2, '1:05.000'→65.0
    """
    s = s.strip()
    m = re.match(r'^(\d+):(\d{2})\.(\d{1,3})$', s)
    if not m:
        raise ValueError(
            f"Formato de tiempo no válido: '{s}'  "
            f"(usar M:SS.mmm, ej: 0:15.200 o 1:05.000)"
        )
    minutes = int(m.group(1))
    seconds = int(m.group(2))
    ms_str  = m.group(3).ljust(3, '0')   # normalizar a 3 dígitos
    millis  = int(ms_str)
    return minutes * 60 + seconds + millis / 1000.0


def format_time(secs: float) -> str:
    """Convierte segundos float a 'M:SS.mmm'."""
    secs = max(0.0, secs)
    m    = int(secs // 60)
    s    = secs - m * 60
    ss   = int(s)
    ms   = round((s - ss) * 1000)
    if ms >= 1000:
        ms -= 1000
        ss += 1
    return f"{m}:{ss:02d}.{ms:03d}"


def parse_anchors(anchor_strings: list) -> list:
    """Parsea lista de strings 'ORIGEN DESTINO' a lista de (t_src, t_dst) en segundos."""
    anchors = []
    for raw in anchor_strings:
        parts = raw.strip().split()
        if len(parts) != 2:
            raise ValueError(
                f"Cada anclaje debe tener exactamente dos tiempos: '{raw}'"
            )
        t_src = parse_time(parts[0])
        t_dst = parse_time(parts[1])
        anchors.append((t_src, t_dst))
    return anchors


def load_anchors_file(path: str) -> list:
    """Carga anclajes desde un archivo de texto."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
    return parse_anchors(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DEL MIDI
# ══════════════════════════════════════════════════════════════════════════════

def get_tempo_map(mid: MidiFile) -> list:
    """Extrae el mapa de tempos del MIDI.

    Devuelve lista de (tick_absoluto, tempo_us) ordenada por tick.
    """
    tempo_map = [(0, 500000)]   # default 120 BPM
    abs_tick = 0
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                # Reemplazar o añadir
                if tempo_map and tempo_map[-1][0] == abs_tick:
                    tempo_map[-1] = (abs_tick, msg.tempo)
                else:
                    tempo_map.append((abs_tick, msg.tempo))
    return sorted(set(tempo_map), key=lambda x: x[0])


def ticks_to_seconds(tick: int, tpb: int, tempo_map: list) -> float:
    """Convierte un tick absoluto a segundos usando el mapa de tempos."""
    secs = 0.0
    prev_tick, prev_tempo = 0, 500000
    for (t, tempo) in tempo_map:
        if t >= tick:
            break
        dt = min(t, tick) - prev_tick
        secs += dt * prev_tempo / (tpb * 1_000_000)
        prev_tick, prev_tempo = t, tempo
    # Resto desde el último cambio de tempo
    dt = tick - prev_tick
    secs += dt * prev_tempo / (tpb * 1_000_000)
    return secs


def seconds_to_ticks(secs: float, tpb: int, tempo_map: list) -> int:
    """Convierte segundos a ticks absolutos usando el mapa de tempos."""
    elapsed = 0.0
    prev_tick, prev_tempo = 0, 500000
    for (t, tempo) in tempo_map:
        seg_ticks = t - prev_tick
        seg_secs  = seg_ticks * prev_tempo / (tpb * 1_000_000)
        if elapsed + seg_secs >= secs:
            remaining = secs - elapsed
            extra_ticks = int(remaining * tpb * 1_000_000 / prev_tempo)
            return prev_tick + extra_ticks
        elapsed  += seg_secs
        prev_tick = t
        prev_tempo = tempo
    # Más allá del último cambio
    remaining = secs - elapsed
    extra_ticks = int(remaining * tpb * 1_000_000 / prev_tempo)
    return prev_tick + extra_ticks


def get_midi_duration_seconds(mid: MidiFile, tempo_map: list) -> float:
    """Duración total del MIDI en segundos."""
    max_tick = 0
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
        max_tick = max(max_tick, abs_tick)
    return ticks_to_seconds(max_tick, mid.ticks_per_beat, tempo_map)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE SEGMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def build_segments(anchors: list, src_duration: float, dst_duration: float) -> list:
    """Construye la lista de segmentos a partir de los anclajes.

    Devuelve lista de dicts:
      src_start, src_end  — tiempos de origen en segundos
      dst_start, dst_end  — tiempos de destino en segundos
      ratio               — factor de escala temporal (dst_dur / src_dur)
    """
    # Añadir anclaje final implícito
    extended = list(anchors) + [(src_duration, dst_duration)]

    segments = []
    for i in range(len(extended) - 1):
        src_start = extended[i][0]
        dst_start = extended[i][1]
        src_end   = extended[i + 1][0]
        dst_end   = extended[i + 1][1]
        src_dur   = src_end - src_start
        dst_dur   = dst_end - dst_start

        if src_dur <= 0:
            ratio = 1.0
        else:
            ratio = dst_dur / src_dur

        segments.append({
            'src_start': src_start,
            'src_end':   src_end,
            'dst_start': dst_start,
            'dst_end':   dst_end,
            'src_dur':   src_dur,
            'dst_dur':   dst_dur,
            'ratio':     ratio,
        })

    return segments


# ══════════════════════════════════════════════════════════════════════════════
#  MODO TEMPO — cambio de BPM por segmento
# ══════════════════════════════════════════════════════════════════════════════

def align_tempo_mode(mid: MidiFile, segments: list,
                     tempo_map: list, verbose: bool) -> MidiFile:
    """Alineación por cambio de BPM.

    Inserta eventos set_tempo al inicio de cada segmento escalando el
    tempo original por el ratio del segmento. El ritmo interno se preserva.
    """
    tpb = mid.ticks_per_beat
    out = MidiFile(ticks_per_beat=tpb)

    # Construir mapa de tempo escalado: tick_src → tempo_nuevo
    # Para cada segmento, todos los tempos dentro del segmento se multiplican
    # por 1/ratio (más rápido = ratio > 1 → tempo_us menor)
    scaled_tempos = {}   # tick_src → tempo_us_escalado

    for seg in segments:
        src_start_tick = seconds_to_ticks(seg['src_start'], tpb, tempo_map)
        src_end_tick   = seconds_to_ticks(seg['src_end'],   tpb, tempo_map)
        ratio          = seg['ratio']

        # Segmento con duración destino cero (cola que desaparece) — omitir
        if ratio <= 0 or seg['dst_dur'] <= 0:
            continue

        for (t, tempo) in tempo_map:
            if src_start_tick <= t < src_end_tick:
                scaled_tempos[t] = max(1, int(tempo * ratio))

        # Asegurar que el inicio del segmento tiene un tempo definido
        # tomando el tempo vigente en src_start y escalándolo
        seg_tempo = 500000
        for (t, tempo) in tempo_map:
            if t <= src_start_tick:
                seg_tempo = tempo
        scaled_tempos[src_start_tick] = max(1, int(seg_tempo * ratio))

    if verbose:
        print("\n  Tempos escalados por segmento:")
        for seg in segments:
            if seg['ratio'] <= 0 or seg['dst_dur'] <= 0:
                print(f"    {format_time(seg['src_start'])} → "
                      f"{format_time(seg['dst_start'])}  "
                      f"ratio=0.0000  (segmento eliminado — cola sin espacio destino)")
                continue
            tick = seconds_to_ticks(seg['src_start'], tpb, tempo_map)
            t_us = scaled_tempos.get(tick, 500000)
            bpm  = round(60_000_000 / t_us, 1)
            print(f"    {format_time(seg['src_start'])} → "
                  f"{format_time(seg['dst_start'])}  "
                  f"ratio={seg['ratio']:.4f}  BPM={bpm}")

    # Reconstruir tracks con los nuevos tempos
    # Calcular el tick de corte — fin del último segmento con ratio > 0
    cutoff_tick = None
    for seg in reversed(segments):
        if seg['ratio'] > 0 and seg['dst_dur'] > 0:
            cutoff_tick = seconds_to_ticks(seg['src_end'], tpb, tempo_map)
            break

    for track in mid.tracks:
        new_track = MidiTrack()
        abs_tick  = 0
        prev_out_tick = 0

        for msg in track:
            abs_tick += msg.time

            # Truncar eventos más allá del corte
            if cutoff_tick is not None and abs_tick > cutoff_tick:
                break

            if msg.type == 'set_tempo':
                new_tempo = scaled_tempos.get(abs_tick, msg.tempo)
                out_tick = abs_tick
                delta = max(0, out_tick - prev_out_tick)
                new_track.append(MetaMessage('set_tempo', tempo=new_tempo, time=delta))
                prev_out_tick = out_tick
            else:
                out_tick = abs_tick
                delta = max(0, out_tick - prev_out_tick)
                new_track.append(msg.copy(time=delta))
                prev_out_tick = out_tick

        out.tracks.append(new_track)

    # Segundo paso: insertar set_tempo en los ticks de inicio de cada segmento
    # en el primer track (track 0, habitualmente el de tempo)
    if out.tracks:
        tempo_track = out.tracks[0]
        for seg in segments:
            tick     = seconds_to_ticks(seg['src_start'], tpb, tempo_map)
            new_tempo = scaled_tempos.get(tick, 500000)
            # Verificar si ya hay un set_tempo en ese tick
            abs_t = 0
            found = False
            for msg in tempo_track:
                abs_t += msg.time
                if abs_t == tick and msg.type == 'set_tempo':
                    found = True
                    break
            if not found:
                # Insertar en posición correcta
                _insert_tempo_event(tempo_track, tick, new_tempo)

    return out


def _insert_tempo_event(track: MidiTrack, target_tick: int, tempo: int):
    """Inserta un MetaMessage set_tempo en el tick correcto del track."""
    events = []
    abs_t  = 0
    for msg in track:
        abs_t += msg.time
        events.append((abs_t, msg))

    # Añadir el nuevo evento
    events.append((target_tick, MetaMessage('set_tempo', tempo=tempo, time=0)))
    events.sort(key=lambda x: (x[0], 0 if x[1].type == 'set_tempo' else 1))

    # Reconstruir con deltas
    track.clear() if hasattr(track, 'clear') else track.__init__()
    prev = 0
    for (t, msg) in events:
        delta = max(0, t - prev)
        track.append(msg.copy(time=delta))
        prev = t


# ══════════════════════════════════════════════════════════════════════════════
#  MODO LINEAR — redistribución proporcional de notas
# ══════════════════════════════════════════════════════════════════════════════

def align_linear_mode(mid: MidiFile, segments: list,
                      tempo_map: list, verbose: bool) -> MidiFile:
    """Alineación por redistribución proporcional.

    Mapea cada evento de su posición temporal en segundos (src) a su
    nueva posición en segundos (dst), luego convierte de vuelta a ticks
    usando el tempo de referencia constante.
    """
    tpb = mid.ticks_per_beat
    ref_tempo = 500000  # tempo fijo de referencia para el MIDI de salida

    def src_secs_to_dst_secs(t_src: float) -> float:
        """Mapea un tiempo fuente a tiempo destino usando los segmentos."""
        for seg in segments:
            if seg['src_start'] <= t_src <= seg['src_end']:
                if seg['src_dur'] <= 0:
                    return seg['dst_start']
                frac = (t_src - seg['src_start']) / seg['src_dur']
                return seg['dst_start'] + frac * seg['dst_dur']
        # Más allá del último segmento — mantener desplazamiento del último
        last = segments[-1]
        overflow = t_src - last['src_end']
        return last['dst_end'] + overflow

    def dst_secs_to_ticks(t_dst: float) -> int:
        return int(t_dst * 1_000_000 / ref_tempo * tpb)

    if verbose:
        print("\n  Redistribución lineal por segmento:")
        for seg in segments:
            print(f"    {format_time(seg['src_start'])}–{format_time(seg['src_end'])} → "
                  f"{format_time(seg['dst_start'])}–{format_time(seg['dst_end'])}  "
                  f"ratio={seg['ratio']:.4f}")

    out = MidiFile(ticks_per_beat=tpb)

    # Track de tempo único con tempo constante de referencia
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage('track_name', name='Tempo', time=0))
    tempo_track.append(MetaMessage('set_tempo', tempo=ref_tempo, time=0))
    tempo_track.append(MetaMessage('end_of_track', time=0))
    out.tracks.append(tempo_track)

    for ti, track in enumerate(mid.tracks):
        new_track = MidiTrack()

        # Recopilar eventos con tiempo absoluto en segundos
        events_secs = []
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.is_meta and msg.type == 'set_tempo':
                continue   # los tempos se gestionan en el track de tempo
            t_secs = ticks_to_seconds(abs_tick, tpb, tempo_map)
            events_secs.append((t_secs, msg))

        # Mapear a tiempo destino y convertir a ticks con tempo fijo
        events_out = []
        for (t_src, msg) in events_secs:
            t_dst  = src_secs_to_dst_secs(t_src)
            t_tick = dst_secs_to_ticks(t_dst)
            events_out.append((t_tick, msg))

        events_out.sort(key=lambda x: (x[0], 0 if x[1].is_meta else 1))

        # Reconstruir con deltas
        prev_tick = 0
        for (t_tick, msg) in events_out:
            delta = max(0, t_tick - prev_tick)
            new_track.append(msg.copy(time=delta))
            prev_tick = t_tick

        new_track.append(MetaMessage('end_of_track', time=0))
        out.tracks.append(new_track)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

def show_preview(segments: list, src_duration: float, dst_duration: float):
    """Muestra el mapa de segmentos sin generar el MIDI."""
    print(f"\n── PREVIEW ──────────────────────────────────────────────────")
    print(f"  Duración fuente  : {format_time(src_duration)}")
    print(f"  Duración destino : {format_time(dst_duration)}")
    print(f"  Segmentos        : {len(segments)}")
    print()
    print(f"  {'Origen':>12}  {'→':1}  {'Destino':>12}  "
          f"{'Dur src':>9}  {'Dur dst':>9}  {'Ratio':>7}")
    print(f"  {'─'*12}  {'─':1}  {'─'*12}  "
          f"{'─'*9}  {'─'*9}  {'─'*7}")
    for i, seg in enumerate(segments):
        delta = seg['dst_dur'] - seg['src_dur']
        sign  = '+' if delta >= 0 else ''
        print(f"  {format_time(seg['src_start']):>12}  →  "
              f"{format_time(seg['dst_start']):>12}  "
              f"{format_time(seg['src_dur']):>9}  "
              f"{format_time(seg['dst_dur']):>9}  "
              f"{seg['ratio']:>7.4f}  ({sign}{format_time(abs(delta))})")
    print(f"  {'─'*70}")
    total_src = sum(s['src_dur'] for s in segments)
    total_dst = sum(s['dst_dur'] for s in segments)
    print(f"  {'TOTAL':>12}     {'':>12}  "
          f"{format_time(total_src):>9}  {format_time(total_dst):>9}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Alineación temporal de MIDIs mediante puntos de anclaje",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("midi",
                   help="MIDI de entrada a alinear")
    p.add_argument("--anchors", nargs="+", metavar="'T T'",
                   help="Pares de tiempos origen→destino (M:SS.mmm M:SS.mmm)")
    p.add_argument("--anchors-file", metavar="FILE",
                   help="Archivo de texto con anclajes (uno por línea)")
    p.add_argument("--ref", metavar="MIDI",
                   help="MIDI de referencia para obtener la duración destino total")
    p.add_argument("--mode", default="tempo",
                   choices=["tempo", "linear"],
                   help="Modo de alineación: tempo (BPM por segmento) | "
                        "linear (redistribución proporcional) (default: tempo)")
    p.add_argument("--output", default=None, metavar="FILE",
                   help="MIDI de salida (default: <input>.aligned.mid)")
    p.add_argument("--preview", action="store_true",
                   help="Mostrar mapa de segmentos sin generar MIDI")
    p.add_argument("--verbose", action="store_true",
                   help="Informe detallado")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Cargar MIDI fuente ───────────────────────────────────────────────────
    if not os.path.isfile(args.midi):
        print(f"ERROR: no se encuentra el archivo: {args.midi}")
        sys.exit(1)
    try:
        mid = MidiFile(args.midi)
    except Exception as e:
        print(f"ERROR al leer el MIDI: {e}")
        sys.exit(1)

    tpb       = mid.ticks_per_beat
    tempo_map = get_tempo_map(mid)
    src_dur   = get_midi_duration_seconds(mid, tempo_map)

    if args.verbose:
        print(f"\n[INFO] Fuente     : {args.midi}")
        print(f"[INFO] TPB        : {tpb}")
        print(f"[INFO] Duración   : {format_time(src_dur)}")
        print(f"[INFO] Tracks     : {len(mid.tracks)}")
        print(f"[INFO] Modo       : {args.mode}")

    # ── Cargar anclajes ──────────────────────────────────────────────────────
    if args.anchors_file:
        anchors = load_anchors_file(args.anchors_file)
    elif args.anchors:
        anchors = parse_anchors(args.anchors)
    else:
        print("ERROR: debes especificar --anchors o --anchors-file")
        sys.exit(1)

    if not anchors:
        print("ERROR: no se encontraron anclajes válidos")
        sys.exit(1)

    # Validar primer anclaje
    if anchors[0][0] != 0.0:
        print(f"AVISO: el primer anclaje no empieza en 0:00.000 "
              f"(empieza en {format_time(anchors[0][0])}). "
              f"Se asume 0:00.000 → 0:00.000 implícito.")
        anchors = [(0.0, 0.0)] + anchors

    # ── Duración destino ─────────────────────────────────────────────────────
    if args.ref:
        if not os.path.isfile(args.ref):
            print(f"ERROR: no se encuentra el MIDI de referencia: {args.ref}")
            sys.exit(1)
        try:
            ref_mid      = MidiFile(args.ref)
            ref_tempo_map = get_tempo_map(ref_mid)
            dst_dur      = get_midi_duration_seconds(ref_mid, ref_tempo_map)
        except Exception as e:
            print(f"ERROR al leer el MIDI de referencia: {e}")
            sys.exit(1)
        if args.verbose:
            print(f"[INFO] Referencia : {args.ref}")
            print(f"[INFO] Dur. dest. : {format_time(dst_dur)}")
    else:
        # Sin referencia: el último anclaje destino define el fin del MIDI.
        # El último segmento del fuente (cola tras el último anclaje) se estira
        # proporcionalmente para llegar exactamente al tiempo destino final.
        dst_dur = anchors[-1][1]
        if args.verbose:
            print(f"[INFO] Dur. dest. : {format_time(dst_dur)} (inferida del último anclaje)")

    # ── Construir segmentos ──────────────────────────────────────────────────
    segments = build_segments(anchors, src_dur, dst_dur)

    # ── Preview ──────────────────────────────────────────────────────────────
    if args.preview:
        show_preview(segments, src_dur, dst_dur)
        return

    if args.verbose:
        show_preview(segments, src_dur, dst_dur)

    # ── Alineación ───────────────────────────────────────────────────────────
    if args.mode == "tempo":
        out_mid = align_tempo_mode(mid, segments, tempo_map, args.verbose)
    else:
        out_mid = align_linear_mode(mid, segments, tempo_map, args.verbose)

    # ── Guardar ──────────────────────────────────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(args.midi)[0]
        out_path = f"{base}.aligned.mid"

    try:
        out_mid.save(out_path)
        print(f"MIDI alineado guardado en: {out_path}")
    except Exception as e:
        print(f"ERROR al guardar el MIDI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
