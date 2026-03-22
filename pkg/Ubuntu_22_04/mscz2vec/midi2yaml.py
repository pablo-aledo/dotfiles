#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MIDI2YAML  v1.0                                     ║
║         Convierte fragmentos MIDI a sintaxis de partitura.yaml              ║
║                                                                              ║
║  Lee un MIDI generado por cualquier herramienta del ecosistema y produce    ║
║  bloques YAML listos para insertar en partitura.yaml: melodia, armonia      ║
║  y/o marcato, en la sintaxis [compas, beat, duracion, nota, velocidad].     ║
║                                                                              ║
║  USO:                                                                        ║
║    python midi2yaml.py fragmento.mid                                         ║
║    python midi2yaml.py fragmento.mid --compas-inicio 15                     ║
║    python midi2yaml.py fragmento.mid --modo melodia                         ║
║    python midi2yaml.py fragmento.mid --modo armonia                         ║
║    python midi2yaml.py fragmento.mid --modo todos                           ║
║    python midi2yaml.py fragmento.mid --compas-inicio 9 --bpm 72            ║
║    python midi2yaml.py fragmento.mid --track 0                              ║
║    python midi2yaml.py fragmento.mid --umbral-silencio 0.25                ║
║    python midi2yaml.py fragmento.mid --redondear 0.125                      ║
║    python midi2yaml.py fragmento.mid --compas-inicio 15 --preview          ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --compas-inicio N   Compás real de inicio en la partitura (default: 1)  ║
║    --bpm N             Tempo en BPM (default: detectado del MIDI)           ║
║    --compas 4/4        Compás de la obra (default: 4/4)                    ║
║    --modo M            Qué extraer: melodia | armonia | marcato | todos     ║
║                        (default: melodia)                                   ║
║    --track N           Track MIDI a usar como melodía (default: auto)      ║
║    --umbral-silencio F Duración mínima de nota en beats (default: 0.0625)  ║
║    --redondear F       Cuantizar beats a múltiplo de F (default: 0.25)     ║
║    --preview           Mostrar estructura sin generar YAML                  ║
║    --verbose           Informe detallado del análisis                       ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    Imprime por stdout el bloque YAML listo para pegar en partitura.yaml.   ║
║    Formato melodia:  [compas, beat, duracion, nota, velocidad]              ║
║    Formato armonia:  {compas: N, acorde: X, grado: "?", funcion: auto}     ║
║    Formato marcato:  notas_bajo: [nota, ...]  (una por compás)             ║
║                                                                              ║
║  NOTAS DE USO:                                                               ║
║    · Los nombres de nota usan la convención del ecosistema:                 ║
║      C4, Cs4 (Do#4), Ds3 (Re#3), Bb3 (Sib3), Fs4 (Fa#4)                  ║
║    · Con --modo todos genera los tres bloques: melodia + armonia + marcato ║
║    · El track con más notas se usa como melodía; el resto como armonía     ║
║    · Usa --track para forzar un track concreto como melodía                ║
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
    from mido import MidiFile
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)


TPB_REF = 480   # ticks per beat de referencia


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN DE NOTAS
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES  = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
FLAT_NAMES  = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Pitch classes que suenan mejor con bemol en contexto menor (Re menor)
PREFER_FLAT = {1, 3, 6, 8, 10}   # Db Eb Gb Ab Bb


def midi_to_note(n: int, usar_bemoles: bool = True) -> str:
    """Convierte número MIDI a nombre en convención del ecosistema.

    Ejemplos: 50→D3, 61→Cs4, 58→Bb3, 66→Fs4
    Usa bemoles para alteraciones típicas de tonalidades menores.
    """
    pc = n % 12
    octave = (n // 12) - 1
    if usar_bemoles and pc in PREFER_FLAT:
        return f"{FLAT_NAMES[pc]}{octave}"
    return f"{NOTE_NAMES[pc]}{octave}"


def ticks_to_beats(ticks: int, tpb: int) -> float:
    """Convierte ticks MIDI a beats."""
    return ticks / tpb


def redondear_beat(beat: float, resolucion: float) -> float:
    """Cuantiza un valor de beat al múltiplo más cercano de resolucion."""
    if resolucion <= 0:
        return round(beat, 4)
    return round(round(beat / resolucion) * resolucion, 4)


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def detectar_tempo(mid: MidiFile) -> float:
    """Extrae el primer tempo del MIDI. Devuelve BPM."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return mido.tempo2bpm(msg.tempo)
    return 120.0


def extraer_notas_track(track, tpb: int) -> list:
    """Extrae notas con tiempo absoluto en ticks de un track MIDI.

    Devuelve lista de (tick_on, tick_off, pitch, velocity).
    """
    abs_tick = 0
    activas  = {}   # pitch → (tick_on, velocity)
    notas    = []

    for msg in track:
        abs_tick += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            activas[msg.note] = (abs_tick, msg.velocity)
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in activas:
                t_on, vel = activas.pop(msg.note)
                notas.append((t_on, abs_tick, msg.note, vel))

    # Cerrar notas que quedaron abiertas al final del track
    for pitch, (t_on, vel) in activas.items():
        notas.append((t_on, abs_tick + tpb, pitch, vel))

    return sorted(notas, key=lambda x: x[0])


def elegir_track_melodia(mid: MidiFile, track_forzado: int = None) -> int:
    """Elige el track con más notas como melodía, o usa track_forzado."""
    if track_forzado is not None:
        return track_forzado
    max_notas = -1
    idx = 0
    for i, track in enumerate(mid.tracks):
        n = sum(1 for msg in track if msg.type == 'note_on' and msg.velocity > 0)
        if n > max_notas:
            max_notas = n
            idx = i
    return idx


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN A COMPÁS / BEAT
# ══════════════════════════════════════════════════════════════════════════════

def tick_a_compas_beat(tick: int, tpb: int, beats_por_compas: float,
                        compas_inicio: int) -> tuple:
    """Convierte tick absoluto a (compas_real, beat).

    compas_inicio: número de compás real en la partitura (ej: 15).
    beat: 0.0 = primer tiempo del compás.
    """
    beats_totales  = tick / tpb
    compas_relativo = int(beats_totales // beats_por_compas)
    beat            = beats_totales % beats_por_compas
    compas_real     = compas_inicio + compas_relativo
    return compas_real, beat


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCIA DE ACORDES
# ══════════════════════════════════════════════════════════════════════════════

CHORD_TEMPLATES = {
    'Dm':  {2, 5, 9},
    'C':   {0, 4, 7},
    'Gm':  {7, 10, 2},
    'Bb':  {10, 2, 5},
    'A':   {9, 1, 4},
    'A7':  {9, 1, 4, 7},
    'Dm7': {2, 5, 9, 0},
    'Eo7': {4, 7, 10, 2},
    'Am':  {9, 0, 4},
    'F':   {5, 9, 0},
    'Gm7': {7, 10, 2, 5},
    'E':   {4, 8, 11},
}


def inferir_acorde(pcs: set) -> str:
    """Infiere el acorde más probable dado un conjunto de pitch classes."""
    if not pcs:
        return '?'
    mejor       = '?'
    mejor_score = -1
    for nombre, template in CHORD_TEMPLATES.items():
        interseccion = len(pcs & template)
        score = interseccion / max(len(template), len(pcs))
        if score > mejor_score:
            mejor_score = score
            mejor       = nombre
    return mejor if mejor_score > 0.4 else '?'


def extraer_acordes_por_compas(tracks_armonia: list, tpb: int,
                                beats_por_compas: float,
                                compas_inicio: int,
                                n_compases: int) -> list:
    """Agrupa notas por compás e infiere el acorde de cada uno."""
    notas_por_compas = {
        c: set()
        for c in range(compas_inicio, compas_inicio + n_compases)
    }
    for track in tracks_armonia:
        notas = extraer_notas_track(track, tpb)
        for (t_on, t_off, pitch, vel) in notas:
            compas, _ = tick_a_compas_beat(t_on, tpb, beats_por_compas,
                                            compas_inicio)
            if compas in notas_por_compas:
                notas_por_compas[compas].add(pitch % 12)

    acordes = []
    for compas in sorted(notas_por_compas):
        pcs    = notas_por_compas[compas]
        acorde = inferir_acorde(pcs)
        acordes.append({'compas': compas, 'acorde': acorde})
    return acordes


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES DE BLOQUES YAML
# ══════════════════════════════════════════════════════════════════════════════

def generar_bloque_melodia(notas: list, tpb: int, beats_por_compas: float,
                            compas_inicio: int, umbral: float,
                            resolucion: float, verbose: bool) -> str:
    """Genera el bloque 'melodia:' en sintaxis YAML del ecosistema."""
    lineas = ["        melodia:"]
    compas_actual = None

    for (t_on, t_off, pitch, vel) in notas:
        dur_beats = ticks_to_beats(t_off - t_on, tpb)
        if dur_beats < umbral:
            if verbose:
                print(f"  [omitida] {midi_to_note(pitch)} "
                      f"dur={dur_beats:.3f} < umbral={umbral}", file=sys.stderr)
            continue

        compas, beat = tick_a_compas_beat(t_on, tpb, beats_por_compas,
                                           compas_inicio)
        beat_r  = redondear_beat(beat, resolucion)
        dur_r   = redondear_beat(dur_beats, resolucion)
        dur_r   = max(resolucion, dur_r)   # duración mínima = resolución
        nota_str = midi_to_note(pitch)

        if compas != compas_actual:
            lineas.append(f"          # c.{compas}")
            compas_actual = compas

        lineas.append(
            f"          - [{compas}, {beat_r}, {dur_r}, {nota_str}, {vel}]"
        )

        if verbose:
            print(f"  c.{compas} beat={beat_r} dur={dur_r} "
                  f"{nota_str} vel={vel}", file=sys.stderr)

    return "\n".join(lineas)


def generar_bloque_armonia(acordes: list) -> str:
    """Genera el bloque 'armonia:' en sintaxis YAML del ecosistema."""
    lineas = ["        armonia:"]
    for item in acordes:
        c = item['compas']
        a = item['acorde']
        lineas.append(
            f"          - {{compas: {c}, acorde: {a}, "
            f"grado: \"?\", funcion: auto}}"
        )
    return "\n".join(lineas)


def generar_bloque_marcato(notas: list, tpb: int, beats_por_compas: float,
                            compas_inicio: int, n_compases: int) -> str:
    """Genera el campo 'notas_bajo:' del marcato — una nota grave por compás."""
    notas_por_compas = {}
    for (t_on, t_off, pitch, vel) in notas:
        compas, _ = tick_a_compas_beat(t_on, tpb, beats_por_compas,
                                        compas_inicio)
        if compas not in notas_por_compas or pitch < notas_por_compas[compas]:
            notas_por_compas[compas] = pitch

    notas_bajo = []
    for c in range(compas_inicio, compas_inicio + n_compases):
        if c in notas_por_compas:
            notas_bajo.append(midi_to_note(notas_por_compas[c]))
        else:
            notas_bajo.append('D2')   # fallback: tónica de Re menor

    return f"          notas_bajo: [{', '.join(notas_bajo)}]"


# ══════════════════════════════════════════════════════════════════════════════
#  PREVIEW
# ══════════════════════════════════════════════════════════════════════════════

def mostrar_preview(mid: MidiFile, bpm: float, num: int, den: int,
                     beats_por_compas: float, track_melodia: int) -> None:
    """Muestra un resumen del MIDI sin generar YAML."""
    tpb = mid.ticks_per_beat
    total_ticks = max(
        (sum(msg.time for msg in track) for track in mid.tracks),
        default=0
    )
    total_beats   = total_ticks / tpb
    total_compases = max(1, math.ceil(total_beats / beats_por_compas))

    print(f"\n── PREVIEW ──────────────────────────────────────────────")
    print(f"  Tracks      : {len(mid.tracks)}")
    print(f"  TPB         : {tpb}")
    print(f"  Tempo       : {bpm:.1f} BPM")
    print(f"  Compás      : {num}/{den}  ({beats_por_compas} beats/compás)")
    print(f"  Duración    : ~{total_compases} compases / {total_beats:.1f} beats")
    print(f"  Track mel.  : {track_melodia}")
    print()
    for i, track in enumerate(mid.tracks):
        n_notas = sum(
            1 for m in track if m.type == 'note_on' and m.velocity > 0
        )
        marca = " ← melodía" if i == track_melodia else ""
        nombre_track = track.name or "(sin nombre)"
        print(f"  Track {i:2d}: {nombre_track:24s}  {n_notas:4d} notas{marca}")
    print(f"────────────────────────────────────────────────────────\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convierte un MIDI a sintaxis de partitura.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("midi",
                   help="Ruta al archivo MIDI de entrada")
    p.add_argument("--compas-inicio", type=int, default=1, metavar="N",
                   help="Compás real de inicio en la partitura (default: 1)")
    p.add_argument("--bpm", type=float, default=None,
                   help="Tempo en BPM (default: detectado del MIDI)")
    p.add_argument("--compas", default="4/4", metavar="N/N",
                   help="Compás de la obra, ej: 4/4  3/4 (default: 4/4)")
    p.add_argument("--modo", default="melodia",
                   choices=["melodia", "armonia", "marcato", "todos"],
                   help="Qué extraer (default: melodia)")
    p.add_argument("--track", type=int, default=None,
                   help="Índice del track a usar como melodía (default: auto)")
    p.add_argument("--umbral-silencio", type=float, default=0.0625,
                   metavar="F",
                   help="Duración mínima de nota en beats (default: 0.0625)")
    p.add_argument("--redondear", type=float, default=0.25, metavar="F",
                   help="Cuantizar beats a múltiplo de F (default: 0.25)")
    p.add_argument("--preview", action="store_true",
                   help="Mostrar estructura sin generar YAML")
    p.add_argument("--verbose", action="store_true",
                   help="Informe detallado del análisis")
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Cargar MIDI ──────────────────────────────────────────────
    if not os.path.isfile(args.midi):
        print(f"ERROR: no se encuentra el archivo: {args.midi}")
        sys.exit(1)

    try:
        mid = MidiFile(args.midi)
    except Exception as e:
        print(f"ERROR al leer el MIDI: {e}")
        sys.exit(1)

    tpb = mid.ticks_per_beat

    # ── Parámetros ───────────────────────────────────────────────
    bpm = args.bpm or detectar_tempo(mid)

    try:
        num_str, den_str = args.compas.split("/")
        num, den = int(num_str), int(den_str)
    except ValueError:
        print(f"ERROR: formato de compás no válido: '{args.compas}'  "
              f"(usar ej: 4/4  o  3/4)")
        sys.exit(1)

    beats_por_compas = num * (4.0 / den)
    track_melodia    = elegir_track_melodia(mid, args.track)

    # ── Preview ──────────────────────────────────────────────────
    if args.preview:
        mostrar_preview(mid, bpm, num, den, beats_por_compas, track_melodia)
        return

    # ── Extraer notas de melodía ─────────────────────────────────
    notas_mel = extraer_notas_track(mid.tracks[track_melodia], tpb)

    # Calcular número de compases totales
    if notas_mel:
        tick_max = max(t_off for (_, t_off, _, _) in notas_mel)
    else:
        tick_max = sum(
            sum(msg.time for msg in track) for track in mid.tracks
        )
    total_beats = ticks_to_beats(tick_max, tpb)
    n_compases  = max(1, math.ceil(total_beats / beats_por_compas))

    if args.verbose:
        print(f"\n[INFO] BPM={bpm:.1f}  compás={num}/{den}  "
              f"beats/compás={beats_por_compas}", file=sys.stderr)
        print(f"[INFO] track melodía={track_melodia}  "
              f"notas={len(notas_mel)}  compases≈{n_compases}",
              file=sys.stderr)

    # ── Tracks de armonía (todos menos el de melodía) ────────────
    tracks_arm = [t for i, t in enumerate(mid.tracks) if i != track_melodia]

    # ── Generar bloques según modo ───────────────────────────────
    bloques = []

    if args.modo in ("melodia", "todos"):
        bloques.append(generar_bloque_melodia(
            notas_mel, tpb, beats_por_compas,
            args.compas_inicio, args.umbral_silencio,
            args.redondear, args.verbose
        ))

    if args.modo in ("armonia", "todos"):
        fuentes_arm = tracks_arm if tracks_arm else [mid.tracks[track_melodia]]
        acordes = extraer_acordes_por_compas(
            fuentes_arm, tpb, beats_por_compas,
            args.compas_inicio, n_compases
        )
        bloques.append(generar_bloque_armonia(acordes))

    if args.modo in ("marcato", "todos"):
        fuentes_arm = tracks_arm if tracks_arm else [mid.tracks[track_melodia]]
        notas_arm   = []
        for t in fuentes_arm:
            notas_arm.extend(extraer_notas_track(t, tpb))
        notas_arm.sort(key=lambda x: x[0])
        bloques.append(generar_bloque_marcato(
            notas_arm, tpb, beats_por_compas,
            args.compas_inicio, n_compases
        ))

    # ── Cabecera informativa ─────────────────────────────────────
    nombre = os.path.basename(args.midi)
    print(f"        # ── generado por midi2yaml.py desde {nombre} ──")
    print(f"        # compas-inicio: {args.compas_inicio}  "
          f"bpm: {bpm:.1f}  modo: {args.modo}")
    print()
    print("\n\n".join(bloques))


if __name__ == "__main__":
    main()
