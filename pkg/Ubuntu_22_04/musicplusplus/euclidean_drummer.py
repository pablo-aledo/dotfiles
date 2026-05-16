#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    EUCLIDEAN DRUMMER  v1.0                                   ║
║         Generador de ritmos matemáticos para percusión MIDI                  ║
║                                                                              ║
║  Portado desde music++ / rhythmGen.h                                         ║
║                                                                              ║
║  Implementa cuatro algoritmos de generación rítmica:                         ║
║    euclidean    — Distribución uniforme de k pulsos en n pasos (Björklund)  ║
║    clough       — Patrón de Clough-Douthett: floor(i·n/k)                  ║
║    deep         — Deep rhythm por multiplicidad modular                      ║
║    tihai        — Estructura tihāī: frase repetida 3x que aterriza en sam  ║
║                                                                              ║
║  USO:                                                                        ║
║    python euclidean_drummer.py --steps 16 --events 5                        ║
║    python euclidean_drummer.py --steps 16 --events 5 --algo euclidean       ║
║    python euclidean_drummer.py --steps 12 --events 7 --algo clough          ║
║    python euclidean_drummer.py --steps 16 --events 6 --algo deep --mult 3  ║
║    python euclidean_drummer.py --steps 16 --reps 3 --algo tihai             ║
║    python euclidean_drummer.py --steps 16 --events 5 --output ritmo.mid    ║
║    python euclidean_drummer.py --steps 16 --events 5 --bars 8              ║
║    python euclidean_drummer.py --compare --steps 16 --events 5             ║
║    python euclidean_drummer.py --family --steps 16                          ║
║                                                                              ║
║  MODOS ESPECIALES:                                                           ║
║    --compare      Genera los 4 algoritmos en paralelo para comparar         ║
║    --family       Genera toda la familia E(k,n) para n fijo                 ║
║    --poly         Polirritmo: múltiples patrones superpuestos en capas      ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --steps N      Número de pasos del ciclo (default: 16)                   ║
║    --events K     Número de pulsos/onsets (default: 5)                      ║
║    --algo A       Algoritmo: euclidean|clough|deep|tihai (default: euclidean)║
║    --mult M       Multiplicidad para deep rhythm (default: auto-optima)     ║
║    --reps R       Repeticiones para tihai (default: 3)                      ║
║    --offset O     Desplazamiento del patrón (default: 0)                    ║
║    --bars N       Número de compases a generar (default: 4)                 ║
║    --tempo BPM    Tempo del MIDI (default: 120)                             ║
║    --note N       Nota MIDI del kick (default: 36 = bass drum)              ║
║    --velocity V   Velocidad MIDI (default: 100)                             ║
║    --output FILE  Exportar a MIDI                                            ║
║    --no-color     Sin colores ANSI                                           ║
║                                                                              ║
║  DEPENDENCIAS: mido (solo para --output)                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import math


# ─────────────────────────────────────────────────────────────────────────────
#  ANSI
# ─────────────────────────────────────────────────────────────────────────────

ANSI = {
    "reset":  "\033[0m", "bold":  "\033[1m",  "green": "\033[32m",
    "yellow": "\033[33m","cyan":  "\033[36m",  "gray":  "\033[90m",
    "red":    "\033[31m","blue":  "\033[34m",  "magenta":"\033[35m",
}
USE_COLOR = True

def c(key: str, text: str) -> str:
    return f"{ANSI[key]}{text}{ANSI['reset']}" if USE_COLOR else text


# ─────────────────────────────────────────────────────────────────────────────
#  DIVISIÓN EUCLÍDEA
#  Portado de mathUtil.h — euclideanDivision()
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_division(dividend: int, divisor: int) -> tuple[int, int]:
    """
    División euclídea con resto siempre no-negativo.
    Portado de euclideanDivision() en mathUtil.h.
    """
    quotient = dividend // divisor
    remainder = dividend - quotient * divisor
    if remainder < 0:
        return quotient - 1, remainder + divisor
    return quotient, remainder


# ─────────────────────────────────────────────────────────────────────────────
#  ALGORITMO EUCLÍDEO (BJÖRKLUND)
#  Portado de euclidean() en rhythmGen.h
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_intervals(steps: int, events: int) -> list[int]:
    """
    Genera el ritmo euclídeo E(events, steps) como lista de intervalos.
    Portado recursivo de euclidean() en rhythmGen.h.
    
    Ejemplo: euclidean(16, 5) → [3, 3, 3, 4, 3]  (clave tresillo típica)
    """
    if events <= 0:
        return []
    if events > steps:
        events = steps
    quotient, remainder = euclidean_division(steps, events)
    if remainder == 0:
        return [quotient] * events
    else:
        a = remainder
        sub = euclidean_intervals(events, a)
        out = []
        for i in range(a):
            for _ in range(sub[i] - 1):
                out.append(quotient)
            out.append(quotient + 1)
        return out


def euclidean_positions(steps: int, events: int, offset: int = 0) -> list[int]:
    """
    Convierte intervalos euclídeos a posiciones (0-based).
    """
    intervals = euclidean_intervals(steps, events)
    positions = []
    pos = offset % steps
    for interval in intervals:
        positions.append(pos % steps)
        pos += interval
    return sorted(set(p % steps for p in positions))


def euclidean_binary(steps: int, events: int, offset: int = 0) -> list[int]:
    """
    Devuelve el ritmo euclídeo como vector binario (0=silencio, 1=onset).
    """
    positions = set(euclidean_positions(steps, events, offset))
    return [1 if i in positions else 0 for i in range(steps)]


# ─────────────────────────────────────────────────────────────────────────────
#  ALGORITMO CLOUGH-DOUTHETT
#  Portado de CloughDouthett() en rhythmGen.h
# ─────────────────────────────────────────────────────────────────────────────

def clough_douthett_positions(steps: int, events: int, offset: int = 0) -> list[int]:
    """
    Genera el patrón de Clough-Douthett como posiciones.
    Fórmula: floor(i * steps / events) para i = 0..events-1
    Portado de CloughDouthett() en rhythmGen.h.
    """
    if events <= 0:
        return []
    positions = []
    for i in range(events):
        pos = int(math.floor(i * steps / events))
        positions.append((pos + offset) % steps)
    return sorted(set(positions))


def clough_douthett_binary(steps: int, events: int, offset: int = 0) -> list[int]:
    positions = set(clough_douthett_positions(steps, events, offset))
    return [1 if i in positions else 0 for i in range(steps)]


# ─────────────────────────────────────────────────────────────────────────────
#  DEEP RHYTHM
#  Portado de deepRhythm() en rhythmGen.h
# ─────────────────────────────────────────────────────────────────────────────

def best_multiplicity(steps: int, events: int) -> int:
    """
    Encuentra la multiplicidad que genera el deep rhythm más uniforme.
    Un buen multiplicador es primo relativo con steps y produce
    la distribución más equidistante.
    """
    from math import gcd
    best_m = 1
    best_score = float("inf")
    for m in range(1, steps):
        if gcd(m, steps) != 1:
            continue
        positions = sorted(set((i * m) % steps for i in range(events)))
        if len(positions) != events:
            continue
        # medir uniformidad: varianza de intervalos
        intervals_check = []
        for j in range(len(positions)):
            diff = (positions[(j+1) % len(positions)] - positions[j]) % steps
            intervals_check.append(diff)
        variance = sum((x - steps/events)**2 for x in intervals_check) / len(intervals_check)
        if variance < best_score:
            best_score = variance
            best_m = m
    return best_m


def deep_rhythm_positions(steps: int, events: int, multiplicity: int | None = None, offset: int = 0) -> list[int]:
    """
    Genera el deep rhythm como posiciones.
    Portado de deepRhythm() en rhythmGen.h.
    Fórmula: (i * multiplicity) % steps  para i = 0..events-1, luego ordenado.
    """
    if multiplicity is None:
        multiplicity = best_multiplicity(steps, events)
    positions = []
    for i in range(events):
        positions.append((i * multiplicity + offset) % steps)
    return sorted(set(positions))


def deep_rhythm_binary(steps: int, events: int, multiplicity: int | None = None, offset: int = 0) -> list[int]:
    positions = set(deep_rhythm_positions(steps, events, multiplicity, offset))
    return [1 if i in positions else 0 for i in range(steps)]


# ─────────────────────────────────────────────────────────────────────────────
#  TIHAI
#  Portado de tihai(), tihaiGenerator(), tihaiReader() en rhythmGen.h
# ─────────────────────────────────────────────────────────────────────────────

def tihai_generator(steps: int, repetitions: int) -> tuple[int, int]:
    """
    Calcula (bols, dams) para un patrón tihai.
    Portado de tihaiGenerator() en rhythmGen.h.
    
    El tihai es una frase de 'bols' onsets + 'dams' silencios, repetida
    'repetitions' veces, aterrizando exactamente en el sam (primer tiempo).
    
    bols + dams = length/repetitions
    dams = length - steps  (número de pasos extra para redondear)
    """
    length = steps
    while length % repetitions != 0:
        length += 1
    dams = length - steps
    bols = length // repetitions - dams
    return bols, dams


def tihai_reader(bols: int, dams: int, repetitions: int, steps: int) -> list[int]:
    """
    Construye el vector binario del tihai.
    Portado de tihaiReader() en rhythmGen.h.
    """
    pattern = []
    # primera repetición
    pattern.extend([1] * bols)
    pattern.extend([0] * dams)
    # repeticiones intermedias
    for _ in range(repetitions - 2):
        pattern.extend([1] * bols)
        pattern.extend([0] * dams)
    # última repetición (sin dam final)
    pattern.extend([1] * bols)
    return pattern


def tihai_binary(steps: int, repetitions: int = 3, pseudo: bool = True) -> list[int]:
    """
    Genera el tihai como vector binario.
    Portado de tihai() en rhythmGen.h.
    
    Si pseudo=True y el resultado es trivial (todo 0 o todo 1),
    intenta con steps-1 recursivamente y rellena con 1s.
    """
    if steps <= 2:
        return [1] * steps
    if repetitions == 1:
        return [1] * steps
    if repetitions <= 0:
        return [0] * steps

    bols, dams = tihai_generator(steps, repetitions)
    pattern = tihai_reader(bols, dams, repetitions, steps)

    # recortar o rellenar
    if len(pattern) > steps:
        pattern = pattern[:steps]

    all_zeros = all(x == 0 for x in pattern)
    all_ones  = all(x == 1 for x in pattern)

    if pseudo and (all_zeros or all_ones):
        shorter = tihai_binary(steps - 1, repetitions, pseudo)
        while len(shorter) < steps:
            shorter.append(1)
        return shorter[:steps]

    return pattern


def tihai_positions(steps: int, repetitions: int = 3, pseudo: bool = True, offset: int = 0) -> list[int]:
    """Posiciones de los onsets del tihai."""
    binary = tihai_binary(steps, repetitions, pseudo)
    return sorted((i + offset) % steps for i, v in enumerate(binary) if v == 1)


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORTACIÓN A MIDI
# ─────────────────────────────────────────────────────────────────────────────

def export_midi(
    patterns: list[tuple[str, list[int]]],   # [(name, binary_pattern), …]
    steps: int,
    bars: int,
    tempo_bpm: int,
    output: str,
    note_map: dict[str, int] | None = None,
    velocity: int = 100,
):
    """
    Exporta uno o varios patrones rítmicos a MIDI.
    Cada patrón va en un canal/pista diferente.
    
    note_map: mapping nombre_patrón → nota MIDI.
    Default: todos los patrones van a canal 9 (percusión) con sus notas asignadas.
    """
    try:
        import mido
    except ImportError:
        print(c("red", "Error: mido no instalado. pip install mido"), file=sys.stderr)
        sys.exit(1)

    if note_map is None:
        drum_notes = [36, 38, 42, 46, 49, 51, 45, 47]
        note_map = {name: drum_notes[i % len(drum_notes)] for i, (name, _) in enumerate(patterns)}

    mid = mido.MidiFile(ticks_per_beat=480)
    mid.tracks.append(mido.MidiTrack())  # track de tempo
    mid.tracks[0].append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    ticks_per_step = 480 // 4  # asume que step = semicorchea

    for name, binary in patterns:
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        note = note_map.get(name, 36)

        events = []  # (tick_absoluto, tipo, nota, velocidad)
        for bar in range(bars):
            for step, val in enumerate(binary):
                tick = (bar * steps + step) * ticks_per_step
                if val:
                    events.append((tick, "on",  note, velocity))
                    events.append((tick + ticks_per_step - 1, "off", note, 0))

        events.sort()
        prev_tick = 0
        for tick, etype, note_v, vel in events:
            delta = tick - prev_tick
            msg_type = "note_on" if etype == "on" else "note_off"
            track.append(mido.Message(msg_type, channel=9, note=note_v, velocity=vel, time=delta))
            prev_tick = tick

    mid.save(output)


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZACIÓN ASCII
# ─────────────────────────────────────────────────────────────────────────────

PULSE_ON  = "█"
PULSE_OFF = "·"

def render_binary(binary: list[int], name: str, extra: str = "") -> str:
    """Renderiza un patrón binario como línea ASCII."""
    pattern_str = "".join(c("cyan", PULSE_ON) if v else c("gray", PULSE_OFF) for v in binary)
    events = sum(binary)
    steps = len(binary)
    info = c("gray", f"  [{events}/{steps}]")
    if extra:
        info += c("gray", f"  {extra}")
    return f"  {c('bold', f'{name:<12}')}{pattern_str}{info}"


def render_intervals(binary: list[int]) -> str:
    """Muestra los intervalos entre onsets."""
    positions = [i for i, v in enumerate(binary) if v]
    n = len(positions)
    if n < 2:
        return ""
    intervals_str = []
    for i in range(n):
        diff = (positions[(i+1) % n] - positions[i]) % len(binary)
        intervals_str.append(str(diff))
    return c("gray", "  intervals: [" + ", ".join(intervals_str) + "]")


def gcd_val(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ─────────────────────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="euclidean_drummer — Generador de ritmos matemáticos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=16, metavar="N",
                        help="Número de pasos del ciclo (default: 16)")
    parser.add_argument("--events", type=int, default=5, metavar="K",
                        help="Número de pulsos/onsets (default: 5)")
    parser.add_argument("--algo",
                        choices=["euclidean", "clough", "deep", "tihai"],
                        default="euclidean",
                        help="Algoritmo a usar (default: euclidean)")
    parser.add_argument("--mult", type=int, metavar="M",
                        help="Multiplicidad para deep rhythm (default: auto)")
    parser.add_argument("--reps", type=int, default=3, metavar="R",
                        help="Repeticiones para tihai (default: 3)")
    parser.add_argument("--offset", type=int, default=0, metavar="O",
                        help="Desplazamiento del patrón (default: 0)")
    parser.add_argument("--bars", type=int, default=4, metavar="N",
                        help="Número de compases en la exportación MIDI (default: 4)")
    parser.add_argument("--tempo", type=int, default=120, metavar="BPM",
                        help="Tempo del MIDI exportado (default: 120)")
    parser.add_argument("--note", type=int, default=36, metavar="N",
                        help="Nota MIDI del patrón (default: 36 = bass drum)")
    parser.add_argument("--velocity", type=int, default=100, metavar="V",
                        help="Velocidad MIDI (default: 100)")
    parser.add_argument("--output", metavar="FILE",
                        help="Exportar a MIDI")
    # modos especiales
    parser.add_argument("--compare", action="store_true",
                        help="Genera los 4 algoritmos en paralelo para comparar")
    parser.add_argument("--family", action="store_true",
                        help="Genera toda la familia E(k,n) para n fijo")
    parser.add_argument("--poly", nargs="+", metavar="K",
                        help="Polirritmo: lista de densidades para superponer (ej: 3 5 7)")
    parser.add_argument("--no-color", action="store_true",
                        help="Sin colores ANSI")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    steps = args.steps
    events = args.events

    print(f"\n{c('bold', 'EUCLIDEAN DRUMMER')}  |  steps={steps}")
    print(c("gray", "─" * 68))

    # ── Modo familia ─────────────────────────────────────────────────────────
    if args.family:
        print(c("yellow", f"\n  Familia E(k, {steps}):\n"))
        for k in range(1, steps + 1):
            binary = euclidean_binary(steps, k, args.offset)
            print(render_binary(binary, f"E({k},{steps})"))
            print(render_intervals(binary))
        print()
        return

    # ── Modo polirritmo ───────────────────────────────────────────────────────
    if args.poly:
        try:
            poly_events = [int(k) for k in args.poly]
        except ValueError:
            print(c("red", "Error: --poly requiere enteros"), file=sys.stderr)
            sys.exit(1)
        print(c("yellow", f"\n  Polirritmo E(k, {steps}):\n"))
        patterns = []
        for k in poly_events:
            binary = euclidean_binary(steps, k, args.offset)
            name = f"E({k},{steps})"
            patterns.append((name, binary))
            print(render_binary(binary, name))
        # superposición
        combined = [max(p[k] for _, p in patterns) for k in range(steps)]
        print(c("gray", "  " + "─" * (steps + 14)))
        print(render_binary(combined, "Superpuesto"))
        print()
        if args.output:
            note_map = {f"E({k},{steps})": [36, 38, 42, 46, 49][i % 5] for i, k in enumerate(poly_events)}
            export_midi(patterns, steps, args.bars, args.tempo, args.output, note_map, args.velocity)
            print(c("green", f"MIDI exportado: {args.output}"))
        return

    # ── Modo comparación ──────────────────────────────────────────────────────
    if args.compare:
        print(c("yellow", f"\n  Comparación E({events}, {steps}):\n"))
        mult_val = args.mult if args.mult else best_multiplicity(steps, events)

        algos = [
            ("euclidean", euclidean_binary(steps, events, args.offset)),
            ("clough",    clough_douthett_binary(steps, events, args.offset)),
            ("deep",      deep_rhythm_binary(steps, events, mult_val, args.offset)),
            ("tihai",     tihai_binary(steps, args.reps)),
        ]
        for name, binary in algos:
            print(render_binary(binary, name))
            print(render_intervals(binary))
        print()
        if args.output:
            export_midi(algos, steps, args.bars, args.tempo, args.output, velocity=args.velocity)
            print(c("green", f"MIDI exportado: {args.output}"))
        return

    # ── Modo normal ───────────────────────────────────────────────────────────
    algo = args.algo

    if algo == "euclidean":
        binary = euclidean_binary(steps, events, args.offset)
        intervals = euclidean_intervals(steps, events)
        name = f"E({events},{steps})"
        extra = f"intervals={intervals}"
    elif algo == "clough":
        binary = clough_douthett_binary(steps, events, args.offset)
        name = f"CD({events},{steps})"
        extra = ""
    elif algo == "deep":
        mult = args.mult if args.mult else best_multiplicity(steps, events)
        binary = deep_rhythm_binary(steps, events, mult, args.offset)
        name = f"D({events},{steps},m={mult})"
        extra = f"mult={mult}"
    elif algo == "tihai":
        binary = tihai_binary(steps, args.reps)
        bols, dams = tihai_generator(steps, args.reps)
        name = f"T({steps},{args.reps})"
        extra = f"bols={bols}  dams={dams}"
    else:
        print(c("red", f"Algoritmo desconocido: {algo}"), file=sys.stderr)
        sys.exit(1)

    print(f"\n{render_binary(binary, name, extra)}")
    print(render_intervals(binary))

    positions = [i for i, v in enumerate(binary) if v]
    print(c("gray", f"  posiciones: {positions}"))
    print()

    if args.output:
        export_midi(
            [(name, binary)],
            steps, args.bars, args.tempo, args.output,
            {name: args.note}, args.velocity,
        )
        print(c("green", f"MIDI exportado: {args.output}"))


if __name__ == "__main__":
    main()
