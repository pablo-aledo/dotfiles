#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SCALE VOICE LEADER  v1.0                                  ║
║         Voice leading por rototranslación matricial                          ║
║                                                                              ║
║  Portado desde music++ / automations.h + matrixDistance.h + matrix.h        ║
║                                                                              ║
║  Dado un acorde de origen y uno de destino (como MIDI o notas),             ║
║  encuentra la inversión/voicing óptimo del acorde destino que minimiza       ║
║  el movimiento total de voces. Utiliza el algoritmo de rototranslación       ║
║  matricial de music++: genera todas las inversiones+transposiciones del      ║
║  destino en un rango de octavas, calcula distancias Manhattan y ordena       ║
║  por coste total. El parámetro --complexity selecciona entre la solución     ║
║  de mínimo coste (0) y la de máximo coste (100).                            ║
║                                                                              ║
║  USO:                                                                        ║
║    # Acordes como MIDI pitch values                                          ║
║    python scale_voice_leader.py "60 64 67" "65 69 72"                       ║
║                                                                              ║
║    # Con nombres de notas                                                    ║
║    python scale_voice_leader.py "C4 E4 G4" "F4 A4 C5"                      ║
║                                                                              ║
║    # Desde archivos MIDI (un acorde por archivo)                             ║
║    python scale_voice_leader.py origen.mid destino.mid                      ║
║                                                                              ║
║    # Controlar la complejidad del voice leading (0=mínimo, 100=máximo)      ║
║    python scale_voice_leader.py "C4 E4 G4" "F4 A4 C5" --complexity 50      ║
║                                                                              ║
║    # Mostrar las N mejores soluciones                                        ║
║    python scale_voice_leader.py "60 64 67" "65 69 72" --top 5              ║
║                                                                              ║
║    # Cadena de acordes (voice leading encadenado)                            ║
║    python scale_voice_leader.py --chain "C4 E4 G4" "A3 C4 E4" "F3 A3 C4"  ║
║                                                                              ║
║    # Exportar MIDI con el resultado                                          ║
║    python scale_voice_leader.py "C4 E4 G4" "F4 A4 C5" --output out.mid    ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    origen destino    Acordes (pitches MIDI o nombres de nota)               ║
║    --complexity N    Complejidad 0-100 (default: 0 = movimiento mínimo)     ║
║    --top N           Mostrar las N mejores soluciones (default: 1)          ║
║    --chain           Modo cadena: encadena varios acordes con VL óptimo     ║
║    --metric M        Métrica de distancia: manhattan|euclidean (default: manhattan) ║
║    --octave-range N  Rango de octavas a explorar (default: 2)              ║
║    --output FILE     Exportar acorde resultado a MIDI                        ║
║    --tempo BPM       Tempo del MIDI exportado (default: 120)                ║
║    --verbose         Mostrar tabla de todas las rototranslaciones            ║
║    --no-color        Sin colores ANSI                                        ║
║                                                                              ║
║  DEPENDENCIAS: mido (solo para --output y lectura de .mid)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys
import math
from itertools import product


# ─────────────────────────────────────────────────────────────────────────────
#  ANSI
# ─────────────────────────────────────────────────────────────────────────────

ANSI = {
    "reset":  "\033[0m", "bold": "\033[1m", "green": "\033[32m",
    "yellow": "\033[33m", "cyan": "\033[36m", "gray": "\033[90m",
    "red":    "\033[31m", "magenta": "\033[35m", "blue": "\033[34m",
}
USE_COLOR = True

def c(key: str, text: str) -> str:
    return f"{ANSI[key]}{text}{ANSI['reset']}" if USE_COLOR else text


# ─────────────────────────────────────────────────────────────────────────────
#  NOMENCLATURA DE NOTAS
# ─────────────────────────────────────────────────────────────────────────────

NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAME_TO_PC  = {n: i for i, n in enumerate(NOTE_NAMES)}
ENHARMONIC  = {"Db":1,"Eb":3,"Fb":4,"Gb":6,"Ab":8,"Bb":10,"Cb":11,"E#":5,"B#":0}

OCTAVE_NAMES = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

def note_name_to_midi(name: str) -> int:
    """Convierte nombre de nota (C4, Bb3, F#5…) a MIDI pitch."""
    name = name.strip()
    # extraer octava del final
    i = len(name) - 1
    while i >= 0 and (name[i].isdigit() or name[i] == "-"):
        i -= 1
    octave = int(name[i+1:]) if i + 1 < len(name) else 4
    note_part = name[:i+1]
    note_part = note_part.capitalize()
    if note_part in NAME_TO_PC:
        pc = NAME_TO_PC[note_part]
    elif note_part in ENHARMONIC:
        pc = ENHARMONIC[note_part]
    else:
        raise ValueError(f"Nota no reconocida: '{name}'")
    return (octave + 1) * 12 + pc


def midi_to_name(midi: int) -> str:
    pc = midi % 12
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[pc]}{octave}"


def parse_chord(token: str) -> list[int]:
    """
    Parsea un acorde desde string.
    Acepta: "60 64 67" o "C4 E4 G4" o un archivo .mid
    """
    token = token.strip()
    if token.endswith(".mid"):
        return read_midi_chord(token)
    parts = token.split()
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(note_name_to_midi(p))
    return sorted(result)


# ─────────────────────────────────────────────────────────────────────────────
#  LECTURA DE MIDI
# ─────────────────────────────────────────────────────────────────────────────

def read_midi_chord(path: str) -> list[int]:
    try:
        import mido
    except ImportError:
        print(c("red", "Error: mido no instalado. pip install mido"), file=sys.stderr)
        sys.exit(1)
    mid = mido.MidiFile(path)
    pitches = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pitches.add(msg.note)
    return sorted(pitches)


def export_midi(chords: list[list[int]], output: str, tempo_bpm: int = 120):
    try:
        import mido
    except ImportError:
        print(c("red", "Error: mido no instalado. pip install mido"), file=sys.stderr)
        sys.exit(1)
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    duration = 480 * 2  # negra doble por acorde
    for chord in chords:
        for note in chord:
            track.append(mido.Message("note_on", note=note, velocity=80, time=0))
        track.append(mido.Message("note_off", note=chord[0], velocity=0, time=duration))
        for note in chord[1:]:
            track.append(mido.Message("note_off", note=note, velocity=0, time=0))
    mid.save(output)


# ─────────────────────────────────────────────────────────────────────────────
#  NÚCLEO: ROTOTRANSLACIÓN MATRICIAL
#  Portado de music++ / matrix.h (rototranslationMatrix) y
#  matrixDistance.h (calculateDistances, getByComplexity)
# ─────────────────────────────────────────────────────────────────────────────

def roto_translate(chord: list[int], translation: int) -> list[int]:
    """
    Rototranslación: rota el vector un paso (sube la nota más grave una octava)
    y traslada globalmente.
    Portado de PositionVector::rotoTranslate() en positionVector.h.
    """
    if not chord:
        return chord
    c_sorted = sorted(chord)
    n = len(c_sorted)
    # Rotación: la nota más grave sube una octava y va al final
    rotated = c_sorted[1:] + [c_sorted[0] + 12]
    # Traslación global
    return [p + translation for p in rotated]


def build_rototranslation_matrix(chord: list[int], center: int, octave_range: int = 2) -> list[tuple[list[int], int]]:
    """
    Genera todas las rototranslaciones del acorde en un rango de traslaciones.
    Portado de rototranslationMatrix() en matrix.h.
    
    center: punto de anclaje (normalmente el primer pitch del acorde de referencia)
    octave_range: cuántas octavas arriba y abajo explorar con traslaciones
    
    Para cada número de rotaciones (0..n-1) y para cada traslación en el rango,
    genera el acorde rotado+trasladado.
    """
    n = len(chord)
    matrix = []
    span = octave_range * 12 * n

    # generar todas las rotaciones base
    rotations = []
    current = sorted(chord)
    for _ in range(n):
        rotations.append(current[:])
        # rotar: primer elemento + 12, al final
        current = current[1:] + [current[0] + 12]

    # para cada rotación, explorar traslaciones alrededor del center
    for rot_idx, rotated in enumerate(rotations):
        for delta in range(-span, span + 1, 1):
            candidate = [p + delta for p in rotated]
            # solo incluir si está en rango razonable de MIDI
            if all(0 <= p <= 127 for p in candidate):
                matrix.append((candidate, delta + rot_idx * 12))

    # deduplicar por contenido
    seen = set()
    unique = []
    for chord_v, idx in matrix:
        key = tuple(sorted(chord_v))
        if key not in seen:
            seen.add(key)
            unique.append((chord_v, idx))

    return unique


def align(reference: list[int], target: list[int]) -> int:
    """
    Encuentra el índice de traslación que alinea target con reference.
    Portado de align() en matrixDistance.h.
    """
    if not reference or not target:
        return 0
    min_ref = min(reference)
    # buscar el punto de traslación que pone target cerca de reference
    target_min = min(target)
    return min_ref - target_min


def manhattan_distance(a: list[int], b: list[int]) -> float:
    """Distancia Manhattan entre dos vectores de igual longitud."""
    if len(a) != len(b):
        return float("inf")
    return sum(abs(x - y) for x, y in zip(sorted(a), sorted(b)))


def euclidean_distance(a: list[int], b: list[int]) -> float:
    """Distancia Euclidiana entre dos vectores."""
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(sorted(a), sorted(b))))


def calculate_distances(
    reference: list[int],
    matrix: list[tuple[list[int], int]],
    metric: str = "manhattan",
) -> list[tuple[list[int], int, float]]:
    """
    Calcula distancias entre reference y cada fila de la matriz.
    Portado de calculateDistances() en matrixDistance.h.
    Devuelve lista de (chord, translation_index, distance) ordenada por distancia.
    """
    dist_fn = euclidean_distance if metric == "euclidean" else manhattan_distance
    n = len(reference)
    results = []
    for chord_v, idx in matrix:
        if len(chord_v) != n:
            continue
        d = dist_fn(reference, chord_v)
        results.append((chord_v, idx, d))
    results.sort(key=lambda x: x[2])
    return results


def get_by_complexity(
    distances: list[tuple[list[int], int, float]],
    complexity: int = 0,
) -> tuple[list[int], int, float]:
    """
    Selecciona un resultado por complejidad (0=más cercano, 100=más lejano).
    Portado de RototranslationMatrixDistance::getByComplexity() en matrixDistance.h.
    """
    if not distances:
        raise ValueError("La lista de distancias está vacía")
    complexity = max(0, min(100, complexity))
    idx = int((complexity / 100.0) * (len(distances) - 1))
    return distances[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  VOICE LEADING PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def voice_lead(
    source: list[int],
    target: list[int],
    complexity: int = 0,
    metric: str = "manhattan",
    octave_range: int = 2,
) -> dict:
    """
    Calcula el voice leading óptimo de source → target.
    
    Portado de voiceLeadingAutomation() en automations.h:
      1. align(reference, target) → center
      2. rototranslationMatrix(target, center)
      3. calculateDistances(reference, matrix)
      4. getByComplexity(complexity)
    
    Devuelve dict con el resultado y metadatos.
    """
    center = align(source, target)
    matrix = build_rototranslation_matrix(target, center, octave_range)
    distances = calculate_distances(source, matrix, metric)

    if not distances:
        return {"error": "No se encontraron rototranslaciones válidas"}

    best_chord, best_idx, best_dist = get_by_complexity(distances, complexity)

    # calcular movimiento por voz
    movements = [b - a for a, b in zip(sorted(source), sorted(best_chord))]

    return {
        "source": sorted(source),
        "target_original": sorted(target),
        "target_voiced": sorted(best_chord),
        "translation_index": best_idx,
        "distance": best_dist,
        "complexity": complexity,
        "metric": metric,
        "voice_movements": movements,
        "total_movement": sum(abs(m) for m in movements),
        "all_solutions": distances,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PRESENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def format_chord(pitches: list[int]) -> str:
    return " ".join(midi_to_name(p) for p in sorted(pitches))


def print_result(result: dict, top: int = 1, verbose: bool = False):
    if "error" in result:
        print(c("red", f"Error: {result['error']}"))
        return

    print(f"\n  {c('gray', 'Origen:  ')}{c('bold', format_chord(result['source']))}")
    print(f"  {c('gray', 'Destino: ')}{format_chord(result['target_original'])}")
    print()

    all_sol = result["all_solutions"]
    shown = all_sol[:top]

    for rank, (chord_v, idx, dist) in enumerate(shown):
        movements = [b - a for a, b in zip(sorted(result["source"]), sorted(chord_v))]
        total_mv = sum(abs(m) for m in movements)
        mov_str = "  ".join(
            (c("green", f"+{m}") if m > 0 else c("red", str(m)) if m < 0 else c("gray", "0"))
            for m in movements
        )
        prefix = c("cyan", f"  #{rank+1}") if top > 1 else "  "
        print(f"{prefix}  {c('bold', format_chord(chord_v))}")
        print(f"        distancia: {c('yellow', str(dist))}  |  movimientos: {mov_str}  |  total: {total_mv}")

    if verbose and len(all_sol) > top:
        print(c("gray", f"\n  ({len(all_sol) - top} soluciones adicionales no mostradas)"))
        if top < len(all_sol):
            print(c("gray", "\n  Tabla de rototranslaciones (mostrando las 20 primeras):"))
            for chord_v, idx, dist in all_sol[:20]:
                print(c("gray", f"    {format_chord(chord_v):30s}  dist={dist:.1f}"))

    print()


def print_chain(chain_result: list[dict]):
    print(f"\n{c('bold', 'CADENA DE VOICE LEADING')}\n")
    for i, step in enumerate(chain_result):
        if i == 0:
            print(f"  {c('gray', f'Acorde {i+1}:')}  {c('bold', format_chord(step['source']))}")
        arrow = c("cyan", "  ↓  ")
        total_mv = step["total_movement"]
        print(f"{arrow}{c('gray', f'[Delta={total_mv}]')}")
        print(f"  {c('gray', f'Acorde {i+2}:')}  {c('bold', format_chord(step['target_voiced']))}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="scale_voice_leader — Voice leading por rototranslación matricial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("chords", nargs="*",
                        help='Acordes: "60 64 67" "65 69 72" o "C4 E4 G4" "F4 A4 C5"')
    parser.add_argument("--complexity", type=int, default=0, metavar="N",
                        help="Complejidad 0-100 (0=mínimo movimiento, 100=máximo)")
    parser.add_argument("--top", type=int, default=1, metavar="N",
                        help="Mostrar las N mejores soluciones")
    parser.add_argument("--chain", action="store_true",
                        help="Modo cadena: encadena todos los acordes con VL óptimo")
    parser.add_argument("--metric", choices=["manhattan", "euclidean"], default="manhattan",
                        help="Métrica de distancia (default: manhattan)")
    parser.add_argument("--octave-range", type=int, default=2, metavar="N",
                        help="Rango de octavas a explorar (default: 2)")
    parser.add_argument("--output", metavar="FILE",
                        help="Exportar acorde/cadena resultado a MIDI")
    parser.add_argument("--tempo", type=int, default=120, metavar="BPM",
                        help="Tempo del MIDI exportado (default: 120)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar tabla completa de rototranslaciones")
    parser.add_argument("--no-color", action="store_true",
                        help="Sin colores ANSI")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    if not args.chords:
        parser.print_help()
        sys.exit(1)

    print(f"\n{c('bold', 'SCALE VOICE LEADER')}  |  métrica={args.metric}  complejidad={args.complexity}")
    print(c("gray", "─" * 68))

    # ── Parsear acordes ────────────────────────────────────────────────────────
    try:
        parsed_chords = [parse_chord(ch) for ch in args.chords]
    except ValueError as e:
        print(c("red", f"Error: {e}"), file=sys.stderr)
        sys.exit(1)

    if args.chain or len(parsed_chords) > 2:
        # ── Modo cadena ────────────────────────────────────────────────────────
        chain_results = []
        current = parsed_chords[0]
        for next_chord in parsed_chords[1:]:
            result = voice_lead(current, next_chord, args.complexity, args.metric, args.octave_range)
            chain_results.append(result)
            current = result["target_voiced"]
        print_chain(chain_results)

        if args.output:
            all_chords = [chain_results[0]["source"]] + [r["target_voiced"] for r in chain_results]
            export_midi(all_chords, args.output, args.tempo)
            print(c("green", f"MIDI exportado: {args.output}"))

    elif len(parsed_chords) == 2:
        # ── Modo par ───────────────────────────────────────────────────────────
        result = voice_lead(parsed_chords[0], parsed_chords[1], args.complexity, args.metric, args.octave_range)
        print_result(result, args.top, args.verbose)

        if args.output:
            export_midi([result["source"], result["target_voiced"]], args.output, args.tempo)
            print(c("green", f"MIDI exportado: {args.output}"))

    else:
        # ── Solo un acorde: mostrar su matriz de rototranslaciones ─────────────
        chord = parsed_chords[0]
        print(f"\n  Acorde: {c('bold', format_chord(chord))}")
        center = chord[0]
        matrix = build_rototranslation_matrix(chord, center, args.octave_range)
        print(c("gray", f"  {len(matrix)} rototranslaciones generadas:\n"))
        for i, (ch, idx) in enumerate(matrix[:24]):
            print(c("gray", f"    [{idx:+4d}]  {format_chord(ch)}"))
        if len(matrix) > 24:
            print(c("gray", f"    … y {len(matrix)-24} más"))
        print()


if __name__ == "__main__":
    main()
