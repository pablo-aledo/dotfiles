#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    SLONIMSKY EXPLORER  v1.0                                  ║
║         Generador de patrones del Thesaurus de Slonimsky                    ║
║                                                                              ║
║  Portado desde music++ / slonimsky.h                                         ║
║                                                                              ║
║  Implementa las operaciones del "Thesaurus of Musical Scales and Patterns"   ║
║  de Nicholas Slonimsky (1947):                                               ║
║                                                                              ║
║    interpolation          — inserta notas ENTRE los intervalos del patrón   ║
║    ultrapolation          — inserta notas ANTES de cada nota (apoyatura sup)║
║    infrapolation          — inserta notas DESPUÉS de cada nota (inf)        ║
║    symmetric_interpolation — interpolación alternada arriba/abajo           ║
║    asymmetric_interpolation — inter con offsets superiores e inferiores dist║
║    infra_interpolation    — combinado: infra + inter                        ║
║    inter_infrapolation    — combinado: inter + infra                        ║
║    infra_ultrapolation    — combinado: infra + ultra                        ║
║    inter_ultrapolation    — combinado: inter + ultra                        ║
║    ultra_interpolation    — combinado: ultra + inter                        ║
║    infra_inter_ultrapolation — triple combinación                           ║
║                                                                              ║
║  USO:                                                                        ║
║    # Interpolación simple de una escala de tonos enteros                    ║
║    python slonimsky_explorer.py --pattern "0 4 8" --op interpolation --k 1  ║
║                                                                              ║
║    # Ultrapolación multi-nota                                                ║
║    python slonimsky_explorer.py --pattern "0 4 8" --op ultrapolation --k "1 2" ║
║                                                                              ║
║    # Infrapolación simétrica                                                 ║
║    python slonimsky_explorer.py --pattern "0 6 12" --op symmetric_interpolation --k 2 ║
║                                                                              ║
║    # Explorar toda la familia para un patrón                                ║
║    python slonimsky_explorer.py --pattern "0 4 8" --explore                 ║
║                                                                              ║
║    # Con raíz MIDI y exportar a MIDI                                        ║
║    python slonimsky_explorer.py --pattern "0 4 8" --op interpolation --k 1  ║
║        --root 60 --output slonimsky.mid                                     ║
║                                                                              ║
║    # Patrón desde nombre de notas                                           ║
║    python slonimsky_explorer.py --notes "C4 E4 Ab4" --op interpolation --k 1 ║
║                                                                              ║
║    # Permutación de infrapolación (verificar ordering constraint)           ║
║    python slonimsky_explorer.py --infra-perm 5                              ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --pattern "n n n …"   Secuencia de pitch-classes o pitches MIDI          ║
║    --notes "N N N …"     Secuencia como nombres de nota (C4, Eb3, …)       ║
║    --op OPERATION        Operación de Slonimsky (ver lista arriba)          ║
║    --k "offset(s)"       Offset superior: entero o "n1 n2 …"               ║
║    --l "offset(s)"       Offset inferior (para operaciones asimétricas)     ║
║    --m offset            Tercer offset (para triples)                       ║
║    --root N              Pitch MIDI raíz (para exportación, default: 60)   ║
║    --explore             Explora todas las operaciones con k=1..3          ║
║    --octaves N           Número de octavas a generar (default: 1)          ║
║    --infra-perm M        Muestra la permutación sigma para infrapolación(M) ║
║    --output FILE         Exportar a MIDI                                     ║
║    --tempo BPM           Tempo del MIDI (default: 120)                      ║
║    --note-dur D          Duración de nota MIDI en ticks (default: 240)      ║
║    --no-color            Sin colores ANSI                                    ║
║                                                                              ║
║  DEPENDENCIAS: mido (solo para --output)                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import sys


# ─────────────────────────────────────────────────────────────────────────────
#  ANSI
# ─────────────────────────────────────────────────────────────────────────────

ANSI = {
    "reset":  "\033[0m", "bold": "\033[1m",  "green": "\033[32m",
    "yellow": "\033[33m","cyan": "\033[36m",  "gray":  "\033[90m",
    "red":    "\033[31m","blue": "\033[34m",  "magenta": "\033[35m",
}
USE_COLOR = True

def c(key: str, text: str) -> str:
    return f"{ANSI[key]}{text}{ANSI['reset']}" if USE_COLOR else text


# ─────────────────────────────────────────────────────────────────────────────
#  NOMENCLATURA
# ─────────────────────────────────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {"Db":1,"Eb":3,"Fb":4,"Gb":6,"Ab":8,"Bb":10,"Cb":11,"E#":5,"B#":0}

def note_name_to_midi(name: str) -> int:
    name = name.strip()
    i = len(name) - 1
    while i >= 0 and (name[i].isdigit() or name[i] == "-"):
        i -= 1
    octave_str = name[i+1:] if i + 1 < len(name) else "4"
    octave = int(octave_str) if octave_str else 4
    note_part = name[:i+1].capitalize()
    if note_part in {n: i for i, n in enumerate(NOTE_NAMES)}:
        pc = list(NOTE_NAMES).index(note_part)
    elif note_part in ENHARMONIC:
        pc = ENHARMONIC[note_part]
    else:
        raise ValueError(f"Nota no reconocida: '{name}'")
    return (octave + 1) * 12 + pc

def midi_to_name(midi: int) -> str:
    pc = midi % 12
    octave = midi // 12 - 1
    return f"{NOTE_NAMES[pc]}{octave}"

def parse_seq(token: str) -> list[int]:
    """Parsea una secuencia de enteros desde string."""
    return [int(x) for x in token.strip().split()]

def parse_offsets(token: str) -> int | list[int]:
    """Parsea offset(s): si hay un solo número devuelve int, si hay varios devuelve lista."""
    parts = [int(x) for x in token.strip().split()]
    return parts[0] if len(parts) == 1 else parts


# ─────────────────────────────────────────────────────────────────────────────
#  OPERACIONES DE SLONIMSKY
#  Portadas directamente de slonimsky.h (music++)
#  Todas las funciones toman:
#    x : list[int] — secuencia ordenada de pitches
#    k : int | list[int] — offset(s) superior(es)
#    l : int | list[int] — offset(s) inferior(es) (para ops asimétricas)
#    m : int — tercer offset (para ops triples)
# ─────────────────────────────────────────────────────────────────────────────

def _seq_intervals(x: list[int]) -> list[int]:
    return [x[i+1] - x[i] for i in range(len(x)-1)]

def _min_interval(x: list[int]) -> int:
    d = _seq_intervals(x)
    return min(d) if d else 0

def infrapolation_permutation(m: int) -> list[int]:
    """
    Devuelve la permutación sigma para infrapolación.
    Portado de infrapolationPermutation() en slonimsky.h.
    Útil para verificar el ordering constraint sobre el vector de offsets.
    """
    h = (m + 1) // 2
    delta = 1 if m % 2 == 0 else 0
    sigma = []
    for i in range(1, m + 1):
        if i <= h:
            val = m - 2 * (i - 1)
        else:
            val = 2 * (i - h) - delta
        sigma.append(val - 1)
    return sigma


# ── Interpolation ─────────────────────────────────────────────────────────────

def interpolation(x: list[int], k: int | list[int]) -> list[int]:
    """
    Inserta k ENTRE cada par de notas consecutivas.
    
    Single note (k: int):
      para i=0..n-2 : (x[i], x[i]+k)  luego x[n-1]
    
    Multiple notes (k: list):
      para i=0..n-2 : (x[i], x[i]+k[0], ..., x[i]+k[m-1])  luego x[n-1]
    
    Portado de interpolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    if isinstance(k, int):
        for i in range(n - 1):
            out.append(x[i])
            out.append(x[i] + k)
    else:
        for i in range(n - 1):
            out.append(x[i])
            for kj in k:
                out.append(x[i] + kj)
    out.append(x[n - 1])
    return out


# ── Ultrapolation ─────────────────────────────────────────────────────────────

def ultrapolation(x: list[int], k: int | list[int]) -> list[int]:
    """
    Inserta k ANTES de cada nota (excepto la primera).
    
    Single note (k: int):
      (x[0]) ++ para i=1..n-1 : (x[i]+k, x[i])
    
    Multiple notes (k: list):
      (x[0]) ++ para i=1..n-1 : (x[i]+k[m-1], ..., x[i]+k[0], x[i])
    
    Portado de ultrapolation() en slonimsky.h.
    """
    n = len(x)
    out = [x[0]]
    if isinstance(k, int):
        for i in range(1, n):
            out.append(x[i] + k)
            out.append(x[i])
    else:
        for i in range(1, n):
            for kj in reversed(k):
                out.append(x[i] + kj)
            out.append(x[i])
    return out


# ── Infrapolation ─────────────────────────────────────────────────────────────

def infrapolation(x: list[int], k: int | list[int]) -> list[int]:
    """
    Inserta k DESPUÉS de cada nota (excepto la última).
    
    Single note (k: int):
      para i=0..n-2 : (x[i], x[i]-k)  luego x[n-1]
    
    Multiple notes (k: list):
      para i=0..n-2 : (x[i], x[i]-k[0], ..., x[i]-k[m-1])  luego x[n-1]
    
    Portado de infrapolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    if isinstance(k, int):
        for i in range(n - 1):
            out.append(x[i])
            out.append(x[i] - k)
    else:
        for i in range(n - 1):
            out.append(x[i])
            for kj in k:
                out.append(x[i] - kj)
    out.append(x[n - 1])
    return out


# ── Symmetric interpolation ───────────────────────────────────────────────────

def symmetric_interpolation(x: list[int], k: int | list[int]) -> list[int]:
    """
    Interpolación simétrica: alterna upper e lower para cada par de notas.
    
    Single note (k: int):
      para j=0..floor((n-1)/2)-1 : (x[2j], x[2j]+k, x[2j+1], x[2j+2]-k)
      luego tau según paridad de n
    
    Portado de symmetricInterpolation() en slonimsky.h.
    """
    n = len(x)
    pairs = (n - 1) // 2
    out = []
    if isinstance(k, int):
        for j in range(pairs):
            out.append(x[2*j])
            out.append(x[2*j] + k)
            out.append(x[2*j + 1])
            out.append(x[2*j + 2] - k)
        if n % 2 == 1:
            out.append(x[n-2])
            out.append(x[n-2] + k)
            out.append(x[n-1])
        else:
            out.append(x[n-1])
    else:
        m = len(k)
        for j in range(pairs):
            out.append(x[2*j])
            for r in range(m):
                out.append(x[2*j] + k[r])
            out.append(x[2*j + 1])
            for r in range(m-1, -1, -1):
                out.append(x[2*j + 2] - k[r])
        if n % 2 == 1:
            out.append(x[n-2])
            for r in range(m):
                out.append(x[n-2] + k[r])
            out.append(x[n-1])
        else:
            out.append(x[n-1])
    return out


# ── Asymmetric interpolation ──────────────────────────────────────────────────

def asymmetric_interpolation(x: list[int], k: list[int], l: list[int]) -> list[int]:
    """
    Interpolación asimétrica: k para la parte superior, l para la inferior.
    
    para j=0..floor((n-1)/2)-1 :
      (x[2j], x[2j]+k[0], ..., x[2j]+k[m-1],
       x[2j+1],
       x[2j+2]-l[m-1], ..., x[2j+2]-l[0])
    luego tau (usa k para la cola impar)
    
    Portado de asymmetricInterpolation() en slonimsky.h.
    """
    n = len(x)
    m = len(k)
    pairs = (n - 1) // 2
    out = []
    for j in range(pairs):
        out.append(x[2*j])
        for r in range(m):
            out.append(x[2*j] + k[r])
        out.append(x[2*j + 1])
        for r in range(m-1, -1, -1):
            out.append(x[2*j + 2] - l[r])
    if n % 2 == 1:
        out.append(x[n-2])
        for r in range(m):
            out.append(x[n-2] + k[r])
        out.append(x[n-1])
    else:
        out.append(x[n-1])
    return out


# ── Composite operations ──────────────────────────────────────────────────────

def infra_interpolation(x: list[int], k: int, l: int) -> list[int]:
    """
    para i=0..n-2 : (x[i], x[i]-k, x[i]+l)  luego x[n-1]
    Portado de infraInterpolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        out.append(x[i] - k)
        out.append(x[i] + l)
    out.append(x[n - 1])
    return out


def inter_infrapolation(x: list[int], k: int, l: int) -> list[int]:
    """
    para i=0..n-2 : (x[i], x[i]+k, x[i]-l)  luego x[n-1]
    Portado de interInfrapolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        out.append(x[i] + k)
        out.append(x[i] - l)
    out.append(x[n - 1])
    return out


def infra_ultrapolation(x: list[int], k: int, l: int) -> list[int]:
    """
    (x[0], x[0]-k)  ++
    para i=1..n-2 : (x[i]+l, x[i], x[i]-k)  ++
    (x[n-1]+l, x[n-1])
    Portado de infraUltrapolation() en slonimsky.h.
    """
    n = len(x)
    out = [x[0], x[0] - k]
    for i in range(1, n - 1):
        out.append(x[i] + l)
        out.append(x[i])
        out.append(x[i] - k)
    out.append(x[n-1] + l)
    out.append(x[n-1])
    return out


def inter_ultrapolation(x: list[int], k: int, l: int) -> list[int]:
    """
    para i=0..n-2 : (x[i], x[i]+k, x[i+1]-l)  luego x[n-1]
    Portado de interUltrapolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        out.append(x[i] + k)
        out.append(x[i + 1] - l)
    out.append(x[n - 1])
    return out


def ultra_interpolation(x: list[int], k: int, l: int) -> list[int]:
    """
    para i=0..n-2 : (x[i], x[i+1]+k, x[i]+l)  luego x[n-1]
    Portado de ultraInterpolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        out.append(x[i + 1] + k)
        out.append(x[i] + l)
    out.append(x[n - 1])
    return out


def infra_inter_ultrapolation(x: list[int], k: int, l: int, m: int) -> list[int]:
    """
    para i=0..n-2 : (x[i], x[i]-k, x[i]+l, x[i+1]+m)  luego x[n-1]
    Portado de infraInterUltrapolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        out.append(x[i] - k)
        out.append(x[i] + l)
        out.append(x[i + 1] + m)
    out.append(x[n - 1])
    return out


def inter_infra_interpolation(x: list[int], k: int, l: int, m: int) -> list[int]:
    """
    para i=0..n-2 : (x[i], x[i+1]-k, x[i]-l, x[i]+m)  luego x[n-1]
    Portado de interInfraInterpolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        out.append(x[i + 1] - k)
        out.append(x[i] - l)
        out.append(x[i] + m)
    out.append(x[n - 1])
    return out


def ultra_infra_interpolation(x: list[int], k: int, l: int, m: int) -> list[int]:
    """
    para i=0..n-2 : (x[i], x[i+1]+k, x[i]-l, x[i]+m)  luego x[n-1]
    Portado de ultraInfraInterpolation() en slonimsky.h.
    """
    n = len(x)
    out = []
    for i in range(n - 1):
        out.append(x[i])
        out.append(x[i + 1] + k)
        out.append(x[i] - l)
        out.append(x[i] + m)
    out.append(x[n - 1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  DISPATCH DE OPERACIONES
# ─────────────────────────────────────────────────────────────────────────────

OPERATIONS = {
    "interpolation":             ("k",     "Inserta notas entre cada par"),
    "ultrapolation":             ("k",     "Inserta notas antes de cada nota"),
    "infrapolation":             ("k",     "Inserta notas después de cada nota"),
    "symmetric_interpolation":   ("k",     "Interpolación simétrica arriba/abajo"),
    "asymmetric_interpolation":  ("k,l",   "Interpolación asimétrica con k sup, l inf"),
    "infra_interpolation":       ("k,l",   "Combinado infra+inter"),
    "inter_infrapolation":       ("k,l",   "Combinado inter+infra"),
    "infra_ultrapolation":       ("k,l",   "Combinado infra+ultra"),
    "inter_ultrapolation":       ("k,l",   "Combinado inter+ultra"),
    "ultra_interpolation":       ("k,l",   "Combinado ultra+inter"),
    "infra_inter_ultrapolation": ("k,l,m", "Triple: infra+inter+ultra"),
    "inter_infra_interpolation": ("k,l,m", "Triple hapax: inter+infra+inter"),
    "ultra_infra_interpolation": ("k,l,m", "Triple hapax: ultra+infra+inter"),
}


def apply_operation(
    op: str,
    x: list[int],
    k: int | list[int] = 1,
    l: int | list[int] = 1,
    m: int = 1,
) -> list[int]:
    """Aplica la operación indicada al patrón x."""
    if op == "interpolation":
        return interpolation(x, k)
    elif op == "ultrapolation":
        return ultrapolation(x, k)
    elif op == "infrapolation":
        return infrapolation(x, k)
    elif op == "symmetric_interpolation":
        return symmetric_interpolation(x, k)
    elif op == "asymmetric_interpolation":
        lv = l if isinstance(l, list) else [l]
        kv = k if isinstance(k, list) else [k]
        return asymmetric_interpolation(x, kv, lv)
    elif op == "infra_interpolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return infra_interpolation(x, ki, li)
    elif op == "inter_infrapolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return inter_infrapolation(x, ki, li)
    elif op == "infra_ultrapolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return infra_ultrapolation(x, ki, li)
    elif op == "inter_ultrapolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return inter_ultrapolation(x, ki, li)
    elif op == "ultra_interpolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return ultra_interpolation(x, ki, li)
    elif op == "infra_inter_ultrapolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return infra_inter_ultrapolation(x, ki, li, m)
    elif op == "inter_infra_interpolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return inter_infra_interpolation(x, ki, li, m)
    elif op == "ultra_infra_interpolation":
        ki = k[0] if isinstance(k, list) else k
        li = l[0] if isinstance(l, list) else l
        return ultra_infra_interpolation(x, ki, li, m)
    else:
        raise ValueError(f"Operación desconocida: '{op}'")


# ─────────────────────────────────────────────────────────────────────────────
#  EXTENSIÓN A MÚLTIPLES OCTAVAS
# ─────────────────────────────────────────────────────────────────────────────

def extend_pattern_octaves(pattern: list[int], base_x: list[int], octaves: int) -> list[int]:
    """
    Extiende el patrón a múltiples octavas sumando el span del patrón base
    de forma repetida.
    """
    if octaves <= 1:
        return pattern
    span = base_x[-1] - base_x[0] if len(base_x) > 1 else 12
    result = list(pattern)
    for oct_i in range(1, octaves):
        # añadir el patrón (menos el primer elemento, que ya está al final del anterior)
        offset = oct_i * span
        result.extend(p + offset for p in pattern[1:])
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  EXPORTACIÓN A MIDI
# ─────────────────────────────────────────────────────────────────────────────

def export_midi(pattern: list[int], root: int, output: str, tempo_bpm: int, note_dur: int):
    """
    Exporta el patrón como secuencia melódica MIDI.
    Cada elemento del patrón se interpreta como semitono relativo a root.
    """
    try:
        import mido
    except ImportError:
        print(c("red", "Error: mido no instalado. pip install mido"), file=sys.stderr)
        sys.exit(1)

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    velocity = 80
    for pitch_offset in pattern:
        midi_pitch = max(0, min(127, root + pitch_offset))
        track.append(mido.Message("note_on",  note=midi_pitch, velocity=velocity, time=0))
        track.append(mido.Message("note_off", note=midi_pitch, velocity=0, time=note_dur))

    mid.save(output)


# ─────────────────────────────────────────────────────────────────────────────
#  PRESENTACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def render_pattern(pattern: list[int], root: int = 60, label: str = "") -> str:
    """Renderiza el patrón como lista de valores y nombres de notas."""
    note_names = [midi_to_name(max(0, min(127, root + p))) for p in pattern]
    vals = " ".join(c("cyan", str(v)) for v in pattern)
    names = c("gray", "  →  " + " ".join(note_names))
    prefix = c("bold", f"{label:<28}") if label else ""
    return f"  {prefix}[{vals}]{names}"


def print_operation_summary(op: str, x: list[int], result: list[int], root: int):
    params, desc = OPERATIONS.get(op, ("?", op))
    print(f"\n  {c('yellow', op)}")
    print(c("gray", f"  {desc}"))
    print(c("gray", f"  Patrón entrada ({len(x)} notas):  {x}"))
    print(c("gray", f"  Resultado     ({len(result)} notas):  {result}"))
    print(render_pattern(result, root))
    # mostrar intervalos del resultado
    intervals = [result[i+1] - result[i] for i in range(len(result)-1)]
    print(c("gray", f"  Intervalos: {intervals}"))


# ─────────────────────────────────────────────────────────────────────────────
#  MODO EXPLORACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def explore_all(x: list[int], root: int, k_range: range):
    """Genera todas las operaciones simples para cada k en k_range."""
    simple_ops = ["interpolation", "ultrapolation", "infrapolation"]
    composite_ops_kl = [
        "infra_interpolation", "inter_infrapolation",
        "infra_ultrapolation", "inter_ultrapolation", "ultra_interpolation",
    ]

    print(c("yellow", f"\n  EXPLORACIÓN COMPLETA — patrón: {x}\n"))

    for op in simple_ops:
        for k in k_range:
            try:
                result = apply_operation(op, x, k=k)
                label = f"{op}(k={k})"
                print(render_pattern(result, root, label))
            except Exception as e:
                pass

    print(c("gray", "\n  — Composites (k=l=1..2) —"))
    for op in composite_ops_kl:
        for k in [1, 2]:
            for l in [1, 2]:
                try:
                    result = apply_operation(op, x, k=k, l=l)
                    label = f"{op}(k={k},l={l})"
                    print(render_pattern(result, root, label))
                except Exception:
                    pass

    print()


# ─────────────────────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global USE_COLOR

    parser = argparse.ArgumentParser(
        description="slonimsky_explorer — Thesaurus of Musical Scales and Patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pattern", metavar="\"n n n\"",
                        help='Secuencia de pitch-offsets (ej: "0 4 8")')
    parser.add_argument("--notes", metavar="\"N N N\"",
                        help='Secuencia como nombres de nota (ej: "C4 E4 Ab4")')
    parser.add_argument("--op", metavar="OPERATION", default="interpolation",
                        help="Operación de Slonimsky (default: interpolation)")
    parser.add_argument("--k", metavar="OFFSET(S)", default="1",
                        help="Offset superior: entero o \"n1 n2 …\" (default: 1)")
    parser.add_argument("--l", metavar="OFFSET(S)", default="1",
                        help="Offset inferior para ops asimétricas (default: 1)")
    parser.add_argument("--m", type=int, default=1, metavar="OFFSET",
                        help="Tercer offset para operaciones triples (default: 1)")
    parser.add_argument("--root", type=int, default=60, metavar="N",
                        help="Pitch MIDI raíz para exportación/nombres (default: 60=C4)")
    parser.add_argument("--octaves", type=int, default=1, metavar="N",
                        help="Número de octavas a generar (default: 1)")
    parser.add_argument("--explore", action="store_true",
                        help="Explora todas las operaciones con k=1..3")
    parser.add_argument("--list-ops", action="store_true",
                        help="Listar todas las operaciones disponibles y salir")
    parser.add_argument("--infra-perm", type=int, metavar="M",
                        help="Mostrar la permutación sigma para infrapolación(M)")
    parser.add_argument("--output", metavar="FILE",
                        help="Exportar resultado a MIDI")
    parser.add_argument("--tempo", type=int, default=120, metavar="BPM",
                        help="Tempo del MIDI (default: 120)")
    parser.add_argument("--note-dur", type=int, default=240, metavar="D",
                        help="Duración de nota MIDI en ticks (default: 240)")
    parser.add_argument("--no-color", action="store_true",
                        help="Sin colores ANSI")
    args = parser.parse_args()

    if args.no_color:
        USE_COLOR = False

    print(f"\n{c('bold', 'SLONIMSKY EXPLORER')}  |  Thesaurus of Musical Scales and Patterns")
    print(c("gray", "─" * 68))

    # ── Listar operaciones ────────────────────────────────────────────────────
    if args.list_ops:
        print(c("yellow", "\n  Operaciones disponibles:\n"))
        for op, (params, desc) in OPERATIONS.items():
            print(f"  {c('cyan', f'{op:<32}')}{c('gray', f'params: {params:<6}')}  {desc}")
        print()
        return

    # ── Permutación de infrapolación ──────────────────────────────────────────
    if args.infra_perm is not None:
        m = args.infra_perm
        sigma = infrapolation_permutation(m)
        print(f"\n  Permutación sigma para infrapolación(m={m}):")
        print(c("cyan", f"  {sigma}"))
        print(c("gray", "  (los offsets del vector k deben satisfacer: k[sigma[i]] < k[sigma[i+1]])"))
        print()
        return

    # ── Parsear patrón ────────────────────────────────────────────────────────
    if args.notes:
        try:
            midi_pitches = [note_name_to_midi(n) for n in args.notes.split()]
            root = midi_pitches[0]
            x = [p - root for p in midi_pitches]
            if args.root == 60:
                args.root = root
        except ValueError as e:
            print(c("red", f"Error: {e}"), file=sys.stderr)
            sys.exit(1)
    elif args.pattern:
        try:
            x = parse_seq(args.pattern)
        except ValueError:
            print(c("red", "Error: --pattern requiere enteros separados por espacios"), file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    root = args.root

    # ── Modo exploración ──────────────────────────────────────────────────────
    if args.explore:
        explore_all(x, root, range(1, 4))
        return

    # ── Parsear offsets ───────────────────────────────────────────────────────
    try:
        k = parse_offsets(args.k)
        l = parse_offsets(args.l)
        m = args.m
    except ValueError as e:
        print(c("red", f"Error en offsets: {e}"), file=sys.stderr)
        sys.exit(1)

    # ── Aplicar operación ─────────────────────────────────────────────────────
    op = args.op
    if op not in OPERATIONS:
        print(c("red", f"Operación '{op}' no reconocida. Usa --list-ops para ver las disponibles."), file=sys.stderr)
        sys.exit(1)

    try:
        result = apply_operation(op, x, k=k, l=l, m=m)
    except Exception as e:
        print(c("red", f"Error al aplicar '{op}': {e}"), file=sys.stderr)
        sys.exit(1)

    # ── Extender a octavas ────────────────────────────────────────────────────
    if args.octaves > 1:
        result = extend_pattern_octaves(result, x, args.octaves)

    # ── Presentar resultado ───────────────────────────────────────────────────
    print_operation_summary(op, x, result, root)
    print()

    # ── Exportar MIDI ─────────────────────────────────────────────────────────
    if args.output:
        export_midi(result, root, args.output, args.tempo, args.note_dur)
        print(c("green", f"MIDI exportado: {args.output}"))
        print()


if __name__ == "__main__":
    main()
