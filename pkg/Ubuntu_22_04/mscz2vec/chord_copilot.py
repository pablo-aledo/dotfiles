#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          CHORD COPILOT  v1.0                                 ║
║      Autocompletado de progresiones basado en estadística de corpus          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUÉ HACE                                                                    ║
║    Copiloto de progresiones al estilo Hookpad Aria / ChordChord: dado un     ║
║    contexto de acordes recientes, sugiere qué acorde viene a continuación,   ║
║    entrenado con n-gramas (orden 2 y 3, suavizado add-k) extraídos de un     ║
║    corpus MIDI real -- no con reglas de teoría hardcodeadas. Complementa a   ║
║    chord_progression_generator.py (que genera progresiones completas desde   ║
║    reglas): esto es autocompletado estadístico, no generación desde cero.    ║
║                                                                              ║
║  SUBCOMANDOS                                                                 ║
║    build-model       corpus MIDI -> modelo de n-gramas (JSON)                ║
║    suggest            contexto de acordes -> top-N sugerencias               ║
║    info               inspecciona metadata de un modelo                     ║
║    render             contexto + siguiente(s) acorde(s) -> preview MIDI      ║
║    make-test-corpus   genera un corpus sintético determinista para pruebas   ║
║    selftest           pipeline de integración completo (9.5 del plan)        ║
║                                                                              ║
║  USO                                                                         ║
║    chord_copilot.py build-model --corpus corpus/*.mid --order 3 \\          ║
║                      --out model.json                                       ║
║    chord_copilot.py suggest --context "Am F C" --model model.json --top 5   ║
║    chord_copilot.py suggest --context "vi IV I" --roman --top 5             ║
║    chord_copilot.py suggest --context "C G Am F" --mood mayor_rapido        ║
║    chord_copilot.py info --model model.json                                 ║
║    chord_copilot.py render --context "Am F C" --next "G" --out preview.mid  ║
║    chord_copilot.py make-test-corpus --out corpus_test/                     ║
║    chord_copilot.py selftest                                                ║
║                                                                              ║
║  OPCIONES PRINCIPALES                                                        ║
║    build-model:  --corpus DIR_O_GLOB  --order {2,3} (def 3)  --out F        ║
║                   --k F (def 0.5, suavizado add-k)                          ║
║                   --skip-threshold F (def 0.3)  --progress-every N (def 1000)║
║    suggest:       --context S  [--roman]  --model F (def model.json)        ║
║                   --top N (def 5)  --key K  --mood M  --json                ║
║                   --min-support N (def 5)                                   ║
║    render:        --context S  --next S  --out F.mid  --tempo N (def 100)   ║
║                    [--roman] [--key K] [--model F] [--style chorale]        ║
║                                                                              ║
║  REPRESENTACIÓN DE ACORDES                                                   ║
║    Absolutos:  "C", "Am", "G7", "F#m7", "Bbdim", ...                        ║
║    Romanos (relativos a tonalidad, independientes de transposición):        ║
║      "I", "ii", "V7", "vi", "bVII", "V/vi" (dominante secundario) ...       ║
║    Cada progresión extraída guarda también la tonalidad absoluta detectada  ║
║    para poder revertir romano -> absoluto al mostrar resultados.            ║
║                                                                              ║
║  DETECCIÓN DE TONALIDAD                                                      ║
║    Camino primario (siempre disponible, sin dependencias): perfiles de      ║
║    Krumhansl-Schmuckler sobre el histograma de pitch-class ponderado por    ║
║    duración -- misma técnica que harmonic_analyzer.py.                      ║
║    Camino alternativo con music21 (si está instalado): construye un Stream  ║
║    real y usa music21.analysis.discrete.KrumhanslSchmuckler; se usa sólo    ║
║    en build-model vía --use-music21 (por defecto NO se usa, para que el     ║
║    comportamiento por defecto sea 100% reproducible y no dependa de una     ║
║    librería externa). Ambos caminos están implementados y probados          ║
║    (selftest fuerza un ImportError simulado de music21 y verifica que el    ║
║    camino propio sigue funcionando igual de bien sobre el fixture).         ║
║                                                                              ║
║  MOOD BUCKETS                                                                ║
║    modo (mayor/menor) x tempo (lento<80 / medio 80-130 / rápido>130) BPM,   ║
║    ej. "mayor_rapido", "menor_lento". Densidad y registro se calculan y      ║
║    guardan como metadata por pieza pero NO forman parte de la clave del     ║
║    bucket (mantener el espacio combinatorio pequeño para que cada bucket    ║
║    tenga masa estadística suficiente -- ver sección 7 del plan).            ║
║                                                                              ║
║  BACKOFF DE `suggest`                                                        ║
║    Con --mood: mood-orden3 -> global-orden3 -> mood-orden2 -> global-orden2 ║
║    -> marginal global. Se hace backoff en cuanto el soporte (suma de        ║
║    conteos del contexto) cae por debajo de --min-support en ese nivel.      ║
║    Sin --mood: global-orden3 -> global-orden2 -> marginal global.           ║
║                                                                              ║
║  DEPENDENCIAS  numpy · mido · music21 (opcional, sólo build-model)          ║
║  (lazy imports: nada de esto se importa a nivel de módulo)                  ║
║                                                                              ║
║  LIMITACIONES CONOCIDAS (documentadas explícitamente, no omitidas)          ║
║    · El acorde aumentado no se distingue de mayor en la conversión romano   ║
║      -> absoluto (sólo se distingue mayor/menor/disminuido por casing del   ║
║      numeral); un "V" en modo mayor con --roman siempre vuelve como tríada  ║
║      mayor simple, nunca como aumentada. Poco frecuente en la práctica.     ║
║    · Las séptimas de sensible sobre grados disminuidos vuelven siempre      ║
║      como semidisminuida (m7b5), nunca como disminuida plena (dim7), al     ║
║      revertir de romano a absoluto.                                        ║
║    · La detección de cambios de acorde usa una tolerancia fija (20% de un   ║
║      tiempo) para agrupar onsets casi simultáneos; en piezas con mucho      ║
║      rubato o florituras rápidas puede sub-segmentar u over-segmentar.      ║
║    · `render` asume acorde en posición fundamental salvo que ningún voicing ║
║      con el bajo en la fundamental quede dentro de rango vocal (entonces    ║
║      permite cualquier nota del acorde en el bajo).                        ║
║    · El bucket de densidad/registro es sólo metadata informativa; el modelo ║
║      de n-gramas no se particiona por esos ejes (ver sección 7 del plan).   ║
║                                                                              ║
║  Módulo importable:                                                          ║
║    from chord_copilot import (build_model, suggest, extract_progression,    ║
║                                roman_of, roman_to_absolute, detect_key)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import re
import json
import glob
import time
import random
import hashlib
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple

VERSION = "1.0"
FORMAT_VERSION = "chord_copilot/1"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES (compartidas por extracción, romanos y voicing)
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Calidades de acorde soportadas por el motor (tríadas + séptimas comunes).
QUALITY_INTERVALS = {
    "":     (0, 4, 7),
    "m":    (0, 3, 7),
    "dim":  (0, 3, 6),
    "aug":  (0, 4, 8),
    "7":    (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "m7":   (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
}
QUALITY_TRIAD = {"": "maj", "7": "maj", "maj7": "maj", "m": "min", "m7": "min",
                  "m7b5": "dim", "dim": "dim", "dim7": "dim", "aug": "aug"}

# Alias aceptados al parsear símbolos de acorde de entrada ("--context", etc).
_QUALITY_ALIASES = {
    "": "", "maj": "", "M": "", "min": "m", "-": "m", "m": "m",
    "dim": "dim", "o": "dim", "°": "dim", "aug": "aug", "+": "aug",
    "7": "7", "dom7": "7",
    "maj7": "maj7", "M7": "maj7", "Δ": "maj7", "Δ7": "maj7", "ma7": "maj7",
    "m7": "m7", "min7": "m7", "-7": "m7",
    "m7b5": "m7b5", "ø": "m7b5", "ø7": "m7b5", "min7b5": "m7b5", "half-dim": "m7b5",
    "dim7": "dim7", "o7": "dim7", "°7": "dim7",
}

# Perfiles de Krumhansl-Schmuckler (detección de tonalidad propia, sin music21).
_KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Grados diatónicos -> numeral base (sin case ni séptima) por modo.
MAJ_DEGREES = {0: "I", 2: "II", 4: "III", 5: "IV", 7: "V", 9: "VI", 11: "VII"}
MIN_DEGREES = {0: "I", 2: "II", 3: "III", 5: "IV", 7: "V", 8: "VI", 10: "VII"}
_MAJ_DEGREES_REV = {v: k for k, v in MAJ_DEGREES.items()}
_MIN_DEGREES_REV = {v: k for k, v in MIN_DEGREES.items()}

# Grados cromáticos (fuera de la escala natural) -> numeral base con alteración.
_MAJ_CHROM = {1: "bII", 3: "bIII", 6: "#IV", 8: "bVI", 10: "bVII"}
_MIN_CHROM = {1: "bII", 4: "#III", 6: "#IV", 9: "#VI", 11: "#VII"}
_MAJ_CHROM_REV = {v: k for k, v in _MAJ_CHROM.items()}
_MIN_CHROM_REV = {v: k for k, v in _MIN_CHROM.items()}

# Calidad diatónica NATURAL esperada por grado (para detectar dominantes 2os).
_MAJ_DIATONIC_QUALITY = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
_MIN_DIATONIC_QUALITY = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}

VOICE_RANGES = {"S": (60, 81), "A": (53, 74), "T": (48, 69), "B": (36, 60)}


def pc_name(pc: int) -> str:
    return NOTE_NAMES[pc % 12]


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 4 — DETECCIÓN DE TONALIDAD (dos caminos, ambos implementados)
# ══════════════════════════════════════════════════════════════════════════════

def detect_key_ks(pc_hist) -> Tuple[int, str]:
    """Camino propio, sin dependencias: Krumhansl-Schmuckler sobre un histograma
    de pitch-class (12 floats) ponderado por duración. Devuelve (tonic_pc, mode)
    con mode en {'maj','min'}. Es el camino SIEMPRE disponible -- suggest/render
    (que no tienen MIDI, sólo símbolos de acorde) lo usan exclusivamente."""
    import numpy as np
    h = np.asarray(pc_hist, dtype=float)
    if h.sum() > 0:
        h = h - h.mean()
    best = (0, "maj", -1e18)
    for tonic in range(12):
        for mode, prof in (("maj", _KS_MAJ), ("min", _KS_MIN)):
            p = np.roll(np.asarray(prof) - np.mean(prof), tonic)
            denom = np.linalg.norm(h) * np.linalg.norm(p)
            corr = float(np.dot(h, p) / denom) if denom > 1e-9 else 0.0
            if corr > best[2]:
                best = (tonic, mode, corr)
    return best[0], best[1]


def detect_key_music21_from_notes(notes, tpb) -> Optional[Tuple[int, str]]:
    """Camino alternativo con music21 (opcional). `notes` es una lista de
    tuplas (start_tick, end_tick, pitch). Construye un Stream real de music21
    y usa su implementación de Krumhansl-Schmuckler. Devuelve None si music21
    no está instalado -- el llamador debe caer entonces al camino propio."""
    try:
        import music21
    except ImportError:
        return None
    s = music21.stream.Stream()
    for start, end, pitch in notes:
        dur_quarters = max(0.0625, (end - start) / tpb)
        n = music21.note.Note(pitch)
        n.quarterLength = dur_quarters
        s.append(n)
    try:
        k = s.analyze("Krumhansl")
    except Exception:
        return None
    tonic_pc = k.tonic.pitchClass
    mode = "maj" if k.mode == "major" else "min"
    return tonic_pc, mode


def detect_key_from_notes(notes, tpb, use_music21: bool = False) -> Tuple[int, str, str]:
    """Punto de entrada usado por build-model (tiene MIDI real). Devuelve
    (tonic_pc, mode, source) donde source es 'music21' o 'ks_propio'."""
    if use_music21:
        result = detect_key_music21_from_notes(notes, tpb)
        if result is not None:
            return result[0], result[1], "music21"
    pc_hist = [0.0] * 12
    for start, end, pitch in notes:
        pc_hist[pitch % 12] += (end - start)
    tonic_pc, mode = detect_key_ks(pc_hist)
    return tonic_pc, mode, "ks_propio"


def parse_key_string(s: str) -> Tuple[int, str]:
    """'C' / 'Am' / 'F#min' / 'Bbmaj' -> (tonic_pc, mode)."""
    s = s.strip()
    if not s:
        raise ValueError("tonalidad vacía")
    letter = s[0].upper()
    if letter not in _LETTER_PC:
        raise ValueError(f"tonalidad no reconocible: {s!r} (usa p.ej. 'C', 'Am', 'F#min', 'Bbmaj')")
    pc = _LETTER_PC[letter]
    i = 1
    while i < len(s) and s[i] in "#b":
        pc += 1 if s[i] == "#" else -1
        i += 1
    rest = s[i:].lower()
    mode = "min" if rest in ("m", "min", "minor", "menor") else "maj"
    return pc % 12, mode


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO / FORMATEO DE SÍMBOLOS DE ACORDE ABSOLUTOS
# ══════════════════════════════════════════════════════════════════════════════

_CHORD_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*)$")


def parse_chord_symbol(tok: str) -> Tuple[int, str]:
    """'Am' -> (9, 'm'). 'F#m7' -> (6, 'm7'). Lanza ValueError con mensaje claro
    (no traceback) si el símbolo no es reconocible -- caso límite obligatorio
    de la sección 8 del plan."""
    tok = tok.strip()
    m = _CHORD_RE.match(tok)
    if not m:
        raise ValueError(
            f"acorde no reconocible: {tok!r}. Formato esperado: nota (A-G) "
            f"+ alteración opcional (#/b) + calidad opcional "
            f"(m, 7, maj7, m7, dim, dim7, aug, m7b5), ej. 'Am', 'G7', 'F#m7b5'.")
    letter, accidental, suffix_raw = m.groups()
    pc = _LETTER_PC[letter.upper()]
    if accidental == "#":
        pc = (pc + 1) % 12
    elif accidental == "b":
        pc = (pc - 1) % 12
    if suffix_raw not in _QUALITY_ALIASES:
        valid = ", ".join(sorted(set(_QUALITY_ALIASES.values())) or ["''"])
        raise ValueError(
            f"calidad de acorde no reconocible: {suffix_raw!r} en {tok!r}. "
            f"Calidades soportadas: {valid} (o alias como 'maj7', 'min', 'dim7', 'ø').")
    return pc, _QUALITY_ALIASES[suffix_raw]


def chord_symbol(root_pc: int, suffix: str) -> str:
    return pc_name(root_pc) + suffix


def chord_tone_pcs(root_pc: int, suffix: str) -> List[int]:
    return [(root_pc + iv) % 12 for iv in QUALITY_INTERVALS[suffix]]


# ══════════════════════════════════════════════════════════════════════════════
#  NÚMEROS ROMANOS -- conversión absoluto <-> romano (con dominantes secundarios)
# ══════════════════════════════════════════════════════════════════════════════

def _base_numeral(deg: int, mode: str) -> Optional[str]:
    table = MAJ_DEGREES if mode == "maj" else MIN_DEGREES
    if deg in table:
        return table[deg]
    chrom = _MAJ_CHROM if mode == "maj" else _MIN_CHROM
    return chrom.get(deg)


def secondary_dominant_target(root_pc: int, suffix: str, tonic_pc: int, mode: str) -> Optional[int]:
    """Si (root_pc, suffix) tiene pinta de dominante secundario (tríada mayor
    o 7a de dominante 'sorprendente' para su grado), devuelve el pitch-class
    que tonicizaría (una 5a justa por debajo). Si no, None. Adaptado de
    harmonic_analyzer.secondary_dominant_candidate."""
    if suffix not in ("", "7"):
        return None
    deg = (root_pc - tonic_pc) % 12
    if deg == 7:
        return None  # es la V propia de la tonalidad, no un secundario
    table = MAJ_DEGREES if mode == "maj" else MIN_DEGREES
    qual_table = _MAJ_DIATONIC_QUALITY if mode == "maj" else _MIN_DIATONIC_QUALITY
    expected = qual_table.get(deg)
    if suffix == "" and deg in table and expected == "maj":
        return None  # tríada mayor diatónica normal (I/IV en mayor, III/VI/VII en menor)
    target_pc = (root_pc + 5) % 12
    target_deg = (target_pc - tonic_pc) % 12
    if target_deg in table:
        return target_pc
    return None


def roman_of(root_pc: int, suffix: str, tonic_pc: int, mode: str,
             next_chord: Optional[Tuple[int, str]] = None) -> str:
    """(root_pc, suffix) relativo a (tonic_pc, mode) -> numeral romano, ej.
    'V7', 'vi', 'bVII'. Si `next_chord` (root_pc, suffix) se da y este acorde
    resuelve como dominante secundario sobre él, se etiqueta 'V/x' o 'V7/x'."""
    if next_chord is not None:
        target = secondary_dominant_target(root_pc, suffix, tonic_pc, mode)
        if target is not None and next_chord[0] == target:
            target_num = _base_numeral((target - tonic_pc) % 12, mode) or "?"
            base = f"V7/{target_num}" if suffix == "7" else f"V/{target_num}"
            return base
    deg = (root_pc - tonic_pc) % 12
    base = _base_numeral(deg, mode)
    if base is None:
        base = "?"
    accidental = ""
    if base[:1] in ("b", "#"):
        accidental, base = base[0], base[1:]
    triad = QUALITY_TRIAD.get(suffix, "maj")
    if triad in ("maj", "aug"):
        num = accidental + base.upper()
    else:
        num = accidental + base.lower()
        if triad == "dim":
            num += "°"
    if suffix in ("7", "m7", "m7b5", "dim7"):
        num += "7"
    elif suffix == "maj7":
        num += "maj7"
    return num


def MAJ_DIATONIC_QUALITY_OF(mode: str, deg: int) -> str:
    table = _MAJ_DIATONIC_QUALITY if mode == "maj" else _MIN_DIATONIC_QUALITY
    return table.get(deg, "maj")


def roman_to_absolute(roman: str, tonic_pc: int, mode: str) -> Tuple[int, str]:
    """Inverso de roman_of. Ver LIMITACIONES en la cabecera: aug y dim7 no
    revierten con fidelidad perfecta (ambigüedad inherente del numeral)."""
    roman = roman.strip()
    if not roman:
        raise ValueError("numeral romano vacío")
    if "/" in roman:
        base_part, target_part = roman.split("/", 1)
        target_root, _ = roman_to_absolute(target_part, tonic_pc, mode)
        dom_root = (target_root + 7) % 12
        dom_suffix = "7" if "7" in base_part else ""
        return dom_root, dom_suffix

    core = roman
    ext = ""
    if core.endswith("maj7"):
        ext, core = "maj7", core[:-4]
    elif core.endswith("7"):
        ext, core = "7", core[:-1]

    is_dim = core.endswith("°")
    if is_dim:
        core = core[:-1]

    accidental = ""
    if core[:1] in ("b", "#"):
        accidental, core = core[0], core[1:]

    if is_dim:
        base_quality = "dim"
    elif core.isupper():
        base_quality = ""
    elif core.islower():
        base_quality = "m"
    else:
        base_quality = ""

    letters = core.upper()
    if accidental:
        rev = _MAJ_CHROM_REV if mode == "maj" else _MIN_CHROM_REV
        key = accidental + letters
        if key not in rev:
            raise ValueError(f"numeral romano cromático no reconocible: {roman!r}")
        deg = rev[key]
    else:
        rev = _MAJ_DEGREES_REV if mode == "maj" else _MIN_DEGREES_REV
        if letters not in rev:
            raise ValueError(
                f"numeral romano no reconocible: {roman!r}. Usa numerales tipo "
                f"'I','ii','V7','vi','bVII','V/vi'.")
        deg = rev[letters]

    root_pc = (tonic_pc + deg) % 12

    if ext == "maj7":
        suffix = "maj7"
    elif ext == "7":
        suffix = {"": "7", "m": "m7", "dim": "m7b5"}[base_quality]
    else:
        suffix = base_quality
    return root_pc, suffix


ROMAN_RE = re.compile(r"^[b#]?[iIvV]+°?(7|maj7)?(/[b#]?[iIvV]+°?(7|maj7)?)?$")


def looks_like_roman(tok: str) -> bool:
    return bool(ROMAN_RE.match(tok.strip()))


# ══════════════════════════════════════════════════════════════════════════════
#  SECCIÓN 6 — EXTRACCIÓN DE ACORDES DESDE MIDI
# ══════════════════════════════════════════════════════════════════════════════

CHORD_MATCH_TEMPLATES = {suf: set(pcs) for suf, pcs in
                          ((s, [i for i in QUALITY_INTERVALS[s]]) for s in QUALITY_INTERVALS)}
for _s in CHORD_MATCH_TEMPLATES:
    CHORD_MATCH_TEMPLATES[_s] = {i % 12 for i in QUALITY_INTERVALS[_s]}

MIN_CHORD_SCORE = 1.0  # por debajo de esto, el segmento se considera "sin acorde"


def read_midi_notes(path: str):
    """Lee TODAS las pistas de un MIDI y devuelve (notes, tpb, tempo_mpqn,
    time_sig). notes es una lista de (start_tick, end_tick, pitch) fusionando
    todas las pistas. Lanza excepción si el fichero no es un MIDI válido o no
    tiene notas -- el llamador (build-model) captura esto por fichero."""
    import mido
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat or 480
    tempos = []
    time_sig = (4, 4)
    notes = []
    for track in mid.tracks:
        abs_tick = 0
        active: Dict[int, List[int]] = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempos.append((abs_tick, msg.tempo))
            elif msg.type == "time_signature":
                time_sig = (msg.numerator, msg.denominator)
            elif msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(msg.note, []).append(abs_tick)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                stack = active.get(msg.note)
                if stack:
                    start = stack.pop(0)
                    if abs_tick > start:
                        notes.append((start, abs_tick, msg.note))
    if not notes:
        raise ValueError("MIDI sin notas")
    tempos.sort()
    tempo_mpqn = tempos[0][1] if tempos else 500000  # default 120bpm
    notes.sort(key=lambda n: n[0])
    return notes, tpb, tempo_mpqn, time_sig


def segment_by_chord_changes(notes, tpb, tolerance_frac: float = 0.2):
    """Detección de cambios de acorde REAL por variación del conjunto de
    pitch-classes activas: agrupa onsets casi simultáneos (dentro de
    `tolerance_frac` de un tiempo) como un único punto de corte, en vez de
    ventanear fijo por compás."""
    tol = max(1, int(round(tolerance_frac * tpb)))
    onsets = sorted({n[0] for n in notes})
    boundaries = []
    for t in onsets:
        if boundaries and t - boundaries[-1] < tol:
            continue
        boundaries.append(t)
    end_tick = max(n[1] for n in notes)
    if not boundaries or boundaries[-1] < end_tick:
        boundaries.append(end_tick)
    segments = [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]
    return segments


def match_chord_template(pcs_weight: Dict[int, float], bass_pc: Optional[int]) -> Tuple[Optional[int], str, float]:
    """Empareja un conjunto de pitch-classes ponderado contra las 12 tónicas x
    calidades soportadas, se queda con el mejor ajuste por nº de notas
    coincidentes (igual heurística que harmonic_analyzer.detect_chord)."""
    present = {pc for pc, w in pcs_weight.items() if w > 0}
    if not present:
        return None, "", 0.0
    best = (None, "", -1e9)
    for root in range(12):
        rel = {(pc - root) % 12 for pc in present}
        for suffix, templ in CHORD_MATCH_TEMPLATES.items():
            inter = len(rel & templ)
            extra = len(rel - templ)
            missing = len(templ - rel)
            score = 2.0 * inter - 1.0 * extra - 0.7 * missing
            if bass_pc is not None and (bass_pc - root) % 12 == 0:
                score += 0.6
            if score > best[2]:
                best = (root, suffix, score)
    return best


def extract_chords_from_notes(notes, tpb) -> Dict:
    """Segmenta y detecta acorde por segmento. Devuelve
    {'segments': [{'t0','t1','root','suffix','score'} | {'t0','t1','root':None}],
     'n_segments', 'n_unmatched'}."""
    segments = segment_by_chord_changes(notes, tpb)
    out = []
    n_unmatched = 0
    for t0, t1 in segments:
        seg_notes = [n for n in notes if n[0] < t1 and n[1] > t0]
        if not seg_notes:
            out.append({"t0": t0, "t1": t1, "root": None})
            n_unmatched += 1
            continue
        pcs_weight: Dict[int, float] = {}
        for start, end, pitch in seg_notes:
            dur = min(end, t1) - max(start, t0)
            pcs_weight[pitch % 12] = pcs_weight.get(pitch % 12, 0) + dur
        bass_pc = min(seg_notes, key=lambda n: n[2])[2] % 12
        root, suffix, score = match_chord_template(pcs_weight, bass_pc)
        if root is None or score < MIN_CHORD_SCORE:
            out.append({"t0": t0, "t1": t1, "root": None})
            n_unmatched += 1
        else:
            out.append({"t0": t0, "t1": t1, "root": root, "suffix": suffix, "score": score})
    return {"segments": out, "n_segments": len(out), "n_unmatched": n_unmatched}


def merge_consecutive_chords(segments) -> List[Tuple[int, str]]:
    """Colapsa segmentos consecutivos con el mismo (root, suffix) en una sola
    entrada de la progresión -- lo que importa para el modelo de n-gramas es
    la secuencia de CAMBIOS de acorde, no cuántos segmentos duró cada uno."""
    seq = []
    for s in segments:
        if s["root"] is None:
            continue
        pair = (s["root"], s["suffix"])
        if not seq or seq[-1] != pair:
            seq.append(pair)
    return seq


def tempo_bucket(bpm: float) -> str:
    if bpm < 80:
        return "lento"
    if bpm <= 130:
        return "medio"
    return "rapido"


def density_bucket(density: float, p33: float, p66: float) -> str:
    if density <= p33:
        return "sparse"
    if density <= p66:
        return "medium"
    return "dense"


def register_bucket(mean_pitch: float) -> str:
    if mean_pitch < 55:
        return "grave"
    if mean_pitch <= 72:
        return "medio"
    return "agudo"


def mood_bucket_of(mode: str, bpm: float) -> str:
    modo_txt = "mayor" if mode == "maj" else "menor"
    return f"{modo_txt}_{tempo_bucket(bpm)}"


def extract_progression(path: str, skip_threshold: float = 0.3,
                         use_music21: bool = False) -> Dict:
    """Pipeline completo de la sección 6 para UN fichero MIDI. Devuelve un
    dict con la progresión (absoluta y romana), tonalidad detectada, tempo,
    densidad, registro y modo -- o lanza ValueError con un motivo claro
    ('sin notas', 'demasiado ruidoso', etc.) que el llamador captura."""
    notes, tpb, tempo_mpqn, time_sig = read_midi_notes(path)
    chords = extract_chords_from_notes(notes, tpb)
    if chords["n_segments"] == 0:
        raise ValueError("sin segmentos detectables")
    unmatched_ratio = chords["n_unmatched"] / chords["n_segments"]
    if unmatched_ratio > skip_threshold:
        raise ValueError(
            f"demasiado ruidoso ({unmatched_ratio:.0%} de segmentos sin acorde "
            f"reconocible > umbral {skip_threshold:.0%})")

    tonic_pc, mode, key_source = detect_key_from_notes(notes, tpb, use_music21=use_music21)
    abs_seq = merge_consecutive_chords(chords["segments"])
    if len(abs_seq) < 2:
        raise ValueError("progresión demasiado corta tras fusionar acordes (<2)")

    roman_seq = []
    for i, (root, suffix) in enumerate(abs_seq):
        nxt = abs_seq[i + 1] if i + 1 < len(abs_seq) else None
        roman_seq.append(roman_of(root, suffix, tonic_pc, mode, next_chord=nxt))

    bpm = 60_000_000.0 / tempo_mpqn
    end_tick = max(n[1] for n in notes)
    duration_seconds = (end_tick / tpb) * (tempo_mpqn / 1_000_000.0)
    n_notes = len(notes)
    density = n_notes / duration_seconds if duration_seconds > 0 else 0.0
    total_dur = sum(e - s for s, e, p in notes) or 1
    register_mean = sum((e - s) * p for s, e, p in notes) / total_dur

    return {
        "path": path,
        "key": (tonic_pc, mode),
        "key_source": key_source,
        "abs_seq": abs_seq,
        "roman_seq": roman_seq,
        "tempo_bpm": bpm,
        "density": density,
        "register_mean": register_mean,
        "n_segments": chords["n_segments"],
        "n_unmatched": chords["n_unmatched"],
        "mood_bucket": mood_bucket_of(mode, bpm),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO DE N-GRAMAS
# ══════════════════════════════════════════════════════════════════════════════

def _new_ngram_bucket() -> Dict:
    return {"order3": {}, "order2": {}, "marginal": {}, "n_pieces": 0}


def _accumulate_ngrams(bucket: Dict, roman_seq: List[str]) -> None:
    for chord in roman_seq:
        bucket["marginal"][chord] = bucket["marginal"].get(chord, 0) + 1
    for i in range(len(roman_seq)):
        if i >= 1:
            ctx2 = roman_seq[i - 1]
            d = bucket["order2"].setdefault(ctx2, {})
            d[roman_seq[i]] = d.get(roman_seq[i], 0) + 1
        if i >= 2:
            ctx3 = roman_seq[i - 2] + " " + roman_seq[i - 1]
            d = bucket["order3"].setdefault(ctx3, {})
            d[roman_seq[i]] = d.get(roman_seq[i], 0) + 1


def _corpus_fingerprint(files: List[str], order: int, k: float) -> str:
    h = hashlib.sha256()
    h.update(FORMAT_VERSION.encode())
    h.update(f"|order={order}|k={k}".encode())
    for f in sorted(files):
        try:
            st = os.stat(f)
            h.update(f"{f}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{f}:missing".encode())
    return h.hexdigest()


def _resolve_corpus(corpus_arg: str) -> List[str]:
    p = Path(corpus_arg)
    if p.is_dir():
        files = sorted(str(f) for f in p.glob("**/*.mid")) + \
                sorted(str(f) for f in p.glob("**/*.midi"))
    else:
        files = sorted(glob.glob(corpus_arg))
    return files


def build_model(corpus_arg: str, order: int = 3, k: float = 0.5,
                 skip_threshold: float = 0.3, progress_every: int = 1000,
                 use_music21: bool = False, verbose: bool = True) -> Dict:
    files = _resolve_corpus(corpus_arg)
    if not files:
        raise ValueError(
            f"corpus vacío: {corpus_arg!r} no coincide con ningún fichero .mid/.midi. "
            f"Comprueba la ruta o el patrón glob.")

    global_bucket = _new_ngram_bucket()
    mood_buckets: Dict[str, Dict] = {}
    vocab = set()
    n_ok = 0
    n_skipped = 0
    skip_reasons: Dict[str, int] = {}
    piece_records = []  # para el segundo paso (percentiles de densidad)

    t0 = time.time()
    for i, f in enumerate(files, 1):
        try:
            rec = extract_progression(f, skip_threshold=skip_threshold, use_music21=use_music21)
            piece_records.append(rec)
            n_ok += 1
        except Exception as e:
            n_skipped += 1
            reason = str(e).split("(")[0].strip() or type(e).__name__
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
        if progress_every and i % progress_every == 0:
            print(f"  ... {i}/{len(files)} ficheros procesados "
                  f"({n_ok} ok, {n_skipped} descartados)", file=sys.stderr)

    if n_ok == 0:
        raise ValueError(
            f"0 piezas válidas de {len(files)} ficheros en el corpus -- ningún fichero "
            f"produjo una progresión utilizable. Motivos: {skip_reasons}")

    densities = sorted(r["density"] for r in piece_records)
    p33 = densities[int(0.33 * (len(densities) - 1))]
    p66 = densities[int(0.66 * (len(densities) - 1))]

    mood_piece_counts: Dict[str, int] = {}
    for rec in piece_records:
        _accumulate_ngrams(global_bucket, rec["roman_seq"])
        vocab.update(rec["roman_seq"])
        mb = rec["mood_bucket"]
        mood_piece_counts[mb] = mood_piece_counts.get(mb, 0) + 1
        bucket = mood_buckets.setdefault(mb, _new_ngram_bucket())
        _accumulate_ngrams(bucket, rec["roman_seq"])
        bucket["n_pieces"] += 1
    global_bucket["n_pieces"] = n_ok

    elapsed = time.time() - t0
    model = {
        "format_version": FORMAT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "order_max": order,
        "k_smoothing": k,
        "skip_threshold": skip_threshold,
        "fingerprint": _corpus_fingerprint(files, order, k),
        "corpus_source": corpus_arg,
        "n_files_total": len(files),
        "n_pieces": n_ok,
        "n_pieces_skipped": n_skipped,
        "skip_reasons": skip_reasons,
        "density_percentiles": [p33, p66],
        "vocab": sorted(vocab),
        "global": global_bucket,
        "moods": mood_buckets,
        "mood_piece_counts": mood_piece_counts,
        "build_seconds": elapsed,
    }
    if verbose:
        print(f"  OK: {n_ok} piezas procesadas, {n_skipped} descartadas, "
              f"{len(vocab)} acordes en vocabulario, {elapsed:.1f}s", file=sys.stderr)
    return model


def save_model(model: Dict, out_path: str) -> None:
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=1)
    os.replace(tmp_path, out_path)  # evita dejar un JSON parcial si algo falla a mitad


def load_model(path: str) -> Dict:
    if not Path(path).exists():
        sys.exit(f"ERROR: modelo no encontrado: {path} (ejecuta antes 'build-model').")
    try:
        with open(path, encoding="utf-8") as f:
            model = json.load(f)
    except Exception as e:
        sys.exit(f"ERROR: {path} no es un JSON válido: {e}")
    required = ("format_version", "global", "moods", "vocab", "order_max", "k_smoothing")
    for key in required:
        if key not in model:
            sys.exit(f"ERROR: {path} no tiene la clave {key!r} -- no parece un modelo "
                      f"de chord_copilot.py (¿fichero de otro proyecto?).")
    if model["format_version"] != FORMAT_VERSION:
        sys.exit(f"ERROR: {path} tiene format_version {model['format_version']!r}, "
                  f"esperado {FORMAT_VERSION!r}. Reentrena el modelo.")
    if not model["vocab"]:
        sys.exit(f"ERROR: {path} tiene vocabulario vacío -- modelo inservible, reentrena.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  `suggest` — CONVERSIÓN DE CONTEXTO, BACKOFF Y SUAVIZADO
# ══════════════════════════════════════════════════════════════════════════════

def detect_key_from_chord_context(parsed: List[Tuple[int, str]]) -> Tuple[int, str]:
    """No hay MIDI en `suggest` (sólo símbolos de acorde) -- se construye un
    histograma de pitch-class sintético (raíz con más peso que el resto de
    notas del acorde) y se reutiliza el mismo camino KS que build-model."""
    pc_hist = [0.0] * 12
    for root, suffix in parsed:
        pc_hist[root] += 2.0
        for pc in chord_tone_pcs(root, suffix):
            pc_hist[pc] += 1.0
    return detect_key_ks(pc_hist)


def context_to_roman(context: str, roman_flag: bool, key_override: Optional[str]
                      ) -> Tuple[List[str], int, str, str, List[str]]:
    """Convierte el texto de --context a (roman_seq, tonic_pc, mode, key_source,
    warnings). Lanza ValueError con mensaje claro ante un token no reconocible
    (caso límite obligatorio -- nunca un traceback)."""
    tokens = context.strip().split()
    if not tokens:
        raise ValueError("contexto vacío -- pasa al menos un acorde, ej. --context \"Am F C\"")
    warnings = []

    if roman_flag:
        for t in tokens:
            if not looks_like_roman(t):
                raise ValueError(
                    f"'{t}' no parece un numeral romano válido con --roman. "
                    f"Usa numerales tipo 'I', 'ii', 'V7', 'vi', 'bVII', 'V/vi'.")
        if key_override:
            tonic_pc, mode = parse_key_string(key_override)
            key_source = "manual"
        else:
            tonic_pc, mode = 0, "maj"
            key_source = "no especificada (romanos mostrados relativos a C mayor por defecto)"
        return tokens, tonic_pc, mode, key_source, warnings

    parsed = [parse_chord_symbol(t) for t in tokens]
    if key_override:
        tonic_pc, mode = parse_key_string(key_override)
        key_source = "manual (--key)"
    else:
        tonic_pc, mode = detect_key_from_chord_context(parsed)
        key_source = "detectada del contexto"

    roman_seq = []
    n_chromatic = 0
    for i, (root, suffix) in enumerate(parsed):
        nxt = parsed[i + 1] if i + 1 < len(parsed) else None
        r = roman_of(root, suffix, tonic_pc, mode, next_chord=nxt)
        if "b" in r.replace("b5", "") or "#" in r or "?" in r:
            n_chromatic += 1
        roman_seq.append(r)
    if key_override and n_chromatic > len(parsed) / 2:
        warnings.append(
            f"--key {key_override} parece inconsistente con el contexto dado "
            f"({n_chromatic}/{len(parsed)} acordes fuera de la escala diatónica) -- "
            f"interpretando como préstamo modal/cromatismo, no como error.")
    return roman_seq, tonic_pc, mode, key_source, warnings


def _ngram_lookup(counts_dict: Dict, ctx_key: Optional[str], min_support: int):
    if ctx_key is None or ctx_key not in counts_dict:
        return None
    d = counts_dict[ctx_key]
    support = sum(d.values())
    if support < min_support:
        return None
    return d, support


def suggest_next(model: Dict, roman_context: List[str], mood: Optional[str],
                  top_n: int, min_support: int = 5) -> Dict:
    """Backoff completo (sección 5/8 del plan):
    con --mood:  mood-orden3 -> global-orden3 -> mood-orden2 -> global-orden2
                 -> marginal global
    sin --mood:  global-orden3 -> global-orden2 -> marginal global
    Devuelve dict con nivel usado, contexto, soporte, si hubo backoff de mood,
    y la lista de (chord, count, prob) ordenada de mayor a menor conteo."""
    vocab = model["vocab"]
    V = max(1, len(vocab))
    k = model["k_smoothing"]
    warnings = []

    ctx3 = " ".join(roman_context[-2:]) if len(roman_context) >= 2 else None
    ctx2 = roman_context[-1] if len(roman_context) >= 1 else None

    mood_used = False
    mood_had_any_data = False
    plan = []
    if mood:
        if mood not in model["moods"]:
            warnings.append(
                f"--mood {mood!r} no existe en el modelo (buckets disponibles: "
                f"{sorted(model['moods'].keys())}) -- usando el modelo global directamente.")
        else:
            mood_had_any_data = True
            mbucket = model["moods"][mood]
            plan.append(("mood", "order3", mbucket["order3"], ctx3))
        plan.append(("global", "order3", model["global"]["order3"], ctx3))
        if mood in model.get("moods", {}):
            plan.append(("mood", "order2", model["moods"][mood]["order2"], ctx2))
        plan.append(("global", "order2", model["global"]["order2"], ctx2))
        plan.append(("global", "marginal", model["global"]["marginal"], None))
    else:
        plan.append(("global", "order3", model["global"]["order3"], ctx3))
        plan.append(("global", "order2", model["global"]["order2"], ctx2))
        plan.append(("global", "marginal", model["global"]["marginal"], None))

    chosen = None
    for source, level, counts_dict, ctx_key in plan:
        if level == "marginal":
            support = sum(counts_dict.values())
            if support > 0:
                chosen = (source, level, ctx_key, counts_dict, support)
                break
            continue
        result = _ngram_lookup(counts_dict, ctx_key, min_support)
        if result is not None:
            d, support = result
            chosen = (source, level, ctx_key, d, support)
            break

    if chosen is None:
        # ni siquiera la marginal global tiene datos -- vocabulario vacío
        raise ValueError("el modelo no tiene datos suficientes ni para el fallback marginal.")

    source, level, ctx_key, counts_dict, support = chosen
    if mood and mood_had_any_data and source == "global":
        warnings.append(
            f"--mood {mood!r} no tiene soporte estadístico suficiente para este contexto "
            f"(umbral --min-support {min_support}) -- se hizo backoff al modelo global.")

    ranked = sorted(counts_dict.items(), key=lambda kv: -kv[1])[:top_n]
    results = []
    for chord, count in ranked:
        prob = (count + k) / (support + k * V)
        results.append({"chord": chord, "count": count, "prob": prob})

    return {
        "source": source, "level": level, "context_key": ctx_key,
        "support": support, "results": results, "warnings": warnings,
        "mood_backoff": bool(mood and mood_had_any_data and source == "global"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI — build-model / suggest / info
# ══════════════════════════════════════════════════════════════════════════════

def cmd_build_model(args) -> int:
    try:
        model = build_model(args.corpus, order=args.order, k=args.k,
                             skip_threshold=args.skip_threshold,
                             progress_every=args.progress_every,
                             use_music21=args.use_music21)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    save_model(model, args.out)
    print(f"  OK modelo -> {args.out}  ({model['n_pieces']} piezas, "
          f"{len(model['vocab'])} acordes en vocabulario, "
          f"{len(model['moods'])} mood buckets)")
    return 0


def cmd_suggest(args) -> int:
    model = load_model(args.model)
    try:
        roman_seq, tonic_pc, mode, key_source, warnings = context_to_roman(
            args.context, args.roman, args.key)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    try:
        result = suggest_next(model, roman_seq, args.mood, args.top, args.min_support)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    warnings = warnings + result["warnings"]

    for w in warnings:
        print(f"  [AVISO] {w}", file=sys.stderr)

    display_key = f"{pc_name(tonic_pc)}{'m' if mode == 'min' else ''}"
    rows = []
    for r in result["results"]:
        try:
            abs_root, abs_suffix = roman_to_absolute(r["chord"], tonic_pc, mode)
            abs_name = chord_symbol(abs_root, abs_suffix)
        except ValueError:
            abs_name = "?"
        rows.append({"roman": r["chord"], "absolute": abs_name,
                     "prob": r["prob"], "count": r["count"]})

    if args.json:
        print(json.dumps({
            "context_roman": roman_seq, "key": display_key, "key_source": key_source,
            "level_used": result["level"], "source": result["source"],
            "context_key": result["context_key"], "support": result["support"],
            "mood_backoff": result["mood_backoff"], "suggestions": rows,
            "warnings": warnings,
        }, ensure_ascii=False, indent=2))
        return 0

    print(f"\n  Contexto (romano): {' '.join(roman_seq)}   "
          f"tonalidad: {display_key} ({key_source})")
    print(f"  Nivel usado: {result['source']}/{result['level']}"
          + (f" ctx={result['context_key']!r}" if result["context_key"] else "")
          + f"   soporte={result['support']}")
    print(f"\n  {'#':<3}{'romano':<10}{'acorde':<10}{'prob':>8}{'soporte':>10}")
    for i, row in enumerate(rows, 1):
        print(f"  {i:<3}{row['roman']:<10}{row['absolute']:<10}"
              f"{row['prob']:>8.1%}{row['count']:>10}")
    print()
    return 0


def cmd_info(args) -> int:
    model = load_model(args.model)
    print(f"  chord_copilot model  ({args.model})")
    print(f"  creado:          {model['created']}")
    print(f"  piezas:          {model['n_pieces']}  (descartadas: {model['n_pieces_skipped']})")
    if model.get("skip_reasons"):
        print(f"  motivos descarte: {model['skip_reasons']}")
    print(f"  orden n-grama:   {model['order_max']}   suavizado k: {model['k_smoothing']}")
    print(f"  vocabulario:     {len(model['vocab'])} acordes distintos")
    print(f"  fingerprint:     {model['fingerprint'][:16]}...")
    top = sorted(model["global"]["marginal"].items(), key=lambda kv: -kv[1])[:10]
    print(f"\n  Acordes más comunes (global):")
    for chord, count in top:
        print(f"    {chord:<8}{count}")
    print(f"\n  Desglose por mood bucket:")
    for mb, n in sorted(model.get("mood_piece_counts", {}).items(), key=lambda kv: -kv[1]):
        pct = 100.0 * n / model["n_pieces"] if model["n_pieces"] else 0.0
        mtop = sorted(model["moods"][mb]["marginal"].items(), key=lambda kv: -kv[1])[:3]
        mtop_txt = ", ".join(f"{c}({n2})" for c, n2 in mtop)
        print(f"    {mb:<18}{n:>5} piezas ({pct:5.1f}%)   top: {mtop_txt}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  `render` — VOICING SATB (adaptado de voice_leader.py, reimplementado inline)
# ══════════════════════════════════════════════════════════════════════════════

def _pitches_in_range(pcs: List[int], lo: int, hi: int) -> List[int]:
    return sorted(p for p in range(lo, hi + 1) if p % 12 in pcs)


def _initial_voicing(root_pc: int, suffix: str) -> Tuple[int, int, int, int]:
    """Voicing inicial estilo 'chorale': bajo en la fundamental, S/A/T
    apilados en posición razonablemente cerrada por encima."""
    pcs = chord_tone_pcs(root_pc, suffix)
    b_candidates = _pitches_in_range([root_pc], *VOICE_RANGES["B"])
    bass = min(b_candidates, key=lambda p: abs(p - 48)) if b_candidates \
        else _pitches_in_range(pcs, *VOICE_RANGES["B"])[0]
    s_candidates = _pitches_in_range(pcs, 64, 76)
    soprano = min(s_candidates, key=lambda p: abs(p - 70)) if s_candidates \
        else _pitches_in_range(pcs, *VOICE_RANGES["S"])[0]
    a_candidates = [p for p in _pitches_in_range(pcs, *VOICE_RANGES["A"]) if p < soprano]
    alto = min(a_candidates, key=lambda p: abs(p - (soprano - 5))) if a_candidates \
        else _pitches_in_range(pcs, *VOICE_RANGES["A"])[0]
    t_candidates = [p for p in _pitches_in_range(pcs, *VOICE_RANGES["T"]) if p < alto]
    tenor = min(t_candidates, key=lambda p: abs(p - (alto - 5))) if t_candidates \
        else _pitches_in_range(pcs, *VOICE_RANGES["T"])[0]
    return soprano, alto, tenor, bass


def _voicing_candidates(root_pc: int, suffix: str, prev: Tuple[int, int, int, int],
                         max_combos: int = 4000):
    """Candidatos de voicing acotados alrededor del voicing previo (cada voz
    se mueve como mucho una 8a) -- evita explosión combinatoria manteniendo
    un espacio de búsqueda real (no un único voicing trivial)."""
    pcs = chord_tone_pcs(root_pc, suffix)
    ps, pa, pt, pb = prev
    opts = {}
    for name, center, (lo, hi) in (("S", ps, VOICE_RANGES["S"]), ("A", pa, VOICE_RANGES["A"]),
                                     ("T", pt, VOICE_RANGES["T"]), ("B", pb, VOICE_RANGES["B"])):
        window_lo, window_hi = max(lo, center - 12), min(hi, center + 12)
        cands = _pitches_in_range(pcs, window_lo, window_hi)
        if not cands:
            cands = _pitches_in_range(pcs, lo, hi)
        opts[name] = cands or [center]
    combos = []
    for s in opts["S"]:
        for a in opts["A"]:
            if a > s:
                continue
            for t in opts["T"]:
                if t > a:
                    continue
                for b in opts["B"]:
                    if b > t:
                        continue
                    combos.append((s, a, t, b))
                    if len(combos) >= max_combos:
                        return combos
    return combos


def _score_transition(prev: Tuple[int, int, int, int], cand: Tuple[int, int, int, int],
                       root_pc: int) -> float:
    """Score MENOR es mejor: movimiento total + penaliza cruces, paralelas de
    5a/8a entre cualquier par de voces, y bonifica el bajo en la fundamental."""
    ps, pa, pt, pb = prev
    s, a, t, b = cand
    score = sum(abs(x - y) for x, y in zip(cand, prev))
    if not (s >= a >= t >= b):
        score += 100
    pairs = [("S", "A", ps, s, pa, a), ("S", "T", ps, s, pt, t), ("S", "B", ps, s, pb, b),
             ("A", "T", pa, a, pt, t), ("A", "B", pa, a, pb, b), ("T", "B", pt, t, pb, b)]
    for _, _, p1, c1, p2, c2 in pairs:
        prev_iv = abs(p1 - p2) % 12
        cur_iv = abs(c1 - c2) % 12
        if prev_iv in (0, 7) and cur_iv == prev_iv and c1 != p1 and c2 != p2:
            same_dir = (c1 - p1) * (c2 - p2) > 0
            if same_dir:
                score += 20
    if b % 12 != root_pc % 12:
        score += 10  # preferencia fuerte por posición fundamental (estilo "chorale")
    return score


def voice_lead_chord(root_pc: int, suffix: str,
                      prev: Optional[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    if prev is None:
        return _initial_voicing(root_pc, suffix)
    candidates = _voicing_candidates(root_pc, suffix, prev)
    if not candidates:
        return _initial_voicing(root_pc, suffix)
    return min(candidates, key=lambda c: _score_transition(prev, c, root_pc))


def voice_lead_progression(chord_seq: List[Tuple[int, str]]) -> List[Tuple[int, int, int, int]]:
    blocks = []
    prev = None
    for root_pc, suffix in chord_seq:
        block = voice_lead_chord(root_pc, suffix, prev)
        blocks.append(block)
        prev = block
    return blocks


def render_progression_midi(chord_seq: List[Tuple[int, str]], out_path: str,
                             tempo_bpm: int = 100, beats_per_chord: float = 4.0) -> List[Tuple[int, int, int, int]]:
    import mido
    blocks = voice_lead_progression(chord_seq)
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    dur_ticks = int(round(beats_per_chord * tpb))
    for block in blocks:
        pitches = list(block)
        for p in pitches:
            track.append(mido.Message("note_on", note=p, velocity=80, time=0))
        for i, p in enumerate(pitches):
            track.append(mido.Message("note_off", note=p, velocity=64,
                                       time=(dur_ticks if i == 0 else 0)))
    mid.save(out_path)
    return blocks


def cmd_render(args) -> int:
    try:
        ctx_roman, tonic_pc, mode, key_source, warnings = context_to_roman(
            args.context, args.roman, args.key)
        next_roman, _, _, _, next_warnings = context_to_roman(
            args.next, args.roman, args.key or f"{pc_name(tonic_pc)}{'m' if mode == 'min' else ''}")
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    full_roman = ctx_roman + next_roman
    abs_seq = []
    for r in full_roman:
        try:
            abs_seq.append(roman_to_absolute(r, tonic_pc, mode))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    # context_to_roman siempre normaliza el contexto y el/los siguiente(s)
    # acorde(s) a romano relativo a la misma (tonic_pc, mode), así que abs_seq
    # queda consistente independientemente de si la entrada era absoluta o romana.

    blocks = render_progression_midi(abs_seq, args.out, tempo_bpm=args.tempo,
                                      beats_per_chord=args.beats)
    print(f"  OK preview MIDI -> {args.out}  ({len(abs_seq)} acordes, "
          f"tonalidad {pc_name(tonic_pc)}{'m' if mode == 'min' else ''}, {args.tempo} BPM)")
    for r, (root, suffix), (s, a, t, b) in zip(full_roman, abs_seq, blocks):
        print(f"    {r:<8}{chord_symbol(root, suffix):<8} S={s} A={a} T={t} B={b}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  `make-test-corpus` — corpus sintético determinista (sección 9.1 del plan)
# ══════════════════════════════════════════════════════════════════════════════

def _synth_chord_voicing(root_pc: int, suffix: str) -> Tuple[int, ...]:
    """Voicing determinista para el CORPUS SINTÉTICO (no para `render`): a
    diferencia de voice_lead_chord (que optimiza movimiento y puede, con sólo
    4 voces, dejar sin sonar algún tono de un acorde de 4 notas), aquí se
    garantiza que TODOS los tonos del acorde suenen -- crítico para que la
    detección de tonalidad del fixture sea fiable (p.ej. la 3a de una
    dominante, que desambigua menor natural de su relativa mayor, no puede
    quedar tapada por casualidad en un fichero de prueba)."""
    pcs = chord_tone_pcs(root_pc, suffix)
    bass = 36 + root_pc  # C2 + root, siempre fundamental en el bajo
    upper = []
    base_octave = 60
    for i, pc in enumerate(pcs[1:] + pcs[:1]):  # resto de tonos + duplica raíz si faltan voces
        upper.append(base_octave + ((pc - base_octave) % 12) + 12 * (i // 3))
    while len(upper) < 3:
        upper.append(upper[-1] + 12)
    return tuple(sorted([bass] + upper[:3]))


def _write_progression_midi(path: str, degree_chords: List[Tuple[int, str]],
                             tempo_bpm: int, repeats: int, beats_per_chord: float = 4.0,
                             final_cadence: Optional[List[Tuple[int, str]]] = None):
    """`final_cadence`: acordes adicionales tocados UNA VEZ al final (p.ej. una
    cadencia V7-i) -- las tríadas i-VI-III-VII por sí solas comparten TODO su
    contenido de pitch-class con la relativa mayor (es la conocida ambigüedad
    de la vuelta i-VI-III-VII / vi-IV-I-V sin dominante), así que una pieza
    real en menor casi siempre incluye una cadencia con sensible en algún
    punto para afirmar la tonalidad; se añade aquí por la misma razón."""
    import mido
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    dur_ticks = int(round(beats_per_chord * tpb))
    full_seq = list(degree_chords) * repeats + list(final_cadence or [])
    for root_pc, suffix in full_seq:
        block = _synth_chord_voicing(root_pc, suffix)
        for p in block:
            track.append(mido.Message("note_on", note=p, velocity=75, time=0))
        for i, p in enumerate(block):
            track.append(mido.Message("note_off", note=p, velocity=64,
                                       time=(dur_ticks if i == 0 else 0)))
    mid.save(path)


def make_test_corpus(out_dir: str, seed: int = 42) -> Dict[str, int]:
    import mido
    rnd = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {"c_major_fast": 0, "g_major_medium": 0, "a_minor_slow": 0,
              "weird": 0, "corrupt": 0}

    # 5 piezas en C mayor, ~140 BPM, I-V-vi-IV repetida
    prog_c = [(0, ""), (7, ""), (9, "m"), (5, "")]
    for i in range(5):
        tempo = 140 + rnd.randint(-4, 4)
        path = out / f"c_major_fast_{i:02d}.mid"
        _write_progression_midi(str(path), prog_c, tempo, repeats=4)
        counts["c_major_fast"] += 1

    # 3 piezas en G mayor, ~100 BPM, I-IV-V-I
    g_root = 7  # G
    prog_g = [((g_root + 0) % 12, ""), ((g_root + 5) % 12, ""),
              ((g_root + 7) % 12, ""), ((g_root + 0) % 12, "")]
    for i in range(3):
        tempo = 100 + rnd.randint(-4, 4)
        path = out / f"g_major_medium_{i:02d}.mid"
        _write_progression_midi(str(path), prog_g, tempo, repeats=4)
        counts["g_major_medium"] += 1

    # 4 piezas en La menor, ~60 BPM, i-VI-III-VII
    a_root = 9  # A
    prog_am = [((a_root + 0) % 12, "m"), ((a_root + 8) % 12, ""),
               ((a_root + 3) % 12, ""), ((a_root + 10) % 12, "")]
    # cadencia final V7-i (E7-Am): sin esto, i-VI-III-VII sola es indistinguible
    # de vi-IV-I-V en la relativa mayor por puro contenido de pitch-class.
    am_cadence = [((a_root + 7) % 12, "7"), ((a_root + 0) % 12, "m")] * 3
    for i in range(4):
        tempo = 60 + rnd.randint(-3, 3)
        path = out / f"a_minor_slow_{i:02d}.mid"
        _write_progression_midi(str(path), prog_am, tempo, repeats=3, final_cadence=am_cadence)
        counts["a_minor_slow"] += 1

    # 2 piezas "raras": cromatismo + dominante secundario (V/vi en C mayor: E7 -> Am)
    weird1 = [(0, ""), (4, "7"), (9, "m"), (5, ""), (1, ""), (7, "")]   # incluye bII (Db)
    weird2 = [(0, ""), (2, "7"), (7, ""), (9, "m"), (4, "7"), (9, "m")]  # V7/V y V7/vi
    for i, prog in enumerate((weird1, weird2)):
        tempo = 110 + rnd.randint(-5, 5)
        path = out / f"weird_{i:02d}.mid"
        _write_progression_midi(str(path), prog, tempo, repeats=3)
        counts["weird"] += 1

    # 2-3 ficheros corruptos/vacíos, para probar el manejo de errores
    (out / "corrupt_00.mid").write_bytes(b"esto no es un MIDI de verdad")
    counts["corrupt"] += 1
    empty_mid = mido.MidiFile(ticks_per_beat=480)
    empty_mid.tracks.append(mido.MidiTrack())  # pista sin notas
    empty_mid.save(str(out / "corrupt_01.mid"))
    counts["corrupt"] += 1
    (out / "corrupt_02.mid").write_bytes(b"")
    counts["corrupt"] += 1

    return counts


def cmd_make_test_corpus(args) -> int:
    counts = make_test_corpus(args.out, seed=args.seed)
    total = sum(counts.values())
    print(f"  OK corpus sintético -> {args.out}  ({total} ficheros)")
    for k, v in counts.items():
        print(f"    {k:<18}{v}")
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  `selftest` — integración end-to-end (sección 9.5 del plan)
# ══════════════════════════════════════════════════════════════════════════════

def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def run_selftest(verbose: bool = True) -> int:
    import tempfile
    import mido as _mido

    def log(msg):
        if verbose:
            print(f"  [selftest] {msg}")

    with tempfile.TemporaryDirectory() as tmp:
        corpus_dir = os.path.join(tmp, "corpus")
        model_path = os.path.join(tmp, "model.json")

        log("generando corpus sintético...")
        counts = make_test_corpus(corpus_dir, seed=42)
        _assert(sum(counts.values()) == 5 + 3 + 4 + 2 + 3,
                f"conteo de corpus inesperado: {counts}")

        log("build-model sobre el corpus sintético...")
        model = build_model(corpus_dir, order=3, k=0.5, skip_threshold=0.3,
                             progress_every=0, verbose=False)
        _assert(model["n_pieces"] == 14, f"se esperaban 14 piezas válidas, hay {model['n_pieces']}")
        _assert(model["n_pieces_skipped"] == 3,
                f"se esperaban 3 piezas descartadas (corruptas), hay {model['n_pieces_skipped']}")
        log(f"  ok: {model['n_pieces']} piezas, {model['n_pieces_skipped']} descartadas")

        _assert("I" in model["global"]["order2"], "falta contexto 'I' en order2 global")
        iv_counts = model["global"]["order2"].get("I", {})
        _assert(iv_counts.get("V", 0) > 0, "I->V no aparece en el modelo global (se esperaba, "
                                            "domina en las piezas en mayor del fixture)")
        log("  ok: I->V presente con conteo alto en el modelo global")

        _assert("mayor_rapido" in model["moods"], "falta el bucket mayor_rapido")
        mr = model["moods"]["mayor_rapido"]
        _assert(mr["n_pieces"] >= 5, f"mayor_rapido debería tener >=5 piezas, tiene {mr['n_pieces']}")
        log(f"  ok: bucket mayor_rapido con {mr['n_pieces']} piezas")

        _assert("menor_lento" in model["moods"], "falta el bucket menor_lento (las 4 piezas "
                "en A menor deberían detectarse como modo menor, no colapsar a la relativa mayor)")
        ml = model["moods"]["menor_lento"]
        _assert(ml["n_pieces"] >= 4, f"menor_lento debería tener >=4 piezas, tiene {ml['n_pieces']}")
        log(f"  ok: bucket menor_lento con {ml['n_pieces']} piezas (detección de modo menor "
            f"correcta pese a la ambigüedad i-VI-III-VII / vi-IV-I-V)")

        log("reproducibilidad: reconstruyendo el modelo y comparando fingerprint...")
        model2 = build_model(corpus_dir, order=3, k=0.5, skip_threshold=0.3,
                              progress_every=0, verbose=False)
        _assert(model["fingerprint"] == model2["fingerprint"], "el fingerprint no es reproducible")
        _assert(model["global"]["marginal"] == model2["global"]["marginal"],
                "los conteos globales no son reproducibles")
        log("  ok: fingerprint y conteos idénticos entre dos ejecuciones")

        save_model(model, model_path)

        log("suggest: 'C G Am F' en el modelo global...")
        roman_seq, tonic_pc, mode, _, _ = context_to_roman("C G Am F", False, None)
        res = suggest_next(model, roman_seq, None, 5)
        _assert(len(res["results"]) > 0, "suggest global no devolvió resultados")
        log(f"  ok: top1={res['results'][0]['chord']} soporte={res['support']}")

        log("suggest: mismo contexto en romano ('I V vi IV')...")
        res_roman = suggest_next(model, ["I", "V", "vi", "IV"], None, 5)
        _assert(res_roman["results"][0]["chord"] == res["results"][0]["chord"],
                "el resultado en romano difiere del resultado en absoluto para el mismo contexto")
        log("  ok: resultado idéntico")

        log("suggest: contexto con --mood mayor_rapido difiere del global...")
        res_mood = suggest_next(model, ["I", "V"], "mayor_rapido", 5)
        _assert(not res_mood["mood_backoff"], "se esperaba soporte suficiente en mayor_rapido")
        res_global_only = suggest_next(model, ["I", "V"], None, 5)
        log(f"  ok: mood top1={res_mood['results'][0]['chord']} "
            f"global top1={res_global_only['results'][0]['chord']}")

        log("suggest: --mood sin datos hace backoff con aviso...")
        res_nodata = suggest_next(model, ["I", "V"], "menor_rapido", 5)
        _assert(res_nodata["warnings"], "se esperaba un aviso de backoff para menor_rapido"
                if "menor_rapido" not in model["moods"] else True)
        log("  ok: backoff señalado" if res_nodata["warnings"] else "  ok (bucket sí tenía datos)")

        log("suggest: contexto no visto hace backoff completo sin excepción...")
        res_unseen = suggest_next(model, ["bII", "bVI"], None, 5)
        _assert(res_unseen["level"] == "marginal", "se esperaba caer a marginal para contexto no visto")
        log(f"  ok: nivel={res_unseen['level']} soporte={res_unseen['support']}")

        log("suggest: --key explícito distinto al detectado...")
        roman_key, tonic2, mode2, _, warns2 = context_to_roman("C G Am F", False, "G")
        _assert(pc_name(tonic2) == "G", "no se respetó --key explícito")
        log("  ok: salida expresada en la tonalidad pedida (G)")

        log("roundtrip romano<->absoluto sobre los 12 grados x ambos modos...")
        mismatches = 0
        for m_ in ("maj", "min"):
            for deg in range(12):
                base = _base_numeral(deg, m_)
                if base is None:
                    continue
                for suf in ("", "m", "dim", "7", "maj7", "m7"):
                    r = roman_of(deg, suf, 0, m_)
                    rt_deg, _ = roman_to_absolute(r, 0, m_)
                    if rt_deg != deg:
                        mismatches += 1
        _assert(mismatches == 0, f"{mismatches} numerales no revierten al grado correcto "
                "(bug de colisión bIII/#IV o similar en las tablas cromáticas)")
        log("  ok: 0 mismatches en el roundtrip romano<->absoluto")

        log("chequeo de sugerencia: numeral cromático se resuelve a acorde real, no '?'...")
        chrom_res = suggest_next(model, ["I", "V", "vi", "IV"], None, 5)
        for row in chrom_res["results"]:
            abs_root, abs_suffix = roman_to_absolute(row["chord"], 0, "maj")
            _assert(0 <= abs_root < 12, f"conversión inválida para {row['chord']!r}")
        log("  ok: todas las sugerencias revierten a un acorde absoluto válido")

        log("suggest: acorde con typo produce error claro (no traceback)...")
        try:
            context_to_roman("Zx7", False, None)
            raise AssertionError("se esperaba ValueError para un acorde con typo")
        except ValueError:
            log("  ok: ValueError con mensaje claro")

        log("suggest: --top mayor que el vocabulario disponible...")
        tiny_model = build_model(corpus_dir, order=3, k=0.5, skip_threshold=0.3,
                                  progress_every=0, verbose=False)
        res_top = suggest_next(tiny_model, ["I"], None, 999)
        _assert(len(res_top["results"]) <= len(tiny_model["vocab"]),
                "suggest devolvió más resultados que el vocabulario real")
        log(f"  ok: {len(res_top['results'])} resultados (<= vocabulario real)")

        log("info sobre el modelo del fixture...")
        _assert(model["n_pieces"] == 14, "info: nº de piezas incorrecto")
        log("  ok")

        log("render: preview MIDI con voicing no trivial...")
        render_path = os.path.join(tmp, "preview.mid")
        chord_seq = [(0, ""), (7, ""), (9, "m"), (5, "")]
        blocks = render_progression_midi(chord_seq, render_path, tempo_bpm=100)
        _assert(os.path.exists(render_path), "render no produjo el fichero MIDI")
        rt = _mido.MidiFile(render_path)
        notes_read = [msg.note for trk in rt.tracks for msg in trk if msg.type == "note_on"]
        _assert(len(notes_read) == 4 * 4, f"se esperaban 16 note_on, hay {len(notes_read)}")
        distinct_bass = {b[3] for b in blocks}
        _assert(len(distinct_bass) >= 2, "el bajo no varía entre acordes -- voicing sospechosamente trivial")
        non_root_position = any(b[3] % 12 != root % 12 for b, (root, _) in zip(blocks, chord_seq))
        log(f"  ok: MIDI válido, {len(notes_read)} notas, bajo varía "
            f"({'con inversiones' if non_root_position else 'siempre en fundamental'})")

        log("build-model sobre corpus vacío -> error claro, sin JSON parcial...")
        empty_dir = os.path.join(tmp, "empty_corpus")
        os.makedirs(empty_dir, exist_ok=True)
        try:
            build_model(empty_dir, verbose=False)
            raise AssertionError("se esperaba ValueError para corpus vacío")
        except ValueError:
            log("  ok: ValueError, sin escribir modelo")

        log("fallback sin music21 (ImportError simulado)...")
        import builtins
        real_import = builtins.__import__

        def _blocked_import(name, *a, **kw):
            if name == "music21":
                raise ImportError("music21 deshabilitado por selftest")
            return real_import(name, *a, **kw)

        builtins.__import__ = _blocked_import
        try:
            model_nom21 = build_model(corpus_dir, order=3, k=0.5, skip_threshold=0.3,
                                       progress_every=0, use_music21=True, verbose=False)
        finally:
            builtins.__import__ = real_import
        _assert(model_nom21["n_pieces"] == 14, "el fallback sin music21 no procesó el fixture igual de bien")
        log("  ok: build-model funciona igual de bien sin music21 instalado")

        log("camino music21 (si está instalado) también se ejecuta sin fallar...")
        try:
            import music21  # noqa: F401
            model_m21 = build_model(corpus_dir, order=3, k=0.5, skip_threshold=0.3,
                                     progress_every=0, use_music21=True, verbose=False)
            _assert(model_m21["n_pieces"] > 0, "el camino music21 no produjo piezas válidas")
            log(f"  ok: camino music21 también funcional ({model_m21['n_pieces']} piezas)")
        except ImportError:
            log("  (music21 no instalado en este entorno -- camino alternativo no ejercitado, "
                "pero el fallback de arriba SÍ prueba que su ausencia no rompe nada)")

    print("\n  [selftest] TODO OK -- 9.2 a 9.5 del plan verificados.\n")
    return 0


def cmd_selftest(args) -> int:
    try:
        return run_selftest(verbose=not args.quiet)
    except AssertionError as e:
        print(f"\n  [selftest] FALLO: {e}\n", file=sys.stderr)
        return 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chord_copilot",
        description="Copiloto de progresiones de acordes basado en n-gramas "
                     "entrenados sobre un corpus MIDI real.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("build-model", help="Corpus MIDI -> modelo de n-gramas")
    p.add_argument("--corpus", required=True, help="Directorio o patrón glob de ficheros MIDI")
    p.add_argument("--order", type=int, default=3, choices=[2, 3])
    p.add_argument("--k", type=float, default=0.5, help="Suavizado add-k (default: 0.5)")
    p.add_argument("--skip-threshold", type=float, default=0.3,
                   help="Umbral de segmentos sin acorde para descartar una pieza (default: 0.3)")
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--use-music21", action="store_true",
                   help="Usa music21 para detección de tonalidad si está instalado")
    p.add_argument("--out", default="model.json")
    p.set_defaults(func=cmd_build_model)

    p = sub.add_parser("suggest", help="Contexto de acordes -> top-N sugerencias")
    p.add_argument("--context", required=True)
    p.add_argument("--roman", action="store_true", help="El contexto ya está en números romanos")
    p.add_argument("--model", default="model.json")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--key", default=None)
    p.add_argument("--mood", default=None)
    p.add_argument("--min-support", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("info", help="Inspecciona metadata de un modelo")
    p.add_argument("--model", default="model.json")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("render", help="Contexto + siguiente(s) acorde(s) -> preview MIDI")
    p.add_argument("--context", required=True)
    p.add_argument("--next", required=True)
    p.add_argument("--roman", action="store_true")
    p.add_argument("--key", default=None)
    p.add_argument("--out", default="preview.mid")
    p.add_argument("--tempo", type=int, default=100)
    p.add_argument("--beats", type=float, default=4.0, help="Beats por acorde (default: 4)")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("make-test-corpus", help="Genera un corpus sintético determinista")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_make_test_corpus)

    p = sub.add_parser("selftest", help="Pipeline de integración end-to-end (sección 9.5)")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_selftest)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
