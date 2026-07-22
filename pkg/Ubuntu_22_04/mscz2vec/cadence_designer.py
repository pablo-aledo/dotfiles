#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      CADENCE DESIGNER  v1.0                                  ║
║         Diseñador de arquitectura de cadencias para obras MIDI              ║
║                                                                              ║
║  Dado un esquema formal, una progresión armónica, o un MIDI incompleto,     ║
║  diseña la ubicación, tipo y fuerza de cada cadencia a lo largo de la obra. ║
║  Asegura coherencia narrativa: las cadencias se organizan en una jerarquía  ║
║  de cierres donde la última es la más fuerte, con variedad y tensión       ║
║  progresiva.                                                                ║
║                                                                              ║
║  A diferencia de expectation_engine (que manipula cadencias EXISTENTES),   ║
║  esta herramienta las DISEÑA desde cero o las PLANIFICA sobre un esqueleto.║
║  A diferencia de chord_progression_generator (que genera acordes),         ║
║  esta se enfoca exclusivamente en los PUNTOS DE CIERRE y sus tipos.         ║
║                                                                              ║
║  CATÁLOGO DE CADENCIAS:                                                      ║
║    authentic_perfect    — V→I, bajo en fundamental. Cierre más fuerte       ║
║    authentic_imperfect  — V→I, bajo en tercera. Cierre fuerte               ║
║    authentic_deceptive  — V→vi. Niega la resolución, sorpresa               ║
║    authentic_interrupted— V→vi, bajo sube. Decepción dramática              ║
║    plagal               — IV→I. Efecto "Amén", suave                         ║
║    plagal_minor         — iv→I en mayor. Color modal                       ║
║    half                 — I→V o cualquier acorde→V. Suspendida              ║
║    phrygian_half        — iv6→V. Descenso cromático en bajo                 ║
║    evaded               — V→I6. Bajo no llega a tónica                      ║
║    imperfect_with_64    — I64→V→I. 6/4 ornamental prepara dominante       ║
║    deceptive_extended   — V7→vi→ii6→V→I. Engaño que se resuelve           ║
║    picardy              — V→I en menor, pero I es mayor. Luz al final       ║
║                                                                              ║
║  SUBCOMANDOS                                                                 ║
║    plan        — Diseña el mapa de cadencias para una forma dada            ║
║    analyze     — Detecta cadencias ya presentes en un MIDI                  ║
║    realize     — Convierte plan en bajo cifrado JSON (voice_leader)        ║
║    insert      — Inserta acordes de cadencia en un MIDI existente           ║
║    strengthen  — Escalona progresivamente la fuerza hacia el final          ║
║    weaken      — Introduce inestabilidad: más engaños y semicadencias       ║
║    sequence    — Organiza cadencias en progresión armónica ascendente/desc.   ║
║                                                                              ║
║  EJEMPLO DE USO COMPLETO (flujo recomendado):                                ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║                                                                              ║
║  # 1. Diseñar arquitectura de cierres para una sonata en La menor          ║
║    python cadence_designer.py plan --bars 64 --key Am --form sonata         ║
║        --tension late_climax -o plan.json --verbose                         ║
║                                                                              ║
║  # 2. Reforzar cadencias intermedias hacia el clímax                        ║
║    python cadence_designer.py strengthen plan.json --intensity 0.6          ║
║        -o plan_ref.json                                                     ║
║                                                                              ║
║  # 3. Convertir a bajo cifrado para voice_leader                            ║
║    python cadence_designer.py realize plan_ref.json -o bajo_cifrado.json    ║
║                                                                              ║
║  # 4. (Opcional) Insertar como pista guía en un MIDI existente              ║
║    python cadence_designer.py insert esqueleto.mid --plan plan_ref.json     ║
║        -o obra_con_cadencias.mid                                            ║
║                                                                              ║
║  INTEGRACIÓN CON EL ECOSISTEMA:                                             ║
║    theorist.py         → --from-theorist lee .theorist.json                 ║
║    tension_designer    → --from-curves lee .curves.json                     ║
║    voice_leader.py     → realize genera bajo cifrado compatible             ║
║    bass_line_composer  → realize genera progresión para bajo                  ║
║    expectation_engine  → insert prepara cadencias para manipulación         ║
║    chord_progression_generator → plan coordina con progresión generada      ║
║    arc_supervisor      → plan exporta tensión_release por cadencia          ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                  ║
║  OPCIONAL: music21 (análisis tonal avanzado)                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, NamedTuple
from dataclasses import dataclass, asdict

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy es requerido. pip install numpy")
    sys.exit(1)

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: mido es requerido. pip install mido")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES Y CATÁLOGO DE CADENCIAS
# ──────────────────────────────────────────────────────────────────────────────

CADENCE_TYPES = {
    # Cadencias auténticas
    "authentic_perfect": {
        "name": "Cadencia Auténtica Perfecta",
        "progression": ["V", "I"],
        "bass": ["5", "1"],
        "strength": 1.0,
        "tension_release": 0.9,
        "description": "V→I con bajo en fundamental. Cierre más fuerte."
    },
    "authentic_imperfect": {
        "name": "Cadencia Auténtica Imperfecta",
        "progression": ["V", "I"],
        "bass": ["5", "3"],
        "strength": 0.75,
        "tension_release": 0.7,
        "description": "V→I con bajo en tercera. Cierre fuerte pero menos conclusivo."
    },
    "authentic_deceptive": {
        "name": "Cadencia de Engaño",
        "progression": ["V", "vi"],
        "bass": ["5", "6"],
        "strength": 0.3,
        "tension_release": 0.1,
        "description": "V→vi. Niega la resolución esperada, crea sorpresa."
    },
    "authentic_interrupted": {
        "name": "Cadencia Interrumpida",
        "progression": ["V", "vi"],
        "bass": ["5", "6"],
        "strength": 0.25,
        "tension_release": 0.05,
        "description": "V→vi con bajo que sube. Decepción dramática."
    },
    # Cadencias plagales
    "plagal": {
        "name": "Cadencia Plagal",
        "progression": ["IV", "I"],
        "bass": ["4", "1"],
        "strength": 0.6,
        "tension_release": 0.65,
        "description": "IV→I. Efecto 'Amén', suave y conclusivo."
    },
    "plagal_minor": {
        "name": "Cadencia Plagal Menor",
        "progression": ["iv", "I"],
        "bass": ["4", "1"],
        "strength": 0.55,
        "tension_release": 0.6,
        "description": "iv→I en mayor. Color modal, melancolía."
    },
    # Semicadencias
    "half": {
        "name": "Semicadencia",
        "progression": ["I", "V"],
        "bass": ["1", "5"],
        "strength": 0.35,
        "tension_release": 0.2,
        "description": "Termina en V. Deja la frase suspendida, sin cierre."
    },
    "phrygian_half": {
        "name": "Semicadencia Frigia",
        "progression": ["iv6", "V"],
        "bass": ["6", "5"],
        "strength": 0.4,
        "tension_release": 0.25,
        "description": "iv6→V con bajo que desciende semitono. Tensión cromática."
    },
    # Cadencias especiales
    "deceptive_extended": {
        "name": "Cadencia de Engaño Extendida",
        "progression": ["V7", "vi", "ii6", "V", "I"],
        "bass": ["5", "6", "2", "5", "1"],
        "strength": 0.85,
        "tension_release": 0.8,
        "description": "V7→vi→ii6→V→I. El engaño se resuelve tras extensión."
    },
    "evaded": {
        "name": "Cadencia Evadida",
        "progression": ["V", "I6"],
        "bass": ["5", "3"],
        "strength": 0.45,
        "tension_release": 0.3,
        "description": "V→I6. El bajo no llega a tónica, el cierre se debilita."
    },
    "imperfect_with_64": {
        "name": "Cadencia con 6/4 de Cadencia",
        "progression": ["I64", "V", "I"],
        "bass": ["1", "5", "1"],
        "strength": 0.9,
        "tension_release": 0.85,
        "description": "I6/4→V→I. El 6/4 ornamental prepara la dominante."
    },
    "picardy": {
        "name": "Tercera Picarda",
        "progression": ["V", "I"],
        "bass": ["5", "1"],
        "strength": 0.8,
        "tension_release": 0.75,
        "description": "V→I en modo menor, pero I es mayor. Luz al final."
    },
}

# Funciones armónicas para cada grado
FUNCTION_MAP = {
    "I": "T", "i": "T",
    "II": "S", "ii": "S", "II7": "S", "ii7": "S",
    "III": "T", "iii": "T", "III7": "T", "iii7": "T",
    "IV": "S", "iv": "S", "IV7": "S", "iv7": "S",
    "V": "D", "v": "D", "V7": "D", "v7": "D", "V+": "D",
    "VI": "T", "vi": "T", "VI7": "T", "vi7": "T",
    "VII": "D", "vii": "D", "vii°": "D", "vii°7": "D", "VII7": "D",
}

ROMAN_TO_PITCH = {
    "I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11,
    "i": 0, "ii": 2, "iii": 4, "iv": 5, "v": 7, "vi": 9, "vii": 11,
}

# ──────────────────────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Cadence:
    position: int           # compás donde ocurre
    cad_type: str           # clave en CADENCE_TYPES
    key: str                # tonalidad local
    strength: float         # 0-1
    tension_before: float   # 0-1
    tension_after: float    # 0-1
    description: str
    progression: List[str]  # acordes de la cadencia
    bass_degrees: List[str] # grados del bajo
    is_final: bool = False
    section: str = ""       # sección formal (A, B, Coda...)

    def to_dict(self):
        d = asdict(self)
        d["type_name"] = CADENCE_TYPES.get(self.cad_type, {}).get("name", self.cad_type)
        return d


@dataclass  
class CadencePlan:
    total_bars: int
    key: str
    mode: str
    form: str
    cadences: List[Cadence]
    tension_curve: List[float]

    def to_dict(self):
        return {
            "total_bars": self.total_bars,
            "key": self.key,
            "mode": self.mode,
            "form": self.form,
            "cadences": [c.to_dict() for c in self.cadences],
            "tension_curve": self.tension_curve,
        }


# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE MIDI
# ──────────────────────────────────────────────────────────────────────────────

def extract_chords_from_midi(midi_path: str, window_beats: float = 1.0) -> List[Tuple[int, str, List[int]]]:
    """
    Extrae acordes de un MIDI por ventanas temporales.
    Retorna: [(compás, acorde_detectado, notas_midi), ...]
    """
    try:
        mid = MidiFile(midi_path)
    except Exception as e:
        print(f"ERROR: No se pudo leer {midi_path}: {e}")
        return []

    # Obtener tempo y ticks por beat
    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # default 120 BPM

    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        if tempo != 500000:
            break

    # Extraer notas con tiempos absolutos
    notes = []
    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append((abs_time, msg.note, msg.velocity))

    if not notes:
        return []

    # Agrupar por ventanas de compás
    # Asumimos 4/4 por defecto
    beats_per_bar = 4
    ticks_per_bar = ticks_per_beat * beats_per_bar

    max_tick = max(n[0] for n in notes)
    num_bars = int(max_tick / ticks_per_bar) + 1

    chords = []
    for bar in range(1, num_bars + 1):
        start = (bar - 1) * ticks_per_bar
        end = bar * ticks_per_bar
        bar_notes = [n[1] for n in notes if start <= n[0] < end]

        if not bar_notes:
            chords.append((bar, "", []))
            continue

        # Detectar acorde por pitch classes
        pitch_classes = sorted(set(n % 12 for n in bar_notes))
        chord_name = detect_chord_from_pcs(pitch_classes)
        chords.append((bar, chord_name, bar_notes))

    return chords


def detect_chord_from_pcs(pitch_classes: List[int]) -> str:
    """Detección simple de acorde por pitch classes."""
    if not pitch_classes:
        return ""

    # Plantillas de acordes (pitch classes relativos a la fundamental)
    templates = {
        "major": [0, 4, 7],
        "minor": [0, 3, 7],
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "maj7": [0, 4, 7, 11],
        "min7": [0, 3, 7, 10],
        "dom7": [0, 4, 7, 10],
        "dim7": [0, 3, 6, 9],
        "halfdim7": [0, 3, 6, 10],
        "sus4": [0, 5, 7],
        "sus2": [0, 2, 7],
    }

    best_match = ""
    best_score = -1

    for root in range(12):
        for name, template in templates.items():
            shifted = [(t + root) % 12 for t in template]
            matches = len(set(shifted) & set(pitch_classes))
            score = matches / len(shifted)
            if score > best_score and score >= 0.6:
                best_score = score
                root_name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][root]
                if name == "major":
                    best_match = root_name
                elif name == "minor":
                    best_match = root_name + "m"
                elif name == "dim":
                    best_match = root_name + "dim"
                elif name == "aug":
                    best_match = root_name + "aug"
                elif name == "maj7":
                    best_match = root_name + "maj7"
                elif name == "min7":
                    best_match = root_name + "m7"
                elif name == "dom7":
                    best_match = root_name + "7"
                elif name == "dim7":
                    best_match = root_name + "dim7"
                elif name == "halfdim7":
                    best_match = root_name + "ø"
                elif name == "sus4":
                    best_match = root_name + "sus4"
                elif name == "sus2":
                    best_match = root_name + "sus2"

    return best_match


def estimate_key_from_midi(midi_path: str) -> Tuple[str, str]:
    """Estima tonalidad y modo de un MIDI."""
    chords = extract_chords_from_midi(midi_path)
    if not chords:
        return ("C", "major")

    # Contar pitch classes totales
    all_pcs = []
    for _, _, notes in chords:
        all_pcs.extend(n % 12 for n in notes)

    if not all_pcs:
        return ("C", "major")

    # Perfil de Krumhansl-Schmuckler simplificado
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    pc_counts = [0] * 12
    for pc in all_pcs:
        pc_counts[pc] += 1

    total = sum(pc_counts)
    if total == 0:
        return ("C", "major")

    pc_dist = [c / total for c in pc_counts]

    best_key = "C"
    best_mode = "major"
    best_corr = -2

    for root in range(12):
        # Correlación con mayor
        shifted_major = [major_profile[(i - root) % 12] for i in range(12)]
        corr_major = np.corrcoef(pc_dist, shifted_major)[0, 1]

        # Correlación con menor
        shifted_minor = [minor_profile[(i - root) % 12] for i in range(12)]
        corr_minor = np.corrcoef(pc_dist, shifted_minor)[0, 1]

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][root]
            best_mode = "major"

        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][root]
            best_mode = "minor"

    return (best_key, best_mode)


# ──────────────────────────────────────────────────────────────────────────────
# PLANIFICACIÓN DE CADENCIAS
# ──────────────────────────────────────────────────────────────────────────────

def plan_cadences(
    total_bars: int,
    key: str,
    mode: str,
    form: str,
    tension_profile: str = "arch",
    custom_tension: Optional[List[float]] = None,
    n_cadences: Optional[int] = None,
    seed: int = 42,
) -> CadencePlan:
    """
    Diseña un plan de cadencias para una obra.

    Args:
        total_bars: compases totales
        key: tónica (C, Am, etc.)
        mode: major/minor
        form: ABA, sonata, verse_chorus, etc.
        tension_profile: arch|crescendo|decrescendo|wave|late_climax|neutral
        custom_tension: curva de tensión explícita [0-1] por compás
        n_cadences: cuántas cadencias (auto si None)
    """
    random.seed(seed)
    np.random.seed(seed)

    # Generar curva de tensión
    if custom_tension:
        tension_curve = custom_tension
    else:
        tension_curve = generate_tension_curve(total_bars, tension_profile)

    # Determinar número y ubicación de cadencias según forma
    cadence_positions = get_cadence_positions(total_bars, form, n_cadences)

    # Asignar tipos de cadencia según posición y tensión
    cadences = []
    for i, pos in enumerate(cadence_positions):
        is_final = (i == len(cadence_positions) - 1)
        cad = select_cadence_for_position(
            pos, total_bars, tension_curve[pos-1], is_final, mode, i, len(cadence_positions)
        )
        cad.position = pos
        cad.key = key
        cad.section = get_section_for_position(pos, total_bars, form)
        cadences.append(cad)

    return CadencePlan(
        total_bars=total_bars,
        key=key,
        mode=mode,
        form=form,
        cadences=cadences,
        tension_curve=tension_curve,
    )


def generate_tension_curve(bars: int, profile: str) -> List[float]:
    """Genera curva de tensión [0-1] por compás."""
    x = np.linspace(0, 1, bars)

    if profile == "arch":
        # Subida hasta 0.618, bajada
        curve = np.where(x < 0.618, x / 0.618, (1 - x) / (1 - 0.618))
    elif profile == "crescendo":
        curve = x
    elif profile == "decrescendo":
        curve = 1 - x
    elif profile == "wave":
        curve = 0.5 + 0.5 * np.sin(x * 2 * np.pi * 2)
    elif profile == "late_climax":
        curve = np.where(x < 0.75, x * 0.5 / 0.75, 0.5 + (x - 0.75) * 0.5 / 0.25)
    elif profile == "neutral":
        curve = np.full(bars, 0.5)
    else:
        curve = x

    # Añadir variabilidad
    noise = np.random.normal(0, 0.05, bars)
    curve = np.clip(curve + noise, 0, 1)

    return curve.tolist()


def get_cadence_positions(total_bars: int, form: str, n_cadences: Optional[int]) -> List[int]:
    """Determina dónde colocar las cadencias según la forma."""
    if n_cadences:
        # Distribución espaciada
        step = total_bars / (n_cadences + 1)
        return [int(round(step * (i + 1))) for i in range(n_cadences)]

    form_positions = {
        "ABA": [8, 16, 24],  # Media frase A, final A, final B, final A
        "ABAB": [8, 16, 24, 32],
        "AABA": [8, 16, 24, 32],  # Jazz standard 32 compases
        "ABACABA": [8, 16, 24, 32, 40, 48, 56],
        "verse_chorus": [8, 16, 24, 32, 40, 48],  # V-C-V-C-B-C
        "sonata": [20, 40, 60, 80, 100],  # E-T-D-R-C (aprox)
        "binary": [8, 16],
        "ternary": [8, 16, 24],
        "rondo": [8, 16, 24, 32, 40],
        "through_composed": [],  # Se determina por tensión
    }

    if form in form_positions:
        positions = [p for p in form_positions[form] if p <= total_bars]
        # Asegurar cadencia final
        if not positions or positions[-1] != total_bars:
            positions.append(total_bars)
        return positions

    # Auto: una cadencia cada 8 compases aprox, más una final
    positions = list(range(8, total_bars + 1, 8))
    if not positions or positions[-1] != total_bars:
        positions.append(total_bars)
    return positions


def get_section_for_position(pos: int, total: int, form: str) -> str:
    """Determina qué sección formal contiene esta posición."""
    if form == "ABA":
        third = total // 3
        if pos <= third:
            return "A"
        elif pos <= 2 * third:
            return "B"
        else:
            return "A"
    elif form == "sonata":
        # Exposición, Transición, Desarrollo, Recapitulación, Coda
        fifth = total // 5
        if pos <= fifth:
            return "Exposición"
        elif pos <= 2 * fifth:
            return "Transición"
        elif pos <= 3 * fifth:
            return "Desarrollo"
        elif pos <= 4 * fifth:
            return "Recapitulación"
        else:
            return "Coda"
    elif form == "verse_chorus":
        # V-C-V-C-B-C
        sections = ["V", "C", "V", "C", "B", "C"]
        section_size = total // len(sections)
        idx = min((pos - 1) // section_size, len(sections) - 1)
        return sections[idx]
    return ""


def select_cadence_for_position(
    pos: int, total: int, tension: float, is_final: bool, mode: str,
    cadence_index: int, total_cadences: int
) -> Cadence:
    """Selecciona el tipo de cadencia óptimo para una posición."""

    if is_final:
        # La última cadencia SIEMPRE debe ser la más fuerte
        if mode == "minor":
            # Tercera picarda como opción dramática
            return Cadence(
                position=pos,
                cad_type="picardy" if random.random() < 0.3 else "authentic_perfect",
                key="",
                strength=1.0,
                tension_before=tension,
                tension_after=0.0,
                description="Cierre final definitivo",
                progression=CADENCE_TYPES["authentic_perfect"]["progression"],
                bass_degrees=CADENCE_TYPES["authentic_perfect"]["bass"],
                is_final=True,
            )
        else:
            return Cadence(
                position=pos,
                cad_type="authentic_perfect",
                key="",
                strength=1.0,
                tension_before=tension,
                tension_after=0.0,
                description="Cierre final definitivo",
                progression=CADENCE_TYPES["authentic_perfect"]["progression"],
                bass_degrees=CADENCE_TYPES["authentic_perfect"]["bass"],
                is_final=True,
            )

    # Cadencias intermedias: estrategia de escalonamiento
    # La fuerza debe ir aumentando hacia el final
    progress = cadence_index / max(total_cadences - 1, 1)

    # En puntos de alta tensión: usar cadencias que NO resuelvan (engaño, semicadencia)
    # En puntos de baja tensión: usar cadencias que resuelvan parcialmente
    if tension > 0.7:
        # Alta tensión: semicadencia o engaño
        candidates = ["half", "phrygian_half", "authentic_deceptive"]
    elif tension > 0.4:
        # Tensión media: cadencias imperfectas o evadidas
        candidates = ["authentic_imperfect", "evaded", "plagal"]
    else:
        # Baja tensión: cadencias conclusivas pero no finales
        candidates = ["authentic_imperfect", "plagal", "imperfect_with_64"]

    # Ajustar por progreso: más fuertes al acercarse al final
    if progress > 0.7:
        candidates = [c for c in candidates if CADENCE_TYPES[c]["strength"] >= 0.5]
        if not candidates:
            candidates = ["authentic_imperfect"]

    chosen = random.choice(candidates)
    info = CADENCE_TYPES[chosen]

    return Cadence(
        position=pos,
        cad_type=chosen,
        key="",
        strength=info["strength"],
        tension_before=tension,
        tension_after=1.0 - info["tension_release"],
        description=info["description"],
        progression=info["progression"],
        bass_degrees=info["bass"],
        is_final=False,
    )


# ──────────────────────────────────────────────────────────────────────────────
# MANIPULACIÓN DE CADENCIAS
# ──────────────────────────────────────────────────────────────────────────────

def strengthen_cadences(plan: CadencePlan, intensity: float = 0.5) -> CadencePlan:
    """Refuerza progresivamente las cadencias hacia el final."""
    if not plan.cadences:
        return plan

    new_cadences = []
    for i, cad in enumerate(plan.cadences):
        if cad.is_final:
            new_cadences.append(cad)
            continue

        progress = i / max(len(plan.cadences) - 1, 1)

        # Si la cadencia es demasiado débil para su posición, subirla de nivel
        if progress > 0.5 and cad.strength < 0.6 * (0.5 + progress * 0.5):
            # Subir a una cadencia más fuerte
            if cad.cad_type in ["half", "phrygian_half"]:
                cad.cad_type = "authentic_imperfect"
                cad.strength = 0.75
                cad.description = "Semicadencia reforzada a imperfecta"
            elif cad.cad_type in ["authentic_deceptive", "authentic_interrupted"]:
                cad.cad_type = "evaded"
                cad.strength = 0.45
                cad.description = "Engaño suavizado a evadida"

        new_cadences.append(cad)

    plan.cadences = new_cadences
    return plan


def weaken_cadences(plan: CadencePlan, intensity: float = 0.5) -> CadencePlan:
    """Introduce inestabilidad: cadencias más débiles, más engaños."""
    if not plan.cadences:
        return plan

    new_cadences = []
    for i, cad in enumerate(plan.cadences):
        if cad.is_final:
            # La final puede volverse picarda o imperfecta
            if random.random() < intensity * 0.3:
                cad.cad_type = "authentic_imperfect"
                cad.strength = 0.75
                cad.description = "Cierre final suavizado a imperfecta"
            new_cadences.append(cad)
            continue

        # Probabilidad de debilitar
        if random.random() < intensity:
            if cad.cad_type in ["authentic_perfect", "authentic_imperfect"]:
                cad.cad_type = "authentic_deceptive"
                cad.strength = 0.3
                cad.description = "Cadencia debilitada a engaño"
            elif cad.cad_type in ["plagal", "plagal_minor"]:
                cad.cad_type = "half"
                cad.strength = 0.35
                cad.description = "Plagal debilitada a semicadencia"

        new_cadences.append(cad)

    plan.cadences = new_cadences
    return plan


def sequence_cadences(plan: CadencePlan, direction: str = "ascending") -> CadencePlan:
    """Crea una secuencia de cadencias en progresión armónica."""
    # Para implementación completa, requeriría análisis de la progresión existente
    # Aquí marcamos las cadencias como parte de una secuencia
    for cad in plan.cadences:
        cad.description += f" [Secuencia {direction}]"
    return plan


# ──────────────────────────────────────────────────────────────────────────────
# REALIZACIÓN: De plan a bajo cifrado
# ──────────────────────────────────────────────────────────────────────────────

def realize_cadence_plan(plan: CadencePlan, output_path: str):
    """
    Convierte el plan de cadencias en un archivo de bajo cifrado
    compatible con voice_leader.py y bass_line_composer.py.
    """
    realization = {
        "key": plan.key,
        "mode": plan.mode,
        "total_bars": plan.total_bars,
        "form": plan.form,
        "cadences": [],
        "bass_line": [],
    }

    # Construir línea de bajo cifrado compás a compás
    bass_line = []
    current_bar = 1

    for cad in sorted(plan.cadences, key=lambda c: c.position):
        # Rellenar hasta la cadencia con acordes de prolongación
        while current_bar < cad.position:
            bass_line.append({
                "bar": current_bar,
                "chord": "I",
                "bass": "1",
                "function": "T",
                "is_cadence": False,
            })
            current_bar += 1

        # Insertar la cadencia
        for j, (chord, bass_deg) in enumerate(zip(cad.progression, cad.bass_degrees)):
            bass_line.append({
                "bar": current_bar,
                "chord": chord,
                "bass": bass_deg,
                "function": FUNCTION_MAP.get(chord.replace("6", "").replace("4", "").replace("7", ""), "?"),
                "is_cadence": True,
                "cadence_type": cad.cad_type,
                "cadence_strength": cad.strength,
                "cadence_position_in_bar": j,
            })
            if j < len(cad.progression) - 1:
                current_bar += 1

        realization["cadences"].append(cad.to_dict())

    # Rellenar hasta el final
    while current_bar <= plan.total_bars:
        bass_line.append({
            "bar": current_bar,
            "chord": "I",
            "bass": "1",
            "function": "T",
            "is_cadence": False,
        })
        current_bar += 1

    realization["bass_line"] = bass_line

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(realization, f, indent=2, ensure_ascii=False)

    return realization


# ──────────────────────────────────────────────────────────────────────────────
# INSERCIÓN EN MIDI
# ──────────────────────────────────────────────────────────────────────────────

def insert_cadences_into_midi(
    midi_path: str,
    plan: CadencePlan,
    output_path: str,
    velocity: int = 80,
):
    """Inserta acordes de cadencia en un MIDI existente (o crea uno nuevo)."""
    try:
        mid = MidiFile(midi_path)
    except:
        mid = MidiFile()
        mid.ticks_per_beat = 480
        track = MidiTrack()
        track.append(MetaMessage("time_signature", numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        track.append(MetaMessage("set_tempo", tempo=500000, time=0))
        mid.tracks.append(track)

    ticks_per_beat = mid.ticks_per_beat
    ticks_per_bar = ticks_per_beat * 4

    # Crear track de cadencias
    cad_track = MidiTrack()
    cad_track.append(MetaMessage("track_name", name="Cadences", time=0))

    for cad in plan.cadences:
        bar_start = (cad.position - 1) * ticks_per_bar

        # Insertar acordes de la cadencia
        duration_per_chord = ticks_per_bar // max(len(cad.progression), 1)

        for i, chord_name in enumerate(cad.progression):
            # Convertir nombre de acorde a notas MIDI
            notes = chord_name_to_midi(chord_name, plan.key, plan.mode)

            offset = bar_start + i * duration_per_chord

            # Notas on
            for j, note in enumerate(notes):
                cad_track.append(Message("note_on", note=note, velocity=velocity, time=offset if j == 0 else 0))

            # Notas off
            for j, note in enumerate(notes):
                cad_track.append(Message("note_off", note=note, velocity=0, time=duration_per_chord if j == 0 else 0))

    mid.tracks.append(cad_track)
    mid.save(output_path)
    return output_path


def chord_name_to_midi(chord_name: str, key: str, mode: str) -> List[int]:
    """Convierte nombre de acorde romano a notas MIDI."""
    # Mapeo de notas a pitch class
    note_to_pc = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
    }

    key_pc = note_to_pc.get(key.replace("m", "").replace("M", ""), 0)

    # Parsear grado romano
    roman = chord_name.replace("6", "").replace("4", "").replace("7", "").replace("°", "").replace("ø", "").replace("+", "")

    degree_pc = ROMAN_TO_PITCH.get(roman, 0)
    root = (key_pc + degree_pc) % 12

    # Determinar calidad
    is_minor = "i" in roman.lower() and "I" not in roman
    is_dim = "°" in chord_name or "dim" in chord_name.lower()
    is_aug = "+" in chord_name
    is_dom7 = "7" in chord_name and not ("maj7" in chord_name or "m7" in chord_name or "dim7" in chord_name)
    is_maj7 = "maj7" in chord_name
    is_min7 = "m7" in chord_name
    is_dim7 = "dim7" in chord_name
    is_halfdim = "ø" in chord_name

    # Construir acorde
    if is_dim7:
        intervals = [0, 3, 6, 9]
    elif is_halfdim:
        intervals = [0, 3, 6, 10]
    elif is_dim:
        intervals = [0, 3, 6]
    elif is_aug:
        intervals = [0, 4, 8]
    elif is_dom7:
        intervals = [0, 4, 7, 10]
    elif is_maj7:
        intervals = [0, 4, 7, 11]
    elif is_min7:
        intervals = [0, 3, 7, 10]
    elif is_minor or (mode == "minor" and roman in ["i", "ii", "iv"]):
        intervals = [0, 3, 7]
    else:
        intervals = [0, 4, 7]

    # Añadir séptima si es dom7 implícito en V
    if roman in ["V", "v"] and "7" in chord_name and not any([is_maj7, is_min7, is_dim7, is_halfdim]):
        intervals = [0, 4, 7, 10]

    notes = [root + interval + 60 for interval in intervals]  # Octava 4
    return notes


# ──────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE CADENCIAS EXISTENTES
# ──────────────────────────────────────────────────────────────────────────────

def analyze_existing_cadences(midi_path: str) -> Dict:
    """Analiza un MIDI y detecta cadencias ya presentes."""
    chords = extract_chords_from_midi(midi_path)
    if not chords:
        return {"error": "No se pudieron extraer acordes"}

    detected = []
    for i in range(len(chords) - 1):
        bar1, chord1, _ = chords[i]
        bar2, chord2, _ = chords[i + 1]

        # Detectar patrones de cadencia
        cad_type = detect_cadence_pattern(chord1, chord2)
        if cad_type:
            info = CADENCE_TYPES.get(cad_type, {})
            detected.append({
                "position": bar1,
                "cadence_type": cad_type,
                "name": info.get("name", "Desconocida"),
                "progression": [chord1, chord2],
                "strength": info.get("strength", 0),
            })

    return {
        "file": midi_path,
        "total_bars": len(chords),
        "detected_cadences": detected,
        "summary": {
            "count": len(detected),
            "average_strength": sum(d["strength"] for d in detected) / max(len(detected), 1),
            "final_cadence": detected[-1]["cadence_type"] if detected else None,
        }
    }


def detect_cadence_pattern(chord1: str, chord2: str) -> Optional[str]:
    """Detecta si dos acordes consecutivos forman una cadencia conocida."""
    # Simplificación: detectar por funciones armónicas básicas
    # En una implementación completa, usaría análisis más sofisticado

    # Normalizar nombres
    def normalize(c):
        c = c.replace("maj", "").replace("min", "m").replace("7", "").replace("m", "").replace("dim", "").replace("aug", "")
        return c

    c1 = normalize(chord1)
    c2 = normalize(chord2)

    # V→I = auténtica
    if c1 in ["V", "G"] and c2 in ["I", "C"]:
        return "authentic_perfect"
    # V→vi = engaño
    if c1 in ["V", "G"] and c2 in ["vi", "Am"]:
        return "authentic_deceptive"
    # IV→I = plagal
    if c1 in ["IV", "F"] and c2 in ["I", "C"]:
        return "plagal"
    # I→V = semicadencia
    if c1 in ["I", "C"] and c2 in ["V", "G"]:
        return "half"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# LECTURA DE ARCHIVOS EXTERNOS
# ──────────────────────────────────────────────────────────────────────────────

def load_theorist_plan(path: str) -> Optional[Dict]:
    """Carga un plan de theorist.py."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def load_tension_curves(path: str) -> Optional[List[float]]:
    """Carga curvas de tension_designer.py."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict) and "tension" in data:
                return data["tension"]
            return data
    except:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# SALIDA Y REPORTE
# ──────────────────────────────────────────────────────────────────────────────

def print_report(plan: CadencePlan, verbose: bool = False):
    """Imprime informe en consola al estilo del ecosistema."""
    R = "\033[0m"
    B = "\033[1m"
    G = "\033[92m"
    Y = "\033[93m"
    C = "\033[96m"

    width = 76
    print(f"{B}╔{'═' * width}╗{R}")
    print(f"{B}║{'CADENCE DESIGNER v1.0 — INFORME':^{width}}║{R}")
    print(f"{B}╠{'═' * width}╣{R}")
    print(f"{B}║{R}  Forma      : {plan.form:<58}{B}║{R}")
    print(f"{B}║{R}  Tonalidad  : {plan.key} {plan.mode:<54}{B}║{R}")
    print(f"{B}║{R}  Compases   : {plan.total_bars:<58}{B}║{R}")
    print(f"{B}║{R}  Cadencias  : {len(plan.cadences):<58}{B}║{R}")
    print(f"{B}╠{'═' * width}╣{R}")

    print(f"{B}║{'POS':^5}│{'TIPO':^28}│{'FUERZA':^8}│{'TENSIÓN':^14}│{'SECCIÓN':^16}║{R}")
    print(f"{B}╠{'═' * width}╣{R}")

    for cad in plan.cadences:
        tipo = CADENCE_TYPES.get(cad.cad_type, {}).get("name", cad.cad_type)[:26]
        fuerza = f"{cad.strength:.2f}"
        tension = f"{cad.tension_before:.2f}→{cad.tension_after:.2f}"
        seccion = cad.section[:14]
        final = f" {Y}[FINAL]{R}" if cad.is_final else ""
        print(f"{B}║{R} {cad.position:>3} │ {tipo:<26} │ {fuerza:>6} │ {tension:>12} │ {seccion:<14}{final:<6}{B}║{R}")

    print(f"{B}╚{'═' * width}╝{R}")

    if verbose:
        print(f"\n{C}Curva de tensión (primeros 16 compases):{R}")
        for i, t in enumerate(plan.tension_curve[:16]):
            bar = "█" * int(t * 20)
            print(f"  Compás {i+1:>3}: {bar:<20} {t:.2f}")


def save_plan(plan: CadencePlan, path: str):
    """Guarda el plan en JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────────────
# CLI PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diseñador de arquitectura de cadencias para obras MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:
  python cadence_designer.py plan --bars 32 --key C --form ABA
  python cadence_designer.py plan --bars 64 --key Am --form sonata --tension late_climax
  python cadence_designer.py analyze obra.mid --json analisis.json
  python cadence_designer.py realize plan.json --output bajo_cifrado.json
  python cadence_designer.py insert obra.mid --plan plan.json --output obra_con_cadencias.mid
  python cadence_designer.py strengthen plan.json --intensity 0.7 --output plan_reforzado.json
  python cadence_designer.py plan --from-theorist obra.theorist.json --output plan.json
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Comando a ejecutar")

    # ── PLAN ──
    cmd_plan = subparsers.add_parser("plan", help="Diseña un plan de cadencias")
    cmd_plan.add_argument("--bars", type=int, required=True, help="Compases totales")
    cmd_plan.add_argument("--key", type=str, default="C", help="Tónica (C, Am, F#...)")
    cmd_plan.add_argument("--mode", type=str, default="major", choices=["major", "minor"], help="Modo")
    cmd_plan.add_argument("--form", type=str, default="through_composed",
                         choices=["ABA", "ABAB", "AABA", "ABACABA", "verse_chorus", 
                                 "sonata", "binary", "ternary", "rondo", "through_composed"],
                         help="Forma musical")
    cmd_plan.add_argument("--tension", type=str, default="arch",
                         choices=["arch", "crescendo", "decrescendo", "wave", "late_climax", "neutral"],
                         help="Perfil de tensión")
    cmd_plan.add_argument("--tension-curve", type=str, help="Curva explícita: 0.2,0.5,0.8,...")
    cmd_plan.add_argument("--n-cadences", type=int, help="Número de cadencias (auto por defecto)")
    cmd_plan.add_argument("--from-theorist", type=str, help="Leer plan de theorist.py")
    cmd_plan.add_argument("--from-curves", type=str, help="Leer curvas de tension_designer")
    cmd_plan.add_argument("--output", "-o", type=str, help="Guardar plan en JSON")
    cmd_plan.add_argument("--realize", type=str, help="También realizar como bajo cifrado")
    cmd_plan.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    cmd_plan.add_argument("--verbose", action="store_true", help="Informe detallado")

    # ── ANALYZE ──
    cmd_analyze = subparsers.add_parser("analyze", help="Analiza cadencias existentes en un MIDI")
    cmd_analyze.add_argument("midi", help="Archivo MIDI de entrada")
    cmd_analyze.add_argument("--json", type=str, help="Exportar análisis a JSON")
    cmd_analyze.add_argument("--verbose", action="store_true", help="Detalle completo")

    # ── REALIZE ──
    cmd_realize = subparsers.add_parser("realize", help="Convierte plan en bajo cifrado")
    cmd_realize.add_argument("plan", help="Plan de cadencias JSON")
    cmd_realize.add_argument("--output", "-o", type=str, required=True, help="Archivo de salida JSON")

    # ── INSERT ──
    cmd_insert = subparsers.add_parser("insert", help="Inserta cadencias en un MIDI")
    cmd_insert.add_argument("midi", help="Archivo MIDI de entrada")
    cmd_insert.add_argument("--plan", type=str, required=True, help="Plan de cadencias JSON")
    cmd_insert.add_argument("--output", "-o", type=str, required=True, help="MIDI de salida")
    cmd_insert.add_argument("--velocity", type=int, default=80, help="Velocity de los acordes")

    # ── STRENGTHEN ──
    cmd_strengthen = subparsers.add_parser("strengthen", help="Refuerza cadencias hacia el final")
    cmd_strengthen.add_argument("plan", help="Plan de cadencias JSON")
    cmd_strengthen.add_argument("--intensity", type=float, default=0.5, help="Intensidad del refuerzo 0-1")
    cmd_strengthen.add_argument("--output", "-o", type=str, required=True, help="Plan modificado")

    # ── WEAKEN ──
    cmd_weaken = subparsers.add_parser("weaken", help="Introduce inestabilidad en cadencias")
    cmd_weaken.add_argument("plan", help="Plan de cadencias JSON")
    cmd_weaken.add_argument("--intensity", type=float, default=0.5, help="Intensidad 0-1")
    cmd_weaken.add_argument("--output", "-o", type=str, required=True, help="Plan modificado")

    # ── SEQUENCE ──
    cmd_sequence = subparsers.add_parser("sequence", help="Secuencia de cadencias en progresión")
    cmd_sequence.add_argument("plan", help="Plan de cadencias JSON")
    cmd_sequence.add_argument("--direction", type=str, default="ascending", choices=["ascending", "descending"], help="Dirección")
    cmd_sequence.add_argument("--output", "-o", type=str, required=True, help="Plan modificado")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # ── EJECUCIÓN ──

    if args.command == "plan":
        # Cargar desde theorist si se indica
        if args.from_theorist:
            theorist = load_theorist_plan(args.from_theorist)
            if theorist:
                key = theorist.get("key", args.key)
                mode = theorist.get("mode", args.mode)
                form = theorist.get("form", args.form)
                bars = theorist.get("bars", args.bars)
            else:
                key, mode, form, bars = args.key, args.mode, args.form, args.bars
        else:
            key, mode, form, bars = args.key, args.mode, args.form, args.bars

        # Curva de tensión
        custom_tension = None
        if args.tension_curve:
            custom_tension = [float(x) for x in args.tension_curve.split(",")]
        elif args.from_curves:
            custom_tension = load_tension_curves(args.from_curves)

        plan = plan_cadences(
            total_bars=bars,
            key=key,
            mode=mode,
            form=form,
            tension_profile=args.tension,
            custom_tension=custom_tension,
            n_cadences=args.n_cadences,
            seed=args.seed,
        )

        print_report(plan, args.verbose)

        if args.output:
            save_plan(plan, args.output)
            print(f"\nPlan guardado en: {args.output}")

        if args.realize:
            realize_cadence_plan(plan, args.realize)
            print(f"Bajo cifrado guardado en: {args.realize}")

    elif args.command == "analyze":
        result = analyze_existing_cadences(args.midi)

        print(f"\033[1mANÁLISIS DE CADENCIAS: {args.midi}\033[0m")
        print(f"Compases totales: {result.get('total_bars', '?')}")
        print(f"Cadencias detectadas: {result['summary']['count']}")
        print(f"Fuerza media: {result['summary']['average_strength']:.2f}")

        if result['summary']['final_cadence']:
            print(f"Cadencia final: {CADENCE_TYPES.get(result['summary']['final_cadence'], {}).get('name', result['summary']['final_cadence'])}")

        for cad in result['detected_cadences']:
            print(f"  Compás {cad['position']:>3}: {cad['name']} (fuerza {cad['strength']:.2f})")

        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nAnálisis exportado a: {args.json}")

    elif args.command == "realize":
        with open(args.plan, "r", encoding="utf-8") as f:
            data = json.load(f)

        plan = CadencePlan(
            total_bars=data["total_bars"],
            key=data["key"],
            mode=data["mode"],
            form=data["form"],
            cadences=[Cadence(**{k: v for k, v in c.items() if k != "type_name"}) for c in data["cadences"]],
            tension_curve=data["tension_curve"],
        )

        realize_cadence_plan(plan, args.output)
        print(f"Bajo cifrado realizado en: {args.output}")

    elif args.command == "insert":
        with open(args.plan, "r", encoding="utf-8") as f:
            data = json.load(f)

        plan = CadencePlan(
            total_bars=data["total_bars"],
            key=data["key"],
            mode=data["mode"],
            form=data["form"],
            cadences=[Cadence(**{k: v for k, v in c.items() if k != "type_name"}) for c in data["cadences"]],
            tension_curve=data["tension_curve"],
        )

        insert_cadences_into_midi(args.midi, plan, args.output, args.velocity)
        print(f"Cadencias insertadas en: {args.output}")

    elif args.command == "strengthen":
        with open(args.plan, "r", encoding="utf-8") as f:
            data = json.load(f)

        plan = CadencePlan(
            total_bars=data["total_bars"],
            key=data["key"],
            mode=data["mode"],
            form=data["form"],
            cadences=[Cadence(**{k: v for k, v in c.items() if k != "type_name"}) for c in data["cadences"]],
            tension_curve=data["tension_curve"],
        )

        plan = strengthen_cadences(plan, args.intensity)
        save_plan(plan, args.output)
        print_report(plan)
        print(f"\nPlan reforzado guardado en: {args.output}")

    elif args.command == "weaken":
        with open(args.plan, "r", encoding="utf-8") as f:
            data = json.load(f)

        plan = CadencePlan(
            total_bars=data["total_bars"],
            key=data["key"],
            mode=data["mode"],
            form=data["form"],
            cadences=[Cadence(**{k: v for k, v in c.items() if k != "type_name"}) for c in data["cadences"]],
            tension_curve=data["tension_curve"],
        )

        plan = weaken_cadences(plan, args.intensity)
        save_plan(plan, args.output)
        print_report(plan)
        print(f"\nPlan debilitado guardado en: {args.output}")

    elif args.command == "sequence":
        with open(args.plan, "r", encoding="utf-8") as f:
            data = json.load(f)

        plan = CadencePlan(
            total_bars=data["total_bars"],
            key=data["key"],
            mode=data["mode"],
            form=data["form"],
            cadences=[Cadence(**{k: v for k, v in c.items() if k != "type_name"}) for c in data["cadences"]],
            tension_curve=data["tension_curve"],
        )

        plan = sequence_cadences(plan, args.direction)
        save_plan(plan, args.output)
        print_report(plan)
        print(f"\nPlan secuenciado guardado en: {args.output}")


if __name__ == "__main__":
    main()
