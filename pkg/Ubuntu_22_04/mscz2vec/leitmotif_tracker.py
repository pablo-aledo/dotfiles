#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      LEITMOTIF TRACKER  v1.0                                ║
║         Gestión, siembra y rastreo de leitmotifs en obras MIDI              ║
║                                                                              ║
║  Un leitmotif es una célula musical corta asociada a una idea semántica     ║
║  (personaje, emoción, concepto). Este módulo gestiona un banco de           ║
║  leitmotifs y los siembra estratégicamente a lo largo de una obra,          ║
║  transformándolos según el contexto emocional de cada sección.              ║
║                                                                              ║
║  COMANDOS PRINCIPALES:                                                       ║
║    register   — Añadir un motivo al banco                                   ║
║    list        — Ver todos los motivos registrados                           ║
║    show        — Detalle de un motivo (fingerprint, apariciones)            ║
║    plan        — Planificar siembra sobre obra_plan.json de narrator        ║
║    suggest     — Modo exploratorio: propone dónde inyectar en un MIDI      ║
║    inject      — Inyectar motivos según un schedule JSON                   ║
║    trace       — Detectar apariciones en una obra ya compuesta             ║
║    report      — Mapa ASCII/JSON de todas las apariciones                  ║
║    propagate   — Propagar mutación semántica a todos los motivos           ║
║    from-theorist — Importar sugerencias de theorist.json                   ║
║    from-harvest  — Importar motivos extraídos por harvester.py             ║
║                                                                              ║
║  INTEGRACIONES:                                                              ║
║    theorist.py    → from-theorist importa suggested_leitmotifs              ║
║    narrator.py    → plan lee leitmotif_hints por sección                   ║
║    harvester.py   → from-harvest importa motivos extraídos                 ║
║    variation_engine.py → transforma motivos para cada contexto             ║
║    mscz2vec.py    → vectoriza motivos para reconocimiento por similitud     ║
║    analyzer.py    → trace usa sliding-window sobre vectores                ║
║    mutator.py     → propagate delega transformaciones semánticas            ║
║    stitcher.py    → inject genera leitmotif_schedule.json compatible       ║
║    orchestrator.py → preferred_instruments guía asignación de timbres      ║
║                                                                              ║
║  IDENTIFICACIÓN DE MOTIVOS (3 capas):                                       ║
║    [1] Huella de contorno: deltas relativos de pitch (invariante a transp.) ║
║    [2] Huella rítmica: duraciones normalizadas                              ║
║    [3] Vector mscz2vec: similitud coseno si music21 disponible             ║
║                                                                              ║
║  USO:                                                                        ║
║    python leitmotif_tracker.py register motivo.mid --name "destino"        ║
║    python leitmotif_tracker.py register motivo.mid --name "luz" \\         ║
║        --tags esperanza redención --instruments violin1 flute              ║
║    python leitmotif_tracker.py list                                         ║
║    python leitmotif_tracker.py show destino                                 ║
║    python leitmotif_tracker.py plan obra_plan.json                         ║
║    python leitmotif_tracker.py plan obra_plan.json --motifs destino luz    ║
║    python leitmotif_tracker.py plan obra_plan.json --auto                  ║
║    python leitmotif_tracker.py suggest seccion_B.mid                       ║
║    python leitmotif_tracker.py inject obra.mid schedule.json               ║
║    python leitmotif_tracker.py trace obra_final.mid                        ║
║    python leitmotif_tracker.py report obra_final.mid                       ║
║    python leitmotif_tracker.py propagate --instruction "más oscuro"        ║
║    python leitmotif_tracker.py from-theorist plan.theorist.json            ║
║    python leitmotif_tracker.py from-harvest obra.motif_01.mid \\           ║
║        --name "destino" --tags fatalidad                                    ║
║                                                                              ║
║  OPCIONES GLOBALES:                                                          ║
║    --bank FILE      Banco de leitmotifs (default: leitmotif_bank.json)     ║
║    --verbose        Salida detallada                                        ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                  ║
║  OPCIONALES:   music21 (vectorización completa), scipy (similitud coseno)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import copy
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False

try:
    from scipy.spatial.distance import cosine as cosine_distance
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

VERSION = "1.0"
DEFAULT_BANK = "leitmotif_bank.json"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

# Afinidad semántica: qué transformaciones aplica según contexto de tensión
TENSION_TRANSFORMATION_MAP = {
    # tensión alta  → aumentación (más lento, más amenazante) o fragmentación
    "high":   ["augmentation", "fragmentation", "inversion"],
    # tensión media → presentación completa o variación armónica
    "mid":    ["full", "harmonic", "ornament"],
    # tensión baja  → presagio (pp, solo las primeras notas), modal
    "low":    ["presage", "modal", "retrograde"],
}

# Afinidad instrumento ↔ mood semántico
INSTRUMENT_MOOD_AFFINITY = {
    "cello":      ["pérdida", "tristeza", "profundidad", "oscuro", "fatalidad"],
    "viola":      ["melancolía", "ambigüedad", "transición"],
    "violin1":    ["amor", "júbilo", "esperanza", "lírico"],
    "violin2":    ["tensión", "diálogo", "duda"],
    "contrabass": ["amenaza", "peso", "inevitabilidad"],
    "flute":      ["luz", "ligereza", "inocencia", "amanecer"],
    "oboe":       ["nostalgia", "soledad", "dolor", "lamentation"],
    "clarinet":   ["misterio", "ambigüedad", "noche"],
    "bassoon":    ["humor oscuro", "gravedad", "destino"],
    "horn":       ["heroísmo", "distancia", "llamada", "búsqueda"],
    "trumpet":    ["triunfo", "fanfarria", "victoria", "proclama"],
    "trombone":   ["fatalidad", "grandiosidad", "poder", "oscuro"],
}

# Variaciones disponibles en variation_engine.py
VARIATION_CODES = {
    "inversion":      "V01",
    "retrograde":     "V02",
    "augmentation":   "V04",
    "diminution":     "V05",
    "ornament":       "V06",
    "modal":          "V09",
    "harmonic":       "V11",
    "full":           None,   # sin transformación
    "fragmentation":  None,   # se maneja internamente
    "presage":        None,   # pp + fragmento inicial
}

# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS MIDI: EXTRACCIÓN DE HUELLA
# ══════════════════════════════════════════════════════════════════════════════

def extract_notes_from_midi(midi_path: str) -> list[tuple]:
    """
    Extrae lista de (pitch, velocity, time_onset_ticks, duration_ticks).
    Solo notas de pistas melódicas (no canal 10).
    """
    if not MIDO_OK:
        return []
    try:
        mid = mido.MidiFile(midi_path)
        tpb = mid.ticks_per_beat
        notes = []
        for track in mid.tracks:
            if any(getattr(m, 'channel', -1) == 9 for m in track if hasattr(m, 'channel')):
                continue
            active = {}
            tick = 0
            for msg in track:
                tick += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    active[msg.note] = (tick, msg.velocity)
                elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
                    if msg.note in active:
                        onset, vel = active.pop(msg.note)
                        notes.append((msg.note, vel, onset, tick - onset))
        notes.sort(key=lambda x: x[2])
        return notes
    except Exception as e:
        return []


def build_contour_fingerprint(notes: list[tuple]) -> list[int]:
    """
    Capa 1: Huella de contorno melódico.
    Deltas relativos en semitonos entre notas consecutivas.
    Invariante a transposición.
    """
    if len(notes) < 2:
        return []
    pitches = [n[0] for n in notes[:16]]  # máx 16 notas
    return [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]


def build_rhythm_fingerprint(notes: list[tuple]) -> list[float]:
    """
    Capa 2: Huella rítmica.
    Duraciones normalizadas respecto a la duración media.
    Invariante a tempo.
    """
    if not notes:
        return []
    durations = [n[3] for n in notes[:16]]
    mean_dur = max(np.mean(durations), 1)
    return [round(d / mean_dur, 2) for d in durations]


def build_tempo_from_midi(midi_path: str) -> float:
    """Extrae el tempo en BPM de un MIDI."""
    if not MIDO_OK:
        return 120.0
    try:
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return round(60_000_000 / msg.tempo, 1)
    except Exception:
        pass
    return 120.0


def fingerprint_motif(midi_path: str, verbose: bool = False) -> dict:
    """
    Construye la firma completa de un motivo MIDI.
    Retorna dict con contour, rhythm, n_notes, duration_bars, tempo.
    """
    notes = extract_notes_from_midi(midi_path)
    if not notes:
        if verbose:
            print(f"  [warn] No se pudieron extraer notas de {midi_path}")
        return {"contour": [], "rhythm": [], "n_notes": 0,
                "mean_pitch": 60, "duration_bars": 2, "tempo": 120.0}

    contour = build_contour_fingerprint(notes)
    rhythm  = build_rhythm_fingerprint(notes)

    if MIDO_OK:
        try:
            mid = mido.MidiFile(midi_path)
            tpb = mid.ticks_per_beat
            total_ticks = sum(
                sum(m.time for m in t) for t in mid.tracks
            )
            # estimación de compases (asume 4/4)
            duration_bars = max(1, round(total_ticks / (tpb * 4)))
        except Exception:
            duration_bars = 2
    else:
        duration_bars = 2

    mean_pitch = round(float(np.mean([n[0] for n in notes])), 1)
    tempo      = build_tempo_from_midi(midi_path)

    return {
        "contour":       contour,
        "rhythm":        rhythm,
        "n_notes":       len(notes),
        "mean_pitch":    mean_pitch,
        "duration_bars": duration_bars,
        "tempo":         tempo,
    }


def contour_similarity(c1: list[int], c2: list[int]) -> float:
    """
    Similitud entre dos contornos melódicos [0, 1].
    Compara los N primeros deltas con tolerancia ±1 semitono.
    """
    if not c1 or not c2:
        return 0.0
    n = min(len(c1), len(c2), 8)
    matches = sum(1 for i in range(n) if abs(c1[i] - c2[i]) <= 1)
    return matches / n


def rhythm_similarity(r1: list[float], r2: list[float]) -> float:
    """Similitud rítmica [0, 1]. Tolerancia ±20%."""
    if not r1 or not r2:
        return 0.0
    n = min(len(r1), len(r2), 8)
    matches = sum(1 for i in range(n) if abs(r1[i] - r2[i]) < 0.2)
    return matches / n


def motif_similarity(fp_a: dict, fp_b: dict) -> float:
    """
    Similitud global entre dos huellas de motivo [0, 1].
    Combina contorno (60%) + ritmo (40%).
    """
    cs = contour_similarity(fp_a.get("contour", []), fp_b.get("contour", []))
    rs = rhythm_similarity(fp_a.get("rhythm", []), fp_b.get("rhythm", []))
    return round(0.6 * cs + 0.4 * rs, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  BANCO DE LEITMOTIFS
# ══════════════════════════════════════════════════════════════════════════════

def load_bank(bank_path: str) -> dict:
    """Carga el banco de leitmotifs desde JSON. Crea uno vacío si no existe."""
    p = Path(bank_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"version": VERSION, "motifs": {}}


def save_bank(bank: dict, bank_path: str):
    """Guarda el banco de leitmotifs en JSON."""
    bank["last_modified"] = datetime.now().isoformat()
    with open(bank_path, "w", encoding="utf-8") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)


def register_motif(bank: dict, name: str, midi_path: str,
                   tags: list[str] = None,
                   preferred_instruments: list[str] = None,
                   description: str = "",
                   tension_affinity: str = "mid",
                   verbose: bool = False) -> dict:
    """
    Registra un nuevo leitmotif en el banco.
    Calcula y almacena su huella (fingerprint).
    """
    midi_abs = str(Path(midi_path).resolve())

    if verbose:
        print(f"  Extrayendo huella de {midi_path}...")

    fp = fingerprint_motif(midi_abs, verbose=verbose)

    # Inferir instrumentos preferidos desde los tags si no se especifican
    if not preferred_instruments and tags:
        preferred_instruments = _infer_instruments_from_tags(tags)
    preferred_instruments = preferred_instruments or ["violin1"]

    motif = {
        "name":                  name,
        "midi":                  midi_abs,
        "fingerprint":           fp,
        "semantics": {
            "tags":                  tags or [],
            "description":           description,
            "tension_affinity":      tension_affinity,  # low | mid | high
            "preferred_instruments": preferred_instruments,
        },
        "appearances":    [],  # historial de apariciones
        "registered_at":  datetime.now().isoformat(),
    }

    bank["motifs"][name] = motif

    if verbose:
        print(f"  ✓ Motivo '{name}' registrado")
        print(f"    Contorno:    {fp['contour'][:6]}...")
        print(f"    Ritmo:       {fp['rhythm'][:6]}...")
        print(f"    Notas:       {fp['n_notes']}")
        print(f"    Instrumentos preferidos: {preferred_instruments}")

    return motif


def _infer_instruments_from_tags(tags: list[str]) -> list[str]:
    """Infiere instrumentos preferidos a partir de las tags semánticas."""
    tag_set = set(t.lower() for t in tags)
    scores = {}
    for instrument, moods in INSTRUMENT_MOOD_AFFINITY.items():
        score = sum(1 for m in moods if any(t in m or m in t for t in tag_set))
        if score > 0:
            scores[instrument] = score
    if scores:
        return [max(scores, key=scores.get)]
    return ["violin1"]


# ══════════════════════════════════════════════════════════════════════════════
#  PLANIFICACIÓN DE SIEMBRA
# ══════════════════════════════════════════════════════════════════════════════

def plan_seeding(bank: dict,
                 narrator_plan: dict,
                 motif_names: list[str] = None,
                 auto: bool = False,
                 verbose: bool = False) -> dict:
    """
    Genera un plan de siembra (leitmotif_schedule.json) a partir del
    plan de narrator. Modo exploratorio: propone opciones sin ejecutar.

    Retorna un schedule dict con lista de inyecciones propuestas.
    """
    motifs = bank.get("motifs", {})
    sections = narrator_plan.get("sections", [])
    curves   = narrator_plan.get("curves", {})
    tension_curve = curves.get("tension", [])
    n_bars   = narrator_plan.get("bars", 32)

    # Filtrar motivos a usar
    if motif_names:
        active_motifs = {k: v for k, v in motifs.items() if k in motif_names}
    else:
        active_motifs = motifs

    if not active_motifs:
        print("  [warn] No hay motivos en el banco. Usa 'register' primero.")
        return {"injections": []}

    schedule = {
        "version":      VERSION,
        "generated":    datetime.now().isoformat(),
        "work_bars":    n_bars,
        "motifs_used":  list(active_motifs.keys()),
        "injections":   [],
    }

    # Calcular el inicio de cada sección en compases absolutos
    bar_cursor = 0
    section_starts = []
    for sec in sections:
        section_starts.append((sec, bar_cursor))
        bar_cursor += sec.get("bars", 8)

    # Para cada sección, generar propuestas de inyección
    for sec, start_bar in section_starts:
        sec_id      = sec.get("id", sec.get("label", "?"))
        sec_bars    = sec.get("bars", 8)
        sec_label   = sec.get("label", sec_id)
        end_bar     = start_bar + sec_bars

        # Tensión media de esta sección
        if tension_curve and len(tension_curve) >= end_bar:
            sec_tension = float(np.mean(tension_curve[start_bar:end_bar]))
        else:
            sec_tension = 0.5

        tension_zone = _tension_zone(sec_tension)

        # Proponer inyecciones para cada motivo activo
        for motif_name, motif in active_motifs.items():
            motif_affinity = motif["semantics"].get("tension_affinity", "mid")

            # ¿Este motivo "encaja" dramáticamente en esta sección?
            fit_score = _dramatic_fit(motif, sec, sec_tension, narrator_plan)

            if fit_score < 0.3:
                continue  # no encaja, saltar

            # Transformación recomendada según tensión
            transforms = TENSION_TRANSFORMATION_MAP.get(tension_zone, ["full"])
            transform  = transforms[0]

            # Posición dentro de la sección: inicio, mitad, final
            injection_bar = _choose_injection_bar(start_bar, sec_bars, tension_zone)

            # Instrumento recomendado
            instrument = _choose_instrument(motif, sec_tension)

            # Dinámica recomendada
            dynamic = _choose_dynamic(sec_tension, transform)

            proposal = {
                "motif":       motif_name,
                "section":     sec_label,
                "bar":         injection_bar,
                "transform":   transform,
                "instrument":  instrument,
                "dynamic":     dynamic,
                "tension":     round(sec_tension, 2),
                "fit_score":   round(fit_score, 2),
                "reason":      _build_reason(motif_name, sec_label, transform,
                                              tension_zone, instrument),
                "approved":    False,   # el usuario aprueba
                "midi_output": None,    # se rellena al inyectar
            }

            schedule["injections"].append(proposal)

    # Ordenar por compás
    schedule["injections"].sort(key=lambda x: x["bar"])

    # Detección de huecos dramáticos
    schedule["gaps"] = _detect_gaps(schedule, active_motifs, n_bars, tension_curve)

    return schedule


def _tension_zone(tension: float) -> str:
    if tension >= 0.65:
        return "high"
    elif tension >= 0.35:
        return "mid"
    return "low"


def _dramatic_fit(motif: dict, section: dict, tension: float,
                  narrator_plan: dict) -> float:
    """
    Calcula cuán bien encaja un motivo en una sección [0, 1].
    Considera: afinidad de tensión, hints del plan, tags semánticas.
    """
    score = 0.0

    # 1. Afinidad de tensión
    affinity = motif["semantics"].get("tension_affinity", "mid")
    zone = _tension_zone(tension)
    if affinity == zone:
        score += 0.5
    elif abs(["low", "mid", "high"].index(affinity) -
             ["low", "mid", "high"].index(zone)) == 1:
        score += 0.25

    # 2. Leitmotif hints en el plan de narrator
    sec_hints = section.get("leitmotif_hints", [])
    if motif["name"] in sec_hints:
        score += 0.4

    # 3. Tags semánticas vs. descripción de la sección
    sec_desc = section.get("description", section.get("label", "")).lower()
    for tag in motif["semantics"].get("tags", []):
        if tag.lower() in sec_desc:
            score += 0.1

    return min(score, 1.0)


def _choose_injection_bar(start_bar: int, sec_bars: int,
                           tension_zone: str) -> int:
    """Elige el compás de inyección dentro de la sección."""
    if tension_zone == "high":
        # En el clímax: inicio o centro de la sección
        return start_bar + sec_bars // 4
    elif tension_zone == "low":
        # Como presagio: justo antes de que empiece la acción
        return start_bar
    else:
        return start_bar + sec_bars // 2


def _choose_instrument(motif: dict, tension: float) -> str:
    """Elige el instrumento más apropiado según tensión y preferencias."""
    preferred = motif["semantics"].get("preferred_instruments", ["violin1"])
    if not preferred:
        return "violin1"
    # En alta tensión usar instrumentos más oscuros/graves si están disponibles
    if tension > 0.7 and any(i in preferred for i in ["cello", "trombone", "contrabass"]):
        for instr in ["cello", "trombone", "contrabass"]:
            if instr in preferred:
                return instr
    return preferred[0]


def _choose_dynamic(tension: float, transform: str) -> str:
    """Elige la dinámica según tensión y tipo de transformación."""
    if transform == "presage":
        return "pp"
    if tension > 0.8:
        return "ff"
    elif tension > 0.6:
        return "mf"
    elif tension > 0.4:
        return "mp"
    return "p"


def _build_reason(motif_name: str, section: str, transform: str,
                   tension_zone: str, instrument: str) -> str:
    """Genera una explicación legible de la propuesta de inyección."""
    transform_descs = {
        "full":          "presentación completa",
        "augmentation":  "aumentación rítmica (más lento, más amenazante)",
        "inversion":     "inversión melódica (giro dramático)",
        "fragmentation": "fragmento inicial (anticipación)",
        "presage":       "presagio en pp (3 primeras notas)",
        "harmonic":      "reharmonización (color diferente)",
        "modal":         "versión modal (cambio de modo)",
        "ornament":      "ornamentado (más expresivo)",
        "retrograde":    "retrógrado (reflexión)",
        "diminution":    "diminución (urgencia, más rápido)",
    }
    t_descs = {"high": "tensión alta", "mid": "tensión media", "low": "tensión baja"}
    return (
        f"Sección '{section}' ({t_descs.get(tension_zone, '?')}): "
        f"'{motif_name}' en {transform_descs.get(transform, transform)} "
        f"— {instrument}"
    )


def _detect_gaps(schedule: dict, motifs: dict, n_bars: int,
                  tension_curve: list) -> list[dict]:
    """
    Detecta compases donde ningún leitmotif aparece pero la tensión sugiere
    que debería aparecer uno.
    """
    if not tension_curve:
        return []

    injected_bars = set(inj["bar"] for inj in schedule.get("injections", []))
    gaps = []

    # Ventana de 4 compases
    for start in range(0, min(n_bars, len(tension_curve)) - 4, 4):
        end = start + 4
        window_tension = float(np.mean(tension_curve[start:end]))
        has_injection  = any(start <= b < end for b in injected_bars)

        if not has_injection and window_tension > 0.65:
            # Hueco dramático: tensión alta sin leitmotif
            gaps.append({
                "bar_start":  start,
                "bar_end":    end,
                "tension":    round(window_tension, 2),
                "suggestion": "Considera inyectar un leitmotif de tensión alta aquí",
            })

    return gaps


# ══════════════════════════════════════════════════════════════════════════════
#  MODO EXPLORATORIO (suggest)
# ══════════════════════════════════════════════════════════════════════════════

def suggest_interactive(bank: dict,
                         midi_path: str,
                         verbose: bool = False) -> dict:
    """
    Analiza un MIDI existente y propone dónde inyectar leitmotifs.
    Modo exploratorio: muestra opciones con contexto, el usuario elige.
    Retorna el schedule con las inyecciones aprobadas.
    """
    motifs = bank.get("motifs", {})
    if not motifs:
        print("  El banco está vacío. Usa 'register' primero.")
        return {"injections": []}

    # Análisis básico del MIDI objetivo
    notes = extract_notes_from_midi(midi_path)
    tempo = build_tempo_from_midi(midi_path)

    if not notes:
        print(f"  [warn] No se pudieron extraer notas de {midi_path}")
        return {"injections": []}

    # Estimar compases y tensión por ventana
    if MIDO_OK:
        try:
            mid = mido.MidiFile(midi_path)
            tpb = mid.ticks_per_beat
            total_ticks = max(n[2] + n[3] for n in notes)
            n_bars = max(1, round(total_ticks / (tpb * 4)))
        except Exception:
            n_bars = 8
    else:
        n_bars = 8

    # Detectar ventanas de oportunidad (silencios, cadencias aproximadas)
    opportunities = _detect_opportunities(notes, n_bars)

    print(f"\n{'═' * 62}")
    print(f"  LEITMOTIF TRACKER — Modo exploratorio")
    print(f"  MIDI: {Path(midi_path).name}  |  ~{n_bars} compases  |  {tempo} BPM")
    print(f"{'═' * 62}")

    if not opportunities:
        print("  No se detectaron ventanas de oportunidad claras.")
        return {"injections": []}

    approved = []

    print(f"\n  Se detectaron {len(opportunities)} oportunidades de inyección:\n")

    for i, opp in enumerate(opportunities, 1):
        bar      = opp["bar"]
        tension  = opp["tension"]
        opp_type = opp["type"]

        print(f"  [{i}] Compás {bar} — {opp_type} (tensión aprox. {tension:.2f})")

        # Proponer el motivo más adecuado para este momento
        best_motif, best_transform = _best_motif_for_context(motifs, tension)
        instr    = _choose_instrument(motifs[best_motif], tension)
        dynamic  = _choose_dynamic(tension, best_transform)

        reason = _build_reason(best_motif, f"compás {bar}",
                               best_transform, _tension_zone(tension), instr)
        print(f"     Propuesta: {reason}")
        print(f"     Dinámica:  {dynamic}")

        # Mostrar alternativas
        alternatives = _build_alternatives(motifs, tension, best_motif, best_transform)
        if alternatives:
            for j, alt in enumerate(alternatives[:3], 1):
                print(f"     Alt {j}:    {alt['motif']} — {alt['transform']} ({alt['instrument']})")

        try:
            choice = input(f"\n     ¿Qué hacemos? [s=sí/n=no/1-3=alternativa/e=editar] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if choice == "s" or choice == "":
            injection = {
                "motif":      best_motif,
                "section":    f"compás {bar}",
                "bar":        bar,
                "transform":  best_transform,
                "instrument": instr,
                "dynamic":    dynamic,
                "tension":    round(tension, 2),
                "reason":     reason,
                "approved":   True,
                "midi_output": None,
            }
            approved.append(injection)
            print(f"     ✓ Aprobado")

        elif choice in ("1", "2", "3"):
            idx = int(choice) - 1
            if idx < len(alternatives):
                alt = alternatives[idx]
                injection = {
                    "motif":      alt["motif"],
                    "section":    f"compás {bar}",
                    "bar":        bar,
                    "transform":  alt["transform"],
                    "instrument": alt["instrument"],
                    "dynamic":    dynamic,
                    "tension":    round(tension, 2),
                    "reason":     _build_reason(alt["motif"], f"compás {bar}",
                                                alt["transform"],
                                                _tension_zone(tension), alt["instrument"]),
                    "approved":   True,
                    "midi_output": None,
                }
                approved.append(injection)
                print(f"     ✓ Alternativa {choice} aprobada")

        elif choice == "e":
            injection = _edit_injection_manually(motifs, bar, tension)
            if injection:
                approved.append(injection)

        else:
            print(f"     — Saltado")

    schedule = {
        "version":    VERSION,
        "generated":  datetime.now().isoformat(),
        "source_midi": midi_path,
        "injections": approved,
        "gaps":        [],
    }

    print(f"\n  {len(approved)} inyecciones aprobadas de {len(opportunities)} propuestas.")
    return schedule


def _detect_opportunities(notes: list[tuple], n_bars: int) -> list[dict]:
    """
    Detecta ventanas de oportunidad para inyectar leitmotifs:
    - Silencios de ≥1 beat entre notas
    - Inicio de cada 4 compases (heurística)
    """
    if not notes:
        return []

    opportunities = []
    seen_bars = set()

    # Heurística: compás cada N ticks (estimado)
    if len(notes) > 1:
        total_span = notes[-1][2] - notes[0][2]
        ticks_per_bar = max(1, total_span // max(n_bars, 1))
    else:
        ticks_per_bar = 480 * 4

    # Silencios entre notas
    for i in range(1, len(notes)):
        gap = notes[i][2] - (notes[i-1][2] + notes[i-1][3])
        if gap > ticks_per_bar * 0.5:  # silencio de ≥ medio compás
            bar = min(n_bars - 1, int(notes[i][2] / ticks_per_bar))
            if bar not in seen_bars:
                seen_bars.add(bar)
                tension = 0.3 + 0.5 * (bar / max(n_bars, 1))
                opportunities.append({
                    "bar":     bar,
                    "tension": round(tension, 2),
                    "type":    "silencio/pausa",
                })

    # Inicio de cada 4 compases como oportunidad estructural
    for bar in range(0, n_bars, 4):
        if bar not in seen_bars:
            seen_bars.add(bar)
            tension = 0.2 + 0.7 * (bar / max(n_bars, 1))
            opportunities.append({
                "bar":     bar,
                "tension": round(tension, 2),
                "type":    "punto estructural",
            })

    opportunities.sort(key=lambda x: x["bar"])
    return opportunities[:8]  # máx 8 propuestas


def _best_motif_for_context(motifs: dict, tension: float) -> tuple[str, str]:
    """Elige el motivo y transformación más adecuados para una tensión dada."""
    zone = _tension_zone(tension)
    best_name   = list(motifs.keys())[0]
    best_score  = -1

    for name, motif in motifs.items():
        affinity = motif["semantics"].get("tension_affinity", "mid")
        zones = ["low", "mid", "high"]
        dist  = abs(zones.index(affinity) - zones.index(zone))
        score = 1.0 - dist * 0.4
        if score > best_score:
            best_score = score
            best_name  = name

    transforms = TENSION_TRANSFORMATION_MAP.get(zone, ["full"])
    return best_name, transforms[0]


def _build_alternatives(motifs: dict, tension: float,
                          exclude_motif: str,
                          exclude_transform: str) -> list[dict]:
    """Construye alternativas de motivo/transformación."""
    zone = _tension_zone(tension)
    transforms = TENSION_TRANSFORMATION_MAP.get(zone, ["full"])
    alts = []

    # Alternativas de transformación del mismo motivo
    for t in transforms[1:]:
        if t != exclude_transform:
            best_name = exclude_motif
            instr = _choose_instrument(motifs[best_name], tension)
            alts.append({"motif": best_name, "transform": t, "instrument": instr})

    # Otros motivos
    for name, motif in motifs.items():
        if name != exclude_motif:
            t     = transforms[0]
            instr = _choose_instrument(motif, tension)
            alts.append({"motif": name, "transform": t, "instrument": instr})

    return alts[:3]


def _edit_injection_manually(motifs: dict, bar: int, tension: float) -> dict | None:
    """Permite editar manualmente los parámetros de una inyección."""
    print(f"     Motivos disponibles: {', '.join(motifs.keys())}")
    try:
        motif_name = input("     Motivo: ").strip()
        if motif_name not in motifs:
            print("     Motivo no encontrado.")
            return None
        transform = input(f"     Transformación [{'/'.join(TENSION_TRANSFORMATION_MAP['mid'])}]: ").strip() or "full"
        instr     = input(f"     Instrumento [{motifs[motif_name]['semantics']['preferred_instruments'][0]}]: ").strip()
        if not instr:
            instr = motifs[motif_name]["semantics"]["preferred_instruments"][0]
        dynamic   = input("     Dinámica [pp/p/mp/mf/f/ff]: ").strip() or "mp"
    except (EOFError, KeyboardInterrupt):
        return None

    return {
        "motif":      motif_name,
        "section":    f"compás {bar}",
        "bar":        bar,
        "transform":  transform,
        "instrument": instr,
        "dynamic":    dynamic,
        "tension":    round(tension, 2),
        "reason":     "(edición manual)",
        "approved":   True,
        "midi_output": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INYECCIÓN: GENERACIÓN DE FRAGMENTOS MIDI
# ══════════════════════════════════════════════════════════════════════════════

def inject_motifs(bank: dict,
                   schedule: dict,
                   output_dir: str = ".",
                   variation_engine: str = None,
                   verbose: bool = False) -> dict:
    """
    Ejecuta las inyecciones aprobadas del schedule.
    Para cada inyección, llama a variation_engine.py con la transformación
    adecuada y guarda el fragmento MIDI resultante.

    Actualiza el schedule con los paths de los MIDIs generados.
    Registra cada aparición en el banco.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    motifs = bank.get("motifs", {})

    approved = [inj for inj in schedule.get("injections", [])
                if inj.get("approved", False)]

    if not approved:
        print("  No hay inyecciones aprobadas en el schedule.")
        return schedule

    ve_script = variation_engine or _find_script("variation_engine.py")

    print(f"\n  Inyectando {len(approved)} leitmotifs...")

    for inj in approved:
        motif_name = inj["motif"]
        if motif_name not in motifs:
            if verbose:
                print(f"  [warn] Motivo '{motif_name}' no encontrado en el banco")
            continue

        motif     = motifs[motif_name]
        transform = inj.get("transform", "full")
        bar       = inj.get("bar", 0)
        section   = inj.get("section", "?")

        # Nombre del archivo de salida
        safe_name = motif_name.replace(" ", "_")
        out_name  = f"leitmotif_{safe_name}_bar{bar:03d}_{transform}.mid"
        out_path  = str(Path(output_dir) / out_name)

        if verbose:
            print(f"  → {motif_name} @ compás {bar}: {transform} → {out_name}")

        # Generar fragmento transformado
        success = _generate_transformed_motif(
            motif_midi=motif["midi"],
            transform=transform,
            out_path=out_path,
            ve_script=ve_script,
            dynamic=inj.get("dynamic", "mp"),
            verbose=verbose,
        )

        inj["midi_output"] = out_path if success else None

        # Registrar aparición en el banco
        appearance = {
            "bar":        bar,
            "section":    section,
            "transform":  transform,
            "instrument": inj.get("instrument", "?"),
            "dynamic":    inj.get("dynamic", "mp"),
            "midi_output": out_path if success else None,
            "timestamp":  datetime.now().isoformat(),
        }
        motifs[motif_name]["appearances"].append(appearance)

        status = "✓" if success else "⚠"
        print(f"  {status} {motif_name} (compás {bar}) → {out_name}")

    return schedule


def _generate_transformed_motif(motif_midi: str, transform: str,
                                  out_path: str, ve_script: str | None,
                                  dynamic: str = "mp",
                                  verbose: bool = False) -> bool:
    """
    Genera un fragmento MIDI transformado usando variation_engine.py.
    Fallback: copia el MIDI original si variation_engine no está disponible.
    """
    variation_code = VARIATION_CODES.get(transform)

    # Casos especiales que manejamos internamente
    if transform in ("full", "presage", "fragmentation") or not variation_code:
        return _handle_special_transform(motif_midi, transform, out_path,
                                          dynamic, verbose)

    if not ve_script or not Path(ve_script).exists():
        if verbose:
            print(f"    [warn] variation_engine.py no encontrado, copiando original")
        return _copy_midi(motif_midi, out_path)

    cmd = [
        sys.executable, ve_script,
        motif_midi,
        "--variations", variation_code,
        "--output-dir", str(Path(out_path).parent),
    ]

    try:
        result = subprocess.run(cmd, capture_output=not verbose,
                                text=True, timeout=60)
        # variation_engine genera el archivo con su propia nomenclatura
        # Intentar encontrar el archivo generado y renombrarlo
        var_glob = list(Path(out_path).parent.glob(
            f"{Path(motif_midi).stem}*{variation_code}*.mid"
        ))
        if var_glob:
            shutil.move(str(var_glob[0]), out_path)
            return True
        # Si no lo encontró, copiar el original
        return _copy_midi(motif_midi, out_path)
    except Exception as e:
        if verbose:
            print(f"    [error] variation_engine: {e}")
        return _copy_midi(motif_midi, out_path)


def _handle_special_transform(motif_midi: str, transform: str,
                                out_path: str, dynamic: str,
                                verbose: bool) -> bool:
    """
    Maneja transformaciones especiales que no delegan a variation_engine:
    - full:         copia directa con ajuste de velocidad
    - presage:      solo las primeras 3 notas, pp
    - fragmentation: solo las primeras N notas
    """
    if not MIDO_OK:
        return _copy_midi(motif_midi, out_path)

    try:
        mid = mido.MidiFile(motif_midi)
        out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

        # Mapa de velocidad según dinámica
        vel_map = {"pp": 25, "p": 45, "mp": 60, "mf": 75, "f": 90, "ff": 110}
        target_vel = vel_map.get(dynamic, 64)

        # Número máximo de notas según transformación
        max_notes = {
            "full":         999,
            "presage":      3,
            "fragmentation": 5,
        }.get(transform, 999)

        for track in mid.tracks:
            new_track = mido.MidiTrack()
            note_count = 0
            for msg in track:
                if msg.type == "note_on" and msg.velocity > 0:
                    note_count += 1
                    if note_count > max_notes:
                        # Agregar note_off para la última nota activa
                        new_track.append(msg.copy(velocity=0))
                        break
                    new_track.append(msg.copy(velocity=min(msg.velocity,
                                                            target_vel)))
                elif msg.type in ("note_off", "note_on"):
                    new_track.append(msg.copy(velocity=0))
                else:
                    new_track.append(msg)
            out.tracks.append(new_track)

        out.save(out_path)
        return True
    except Exception as e:
        if verbose:
            print(f"    [error] transformación especial: {e}")
        return _copy_midi(motif_midi, out_path)


def _copy_midi(src: str, dst: str) -> bool:
    """Copia un archivo MIDI como fallback."""
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  RASTREO (trace): DETECTAR APARICIONES EN OBRA EXISTENTE
# ══════════════════════════════════════════════════════════════════════════════

def trace_appearances(bank: dict,
                       midi_path: str,
                       threshold: float = 0.55,
                       verbose: bool = False) -> dict:
    """
    Busca apariciones de los motivos del banco en una obra ya compuesta.
    Usa sliding-window de 4-8 compases sobre el MIDI objetivo.

    Retorna un informe de apariciones detectadas con similitud.
    """
    motifs = bank.get("motifs", {})
    if not motifs:
        print("  El banco está vacío.")
        return {"detections": [], "gaps": []}

    notes = extract_notes_from_midi(midi_path)
    if not notes:
        print(f"  [warn] No se pudieron extraer notas de {midi_path}")
        return {"detections": [], "gaps": []}

    if MIDO_OK:
        try:
            mid = mido.MidiFile(midi_path)
            tpb = mid.ticks_per_beat
            total_ticks = max(n[2] + n[3] for n in notes)
            n_bars = max(1, round(total_ticks / (tpb * 4)))
            ticks_per_bar = tpb * 4
        except Exception:
            n_bars = 8
            ticks_per_bar = 480 * 4
    else:
        n_bars = 8
        ticks_per_bar = 480 * 4

    detections = []

    # Sliding window de ~4 notas
    window_size = 4
    for start_idx in range(len(notes) - window_size + 1):
        window_notes = notes[start_idx:start_idx + window_size]
        w_fp = {
            "contour": build_contour_fingerprint(window_notes),
            "rhythm":  build_rhythm_fingerprint(window_notes),
        }
        w_bar = int(window_notes[0][2] / ticks_per_bar)

        for motif_name, motif in motifs.items():
            m_fp = motif.get("fingerprint", {})
            sim  = motif_similarity(m_fp, w_fp)

            if sim >= threshold:
                # Evitar duplicados en el mismo compás
                already = any(
                    d["motif"] == motif_name and abs(d["bar"] - w_bar) < 2
                    for d in detections
                )
                if not already:
                    detections.append({
                        "motif":      motif_name,
                        "bar":        w_bar,
                        "similarity": round(sim, 3),
                        "notes_idx":  start_idx,
                        "type":       "detected",
                    })
                    if verbose:
                        print(f"  ✓ '{motif_name}' detectado en compás {w_bar} "
                              f"(similitud {sim:.2f})")

    detections.sort(key=lambda x: x["bar"])

    # Detectar huecos (tramos sin leitmotif)
    detected_bars = set(d["bar"] for d in detections)
    gaps = []
    for bar in range(n_bars):
        if bar not in detected_bars and bar % 8 == 0:
            gaps.append({"bar": bar, "note": "Sin leitmotif detectado"})

    return {
        "midi":       midi_path,
        "n_bars":     n_bars,
        "detections": detections,
        "gaps":       gaps,
        "threshold":  threshold,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTE VISUAL
# ══════════════════════════════════════════════════════════════════════════════

def print_report(trace_result: dict, bank: dict):
    """
    Imprime un mapa ASCII de apariciones de leitmotifs a lo largo de la obra.
    Cada motivo ocupa una fila; las columnas son grupos de 4 compases.
    """
    n_bars    = trace_result.get("n_bars", 32)
    detections = trace_result.get("detections", [])
    gaps       = trace_result.get("gaps", [])
    motifs     = bank.get("motifs", {})

    print(f"\n{'═' * 62}")
    print(f"  LEITMOTIF TRACKER — Mapa de apariciones")
    print(f"  Obra: {trace_result.get('midi', '?')}  |  {n_bars} compases")
    print(f"{'═' * 62}")

    if not detections:
        print("  No se detectaron apariciones.")
    else:
        cols = max(1, n_bars // 4)

        # Cabecera de compases
        header = "  Motivo          |"
        for c in range(cols):
            header += f"{c*4:3d}|"
        print(f"\n{header}")
        print(f"  {'─'*16}|{'─'*4*cols}")

        for motif_name in motifs:
            row = f"  {motif_name:<16}|"
            for c in range(cols):
                bar_start = c * 4
                bar_end   = bar_start + 4
                found = [d for d in detections
                         if d["motif"] == motif_name
                         and bar_start <= d["bar"] < bar_end]
                if found:
                    sim = found[0]["similarity"]
                    if sim >= 0.8:
                        cell = " ██ "
                    elif sim >= 0.65:
                        cell = " ▓▓ "
                    else:
                        cell = " ░░ "
                else:
                    cell = "    "
                row += f"{cell}|"
            print(row)

        print(f"\n  Leyenda: ██=alta similitud  ▓▓=media  ░░=baja")
        print(f"\n  Detecciones totales: {len(detections)}")
        for d in detections:
            print(f"    compás {d['bar']:3d}: '{d['motif']}' "
                  f"(similitud {d['similarity']:.2f})")

    if gaps:
        print(f"\n  Huecos dramáticos detectados:")
        for g in gaps[:5]:
            print(f"    compás {g['bar']:3d}: {g['note']}")

    print(f"\n{'═' * 62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PROPAGACIÓN: MUTACIÓN SEMÁNTICA DE TODOS LOS MOTIVOS
# ══════════════════════════════════════════════════════════════════════════════

def propagate_mutation(bank: dict,
                        instruction: str,
                        motif_names: list[str] = None,
                        mutator_script: str = None,
                        verbose: bool = False) -> dict:
    """
    Propaga una instrucción semántica a todos los motivos del banco
    (o a los especificados), llamando a mutator.py por cada uno.

    Retorna el banco actualizado con los nuevos MIDIs mutados.
    """
    motifs    = bank.get("motifs", {})
    mut_script = mutator_script or _find_script("mutator.py")

    target_motifs = motif_names or list(motifs.keys())

    print(f"\n  Propagando '{instruction}' a {len(target_motifs)} motivos...")

    for name in target_motifs:
        if name not in motifs:
            print(f"  [warn] Motivo '{name}' no encontrado")
            continue

        motif    = motifs[name]
        midi_src = motif["midi"]

        if not Path(midi_src).exists():
            print(f"  [warn] MIDI no encontrado: {midi_src}")
            continue

        if verbose:
            print(f"  → mutando '{name}': {instruction}")

        # Ruta de salida: mismo directorio con sufijo _mutated
        src_path = Path(midi_src)
        out_path = str(src_path.parent /
                       f"{src_path.stem}_mutated_{name.replace(' ', '_')}.mid")

        if mut_script and Path(mut_script).exists():
            cmd = [
                sys.executable, mut_script,
                midi_src,
                instruction,
                "--output", out_path,
                "--batch-mode",  # modo silencioso para lote (implica --no-llm)
            ]
            try:
                result = subprocess.run(cmd, capture_output=not verbose,
                                        text=True, timeout=120)
                if result.returncode == 0 and Path(out_path).exists():
                    # Actualizar el banco con la nueva versión mutada
                    motifs[name]["midi"] = out_path
                    motifs[name]["fingerprint"] = fingerprint_motif(out_path, verbose)
                    print(f"  ✓ '{name}' mutado → {Path(out_path).name}")
                else:
                    print(f"  ⚠ '{name}' — mutator falló, motivo sin cambios")
            except Exception as e:
                print(f"  ⚠ '{name}' — error: {e}")
        else:
            print(f"  [warn] mutator.py no disponible — '{name}' sin cambios")
            print(f"         Instala mutator.py en el mismo directorio.")

    return bank


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRACIÓN CON THEORIST
# ══════════════════════════════════════════════════════════════════════════════

def import_from_theorist(bank: dict,
                          theorist_json_path: str,
                          verbose: bool = False) -> dict:
    """
    Importa suggested_leitmotifs desde un archivo .theorist.json.

    theorist.py puede incluir en su output un bloque 'suggested_leitmotifs'
    con nombre, tags, tension_affinity y preferred_instruments.
    Si no existe, sugiere a theorist que lo añada.
    """
    try:
        with open(theorist_json_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
    except Exception as e:
        print(f"  Error al leer {theorist_json_path}: {e}")
        return bank

    suggested = plan.get("suggested_leitmotifs", [])

    if not suggested:
        print(f"  El archivo .theorist.json no contiene 'suggested_leitmotifs'.")
        print(f"  Para habilitarlo, añade al system prompt de theorist.py:")
        print(f"    '\"suggested_leitmotifs\": [{{\"name\": ..., \"tags\": [...], ...}}]'")
        print(f"\n  Información de la obra en el plan:")
        sem = plan.get("semantic_layer", {})
        print(f"    Emoción central: {sem.get('core_emotion', '—')}")
        print(f"    Secundarias:     {', '.join(sem.get('secondary_emotions', []))}")
        print(f"\n  Sugerencia automática de leitmotifs basada en la semántica:")
        auto_suggested = _auto_suggest_from_semantics(sem)
        for s in auto_suggested:
            print(f"    → '{s['name']}': tags={s['tags']}")
        suggested = auto_suggested

    imported = 0
    for s in suggested:
        name = s.get("name", "").strip()
        if not name:
            continue
        if name in bank["motifs"]:
            if verbose:
                print(f"  [skip] '{name}' ya existe en el banco")
            continue

        # Crear entrada de motivo sin MIDI (pendiente de registrar)
        bank["motifs"][name] = {
            "name":    name,
            "midi":    None,   # pendiente: el usuario debe proveer el MIDI
            "fingerprint": {"contour": [], "rhythm": [], "n_notes": 0,
                             "mean_pitch": 60, "duration_bars": 2, "tempo": 120.0},
            "semantics": {
                "tags":                  s.get("tags", []),
                "description":           s.get("description", ""),
                "tension_affinity":      s.get("tension_affinity", "mid"),
                "preferred_instruments": s.get("preferred_instruments",
                                               _infer_instruments_from_tags(
                                                   s.get("tags", []))),
            },
            "appearances":    [],
            "registered_at":  datetime.now().isoformat(),
            "status":         "pending_midi",  # recordatorio
        }
        imported += 1
        print(f"  + '{name}' importado (MIDI pendiente)")
        if verbose:
            print(f"    Tags: {s.get('tags', [])}")
            print(f"    Afinidad de tensión: {s.get('tension_affinity', 'mid')}")

    print(f"\n  {imported} leitmotifs importados desde {Path(theorist_json_path).name}")
    if imported > 0:
        print(f"  Próximo paso: registra los MIDIs de cada motivo con:")
        for name in list(bank["motifs"].keys())[-imported:]:
            print(f"    python leitmotif_tracker.py register motivo.mid --name \"{name}\"")

    return bank


def _auto_suggest_from_semantics(semantic_layer: dict) -> list[dict]:
    """Genera sugerencias automáticas de leitmotifs basadas en la semántica."""
    suggestions = []
    core = semantic_layer.get("core_emotion", "")
    secondary = semantic_layer.get("secondary_emotions", [])

    all_emotions = ([core] + secondary)[:3]  # máx 3 leitmotifs sugeridos

    tension_map = {
        "tristeza": "mid", "pérdida": "high", "alegría": "low",
        "esperanza": "mid", "misterio": "high", "calma": "low",
        "melancolía": "mid", "miedo": "high", "amor": "low",
    }

    for emotion in all_emotions:
        if not emotion:
            continue
        suggestions.append({
            "name":            emotion,
            "tags":            [emotion],
            "tension_affinity": tension_map.get(emotion.lower(), "mid"),
            "description":     f"Leitmotif sugerido para '{emotion}'",
            "preferred_instruments": _infer_instruments_from_tags([emotion]),
        })

    return suggestions


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRACIÓN CON HARVESTER
# ══════════════════════════════════════════════════════════════════════════════

def import_from_harvest(bank: dict,
                         midi_path: str,
                         name: str,
                         tags: list[str] = None,
                         verbose: bool = False) -> dict:
    """
    Importa un motivo extraído por harvester.py (archivos .motif_NN.mid).
    Es un atajo de 'register' pensado para flujo harvester → tracker.
    """
    print(f"  Importando motivo de harvester: {Path(midi_path).name}")
    register_motif(bank, name, midi_path, tags=tags, verbose=verbose)
    return bank


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRACIÓN CON STITCHER: leitmotif_schedule.json
# ══════════════════════════════════════════════════════════════════════════════

def export_schedule_for_stitcher(schedule: dict, output_path: str):
    """
    Exporta el schedule en formato compatible con stitcher.py.
    stitcher.py puede leer leitmotif_schedule.json para insertar
    fragmentos en los puntos de unión entre secciones.

    Formato del schedule para stitcher:
    {
      "leitmotif_injections": [
        { "bar": 4, "midi": "leitmotif_destino_bar004_presage.mid",
          "instrument": "cello", "dynamic": "pp" }
      ]
    }
    """
    approved = [inj for inj in schedule.get("injections", [])
                if inj.get("approved") and inj.get("midi_output")]

    stitcher_schedule = {
        "version":              VERSION,
        "generated":            datetime.now().isoformat(),
        "leitmotif_injections": [
            {
                "bar":        inj["bar"],
                "midi":       inj["midi_output"],
                "instrument": inj.get("instrument", "violin1"),
                "dynamic":    inj.get("dynamic", "mp"),
                "motif":      inj["motif"],
                "transform":  inj.get("transform", "full"),
            }
            for inj in approved
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stitcher_schedule, f, indent=2, ensure_ascii=False)

    print(f"  Schedule para stitcher.py exportado: {output_path}")
    print(f"  Uso: python stitcher.py *.fingerprint.json "
          f"--leitmotif-schedule {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  INTEGRACIÓN CON ORCHESTRATOR: preferred_instruments
# ══════════════════════════════════════════════════════════════════════════════

def export_orchestrator_hints(bank: dict, output_path: str):
    """
    Exporta un JSON con preferred_instruments de cada leitmotif,
    para que orchestrator.py asigne los timbres correctos al generar
    el MIDI multitracks.

    El orchestrator puede leer este archivo con --leitmotif-hints.
    """
    motifs = bank.get("motifs", {})
    hints = {
        "version": VERSION,
        "leitmotif_instrument_map": {
            name: {
                "preferred_instruments": m["semantics"].get("preferred_instruments", []),
                "tags":                  m["semantics"].get("tags", []),
                "tension_affinity":      m["semantics"].get("tension_affinity", "mid"),
            }
            for name, m in motifs.items()
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(hints, f, indent=2, ensure_ascii=False)

    print(f"  Hints para orchestrator.py exportados: {output_path}")
    print(f"  Uso: python orchestrator.py obra.mid --leitmotif-hints {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY: list y show
# ══════════════════════════════════════════════════════════════════════════════

def print_bank_list(bank: dict):
    """Lista todos los motivos del banco con información resumida."""
    motifs = bank.get("motifs", {})
    if not motifs:
        print("  El banco está vacío.")
        return

    print(f"\n{'═' * 62}")
    print(f"  LEITMOTIF TRACKER — Banco de motivos  ({len(motifs)} registrados)")
    print(f"{'═' * 62}\n")

    for name, motif in motifs.items():
        status = motif.get("status", "ok")
        midi   = motif.get("midi", "—")
        midi_name = Path(midi).name if midi else "— (pendiente)"
        n_apps = len(motif.get("appearances", []))
        tags   = ", ".join(motif["semantics"].get("tags", [])[:4])
        instrs = ", ".join(motif["semantics"].get("preferred_instruments", [])[:3])
        affin  = motif["semantics"].get("tension_affinity", "mid")
        fp     = motif.get("fingerprint", {})
        n_notes = fp.get("n_notes", 0)

        marker = "⚠" if status == "pending_midi" else "◆"
        print(f"  {marker} {name}")
        print(f"    MIDI:        {midi_name}")
        print(f"    Tags:        {tags or '—'}")
        print(f"    Instrumentos:{instrs or '—'}")
        print(f"    Afinidad:    {affin}  |  Notas: {n_notes}  |  Apariciones: {n_apps}")
        if motif.get("fingerprint", {}).get("contour"):
            c = motif["fingerprint"]["contour"][:6]
            print(f"    Contorno:    {c}...")
        print()


def print_motif_detail(bank: dict, name: str):
    """Muestra el detalle completo de un motivo."""
    motifs = bank.get("motifs", {})
    if name not in motifs:
        print(f"  Motivo '{name}' no encontrado.")
        return

    m  = motifs[name]
    fp = m.get("fingerprint", {})

    print(f"\n{'═' * 62}")
    print(f"  LEITMOTIF: {name}")
    print(f"{'═' * 62}")
    print(f"  MIDI:         {m.get('midi', '—')}")
    print(f"  Registrado:   {m.get('registered_at', '—')}")
    print(f"  Estado:       {m.get('status', 'ok')}")
    print(f"\n  Semántica:")
    sem = m.get("semantics", {})
    print(f"    Tags:        {', '.join(sem.get('tags', []))}")
    print(f"    Descripción: {sem.get('description', '—')}")
    print(f"    Afinidad:    {sem.get('tension_affinity', '—')}")
    print(f"    Instrumentos:{', '.join(sem.get('preferred_instruments', []))}")
    print(f"\n  Huella:")
    print(f"    Contorno:    {fp.get('contour', [])}")
    print(f"    Ritmo:       {fp.get('rhythm', [])}")
    print(f"    Notas:       {fp.get('n_notes', 0)}")
    print(f"    Pitch medio: {fp.get('mean_pitch', '—')}")
    print(f"    Duración:    {fp.get('duration_bars', '—')} compases")
    print(f"    Tempo:       {fp.get('tempo', '—')} BPM")

    appearances = m.get("appearances", [])
    if appearances:
        print(f"\n  Apariciones ({len(appearances)}):")
        for ap in appearances:
            print(f"    compás {ap.get('bar', '?'):3d} | {ap.get('section', '?'):<16} | "
                  f"{ap.get('transform', '?'):<15} | {ap.get('instrument', '?'):<12} | "
                  f"{ap.get('dynamic', '?')}")
    else:
        print(f"\n  Sin apariciones registradas.")

    print(f"\n{'═' * 62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _find_script(name: str) -> str | None:
    """Busca un script del pipeline en el mismo directorio o PATH."""
    here = Path(__file__).parent / name
    if here.exists():
        return str(here)
    cwd = Path.cwd() / name
    if cwd.exists():
        return str(cwd)
    return shutil.which(name)


def save_schedule(schedule: dict, path: str):
    """Guarda el schedule de inyecciones en JSON. Garantiza JSON valido siempre."""
    if not isinstance(schedule, dict):
        schedule = {"injections": [], "gaps": []}
    schedule.setdefault("injections", [])
    schedule.setdefault("gaps", [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)
    n = len(schedule.get("injections", []))
    print(f"  Schedule guardado: {path}  ({n} inyecciones)")


def load_schedule(path: str) -> dict:
    """Carga un schedule desde JSON. Tolerante con archivos vacios o corruptos."""
    _empty = {"injections": [], "gaps": [], "version": "1.0"}
    p = Path(path)
    if not p.exists():
        print(f"  [warn] Schedule no encontrado: {path}")
        return _empty
    try:
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  [warn] Schedule vacio: {path}")
            print(f"         Generalo con: python leitmotif_tracker.py plan obra_plan.json")
            return _empty
        return json.loads(content)
    except Exception as e:
        print(f"  [error] JSON invalido en {path}: {e}")
        print(f"          Regenera con: python leitmotif_tracker.py plan obra_plan.json")
        return _empty


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN / CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="leitmotif_tracker.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            LEITMOTIF TRACKER v{VERSION} — Gestión y siembra de leitmotifs en obras MIDI
            ─────────────────────────────────────────────────────────────────────
            Gestiona un banco de motivos musicales y los siembra estratégicamente
            a lo largo de una obra, integrándose con todo el ecosistema MIDI.

            Ejemplos:
              python leitmotif_tracker.py register motivo.mid --name "destino"
              python leitmotif_tracker.py list
              python leitmotif_tracker.py plan obra_plan.json --auto
              python leitmotif_tracker.py suggest seccion_B.mid
              python leitmotif_tracker.py inject obra.mid schedule.json
              python leitmotif_tracker.py trace obra_final.mid
              python leitmotif_tracker.py report obra_final.mid
              python leitmotif_tracker.py from-theorist plan.theorist.json
              python leitmotif_tracker.py propagate --instruction "más oscuro"
        """),
    )

    parser.add_argument("--bank", default=DEFAULT_BANK,
                        help=f"Banco de leitmotifs (default: {DEFAULT_BANK})")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # ── register ──────────────────────────────────────────────────────────────
    p_reg = sub.add_parser("register", help="Añadir un motivo al banco")
    p_reg.add_argument("midi", help="Archivo MIDI del motivo")
    p_reg.add_argument("--name", required=True, help="Nombre semántico del leitmotif")
    p_reg.add_argument("--tags", nargs="+", default=[],
                        help="Etiquetas semánticas (ej: pérdida tristeza)")
    p_reg.add_argument("--instruments", nargs="+", default=None,
                        help="Instrumentos preferidos (ej: cello viola)")
    p_reg.add_argument("--description", default="", help="Descripción libre")
    p_reg.add_argument("--tension-affinity", default="mid",
                        choices=["low", "mid", "high"],
                        help="Afinidad de tensión (default: mid)")

    # ── list ──────────────────────────────────────────────────────────────────
    sub.add_parser("list", help="Listar todos los motivos del banco")

    # ── show ──────────────────────────────────────────────────────────────────
    p_show = sub.add_parser("show", help="Detalle de un motivo")
    p_show.add_argument("name", help="Nombre del motivo")

    # ── plan ──────────────────────────────────────────────────────────────────
    p_plan = sub.add_parser("plan", help="Planificar siembra sobre obra_plan.json")
    p_plan.add_argument("narrator_plan", help="Archivo obra_plan.json de narrator")
    p_plan.add_argument("--motifs", nargs="+", default=None,
                         help="Motivos a sembrar (default: todos)")
    p_plan.add_argument("--auto", action="store_true",
                         help="Planificación automática sin confirmación")
    p_plan.add_argument("--output", default="leitmotif_schedule.json",
                         help="Archivo de salida (default: leitmotif_schedule.json)")

    # ── suggest ───────────────────────────────────────────────────────────────
    p_sug = sub.add_parser("suggest", help="Modo exploratorio: propone inyecciones")
    p_sug.add_argument("midi", help="MIDI de la sección a analizar")
    p_sug.add_argument("--output", default="leitmotif_schedule.json")

    # ── inject ────────────────────────────────────────────────────────────────
    p_inj = sub.add_parser("inject", help="Ejecutar inyecciones del schedule")
    p_inj.add_argument("midi", help="MIDI destino (informativo)")
    p_inj.add_argument("schedule", help="Schedule JSON (de plan o suggest)")
    p_inj.add_argument("--output-dir", default="leitmotifs_out",
                        help="Directorio de salida (default: leitmotifs_out)")
    p_inj.add_argument("--variation-engine", default=None,
                        help="Ruta a variation_engine.py")
    p_inj.add_argument("--export-stitcher", default=None,
                        help="Exportar schedule para stitcher.py")
    p_inj.add_argument("--export-orchestrator", default=None,
                        help="Exportar hints para orchestrator.py")

    # ── trace ─────────────────────────────────────────────────────────────────
    p_trace = sub.add_parser("trace", help="Detectar apariciones en obra existente")
    p_trace.add_argument("midi", help="MIDI de la obra completa")
    p_trace.add_argument("--threshold", type=float, default=0.55,
                          help="Umbral de similitud 0-1 (default: 0.55)")
    p_trace.add_argument("--output", default=None,
                          help="Guardar resultado en JSON")

    # ── report ────────────────────────────────────────────────────────────────
    p_rep = sub.add_parser("report", help="Mapa visual de apariciones")
    p_rep.add_argument("midi", help="MIDI de la obra completa")
    p_rep.add_argument("--threshold", type=float, default=0.55)
    p_rep.add_argument("--output", default=None, help="Guardar resultado en JSON")

    # ── propagate ─────────────────────────────────────────────────────────────
    p_prop = sub.add_parser("propagate",
                              help="Propagar mutación semántica a todos los motivos")
    p_prop.add_argument("--instruction", required=True,
                         help="Instrucción semántica (ej: 'más oscuro')")
    p_prop.add_argument("--motifs", nargs="+", default=None,
                         help="Motivos a mutar (default: todos)")
    p_prop.add_argument("--mutator", default=None,
                         help="Ruta a mutator.py")

    # ── from-theorist ─────────────────────────────────────────────────────────
    p_ft = sub.add_parser("from-theorist",
                            help="Importar sugerencias desde .theorist.json")
    p_ft.add_argument("theorist_json", help="Archivo .theorist.json")

    # ── from-harvest ──────────────────────────────────────────────────────────
    p_fh = sub.add_parser("from-harvest",
                            help="Importar motivo extraído por harvester.py")
    p_fh.add_argument("midi", help="Archivo .motif_NN.mid de harvester")
    p_fh.add_argument("--name", required=True, help="Nombre del leitmotif")
    p_fh.add_argument("--tags", nargs="+", default=[])

    args = parser.parse_args()

    # Cargar banco
    bank = load_bank(args.bank)

    # ── Despacho de comandos ──────────────────────────────────────────────────

    if args.command == "register":
        if not Path(args.midi).exists():
            print(f"ERROR: {args.midi} no existe")
            sys.exit(1)
        register_motif(bank, args.name, args.midi,
                       tags=args.tags,
                       preferred_instruments=args.instruments,
                       description=args.description,
                       tension_affinity=args.tension_affinity,
                       verbose=args.verbose)
        save_bank(bank, args.bank)
        print(f"  Banco guardado: {args.bank}")

    elif args.command == "list":
        print_bank_list(bank)

    elif args.command == "show":
        print_motif_detail(bank, args.name)

    elif args.command == "plan":
        if not Path(args.narrator_plan).exists():
            print(f"ERROR: {args.narrator_plan} no existe")
            sys.exit(1)
        with open(args.narrator_plan, "r", encoding="utf-8") as f:
            narrator_plan = json.load(f)

        schedule = plan_seeding(bank, narrator_plan,
                                 motif_names=args.motifs,
                                 auto=args.auto,
                                 verbose=args.verbose)

        # Modo exploratorio: mostrar propuestas y pedir confirmación
        if not args.auto:
            injections = schedule.get("injections", [])
            print(f"\n{'═' * 62}")
            print(f"  Plan de siembra — {len(injections)} propuestas")
            print(f"{'═' * 62}\n")

            approved_injections = []
            for i, inj in enumerate(injections, 1):
                print(f"  [{i:2d}] {inj['reason']}")
                print(f"       Dinámica: {inj['dynamic']}  |  Fit: {inj['fit_score']:.2f}")
                try:
                    r = input("       [s=aprobar / n=saltar / enter=sí] > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    break
                if r in ("s", "", "si", "sí", "y", "yes"):
                    inj["approved"] = True
                    approved_injections.append(inj)
                    print("       ✓ Aprobada")
                else:
                    print("       — Saltada")

            schedule["injections"] = approved_injections
            n_approved = len(approved_injections)
            print(f"\n  {n_approved} inyecciones aprobadas de {len(injections)} propuestas.")
        else:
            # Auto: aprobar todas
            for inj in schedule["injections"]:
                inj["approved"] = True

        if schedule.get("gaps"):
            print(f"\n  Huecos dramáticos detectados ({len(schedule['gaps'])}):")
            for g in schedule["gaps"]:
                print(f"    compases {g['bar_start']}-{g['bar_end']}: "
                      f"tensión {g['tension']:.2f} — {g['suggestion']}")

        save_schedule(schedule, args.output)

    elif args.command == "suggest":
        if not Path(args.midi).exists():
            print(f"ERROR: {args.midi} no existe")
            sys.exit(1)
        schedule = suggest_interactive(bank, args.midi, verbose=args.verbose)
        save_schedule(schedule, args.output)

    elif args.command == "inject":
        if not Path(args.schedule).exists():
            print(f"ERROR: schedule {args.schedule} no existe")
            sys.exit(1)
        schedule = load_schedule(args.schedule)
        schedule = inject_motifs(bank, schedule,
                                  output_dir=args.output_dir,
                                  variation_engine=args.variation_engine,
                                  verbose=args.verbose)
        save_bank(bank, args.bank)
        save_schedule(schedule, args.schedule)

        if args.export_stitcher:
            export_schedule_for_stitcher(schedule, args.export_stitcher)
        if args.export_orchestrator:
            export_orchestrator_hints(bank, args.export_orchestrator)

    elif args.command == "trace":
        if not Path(args.midi).exists():
            print(f"ERROR: {args.midi} no existe")
            sys.exit(1)
        result = trace_appearances(bank, args.midi,
                                    threshold=args.threshold,
                                    verbose=args.verbose)
        print(f"\n  Apariciones detectadas: {len(result['detections'])}")
        for d in result["detections"]:
            print(f"    compás {d['bar']:3d}: '{d['motif']}' "
                  f"(similitud {d['similarity']:.2f})")
        if result["gaps"]:
            print(f"  Huecos: {len(result['gaps'])} tramos sin leitmotif")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Resultado guardado: {args.output}")

    elif args.command == "report":
        if not Path(args.midi).exists():
            print(f"ERROR: {args.midi} no existe")
            sys.exit(1)
        result = trace_appearances(bank, args.midi,
                                    threshold=args.threshold,
                                    verbose=args.verbose)
        print_report(result, bank)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Resultado guardado: {args.output}")

    elif args.command == "propagate":
        bank = propagate_mutation(bank,
                                   instruction=args.instruction,
                                   motif_names=args.motifs,
                                   mutator_script=args.mutator,
                                   verbose=args.verbose)
        save_bank(bank, args.bank)
        print(f"  Banco actualizado: {args.bank}")

    elif args.command == "from-theorist":
        if not Path(args.theorist_json).exists():
            print(f"ERROR: {args.theorist_json} no existe")
            sys.exit(1)
        bank = import_from_theorist(bank, args.theorist_json, verbose=args.verbose)
        save_bank(bank, args.bank)

    elif args.command == "from-harvest":
        if not Path(args.midi).exists():
            print(f"ERROR: {args.midi} no existe")
            sys.exit(1)
        bank = import_from_harvest(bank, args.midi, args.name,
                                    tags=args.tags, verbose=args.verbose)
        save_bank(bank, args.bank)
        print(f"  Banco guardado: {args.bank}")


if __name__ == "__main__":
    main()
