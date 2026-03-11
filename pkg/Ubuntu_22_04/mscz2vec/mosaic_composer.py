#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     MOSAIC COMPOSER  v2.0                                    ║
║         Collage y ensamblaje dramático de fragmentos musicales               ║
║                                                                              ║
║  Segmenta un corpus de MIDIs en fragmentos etiquetados, los indexa por      ║
║  función dramática, tensión, estilo y compatibilidad de costura, y los      ║
║  reensambla creativamente siguiendo un arco narrativo explícito.             ║
║                                                                              ║
║  El material resultante no es una concatenación: cada sección se genera     ║
║  fusionando el ADN de dos fragmentos distintos (melodía de uno, ritmo del   ║
║  otro) y aplicando transformaciones diatónicas que hacen irreconocibles     ║
║  las fuentes originales.                                                     ║
║                                                                              ║
║  PIPELINE INTERNO:                                                           ║
║  [1] HARVEST   — Segmenta corpus con harvester.py (o internamente)          ║
║  [2] INDEX     — Etiqueta cada fragmento: tensión, rol, cadencia, estilo    ║
║  [3] PLAN      — Diseña el arco dramático (slots con función y tensión)     ║
║  [4] SELECT    — Para cada slot elige DOS fragmentos de fuentes distintas   ║
║  [5] BLEND     — Fusiona sus ADNs (deidentify + graft + midi_dna_unified)   ║
║  [6] STITCH    — Une los slots con puentes morpher o transiciones tonales   ║
║  [7] POLISH    — Reharmoniza y/o unifica estilo con style_transfer          ║
║                                                                              ║
║  MODOS DE OPERACIÓN:                                                         ║
║    build       — Pipeline completo [1]→[7] desde corpus hasta MIDI final   ║
║    harvest     — Sólo [1+2]: segmentar corpus y exportar banco etiquetado   ║
║    plan        — Sólo [3]: diseñar y exportar arco dramático (JSON)         ║
║    assemble    — [4]→[7]: ensamblar desde banco+plan ya existentes          ║
║    inspect     — Analizar el banco de fragmentos                            ║
║    stitch      — Coser una lista manual de MIDIs con transiciones           ║
║                                                                              ║
║  ARCOS DRAMÁTICOS (--arc):                                                   ║
║    sonata      — Exposición A → Exposición B → Desarrollo → Recapitulación ║
║    rondo       — A-B-A-C-A-D-A (refrán + episodios alternados)             ║
║    arch        — Tensión creciente → Clímax → Descenso espejo (default)    ║
║    journey     — Llamada → Partida → Prueba → Noche oscura →               ║
║                  Clímax → Regreso → Retorno                                 ║
║    wave        — Ciclos tensión/reposo (ambient, minimalismo)               ║
║    mosaic      — Collage puro por contraste máximo entre bloques            ║
║                                                                              ║
║  ESTRATEGIAS DE COSTURA (--stitch):                                          ║
║    auto        — Detecta la técnica óptima por compatibilidad tonal        ║
║    bridge      — Puente de morphing con morpher.py (más orgánico)          ║
║    pivot       — Transposición al acorde pivote entre tonalidades           ║
║    silence     — Silencio de 1 compás entre secciones                      ║
║    overlap     — Superposición temporal de compases finales/iniciales       ║
║    direct      — Corte directo sin transición                               ║
║                                                                              ║
║  TRANSFORMACIONES DE BLEND (automáticas, por función del slot):             ║
║    introduction/coda     — Aumentación ×1.5 + transporte ±3-4 grados      ║
║    theme_a/recapitulation— Inversión diatónica + transporte ±2-3 grados   ║
║    development           — Retrógrado + inversión + transporte ±4-5 grados ║
║    climax                — Diminución ×0.65 + inversión + octava alta      ║
║    theme_b/episode       — Permutación de frases + transporte ±3-4 grados  ║
║                                                                              ║
║  INTEGRACIONES CON EL ECOSISTEMA:                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │  harvester.py      → [1] segmentación y etiquetado del corpus        │   ║
║  │  midi_dna_unified  → [5] blend DNA entre pares de fragmentos         │   ║
║  │  morpher.py        → [6] puentes de transición entre slots           │   ║
║  │  variation_engine  → [4] variantes si no hay fragmento adecuado      │   ║
║  │                         o en harvest --variations para enriquecer     │   ║
║  │  reharmonizer.py   → [7] reharmonización diatónica del resultado     │   ║
║  │  style_transfer.py → [7] unificar timbre/textura de la obra final    │   ║
║  │  phrase_builder.py → [6] costura sintáctica ant/cons entre frases    │   ║
║  │  completer.py      → [4] rellenar huecos no cubiertos por el banco   │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EJEMPLOS DE USO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

── BUILD: pipeline completo en un solo comando ──────────────────────────────

  # Mínimo — detecta tonalidad y tempo del corpus
  python mosaic_composer.py build \\
      --corpus midis/ --arc arch --bars 64 --output obra.mid

  # Completo con todas las herramientas activas
  python mosaic_composer.py build \\
      --corpus midis/ \\
      --arc journey \\
      --bars 64 \\
      --key "D minor" \\
      --stitch bridge --bridge-bars 2 \\
      --candidates 10 --diversity 0.7 \\
      --allow-variations \\
      --variations V01 V04 V10 \\
      --reharmonize \\
      --style-unify midis/referencia.mid \\
      --no-percussion \\
      --out-dir salida/ --output obra_final.mid \\
      --report --verbose --seed 42

  # Arcos disponibles: arch | journey | sonata | rondo | wave | mosaic
  python mosaic_composer.py build --corpus midis/ --arc sonata --bars 80
  python mosaic_composer.py build --corpus midis/ --arc rondo  --bars 96
  python mosaic_composer.py build --corpus midis/ --arc wave   --bars 48

── HARVEST: segmentar corpus y enriquecer banco con variaciones ─────────────

  # Segmentación básica (2-5 compases por fragmento)
  python mosaic_composer.py harvest \\
      --corpus midis/ --out-dir banco/ --min-bars 2 --max-bars 5

  # Con variaciones específicas (multiplica el banco x4)
  python mosaic_composer.py harvest \\
      --corpus midis/ --out-dir banco/ \\
      --min-bars 2 --max-bars 8 \\
      --variations V01 V04 V10     # inversión, aumentación, variación rítmica

  # Con todas las variaciones disponibles
  python mosaic_composer.py harvest \\
      --corpus midis/ --out-dir banco/ --variations all

  # Variaciones disponibles:
  #   V01 inversión melódica      → útil para development, climax
  #   V02 retrógrado              → útil para development
  #   V03 inversión retrógrada    → útil para development
  #   V04 aumentación rítmica     → útil para coda, introduction
  #   V05 diminución rítmica      → útil para climax, episode
  #   V06 ornamentación           → útil para theme_a, theme_b
  #   V07 cambio de acompañamiento→ cualquier función
  #   V08 transposición           → bridge, recapitulation
  #   V10 variación rítmica       → episode, bridge
  #   V14 estocástica             → diversidad general

── PLAN: diseñar y exportar el arco dramático ───────────────────────────────

  # Exportar plan para editarlo manualmente antes de ensamblar
  python mosaic_composer.py plan \\
      --arc journey --bars 48 --key "A minor" --out-dir salida/

  # Usar --stdout para encadenar con assemble via sustitución de proceso
  python mosaic_composer.py assemble \\
      --bank banco/.mosaic_bank.json \\
      --plan <(python mosaic_composer.py plan --arc arch --bars 32 --stdout) \\
      --output obra.mid

── ASSEMBLE: ensamblar desde banco y plan existentes ────────────────────────

  # Básico
  python mosaic_composer.py assemble \\
      --bank banco/.mosaic_bank.json --plan plan.json --output obra.mid

  # Con todas las herramientas activas
  python mosaic_composer.py assemble \\
      --bank banco/.mosaic_bank.json \\
      --plan plan.json \\
      --stitch bridge --bridge-bars 2 \\
      --candidates 10 --diversity 0.7 \\
      --allow-variations \\
      --reharmonize \\
      --style-unify midis/referencia.mid \\
      --no-percussion \\
      --out-dir salida/ --output obra.mid \\
      --verbose --report --seed 42

  # Generar múltiples variaciones con distintas semillas (una por seed)
  python mosaic_composer.py assemble \\
      --bank banco/.mosaic_bank.json --plan plan.json \\
      --stitch bridge --candidates 10 --diversity 0.7 \\
      --allow-variations --reharmonize --no-percussion \\
      --output viaje.mid \\
      --seeds 42 123 999 777 555
  # → produce: viaje_s000042.mid  viaje_s000123.mid  viaje_s000999.mid ...

  # Costura bridge máxima diversidad
  python mosaic_composer.py assemble \\
      --bank banco/.mosaic_bank.json --plan plan.json \\
      --stitch bridge --bridge-bars 4 \\
      --candidates 15 --diversity 0.9 \\
      --allow-variations --fill-gaps \\
      --output obra_maxima_diversidad.mid

── INDEX: indexar / re-indexar un directorio de MIDIs ───────────────────────

  # Indexar banco/bank/ y crear (o sobreescribir) el banco
  python mosaic_composer.py index banco/bank/ \
      --bank banco/.mosaic_bank.json

  # Re-indexar añadiendo MIDIs nuevos sin perder los ya indexados
  python mosaic_composer.py index banco/bank/ \
      --bank banco/.mosaic_bank.json --merge

  # Indexar en subdirectorios (corpus jerárquico) con detalle por fragmento
  python mosaic_composer.py index banco/bank/ \
      --bank banco/.mosaic_bank.json --recursive --verbose

  # Fusionar dos directorios en un banco único
  python mosaic_composer.py index banco_a/bank/ \
      --bank banco_combinado/.mosaic_bank.json
  python mosaic_composer.py index banco_b/bank/ \
      --bank banco_combinado/.mosaic_bank.json --merge

  # Qué calcula por cada MIDI:
  #   tonalidad  →  detect_key_simple()
  #   tempo      →  get_tempo()
  #   compases   →  count_bars()
  #   tensión    →  estimate_tension()  → tension_mean, tension_end
  #   rol        →  infer_role()        → motif|antecedent|consequent|cadence|…
  #   cadencia   →  heurística sobre tension_end  → AC|IAC|HC|DC|none
  #   estilo     →  detección por nombre de directorio/archivo

── INSPECT: analizar el banco de fragmentos ─────────────────────────────────

  # Vista general: lista todos los fragmentos ordenados por tensión,
  # con roles, cadencias, tonalidades y estadísticas de cobertura.
  # Útil para comprobar si el banco tiene suficiente diversidad de tensión
  # antes de lanzar assemble, o detectar fuentes sobrerepresentadas.
  python mosaic_composer.py inspect --bank banco/.mosaic_bank.json

  # Salida típica:
  #   ID                                Bars  Key          T_mean  Role         Cad
  #   ────────────────────────────────────────────────────────────────────────────
  #   Bach_Konzert.phrase_01              4   C major       0.23   antecedent   HC
  #   ABBA_Alaska.frag_B                  3   A minor       0.45   phrase       none
  #   1-10-Fighting-Xg.climax_01          6   D minor       0.87   climax       AC
  #   ...
  #   Roles:   {'phrase': 12, 'cadence': 8, 'climax': 3, 'motif': 5}
  #   Estilos: {'classical': 7, 'pop': 6, 'unknown': 9}
  #   Tensión: min=0.12  max=0.91  media=0.54

── STITCH: coser una lista manual de MIDIs ──────────────────────────────────

  python mosaic_composer.py stitch \\
      intro.mid desarrollo.mid climax.mid coda.mid \\
      --stitch auto --output obra_manual.mid

  python mosaic_composer.py stitch \\
      frag_a.mid frag_b.mid frag_c.mid \\
      --stitch bridge --bridge-bars 2 --key "G minor" --output costura.mid

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FLUJOS RECOMENDADOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # A) Exploración rápida: un comando, múltiples variaciones
  python mosaic_composer.py build \\
      --corpus midis/ --arc arch --bars 48 --output prueba.mid --seed 42
  # luego probar semillas distintas: --seed 123, --seed 999, etc.

  # B) Control total: harvest → plan → múltiples assemble
  python mosaic_composer.py harvest \\
      --corpus midis/ --out-dir banco/ --variations V01 V04 V10

  python mosaic_composer.py plan --arc journey --bars 64 --out-dir banco/
  # (editar banco/mosaic_plan.json si se desea ajustar tensiones/roles)

  python mosaic_composer.py assemble \\
      --bank banco/.mosaic_bank.json \\
      --plan banco/mosaic_plan.json \\
      --stitch bridge --candidates 10 --diversity 0.7 \\
      --allow-variations --reharmonize --no-percussion \\
      --output obra.mid --seeds 1 2 3 4 5

  # C) Rehacer solo el ensamblaje con otro arco (reutilizar banco)
  python mosaic_composer.py assemble \\
      --bank banco/.mosaic_bank.json \\
      --plan <(python mosaic_composer.py plan --arc sonata --bars 80 --stdout) \\
      --stitch auto --reharmonize --output sonata.mid

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DEPENDENCIAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Requeridas:   mido  numpy
  Opcionales:   scipy  music21
  Ecosistema:   harvester.py  midi_dna_unified.py  morpher.py
                reharmonizer.py  style_transfer.py  variation_engine.py
                phrase_builder.py  completer.py
                (todos deben estar en el mismo directorio que mosaic_composer.py)
"""

import os
import sys
import json
import argparse
import random
import tempfile
import traceback
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# ─── Importaciones opcionales del ecosistema ───────────────────────────────
# Todos los scripts del ecosistema se invocan via subprocess — sin imports directos.


# ═══════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Fragment:
    """Un fragmento del banco con todas sus etiquetas."""
    id: str
    file: str
    bars: int = 4
    key: str = "C major"
    tempo: float = 120.0
    tension_mean: float = 0.5
    tension_end: float = 0.5
    arousal: float = 0.5
    role: str = "phrase"          # motif|antecedent|consequent|cadence|bridge|climax|coda
    cadence_end: str = "none"     # AC|IAC|HC|DC|PC|none
    cadence_start: str = "none"
    style: str = "unknown"
    source: str = ""
    fingerprint: Dict = field(default_factory=dict)
    score: float = 0.0            # calidad musical (score_candidate)


@dataclass
class Slot:
    """Un hueco en el plan dramático que debe llenarse con un fragmento."""
    id: str
    label: str
    function: str                         # introduction|theme_a|theme_b|development|
                                          # recapitulation|climax|coda|episode|bridge
    bars: int = 8
    tension_target: Tuple[float, float] = (0.3, 0.6)
    role_required: List[str] = field(default_factory=list)
    cadence_end: str = "any"
    style_hint: Optional[str] = None
    stitch_next: str = "auto"
    assigned_fragment: Optional[str] = None   # id del fragmento primario (ADN melódico)
    assigned_file: Optional[str] = None
    assigned_fragment_b: Optional[str] = None # id del fragmento secundario (ADN rítmico/armónico)
    assigned_file_b: Optional[str] = None


@dataclass
class MosaicPlan:
    arc: str = "arch"
    bars_total: int = 64
    key: str = "C major"
    tempo: float = 120.0
    slots: List[Slot] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# ARCOS DRAMÁTICOS
# ═══════════════════════════════════════════════════════════════════════════

ARC_TEMPLATES = {

    "sonata": [
        # label, function, bars_frac, tension_range, roles, cadence_end
        ("Exposición A",      "theme_a",         0.15, (0.2, 0.45), ["antecedent","motif"], "HC"),
        ("Exposición B",      "theme_b",         0.15, (0.3, 0.55), ["consequent","phrase"], "AC"),
        ("Transición",        "bridge",           0.10, (0.4, 0.65), ["bridge","phrase"],    "HC"),
        ("Desarrollo I",      "development",      0.15, (0.5, 0.80), ["phrase","motif"],     "DC"),
        ("Desarrollo II",     "development",      0.15, (0.6, 0.95), ["climax","phrase"],    "HC"),
        ("Recapitulación A",  "recapitulation",   0.15, (0.3, 0.55), ["antecedent","motif"], "AC"),
        ("Coda",              "coda",             0.15, (0.1, 0.30), ["coda","consequent"],  "AC"),
    ],

    "rondo": [
        ("Refrán A",          "theme_a",   0.18, (0.2, 0.50), ["antecedent","motif"],    "AC"),
        ("Episodio B",        "episode",   0.14, (0.4, 0.70), ["phrase","bridge"],        "HC"),
        ("Refrán A'",         "theme_a",   0.12, (0.2, 0.45), ["antecedent","motif"],    "AC"),
        ("Episodio C",        "episode",   0.14, (0.5, 0.80), ["climax","phrase"],        "DC"),
        ("Refrán A''",        "theme_a",   0.12, (0.2, 0.45), ["antecedent","motif"],    "AC"),
        ("Episodio D",        "episode",   0.16, (0.6, 0.90), ["climax","phrase"],        "HC"),
        ("Refrán A (coda)",   "coda",      0.14, (0.1, 0.30), ["coda","consequent"],     "AC"),
    ],

    "arch": [
        ("Inicio",            "introduction", 0.15, (0.1, 0.35), ["motif","antecedent"], "HC"),
        ("Ascenso I",         "development",  0.15, (0.3, 0.55), ["phrase","bridge"],    "HC"),
        ("Ascenso II",        "development",  0.15, (0.5, 0.75), ["phrase","climax"],    "DC"),
        ("Clímax",            "climax",       0.15, (0.7, 1.00), ["climax"],             "DC"),
        ("Descenso I",        "development",  0.15, (0.5, 0.75), ["phrase","bridge"],    "HC"),
        ("Descenso II",       "recapitulation",0.12,(0.3, 0.55), ["phrase","antecedent"],"AC"),
        ("Desenlace",         "coda",         0.13, (0.1, 0.30), ["coda","consequent"],  "AC"),
    ],

    "journey": [
        ("Llamada",           "introduction", 0.12, (0.1, 0.35), ["motif"],              "HC"),
        ("Partida",           "theme_a",      0.15, (0.3, 0.55), ["antecedent","phrase"],"HC"),
        ("Primera prueba",    "development",  0.15, (0.5, 0.75), ["phrase","bridge"],    "DC"),
        ("Noche oscura",      "development",  0.13, (0.6, 0.90), ["climax","phrase"],    "DC"),
        ("Clímax / Victoria", "climax",       0.15, (0.75,1.00), ["climax"],             "DC"),
        ("Regreso",           "recapitulation",0.15,(0.25,0.50), ["consequent","phrase"],"AC"),
        ("Retorno al hogar",  "coda",         0.15, (0.05,0.25), ["coda"],               "AC"),
    ],

    "wave": [
        ("Ola 1 — subida",    "development",  0.17, (0.2, 0.65), ["phrase","motif"],     "HC"),
        ("Ola 1 — bajada",    "theme_a",      0.17, (0.1, 0.35), ["consequent","coda"],  "AC"),
        ("Ola 2 — subida",    "development",  0.17, (0.35,0.80), ["phrase","climax"],    "HC"),
        ("Ola 2 — bajada",    "recapitulation",0.16,(0.1, 0.35), ["consequent"],         "AC"),
        ("Ola 3 — clímax",    "climax",       0.17, (0.6, 1.00), ["climax"],             "DC"),
        ("Disolución",        "coda",         0.16, (0.05,0.20), ["coda"],               "AC"),
    ],

    "mosaic": [
        # Collage puro: máximo contraste entre fragmentos consecutivos
        ("Bloque 1",  "episode", 0.17, (0.1, 1.0), [], "any"),
        ("Bloque 2",  "episode", 0.17, (0.1, 1.0), [], "any"),
        ("Bloque 3",  "episode", 0.17, (0.1, 1.0), [], "any"),
        ("Bloque 4",  "episode", 0.17, (0.1, 1.0), [], "any"),
        ("Bloque 5",  "episode", 0.16, (0.1, 1.0), [], "any"),
        ("Bloque 6",  "episode", 0.16, (0.1, 1.0), [], "any"),
    ],
}


def build_plan(arc: str, bars_total: int, key: str, tempo: float) -> MosaicPlan:
    """Construye un MosaicPlan a partir de una plantilla de arco."""
    template = ARC_TEMPLATES.get(arc, ARC_TEMPLATES["arch"])
    plan = MosaicPlan(arc=arc, bars_total=bars_total, key=key, tempo=tempo)

    for i, (label, function, frac, tension_range, roles, cadence_end) in enumerate(template):
        slot_bars = max(2, round(bars_total * frac))
        slot = Slot(
            id=f"slot_{i+1:02d}",
            label=label,
            function=function,
            bars=slot_bars,
            tension_target=tension_range,
            role_required=roles,
            cadence_end=cadence_end,
            stitch_next="auto",
        )
        plan.slots.append(slot)

    # Ajustar total de barras por redondeo
    current = sum(s.bars for s in plan.slots)
    if current != bars_total:
        plan.slots[-1].bars += bars_total - current

    return plan


# ═══════════════════════════════════════════════════════════════════════════
# INDEXACIÓN DE FRAGMENTOS
# ═══════════════════════════════════════════════════════════════════════════

def detect_key_simple(mid: MidiFile) -> str:
    """Detección de tonalidad simple por distribución de alturas (Krumhansl)."""
    pitch_counts = np.zeros(12)
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                pitch_counts[msg.note % 12] += 1
    if pitch_counts.sum() == 0:
        return "C major"
    # Perfiles de Krumhansl
    major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    best_key, best_score, best_mode = 0, -np.inf, "major"
    names = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    for root in range(12):
        rolled_counts = np.roll(pitch_counts, -root)
        rolled_counts = rolled_counts / (rolled_counts.sum() + 1e-9)
        maj_score = np.corrcoef(rolled_counts, major_profile / major_profile.sum())[0,1]
        min_score = np.corrcoef(rolled_counts, minor_profile / minor_profile.sum())[0,1]
        if maj_score > best_score:
            best_score, best_key, best_mode = maj_score, root, "major"
        if min_score > best_score:
            best_score, best_key, best_mode = min_score, root, "minor"
    return f"{names[best_key]} {best_mode}"


def estimate_tension(mid: MidiFile) -> Tuple[float, float]:
    """Estima tensión media y final a partir de densidad y cromatismo."""
    events = []
    for track in mid.tracks:
        t = 0
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                events.append((t, msg.note))
    if not events:
        return 0.5, 0.5
    events.sort()
    total_time = events[-1][0] + 1
    window = total_time // 4
    # Tensión en la última cuarta parte
    last_events = [n for t, n in events if t >= 3 * window]
    all_pitches = [n for _, n in events]
    last_pitches = last_events if last_events else all_pitches[-4:]
    chromatic_ratio = len(set(p % 12 for p in all_pitches)) / 12.0
    density = min(1.0, len(events) / max(1, total_time / 480) / 8)
    tension_mean = 0.4 * chromatic_ratio + 0.6 * density
    chromatic_end = len(set(p % 12 for p in last_pitches)) / 12.0 if last_pitches else chromatic_ratio
    tension_end = 0.4 * chromatic_end + 0.6 * density
    return float(np.clip(tension_mean, 0, 1)), float(np.clip(tension_end, 0, 1))


def count_bars(mid: MidiFile) -> int:
    """Cuenta compases aproximados."""
    ticks_per_beat = mid.ticks_per_beat or 480
    total_ticks = max(
        sum(msg.time for msg in track) for track in mid.tracks
    )
    beats = total_ticks / ticks_per_beat
    return max(1, round(beats / 4))


def get_tempo(mid: MidiFile) -> float:
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return round(60_000_000 / msg.tempo, 1)
    return 120.0


def infer_role(fragment_id: str, tension_mean: float, tension_end: float, bars: int) -> str:
    """Infiere el rol dramático del fragmento a partir de sus atributos."""
    fname = fragment_id.lower()
    if "motif" in fname:
        return "motif"
    if "cadence_auth" in fname:
        return "consequent"
    if "cadence_half" in fname:
        return "antecedent"
    if "coda" in fname or (tension_mean < 0.3 and tension_end < 0.3):
        return "coda"
    if tension_mean > 0.75:
        return "climax"
    if "phrase_01" in fname or "frag_a" in fname:
        return "antecedent"
    if tension_end < tension_mean - 0.15:
        return "consequent"
    if bars <= 2:
        return "motif"
    if bars >= 12:
        return "climax"
    return "phrase"


def index_fragment(mid_path: str) -> Fragment:
    """Crea un Fragment etiquetado desde un archivo MIDI."""
    mid = MidiFile(mid_path)
    frag_id = Path(mid_path).stem
    key = detect_key_simple(mid)
    tempo = get_tempo(mid)
    bars = count_bars(mid)
    tension_mean, tension_end = estimate_tension(mid)
    role = infer_role(frag_id, tension_mean, tension_end, bars)

    # Cadencia final heurística
    cadence_end = "none"
    if tension_end < 0.25:
        cadence_end = "AC"
    elif tension_end < 0.45:
        cadence_end = "IAC"
    elif tension_end < 0.65:
        cadence_end = "HC"
    elif tension_end < 0.80:
        cadence_end = "DC"

    # Estilo aproximado por corpus de origen
    style = "unknown"
    parent = Path(mid_path).parent.name.lower()
    for tag in ["bach","baroque","mozart","classical","jazz","pop","flamenco","latin","blues","romantic"]:
        if tag in parent or tag in frag_id.lower():
            style = tag
            break

    return Fragment(
        id=frag_id,
        file=str(mid_path),
        bars=bars,
        key=key,
        tempo=tempo,
        tension_mean=tension_mean,
        tension_end=tension_end,
        arousal=tension_mean * 0.8 + (bars / 16) * 0.2,
        role=role,
        cadence_end=cadence_end,
        cadence_start="none",
        style=style,
        source=str(Path(mid_path).parent),
    )


# ═══════════════════════════════════════════════════════════════════════════
# SELECCIÓN DE FRAGMENTOS POR SLOT
# ═══════════════════════════════════════════════════════════════════════════

def fragment_slot_score(frag: Fragment, slot: Slot) -> float:
    """
    Puntúa la adecuación de un fragmento para un slot.
    Retorna un valor en [0, 1], mayor = mejor.
    """
    score = 0.0

    # Tensión dentro del rango objetivo
    t_lo, t_hi = slot.tension_target
    if t_lo <= frag.tension_mean <= t_hi:
        score += 0.35
    else:
        dist = min(abs(frag.tension_mean - t_lo), abs(frag.tension_mean - t_hi))
        score += 0.35 * max(0, 1 - dist * 3)

    # Rol requerido
    if slot.role_required:
        if frag.role in slot.role_required:
            score += 0.30
        elif any(r in frag.id.lower() for r in slot.role_required):
            score += 0.15
    else:
        score += 0.30  # sin restricción de rol

    # Cadencia de cierre
    if slot.cadence_end == "any" or slot.cadence_end == frag.cadence_end:
        score += 0.20
    elif slot.cadence_end in ("AC","IAC") and frag.cadence_end in ("AC","IAC"):
        score += 0.10

    # Estilo
    if slot.style_hint is None or slot.style_hint == frag.style:
        score += 0.15
    elif frag.style == "unknown":
        score += 0.07

    return float(np.clip(score, 0.0, 1.0))


def select_fragments(
    bank: List[Fragment],
    plan: MosaicPlan,
    n_candidates: int = 5,
    diversity: float = 0.4,
    allow_variations: bool = False,
    variation_out_dir: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
    args_ref=None,
) -> MosaicPlan:
    """
    Asigna DOS fragmentos por slot (de fuentes distintas si es posible).
    El slot guarda assigned_fragment (primario) y assigned_fragment_b (secundario)
    para que blend_slot() los fusione con midi_dna_unified en lugar de
    limitarse a concatenarlos.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    used_ids: Dict[str, int] = {}

    for slot in plan.slots:
        scored = []
        for frag in bank:
            base_score = fragment_slot_score(frag, slot)
            times_used = used_ids.get(frag.id, 0)
            final_score = base_score - diversity * times_used
            scored.append((final_score, frag))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored or scored[0][0] < 0.05:
            if verbose:
                print(f"[WARN] {slot.id} ({slot.label}): banco insuficiente.")

            # Intentar completar el mejor fragmento disponible con completer.py
            fill_gaps = getattr(args_ref, "fill_gaps", False) if args_ref else False
            if fill_gaps and scored:
                best_frag = scored[0][1]
                corpus_dir = getattr(args_ref, "corpus", None) if args_ref else None
                completed  = _complete_fragment(
                    best_frag, slot,
                    corpus_dir=corpus_dir or "",
                    out_dir=variation_out_dir or ".",
                    verbose=verbose,
                )
                if completed is not best_frag:
                    bank.append(completed)
                    scored.insert(0, (0.6, completed))

            var_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "variation_engine.py")
            if allow_variations and os.path.exists(var_script) and scored:
                best_frag = scored[0][1]
                try:
                    var_path = _generate_variation_for_slot(best_frag, slot, variation_out_dir)
                    var_frag = index_fragment(var_path)
                    var_frag.id = f"{best_frag.id}_var_{slot.id}"
                    var_frag.file = var_path
                    bank.append(var_frag)
                    scored.insert(0, (0.5, var_frag))
                    if verbose:
                        print(f"       → Variación generada: {var_path}")
                except Exception as e:
                    if verbose:
                        print(f"       [ERROR] variation_engine: {e}")
            if not scored:
                continue

        top = scored[:n_candidates]

        # ── Candidato A: mejor puntuación con temperatura baja ──
        scores_arr = np.array([s for s, _ in top])
        scores_arr = np.exp(scores_arr * 5)
        probs = scores_arr / scores_arr.sum()
        idx_a = rng.choice(len(top), p=probs)
        _, frag_a = top[idx_a]

        # ── Candidato B: mejor puntuación de una FUENTE distinta a frag_a ──
        frag_b = None
        for _, f in scored:
            if f.id != frag_a.id and f.source != frag_a.source:
                frag_b = f
                break
        # Si no hay fuente distinta, tomamos simplemente el segundo mejor
        if frag_b is None:
            for _, f in scored:
                if f.id != frag_a.id:
                    frag_b = f
                    break

        slot.assigned_fragment  = frag_a.id
        slot.assigned_file      = frag_a.file
        slot.assigned_fragment_b = frag_b.id   if frag_b else None
        slot.assigned_file_b     = frag_b.file if frag_b else None

        used_ids[frag_a.id] = used_ids.get(frag_a.id, 0) + 1
        if frag_b:
            used_ids[frag_b.id] = used_ids.get(frag_b.id, 0) + 1

        if verbose:
            b_str = f"+ '{frag_b.id}'" if frag_b else "(sin par)"
            print(f"  {slot.id} [{slot.label:20s}] "
                  f"A='{frag_a.id}' {b_str}  "
                  f"t={frag_a.tension_mean:.2f}")

    return plan


def _generate_variation_for_slot(frag: Fragment, slot: Slot, out_dir: Optional[str]) -> str:
    """Llama a variation_engine para generar una variante del fragmento."""
    if out_dir is None:
        out_dir = tempfile.mkdtemp()
    # Elegir variación según función del slot
    var_map = {
        "development": "V01",   # inversión
        "climax": "V10",        # rítmica intensificada
        "coda": "V04",          # aumentación
        "bridge": "V06",        # ornamentación
    }
    var_type = var_map.get(slot.function, "V14")  # estocástica por defecto
    out_path = str(Path(out_dir) / f"{frag.id}_{var_type}.mid")
    generate_variation(frag.file, [var_type], output_dir=out_dir)
    # variation_engine nombra el archivo automáticamente
    candidates = list(Path(out_dir).glob(f"*{var_type}*.mid"))
    if candidates:
        return str(candidates[0])
    return frag.file  # fallback al original


# ═══════════════════════════════════════════════════════════════════════════
# COSTURA DE FRAGMENTOS
# ═══════════════════════════════════════════════════════════════════════════

def detect_stitch_mode(frag_a: Fragment, frag_b: Fragment) -> str:
    """Selecciona automáticamente la técnica de costura entre dos fragmentos."""
    same_key = frag_a.key.split()[0] == frag_b.key.split()[0]
    tension_jump = abs(frag_b.tension_mean - frag_a.tension_end)
    cadence_a = frag_a.cadence_end

    if cadence_a in ("HC", "DC") and same_key:
        return "pivot"
    if tension_jump > 0.4:
        return "silence"
    if cadence_a == "AC" and not same_key:
        return "pivot"
    if frag_a.tempo and frag_b.tempo and abs(frag_a.tempo - frag_b.tempo) > 20:
        return "bridge"
    return "overlap"


def transpose_midi(mid: MidiFile, semitones: int) -> MidiFile:
    """Transpone un MidiFile en semitones."""
    new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.type in ("note_on", "note_off"):
                new_note = max(0, min(127, msg.note + semitones))
                new_track.append(msg.copy(note=new_note))
            else:
                new_track.append(msg.copy())
        new_mid.tracks.append(new_track)
    return new_mid



def strip_percussion(mid: MidiFile) -> MidiFile:
    """
    Elimina todos los eventos del canal 9 (percusión General MIDI).
    Devuelve un MidiFile nuevo sin modificar el original.
    """
    new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    for track in mid.tracks:
        new_track = MidiTrack()
        for msg in track:
            if msg.is_meta:
                new_track.append(msg)
            elif getattr(msg, "channel", -1) == 9:
                # Convertir note_on/off de canal 9 en silencio preservando el tiempo
                new_track.append(Message("note_on", channel=0, note=0,
                                         velocity=0, time=msg.time))
            else:
                new_track.append(msg)
        new_mid.tracks.append(new_track)
    return new_mid

def key_to_midi_root(key_str: str) -> int:
    """Convierte 'D minor' → 2, 'Ab major' → 8, etc."""
    name_map = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,
                "F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
    parts = key_str.strip().split()
    return name_map.get(parts[0], 0)


def stitch_midi_pair(
    frag_a: Fragment,
    frag_b: Fragment,
    stitch_mode: str,
    target_key: str,
    bridge_bars: int = 2,
    out_dir: str = ".",
    verbose: bool = False,
) -> str:
    """
    Cose dos fragmentos con la técnica indicada.
    Retorna la ruta del MIDI resultante (fragmento B transpuesto/ajustado).
    """
    if stitch_mode == "auto":
        stitch_mode = detect_stitch_mode(frag_a, frag_b)

    if verbose:
        print(f"    stitching '{frag_a.id}' → '{frag_b.id}' [{stitch_mode}]")

    mid_b = MidiFile(frag_b.file)

    # ── PIVOT: transponer frag_b a la tonalidad del resultado ──
    if stitch_mode in ("pivot", "direct", "overlap"):
        target_root = key_to_midi_root(target_key)
        b_root = key_to_midi_root(frag_b.key)
        semitones = (target_root - b_root) % 12
        if semitones > 6:
            semitones -= 12
        if semitones != 0:
            mid_b = transpose_midi(mid_b, semitones)

    # ── BRIDGE: generar puente con morpher.py ──
    elif stitch_mode == "bridge" and os.path.exists(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "morpher.py")):
        try:
            bridge_path = os.path.join(out_dir, f"bridge_{frag_a.id}_{frag_b.id}.mid")
            morph_midi(
                frag_a.file, frag_b.file,
                steps=3,
                mode="sigmoid",
                bars=bridge_bars,
                output_dir=out_dir,
                catalog=True,
            )
            # El puente ya está en out_dir, se concatenará en assemble_sequence
        except Exception as e:
            if verbose:
                print(f"    [WARN] morpher falló: {e}. Usando pivot.")
            stitch_mode = "pivot"

    # ── SILENCE: añadir silencio al inicio de frag_b ──
    elif stitch_mode == "silence":
        ticks_per_beat = mid_b.ticks_per_beat or 480
        silence_beats = 2
        new_mid = MidiFile(ticks_per_beat=ticks_per_beat, type=mid_b.type)
        for track in mid_b.tracks:
            new_track = MidiTrack()
            first = True
            for msg in track:
                if first and not msg.is_meta:
                    new_track.append(msg.copy(time=msg.time + silence_beats * ticks_per_beat))
                    first = False
                else:
                    new_track.append(msg.copy())
            new_mid.tracks.append(new_track)
        mid_b = new_mid

    out_path = os.path.join(out_dir, f"stitched_{frag_b.id}.mid")
    mid_b.save(out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# TRANSFORMACIONES DIATÓNICAS COHERENTES
# ═══════════════════════════════════════════════════════════════════════════

# Escalas diatónicas por modo: lista de semitonos relativos a la tónica
SCALE_INTERVALS = {
    "major":             [0, 2, 4, 5, 7, 9, 11],
    "minor":             [0, 2, 3, 5, 7, 8, 10],
    "dorian":            [0, 2, 3, 5, 7, 9, 10],
    "phrygian":          [0, 1, 3, 5, 7, 8, 10],
    "lydian":            [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":        [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor":    [0, 2, 3, 5, 7, 8, 11],
}


def _get_scale(key_str: str) -> list:
    """Devuelve la lista de MIDI pitches de la escala en todas las octavas (21–108)."""
    parts   = key_str.strip().lower().split()
    root    = key_to_midi_root(key_str)
    mode    = "major"
    for m in SCALE_INTERVALS:
        if m in " ".join(parts):
            mode = m
            break
    if "minor" in parts and "harmonic" not in parts:
        mode = "minor"
    intervals = SCALE_INTERVALS[mode]
    pitches = []
    for octave in range(-1, 9):
        for iv in intervals:
            p = root + iv + octave * 12
            if 21 <= p <= 108:
                pitches.append(p)
    return sorted(set(pitches))


def _snap_to_scale(pitch: int, scale: list) -> int:
    """Ajusta un pitch al grado más cercano de la escala."""
    if not scale:
        return int(np.clip(pitch, 21, 108))
    return min(scale, key=lambda p: (abs(p - pitch), p))


def _extract_notes(mid: MidiFile) -> list:
    """Extrae lista de (tick_abs, channel, note, velocity, duration_ticks)."""
    events = []
    for track in mid.tracks:
        t = 0
        pending = {}
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in pending:
                    t_on, vel = pending.pop(key)
                    events.append([t_on, msg.channel, msg.note, vel, max(1, t - t_on)])
    events.sort()
    return events


def _notes_to_midi(notes: list, ticks_per_beat: int, tempo_us: int = 500000) -> MidiFile:
    """Convierte lista de notas absolutas a MidiFile tipo 0."""
    mid   = MidiFile(ticks_per_beat=ticks_per_beat, type=0)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))
    raw = []
    for (t_on, ch, note, vel, dur) in notes:
        raw.append((t_on,         "note_on",  ch, note, vel))
        raw.append((t_on + dur,   "note_off", ch, note, 0))
    raw.sort(key=lambda x: (x[0], 0 if x[1] == "note_off" else 1))
    prev_t = 0
    for (t_abs, mtype, ch, note, vel) in raw:
        delta  = max(0, t_abs - prev_t)
        track.append(Message(mtype, channel=ch, note=note,
                             velocity=vel, time=delta))
        prev_t = t_abs
    track.append(MetaMessage("end_of_track", time=0))
    return mid


def _split_phrases(notes: list, tpb: int) -> list:
    """
    Divide la lista de notas en frases separadas por silencios >= 1 beat.
    Devuelve lista de listas de notas.
    """
    if not notes:
        return [[]]
    phrases   = []
    current   = [notes[0]]
    threshold = tpb  # 1 beat de silencio = nueva frase
    for i in range(1, len(notes)):
        gap = notes[i][0] - (notes[i-1][0] + notes[i-1][4])
        if gap >= threshold:
            phrases.append(current)
            current = []
        current.append(notes[i])
    if current:
        phrases.append(current)
    return phrases if phrases else [[]]


def _diatonic_invert(notes: list, scale: list, pivot_idx: int = None) -> list:
    """
    Invierte el contorno melódico en el espacio diatónico.
    El eje de inversión es el grado más cercano a la mediana de alturas.
    Cada intervalo ascendente se convierte en descendente del mismo número
    de GRADOS (no semitonos), garantizando que el resultado quede en escala.
    """
    if not notes or not scale:
        return notes
    pitches = [n[2] for n in notes]
    # Encontrar grado de cada pitch en la escala
    def nearest_degree(p):
        return min(range(len(scale)), key=lambda i: abs(scale[i] - p))
    degrees   = [nearest_degree(p) for p in pitches]
    pivot_deg = nearest_degree(int(np.median(pitches)))
    new_degrees = [2 * pivot_deg - d for d in degrees]
    new_pitches = [scale[max(0, min(len(scale)-1, d))] for d in new_degrees]
    result = []
    for i, n in enumerate(notes):
        result.append([n[0], n[1], new_pitches[i], n[3], n[4]])
    return result


def _diatonic_transpose(notes: list, scale: list, steps: int) -> list:
    """
    Transpone en N grados diatónicos (no semitonos).
    steps=2 sube una tercera, steps=-1 baja una segunda, etc.
    """
    if not notes or not scale:
        return notes
    def nearest_degree(p):
        return min(range(len(scale)), key=lambda i: abs(scale[i] - p))
    result = []
    for n in notes:
        deg      = nearest_degree(n[2])
        new_deg  = max(0, min(len(scale)-1, deg + steps))
        new_pitch = scale[new_deg]
        result.append([n[0], n[1], new_pitch, n[3], n[4]])
    return result


def _retrograde(notes: list) -> list:
    """Invierte el orden temporal manteniendo duraciones."""
    if not notes:
        return notes
    total = notes[-1][0] + notes[-1][4]
    result = []
    for n in reversed(notes):
        new_t = total - n[0] - n[4]
        result.append([max(0, new_t), n[1], n[2], n[3], n[4]])
    result.sort()
    return result


def _augment(notes: list, factor: float) -> list:
    """Aumenta duraciones y posiciones temporales por un factor."""
    return [[round(n[0]*factor), n[1], n[2], n[3], max(1, round(n[4]*factor))]
            for n in notes]


def _diminish(notes: list, factor: float, tpb: int) -> list:
    """Disminuye duraciones; asegura duración mínima de 1/16."""
    min_dur = tpb // 4
    return [[round(n[0]*factor), n[1], n[2], n[3], max(min_dur, round(n[4]*factor))]
            for n in notes]


def _permute_phrases(notes: list, tpb: int) -> list:
    """Permuta el orden de las frases para romper la estructura reconocible."""
    phrases = _split_phrases(notes, tpb)
    if len(phrases) < 2:
        return notes
    # Rotar: la segunda frase va primero
    reordered = phrases[1:] + phrases[:1]
    # Recalcular tiempos absolutos consecutivos
    result   = []
    cursor   = 0
    gap      = tpb * 2   # 2 beats entre frases permutadas
    for phrase in reordered:
        if not phrase:
            continue
        offset = cursor - phrase[0][0]
        for n in phrase:
            result.append([n[0] + offset, n[1], n[2], n[3], n[4]])
        cursor = result[-1][0] + result[-1][4] + gap
    return result


def _deidentify(mid: MidiFile, slot_function: str,
                target_key: str, rng: np.random.Generator) -> MidiFile:
    """
    Aplica transformaciones diatónicas para que el fragmento sea irreconocible
    pero musicalmente coherente (todas las notas quedan en la escala objetivo).

    Transformaciones por función:
      introduction / coda      → aumentación x1.5 + transporte diatónico ±3 grados
      theme_a / recapitulation → inversión diatónica + transporte ±2 grados
      development              → retrógrado + inversión diatónica + ±4 grados
      climax                   → diminución x0.65 + inversión diatónica
      theme_b / episode        → permutación de frases + transporte ±3 grados
      bridge y otros           → transporte diatónico ±2 grados
    """
    tpb   = mid.ticks_per_beat or 480
    scale = _get_scale(target_key)
    notes = _extract_notes(mid)
    if not notes:
        return mid

    func  = slot_function

    # Snap inicial: llevar todas las notas a la escala objetivo
    notes = [[n[0], n[1], _snap_to_scale(n[2], scale), n[3], n[4]]
             for n in notes]

    if func in ("introduction", "coda"):
        steps  = int(rng.choice([-4, -3, 3, 4]))
        notes  = _augment(notes, 1.5)
        notes  = _diatonic_transpose(notes, scale, steps)

    elif func in ("theme_a", "recapitulation"):
        steps  = int(rng.choice([-3, -2, 2, 3]))
        notes  = _diatonic_invert(notes, scale)
        notes  = _diatonic_transpose(notes, scale, steps)

    elif func == "development":
        steps  = int(rng.choice([-5, -4, 4, 5]))
        notes  = _retrograde(notes)
        notes  = _diatonic_invert(notes, scale)
        notes  = _augment(notes, 1.25)
        notes  = _diatonic_transpose(notes, scale, steps)

    elif func == "climax":
        notes  = _diminish(notes, 0.65, tpb)
        notes  = _diatonic_invert(notes, scale)
        # Subir una octava para dar energía
        notes  = [[n[0], n[1], min(108, n[2]+12), n[3], n[4]] for n in notes]

    elif func in ("theme_b", "episode"):
        steps  = int(rng.choice([-4, -3, 3, 4]))
        notes  = _permute_phrases(notes, tpb)
        notes  = _diatonic_transpose(notes, scale, steps)

    else:  # bridge, etc.
        steps  = int(rng.choice([-3, -2, 2, 3]))
        notes  = _diatonic_transpose(notes, scale, steps)

    # Snap final por si alguna transformación aritmética salió de escala
    notes = [[n[0], n[1], _snap_to_scale(n[2], scale), n[3], n[4]]
             for n in notes]
    # Asegurar que los tiempos no sean negativos
    if notes:
        t_min = min(n[0] for n in notes)
        if t_min < 0:
            notes = [[n[0]-t_min, n[1], n[2], n[3], n[4]] for n in notes]

    return _notes_to_midi(notes, tpb)


def _rhythmic_graft(mid_melody: MidiFile, mid_rhythm: MidiFile,
                    scale: list) -> MidiFile:
    """
    Injerta el ritmo de mid_rhythm en las alturas de mid_melody,
    respetando el fraseo: empareja frases de ambas fuentes.

    El resultado tiene:
      - Onsets y duraciones de mid_rhythm (groove original)
      - Alturas de mid_melody ajustadas a la escala (coherencia tonal)
      - Dinámica (velocity) de mid_rhythm (expresividad original)
    """
    tpb      = mid_melody.ticks_per_beat or 480
    notes_m  = _extract_notes(mid_melody)
    notes_r  = _extract_notes(mid_rhythm)

    if not notes_m or not notes_r:
        return mid_melody

    # Snap alturas de la melodía a la escala
    pitches_m  = [_snap_to_scale(n[2], scale) for n in notes_m]
    channels_m = [n[1] for n in notes_m]

    # Construir nuevas notas: timing del ritmo, alturas de la melodía
    new_notes = []
    for i, (t_on, ch_r, note_r, vel_r, dur_r) in enumerate(notes_r):
        idx      = i % len(pitches_m)
        new_notes.append((t_on, channels_m[idx], pitches_m[idx], vel_r, dur_r))

    return _notes_to_midi(new_notes, tpb)


def _phrase_build_slot(mid_path: str, slot, target_key: str,
                       out_dir: str, verbose: bool = False) -> str:
    """
    Reestructura el material del slot en forma antecedente/consecuente
    usando phrase_builder.py via subprocess.

    Solo se aplica a funciones temáticas (introduction, theme_a, theme_b,
    recapitulation) donde la estructura de frase es musicalmente relevante.
    Si falla, devuelve mid_path sin modificar.
    """
    PHRASE_FUNCS = {"introduction", "theme_a", "theme_b", "recapitulation", "coda"}
    if slot.function not in PHRASE_FUNCS:
        return mid_path

    out_path = os.path.join(out_dir, f"slot_{slot.id}_phrase.mid")
    if os.path.exists(out_path):
        return out_path

    # Mapear función del slot → tipo de frase
    form_map = {
        "introduction":    ("period",   "parallel"),
        "theme_a":         ("period",   "parallel"),
        "theme_b":         ("period",   "contrasting"),
        "recapitulation":  ("sentence", "parallel"),
        "coda":            ("sentence", "contrasting"),
    }
    form, ptype = form_map.get(slot.function, ("period", "parallel"))

    # Cadencias: antecedente → HC, consecuente → AC
    cadences = ["HC", "AC"]

    args_list = [
        "--motif", mid_path,
        "--key",   target_key,
        "--form",  form,
        "--type",  ptype,
        "--cadences", cadences[0], cadences[1],
        "--bars",  str(slot.bars),
        "--output", out_path,
        "--output-dir", out_dir,
    ]

    ok = _run_script("phrase_builder.py", args_list, out_path,
                     verbose=verbose, timeout=60)

    if ok:
        if verbose:
            print(f"    [PHRASE] {form}/{ptype} → {os.path.basename(out_path)}")
        return out_path

    # fallback: devolver el material sin restructurar
    return mid_path


def _complete_fragment(frag, slot, corpus_dir: str,
                       out_dir: str, verbose: bool = False):
    """
    Usa completer.py para extender un fragmento corto hasta cubrir los
    compases requeridos por el slot.

    Se llama desde select_fragments cuando --fill-gaps está activo y
    ningún fragmento del banco cubre los compases necesarios.
    Devuelve un nuevo Fragment con el archivo completado, o el original si falla.
    """
    target_bars = slot.bars
    if frag.bars >= target_bars:
        return frag   # ya es suficientemente largo

    out_path = os.path.join(out_dir, f"{frag.id}_completed_{target_bars}b.mid")
    if os.path.exists(out_path):
        try:
            cf = index_fragment(out_path)
            cf.id     = f"{frag.id}_completed"
            cf.source = frag.source
            return cf
        except Exception:
            pass

    args_list = [
        frag.file,
        "--bars",     str(target_bars - frag.bars),   # compases a añadir
        "--strategy", "harvest",
        "--key",      frag.key,
        "--output",   out_path,
    ]
    if corpus_dir and os.path.isdir(corpus_dir):
        args_list += ["--collection", corpus_dir]

    ok = _run_script("completer.py", args_list, out_path,
                     verbose=verbose, timeout=90)

    if ok:
        try:
            cf = index_fragment(out_path)
            cf.id     = f"{frag.id}_completed"
            cf.source = frag.source
            if verbose:
                print(f"    [COMPLETE] {frag.id} → {cf.bars} compases")
            return cf
        except Exception as e:
            if verbose:
                print(f"    [WARN] completer indexing: {e}")

    return frag   # fallback al original


def blend_slot(
    frag_a,
    frag_b,
    slot,
    target_key,
    out_dir,
    verbose=False,
    rng=None,
):
    """
    Genera material nuevo para el slot en tres capas:

    [1] DEIDENTIFY  — transforma frag_a con inversión/retrógrado/aumentación
                      según la función del slot, haciéndolo irreconocible.
    [2] GRAFT       — si hay frag_b, injerta su ritmo en las alturas
                      transformadas de frag_a (rhythmic_graft).
    [3] BLEND DNA   — si midi_dna_unified está disponible, lo usa como
                      capa adicional encima del resultado anterior.

    Resultado: material que no suena a ninguna de las fuentes originales.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    slot_path = os.path.join(out_dir, f"slot_{slot.id}.mid")

    # ── [1] Deidentificar frag_a ─────────────────────────────────────────────
    try:
        mid_a = MidiFile(frag_a.file)
        mid_transformed = _deidentify(mid_a, slot.function, target_key, rng)
        if verbose:
            print(f"    [DEIDENTIFY] {slot.function} → transformado")
    except Exception as e:
        if verbose:
            print(f"    [WARN] deidentify falló: {e}")
        mid_transformed = MidiFile(frag_a.file)

    # ── [2] Injertar ritmo de frag_b (en espacio diatónico) ─────────────────
    scale = _get_scale(target_key)
    if frag_b is not None:
        try:
            mid_b = MidiFile(frag_b.file)
            mid_grafted = _rhythmic_graft(mid_transformed, mid_b, scale)
            if verbose:
                print(f"    [GRAFT] ritmo de '{frag_b.id}' → injertado")
        except Exception as e:
            if verbose:
                print(f"    [WARN] graft falló: {e}")
            mid_grafted = mid_transformed
    else:
        mid_grafted = mid_transformed

    # Guardar resultado intermedio para midi_dna_unified
    intermediate = os.path.join(out_dir, f"slot_{slot.id}_intermediate.mid")
    try:
        mid_grafted.save(intermediate)
    except Exception as e:
        if verbose:
            print(f"    [WARN] no se pudo guardar intermedio: {e}")
        intermediate = frag_a.file

    # ── [3] Blend DNA con midi_dna_unified (subprocess, sin import) ──────────
    dna_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "midi_dna_unified.py"
    )
    if frag_b is not None and os.path.exists(dna_script):
        try:
            import subprocess, sys as _sys
            func = slot.function
            if func in ("development", "climax"):
                sources = "melody=0,rhythm=1,harmony=1,emotion=0"
            elif func in ("theme_b", "episode"):
                sources = "melody=1,rhythm=0,harmony=1,emotion=1"
            else:
                sources = "melody=0,rhythm=1,harmony=0,emotion=0"

            cmd = [
                _sys.executable, dna_script,
                intermediate, frag_b.file,
                "--mode", "full_blend",
                "--sources", sources,
                "--bars", str(slot.bars),
                "--key", target_key,
                "--output", slot_path,
                "--no-percussion",
            ]
            if verbose:
                print(f"    [DNA_BLEND] {intermediate} + '{frag_b.id}'")

            result = subprocess.run(cmd, capture_output=True,
                                    text=True, timeout=120)
            if result.returncode == 0 and os.path.exists(slot_path):
                if verbose:
                    print(f"    → DNA blend OK")
                return slot_path
            if verbose:
                print(f"    [WARN] midi_dna_unified rc={result.returncode}")
                if result.stderr:
                    print(f"           {result.stderr.strip()[:200]}")
        except Exception as e:
            if verbose:
                print(f"    [WARN] DNA blend: {e}")

    # ── Fallback: usar el resultado del graft directamente ───────────────────
    try:
        mid_grafted.save(slot_path)
    except Exception:
        import shutil
        shutil.copy(frag_a.file, slot_path)

    # ── [4] Restructurar en forma de frase (phrase_builder) ──────────────────
    slot_path = _phrase_build_slot(
        slot_path, slot, target_key, out_dir, verbose=verbose
    )

    return slot_path


def _morph_transition(frag_a, frag_b, bridge_bars, out_dir, target_key, verbose=False):
    """
    Genera un puente de morphing entre frag_a y frag_b con morpher.py via subprocess.
    Retorna ruta del MIDI del puente, o None si no disponible / falla.
    """
    bridge_path = os.path.join(
        out_dir, f"bridge_{frag_a.id[:20]}_{frag_b.id[:20]}.mid"
    )
    # No regenerar si ya existe
    if os.path.exists(bridge_path):
        return bridge_path

    morph_dir = os.path.join(out_dir, "morph_tmp")
    os.makedirs(morph_dir, exist_ok=True)

    # Tomar nota de los archivos presentes antes de llamar
    before = set(Path(morph_dir).glob("*.mid"))

    args_list = [
        frag_a.file, frag_b.file,
        "--steps", "3",
        "--mode", "sigmoid",
        "--catalog",
        "--output-dir", morph_dir,
        "--prefix", "bridge",
    ]
    if bridge_bars:
        args_list += ["--bars", str(bridge_bars)]

    ok = _run_script("morpher.py", args_list, "",  # out_path vacío; buscamos después
                     verbose=verbose, timeout=60)

    after = set(Path(morph_dir).glob("*.mid"))
    new_files = sorted(after - before)

    if not new_files:
        if verbose:
            print(f"    [WARN] morpher no generó archivos")
        return None

    # Preferir el que tiene "catalog" en el nombre (paso intermedio del morphing)
    catalog = [f for f in new_files if "catalog" in f.stem.lower()]
    chosen  = catalog[0] if catalog else new_files[-1]   # último = más próximo a B

    import shutil
    shutil.copy(str(chosen), bridge_path)
    if verbose:
        print(f"    [MORPH] puente: {chosen.name} → {os.path.basename(bridge_path)}")
    return bridge_path


def _append_midi(output_track, mid_path, ticks_per_beat, verbose=False):
    """Añade los eventos de un MIDI al track de salida, escalando ticks."""
    try:
        mid = MidiFile(mid_path)
    except Exception as e:
        if verbose:
            print(f"  [ERROR] no se puede leer {mid_path}: {e}")
        return
    ratio = ticks_per_beat / (mid.ticks_per_beat or 480)
    for track in mid.tracks:
        for msg in track:
            if msg.is_meta:
                if msg.type not in ("track_name", "end_of_track"):
                    output_track.append(msg)
            else:
                output_track.append(msg.copy(time=round(msg.time * ratio)))


def assemble_sequence(
    slots,
    bank,
    stitch,
    target_key,
    bridge_bars,
    out_dir,
    verbose=False,
    seed=42,
):
    """
    Ensambla la obra slot a slot usando fusión real de ADNs.

    Para cada slot:
      1. blend_slot() llama a midi_dna_unified --mode full_blend con los
         dos fragmentos seleccionados: melodía de A + ritmo/armonía de B.
         El resultado es material NUEVO que nunca existió en ninguna obra
         original — no una concatenación.

      2. Entre slots consecutivos, inserta un puente de morphing (morpher.py)
         si stitch==bridge, o ajusta tonalidad/silencio en otros modos.
    """
    bank_by_id = {f.id: f for f in bank}
    ticks_per_beat = 480
    output_track = MidiTrack()
    output_mid = MidiFile(ticks_per_beat=ticks_per_beat, type=0)
    output_mid.tracks.append(output_track)

    prev_frag_a = None
    rng = np.random.default_rng(seed)

    for slot in slots:
        if not slot.assigned_file or not os.path.exists(slot.assigned_file):
            if verbose:
                print(f"  [SKIP] {slot.id}: sin fragmento asignado.")
            continue

        frag_a = bank_by_id.get(slot.assigned_fragment)
        if frag_a is None:
            frag_a = Fragment(id=slot.assigned_fragment or "unknown",
                              file=slot.assigned_file)

        frag_b = None
        if getattr(slot, 'assigned_fragment_b', None) and getattr(slot, 'assigned_file_b', None):
            frag_b = bank_by_id.get(slot.assigned_fragment_b)
            if frag_b is None:
                frag_b = Fragment(id=slot.assigned_fragment_b,
                                  file=slot.assigned_file_b)

        # ── Transición desde slot anterior ──────────────────────────────────
        if prev_frag_a is not None:
            slot_stitch = slot.stitch_next if stitch == "auto" else stitch
            if slot_stitch == "auto":
                slot_stitch = detect_stitch_mode(prev_frag_a, frag_a)

            if slot_stitch == "bridge":
                bridge_mid = _morph_transition(
                    prev_frag_a, frag_a,
                    bridge_bars=bridge_bars,
                    out_dir=out_dir,
                    target_key=target_key,
                    verbose=verbose,
                )
                if bridge_mid:
                    _append_midi(output_track, bridge_mid, ticks_per_beat, verbose)
            elif slot_stitch == "silence":
                silence_ticks = ticks_per_beat * 4
                output_track.append(
                    Message("note_on", channel=0, note=0, velocity=0,
                            time=silence_ticks)
                )

        # ── Fusionar ADNs y generar material del slot ────────────────────────
        if verbose:
            b_label = f"'{frag_b.id}'" if frag_b else "—"
            print(f"\n  [{slot.id}] {slot.label}")
            print(f"    melodia : '{frag_a.id}'")
            print(f"    ritmo   : {b_label}")

        slot_mid = blend_slot(
            frag_a=frag_a,
            frag_b=frag_b,
            slot=slot,
            target_key=target_key,
            out_dir=out_dir,
            verbose=verbose,
            rng=rng,
        )
        _append_midi(output_track, slot_mid, ticks_per_beat, verbose)
        prev_frag_a = frag_a

    output_track.append(MetaMessage("end_of_track", time=0))

    assembled_path = os.path.join(out_dir, "mosaic_assembled.mid")
    output_mid.save(assembled_path)
    if verbose:
        print(f"\n  ✓ Ensamblaje guardado: {assembled_path}")
    return assembled_path


# ═══════════════════════════════════════════════════════════════════════════
# PASO DE PULIDO
# ═══════════════════════════════════════════════════════════════════════════

def _run_script(script_name: str, args_list: list,
                out_path: str, verbose: bool = False,
                timeout: int = 120) -> bool:
    """
    Llama a un script del ecosistema via subprocess.
    Devuelve True si tuvo éxito y out_path existe.
    Nunca usa import — todos los scripts tienen código de nivel superior.
    """
    import subprocess, sys as _sys
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), script_name
    )
    if not os.path.exists(script):
        if verbose:
            print(f"  [WARN] {script_name} no encontrado en {os.path.dirname(script)}")
        return False
    cmd = [_sys.executable, script] + args_list
    if verbose:
        print(f"  cmd: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=not verbose,
                             text=True, timeout=timeout)
        if res.returncode != 0:
            print(f"  [WARN] {script_name} rc={res.returncode}")
            if res.stderr:
                print(f"         {res.stderr.strip()[:200]}")
            return False
        if not os.path.exists(out_path):
            print(f"  [WARN] {script_name} no generó {out_path}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [WARN] {script_name}: timeout ({timeout}s)")
        return False
    except Exception as e:
        print(f"  [WARN] {script_name}: {e}")
        return False


def reharmonize_midi(input_path: str, out_dir: str,
                     strategy: str = "diatonic",
                     verbose: bool = False) -> str:
    """
    Reharmoniza un MIDI via reharmonizer.py subprocess.
    Devuelve la ruta del resultado (o input_path si falla).
    """
    rh_dir = os.path.join(out_dir, "rh_tmp")
    os.makedirs(rh_dir, exist_ok=True)
    expected = os.path.join(rh_dir, Path(input_path).stem + ".mid")
    ok = _run_script(
        "reharmonizer.py",
        [input_path, "--strategy", strategy, "--out-dir", rh_dir],
        expected, verbose=verbose,
    )
    if ok:
        return expected
    # fallback: buscar cualquier .mid generado
    candidates = list(Path(rh_dir).glob("*.mid"))
    return str(candidates[0]) if candidates else input_path


def polish(
    assembled_path: str,
    out_dir: str,
    reharmonize: bool = False,
    style_unify_file: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Aplica reharmonización y/o unificación de estilo al MIDI ensamblado.
    Usa subprocess para todos los scripts externos — nunca import directo.
    Siempre imprime qué está haciendo, con o sin --verbose.
    """
    current_path = assembled_path

    if reharmonize:
        print("[POLISH] Reharmonizando …")
        # reharmonizer guarda el resultado en --out-dir con el mismo nombre base
        rh_dir  = os.path.join(out_dir, "rh_tmp")
        os.makedirs(rh_dir, exist_ok=True)
        rh_name = Path(current_path).stem + ".mid"
        rh_path = os.path.join(rh_dir, rh_name)
        ok = _run_script(
            "reharmonizer.py",
            [current_path,
             "--strategy", "diatonic",
             "--out-dir", rh_dir],
            rh_path, verbose=verbose,
        )
        # reharmonizer nombra los archivos con sufijo _diatonic_01, etc.
        # buscar el mejor candidato generado aunque el nombre exacto no coincida
        candidates = sorted(Path(rh_dir).glob("*.mid"))
        if candidates:
            current_path = str(candidates[0])
            print(f"  ✓ {current_path}")
        elif ok:
            current_path = rh_path
            print(f"  ✓ {rh_path}")
        else:
            print("  [WARN] reharmonizer.py falló — omitido")

    if style_unify_file:
        print(f"[POLISH] Unificando estilo con '{Path(style_unify_file).name}' …")
        st_path = os.path.join(out_dir, "mosaic_styled.mid")
        ok = _run_script(
            "style_transfer.py",
            [current_path, style_unify_file,
             "--transfer", "rhythm", "texture", "dynamics",
             "--preserve", "contour", "motif",
             "--strength", "0.5",
             "--output", st_path],
            st_path, verbose=verbose,
        )
        if ok:
            current_path = st_path
            print(f"  ✓ {st_path}")
        else:
            print("  [WARN] style_transfer.py no disponible o falló — omitido")

    return current_path


# ═══════════════════════════════════════════════════════════════════════════
# COMANDO: HARVEST
# ═══════════════════════════════════════════════════════════════════════════

def _harvest_via_subprocess(mid_path: str, bank_dir: str,
                             min_bars: int, max_bars: int,
                             verbose: bool = False) -> List[str]:
    """
    Llama a harvester.py via subprocess y devuelve las rutas de los
    fragmentos generados en bank_dir.
    Nunca usa import (harvester tiene código de nivel superior que falla).
    """
    import subprocess, sys as _sys
    harvester_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "harvester.py"
    )
    if not os.path.exists(harvester_script):
        return []   # script no disponible → el caller usará fallback

    # Número de fragmentos antes de llamar
    before = set(Path(bank_dir).glob("*.mid"))

    cmd = [
        _sys.executable, harvester_script,
        mid_path,
        "--out-dir", bank_dir,
        "--mode", "all",
        "--min-bars", str(min_bars),
        "--max-bars", str(max_bars),
    ]
    result = subprocess.run(cmd, capture_output=not verbose,
                            text=True, timeout=60)
    if result.returncode != 0 and verbose:
        print(f"    [WARN] harvester rc={result.returncode}: "
              f"{result.stderr.strip()[:120]}")

    # Fragmentos nuevos = los que aparecieron después de la llamada
    after = set(Path(bank_dir).glob("*.mid"))
    new_files = sorted(after - before)
    return [str(f) for f in new_files]


def _split_midi_into_fragments(mid_path: str, bank_dir: str,
                                min_bars: int, max_bars: int) -> List[str]:
    """
    Fallback interno: divide el MIDI en fragmentos de min_bars–max_bars compases
    usando solo mido, sin depender de harvester.py.
    Detecta fronteras por silencios largos (>= 1 beat) y por tamaño de ventana.
    """
    try:
        mid   = MidiFile(mid_path)
        tpb   = mid.ticks_per_beat or 480
        stem  = Path(mid_path).stem

        # Recopilar todos los eventos con tiempo absoluto
        all_events = []
        for track in mid.tracks:
            t = 0
            for msg in track:
                t += msg.time
                all_events.append((t, msg))
        all_events.sort(key=lambda x: x[0])

        if not all_events:
            return []

        total_ticks  = all_events[-1][0]
        beats_total  = total_ticks / tpb
        bars_total   = max(1, round(beats_total / 4))

        # Si el MIDI es más corto que min_bars, guardarlo entero
        if bars_total <= min_bars:
            import shutil
            dest = str(Path(bank_dir) / f"{stem}.mid")
            shutil.copy(mid_path, dest)
            return [dest]

        # Detectar fronteras: silencios >= 2 beats entre notas
        note_ticks = sorted(
            t for t, msg in all_events
            if msg.type == "note_on" and getattr(msg, "velocity", 0) > 0
        )
        if not note_ticks:
            import shutil
            dest = str(Path(bank_dir) / f"{stem}.mid")
            shutil.copy(mid_path, dest)
            return [dest]

        silence_threshold = tpb * 2   # 2 beats
        boundaries = [note_ticks[0]]
        for i in range(1, len(note_ticks)):
            if note_ticks[i] - note_ticks[i-1] >= silence_threshold:
                boundaries.append(note_ticks[i])
        boundaries.append(total_ticks + tpb)

        # Construir ventanas respetando min/max_bars
        windows   = []
        win_start = note_ticks[0]
        for b in boundaries[1:]:
            win_ticks = b - win_start
            win_bars  = win_ticks / tpb / 4
            if win_bars >= min_bars:
                # Partir en sub-ventanas si es mayor que max_bars
                cursor = win_start
                while cursor < b:
                    end = min(cursor + max_bars * 4 * tpb, b)
                    seg_bars = (end - cursor) / tpb / 4
                    if seg_bars >= min_bars:
                        windows.append((cursor, end))
                    cursor = end
                win_start = b

        if not windows:
            import shutil
            dest = str(Path(bank_dir) / f"{stem}.mid")
            shutil.copy(mid_path, dest)
            return [dest]

        # Exportar cada ventana como MIDI independiente
        saved = []
        for idx, (t_start, t_end) in enumerate(windows):
            new_mid   = MidiFile(ticks_per_beat=tpb, type=0)
            new_track = MidiTrack()
            new_mid.tracks.append(new_track)

            pending_off: dict = {}
            prev_abs = t_start
            for t_abs, msg in all_events:
                if t_abs < t_start:
                    continue
                if t_abs >= t_end:
                    # Añadir note_off pendientes
                    for (ch, note), t_on in list(pending_off.items()):
                        new_track.append(Message(
                            "note_off", channel=ch, note=note,
                            velocity=0, time=max(0, t_end - prev_abs)
                        ))
                        prev_abs = t_end
                    break
                delta = max(0, t_abs - prev_abs)
                if msg.is_meta:
                    if msg.type not in ("end_of_track",):
                        new_track.append(msg.copy(time=delta))
                else:
                    new_track.append(msg.copy(time=delta))
                    if msg.type == "note_on" and getattr(msg,"velocity",0) > 0:
                        pending_off[(msg.channel, msg.note)] = t_abs
                    elif msg.type in ("note_off",) or (
                            msg.type == "note_on" and getattr(msg,"velocity",0) == 0):
                        pending_off.pop((msg.channel, getattr(msg,"note",0)), None)
                prev_abs = t_abs

            new_track.append(MetaMessage("end_of_track", time=0))

            # Solo guardar si tiene notas
            has_notes = any(
                m.type == "note_on" and getattr(m,"velocity",0) > 0
                for m in new_track
            )
            if has_notes:
                dest = str(Path(bank_dir) / f"{stem}_frag{idx:03d}.mid")
                new_mid.save(dest)
                saved.append(dest)

        return saved

    except Exception as e:
        print(f"    [WARN] split fallback: {e}")
        return []



# Variaciones disponibles en variation_engine con su rol dramático natural
VARIATION_ROLES = {
    "V01": ("inversión melódica",      ["development", "climax"]),
    "V02": ("retrógrado",              ["development"]),
    "V03": ("inversión retrógrada",    ["development"]),
    "V04": ("aumentación rítmica",     ["coda", "introduction"]),
    "V05": ("diminución rítmica",      ["climax", "episode"]),
    "V06": ("ornamentación",           ["theme_a", "theme_b"]),
    "V07": ("cambio de acompañamiento",["theme_b", "episode"]),
    "V08": ("transposición",           ["bridge", "recapitulation"]),
    "V10": ("variación rítmica",       ["episode", "bridge"]),
    "V14": ("estocástica",             ["any"]),
}


def _generate_variations_for_fragment(
    frag: "Fragment",
    variations: List[str],
    var_dir: str,
    verbose: bool = False,
) -> List["Fragment"]:
    """
    Genera variaciones de un fragmento usando variation_engine.py via subprocess.
    Devuelve lista de Fragment nuevos (uno por variación generada con éxito).
    Los archivos se guardan en var_dir con sufijo _{Vxx}.mid.
    """
    import subprocess, sys as _sys

    var_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "variation_engine.py"
    )
    if not os.path.exists(var_script):
        if verbose:
            print(f"      [WARN] variation_engine.py no encontrado, omitiendo variaciones")
        return []

    Path(var_dir).mkdir(parents=True, exist_ok=True)
    result_frags = []

    for v_code in variations:
        out_path = os.path.join(var_dir, f"{frag.id}_{v_code}.mid")
        # No regenerar si ya existe (útil para re-runs)
        if os.path.exists(out_path):
            try:
                vf = index_fragment(out_path)
                vf.id     = f"{frag.id}_{v_code}"
                vf.source = frag.source
                result_frags.append(vf)
                continue
            except Exception:
                pass

        cmd = [
            _sys.executable, var_script,
            frag.file,
            "--variations", v_code,
            "--output-dir", var_dir,
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if res.returncode != 0:
                if verbose:
                    print(f"      [WARN] {v_code} rc={res.returncode}: "
                          f"{res.stderr.strip()[:80]}")
                continue

            # variation_engine puede nombrar el archivo de varias maneras;
            # buscamos cualquier nuevo .mid que contenga el código de variación
            candidates = list(Path(var_dir).glob(f"*{v_code}*.mid"))
            # Preferir el que también contiene el id del fragmento
            named = [c for c in candidates if frag.id[:10] in c.stem]
            chosen = named[0] if named else (candidates[0] if candidates else None)

            if chosen is None or not chosen.exists():
                if verbose:
                    print(f"      [WARN] {v_code}: archivo no encontrado tras ejecución")
                continue

            # Renombrar al nombre canónico para evitar colisiones
            if str(chosen) != out_path:
                import shutil
                shutil.move(str(chosen), out_path)

            vf = index_fragment(out_path)
            vf.id     = f"{frag.id}_{v_code}"
            vf.source = frag.source          # misma fuente que el original
            result_frags.append(vf)
            if verbose:
                descr = VARIATION_ROLES.get(v_code, (v_code,))[0]
                print(f"      + {v_code} ({descr}): {os.path.basename(out_path)}")

        except subprocess.TimeoutExpired:
            if verbose:
                print(f"      [WARN] {v_code}: timeout")
        except Exception as e:
            if verbose:
                print(f"      [WARN] {v_code}: {e}")

    return result_frags


def cmd_harvest(args) -> List[Fragment]:
    """
    Segmenta el corpus y construye el banco de fragmentos.

    Estrategia en cascada:
    [1] Intenta usar harvester.py via subprocess (segmentación musical completa)
    [2] Si no está disponible, usa _split_midi_into_fragments (segmentación
        por silencios y ventana de tamaño, sin dependencias externas)
    Nunca cae en el fallback de copiar el MIDI entero sin fragmentar.
    """
    corpus_dir = Path(args.corpus)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_dir   = out_dir / "bank"
    bank_dir.mkdir(exist_ok=True)

    min_bars   = getattr(args, "min_bars", 2)
    max_bars   = getattr(args, "max_bars", 16)
    verbose    = getattr(args, "verbose", False)
    variations = getattr(args, "variations", []) or []
    # --variations all → expandir a todos los códigos disponibles
    if variations == ["all"]:
        variations = list(VARIATION_ROLES.keys())

    midi_files = (list(corpus_dir.glob("**/*.mid")) +
                  list(corpus_dir.glob("**/*.midi")))
    if not midi_files:
        print(f"[ERROR] No se encontraron MIDIs en '{corpus_dir}'")
        sys.exit(1)

    # Detectar si harvester.py está disponible (una sola vez)
    harvester_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "harvester.py"
    )
    use_harvester = os.path.exists(harvester_script)
    if use_harvester:
        print(f"[HARVEST] harvester.py encontrado → segmentación musical")
    else:
        print(f"[HARVEST] harvester.py no encontrado → segmentación por silencios")

    if variations:
        print(f"[HARVEST] Variaciones: {' '.join(variations)}")
        var_dir = bank_dir / "variations"
        var_dir.mkdir(exist_ok=True)
    else:
        var_dir = None

    print(f"[HARVEST] {len(midi_files)} MIDIs | min={min_bars} max={max_bars} compases")

    fragments: List[Fragment] = []
    total_frags = 0
    total_vars  = 0

    for mid_path in midi_files:
        print(f"  {mid_path.name}", end=" ", flush=True)
        frag_paths = []
        try:
            if use_harvester:
                frag_paths = _harvest_via_subprocess(
                    str(mid_path), str(bank_dir),
                    min_bars=min_bars, max_bars=max_bars,
                    verbose=verbose,
                )
                if not frag_paths:
                    frag_paths = _split_midi_into_fragments(
                        str(mid_path), str(bank_dir),
                        min_bars=min_bars, max_bars=max_bars,
                    )
            else:
                frag_paths = _split_midi_into_fragments(
                    str(mid_path), str(bank_dir),
                    min_bars=min_bars, max_bars=max_bars,
                )
        except Exception as e:
            print(f"→ ERROR: {e}")
            continue

        n_ok  = 0
        n_var = 0
        for frag_path in frag_paths:
            try:
                frag = index_fragment(frag_path)
                fragments.append(frag)
                n_ok += 1

                # Generar variaciones del fragmento recién indexado
                if variations and var_dir is not None:
                    var_frags = _generate_variations_for_fragment(
                        frag, variations, str(var_dir), verbose=verbose
                    )
                    fragments.extend(var_frags)
                    n_var += len(var_frags)

            except Exception as e:
                if verbose:
                    print(f"\n    [WARN] no indexado '{frag_path}': {e}")

        total_frags += n_ok
        total_vars  += n_var
        suffix = f" (+{n_var} vars)" if n_var else ""
        print(f"→ {n_ok} fragmentos{suffix}")

    var_summary = f" + {total_vars} variaciones" if total_vars else ""
    print(f"\n[HARVEST] {total_frags} fragmentos{var_summary} "
          f"({total_frags + total_vars} total) de {len(midi_files)} MIDIs.")

    # Guardar banco
    bank_path = out_dir / ".mosaic_bank.json"
    bank_data = {"fragments": [asdict(f) for f in fragments]}
    bank_path.write_text(
        json.dumps(bank_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[HARVEST] Banco guardado: {bank_path.resolve()}")

    return fragments


# ═══════════════════════════════════════════════════════════════════════════
# COMANDO: PLAN
# ═══════════════════════════════════════════════════════════════════════════

def _plan_to_dict(plan: MosaicPlan) -> dict:
    """Serializa MosaicPlan a dict; tension_target queda como lista (JSON-safe)."""
    slots_data = []
    for s in plan.slots:
        sd = asdict(s)
        # asdict convierte Tuple→list automáticamente; _load_plan lo reconvierte
        slots_data.append(sd)
    return {
        "arc": plan.arc,
        "bars_total": plan.bars_total,
        "key": plan.key,
        "tempo": plan.tempo,
        "slots": slots_data,
    }


def _save_plan(plan: MosaicPlan, path: Path) -> Path:
    """Escribe el plan en disco y verifica que el archivo no quede vacío."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_plan_to_dict(plan), indent=2, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    written = path.read_text(encoding="utf-8").strip()
    if not written or written in ("{}", ""):
        raise RuntimeError(f"El plan se escribió vacío en '{path}'. Revisa permisos.")
    return path


def cmd_plan(args) -> MosaicPlan:
    """Diseña y exporta el arco dramático.

    Con --stdout escribe el JSON puro en stdout y los mensajes informativos
    en stderr, de modo que funciona correctamente con sustitución de proceso
    bash:  --plan <(python mosaic_composer.py plan --arc journey --stdout)
    """
    plan = build_plan(
        arc=args.arc,
        bars_total=args.bars,
        key=getattr(args, "key", "C major"),
        tempo=getattr(args, "tempo", 120.0),
    )

    stdout_mode = getattr(args, "stdout", False)
    # En modo --stdout todos los mensajes van a stderr para no contaminar el JSON
    log = (lambda *a, **k: print(*a, **k, file=sys.stderr)) if stdout_mode else print

    if stdout_mode:
        # Escribir JSON puro en stdout — esto es lo que captura bash <(...)
        sys.stdout.write(json.dumps(_plan_to_dict(plan), indent=2, ensure_ascii=False))
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        out_dir = Path(getattr(args, "out_dir", "./mosaic_out"))
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / "mosaic_plan.json"
        try:
            _save_plan(plan, plan_path)
        except RuntimeError as e:
            log(f"[ERROR] {e}")
            sys.exit(1)
        log(f"[PLAN] Guardado: {plan_path.resolve()}")

    log(f"[PLAN] Arco '{plan.arc}' con {len(plan.slots)} slots ({plan.bars_total} compases)")
    for slot in plan.slots:
        log(f"  {slot.id}  {slot.label:22s}  {slot.bars:2d} compases  "
            f"tensión {slot.tension_target[0]:.1f}–{slot.tension_target[1]:.1f}  "
            f"cadencia {slot.cadence_end}")
    return plan


# ═══════════════════════════════════════════════════════════════════════════
# COMANDO: ASSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

def _read_any(path: str) -> str:
    """
    Lee el contenido de texto de cualquier ruta, incluyendo file descriptors
    del kernel como /proc/self/fd/N (usados por la sustitución de proceso
    de bash: --plan <(python ... plan ...) ).
    No usa Path.exists() ni Path.read_text() porque fallan en /proc/self/fd/.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as e:
        return ""   # el caller comprobará si está vacío


def _load_bank(path: str) -> List[Fragment]:
    """Carga el banco desde JSON con validación explícita.
    Acepta rutas normales y file descriptors (/proc/self/fd/N).
    """
    raw = _read_any(path).strip()
    if not raw:
        print(f"[ERROR] El banco está vacío o no se pudo leer: {path}")
        print( "        Si usas sustitución de proceso bash <(...), asegúrate de")
        print( "        guardar primero el banco en un archivo real con 'harvest'.")
        sys.exit(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON inválido en banco '{path}': {e}")
        sys.exit(1)
    if "fragments" not in data:
        print(f"[ERROR] El banco no contiene la clave 'fragments': {path}")
        sys.exit(1)
    fragments = []
    for f in data["fragments"]:
        f.setdefault("fingerprint", {})
        f.setdefault("score", 0.0)
        fragments.append(Fragment(**f))
    return fragments


def _load_plan(path: str) -> MosaicPlan:
    """Carga el plan desde JSON con validación y coerciones de tipo.
    Acepta rutas normales y file descriptors (/proc/self/fd/N).

    NOTA: si pasas --plan <(python mosaic_composer.py plan ...) en bash,
    el descriptor de pipe solo puede leerse UNA vez. Esta función lo lee
    completamente en memoria antes de hacer cualquier validación, lo que
    garantiza que el contenido no se pierda.
    """
    raw = _read_any(path).strip()
    if not raw:
        print(f"[ERROR] El plan está vacío o no se pudo leer: {path}")
        print( "        Causa probable: sustitución de proceso bash <(...) cuyo")
        print( "        descriptor ya fue consumido, o archivo no generado aún.")
        print( "        Solución: guarda el plan primero con el subcomando 'plan'")
        print( "        y luego pasa la ruta del archivo .json a --plan.")
        sys.exit(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Mostrar contexto para ayudar a depurar planes editados a mano
        lines = raw.splitlines()
        ctx_start = max(0, e.lineno - 2)
        ctx_end   = min(len(lines), e.lineno + 2)
        print(f"[ERROR] JSON inválido en plan '{path}': {e}")
        for i, ln in enumerate(lines[ctx_start:ctx_end], ctx_start + 1):
            marker = ">>>" if i == e.lineno else "   "
            print(f"  {marker} {i:4d}: {ln}")
        sys.exit(1)
    for required_key in ("arc", "bars_total", "slots"):
        if required_key not in data:
            print(f"[ERROR] El plan no contiene la clave requerida '{required_key}': {path}")
            sys.exit(1)
    slots = []
    for s in data["slots"]:
        # tension_target viene como lista desde JSON → convertir a tupla
        if "tension_target" in s and isinstance(s["tension_target"], list):
            s["tension_target"] = tuple(s["tension_target"])
        # role_required puede venir como None desde JSON
        s.setdefault("role_required", [])
        if s["role_required"] is None:
            s["role_required"] = []
        # Campos opcionales que pueden no existir en planes editados a mano
        s.setdefault("style_hint", None)
        s.setdefault("stitch_next", "auto")
        s.setdefault("assigned_fragment", None)
        s.setdefault("assigned_file", None)
        s.setdefault("assigned_fragment_b", None)
        s.setdefault("assigned_file_b", None)
        slots.append(Slot(**s))
    return MosaicPlan(
        arc=data["arc"],
        bars_total=data["bars_total"],
        key=data.get("key", "C major"),
        tempo=data.get("tempo", 120.0),
        slots=slots,
    )


def _run_single_assemble(
    bank: List[Fragment],
    plan: "MosaicPlan",
    seed: int,
    out_dir: Path,
    output_name: str,
    stitch_mode: str,
    n_candidates: int,
    diversity: float,
    allow_variations: bool,
    bridge_bars: int,
    reharmonize: bool,
    style_unify_file: Optional[str],
    no_percussion: bool,
    verbose: bool,
    report: bool,
) -> str:
    """
    Ejecuta un único pase de ensamblaje con la semilla dada.
    Devuelve la ruta del archivo final generado.
    Usado tanto por cmd_assemble (una o varias semillas) como por cmd_build.
    """
    import copy, shutil

    # Cada semilla necesita su propio plan (las asignaciones se escriben en él)
    plan_copy = copy.deepcopy(plan)
    rng       = np.random.default_rng(seed)

    # Subdirectorio propio para no mezclar temporales entre semillas
    seed_dir = out_dir / f"seed_{seed:06d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[ASSEMBLE] seed={seed} — seleccionando fragmentos para "
          f"{len(plan_copy.slots)} slots …")
    plan_copy = select_fragments(
        bank=bank, plan=plan_copy,
        n_candidates=n_candidates,
        diversity=diversity,
        allow_variations=allow_variations,
        variation_out_dir=str(seed_dir),
        rng=rng,
        verbose=True,
    )

    print(f"\n[ASSEMBLE] Cosiendo secuencia [{stitch_mode}] …")
    assembled = assemble_sequence(
        slots=plan_copy.slots,
        bank=bank,
        stitch=stitch_mode,
        target_key=plan_copy.key,
        bridge_bars=bridge_bars,
        out_dir=str(seed_dir),
        verbose=verbose,
        seed=seed,
    )

    print(f"\n[POLISH] Puliendo resultado …")
    final = polish(
        assembled_path=assembled,
        out_dir=str(seed_dir),
        reharmonize=reharmonize,
        style_unify_file=style_unify_file,
        verbose=verbose,
    )

    # Nombre final: insertar seed antes de la extensión si hay varias semillas
    final_path = out_dir / output_name
    if no_percussion:
        print("[POLISH] Eliminando percusión (canal 9) …")
        try:
            mid_clean = strip_percussion(MidiFile(final))
            mid_clean.save(str(final_path))
            print(f"  ✓ percusión eliminada")
        except Exception as e:
            print(f"  [WARN] strip_percussion: {e} — copiando sin modificar")
            shutil.copy(final, str(final_path))
    else:
        shutil.copy(final, str(final_path))

    print(f"\n✓ OBRA FINAL: {final_path}")

    if report:
        _print_report(plan_copy, bank)

    return str(final_path)


def cmd_assemble(args) -> List[str]:
    """
    Ensambla la obra desde banco y plan existentes.
    Con --seeds s1 s2 s3 genera una variación por semilla,
    cada una en su propio subdirectorio y con sufijo _sXXXXXX en el nombre.
    Sin --seeds usa una única semilla (--seed, por defecto 42).
    """
    bank = _load_bank(args.bank)
    plan = _load_plan(args.plan)

    out_dir = Path(getattr(args, "out_dir", "./mosaic_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolver lista de semillas
    seeds_arg = getattr(args, "seeds", None) or []
    if seeds_arg:
        seeds = [int(s) for s in seeds_arg]
    else:
        seeds = [int(getattr(args, "seed", 42))]

    output_base = getattr(args, "output", "mosaic_output.mid")
    stem        = Path(output_base).stem
    ext         = Path(output_base).suffix or ".mid"
    multi       = len(seeds) > 1

    results = []
    for i, seed in enumerate(seeds, 1):
        if multi:
            sep = "═" * 60
            print(f"\n{sep}")
            print(f"  Variación {i}/{len(seeds)}  (seed={seed})")
            print(sep)
            output_name = f"{stem}_s{seed:06d}{ext}"
        else:
            output_name = output_base

        path = _run_single_assemble(
            bank=bank,
            plan=plan,
            seed=seed,
            out_dir=out_dir,
            output_name=output_name,
            stitch_mode=getattr(args, "stitch", "auto"),
            n_candidates=getattr(args, "candidates", 5),
            diversity=getattr(args, "diversity", 0.4),
            allow_variations=getattr(args, "allow_variations", False),
            bridge_bars=getattr(args, "bridge_bars", 2),
            reharmonize=getattr(args, "reharmonize", False),
            style_unify_file=getattr(args, "style_unify", None),
            no_percussion=getattr(args, "no_percussion", False),
            verbose=getattr(args, "verbose", False),
            report=getattr(args, "report", False),
        )
        results.append(path)

    if multi:
        print(f"\n{'═'*60}")
        print(f"  {len(results)} variaciones generadas:")
        for p in results:
            print(f"    {p}")
        print(f"{'═'*60}")

    return results

    return str(final_path)


# ═══════════════════════════════════════════════════════════════════════════
# COMANDO: STITCH (costura manual)
# ═══════════════════════════════════════════════════════════════════════════

def cmd_stitch(args) -> str:
    """Cose una lista de MIDIs con transiciones automáticas."""
    mid_files = args.midis
    out_dir = Path(getattr(args, "out_dir", "./mosaic_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    stitch_mode = getattr(args, "stitch", "auto")
    target_key = getattr(args, "key", None)
    bridge_bars = getattr(args, "bridge_bars", 2)
    verbose = getattr(args, "verbose", False)
    output_name = getattr(args, "output", "mosaic_stitch.mid")

    # Indexar todos los MIDIs
    fragments = []
    for f in mid_files:
        try:
            frag = index_fragment(f)
            fragments.append(frag)
        except Exception as e:
            print(f"[WARN] no indexado '{f}': {e}")

    if not fragments:
        print("[ERROR] Ningún MIDI válido.")
        sys.exit(1)

    if target_key is None:
        target_key = fragments[0].key

    # Crear slots mínimos
    slots = []
    for i, frag in enumerate(fragments):
        slot = Slot(
            id=f"slot_{i+1:02d}",
            label=frag.id,
            function="phrase",
            bars=frag.bars,
        )
        slot.assigned_fragment = frag.id
        slot.assigned_file = frag.file
        slots.append(slot)

    assembled = assemble_sequence(
        slots=slots,
        bank=fragments,
        stitch=stitch_mode,
        target_key=target_key,
        bridge_bars=bridge_bars,
        out_dir=str(out_dir),
        verbose=verbose,
    )

    final_path = out_dir / output_name
    import shutil
    shutil.copy(assembled, str(final_path))
    print(f"\n✓ COSTURA FINAL: {final_path}")
    return str(final_path)


# ═══════════════════════════════════════════════════════════════════════════
# COMANDO: INSPECT
# ═══════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    """Muestra estadísticas del banco de fragmentos."""
    with open(args.bank, encoding="utf-8") as fh:
        bank_data = json.load(fh)
    fragments = [Fragment(**f) for f in bank_data["fragments"]]

    print(f"\n{'═'*60}")
    print(f" MOSAIC BANK INSPECTOR  —  {len(fragments)} fragmentos")
    print(f"{'═'*60}")
    print(f"{'ID':35s} {'Bars':4s} {'Key':12s} {'T_mean':6s} {'Role':12s} {'Cad':4s}")
    print(f"{'─'*60}")
    for f in sorted(fragments, key=lambda x: x.tension_mean):
        print(f"  {f.id[:33]:33s} {f.bars:4d} {f.key[:12]:12s} "
              f"{f.tension_mean:.2f}   {f.role[:12]:12s} {f.cadence_end}")

    # Resumen por rol
    from collections import Counter
    roles = Counter(f.role for f in fragments)
    styles = Counter(f.style for f in fragments)
    print(f"\n{'─'*60}")
    print(f"  Roles:  {dict(roles)}")
    print(f"  Estilos:{dict(styles)}")
    tension_vals = [f.tension_mean for f in fragments]
    print(f"  Tensión: min={min(tension_vals):.2f}  max={max(tension_vals):.2f}  "
          f"media={np.mean(tension_vals):.2f}")
    print(f"{'═'*60}\n")



# ═══════════════════════════════════════════════════════════════════════════
# COMANDO: INDEX
# ═══════════════════════════════════════════════════════════════════════════

def cmd_index(args):
    """
    Re-indexa un directorio de MIDIs y actualiza (o crea) el banco.

    Casos de uso:
      - Re-indexar banco/bank/ tras añadir MIDIs manualmente
      - Forzar re-cálculo de tensión/rol/cadencia con parámetros nuevos
      - Crear un banco desde un directorio de fragmentos ya segmentados
        sin pasar por harvest (útil si se usa un segmentador externo)
      - Fusionar varios directorios en un único banco con --merge

    Si --bank existe y se pasa --merge, los fragmentos nuevos se añaden
    a los existentes (sin duplicar por ruta de archivo).
    Sin --merge, el banco se sobreescribe completamente.
    """
    import shutil

    midi_dir = Path(args.dir)
    if not midi_dir.exists():
        print(f"[ERROR] Directorio no encontrado: {midi_dir}")
        sys.exit(1)

    out_dir = Path(getattr(args, "out_dir", str(midi_dir.parent)))
    out_dir.mkdir(parents=True, exist_ok=True)

    recursive = getattr(args, "recursive", False)
    merge     = getattr(args, "merge", False)
    bank_path = Path(getattr(args, "bank", str(out_dir / ".mosaic_bank.json")))
    verbose   = getattr(args, "verbose", False)

    # Recoger MIDIs
    pattern = "**/*.mid" if recursive else "*.mid"
    midi_files = sorted(midi_dir.glob(pattern))
    midi_files += sorted(midi_dir.glob(pattern.replace(".mid", ".midi")))
    if not midi_files:
        print(f"[ERROR] No se encontraron MIDIs en '{midi_dir}'")
        sys.exit(1)

    print(f"[INDEX] {len(midi_files)} MIDIs en '{midi_dir}'"
          f"{'  (recursivo)' if recursive else ''}")

    # Cargar banco existente si --merge
    existing: Dict[str, Fragment] = {}
    if merge and bank_path.exists():
        try:
            old_bank = _load_bank(str(bank_path))
            existing = {f.file: f for f in old_bank}
            print(f"[INDEX] Banco existente cargado: {len(existing)} fragmentos")
        except Exception as e:
            print(f"[INDEX] [WARN] No se pudo cargar banco existente: {e}")

    fragments: List[Fragment] = []
    n_new = n_skip = n_err = 0

    for mid_path in midi_files:
        path_str = str(mid_path)
        if merge and path_str in existing:
            fragments.append(existing[path_str])
            n_skip += 1
            continue
        try:
            frag = index_fragment(path_str)
            fragments.append(frag)
            n_new += 1
            if verbose:
                print(f"  ✓  {frag.id:40s}  t={frag.tension_mean:.2f}  "
                      f"role={frag.role:12s}  cad={frag.cadence_end}  "
                      f"key={frag.key}")
        except Exception as e:
            n_err += 1
            print(f"  [WARN] {mid_path.name}: {e}")

    # Guardar banco
    bank_data = {"fragments": [asdict(f) for f in fragments]}
    bank_path.write_text(
        json.dumps(bank_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    merge_str = f"  ({n_skip} existentes conservados)" if merge and n_skip else ""
    print(f"\n[INDEX] {n_new} indexados  {n_err} errores{merge_str}")
    print(f"[INDEX] Banco guardado: {bank_path.resolve()}  "
          f"({len(fragments)} fragmentos total)")

# ═══════════════════════════════════════════════════════════════════════════
# REPORTE
# ═══════════════════════════════════════════════════════════════════════════

def _print_report(plan: MosaicPlan, bank: List[Fragment]):
    bank_by_id = {f.id: f for f in bank}
    print(f"\n{'═'*70}")
    print(f" MOSAIC COMPOSER — REPORTE DE ENSAMBLAJE")
    print(f" Arco: {plan.arc}  |  Tonalidad: {plan.key}  |  Compases: {plan.bars_total}")
    print(f"{'═'*70}")
    for slot in plan.slots:
        frag = bank_by_id.get(slot.assigned_fragment)
        frag_str = f"'{frag.id}' (t={frag.tension_mean:.2f}, {frag.key})" if frag else "—"
        print(f"  [{slot.id}] {slot.label:22s} {slot.bars:3d}c  →  {frag_str}")
    print(f"{'═'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# COMANDO: BUILD (pipeline completo)
# ═══════════════════════════════════════════════════════════════════════════

def cmd_build(args):
    """Pipeline completo: harvest → plan → assemble → polish."""
    out_dir = Path(getattr(args, "out_dir", "./mosaic_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)

    print(f"\n{'═'*60}")
    print(f"  MOSAIC COMPOSER — BUILD")
    print(f"  Corpus: {args.corpus}")
    print(f"  Arco:   {args.arc}  |  Compases: {args.bars}")
    print(f"{'═'*60}\n")

    # [1+2] Harvest
    bank = cmd_harvest(args)

    # [3] Plan
    plan = build_plan(
        arc=args.arc,
        bars_total=args.bars,
        key=getattr(args, "key", "C major"),
        tempo=getattr(args, "tempo", 120.0),
    )

    # Guardar banco con rutas absolutas
    bank_path = out_dir.resolve() / ".mosaic_bank.json"
    bank_data = {"fragments": [asdict(f) for f in bank]}
    bank_text = json.dumps(bank_data, indent=2, ensure_ascii=False)
    bank_path.write_text(bank_text, encoding="utf-8")
    # Verificar escritura del banco
    if not bank_path.read_text(encoding="utf-8").strip():
        print(f"[ERROR] El banco se escribió vacío en '{bank_path}'")
        sys.exit(1)

    # Guardar plan con _save_plan (incluye verificación)
    plan_path = out_dir.resolve() / ".mosaic_plan.json"
    try:
        _save_plan(plan, plan_path)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    args.bank = str(bank_path)
    args.plan = str(plan_path)

    print(f"[BUILD] Banco: {bank_path}")
    print(f"[BUILD] Plan:  {plan_path}")

    # [4+5+6] Assemble
    return cmd_assemble(args)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _check_ecosystem(verbose: bool = False):
    """
    Comprueba qué scripts del ecosistema están disponibles en el mismo
    directorio que mosaic_composer.py. Imprime un resumen siempre visible
    (no depende de --verbose) para que el usuario sepa qué funciones estarán
    activas antes de que empiece el pipeline.

    Distingue tres niveles:
      ✓  disponible como script .py  (subprocess, funciona siempre)
      ~  importable como módulo      (import, solo si no tiene código de nivel superior)
      ✗  no encontrado               (esa función quedará desactivada)
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # (nombre_script, descripción, es_critico)
    tools = [
        ("harvester.py",        "segmentación de corpus",        True),
        ("midi_dna_unified.py", "fusión de ADN musical (blend)", True),
        ("reharmonizer.py",     "reharmonización",               False),
        ("style_transfer.py",   "transferencia de estilo",       False),
        ("variation_engine.py", "generación de variaciones",     False),
        ("morpher.py",          "morphing entre fragmentos",      False),
        ("phrase_builder.py",   "construcción de frases",         False),
        ("completer.py",        "completado de obras",            False),
    ]

    missing_critical = []
    warnings = []

    lines = []
    for script, desc, critical in tools:
        exists = os.path.exists(os.path.join(here, script))
        if exists:
            symbol = "✓"
            status = "disponible"
        else:
            symbol = "✗"
            if critical:
                status = "NO ENCONTRADO  ← funcionalidad reducida"
                missing_critical.append(script)
            else:
                status = "no encontrado  (función desactivada)"
                warnings.append(script)

        lines.append(f"  {symbol}  {script:<26s}  {desc}  [{status}]")

    header = "─" * 72
    err = sys.stderr
    print(file=err)
    print(header, file=err)
    print("  MOSAIC COMPOSER — herramientas del ecosistema", file=err)
    print(header, file=err)
    for line in lines:
        print(line, file=err)
    print(header, file=err)

    if missing_critical:
        print(f"  AVISO: {', '.join(missing_critical)} no encontrado/s.", file=err)
        print(f"         Copia el/los script/s al mismo directorio que mosaic_composer.py", file=err)
        print(f"         para activar las funciones indicadas.", file=err)
    if warnings:
        print(f"  INFO:  Las funciones opcionales desactivadas pueden reactivarse", file=err)
        print(f"         añadiendo los scripts al directorio actual.", file=err)
    if not missing_critical and not warnings:
        print("  Todos los scripts del ecosistema están disponibles.", file=err)
    print(header + "\n", file=err)


def main():
    parser = argparse.ArgumentParser(
        description="MOSAIC COMPOSER — Collage y ensamblaje dramático de fragmentos MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ── build ──
    p_build = sub.add_parser("build", help="Pipeline completo (harvest+plan+assemble)")
    p_build.add_argument("--corpus", required=True, help="Directorio de MIDIs fuente")
    p_build.add_argument("--arc", default="arch",
                         choices=list(ARC_TEMPLATES), help="Arco dramático")
    p_build.add_argument("--bars", type=int, default=64, help="Compases totales")
    p_build.add_argument("--key", default="C major", help="Tonalidad destino")
    p_build.add_argument("--tempo", type=float, default=120.0)
    p_build.add_argument("--stitch", default="auto",
                         choices=["auto","pivot","fade","overlap","silence","bridge","direct"])
    p_build.add_argument("--bridge-bars", type=int, default=2)
    p_build.add_argument("--candidates", type=int, default=5)
    p_build.add_argument("--diversity", type=float, default=0.4)
    p_build.add_argument("--allow-variations", action="store_true")
    p_build.add_argument("--fill-gaps", action="store_true")
    p_build.add_argument("--reharmonize", action="store_true")
    p_build.add_argument("--style-unify", metavar="FILE")
    p_build.add_argument("--out-dir", default="./mosaic_out")
    p_build.add_argument("--output", default="mosaic_output.mid")
    p_build.add_argument("--report", action="store_true")
    p_build.add_argument("--no-percussion", action="store_true",
                         help="Eliminar canal 9 (percusión) del resultado final")
    p_build.add_argument("--verbose", action="store_true")
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--min-bars", type=int, default=2)
    p_build.add_argument("--max-bars", type=int, default=16)
    p_build.add_argument("--variations", nargs="*", metavar="V",
                        help="Variaciones por fragmento: V01 V02 V04 ... o all. "
                             "V01=inversion V02=retrogrado V04=aumentacion "
                             "V05=diminucion V06=ornamentacion V08=transposicion "
                             "V10=ritmica V14=estocastica. "
                             "Ejemplo: --variations V01 V04 V10")

    # ── harvest ──
    p_harvest = sub.add_parser("harvest", help="Segmentar corpus y construir banco")
    p_harvest.add_argument("--corpus", required=True)
    p_harvest.add_argument("--out-dir", default="./mosaic_out")
    p_harvest.add_argument("--min-bars", type=int, default=2)
    p_harvest.add_argument("--max-bars", type=int, default=16)
    p_harvest.add_argument("--variations", nargs="*", metavar="V",
                        help="Variaciones por fragmento: V01 V02 V04 ... o all. "
                             "V01=inversion V02=retrogrado V04=aumentacion "
                             "V05=diminucion V06=ornamentacion V08=transposicion "
                             "V10=ritmica V14=estocastica. "
                             "Ejemplo: --variations V01 V04 V10")
    p_harvest.add_argument("--verbose", action="store_true")

    # ── plan ──
    p_plan = sub.add_parser("plan", help="Diseñar y exportar arco dramático")
    p_plan.add_argument("--arc", default="arch", choices=list(ARC_TEMPLATES))
    p_plan.add_argument("--bars", type=int, default=64)
    p_plan.add_argument("--key", default="C major")
    p_plan.add_argument("--tempo", type=float, default=120.0)
    p_plan.add_argument("--out-dir", default="./mosaic_out")
    p_plan.add_argument("--stdout", action="store_true",
                        help="Escribir JSON en stdout (para uso con bash <(...))."
                             " Los mensajes informativos van a stderr.")

    # ── assemble ──
    p_assemble = sub.add_parser("assemble", help="Ensamblar desde banco y plan existentes")
    p_assemble.add_argument("--bank", required=True)
    p_assemble.add_argument("--plan", required=True)
    p_assemble.add_argument("--stitch", default="auto")
    p_assemble.add_argument("--bridge-bars", type=int, default=2)
    p_assemble.add_argument("--candidates", type=int, default=5)
    p_assemble.add_argument("--diversity", type=float, default=0.4)
    p_assemble.add_argument("--allow-variations", action="store_true")
    p_assemble.add_argument("--fill-gaps", action="store_true",
                            help="Usar completer.py para extender fragmentos cortos")
    p_assemble.add_argument("--reharmonize", action="store_true")
    p_assemble.add_argument("--style-unify", metavar="FILE")
    p_assemble.add_argument("--out-dir", default="./mosaic_out")
    p_assemble.add_argument("--output", default="mosaic_output.mid")
    p_assemble.add_argument("--report", action="store_true")
    p_assemble.add_argument("--verbose", action="store_true")
    p_assemble.add_argument("--seed", type=int, default=42,
                            help="Semilla aleatoria (una sola variación)")
    p_assemble.add_argument("--seeds", nargs="+", metavar="S",
                            help="Lista de semillas para generar múltiples variaciones. "
                                 "Ejemplo: --seeds 42 123 999  genera 3 variaciones. "
                                 "Incompatible con --seed (--seeds tiene prioridad).")
    p_assemble.add_argument("--no-percussion", action="store_true",
                            help="Eliminar canal 9 (percusión) del resultado final")

    # ── stitch ──
    p_stitch = sub.add_parser("stitch", help="Coser lista manual de MIDIs")
    p_stitch.add_argument("midis", nargs="+", help="Archivos MIDI a coser en orden")
    p_stitch.add_argument("--stitch", default="auto")
    p_stitch.add_argument("--key")
    p_stitch.add_argument("--bridge-bars", type=int, default=2)
    p_stitch.add_argument("--out-dir", default="./mosaic_out")
    p_stitch.add_argument("--output", default="mosaic_stitch.mid")
    p_stitch.add_argument("--verbose", action="store_true")

    # ── inspect ──
    p_index = sub.add_parser("index",
                             help="Indexar/re-indexar un directorio de MIDIs y crear o actualizar el banco")
    p_index.add_argument("dir",
                         help="Directorio con los MIDIs a indexar (p.ej. banco/bank/)")
    p_index.add_argument("--bank", default=None,
                         help="Ruta del banco de salida (por defecto: <dir>/../.mosaic_bank.json)")
    p_index.add_argument("--out-dir", default=None,
                         help="Directorio donde guardar el banco (ignorado si --bank es absoluto)")
    p_index.add_argument("--recursive", action="store_true",
                         help="Buscar MIDIs en subdirectorios recursivamente")
    p_index.add_argument("--merge", action="store_true",
                         help="Añadir al banco existente en lugar de sobreescribirlo. "
                              "Los fragmentos ya indexados (misma ruta) se conservan sin re-calcular.")
    p_index.add_argument("--verbose", action="store_true",
                         help="Mostrar detalles de cada fragmento indexado")

    p_inspect = sub.add_parser("inspect", help="Analizar banco de fragmentos")
    p_inspect.add_argument("--bank", required=True)

    args = parser.parse_args()

    _check_ecosystem()

    dispatch = {
        "build":   cmd_build,
        "harvest": cmd_harvest,
        "index":   cmd_index,
        "plan":    cmd_plan,
        "assemble":cmd_assemble,
        "stitch":  cmd_stitch,
        "inspect": cmd_inspect,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
