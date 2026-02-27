#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MUTATOR  v1.0                                        ║
║         Mutación semántica de MIDIs guiada por lenguaje natural              ║
║                                                                              ║
║  Transforma instrucciones en lenguaje natural ("más oscuro", "añade         ║
║  urgencia en el clímax", "que suene como lluvia") en parámetros concretos   ║
║  del pipeline midi_dna_unified y los aplica sobre un MIDI existente.        ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] Lee el MIDI fuente y su fingerprint (estado actual)                     ║
║  [2] Parsea la instrucción: diccionario semántico + LLM (fallback/enriquece) ║
║  [3] Resuelve deltas relativos sobre el estado actual                        ║
║  [4] Aplica parámetros llamando a variation_engine / morpher / tension       ║
║  [5] Verifica con mscz2vec que el resultado se movió en la dirección pedida  ║
║  [6] Guarda historial de mutaciones (árbol, revertible)                      ║
║                                                                              ║
║  USO:                                                                        ║
║    python mutator.py obra.mid "más oscuro"                                   ║
║    python mutator.py obra.mid "añade urgencia en los últimos 8 compases"    ║
║    python mutator.py obra.mid "hazlo más lento y melancólico"               ║
║    python mutator.py obra.mid "quiero que suene como si lloviera"           ║
║    python mutator.py obra.mid --interactive                                  ║
║    python mutator.py obra.mid --history                                      ║
║    python mutator.py obra.mid --revert 2                                     ║
║    python mutator.py obra.mid --tree                                         ║
║    python mutator.py obra.mid --branch mut_003.mid "más luminoso"           ║
║    python mutator.py obra.mid --no-llm "oscuro"                             ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    instruction     Instrucción en lenguaje natural (entre comillas)          ║
║    --interactive   Modo conversacional: itera hasta que estés satisfecho     ║
║    --intensity F   Intensidad global de la mutación 0-1 (default: 0.7)      ║
║    --bars N        Compases de salida (default: auto desde fuente)           ║
║    --scope S:E     Aplicar solo entre compases S y E (ej: --scope 8:16)     ║
║    --output FILE   MIDI de salida (default: auto desde historial)            ║
║    --history       Mostrar historial de mutaciones de esta obra              ║
║    --revert N      Revertir a la mutación N del historial                    ║
║    --tree          Visualizar el árbol de mutaciones                         ║
║    --branch MIDI   Partir desde un MIDI específico del historial             ║
║    --no-llm        Usar solo el diccionario semántico (sin API)              ║
║    --llm-provider  Proveedor LLM: anthropic (default) | openai               ║
║    --llm-model     Modelo específico a usar (default: auto según proveedor)  ║
║    --api-key KEY   API key del proveedor (o vars de entorno)                 ║
║                    Anthropic: ANTHROPIC_API_KEY                              ║
║                    OpenAI:    OPENAI_API_KEY                                 ║
║    --dry-run       Mostrar parámetros resueltos sin generar MIDI             ║
║    --verbose       Informe detallado de decisiones                           ║
║    --listen        Reproducir el resultado (requiere pygame)                 ║
║    --verify        Verificar con mscz2vec que la mutación fue en la          ║
║                    dirección correcta (requiere music21)                     ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                   ║
║  LLM (opcional): anthropic  →  pip install anthropic                        ║
║                  openai     →  pip install openai                            ║
║  SCRIPTS DEL PIPELINE: variation_engine.py, morpher.py, tension_designer.py ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import time
import copy
import re
import math
import textwrap
import tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0"

# Modos escalares válidos (los que acepta midi_dna_unified)
VALID_MODES = ["major", "minor", "dorian", "phrygian", "lydian",
               "mixolydian", "locrian", "harmonic_minor", "melodic_minor"]

# Tonalidades cromáticas
TONICS = ["C", "C#", "Db", "D", "D#", "Eb", "E", "F",
          "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"]

# "Oscuridad" relativa de los modos (0=más brillante, 1=más oscuro)
MODE_DARKNESS = {
    "lydian": 0.0,
    "major": 0.15,
    "mixolydian": 0.3,
    "dorian": 0.45,
    "minor": 0.6,
    "harmonic_minor": 0.65,
    "melodic_minor": 0.65,
    "phrygian": 0.8,
    "locrian": 1.0,
}

# Distancia de los modos en el espectro (para navegación incremental)
MODE_DARKNESS_SORTED = sorted(MODE_DARKNESS.items(), key=lambda x: x[1])

# Relaciones de quinta entre tonalidades (para navegación tonal)
CIRCLE_OF_FIFTHS = ["C", "G", "D", "A", "E", "B", "F#",
                    "Db", "Ab", "Eb", "Bb", "F"]

# ══════════════════════════════════════════════════════════════════════════════
#  MAPA SEMÁNTICO: instrucción → deltas de parámetros
#  Cada entrada puede tener:
#    key_mode_delta: float -1..+1 (hacia oscuro/brillante)
#    key_tonic_delta: int en círculo de quintas (+ = subir quinta, - = bajar)
#    tempo_delta: float como factor multiplicativo (1.0 = sin cambio)
#    tension_delta: float -1..+1
#    register_delta: float -1..+1
#    density_delta: float -1..+1
#    harmony_complexity_delta: float -1..+1
#    swing_delta: float -1..+1
#    velocity_mean_delta: float -1..+1 (dinámica general)
#    velocity_var_delta: float -1..+1 (expresividad dinámica)
#    ornament_density_delta: float -1..+1
#    variation_type: str  (V01-V15 de variation_engine)
#    tension_shape: str (preset de tension_designer)
#    aliases: list[str]  (sinónimos)
# ══════════════════════════════════════════════════════════════════════════════

SEMANTIC_MAP = {
    # ── CARÁCTER TONAL ────────────────────────────────────────────────────────
    "oscuro": {
        "aliases": ["dark", "sombrío", "sombrio", "tenebroso", "siniestro",
                    "lúgubre", "lugubre", "fúnebre", "funesto"],
        "key_mode_delta": +0.6,         # hacia modos más oscuros
        "tension_delta": +0.2,
        "register_delta": -0.2,
        "velocity_mean_delta": -0.1,
        "harmony_complexity_delta": +0.2,
    },
    "muy oscuro": {
        "aliases": ["muy sombrio", "muy tenebroso", "muy oscuro"],
        "key_mode_delta": +0.9,
        "tension_delta": +0.35,
        "register_delta": -0.3,
        "velocity_mean_delta": -0.15,
        "harmony_complexity_delta": +0.35,
    },
    "luminoso": {
        "aliases": ["brillante", "claro", "alegre", "light", "bright",
                    "soleado", "radiante", "optimista"],
        "key_mode_delta": -0.6,         # hacia modos más brillantes
        "tension_delta": -0.15,
        "register_delta": +0.15,
        "velocity_mean_delta": +0.1,
        "harmony_complexity_delta": -0.1,
    },
    "melancólico": {
        "aliases": ["melancolico", "melancholic", "nostálgico", "nostalgico",
                    "añoranza", "anoranza", "triste"],
        "key_mode_delta": +0.4,
        "tempo_delta": 0.88,
        "tension_delta": +0.1,
        "register_delta": -0.1,
        "ornament_density_delta": +0.2,  # suspiros melódicos
        "velocity_var_delta": +0.15,     # más expresivo
    },
    "épico": {
        "aliases": ["epico", "heroico", "majestuoso", "grandioso",
                    "poderoso", "epic", "heroic"],
        "key_mode_delta": -0.2,
        "tension_delta": +0.3,
        "register_delta": +0.2,
        "velocity_mean_delta": +0.25,
        "tempo_delta": 1.05,
        "density_delta": +0.2,
        "harmony_complexity_delta": +0.15,
    },
    "misterioso": {
        "aliases": ["mysterious", "enigmático", "enigmatico", "inquietante",
                    "ambiguo"],
        "key_mode_delta": +0.5,
        "tension_delta": +0.25,
        "harmony_complexity_delta": +0.4,
        "density_delta": -0.15,
        "velocity_var_delta": +0.2,
        "ornament_density_delta": +0.1,
    },
    "romántico": {
        "aliases": ["romantico", "romantic", "apasionado", "expresivo",
                    "lírico", "lirico"],
        "key_mode_delta": +0.2,
        "ornament_density_delta": +0.35,
        "velocity_var_delta": +0.3,
        "density_delta": -0.1,
        "harmony_complexity_delta": +0.2,
        "swing_delta": +0.1,
    },

    # ── ENERGÍA / TEMPO ───────────────────────────────────────────────────────
    "urgente": {
        "aliases": ["urgent", "apremiante", "angustioso", "ansioso",
                    "nervioso", "frenético", "frenetico"],
        "tempo_delta": 1.15,
        "density_delta": +0.3,
        "tension_delta": +0.25,
        "register_delta": +0.1,
        "velocity_mean_delta": +0.2,
        "velocity_var_delta": +0.2,
    },
    "tranquilo": {
        "aliases": ["tranquil", "sereno", "calm", "calmado", "quieto",
                    "reposado", "plácido", "placido", "pacífico", "pacifico"],
        "tempo_delta": 0.82,
        "density_delta": -0.3,
        "tension_delta": -0.2,
        "velocity_mean_delta": -0.1,
        "velocity_var_delta": -0.1,
    },
    "más rápido": {
        "aliases": ["mas rapido", "faster", "accelerar", "acelerar",
                    "vivace", "presto", "allegro"],
        "tempo_delta": 1.2,
        "density_delta": +0.15,
    },
    "más lento": {
        "aliases": ["mas lento", "slower", "ralentizar", "retardar",
                    "adagio", "lento", "largo"],
        "tempo_delta": 0.78,
        "density_delta": -0.1,
    },
    "enérgico": {
        "aliases": ["energico", "energetic", "vigoroso", "potente",
                    "intenso"],
        "tempo_delta": 1.08,
        "density_delta": +0.2,
        "velocity_mean_delta": +0.2,
        "tension_delta": +0.1,
    },

    # ── TENSIÓN / ARCO ────────────────────────────────────────────────────────
    "tenso": {
        "aliases": ["tenso", "tenso", "tense", "cargado", "opresivo"],
        "tension_delta": +0.35,
        "harmony_complexity_delta": +0.25,
        "velocity_var_delta": +0.15,
    },
    "relajado": {
        "aliases": ["relaxed", "suelto", "distendido", "descanso"],
        "tension_delta": -0.3,
        "harmony_complexity_delta": -0.2,
        "density_delta": -0.15,
    },
    "climax": {
        "aliases": ["clímax", "climax", "culmen", "punto más alto",
                    "punto mas alto", "apogeo"],
        "tension_shape": "late_climax",
        "tension_delta": +0.2,
        "register_delta": +0.2,
        "velocity_mean_delta": +0.15,
    },
    "arco": {
        "aliases": ["arch", "arco dramático", "arco dramatico",
                    "subir y bajar"],
        "tension_shape": "arch",
    },

    # ── DENSIDAD / TEXTURA ────────────────────────────────────────────────────
    "denso": {
        "aliases": ["dense", "tupido", "lleno", "saturado", "contrapuntístico",
                    "contrapuntistico"],
        "density_delta": +0.35,
        "harmony_complexity_delta": +0.15,
    },
    "sparse": {
        "aliases": ["esparso", "escaso", "vacío", "vacio", "ligero",
                    "minimalista", "minimal", "desnudo"],
        "density_delta": -0.35,
        "harmony_complexity_delta": -0.2,
        "ornament_density_delta": -0.2,
    },
    "más voces": {
        "aliases": ["mas voces", "más contrapunto", "mas contrapunto",
                    "contrapunto", "polifónico", "polifonico"],
        "density_delta": +0.2,
        "harmony_complexity_delta": +0.3,
        "variation_type": "V15",        # contrapuntística
    },

    # ── REGISTRO ──────────────────────────────────────────────────────────────
    "más grave": {
        "aliases": ["mas grave", "bajo", "low", "profundo", "grave",
                    "bajo registro"],
        "register_delta": -0.4,
    },
    "más agudo": {
        "aliases": ["mas agudo", "alto", "high", "agudo", "brillante registro",
                    "alto registro"],
        "register_delta": +0.4,
    },

    # ── GROOVE / RITMO ────────────────────────────────────────────────────────
    "swing": {
        "aliases": ["swingado", "jazz", "sincopado", "groovy"],
        "swing_delta": +0.5,
        "harmony_complexity_delta": +0.1,
    },
    "recto": {
        "aliases": ["straight", "sin swing", "cuadrado", "mecánico",
                    "mecanico"],
        "swing_delta": -0.5,
    },
    "rítmico": {
        "aliases": ["ritmico", "rhythmic", "percusivo", "pulsante",
                    "enérgico rítmico"],
        "density_delta": +0.2,
        "swing_delta": +0.1,
        "velocity_var_delta": +0.2,
    },

    # ── ORNAMENTACIÓN / EXPRESIÓN ─────────────────────────────────────────────
    "ornamentado": {
        "aliases": ["ornamental", "floreado", "adornado", "barroco",
                    "coloraturas"],
        "ornament_density_delta": +0.5,
        "variation_type": "V06",
    },
    "simple": {
        "aliases": ["sencillo", "limpio", "clean", "clásico simple",
                    "directo"],
        "ornament_density_delta": -0.4,
        "harmony_complexity_delta": -0.2,
        "density_delta": -0.1,
    },
    "expresivo": {
        "aliases": ["expressive", "dinámico", "dinamico", "vivo",
                    "con sentimiento"],
        "velocity_var_delta": +0.4,
        "ornament_density_delta": +0.2,
    },

    # ── IMÁGENES / METÁFORAS VISUALES ─────────────────────────────────────────
    "lluvia": {
        "aliases": ["rain", "llovizna", "tormenta leve", "gotas"],
        "density_delta": +0.3,
        "register_delta": +0.2,
        "velocity_var_delta": +0.35,
        "swing_delta": +0.1,
        "ornament_density_delta": +0.2,
        "tempo_delta": 0.95,
    },
    "tormenta": {
        "aliases": ["storm", "tempestad", "truenos", "caos",
                    "turbulento", "violento"],
        "density_delta": +0.45,
        "tension_delta": +0.4,
        "tempo_delta": 1.1,
        "velocity_mean_delta": +0.3,
        "velocity_var_delta": +0.3,
        "register_delta": +0.1,
        "harmony_complexity_delta": +0.3,
    },
    "amanecer": {
        "aliases": ["alba", "dawn", "aurora", "despertar", "opening"],
        "key_mode_delta": -0.4,
        "register_delta": +0.1,
        "tension_shape": "crescendo",
        "velocity_mean_delta": +0.1,
        "tempo_delta": 1.05,
    },
    "noche": {
        "aliases": ["night", "nocturno", "oscuridad", "medianoche"],
        "key_mode_delta": +0.5,
        "register_delta": -0.15,
        "tension_delta": +0.1,
        "density_delta": -0.1,
        "velocity_mean_delta": -0.1,
    },
    "mar": {
        "aliases": ["ocean", "oceano", "olas", "waves", "marítimo",
                    "maritimo"],
        "tension_shape": "wave",
        "swing_delta": +0.2,
        "density_delta": +0.1,
        "register_delta": -0.05,
    },
    "vacío": {
        "aliases": ["vacio", "empty", "soledad", "alone", "abandono",
                    "desierto"],
        "density_delta": -0.45,
        "register_delta": -0.1,
        "tension_delta": -0.1,
        "ornament_density_delta": -0.3,
        "velocity_mean_delta": -0.15,
    },
    "vals": {
        "aliases": ["waltz", "valse", "dance", "baile"],
        "swing_delta": +0.3,
        "density_delta": +0.1,
        "tempo_delta": 1.05,
        "variation_type": "V07",
    },

    # ── VARIACIONES CLÁSICAS ──────────────────────────────────────────────────
    "al revés": {
        "aliases": ["al reves", "invertido", "retrograde", "retrógrado",
                    "retrogrado"],
        "variation_type": "V02",
    },
    "invertido": {
        "aliases": ["inversion", "inversión", "reflejo", "espejo"],
        "variation_type": "V01",
    },
    "aumentado": {
        "aliases": ["más lento rítmicamente", "augmentation", "aumentación"],
        "variation_type": "V04",
    },
    "disminuido": {
        "aliases": ["diminución", "diminucion", "más rápido rítmicamente"],
        "variation_type": "V05",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT PARA EL LLM
# ══════════════════════════════════════════════════════════════════════════════

LLM_SYSTEM_PROMPT = """Eres un intérprete musical experto. Tu tarea es traducir instrucciones \
en lenguaje natural a parámetros musicales estructurados para un sistema de composición MIDI.

Debes devolver EXCLUSIVAMENTE un objeto JSON válido (sin texto adicional, sin backticks, \
sin explicaciones antes o después). El JSON tiene esta estructura:

{
  "matched_concepts": ["concepto1", "concepto2"],
  "params": {
    "key_mode_delta": <float -1..1, positivo=más oscuro, negativo=más brillante>,
    "key_tonic_delta": <int -6..6, círculo de quintas, positivo=subir quinta>,
    "tempo_delta": <float, factor multiplicativo, 1.0=sin cambio, 0.8=20% más lento>,
    "tension_delta": <float -1..1>,
    "register_delta": <float -1..1, positivo=más agudo>,
    "density_delta": <float -1..1, densidad rítmica>,
    "harmony_complexity_delta": <float -1..1>,
    "swing_delta": <float -1..1>,
    "velocity_mean_delta": <float -1..1, volumen medio>,
    "velocity_var_delta": <float -1..1, expresividad dinámica>,
    "ornament_density_delta": <float -1..1>,
    "variation_type": <"V01"-"V15" o null>,
    "tension_shape": <"arch"|"crescendo"|"decrescendo"|"plateau"|"late_climax"|"wave"|"neutral"|null>,
    "scope_start_frac": <float 0..1, inicio del efecto como fracción de la obra>,
    "scope_end_frac": <float 0..1, fin del efecto>
  },
  "reasoning": "<explicación breve de las decisiones musicales>",
  "confidence": <float 0..1>
}

Omite campos que sean 0 o null. Incluye SOLO los parámetros que realmente cambian.

Variaciones disponibles (variation_type):
V01=Inversión melódica, V02=Retrógrado, V03=Inversión retrógrada,
V04=Aumentación rítmica, V05=Diminución rítmica, V06=Ornamentación,
V07=Nuevo acompañamiento, V08=Transposición, V09=Cambio modal,
V10=Variación rítmica, V11=Reharmonización, V12=Variación textural,
V13=Arco emocional distinto, V14=Estocástica, V15=Contrapunto.
"""

def build_llm_prompt(instruction: str, current_state: dict) -> str:
    """Construye el prompt de usuario con contexto del MIDI actual."""
    state_lines = []
    if current_state.get("key_tonic"):
        state_lines.append(f"- Tonalidad: {current_state['key_tonic']} {current_state.get('key_mode','major')}")
    if current_state.get("tempo_bpm"):
        state_lines.append(f"- Tempo: {current_state['tempo_bpm']:.0f} BPM")
    if current_state.get("tension_mean") is not None:
        state_lines.append(f"- Tensión media: {current_state['tension_mean']:.2f}")
    if current_state.get("register"):
        state_lines.append(f"- Registro: {current_state['register']}")
    if current_state.get("density") is not None:
        state_lines.append(f"- Densidad rítmica: {current_state['density']:.2f}")
    if current_state.get("n_bars"):
        state_lines.append(f"- Compases: {current_state['n_bars']}")

    state_str = "\n".join(state_lines) if state_lines else "(estado no disponible)"
    return f"""Estado actual del MIDI:
{state_str}

Instrucción del compositor: "{instruction}"

Traduce esta instrucción a parámetros JSON."""


# ══════════════════════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL: MutationState
# ══════════════════════════════════════════════════════════════════════════════

class MutationState:
    """
    Gestiona el estado actual del MIDI y su historial de mutaciones.
    Carga/guarda un archivo .mutator.json junto al MIDI original.
    """

    def __init__(self, source_midi: str):
        self.source_midi = str(Path(source_midi).resolve())
        self.history_path = str(Path(source_midi).with_suffix(".mutator.json"))
        self.history = self._load_history()

    def _load_history(self) -> dict:
        if Path(self.history_path).exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "source": self.source_midi,
            "created": datetime.now().isoformat(),
            "mutations": []
        }

    def save_history(self):
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

    def add_mutation(self, instruction: str, params: dict, result_path: str,
                     parent_path: str, reasoning: str = ""):
        entry = {
            "id": len(self.history["mutations"]) + 1,
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction,
            "parent": parent_path,
            "result": result_path,
            "params": params,
            "reasoning": reasoning,
        }
        self.history["mutations"].append(entry)
        self.save_history()
        return entry["id"]

    def get_mutation(self, n: int) -> dict | None:
        for m in self.history["mutations"]:
            if m["id"] == n:
                return m
        return None

    def print_history(self):
        print(f"\n{'─'*60}")
        print(f"  Historial de mutaciones: {Path(self.source_midi).name}")
        print(f"{'─'*60}")
        if not self.history["mutations"]:
            print("  (sin mutaciones)")
            return
        for m in self.history["mutations"]:
            ts = m["timestamp"][:16].replace("T", " ")
            parent_short = Path(m["parent"]).name
            result_short = Path(m["result"]).name
            print(f"  [{m['id']:02d}] {ts}  ←{parent_short}")
            print(f"       \"{m['instruction']}\"")
            print(f"       → {result_short}")
            if m.get("reasoning"):
                print(f"       Razón: {m['reasoning'][:80]}")
            print()

    def print_tree(self):
        """Visualización ASCII del árbol de mutaciones."""
        print(f"\n{'─'*60}")
        print(f"  Árbol de mutaciones")
        print(f"{'─'*60}")
        source_name = Path(self.source_midi).name
        print(f"  ◉ {source_name}  [ORIGEN]")
        _build_tree(self.history["mutations"], self.source_midi, "  ")


def _build_tree(mutations: list, parent: str, indent: str):
    children = [m for m in mutations if m["parent"] == parent]
    for i, m in enumerate(children):
        is_last = (i == len(children) - 1)
        connector = "└─" if is_last else "├─"
        result_name = Path(m["result"]).name
        instr_short = m["instruction"][:40]
        print(f"{indent}{connector} [{m['id']:02d}] {result_name}")
        print(f"{indent}{'  ' if is_last else '│ '}     \"{instr_short}\"")
        new_indent = indent + ("   " if is_last else "│  ")
        _build_tree(mutations, m["result"], new_indent)


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DEL ESTADO ACTUAL
# ══════════════════════════════════════════════════════════════════════════════

def read_midi_state(midi_path: str, verbose: bool = False) -> dict:
    """
    Extrae el estado musical del MIDI usando:
    1. El fingerprint .fingerprint.json si existe.
    2. Análisis directo con mido como fallback.
    """
    state = {
        "midi_path": midi_path,
        "key_tonic": "C",
        "key_mode": "major",
        "tempo_bpm": 120.0,
        "n_bars": 16,
        "tension_mean": 0.5,
        "register": "mid",
        "density": 0.5,
        "harmony_complexity": 0.3,
        "swing": 0.1,
    }

    # Intentar cargar fingerprint
    fp_candidates = [
        str(Path(midi_path).with_suffix("").with_suffix(".fingerprint.json")),
        str(Path(midi_path)) + ".fingerprint.json",
    ]
    for fp_path in fp_candidates:
        if Path(fp_path).exists():
            try:
                with open(fp_path, "r") as f:
                    fp = json.load(f)
                meta = fp.get("meta", {})
                state["key_tonic"]   = meta.get("key_tonic", state["key_tonic"])
                state["key_mode"]    = meta.get("key_mode", state["key_mode"])
                state["tempo_bpm"]   = float(meta.get("tempo_bpm", state["tempo_bpm"]))
                state["n_bars"]      = int(meta.get("n_bars", state["n_bars"]))
                tc = fp.get("tension_curve", {})
                vals = tc.get("values", None)
                if vals:
                    state["tension_mean"] = float(np.mean(vals))
                reg_exit = fp.get("exit", {}).get("melody_register", "mid")
                state["register"] = reg_exit
                if verbose:
                    print(f"  [estado] Fingerprint cargado: {fp_path}")
                return state
            except Exception as e:
                if verbose:
                    print(f"  [estado] No se pudo leer fingerprint: {e}")
                break

    # Fallback: análisis con mido
    try:
        import mido
        mid = mido.MidiFile(midi_path)
        tempo = 500000
        notes = []
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                if msg.type == "note_on" and msg.velocity > 0:
                    notes.append(msg.note)
        state["tempo_bpm"] = round(60_000_000 / tempo, 1)
        if notes:
            mean_note = np.mean(notes)
            if mean_note < 52:
                state["register"] = "low"
            elif mean_note > 72:
                state["register"] = "high"
            else:
                state["register"] = "mid"
            state["density"] = min(1.0, len(notes) / (state["n_bars"] * 8))
        if verbose:
            print(f"  [estado] Analizado con mido (sin fingerprint)")
    except Exception as e:
        if verbose:
            print(f"  [estado] Análisis mido falló: {e}")

    return state


# ══════════════════════════════════════════════════════════════════════════════
#  PARSING SEMÁNTICO
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Normaliza texto para búsqueda: lowercase, sin tildes, sin signos."""
    replacements = {"á":"a","é":"e","í":"i","ó":"o","ú":"u",
                    "à":"a","è":"e","ì":"i","ò":"o","ù":"u",
                    "ñ":"n","ü":"u"}
    t = text.lower().strip()
    for k, v in replacements.items():
        t = t.replace(k, v)
    return t


def parse_scope(instruction: str, n_bars: int) -> tuple[int | None, int | None]:
    """
    Extrae scope temporal de la instrucción si existe.
    Ej: "en los últimos 8 compases" → (n_bars-8, n_bars)
        "en los primeros 4 compases" → (0, 4)
        "entre los compases 8 y 16"  → (8, 16)
        "en el compás 12"            → (12, 13)
    Retorna (start_bar, end_bar) o (None, None).
    """
    t = instruction.lower()

    # "entre compases X y Y"
    m = re.search(r"entre (?:los )?comp[aá]ses? (\d+) y (\d+)", t)
    if m:
        return int(m.group(1)), int(m.group(2))

    # "en el compás X"
    m = re.search(r"en (?:el )?comp[aá]s (\d+)", t)
    if m:
        bar = int(m.group(1))
        return bar, min(bar + 1, n_bars)

    # "en los primeros X compases"
    m = re.search(r"(?:primeros|primer) (\d+) comp[aá]ses?", t)
    if m:
        return 0, int(m.group(1))

    # "en los últimos X compases"
    m = re.search(r"(?:[uú]ltimos|[uú]ltimo) (\d+) comp[aá]ses?", t)
    if m:
        n = int(m.group(1))
        return max(0, n_bars - n), n_bars

    # "en la segunda mitad" / "en la primera mitad"
    if re.search(r"segunda mitad", t):
        return n_bars // 2, n_bars
    if re.search(r"primera mitad", t):
        return 0, n_bars // 2

    # "en el clímax" → último tercio
    if re.search(r"cl[íi]max|climax|culmen|apogeo|punto m[aá]s alto", t):
        return int(n_bars * 0.65), n_bars

    # "al final"
    if re.search(r"\bal final\b|\ben el final\b", t):
        return int(n_bars * 0.75), n_bars

    # "al inicio" / "al principio"
    if re.search(r"\bal inicio\b|\bal principio\b|\ben el inicio\b", t):
        return 0, int(n_bars * 0.25)

    return None, None


def parse_intensity_modifier(instruction: str) -> float:
    """Extrae modificador de intensidad de la instrucción."""
    t = normalize_text(instruction)

    if re.search(r"\bmuy\b|\bmucho\b|\bbastante\b|\bextremadamente\b|\btotalmente\b", t):
        return 1.3
    if re.search(r"\bun poco\b|\bligeramente\b|\blevemente\b|\bsuavemente\b|\bpoco\b", t):
        return 0.5
    if re.search(r"\bmucho mas\b|\bmucho más\b|\bbastante mas\b", t):
        return 1.5
    return 1.0


def dictionary_parse(instruction: str, verbose: bool = False) -> dict:
    """
    Busca coincidencias en el SEMANTIC_MAP.
    Devuelve un dict de parámetros merged (puede haber múltiples conceptos).
    """
    t_norm = normalize_text(instruction)
    matched = []
    params_merged = {}

    for concept, data in SEMANTIC_MAP.items():
        # Buscar el concepto o sus aliases en la instrucción
        terms = [normalize_text(concept)] + [normalize_text(a) for a in data.get("aliases", [])]
        for term in terms:
            if term in t_norm:
                matched.append(concept)
                # Merge params (si hay conflicto, promediamos)
                for k, v in data.items():
                    if k == "aliases":
                        continue
                    if k in params_merged and isinstance(v, (int, float)):
                        params_merged[k] = (params_merged[k] + v) / 2
                    else:
                        params_merged[k] = v
                break  # solo matchear una vez por concepto

    if verbose:
        if matched:
            print(f"  [dict] Conceptos: {matched}")
        else:
            print(f"  [dict] Sin coincidencias directas")

    return params_merged, matched


def _llm_debug_block(label: str, text: str):
    """Imprime un bloque de debug con borde visual."""
    bar = "─" * max(0, 44 - len(label))
    print(f"\n  ┌─ [llm:debug] {label} {bar}")
    for line in text.splitlines():
        print(f"  │ {line}")
    print(f"  └{'─' * 58}")


def _parse_llm_response(raw: str, verbose: bool,
                        debug: bool = False) -> tuple[dict, list, str]:
    """Parsea la respuesta JSON del LLM (compartido entre proveedores)."""
    if debug:
        _llm_debug_block("RESPUESTA RAW", raw)

    raw_clean = re.sub(r"```(?:json)?", "", raw).strip()
    data = json.loads(raw_clean)
    params     = data.get("params", {})
    matched    = data.get("matched_concepts", [])
    reasoning  = data.get("reasoning", "")
    confidence = data.get("confidence", 0.5)

    if verbose:
        print(f"  [llm] Conceptos: {matched}")
        print(f"  [llm] Razonamiento: {reasoning}")
        print(f"  [llm] Confianza: {confidence:.2f}")
    return params, matched, reasoning


def _llm_parse_anthropic(prompt: str, api_key: str,
                          model: str | None, verbose: bool,
                          debug: bool = False) -> tuple[dict, list, str]:
    """Backend Anthropic (claude-sonnet-4-6 por defecto)."""
    try:
        import anthropic
    except ImportError:
        if verbose:
            print("  [llm:anthropic] No instalado → pip install anthropic")
        return {}, [], ""

    model = model or "claude-sonnet-4-6"
    if debug:
        _llm_debug_block("PROMPT → anthropic/" + model, prompt)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=800,
            system=LLM_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        raw = msg.content[0].text.strip()
        return _parse_llm_response(raw, verbose, debug=debug)
    except Exception as e:
        if verbose:
            print(f"  [llm:anthropic] Error: {e}")
        return {}, [], ""


def _llm_parse_openai(prompt: str, api_key: str,
                       model: str | None, verbose: bool,
                       debug: bool = False) -> tuple[dict, list, str]:
    """Backend OpenAI (gpt-4o por defecto)."""
    try:
        import openai
    except ImportError:
        if verbose:
            print("  [llm:openai] No instalado → pip install openai")
        return {}, [], ""

    model = model or "gpt-4o"
    if debug:
        _llm_debug_block("PROMPT → openai/" + model, prompt)
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=800,
            response_format={"type": "json_object"},  # JSON mode nativo
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        return _parse_llm_response(raw, verbose, debug=debug)
    except Exception as e:
        if verbose:
            print(f"  [llm:openai] Error: {e}")
        return {}, [], ""


# Registro de proveedores: nombre → (función, env_var, modelos_recomendados)
LLM_PROVIDERS = {
    "anthropic": (
        _llm_parse_anthropic,
        "ANTHROPIC_API_KEY",
        ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"],
    ),
    "openai": (
        _llm_parse_openai,
        "OPENAI_API_KEY",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
    ),
}


def llm_parse(instruction: str, current_state: dict,
              api_key: str | None = None,
              provider: str = "anthropic",
              model: str | None = None,
              verbose: bool = False,
              debug: bool = False) -> tuple[dict, list, str]:
    """
    Parsea una instrucción usando el proveedor LLM configurado.

    Proveedores soportados: anthropic, openai
    La API key se toma de:
      1. El argumento api_key
      2. La variable de entorno del proveedor (ANTHROPIC_API_KEY / OPENAI_API_KEY)
    Si no hay key disponible, retorna vacío sin error fatal.
    Con debug=True imprime el prompt enviado y la respuesta raw recibida.
    """
    provider = provider.lower().strip()

    if provider not in LLM_PROVIDERS:
        if verbose:
            print(f"  [llm] Proveedor desconocido: '{provider}'. "
                  f"Opciones: {list(LLM_PROVIDERS.keys())}")
        return {}, [], ""

    parse_fn, env_var, _ = LLM_PROVIDERS[provider]

    # Resolver API key
    if api_key is None:
        api_key = os.environ.get(env_var)
    if not api_key:
        if verbose:
            print(f"  [llm:{provider}] Sin API key "
                  f"(usa --api-key o exporta {env_var})")
        return {}, [], ""

    if verbose or debug:
        model_str = model or "(default)"
        print(f"  [llm:{provider}] modelo={model_str}")

    if debug:
        _llm_debug_block("SYSTEM PROMPT", LLM_SYSTEM_PROMPT)

    prompt = build_llm_prompt(instruction, current_state)
    return parse_fn(prompt, api_key, model, verbose, debug=debug)


def merge_params(dict_params: dict, llm_params: dict,
                 intensity: float = 1.0) -> dict:
    """
    Combina parámetros del diccionario y el LLM.
    El LLM enriquece/refina lo que el diccionario no captura.
    El diccionario tiene prioridad en campos que sí capturó.
    Aplica el modificador de intensidad.
    """
    merged = copy.deepcopy(dict_params)

    for k, v in llm_params.items():
        if k not in merged:
            merged[k] = v
        elif isinstance(v, (int, float)) and isinstance(merged.get(k), (int, float)):
            # Promedio ponderado: dict 60%, llm 40%
            merged[k] = merged[k] * 0.6 + v * 0.4

    # Aplicar intensidad a todos los deltas numéricos
    delta_keys = [
        "key_mode_delta", "tension_delta", "register_delta",
        "density_delta", "harmony_complexity_delta", "swing_delta",
        "velocity_mean_delta", "velocity_var_delta", "ornament_density_delta",
    ]
    for k in delta_keys:
        if k in merged and isinstance(merged[k], (int, float)):
            merged[k] = merged[k] * intensity

    # tempo_delta: centrado en 1.0
    if "tempo_delta" in merged:
        d = (merged["tempo_delta"] - 1.0) * intensity
        merged["tempo_delta"] = 1.0 + d

    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  RESOLUCIÓN DE PARÁMETROS
# ══════════════════════════════════════════════════════════════════════════════

def resolve_key(current_tonic: str, current_mode: str,
                mode_delta: float = 0.0,
                tonic_delta: int = 0) -> tuple[str, str]:
    """
    Navega el mapa de modos y el círculo de quintas para
    resolver la nueva tonalidad.
    """
    # ── Modo ──────────────────────────────────────────────────────────────────
    current_darkness = MODE_DARKNESS.get(current_mode, 0.5)
    new_darkness = np.clip(current_darkness + mode_delta, 0.0, 1.0)
    # Encontrar el modo más cercano al nuevo valor de oscuridad
    new_mode = min(MODE_DARKNESS_SORTED,
                   key=lambda x: abs(x[1] - new_darkness))[0]

    # ── Tónica ────────────────────────────────────────────────────────────────
    if tonic_delta != 0:
        try:
            idx = CIRCLE_OF_FIFTHS.index(current_tonic)
        except ValueError:
            # Manejar alteraciones enarmónicas
            enharmonic = {"C#": "Db", "Db": "C#", "F#": "Gb", "Gb": "F#",
                          "G#": "Ab", "Ab": "G#", "D#": "Eb", "Eb": "D#",
                          "A#": "Bb", "Bb": "A#"}
            enh = enharmonic.get(current_tonic, current_tonic)
            idx = CIRCLE_OF_FIFTHS.index(enh) if enh in CIRCLE_OF_FIFTHS else 0
        new_idx = (idx + tonic_delta) % len(CIRCLE_OF_FIFTHS)
        new_tonic = CIRCLE_OF_FIFTHS[new_idx]
    else:
        new_tonic = current_tonic

    return new_tonic, new_mode


def params_to_pipeline_args(params: dict, state: dict,
                              n_bars: int,
                              scope_start: int | None,
                              scope_end: int | None,
                              output_path: str,
                              verbose: bool = False) -> dict:
    """
    Convierte los parámetros semánticos a argumentos concretos
    para los scripts del pipeline.

    Retorna un dict con:
        "tool": nombre del script a usar
        "args": lista de argumentos CLI
        "summary": descripción legible
    """
    tool = "variation_engine"
    args = []
    summary_parts = []

    # ── Resolución de tonalidad ───────────────────────────────────────────────
    mode_delta    = params.get("key_mode_delta", 0.0)
    tonic_delta   = params.get("key_tonic_delta", 0)
    new_tonic, new_mode = resolve_key(
        state["key_tonic"], state["key_mode"],
        mode_delta, tonic_delta
    )
    key_changed = (new_tonic != state["key_tonic"] or new_mode != state["key_mode"])

    # ── Resolución de tempo ───────────────────────────────────────────────────
    tempo_factor  = params.get("tempo_delta", 1.0)
    new_tempo     = round(state["tempo_bpm"] * tempo_factor, 1)
    tempo_changed = abs(new_tempo - state["tempo_bpm"]) > 2.0

    # ── Construcción de curvas mt-* ───────────────────────────────────────────
    mt_curves = {}

    tension_delta = params.get("tension_delta", 0.0)
    tension_shape = params.get("tension_shape", None)

    if tension_delta != 0.0 or tension_shape:
        base_tension = state.get("tension_mean", 0.5)
        if tension_shape:
            t_arr = _build_tension_shape(tension_shape, n_bars)
            # Aplicar el delta encima del shape
            t_arr = np.clip(t_arr + tension_delta * 0.5, 0, 1)
        else:
            t_arr = np.ones(n_bars) * np.clip(base_tension + tension_delta, 0, 1)
        mt_curves["tension"] = t_arr

    reg_delta = params.get("register_delta", 0.0)
    if abs(reg_delta) > 0.05:
        reg_map = {"low": 0.2, "mid": 0.5, "high": 0.8}
        base_reg = reg_map.get(state.get("register", "mid"), 0.5)
        new_reg = np.clip(base_reg + reg_delta, 0, 1)
        mt_curves["register"] = np.ones(n_bars) * new_reg

    density_delta = params.get("density_delta", 0.0)
    if abs(density_delta) > 0.05:
        base_dens = state.get("density", 0.5)
        mt_curves["density"] = np.ones(n_bars) * np.clip(base_dens + density_delta, 0.05, 1)

    harmony_delta = params.get("harmony_complexity_delta", 0.0)
    if abs(harmony_delta) > 0.05:
        base_harm = state.get("harmony_complexity", 0.3)
        mt_curves["harmony"] = np.ones(n_bars) * np.clip(base_harm + harmony_delta, 0, 1)

    swing_delta = params.get("swing_delta", 0.0)
    if abs(swing_delta) > 0.05:
        base_swing = state.get("swing", 0.1)
        mt_curves["swing"] = np.ones(n_bars) * np.clip(base_swing + swing_delta, 0, 1)

    # Aplicar scope temporal a las curvas si existe
    if scope_start is not None and scope_end is not None:
        for k in mt_curves:
            full = np.ones(n_bars) * (mt_curves[k][0] if np.isscalar(mt_curves[k]) else
                                       mt_curves[k][0])
            # Fuera del scope: mantener valor base
            base_val_map = {
                "tension": state.get("tension_mean", 0.5),
                "register": {"low":0.2,"mid":0.5,"high":0.8}.get(state.get("register","mid"),0.5),
                "density": state.get("density", 0.5),
                "harmony": state.get("harmony_complexity", 0.3),
                "swing": state.get("swing", 0.1),
            }
            base_val = base_val_map.get(k, 0.5)
            arr = np.ones(n_bars) * base_val
            scope_vals = mt_curves[k]
            if hasattr(scope_vals, "__len__"):
                scope_len = len(scope_vals)
                target_len = scope_end - scope_start
                if scope_len != target_len:
                    scope_vals = np.interp(
                        np.linspace(0, 1, target_len),
                        np.linspace(0, 1, scope_len),
                        scope_vals
                    )
                arr[scope_start:scope_end] = scope_vals[:target_len]
            else:
                arr[scope_start:scope_end] = scope_vals
            mt_curves[k] = arr

    # ── Decidir herramienta ───────────────────────────────────────────────────
    variation_type = params.get("variation_type", None)

    # Si hay un tipo de variación específico, usamos variation_engine
    if variation_type:
        tool = "variation_engine"
        var_code = variation_type
        summary_parts.append(f"Variación {var_code}")
    else:
        # Si hay cambios de tonalidad o curvas, usamos midi_dna_unified via tension_designer
        tool = "midi_dna_via_tension"

    # ── Summary ───────────────────────────────────────────────────────────────
    if key_changed:
        summary_parts.append(f"Tonalidad: {state['key_tonic']} {state['key_mode']} → {new_tonic} {new_mode}")
    if tempo_changed:
        summary_parts.append(f"Tempo: {state['tempo_bpm']:.0f} → {new_tempo:.0f} BPM")
    if mt_curves.get("tension") is not None:
        t_mean = float(np.mean(mt_curves["tension"]))
        summary_parts.append(f"Tensión media: {state.get('tension_mean',0.5):.2f} → {t_mean:.2f}")
    if mt_curves.get("density") is not None:
        summary_parts.append(f"Densidad: {state.get('density',0.5):.2f} → {float(mt_curves['density'].mean()):.2f}")

    return {
        "tool": tool,
        "new_key": f"{new_tonic} {new_mode}",
        "new_tonic": new_tonic,
        "new_mode": new_mode,
        "new_tempo": new_tempo,
        "key_changed": key_changed,
        "tempo_changed": tempo_changed,
        "mt_curves": mt_curves,
        "variation_type": variation_type,
        "output_path": output_path,
        "summary": " | ".join(summary_parts) if summary_parts else "sin cambios significativos",
    }


def _build_tension_shape(shape: str, n_bars: int) -> np.ndarray:
    """Construye un array de tensión según un preset."""
    t = np.linspace(0, 1, n_bars)
    if shape == "arch":
        return 0.1 + 0.85 * np.sin(np.pi * t)
    elif shape == "crescendo":
        return 0.1 + 0.85 * t
    elif shape == "decrescendo":
        return 0.95 - 0.85 * t
    elif shape == "plateau":
        return np.where((t > 0.2) & (t < 0.8), 0.8, 0.2)
    elif shape == "late_climax":
        return np.clip(t ** 0.5 * 0.9 + 0.05, 0, 1)
    elif shape == "wave":
        return 0.5 + 0.4 * np.sin(2 * np.pi * t)
    else:  # neutral
        return np.ones(n_bars) * 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN: ejecutar la mutación
# ══════════════════════════════════════════════════════════════════════════════

def apply_mutation(resolved: dict, source_midi: str, state: dict,
                   verbose: bool = False) -> bool:
    """
    Aplica la mutación resuelta llamando al script correspondiente.
    Retorna True si el archivo de salida fue creado.
    """
    output = resolved["output_path"]
    tool   = resolved["tool"]

    # ── Directorio de salida ──────────────────────────────────────────────────
    out_dir = str(Path(output).parent)
    os.makedirs(out_dir, exist_ok=True)

    # ── Opción 1: Variation Engine ────────────────────────────────────────────
    if tool == "variation_engine" and resolved.get("variation_type"):
        return _run_variation_engine(resolved, source_midi, state, verbose)

    # ── Opción 2: midi_dna_unified via tension + key + tempo ──────────────────
    return _run_midi_dna(resolved, source_midi, state, verbose)


def _run_variation_engine(resolved: dict, source_midi: str,
                           state: dict, verbose: bool) -> bool:
    """Llama a variation_engine.py con la variación adecuada."""
    script = _find_script("variation_engine.py")
    if not script:
        print("  [ERROR] variation_engine.py no encontrado")
        return False

    vtype = resolved["variation_type"]
    output = resolved["output_path"]
    out_dir = str(Path(output).parent)

    cmd = [
        sys.executable, script, source_midi,
        "--variations", vtype,
        "--output-dir", out_dir,
    ]
    if state.get("n_bars"):
        cmd += ["--bars", str(state["n_bars"])]
    if verbose:
        cmd += ["--verbose"]

    if verbose:
        print(f"  [cmd] {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=120)
        # variation_engine guarda con nombre automático; buscar el generado
        var_name_map = {
            "V01": "V01_inversion", "V02": "V02_retrogrado",
            "V03": "V03_inv_retrogrado", "V04": "V04_aumentacion",
            "V05": "V05_diminucion", "V06": "V06_ornamentacion",
            "V07": "V07_acompanamiento", "V08": "V08_transportada",
            "V09": "V09_modal", "V10": "V10_ritmica",
            "V11": "V11_armonica", "V12": "V12_textural",
            "V13": "V13_emocional", "V14": "V14_estocastica",
            "V15": "V15_contrapunto",
        }
        stem = Path(source_midi).stem
        var_label = var_name_map.get(vtype, vtype)
        generated = Path(out_dir) / f"{stem}_{var_label}.mid"
        if generated.exists():
            shutil.copy(str(generated), output)
            return True
        # Buscar cualquier .mid recién generado
        candidates = sorted(Path(out_dir).glob("*.mid"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            shutil.copy(str(candidates[0]), output)
            return True
        return False
    except subprocess.TimeoutExpired:
        print("  [ERROR] variation_engine tardó demasiado")
        return False
    except Exception as e:
        print(f"  [ERROR] variation_engine: {e}")
        return False


def _run_midi_dna(resolved: dict, source_midi: str,
                   state: dict, verbose: bool) -> bool:
    """
    Aplica la mutación usando tension_designer + midi_dna_unified.
    Si no hay midi_dna_unified disponible, aplica transformaciones
    directas con mido como fallback.
    """
    script_td  = _find_script("tension_designer.py")
    script_dna = _find_script("midi_dna_unified.py")
    output     = resolved["output_path"]

    # Si tenemos midi_dna_unified, lo usamos correctamente
    if script_dna:
        return _run_via_midi_dna_direct(resolved, source_midi, state, verbose, script_dna)

    # Si tenemos tension_designer, lo usamos
    if script_td:
        return _run_via_tension_designer(resolved, source_midi, state, verbose, script_td)

    # Fallback: transformaciones directas con mido
    print("  [warn] midi_dna_unified.py y tension_designer.py no encontrados.")
    print("         Aplicando transformaciones básicas con mido.")
    return _apply_mido_transforms(resolved, source_midi, state, verbose)


def _run_via_midi_dna_direct(resolved: dict, source_midi: str,
                               state: dict, verbose: bool,
                               script_path: str) -> bool:
    """Llama directamente a midi_dna_unified con los parámetros resueltos."""
    output = resolved["output_path"]
    n_bars = state.get("n_bars", 16)
    mt_curves = resolved.get("mt_curves", {})

    cmd = [
        sys.executable, script_path,
        source_midi,
        "--output", output,
        "--bars", str(n_bars),
        "--mode", "auto",
    ]

    if resolved["key_changed"]:
        cmd += ["--key", resolved["new_tonic"]]

    if resolved["tempo_changed"]:
        cmd += ["--tempo", str(int(resolved["new_tempo"]))]

    # Serializar curvas como especificaciones mt-*
    curve_flag_map = {
        "tension": "--mt-tension",
        "density": "--mt-density",
        "register": "--mt-register",
        "harmony": "--mt-harmony-complexity",
        "swing": "--mt-swing",
    }
    for curve_name, flag in curve_flag_map.items():
        if curve_name in mt_curves:
            spec = _array_to_mt_spec(mt_curves[curve_name], n_bars)
            cmd += [flag, spec]

    if verbose:
        print(f"  [cmd] {' '.join(cmd[:8])} ...")

    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=180)
        return Path(output).exists()
    except Exception as e:
        print(f"  [ERROR] midi_dna_unified: {e}")
        return False


def _run_via_tension_designer(resolved: dict, source_midi: str,
                                state: dict, verbose: bool,
                                script_path: str) -> bool:
    """Usa tension_designer en modo headless."""
    output  = resolved["output_path"]
    n_bars  = state.get("n_bars", 16)
    mt_curves = resolved.get("mt_curves", {})

    # Escribir curvas a JSON temporal
    curves_dict = {}
    if "tension" in mt_curves:
        curves_dict["tension"]  = mt_curves["tension"].tolist()
    if "density" in mt_curves:
        curves_dict["activity"] = mt_curves["density"].tolist()
    if "register" in mt_curves:
        curves_dict["register"] = mt_curves["register"].tolist()
    if "harmony" in mt_curves:
        curves_dict["harmony"]  = mt_curves["harmony"].tolist()
    if "swing" in mt_curves:
        curves_dict["swing"]    = mt_curves["swing"].tolist()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(curves_dict, f)
        curves_file = f.name

    cmd = [
        sys.executable, script_path,
        source_midi,
        "--no-gui",
        "--load", curves_file,
        "--bars", str(n_bars),
        "--output", output,
    ]
    if verbose:
        cmd += ["--verbose"]

    if verbose:
        print(f"  [cmd] {' '.join(cmd[:6])} ...")

    try:
        result = subprocess.run(cmd, capture_output=not verbose, text=True, timeout=180)
        return Path(output).exists()
    finally:
        Path(curves_file).unlink(missing_ok=True)


def _apply_mido_transforms(resolved: dict, source_midi: str,
                            state: dict, verbose: bool) -> bool:
    """
    Transformaciones básicas directamente sobre el MIDI con mido.
    Fallback cuando no hay midi_dna_unified disponible.
    """
    try:
        import mido
        from mido import MidiFile, MidiTrack, Message, MetaMessage
    except ImportError:
        print("  [ERROR] mido no disponible")
        return False

    output = resolved["output_path"]

    try:
        mid = MidiFile(source_midi)
        new_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)

        # Calcular factor de transpopición si hay cambio de tonalidad
        # (simplificado: semitono según relación de tonalidades)
        semitone_shift = 0
        if resolved["key_changed"]:
            old_tonic = state["key_tonic"]
            new_tonic = resolved["new_tonic"]
            try:
                old_idx = TONICS.index(old_tonic) if old_tonic in TONICS else 0
                new_idx = TONICS.index(new_tonic) if new_tonic in TONICS else 0
                semitone_shift = new_idx - old_idx
                if semitone_shift > 6:
                    semitone_shift -= 12
                elif semitone_shift < -6:
                    semitone_shift += 12
            except Exception:
                semitone_shift = 0

        # Factor de tempo
        tempo_factor = resolved["new_tempo"] / max(state["tempo_bpm"], 1.0)

        # Cambio de velocidades (dinámica)
        vel_mean_delta = resolved.get("velocity_mean_delta", 0.0) if hasattr(resolved, "get") else 0.0
        # Buscamos en params originales — pasamos por resolved solo si lo llevamos
        vel_delta_mult = 1.0 + vel_mean_delta * 0.3

        for track in mid.tracks:
            new_track = MidiTrack()
            for msg in track:
                if msg.is_meta:
                    if msg.type == "set_tempo" and resolved["tempo_changed"]:
                        new_tempo = int(msg.tempo / tempo_factor)
                        new_track.append(MetaMessage("set_tempo", tempo=new_tempo, time=msg.time))
                    else:
                        new_track.append(msg.copy())
                elif msg.type in ("note_on", "note_off"):
                    new_note = msg.note + semitone_shift
                    new_note = max(0, min(127, new_note))
                    new_vel  = int(msg.velocity * vel_delta_mult)
                    new_vel  = max(1, min(127, new_vel))
                    new_track.append(msg.copy(note=new_note, velocity=new_vel))
                else:
                    new_track.append(msg.copy())
            new_mid.tracks.append(new_track)

        new_mid.save(output)
        return True

    except Exception as e:
        print(f"  [ERROR] Transformación mido: {e}")
        return False


def _array_to_mt_spec(arr: np.ndarray, n_bars: int) -> str:
    """
    Convierte un array de valores (0-1, longitud n_bars) al formato
    de especificación mt-* de midi_dna_unified:
    "0:0.3, 4:0.6, 8:0.8, 12:0.9" (solo puntos de cambio significativo).
    """
    if len(arr) == 0:
        return "0:0.5"
    # Simplificar: tomar ~8 puntos representativos
    n_points = min(8, len(arr))
    indices = np.round(np.linspace(0, len(arr)-1, n_points)).astype(int)
    points = [(int(idx), float(np.clip(arr[idx], 0, 1))) for idx in indices]
    return ", ".join(f"{bar}:{val:.2f}" for bar, val in points)


def _find_script(name: str) -> str | None:
    """Busca un script del pipeline en el mismo directorio o en PATH."""
    # Mismo directorio que mutator.py
    here = Path(__file__).parent / name
    if here.exists():
        return str(here)
    # CWD
    cwd = Path.cwd() / name
    if cwd.exists():
        return str(cwd)
    # PATH
    found = shutil.which(name)
    if found:
        return found
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICACIÓN CON MSCZ2VEC
# ══════════════════════════════════════════════════════════════════════════════

def verify_mutation(original_midi: str, result_midi: str,
                    expected_params: dict, verbose: bool = False) -> dict:
    """
    Compara el estado original con el resultado usando análisis básico.
    Retorna un dict con métricas de verificación.
    """
    before = read_midi_state(original_midi, verbose=False)
    after  = read_midi_state(result_midi, verbose=False)

    report = {"verified": True, "checks": []}

    tension_exp = expected_params.get("tension_delta", 0.0)
    tension_actual = after.get("tension_mean", 0.5) - before.get("tension_mean", 0.5)
    if abs(tension_exp) > 0.1:
        correct = (tension_exp * tension_actual) > 0
        report["checks"].append({
            "dimension": "tensión",
            "expected_direction": "+" if tension_exp > 0 else "-",
            "actual_change": round(tension_actual, 3),
            "ok": correct,
        })
        if not correct:
            report["verified"] = False

    if verbose:
        print(f"\n  [verificación]")
        for c in report["checks"]:
            status = "✓" if c["ok"] else "✗"
            print(f"    {status} {c['dimension']}: dir={c['expected_direction']} "
                  f"cambio={c['actual_change']:+.3f}")

    return report


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT NOMBRE AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════

def auto_output_name(source_midi: str, instruction: str,
                     mutation_id: int, output_dir: str | None = None) -> str:
    """Genera un nombre de archivo de salida descriptivo."""
    stem = Path(source_midi).stem
    # Extraer palabras clave de la instrucción
    words = re.findall(r'\b[a-záéíóúñA-ZÁÉÍÓÚÑ]{4,}\b', instruction)
    keyword = normalize_text("_".join(words[:2])) if words else "mut"
    keyword = re.sub(r'[^a-z0-9_]', '', keyword)[:20]
    filename = f"{stem}_mut{mutation_id:03d}_{keyword}.mid"
    if output_dir:
        return str(Path(output_dir) / filename)
    return str(Path(source_midi).parent / filename)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════

def interactive_loop(source_midi: str, args, mutation_state: MutationState):
    """Bucle de mutación interactivo."""
    current_midi = source_midi
    print(f"\n{'═'*60}")
    print(f"  MUTATOR  v{VERSION}  —  Modo interactivo")
    print(f"  Obra actual: {Path(current_midi).name}")
    print(f"{'═'*60}")
    print("  Comandos especiales:")
    print("    'historia'  — ver historial de mutaciones")
    print("    'arbol'     — ver árbol de mutaciones")
    print("    'revertir N'— volver a la mutación N")
    print("    'ramificar MIDI' — partir desde otro MIDI del historial")
    print("    'salir'     — terminar")
    print()

    mutation_id = len(mutation_state.history["mutations"]) + 1

    while True:
        try:
            instruction = input(f"  [mut{mutation_id:03d}] Instrucción > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Terminando.")
            break

        if not instruction:
            continue
        if instruction.lower() in ("salir", "exit", "quit", "q"):
            break
        if instruction.lower() in ("historia", "historial", "history"):
            mutation_state.print_history()
            continue
        if instruction.lower() in ("arbol", "árbol", "tree"):
            mutation_state.print_tree()
            continue

        m = re.match(r"revertir\s+(\d+)", instruction.lower())
        if m:
            n = int(m.group(1))
            entry = mutation_state.get_mutation(n)
            if entry:
                current_midi = entry["result"]
                print(f"  → Revertido a: {Path(current_midi).name}")
            else:
                print(f"  → Mutación {n} no encontrada")
            continue

        m = re.match(r"ramificar\s+(.+\.mid)", instruction, re.IGNORECASE)
        if m:
            branch_midi = m.group(1).strip()
            if Path(branch_midi).exists():
                current_midi = branch_midi
                print(f"  → Rama desde: {Path(current_midi).name}")
            else:
                print(f"  → Archivo no encontrado: {branch_midi}")
            continue

        # Ejecutar mutación
        success, output_path = run_single_mutation(
            current_midi, instruction,
            args, mutation_state, mutation_id, verbose=args.verbose
        )

        if success:
            current_midi = output_path
            mutation_id += 1
            print(f"  ✓ Generado: {Path(output_path).name}")
            if args.listen:
                _play_midi(output_path, seconds=args.play_seconds)
        else:
            print("  ✗ La mutación falló. Prueba con otra instrucción.")


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL DE MUTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def run_single_mutation(source_midi: str, instruction: str,
                         args, mutation_state: MutationState,
                         mutation_id: int | None = None,
                         verbose: bool = False) -> tuple[bool, str]:
    """Ejecuta una mutación completa y retorna (success, output_path)."""

    if mutation_id is None:
        mutation_id = len(mutation_state.history["mutations"]) + 1

    print(f"\n  Instrucción: \"{instruction}\"")
    print(f"  Fuente:      {Path(source_midi).name}")

    # ── 1. Leer estado ────────────────────────────────────────────────────────
    state = read_midi_state(source_midi, verbose=verbose)
    n_bars = args.bars if args.bars else state.get("n_bars", 16)
    state["n_bars"] = n_bars

    if verbose:
        print(f"  [estado] {state['key_tonic']} {state['key_mode']} | "
              f"{state['tempo_bpm']:.0f} BPM | {n_bars} bars | "
              f"tensión: {state.get('tension_mean',0.5):.2f}")

    # ── 2. Extraer scope ──────────────────────────────────────────────────────
    scope_start_bar = None
    scope_end_bar   = None

    if args.scope:
        try:
            parts = args.scope.split(":")
            scope_start_bar = int(parts[0])
            scope_end_bar   = int(parts[1])
        except Exception:
            print(f"  [warn] --scope mal formado: {args.scope} (esperado S:E)")
    else:
        scope_start_bar, scope_end_bar = parse_scope(instruction, n_bars)
        if scope_start_bar is not None and verbose:
            print(f"  [scope] Compases {scope_start_bar}–{scope_end_bar}")

    # ── 3. Intensidad ─────────────────────────────────────────────────────────
    intensity_mod = parse_intensity_modifier(instruction)
    intensity = np.clip(args.intensity * intensity_mod, 0.1, 1.5)
    if verbose and abs(intensity_mod - 1.0) > 0.05:
        print(f"  [intensidad] {args.intensity:.1f} × {intensity_mod:.1f} = {intensity:.2f}")

    # ── 4. Parsing ────────────────────────────────────────────────────────────
    print("  Parseando instrucción...")
    dict_params, dict_matched = dictionary_parse(instruction, verbose=verbose)

    llm_params, llm_matched, reasoning = {}, [], ""
    if not args.no_llm:
        llm_params, llm_matched, reasoning = llm_parse(
            instruction, state,
            api_key=args.api_key,
            provider=getattr(args, "llm_provider", "anthropic"),
            model=getattr(args, "llm_model", None),
            verbose=verbose,
            debug=getattr(args, "llm_debug", False),
        )

    all_params = merge_params(dict_params, llm_params, intensity)

    if not all_params:
        print("  ✗ No se reconoció ningún concepto en la instrucción.")
        print("    Prueba con términos como: oscuro, tranquilo, urgente,")
        print("    más lento, luminoso, misterioso, lluvia, vals...")
        return False, ""

    all_matched = list(set(dict_matched + llm_matched))
    if verbose:
        print(f"  [params] {json.dumps({k:round(v,3) if isinstance(v,float) else v for k,v in all_params.items()}, ensure_ascii=False)}")

    # ── 5. Output path ────────────────────────────────────────────────────────
    if args.output:
        output_path = args.output
    else:
        out_dir = args.output_dir or str(Path(source_midi).parent)
        output_path = auto_output_name(source_midi, instruction, mutation_id, out_dir)

    # ── 6. Dry run ────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n  [dry-run] Parámetros resueltos:")
        for k, v in all_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:+.3f}")
            else:
                print(f"    {k}: {v}")
        print(f"  [dry-run] Output: {output_path}")
        return True, output_path

    # ── 7. Resolver parámetros ────────────────────────────────────────────────
    resolved = params_to_pipeline_args(
        all_params, state, n_bars,
        scope_start_bar, scope_end_bar,
        output_path, verbose=verbose
    )
    print(f"  Cambios: {resolved['summary']}")

    # ── 8. Aplicar ────────────────────────────────────────────────────────────
    print(f"  Generando {Path(output_path).name}...")
    success = apply_mutation(resolved, source_midi, state, verbose=verbose)

    if not success:
        # Si el script no generó el archivo pero hay transformaciones básicas,
        # intentar fallback directo con mido
        if not Path(output_path).exists():
            print("  [fallback] Intentando transformación básica con mido...")
            resolved["velocity_mean_delta"] = all_params.get("velocity_mean_delta", 0.0)
            success = _apply_mido_transforms(resolved, source_midi, state, verbose)

    if not success:
        print(f"  ✗ No se generó el archivo de salida.")
        return False, ""

    # ── 9. Verificación ───────────────────────────────────────────────────────
    if args.verify:
        verify_mutation(source_midi, output_path, all_params, verbose=True)

    # ── 10. Guardar historial ─────────────────────────────────────────────────
    mid = mutation_state.add_mutation(
        instruction=instruction,
        params={k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in all_params.items()},
        result_path=output_path,
        parent_path=source_midi,
        reasoning=reasoning or resolved["summary"],
    )

    print(f"  ✓ {Path(output_path).name}  [mut{mid:03d}]")
    if reasoning and verbose:
        print(f"  Razonamiento: {reasoning}")

    return True, output_path


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _play_midi(file_path: str, seconds: int = 10):
    """Reproduce un MIDI usando pygame."""
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        start = time.time()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            if time.time() - start >= seconds:
                pygame.mixer.music.stop()
                break
    except Exception as e:
        print(f"  [warn] Reproducción: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  LISTADO DE CONCEPTOS
# ══════════════════════════════════════════════════════════════════════════════

def print_concepts():
    """Lista todos los conceptos semánticos disponibles."""
    print(f"\n{'═'*60}")
    print(f"  Conceptos semánticos disponibles en MUTATOR v{VERSION}")
    print(f"{'═'*60}")

    categories = {
        "Carácter tonal": ["oscuro", "muy oscuro", "luminoso", "melancólico",
                            "épico", "misterioso", "romántico"],
        "Energía / Tempo": ["urgente", "tranquilo", "más rápido", "más lento", "enérgico"],
        "Tensión / Arco":  ["tenso", "relajado", "climax", "arco"],
        "Densidad / Textura": ["denso", "sparse", "más voces"],
        "Registro":        ["más grave", "más agudo"],
        "Groove / Ritmo":  ["swing", "recto", "rítmico"],
        "Ornamentación":   ["ornamentado", "simple", "expresivo"],
        "Imágenes":        ["lluvia", "tormenta", "amanecer", "noche", "mar", "vacío", "vals"],
        "Variaciones clásicas": ["al revés", "invertido", "aumentado", "disminuido"],
    }

    for cat, concepts in categories.items():
        print(f"\n  {cat}:")
        for c in concepts:
            data = SEMANTIC_MAP.get(c, {})
            aliases = data.get("aliases", [])
            alias_str = f"  ({', '.join(aliases[:3])})" if aliases else ""
            print(f"    • {c}{alias_str}")

    print(f"\n  También puedes combinar conceptos:")
    print(f"    'más oscuro y urgente'")
    print(f"    'tranquilo y melancólico en los últimos 8 compases'")
    print(f"    'lluvia con tensión creciente'")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="mutator.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            MUTATOR v1.0 — Mutación semántica de MIDIs en lenguaje natural
            ─────────────────────────────────────────────────────────────
            Transforma instrucciones en lenguaje natural en mutaciones
            musicales concretas aplicadas sobre un MIDI existente.

            Ejemplos:
              python mutator.py obra.mid "más oscuro"
              python mutator.py obra.mid "urgente en la segunda mitad"
              python mutator.py obra.mid "que suene como lluvia"
              python mutator.py obra.mid --interactive
              python mutator.py obra.mid --list-concepts
        """),
    )

    parser.add_argument("midi", nargs="?", help="MIDI fuente")
    parser.add_argument("instruction", nargs="?", default=None,
                        help="Instrucción en lenguaje natural")

    # Opciones de operación
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Modo conversacional iterativo")
    parser.add_argument("--history",     action="store_true",
                        help="Mostrar historial de mutaciones")
    parser.add_argument("--tree",        action="store_true",
                        help="Visualizar árbol de mutaciones")
    parser.add_argument("--revert",      type=int, default=None, metavar="N",
                        help="Revertir a la mutación N")
    parser.add_argument("--branch",      default=None, metavar="MIDI",
                        help="Partir desde un MIDI del historial")
    parser.add_argument("--list-concepts", action="store_true",
                        help="Listar conceptos semánticos disponibles")

    # Parámetros de mutación
    parser.add_argument("--intensity", type=float, default=0.7,
                        help="Intensidad global 0-1 (default: 0.7)")
    parser.add_argument("--bars",      type=int, default=None,
                        help="Compases de salida (default: auto)")
    parser.add_argument("--scope",     default=None, metavar="S:E",
                        help="Aplicar solo en compases S a E (ej: 8:16)")

    # Salida
    parser.add_argument("--output",     default=None,
                        help="Archivo de salida (default: auto)")
    parser.add_argument("--output-dir", default=None,
                        help="Directorio para archivos de salida")

    # LLM
    parser.add_argument("--no-llm",       action="store_true",
                        help="Usar solo diccionario semántico, sin LLM")
    parser.add_argument("--llm-provider", default="anthropic",
                        choices=list(LLM_PROVIDERS.keys()),
                        help="Proveedor LLM: anthropic (default) | openai")
    parser.add_argument("--llm-model",    default=None,
                        help="Modelo específico (default: auto según proveedor). "
                             "Anthropic: claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5-20251001 | "
                             "OpenAI: gpt-4o, gpt-4o-mini, gpt-4-turbo")
    parser.add_argument("--api-key",      default=None,
                        help="API key del proveedor elegido "
                             "(o vars de entorno: ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--llm-debug",    action="store_true",
                        help="Mostrar prompt enviado y respuesta raw del LLM")

    # Utilidades
    parser.add_argument("--dry-run",  action="store_true",
                        help="Mostrar parámetros sin generar MIDI")
    parser.add_argument("--verify",   action="store_true",
                        help="Verificar que la mutación fue en la dirección correcta")
    parser.add_argument("--listen",   action="store_true",
                        help="Reproducir el resultado (requiere pygame)")
    parser.add_argument("--play-seconds", type=int, default=15,
                        help="Segundos de reproducción (default: 15)")
    parser.add_argument("--verbose",  "-v", action="store_true",
                        help="Informe detallado de decisiones")

    args = parser.parse_args()

    # ── Listar conceptos ──────────────────────────────────────────────────────
    if args.list_concepts:
        print_concepts()
        return

    # ── Verificar MIDI ────────────────────────────────────────────────────────
    if not args.midi:
        parser.print_help()
        return

    if not Path(args.midi).exists():
        print(f"ERROR: no existe el archivo: {args.midi}")
        sys.exit(1)

    mutation_state = MutationState(args.midi)

    # ── Acciones de historial ─────────────────────────────────────────────────
    if args.history:
        mutation_state.print_history()
        return

    if args.tree:
        mutation_state.print_tree()
        return

    if args.revert is not None:
        entry = mutation_state.get_mutation(args.revert)
        if entry:
            print(f"\n  Mutación {args.revert}: \"{entry['instruction']}\"")
            print(f"  Archivo: {entry['result']}")
            print(f"  Usa este archivo como fuente para continuar:")
            print(f"    python mutator.py {entry['result']} \"tu instrucción\"")
        else:
            print(f"  Mutación {args.revert} no encontrada.")
        return

    # ── Modo interactivo ──────────────────────────────────────────────────────
    source = args.branch if args.branch else args.midi
    if args.branch and not Path(args.branch).exists():
        print(f"ERROR: --branch archivo no existe: {args.branch}")
        sys.exit(1)

    if args.interactive:
        interactive_loop(source, args, mutation_state)
        return

    # ── Modo de instrucción única ─────────────────────────────────────────────
    if not args.instruction:
        print("ERROR: proporciona una instrucción o usa --interactive")
        print()
        print("  Ejemplos:")
        print("    python mutator.py obra.mid 'más oscuro'")
        print("    python mutator.py obra.mid 'urgente en la segunda mitad'")
        print("    python mutator.py obra.mid --interactive")
        print("    python mutator.py --list-concepts")
        sys.exit(1)

    mutation_id = len(mutation_state.history["mutations"]) + 1
    success, output_path = run_single_mutation(
        source, args.instruction,
        args, mutation_state, mutation_id,
        verbose=args.verbose
    )

    if success and args.listen and not args.dry_run:
        print(f"\n  ♪ Reproduciendo {Path(output_path).name}...")
        _play_midi(output_path, args.play_seconds)

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
