#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         THEORIST  v1.0                                       ║
║         Intención musical → teoría fundamentada → parámetros de pipeline    ║
║                                                                              ║
║  Traduce descripciones vagas de intención musical en planes de composición  ║
║  completos, justificados paso a paso en lenguaje teórico musical.           ║
║  Actúa como teórico/compositor que razona en voz alta antes de producir     ║
║  ningún número: primero semántica, luego teoría, luego parámetros.          ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] SEMÁNTICA    — interpreta la descripción libre, detecta contradicciones ║
║  [2] MARCO TEÓRICO— elige escuela compositiva, referencias, reglas          ║
║  [3] DECISIONES   — cada parámetro con su justificación teórica explícita   ║
║  [4] CURVAS       — genera curvas emocionales para tension_designer          ║
║  [5] PIPELINE     — traduce a argumentos CLI exactos para cada script        ║
║  [6] EXPORTACIÓN  — .theorist.json, .curves.json, plan.yaml, plan.json      ║
║                                                                              ║
║  MODOS DE ENTRADA:                                                           ║
║    Descripción libre:                                                        ║
║      python theorist.py "una pieza que suene a esperar algo que nunca llega"║
║    Con restricciones duras:                                                  ║
║      python theorist.py "melancolía urbana" --key Dm --bars 32 --tempo 72   ║
║    Anclado a un MIDI de referencia:                                          ║
║      python theorist.py "hazlo más oscuro" --from obra.mid                  ║
║    Modo conversacional iterativo:                                            ║
║      python theorist.py --interactive                                        ║
║    Refinar un plan existente:                                                ║
║      python theorist.py --load plan.theorist.json --refine "clímax más ambiguo"║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    description     Descripción libre de la intención (entre comillas)       ║
║    --from MIDI     Anclar interpretación a un MIDI de referencia             ║
║    --key KEY       Tonalidad forzada, ej. Dm, G, Bb (default: auto)         ║
║    --bars N        Compases totales (default: 32)                            ║
║    --tempo N       Tempo base BPM (default: auto)                            ║
║    --arc NAME      Arco narrativo base: hero|tragedy|romance|mystery|        ║
║                    meditation|rondo|sonata|custom (default: auto)            ║
║    --persona NAME  Persona teórica: schenker|fux|ravel|jazz|spectral|       ║
║                    baroque|romantic|modal (default: auto)                    ║
║    --dialectical   Generar 3 lecturas alternativas de la misma intención    ║
║    --interactive   Modo conversacional con compromisos teóricos por turno   ║
║    --load FILE     Cargar y refinar un plan .theorist.json existente         ║
║    --refine TEXT   Refinar el plan cargado con una nueva instrucción         ║
║    --dry-run       Solo mostrar el razonamiento, no generar archivos         ║
║    --execute       Ejecutar el pipeline completo tras generar el plan        ║
║    --until SCRIPT  Ejecutar hasta un script concreto (ej: midi_dna_unified) ║
║    --output BASE   Nombre base para archivos de salida (default: obra)      ║
║    --no-llm        Usar solo el motor local (sin API)                        ║
║    --llm-provider  anthropic (default) | openai                              ║
║    --llm-model     Modelo específico (default: auto)                         ║
║    --api-key KEY   API key (o ANTHROPIC_API_KEY / OPENAI_API_KEY)           ║
║    --verbose       Mostrar razonamiento completo durante el proceso          ║
║    --list-personas Listar personas teóricas disponibles                      ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    <base>.theorist.json  — Plan narrativo completo con justificaciones       ║
║    <base>.curves.json    — Curvas para tension_designer                      ║
║    <base>_plan.json      — Plan de narrator (editable en su GUI)             ║
║    <base>_plan.yaml      — Pipeline para runner.py                           ║
║                                                                              ║
║  DEPENDENCIAS: numpy                                                         ║
║  LLM (opcional): anthropic  →  pip install anthropic                        ║
║                  openai     →  pip install openai                            ║
║  SCRIPTS DEL PIPELINE: narrator.py, tension_designer.py,                    ║
║                         midi_dna_unified.py, reharmonizer.py, orchestrator  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import re
import textwrap
import copy
from pathlib import Path
from datetime import datetime

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0"

VALID_KEYS = [
    "C", "C#", "Db", "D", "D#", "Eb", "E", "F",
    "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B",
    "Cm", "C#m", "Dbm", "Dm", "D#m", "Ebm", "Em", "Fm",
    "F#m", "Gbm", "Gm", "G#m", "Abm", "Am", "A#m", "Bbm", "Bm",
]

VALID_MODES = [
    "major", "minor", "dorian", "phrygian", "lydian",
    "mixolydian", "locrian", "harmonic_minor", "melodic_minor",
]

VALID_ARCS = [
    "hero", "tragedy", "romance", "mystery",
    "meditation", "rondo", "sonata", "custom", "auto",
]

PIPELINE_SCRIPTS = [
    "narrator.py", "tension_designer.py", "midi_dna_unified.py",
    "reharmonizer.py", "orchestrator.py",
]

# ══════════════════════════════════════════════════════════════════════════════
#  CONTRATOS DE SCRIPTS DEL PIPELINE
#  Valores válidos según las interfaces reales (inspeccionadas del código).
# ══════════════════════════════════════════════════════════════════════════════

# Arcos válidos de narrator.py
NARRATOR_VALID_ARCS = [
    "hero", "tragedy", "romance", "mystery", "meditation", "rondo", "sonata",
]

# Mapeo de texto libre → arco válido
ARC_SEMANTIC_MAP = {
    # hero
    "hero": "hero", "heroico": "hero", "épico": "hero", "epico": "hero",
    "triunfo": "hero", "triumph": "hero", "victoria": "hero", "journey": "hero",
    "viaje": "hero", "aventura": "hero", "esperanza": "hero", "hope": "hero",
    "insatisfecha": "hero", "expectativa": "hero", "growth": "hero",
    # tragedy
    "tragedy": "tragedy", "tragedia": "tragedy", "pérdida": "tragedy",
    "perdida": "tragedy", "dolor": "tragedy", "duelo": "tragedy", "luto": "tragedy",
    "muerte": "tragedy", "fall": "tragedy", "caída": "tragedy", "caida": "tragedy",
    # romance
    "romance": "romance", "amor": "romance", "love": "romance",
    "ternura": "romance", "dulzura": "romance", "intimidad": "romance",
    # mystery
    "mystery": "mystery", "misterio": "mystery", "enigma": "mystery",
    "suspenso": "mystery", "suspense": "mystery", "ambiguo": "mystery",
    "inquietante": "mystery", "intriga": "mystery",
    # meditation
    "meditation": "meditation", "meditación": "meditation", "meditacion": "meditation",
    "calma": "meditation", "paz": "meditation", "contemplación": "meditation",
    "contemplacion": "meditation", "nocturno": "meditation", "noche": "meditation",
    "introspección": "meditation", "introspection": "meditation",
    "espera": "meditation", "waiting": "meditation", "melancolia": "meditation",
    "melancolía": "meditation", "soledad": "meditation",
    # rondo
    "rondo": "rondo", "rondó": "rondo", "dance": "rondo", "danza": "rondo",
    "playful": "rondo", "juguetón": "rondo", "jugeton": "rondo",
    # sonata
    "sonata": "sonata", "contraste": "sonata", "desarrollo": "sonata",
    "development": "sonata", "dialéctica": "sonata", "dialectica": "sonata",
}

# Presets válidos de tension_designer.py
TENSION_DESIGNER_PRESETS = [
    "arch", "crescendo", "decrescendo", "plateau", "late_climax",
    "wave", "neutral", "dramatic",
]

# Mapeo de nombre de arco → preset de tensión más apropiado
ARC_TO_TENSION_PRESET = {
    "hero":       "late_climax",
    "tragedy":    "arch",
    "romance":    "arch",
    "mystery":    "plateau",
    "meditation": "neutral",
    "rondo":      "wave",
    "sonata":     "arch",
}

# Templates válidos de orchestrator.py
ORCHESTRATOR_VALID_TEMPLATES = ["chamber", "full", "strings_only"]

# Estrategias válidas de reharmonizer.py
REHARMONIZER_VALID_STRATEGIES = [
    "diatonic", "tritone", "secondary", "modal_interchange",
    "chromatic_med", "coltrane", "neapolitan", "pedal",
    "minor_modal", "baroque", "impressionist",
]


def sanitize_arc(arc_raw: str) -> str:
    """
    Convierte texto libre del LLM en un arco válido de narrator.py.
    Busca por palabras clave en la descripción.
    """
    if not arc_raw:
        return "hero"
    arc_raw = arc_raw.strip().lower()

    # Coincidencia exacta primero
    if arc_raw in NARRATOR_VALID_ARCS:
        return arc_raw

    # Búsqueda por palabras clave
    for word, arc in ARC_SEMANTIC_MAP.items():
        if word in arc_raw:
            return arc

    # Fallback: intentar las palabras individuales del arc_raw
    for token in arc_raw.replace("_", " ").split():
        if token in NARRATOR_VALID_ARCS:
            return token
        if token in ARC_SEMANTIC_MAP:
            return ARC_SEMANTIC_MAP[token]

    return "hero"  # fallback seguro


def sanitize_orchestrator_template(template_raw: str) -> str:
    """Convierte texto libre en template válido de orchestrator.py."""
    if not template_raw:
        return "chamber"
    t = template_raw.strip().lower()
    if t in ORCHESTRATOR_VALID_TEMPLATES:
        return t
    # Mapeos comunes del LLM
    mapping = {
        "orchestra":    "full",
        "orquesta":     "full",
        "full_orchestra": "full",
        "strings":      "strings_only",
        "cuerdas":      "strings_only",
        "solo":         "chamber",
        "piano":        "chamber",
        "quartet":      "chamber",
        "quinteto":     "chamber",
    }
    for k, v in mapping.items():
        if k in t:
            return v
    return "chamber"


def sanitize_reharmonizer_strategies(strategy_raw) -> list[str]:
    """
    Convierte una estrategia del LLM (str o None) en lista de estrategias válidas.
    reharmonizer acepta múltiples estrategias como argumentos posicionales.
    """
    if not strategy_raw:
        return ["diatonic"]
    if isinstance(strategy_raw, list):
        strats = strategy_raw
    else:
        strats = [s.strip() for s in str(strategy_raw).replace(",", " ").split()]
    valid = [s for s in strats if s in REHARMONIZER_VALID_STRATEGIES]
    return valid if valid else ["diatonic"]


# ══════════════════════════════════════════════════════════════════════════════
#  PERSONAS TEÓRICAS
#  Cada persona activa un vocabulario y un conjunto de reglas distintos.
#  El LLM usa la persona como system prompt adicional para anclar el
#  razonamiento en una tradición teórica concreta.
# ══════════════════════════════════════════════════════════════════════════════

PERSONAS = {
    "schenker": {
        "label": "Analista schenkeriano",
        "description": "Identifica la estructura profunda (Ursatz) y trabaja desde ella hacia la superficie.",
        "preferred_modes": ["major", "minor"],
        "preferred_arcs": ["sonata", "hero"],
        "style_note": "Prioriza la línea fundamental, el bajo alberti y las dominantes estructurales.",
        "system_addendum": (
            "Razona como Heinrich Schenker. Identifica el Ursatz (línea fundamental + bajo "
            "fundamental), las dominantes estructurales y el prolongamiento. Cada decisión "
            "debe poder justificarse como prolongación de la estructura profunda. "
            "Usa terminología: Ursatz, Urlinie, Bassbrechung, Zug, Nebennoten, Aufstieg."
        ),
    },
    "fux": {
        "label": "Contrapuntista (Fux)",
        "description": "Aplica las reglas de contrapunto por especies de Gradus ad Parnassum.",
        "preferred_modes": ["dorian", "phrygian", "mixolydian", "major"],
        "preferred_arcs": ["meditation", "rondo"],
        "style_note": "Prioriza el movimiento contrario, evita paralelas de 5ª/8ª, usa cadencias clausulares.",
        "system_addendum": (
            "Razona como Johann Joseph Fux. Aplica las reglas de contrapunto estricto: "
            "movimiento contrario preferido, prohibición de paralelas de 5ª y 8ª perfectas, "
            "uso de disonancias solo como notas de paso en tiempos débiles (2ª especie) o "
            "como retardos resueltos por grado descendente (4ª especie). "
            "Justifica cada voz según su función contrapuntística."
        ),
    },
    "ravel": {
        "label": "Impresionista (Ravel/Debussy)",
        "description": "Trabaja con modos, quintas paralelas, escalas de tonos enteros y armonía por color.",
        "preferred_modes": ["lydian", "dorian", "major"],
        "preferred_arcs": ["meditation", "mystery"],
        "style_note": "Prioriza el color armónico sobre la función. Evita la tónica como destino obvio.",
        "system_addendum": (
            "Razona como Maurice Ravel o Claude Debussy. La armonía es color, no función. "
            "Usa escalas de tonos enteros, modos eclesiásticos, quintas y cuartas paralelas, "
            "acordes de 9ª sin resolución obligatoria, pedales y ostinati como textura. "
            "Evita la cadencia auténtica V7→I como cierre definitivo. "
            "Justifica decisiones en términos de timbre armónico y atmósfera."
        ),
    },
    "jazz": {
        "label": "Teórico de jazz",
        "description": "Aplica sustituciones de tritono, ii-V-I, tensiones extendidas y reharmonización cromática.",
        "preferred_modes": ["dorian", "mixolydian", "lydian"],
        "preferred_arcs": ["rondo", "hero"],
        "style_note": "Prioriza la conducción de voces, las tensiones 9ª/11ª/13ª y los ii-V-I.",
        "system_addendum": (
            "Razona como Mark Levine (The Jazz Theory Book). "
            "Usa progresiones ii-V-I, sustituciones de tritono (bII7 en lugar de V7), "
            "dominantes secundarios, préstamos modales, reharmonización cromática. "
            "Las tensiones 9ª, 11ª aumentada y 13ª son parte del vocabulario estándar. "
            "Justifica cada acorde en términos de conducción de voces y tensión-resolución."
        ),
    },
    "spectral": {
        "label": "Compositor espectral",
        "description": "Basa las decisiones en la física del sonido: series de armónicos, microtonalidad, timbre.",
        "preferred_modes": ["major", "lydian"],
        "preferred_arcs": ["meditation", "mystery"],
        "style_note": "Las alturas emergen de la serie armónica. El timbre es parámetro compositivo primario.",
        "system_addendum": (
            "Razona como Gérard Grisey o Tristan Murail. "
            "Las alturas se derivan de la serie de armónicos de un fundamental elegido. "
            "El timbre —no la melodía— es el parámetro compositivo central. "
            "Las transiciones son morphings espectrales graduales. "
            "Justifica tonos y densidades en términos de parciales del espectro."
        ),
    },
    "baroque": {
        "label": "Contrapuntista barroco (Bach)",
        "description": "Fuga, coral, passacaglia: reglas estrictas de conducción de voces y forma.",
        "preferred_modes": ["minor", "major", "dorian"],
        "preferred_arcs": ["sonata", "rondo"],
        "style_note": "Prioriza la coherencia motívica, las secuencias y el contrapunto imitativo.",
        "system_addendum": (
            "Razona como J.S. Bach. Cada voz tiene independencia melódica. "
            "Usa secuencias (ciclos de quintas, Rosalia), inversión y aumentación del motivo, "
            "stretto en la fuga, coral a 4 voces con conducción estricta. "
            "Las disonancias preparan y resuelven. "
            "Justifica decisiones en términos de contrapunto y forma fugada."
        ),
    },
    "romantic": {
        "label": "Compositor romántico",
        "description": "Arco emocional largo, cromatismo, modulaciones a terceras, rubato y expresividad.",
        "preferred_modes": ["minor", "major", "harmonic_minor"],
        "preferred_arcs": ["hero", "romance", "tragedy"],
        "style_note": "Prioriza el arco emocional sobre la corrección contrapuntística.",
        "system_addendum": (
            "Razona como Schubert, Brahms o Chopin. "
            "El arco emocional es el argumento central. "
            "Usa modulaciones a terceras (mediante cromática), "
            "progresiones de acordes disminuidos y aumentados, "
            "armonía cromática con cromatismo ascendente y descendente. "
            "El rubato y las dinámicas son parte del vocabulario compositivo. "
            "Justifica en términos de tensión emocional y narrativa."
        ),
    },
    "modal": {
        "label": "Compositor modal",
        "description": "Trabaja con los modos griegos, música antigua y exploración modal postmoderna.",
        "preferred_modes": ["dorian", "phrygian", "lydian", "mixolydian"],
        "preferred_arcs": ["meditation", "mystery", "rondo"],
        "style_note": "Evita la tonalidad funcional. El modo define el color, no la dominante.",
        "system_addendum": (
            "Razona como Olivier Messiaen o Béla Bartók (en su fase modal). "
            "Los modos no son tonalidades con accidentes: cada modo tiene una "
            "sonoridad intrínseca que justifica las elecciones de altura. "
            "Evita la dominante como agente cadencial. "
            "Usa pedales, ostinati, y cadencias modales (frigia, dórica). "
            "Justifica en términos del color propio de cada modo."
        ),
    },
    "auto": {
        "label": "Auto (el LLM elige)",
        "description": "El sistema elige la persona más adecuada según la descripción.",
        "preferred_modes": [],
        "preferred_arcs": [],
        "style_note": "",
        "system_addendum": "",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  DICCIONARIO SEMÁNTICO LOCAL
#  Para funcionar sin LLM. Mapea conceptos frecuentes a parámetros musicales
#  con justificaciones teóricas.
# ══════════════════════════════════════════════════════════════════════════════

SEMANTIC_DICTIONARY = {
    # ── ESTADOS EMOCIONALES ───────────────────────────────────────────────────
    "tristeza": {
        "aliases": ["triste", "sad", "melancólico", "melancolico", "melancholy",
                    "nostalgia", "nostálgico", "nostalgico", "pena", "dolor"],
        "key_mode": "minor",
        "tension_profile": "medium_low",
        "tempo_range": (52, 84),
        "density": "light",
        "harmony_complexity": "moderate",
        "arc": "tragedy",
        "justification": "La modalidad menor es la asociación más consolidada en la teoría tonal occidental (Mattheson, 1739). Tempo lento y densidad ligera evitan la agitación, sosteniendo la contemplación.",
    },
    "alegría": {
        "aliases": ["alegre", "happy", "feliz", "joyful", "festivo", "jubiloso", "animado"],
        "key_mode": "major",
        "tension_profile": "medium",
        "tempo_range": (108, 160),
        "density": "medium",
        "harmony_complexity": "simple",
        "arc": "hero",
        "justification": "Modo mayor, tempo allegro y armonías diatónicas simples son los marcadores canónicos de júbilo (Kirnberger, Affektenlehre).",
    },
    "ansiedad": {
        "aliases": ["ansioso", "anxious", "angustia", "inquietud", "nervioso",
                    "agitación", "agitacion", "urgencia", "urgente"],
        "key_mode": "minor",
        "tension_profile": "high_unstable",
        "tempo_range": (126, 176),
        "density": "high",
        "harmony_complexity": "complex",
        "arc": "hero",
        "justification": "Alta densidad rítmica con tensión armónica inestable y registro agudo: los correlatos físicos de la ansiedad (pulso acelerado, respiración corta) se traducen directamente en parámetros musicales.",
    },
    "calma": {
        "aliases": ["tranquilo", "sereno", "peaceful", "calm", "reposo",
                    "quietud", "paz", "placid", "plácido"],
        "key_mode": "major",
        "tension_profile": "low",
        "tempo_range": (52, 76),
        "density": "sparse",
        "harmony_complexity": "simple",
        "arc": "meditation",
        "justification": "Tensión armónica mínima, densidad esparsa y tempo lento reducen la carga cognitiva y producen reposo perceptual (Huron, Sweet Anticipation, 2006).",
    },
    "misterio": {
        "aliases": ["misterioso", "mysterious", "enigmático", "enigmatico",
                    "ambiguo", "oscuro", "inquietante", "siniestro"],
        "key_mode": "phrygian",
        "tension_profile": "medium_high",
        "tempo_range": (58, 96),
        "density": "light",
        "harmony_complexity": "complex",
        "arc": "mystery",
        "justification": "El modo frigio con su b2 crea una ambigüedad tonal que impide la resolución predecible. La densidad baja aumenta la tensión por ausencia.",
    },
    "grandiosidad": {
        "aliases": ["grandioso", "majestuoso", "épico", "epico", "heroico",
                    "poderoso", "sublime", "monumental", "imponente"],
        "key_mode": "major",
        "tension_profile": "high_climax",
        "tempo_range": (88, 120),
        "density": "high",
        "harmony_complexity": "complex",
        "arc": "hero",
        "justification": "Modo mayor con armonías extendidas, densidad alta y registro amplio activan la respuesta de 'escalofrío musical' asociada a lo sublime (Sloboda, 1991).",
    },
    "ternura": {
        "aliases": ["tierno", "delicado", "suave", "gentle", "sweet", "dulce",
                    "íntimo", "intimo", "cariño"],
        "key_mode": "major",
        "tension_profile": "low_warm",
        "tempo_range": (60, 88),
        "density": "light",
        "harmony_complexity": "moderate",
        "arc": "romance",
        "justification": "Registro medio-agudo, dinámica suave y cadencias evitadas o a la mediante (no a la tónica dura): la ternura evita la conclusividad.",
    },
    "expectativa": {
        "aliases": ["espera", "suspense", "anticipación", "anticipacion",
                    "tensión sin resolver", "tension sin resolver",
                    "esperar algo", "inminente"],
        "key_mode": "phrygian",
        "tension_profile": "sustained_high",
        "tempo_range": (60, 96),
        "density": "medium",
        "harmony_complexity": "complex",
        "arc": "mystery",
        "justification": "Dominantes sin resolver y semicadencias repetidas como marca estructural. El modo frigio resiste la resolución conclusiva (Ravel, Ma mère l'Oye).",
    },
    "melancolía": {
        "aliases": ["melancolia", "wistful", "añoranza", "anoranza",
                    "nostalgia urbana", "soledad", "alone", "desolación"],
        "key_mode": "dorian",
        "tension_profile": "medium",
        "tempo_range": (56, 84),
        "density": "light",
        "harmony_complexity": "moderate",
        "arc": "romance",
        "justification": "El dórico combina la oscuridad del menor con la VI mayor, creando la ambigüedad tonal característica de la melancolía (escala de blues, jazz modal).",
    },
    # ── IMÁGENES Y METÁFORAS ──────────────────────────────────────────────────
    "lluvia": {
        "aliases": ["rain", "llovizna", "gotas", "aguacero"],
        "key_mode": "minor",
        "tension_profile": "medium",
        "tempo_range": (76, 108),
        "density": "medium",
        "harmony_complexity": "moderate",
        "arc": "meditation",
        "justification": "Figuración rápida en registro agudo (gotas), bajo pedal estático (suelo), variación dinámica irregular (intensidad variable de la lluvia).",
    },
    "noche": {
        "aliases": ["night", "nocturno", "medianoche", "oscuridad nocturna"],
        "key_mode": "minor",
        "tension_profile": "low_dark",
        "tempo_range": (48, 72),
        "density": "sparse",
        "harmony_complexity": "moderate",
        "arc": "meditation",
        "justification": "El nocturno como género establece el modelo: modo menor, registro grave-medio, densidad esparsa, dinámicas pp-mp, cadencias evitadas (Chopin, Nocturnos op.9).",
    },
    "amanecer": {
        "aliases": ["alba", "dawn", "aurora", "despertar", "opening"],
        "key_mode": "major",
        "tension_profile": "crescendo",
        "tempo_range": (60, 96),
        "density": "sparse_to_medium",
        "harmony_complexity": "simple",
        "arc": "hero",
        "justification": "Crescendo de densidad y dinámica desde pp hasta mf-f, registro que sube del grave al agudo. Modelo: apertura del Don Juan de Strauss.",
    },
    "tormenta": {
        "aliases": ["storm", "tempestad", "turbulento", "caos", "furioso"],
        "key_mode": "minor",
        "tension_profile": "chaotic_high",
        "tempo_range": (132, 200),
        "density": "very_high",
        "harmony_complexity": "very_complex",
        "arc": "tragedy",
        "justification": "Densidad máxima, tempo allegro agitato, armonías disminuidas y aumentadas, registro extremo. Modelo: La Tempestad de Beethoven, op.31 no.2.",
    },
    "mar": {
        "aliases": ["ocean", "oceano", "olas", "waves", "marítimo", "maritimo"],
        "key_mode": "dorian",
        "tension_profile": "wave",
        "tempo_range": (60, 88),
        "density": "medium",
        "harmony_complexity": "moderate",
        "arc": "meditation",
        "justification": "Curva de tensión ondulatoria, figuración de arpegio ascendente-descendente (olas), modo dórico o mixolidio (asociados al mar en tradición celta y modal).",
    },
    "vacío": {
        "aliases": ["vacio", "empty", "desolado", "desierto", "abandono",
                    "nada", "silencio interior"],
        "key_mode": "phrygian",
        "tension_profile": "very_low",
        "tempo_range": (44, 64),
        "density": "minimal",
        "harmony_complexity": "simple",
        "arc": "meditation",
        "justification": "El vacío como principio compositivo: Feldman (Rothko Chapel) y Satie usan la ausencia de densidad como argumento. Los silencios son parte del material.",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  PERFILES DE TENSIÓN
#  Cada perfil es una función que genera un array de n_bars valores 0-1.
# ══════════════════════════════════════════════════════════════════════════════

def build_tension_profile(profile_name: str, n_bars: int) -> np.ndarray:
    """Construye un array de tensión 0-1 de longitud n_bars."""
    t = np.linspace(0, 1, n_bars)
    profiles = {
        "low":             lambda: np.ones(n_bars) * 0.2,
        "low_dark":        lambda: 0.15 + 0.1 * np.sin(np.pi * t),
        "low_warm":        lambda: 0.25 + 0.1 * np.sin(2 * np.pi * t),
        "medium_low":      lambda: np.ones(n_bars) * 0.38,
        "medium":          lambda: 0.4 + 0.15 * np.sin(np.pi * t),
        "medium_high":     lambda: 0.5 + 0.2 * np.sin(np.pi * t),
        "high_unstable":   lambda: np.clip(0.65 + 0.2 * np.sin(4 * np.pi * t) +
                                           0.1 * np.random.default_rng(42).standard_normal(n_bars), 0, 1),
        "high_climax":     lambda: np.clip(t ** 0.5 * 0.85 + 0.1, 0, 1),
        "sustained_high":  lambda: np.clip(0.60 + 0.12 * np.sin(3 * np.pi * t), 0, 1),
        "chaotic_high":    lambda: np.clip(0.7 + 0.25 * np.sin(6 * np.pi * t) +
                                           0.08 * np.random.default_rng(7).standard_normal(n_bars), 0, 1),
        "crescendo":       lambda: np.clip(0.1 + 0.85 * t, 0, 1),
        "decrescendo":     lambda: np.clip(0.95 - 0.85 * t, 0, 1),
        "arch":            lambda: 0.1 + 0.85 * np.sin(np.pi * t),
        "wave":            lambda: 0.5 + 0.4 * np.sin(2 * np.pi * t),
        "sparse_to_medium":lambda: np.clip(0.2 + 0.5 * t, 0, 1),
        "very_low":        lambda: np.ones(n_bars) * 0.1,
    }
    fn = profiles.get(profile_name, profiles["medium"])
    return np.clip(fn(), 0.0, 1.0)


def build_density_profile(density_label: str, n_bars: int) -> np.ndarray:
    """Construye un array de densidad 0-1."""
    t = np.linspace(0, 1, n_bars)
    mapping = {
        "minimal":         0.05,
        "sparse":          0.2,
        "light":           0.32,
        "sparse_to_medium":None,
        "medium":          0.55,
        "high":            0.75,
        "very_high":       0.92,
    }
    if density_label == "sparse_to_medium":
        return np.clip(0.2 + 0.4 * t, 0, 1)
    val = mapping.get(density_label, 0.5)
    return np.ones(n_bars) * val


# ══════════════════════════════════════════════════════════════════════════════
#  LLM: SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

LLM_SYSTEM_BASE = """\
Eres un teórico musical y compositor experto. Tu misión es traducir descripciones \
vagas de intención musical en planes de composición completamente fundamentados en \
teoría musical. Tu respuesta debe ser un objeto JSON válido EXCLUSIVAMENTE \
(sin backticks, sin texto antes o después, sin explicaciones fuera del JSON).

Debes razonar en CAPAS, explicitando cada decisión:
1. Semántica: qué emociones/imágenes contiene la descripción
2. Marco teórico: qué escuela o tradición compositiva aplica mejor
3. Contradicciones detectadas y cómo las resuelves
4. Cada parámetro musical con su justificación teórica

El JSON debe seguir EXACTAMENTE esta estructura:

{
  "semantic_layer": {
    "core_emotion": "emoción/imagen principal identificada",
    "secondary_emotions": ["lista", "de", "secundarias"],
    "contradictions": [
      {"detected": "descripción de la contradicción", "resolution": "cómo se resuelve"}
    ],
    "metaphors_found": ["metáforas o imágenes detectadas"]
  },
  "theoretical_frame": {
    "school": "tradición compositiva elegida (impresionismo/barroco/romanticismo/jazz/modal/etc.)",
    "persona_used": "nombre del teórico/compositor como referencia principal",
    "references": ["Compositor, Obra (año)", "..."],
    "governing_principle": "regla o principio teórico central que guía toda la obra"
  },
  "musical_decisions": [
    {
      "parameter": "nombre del parámetro",
      "value": "valor elegido",
      "justification": "justificación teórica específica (no genérica)",
      "rule_applied": "regla teórica concreta aplicada"
    }
  ],
  "warnings": ["advertencias sobre decisiones que podrían causar problemas"],
  "pipeline_params": {
    "narrator": {
      "arc": "nombre del arco",
      "bars": número_entero,
      "key": "tonalidad",
      "tempo": número_entero,
      "sections": "A:N,B:N,..."
    },
    "midi_dna_unified": {
      "--key": "tónica",
      "--mode": "auto",
      "--bars": número_entero,
      "--tempo": número_entero,
      "--mt-density": "especificación mt-*",
      "--mt-harmony-complexity": "especificación mt-*",
      "--mt-register": "especificación mt-*",
      "--mt-swing": "especificación mt-*",
      "--mt-emotion-morph": "especificación mt-* o null"
    },
    "reharmonizer": {
      "--strategy": "una o más estrategias separadas por espacio (ver lista abajo)",
      "--candidates": número_entero
    },
    "orchestrator": {
      "--template": "chamber, full, o strings_only"
    }
  },
  "tension_curves": {
    "tension":  [lista de N floats 0-1, uno por compás],
    "activity": [lista de N floats 0-1],
    "register": [lista de N floats 0-1],
    "harmony":  [lista de N floats 0-1],
    "swing":    [lista de N floats 0-1]
  },
  "essay": "párrafo de 80-120 palabras que explica la pieza como si fuera una nota de programa"
}

Parámetros del pipeline (formato mt-*):
  mt-* acepta especificaciones como "0:valor, 8:valor, 16:valor" donde el número
  es el compás y el valor es el nombre del preset o un float.
  Presets de density: sparse, light, medium, dense, full
  Presets de register: low, mid-low, mid, mid-high, high
  Presets de harmony-complexity: simple, diatonic, moderate, extended, chromatic
  Presets de swing: 0.0 a 1.0 como float

RESTRICCIONES CRÍTICAS — USA EXACTAMENTE ESTOS VALORES:

Arc de narrator (solo estos 7, NUNCA texto libre):
  hero, tragedy, romance, mystery, meditation, rondo, sonata

Tonalidades válidas: C, C#, Db, D, D#, Eb, E, F, F#, Gb, G, G#, Ab, A, A#, Bb, B

--mode de midi_dna_unified (modo de GENERACIÓN, NO escala):
  Siempre usa: "auto"

Estrategias de reharmonizer (una o más):
  diatonic, tritone, secondary, modal_interchange, chromatic_med,
  coltrane, neapolitan, pedal, minor_modal, baroque, impressionist

Templates de orchestrator (solo estos 3):
  chamber (default), full, strings_only

reharmonizer NO tiene --versions. Usa --candidates (entero, default 3).
"""

LLM_SYSTEM_DIALECTICAL = """\
Eres un teórico musical y compositor experto. Se te pide generar TRES lecturas \
alternativas de la misma intención musical, cada una desde un marco teórico distinto. \
Cada lectura debe producir parámetros y justificaciones diferentes pero igualmente \
válidas. Devuelve EXCLUSIVAMENTE JSON, sin texto adicional.

Estructura exacta:
{
  "description_received": "texto original",
  "readings": [
    {
      "label": "nombre corto de la lectura (ej: Debussy, Shostakovich, Satie)",
      "school": "tradición compositiva",
      "governing_principle": "principio teórico central",
      "key": "tonalidad",
      "mode": "modo",
      "tempo": número_entero,
      "arc": "uno de: hero, tragedy, romance, mystery, meditation, rondo, sonata",
      "tension_profile": "nombre del perfil de tensión",
      "density": "etiqueta de densidad",
      "harmony_strategy": "nombre de estrategia",
      "justification": "2-3 frases que justifican la lectura desde este marco teórico",
      "reference": "Compositor, Obra canónica de referencia"
    }
  ]
}
Genera exactamente 3 lecturas con marcos teóricos bien diferenciados.
"""

LLM_SYSTEM_INTERACTIVE_TURN = """\
Eres un teórico musical en diálogo. Tu tarea en cada turno es:
1. Confirmar un COMPROMISO TEÓRICO concreto (una regla o decisión musical) antes de preguntar.
2. Hacer UNA sola pregunta de refinamiento breve.
3. Actualizar el plan parcial.

Devuelve EXCLUSIVAMENTE JSON:
{
  "commitment": {
    "parameter": "parámetro que estás comprometiendo ahora",
    "value": "valor elegido",
    "justification": "justificación en 1 frase"
  },
  "question": "pregunta de refinamiento (máx 15 palabras)",
  "partial_plan": {
    "key": "tonalidad o null",
    "mode": "modo o null",
    "tempo": número o null,
    "arc": "arco o null",
    "tension_profile": "perfil o null",
    "density": "etiqueta o null",
    "harmony_strategy": "estrategia o null"
  },
  "is_complete": false
}
Cuando el plan esté suficientemente definido, pon "is_complete": true y añade
"final_summary": "resumen en 2 frases de las decisiones tomadas".
"""


def _add_persona_to_system(system: str, persona_name: str) -> str:
    """Añade el addendum de la persona al system prompt."""
    persona = PERSONAS.get(persona_name, PERSONAS["auto"])
    addendum = persona.get("system_addendum", "")
    if addendum:
        return system + f"\n\nPERSONA TEÓRICA ACTIVA:\n{addendum}"
    return system


# ══════════════════════════════════════════════════════════════════════════════
#  BACKEND LLM (mismo patrón que mutator.py)
# ══════════════════════════════════════════════════════════════════════════════

def _llm_debug_block(label: str, text: str):
    bar = "─" * max(0, 44 - len(label))
    print(f"\n  ┌─ [llm:debug] {label} {bar}")
    for line in text.splitlines():
        print(f"  │ {line}")
    print(f"  └{'─' * 58}")


def _clean_and_parse_json(raw: str) -> dict:
    """Limpia y parsea JSON de la respuesta LLM."""
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    # Eliminar posible texto antes del primer '{'
    first_brace = clean.find("{")
    if first_brace > 0:
        clean = clean[first_brace:]
    last_brace = clean.rfind("}")
    if last_brace >= 0:
        clean = clean[:last_brace + 1]
    return json.loads(clean)


def _call_anthropic(system: str, user_prompt: str, api_key: str,
                    model: str | None, verbose: bool,
                    debug: bool = False) -> dict:
    try:
        import anthropic
    except ImportError:
        if verbose:
            print("  [llm:anthropic] No instalado → pip install anthropic")
        return {}

    model = model or "claude-sonnet-4-6"
    if debug:
        _llm_debug_block("SYSTEM PROMPT (primeras 400 chars)", system[:400])
        _llm_debug_block("USER PROMPT", user_prompt)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        if debug:
            _llm_debug_block("RESPUESTA RAW (primeras 600 chars)", raw[:600])
        return _clean_and_parse_json(raw)
    except Exception as e:
        if verbose:
            print(f"  [llm:anthropic] Error: {e}")
        return {}


def _call_openai(system: str, user_prompt: str, api_key: str,
                 model: str | None, verbose: bool,
                 debug: bool = False) -> dict:
    try:
        import openai
    except ImportError:
        if verbose:
            print("  [llm:openai] No instalado → pip install openai")
        return {}

    model = model or "gpt-4o"
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            max_tokens=4096,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        if debug:
            _llm_debug_block("RESPUESTA RAW (primeras 600 chars)", raw[:600])
        return _clean_and_parse_json(raw)
    except Exception as e:
        if verbose:
            print(f"  [llm:openai] Error: {e}")
        return {}


LLM_BACKENDS = {
    "anthropic": (_call_anthropic, "ANTHROPIC_API_KEY",
                  ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"]),
    "openai":    (_call_openai,    "OPENAI_API_KEY",
                  ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]),
}


def call_llm(system: str, user_prompt: str,
             provider: str = "anthropic",
             api_key: str | None = None,
             model: str | None = None,
             verbose: bool = False,
             debug: bool = False) -> dict:
    """Llama al LLM configurado. Retorna dict vacío si falla."""
    provider = provider.lower()
    if provider not in LLM_BACKENDS:
        if verbose:
            print(f"  [llm] Proveedor desconocido: {provider}")
        return {}

    fn, env_var, _ = LLM_BACKENDS[provider]
    if api_key is None:
        api_key = os.environ.get(env_var)
    if not api_key:
        if verbose:
            print(f"  [llm:{provider}] Sin API key (usa --api-key o exporta {env_var})")
        return {}

    if verbose:
        print(f"  [llm:{provider}] modelo={model or '(default)'}")

    return fn(system, user_prompt, api_key, model, verbose, debug=debug)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR LOCAL (sin LLM)
#  Mapeo semántico + heurísticas para funcionar offline.
# ══════════════════════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    replacements = {
        "á":"a","é":"e","í":"i","ó":"o","ú":"u",
        "à":"a","è":"e","ì":"i","ò":"o","ù":"u",
        "ñ":"n","ü":"u",
    }
    t = text.lower().strip()
    for k, v in replacements.items():
        t = t.replace(k, v)
    return t


def local_parse(description: str, n_bars: int = 32,
                forced_key: str | None = None,
                forced_tempo: int | None = None,
                verbose: bool = False) -> dict:
    """
    Motor local: busca el concepto más cercano en SEMANTIC_DICTIONARY
    y construye un plan mínimo sin LLM.
    """
    t_norm = normalize_text(description)
    best_match = None
    best_score = 0

    for concept, data in SEMANTIC_DICTIONARY.items():
        terms = [normalize_text(concept)] + [normalize_text(a) for a in data.get("aliases", [])]
        score = sum(1 for term in terms if term in t_norm)
        if score > best_score:
            best_score = score
            best_match = (concept, data)

    if not best_match:
        # Fallback genérico
        if verbose:
            print("  [local] Sin coincidencias; usando plan genérico.")
        return _generic_plan(n_bars, forced_key, forced_tempo)

    concept_name, data = best_match
    if verbose:
        print(f"  [local] Concepto detectado: '{concept_name}' (score={best_score})")

    key_mode = data["key_mode"]
    tonic = forced_key if forced_key else _default_tonic_for_mode(key_mode)
    tempo = forced_tempo if forced_tempo else int(np.mean(data["tempo_range"]))
    tension_arr = build_tension_profile(data["tension_profile"], n_bars)
    density_arr = build_density_profile(data["density"], n_bars)

    # Parámetros mt-* básicos
    tension_spec = _array_to_mt_spec(tension_arr, n_bars)
    density_spec  = _density_label_to_mt_spec(data["density"], n_bars)

    return {
        "semantic_layer": {
            "core_emotion": concept_name,
            "secondary_emotions": [],
            "contradictions": [],
            "metaphors_found": [],
        },
        "theoretical_frame": {
            "school": "(motor local, sin LLM)",
            "persona_used": "heurísticas del diccionario semántico",
            "references": [],
            "governing_principle": data.get("justification", ""),
        },
        "musical_decisions": [
            {
                "parameter": "tonalidad",
                "value": f"{tonic} {key_mode}",
                "justification": data.get("justification", ""),
                "rule_applied": f"Modo {key_mode} para '{concept_name}'",
            },
            {
                "parameter": "tempo",
                "value": f"{tempo} BPM",
                "justification": f"Rango recomendado para '{concept_name}': {data['tempo_range']}",
                "rule_applied": "correlato tempo-emoción (Affektenlehre)",
            },
        ],
        "warnings": ["Plan generado con motor local (sin LLM). Justificaciones simplificadas."],
        "pipeline_params": {
            "narrator": {
                "arc": data.get("arc", "hero"),
                "bars": n_bars,
                "key": f"{tonic}",
                "tempo": tempo,
                "sections": _default_sections(data.get("arc", "hero"), n_bars),
            },
            "midi_dna_unified": {
                "--key": tonic,
                "--mode": "auto",
                "--bars": n_bars,
                "--tempo": tempo,
                "--mt-density": density_spec,
                "--mt-harmony-complexity": _harmony_label_to_spec(data["harmony_complexity"], n_bars),
                "--mt-register": "0:mid",
                "--mt-swing": "0:0.0",
                "--mt-emotion-morph": None,
            },
            "reharmonizer": {
                "--strategy": _mode_to_reharmonizer_strategy(key_mode),
                "--candidates": 3,
            },
            "orchestrator": {"--template": "chamber"},
        },
        "tension_curves": {
            "tension":  tension_arr.tolist(),
            "activity": density_arr.tolist(),
            "register": (np.ones(n_bars) * 0.5).tolist(),
            "harmony":  (np.ones(n_bars) * 0.4).tolist(),
            "swing":    (np.zeros(n_bars) + 0.05).tolist(),
        },
        "essay": (
            f"Una pieza de {n_bars} compases en {tonic} {key_mode}, a {tempo} BPM, "
            f"construida en torno a la idea de '{concept_name}'. "
            f"{data.get('justification', '')}"
        ),
    }


def _default_tonic_for_mode(mode: str) -> str:
    defaults = {
        "major": "C", "minor": "A", "dorian": "D", "phrygian": "E",
        "lydian": "F", "mixolydian": "G", "locrian": "B",
        "harmonic_minor": "A", "melodic_minor": "A",
    }
    return defaults.get(mode, "C")


def _generic_plan(n_bars: int, forced_key: str | None, forced_tempo: int | None) -> dict:
    tonic = forced_key or "C"
    tempo = forced_tempo or 120
    return {
        "semantic_layer": {"core_emotion": "indefinido", "secondary_emotions": [],
                            "contradictions": [], "metaphors_found": []},
        "theoretical_frame": {"school": "tonal clásico", "persona_used": "genérico",
                               "references": [], "governing_principle": "arco narrativo estándar"},
        "musical_decisions": [],
        "warnings": ["Descripción no reconocida. Plan genérico aplicado. Usa --verbose o --llm para mejor resultado."],
        "pipeline_params": {
            "narrator": {"arc": "hero", "bars": n_bars, "key": tonic, "tempo": tempo,
                          "sections": _default_sections("hero", n_bars)},
            "midi_dna_unified": {"--key": tonic, "--mode": "auto", "--bars": n_bars,
                                  "--tempo": tempo, "--mt-density": "0:medium",
                                  "--mt-harmony-complexity": "0:moderate",
                                  "--mt-register": "0:mid", "--mt-swing": "0:0.0",
                                  "--mt-emotion-morph": None},
            "reharmonizer": {"--strategy": "diatonic", "--candidates": 3},
            "orchestrator": {"--template": "chamber"},
        },
        "tension_curves": {
            "tension":  build_tension_profile("arch", n_bars).tolist(),
            "activity": (np.ones(n_bars) * 0.5).tolist(),
            "register": (np.ones(n_bars) * 0.5).tolist(),
            "harmony":  (np.ones(n_bars) * 0.4).tolist(),
            "swing":    (np.zeros(n_bars) + 0.05).tolist(),
        },
        "essay": f"Pieza en {tonic} mayor, {n_bars} compases, {tempo} BPM. Arco narrativo estándar.",
    }


def _default_sections(arc: str, n_bars: int) -> str:
    q = n_bars // 4
    h = n_bars // 2
    sections = {
        "hero":      f"intro:{q},development:{h - q},climax:{q},resolution:{n_bars - h - q},coda:{q}",
        "tragedy":   f"A:{h},climax:{q},fall:{n_bars - h - q}",
        "romance":   f"A:{q},B:{q},tension:{q},resolution:{n_bars - 3*q}",
        "mystery":   f"intro:{q},development:{h},revelation:{q},coda:{n_bars - h - 2*q}",
        "meditation":f"A:{h},B:{n_bars - h}",
        "rondo":     f"A:{q},B:{q},A:{q},C:{n_bars - 3*q}",
        "sonata":    f"exposition:{h},development:{q},recapitulation:{n_bars - h - q}",
        "custom":    f"A:{n_bars}",
    }
    return sections.get(arc, f"A:{n_bars}")


def _density_label_to_mt_spec(label: str, n_bars: int) -> str:
    presets = {
        "minimal":         "0:sparse",
        "sparse":          "0:sparse",
        "light":           "0:light",
        "sparse_to_medium":f"0:sparse, {n_bars//2}:medium",
        "medium":          "0:medium",
        "high":            "0:dense",
        "very_high":       "0:full",
    }
    return presets.get(label, "0:medium")


def _harmony_label_to_spec(label: str, n_bars: int) -> str:
    presets = {
        "simple":   "0:simple",
        "moderate": "0:moderate",
        "complex":  "0:extended",
        "very_complex": f"0:extended, {n_bars//2}:chromatic",
    }
    return presets.get(label, "0:moderate")


def _mode_to_reharmonizer_strategy(mode: str) -> str:
    mapping = {
        "major":         "diatonic",
        "minor":         "minor_modal",
        "dorian":        "modal_interchange",
        "phrygian":      "pedal",
        "lydian":        "impressionist",
        "mixolydian":    "secondary",
        "locrian":       "chromatic_med",
        "harmonic_minor":"neapolitan",
        "melodic_minor": "coltrane",
    }
    return mapping.get(mode, "diatonic")


def _array_to_mt_spec(arr: np.ndarray, n_bars: int) -> str:
    """Convierte array de valores 0-1 a especificación mt-* de midi_dna_unified."""
    n_points = min(8, len(arr))
    indices = np.round(np.linspace(0, len(arr) - 1, n_points)).astype(int)
    points = [(int(idx), float(np.clip(arr[idx], 0, 1))) for idx in indices]
    return ", ".join(f"{bar}:{val:.2f}" for bar, val in points)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL PLAN COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def build_user_prompt(description: str, n_bars: int,
                      forced_key: str | None,
                      forced_tempo: int | None,
                      ref_state: dict | None,
                      persona_name: str) -> str:
    """Construye el prompt de usuario para el LLM."""
    lines = [f'Descripción musical: "{description}"', ""]
    lines.append(f"Parámetros de la obra:")
    lines.append(f"  - Compases: {n_bars}")
    if forced_key:
        lines.append(f"  - Tonalidad forzada: {forced_key} (DEBES usar esta tonalidad)")
    if forced_tempo:
        lines.append(f"  - Tempo forzado: {forced_tempo} BPM (DEBES usar este tempo)")

    if ref_state:
        lines.append("")
        lines.append("MIDI de referencia (ancla la interpretación):")
        for k, v in ref_state.items():
            if k != "midi_path":
                lines.append(f"  - {k}: {v}")

    persona = PERSONAS.get(persona_name, PERSONAS["auto"])
    if persona_name != "auto" and persona.get("style_note"):
        lines.append("")
        lines.append(f"Persona teórica activa: {persona['label']}")
        lines.append(f"Nota de estilo: {persona['style_note']}")

    lines.append("")
    lines.append(f"Las curvas tension/activity/register/harmony/swing deben tener exactamente {n_bars} valores.")
    lines.append("Genera el plan completo en JSON según la estructura indicada.")

    return "\n".join(lines)


def sanitize_mt_spec(spec_val) -> str | None:
    """
    Normaliza un valor de flag --mt-* asegurando que el primer punto
    tenga siempre el prefijo de compás "0:".

    Casos que corrige:
      "0.0, 16:0.0, 32:0.0"   → "0:0.0, 16:0.0, 32:0.0"
      "sparse, 16:dense"       → "0:sparse, 16:dense"
      "0:sparse, 16:dense"     → sin cambio (ya correcto)
      None / ""                → None
    """
    if not spec_val:
        return None
    s = str(spec_val).strip()
    if not s:
        return None

    parts = [p.strip() for p in s.split(',')]
    if parts and ':' not in parts[0]:
        parts[0] = '0:' + parts[0]
    return ', '.join(parts)


def apply_forced_params(plan: dict, forced_key: str | None,
                        forced_tempo: int | None, n_bars: int) -> dict:
    """
    Sobreescribe parámetros forzados y sanitiza valores LLM a contratos reales.
    """
    if not plan:
        return plan

    pp = plan.get("pipeline_params", {})

    # ── Sanitizar arc: texto libre → enum válido de narrator ─────────────────
    narrator_pp = pp.get("narrator", {})
    if narrator_pp:
        raw_arc = narrator_pp.get("arc", "hero")
        narrator_pp["arc"] = sanitize_arc(raw_arc)

    # ── Sanitizar orchestrator template ──────────────────────────────────────
    orch_pp = pp.get("orchestrator", {})
    if orch_pp:
        raw_tpl = orch_pp.get("--template", "chamber")
        orch_pp["--template"] = sanitize_orchestrator_template(raw_tpl)

    # ── Sanitizar reharmonizer: --versions → --candidates, estrategias ───────
    reharm_pp = pp.get("reharmonizer", {})
    if reharm_pp:
        # Eliminar --versions si el LLM lo puso
        if "--versions" in reharm_pp:
            candidates = reharm_pp.pop("--versions")
            reharm_pp.setdefault("--candidates", candidates)
        # Sanitizar estrategias
        raw_strat = reharm_pp.get("--strategy", "diatonic")
        reharm_pp["--strategy"] = " ".join(sanitize_reharmonizer_strategies(raw_strat))

    # ── Sanitizar midi_dna_unified --mode y flags --mt-* ────────────────────
    dna_pp = pp.get("midi_dna_unified", {})
    if dna_pp:
        # --mode en midi_dna_unified es el modo de GENERACIÓN, no la escala
        # Valores válidos: auto, rhythm_melody, harmony_melody, full_blend, ...
        # El LLM a veces pone major/minor/phrygian etc. → corregir a "auto"
        dna_valid_modes = {
            "auto", "rhythm_melody", "harmony_melody", "full_blend",
            "rhythm_only", "melody_only", "harmony_only",
        }
        mode_val = str(dna_pp.get("--mode", "auto"))
        if mode_val not in dna_valid_modes:
            dna_pp["--mode"] = "auto"

        # Sanitizar flags --mt-*: garantizar que el primer punto tenga prefijo "0:"
        for mt_flag in ("--mt-density", "--mt-harmony-complexity", "--mt-register",
                        "--mt-swing", "--mt-emotion-morph", "--mt-rhythm-morph",
                        "--mt-acc-style"):
            if mt_flag in dna_pp:
                dna_pp[mt_flag] = sanitize_mt_spec(dna_pp[mt_flag])

    if forced_key:
        tonic, mode = _parse_key_string(forced_key)
        if narrator_pp:
            narrator_pp["key"] = tonic
            narrator_pp["mode"] = mode
        if dna_pp:
            # parse_key_arg en midi_dna_unified acepta "C# minor" (tónica + modo)
            dna_pp["--key"] = f"{tonic} {mode}" if mode != "major" else tonic

    if forced_tempo:
        if narrator_pp:
            narrator_pp["tempo"] = forced_tempo
        if dna_pp:
            dna_pp["--tempo"] = forced_tempo

    # Aseguramos que bars sea correcto
    if narrator_pp:
        narrator_pp["bars"] = n_bars
    if dna_pp:
        dna_pp["--bars"] = n_bars

    # Normalizar longitud de curvas
    curves = plan.get("tension_curves", {})
    for curve_name in ["tension", "activity", "register", "harmony", "swing"]:
        if curve_name in curves:
            arr = np.array(curves[curve_name], dtype=float)
            if len(arr) != n_bars:
                arr = np.interp(
                    np.linspace(0, 1, n_bars),
                    np.linspace(0, 1, len(arr)),
                    arr
                )
            curves[curve_name] = np.clip(arr, 0, 1).tolist()

    return plan


def _parse_key_string(key_str: str) -> tuple[str, str]:
    """'Dm' → ('D', 'minor'), 'F#' → ('F#', 'major'), 'Bb minor' → ('Bb', 'minor')."""
    key_str = key_str.strip()
    if " " in key_str:
        parts = key_str.split()
        tonic = parts[0]
        mode_str = parts[1].lower()
        mode = "minor" if mode_str in ("m", "minor", "menor") else "major"
        return tonic, mode
    if key_str.endswith("m") and len(key_str) >= 2:
        return key_str[:-1], "minor"
    return key_str, "major"


def generate_plan(description: str,
                  n_bars: int = 32,
                  forced_key: str | None = None,
                  forced_tempo: int | None = None,
                  ref_state: dict | None = None,
                  persona: str = "auto",
                  arc: str | None = None,
                  use_llm: bool = True,
                  provider: str = "anthropic",
                  api_key: str | None = None,
                  model: str | None = None,
                  verbose: bool = False,
                  debug: bool = False) -> dict:
    """
    Genera el plan completo a partir de la descripción.
    Si use_llm=False o el LLM falla, usa el motor local.
    """
    plan = {}

    if use_llm:
        system = _add_persona_to_system(LLM_SYSTEM_BASE, persona)
        user_prompt = build_user_prompt(
            description, n_bars, forced_key, forced_tempo, ref_state, persona
        )
        plan = call_llm(system, user_prompt, provider, api_key, model, verbose, debug)

    if not plan:
        if use_llm and verbose:
            print("  [warn] LLM no disponible o falló. Usando motor local.")
        plan = local_parse(description, n_bars, forced_key, forced_tempo, verbose)

    # Inyectar arc forzado si se especificó, sanitizando
    if arc and arc != "auto":
        pp = plan.get("pipeline_params", {})
        valid_arc = sanitize_arc(arc)
        if pp.get("narrator"):
            pp["narrator"]["arc"] = valid_arc
            pp["narrator"]["sections"] = _default_sections(valid_arc, n_bars)

    plan = apply_forced_params(plan, forced_key, forced_tempo, n_bars)
    return plan


def generate_dialectical(description: str,
                          n_bars: int,
                          provider: str, api_key: str | None, model: str | None,
                          verbose: bool, debug: bool) -> list[dict]:
    """Genera 3 lecturas dialecticas de la misma descripción."""
    user_prompt = (
        f'Descripción: "{description}"\n'
        f"Compases: {n_bars}\n"
        "Genera exactamente 3 lecturas alternativas con marcos teóricos bien diferenciados."
    )
    result = call_llm(LLM_SYSTEM_DIALECTICAL, user_prompt,
                      provider, api_key, model, verbose, debug)
    return result.get("readings", [])


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════

def export_theorist_json(plan: dict, description: str, output_base: str,
                          description_args: dict) -> str:
    """Exporta el plan completo como .theorist.json."""
    output = {
        "version": VERSION,
        "generated": datetime.now().isoformat(),
        "description_original": description,
        "args": description_args,
        **plan,
    }
    path = f"{output_base}.theorist.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return path


def export_curves_json(plan: dict, output_base: str) -> str:
    """Exporta las curvas en el formato exacto de tension_designer.py."""
    curves = plan.get("tension_curves", {})
    path = f"{output_base}.curves.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2, ensure_ascii=False)
    return path


def export_narrator_plan(plan: dict, output_base: str) -> str:
    """Exporta un plan.json compatible con narrator.py."""
    pp = plan.get("pipeline_params", {})
    narrator_params = pp.get("narrator", {})

    arc      = narrator_params.get("arc", "hero")
    n_bars   = narrator_params.get("bars", 32)
    key      = narrator_params.get("key", "C")
    tempo    = narrator_params.get("tempo", 120)
    sections_str = narrator_params.get("sections", f"A:{n_bars}")

    # Parsear secciones
    sections = []
    for part in sections_str.split(","):
        part = part.strip()
        if ":" in part:
            label, bars_str = part.split(":", 1)
            try:
                bars = int(bars_str.strip())
            except ValueError:
                bars = n_bars // 4
            sections.append({
                "id": label.strip().lower().replace(" ", "_"),
                "label": label.strip(),
                "bars": bars,
                "output": f"seccion_{label.strip()}.mid",
            })

    curves = plan.get("tension_curves", {})

    narrator_plan = {
        "arc": arc,
        "bars": n_bars,
        "key": key,
        "tempo": tempo,
        "sections": sections,
        "curves": curves,
        "pipeline": [
            {
                "step": "midi_dna_unified",
                "params": pp.get("midi_dna_unified", {}),
            },
            {
                "step": "reharmonizer",
                "params": pp.get("reharmonizer", {}),
            },
            {
                "step": "orchestrator",
                "params": pp.get("orchestrator", {}),
            },
        ],
        "theorist_source": True,
    }

    path = f"{output_base}_plan.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(narrator_plan, f, indent=2, ensure_ascii=False)
    return path


def export_pipeline_yaml(plan: dict, output_base: str,
                          ref_midi: str | None = None) -> str:
    """Exporta un plan.yaml para runner.py."""
    pp = plan.get("pipeline_params", {})
    narrator_params = pp.get("narrator", {})
    dna_params      = pp.get("midi_dna_unified", {})
    reharm_params   = pp.get("reharmonizer", {})
    orch_params     = pp.get("orchestrator", {})

    arc    = narrator_params.get("arc", "hero")
    n_bars = narrator_params.get("bars", 32)
    key    = narrator_params.get("key", "C")
    mode   = narrator_params.get("mode", "major")
    key_full = f"{key} {mode}" if mode and mode != "major" else key
    tempo  = narrator_params.get("tempo", 120)

    # Construir el comando CLI de midi_dna_unified
    dna_cmd_parts = ["python midi_dna_unified.py"]
    for flag, val in dna_params.items():
        if val is not None and flag.startswith("--"):
            dna_cmd_parts.append(f"{flag} {val}")
    dna_cmd = " \\\n        ".join(dna_cmd_parts)

    reharm_strategy = reharm_params.get("--strategy", "diatonic")
    reharm_versions = reharm_params.get("--versions", 2)
    orch_template   = orch_params.get("--template", "chamber")

    # Sanitizar todos los valores antes de emitir YAML
    arc           = sanitize_arc(arc)
    orch_template = sanitize_orchestrator_template(orch_template)
    reharm_strats = sanitize_reharmonizer_strategies(reharm_strategy)
    reharm_strategy_str = " ".join(reharm_strats)
    reharm_candidates   = reharm_params.get("--candidates",
                            reharm_params.get("--versions", 3))

    # tension_designer --no-gui necesita midi fuente + --preset, NO --load sin midi
    # Usamos el .mid generado por midi_dna_unified como fuente
    tension_preset = ARC_TO_TENSION_PRESET.get(arc, "arch")

    ref_stem = Path(ref_midi).stem if ref_midi else None

    # Fragmentos que harvester generará, usados como fuentes posicionales
    # para midi_dna_unified y tension_designer
    if ref_stem:
        # Pasar el MIDI de referencia directamente como fuente posicional.
        # harvester habrá extraído ya los fragmentos antes de este paso,
        # pero midi_dna_unified también puede usar el MIDI completo como fuente.
        dna_cmd = dna_cmd.replace("python midi_dna_unified.py",
                                   f'python midi_dna_unified.py "{ref_midi}"')
        td_ref_arg = f' "{ref_midi}"'
    else:
        td_ref_arg = ""

    yaml_lines = [
        f"# Plan de pipeline generado por theorist.py v{VERSION}",
        f"# Descripción: {plan.get('description_original', '(sin descripción)')}",
        f"# Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    ]
    if ref_midi:
        yaml_lines.append(f"# MIDI de referencia: {ref_midi}")
    yaml_lines += [
        "",
        "obra:",
        f"  arc: {arc}",
        f"  bars: {n_bars}",
        f"  key: {key_full}",
        f"  tempo: {tempo}",
    ]
    if ref_midi:
        yaml_lines.append(f"  ref_midi: {ref_midi}")
    yaml_lines += ["", "steps:", ""]

    # Paso 0 (opcional): harvester extrae fragmentos del MIDI de referencia
    if ref_midi:
        yaml_lines += [
            "  - name: harvester",
            f'    cmd: python harvester.py "{ref_midi}" --mode motif cadence texture --report',
            f"    output: {ref_stem}.harvest_report.json",
            "",
        ]

    yaml_lines += [
        "  - name: narrator",
        f"    cmd: python narrator.py --arc {arc} --bars {n_bars} --key {key_full} --tempo {tempo} --no-gui --export-curves --export-yaml",
        f"    output: {output_base}_plan.json",
        "",
        "  - name: midi_dna_unified",
        f"    cmd: {dna_cmd} --export-fingerprint --output {output_base}.mid",
        f"    output: {output_base}.mid",
        "",
        "  - name: tension_designer",
        f"    cmd: python tension_designer.py {output_base}.mid{td_ref_arg} --no-gui --preset {tension_preset} --bars {n_bars} --output {output_base}_tension.mid",
        f"    output: {output_base}_tension.mid",
        "",
        "  - name: reharmonizer",
        f"    cmd: python reharmonizer.py {output_base}.mid --strategy {reharm_strategy_str} --candidates {reharm_candidates}",
        "",
        "  - name: orchestrator",
        f"    cmd: python orchestrator.py {output_base}.mid --template {orch_template} --auto-fp --output {output_base}_orquestado.mid",
        f"    output: {output_base}_orquestado.mid",
        "",
    ]

    path = f"{output_base}_plan.yaml"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines))
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  IMPRESIÓN DEL RAZONAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def print_reasoning(plan: dict, verbose: bool = False):
    """Imprime el razonamiento del plan de forma legible."""
    sep = "─" * 62

    print(f"\n{'═' * 62}")
    print(f"  THEORIST v{VERSION}  —  Plan de composición")
    print(f"{'═' * 62}")

    # Capa semántica
    sem = plan.get("semantic_layer", {})
    print(f"\n  [1/5] Interpretación semántica")
    print(f"  {sep}")
    print(f"  Emoción central: {sem.get('core_emotion', '—')}")
    if sem.get("secondary_emotions"):
        print(f"  Secundarias:     {', '.join(sem['secondary_emotions'])}")
    if sem.get("contradictions"):
        for c in sem["contradictions"]:
            print(f"  ⚠  Contradicción: {c.get('detected', '')}")
            print(f"     Resolución:    {c.get('resolution', '')}")

    # Marco teórico
    frame = plan.get("theoretical_frame", {})
    print(f"\n  [2/5] Marco teórico")
    print(f"  {sep}")
    print(f"  Escuela:    {frame.get('school', '—')}")
    print(f"  Referencia: {frame.get('persona_used', '—')}")
    if frame.get("references"):
        for ref in frame["references"][:3]:
            print(f"    • {ref}")
    if frame.get("governing_principle"):
        wrapped = textwrap.fill(
            frame["governing_principle"], width=56,
            initial_indent="  Principio: ", subsequent_indent="             "
        )
        print(wrapped)

    # Decisiones musicales
    decisions = plan.get("musical_decisions", [])
    print(f"\n  [3/5] Decisiones musicales ({len(decisions)} parámetros)")
    print(f"  {sep}")
    for d in decisions:
        print(f"  ◆ {d.get('parameter', '?')}: {d.get('value', '?')}")
        if d.get("rule_applied"):
            print(f"    Regla:          {d['rule_applied']}")
        if verbose and d.get("justification"):
            just = textwrap.fill(
                d["justification"], width=54,
                initial_indent="    Justificación: ",
                subsequent_indent="                  "
            )
            print(just)

    # Pipeline params
    pp = plan.get("pipeline_params", {})
    dna = pp.get("midi_dna_unified", {})
    print(f"\n  [4/5] Parámetros de pipeline")
    print(f"  {sep}")
    if dna:
        key_val  = dna.get("--key", "?")
        mode_val = dna.get("--mode", "auto")
        bars_val = dna.get("--bars", "?")
        tempo_val = dna.get("--tempo", "?")
        print(f"  midi_dna_unified: {key_val} {mode_val}, {bars_val} bars, {tempo_val} BPM")
        for flag in ["--mt-density", "--mt-harmony-complexity", "--mt-register", "--mt-swing"]:
            if dna.get(flag):
                print(f"    {flag} \"{dna[flag]}\"")
    reharm = pp.get("reharmonizer", {})
    if reharm.get("--strategy"):
        candidates = reharm.get("--candidates", reharm.get("--versions", 3))
        print(f"  reharmonizer:     strategy={reharm['--strategy']}, candidates={candidates}")
    orch = pp.get("orchestrator", {})
    if orch.get("--template"):
        print(f"  orchestrator:     template={orch['--template']}")

    # Advertencias
    warnings = plan.get("warnings", [])
    if warnings:
        print(f"\n  [!] Advertencias")
        print(f"  {sep}")
        for w in warnings:
            wrapped = textwrap.fill(w, width=58, initial_indent="  → ", subsequent_indent="    ")
            print(wrapped)

    # Nota de programa
    essay = plan.get("essay", "")
    if essay:
        print(f"\n  [5/5] Nota de programa")
        print(f"  {sep}")
        wrapped = textwrap.fill(essay, width=58, initial_indent="  ", subsequent_indent="  ")
        print(wrapped)

    print(f"\n{'═' * 62}\n")


def print_dialectical(readings: list[dict]):
    """Imprime las 3 lecturas dialecticas."""
    print(f"\n{'═' * 62}")
    print(f"  THEORIST v{VERSION}  —  Modo dialectico: 3 lecturas")
    print(f"{'═' * 62}")

    for i, r in enumerate(readings, 1):
        print(f"\n  [Lectura {i}] {r.get('label', f'Lectura {i}')}")
        print(f"  {'─' * 58}")
        print(f"  Escuela:   {r.get('school', '—')}")
        print(f"  Tonalidad: {r.get('key', '?')} {r.get('mode', '')}")
        print(f"  Tempo:     {r.get('tempo', '?')} BPM")
        print(f"  Arco:      {r.get('arc', '?')}")
        print(f"  Armonía:   {r.get('harmony_strategy', '?')}")
        if r.get("governing_principle"):
            wrapped = textwrap.fill(
                r["governing_principle"], width=54,
                initial_indent="  Principio: ", subsequent_indent="             "
            )
            print(wrapped)
        if r.get("justification"):
            wrapped = textwrap.fill(
                r["justification"], width=56,
                initial_indent="  Razón: ", subsequent_indent="         "
            )
            print(wrapped)
        if r.get("reference"):
            print(f"  Ref:       {r['reference']}")

    print(f"\n{'═' * 62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DE MIDI DE REFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def read_ref_state(midi_path: str, verbose: bool = False) -> dict:
    """Lee el estado musical de un MIDI de referencia."""
    state = {"midi_path": midi_path}
    fp_candidates = [
        str(Path(midi_path).with_suffix(".fingerprint.json")),
        str(Path(midi_path)) + ".fingerprint.json",
    ]
    for fp_path in fp_candidates:
        if Path(fp_path).exists():
            try:
                with open(fp_path, "r") as f:
                    fp = json.load(f)
                meta = fp.get("meta", {})
                state["key_tonic"]       = meta.get("key_tonic", "C")
                state["key_mode"]        = meta.get("key_mode", "major")
                state["tempo_bpm"]       = float(meta.get("tempo_bpm", 120))
                state["n_bars"]          = int(meta.get("n_bars", 16))
                tc = fp.get("tension_curve", {})
                vals = tc.get("values", None)
                if vals:
                    state["tension_mean"] = float(np.mean(vals))
                if verbose:
                    print(f"  [ref] Fingerprint cargado: {fp_path}")
                return state
            except Exception as e:
                if verbose:
                    print(f"  [ref] No se pudo leer fingerprint: {e}")

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
            mean_note = float(np.mean(notes))
            state["pitch_mean"] = round(mean_note, 1)
        if verbose:
            print(f"  [ref] Analizado con mido (sin fingerprint)")
    except Exception as e:
        if verbose:
            print(f"  [ref] Análisis mido falló: {e}")

    return state


# ══════════════════════════════════════════════════════════════════════════════
#  EJECUCIÓN DEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def _find_script(name: str) -> str | None:
    here = Path(__file__).parent / name
    if here.exists():
        return str(here)
    cwd = Path.cwd() / name
    if cwd.exists():
        return str(cwd)
    found = shutil.which(name)
    return found


PIPELINE_ORDER = [
    "tension_designer.py",
    "midi_dna_unified.py",
    "reharmonizer.py",
    "orchestrator.py",
]


def execute_pipeline(plan: dict, output_base: str,
                     until: str | None = None,
                     verbose: bool = False,
                     ref_midi: str | None = None) -> bool:
    """
    Ejecuta el pipeline completo o hasta el script indicado.
    Retorna True si todos los pasos ejecutados tuvieron éxito.
    """
    pp = plan.get("pipeline_params", {})
    curves_file = f"{output_base}.curves.json"
    midi_out    = f"{output_base}.mid"

    steps = []
    ref_stem = Path(ref_midi).stem if ref_midi else None

    # harvester — extrae fragmentos del MIDI de referencia si se proporcionó
    if ref_midi and Path(ref_midi).exists():
        script_harv = _find_script("harvester.py")
        if script_harv:
            steps.append({
                "name": "harvester",
                "script": script_harv,
                "cmd": [
                    sys.executable, script_harv,
                    ref_midi,
                    "--mode", "motif", "cadence", "texture",
                    "--report",
                ],
            })

    # tension_designer — necesita MIDI fuente + --preset (no --load sin midi)
    # Si hay ref_midi, se pasa también como fuente de análisis tímbrico
    script_td = _find_script("tension_designer.py")
    if script_td:
        arc_val = sanitize_arc(pp.get("narrator", {}).get("arc", "hero"))
        tension_preset = ARC_TO_TENSION_PRESET.get(arc_val, "arch")
        n_bars_td = pp.get("narrator", {}).get("bars", 32)
        td_cmd = [sys.executable, script_td, midi_out]
        if ref_midi and Path(ref_midi).exists():
            td_cmd.append(ref_midi)
        td_cmd += [
            "--no-gui",
            "--preset", tension_preset,
            "--bars", str(n_bars_td),
            "--output", f"{output_base}_tension.mid",
        ]
        steps.append({"name": "tension_designer", "script": script_td, "cmd": td_cmd})

    # midi_dna_unified — fragmentos cosechados como fuentes posicionales
    script_dna = _find_script("midi_dna_unified.py")
    if script_dna:
        dna_params = pp.get("midi_dna_unified", {})
        cmd = [sys.executable, script_dna]
        if ref_midi and ref_stem:
            ref_dir = str(Path(ref_midi).parent)
            harvested = sorted(
                [str(p) for p in Path(ref_dir).glob(f"{ref_stem}.motif_*.mid")] +
                [str(p) for p in Path(ref_dir).glob(f"{ref_stem}.cadence_*.mid")]
            )
            cmd.extend(harvested)
        for flag, val in dna_params.items():
            if val is not None and flag.startswith("--"):
                cmd += [flag, str(val)]
        cmd += ["--export-fingerprint", "--output", midi_out]
        steps.append({"name": "midi_dna_unified", "script": script_dna, "cmd": cmd})

    # reharmonizer — --strategy acepta múltiples valores; no tiene --versions
    script_rh = _find_script("reharmonizer.py")
    if script_rh:
        rh = pp.get("reharmonizer", {})
        cmd = [sys.executable, script_rh, midi_out]
        for flag, val in rh.items():
            if val is None or not flag.startswith("--"):
                continue
            if flag == "--strategy":
                # estrategias como argumentos separados
                for s in sanitize_reharmonizer_strategies(val):
                    cmd += ["--strategy", s]
            elif flag == "--versions":
                # alias → --candidates
                cmd += ["--candidates", str(val)]
            else:
                cmd += [flag, str(val)]
        steps.append({"name": "reharmonizer", "script": script_rh, "cmd": cmd})

    # orchestrator
    script_orch = _find_script("orchestrator.py")
    if script_orch:
        orch = pp.get("orchestrator", {})
        orch_template = sanitize_orchestrator_template(orch.get("--template", "chamber"))
        orch_out = f"{output_base}_orquestado.mid"
        cmd = [
            sys.executable, script_orch, midi_out,
            "--template", orch_template,
            "--auto-fp",
            "--output", orch_out,
        ]
        steps.append({"name": "orchestrator", "script": script_orch, "cmd": cmd})

    if not steps:
        print("  [pipeline] No se encontraron scripts del pipeline en este directorio.")
        print("             Exportados los archivos de plan. Ejecútalos manualmente.")
        return False

    for step in steps:
        name = step["name"]
        if until and name not in PIPELINE_ORDER[:PIPELINE_ORDER.index(f"{until}.py") + 1] if f"{until}.py" in PIPELINE_ORDER else False:
            break

        script_short = Path(step["script"]).name
        print(f"  → Ejecutando {script_short}...")
        if verbose:
            print(f"    {' '.join(step['cmd'][:6])} ...")

        try:
            result = subprocess.run(
                step["cmd"],
                capture_output=not verbose,
                text=True, timeout=300
            )
            if result.returncode != 0 and verbose:
                print(f"    [warn] {script_short} salió con código {result.returncode}")
            else:
                print(f"    ✓ {script_short}")
        except subprocess.TimeoutExpired:
            print(f"    [warn] {script_short} tardó demasiado (timeout 300s)")
        except Exception as e:
            print(f"    [error] {script_short}: {e}")

        if until and name == until.replace(".py", ""):
            print(f"  [pipeline] Detenido en {name} (--until)")
            break

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MODO INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════

def interactive_loop(args):
    """Bucle conversacional con compromisos teóricos por turno."""
    print(f"\n{'═' * 62}")
    print(f"  THEORIST v{VERSION}  —  Modo conversacional")
    print(f"{'═' * 62}")
    print("  En cada turno el sistema se compromete con una decisión teórica")
    print("  y hace UNA pregunta de refinamiento.")
    print("  Escribe 'generar' cuando estés listo para producir el plan.")
    print("  Escribe 'salir' para terminar sin generar.")
    print()

    api_key  = args.api_key or os.environ.get(
        "ANTHROPIC_API_KEY" if args.llm_provider == "anthropic" else "OPENAI_API_KEY"
    )

    conversation_history = []
    partial_plan = {}
    description_parts = []

    turn = 0
    while True:
        turn += 1
        try:
            user_input = input(f"  [turno {turn}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Terminando.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("salir", "exit", "quit", "q"):
            break
        if user_input.lower() in ("generar", "generate", "go", "listo"):
            break

        description_parts.append(user_input)
        conversation_history.append({"role": "user", "content": user_input})

        # Construir prompt con historial
        history_str = "\n".join(
            f"  {m['role'].upper()}: {m['content']}"
            for m in conversation_history
        )
        plan_str = json.dumps(partial_plan, ensure_ascii=False) if partial_plan else "{}"

        llm_user_prompt = (
            f"Conversación hasta ahora:\n{history_str}\n\n"
            f"Plan parcial acumulado:\n{plan_str}\n\n"
            f"Compases planificados: {args.bars}\n"
            "Genera el próximo turno: un compromiso teórico + una pregunta."
        )

        response = call_llm(
            LLM_SYSTEM_INTERACTIVE_TURN,
            llm_user_prompt,
            provider=args.llm_provider,
            api_key=api_key,
            model=args.llm_model,
            verbose=args.verbose,
        )

        if response:
            commitment = response.get("commitment", {})
            question   = response.get("question", "")
            new_partial = response.get("partial_plan", {})
            is_complete = response.get("is_complete", False)

            # Actualizar plan parcial
            for k, v in new_partial.items():
                if v is not None:
                    partial_plan[k] = v

            # Mostrar compromiso
            if commitment:
                print(f"\n  ◆ Compromiso: {commitment.get('parameter', '?')} = {commitment.get('value', '?')}")
                print(f"    {commitment.get('justification', '')}")

            # Mostrar pregunta
            if question and not is_complete:
                print(f"\n  ? {question}")

            conversation_history.append({"role": "assistant", "content": json.dumps(response)})

            if is_complete:
                print(f"\n  {response.get('final_summary', 'Plan completo.')}")
                print("  Escribe 'generar' para producir el plan o continúa refinando.")
        else:
            # Sin LLM: mostrar estado parcial
            print(f"  (Sin LLM) Descripción acumulada: {' / '.join(description_parts)}")
            print("  Escribe 'generar' cuando estés listo.")

    # Generar plan final
    full_description = " ".join(description_parts)
    if not full_description:
        print("  Sin descripción. Terminando.")
        return

    print(f"\n  Generando plan para: \"{full_description}\"")

    plan = generate_plan(
        full_description,
        n_bars=args.bars,
        forced_key=getattr(args, "key", None),
        forced_tempo=getattr(args, "tempo", None),
        persona=getattr(args, "persona", "auto"),
        arc=getattr(args, "arc", None),
        use_llm=not args.no_llm,
        provider=args.llm_provider,
        api_key=api_key,
        model=args.llm_model,
        verbose=args.verbose,
    )

    _run_export_and_execute(plan, full_description, args, vars(args))


# ══════════════════════════════════════════════════════════════════════════════
#  FLUJO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _run_export_and_execute(plan: dict, description: str, args, description_args: dict):
    """Exporta los archivos y opcionalmente ejecuta el pipeline."""
    if args.dry_run:
        print_reasoning(plan, verbose=True)
        print("  [dry-run] No se generaron archivos.")
        return

    output_base = args.output

    print_reasoning(plan, verbose=args.verbose)

    # Exportar
    path_theorist = export_theorist_json(plan, description, output_base, description_args)
    path_curves   = export_curves_json(plan, output_base)
    path_narrator = export_narrator_plan(plan, output_base)
    path_yaml     = export_pipeline_yaml(plan, output_base, ref_midi=args.ref_midi)

    print(f"  Archivos generados:")
    print(f"    ◆ {path_theorist}   (plan completo con justificaciones)")
    print(f"    ◆ {path_curves}     (curvas → tension_designer)")
    print(f"    ◆ {path_narrator}   (plan → narrator GUI)")
    print(f"    ◆ {path_yaml}       (pipeline → runner.py)")
    print()

    # Ejecutar pipeline si se pidió
    if args.execute:
        print(f"  Ejecutando pipeline...")
        execute_pipeline(plan, output_base, until=args.until, verbose=args.verbose,
                       ref_midi=args.ref_midi)


def print_personas():
    """Lista todas las personas teóricas disponibles."""
    print(f"\n{'═' * 62}")
    print(f"  Personas teóricas disponibles  —  THEORIST v{VERSION}")
    print(f"{'═' * 62}")
    for name, data in PERSONAS.items():
        if name == "auto":
            continue
        print(f"\n  --persona {name}")
        print(f"    {data['label']}")
        wrapped = textwrap.fill(
            data["description"], width=54,
            initial_indent="    ", subsequent_indent="    "
        )
        print(wrapped)
        if data.get("preferred_modes"):
            print(f"    Modos preferidos: {', '.join(data['preferred_modes'][:4])}")
    print(f"\n  --persona auto  (el sistema elige según la descripción)")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="theorist.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            THEORIST v{VERSION} — Intención musical → teoría fundamentada → pipeline
            ─────────────────────────────────────────────────────────────────────
            Traduce descripciones libres de intención musical en planes de
            composición completos, justificados en teoría musical, listos
            para alimentar el pipeline de midi_dna_unified.

            Ejemplos:
              python theorist.py "esperar algo que nunca llega"
              python theorist.py "melancolía urbana" --key Dm --bars 32
              python theorist.py "tormenta que se calma" --persona romantic --execute
              python theorist.py "lluvia" --dialectical
              python theorist.py "misterio" --from ref.mid --persona ravel
              python theorist.py --interactive
              python theorist.py --load plan.theorist.json --refine "clímax más ambiguo"
        """),
    )

    # Entrada principal
    parser.add_argument("description", nargs="?", default=None,
                        help="Descripción libre de la intención musical")

    # Restricciones duras
    parser.add_argument("--key",   default=None,
                        help="Tonalidad forzada, ej. Dm, G#, Bb (default: auto)")
    parser.add_argument("--bars",  type=int, default=32,
                        help="Compases totales (default: 32)")
    parser.add_argument("--tempo", type=int, default=None,
                        help="Tempo BPM (default: auto)")
    parser.add_argument("--arc",   default=None, choices=VALID_ARCS,
                        help="Arco narrativo base (default: auto)")
    parser.add_argument("--persona", default="auto",
                        choices=list(PERSONAS.keys()),
                        help="Persona teórica (default: auto)")

    # Modos especiales
    parser.add_argument("--from", dest="ref_midi", default=None, metavar="MIDI",
                        help="MIDI de referencia para anclar la interpretación")
    parser.add_argument("--dialectical", action="store_true",
                        help="Generar 3 lecturas alternativas de la misma intención")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Modo conversacional con compromisos teóricos por turno")
    parser.add_argument("--load", default=None, metavar="FILE",
                        help="Cargar un plan .theorist.json existente")
    parser.add_argument("--refine", default=None, metavar="TEXT",
                        help="Refinar el plan cargado con una nueva instrucción")

    # Ejecución
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo mostrar el razonamiento, no generar archivos")
    parser.add_argument("--execute", action="store_true",
                        help="Ejecutar el pipeline completo tras generar el plan")
    parser.add_argument("--until",   default=None, metavar="SCRIPT",
                        help="Ejecutar hasta este script del pipeline (sin .py)")

    # Salida
    parser.add_argument("--output", default="obra",
                        help="Nombre base para archivos de salida (default: obra)")

    # LLM
    parser.add_argument("--no-llm",       action="store_true",
                        help="Usar solo motor local, sin LLM")
    parser.add_argument("--llm-provider", default="anthropic",
                        choices=list(LLM_BACKENDS.keys()),
                        help="Proveedor LLM: anthropic (default) | openai")
    parser.add_argument("--llm-model",    default=None,
                        help="Modelo específico. "
                             "Anthropic: claude-sonnet-4-6, claude-opus-4-6, claude-haiku-4-5-20251001 | "
                             "OpenAI: gpt-4o, gpt-4o-mini")
    parser.add_argument("--api-key",      default=None,
                        help="API key (o ANTHROPIC_API_KEY / OPENAI_API_KEY)")
    parser.add_argument("--llm-debug",    action="store_true",
                        help="Mostrar prompt enviado y respuesta raw del LLM")

    # Utilidades
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Razonamiento detallado durante el proceso")
    parser.add_argument("--list-personas", action="store_true",
                        help="Listar personas teóricas disponibles")

    args = parser.parse_args()

    # ── Listar personas ───────────────────────────────────────────────────────
    if args.list_personas:
        print_personas()
        return

    # ── Modo interactivo ──────────────────────────────────────────────────────
    if args.interactive:
        interactive_loop(args)
        return

    # ── Verificar entrada ─────────────────────────────────────────────────────
    if not args.description and not args.load:
        parser.print_help()
        print("\n  Ejemplos rápidos:")
        print("    python theorist.py \"melancolía urbana\"")
        print("    python theorist.py \"tormenta\" --persona romantic --execute")
        print("    python theorist.py --interactive")
        sys.exit(0)

    # ── Cargar plan existente ─────────────────────────────────────────────────
    existing_plan = None
    if args.load:
        if not Path(args.load).exists():
            print(f"ERROR: no existe el archivo: {args.load}")
            sys.exit(1)
        with open(args.load, "r", encoding="utf-8") as f:
            existing_plan = json.load(f)
        print(f"  Plan cargado: {args.load}")

        if args.refine:
            # Refinar: la descripción es la refinación + contexto del plan cargado
            original_desc = existing_plan.get("description_original", "")
            description = f"{original_desc} | refinamiento: {args.refine}"
            print(f"  Refinando: \"{args.refine}\"")
        else:
            print_reasoning(existing_plan, verbose=args.verbose)
            return
    else:
        description = args.description

    # ── Leer MIDI de referencia ───────────────────────────────────────────────
    ref_state = None
    if args.ref_midi:
        if not Path(args.ref_midi).exists():
            print(f"ERROR: --from archivo no existe: {args.ref_midi}")
            sys.exit(1)
        print(f"  Referencia MIDI: {args.ref_midi}")
        ref_state = read_ref_state(args.ref_midi, verbose=args.verbose)

    api_key = args.api_key or os.environ.get(
        "ANTHROPIC_API_KEY" if args.llm_provider == "anthropic" else "OPENAI_API_KEY"
    )

    # ── Modo dialectico ───────────────────────────────────────────────────────
    if args.dialectical:
        if args.no_llm:
            print("  [warn] --dialectical requiere LLM. Usa --api-key o exporta la variable de entorno.")
            sys.exit(1)
        print(f"  Generando 3 lecturas para: \"{description}\"")
        readings = generate_dialectical(
            description, args.bars,
            args.llm_provider, api_key, args.llm_model,
            args.verbose, args.llm_debug,
        )
        if not readings:
            print("  [error] El LLM no devolvió lecturas. Comprueba la API key.")
            sys.exit(1)
        print_dialectical(readings)

        if not args.dry_run:
            # Exportar las 3 lecturas como planes individuales
            for i, r in enumerate(readings, 1):
                base = f"{args.output}_lectura{i}_{r.get('label', str(i)).replace(' ', '_').lower()}"
                # Construir un plan mínimo por cada lectura para exportar
                mini_plan = local_parse(
                    description, args.bars,
                    forced_key=r.get("key"),
                    forced_tempo=r.get("tempo"),
                )
                mini_plan["theoretical_frame"]["school"] = r.get("school", "")
                mini_plan["theoretical_frame"]["governing_principle"] = r.get("governing_principle", "")
                export_theorist_json(mini_plan, description, base, vars(args))
                export_curves_json(mini_plan, base)
                print(f"    ✓ {base}.theorist.json")
        return

    # ── Plan principal ────────────────────────────────────────────────────────
    if args.verbose:
        print(f"\n  Procesando: \"{description}\"")

    print(f"\n  [1/5] Interpretando descripción...")
    print(f"  [2/5] Consultando marco teórico...")
    print(f"  [3/5] Decidiendo parámetros musicales...")

    plan = generate_plan(
        description,
        n_bars=args.bars,
        forced_key=args.key,
        forced_tempo=args.tempo,
        ref_state=ref_state,
        persona=args.persona,
        arc=args.arc,
        use_llm=not args.no_llm,
        provider=args.llm_provider,
        api_key=api_key,
        model=args.llm_model,
        verbose=args.verbose,
        debug=args.llm_debug,
    )

    if not plan:
        print("  [error] No se pudo generar el plan.")
        sys.exit(1)

    print(f"  [4/5] Generando curvas emocionales...")
    print(f"  [5/5] Exportando archivos de pipeline...")

    _run_export_and_execute(plan, description, args, vars(args))


if __name__ == "__main__":
    main()
