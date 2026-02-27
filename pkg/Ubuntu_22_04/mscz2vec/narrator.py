#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         NARRATOR  v1.0                                       ║
║         Diseñador de arquitectura narrativa para obras musicales             ║
║                                                                              ║
║  Define la estructura dramática completa de una obra ANTES de generar        ║
║  ningún MIDI. Actúa como "director de orquesta" del pipeline:                ║
║                                                                              ║
║  [1] NARRATIVA — Define la historia emocional en lenguaje natural o GUI      ║
║  [2] ESTRUCTURA — Genera forma musical (A/B/C), duraciones, cadencias        ║
║  [3] CURVAS — Exporta curvas para tension_designer.py                        ║
║  [4] PLAN DE PIPELINE — Genera el plan.yaml que runner.py ejecuta           ║
║  [5] TONALIDAD — Define el mapa tonal (qué tonalidades, cuándo, cómo)       ║
║                                                                              ║
║  ARCOS NARRATIVOS PREDEFINIDOS:                                              ║
║    hero       — Llamada → Prueba → Clímax → Resolución (arco de héroe)      ║
║    tragedy    — Ascenso → Clímax → Caída abrupta                            ║
║    romance    — Presentación → Tensión → Unión → Reafirmación               ║
║    mystery    — Ambigüedad → Revelaciones → Crisis → Resolución              ║
║    meditation — Plano → Ondulaciones sutiles → Retorno al reposo             ║
║    rondo      — Tema A reiterado con episodios contrastantes                 ║
║    sonata     — Exposición → Desarrollo → Reexposición                      ║
║    custom     — Diseño libre sección a sección                               ║
║                                                                              ║
║  USO:                                                                        ║
║    python narrator.py                        # GUI interactivo               ║
║    python narrator.py --arc hero --bars 64   # desde arco predefinido        ║
║    python narrator.py --arc sonata --bars 96 --key Dm --output plan.json    ║
║    python narrator.py --load plan.json --edit                                ║
║    python narrator.py --arc custom --sections "A:16,B:8,A:16,C:12,A:8"     ║
║    python narrator.py --arc hero --bars 48 --no-gui --export-curves         ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --arc NAME      Arco narrativo (default: hero)                            ║
║    --bars N        Compases totales de la obra (default: 64)                 ║
║    --key KEY       Tonalidad base, p.ej. Cm, G, Bb (default: C)             ║
║    --tempo N       Tempo base en BPM (default: 120)                          ║
║    --sections S    Override manual de secciones "A:16,B:8,C:24"             ║
║    --output FILE   Archivo de salida (default: obra_plan.json)               ║
║    --export-curves Exportar JSON de curvas para tension_designer             ║
║    --export-yaml   Exportar plan.yaml para runner.py                         ║
║    --no-gui        Modo headless: genera plan sin GUI                        ║
║    --load FILE     Cargar y editar un plan existente                         ║
║    --verbose       Informe detallado de decisiones                           ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    obra_plan.json         — Plan narrativo completo                          ║
║    obra_plan.curves.json  — Curvas para tension_designer (con --export-curves)║
║    obra_plan.yaml         — Pipeline para runner.py (con --export-yaml)      ║
║                                                                              ║
║  DEPENDENCIAS: matplotlib, numpy                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import copy
import textwrap
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.widgets import Button, Slider, RadioButtons, TextBox
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

SECTION_COLORS = {
    'A': '#3498db', 'B': '#e74c3c', 'C': '#2ecc71',
    'D': '#f39c12', 'E': '#9b59b6', 'T': '#1abc9c',  # T = transición
}

# Tonalidades y sus modulaciones típicas
TONALITY_GRAPH = {
    'C':  {'relative': 'Am', 'dominant': 'G',  'subdominant': 'F',  'parallel': 'Cm',  'mediant': 'Em'},
    'G':  {'relative': 'Em', 'dominant': 'D',  'subdominant': 'C',  'parallel': 'Gm',  'mediant': 'Bm'},
    'D':  {'relative': 'Bm', 'dominant': 'A',  'subdominant': 'G',  'parallel': 'Dm',  'mediant': 'F#m'},
    'A':  {'relative': 'F#m','dominant': 'E',  'subdominant': 'D',  'parallel': 'Am',  'mediant': 'C#m'},
    'E':  {'relative': 'C#m','dominant': 'B',  'subdominant': 'A',  'parallel': 'Em',  'mediant': 'G#m'},
    'F':  {'relative': 'Dm', 'dominant': 'C',  'subdominant': 'Bb', 'parallel': 'Fm',  'mediant': 'Am'},
    'Bb': {'relative': 'Gm', 'dominant': 'F',  'subdominant': 'Eb', 'parallel': 'Bbm', 'mediant': 'Dm'},
    'Eb': {'relative': 'Cm', 'dominant': 'Bb', 'subdominant': 'Ab', 'parallel': 'Ebm', 'mediant': 'Gm'},
    'Cm': {'relative': 'Eb', 'dominant': 'Gm', 'subdominant': 'Fm', 'parallel': 'C',   'mediant': 'Eb'},
    'Am': {'relative': 'C',  'dominant': 'Em', 'subdominant': 'Dm', 'parallel': 'A',   'mediant': 'C'},
    'Dm': {'relative': 'F',  'dominant': 'Am', 'subdominant': 'Gm', 'parallel': 'D',   'mediant': 'F'},
    'Gm': {'relative': 'Bb', 'dominant': 'Dm', 'subdominant': 'Cm', 'parallel': 'G',   'mediant': 'Bb'},
}

# ══════════════════════════════════════════════════════════════════════════════
#  ARCOS NARRATIVOS
# ══════════════════════════════════════════════════════════════════════════════

def _sigmoid(x): return 1 / (1 + np.exp(-x))

def build_arc_curves(arc_name: str, n_bars: int, key: str = 'C') -> dict:
    """
    Construye las 5 curvas emocionales para un arco narrativo dado.
    Retorna dict con tension, activity, register, harmony, swing (listas de n_bars valores).
    """
    t = np.linspace(0, 1, n_bars)

    arcs = {
        'hero': {
            'tension':  0.1 + 0.85 * np.concatenate([
                np.linspace(0.1, 0.3, n_bars//4),
                np.linspace(0.3, 0.95, n_bars//2 - n_bars//4),
                np.linspace(0.95, 0.15, n_bars - n_bars//2)
            ])[:n_bars],
            'activity': 0.2 + 0.7 * np.clip(
                np.concatenate([
                    np.linspace(0.2, 0.5, n_bars//4),
                    np.linspace(0.5, 1.0, n_bars//2 - n_bars//4),
                    np.linspace(1.0, 0.3, n_bars - n_bars//2)
                ])[:n_bars], 0, 1),
            'register': np.clip(np.concatenate([
                np.linspace(0.4, 0.5, n_bars//4),
                np.linspace(0.5, 0.9, n_bars//2),
                np.linspace(0.9, 0.4, n_bars - (n_bars//4 + n_bars//2))
            ])[:n_bars], 0, 1),
            'harmony':  np.clip(t**1.5 * 0.8 + 0.1, 0, 1),
            'swing':    np.zeros(n_bars) + 0.1,
        },
        'tragedy': {
            'tension':  np.clip(
                np.concatenate([
                    np.linspace(0.2, 0.95, int(n_bars*0.65)),
                    np.linspace(0.95, 0.05, n_bars - int(n_bars*0.65))
                ])[:n_bars], 0, 1),
            'activity': np.clip(
                np.concatenate([
                    np.linspace(0.3, 0.9, int(n_bars*0.65)),
                    np.linspace(0.9, 0.1, n_bars - int(n_bars*0.65))
                ])[:n_bars], 0, 1),
            'register': np.clip(
                np.concatenate([
                    np.linspace(0.5, 0.95, int(n_bars*0.65)),
                    np.linspace(0.95, 0.1, n_bars - int(n_bars*0.65))
                ])[:n_bars], 0, 1),
            'harmony':  np.clip(t * 0.9 + 0.1, 0, 1),
            'swing':    np.zeros(n_bars) + 0.05,
        },
        'romance': {
            'tension':  np.clip(
                0.3 + 0.4 * np.sin(np.pi * t) + 0.1 * np.sin(3 * np.pi * t),
                0.1, 0.9),
            'activity': np.clip(
                0.35 + 0.4 * np.sin(np.pi * t * 0.8), 0.2, 0.85),
            'register': np.clip(
                0.4 + 0.4 * np.sin(np.pi * t), 0.3, 0.9),
            'harmony':  np.clip(0.3 + 0.3 * t + 0.2 * np.sin(2*np.pi*t), 0.2, 0.8),
            'swing':    np.clip(0.1 + 0.3 * np.sin(np.pi * t), 0, 0.6),
        },
        'mystery': {
            'tension':  np.clip(
                0.5 + 0.4 * np.sin(2 * np.pi * t + 0.5) * t,
                0.2, 0.95),
            'activity': np.clip(
                0.2 + 0.5 * t + 0.15 * np.sin(4 * np.pi * t),
                0.1, 0.85),
            'register': np.clip(
                0.5 + 0.3 * np.sin(3 * np.pi * t) * (1 - t * 0.3),
                0.2, 0.85),
            'harmony':  np.clip(0.6 + 0.3 * t, 0.4, 1.0),
            'swing':    np.zeros(n_bars) + 0.05,
        },
        'meditation': {
            'tension':  np.clip(0.25 + 0.15 * np.sin(2 * np.pi * t * 2), 0.05, 0.5),
            'activity': np.clip(0.2 + 0.1 * np.sin(np.pi * t), 0.1, 0.4),
            'register': np.clip(0.4 + 0.1 * np.sin(np.pi * t * 1.5), 0.3, 0.6),
            'harmony':  np.clip(0.2 + 0.15 * np.sin(np.pi * t), 0.1, 0.45),
            'swing':    np.clip(0.2 + 0.1 * np.sin(np.pi * t), 0.1, 0.4),
        },
        'rondo': {
            'tension': np.array([
                0.4 + 0.05 * np.sin(10 * np.pi * i / (n_bars-1))
                + 0.35 * (1 - abs(2 * (i/(n_bars-1) % (1/5)) - 1/5))
                for i in range(n_bars)
            ]).clip(0.1, 0.9),
            'activity': np.array([
                0.4 + 0.25 * np.sin(4 * np.pi * i / (n_bars-1))
                for i in range(n_bars)
            ]).clip(0.2, 0.85),
            'register': np.array([
                0.45 + 0.2 * np.cos(4 * np.pi * i / (n_bars-1))
                for i in range(n_bars)
            ]).clip(0.2, 0.8),
            'harmony': np.clip(0.3 + 0.2 * np.sin(3 * np.pi * t), 0.2, 0.7),
            'swing':   np.clip(0.15 + 0.1 * np.sin(2 * np.pi * t), 0.05, 0.4),
        },
        'sonata': {
            'tension': np.array([
                # Exposición: suave/media — Desarrollo: alta — Reexposición: media
                0.3 + 0.2 * (i/(n_bars-1))             if i < n_bars//3 else
                0.5 + 0.4 * np.sin(np.pi * (i - n_bars//3) / (n_bars//3))
                                                         if i < 2*n_bars//3 else
                0.5 - 0.2 * ((i - 2*n_bars//3) / (n_bars//3))
                for i in range(n_bars)
            ]).clip(0.1, 0.95),
            'activity': np.array([
                0.3 + 0.1 * (i/(n_bars-1))  if i < n_bars//3 else
                0.4 + 0.5 * (i - n_bars//3) / (n_bars//3) if i < 2*n_bars//3 else
                0.7 - 0.1 * ((i - 2*n_bars//3) / (n_bars//3))
                for i in range(n_bars)
            ]).clip(0.2, 1.0),
            'register': np.array([
                0.45 + 0.1 * (i/(n_bars//3))   if i < n_bars//3 else
                0.55 + 0.35 * np.sin(np.pi * (i - n_bars//3) / (n_bars//3))
                                                 if i < 2*n_bars//3 else
                0.55 + 0.1 * (1 - (i - 2*n_bars//3)/(n_bars//3))
                for i in range(n_bars)
            ]).clip(0.3, 1.0),
            'harmony':  np.array([
                0.3 if i < n_bars//3 else
                0.3 + 0.6 * ((i - n_bars//3) / (n_bars//3)) if i < 2*n_bars//3 else
                0.4
                for i in range(n_bars)
            ]).clip(0.2, 1.0),
            'swing':   np.zeros(n_bars) + 0.08,
        },
    }

    if arc_name not in arcs:
        arc_name = 'hero'

    curves = arcs[arc_name]
    return {k: v.tolist() for k, v in curves.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADOR DE SECCIONES
# ══════════════════════════════════════════════════════════════════════════════

ARC_FORMS = {
    'hero':       [('A', 'Llamada',     0.20), ('B', 'Prueba',      0.30),
                   ('C', 'Clímax',      0.25), ('A', 'Resolución',  0.25)],
    'tragedy':    [('A', 'Ascenso',     0.35), ('B', 'Punto alto',  0.30),
                   ('C', 'Caída',       0.35)],
    'romance':    [('A', 'Encuentro',   0.25), ('B', 'Tensión',     0.30),
                   ('C', 'Unión',       0.25), ('A', 'Reafirmación',0.20)],
    'mystery':    [('A', 'Ambigüedad',  0.20), ('B', 'Pistas',      0.25),
                   ('C', 'Crisis',      0.30), ('A', 'Resolución',  0.25)],
    'meditation': [('A', 'Reposo',      0.20), ('B', 'Ondulación',  0.40),
                   ('A', 'Retorno',     0.40)],
    'rondo':      [('A', 'Tema',        0.15), ('B', 'Episodio 1',  0.20),
                   ('A', 'Tema',        0.10), ('C', 'Episodio 2',  0.20),
                   ('A', 'Tema',        0.10), ('D', 'Episodio 3',  0.15),
                   ('A', 'Coda',        0.10)],
    'sonata':     [('A', 'Exposición',  0.33), ('B', 'Desarrollo',  0.34),
                   ('A', 'Reexposición',0.33)],
    'custom':     [('A', 'Sección A',   1.00)],
}

ARC_TONAL_MAPS = {
    'hero':       ['base', 'parallel', 'dominant', 'base'],
    'tragedy':    ['base', 'relative', 'parallel'],
    'romance':    ['base', 'relative', 'dominant', 'base'],
    'mystery':    ['base', 'relative', 'parallel', 'base'],
    'meditation': ['base', 'relative', 'base'],
    'rondo':      ['base', 'dominant', 'base', 'relative', 'base', 'subdominant', 'base'],
    'sonata':     ['base', 'dominant', 'base'],
    'custom':     ['base'],
}


def build_sections(arc_name: str, n_bars: int, key: str = 'C',
                   sections_override: str = None) -> list:
    """
    Construye la lista de secciones de la obra.
    Cada sección: {'id': str, 'label': str, 'bars': int, 'start_bar': int,
                   'key': str, 'tonal_function': str, 'role': str}
    """
    if sections_override:
        # Parsear "A:16,B:8,A:16,C:12"
        raw = [s.strip().split(':') for s in sections_override.split(',')]
        template = [(r[0], r[0], int(r[1])) for r in raw if len(r) == 2]
    else:
        form = ARC_FORMS.get(arc_name, ARC_FORMS['hero'])
        template = [(sec, label, max(4, int(n_bars * frac)))
                    for sec, label, frac in form]
        # Ajustar para que sumen n_bars exactamente
        total = sum(b for _, _, b in template)
        if total != n_bars:
            diff = n_bars - total
            # Distribuir diferencia en la sección más larga
            max_idx = max(range(len(template)), key=lambda i: template[i][2])
            template[max_idx] = (template[max_idx][0], template[max_idx][1],
                                  template[max_idx][2] + diff)

    tonal_map = ARC_TONAL_MAPS.get(arc_name, ['base'] * len(template))
    tonal_relations = TONALITY_GRAPH.get(key, TONALITY_GRAPH.get('C'))

    sections = []
    start = 0
    for i, (sec_id, label, bars) in enumerate(template):
        tonal_fn = tonal_map[i % len(tonal_map)]
        if tonal_fn == 'base':
            sec_key = key
        else:
            sec_key = tonal_relations.get(tonal_fn, key) if tonal_relations else key

        sections.append({
            'id':           sec_id,
            'label':        label,
            'bars':         bars,
            'start_bar':    start,
            'key':          sec_key,
            'tonal_function': tonal_fn,
            'role':         label.lower(),
        })
        start += bars

    return sections


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

def build_plan(arc_name: str, n_bars: int, key: str = 'C', tempo: int = 120,
               sections_override: str = None, verbose: bool = False) -> dict:
    """Construye el plan narrativo completo."""
    sections  = build_sections(arc_name, n_bars, key, sections_override)
    curves    = build_arc_curves(arc_name, n_bars, key)

    # Calcular climax bar
    tension_arr = np.array(curves['tension'])
    climax_bar  = int(np.argmax(tension_arr))

    # Forma como string (p.ej. "ABCA")
    form_string = ''.join(s['id'] for s in sections)

    plan = {
        'arc':         arc_name,
        'n_bars':      n_bars,
        'base_key':    key,
        'tempo_bpm':   tempo,
        'form_string': form_string,
        'climax_bar':  climax_bar,
        'sections':    sections,
        'curves':      curves,
        'tonal_map': [s['key'] for s in sections],
        'pipeline_steps': _build_pipeline_steps(sections, curves, key, tempo, arc_name),
    }

    if verbose:
        _print_plan_summary(plan)

    return plan


def _curve_to_spec(values: list, bar_offset: int, n_points: int = 4) -> str:
    """
    Convierte un subarray de curva a string 'BAR:VAL, BAR:VAL, ...'
    muestreando n_points puntos representativos (inicio, puntos intermedios, fin).
    Los índices de compás son absolutos (bar_offset-based).
    """
    n = len(values)
    if n == 0:
        return None
    if n == 1:
        return f"{bar_offset}:{values[0]:.2f}"
    # Seleccionar índices uniformemente distribuidos
    indices = sorted(set(
        [0] +
        [int(i * (n - 1) / (n_points - 1)) for i in range(1, n_points - 1)] +
        [n - 1]
    ))
    parts = []
    for idx in indices:
        bar = bar_offset + idx
        val = round(float(values[idx]), 2)
        parts.append(f"{bar}:{val}")
    return ", ".join(parts)


def _build_pipeline_steps(sections: list, curves: dict, key: str,
                           tempo: int, arc_name: str) -> list:
    """
    Genera los pasos de pipeline sugeridos para este plan:
    qué herramientas llamar, en qué orden, con qué parámetros.
    Incluye curvas de mutación temporal (--mt-*) derivadas del arco narrativo.
    """
    steps = []
    for i, sec in enumerate(sections):
        start = sec['start_bar']
        n     = sec['bars']

        def _sec_curve(key_name):
            c = curves.get(key_name, [])
            if not c:
                return None
            return c[start:start + n]

        density_spec  = _curve_to_spec(_sec_curve('activity')  or [], start)
        register_spec = _curve_to_spec(_sec_curve('register')  or [], start)
        harmony_spec  = _curve_to_spec(_sec_curve('harmony')   or [], start)
        swing_spec    = _curve_to_spec(_sec_curve('swing')     or [], start)

        flags = {
            '--bars':              n,
            '--key':               sec['key'],
            '--tempo':             tempo,
            '--mode':              'auto',
            '--export-fingerprint': True,
            '--no-gui':            True,
        }
        if density_spec:
            flags['--mt-density'] = density_spec
        if register_spec:
            flags['--mt-register'] = register_spec
        if harmony_spec:
            flags['--mt-harmony-complexity'] = harmony_spec
        if swing_spec:
            flags['--mt-swing'] = swing_spec

        step = {
            'step':    i + 1,
            'tool':    'midi_dna_unified.py',
            'section': sec['id'],
            'label':   sec['label'],
            'bars':    n,
            'key':     sec['key'],
            'tempo':   tempo,
            'output':  f"seccion_{i+1:02d}_{sec['id']}_{sec['label'].replace(' ','_')}.mid",
            'flags':   flags,
        }
        steps.append(step)

    steps.append({
        'step':   len(sections) + 1,
        'tool':   'stitcher.py',
        'label':  'Ensamblaje final',
        'inputs': [s['output'] for s in steps if 'output' in s],
        'output': f'obra_{arc_name}_final.mid',
        'flags': {'--verbose': True},
    })

    steps.append({
        'step':   len(sections) + 2,
        'tool':   'orchestrator.py',
        'label':  'Orquestación',
        'input':  f'obra_{arc_name}_final.mid',
        'output': f'obra_{arc_name}_orquestada.mid',
        'flags': {'--template': 'chamber', '--auto-fp': True},
    })

    return steps


def _print_plan_summary(plan: dict):
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  NARRATOR — Plan Narrativo: {plan['arc'].upper():<34}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Forma:    {plan['form_string']:<51}║")
    print(f"║  Compases: {plan['n_bars']:<51}║")
    print(f"║  Tonalidad:{plan['base_key']:<51}║")
    print(f"║  Tempo:    {plan['tempo_bpm']} BPM{' '*47}║")
    print(f"║  Clímax en compás {plan['climax_bar']:<43}║")
    print("╠══════════════════════════════════════════════════════════════╣")
    for sec in plan['sections']:
        line = f"  {sec['id']}  {sec['label']:20s}  {sec['bars']:3d} compases  ({sec['key']})"
        print(f"║  {line:<60}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTADORES
# ══════════════════════════════════════════════════════════════════════════════

def export_plan_json(plan: dict, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(f"[narrator] Plan guardado en: {path}")


def export_curves_json(plan: dict, path: str):
    """Exporta curvas en formato tension_designer.py."""
    out = {
        'n_bars':   plan['n_bars'],
        'arc':      plan['arc'],
        'base_key': plan['base_key'],
        **plan['curves']
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[narrator] Curvas exportadas en: {path}")


def export_yaml(plan: dict, path: str):
    """Exporta plan.yaml para runner.py (formato legible)."""
    lines = [
        f"arc: {plan['arc']}",
        f"bars: {plan['n_bars']}",
        f"key: {plan['base_key']}",
        f"tempo: {plan['tempo_bpm']}",
        f"form: {plan['form_string']}",
        "",
        "sections:",
    ]
    for sec in plan['sections']:
        lines += [
            f"  - id: {sec['id']}",
            f"    label: {sec['label']}",
            f"    bars: {sec['bars']}",
            f"    key: {sec['key']}",
        ]
    lines += ["", "pipeline:"]
    for step in plan['pipeline_steps']:
        lines += [
            f"  - step: {step['step']}",
            f"    tool: {step['tool']}",
            f"    label: {step['label']}",
        ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"[narrator] YAML exportado en: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  GUI MATPLOTLIB
# ══════════════════════════════════════════════════════════════════════════════

class NarratorGUI:
    def __init__(self, plan: dict, output_base: str = 'obra_plan'):
        self.plan        = plan
        self.output_base = output_base
        self.modified    = False
        self._setup_figure()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#0d1117')
        self.fig.canvas.manager.set_window_title('NARRATOR v1.0 — Arquitectura Narrativa')

        # Área de curvas (arriba)
        self.ax_curves = self.fig.add_axes([0.05, 0.42, 0.60, 0.50])
        self.ax_curves.set_facecolor('#161b22')

        # Área de secciones (abajo izquierda)
        self.ax_form = self.fig.add_axes([0.05, 0.08, 0.60, 0.28])
        self.ax_form.set_facecolor('#161b22')

        # Panel de info (derecha)
        self.ax_info = self.fig.add_axes([0.70, 0.08, 0.28, 0.84])
        self.ax_info.set_facecolor('#161b22')
        self.ax_info.axis('off')

        self._draw_all()
        self._add_buttons()
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _draw_all(self):
        self._draw_curves()
        self._draw_form()
        self._draw_info()

    def _draw_curves(self):
        ax = self.ax_curves
        ax.clear()
        ax.set_facecolor('#161b22')
        curves = self.plan['curves']
        n = self.plan['n_bars']
        xs = np.arange(n)

        curve_meta = [
            ('tension',  '#e74c3c', 'Tensión'),
            ('activity', '#3498db', 'Actividad'),
            ('register', '#2ecc71', 'Registro'),
            ('harmony',  '#9b59b6', 'Armonía'),
            ('swing',    '#f39c12', 'Swing'),
        ]
        for cname, color, label in curve_meta:
            vals = curves.get(cname, [0.5]*n)
            ax.plot(xs, vals, color=color, linewidth=2, label=label, alpha=0.85)

        # Marcar inicio de secciones
        for sec in self.plan['sections']:
            sb = sec['start_bar']
            ax.axvline(sb, color='white', linestyle='--', alpha=0.25, linewidth=1)
            ax.text(sb + 0.3, 0.97, sec['id'], color='white', fontsize=8, alpha=0.7,
                    transform=ax.get_xaxis_transform())

        # Marcar clímax
        cb = self.plan['climax_bar']
        ax.axvline(cb, color='#ff6b6b', linestyle='-', alpha=0.6, linewidth=2)
        ax.text(cb + 0.3, 0.87, '▲clímax', color='#ff6b6b', fontsize=8,
                transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Compás', color='#8b949e', fontsize=9)
        ax.set_title(f"Curvas emocionales — Arco: {self.plan['arc'].upper()}",
                     color='white', fontsize=11, pad=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.spines[:].set_color('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5)
        ax.legend(loc='lower right', fontsize=8, facecolor='#21262d',
                  labelcolor='white', framealpha=0.8)

    def _draw_form(self):
        ax = self.ax_form
        ax.clear()
        ax.set_facecolor('#161b22')
        n = self.plan['n_bars']

        # Bloques de secciones
        for sec in self.plan['sections']:
            color = SECTION_COLORS.get(sec['id'], '#555')
            rect = mpatches.FancyBboxPatch(
                (sec['start_bar'], 0.1), sec['bars'] - 0.4, 0.8,
                boxstyle='round,pad=0.02', facecolor=color, alpha=0.75,
                edgecolor='white', linewidth=0.8)
            ax.add_patch(rect)
            cx = sec['start_bar'] + sec['bars'] / 2 - 0.2
            ax.text(cx, 0.5, f"{sec['id']}\n{sec['bars']}c", ha='center', va='center',
                    color='white', fontsize=9, fontweight='bold')
            ax.text(cx, 0.1, sec['key'], ha='center', va='bottom',
                    color='#f0f0f0', fontsize=7, alpha=0.8)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Compás', color='#8b949e', fontsize=9)
        ax.set_title('Forma musical — Mapa tonal', color='white', fontsize=11, pad=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.spines[:].set_color('#30363d')
        ax.set_yticks([])

    def _draw_info(self):
        ax = self.ax_info
        ax.clear()
        ax.set_facecolor('#161b22')
        ax.axis('off')
        p = self.plan

        info_lines = [
            ('NARRATOR v1.0', '#e6edf3', 13, 'bold'),
            ('', None, 8, 'normal'),
            (f"Arco:  {p['arc'].upper()}", '#58a6ff', 10, 'bold'),
            (f"Forma: {p['form_string']}", '#e6edf3', 10, 'normal'),
            (f"Bars:  {p['n_bars']}", '#e6edf3', 10, 'normal'),
            (f"Key:   {p['base_key']}", '#e6edf3', 10, 'normal'),
            (f"BPM:   {p['tempo_bpm']}", '#e6edf3', 10, 'normal'),
            (f"Clímax: compás {p['climax_bar']}", '#ff6b6b', 10, 'normal'),
            ('', None, 8, 'normal'),
            ('SECCIONES', '#58a6ff', 10, 'bold'),
        ]
        for sec in p['sections']:
            info_lines.append(
                (f"  {sec['id']} {sec['label']:18s} {sec['bars']:3d}c  ({sec['key']})",
                 '#8b949e', 9, 'normal')
            )

        info_lines += [
            ('', None, 8, 'normal'),
            ('CONTROLES', '#58a6ff', 10, 'bold'),
            ('  S — guardar plan JSON',   '#8b949e', 9, 'normal'),
            ('  C — exportar curvas',     '#8b949e', 9, 'normal'),
            ('  Y — exportar YAML',       '#8b949e', 9, 'normal'),
            ('  Q — salir',               '#8b949e', 9, 'normal'),
        ]

        y = 0.97
        for text, color, size, weight in info_lines:
            if color is None:
                y -= 0.02
                continue
            ax.text(0.04, y, text, color=color, fontsize=size,
                    fontweight=weight, transform=ax.transAxes, va='top',
                    fontfamily='monospace')
            y -= size * 0.016

    def _add_buttons(self):
        # Botones en la figura
        btn_defs = [
            ([0.05, 0.02, 0.08, 0.04], 'Guardar JSON', self._save_json),
            ([0.14, 0.02, 0.08, 0.04], 'Exportar Curvas', self._export_curves),
            ([0.23, 0.02, 0.08, 0.04], 'Exportar YAML', self._export_yaml),
            ([0.55, 0.02, 0.06, 0.04], 'Salir', self._quit),
        ]
        self._buttons = []
        for pos, label, callback in btn_defs:
            ax_btn = self.fig.add_axes(pos)
            btn = Button(ax_btn, label, color='#21262d', hovercolor='#30363d')
            btn.label.set_color('white')
            btn.label.set_fontsize(9)
            btn.on_clicked(callback)
            self._buttons.append(btn)

    def _on_key(self, event):
        if event.key in ('q', 'Q', 'escape'):
            self._quit(None)
        elif event.key in ('s', 'S'):
            self._save_json(None)
        elif event.key in ('c', 'C'):
            self._export_curves(None)
        elif event.key in ('y', 'Y'):
            self._export_yaml(None)

    def _save_json(self, _):
        path = self.output_base + '.json'
        export_plan_json(self.plan, path)

    def _export_curves(self, _):
        path = self.output_base + '.curves.json'
        export_curves_json(self.plan, path)

    def _export_yaml(self, _):
        path = self.output_base + '.yaml'
        export_yaml(self.plan, path)

    def _quit(self, _):
        plt.close('all')

    def run(self):
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE Y MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='NARRATOR v1.0 — Arquitectura narrativa para obras musicales',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arc',      default='hero',
                   choices=['hero','tragedy','romance','mystery','meditation',
                            'rondo','sonata','custom'],
                   help='Arco narrativo (default: hero)')
    p.add_argument('--bars',     type=int, default=64,
                   help='Compases totales (default: 64)')
    p.add_argument('--key',      default='C',
                   help='Tonalidad base (default: C)')
    p.add_argument('--tempo',    type=int, default=120,
                   help='Tempo en BPM (default: 120)')
    p.add_argument('--sections', default=None,
                   help='Override manual de secciones: "A:16,B:8,C:24"')
    p.add_argument('--output',   default='obra_plan',
                   help='Base del nombre de salida (default: obra_plan)')
    p.add_argument('--export-curves', action='store_true',
                   help='Exportar JSON de curvas para tension_designer')
    p.add_argument('--export-yaml',   action='store_true',
                   help='Exportar plan.yaml para runner.py')
    p.add_argument('--no-gui',   action='store_true',
                   help='Modo headless: sin GUI')
    p.add_argument('--load',     default=None,
                   help='Cargar plan existente desde JSON')
    p.add_argument('--verbose',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if args.load and os.path.exists(args.load):
        with open(args.load, encoding='utf-8') as f:
            plan = json.load(f)
        print(f"[narrator] Plan cargado desde {args.load}")
    else:
        plan = build_plan(
            arc_name  = args.arc,
            n_bars    = args.bars,
            key       = args.key,
            tempo     = args.tempo,
            sections_override = args.sections,
            verbose   = args.verbose,
        )

    out_base = args.output.rstrip('.json').rstrip('.yaml')

    # Siempre guardar JSON
    export_plan_json(plan, out_base + '.json')

    if args.export_curves:
        export_curves_json(plan, out_base + '.curves.json')

    if args.export_yaml:
        export_yaml(plan, out_base + '.yaml')

    if not args.no_gui:
        if not MATPLOTLIB_OK:
            print("[narrator] matplotlib no disponible. Usa --no-gui.")
            return
        gui = NarratorGUI(plan, out_base)
        gui.run()
    else:
        if args.verbose:
            _print_plan_summary(plan)


if __name__ == '__main__':
    main()
