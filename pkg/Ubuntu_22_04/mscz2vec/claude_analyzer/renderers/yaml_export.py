"""
renderers/yaml_export.py
════════════════════════
Renderer de exportación estructurada (salida .yaml).

Serializa todos los resultados del análisis en un schema YAML
tipado, documentado y deserializable sin pérdida por otra herramienta.

Convenciones del schema:
  - Curvas temporales: lista de [tiempo_seg, valor_float]
  - Conteos: dict {tipo: int}
  - Enums: strings en minúsculas
  - Opcionales ausentes: null explícito (no omitidos)
  - Números: redondeados a 4 decimales para legibilidad
  - Timestamps: floats en segundos (no strings)

Uso:
    from midi_analyzer.renderers.yaml_export import render_yaml
    yaml_str = render_yaml(results)
    with open("pieza.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_str)

    # Leer de vuelta:
    import yaml
    data = yaml.safe_load(open("pieza.yaml"))
"""

from __future__ import annotations
import math
from typing import Any


# ── YAML serializer (sin dependencia externa) ────────────────────

def _r(val: float, digits: int = 4) -> float:
    """Round float for clean YAML output."""
    if val is None:
        return None
    try:
        return round(float(val), digits)
    except (TypeError, ValueError):
        return val


def _curve(data, key_t='time', key_v=None, max_points: int = 500) -> list:
    """Serialize a curve to [[t, v], ...] with downsampling."""
    if not data:
        return []
    kv = key_v or 'tension'
    first = data[0]
    if isinstance(first, (list, tuple)):
        pts = [[_r(p[0], 3), _r(p[1], 4)] for p in data]
    elif isinstance(first, dict):
        pts = [[_r(p.get(key_t, 0), 3), _r(p.get(kv, 0), 4)] for p in data]
    elif hasattr(first, 'time') and hasattr(first, 'tension'):
        pts = [[_r(p.time, 3), _r(p.tension, 4)] for p in data]
    elif hasattr(first, key_t):
        pts = [[_r(getattr(p, key_t, 0), 3), _r(getattr(p, kv, 0), 4)] for p in data]
    else:
        pts = [[i, _r(float(v), 4)] for i, v in enumerate(data)]
    # Downsample
    if len(pts) > max_points:
        step = len(pts) // max_points
        pts = pts[::step]
    return pts


def _counter(c) -> dict:
    """Serialize Counter or dict to plain dict."""
    if c is None:
        return {}
    if hasattr(c, 'most_common'):
        return {str(k): int(v) for k, v in c.items()}
    if isinstance(c, dict):
        return {str(k): int(v) for k, v in c.items()}
    return {}


def _safe(val, default=None):
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return val


def _yaml_str(key: str, val: Any, indent: int = 0) -> str:
    """Minimal YAML serializer — handles nested dicts/lists."""
    pad = '  ' * indent
    if val is None:
        return f"{pad}{key}: null\n"
    if isinstance(val, bool):
        return f"{pad}{key}: {'true' if val else 'false'}\n"
    if isinstance(val, int):
        return f"{pad}{key}: {val}\n"
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return f"{pad}{key}: null\n"
        return f"{pad}{key}: {val:.4f}\n"
    if isinstance(val, str):
        # Quote if needed
        if any(c in val for c in [':', '#', '{', '}', '[', ']', ',', '&', '*', '?', '|', '-', '<', '>', '=', '!', '"', "'"]):
            escaped = val.replace("'", "''")
            return f"{pad}{key}: '{escaped}'\n"
        return f"{pad}{key}: {val}\n"
    if isinstance(val, list):
        if not val:
            return f"{pad}{key}: []\n"
        # Check if it's a list of simple scalars or pairs
        if all(isinstance(v, (int, float)) for v in val):
            items = ', '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in val[:20])
            if len(val) > 20:
                items += f', ...({len(val)} items)'
            return f"{pad}{key}: [{items}]\n"
        if all(isinstance(v, list) and len(v) == 2 for v in val[:5]):
            # Curve data — inline pairs
            pairs = ', '.join(f'[{p[0]:.3f}, {p[1]:.4f}]' for p in val[:5])
            if len(val) > 5:
                pairs += f', ...({len(val)} points)'
            return f"{pad}{key}: [{pairs}]\n"
        # List of dicts or mixed
        lines = [f"{pad}{key}:\n"]
        for item in val[:10]:
            if isinstance(item, dict):
                lines.append(f"{pad}  -\n")
                for k2, v2 in item.items():
                    lines.append(_yaml_str(str(k2), v2, indent + 2))
            elif isinstance(item, (list, tuple)):
                inner = ', '.join(str(x) for x in item)
                lines.append(f"{pad}  - [{inner}]\n")
            else:
                lines.append(f"{pad}  - {item}\n")
        if len(val) > 10:
            lines.append(f"{pad}  # ... {len(val)-10} more items\n")
        return ''.join(lines)
    if isinstance(val, dict):
        if not val:
            return f"{pad}{key}: {{}}\n"
        lines = [f"{pad}{key}:\n"]
        for k2, v2 in val.items():
            lines.append(_yaml_str(str(k2), v2, indent + 1))
        return ''.join(lines)
    return f"{pad}{key}: {str(val)!r}\n"


# ── Schema sections ──────────────────────────────────────────────

def _build_identity(r: dict) -> dict:
    return {
        'filepath':    r.get('filepath', ''),
        'title':       r.get('title', ''),
        'key_root':    int(_safe(r.get('key_root'), 0)),
        'key_name':    r.get('key_name', ''),
        'mode':        r.get('mode', ''),
        'ks_confidence': _r(_safe(r.get('kconf'), 0)),
        'avg_bpm':     _r(_safe(r.get('avg_bpm'), 120)),
        'main_bpm':    _r(_safe(r.get('main_bpm'), 120)),
        'time_signature': r.get('ts_str_val', '4/4'),
        'ts_num':      int(_safe(r.get('ts_num'), 4)),
        'total_dur_sec': _r(_safe(r.get('total_dur'), 0)),
        'notes_count': int(_safe(r.get('notes_count'), 0)),
        'n_sections':  int(_safe(r.get('n_sections'), 6)),
    }


def _build_genre(r: dict) -> dict:
    gd = r.get('genre_detection') or {}
    se = r.get('semantic_enrichment') or {}
    return {
        'genre': gd.get('genre', 'indeterminado'),
        'genre_label': gd.get('genre_label', ''),
        'confidence': _r(_safe(gd.get('confidence'), 0)),
        'subgenre': gd.get('subgenre'),
        'ambiguous': bool(gd.get('ambiguous', False)),
        'key_signature': gd.get('key_signature', ''),
        'regions': gd.get('regions', ''),
        'top3': [
            {'genre': g['genre'], 'label': g['label'], 'score': _r(g['score'])}
            for g in (gd.get('top3') or [])
        ],
        'affekt_concept': se.get('concept'),
        'affekt_fit': _r(_safe(se.get('fit_score'), 0)),
        'affekt_arc': se.get('arc'),
        'persona': se.get('persona'),
    }


def _build_harmony(r: dict) -> dict:
    cad = r.get('cadences') or {}
    hg  = r.get('harmonic_graph') or {}
    tg  = r.get('tonal_gravity') or {}
    ta  = r.get('tonal_ambiguity') or {}
    ac  = r.get('anti_conventional') or {}
    return {
        'cadences': {
            'total': int(_safe(cad.get('total'), 0)),
            'dominant_type': cad.get('dominant_type', ''),
            'counts': _counter(cad.get('counts')),
        },
        'harmonic_graph': {
            'entropy': _r(_safe(hg.get('entropy'), 0)),
            'center': hg.get('center', ''),
        },
        'tonal_gravity': {
            'n_modulations': int(_safe(tg.get('n_modulations'), 0)),
            'dominant_key':  tg.get('dominant_key', ''),
            'total_distance': _r(_safe(tg.get('total_distance'), 0)),
        },
        'tonal_ambiguity': {
            'ambiguity_index': _r(_safe(ta.get('ambiguity_index'), 0)),
            'whole_tone_coverage': _r(_safe(ta.get('whole_tone_coverage'), 0)),
            'octatonic_coverage': _r(_safe(ta.get('octatonic_coverage'), 0)),
        },
        'canon_progressions': list(r.get('canon_progs') or [])[:8],
        'modal_borrowings': list(r.get('modal_borrow') or [])[:6],
        'voice_leading_smoothness': _r(_safe((r.get('voice_leading') or {}).get('smoothness'), 0)),
        'anti_conventional_score': _r(_safe(ac.get('deviation_score'), 0)),
        'signature_choices': list(ac.get('signature_choices') or [])[:5],
    }


def _build_melody(r: dict) -> dict:
    ia  = r.get('interval_anal') or {}
    mm  = r.get('melodic_markov') or {}
    exp = r.get('expectation') or {}
    return {
        'contour': r.get('contour', ''),
        'intervals': {
            'conjunct_ratio': _r(_safe(ia.get('conjunct'), 0)),
            'leaps_ratio': _r(_safe(ia.get('leaps'), 0)),
            'ascending_ratio': _r(_safe(ia.get('ascending'), 0)),
            'avg_interval_st': _r(_safe(ia.get('avg_interval'), 0)),
            'tritone_ratio': _r(_safe(ia.get('tritones'), 0)),
        },
        'motifs_count': len(r.get('motifs') or []),
        'expectation': {
            'mean_surprise': _r(_safe(exp.get('mean_surprise'), 0)),
            'surprise_peaks_count': len(exp.get('surprise_peaks') or []),
        },
        'markov': {
            'style_profile': mm.get('style_profile', ''),
            'mean_probability': _r(_safe(mm.get('mean_probability'), 0)),
            'entropy_bits': _r(_safe(mm.get('entropy_markov'), 0)),
            'n_unique_states': int(_safe(mm.get('n_unique_states'), 0)),
            'n_transitions': int(_safe(mm.get('n_transitions'), 0)),
            'hapax_count': len(mm.get('singular_gestures') or []),
        },
    }


def _build_rhythm(r: dict) -> dict:
    rh = r.get('rhythm') or {}
    mt = r.get('micro_timing') or {}
    gr = r.get('groove') or {}
    return {
        'syncopation': _r(_safe(rh.get('syncopation'), 0)),
        'variety': _r(_safe(rh.get('variety'), 0)),
        'swing_ratio': _r(_safe(rh.get('swing_ratio'), 0)),
        'groove_strength': _r(_safe(gr.get('groove_strength'), 0)),
        'micro_timing': {
            'humanization': _r(_safe(mt.get('humanization'), 0)),
            'mean_deviation_ms': _r(_safe(mt.get('mean_deviation_ms'), 0)),
            'drag_ratio': _r(_safe(mt.get('drag_ratio'), 0)),
            'anticipation_ratio': _r(_safe(mt.get('anticipation_ratio'), 0)),
            'style': mt.get('style', ''),
        },
    }


def _build_emotional_curves(r: dict) -> dict:
    tc  = r.get('tension_curve') or []
    dv  = r.get('dynamic_valence') or {}
    ep  = r.get('energy_profile') or {}
    rg  = r.get('roughness') or {}
    uem = r.get('unified_emotional_map') or {}
    pc  = r.get('perceptual') or {}

    return {
        'tension': {
            'curve': _curve(tc, key_v='tension'),
            'overall': _r(_safe(r.get('overall_tension'), 0)),
        },
        'valence': {
            'curve': _curve(dv.get('curve') or []),
            'mean': _r(_safe(dv.get('mean_valence'), 0)),
            'volatility': _r(_safe(dv.get('volatility'), 0)),
            'arc_shape': dv.get('arc_shape', ''),
            'peaks_light': [
                {'time': _r(p['time'], 3), 'value': _r(p['valence'], 4)}
                for p in (dv.get('peaks_light') or [])[:4]
            ],
            'peaks_dark': [
                {'time': _r(p['time'], 3), 'value': _r(p['valence'], 4)}
                for p in (dv.get('peaks_dark') or [])[:4]
            ],
        },
        'energy': {
            'curve': _curve(ep.get('curve') or []),
            'mean': _r(_safe(ep.get('mean'), 0)),
            'peak_time': _r(_safe(ep.get('peak_time'), 0), 3),
        },
        'roughness': {
            'curve_norm': _curve(rg.get('curve_norm') or []),
            'mean': _r(_safe(rg.get('mean_roughness'), 0)),
            'max': _r(_safe(rg.get('max_roughness'), 0)),
            'arc': rg.get('roughness_arc', ''),
            'register_impact': _r(_safe(rg.get('register_impact'), 0)),
            'peak_time': _r(_safe((rg.get('peak_roughness') or {}).get('time'), 0), 3),
        },
        'perceptual_intensity': {
            'mean': _r(_safe(pc.get('mean'), 0)),
            'peak_time': _r(_safe(pc.get('peak_time'), 0), 3),
        },
        'unified_map': {
            'mean_coherence': _r(_safe(uem.get('mean_coherence'), 0)),
            'n_phases': len(uem.get('emotional_phases') or []),
            'n_inflections': len(uem.get('inflection_points') or []),
            'peak_complexity_time': _r(_safe(uem.get('peak_complexity'), 0), 3),
            'phases': [
                {
                    'label': ph['label'],
                    'start': _r(ph['start'], 3),
                    'end':   _r(ph['end'], 3),
                    'duration': _r(ph['duration'], 3),
                    'tension_mean':  _r((ph.get('means') or {}).get('tension'), 0),
                    'valence_mean':  _r((ph.get('means') or {}).get('valence_n'), 0),
                    'energy_mean':   _r((ph.get('means') or {}).get('energy'), 0),
                    'arousal_mean':  _r((ph.get('means') or {}).get('arousal'), 0),
                }
                for ph in (uem.get('emotional_phases') or [])
            ],
            'inflections': [
                {
                    'time': _r(ip['time'], 3),
                    'delta': _r(ip['delta'], 4),
                    'from_label': ip.get('from_label', ''),
                    'to_label': ip.get('to_label', ''),
                }
                for ip in (uem.get('inflection_points') or [])[:8]
            ],
        },
    }


def _build_narrative(r: dict) -> dict:
    ni  = r.get('narrative_intention') or {}
    cat = r.get('catharsis') or {}
    traj= r.get('emotional_trajectory') or {}
    pol = r.get('polarity') or {}
    amb = r.get('emotional_ambivalence') or {}
    co  = r.get('coherence') or {}
    mld = r.get('multilevel_density') or {}

    return {
        'narrative_intention': {
            'archetype': ni.get('archetype', ''),
            'confidence': _r(_safe(ni.get('confidence'), 0)),
            'alternative': ni.get('alternative'),
            'time_relationship': ni.get('time_relationship', ''),
            'expectation_strategy': ni.get('expectation_strategy', ''),
        },
        'catharsis': {
            'present': bool(cat.get('present', False)),
            'type': cat.get('catharsis_type', 'ausente'),
            'moment_sec': _r(_safe(cat.get('moment')), 3),
            'buildup_start_sec': _r(_safe(cat.get('buildup_start')), 3),
            'buildup_duration_sec': _r(_safe(cat.get('buildup_duration'), 0), 3),
            'release_strength': _r(_safe(cat.get('release_strength'), 0)),
            'resolution_quality': _r(_safe(cat.get('resolution_quality'), 0)),
            'n_waves': len(cat.get('waves') or []),
        },
        'emotional_trajectory': {
            'path_shape': traj.get('path_shape', ''),
            'total_distance': _r(_safe(traj.get('total_distance'), 0)),
            'net_displacement': _r(_safe(traj.get('net_displacement'), 0)),
            'efficiency': _r(_safe(traj.get('efficiency'), 0)),
            'centroid_valence': _r(_safe((traj.get('centroid') or (0, 0))[0])),
            'centroid_arousal': _r(_safe((traj.get('centroid') or (0, 0))[1])),
            'crossings': int(_safe(traj.get('crossings'), 0)),
            'dominant_quadrant': traj.get('dominant_quadrant', ''),
        },
        'polarity_type': pol.get('polarity_type', ''),
        'emotional_ambivalence': {
            'mean': _r(_safe(amb.get('mean_ambivalence'), 0)),
            'profile': amb.get('profile', ''),
            'n_zones': len(amb.get('zones') or []),
        },
        'coherence': {
            'index': _r(_safe(co.get('coherence'), 1)),
            'n_contradictions': int(_safe(co.get('contradiction_count'), 0)),
        },
        'multilevel_density': {
            'micro_entropy': _r(_safe(mld.get('micro_entropy'), 0)),
            'meso_entropy':  _r(_safe(mld.get('meso_entropy'), 0)),
            'macro_entropy': _r(_safe(mld.get('macro_entropy'), 0)),
            'composer_economy': _r(_safe(mld.get('composer_economy'), 0)),
            'profile': mld.get('density_profile', ''),
        },
    }


def _build_structure(r: dict) -> dict:
    ssm = r.get('ssm') or {}
    mf  = r.get('musical_form') or {}
    na  = r.get('narrative_arc') or {}
    pnr = r.get('pnr') or {}
    gld = r.get('golden') or {}
    sa  = r.get('silence_analysis') or {}

    return {
        'musical_form': {
            'form': mf.get('form', ''),
            'pattern': mf.get('pattern', ''),
        },
        'ssm': {
            'form_str': ssm.get('form_str', ''),
            'form_canon': ssm.get('form_canon', ''),
            'n_unique': int(_safe(ssm.get('n_unique'), 0)),
            'n_repetitions': len(ssm.get('repetitions') or []),
            'symmetry': _r(_safe(ssm.get('symmetry'), 0)),
            'block_structure': [
                {
                    'label':    b.get('label', '?'),
                    'start':    _r(b.get('start', 0), 3),
                    'end':      _r(b.get('end', 0), 3),
                    'duration': _r(b.get('duration', b.get('end', 0) - b.get('start', 0)), 3),
                }
                for b in (ssm.get('block_structure') or [])
            ],
        },
        'narrative_arc': {
            'arc_type': na.get('arc_type', ''),
            'climax_time': _r(_safe(na.get('climax_time')), 3),
            'climax_position': _r(_safe(na.get('climax_position'), 0)),
        },
        'point_of_no_return': {
            'time': _r(_safe(pnr.get('time')), 3),
            'position': _r(_safe(pnr.get('position')), 3),
        },
        'golden_ratio': {
            'golden_climax': bool(gld.get('golden_climax', False)),
        },
        'silence': {
            'ratio': _r(_safe(sa.get('ratio'), 0)),
            'n_silences': int(_safe(sa.get('n_silences'), 0)),
        },
    }


# ── YAML document builder ────────────────────────────────────────

def render_yaml(results: dict) -> str:
    """
    Genera la exportación YAML estructurada a partir del dict de resultados.

    Args:
        results: dict devuelto por run_analysis()

    Returns:
        str: documento YAML completo, listo para guardar como .yaml
    """
    r = results

    # Build all sections
    identity  = _build_identity(r)
    genre     = _build_genre(r)
    harmony   = _build_harmony(r)
    melody    = _build_melody(r)
    rhythm    = _build_rhythm(r)
    curves    = _build_emotional_curves(r)
    narrative = _build_narrative(r)
    structure = _build_structure(r)

    # Assemble YAML manually for full control over formatting
    lines = [
        "# ════════════════════════════════════════════════════════",
        "# midi_analyzer v12.0 — exportación estructurada YAML",
        f"# generado desde: {r.get('filepath','')}",
        "# ════════════════════════════════════════════════════════",
        "",
        "---",
        "",
        "# ── IDENTIDAD ───────────────────────────────────────────",
    ]

    for k, v in identity.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "# ── GÉNERO Y ESTILO ─────────────────────────────────────"]
    for k, v in genre.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "# ── ARMONÍA ─────────────────────────────────────────────"]
    for k, v in harmony.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "# ── MELODÍA ─────────────────────────────────────────────"]
    for k, v in melody.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "# ── RITMO ───────────────────────────────────────────────"]
    for k, v in rhythm.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "# ── ESTRUCTURA FORMAL ───────────────────────────────────"]
    for k, v in structure.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "# ── ANÁLISIS NARRATIVO E INTENCIONAL ───────────────────"]
    for k, v in narrative.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "# ── CURVAS EMOCIONALES TEMPORALES ──────────────────────",
              "# Formato curvas: [[tiempo_seg, valor], ...]",
              "# Nota: downsampled a max 500 puntos por curva"]
    for k, v in curves.items():
        lines.append(_yaml_str(k, v).rstrip())

    lines += ["", "...", ""]
    return '\n'.join(lines)
