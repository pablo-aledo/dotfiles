"""
renderers/essay.py
══════════════════
Renderer de ensayo narrativo (salida .txt).

Consume el dict devuelto por run_analysis() y produce
prosa continua estructurada en 10 secciones.

No contiene datos numéricos crudos — todo se transforma
en lenguaje natural. Depende exclusivamente de las
funciones write_* del módulo core.

Uso:
    from midi_analyzer.renderers.essay import render_essay
    text = render_essay(results)
    with open("pieza.txt", "w") as f:
        f.write(text)
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ── Importar funciones de escritura del core ────────────────────
from midi_analyzer.core import (
    write_executive_summary,
    write_identity,
    write_architecture,
    write_harmony,
    write_melody,
    write_rhythm,
    write_listener_experience,
    write_compositional_intention,
    write_emotional_evolution,
    write_emotional_story_section,
    write_synthesis,
    MODE_CHARS,
)


def render_essay(results: dict) -> str:
    """
    Genera el ensayo narrativo completo a partir del dict de resultados.

    Estructura de salida:
        RESUMEN EJECUTIVO (7 hallazgos clave)
        I.   Identidad
        II.  Arquitectura temporal
        III. Lenguaje armónico
        IV.  Discurso melódico
        V.   Dimensión rítmica
        VI.  Experiencia del oyente
        VII. Intención compositiva
        VIII.Evolución emocional a lo largo del tiempo
        IX.  Historia emocional y lectura narrativa
        X.   Síntesis

    Args:
        results: dict devuelto por run_analysis()

    Returns:
        str: texto completo del ensayo, listo para guardar como .txt
    """
    r = results
    kw = r  # alias — generate_essay usa kw.get(...)

    title = r.get('title', 'sin título').upper()
    SEP   = '═' * 66

    parts = []

    # ── Cabecera ────────────────────────────────────────────────
    parts += [SEP, f"  ANÁLISIS MUSICAL — {title}",
              f"  Ensayo narrativo · midi_analyzer v12.0", SEP, ""]

    # ── Resumen ejecutivo ────────────────────────────────────────
    parts.append(write_executive_summary(
        r.get('key_name', f"{r.get('key_root',0)} {r.get('mode','')}"),
        r.get('mode', ''),
        r.get('avg_bpm', 120),
        r.get('total_dur', 0),
        genre_detection=r.get('genre_detection'),
        dynamic_valence=r.get('dynamic_valence'),
        narrative_intention=r.get('narrative_intention'),
        catharsis=r.get('catharsis'),
        anti_conventional=r.get('anti_conventional'),
        semantic_enrichment=r.get('semantic_enrichment'),
        unified_emotional_map=r.get('unified_emotional_map'),
        roughness=r.get('roughness'),
        melodic_markov=r.get('melodic_markov'),
    ))

    # ── I. Identidad ─────────────────────────────────────────────
    parts.append(write_identity(
        r.get('key_root', 0), r.get('mode', ''),
        r.get('kconf', 0), r.get('avg_bpm', 120),
        r.get('total_dur', 0),
        r.get('fingerprint'), r.get('cultural'),
        r.get('ts_str_val'), r.get('instruments'), r.get('key_name', ''),
        genre_detection=r.get('genre_detection'),
        semantic_enrichment=r.get('semantic_enrichment'),
    ))

    # ── II. Arquitectura temporal ────────────────────────────────
    parts.append(write_architecture(
        r.get('musical_form'), r.get('golden'), r.get('pnr'),
        r.get('narrative_arc'), r.get('dyn_segments'),
        r.get('sections', []), r.get('tension_curve', []),
        r.get('total_dur', 0), r.get('avg_bpm', 120),
        ssm=r.get('ssm'),
        narrative_intention=r.get('narrative_intention'),
        catharsis=r.get('catharsis'),
    ))

    # ── III. Lenguaje armónico ────────────────────────────────────
    parts.append(write_harmony(
        r.get('chords', []), r.get('key_root', 0), r.get('mode', ''),
        r.get('canon_progs', []), r.get('modal_borrow', []),
        r.get('cadences'), r.get('harmonic_graph'),
        r.get('tonal_gravity'), r.get('tonal_ambiguity'),
        r.get('harmonic_color'), r.get('harmonic_semantic'),
        r.get('subtext'), r.get('voice_leading'),
        r.get('pedal_points', []), r.get('elliptical'),
        chromaticism=r.get('chromaticism'),
        dissonance_res=r.get('dissonance_res'),
    ))

    # ── IV. Discurso melódico ─────────────────────────────────────
    parts.append(write_melody(
        r.get('interval_anal'), r.get('motifs', []),
        r.get('contour', ''), r.get('phrasing'),
        r.get('thematic_transforms'), r.get('thematic_recurrence'),
        r.get('expectation'), r.get('narrative_voice'),
        r.get('comfort_zones'), r.get('accomp', ''),
        r.get('inner_voice'),
        counterpoint=r.get('counterpoint'),
        story=r.get('story'),
    ))

    # ── V. Dimensión rítmica ──────────────────────────────────────
    parts.append(write_rhythm(
        r.get('rhythm'), r.get('micro_timing'),
        r.get('metric_hierarchy'), r.get('polyrhythm'),
        r.get('groove'), r.get('bass_narrative'),
        r.get('layer_sync'), r.get('avg_bpm', 120),
        r.get('ts_str_val'),
        tempo_gestures=r.get('tempo_gestures'),
        builds=r.get('builds'),
    ))

    # ── VI. Experiencia del oyente ────────────────────────────────
    parts.append(write_listener_experience(
        r.get('listener_model'), r.get('cumulative_state'),
        r.get('auditory_fatigue'), r.get('emotional_transitions'),
        r.get('suspense'), r.get('param_convergence'),
        r.get('tension_curve', []), r.get('energy_profile'),
        r.get('silence_analysis'), r.get('eternity'),
        r.get('total_dur', 0),
        dynamic_valence=r.get('dynamic_valence'),
        emotional_ambivalence=r.get('emotional_ambivalence'),
        emotional_trajectory=r.get('emotional_trajectory'),
        unified_emotional_map=r.get('unified_emotional_map'),
        roughness=r.get('roughness'),
        perceptual=r.get('perceptual'),
        cinematic=r.get('cinematic'),
        climax_comparison=r.get('climax_comparison'),
        momentum=r.get('momentum'),
        polarity=r.get('polarity'),
    ))

    # ── VII. Intención compositiva ────────────────────────────────
    parts.append(write_compositional_intention(
        r.get('fractal'), r.get('emotional_weight'),
        r.get('golden'), r.get('coherence'),
        r.get('tonal_ambiguity'), r.get('harmonic_semantic'),
        r.get('narrative_arc'), r.get('microexpression'),
        r.get('fingerprint'), r.get('motifs', []),
        r.get('thematic_transforms'), r.get('mode', ''),
        r.get('avg_bpm', 120),
        anti_conventional=r.get('anti_conventional'),
        multilevel_density=r.get('multilevel_density'),
        melodic_markov=r.get('melodic_markov'),
        semantic_enrichment=r.get('semantic_enrichment'),
    ))

    # ── VIII. Evolución emocional ─────────────────────────────────
    parts.append(write_emotional_evolution(
        unified_emotional_map=r.get('unified_emotional_map'),
        dynamic_valence=r.get('dynamic_valence'),
        emotional_ambivalence=r.get('emotional_ambivalence'),
        emotional_trajectory=r.get('emotional_trajectory'),
        catharsis=r.get('catharsis'),
        polarity=r.get('polarity'),
        momentum=r.get('momentum'),
        total_dur=r.get('total_dur', 0),
    ))

    # ── IX. Historia emocional ────────────────────────────────────
    parts.append(write_emotional_story_section(
        story=r.get('story'),
        catharsis=r.get('catharsis'),
        narrative_intention=r.get('narrative_intention'),
        emotional_ambivalence=r.get('emotional_ambivalence'),
        cinematic=r.get('cinematic'),
        anti_conventional=r.get('anti_conventional'),
        semantic_enrichment=r.get('semantic_enrichment'),
        coherence=r.get('coherence'),
    ))

    # ── X. Síntesis ───────────────────────────────────────────────
    parts.append(write_synthesis(
        r.get('key_name', ''), r.get('mode', ''),
        r.get('avg_bpm', 120), r.get('total_dur', 0),
        r.get('emotional_genre'),
        r.get('narrative_arc'), r.get('cumulative_state'),
        r.get('silence_analysis'), r.get('tension_curve', []),
        r.get('pnr'), r.get('motifs', []),
        r.get('fingerprint'), r.get('cultural'),
        genre_detection=r.get('genre_detection'),
        catharsis=r.get('catharsis'),
        emotional_trajectory=r.get('emotional_trajectory'),
        narrative_intention=r.get('narrative_intention'),
    ))

    return '\n'.join(p for p in parts if p)
