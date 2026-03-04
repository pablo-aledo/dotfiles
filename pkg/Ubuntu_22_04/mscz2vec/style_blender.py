#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       STYLE BLENDER  v1.0                                    ║
║       Fusión dinámica de dos estilos a lo largo de una obra                  ║
║                                                                              ║
║  A diferencia de style_transfer.py (un estilo → toda la obra con strength   ║
║  fija), este módulo mezcla DOS fuentes de estilo usando una curva de blend   ║
║  que cambia compás a compás. Permite fusiones como épico→tango, clásico→    ║
║  jazz, o cualquier viaje entre dos mundos sonoros.                           ║
║                                                                              ║
║  MODELO DE BLEND:                                                             ║
║    blend=0.0  →  estilo A puro   (épico, clásico, etc.)                     ║
║    blend=0.5  →  fusión equitativa                                           ║
║    blend=1.0  →  estilo B puro   (tango, jazz, etc.)                        ║
║                                                                              ║
║  La curva puede definirse de tres formas:                                    ║
║    [1] --blend-curve "0:0.0, 16:0.5, 32:1.0"   puntos compás:valor         ║
║    [2] --blend-from-curves obra.curves.json      desde tension_designer      ║
║    [3] --blend-preset NAME                       forma predefinida           ║
║                                                                              ║
║  PRESETS DE BLEND:                                                            ║
║    gradual      — sube linealmente de 0.0 a 1.0 (A→B progresivo)           ║
║    inverse      — baja linealmente de 1.0 a 0.0 (B→A progresivo)           ║
║    arch         — A en extremos, B en el centro (A→B→A)                     ║
║    inverse_arch — B en extremos, A en el centro (B→A→B)                    ║
║    late_blend   — A dominante hasta el último tercio, luego B               ║
║    early_blend  — B irrumpe al principio, luego retorna A                   ║
║    wave         — oscila entre A y B repetidamente                          ║
║    sudden       — A durante la primera mitad, B de golpe en la segunda      ║
║                                                                              ║
║  DIMENSIONES CONTROLABLES (--blend-dims):                                    ║
║    rhythm    — patrón rítmico y groove                                       ║
║    harmony   — progresión de acordes y complejidad armónica                 ║
║    emotion   — curvas de tensión/arousal/valencia                           ║
║    texture   — estilo de acompañamiento                                     ║
║    dynamics  — envolvente de velocidades                                    ║
║    melody    — mezcla de las líneas melódicas generadas por cada estilo     ║
║                                                                              ║
║  INTEGRACIÓN CON EL ECOSISTEMA:                                              ║
║    theorist.py   → el plan .theorist.json puede guiar --blend-from-curves   ║
║    narrator.py   → las secciones del plan definen bloques de blend           ║
║    tension_designer.py → curvas exportadas como proxy directo del blend     ║
║    harvester.py  → motivos extraídos como contenido de entrada              ║
║    orchestrator.py → el MIDI resultante se orquesta normalmente             ║
║                                                                              ║
║  USO:                                                                        ║
║    # Blend gradual style1→style2 en 32 compases                               ║
║    python style_blender.py contenido.mid style1.mid style2.mid \              ║
║        --blend-preset gradual --bars 32                                      ║
║                                                                              ║
║    # Curva manual: style1 al inicio, style2 en el clímax, style1 al final      ║
║    python style_blender.py contenido.mid style1.mid style2.mid \              ║
║        --blend-curve "0:0.0, 8:0.1, 24:0.9, 32:0.1, 40:0.0"               ║
║                                                                              ║
║    # Usar la curva de tensión de la obra como proxy del blend               ║
║    python style_blender.py contenido.mid style1.mid style2.mid \              ║
║        --blend-from-curves obra.curves.json                                 ║
║                                                                              ║
║    # Control por secciones del narrator                                      ║
║    python style_blender.py contenido.mid style1.mid style2.mid \              ║
║        --blend-from-narrator obra_plan.json \                               ║
║        --section-blend "A:0.0, B:0.8, C:0.4, A2:0.1"                      ║
║                                                                              ║
║    # Solo ciertas dimensiones                                                ║
║    python style_blender.py contenido.mid style1.mid style2.mid \              ║
║        --blend-dims rhythm texture --blend-preset arch                      ║
║                                                                              ║
║    # Exportar la curva de blend usada (para inspeccionarla)                 ║
║    python style_blender.py contenido.mid style1.mid style2.mid \              ║
║        --blend-preset wave --export-blend-curve                             ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --blend-curve STR   Puntos de la curva "bar:value, ..."                  ║
║    --blend-preset NAME Forma predefinida de la curva (ver presets arriba)   ║
║    --blend-from-curves Leer curva de tensión de un .curves.json             ║
║    --blend-from-narrator Leer secciones de un obra_plan.json               ║
║    --section-blend STR Valor de blend por sección "A:0.0, B:0.8, ..."      ║
║    --blend-dims DIMS   Dimensiones a mezclar (default: todas)               ║
║    --preserve STRATS   Preservación melódica: contour motif scale intervals ║
║    --bars N            Compases de salida (default: auto desde contenido)   ║
║    --candidates N      Candidatos por segmento (default: 2)                 ║
║    --output FILE       MIDI de salida (default: blend_out.mid)              ║
║    --export-blend-curve Guardar la curva de blend usada en JSON             ║
║    --export-fingerprint Exportar fingerprint del resultado                  ║
║    --no-percussion     No generar percusión                                 ║
║    --seed N            Semilla aleatoria (default: 42)                      ║
║    --verbose           Informe detallado compás a compás                    ║
║                                                                              ║
║  DEPENDENCIAS: midi_dna_unified.py, style_transfer.py, mido, numpy         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import math
import random
import copy

import numpy as np

# ── Importar desde el ecosistema ──────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import midi_dna_unified as dna_mod
    from midi_dna_unified import (
        UnifiedDNA, EmotionalController, FormGenerator,
        MarkovMelody, GrooveMap,
        generate_accompaniment, generate_bass, generate_counterpoint,
        generate_percussion, add_ornamentation, humanize,
        build_midi, score_candidate,
        _snap_to_scale, _get_scale_midi, _get_scale_pcs,
        _quarter_to_ticks, _generate_melody_with_modulation,
        INSTRUMENT_RANGES,
    )
    from music21 import pitch as m21pitch, key as m21key
except ImportError as e:
    print(f"ERROR importando midi_dna_unified: {e}")
    print("Asegúrate de que midi_dna_unified.py está en el mismo directorio.")
    sys.exit(1)

try:
    from style_transfer import (
        extract_raw_melody,
        blend_rhythm_pattern,
        blend_harmony_prog,
        blend_emotional_curves,
        blend_groove,
        get_acc_style_from_style_dna,
        preserve_contour,
        preserve_intervals,
        preserve_motif,
        snap_to_style_scale,
        transpose_to_style_key,
        _blend_melodies,
    )
except ImportError as e:
    print(f"ERROR importando style_transfer: {e}")
    print("Asegúrate de que style_transfer.py está en el mismo directorio.")
    sys.exit(1)

import mido


# ══════════════════════════════════════════════════════════════════════════════
#  CURVAS DE BLEND
#  Todas las curvas devuelven un array de longitud n_bars con valores [0.0, 1.0]
#  donde 0.0 = estilo A puro y 1.0 = estilo B puro.
# ══════════════════════════════════════════════════════════════════════════════

def _interp_curve(points_dict, n_bars):
    """
    Interpola linealmente entre puntos {bar: value} para producir
    un array de n_bars valores.
    points_dict puede contener claves fuera de [0, n_bars-1]; se clampan.
    """
    if not points_dict:
        return np.full(n_bars, 0.5)

    sorted_pts = sorted(points_dict.items())
    xs = np.array([p[0] for p in sorted_pts], dtype=float)
    ys = np.array([p[1] for p in sorted_pts], dtype=float)
    bars = np.arange(n_bars, dtype=float)
    return np.clip(np.interp(bars, xs, ys), 0.0, 1.0)


BLEND_PRESETS = {
    "gradual":      lambda n: np.linspace(0.0, 1.0, n),
    "inverse":      lambda n: np.linspace(1.0, 0.0, n),
    "arch":         lambda n: np.array([
                        0.5 - 0.5 * math.cos(math.pi * i / max(n - 1, 1))
                        for i in range(n)]),
    "inverse_arch": lambda n: np.array([
                        0.5 + 0.5 * math.cos(math.pi * i / max(n - 1, 1))
                        for i in range(n)]),
    "late_blend":   lambda n: np.array([
                        0.0 if i < n * 2 / 3
                        else (i - n * 2 / 3) / max(n / 3, 1)
                        for i in range(n)]),
    "early_blend":  lambda n: np.array([
                        1.0 - i / max(n / 3, 1) if i < n / 3 else 0.0
                        for i in range(n)]),
    "wave":         lambda n: np.array([
                        0.5 - 0.5 * math.cos(2 * math.pi * i / max(n / 2, 1))
                        for i in range(n)]),
    "sudden":       lambda n: np.array([
                        0.0 if i < n / 2 else 1.0
                        for i in range(n)]),
}


def build_blend_curve_from_string(curve_str, n_bars):
    """
    Parsea "--blend-curve '0:0.0, 8:0.5, 32:1.0'" en un array de n_bars valores.
    """
    points = {}
    for token in curve_str.split(","):
        token = token.strip()
        if ":" not in token:
            continue
        bar_str, val_str = token.split(":", 1)
        try:
            bar = int(bar_str.strip())
            val = float(val_str.strip())
            points[bar] = val
        except ValueError:
            continue
    if not points:
        raise ValueError(f"Curva de blend inválida: '{curve_str}'")
    return _interp_curve(points, n_bars)


def build_blend_curve_from_curves_json(curves_path, n_bars):
    """
    Lee un .curves.json de tension_designer y usa la curva de tensión
    como proxy directo del blend (tensión alta = más estilo B).
    """
    with open(curves_path) as f:
        data = json.load(f)

    # tension_designer exporta {"T": [...], "A": [...], ...}
    # Intentar distintas claves según el formato
    tension = None
    for key in ("T", "tension", "tension_curve"):
        if key in data and data[key]:
            tension = data[key]
            break

    if tension is None:
        # Buscar el primer array que haya
        for v in data.values():
            if isinstance(v, list) and len(v) > 1:
                tension = v
                break

    if tension is None:
        raise ValueError(f"No se encontró curva de tensión en {curves_path}")

    # Interpolar a n_bars
    src = np.array(tension, dtype=float)
    bars = np.linspace(0, len(src) - 1, n_bars)
    lo = np.floor(bars).astype(int)
    hi = np.clip(lo + 1, 0, len(src) - 1)
    t = bars - lo
    curve = src[lo] * (1 - t) + src[hi] * t
    return np.clip(curve, 0.0, 1.0)


def build_blend_curve_from_narrator(plan_path, section_blend_str, n_bars):
    """
    Usa el plan de narrator.py + un mapeo sección→valor para construir
    la curva de blend. section_blend_str: "A:0.0, B:0.8, C:0.4"
    """
    with open(plan_path) as f:
        plan = json.load(f)

    # Parsear section_blend
    sec_map = {}
    for token in section_blend_str.split(","):
        token = token.strip()
        if ":" not in token:
            continue
        sec_str, val_str = token.split(":", 1)
        try:
            sec_map[sec_str.strip()] = float(val_str.strip())
        except ValueError:
            continue

    # Leer secciones del plan
    sections = plan.get("sections", [])
    if not sections:
        raise ValueError(f"No se encontraron secciones en {plan_path}")

    # Construir puntos compás→valor
    points = {}
    for sec in sections:
        label = sec.get("label", "A")
        start = sec.get("start_bar", 0)
        end   = sec.get("end_bar",   start + sec.get("bars", 8))
        blend_val = sec_map.get(label, sec_map.get(label.rstrip("0123456789"), 0.0))
        points[start] = blend_val
        points[end]   = blend_val  # mantener el valor hasta el final de la sección

    return _interp_curve(points, n_bars)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERPOLACIÓN DE DNA POR COMPÁS
#  El núcleo del blender: mezcla dos DNAs con un alpha variable.
# ══════════════════════════════════════════════════════════════════════════════

def _lerp(a, b, t):
    """Interpolación lineal escalar."""
    return a * (1.0 - t) + b * t


def _interpolate_dna_at_bar(dna_a, dna_b, alpha, bar_idx, n_bars):
    """
    Construye un DNA mezclado para el compás bar_idx con el alpha dado.
    No copia arrays completos — sólo los campos relevantes para la generación
    de ese compás.
    """
    d = copy.copy(dna_a)  # shallow copy como base

    # ── Tonalidad: usar la de mayor peso ─────────────────────────────────────
    d.key_obj = dna_b.key_obj if alpha >= 0.5 else dna_a.key_obj

    # ── Tempo: interpolado ────────────────────────────────────────────────────
    d.tempo_bpm = _lerp(dna_a.tempo_bpm, dna_b.tempo_bpm, alpha)

    # ── Complejidad armónica: interpolada ─────────────────────────────────────
    d.harmony_complexity = _lerp(dna_a.harmony_complexity, dna_b.harmony_complexity, alpha)

    # ── Grooves: elegir según alpha ────────────────────────────────────────────
    if alpha >= 0.5 and dna_b.groove_map.trained:
        d.groove_map = dna_b.groove_map
    elif dna_a.groove_map.trained:
        d.groove_map = dna_a.groove_map

    # ── Estilo de acompañamiento: del DNA dominante ───────────────────────────
    d.style = dna_b.style if alpha >= 0.5 else dna_a.style

    # ── Swing ─────────────────────────────────────────────────────────────────
    d.swing = dna_b.swing if alpha >= 0.5 else dna_a.swing

    # ── Dinámica media: interpolada ────────────────────────────────────────────
    d.dynamics_mean = _lerp(dna_a.dynamics_mean, dna_b.dynamics_mean, alpha)
    d.dynamics_std  = _lerp(dna_a.dynamics_std,  dna_b.dynamics_std,  alpha)

    # ── Sincopación: interpolada ───────────────────────────────────────────────
    d.syncopation_ratio = _lerp(dna_a.syncopation_ratio, dna_b.syncopation_ratio, alpha)

    return d


def _blend_rhythm_grids(dna_a, dna_b, alpha):
    """Interpola los grids rítmicos de 16 subdivisiones."""
    ga = np.array(dna_a.rhythm_grid,        dtype=float)
    gb = np.array(dna_b.rhythm_grid,        dtype=float)
    aa = np.array(dna_a.rhythm_accent_grid, dtype=float)
    ab = np.array(dna_b.rhythm_accent_grid, dtype=float)

    n = max(len(ga), len(gb), 16)

    def _pad(arr, n):
        if len(arr) < n:
            return np.pad(arr, (0, n - len(arr)))
        return arr[:n]

    ga, gb, aa, ab = _pad(ga, n), _pad(gb, n), _pad(aa, n), _pad(ab, n)
    return ga * (1 - alpha) + gb * alpha, aa * (1 - alpha) + ab * alpha


def _blend_harmony_at_bar(dna_a, dna_b, alpha, bar_idx, bars_per_repeat=4):
    """
    Selecciona el acorde del compás bar_idx desde la progresión mezclada.
    Usa blend_harmony_prog de style_transfer con el alpha de este compás.
    """
    prog = blend_harmony_prog(dna_a, dna_b, alpha)
    if not prog:
        return [('I', 4.0)]
    # Seleccionar el compás correcto cíclicamente
    idx = bar_idx % max(len(prog), 1)
    return [prog[idx]]


def _blend_curves_at_bar(dna_a, dna_b, alpha, bar_idx, n_bars):
    """
    Obtiene los valores de las curvas emocionales para un compás concreto,
    interpolados entre los dos DNAs.
    """
    def _sample(curve, bar_idx, n_bars):
        if not curve:
            return 0.5
        pos = bar_idx * (len(curve) - 1) / max(n_bars - 1, 1)
        lo = int(pos)
        hi = min(lo + 1, len(curve) - 1)
        t = pos - lo
        return curve[lo] * (1 - t) + curve[hi] * t

    tension   = _lerp(_sample(dna_a.tension_curve,   bar_idx, n_bars),
                      _sample(dna_b.tension_curve,   bar_idx, n_bars), alpha)
    arousal   = _lerp(_sample(dna_a.arousal_curve,   bar_idx, n_bars),
                      _sample(dna_b.arousal_curve,   bar_idx, n_bars), alpha)
    valence   = _lerp(_sample(dna_a.valence_curve,   bar_idx, n_bars),
                      _sample(dna_b.valence_curve,   bar_idx, n_bars), alpha)
    stability = _lerp(_sample(dna_a.stability_curve, bar_idx, n_bars),
                      _sample(dna_b.stability_curve, bar_idx, n_bars), alpha)
    activity  = _lerp(_sample(dna_a.activity_curve,  bar_idx, n_bars),
                      _sample(dna_b.activity_curve,  bar_idx, n_bars), alpha)
    return tension, arousal, valence, stability, activity


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN POR SEGMENTOS
#  La obra se divide en segmentos de blend_segment_bars compases.
#  Cada segmento se genera con el alpha medio de ese rango.
#  Los segmentos se concatenan con crossfade de notas en las uniones.
# ══════════════════════════════════════════════════════════════════════════════

SEGMENT_BARS = 4   # Resolución de generación: cada 4 compases


def _alpha_mean(blend_curve, start_bar, end_bar):
    """Alpha medio para un rango de compases."""
    segment = blend_curve[start_bar:end_bar]
    return float(np.mean(segment)) if len(segment) > 0 else 0.5


def _alpha_dominant(blend_curve, start_bar, end_bar):
    """
    Devuelve (alpha_mean, needs_crossfade).
    needs_crossfade es True si hay un cambio brusco dentro del segmento.
    """
    segment = blend_curve[start_bar:end_bar]
    if len(segment) == 0:
        return 0.5, False
    mean_alpha = float(np.mean(segment))
    spread = float(np.max(segment) - np.min(segment)) if len(segment) > 1 else 0.0
    return mean_alpha, spread > 0.3


def run_blend_segment(
    dna_a, dna_b, content_dna, original_melody,
    blend_curve, start_bar, end_bar,
    blend_dims, preserve_strats,
    candidates, seed, verbose
):
    """
    Genera un segmento de (end_bar - start_bar) compases con el alpha
    correspondiente a esa zona de la curva.
    Devuelve (mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig).
    """
    alpha, has_transition = _alpha_dominant(blend_curve, start_bar, end_bar)
    seg_bars = end_bar - start_bar
    bpb = dna_a.time_sig[0]

    if verbose:
        print(f"    Segmento [{start_bar}–{end_bar}]  alpha={alpha:.2f}"
              f"{'  ⚡ transición' if has_transition else ''}")

    # ── DNA mezclado para este segmento ──────────────────────────────────────
    mixed = _interpolate_dna_at_bar(dna_a, dna_b, alpha, start_bar, end_bar - start_bar)

    # ── Progresión armónica ────────────────────────────────────────────────────
    if 'harmony' in blend_dims:
        mixed.harmony_prog = blend_harmony_prog(dna_a, dna_b, alpha)
        if not mixed.harmony_prog:
            mixed.harmony_prog = dna_a.harmony_prog or [('I', float(bpb))]
    else:
        mixed.harmony_prog = content_dna.harmony_prog or dna_a.harmony_prog or [('I', float(bpb))]

    # ── Patrón rítmico ────────────────────────────────────────────────────────
    if 'rhythm' in blend_dims:
        mixed.rhythm_pattern = blend_rhythm_pattern(dna_a, dna_b, alpha)
        mixed.rhythm_grid, mixed.rhythm_accent_grid = _blend_rhythm_grids(dna_a, dna_b, alpha)
    else:
        mixed.rhythm_pattern = content_dna.rhythm_pattern or dna_a.rhythm_pattern

    # ── Curvas emocionales ────────────────────────────────────────────────────
    if 'emotion' in blend_dims:
        tc_full, ac_full, vc_full, sc_full, ec_full = blend_emotional_curves(dna_a, dna_b, alpha)
        # Recortar al segmento
        def _crop(c, s, e, total):
            if not c:
                return [0.5] * (e - s)
            n = len(c)
            return [c[min(int(s + i * (e - s) / max(e - s, 1) * n / total), n - 1)]
                    for i in range(e - s)]
        mixed.tension_curve  = tc_full
        mixed.arousal_curve  = ac_full
        mixed.valence_curve  = vc_full
        mixed.stability_curve = sc_full
        mixed.activity_curve  = ec_full
    else:
        pass  # mantener las del DNA base

    # ── Target key ────────────────────────────────────────────────────────────
    if 'melody' in blend_dims or 'harmony' in blend_dims:
        target_key = dna_b.key_obj if alpha >= 0.5 else dna_a.key_obj
    else:
        target_key = content_dna.key_obj

    # ── Controllers ───────────────────────────────────────────────────────────
    ec = EmotionalController(
        tension_curve    = mixed.tension_curve   or [0.5],
        arousal_curve    = mixed.arousal_curve   or [0.0],
        valence_curve    = mixed.valence_curve   or [0.0],
        stability_curve  = mixed.stability_curve or [0.7],
        activity_curve   = mixed.activity_curve  or [0.5],
        emotional_arc_label = mixed.emotional_arc_label,
        n_bars = seg_bars,
    )

    # Forma: heredar del DNA con mayor peso
    src_form = dna_b if alpha >= 0.5 else dna_a
    fg = FormGenerator(
        form_string       = src_form.form_string,
        section_map       = src_form.section_map,
        phrase_lengths    = src_form.phrase_lengths,
        cadence_positions = src_form.cadence_positions,
        n_bars_out        = seg_bars,
    )

    # ── Melodía de contenido recortada al segmento ────────────────────────────
    beats_start = start_bar * bpb
    beats_end   = end_bar   * bpb
    seg_melody  = [
        (o - beats_start, p, d, v)
        for o, p, d, v in original_melody
        if beats_start <= o < beats_end
    ]

    # Aplicar key transfer si corresponde
    if 'melody' in blend_dims and seg_melody:
        seg_melody = transpose_to_style_key(seg_melody, content_dna.key_obj, target_key)

    # Preservación melódica
    if 'contour' in preserve_strats and seg_melody:
        seg_melody = preserve_contour(seg_melody, target_key)
    if 'scale' in preserve_strats and seg_melody:
        seg_melody = snap_to_style_scale(seg_melody, target_key)

    # ── Generación de melodía con el DNA mezclado ─────────────────────────────
    best_score  = -1.0
    best_result = None

    markov_src = dna_b.markov if alpha >= 0.5 else dna_a.markov
    seq_src    = dna_b.sequitur_phrases if alpha >= 0.5 else dna_a.sequitur_phrases

    for cand_i in range(max(1, candidates)):
        random.seed(seed + start_bar * 100 + cand_i * 13)
        np.random.seed(seed + start_bar * 100 + cand_i * 13)

        generated = _generate_melody_with_modulation(
            h_prog       = mixed.harmony_prog,
            target_key   = target_key,
            r_pat        = mixed.rhythm_pattern,
            contour      = mixed.pitch_contour or content_dna.pitch_contour,
            reg          = content_dna.pitch_register,
            motif        = content_dna.motif_intervals,
            n_bars       = seg_bars,
            ec           = ec,
            fg           = fg,
            bpb          = bpb,
            rhythm_strength = 1.0,
            markov          = markov_src,
            seq_phrases     = seq_src,
            melody_mode     = 'markov',
            surprise_rate   = 0.06 + alpha * 0.06,
            use_motif_coherence = True,
            use_tension_markov  = True,
        )

        # Mezclar melodía preservada y generada
        if seg_melody and generated and 'melody' in blend_dims:
            # alpha alto = más estilo B → más generada desde B
            # La melodía del contenido se va diluyendo según alpha
            blend_mel = max(0.0, 1.0 - alpha)
            final_mel = _blend_melodies(seg_melody, generated, blend_mel, target_key)
        elif seg_melody:
            final_mel = seg_melody
        else:
            final_mel = generated

        # Siembra del motivo original
        if 'motif' in preserve_strats and original_melody and final_mel:
            final_mel = preserve_motif(original_melody, final_mel, target_key)

        # Ornamentación según estilo dominante
        final_mel = add_ornamentation(final_mel, target_key,
                                      dna_b.style if alpha >= 0.5 else dna_a.style)

        # Humanización
        groove = blend_groove(dna_a, dna_b, alpha)
        if groove and groove.trained:
            final_mel = humanize(final_mel, groove, bpb)

        # Acompañamiento
        acc_style = None
        if 'texture' in blend_dims:
            acc_style = get_acc_style_from_style_dna(dna_b if alpha >= 0.5 else dna_a, alpha)
        acc = generate_accompaniment(
            mixed.harmony_prog, target_key, seg_bars, ec, fg, bpb,
            groove_map=groove,
            force_style=acc_style,
            harmony_complexity=mixed.harmony_complexity,
        )

        # Bajo
        bass = generate_bass(mixed.harmony_prog, target_key, seg_bars, bpb, groove)

        # Contrapunto
        cp = generate_counterpoint(final_mel, mixed.harmony_prog, target_key, seg_bars, bpb, ec)

        sc_val = score_candidate(final_mel, acc, target_key)
        if verbose:
            print(f"      candidato {cand_i+1}: score={sc_val:.3f}")
        if sc_val > best_score:
            best_score  = sc_val
            best_result = (final_mel, acc, bass, cp)

    mel, acc, bass, cp = best_result

    # Percusión mezclada
    rg, rag = _blend_rhythm_grids(dna_a, dna_b, alpha)
    perc = generate_percussion(
        rg, rag, seg_bars, bpb,
        groove_map=blend_groove(dna_a, dna_b, alpha),
        style=dna_b.style if alpha >= 0.5 else dna_a.style,
    )

    tempo_bpm = _lerp(dna_a.tempo_bpm, dna_b.tempo_bpm, alpha)
    time_sig  = dna_b.time_sig if alpha >= 0.5 else dna_a.time_sig

    return mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg


# ══════════════════════════════════════════════════════════════════════════════
#  CONCATENACIÓN DE SEGMENTOS
#  Une los segmentos generados desplazando sus offsets temporales y
#  aplicando crossfade de velocidades en las uniones para suavizar las
#  transiciones entre zonas de blend muy distintas.
# ══════════════════════════════════════════════════════════════════════════════

def _offset_notes(notes, beat_offset):
    """Desplaza todos los offsets de una lista de notas."""
    return [(o + beat_offset, p, d, v) for o, p, d, v in notes]


def _crossfade_velocities(notes_out, notes_in, crossfade_beats=2.0):
    """
    Aplica un fade-out suave al final de notes_out y fade-in al inicio
    de notes_in. Modifica in-place las velocidades en esa zona.
    """
    if not notes_out or not notes_in:
        return notes_out, notes_in

    max_out = max(o for o, _, _, _ in notes_out)
    min_in  = min(o for o, _, _, _ in notes_in)

    faded_out = []
    for o, p, d, v in notes_out:
        dist = max_out - o
        if dist < crossfade_beats:
            factor = 0.5 + 0.5 * (dist / crossfade_beats)
            v = max(1, int(v * factor))
        faded_out.append((o, p, d, v))

    faded_in = []
    for o, p, d, v in notes_in:
        dist = o - min_in
        if dist < crossfade_beats:
            factor = 0.5 + 0.5 * (dist / crossfade_beats)
            v = max(1, int(v * factor))
        faded_in.append((o, p, d, v))

    return faded_out, faded_in


def concatenate_segments(segments, blend_curve, bpb, crossfade=True):
    """
    Concatena una lista de segmentos [(mel, acc, bass, cp, perc, key, tempo, ts, fg), bars].
    Devuelve (mel, acc, bass, cp, perc) concatenados con offsets correctos.
    """
    all_mel, all_acc, all_bass, all_cp, all_perc = [], [], [], [], []
    beat_cursor = 0.0

    for i, (seg, seg_bars) in enumerate(segments):
        mel, acc, bass, cp, perc, key, tempo, ts, fg = seg
        offset = beat_cursor

        mel_shifted  = _offset_notes(mel,  offset)
        acc_shifted  = _offset_notes(acc,  offset)
        bass_shifted = _offset_notes(bass, offset)
        cp_shifted   = _offset_notes(cp,   offset)
        perc_shifted = _offset_notes(perc, offset)

        # Crossfade en la unión con el segmento anterior
        if crossfade and i > 0 and all_mel:
            # Detectar si hay un salto brusco de blend
            prev_end   = (beat_cursor - seg_bars * bpb) / bpb
            prev_alpha = blend_curve[max(0, int(prev_end))]
            curr_alpha = blend_curve[min(len(blend_curve) - 1, int(beat_cursor / bpb))]
            if abs(curr_alpha - prev_alpha) > 0.25:
                all_mel[-1], mel_shifted = _crossfade_velocities(all_mel[-1], mel_shifted)

        all_mel.append(mel_shifted)
        all_acc.append(acc_shifted)
        all_bass.append(bass_shifted)
        all_cp.append(cp_shifted)
        all_perc.append(perc_shifted)

        beat_cursor += seg_bars * bpb

    # Aplanar
    flat = lambda lists: [note for seg in lists for note in seg]
    return flat(all_mel), flat(all_acc), flat(all_bass), flat(all_cp), flat(all_perc)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_style_blender(
    content_dna, dna_a, dna_b,
    original_melody,
    blend_curve,             # np.array [n_bars]
    blend_dims=None,
    preserve_strats=None,
    n_bars=32,
    candidates=2,
    include_percussion=True,
    seed=42,
    verbose=False,
):
    """
    Motor principal del blender.
    Divide la obra en segmentos de SEGMENT_BARS compases y genera cada
    uno con el alpha correspondiente. Devuelve el material completo concatenado.
    """
    bpb = dna_a.time_sig[0]

    ALL_DIMS     = {'rhythm', 'harmony', 'emotion', 'texture', 'melody'}
    ALL_PRESERVE = {'contour', 'motif', 'scale'}

    if blend_dims is None:
        blend_dims = ALL_DIMS
    if preserve_strats is None:
        preserve_strats = ALL_PRESERVE

    # Asegurar que la curva de blend tiene exactamente n_bars puntos
    if len(blend_curve) != n_bars:
        xs = np.linspace(0, len(blend_curve) - 1, n_bars)
        lo = np.floor(xs).astype(int)
        hi = np.clip(lo + 1, 0, len(blend_curve) - 1)
        t  = xs - lo
        blend_curve = blend_curve[lo] * (1 - t) + blend_curve[hi] * t

    # Dividir en segmentos
    segments = []
    bar = 0
    while bar < n_bars:
        seg_end  = min(bar + SEGMENT_BARS, n_bars)
        seg_bars = seg_end - bar

        seg = run_blend_segment(
            dna_a, dna_b, content_dna, original_melody,
            blend_curve, bar, seg_end,
            blend_dims, preserve_strats,
            candidates, seed, verbose,
        )
        segments.append((seg, seg_bars))
        bar = seg_end

    # Concatenar
    mel, acc, bass, cp, perc = concatenate_segments(
        segments, blend_curve, bpb, crossfade=True
    )

    # Forma y tempo finales: del DNA con mayor peso medio
    mean_alpha = float(np.mean(blend_curve))
    final_dna  = dna_b if mean_alpha >= 0.5 else dna_a
    target_key = dna_b.key_obj if mean_alpha >= 0.5 else dna_a.key_obj

    # Tempo: usar el del último segmento (respeta la evolución)
    _, _, _, _, _, _, tempo_bpm, time_sig, fg = segments[-1][0]

    if not include_percussion:
        perc = []

    return mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg, blend_curve


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME DE BLEND
# ══════════════════════════════════════════════════════════════════════════════

def print_blend_report(blend_curve, dna_a, dna_b, n_bars):
    """Imprime un mapa ASCII de la curva de blend."""
    print("\n  CURVA DE BLEND (0=estilo A puro, 1=estilo B puro)")
    print(f"  {'Compás':<6}  {'Blend':>5}  Visualización")
    print("  " + "─" * 52)

    step = max(1, n_bars // 20)
    for bar in range(0, n_bars, step):
        alpha = blend_curve[min(bar, len(blend_curve) - 1)]
        bar_len  = int(alpha * 30)
        bar_vis  = "█" * bar_len + "░" * (30 - bar_len)
        dominant = dna_b.name if alpha >= 0.5 else dna_a.name
        print(f"  {bar:<6}  {alpha:>5.2f}  {bar_vis}  {dominant}")
    print()

    # Estadísticas
    transitions = np.sum(np.abs(np.diff(blend_curve)) > 0.2)
    print(f"  Alpha medio : {np.mean(blend_curve):.2f}")
    print(f"  Rango       : [{np.min(blend_curve):.2f}, {np.max(blend_curve):.2f}]")
    print(f"  Transiciones bruscas: {transitions}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="STYLE BLENDER — Fusión dinámica de dos estilos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("content",  help="MIDI de contenido (tu idea melódica)")
    parser.add_argument("style_a",  help="MIDI de estilo A (blend=0.0)")
    parser.add_argument("style_b",  help="MIDI de estilo B (blend=1.0)")

    # Curva de blend
    g_curve = parser.add_mutually_exclusive_group()
    g_curve.add_argument("--blend-curve",         metavar="STR",
                         help='Puntos de la curva "bar:value, ..." ej: "0:0.0,16:0.8,32:0.1"')
    g_curve.add_argument("--blend-preset",        metavar="NAME",
                         choices=list(BLEND_PRESETS.keys()),
                         help=f"Preset: {', '.join(BLEND_PRESETS.keys())}")
    g_curve.add_argument("--blend-from-curves",   metavar="FILE",
                         help="Curva desde .curves.json de tension_designer")
    g_curve.add_argument("--blend-from-narrator", metavar="FILE",
                         help="Plan de narrator.py como guía de blend")

    parser.add_argument("--section-blend", metavar="STR",
                        help='Con --blend-from-narrator: "A:0.0, B:0.8, C:0.4"')

    # Dimensiones
    parser.add_argument("--blend-dims", nargs="+",
                        choices=["rhythm", "harmony", "emotion", "texture", "melody"],
                        default=None,
                        help="Dimensiones a mezclar (default: todas)")
    parser.add_argument("--preserve", nargs="+",
                        choices=["contour", "intervals", "motif", "scale"],
                        default=None,
                        help="Estrategias de preservación melódica")

    # Generación
    parser.add_argument("--bars",       type=int, default=None)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--seed",       type=int, default=42)

    # Salida
    parser.add_argument("--output",               default="blend_out.mid")
    parser.add_argument("--export-blend-curve",   action="store_true",
                        help="Guardar la curva de blend usada en JSON")
    parser.add_argument("--export-fingerprint",   action="store_true")
    parser.add_argument("--no-percussion",        action="store_true")
    parser.add_argument("--verbose",              action="store_true")

    args = parser.parse_args()

    # Validar archivos
    for path in [args.content, args.style_a, args.style_b]:
        if not os.path.exists(path):
            print(f"ERROR: No encontrado: {path}")
            sys.exit(1)

    print("═" * 65)
    print("  STYLE BLENDER v1.0")
    print("═" * 65)
    print(f"  Contenido : {os.path.basename(args.content)}")
    print(f"  Estilo A  : {os.path.basename(args.style_a)}  (blend=0.0)")
    print(f"  Estilo B  : {os.path.basename(args.style_b)}  (blend=1.0)")

    # ── [1] Extraer melodía del contenido ─────────────────────────────────────
    print("\n[1/4] Extrayendo melodía del contenido…")
    original_melody, content_tempo, content_ts = extract_raw_melody(
        args.content, verbose=args.verbose)
    print(f"  ✓ {len(original_melody)} notas extraídas")

    # ── [2] Extraer DNAs ───────────────────────────────────────────────────────
    print("\n[2/4] Extrayendo ADN de los tres MIDIs…")
    print(f"  ▶ Contenido : {os.path.basename(args.content)}")
    content_dna = UnifiedDNA(args.content)
    content_dna.extract(verbose=args.verbose)

    print(f"  ▶ Estilo A  : {os.path.basename(args.style_a)}")
    dna_a = UnifiedDNA(args.style_a)
    dna_a.extract(verbose=args.verbose)

    print(f"  ▶ Estilo B  : {os.path.basename(args.style_b)}")
    dna_b = UnifiedDNA(args.style_b)
    dna_b.extract(verbose=args.verbose)

    # ── [3] Determinar número de compases ─────────────────────────────────────
    if args.bars:
        n_bars = args.bars
    else:
        bpb = content_dna.time_sig[0]
        if original_melody:
            total_beats = max(o + d for o, _, d, _ in original_melody)
            n_bars = max(4, int(np.ceil(total_beats / bpb)))
        else:
            n_bars = 32
        n_bars = max(4, (n_bars // SEGMENT_BARS) * SEGMENT_BARS)

    print(f"\n  Compases de salida: {n_bars}")

    # ── [4] Construir curva de blend ───────────────────────────────────────────
    print("\n[3/4] Construyendo curva de blend…")

    if args.blend_curve:
        blend_curve = build_blend_curve_from_string(args.blend_curve, n_bars)
        print(f"  ✓ Curva manual con {len(args.blend_curve.split(','))} puntos")

    elif args.blend_preset:
        fn = BLEND_PRESETS[args.blend_preset]
        blend_curve = fn(n_bars)
        print(f"  ✓ Preset '{args.blend_preset}'")

    elif args.blend_from_curves:
        blend_curve = build_blend_curve_from_curves_json(args.blend_from_curves, n_bars)
        print(f"  ✓ Curva desde {os.path.basename(args.blend_from_curves)}")

    elif args.blend_from_narrator:
        if not args.section_blend:
            print("ERROR: --blend-from-narrator requiere --section-blend")
            sys.exit(1)
        blend_curve = build_blend_curve_from_narrator(
            args.blend_from_narrator, args.section_blend, n_bars)
        print(f"  ✓ Curva desde {os.path.basename(args.blend_from_narrator)}")

    else:
        # Default: gradual A→B
        blend_curve = BLEND_PRESETS["gradual"](n_bars)
        print("  ✓ Preset por defecto: 'gradual'")

    # Mostrar informe de la curva
    if args.verbose:
        print_blend_report(blend_curve, dna_a, dna_b, n_bars)

    print(f"  Alpha medio: {np.mean(blend_curve):.2f}  "
          f"rango: [{np.min(blend_curve):.2f}, {np.max(blend_curve):.2f}]")

    # Dimensiones y preservación
    blend_dims     = set(args.blend_dims)    if args.blend_dims else None
    preserve_strats = set(args.preserve)    if args.preserve   else None

    dims_str = ', '.join(sorted(blend_dims or {'rhythm','harmony','emotion','texture','melody'}))
    print(f"  Dimensiones: {dims_str}")

    # ── [4] Generar ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Generando {n_bars // SEGMENT_BARS} segmentos "
          f"({SEGMENT_BARS} compases c/u)…")

    mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg, blend_curve = \
        run_style_blender(
            content_dna, dna_a, dna_b,
            original_melody,
            blend_curve,
            blend_dims=blend_dims,
            preserve_strats=preserve_strats,
            n_bars=n_bars,
            candidates=args.candidates,
            include_percussion=not args.no_percussion,
            seed=args.seed,
            verbose=args.verbose,
        )

    print(f"\n  ✓ Melodía        : {len(mel)} notas")
    print(f"  ✓ Acompañamiento : {len(acc)} eventos")
    print(f"  ✓ Bajo           : {len(bass)} notas")
    print(f"  ✓ Contrapunto    : {len(cp)} notas")
    if not args.no_percussion:
        print(f"  ✓ Percusión      : {len(perc)} golpes")

    # ── Exportar MIDI ──────────────────────────────────────────────────────────
    build_midi(
        mel, acc, bass, cp,
        target_key, tempo_bpm, time_sig, n_bars,
        form_gen=fg,
        output_path=args.output,
        percussion_notes=None if args.no_percussion else perc,
    )
    print(f"\n  → MIDI: {args.output}")

    # ── Exportar curva de blend ────────────────────────────────────────────────
    if args.export_blend_curve:
        curve_path = args.output.replace(".mid", ".blend_curve.json")
        curve_data = {
            "n_bars":      n_bars,
            "style_a":     os.path.basename(args.style_a),
            "style_b":     os.path.basename(args.style_b),
            "blend_curve": blend_curve.tolist(),
            "alpha_mean":  float(np.mean(blend_curve)),
            "alpha_min":   float(np.min(blend_curve)),
            "alpha_max":   float(np.max(blend_curve)),
        }
        with open(curve_path, "w") as f:
            json.dump(curve_data, f, indent=2)
        print(f"  → Curva de blend: {curve_path}")

    # ── Exportar fingerprint ───────────────────────────────────────────────────
    if args.export_fingerprint:
        fp = dna_mod.extract_fingerprint(
            mel, bass, acc,
            target_key, tempo_bpm, n_bars, time_sig,
            fg, dna_b if np.mean(blend_curve) >= 0.5 else dna_a,
            args.output,
        )
        json_path = dna_mod.export_fingerprint(fp, args.output)
        print(f"  → Fingerprint: {json_path}")

    print("\n" + "═" * 65)
    print(f"  Blend completado: {args.output}")
    print(f"  Tonalidad final : {target_key.tonic.name} {target_key.mode}")
    print(f"  Tempo           : {tempo_bpm:.0f} BPM")
    print(f"  A→B completado  : {np.mean(blend_curve):.0%} promedio hacia estilo B")
    print("═" * 65)


if __name__ == "__main__":
    main()
