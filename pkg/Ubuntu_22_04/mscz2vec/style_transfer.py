"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     STYLE TRANSFER  v1.0                                     ║
║       Transferencia de estilo: aplica el ADN de un MIDI a otro               ║
║                                                                              ║
║  Dado un MIDI de «contenido» (tu idea melódica) y un MIDI de «estilo»,       ║
║  preserva la esencia melódica del primero mientras aplica el ADN rítmico,    ║
║  armónico, emocional y de textura del segundo. Es un «filtro de estilo»      ║
║  musical que mantiene tu voz creativa.                                       ║
║                                                                              ║
║  DIMENSIONES DE TRANSFERENCIA (controlables individualmente):                ║
║  [R] Ritmo/Groove     — patrón rítmico y timing humanizado del estilo        ║
║  [H] Armonía          — progresión armónica y complejidad del estilo         ║
║  [E] Emoción          — curvas de tensión/arousal/valencia del estilo        ║
║  [T] Timbre/Textura   — estilo de acompañamiento (alberti/arpeggio/bloque)   ║
║  [D] Dinámica         — envolvente de velocidades del estilo                 ║
║  [K] Tonalidad        — transponer contenido a la tonalidad del estilo       ║
║                                                                              ║
║  ESTRATEGIAS DE PRESERVACIÓN MELÓDICA:                                      ║
║  [P1] Preservar contorno: mantiene las subidas/bajadas del original          ║
║  [P2] Preservar intervalos: mantiene los saltos exactos                      ║
║  [P3] Preservar motivo: extrae y re-siembra el motivo del contenido          ║
║  [P4] Pitch-snapping: ajusta notas a la escala del estilo                   ║
║                                                                              ║
║  USO:                                                                        ║
║    python style_transfer.py contenido.mid estilo.mid                        ║
║    python style_transfer.py mi_melodia.mid bach.mid --strength 0.8          ║
║    python style_transfer.py mi_melodia.mid jazz.mid --no-key-transfer        ║
║    python style_transfer.py mi_melodia.mid estilo.mid \\                    ║
║        --transfer rhythm harmony --preserve contour motif                   ║
║    python style_transfer.py mi_melodia.mid estilo.mid \\                    ║
║        --strength 0.5 --blend-melody 0.6 --bars 32                         ║
║    python style_transfer.py mi_melodia.mid estilo.mid --verbose             ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --strength F       Intensidad global de transferencia 0-1 (default: 0.7) ║
║    --transfer DIMS    Dimensiones a transferir (default: todas)              ║
║                       rhythm harmony emotion texture dynamics key            ║
║    --no-KEY-transfer  Atajo para desactivar dimensiones individuales         ║
║    --preserve STRATS  Estrategias de preservación melódica (default: todas) ║
║                       contour intervals motif scale                         ║
║    --blend-melody F   Mezcla melodía original vs generada 0-1 (default: 0.5)║
║    --bars N           Compases de salida (default: auto desde contenido)     ║
║    --candidates N     Candidatos a evaluar (default: 3)                      ║
║    --output FILE      Fichero de salida (default: transfer_out.mid)          ║
║    --export-fingerprint  Exportar fingerprint del resultado                  ║
║    --seed N           Semilla (default: 42)                                  ║
║    --verbose          Informe detallado                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import random
import copy
import math

import numpy as np

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import midi_dna_unified as dna_mod
    from midi_dna_unified import (
        UnifiedDNA, EmotionalController, FormGenerator,
        MarkovMelody, GrooveMap,
        generate_accompaniment, generate_bass, generate_counterpoint,
        generate_percussion, add_ornamentation, humanize,
        humanize_with_swing, build_midi, score_candidate,
        _snap_to_scale, _get_scale_midi, _get_scale_pcs,
        _get_relative_key, _get_dominant_key,
        _quarter_to_ticks, INSTRUMENT_RANGES,
    )
    from music21 import pitch as m21pitch, key as m21key
except ImportError as e:
    print(f"ERROR: {e}\nmidi_dna_unified.py no encontrado en el mismo directorio.")
    sys.exit(1)

import mido


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE MELODÍA CRUDA DEL MIDI DE CONTENIDO
# ══════════════════════════════════════════════════════════════════════════════

def extract_raw_melody(midi_path, verbose=False):
    """
    Extrae la línea melódica del MIDI de contenido como lista de
    (offset_beats, pitch_midi, duration_beats, velocity).
    Devuelve también el tempo y la firma de tiempo.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000  # 120 BPM por defecto
    ts_num, ts_den = 4, 4

    # Recopilar notas por canal
    notes_by_channel = {}
    pending = {}

    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            abs_beats = abs_ticks / tpb

            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator

            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_beats, msg.velocity)

            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                key_ = (msg.channel, msg.note)
                if key_ in pending:
                    onset, vel = pending.pop(key_)
                    dur = max(0.1, abs_beats - onset)
                    ch = msg.channel
                    if ch != 9:  # ignorar percusión
                        notes_by_channel.setdefault(ch, []).append(
                            (onset, msg.note, dur, vel))

    tempo_bpm = 60_000_000 / max(tempo_us, 1)

    if not notes_by_channel:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    # Elegir el canal con pitch medio más alto (probable melodía)
    def _mean_pitch(notes):
        return sum(p for _, p, _, _ in notes) / len(notes) if notes else 0

    melody_ch = max(notes_by_channel.keys(), key=lambda ch: _mean_pitch(notes_by_channel[ch]))
    melody = sorted(notes_by_channel[melody_ch], key=lambda x: x[0])

    if verbose:
        print(f"    Melodía contenido: {len(melody)} notas  |  canal {melody_ch}")
        print(f"    Tempo: {tempo_bpm:.1f} BPM  |  compás: {ts_num}/{ts_den}")

    return melody, tempo_bpm, (ts_num, ts_den)


# ══════════════════════════════════════════════════════════════════════════════
#  PRESERVACIÓN MELÓDICA
# ══════════════════════════════════════════════════════════════════════════════

def _melody_contour(melody):
    """Extrae el contorno relativo (lista de intervalos) de la melodía."""
    pitches = [p for _, p, _, _ in melody]
    return [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]


def preserve_contour(original_melody, style_key, register_offset=0):
    """
    [P1] Reconstruye la melodía con el contorno original pero en la
    tonalidad y registro del estilo.
    """
    if not original_melody:
        return original_melody
    contour = _melody_contour(original_melody)
    if not contour:
        return original_melody

    # Punto de partida: tónica del estilo en registro medio
    tonic_pc = m21pitch.Pitch(style_key.tonic.name).pitchClass
    start = tonic_pc + 60 + register_offset
    start = _snap_to_scale(start, style_key)

    result = []
    current = start
    offsets = [o for o, _, _, _ in original_melody]
    durs    = [d for _, _, d, _ in original_melody]
    vels    = [v for _, _, _, v in original_melody]

    result.append((offsets[0], current, durs[0], vels[0]))
    for i, step in enumerate(contour):
        current = _snap_to_scale(current + step, style_key)
        current = max(52, min(84, current))
        idx = i + 1
        if idx < len(offsets):
            result.append((offsets[idx], current, durs[idx], vels[idx]))

    return result


def preserve_intervals(original_melody, style_key):
    """
    [P2] Mantiene los intervalos exactos pero ajusta la nota inicial
    a la escala del estilo.
    """
    if not original_melody:
        return original_melody
    start_pitch = original_melody[0][1]
    snapped_start = _snap_to_scale(start_pitch, style_key)
    shift = snapped_start - start_pitch

    return [
        (o, max(40, min(96, p + shift)), d, v)
        for o, p, d, v in original_melody
    ]


def preserve_motif(original_melody, generated_melody, style_key, n_motif=6):
    """
    [P3] Extrae el motivo principal del contenido (primeras n_motif notas)
    y lo siembra al inicio de la melodía generada.
    """
    if not original_melody or not generated_melody or len(original_melody) < 2:
        return generated_melody

    # Extraer motivo del original (intervalos relativos)
    motif_intervals = []
    orig_pitches = [p for _, p, _, _ in original_melody[:n_motif + 1]]
    for i in range(min(n_motif, len(orig_pitches) - 1)):
        motif_intervals.append(orig_pitches[i+1] - orig_pitches[i])

    if not motif_intervals:
        return generated_melody

    # Reconstruir el motivo desde la primera nota de la melodía generada
    start = generated_melody[0][1]
    seeded = [generated_melody[0]]
    current = start
    for i, iv in enumerate(motif_intervals):
        if i + 1 >= len(generated_melody):
            break
        current = _snap_to_scale(current + iv, style_key)
        current = max(52, min(84, current))
        o, _, d, v = generated_melody[i + 1]
        seeded.append((o, current, d, v))

    # Combinar motivo sembrado + resto de la melodía generada
    rest = list(generated_melody[len(seeded):])
    return seeded + rest


def snap_to_style_scale(melody, style_key):
    """[P4] Ajusta todas las notas a la escala del estilo."""
    return [
        (o, _snap_to_scale(p, style_key), d, v)
        for o, p, d, v in melody
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFERENCIA DE DIMENSIONES
# ══════════════════════════════════════════════════════════════════════════════

def blend_rhythm_pattern(content_dna, style_dna, strength):
    """[R] Mezcla el patrón rítmico entre contenido y estilo."""
    pat_c = content_dna.rhythm_pattern or [[]]
    pat_s = style_dna.rhythm_pattern or [[]]
    if not pat_s:
        return pat_c
    n = max(len(pat_c), len(pat_s))
    pat_c = (pat_c * (n // max(len(pat_c), 1) + 1))[:n]
    pat_s = (pat_s * (n // max(len(pat_s), 1) + 1))[:n]
    blended = []
    for bar_c, bar_s in zip(pat_c, pat_s):
        if random.random() < strength:
            blended.append(bar_s)  # usar patrón del estilo
        else:
            blended.append(bar_c)  # mantener patrón del contenido
    return blended


def blend_harmony_prog(content_dna, style_dna, strength):
    """[H] Mezcla la progresión armónica."""
    prog_c = content_dna.harmony_prog or [('I', 2.0)]
    prog_s = style_dna.harmony_prog or [('I', 2.0)]
    if strength >= 0.9:
        return prog_s
    if strength <= 0.1:
        return prog_c
    # Mezcla compás a compás
    n = max(len(prog_c), len(prog_s))
    prog_c = (prog_c * (n // max(len(prog_c), 1) + 1))[:n]
    prog_s = (prog_s * (n // max(len(prog_s), 1) + 1))[:n]
    blended = []
    for (fc, dc), (fs, ds) in zip(prog_c, prog_s):
        if random.random() < strength:
            blended.append((fs, ds))
        else:
            blended.append((fc, dc))
    return blended


def blend_emotional_curves(content_dna, style_dna, strength):
    """[E] Mezcla las curvas emocionales."""
    def _interp(ca, cb, alpha, n):
        """Interpola dos curvas de longitud variable a n puntos."""
        def _sample(c, i, n):
            if not c:
                return 0.5
            pos = i * (len(c) - 1) / max(n - 1, 1)
            lo = int(pos)
            hi = min(lo + 1, len(c) - 1)
            t = pos - lo
            return c[lo] * (1 - t) + c[hi] * t
        return [_sample(ca, i, n) * (1 - alpha) + _sample(cb, i, n) * alpha
                for i in range(n)]

    n_bars = max(
        len(style_dna.tension_curve or [0.5]),
        len(content_dna.tension_curve or [0.5]),
        4
    )

    tc = _interp(content_dna.tension_curve, style_dna.tension_curve, strength, n_bars)
    ac = _interp(content_dna.arousal_curve, style_dna.arousal_curve, strength, n_bars)
    vc = _interp(content_dna.valence_curve, style_dna.valence_curve, strength, n_bars)
    sc = _interp(content_dna.stability_curve, style_dna.stability_curve, strength, n_bars)
    ec = _interp(content_dna.activity_curve, style_dna.activity_curve, strength, n_bars)
    return tc, ac, vc, sc, ec


def get_acc_style_from_style_dna(style_dna, strength):
    """[T] Determina el estilo de acompañamiento basándose en el DNA de estilo."""
    if strength < 0.3:
        return None  # dejar que la emoción decida
    style = getattr(style_dna, 'style', 'generic')
    swing = getattr(style_dna, 'swing', False)
    complexity = getattr(style_dna, 'harmony_complexity', 0.3)
    # Inferir estilo de acompañamiento típico
    if swing:
        return 'block'
    elif complexity > 0.6:
        return 'arpeggio'
    elif style == 'baroque':
        return 'alberti'
    elif style == 'romantic':
        return 'arpeggio'
    else:
        return None


def transpose_to_style_key(melody, content_key, style_key):
    """[K] Transpone la melodía a la tonalidad del estilo."""
    if content_key is None or style_key is None:
        return melody
    try:
        src_pc = m21pitch.Pitch(content_key.tonic.name).pitchClass
        tgt_pc = m21pitch.Pitch(style_key.tonic.name).pitchClass
        shift = (tgt_pc - src_pc)
        if shift > 6:  shift -= 12
        if shift < -6: shift += 12
        return [(o, max(40, min(96, p + shift)), d, v) for o, p, d, v in melody]
    except Exception:
        return melody


def blend_groove(content_dna, style_dna, strength):
    """[D] Devuelve el groove_map a usar: del estilo si strength > umbral."""
    if strength > 0.5 and style_dna.groove_map.trained:
        return style_dna.groove_map
    elif content_dna.groove_map.trained:
        return content_dna.groove_map
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFERENCIA COMPLETA
# ══════════════════════════════════════════════════════════════════════════════

def run_style_transfer(
    content_dna, style_dna, original_melody,
    strength=0.7, n_bars=16,
    transfer_dims=None,   # conjunto de dimensiones a transferir
    preserve_strats=None, # estrategias de preservación
    blend_melody=0.5,     # 0=todo generado, 1=todo original preservado
    candidates=3,
    seed=42,
    verbose=False,
):
    """
    Motor principal de transferencia de estilo.
    Retorna (mel, acc, bass, cp, perc, target_key, tempo, time_sig)
    """
    random.seed(seed)
    np.random.seed(seed)

    ALL_DIMS     = {'rhythm', 'harmony', 'emotion', 'texture', 'dynamics', 'key'}
    ALL_PRESERVE = {'contour', 'intervals', 'motif', 'scale'}

    if transfer_dims is None:
        transfer_dims = ALL_DIMS
    if preserve_strats is None:
        preserve_strats = ALL_PRESERVE

    # ── Tonalidad objetivo ────────────────────────────────────────────────────
    target_key = style_dna.key_obj if 'key' in transfer_dims else content_dna.key_obj
    tempo_bpm  = style_dna.tempo_bpm if 'rhythm' in transfer_dims else content_dna.tempo_bpm
    time_sig   = style_dna.time_sig  if 'rhythm' in transfer_dims else content_dna.time_sig
    bpb = time_sig[0]

    if verbose:
        print(f"    Tonalidad destino: {target_key.tonic.name} {target_key.mode}")
        print(f"    Tempo destino    : {tempo_bpm:.0f} BPM")

    # ── Construir el ADN mezclado ─────────────────────────────────────────────
    mixed_dna = copy.deepcopy(content_dna)

    # [R] Ritmo
    if 'rhythm' in transfer_dims:
        mixed_dna.rhythm_pattern = blend_rhythm_pattern(content_dna, style_dna, strength)
        mixed_dna.rhythm_grid    = style_dna.rhythm_grid
        mixed_dna.groove_map     = style_dna.groove_map if style_dna.groove_map.trained \
                                   else content_dna.groove_map

    # [H] Armonía
    if 'harmony' in transfer_dims:
        mixed_dna.harmony_prog      = blend_harmony_prog(content_dna, style_dna, strength)
        mixed_dna.harmony_functions = [f for f, _ in mixed_dna.harmony_prog]
        mixed_dna.harmony_complexity = (
            content_dna.harmony_complexity * (1 - strength) +
            style_dna.harmony_complexity * strength
        )

    # [E] Emoción → curvas mezcladas
    if 'emotion' in transfer_dims:
        tc, ac, vc, sc, ec_c = blend_emotional_curves(content_dna, style_dna, strength)
        mixed_dna.tension_curve  = tc
        mixed_dna.arousal_curve  = ac
        mixed_dna.valence_curve  = vc
        mixed_dna.stability_curve = sc
        mixed_dna.activity_curve  = ec_c
        mixed_dna.emotional_arc_label = (
            style_dna.emotional_arc_label if strength > 0.5
            else content_dna.emotional_arc_label
        )

    # ── Construir controladores ───────────────────────────────────────────────
    ec = EmotionalController(
        tension_curve   = mixed_dna.tension_curve   or [0.5],
        arousal_curve   = mixed_dna.arousal_curve   or [0.0],
        valence_curve   = mixed_dna.valence_curve   or [0.0],
        stability_curve = mixed_dna.stability_curve or [0.7],
        activity_curve  = mixed_dna.activity_curve  or [0.5],
        emotional_arc_label = mixed_dna.emotional_arc_label,
        n_bars = n_bars,
    )

    fg = FormGenerator(
        form_string       = style_dna.form_string if 'emotion' in transfer_dims
                            else content_dna.form_string,
        section_map       = style_dna.section_map,
        phrase_lengths    = style_dna.phrase_lengths,
        cadence_positions = style_dna.cadence_positions,
        n_bars_out        = n_bars,
    )

    groove = blend_groove(content_dna, style_dna, strength)
    acc_style_forced = get_acc_style_from_style_dna(style_dna, strength) \
                       if 'texture' in transfer_dims else None

    # ── Transformar melodía original ──────────────────────────────────────────
    # Adaptar la melodía al número de compases de salida
    total_beats_out = n_bars * bpb
    if original_melody:
        # Escalar temporalmente la melodía al número de compases de salida
        total_beats_in = max(o + d for o, _, d, _ in original_melody)
        if total_beats_in > 0:
            time_scale = total_beats_out / total_beats_in
        else:
            time_scale = 1.0
        scaled_melody = [
            (o * time_scale, p, d * time_scale, v)
            for o, p, d, v in original_melody
        ]
    else:
        scaled_melody = []

    # Aplicar transferencia de tonalidad
    if 'key' in transfer_dims and scaled_melody:
        scaled_melody = transpose_to_style_key(scaled_melody, content_dna.key_obj, target_key)

    # Aplicar estrategias de preservación
    preserved_melody = scaled_melody
    if 'contour' in preserve_strats and preserved_melody:
        preserved_melody = preserve_contour(preserved_melody, target_key)
    if 'intervals' in preserve_strats and preserved_melody and 'contour' not in preserve_strats:
        preserved_melody = preserve_intervals(preserved_melody, target_key)
    if 'scale' in preserve_strats and preserved_melody:
        preserved_melody = snap_to_style_scale(preserved_melody, target_key)

    # ── Generar melodía desde el estilo (para mezclar con la preservada) ─────
    n_candidates = max(1, candidates)
    best_score = -1.0
    best_result = None

    for cand_idx in range(n_candidates):
        random.seed(seed + cand_idx * 13)
        np.random.seed(seed + cand_idx * 13)

        # Generar melodía con el DNA mezclado
        generated = dna_mod._generate_melody_with_modulation(
            h_prog       = mixed_dna.harmony_prog,
            target_key   = target_key,
            r_pat        = mixed_dna.rhythm_pattern,
            contour      = mixed_dna.pitch_contour or content_dna.pitch_contour,
            reg          = content_dna.pitch_register,
            motif        = content_dna.motif_intervals,
            n_bars       = n_bars,
            ec           = ec,
            fg           = fg,
            bpb          = bpb,
            rhythm_strength = 1.0,
            markov          = style_dna.markov,
            seq_phrases     = style_dna.sequitur_phrases,
            melody_mode     = 'markov',
            surprise_rate   = 0.08,
            use_motif_coherence = True,
            use_tension_markov  = True,
        )

        # Mezclar melodía preservada y generada según blend_melody
        if preserved_melody and generated:
            final_mel = _blend_melodies(preserved_melody, generated, blend_melody, target_key)
        elif preserved_melody:
            final_mel = preserved_melody
        else:
            final_mel = generated

        # Siembra del motivo del contenido
        if 'motif' in preserve_strats and original_melody and final_mel:
            final_mel = preserve_motif(original_melody, final_mel, target_key)

        # Ornamentación
        final_mel = add_ornamentation(final_mel, target_key, style_dna.style)

        # Humanización
        if groove and groove.trained:
            final_mel = humanize(final_mel, groove, bpb)

        # Acompañamiento
        acc = generate_accompaniment(
            mixed_dna.harmony_prog, target_key, n_bars, ec, fg, bpb,
            groove_map=groove,
            force_style=acc_style_forced,
            harmony_complexity=mixed_dna.harmony_complexity,
        )

        # Bajo
        bass = generate_bass(mixed_dna.harmony_prog, target_key, n_bars, bpb, groove)

        # Contrapunto
        cp = generate_counterpoint(final_mel, mixed_dna.harmony_prog, target_key, n_bars, bpb, ec)

        # Scoring
        sc = score_candidate(final_mel, acc, target_key)
        if verbose:
            print(f"    Candidato {cand_idx+1}/{n_candidates}: score={sc:.3f}")

        if sc > best_score:
            best_score = sc
            best_result = (final_mel, acc, bass, cp)

    mel, acc, bass, cp = best_result

    # Percusión desde el estilo
    perc = generate_percussion(
        style_dna.rhythm_grid,
        style_dna.rhythm_accent_grid,
        n_bars, bpb,
        groove_map=groove,
        style=style_dna.style,
    )

    return mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg


def _blend_melodies(mel_a, mel_b, alpha, key_obj):
    """
    Mezcla dos melodías nota a nota interpolando los pitches.
    alpha=1.0 → todo mel_a, alpha=0.0 → todo mel_b.
    Se alinean por offset temporal.
    """
    if alpha >= 0.99:
        return mel_a
    if alpha <= 0.01:
        return mel_b

    # Construir lookup de mel_b por offset
    n = min(len(mel_a), len(mel_b))
    result = []
    mel_b_sorted = sorted(mel_b, key=lambda x: x[0])

    for i, (oa, pa, da, va) in enumerate(sorted(mel_a, key=lambda x: x[0])[:n]):
        if i < len(mel_b_sorted):
            ob, pb, db, vb = mel_b_sorted[i]
            # Interpolar pitch (redondeado al semitono) y velocidad
            blended_pitch = int(round(pa * alpha + pb * (1 - alpha)))
            blended_pitch = _snap_to_scale(blended_pitch, key_obj)
            blended_vel   = int(va * alpha + vb * (1 - alpha))
            blended_dur   = da * alpha + db * (1 - alpha)
            result.append((oa, blended_pitch, blended_dur, blended_vel))
        else:
            result.append((oa, pa, da, va))

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='STYLE TRANSFER — Transferencia de estilo entre MIDIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('content', help='MIDI de contenido (tu idea melódica)')
    parser.add_argument('style',   help='MIDI de estilo (el ADN a transferir)')
    parser.add_argument('--strength',     type=float, default=0.7,
                        help='Intensidad global de transferencia 0-1 (default: 0.7)')
    parser.add_argument('--transfer',     nargs='+',
                        choices=['rhythm','harmony','emotion','texture','dynamics','key'],
                        default=None,
                        help='Dimensiones a transferir (default: todas)')
    parser.add_argument('--no-rhythm',    action='store_true', help='No transferir ritmo')
    parser.add_argument('--no-harmony',   action='store_true', help='No transferir armonía')
    parser.add_argument('--no-emotion',   action='store_true', help='No transferir emoción')
    parser.add_argument('--no-texture',   action='store_true', help='No transferir textura')
    parser.add_argument('--no-dynamics',  action='store_true', help='No transferir dinámica')
    parser.add_argument('--no-key',       action='store_true', help='No transferir tonalidad')
    parser.add_argument('--preserve',     nargs='+',
                        choices=['contour','intervals','motif','scale'],
                        default=None,
                        help='Estrategias de preservación melódica (default: todas)')
    parser.add_argument('--blend-melody', type=float, default=0.5,
                        help='Mezcla melodía original vs generada 0-1 (default: 0.5)')
    parser.add_argument('--bars',         type=int,   default=None,
                        help='Compases de salida (default: auto)')
    parser.add_argument('--candidates',   type=int,   default=3,
                        help='Candidatos a evaluar (default: 3)')
    parser.add_argument('--output',       default='transfer_out.mid')
    parser.add_argument('--export-fingerprint', action='store_true')
    parser.add_argument('--no-percussion', action='store_true',
                        help='No generar pista de percusión')
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--verbose',      action='store_true')
    args = parser.parse_args()

    for path in [args.content, args.style]:
        if not os.path.exists(path):
            print(f"ERROR: No encontrado: {path}")
            sys.exit(1)

    # Resolver dimensiones de transferencia
    ALL_DIMS = {'rhythm', 'harmony', 'emotion', 'texture', 'dynamics', 'key'}
    if args.transfer:
        transfer_dims = set(args.transfer)
    else:
        transfer_dims = set(ALL_DIMS)
    if args.no_rhythm:   transfer_dims.discard('rhythm')
    if args.no_harmony:  transfer_dims.discard('harmony')
    if args.no_emotion:  transfer_dims.discard('emotion')
    if args.no_texture:  transfer_dims.discard('texture')
    if args.no_dynamics: transfer_dims.discard('dynamics')
    if args.no_key:      transfer_dims.discard('key')

    preserve_strats = set(args.preserve) if args.preserve else {'contour', 'motif', 'scale'}

    print("═" * 65)
    print("  STYLE TRANSFER v1.0")
    print("═" * 65)
    print(f"  Contenido : {os.path.basename(args.content)}")
    print(f"  Estilo    : {os.path.basename(args.style)}")
    print(f"  Strength  : {args.strength:.2f}")
    print(f"  Transfer  : {', '.join(sorted(transfer_dims)) or '(ninguna)'}")
    print(f"  Preserve  : {', '.join(sorted(preserve_strats)) or '(ninguna)'}")
    print(f"  Blend mel.: {args.blend_melody:.2f}")

    # Extraer melodía del contenido
    print("\n[1/4] Extrayendo melodía del contenido…")
    original_melody, content_tempo, content_ts = extract_raw_melody(
        args.content, verbose=args.verbose)
    print(f"  ✓ {len(original_melody)} notas extraídas")

    # Extraer ADN del contenido y del estilo
    print("\n[2/4] Extrayendo ADN…")
    print(f"  ▶ Contenido: {os.path.basename(args.content)}")
    content_dna = UnifiedDNA(args.content)
    content_dna.extract(verbose=args.verbose)

    print(f"  ▶ Estilo   : {os.path.basename(args.style)}")
    style_dna = UnifiedDNA(args.style)
    style_dna.extract(verbose=args.verbose)

    # Determinar número de compases
    if args.bars:
        n_bars = args.bars
    else:
        bpb = (style_dna.time_sig[0] if 'rhythm' in transfer_dims
               else content_dna.time_sig[0])
        if original_melody:
            total_beats = max(o + d for o, _, d, _ in original_melody)
            n_bars = max(4, int(np.ceil(total_beats / bpb)))
        else:
            n_bars = 16
        # Redondear a múltiplo de 4
        n_bars = max(4, (n_bars // 4) * 4)

    print(f"\n[3/4] Transfiriendo estilo ({n_bars} compases, {args.candidates} candidatos)…")

    mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg = run_style_transfer(
        content_dna      = content_dna,
        style_dna        = style_dna,
        original_melody  = original_melody,
        strength         = args.strength,
        n_bars           = n_bars,
        transfer_dims    = transfer_dims,
        preserve_strats  = preserve_strats,
        blend_melody     = args.blend_melody,
        candidates       = args.candidates,
        seed             = args.seed,
        verbose          = args.verbose,
    )

    print(f"    → Melodía        : {len(mel)} notas")
    print(f"    → Acompañamiento : {len(acc)} eventos")
    print(f"    → Bajo           : {len(bass)} notas")
    print(f"    → Contrapunto    : {len(cp)} notas")
    if not args.no_percussion:
        print(f"    → Percusión      : {len(perc)} golpes")

    print(f"\n[4/4] Exportando → {args.output}")
    build_midi(mel, acc, bass, cp,
               target_key, tempo_bpm, time_sig, n_bars,
               form_gen=fg, output_path=args.output,
               percussion_notes=None if args.no_percussion else perc)

    if args.export_fingerprint:
        fp = dna_mod.extract_fingerprint(
            mel, bass, acc,
            target_key, tempo_bpm, n_bars, time_sig,
            fg, style_dna, args.output
        )
        json_path = dna_mod.export_fingerprint(fp, args.output)
        print(f"  → Fingerprint: {json_path}")

    print("\n" + "═" * 65)
    print(f"  Transferencia completada: {args.output}")
    print(f"  Tonalidad: {target_key.tonic.name} {target_key.mode}  |  {tempo_bpm:.0f} BPM")
    print("═" * 65)


if __name__ == '__main__':
    main()
