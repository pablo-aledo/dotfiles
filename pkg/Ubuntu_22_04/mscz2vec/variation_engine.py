"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    VARIATION ENGINE  v1.0                                    ║
║         Motor de variaciones musicales automáticas                           ║
║                                                                              ║
║  Dado un MIDI fuente (tu idea), genera un catálogo de variaciones            ║
║  clásicas y contemporáneas manteniendo la identidad temática.                ║
║                                                                              ║
║  VARIACIONES IMPLEMENTADAS:                                                  ║
║  [V01] Inversión          — refleja el contorno melódico                    ║
║  [V02] Retrógrado         — invierte el orden temporal                      ║
║  [V03] Inversión retrógrada — combina inversión + retrógrado                ║
║  [V04] Aumentación rítmica — duplica duraciones (va más lento)              ║
║  [V05] Diminución rítmica  — reduce duraciones a la mitad (más rápido)     ║
║  [V06] Ornamentación       — añade notas de paso, apoyaturas, mordentes     ║
║  [V07] Acompañamiento      — cambia el patrón de acompañamiento             ║
║  [V08] Transportada        — transpone a otra tonalidad                     ║
║  [V09] Modal               — cambia mayor ↔ menor (u otros modos)           ║
║  [V10] Rítmica             — reinterpreta con nuevo patrón rítmico          ║
║  [V11] Armónica            — reharmonización con progresión alternativa     ║
║  [V12] Textural            — cambia el número de voces y densidad           ║
║  [V13] Emocional           — aplica un arco emocional distinto              ║
║  [V14] Estocástica         — variación aleatoria controlada (por semilla)   ║
║  [V15] Contrapuntística    — añade contrapunto elaborado                    ║
║                                                                              ║
║  USO:                                                                        ║
║    python variation_engine.py fuente.mid                                    ║
║    python variation_engine.py fuente.mid --variations V01 V02 V04          ║
║    python variation_engine.py fuente.mid --all                              ║
║    python variation_engine.py fuente.mid --list                             ║
║    python variation_engine.py fuente.mid --bars 32 --output-dir ./vars      ║
║    python variation_engine.py fuente.mid --catalog --listen                 ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --variations V..  Lista de variaciones a generar (default: V01-V08)      ║
║    --all             Generar todas las variaciones disponibles              ║
║    --list            Listar variaciones disponibles y salir                  ║
║    --bars N          Compases por variación (default: auto)                  ║
║    --output-dir DIR  Directorio de salida (default: ./variations)           ║
║    --catalog         Generar un MIDI concatenado con todas las variaciones  ║
║    --listen          Reproducir las variaciones al final (requiere pygame)  ║
║    --play-seconds N  Segundos de reproducción por variación (default: 10)  ║
║    --seed N          Semilla aleatoria (default: 42)                         ║
║    --verbose         Informe detallado                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import random
import copy
import time
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
        build_midi, score_candidate,
        _snap_to_scale, _get_scale_midi, _get_scale_pcs,
        _get_relative_key, _get_dominant_key,
        _quarter_to_ticks, INSTRUMENT_RANGES,
    )
    from music21 import pitch as m21pitch, key as m21key, scale as m21scale
except ImportError as e:
    print(f"ERROR: {e}\nmidi_dna_unified.py no encontrado.")
    sys.exit(1)

TENSION_PRESETS = {
    'arch': lambda n: [
        0.1 + 0.85 * np.sin(np.pi * i / (n - 1))
        for i in range(n)
    ],
    'crescendo': lambda n: [
        0.05 + 0.9 * (i / (n - 1))
        for i in range(n)
    ],
    'decrescendo': lambda n: [
        0.95 - 0.9 * (i / (n - 1))
        for i in range(n)
    ],
    'plateau': lambda n: [
        0.2 + 0.7 * np.sin(np.pi * i / (n - 1)) ** 0.3
        for i in range(n)
    ],
    'late_climax': lambda n: [
        0.1 + 0.85 * np.sin(np.pi * i / (1.4 * (n - 1))) ** 2
        for i in range(n)
    ],
    'wave': lambda n: [
        0.4 + 0.5 * np.sin(2 * np.pi * i / (n - 1))
        for i in range(n)
    ],
    'neutral': lambda n: [0.5] * n,
    'dramatic': lambda n: [
        0.1 if i < n // 8 else
        0.1 + 0.85 * ((i - n//8) / (0.6 * (n - 1))) if i < int(0.7 * n) else
        0.95 - 0.8 * ((i - int(0.7 * n)) / (0.3 * (n - 1)))
        for i in range(n)
    ],
}
MT_PRESETS = TENSION_PRESETS

import mido


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTRO DE VARIACIONES
# ══════════════════════════════════════════════════════════════════════════════

VARIATIONS = {}  # id → (nombre, descripción, función)

def register_variation(var_id, name, description):
    """Decorador para registrar variaciones."""
    def decorator(fn):
        VARIATIONS[var_id] = (name, description, fn)
        return fn
    return decorator


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def extract_melody_from_midi(midi_path):
    """Extrae notas del MIDI como lista de (offset, pitch, dur, vel)."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4

    notes_by_channel = {}
    pending = {}

    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            ab = abs_ticks / tpb
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (ab, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                k_ = (msg.channel, msg.note)
                if k_ in pending:
                    onset, vel = pending.pop(k_)
                    dur = max(0.1, ab - onset)
                    if msg.channel != 9:
                        notes_by_channel.setdefault(msg.channel, []).append(
                            (onset, msg.note, dur, vel))

    tempo_bpm = 60_000_000 / max(tempo_us, 1)
    if not notes_by_channel:
        raise RuntimeError("No se encontraron notas en el MIDI")

    # Canal con pitch medio más alto = melodía
    def _mean_p(notes): return sum(p for _, p, _, _ in notes) / len(notes) if notes else 0
    mel_ch = max(notes_by_channel.keys(), key=lambda ch: _mean_p(notes_by_channel[ch]))
    melody = sorted(notes_by_channel[mel_ch], key=lambda x: x[0])
    return melody, tempo_bpm, (ts_num, ts_den)


def _write_variation(var_id, var_name, mel, acc, bass, cp, perc,
                      key_obj, tempo, time_sig, n_bars, fg, output_dir):
    """Escribe el MIDI de una variación y retorna la ruta."""
    safe_name = var_name.lower().replace(' ', '_').replace('/', '_')
    fname = f"{var_id}_{safe_name}.mid"
    out_path = os.path.join(output_dir, fname)
    try:
        build_midi(
            mel, acc, bass, cp,
            key_obj, tempo, time_sig, n_bars,
            form_gen=fg, output_path=out_path,
            percussion_notes=perc,
        )
        return out_path
    except Exception as e:
        print(f"    ⚠ Error escribiendo {out_path}: {e}")
        return None


def _build_controllers(dna, n_bars, tension_override=None):
    """Construye EmotionalController y FormGenerator desde un DNA."""
    tc = tension_override or dna.tension_curve or [0.5]
    ec = EmotionalController(
        tension_curve       = tc,
        arousal_curve       = dna.arousal_curve   or [0.0],
        valence_curve       = dna.valence_curve   or [0.0],
        stability_curve     = dna.stability_curve or [0.7],
        activity_curve      = dna.activity_curve  or [0.5],
        emotional_arc_label = dna.emotional_arc_label,
        n_bars              = n_bars,
    )
    fg = FormGenerator(
        form_string       = dna.form_string,
        section_map       = dna.section_map,
        phrase_lengths    = dna.phrase_lengths,
        cadence_positions = dna.cadence_positions,
        n_bars_out        = n_bars,
    )
    return ec, fg


def _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, mel,
                            force_acc_style=None, groove=None):
    """Genera acompañamiento, bajo, contrapunto y percusión estándar."""
    bpb = dna.time_sig[0]
    acc = generate_accompaniment(
        dna.harmony_prog, key_obj, n_bars, ec, fg, bpb,
        groove_map=groove, force_style=force_acc_style,
        harmony_complexity=dna.harmony_complexity,
    )
    bass = generate_bass(dna.harmony_prog, key_obj, n_bars, bpb, groove)
    cp   = generate_counterpoint(mel, dna.harmony_prog, key_obj, n_bars, bpb, ec)
    perc = generate_percussion(
        dna.rhythm_grid, dna.rhythm_accent_grid, n_bars, bpb,
        groove_map=groove, style=dna.style
    )
    return acc, bass, cp, perc


# ══════════════════════════════════════════════════════════════════════════════
#  VARIACIONES
# ══════════════════════════════════════════════════════════════════════════════

@register_variation('V01', 'Inversión', 'Refleja el contorno melódico (los ascensos se vuelven descensos y viceversa)')
def variation_inversion(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    # Invertir la melodía respecto a la nota central
    pitches = [p for _, p, _, _ in melody]
    center = sum(pitches) / len(pitches) if pitches else 60
    inv_melody = [
        (o, _snap_to_scale(int(2 * center - p), key_obj), d, v)
        for o, p, d, v in melody
    ]
    # Escalar temporalmente a n_bars
    inv_melody = _scale_melody_to_bars(inv_melody, n_bars, bpb)

    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, inv_melody)
    inv_melody = add_ornamentation(inv_melody, key_obj, dna.style)
    return inv_melody, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V02', 'Retrógrado', 'Invierte el orden temporal de las notas')
def variation_retrograde(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    if not melody:
        return _empty_variation(dna, n_bars)

    # Reordenar notas en orden inverso, manteniendo sus duraciones
    total_dur = sum(d for _, _, d, _ in melody)
    retro = []
    cursor = 0.0
    for (_, p, d, v) in reversed(melody):
        retro.append((cursor, p, d, v))
        cursor += d

    retro = _scale_melody_to_bars(retro, n_bars, bpb)
    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, retro)
    retro = add_ornamentation(retro, key_obj, dna.style)
    return retro, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V03', 'Inversión retrógrada', 'Combina inversión y retrógrado simultáneamente')
def variation_retrograde_inversion(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    pitches = [p for _, p, _, _ in melody]
    center = sum(pitches) / len(pitches) if pitches else 60

    # Invertir
    inv = [(o, int(2 * center - p), d, v) for o, p, d, v in melody]
    # Retrógrado
    cursor = 0.0
    retro_inv = []
    for (_, p, d, v) in reversed(inv):
        snapped = _snap_to_scale(p, key_obj)
        retro_inv.append((cursor, snapped, d, v))
        cursor += d

    retro_inv = _scale_melody_to_bars(retro_inv, n_bars, bpb)
    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, retro_inv)
    return retro_inv, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V04', 'Aumentación rítmica', 'Duplica las duraciones (tempo efectivo a la mitad)')
def variation_augmentation(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    augmented = []
    cursor = 0.0
    for (_, p, d, v) in melody:
        augmented.append((cursor, p, d * 2.0, v))
        cursor += d * 2.0

    augmented = _scale_melody_to_bars(augmented, n_bars, bpb)
    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, augmented)
    return augmented, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V05', 'Diminución rítmica', 'Reduce duraciones a la mitad (más rápido y enérgico)')
def variation_diminution(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    diminished = []
    cursor = 0.0
    for (_, p, d, v) in melody:
        diminished.append((cursor, p, max(0.1, d * 0.5), v))
        cursor += d * 0.5

    diminished = _scale_melody_to_bars(diminished, n_bars, bpb)
    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, diminished,
                                                  force_acc_style='alberti')
    return diminished, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V06', 'Ornamentación', 'Añade notas de paso, apoyaturas y mordentes elaborados')
def variation_ornamentation(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    base = _scale_melody_to_bars(melody, n_bars, bpb)
    # Ornamentación agresiva cambiando el estilo a 'baroque'
    ornamented = add_ornamentation(base, key_obj, 'baroque')
    # Segunda pasada de ornamentación para enriquecer
    ornamented = add_ornamentation(ornamented, key_obj, 'romantic')

    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, ornamented)
    return ornamented, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V07', 'Acompañamiento alternativo', 'Reinterpreta con diferente patrón de acompañamiento')
def variation_accompaniment(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    base = _scale_melody_to_bars(melody, n_bars, bpb)
    ec, fg = _build_controllers(dna, n_bars)

    # Rotar el estilo de acompañamiento
    styles = ['alberti', 'arpeggio', 'block', 'waltz']
    current_style = getattr(dna, 'style', 'generic')
    style_map = {'baroque': 'alberti', 'classical': 'arpeggio',
                 'romantic': 'block', 'jazz': 'block', 'generic': 'arpeggio'}
    default = style_map.get(current_style, 'arpeggio')
    # Elegir el siguiente en la lista
    idx = styles.index(default) if default in styles else 0
    new_style = styles[(idx + 1) % len(styles)]

    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, base,
                                                  force_acc_style=new_style)
    return base, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V08', 'Transportada', 'Transpone a la tonalidad relativa o dominante')
def variation_transposed(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj

    # Alternar entre relativo y dominante según la semilla
    if seed % 2 == 0:
        new_key = _get_relative_key(key_obj)
        label = "relativa"
    else:
        new_key = _get_dominant_key(key_obj)
        label = "dominante"

    src_pc = m21pitch.Pitch(key_obj.tonic.name).pitchClass
    tgt_pc = m21pitch.Pitch(new_key.tonic.name).pitchClass
    shift = (tgt_pc - src_pc)
    if shift > 6:  shift -= 12
    if shift < -6: shift += 12

    bpb = dna.time_sig[0]
    transposed = [
        (o, _snap_to_scale(max(40, min(96, p + shift)), new_key), d, v)
        for o, p, d, v in melody
    ]
    transposed = _scale_melody_to_bars(transposed, n_bars, bpb)

    # Modificar el DNA para usar la nueva tonalidad
    dna_t = copy.deepcopy(dna)
    dna_t.key_obj = new_key
    ec, fg = _build_controllers(dna_t, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna_t, new_key, n_bars, ec, fg, transposed)
    transposed = add_ornamentation(transposed, new_key, dna.style)
    return transposed, acc, bass, cp, perc, new_key, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V09', 'Modal', 'Reinterpreta en modo paralelo (mayor↔menor) o modo dórico/frigio')
def variation_modal(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    # Elegir el modo destino
    tonic_name = key_obj.tonic.name
    options = {
        'major': ['minor', 'dorian', 'phrygian'],
        'minor': ['major', 'dorian', 'mixolydian'],
    }
    current_mode = key_obj.mode
    mode_choices = options.get(current_mode, ['minor'])
    new_mode = mode_choices[seed % len(mode_choices)]

    try:
        new_key = m21key.Key(tonic_name, new_mode)
    except Exception:
        new_key = _get_relative_key(key_obj)

    # Re-snap las notas a la nueva escala
    base = _scale_melody_to_bars(melody, n_bars, bpb)
    modal_mel = [
        (o, _snap_to_scale(p, new_key), d, v)
        for o, p, d, v in base
    ]

    dna_m = copy.deepcopy(dna)
    dna_m.key_obj = new_key
    ec, fg = _build_controllers(dna_m, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna_m, new_key, n_bars, ec, fg, modal_mel)
    return modal_mel, acc, bass, cp, perc, new_key, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V10', 'Rítmica', 'Reinterpreta el material melódico con un patrón rítmico completamente distinto')
def variation_rhythmic(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    pitches = [p for _, p, _, _ in melody]
    vels    = [v for _, _, _, v in melody]

    # Generar nuevo patrón rítmico: valores de nota alternos
    durations_new = []
    patterns = [
        [1.0, 0.5, 0.5, 1.0, 1.0],  # negra, corchea, corchea, negra, negra
        [0.5, 0.5, 0.5, 0.5, 1.0, 1.0],  # corcheas + negras
        [0.25, 0.25, 0.5, 1.0, 0.5, 0.5],  # semicorcheas + combinado
        [2.0, 0.5, 0.5, 1.0],  # blanca + corcheas
    ]
    chosen_pattern = patterns[seed % len(patterns)]

    cursor = 0.0
    rhythmic_mel = []
    p_idx = 0
    pat_idx = 0
    total_beats = n_bars * bpb
    while cursor < total_beats - 0.05 and p_idx < len(pitches):
        d = chosen_pattern[pat_idx % len(chosen_pattern)]
        d = min(d, total_beats - cursor)
        p = _snap_to_scale(pitches[p_idx % len(pitches)], key_obj)
        v = vels[p_idx % len(vels)]
        rhythmic_mel.append((cursor, p, d, v))
        cursor += d
        p_idx += 1
        pat_idx += 1

    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, rhythmic_mel)
    return rhythmic_mel, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V11', 'Armónica', 'Reharmonización con progresión alternativa manteniendo la melodía')
def variation_harmonic(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    base = _scale_melody_to_bars(melody, n_bars, bpb)

    # Progresiones alternativas predefinidas
    bpb_f = float(bpb)
    reharmonizations = [
        [('I', bpb_f), ('vi', bpb_f), ('IV', bpb_f), ('V', bpb_f)],      # I-VI-IV-V
        [('I', bpb_f), ('V', bpb_f), ('vi', bpb_f), ('IV', bpb_f)],      # I-V-VI-IV
        [('i', bpb_f), ('VII', bpb_f), ('VI', bpb_f), ('VII', bpb_f)],   # i-VII-VI-VII
        [('I', bpb_f/2), ('IV', bpb_f/2), ('I', bpb_f/2), ('V', bpb_f/2),
         ('I', bpb_f), ('IV', bpb_f), ('V', bpb_f), ('I', bpb_f)],        # clásica extendida
        [('ii', bpb_f), ('V', bpb_f), ('I', bpb_f), ('vi', bpb_f)],      # jazz ii-V-I
    ]
    new_prog = reharmonizations[seed % len(reharmonizations)]
    # Expandir para cubrir n_bars
    total = sum(d for _, d in new_prog)
    reps = max(1, math.ceil(n_bars * bpb / total))
    full_prog = (new_prog * reps)[:n_bars * bpb]

    dna_h = copy.deepcopy(dna)
    dna_h.harmony_prog = full_prog
    dna_h.harmony_functions = [f for f, _ in full_prog]

    ec, fg = _build_controllers(dna_h, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna_h, key_obj, n_bars, ec, fg, base)
    return base, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V12', 'Textural', 'Cambia la densidad de voces: de monódica a rica en texturas')
def variation_textural(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    base = _scale_melody_to_bars(melody, n_bars, bpb)

    # Aumentar densidad de voces con contrapunto elaborado
    ec, fg = _build_controllers(dna, n_bars)
    groove = dna.groove_map if dna.groove_map.trained else None

    # Acompañamiento bloque (más denso)
    acc = generate_accompaniment(
        dna.harmony_prog, key_obj, n_bars, ec, fg, bpb,
        groove_map=groove, force_style='block',
        harmony_complexity=min(1.0, dna.harmony_complexity + 0.3),
    )
    bass = generate_bass(dna.harmony_prog, key_obj, n_bars, bpb, groove)
    # Contrapunto más elaborado
    cp = generate_counterpoint(base, dna.harmony_prog, key_obj, n_bars, bpb, ec)
    # Añadir ornamentación al contrapunto
    cp = add_ornamentation(cp, key_obj, 'romantic')
    perc = generate_percussion(dna.rhythm_grid, dna.rhythm_accent_grid, n_bars, bpb,
                               groove_map=groove, style=dna.style)
    return base, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V13', 'Emocional', 'Aplica un arco emocional opuesto (lullaby vs awakening, crescendo vs decrescendo)')
def variation_emotional(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    base = _scale_melody_to_bars(melody, n_bars, bpb)

    # Invertir la curva de tensión
    orig_tension = dna.tension_curve or [0.5] * n_bars
    # Interpolar a n_bars
    arr = np.array(orig_tension)
    xs = np.linspace(0, len(arr) - 1, n_bars)
    interp = np.interp(xs, np.arange(len(arr)), arr)
    inverted_tension = (1.0 - interp).tolist()

    dna_e = copy.deepcopy(dna)
    dna_e.tension_curve = inverted_tension

    # Clasificar el nuevo arco
    arr2 = np.array(inverted_tension)
    if arr2[-1] > arr2[0] + 0.2:
        dna_e.emotional_arc_label = 'crescendo'
    elif arr2[0] > arr2[-1] + 0.2:
        dna_e.emotional_arc_label = 'decrescendo'
    else:
        dna_e.emotional_arc_label = 'arch'

    ec, fg = _build_controllers(dna_e, n_bars, tension_override=inverted_tension)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna_e, key_obj, n_bars, ec, fg, base)
    return base, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V14', 'Estocástica', 'Variación aleatoria controlada por semilla: única cada vez')
def variation_stochastic(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    base = _scale_melody_to_bars(melody, n_bars, bpb)

    # Mutaciones aleatorias controladas
    stochastic_mel = []
    for o, p, d, v in base:
        new_p = p
        new_d = d
        new_v = v

        # Mutación de pitch con probabilidad 20%
        if random.random() < 0.20:
            shift = random.choice([-2, -1, 1, 2])
            new_p = _snap_to_scale(p + shift, key_obj)
            new_p = max(48, min(84, new_p))

        # Mutación de duración con probabilidad 15%
        if random.random() < 0.15:
            scale = random.choice([0.5, 0.75, 1.5, 2.0])
            new_d = max(0.1, d * scale)

        # Mutación de velocidad con probabilidad 25%
        if random.random() < 0.25:
            new_v = max(30, min(110, v + random.randint(-20, 20)))

        stochastic_mel.append((o, new_p, new_d, new_v))

    # Re-normalizar offsets
    cursor = 0.0
    normalized = []
    for _, p, d, v in stochastic_mel:
        normalized.append((cursor, p, d, v))
        cursor += d
    normalized = _scale_melody_to_bars(normalized, n_bars, bpb)

    ec, fg = _build_controllers(dna, n_bars)
    acc, bass, cp, perc = _std_acc_bass_cp_perc(dna, key_obj, n_bars, ec, fg, normalized)
    return normalized, acc, bass, cp, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


@register_variation('V15', 'Contrapuntística', 'Eleva el contrapunto a voz principal, la melodía pasa a segundo plano')
def variation_counterpoint(dna, melody, n_bars, seed=42):
    random.seed(seed); np.random.seed(seed)
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]

    base = _scale_melody_to_bars(melody, n_bars, bpb)
    ec, fg = _build_controllers(dna, n_bars)
    groove = dna.groove_map if dna.groove_map.trained else None

    # Generar contrapunto elaborado y usarlo como melodía
    cp_as_mel = generate_counterpoint(base, dna.harmony_prog, key_obj, n_bars, bpb, ec)
    cp_as_mel = add_ornamentation(cp_as_mel, key_obj, 'baroque')
    if groove and groove.trained:
        cp_as_mel = dna_mod.humanize(cp_as_mel, groove, bpb)

    # La melodía original pasa a ser el acompañamiento interior
    base_as_inner = [(o, max(36, p - 12), d, max(20, v - 15)) for o, p, d, v in base]

    acc = generate_accompaniment(
        dna.harmony_prog, key_obj, n_bars, ec, fg, bpb,
        groove_map=groove, force_style='arpeggio',
        harmony_complexity=dna.harmony_complexity,
    )
    bass = generate_bass(dna.harmony_prog, key_obj, n_bars, bpb, groove)
    perc = generate_percussion(dna.rhythm_grid, dna.rhythm_accent_grid, n_bars, bpb,
                               groove_map=groove, style=dna.style)

    # cp_as_mel es la melodía, base_as_inner reemplaza al contrapunto
    return cp_as_mel, acc, bass, base_as_inner, perc, key_obj, dna.tempo_bpm, dna.time_sig, fg


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE ESCALADO
# ══════════════════════════════════════════════════════════════════════════════

def _scale_melody_to_bars(melody, n_bars, bpb):
    """Escala temporalmente la melodía para que llene exactamente n_bars compases."""
    if not melody:
        return melody
    total_beats_target = n_bars * bpb
    total_beats_src = max(o + d for o, _, d, _ in melody)
    if total_beats_src <= 0:
        return melody
    scale = total_beats_target / total_beats_src
    return [(o * scale, p, d * scale, v) for o, p, d, v in melody]


def _empty_variation(dna, n_bars):
    """Retorna una variación vacía para casos de error."""
    key_obj = dna.key_obj
    bpb = dna.time_sig[0]
    ec, fg = _build_controllers(dna, n_bars)
    # Melodía mínima: tónica sostenida
    tonic = m21pitch.Pitch(key_obj.tonic.name).midi + 60
    mel = [(i * bpb, tonic, bpb * 0.9, 64) for i in range(n_bars)]
    acc = generate_accompaniment(dna.harmony_prog, key_obj, n_bars, ec, fg, bpb)
    bass = generate_bass(dna.harmony_prog, key_obj, n_bars, bpb)
    return mel, acc, bass, [], [], key_obj, dna.tempo_bpm, dna.time_sig, fg


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO: concatenar todas las variaciones en un solo MIDI
# ══════════════════════════════════════════════════════════════════════════════

def build_catalog_midi(variation_results, output_path, tempo_bpm, time_sig):
    """
    Concatena todas las variaciones en un único MIDI con marcadores.
    variation_results: lista de (var_id, var_name, mel, acc, bass, cp, perc, key_obj, fg, n_bars)
    """
    TICKS = 480
    bpb, bu = time_sig
    us_per_beat = int(60_000_000 / max(tempo_bpm, 1))

    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS)
    mel_trk  = mido.MidiTrack(); mel_trk.name  = 'Melody'
    acc_trk  = mido.MidiTrack(); acc_trk.name  = 'Accompaniment'
    bass_trk = mido.MidiTrack(); bass_trk.name = 'Bass'
    cp_trk   = mido.MidiTrack(); cp_trk.name   = 'Counterpoint'

    for trk, prog in [(mel_trk, 0), (acc_trk, 0), (bass_trk, 32), (cp_trk, 0)]:
        trk.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
        trk.append(mido.MetaMessage('time_signature',
                                     numerator=bpb, denominator=bu,
                                     clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        trk.append(mido.Message('program_change', channel=0, program=prog, time=0))

    mid.tracks += [mel_trk, acc_trk, bass_trk, cp_trk]

    def _add_notes_to_track(track, notes_list, ch, offset_bars, n_bars_section):
        beats_offset = offset_bars * bpb
        events = []
        for o, p, d, v in notes_list:
            p   = max(0, min(127, int(p)))
            v   = max(1, min(127, int(v)))
            d   = max(0.05, float(d))
            t_on  = _quarter_to_ticks(float(o) + beats_offset, TICKS)
            t_off = _quarter_to_ticks(float(o) + beats_offset + d, TICKS)
            events.append((t_on,  'on',  p, v))
            events.append((t_off, 'off', p, 0))
        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
        prev = getattr(track, '_catalog_last_tick', 0)
        for tick, kind, note, vel in events:
            delta = max(0, tick - prev)
            prev  = tick
            msg_type = 'note_on' if kind == 'on' else 'note_off'
            track.append(mido.Message(msg_type, channel=ch, note=note, velocity=vel, time=delta))
        track._catalog_last_tick = prev

    # Añadir silencio de separación entre secciones (1 compás)
    GAP = 1
    cursor_bar = 0
    for var_id, var_name, mel, acc, bass, cp, perc, key_obj, fg, n_bars in variation_results:
        # Marcador
        marker_tick = _quarter_to_ticks(cursor_bar * bpb, TICKS)
        for trk in [mel_trk]:
            prev = getattr(trk, '_catalog_last_tick', 0)
            delta = max(0, marker_tick - prev)
            trk.append(mido.MetaMessage('marker', text=f"{var_id}: {var_name}", time=delta))
            trk._catalog_last_tick = marker_tick

        _add_notes_to_track(mel_trk,  mel,  0, cursor_bar, n_bars)
        _add_notes_to_track(acc_trk,  acc,  2, cursor_bar, n_bars)
        _add_notes_to_track(bass_trk, bass, 3, cursor_bar, n_bars)
        _add_notes_to_track(cp_trk,   cp,   1, cursor_bar, n_bars)
        cursor_bar += n_bars + GAP

    for trk in [mel_trk, acc_trk, bass_trk, cp_trk]:
        trk.append(mido.MetaMessage('end_of_track', time=0))

    mid.save(output_path)
    print(f"  → Catálogo MIDI: {output_path}  ({cursor_bar} compases totales)")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='VARIATION ENGINE — Motor de variaciones musicales automáticas',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', nargs='?', help='MIDI fuente')
    parser.add_argument('--variations', nargs='+', metavar='VXX',
                        help='Variaciones a generar (ej: V01 V02 V04)')
    parser.add_argument('--all',        action='store_true', help='Generar todas las variaciones')
    parser.add_argument('--list',       action='store_true', help='Listar variaciones disponibles')
    parser.add_argument('--bars',       type=int, default=None, help='Compases por variación')
    parser.add_argument('--output-dir', default='./variations', help='Directorio de salida')
    parser.add_argument('--catalog',    action='store_true',
                        help='Generar MIDI catálogo con todas las variaciones concatenadas')
    parser.add_argument('--listen',     action='store_true',
                        help='Reproducir las variaciones al final (requiere pygame)')
    parser.add_argument('--play-seconds', type=int, default=10)
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--no-percussion', action='store_true',
                        help='Desactivar generación de percusión en todas las variaciones')
    parser.add_argument('--verbose',    action='store_true')
    args = parser.parse_args()

    # Listar variaciones
    if args.list:
        print("\n  VARIACIONES DISPONIBLES:")
        print("  " + "─" * 55)
        for var_id in sorted(VARIATIONS):
            name, desc, _ = VARIATIONS[var_id]
            print(f"  [{var_id}] {name:<25} {desc[:45]}")
        print()
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"ERROR: No encontrado: {args.input}")
        sys.exit(1)

    # Seleccionar variaciones a generar
    if args.all:
        selected_ids = sorted(VARIATIONS.keys())
    elif args.variations:
        selected_ids = [v.upper() for v in args.variations if v.upper() in VARIATIONS]
        invalid = [v for v in args.variations if v.upper() not in VARIATIONS]
        if invalid:
            print(f"  ⚠ Variaciones desconocidas: {invalid}")
    else:
        # Default: las 8 primeras
        selected_ids = sorted(VARIATIONS.keys())[:8]

    if not selected_ids:
        print("ERROR: No se seleccionó ninguna variación válida.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("═" * 65)
    print("  VARIATION ENGINE v1.0")
    print("═" * 65)
    print(f"  Fuente       : {os.path.basename(args.input)}")
    print(f"  Variaciones  : {', '.join(selected_ids)}")
    print(f"  Directorio   : {args.output_dir}")

    # Extraer melodía
    print("\n[1/3] Extrayendo melodía fuente…")
    try:
        melody, tempo_bpm, time_sig = extract_melody_from_midi(args.input)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    print(f"  ✓ {len(melody)} notas  |  {tempo_bpm:.0f} BPM  |  {time_sig[0]}/{time_sig[1]}")

    # Extraer ADN
    print("\n[2/3] Extrayendo ADN musical…")
    dna = UnifiedDNA(args.input)
    if not dna.extract(verbose=args.verbose):
        print("ERROR: No se pudo extraer el ADN del MIDI.")
        sys.exit(1)
    print(f"  ✓ ADN extraído: {dna.key_obj.tonic.name} {dna.key_obj.mode}  |  "
          f"{dna.form_string}  |  {dna.style}")

    # Determinar número de compases
    bpb = time_sig[0]
    if args.bars:
        n_bars = args.bars
    else:
        total_beats = max(o + d for o, _, d, _ in melody)
        n_bars = max(4, int(np.ceil(total_beats / bpb)))
        n_bars = (n_bars // 4) * 4  # redondear a múltiplo de 4

    print(f"  Compases por variación: {n_bars}")

    # Generar variaciones
    print(f"\n[3/3] Generando {len(selected_ids)} variaciones…")
    generated_paths = []
    catalog_data = []

    for var_id in selected_ids:
        name, desc, fn = VARIATIONS[var_id]
        print(f"\n  [{var_id}] {name}")
        print(f"        {desc}")
        try:
            result = fn(dna, melody, n_bars, seed=args.seed)
            mel, acc, bass, cp, perc, key_obj, tempo, ts, fg = result
            if args.no_percussion:
                perc = []
            out_path = _write_variation(
                var_id, name, mel, acc, bass, cp, perc,
                key_obj, tempo, ts, n_bars, fg, args.output_dir
            )
            if out_path:
                sc = score_candidate(mel, acc, key_obj)
                print(f"        → {os.path.basename(out_path)}  [score: {sc:.3f}]")
                generated_paths.append(out_path)
                catalog_data.append((var_id, name, mel, acc, bass, cp, perc, key_obj, fg, n_bars))
            else:
                print("        ✗ Error al escribir el MIDI")
        except Exception as e:
            print(f"        ✗ Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # Catálogo
    if args.catalog and catalog_data:
        cat_path = os.path.join(args.output_dir, '00_catalog_all_variations.mid')
        build_catalog_midi(catalog_data, cat_path, tempo_bpm, time_sig)
        generated_paths.insert(0, cat_path)

    # Guardar índice JSON
    index = {
        'source': os.path.abspath(args.input),
        'n_bars': n_bars,
        'tempo_bpm': tempo_bpm,
        'key': f"{dna.key_obj.tonic.name} {dna.key_obj.mode}",
        'variations': [
            {'id': var_id, 'name': VARIATIONS[var_id][0],
             'description': VARIATIONS[var_id][1]}
            for var_id in selected_ids
            if var_id in VARIATIONS
        ]
    }
    idx_path = os.path.join(args.output_dir, 'index.json')
    with open(idx_path, 'w') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\n  ═══ RESUMEN ═══")
    print(f"  Variaciones generadas : {len(generated_paths)}")
    print(f"  Directorio            : {args.output_dir}")
    print(f"  Índice                : {idx_path}")

    # Reproducir
    if args.listen and generated_paths:
        try:
            import pygame
            pygame.init()
            pygame.mixer.init()
            print("\n  ▶ Reproduciendo variaciones…")
            for path in generated_paths:
                print(f"    → {os.path.basename(path)}")
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                t0 = time.time()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    if time.time() - t0 >= args.play_seconds:
                        pygame.mixer.music.stop()
                        break
                time.sleep(0.5)
            pygame.quit()
        except Exception as e:
            print(f"  ⚠ No se pudo reproducir: {e}")


if __name__ == '__main__':
    main()
