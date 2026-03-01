"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_REPR2  v1.0  —  Representación piano roll exacta MIDI ↔ Tensor        ║
║                                                                              ║
║  A diferencia de lm_repr.py (estadísticas agregadas), esta representación  ║
║  guarda qué nota exacta suena en cada frame → reconstrucción fiel posible. ║
║                                                                              ║
║  ESTRUCTURA DEL TENSOR [T, F*P]:                                            ║
║    T = frames de 125ms (8 fps — resolución suficiente para melodía)        ║
║    F = 5 familias (keys, strings, bass, winds, percussion)                 ║
║    P = 72 pitches (MIDI 24–95, Do1–Si6)                                   ║
║    Valor = velocity/127 ∈ [0,1] si la nota suena, 0 si no.                ║
║                                                                              ║
║  Memoria: 256 × 360 = 92 KB por pieza, 715 piezas = 66 MB               ║
║  Resolución: 125ms por frame (2 fps del anterior → 8 fps ahora)           ║
║                                                                              ║
║  USO:                                                                        ║
║    python lm_repr2.py archivo.mid --info                                    ║
║    python lm_repr2.py archivo.mid --roundtrip --output rec.mid             ║
║    python lm_repr2.py *.mid --batch-check                                   ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install mido numpy                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import argparse
import numpy as np
from collections import defaultdict

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

# Importar utilidades de lm_repr (detección de familia, rangos, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from lm_repr import (
        midi_to_note_events, NoteEvent,
        FAMILIES, N_FAMILIES, FAMILY_GM, FAMILY_RANGES,
        INSTRUMENT_RANGES, INSTR_TO_FAMILY, INSTR_PROGRAM,
        DEFAULT_BPM, TICKS_OUT, PERC_NOTES_GM,
    )
except ImportError as e:
    print(f"ERROR: lm_repr.py debe estar en el mismo directorio. ({e})")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DEL PIANO ROLL
# ══════════════════════════════════════════════════════════════════════════════

FRAME_MS2    = 125            # 8 fps — resolución melódica
PITCH_LO     = 24             # Do1  — MIDI note 24
PITCH_HI     = 95             # Si6  — MIDI note 95
N_PITCHES    = PITCH_HI - PITCH_LO + 1    # 72
TENSOR_DIMS2 = N_FAMILIES * N_PITCHES     # 360

# Índice en el tensor para (familia, pitch)
def fam_pitch_idx(fam_idx: int, pitch: int) -> int:
    """Índice en el tensor [T, 360] para familia fam_idx y pitch MIDI."""
    p = int(np.clip(pitch - PITCH_LO, 0, N_PITCHES - 1))
    return fam_idx * N_PITCHES + p

# Percusión GM: usamos pitches fijos del mapa GM, proyectados a rango [35,81]
# Los mapeamos al bloque de percusión (fam_idx=4) con su pitch real
PERC_PITCHES = sorted([p for p in PERC_NOTES_GM if PITCH_LO <= p <= PITCH_HI])


# ══════════════════════════════════════════════════════════════════════════════
#  MIDI → TENSOR PIANO ROLL [T, 360]
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_roll(file_path: str,
                 frame_ms: int = FRAME_MS2,
                 max_frames: int = 256,
                 ) -> tuple[np.ndarray | None, float]:
    """
    Convierte un fichero MIDI en tensor piano roll [T, F*P].
    
    Retorna (tensor [max_frames, 360] float32, bpm)
    o (None, 0) si el fichero falla.
    
    Cada valor es velocity/127 ∈ [0,1] si la nota suena en ese frame,
    0 si no suena. Las notas que duran múltiples frames aparecen en todos.
    """
    try:
        events, total_sec, bpm = midi_to_note_events(file_path)
    except Exception as e:
        return None, 0.0

    if not events or total_sec <= 0:
        return None, 0.0

    frame_sec = frame_ms / 1000.0
    n_frames  = max(1, math.ceil(total_sec / frame_sec))

    # Tensor completo (puede ser más largo que max_frames)
    roll = np.zeros((n_frames, TENSOR_DIMS2), dtype=np.float32)

    fam_to_idx = {f: i for i, f in enumerate(FAMILIES)}

    for ev in events:
        fam_idx  = fam_to_idx.get(ev.family, 0)
        pitch    = int(ev.pitch)

        # Ignorar pitches fuera de rango
        if not (PITCH_LO <= pitch <= PITCH_HI):
            continue

        vel_norm = float(ev.velocity) / 127.0
        col      = fam_pitch_idx(fam_idx, pitch)

        f_start = int(ev.time_sec / frame_sec)
        # Duración mínima: 1 frame; máxima: hasta el final de la nota
        f_end   = max(f_start + 1,
                      int((ev.time_sec + ev.duration_sec) / frame_sec))

        for f in range(f_start, min(f_end, n_frames)):
            # Tomar el máximo si hay superposición de notas en el mismo pitch
            roll[f, col] = max(roll[f, col], vel_norm)

    # Recortar o rellenar a max_frames
    if n_frames >= max_frames:
        roll = roll[:max_frames]
    else:
        pad  = np.zeros((max_frames - n_frames, TENSOR_DIMS2), dtype=np.float32)
        roll = np.concatenate([roll, pad], axis=0)

    return roll, bpm


# ══════════════════════════════════════════════════════════════════════════════
#  TENSOR PIANO ROLL → MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _segment_pitch(sig: np.ndarray,
                   min_peak: float = 0.30,
                   valley_ratio: float = 0.40,
                   max_frames: int = 24) -> list[tuple[int, int, float]]:
    """
    Segmenta la activación 1-D de un pitch en notas discretas.

    Diseñado para outputs de VAE: la señal nunca llega a cero entre notas,
    sino que forma colinas con valles parciales.

    Algoritmo de avance lineal estricto (sin backtrack, sin solapamiento):
      1. Ignorar todo lo que esté bajo min_peak (fondo del decoder).
      2. Al encontrar min_peak, buscar el pico local avanzando mientras sube.
      3. Definir cutoff = pico * valley_ratio.
      4. Avanzar hasta que la señal caiga bajo cutoff o se alcance max_frames.
      5. Emitir nota y continuar desde el fin (sin solapamiento garantizado).

    min_peak:     vel mínima para considerar que hay nota real (ignora fondo ~0.08)
    valley_ratio: fracción del pico bajo la cual se considera fin de nota
    max_frames:   duración máxima en frames (hard cap para evitar notas eternas)

    Retorna lista de (frame_on, frame_off, peak_vel).
    """
    T = len(sig)
    notes = []
    f = 0

    while f < T:
        # Saltar fondo (activación residual del decoder)
        if sig[f] < min_peak:
            f += 1
            continue

        # Encontrar el pico local: avanzar mientras la señal sube o se mantiene
        note_start = f
        peak_val   = sig[f]
        f += 1
        while f < T and f - note_start < max_frames:
            v = sig[f]
            if v > peak_val:
                peak_val = v
                f += 1
            elif v >= peak_val * 0.90:   # plateau — mismo pico
                f += 1
            else:
                break   # empieza a descender: hemos pasado el pico

        # Verificación: el pico real supera min_peak (puede haberse colado ruido)
        if peak_val < min_peak:
            continue

        # Cutoff dinámico basado en el pico real
        cutoff = peak_val * valley_ratio

        # Avanzar hasta el fin de la nota (valle o max_frames)
        note_end = f
        while note_end < T and sig[note_end] >= cutoff:
            note_end += 1
            if note_end - note_start >= max_frames:
                break

        notes.append((note_start, note_end, peak_val))
        f = note_end   # continuar justo después — sin solapamiento

    return notes


def roll_to_midi(roll: np.ndarray,
                 bpm: float,
                 instruments: list[dict] | None = None,
                 output_path: str = 'output.mid',
                 frame_ms: int = FRAME_MS2,
                 min_peak: float = 0.30,
                 valley_ratio: float = 0.40,
                 max_note_frames: int = 24,
                 ) -> str:
    """
    Convierte tensor piano roll [T, F*P] a fichero MIDI.

    Utiliza _segment_pitch() para separar las notas del output del VAE,
    que produce activaciones continuas (nunca exactamente cero) en lugar
    del piano roll original con ceros limpios.

    min_peak:       vel mínima para considerar nota real (default 0.30).
                    Aumentar si hay demasiadas notas fantasma.
                    Reducir si faltan notas reales.
    valley_ratio:   fracción del pico que define el fin de la nota (default 0.40).
                    Reducir para notas más largas; aumentar para más cortas.
    max_note_frames: duración máxima en frames (default 24 = 3s a 8fps).

    instruments: lista de dicts {'name', 'role'} compatible con orchestrator.py.
    """
    tempo_us    = int(60_000_000 / max(bpm, 20))
    frame_ticks = int(TICKS_OUT * bpm * frame_ms / 60_000)

    mid = MidiFile(type=1, ticks_per_beat=TICKS_OUT)

    # Pista de tempo
    t0 = MidiTrack()
    t0.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    mid.tracks.append(t0)

    INSTR_PROG = {
        'piano': 0, 'electric_piano': 4, 'organ': 19, 'harpsichord': 6,
        'vibraphone': 11, 'celesta': 8,
        'violin1': 40, 'violin2': 40, 'viola': 41, 'cello': 42, 'contrabass': 43,
        'string_ensemble': 48, 'choir': 52,
        'guitar': 25, 'nylon_guitar': 24, 'steel_guitar': 25,
        'bass_guitar': 33, 'electric_bass': 33, 'acoustic_bass': 32,
        'fretless_bass': 35, 'synth_bass': 38,
        'flute': 73, 'oboe': 68, 'clarinet': 71, 'bassoon': 70,
        'horn': 60, 'trumpet': 56, 'trombone': 57, 'tuba': 58,
        'alto_sax': 65, 'tenor_sax': 66, 'soprano_sax': 64,
        'baritone_sax': 67, 'brass_section': 61,
        'timpani': 47, 'drums': 0,
    }
    DEFAULT_INSTRS = {
        'keys':       {'name': 'piano',      'program': 0},
        'strings':    {'name': 'violin1',    'program': 40},
        'bass':       {'name': 'bass_guitar','program': 33},
        'winds':      {'name': 'trumpet',    'program': 56},
        'percussion': {'name': 'drums',      'program': 0},
    }

    instr_by_family: dict[str, dict] = {}
    if instruments:
        for instr in instruments:
            name = instr.get('name', '')
            fam  = INSTR_TO_FAMILY.get(name, 'strings')
            if fam not in instr_by_family:
                prog = INSTR_PROG.get(name, 40)
                instr_by_family[fam] = {'name': name, 'program': prog}
    for fam, cfg in DEFAULT_INSTRS.items():
        if fam not in instr_by_family:
            instr_by_family[fam] = cfg

    T = roll.shape[0]

    for fam_idx, fam in enumerate(FAMILIES):
        cfg = instr_by_family.get(fam)
        if cfg is None:
            continue

        name    = cfg['name']
        program = cfg['program']
        channel = fam_idx if fam_idx != 4 else 9
        if channel == 9 and fam != 'percussion':
            channel = fam_idx + 10

        track = MidiTrack()
        track.append(MetaMessage('track_name', name=name, time=0))
        if fam != 'percussion':
            track.append(Message('program_change', channel=channel,
                                 program=program, time=0))

        note_events: list[tuple[int, str, int, int]] = []
        base_col = fam_idx * N_PITCHES

        for pi in range(N_PITCHES):
            pitch = PITCH_LO + pi
            col   = base_col + pi

            sig = roll[:, col]   # vista 1-D del pitch a lo largo del tiempo

            for f_on, f_off, peak_vel in _segment_pitch(
                    sig, min_peak=min_peak,
                    valley_ratio=valley_ratio,
                    max_frames=max_note_frames):
                vel_midi = int(np.clip(peak_vel * 127, 1, 127))
                note_events.append((f_on  * frame_ticks, 'on',  pitch, vel_midi))
                note_events.append((f_off * frame_ticks, 'off', pitch, 0))

        # off antes que on en el mismo tick (evita solapamientos en reproductores)
        note_events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

        prev_tick = 0
        for tick_abs, etype, pitch, vel in note_events:
            dt = max(0, tick_abs - prev_tick)
            if etype == 'on':
                track.append(Message('note_on',  channel=channel,
                                     note=pitch, velocity=vel, time=dt))
            else:
                track.append(Message('note_off', channel=channel,
                                     note=pitch, velocity=0,   time=dt))
            prev_tick = tick_abs

        if note_events:
            mid.tracks.append(track)

    mid.save(output_path)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  ESTADÍSTICAS DEL ROLL
# ══════════════════════════════════════════════════════════════════════════════

def get_roll_stats(roll: np.ndarray) -> dict:
    """Estadísticas del piano roll por familia."""
    T = roll.shape[0]
    stats = {'families': {}, 'global': {}}

    total_active = 0.0
    for fam_idx, fam in enumerate(FAMILIES):
        base = fam_idx * N_PITCHES
        block = roll[:, base:base + N_PITCHES]   # [T, 72]

        # Frames con al menos una nota
        frames_active = (block.max(axis=1) > 0.05).sum()
        active_pct    = frames_active / max(T, 1)

        # Nota más frecuente
        note_counts = block.sum(axis=0)
        top_pi      = int(note_counts.argmax())
        top_pitch   = PITCH_LO + top_pi
        top_name    = _pitch_name(top_pitch)

        # Densidad media (notas simultáneas)
        density = float((block > 0.05).sum(axis=1).mean())

        # Dinámica media
        dynamic = float(block[block > 0.05].mean()) if (block > 0.05).any() else 0.0

        stats['families'][fam] = {
            'active_pct': float(active_pct),
            'top_pitch':  top_pitch,
            'top_name':   top_name,
            'density':    density,
            'dynamic':    dynamic,
        }
        total_active += active_pct

    stats['global']['mean_family_activity'] = total_active / N_FAMILIES
    stats['global']['sparsity'] = float((roll < 0.05).mean())
    return stats


def _pitch_name(midi: int) -> str:
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[midi % 12]}{midi // 12 - 1}"


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH CHECK
# ══════════════════════════════════════════════════════════════════════════════

def batch_check_roll(midi_files: list[str],
                     frame_ms: int = FRAME_MS2,
                     max_frames: int = 256) -> dict:
    """Valida corpus y reporta estadísticas del piano roll."""
    ok, failed = [], []
    all_bpms, all_active, all_sparsity = [], [], []
    family_coverage = defaultdict(int)
    total = len(midi_files)

    for i, f in enumerate(midi_files):
        roll, bpm = midi_to_roll(f, frame_ms=frame_ms, max_frames=max_frames)
        if roll is None:
            failed.append(f)
        else:
            ok.append(f)
            all_bpms.append(bpm)
            stats = get_roll_stats(roll)
            all_active.append(stats['global']['mean_family_activity'])
            all_sparsity.append(stats['global']['sparsity'])
            for fam, s in stats['families'].items():
                if s['active_pct'] > 0.05:
                    family_coverage[fam] += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total}  OK={len(ok)}  Fail={len(failed)}", end='\r')

    print()
    return {
        'total':    total,
        'valid':    len(ok),
        'failed':   len(failed),
        'failed_files': failed[:10],
        'bpm_mean': float(np.mean(all_bpms)) if all_bpms else 0,
        'bpm_std':  float(np.std(all_bpms))  if all_bpms else 0,
        'activity': float(np.mean(all_active))    if all_active else 0,
        'sparsity': float(np.mean(all_sparsity))  if all_sparsity else 0,
        'family_coverage': dict(family_coverage),
        'tensor_shape': (max_frames, TENSOR_DIMS2),
        'mb_per_midi':  max_frames * TENSOR_DIMS2 * 4 / 1024 / 1024,
        'mb_corpus':    len(ok) * max_frames * TENSOR_DIMS2 * 4 / 1024 / 1024,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='LM_REPR2: representación piano roll exacta')
    parser.add_argument('midi', nargs='+')
    parser.add_argument('--info',        action='store_true')
    parser.add_argument('--roundtrip',   action='store_true',
                        help='MIDI → roll → MIDI (test de fidelidad)')
    parser.add_argument('--batch-check', action='store_true')
    parser.add_argument('--output',      default='reconstructed2.mid')
    parser.add_argument('--frame-ms',    type=int, default=FRAME_MS2)
    parser.add_argument('--max-frames',  type=int, default=256)
    parser.add_argument('--threshold',   type=float, default=0.05,
                        help='Umbral de activación de nota (default: 0.05)')
    args = parser.parse_args()

    if args.batch_check:
        print(f"\nValidando corpus: {len(args.midi)} ficheros...")
        r = batch_check_roll(args.midi, args.frame_ms, args.max_frames)
        print(f"\n═══ RESUMEN PIANO ROLL ═══")
        print(f"  Total:     {r['total']}")
        print(f"  Válidos:   {r['valid']}  ({100*r['valid']/max(r['total'],1):.1f}%)")
        print(f"  BPM:       {r['bpm_mean']:.1f} ± {r['bpm_std']:.1f}")
        print(f"  Actividad: {r['activity']:.3f}")
        print(f"  Sparsidad: {r['sparsity']*100:.1f}%  (% de ceros en el tensor)")
        print(f"  Tensor:    {r['tensor_shape']}  ({r['mb_per_midi']:.2f} MB/MIDI)")
        print(f"  Corpus:    {r['mb_corpus']:.0f} MB en RAM")
        print(f"  Cobertura por familia:")
        for fam, count in r['family_coverage'].items():
            pct = 100 * count / max(r['valid'], 1)
            print(f"    {fam:<12}: {count:>4}  ({pct:.1f}%)")
        if r['failed_files']:
            print(f"  Fallidos: {r['failed_files'][:5]}")
        return

    midi_file = args.midi[0]
    print(f"\nProcesando: {midi_file}")
    roll, bpm = midi_to_roll(midi_file, args.frame_ms, args.max_frames)

    if roll is None:
        print("ERROR: no se pudo procesar el fichero")
        sys.exit(1)

    print(f"  Shape: {roll.shape}  BPM: {bpm:.1f}")
    print(f"  Rango de valores: [{roll.min():.3f}, {roll.max():.3f}]")
    print(f"  Notas activas: {(roll > args.threshold).sum()} "
          f"({(roll > args.threshold).mean()*100:.1f}%)")

    if args.info:
        stats = get_roll_stats(roll)
        print(f"\n  Estadísticas por familia:")
        print(f"  {'Familia':<12}  {'Activo':>7}  {'Top nota':>8}  "
              f"{'Densidad':>9}  {'Dinámica':>9}")
        for fam, s in stats['families'].items():
            bar = '█' * int(s['active_pct'] * 20)
            print(f"  {fam:<12}  {s['active_pct']*100:6.1f}%  "
                  f"{s['top_name']:>8}  {s['density']:>9.2f}  "
                  f"{s['dynamic']:>9.3f}  {bar}")

    if args.roundtrip:
        print(f"\n  Roundtrip: roll → MIDI → {args.output}")
        roll_to_midi(roll, bpm, output_path=args.output,
                     frame_ms=args.frame_ms)
        # Verificar calidad del roundtrip
        roll2, bpm2 = midi_to_roll(args.output, args.frame_ms, args.max_frames)
        if roll2 is not None:
            mse  = float(((roll - roll2)**2).mean())
            # Correlación solo sobre frames activos
            mask = (roll > args.threshold) | (roll2 > args.threshold)
            if mask.any():
                corr = float(np.corrcoef(
                    roll[mask].flatten(), roll2[mask].flatten())[0, 1])
            else:
                corr = 0.0
            print(f"  MSE roundtrip:         {mse:.5f}")
            print(f"  Correlación (activos): {corr:.4f}")
            print(f"  Guardado: {args.output}")


if __name__ == '__main__':
    main()
