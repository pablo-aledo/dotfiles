#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      LEITMOTIF ANALYST  v1.1                                 ║
║     Identificación de leitmotifs y sus transformaciones en una obra MIDI     ║
║                                                                              ║
║  Dado un archivo MIDI, el programa:                                          ║
║    1. Extrae motivos candidatos usando Re-Pair (como harvester.py)           ║
║    2. Agrupa los más significativos como «leitmotifs» de la obra             ║
║    3. Rastrea cada leitmotif y sus variaciones a lo largo de la pieza        ║
║       detectando las siguientes transformaciones (según variation_engine):   ║
║         V01 — Inversión del contorno melódico                                ║
║         V02 — Retrógrado (orden temporal invertido)                          ║
║         V03 — Inversión retrógrada                                           ║
║         V04 — Aumentación rítmica (duraciones duplicadas)                    ║
║         V05 — Diminución rítmica (duraciones reducidas)                      ║
║         V08 — Transposición (mismo contorno, otra altura)                    ║
║         V09 — Modal (cambio mayor ↔ menor)                                   ║
║    4. Presenta el resultado como:                                            ║
║         · Tabla de leitmotifs con sus apariciones por compás                 ║
║         · Tipo de aparición (original / transformación)                      ║
║         · Mapa ASCII de presencia a lo largo de la obra                      ║
║         · Reporte JSON opcional (--report)                                   ║
║    5. Exporta los leitmotifs como archivos MIDI independientes               ║
║                                                                              ║
║  USO:                                                                        ║
║    python leitmotif_analyst.py obra.mid                                      ║
║    python leitmotif_analyst.py obra.mid --min-bars 2 --max-bars 8           ║
║    python leitmotif_analyst.py obra.mid --threshold 0.60                    ║
║    python leitmotif_analyst.py obra.mid --top 5                             ║
║    python leitmotif_analyst.py obra.mid --report                            ║
║    python leitmotif_analyst.py obra.mid --verbose                           ║
║                                                                              ║
║  MODO LEITMOTIF EXTERNO (--leitmotif):                                       ║
║    Si se proporciona un MIDI con el leitmotif de referencia, el programa    ║
║    omite la detección automática y busca únicamente ese motivo (y sus       ║
║    variaciones) a lo largo de la obra. La salida es idéntica al modo        ║
║    automático. Con --save-leitmotifs o --save-all se exporta además un      ║
║    MIDI por cada variación del motivo dado (forma fundamental + V01–V09).   ║
║                                                                              ║
║    python leitmotif_analyst.py obra.mid --leitmotif motivo.mid              ║
║    python leitmotif_analyst.py obra.mid --leitmotif motivo.mid --save-all   ║
║                                                                              ║
║    --save-leitmotifs   Guarda un MIDI por leitmotif (forma fundamental,     ║
║                        extraída de su primera aparición en la obra)         ║
║    --save-all          Guarda además un MIDI por cada aparición encontrada  ║
║                        (incluye originales y todas las variaciones)         ║
║    --save-dir DIR      Carpeta de destino (default: <stem>_leitmotifs/)     ║
║                                                                              ║
║  Nomenclatura de archivos generados:                                        ║
║    <stem>.LM-01.mid                   — forma fundamental del leitmotif 1  ║
║    <stem>.LM-01.bar042.original.mid   — aparición original en compás 42    ║
║    <stem>.LM-01.bar067.V01.mid        — inversión en compás 67             ║
║    <stem>.LM-01.bar089.V04.mid        — aumentación en compás 89           ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --leitmotif FILE    MIDI con el leitmotif de referencia (omite           ║
║                        detección automática y busca solo ese motivo)        ║
║    --min-bars N        Longitud mínima de leitmotif en compases (def: 1)    ║
║    --max-bars N        Longitud máxima de leitmotif en compases (def: 8)    ║
║    --threshold F       Umbral de similitud para detectar apariciones        ║
║                        (0-1, default: 0.55)                                  ║
║    --var-threshold F   Umbral para reconocer una variación (def: 0.50)      ║
║    --top N             Número máximo de leitmotifs a identificar (def: 8)   ║
║    --report            Guardar reporte JSON                                  ║
║    --save-leitmotifs   Guardar forma fundamental de cada leitmotif como MIDI║
║    --save-all          Guardar forma fundamental + todas las apariciones     ║
║    --save-dir DIR      Carpeta de destino para los MIDIs exportados         ║
║    --verbose           Salida detallada                                      ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                   ║
║  OPCIONALES:   scipy (similitud coseno mejorada)                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np

try:
    import mido
    MIDO_OK = True
except ImportError:
    print("ERROR: mido no instalado. Ejecuta: pip install mido")
    sys.exit(1)

try:
    from scipy.spatial.distance import cosine as cosine_distance
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

VERSION = "1.1"

# ══════════════════════════════════════════════════════════════════════════════
#  NOMBRES DESCRIPTIVOS PARA LAS VARIACIONES
# ══════════════════════════════════════════════════════════════════════════════

VARIATION_NAMES = {
    "original":    "Original",
    "V01":         "Inversión",
    "V02":         "Retrógrado",
    "V03":         "Inv. retrógrada",
    "V04":         "Aumentación",
    "V05":         "Diminución",
    "V08":         "Transposición",
    "V09":         "Modal",
}

# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE NOTAS DESDE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def extract_notes(midi_path: str) -> tuple:
    """
    Extrae todas las notas melódicas (no canal 10) del MIDI.

    Devuelve:
        notes       : lista de (pitch, velocity, onset_tick, duration_ticks)
        tpb         : ticks por beat
        total_ticks : duración total en ticks
        bar_ticks   : duración de un compás en ticks
        tempo_bpm   : tempo estimado en BPM
        time_sig    : (numerator, denominator)
        tempo_us    : tempo en microsegundos (para reconstrucción MIDI)
    """
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat or 480

    # Extraer tempo y compás
    tempo_us = 500_000
    ts_num, ts_den = 4, 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator

    tempo_bpm = round(60_000_000 / max(tempo_us, 1), 1)
    beats_per_bar = ts_num * (4.0 / ts_den)
    bar_ticks = int(beats_per_bar * tpb)

    # Extraer notas con tiempo absoluto
    notes = []
    for track in mid.tracks:
        # Saltar pistas de percusión
        if any(getattr(m, 'channel', -1) == 9 for m in track if hasattr(m, 'channel')):
            continue
        active = {}
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[msg.note] = (tick, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    onset, vel = active.pop(msg.note)
                    dur = max(1, tick - onset)
                    notes.append((msg.note, vel, onset, dur))

    notes.sort(key=lambda x: x[2])
    total_ticks = max((n[2] + n[3] for n in notes), default=bar_ticks)
    return notes, tpb, total_ticks, bar_ticks, tempo_bpm, (ts_num, ts_den), tempo_us


def tick_to_bar(tick: int, bar_ticks: int) -> int:
    """Convierte un tick absoluto al número de compás (base 1)."""
    return int(tick / bar_ticks) + 1


def bar_to_tick(bar: int, bar_ticks: int) -> int:
    """Convierte número de compás (base 1) a tick de inicio."""
    return (bar - 1) * bar_ticks


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE SEGMENTOS MIDI
# ══════════════════════════════════════════════════════════════════════════════

def slice_midi(mid: mido.MidiFile, start_tick: int, end_tick: int) -> mido.MidiFile:
    """
    Extrae el segmento [start_tick, end_tick) del MIDI original preservando
    los meta-mensajes de tempo, compás y armadura previos al corte.
    """
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)

    for track in mid.tracks:
        new_track = mido.MidiTrack()

        # Recoger meta-mensajes anteriores al inicio (tempo, compás, etc.)
        preamble = []
        events_in_range = []
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.is_meta:
                if abs_t <= start_tick:
                    preamble.append(msg.copy(time=0))
                elif abs_t < end_tick:
                    events_in_range.append((abs_t, msg))
            else:
                if start_tick <= abs_t < end_tick:
                    events_in_range.append((abs_t, msg))

        for msg in preamble:
            new_track.append(msg)

        prev = start_tick
        for abs_t2, msg in sorted(events_in_range, key=lambda x: x[0]):
            dt = max(0, abs_t2 - prev)
            new_track.append(msg.copy(time=dt))
            prev = abs_t2

        new_track.append(mido.MetaMessage('end_of_track', time=0))
        out.tracks.append(new_track)

    return out


def notes_to_midi(notes: list, tpb: int, tempo_us: int = 500_000,
                   ts_num: int = 4, ts_den: int = 4) -> mido.MidiFile:
    """
    Construye un MIDI mínimo a partir de una lista de notas
    (pitch, velocity, onset_tick, duration_ticks).
    Los ticks de onset se reescriben para que el primer onset sea 0.
    """
    mid = mido.MidiFile(type=0, ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Meta-mensajes de cabecera
    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    track.append(mido.MetaMessage('time_signature',
                                   numerator=ts_num, denominator=ts_den,
                                   clocks_per_click=24, notated_32nd_notes_per_beat=8,
                                   time=0))

    if not notes:
        track.append(mido.MetaMessage('end_of_track', time=0))
        return mid

    # Recentrar los ticks al inicio
    offset = notes[0][2]
    events = []
    for pitch, vel, onset, dur in notes:
        t_on  = onset - offset
        t_off = t_on + dur
        events.append((t_on,  'on',  pitch, vel))
        events.append((t_off, 'off', pitch, 0))

    events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

    prev = 0
    for tick, kind, pitch, vel in events:
        dt = max(0, tick - prev)
        if kind == 'on':
            track.append(mido.Message('note_on',  note=pitch,
                                       velocity=vel, time=dt))
        else:
            track.append(mido.Message('note_off', note=pitch,
                                       velocity=0,   time=dt))
        prev = tick

    track.append(mido.MetaMessage('end_of_track', time=0))
    return mid


def save_leitmotif_midi(leitmotif: dict, out_path: str,
                         tpb: int, tempo_us: int,
                         ts_num: int, ts_den: int) -> bool:
    """
    Guarda la forma fundamental de un leitmotif (sus notas canónicas)
    como archivo MIDI independiente.
    """
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        mid = notes_to_midi(leitmotif["notes"], tpb, tempo_us, ts_num, ts_den)
        mid.save(out_path)
        return True
    except Exception as e:
        print(f"    [error] No se pudo guardar {out_path}: {e}")
        return False


def save_appearance_midi(appearance: dict, notes_all: list,
                          out_path: str, tpb: int,
                          tempo_us: int, ts_num: int, ts_den: int,
                          n_notes_motif: int) -> bool:
    """
    Guarda una aparición concreta (original o variación) extraída directamente
    de la obra fuente como MIDI independiente.
    """
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        start_idx = appearance["note_idx"]
        end_idx   = min(start_idx + n_notes_motif, len(notes_all))
        segment_notes = notes_all[start_idx:end_idx]
        if not segment_notes:
            return False
        mid = notes_to_midi(segment_notes, tpb, tempo_us, ts_num, ts_den)
        mid.save(out_path)
        return True
    except Exception as e:
        print(f"    [error] No se pudo guardar {out_path}: {e}")
        return False


def export_midis(leitmotifs: list,
                  all_appearances: dict,
                  notes_all: list,
                  midi_path: str,
                  tpb: int,
                  tempo_us: int,
                  ts_num: int,
                  ts_den: int,
                  save_dir: str,
                  save_all: bool = False,
                  verbose: bool = False) -> dict:
    """
    Exporta los leitmotifs (y opcionalmente sus apariciones) como MIDIs.

    Modos:
        save_all=False  → solo forma fundamental (un archivo por leitmotif)
        save_all=True   → fundamental + cada aparición individual en la obra

    Devuelve dict con rutas guardadas agrupadas por leitmotif.
    """
    stem = Path(midi_path).stem
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    total_files = 0

    print(f"\n{'═' * 68}")
    print(f"  EXPORTANDO MIDI{'S' if save_all else ''} DE LEITMOTIFS")
    print(f"  Carpeta: {out_dir}")
    print(f"{'═' * 68}")

    for lm in leitmotifs:
        lm_id = lm["id"]
        saved[lm_id] = {"fundamental": None, "appearances": []}

        # ── Forma fundamental ──────────────────────────────────────────────
        fund_name = f"{stem}.{lm_id}.mid"
        fund_path = str(out_dir / fund_name)
        ok = save_leitmotif_midi(lm, fund_path, tpb, tempo_us, ts_num, ts_den)
        if ok:
            saved[lm_id]["fundamental"] = fund_path
            total_files += 1
            print(f"  ✓ {lm_id}  fundamental  →  {fund_name}")
        else:
            print(f"  ✗ {lm_id}  fundamental  →  error")

        if not save_all:
            continue

        # ── Apariciones individuales ───────────────────────────────────────
        appearances = all_appearances.get(lm_id, [])
        for a in appearances:
            bar       = a["bar"]
            var_code  = a["type"]          # "original" | "V01" ...
            app_name  = f"{stem}.{lm_id}.bar{bar:03d}.{var_code}.mid"
            app_path  = str(out_dir / app_name)

            ok = save_appearance_midi(
                a, notes_all, app_path,
                tpb, tempo_us, ts_num, ts_den,
                n_notes_motif=lm["n_notes"],
            )
            if ok:
                saved[lm_id]["appearances"].append({
                    "bar":      bar,
                    "type":     var_code,
                    "path":     app_path,
                })
                total_files += 1
                if verbose:
                    type_name = VARIATION_NAMES.get(var_code, var_code)
                    print(f"    ◌ compás {bar:3d}  {type_name:<18}  →  {app_name}")
            # Solo reportar errores en modo no-verbose para no saturar la salida
            elif verbose:
                print(f"    ✗ compás {bar:3d}  {var_code}  →  error")

        n_app_saved = len(saved[lm_id]["appearances"])
        if save_all and n_app_saved > 0:
            n_orig = sum(1 for x in saved[lm_id]["appearances"]
                         if x["type"] == "original")
            n_var  = n_app_saved - n_orig
            print(f"    → {n_app_saved} apariciones guardadas  "
                  f"({n_orig} originales + {n_var} variaciones)")

    print(f"\n  Total archivos exportados: {total_files}")
    print(f"{'═' * 68}\n")
    return saved


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE MOTIVOS REPETIDOS (Re-Pair, adaptado de harvester.py)
# ══════════════════════════════════════════════════════════════════════════════

def _quantize_dur(dur_ticks: int, tpb: int) -> str:
    """Cuantiza una duración en ticks a símbolo rítmico."""
    q = dur_ticks / tpb
    if q < 0.3:
        return 'X'   # muy corta (fusa)
    elif q < 0.6:
        return 'S'   # corchea
    elif q < 1.1:
        return 'Q'   # negra
    elif q < 1.9:
        return 'H'   # negra con puntillo / blanca ~
    else:
        return 'L'   # blanca o mayor


def notes_to_interval_sequence(notes: list, tpb: int) -> tuple:
    """
    Convierte lista de notas a secuencia de (intervalo_semitono, dur_símbolo).
    Los intervalos se limitan a ±12 (invarianza a octava parcial).
    Devuelve (secuencia, índices_de_notas).
    """
    if len(notes) < 2:
        return [], []
    seq = []
    idxs = []
    for i in range(len(notes) - 1):
        interval = notes[i + 1][0] - notes[i][0]
        interval = max(-12, min(12, interval))
        dur_sym = _quantize_dur(notes[i][3], tpb)
        seq.append((interval, dur_sym))
        idxs.append(i)
    return seq, idxs


def repair_find_motifs(seq: list, min_len: int = 3,
                        max_len: int = 10, min_count: int = 2) -> list:
    """
    Algoritmo Re-Pair simplificado: encuentra sub-secuencias repetidas.
    Devuelve lista de (motif_tuple, count, [start_positions]) ordenada por score.
    """
    best = {}
    for length in range(min_len, min(max_len + 1, len(seq) - 1)):
        counts = defaultdict(list)
        for i in range(len(seq) - length + 1):
            key = tuple(seq[i: i + length])
            counts[key].append(i)
        for key, positions in counts.items():
            # Filtrar posiciones solapadas
            filtered = []
            prev = -length
            for pos in sorted(positions):
                if pos >= prev + length:
                    filtered.append(pos)
                    prev = pos
            if len(filtered) >= min_count:
                score = length * len(filtered)
                if key not in best or score > best[key][0]:
                    best[key] = (score, len(filtered), filtered)

    motifs_sorted = sorted(best.items(), key=lambda x: -x[1][0])
    result = []
    covered = set()
    for motif_key, (score, count, positions) in motifs_sorted[:30]:
        clean_pos = []
        for pos in positions:
            span = set(range(pos, pos + len(motif_key)))
            if not span & covered:
                clean_pos.append(pos)
                covered |= span
        if len(clean_pos) >= min_count:
            result.append((motif_key, len(clean_pos), clean_pos))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  FINGERPRINT DE MOTIVOS (basado en leitmotif_tracker.py)
# ══════════════════════════════════════════════════════════════════════════════

def build_contour(notes: list) -> list:
    """
    Huella de contorno melódico: deltas de pitch entre notas consecutivas.
    Invariante a transposición.
    """
    if len(notes) < 2:
        return []
    pitches = [n[0] for n in notes[:16]]
    return [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]


def build_rhythm(notes: list, tpb: int) -> list:
    """
    Huella rítmica: duraciones normalizadas respecto a la media.
    Invariante a tempo.
    """
    if not notes:
        return []
    durations = [n[3] for n in notes[:16]]
    mean_dur = max(np.mean(durations), 1)
    return [round(d / mean_dur, 2) for d in durations]


def make_fingerprint(notes: list, tpb: int) -> dict:
    """Construye la huella completa de un fragmento de notas."""
    return {
        "contour": build_contour(notes),
        "rhythm":  build_rhythm(notes, tpb),
        "pitches": [n[0] for n in notes[:16]],
        "n_notes": len(notes),
    }


def contour_similarity(c1: list, c2: list, tolerance: int = 1) -> float:
    """Similitud de contorno [0,1] con tolerancia en semitonos."""
    if not c1 or not c2:
        return 0.0
    n = min(len(c1), len(c2), 8)
    matches = sum(1 for i in range(n) if abs(c1[i] - c2[i]) <= tolerance)
    return matches / n


def rhythm_similarity(r1: list, r2: list) -> float:
    """Similitud rítmica [0,1] con tolerancia ±25%."""
    if not r1 or not r2:
        return 0.0
    n = min(len(r1), len(r2), 8)
    matches = sum(1 for i in range(n) if abs(r1[i] - r2[i]) < 0.25)
    return matches / n


def motif_similarity(fp_a: dict, fp_b: dict) -> float:
    """Similitud global entre dos huellas (contorno 60% + ritmo 40%)."""
    cs = contour_similarity(fp_a.get("contour", []), fp_b.get("contour", []))
    rs = rhythm_similarity(fp_a.get("rhythm", []), fp_b.get("rhythm", []))
    return round(0.6 * cs + 0.4 * rs, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSFORMACIONES PARA DETECCIÓN (según variation_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

def apply_inversion(fp: dict) -> dict:
    """V01: Inversión — refleja el contorno melódico (negar deltas)."""
    inv_contour = [-d for d in fp["contour"]]
    # Reconstruir pitches desde el primer pitch de referencia
    ref = fp["pitches"][0] if fp["pitches"] else 60
    pitches = [ref]
    for delta in inv_contour:
        pitches.append(pitches[-1] + delta)
    return {
        "contour": inv_contour,
        "rhythm":  fp["rhythm"][:],
        "pitches": pitches,
        "n_notes": fp["n_notes"],
    }


def apply_retrograde(fp: dict) -> dict:
    """V02: Retrógrado — invierte el orden temporal."""
    rev_pitches = list(reversed(fp["pitches"]))
    rev_contour = [rev_pitches[i + 1] - rev_pitches[i]
                   for i in range(len(rev_pitches) - 1)]
    return {
        "contour": rev_contour,
        "rhythm":  list(reversed(fp["rhythm"])),
        "pitches": rev_pitches,
        "n_notes": fp["n_notes"],
    }


def apply_retro_inversion(fp: dict) -> dict:
    """V03: Inversión retrógrada — invierte y luego refleja."""
    retro = apply_retrograde(fp)
    return apply_inversion(retro)


def apply_augmentation(fp: dict) -> dict:
    """V04: Aumentación — las duraciones relativas se hacen más largas (×2)."""
    # El contorno no cambia, pero el ritmo normalizado se duplica
    aug_rhythm = [r * 2.0 for r in fp["rhythm"]]
    mean_aug = max(np.mean(aug_rhythm), 0.01) if aug_rhythm else 1.0
    norm_rhythm = [round(r / mean_aug, 2) for r in aug_rhythm]
    return {
        "contour": fp["contour"][:],
        "rhythm":  norm_rhythm,
        "pitches": fp["pitches"][:],
        "n_notes": fp["n_notes"],
    }


def apply_diminution(fp: dict) -> dict:
    """V05: Diminución — duraciones reducidas a la mitad."""
    dim_rhythm = [r * 0.5 for r in fp["rhythm"]]
    mean_dim = max(np.mean(dim_rhythm), 0.01) if dim_rhythm else 1.0
    norm_rhythm = [round(r / mean_dim, 2) for r in dim_rhythm]
    return {
        "contour": fp["contour"][:],
        "rhythm":  norm_rhythm,
        "pitches": fp["pitches"][:],
        "n_notes": fp["n_notes"],
    }


def apply_transposition(fp: dict, semitones: int = 5) -> dict:
    """V08: Transposición — mismo contorno, pitches desplazados."""
    new_pitches = [p + semitones for p in fp["pitches"]]
    return {
        "contour": fp["contour"][:],   # contorno idéntico
        "rhythm":  fp["rhythm"][:],
        "pitches": new_pitches,
        "n_notes": fp["n_notes"],
    }


def apply_modal(fp: dict) -> dict:
    """
    V09: Modal — cambio mayor ↔ menor.
    Afecta al contorno: el 3er grado (±4 o ±3 semitonos) se altera ±1.
    """
    modal_contour = fp["contour"][:]
    # Buscar intervalos de tercera y alterarlos
    for i, d in enumerate(modal_contour):
        if d == 4:
            modal_contour[i] = 3   # mayor → menor
        elif d == -4:
            modal_contour[i] = -3
        elif d == 3:
            modal_contour[i] = 4   # menor → mayor
        elif d == -3:
            modal_contour[i] = -4
    return {
        "contour": modal_contour,
        "rhythm":  fp["rhythm"][:],
        "pitches": fp["pitches"][:],
        "n_notes": fp["n_notes"],
    }


# Mapa de variaciones: código → (nombre, función)
VARIATION_TRANSFORMS = {
    "V01": ("Inversión",         apply_inversion),
    "V02": ("Retrógrado",        apply_retrograde),
    "V03": ("Inv. retrógrada",   apply_retro_inversion),
    "V04": ("Aumentación",       apply_augmentation),
    "V05": ("Diminución",        apply_diminution),
    "V08": ("Transposición",     lambda fp: apply_transposition(fp, 5)),
    "V09": ("Modal",             apply_modal),
}


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  LEITMOTIF EXTERNO: carga desde MIDI proporcionado por el usuario
# ══════════════════════════════════════════════════════════════════════════════

def leitmotif_from_midi(leitmotif_path: str, tpb_target: int) -> dict:
    """
    Carga un MIDI externo como leitmotif de referencia.
    Las notas se re-escalan a tpb_target para que la comparación de
    duraciones sea coherente con la obra a analizar.

    Devuelve un dict con la misma estructura que los producidos por
    extract_leitmotifs(), marcado con source="external".
    """
    lm_notes, lm_tpb, _, _, _, _, _ = extract_notes(leitmotif_path)
    if not lm_notes:
        raise ValueError(f"No se encontraron notas en {leitmotif_path}")

    # Re-escalar ticks si el tpb del leitmotif difiere del de la obra
    if lm_tpb != tpb_target and lm_tpb > 0:
        ratio = tpb_target / lm_tpb
        lm_notes = [
            (p, v, int(onset * ratio), max(1, int(dur * ratio)))
            for p, v, onset, dur in lm_notes
        ]

    fp = make_fingerprint(lm_notes, tpb_target)

    # Construir un interval_pattern sintético para el resumen
    pitches = [n[0] for n in lm_notes]
    interval_pattern = [pitches[i+1] - pitches[i]
                        for i in range(len(pitches) - 1)]

    return {
        "id":               "LM-EXT",
        "notes":            lm_notes,
        "fingerprint":      fp,
        "bar_start":        1,
        "n_notes":          len(lm_notes),
        "repetitions":      0,          # no calculado por Re-Pair
        "interval_pattern": interval_pattern,
        "dur_pattern":      [],
        "source":           "external",
        "source_path":      str(Path(leitmotif_path).resolve()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE VARIACIONES DEL LEITMOTIF EXTERNO
# ══════════════════════════════════════════════════════════════════════════════

def _apply_transform_to_notes(notes: list, var_code: str,
                                tpb: int) -> list:
    """
    Aplica una transformación directamente sobre la lista de notas
    (pitch, velocity, onset_tick, duration_ticks), devolviendo una nueva
    lista lista para escribir a MIDI.

    Las transformaciones de pitch reconstruyen la melodía a partir del
    primer pitch de referencia y de los nuevos intervalos.
    Las de ritmo escalan las duraciones manteniendo los onsets relativos.
    """
    if not notes:
        return []

    pitches   = [n[0] for n in notes]
    vels      = [n[1] for n in notes]
    durations = [n[3] for n in notes]
    mean_dur  = max(np.mean(durations), 1)

    # ── Transformaciones de pitch ────────────────────────────────────────────
    if var_code == "V01":           # Inversión
        deltas = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        new_pitches = [pitches[0]]
        for d in deltas:
            new_pitches.append(max(0, min(127, new_pitches[-1] - d)))
        new_durations = durations[:]

    elif var_code == "V02":         # Retrógrado
        new_pitches   = list(reversed(pitches))
        new_durations = list(reversed(durations))

    elif var_code == "V03":         # Inversión retrógrada
        deltas = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        inv = [pitches[0]]
        for d in deltas:
            inv.append(max(0, min(127, inv[-1] - d)))
        new_pitches   = list(reversed(inv))
        new_durations = list(reversed(durations))

    elif var_code == "V04":         # Aumentación rítmica (×2)
        new_pitches   = pitches[:]
        new_durations = [d * 2 for d in durations]

    elif var_code == "V05":         # Diminución rítmica (×½)
        new_pitches   = pitches[:]
        new_durations = [max(1, d // 2) for d in durations]

    elif var_code == "V08":         # Transposición (+5 semitonos)
        new_pitches   = [max(0, min(127, p + 5)) for p in pitches]
        new_durations = durations[:]

    elif var_code == "V09":         # Modal (mayor ↔ menor, altera terceras)
        deltas = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
        modal_deltas = []
        for d in deltas:
            if d == 4:   modal_deltas.append(3)
            elif d == -4: modal_deltas.append(-3)
            elif d == 3:  modal_deltas.append(4)
            elif d == -3: modal_deltas.append(-4)
            else:         modal_deltas.append(d)
        new_pitches = [pitches[0]]
        for d in modal_deltas:
            new_pitches.append(max(0, min(127, new_pitches[-1] + d)))
        new_durations = durations[:]

    else:
        return notes[:]   # sin transformación

    # Reconstruir onsets relativos a partir de las nuevas duraciones
    new_notes = []
    onset = 0
    for i, (p, v, dur) in enumerate(zip(new_pitches, vels, new_durations)):
        new_notes.append((p, v, onset, dur))
        onset += dur

    return new_notes


def export_transformed_midis(leitmotif: dict,
                               midi_path: str,
                               tpb: int,
                               tempo_us: int,
                               ts_num: int,
                               ts_den: int,
                               save_dir: str,
                               verbose: bool = False) -> dict:
    """
    Genera un MIDI por cada variación del leitmotif externo:
      · <stem>.LM-EXT.mid            — forma original tal como fue dada
      · <stem>.LM-EXT.V01.mid        — inversión
      · <stem>.LM-EXT.V02.mid        — retrógrado
      · … (V03, V04, V05, V08, V09)

    Devuelve dict {var_code: ruta_guardada}.
    """
    stem    = Path(midi_path).stem
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lm_notes = leitmotif["notes"]

    saved = {}

    print(f"\n{'═' * 68}")
    print(f"  EXPORTANDO VARIACIONES DEL LEITMOTIF EXTERNO")
    print(f"  Carpeta: {out_dir}")
    print(f"{'═' * 68}")

    # Forma original
    orig_name = f"{stem}.LM-EXT.original.mid"
    orig_path = str(out_dir / orig_name)
    ok = save_leitmotif_midi(leitmotif, orig_path, tpb, tempo_us, ts_num, ts_den)
    if ok:
        saved["original"] = orig_path
        print(f"  ✓ original        →  {orig_name}")
    else:
        print(f"  ✗ original        →  error")

    # Cada variación
    for var_code in ["V01", "V02", "V03", "V04", "V05", "V08", "V09"]:
        var_name = VARIATION_NAMES.get(var_code, var_code)
        transformed = _apply_transform_to_notes(lm_notes, var_code, tpb)
        if not transformed:
            continue

        # Crear un leitmotif temporal con las notas transformadas
        lm_var = dict(leitmotif)
        lm_var["notes"] = transformed

        fname = f"{stem}.LM-EXT.{var_code}.mid"
        fpath = str(out_dir / fname)
        ok = save_leitmotif_midi(lm_var, fpath, tpb, tempo_us, ts_num, ts_den)
        if ok:
            saved[var_code] = fpath
            print(f"  ✓ {var_code} {var_name:<18}  →  {fname}")
            if verbose:
                transformed_fp = make_fingerprint(transformed, tpb)
                print(f"      contorno: {transformed_fp['contour'][:8]}")
        else:
            print(f"  ✗ {var_code} {var_name:<18}  →  error")

    print(f"\n  {len(saved)} archivos exportados.")
    print(f"{'═' * 68}\n")
    return saved


def extract_leitmotifs(notes: list, tpb: int, bar_ticks: int,
                        total_ticks: int,
                        min_bars: int = 1, max_bars: int = 8,
                        top_n: int = 8,
                        verbose: bool = False) -> list:
    """
    Identifica los leitmotifs principales de la obra usando Re-Pair.

    Devuelve lista de dicts con:
        id          : identificador (LM-01, LM-02, ...)
        notes       : lista de notas del motivo
        fingerprint : huella del motivo
        bar_start   : compás donde aparece por primera vez
        n_notes     : número de notas
        repetitions : número de repeticiones detectadas por Re-Pair
    """
    if not notes:
        return []

    seq, idxs = notes_to_interval_sequence(notes, tpb)
    if len(seq) < 3:
        return []

    min_len = max(2, int(min_bars * (len(notes) / max(1, total_ticks / bar_ticks))))
    max_len = min(16, int(max_bars * (len(notes) / max(1, total_ticks / bar_ticks))))
    min_len = max(2, min_len)
    max_len = max(min_len + 1, max_len)

    if verbose:
        print(f"  Re-Pair: ventana de notas [{min_len}–{max_len}], {len(seq)} intervalos")

    motifs_raw = repair_find_motifs(seq, min_len=min_len,
                                     max_len=max_len, min_count=2)

    if not motifs_raw:
        if verbose:
            print("  Re-Pair no encontró motivos repetidos. Usando ventanas fijas.")
        # Fallback: extraer motivos de 4-6 notas de secciones iniciales
        motifs_raw = _fallback_motifs(notes, seq, tpb, min_len, max_len)

    if not motifs_raw:
        return []

    leitmotifs = []
    for lm_idx, (motif_key, count, positions) in enumerate(motifs_raw[:top_n]):
        # Las posiciones son índices en la secuencia de intervalos
        # → recuperar las notas reales
        note_start = positions[0]
        n_notes_motif = len(motif_key) + 1  # intervalos + 1 = notas
        note_end = min(note_start + n_notes_motif, len(notes))
        motif_notes = notes[note_start:note_end]

        if len(motif_notes) < 2:
            continue

        fp = make_fingerprint(motif_notes, tpb)
        bar_start = tick_to_bar(motif_notes[0][2], bar_ticks)

        leitmotifs.append({
            "id":          f"LM-{lm_idx + 1:02d}",
            "notes":       motif_notes,
            "fingerprint": fp,
            "bar_start":   bar_start,
            "n_notes":     len(motif_notes),
            "repetitions": count,
            "interval_pattern": [iv for iv, _ in motif_key],
            "dur_pattern":  [d for _, d in motif_key],
        })

    if verbose:
        print(f"  {len(leitmotifs)} leitmotifs identificados")

    return leitmotifs


def _fallback_motifs(notes: list, seq: list, tpb: int,
                      min_len: int, max_len: int) -> list:
    """Fallback: busca motivos de longitud fija sin restricción de repeticiones."""
    target_len = min(6, max_len)
    if len(seq) < target_len:
        return []
    counts = defaultdict(list)
    for i in range(len(seq) - target_len + 1):
        key = tuple(seq[i: i + target_len])
        counts[key].append(i)
    # Ordenar por frecuencia
    sorted_motifs = sorted(counts.items(), key=lambda x: -len(x[1]))
    result = []
    for motif_key, positions in sorted_motifs[:10]:
        if positions:
            result.append((motif_key, len(positions), positions))
    return result


def trace_leitmotif(leitmotif: dict, notes: list, tpb: int,
                     bar_ticks: int, total_ticks: int,
                     threshold: float = 0.55,
                     var_threshold: float = 0.50,
                     verbose: bool = False) -> list:
    """
    Busca todas las apariciones de un leitmotif (y sus variaciones) a lo
    largo de la obra mediante sliding window.

    Devuelve lista de dicts:
        bar         : número de compás (base 1)
        type        : "original" | código de variación (V01-V09)
        type_name   : nombre legible de la transformación
        similarity  : valor de similitud [0,1]
        onset_tick  : tick de inicio
        note_idx    : índice de la nota inicial en la lista global
    """
    fp_ref = leitmotif["fingerprint"]
    n_notes_motif = leitmotif["n_notes"]
    lm_id = leitmotif["id"]

    # Pre-calcular huellas de todas las variaciones
    variation_fps = {}
    for var_code, (var_name, transform_fn) in VARIATION_TRANSFORMS.items():
        try:
            variation_fps[var_code] = (var_name, transform_fn(fp_ref))
        except Exception:
            pass

    appearances = []
    # La ventana tiene el mismo número de notas que el leitmotif
    window = n_notes_motif
    if window < 2:
        window = 2

    # Para evitar duplicados cercanos
    MIN_DISTANCE_BARS = 1

    for start_idx in range(len(notes) - window + 1):
        window_notes = notes[start_idx: start_idx + window]
        w_fp = make_fingerprint(window_notes, tpb)
        onset_tick = window_notes[0][2]
        bar = tick_to_bar(onset_tick, bar_ticks)

        # --- Comprobar similitud con el original ---
        sim_orig = motif_similarity(fp_ref, w_fp)
        best_type = None
        best_sim = 0.0

        if sim_orig >= threshold:
            best_type = "original"
            best_sim = sim_orig
        else:
            # --- Comprobar similitud con cada variación ---
            for var_code, (var_name, var_fp) in variation_fps.items():
                sim_var = motif_similarity(var_fp, w_fp)
                if sim_var >= var_threshold and sim_var > best_sim:
                    best_type = var_code
                    best_sim = sim_var

        if best_type is None:
            continue

        # Comprobar que no hay ya una aparición muy cercana del mismo tipo
        too_close = any(
            a["bar"] == bar or
            (a["type"] == best_type and abs(a["bar"] - bar) < MIN_DISTANCE_BARS)
            for a in appearances
        )
        if too_close:
            # Actualizar si la nueva similitud es mejor
            for a in appearances:
                if a["bar"] == bar and a["similarity"] < best_sim:
                    a["type"] = best_type
                    a["type_name"] = VARIATION_NAMES.get(best_type, best_type)
                    a["similarity"] = round(best_sim, 3)
            continue

        type_name = VARIATION_NAMES.get(best_type, best_type)
        appearances.append({
            "bar":        bar,
            "type":       best_type,
            "type_name":  type_name,
            "similarity": round(best_sim, 3),
            "onset_tick": onset_tick,
            "note_idx":   start_idx,
        })

        if verbose:
            print(f"    [{lm_id}] compás {bar:3d}: {type_name:<18} sim={best_sim:.2f}")

    appearances.sort(key=lambda x: x["bar"])
    return appearances


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║               LEITMOTIF ANALYST  v1.0                           ║")
    print("║  Identificación de leitmotifs y variaciones en obra MIDI        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")


def print_leitmotif_summary(leitmotif: dict, appearances: list):
    """Imprime resumen de un leitmotif con sus apariciones."""
    lm = leitmotif
    total_app = len(appearances)
    type_counts = Counter(a["type"] for a in appearances)

    print(f"\n  ┌─ {lm['id']}  ({lm['n_notes']} notas | {lm['repetitions']} repeticiones Re-Pair | "
          f"primera aparición: compás {lm['bar_start']})")
    print(f"  │  Contorno: {lm['interval_pattern'][:8]}")

    if total_app == 0:
        print(f"  │  Sin apariciones detectadas.")
    else:
        print(f"  │  Total apariciones rastreadas: {total_app}")
        for t, cnt in sorted(type_counts.items(),
                              key=lambda x: (0 if x[0] == "original" else 1, x[0])):
            name = VARIATION_NAMES.get(t, t)
            bars_str = ", ".join(
                str(a["bar"]) for a in appearances if a["type"] == t
            )
            print(f"  │    {name:<20} ×{cnt:2d}  →  compases: {bars_str}")

    print(f"  └{'─'*65}")


def print_ascii_map(leitmotifs: list, all_appearances: dict,
                     n_bars: int, bars_per_col: int = 1):
    """
    Mapa ASCII de presencia de leitmotifs a lo largo de la obra.
    Filas = leitmotifs (+ fila por cada variación presente).
    Columnas = grupos de `bars_per_col` compases.
    """
    cpg = max(1, bars_per_col)   # compases por grupo / columna
    n_groups = max(1, (n_bars + cpg - 1) // cpg)

    # Ancho de celda: lo suficiente para la etiqueta del compás más un espacio
    # mínimo 3 caracteres para que quepan los bloques ██
    max_bar_label = str((n_groups - 1) * cpg + 1)
    cell_w = max(3, len(max_bar_label) + 1)
    cell_fmt  = f"{{:^{cell_w}}}"    # para etiquetas de cabecera
    empty_cell = " " * cell_w        # celda vacía
    sep_cell   = "─" * cell_w        # separador horizontal

    # Bloques de similitud escalados al ancho de celda
    def _cell(sim: float) -> str:
        if sim >= 0.80:
            inner = "██"
        elif sim >= 0.65:
            inner = "▓▓"
        else:
            inner = "░░"
        return inner.center(cell_w)

    print(f"\n{'═' * 68}")
    unit = "compás" if cpg == 1 else f"{cpg} compases"
    print(f"  MAPA DE APARICIONES  (cada columna = {unit})")
    print(f"{'═' * 68}")

    # Cabecera con número de compás de inicio de cada columna
    header = f"  {'ID':<8} | {'Tipo':<6} |"
    for g in range(n_groups):
        bar_label = str(g * cpg + 1)
        header += cell_fmt.format(bar_label) + "|"
    print(header)
    print(f"  {'-'*8}-+-{'-'*6}-+" + f"{sep_cell}+" * n_groups)

    for lm in leitmotifs:
        appearances = all_appearances.get(lm["id"], [])

        # Agrupar apariciones por columna y tipo
        row_vars: dict[str, list] = defaultdict(lambda: [None] * n_groups)
        for a in appearances:
            g = (a["bar"] - 1) // cpg
            if g < 0 or g >= n_groups:
                continue
            t = a["type"]
            # Guardar la de mayor similitud si hay varias en el mismo grupo
            prev = row_vars[t][g]
            if prev is None or a["similarity"] > prev:
                row_vars[t][g] = a["similarity"]

        # Fila del original
        row_orig = f"  {lm['id']:<8} | {'orig':<6} |"
        for g in range(n_groups):
            sim = row_vars["original"][g]
            row_orig += (_cell(sim) if sim is not None else empty_cell) + "|"
        print(row_orig)

        # Filas de variaciones presentes (al menos una celda no vacía)
        present_vars = sorted(
            t for t in row_vars
            if t != "original" and any(v is not None for v in row_vars[t])
        )
        for var_code in present_vars:
            row_v = f"  {'':8} | {var_code:<6} |"
            for g in range(n_groups):
                sim = row_vars[var_code][g]
                row_v += (_cell(sim) if sim is not None else empty_cell) + "|"
            print(row_v)

        if len(leitmotifs) > 1:
            print(f"  {'-'*8}-+-{'-'*6}-+" + f"{sep_cell}+" * n_groups)

    print(f"\n  Leyenda: ██ sim≥0.80  ▓▓ sim≥0.65  ░░ sim≥umbral")


def print_detailed_list(leitmotifs: list, all_appearances: dict):
    """Lista detallada de apariciones por compás."""
    print(f"\n{'═' * 68}")
    print(f"  LISTA DETALLADA DE APARICIONES POR COMPÁS")
    print(f"{'═' * 68}")
    for lm in leitmotifs:
        appearances = all_appearances.get(lm["id"], [])
        if not appearances:
            continue
        print(f"\n  {lm['id']}:")
        for a in appearances:
            marker = "●" if a["type"] == "original" else "◌"
            print(f"    {marker} compás {a['bar']:3d}  {a['type_name']:<20} sim={a['similarity']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def analyse(midi_path: str,
             leitmotif_midi: str = None,
             min_bars: int = 1,
             max_bars: int = 8,
             threshold: float = 0.55,
             var_threshold: float = 0.50,
             top_n: int = 8,
             map_cols: int = 1,
             report: bool = False,
             save_leitmotifs: bool = False,
             save_all: bool = False,
             save_dir: str = None,
             verbose: bool = False) -> dict:
    """
    Análisis completo de leitmotifs en un MIDI.

    Si se proporciona leitmotif_midi, omite la detección automática y busca
    únicamente ese motivo (y sus variaciones) en la obra.

    Devuelve el resultado como dict (también lo imprime en consola).
    """
    print_banner()
    print(f"\n  Archivo   : {os.path.basename(midi_path)}")
    if leitmotif_midi:
        print(f"  Leitmotif : {os.path.basename(leitmotif_midi)}  [externo]")
    print(f"  Umbral orig: {threshold}  |  umbral var: {var_threshold}  "
          f"|  top leitmotifs: {top_n if not leitmotif_midi else '—'}")

    # ── 1. Cargar MIDI ────────────────────────────────────────────────────────
    print(f"\n[1/3] Cargando MIDI...")
    try:
        notes, tpb, total_ticks, bar_ticks, tempo_bpm, time_sig, tempo_us = \
            extract_notes(midi_path)
    except Exception as e:
        print(f"ERROR al cargar {midi_path}: {e}")
        sys.exit(1)

    if not notes:
        print("ERROR: No se encontraron notas melódicas en el MIDI.")
        sys.exit(1)

    ts_num, ts_den = time_sig
    n_bars = max(1, int(total_ticks / bar_ticks))
    print(f"  ✓ {len(notes)} notas  |  {tempo_bpm} BPM  |  {ts_num}/{ts_den}  "
          f"|  ~{n_bars} compases  |  tpb={tpb}")

    # ── 2. Identificar leitmotifs ─────────────────────────────────────────────
    if leitmotif_midi:
        # Modo externo: usar el MIDI proporcionado como único leitmotif
        print(f"\n[2/3] Cargando leitmotif externo...")
        try:
            ext_lm = leitmotif_from_midi(leitmotif_midi, tpb)
        except Exception as e:
            print(f"ERROR al cargar el leitmotif: {e}")
            sys.exit(1)
        leitmotifs = [ext_lm]
        print(f"  ✓ Leitmotif cargado: {ext_lm['n_notes']} notas  "
              f"contorno: {ext_lm['interval_pattern'][:8]}")
    else:
        # Modo automático: detección con Re-Pair
        print(f"\n[2/3] Identificando leitmotifs...")
        leitmotifs = extract_leitmotifs(
            notes, tpb, bar_ticks, total_ticks,
            min_bars=min_bars, max_bars=max_bars,
            top_n=top_n, verbose=verbose
        )
        if not leitmotifs:
            print("  No se encontraron leitmotifs repetidos en esta obra.")
            return {"leitmotifs": [], "appearances": {}}
        print(f"  ✓ {len(leitmotifs)} leitmotifs identificados")
        for lm in leitmotifs:
            print(f"    {lm['id']}  {lm['n_notes']} notas  "
                  f"×{lm['repetitions']} repeticiones  "
                  f"contorno: {lm['interval_pattern'][:6]}")

    # ── 3. Rastrear apariciones y variaciones ─────────────────────────────────
    print(f"\n[3/3] Rastreando apariciones y variaciones...")
    all_appearances = {}
    for lm in leitmotifs:
        if verbose:
            print(f"  → Rastreando {lm['id']}...")
        aprs = trace_leitmotif(
            lm, notes, tpb, bar_ticks, total_ticks,
            threshold=threshold,
            var_threshold=var_threshold,
            verbose=verbose,
        )
        all_appearances[lm["id"]] = aprs
        n_orig = sum(1 for a in aprs if a["type"] == "original")
        n_var  = len(aprs) - n_orig
        print(f"  {lm['id']}: {len(aprs)} apariciones  "
              f"({n_orig} originales + {n_var} variaciones)")

    # ── Presentación ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print(f"  LEITMOTIFS Y SUS APARICIONES")
    print(f"{'═' * 68}")
    for lm in leitmotifs:
        print_leitmotif_summary(lm, all_appearances.get(lm["id"], []))

    print_ascii_map(leitmotifs, all_appearances, n_bars, bars_per_col=map_cols)
    print_detailed_list(leitmotifs, all_appearances)

    print(f"\n{'═' * 68}")
    total_aprs = sum(len(v) for v in all_appearances.values())
    print(f"  RESUMEN FINAL")
    print(f"  Leitmotifs identificados : {len(leitmotifs)}")
    print(f"  Apariciones totales      : {total_aprs}")
    type_totals = Counter()
    for aprs in all_appearances.values():
        for a in aprs:
            type_totals[a["type_name"]] += 1
    for t_name, cnt in sorted(type_totals.items(),
                               key=lambda x: (-x[1], x[0])):
        print(f"    {t_name:<22}: {cnt}")
    print(f"{'═' * 68}\n")

    # ── Reporte JSON ──────────────────────────────────────────────────────────
    result = {
        "version":   VERSION,
        "generated": datetime.now().isoformat(),
        "midi":      os.path.abspath(midi_path),
        "n_bars":    n_bars,
        "tempo_bpm": tempo_bpm,
        "time_sig":  f"{ts_num}/{ts_den}",
        "parameters": {
            "threshold":       threshold,
            "var_threshold":   var_threshold,
            "top_n":           top_n,
            "min_bars":        min_bars,
            "max_bars":        max_bars,
            "leitmotif_midi":  leitmotif_midi,
        },
        "leitmotifs": [
            {
                "id":               lm["id"],
                "n_notes":          lm["n_notes"],
                "bar_start":        lm["bar_start"],
                "repetitions":      lm["repetitions"],
                "interval_pattern": lm["interval_pattern"],
                "dur_pattern":      lm.get("dur_pattern", []),
                "source":           lm.get("source", "auto"),
                "fingerprint": {
                    "contour": lm["fingerprint"]["contour"],
                    "rhythm":  lm["fingerprint"]["rhythm"],
                },
                "appearances": all_appearances.get(lm["id"], []),
            }
            for lm in leitmotifs
        ],
    }

    if report:
        stem = Path(midi_path).stem
        report_path = Path(midi_path).parent / f"{stem}.leitmotif_analysis.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Reporte JSON guardado: {report_path}")

    # ── Exportar MIDIs ────────────────────────────────────────────────────────
    if save_leitmotifs or save_all:
        stem    = Path(midi_path).stem
        out_dir = save_dir or f"{stem}_leitmotifs"

        if leitmotif_midi:
            # Modo externo: exportar original + todas las variaciones generadas
            saved_files = export_transformed_midis(
                leitmotif=leitmotifs[0],
                midi_path=midi_path,
                tpb=tpb,
                tempo_us=tempo_us,
                ts_num=ts_num,
                ts_den=ts_den,
                save_dir=out_dir,
                verbose=verbose,
            )
            # Si además se pidió --save-all, también exportar las apariciones
            if save_all:
                app_files = export_midis(
                    leitmotifs=leitmotifs,
                    all_appearances=all_appearances,
                    notes_all=notes,
                    midi_path=midi_path,
                    tpb=tpb,
                    tempo_us=tempo_us,
                    ts_num=ts_num,
                    ts_den=ts_den,
                    save_dir=out_dir,
                    save_all=True,
                    verbose=verbose,
                )
                result["exported_transformed"] = saved_files
                result["exported_appearances"]  = app_files
            else:
                result["exported_transformed"] = saved_files
        else:
            # Modo automático: comportamiento original
            saved_files = export_midis(
                leitmotifs=leitmotifs,
                all_appearances=all_appearances,
                notes_all=notes,
                midi_path=midi_path,
                tpb=tpb,
                tempo_us=tempo_us,
                ts_num=ts_num,
                ts_den=ts_den,
                save_dir=out_dir,
                save_all=save_all,
                verbose=verbose,
            )
            result["exported_files"] = saved_files

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description="LEITMOTIF ANALYST — Identifica leitmotifs y sus variaciones en una obra MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Detección automática
  python leitmotif_analyst.py beethoven.mid
  python leitmotif_analyst.py wagner.mid --top 6 --report
  python leitmotif_analyst.py obra.mid --min-bars 1 --max-bars 4 --verbose

  # Leitmotif externo: buscar solo el motivo dado
  python leitmotif_analyst.py obra.mid --leitmotif destino.mid
  python leitmotif_analyst.py obra.mid --leitmotif destino.mid --threshold 0.60

  # Guardar solo la forma fundamental (o, con --leitmotif, original + variaciones):
  python leitmotif_analyst.py obra.mid --save-leitmotifs
  python leitmotif_analyst.py obra.mid --leitmotif motivo.mid --save-leitmotifs

  # Guardar forma fundamental + apariciones individuales en la obra:
  python leitmotif_analyst.py obra.mid --save-all
  python leitmotif_analyst.py obra.mid --leitmotif motivo.mid --save-all

  # Especificar carpeta de destino:
  python leitmotif_analyst.py obra.mid --save-all --save-dir ./mis_motivos

  # Controlar la resolución del mapa:
  python leitmotif_analyst.py obra.mid --map-cols 1   # una columna por compás (default)
  python leitmotif_analyst.py obra.mid --map-cols 4   # agrupar de 4 en 4
        """
    )
    p.add_argument("midi",             help="Archivo MIDI de la obra a analizar")
    p.add_argument("--leitmotif",      default=None, metavar="FILE",
                   help=("MIDI con el leitmotif de referencia. Si se proporciona, "
                         "omite la detección automática y busca solo ese motivo "
                         "(y sus variaciones) en la obra"))
    p.add_argument("--min-bars",       type=int,   default=1,
                   help="Longitud mínima de leitmotif en compases (default: 1)")
    p.add_argument("--max-bars",       type=int,   default=8,
                   help="Longitud máxima de leitmotif en compases (default: 8)")
    p.add_argument("--threshold",      type=float, default=0.55,
                   help="Umbral de similitud para forma original (default: 0.55)")
    p.add_argument("--var-threshold",  type=float, default=0.50,
                   help="Umbral de similitud para variaciones (default: 0.50)")
    p.add_argument("--top",            type=int,   default=8,
                   help="Número máximo de leitmotifs en modo automático (default: 8)")
    p.add_argument("--report",         action="store_true",
                   help="Guardar reporte JSON con el análisis completo")
    p.add_argument("--map-cols",       type=int, default=1, metavar="N",
                   help=("Compases por columna en el mapa de apariciones "
                         "(default: 1 — una columna por compás)"))

    # ── Opciones de exportación MIDI ─────────────────────────────────────────
    save_group = p.add_argument_group("exportación MIDI")
    save_group.add_argument(
        "--save-leitmotifs", action="store_true",
        help=("Guardar la forma fundamental de cada leitmotif como MIDI. "
              "Con --leitmotif, exporta también todas las variaciones generadas (V01–V09)")
    )
    save_group.add_argument(
        "--save-all", action="store_true",
        help=("Guardar forma fundamental + un MIDI por cada aparición detectada. "
              "Con --leitmotif, añade además los MIDIs de variaciones generadas")
    )
    save_group.add_argument(
        "--save-dir", default=None, metavar="DIR",
        help=("Carpeta de destino para los MIDIs exportados "
              "(default: <nombre_obra>_leitmotifs/)")
    )

    p.add_argument("--verbose",        action="store_true",
                   help="Salida detallada")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not Path(args.midi).exists():
        print(f"ERROR: No se encontró la obra: {args.midi}")
        sys.exit(1)

    if args.leitmotif and not Path(args.leitmotif).exists():
        print(f"ERROR: No se encontró el leitmotif: {args.leitmotif}")
        sys.exit(1)

    # --save-all implica --save-leitmotifs
    save_leitmotifs = args.save_leitmotifs or args.save_all

    analyse(
        midi_path=args.midi,
        leitmotif_midi=args.leitmotif,
        min_bars=args.min_bars,
        max_bars=args.max_bars,
        threshold=args.threshold,
        var_threshold=args.var_threshold,
        top_n=args.top,
        map_cols=args.map_cols,
        report=args.report,
        save_leitmotifs=save_leitmotifs,
        save_all=args.save_all,
        save_dir=args.save_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
