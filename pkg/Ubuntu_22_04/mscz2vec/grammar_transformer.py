#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    GRAMMAR TRANSFORMER  v2.0                                 ║
║         Transformación estructural de MIDI mediante gramática Re-Pair        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DESCRIPCIÓN:                                                                ║
║    Analiza un MIDI de entrada y extrae su gramática intrínseca a nivel de   ║
║    nota mediante el algoritmo Re-Pair (cada token hoja es una nota con su   ║
║    duración en beats). Calcula matrices de compatibilidad musical entre      ║
║    los elementos de la gramática y genera un nuevo MIDI sustituyendo los    ║
║    elementos de la longitud indicada por otros compatibles, sampleados con  ║
║    una temperatura configurable.                                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PIPELINE                                                                    ║
║    1. Extrae notas del MIDI → secuencia de tokens (pitch, duración)         ║
║    2. Re-Pair aprende reglas jerárquicas N1, N2 … sobre esa secuencia       ║
║    3. Cada regla se expande a sus notas y se vectoriza (47 dims)            ║
║    4. Se construye una matriz de compatibilidad coseno por longitud          ║
║    5. Los elementos de la longitud objetivo se sustituyen por sampling       ║
║    6. Se reconstruye la secuencia plana y se escribe el MIDI de salida      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODOS                                                                       ║
║                                                                              ║
║  ── TRANSFORM (por defecto) ───────────────────────────────────────────── ║
║    Requiere --length. Sustituye los elementos de esa duración por otros     ║
║    compatibles sampleados con temperatura T.                                 ║
║      T=0   → greedy: siempre el más compatible (mínima variación)          ║
║      T=0.5 → equilibrio entre fidelidad y variedad (recomendado)            ║
║      T=1   → softmax uniforme: máxima variedad respetando compatibilidad   ║
║      T>1   → distribución más plana que uniforme (exploración extrema)      ║
║                                                                              ║
║  ── BINS ──────────────────────────────────────────────────────────────── ║
║    Requiere --bin-size. Agrupa todos los elementos en intervalos de ese     ║
║    ancho (en beats) y para cada elemento elige un reemplazo de su mismo    ║
║    bin, independientemente de la longitud exacta. La diferencia de         ║
║    duración entre el elemento elegido y el hueco destino se resuelve con   ║
║    --fit:                                                                    ║
║      scale  — escala proporcional todas las duraciones al hueco destino    ║
║      pad    — añade notas de paso cromáticas para completar el hueco        ║
║      trim   — recorta el elemento al final para ajustar al hueco            ║
║                                                                              ║
║  ── REPORT ────────────────────────────────────────────────────────────── ║
║    Muestra estadísticas de la gramática sin transformar el MIDI.            ║
║    Con --report-format html genera un visor interactivo en el navegador.    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  EJEMPLOS DE USO                                                             ║
║                                                                              ║
║  · Exploración inicial — ver la gramática antes de transformar:             ║
║      python grammar_transformer.py obra.mid --report                        ║
║      python grammar_transformer.py obra.mid --report --report-format html   ║
║      python grammar_transformer.py obra.mid --report --report-format both   ║
║                                                                              ║
║  · Ver la gramática y la matriz de compatibilidad para 1 beat:              ║
║      python grammar_transformer.py obra.mid --report --length 1.0           ║
║                                                                              ║
║  · Transformación básica — sustituir motivos de 1 beat, T moderada:        ║
║      python grammar_transformer.py obra.mid --length 1.0                    ║
║      python grammar_transformer.py obra.mid --length 1.0 -o variacion.mid  ║
║                                                                              ║
║  · Modo bins — agrupar por intervalos de 2 beats y escalar al insertar:    ║
║      python grammar_transformer.py obra.mid --bin-size 2.0                  ║
║      python grammar_transformer.py obra.mid --bin-size 2.0 --fit scale     ║
║        # escala proporcional (default)                                       ║
║      python grammar_transformer.py obra.mid --bin-size 2.0 --fit pad       ║
║        # rellena con notas de paso cromáticas                               ║
║      python grammar_transformer.py obra.mid --bin-size 2.0 --fit trim      ║
║        # recorta el final del elemento insertado                            ║
║      python grammar_transformer.py obra.mid --bin-size 1.0 --fit pad       ║
║        --temp 0.8 --pitch-norm -o variacion.mid                            ║
║      python grammar_transformer.py obra.mid --bin-size 2.0                 ║
║        --bin-length 1.0 4.0  # solo transforma elementos entre 1 y 4 beats ║
║      python grammar_transformer.py obra.mid --bin-size 1.0 --fit scale     ║
║        --bin-length 0.5 2.0 --temp 0.5 -o variacion.mid                   ║
║                                                                              ║
║  · Controlar la variación con temperatura:                                  ║
║      python grammar_transformer.py obra.mid --length 1.0 --temp 0.0        ║
║        # greedy: solo intercambia si hay algo más compatible                ║
║      python grammar_transformer.py obra.mid --length 1.0 --temp 0.5        ║
║        # equilibrado (default)                                               ║
║      python grammar_transformer.py obra.mid --length 1.0 --temp 1.0        ║
║        # máxima variedad dentro de los candidatos compatibles               ║
║                                                                              ║
║  · Transformación invariante a tonalidad (normaliza pitch al comparar       ║
║    y retranspone el elemento insertado al registro original):               ║
║      python grammar_transformer.py obra.mid --length 2.0 --pitch-norm      ║
║                                                                              ║
║  · Gramática más granular con --max-rules (más candidatos por longitud,     ║
║    transformaciones más ricas):                                              ║
║      python grammar_transformer.py obra.mid --length 0.5 --max-rules 20    ║
║                                                                              ║
║  · Resultado reproducible con semilla fija:                                 ║
║      python grammar_transformer.py obra.mid --length 1.0 --seed 42         ║
║                                                                              ║
║  · Exportar los fragmentos de la gramática para editarlos o reutilizarlos  ║
║    en otras obras:                                                           ║
║      python grammar_transformer.py obra.mid --export-elements ./frags/      ║
║        # genera un JSON por elemento en ./frags/ (default)                  ║
║      python grammar_transformer.py obra.mid --export-elements ./frags/      ║
║        --export-format midi   # solo MIDIs, uno por elemento                ║
║      python grammar_transformer.py obra.mid --export-elements ./frags/      ║
║        --export-format both   # JSON + MIDI por elemento                    ║
║                                                                              ║
║  · Importar fragmentos externos (de otra obra o variantes manuales)         ║
║    para ampliar el pool de candidatos antes de transformar:                 ║
║      python grammar_transformer.py obra.mid --length 2.0                   ║
║        --import-elements ./frags_otra_obra/                                 ║
║        # acepta *.json, *.mid o mezcla de ambos en el mismo directorio     ║
║                                                                              ║
║  · Flujo completo — exportar, enriquecer e importar:                        ║
║      python grammar_transformer.py obraA.mid --export-elements ./pool/      ║
║      python grammar_transformer.py obraB.mid --export-elements ./pool/      ║
║        # editar JSONs en ./pool/ si se desean variantes manuales            ║
║      python grammar_transformer.py obraA.mid --length 1.0 --temp 0.5       ║
║        --import-elements ./pool/ -o obraA_enriquecida.mid                  ║
║                                                                              ║
║  · Combinar reporte + transformación en un solo paso:                       ║
║      python grammar_transformer.py obra.mid --length 1.0 --temp 0.5        ║
║        --report --report-format both -o salida.mid                          ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OPCIONES                                                                    ║
║    --length / -l BEATS    Longitud en beats de los elementos a transformar  ║
║    --bin-size BEATS       Ancho del bin (activa modo bins)                  ║
║    --bin-length MIN MAX   Rango de duraciones a transformar en modo bins    ║
║                           (beats). Elementos fuera quedan intactos.         ║
║    --fit MODE             Ajuste al insertar: scale | pad | trim (def: scale)║
║    --temp / -t FLOAT      Temperatura de sampling (default: 0.5)            ║
║    --pitch-norm           Normalizar pitch al comparar y transponer al      ║
║                           insertar (compatibilidad invariante a tonalidad)  ║
║    --report               Activar modo informe                              ║
║    --report-format FMT    text | html | both  (default: text)              ║
║    --export-elements DIR  Exportar elementos de la gramática a directorio   ║
║    --export-format FMT    Formato de exportación: json, midi, both (def: json)║
║    --import-elements DIR  Importar elementos adicionales desde directorio   ║
║                           Acepta *.json y *.mid (se pueden mezclar).        ║
║                           Si hay JSON y MIDI con el mismo nombre, el JSON   ║
║                           tiene prioridad (evita duplicados).               ║
║    --min-pair N           Frecuencia mínima para crear regla (default: 2)   ║
║    --max-rules N          Máximo de reglas a generar (default: ilimitado)   ║
║    --track N              Track del MIDI a usar (default: auto)             ║
║    --out / -o FILE        Fichero MIDI de salida                            ║
║    --seed FLOAT           Semilla aleatoria para reproducibilidad           ║
║    --verbose              Traza detallada del proceso                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  VECTOR DE COMPATIBILIDAD (47 dimensiones)                                   ║
║    · Clases de pitch — histograma de las 12 notas cromáticas (12 dims)     ║
║    · Intervalos melódicos — histograma [-12, +12] semítonos  (25 dims)     ║
║    · Contorno melódico — pendiente, varianza, rango, ratio↑  ( 4 dims)     ║
║    · Perfil rítmico — densidad, dur. media/std, ratio ♪/♩/♩♩ ( 6 dims)     ║
║    Similitud = coseno entre vectores. Con --pitch-norm los pitches se       ║
║    normalizan al Do central antes de calcular el vector.                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS: mido, numpy, scipy                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import re
import json
import math
import copy
import random
import argparse
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, defaultdict

import numpy as np
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "2.0"
TICKS_EPSILON = 1

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
INTERVAL_TENSION = {0: 0.0, 1: 1.0, 2: 0.4, 3: 0.6, 4: 0.2, 5: 0.2,
                    6: 0.9, 7: 0.1, 8: 0.5, 9: 0.3, 10: 0.7, 11: 0.8}



# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GrammarNote:
    """Nota a nivel de gramática: pitch MIDI + duración en beats."""
    pitch: int          # MIDI 0-127
    duration: float     # en beats (negras)
    velocity: int = 80
    channel: int = 0
    offset: float = 0.0  # offset dentro del elemento (en beats)


@dataclass
class GrammarElement:
    """
    Elemento de la gramática: secuencia de notas con su duración total.
    Puede ser una hoja (notes directas) o un nodo interno (refs a subelementos).
    """
    name: str
    notes: list = field(default_factory=list)  # lista de GrammarNote
    duration_beats: float = 0.0
    occurrences: int = 0
    vector: Optional[np.ndarray] = None        # vector de compatibilidad
    source_bars: list = field(default_factory=list)  # compases de origen


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def get_tempo(mid: MidiFile) -> int:
    """Devuelve el primer tempo encontrado (µs/negra) o 500 000 (120 BPM)."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500_000


def choose_main_track(mid: MidiFile) -> int:
    """Elige el track con más note_on como track principal."""
    best_idx, best_count = 0, -1
    for i, track in enumerate(mid.tracks):
        count = sum(1 for m in track if m.type == 'note_on' and m.velocity > 0)
        if count > best_count:
            best_count, best_idx = count, i
    return best_idx


def get_time_signatures(mid: MidiFile) -> list:
    """Extrae lista de (abs_tick, numerator, denominator) del track 0."""
    sigs = [(0, 4, 4)]
    abs_t = 0
    for msg in mid.tracks[0]:
        abs_t += msg.time
        if msg.type == 'time_signature':
            sigs.append((abs_t, msg.numerator, msg.denominator))
    sigs.sort(key=lambda x: x[0])
    return sigs


def ticks_per_beat_at(tick: int, time_sigs: list, tpb: int) -> float:
    """Devuelve los beats por compás en el momento 'tick'."""
    num, denom = 4, 4
    for ts_tick, ts_num, ts_denom in time_sigs:
        if ts_tick <= tick:
            num, denom = ts_num, ts_denom
    return tpb * num * 4 / denom


def extract_notes_flat(mid: MidiFile, track_idx: int) -> list:
    """
    Extrae todas las notas del track como lista de GrammarNote con offset
    absoluto en beats desde el inicio de la pieza.

    Devuelve [(abs_beat_offset, GrammarNote), ...]
    """
    tpb = mid.ticks_per_beat
    time_sigs = get_time_signatures(mid)

    track = mid.tracks[track_idx]
    abs_t = 0
    events = []
    for msg in track:
        abs_t += msg.time
        events.append((abs_t, msg))

    # Recolectar time_sigs también del track principal si no es el 0
    if track_idx != 0:
        abs_t2 = 0
        for msg in mid.tracks[0]:
            abs_t2 += msg.time
            if msg.type == 'time_signature':
                time_sigs.append((abs_t2, msg.numerator, msg.denominator))
        time_sigs.sort(key=lambda x: x[0])

    def ticks_to_beats(t: int) -> float:
        return t / tpb

    # Extraer notas con note_on/note_off
    active: dict = {}   # (channel, pitch) -> [(abs_tick, velocity)]
    raw: list = []      # (start_tick, end_tick, pitch, velocity, channel)

    for abs_t, msg in events:
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (msg.channel, msg.note)
            active.setdefault(key, []).append((abs_t, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active and active[key]:
                start_t, vel = active[key].pop(0)
                raw.append((start_t, abs_t, msg.note, vel, msg.channel))

    result = []
    for start_t, end_t, pitch, vel, ch in sorted(raw, key=lambda x: x[0]):
        offset_beats = ticks_to_beats(start_t)
        dur_beats = ticks_to_beats(end_t - start_t)
        note = GrammarNote(
            pitch=pitch,
            duration=max(dur_beats, 1/64),
            velocity=vel,
            channel=ch,
            offset=offset_beats,
        )
        result.append((offset_beats, note))

    return result


def notes_to_midi_file(
    notes: list,          # [(abs_offset_beats, GrammarNote), ...]
    tpb: int,
    tempo: int,
    out_path: Path,
) -> None:
    """Escribe una lista plana de notas a un fichero MIDI."""
    mid = MidiFile(ticks_per_beat=tpb)
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

    events = []
    for abs_beat, note in notes:
        abs_tick = int(abs_beat * tpb)
        dur_tick = max(1, int(note.duration * tpb))
        events.append((abs_tick,            'on',  note.pitch, note.velocity, note.channel))
        events.append((abs_tick + dur_tick,  'off', note.pitch, 0,            note.channel))

    events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

    cursor = 0
    for abs_tick, kind, pitch, vel, ch in events:
        delta = abs_tick - cursor
        cursor = abs_tick
        mtype = 'note_on'
        track.append(Message(mtype, channel=ch, note=pitch, velocity=vel, time=delta))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))


def extract_all_tracks_flat(mid: MidiFile) -> dict:
    """Extrae notas de TODOS los tracks con notas. Devuelve {track_idx: notes_flat}."""
    result = {}
    for i, track in enumerate(mid.tracks):
        if any(m.type == 'note_on' and m.velocity > 0 for m in track):
            nf = extract_notes_flat(mid, i)
            if nf:
                result[i] = nf
    return result


def multitrack_to_midi(
    tracks_notes: dict,
    tpb: int,
    tempo: int,
    out_path: Path,
    original_mid: MidiFile | None = None,
) -> None:
    """
    Escribe {track_idx: [(abs_beat, GrammarNote),...]} a MIDI multipista.

    Si original_mid se proporciona, replica su estructura de tracks (número,
    nombres, presencia de track de tempo) para máxima compatibilidad con el
    DAW de origen.
    """
    # Detectar si el original tiene track 0 dedicado a tempo (sin notas)
    has_dedicated_tempo_track = (
        original_mid is not None
        and not any(
            msg.type == 'note_on'
            for msg in original_mid.tracks[0]
        )
    ) if (original_mid is not None and original_mid.tracks) else True

    mid = MidiFile(type=1, ticks_per_beat=tpb)

    if has_dedicated_tempo_track:
        t0 = MidiTrack()
        mid.tracks.append(t0)
        t0.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        t0.append(MetaMessage('end_of_track', time=0))

    for orig_idx in sorted(tracks_notes.keys()):
        notes = tracks_notes[orig_idx]
        track = MidiTrack()

        # Preservar nombre y tempo inline si el original no tiene track dedicado
        if not has_dedicated_tempo_track:
            track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
        if original_mid is not None and orig_idx < len(original_mid.tracks):
            orig_name = original_mid.tracks[orig_idx].name
            if orig_name:
                track.append(MetaMessage('track_name', name=orig_name, time=0))

        mid.tracks.append(track)

        events = []
        for abs_beat, note in notes:
            abs_tick = int(abs_beat * tpb)
            dur_tick = max(1, int(note.duration * tpb))
            events.append((abs_tick, 'on',  note.pitch, note.velocity, note.channel))
            events.append((abs_tick + dur_tick, 'off', note.pitch, 0, note.channel))

        events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))
        cursor = 0
        for abs_tick, _, pitch, vel, ch in events:
            delta = abs_tick - cursor
            cursor = abs_tick
            track.append(Message('note_on', channel=ch, note=pitch, velocity=vel, time=delta))
        track.append(MetaMessage('end_of_track', time=0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(out_path))


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE GRAMÁTICA A NIVEL DE NOTA
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_duration(dur_beats: float, grid: float = 1/16) -> float:
    """Cuantiza una duración al grid indicado (en beats)."""
    if dur_beats <= 0:
        return grid
    return max(grid, round(dur_beats / grid) * grid)


def note_token(note: GrammarNote, quantize: bool = True) -> tuple:
    """Crea un token hashable para una nota (pitch, dur_cuantizado)."""
    dur = quantize_duration(note.duration) if quantize else note.duration
    return (note.pitch, dur)


def extract_note_sequence(notes_flat: list) -> list:
    """
    Transforma la lista plana de (offset, GrammarNote) en una secuencia
    de tokens (pitch, dur_cuantizado) apta para Re-Pair.
    """
    return [note_token(n) for _, n in notes_flat]


def extract_chord_sequence(notes_flat: list, onset_tol: float = 0.05) -> tuple:
    """
    Agrupa las notas de notes_flat por onset simultáneo (tolerancia onset_tol beats)
    y devuelve (sequence, chord_map) donde:
        sequence  = [chord_token_int, ...]  — un int opaco por posición de acorde
        chord_map = {token: [GrammarNote, ...]}  — notas del acorde con offsets
                                                   relativos al inicio del acorde

    Dos acordes con exactamente las mismas notas (pitch + dur cuantizada) comparten
    el mismo token. Acordes de una sola nota también son válidos.
    """
    if not notes_flat:
        return [], {}

    # Agrupar por onset
    onset_groups: dict = defaultdict(list)   # onset_beat → [GrammarNote]
    for off, note in sorted(notes_flat, key=lambda x: x[0]):
        # Buscar un grupo existente dentro de la tolerancia
        matched = None
        for existing_off in onset_groups:
            if abs(off - existing_off) <= onset_tol:
                matched = existing_off
                break
        key = matched if matched is not None else off
        onset_groups[key].append(GrammarNote(
            pitch=note.pitch,
            duration=note.duration,
            velocity=note.velocity,
            channel=note.channel,
            offset=0.0,   # relativo al acorde: todas las notas empiezan en 0
        ))

    # Ordenar onsets y construir tokens
    def chord_fp(notes):
        return tuple(sorted(
            (n.pitch, quantize_duration(n.duration))
            for n in notes
        ))

    fp_to_token: dict = {}
    next_tok = [0]
    sequence = []
    chord_map: dict = {}       # token → [GrammarNote]
    onset_list = []            # [(onset_beat, token)]

    for onset in sorted(onset_groups):
        notes = onset_groups[onset]
        fp = chord_fp(notes)
        if fp not in fp_to_token:
            tok = next_tok[0]
            next_tok[0] += 1
            fp_to_token[fp] = tok
            chord_map[tok] = notes
        token = fp_to_token[fp]
        sequence.append(token)
        onset_list.append((onset, token))

    return sequence, chord_map, onset_list


def build_grammar_elements_from_chords(
    rules: dict,
    root: list,
    chord_map: dict,
    onset_list: list,
    verbose: bool = False,
) -> tuple:
    """
    Construye GrammarElements desde gramática a nivel de acorde.

    Los tokens hoja (int) de la raíz y los cuerpos de reglas se convierten
    en GrammarElements con nombre propio ('C0', 'C1'…) para que reconstruct_notes
    los encuentre normalmente sin necesitar acceso a chord_map.

    Devuelve (elements, new_rules, new_root) donde new_rules y new_root
    tienen los int reemplazados por los nombres 'C{tok}'.
    """
    memo: dict = {}
    occ: Counter = Counter()
    all_tokens = list(root)
    for body in rules.values():
        all_tokens.extend(body)
    for tok in all_tokens:
        if isinstance(tok, str):
            occ[tok] += 1

    # ── Paso 1: crear GrammarElement para cada acorde hoja único ────────────
    elements: dict = {}
    for tok, notes in chord_map.items():
        cname = f'C{tok}'
        dur = max(n.duration for n in notes) if notes else 0.0
        chord_notes = [GrammarNote(
            pitch=n.pitch, duration=n.duration,
            velocity=n.velocity, channel=n.channel,
            offset=0.0,          # todas simultáneas, base 0
        ) for n in notes]
        elem = GrammarElement(name=cname, notes=chord_notes,
                              duration_beats=round(dur, 6), occurrences=0)
        elements[cname] = elem

    # ── Paso 2: sustituir int → 'C{tok}' en rules y root ───────────────────
    def tok_to_name(tok):
        return f'C{tok}' if isinstance(tok, int) else tok

    new_rules = {
        rname: [tok_to_name(t) for t in body]
        for rname, body in rules.items()
    }
    new_root = [tok_to_name(t) for t in root]

    # Actualizar conteo de ocurrencias
    occ2: Counter = Counter()
    for tok in new_root:
        if isinstance(tok, str):
            occ2[tok] += 1
    for body in new_rules.values():
        for tok in body:
            if isinstance(tok, str):
                occ2[tok] += 1
    for cname in list(elements.keys()):
        elements[cname].occurrences = occ2.get(cname, 0)

    # ── Paso 3: construir GrammarElements para las reglas Re-Pair ───────────
    def expand_to_notes(name, depth=0) -> tuple:
        """Expande devolviendo (notes_with_local_offsets, span_total)."""
        if depth > 128:
            return [], 0.0
        if name in memo:
            return memo[name]

        # Acorde hoja
        if name.startswith('C') and name not in new_rules:
            elem = elements.get(name)
            if elem:
                result = (list(elem.notes), elem.duration_beats)
                memo[name] = result
                return result
            return [], 0.0

        if name not in new_rules:
            return [], 0.0

        body = new_rules[name]
        result_notes, cursor = [], 0.0
        for tok in body:
            sub_notes, sub_span = expand_to_notes(tok, depth + 1)
            for n in sub_notes:
                result_notes.append(GrammarNote(
                    pitch=n.pitch, duration=n.duration,
                    velocity=n.velocity, channel=n.channel,
                    offset=round(cursor + n.offset, 6)))
            cursor += sub_span
        memo[name] = (result_notes, cursor)
        return result_notes, cursor

    for name in new_rules:
        notes, span = expand_to_notes(name)
        elem = GrammarElement(name=name, notes=notes,
                              duration_beats=round(span, 6),
                              occurrences=occ2.get(name, 0))
        elements[name] = elem
        if verbose:
            n_chords = len(set(round(n.offset, 4) for n in notes))
            print(f"  [elem] {name}: {len(notes)} notas ({n_chords} acordes), "
                  f"{span:.3f}b, x{occ2.get(name, 0)}")

    # ── Paso 4: elemento raíz ────────────────────────────────────────────────
    root_notes, cursor = [], 0.0
    for tok in new_root:
        sub_notes, sub_span = expand_to_notes(tok)
        for n in sub_notes:
            root_notes.append(GrammarNote(
                pitch=n.pitch, duration=n.duration,
                velocity=n.velocity, channel=n.channel,
                offset=round(cursor + n.offset, 6)))
        cursor += sub_span
    elements['__root__'] = GrammarElement(
        name='__root__', notes=root_notes,
        duration_beats=round(cursor, 6), occurrences=1)

    return elements, new_rules, new_root


def extract_bar_sequence(notes_flat: list, tpb: int, mid: MidiFile) -> tuple:
    """
    Segmenta notes_flat en compases y devuelve (sequence, bar_notes_map).
    sequence = [bar_token_int, ...]; bar_notes_map = {token: [GrammarNote,...]}
    """
    time_sigs = get_time_signatures(mid)

    def bar_ticks_at(tick: int) -> int:
        num, denom = 4, 4
        for ts_tick, ts_num, ts_denom in time_sigs:
            if ts_tick <= tick:
                num, denom = ts_num, ts_denom
        return int(tpb * num * 4 / denom)

    bar_groups: dict = defaultdict(list)
    for abs_beat, note in notes_flat:
        abs_tick = int(abs_beat * tpb)
        bar, cursor = 1, 0
        while True:
            blen = bar_ticks_at(cursor)
            if cursor + blen > abs_tick:
                break
            cursor += blen
            bar += 1
        bar_start_beat = cursor / tpb
        local_offset = abs_beat - bar_start_beat
        bar_groups[bar].append(GrammarNote(
            pitch=note.pitch, duration=note.duration,
            velocity=note.velocity, channel=note.channel,
            offset=round(local_offset, 6),
        ))

    if not bar_groups:
        return [], {}

    def bar_fp(notes):
        return tuple(sorted(
            (n.pitch, quantize_duration(n.duration), round(n.offset * 16) / 16)
            for n in notes
        ))

    fp_to_token: dict = {}
    next_tok = [0]
    sequence, bar_notes_map = [], {}

    for bar_n in sorted(bar_groups.keys()):
        notes = bar_groups[bar_n]
        fp = bar_fp(notes)
        if fp not in fp_to_token:
            tok = next_tok[0]
            next_tok[0] += 1
            fp_to_token[fp] = tok
            bar_notes_map[tok] = notes
        sequence.append(fp_to_token[fp])

    return sequence, bar_notes_map


def repaint(
    sequence: list,
    min_pair: int = 2,
    max_rules: int | None = None,
    prefix: str = 'N',
    verbose: bool = False,
) -> tuple[dict, list]:
    """
    Algoritmo Re-Pair genérico. prefix controla los nombres ('N' para notas, 'B' para compases).
    """
    rules: dict = {}
    next_id = [1]

    def new_name() -> str:
        while f"{prefix}{next_id[0]}" in rules:
            next_id[0] += 1
        name = f"{prefix}{next_id[0]}"
        next_id[0] += 1
        return name

    current = list(sequence)

    while True:
        if max_rules is not None and len(rules) >= max_rules:
            break

        pairs: Counter = Counter()
        for i in range(len(current) - 1):
            pairs[(current[i], current[i + 1])] += 1

        if not pairs:
            break

        best_pair, count = pairs.most_common(1)[0]
        if count < min_pair:
            break

        rule_name = new_name()
        rules[rule_name] = list(best_pair)

        if verbose:
            print(f"  [repaint] {rule_name} = {list(best_pair)}  (×{count})")

        new_seq = []
        i = 0
        while i < len(current):
            if (i < len(current) - 1
                    and current[i] == best_pair[0]
                    and current[i + 1] == best_pair[1]):
                new_seq.append(rule_name)
                i += 2
            else:
                new_seq.append(current[i])
                i += 1
        current = new_seq

    return rules, current


repaint_notes = repaint  # alias de compatibilidad


def expand_rule(name, rules: dict, memo: dict = None) -> list:
    """
    Expande recursivamente una regla hasta obtener la secuencia de tokens hoja
    (tuplas de nota). Usa memoization.
    """
    if memo is None:
        memo = {}
    if name in memo:
        return memo[name]
    if name not in rules:
        # Es un token hoja (tupla nota)
        return [name]

    result = []
    for tok in rules[name]:
        if isinstance(tok, str):
            result.extend(expand_rule(tok, rules, memo))
        else:
            result.append(tok)
    memo[name] = result
    return result


def compute_element_duration(tokens: list, rules: dict) -> float:
    """
    Calcula la duración total en beats de una secuencia de tokens.
    Los tokens hoja son tuplas (pitch, dur_beats).
    """
    total = 0.0
    for tok in tokens:
        if isinstance(tok, str):
            sub = expand_rule(tok, rules)
            total += sum(dur for _, dur in sub)
        else:
            # tok = (pitch, dur)
            total += tok[1]
    return total


def build_grammar_elements_from_notes(
    rules: dict,
    root: list,
    notes_flat: list,
    verbose: bool = False,
) -> dict:
    """
    Construye GrammarElement para cada regla, calculando:
        - notes: lista de GrammarNote expandidas
        - duration_beats: duración total en beats
        - occurrences: número de veces que aparece en la gramática completa

    También incluye un elemento especial '__root__' para la raíz.

    Devuelve {nombre: GrammarElement}
    """
    memo: dict = {}
    elements: dict = {}

    # Contar ocurrencias de cada regla en toda la gramática (raíz + cuerpos)
    occ: Counter = Counter()
    all_tokens = list(root)
    for body in rules.values():
        all_tokens.extend(body)
    for tok in all_tokens:
        if isinstance(tok, str):
            occ[tok] += 1

    # Construir mapa de offset→nota para recuperar velocidades y canales reales
    pitch_dur_to_notes: dict = defaultdict(list)
    for _, gn in notes_flat:
        key = note_token(gn)
        pitch_dur_to_notes[key].append(gn)

    def token_to_gramnote(tok: tuple, offset_within: float) -> GrammarNote:
        pitch, dur = tok
        candidates = pitch_dur_to_notes.get(tok, [])
        if candidates:
            ref = candidates[0]
            return GrammarNote(pitch=pitch, duration=dur,
                               velocity=ref.velocity, channel=ref.channel,
                               offset=offset_within)
        return GrammarNote(pitch=pitch, duration=dur, offset=offset_within)

    for name, body in rules.items():
        expanded = expand_rule(name, rules, memo)
        cur_offset = 0.0
        gnotes = []
        for tok in expanded:
            gn = token_to_gramnote(tok, cur_offset)
            gnotes.append(gn)
            cur_offset += gn.duration

        elem = GrammarElement(
            name=name,
            notes=gnotes,
            duration_beats=cur_offset,
            occurrences=occ.get(name, 0),
        )
        elements[name] = elem
        if verbose:
            print(f"  [elem] {name}: {len(gnotes)} notas, {cur_offset:.3f} beats, ×{occ.get(name,0)}")

    # Elemento raíz
    root_notes = []
    root_offset = 0.0
    for tok in root:
        if isinstance(tok, str):
            sub = expand_rule(tok, rules, memo)
            for t in sub:
                gn = token_to_gramnote(t, root_offset)
                root_notes.append(gn)
                root_offset += gn.duration
        else:
            gn = token_to_gramnote(tok, root_offset)
            root_notes.append(gn)
            root_offset += gn.duration

    elements['__root__'] = GrammarElement(
        name='__root__',
        notes=root_notes,
        duration_beats=root_offset,
        occurrences=1,
    )

    return elements



def build_grammar_elements_from_bars(
    rules: dict,
    root: list,
    bar_notes_map: dict,
    verbose: bool = False,
) -> dict:
    """Construye GrammarElements desde gramática a nivel de compás."""
    memo: dict = {}
    occ: Counter = Counter()
    all_tokens = list(root)
    for body in rules.values():
        all_tokens.extend(body)
    for tok in all_tokens:
        if isinstance(tok, str):
            occ[tok] += 1

    def bar_span(tok) -> float:
        """
        Span real de un compás: distancia desde el offset de la primera nota
        hasta el fin de la última nota. Preserva los huecos internos del compás
        para que el cursor avance correctamente al encadenar compases.
        """
        raw = bar_notes_map.get(tok, [])
        if not raw:
            return 0.0
        return raw[-1].offset + raw[-1].duration - raw[0].offset

    def tok_to_notes(tok, base_offset: float) -> tuple:
        """
        Convierte un token de compás en notas con offsets relativos a base_offset.
        Los offsets internos del compás se normalizan restando el offset mínimo,
        de modo que la primera nota siempre queda en base_offset (offset=0 local).
        Devuelve (notes, span) donde span es la duración real del compás.
        """
        raw = bar_notes_map.get(tok, [])
        if not raw:
            return [], 0.0
        min_off = raw[0].offset          # offset de la primera nota del compás
        span = raw[-1].offset + raw[-1].duration - min_off
        notes = [GrammarNote(pitch=n.pitch, duration=n.duration,
                             velocity=n.velocity, channel=n.channel,
                             offset=round(base_offset + n.offset - min_off, 6))
                 for n in raw]
        return notes, span

    def expand_to_notes(name, depth=0) -> tuple:
        """
        Expande recursivamente devolviendo (notes, span_total).
        span_total es la duración real del elemento incluyendo huecos internos.
        """
        if depth > 128:
            return [], 0.0
        if name in memo:
            return memo[name]
        if name not in rules:
            if isinstance(name, int):
                result = tok_to_notes(name, 0.0)
                memo[name] = result
                return result
            return [], 0.0
        body = rules[name]
        result_notes, cursor = [], 0.0
        for tok in body:
            if isinstance(tok, str):
                sub_notes, sub_span = expand_to_notes(tok, depth + 1)
                # Rebasar los offsets de las sub-notas al cursor actual
                for n in sub_notes:
                    result_notes.append(GrammarNote(
                        pitch=n.pitch, duration=n.duration,
                        velocity=n.velocity, channel=n.channel,
                        offset=round(cursor + n.offset, 6)))
                cursor += sub_span
            elif isinstance(tok, int):
                bar_notes, bar_sp = tok_to_notes(tok, cursor)
                result_notes.extend(bar_notes)
                cursor += bar_sp
        memo[name] = (result_notes, cursor)
        return result_notes, cursor

    elements: dict = {}
    for name in rules:
        notes, span = expand_to_notes(name)
        elem = GrammarElement(name=name, notes=notes, duration_beats=round(span, 6),
                              occurrences=occ.get(name, 0))
        elements[name] = elem
        if verbose:
            print(f"  [elem] {name}: {len(notes)} notas, {span:.3f}b, x{occ.get(name,0)}")

    root_notes, cursor = [], 0.0
    for tok in root:
        if isinstance(tok, str):
            sub_notes, sub_span = expand_to_notes(tok)
            for n in sub_notes:
                root_notes.append(GrammarNote(
                    pitch=n.pitch, duration=n.duration,
                    velocity=n.velocity, channel=n.channel,
                    offset=round(cursor + n.offset, 6)))
            cursor += sub_span
        elif isinstance(tok, int):
            bar_notes, bar_sp = tok_to_notes(tok, cursor)
            root_notes.extend(bar_notes)
            cursor += bar_sp
    elements['__root__'] = GrammarElement(name='__root__', notes=root_notes,
                                           duration_beats=round(cursor, 6), occurrences=1)
    return elements

# Alias por compatibilidad
build_grammar_elements = build_grammar_elements_from_notes


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZACIÓN DE COMPATIBILIDAD MUSICAL
# ═══════════════════════════════════════════════════════════════════════════════

def pitch_class_histogram(notes: list, normalize_pitch: bool = False) -> np.ndarray:
    """
    Distribución de clases de pitch (12 dimensiones).
    Si normalize_pitch=True, transpone las notas al Do central antes de calcular.
    """
    hist = np.zeros(12)
    if not notes:
        return hist
    pitches = [n.pitch for n in notes]
    if normalize_pitch and pitches:
        mean_pitch = np.mean(pitches)
        transpose = 60 - round(mean_pitch)
        pitches = [p + transpose for p in pitches]
    for p in pitches:
        hist[p % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def interval_histogram(notes: list, normalize_pitch: bool = False) -> np.ndarray:
    """
    Histograma de intervalos melódicos en semítonos [-12, 12], 25 bins.
    """
    hist = np.zeros(25)
    if len(notes) < 2:
        return hist
    pitches = [n.pitch for n in notes]
    if normalize_pitch and pitches:
        mean_pitch = np.mean(pitches)
        transpose = 60 - round(mean_pitch)
        pitches = [p + transpose for p in pitches]
    intervals = np.diff(pitches)
    for iv in intervals:
        idx = int(np.clip(iv + 12, 0, 24))
        hist[idx] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def melodic_contour_features(notes: list) -> np.ndarray:
    """
    Características de contorno melódico (4 dimensiones):
    [pendiente_media, std_intervalos, rango_pitch_norm, ratio_ascendente]
    """
    if len(notes) < 2:
        return np.zeros(4)
    pitches = np.array([n.pitch for n in notes], dtype=float)
    intervals = np.diff(pitches)
    slope = np.mean(intervals) / 12.0           # normalizado a semioctava
    std_iv = np.std(intervals) / 12.0
    pitch_range = (pitches.max() - pitches.min()) / 48.0  # normalizado a 4 octavas
    ratio_up = np.sum(intervals > 0) / max(len(intervals), 1)
    return np.array([slope, std_iv, pitch_range, ratio_up])


def rhythmic_profile(notes: list) -> np.ndarray:
    """
    Perfil rítmico (6 dimensiones):
    [densidad, dur_media_norm, dur_std_norm, ratio_corcheas, ratio_negras, ratio_redondas]
    """
    if not notes:
        return np.zeros(6)
    total_dur = sum(n.duration for n in notes)
    density = len(notes) / max(total_dur, 0.001)  # notas/beat
    durs = np.array([n.duration for n in notes])
    dur_mean = np.mean(durs)
    dur_std  = np.std(durs)

    def ratio_near(target, tol=0.1):
        return np.sum(np.abs(durs - target) < tol) / len(durs)

    return np.array([
        np.clip(density / 4.0, 0, 1),   # normalizado: 4 notas/beat = máx
        np.clip(dur_mean / 4.0, 0, 1),
        np.clip(dur_std  / 2.0, 0, 1),
        ratio_near(0.5),   # corcheas
        ratio_near(1.0),   # negras
        ratio_near(2.0),   # blancas
    ])


def compute_compatibility_vector(
    elem: GrammarElement,
    normalize_pitch: bool = False,
) -> np.ndarray:
    """
    Construye el vector de compatibilidad para un GrammarElement.

    Dimensiones:
        12  pitch class histogram
        25  interval histogram
         4  melodic contour
         6  rhythmic profile
        10  harmonic profile
    Total: 57 dimensiones
    """
    notes = elem.notes
    if not notes:
        return np.zeros(57)

    pc  = pitch_class_histogram(notes, normalize_pitch=normalize_pitch)    # 12
    iv  = interval_histogram(notes, normalize_pitch=normalize_pitch)       # 25
    mc  = melodic_contour_features(notes)                                  #  4
    rh  = rhythmic_profile(notes)                                          #  6
    ha  = harmonic_profile(notes)                                          # 10

    return np.concatenate([pc, iv, mc, rh, ha])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similitud coseno entre dos vectores."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def build_compatibility_matrix(
    elements: dict,
    target_duration: float,
    normalize_pitch: bool = False,
    dur_tolerance: float = 0.01,
) -> tuple[list, np.ndarray]:
    """
    Calcula la matriz de compatibilidad para todos los elementos con
    duración ≈ target_duration.

    Devuelve:
        names  — lista de nombres de elementos con esa duración
        matrix — np.ndarray (N×N) de similitudes coseno
    """
    # Asegurar vectores calculados
    candidates = []
    for name, elem in elements.items():
        if name == '__root__':
            continue
        if abs(elem.duration_beats - target_duration) <= dur_tolerance:
            if elem.vector is None:
                elem.vector = compute_compatibility_vector(elem, normalize_pitch)
            candidates.append(name)

    n = len(candidates)
    if n == 0:
        return [], np.array([])

    vecs = np.stack([elements[c].vector for c in candidates])
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = cosine_similarity(vecs[i], vecs[j])

    return candidates, matrix



def harmonic_profile(notes: list) -> np.ndarray:
    """Perfil armónico desde MIDI (10 dims): tensión, disonancia, hist intervalos, polifonía."""
    if len(notes) < 2:
        return np.zeros(10)
    tension_vals = []
    diss_count = 0
    interval_hist = np.zeros(7)
    n_pairs = 0
    poly_counts = []
    sorted_notes = sorted(notes, key=lambda n: n.offset)
    for i, ni in enumerate(sorted_notes):
        ni_end = ni.offset + ni.duration
        simultaneous = 1
        for j, nj in enumerate(sorted_notes):
            if i == j:
                continue
            nj_end = nj.offset + nj.duration
            if nj.offset < ni_end and nj_end > ni.offset:
                simultaneous += 1
                if j > i:
                    iv = abs(ni.pitch - nj.pitch) % 12
                    tension = INTERVAL_TENSION.get(iv, 0.5)
                    tension_vals.append(tension)
                    if iv in (1, 2, 6, 10, 11):
                        diss_count += 1
                    interval_hist[min(iv, 6)] += 1
                    n_pairs += 1
        poly_counts.append(simultaneous)
    mean_tension = float(np.mean(tension_vals)) if tension_vals else 0.0
    diss_ratio = diss_count / max(n_pairs, 1)
    if interval_hist.sum() > 0:
        interval_hist /= interval_hist.sum()
    mean_poly = np.mean(poly_counts) / 4.0 if poly_counts else 0.0
    return np.concatenate([[mean_tension, diss_ratio], interval_hist, [np.clip(mean_poly, 0, 1)]])


def detect_scale(notes: list) -> tuple:
    """Detecta tonalidad con perfiles de Krumhansl. Devuelve (tonic_st, mode)."""
    if not notes:
        return 0, 'major'
    major_p = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor_p = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += n.duration
    if hist.sum() > 0:
        hist /= hist.sum()
    best_score, best_tonic, best_mode = -1, 0, 'major'
    for tonic in range(12):
        rotated = np.roll(hist, -tonic)
        for profile, mode in [(major_p, 'major'), (minor_p, 'minor')]:
            score = np.corrcoef(rotated, profile)[0, 1]
            if score > best_score:
                best_score, best_tonic, best_mode = score, tonic, mode
    return best_tonic, best_mode


def scale_pitches(tonic: int, mode: str) -> list:
    template = MAJOR_SCALE if mode == 'major' else MINOR_SCALE
    return [(tonic + st) % 12 for st in template]


def parse_target_key(key_str: str) -> tuple:
    """Parsea 'C', 'Dm', 'F#m', 'Bb', 'Am' -> (tonic_semitone, mode)."""
    key_str = key_str.strip()
    mode = 'major'
    if key_str.lower().endswith('m') and not key_str.lower().endswith('maj'):
        mode = 'minor'; key_str = key_str[:-1]
    elif key_str.lower().endswith('maj'):
        key_str = key_str[:-3]
    elif key_str.lower().endswith('min'):
        mode = 'minor'; key_str = key_str[:-3]
    key_str = key_str.strip()
    note_upper = key_str[0].upper()
    alter = key_str[1:] if len(key_str) > 1 else ''
    base = NOTE_NAMES.index(note_upper) if note_upper in NOTE_NAMES else 0
    if '#' in alter: base = (base + 1) % 12
    elif 'b' in alter: base = (base - 1) % 12
    return base, mode


def transpose_to_key(elem: GrammarElement, target_tonic: int, target_mode: str) -> GrammarElement:
    """Transpone el elemento para alinear su tónica detectada con target_tonic."""
    if not elem.notes:
        return elem
    src_tonic, _ = detect_scale(elem.notes)
    semitones = (target_tonic - src_tonic) % 12
    if semitones > 6:
        semitones -= 12
    return transpose_element(elem, semitones)


# ═══════════════════════════════════════════════════════════════════════════════
# ESTIMACIÓN DE TRANSPOSICIÓN (modo --pitch-norm)
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_transpose_semitones(
    source_elem: GrammarElement,
    target_elem: GrammarElement,
) -> int:
    """
    Calcula los semitonos de transposición para llevar el target al registro
    del source. Usa la diferencia entre pitch medios.
    """
    if not source_elem.notes or not target_elem.notes:
        return 0
    src_mean = np.mean([n.pitch for n in source_elem.notes])
    tgt_mean = np.mean([n.pitch for n in target_elem.notes])
    return int(round(src_mean - tgt_mean))


def transpose_element(elem: GrammarElement, semitones: int) -> GrammarElement:
    """
    Devuelve una copia del elemento con todos los pitches transpuestos.
    """
    new_elem = copy.deepcopy(elem)
    for note in new_elem.notes:
        note.pitch = int(np.clip(note.pitch + semitones, 0, 127))
    return new_elem


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING CON TEMPERATURA
# ═══════════════════════════════════════════════════════════════════════════════

def temperature_sample(
    scores: np.ndarray,
    temperature: float,
    rng: random.Random,
) -> int:
    """
    Samplea un índice de la distribución de puntuaciones con temperatura.
    temperature=0 → greedy (argmax)
    temperature=1 → softmax estándar
    temperature>1 → más plana
    """
    if temperature <= 0:
        return int(np.argmax(scores))

    # Softmax con temperatura
    s = np.array(scores, dtype=float)
    s = s / temperature
    s = s - s.max()   # estabilidad numérica
    exp_s = np.exp(s)
    probs = exp_s / exp_s.sum()
    cumulative = np.cumsum(probs)
    r = rng.random()
    for i, c in enumerate(cumulative):
        if r <= c:
            return i
    return len(scores) - 1



# ═══════════════════════════════════════════════════════════════════════════════
# RESTRICCIÓN DE CONTINUIDAD EN FRONTERA
# ═══════════════════════════════════════════════════════════════════════════════

def build_next_element_map(root: list, elements: dict) -> dict:
    """Para cada token de la raíz, devuelve el pitch inicial del token siguiente."""
    result = {}
    root_strs = [t for t in root if isinstance(t, str)]
    for i, tok in enumerate(root_strs):
        if i + 1 < len(root_strs):
            next_elem = elements.get(root_strs[i + 1])
            result[tok] = next_elem.notes[0].pitch if (next_elem and next_elem.notes) else None
        else:
            result[tok] = None
    return result


def boundary_score(
    candidate_elem: GrammarElement,
    next_first_pitch,
    continuity_weight: float,
) -> float:
    """exp(-W * |last_pitch - next_pitch| / 12). Devuelve 1.0 si W=0 o no hay siguiente."""
    if continuity_weight <= 0 or next_first_pitch is None or not candidate_elem.notes:
        return 1.0
    semitones = abs(candidate_elem.notes[-1].pitch - next_first_pitch)
    return math.exp(-continuity_weight * semitones / 12.0)


# ═══════════════════════════════════════════════════════════════════════════════
# MODO BIN — agrupación por longitud y ajuste de duración
# ═══════════════════════════════════════════════════════════════════════════════

def build_bins(
    elements: dict,
    bin_size: float,
) -> dict:
    """
    Agrupa los elementos en bins de ancho bin_size beats.
    Devuelve {bin_idx: [nombre, ...]} donde bin_idx = floor(duration / bin_size).
    """
    bins: dict = defaultdict(list)
    for name, elem in elements.items():
        if name == '__root__':
            continue
        idx = int(elem.duration_beats / bin_size)
        bins[idx].append(name)
    return dict(bins)


def fit_scale(notes: list, src_dur: float, dst_dur: float) -> list:
    """
    Escala todas las duraciones y offsets proporcionalmente para que el
    elemento ocupe exactamente dst_dur beats.
    """
    if src_dur <= 0:
        return notes
    ratio = dst_dur / src_dur
    result = []
    for n in notes:
        result.append(GrammarNote(
            pitch=n.pitch,
            duration=round(n.duration * ratio, 6),
            velocity=n.velocity,
            channel=n.channel,
            offset=round(n.offset * ratio, 6),
        ))
    return result


def fit_trim(notes: list, dst_dur: float) -> list:
    """
    Trim inteligente: busca el punto de menor tension en [0.75*dst_dur, dst_dur]
    (silencio, nota larga, nota estable). Si no hay punto bueno corta en dst_dur.
    """
    if not notes:
        return notes
    tonic, _ = detect_scale(notes)
    window_start = dst_dur * 0.75
    best_cut, best_score = dst_dur, -1
    for n in notes:
        if n.offset < window_start or n.offset > dst_dur:
            continue
        score = min(n.duration / 2.0, 1.0)
        if n.pitch % 12 == tonic: score += 1.0
        elif n.pitch % 12 == (tonic + 7) % 12: score += 0.5
        if score > best_score:
            best_score = score
            best_cut = min(n.offset + n.duration, dst_dur)
    cut = min(best_cut, dst_dur)
    result = []
    for n in notes:
        if n.offset >= cut:
            break
        dur = min(n.duration, cut - n.offset)
        result.append(GrammarNote(pitch=n.pitch, duration=max(dur, 1/64),
                                  velocity=n.velocity, channel=n.channel, offset=n.offset))
    return result


def fit_pad(notes: list, src_dur: float, dst_dur: float) -> list:
    """Pad tonal: notas de paso dentro de la escala detectada del fragmento."""
    if not notes:
        return notes
    if dst_dur <= src_dur:
        return fit_trim(notes, dst_dur)
    gap = dst_dur - src_dur
    last = notes[-1]
    tonic, mode = detect_scale(notes)
    scale = scale_pitches(tonic, mode)
    n_fill = max(1, int(gap / (1/16)))
    fill_dur = gap / n_fill
    start_pitch = last.pitch
    target_pitch = notes[0].pitch
    fill_notes = []
    for i in range(n_fill):
        direction = 1 if target_pitch >= start_pitch else -1
        candidate = start_pitch + direction * (i + 1)
        candidate = max(0, min(127, candidate))
        best_p, best_d = candidate, 127
        for octave in range(0, 128, 12):
            for sc in scale:
                p = octave + sc
                if abs(p - candidate) < best_d:
                    best_d, best_p = abs(p - candidate), p
        fill_notes.append(GrammarNote(
            pitch=best_p, duration=round(fill_dur, 6),
            velocity=max(40, last.velocity - 20),
            channel=last.channel,
            offset=round(src_dur + i * fill_dur, 6),
        ))
    return list(notes) + fill_notes


def fit_element(
    elem: 'GrammarElement',
    dst_dur: float,
    fit_mode: str,          # 'scale' | 'pad' | 'trim'
) -> 'GrammarElement':
    """
    Devuelve una copia del elemento ajustada a dst_dur beats según fit_mode.
    Si la duración ya coincide (tolerancia 1e-4) devuelve el elemento sin tocar.
    """
    src_dur = elem.duration_beats
    if abs(src_dur - dst_dur) < 1e-4:
        return elem

    if fit_mode == 'scale':
        new_notes = fit_scale(elem.notes, src_dur, dst_dur)
    elif fit_mode == 'trim':
        new_notes = fit_trim(elem.notes, dst_dur)
    elif fit_mode == 'pad':
        new_notes = fit_pad(elem.notes, src_dur, dst_dur)
    else:
        new_notes = list(elem.notes)

    new_elem = copy.deepcopy(elem)
    new_elem.notes = new_notes
    new_elem.duration_beats = round(sum(n.duration for n in new_notes), 6)
    return new_elem


def transform_grammar_bins(
    rules: dict,
    root: list,
    elements: dict,
    bin_size: float,
    fit_mode: str,
    temperature: float,
    normalize_pitch: bool,
    rng: random.Random,
    bin_length_range: tuple | None = None,
    continuity_weight: float = 0.0,
    target_key: tuple | None = None,
    verbose: bool = False,
) -> tuple[dict, list, dict]:
    """
    Variante de transform_grammar que trabaja con bins de longitud.

    Para cada elemento de la gramática:
      1. Calcula su bin (floor(duration / bin_size)).
      2. Recoge todos los candidatos del mismo bin.
      3. Calcula similitudes coseno entre el elemento origen y los candidatos.
      4. Samplea un reemplazo con temperatura.
      5. Ajusta la duración del elegido a la del hueco destino con fit_mode.
      6. Registra el elemento ajustado en elements con un nombre derivado.

    bin_length_range: (min_beats, max_beats) opcional. Si se indica, solo se
    transforman los elementos cuya duración cae en ese rango [min, max].
    Los elementos fuera del rango se dejan intactos aunque tengan candidatos
    en su bin.

    Devuelve (new_rules, new_root, elements_ampliados).
    """
    bins = build_bins(elements, bin_size)

    lo_filter = bin_length_range[0] if bin_length_range else None
    hi_filter = bin_length_range[1] if bin_length_range else None

    if verbose:
        range_tag = (f", rango [{lo_filter:.3f}, {hi_filter:.3f}]b"
                     if bin_length_range else "")
        print(f"  [bins] {len(bins)} bins de {bin_size}b{range_tag}: " +
              ", ".join(f"bin{k}({len(v)})" for k, v in sorted(bins.items())))

    # Pre-calcular vectores de todos los elementos
    for name, elem in elements.items():
        if name == '__root__':
            continue
        if elem.vector is None:
            elem.vector = compute_compatibility_vector(elem, normalize_pitch)

    next_pitch_map = build_next_element_map(root, elements) if continuity_weight > 0 else {}
    substitutions: dict = {}
    skipped_range = 0

    for name, elem in list(elements.items()):
        if name == '__root__':
            continue

        # Filtro de rango
        if lo_filter is not None and elem.duration_beats < lo_filter:
            skipped_range += 1
            continue
        if hi_filter is not None and elem.duration_beats > hi_filter:
            skipped_range += 1
            continue

        bin_idx = int(elem.duration_beats / bin_size)
        candidates = bins.get(bin_idx, [])
        if len(candidates) <= 1:
            continue   # sin alternativas en el bin

        # Similitudes del origen con cada candidato del bin
        src_vec = elem.vector
        scores = np.array([
            cosine_similarity(src_vec, elements[c].vector)
            for c in candidates
        ])

        chosen_idx = temperature_sample(scores, temperature, rng)
        chosen_name = candidates[chosen_idx]
        if chosen_name == name:
            continue   # sin cambio

        # Ajustar duración del elegido a la del elemento origen
        chosen_elem = elements[chosen_name]
        fitted = fit_element(chosen_elem, elem.duration_beats, fit_mode)

        # Nombre único para el elemento ajustado
        fit_tag = f"_fit{fit_mode[0]}{elem.duration_beats:.3f}"
        fitted_name = f"{chosen_name}{fit_tag}"
        fitted.name = fitted_name
        elements[fitted_name] = fitted

        substitutions[name] = fitted_name

        if verbose:
            print(f"    {name}({elem.duration_beats:.3f}b) → "
                  f"{chosen_name}({chosen_elem.duration_beats:.3f}b) "
                  f"→ {fitted_name} [{fit_mode}]  "
                  f"sim={scores[chosen_idx]:.3f}")

    # Aplicar sustituciones a reglas y raíz
    new_rules = {
        rname: [substitutions.get(t, t) if isinstance(t, str) else t
                for t in body]
        for rname, body in rules.items()
    }
    new_root = [substitutions.get(t, t) if isinstance(t, str) else t
                for t in root]

    # normalize_pitch: transponer elementos insertados al registro del origen
    if normalize_pitch:
        for old_name, fitted_name in substitutions.items():
            src_elem    = elements.get(old_name)
            fitted_elem = elements.get(fitted_name)
            if src_elem and fitted_elem:
                semitones = estimate_transpose_semitones(src_elem, fitted_elem)
                if semitones != 0:
                    transposed = transpose_element(fitted_elem, semitones)
                    tp_name = f"{fitted_name}_tp{semitones:+d}"
                    transposed.name = tp_name
                    elements[tp_name] = transposed
                    new_rules = {
                        rn: [tp_name if t == fitted_name else t for t in body]
                        for rn, body in new_rules.items()
                    }
                    new_root = [tp_name if t == fitted_name else t for t in new_root]

    n_subs = len(substitutions)
    range_msg = (f", rango [{lo_filter:.3f}, {hi_filter:.3f}]b — "
                 f"{skipped_range} elem. fuera de rango ignorados"
                 if bin_length_range else "")
    print(f"  [bins] {n_subs} sustituciones aplicadas "
          f"(bin_size={bin_size}b, fit={fit_mode}{range_msg})")

    return new_rules, new_root, elements

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSFORMACIÓN DE LA GRAMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

def transform_grammar(
    rules: dict,
    root: list,
    elements: dict,
    target_duration: float,
    temperature: float,
    normalize_pitch: bool,
    rng: random.Random,
    continuity_weight: float = 0.0,
    target_key: tuple | None = None,
    verbose: bool = False,
) -> tuple[dict, list]:
    """
    Transforma la gramática sustituyendo los elementos con duración
    ≈ target_duration por otros compatibles sampleados.
    Aplica penalización de continuidad y/o transposición a tonalidad si se indica.
    """
    # Construir matriz de compatibilidad
    candidates, matrix = build_compatibility_matrix(
        elements, target_duration, normalize_pitch=normalize_pitch
    )

    if len(candidates) == 0:
        print(f"  [AVISO] No hay elementos con longitud {target_duration:.3f} beats.")
        return rules, root

    if verbose:
        print(f"  [transform] {len(candidates)} candidatos con duración {target_duration:.3f} beats")

    name_to_idx = {name: i for i, name in enumerate(candidates)}
    next_pitch_map = build_next_element_map(root, elements) if continuity_weight > 0 else {}

    def sample_replacement(src_name: str) -> str:
        if src_name not in name_to_idx:
            return src_name
        idx = name_to_idx[src_name]
        scores = matrix[idx].copy()
        if continuity_weight > 0:
            next_p = next_pitch_map.get(src_name)
            for ci, cname in enumerate(candidates):
                scores[ci] *= boundary_score(elements[cname], next_p, continuity_weight)
        chosen_idx = temperature_sample(scores, temperature, rng)
        chosen_name = candidates[chosen_idx]
        if verbose and chosen_name != src_name:
            print(f"    {src_name} → {chosen_name}  (sim={matrix[idx][chosen_idx]:.3f})")
        return chosen_name

    # Construir mapa de sustituciones (uno por nombre de elemento transformado)
    substitutions: dict = {}  # old_name → new_name
    for name in candidates:
        new_name = sample_replacement(name)
        substitutions[name] = new_name

    # Aplicar sustituciones con transposición si normalize_pitch
    new_rules = {}
    for rule_name, body in rules.items():
        new_body = []
        for tok in body:
            if isinstance(tok, str) and tok in substitutions:
                new_tok = substitutions[tok]
                new_body.append(new_tok)
            else:
                new_body.append(tok)
        new_rules[rule_name] = new_body

    new_root = []
    for tok in root:
        if isinstance(tok, str) and tok in substitutions:
            new_root.append(substitutions[tok])
        else:
            new_root.append(tok)

    # Transponer elementos insertados
    for old_name, new_name in substitutions.items():
        if old_name == new_name:
            continue
        src_elem = elements.get(old_name)
        tgt_elem = elements.get(new_name)
        if not src_elem or not tgt_elem:
            continue
        working = tgt_elem
        if normalize_pitch:
            st = estimate_transpose_semitones(src_elem, working)
            if st != 0:
                working = transpose_element(working, st)
        if target_key is not None:
            working = transpose_to_key(working, target_key[0], target_key[1])
        if working is not tgt_elem:
            tp_name = f"{new_name}_tp"
            working.name = tp_name
            elements[tp_name] = working
            new_rules = {rn: [tp_name if t == new_name else t for t in body]
                         for rn, body in new_rules.items()}
            new_root = [tp_name if t == new_name else t for t in new_root]

    return new_rules, new_root


# ═══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCCIÓN DEL MIDI DESDE LA GRAMÁTICA TRANSFORMADA
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_notes(
    root: list,
    rules: dict,
    elements: dict,
) -> list:
    """
    Reconstruye la secuencia plana de (abs_beat, GrammarNote) desde la
    gramática transformada.

    Expande recursivamente desde new_rules (que contiene las sustituciones)
    en lugar de usar los snapshots pre-transformación de elements. Así los
    cambios en nodos internos de la jerarquía llegan al audio.
    """
    memo: dict = {}
    tpb_fallback = 480   # solo para construcción de GrammarNote desde tupla

    def expand_to_notes(tok, depth: int = 0) -> list:
        """
        Expande un token a lista de GrammarNote con offsets relativos al
        inicio del token (base 0). Usa memoization sobre el nombre del token.
        """
        if depth > 128:
            return []

        # Token hoja: tupla (pitch, dur)
        if isinstance(tok, tuple):
            pitch, dur = tok
            return [GrammarNote(pitch=pitch, duration=dur,
                                velocity=80, channel=0, offset=0.0)]

        # Token string: buscar primero en elements (puede ser un elemento
        # fitted/transpuesto creado durante la transformación), luego en rules
        if tok in memo:
            return memo[tok]

        # Prioridad 1: elemento materializado (fitted, transpuesto, importado)
        # cuyas notas YA son el resultado final correcto
        elem = elements.get(tok)
        if elem is not None and tok not in rules:
            # Elemento sin regla en rules: es un leaf o un elemento externo/fitted
            memo[tok] = list(elem.notes)
            return memo[tok]

        # Prioridad 2: expandir desde rules (captura las sustituciones)
        if tok in rules:
            body = rules[tok]
            result = []
            cursor = 0.0
            for sub_tok in body:
                sub_notes = expand_to_notes(sub_tok, depth + 1)
                for n in sub_notes:
                    result.append(GrammarNote(
                        pitch=n.pitch,
                        duration=n.duration,
                        velocity=n.velocity,
                        channel=n.channel,
                        offset=round(cursor + n.offset, 6),
                    ))
                # Avanzar por el span real del subelemento:
                # max(offset+dur) captura correctamente tanto notas
                # monofónicas (secuenciales) como acordes (simultáneas).
                if sub_notes:
                    sub_span = max(n.offset + n.duration for n in sub_notes)
                    cursor += sub_span
            memo[tok] = result
            return result

        # Fallback: elemento en elements aunque tenga regla (snapshot)
        if elem is not None:
            memo[tok] = list(elem.notes)
            return memo[tok]

        return []

    result = []
    cursor = 0.0
    for tok in root:
        notes = expand_to_notes(tok)
        for n in notes:
            result.append((round(cursor + n.offset, 6), n))
        if notes:
            cursor += max(n.offset + n.duration for n in notes)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT / IMPORT DE ELEMENTOS
# ═══════════════════════════════════════════════════════════════════════════════

def element_to_midi_file(
    elem: 'GrammarElement',
    tpb: int,
    tempo: int,
    out_path: Path,
) -> None:
    """Escribe un GrammarElement como fichero MIDI independiente (offsets base 0)."""
    notes_flat = [(n.offset, n) for n in elem.notes]
    notes_to_midi_file(notes_flat, tpb, tempo, out_path)


def export_elements(
    elements: dict,
    directory: Path,
    export_format: str = 'json',
    tpb: int = 480,
    tempo: int = 500_000,
    verbose: bool = False,
) -> None:
    """
    Exporta cada GrammarElement al directorio indicado.

    export_format:
        'json'  — solo fichero JSON (comportamiento original)
        'midi'  — solo fichero MIDI
        'both'  — JSON + MIDI con el mismo nombre base
    """
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, elem in elements.items():
        if name == '__root__':
            continue

        if export_format in ('json', 'both'):
            data = {
                'name': name,
                'duration_beats': elem.duration_beats,
                'occurrences': elem.occurrences,
                'notes': [
                    {
                        'pitch': n.pitch,
                        'duration': n.duration,
                        'velocity': n.velocity,
                        'channel': n.channel,
                        'offset': n.offset,
                    }
                    for n in elem.notes
                ]
            }
            json_file = directory / f"{name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            if verbose:
                print(f"  [export] {name} → {json_file}")

        if export_format in ('midi', 'both'):
            mid_file = directory / f"{name}.mid"
            try:
                element_to_midi_file(elem, tpb, tempo, mid_file)
                if verbose:
                    print(f"  [export] {name} → {mid_file}")
            except Exception as e:
                print(f"  [AVISO] No se pudo exportar MIDI de {name}: {e}")

        count += 1

    fmt_label = {'json': 'JSON', 'midi': 'MIDI', 'both': 'JSON + MIDI'}[export_format]
    print(f"  → {count} elementos exportados ({fmt_label}) a {directory}")


def _midi_file_to_element(fpath: Path, imp_name: str) -> 'GrammarElement':
    """
    Lee un fichero MIDI y lo convierte en un GrammarElement.

    Usa el track con más note_on. Los offsets de las notas se calculan
    desde el inicio del fichero (base 0, en beats). Si el MIDI tiene
    un JSON homónimo junto a él, lee la velocidad/canal desde el JSON
    para mayor fidelidad; en caso contrario usa los valores del MIDI.
    """
    mid = mido.MidiFile(str(fpath))
    tpb = mid.ticks_per_beat

    # Elegir track con más note_on
    best_idx, best_count = 0, -1
    for i, track in enumerate(mid.tracks):
        c = sum(1 for m in track if m.type == 'note_on' and m.velocity > 0)
        if c > best_count:
            best_count, best_idx = c, i

    track = mid.tracks[best_idx]
    abs_t = 0
    active: dict = {}
    raw: list = []
    for msg in track:
        abs_t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            active.setdefault((msg.channel, msg.note), []).append((abs_t, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active and active[key]:
                start_t, vel = active[key].pop(0)
                raw.append((start_t, abs_t, msg.note, vel, msg.channel))

    raw.sort(key=lambda x: x[0])
    notes = []
    for start_t, end_t, pitch, vel, ch in raw:
        offset_b = start_t / tpb
        dur_b    = max((end_t - start_t) / tpb, 1 / 64)
        notes.append(GrammarNote(
            pitch=pitch, duration=round(dur_b, 6),
            velocity=vel, channel=ch,
            offset=round(offset_b, 6),
        ))

    dur_beats = sum(n.duration for n in notes)
    return GrammarElement(
        name=imp_name,
        notes=notes,
        duration_beats=round(dur_beats, 6),
        occurrences=0,
    )


def import_elements(directory: Path, verbose: bool = False) -> dict:
    """
    Importa GrammarElements desde el directorio.

    Formatos aceptados (se pueden mezclar libremente en el mismo directorio):
        *.json  — formato nativo (pitch, duration, velocity, channel, offset)
        *.mid   — MIDI independiente; si existe un JSON homónimo se usa el
                  JSON y el MIDI se ignora para evitar duplicados

    Los elementos importados reciben el prefijo 'imp_' si no lo tienen ya.
    """
    imported = {}
    if not directory.exists():
        print(f"  [AVISO] Directorio de importación no encontrado: {directory}")
        return imported

    # ── Recopilar ficheros, priorizando JSON sobre MIDI homónimo ──────────
    json_stems = {f.stem for f in directory.glob("*.json")}
    files_to_load: list[Path] = []
    for fpath in sorted(directory.iterdir()):
        if fpath.suffix == '.json':
            files_to_load.append(fpath)
        elif fpath.suffix in ('.mid', '.midi'):
            if fpath.stem not in json_stems:   # solo si no hay JSON gemelo
                files_to_load.append(fpath)

    for fpath in files_to_load:
        try:
            if fpath.suffix == '.json':
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                name = data.get('name', fpath.stem)
                imp_name = f"imp_{name}" if not name.startswith('imp_') else name
                notes = [
                    GrammarNote(
                        pitch=n['pitch'],
                        duration=n['duration'],
                        velocity=n.get('velocity', 80),
                        channel=n.get('channel', 0),
                        offset=n.get('offset', 0.0),
                    )
                    for n in data.get('notes', [])
                ]
                elem = GrammarElement(
                    name=imp_name,
                    notes=notes,
                    duration_beats=data.get('duration_beats', 0.0),
                    occurrences=0,
                )
            else:
                # MIDI sin JSON gemelo
                name = fpath.stem
                imp_name = f"imp_{name}" if not name.startswith('imp_') else name
                elem = _midi_file_to_element(fpath, imp_name)

            imported[imp_name] = elem
            if verbose:
                src_tag = 'json' if fpath.suffix == '.json' else 'midi'
                print(f"  [import] {imp_name} ← {fpath.name}  "
                      f"({elem.duration_beats:.3f} beats, {src_tag})")
        except Exception as e:
            print(f"  [AVISO] Error importando {fpath.name}: {e}")

    n_json = sum(1 for f in files_to_load if f.suffix == '.json')
    n_midi = len(files_to_load) - n_json
    parts = []
    if n_json: parts.append(f"{n_json} JSON")
    if n_midi: parts.append(f"{n_midi} MIDI")
    print(f"  → {len(imported)} elementos importados ({', '.join(parts)}) desde {directory}")
    return imported


# ═══════════════════════════════════════════════════════════════════════════════
# INFORME
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html_report(
    rules: dict,
    root: list,
    elements: dict,
    notes_flat: list,
    target_duration: float | None = None,
    normalize_pitch: bool = False,
    bin_size: float | None = None,
    out_path: Path | None = None,
    midi_name: str = "",
) -> Path:
    """
    Genera un informe interactivo HTML con piano-roll, tabla de elementos,
    distribuciones y matrices de compatibilidad.

    Si out_path es None se guarda junto al MIDI con extensión .grammar.html.
    Devuelve la ruta del fichero generado.
    """
    import json as _json

    NOTE_NAMES_PY = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

    # ── Construir datos para incrustar en el JS ──────────────────────────────
    root_dur = elements.get('__root__', GrammarElement('__root__', [], 0.0)).duration_beats

    data: dict = {
        'total_notes':    len(notes_flat),
        'total_rules':    len(rules),
        'root_len':       len(root),
        'duration_beats': root_dur,
        'midi_name':      midi_name,
        'rules':          {},
        'by_duration':    {},
    }

    for name, elem in elements.items():
        if name == '__root__':
            continue
        body = rules.get(name, [])
        vec  = compute_compatibility_vector(elem, normalize_pitch)
        data['rules'][name] = {
            'duration':    round(elem.duration_beats, 4),
            'n_notes':     len(elem.notes),
            'occurrences': elem.occurrences,
            'body':        [str(t) for t in body[:8]],
            'notes':       [{'pitch': n.pitch,
                             'duration': round(n.duration, 4),
                             'offset':   round(n.offset,   4)} for n in elem.notes],
            'vector':      vec.tolist(),
        }
        key = str(round(elem.duration_beats, 3))
        data['by_duration'].setdefault(key, []).append(name)

    # Matrices de compatibilidad para las 3 longitudes más pobladas
    dur_counts = {k: len(v) for k, v in data['by_duration'].items()}
    top_durs = sorted(dur_counts, key=lambda k: (-dur_counts[k], float(k)))[:3]
    compat_keys = []
    for dk in top_durs:
        d = float(dk)
        cands, matrix = build_compatibility_matrix(elements, d, normalize_pitch)
        js_key = f'compat_{dk.replace(".", "_")}'
        data[js_key] = {'names': cands, 'matrix': matrix.tolist(), 'dur': d}
        compat_keys.append({'key': js_key, 'label': f'{d}b'})
    data['compat_keys'] = compat_keys

    # Secuencia de pitches para piano-roll
    data['pitch_sequence'] = [
        {'pitch': n.pitch, 'offset': round(n.offset, 4), 'dur': round(n.duration, 4)}
        for _, n in notes_flat
    ]

    # Datos de bins
    data['bin_size'] = bin_size
    if bin_size is not None:
        raw_bins = build_bins(elements, bin_size)
        data['bins'] = {
            str(idx): {
                'lo': round(idx * bin_size, 4),
                'hi': round((idx + 1) * bin_size, 4),
                'names': names,
            }
            for idx, names in sorted(raw_bins.items())
        }
    else:
        data['bins'] = {}

    grammar_js = 'const GRAMMAR = ' + _json.dumps(data, separators=(',', ':')) + ';'
    compat_btns_js = _json.dumps(compat_keys)

    # ── Plantilla HTML ───────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Grammar Transformer — {midi_name}</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
:root{{
  --bg:#0a0a0f;--surface:#12121a;--border:#1e1e2e;
  --accent:#c8a96e;--accent2:#6e9ec8;--accent3:#9ec86e;
  --dim:#3a3a52;--text:#d4d0c8;--textdim:#6e6a5e;
  --mono:'Space Mono',monospace;--serif:'Instrument Serif',serif;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{font-size:13px}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);min-height:100vh}}
header{{display:grid;grid-template-columns:1fr auto;align-items:end;
  padding:2.5rem 3rem 1.5rem;border-bottom:1px solid var(--border);
  position:relative;overflow:hidden}}
header::before{{content:'';position:absolute;inset:0;
  background:repeating-linear-gradient(90deg,transparent 0px,transparent 59px,var(--border) 60px);
  opacity:.3;pointer-events:none}}
.header-title{{font-family:var(--serif);font-size:2.8rem;font-style:italic;
  color:var(--accent);letter-spacing:-.02em;line-height:1;position:relative}}
.header-subtitle{{font-size:.75rem;color:var(--textdim);letter-spacing:.15em;
  text-transform:uppercase;margin-top:.4rem;position:relative}}
.header-stats{{display:flex;gap:2rem;text-align:right;position:relative}}
.stat-item{{display:flex;flex-direction:column}}
.stat-val{{font-size:1.8rem;color:var(--accent);font-weight:700;line-height:1}}
.stat-lbl{{font-size:.65rem;color:var(--textdim);text-transform:uppercase;
  letter-spacing:.1em;margin-top:.2rem}}
nav{{display:flex;border-bottom:1px solid var(--border);padding:0 3rem;
  background:var(--surface)}}
nav button{{background:none;border:none;color:var(--textdim);font-family:var(--mono);
  font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;padding:.8rem 1.4rem;
  cursor:pointer;border-bottom:2px solid transparent;transition:color .15s,border-color .15s;
  margin-bottom:-1px}}
nav button:hover{{color:var(--text)}}
nav button.active{{color:var(--accent);border-bottom-color:var(--accent)}}
.panel{{display:none;padding:2rem 3rem}}
.panel.active{{display:block}}
.section-label{{font-size:.65rem;text-transform:uppercase;letter-spacing:.2em;
  color:var(--textdim);margin-bottom:1rem;display:flex;align-items:center;gap:.8rem}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
canvas.full{{width:100%;display:block;background:var(--surface);border:1px solid var(--border);border-radius:2px}}
.dur-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
  gap:.6rem;margin-top:1rem}}
.dur-row{{background:var(--surface);border:1px solid var(--border);border-radius:2px;
  padding:.7rem 1rem;display:grid;grid-template-columns:60px 36px 1fr;
  align-items:center;gap:.6rem;cursor:pointer;transition:border-color .15s}}
.dur-row:hover{{border-color:var(--accent)}}
.dur-row.selected{{border-color:var(--accent);background:#1a1810}}
.dur-val{{color:var(--accent);font-weight:700;font-size:.9rem}}
.dur-count{{background:var(--dim);color:var(--text);font-size:.65rem;
  text-align:center;padding:.1rem .3rem;border-radius:2px}}
.dur-names{{font-size:.7rem;color:var(--textdim);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}}
.elem-table{{width:100%;border-collapse:collapse;font-size:.75rem;margin-top:1rem}}
.elem-table th{{text-align:left;color:var(--textdim);font-size:.65rem;
  text-transform:uppercase;letter-spacing:.1em;padding:.4rem .6rem;
  border-bottom:1px solid var(--border);font-weight:400}}
.elem-table td{{padding:.45rem .6rem;border-bottom:1px solid #16161e;vertical-align:middle}}
.elem-table tr{{cursor:pointer;transition:background .1s}}
.elem-table tr:hover td{{background:var(--surface)}}
.elem-table tr.selected td{{background:#1a1810}}
.name-badge{{display:inline-block;background:var(--dim);color:var(--accent);
  padding:.1rem .4rem;border-radius:2px;font-weight:700;font-size:.72rem}}
.dur-pill{{display:inline-block;border:1px solid var(--accent);color:var(--accent);
  padding:.05rem .35rem;border-radius:2px;font-size:.68rem}}
.body-tok{{color:var(--textdim);font-size:.68rem}}
.body-tok .ref{{color:var(--accent2)}}.body-tok .leaf{{color:var(--accent3)}}
.compat-controls{{display:flex;gap:.8rem;margin-bottom:1rem;flex-wrap:wrap}}
.compat-btn{{background:var(--surface);border:1px solid var(--border);color:var(--textdim);
  font-family:var(--mono);font-size:.68rem;padding:.4rem .8rem;cursor:pointer;
  border-radius:2px;transition:all .15s;letter-spacing:.08em}}
.compat-btn:hover{{border-color:var(--accent);color:var(--accent)}}
.compat-btn.active{{border-color:var(--accent);color:var(--accent);background:#1a1810}}
#compat-canvas-wrap{{overflow-x:auto;background:var(--surface);
  border:1px solid var(--border);padding:1rem;border-radius:2px}}
.detail-panel{{position:fixed;right:0;top:0;bottom:0;width:320px;
  background:var(--surface);border-left:1px solid var(--border);padding:1.5rem;
  overflow-y:auto;transform:translateX(100%);
  transition:transform .25s cubic-bezier(.16,1,.3,1);z-index:100}}
.detail-panel.open{{transform:translateX(0)}}
.detail-close{{position:absolute;top:1rem;right:1rem;background:none;border:none;
  color:var(--textdim);font-size:1.2rem;cursor:pointer}}
.detail-name{{font-size:1.6rem;color:var(--accent);font-weight:700;margin-bottom:.3rem}}
.detail-meta{{font-size:.72rem;color:var(--textdim);margin-bottom:1.2rem}}
.detail-section{{margin-bottom:1rem}}
.detail-section-title{{font-size:.6rem;text-transform:uppercase;letter-spacing:.15em;
  color:var(--textdim);margin-bottom:.5rem;padding-bottom:.3rem;
  border-bottom:1px solid var(--border)}}
.note-row{{display:grid;grid-template-columns:40px 60px 1fr;gap:.3rem;
  font-size:.68rem;padding:.2rem 0;border-bottom:1px solid #16161e}}
.note-pitch{{color:var(--accent)}}.note-dur{{color:var(--accent2)}}
.note-bar{{height:8px;background:var(--accent3);border-radius:1px;margin-top:2px}}
.body-line{{font-size:.7rem;padding:.15rem 0;color:var(--textdim)}}
.body-line .ref{{color:var(--accent2)}}.body-line .leaf{{color:var(--accent3)}}
.vec-row{{display:grid;grid-template-columns:80px 1fr 40px;gap:.4rem;
  align-items:center;margin-bottom:.2rem;font-size:.62rem}}
.vec-label{{color:var(--textdim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.vec-bar-bg{{background:var(--dim);height:6px;border-radius:1px;overflow:hidden}}
.vec-bar-fill{{height:100%;background:var(--accent);border-radius:1px;transition:width .3s}}
.vec-val{{color:var(--accent);text-align:right}}
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-track{{background:var(--bg)}}
::-webkit-scrollbar-thumb{{background:var(--dim);border-radius:2px}}
#filter-btns .compat-btn{{font-size:.65rem;padding:.3rem .65rem}}
</style>
</head>
<body>
<header>
  <div>
    <div class="header-title">Grammar Transformer</div>
    <div class="header-subtitle" id="hdr-sub">Reporte de análisis</div>
  </div>
  <div class="header-stats">
    <div class="stat-item"><span class="stat-val" id="s-notes">—</span><span class="stat-lbl">notas</span></div>
    <div class="stat-item"><span class="stat-val" id="s-rules">—</span><span class="stat-lbl">reglas</span></div>
    <div class="stat-item"><span class="stat-val" id="s-dur">—</span><span class="stat-lbl">beats</span></div>
  </div>
</header>
<nav>
  <button class="active" onclick="showPanel('piano')">Piano Roll</button>
  <button onclick="showPanel('durations')">Por Longitud</button>
  <button onclick="showPanel('elements')">Elementos</button>
  <button onclick="showPanel('compat')">Compatibilidad</button>
  <button onclick="showPanel('bins')" id="nav-bins" style="display:none">Bins</button>
</nav>
<div class="panel active" id="panel-piano">
  <div class="section-label">Secuencia completa</div>
  <canvas id="pianoroll-canvas" class="full" height="200"></canvas>
  <div style="display:flex;gap:2rem;margin-top:.8rem;font-size:.68rem;color:var(--textdim)">
    <span>↑ pitch más alto</span><span>Ancho = duración</span>
    <span style="color:var(--accent)">█</span><span>nota (color = clase de pitch)</span>
  </div>
  <div style="margin-top:1.5rem" class="section-label">Distribución de clases de pitch</div>
  <canvas id="pitch-hist-canvas" class="full" height="100"></canvas>
  <div style="margin-top:1.5rem" class="section-label">Distribución de duraciones</div>
  <canvas id="dur-hist-canvas" class="full" height="100"></canvas>
</div>
<div class="panel" id="panel-durations">
  <div class="section-label">Elementos agrupados por duración en beats</div>
  <div class="dur-grid" id="dur-grid"></div>
  <div style="margin-top:1.5rem" class="section-label">Elementos por longitud</div>
  <canvas id="durlen-canvas" class="full" height="120"></canvas>
</div>
<div class="panel" id="panel-elements">
  <div class="section-label">Todas las reglas Re-Pair</div>
  <div style="display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap" id="filter-btns"></div>
  <table class="elem-table" id="elem-table">
    <thead><tr>
      <th>Regla</th><th>Dur (b)</th><th>Notas</th><th>Ocurr.</th>
      <th>Cuerpo</th><th style="width:140px">Roll</th>
    </tr></thead>
    <tbody id="elem-tbody"></tbody>
  </table>
</div>
<div class="panel" id="panel-compat">
  <div class="section-label">Matrices de similitud coseno entre elementos</div>
  <div class="compat-controls" id="compat-controls"></div>
  <div id="compat-canvas-wrap"><canvas id="compat-canvas"></canvas></div>
  <div style="margin-top:.8rem;font-size:.68rem;color:var(--textdim)">
    Vector 47-dim: pitch-class (12) + intervalos (25) + contorno (4) + ritmo (6).
    <span style="color:#fff">Blanco=máx</span> · <span style="color:var(--accent)">Dorado=alto</span> · <span style="color:var(--dim)">Oscuro=bajo</span>
  </div>
</div>
<div class="panel" id="panel-bins">
  <div class="section-label" id="bins-label">Distribución en bins</div>
  <canvas id="bins-bar-canvas" class="full" height="160"></canvas>
  <div style="margin-top:1.5rem" class="section-label">Detalle por bin</div>
  <div id="bins-detail"></div>
</div>
<div class="detail-panel" id="detail-panel">
  <button class="detail-close" onclick="closeDetail()">✕</button>
  <div class="detail-name" id="d-name"></div>
  <div class="detail-meta"  id="d-meta"></div>
  <canvas id="detail-roll" style="width:100%;height:80px;display:block"></canvas>
  <div class="detail-section" style="margin-top:.8rem">
    <div class="detail-section-title">Cuerpo de la regla</div>
    <div id="d-body"></div>
  </div>
  <div class="detail-section">
    <div class="detail-section-title">Notas expandidas</div>
    <div id="d-notes"></div>
  </div>
  <div class="detail-section">
    <div class="detail-section-title">Vector musical (top 10 dims)</div>
    <div id="d-vector"></div>
  </div>
</div>
<script>
{grammar_js}
const COMPAT_KEYS = {compat_btns_js};
const NOTE_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
function pitchName(p){{return NOTE_NAMES[p%12]+Math.floor(p/12-1)}}
// ── init header ──
document.getElementById('s-notes').textContent = GRAMMAR.total_notes;
document.getElementById('s-rules').textContent = GRAMMAR.total_rules;
document.getElementById('s-dur').textContent   = Math.round(GRAMMAR.duration_beats*10)/10;
if(GRAMMAR.midi_name) document.getElementById('hdr-sub').textContent = 'Reporte — '+GRAMMAR.midi_name;
// ── panels ──
const PANELS = ['piano','durations','elements','compat','bins'];
function showPanel(name){{
  PANELS.forEach(p=>{{
    document.getElementById('panel-'+p).classList.toggle('active',p===name);
  }});
  document.querySelectorAll('nav button').forEach((b,i)=>b.classList.toggle('active',PANELS[i]===name));
  if(name==='piano')     {{drawPianoRoll();drawPitchHist();drawDurHist()}}
  if(name==='durations') {{buildDurGrid();drawDurLenChart()}}
  if(name==='elements')  {{buildElemTable()}}
  if(name==='compat')    {{buildCompatControls();showCompat(COMPAT_KEYS[0]?.key)}}
  if(name==='bins')      {{buildBinsPanel()}}
}}
// ── bins panel ──
function initBinsNav(){{
  const hasBins = GRAMMAR.bin_size !== null && Object.keys(GRAMMAR.bins).length > 0;
  const btn = document.getElementById('nav-bins');
  if(hasBins && btn) btn.style.display='';
  if(hasBins){{
    document.getElementById('bins-label').textContent =
      `Distribución en bins — bin_size = ${{GRAMMAR.bin_size}}b`;
  }}
}}
function buildBinsPanel(){{
  drawBinsBarChart();
  buildBinsDetail();
}}
function drawBinsBarChart(){{
  const canvas = document.getElementById('bins-bar-canvas');
  const {{ctx,w,h}} = resizeCanvas(canvas);
  const bins = GRAMMAR.bins;
  const keys = Object.keys(bins).map(Number).sort((a,b)=>a-b);
  if(!keys.length){{
    ctx.fillStyle='#12121a'; ctx.fillRect(0,0,w,h);
    ctx.fillStyle='#6e6a5e'; ctx.font='11px Space Mono'; ctx.textAlign='left';
    ctx.fillText('Sin datos de bins (usa --bin-size al generar el reporte)',12,h/2);
    return;
  }}
  const counts = keys.map(k=>bins[String(k)].names.length);
  const maxC = Math.max(...counts,1);
  const bw = w / keys.length;
  ctx.fillStyle='#12121a'; ctx.fillRect(0,0,w,h);
  const labelH = 32;
  keys.forEach((k,i)=>{{
    const bh = (counts[i]/maxC)*(h-labelH-8);
    const x = i*bw+1;
    const y = h-labelH-bh;
    // barra con gradiente
    const grad = ctx.createLinearGradient(0,y,0,y+bh);
    grad.addColorStop(0,'#e8c97e');
    grad.addColorStop(1,'#8a6a30');
    ctx.fillStyle=grad;
    ctx.fillRect(x,y,bw-2,bh);
    // etiqueta conteo
    if(bh > 14){{
      ctx.fillStyle='#0a0a0f'; ctx.font=`bold ${{Math.min(11,Math.floor(bw*0.35))}}px Space Mono`;
      ctx.textAlign='center';
      ctx.fillText(counts[i], x+bw/2-1, y+bh/2+4);
    }}
    // etiqueta bin
    ctx.fillStyle='#6e6a5e';
    ctx.font=`${{Math.min(9,Math.floor(bw*0.28))}}px Space Mono`;
    ctx.textAlign='center';
    const lo=bins[String(k)].lo, hi=bins[String(k)].hi;
    if(bw>36){{
      ctx.fillText(`${{lo}}–${{hi}}`, x+bw/2-1, h-labelH+12);
      ctx.fillStyle='#c8a96e';
      ctx.fillText(`bin${{k}}`, x+bw/2-1, h-labelH+24);
    }} else {{
      ctx.fillText(`b${{k}}`, x+bw/2-1, h-labelH+14);
    }}
  }});
}}
function buildBinsDetail(){{
  const container = document.getElementById('bins-detail');
  container.innerHTML='';
  const bins = GRAMMAR.bins;
  const keys = Object.keys(bins).map(Number).sort((a,b)=>a-b);
  if(!keys.length){{
    container.innerHTML='<p style="color:var(--textdim);font-size:.75rem">Sin datos de bins.</p>';
    return;
  }}
  const maxC = Math.max(...keys.map(k=>bins[String(k)].names.length),1);
  keys.forEach(k=>{{
    const bin = bins[String(k)];
    const section = document.createElement('div');
    section.style.cssText='margin-bottom:1.2rem;border:1px solid var(--border);border-radius:2px;overflow:hidden';
    // cabecera
    const hdr = document.createElement('div');
    hdr.style.cssText='display:grid;grid-template-columns:auto auto 1fr auto;gap:.8rem;align-items:center;padding:.5rem 1rem;background:var(--surface);cursor:pointer';
    const pct = Math.round(bin.names.length/maxC*100);
    hdr.innerHTML=`
      <span style="color:var(--accent);font-weight:700;font-size:.85rem">bin${{k}}</span>
      <span style="color:var(--textdim);font-size:.68rem">[${{bin.lo}}b, ${{bin.hi}}b)</span>
      <div style="background:var(--dim);height:6px;border-radius:1px;overflow:hidden">
        <div style="height:100%;width:${{pct}}%;background:var(--accent);border-radius:1px"></div>
      </div>
      <span style="color:var(--accent2);font-size:.75rem;font-weight:700">${{bin.names.length}} elem</span>`;
    // lista de elementos (colapsable)
    const body = document.createElement('div');
    body.style.cssText='padding:.6rem 1rem;display:flex;flex-wrap:wrap;gap:.4rem';
    bin.names.forEach(name=>{{
      const badge = document.createElement('span');
      const r = GRAMMAR.rules[name];
      const dur = r ? r.duration : '?';
      badge.style.cssText='background:var(--dim);color:var(--accent);padding:.15rem .5rem;border-radius:2px;font-size:.7rem;cursor:pointer;border:1px solid transparent;transition:border-color .1s';
      badge.textContent=`${{name}} (${{dur}}b)`;
      badge.title=`${{name}} — ${{r?r.n_notes:0}} notas`;
      badge.onmouseenter=()=>badge.style.borderColor='var(--accent)';
      badge.onmouseleave=()=>badge.style.borderColor='transparent';
      badge.onclick=()=>{{showPanel('elements');openDetail(name)}};
      body.appendChild(badge);
    }});
    let open=true;
    hdr.onclick=()=>{{ open=!open; body.style.display=open?'flex':'none'; }};
    section.appendChild(hdr);
    section.appendChild(body);
    container.appendChild(section);
  }});
}}
}}
// ── canvas helpers ──
function resizeCanvas(canvas){{
  const dpr=window.devicePixelRatio||1;
  const r=canvas.getBoundingClientRect();
  canvas.width=r.width*dpr; canvas.height=r.height*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  return {{ctx,w:r.width,h:r.height}};
}}
function simColor(v){{
  const r=Math.round(30+(225-30)*v*v);
  const g=Math.round(30+(200-30)*v*v);
  const b2=Math.round(46+(180-46)*v*v);
  return `rgb(${{r}},${{g}},${{b2}})`;
}}
// ── piano roll ──
function drawPianoRoll(){{
  const canvas=document.getElementById('pianoroll-canvas');
  const {{ctx,w,h}}=resizeCanvas(canvas);
  const notes=GRAMMAR.pitch_sequence;
  const totalDur=GRAMMAR.duration_beats;
  const minP=Math.min(...notes.map(n=>n.pitch))-1;
  const maxP=Math.max(...notes.map(n=>n.pitch))+1;
  const pRange=maxP-minP;
  ctx.fillStyle='#12121a'; ctx.fillRect(0,0,w,h);
  for(let p=minP;p<=maxP;p++){{
    if(p%12===0){{
      const y=h-((p-minP)/pRange)*h;
      ctx.strokeStyle='#1e1e2e'; ctx.lineWidth=0.5;
      ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
    }}
  }}
  notes.forEach(n=>{{
    const x=(n.offset/totalDur)*w;
    const nw=Math.max(1.5,(n.dur/totalDur)*w-0.5);
    const y=h-((n.pitch-minP+0.1)/pRange)*h;
    const nh=Math.max(2,h/pRange-0.5);
    ctx.fillStyle=`hsl(${{(n.pitch%12)*30}},60%,55%)`;
    ctx.fillRect(x,y,nw,nh);
  }});
}}
function drawPitchHist(){{
  const canvas=document.getElementById('pitch-hist-canvas');
  const {{ctx,w,h}}=resizeCanvas(canvas);
  const counts=new Array(12).fill(0);
  GRAMMAR.pitch_sequence.forEach(n=>counts[n.pitch%12]++);
  const max=Math.max(...counts);
  const bw=w/12;
  ctx.fillStyle='#12121a'; ctx.fillRect(0,0,w,h);
  counts.forEach((c,i)=>{{
    const bh=(c/max)*(h-24);
    ctx.fillStyle=`hsl(${{i*30}},60%,50%)`;
    ctx.fillRect(i*bw+1,h-bh-20,bw-2,bh);
    ctx.fillStyle='#6e6a5e'; ctx.font='8px Space Mono';
    ctx.textAlign='center';
    ctx.fillText(NOTE_NAMES[i],i*bw+bw/2,h-6);
  }});
}}
function drawDurHist(){{
  const canvas=document.getElementById('dur-hist-canvas');
  const {{ctx,w,h}}=resizeCanvas(canvas);
  const counts={{}};
  GRAMMAR.pitch_sequence.forEach(n=>{{
    const k=Math.round(n.dur*16)/16;
    counts[k]=(counts[k]||0)+1;
  }});
  const keys=Object.keys(counts).map(Number).sort((a,b)=>a-b);
  const max=Math.max(...Object.values(counts));
  const bw=w/keys.length;
  ctx.fillStyle='#12121a'; ctx.fillRect(0,0,w,h);
  keys.forEach((k,i)=>{{
    const bh=(counts[k]/max)*(h-20);
    ctx.fillStyle='#c8a96e';
    ctx.fillRect(i*bw+1,h-bh-18,bw-2,bh);
    ctx.fillStyle='#6e6a5e'; ctx.font='8px Space Mono'; ctx.textAlign='center';
    ctx.fillText(k+'b',i*bw+bw/2,h-4);
  }});
}}
// ── dur grid ──
function buildDurGrid(){{
  const grid=document.getElementById('dur-grid');
  if(grid.children.length>0) return;
  const durs=Object.keys(GRAMMAR.by_duration).map(Number).sort((a,b)=>a-b);
  durs.forEach(d=>{{
    const names=GRAMMAR.by_duration[String(d)];
    const row=document.createElement('div');
    row.className='dur-row';
    row.innerHTML=`<span class="dur-val">${{d}}b</span><span class="dur-count">${{names.length}}</span><span class="dur-names">${{names.join(' · ')}}</span>`;
    row.onclick=()=>{{
      document.querySelectorAll('.dur-row').forEach(r=>r.classList.remove('selected'));
      row.classList.add('selected');
      showPanel('elements'); filterByDur(d);
    }};
    grid.appendChild(row);
  }});
}}
function drawDurLenChart(){{
  const canvas=document.getElementById('durlen-canvas');
  const {{ctx,w,h}}=resizeCanvas(canvas);
  const durs=Object.keys(GRAMMAR.by_duration).map(Number).sort((a,b)=>a-b);
  const counts=durs.map(d=>GRAMMAR.by_duration[String(d)].length);
  const max=Math.max(...counts);
  const bw=w/durs.length;
  ctx.fillStyle='#12121a'; ctx.fillRect(0,0,w,h);
  durs.forEach((d,i)=>{{
    const bh=(counts[i]/max)*(h-22);
    ctx.fillStyle=counts[i]>1?'#c8a96e':'#3a3a52';
    ctx.fillRect(i*bw+1,h-bh-18,bw-2,bh);
    if(bw>20){{
      ctx.fillStyle='#6e6a5e'; ctx.font='7px Space Mono'; ctx.textAlign='center';
      ctx.fillText(d+'b',i*bw+bw/2,h-4);
    }}
  }});
}}
// ── elements table ──
let currentFilter=null;
function buildElemTable(filterDur){{
  const tbody=document.getElementById('elem-tbody');
  tbody.innerHTML='';
  const fb=document.getElementById('filter-btns');
  if(!fb.children.length){{
    const allBtn=document.createElement('button');
    allBtn.className='compat-btn active'; allBtn.textContent='Todos';
    allBtn.onclick=()=>{{currentFilter=null;buildElemTable();setActiveFilterBtn(allBtn)}};
    fb.appendChild(allBtn);
    Object.keys(GRAMMAR.by_duration).map(Number).sort((a,b)=>a-b).forEach(d=>{{
      const btn=document.createElement('button');
      btn.className='compat-btn'; btn.textContent=d+'b';
      btn.onclick=()=>{{currentFilter=d;buildElemTable(d);setActiveFilterBtn(btn)}};
      fb.appendChild(btn);
    }});
  }}
  const dur=filterDur!==undefined?filterDur:currentFilter;
  let names=Object.keys(GRAMMAR.rules).sort((a,b)=>parseInt(a.slice(1))-parseInt(b.slice(1)));
  if(dur!==null&&dur!==undefined) names=names.filter(n=>Math.abs(GRAMMAR.rules[n].duration-dur)<0.01);
  names.forEach(name=>{{
    const r=GRAMMAR.rules[name];
    const rollId='roll-'+name;
    const tr=document.createElement('tr');
    tr.innerHTML=`
      <td><span class="name-badge">${{name}}</span></td>
      <td><span class="dur-pill">${{r.duration}}b</span></td>
      <td>${{r.n_notes}}</td>
      <td>${{r.occurrences>0?r.occurrences:'<span style="color:var(--dim)">—</span>'}}</td>
      <td class="body-tok">${{formatBody(r.body)}}</td>
      <td><canvas id="${{rollId}}" width="140" height="28" style="width:140px;height:28px;display:inline-block"></canvas></td>`;
    tr.onclick=()=>openDetail(name);
    tbody.appendChild(tr);
    requestAnimationFrame(()=>drawMiniRoll(rollId,r.notes));
  }});
}}
function setActiveFilterBtn(btn){{
  document.querySelectorAll('#filter-btns .compat-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
}}
function filterByDur(d){{
  currentFilter=d; buildElemTable(d);
  document.querySelectorAll('#filter-btns .compat-btn').forEach(b=>{{
    b.classList.toggle('active',b.textContent===d+'b'||(d===null&&b.textContent==='Todos'));
  }});
}}
function formatBody(body){{
  return body.map(t=>{{
    if(t.startsWith("'N")) return `<span class="ref">${{t.replace(/'/g,'')}}</span>`;
    return `<span class="leaf">${{t}}</span>`;
  }}).join(' ');
}}
function drawMiniRoll(id,notes){{
  const canvas=document.getElementById(id);
  if(!canvas) return;
  const dpr=window.devicePixelRatio||1;
  canvas.width=140*dpr; canvas.height=28*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.fillStyle='#0e0e16'; ctx.fillRect(0,0,140,28);
  if(!notes.length) return;
  const totalDur=notes.reduce((s,n)=>s+n.duration,0);
  const minP=Math.min(...notes.map(n=>n.pitch));
  const maxP=Math.max(...notes.map(n=>n.pitch));
  const pRange=Math.max(maxP-minP,1);
  notes.forEach(n=>{{
    const x=(n.offset/totalDur)*140;
    const nw=Math.max(1.5,(n.duration/totalDur)*140-0.5);
    const y=28-((n.pitch-minP)/pRange)*(28-4)-3;
    ctx.fillStyle=`hsl(${{(n.pitch%12)*30}},65%,55%)`;
    ctx.fillRect(x,y,nw,3);
  }});
}}
// ── compat matrix ──
let currentCompatKey=null;
function buildCompatControls(){{
  const ctrl=document.getElementById('compat-controls');
  if(ctrl.children.length) return;
  COMPAT_KEYS.forEach((ck,idx)=>{{
    const btn=document.createElement('button');
    btn.className='compat-btn'+(idx===0?' active':'');
    btn.textContent=ck.label;
    btn.onclick=()=>{{
      document.querySelectorAll('#compat-controls .compat-btn').forEach(b=>b.classList.remove('active'));
      btn.classList.add('active'); showCompat(ck.key);
    }};
    ctrl.appendChild(btn);
  }});
}}
function showCompat(key){{
  if(!key) return;
  currentCompatKey=key;
  const data=GRAMMAR[key];
  if(!data) return;
  drawCompatMatrix(data.names,data.matrix);
}}
function drawCompatMatrix(names,matrix){{
  const canvas=document.getElementById('compat-canvas');
  const n=names.length;
  if(n===0){{
    canvas.width=300; canvas.height=60;
    const ctx=canvas.getContext('2d');
    ctx.fillStyle='#12121a'; ctx.fillRect(0,0,300,60);
    ctx.fillStyle='#6e6a5e'; ctx.font='12px Space Mono';
    ctx.fillText('Sin elementos',10,35); return;
  }}
  const cell=Math.max(44,Math.min(80,Math.floor(500/n)));
  const lW=52,lH=52,W=lW+n*cell,H=lH+n*cell;
  const dpr=window.devicePixelRatio||1;
  canvas.width=W*dpr; canvas.height=H*dpr;
  canvas.style.width=W+'px'; canvas.style.height=H+'px';
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.fillStyle='#0e0e16'; ctx.fillRect(0,0,W,H);
  for(let i=0;i<n;i++) for(let j=0;j<n;j++){{
    const v=matrix[i][j];
    ctx.fillStyle=simColor(v);
    ctx.fillRect(lW+j*cell+1,lH+i*cell+1,cell-2,cell-2);
    ctx.fillStyle=v>0.6?'#0a0a0f':'#d4d0c8';
    ctx.font=`${{Math.floor(cell*.22)}}px Space Mono`;
    ctx.textAlign='center'; ctx.textBaseline='middle';
    ctx.fillText(v.toFixed(2),lW+j*cell+cell/2,lH+i*cell+cell/2);
  }}
  ctx.fillStyle='#c8a96e';
  ctx.font=`bold ${{Math.floor(cell*.24)}}px Space Mono`;
  ctx.textAlign='center'; ctx.textBaseline='middle';
  names.forEach((name,i)=>{{
    ctx.save(); ctx.translate(lW+i*cell+cell/2,lH*.5); ctx.fillText(name,0,0); ctx.restore();
    ctx.textAlign='right'; ctx.fillText(name,lW-6,lH+i*cell+cell/2); ctx.textAlign='center';
  }});
}}
// ── detail ──
function openDetail(name){{
  const r=GRAMMAR.rules[name];
  document.getElementById('d-name').textContent=name;
  document.getElementById('d-meta').textContent=`${{r.duration}} beats · ${{r.n_notes}} notas · ${{r.occurrences}} ocurrencias`;
  const canvas=document.getElementById('detail-roll');
  const dpr=window.devicePixelRatio||1;
  canvas.width=canvas.offsetWidth*dpr; canvas.height=80*dpr;
  const ctx=canvas.getContext('2d'); ctx.scale(dpr,dpr);
  const w=canvas.offsetWidth,h=80;
  ctx.fillStyle='#0e0e16'; ctx.fillRect(0,0,w,h);
  const notes=r.notes;
  if(notes.length){{
    const totalDur=notes.reduce((s,n)=>s+n.duration,0);
    const minP=Math.min(...notes.map(n=>n.pitch));
    const maxP=Math.max(...notes.map(n=>n.pitch));
    const pRange=Math.max(maxP-minP,1);
    notes.forEach(n=>{{
      const x=(n.offset/totalDur)*w;
      const nw=Math.max(2,(n.duration/totalDur)*w-1);
      const y=h-((n.pitch-minP)/pRange)*(h-8)-5;
      ctx.fillStyle=`hsl(${{(n.pitch%12)*30}},65%,55%)`;
      ctx.fillRect(x,y,nw,6);
    }});
  }}
  document.getElementById('d-body').innerHTML=r.body.map(t=>{{
    const isRef=t.startsWith("'N");
    return `<div class="body-line ${{isRef?'ref':'leaf'}}">${{t}}</div>`;
  }}).join('');
  const maxDur=Math.max(...r.notes.map(n=>n.duration),0.001);
  document.getElementById('d-notes').innerHTML=r.notes.map(n=>`
    <div class="note-row">
      <span class="note-pitch">${{pitchName(n.pitch)}}</span>
      <span class="note-dur">${{n.duration}}b</span>
      <div><div class="note-bar" style="width:${{Math.round((n.duration/maxDur)*100)}}%"></div></div>
    </div>`).join('');
  const vecLabels=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B',
    '-12','-11','-10','-9','-8','-7','-6','-5','-4','-3','-2','-1','0',
    '+1','+2','+3','+4','+5','+6','+7','+8','+9','+10','+11','+12',
    'slope','σ_iv','range','↑ratio','dens','dur_μ','dur_σ','♪½','♩','♩♩'];
  const vec=r.vector;
  const maxV=Math.max(...vec.map(Math.abs),0.001);
  const top10=vec.map((v,i)=>{{return{{v,i}}}}).sort((a,b)=>Math.abs(b.v)-Math.abs(a.v)).slice(0,10);
  document.getElementById('d-vector').innerHTML=top10.map((item)=>{{const v=item.v,i=item.i; return `
    <div class="vec-row">
      <span class="vec-label" title="${{vecLabels[i]||i}}">${{vecLabels[i]||'dim'+i}}</span>
      <div class="vec-bar-bg"><div class="vec-bar-fill" style="width:${{Math.round(Math.abs(v)/maxV*100)}}%"></div></div>
      <span class="vec-val">${{v.toFixed(3)}}</span>
    </div>`}}).join('');
  document.getElementById('detail-panel').classList.add('open');
  document.querySelectorAll('#elem-tbody tr').forEach(tr=>{{
    tr.classList.toggle('selected',tr.querySelector('.name-badge')?.textContent===name);
  }});
}}
function closeDetail(){{document.getElementById('detail-panel').classList.remove('open')}}
window.addEventListener('load',()=>{{drawPianoRoll();drawPitchHist();drawDurHist();initBinsNav()}});
window.addEventListener('resize',()=>{{
  const a=document.querySelector('.panel.active')?.id;
  if(a==='panel-piano'){{drawPianoRoll();drawPitchHist();drawDurHist()}}
  if(a==='panel-compat'){{showCompat(currentCompatKey)}}
}});
</script>
</body></html>"""

    if out_path is None:
        out_path = Path(midi_name).with_suffix('.grammar.html') if midi_name else Path('grammar_report.html')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding='utf-8')
    print(f"  → Reporte HTML guardado: {out_path}")
    return out_path


def print_report(
    rules: dict,
    root: list,
    elements: dict,
    target_duration: float | None = None,
    normalize_pitch: bool = False,
    bin_size: float | None = None,
) -> None:
    """Imprime el informe detallado de la gramática."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            GRAMMAR TRANSFORMER — Informe de gramática        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Estadísticas globales
    root_dur = sum(
        elements[t].duration_beats if isinstance(t, str) and t in elements else (t[1] if isinstance(t, tuple) else 0)
        for t in root
    )
    print(f"\n  Reglas aprendidas : {len(rules)}")
    print(f"  Tokens en raíz    : {len(root)}")
    print(f"  Duración total    : {root_dur:.2f} beats")

    # Agrupar por duración exacta
    dur_groups: dict = defaultdict(list)
    for name, elem in elements.items():
        if name == '__root__':
            continue
        key = round(elem.duration_beats, 3)
        dur_groups[key].append(name)

    print(f"\n  Elementos por longitud (beats):")
    print(f"  {'Longitud':>10}  {'N elem':>7}  {'Nombres'}")
    print("  " + "─" * 60)
    for dur in sorted(dur_groups.keys()):
        names = dur_groups[dur]
        names_str = ", ".join(names[:8])
        if len(names) > 8:
            names_str += f" … +{len(names)-8} más"
        print(f"  {dur:>10.3f}  {len(names):>7}  {names_str}")

    # ── Sección bins ──────────────────────────────────────────────────────
    if bin_size is not None:
        bins = build_bins(elements, bin_size)
        max_count = max((len(v) for v in bins.values()), default=1)
        bar_width = 30

        print(f"\n  Bins (bin_size={bin_size}b):")
        print(f"  {'Bin':>5}  {'Rango':>14}  {'N':>4}  {'Elementos'}")
        print("  " + "─" * 70)
        for idx in sorted(bins.keys()):
            names = bins[idx]
            lo = idx * bin_size
            hi = (idx + 1) * bin_size
            bar = "█" * int(len(names) / max_count * bar_width)
            names_str = ", ".join(names[:6])
            if len(names) > 6:
                names_str += f" +{len(names)-6}"
            print(f"  {idx:>5}  [{lo:.2f}, {hi:.2f})  {len(names):>4}  {bar}  {names_str}")

    # Detalle de cada regla
    print(f"\n  Detalle de reglas:")
    print(f"  {'Nombre':>8}  {'Dur(b)':>7}  {'N notas':>7}  {'Ocurr.':>7}  Tokens")
    print("  " + "─" * 70)
    for name in sorted(rules.keys()):
        elem = elements.get(name)
        if not elem:
            continue
        body_preview = str(rules[name][:4])
        if len(rules[name]) > 4:
            body_preview = body_preview[:-1] + ", ...]"
        # Anotar el bin si aplica
        bin_tag = ""
        if bin_size is not None:
            bin_tag = f"  bin{int(elem.duration_beats / bin_size)}"
        print(f"  {name:>8}  {elem.duration_beats:>7.3f}  {len(elem.notes):>7}  {elem.occurrences:>7}  {body_preview}{bin_tag}")

    # Matriz de compatibilidad para la longitud objetivo (si se indica)
    if target_duration is not None:
        candidates, matrix = build_compatibility_matrix(
            elements, target_duration, normalize_pitch=normalize_pitch
        )
        print(f"\n  Matriz de compatibilidad — longitud {target_duration:.3f} beats "
              f"({'pitch-norm' if normalize_pitch else 'absoluto'}):")
        if not candidates:
            print("    (ningún elemento con esa longitud)")
        else:
            header = "         " + "".join(f"{c:>9}" for c in candidates)
            print(f"  {header}")
            for i, ri in enumerate(candidates):
                row = "".join(f"  {matrix[i,j]:.4f}" for j in range(len(candidates)))
                print(f"  {ri:>8}{row}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def run_transform(
    midi_path: Path,
    out_path: Path,
    target_duration: float,
    temperature: float,
    normalize_pitch: bool,
    bin_size: float | None,
    bin_length_range: tuple | None,
    fit_mode: str,
    bar_tokens: bool,
    multitrack: bool,
    continuity_weight: float,
    target_key: tuple | None,
    min_pair: int,
    max_rules: int | None,
    track_idx: int | None,
    export_dir: Path | None,
    export_format: str,
    import_dir: Path | None,
    report: bool,
    report_format: str,
    seed: float | None,
    verbose: bool,
) -> None:

    rng = random.Random(seed)

    # ── 1. Cargar MIDI
    print(f"[1/6] Cargando MIDI: {midi_path} …")
    try:
        mid = MidiFile(str(midi_path))
    except Exception as e:
        sys.exit(f"  Error al leer el MIDI: {e}")

    t_idx = track_idx if track_idx is not None else choose_main_track(mid)
    tempo = get_tempo(mid)
    tpb   = mid.ticks_per_beat
    bpm   = round(60_000_000 / tempo)
    print(f"  → {len(mid.tracks)} track(s), tpb={tpb}, tempo={bpm} BPM, track={t_idx}")

    # ── 2. Extraer notas
    print(f"[2/6] Extrayendo notas …")
    if multitrack:
        all_tracks_flat = extract_all_tracks_flat(mid)
        notes_flat = all_tracks_flat.get(t_idx, extract_notes_flat(mid, t_idx))
        n_tracks = len(all_tracks_flat)
        n_notes = sum(len(v) for v in all_tracks_flat.values())
        print(f"  → {n_tracks} tracks con notas, {n_notes} notas totales (gramática track {t_idx}: {len(notes_flat)} notas)")
    else:
        notes_flat = extract_notes_flat(mid, t_idx)
        all_tracks_flat = {t_idx: notes_flat}
        print(f"  → {len(notes_flat)} notas extraídas")

    # ── 3. Re-Pair
    mode_tag = "compases" if bar_tokens else "notas"
    print(f"[3/6] Aprendiendo gramática Re-Pair (nivel {mode_tag}, min_pair={min_pair}) …")
    bar_notes_map = {}
    if bar_tokens:
        sequence, bar_notes_map = extract_bar_sequence(notes_flat, tpb, mid)
        print(f"  → {len(sequence)} compases, {len(set(sequence))} tipos únicos")
        rules, root = repaint(sequence, min_pair=min_pair, max_rules=max_rules, prefix='B', verbose=verbose)
    else:
        note_seq = [note_token(n) for _, n in notes_flat]
        rules, root = repaint(note_seq, min_pair=min_pair, max_rules=max_rules, prefix='N', verbose=verbose)
    print(f"  → {len(rules)} reglas aprendidas, {len(root)} tokens en raíz")

    # ── 4. Construir elementos
    print(f"[4/6] Construyendo elementos de la gramática …")
    if bar_tokens:
        elements = build_grammar_elements_from_bars(rules, root, bar_notes_map, verbose=verbose)
    else:
        elements = build_grammar_elements_from_notes(rules, root, notes_flat, verbose=verbose)
    print(f"  → {len(elements)} elementos construidos")

    # Importar elementos adicionales
    if import_dir is not None:
        print(f"  Importando elementos desde {import_dir} …")
        imported = import_elements(import_dir, verbose=verbose)
        elements.update(imported)
        print(f"  → Pool total: {len(elements)} elementos")

    # Exportar elementos
    if export_dir is not None:
        print(f"  Exportando elementos a {export_dir} …")
        export_elements(elements, export_dir,
                        export_format=export_format,
                        tpb=tpb, tempo=tempo,
                        verbose=verbose)

    # Informe
    if report:
        if report_format in ('text', 'both'):
            print_report(rules, root, elements, target_duration, normalize_pitch,
                         bin_size=bin_size)
        if report_format in ('html', 'both'):
            html_path = midi_path.with_suffix('.grammar.html')
            generate_html_report(
                rules, root, elements, notes_flat,
                target_duration=target_duration,
                normalize_pitch=normalize_pitch,
                bin_size=bin_size,
                out_path=html_path,
                midi_name=midi_path.name,
            )

    if target_duration is None and bin_size is None:
        print("\n  (Sin --length ni --bin-size: no se aplica transformación. Usa --report para explorar.)")
        return

    # ── 5. Transformar gramática
    if bin_size is not None:
        range_tag = (f", rango [{bin_length_range[0]:.2f}, {bin_length_range[1]:.2f}]b"
                     if bin_length_range else "")
        print(f"[5/6] Transformando en modo bins "
              f"(bin_size={bin_size}b{range_tag}, fit={fit_mode}, T={temperature}"
              + (", pitch-norm" if normalize_pitch else "") + ") …")
        new_rules, new_root, elements = transform_grammar_bins(
            rules, root, elements,
            bin_size=bin_size,
            fit_mode=fit_mode,
            temperature=temperature,
            normalize_pitch=normalize_pitch,
            rng=rng,
            bin_length_range=bin_length_range,
            verbose=verbose,
        )
    else:
        print(f"[5/6] Transformando (length={target_duration:.2f}b, T={temperature}"
              + (", pitch-norm" if normalize_pitch else "") + ") …")
        new_rules, new_root = transform_grammar(
            rules, root, elements,
            target_duration=target_duration,
            temperature=temperature,
            normalize_pitch=normalize_pitch,
            rng=rng,
            verbose=verbose,
        )

    # ── 6. Reconstruir y escribir MIDI
    print(f"[6/6] Reconstruyendo MIDI …")
    new_notes = reconstruct_notes(new_root, new_rules, elements)
    print(f"  → {len(new_notes)} notas en track principal")

    if not new_notes:
        print("  [AVISO] No se generaron notas. Revisa los parámetros.")
        return

    print(f"  Escribiendo: {out_path} …")
    if multitrack and len(all_tracks_flat) > 1:
        # El track principal ya fue transformado. Para los tracks secundarios:
        # construimos su propia gramática, aplicamos las mismas sustituciones
        # que aprendimos del principal (mismos nombres de regla → mismo patrón
        # de reemplazo), y los reconstruimos independientemente.
        tracks_notes: dict = {}
        for tidx, nf in all_tracks_flat.items():
            if tidx == t_idx:
                tracks_notes[tidx] = new_notes
            else:
                # Track secundario: pipeline de acordes para preservar polifonía
                sec_seq, sec_chord_map, sec_onset_list = extract_chord_sequence(nf)
                n_chords = len(sec_seq)
                n_unique = len(set(sec_seq))
                print(f"  Track {tidx} (acompañamiento): "
                      f"{len(nf)} notas, {n_chords} acordes, {n_unique} únicos")
                sec_rules, sec_root = repaint(
                    sec_seq, min_pair=2, prefix='S')
                sec_elements, sec_rules, sec_root = build_grammar_elements_from_chords(
                    sec_rules, sec_root, sec_chord_map, sec_onset_list)
                if bin_size is not None:
                    sec_new_rules, sec_new_root, sec_elements = transform_grammar_bins(
                        sec_rules, sec_root, sec_elements,
                        bin_size=bin_size, fit_mode=fit_mode,
                        temperature=temperature,
                        normalize_pitch=normalize_pitch,
                        rng=random.Random(rng.randint(0, 99999)),
                        bin_length_range=bin_length_range,
                        continuity_weight=continuity_weight,
                        target_key=target_key,
                    )
                else:
                    sec_new_rules, sec_new_root = transform_grammar(
                        sec_rules, sec_root, sec_elements,
                        target_duration=target_duration,
                        temperature=temperature,
                        normalize_pitch=normalize_pitch,
                        rng=random.Random(rng.randint(0, 99999)),
                        continuity_weight=continuity_weight,
                        target_key=target_key,
                    )
                sec_notes = reconstruct_notes(
                    sec_new_root, sec_new_rules, sec_elements)
                tracks_notes[tidx] = sec_notes
                n_out_chords = len(set(round(off, 4) for off, _ in sec_notes))
                n_out_poly = len(sec_notes)
                print(f"  → Track {tidx}: {n_out_poly} notas, "
                      f"~{n_out_chords} posiciones de acorde")
        total = sum(len(v) for v in tracks_notes.values())
        print(f"  → {len(tracks_notes)} tracks, {total} notas en total")
        multitrack_to_midi(tracks_notes, tpb, tempo, out_path, original_mid=mid)
    else:
        notes_to_midi_file(new_notes, tpb, tempo, out_path)

    print(f"  → Fichero guardado: {out_path}")
    print("\n✓ Grammar Transformer finalizado correctamente.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE LÍNEA DE COMANDOS
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='GRAMMAR TRANSFORMER — Transformación MIDI por gramática Re-Pair',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('midi', type=Path, help='MIDI de entrada')
    p.add_argument('--out', '-o', type=Path, default=None,
                   metavar='FILE', help='MIDI de salida (default: entrada.transformed.mid)')
    p.add_argument('--length', '-l', type=float, default=None,
                   metavar='BEATS',
                   help='Longitud en beats de los elementos a transformar (modo exacto)')
    p.add_argument('--bin-size', type=float, default=None,
                   metavar='BEATS',
                   help='Ancho del bin en beats (modo bins; activa agrupación por longitud)')
    p.add_argument('--bin-length', type=float, nargs=2, default=None,
                   metavar=('MIN', 'MAX'),
                   help='Rango de duraciones a transformar en modo bins, en beats '
                        '(ej: --bin-length 0.5 4.0). Elementos fuera del rango se dejan intactos.')
    p.add_argument('--fit', choices=['scale', 'pad', 'trim'], default='scale',
                   metavar='MODE',
                   help='Ajuste al insertar: scale|pad|trim (default: scale)')
    p.add_argument('--bar-tokens', action='store_true',
                   help='Re-Pair a nivel de compas en lugar de nota')
    p.add_argument('--multitrack', action='store_true',
                   help='Procesar todos los tracks coordinadamente')
    p.add_argument('--continuity', type=float, default=0.0, metavar='W',
                   help='Peso penalizacion de frontera [0,1] (0=desactivado, rec: 0.3-0.8)')
    p.add_argument('--target-key', type=str, default=None, metavar='KEY',
                   help='Transponer insercion a tonalidad (ej: C, Dm, F#, Bbm, Am)')
    p.add_argument('--temp', '-t', type=float, default=0.5,
                   metavar='FLOAT',
                   help='Temperatura de sampling (0=greedy, 1=uniforme, default=0.5)')
    p.add_argument('--pitch-norm', action='store_true',
                   help='Normalizar pitch para compatibilidad y transponer al insertar')
    p.add_argument('--report', action='store_true',
                   help='Mostrar informe de gramática y estadísticas')
    p.add_argument('--report-format', choices=['text', 'html', 'both'], default='text',
                   metavar='FMT',
                   help='Formato del informe: text (stdout), html (fichero), both (default: text)')
    p.add_argument('--export-elements', type=Path, default=None,
                   metavar='DIR', help='Exportar elementos de la gramática a directorio')
    p.add_argument('--export-format', choices=['json', 'midi', 'both'], default='json',
                   metavar='FMT',
                   help='Formato de exportación: json, midi, both (default: json)')
    p.add_argument('--import-elements', type=Path, default=None,
                   metavar='DIR', help='Importar elementos adicionales desde directorio')
    p.add_argument('--min-pair', type=int, default=2,
                   metavar='N', help='Frecuencia mínima para crear regla (default: 2)')
    p.add_argument('--max-rules', type=int, default=None,
                   metavar='N', help='Número máximo de reglas a generar')
    p.add_argument('--track', type=int, default=None,
                   metavar='N', help='Track del MIDI a usar (default: auto)')
    p.add_argument('--seed', type=float, default=None,
                   metavar='FLOAT', help='Semilla aleatoria para reproducibilidad')
    p.add_argument('--verbose', action='store_true',
                   help='Traza detallada del proceso')
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.midi.exists():
        parser.error(f"Fichero MIDI no encontrado: {args.midi}")

    out_path = args.out or args.midi.with_suffix('.transformed.mid')

    target_key = None
    if args.target_key:
        try:
            target_key = parse_target_key(args.target_key)
        except Exception:
            parser.error(f"Tonalidad no reconocida: '{args.target_key}'. Ej: C, Dm, F#m, Bb, Am")

    run_transform(
        midi_path        = args.midi,
        out_path         = out_path,
        target_duration  = args.length,
        temperature      = args.temp,
        normalize_pitch  = args.pitch_norm,
        bin_size         = args.bin_size,
        bin_length_range = tuple(args.bin_length) if args.bin_length else None,
        fit_mode         = args.fit,
        bar_tokens       = args.bar_tokens,
        multitrack       = args.multitrack,
        continuity_weight= args.continuity,
        target_key       = target_key,
        min_pair         = args.min_pair,
        max_rules       = args.max_rules,
        track_idx        = args.track,
        export_dir       = args.export_elements,
        export_format    = args.export_format,
        import_dir       = args.import_elements,
        report           = args.report,
        report_format    = args.report_format,
        seed             = args.seed,
        verbose          = args.verbose,
    )


if __name__ == '__main__':
    main()
