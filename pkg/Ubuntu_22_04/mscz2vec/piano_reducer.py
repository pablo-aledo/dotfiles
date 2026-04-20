#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       PIANO REDUCER  v1.0                                    ║
║     Reducción para piano de obras orquestales / multitracks MIDI            ║
║                                                                              ║
║  Lee un MIDI multitracks (o una partitura.yaml) y genera una reducción      ║
║  pianística idiomática en dos pentagramas (mano derecha / mano izquierda)   ║
║  lista para evaluar en FL Studio o imprimir con MuseScore/LilyPond.         ║
║                                                                              ║
║  ESTRATEGIAS DE REDUCCIÓN (--strategy):                                      ║
║    smart      Asignación inteligente por rol musical detectado (default)    ║
║    register   MD = notas > umbral, MI = notas <= umbral (--split C4)        ║
║    layer      Capas impares → MD, capas pares → MI                         ║
║    priority   Lista de pistas prioritarias explícita (--rh-tracks / --lh)  ║
║    melody+bass  Melodía principal en MD, bajo+armonía en MI                 ║
║                                                                              ║
║  TÉCNICAS PIANÍSTICAS APLICADAS:                                             ║
║  [P1]  Reducción de notas simultáneas al límite polifónico por mano         ║
║  [P2]  Octavación idiomática: notas fuera de rango transpuestas ±1 oct      ║
║  [P3]  Bajo de Alberti / arpegios en MI para acordes bloques > 3 voces     ║
║  [P4]  Compresión armónica: acordes de ≥5 notas → tétradas representativas  ║
║  [P5]  Notas de paso melódicas inferidas entre saltos de 3ª/4ª              ║
║  [P6]  Cruce de manos detectado y resuelto                                  ║
║  [P7]  Unificación de duplicaciones: misma nota en ambas manos → una sola  ║
║  [P8]  Humanización dinámica por voz: melodía ff, acomp. mp, bajo mf       ║
║                                                                              ║
║  MODOS DE ENTRADA:                                                           ║
║    piano_reducer.py obra.mid                  — MIDI multitracks           ║
║    piano_reducer.py obra.mid --strategy smart                               ║
║    piano_reducer.py obra.mid --rh-tracks "Violin I,Oboe" --lh-tracks "Cello,Bass"
║    piano_reducer.py obra.mid --from-yaml partitura.yaml                     ║
║    piano_reducer.py obra.mid --split C4 --strategy register                 ║
║    piano_reducer.py obra.mid --max-voices-rh 4 --max-voices-lh 3           ║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    <stem>_piano.mid          — MIDI de dos pistas: MD (ch1) y MI (ch2)     ║
║    <stem>_piano_report.txt   — Informe de reducción: decisiones, análisis  ║
║    <stem>_piano.yaml         — Partitura reducida en formato yaml (opt.)   ║
║    <stem>_piano_combined.mid — Reducción + original combinados (opt.)      ║
║                                                                              ║
║  USO:                                                                        ║
║    python piano_reducer.py obra.mid                                          ║
║    python piano_reducer.py obra.mid --strategy melody+bass --verbose        ║
║    python piano_reducer.py obra.mid --split C4 --strategy register          ║
║    python piano_reducer.py obra.mid --rh-tracks "Violin I,Flute"            ║
║    python piano_reducer.py obra.mid --no-alberti --no-passing-notes         ║
║    python piano_reducer.py obra.mid --from-yaml partitura.yaml              ║
║    python piano_reducer.py obra.mid --output reduccion --combine-output     ║
║    python piano_reducer.py obra.mid --report-only                           ║
║    python piano_reducer.py obra.mid --export-yaml                           ║
║    python piano_reducer.py obra.mid --max-polyphony 5                       ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --strategy      smart | register | layer | priority | melody+bass        ║
║    --split         Nota de split register (default: C4 = 60)                ║
║    --rh-tracks     Nombres de pistas → mano derecha (coma-separados)        ║
║    --lh-tracks     Nombres de pistas → mano izquierda (coma-separados)      ║
║    --max-polyphony Límite de voces simultáneas total (default: 7)           ║
║    --max-voices-rh Límite de voces MD (default: 4)                          ║
║    --max-voices-lh Límite de voces MI (default: 3)                          ║
║    --no-alberti    No aplicar bajo de Alberti en MI                         ║
║    --no-compress   No comprimir acordes > 4 notas                           ║
║    --no-octave     No corregir notas fuera de rango idiomático               ║
║    --no-passing    No añadir notas de paso inferidas                        ║
║    --from-yaml     Leer también la partitura.yaml para metadatos            ║
║    --export-yaml   Exportar reducción en formato yaml                       ║
║    --combine-output  Generar MIDI combinado (original + reducción)          ║
║    --report-only   Solo análisis, sin generar MIDI                          ║
║    --output        Nombre base de salida (default: <stem>_piano)            ║
║    --verbose       Decisiones detalladas por compás                         ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install mido numpy                                                    ║
║    (opcional) pip install PyYAML   ← para --from-yaml y --export-yaml      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import argparse
import random
import copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy")
    sys.exit(1)

# yaml es opcional
try:
    import yaml
    YAML_OK = True
except ImportError:
    YAML_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

TICKS = 480       # ticks per beat (se sobreescribe al leer el MIDI)
TICKS_PER_BAR = TICKS * 4

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

# Rango idiomático del piano
PIANO_RANGE   = (21, 108)   # A0 – C8
PIANO_RH_LOW  = 48          # C3  — nota más baja "natural" para MD
PIANO_LH_HIGH = 72          # C5  — nota más alta "natural" para MI
PIANO_SPLIT_DEFAULT = 60    # C4

# Roles musicales detectables
ROLE_MELODY      = 'melody'
ROLE_BASS        = 'bass'
ROLE_HARMONY     = 'harmony'
ROLE_COUNTERPOINT = 'counterpoint'
ROLE_PERCUSSION  = 'percussion'
ROLE_FILLER      = 'filler'

# Pistas de percusión (canal MIDI 10, canal 9 en 0-indexed)
PERC_CHANNEL = 9

# Límite de voces simultáneas por mano
MAX_VOICES_RH = 4
MAX_VOICES_LH = 3


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE NOTAS
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"

def name_to_midi(name: str) -> int:
    """'C4' → 60, 'A#3' → 58, 'Bb4' → 70, etc."""
    note_map = {n: i for i, n in enumerate(NOTE_NAMES)}
    # también aceptar bemoles con 'b'
    flat_map = {'Db':1,'Eb':3,'Fb':4,'Gb':6,'Ab':8,'Bb':10,'Cb':11}
    name = name.strip()
    upper = name[0].upper() + name[1:]
    # intento bemol
    if len(upper) >= 2 and upper[1] == 'b' and upper[:2] in flat_map:
        pc = flat_map[upper[:2]]
        octave = int(upper[2:])
    elif len(upper) >= 2 and upper[1] == '#':
        pc = note_map.get(upper[:2], 0)
        octave = int(upper[2:])
    else:
        pc = note_map.get(upper[0], 0)
        octave = int(upper[1:])
    return (octave + 1) * 12 + pc

def note_in_piano_range(n: int) -> bool:
    return PIANO_RANGE[0] <= n <= PIANO_RANGE[1]

def clamp_to_piano(n: int) -> int:
    """Transpone por octavas hasta que quede en rango pianístico."""
    while n < PIANO_RANGE[0]:
        n += 12
    while n > PIANO_RANGE[1]:
        n -= 12
    return n

def interval_semitones(a: int, b: int) -> int:
    return abs(a - b)


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DEL MIDI
# ══════════════════════════════════════════════════════════════════════════════

def load_midi(path: str) -> Tuple[MidiFile, int, int]:
    """Carga MIDI y devuelve (midi_obj, tpb, tempo_us)."""
    mid = MidiFile(path)
    tpb = mid.ticks_per_beat
    tempo = 500000  # 120 BPM por defecto
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    return mid, tpb, tempo

def extract_tracks(mid: MidiFile, tpb: int) -> Dict[str, List[dict]]:
    """
    Extrae notas absolutas por pista.
    Devuelve dict {track_name: [ {tick, pitch, velocity, duration, channel}, ... ] }
    """
    tracks_notes = {}

    for i, track in enumerate(mid.tracks):
        name = track.name.strip() if track.name.strip() else f"Track_{i}"
        notes_on = {}   # (channel, pitch) → tick_on
        notes = []
        abs_tick = 0

        for msg in track:
            abs_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes_on[(msg.channel, msg.note)] = (abs_tick, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in notes_on:
                    t_on, vel = notes_on.pop(key)
                    dur = abs_tick - t_on
                    if dur > 0 and msg.channel != PERC_CHANNEL:
                        notes.append({
                            'tick': t_on,
                            'pitch': msg.note,
                            'velocity': vel,
                            'duration': dur,
                            'channel': msg.channel
                        })

        if notes:
            tracks_notes[name] = notes

    return tracks_notes


def collect_programs(mid: MidiFile) -> Dict[int, int]:
    """Devuelve {channel: program} para cada canal."""
    programs = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'program_change':
                programs[msg.channel] = msg.program
    return programs


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE ROLES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

def analyze_track_role(notes: List[dict], all_tracks: Dict[str, List[dict]]) -> str:
    """
    Detecta el rol musical de una pista:
      - melody      : registro agudo, mayor variabilidad de alturas, línea continua
      - bass        : registro grave consistente (< C3)
      - harmony     : densidad de notas alta, acordes frecuentes, registro medio
      - counterpoint: registro medio-agudo, melodía independiente de la principal
      - filler      : notas muy cortas, tremolo, relleno textural
    """
    if not notes:
        return ROLE_FILLER

    pitches = [n['pitch'] for n in notes]
    mean_pitch = np.mean(pitches)
    std_pitch  = np.std(pitches)
    durations  = [n['duration'] for n in notes]
    mean_dur   = np.mean(durations)

    # Detectar acordes (notas solapadas)
    notes_sorted = sorted(notes, key=lambda x: x['tick'])
    chords = 0
    for i in range(1, len(notes_sorted)):
        prev = notes_sorted[i-1]
        curr = notes_sorted[i]
        if curr['tick'] < prev['tick'] + prev['duration']:
            chords += 1
    chord_ratio = chords / max(len(notes), 1)

    # Reglas de clasificación
    if mean_pitch < 48:                      # por debajo de C3
        return ROLE_BASS
    if mean_pitch > 72 and std_pitch > 8:    # agudo y variable → melodía
        return ROLE_MELODY
    if chord_ratio > 0.4:                    # muchas notas simultáneas → armonía
        return ROLE_HARMONY
    if mean_pitch > 60 and std_pitch > 6:    # registro medio-alto, variable
        return ROLE_COUNTERPOINT
    if mean_dur < TICKS * 0.25:              # muy cortas → relleno
        return ROLE_FILLER
    if mean_pitch > 55:
        return ROLE_MELODY                   # default agudo
    return ROLE_HARMONY                      # default medio


def detect_melody_track(tracks: Dict[str, List[dict]]) -> Optional[str]:
    """Identifica la pista de melodía principal (pitch más agudo + continuidad)."""
    best_name = None
    best_score = -1
    for name, notes in tracks.items():
        if not notes:
            continue
        pitches = [n['pitch'] for n in notes]
        score = np.mean(pitches) + np.std(pitches) * 0.5
        if score > best_score:
            best_score = score
            best_name = name
    return best_name

def detect_bass_track(tracks: Dict[str, List[dict]]) -> Optional[str]:
    """Identifica la pista de bajo (pitch más grave consistente)."""
    best_name = None
    best_score = 999
    for name, notes in tracks.items():
        if not notes:
            continue
        mean_p = np.mean([n['pitch'] for n in notes])
        if mean_p < best_score:
            best_score = mean_p
            best_name = name
    return best_name


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRATEGIAS DE ASIGNACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def assign_smart(tracks: Dict[str, List[dict]], args) -> Tuple[List[str], List[str]]:
    """
    Asignación inteligente por rol:
      MD ← melody, counterpoint
      MI ← bass, harmony, filler
    """
    rh_tracks, lh_tracks = [], []
    roles = {name: analyze_track_role(notes, tracks) for name, notes in tracks.items()}

    if args.verbose:
        print("\n  Roles detectados:")
        for name, role in roles.items():
            print(f"    [{name:<24}] → {role}")

    for name, role in roles.items():
        if role in (ROLE_MELODY, ROLE_COUNTERPOINT):
            rh_tracks.append(name)
        else:
            lh_tracks.append(name)

    # Fallback: si una mano queda vacía, balancear por registro
    if not rh_tracks or not lh_tracks:
        rh_tracks, lh_tracks = assign_register(tracks, args)

    return rh_tracks, lh_tracks


def assign_register(tracks: Dict[str, List[dict]], args) -> Tuple[List[str], List[str]]:
    """Asignación por umbral de registro."""
    split = args.split_note
    rh_tracks, lh_tracks = [], []
    for name, notes in tracks.items():
        if not notes:
            continue
        mean_p = np.mean([n['pitch'] for n in notes])
        if mean_p >= split:
            rh_tracks.append(name)
        else:
            lh_tracks.append(name)
    return rh_tracks, lh_tracks


def assign_layer(tracks: Dict[str, List[dict]], args) -> Tuple[List[str], List[str]]:
    """Pistas impares → MD, pares → MI."""
    names = list(tracks.keys())
    rh = [names[i] for i in range(0, len(names), 2)]
    lh = [names[i] for i in range(1, len(names), 2)]
    return rh, lh


def assign_priority(tracks: Dict[str, List[dict]], args) -> Tuple[List[str], List[str]]:
    """Asignación por listas explícitas (--rh-tracks / --lh-tracks)."""
    rh_names = set(t.strip() for t in (args.rh_tracks or '').split(',') if t.strip())
    lh_names = set(t.strip() for t in (args.lh_tracks or '').split(',') if t.strip())
    rh_tracks, lh_tracks = [], []

    for name in tracks:
        matched_rh = any(r.lower() in name.lower() or name.lower() in r.lower() for r in rh_names)
        matched_lh = any(l.lower() in name.lower() or name.lower() in l.lower() for l in lh_names)
        if matched_rh:
            rh_tracks.append(name)
        elif matched_lh:
            lh_tracks.append(name)
        else:
            # No especificado → smart fallback
            role = analyze_track_role(tracks[name], tracks)
            if role in (ROLE_MELODY, ROLE_COUNTERPOINT):
                rh_tracks.append(name)
            else:
                lh_tracks.append(name)

    return rh_tracks, lh_tracks


def assign_melody_bass(tracks: Dict[str, List[dict]], args) -> Tuple[List[str], List[str]]:
    """Melodía + contrapunto en MD, bajo + armonía en MI."""
    melody  = detect_melody_track(tracks)
    bass    = detect_bass_track(tracks)
    rh_tracks = [melody] if melody else []
    lh_tracks = [bass] if bass else []

    for name in tracks:
        if name == melody or name == bass:
            continue
        role = analyze_track_role(tracks[name], tracks)
        if role == ROLE_COUNTERPOINT:
            rh_tracks.append(name)
        else:
            lh_tracks.append(name)

    return rh_tracks, lh_tracks


STRATEGIES = {
    'smart':       assign_smart,
    'register':    assign_register,
    'layer':       assign_layer,
    'priority':    assign_priority,
    'melody+bass': assign_melody_bass,
}


# ══════════════════════════════════════════════════════════════════════════════
#  FUSIÓN DE PISTAS EN UNA VOZ
# ══════════════════════════════════════════════════════════════════════════════

def merge_tracks(track_names: List[str], all_tracks: Dict[str, List[dict]]) -> List[dict]:
    """Combina múltiples pistas en una sola lista de notas ordenada por tick."""
    merged = []
    for name in track_names:
        merged.extend(all_tracks.get(name, []))
    merged.sort(key=lambda n: (n['tick'], -n['pitch']))
    return merged


# ══════════════════════════════════════════════════════════════════════════════
#  SPLIT INTERNO DE PISTA ÚNICA (caso piano solo / MIDI monolítico)
# ══════════════════════════════════════════════════════════════════════════════

def split_single_track(notes: List[dict], split_note: int,
                       tpb: int, verbose: bool = False) -> Tuple[List[dict], List[dict]]:
    """
    Divide las notas de una pista única en dos manos usando tres criterios:

    1. SPLIT POR REGISTRO: notas >= split_note → MD, notas < split_note → MI.
    2. SEPARACIÓN POLIFÓNICA: en acordes simultáneos, voces superiores → MD,
       inferiores → MI (captura piano escrito con ambas manos en una pista).
    3. REFINAMIENTO melódico: notas MI que son más agudas que notas MD
       cercanas se repatrían a MD (melodía que baja temporalmente).

    Devuelve (rh_notes, lh_notes).
    """
    SLACK = tpb // 8  # ventana de simultaneidad

    sorted_notes = sorted(notes, key=lambda n: (n['tick'], -n['pitch']))
    clusters: List[List[dict]] = []

    i = 0
    while i < len(sorted_notes):
        cluster = [sorted_notes[i]]
        j = i + 1
        while j < len(sorted_notes) and sorted_notes[j]['tick'] - sorted_notes[i]['tick'] <= SLACK:
            cluster.append(sorted_notes[j])
            j += 1
        clusters.append(cluster)
        i = j

    rh_notes: List[dict] = []
    lh_notes: List[dict] = []

    for cluster in clusters:
        cs = sorted(cluster, key=lambda n: -n['pitch'])  # agudas primero
        n_total = len(cs)

        if n_total == 1:
            if cs[0]['pitch'] >= split_note:
                rh_notes.extend(cs)
            else:
                lh_notes.extend(cs)
        elif n_total == 2:
            rh_notes.append(cs[0])
            lh_notes.append(cs[1])
        else:
            above = [n for n in cs if n['pitch'] >= split_note]
            below = [n for n in cs if n['pitch'] <  split_note]
            if above and below:
                rh_notes.extend(above)
                lh_notes.extend(below)
            else:
                rh_notes.append(cs[0])
                lh_notes.extend(cs[1:])

    # Refinamiento: recuperar notas MI más agudas que la voz MD cercana
    if rh_notes and lh_notes:
        rh_by_slot: Dict[int, List[dict]] = defaultdict(list)
        for n in rh_notes:
            rh_by_slot[n['tick'] // max(SLACK, 1)].append(n)

        lh_keep = []
        lh_promote = []
        for n in lh_notes:
            slot = n['tick'] // max(SLACK, 1)
            nearby = rh_by_slot.get(slot) or rh_by_slot.get(slot-1) or rh_by_slot.get(slot+1)
            if nearby:
                rh_top = max(nearby, key=lambda x: x['pitch'])
                if n['pitch'] > rh_top['pitch'] + 2:
                    lh_promote.append(n)
                    continue
            lh_keep.append(n)

        rh_notes = sorted(rh_notes + lh_promote,
                          key=lambda n: (n['tick'], -n['pitch']))
        lh_notes = lh_keep

    if verbose:
        print(f"  Split interno ({midi_to_name(split_note)}): "
              f"MD={len(rh_notes)}  MI={len(lh_notes)}")

    return rh_notes, lh_notes


def detect_single_track_situation(tracks: Dict[str, List[dict]]) -> bool:
    """
    True si el MIDI tiene una sola pista con notas, o todas en el mismo canal
    (piano solo escrito en una pista).
    """
    non_empty = {k: v for k, v in tracks.items() if v}
    if len(non_empty) <= 1:
        return True
    all_channels = set()
    for notes in non_empty.values():
        all_channels.update(n['channel'] for n in notes)
    return len(all_channels) == 1


# ══════════════════════════════════════════════════════════════════════════════
#  TÉCNICAS DE REDUCCIÓN PIANÍSTICA
# ══════════════════════════════════════════════════════════════════════════════

def limit_polyphony(notes: List[dict], max_voices: int, prefer='top') -> List[dict]:
    """
    [P1] Limita el número de notas simultáneas.
    prefer='top'    → conserva las notas más agudas (MD)
    prefer='bottom' → conserva las más graves (MI)
    """
    if not notes:
        return notes

    # Agrupar por ventanas de simultaneidad (tick idéntico o muy cercano)
    SLACK = TICKS // 16   # 1/64 de negra — notas "casi simultáneas"
    result = []
    i = 0
    while i < len(notes):
        group = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j]['tick'] - notes[i]['tick'] <= SLACK:
            group.append(notes[j])
            j += 1

        if len(group) > max_voices:
            group.sort(key=lambda n: n['pitch'], reverse=(prefer == 'top'))
            group = group[:max_voices]

        result.extend(group)
        i = j

    return sorted(result, key=lambda n: (n['tick'], n['pitch']))


def fix_octaves(notes: List[dict], is_rh: bool, force: bool = True) -> List[dict]:
    """
    [P2] Transpone notas fuera del rango pianístico por octavas.
    También aplica heurística de idiomatismo:
      MD: preferiblemente > C3 (48)
      MI: preferiblemente < C5 (72)
    """
    result = []
    for note in notes:
        p = note['pitch']
        p = clamp_to_piano(p)
        if is_rh and p < PIANO_RH_LOW and force:
            while p < PIANO_RH_LOW and p + 12 <= PIANO_RANGE[1]:
                p += 12
        elif not is_rh and p > PIANO_LH_HIGH and force:
            while p > PIANO_LH_HIGH and p - 12 >= PIANO_RANGE[0]:
                p -= 12
        result.append({**note, 'pitch': p})
    return result


def compress_chords(notes: List[dict], max_notes: int = 4) -> List[dict]:
    """
    [P4] Acortes de > max_notes notas simultáneas → selección representativa.
    Conserva: raíz (más grave), 3ª, 5ª/7ª, y nota más aguda (melodía).
    """
    SLACK = TICKS // 16
    result = []
    i = 0
    sorted_notes = sorted(notes, key=lambda n: (n['tick'], n['pitch']))

    while i < len(sorted_notes):
        group = [sorted_notes[i]]
        j = i + 1
        while j < len(sorted_notes) and sorted_notes[j]['tick'] - sorted_notes[i]['tick'] <= SLACK:
            group.append(sorted_notes[j])
            j += 1

        if len(group) > max_notes:
            group_by_pitch = sorted(group, key=lambda n: n['pitch'])
            kept = [group_by_pitch[0]]                    # raíz
            kept.append(group_by_pitch[-1])               # nota más aguda
            # rellenar con notas intermedias si caben
            for idx in [len(group_by_pitch)//3, 2*len(group_by_pitch)//3]:
                if len(kept) < max_notes:
                    candidate = group_by_pitch[idx]
                    if candidate not in kept:
                        kept.append(candidate)
            group = sorted(kept, key=lambda n: n['pitch'])

        result.extend(group)
        i = j

    return sorted(result, key=lambda n: (n['tick'], n['pitch']))


def resolve_hand_crossing(rh_notes: List[dict], lh_notes: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    [P6] Detecta y resuelve cruce de manos:
    Si hay nota en MI > nota más baja de MD en el mismo tick, la sube una octava.
    """
    SLACK = TICKS // 8

    # Índice RH por tick
    rh_by_tick: Dict[int, List[int]] = defaultdict(list)
    for n in rh_notes:
        rh_by_tick[n['tick']].append(n['pitch'])

    lh_fixed = []
    crossings = 0
    for note in lh_notes:
        p = note['pitch']
        # Buscar si hay nota RH cercana más grave
        for t_rh, rh_pitches in rh_by_tick.items():
            if abs(note['tick'] - t_rh) <= SLACK:
                rh_min = min(rh_pitches)
                if p > rh_min - 2:          # cruce o muy cercano
                    while p > rh_min - 12 and p - 12 >= PIANO_RANGE[0]:
                        p -= 12
                    crossings += 1
                break
        lh_fixed.append({**note, 'pitch': p})

    return rh_notes, lh_fixed, crossings


def remove_duplicates(rh_notes: List[dict], lh_notes: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    [P7] Elimina notas duplicadas entre manos en el mismo tick.
    La MD conserva la nota; la MI la pierde.
    """
    SLACK = TICKS // 8
    rh_set: Set[Tuple[int,int]] = set()
    for n in rh_notes:
        rh_set.add((n['tick'] // SLACK, n['pitch']))

    lh_clean = []
    removed = 0
    for n in lh_notes:
        key = (n['tick'] // SLACK, n['pitch'])
        if key in rh_set:
            removed += 1
        else:
            lh_clean.append(n)

    return rh_notes, lh_clean, removed


def apply_alberti(lh_notes: List[dict], tpb: int) -> List[dict]:
    """
    [P3] Desglosa acordes de la MI en patrón Alberti (bajo-agudo-medio-agudo)
    cuando detecta acordes de 3+ notas con duración >= negra.
    Solo se aplica en pasajes con densidad armónica alta y tempo moderado.
    """
    SLACK = tpb // 8
    BEAT  = tpb

    # Agrupar notas simultáneas
    groups: Dict[int, List[dict]] = defaultdict(list)
    singles = []
    sorted_lh = sorted(lh_notes, key=lambda n: (n['tick'], n['pitch']))

    i = 0
    while i < len(sorted_lh):
        group = [sorted_lh[i]]
        j = i + 1
        while j < len(sorted_lh) and sorted_lh[j]['tick'] - sorted_lh[i]['tick'] <= SLACK:
            group.append(sorted_lh[j])
            j += 1

        tick = sorted_lh[i]['tick']
        dur  = min(n['duration'] for n in group)

        if len(group) >= 3 and dur >= BEAT:
            pitches = sorted(set(n['pitch'] for n in group))
            low, mid, high = pitches[0], pitches[len(pitches)//2], pitches[-1]
            vel_base = int(np.mean([n['velocity'] for n in group]))
            sub_dur  = dur // 4

            # Patrón: bajo – alto – medio – alto
            pattern = [low, high, mid, high]
            for k, p in enumerate(pattern):
                groups[tick].append({
                    'tick':     tick + k * sub_dur,
                    'pitch':    p,
                    'velocity': max(vel_base - k*8, 40),
                    'duration': sub_dur,
                    'channel':  group[0]['channel']
                })
        else:
            singles.extend(group)
        i = j

    alberti_notes = [n for grp in groups.values() for n in grp]
    result = sorted(singles + alberti_notes, key=lambda n: (n['tick'], n['pitch']))
    return result


def add_passing_notes(notes: List[dict], tpb: int) -> List[dict]:
    """
    [P5] Infiere notas de paso entre saltos melódicos de 3ª o 4ª
    cuando hay espacio rítmico disponible (duración > corchea).
    Solo en la melodía / línea superior de cada mano.
    """
    if len(notes) < 2:
        return notes

    CORCHEA = tpb // 2
    result = list(notes)
    extras = []

    # Trabajar solo con la nota más aguda en cada tick
    melody_line = []
    by_tick: Dict[int, List[dict]] = defaultdict(list)
    for n in notes:
        by_tick[n['tick']].append(n)
    for tick in sorted(by_tick):
        top = max(by_tick[tick], key=lambda n: n['pitch'])
        melody_line.append(top)

    for i in range(len(melody_line) - 1):
        curr = melody_line[i]
        nxt  = melody_line[i+1]
        interval = nxt['pitch'] - curr['pitch']
        time_gap = nxt['tick'] - curr['tick']

        if 2 < abs(interval) <= 5 and time_gap >= CORCHEA * 2:
            # Hay espacio para insertar una nota de paso
            step = 1 if interval > 0 else -1
            pass_pitch = curr['pitch'] + step
            pass_tick  = curr['tick'] + time_gap // 2
            pass_dur   = time_gap // 4
            extras.append({
                'tick':     pass_tick,
                'pitch':    pass_pitch,
                'velocity': max(curr['velocity'] - 15, 40),
                'duration': pass_dur,
                'channel':  curr['channel']
            })

    combined = sorted(result + extras, key=lambda n: (n['tick'], n['pitch']))
    return combined


def humanize_dynamics(rh_notes: List[dict], lh_notes: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    [P8] Ajusta velocidades para que MD suene más prominente que MI.
    MD: velocidades mínimo 60, escaladas hacia arriba.
    MI: velocidades reducidas un 20-30% para no competir.
    """
    SCALE_RH = 1.05
    SCALE_LH = 0.75
    MIN_VEL  = 40

    rh_dyn = [{**n, 'velocity': min(127, max(MIN_VEL, int(n['velocity'] * SCALE_RH)))} for n in rh_notes]
    lh_dyn = [{**n, 'velocity': min(127, max(MIN_VEL, int(n['velocity'] * SCALE_LH)))} for n in lh_notes]
    return rh_dyn, lh_dyn


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL DE REDUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def reduce(tracks: Dict[str, List[dict]], tpb: int, args) -> Tuple[List[dict], List[dict], dict]:
    """
    Pipeline completo de reducción.
    Devuelve (rh_notes, lh_notes, report_data).
    """
    report = {
        'strategy':     args.strategy,
        'tracks_in':    list(tracks.keys()),
        'rh_tracks':    [],
        'lh_tracks':    [],
        'steps':        [],
        'stats':        {}
    }

    # ── 0. Detección de MIDI monolítico (pista única o mismo canal) ────────
    is_mono = detect_single_track_situation(tracks)
    rh: List[dict] = []
    lh: List[dict] = []

    if is_mono:
        # Unir todas las notas en una masa y hacer split polifónico interno
        all_notes = []
        for notes in tracks.values():
            all_notes.extend(notes)
        all_notes.sort(key=lambda n: (n['tick'], -n['pitch']))

        print(f"\n  Pista única detectada — aplicando split interno "
              f"(split={midi_to_name(args.split_note)})")
        rh, lh = split_single_track(all_notes, args.split_note, tpb,
                                    verbose=args.verbose)
        rh_track_names = ['[interno MD]']
        lh_track_names = ['[interno MI]']
        report['rh_tracks'] = rh_track_names
        report['lh_tracks'] = lh_track_names
        report['steps'].append(
            f"Split interno pista única (split={midi_to_name(args.split_note)})")
        print(f"  Mano Derecha : {len(rh)} notas")
        print(f"  Mano Izquierda: {len(lh)} notas")
    else:
        # ── 1. Asignación de pistas a manos (flujo normal) ──────────────────
        strategy_fn = STRATEGIES.get(args.strategy, assign_smart)
        rh_track_names, lh_track_names = strategy_fn(tracks, args)

        report['rh_tracks'] = rh_track_names
        report['lh_tracks'] = lh_track_names

        print(f"\n  Mano Derecha ← {rh_track_names or ['(ninguna)']}")
        print(f"  Mano Izquierda ← {lh_track_names or ['(ninguna)']}")

        # ── 2. Fusión ────────────────────────────────────────────────────────
        rh = merge_tracks(rh_track_names, tracks)
        lh = merge_tracks(lh_track_names, tracks)

    n_rh_raw = len(rh)
    n_lh_raw = len(lh)
    if not is_mono:
        report['steps'].append(f"Fusión: MD={n_rh_raw} notas, MI={n_lh_raw} notas")
    print(f"  Notas brutas:  MD={n_rh_raw}  MI={n_lh_raw}")

    # ── 3. Compresión armónica ──────────────────────────────────────────────
    if not args.no_compress:
        rh = compress_chords(rh, max_notes=args.max_voices_rh)
        lh = compress_chords(lh, max_notes=args.max_voices_lh)
        report['steps'].append(f"Compresión: MD≤{args.max_voices_rh}, MI≤{args.max_voices_lh} voces/acorde")

    # ── 4. Límite de polifonía ──────────────────────────────────────────────
    rh = limit_polyphony(rh, args.max_voices_rh, prefer='top')
    lh = limit_polyphony(lh, args.max_voices_lh, prefer='bottom')
    report['steps'].append(f"Límite polifonía: MD={args.max_voices_rh} MI={args.max_voices_lh}")

    # ── 5. Corrección de octavas ────────────────────────────────────────────
    if not args.no_octave:
        rh = fix_octaves(rh, is_rh=True)
        lh = fix_octaves(lh, is_rh=False)
        report['steps'].append("Corrección de octavas aplicada")

    # ── 6. Bajo de Alberti ──────────────────────────────────────────────────
    if not args.no_alberti:
        lh_before = len(lh)
        lh = apply_alberti(lh, tpb)
        added = len(lh) - lh_before
        if added > 0:
            report['steps'].append(f"Bajo de Alberti: +{added} notas en MI")
            if args.verbose:
                print(f"  Alberti: +{added} notas añadidas en MI")

    # ── 7. Notas de paso ────────────────────────────────────────────────────
    if not args.no_passing:
        rh_before = len(rh)
        rh = add_passing_notes(rh, tpb)
        added = len(rh) - rh_before
        if added > 0:
            report['steps'].append(f"Notas de paso: +{added} en MD")
            if args.verbose:
                print(f"  Notas de paso: +{added} en MD")

    # ── 8. Resolución de cruce ──────────────────────────────────────────────
    rh, lh, crossings = resolve_hand_crossing(rh, lh)
    if crossings:
        report['steps'].append(f"Cruces de manos resueltos: {crossings}")
        if args.verbose:
            print(f"  Cruces resueltos: {crossings}")

    # ── 9. Eliminación de duplicados ───────────────────────────────────────
    rh, lh, dups = remove_duplicates(rh, lh)
    if dups:
        report['steps'].append(f"Duplicados eliminados de MI: {dups}")

    # ── 10. Humanización dinámica ───────────────────────────────────────────
    rh, lh = humanize_dynamics(rh, lh)
    report['steps'].append("Humanización dinámica aplicada")

    # ── Stats finales ───────────────────────────────────────────────────────
    report['stats'] = {
        'rh_notes_final': len(rh),
        'lh_notes_final': len(lh),
        'rh_notes_raw':   n_rh_raw,
        'lh_notes_raw':   n_lh_raw,
        'reduction_ratio': round((len(rh)+len(lh)) / max(n_rh_raw+n_lh_raw, 1), 3),
        'crossings_fixed': crossings,
        'duplicates_removed': dups,
    }

    print(f"  Notas finales: MD={len(rh)}  MI={len(lh)}")
    print(f"  Ratio de reducción: {report['stats']['reduction_ratio']:.1%}")

    return rh, lh, report


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL MIDI DE SALIDA
# ══════════════════════════════════════════════════════════════════════════════

def notes_to_track(notes: List[dict], tpb: int, channel: int,
                   track_name: str, program: int = 0) -> MidiTrack:
    """Convierte lista de notas absolutas a MidiTrack con tiempos delta."""
    track = MidiTrack()
    track.append(MetaMessage('track_name', name=track_name, time=0))
    track.append(Message('program_change', channel=channel, program=program, time=0))

    # Construir lista de eventos (on + off)
    events = []
    for note in notes:
        t   = note['tick']
        dur = max(note['duration'], 1)
        events.append((t,       'on',  note['pitch'], note['velocity']))
        events.append((t + dur, 'off', note['pitch'], 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))

    prev_tick = 0
    for abs_tick, ev_type, pitch, vel in events:
        delta = abs_tick - prev_tick
        prev_tick = abs_tick
        if ev_type == 'on':
            track.append(Message('note_on',  channel=channel, note=pitch, velocity=vel, time=delta))
        else:
            track.append(Message('note_off', channel=channel, note=pitch, velocity=0,   time=delta))

    return track


def build_piano_midi(rh_notes: List[dict], lh_notes: List[dict],
                     tpb: int, tempo: int, output_path: str) -> None:
    """Genera MIDI con dos pistas: MD (canal 0) y MI (canal 1)."""
    mid = MidiFile(type=1, ticks_per_beat=tpb)

    # Pista de tempo
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    tempo_track.append(MetaMessage('time_signature', numerator=4, denominator=4,
                                   clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    mid.tracks.append(tempo_track)

    # Pista MD — piano acústico (program 0)
    rh_track = notes_to_track(rh_notes, tpb, channel=0,
                               track_name='Piano MD', program=0)
    mid.tracks.append(rh_track)

    # Pista MI — piano acústico (program 0)
    lh_track = notes_to_track(lh_notes, tpb, channel=1,
                               track_name='Piano MI', program=0)
    mid.tracks.append(lh_track)

    mid.save(output_path)


def build_combined_midi(rh_notes: List[dict], lh_notes: List[dict],
                        original_mid: MidiFile, tpb: int, tempo: int,
                        output_path: str) -> None:
    """MIDI combinado: todas las pistas originales + las dos de reducción."""
    combined = MidiFile(type=1, ticks_per_beat=tpb)

    # Pistas originales
    for track in original_mid.tracks:
        combined.tracks.append(track)

    # Reducción
    ch_base = 14  # canales 14 y 15 para no colisionar
    rh_track = notes_to_track(rh_notes, tpb, channel=ch_base,
                               track_name='[REDUCCION] Piano MD', program=0)
    lh_track = notes_to_track(lh_notes, tpb, channel=ch_base+1,
                               track_name='[REDUCCION] Piano MI', program=0)
    combined.tracks.append(rh_track)
    combined.tracks.append(lh_track)
    combined.save(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN YAML
# ══════════════════════════════════════════════════════════════════════════════

def notes_to_yaml_section(notes: List[dict], tpb: int, hand: str) -> dict:
    """Convierte lista de notas a estructura yaml legible."""
    bars: Dict[int, List[dict]] = defaultdict(list)
    bar_ticks = tpb * 4
    for n in notes:
        bar = n['tick'] // bar_ticks + 1
        bars[bar].append({
            'beat':     round((n['tick'] % bar_ticks) / tpb + 1, 2),
            'pitch':    midi_to_name(n['pitch']),
            'velocity': n['velocity'],
            'dur_beats': round(n['duration'] / tpb, 3),
        })
    return {
        'hand': hand,
        'bars': {bar: sorted(ns, key=lambda x: x['beat']) for bar, ns in sorted(bars.items())}
    }


def export_yaml(rh_notes: List[dict], lh_notes: List[dict],
                tpb: int, output_path: str, source_yaml: Optional[str] = None) -> None:
    if not YAML_OK:
        print("  ⚠ PyYAML no disponible. Instala con: pip install PyYAML")
        return

    data = {
        'reduccion_piano': {
            'fuente': source_yaml or 'MIDI',
            'mano_derecha': notes_to_yaml_section(rh_notes, tpb, 'MD'),
            'mano_izquierda': notes_to_yaml_section(lh_notes, tpb, 'MI'),
        }
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    print(f"  YAML exportado: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(report_data: dict, tracks: Dict[str, List[dict]],
                    tpb: int, midi_path: str, output_midi: str) -> str:
    """Genera el informe de texto."""
    lines = []
    W = 70
    lines.append("═" * W)
    lines.append("  PIANO REDUCER — INFORME DE REDUCCIÓN")
    lines.append("═" * W)
    lines.append(f"  Fuente        : {midi_path}")
    lines.append(f"  Salida        : {output_midi}")
    lines.append(f"  Estrategia    : {report_data['strategy']}")
    lines.append("")

    lines.append("  PISTAS DE ENTRADA")
    lines.append("  " + "─" * (W-2))
    for name in report_data['tracks_in']:
        notes = tracks.get(name, [])
        if notes:
            pitches = [n['pitch'] for n in notes]
            role = analyze_track_role(notes, tracks)
            lines.append(f"    {name:<26} {len(notes):>5} notas  "
                         f"rango [{midi_to_name(min(pitches))}–{midi_to_name(max(pitches))}]  "
                         f"rol={role}")
        else:
            lines.append(f"    {name:<26} (vacía)")
    lines.append("")

    lines.append("  ASIGNACIÓN DE MANOS")
    lines.append("  " + "─" * (W-2))
    lines.append(f"    MD: {', '.join(report_data['rh_tracks']) or '—'}")
    lines.append(f"    MI: {', '.join(report_data['lh_tracks']) or '—'}")
    lines.append("")

    lines.append("  PASOS DE PROCESAMIENTO")
    lines.append("  " + "─" * (W-2))
    for step in report_data['steps']:
        lines.append(f"    • {step}")
    lines.append("")

    lines.append("  ESTADÍSTICAS")
    lines.append("  " + "─" * (W-2))
    s = report_data['stats']
    lines.append(f"    Notas brutas       : MD={s['rh_notes_raw']}  MI={s['lh_notes_raw']}")
    lines.append(f"    Notas finales      : MD={s['rh_notes_final']}  MI={s['lh_notes_final']}")
    lines.append(f"    Ratio reducción    : {s['reduction_ratio']:.1%}")
    lines.append(f"    Cruces resueltos   : {s['crossings_fixed']}")
    lines.append(f"    Duplicados elim.   : {s['duplicates_removed']}")
    lines.append("")

    lines.append("  NOTA PARA FL STUDIO")
    lines.append("  " + "─" * (W-2))
    lines.append("    Canal 0 (MIDI ch 1) → Piano MD — asigna piano sample")
    lines.append("    Canal 1 (MIDI ch 2) → Piano MI — mismo instrumento,")
    lines.append("    o usa dos instancias para control de velocidad independiente.")
    lines.append("")
    lines.append("═" * W)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS RÁPIDO SIN MIDI (--report-only)
# ══════════════════════════════════════════════════════════════════════════════

def report_only_analysis(tracks: Dict[str, List[dict]], tpb: int, args) -> None:
    """Muestra análisis de pistas y asignación propuesta sin generar MIDI."""
    print("\n  ANÁLISIS DE PISTAS (sin generar MIDI)")
    print("  " + "─" * 60)

    is_mono = detect_single_track_situation(tracks)

    if is_mono:
        all_notes = [n for notes in tracks.values() for n in notes]
        if all_notes:
            pitches = [n['pitch'] for n in all_notes]
            print(f"  Pista única / MIDI monolítico detectado")
            print(f"  Total notas : {len(all_notes)}")
            print(f"  Rango       : {midi_to_name(min(pitches))} – {midi_to_name(max(pitches))}")
            print(f"  Split en    : {midi_to_name(args.split_note)}")
            rh, lh = split_single_track(all_notes, args.split_note, tpb, verbose=True)
            print(f"  → MD estimado: {len(rh)} notas")
            print(f"  → MI estimado: {len(lh)} notas")
        return

    strategy_fn = STRATEGIES.get(args.strategy, assign_smart)
    rh_track_names, lh_track_names = strategy_fn(tracks, args)

    for name, notes in tracks.items():
        if not notes:
            continue
        pitches = [n['pitch'] for n in notes]
        role    = analyze_track_role(notes, tracks)
        hand    = "MD" if name in rh_track_names else "MI"
        print(f"  {hand}  {name:<26} {len(notes):>5} notas  "
              f"[{midi_to_name(min(pitches))}–{midi_to_name(max(pitches))}]  "
              f"rol={role}")

    print(f"\n  MD ← {rh_track_names}")
    print(f"  MI ← {lh_track_names}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Piano Reducer — Reducción pianística de obras orquestales MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('midi', help='MIDI de entrada (multitracks)')
    parser.add_argument('--strategy',
                        choices=['smart', 'register', 'layer', 'priority', 'melody+bass'],
                        default='smart',
                        help='Estrategia de reducción (default: smart)')
    parser.add_argument('--split',       default='C4',
                        help='Nota de split para --strategy register (default: C4)')
    parser.add_argument('--rh-tracks',   default='',
                        help='Pistas para MD (coma-separados, para --strategy priority)')
    parser.add_argument('--lh-tracks',   default='',
                        help='Pistas para MI (coma-separados, para --strategy priority)')
    parser.add_argument('--max-polyphony', type=int, default=7,
                        help='Límite de voces simultáneas total (default: 7)')
    parser.add_argument('--max-voices-rh', type=int, default=MAX_VOICES_RH,
                        help=f'Voces máximas en MD (default: {MAX_VOICES_RH})')
    parser.add_argument('--max-voices-lh', type=int, default=MAX_VOICES_LH,
                        help=f'Voces máximas en MI (default: {MAX_VOICES_LH})')
    parser.add_argument('--no-alberti',  action='store_true',
                        help='No aplicar bajo de Alberti en MI')
    parser.add_argument('--no-compress', action='store_true',
                        help='No comprimir acordes de >4 notas')
    parser.add_argument('--no-octave',   action='store_true',
                        help='No corregir notas fuera de rango idiomático')
    parser.add_argument('--no-passing',  action='store_true',
                        help='No añadir notas de paso inferidas')
    parser.add_argument('--from-yaml',   default=None,
                        help='Partitura YAML para metadatos adicionales')
    parser.add_argument('--export-yaml', action='store_true',
                        help='Exportar reducción en formato YAML')
    parser.add_argument('--combine-output', action='store_true',
                        help='Generar MIDI combinado (original + reducción)')
    parser.add_argument('--report-only', action='store_true',
                        help='Solo análisis, sin generar MIDI')
    parser.add_argument('--output',      default=None,
                        help='Nombre base de salida (default: <stem>_piano)')
    parser.add_argument('--verbose',     action='store_true',
                        help='Decisiones detalladas')
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Resolver nota de split
    try:
        args.split_note = name_to_midi(args.split)
    except Exception:
        args.split_note = PIANO_SPLIT_DEFAULT
        print(f"  ⚠ No se pudo parsear --split '{args.split}'. Usando C4.")

    print("═" * 65)
    print("  PIANO REDUCER  v1.0")
    print("═" * 65)
    print(f"  Entrada  : {args.midi}")
    print(f"  Estrategia: {args.strategy}")

    # ── Cargar MIDI ─────────────────────────────────────────────────────────
    if not Path(args.midi).exists():
        print(f"  ERROR: Fichero no encontrado: {args.midi}")
        sys.exit(1)

    mid, tpb, tempo = load_midi(args.midi)
    global TICKS
    TICKS = tpb
    bpm   = round(60_000_000 / tempo)
    print(f"  TPB={tpb}  Tempo={bpm} BPM")

    tracks = extract_tracks(mid, tpb)
    if not tracks:
        print("  ERROR: No se encontraron pistas con notas.")
        sys.exit(1)

    print(f"  Pistas detectadas: {len(tracks)}")
    if args.verbose:
        for name, notes in tracks.items():
            print(f"    [{name}]: {len(notes)} notas")

    # ── Solo análisis ────────────────────────────────────────────────────────
    if args.report_only:
        report_only_analysis(tracks, tpb, args)
        sys.exit(0)

    # ── Base name de salida ──────────────────────────────────────────────────
    stem = Path(args.midi).stem
    base = args.output or (stem + '_piano')
    if base.endswith('.mid'):
        base = base[:-4]

    output_midi    = base + '.mid'
    output_report  = base + '_report.txt'
    output_yaml    = base + '.yaml'
    output_combined = base + '_combined.mid'

    # ── Pipeline ─────────────────────────────────────────────────────────────
    print("\n  Reduciendo...")
    rh_notes, lh_notes, report_data = reduce(tracks, tpb, args)

    # ── Escribir MIDI ────────────────────────────────────────────────────────
    print(f"\n  Escribiendo: {output_midi}")
    build_piano_midi(rh_notes, lh_notes, tpb, tempo, output_midi)

    # ── Combinado ────────────────────────────────────────────────────────────
    if args.combine_output:
        print(f"  Escribiendo combinado: {output_combined}")
        build_combined_midi(rh_notes, lh_notes, mid, tpb, tempo, output_combined)

    # ── YAML ─────────────────────────────────────────────────────────────────
    if args.export_yaml:
        export_yaml(rh_notes, lh_notes, tpb, output_yaml, args.from_yaml)

    # ── Informe ───────────────────────────────────────────────────────────────
    report_text = generate_report(report_data, tracks, tpb, args.midi, output_midi)
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  Informe: {output_report}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESUMEN")
    print("═" * 65)
    s = report_data['stats']
    print(f"  MIDI reducción   : {output_midi}")
    print(f"  Informe          : {output_report}")
    print(f"  Notas MD         : {s['rh_notes_final']} (de {s['rh_notes_raw']} originales)")
    print(f"  Notas MI         : {s['lh_notes_final']} (de {s['lh_notes_raw']} originales)")
    print(f"  Ratio reducción  : {s['reduction_ratio']:.1%}")
    print(f"  Estrategia       : {args.strategy}")
    if args.combine_output:
        print(f"  MIDI combinado   : {output_combined}")
    if args.export_yaml:
        print(f"  YAML             : {output_yaml}")
    print("═" * 65)
    print(f"\n  Importa {output_midi} en FL Studio.")
    print("  Canal 1 = Piano MD, Canal 2 = Piano MI.")


if __name__ == '__main__':
    main()
