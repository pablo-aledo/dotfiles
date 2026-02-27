"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           ORCHESTRATOR  v1.0                                                 ║
║   Orquestación automática + percusión para mocks orquestales                 ║
║                                                                              ║
║  Lee el MIDI de stitcher.py + los fingerprints JSON y genera un MIDI        ║
║  multitracks listo para importar en FL Studio con:                           ║
║                                                                              ║
║  [A] Distribución de material a instrumentos reales por sección             ║
║  [B] Keyswitches automáticos según duración y contexto de cada nota         ║
║  [C] CC1 (dynamics xfade) interpolado desde la tension_curve del FP         ║
║  [D] CC11 (expression) con inflexiones de frase automáticas                 ║
║  [E] Percusión orquestal completa: timbal, bombo, platos, tam-tam           ║
║  [F] Registro idiomático: transpone líneas fuera de rango al rango real     ║
║  [G] Informe de sesión: mapa de FL Studio con canales, libraries, CCs       ║
║                                                                              ║
║  PLANTILLA ORQUESTAL GENERADA:                                               ║
║    Cuerdas   — Violin I, Violin II, Viola, Cello, Contrabajo                ║
║    Maderas   — Flauta, Oboe, Clarinete, Fagot                               ║
║    Metales   — Trompa, Trompeta, Trombón                                    ║
║    Percusión — Timbal, Bombo, Platos, Tam-tam, Caja                        ║
║                                                                              ║
║  USO:                                                                        ║
║    python orchestrator.py obra_final.mid [fingerprints...] [opciones]       ║
║                                                                              ║
║    # Con fingerprints explícitos:                                            ║
║    python orchestrator.py obra_final.mid                                \   ║
║        secA.fingerprint.json secB.fingerprint.json secC.fingerprint.json \  ║
║        --template chamber                                                    ║
║                                                                              ║
║    # Auto-detecta fingerprints en el mismo directorio:                      ║
║    python orchestrator.py obra_final.mid --auto-fingerprints                ║
║                                                                              ║
║    # Con plantilla orquestal y library target:                              ║
║    python orchestrator.py obra_final.mid *.fingerprint.json             \   ║
║        --template full --library nucleus                                     ║
║                                                                              ║
║    # Solo generar informe sin MIDI (para planificar):                       ║
║    python orchestrator.py obra_final.mid --report-only                      ║
║                                                                              ║
║    # Personalizar keyswitches para tu patch específico:                     ║
║    python orchestrator.py obra_final.mid --ks-config my_ks.json             ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --template     chamber | full | strings-only   (default: chamber)        ║
║    --library      nucleus | metropolis | generic  (default: nucleus)        ║
║    --ks-config    JSON con keyswitches personalizados (ver --dump-ks)       ║
║    --dump-ks      Exportar tabla de KS defaults a JSON y salir              ║
║    --auto-fp      Buscar .fingerprint.json en el mismo dir que el MIDI      ║
║    --report-only  Solo generar informe .txt, sin producir MIDI              ║
║    --output       Nombre base del MIDI de salida (default: orquestado)      ║
║    --no-perc      No generar percusión orquestal                            ║
║    --no-ks        No insertar keyswitches                                   ║
║    --no-cc        No insertar CC1/CC11                                      ║
║    --tempo-humanize F  Microvariación de tempo 0.0-1.0 (default: 0.15)     ║
║    --verbose      Informe detallado de decisiones                           ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install mido numpy                                                    ║
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


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MIDI
# ══════════════════════════════════════════════════════════════════════════════

TICKS = 480  # ticks per beat

# Nota MIDI → nombre
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def midi_to_name(n):
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"

def name_to_midi(name):
    """'C4' → 60, 'A#3' → 58, etc."""
    note_map = {n: i for i, n in enumerate(NOTE_NAMES)}
    name = name.strip().upper()
    if len(name) >= 2 and name[1] == '#':
        pc = note_map.get(name[:2], 0)
        octave = int(name[2:])
    else:
        pc = note_map.get(name[0], 0)
        octave = int(name[1:])
    return (octave + 1) * 12 + pc


# ══════════════════════════════════════════════════════════════════════════════
#  RANGOS IDIOMÁTICOS POR INSTRUMENTO
#  (MIDI note numbers: 60=C4, 69=A4)
# ══════════════════════════════════════════════════════════════════════════════

INSTRUMENT_RANGES = {
    # Cuerdas
    'violin1':    (55, 96),   # G3 – C7
    'violin2':    (55, 91),   # G3 – G6
    'viola':      (48, 84),   # C3 – C6
    'cello':      (36, 76),   # C2 – E5
    'contrabass': (28, 60),   # E1 – C4 (suena 8vb)
    # Maderas
    'flute':      (60, 96),   # C4 – C7
    'oboe':       (58, 91),   # Bb3 – G6
    'clarinet':   (50, 94),   # D3 – Bb6 (sonido real, Bb clarinete)
    'bassoon':    (34, 75),   # Bb1 – Eb5
    # Metales
    'horn':       (34, 77),   # Bb1 – F5
    'trumpet':    (52, 82),   # E3 – Bb5
    'trombone':   (34, 72),   # Bb1 – C5
    'tuba':       (28, 58),   # E1 – Bb3
    # Percusión de timbal (afinada)
    'timpani':    (41, 65),   # F2 – F4 (par estándar)
}

# Registro más cómodo (zona de mayor expresividad)
INSTRUMENT_SWEET_SPOT = {
    'violin1':    (62, 88),
    'violin2':    (60, 84),
    'viola':      (55, 79),
    'cello':      (43, 72),
    'contrabass': (33, 52),
    'flute':      (65, 90),
    'oboe':       (62, 86),
    'clarinet':   (55, 86),
    'bassoon':    (38, 67),
    'horn':       (40, 70),
    'trumpet':    (56, 77),
    'trombone':   (40, 67),
}


# ══════════════════════════════════════════════════════════════════════════════
#  PLANTILLAS ORQUESTALES
#  Cada plantilla define qué pistas del pipeline van a qué instrumentos
# ══════════════════════════════════════════════════════════════════════════════

# Rol de cada instrumento en la plantilla
# source_track: qué pista del MIDI del pipeline usa como material base
# doubles: lista de instrumentos que doblan (se crean como copias transpuestas/filtradas)
# role: melody | counterpoint | accompaniment | bass | pad

TEMPLATES = {
    'chamber': {
        # Orquesta de cámara: cuerdas + maderas básicas + 2 trompas
        'instruments': [
            {'name': 'violin1',    'source': 'Melody',        'role': 'melody',        'channel': 0},
            {'name': 'violin2',    'source': 'Counterpoint',  'role': 'counterpoint',  'channel': 1},
            {'name': 'viola',      'source': 'Accompaniment', 'role': 'accompaniment', 'channel': 2},
            {'name': 'cello',      'source': 'Bass',          'role': 'bass',          'channel': 3},
            {'name': 'contrabass', 'source': 'Bass',          'role': 'bass',          'channel': 4, 'octave_shift': -1},
            {'name': 'oboe',       'source': 'Melody',        'role': 'melody_double', 'channel': 5, 'section_filter': ['A', 'E']},
            {'name': 'clarinet',   'source': 'Counterpoint',  'role': 'counterpoint',  'channel': 6, 'section_filter': ['B', 'C']},
            {'name': 'bassoon',    'source': 'Bass',          'role': 'bass_double',   'channel': 7, 'section_filter': ['B', 'C', 'D']},
            {'name': 'horn',       'source': 'Accompaniment', 'role': 'pad',           'channel': 8, 'section_filter': ['C', 'D']},
        ],
        'perc_channel': 9,
    },
    'full': {
        # Orquesta completa
        'instruments': [
            {'name': 'violin1',    'source': 'Melody',        'role': 'melody',        'channel': 0},
            {'name': 'violin2',    'source': 'Counterpoint',  'role': 'counterpoint',  'channel': 1},
            {'name': 'viola',      'source': 'Accompaniment', 'role': 'accompaniment', 'channel': 2},
            {'name': 'cello',      'source': 'Bass',          'role': 'bass',          'channel': 3},
            {'name': 'contrabass', 'source': 'Bass',          'role': 'bass',          'channel': 4, 'octave_shift': -1},
            {'name': 'flute',      'source': 'Melody',        'role': 'melody_double', 'channel': 5, 'section_filter': ['A', 'E']},
            {'name': 'oboe',       'source': 'Melody',        'role': 'melody_double', 'channel': 6, 'section_filter': ['A', 'B']},
            {'name': 'clarinet',   'source': 'Counterpoint',  'role': 'counterpoint',  'channel': 7},
            {'name': 'bassoon',    'source': 'Bass',          'role': 'bass_double',   'channel': 8},
            {'name': 'horn',       'source': 'Accompaniment', 'role': 'pad',           'channel': 10},
            {'name': 'trumpet',    'source': 'Melody',        'role': 'melody_double', 'channel': 11, 'section_filter': ['C']},
            {'name': 'trombone',   'source': 'Accompaniment', 'role': 'pad',           'channel': 12, 'section_filter': ['C', 'D']},
        ],
        'perc_channel': 9,
    },
    'strings_only': {
        'instruments': [
            {'name': 'violin1',    'source': 'Melody',        'role': 'melody',        'channel': 0},
            {'name': 'violin2',    'source': 'Counterpoint',  'role': 'counterpoint',  'channel': 1},
            {'name': 'viola',      'source': 'Accompaniment', 'role': 'accompaniment', 'channel': 2},
            {'name': 'cello',      'source': 'Bass',          'role': 'bass',          'channel': 3},
            {'name': 'contrabass', 'source': 'Bass',          'role': 'bass',          'channel': 4, 'octave_shift': -1},
        ],
        'perc_channel': 9,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  KEYSWITCHES
#  IMPORTANTE: Nucleus y Metropolis Ark usan KS REASIGNABLES en SINE/Capsule.
#  Los valores aquí son los defaults más comunes reportados por usuarios.
#  Usa --dump-ks para exportar y --ks-config para personalizar según tu patch.
#
#  Convención: notas negativas = sin KS para esa articulación en esa library
# ══════════════════════════════════════════════════════════════════════════════

# Duración en beats para clasificar articulaciones
ART_THRESHOLDS = {
    'legato':   2.0,   # notas de 2+ beats → legato
    'sustain':  0.75,  # notas de 0.75-2 beats → sustain
    'portato':  0.4,   # notas de 0.4-0.75 → portato / détaché
    'spiccato': 0.0,   # notas < 0.4 → spiccato / staccato
}

# Tabla de keyswitches por library
# Formato: {instrumento: {articulacion: nota_midi}}
# Usa -1 para "no disponible" (se cae a la articulación más cercana)
KEYSWITCH_TABLES = {

    'nucleus': {
        # Audio Imperia Nucleus — KS defaults reportados (ajustables en Kontakt)
        # Los patches "multi" de Nucleus asignan KS desde A0 (21) hacia arriba
        # El orden típico es: Legato, Sustain, Tremolo, Spiccato, Pizzicato
        '_default_start': 21,  # A0
        'strings': {
            'legato':   21,   # A0
            'sustain':  22,   # A#0
            'tremolo':  23,   # B0
            'spiccato': 24,   # C1
            'pizzicato':25,   # C#1
            'portato':  22,   # = sustain (no hay portato separado)
        },
        'winds': {
            'legato':   21,   # A0
            'sustain':  22,   # A#0
            'staccato': 23,   # B0
            'trill_ht': 24,   # C1
            'trill_wt': 25,   # C#1
            'portato':  22,   # = sustain
        },
        'brass': {
            'legato':   21,   # A0
            'sustain':  22,   # A#0
            'staccato': 23,   # B0
            'portato':  22,
        },
    },

    'metropolis': {
        # Orchestral Tools Metropolis Ark 1 en SINE Player
        # KS se asignan automáticamente desde el KS Area Start configurado
        # Default en SINE: KS High = C8 (arriba del rango), KS Low = A0 (abajo)
        # Los instrumentos de MA1 son grabados fff — usar con moderación en pp
        '_default_start': 21,  # A0 (Low range por defecto en SINE)
        'strings': {
            # Finckenstein High Strings / Wolfenstein Low Strings
            'legato':    21,  # A0  — 8va legato (característica de MA1)
            'sustain':   22,  # A#0 — unison sustain
            'tremolo':   23,  # B0
            'spiccato':  24,  # C1
            'col_legno': 25,  # C#1 — específico de MA1 (alta tensión)
            'portato':   22,  # = sustain
            'pizzicato': -1,  # MA1 no tiene pizzicato en high strings
        },
        'brass': {
            # Schwarzdorn / Kaskelturm horns, Bismarckhütte trombones
            'legato':     21,  # A0
            'sustain':    22,  # A#0
            'marcato_l':  23,  # B0  — marcato largo
            'marcato_s':  24,  # C1  — marcato corto
            'staccato':   25,  # C#1
            'swell':      26,  # D1  — crescendo característico MA1
            'portato':    22,
        },
        'winds': {
            'sustain':   21,
            'staccato':  22,
            'portato':   21,
            'legato':    -1,  # MA1 no incluye maderas
        },
        'choir': {
            # Viktoria / Aarauer choir
            'legato':    21,
            'sustain':   22,
            'marcato':   23,
            'staccato':  24,
        },
    },

    'generic': {
        # Sin library específica — no inserta KS, solo CC1/CC11
        '_default_start': 0,
        'strings': {
            'legato': -1, 'sustain': -1, 'spiccato': -1,
            'tremolo': -1, 'pizzicato': -1, 'portato': -1,
        },
        'winds':   {'legato': -1, 'sustain': -1, 'staccato': -1, 'portato': -1},
        'brass':   {'legato': -1, 'sustain': -1, 'staccato': -1, 'portato': -1},
    },
}

# Clasificación de instrumento → familia
INSTRUMENT_FAMILY = {
    'violin1': 'strings', 'violin2': 'strings', 'viola': 'strings',
    'cello': 'strings', 'contrabass': 'strings',
    'flute': 'winds', 'oboe': 'winds', 'clarinet': 'winds', 'bassoon': 'winds',
    'horn': 'brass', 'trumpet': 'brass', 'trombone': 'brass', 'tuba': 'brass',
}


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRAMS GM POR INSTRUMENTO (para referencia / GM fallback)
# ══════════════════════════════════════════════════════════════════════════════

GM_PROGRAMS = {
    'violin1': 40, 'violin2': 40, 'viola': 41, 'cello': 42, 'contrabass': 43,
    'flute': 73, 'oboe': 68, 'clarinet': 71, 'bassoon': 70,
    'horn': 60, 'trumpet': 56, 'trombone': 57, 'tuba': 58,
    'timpani': 47,
}

# Nombres legibles para el informe
INSTRUMENT_NAMES = {
    'violin1': 'Violin I', 'violin2': 'Violin II', 'viola': 'Viola',
    'cello': 'Cello', 'contrabass': 'Contrabajo',
    'flute': 'Flauta', 'oboe': 'Oboe', 'clarinet': 'Clarinete', 'bassoon': 'Fagot',
    'horn': 'Trompa', 'trumpet': 'Trompeta', 'trombone': 'Trombón', 'tuba': 'Tuba',
    'timpani': 'Timbal', 'bass_drum': 'Bombo', 'crash': 'Platos', 'tamtam': 'Tam-tam',
    'snare': 'Caja', 'sus_cym': 'Plato susp.',
}

# Pistas del MIDI de stitcher
PIPELINE_TRACKS = ['Melody', 'Counterpoint', 'Accompaniment', 'Bass', 'Percussion']


# ══════════════════════════════════════════════════════════════════════════════
#  PERCUSIÓN ORQUESTAL
#  Notas GM canal 9: Kick=36, Snare=38, HHclosed=42, HHopen=46, Crash=49
#  Notas de percusión de orquesta que mapearemos:
#    Timbal    → nota afinada según tonalidad (no canal 9, canal propio)
#    Bombo     → 36 (canal 9)
#    Caja      → 38 (canal 9)
#    Plato sus → 51 (canal 9, ride bell = plato de suspensión)
#    Tam-tam   → 54 (canal 9, tambourine = más grave en muchas libraries)
#    Crash     → 49 (canal 9)
# ══════════════════════════════════════════════════════════════════════════════

PERC_GM = {
    'kick':      36,
    'snare':     38,
    'hihat':     42,
    'crash':     49,
    'sus_cym':   51,
    'tamtam':    54,
    'low_floor': 41,  # floor tom como bombo pesado
}


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA DEL MIDI DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def load_midi_tracks(midi_path):
    """
    Carga el MIDI de stitcher y devuelve un dict:
    {nombre_pista: [(abs_tick, pitch, duration_ticks, velocity), ...]}
    También devuelve el tempo en microsegundos y ticks_per_beat.
    """
    mid = MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    tempo = 500000  # default 120 BPM

    tracks_raw = {}
    section_markers = []  # [(abs_tick, label)]

    for track_idx, track in enumerate(mid.tracks):
        name = track.name.strip() if track.name else f'track_{track_idx}'
        abs_tick = 0
        note_on_events = {}  # pitch → abs_tick_on, velocity
        notes = []

        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'marker':
                # Recolectar marcadores solo del primer track (tempo track en MIDI tipo 1)
                # para evitar duplicados si cada pista repite los mismos marcadores.
                if track_idx == 0:
                    section_markers.append((abs_tick, msg.text))
            elif msg.type == 'note_on' and msg.velocity > 0:
                note_on_events[msg.note] = (abs_tick, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in note_on_events:
                    t_on, vel = note_on_events.pop(msg.note)
                    dur = abs_tick - t_on
                    if dur > 0:
                        notes.append((t_on, msg.note, dur, vel))

        if notes:
            tracks_raw[name] = sorted(notes, key=lambda x: x[0])

    # Deduplicar marcadores: si el mismo tick aparece varias veces (porque cada pista
    # del MIDI repite los marcadores), quedarse solo con la primera ocurrencia por tick.
    seen_ticks = set()
    deduped_markers = []
    for tick, label in sorted(section_markers):
        if tick not in seen_ticks:
            seen_ticks.add(tick)
            deduped_markers.append((tick, label))
    section_markers = deduped_markers

    return tracks_raw, tempo, tpb, section_markers


def ticks_to_beats(ticks, tpb):
    return ticks / tpb

def beats_to_ticks(beats, tpb):
    return int(beats * tpb)


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA Y PROCESADO DE FINGERPRINTS
# ══════════════════════════════════════════════════════════════════════════════

def load_fingerprints(fp_paths):
    """Carga una lista de fingerprint JSONs y los indexa por nombre de sección."""
    fps = []
    for p in fp_paths:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                fp = json.load(f)
            fps.append(fp)
        except Exception as e:
            print(f"  ⚠ No se pudo cargar {p}: {e}")
    return fps


def build_section_map(section_markers, fps, tpb, tempo):
    """
    Crea un mapa de secciones con su rango de ticks y fingerprint asociado.
    Devuelve lista de:
    {label, start_tick, end_tick, fp (o None), key_tonic, key_mode,
     tension_curve, arc, tempo_bpm, bars}
    """
    if not section_markers:
        # Sin marcadores: toda la obra es una sola sección
        return [{
            'label': 'A', 'start_tick': 0, 'end_tick': None,
            'fp': fps[0] if fps else None,
            'key_tonic': fps[0]['meta']['key_tonic'] if fps else 'C',
            'key_mode':  fps[0]['meta']['key_mode']  if fps else 'major',
            'tension_curve': fps[0]['tension_curve'] if fps else {'mean': 0.5},
            'arc':  fps[0]['meta'].get('emotional_arc', 'neutral') if fps else 'neutral',
            'tempo_bpm': fps[0]['meta']['tempo_bpm'] if fps else 120.0,
            'bars': fps[0]['meta']['n_bars'] if fps else 16,
            'harmony_complexity': fps[0]['meta'].get('harmony_complexity', 0.5) if fps else 0.5,
        }]

    # Detectar si todos los marcadores tienen el mismo label (pipeline repetitivo)
    # En ese caso, reasignar labels únicos secuenciales: A, B, C, ...
    SECTION_LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    raw_labels = [m[1].strip('[]') for m in section_markers]
    all_same = len(set(raw_labels)) == 1

    sections = []
    # Los marcadores del pipeline tienen formato [A], [B], etc.
    for i, (tick, label) in enumerate(section_markers):
        if all_same:
            # Reasignar label único: A, B, C, ... Z, AA, AB, ...
            if i < 26:
                label_clean = SECTION_LETTERS[i]
            else:
                label_clean = SECTION_LETTERS[i // 26 - 1] + SECTION_LETTERS[i % 26]
        else:
            label_clean = label.strip('[]')

        end_tick = section_markers[i + 1][0] if i + 1 < len(section_markers) else None
        # Buscar fingerprint que coincida con esta sección (por nombre o posición)
        fp = fps[i] if i < len(fps) else (fps[-1] if fps else None)
        sections.append({
            'label': label_clean,
            'start_tick': tick,
            'end_tick': end_tick,
            'fp': fp,
            'key_tonic': fp['meta']['key_tonic'] if fp else 'C',
            'key_mode':  fp['meta']['key_mode']  if fp else 'major',
            'tension_curve': fp['tension_curve'] if fp else {'mean': 0.5},
            'arc':  fp['meta'].get('emotional_arc', 'neutral') if fp else 'neutral',
            'tempo_bpm': fp['meta']['tempo_bpm'] if fp else 120.0,
            'bars': fp['meta']['n_bars'] if fp else 16,
            'harmony_complexity': fp['meta'].get('harmony_complexity', 0.5) if fp else 0.5,
        })

    # Asegurarse de que la primera sección empiece en tick 0.
    if sections and sections[0]['start_tick'] > 0:
        sections[0] = dict(sections[0], start_tick=0)

    if all_same:
        print(f"  ⚠ Todos los marcadores tenían el mismo label. "
              f"Reasignados como: {[s['label'] for s in sections]}")

    return sections


def tension_at_tick(tick, section, total_ticks_section, tpb):
    """Devuelve tensión 0-1 en un tick dado dentro de una sección."""
    fp = section.get('fp')
    if not fp:
        return section['tension_curve'].get('mean', 0.5)

    # Si hay curva completa en el fingerprint, interpolar
    tc_full = fp.get('tension_curve_full')
    if tc_full and len(tc_full) > 1:
        if total_ticks_section <= 0:
            return tc_full[0]
        pos = tick / total_ticks_section
        idx = pos * (len(tc_full) - 1)
        i0 = int(idx)
        i1 = min(i0 + 1, len(tc_full) - 1)
        alpha = idx - i0
        return tc_full[i0] * (1 - alpha) + tc_full[i1] * alpha

    # Fallback: interpolar con los stats del fingerprint
    t_entry = fp['tension_curve']['entry']
    t_exit  = fp['tension_curve']['exit']
    t_peak  = fp['tension_curve']['peak']
    peak_pos = fp['tension_curve']['peak_bar'] / max(fp['meta']['n_bars'], 1)
    pos = tick / max(total_ticks_section, 1)

    if pos < peak_pos:
        alpha = pos / max(peak_pos, 0.01)
        return t_entry + (t_peak - t_entry) * alpha
    else:
        alpha = (pos - peak_pos) / max(1.0 - peak_pos, 0.01)
        return t_peak + (t_exit - t_peak) * alpha


# ══════════════════════════════════════════════════════════════════════════════
#  CLASIFICACIÓN DE ARTICULACIÓN POR NOTA
# ══════════════════════════════════════════════════════════════════════════════

def classify_articulation(note_tuple, next_note, prev_note, tension, family, role):
    """
    Clasifica la articulación apropiada para una nota.
    note_tuple: (abs_tick, pitch, duration_ticks, velocity)
    Devuelve: 'legato' | 'sustain' | 'portato' | 'spiccato' | 'pizzicato' | 'tremolo'
    """
    t_on, pitch, dur_ticks, vel = note_tuple
    dur_beats = ticks_to_beats(dur_ticks, TICKS)

    # Contexto de tensión
    high_tension = tension > 0.65
    low_tension  = tension < 0.3

    # Brecha con la nota anterior (si hay)
    gap_beats = 0.0
    if prev_note:
        prev_end = prev_note[0] + prev_note[2]
        gap_beats = ticks_to_beats(t_on - prev_end, TICKS)

    # Brecha con la nota siguiente (si hay)
    next_gap = 0.0
    if next_note:
        my_end = t_on + dur_ticks
        next_gap = ticks_to_beats(next_note[0] - my_end, TICKS)

    # Reglas principales
    if family == 'strings':
        if role in ('melody', 'melody_double') and dur_beats >= ART_THRESHOLDS['legato'] and gap_beats < 0.25:
            return 'legato'
        elif role in ('melody', 'melody_double') and dur_beats >= ART_THRESHOLDS['sustain']:
            return 'sustain'
        elif high_tension and dur_beats < ART_THRESHOLDS['portato']:
            # Alta tensión + notas cortas → spiccato agresivo
            return 'spiccato'
        elif role == 'accompaniment' and low_tension and dur_beats >= 0.4:
            # Acompañamiento en baja tensión: considerar pizzicato
            return 'pizzicato'
        elif role == 'accompaniment' and high_tension and dur_beats >= 1.0:
            # Acompañamiento en alta tensión: tremolo
            return 'tremolo'
        elif dur_beats >= ART_THRESHOLDS['portato']:
            return 'portato'
        elif dur_beats >= ART_THRESHOLDS['spiccato']:
            return 'spiccato'
        else:
            return 'sustain'

    elif family == 'winds':
        if dur_beats >= ART_THRESHOLDS['legato'] and gap_beats < 0.2:
            return 'legato'
        elif dur_beats >= ART_THRESHOLDS['sustain']:
            return 'sustain'
        elif dur_beats >= ART_THRESHOLDS['portato']:
            return 'portato'
        else:
            return 'staccato'

    elif family == 'brass':
        if dur_beats >= ART_THRESHOLDS['legato']:
            if high_tension:
                return 'marcato_l' if dur_beats >= 2.0 else 'marcato_s'
            return 'legato'
        elif dur_beats >= ART_THRESHOLDS['sustain']:
            return 'sustain'
        elif dur_beats >= ART_THRESHOLDS['portato']:
            return 'portato'
        else:
            return 'staccato'

    return 'sustain'


# ══════════════════════════════════════════════════════════════════════════════
#  PROCESADO DE NOTAS POR INSTRUMENTO
# ══════════════════════════════════════════════════════════════════════════════

def fit_to_range(pitch, lo, hi, sweet_lo, sweet_hi):
    """
    Transpone por octavas hasta que la nota cabe en [lo, hi].
    Prioriza el sweet spot. Devuelve (nuevo_pitch, semitones_shifted).
    """
    if lo <= pitch <= hi:
        return pitch, 0

    original = pitch
    # Intentar mover hacia el sweet spot por octavas
    for direction in [1, -1]:
        p = pitch
        for _ in range(4):
            p += 12 * direction
            if lo <= p <= hi:
                if sweet_lo <= p <= sweet_hi:
                    return p, p - original
        p = pitch
        for _ in range(4):
            p -= 12 * direction
            if lo <= p <= hi:
                return p, p - original

    # Clamp duro
    return max(lo, min(hi, pitch)), 0


def apply_octave_shift(notes, shift):
    """Transpone todas las notas en semitones (múltiplos de 12)."""
    return [(t, p + shift, d, v) for t, p, d, v in notes]


def filter_by_section(notes, section, section_filter):
    """Mantiene solo las notas que caen en las secciones del filtro."""
    if not section_filter:
        return notes
    start = section['start_tick']
    end   = section['end_tick'] if section['end_tick'] else float('inf')
    label = section['label']
    if label not in section_filter:
        return []
    return [(t, p, d, v) for t, p, d, v in notes if start <= t < end]


def process_instrument_notes(raw_notes, instr_cfg, sections, ks_table, library,
                              add_ks, add_cc, tempo_humanize, verbose):
    """
    Procesa las notas de una pista para un instrumento específico.
    Devuelve lista de eventos MIDI: (abs_tick, type, data...)
    type puede ser: 'note', 'cc', 'ks' (keyswitch note)
    """
    name     = instr_cfg['name']
    role     = instr_cfg['role']
    octshift = instr_cfg.get('octave_shift', 0)
    sec_filt = instr_cfg.get('section_filter', None)
    family   = INSTRUMENT_FAMILY.get(name, 'strings')

    lo, hi    = INSTRUMENT_RANGES.get(name, (48, 84))
    sw_lo, sw_hi = INSTRUMENT_SWEET_SPOT.get(name, (lo + 5, hi - 5))
    ks_map    = ks_table.get(family, {})

    events = []
    prev_art = None
    prev_note = None

    # Filtrar notas por sección si hay section_filter
    if sec_filt:
        filtered = []
        for section in sections:
            if section['label'] in sec_filt:
                s = section['start_tick']
                e = section['end_tick'] if section['end_tick'] else float('inf')
                filtered += [(t, p, d, v) for t, p, d, v in raw_notes if s <= t < e]
        raw_notes = sorted(filtered, key=lambda x: x[0])

    if not raw_notes:
        return events

    # Aplicar octave_shift base
    if octshift:
        raw_notes = apply_octave_shift(raw_notes, octshift * 12)

    # Construir mapa de sección por tick
    def get_section_for_tick(tick):
        for sec in sections:
            s = sec['start_tick']
            e = sec['end_tick'] if sec['end_tick'] else float('inf')
            if s <= tick < e:
                return sec
        # Si el tick es anterior a la primera sección, devolver la primera
        if tick < sections[0]['start_tick']:
            return sections[0]
        return sections[-1]

    # Variables para CC1 (evitar repetir el mismo valor consecutivo)
    last_cc1 = -1
    last_cc11 = -1
    cc1_resolution = beats_to_ticks(0.25, TICKS)  # CC1 cada corchea

    # Calcular ticks totales de cada sección
    section_total_ticks = {}
    for sec in sections:
        if sec['end_tick'] is not None:
            section_total_ticks[sec['label']] = sec['end_tick'] - sec['start_tick']
        else:
            if raw_notes:
                section_total_ticks[sec['label']] = raw_notes[-1][0] + raw_notes[-1][2]
            else:
                section_total_ticks[sec['label']] = TICKS * 64

    # CC1 al inicio de la pista (valor medio para iniciar la dinámica)
    first_section = get_section_for_tick(raw_notes[0][0])
    init_tension  = first_section['tension_curve'].get('entry', 0.4)
    init_cc1 = int(np.clip(init_tension * 100 + 15, 20, 115))
    if add_cc:
        events.append((raw_notes[0][0], 'cc', 1, init_cc1))
        events.append((raw_notes[0][0], 'cc', 11, 100))

    next_cc1_tick = raw_notes[0][0] + cc1_resolution

    for i, note in enumerate(raw_notes):
        t_on, pitch, dur_ticks, vel = note
        next_n = raw_notes[i + 1] if i + 1 < len(raw_notes) else None
        prev_n = raw_notes[i - 1] if i > 0 else None

        # Transponer al rango idiomático
        pitch, shift = fit_to_range(pitch, lo, hi, sw_lo, sw_hi)
        if shift and verbose:
            print(f"    [{name}] nota transpuesta {shift:+d} semitones → {midi_to_name(pitch)}")

        # Humanize timing (micro-jitter)
        if tempo_humanize > 0:
            jitter_range = int(tempo_humanize * TICKS * 0.04)  # max ~20 ticks
            if jitter_range > 0:
                t_on = max(0, t_on + random.randint(-jitter_range, jitter_range))

        # Obtener sección y tensión
        section = get_section_for_tick(t_on)
        total_sec = section_total_ticks.get(section['label'], TICKS * 64)
        local_tick = t_on - section['start_tick']
        tension = tension_at_tick(local_tick, section, total_sec, TICKS)

        # Clasificar articulación
        art = classify_articulation(note, next_n, prev_n, tension, family, role)

        # Emitir keyswitch si cambia la articulación
        if add_ks and art != prev_art:
            ks_note = ks_map.get(art, ks_map.get('sustain', -1))
            if ks_note >= 0:
                # KS antes de la nota, con 5 ticks de anticipación
                ks_tick = max(0, t_on - 5)
                events.append((ks_tick, 'ks', ks_note, 100))
                if verbose:
                    print(f"    [{name}] KS {midi_to_name(ks_note)} → {art} @ tick {ks_tick}")
            prev_art = art

        # Emitir CC1 si toca
        if add_cc and t_on >= next_cc1_tick:
            cc1_val = int(np.clip(tension * 100 + 15, 15, 120))
            # Suavizar: no saltar más de 15 en un step
            if last_cc1 >= 0:
                cc1_val = int(np.clip(cc1_val, last_cc1 - 15, last_cc1 + 15))
            if cc1_val != last_cc1:
                events.append((t_on, 'cc', 1, cc1_val))
                last_cc1 = cc1_val
            next_cc1_tick = t_on + cc1_resolution

        # CC11 expression: inflexión en notas largas (peak a mitad, decae al final)
        if add_cc:
            dur_beats = ticks_to_beats(dur_ticks, TICKS)
            if dur_beats >= 1.5:
                # Swell: sube a mitad de la nota, baja al 80% hacia el final
                mid_tick = t_on + dur_ticks // 2
                end_tick = t_on + int(dur_ticks * 0.85)
                cc11_peak = min(127, int(vel * 1.1 + tension * 20))
                cc11_end  = max(40, int(vel * 0.8))
                events.append((mid_tick, 'cc', 11, cc11_peak))
                events.append((end_tick, 'cc', 11, cc11_end))

        # Emitir la nota
        # Velocidad: en articulaciones largas la dinámica viene del CC1
        # → normalizamos la vel a un valor medio para que el CC1 controle
        if family in ('strings', 'brass') and art in ('legato', 'sustain', 'tremolo'):
            effective_vel = 80  # CC1 es quien manda en longs
        else:
            # Shorts: la velocity sí importa; escalar con tensión
            effective_vel = int(np.clip(vel * (0.7 + tension * 0.5), 30, 127))

        events.append((t_on, 'note', pitch, effective_vel, dur_ticks))

    return events


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE PERCUSIÓN ORQUESTAL
# ══════════════════════════════════════════════════════════════════════════════

def get_tonic_midi(key_tonic, octave=2):
    """Devuelve la nota MIDI de la tónica en la octava dada."""
    pc_map = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,
              'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
    pc = pc_map.get(key_tonic.replace('b', '#'), 0)  # simplificación
    return (octave + 1) * 12 + pc


def dominant_midi(key_tonic, key_mode, octave=2):
    """Nota de la dominante (quinta sobre la tónica)."""
    tonic = get_tonic_midi(key_tonic, octave)
    # Dominante = tónica + 7 semitones (quinta justa)
    dom = tonic + 7
    # Ajustar al rango del timbal (41-65)
    while dom > 65: dom -= 12
    while dom < 41: dom += 12
    return dom


def generate_orchestral_percussion(sections, tpb, tempo, no_perc=False):
    """
    Genera percusión orquestal completa desde los fingerprints de sección.
    Devuelve dict: {'timpani': events_list, 'perc_gm': events_list}

    timpani events: (abs_tick, pitch, dur_ticks, vel)
    perc_gm events: (abs_tick, gm_note, dur_ticks, vel)
    """
    if no_perc:
        return {'timpani': [], 'perc_gm': []}

    beats_per_bar  = 4  # asumimos 4/4
    ticks_per_bar  = beats_per_bar * tpb
    subdivisions   = tpb // 4  # semicorchea

    timpani_events = []
    gm_events      = []

    for sec_idx, section in enumerate(sections):
        fp    = section.get('fp')
        arc   = section.get('arc', 'neutral')
        key_t = section.get('key_tonic', 'C')
        key_m = section.get('key_mode', 'major')
        start_tick = section['start_tick']
        end_tick   = section.get('end_tick')
        n_bars     = section.get('bars', 16)
        tension_mean  = section['tension_curve'].get('mean', 0.5)
        tension_peak  = section['tension_curve'].get('peak', 0.6)
        tension_entry = section['tension_curve'].get('entry', 0.4)
        tension_exit  = section['tension_curve'].get('exit', 0.4)

        # Afinar timbal en tónica y dominante
        tonic_note = get_tonic_midi(key_t, octave=2)
        dom_note   = dominant_midi(key_t, key_m, octave=2)

        # Asegurar rango timbal
        tonic_note = max(41, min(65, tonic_note))
        dom_note   = max(41, min(65, dom_note))

        if end_tick is None:
            end_tick = start_tick + n_bars * ticks_per_bar

        # ── ESTRATEGIA POR ARCO EMOCIONAL ──────────────────────────────────
        # La percusión cambia completamente según el arco de la sección

        for bar in range(n_bars):
            bar_tick = start_tick + bar * ticks_per_bar

            # Tensión relativa dentro de la sección
            bar_pos    = bar / max(n_bars - 1, 1)
            tension    = tension_entry + (tension_exit - tension_entry) * bar_pos
            if arc == 'arch':
                t_mid = 2 * tension_peak * bar_pos * (1 - bar_pos)
                tension = tension_entry * (1 - bar_pos) + tension_exit * bar_pos
                tension = max(tension, t_mid)
            elif arc == 'crescendo':
                tension = tension_entry + (tension_peak - tension_entry) * bar_pos
            elif arc == 'decrescendo':
                tension = tension_peak - (tension_peak - tension_exit) * bar_pos
            elif arc in ('plateau',):
                tension = tension_mean

            tension = float(np.clip(tension, 0.0, 1.0))
            high = tension > 0.65
            mid  = 0.35 <= tension <= 0.65
            low  = tension < 0.35

            # ── TIMBAL ────────────────────────────────────────────────────
            # Patrón según arco y tensión

            if arc in ('arch', 'crescendo') and high:
                # Clímax: ritmo de cuartas en tónica + dom
                for beat in range(beats_per_bar):
                    t = bar_tick + beat * tpb
                    pitch_t = tonic_note if beat % 2 == 0 else dom_note
                    vel_t   = int(np.clip(85 + tension * 30, 85, 115))
                    timpani_events.append((t, pitch_t, tpb - 5, vel_t))

            elif arc in ('arch', 'crescendo') and mid:
                # Desarrollo: cuartas en tiempo 1 y 3
                for beat in [0, 2]:
                    t = bar_tick + beat * tpb
                    vel_t = int(np.clip(65 + tension * 25, 60, 95))
                    timpani_events.append((t, tonic_note, tpb - 5, vel_t))

            elif arc in ('decrescendo', 'lullaby') and low:
                # Resolución: solo primer tiempo, piano
                if bar % 2 == 0:
                    vel_t = int(np.clip(45 + tension * 20, 40, 70))
                    timpani_events.append((bar_tick, tonic_note, tpb * 2 - 5, vel_t))

            elif arc == 'neutral' and mid:
                # Neutro medio: tiempo 1 de cada dos compases
                if bar % 2 == 0:
                    vel_t = int(70 + tension * 15)
                    timpani_events.append((bar_tick, tonic_note, tpb - 5, vel_t))

            elif arc in ('late_climax',):
                # Clímax tardío: roll creciente hacia el final
                if bar > n_bars * 0.6:
                    roll_density = int((bar - n_bars * 0.6) / (n_bars * 0.4) * 4)
                    for sub in range(roll_density):
                        t = bar_tick + sub * (tpb // 2)
                        vel_t = int(55 + (bar / n_bars) * 60)
                        timpani_events.append((t, tonic_note, tpb // 2 - 3, vel_t))

            # ── GM PERCUSSION (Canal 9) ────────────────────────────────────

            if arc in ('crescendo', 'arch') and high:
                # Clímax: bombo en 1 y 3, platos en 2 y 4
                for beat in range(beats_per_bar):
                    t = bar_tick + beat * tpb
                    if beat % 2 == 0:
                        vel_bd = int(np.clip(90 + tension * 25, 90, 115))
                        gm_events.append((t, PERC_GM['kick'], tpb // 2, vel_bd))
                    else:
                        vel_cr = int(np.clip(75 + tension * 20, 70, 100))
                        gm_events.append((t, PERC_GM['crash'], tpb - 5, vel_cr))

                # Crash específico en downbeat del primer compás de clímax
                if bar == 0 and arc == 'crescendo':
                    gm_events.append((bar_tick, PERC_GM['crash'], tpb * 2, 110))

            elif arc in ('arch',) and mid:
                # Desarrollo medio: bombo en 1, caja en 3
                gm_events.append((bar_tick, PERC_GM['kick'], tpb // 2, 75))
                gm_events.append((bar_tick + 2 * tpb, PERC_GM['snare'], tpb // 2, 65))

            elif arc in ('decrescendo', 'falling', 'lullaby') or (arc == 'neutral' and low):
                # Resolución / reposo: nada o un golpe sutil de plato
                if bar % 4 == 0 and tension > 0.2:
                    gm_events.append((bar_tick, PERC_GM['sus_cym'], tpb * 4, 45))

            # Tam-tam: solo al inicio de secciones de ruptura
            if sec_idx > 0 and bar == 0 and arc in ('neutral',) and tension_entry > 0.5:
                # Primera sección con entrada tensa y arco neutral → posible fractura
                gm_events.append((bar_tick, PERC_GM['tamtam'], tpb * 8, 95))

            # Tam-tam al inicio de sección con alta apertura tras clímax
            if fp and fp['stitching_hints']['openness'] > 0.7 and bar == 0:
                gm_events.append((bar_tick, PERC_GM['tamtam'], tpb * 6, 85))

    return {
        'timpani': sorted(timpani_events, key=lambda x: x[0]),
        'perc_gm': sorted(gm_events, key=lambda x: x[0]),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL MIDI DE SALIDA
# ══════════════════════════════════════════════════════════════════════════════

def events_to_track(events, channel, track_name, program, tempo, tpb):
    """
    Convierte una lista de eventos (abs_tick, type, ...) a MidiTrack.
    Tipos:
      ('note',  pitch, vel, dur_ticks)
      ('cc',    cc_num, value)
      ('ks',    ks_note, vel)  — keyswitch: note_on + note_off inmediato
    """
    if not events:
        return None

    trk = MidiTrack()
    trk.name = track_name
    trk.append(MetaMessage('set_tempo', tempo=int(tempo), time=0))
    trk.append(Message('program_change', channel=channel, program=program, time=0))

    # Expandir eventos a (abs_tick, priority, msg)
    # priority: note_off=0, cc=1, ks=2, note_on=3
    raw = []

    for ev in events:
        abs_tick = ev[0]
        etype    = ev[1]

        if etype == 'note':
            _, _, pitch, vel, dur = ev
            pitch = max(0, min(127, pitch))
            vel   = max(1, min(127, vel))
            raw.append((abs_tick,           3, Message('note_on',  channel=channel, note=pitch, velocity=vel,   time=0)))
            raw.append((abs_tick + dur,     0, Message('note_off', channel=channel, note=pitch, velocity=0,     time=0)))

        elif etype == 'cc':
            _, _, cc_num, val = ev
            val = max(0, min(127, val))
            raw.append((abs_tick, 1, Message('control_change', channel=channel, control=cc_num, value=val, time=0)))

        elif etype == 'ks':
            _, _, ks_note, vel = ev
            ks_note = max(0, min(127, ks_note))
            raw.append((abs_tick,     2, Message('note_on',  channel=channel, note=ks_note, velocity=vel, time=0)))
            raw.append((abs_tick + 2, 0, Message('note_off', channel=channel, note=ks_note, velocity=0,   time=0)))

    # Ordenar: por tick, luego priority (note_off primero, luego cc, luego ks, luego note_on)
    raw.sort(key=lambda x: (x[0], x[1]))

    prev_tick = 0
    for abs_tick, _, msg in raw:
        delta = max(0, abs_tick - prev_tick)
        prev_tick = abs_tick
        trk.append(msg.copy(time=delta))

    trk.append(MetaMessage('end_of_track', time=0))
    return trk


def build_output_midi(instrument_events, timpani_events, gm_perc_events,
                      template, tempo, tpb, library, output_path):
    """
    Ensambla todos los eventos en un MIDI tipo 1 multitracks.
    """
    mid = MidiFile(type=1, ticks_per_beat=tpb)

    instr_cfgs = template['instruments']
    perc_ch    = template.get('perc_channel', 9)

    created_tracks = []

    # Pistas de instrumentos
    for instr_cfg in instr_cfgs:
        name    = instr_cfg['name']
        channel = instr_cfg['channel']
        if channel == 9:
            channel = 10  # saltar canal de percusión GM

        events  = instrument_events.get(name, [])
        if not events:
            continue

        program  = GM_PROGRAMS.get(name, 0)
        disp_name = INSTRUMENT_NAMES.get(name, name)
        trk = events_to_track(events, channel, disp_name, program, tempo, tpb)
        if trk:
            mid.tracks.append(trk)
            created_tracks.append(disp_name)

    # Timbal (canal propio, no el 9)
    if timpani_events:
        timpani_ch = 15  # canal reservado para timbal
        trk = events_to_track(
            [(t, 'note', p, v, d) for t, p, d, v in timpani_events],
            timpani_ch, 'Timbal', GM_PROGRAMS['timpani'], tempo, tpb
        )
        if trk:
            mid.tracks.append(trk)
            created_tracks.append('Timbal')

    # Percusión GM (canal 9)
    if gm_perc_events:
        trk = events_to_track(
            [(t, 'note', n, v, d) for t, n, d, v in gm_perc_events],
            9, 'Percusión', 0, tempo, tpb
        )
        if trk:
            mid.tracks.append(trk)
            created_tracks.append('Percusión GM')

    mid.save(output_path)
    return created_tracks


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME DE SESIÓN FL STUDIO
# ══════════════════════════════════════════════════════════════════════════════

def generate_session_report(sections, template, library, output_midi,
                             created_tracks, ks_table, fps):
    """
    Genera un informe .txt con el mapa completo de la sesión FL Studio:
    - Canales del Mixer a crear
    - Qué library / patch cargar en cada canal
    - Qué CCs escuchar y para qué
    - Observaciones por sección (tensión, articulación dominante, percusión usada)
    """
    lines = []
    lines.append("═" * 72)
    lines.append("  ORCHESTRATOR — INFORME DE SESIÓN FL STUDIO")
    lines.append(f"  MIDI generado: {output_midi}")
    lines.append(f"  Library target: {library.upper()}")
    lines.append(f"  Plantilla: {', '.join(INSTRUMENT_NAMES.get(i['name'], i['name']) for i in template['instruments'])}")
    lines.append("═" * 72)
    lines.append("")

    # ── Mapa de canales ────────────────────────────────────────────────────
    lines.append("▸ CANALES FL STUDIO (Piano Roll → Mixer)")
    lines.append("─" * 60)
    lines.append(f"  {'Pista MIDI':<22} {'Ch':<4} {'Library / Patch sugerido':<32} {'CCs'}")
    lines.append(f"  {'-'*22} {'-'*4} {'-'*32} {'-'*20}")

    library_hints = {
        'nucleus': {
            'violin1':    'Nucleus — 16 Violins Legato / Sustain',
            'violin2':    'Nucleus — 16 Violins Sustain',
            'viola':      'Nucleus — 10 Violas Sustain / Spiccato',
            'cello':      'Nucleus — 6 Celli Legato / Sustain',
            'contrabass': 'Nucleus — 4 Basses Sustain',
            'flute':      'Nucleus — 2 Flutes Sustained',
            'oboe':       'Nucleus — 2 Oboes Sustained',
            'clarinet':   'Nucleus — 2 Clarinets Sustained',
            'bassoon':    'Nucleus — 2 Bassoons Sustained',
            'horn':       'Nucleus — 6 Horns Sustain',
            'trumpet':    'Nucleus — 3 Trumpets Legato',
            'trombone':   'Nucleus — Trombones Sustained',
            'timpani':    'Nucleus — Timpani (o Metropolis Ark Kopernikus)',
        },
        'metropolis': {
            'violin1':    'MA1 — Finckenstein High Strings Legato',
            'violin2':    'MA1 — Finckenstein High Strings Sustain',
            'viola':      'MA1 — Finckenstein High Strings (violas)',
            'cello':      'MA1 — Wolfenstein Low Strings Legato',
            'contrabass': 'MA1 — Wolfenstein Low Strings Sustain',
            'flute':      'Nucleus — 2 Flutes (MA1 no tiene maderas)',
            'oboe':       'Nucleus — 2 Oboes',
            'clarinet':   'Nucleus — 2 Clarinets',
            'bassoon':    'Nucleus — 2 Bassoons',
            'horn':       'MA1 — Schwarzdorn Horns a9 Legato',
            'trumpet':    'MA1 — Bismarckhütte Trumpets Sustain',
            'trombone':   'MA1 — Bismarckhütte Trombones Sustain',
            'timpani':    'MA1 — Kopernikus Epic Percussion',
        },
        'generic': {name: f'GM — program {GM_PROGRAMS.get(name, 0)}' for name in GM_PROGRAMS},
    }

    cc_hints = {
        'strings': 'CC1=dinámica(XFade), CC11=expression, CC64=sustain',
        'winds':   'CC1=dinámica(XFade), CC11=expression',
        'brass':   'CC1=dinámica(XFade), CC11=expression',
        'timpani': 'velocity=dinámica',
    }

    for instr_cfg in template['instruments']:
        name  = instr_cfg['name']
        ch    = instr_cfg['channel']
        disp  = INSTRUMENT_NAMES.get(name, name)
        patch = library_hints.get(library, library_hints['generic']).get(name, '—')
        family = INSTRUMENT_FAMILY.get(name, 'strings')
        ccs   = cc_hints.get(family, 'CC1=dinámica')
        lines.append(f"  {disp:<22} {ch:<4} {patch:<32} {ccs}")

    lines.append(f"  {'Timbal':<22} {'15':<4} {'(ver arriba)':<32} velocity")
    lines.append(f"  {'Percusión GM':<22} {'9':<4} {'Kit de batería / orquesta':<32} velocity")
    lines.append("")

    # ── Keyswitches usados ─────────────────────────────────────────────────
    lines.append("▸ KEYSWITCHES INSERTADOS EN EL MIDI")
    lines.append("─" * 60)
    if library == 'generic':
        lines.append("  (library=generic: no se han insertado keyswitches)")
    else:
        ks_by_family = {}
        for family, ks_map in ks_table.items():
            if family.startswith('_'):
                continue
            ks_by_family[family] = {}
            for art, note in ks_map.items():
                if note >= 0:
                    ks_by_family[family][art] = f"{midi_to_name(note)} ({note})"

        for family, arts in ks_by_family.items():
            lines.append(f"  {family.upper()}:")
            for art, note_str in arts.items():
                lines.append(f"    {art:<18} → {note_str}")
        lines.append("")
        lines.append("  ⚠ IMPORTANTE: Verifica que estos KS coincidan con tu configuración")
        lines.append("    en SINE Player / Capsule. Si has reasignado los KS, usa --ks-config.")
        lines.append("    Para exportar la tabla actual: python orchestrator.py --dump-ks")

    lines.append("")

    # ── Análisis por sección ───────────────────────────────────────────────
    lines.append("▸ ANÁLISIS DE SECCIONES")
    lines.append("─" * 60)
    for sec in sections:
        label = sec['label']
        arc   = sec['arc']
        key   = f"{sec['key_tonic']} {sec['key_mode']}"
        bpm   = sec['tempo_bpm']
        bars  = sec['bars']
        t_mean = sec['tension_curve'].get('mean', 0.5)
        t_peak = sec['tension_curve'].get('peak', 0.5)
        harm   = sec.get('harmony_complexity', 0.5)

        lines.append(f"  [{label}]  {key}  {bpm:.0f}BPM  {bars}c  arco={arc}")
        lines.append(f"       tensión media={t_mean:.2f}  pico={t_peak:.2f}  armonía={harm:.2f}")

        # Recomendaciones específicas
        if t_mean > 0.65:
            lines.append("       → Metales abiertos, bombo en tiempos 1+3, timbal activo")
        elif t_mean < 0.3:
            lines.append("       → Solo cuerdas líricas, maderas en legato, sin percusión")
        else:
            lines.append("       → Plantilla completa en mezzo-forte, timbal puntual")

        if arc == 'crescendo':
            lines.append("       → Sección en crecimiento: asegúrate de que CC1 sube progresivamente")
        elif arc == 'decrescendo':
            lines.append("       → Sección en decrescendo: CC1 baja, considerar pizzicato en cuerdas")
        elif arc == 'arch':
            lines.append("       → Sección en arco: CC1 sube y baja, revisa el pico del timbal")

        if harm > 0.7:
            lines.append(f"       → Alta complejidad armónica: considera voicings densos en acordes")

        lines.append("")

    # ── Flujo de trabajo recomendado ──────────────────────────────────────
    lines.append("▸ FLUJO DE TRABAJO EN FL STUDIO")
    lines.append("─" * 60)
    lines.append("  1. Importar el MIDI (File → Import MIDI → Split by Channel)")
    lines.append("  2. Crear una pista de instrumento por canal (ver tabla arriba)")
    lines.append("  3. Cargar la library y el patch sugerido en cada canal")
    lines.append(f"  4. En SINE/Capsule: verificar que los KS de {library.upper()} coinciden")
    lines.append("     con los de la tabla de keyswitches de este informe")
    lines.append("  5. CC1 está ya dibujado en el MIDI — en tu library debería")
    lines.append("     controlar el XFade dinámico (normalmente lo hace por defecto)")
    lines.append("  6. Añadir reverb de sala como Send Bus a todas las pistas:")
    lines.append("     Recomendado: IR de sala de conciertos o Valhalla Room")
    lines.append("     Pre-delay sugerido: 20-30ms. Wet: 25-40%")
    lines.append("  7. Escuchar cada sección aislada antes del mix global")
    lines.append("  8. Las pistas de Timbal (ch.15) y Percusión GM (ch.9) pueden")
    lines.append("     necesitar velocidades ajustadas a tu library de percusión")
    lines.append("")
    lines.append("▸ DECISIONES QUE QUEDAN PARA TI EN EL DAW")
    lines.append("─" * 60)
    lines.append("  □ Revisar transposes idiomáticos (notas fuera de rango)")
    lines.append("  □ Ajustar KS si no coinciden con tu patch")
    lines.append("  □ Dibujar CC1 en zonas donde la automatización sea demasiado brusca")
    lines.append("  □ Añadir material nuevo desde cero: tuttis, pedales armónicos")
    lines.append("  □ Percusión: ajustar pitches del timbal a las tonalidades reales")
    lines.append("  □ Revisar transiciones entre secciones (los marcadores [A][B]...")
    lines.append("    son visibles en el timeline de FL Studio)")
    lines.append("  □ Balance de mezcla por familia antes de la reverb")
    lines.append("")
    lines.append("═" * 72)

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS POR COMPÁS
#  Extrae métricas musicales bar a bar desde el MIDI crudo.
#  Son la base empírica de todas las sugerencias.
# ══════════════════════════════════════════════════════════════════════════════

def analyze_bars(tracks_raw, sections, tpb, beats_per_bar=4):
    """
    Para cada compás de cada sección calcula métricas musicales desde
    las notas reales del MIDI. Devuelve:
      {section_label: [bar_metrics_dict, ...]}

    bar_metrics incluye:
      bar_idx, abs_tick_start,
      pitch_mean, pitch_std, pitch_min, pitch_max, pitch_range,
      pitch_direction  (-1 descending, 0 flat, 1 ascending),
      velocity_mean, velocity_std,
      note_count, note_density (notes/beat),
      dur_mean_beats, gap_beats (total silence),
      strong_beat_ratio (notes on beats 1+3 / total),
      max_leap, interval_mean,
      tension  (from fingerprint curve, 0-1)
    """
    ticks_per_bar = beats_per_bar * tpb

    # Fusionar todas las pistas melódicas para análisis global
    all_melodic = []
    for tname in ('Melody', 'Counterpoint', 'Accompaniment', 'Bass'):
        all_melodic += tracks_raw.get(tname, [])
    all_melodic.sort(key=lambda x: x[0])

    result = {}

    for sec in sections:
        label      = sec['label']
        start_tick = sec['start_tick']
        n_bars     = sec['bars']
        end_tick   = sec.get('end_tick') or (start_tick + n_bars * ticks_per_bar)

        # Calcular total de ticks en la sección para interpolar tensión
        total_sec_ticks = end_tick - start_tick

        bar_list = []
        for bar_idx in range(n_bars):
            bar_start = start_tick + bar_idx * ticks_per_bar
            bar_end   = bar_start  + ticks_per_bar

            # Notas que caen en este compás
            bar_notes = [(t, p, d, v) for t, p, d, v in all_melodic
                         if bar_start <= t < bar_end]

            # Notas solo de melodía (para intervalos y dirección)
            mel_notes = [(t, p, d, v) for t, p, d, v in
                         tracks_raw.get('Melody', [])
                         if bar_start <= t < bar_end]

            m = {'bar_idx': bar_idx, 'abs_tick': bar_start}

            if not bar_notes:
                m.update({
                    'pitch_mean': 0, 'pitch_std': 0,
                    'pitch_min': 0, 'pitch_max': 0, 'pitch_range': 0,
                    'pitch_direction': 0,
                    'velocity_mean': 0, 'velocity_std': 0,
                    'note_count': 0, 'note_density': 0.0,
                    'dur_mean_beats': 0, 'gap_beats': beats_per_bar,
                    'strong_beat_ratio': 0.0,
                    'max_leap': 0, 'interval_mean': 0,
                    'tension': tension_at_tick(
                        bar_idx * ticks_per_bar, sec, total_sec_ticks, tpb),
                    'is_empty': True,
                })
                bar_list.append(m)
                continue

            pitches   = [p for _, p, _, _ in bar_notes]
            vels      = [v for _, _, _, v in bar_notes]
            durs      = [d / tpb for _, _, d, _ in bar_notes]  # en beats
            offsets   = [t - bar_start for t, _, _, _ in bar_notes]

            m['pitch_mean']  = float(np.mean(pitches))
            m['pitch_std']   = float(np.std(pitches))
            m['pitch_min']   = int(min(pitches))
            m['pitch_max']   = int(max(pitches))
            m['pitch_range'] = int(max(pitches) - min(pitches))
            m['velocity_mean'] = float(np.mean(vels))
            m['velocity_std']  = float(np.std(vels))
            m['note_count']    = len(bar_notes)
            m['note_density']  = len(bar_notes) / beats_per_bar
            m['dur_mean_beats']= float(np.mean(durs))
            m['is_empty'] = False

            # Dirección melódica desde pista Melody si hay
            if len(mel_notes) >= 2:
                first_p = mel_notes[0][1]
                last_p  = mel_notes[-1][1]
                diff = last_p - first_p
                m['pitch_direction'] = 1 if diff > 1 else (-1 if diff < -1 else 0)
            else:
                m['pitch_direction'] = 0

            # Silencio total del compás (hueco entre notas)
            occupied = sum(min(d, beats_per_bar) for d in durs)
            m['gap_beats'] = max(0.0, beats_per_bar - occupied)

            # Strong-beat ratio: notas que empiezan en beat 1 o 3
            strong_ticks = {0, tpb * 2}  # beat 1 y 3 en 4/4
            strong_count = sum(1 for off in offsets
                               if min(abs(off - s) for s in strong_ticks) < tpb // 4)
            m['strong_beat_ratio'] = strong_count / max(len(bar_notes), 1)

            # Intervalos (solo melodía)
            mel_pitches = [p for _, p, _, _ in mel_notes]
            if len(mel_pitches) >= 2:
                intervals = [abs(mel_pitches[i+1] - mel_pitches[i])
                             for i in range(len(mel_pitches) - 1)]
                m['max_leap']      = max(intervals)
                m['interval_mean'] = float(np.mean(intervals))
            else:
                m['max_leap']      = 0
                m['interval_mean'] = 0.0

            # Tensión interpolada del fingerprint
            m['tension'] = tension_at_tick(
                bar_idx * ticks_per_bar, sec, total_sec_ticks, tpb)

            bar_list.append(m)

        # Gradientes cross-bar (se calculan sobre la lista completa)
        for i, m in enumerate(bar_list):
            prev = bar_list[i-1] if i > 0 else m
            nxt  = bar_list[i+1] if i+1 < len(bar_list) else m

            m['density_gradient'] = m['note_density'] - prev['note_density']
            m['velocity_gradient']= m['velocity_mean'] - prev['velocity_mean']
            m['tension_gradient'] = m['tension'] - prev['tension']
            m['pitch_gradient']   = m['pitch_mean'] - prev['pitch_mean']

            # Plateau: densidad similar en compás actual, anterior y siguiente
            density_stable = abs(m['note_density'] - prev['note_density']) < 0.5 \
                          and abs(m['note_density'] - nxt['note_density'])  < 0.5
            tension_stable = abs(m['tension'] - prev['tension']) < 0.08 \
                          and abs(m['tension'] - nxt['tension'])  < 0.08
            m['is_plateau'] = density_stable and tension_stable

        result[label] = bar_list

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE SUGERENCIAS COMPOSITIVAS
#  Cada función de detección recibe (bar_metrics_list, section, bar_idx)
#  y devuelve None o un dict de sugerencia.
# ══════════════════════════════════════════════════════════════════════════════

# Estructura de una sugerencia:
# {
#   'type':       str  — id del elemento ('ostinato', 'pedal', ...)
#   'category':   str  — 'structural' | 'textural' | 'expressive' | 'percussive'
#   'priority':   int  — 1=urgente 2=recomendado 3=opcional
#   'section':    str  — label de sección
#   'bar':        int  — compás dentro de la sección (0-indexed)
#   'bar_abs':    int  — compás absoluto en la obra
#   'duration':   int  — compases de duración sugerida
#   'instrument': str  — instrumento sugerido (o None = varios)
#   'title':      str  — nombre legible
#   'reason':     str  — por qué aquí (datos concretos)
#   'how':        str  — cómo implementarlo en FL Studio
#   'data':       dict — métricas relevantes
# }

BAR_OFFSET_CUMULATIVE = {}  # se rellena en analyze_suggestions

def _bar_abs(section_label, bar_idx):
    return BAR_OFFSET_CUMULATIVE.get(section_label, 0) + bar_idx


def detect_ostinato_opportunities(bars, section):
    """
    Ostinato: compases con densidad baja-media y tensión estable o creciente.
    Ideal para construir energía bajo una línea lenta.
    """
    suggestions = []
    n = len(bars)
    for i, m in enumerate(bars):
        # Zona de plateau con densidad baja = espacio para ostinato
        if (m['is_plateau']
                and m['note_density'] < 1.5
                and m['tension'] > 0.25
                and not m['is_empty']
                and i < n - 2):

            # ¿Hay al menos 4 compases de plateau?
            run = 1
            for j in range(i+1, min(i+6, n)):
                if bars[j]['is_plateau'] and bars[j]['note_density'] < 1.5:
                    run += 1
                else:
                    break
            if run < 3:
                continue

            suggestions.append({
                'type': 'ostinato',
                'category': 'textural',
                'priority': 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': run,
                'instrument': 'cello / viola',
                'title': 'Ostinato de cuerdas bajas',
                'reason': (
                    f"{run} compases de densidad estable "
                    f"({m['note_density']:.1f} notas/beat) con tensión "
                    f"{m['tension']:.2f} — zona ideal para un patrón repetitivo "
                    f"que acumule energía bajo la melodía."
                ),
                'how': (
                    "Crear nueva pista de Cello/Viola. Dibujar un patrón "
                    "rítmico de 1-2 compases (p.ej. corcheas en la raíz del "
                    "acorde) y repetirlo. CC1 debe subir gradualmente durante "
                    f"los {run} compases para reforzar el crescendo."
                ),
                'data': {'density': m['note_density'], 'tension': m['tension'],
                         'run_bars': run},
            })
            i += run  # saltar el bloque ya marcado
    return suggestions


def detect_pedal_opportunities(bars, section):
    """
    Pedal armónico: tensión alta sostenida + acorde de dominante o tónica.
    Una nota sostenida en metales bajos o contrabajo crea masa sin añadir movimiento.
    """
    suggestions = []
    n = len(bars)
    fp = section.get('fp')
    if not fp:
        return suggestions

    chord_exit = fp['exit'].get('chord_roman', 'unknown')
    is_dominant_zone = chord_exit in ('V', 'VII', 'vii°')

    for i, m in enumerate(bars):
        if (m['tension'] > 0.6
                and not m['is_empty']
                and m['note_density'] > 1.0):

            # Buscar cuántos compases seguidos de alta tensión
            run = 1
            for j in range(i+1, min(i+8, n)):
                if bars[j]['tension'] > 0.55:
                    run += 1
                else:
                    break

            if run < 2:
                continue

            # La nota de pedal sugerida es la dominante de la tonalidad
            dom_name = f"{section['key_tonic']} (dominante: "
            # Calcular dominante
            pc_map = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,
                      'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
            note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            tonic_pc = pc_map.get(section['key_tonic'], 0)
            dom_pc = (tonic_pc + 7) % 12
            dom_name = note_names[dom_pc]

            suggestions.append({
                'type': 'pedal',
                'category': 'textural',
                'priority': 1 if m['tension'] > 0.75 else 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': run,
                'instrument': 'trompa / tuba / contrabajo',
                'title': f'Pedal armónico en {dom_name}',
                'reason': (
                    f"Tensión {m['tension']:.2f} sostenida durante {run} compases. "
                    f"Una nota larga de {dom_name} en el bajo crea masa y presión "
                    f"sin añadir movimiento armónico."
                    + (" La zona es dominante — el pedal reforzará la tensión "
                       "de la cadencia." if is_dominant_zone else "")
                ),
                'how': (
                    f"Nueva pista Trompa o Tuba. Dibujar una redonda de {dom_name} "
                    f"(octava 2) sostenida {run} compases. CC1 estático en 60-80. "
                    f"En Nucleus: articulación Sustain (KS A#0). "
                    f"Alternativamente en Contrabajo: nota larga con arco, sul tasto."
                ),
                'data': {'tension': m['tension'], 'run_bars': run,
                         'pedal_note': dom_name,
                         'tonality': f"{section['key_tonic']} {section['key_mode']}"},
            })
            break  # una sugerencia por sección es suficiente para pedal
    return suggestions


def detect_syncopation_opportunities(bars, section):
    """
    Síncopa: compases con strong_beat_ratio alto (todo en tiempos fuertes).
    Añadir un contra-ritmo sincopado en vientos o cuerdas da vida rítmica.
    """
    suggestions = []
    fp = section.get('fp')
    natural_syncopation = fp['meta'].get('syncopation', 0.3) if fp else 0.3

    for i, m in enumerate(bars):
        # Zona con pocas síncopa natural + densidad media + tensión media
        if (m['strong_beat_ratio'] > 0.75  # casi todo en tiempos fuertes
                and m['note_density'] > 0.8
                and 0.2 < m['tension'] < 0.75
                and not m['is_empty']
                and natural_syncopation < 0.25):  # la pieza en sí no es sincopada

            suggestions.append({
                'type': 'sincopa',
                'category': 'textural',
                'priority': 3,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': 2,
                'instrument': 'clarinete / viola',
                'title': 'Contrariedad rítmica (síncopa)',
                'reason': (
                    f"El {int(m['strong_beat_ratio']*100)}% de las notas cae en "
                    f"tiempos fuertes — el ritmo es muy cuadrado aquí. "
                    f"Un patrón sincopado en voz interior daría impulso."
                ),
                'how': (
                    "En la pista de Clarinete o Viola, añadir un motivo de 2 notas "
                    "que caiga en el 'y' del 2 y el 'y' del 4 (offbeats). "
                    "Las notas deben ser del acorde del compás. "
                    "Velocidad media (65-75) para no competir con la melodía."
                ),
                'data': {'strong_ratio': m['strong_beat_ratio'],
                         'tension': m['tension']},
            })
    return suggestions


def detect_crescendo_opportunities(bars, section):
    """
    Crescendo estructural: zona de tensión creciente antes del peak_bar.
    Si la dinámica de velocidades no sube con la tensión → hay que añadirlo.
    """
    suggestions = []
    fp = section.get('fp')
    if not fp:
        return suggestions

    peak_bar = fp['tension_curve'].get('peak_bar', len(bars) // 2)
    arc      = section['arc']

    if arc not in ('crescendo', 'arch', 'late_climax', 'awakening'):
        return suggestions

    # Buscar zona de 4+ compases antes del pico donde la velocidad NO sube
    # aunque la tensión sí
    for i in range(max(0, peak_bar - 8), peak_bar - 1):
        if i >= len(bars) - 1:
            break
        m     = bars[i]
        m_nxt = bars[i+1]
        tension_rising  = m['tension_gradient'] > 0.04
        velocity_flat   = abs(m['velocity_gradient']) < 3.0

        if tension_rising and velocity_flat and not m['is_empty']:
            run = 1
            for j in range(i+1, min(peak_bar, len(bars))):
                if bars[j]['tension_gradient'] > 0.02:
                    run += 1
                else:
                    break

            suggestions.append({
                'type': 'crescendo',
                'category': 'expressive',
                'priority': 1,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': run,
                'instrument': 'cuerdas / tutti',
                'title': f'Crescendo estructural hacia c.{peak_bar+1}',
                'reason': (
                    f"La tensión sube {m['tension_gradient']:+.2f}/compás durante "
                    f"{run} compases pero la dinámica de velocidades no lo refleja "
                    f"(variación {m['velocity_gradient']:+.1f}/compás). "
                    f"El pico de tensión está en c.{peak_bar+1}."
                ),
                'how': (
                    f"Automatización de CC1 en TODAS las pistas de cuerda: "
                    f"subir de {int(m['velocity_mean'])} a 110+ en {run} compases. "
                    f"Añadir cuerdas en tremolo si no las hay. "
                    f"En percusión: iniciar un roll de timbal en c.{i+1} "
                    f"que llegue a fortissimo en c.{peak_bar+1}."
                ),
                'data': {'tension_gradient': m['tension_gradient'],
                         'velocity_gradient': m['velocity_gradient'],
                         'peak_bar': peak_bar, 'run': run},
            })
            break
    return suggestions


def detect_silence_opportunities(bars, section):
    """
    Silencio estructural: tras una zona densa o de alta tensión,
    un compás de silencio total es más impactante que cualquier nota.
    """
    suggestions = []
    fp = section.get('fp')
    n  = len(bars)

    for i in range(1, n - 1):
        m     = bars[i]
        prev  = bars[i-1]
        nxt   = bars[i+1]

        # Condición: compás actual relativamente vacío entre dos compases densos
        density_drop = prev['note_density'] > 1.5 and m['note_density'] < 0.5
        tension_drop = prev['tension'] > 0.5 and m['tension'] < 0.4

        if density_drop and tension_drop and not nxt['is_empty']:
            suggestions.append({
                'type': 'silencio',
                'category': 'structural',
                'priority': 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': 1,
                'instrument': 'tutti',
                'title': 'Silencio estructural de impacto',
                'reason': (
                    f"El c.{i+1} ya tiene densidad {m['note_density']:.1f} "
                    f"(bajó de {prev['note_density']:.1f}). "
                    f"Ampliar a silencio total entre la zona densa anterior "
                    f"y lo que sigue crearía un contraste de máximo impacto."
                ),
                'how': (
                    "Eliminar o atenuar todas las notas de este compás en todas "
                    "las pistas. CC1 a 0 en el beat 1. Dejar que la reverb "
                    "de la sala decaiga naturalmente. "
                    "La duración exacta del silencio depende del tempo: "
                    f"a {section['tempo_bpm']:.0f} BPM, 1 compás = "
                    f"{60/section['tempo_bpm']*4:.1f} segundos."
                ),
                'data': {'prev_density': prev['note_density'],
                         'curr_density': m['note_density'],
                         'prev_tension': prev['tension']},
            })
    return suggestions


def detect_melancholy_moments(bars, section):
    """
    Melancolía: modo menor + tensión media + melodía descendente + densidad baja.
    Sugiere reforzar con maderas en registro grave o cello cantabile.
    """
    suggestions = []
    if section['key_mode'] not in ('minor', 'aeolian', 'dorian', 'phrygian'):
        return suggestions

    for i, m in enumerate(bars):
        descending = m['pitch_direction'] == -1
        mid_tension = 0.2 < m['tension'] < 0.55
        low_density = m['note_density'] < 1.2
        long_notes  = m['dur_mean_beats'] > 1.0

        if descending and mid_tension and low_density and long_notes and not m['is_empty']:
            suggestions.append({
                'type': 'melancolia',
                'category': 'expressive',
                'priority': 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': 2,
                'instrument': 'oboe / cello solístico',
                'title': 'Momento melancólico — voz solista',
                'reason': (
                    f"Modo {section['key_mode']}, melodía descendente "
                    f"({m['pitch_mean']:.0f}→, dirección {m['pitch_direction']:+d}), "
                    f"densidad baja ({m['note_density']:.1f}) con notas largas "
                    f"({m['dur_mean_beats']:.1f} beats). "
                    f"Tensión {m['tension']:.2f} — zona de introspección."
                ),
                'how': (
                    "Doble la melodía con Oboe en su registro de pecho (Bb3-E5) "
                    "o extrae la línea a un Cello solístico en sul tasto. "
                    "CC1 muy bajo (20-35), CC11 con inflexiones de frase suaves. "
                    "En Nucleus: Legato o Espressivo si está disponible. "
                    "Evitar vibrato excesivo — dejar la desnudez del timbre."
                ),
                'data': {'mode': section['key_mode'],
                         'pitch_direction': m['pitch_direction'],
                         'tension': m['tension'],
                         'dur_mean': m['dur_mean_beats']},
            })
    return suggestions


def detect_accent_candidates(bars, section):
    """
    Acentos: tiempo fuerte de compás con cambio de acorde o alta velocidad.
    Sugiere refuerzo con sforzando en metales o percusión puntual.
    """
    suggestions = []
    fp = section.get('fp')
    if not fp:
        return suggestions

    # Solo en zonas de tensión alta
    for i, m in enumerate(bars[1:], 1):
        prev = bars[i-1]
        tension_jump = m['tension'] - prev['tension'] > 0.12
        vel_jump     = m['velocity_mean'] - prev['velocity_mean'] > 8
        high_tension = m['tension'] > 0.5

        if (tension_jump or vel_jump) and high_tension and not m['is_empty']:
            suggestions.append({
                'type': 'acento',
                'category': 'expressive',
                'priority': 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': 1,
                'instrument': 'metales / percusión',
                'title': 'Acento sforzando',
                'reason': (
                    f"Salto de tensión {prev['tension']:.2f}→{m['tension']:.2f} "
                    + (f"y velocidad {prev['velocity_mean']:.0f}→{m['velocity_mean']:.0f} " if vel_jump else "")
                    + f"en el c.{i+1}. El downbeat aquí pide refuerzo percusivo."
                ),
                'how': (
                    "Beat 1 del compás: acorde de Trompas en sforzando (vel 110+, "
                    "articulación Marcato en Metropolis Ark). "
                    "Simultáneamente un golpe de bombo de orquesta (vel 100). "
                    "El resto del compás puede ser más suave (el contraste es clave). "
                    "CC1 baja bruscamente después del beat 1."
                ),
                'data': {'tension_jump': m['tension'] - prev['tension'],
                         'vel_jump': m['velocity_mean'] - prev['velocity_mean']},
            })
    return suggestions


def detect_tension_builds(bars, section):
    """
    Build de tensión: zona de aumento gradual antes del clímax.
    Sugiere técnicas de acumulación (tremolo, densidad creciente, cols legno).
    """
    suggestions = []
    fp = section.get('fp')
    if not fp:
        return suggestions

    arc = section['arc']
    if arc not in ('crescendo', 'arch', 'late_climax'):
        return suggestions

    peak_bar = fp['tension_curve'].get('peak_bar', len(bars) // 2)
    build_start = max(0, peak_bar - 6)

    if build_start >= len(bars):
        return suggestions

    build_bars = bars[build_start:peak_bar]
    if not build_bars:
        return suggestions

    avg_tension_gradient = np.mean([m['tension_gradient'] for m in build_bars])

    if avg_tension_gradient > 0.03:
        suggestions.append({
            'type': 'tension_build',
            'category': 'structural',
            'priority': 1,
            'section': section['label'],
            'bar': build_start,
            'bar_abs': _bar_abs(section['label'], build_start),
            'duration': peak_bar - build_start,
            'instrument': 'cuerdas (tremolo) + percusión',
            'title': f'Build de tensión hacia el clímax (c.{peak_bar+1})',
            'reason': (
                f"Arco {arc}: tensión sube "
                f"{bars[build_start]['tension']:.2f}→"
                f"{bars[min(peak_bar, len(bars)-1)]['tension']:.2f} "
                f"en {peak_bar - build_start} compases. "
                f"Gradiente medio: {avg_tension_gradient:+.3f}/compás."
            ),
            'how': (
                f"c.{build_start+1}: Cuerdas entran en tremolo (KS B0 en Nucleus), "
                f"CC1 en 40.\n"
                f"  c.{build_start+2}-{peak_bar}: CC1 sube linealmente hasta 110.\n"
                f"  c.{peak_bar-2}: Timbal inicia roll (vel 60→100).\n"
                f"  c.{peak_bar}: Brass entra en fff con acorde abierto. "
                f"Crash de platos en beat 1.\n"
                f"  Considerar densidad creciente: añadir una voz nueva "
                f"cada 2 compases durante el build."
            ),
            'data': {'build_start': build_start, 'peak_bar': peak_bar,
                     'avg_gradient': float(avg_tension_gradient)},
        })
    return suggestions


def detect_melodic_arc_opportunities(bars, section):
    """
    Arco melódico: zona de notas largas en registro medio-alto con baja tensión.
    Ideal para línea cantabile solista o legato amplio.
    """
    suggestions = []

    for i, m in enumerate(bars):
        high_register = m['pitch_mean'] > 66  # sobre F#4 aprox
        long_notes    = m['dur_mean_beats'] > 1.5
        low_tension   = m['tension'] < 0.4
        smooth        = m['max_leap'] < 5  # movimiento por grado

        if high_register and long_notes and low_tension and smooth and not m['is_empty']:
            # ¿Cuántos compases seguidos con estas condiciones?
            run = 1
            for j in range(i+1, min(i+8, len(bars))):
                bj = bars[j]
                if (bj['dur_mean_beats'] > 1.0
                        and bj['tension'] < 0.45
                        and not bj['is_empty']):
                    run += 1
                else:
                    break

            if run < 2:
                continue

            suggestions.append({
                'type': 'arco_melodico',
                'category': 'expressive',
                'priority': 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': run,
                'instrument': 'violin I solístico / flauta',
                'title': 'Línea cantabile en arco amplio',
                'reason': (
                    f"Registro {m['pitch_mean']:.0f} MIDI ({midi_to_name(int(m['pitch_mean']))}), "
                    f"notas largas ({m['dur_mean_beats']:.1f} beats), "
                    f"movimiento suave (max leap {m['max_leap']} st), "
                    f"tensión {m['tension']:.2f}. "
                    f"{run} compases de condiciones ideales para una línea solista."
                ),
                'how': (
                    "Extraer la melodía de Violin I a una pista independiente "
                    f"marcada como 'solista'. En Nucleus: Legato solo (no sección). "
                    "CC1 en 45-60 (no en forte — la belleza es en mezzo-piano). "
                    "CC11 con pequeño swell en la nota de llegada de cada frase. "
                    "Reducir velocidad de la sección de violines para que el "
                    "solista destaque naturalmente."
                ),
                'data': {'pitch_mean': m['pitch_mean'], 'dur_mean': m['dur_mean_beats'],
                         'max_leap': m['max_leap'], 'run': run},
            })
            break
    return suggestions


def detect_col_legno_sul_ponticello(bars, section):
    """
    Col legno / sul ponticello: alta tensión + arco dramático.
    Técnicas extendidas de cuerdas para zonas de máxima disrupción.
    """
    suggestions = []
    arc = section['arc']

    if arc not in ('arch', 'neutral', 'crescendo', 'late_climax'):
        return suggestions

    for i, m in enumerate(bars):
        very_high = m['tension'] > 0.72
        high_density = m['note_density'] > 1.5

        if very_high and high_density and not m['is_empty']:
            run = 1
            for j in range(i+1, min(i+5, len(bars))):
                if bars[j]['tension'] > 0.65:
                    run += 1
                else:
                    break

            # Elegir técnica según la densidad y el arc
            if m['note_density'] > 2.5 and arc in ('arch', 'crescendo'):
                technique = 'col legno'
                how_detail = (
                    "Duplicar la pista de Viola/Cello y cambiar la articulación "
                    "a col legno (KS C#1 en Nucleus si está disponible, o usar "
                    "LABS Spitfire — 'Bones' o 'Scary Strings' para este efecto). "
                    "La pista original puede quedar en tremolo mientras la nueva "
                    "hace el col legno rítmico."
                )
            else:
                technique = 'sul ponticello'
                how_detail = (
                    "En la pista de cuerdas existente, añadir una automatización "
                    "de CC que active sul ponticello si tu library lo soporta. "
                    "En Nucleus: no hay KS directo — usar una pista nueva con "
                    "LABS 'Scary Strings' o similar mesclada a -6dB."
                )

            suggestions.append({
                'type': technique.replace(' ', '_'),
                'category': 'textural',
                'priority': 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': run,
                'instrument': 'cuerdas',
                'title': f'Técnica extendida: {technique}',
                'reason': (
                    f"Tensión {m['tension']:.2f} sostenida {run} compases, "
                    f"densidad {m['note_density']:.1f}. "
                    f"Esta zona necesita un color tímbrico disruptivo "
                    f"para sustentar la alta tensión sin añadir más notas."
                ),
                'how': how_detail,
                'data': {'tension': m['tension'], 'density': m['note_density'],
                         'run': run},
            })
            break
    return suggestions


def detect_pizzicato_pulse(bars, section):
    """
    Pizzicato pulsado: tensión baja + acompañamiento + arco de resolución.
    Cambio de color muy efectivo en zonas líricas.
    """
    suggestions = []
    arc = section['arc']

    if arc not in ('decrescendo', 'lullaby', 'falling'):
        return suggestions

    for i, m in enumerate(bars):
        if (m['tension'] < 0.35
                and m['note_density'] > 0.5
                and m['note_density'] < 1.8
                and not m['is_empty']):

            run = 1
            for j in range(i+1, min(i+6, len(bars))):
                if bars[j]['tension'] < 0.4:
                    run += 1
                else:
                    break

            if run < 3:
                continue

            suggestions.append({
                'type': 'pizzicato_pulse',
                'category': 'textural',
                'priority': 2,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': run,
                'instrument': 'viola / violin II',
                'title': 'Acompañamiento en pizzicato',
                'reason': (
                    f"Arco {arc}, tensión {m['tension']:.2f} durante {run} compases. "
                    f"El acompañamiento actual (arco) puede sustituirse por pizzicato "
                    f"para aligerar la textura y subrayar el carácter de reposo."
                ),
                'how': (
                    "En la pista de Viola/Violin II que lleva el acompañamiento, "
                    "insertar KS de pizzicato (C#1 en Nucleus) al inicio de c."
                    f"{i+1}. Velocidades entre 55-70. "
                    "Al final del bloque, volver a arco con KS de sustain. "
                    "El patrón puede ser un simple pulso en corcheas o los "
                    "tiempos 1 y 3 según lo que diga el acompañamiento."
                ),
                'data': {'tension': m['tension'], 'arc': arc, 'run': run},
            })
            break
    return suggestions


def detect_brass_chord_entries(bars, section):
    """
    Entrada de metales: inicio de sección con tensión media-alta.
    Un acorde sostenido de trompas marca la estructura formal.
    """
    suggestions = []
    fp = section.get('fp')
    if not fp:
        return suggestions

    tension_entry = fp['tension_curve'].get('entry', 0.3)
    if tension_entry < 0.35:
        return suggestions

    # Solo el primer compás de la sección
    if bars and not bars[0]['is_empty']:
        chord = fp['entry'].get('chord_roman', 'I')
        suggestions.append({
            'type': 'brass_entry',
            'category': 'structural',
            'priority': 2,
            'section': section['label'],
            'bar': 0, 'bar_abs': _bar_abs(section['label'], 0),
            'duration': 2,
            'instrument': 'trompas',
            'title': f'Entrada de trompas en [{section["label"]}]',
            'reason': (
                f"La sección [{section['label']}] entra con tensión {tension_entry:.2f} "
                f"y acorde {chord}. Una entrada de trompas en el beat 1 "
                f"marca el cambio de sección formalmente y llena el espectro medio."
            ),
            'how': (
                f"Nueva pista Trompa (4 voces si tienes Nucleus Horn Section). "
                f"Acorde de {chord} en {section['key_tonic']} {section['key_mode']}, "
                f"2 compases de duración. "
                f"CC1 en 70-85 (no demasiado fuerte — función de relleno). "
                f"En Metropolis Ark: Sustain con swell final al c.2. "
                f"Articulación: nota larga con ataque suave."
            ),
            'data': {'tension_entry': tension_entry, 'chord': chord,
                     'key': f"{section['key_tonic']} {section['key_mode']}"},
        })
    return suggestions


def detect_hemiola(bars, section):
    """
    Hemiola: zona de densidad constante con notas en grupos de 3.
    Sugiere reorganizar el fraseo para crear 3 contra 2.
    """
    suggestions = []

    for i, m in enumerate(bars[1:-1], 1):
        prev = bars[i-1]
        nxt  = bars[i+1]

        # Densidad estable alrededor de 1.5 (3 notas por 2 tiempos)
        stable_density = abs(m['note_density'] - 1.5) < 0.4
        no_leap        = m['max_leap'] < 4
        mid_tension    = 0.25 < m['tension'] < 0.65
        not_too_fast   = m['dur_mean_beats'] > 0.4

        if stable_density and no_leap and mid_tension and not_too_fast and not m['is_empty']:
            suggestions.append({
                'type': 'hemiola',
                'category': 'textural',
                'priority': 3,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': 2,
                'instrument': 'cuerdas / maderas',
                'title': 'Hemiola (3 contra 2)',
                'reason': (
                    f"Densidad {m['note_density']:.1f} notas/beat (≈1.5) "
                    f"con movimiento suave (max leap {m['max_leap']}). "
                    f"Reorganizar en grupos de 3 semínimas crea una hemiola "
                    f"que desplaza el acento métrico — efecto de suspensión temporal."
                ),
                'how': (
                    "En el Piano Roll: seleccionar las notas de 2 compases y "
                    "reagrupar el timing en 3 grupos iguales de 4/3 beats cada uno. "
                    "La clave es que los acentos de velocidad caigan en los "
                    "nuevos downbeats del grupo de 3, no en el beat 1 del compás. "
                    "El bajo puede mantener el pulso de 4/4 para que la tensión "
                    "métrica sea explícita."
                ),
                'data': {'density': m['note_density'], 'max_leap': m['max_leap']},
            })
            break  # una sugerencia de hemiola por sección
    return suggestions


def detect_flutter_trills(bars, section):
    """
    Flutter tongue / trino: notas largas en maderas con tensión media-alta.
    """
    suggestions = []

    for i, m in enumerate(bars):
        long_notes   = m['dur_mean_beats'] > 1.0
        mid_high_ten = m['tension'] > 0.4
        not_climax   = m['tension'] < 0.8  # el flutter no es para el tutti

        if long_notes and mid_high_ten and not_climax and not m['is_empty']:
            if m['tension'] > 0.6:
                ttype, instrument = 'flutter tongue', 'flauta / clarinete'
                how = (
                    "En la pista de Flauta: insertar notas cortas repetidas muy "
                    "rápidas (32 compases, vel 70) sobre la nota larga. "
                    "Alternativamente, usar un patch de flutter si tu library "
                    "lo tiene (BBCSO Discover tiene 'FLT' como articulación). "
                    "Mezclado con la nota larga, crea una textura orgánica tensa."
                )
            else:
                ttype, instrument = 'trino', 'oboe / clarinete'
                how = (
                    "Añadir un trino (nota principal + semitono superior alternado) "
                    "sobre la nota larga en Oboe o Clarinete. "
                    "En FL Studio: duplicar la nota y la nota +1 semitono "
                    "en semicorcheas rápidas, vel 60-75. "
                    "El trino crea actividad sin añadir densidad armónica."
                )

            suggestions.append({
                'type': ttype.replace(' ', '_'),
                'category': 'textural',
                'priority': 3,
                'section': section['label'],
                'bar': i, 'bar_abs': _bar_abs(section['label'], i),
                'duration': 2,
                'instrument': instrument,
                'title': f'{ttype.title()} en maderas',
                'reason': (
                    f"Nota larga ({m['dur_mean_beats']:.1f} beats), "
                    f"tensión {m['tension']:.2f}. "
                    f"Un {ttype} añade movimiento tímbrico sin cambiar la armonía."
                ),
                'how': how,
                'data': {'dur_mean': m['dur_mean_beats'], 'tension': m['tension']},
            })
            break
    return suggestions


# ── Dispatcher principal ──────────────────────────────────────────────────────

DETECTORS = [
    detect_tension_builds,        # priority 1 — estructurales primero
    detect_crescendo_opportunities,
    detect_silence_opportunities,
    detect_brass_chord_entries,
    detect_pedal_opportunities,
    detect_ostinato_opportunities,
    detect_melodic_arc_opportunities,
    detect_melancholy_moments,
    detect_accent_candidates,
    detect_pizzicato_pulse,
    detect_col_legno_sul_ponticello,
    detect_syncopation_opportunities,
    detect_hemiola,
    detect_flutter_trills,
]

def analyze_suggestions(sections, tracks_raw, tpb, beats_per_bar=4):
    """
    Ejecuta todos los detectores sobre todas las secciones y devuelve
    la lista completa de sugerencias, ordenadas por prioridad y compás absoluto.
    """
    global BAR_OFFSET_CUMULATIVE
    BAR_OFFSET_CUMULATIVE = {}
    bar_cursor = 0
    for sec in sections:
        BAR_OFFSET_CUMULATIVE[sec['label']] = bar_cursor
        bar_cursor += sec['bars']

    print("\n  Analizando el material bar a bar...")
    bar_analysis = analyze_bars(tracks_raw, sections, tpb, beats_per_bar)

    all_suggestions = []

    for sec in sections:
        label = sec['label']
        bars  = bar_analysis.get(label, [])
        if not bars:
            continue

        sec_suggestions = []
        for detector in DETECTORS:
            found = detector(bars, sec)
            sec_suggestions.extend(found)

        print(f"  [{label}] {len(bars)} compases → {len(sec_suggestions)} sugerencias")
        all_suggestions.extend(sec_suggestions)

    # Ordenar: prioridad, luego compás absoluto
    all_suggestions.sort(key=lambda s: (s['priority'], s['bar_abs']))
    return all_suggestions


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATEADOR DE SUGERENCIAS
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_ICONS = {
    'structural':  '🏛',
    'textural':    '🎨',
    'expressive':  '🎭',
    'percussive':  '🥁',
}

PRIORITY_LABELS = {1: '★★★ URGENTE', 2: '★★  Recomendado', 3: '★   Opcional'}
PRIORITY_BARS   = {1: '████', 2: '██░░', 3: '█░░░'}

def format_suggestions_report(suggestions, sections, tracks_raw, tpb):
    """Genera el bloque de sugerencias para el informe de sesión."""
    if not suggestions:
        return "\n▸ SUGERENCIAS COMPOSITIVAS\n  (Sin sugerencias — el material está bien equilibrado)\n"

    lines = []
    lines.append("\n" + "═" * 72)
    lines.append("  SUGERENCIAS COMPOSITIVAS")
    lines.append("  Elementos a añadir manualmente en FL Studio")
    lines.append("═" * 72)

    # Resumen por categoría
    from collections import Counter
    cat_count = Counter(s['category'] for s in suggestions)
    type_count = Counter(s['type'] for s in suggestions)
    lines.append(f"\n  Total: {len(suggestions)} sugerencias  |  "
                 + "  ".join(f"{CATEGORY_ICONS.get(c,'·')} {c}: {n}"
                             for c, n in sorted(cat_count.items())))
    lines.append("")

    # Agrupar por sección
    by_section = defaultdict(list)
    for s in suggestions:
        by_section[s['section']].append(s)

    # Orden de secciones original
    section_order = [sec['label'] for sec in sections]

    for label in section_order:
        sec_suggs = by_section.get(label, [])
        if not sec_suggs:
            continue

        sec = next(s for s in sections if s['label'] == label)
        lines.append(f"  ┌─ SECCIÓN [{label}]  "
                     f"{sec['key_tonic']} {sec['key_mode']}  "
                     f"{sec['tempo_bpm']:.0f}BPM  "
                     f"arco={sec['arc']}  "
                     f"{len(sec_suggs)} sugerencias")
        lines.append("  │")

        for s in sec_suggs:
            icon     = CATEGORY_ICONS.get(s['category'], '·')
            priority = PRIORITY_LABELS.get(s['priority'], '')
            pbar     = PRIORITY_BARS.get(s['priority'], '')
            bar_disp = s['bar'] + 1  # 1-indexed para el usuario

            lines.append(f"  │  {icon} [{priority}] {pbar}")
            lines.append(f"  │  {s['title'].upper()}")
            lines.append(f"  │  Dónde: c.{bar_disp} de [{label}]  "
                         f"(c.{s['bar_abs']+1} absoluto)  "
                         f"duración: {s['duration']} compás/es  "
                         f"→ {s['instrument']}")
            lines.append(f"  │")

            # Razón: envolver a 65 chars
            reason_words = s['reason'].split()
            line_buf = "  │  Por qué: "
            for word in reason_words:
                if len(line_buf) + len(word) > 72:
                    lines.append(line_buf)
                    line_buf = "  │           " + word + " "
                else:
                    line_buf += word + " "
            lines.append(line_buf.rstrip())

            lines.append("  │")

            # Cómo: mismo wrap
            how_lines = s['how'].split('\n')
            first = True
            for hl in how_lines:
                words = hl.split()
                prefix_first = "  │  Cómo:   " if first else "  │           "
                prefix_cont  = "  │           "
                first = False
                line_buf = prefix_first
                for word in words:
                    if len(line_buf) + len(word) > 72:
                        lines.append(line_buf)
                        line_buf = prefix_cont + word + " "
                    else:
                        line_buf += word + " "
                lines.append(line_buf.rstrip())

            lines.append("  │")
            lines.append("  │  " + "─" * 60)

        lines.append("  └" + "─" * 60)
        lines.append("")

    # Checklist al final
    lines.append("▸ CHECKLIST DE IMPLEMENTACIÓN")
    lines.append("─" * 60)
    lines.append("  Elementos urgentes (★★★):")
    urgent = [s for s in suggestions if s['priority'] == 1]
    if urgent:
        for s in urgent:
            lines.append(f"  □  [{s['section']}] c.{s['bar']+1}  {s['title']}")
    else:
        lines.append("  (ninguno)")

    lines.append("\n  Elementos recomendados (★★):")
    rec = [s for s in suggestions if s['priority'] == 2]
    if rec:
        for s in rec:
            lines.append(f"  □  [{s['section']}] c.{s['bar']+1}  {s['title']}")
    else:
        lines.append("  (ninguno)")

    lines.append("\n  Elementos opcionales (★):")
    opt = [s for s in suggestions if s['priority'] == 3]
    if opt:
        for s in opt:
            lines.append(f"  □  [{s['section']}] c.{s['bar']+1}  {s['title']}")
    else:
        lines.append("  (ninguno)")

    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Orquestación automática + percusión para mocks orquestales',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('midi', help='MIDI de entrada (obra_final.mid de stitcher.py)')
    parser.add_argument('fingerprints', nargs='*', help='Archivos .fingerprint.json')
    parser.add_argument('--template',   default='chamber',
                        choices=['chamber', 'full', 'strings_only'])
    parser.add_argument('--library',    default='nucleus',
                        choices=['nucleus', 'metropolis', 'generic'])
    parser.add_argument('--ks-config',  default=None,
                        help='JSON con tabla de keyswitches personalizada')
    parser.add_argument('--dump-ks',    action='store_true',
                        help='Exportar tabla KS defaults a JSON y salir')
    parser.add_argument('--auto-fp',    action='store_true',
                        help='Buscar .fingerprint.json en el mismo dir que el MIDI')
    parser.add_argument('--report-only',action='store_true',
                        help='Solo generar informe, sin producir MIDI')
    parser.add_argument('--output',     default=None,
                        help='Nombre base de salida (default: <midi_base>_orquestado)')
    parser.add_argument('--no-perc',    action='store_true')
    parser.add_argument('--no-ks',      action='store_true')
    parser.add_argument('--no-cc',      action='store_true')
    parser.add_argument('--tempo-humanize', type=float, default=0.15)
    parser.add_argument('--verbose',    action='store_true')
    parser.add_argument('--map', nargs='+', default=[],
                        metavar='PISTA=ROL',
                        help='Reasignar pistas del MIDI a roles. Ej: --map Piano=Melody Piano=Bass '
                             'Roles válidos: Melody, Counterpoint, Accompaniment, Bass. '
                             'Una misma pista puede mapearse a varios roles.')
    args = parser.parse_args()

    # ── Dump KS ─────────────────────────────────────────────────────────────
    if args.dump_ks:
        out_path = 'keyswitch_defaults.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(KEYSWITCH_TABLES, f, indent=2, ensure_ascii=False)
        print(f"  Tabla de keyswitches exportada a: {out_path}")
        print("  Edítala y pásala con --ks-config para personalizar.")
        sys.exit(0)

    # ── Cargar KS personalizado ──────────────────────────────────────────────
    ks_table = KEYSWITCH_TABLES[args.library]
    if args.ks_config:
        try:
            with open(args.ks_config, 'r', encoding='utf-8') as f:
                custom_ks = json.load(f)
            ks_table = custom_ks.get(args.library, ks_table)
            print(f"  KS personalizados cargados desde {args.ks_config}")
        except Exception as e:
            print(f"  ⚠ No se pudo cargar --ks-config: {e}. Usando defaults.")

    # ── Cargar MIDI ─────────────────────────────────────────────────────────
    print(f"\n  Cargando MIDI: {args.midi}")
    tracks_raw, tempo, tpb, section_markers = load_midi_tracks(args.midi)
    bpm = 60_000_000 / tempo
    print(f"  Tempo: {bpm:.1f} BPM  |  Pistas detectadas: {list(tracks_raw.keys())}")
    print(f"  Marcadores de sección: {[m[1] for m in section_markers]}")
    for tname, tnotes in tracks_raw.items():
        if tnotes:
            ticks = [n[0] for n in tnotes]
            print(f"    {tname:<20}: {len(tnotes):>4} notas  tick_min={min(ticks)}  tick_max={max(ticks)}")

    # ── Remap de pistas ─────────────────────────────────────────────────────
    # Aplicar --map explícito: ej. Piano=Melody añade tracks_raw['Melody'] = tracks_raw['Piano']
    if args.map:
        for mapping in args.map:
            if '=' not in mapping:
                print(f"  ⚠ --map ignorado (formato incorrecto): '{mapping}'. Usa PISTA=ROL")
                continue
            src, dst = mapping.split('=', 1)
            src = src.strip()
            dst = dst.strip()
            # Buscar la pista fuente (exacta o parcial)
            matched = None
            if src in tracks_raw:
                matched = src
            else:
                for tname in tracks_raw:
                    if src.lower() in tname.lower() or tname.lower() in src.lower():
                        matched = tname
                        break
            if matched:
                if dst in tracks_raw:
                    # Combinar notas si el destino ya existe
                    combined = sorted(tracks_raw[dst] + tracks_raw[matched], key=lambda x: x[0])
                    tracks_raw[dst] = combined
                else:
                    tracks_raw[dst] = tracks_raw[matched]
                print(f"  Mapeado: '{matched}' → '{dst}' ({len(tracks_raw[dst])} notas)")
            else:
                print(f"  ⚠ --map: pista '{src}' no encontrada. "
                      f"Pistas disponibles: {list(tracks_raw.keys())}")

    # Auto-split: si faltan roles básicos pero hay pistas con muchas notas,
    # intentar dividir automáticamente por registro de pitch.
    PIANO_ROLES = ['Melody', 'Counterpoint', 'Accompaniment', 'Bass']
    missing_roles = [r for r in PIANO_ROLES if r not in tracks_raw or not tracks_raw[r]]
    if missing_roles:
        # Buscar pistas candidatas: las que no sean roles ya existentes y tengan notas
        existing_roles = set(PIANO_ROLES) - set(missing_roles)
        candidate_tracks = {
            n: notes for n, notes in tracks_raw.items()
            if n not in PIANO_ROLES and notes
        }
        if candidate_tracks:
            # Usar la pista con más notas como fuente de split
            best = max(candidate_tracks, key=lambda n: len(candidate_tracks[n]))
            best_notes = candidate_tracks[best]
            print(f"  Auto-split: '{best}' ({len(best_notes)} notas) → {missing_roles}")

            # Calcular percentiles de pitch para dividir en registros
            pitches = sorted(set(n[1] for n in best_notes))
            n_p = len(pitches)

            if 'Bass' in missing_roles and 'Melody' in missing_roles:
                # División tripartita: agudo=Melody, medio=Accompaniment/Counterpoint, grave=Bass
                p33 = pitches[n_p // 3]
                p66 = pitches[(2 * n_p) // 3]
                melody_notes  = [n for n in best_notes if n[1] >= p66]
                mid_notes     = [n for n in best_notes if p33 <= n[1] < p66]
                bass_notes    = [n for n in best_notes if n[1] < p33]
                if 'Melody'         in missing_roles and melody_notes:
                    tracks_raw['Melody']         = melody_notes
                    print(f"    Melody:         {len(melody_notes):>4} notas  (pitch >= {p66})")
                if 'Counterpoint'   in missing_roles and mid_notes:
                    tracks_raw['Counterpoint']   = mid_notes
                    print(f"    Counterpoint:   {len(mid_notes):>4} notas  ({p33} ≤ pitch < {p66})")
                if 'Accompaniment'  in missing_roles and mid_notes:
                    tracks_raw['Accompaniment']  = mid_notes
                    print(f"    Accompaniment:  {len(mid_notes):>4} notas  ({p33} ≤ pitch < {p66})")
                if 'Bass'           in missing_roles and bass_notes:
                    tracks_raw['Bass']           = bass_notes
                    print(f"    Bass:           {len(bass_notes):>4} notas  (pitch < {p33})")
            elif 'Bass' in missing_roles:
                # Solo falta el bajo: partir por la mediana
                mid_pitch = pitches[n_p // 2]
                tracks_raw['Bass'] = [n for n in best_notes if n[1] < mid_pitch]
                print(f"    Bass:  {len(tracks_raw['Bass']):>4} notas  (pitch < {mid_pitch})")
            elif 'Melody' in missing_roles:
                mid_pitch = pitches[n_p // 2]
                tracks_raw['Melody'] = [n for n in best_notes if n[1] >= mid_pitch]
                print(f"    Melody: {len(tracks_raw['Melody']):>4} notas  (pitch >= {mid_pitch})")

    # ── Buscar fingerprints ──────────────────────────────────────────────────
    fp_paths = list(args.fingerprints)
    if args.auto_fp and not fp_paths:
        midi_dir = Path(args.midi).parent
        fp_paths = sorted(midi_dir.glob('*.fingerprint.json'))
        print(f"  Auto-fingerprints: {[str(p) for p in fp_paths]}")

    fps = load_fingerprints(fp_paths)
    if not fps:
        print("  ⚠ Sin fingerprints: usando valores por defecto (tensión media 0.5)")

    # ── Construir mapa de secciones ─────────────────────────────────────────
    sections = build_section_map(section_markers, fps, tpb, tempo)
    print(f"  Secciones detectadas: {[s['label'] for s in sections]}")

    # ── Template ─────────────────────────────────────────────────────────────
    template = TEMPLATES[args.template]
    print(f"  Plantilla: {args.template} ({len(template['instruments'])} instrumentos)")

    # ── Solo informe ──────────────────────────────────────────────────────────
    _raw_output = args.output or Path(args.midi).stem + '_orquestado'
    base_name = _raw_output[:-4] if _raw_output.endswith('.mid') else _raw_output
    report_path = base_name + '_sesion_FL.txt'

    if args.report_only:
        suggestions = analyze_suggestions(sections, tracks_raw, tpb)
        report = generate_session_report(sections, template, args.library,
                                         base_name + '.mid', [], ks_table, fps)
        report += format_suggestions_report(suggestions, sections, tracks_raw, tpb)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  Informe guardado: {report_path}")
        print(f"  Sugerencias generadas: {len(suggestions)}")
        sys.exit(0)

    # ── Procesar instrumentos ────────────────────────────────────────────────
    print(f"\n  Procesando instrumentos...")
    instrument_events = {}

    for instr_cfg in template['instruments']:
        name       = instr_cfg['name']
        src_track  = instr_cfg['source']
        family     = INSTRUMENT_FAMILY.get(name, 'strings')
        ks_map_fam = ks_table.get(family, {})

        raw_notes = tracks_raw.get(src_track, [])
        if not raw_notes:
            # Buscar la pista por nombre parcial (ambas direcciones)
            for tname, tnotes in tracks_raw.items():
                if src_track.lower() in tname.lower() or tname.lower() in src_track.lower():
                    raw_notes = tnotes
                    if args.verbose:
                        print(f"    [{name}] pista '{src_track}' → encontrada como '{tname}'")
                    break

        if not raw_notes and len(tracks_raw) == 1:
            # MIDI de una sola pista: usar esa pista para todos los instrumentos
            raw_notes = next(iter(tracks_raw.values()))
            if args.verbose:
                only_name = next(iter(tracks_raw.keys()))
                print(f"    [{name}] MIDI de una sola pista → usando '{only_name}'")

        if not raw_notes:
            print(f"  ⚠ [{name}] Sin notas en pista '{src_track}' "
                  f"(pistas disponibles: {list(tracks_raw.keys())}) — pista vacía")
            instrument_events[name] = []
            continue

        events = process_instrument_notes(
            raw_notes, instr_cfg, sections,
            ks_map_fam, args.library,
            add_ks=not args.no_ks,
            add_cc=not args.no_cc,
            tempo_humanize=args.tempo_humanize,
            verbose=args.verbose
        )

        instrument_events[name] = events
        n_notes = sum(1 for e in events if e[1] == 'note')
        n_ks    = sum(1 for e in events if e[1] == 'ks')
        n_cc    = sum(1 for e in events if e[1] == 'cc')
        print(f"  [{INSTRUMENT_NAMES.get(name, name):<18}] {n_notes:>4} notas  "
              f"{n_ks:>3} KS  {n_cc:>4} CC")

    # ── Generar percusión ────────────────────────────────────────────────────
    print(f"\n  Generando percusión orquestal...")
    perc = generate_orchestral_percussion(sections, tpb, tempo, no_perc=args.no_perc)
    print(f"  Timbal:      {len(perc['timpani'])} eventos")
    print(f"  Percusión:   {len(perc['perc_gm'])} eventos")

    # ── Construir MIDI de salida ─────────────────────────────────────────────
    output_midi = base_name + '.mid'
    print(f"\n  Construyendo MIDI: {output_midi}")
    created = build_output_midi(
        instrument_events,
        perc['timpani'],
        perc['perc_gm'],
        template, tempo, tpb,
        args.library, output_midi
    )
    print(f"  Pistas creadas: {created}")

    # ── Generar informe ──────────────────────────────────────────────────────
    suggestions = analyze_suggestions(sections, tracks_raw, tpb)
    report = generate_session_report(sections, template, args.library,
                                     output_midi, created, ks_table, fps)
    report += format_suggestions_report(suggestions, sections, tracks_raw, tpb)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    # ── Resumen ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESUMEN FINAL")
    print("═" * 65)
    print(f"  MIDI orquestado : {output_midi}")
    print(f"  Informe sesión  : {report_path}")
    print(f"  Pistas          : {len(created)}")
    print(f"  Library         : {args.library}")
    print(f"  Plantilla       : {args.template}")
    print(f"  KS insertados   : {'sí' if not args.no_ks else 'no'}")
    print(f"  CC1/CC11        : {'sí' if not args.no_cc else 'no'}")
    print(f"  Percusión       : {'sí' if not args.no_perc else 'no'}")
    print(f"  Sugerencias     : {len(suggestions)}")
    print("═" * 65)
    print(f"\n  Abre {report_path} para el mapa completo de la sesión FL Studio.")


if __name__ == '__main__':
    main()
