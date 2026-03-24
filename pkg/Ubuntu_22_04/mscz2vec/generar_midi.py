#!/usr/bin/env python3
"""
generar_midi.py
═══════════════
Lee un fichero YAML de partitura y genera ficheros MIDI por sección.

Uso:
    python generar_midi.py                         # genera todas las secciones
    python generar_midi.py --seccion I             # solo Sección I
    python generar_midi.py --frase I.2             # solo la frase I.2
    python generar_midi.py --preview               # muestra estructura
    python generar_midi.py --completa              # obra completa en un único MIDI
    python generar_midi.py --diff v1.yaml v2.yaml  # compara dos versiones
    python generar_midi.py --estadisticas          # análisis de la obra
    python generar_midi.py --nueva-frase I.4 --compases 15-18 --leitmotiv TEMA_A
    python generar_midi.py --stems                 # un MIDI por instrumento en output/stems/
    python generar_midi.py --stems --completa      # stems de la obra completa concatenada
    python generar_midi.py --stems --seccion I     # stems solo de la sección I
    python generar_midi.py --leitmotivs            # un MIDI por leitmotiv en output/leitmotivs/
    python generar_midi.py --harmony               # un MIDI de armonía por sección en output/harmony/
    python generar_midi.py --melody                # un MIDI de melodía por sección en output/melody/
    python generar_midi.py --rhythm                # un MIDI de ritmo por sección en output/rhythm/
    python generar_midi.py --html                  # esquema visual HTML de la obra
    python generar_midi.py --render-audio          # genera MIDI y lo convierte a WAV
    python generar_midi.py --csv                   # exporta CSV de análisis
    python generar_midi.py --validar               # valida sin generar
    python generar_midi.py --yaml mi_obra.yaml --output carpeta/

ESTRUCTURA DEL YAML:
    obra:       título, subtítulo, compás, tonalidad, compases_totales
    leitmotivs: células temáticas con id, nombre, notas canónicas
    tracks:     definición de instrumentos/canales MIDI (opcional)
    secciones:  bloques de compases con frases, armonía, melodía, etc.

    Ver partitura.yaml para la estructura completa con todos los campos.
"""

import yaml
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import argparse
import os
import sys
import math
import subprocess
import shutil
from collections import defaultdict, Counter

TPB = 480
TICKS_PER_BAR = TPB * 4

# ══════════════════════════════════════════════════════════════════════════════
#  TABLA DE NOTAS
# ══════════════════════════════════════════════════════════════════════════════
NOTE_BASE  = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
ACCIDENTAL = {'s': 1, '#': 1, 'b': -1, 'f': -1}
NOTE_NAMES = ['C','Cs','D','Eb','E','F','Fs','G','Ab','A','Bb','B']

def note_to_midi(name):
    if not name or name in ('null','None',None,'-','—'):
        return None
    name = str(name).strip()
    base = NOTE_BASE.get(name[0].upper())
    if base is None:
        raise ValueError(f"Nota no reconocida: '{name}'")
    i, acc = 1, 0
    if i < len(name) and name[i] in ('s','#','b','f'):
        acc = ACCIDENTAL[name[i]]; i += 1
    return (int(name[i:]) + 1) * 12 + base + acc

def midi_to_note(n):
    return f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"

def beats_to_ticks(beats):
    return int(round(beats * TPB))

def bar_beat_to_tick(bar, beat, bar1):
    return (bar - bar1) * TICKS_PER_BAR + beats_to_ticks(beat)

# ══════════════════════════════════════════════════════════════════════════════
#  TABLA DE ACORDES
# ══════════════════════════════════════════════════════════════════════════════
CHORD_VOICINGS = {
    'C':['C2','C3','E3','G3'],'D':['D2','D3','Fs3','A3'],'E':['E2','E3','Gs3','B3'],
    'F':['F1','F2','A2','C3'],'G':['G1','G2','B2','D3'],'A':['A1','A2','Cs3','E3'],
    'B':['B1','B2','Ds3','Fs3'],'Bb':['Bb1','Bb2','D3','F3'],'Eb':['Eb2','Eb3','G3','Bb3'],
    'Ab':['Ab1','Ab2','C3','Eb3'],
    'Am':['A1','A2','C3','E3'],'Bm':['B1','B2','D3','Fs3'],'Cm':['C2','C3','Eb3','G3'],
    'Dm':['D2','D3','F3','A3'],'Em':['E2','E3','G3','B3'],'Fm':['F1','F2','Ab2','C3'],
    'Gm':['G1','G2','Bb2','D3'],
    'A7':['A1','A2','Cs3','E3','G3'],'B7':['B1','B2','Ds3','Fs3','A3'],
    'C7':['C2','C3','E3','G3','Bb3'],'D7':['D2','D3','Fs3','A3','C4'],
    'E7':['E2','E3','Gs3','B3','D4'],'F7':['F1','F2','A2','C3','Eb3'],
    'G7':['G1','G2','B2','D3','F3'],'Bb7':['Bb1','Bb2','D3','F3','Ab3'],
    'Am7':['A1','A2','C3','E3','G3'],'Dm7':['D2','D3','F3','A3','C4'],
    'Em7':['E2','E3','G3','B3','D4'],'Gm7':['G1','G2','Bb2','D3','F3'],
    'Cmaj7':['C2','C3','E3','G3','B3'],'Fmaj7':['F1','F2','A2','C3','E3'],
    'Gmaj7':['G1','G2','B2','D3','Fs3'],
    'Eo7':['E1','E2','G2','Bb2','D3'],'Bo7':['B1','B2','D3','F3'],'Co7':['C2','C3','Eb3','Gb3'],
    'Csus4':['C2','C3','F3','G3'],'Gsus4':['G1','G2','C3','D3'],
    'Dsus4':['D2','D3','G3','A3'],'Asus4':['A1','A2','D3','E3'],
    'Caug':['C2','C3','E3','Gs3'],'Gaug':['G1','G2','B2','Ds3'],
    # Aliases
    'Eø':  ['E1','E2','G2','Bb2','D3'], 'Eø7': ['E1','E2','G2','Bb2','D3'],
    'Bø':  ['B1','B2','D3','F3'],       'Bø7': ['B1','B2','D3','F3'],
    'Cø':  ['C2','C3','Eb3','Gb3'],     'Cø7': ['C2','C3','Eb3','Gb3'],
    'Bf':  ['Bb1','Bb2','D3','F3'],     'Bf7': ['Bb1','Bb2','D3','F3','Ab3'],
    'Ef':  ['Eb2','Eb3','G3','Bb3'],    'Af':  ['Ab1','Ab2','C3','Eb3'],
}
CHORD_DEFAULT = 'C'

def parse_chord(acorde_str):
    if not acorde_str or acorde_str in ('null','None',None,'—','-'):
        return []
    voicing = CHORD_VOICINGS.get(str(acorde_str))
    if voicing is None:
        print(f"  ⚠ Acorde no reconocido: '{acorde_str}' — usando {CHORD_DEFAULT}")
        voicing = CHORD_VOICINGS[CHORD_DEFAULT]
    return [note_to_midi(n) for n in voicing]

# ══════════════════════════════════════════════════════════════════════════════
#  PATRONES RÍTMICOS
# ══════════════════════════════════════════════════════════════════════════════
MARCATO_PATTERNS = {
    'cuatro_negras':[(0.0,0.82,1.0),(1.0,0.82,0.75),(2.0,0.82,0.85),(3.0,0.82,0.75)],
    'tango':[(0.0,0.82,1.0),(1.0,0.38,0.65),(1.5,0.38,0.55),(2.0,0.82,0.82)],
    'ostinato_332':[(0.0,0.42,1.0),(0.5,0.42,0.55),(1.0,0.42,0.55),
                    (1.5,0.42,0.92),(2.0,0.42,0.55),(2.5,0.42,0.55),
                    (3.0,0.42,0.88),(3.5,0.42,0.55)],
    'ostinato_332_cortado':[(0.0,0.42,1.0),(0.5,0.42,0.55),(1.0,0.42,0.55)],
    'habanera':[(0.0,0.45,1.0),(0.75,0.45,0.7),(1.0,0.45,0.8),
                (2.0,0.45,0.9),(2.75,0.45,0.65),(3.0,0.45,0.75)],
    'waltz':[(0.0,0.82,1.0),(1.0,0.82,0.6),(2.0,0.82,0.65)],
    'bolero':[(0.0,0.45,1.0),(0.5,0.45,0.5),(1.0,0.45,0.7),(1.5,0.45,0.5),
              (2.0,0.45,0.95),(2.5,0.45,0.5),(3.0,0.45,0.7),(3.5,0.45,0.5)],
    'pulso_1':[(0.0,0.5,1.0)],
    'pulso_12':[(0.0,0.5,1.0),(2.0,0.5,0.8)],
    'corcheas':[(b*0.5,0.42,1.0 if b==0 else 0.6) for b in range(8)],
}
MARCATO_PATTERNS['default'] = MARCATO_PATTERNS['tango']

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE TRACKS MIDI
# ══════════════════════════════════════════════════════════════════════════════
def ascii_safe(s):
    return s.encode('ascii','ignore').decode('ascii')

def build_track(name, channel, program, events):
    track = MidiTrack()
    track.append(MetaMessage('track_name', name=ascii_safe(name), time=0))
    track.append(Message('program_change', channel=channel, program=program, time=0))
    events.sort(key=lambda e: (e[0], 0 if e[1]=='off' else 1))
    prev = 0
    for (tick, on_off, note, vel) in events:
        delta = max(0, tick - prev)
        v = max(0, min(127, int(vel) if on_off=='on' else 0))
        track.append(Message('note_on', channel=channel, note=note, velocity=v, time=delta))
        prev = tick
    track.append(MetaMessage('end_of_track', time=0))
    return track

def check_note_overlaps(events, track_name):
    active = {}
    warnings = 0
    for (tick, on_off, note, vel) in sorted(events, key=lambda e: e[0]):
        if on_off == 'on':
            if note in active:
                print(f"  ⚠ SOLAPAMIENTO [{track_name}] {midi_to_note(note)} "
                      f"activa desde tick {active[note]}, nuevo on en tick {tick}")
                warnings += 1
            active[note] = tick
        else:
            active.pop(note, None)
    return warnings

# ══════════════════════════════════════════════════════════════════════════════
#  PISTA DE TEMPO
# ══════════════════════════════════════════════════════════════════════════════
def get_time_signature(obj, fallback):
    raw = obj.get('compas') or fallback.get('compas','4/4')
    try:
        n, d = str(raw).split('/')
        return int(n), int(d)
    except Exception:
        return 4, 4

def get_bar_ticks(seccion, bar_abs, obra=None):
    """Devuelve los ticks que ocupa un compás concreto, teniendo en cuenta
    cambios de compás en la sección (cambios_de_compas) y overrides por frase.

    El tamaño del compás puede variar compás a compás si la sección tiene
    'cambios_de_compas'. Esta función es el punto centralizado para calcular
    duraciones.

    Retorna (ticks_del_compas, (numerador, denominador)).
    """
    obra = obra or {}
    cambios = _build_meter_map(seccion, obra)
    # cambios: lista ordenada de (compas_abs, num, den)
    num, den = get_time_signature(seccion, obra)
    for (c, n, d) in reversed(cambios):
        if bar_abs >= c:
            num, den = n, d
            break
    beats_per_bar = num * (4 / den)
    ticks = int(round(beats_per_bar * TPB))
    return ticks, (num, den)


def _build_meter_map(seccion, obra=None):
    """Construye lista ordenada de (compas_abs, num, den) para la sección.

    Fuentes de cambios de compás (se fusionan en orden de prioridad):
      1. seccion.cambios_de_compas  → lista [{compas: N, compas: "X/Y"}, ...]
      2. Por cada frase con campo 'compas' propio → aplica desde frase['compases'][0]

    El compás de la sección (o de la obra) se usa como valor de arranque.
    """
    obra = obra or {}
    c_ini_sec = seccion['compases'][0]
    num0, den0 = get_time_signature(seccion, obra)

    cambios = [(c_ini_sec, num0, den0)]

    # Cambios explícitos en la sección
    for entry in seccion.get('cambios_de_compas', []):
        c_abs = int(entry['compas'])
        raw   = entry.get('nuevo_compas') or entry.get('compas_nuevo', '4/4')
        try:
            n, d = str(raw).split('/')
            cambios.append((c_abs, int(n), int(d)))
        except Exception:
            print(f"  ⚠ cambios_de_compas: formato inválido '{raw}' en c.{c_abs}")

    # Override por frase (campo 'compas' en la frase)
    for frase in seccion.get('frases', []):
        compas_frase = frase.get('compas')
        if compas_frase and '/' in str(compas_frase):
            try:
                n, d = str(compas_frase).split('/')
                cambios.append((int(frase['compases'][0]), int(n), int(d)))
            except Exception:
                pass

    cambios.sort(key=lambda x: x[0])
    return cambios


def _build_key_map_for_section(seccion, obra=None):
    """Construye lista ordenada de (compas_abs, tonalidad_str) para UNA sección.

    Fuentes (en orden de prioridad, de menos a más):
      1. obra.tonalidad                    → valor de arranque global
      2. seccion.tonalidad                 → cambia al inicio de la sección
      3. seccion.cambios_de_tonalidad[]    → cambios dentro de la sección
         formato: [{compas: N, nueva_tonalidad: "C"}, ...]
      4. Por cada frase con campo 'tonalidad' → aplica desde frase['compases'][0]

    Retorna lista ordenada: [(compas_abs, key_str), ...]
    """
    obra = obra or {}
    key_global = obra.get('obra', {}).get('tonalidad') or obra.get('tonalidad', 'C')
    key_sec    = seccion.get('tonalidad') or key_global
    c_ini      = seccion['compases'][0]

    result = [(c_ini, key_sec)]

    # Cambios explícitos dentro de la sección
    for entry in seccion.get('cambios_de_tonalidad', []):
        c_abs = int(entry['compas'])
        nueva = entry.get('nueva_tonalidad') or entry.get('tonalidad')
        if nueva:
            result.append((c_abs, str(nueva)))
        else:
            print(f"  ⚠ cambios_de_tonalidad: falta 'nueva_tonalidad' en c.{c_abs}")

    # Override por frase
    for frase in seccion.get('frases', []):
        ton_frase = frase.get('tonalidad')
        if ton_frase:
            result.append((int(frase['compases'][0]), str(ton_frase)))

    result.sort(key=lambda x: x[0])
    # Eliminar duplicados consecutivos con la misma tonalidad
    deduped = [result[0]]
    for item in result[1:]:
        if item[1] != deduped[-1][1]:
            deduped.append(item)
    return deduped


def _build_global_key_map(secciones, obra=None):
    """Construye el mapa de armaduras para TODA la obra como lista de
    (tick_abs, key_str), lista de eventos listos para insertar en el
    track de tempo global.

    tick_abs es relativo al inicio de la obra (tick 0 = compás 1 beat 0).
    Respeta cambios de compás dentro de cada sección para calcular los
    ticks correctamente.
    """
    obra = obra or {}
    events = []       # (tick_abs, key_str)
    tick_cursor = 0
    prev_key = None

    for sec in secciones:
        key_map = _build_key_map_for_section(sec, obra)
        c_ini   = sec['compases'][0]

        # Calcular tick de cada compás dentro de la sección
        def _bar_tick_in_section(bar_abs):
            """Tick relativo al inicio de la sección para bar_abs."""
            cambios = _build_meter_map(sec, obra)
            acc = 0
            current_num, current_den = cambios[0][1], cambios[0][2]
            for i in range(bar_abs - c_ini):
                b = c_ini + i
                for (c_abs, n, d) in cambios:
                    if c_abs == b:
                        current_num, current_den = n, d
                beats = current_num * (4 / current_den)
                acc += int(round(beats * TPB))
            return acc

        for (bar_abs, key_str) in key_map:
            tick_abs = tick_cursor + _bar_tick_in_section(bar_abs)
            if key_str != prev_key:
                events.append((tick_abs, key_str))
                prev_key = key_str

        # Avanzar tick_cursor al final de esta sección
        cambios = _build_meter_map(sec, obra)
        current_num, current_den = cambios[0][1], cambios[0][2]
        n_bars = sec['compases'][1] - sec['compases'][0] + 1
        for i in range(n_bars):
            bar_abs = c_ini + i
            for (c_abs, n, d) in cambios:
                if c_abs == bar_abs:
                    current_num, current_den = n, d
            beats = current_num * (4 / current_den)
            tick_cursor += int(round(beats * TPB))

    return events


def build_global_tempo_track(secciones, obra, titulo_override=None):
    """Construye el track de tempo maestro para toda la obra.

    Incluye correctamente:
      - time_signature inicial y todos los cambios de compás por sección
      - key_signature inicial y todos los cambios de armadura por sección
        (tanto cambios entre secciones como cambios_de_tonalidad intra-sección)
      - set_tempo con interpolación según tipo (fijo/accelerando/rallentando/rubato)

    Sustituye a las tres implementaciones duplicadas que existían antes
    (_make_global_tempo_track en stems_completa, _global_tempo_track en
    leitmotivs, y el bloque inline en generate_full_obra).
    """
    obra_meta   = obra.get('obra', {})
    titulo      = titulo_override or obra_meta.get('titulo', '')
    key_global  = obra_meta.get('tonalidad') or obra.get('tonalidad', 'C')

    tt = MidiTrack()
    tt.append(MetaMessage('track_name', name=ascii_safe(titulo), time=0))

    # ── Recopilar todos los eventos (tick_abs, tipo, valor) ─────────────────
    all_events = []   # (tick_abs, 'ts'/'key'/'tempo', valor)

    tick_cursor = 0
    prev_key = None

    for sec in secciones:
        c_ini, c_fin = sec['compases']
        n_bars       = c_fin - c_ini + 1
        t_cfg        = sec.get('tempo', {})
        bpm_i        = float(t_cfg.get('bpm_inicio', 120))
        bpm_f        = float(t_cfg.get('bpm_fin', bpm_i))
        tipo         = t_cfg.get('tipo', 'fijo')
        amp          = float(t_cfg.get('rubato_amplitud', 2.5))
        cambios_ts   = _build_meter_map(sec, obra)
        key_map      = _build_key_map_for_section(sec, obra)

        # Precalcular tick relativo de cada compás dentro de la sección
        bar_ticks = []   # bar_ticks[i] = tick relativo al inicio de la sección
        acc = 0
        cur_num, cur_den = cambios_ts[0][1], cambios_ts[0][2]
        for i in range(n_bars + 1):
            bar_ticks.append(acc)
            if i < n_bars:
                b = c_ini + i
                for (c_abs, n, d) in cambios_ts:
                    if c_abs == b:
                        cur_num, cur_den = n, d
                acc += int(round(cur_num * (4 / cur_den) * TPB))

        # time_signature events
        cur_num, cur_den = cambios_ts[0][1], cambios_ts[0][2]
        all_events.append((tick_cursor + bar_ticks[0], 'ts', (cur_num, cur_den)))
        for i in range(n_bars):
            b = c_ini + i
            for (c_abs, n, d) in cambios_ts:
                if c_abs == b and (n != cur_num or d != cur_den):
                    cur_num, cur_den = n, d
                    all_events.append((tick_cursor + bar_ticks[i], 'ts', (cur_num, cur_den)))

        # key_signature events
        for (bar_abs, key_str) in key_map:
            idx = bar_abs - c_ini
            tick_abs = tick_cursor + bar_ticks[min(idx, n_bars)]
            if key_str != prev_key:
                all_events.append((tick_abs, 'key', key_str))
                prev_key = key_str

        # tempo events
        for i in range(n_bars + 1):
            t = i / n_bars if n_bars > 0 else 0
            if tipo in ('accelerando', 'rallentando'):
                bpm = bpm_i + (bpm_f - bpm_i) * t
                all_events.append((tick_cursor + bar_ticks[i], 'tempo', mido.bpm2tempo(bpm)))
            elif tipo == 'rubato':
                bpm = bpm_i + (bpm_f - bpm_i) * t + amp * math.sin(i * 0.8)
                all_events.append((tick_cursor + bar_ticks[i], 'tempo', mido.bpm2tempo(max(30, bpm))))
            elif i == 0:
                all_events.append((tick_cursor, 'tempo', mido.bpm2tempo(bpm_i)))

        tick_cursor += bar_ticks[n_bars]   # avanzar al inicio de la siguiente sección

    # ── Emitir eventos ordenados ─────────────────────────────────────────────
    # Orden de prioridad dentro del mismo tick: ts → key → tempo
    TYPE_ORDER = {'ts': 0, 'key': 1, 'tempo': 2}
    all_events.sort(key=lambda e: (e[0], TYPE_ORDER.get(e[1], 9)))

    # Deduplicar key_signature consecutivos con la misma tonalidad
    deduped = []
    last_key_emitted = None
    for ev in all_events:
        if ev[1] == 'key':
            if ev[2] != last_key_emitted:
                deduped.append(ev)
                last_key_emitted = ev[2]
        else:
            deduped.append(ev)
    all_events = deduped

    # Asegurar que el primer evento sea una armadura
    if not any(e[1] == 'key' for e in all_events):
        all_events.insert(0, (0, 'key', key_global))

    prev = 0
    for (tick, etype, val) in all_events:
        delta = max(0, tick - prev)
        if etype == 'ts':
            num, den = val
            tt.append(MetaMessage('time_signature', numerator=num, denominator=den,
                       clocks_per_click=24, notated_32nd_notes_per_beat=8, time=delta))
        elif etype == 'key':
            tt.append(MetaMessage('key_signature', key=val, time=delta))
        else:
            tt.append(MetaMessage('set_tempo', tempo=val, time=delta))
        prev = tick

    tt.append(MetaMessage('end_of_track', time=0))
    return tt


def bar_beat_to_tick_v2(bar, beat, seccion, obra=None):
    """Versión de bar_beat_to_tick que respeta cambios de compás.

    Para secciones sin cambios_de_compas se comporta exactamente igual
    que la función original (retrocompatible).
    """
    obra = obra or {}
    cambios = _build_meter_map(seccion, obra)
    c_ini = seccion['compases'][0]

    # Si no hay cambios de compás, usar el camino rápido original
    if len(cambios) <= 1:
        return bar_beat_to_tick(bar, beat, c_ini)

    # Acumular ticks compás a compás
    tick = 0
    current_bar = c_ini
    num, den = cambios[0][1], cambios[0][2]

    for i in range(bar - c_ini):
        current_bar_abs = c_ini + i
        # ¿Cambia el compás en este compás?
        for (c_abs, n, d) in cambios:
            if c_abs == current_bar_abs:
                num, den = n, d
        beats_per_bar = num * (4 / den)
        tick += int(round(beats_per_bar * TPB))

    # Ahora añadir el beat dentro del compás 'bar'
    tick += beats_to_ticks(beat)
    return tick


def build_tempo_track(seccion, bar1, obra=None):
    obra = obra or {}
    track = MidiTrack()
    track.append(MetaMessage('track_name',
                  name=ascii_safe(f"Sec.{seccion['id']} -- tempo"), time=0))

    cambios = _build_meter_map(seccion, obra)
    key_map = _build_key_map_for_section(seccion, obra)   # NEW
    c_ini, c_fin = seccion['compases']
    n_bars = c_fin - c_ini + 1

    t_cfg   = seccion.get('tempo',{})
    bpm_ini = float(t_cfg.get('bpm_inicio',120))
    bpm_fin = float(t_cfg.get('bpm_fin', bpm_ini))
    tipo    = t_cfg.get('tipo','fijo')
    amp     = float(t_cfg.get('rubato_amplitud',2.5))

    # Precalcular tick relativo de cada compás
    bar_ticks = []
    acc = 0
    cur_num, cur_den = cambios[0][1], cambios[0][2]
    for i in range(n_bars + 1):
        bar_ticks.append(acc)
        if i < n_bars:
            b = c_ini + i
            for (c_abs, n, d) in cambios:
                if c_abs == b:
                    cur_num, cur_den = n, d
            acc += int(round(cur_num * (4 / cur_den) * TPB))

    # Construir lista de eventos (tick_rel, tipo, valor)
    events = []

    # time_signature
    cur_num, cur_den = cambios[0][1], cambios[0][2]
    events.append((bar_ticks[0], 'ts', (cur_num, cur_den)))
    for i in range(n_bars):
        b = c_ini + i
        for (c_abs, n, d) in cambios:
            if c_abs == b and (n != cur_num or d != cur_den):
                cur_num, cur_den = n, d
                events.append((bar_ticks[i], 'ts', (cur_num, cur_den)))

    # key_signature — inicial + cambios dentro de la sección
    prev_key = None
    for (bar_abs, key_str) in key_map:
        idx = bar_abs - c_ini
        tick_rel = bar_ticks[min(idx, n_bars)]
        if key_str != prev_key:
            events.append((tick_rel, 'key', key_str))
            prev_key = key_str

    # tempo
    for i in range(n_bars + 1):
        t = i / n_bars if n_bars > 0 else 0
        if tipo in ('accelerando', 'rallentando'):
            bpm = bpm_ini + (bpm_fin - bpm_ini) * t
            events.append((bar_ticks[i], 'tempo', mido.bpm2tempo(bpm)))
        elif tipo == 'rubato':
            bpm = bpm_ini + (bpm_fin - bpm_ini) * t + amp * math.sin(i * 0.8)
            events.append((bar_ticks[i], 'tempo', mido.bpm2tempo(max(30, bpm))))
        elif i == 0:
            events.append((0, 'tempo', mido.bpm2tempo(bpm_ini)))

    TYPE_ORDER = {'ts': 0, 'key': 1, 'tempo': 2}
    events.sort(key=lambda e: (e[0], TYPE_ORDER.get(e[1], 9)))

    prev = 0
    for (tick, etype, val) in events:
        delta = max(0, tick - prev)
        if etype == 'ts':
            num, den = val
            track.append(MetaMessage('time_signature', numerator=num, denominator=den,
                          clocks_per_click=24, notated_32nd_notes_per_beat=8, time=delta))
        elif etype == 'key':
            track.append(MetaMessage('key_signature', key=val, time=delta))
        else:
            track.append(MetaMessage('set_tempo', tempo=val, time=delta))
        prev = tick

    track.append(MetaMessage('end_of_track', time=0))
    return track

# ══════════════════════════════════════════════════════════════════════════════
#  CURVAS DE DINÁMICA POR NOTA
# ══════════════════════════════════════════════════════════════════════════════
def apply_dynamics_curve(events, dinamica):
    """Interpola crescendo/decrescendo/arco sobre la velocidad de los eventos 'on'.

    Campos en dinamica:
        tipo:    crescendo | decrescendo | arco | fijo
        vel_ini: velocidad al inicio (usa la propia nota si se omite)
        vel_fin: velocidad al final
        escala:  factor multiplicador global (0.0–2.0, default 1.0)
    """
    if not dinamica or dinamica.get('tipo','fijo') == 'fijo':
        escala = float(dinamica.get('escala',1.0)) if dinamica else 1.0
        if escala == 1.0:
            return events
        return [(t,oo,n,max(1,min(127,int(v*escala)))) for (t,oo,n,v) in events]

    tipo   = dinamica.get('tipo','fijo')
    escala = float(dinamica.get('escala',1.0))
    on_evs = [(i,t,n,v) for i,(t,oo,n,v) in enumerate(events) if oo=='on']
    if not on_evs:
        return events
    tmin = min(t for _,t,_,_ in on_evs); tmax = max(t for _,t,_,_ in on_evs)
    span = max(tmax-tmin,1)
    vel_ini = int(dinamica.get('vel_ini', on_evs[0][3]))
    vel_fin = int(dinamica.get('vel_fin', on_evs[-1][3]))
    result = list(events)
    for (orig_i, tick, note, vel) in on_evs:
        pos = (tick-tmin)/span
        if tipo in ('crescendo','crescendo_gradual'):  factor = pos
        elif tipo in ('decrescendo','diminuendo'):     factor = 1.0-pos
        elif tipo == 'arco':                           factor = math.sin(pos*math.pi)
        else:                                          factor = 0.5
        nv = max(1,min(127,int((vel_ini+(vel_fin-vel_ini)*factor)*escala)))
        t,oo,n,_ = result[orig_i]; result[orig_i] = (t,oo,n,nv)
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  ARTICULACIONES
# ══════════════════════════════════════════════════════════════════════════════
#
#  Cada nota puede tener articulación explícita (formato dict en melodia) o
#  la frase puede tener una articulación global en el campo 'articulacion'.
#  Las articulaciones modifican:
#    - ratio de duración (legato)
#    - desplazamiento de velocidad (vel_delta)
#    - CC opcionales (sfz → CC11 spike, etc.)
#
#  Articulaciones soportadas:
#    legato       ratio 0.97   vel_delta  0   (ligado, sostenido)
#    tenuto       ratio 0.98   vel_delta +2   (pleno valor, ligero peso)
#    normale      ratio 0.91   vel_delta  0   (por defecto del script)
#    portato      ratio 0.80   vel_delta -4   (semi-separado)
#    marcato      ratio 0.78   vel_delta +12  (acentuado, acortado)
#    staccato     ratio 0.45   vel_delta -6   (separado)
#    staccatissimo ratio 0.22  vel_delta -10  (muy separado)
#    accent       ratio 0.88   vel_delta +18  (acento sf)
#    sforzando    ratio 0.85   vel_delta +28  (sfz, pico brusco)
#    flutter      ratio 0.91   vel_delta  0   (flutter-tongue: trémolo rápido)
#    pizzicato    ratio 0.30   vel_delta +8   (pizz: corto y seco)
#    snap_pizz    ratio 0.18   vel_delta +22  (Bartók pizz)
#    sul_pont     ratio 0.91   vel_delta -15  (sul ponticello: más agudo y fino)
#    col_legno    ratio 0.40   vel_delta -20  (col legno: seco, percutido)
#    harmonic     ratio 0.88   vel_delta -25  (armónico: piano y etéreo)
#
#  Uso en la frase (articulación global):
#    articulacion: staccato
#
#  Uso por nota (formato dict en melodia):
#    - {compas: 3, beat: 0.0, dur: 1.0, nota: A4, vel: 80, art: sforzando}
#
#  Combinar con dinámica: la dinámica se aplica primero, la articulación después.
# ══════════════════════════════════════════════════════════════════════════════

ARTICULATIONS = {
    #                      ratio    vel_delta
    'legato':         (0.97,   0),
    'tenuto':         (0.98,  +2),
    'normale':        (0.91,   0),
    'portato':        (0.80,  -4),
    'marcato':        (0.78, +12),
    'staccato':       (0.45,  -6),
    'staccatissimo':  (0.22, -10),
    'accent':         (0.88, +18),
    'sforzando':      (0.85, +28),
    'flutter':        (0.91,   0),   # el trémolo se añade como eventos extra
    'pizzicato':      (0.30,  +8),
    'snap_pizz':      (0.18, +22),
    'sul_pont':       (0.91, -15),
    'col_legno':      (0.40, -20),
    'harmonic':       (0.88, -25),
}
ART_DEFAULT_RATIO = 0.91  # 'normale'


def resolve_articulation(art_name):
    """Devuelve (ratio, vel_delta) para una articulación dada.
    Si es None o desconocida devuelve los valores de 'normale'."""
    if not art_name:
        return ART_DEFAULT_RATIO, 0
    art_name = str(art_name).lower()
    if art_name not in ARTICULATIONS:
        print(f"  ⚠ articulación desconocida: '{art_name}' — usando normale")
        return ART_DEFAULT_RATIO, 0
    return ARTICULATIONS[art_name]


def apply_articulation_to_events(events, art_name):
    """Aplica articulación global a una lista de eventos (on/off).

    Recalcula los ticks 'off' para ajustar la duración real de cada nota
    según el ratio de la articulación. La velocidad de los 'on' se ajusta
    con vel_delta.

    Para flutter-tongue genera trémolo de semicorcheas sobre cada nota.
    """
    if not art_name or art_name == 'normale':
        return events

    ratio, vel_delta = resolve_articulation(art_name)

    # Parear on/off para recalcular duración
    paired = []
    pending = {}
    for ev in sorted(events, key=lambda e: e[0]):
        tick, on_off, note, vel = ev
        if on_off == 'on':
            pending[note] = (tick, vel)
        elif note in pending:
            t_on, v = pending.pop(note)
            paired.append((t_on, tick - t_on, note, v))

    if not paired:
        return events

    result = []

    if art_name == 'flutter':
        # Trémolo rápido: sustituir cada nota por una secuencia de semicorcheas
        step = beats_to_ticks(0.25)
        for (t_on, dur, note, vel) in paired:
            t = t_on
            while t < t_on + dur - step // 2:
                t_off = min(t + int(step * 0.80), t_on + dur)
                new_vel = max(1, min(127, vel + vel_delta))
                result.append((t,     'on',  note, new_vel))
                result.append((t_off, 'off', note, 0))
                t += step
    else:
        for (t_on, dur, note, vel) in paired:
            new_dur  = max(1, int(dur * ratio))
            new_vel  = max(1, min(127, vel + vel_delta))
            result.append((t_on,           'on',  note, new_vel))
            result.append((t_on + new_dur, 'off', note, 0))

    return sorted(result, key=lambda e: (e[0], 0 if e[1] == 'off' else 1))


def _extract_note_art(nota_dict):
    """Extrae la articulación de una nota en formato dict, si existe."""
    if not isinstance(nota_dict, dict):
        return None
    return nota_dict.get('art') or nota_dict.get('articulacion')



def transpose_events(events, semitonos):
    return [(t,oo,max(0,min(127,note+semitonos)),v) for (t,oo,note,v) in events]

def invert_events(events, eje_midi=None):
    on_notes = [note for (_,oo,note,_) in events if oo=='on']
    if not on_notes:
        return events
    eje = eje_midi if eje_midi is not None else on_notes[0]
    return [(t,oo,max(0,min(127,2*eje-note)),v) for (t,oo,note,v) in events]

def retrograde_events(events):
    if not events:
        return events
    paired = []; pending = {}
    for (tick,on_off,note,vel) in sorted(events, key=lambda e: e[0]):
        if on_off=='on':
            pending[note] = (tick,vel)
        elif note in pending:
            t_on,v = pending.pop(note); paired.append((t_on, tick-t_on, note, v))
    if not paired:
        return events
    total = max(t_on+dur for (t_on,dur,_,_) in paired)
    result = []
    for (t_on,dur,note,vel) in paired:
        new_on = total-t_on-dur
        result.append((int(new_on),'on',note,vel)); result.append((int(new_on+dur),'off',note,0))
    return sorted(result, key=lambda e:(e[0],0 if e[1]=='off' else 1))

def augment_events(events, factor=2.0):
    return [(int(t*factor),oo,note,vel) for (t,oo,note,vel) in events]

def fragment_events(events, beats_inicio=0.0, beats_fin=None):
    t0 = beats_to_ticks(beats_inicio)
    t1 = beats_to_ticks(beats_fin) if beats_fin is not None else float('inf')
    return [(t-t0,oo,note,vel) for (t,oo,note,vel) in events if t0<=t<=t1]

def apply_transformations(events, transformaciones):
    """Aplica en cadena las transformaciones del YAML.

    Tipos disponibles:
        transponer  → semitonos: N
        invertir    → eje_midi: N (opcional)
        retrogradar
        aumentar    → factor: 2.0
        fragmentar  → beats_inicio: 0.0, beats_fin: 4.0
    """
    if not transformaciones:
        return events
    for t in transformaciones:
        tipo = t.get('tipo','')
        if tipo == 'transponer':
            events = transpose_events(events, int(t.get('semitonos',0)))
        elif tipo == 'invertir':
            events = invert_events(events, t.get('eje_midi'))
        elif tipo == 'retrogradar':
            events = retrograde_events(events)
        elif tipo == 'aumentar':
            events = augment_events(events, float(t.get('factor',2.0)))
        elif tipo == 'fragmentar':
            events = fragment_events(events, float(t.get('beats_inicio',0.0)), t.get('beats_fin'))
    return events

# ══════════════════════════════════════════════════════════════════════════════
#  CONTRAPUNTO AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════
INTERVALO_ST = {3:4, 4:5, 5:7, 6:9, 7:11, 10:16}

def generate_contrapunto(events, intervalo=3, direccion='abajo'):
    """Genera voz paralela a la melodía.

    Definición en track:
        contrapunto:
          intervalo: 3      # 3=terceras  6=sextas  10=décimas
          direccion: abajo  # arriba | abajo
    """
    st = INTERVALO_ST.get(intervalo, intervalo)
    signo = 1 if direccion=='arriba' else -1
    return [(t,oo,max(0,min(127,note+signo*st)),int(v*0.85)) for (t,oo,note,v) in events]

# ══════════════════════════════════════════════════════════════════════════════
#  ARPEGGIADOR
# ══════════════════════════════════════════════════════════════════════════════
def arpeggiate_harmony(harmony_events, patron='ascendente', subdivision=0.5):
    """Convierte acordes simultáneos en arpegios.

    Activación en cada frase:
        arpegio:
          patron: ascendente     # ascendente | descendente | alternado
          subdivision: 0.5       # beats entre notas
    """
    import random
    chords = defaultdict(list)
    for (tick,on_off,note,vel) in harmony_events:
        if on_off=='on':
            chords[tick].append((note,vel))
    result = []
    ticks_sorted = sorted(chords.keys())
    for idx, t_on in enumerate(ticks_sorted):
        notes_vel = chords[t_on]
        dur_ticks = ticks_sorted[idx+1]-t_on if idx+1 < len(ticks_sorted) else TICKS_PER_BAR
        notes_vel.sort(key=lambda x:x[0])
        if patron=='descendente':       notes_vel = list(reversed(notes_vel))
        elif patron=='alternado':
            if idx%2==1:                notes_vel = list(reversed(notes_vel))
        elif patron=='aleatorio_fijo':
            rng = random.Random(t_on); rng.shuffle(notes_vel)
        step = beats_to_ticks(subdivision)
        tick_ptr=t_on; i=0
        while tick_ptr < t_on+dur_ticks:
            note,vel = notes_vel[i%len(notes_vel)]
            note_end = min(tick_ptr+step, t_on+dur_ticks)
            result.append((tick_ptr,'on',note,vel)); result.append((note_end,'off',note,0))
            tick_ptr+=step; i+=1
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  OSTINATO MELÓDICO
# ══════════════════════════════════════════════════════════════════════════════
def process_ostinato(frases, bar1):
    """Capa 'ostinato': secuencia de notas repetida cíclicamente.

    Formato YAML:
        ostinato:
          notas: [C3, E3, G3, E3]
          duracion_nota: 0.5
          velocidad: 68
          compases: [ini, fin]   # opcional, default = rango de la frase
          crescendo: false
    """
    events = []
    for frase in frases:
        ost = frase.get('ostinato')
        if not ost:
            continue
        notas_names = ost.get('notas',[])
        if not notas_names:
            continue
        dur_nota = float(ost.get('duracion_nota',0.5))
        vel_base = int(ost.get('velocidad',68))
        crescendo= bool(ost.get('crescendo',False))
        f_ini, f_fin = frase['compases']
        c_ini = int(ost.get('compases',[f_ini,f_fin])[0]) if 'compases' in ost else f_ini
        c_fin2 = int(ost.get('compases',[f_ini,f_fin])[1]) if 'compases' in ost else f_fin
        n_bars = c_fin2-c_ini+1
        notas_midi = []
        for nn in notas_names:
            try: notas_midi.append(note_to_midi(str(nn)))
            except ValueError as e: print(f"  ⚠ ostinato: {e}")
        if not notas_midi:
            continue
        step = beats_to_ticks(dur_nota)
        total_ticks = n_bars*TICKS_PER_BAR
        tick = bar_beat_to_tick(c_ini,0.0,bar1)
        t_end = tick+total_ticks; idx=0
        while tick < t_end:
            note = notas_midi[idx%len(notas_midi)]
            pos  = (tick-bar_beat_to_tick(c_ini,0.0,bar1))/max(total_ticks,1)
            vel  = int(vel_base*(0.6+0.4*pos)) if crescendo else vel_base
            vel  = max(1,min(127,vel))
            t_off = min(tick+int(step*0.88),t_end)
            events.append((tick,'on',note,vel)); events.append((t_off,'off',note,0))
            tick+=step; idx+=1
    return events

# ══════════════════════════════════════════════════════════════════════════════
#  TRÉMOLO Y TRINO
# ══════════════════════════════════════════════════════════════════════════════
def expand_tremolo_trino(events, frases, bar1):
    """Expande notas dict con tremolo: o trino: en la melodía.

    Formato dict en melodia:
        {compas: N, beat: 0.0, dur: 2.0, nota: A4, vel: 80, tremolo: 16}
        {compas: N, beat: 0.0, dur: 2.0, nota: A4, vel: 80, trino: Bb4}
    """
    extra = []
    for frase in frases:
        for entrada in frase.get('melodia',[]):
            if not isinstance(entrada, dict):
                continue
            bar=int(entrada['compas']); beat=float(entrada['beat'])
            dur=float(entrada['dur']); vel=int(entrada['vel'])
            try: note_midi=note_to_midi(entrada['nota'])
            except ValueError as e: print(f"  ⚠ tremolo/trino: {e}"); continue
            t_start = bar_beat_to_tick(bar,beat,bar1)
            t_total = beats_to_ticks(dur)
            if 'tremolo' in entrada:
                subdiv=int(entrada['tremolo'])
                step=beats_to_ticks(1.0/(subdiv/4)); t=t_start
                while t < t_start+t_total-step//2:
                    t_off=min(t+int(step*0.85),t_start+t_total)
                    extra.append((t,'on',note_midi,vel)); extra.append((t_off,'off',note_midi,0))
                    t+=step
            elif 'trino' in entrada:
                try: nota2=note_to_midi(str(entrada['trino']))
                except ValueError as e: print(f"  ⚠ trino: {e}"); continue
                step=beats_to_ticks(0.25); t=t_start; n_idx=0
                while t < t_start+t_total-step//2:
                    n=note_midi if n_idx%2==0 else nota2
                    t_off=min(t+int(step*0.85),t_start+t_total)
                    extra.append((t,'on',n,vel)); extra.append((t_off,'off',n,0))
                    t+=step; n_idx+=1
    return events+extra

# ══════════════════════════════════════════════════════════════════════════════
#  PITCH BEND (vibrato / portamento / glissando)
# ══════════════════════════════════════════════════════════════════════════════
def build_pitchbend_track(name, channel, frases, bar1):
    """Genera pista de pitch bend.

    Formato YAML en la frase (campo 'pitch_bend'):
        pitch_bend:
          - {compas: N, beat: 0.0, tipo: vibrato,    amplitud: 0.3, velocidad: 5.0, beats: 2.0}
          - {compas: N, beat: 2.0, tipo: portamento, desde: E4, hasta: F4, beats: 0.5}
          - {compas: N, beat: 0.0, tipo: glissando,  desde: C4, hasta: A4, beats: 2.0}
    """
    msgs = []; RES=16
    for frase in frases:
        for pb in frase.get('pitch_bend',[]):
            bar=int(pb['compas']); beat=float(pb.get('beat',0.0))
            tipo=pb.get('tipo','vibrato'); t0=bar_beat_to_tick(bar,beat,bar1)
            if tipo=='vibrato':
                amp=float(pb.get('amplitud',0.3)); freq=float(pb.get('velocidad',5.0))
                beats=float(pb.get('beats',2.0)); total=beats_to_ticks(beats); n_pts=int(beats*RES)
                for k in range(n_pts+1):
                    t=k/max(n_pts,1)
                    bv=int(amp*8192*math.sin(2*math.pi*freq*t*beats/4))
                    msgs.append((t0+int(t*total),max(-8192,min(8191,bv))))
            elif tipo in ('portamento','glissando'):
                desde=note_to_midi(str(pb.get('desde','C4'))); hasta=note_to_midi(str(pb.get('hasta','C4')))
                beats=float(pb.get('beats',0.5)); total=beats_to_ticks(beats); n_pts=int(beats*RES)
                delta_st=hasta-desde
                for k in range(n_pts+1):
                    t=k/max(n_pts,1); bv=int(t*delta_st*4096)
                    msgs.append((t0+int(t*total),max(-8192,min(8191,bv))))
                msgs.append((t0+total,0))
    if not msgs:
        return None
    msgs.sort(key=lambda m:m[0])
    track=MidiTrack(); track.append(MetaMessage('track_name',name=ascii_safe(f"{name} PB"),time=0))
    prev=0
    for (tick,val) in msgs:
        track.append(Message('pitchwheel',channel=channel,pitch=val,time=max(0,tick-prev))); prev=tick
    track.append(MetaMessage('end_of_track',time=0))
    return track

# ══════════════════════════════════════════════════════════════════════════════
#  CONTROL CHANGE (pedal, expression, mod wheel)
# ══════════════════════════════════════════════════════════════════════════════
def build_cc_track(name, channel, frases, bar1):
    """Genera pista de mensajes CC.

    Formato YAML en la frase (campo 'control'):
        control:
          - {compas: N, beat: 0.0, cc: 64, valor: 127}             # pedal on
          - {compas: N, beat: 2.0, cc: 11, curva: crescendo,
             beats: 4.0, valor_ini: 40, valor_fin: 127}            # expression
          - {compas: N, beat: 0.0, cc: 1, curva: vibrato, beats: 2.0, amplitud: 64}

    CC habituales:
        1=ModWheel  7=Volume  10=Pan  11=Expression  64=Sustain  91=Reverb
    """
    msgs = []; RES=8
    for frase in frases:
        for ctrl in frase.get('control',[]):
            bar=int(ctrl['compas']); beat=float(ctrl.get('beat',0.0))
            cc=int(ctrl['cc']); t0=bar_beat_to_tick(bar,beat,bar1)
            curva=ctrl.get('curva')
            if curva is None:
                msgs.append((t0,cc,int(ctrl.get('valor',0))))
            elif curva in ('crescendo','decrescendo','lineal'):
                v_ini=int(ctrl.get('valor_ini',0)); v_fin=int(ctrl.get('valor_fin',127))
                beats=float(ctrl.get('beats',4.0)); total=beats_to_ticks(beats); n_pts=int(beats*RES)
                for k in range(n_pts+1):
                    t=k/max(n_pts,1)
                    if curva=='decrescendo': t=1.0-t
                    v=int(v_ini+(v_fin-v_ini)*t)
                    msgs.append((t0+int(k/max(n_pts,1)*total),cc,max(0,min(127,v))))
            elif curva=='vibrato':
                amp=int(ctrl.get('amplitud',32)); beats=float(ctrl.get('beats',2.0))
                total=beats_to_ticks(beats); n_pts=int(beats*RES)
                for k in range(n_pts+1):
                    t=k/max(n_pts,1); v=int(64+amp*math.sin(2*math.pi*5*t*beats/4))
                    msgs.append((t0+int(t*total),cc,max(0,min(127,v))))
    if not msgs:
        return None
    msgs.sort(key=lambda m:m[0])
    track=MidiTrack(); track.append(MetaMessage('track_name',name=ascii_safe(f"{name} CC"),time=0))
    prev=0
    for (tick,cc,val) in msgs:
        track.append(Message('control_change',channel=channel,control=cc,value=val,time=max(0,tick-prev))); prev=tick
    track.append(MetaMessage('end_of_track',time=0))
    return track

# ══════════════════════════════════════════════════════════════════════════════
#  PROCESADORES DE CAPAS
# ══════════════════════════════════════════════════════════════════════════════
FUNCION_VEL = {
    'fff':110,'ff':96,'impacto':88,'dominante':84,'V7':84,'fortissimo':96,'forte':80,
    'mf':72,'mp':64,'tonica':56,'subdominante':56,'subtonica':52,'mediante':52,
    'pedal':60,'cromatismo':48,'pp':36,'ppp':28,'pianissimo':28,'deconstruccion':28,
}


# ══════════════════════════════════════════════════════════════════════════════
#  RESOLUCIÓN DE LEITMOTIV → MELODÍA CONCRETA  (Nivel 3)
# ══════════════════════════════════════════════════════════════════════════════
def _build_leitmotiv_index(obra):
    """Devuelve {lm_id: lm_dict} para búsqueda rápida."""
    return {lm['id']: lm for lm in obra.get('leitmotivs', [])}

def resolve_leitmotiv_melody(frase, bar1, lm_index, legato=0.91):
    """Construye eventos de melodía combinando alturas del leitmotiv canónico
    con el ritmo de la frase. Solo se invoca si origen_leitmotiv: true.

    Reglas
    ──────
    1. Alturas  → notas_canonicas del leitmotiv, ciclando si hay más posiciones.
    2. Ritmo    → melodia: de la frase (columna nota ignorada si es '-'/null).
                  Si melodia: vacía → ritmo_canonico: del leitmotiv.
                  Si tampoco → una nota por beat del primer compás, dur=1.0, vel=72.
    3. Transformaciones y dinámica de la frase se aplican en process_melody.

    YAML de ejemplo:
        leitmotiv: TEMA_A
        origen_leitmotiv: true
        melodia:
          - [25, 0.0, 2.0, -, 60]   # nota ignorada
          - [25, 2.0, 2.0, -, 64]
          - [26, 0.0, 4.0, -, 68]
        transformaciones:
          - tipo: transponer
            semitonos: 3

    Devuelve lista de eventos (tick,'on'/'off',midi,vel) o None si hay error.
    """
    lm_id = frase.get('leitmotiv')
    if not lm_id or lm_id in ('null','None',None,'-','—'):
        print(f"  ⚠ {frase.get('id','?')}: origen_leitmotiv=true pero sin leitmotiv asignado")
        return None
    lm = lm_index.get(lm_id)
    if lm is None:
        print(f"  ⚠ {frase.get('id','?')}: leitmotiv '{lm_id}' no encontrado")
        return None
    notas_canonicas = lm.get('notas_canonicas') or []
    if not notas_canonicas:
        print(f"  ⚠ {frase.get('id','?')}: '{lm_id}' sin notas_canonicas")
        return None

    notas_midi = []
    for nn in notas_canonicas:
        try: notas_midi.append(note_to_midi(str(nn)))
        except ValueError as e: print(f"  ⚠ leitmotiv {lm_id}: {e}")
    if not notas_midi:
        return None

    # Posiciones rítmicas
    posiciones = []
    for entrada in frase.get('melodia', []):
        if isinstance(entrada, dict):
            posiciones.append((int(entrada.get('compas', frase['compases'][0])),
                               float(entrada.get('beat', 0.0)),
                               float(entrada.get('dur', 1.0)),
                               int(entrada.get('vel', 72))))
        elif isinstance(entrada, (list, tuple)) and len(entrada) >= 5:
            bar, beat, dur, _nota, vel = entrada
            posiciones.append((int(bar), float(beat), float(dur), int(vel)))

    if not posiciones:
        ritmo = lm.get('ritmo_canonico', [])
        if ritmo:
            bar0 = frase['compases'][0]
            for entry in ritmo:
                beat_rel = float(entry[0]); dur_r = float(entry[1])
                vel_r    = int(entry[2]) if len(entry) > 2 else 72
                posiciones.append((bar0 + int(beat_rel//4), beat_rel%4, dur_r, vel_r))
        else:
            bar0 = frase['compases'][0]
            for b in range(4):
                posiciones.append((bar0, float(b), 1.0, 72))

    events = []
    for i, (bar, beat, dur, vel) in enumerate(posiciones):
        midi_note = notas_midi[i % len(notas_midi)]
        t_on  = bar_beat_to_tick(bar, beat, bar1)
        t_off = t_on + beats_to_ticks(dur * legato)
        events.append((t_on,  'on',  midi_note, vel))
        events.append((t_off, 'off', midi_note, 0))
    return events

def process_melody(frases, bar1, legato=0.91, lm_index=None, seccion=None, obra=None):
    """lm_index: dict {lm_id: lm_dict} para resolver frases con origen_leitmotiv.
    seccion/obra: necesarios para bar_beat_to_tick_v2 (cambios de compás).
    """
    lm_index = lm_index or {}
    events = []

    def _bbtt(bar, beat):
        if seccion is not None:
            return bar_beat_to_tick_v2(int(bar), float(beat), seccion, obra)
        return bar_beat_to_tick(int(bar), float(beat), bar1)

    for frase in frases:
        # Articulación global de la frase (puede sobreescribirse por nota)
        art_global = frase.get('articulacion')

        # ── Nivel 3: alturas del leitmotiv + ritmo de la frase ────────────
        if frase.get('origen_leitmotiv', False):
            resolved = resolve_leitmotiv_melody(frase, bar1, lm_index, legato)
            if resolved is not None:
                resolved = apply_transformations(resolved, frase.get('transformaciones'))
                resolved = apply_dynamics_curve(resolved, frase.get('dinamica'))
                resolved = apply_articulation_to_events(resolved, art_global)
                events.extend(resolved)
                n_on = sum(1 for e in resolved if e[1]=='on')
                print(f"    ↳ {frase.get('id','?')}: [{frase.get('leitmotiv','?')}] → {n_on} notas")
                continue

        # ── Melodía normal ─────────────────────────────────────────────────
        fe = []
        for nota in frase.get('melodia',[]):
            if isinstance(nota, dict):
                # Las notas dict con tremolo/trino se procesan en expand_tremolo_trino
                # pero aquí gestionamos las que tienen articulación por nota
                if 'tremolo' in nota or 'trino' in nota:
                    continue
                # Nota dict sin tremolo/trino: procesarla aquí con art por nota
                try:
                    bar  = int(nota.get('compas', frase['compases'][0]))
                    beat = float(nota.get('beat', 0.0))
                    dur  = float(nota.get('dur', 1.0))
                    vel  = int(nota.get('vel', 72))
                    n    = note_to_midi(str(nota['nota']))
                except (ValueError, KeyError) as e:
                    print(f"  ⚠ melodía dict: {e}"); continue

                art_nota = _extract_note_art(nota) or art_global
                ratio, vel_delta = resolve_articulation(art_nota)
                new_vel = max(1, min(127, vel + vel_delta))
                t_on  = _bbtt(bar, beat)
                t_off = t_on + beats_to_ticks(dur * ratio)
                fe.append((t_on,  'on',  n, new_vel))
                fe.append((t_off, 'off', n, 0))
                continue

            # Formato lista: [compas, beat, dur, nota, vel]
            bar, beat, dur, note_name, vel = nota
            try: n = note_to_midi(note_name)
            except ValueError as e: print(f"  ⚠ melodía: {e}"); continue

            ratio, vel_delta = resolve_articulation(art_global)
            new_vel = max(1, min(127, int(vel) + vel_delta))
            t_on  = _bbtt(bar, beat)
            t_off = t_on + beats_to_ticks(float(dur) * ratio)
            fe.append((t_on,  'on',  n, new_vel))
            fe.append((t_off, 'off', n, 0))

        fe = apply_transformations(fe, frase.get('transformaciones'))
        fe = apply_dynamics_curve(fe, frase.get('dinamica'))
        # No aplicar articulación global otra vez aquí: ya se aplicó nota a nota
        events.extend(fe)
    events = expand_tremolo_trino(events, frases, bar1)
    return events


# ══════════════════════════════════════════════════════════════════════════════
#  INVERSIONES Y BAJO INDEPENDIENTE
# ══════════════════════════════════════════════════════════════════════════════
#
#  Campos nuevos en cada entrada de armonia:
#
#    inversion: raiz | primera | segunda | tercera
#      Rota el voicing del acorde para que la nota más grave sea la correcta.
#      'primera' → primera inversión (la tercera en el bajo)
#      'segunda' → segunda inversión (la quinta en el bajo)
#      'tercera' → tercera inversión (la séptima en el bajo, solo acordes de 4 notas)
#
#    bajo: "E2"
#      Nota concreta del bajo (sustituye la nota más grave del voicing).
#      Útil para bajo pedal, cromatismo en el bajo, o acordes con bajo añadido.
#      Si se especifica junto con inversion, bajo tiene prioridad.
#
#    pedal: "A1"
#      Nota de pedal que suena durante todo el compás independientemente
#      del acorde. Se añade como capa separada por debajo del voicing.
#      Útil para pedal de tónica, dominante pedal, etc.
#
#    bajo_independiente: ["E2", "D2", "C2", "B1"]
#      Lista de notas de bajo (una por compás) que forman una línea
#      melódica independiente. Anula 'bajo' si se especifica.
#      Se puede definir a nivel de frase para toda la sección armónica.
#
#  Ejemplos:
#    - {compas: 5, acorde: Am, inversion: primera}           → Am/C
#    - {compas: 6, acorde: G,  bajo: D2}                     → G/D (seg. inversión explícita)
#    - {compas: 7, acorde: C,  pedal: G1}                    → C sobre pedal de Sol
#    - {compas: 8, acorde: Am, bajo: E2, inversion: primera} → bajo prevalece
# ══════════════════════════════════════════════════════════════════════════════

def _apply_inversion(notas_midi, inversion):
    """Rota el voicing para obtener la inversión deseada.

    Asume notas_midi ordenadas de grave a agudo.
    Cada rotación sube la nota más grave una octava.
    """
    if not notas_midi or not inversion or inversion == 'raiz':
        return notas_midi
    orden = {'primera': 1, 'segunda': 2, 'tercera': 3}
    n = orden.get(str(inversion).lower(), 0)
    notas = sorted(notas_midi)
    for _ in range(n):
        if len(notas) < 2:
            break
        notas = notas[1:] + [notas[0] + 12]
    return notas


def _apply_bajo(notas_midi, bajo_name):
    """Sustituye la nota más grave del voicing por la nota de bajo indicada.

    Si bajo_name es más grave que la nota más baja actual, se añade por debajo.
    Si es más agudo, simplemente reemplaza la nota más baja.
    """
    if not notas_midi or not bajo_name:
        return notas_midi
    try:
        bajo_midi = note_to_midi(str(bajo_name))
    except ValueError as e:
        print(f"  ⚠ bajo: {e}")
        return notas_midi
    notas = sorted(notas_midi)
    # Eliminar la nota más grave y añadir el bajo
    notas = notas[1:] + [bajo_midi]
    return sorted(notas)


def process_harmony(frases, bar1, seccion=None, obra=None):
    events = []

    def _bbtt(bar, beat):
        if seccion is not None:
            return bar_beat_to_tick_v2(int(bar), float(beat), seccion, obra)
        return bar_beat_to_tick(int(bar), float(beat), bar1)

    for frase in frases:
        fe = []
        pedal_events = []
        bajo_ind = frase.get('bajo_independiente', [])  # línea de bajo de la frase

        for idx, item in enumerate(frase.get('armonia', [])):
            bar       = int(item['compas'])
            acorde_str = item.get('acorde')
            notes     = parse_chord(acorde_str)
            if not notes:
                continue

            funcion   = str(item.get('funcion', ''))
            vel_base  = next((v for k, v in FUNCION_VEL.items() if k in funcion), 56)
            t_on      = _bbtt(bar, 0.0)
            t_off     = t_on + beats_to_ticks(3.85)

            # ── Inversión ────────────────────────────────────────────────────
            inversion = item.get('inversion')
            if inversion:
                notes = _apply_inversion(notes, inversion)

            # ── Bajo explícito (prioridad sobre inversión) ───────────────────
            bajo_name = item.get('bajo')
            if not bajo_name and bajo_ind:
                bajo_name = bajo_ind[idx % len(bajo_ind)]
            if bajo_name:
                notes = _apply_bajo(notes, bajo_name)

            for n in notes:
                fe.append((t_on,  'on',  n, vel_base))
                fe.append((t_off, 'off', n, 0))

            # ── Pedal armónico ───────────────────────────────────────────────
            pedal_name = item.get('pedal')
            if pedal_name:
                try:
                    pedal_midi = note_to_midi(str(pedal_name))
                    vel_pedal  = max(1, vel_base - 18)  # más suave que el acorde
                    pedal_events.append((t_on,  'on',  pedal_midi, vel_pedal))
                    pedal_events.append((t_off, 'off', pedal_midi, 0))
                except ValueError as e:
                    print(f"  ⚠ pedal: {e}")

        # Arpegio
        arpegio_cfg = frase.get('arpegio')
        if arpegio_cfg and fe:
            fe = arpeggiate_harmony(fe, arpegio_cfg.get('patron', 'ascendente'),
                                    float(arpegio_cfg.get('subdivision', 0.5)))

        events.extend(fe)
        events.extend(pedal_events)

    return events


def process_marcato(frases, bar1, patrones_extra=None):
    """patrones_extra: dict de patrones adicionales leídos del YAML (se fusionan
    con MARCATO_PATTERNS; los del YAML tienen prioridad sobre los del script)."""
    patrones = {**MARCATO_PATTERNS, **(patrones_extra or {})}
    events = []
    for frase in frases:
        marc=frase.get('marcato',{})
        if not marc or not marc.get('activo',False): continue
        c_ini,c_fin=frase['compases']; n_bars=c_fin-c_ini+1
        int_ini=float(marc.get('intensidad_inicio',marc.get('intensidad',0.5)))
        int_fin=float(marc.get('intensidad_fin',int_ini))
        notas_bajo=marc.get('notas_bajo',['C2']*n_bars)
        patron=patrones.get(marc.get('patron_ritmico','default'),patrones['default'])
        for i in range(n_bars):
            bar=c_ini+i; t=i/max(n_bars-1,1); vol=int_ini+(int_fin-int_ini)*t
            bass_name=notas_bajo[i%len(notas_bajo)] if notas_bajo else 'C2'
            try: bass=note_to_midi(str(bass_name))
            except ValueError: bass=note_to_midi('C2')
            for (beat,dur,factor) in patron:
                vel=int(min(127,vol*factor*127))
                if vel<5: continue
                t_on=bar_beat_to_tick(bar,beat,bar1); t_off=t_on+beats_to_ticks(dur*0.86)
                events.append((t_on,'on',bass,vel)); events.append((t_off,'off',bass,0))
    return events

def process_solo(frases, bar1, capa='solo', legato=0.91):
    events = []
    for frase in frases:
        bloque = frase.get(capa)
        if not bloque: continue
        fe = []
        art_global = frase.get('articulacion')
        if isinstance(bloque, list):
            for nota in bloque:
                if isinstance(nota, dict):
                    try:
                        bar  = int(nota.get('compas', frase['compases'][0]))
                        beat = float(nota.get('beat', 0.0))
                        dur  = float(nota.get('dur', 1.0))
                        vel  = int(nota.get('vel', 72))
                        n    = note_to_midi(str(nota['nota']))
                    except (ValueError, KeyError) as e:
                        print(f"  ⚠ {capa} dict: {e}"); continue
                    art_nota = nota.get('art') or art_global
                    ratio, vel_delta = resolve_articulation(art_nota)
                    new_vel = max(1, min(127, vel + vel_delta))
                    t_on  = bar_beat_to_tick(int(bar), float(beat), bar1)
                    t_off = t_on + beats_to_ticks(dur * ratio)
                    fe.append((t_on, 'on', n, new_vel))
                    fe.append((t_off, 'off', n, 0))
                else:
                    bar, beat, dur, note_name, vel = nota
                    try: n = note_to_midi(note_name)
                    except ValueError as e: print(f"  ⚠ {capa}: {e}"); continue
                    ratio, vel_delta = resolve_articulation(art_global)
                    new_vel = max(1, min(127, int(vel) + vel_delta))
                    t_on  = bar_beat_to_tick(int(bar), float(beat), bar1)
                    t_off = t_on + beats_to_ticks(float(dur) * ratio)
                    fe.append((t_on, 'on', n, new_vel))
                    fe.append((t_off, 'off', n, 0))
        elif 'melodia' in bloque:
            for nota in bloque['melodia']:
                if isinstance(nota, dict):
                    try:
                        bar  = int(nota.get('compas', frase['compases'][0]))
                        beat = float(nota.get('beat', 0.0))
                        dur  = float(nota.get('dur', 1.0))
                        vel  = int(nota.get('vel', 72))
                        n    = note_to_midi(str(nota['nota']))
                    except (ValueError, KeyError) as e:
                        print(f"  ⚠ {capa} dict: {e}"); continue
                    art_nota = nota.get('art') or art_global
                    ratio, vel_delta = resolve_articulation(art_nota)
                    new_vel = max(1, min(127, vel + vel_delta))
                    t_on  = bar_beat_to_tick(int(bar), float(beat), bar1)
                    t_off = t_on + beats_to_ticks(dur * ratio)
                    fe.append((t_on, 'on', n, new_vel))
                    fe.append((t_off, 'off', n, 0))
                else:
                    bar, beat, dur, note_name, vel = nota
                    try: n = note_to_midi(note_name)
                    except ValueError as e: print(f"  ⚠ {capa}: {e}"); continue
                    ratio, vel_delta = resolve_articulation(art_global)
                    new_vel = max(1, min(127, int(vel) + vel_delta))
                    t_on  = bar_beat_to_tick(int(bar), float(beat), bar1)
                    t_off = t_on + beats_to_ticks(float(dur) * ratio)
                    fe.append((t_on, 'on', n, new_vel))
                    fe.append((t_off, 'off', n, 0))
        else:
            patron_raw = bloque.get('patron', [])
            apariciones = bloque.get('apariciones', [])
            for bar_start in apariciones:
                for entry in patron_raw:
                    if len(entry) < 4: continue
                    beat_abs, dur, note_name, vel = entry
                    bar_offset = int(beat_abs // 4); beat_in_bar = beat_abs % 4
                    actual_bar = bar_start + bar_offset
                    try: n = note_to_midi(str(note_name))
                    except ValueError as e: print(f"  ⚠ {capa}: {e}"); continue
                    t_on  = bar_beat_to_tick(actual_bar, beat_in_bar, bar1)
                    t_off = t_on + beats_to_ticks(float(dur) * legato)
                    fe.append((t_on, 'on', n, int(min(vel, 127))))
                    fe.append((t_off, 'off', n, 0))
        fe = apply_transformations(fe, bloque.get('transformaciones') if isinstance(bloque, dict) else None)
        fe = apply_dynamics_curve(fe, bloque.get('dinamica') if isinstance(bloque, dict) else None)
        events.extend(fe)
    return events


def process_pad(frases, bar1):
    events=[]
    for frase in frases:
        for entry in frase.get('pad',[]):
            bar=int(entry['compas']); vel=int(entry.get('velocidad',52))
            dur=float(entry.get('duracion',3.8)); notas=entry.get('notas',[])
            t_on=bar_beat_to_tick(bar,0.0,bar1); t_off=t_on+beats_to_ticks(dur)
            for n_name in notas:
                try: n=note_to_midi(str(n_name)); events.append((t_on,'on',n,vel)); events.append((t_off,'off',n,0))
                except ValueError as e: print(f"  ⚠ pad: {e}")
    return events

def process_percusion(frases, bar1):
    events = []
    for frase in frases:
        perc = frase.get('percusion')
        if not perc: continue
        if isinstance(perc, dict):
            patron_raw  = perc.get('patron', [])
            apariciones = perc.get('apariciones', [])
            for bar_start in apariciones:
                for entry in patron_raw:
                    if len(entry) < 3: continue
                    if len(entry) == 3:
                        beat_abs, dur, vel = entry; nota_num = 36
                    else:
                        beat_abs, dur, nota_num, vel = entry[:4]
                    bar_offset  = int(float(beat_abs) // 4)
                    beat_in_bar = float(beat_abs) % 4
                    actual_bar  = bar_start + bar_offset
                    t_on  = bar_beat_to_tick(actual_bar, beat_in_bar, bar1)
                    t_off = t_on + beats_to_ticks(float(dur))
                    events.append((t_on,  'on',  int(nota_num), int(min(vel, 127))))
                    events.append((t_off, 'off', int(nota_num), 0))
        else:
            for entrada in perc:
                bar, beat, nota_num, vel = entrada
                t_on  = bar_beat_to_tick(int(bar), float(beat), bar1)
                t_off = t_on + beats_to_ticks(0.4)
                events.append((t_on,  'on',  int(nota_num), int(vel)))
                events.append((t_off, 'off', int(nota_num), 0))
    return events

DEFAULT_TRACKS = [
    {'id':'melodia','nombre':'Melodia', 'canal':0,'programa':40,'capa':'melodia'},
    {'id':'armonia','nombre':'Armonia', 'canal':1,'programa':48,'capa':'armonia'},
    {'id':'bajo',   'nombre':'Bajo',    'canal':2,'programa':32,'capa':'marcato'},
    {'id':'solo',   'nombre':'Solo',    'canal':3,'programa':40,'capa':'solo'},
    {'id':'pad',    'nombre':'Pad',     'canal':4,'programa':88,'capa':'pad'},
    {'id':'perc',   'nombre':'Percusion','canal':9,'programa':0,'capa':'percusion'},
    {'id':'ostinato','nombre':'Ostinato','canal':5,'programa':48,'capa':'ostinato'},
]
def get_capa_events(capa, frases, bar1, patrones_extra=None, lm_index=None,
                    seccion=None, obra=None):
    """Despacha al procesador correcto.

    patrones_extra: patrones rítmicos del YAML (se fusionan con MARCATO_PATTERNS).
    lm_index:       {lm_id: lm_dict} para resolver frases con origen_leitmotiv.
    seccion/obra:   para bar_beat_to_tick_v2 (cambios de compás).
    """
    if capa == 'melodia':   return process_melody(frases, bar1, lm_index=lm_index,
                                                   seccion=seccion, obra=obra)
    if capa == 'armonia':   return process_harmony(frases, bar1,
                                                    seccion=seccion, obra=obra)
    if capa == 'marcato':   return process_marcato(frases, bar1, patrones_extra)
    if capa == 'pad':       return process_pad(frases, bar1)
    if capa == 'percusion': return process_percusion(frases, bar1)
    if capa == 'ostinato':  return process_ostinato(frases, bar1)
    return process_solo(frases, bar1, capa=capa)

# ══════════════════════════════════════════════════════════════════════════════
#  VALIDADOR DE YAML
# ══════════════════════════════════════════════════════════════════════════════
def validate_yaml(obra):
    errors=0; warnings=0
    def err(msg): nonlocal errors; print(f"  ✗ {msg}"); errors+=1
    def warn(msg): nonlocal warnings; print(f"  ⚠ {msg}")
    print(f"\n{'─'*60}\n  VALIDANDO YAML…\n{'─'*60}")
    if 'obra' not in obra: err("Falta la clave 'obra'"); return errors
    if 'titulo' not in obra['obra']: err("obra.titulo es obligatorio")
    lm_ids={lm['id'] for lm in obra.get('leitmotivs',[])}
    # Patrones válidos = los del script + los definidos en el YAML
    patrones_validos = set(MARCATO_PATTERNS.keys()) | set(obra.get('patrones_ritmicos', {}).keys())
    secciones=obra.get('secciones',[])
    if not secciones: warn("No hay secciones definidas")
    prev_fin=None
    for sec in secciones:
        sid=sec.get('id','?'); prefix=f"Sec.{sid}"
        for campo in ('id','nombre','compases','tempo'):
            if campo not in sec: err(f"{prefix}: falta '{campo}'")
        if 'compases' in sec:
            c_ini,c_fin=sec['compases']
            if c_ini>c_fin: err(f"{prefix}: c_ini({c_ini}) > c_fin({c_fin})")
            if prev_fin is not None and c_ini!=prev_fin+1:
                warn(f"{prefix}: c.{c_ini} no contiguo (anterior termina en c.{prev_fin})")
            prev_fin=c_fin
        # Validar cambios de compás
        for cc_entry in sec.get('cambios_de_compas', []):
            raw = cc_entry.get('nuevo_compas') or cc_entry.get('compas_nuevo', '')
            if '/' not in str(raw):
                err(f"{prefix}: cambios_de_compas: formato inválido '{raw}' (usar 'N/D')")
            c_abs = cc_entry.get('compas')
            if c_abs and 'compases' in sec:
                if not (sec['compases'][0] <= int(c_abs) <= sec['compases'][1]):
                    err(f"{prefix}: cambios_de_compas: c.{c_abs} fuera de la sección")
        # Validar cambios de tonalidad
        for ct_entry in sec.get('cambios_de_tonalidad', []):
            nueva = ct_entry.get('nueva_tonalidad') or ct_entry.get('tonalidad')
            if not nueva:
                err(f"{prefix}: cambios_de_tonalidad: falta 'nueva_tonalidad'")
            c_abs = ct_entry.get('compas')
            if c_abs and 'compases' in sec:
                if not (sec['compases'][0] <= int(c_abs) <= sec['compases'][1]):
                    err(f"{prefix}: cambios_de_tonalidad: c.{c_abs} fuera de la sección")
        if 'tempo' in sec:
            t=sec['tempo']
            if 'bpm_inicio' not in t: err(f"{prefix}: falta tempo.bpm_inicio")
            if t.get('tipo','fijo') not in ('fijo','accelerando','rallentando','rubato'):
                err(f"{prefix}: tempo.tipo '{t.get('tipo')}' no reconocido")
        for frase in sec.get('frases',[]):
            fid=frase.get('id','?'); fp=f"{prefix} F.{fid}"
            for campo in ('id','nombre','compases'):
                if campo not in frase: err(f"{fp}: falta '{campo}'")
            if 'compases' in frase and 'compases' in sec:
                fc_ini,fc_fin=frase['compases']; sc_ini,sc_fin=sec['compases']
                if fc_ini<sc_ini or fc_fin>sc_fin:
                    err(f"{fp}: compases [{fc_ini},{fc_fin}] fuera de [{sc_ini},{sc_fin}]")
            lm=frase.get('leitmotiv')
            if lm and lm not in lm_ids and lm not in ('null','None',None):
                warn(f"{fp}: leitmotiv '{lm}' no definido")
            if frase.get('origen_leitmotiv', False):
                if not lm or lm in ('null','None',None):
                    err(f"{fp}: origen_leitmotiv=true pero falta campo leitmotiv:")
                elif lm in lm_ids:
                    lm_obj=next((l for l in obra.get('leitmotivs',[]) if l['id']==lm),None)
                    if lm_obj and not lm_obj.get('notas_canonicas'):
                        err(f"{fp}: origen_leitmotiv=true pero '{lm}' sin notas_canonicas")
            for item in frase.get('armonia',[]):
                ac=item.get('acorde')
                if ac and ac not in ('null','None',None,'—','-') and ac not in CHORD_VOICINGS:
                    err(f"{fp}: acorde '{ac}' no en CHORD_VOICINGS")
                inv = item.get('inversion')
                if inv and inv not in ('raiz','primera','segunda','tercera'):
                    err(f"{fp}: inversion '{inv}' no válida (raiz|primera|segunda|tercera)")
                bajo = item.get('bajo')
                if bajo:
                    try: note_to_midi(str(bajo))
                    except (ValueError, IndexError) as e: err(f"{fp}: bajo inválido: {e}")
                pedal = item.get('pedal')
                if pedal:
                    try: note_to_midi(str(pedal))
                    except (ValueError, IndexError) as e: err(f"{fp}: pedal inválido: {e}")
            # Articulación global de la frase
            art = frase.get('articulacion')
            if art and art not in ARTICULATIONS:
                warn(f"{fp}: articulacion '{art}' desconocida")
            for nota in frase.get('melodia',[]):
                nn=(nota.get('nota') if isinstance(nota,dict) else nota[3] if len(nota)>3 else None)
                if nn:
                    try: note_to_midi(str(nn))
                    except (ValueError,IndexError) as e: err(f"{fp}: nota inválida: {e}")
                # Articulación por nota
                if isinstance(nota, dict):
                    art_n = nota.get('art') or nota.get('articulacion')
                    if art_n and art_n not in ARTICULATIONS:
                        warn(f"{fp}: articulacion por nota '{art_n}' desconocida")
            marc=frase.get('marcato',{})
            if marc and marc.get('activo'):
                patron=marc.get('patron_ritmico','default')
                if patron not in patrones_validos: err(f"{fp}: patron_ritmico '{patron}' no definido")
    print(f"\n  {'✓ Sin errores' if not errors else f'✗ {errors} error(es)'}"
          f"{'  ·  '+str(warnings)+' aviso(s)' if warnings else ''}")
    print(f"{'─'*60}")
    return errors

# ══════════════════════════════════════════════════════════════════════════════
#  ESTADÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════
def estadisticas(obra):
    print(f"\n{'═'*60}\n  ESTADÍSTICAS — {obra['obra']['titulo']}\n{'═'*60}")
    secciones=obra.get('secciones',[])
    total_bars=sum(s['compases'][1]-s['compases'][0]+1 for s in secciones)
    avg_bpm=sum((s['tempo']['bpm_inicio']+s['tempo'].get('bpm_fin',s['tempo']['bpm_inicio']))/2
                for s in secciones)/max(len(secciones),1)
    dur_sec=total_bars*4/avg_bpm*60
    print(f"\n  Secciones:  {len(secciones)}  |  Compases: {total_bars}  |  BPM medio: {avg_bpm:.1f}  |  Duración ~{int(dur_sec//60)}'{int(dur_sec%60):02d}\"")
    all_notes=[]; all_chords=Counter(); all_vels=[]
    print(f"\n  {'ID':5s} {'Nombre':30s} {'cc':5s} {'BPM':16s} {'Frases':7s} {'Notas':6s}")
    print(f"  {'─'*5} {'─'*30} {'─'*5} {'─'*16} {'─'*7} {'─'*6}")
    for sec in secciones:
        n_bars=sec['compases'][1]-sec['compases'][0]+1; t=sec['tempo']
        bpm_str=f"{t['bpm_inicio']}→{t.get('bpm_fin',t['bpm_inicio'])} ({t['tipo'][:5]})"
        frases=sec.get('frases',[]); n_notas=sum(len(f.get('melodia',[])) for f in frases)
        print(f"  {sec['id']:5s} {sec['nombre'][:30]:30s} {n_bars:5d} {bpm_str:16s} {len(frases):7d} {n_notas:6d}")
        for frase in frases:
            for nota in frase.get('melodia',[]):
                if isinstance(nota,list) and len(nota)>=5:
                    try: all_notes.append(note_to_midi(str(nota[3]))%12); all_vels.append(int(nota[4]))
                    except Exception: pass
            for item in frase.get('armonia',[]):
                if item.get('acorde'): all_chords[item['acorde']]+=1
    if all_notes:
        pc_count=Counter(all_notes)
        print(f"\n  DISTRIBUCIÓN DE ALTURAS (pitch class):")
        for pc in range(12):
            cnt=pc_count.get(pc,0)
            bar_str='█'*int(cnt/max(max(pc_count.values()),1)*20)
            print(f"  {NOTE_NAMES[pc]:3s} {bar_str:<20s} {cnt:4d}")
    if all_vels:
        print(f"\n  DINÁMICA: mín={min(all_vels)}  máx={max(all_vels)}  media={sum(all_vels)/len(all_vels):.1f}")
    if all_chords:
        print(f"\n  ACORDES MÁS FRECUENTES:")
        for ac,cnt in all_chords.most_common(10):
            bar_str='█'*int(cnt/all_chords.most_common(1)[0][1]*20)
            print(f"  {ac:8s} {bar_str:<20s} {cnt:4d}x")
    print(f"\n{'═'*60}")

# ══════════════════════════════════════════════════════════════════════════════
#  DIFF
# ══════════════════════════════════════════════════════════════════════════════
def diff_obras(path_a, path_b):
    with open(path_a,'r',encoding='utf-8') as f: obra_a=yaml.safe_load(f)
    with open(path_b,'r',encoding='utf-8') as f: obra_b=yaml.safe_load(f)
    print(f"\n{'═'*60}\n  DIFF: {os.path.basename(path_a)}  →  {os.path.basename(path_b)}\n{'═'*60}")
    def frases_map(o): return {fr['id']:fr for s in o.get('secciones',[]) for fr in s.get('frases',[])}
    def secs_map(o):   return {s['id']:s for s in o.get('secciones',[])}
    secs_a=secs_map(obra_a); secs_b=secs_map(obra_b)
    frases_a=frases_map(obra_a); frases_b=frases_map(obra_b)
    cambios=0
    for sid in sorted(set(secs_a)|set(secs_b)):
        if sid not in secs_a: print(f"\n  [+] Sección {sid} AÑADIDA: {secs_b[sid]['nombre']}"); cambios+=1; continue
        if sid not in secs_b: print(f"\n  [-] Sección {sid} ELIMINADA: {secs_a[sid]['nombre']}"); cambios+=1; continue
        sa,sb=secs_a[sid],secs_b[sid]; diffs=[]
        if sa['compases']!=sb['compases']: diffs.append(f"compases: {sa['compases']} → {sb['compases']}")
        if sa.get('tempo',{})!=sb.get('tempo',{}):
            ta,tb=sa.get('tempo',{}),sb.get('tempo',{})
            diffs.append(f"tempo: {ta.get('bpm_inicio')}→{ta.get('bpm_fin',ta.get('bpm_inicio'))} BPM "
                         f"⟶ {tb.get('bpm_inicio')}→{tb.get('bpm_fin',tb.get('bpm_inicio'))} BPM")
        if diffs:
            print(f"\n  [~] Sección {sid}: {sa['nombre']}")
            for d in diffs: print(f"      {d}")
            cambios+=len(diffs)
    for fid in sorted(set(frases_a)|set(frases_b)):
        if fid not in frases_a: fb=frases_b[fid]; print(f"\n  [+] Frase {fid} AÑADIDA: '{fb['nombre']}' cc.{fb['compases'][0]}–{fb['compases'][1]}"); cambios+=1; continue
        if fid not in frases_b: print(f"\n  [-] Frase {fid} ELIMINADA: '{frases_a[fid]['nombre']}'"); cambios+=1; continue
        fa,fb=frases_a[fid],frases_b[fid]; diffs=[]
        na_mel=len(fa.get('melodia',[])); nb_mel=len(fb.get('melodia',[]))
        if na_mel!=nb_mel: diffs.append(f"melodía: {na_mel} → {nb_mel} notas ({'+' if nb_mel>na_mel else ''}{nb_mel-na_mel})")
        na_arm=len(fa.get('armonia',[])); nb_arm=len(fb.get('armonia',[]))
        if na_arm!=nb_arm: diffs.append(f"armonía: {na_arm} → {nb_arm} acordes")
        arm_a={item['compas']:item.get('acorde') for item in fa.get('armonia',[])}
        arm_b={item['compas']:item.get('acorde') for item in fb.get('armonia',[])}
        for c,ac_a in arm_a.items():
            ac_b=arm_b.get(c)
            if ac_b and ac_b!=ac_a: diffs.append(f"c.{c}: acorde {ac_a} → {ac_b}")
        if fa.get('compases')!=fb.get('compases'): diffs.append(f"compases: {fa['compases']} → {fb['compases']}")
        if diffs:
            print(f"\n  [~] Frase {fid}: '{fa['nombre']}'")
            for d in diffs: print(f"      {d}")
            cambios+=len(diffs)
    print(f"\n  {'✓ Sin diferencias' if cambios==0 else f'Total: {cambios} cambio(s)'}\n{'═'*60}\n")

# ══════════════════════════════════════════════════════════════════════════════
#  PLANTILLA DE NUEVA FRASE
# ══════════════════════════════════════════════════════════════════════════════
def nueva_frase_template(frase_id, compases, leitmotiv=None, yaml_path='partitura.yaml'):
    try: c_ini,c_fin=[int(x) for x in compases.split('-')]
    except Exception: print(f"✗ Formato inválido: '{compases}'. Usa 'ini-fin'"); return
    sec_id=frase_id.split('.')[0]; lm_str=f'"{leitmotiv}"' if leitmotiv else 'null'
    n_bars=c_fin-c_ini+1; notas_bajo=', '.join(['C2']*n_bars)
    arm_lines='\n'.join(f'    - {{compas: {c_ini+i}, acorde: Am, grado: "i", funcion: tonica}}' for i in range(n_bars))
    template=f"""
# ── Frase {frase_id}: [NOMBRE] ────────────────────────────────────────────
- id: "{frase_id}"
  nombre: "[Nombre descriptivo]"
  compases: [{c_ini}, {c_fin}]
  leitmotiv: {lm_str}
  descripcion: >
    [Descripción del carácter y función dramática.]

  armonia:
{arm_lines}

  melodia:
    # [compas, beat, duracion, nota, velocidad]
    # Dict para tremolo/trino:
    #   {{compas: {c_ini}, beat: 0.0, dur: 2.0, nota: A4, vel: 80, tremolo: 16}}
    #   {{compas: {c_ini}, beat: 0.0, dur: 2.0, nota: A4, vel: 80, trino: Bb4}}
    - [{c_ini}, 0.0, 2.0, A4, 72]
    - [{c_ini}, 2.0, 1.8, E4, 68]

  # Transformaciones temáticas (se aplican a toda la capa melodia):
  # transformaciones:
  #   - tipo: transponer
  #     semitonos: 5
  #   - tipo: invertir
  #   - tipo: retrogradar
  #   - tipo: aumentar
  #     factor: 2.0
  #   - tipo: fragmentar
  #     beats_inicio: 0.0
  #     beats_fin: 4.0

  marcato:
    activo: false
    patron_ritmico: cuatro_negras
    intensidad_inicio: 0.35
    intensidad_fin: 0.45
    notas_bajo: [{notas_bajo}]

  # arpegio:
  #   patron: ascendente       # ascendente | descendente | alternado
  #   subdivision: 0.5

  # ostinato:
  #   notas: [C3, E3, G3, E3]
  #   duracion_nota: 0.5
  #   velocidad: 68

  # dinamica:
  #   tipo: crescendo          # crescendo | decrescendo | arco | fijo
  #   vel_ini: 48
  #   vel_fin: 96

  # pitch_bend:
  #   - {{compas: {c_ini}, beat: 0.0, tipo: vibrato, amplitud: 0.3, velocidad: 5.0, beats: 2.0}}
  #   - {{compas: {c_ini}, beat: 2.0, tipo: portamento, desde: E4, hasta: F4, beats: 0.5}}

  # control:
  #   - {{compas: {c_ini}, beat: 0.0, cc: 64, valor: 127}}
  #   - {{compas: {c_fin}, beat: 3.5, cc: 64, valor: 0}}
  #   - {{compas: {c_ini}, beat: 0.0, cc: 11, curva: crescendo, beats: {n_bars*4:.1f}, valor_ini: 40, valor_fin: 100}}

  pad: []

  dinamica:
    inicio: mp
    fin: mf
    tipo: crescendo_gradual
"""
    print(f"\n{'═'*60}\n  PLANTILLA — Frase {frase_id}  (insertar en Sección {sec_id})\n{'═'*60}")
    print(template)
    out=f"plantilla_{frase_id.replace('.','_')}.yaml"
    with open(out,'w',encoding='utf-8') as f: f.write(template)
    print(f"  → Guardado: {out}")

# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN CSV
# ══════════════════════════════════════════════════════════════════════════════
def export_csv(obra, output_dir):
    import csv
    rows=[]
    for sec in obra.get('secciones',[]):
        sid=sec['id']
        for frase in sec.get('frases',[]):
            fid=frase['id']
            for nota in frase.get('melodia',[]):
                if isinstance(nota,dict):
                    row={'seccion':sid,'frase':fid,'instrumento':'melodia',
                         'compas':nota.get('compas','?'),'beat':nota.get('beat','?'),
                         'duracion':nota.get('dur','?'),'nota':nota.get('nota','?'),
                         'nota_midi':'','velocidad':nota.get('vel','?'),
                         'leitmotiv':frase.get('leitmotiv','')}
                    try: row['nota_midi']=note_to_midi(str(nota.get('nota','')))
                    except Exception: pass
                elif len(nota)>=5:
                    bar,beat,dur,nn,vel=nota
                    row={'seccion':sid,'frase':fid,'instrumento':'melodia',
                         'compas':bar,'beat':beat,'duracion':dur,'nota':nn,'nota_midi':'',
                         'velocidad':vel,'leitmotiv':frase.get('leitmotiv','')}
                    try: row['nota_midi']=note_to_midi(str(nn))
                    except Exception: pass
                else:
                    continue
                rows.append(row)
    if not rows: print("  ⚠ No hay datos de melodía para exportar"); return
    out_path=os.path.join(output_dir,'analisis_notas.csv')
    campos=['seccion','frase','instrumento','compas','beat','duracion','nota','nota_midi','velocidad','leitmotiv']
    with open(out_path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=campos); w.writeheader(); w.writerows(rows)
    print(f"  → CSV exportado: {out_path}  ({len(rows)} notas)")

# ══════════════════════════════════════════════════════════════════════════════
#  RENDER DE AUDIO
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_SF=['/usr/share/sounds/sf2/FluidR3_GM.sf2','/usr/share/soundfonts/FluidR3_GM.sf2',
            '/usr/share/sounds/sf2/default.sf2']

def render_audio(midi_paths, soundfont=None):
    has_fluid=shutil.which('fluidsynth') is not None
    has_timi =shutil.which('timidity')   is not None
    if not has_fluid and not has_timi:
        print("\n  ⚠ No se encontró FluidSynth ni TiMidity.")
        print("    Ubuntu/Debian:  sudo apt install fluidsynth fluid-soundfont-gm")
        print("    macOS:          brew install fluidsynth"); return
    sf=soundfont
    if has_fluid and not sf:
        for p in DEFAULT_SF:
            if os.path.exists(p): sf=p; break
    rendered=[]
    for midi_path in midi_paths:
        wav_path=midi_path.replace('.mid','.wav')
        try:
            if has_fluid and sf: cmd=['fluidsynth','-ni',sf,midi_path,'-F',wav_path,'-r','44100']
            else:                 cmd=['timidity',midi_path,'-Ow','-o',wav_path]
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            if result.returncode==0 and os.path.exists(wav_path):
                print(f"  ✓ {os.path.basename(wav_path)}  ({os.path.getsize(wav_path)//1024} KB)")
                rendered.append(wav_path)
            else:
                print(f"  ✗ Error al renderizar {os.path.basename(midi_path)}")
                if result.stderr: print(f"    {result.stderr[:200]}")
        except Exception as e: print(f"  ✗ {e}")
    if rendered: print(f"\n  ✓ {len(rendered)} fichero(s) WAV generado(s)")

# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN HTML — ESQUEMA VISUAL DE LA OBRA
# ══════════════════════════════════════════════════════════════════════════════
def export_html(obra: dict, output_dir: str):
    """Genera un fichero HTML con un esquema visual completo de la obra.

    Incluye:
    - Cabecera con título, tonalidad, compás, duración estimada
    - Línea de tiempo proporcional de secciones con BPM y tempo
    - Por sección: frases como bloques con altura proporcional a la densidad
      de notas, acordes, leitmotiv, marcadores de transformaciones/arpegio/etc.
    - Panel de leitmotivs con sus notas canónicas y color asignado
    - Gráfico de distribución de pitch class
    - Gráfico de acordes más frecuentes
    - Tabla de tracks/instrumentación
    """
    import json

    # ── datos calculados ──────────────────────────────────────────────────────
    secciones   = obra.get('secciones', [])
    leitmotivs  = obra.get('leitmotivs', [])
    tracks      = obra.get('tracks', DEFAULT_TRACKS)
    obra_meta   = obra.get('obra', {})

    total_bars  = sum(s['compases'][1] - s['compases'][0] + 1 for s in secciones)
    avg_bpm     = sum(
        (s['tempo']['bpm_inicio'] + s['tempo'].get('bpm_fin', s['tempo']['bpm_inicio'])) / 2
        for s in secciones
    ) / max(len(secciones), 1)
    dur_sec     = total_bars * 4 / avg_bpm * 60

    # Estadísticas globales
    # Claves de instrumento que pueden contener notas melódicas en una frase
    _INSTR_KEYS = {'viola','violin','melodia','cello','contrabajo','flauta',
                   'oboe','clarinete','fagot','trompa','trompeta','trombon',
                   'arpa','piano','maderas','metales','cuerdas_seccion'}
    # Campos que NO son notas en una frase (evitar confusión)
    _NON_NOTE_KEYS = {'id','nombre','compases','leitmotiv','articulacion',
                      'descripcion','dinamica','armonia','marcato','control',
                      'ostinato','arpegio','pitch_bend','transformaciones',
                      'origen_leitmotiv','pad','percusion','tonalidad','compas',
                      'tempo','duracion_aprox','instrumentacion'}

    def _extract_note_vel(entrada):
        """Extrae (nota_str, velocidad) de una entrada que puede ser dict o lista."""
        if isinstance(entrada, dict):
            nota = entrada.get('nota') or entrada.get('note')
            vel  = entrada.get('vel') or entrada.get('velocidad') or entrada.get('velocity')
            if nota and vel is not None:
                return str(nota), int(vel)
        elif isinstance(entrada, list) and len(entrada) >= 5:
            return str(entrada[3]), int(entrada[4])
        return None, None

    all_notes = []; all_chords = Counter(); all_vels = []
    for sec in secciones:
        for frase in sec.get('frases', []):
            # Buscar notas en claves de instrumento conocidas o en cualquier
            # clave que no sea metadata conocida
            for key, val in frase.items():
                if key in _NON_NOTE_KEYS:
                    continue
                if not isinstance(val, list):
                    continue
                for entrada in val:
                    nota_str, vel = _extract_note_vel(entrada)
                    if nota_str and vel is not None:
                        try:
                            all_notes.append(note_to_midi(nota_str) % 12)
                            all_vels.append(vel)
                        except Exception:
                            pass
            for item in frase.get('armonia', []):
                if isinstance(item, dict) and item.get('acorde'):
                    all_chords[item['acorde']] += 1

    pc_count    = Counter(all_notes)
    pc_max      = max(pc_count.values()) if pc_count else 1
    chord_top   = all_chords.most_common(12)
    chord_max   = chord_top[0][1] if chord_top else 1

    # Paleta de colores para leitmotivs
    LM_COLORS = [
        '#e8925a','#5ab4e8','#a8e85a','#e85ab4',
        '#e8d45a','#5ae8c8','#c85ae8','#e8725a',
    ]
    lm_color = {lm['id']: LM_COLORS[i % len(LM_COLORS)]
                for i, lm in enumerate(leitmotivs)}

    # Datos JSON para el JS del HTML
    sections_data = []
    for sec in secciones:
        c_ini, c_fin = sec['compases']
        n_bars = c_fin - c_ini + 1
        t = sec['tempo']
        frases_data = []
        for frase in sec.get('frases', []):
            # Contar notas en todas las claves de instrumento (no solo 'melodia')
            n_mel = 0
            for key, val in frase.items():
                if key in _NON_NOTE_KEYS:
                    continue
                if isinstance(val, list):
                    for entrada in val:
                        nota_str, _ = _extract_note_vel(entrada)
                        if nota_str:
                            n_mel += 1
            n_arm  = len(frase.get('armonia', []))
            badges = []
            if frase.get('transformaciones'):
                badges.append('⟳ transf.')
            if frase.get('arpegio'):
                badges.append('♜ arpegio')
            if frase.get('ostinato'):
                badges.append('∞ ostinato')
            if frase.get('pitch_bend'):
                badges.append('〜 bend')
            if frase.get('control'):
                badges.append('CC')
            marc = frase.get('marcato', {})
            if marc and marc.get('activo'):
                badges.append(f"♩ {marc.get('patron_ritmico','?')}")
            # Acordes de la frase
            acordes = [item.get('acorde','') for item in frase.get('armonia', [])
                       if item.get('acorde')]
            frases_data.append({
                'id':      frase.get('id',''),
                'nombre':  frase.get('nombre',''),
                'cc_ini':  frase['compases'][0],
                'cc_fin':  frase['compases'][1],
                'n_bars':  frase['compases'][1] - frase['compases'][0] + 1,
                'n_mel':   n_mel,
                'n_arm':   n_arm,
                'lm':      frase.get('leitmotiv') or '',
                'lm_color':lm_color.get(frase.get('leitmotiv',''), '#666'),
                'badges':  badges,
                'acordes': acordes,
                'desc':    (frase.get('descripcion') or '').strip(),
                'din_ini': (frase.get('dinamica') or {}).get('vel_ini',''),
                'din_fin': (frase.get('dinamica') or {}).get('vel_fin',''),
                'din_tipo':(frase.get('dinamica') or {}).get('tipo',''),
            })
        sections_data.append({
            'id':       sec.get('id',''),
            'nombre':   sec.get('nombre',''),
            'cc_ini':   c_ini,
            'cc_fin':   c_fin,
            'n_bars':   n_bars,
            'pct':      n_bars / max(total_bars, 1) * 100,
            'bpm_ini':  t.get('bpm_inicio', 120),
            'bpm_fin':  t.get('bpm_fin', t.get('bpm_inicio', 120)),
            'tipo':     t.get('tipo', 'fijo'),
            'dur':      sec.get('duracion_aprox', ''),
            'tonalidad':sec.get('tonalidad', obra_meta.get('tonalidad', '')),
            'frases':   frases_data,
            'desc':     (sec.get('descripcion') or '').strip(),
        })

    lm_data = [{
        'id':    lm.get('id',''),
        'nombre':lm.get('nombre',''),
        'notas': lm.get('notas_canonicas') or [],
        'desc':  (lm.get('descripcion') or '').strip(),
        'color': lm_color.get(lm.get('id',''), '#666'),
    } for lm in leitmotivs]

    tracks_data = [{'nombre': t.get('nombre', t.get('id','')),
                    'canal':  t.get('canal', 0),
                    'prog':   t.get('programa', 0),
                    'capa':   t.get('capa', '')} for t in tracks]

    pc_data  = [{'pc': NOTE_NAMES[i], 'n': pc_count.get(i, 0)} for i in range(12)]
    vel_data = {'min': min(all_vels) if all_vels else 0,
                'max': max(all_vels) if all_vels else 0,
                'avg': round(sum(all_vels)/len(all_vels), 1) if all_vels else 0}

    # ── HTML ──────────────────────────────────────────────────────────────────
    titulo    = obra_meta.get('titulo', 'Sin título')
    subtitulo = obra_meta.get('subtitulo', '')
    tonalidad = obra_meta.get('tonalidad', '?')
    compas    = obra_meta.get('compas', '?')
    mins      = int(dur_sec // 60); secs_r = int(dur_sec % 60)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{titulo}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IM+Fell+English:ital@0;1&family=Inconsolata:wght@300;400;600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
:root {{
  --ink:      #0f0c08;
  --paper:    #f5f0e8;
  --paper2:   #ede7d6;
  --rule:     #c8b99a;
  --rule2:    #a89070;
  --accent:   #8b1a1a;
  --accent2:  #2a4a6b;
  --dim:      #4a3e2e;
  --staff:    #d4c5a9;
  --shadow:   rgba(0,0,0,0.15);
}}
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background: var(--paper);
  color: var(--ink);
  font-family: 'Inconsolata', monospace;
  font-size: 15px;
  line-height: 1.6;
  min-height: 100vh;
}}

/* ── Cabecera ─────────────────────────────────────────────────── */
header {{
  background: var(--ink);
  color: var(--paper);
  padding: 48px 60px 36px;
  position: relative;
  overflow: hidden;
}}
header::before {{
  content: '';
  position: absolute; inset: 0;
  background-image:
    repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(255,255,255,.04) 60px);
  pointer-events: none;
}}
.header-inner {{ position: relative; z-index: 1; max-width: 1400px; margin: 0 auto; }}
h1 {{
  font-family: 'IM Fell English', serif;
  font-size: clamp(2.2rem, 5vw, 3.8rem);
  letter-spacing: .01em;
  line-height: 1.1;
  font-weight: 400;
}}
.subtitulo {{
  font-family: 'IM Fell English', serif;
  font-style: italic;
  color: var(--rule);
  font-size: 1.15rem;
  margin-top: 6px;
}}
.meta-pills {{
  display: flex; flex-wrap: wrap; gap: 10px;
  margin-top: 22px;
}}
.pill {{
  border: 1px solid rgba(255,255,255,.25);
  padding: 4px 14px;
  font-size: 13px;
  letter-spacing: .10em;
  text-transform: uppercase;
  color: rgba(255,255,255,.85);
}}
.pill strong {{ color: #fff; }}

/* ── Contenido ────────────────────────────────────────────────── */
.page {{ max-width: 1400px; margin: 0 auto; padding: 0 48px 80px; }}

.section-title {{
  font-family: 'Libre Baskerville', serif;
  font-size: .8rem;
  letter-spacing: .20em;
  text-transform: uppercase;
  color: var(--dim);
  margin: 48px 0 16px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--rule);
}}

/* ── Línea de tiempo de secciones ─────────────────────────────── */
.timeline {{
  display: flex;
  gap: 3px;
  height: 56px;
  margin-bottom: 6px;
  align-items: stretch;
}}
.tl-seg {{
  display: flex; flex-direction: column;
  justify-content: center; align-items: center;
  background: var(--accent2);
  color: #fff;
  font-size: 12px;
  letter-spacing: .08em;
  text-align: center;
  padding: 4px 6px;
  position: relative;
  cursor: pointer;
  transition: filter .15s;
  overflow: hidden;
  min-width: 0;
}}
.tl-seg:hover {{ filter: brightness(1.25); }}
.tl-seg::after {{
  content: '';
  position: absolute; inset: 0;
  pointer-events: none;
}}
.tl-seg .tl-id   {{ font-weight: 600; font-size: 15px; }}
.tl-seg .tl-cc   {{ opacity: .8; font-size: 11px; }}
.tl-ruler {{
  display: flex; gap: 3px;
  margin-bottom: 32px;
}}
.tl-ruler-seg {{
  text-align: center;
  font-size: 11px;
  color: var(--dim);
  letter-spacing: .05em;
}}

/* ── Bloques de sección ───────────────────────────────────────── */
.sec-block {{
  margin-bottom: 52px;
  border-left: 3px solid var(--accent2);
  padding-left: 20px;
  animation: fadeUp .4s ease both;
}}
@keyframes fadeUp {{
  from {{ opacity:0; transform:translateY(12px); }}
  to   {{ opacity:1; transform:translateY(0); }}
}}
.sec-header {{
  display: flex; align-items: baseline; gap: 18px;
  margin-bottom: 14px;
}}
.sec-id {{
  font-family: 'IM Fell English', serif;
  font-size: 2rem;
  color: var(--accent2);
  line-height: 1;
  min-width: 2.5rem;
}}
.sec-nombre {{
  font-family: 'Libre Baskerville', serif;
  font-size: 1.05rem;
  font-weight: 700;
}}
.sec-badges {{
  display: flex; gap: 8px; flex-wrap: wrap; margin-left: auto;
}}
.tempo-badge {{
  font-size: 12px;
  color: var(--dim);
  border: 1px solid var(--rule);
  padding: 2px 8px;
  white-space: nowrap;
}}
.sec-desc {{
  font-family: 'Inconsolata', monospace;
  font-style: normal;
  color: var(--ink);
  font-size: 1.0rem;
  margin-bottom: 18px;
  max-width: 100%;
  line-height: 1.7;
}}

/* ── Frases ───────────────────────────────────────────────────── */
.frases-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: flex-start;
  padding-bottom: 4px;
}}
.frase-block {{
  background: var(--paper2);
  border: 1px solid var(--rule);
  border-top: 3px solid var(--rule2);
  padding: 10px 12px 8px;
  position: relative;
  min-width: 140px;
}}
.frase-block.has-lm {{ border-top-width: 4px; }}
.frase-id {{
  font-size: 11px;
  letter-spacing: .10em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: 4px;
}}
.frase-nombre {{
  font-family: 'Libre Baskerville', serif;
  font-size: .88rem;
  font-weight: 700;
  line-height: 1.3;
  margin-bottom: 6px;
}}
.frase-cc {{
  font-size: 12px;
  color: var(--dim);
  margin-bottom: 6px;
}}
.frase-stats {{
  display: flex; gap: 6px;
  font-size: 12px;
  color: var(--dim);
  margin-bottom: 6px;
}}
.stat-n {{
  background: var(--rule);
  color: var(--ink);
  padding: 1px 6px;
  font-weight: 600;
}}
/* Barra de densidad de notas */
.density-bar {{
  height: 3px;
  background: var(--rule);
  margin-bottom: 6px;
  position: relative;
}}
.density-fill {{
  position: absolute; top:0; left:0; bottom:0;
  background: var(--accent2);
  transition: width .4s ease;
}}
/* Acordes en miniatura */
.acordes-row {{
  display: flex; flex-wrap: wrap; gap: 3px;
  margin-bottom: 6px;
}}
.acorde-chip {{
  font-size: 11px;
  border: 1px solid var(--rule2);
  padding: 1px 5px;
  color: var(--accent);
  font-family: 'Libre Baskerville', serif;
  font-weight: 700;
  white-space: nowrap;
}}
/* Badges de funcionalidades */
.badge-row {{
  display: flex; flex-wrap: wrap; gap: 3px;
}}
.badge {{
  font-size: 10px;
  letter-spacing: .05em;
  padding: 1px 6px;
  background: var(--ink);
  color: var(--paper);
  opacity: .75;
  white-space: nowrap;
}}
/* Leitmotiv dot */
.lm-dot {{
  display: inline-block;
  width: 9px; height: 9px;
  border-radius: 50%;
  margin-right: 4px;
  vertical-align: middle;
}}
.lm-label {{
  font-size: 11px;
  letter-spacing: .07em;
  text-transform: uppercase;
}}

/* (tooltip eliminado) */

/* ── Panel de análisis ────────────────────────────────────────── */
.analysis-grid {{
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 24px;
  margin-top: 8px;
}}
@media (max-width: 800px) {{
  .analysis-grid {{ grid-template-columns: 1fr; }}
  .page {{ padding: 0 20px 60px; }}
  header {{ padding: 32px 20px 28px; }}
}}
.analysis-card {{
  background: var(--paper2);
  border: 1px solid var(--rule);
  padding: 20px;
}}
.card-title {{
  font-size: .75rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--dim);
  margin-bottom: 14px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--rule);
}}

/* Pitch class chart */
.pc-chart {{ display: flex; flex-direction: column; gap: 5px; }}
.pc-row {{ display: flex; align-items: center; gap: 8px; font-size: 12px; }}
.pc-name {{ width: 26px; font-weight: 600; color: var(--ink); }}
.pc-bar-wrap {{ flex: 1; height: 14px; background: var(--rule); position: relative; }}
.pc-bar-fill {{ position: absolute; top:0; left:0; bottom:0; background: var(--accent2); }}
.pc-val {{ width: 28px; text-align: right; color: var(--dim); }}

/* Chord chart */
.chord-chart {{ display: flex; flex-direction: column; gap: 6px; }}
.chord-row {{ display: flex; align-items: center; gap: 8px; font-size: 12px; }}
.chord-name {{
  width: 50px;
  font-family: 'Libre Baskerville', serif;
  font-weight: 700;
  font-size: 13px;
  color: var(--accent);
}}
.chord-bar-wrap {{ flex: 1; height: 14px; background: var(--rule); position: relative; }}
.chord-bar-fill {{ position: absolute; top:0; left:0; bottom:0; background: var(--accent); }}
.chord-val {{ width: 28px; text-align: right; color: var(--dim); }}

/* Leitmotivs */
.lm-list {{ display: flex; flex-direction: column; gap: 12px; }}
.lm-item {{ display: flex; gap: 10px; align-items: flex-start; }}
.lm-swatch {{
  width: 4px; min-height: 40px;
  flex-shrink: 0;
  margin-top: 2px;
}}
.lm-info .lm-id {{ font-weight: 600; font-size: 12px; letter-spacing: .08em; color: var(--ink); }}
.lm-info .lm-nombre {{
  font-family: 'Libre Baskerville', serif;
  font-size: .92rem;
  font-weight: 700;
}}
.lm-info .lm-notas {{
  font-size: 12px; color: var(--dim);
  letter-spacing: .05em;
  margin: 2px 0;
}}
.lm-info .lm-desc {{
  font-family: 'Inconsolata', monospace;
  font-style: normal;
  font-size: .85rem;
  color: var(--dim);
  line-height: 1.5;
}}

/* Dinámica */
.din-box {{ display: flex; flex-direction: column; gap: 10px; }}
.din-stat {{ display: flex; justify-content: space-between; font-size: 13px; }}
.din-label {{ color: var(--dim); }}
.din-val {{ font-weight: 600; color: var(--ink); }}
.din-bar-wrap {{ height: 18px; background: var(--rule); position: relative; margin-top: 8px; }}
.din-bar-min, .din-bar-max, .din-bar-avg {{
  position: absolute; top:0; bottom:0; width: 2px;
}}
.din-bar-min {{ background: var(--accent2); }}
.din-bar-max {{ background: var(--accent); }}
.din-bar-avg {{ background: var(--ink); }}

/* Tracks */
.tracks-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.tracks-table th {{
  text-align: left; padding: 5px 8px;
  font-size: 11px; letter-spacing: .10em; text-transform: uppercase;
  color: var(--dim); border-bottom: 1px solid var(--rule);
}}
.tracks-table td {{
  padding: 6px 8px;
  border-bottom: 1px solid var(--staff);
  vertical-align: top;
}}
.tracks-table tr:last-child td {{ border-bottom: none; }}
.capa-badge {{
  display: inline-block;
  background: var(--accent2);
  color: #fff;
  font-size: 8px;
  padding: 1px 5px;
  letter-spacing: .06em;
}}

/* Patrón BPM visual */
.bpm-viz {{
  display: flex; align-items: center; gap: 6px;
  font-size: 10px; color: var(--dim);
}}
.bpm-arrow {{ color: var(--accent); }}

footer {{
  text-align: center;
  padding: 24px;
  font-size: 12px;
  color: var(--dim);
  letter-spacing: .08em;
  border-top: 1px solid var(--rule);
  margin-top: 40px;
}}
</style>
</head>
<body>

<!-- ════ CABECERA ════════════════════════════════════════════════════════ -->
<header>
  <div class="header-inner">
    <h1>{titulo}</h1>
    {'<p class="subtitulo">' + subtitulo + '</p>' if subtitulo else ''}
    <div class="meta-pills">
      <span class="pill">Tonalidad <strong>{tonalidad}</strong></span>
      <span class="pill">Compás <strong>{compas}</strong></span>
      <span class="pill">Compases <strong>{total_bars}</strong></span>
      <span class="pill">Secciones <strong>{len(secciones)}</strong></span>
      <span class="pill">BPM medio <strong>{avg_bpm:.0f}</strong></span>
      <span class="pill">Duración <strong>~{mins}'{secs_r:02d}"</strong></span>
      <span class="pill">Tracks <strong>{len(tracks)}</strong></span>
    </div>
  </div>
</header>

<div class="page">

<!-- ════ LÍNEA DE TIEMPO ═════════════════════════════════════════════════ -->
<p class="section-title">Estructura global</p>
<div class="timeline" id="timeline"></div>
<div class="tl-ruler" id="tl-ruler"></div>

<!-- ════ SECCIONES ══════════════════════════════════════════════════════ -->
<p class="section-title">Secciones y frases</p>
<div id="secciones-container"></div>

<!-- ════ ANÁLISIS ═══════════════════════════════════════════════════════ -->
<p class="section-title">Análisis</p>
<div class="analysis-grid">
  <div class="analysis-card">
    <p class="card-title">Distribución de alturas</p>
    <div class="pc-chart" id="pc-chart"></div>
  </div>
  <div class="analysis-card">
    <p class="card-title">Acordes más frecuentes</p>
    <div class="chord-chart" id="chord-chart"></div>
  </div>
  <div class="analysis-card">
    <p class="card-title">Leitmotivs</p>
    <div class="lm-list" id="lm-list"></div>
  </div>
  <div class="analysis-card">
    <p class="card-title">Dinámica</p>
    <div class="din-box" id="din-box"></div>
  </div>
  <div class="analysis-card" style="grid-column: span 2;">
    <p class="card-title">Instrumentación</p>
    <table class="tracks-table" id="tracks-table">
      <thead><tr>
        <th>Instrumento</th><th>Canal</th><th>Programa GM</th><th>Capa</th>
      </tr></thead>
      <tbody id="tracks-tbody"></tbody>
    </table>
  </div>
</div>

</div><!-- /page -->

<footer>
  Generado por generar_midi.py &nbsp;·&nbsp; {titulo}
</footer>

<script>
const SECTIONS = {json.dumps(sections_data, ensure_ascii=False)};
const PC_DATA  = {json.dumps(pc_data,       ensure_ascii=False)};
const PC_MAX   = {pc_max};
const CHORD_TOP= {json.dumps(chord_top,     ensure_ascii=False)};
const CHORD_MAX= {chord_max};
const LM_DATA  = {json.dumps(lm_data,       ensure_ascii=False)};
const VEL_DATA = {json.dumps(vel_data,      ensure_ascii=False)};
const TRACKS   = {json.dumps(tracks_data,   ensure_ascii=False)};
const TOTAL_BARS = {total_bars};

// ── Utilidades ──────────────────────────────────────────────────────────
function el(tag, cls, html) {{
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}}
function pct(n, max) {{ return Math.round(n / Math.max(max,1) * 100); }}

// ── Línea de tiempo ──────────────────────────────────────────────────────
const tl = document.getElementById('timeline');
const ruler = document.getElementById('tl-ruler');
const SEC_COLORS = ['#2a4a6b','#3a6b4a','#6b2a2a','#4a3a6b','#6b5a2a','#2a6b6b'];

SECTIONS.forEach((sec, i) => {{
  const seg = el('div','tl-seg');
  seg.style.flex = sec.pct;
  seg.style.background = SEC_COLORS[i % SEC_COLORS.length];
  seg.innerHTML = `<span class="tl-id">${{sec.id}}</span><span class="tl-cc">cc.${{sec.cc_ini}}–${{sec.cc_fin}}</span>`;
  seg.title = sec.nombre;
  seg.onclick = () => document.getElementById('sec-'+sec.id)?.scrollIntoView({{behavior:'smooth',block:'start'}});
  tl.appendChild(seg);

  const rg = el('div','tl-ruler-seg');
  rg.style.flex = sec.pct;
  const bpmStr = sec.bpm_ini === sec.bpm_fin
    ? `${{sec.bpm_ini}} BPM`
    : `${{sec.bpm_ini}}→${{sec.bpm_fin}} BPM`;
  rg.textContent = bpmStr;
  ruler.appendChild(rg);
}});

// ── Secciones ────────────────────────────────────────────────────────────
const container = document.getElementById('secciones-container');
const MAX_NOTAS_FRASE = Math.max(1, ...SECTIONS.flatMap(s => s.frases.map(f => f.n_mel)));

SECTIONS.forEach((sec, si) => {{
  const block = el('div','sec-block');
  block.id = 'sec-' + sec.id;
  block.style.animationDelay = (si * 0.08) + 's';

  // Cabecera
  const tipoLabel = {{fijo:'fijo',accelerando:'accel.',rallentando:'rall.',rubato:'rubato'}}[sec.tipo] || sec.tipo;
  const bpmStr = sec.bpm_ini === sec.bpm_fin
    ? `${{sec.bpm_ini}} BPM (${{tipoLabel}})`
    : `${{sec.bpm_ini}} → ${{sec.bpm_fin}} BPM (${{tipoLabel}})`;
  const tonStr = sec.tonalidad ? `${{sec.tonalidad}}` : '';

  block.innerHTML = `
    <div class="sec-header">
      <span class="sec-id">${{sec.id}}</span>
      <span class="sec-nombre">${{sec.nombre}}</span>
      <div class="sec-badges">
        <span class="tempo-badge">♩ ${{bpmStr}}</span>
        ${{tonStr ? `<span class="tempo-badge">${{tonStr}}</span>` : ''}}
        <span class="tempo-badge">cc.${{sec.cc_ini}}–${{sec.cc_fin}} (${{sec.n_bars}} cc.)</span>
        ${{sec.dur ? `<span class="tempo-badge">~${{sec.dur}}</span>` : ''}}
      </div>
    </div>
    ${{sec.desc ? `<p class="sec-desc">${{sec.desc.replace(/\\n/g,' ')}}</p>` : ''}}
  `;

  // Frases
  const row = el('div','frases-row');
  sec.frases.forEach(frase => {{
    const fb = el('div', 'frase-block' + (frase.lm ? ' has-lm' : ''));
    if (frase.lm) fb.style.borderTopColor = frase.lm_color;

    // Ancho proporcional a compases
    fb.style.width = Math.max(100, frase.n_bars * 54) + 'px';

    const densityPct = pct(frase.n_mel, MAX_NOTAS_FRASE);
    const acordesHtml = frase.acordes.slice(0,8).map(a =>
      `<span class="acorde-chip">${{a}}</span>`
    ).join('') + (frase.acordes.length > 8 ? `<span class="acorde-chip">+${{frase.acordes.length-8}}</span>` : '');

    const badgesHtml = frase.badges.map(b => `<span class="badge">${{b}}</span>`).join('');

    const lmHtml = frase.lm
      ? `<div style="margin-bottom:5px"><span class="lm-dot" style="background:${{frase.lm_color}}"></span><span class="lm-label">${{frase.lm}}</span></div>`
      : '';

    const dinIni = frase.din_ini !== undefined && frase.din_ini !== '' ? frase.din_ini : null;
    const dinFin = frase.din_fin !== undefined && frase.din_fin !== '' ? frase.din_fin : null;
    const dinTipo = frase.din_tipo || '';
    const dinSame = dinIni !== null && dinFin !== null && String(dinIni) === String(dinFin);
    const dinHtml = (dinTipo && dinTipo !== 'fijo')
      ? (dinSame
          ? `<div style="font-size:12px;color:var(--dim);margin-top:4px">din: ${{dinIni}} <em>(${{dinTipo}})</em></div>`
          : `<div style="font-size:12px;color:var(--dim);margin-top:4px">din: ${{dinIni !== null ? dinIni : '?'}}→${{dinFin !== null ? dinFin : '?'}} <em>(${{dinTipo}})</em></div>`)
      : '';

    fb.innerHTML = `
      <div class="frase-id">${{frase.id}}</div>
      <div class="frase-nombre">${{frase.nombre}}</div>
      <div class="frase-cc">cc.${{frase.cc_ini}}–${{frase.cc_fin}}</div>
      <div class="frase-stats">
        <span class="stat-n">${{frase.n_mel}}♩</span>
        <span class="stat-n">${{frase.n_arm}}⬛</span>
      </div>
      <div class="density-bar"><div class="density-fill" style="width:${{densityPct}}%"></div></div>
      ${{lmHtml}}
      <div class="acordes-row">${{acordesHtml}}</div>
      <div class="badge-row">${{badgesHtml}}</div>
      ${{dinHtml}}
    `;
    row.appendChild(fb);
  }});

  block.appendChild(row);
  container.appendChild(block);
}});

// ── Pitch class chart ─────────────────────────────────────────────────────
const pcChart = document.getElementById('pc-chart');
PC_DATA.forEach(d => {{
  const w = pct(d.n, PC_MAX);
  const row = el('div','pc-row');
  row.innerHTML = `
    <span class="pc-name">${{d.pc}}</span>
    <div class="pc-bar-wrap"><div class="pc-bar-fill" style="width:${{w}}%"></div></div>
    <span class="pc-val">${{d.n}}</span>
  `;
  pcChart.appendChild(row);
}});

// ── Chord chart ───────────────────────────────────────────────────────────
const chordChart = document.getElementById('chord-chart');
CHORD_TOP.forEach(([ac, cnt]) => {{
  const w = pct(cnt, CHORD_MAX);
  const row = el('div','chord-row');
  row.innerHTML = `
    <span class="chord-name">${{ac}}</span>
    <div class="chord-bar-wrap"><div class="chord-bar-fill" style="width:${{w}}%"></div></div>
    <span class="chord-val">${{cnt}}×</span>
  `;
  chordChart.appendChild(row);
}});

// ── Leitmotivs ────────────────────────────────────────────────────────────
const lmList = document.getElementById('lm-list');
LM_DATA.forEach(lm => {{
  const item = el('div','lm-item');
  const notasStr = Array.isArray(lm.notas) ? lm.notas.join(' – ') : '—';
  item.innerHTML = `
    <div class="lm-swatch" style="background:${{lm.color}}"></div>
    <div class="lm-info">
      <div class="lm-id">${{lm.id}}</div>
      <div class="lm-nombre">${{lm.nombre}}</div>
      <div class="lm-notas">${{notasStr}}</div>
      ${{lm.desc ? `<div class="lm-desc">${{lm.desc}}</div>` : ''}}
    </div>
  `;
  lmList.appendChild(item);
}});
if (LM_DATA.length === 0) lmList.innerHTML = '<p style="color:var(--dim);font-size:.8rem">No hay leitmotivs definidos.</p>';

// ── Dinámica ──────────────────────────────────────────────────────────────
const dinBox = document.getElementById('din-box');
const {{min: vMin, max: vMax, avg: vAvg}} = VEL_DATA;
dinBox.innerHTML = `
  <div class="din-stat"><span class="din-label">Velocidad mínima</span><span class="din-val">${{vMin}} <em style="color:var(--dim);font-weight:normal;font-size:.85em">(pp)</em></span></div>
  <div class="din-stat"><span class="din-label">Velocidad máxima</span><span class="din-val">${{vMax}} <em style="color:var(--dim);font-weight:normal;font-size:.85em">(ff+)</em></span></div>
  <div class="din-stat"><span class="din-label">Media</span><span class="din-val">${{vAvg}}</span></div>
  <div class="din-bar-wrap">
    <div class="din-bar-min"  style="left:${{pct(vMin,127)}}%"></div>
    <div class="din-bar-avg"  style="left:${{pct(vAvg,127)}}%"></div>
    <div class="din-bar-max"  style="left:${{pct(vMax,127)}}%"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--dim);margin-top:4px">
    <span>ppp&nbsp;0</span><span>mp&nbsp;64</span><span>fff&nbsp;127</span>
  </div>
`;

// ── Tracks ────────────────────────────────────────────────────────────────
const tbody = document.getElementById('tracks-tbody');
TRACKS.forEach(t => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><strong>${{t.nombre}}</strong></td>
    <td>${{t.canal}}</td>
    <td>${{t.prog}}</td>
    <td><span class="capa-badge">${{t.capa}}</span></td>
  `;
  tbody.appendChild(tr);
}});
</script>
</body>
</html>"""

    safe = _safe_filename(titulo)
    out_path = os.path.join(output_dir, f"{safe}_esquema.html")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  → HTML generado: {out_path}")
    return out_path

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE PATRONES RÍTMICOS DEFINIDOS EN EL YAML
# ══════════════════════════════════════════════════════════════════════════════
def _load_patrones(obra: dict) -> dict:
    """Lee la sección 'patrones_ritmicos:' del YAML y la devuelve como dict.

    Los patrones del YAML se AÑADEN a MARCATO_PATTERNS. Si un nombre coincide
    con uno del script, el del YAML tiene prioridad (permite redefinir).

    Formato YAML:
        patrones_ritmicos:
          compas_5_4:                        # nombre libre, usable en patron_ritmico:
            - [0.0, 0.45, 1.0]              # [beat, duracion_beats, factor_volumen]
            - [1.0, 0.45, 0.80]
            - [2.0, 0.45, 0.90]
            - [3.0, 0.45, 0.70]
            - [4.0, 0.45, 0.75]
          clave_son:
            - [0.0, 0.45, 1.00]
            - [0.75,0.45, 0.70]
            - [1.5, 0.45, 0.85]
            - [2.5, 0.45, 0.80]
            - [3.5, 0.45, 0.75]

    Cada entrada es una lista [beat, duración, factor_volumen]:
        beat          posición dentro del compás (0.0–N, sin límite de compás)
        duración      en beats (afectada por staccato ×0.86 en process_marcato)
        factor_volumen escalar 0.0–1.0 relativo a la intensidad de la frase

    Los patrones no tienen que limitarse a un compás de 4/4: pueden tener
    tantos beats como necesites (5/4, 7/8, etc.) siempre que la sección tenga
    el mismo compás.

    Devuelve {} si no hay sección 'patrones_ritmicos' en el YAML.
    """
    raw = obra.get('patrones_ritmicos', {})
    if not raw:
        return {}
    result = {}
    for nombre, entradas in raw.items():
        try:
            patron = [tuple(e) for e in entradas]
            result[nombre] = patron
        except Exception as exc:
            print(f"  ⚠ patrones_ritmicos.{nombre}: formato inválido — {exc}")
    if result:
        nombres = ', '.join(result.keys())
        print(f"  + Patrones del YAML cargados: {nombres}")
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE SECCIÓN
# ══════════════════════════════════════════════════════════════════════════════
def generate_section(seccion, output_dir, obra=None):
    obra=obra or {}; sid=seccion['id']; c_ini=seccion['compases'][0]
    frases=seccion.get('frases',[])
    patrones_extra=_load_patrones(obra)
    lm_index=_build_leitmotiv_index(obra)
    print(f"\n  Generando Sección {sid}: {seccion['nombre']}")
    print(f"  Cc.{seccion['compases'][0]}–{seccion['compases'][1]}")
    # Mostrar cambios de compás si los hay
    cambios = _build_meter_map(seccion, obra)
    if len(cambios) > 1:
        cc_str = '  '.join(f"c.{c}: {n}/{d}" for c,n,d in cambios)
        print(f"  Compases: {cc_str}")
    mid=MidiFile(type=1,ticks_per_beat=TPB)
    mid.tracks.append(build_tempo_track(seccion,c_ini,obra))
    track_defs=obra.get('tracks',DEFAULT_TRACKS)
    total_overlaps=0
    for tdef in track_defs:
        tid=tdef.get('id',tdef.get('capa','?'))
        nom=tdef.get('nombre',tid); canal=int(tdef.get('canal',0))
        prog=int(tdef.get('programa',0)); capa=tdef.get('capa',tid)
        if tdef.get('contrapunto'):
            cfg=tdef['contrapunto']
            mel_ev=get_capa_events('melodia',frases,c_ini,patrones_extra,lm_index,
                                    seccion=seccion,obra=obra)
            events=generate_contrapunto(mel_ev,int(cfg.get('intervalo',3)),cfg.get('direccion','abajo'))
        else:
            events=get_capa_events(capa,frases,c_ini,patrones_extra,lm_index,
                                    seccion=seccion,obra=obra)
        if events:
            total_overlaps+=check_note_overlaps(events,nom)
            mid.tracks.append(build_track(f'{nom} — Sec.{sid}',canal,prog,events))
            print(f"  ✓ {nom}: {sum(1 for e in events if e[1]=='on')} eventos")
        pb=build_pitchbend_track(nom,canal,frases,c_ini)
        if pb: mid.tracks.append(pb); print(f"  ✓ {nom} (pitch bend)")
        cc=build_cc_track(nom,canal,frases,c_ini)
        if cc: mid.tracks.append(cc); print(f"  ✓ {nom} (CC)")
    if total_overlaps: print(f"  ⚠ {total_overlaps} solapamiento(s) detectado(s)")
    filename=f"Seccion{_safe_filename(str(sid))}_{_safe_filename(seccion['nombre'])[:40]}.mid"
    out_path=os.path.join(output_dir,filename)
    mid.save(out_path); print(f"  → Guardado: {out_path}"); return out_path


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE ARMONÍA (un MIDI por sección con solo la armonía)
# ══════════════════════════════════════════════════════════════════════════════
def generate_harmony(secciones, output_dir, obra):
    """Genera un MIDI por sección exportando únicamente la capa de armonía
    (acordes según process_harmony) en el canal/programa de cuerdas o piano.

    Estructura de salida:
        output_dir/harmony/
          SeccionI_harmony.mid
          SeccionII_harmony.mid
          ...
    """
    harmony_dir = os.path.join(output_dir, 'harmony')
    os.makedirs(harmony_dir, exist_ok=True)
    generated   = []

    # Buscar un track definido con capa 'armonia'; si no hay, usar strings ensemble
    track_defs = obra.get('tracks', DEFAULT_TRACKS)
    arm_tdef   = next((t for t in track_defs if t.get('capa') == 'armonia'), None)
    canal      = int(arm_tdef['canal'])   if arm_tdef else 2
    prog       = int(arm_tdef['programa']) if arm_tdef else 48   # Strings ensemble GM
    nom        = arm_tdef.get('nombre', 'Armonia') if arm_tdef else 'Armonia'

    print(f"\n  Exportando armonía → {harmony_dir}/")
    print(f"  Track: {nom}  canal:{canal}  prog:{prog}")

    for sec in secciones:
        sid    = sec['id']
        c_ini  = sec['compases'][0]
        frases = sec.get('frases', [])

        events = process_harmony(frases, c_ini, seccion=sec, obra=obra)
        if not events:
            print(f"  — Sección {sid}: sin eventos de armonía, omitida")
            continue

        check_note_overlaps(events, f'armonia-{sid}')
        n_on = sum(1 for e in events if e[1] == 'on')

        mid = MidiFile(type=1, ticks_per_beat=TPB)
        mid.tracks.append(build_tempo_track(sec, c_ini, obra))
        mid.tracks.append(build_track(f'Armonia — Sec.{sid}', canal, prog, events))

        fname    = f"Seccion{_safe_filename(str(sid))}_harmony.mid"
        out_path = os.path.join(harmony_dir, fname)
        mid.save(out_path)
        generated.append(out_path)
        print(f"  ✓ {fname}  ({n_on} acordes/notas)")

    return generated


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE MELODÍA (un MIDI por sección con solo la melodía)
# ══════════════════════════════════════════════════════════════════════════════
def generate_melody(secciones, output_dir, obra):
    """Genera un MIDI por sección exportando únicamente la capa de melodía.

    Estructura de salida:
        output_dir/melody/
          SeccionI_melody.mid
          SeccionII_melody.mid
          ...
    """
    melody_dir     = os.path.join(output_dir, 'melody')
    os.makedirs(melody_dir, exist_ok=True)
    generated      = []
    patrones_extra = _load_patrones(obra)
    lm_index       = _build_leitmotiv_index(obra)

    # Buscar el track principal de melodía (capa 'viola', 'violin' o 'melodia')
    track_defs = obra.get('tracks', DEFAULT_TRACKS)
    mel_tdef   = next((t for t in track_defs
                       if t.get('capa') in ('melodia', 'viola', 'violin')), None)
    canal = int(mel_tdef['canal'])    if mel_tdef else 0
    prog  = int(mel_tdef['programa']) if mel_tdef else 40   # Violin GM
    nom   = mel_tdef.get('nombre', 'Melodia') if mel_tdef else 'Melodia'

    print(f"\n  Exportando melodía → {melody_dir}/")
    print(f"  Track: {nom}  canal:{canal}  prog:{prog}")

    for sec in secciones:
        sid    = sec['id']
        c_ini  = sec['compases'][0]
        frases = sec.get('frases', [])

        events = get_capa_events('melodia', frases, c_ini, patrones_extra,
                                 lm_index, seccion=sec, obra=obra)
        if not events:
            print(f"  — Sección {sid}: sin eventos de melodía, omitida")
            continue

        check_note_overlaps(events, f'melodia-{sid}')
        n_on = sum(1 for e in events if e[1] == 'on')

        mid = MidiFile(type=1, ticks_per_beat=TPB)
        mid.tracks.append(build_tempo_track(sec, c_ini, obra))
        mid.tracks.append(build_track(f'Melodia — Sec.{sid}', canal, prog, events))

        # Incluir pitch bend si lo hay
        pb = build_pitchbend_track(nom, canal, frases, c_ini)
        if pb:
            mid.tracks.append(pb)

        fname    = f"Seccion{_safe_filename(str(sid))}_melody.mid"
        out_path = os.path.join(melody_dir, fname)
        mid.save(out_path)
        generated.append(out_path)
        print(f"  ✓ {fname}  ({n_on} notas)")

    return generated


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE RITMO (un MIDI por sección con marcato + percusión)
# ══════════════════════════════════════════════════════════════════════════════
def generate_rhythm(secciones, output_dir, obra):
    """Genera un MIDI por sección exportando las capas rítmicas: marcato y
    percusión. Si una sección no tiene percusión, exporta solo el marcato,
    y viceversa.

    Estructura de salida:
        output_dir/rhythm/
          SeccionI_rhythm.mid
          SeccionII_rhythm.mid
          ...
    """
    rhythm_dir     = os.path.join(output_dir, 'rhythm')
    os.makedirs(rhythm_dir, exist_ok=True)
    generated      = []
    patrones_extra = _load_patrones(obra)
    track_defs     = obra.get('tracks', DEFAULT_TRACKS)

    # Buscar tracks de marcato y percusión
    marc_tdef = next((t for t in track_defs if t.get('capa') == 'marcato'), None)
    perc_tdef = next((t for t in track_defs if t.get('capa') == 'percusion'), None)

    marc_canal = int(marc_tdef['canal'])    if marc_tdef else 3
    marc_prog  = int(marc_tdef['programa']) if marc_tdef else 0
    marc_nom   = marc_tdef.get('nombre', 'Marcato') if marc_tdef else 'Marcato'

    perc_canal = int(perc_tdef['canal'])    if perc_tdef else 9
    perc_nom   = perc_tdef.get('nombre', 'Percusion') if perc_tdef else 'Percusion'

    print(f"\n  Exportando ritmo → {rhythm_dir}/")
    print(f"  Marcato: {marc_nom}  canal:{marc_canal}  prog:{marc_prog}")
    print(f"  Percusión: {perc_nom}  canal:{perc_canal}")

    for sec in secciones:
        sid    = sec['id']
        c_ini  = sec['compases'][0]
        frases = sec.get('frases', [])

        marc_events = get_capa_events('marcato',   frases, c_ini, patrones_extra)
        perc_events = get_capa_events('percusion', frases, c_ini, patrones_extra)

        if not marc_events and not perc_events:
            print(f"  — Sección {sid}: sin eventos rítmicos, omitida")
            continue

        mid = MidiFile(type=1, ticks_per_beat=TPB)
        mid.tracks.append(build_tempo_track(sec, c_ini, obra))

        if marc_events:
            check_note_overlaps(marc_events, f'marcato-{sid}')
            mid.tracks.append(build_track(f'Marcato — Sec.{sid}',
                                          marc_canal, marc_prog, marc_events))
            n_marc = sum(1 for e in marc_events if e[1] == 'on')
        else:
            n_marc = 0

        if perc_events:
            mid.tracks.append(build_track(f'Percusion — Sec.{sid}',
                                          perc_canal, 0, perc_events))
            n_perc = sum(1 for e in perc_events if e[1] == 'on')
        else:
            n_perc = 0

        fname    = f"Seccion{_safe_filename(str(sid))}_rhythm.mid"
        out_path = os.path.join(rhythm_dir, fname)
        mid.save(out_path)
        generated.append(out_path)
        print(f"  ✓ {fname}  (marcato:{n_marc}  perc:{n_perc})")

    return generated


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE STEMS (un MIDI por instrumento)
# ══════════════════════════════════════════════════════════════════════════════
def generate_stems(secciones, output_dir, obra):
    """Genera un MIDI independiente por instrumento y sección.

    Estructura de salida:
        output_dir/stems/
          SeccionI/
            01_Violin_solista.mid
            02_Cuerdas_armonia.mid   ...
          SeccionII/
            ...
    Todos los stems comparten la misma pista de tempo, por lo que
    son directamente importables y alineables en cualquier DAW.
    """
    track_defs     = obra.get('tracks', DEFAULT_TRACKS)
    patrones_extra = _load_patrones(obra)
    lm_index       = _build_leitmotiv_index(obra)
    stems_root     = os.path.join(output_dir, 'stems')
    generated      = []
    print(f"\n  Exportando stems → {stems_root}/")

    for sec in secciones:
        sid    = sec['id']
        c_ini  = sec['compases'][0]
        frases = sec.get('frases', [])
        sec_dir= os.path.join(stems_root, f"Seccion{_safe_filename(str(sid))}")
        os.makedirs(sec_dir, exist_ok=True)
        print(f"\n  Sección {sid}: {sec['nombre']}")

        for idx, tdef in enumerate(track_defs, 1):
            tid   = tdef.get('id', tdef.get('capa','?'))
            nom   = tdef.get('nombre', tid)
            canal = int(tdef.get('canal', 0))
            prog  = int(tdef.get('programa', 0))
            capa  = tdef.get('capa', tid)

            if tdef.get('contrapunto'):
                cfg    = tdef['contrapunto']
                mel_ev = get_capa_events('melodia', frases, c_ini, patrones_extra, lm_index)
                events = generate_contrapunto(mel_ev,
                             int(cfg.get('intervalo',3)), cfg.get('direccion','abajo'))
            else:
                events = get_capa_events(capa, frases, c_ini, patrones_extra, lm_index)

            pb_track = build_pitchbend_track(nom, canal, frases, c_ini)
            cc_track = build_cc_track(nom, canal, frases, c_ini)

            if not events and not pb_track and not cc_track:
                print(f"    — {nom}: sin eventos, omitido")
                continue

            mid = MidiFile(type=1, ticks_per_beat=TPB)
            mid.tracks.append(build_tempo_track(sec, c_ini, obra))
            if events:
                check_note_overlaps(events, nom)
                mid.tracks.append(build_track(f'{nom} — Sec.{sid}', canal, prog, events))
            if pb_track: mid.tracks.append(pb_track)
            if cc_track: mid.tracks.append(cc_track)

            n_on  = sum(1 for e in events if e[1]=='on') if events else 0
            fname = f"{idx:02d}_{_safe_filename(nom)}.mid"
            path  = os.path.join(sec_dir, fname)
            mid.save(path)
            generated.append(path)
            print(f"    ✓ {fname}  ({n_on} notas)")

    return generated


def generate_stems_completa(obra, output_dir):
    """Stems de la obra completa concatenada, uno por instrumento."""
    track_defs     = obra.get('tracks', DEFAULT_TRACKS)
    patrones_extra = _load_patrones(obra)
    lm_index       = _build_leitmotiv_index(obra)
    secciones      = obra.get('secciones', [])
    stems_root     = os.path.join(output_dir, 'stems', 'COMPLETA')
    os.makedirs(stems_root, exist_ok=True)
    generated      = []
    print(f"\n  Stems obra completa → {stems_root}/")

    # Calcular tick_offset de cada sección respetando cambios de compás
    tick_offsets = []
    cursor = 0
    for sec in secciones:
        tick_offsets.append(cursor)
        cambios = _build_meter_map(sec, obra)
        c_ini, c_fin = sec['compases']
        n_bars = c_fin - c_ini + 1
        cur_num, cur_den = cambios[0][1], cambios[0][2]
        for i in range(n_bars):
            b = c_ini + i
            for (c_abs, n, d) in cambios:
                if c_abs == b:
                    cur_num, cur_den = n, d
            cursor += int(round(cur_num * (4 / cur_den) * TPB))

    # Pista de tempo global con armaduras correctas — una sola vez
    import copy as _copy
    global_tempo = build_global_tempo_track(secciones, obra)

    for idx, tdef in enumerate(track_defs, 1):
        tid   = tdef.get('id', tdef.get('capa','?'))
        nom   = tdef.get('nombre', tid)
        canal = int(tdef.get('canal', 0))
        prog  = int(tdef.get('programa', 0))
        capa  = tdef.get('capa', tid)

        all_events = []
        for sec, offset in zip(secciones, tick_offsets):
            c_ini  = sec['compases'][0]; frases = sec.get('frases', [])
            if tdef.get('contrapunto'):
                cfg    = tdef['contrapunto']
                mel_ev = get_capa_events('melodia', frases, c_ini, patrones_extra,
                                          lm_index, seccion=sec, obra=obra)
                evs    = generate_contrapunto(mel_ev,
                             int(cfg.get('intervalo', 3)), cfg.get('direccion', 'abajo'))
            else:
                evs = get_capa_events(capa, frases, c_ini, patrones_extra,
                                       lm_index, seccion=sec, obra=obra)
            for (t, oo, n, v) in evs:
                all_events.append((t + offset, oo, n, v))

        if not all_events:
            print(f"    — {nom}: sin eventos, omitido"); continue

        mid = MidiFile(type=1, ticks_per_beat=TPB)
        mid.tracks.append(_copy.deepcopy(global_tempo))
        check_note_overlaps(all_events, nom)
        mid.tracks.append(build_track(nom, canal, prog, all_events))

        n_on  = sum(1 for e in all_events if e[1] == 'on')
        fname = f"{idx:02d}_{_safe_filename(nom)}_COMPLETA.mid"
        path  = os.path.join(stems_root, fname)
        mid.save(path)
        generated.append(path)
        print(f"    ✓ {fname}  ({n_on} notas)")

    return generated


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE LEITMOTIVS (un MIDI por leitmotiv)
# ══════════════════════════════════════════════════════════════════════════════
def generate_leitmotiv_midis(obra, output_dir):
    """Genera un MIDI por leitmotiv con todas las frases que lo usan.

    Estructura de salida:
        output_dir/leitmotivs/
          TEMA_A.mid          ← todas las frases con leitmotiv: TEMA_A
          TEMA_B.mid
          MOTIVO_RITMICO.mid
          ...
          _sin_leitmotiv.mid  ← frases sin leitmotiv asignado (si las hay)

    Cada fichero contiene:
      - Track 0: pista de tempo global (igual que el MIDI completo, con los
        offsets reales de cada sección, así los leitmotivs son directamente
        comparables y alineables en un DAW)
      - Un track por capa de instrumento (melodia, armonia, marcato, pad,
        ostinato, solistas...) con solo las notas de las frases que usan
        ese leitmotiv

    Esto permite:
      - Escuchar de un golpe todas las apariciones de un tema
      - Importar en el DAW como una capa sobre el MIDI completo
      - Analizar la evolución armónica/melódica de cada célula temática
    """
    leitmotivs = obra.get('leitmotivs', [])
    secciones  = obra.get('secciones', [])
    track_defs = obra.get('tracks', DEFAULT_TRACKS)
    patrones_extra = _load_patrones(obra)
    lm_index   = _build_leitmotiv_index(obra)
    lm_dir     = os.path.join(output_dir, 'leitmotivs')
    os.makedirs(lm_dir, exist_ok=True)
    generated  = []

    print(f"\n  Exportando leitmotivs → {lm_dir}/")

    # Calcular tick_offset de cada sección respetando cambios de compás
    tick_offsets = {}
    cursor = 0
    for sec in secciones:
        tick_offsets[sec['id']] = cursor
        cambios = _build_meter_map(sec, obra)
        c_ini, c_fin = sec['compases']
        n_bars = c_fin - c_ini + 1
        cur_num, cur_den = cambios[0][1], cambios[0][2]
        for i in range(n_bars):
            b = c_ini + i
            for (c_abs, n, d) in cambios:
                if c_abs == b:
                    cur_num, cur_den = n, d
            cursor += int(round(cur_num * (4 / cur_den) * TPB))

    # Pista de tempo global con armaduras correctas
    import copy as _copy
    _cached_global_tempo = build_global_tempo_track(secciones, obra)

    # Recopilar frases agrupadas por leitmotiv
    # Estructura: {lm_id: [(sec_id, offset, c_ini, frase), ...]}
    grupos = defaultdict(list)
    for sec in secciones:
        offset = tick_offsets[sec['id']]
        c_ini  = sec['compases'][0]
        for frase in sec.get('frases', []):
            lm = frase.get('leitmotiv') or '_sin_leitmotiv'
            if lm in ('null', 'None', None, '-', '—'):
                lm = '_sin_leitmotiv'
            grupos[lm].append((sec['id'], offset, c_ini, frase))

    if not grupos:
        print("  ⚠ No hay frases con leitmotiv asignado")
        return []

    # Orden: primero los leitmotivs definidos (en su orden), luego _sin_leitmotiv
    lm_ids_ordenados = [lm['id'] for lm in leitmotivs]
    lm_ids_ordenados += [k for k in grupos if k not in lm_ids_ordenados]

    for lm_id in lm_ids_ordenados:
        apariciones = grupos.get(lm_id)
        if not apariciones:
            continue

        # Nombre legible
        lm_meta = next((lm for lm in leitmotivs if lm['id'] == lm_id), None)
        lm_nombre = lm_meta['nombre'] if lm_meta else lm_id
        n_apariciones = len(apariciones)

        print(f"\n  [{lm_id}]  {lm_nombre}  ({n_apariciones} aparición/es)")

        mid = MidiFile(type=1, ticks_per_beat=TPB)
        mid.tracks.append(_copy.deepcopy(_cached_global_tempo))

        total_notas = 0

        for idx, tdef in enumerate(track_defs, 1):
            tid   = tdef.get('id', tdef.get('capa', '?'))
            nom   = tdef.get('nombre', tid)
            canal = int(tdef.get('canal', 0))
            prog  = int(tdef.get('programa', 0))
            capa  = tdef.get('capa', tid)

            all_events = []
            for (sid, offset, c_ini, frase) in apariciones:
                if tdef.get('contrapunto'):
                    cfg    = tdef['contrapunto']
                    mel_ev = get_capa_events('melodia', [frase], c_ini, patrones_extra, lm_index)
                    evs    = generate_contrapunto(mel_ev,
                                 int(cfg.get('intervalo', 3)),
                                 cfg.get('direccion', 'abajo'))
                else:
                    evs = get_capa_events(capa, [frase], c_ini, patrones_extra, lm_index)
                for (t, oo, n, v) in evs:
                    all_events.append((t + offset, oo, n, v))

            if not all_events:
                continue

            check_note_overlaps(all_events, f"{lm_id}/{nom}")
            mid.tracks.append(build_track(f"{nom} [{lm_id}]", canal, prog, all_events))
            n_on = sum(1 for e in all_events if e[1] == 'on')
            total_notas += n_on
            print(f"    {nom}: {n_on} notas")

        if len(mid.tracks) <= 1:
            print(f"    — sin notas, omitido")
            continue

        fname = f"{_safe_filename(lm_id)}.mid"
        path  = os.path.join(lm_dir, fname)
        mid.save(path)
        generated.append(path)
        print(f"    → {fname}  ({total_notas} notas totales, {n_apariciones} frases)")

    return generated

# ══════════════════════════════════════════════════════════════════════════════
#  OBRA COMPLETA CONCATENADA
# ══════════════════════════════════════════════════════════════════════════════
def generate_full_obra(obra, output_dir):
    titulo    = obra['obra']['titulo']
    secciones = obra.get('secciones', [])
    track_defs = obra.get('tracks', DEFAULT_TRACKS)
    print(f"\n  Generando obra completa: {titulo}")

    mid = MidiFile(type=1, ticks_per_beat=TPB)
    mid.tracks.append(build_global_tempo_track(secciones, obra, titulo))

    patrones_extra = _load_patrones(obra)
    lm_index       = _build_leitmotiv_index(obra)
    all_events     = {tdef['id']: [] for tdef in track_defs}

    # Calcular tick offsets respetando cambios de compás por sección
    tick_offsets = []
    tick_cursor  = 0
    for sec in secciones:
        tick_offsets.append(tick_cursor)
        cambios = _build_meter_map(sec, obra)
        c_ini, c_fin = sec['compases']
        n_bars = c_fin - c_ini + 1
        cur_num, cur_den = cambios[0][1], cambios[0][2]
        for i in range(n_bars):
            b = c_ini + i
            for (c_abs, n, d) in cambios:
                if c_abs == b:
                    cur_num, cur_den = n, d
            tick_cursor += int(round(cur_num * (4 / cur_den) * TPB))

    for sec, tick_offset in zip(secciones, tick_offsets):
        c_ini  = sec['compases'][0]
        frases = sec.get('frases', [])
        for tdef in track_defs:
            tid  = tdef['id']
            capa = tdef.get('capa', tid)
            if tdef.get('contrapunto'):
                cfg    = tdef['contrapunto']
                mel_ev = get_capa_events('melodia', frases, c_ini, patrones_extra,
                                          lm_index, seccion=sec, obra=obra)
                evs    = generate_contrapunto(mel_ev,
                             int(cfg.get('intervalo', 3)), cfg.get('direccion', 'abajo'))
            else:
                evs = get_capa_events(capa, frases, c_ini, patrones_extra,
                                       lm_index, seccion=sec, obra=obra)
            for (tick, on_off, note, vel) in evs:
                all_events[tid].append((tick + tick_offset, on_off, note, vel))
        n_mel = sum(1 for e in all_events.get('melodia', [])
                    if e[1] == 'on' and e[0] >= tick_offset)
        print(f"    Sec.{sec['id']:3s}  mel:{n_mel:3d}  offset:{tick_offset}")

    for tdef in track_defs:
        tid    = tdef['id']
        events = all_events.get(tid, [])
        if events:
            mid.tracks.append(build_track(tdef.get('nombre', tid),
                               int(tdef.get('canal', 0)),
                               int(tdef.get('programa', 0)), events))

    total_bars = sum(s['compases'][1] - s['compases'][0] + 1 for s in secciones)
    avg_bpm    = sum((s['tempo']['bpm_inicio'] +
                      s['tempo'].get('bpm_fin', s['tempo']['bpm_inicio'])) / 2
                     for s in secciones) / max(len(secciones), 1)
    dur_sec    = total_bars * 4 / avg_bpm * 60
    out_path   = os.path.join(output_dir, f"{_safe_filename(titulo)}_COMPLETA.mid")
    mid.save(out_path)
    print(f"\n  Obra completa: {total_bars} cc · ~{int(dur_sec//60)}'{int(dur_sec%60):02d}\" · {len(mid.tracks)} tracks")
    print(f"  → Guardado: {out_path}")
    return out_path

# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════
def _safe_filename(s):
    for src,dst in {'á':'a','é':'e','í':'i','ó':'o','ú':'u','ü':'u','ñ':'n',
                    'Á':'A','É':'E','Í':'I','Ó':'O','Ú':'U','Ñ':'N',
                    'ç':'c','Ç':'C',' ':'_','/':'-','\\':'-','—':'-','–':'-',
                    '(':'',')':'','[':'',']':'','"':'',  "'":''}.items():
        s=s.replace(src,dst)
    return s.encode('ascii','ignore').decode()

# ══════════════════════════════════════════════════════════════════════════════
#  PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
def preview(obra):
    print(f"\n{'═'*60}\n  {obra['obra']['titulo']}")
    if obra['obra'].get('subtitulo'): print(f"  {obra['obra']['subtitulo']}")
    print(f"  {obra['obra'].get('compas','?')} · {obra['obra'].get('tonalidad','?')} · {obra['obra'].get('compases_totales','?')} cc.")
    print(f"{'─'*60}")
    lms=obra.get('leitmotivs',[])
    if lms:
        print(f"\n  LEITMOTIVS ({len(lms)}):")
        for lm in lms: print(f"    [{lm['id']:20s}] {lm['nombre']}")
    for sec in obra.get('secciones',[]):
        c=sec['compases']; t=sec['tempo']
        print(f"\n  SECCIÓN {sec['id']}: {sec['nombre']}")
        print(f"  Cc.{c[0]}–{c[1]} · {sec.get('duracion_aprox','?')} · {t['bpm_inicio']}→{t.get('bpm_fin',t['bpm_inicio'])} BPM ({t['tipo']})")
        for frase in sec.get('frases',[]):
            fc=frase['compases']
            print(f"    {frase['id']:8s} cc.{fc[0]:3d}–{fc[1]:3d}  {frase['nombre'][:35]:35s} mel:{len(frase.get('melodia',[])):3d}  arm:{len(frase.get('armonia',[])):2d}  lm:{frase.get('leitmotiv','—')}")
    print(f"\n{'═'*60}")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser=argparse.ArgumentParser(description='Genera MIDIs desde un fichero YAML de partitura.',
                                    formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__)
    parser.add_argument('--yaml',         default='partitura.yaml')
    parser.add_argument('--output',       default='output')
    parser.add_argument('--seccion',      help='Solo esta sección (ej: I, II)')
    parser.add_argument('--frase',        help='Solo esta frase (ej: I.2)')
    parser.add_argument('--preview',      action='store_true')
    parser.add_argument('--completa',     action='store_true')
    parser.add_argument('--validar',      action='store_true')
    parser.add_argument('--estadisticas', action='store_true')
    parser.add_argument('--diff',         nargs=2, metavar=('YAML_A','YAML_B'))
    parser.add_argument('--nueva-frase',  metavar='ID')
    parser.add_argument('--compases',     metavar='INI-FIN')
    parser.add_argument('--leitmotiv')
    parser.add_argument('--render-audio', action='store_true')
    parser.add_argument('--soundfont')
    parser.add_argument('--csv',          action='store_true')
    parser.add_argument('--sin-validar',  action='store_true')
    parser.add_argument('--html',         action='store_true',
                        help='Genera esquema visual HTML de la obra')
    parser.add_argument('--stems',        action='store_true',
                        help='Exporta un MIDI por instrumento en output/stems/')
    parser.add_argument('--leitmotivs',   action='store_true',
                        help='Exporta un MIDI por leitmotiv en output/leitmotivs/')
    parser.add_argument('--harmony',      action='store_true',
                        help='Exporta un MIDI de armonía por sección en output/harmony/')
    parser.add_argument('--melody',       action='store_true',
                        help='Exporta un MIDI de melodía por sección en output/melody/')
    parser.add_argument('--rhythm',       action='store_true',
                        help='Exporta un MIDI de ritmo (marcato+percusión) por sección en output/rhythm/')
    args=parser.parse_args()

    if args.diff: diff_obras(args.diff[0],args.diff[1]); return

    if not os.path.exists(args.yaml): print(f"✗ No encontrado: {args.yaml}"); sys.exit(1)
    with open(args.yaml,'r',encoding='utf-8') as f: obra=yaml.safe_load(f)

    if args.preview:      preview(obra); return
    if args.estadisticas: estadisticas(obra); return
    if args.html:
        os.makedirs(args.output, exist_ok=True)
        export_html(obra, args.output); return
    if args.validar:      n=validate_yaml(obra); sys.exit(0 if n==0 else 1)
    if args.nueva_frase:
        if not args.compases: print("✗ --nueva-frase requiere --compases INI-FIN"); sys.exit(1)
        nueva_frase_template(args.nueva_frase, args.compases, args.leitmotiv, args.yaml); return

    if not args.sin_validar:
        n_errors=validate_yaml(obra)
        if n_errors>0:
            print(f"\n  ✗ Corrije los {n_errors} error(es) antes de generar.")
            print("    Usa --sin-validar para omitir."); sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    if args.csv: export_csv(obra, args.output)

    secciones=obra.get('secciones',[])
    if args.seccion:
        secciones=[s for s in secciones if s['id']==args.seccion]
        if not secciones: print(f"✗ Sección '{args.seccion}' no encontrada"); sys.exit(1)
    if args.frase:
        sec_id=args.frase.split('.')[0]
        secciones=[s for s in secciones if s['id']==sec_id]
        for sec in secciones: sec['frases']=[f for f in sec.get('frases',[]) if f['id']==args.frase]
        if not any(s.get('frases') for s in secciones): print(f"✗ Frase '{args.frase}' no encontrada"); sys.exit(1)

    print(f"\n{'═'*60}\n  {obra['obra']['titulo']}\n  Generando MIDIs desde {args.yaml}\n{'═'*60}")

    generated=[]
    if args.completa:
        generated.append(generate_full_obra(obra,args.output))
        print(f"\n  Generando también MIDIs por sección…")
        for sec in secciones: generated.append(generate_section(sec,args.output,obra))
    else:
        for sec in secciones: generated.append(generate_section(sec,args.output,obra))

    if args.stems:
        print(f"\n{'─'*60}")
        stems = generate_stems(secciones, args.output, obra)
        if args.completa:
            stems += generate_stems_completa(obra, args.output)
        generated.extend(stems)
        print(f"\n  ✓ {len(stems)} stem(s) en '{args.output}/stems/'")

    if args.leitmotivs:
        print(f"\n{'─'*60}")
        lm_files = generate_leitmotiv_midis(obra, args.output)
        generated.extend(lm_files)
        print(f"\n  ✓ {len(lm_files)} leitmotiv(s) en '{args.output}/leitmotivs/'")

    if args.harmony:
        print(f"\n{'─'*60}")
        harm_files = generate_harmony(secciones, args.output, obra)
        generated.extend(harm_files)
        print(f"\n  ✓ {len(harm_files)} fichero(s) de armonía en '{args.output}/harmony/'")

    if args.melody:
        print(f"\n{'─'*60}")
        mel_files = generate_melody(secciones, args.output, obra)
        generated.extend(mel_files)
        print(f"\n  ✓ {len(mel_files)} fichero(s) de melodía en '{args.output}/melody/'")

    if args.rhythm:
        print(f"\n{'─'*60}")
        rhy_files = generate_rhythm(secciones, args.output, obra)
        generated.extend(rhy_files)
        print(f"\n  ✓ {len(rhy_files)} fichero(s) de ritmo en '{args.output}/rhythm/'")

    print(f"\n{'─'*60}\n  ✓ {len(generated)} fichero(s) generado(s) en '{args.output}/'")
    for p in generated:
        flag=" ← OBRA COMPLETA" if p.endswith('_COMPLETA.mid') else ""
        print(f"    {os.path.basename(p)}{flag}")
    print(f"{'═'*60}\n")

    if args.render_audio:
        render_audio([p for p in generated if p.endswith('.mid')], args.soundfont)

if __name__ == '__main__':
    main()
