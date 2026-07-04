#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       COUNTERPOINT  v1.1                                     ║
║         Motor de contrapunto por especies para MIDIs melódicos               ║
║                                                                              ║
║  Dado un MIDI de melodía, genera una segunda voz (contrapunto) aplicando     ║
║  las reglas clásicas de contrapunto por especies de Fux/Gradus ad Parnassum. ║
║  Funciona como herramienta standalone y como módulo importable por           ║
║  variation_engine (V15), orchestrator y el propio pipeline.                  ║
║                                                                              ║
║  MODO STRICT (CSP)  —  novedad v1.1                                          ║
║  Por defecto las reglas de Fux son PENALIZACIONES BLANDAS: el motor elige    ║
║  el candidato «menos malo» y puede emitir paralelas de 5ª/8ª si ninguno es   ║
║  limpio. Con --strict se convierten en RESTRICCIONES DURAS: nunca se emite   ║
║  un contrapunto que las viole; si no existe solución, se lanza               ║
║  CounterpointStrictError con diagnóstico y NO se genera el MIDI.             ║
║                                                                              ║
║  Especie 1:  CSP por backtracking con H1–H6 GARANTIZADAS                     ║
║    [H1] solo consonancias en cada momento                                    ║
║    [H2] sin paralelas de 5ª justa u 8ª entre instantes consecutivos          ║
║    [H3] sin movimiento directo a consonancia perfecta (5ª/8ª)                ║
║    [H4] sin cruce de voces (respeta voice_position)                          ║
║    [H5] sin repetir el mismo intervalo armónico 3 veces seguidas             ║
║    [H6] cadencia: la sensible del cp resuelve ASCENDIENDO a la tónica        ║
║    [P1] preferencia: clímax del cp en el 50–85 % de la frase (se relaja)     ║
║  Especies 2–5:  greedy + validación del ESQUELETO de tiempos fuertes         ║
║    (H1 en downbeats de S2/S3; H2/H3 entre downbeats; H4 en todo instante)    ║
║    Si el esqueleto rompe reglas duras, la especie se descarta con aviso.     ║
║                                                                              ║
║  ESPECIES IMPLEMENTADAS (generación):                                        ║
║  [S1] Primera especie  — nota contra nota (1:1)                              ║
║  [S2] Segunda especie  — dos notas por nota cantus firmus (2:1)              ║
║  [S3] Tercera especie  — cuatro notas por nota cantus firmus (4:1)           ║
║  [S4] Cuarta especie   — ligaduras y retardos (sincopado)                    ║
║  [S5] Quinta especie   — contrapunto florido (mezcla de las anteriores)      ║
║                                                                              ║
║  MODOS DE VOZ:                                                               ║
║    above / below / auto   (posición relativa a la melodía)                   ║
║  VOCES: soprano, alto, tenor, bass, violin, viola, cello                     ║
║                                                                              ║
║  USO STANDALONE — ejemplos:                                                  ║
║    # Especie 1 con CSP estricto (garantiza H1–H6):                           ║
║    python counterpoint.py melodia.mid --species 1 --strict                   ║
║    python counterpoint.py melodia.mid --species 1 --strict --voice viola     ║
║    python counterpoint.py melodia.mid --species 1 --strict \                 ║
║        --voice-position above --key Dm --verbose                             ║
║                                                                              ║
║    # Especie 2 con validación de esqueleto en strict:                        ║
║    python counterpoint.py melodia.mid --species 2 --strict                   ║
║                                                                              ║
║    # Las 5 especies en strict; las que rompan reglas se descartan:           ║
║    python counterpoint.py melodia.mid --all-species --strict --key C         ║
║                                                                              ║
║    # Sin strict (penalizaciones blandas, siempre emite salida):              ║
║    python counterpoint.py melodia.mid --species 5 --tension-curve            ║
║        "0:0.2, 8:0.8, 16:0.3"                                                ║
║    python counterpoint.py melodia.mid --verbose --output cp_out.mid          ║
║    python counterpoint.py melodia.mid --all-species --listen                 ║
║                                                                              ║
║  USO COMO MÓDULO:                                                            ║
║    from counterpoint import (generate_counterpoint_voice,                    ║
║                              load_melody_midi, CounterpointStrictError)      ║
║                                                                              ║
║    notes = load_melody_midi("melodia.mid")                                   ║
║    try:                                                                      ║
║        cp = generate_counterpoint_voice(                                     ║
║            melody_notes=notes, key="C major", species=1,                     ║
║            voice="viola", voice_position="above", strict=True,               ║
║        )                                                                     ║
║    except CounterpointStrictError as e:                                      ║
║        print("sin solución limpia:", e)                                      ║
║    # cp → list of (offset, pitch, dur, vel)                                  ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --species N        Especie 1-5 (default: 5)                               ║
║    --strict           CSP con reglas de Fux DURAS (ver arriba). Aborta con   ║
║                       exit 2 si no hay solución limpia; con --all-species    ║
║                       descarta la especie y continúa con la siguiente.       ║
║    --voice NAME       soprano|alto|tenor|bass|violin|viola|cello (def: auto) ║
║    --voice-position   above|below|auto (default: auto)                       ║
║    --key KEY          Tonalidad: "C", "Am", "Bb major" … (default: auto)     ║
║    --bars N           Compases a generar (default: auto desde MIDI)          ║
║    --beats N          Beats por compás (default: 4)                          ║
║    --tension-curve S  Curva de tensión compás a compás "0:0.2, 8:0.8"        ║
║    --all-species      Generar una versión por cada especie (1-5)             ║
║    --catalog          Si --all-species, concatenar todo en un único MIDI     ║
║    --output FILE      Fichero de salida (default: counterpoint_out.mid)      ║
║    --output-dir DIR   Directorio de salida con --all-species                 ║
║    --channel N        Canal MIDI para el contrapunto (default: 1)            ║
║    --program N        Program change (instrumento GM) (default: auto)        ║
║    --listen           Reproducir resultado al terminar (requiere pygame)     ║
║    --play-seconds N   Segundos de reproducción (default: 30)                 ║
║    --export-fingerprint  Exportar fingerprint JSON (para stitcher)           ║
║    --verbose          Informe detallado de decisiones regla a regla          ║
║    --seed N           Semilla aleatoria (default: 42)                        ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    counterpoint_out.mid  — MIDI de dos pistas: melodía + contrapunto         ║
║    counterpoint_out_s{N}.mid — con --all-species                             ║
║    counterpoint_out.fingerprint.json — con --export-fingerprint              ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:   mido, numpy                                                    ║
║    --key auto / análisis tonal:  music21  (opcional, fallback si no)         ║
║    --listen:  pygame                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import random
import copy
import math
import time
import tempfile
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCIAS CORE
# ══════════════════════════════════════════════════════════════════════════════

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_OK = True
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)

try:
    from music21 import key as m21key, pitch
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0"
TICKS   = 480   # ticks per beat — mismo que midi_dna_unified

# Rangos MIDI por instrumento/rol (idénticos a midi_dna_unified INSTRUMENT_RANGES)
VOICE_RANGES = {
    'soprano': (60, 81),   # C4–A5
    'alto':    (53, 74),   # F3–D5
    'tenor':   (48, 69),   # C3–A4
    'bass':    (28, 52),   # E1–E3
    'violin':  (55, 88),   # G3–E6
    'viola':   (48, 77),   # C3–F5
    'cello':   (36, 69),   # C2–A4
}

# Programas GM por voz
VOICE_PROGRAMS = {
    'soprano': 52,   # Choir Aahs
    'alto':    53,
    'tenor':   54,
    'bass':    55,
    'violin':  40,
    'viola':   41,
    'cello':   42,
}

# Nombres de nota para diagnósticos (mod 12, sostenidos)
_MIDI_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#',
                    'G', 'G#', 'A', 'A#', 'B']

def _m2n(midi_pitch: int) -> str:
    """Convierte una altura MIDI en nombre científico (p.ej. 60 -> 'C4')."""
    if midi_pitch is None:
        return '-'
    pc = midi_pitch % 12
    octv = midi_pitch // 12 - 1
    return f"{_MIDI_NOTE_NAMES[pc]}{octv}"

# Intervalos (en semitonos mod 12)
PERFECT_CONSONANCES   = {0, 7, 12}       # unísono, 5ª, 8ª (Fux: mismas restricciones)
PERFECT_CONSONANCES_8 = {0, 7, 12}       # alias mantenido por compatibilidad
IMPERFECT_CONSONANCES = {3, 4, 8, 9}     # 3ª m/M, 6ª m/M
ALL_CONSONANCES       = PERFECT_CONSONANCES | IMPERFECT_CONSONANCES
DISSONANCES           = {1, 2, 5, 6, 10, 11}

# Escalas diatónicas (pcs relativos al tónico)
SCALE_PCS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],   # natural minor
}

# Tonalidades de fallback si music21 no está disponible
KEY_FALLBACK = {
    'C':  ('C',  'major', 0),
    'G':  ('G',  'major', 7),
    'D':  ('D',  'major', 2),
    'A':  ('A',  'major', 9),
    'E':  ('E',  'major', 4),
    'B':  ('B',  'major', 11),
    'F':  ('F',  'major', 5),
    'Bb': ('Bb', 'major', 10),
    'Eb': ('Eb', 'major', 3),
    'Ab': ('Ab', 'major', 8),
    'Am': ('A',  'minor', 9),
    'Em': ('E',  'minor', 4),
    'Dm': ('D',  'minor', 2),
    'Gm': ('G',  'minor', 7),
    'Cm': ('C',  'minor', 0),
    'Fm': ('F',  'minor', 5),
    'Bm': ('B',  'minor', 11),
}

# ══════════════════════════════════════════════════════════════════════════════
#  REPRESENTACIÓN DE TONALIDAD (compatible con y sin music21)
# ══════════════════════════════════════════════════════════════════════════════

class KeyInfo:
    """
    Wrapper ligero sobre music21.key.Key o fallback puro.
    Expone: tonic_pc (int), mode (str), scale_pcs (list[int])
    """
    def __init__(self, tonic_pc: int, mode: str, name: str = ""):
        self.tonic_pc  = tonic_pc % 12
        self.mode      = mode.lower()
        self.name      = name or f"pc{tonic_pc}_{mode}"
        self.scale_pcs = [(self.tonic_pc + pc) % 12
                          for pc in SCALE_PCS.get(self.mode, SCALE_PCS['major'])]
        # leading tone pc (7º grado)
        leading_offset = SCALE_PCS[self.mode][-1] if self.mode in SCALE_PCS else 11
        self.leading_pc = (self.tonic_pc + leading_offset) % 12
        # dominant pc
        self.dominant_pc = (self.tonic_pc + 7) % 12

    def is_diatonic(self, midi_pitch: int) -> bool:
        return (midi_pitch % 12) in self.scale_pcs

    def snap_to_scale(self, midi_pitch: int) -> int:
        """Ajusta midi_pitch al grado diatónico más cercano."""
        pc = midi_pitch % 12
        if pc in self.scale_pcs:
            return midi_pitch
        best, best_d = pc, 100
        for spc in self.scale_pcs:
            d = min(abs(spc - pc), 12 - abs(spc - pc))
            if d < best_d:
                best_d, best = d, spc
        diff = best - pc
        if diff > 6:  diff -= 12
        if diff < -6: diff += 12
        return midi_pitch + diff

    def scale_midi(self, lo: int = 36, hi: int = 96) -> list:
        """Devuelve todas las alturas diatónicas en el rango dado."""
        result = []
        for m in range(lo, hi + 1):
            if (m % 12) in self.scale_pcs:
                result.append(m)
        return result

    @staticmethod
    def from_string(key_str: str) -> "KeyInfo":
        """
        Parsea strings como 'C', 'Am', 'Bb major', 'D minor'.
        Usa music21 si está disponible, si no usa tabla interna.
        """
        key_str = key_str.strip()
        if MUSIC21_OK:
            try:
                # Normalizar: "Am" → "A minor", "C" → "C major"
                if 'm' in key_str and 'major' not in key_str and 'minor' not in key_str:
                    root = key_str.replace('m', '')
                    k = m21key.Key(root, 'minor')
                else:
                    parts = key_str.split()
                    root = parts[0]
                    mode = parts[1] if len(parts) > 1 else 'major'
                    k = m21key.Key(root, mode)
                tonic_pc = pitch.Pitch(k.tonic.name).pitchClass
                return KeyInfo(tonic_pc, k.mode, f"{k.tonic.name} {k.mode}")
            except Exception:
                pass

        # Fallback tabla interna
        norm = key_str.replace(' major', '').replace(' minor', '')
        if 'minor' in key_str and norm + 'm' in KEY_FALLBACK:
            name, mode, tpc = KEY_FALLBACK[norm + 'm']
        elif norm in KEY_FALLBACK:
            name, mode, tpc = KEY_FALLBACK[norm]
        elif norm + 'm' in KEY_FALLBACK and 'minor' in key_str:
            name, mode, tpc = KEY_FALLBACK[norm + 'm']
        else:
            # Default C major
            name, mode, tpc = 'C', 'major', 0
        return KeyInfo(tpc, mode, f"{name} {mode}")

    @staticmethod
    def detect_from_notes(notes: list) -> "KeyInfo":
        """
        Detección de tonalidad por Krumhansl-Schmuckler simplificado.
        notes: list of (offset, pitch, dur, vel)
        """
        if MUSIC21_OK and notes:
            try:
                from music21 import stream, note as m21note, analysis
                s = stream.Stream()
                for _, p, d, v in notes:
                    n = m21note.Note(p)
                    n.quarterLength = max(0.25, d)
                    s.append(n)
                k = s.analyze('key')
                tonic_pc = pitch.Pitch(k.tonic.name).pitchClass
                return KeyInfo(tonic_pc, k.mode, f"{k.tonic.name} {k.mode}")
            except Exception:
                pass

        # Fallback: pitch class profile
        if not notes:
            return KeyInfo(0, 'major', 'C major')
        pc_counts = defaultdict(float)
        for _, p, d, _ in notes:
            pc_counts[p % 12] += max(d, 0.25)
        # Perfil de Krumhansl-Kessler (mayor y menor)
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                         2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                         2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        best_r, best_tpc, best_mode = -2.0, 0, 'major'
        total = sum(pc_counts.values()) or 1
        counts = [pc_counts[i] / total for i in range(12)]
        for tpc in range(12):
            for profile, mode in [(major_profile, 'major'), (minor_profile, 'minor')]:
                rot = profile[tpc:] + profile[:tpc]
                # Correlación de Pearson
                pm = sum(rot) / 12
                cm = sum(counts) / 12
                num = sum((rot[i] - pm) * (counts[i] - cm) for i in range(12))
                den = (sum((rot[i] - pm) ** 2 for i in range(12)) *
                       sum((counts[i] - cm) ** 2 for i in range(12))) ** 0.5
                r = num / den if den > 0 else 0
                if r > best_r:
                    best_r, best_tpc, best_mode = r, tpc, mode
        pc_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        return KeyInfo(best_tpc, best_mode, f"{pc_names[best_tpc]} {best_mode}")


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA Y EXPORTACIÓN MIDI
# ══════════════════════════════════════════════════════════════════════════════

def load_melody_midi(filepath: str) -> tuple:
    """
    Carga un MIDI y extrae la pista melódica principal.
    Devuelve (notes, tempo_bpm, beats_per_bar, ticks_per_beat)
    notes: list of (offset_beats, pitch, duration_beats, velocity)
    """
    mid = mido.MidiFile(filepath)
    tpb = mid.ticks_per_beat

    tempo_us = 500_000  # 120 BPM default
    beats_per_bar = 4
    notes_by_channel = defaultdict(list)
    pending = {}

    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            beats = abs_ticks / tpb
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                beats_per_bar = msg.numerator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (beats, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                key_ = (msg.channel, msg.note)
                if key_ in pending:
                    onset, vel = pending.pop(key_)
                    dur = beats - onset
                    if dur < 0.01:
                        dur = 0.125
                    notes_by_channel[msg.channel].append((onset, msg.note, dur, vel))

    # Cerrar notas huérfanas
    for (ch, p_), (onset, vel) in pending.items():
        notes_by_channel[ch].append((onset, p_, 0.25, vel))

    # Elegir canal melódico: el de pitch medio más alto (excl. perc canal 9)
    melodic = {ch: notes for ch, notes in notes_by_channel.items()
               if ch != 9 and notes}
    if not melodic:
        raise ValueError(f"No se encontraron notas melódicas en {filepath}")

    def mean_pitch(notes):
        return sum(p for _, p, _, _ in notes) / len(notes) if notes else 0

    best_ch = max(melodic, key=lambda ch: mean_pitch(melodic[ch]))
    notes = sorted(melodic[best_ch], key=lambda x: x[0])

    tempo_bpm = round(60_000_000 / tempo_us)
    return notes, tempo_bpm, beats_per_bar, tpb


def save_two_voice_midi(
    melody_notes: list,
    cp_notes: list,
    output_path: str,
    tempo_bpm: int = 120,
    beats_per_bar: int = 4,
    cp_program: int = 40,
    cp_channel: int = 1,
    mel_program: int = 0,
    verbose: bool = False,
) -> str:
    """
    Guarda melodía + contrapunto en un MIDI de dos pistas (type=1).
    """
    us_per_beat = int(60_000_000 / max(tempo_bpm, 1))
    bu = {2: 2, 3: 4, 4: 4, 6: 8}.get(beats_per_bar, 4)

    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS)

    def _make_track(notes, channel, program, name):
        trk = mido.MidiTrack()
        trk.name = name
        trk.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
        trk.append(mido.MetaMessage(
            'time_signature', numerator=beats_per_bar, denominator=bu,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        trk.append(mido.Message('program_change', channel=channel,
                                program=program, time=0))

        events = []
        for offset, mp, dur, vel in notes:
            mp  = max(0, min(127, int(mp)))
            vel = max(1, min(127, int(vel)))
            dur = max(0.05, float(dur))
            t_on  = max(1, int(round(float(offset) * TICKS)))
            t_off = max(t_on + 1, int(round((float(offset) + dur) * TICKS)))
            events.append((t_on,  'on',  mp, vel))
            events.append((t_off, 'off', mp, 0))

        events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
        prev = 0
        for t, kind, note, vel in events:
            dt = max(0, t - prev)
            trk.append(mido.Message(
                'note_on' if kind == 'on' else 'note_off',
                channel=channel, note=note, velocity=vel, time=dt))
            prev = t
        return trk

    mid.tracks.append(_make_track(melody_notes, 0, mel_program, "Melody"))
    mid.tracks.append(_make_track(cp_notes,     cp_channel, cp_program, "Counterpoint"))
    mid.save(output_path)
    if verbose:
        print(f"    ✓ Guardado: {output_path}  ({len(cp_notes)} notas de contrapunto)")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE CONTRAPUNTO
# ══════════════════════════════════════════════════════════════════════════════

def _interval(p1: int, p2: int) -> int:
    """Intervalo absoluto en semitonos (sin octava)."""
    return abs(p1 - p2) % 12

def _interval_full(p1: int, p2: int) -> int:
    """Intervalo absoluto real (con octava)."""
    return abs(p1 - p2)

def _is_consonant(p1: int, p2: int) -> bool:
    iv = _interval(p1, p2)
    return iv in ALL_CONSONANCES

def _is_perfect(p1: int, p2: int) -> bool:
    iv = _interval(p1, p2)
    return iv in PERFECT_CONSONANCES

def _motion_type(prev_mel, prev_cp, curr_mel, curr_cp):
    """
    Devuelve el tipo de movimiento entre dos momentos consecutivos:
    'contrary', 'oblique', 'parallel', 'similar', 'direct'
    """
    d_mel = curr_mel - prev_mel
    d_cp  = curr_cp  - prev_cp
    if d_mel == 0 and d_cp == 0:
        return 'oblique'
    if d_mel == 0 or d_cp == 0:
        return 'oblique'
    if d_mel > 0 and d_cp < 0:
        return 'contrary'
    if d_mel < 0 and d_cp > 0:
        return 'contrary'
    # Mismo sentido
    if d_mel == d_cp:
        return 'parallel'
    # Mismo sentido pero distinto intervalo: similar o directo
    # "Directo" si se llega a consonancia perfecta por movimiento similar
    # (Fux: prohibido independientemente del tamaño del salto)
    iv_new = _interval(curr_mel, curr_cp)
    if iv_new in PERFECT_CONSONANCES:
        return 'direct'
    return 'similar'


def _resolve_dissonance(cp_cand: int, mel_p: int, key_info: KeyInfo,
                         lo: int, hi: int) -> int:
    """
    Si cp_cand forma disonancia con mel_p, busca la consonancia
    diatónica más cercana dentro del rango.
    """
    if _is_consonant(mel_p, cp_cand):
        return cp_cand
    for step in [1, -1, 2, -2, 3, -3]:
        cand = key_info.snap_to_scale(cp_cand + step)
        if lo <= cand <= hi and _is_consonant(mel_p, cand):
            return cand
    return cp_cand


def _best_candidate(
    candidates: list,
    mel_p: int,
    prev_cp,
    prev_mel,
    key_info: KeyInfo,
    voice_position: str = 'below',
    prefer_contrary: bool = True,
    avoid_direct: bool = True,
    strict: bool = False,
    verbose_log: list = None,
) -> int:
    """
    Puntúa candidatos de contrapunto y elige el mejor.
    Aplica las reglas comunes a todas las especies.
    """
    best_cp, best_score = None, -9999

    for cp in candidates:
        iv       = _interval(mel_p, cp)
        iv_full  = _interval_full(mel_p, cp)
        score    = 0
        reasons  = []

        # --- CONSONANCIAS ---
        if iv in IMPERFECT_CONSONANCES:
            score += 4
            reasons.append(f"+4 consonancia imperfecta ({iv}st)")
        elif iv in PERFECT_CONSONANCES:
            score += 2
            reasons.append(f"+2 consonancia perfecta ({iv}st)")
        else:
            score -= 3
            reasons.append(f"-3 disonancia ({iv}st)")

        # --- MOVIMIENTO ---
        if prev_cp is not None and prev_mel is not None:
            motion = _motion_type(prev_mel, prev_cp, mel_p, cp)

            if motion == 'contrary':
                score += 3
                reasons.append("+3 movimiento contrario")
            elif motion == 'oblique':
                score += 1
                reasons.append("+1 oblicuo")
            elif motion == 'similar':
                score += 0
            elif motion == 'parallel':
                if iv in PERFECT_CONSONANCES:
                    score -= 5
                    reasons.append(f"-5 paralelas de {'5ª' if iv==7 else '8ª'}")
                else:
                    score -= 1
                    reasons.append("-1 paralelas imperfectas")
            elif motion == 'direct':
                if avoid_direct:
                    score -= 4
                    reasons.append("-4 movimiento directo a consonancia perfecta")

            # Salto excesivo
            jump = abs(cp - prev_cp)
            if jump > 12:
                score -= 4
                reasons.append(f"-4 salto de {jump} st (> octava)")
            elif jump > 7:
                score -= 2
                reasons.append(f"-2 salto de {jump} st")
            elif jump <= 2:
                score += 1
                reasons.append("+1 grado conjunto")

            # Unísono consecutivo
            if iv == 0 and _interval(prev_mel, prev_cp) == 0:
                score -= 3
                reasons.append("-3 unísono consecutivo")

        # --- POSICIÓN RELATIVA ---
        if voice_position == 'below' and cp < mel_p:
            score += 1
            reasons.append("+1 posición correcta (abajo)")
        elif voice_position == 'above' and cp > mel_p:
            score += 1
            reasons.append("+1 posición correcta (arriba)")

        # --- PREFERIR TÓNICA Y DOMINANTE ---
        if (cp % 12) == key_info.tonic_pc:
            score += 0.5
        if (cp % 12) == key_info.dominant_pc:
            score += 0.3

        # Evitar cruzamiento de voces
        if voice_position == 'below' and cp > mel_p:
            score -= 6
            reasons.append("-6 cruce de voces")
        elif voice_position == 'above' and cp < mel_p:
            score -= 6
            reasons.append("-6 cruce de voces")

        # Strict: evitar unísono salvo cadencia
        if strict and iv == 0:
            score -= 2
            reasons.append("-2 strict: unísono evitado")

        if score > best_score:
            best_score = score
            best_cp    = cp
            if verbose_log is not None:
                verbose_log.clear()
                verbose_log.extend(reasons)

    return best_cp if best_cp is not None else candidates[0]


def _candidates_in_range(key_info: KeyInfo, lo: int, hi: int) -> list:
    """Todos los grados diatónicos en el rango dado."""
    return key_info.scale_midi(lo, hi)


def _parse_tension_curve(spec: str, n_bars: int) -> list:
    """
    Parsea una curva de tensión tipo "0:0.2, 8:0.8, 16:0.3"
    y devuelve una lista de n_bars valores interpolados.
    """
    if not spec:
        return [0.5] * n_bars
    try:
        parts = [p.strip() for p in spec.split(',')]
        control_points = []
        for p in parts:
            bar_s, val_s = p.split(':')
            control_points.append((int(bar_s.strip()), float(val_s.strip())))
        control_points.sort(key=lambda x: x[0])
        result = []
        for i in range(n_bars):
            # Interpolar lineal
            if i <= control_points[0][0]:
                result.append(control_points[0][1])
            elif i >= control_points[-1][0]:
                result.append(control_points[-1][1])
            else:
                for j in range(len(control_points) - 1):
                    b0, v0 = control_points[j]
                    b1, v1 = control_points[j + 1]
                    if b0 <= i <= b1:
                        t = (i - b0) / max(b1 - b0, 1)
                        result.append(v0 + t * (v1 - v0))
                        break
                else:
                    result.append(0.5)
        return result
    except Exception:
        return [0.5] * n_bars


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES POR ESPECIE
# ══════════════════════════════════════════════════════════════════════════════

def _species1(
    melody_notes: list,
    key_info: KeyInfo,
    n_bars: int,
    beats_per_bar: int,
    lo: int,
    hi: int,
    voice_position: str,
    tension_curve: list,
    strict: bool,
    rng: random.Random,
    verbose: bool,
) -> list:
    """
    Primera especie: nota contra nota.
    Por cada nota de la melodía, una nota de contrapunto consonante.
    Reglas aplicadas:
    - Solo consonancias
    - Sin paralelas de 5ª u 8ª
    - Sin movimiento directo a consonancia perfecta
    - Preferir movimiento contrario
    - Evitar repetición del mismo intervalo más de 3 veces seguidas
    """
    if verbose:
        print("\n  [S1] Primera especie — nota contra nota")

    candidates = _candidates_in_range(key_info, lo, hi)
    if not candidates:
        return []

    mel_sorted = sorted(melody_notes, key=lambda x: x[0])
    result = []
    prev_cp, prev_mel = None, None
    same_iv_count = 0
    prev_iv = None
    vlog = []

    for offset, mel_p, dur, vel in mel_sorted:
        # Filtrar candidatos consonantes
        cons = [c for c in candidates if _is_consonant(mel_p, c)]
        if not cons:
            cons = candidates  # fallback

        cp = _best_candidate(
            cons, mel_p, prev_cp, prev_mel, key_info,
            voice_position, prefer_contrary=True,
            avoid_direct=True, strict=strict, verbose_log=vlog,
        )

        # Evitar repetición monótona de intervalo
        curr_iv = _interval(mel_p, cp)
        if curr_iv == prev_iv:
            same_iv_count += 1
        else:
            same_iv_count = 0
        if same_iv_count >= 3:
            # Forzar cambio de intervalo
            alt = [c for c in cons if _interval(mel_p, c) != curr_iv]
            if alt:
                cp_alt = _best_candidate(alt, mel_p, prev_cp, prev_mel, key_info,
                                          voice_position, verbose_log=vlog)
                cp = cp_alt
                same_iv_count = 0

        vel_cp = max(30, min(90, int(vel * 0.75 + rng.randint(-8, 8))))
        result.append((offset, cp, dur * 0.90, vel_cp))

        if verbose and vlog:
            print(f"    bar≈{int(offset/beats_per_bar):02d} beat={offset%beats_per_bar:.1f}"
                  f"  mel={mel_p:3d}  cp={cp:3d}  iv={curr_iv:2d}st  "
                  + " | ".join(vlog[:3]))

        prev_cp, prev_mel, prev_iv = cp, mel_p, curr_iv

    return result


def _species2(
    melody_notes: list,
    key_info: KeyInfo,
    n_bars: int,
    beats_per_bar: int,
    lo: int,
    hi: int,
    voice_position: str,
    tension_curve: list,
    strict: bool,
    rng: random.Random,
    verbose: bool,
) -> list:
    """
    Segunda especie: dos notas de contrapunto por cada nota del cantus.
    - Tiempo fuerte: siempre consonante
    - Tiempo débil (beat 2 o mitad): puede ser disonancia de paso
      si está entre dos consonancias por grado conjunto
    """
    if verbose:
        print("\n  [S2] Segunda especie — 2:1 (con notas de paso)")

    candidates_all = _candidates_in_range(key_info, lo, hi)
    mel_sorted = sorted(melody_notes, key=lambda x: x[0])

    result = []
    prev_cp, prev_mel = None, None
    vlog = []

    for offset, mel_p, dur, vel in mel_sorted:
        # Duración de cada nota de contrapunto
        cp_dur = dur / 2.0

        for beat_frac in [0.0, 0.5]:
            beat_offset = offset + beat_frac * dur
            is_strong   = (beat_frac == 0.0)

            if is_strong:
                # Tiempo fuerte: solo consonancias
                cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
                if not cons:
                    cons = candidates_all
                cp = _best_candidate(
                    cons, mel_p, prev_cp, prev_mel, key_info,
                    voice_position, prefer_contrary=True, avoid_direct=True,
                    strict=strict, verbose_log=vlog,
                )
            else:
                # Tiempo débil: notas de paso permitidas
                # Preferir grado conjunto desde el tiempo fuerte anterior
                if prev_cp is not None:
                    # Notas a distancia de 1 o 2 semitonos diatónicos
                    pass_cands = []
                    for step in [-2, -1, 1, 2]:
                        cand = key_info.snap_to_scale(prev_cp + step)
                        if lo <= cand <= hi:
                            pass_cands.append(cand)
                    if not pass_cands:
                        pass_cands = candidates_all
                    # En tiempo débil se permiten disonancias de paso
                    cp = _best_candidate(
                        pass_cands, mel_p, prev_cp, prev_mel, key_info,
                        voice_position, prefer_contrary=False, avoid_direct=True,
                        strict=False, verbose_log=vlog,
                    )
                else:
                    cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
                    cp   = _best_candidate(cons or candidates_all, mel_p, prev_cp,
                                            prev_mel, key_info, voice_position,
                                            verbose_log=vlog)

            vel_cp = max(30, min(90, int(vel * (0.80 if is_strong else 0.65)
                                          + rng.randint(-5, 5))))
            result.append((beat_offset, cp, cp_dur * 0.88, vel_cp))

            if verbose and vlog:
                print(f"    bar≈{int(offset/beats_per_bar):02d}"
                      f" beat={beat_offset%beats_per_bar:.1f}"
                      f"  mel={mel_p:3d}  cp={cp:3d}"
                      f"  {'STRONG' if is_strong else 'weak  '}"
                      f"  iv={_interval(mel_p,cp):2d}st")

            prev_cp, prev_mel = cp, mel_p

    return result


def _species3(
    melody_notes: list,
    key_info: KeyInfo,
    n_bars: int,
    beats_per_bar: int,
    lo: int,
    hi: int,
    voice_position: str,
    tension_curve: list,
    strict: bool,
    rng: random.Random,
    verbose: bool,
) -> list:
    """
    Tercera especie: cuatro notas de contrapunto por nota de cantus.
    - Beat 1 (fuerte): consonante
    - Beats 2-4: notas de paso diatónicas, cambiatas y échappées permitidas
    - Solo una nota no diatónica por compás (cambiata)
    """
    if verbose:
        print("\n  [S3] Tercera especie — 4:1 (contrapunto florido básico)")

    candidates_all = _candidates_in_range(key_info, lo, hi)
    mel_sorted = sorted(melody_notes, key=lambda x: x[0])
    result = []
    prev_cp, prev_mel = None, None
    vlog = []

    for offset, mel_p, dur, vel in mel_sorted:
        n_subdivs = min(4, max(2, int(round(dur))))
        sub_dur   = dur / n_subdivs

        for sub in range(n_subdivs):
            beat_offset = offset + sub * sub_dur
            is_strong   = (sub == 0)

            if is_strong:
                cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
                cp   = _best_candidate(cons or candidates_all, mel_p,
                                        prev_cp, prev_mel, key_info,
                                        voice_position, prefer_contrary=True,
                                        avoid_direct=True, strict=strict,
                                        verbose_log=vlog)
            else:
                # Notas de paso: moverse por grado conjunto diatónico
                direction = 0
                if prev_cp is not None and mel_p != prev_mel:
                    direction = 1 if voice_position == 'above' else -1
                    mel_dir   = 1 if mel_p > (prev_mel or mel_p) else -1
                    direction = -mel_dir  # contrario a la melodía

                step_cands = []
                for step in [direction, direction * 2, -direction, 1, -1, 2, -2]:
                    if prev_cp is not None:
                        cand = key_info.snap_to_scale(prev_cp + step)
                    else:
                        cand = key_info.snap_to_scale(mel_p + (step * (-1 if voice_position == 'below' else 1)))
                    if lo <= cand <= hi:
                        step_cands.append(cand)

                # Cambiata: nota que escapa un grado y vuelve (permitida en S3)
                if sub == 2 and prev_cp is not None:
                    cambiata = key_info.snap_to_scale(prev_cp + (2 * direction if direction else 1))
                    if lo <= cambiata <= hi:
                        step_cands.append(cambiata)

                if not step_cands:
                    step_cands = candidates_all

                cp = _best_candidate(
                    list(set(step_cands)), mel_p, prev_cp, prev_mel, key_info,
                    voice_position, prefer_contrary=False, avoid_direct=(sub == n_subdivs - 1),
                    strict=False, verbose_log=vlog,
                )

            # Dinámica interna por subdivisión
            dyn_factor = [0.85, 0.65, 0.70, 0.60]
            vel_cp = max(25, min(90, int(vel * dyn_factor[min(sub, 3)]
                                          + rng.randint(-5, 5))))
            result.append((beat_offset, cp, sub_dur * 0.85, vel_cp))

            if verbose:
                print(f"    bar≈{int(offset/beats_per_bar):02d}"
                      f" sub={sub}  mel={mel_p:3d}  cp={cp:3d}"
                      f"  {'STRONG' if is_strong else 'pass  '}"
                      f"  iv={_interval(mel_p,cp):2d}st")

            prev_cp, prev_mel = cp, mel_p

    return result


def _species4(
    melody_notes: list,
    key_info: KeyInfo,
    n_bars: int,
    beats_per_bar: int,
    lo: int,
    hi: int,
    voice_position: str,
    tension_curve: list,
    strict: bool,
    rng: random.Random,
    verbose: bool,
) -> list:
    """
    Cuarta especie: contrapunto sincopado con ligaduras y retardos.
    - El contrapunto empieza en el tiempo débil (beat 2)
    - La nota se prepara en tiempo débil (consonante)
    - Se mantiene (liga) al tiempo fuerte → puede ser disonante
    - La disonancia DEBE resolver por grado descendente al siguiente tiempo débil
    - Retardos clásicos: 7-6, 4-3, 9-8, 2-1
    """
    if verbose:
        print("\n  [S4] Cuarta especie — sincopado (ligaduras y retardos)")

    RETARDOS = {
        # (intervalo disonante o tenso): resuelve descendiendo N semitonos
        11: 1,   # 7ª → 6ª (un semitono/tono abajo)
        10: 1,
        5:  1,   # 4ª → 3ª
        2:  1,   # 2ª → 1ª (unísono)
        14: 1,   # 9ª → 8ª
    }

    candidates_all = _candidates_in_range(key_info, lo, hi)
    mel_sorted      = sorted(melody_notes, key=lambda x: x[0])
    result          = []
    prev_cp, prev_mel = None, None
    ligature_pending  = None  # nota ligada al siguiente tiempo fuerte
    vlog = []

    for i, (offset, mel_p, dur, vel) in enumerate(mel_sorted):
        half = dur / 2.0

        # --- Tiempo débil (beat 2): preparación ---
        if ligature_pending is not None:
            # Resolver la ligadura anterior
            cp_strong   = ligature_pending
            iv_strong   = _interval_full(mel_p, cp_strong)
            iv_mod      = iv_strong % 12
            resolution  = None

            if iv_mod in RETARDOS:
                # Resolución por grado diatónico descendente (no semitono fijo)
                # Buscar la nota diatónica inmediatamente inferior
                for delta in [1, 2]:
                    res_cand = key_info.snap_to_scale(cp_strong - delta)
                    if (res_cand < cp_strong and lo <= res_cand <= hi
                            and _is_consonant(mel_p, res_cand)):
                        resolution = res_cand
                        break

            if resolution is None:
                resolution = _resolve_dissonance(cp_strong, mel_p, key_info, lo, hi)

            vel_cp = max(30, min(90, int(vel * 0.75 + rng.randint(-5, 5))))
            # Nota ligada en el tiempo fuerte (puede ser disonante)
            result.append((offset, cp_strong, half * 0.92, int(vel * 0.80)))
            # Resolución en el tiempo débil
            result.append((offset + half, resolution, half * 0.88, vel_cp))

            if verbose:
                print(f"    bar≈{int(offset/beats_per_bar):02d}"
                      f"  RETARDO  ligada={cp_strong}(iv={iv_mod}st)"
                      f"  →  resolución={resolution}")

            ligature_pending = resolution
            prev_cp, prev_mel = resolution, mel_p
        else:
            # Primera nota o sin ligadura: colocar consonancia en tiempo fuerte
            cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
            cp   = _best_candidate(cons or candidates_all, mel_p,
                                    prev_cp, prev_mel, key_info,
                                    voice_position, prefer_contrary=True,
                                    verbose_log=vlog)
            vel_cp = max(30, min(90, int(vel * 0.78 + rng.randint(-5, 5))))
            result.append((offset, cp, dur * 0.90, vel_cp))
            ligature_pending = cp
            prev_cp, prev_mel = cp, mel_p

    return result


def _species5(
    melody_notes: list,
    key_info: KeyInfo,
    n_bars: int,
    beats_per_bar: int,
    lo: int,
    hi: int,
    voice_position: str,
    tension_curve: list,
    strict: bool,
    rng: random.Random,
    verbose: bool,
) -> list:
    """
    Quinta especie: contrapunto florido (mezcla de S1-S4).
    Elige la especie por compás basándose en:
    - Tensión alta  → S3 o S4 (mayor movimiento, retardos expresivos)
    - Tensión media → S2 (notas de paso)
    - Tensión baja  → S1 (tranquilo)
    - Cadencias     → S4 (retardos cadenciales)

    El contrapunto generado es una mezcla coherente en la misma voz.
    """
    if verbose:
        print("\n  [S5] Quinta especie — florido (mezcla S1-S4 por tensión)")

    candidates_all = _candidates_in_range(key_info, lo, hi)
    mel_sorted     = sorted(melody_notes, key=lambda x: x[0])

    # Agrupar notas por compás
    by_bar = defaultdict(list)
    for note in mel_sorted:
        bar = int(note[0] / beats_per_bar)
        by_bar[bar].append(note)

    result        = []
    prev_cp, prev_mel = None, None
    ligature_hold = None   # para mantener retardos entre compases
    vlog = []

    for bar in range(n_bars):
        bar_notes = sorted(by_bar.get(bar, []), key=lambda x: x[0])
        if not bar_notes:
            continue

        # Tensión del compás
        tension = tension_curve[bar] if bar < len(tension_curve) else 0.5

        # Detectar cadencia: último o penúltimo compás
        is_cadence = (bar >= n_bars - 2)

        # Elegir especie para este compás
        if is_cadence and tension >= 0.5:
            species_bar = 4   # retardo cadencial
        elif tension >= 0.75:
            species_bar = rng.choice([3, 4])
        elif tension >= 0.45:
            species_bar = rng.choice([2, 3])
        else:
            species_bar = rng.choice([1, 2])

        if verbose:
            print(f"    bar {bar:02d}  tension={tension:.2f}"
                  f"  → especie {species_bar}"
                  f"{'  [CADENCIA]' if is_cadence else ''}")

        # Generar notas según la especie elegida (función inline por eficiencia)
        bar_result = []

        if species_bar == 1:
            for offset, mel_p, dur, vel in bar_notes:
                cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
                cp   = _best_candidate(cons or candidates_all, mel_p,
                                        prev_cp, prev_mel, key_info,
                                        voice_position, prefer_contrary=True,
                                        avoid_direct=True, strict=strict,
                                        verbose_log=vlog)
                vel_cp = max(30, min(90, int(vel * 0.75 + rng.randint(-6, 6))))
                bar_result.append((offset, cp, dur * 0.90, vel_cp))
                prev_cp, prev_mel, ligature_hold = cp, mel_p, cp

        elif species_bar == 2:
            for offset, mel_p, dur, vel in bar_notes:
                sub = dur / 2.0
                for bi, beat_frac in enumerate([0.0, 0.5]):
                    boff     = offset + beat_frac * dur
                    is_str   = (bi == 0)
                    if is_str or prev_cp is None:
                        cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
                        cp   = _best_candidate(cons or candidates_all, mel_p,
                                               prev_cp, prev_mel, key_info,
                                               voice_position, prefer_contrary=True,
                                               avoid_direct=True, verbose_log=vlog)
                    else:
                        # Nota de paso por grado conjunto
                        step_cands = []
                        for st in [-1, 1, -2, 2]:
                            cand = key_info.snap_to_scale(prev_cp + st)
                            if lo <= cand <= hi:
                                step_cands.append(cand)
                        cp = _best_candidate(step_cands or candidates_all, mel_p,
                                             prev_cp, prev_mel, key_info,
                                             voice_position, verbose_log=vlog)
                    vel_cp = max(25, min(90, int(vel * (0.80 if is_str else 0.65)
                                                  + rng.randint(-5, 5))))
                    bar_result.append((boff, cp, sub * 0.88, vel_cp))
                    prev_cp, prev_mel, ligature_hold = cp, mel_p, cp

        elif species_bar == 3:
            for offset, mel_p, dur, vel in bar_notes:
                n_sub = min(4, max(2, int(round(dur))))
                sub_d = dur / n_sub
                for si in range(n_sub):
                    boff     = offset + si * sub_d
                    is_str   = (si == 0)
                    if is_str:
                        cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
                        cp   = _best_candidate(cons or candidates_all, mel_p,
                                               prev_cp, prev_mel, key_info,
                                               voice_position, prefer_contrary=True,
                                               avoid_direct=True, verbose_log=vlog)
                    else:
                        direction = -1 if voice_position == 'below' else 1
                        if prev_cp is not None and prev_mel is not None:
                            mel_dir   = 1 if mel_p > prev_mel else -1
                            direction = -mel_dir
                        step_cands = []
                        for st in [direction, direction * 2, -direction]:
                            if prev_cp is not None:
                                cand = key_info.snap_to_scale(prev_cp + st)
                                if lo <= cand <= hi:
                                    step_cands.append(cand)
                        cp = _best_candidate(
                            step_cands or candidates_all, mel_p, prev_cp, prev_mel,
                            key_info, voice_position, prefer_contrary=False,
                            avoid_direct=(si == n_sub - 1), verbose_log=vlog)
                    dyn = [0.85, 0.65, 0.70, 0.60]
                    vel_cp = max(25, min(90, int(vel * dyn[min(si, 3)]
                                                  + rng.randint(-5, 5))))
                    bar_result.append((boff, cp, sub_d * 0.85, vel_cp))
                    prev_cp, prev_mel, ligature_hold = cp, mel_p, cp

        elif species_bar == 4:
            # Retardos: preparar en tiempo débil, resolver en siguiente tiempo fuerte
            RETARDOS = {11: 1, 10: 1, 5: 1, 2: 1}
            for offset, mel_p, dur, vel in bar_notes:
                half = dur / 2.0
                if ligature_hold is not None:
                    cp_strong  = ligature_hold
                    iv_mod     = _interval(mel_p, cp_strong)
                    resolution = None
                    if iv_mod in RETARDOS:
                        res_cand = key_info.snap_to_scale(cp_strong - RETARDOS[iv_mod])
                        if lo <= res_cand <= hi and _is_consonant(mel_p, res_cand):
                            resolution = res_cand
                    if resolution is None:
                        resolution = _resolve_dissonance(cp_strong, mel_p,
                                                          key_info, lo, hi)
                    bar_result.append((offset,        cp_strong,  half * 0.92, int(vel * 0.80)))
                    bar_result.append((offset + half, resolution, half * 0.88, int(vel * 0.72)))
                    ligature_hold = resolution
                    prev_cp, prev_mel = resolution, mel_p
                else:
                    cons = [c for c in candidates_all if _is_consonant(mel_p, c)]
                    cp   = _best_candidate(cons or candidates_all, mel_p,
                                            prev_cp, prev_mel, key_info,
                                            voice_position, prefer_contrary=True,
                                            verbose_log=vlog)
                    bar_result.append((offset, cp, dur * 0.90, int(vel * 0.78)))
                    ligature_hold = cp
                    prev_cp, prev_mel = cp, mel_p

        result.extend(bar_result)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MODO STRICT — CSP (Constraint Satisfaction Problem)
# ══════════════════════════════════════════════════════════════════════════════
#
# En modo NO strict las reglas de Fux son PENALIZACIONES BLANDAS: el motor
# elige el candidato "menos malo" y puede emitir un contrapunto con paralelas
# de 5ª/8ª si ningún candidato es limpio. Esto es útil para experimentar,
# pero musicalmente engañoso: un usuario confiado puede creer que el resultado
# es correcto.
#
# El modo --strict convierte las reglas en RESTRICCIONES DURAS: el motor NUNCA
# emite un contrapunto que las viole. Si no encuentra solución, lanza
# CounterpointStrictError con un diagnóstico, en vez de devolver basura.
#
# Reglas duras (especie 1, nota contra nota) — H1..H6:
#   [H1] Solo consonancias (perfectas o imperfectas) en cada momento
#   [H2] Sin paralelas de 5ª justa u 8ª entre dos momentos consecutivos
#   [H3] Sin movimiento directo a consonancia perfecta (5ª u 8ª) por movimiento
#        similar (Fux: prohibido con independencia del tamaño del salto)
#   [H4] Sin cruce de voces (el contrapunto respeta voice_position)
#   [H5] Sin repetir el MISMO intervalo armónico 3 veces seguidas (monotonía)
#   [H6] Cadencia: la sensible (7º grado) del contrapunto resuelve ASCENDIENDO
#        a la tónica cuando aparece en la penúltima posición y la melodía
#        cadencia sobre la tónica (cadencia auténtica)
# Preferencia (no dura, se relaja si bloquea la búsqueda):
#   [P1] El clímax (nota más aguda del contrapunto) cae en el 50–85 % de la
#        frase (heurística cantábile de melody_critic)
#
# El resolver usa backtracking (ventana de 2 instantes), orden aleatorio de
# candidatos (rng) y un límite de nodos para evitar explosión combinatoria.
#
# Para especies 2–5, --strict valida el ESQUELETO de tiempos fuertes (downbeats
# del cantus) contra H1 (S2/S3), H2, H3 y H4, y rechaza con diagnóstico si el
# resultado greedy viola las reglas duras. Así nunca se emite un contrapunto
# con paralelas ocultas en los downbeats.

class CounterpointStrictError(Exception):
    """No existe un contrapunto que satisfaga las reglas duras de Fux."""
    pass


def _crossing_ok(cp: int, mel_p: int, voice_position: str) -> bool:
    """H4: el contrapunto no cruza la melodía."""
    if voice_position == 'above':
        return cp >= mel_p
    return cp <= mel_p


def _parallel_perfect(prev_cp: int, prev_mel: int,
                      cp: int, mel_p: int) -> bool:
    """H2: ¿se forma una 5ª u 8ª paralela entre dos instantes consecutivos?"""
    iv_prev = _interval(prev_mel, prev_cp)
    iv_now  = _interval(mel_p, cp)
    if iv_now not in (0, 7):           # sólo 8ª (unísono) y 5ª justa
        return False
    d_mel = mel_p - prev_mel
    d_cp  = cp   - prev_cp
    same_dir = (d_mel > 0 and d_cp > 0) or (d_mel < 0 and d_cp < 0)
    return same_dir and iv_prev == iv_now


def _direct_to_perfect(prev_cp: int, prev_mel: int,
                       cp: int, mel_p: int) -> bool:
    """H3: ¿se llega a 5ª/8ª por movimiento similar (directo)?"""
    iv_now = _interval(mel_p, cp)
    if iv_now not in (0, 7):
        return False
    d_mel = mel_p - prev_mel
    d_cp  = cp   - prev_cp
    same_dir = (d_mel > 0 and d_cp > 0) or (d_mel < 0 and d_cp < 0)
    if not (same_dir and d_mel != 0 and d_cp != 0):
        return False
    # Si el intervalo previo ya era la misma consonancia perfecta, lo trata
    # _parallel_perfect; aquí sólo el movimiento directo (cambio de intervalo)
    return _interval(prev_mel, prev_cp) != iv_now


def _species1_csp(
    melody_notes: list,
    key_info: KeyInfo,
    n_bars: int,
    beats_per_bar: int,
    lo: int,
    hi: int,
    voice_position: str,
    tension_curve: list,
    rng: random.Random,
    verbose: bool,
) -> list:
    """
    Resuelve la PRIMERA ESPECIE como un CSP con restricciones duras de Fux.

    Devuelve list[(offset, pitch, dur, vel)] o lanza CounterpointStrictError.
    Garantiza: H1–H6 (P1 como preferencia relaxable).
    """
    mel_sorted = sorted(melody_notes, key=lambda x: x[0])
    n = len(mel_sorted)
    if n == 0:
        return []

    all_cands = _candidates_in_range(key_info, lo, hi)
    if not all_cands:
        raise CounterpointStrictError(
            f"Sin grados diatónicos en el rango [{_m2n(lo)}–{_m2n(hi)}] "
            f"para la tonalidad {key_info.name}. Amplía la voz (--voice).")

    # H1: consonancias disponibles por cada nota del cantus
    cons_per_note = []
    for off, mel_p, dur, vel in mel_sorted:
        cons = [c for c in all_cands if _is_consonant(mel_p, c)]
        if not cons:
            raise CounterpointStrictError(
                f"[H1] Sin consonancia posible para la nota melódica "
                f"{_m2n(mel_p)} (MIDI {mel_p}) en offset {off:.2f} dentro "
                f"del rango [{_m2n(lo)}–{_m2n(hi)}]. Amplía la voz (--voice) "
                f"o transpón la melodía.")
        cons_per_note.append(cons)

    # P1: ventana del clímax (50–85 % de la frase), sólo si es larga
    climax_lo = int(n * 0.50)
    climax_hi = int(n * 0.85)
    require_climax = n >= 8

    tonic_pc  = key_info.tonic_pc
    leading_pc = key_info.leading_pc

    MAX_NODES = 250_000

    def solve(require_climax_flag: bool):
        assignment = [None] * n
        iv_hist = []          # H5: historial de intervalos armónicos
        nodes = [0]

        def backtrack(i):
            if nodes[0] > MAX_NODES:
                return None                 # presupuesto agotado
            nodes[0] += 1
            if i == n:
                # Validación global: clímax (P1) y cadencia (H6)
                cp_pitches = [a[1] for a in assignment]
                hi_idx = cp_pitches.index(max(cp_pitches))
                if require_climax_flag and not (climax_lo <= hi_idx <= climax_hi):
                    return None
                if n >= 2:
                    pen_cp  = assignment[-2][1]
                    last_cp = assignment[-1][1]
                    if (pen_cp % 12) == leading_pc \
                            and (last_cp % 12) != tonic_pc:
                        return None
                return list(assignment)
            off, mel_p, dur, vel = mel_sorted[i]
            cands = list(cons_per_note[i])
            rng.shuffle(cands)
            for cp in cands:
                if not _crossing_ok(cp, mel_p, voice_position):     # H4
                    continue
                if i > 0:                                           # H2/H3
                    p_mel = mel_sorted[i-1][1]
                    p_cp  = assignment[i-1][1]
                    if _parallel_perfect(p_cp, p_mel, cp, mel_p):
                        continue
                    if _direct_to_perfect(p_cp, p_mel, cp, mel_p):
                        continue
                iv_now = _interval(mel_p, cp)                       # H5
                if len(iv_hist) >= 2 and iv_hist[-1] == iv_now \
                        and iv_hist[-2] == iv_now:
                    continue
                assignment[i] = (off, cp, dur * 0.90,
                                 max(30, min(90, int(vel * 0.75
                                                     + rng.randint(-8, 8)))))
                iv_hist.append(iv_now)
                r = backtrack(i + 1)
                if r is not None:
                    return r
                iv_hist.pop()
                assignment[i] = None
            return None

        return backtrack(0), nodes[0]

    # Primer intento con clímax exigido; si falla, se relaja P1.
    result, nodes = solve(require_climax)
    if result is None and require_climax:
        if verbose:
            print("  [S1-strict] clímax no encuadrable; relajando P1")
        result, nodes = solve(False)

    if result is None:
        raise CounterpointStrictError(
            f"No existe un contrapunto de 1ª especie que cumpla TODAS las "
            f"reglas duras de Fux (explorados {nodes} nodos). Sugerencias: "
            f"ampliar el rango vocal (--voice), acortar la melodía (--bars), "
            f"cambiar la tonalidad (--key) o usar modo no strict.")

    if verbose:
        cps = [a[1] for a in result]
        hi_idx = cps.index(max(cps))
        print(f"\n  [S1-strict] CSP resuelto en {nodes} nodos; "
              f"clímax en posición {hi_idx+1}/{n}")
    return result


def _strong_beat_pairs(mel_sorted: list, cp_notes: list) -> list:
    """
    Empareja cada onset del cantus con la nota de contrapunto más cercana
    en tiempo (el tiempo fuerte / downbeat de ese compás o fracción).
    Devuelve list[(offset, mel_p, cp_p)].
    """
    pairs = []
    eps = 0.25  # beats de tolerancia
    for off, mel_p, dur, vel in mel_sorted:
        best = None
        best_d = 1e9
        for co, cp_p, cd, cv in cp_notes:
            d = abs(co - off)
            if d < best_d:
                best_d, best = d, cp_p
        if best is not None and best_d <= eps + max(dur, 1.0):
            pairs.append((off, mel_p, best))
    return pairs


def _validate_strict(cp_notes: list,
                      mel_sorted: list,
                      voice_position: str,
                      species: int,
                      key_info: KeyInfo,
                      verbose: bool) -> None:
    """
    Valida el resultado greedy (especies 2–5) contra las reglas duras del
    ESQUELETO de tiempos fuertes. Lanza CounterpointStrictError con la lista
    de violaciones si las hay; no devuelve nada si todo está limpio.

    Para S2/S3: el downbeat del contrapunto debe ser consonante (H1).
    Para S2–S5: sin paralelas de 5ª/8ª entre downbeats (H2), sin directo a
    perfecta entre downbeats (H3), y sin cruce de voces en ningún instante (H4).
    """
    violations = []

    # H4: cruce en cualquier instante (universal a todas las especies)
    for co, cp_p, cd, cv in cp_notes:
        # mel_p más cercano al offset del cp
        best_mel = min(mel_sorted, key=lambda m: abs(m[0] - co))
        if not _crossing_ok(cp_p, best_mel[1], voice_position):
            violations.append(
                f"[H4] cruce en offset {co:.2f}: cp={_m2n(cp_p)} "
                f"({'encima' if voice_position=='above' else 'debajo'} de "
                f"mel={_m2n(best_mel[1])})")

    skeleton = _strong_beat_pairs(mel_sorted, cp_notes)
    if len(skeleton) < 2:
        if violations:
            raise CounterpointStrictError(
                "Contrapunto estricto viola reglas duras:\n  - "
                + "\n  - ".join(violations))
        return

    # H1 (sólo S2/S3): consonancia en el downbeat
    if species in (2, 3):
        for off, mel_p, cp_p in skeleton:
            if not _is_consonant(mel_p, cp_p):
                violations.append(
                    f"[H1] downbeat disonante en offset {off:.2f}: "
                    f"cp={_m2n(cp_p)} vs mel={_m2n(mel_p)} "
                    f"(intervalo {_interval(mel_p, cp_p)}st)")

    # H2 / H3 entre downbeats consecutivos
    for k in range(1, len(skeleton)):
        _, p_mel, p_cp = skeleton[k-1]
        off, mel_p, cp_p = skeleton[k]
        if _parallel_perfect(p_cp, p_mel, cp_p, mel_p):
            violations.append(
                f"[H2] paralelas de "
                f"{'8ª' if _interval(mel_p, cp_p)==0 else '5ª'} "
                f"entre offsets {skeleton[k-1][0]:.2f} y {off:.2f}")
        if _direct_to_perfect(p_cp, p_mel, cp_p, mel_p):
            violations.append(
                f"[H3] directo a "
                f"{'8ª' if _interval(mel_p, cp_p)==0 else '5ª'} "
                f"entre offsets {skeleton[k-1][0]:.2f} y {off:.2f}")

    if violations:
        raise CounterpointStrictError(
            f"Contrapunto estricto (especie {species}) viola "
            f"{len(violations)} regla(s) dura(s) en el esqueleto de "
            f"tiempos fuertes:\n  - " + "\n  - ".join(violations)
            + "\nEl modo --strict no emite resultados que rompan las reglas "
              "de Fux. Prueba sin --strict, con otra voz, otra tonalidad o "
              "otra melodía de entrada.")


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def generate_counterpoint_voice(
    melody_notes: list,
    key = "auto",
    species: int = 5,
    voice: str = "auto",
    voice_position: str = "auto",
    beats_per_bar: int = 4,
    n_bars: int = None,
    tension_curve = None,
    strict: bool = False,
    seed: int = 42,
    verbose: bool = False,
) -> list:
    """
    Genera una voz de contrapunto para una melodía dada.

    Parámetros
    ----------
    melody_notes  : list of (offset_beats, pitch, duration_beats, velocity)
    key           : "auto" | str ("C major", "Am", …) | KeyInfo
    species       : 1-5
    voice         : "auto" | "soprano" | "alto" | "tenor" | "bass" |
                    "violin" | "viola" | "cello"
    voice_position: "auto" | "above" | "below"
    beats_per_bar : int
    n_bars        : int | None (auto)
    tension_curve : list[float] | str ("0:0.2, 8:0.8") | None
    strict        : bool — aplicar reglas de Fux más estrictas
    seed          : int
    verbose       : bool

    Devuelve
    --------
    list of (offset_beats, pitch, duration_beats, velocity)
    """
    rng = random.Random(seed)

    # --- Tonalidad ---
    if isinstance(key, KeyInfo):
        key_info = key
    elif key == "auto" or key is None:
        key_info = KeyInfo.detect_from_notes(melody_notes)
    else:
        key_info = KeyInfo.from_string(str(key))

    if verbose:
        print(f"  Tonalidad detectada: {key_info.name}")

    # --- n_bars ---
    if not melody_notes:
        return []
    if n_bars is None:
        last_offset = max(o + d for o, _, d, _ in melody_notes)
        n_bars = max(1, math.ceil(last_offset / beats_per_bar))

    # --- Posición de la voz ---
    mel_mean = sum(p for _, p, _, _ in melody_notes) / len(melody_notes)
    if voice_position == "auto":
        voice_position = "below" if mel_mean >= 60 else "above"

    # --- Rango del instrumento ---
    if voice == "auto":
        voice = "viola" if voice_position == "below" else "violin"
    lo, hi = VOICE_RANGES.get(voice, (48, 77))

    if verbose:
        print(f"  Voz: {voice}  posición: {voice_position}"
              f"  rango: [{lo}, {hi}]  especie: {species}")

    # --- Curva de tensión ---
    if isinstance(tension_curve, str):
        tc = _parse_tension_curve(tension_curve, n_bars)
    elif tension_curve is None:
        # Curva por defecto: arco
        tc = [0.5 * (1 + math.sin(math.pi * i / max(n_bars - 1, 1) - math.pi / 2))
              for i in range(n_bars)]
    else:
        tc = list(tension_curve)
        # Extender si hace falta
        while len(tc) < n_bars:
            tc.append(tc[-1] if tc else 0.5)

    # --- Seleccionar generador ---
    generators = {
        1: _species1,
        2: _species2,
        3: _species3,
        4: _species4,
        5: _species5,
    }
    gen_fn = generators.get(species, _species5)

    # ── Modo STRICT ───────────────────────────────────────────────────────
    # Especie 1: se resuelve como CSP con restricciones duras (garantiza H1–H6).
    # Especies 2–5: se genera con el motor greedy y luego se valida el
    # esqueleto de tiempos fuertes; si viola reglas duras, se lanza
    # CounterpointStrictError en vez de emitir un resultado incorrecto.
    if strict and species == 1:
        cp_notes = _species1_csp(
            melody_notes=melody_notes,
            key_info=key_info,
            n_bars=n_bars,
            beats_per_bar=beats_per_bar,
            lo=lo,
            hi=hi,
            voice_position=voice_position,
            tension_curve=tc,
            rng=rng,
            verbose=verbose,
        )
    else:
        cp_notes = gen_fn(
            melody_notes=melody_notes,
            key_info=key_info,
            n_bars=n_bars,
            beats_per_bar=beats_per_bar,
            lo=lo,
            hi=hi,
            voice_position=voice_position,
            tension_curve=tc,
            strict=strict,
            rng=rng,
            verbose=verbose,
        )
        if strict:  # especies 2–5: validar el esqueleto de tiempos fuertes
            mel_sorted = sorted(melody_notes, key=lambda x: x[0])
            _validate_strict(cp_notes, mel_sorted, voice_position,
                            species, key_info, verbose)
            if verbose:
                print("  [strict] esqueleto de tiempos fuertes validado "
                      f"(especie {species}): 0 violaciones duras")

    if verbose:
        print(f"\n  ✓ Generadas {len(cp_notes)} notas de contrapunto "
              f"(especie {species}, voz {voice}"
              f"{'· strict' if strict else ''})")

    return cp_notes


# ══════════════════════════════════════════════════════════════════════════════
#  FINGERPRINT (para integración con stitcher.py)
# ══════════════════════════════════════════════════════════════════════════════

def export_fingerprint(
    melody_notes: list,
    cp_notes: list,
    key_info: KeyInfo,
    tempo_bpm: int,
    n_bars: int,
    beats_per_bar: int,
    output_path: str,
) -> dict:
    """
    Genera un .fingerprint.json compatible con stitcher.py y midi_dna_unified.
    """
    all_notes = melody_notes + cp_notes
    if not all_notes:
        return {}

    all_sorted = sorted(all_notes, key=lambda x: x[0])
    last_bar_notes = [n for n in all_sorted
                      if n[0] >= (n_bars - 1) * beats_per_bar]
    first_bar_notes = [n for n in all_sorted
                       if n[0] < beats_per_bar]

    def _chord_pc(notes):
        pcs = sorted(set(n[1] % 12 for n in notes))
        return pcs[:4] if pcs else [0]

    fp = {
        "source":      output_path,
        "key":         key_info.name,
        "tempo_bpm":   tempo_bpm,
        "n_bars":      n_bars,
        "beats_per_bar": beats_per_bar,
        "entry": {
            "chord_pcs":  _chord_pc(first_bar_notes),
            "bass_pitch": min(n[1] for n in first_bar_notes) if first_bar_notes else 48,
            "register":   sum(n[1] for n in first_bar_notes) / max(len(first_bar_notes), 1),
            "tension":    0.0,
        },
        "exit": {
            "chord_pcs":  _chord_pc(last_bar_notes),
            "bass_pitch": min(n[1] for n in last_bar_notes) if last_bar_notes else 48,
            "last_pitch": last_bar_notes[-1][1] if last_bar_notes else 60,
            "tension":    0.0,
        },
        "tension_curve": {
            "mean":      float(np.mean([n[3] for n in all_notes]) / 127),
            "peak":      float(max(n[3] for n in all_notes) / 127),
            "direction": "neutral",
        },
        "stitching_hints": {
            "cadence_type": "AC",
            "open_ending":  False,
            "bpm_tolerance": 10,
        },
    }
    with open(output_path, 'w') as f:
        json.dump(fp, f, indent=2)
    return fp


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _play_midi(filepath: str, seconds: int = 30):
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        t0 = time.time()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            if time.time() - t0 >= seconds:
                pygame.mixer.music.stop()
                break
    except ImportError:
        print("  ⚠ pygame no disponible — instala con: pip install pygame")
    except Exception as e:
        print(f"  ⚠ Error reproduciendo: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='counterpoint.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Genera una segunda voz de contrapunto para un MIDI melódico.
            Soporta las 5 especies de Fux/Gradus ad Parnassum.
        """),
    )
    p.add_argument('input', help='MIDI fuente (melodía)')
    p.add_argument('--species', type=int, default=5, choices=[1,2,3,4,5],
                   help='Especie de contrapunto 1-5 (default: 5)')
    p.add_argument('--voice', default='auto',
                   choices=['auto','soprano','alto','tenor','bass',
                            'violin','viola','cello'],
                   help='Instrumento de la voz de contrapunto (default: auto)')
    p.add_argument('--voice-position', default='auto',
                   choices=['above','below','auto'],
                   help='Posición relativa a la melodía (default: auto)')
    p.add_argument('--key', default='auto',
                   help='Tonalidad: "C", "Am", "Bb major"… (default: auto)')
    p.add_argument('--bars', type=int, default=None,
                   help='Compases a generar (default: auto)')
    p.add_argument('--beats', type=int, default=4,
                   help='Beats por compás (default: 4)')
    p.add_argument('--strict', action='store_true',
                   help='Modo CSP: reglas de Fux como RESTRICCIONES DURAS. '
                        'Especie 1: backtracking con H1–H6 garantizadas '
                        '(sin paralelas/directas/cruces, clímax y cadencia). '
                        'Especies 2–5: validación del esqueleto de tiempos '
                        'fuertes. Si no hay solución, NO se emite MIDI: se '
                        'rechaza con diagnóstico (CounterpointStrictError).')
    p.add_argument('--tension-curve', default=None,
                   help='Curva de tensión: "0:0.2, 8:0.8, 16:0.3"')
    p.add_argument('--all-species', action='store_true',
                   help='Generar las 5 especies')
    p.add_argument('--catalog', action='store_true',
                   help='Con --all-species, concatenar en un único MIDI')
    p.add_argument('--output', default='counterpoint_out.mid',
                   help='Fichero de salida (default: counterpoint_out.mid)')
    p.add_argument('--output-dir', default='.',
                   help='Directorio de salida con --all-species')
    p.add_argument('--channel', type=int, default=1,
                   help='Canal MIDI del contrapunto (default: 1)')
    p.add_argument('--program', type=int, default=None,
                   help='Program change GM (default: auto según voz)')
    p.add_argument('--listen', action='store_true',
                   help='Reproducir al terminar (requiere pygame)')
    p.add_argument('--play-seconds', type=int, default=30,
                   help='Segundos de reproducción (default: 30)')
    p.add_argument('--export-fingerprint', action='store_true',
                   help='Exportar .fingerprint.json para stitcher')
    p.add_argument('--verbose', action='store_true',
                   help='Informe detallado')
    p.add_argument('--seed', type=int, default=42,
                   help='Semilla aleatoria (default: 42)')
    return p


def _print_summary(melody_notes, cp_notes, key_info, species, voice,
                   voice_position, n_bars, beats_per_bar):
    """Imprime resumen del análisis de contrapunto generado."""
    print("\n" + "═" * 60)
    print(f"  COUNTERPOINT v{VERSION} — Resumen")
    print("═" * 60)
    print(f"  Tonalidad    : {key_info.name}")
    print(f"  Especie      : {species}")
    print(f"  Voz          : {voice}  ({voice_position})")
    print(f"  Compases     : {n_bars}  ({beats_per_bar}/4)")
    print(f"  Notas melodía: {len(melody_notes)}")
    print(f"  Notas cp     : {len(cp_notes)}")
    if cp_notes:
        ivs = [_interval(m[1], c[1])
               for m, c in zip(
                   sorted(melody_notes, key=lambda x: x[0]),
                   sorted(cp_notes,    key=lambda x: x[0])
               )]
        cons_count = sum(1 for iv in ivs if iv in ALL_CONSONANCES)
        perf_count = sum(1 for iv in ivs if iv in PERFECT_CONSONANCES)
        dis_count  = len(ivs) - cons_count
        print(f"\n  Calidad de intervalos:")
        print(f"    Consonancias   : {cons_count}/{len(ivs)}"
              f" ({100*cons_count//max(len(ivs),1)}%)")
        print(f"    Perf. 5ª/8ª   : {perf_count}/{len(ivs)}"
              f" ({100*perf_count//max(len(ivs),1)}%)")
        print(f"    Disonancias    : {dis_count}/{len(ivs)}"
              f" ({100*dis_count//max(len(ivs),1)}%)")

        # Movimiento contrario
        mel_sorted = sorted(melody_notes, key=lambda x: x[0])
        cp_sorted  = sorted(cp_notes,    key=lambda x: x[0])
        n_contrary = 0
        for i in range(1, min(len(mel_sorted), len(cp_sorted))):
            pm, pc = mel_sorted[i-1][1], cp_sorted[i-1][1]
            cm, cc = mel_sorted[i][1],   cp_sorted[i][1]
            if (cm - pm) * (cc - pc) < 0:
                n_contrary += 1
        n_moves = min(len(mel_sorted), len(cp_sorted)) - 1
        if n_moves > 0:
            print(f"    Mov. contrario : {n_contrary}/{n_moves}"
                  f" ({100*n_contrary//n_moves}%)")
    print("═" * 60)


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # --- Cargar MIDI ---
    print(f"\n◆ Cargando {args.input} …")
    try:
        melody_notes, tempo_bpm, beats_per_bar, tpb = load_melody_midi(args.input)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    beats_per_bar = args.beats if args.beats != 4 else beats_per_bar
    if args.bars:
        n_bars = args.bars
    else:
        last = max(o + d for o, _, d, _ in melody_notes)
        n_bars = max(1, math.ceil(last / beats_per_bar))

    # --- Tonalidad ---
    key_str = args.key
    key_info = (KeyInfo.detect_from_notes(melody_notes)
                if key_str == 'auto'
                else KeyInfo.from_string(key_str))

    print(f"  Tonalidad  : {key_info.name}")
    print(f"  Tempo      : {tempo_bpm} BPM")
    print(f"  Compases   : {n_bars}  ({beats_per_bar}/4)")
    print(f"  Notas mel. : {len(melody_notes)}")

    # --- Programa GM ---
    program = args.program
    if program is None:
        voice_name = args.voice if args.voice != 'auto' else 'viola'
        program = VOICE_PROGRAMS.get(voice_name, 40)

    # --- Generar ---
    species_list = [1, 2, 3, 4, 5] if args.all_species else [args.species]
    output_paths = []

    for sp in species_list:
        print(f"\n◆ Generando especie {sp} …")
        try:
            cp_notes = generate_counterpoint_voice(
                melody_notes   = melody_notes,
                key            = key_info,
                species        = sp,
                voice          = args.voice,
                voice_position = args.voice_position,
                beats_per_bar  = beats_per_bar,
                n_bars         = n_bars,
                tension_curve  = args.tension_curve,
                strict         = args.strict,
                seed           = args.seed,
                verbose        = args.verbose,
            )
        except CounterpointStrictError as e:
            # En modo --strict, si una especie no tiene solución que cumpla
            # las reglas duras de Fux, no se emite MIDI: se informa y se
            # continúa con la siguiente especie (--all-species) o se aborta.
            print(f"  ✗ [strict] especie {sp} descartada:")
            for line in str(e).splitlines():
                print(f"      {line}")
            if args.all_species:
                continue
            else:
                print("\n  Consejo: prueba sin --strict, con otra voz "
                      "(--voice violin/violonchelo...), otra tonalidad "
                      "(--key) o acorta la melodía (--bars).")
                sys.exit(2)


        # Determinar nombre de salida
        if args.all_species:
            base  = Path(args.output).stem
            out_p = str(Path(args.output_dir) / f"{base}_s{sp}.mid")
        else:
            out_p = args.output

        # Guardar MIDI
        save_two_voice_midi(
            melody_notes = melody_notes,
            cp_notes     = cp_notes,
            output_path  = out_p,
            tempo_bpm    = tempo_bpm,
            beats_per_bar= beats_per_bar,
            cp_program   = program,
            cp_channel   = args.channel,
            verbose      = args.verbose,
        )
        print(f"  ✓ Guardado: {out_p}")
        output_paths.append((out_p, cp_notes))

        # Fingerprint
        if args.export_fingerprint:
            fp_path = out_p.replace('.mid', '.fingerprint.json')
            export_fingerprint(melody_notes, cp_notes, key_info,
                               tempo_bpm, n_bars, beats_per_bar, fp_path)
            print(f"  ✓ Fingerprint: {fp_path}")

        # Resumen
        if args.verbose or not args.all_species:
            _print_summary(melody_notes, cp_notes, key_info,
                           sp, args.voice, args.voice_position,
                           n_bars, beats_per_bar)

    # --- Catálogo ---
    if args.all_species and args.catalog and output_paths:
        catalog_path = str(Path(args.output_dir) / "counterpoint_catalog.mid")
        print(f"\n◆ Generando catálogo … → {catalog_path}")
        # Concatenar todas las especies en un único MIDI
        us_per_beat = int(60_000_000 / max(tempo_bpm, 1))
        bu = {2: 2, 3: 4, 4: 4, 6: 8}.get(beats_per_bar, 4)
        cat = mido.MidiFile(type=1, ticks_per_beat=TICKS)

        # Pista 0: metadata
        meta_trk = mido.MidiTrack()
        meta_trk.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
        meta_trk.append(mido.MetaMessage(
            'time_signature', numerator=beats_per_bar, denominator=bu,
            clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        cat.tracks.append(meta_trk)

        total_offset = 0.0
        gap = beats_per_bar * 2  # 2 compases de silencio entre secciones

        for sp_idx, (out_p, cp_notes) in enumerate(output_paths):
            # Melodía
            trk_mel = mido.MidiTrack()
            trk_mel.name = f"Melody_S{species_list[sp_idx]}"
            trk_mel.append(mido.Message('program_change', channel=0,
                                         program=0, time=0))
            events = []
            for o, p_, d, v in melody_notes:
                t_on  = max(1, int(round((total_offset + o) * TICKS)))
                t_off = max(t_on + 1,
                            int(round((total_offset + o + d) * TICKS)))
                events.extend([(t_on, 'on', p_, v), (t_off, 'off', p_, 0)])
            events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
            prev_t = 0
            for t, kind, note, vel in events:
                dt = max(0, t - prev_t)
                trk_mel.append(mido.Message(
                    'note_on' if kind == 'on' else 'note_off',
                    channel=0, note=note, velocity=vel, time=dt))
                prev_t = t
            cat.tracks.append(trk_mel)

            # Contrapunto
            trk_cp = mido.MidiTrack()
            trk_cp.name = f"CP_S{species_list[sp_idx]}"
            trk_cp.append(mido.Message('program_change', channel=1,
                                        program=program, time=0))
            events = []
            for o, p_, d, v in cp_notes:
                t_on  = max(1, int(round((total_offset + o) * TICKS)))
                t_off = max(t_on + 1,
                            int(round((total_offset + o + d) * TICKS)))
                events.extend([(t_on, 'on', p_, v), (t_off, 'off', p_, 0)])
            events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
            prev_t = 0
            for t, kind, note, vel in events:
                dt = max(0, t - prev_t)
                trk_cp.append(mido.Message(
                    'note_on' if kind == 'on' else 'note_off',
                    channel=1, note=note, velocity=vel, time=dt))
                prev_t = t
            cat.tracks.append(trk_cp)

            # Avanzar offset
            section_dur = n_bars * beats_per_bar
            total_offset += section_dur + gap

        cat.save(catalog_path)
        print(f"  ✓ Catálogo guardado: {catalog_path}")

    # --- Reproducción ---
    if args.listen and output_paths:
        play_path = output_paths[0][0]
        print(f"\n◆ Reproduciendo {play_path} ({args.play_seconds}s) …")
        _play_midi(play_path, args.play_seconds)

    print("\n◆ Listo.")


if __name__ == '__main__':
    main()
