#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        CSL-L2M  v1.1                                         ║
║         Generación controlable de melodía a partir de letra (Lyric-to-Melody)║
║                                                                              ║
║  Adaptación de CSL-L2M (AAAI-2025) al estilo single-file del ecosistema.    ║
║  Encoder Transformer sobre letras → Decoder Transformer autorregresivo       ║
║  condicionado en 14 controles musicales por frase (pitch, duración, ritmo,   ║
║  tonalidad, emoción, estructura). Representación: REMI-Aligned.              ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare    — MIDI + letras alineadas → eventos REMI-Aligned (.pkl)        ║
║    vocab      — eventos .pkl → diccionarios melody + lyric                   ║
║    splits     — carpeta de .pkl → train/val/test splits                      ║
║    attributes — extrae 12 atributos estadísticos por frase                   ║
║    train-vqvae— entrena VQ-VAE para extraer features melódicas               ║
║    extract-feats— VQ-VAE checkpoint → features latentes por canción          ║
║    train      — entrena el modelo CSL-L2M completo                           ║
║    generate   — genera MIDI dado un .pkl con letra alineada                  ║
║    round-trip — REMI-Aligned → MIDI (diagnóstico sin modelo)                 ║
║    inspect    — diagnóstico de datos y checkpoints                            ║
║                                                                              ║
║  FLUJO TÍPICO:                                                               ║
║    1. Preparar corpus (MIDI + letras alineadas en lyrics MIDI markers):      ║
║       python csl_l2m.py prepare --midi-dir midis/ --output-dir data/events/ ║
║    2. Construir vocabulario:                                                  ║
║       python csl_l2m.py vocab --events-dir data/events/ --output-dir data/  ║
║    3. Crear splits:                                                           ║
║       python csl_l2m.py splits --events-dir data/events/ --output-dir data/ ║
║    4. Calcular atributos:                                                     ║
║       python csl_l2m.py attributes --events-dir data/events/                 ║
║                         --output-dir data/attributes/                         ║
║    5. (Opcional) VQ-VAE para features aprendidas:                            ║
║       python csl_l2m.py train-vqvae --events-dir data/events/               ║
║                         --vocab data/dictionary_melody.pkl                   ║
║                         --model-dir model_vqvae/                             ║
║       python csl_l2m.py extract-feats --events-dir data/events/             ║
║                         --vocab data/dictionary_melody.pkl                   ║
║                         --model-dir model_vqvae/ --output-dir data/feats/   ║
║    6. Entrenar:                                                               ║
║       python csl_l2m.py train --events-dir data/events/                     ║
║                         --vocab-melody data/dictionary_melody.pkl            ║
║                         --vocab-lyric  data/dictionary_lyric.pkl             ║
║                         --train-split  data/train.pkl                        ║
║                         --val-split    data/val.pkl                          ║
║                         --attr-dir     data/attributes/                      ║
║                         --model-dir    model_csll2m/                         ║
║    7. Generar:                                                                ║
║       python csl_l2m.py generate --events data/events/cancion.pkl           ║
║                         --vocab-melody data/dictionary_melody.pkl            ║
║                         --vocab-lyric  data/dictionary_lyric.pkl             ║
║                         --attr-dir     data/attributes/                      ║
║                         --model-dir    model_csll2m/ --output salida.mid     ║
║                                                                              ║
║  FORMATO MIDI DE ENTRADA:                                                    ║
║    - Un único instrumento (pista 0), velocity=126 en todas las notas         ║
║    - Letras en MIDI Lyric markers, una por nota                              ║
║    - Convención de letras:                                                    ║
║        "hola"   → sílaba normal (una nota)                                   ║
║        "ho*"    → extensión de la sílaba anterior (nota extra, sin avanzar)  ║
║        "hola."  → última sílaba de la frase (marca fin de SEQ)               ║
║        "*."     → extensión + fin de frase                                   ║
║                                                                              ║
║  FORMATO ABC DE ENTRADA (sin dependencias externas):                         ║
║    - Cabeceras estándar: X: T: M: L: Q: K:                                   ║
║    - Notas en compases separados por | con altura estándar ABC               ║
║    - Letras bajo cada voz con w: (una sílaba por nota)                       ║
║    - Convención w: estándar:                                                  ║
║        sílaba    → una nota                                                   ║
║        síla-ba   → guión une la siguiente sílaba (misma palabra)             ║
║        *  _      → melisma: extiende la sílaba anterior (nota sin avanzar)   ║
║        |         → barra de compás en la letra (se ignora)                   ║
║    - Cada línea de notas con su w: equivale a una frase (SEQ)                ║
║    - Ejemplo mínimo:                                                          ║
║        X:1                                                                    ║
║        T:Prueba  M:4/4  L:1/8  Q:90  K:C                                    ║
║        |G2 A2 B2 c2|d4 c4|                                                   ║
║        w:ho- la mun- do bue- nas no- ches                                    ║
║        |e4 d4|c8|                                                             ║
║        w:có- mo es- tás
║                                                                              ║
║  CONTROLES EN GENERACIÓN:                                                    ║
║    --key KEY        Tonalidad: C, Dm, F#m, Bb … (default: auto del .pkl)    ║
║    --emotion E      Emoción: Positive | Neutral | Negative (default: auto)  ║
║    --temperature F  Temperatura de muestreo (default: 1.2)                  ║
║    --nucleus-p F    Probabilidad acumulada nucleus (default: 0.9)            ║
║    --shift-pm N     Desplazar atributo pitch-mean ±N clases (default: 0)    ║
║    --shift-nd N     Desplazar note-density ±N clases (default: 0)           ║
║    --n-samples N    Número de melodías a generar (default: 1)               ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    miditoolkit, numpy, torch, scipy, PyYAML                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
import textwrap
import time
from copy import deepcopy
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

VERSION        = '1.0'
BEAT_RESOL     = 480
BAR_RESOL      = BEAT_RESOL * 4
TICK_RESOL     = BEAT_RESOL // 16   # 30 ticks por subdivisión
MEASURE_LENGTH = 64                 # BAR_RESOL // TICK_RESOL
ATTR_DIM       = 64                 # clases por atributo estadístico

ATTR_NAMES = ['PM', 'PV', 'PR', 'DM', 'DV', 'DR', 'ND', 'MCD', 'AA', 'CM', 'DMM', 'Align']

KEY_DICT = {
    'A': 0, 'Ab': 1, 'Am': 2, 'B': 3, 'Bb': 4, 'Bbm': 5, 'Bm': 6,
    'C': 7, 'C#m': 8, 'Cm': 9, 'D': 10, 'D#m': 11, 'Db': 12, 'Dm': 13,
    'E': 14, 'Eb': 15, 'Em': 16, 'F': 17, 'F#': 18, 'F#m': 19, 'Fm': 20,
    'G': 21, 'G#m': 22, 'Gm': 23,
}
IDX2KEY   = {v: k for k, v in KEY_DICT.items()}

EMOTION_DICT = {'Neutral': 0, 'Negative': 1, 'Positive': 2}
IDX2EMOTION  = {v: k for k, v in EMOTION_DICT.items()}

STRUCT_DICT = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════════════════════

def _pkl_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def _pkl_save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _time_to_pos(t):
    return round(t / TICK_RESOL)

def _create_event(name, value):
    return {'name': name, 'value': value}

def _separator(char='═', width=68):
    print(char * width)

def _header(title):
    _separator()
    print(f'  {title}')
    _separator()


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER ABC — sin dependencias externas
# ══════════════════════════════════════════════════════════════════════════════

# Tabla cromática ABC → semitono relativo (C=0)
_ABC_SEMITONES = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# Octava base según mayúsculas/minúsculas ABC:
# Mayúsculas C,D,E,F,G,A,B = C4..B4  (C4=60 en MIDI)
# Minúsculas c,d,e,f,g,a,b = C5..B5  (c5=72 en MIDI)
# MIDI: octava N → nota_base = (N+1)*12, así C4 = 5*12 = 60 ✓
_ABC_OCTAVE_UPPER = 5   # multiplicador: pitch = octave*12 + semitone → 5*12+0 = 60 = C4
_ABC_OCTAVE_LOWER = 6   # 6*12+0 = 72 = C5


def _abc_note_to_midi(note_str, key_acc):
    """
    Convierte una nota ABC (ej. "^G," "c2" "_B") a pitch MIDI y duración relativa.
    key_acc: dict de accidentales de la armadura {letra_mayúscula: semitono_delta}.
    Devuelve (pitch_midi, dur_num, dur_den) donde dur = dur_num/dur_den * L.
    """
    i = 0
    n = len(note_str)

    # Accidentales explícitas
    acc = 0
    while i < n and note_str[i] in ('^', '_', '='):
        if note_str[i] == '^':
            acc += 1
        elif note_str[i] == '_':
            acc -= 1
        else:
            acc = 0   # becuadro
        i += 1

    if i >= n or note_str[i] not in 'CDEFGABcdefgab':
        return None

    letter = note_str[i]
    upper  = letter.upper()
    octave = _ABC_OCTAVE_LOWER if letter.islower() else _ABC_OCTAVE_UPPER
    i += 1

    # Comas y apóstrofos ajustan octava
    while i < n and note_str[i] == ',':
        octave -= 1
        i += 1
    while i < n and note_str[i] == "'":
        octave += 1
        i += 1

    # Accidental de armadura si no hay accidental explícita
    if acc == 0 and upper in key_acc:
        acc = key_acc[upper]

    semitone = _ABC_SEMITONES[upper] + acc
    pitch    = octave * 12 + semitone

    # Duración: [num] o [num/den] o [/den] o [num/] etc.
    dur_num, dur_den = 1, 1
    if i < n and note_str[i].isdigit():
        s = ''
        while i < n and note_str[i].isdigit():
            s += note_str[i]; i += 1
        dur_num = int(s)
    if i < n and note_str[i] == '/':
        i += 1
        if i < n and note_str[i].isdigit():
            s = ''
            while i < n and note_str[i].isdigit():
                s += note_str[i]; i += 1
            dur_den = int(s)
        else:
            dur_den = 2   # bare / means /2

    return pitch, dur_num, dur_den


def _abc_key_accidentals(key_str):
    """
    Devuelve dict {letra_mayúscula: delta_semitono} para la armadura.
    Soporta las 15 tonalidades mayores/menores estándar.
    """
    sharps_order = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    flats_order  = ['B', 'E', 'A', 'D', 'G', 'C', 'F']

    # Número de sostenidos (+) o bemoles (-) por tónica mayor
    major_n = {
        'C':  0,
        'G':  1, 'D':  2, 'A':  3, 'E':  4, 'B':  5, 'F#': 6, 'C#': 7,
        'F': -1, 'Bb':-2, 'Eb':-3, 'Ab':-4, 'Db':-5, 'Gb':-6, 'Cb':-7,
    }
    minor_to_major = {
        'Am':'C',  'Em':'G',  'Bm':'D',  'F#m':'A', 'C#m':'E',
        'G#m':'B', 'D#m':'F#','A#m':'C#','Dm':'F',  'Gm':'Bb',
        'Cm':'Eb', 'Fm':'Ab', 'Bbm':'Db','Ebm':'Gb','Abm':'Cb',
    }

    k     = key_str.strip()
    minor = k.endswith('m') and not k.endswith('Maj')
    if minor:
        root = minor_to_major.get(k, 'C')
    else:
        root = k.replace('maj','').replace('Maj','').strip()

    n = major_n.get(root, 0)
    acc = {}
    if n > 0:
        for letter in sharps_order[:n]:
            acc[letter] = 1
    elif n < 0:
        for letter in flats_order[:-n]:   # n is negative, so -n is positive
            acc[letter] = -1
    return acc


def _parse_abc_body(body_lines):
    """
    Parsea las líneas del cuerpo ABC y devuelve lista de tuplas:
      (notes_line, w_line)
    donde notes_line es la línea de notación musical y w_line la línea w: asociada.
    Una línea de notas sin w: inmediata se agrupa con la siguiente w:.
    """
    pairs = []
    i = 0
    while i < len(body_lines):
        line = body_lines[i].strip()
        if not line or line.startswith('%'):
            i += 1
            continue
        if line.startswith('w:') or line.startswith('W:'):
            i += 1
            continue
        # Es una línea de notas
        notes_line = line
        # Buscar la w: que le sigue (puede haber líneas vacías entre medias)
        j = i + 1
        w_line = ''
        while j < len(body_lines):
            nxt = body_lines[j].strip()
            if not nxt or nxt.startswith('%'):
                j += 1
                continue
            if nxt.lower().startswith('w:'):
                w_line = nxt[2:].strip()
                j += 1
            break
        pairs.append((notes_line, w_line))
        i = j
    return pairs


def _tokenize_abc_notes(notes_line):
    """
    Extrae tokens de notas de una línea ABC.
    Ignora barras |, decoraciones !!, acordes [..], repeticiones y texto.
    Devuelve lista de strings de nota (con accidentales, octava y duración).
    """
    import re
    # Eliminar decoraciones tipo !staccato! y texto "..."
    line = re.sub(r'![^!]*!', '', notes_line)
    line = re.sub(r'"[^"]*"', '', line)
    # Eliminar acordes [CEG] — tomamos solo la primera nota del acorde
    line = re.sub(r'\[([^\]]+)\]', lambda m: m.group(1).split()[0], line)
    # Eliminar barras de compás, espacios de repetición, símbolos de sección
    line = re.sub(r'[|:\[\]0-9]', ' ', line)
    # Tokenizar: accidental(es) + letra + comas/apóstrofos + dígitos + /dígitos
    tokens = re.findall(r'[_^=]*[CDEFGABcdefgab][,\']*(?:\d+(?:/\d*)?|/\d*)?', line)
    # También recoger silencios z (los contamos pero no generamos nota)
    tokens_full = re.findall(r'(?:[_^=]*[CDEFGABcdefgab][,\']*(?:\d+(?:/\d*)?|/\d*)?|z\d*)', line)
    return tokens_full


def _tokenize_abc_lyrics(w_line):
    """
    Tokeniza una línea w: en tokens alineados nota-a-nota, siguiendo el
    estándar ABC real:

      "síla-"  → sílaba con guión final: la siguiente nota lleva la
                 continuación de la misma palabra (NO es melisma de nota,
                 sino inicio de la siguiente sílaba en la nota siguiente).
      "ba"     → sílaba normal (continuación de palabra si venía con guión).
      "-"      → token solo: repite/extiende la última sílaba a la nota
                 siguiente (melisma explícito).
      "*" "_"  → melisma: la nota actual extiende la sílaba anterior.
      "|"      → barra de compás en la letra, se ignora.

    En resumen: UN token por nota, y los guiones son solo marcadores
    visuales de unión de palabra — no generan tokens extra.

    Devuelve lista de strings donde '*' indica melisma.
    """
    tokens = []
    raw = w_line.split()
    for part in raw:
        if not part or part == '|':
            continue
        if part == '-':
            tokens.append('*')    # guión solo = melisma explícito
        elif part in ('*', '_'):
            tokens.append('*')    # melisma estándar
        else:
            tokens.append(part)   # sílaba (con o sin guión final, no importa)
    return tokens


def _abc_to_remi(abc_path, default_tempo=90, default_unit=8):
    """
    Parsea un fichero ABC y devuelve (seq_lyrics, pos_seq, melody_events).
    Sin dependencias externas.

    Asunciones:
      - Una línea de notas + su w: = una frase (evento SEQ)
      - Compás 4/4 si M: no está definido
      - L: 1/8 si no está definido
      - Tempo Q: en BPM (número simple o "1/4=90")
    """
    text = Path(abc_path).read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()

    # ── Parsear cabeceras ─────────────────────────────────────────────────────
    headers = {}
    body_start = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) >= 2 and stripped[1] == ':' and stripped[0].isalpha():
            headers[stripped[0]] = stripped[2:].strip()
            if stripped[0] == 'K':
                body_start = idx + 1
                break

    key_str   = headers.get('K', 'C')
    key_acc   = _abc_key_accidentals(key_str)

    # Métrica
    meter_str = headers.get('M', '4/4')
    try:
        mn, md   = [int(x) for x in meter_str.split('/')]
        beats_per_bar = mn
    except Exception:
        beats_per_bar = 4

    # Unidad de nota (L:)
    unit_str = headers.get('L', '1/8')
    try:
        un, ud   = [int(x) for x in unit_str.split('/')]
        unit_den = ud   # p.ej. L:1/8 → unit_den=8
    except Exception:
        unit_den = default_unit

    # Tempo Q:
    q_str = headers.get('Q', str(default_tempo))
    import re as _re
    m = _re.search(r'(\d+)\s*$', q_str)
    tempo_bpm = int(m.group(1)) if m else default_tempo

    # Ticks por unidad de nota (L:)
    # 1 beat = BEAT_RESOL ticks; 1 beat = 1/4 nota → L:1/8 = BEAT_RESOL/2
    ticks_per_unit = int(BEAT_RESOL * 4 / unit_den)

    # ── Parsear cuerpo ────────────────────────────────────────────────────────
    body_lines = lines[body_start:]
    pairs      = _parse_abc_body(body_lines)

    if not pairs:
        raise RuntimeError('No se encontraron líneas de notas en el fichero ABC')

    melody_events = [_create_event('SEQ', None)]
    seq_lyrics    = []
    cur_tick      = 0
    cur_bar       = 0

    ticks_per_bar = ticks_per_unit * beats_per_bar * (unit_den // 4) if unit_den >= 4 else BEAT_RESOL * beats_per_bar

    for phrase_idx, (notes_line, w_line) in enumerate(pairs):
        note_tokens   = _tokenize_abc_notes(notes_line)
        lyric_tokens  = _tokenize_abc_lyrics(w_line) if w_line else []

        # Alinear sílabas con notas (avanzar lírico solo en no-melismas)
        phrase_notes  = []   # (pitch, dur_ticks)
        phrase_lyrics = []   # sílaba por nota ('' si melisma/silencio)

        lyr_idx = 0
        for tok in note_tokens:
            if tok.startswith('z'):
                # Silencio: calcular duración y avanzar tiempo pero no sílaba
                s = tok[1:] if len(tok) > 1 else '1'
                try:    dur_n, dur_d = int(s), 1
                except Exception: dur_n, dur_d = 1, 1
                dur_ticks = max(TICK_RESOL, ticks_per_unit * dur_n // dur_d)
                phrase_notes.append(None)
                phrase_lyrics.append('')
                cur_tick += dur_ticks
                continue

            parsed = _abc_note_to_midi(tok, key_acc)
            if parsed is None:
                continue
            pitch, dur_n, dur_d = parsed
            dur_ticks = max(TICK_RESOL, ticks_per_unit * dur_n // dur_d)

            # Determinar sílaba
            if lyr_idx < len(lyric_tokens):
                syl = lyric_tokens[lyr_idx]
                if syl == '*':
                    phrase_notes.append((pitch, dur_ticks))
                    phrase_lyrics.append('*')   # melisma — no avanza sílaba
                    lyr_idx += 1
                else:
                    phrase_notes.append((pitch, dur_ticks))
                    phrase_lyrics.append(syl)
                    lyr_idx += 1
            else:
                phrase_notes.append((pitch, dur_ticks))
                phrase_lyrics.append('')

        # ── Convertir a eventos REMI ──────────────────────────────────────────
        lyrics_list    = []
        prev_bar_ev    = cur_bar
        phrase_ev_start = len(melody_events)

        j = 0
        while j < len(phrase_notes):
            entry = phrase_notes[j]
            if entry is None:
                cur_tick += 0   # ya sumado arriba
                j += 1
                continue

            pitch, dur_ticks = entry
            syl              = phrase_lyrics[j]

            bar_now  = cur_tick // (ticks_per_bar if ticks_per_bar > 0 else BEAT_RESOL * 4)
            beat_now = (cur_tick % (ticks_per_bar if ticks_per_bar > 0 else BEAT_RESOL * 4)) // ticks_per_unit

            if bar_now != prev_bar_ev:
                melody_events.append(_create_event('Bar', None))
                prev_bar_ev = bar_now

            melody_events.append(_create_event('Beat', int(beat_now)))
            melody_events.append(_create_event('Note_Pitch', int(pitch)))
            dur_aligned = max(TICK_RESOL, round(dur_ticks / TICK_RESOL) * TICK_RESOL)
            melody_events.append(_create_event('Note_Duration', dur_aligned))

            if syl == '*':
                # Nota de melisma — emitir ALIGN solo si la siguiente ya no es melisma
                next_is_mel = (j + 1 < len(phrase_notes)
                               and phrase_notes[j + 1] is not None
                               and phrase_lyrics[j + 1] == '*')
                if not next_is_mel:
                    melody_events.append(_create_event('ALIGN', None))
            else:
                # Sílaba normal: acumular y emitir ALIGN
                clean = syl.rstrip('-').strip()
                if clean:
                    lyrics_list.append(clean)
                # ¿La siguiente nota es melisma de esta sílaba?
                next_is_mel = (j + 1 < len(phrase_notes)
                               and phrase_notes[j + 1] is not None
                               and phrase_lyrics[j + 1] == '*')
                if not next_is_mel:
                    melody_events.append(_create_event('ALIGN', None))

            cur_tick  += dur_ticks
            cur_bar    = cur_tick // (ticks_per_bar if ticks_per_bar > 0 else BEAT_RESOL * 4)
            j         += 1

        # Fin de frase
        melody_events.append(_create_event('SEQ', None))
        if lyrics_list:
            seq_lyrics.append(lyrics_list)

    melody_events.append(_create_event('Bar', None))
    melody_events.append(_create_event('EOS', None))

    pos_seq = [i for i, ev in enumerate(melody_events) if ev['name'] == 'SEQ']
    return seq_lyrics, pos_seq, melody_events, tempo_bpm


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARE — MIDI / ABC → REMI-Aligned .pkl
# ══════════════════════════════════════════════════════════════════════════════

def _midi_to_remi(midi_path):
    """
    Convierte un MIDI con letras alineadas al formato REMI-Aligned.
    Devuelve (seq_lyrics, pos_seq, melody_events) o lanza RuntimeError.

    Convención de letras en los markers MIDI:
      - Sílaba normal: "hola"
      - Extensión (melisma): "ho*" (no avanza la sílaba)
      - Fin de frase (SEQ):  "hola." (el punto marca el fin del segmento)
      - Extensión + fin:     "*."
    """
    try:
        import miditoolkit
        midi_obj = miditoolkit.midi.parser.MidiFile(str(midi_path), charset='utf-8')
    except Exception as e:
        raise RuntimeError(f'No se pudo abrir {midi_path}: {e}')

    if len(midi_obj.instruments) != 1:
        raise RuntimeError(f'Se esperaba 1 instrumento, hay {len(midi_obj.instruments)}')
    if len(midi_obj.instruments[0].notes) != len(midi_obj.lyrics):
        raise RuntimeError(
            f'Notas ({len(midi_obj.instruments[0].notes)}) ≠ '
            f'lyrics ({len(midi_obj.lyrics)})'
        )

    notes = midi_obj.instruments[0].notes
    lyrics_raw = [l.text for l in midi_obj.lyrics]

    # Mapa posición → (bar, beat)
    max_pos = min(max(_time_to_pos(n.start) for n in notes) + 1, 2 ** 16)
    pos_to_info = [[None, None]] * max_pos
    cnt, bar = 0, 0
    for j in range(max_pos):
        pos_to_info[j] = [bar, cnt]
        cnt += 1
        if cnt >= MEASURE_LENGTH:
            cnt -= MEASURE_LENGTH
            bar += 1

    melody_events = [_create_event('SEQ', None)]
    lyrics_list, seq_lyrics = [], []
    current_bar = None

    for idx, note in enumerate(notes):
        lyric = lyrics_raw[idx]
        info  = pos_to_info[_time_to_pos(note.start)]

        if current_bar != info[0]:
            melody_events.append(_create_event('Bar', None))
        melody_events.append(_create_event('Beat', info[1]))
        melody_events.append(_create_event('Note_Pitch', note.pitch))
        dur = round((note.end - note.start) / TICK_RESOL) * TICK_RESOL
        melody_events.append(_create_event('Note_Duration', dur))
        current_bar = info[0]

        # ALIGN: avanza al siguiente carácter en la letra
        next_is_ext = (idx < len(notes) - 1 and '*' in lyrics_raw[idx + 1])
        if not ('*' in lyric and next_is_ext):
            melody_events.append(_create_event('ALIGN', None))

        # Acumular letra limpia
        clean = lyric.replace('*', '').replace('.', '')
        if clean:
            lyrics_list.append(clean)

        # Fin de frase
        if '.' in lyric:
            melody_events.append(_create_event('SEQ', None))
            seq_lyrics.append(lyrics_list)
            lyrics_list = []

    melody_events.append(_create_event('Bar', None))
    melody_events.append(_create_event('SEQ', None))
    melody_events.append(_create_event('EOS', None))

    if lyrics_list:                     # última frase sin punto final
        seq_lyrics.append(lyrics_list)

    pos_seq = [i for i, ev in enumerate(melody_events) if ev['name'] == 'SEQ']
    return seq_lyrics, pos_seq, melody_events


def cmd_prepare(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recoger MIDI y ABC
    midi_files = (sorted(input_dir.glob('*.mid')) +
                  sorted(input_dir.glob('*.midi')))
    abc_files  = sorted(input_dir.glob('*.abc'))
    all_files  = midi_files + abc_files

    if not all_files:
        print(f'[prepare] ✗  No se encontraron archivos .mid/.midi/.abc en {input_dir}')
        sys.exit(1)

    # Comprobar miditoolkit solo si hay MIDI
    if midi_files:
        try:
            import miditoolkit  # noqa: F401
        except ImportError:
            print('[prepare] ✗  miditoolkit no encontrado — pip install miditoolkit')
            print('[prepare]    (necesario solo para .mid/.midi; los .abc no lo requieren)')
            if not abc_files:
                sys.exit(1)
            print('[prepare]    Procesando solo los .abc\n')
            all_files = abc_files

    print(f'[prepare] {len(midi_files)} MIDI  +  {len(abc_files)} ABC  =  {len(all_files)} archivos')
    print(f'[prepare] Salida → {output_dir}\n')

    ok, skipped = 0, 0
    for mf in all_files:
        try:
            ext = mf.suffix.lower()
            if ext == '.abc':
                seq_lyrics, pos_seq, melody_events, tempo = _abc_to_remi(
                    mf, default_tempo=args.default_tempo
                )
                source = 'abc'
            else:
                seq_lyrics, pos_seq, melody_events = _midi_to_remi(mf)
                source = 'mid'

            out_path = output_dir / (mf.stem + '.pkl')
            _pkl_save((seq_lyrics, pos_seq, melody_events), out_path)
            n_seqs = len(seq_lyrics)
            n_evs  = len(melody_events)
            print(f'  ✓  [{source}]  {mf.stem:<38}  {n_seqs} frases  {n_evs} eventos')
            ok += 1
        except Exception as e:
            print(f'  ✗  {mf.stem:<44}  {e}')
            skipped += 1

    print()
    _separator()
    print(f'  RESUMEN PREPARE')
    _separator()
    print(f'  Procesados : {ok}')
    print(f'  Omitidos   : {skipped}')
    _separator()


# ══════════════════════════════════════════════════════════════════════════════
#  VOCAB — eventos → diccionarios
# ══════════════════════════════════════════════════════════════════════════════

def cmd_vocab(args):
    events_dir = Path(args.events_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(events_dir.glob('*.pkl'))
    if not pkl_files:
        print(f'[vocab] ✗  No se encontraron .pkl en {events_dir}')
        sys.exit(1)

    print(f'[vocab] {len(pkl_files)} archivos .pkl')

    all_events, all_lyrics = [], []
    for p in pkl_files:
        seq_lyrics, _, events = _pkl_load(p)
        for ev in events:
            all_events.append(f"{ev['name']}_{ev['value']}")
        for phrase in seq_lyrics:
            all_lyrics.extend(phrase)

    unique_ev = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2idx  = {k: i for i, k in enumerate(unique_ev)}
    idx2event  = {i: k for i, k in enumerate(unique_ev)}

    unique_ly = sorted(set(all_lyrics), key=lambda x: (not isinstance(x, int), x))
    lyric2idx  = {k: i for i, k in enumerate(unique_ly)}
    idx2lyric  = {i: k for i, k in enumerate(unique_ly)}

    out_mel = output_dir / 'dictionary_melody.pkl'
    out_lyr = output_dir / 'dictionary_lyric.pkl'
    _pkl_save((event2idx, idx2event), out_mel)
    _pkl_save((lyric2idx, idx2lyric), out_lyr)

    print(f'[vocab] ✓  Vocabulario melodía : {len(event2idx)} tokens  → {out_mel}')
    print(f'[vocab] ✓  Vocabulario letra   : {len(lyric2idx)} tokens  → {out_lyr}')


# ══════════════════════════════════════════════════════════════════════════════
#  SPLITS — train / val / test
# ══════════════════════════════════════════════════════════════════════════════

def cmd_splits(args):
    events_dir = Path(args.events_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces = sorted([p.name for p in events_dir.glob('*.pkl')])
    if not pieces:
        print(f'[splits] ✗  No se encontraron .pkl en {events_dir}')
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(pieces)

    n      = len(pieces)
    n_test = max(1, int(n * args.test_ratio))
    n_val  = max(1, int(n * args.val_ratio))
    n_tr   = n - n_val - n_test

    train = pieces[:n_tr]
    val   = pieces[n_tr:n_tr + n_val]
    test  = pieces[n_tr + n_val:]

    _pkl_save(train, output_dir / 'train.pkl')
    _pkl_save(val,   output_dir / 'val.pkl')
    _pkl_save(test,  output_dir / 'test.pkl')

    print(f'[splits] {n} canciones  →  train={len(train)}  val={len(val)}  test={len(test)}')
    print(f'[splits] ✓  {output_dir / "train.pkl"}')
    print(f'[splits] ✓  {output_dir / "val.pkl"}')
    print(f'[splits] ✓  {output_dir / "test.pkl"}')


# ══════════════════════════════════════════════════════════════════════════════
#  ATTRIBUTES — 12 atributos estadísticos por frase
# ══════════════════════════════════════════════════════════════════════════════

def _compute_raw_attrs(events):
    """
    Calcula los 12 atributos en escala continua, por frase (SEQ).
    Devuelve dict attr→lista_de_valores (uno por frase).
    """
    import numpy as np

    raw = {a: [] for a in ATTR_NAMES}

    seq_pitch, seq_dur = [], []
    jj = nn = num_notes = cur_bar = mm = 0
    first_pos = cur_pos = 0

    for ev in events:
        name = ev['name']
        if name == 'Note_Pitch':
            seq_pitch.append(ev['value'])
            num_notes += 1
        elif name == 'Note_Duration':
            seq_dur.append(ev['value'] // TICK_RESOL)
        elif name == 'ALIGN':
            mm += 1
        elif name == 'Bar':
            prev = events[jj - 1]['name'] if jj > 0 else ''
            nxt  = events[jj + 1]['name'] if jj + 1 < len(events) else ''
            if prev != 'SEQ' and nxt not in ('SEQ', '') and nxt == 'Beat':
                cur_bar += 1
        elif name == 'Beat':
            cur_pos = int(ev['value'])
            nn += 1
            if nn == 1:
                first_pos = cur_pos
        elif name == 'SEQ' and jj != 0 and seq_pitch:
            assert len(seq_pitch) == len(seq_dur), 'Desalineamiento pitch/dur'
            rec_idx = max(1, cur_bar * MEASURE_LENGTH + cur_pos - first_pos)
            p = np.array(seq_pitch)
            d = np.array(seq_dur)

            raw['PM'].append(float(np.mean(p)))
            raw['PV'].append(float(np.std(p)))
            raw['PR'].append(float(p.max() - p.min()))
            raw['DM'].append(float(np.mean(d)))
            raw['DV'].append(float(np.std(d)))
            raw['DR'].append(float(d.max() - d.min()))
            raw['ND'].append(num_notes / rec_idx)
            raw['Align'].append(mm / num_notes)
            raw['MCD'].append((d == 8).sum() / len(d))

            if len(p) > 1:
                diff = np.abs(p[1:] - p[:-1]).tolist()
                arpeggios = {0, 3, 4, 7, 10, 11, 12, 13, 14}
                raw['AA'].append(sum(x in arpeggios for x in diff) / len(diff))
                raw['CM'].append(diff.count(1) / len(diff))
                raw['DMM'].append(float(np.sum(p[1:] - p[:-1] > 0) / len(diff)))
            else:
                raw['AA'].append(0.0)
                raw['CM'].append(0.0)
                raw['DMM'].append(0.0)

            seq_pitch, seq_dur = [], []
            num_notes = cur_bar = mm = nn = 0
        jj += 1

    return raw


def cmd_attributes(args):
    import numpy as np

    events_dir = Path(args.events_dir)
    output_dir = Path(args.output_dir)

    pkl_files = sorted(events_dir.glob('*.pkl'))
    if not pkl_files:
        print(f'[attributes] ✗  No se encontraron .pkl en {events_dir}')
        sys.exit(1)

    print(f'[attributes] {len(pkl_files)} archivos  |  {ATTR_DIM} clases por atributo')

    # Paso 1: acumular todos los valores continuos del corpus
    corpus_raw = {a: [] for a in ATTR_NAMES}
    for p in pkl_files:
        _, _, events = _pkl_load(p)
        piece_raw = _compute_raw_attrs(events)
        for a in ATTR_NAMES:
            corpus_raw[a].extend(piece_raw[a])

    # Calcular fronteras de cuantización (equiprobable en el corpus)
    n    = len(corpus_raw['PM'])
    step = max(1, n // ATTR_DIM)
    bounds = {}
    for a in ATTR_NAMES:
        sorted_vals = sorted(corpus_raw[a])
        bounds[a] = [sorted_vals[j] for j in range(step, n - step, step)]

    bounds_path = output_dir / 'attr_bounds.pkl'
    output_dir.mkdir(parents=True, exist_ok=True)
    _pkl_save(bounds, bounds_path)
    print(f'[attributes] ✓  Fronteras guardadas → {bounds_path}')

    # Paso 2: cuantizar y guardar por canción
    ok, skipped = 0, 0
    for p in pkl_files:
        try:
            _, _, events = _pkl_load(p)
            piece_raw  = _compute_raw_attrs(events)
            piece_cls  = {}
            for a in ATTR_NAMES:
                piece_cls[a] = np.searchsorted(bounds[a], piece_raw[a]).tolist()

            out_path = output_dir / p.name
            _pkl_save(piece_cls, out_path)
            n_seqs = len(piece_cls['PM'])
            print(f'  ✓  {p.stem:<40}  {n_seqs} frases')
            ok += 1
        except Exception as e:
            print(f'  ✗  {p.stem:<40}  {e}')
            skipped += 1

    print()
    _separator()
    print(f'  RESUMEN ATTRIBUTES')
    _separator()
    print(f'  Procesados : {ok}')
    print(f'  Omitidos   : {skipped}')
    _separator()


# ══════════════════════════════════════════════════════════════════════════════
#  MODELOS PYTORCH — helpers compartidos
# ══════════════════════════════════════════════════════════════════════════════

def _build_model_classes():
    """
    Define y devuelve las clases de modelo bajo demanda (import torch lazy).
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _generate_causal_mask(seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        mask.requires_grad = False
        return mask

    def _weights_init(m):
        cn = m.__class__.__name__
        if cn.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 0.0, 0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif cn.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 0.0, 0.01)
        elif cn.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, 0.01)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ── Bloques básicos ───────────────────────────────────────────────────────

    class PositionalEncoding(nn.Module):
        def __init__(self, d_embed, max_pos=20480):
            super().__init__()
            pe  = torch.zeros(max_pos, d_embed)
            pos = torch.arange(0, max_pos, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_embed, 2).float() * (-math.log(10000.0) / d_embed))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, seq_len, bsz=None):
            enc = self.pe[:seq_len, :]
            if bsz is not None:
                enc = enc.expand(seq_len, bsz, -1)
            return enc

    class TokenEmbedding(nn.Module):
        def __init__(self, n_token, d_embed, d_proj):
            super().__init__()
            self.emb_scale  = d_proj ** 0.5
            self.emb_lookup = nn.Embedding(n_token, d_embed)
            self.emb_proj   = nn.Linear(d_embed, d_proj, bias=False) if d_proj != d_embed else None

        def forward(self, inp):
            x = self.emb_lookup(inp)
            if self.emb_proj is not None:
                x = self.emb_proj(x)
            return x.mul_(self.emb_scale)

    # ── Encoder Transformer (para letras) ─────────────────────────────────────

    class TransformerEncoder(nn.Module):
        def __init__(self, n_layer, n_head, d_model, d_ff, d_latent, dropout=0.1):
            super().__init__()
            layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, 'relu')
            self.tr  = nn.TransformerEncoder(layer, n_layer)
            self.fc  = nn.Linear(d_model, d_latent)

        def forward(self, x, padding_mask=None):
            out    = self.tr(x, src_key_padding_mask=padding_mask)
            return self.fc(out[0])          # CLS token → latente

    # ── VQ-VAE encoder  ───────────────────────────────────────────────────────

    class VectorQuantizeEMA(nn.Module):
        def __init__(self, d_latent, n_codes=2048, n_groups=64, decay=0.995, eps=1e-4):
            super().__init__()
            assert d_latent % n_groups == 0
            self.n_groups = n_groups
            self.dim      = d_latent // n_groups
            self.n_codes  = n_codes
            self.decay, self.eps = decay, eps
            self.init = False

            embed = torch.randn(n_codes, self.dim)
            self.register_buffer('embedding',    embed)
            self.register_buffer('cluster_size', torch.ones(n_codes))
            self.register_buffer('cluster_sum',  embed.clone().detach())

        def forward(self, x):
            x_ = x.reshape(-1, self.dim)
            if self.training and not self.init:
                self._init(x_)
            emb_t = self.embedding.t()
            dist  = x_.pow(2).sum(1, keepdim=True) - 2 * x_ @ emb_t + emb_t.pow(2).sum(0, keepdim=True)
            _, idx = (-dist).max(1)
            onehot  = F.one_hot(idx, self.n_codes).float()
            quantize = F.embedding(idx, self.embedding).view(-1, self.n_groups * self.dim)
            diff     = (quantize.detach() - x).pow(2).mean()
            quantize = x + (quantize - x).detach()
            if self.training:
                self._ema_update(x_, onehot)
            return quantize, diff

        def _init(self, x):
            self.init = True
            rand = self._randomize(x)
            self.cluster_sum.data.copy_(rand)
            self.cluster_size.data.fill_(1)

        def _randomize(self, x):
            n = x.size(0)
            if n < self.n_codes:
                r  = (self.n_codes + n - 1) // n
                x  = x.repeat(r, 1)
                x += 0.01 / (self.dim ** 0.5) * torch.randn_like(x)
            return x[torch.randperm(x.size(0))][:self.n_codes]

        def _ema_update(self, x, assign):
            with torch.no_grad():
                cs = assign.sum(0)
                cv = assign.t() @ x
                rand = self._randomize(x)
                self.cluster_size.data.copy_(self.decay * self.cluster_size + (1 - self.decay) * cs)
                self.cluster_sum.data.copy_(self.decay * self.cluster_sum  + (1 - self.decay) * cv)
                used  = (self.cluster_size >= 0.99).float().unsqueeze(-1)
                n     = self.cluster_size.sum()
                count = (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
                centers = self.cluster_sum / count.unsqueeze(-1)
                self.embedding.data.copy_(used * centers + (1 - used) * rand)

    class TransformerEncoderVQ(nn.Module):
        def __init__(self, n_layer, n_head, d_model, d_ff, d_latent,
                     dropout=0.1, n_codes=2048, n_groups=64):
            super().__init__()
            layer   = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, 'relu')
            self.tr = nn.TransformerEncoder(layer, n_layer)
            self.fc = nn.Linear(d_model, d_latent)
            self.ln = nn.LayerNorm(d_latent)   # estabiliza latentes antes del VQ
            self.vq = VectorQuantizeEMA(d_latent, n_codes, n_groups)

        def forward(self, x, padding_mask=None):
            out    = self.tr(x, src_key_padding_mask=padding_mask)
            latent = self.ln(self.fc(out[0]))
            z, diff = self.vq(latent)
            return latent, z, diff

    # ── Decoder Transformer ───────────────────────────────────────────────────

    class FullSongDecoder(nn.Module):
        def __init__(self, n_layer, n_head, d_model, d_ff, d_seg, dropout=0.1):
            super().__init__()
            self.seg_proj = nn.Linear(d_seg, d_model, bias=False)
            layer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, 'relu')
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout, 'relu')
                for _ in range(n_layer)
            ])

        def forward(self, x, seg_emb):
            mask = _generate_causal_mask(x.size(0)).to(x.device)
            seg  = self.seg_proj(seg_emb)
            out  = x
            for layer in self.layers:
                out = out + seg
                out = layer(out, src_mask=mask)
            return out

    # ── VQ-VAE completo ───────────────────────────────────────────────────────

    class VQVAE(nn.Module):
        def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff,
                     dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
                     d_latent, d_embed, n_token,
                     enc_dropout=0.1, dec_dropout=0.1,
                     n_codes=2048, n_groups=64):
            super().__init__()
            self.d_latent = d_latent
            self.n_token  = n_token

            self.token_emb  = TokenEmbedding(n_token, d_embed, enc_d_model)
            self.pe         = PositionalEncoding(d_embed)
            self.dec_proj   = nn.Linear(dec_d_model, n_token)
            self.emb_drop   = nn.Dropout(enc_dropout)
            self.encoder    = TransformerEncoderVQ(
                enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_latent,
                enc_dropout, n_codes, n_groups
            )
            self.decoder    = FullSongDecoder(
                dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_latent, dec_dropout
            )
            self.apply(_weights_init)

        def forward(self, enc_inp, dec_inp, dec_seq_pos, padding_mask=None):
            bt, n_seqs = enc_inp.size(1), enc_inp.size(2)
            enc_emb = self.token_emb(enc_inp)
            dec_emb = self.token_emb(dec_inp)
            enc_emb = enc_emb.reshape(enc_inp.size(0), -1, enc_emb.size(-1))
            enc_inp_drop = self.emb_drop(enc_emb) + self.pe(enc_inp.size(0))
            dec_inp_drop = self.emb_drop(dec_emb) + self.pe(dec_inp.size(0))
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))
            mu, z, diff = self.encoder(enc_inp_drop, padding_mask)
            z_reshaped  = z.reshape(bt, n_seqs, -1)
            seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_latent,
                                  device=z.device)
            for n in range(dec_inp.size(1)):
                for b, (st, ed) in enumerate(zip(dec_seq_pos[n, :-1], dec_seq_pos[n, 1:])):
                    seg_emb[st:ed, n, :] = z_reshaped[n, b, :]
            dec_out    = self.decoder(dec_inp_drop, seg_emb)
            dec_logits = self.dec_proj(dec_out)
            return mu, diff, dec_logits

        def get_latent(self, enc_inp, padding_mask=None):
            emb = self.token_emb(enc_inp)
            emb = self.emb_drop(emb) + self.pe(enc_inp.size(0))
            _, z, _ = self.encoder(emb, padding_mask)
            return z

        def compute_loss(self, diff, dec_logits, dec_tgt, beta=1.0):
            rc = F.cross_entropy(
                dec_logits.view(-1, dec_logits.size(-1)),
                dec_tgt.contiguous().view(-1),
                ignore_index=self.n_token - 1, reduction='mean'
            ).float()
            return {'total': rc + beta * diff, 'recons': rc, 'vq': diff}

    # ── CSL-L2M completo ──────────────────────────────────────────────────────

    class CSLL2M(nn.Module):
        """
        Encoder de letras + Decoder condicionado en 14 controles concatenados.
        Los controles activables son: key, emotion, struct, Align, ND,
        PM, PV, PR, DMM, AA, CM, DM, DV, DR, MCD, learned_feats.
        """

        # Dimensiones fijas de los embeddings de control
        _CTRL_CFG = {
            'key':     (32, 24),    # (d_emb, n_cls)
            'emotion': (32,  3),
            'struct':  (32,  5),
            'Align':   (32, 64),
            'ND':      (32, 64),
            'PM':      (32, 64),
            'PV':      (32, 64),
            'PR':      (32, 64),
            'DMM':     (32, 64),
            'AA':      (32, 64),
            'CM':      (32, 64),
            'DM':      (32, 64),
            'DV':      (32, 64),
            'DR':      (32, 64),
            'MCD':     (32, 64),
        }

        def __init__(self, enc_n_layer, enc_n_head, enc_d_model, enc_d_ff,
                     dec_n_layer, dec_n_head, dec_d_model, dec_d_ff,
                     d_latent, d_embed, n_token, n_token_lyric, pad_token_melody,
                     enc_dropout=0.1, dec_dropout=0.1,
                     d_learned=128, controls=None):
            """
            controls: conjunto de nombres de control activos.
                      None → todos activos excepto learned_feats.
            """
            super().__init__()
            self.d_latent          = d_latent
            self.n_token           = n_token
            self.pad_token_melody  = pad_token_melody

            if controls is None:
                controls = set(self._CTRL_CFG.keys())
            self.controls    = controls
            self.d_learned   = d_learned
            self.use_learned = 'learned_feats' in controls

            # Embeddings de tokens
            self.token_emb       = TokenEmbedding(n_token,       d_embed, enc_d_model)
            self.token_emb_lyric = TokenEmbedding(n_token_lyric, d_embed, enc_d_model)
            self.pe              = PositionalEncoding(d_embed)
            self.emb_drop        = nn.Dropout(enc_dropout)
            self.encoder         = TransformerEncoder(
                enc_n_layer, enc_n_head, enc_d_model, enc_d_ff, d_latent, enc_dropout
            )

            # Embeddings de control (uno por control activo)
            self.ctrl_embs = nn.ModuleDict()
            d_seg = d_latent
            for name in self._CTRL_CFG:
                if name in controls:
                    d_emb, n_cls = self._CTRL_CFG[name]
                    self.ctrl_embs[name] = TokenEmbedding(n_cls, d_emb, d_emb)
                    d_seg += d_emb
            if self.use_learned:
                d_seg += d_learned

            self.decoder  = FullSongDecoder(
                dec_n_layer, dec_n_head, dec_d_model, dec_d_ff, d_seg, dec_dropout
            )
            self.dec_proj = nn.Linear(dec_d_model, n_token)
            self.apply(_weights_init)

        def _cat_controls(self, seg_emb, ctrl_dict):
            """Concatena los embeddings de control activos al seg_emb."""
            parts = [seg_emb]
            for name in self._CTRL_CFG:
                if name in self.controls and name in ctrl_dict:
                    parts.append(self.ctrl_embs[name](ctrl_dict[name]))
            if self.use_learned and 'learned_feats' in ctrl_dict:
                parts.append(ctrl_dict['learned_feats'])
            return torch.cat(parts, dim=-1)

        def get_lyric_emb(self, enc_inp, padding_mask=None):
            emb = self.token_emb_lyric(enc_inp)
            enc = self.emb_drop(emb) + self.pe(enc_inp.size(0))
            return self.encoder(enc, padding_mask)

        def forward(self, enc_inp, dec_inp, dec_seq_pos, ctrl_dict,
                    padding_mask=None):
            bt, n_seqs = enc_inp.size(1), enc_inp.size(2)
            enc_emb = self.token_emb_lyric(enc_inp)
            dec_emb = self.token_emb(dec_inp)
            enc_emb = enc_emb.reshape(enc_inp.size(0), -1, enc_emb.size(-1))
            enc_inp_drop = self.emb_drop(enc_emb) + self.pe(enc_inp.size(0))
            dec_inp_drop = self.emb_drop(dec_emb) + self.pe(dec_inp.size(0))
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(-1, padding_mask.size(-1))
            lyric_latent  = self.encoder(enc_inp_drop, padding_mask)
            lat_reshaped  = lyric_latent.reshape(bt, n_seqs, -1)
            seg_emb = torch.zeros(dec_inp.size(0), dec_inp.size(1), self.d_latent,
                                  device=lyric_latent.device)
            for n in range(dec_inp.size(1)):
                for b, (st, ed) in enumerate(zip(dec_seq_pos[n, :-1], dec_seq_pos[n, 1:])):
                    seg_emb[st:ed, n, :] = lat_reshaped[n, b, :]
            seg_cat   = self._cat_controls(seg_emb, ctrl_dict)
            dec_out   = self.decoder(dec_inp_drop, seg_cat)
            dec_logits = self.dec_proj(dec_out)
            return dec_logits

        def generate_step(self, dec_inp, seg_emb_full, ctrl_dict, keep_last=True):
            emb = self.emb_drop(self.token_emb(dec_inp)) + self.pe(dec_inp.size(0))
            seg_cat = self._cat_controls(seg_emb_full, ctrl_dict)
            out    = self.decoder(emb, seg_cat)
            logits = self.dec_proj(out)
            return logits[-1] if keep_last else logits

        def compute_loss(self, dec_logits, dec_tgt):
            return F.cross_entropy(
                dec_logits.view(-1, dec_logits.size(-1)),
                dec_tgt.contiguous().view(-1),
                ignore_index=self.pad_token_melody, reduction='mean'
            ).float()

    return {
        'VQVAE':     VQVAE,
        'CSLL2M':    CSLL2M,
        'PositionalEncoding': PositionalEncoding,
        'TokenEmbedding':     TokenEmbedding,
        'TransformerEncoder': TransformerEncoder,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _build_dataset_class():
    import torch
    import numpy as np
    from torch.utils.data import Dataset

    class REMIAlignedDataset(Dataset):
        def __init__(self, events_dir, vocab_melody, vocab_lyric, attr_dir,
                     pieces, enc_seqlen=128, dec_seqlen=2048, max_seqs=64,
                     pad_to_same=True, use_controls=True,
                     learned_feats_dir=None, struct_dir=None, key_dir=None, emotion_dir=None):
            self.events_dir       = Path(events_dir)
            self.attr_dir         = Path(attr_dir) if attr_dir else None
            self.learned_feats_dir = Path(learned_feats_dir) if learned_feats_dir else None
            self.struct_dir       = Path(struct_dir) if struct_dir else None
            self.key_dir          = Path(key_dir) if key_dir else None
            self.emotion_dir      = Path(emotion_dir) if emotion_dir else None

            self.enc_seqlen   = enc_seqlen
            self.dec_seqlen   = dec_seqlen
            self.max_seqs     = max_seqs
            self.pad_to_same  = pad_to_same
            self.use_controls = use_controls

            # Vocabularios
            self.event2idx, self.idx2event = _pkl_load(vocab_melody)
            self.lyric2idx, self.idx2lyric = _pkl_load(vocab_lyric)
            self.bar_token        = self.event2idx['Bar_None']
            self.eos_token        = self.event2idx['EOS_None']
            self.pad_token_melody = len(self.event2idx)
            self.vocab_size       = len(self.event2idx) + 1
            self.pad_token_lyric  = len(self.lyric2idx) + 1
            self.vocab_size_lyric = len(self.lyric2idx) + 2
            self.seq_token_lyric  = len(self.lyric2idx)

            # Lista de piezas y posiciones de SEQ
            self.pieces        = sorted([self.events_dir / p for p in pieces])
            self.piece_seq_pos = []
            for i, p in enumerate(self.pieces):
                _, seq_pos, evs = _pkl_load(p)
                if seq_pos[-1] == len(evs):
                    seq_pos = seq_pos[:-1]
                if len(evs) - seq_pos[-1] == 2:
                    seq_pos = seq_pos[:-1]
                seq_pos.append(len(evs))
                self.piece_seq_pos.append(seq_pos)

        def __len__(self):
            return len(self.pieces)

        def _pad_seq(self, seq, maxlen, val):
            seq = list(seq)
            seq.extend([val] * (maxlen - len(seq)))
            return seq

        def _enc_input(self, seq_lyrics):
            mask = np.ones((self.max_seqs, self.enc_seqlen), dtype=bool)
            mask[:, :2] = False
            enc  = np.full((self.max_seqs, self.enc_seqlen), self.pad_token_lyric, dtype=int)
            for i, phrase in enumerate(seq_lyrics):
                tokens = [self.seq_token_lyric] + [self.lyric2idx.get(ch, 0) for ch in phrase]
                mask[i, :len(tokens)] = False
                enc[i, :] = self._pad_seq(tokens, self.enc_seqlen, self.pad_token_lyric)
            return enc, mask

        def __getitem__(self, idx):
            p          = self.pieces[idx]
            seq_lyrics, _, evs = _pkl_load(p)
            seq_pos    = self.piece_seq_pos[idx]
            n_seqs_all = len(seq_pos) - 1

            if n_seqs_all > self.max_seqs:
                st = random.randrange(n_seqs_all - self.max_seqs)
            else:
                st = 0
            n_seqs = min(n_seqs_all - st, self.max_seqs)

            # Slice eventos y posiciones de SEQ
            ev_st  = seq_pos[st]
            ev_ed  = seq_pos[st + n_seqs] if st + n_seqs < len(seq_pos) else len(evs)
            sliced_evs = evs[ev_st:ev_ed]
            sliced_pos = np.array(seq_pos[st:st + n_seqs + 1]) - ev_st
            padded_pos = np.array(
                sliced_pos.tolist() + [sliced_pos[-1]] * (self.max_seqs + 1 - len(sliced_pos))
            )

            # Tokens de melodía
            mel_tokens = [self.event2idx.get(f"{e['name']}_{e['value']}", 0) for e in sliced_evs]
            length = len(mel_tokens)
            if self.pad_to_same:
                mel_tokens = self._pad_seq(mel_tokens, self.dec_seqlen + 1, self.pad_token_melody)
            else:
                mel_tokens = mel_tokens + [self.pad_token_melody]
            target = np.array(mel_tokens[1:self.dec_seqlen + 1], dtype=int)
            inp    = np.array(mel_tokens[:self.dec_seqlen], dtype=int)

            # Encoder input (letras)
            phrase_sl  = seq_lyrics[st:st + n_seqs]
            enc_inp, enc_mask = self._enc_input(phrase_sl)

            item = {
                'piece_id':       p.stem,
                'st_seq_id':      st,
                'enc_n_seqs':     n_seqs,
                'seq_pos':        padded_pos,
                'enc_input':      enc_inp,
                'enc_padding_mask': enc_mask,
                'dec_input':      inp,
                'dec_target':     target,
                'length':         min(length, self.dec_seqlen),
            }

            if self.use_controls and self.attr_dir is not None:
                attr_path = self.attr_dir / p.name
                if attr_path.exists():
                    attrs = _pkl_load(attr_path)
                    for a in ATTR_NAMES:
                        vals = attrs.get(a, [])
                        sliced = vals[st:st + self.max_seqs]
                        sliced = sliced + [0] * (self.max_seqs - len(sliced))
                        exp = np.zeros(self.dec_seqlen, dtype=int)
                        for i, (s, e) in enumerate(zip(padded_pos[:-1], padded_pos[1:])):
                            if i < len(sliced):
                                exp[s:e] = sliced[i]
                        item[a] = exp
                        item[f'{a}_seq'] = np.array(sliced, dtype=int)

                # Key global
                if self.key_dir:
                    k = _pkl_load(self.key_dir / p.name)
                    item['global_key'] = KEY_DICT.get(k, 7)
                else:
                    item['global_key'] = 7
                key_exp = np.full(self.dec_seqlen, item['global_key'], dtype=int)
                item['key'] = key_exp

                # Emotion global
                if self.emotion_dir:
                    em = _pkl_load(self.emotion_dir / p.name)
                    item['global_emotion'] = EMOTION_DICT.get(em, 0)
                else:
                    item['global_emotion'] = 0
                emo_exp = np.full(self.dec_seqlen, item['global_emotion'], dtype=int)
                item['emotion'] = emo_exp

                # Struct (etiqueta por frase, A/B/C/D/E)
                if self.struct_dir:
                    struct_vals = _pkl_load(self.struct_dir / p.name)
                    sliced_st   = struct_vals[st:st + self.max_seqs]
                    if sliced_st and isinstance(sliced_st[0], str):
                        sliced_st = [STRUCT_DICT.get(s, 0) for s in sliced_st]
                    sliced_st = sliced_st + [0] * (self.max_seqs - len(sliced_st))
                    exp_st = np.zeros(self.dec_seqlen, dtype=int)
                    for i, (s, e) in enumerate(zip(padded_pos[:-1], padded_pos[1:])):
                        if i < len(sliced_st):
                            exp_st[s:e] = sliced_st[i]
                    item['struct']     = exp_st
                    item['struct_seq'] = np.array(sliced_st, dtype=int)
                else:
                    item['struct']     = np.zeros(self.dec_seqlen, dtype=int)
                    item['struct_seq'] = np.zeros(self.max_seqs, dtype=int)

                # Learned features (VQ-VAE)
                if self.learned_feats_dir:
                    feat_path = self.learned_feats_dir / p.name
                    if feat_path.exists():
                        feats = _pkl_load(feat_path)            # (n_seqs_all, d)
                        feats_sl = feats[st:st + self.max_seqs]
                        d = feats.shape[-1]
                        feats_pad = np.zeros((self.max_seqs, d), dtype=np.float32)
                        feats_pad[:len(feats_sl)] = feats_sl
                        exp_f = np.zeros((self.dec_seqlen, d), dtype=np.float32)
                        for i, (s, e) in enumerate(zip(padded_pos[:-1], padded_pos[1:])):
                            if i < self.max_seqs:
                                exp_f[s:e] = feats_pad[i]
                        item['learned_feats']     = exp_f
                        item['learned_feats_seq'] = feats_pad

            return item

    return REMIAlignedDataset


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN VQ-VAE
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train_vqvae(args):
    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    print(f'[train-vqvae] Dispositivo: {device}')

    classes    = _build_model_classes()
    DSClass    = _build_dataset_class()
    VQVAE_cls  = classes['VQVAE']

    # Dividir las piezas en train/val 90/10
    all_pieces = [p.name for p in Path(args.events_dir).glob('*.pkl')]
    if not all_pieces:
        print(f'[train-vqvae] ✗  No se encontraron .pkl en {args.events_dir}')
        sys.exit(1)

    random.seed(42)
    random.shuffle(all_pieces)

    if len(all_pieces) == 1:
        # Corpus mínimo: misma pieza en train y val
        tr_pieces  = all_pieces
        val_pieces = all_pieces
        print(f'[train-vqvae] ⚠  Solo 1 pieza — se usa para train y val')
    else:
        n_val      = max(1, int(len(all_pieces) * 0.1))
        n_val      = min(n_val, len(all_pieces) - 1)   # siempre al menos 1 en train
        tr_pieces  = all_pieces[n_val:]
        val_pieces = all_pieces[:n_val]

    dset     = DSClass(args.events_dir, args.vocab, args.vocab,
                       attr_dir=None, pieces=tr_pieces,
                       pad_to_same=True, use_controls=False)
    dset_val = DSClass(args.events_dir, args.vocab, args.vocab,
                       attr_dir=None, pieces=val_pieces,
                       pad_to_same=True, use_controls=False)

    print(f'[train-vqvae] Train={len(dset)}  Val={len(dset_val)}')

    dloader     = DataLoader(dset,     batch_size=args.batch_size, shuffle=True,  num_workers=0)
    dloader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VQVAE_cls(
        args.enc_layers, args.enc_heads, args.enc_dim, args.enc_dim * 4,
        args.dec_layers, args.dec_heads, args.dec_dim, args.dec_dim * 4,
        args.d_latent, args.d_embed, dset.vocab_size,
        n_codes=args.n_codes, n_groups=args.n_groups,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train-vqvae] Parámetros: {n_params:,}')

    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(dloader),
                                                      eta_min=args.lr * 0.05)
    best_val  = float('inf')

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        t0 = time.time()
        n_nan = 0
        for batch in dloader:
            optimizer.zero_grad()
            dec_raw = batch['dec_input'].long()          # (B, dec_seqlen)
            pos_raw = batch['seq_pos'].long()            # (B, max_seqs+1)
            B, dec_sl = dec_raw.shape
            enc_sl  = 128
            n_seqs  = pos_raw.shape[1] - 1
            pad_val = dset.pad_token_melody
            # Construir enc_inp: (enc_sl, B, n_seqs)
            enc_arr = torch.full((enc_sl, B, n_seqs), pad_val, dtype=torch.long)
            msk_arr = torch.ones(B, n_seqs, enc_sl, dtype=torch.bool)
            for b in range(B):
                for s in range(n_seqs):
                    st = pos_raw[b, s].item()
                    ed = pos_raw[b, s+1].item()
                    seg = dec_raw[b, st:ed][:enc_sl]
                    L = len(seg)
                    if L > 0:
                        enc_arr[:L, b, s] = seg
                        msk_arr[b, s, :L] = False
            # Garantizar que al menos la posición 0 de cada frase no esté enmascarada
            # (evita atención completamente enmascarada → NaN en softmax)
            msk_arr[:, :, 0] = False
            enc = enc_arr.to(device)
            msk = msk_arr.to(device)
            dec = dec_raw.permute(1, 0).long().to(device)
            tgt = batch['dec_target'].permute(1, 0).long().to(device)
            pos = pos_raw.to(device)
            _, diff, logits = model(enc, dec, pos, msk)
            losses = model.compute_loss(diff, logits, tgt, beta=0.1)
            loss = losses['total']
            if torch.isnan(loss):
                n_nan += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            scheduler.step()
            ep_loss += losses['recons'].item()

        ep_loss /= max(1, len(dloader) - n_nan)
        elapsed  = time.time() - t0

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dloader_val:
                dec_raw = batch['dec_input'].long()
                pos_raw = batch['seq_pos'].long()
                B, dec_sl = dec_raw.shape
                n_seqs  = pos_raw.shape[1] - 1
                enc_arr = torch.full((enc_sl, B, n_seqs), pad_val, dtype=torch.long)
                msk_arr = torch.ones(B, n_seqs, enc_sl, dtype=torch.bool)
                for b in range(B):
                    for s in range(n_seqs):
                        st = pos_raw[b, s].item()
                        ed = pos_raw[b, s+1].item()
                        seg = dec_raw[b, st:ed][:enc_sl]
                        L = len(seg)
                        if L > 0:
                            enc_arr[:L, b, s] = seg
                            msk_arr[b, s, :L] = False
                msk_arr[:, :, 0] = False   # evitar atención totalmente enmascarada
                enc = enc_arr.to(device)
                msk = msk_arr.to(device)
                dec = dec_raw.permute(1, 0).long().to(device)
                tgt = batch['dec_target'].permute(1, 0).long().to(device)
                pos = pos_raw.to(device)
                _, diff, logits = model(enc, dec, pos, msk)
                vl = model.compute_loss(diff, logits, tgt)['recons']
                if not torch.isnan(vl):
                    val_loss += vl.item()
        val_loss /= max(1, len(dloader_val))

        marker = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / 'best_vqvae.pt')
            marker = '  ✓ mejor'

        print(f'  ep {ep+1:03d}/{args.epochs}  '
              f'train={ep_loss:.4f}  val={val_loss:.4f}  '
              f't={elapsed:.1f}s{marker}')

    # Guardar config del modelo para extract-feats
    cfg = {
        'enc_layers': args.enc_layers, 'enc_heads': args.enc_heads,
        'enc_dim': args.enc_dim, 'dec_layers': args.dec_layers,
        'dec_heads': args.dec_heads, 'dec_dim': args.dec_dim,
        'd_latent': args.d_latent, 'd_embed': args.d_embed,
        'n_codes': args.n_codes, 'n_groups': args.n_groups,
    }
    with open(out_dir / 'vqvae_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'[train-vqvae] ✓  Modelo guardado en {out_dir}')


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACT-FEATS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_extract_feats(args):
    import torch
    import numpy as np

    device  = torch.device('cpu')
    classes = _build_model_classes()
    VQVAE   = classes['VQVAE']

    model_dir  = Path(args.model_dir)
    cfg_path   = model_dir / 'vqvae_config.json'
    if not cfg_path.exists():
        print(f'[extract-feats] ✗  No se encontró {cfg_path}')
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    vocab_mel = args.vocab if hasattr(args, 'vocab') else args.vocab_melody
    event2idx, _ = _pkl_load(vocab_mel)
    vocab_size    = len(event2idx) + 1

    model = VQVAE(
        cfg['enc_layers'], cfg['enc_heads'], cfg['enc_dim'], cfg['enc_dim'] * 4,
        cfg['dec_layers'], cfg['dec_heads'], cfg['dec_dim'], cfg['dec_dim'] * 4,
        cfg['d_latent'], cfg['d_embed'], vocab_size,
        n_codes=cfg['n_codes'], n_groups=cfg['n_groups'],
    ).to(device)
    model.load_state_dict(torch.load(model_dir / 'best_vqvae.pt', map_location='cpu'))
    model.eval()

    events_dir = Path(args.events_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted(events_dir.glob('*.pkl'))
    print(f'[extract-feats] {len(pkl_files)} canciones  →  {output_dir}')

    for p in pkl_files:
        try:
            _, seq_pos, events = _pkl_load(p)
            # Tokenizar
            tokens = [event2idx.get(f"{e['name']}_{e['value']}", 0) for e in events]
            # Por frase (SEQ), encodear
            feats = []
            n_seqs = len(seq_pos) - 1
            enc_seqlen = 128
            pad_tok    = vocab_size - 1
            for i in range(n_seqs):
                st, ed = seq_pos[i], seq_pos[i + 1] if i + 1 < len(seq_pos) else len(tokens)
                seg = tokens[st:ed]
                seg = (seg + [pad_tok] * enc_seqlen)[:enc_seqlen]
                inp = torch.tensor([seg], dtype=torch.long).t()  # (L, 1)
                with torch.no_grad():
                    z = model.get_latent(inp)
                feats.append(z.squeeze(0).numpy())
            out = np.array(feats, dtype=np.float32)
            _pkl_save(out, output_dir / p.name)
            print(f'  ✓  {p.stem}  shape={out.shape}')
        except Exception as e:
            print(f'  ✗  {p.stem}  {e}')

    print(f'[extract-feats] ✓  Completado')


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN CSL-L2M
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import torch
    from torch import optim
    from torch.utils.data import DataLoader

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu'
                          else 'cpu')
    print(f'[train] Dispositivo: {device}')

    classes  = _build_model_classes()
    DSClass  = _build_dataset_class()
    CSLL2M   = classes['CSLL2M']

    tr_pieces  = _pkl_load(args.train_split)
    val_pieces = _pkl_load(args.val_split)

    dset_kw = dict(
        events_dir=args.events_dir,
        vocab_melody=args.vocab_melody,
        vocab_lyric=args.vocab_lyric,
        attr_dir=args.attr_dir,
        enc_seqlen=args.enc_seqlen,
        dec_seqlen=args.dec_seqlen,
        max_seqs=args.max_seqs,
        use_controls=True,
        struct_dir=args.struct_dir or None,
        key_dir=args.key_dir or None,
        emotion_dir=args.emotion_dir or None,
        learned_feats_dir=args.feats_dir or None,
    )
    dset     = DSClass(**dset_kw, pieces=tr_pieces,  pad_to_same=True)
    dset_val = DSClass(**dset_kw, pieces=val_pieces, pad_to_same=True)

    print(f'[train] Train={len(dset)}  Val={len(dset_val)}')
    dloader     = DataLoader(dset,     batch_size=args.batch_size, shuffle=True,  num_workers=0)
    dloader_val = DataLoader(dset_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Determinar controles activos
    controls = set()
    ctrl_flags = ['key', 'emotion', 'struct', 'Align', 'ND',
                  'PM', 'PV', 'PR', 'DMM', 'AA', 'CM', 'DM', 'DV', 'DR', 'MCD']
    for c in ctrl_flags:
        if getattr(args, f'no_{c.lower()}', False) is False:
            controls.add(c)

    # Determinar d_learned desde el config del VQ-VAE si se usan features aprendidas
    d_learned = 128   # default si no hay VQ-VAE
    if args.feats_dir:
        controls.add('learned_feats')
        vqvae_cfg_path = Path(args.feats_dir).parent / 'model_vqvae' / 'vqvae_config.json'
        # Buscar vqvae_config.json: primero junto a feats_dir, luego model_vqvae/
        for candidate in [
            Path(args.feats_dir).parent / 'vqvae_config.json',
            Path(args.feats_dir).parent / 'model_vqvae' / 'vqvae_config.json',
            Path('model_vqvae') / 'vqvae_config.json',
        ]:
            if candidate.exists():
                with open(candidate) as f:
                    vcfg = json.load(f)
                d_learned = vcfg.get('d_latent', 128)
                print(f'[train] VQ-VAE d_latent={d_learned}  ({candidate})')
                break
        else:
            print(f'[train] ⚠  No se encontró vqvae_config.json — asumiendo d_learned={d_learned}')

    model = CSLL2M(
        args.enc_layers, args.enc_heads, args.enc_dim, args.enc_dim * 4,
        args.dec_layers, args.dec_heads, args.dec_dim, args.dec_dim * 4,
        args.d_latent, args.d_embed,
        dset.vocab_size, dset.vocab_size_lyric, dset.pad_token_melody,
        controls=controls, d_learned=d_learned,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train] Parámetros: {n_params:,}  |  Controles: {sorted(controls)}')

    out_dir = Path(args.model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Guardar config
    cfg = {
        'enc_layers': args.enc_layers, 'enc_heads': args.enc_heads, 'enc_dim': args.enc_dim,
        'dec_layers': args.dec_layers, 'dec_heads': args.dec_heads, 'dec_dim': args.dec_dim,
        'd_latent': args.d_latent, 'd_embed': args.d_embed, 'd_learned': d_learned,
        'controls': sorted(controls),
        'vocab_melody': str(args.vocab_melody),
        'vocab_lyric':  str(args.vocab_lyric),
        'attr_dir':     str(args.attr_dir),
    }
    with open(out_dir / 'model_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    optimizer = optim.Adam(model.parameters(), lr=args.max_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * len(dloader), eta_min=args.min_lr
    )
    best_val  = float('inf')
    step      = 0

    def _to_ctrl(batch, device, controls):
        """Extrae y permuta los tensores de control."""
        cd = {}
        ctrl_keys = ['key', 'emotion', 'struct'] + ATTR_NAMES
        for k in ctrl_keys:
            if k in controls and k in batch:
                cd[k] = batch[k].permute(1, 0).long().to(device)
        if 'learned_feats' in controls and 'learned_feats' in batch:
            # batch['learned_feats']: (B, dec_seqlen, d) → (dec_seqlen, B, d)
            cd['learned_feats'] = batch['learned_feats'].permute(1, 0, 2).float().to(device)
        return cd

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        t0 = time.time()
        for batch in dloader:
            optimizer.zero_grad()
            enc = batch['enc_input'].permute(2, 0, 1).long().to(device)
            dec = batch['dec_input'].permute(1, 0).long().to(device)
            tgt = batch['dec_target'].permute(1, 0).long().to(device)
            pos = batch['seq_pos'].to(device)
            msk = batch['enc_padding_mask'].bool()
            msk[:, :, 0] = False   # evitar filas totalmente enmascaradas → NaN
            msk = msk.to(device)
            cd  = _to_ctrl(batch, device, controls)

            logits = model(enc, dec, pos, cd, padding_mask=msk)
            loss   = model.compute_loss(logits, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            ep_loss += loss.item()
            step    += 1

        ep_loss /= len(dloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dloader_val:
                enc = batch['enc_input'].permute(2, 0, 1).long().to(device)
                dec = batch['dec_input'].permute(1, 0).long().to(device)
                tgt = batch['dec_target'].permute(1, 0).long().to(device)
                pos = batch['seq_pos'].to(device)
                msk = batch['enc_padding_mask'].bool()
                msk[:, :, 0] = False
                msk = msk.to(device)
                cd  = _to_ctrl(batch, device, controls)
                logits = model(enc, dec, pos, cd, padding_mask=msk)
                vl = model.compute_loss(logits, tgt)
                if not torch.isnan(vl):
                    val_loss += vl.item()
        val_loss /= max(1, len(dloader_val))
        elapsed   = time.time() - t0

        marker = ''
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / 'best_csll2m.pt')
            marker = '  ✓ mejor'

        print(f'  ep {ep+1:03d}/{args.epochs}  '
              f'train={ep_loss:.4f}  val={val_loss:.4f}  '
              f'steps={step}  t={elapsed:.1f}s{marker}')

    print(f'[train] ✓  Modelo guardado en {out_dir}')


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATE
# ══════════════════════════════════════════════════════════════════════════════

def _temperatured_softmax(logits, temperature):
    import numpy as np
    try:
        probs = np.exp(logits / temperature)
        probs /= probs.sum()
        if np.any(np.isnan(probs)):
            raise ValueError
    except (ValueError, OverflowError):
        logits = logits.astype(np.float64)
        probs  = np.exp(logits / temperature)
        probs /= probs.sum()
    return probs


def _nucleus(probs, p):
    import numpy as np
    probs /= probs.sum()
    idx_sorted = np.argsort(probs)[::-1]
    cumsum     = np.cumsum(probs[idx_sorted])
    cutoff     = np.where(cumsum > p)[0]
    last       = idx_sorted[cutoff[1]] if len(cutoff) > 1 else idx_sorted[min(2, len(idx_sorted)-1)]
    candidates = idx_sorted[:np.where(idx_sorted == last)[0][0] + 1]
    cprobs     = probs[candidates]
    cprobs    /= cprobs.sum()
    return int(np.random.choice(candidates, p=cprobs))


def _remi_to_midi(seq_lyrics, events, tempo, output_path):
    """Convierte secuencia REMI-Aligned a MIDI con letras."""
    try:
        import miditoolkit
    except ImportError:
        print('[generate] ✗  miditoolkit no instalado')
        return

    DEFAULT_FRACTION = 64
    DEFAULT_BAR_RESOL = 480 * 4

    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.lyrics = []
    midi_obj.instruments = [miditoolkit.Instrument(program=0, is_drum=False, name='Piano')]
    midi_obj.tempo_changes.append(miditoolkit.TempoChange(tempo, 0))

    temp_notes = []
    cur_bar = cur_pos = 0
    num_notes = 0
    pp = 0
    flat_lyrics = [ch for phrase in seq_lyrics for ch in phrase]

    for i, ev in enumerate(events):
        name = ev['name'] if isinstance(ev, dict) else ev.split('_')[0]
        val  = ev['value'] if isinstance(ev, dict) else '_'.join(ev.split('_')[1:])

        if name == 'Bar' and i > 1:
            cur_bar += 1
        elif name == 'Beat':
            cur_pos = int(val)
        elif name == 'Note_Pitch' and i + 1 < len(events):
            nxt = events[i + 1]
            nxt_name = nxt['name'] if isinstance(nxt, dict) else nxt.split('_')[0]
            if 'Note_Duration' in nxt_name:
                dur   = int(nxt['value'] if isinstance(nxt, dict) else nxt.split('_')[-1])
                start = cur_bar * DEFAULT_BAR_RESOL + cur_pos * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
                temp_notes.append((int(val), start, dur))
                num_notes += 1
        elif name == 'ALIGN':
            if num_notes >= 1 and pp < len(flat_lyrics):
                start_t = temp_notes[-num_notes][1]
                midi_obj.lyrics.append(
                    miditoolkit.Lyric(text=flat_lyrics[pp], time=start_t)
                )
                for k in range(num_notes - 1):
                    midi_obj.lyrics.append(
                        miditoolkit.Lyric(text='*', time=temp_notes[-num_notes + k + 1][1])
                    )
                pp = min(pp + 1, len(flat_lyrics) - 1)
            num_notes = 0
        elif name in ('EOS', 'PAD'):
            break

    for pitch, start, dur in temp_notes:
        midi_obj.instruments[0].notes.append(
            miditoolkit.Note(126, pitch, int(start), int(start + dur))
        )

    if output_path:
        midi_obj.dump(str(output_path), charset='utf-8')
    return midi_obj


def cmd_generate(args):
    import torch
    import numpy as np

    device  = torch.device('cpu')
    classes = _build_model_classes()
    CSLL2M  = classes['CSLL2M']

    model_dir = Path(args.model_dir)
    cfg_path  = model_dir / 'model_config.json'
    if not cfg_path.exists():
        print(f'[generate] ✗  No se encontró {cfg_path}')
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Vocabularios
    event2idx, idx2event = _pkl_load(cfg['vocab_melody'])
    lyric2idx, idx2lyric = _pkl_load(cfg['vocab_lyric'])
    vocab_size       = len(event2idx) + 1
    vocab_size_lyric = len(lyric2idx) + 2
    pad_token_melody = len(event2idx)
    pad_token_lyric  = len(lyric2idx) + 1
    seq_token_lyric  = len(lyric2idx)

    controls = set(cfg.get('controls', []))
    d_learned = cfg.get('d_learned', 128)
    model = CSLL2M(
        cfg['enc_layers'], cfg['enc_heads'], cfg['enc_dim'], cfg['enc_dim'] * 4,
        cfg['dec_layers'], cfg['dec_heads'], cfg['dec_dim'], cfg['dec_dim'] * 4,
        cfg['d_latent'], cfg['d_embed'],
        vocab_size, vocab_size_lyric, pad_token_melody,
        controls=controls, d_learned=d_learned,
    ).to(device)
    ckpt = model_dir / 'best_csll2m.pt'
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=False)
    model.eval()

    # Cargar la pieza
    events_path = Path(args.events)
    seq_lyrics, seq_pos, raw_events = _pkl_load(events_path)
    n_seqs = len(seq_lyrics)

    # Atributos
    attr_dir  = Path(cfg.get('attr_dir', args.attr_dir or ''))
    attr_path = attr_dir / events_path.name
    if attr_path.exists():
        attrs = _pkl_load(attr_path)
    else:
        attrs = {a: [32] * n_seqs for a in ATTR_NAMES}

    # Desplazamientos de atributos
    def _shifted(a_name, shift):
        return [max(0, min(ATTR_DIM - 1, v + shift)) for v in attrs.get(a_name, [32] * n_seqs)]

    # Controles globales
    key_val     = KEY_DICT.get(args.key, 7) if args.key else 7
    emotion_val = EMOTION_DICT.get(args.emotion, 0) if args.emotion else 0

    # Latentes de letra (encoder run)
    enc_seqlen = cfg.get('enc_seqlen', 128)
    lyric_latents = []
    with torch.no_grad():
        for phrase in seq_lyrics:
            tokens = [seq_token_lyric] + [lyric2idx.get(ch, 0) for ch in phrase]
            tokens = (tokens + [pad_token_lyric] * enc_seqlen)[:enc_seqlen]
            inp    = torch.tensor(tokens, dtype=torch.long).unsqueeze(1)  # (L, 1)
            lat    = model.get_lyric_emb(inp)
            lyric_latents.append(lat.squeeze(0))

    max_events = args.dec_seqlen
    d_lat      = cfg['d_latent']

    # Calcular dimensión total del seg_emb para el placeholder
    ctrl_cfg_ref = CSLL2M._CTRL_CFG
    d_seg = d_lat + sum(ctrl_cfg_ref[c][0] for c in controls if c in ctrl_cfg_ref)
    if 'learned_feats' in controls:
        d_seg += cfg.get('d_learned', 128)

    def _run_sample():
        seg_placeholder  = torch.zeros(max_events, 1, d_lat)
        ctrl_placeholders = {c: torch.zeros(max_events, 1, dtype=torch.long)
                             for c in controls if c in ctrl_cfg_ref}
        # Placeholder para learned_feats (zeros = estilo neutro)
        d_learned_gen = cfg.get('d_learned', 128)
        if 'learned_feats' in controls:
            feats_placeholder = torch.zeros(max_events, 1, d_learned_gen)

        # Límites válidos por control
        ctrl_maxval = {**{a: ATTR_DIM - 1 for a in ATTR_NAMES},
                       'key': 23, 'emotion': 2, 'struct': 4}

        generated    = [event2idx.get('SEQ_None', 0)]
        gen_bars     = 0
        target_bars  = n_seqs
        failed_pos   = failed_eos = seq_align = 0
        last_tick    = cur_dur = cur_bar = 0
        DEFAULT_FRAC      = 64
        DEFAULT_BAR_RESOL = BEAT_RESOL * 4

        while gen_bars < target_bars:
            pos_idx = len(generated) - 1
            if pos_idx >= max_events:
                break
            seg_placeholder[pos_idx, 0, :] = lyric_latents[min(gen_bars, len(lyric_latents)-1)]

            # Rellenar controles con clamp
            for c in ctrl_placeholders:
                maxv = ctrl_maxval.get(c, 63)
                if c == 'key':
                    val = key_val
                elif c == 'emotion':
                    val = emotion_val
                elif c == 'struct':
                    val = 0
                else:
                    seq_vals = _shifted(c, getattr(args, f'shift_{c.lower()}', 0) or 0)
                    val = seq_vals[gen_bars] if gen_bars < len(seq_vals) else 0
                ctrl_placeholders[c][pos_idx, 0] = max(0, min(int(val), maxv))

            # Input token
            if len(generated) == 1:
                dec_inp = torch.tensor([generated], dtype=torch.long)
            else:
                dec_inp = torch.tensor([generated], dtype=torch.long).permute(1, 0)

            seg_now  = seg_placeholder[:len(generated), :]
            ctrl_now = {c: ctrl_placeholders[c][:len(generated), :]
                        for c in ctrl_placeholders}
            if 'learned_feats' in controls:
                ctrl_now['learned_feats'] = feats_placeholder[:len(generated), :]

            with torch.no_grad():
                logits = model.generate_step(dec_inp, seg_now, ctrl_now)
            logits_np = logits.numpy().flatten()
            probs     = _temperatured_softmax(logits_np, args.temperature)

            # Forzar / bloquear SEQ según sincronía de letras
            n_aligns = len(seq_lyrics[gen_bars]) if gen_bars < target_bars else 0
            seq_tok  = event2idx.get('SEQ_None', 0)
            if seq_align >= n_aligns:
                word = seq_tok
            else:
                word = _nucleus(probs, args.nucleus_p)
                if word == seq_tok:
                    probs[seq_tok] = 0.0
                    probs /= probs.sum() + 1e-9
                    word = _nucleus(probs, args.nucleus_p)

            word_ev = idx2event.get(word, 'PAD_None')

            # Validación de posición Beat
            if 'Beat' in word_ev:
                beat_pos = int(word_ev.split('_')[-1])
                start_t  = cur_bar * DEFAULT_BAR_RESOL + beat_pos * (DEFAULT_BAR_RESOL // DEFAULT_FRAC)
                if start_t < last_tick + cur_dur:
                    failed_pos += 1
                    if failed_pos >= 128:
                        print('[generate] ⚠  Modelo bloqueado (posición), abortando')
                        break
                    continue
                last_tick  = start_t
                failed_pos = 0

            if 'Bar' in word_ev and len(generated) > 2:
                cur_bar += 1
            if 'Note_Duration' in word_ev:
                cur_dur = int(word_ev.split('_')[-1])
            if 'ALIGN' in word_ev:
                seq_align += 1
            if 'SEQ' in word_ev:
                gen_bars  += 1
                seq_align  = 0
            if gen_bars < target_bars - 1 and 'EOS' in word_ev:
                failed_eos += 1
                if failed_eos >= 128:
                    print('[generate] ⚠  EOS prematuro repetido, abortando')
                    break
                continue
            if len(generated) >= max_events or ('EOS' in word_ev and gen_bars >= target_bars - 1):
                generated.append(event2idx.get('SEQ_None', 0))
                break

            generated.append(word)

        return [idx2event.get(w, 'PAD_None') for w in generated[:-1]]

    out_dir = Path(args.output).parent
    out_stem = Path(args.output).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n_samples):
        ev_seq = _run_sample()
        suffix = f'_{i+1:02d}' if args.n_samples > 1 else ''
        out_path = out_dir / f'{out_stem}{suffix}.mid'

        # Convertir eventos a dict para _remi_to_midi
        ev_dicts = []
        for e in ev_seq:
            parts = e.split('_', 1)
            ev_dicts.append({'name': parts[0], 'value': parts[1] if len(parts) > 1 else None})
        _remi_to_midi(seq_lyrics, ev_dicts, args.tempo, out_path)
        print(f'[generate] ✓  {out_path}  ({len(ev_seq)} eventos)')


# ══════════════════════════════════════════════════════════════════════════════
#  ROUND-TRIP — REMI-Aligned .pkl → MIDI (diagnóstico)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_round_trip(args):
    events_path = Path(args.events)
    if not events_path.exists():
        print(f'[round-trip] ✗  No se encontró {events_path}')
        sys.exit(1)

    seq_lyrics, _, events = _pkl_load(events_path)
    out_path = Path(args.output)
    _remi_to_midi(seq_lyrics, events, args.tempo, out_path)
    print(f'[round-trip] ✓  {out_path}  ({len(events)} eventos  /  {len(seq_lyrics)} frases)')


# ══════════════════════════════════════════════════════════════════════════════
#  INSPECT — diagnóstico
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    _header('INSPECT')

    # — Datos —
    if args.events_dir:
        events_dir = Path(args.events_dir)
        pkl_files  = sorted(events_dir.glob('*.pkl'))
        print(f'\n  Directorio eventos : {events_dir}')
        print(f'  Archivos .pkl      : {len(pkl_files)}')
        if pkl_files:
            total_evs, total_seqs = 0, 0
            for p in pkl_files[:args.max_show]:
                try:
                    sl, sp, evs = _pkl_load(p)
                    n_s = len(sl)
                    n_e = len(evs)
                    total_evs  += n_e
                    total_seqs += n_s
                    print(f'    {p.stem:<40}  {n_s:3d} frases  {n_e:5d} eventos')
                except Exception as e:
                    print(f'    {p.stem:<40}  ERROR: {e}')
            if len(pkl_files) > args.max_show:
                print(f'    … ({len(pkl_files) - args.max_show} más)')
            print(f'\n  Total eventos  : {total_evs}')
            print(f'  Total frases   : {total_seqs}')

    # — Vocabularios —
    if args.vocab_melody:
        e2i, i2e = _pkl_load(args.vocab_melody)
        print(f'\n  Vocabulario melodía : {len(e2i)} tokens')
        sample = list(e2i.keys())[:8]
        print(f'    Ejemplos: {sample}')
    if args.vocab_lyric:
        l2i, i2l = _pkl_load(args.vocab_lyric)
        print(f'  Vocabulario letra   : {len(l2i)} tokens')

    # — Atributos —
    if args.attr_dir:
        attr_dir = Path(args.attr_dir)
        bounds_f = attr_dir / 'attr_bounds.pkl'
        if bounds_f.exists():
            bounds = _pkl_load(bounds_f)
            print(f'\n  Atributos ({ATTR_DIM} clases/atributo):')
            for a in ATTR_NAMES:
                b = bounds.get(a, [])
                print(f'    {a:<8}  rango [{b[0]:.3f} … {b[-1]:.3f}]  fronteras={len(b)}')

    # — Checkpoint —
    if args.model_dir:
        model_dir = Path(args.model_dir)
        cfg_f     = model_dir / 'model_config.json'
        ckpt_f    = model_dir / 'best_csll2m.pt'
        vqvae_f   = model_dir / 'best_vqvae.pt'
        vqcfg_f   = model_dir / 'vqvae_config.json'
        print(f'\n  Directorio modelo  : {model_dir}')
        for label, path in [('CSL-L2M config', cfg_f), ('CSL-L2M ckpt', ckpt_f),
                             ('VQ-VAE config', vqcfg_f), ('VQ-VAE ckpt', vqvae_f)]:
            mark = '✓' if path.exists() else '✗'
            sz   = f'  {path.stat().st_size / 1e6:.1f} MB' if path.exists() else ''
            print(f'    {mark}  {label:<18}{sz}')
        if cfg_f.exists():
            with open(cfg_f) as f:
                cfg = json.load(f)
            print(f'\n  Controles activos  : {cfg.get("controls", [])}')
            for k in ['enc_layers', 'enc_heads', 'enc_dim', 'dec_layers', 'dec_heads', 'dec_dim',
                      'd_latent', 'd_embed']:
                print(f'    {k:<20} {cfg.get(k, "?")}')

    _separator()


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE + MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='csl_l2m',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            CSL-L2M v1.0 — Generación controlable de melodía a partir de letra.
            Adaptación single-file de CSL-L2M (AAAI-2025).

            Flujo típico:
              1. prepare    → convierte corpus MIDI+letras a REMI-Aligned
              2. vocab      → construye los diccionarios
              3. splits     → divide en train/val/test
              4. attributes → extrae atributos estadísticos
              5. train      → entrena el modelo
              6. generate   → genera melodías a partir de letras alineadas
        """),
    )
    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser('prepare', help='MIDI/ABC + letras → REMI-Aligned .pkl')
    p.add_argument('--input-dir',      required=True, metavar='DIR',
                   help='Carpeta con archivos .mid, .midi y/o .abc')
    p.add_argument('--output-dir',     required=True, metavar='DIR',
                   help='Carpeta de salida para los .pkl')
    p.add_argument('--default-tempo',  type=float, default=90.0, metavar='BPM',
                   help='Tempo por defecto para ABC sin cabecera Q: (default: 90)')
    p.set_defaults(func=cmd_prepare)

    # ── vocab ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('vocab', help='eventos .pkl → diccionarios')
    p.add_argument('--events-dir',  required=True, metavar='DIR')
    p.add_argument('--output-dir',  required=True, metavar='DIR')
    p.set_defaults(func=cmd_vocab)

    # ── splits ────────────────────────────────────────────────────────────────
    p = sub.add_parser('splits', help='Divide .pkl en train/val/test')
    p.add_argument('--events-dir',  required=True, metavar='DIR')
    p.add_argument('--output-dir',  required=True, metavar='DIR')
    p.add_argument('--val-ratio',   type=float, default=0.05, metavar='F')
    p.add_argument('--test-ratio',  type=float, default=0.05, metavar='F')
    p.add_argument('--seed',        type=int,   default=42)
    p.set_defaults(func=cmd_splits)

    # ── attributes ────────────────────────────────────────────────────────────
    p = sub.add_parser('attributes', help='12 atributos estadísticos por frase')
    p.add_argument('--events-dir',  required=True, metavar='DIR')
    p.add_argument('--output-dir',  required=True, metavar='DIR')
    p.set_defaults(func=cmd_attributes)

    # ── train-vqvae ───────────────────────────────────────────────────────────
    p = sub.add_parser('train-vqvae', help='Entrena VQ-VAE para features melódicas')
    p.add_argument('--events-dir',  required=True, metavar='DIR')
    p.add_argument('--vocab',       required=True, metavar='FILE',
                   help='dictionary_melody.pkl')
    p.add_argument('--model-dir',   required=True, metavar='DIR')
    p.add_argument('--epochs',      type=int,   default=12)
    p.add_argument('--batch-size',  type=int,   default=2)
    p.add_argument('--lr',          type=float, default=3e-5)
    # Arquitectura: defaults CPU-friendly (~1M params).
    # Para GPU usar: --enc-layers 12 --enc-dim 512 --dec-layers 12 --dec-dim 512
    p.add_argument('--enc-layers',  type=int,   default=2)
    p.add_argument('--enc-heads',   type=int,   default=4)
    p.add_argument('--enc-dim',     type=int,   default=128)
    p.add_argument('--dec-layers',  type=int,   default=2)
    p.add_argument('--dec-heads',   type=int,   default=4)
    p.add_argument('--dec-dim',     type=int,   default=128)
    p.add_argument('--d-latent',    type=int,   default=64)
    p.add_argument('--d-embed',     type=int,   default=128)
    p.add_argument('--n-codes',     type=int,   default=512)
    p.add_argument('--n-groups',    type=int,   default=16)
    p.add_argument('--device',      default='cpu')
    p.set_defaults(func=cmd_train_vqvae)

    # ── extract-feats ─────────────────────────────────────────────────────────
    p = sub.add_parser('extract-feats', help='VQ-VAE → features latentes por canción')
    p.add_argument('--events-dir',    required=True, metavar='DIR')
    p.add_argument('--vocab-melody',  required=True, metavar='FILE')
    p.add_argument('--model-dir',     required=True, metavar='DIR')
    p.add_argument('--output-dir',    required=True, metavar='DIR')
    p.set_defaults(func=cmd_extract_feats)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Entrena el modelo CSL-L2M')
    p.add_argument('--events-dir',    required=True, metavar='DIR')
    p.add_argument('--vocab-melody',  required=True, metavar='FILE')
    p.add_argument('--vocab-lyric',   required=True, metavar='FILE')
    p.add_argument('--train-split',   required=True, metavar='FILE')
    p.add_argument('--val-split',     required=True, metavar='FILE')
    p.add_argument('--attr-dir',      required=True, metavar='DIR')
    p.add_argument('--model-dir',     required=True, metavar='DIR')
    p.add_argument('--struct-dir',    default=None,  metavar='DIR',
                   help='Carpeta con etiquetas de estructura A/B/C… (.pkl por canción)')
    p.add_argument('--key-dir',       default=None,  metavar='DIR',
                   help='Carpeta con tonalidades (.pkl por canción, ej: "Dm")')
    p.add_argument('--emotion-dir',   default=None,  metavar='DIR',
                   help='Carpeta con emociones (.pkl por canción, ej: "Positive")')
    p.add_argument('--feats-dir',     default=None,  metavar='DIR',
                   help='Carpeta con features VQ-VAE (opcional)')
    p.add_argument('--epochs',        type=int,   default=12)
    p.add_argument('--batch-size',    type=int,   default=2)
    p.add_argument('--max-lr',        type=float, default=1e-4)
    p.add_argument('--min-lr',        type=float, default=5e-6)
    # Arquitectura: defaults CPU-friendly (~2M params).
    # Para GPU usar: --enc-layers 12 --enc-dim 512 --dec-layers 12 --dec-dim 512
    p.add_argument('--enc-layers',    type=int,   default=2)
    p.add_argument('--enc-heads',     type=int,   default=4)
    p.add_argument('--enc-dim',       type=int,   default=128)
    p.add_argument('--dec-layers',    type=int,   default=2)
    p.add_argument('--dec-heads',     type=int,   default=4)
    p.add_argument('--dec-dim',       type=int,   default=128)
    p.add_argument('--d-latent',      type=int,   default=64)
    p.add_argument('--d-embed',       type=int,   default=128)
    p.add_argument('--enc-seqlen',    type=int,   default=128)
    p.add_argument('--dec-seqlen',    type=int,   default=512)
    p.add_argument('--max-seqs',      type=int,   default=16)
    p.add_argument('--device',        default='cpu')
    # Flags para desactivar controles individuales
    for ctrl in ['key', 'emotion', 'struct', 'Align', 'ND',
                 'PM', 'PV', 'PR', 'DMM', 'AA', 'CM', 'DM', 'DV', 'DR', 'MCD']:
        p.add_argument(f'--no-{ctrl.lower()}', action='store_true',
                       help=f'Desactivar control {ctrl}')
    p.set_defaults(func=cmd_train)

    # ── generate ──────────────────────────────────────────────────────────────
    p = sub.add_parser('generate', help='Genera MIDI desde letras alineadas')
    p.add_argument('--events',        required=True, metavar='FILE',
                   help='Archivo .pkl con REMI-Aligned (output de prepare)')
    p.add_argument('--vocab-melody',  default=None,  metavar='FILE')
    p.add_argument('--vocab-lyric',   default=None,  metavar='FILE')
    p.add_argument('--attr-dir',      default=None,  metavar='DIR')
    p.add_argument('--model-dir',     required=True, metavar='DIR')
    p.add_argument('--output',        default='salida.mid', metavar='FILE')
    p.add_argument('--key',           default=None,  metavar='STR',
                   help='Tonalidad: C, Dm, F#m… (default: auto)')
    p.add_argument('--emotion',       default=None,
                   choices=['Positive', 'Neutral', 'Negative'])
    p.add_argument('--temperature',   type=float, default=1.2)
    p.add_argument('--nucleus-p',     type=float, default=0.9)
    p.add_argument('--tempo',         type=float, default=90.0)
    p.add_argument('--n-samples',     type=int,   default=1,
                   help='Número de melodías a generar (default: 1)')
    p.add_argument('--dec-seqlen',    type=int,   default=2048)
    # Desplazamientos de atributos
    for a in ATTR_NAMES:
        p.add_argument(f'--shift-{a.lower()}', type=int, default=0, metavar='N',
                       help=f'Desplazar atributo {a} ±N clases')
    p.set_defaults(func=cmd_generate)

    # ── round-trip ────────────────────────────────────────────────────────────
    p = sub.add_parser('round-trip', help='REMI-Aligned .pkl → MIDI (diagnóstico)')
    p.add_argument('--events',  required=True, metavar='FILE')
    p.add_argument('--output',  default='round_trip.mid', metavar='FILE')
    p.add_argument('--tempo',   type=float, default=90.0)
    p.set_defaults(func=cmd_round_trip)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect', help='Diagnóstico de datos y checkpoints')
    p.add_argument('--events-dir',   default=None, metavar='DIR')
    p.add_argument('--vocab-melody', default=None, metavar='FILE')
    p.add_argument('--vocab-lyric',  default=None, metavar='FILE')
    p.add_argument('--attr-dir',     default=None, metavar='DIR')
    p.add_argument('--model-dir',    default=None, metavar='DIR')
    p.add_argument('--max-show',     type=int, default=20,
                   help='Máximo de archivos a listar (default: 20)')
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
