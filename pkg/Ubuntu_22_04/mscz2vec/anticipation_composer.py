#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  anticipation_composer.py                                                ║
# ║  Anticipatory Music Transformer — herramienta de fichero unico           ║
# ╟──────────────────────────────────────────────────────────────────────────╢
# ║  Adaptacion al estilo mutopia del repo "anticipation" (Stanford CRFM,    ║
# ║  Thickstun / Hall / Donahue / Liang, arXiv:2306.08620). Convierte MIDI   ║
# ║  a la tokenizacion "arrival-time" (la que usa el modelo generativo) o    ║
# ║  a la tokenizacion "interarrival" (estilo MIDI-like), manipula esas      ║
# ║  secuencias, y genera musica muestreando de un checkpoint preentrenado   ║
# ║  de HuggingFace (p.ej. stanford-crfm/music-medium-800k).                 ║
# ║                                                                          ║
# ║  Subcomandos:                                                            ║
# ║    tokenize        MIDI -> tokens (.json)                                ║
# ║    render          tokens (.json) -> MIDI                                ║
# ║    info            estadisticas de un MIDI o de un fichero de tokens     ║
# ║    clip            recorta una ventana temporal de un fichero de tokens  ║
# ║    extract-melody  separa instrumentos (melodia) del resto (acomp.)      ║
# ║    list-models     lista los checkpoints publicados por Stanford CRFM    ║
# ║    download-model  descarga y cachea un checkpoint de HuggingFace        ║
# ║    generate        genera musica desde cero o como continuacion         ║
# ║    accompany       genera un acompanamiento condicionado a una melodia   ║
# ║                                                                          ║
# ║  Ejemplos:                                                                ║
# ║    ./anticipation_composer.py tokenize in.mid tokens.json                ║
# ║    ./anticipation_composer.py info tokens.json                           ║
# ║    ./anticipation_composer.py render tokens.json out.mid                 ║
# ║    ./anticipation_composer.py clip tokens.json clipped.json --start 10 --end 30 ║
# ║    ./anticipation_composer.py extract-melody tokens.json --instruments 0 \ ║
# ║        --melody-out mel.json --accomp-out acc.json                       ║
# ║    ./anticipation_composer.py list-models                                ║
# ║    ./anticipation_composer.py download-model --model stanford-crfm/music-small-800k ║
# ║    ./anticipation_composer.py generate --model stanford-crfm/music-small-800k \ ║
# ║        --length 20 --output generated.json --top-p .98                  ║
# ║    ./anticipation_composer.py accompany --model stanford-crfm/music-medium-800k \ ║
# ║        --melody mel.json --length 20 --output accomp.json                ║
# ║                                                                          ║
# ║  Dependencias: mido, numpy (siempre). torch + transformers +             ║
# ║  huggingface_hub solo se importan bajo demanda para generate/accompany/  ║
# ║  download-model.                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
anticipation_composer.py -- herramientas de linea de comandos para trabajar
con el formato de tokens del Anticipatory Music Transformer y para muestrear
musica de modelos entrenados con ese formato.
"""

import argparse
import json
import math
import sys
from collections import defaultdict

import mido
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# Color de terminal (ANSI), al estilo del resto de herramientas mutopia
# ──────────────────────────────────────────────────────────────────────────

class C:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'


def info_msg(msg):
    print(f'{C.CYAN}[i]{C.RESET} {msg}')


def ok_msg(msg):
    print(f'{C.GREEN}[ok]{C.RESET} {msg}')


def warn_msg(msg):
    print(f'{C.YELLOW}[!]{C.RESET} {msg}', file=sys.stderr)


def err_msg(msg):
    print(f'{C.RED}[error]{C.RESET} {msg}', file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────
# Configuracion global (de anticipation/config.py)
# ──────────────────────────────────────────────────────────────────────────

CONTEXT_SIZE = 1024                # contexto del modelo
EVENT_SIZE = 3                     # cada evento/control se codifica en 3 tokens
M = 341                            # contexto del modelo (1024 = 1 + EVENT_SIZE*M)
DELTA = 5                          # intervalo de anticipacion, en segundos

assert CONTEXT_SIZE == 1 + EVENT_SIZE * M

MAX_TIME_IN_SECONDS = 100          # excluye secuencias de entrenamiento muy largas
MAX_DURATION_IN_SECONDS = 10       # duracion maxima de una nota
TIME_RESOLUTION = 100              # resolucion temporal de 10ms = 100 bins/segundo

MAX_PITCH = 128                    # 128 alturas MIDI
MAX_INSTR = 129                    # 129 instrumentos MIDI (128 + percusion)
MAX_NOTE = MAX_PITCH * MAX_INSTR   # nota = altura x instrumento

MAX_INTERARRIVAL_IN_SECONDS = 10   # tiempo maximo entre eventos (codificacion MIDI-like)

MAX_TRACK_INSTR = 16               # excluye pistas con demasiados instrumentos

MAX_TIME = TIME_RESOLUTION * MAX_TIME_IN_SECONDS
MAX_DUR = TIME_RESOLUTION * MAX_DURATION_IN_SECONDS
MAX_INTERARRIVAL = TIME_RESOLUTION * MAX_INTERARRIVAL_IN_SECONDS


# ──────────────────────────────────────────────────────────────────────────
# Vocabulario (de anticipation/vocab.py)
# ──────────────────────────────────────────────────────────────────────────

# --- vocabulario "arrival-time" (secuencia de entrenamiento del modelo) ---

# bloque de eventos
EVENT_OFFSET = 0
TIME_OFFSET = EVENT_OFFSET
DUR_OFFSET = TIME_OFFSET + MAX_TIME
NOTE_OFFSET = DUR_OFFSET + MAX_DUR
REST = NOTE_OFFSET + MAX_NOTE

# bloque de controles (anticipacion)
CONTROL_OFFSET = NOTE_OFFSET + MAX_NOTE + 1
ATIME_OFFSET = CONTROL_OFFSET + 0
ADUR_OFFSET = ATIME_OFFSET + MAX_TIME
ANOTE_OFFSET = ADUR_OFFSET + MAX_DUR

# bloque especial
SPECIAL_OFFSET = ANOTE_OFFSET + MAX_NOTE
SEPARATOR = SPECIAL_OFFSET
AUTOREGRESS = SPECIAL_OFFSET + 1
ANTICIPATE = SPECIAL_OFFSET + 2
VOCAB_SIZE = ANTICIPATE + 1

# --- vocabulario "interarrival" (codificacion estilo MIDI-like) ---

MIDI_TIME_OFFSET = 0
MIDI_START_OFFSET = MIDI_TIME_OFFSET + MAX_INTERARRIVAL
MIDI_END_OFFSET = MIDI_START_OFFSET + MAX_NOTE
MIDI_SEPARATOR = MIDI_END_OFFSET + MAX_NOTE
MIDI_VOCAB_SIZE = MIDI_SEPARATOR + 1


# ──────────────────────────────────────────────────────────────────────────
# ops -- algebra de secuencias de tokens "arrival-time" (de anticipation/ops.py)
# ──────────────────────────────────────────────────────────────────────────

def op_print_tokens(tokens):
    print('---------------------')
    for j, (tm, dur, note) in enumerate(zip(tokens[0::3], tokens[1::3], tokens[2::3])):
        if note == SEPARATOR:
            print(j, 'SEPARATOR')
            continue
        if note == REST:
            print(j, tm - TIME_OFFSET, 'REST')
            continue
        if note < CONTROL_OFFSET:
            tm2, dur2, note2 = tm - TIME_OFFSET, dur - DUR_OFFSET, note - NOTE_OFFSET
            instr, pitch = note2 // 2**7, note2 - (2**7) * (note2 // 2**7)
            print(j, tm2, dur2, instr, pitch)
        else:
            tm2, dur2, note2 = tm - ATIME_OFFSET, dur - ADUR_OFFSET, note - ANOTE_OFFSET
            instr, pitch = note2 // 2**7, note2 - (2**7) * (note2 // 2**7)
            print(j, tm2, dur2, instr, pitch, '(A)')


def op_clip(tokens, start, end, clip_duration=True, seconds=True):
    if seconds:
        start = int(TIME_RESOLUTION * start)
        end = int(TIME_RESOLUTION * end)

    new_tokens = []
    for (time, dur, note) in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note < CONTROL_OFFSET:
            this_time, this_dur = time - TIME_OFFSET, dur - DUR_OFFSET
        else:
            this_time, this_dur = time - ATIME_OFFSET, dur - ADUR_OFFSET

        if this_time < start or end < this_time:
            continue

        if clip_duration and end < this_time + this_dur:
            dur -= this_time + this_dur - end

        new_tokens.extend([time, dur, note])

    return new_tokens


def op_mask(tokens, start, end):
    new_tokens = []
    for (time, dur, note) in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        this_time = (time - TIME_OFFSET if note < CONTROL_OFFSET else time - ATIME_OFFSET)
        this_time /= float(TIME_RESOLUTION)
        if start < this_time < end:
            continue
        new_tokens.extend([time, dur, note])
    return new_tokens


def op_sort(tokens):
    """ordena una secuencia de eventos o de controles (pero no una mezcla)"""
    times = tokens[0::3]
    indices = sorted(range(len(times)), key=times.__getitem__)
    sorted_tokens = []
    for idx in indices:
        sorted_tokens.extend(tokens[3 * idx:3 * (idx + 1)])
    return sorted_tokens


def op_split(tokens):
    """separa una secuencia intercalada en eventos y controles"""
    events, controls = [], []
    for (time, dur, note) in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note < CONTROL_OFFSET:
            events.extend([time, dur, note])
        else:
            controls.extend([time, dur, note])
    return events, controls


def op_pad(tokens, end_time=None, density=TIME_RESOLUTION):
    end_time = TIME_OFFSET + (end_time if end_time else op_max_time(tokens, seconds=False))
    new_tokens = []
    previous_time = TIME_OFFSET + 0
    for (time, dur, note) in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        assert note < CONTROL_OFFSET
        while time > previous_time + density:
            new_tokens.extend([previous_time + density, DUR_OFFSET + 0, REST])
            previous_time += density
        new_tokens.extend([time, dur, note])
        previous_time = time
    while end_time > previous_time + density:
        new_tokens.extend([previous_time + density, DUR_OFFSET + 0, REST])
        previous_time += density
    return new_tokens


def op_unpad(tokens):
    new_tokens = []
    for (time, dur, note) in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note == REST:
            continue
        new_tokens.extend([time, dur, note])
    return new_tokens


def op_anticipate(events, controls, delta=DELTA * TIME_RESOLUTION):
    """intercala una secuencia de eventos con controles anticipados"""
    if len(controls) == 0:
        return events, controls

    tokens = []
    event_time = 0
    control_time = controls[0] - ATIME_OFFSET
    for time, dur, note in zip(events[0::3], events[1::3], events[2::3]):
        while event_time >= control_time - delta:
            tokens.extend(controls[0:3])
            controls = controls[3:]
            control_time = controls[0] - ATIME_OFFSET if len(controls) > 0 else float('inf')
        assert note < CONTROL_OFFSET
        event_time = time - TIME_OFFSET
        tokens.extend([time, dur, note])

    return tokens, controls


def op_sparsity(tokens):
    max_dt = 0
    previous_time = TIME_OFFSET + 0
    for (time, dur, note) in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note == SEPARATOR:
            continue
        max_dt = max(max_dt, time - previous_time)
        previous_time = time
    return max_dt


def op_min_time(tokens, seconds=True, instr=None):
    mt = None
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note == SEPARATOR:
            break
        if note < CONTROL_OFFSET:
            time, note = time - TIME_OFFSET, note - NOTE_OFFSET
        else:
            time, note = time - ATIME_OFFSET, note - ANOTE_OFFSET
        if instr is not None and instr != note // 2**7:
            continue
        mt = time if mt is None else min(mt, time)
    if mt is None:
        mt = 0
    return mt / float(TIME_RESOLUTION) if seconds else mt


def op_max_time(tokens, seconds=True, instr=None):
    mt = 0
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note == SEPARATOR:
            continue
        if note < CONTROL_OFFSET:
            time, note = time - TIME_OFFSET, note - NOTE_OFFSET
        else:
            time, note = time - ATIME_OFFSET, note - ANOTE_OFFSET
        if instr is not None and instr != note // 2**7:
            continue
        mt = max(mt, time)
    return mt / float(TIME_RESOLUTION) if seconds else mt


def op_get_instruments(tokens):
    instruments = defaultdict(int)
    for time, dur, note in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note >= SPECIAL_OFFSET:
            continue
        note = note - NOTE_OFFSET if note < CONTROL_OFFSET else note - ANOTE_OFFSET
        instruments[note // 2**7] += 1
    return instruments


def op_translate(tokens, dt, seconds=False):
    if seconds:
        dt = int(TIME_RESOLUTION * dt)
    new_tokens = []
    for (time, dur, note) in zip(tokens[0::3], tokens[1::3], tokens[2::3]):
        if note == SEPARATOR:
            new_tokens.extend([time, dur, note])
            dt = 0
            continue
        this_time = time - TIME_OFFSET if note < CONTROL_OFFSET else time - ATIME_OFFSET
        assert 0 <= this_time + dt
        new_tokens.extend([time + dt, dur, note])
    return new_tokens


def op_combine(events, controls):
    return op_sort(events + [token - CONTROL_OFFSET for token in controls])


def op_extract_instruments(all_events, instruments):
    """separa los eventos de `instruments` (controles) del resto (eventos)"""
    events, controls = [], []
    for time, dur, note in zip(all_events[0::3], all_events[1::3], all_events[2::3]):
        instr = (note - NOTE_OFFSET) // 2**7
        if instr in instruments:
            controls.extend([CONTROL_OFFSET + time, CONTROL_OFFSET + dur, CONTROL_OFFSET + note])
        else:
            events.extend([time, dur, note])
    return events, controls


# ──────────────────────────────────────────────────────────────────────────
# convert -- MIDI <-> tokens (de anticipation/convert.py)
# ──────────────────────────────────────────────────────────────────────────

_IGNORED_MIDI_TYPES = {
    'aftertouch', 'polytouch', 'pitchwheel', 'sequencer_specific', 'control_change',
    'track_name', 'text', 'end_of_track', 'lyrics', 'key_signature', 'copyright',
    'marker', 'instrument_name', 'cue_marker', 'device_name', 'sequence_number',
    'channel_prefix', 'midi_port', 'smpte_offset', 'sysex', 'time_signature',
}


def midi_to_compound(midifile, debug=False):
    midi = mido.MidiFile(midifile) if isinstance(midifile, str) else midifile

    tokens = []
    note_idx = 0
    open_notes = defaultdict(list)
    time = 0
    instruments = defaultdict(int)

    for message in midi:
        time += message.time
        if message.time < 0:
            raise ValueError('negative delta time in MIDI stream')

        if message.type == 'program_change':
            instruments[message.channel] = message.program
        elif message.type in ('note_on', 'note_off'):
            instr = 128 if message.channel == 9 else instruments[message.channel]
            if message.type == 'note_on' and message.velocity > 0:
                time_in_ticks = round(TIME_RESOLUTION * time)
                tokens.extend([time_in_ticks, -1, message.note, instr, message.velocity])
                open_notes[(instr, message.note, message.channel)].append((note_idx, time))
                note_idx += 1
            else:
                try:
                    open_idx, onset_time = open_notes[(instr, message.note, message.channel)].pop(0)
                except IndexError:
                    if debug:
                        warn_msg('ignorando note_off sin note_on correspondiente')
                else:
                    tokens[5 * open_idx + 1] = round(TIME_RESOLUTION * (time - onset_time))
        elif message.type == 'set_tempo':
            pass
        elif message.type in _IGNORED_MIDI_TYPES:
            pass
        elif debug:
            warn_msg(f'mensaje MIDI no gestionado: {message.type} {message}')

    unclosed = sum(len(v) for v in open_notes.values())
    if debug and unclosed > 0:
        warn_msg(f'{unclosed} notas sin cerrar')

    return tokens


def compound_to_midi(tokens, debug=False):
    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION // 2

    it = iter(tokens)
    time_index = defaultdict(list)
    for (time_in_ticks, duration, note, instrument, velocity) in zip(it, it, it, it, it):
        time_index[(time_in_ticks, 0)].append((note, instrument, velocity))
        time_index[(time_in_ticks + duration, 1)].append((note, instrument, velocity))

    track_idx = {}
    num_tracks = 0
    for time_in_ticks, event_type in sorted(time_index.keys()):
        for (note, instrument, velocity) in time_index[(time_in_ticks, event_type)]:
            if event_type == 0:
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    idx = num_tracks
                    previous_time = 0
                    track = mido.MidiTrack()
                    mid.tracks.append(track)
                    if instrument == 128:
                        idx = 9
                        track.append(mido.Message('program_change', channel=idx, program=0))
                    else:
                        track.append(mido.Message('program_change', channel=idx, program=instrument))
                    num_tracks += 1
                    if num_tracks == 9:
                        num_tracks += 1  # se salta la pista de percusion
                track.append(mido.Message('note_on', note=note, channel=idx, velocity=velocity,
                                           time=time_in_ticks - previous_time))
                track_idx[instrument] = (track, time_in_ticks, idx)
            else:
                try:
                    track, previous_time, idx = track_idx[instrument]
                except KeyError:
                    if debug:
                        warn_msg('ignorando note_off sin onset correspondiente')
                    continue
                track.append(mido.Message('note_off', note=note, channel=idx,
                                           time=time_in_ticks - previous_time))
                track_idx[instrument] = (track, time_in_ticks, idx)

    return mid


def compound_to_events(tokens, stats=False):
    assert len(tokens) % 5 == 0
    tokens = tokens.copy()

    del tokens[4::5]  # velocidades

    tokens[2::4] = [SEPARATOR if note == -1 else MAX_PITCH * instr + note
                    for note, instr in zip(tokens[2::4], tokens[3::4])]
    tokens[2::4] = [NOTE_OFFSET + tok for tok in tokens[2::4]]
    del tokens[3::4]

    truncations = sum(1 for tok in tokens[1::3] if tok >= MAX_DUR)
    tokens[1::3] = [TIME_RESOLUTION // 4 if tok == -1 else min(tok, MAX_DUR - 1) for tok in tokens[1::3]]
    tokens[1::3] = [DUR_OFFSET + tok for tok in tokens[1::3]]

    tokens[0::3] = [TIME_OFFSET + tok for tok in tokens[0::3]]

    return (tokens, truncations) if stats else tokens


def events_to_compound(tokens, debug=False):
    tokens = op_unpad(tokens)
    tokens = [tok - CONTROL_OFFSET if tok >= CONTROL_OFFSET and tok != SEPARATOR else tok for tok in tokens]

    tokens[0::3] = [tok - TIME_OFFSET if tok != SEPARATOR else tok for tok in tokens[0::3]]
    tokens[1::3] = [tok - DUR_OFFSET if tok != SEPARATOR else tok for tok in tokens[1::3]]
    tokens[2::3] = [tok - NOTE_OFFSET if tok != SEPARATOR else tok for tok in tokens[2::3]]

    offset, track_max = 0, 0
    for j, (time, dur, note) in enumerate(zip(tokens[0::3], tokens[1::3], tokens[2::3])):
        if note == SEPARATOR:
            offset += track_max
            track_max = 0
        else:
            track_max = max(track_max, time + dur)
            tokens[3 * j] += offset

    tokens = [tok for tok in tokens if tok != SEPARATOR]

    out = 5 * (len(tokens) // 3) * [0]
    out[0::5] = tokens[0::3]
    out[1::5] = tokens[1::3]
    out[2::5] = [tok - (2**7) * (tok // 2**7) for tok in tokens[2::3]]
    out[3::5] = [tok // 2**7 for tok in tokens[2::3]]
    out[4::5] = (len(tokens) // 3) * [72]  # velocidad por defecto

    return out


def events_to_midi(tokens, debug=False):
    return compound_to_midi(events_to_compound(tokens, debug=debug), debug=debug)


def midi_to_events(midifile, debug=False):
    return compound_to_events(midi_to_compound(midifile, debug=debug))


def midi_to_interarrival(midifile, debug=False, stats=False):
    midi = mido.MidiFile(midifile) if isinstance(midifile, str) else midifile

    tokens = []
    dt = 0
    instruments = defaultdict(int)
    truncations = 0

    for message in midi:
        dt += message.time
        if message.time < 0:
            raise ValueError('negative delta time in MIDI stream')

        if message.type == 'program_change':
            instruments[message.channel] = message.program
        elif message.type in ('note_on', 'note_off'):
            delta_ticks = min(round(TIME_RESOLUTION * dt), MAX_INTERARRIVAL - 1)
            if delta_ticks != round(TIME_RESOLUTION * dt):
                truncations += 1
            if delta_ticks > 0:
                tokens.append(MIDI_TIME_OFFSET + delta_ticks)

            inst = 128 if message.channel == 9 else instruments[message.channel]
            offset = MIDI_START_OFFSET if message.type == 'note_on' and message.velocity > 0 else MIDI_END_OFFSET
            tokens.append(offset + (2**7) * inst + message.note)
            dt = 0
        elif message.type == 'set_tempo':
            pass
        elif message.type in _IGNORED_MIDI_TYPES:
            pass
        elif debug:
            warn_msg(f'mensaje MIDI no gestionado: {message.type} {message}')

    return (tokens, truncations) if stats else tokens


def interarrival_to_midi(tokens, debug=False):
    mid = mido.MidiFile()
    mid.ticks_per_beat = TIME_RESOLUTION // 2

    track_idx = {}
    time_in_ticks = 0
    num_tracks = 0
    for token in tokens:
        if token == MIDI_SEPARATOR:
            continue
        if token < MIDI_START_OFFSET:
            time_in_ticks += token - MIDI_TIME_OFFSET
        elif token < MIDI_END_OFFSET:
            token -= MIDI_START_OFFSET
            instrument, pitch = token // 2**7, token - (2**7) * (token // 2**7)
            try:
                track, previous_time, idx = track_idx[instrument]
            except KeyError:
                idx = num_tracks
                previous_time = 0
                track = mido.MidiTrack()
                mid.tracks.append(track)
                if instrument == 128:
                    idx = 9
                    track.append(mido.Message('program_change', channel=idx, program=0))
                else:
                    track.append(mido.Message('program_change', channel=idx, program=instrument))
                num_tracks += 1
                if num_tracks == 9:
                    num_tracks += 1
            track.append(mido.Message('note_on', note=pitch, channel=idx, velocity=96,
                                       time=time_in_ticks - previous_time))
            track_idx[instrument] = (track, time_in_ticks, idx)
        else:
            token -= MIDI_END_OFFSET
            instrument, pitch = token // 2**7, token - (2**7) * (token // 2**7)
            try:
                track, previous_time, idx = track_idx[instrument]
            except KeyError:
                if debug:
                    warn_msg('ignorando offset sin onset correspondiente')
                continue
            track.append(mido.Message('note_off', note=pitch, channel=idx,
                                       time=time_in_ticks - previous_time))
            track_idx[instrument] = (track, time_in_ticks, idx)

    return mid


# ──────────────────────────────────────────────────────────────────────────
# I/O de tokens: formato JSON autodescriptivo {"format": ..., "tokens": [...]}
# ──────────────────────────────────────────────────────────────────────────

def save_tokens(path, tokens, fmt):
    with open(path, 'w') as f:
        json.dump({'format': fmt, 'tokens': tokens}, f)


def load_tokens(path):
    with open(path) as f:
        data = json.load(f)
    return data['tokens'], data['format']


def require_format(fmt, expected, cmd):
    if fmt != expected:
        err_msg(f"'{cmd}' espera tokens en formato '{expected}', pero el fichero es '{fmt}'")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────
# catalogo de checkpoints publicados por Stanford CRFM
# (https://huggingface.co/stanford-crfm, ver tabla 1 del paper arXiv:2306.08620)
# ──────────────────────────────────────────────────────────────────────────

KNOWN_MODELS = [
    # (nombre HF, tokenizacion, pasos, nota)
    ('stanford-crfm/music-small-ar-inter-100k', 'interarrival', '100k', 'autoregresivo, small'),
    ('stanford-crfm/music-small-ar-100k',       'arrival',      '100k', 'autoregresivo, small'),
    ('stanford-crfm/music-small-100k',          'arrival',      '100k', 'anticipatorio, small'),
    ('stanford-crfm/music-small-ar-800k',       'arrival',      '800k', 'autoregresivo, small'),
    ('stanford-crfm/music-small-800k',          'arrival',      '800k', 'anticipatorio, small -- rapido, buena calidad'),
    ('stanford-crfm/music-medium-100k',         'arrival',      '100k', 'anticipatorio, medium'),
    ('stanford-crfm/music-medium-200k',         'arrival',      '200k', 'anticipatorio, medium'),
    ('stanford-crfm/music-medium-800k',         'arrival',      '800k', 'anticipatorio, medium -- mejor calidad, el usado en el paper'),
    ('stanford-crfm/music-large-100k',          'arrival',      '100k', 'anticipatorio, large'),
    ('stanford-crfm/music-large-800k',          'arrival',      '800k', 'anticipatorio, large -- el mejor, mas lento, mas datos de entrenamiento'),
]


def cmd_list_models(args):
    print(f'{C.BOLD}Checkpoints publicados por Stanford CRFM{C.RESET}  (usar con --model en generate/accompany/download-model)')
    print()
    for name, fmt, steps, note in KNOWN_MODELS:
        print(f'  {C.CYAN}{name:<42}{C.RESET} [{fmt:<12}] {steps:>4} pasos  -- {note}')
    print()
    print('nota: todos salvo el primero usan la tokenizacion "arrival" (la que produce este script por defecto).')


def cmd_download_model(args):
    from huggingface_hub import snapshot_download

    info_msg(f'descargando {args.model}...')
    path = snapshot_download(repo_id=args.model, revision=args.revision)
    ok_msg(f'{args.model} disponible en cache local: {path}')



def _load_model(model_name, device):
    import torch
    from transformers import AutoModelForCausalLM

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    info_msg(f'cargando modelo {model_name} en {device}...')
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    ok_msg('modelo cargado')
    return model, device


def _safe_logits(torch, logits, idx):
    logits[CONTROL_OFFSET:SPECIAL_OFFSET] = -float('inf')
    logits[SPECIAL_OFFSET:] = -float('inf')
    if idx % 3 == 0:
        logits[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET + MAX_NOTE] = -float('inf')
    elif idx % 3 == 1:
        logits[TIME_OFFSET:TIME_OFFSET + MAX_TIME] = -float('inf')
        logits[NOTE_OFFSET:NOTE_OFFSET + MAX_NOTE] = -float('inf')
    else:
        logits[TIME_OFFSET:TIME_OFFSET + MAX_TIME] = -float('inf')
        logits[DUR_OFFSET:DUR_OFFSET + MAX_DUR] = -float('inf')
    return logits


def _nucleus(torch, F, logits, top_p):
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('inf')
    return logits


def _future_logits(logits, curtime):
    if curtime > 0:
        logits[TIME_OFFSET:TIME_OFFSET + curtime] = -float('inf')
    return logits


def _instr_logits(logits, full_history):
    instrs = op_get_instruments(full_history)
    if len(instrs) < 15:
        return logits
    for instr in range(MAX_INSTR):
        if instr not in instrs:
            logits[NOTE_OFFSET + instr * MAX_PITCH:NOTE_OFFSET + (instr + 1) * MAX_PITCH] = -float('inf')
    return logits


def _add_token(model, z, tokens, top_p, current_time):
    import torch
    import torch.nn.functional as F

    assert len(tokens) % 3 == 0
    history = tokens.copy()
    lookback = max(len(tokens) - 1017, 0)
    history = history[lookback:]
    offset = op_min_time(history, seconds=False)
    history[::3] = [tok - offset for tok in history[::3]]

    new_token = []
    with torch.no_grad():
        for i in range(3):
            input_tokens = torch.tensor(z + history + new_token).unsqueeze(0).to(model.device)
            logits = model(input_tokens).logits[0, -1]
            idx = input_tokens.shape[1] - 1
            logits = _safe_logits(torch, logits, idx)
            if i == 0:
                logits = _future_logits(logits, current_time - offset)
            elif i == 2:
                logits = _instr_logits(logits, tokens)
            logits = _nucleus(torch, F, logits, top_p)
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, 1)
            new_token.append(int(token))

    new_token[0] += offset
    return new_token


def sample_generate(model, start_time, end_time, inputs=None, controls=None, top_p=1.0, delta=DELTA * TIME_RESOLUTION):
    """modo AAR (autoregresivo + anticipacion): genera eventos condicionados a `controls`"""
    inputs = inputs or []
    controls = controls or []

    start_time = int(TIME_RESOLUTION * start_time)
    end_time = int(TIME_RESOLUTION * end_time)

    prompt = op_pad(op_clip(inputs, 0, start_time, clip_duration=False, seconds=False), start_time)
    future = op_clip(inputs, start_time + 1, op_max_time(inputs, seconds=False), clip_duration=False, seconds=False)
    controls = op_clip(controls, DELTA, op_max_time(controls, seconds=False), clip_duration=False, seconds=False)

    z = [ANTICIPATE] if len(controls) > 0 or len(future) > 0 else [AUTOREGRESS]
    tokens, controls = op_anticipate(prompt, op_sort(controls + [CONTROL_OFFSET + tok for tok in future]))
    current_time = op_max_time(prompt, seconds=False)

    if controls:
        atime, adur, anote = controls[0:3]
        anticipated_tokens = controls[3:]
        anticipated_time = atime - ATIME_OFFSET
    else:
        anticipated_time = math.inf

    n_steps = end_time - start_time
    report_every = max(n_steps // 20, 1)
    while True:
        while current_time >= anticipated_time - delta:
            tokens.extend([atime, adur, anote])
            if len(anticipated_tokens) > 0:
                atime, adur, anote = anticipated_tokens[0:3]
                anticipated_tokens = anticipated_tokens[3:]
                anticipated_time = atime - ATIME_OFFSET
            else:
                anticipated_time = math.inf

        new_token = _add_token(model, z, tokens, top_p, max(start_time, current_time))
        new_time = new_token[0] - TIME_OFFSET
        if new_time >= end_time:
            break

        tokens.extend(new_token)
        dt = new_time - current_time
        current_time = new_time
        if (current_time // TIME_RESOLUTION) % report_every == 0:
            info_msg(f'  {current_time / TIME_RESOLUTION:.1f}s / {end_time / TIME_RESOLUTION:.1f}s')

    events, _ = op_split(tokens)
    return op_sort(op_unpad(events) + future)


def sample_generate_ar(model, start_time, end_time, inputs=None, controls=None, top_p=1.0):
    """modo autoregresivo puro (sin anticipacion explicita)"""
    inputs = inputs or []
    controls = [tok - CONTROL_OFFSET for tok in (controls or [])]

    start_time = int(TIME_RESOLUTION * start_time)
    end_time = int(TIME_RESOLUTION * end_time)

    inputs = op_sort(inputs + controls)
    prompt = op_pad(op_clip(inputs, 0, start_time, clip_duration=False, seconds=False), start_time)
    controls = op_clip(inputs, start_time + 1, op_max_time(inputs, seconds=False), clip_duration=False, seconds=False)

    z = [AUTOREGRESS]
    current_time = op_max_time(prompt, seconds=False)
    tokens = prompt

    if controls:
        atime, adur, anote = controls[0:3]
        anticipated_tokens = controls[3:]
        anticipated_time = atime - TIME_OFFSET
    else:
        anticipated_time = math.inf

    while True:
        new_token = _add_token(model, z, tokens, top_p, max(start_time, current_time))
        new_time = new_token[0] - TIME_OFFSET
        if new_time >= end_time:
            break

        dt = new_time - current_time
        current_time = new_time

        while current_time >= anticipated_time:
            tokens.extend([atime, adur, anote])
            if len(anticipated_tokens) > 0:
                atime, adur, anote = anticipated_tokens[0:3]
                anticipated_tokens = anticipated_tokens[3:]
                anticipated_time = atime - TIME_OFFSET
            else:
                anticipated_time = math.inf

        tokens.extend(new_token)

    if anticipated_time != math.inf:
        tokens.extend([atime, adur, anote])

    return op_sort(op_unpad(tokens) + controls)


# ──────────────────────────────────────────────────────────────────────────
# subcomandos
# ──────────────────────────────────────────────────────────────────────────

def cmd_tokenize(args):
    info_msg(f'leyendo {args.input}...')
    if args.format == 'arrival':
        tokens = midi_to_events(args.input, debug=args.debug)
    else:
        tokens = midi_to_interarrival(args.input, debug=args.debug)

    save_tokens(args.output, tokens, args.format)
    n_events = len(tokens) // 3 if args.format == 'arrival' else None
    if n_events is not None:
        ok_msg(f'{n_events} eventos escritos en {args.output} (formato {args.format})')
    else:
        ok_msg(f'{len(tokens)} tokens escritos en {args.output} (formato {args.format})')


def cmd_render(args):
    tokens, fmt = load_tokens(args.input)
    info_msg(f'tokens en formato {fmt}, generando MIDI...')
    if fmt == 'arrival':
        mid = events_to_midi(tokens, debug=args.debug)
    else:
        mid = interarrival_to_midi(tokens, debug=args.debug)
    mid.save(args.output)
    ok_msg(f'MIDI escrito en {args.output}')


def cmd_info(args):
    if args.input.endswith('.mid') or args.input.endswith('.midi'):
        tokens = midi_to_events(args.input)
        fmt = 'arrival'
        source = 'MIDI'
    else:
        tokens, fmt = load_tokens(args.input)
        source = 'tokens'

    print(f'{C.BOLD}{args.input}{C.RESET}  ({source}, formato {fmt})')

    if fmt == 'arrival':
        n_events = len(tokens) // 3
        duration = op_max_time(tokens)
        instruments = op_get_instruments(tokens)
        sparsity = op_sparsity(tokens) / TIME_RESOLUTION
        print(f'  eventos       : {n_events}')
        print(f'  duracion      : {duration:.2f}s')
        print(f'  instrumentos  : {len(instruments)}')
        for instr, count in sorted(instruments.items()):
            label = 'percusion' if instr == 128 else f'programa {instr}'
            print(f'    - {label:<14} {count} notas')
        print(f'  intervalo max : {sparsity:.2f}s entre eventos')
    else:
        n_notes = sum(1 for tok in tokens if MIDI_START_OFFSET <= tok < MIDI_END_OFFSET)
        n_time = sum(1 for tok in tokens if tok < MIDI_START_OFFSET)
        print(f'  tokens totales : {len(tokens)}')
        print(f'  note-on        : {n_notes}')
        print(f'  saltos de tiempo: {n_time}')


def cmd_clip(args):
    if args.start < 0:
        err_msg('--start no puede ser negativo')
        sys.exit(1)
    if args.start >= args.end:
        err_msg('--start debe ser menor que --end')
        sys.exit(1)

    tokens, fmt = load_tokens(args.input)
    require_format(fmt, 'arrival', 'clip')
    clipped = op_clip(tokens, args.start, args.end, clip_duration=not args.no_clip_duration)
    clipped = op_translate(clipped, -args.start, seconds=True)
    save_tokens(args.output, clipped, fmt)
    ok_msg(f'{len(clipped) // 3} eventos entre {args.start}s y {args.end}s escritos en {args.output}')


def cmd_extract_melody(args):
    tokens, fmt = load_tokens(args.input)
    require_format(fmt, 'arrival', 'extract-melody')
    try:
        instruments = set(int(i) for i in args.instruments.split(','))
    except ValueError:
        err_msg(f"--instruments debe ser una lista de enteros separados por coma, p.ej. '0,1' (recibido: '{args.instruments}')")
        sys.exit(1)

    events, controls = op_extract_instruments(tokens, instruments)
    melody = [tok - CONTROL_OFFSET for tok in controls]  # de vuelta a offsets de evento normales

    save_tokens(args.melody_out, melody, fmt)
    save_tokens(args.accomp_out, events, fmt)
    ok_msg(f'melodia ({len(melody) // 3} eventos, instrumentos {sorted(instruments)}) -> {args.melody_out}')
    ok_msg(f'resto ({len(events) // 3} eventos) -> {args.accomp_out}')


def cmd_generate(args):
    model, device = _load_model(args.model, args.device)

    inputs, controls = [], []
    if args.input:
        inputs, fmt = load_tokens(args.input)
        require_format(fmt, 'arrival', 'generate --input')

    start_time = args.start
    end_time = args.start + args.length

    info_msg(f'generando de {start_time}s a {end_time}s (top_p={args.top_p}, modo={args.mode})...')
    if args.mode == 'aar':
        tokens = sample_generate(model, start_time, end_time, inputs=inputs, top_p=args.top_p)
    else:
        tokens = sample_generate_ar(model, start_time, end_time, inputs=inputs, top_p=args.top_p)

    save_tokens(args.output, tokens, 'arrival')
    ok_msg(f'{len(tokens) // 3} eventos generados -> {args.output}')


def cmd_accompany(args):
    model, device = _load_model(args.model, args.device)

    melody, fmt = load_tokens(args.melody)
    require_format(fmt, 'arrival', 'accompany')
    controls = [CONTROL_OFFSET + tok for tok in melody]

    start_time = args.start
    end_time = args.start + args.length

    info_msg(f'generando acompanamiento de {start_time}s a {end_time}s (top_p={args.top_p})...')
    accompaniment = sample_generate(model, start_time, end_time, controls=controls, top_p=args.top_p)

    combined = op_combine(accompaniment, controls) if args.combine else accompaniment
    save_tokens(args.output, combined, 'arrival')
    label = 'acompanamiento + melodia combinados' if args.combine else 'acompanamiento'
    ok_msg(f'{len(combined) // 3} eventos ({label}) -> {args.output}')


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog='anticipation_composer.py',
        description='Anticipatory Music Transformer -- tokenizacion, manipulacion y generacion de musica MIDI.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('tokenize', help='MIDI -> tokens (.json)')
    p.add_argument('input', help='fichero MIDI de entrada')
    p.add_argument('output', help='fichero de tokens (.json) de salida')
    p.add_argument('--format', choices=['arrival', 'interarrival'], default='arrival')
    p.add_argument('--debug', action='store_true')
    p.set_defaults(func=cmd_tokenize)

    p = sub.add_parser('render', help='tokens (.json) -> MIDI')
    p.add_argument('input', help='fichero de tokens (.json) de entrada')
    p.add_argument('output', help='fichero MIDI de salida')
    p.add_argument('--debug', action='store_true')
    p.set_defaults(func=cmd_render)

    p = sub.add_parser('info', help='estadisticas de un MIDI o de un fichero de tokens')
    p.add_argument('input', help='fichero .mid o .json de tokens')
    p.set_defaults(func=cmd_info)

    p = sub.add_parser('clip', help='recorta una ventana temporal de un fichero de tokens (arrival-time)')
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--start', type=float, required=True, help='inicio en segundos')
    p.add_argument('--end', type=float, required=True, help='fin en segundos')
    p.add_argument('--no-clip-duration', action='store_true',
                    help='no truncar la duracion de notas que cruzan el limite')
    p.set_defaults(func=cmd_clip)

    p = sub.add_parser('extract-melody', help='separa instrumentos (melodia) del resto (arrival-time)')
    p.add_argument('input')
    p.add_argument('--instruments', required=True,
                    help='lista de programas MIDI separados por coma, p.ej. "0,1" (128 = percusion)')
    p.add_argument('--melody-out', required=True)
    p.add_argument('--accomp-out', required=True)
    p.set_defaults(func=cmd_extract_melody)

    p = sub.add_parser('list-models', help='lista los checkpoints publicados por Stanford CRFM')
    p.set_defaults(func=cmd_list_models)

    p = sub.add_parser('download-model', help='descarga y cachea localmente un checkpoint de HuggingFace')
    p.add_argument('--model', required=True, help='nombre del checkpoint, p.ej. stanford-crfm/music-medium-800k')
    p.add_argument('--revision', default=None, help='revision/commit concreto (opcional)')
    p.set_defaults(func=cmd_download_model)

    p = sub.add_parser('generate', help='genera musica desde cero o como continuacion de un input')
    p.add_argument('--model', required=True, help='nombre o ruta de un checkpoint HuggingFace')
    p.add_argument('--input', help='tokens (arrival-time) a continuar; opcional')
    p.add_argument('--start', type=float, default=0.0, help='instante en el que empieza la generacion')
    p.add_argument('--length', type=float, default=20.0, help='segundos a generar')
    p.add_argument('--top-p', type=float, default=1.0, dest='top_p')
    p.add_argument('--mode', choices=['aar', 'ar'], default='aar',
                    help='aar = anticipatorio (usa --input como controles futuros), ar = autoregresivo puro')
    p.add_argument('--device', default=None, help='cuda|cpu; por defecto detecta automaticamente')
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser('accompany', help='genera un acompanamiento condicionado a una melodia (arrival-time)')
    p.add_argument('--model', required=True)
    p.add_argument('--melody', required=True, help='tokens de la melodia (usar extract-melody antes)')
    p.add_argument('--start', type=float, default=0.0)
    p.add_argument('--length', type=float, default=20.0)
    p.add_argument('--top-p', type=float, default=1.0, dest='top_p')
    p.add_argument('--combine', action='store_true', help='incluir la melodia en la salida, ya combinada')
    p.add_argument('--device', default=None)
    p.add_argument('--output', required=True)
    p.set_defaults(func=cmd_accompany)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except FileNotFoundError as e:
        if e.filename:
            err_msg(f'fichero no encontrado: {e.filename}')
        else:
            # algunas excepciones de terceros (p.ej. huggingface_hub) heredan de
            # FileNotFoundError sin ser realmente "fichero local no encontrado";
            # su propio mensaje es mas util que nuestra plantilla generica
            err_msg(str(e))
        sys.exit(1)
    except Exception as e:
        err_msg(str(e))
        if args_debug_enabled(args):
            raise
        sys.exit(1)


def args_debug_enabled(args):
    return bool(getattr(args, 'debug', False))


if __name__ == '__main__':
    main()
