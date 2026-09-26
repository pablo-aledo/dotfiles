#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════╗
║ miditok_algoritmos.py                                                     ║
║                                                                            ║
║ Los 9 algoritmos de tokenización de MIDI (MIDI-Like, TSD, REMI,           ║
║ Structured, CPWord, Octuple, MuMIDI, MMM, PerTok) REIMPLEMENTADOS DESDE   ║
║ CERO, en Python puro sobre `mido`, para poder leer y entender el          ║
║ algoritmo real de cada uno -- sin la caja negra de la librería miditok.   ║
║                                                                            ║
║ Este fichero es la versión "para entender cómo funciona"; para uso        ║
║ productivo (fidelidad exacta, BPE, etc.) usa miditok_composer.py, que      ║
║ envuelve la librería original.                                            ║
║                                                                            ║
║ Simplificaciones deliberadas (documentadas también en --explain):         ║
║   - se asume compás 4/4 para la rejilla de Bar/Position                   ║
║   - velocity se cuantiza a 32 bins, duración/time-shift a una rejilla      ║
║     de semicorcheas (configurable con --resolution)                       ║
║   - sin BPE/entrenamiento de vocabulario (eso ya lo cubre miditok_composer)║
║                                                                            ║
║ Uso:                                                                      ║
║   miditok_algoritmos.py list-schemes                                     ║
║   miditok_algoritmos.py explain REMI                                     ║
║   miditok_algoritmos.py tokenize song.mid -s REMI -o tokens.json         ║
║   miditok_algoritmos.py detokenize tokens.json -o song_back.mid          ║
║   miditok_algoritmos.py compare song.mid          (los 9 a la vez)       ║
║   miditok_algoritmos.py info song.mid                                    ║
║                                                                            ║
║ Deps: pip install mido                                                   ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import mido
except ImportError:
    mido = None


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"


def ok(msg): print(f"{C.GREEN}✓{C.RESET} {msg}")
def info_line(msg): print(f"{C.CYAN}i{C.RESET} {msg}")
def fail(msg):
    print(f"{C.RED}✗ error:{C.RESET} {msg}", file=sys.stderr)
    sys.exit(1)


def need_mido():
    if mido is None:
        fail("falta 'mido'. Instala con:\n    pip install mido")


# --------------------------------------------------------------------------- #
# representación común de una nota + E/S de MIDI (independiente del esquema)
# --------------------------------------------------------------------------- #

@dataclass
class Note:
    pitch: int
    velocity: int
    start: int          # ticks absolutos
    duration: int        # ticks
    track: int = 0
    program: int = 0
    is_drum: bool = False


def load_notes(path: str):
    """Lee un MIDI con mido y devuelve (lista de Note, ticks_per_beat).

    Usa una cola por (track, pitch) -- no un único valor -- para poder
    emparejar correctamente NoteOn/NoteOff cuando dos notas del mismo pitch
    se solapan (nota B empieza antes de que acabe la nota A).
    """
    need_mido()
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    notes = []
    for ti, track in enumerate(mid.tracks):
        abs_t = 0
        program = 0
        is_drum = False
        active = {}  # (pitch) -> lista de (start, vel), FIFO
        for msg in track:
            abs_t += msg.time
            if msg.type == "program_change":
                program = msg.program
            if msg.type in ("note_on", "note_off"):
                if getattr(msg, "channel", 0) == 9:
                    is_drum = True
                if msg.type == "note_on" and msg.velocity > 0:
                    active.setdefault(msg.note, []).append((abs_t, msg.velocity))
                elif active.get(msg.note):
                    st, vel = active[msg.note].pop(0)
                    notes.append(Note(msg.note, vel, st, max(1, abs_t - st), ti, program, is_drum))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, tpb


def notes_to_midi(notes, tpb: int, path: str):
    """Escribe una lista de Note a un fichero .mid con mido, una pista MIDI por track."""
    need_mido()
    mid = mido.MidiFile(ticks_per_beat=tpb)
    by_track = {}
    for n in notes:
        by_track.setdefault(n.track, []).append(n)
    if not by_track:
        by_track[0] = []
    for ti in sorted(by_track):
        track = mido.MidiTrack()
        mid.tracks.append(track)
        tn = by_track[ti]
        chan = 9 if any(n.is_drum for n in tn) else 0
        prog = next((n.program for n in tn if not n.is_drum), None)
        events = []
        for n in tn:
            events.append((n.start, 1, "on", n.pitch, n.velocity))
            events.append((n.start + n.duration, 0, "off", n.pitch, 0))
        events.sort()  # (tiempo, off-antes-que-on, ...)
        if prog:
            track.append(mido.Message("program_change", program=prog, time=0, channel=chan))
        last = 0
        for t, _, kind, pitch, vel in events:
            delta = max(0, t - last)
            if kind == "on":
                track.append(mido.Message("note_on", note=pitch, velocity=vel, time=delta, channel=chan))
            else:
                track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta, channel=chan))
            last = t
    mid.save(path)


# --------------------------------------------------------------------------- #
# cuantización compartida: rejilla de posición, bins de velocity
# --------------------------------------------------------------------------- #

POSITIONS_PER_BAR = 16   # semicorcheas; asume compás 4/4
VELOCITY_BINS = 32
MAX_DURATION_STEPS = 64  # 4 compases
MAX_TIMESHIFT_STEPS = 64
MICROTIMING_RANGE = 4    # PerTok: bins de -4 a +3 alrededor de la rejilla


def ticks_per_position(tpb: int) -> int:
    return max(1, (tpb * 4) // POSITIONS_PER_BAR)


def quantize(tick: int, tpp: int) -> int:
    return round(tick / tpp)


def vel_to_bin(v: int) -> int:
    return max(0, min(VELOCITY_BINS - 1, (v * VELOCITY_BINS) // 128))


def bin_to_vel(b: int) -> int:
    span = 128 // VELOCITY_BINS
    return max(1, min(127, b * span + span // 2))


def tok_int(prefix: str, tok: str) -> int:
    return int(tok.split(prefix + "_", 1)[1])


# --------------------------------------------------------------------------- #
# los 9 esquemas
# --------------------------------------------------------------------------- #

class BaseScheme:
    name = "Base"
    explain = "(sin descripción)"

    def tokenize(self, notes, tpb):
        raise NotImplementedError

    def detokenize(self, tokens, tpb):
        raise NotImplementedError


class MIDILikeScheme(BaseScheme):
    name = "MIDILike"
    explain = (
        "Reproduce el propio protocolo MIDI: una secuencia de eventos NoteOn/NoteOff\n"
        "independientes, con TimeShift entre eventos y Velocity como 'estado' que se\n"
        "actualiza cuando cambia. Es el esquema más simple y el que menos estructura\n"
        "impone -- por eso también es el que genera secuencias más largas e\n"
        "irregulares (nada obliga a que un NoteOn tenga su NoteOff cerca).\n"
        "Vocabulario: NoteOn_<pitch>, NoteOff_<pitch>, TimeShift_<n>, Velocity_<bin>"
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        events = []
        for n in notes:
            start_step = quantize(n.start, tpp)
            end_step = max(start_step + 1, quantize(n.start + n.duration, tpp))
            events.append((start_step, 1, "on", n.pitch, vel_to_bin(n.velocity)))
            events.append((end_step, 0, "off", n.pitch, None))
        events.sort()
        tokens, last_t, last_vel = [], 0, None
        for t, _, kind, pitch, velbin in events:
            if t > last_t:
                tokens.append(f"TimeShift_{min(t - last_t, MAX_TIMESHIFT_STEPS)}")
                last_t = t
            if kind == "on":
                if velbin != last_vel:
                    tokens.append(f"Velocity_{velbin}")
                    last_vel = velbin
                tokens.append(f"NoteOn_{pitch}")
            else:
                tokens.append(f"NoteOff_{pitch}")
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        t, velbin, active, notes = 0, VELOCITY_BINS // 2, {}, []
        for tok in tokens:
            if tok.startswith("TimeShift_"):
                t += tok_int("TimeShift", tok) * tpp
            elif tok.startswith("Velocity_"):
                velbin = tok_int("Velocity", tok)
            elif tok.startswith("NoteOn_"):
                active.setdefault(tok_int("NoteOn", tok), []).append((t, bin_to_vel(velbin)))
            elif tok.startswith("NoteOff_"):
                p = tok_int("NoteOff", tok)
                if active.get(p):
                    st, v = active[p].pop(0)
                    notes.append(Note(p, v, st, max(1, t - st)))
        return notes


class TSDScheme(BaseScheme):
    name = "TSD"
    explain = (
        "'Time-Shift Duration': como MIDI-Like pero cada nota lleva su Duration\n"
        "explícita en vez de depender de un NoteOff futuro. Ventaja sobre MIDI-Like:\n"
        "cada nota es autocontenida (Pitch+Velocity+Duration juntos), no hace falta\n"
        "'recordar' qué notas siguen sonando al decodificar.\n"
        "Vocabulario: TimeShift_<n>, Pitch_<p>, Velocity_<bin>, Duration_<pasos>"
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        tokens, last_t = [], 0
        for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
            t = quantize(n.start, tpp)
            if t > last_t:
                tokens.append(f"TimeShift_{min(t - last_t, MAX_TIMESHIFT_STEPS)}")
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens += [f"Pitch_{n.pitch}", f"Velocity_{vel_to_bin(n.velocity)}", f"Duration_{d}"]
            last_t = t
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        t, notes, i = 0, [], 0
        while i < len(tokens):
            if tokens[i].startswith("TimeShift_"):
                t += tok_int("TimeShift", tokens[i]) * tpp
                i += 1
            if i + 2 >= len(tokens):
                break
            pitch = tok_int("Pitch", tokens[i]); i += 1
            velbin = tok_int("Velocity", tokens[i]); i += 1
            dur = tok_int("Duration", tokens[i]); i += 1
            notes.append(Note(pitch, bin_to_vel(velbin), t, dur * tpp))
        return notes


class REMIScheme(BaseScheme):
    name = "REMI"
    explain = (
        "'Revamped MIDI': sustituye el TimeShift libre de TSD por marcadores\n"
        "explícitos Bar_None / Position_<i>. El modelo ya no tiene que sumar deltas\n"
        "para saber 'dónde' está métricamente -- lo lee directo del token. Es el\n"
        "esquema más usado para generación con Transformer porque hace explícita la\n"
        "estructura métrica (compás/posición) que MIDI-Like y TSD dejan implícita.\n"
        "Vocabulario: Bar_None, Position_<0..15>, Pitch_<p>, Velocity_<bin>, Duration_<pasos>"
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        tokens, cur_bar = [], -1
        for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
            step = quantize(n.start, tpp)
            bar, pos = step // POSITIONS_PER_BAR, step % POSITIONS_PER_BAR
            while cur_bar < bar:
                tokens.append("Bar_None")
                cur_bar += 1
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens += [f"Position_{pos}", f"Pitch_{n.pitch}", f"Velocity_{vel_to_bin(n.velocity)}", f"Duration_{d}"]
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        bar, notes, i = -1, [], 0
        while i < len(tokens):
            if tokens[i] == "Bar_None":
                bar += 1
                i += 1
                continue
            if tokens[i].startswith("Position_") and i + 3 < len(tokens):
                pos = tok_int("Position", tokens[i]); i += 1
                pitch = tok_int("Pitch", tokens[i]); i += 1
                velbin = tok_int("Velocity", tokens[i]); i += 1
                dur = tok_int("Duration", tokens[i]); i += 1
                t = (max(bar, 0) * POSITIONS_PER_BAR + pos) * tpp
                notes.append(Note(pitch, bin_to_vel(velbin), t, dur * tpp))
                continue
            i += 1
        return notes


class StructuredScheme(BaseScheme):
    name = "Structured"
    explain = (
        "Gramática estrictamente periódica: SIEMPRE TimeShift, Pitch, Velocity,\n"
        "Duration en ese orden, nota tras nota, sin excepciones ni tokens\n"
        "opcionales (el TimeShift inicial codifica el instante de la primera nota).\n"
        "Frente a MIDI-Like/TSD (donde la forma de la secuencia varía según la\n"
        "música), aquí la posición de cada tipo de token dentro del ciclo es\n"
        "siempre la misma -- más fácil de aprender para un modelo pequeño, a costa\n"
        "de menos flexibilidad (acordes -> varios ciclos con TimeShift_0).\n"
        "Vocabulario: TimeShift_<n>, Pitch_<p>, Velocity_<bin>, Duration_<pasos>"
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        srt = sorted(notes, key=lambda n: (n.start, n.pitch))
        tokens, last_step = [], 0
        for n in srt:
            step = quantize(n.start, tpp)
            shift = max(0, min(step - last_step, MAX_TIMESHIFT_STEPS))
            tokens.append(f"TimeShift_{shift}")
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens += [f"Pitch_{n.pitch}", f"Velocity_{vel_to_bin(n.velocity)}", f"Duration_{d}"]
            last_step = step
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        step, notes = 0, []
        for i in range(0, len(tokens) - 3, 4):
            step += tok_int("TimeShift", tokens[i])
            pitch = tok_int("Pitch", tokens[i + 1])
            velbin = tok_int("Velocity", tokens[i + 2])
            dur = tok_int("Duration", tokens[i + 3])
            notes.append(Note(pitch, bin_to_vel(velbin), step * tpp, dur * tpp))
        return notes


IGNORE = "Ignore"  # placeholder para sub-vocabularios que no aplican en CPWord


class CPWordScheme(BaseScheme):
    name = "CPWord"
    explain = (
        "'Compound Word': en vez de una cadena de tokens ESCALARES, cada paso es un\n"
        "único token COMPUESTO -- un vector con una sub-entrada por familia\n"
        "(Bar/Position, Pitch, Velocity, Duration), con 'Ignore' donde no aplica.\n"
        "El modelo predice el vector entero en paralelo en vez de token a token, lo\n"
        "que acorta la secuencia ~4-5x frente a REMI a cambio de una cabeza de\n"
        "predicción más compleja (varias sub-cabezas en paralelo).\n"
        "Cada 'token' aquí es un dict: {Family, Bar, Position, Pitch, Velocity, Duration}"
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        tokens, cur_bar = [], -1
        for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
            step = quantize(n.start, tpp)
            bar, pos = step // POSITIONS_PER_BAR, step % POSITIONS_PER_BAR
            while cur_bar < bar:
                tokens.append({"Family": "Metric", "Bar": "New", "Position": IGNORE,
                                "Pitch": IGNORE, "Velocity": IGNORE, "Duration": IGNORE})
                cur_bar += 1
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens.append({"Family": "Note", "Bar": IGNORE, "Position": pos,
                            "Pitch": n.pitch, "Velocity": vel_to_bin(n.velocity), "Duration": d})
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        bar, notes = -1, []
        for tok in tokens:
            if tok["Family"] == "Metric":
                bar += 1
                continue
            t = (max(bar, 0) * POSITIONS_PER_BAR + tok["Position"]) * tpp
            notes.append(Note(tok["Pitch"], bin_to_vel(tok["Velocity"]), t, tok["Duration"] * tpp))
        return notes


class OctupleScheme(BaseScheme):
    name = "Octuple"
    explain = (
        "Un único token compuesto POR NOTA con todos sus atributos en paralelo:\n"
        "Bar, Position, Pitch, Velocity, Duration, Program, TimeSignature, Tempo.\n"
        "A diferencia de CPWord (compuesto por PASO temporal), aquí cada nota es\n"
        "autocontenida -- no hace falta ni siquiera leer un Bar_None previo para\n"
        "saber cuándo suena. Da las secuencias más cortas de todas (una \"palabra\"\n"
        "por nota), al precio de una cabeza de predicción con 8 salidas paralelas.\n"
        "Cada 'token' aquí es un dict con los 8 campos."
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        tokens = []
        for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
            step = quantize(n.start, tpp)
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens.append({
                "Bar": step // POSITIONS_PER_BAR, "Position": step % POSITIONS_PER_BAR,
                "Pitch": n.pitch, "Velocity": vel_to_bin(n.velocity), "Duration": d,
                "Program": "drums" if n.is_drum else n.program, "TimeSig": "4/4", "Tempo": 120,
            })
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        notes = []
        for tok in tokens:
            t = (tok["Bar"] * POSITIONS_PER_BAR + tok["Position"]) * tpp
            is_drum = tok["Program"] == "drums"
            prog = 0 if is_drum else tok["Program"]
            notes.append(Note(tok["Pitch"], bin_to_vel(tok["Velocity"]), t, tok["Duration"] * tpp,
                               program=prog, is_drum=is_drum))
        return notes


class MuMIDIScheme(BaseScheme):
    name = "MuMIDI"
    explain = (
        "Pensado para multitrack: en vez de una secuencia por pista, TODAS las\n"
        "pistas se combinan en un único stream ordenado por tiempo absoluto\n"
        "('one_token_stream'). Como ya no hay Bar_None/Position_i secuenciales que\n"
        "marquen 'dónde estamos', cada token lleva pegada su propia codificación\n"
        "posicional (BarPosEnc, PositionPosEnc) -- el modelo sabe su posición\n"
        "métrica sin tener que contar tokens hacia atrás.\n"
        "Limitación real (y de esta reimplementación): BarPosEnc se acota a 0-15\n"
        "(mod 16 compases) para no disparar el vocabulario, así que distinguir\n"
        "compás 3 de compás 19 depende del orden de la secuencia, no del token en sí."
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        tokens = []
        for n in sorted(notes, key=lambda n: (n.start, n.track, n.pitch)):
            step = quantize(n.start, tpp)
            bar, pos = step // POSITIONS_PER_BAR, step % POSITIONS_PER_BAR
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens.append({
                "Program": "drums" if n.is_drum else n.program,
                "Pitch": n.pitch, "Velocity": vel_to_bin(n.velocity), "Duration": d,
                "BarPosEnc": bar % 16, "PositionPosEnc": pos,
            })
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        notes, bar, last_barenc, last_pos = [], 0, None, -1
        for tok in tokens:
            barenc, pos = tok["BarPosEnc"], tok["PositionPosEnc"]
            if last_barenc is not None and (barenc != last_barenc or pos < last_pos):
                bar += 1
            last_barenc, last_pos = barenc, pos
            t = (bar * POSITIONS_PER_BAR + pos) * tpp
            is_drum = tok["Program"] == "drums"
            prog = 0 if is_drum else tok["Program"]
            notes.append(Note(tok["Pitch"], bin_to_vel(tok["Velocity"]), t, tok["Duration"] * tpp,
                               program=prog, is_drum=is_drum))
        return notes


class MMMScheme(BaseScheme):
    name = "MMM"
    base_name = "TSD"
    explain = (
        "'Multi-Track Music Machine': cada pista se serializa como un bloque\n"
        "independiente -- Track_Start_<programa>, ... tokens de la pista con un\n"
        "esquema base (aquí TSD) ..., Track_End -- y los bloques se concatenan.\n"
        "Pensado para generación/inpainting por pista: el modelo puede aprender a\n"
        "generar 'una pista de bajo dado que ya existen estos otros bloques',\n"
        "reordenando o añadiendo bloques de pista libremente.\n"
        "Vocabulario: Track_Start_<programa|drums>, Track_End, + vocabulario TSD"
    )

    def tokenize(self, notes, tpb):
        base = SCHEMES[self.base_name]()
        tracks = sorted(set(n.track for n in notes)) or [0]
        tokens = []
        for ti in tracks:
            tn = [n for n in notes if n.track == ti]
            is_drum = any(n.is_drum for n in tn)
            prog = next((n.program for n in tn if not n.is_drum), 0)
            label = "drums" if is_drum else prog
            tokens.append(f"Track_Start_{label}")
            tokens += base.tokenize(tn, tpb)
            tokens.append("Track_End")
        return tokens

    def detokenize(self, tokens, tpb):
        base = SCHEMES[self.base_name]()
        notes, i, ti = [], 0, 0
        while i < len(tokens):
            if tokens[i].startswith("Track_Start_"):
                label = tokens[i].split("Track_Start_", 1)[1]
                i += 1
                block = []
                while i < len(tokens) and tokens[i] != "Track_End":
                    block.append(tokens[i]); i += 1
                i += 1
                is_drum = label == "drums"
                prog = 0 if is_drum else int(label)
                for n in base.detokenize(block, tpb):
                    notes.append(Note(n.pitch, n.velocity, n.start, n.duration, ti, prog, is_drum))
                ti += 1
            else:
                i += 1
        return notes


class PerTokScheme(BaseScheme):
    name = "PerTok"
    explain = (
        "'Performance Tokenizer': orientado a preservar el microtiming expresivo\n"
        "que el resto de esquemas pierde al cuantizar cada nota a la rejilla más\n"
        "cercana. Además de Position_<i> (la casilla de rejilla más cercana), cada\n"
        "nota lleva un MicroTiming_<±n> con el residuo exacto respecto a esa\n"
        "casilla -- así el modelo puede aprender el 'groove' (notas ligeramente\n"
        "antes/después del click) en vez de forzar todo a metrónomo perfecto.\n"
        "Vocabulario: Bar_None, Position_<i>, MicroTiming_<±n>, Pitch_<p>, Velocity_<bin>, Duration_<pasos>"
    )

    def tokenize(self, notes, tpb):
        tpp = ticks_per_position(tpb)
        tokens, cur_bar = [], -1
        for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
            step = round(n.start / tpp)
            residual = n.start - step * tpp
            mt = int(round(residual / tpp * MICROTIMING_RANGE * 2))
            mt = max(-MICROTIMING_RANGE, min(MICROTIMING_RANGE - 1, mt))
            bar, pos = step // POSITIONS_PER_BAR, step % POSITIONS_PER_BAR
            while cur_bar < bar:
                tokens.append("Bar_None")
                cur_bar += 1
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens += [f"Position_{pos}", f"MicroTiming_{mt:+d}", f"Pitch_{n.pitch}",
                       f"Velocity_{vel_to_bin(n.velocity)}", f"Duration_{d}"]
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        bar, notes, i = -1, [], 0
        while i < len(tokens):
            if tokens[i] == "Bar_None":
                bar += 1
                i += 1
                continue
            if tokens[i].startswith("Position_") and i + 4 < len(tokens):
                pos = tok_int("Position", tokens[i]); i += 1
                mt = tok_int("MicroTiming", tokens[i]); i += 1
                pitch = tok_int("Pitch", tokens[i]); i += 1
                velbin = tok_int("Velocity", tokens[i]); i += 1
                dur = tok_int("Duration", tokens[i]); i += 1
                step = max(bar, 0) * POSITIONS_PER_BAR + pos
                t = step * tpp + round(mt / (MICROTIMING_RANGE * 2) * tpp)
                notes.append(Note(pitch, bin_to_vel(velbin), max(0, t), dur * tpp))
                continue
            i += 1
        return notes


SCHEMES = {
    "MIDILike": MIDILikeScheme, "TSD": TSDScheme, "REMI": REMIScheme,
    "Structured": StructuredScheme, "CPWord": CPWordScheme, "Octuple": OctupleScheme,
    "MuMIDI": MuMIDIScheme, "MMM": MMMScheme, "PerTok": PerTokScheme,
}


# --------------------------------------------------------------------------- #
# subcomandos
# --------------------------------------------------------------------------- #

def cmd_list_schemes(_a):
    print(f"{C.BOLD}esquemas implementados desde cero:{C.RESET}")
    for name in SCHEMES:
        first_line = SCHEMES[name].explain.split("\n")[0]
        print(f"  {C.CYAN}{name:<12}{C.RESET} {first_line}")
    print(f"\n{C.DIM}usa 'explain <esquema>' para el detalle de cada uno{C.RESET}")


def cmd_explain(args):
    if args.scheme not in SCHEMES:
        fail(f"esquema desconocido: {args.scheme} (usa list-schemes)")
    s = SCHEMES[args.scheme]
    print(f"{C.BOLD}{s.name}{C.RESET}\n")
    print(s.explain)


def cmd_tokenize(args):
    src = Path(args.midi)
    if not src.exists():
        fail(f"no existe: {src}")
    scheme = SCHEMES[args.scheme]()
    notes, tpb = load_notes(str(src))
    tokens = scheme.tokenize(notes, tpb)
    out = {"scheme": args.scheme, "source": str(src), "ticks_per_beat": tpb, "tokens": tokens}
    dest = Path(args.output) if args.output else src.with_suffix(f".{args.scheme}.tokens.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    ok(f"{len(tokens)} tokens ({args.scheme}) -> {dest}")


def cmd_detokenize(args):
    src = Path(args.tokens)
    if not src.exists():
        fail(f"no existe: {src}")
    data = json.loads(src.read_text())
    scheme_name = data.get("scheme") or args.scheme
    if not scheme_name:
        fail("el fichero no indica 'scheme'; pásalo con --scheme")
    if scheme_name not in SCHEMES:
        fail(f"esquema desconocido: {scheme_name}")
    scheme = SCHEMES[scheme_name]()
    tpb = data.get("ticks_per_beat", 480)
    notes = scheme.detokenize(data["tokens"], tpb)
    dest = Path(args.output) if args.output else src.with_suffix(".mid")
    notes_to_midi(notes, tpb, str(dest))
    ok(f"{len(notes)} notas reconstruidas -> {dest}")


def cmd_compare(args):
    src = Path(args.midi)
    if not src.exists():
        fail(f"no existe: {src}")
    notes, tpb = load_notes(str(src))
    print(f"{C.BOLD}{src.name}{C.RESET} — {len(notes)} notas, {tpb} ticks/negra\n")
    print(f"{'esquema':<12} {'nº tokens':>10}   primeros tokens")
    for name, cls in SCHEMES.items():
        toks = cls().tokenize(notes, tpb)
        preview = toks[:4]
        preview_s = ", ".join(t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in preview)
        print(f"{C.CYAN}{name:<12}{C.RESET} {len(toks):>10}   {preview_s} ...")


def cmd_info(args):
    path = Path(args.path)
    if not path.exists():
        fail(f"no existe: {path}")
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        print(f"{C.BOLD}{path.name}{C.RESET} — tokens ({data.get('scheme', '?')})")
        info_line(f"origen:  {data.get('source', '?')}")
        info_line(f"tokens:  {len(data.get('tokens', []))}")
        return
    notes, tpb = load_notes(str(path))
    tracks = sorted(set(n.track for n in notes))
    print(f"{C.BOLD}{path.name}{C.RESET} — partitura")
    info_line(f"notas:        {len(notes)}")
    info_line(f"pistas:       {len(tracks)}")
    info_line(f"ticks/negra:  {tpb}")


def main():
    p = argparse.ArgumentParser(prog="miditok_algoritmos.py",
                                 description="Las 9 tokenizaciones de MIDI reimplementadas desde cero, para entenderlas.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-schemes"); sp.set_defaults(func=cmd_list_schemes)

    sp = sub.add_parser("explain", help="explica el algoritmo de un esquema")
    sp.add_argument("scheme", choices=list(SCHEMES))
    sp.set_defaults(func=cmd_explain)

    sp = sub.add_parser("tokenize")
    sp.add_argument("midi")
    sp.add_argument("-s", "--scheme", default="REMI", choices=list(SCHEMES))
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_tokenize)

    sp = sub.add_parser("detokenize")
    sp.add_argument("tokens")
    sp.add_argument("--scheme", default=None, choices=list(SCHEMES))
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_detokenize)

    sp = sub.add_parser("compare", help="tokeniza el mismo MIDI con los 9 esquemas y compara")
    sp.add_argument("midi")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("info")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_info)

    args = p.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
