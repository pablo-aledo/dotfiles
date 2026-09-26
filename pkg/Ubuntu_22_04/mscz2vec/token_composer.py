#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TOKEN COMPOSER  v1                                    ║
║     Transformer autoregresivo sobre las 9 tokenizaciones de miditok_algoritmos║
║                                                                                ║
║  Un único modelo GPT-style (bloques causales pre-LN, posición aprendida) que  ║
║  se adapta a la forma del esquema elegido con --scheme:                      ║
║                                                                                ║
║    ESCALARES  (MIDILike, TSD, REMI, Structured, PerTok, MMM)                 ║
║      secuencia plana de tokens → un embedding, una cabeza de salida.         ║
║                                                                                ║
║    COMPUESTOS (CPWord, Octuple, MuMIDI)                                      ║
║      cada paso es un vector de varios campos (Bar/Position/Pitch/...) →      ║
║      un embedding y una cabeza POR CAMPO sobre un backbone compartido.       ║
║      Simplificación respecto al paper CP-Word: los campos se predicen en     ║
║      paralelo (no Family condicionando al resto) -- suficiente para          ║
║      explorar el esquema, no para igualar el estado del arte.                ║
║                                                                                ║
║  COMANDOS:                                                                    ║
║    prepare   — corpus MIDI → tokens + vocab.json + pickles de ids            ║
║    train     — entrena el Transformer (arquitectura según el esquema)        ║
║    compose   — genera un MIDI nuevo desde un checkpoint (con --prime         ║
║                opcional para "priming"/transferencia de estilo aproximada)   ║
║    infill    — regenera o añade una pista completa de un MIDI existente,     ║
║                condicionando en las demás pistas (solo --scheme MMM)         ║
║    inspect   — diagnóstico de dataset y/o checkpoint                          ║
║                                                                                ║
║  DEPENDENCIAS: mido, numpy, torch                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

python token_composer.py prepare --input-dir midis/ --output-dir data_remi/ --scheme REMI

python token_composer.py train \
    --data-dir data_remi/ --model-dir runs/remi1/ \
    --dim 256 --layers 6 --heads 4 --max-seq 512 \
    --batch-size 8 --epochs 100

python token_composer.py compose --model-dir runs/remi1/ \
    --output generado.mid --length 512 --temperature 0.9

python token_composer.py compose --model-dir runs/remi1/ --output continuacion.mid \
    --prime referencia.mid --prime-length 128 --length 512

python token_composer.py infill --model-dir runs/mmm1/ --input pieza.mid \
    --track 1 --output pieza_bajo_nuevo.mid

python token_composer.py inspect --data-dir data_remi/ --model-dir runs/remi1/
"""

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

POSITIONS_PER_BAR   = 16
VELOCITY_BINS       = 32
MAX_DURATION_STEPS  = 64
MAX_TIMESHIFT_STEPS = 64
MICROTIMING_RANGE   = 4

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, BOS, EOS]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2  # garantizado por el orden de SPECIAL_TOKENS


# ══════════════════════════════════════════════════════════════════════════════
#  NOTA + E/S DE MIDI  (idéntico a miditok_algoritmos.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Note:
    pitch: int
    velocity: int
    start: int
    duration: int
    track: int = 0
    program: int = 0
    is_drum: bool = False


def load_notes(path: str):
    """Lee un MIDI con mido y devuelve (lista de Note, ticks_per_beat).

    Usa una cola por (pista, pitch) para emparejar bien NoteOn/NoteOff cuando
    dos notas del mismo pitch se solapan.
    """
    import mido
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    notes = []
    for ti, track in enumerate(mid.tracks):
        abs_t, program, is_drum, active = 0, 0, False, {}
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
    import mido
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
        events.sort()
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


def safe_tok_int(prefix: str, tok: str):
    """Como tok_int pero devuelve None si tok no tiene el prefijo esperado o
    no termina en un entero válido -- usado al decodificar secuencias
    generadas por el modelo, que pueden no seguir la gramática del esquema."""
    if not tok.startswith(prefix + "_"):
        return None
    try:
        return int(tok.split(prefix + "_", 1)[1])
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  LOS 9 ESQUEMAS  (mismo algoritmo que miditok_algoritmos.py)
# ══════════════════════════════════════════════════════════════════════════════

class BaseScheme:
    name = "Base"
    is_compound = False
    FIELDS = None  # solo para esquemas compuestos

    def tokenize(self, notes, tpb):
        raise NotImplementedError

    def detokenize(self, tokens, tpb):
        raise NotImplementedError


class MIDILikeScheme(BaseScheme):
    name = "MIDILike"

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
                shift = safe_tok_int("TimeShift", tokens[i])
                if shift is None:
                    i += 1; continue
                t += shift * tpp
                i += 1
                continue
            pitch = safe_tok_int("Pitch", tokens[i])
            if pitch is None or i + 2 >= len(tokens):
                i += 1; continue
            velbin = safe_tok_int("Velocity", tokens[i + 1])
            dur = safe_tok_int("Duration", tokens[i + 2])
            if velbin is None or dur is None:
                i += 1; continue
            notes.append(Note(pitch, bin_to_vel(velbin), t, dur * tpp))
            i += 3
        return notes


class REMIScheme(BaseScheme):
    name = "REMI"

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
                bar += 1; i += 1; continue
            pos = safe_tok_int("Position", tokens[i])
            if pos is None or i + 3 >= len(tokens):
                i += 1; continue
            pitch = safe_tok_int("Pitch", tokens[i + 1])
            velbin = safe_tok_int("Velocity", tokens[i + 2])
            dur = safe_tok_int("Duration", tokens[i + 3])
            if pitch is None or velbin is None or dur is None:
                i += 1; continue
            t = (max(bar, 0) * POSITIONS_PER_BAR + pos) * tpp
            notes.append(Note(pitch, bin_to_vel(velbin), t, dur * tpp))
            i += 4
        return notes


class StructuredScheme(BaseScheme):
    name = "Structured"

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
        step, notes, i = 0, [], 0
        while i < len(tokens):
            shift = safe_tok_int("TimeShift", tokens[i])
            if shift is None or i + 3 >= len(tokens):
                i += 1; continue
            pitch = safe_tok_int("Pitch", tokens[i + 1])
            velbin = safe_tok_int("Velocity", tokens[i + 2])
            dur = safe_tok_int("Duration", tokens[i + 3])
            if pitch is None or velbin is None or dur is None:
                i += 1; continue
            step += shift
            notes.append(Note(pitch, bin_to_vel(velbin), step * tpp, dur * tpp))
            i += 4
        return notes


class PerTokScheme(BaseScheme):
    name = "PerTok"

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
                tokens.append("Bar_None"); cur_bar += 1
            d = max(1, min(MAX_DURATION_STEPS, round(n.duration / tpp)))
            tokens += [f"Position_{pos}", f"MicroTiming_{mt:+d}", f"Pitch_{n.pitch}",
                       f"Velocity_{vel_to_bin(n.velocity)}", f"Duration_{d}"]
        return tokens

    def detokenize(self, tokens, tpb):
        tpp = ticks_per_position(tpb)
        bar, notes, i = -1, [], 0
        while i < len(tokens):
            if tokens[i] == "Bar_None":
                bar += 1; i += 1; continue
            pos = safe_tok_int("Position", tokens[i])
            if pos is None or i + 4 >= len(tokens):
                i += 1; continue
            mt = safe_tok_int("MicroTiming", tokens[i + 1])
            pitch = safe_tok_int("Pitch", tokens[i + 2])
            velbin = safe_tok_int("Velocity", tokens[i + 3])
            dur = safe_tok_int("Duration", tokens[i + 4])
            if mt is None or pitch is None or velbin is None or dur is None:
                i += 1; continue
            step = max(bar, 0) * POSITIONS_PER_BAR + pos
            t = step * tpp + round(mt / (MICROTIMING_RANGE * 2) * tpp)
            notes.append(Note(pitch, bin_to_vel(velbin), max(0, t), dur * tpp))
            i += 5
        return notes


IGNORE = "Ignore"


class CPWordScheme(BaseScheme):
    name = "CPWord"
    is_compound = True
    FIELDS = ["Family", "Bar", "Position", "Pitch", "Velocity", "Duration"]

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
            if tok.get("Family") == "Metric":
                bar += 1; continue
            if tok.get("Family") != "Note":
                continue
            pos, pitch, velbin, dur = tok.get("Position"), tok.get("Pitch"), tok.get("Velocity"), tok.get("Duration")
            if not all(isinstance(v, int) for v in (pos, pitch, velbin, dur)):
                continue  # combinación de campos inconsistente (posible en un modelo poco entrenado)
            t = (max(bar, 0) * POSITIONS_PER_BAR + pos) * tpp
            notes.append(Note(pitch, bin_to_vel(velbin), t, dur * tpp))
        return notes


class OctupleScheme(BaseScheme):
    name = "Octuple"
    is_compound = True
    FIELDS = ["Bar", "Position", "Pitch", "Velocity", "Duration", "Program", "TimeSig", "Tempo"]

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
            bar, pos, pitch, velbin, dur, prog = (tok.get("Bar"), tok.get("Position"), tok.get("Pitch"),
                                                    tok.get("Velocity"), tok.get("Duration"), tok.get("Program"))
            if not all(isinstance(v, int) for v in (bar, pos, pitch, velbin, dur)):
                continue
            is_drum = prog == "drums"
            if not is_drum and not isinstance(prog, int):
                continue
            t = (bar * POSITIONS_PER_BAR + pos) * tpp
            notes.append(Note(pitch, bin_to_vel(velbin), t, dur * tpp,
                               program=0 if is_drum else prog, is_drum=is_drum))
        return notes


class MuMIDIScheme(BaseScheme):
    name = "MuMIDI"
    is_compound = True
    FIELDS = ["Program", "Pitch", "Velocity", "Duration", "BarPosEnc", "PositionPosEnc"]

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
            barenc, pos, pitch, velbin, dur, prog = (tok.get("BarPosEnc"), tok.get("PositionPosEnc"),
                                                       tok.get("Pitch"), tok.get("Velocity"),
                                                       tok.get("Duration"), tok.get("Program"))
            if not all(isinstance(v, int) for v in (barenc, pos, pitch, velbin, dur)):
                continue
            is_drum = prog == "drums"
            if not is_drum and not isinstance(prog, int):
                continue
            if last_barenc is not None and (barenc != last_barenc or pos < last_pos):
                bar += 1
            last_barenc, last_pos = barenc, pos
            t = (bar * POSITIONS_PER_BAR + pos) * tpp
            notes.append(Note(pitch, bin_to_vel(velbin), t, dur * tpp,
                               program=0 if is_drum else prog, is_drum=is_drum))
        return notes


class MMMScheme(BaseScheme):
    name = "MMM"
    base_name = "TSD"

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


SCHEMES = {
    "MIDILike": MIDILikeScheme, "TSD": TSDScheme, "REMI": REMIScheme,
    "Structured": StructuredScheme, "CPWord": CPWordScheme, "Octuple": OctupleScheme,
    "MuMIDI": MuMIDIScheme, "MMM": MMMScheme, "PerTok": PerTokScheme,
}


# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARIO
# ══════════════════════════════════════════════════════════════════════════════

class Vocab:
    """token <-> id, con PAD=0 / BOS=1 / EOS=2 garantizados en toda instancia."""

    def __init__(self):
        self.token2id = {}
        self.id2token = []
        for t in SPECIAL_TOKENS:
            self._add(t)

    def _add(self, tok):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)
        return self.token2id[tok]

    def encode(self, tok):
        return self.token2id.get(tok, PAD_ID)

    def decode(self, idx):
        return self.id2token[idx] if 0 <= idx < len(self.id2token) else PAD

    def __len__(self):
        return len(self.id2token)

    def to_json(self):
        return {"tokens": self.id2token}

    @classmethod
    def from_json(cls, data):
        v = cls()
        for t in data["tokens"]:
            if t not in SPECIAL_TOKENS:
                v._add(t)
        return v


class SchemeVocab:
    """Vocabulario de un esquema completo: un Vocab plano o uno por campo."""

    def __init__(self, scheme_name, is_compound, fields=None):
        self.scheme_name = scheme_name
        self.is_compound = is_compound
        self.fields = fields or []
        if is_compound:
            self.per_field = {f: Vocab() for f in self.fields}
        else:
            self.flat = Vocab()

    def add(self, tok):
        if self.is_compound:
            for f in self.fields:
                self.per_field[f]._add(str(tok[f]))
        else:
            self.flat._add(tok)

    def encode(self, tok):
        if self.is_compound:
            return tuple(self.per_field[f].encode(str(tok[f])) for f in self.fields)
        return self.flat.encode(tok)

    def sizes(self):
        if self.is_compound:
            return {f: len(self.per_field[f]) for f in self.fields}
        return len(self.flat)

    def to_json(self):
        if self.is_compound:
            return {"scheme": self.scheme_name, "is_compound": True, "fields": self.fields,
                    "vocabs": {f: self.per_field[f].to_json() for f in self.fields}}
        return {"scheme": self.scheme_name, "is_compound": False, "vocab": self.flat.to_json()}

    @classmethod
    def from_json(cls, data):
        v = cls(data["scheme"], data["is_compound"], data.get("fields"))
        if v.is_compound:
            for f in v.fields:
                v.per_field[f] = Vocab.from_json(data["vocabs"][f])
        else:
            v.flat = Vocab.from_json(data["vocab"])
        return v


def _coerce_compound_value(v):
    """'4' -> 4, pero 'Ignore'/'New'/'drums'/'4/4' se quedan como string."""
    if isinstance(v, str) and v.lstrip("-").isdigit():
        return int(v)
    return v


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class TokenDataset:
    """Pickles de ids (uno por pieza) → batches de tipo slide_seq2seq."""

    def __init__(self, data_dir, is_compound: bool, seed: int = 42):
        files = sorted(Path(data_dir).glob("*.pickle"))
        if not files:
            raise FileNotFoundError(f"No se encontraron .pickle en {data_dir}")
        rng = random.Random(seed)
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        self.splits = {
            "train": files[:int(n * 0.8)],
            "eval": files[int(n * 0.8):int(n * 0.9)],
            "test": files[int(n * 0.9):],
        }
        self.is_compound = is_compound

    def __repr__(self):
        return (f"<TokenDataset train={len(self.splits['train'])} "
                f"eval={len(self.splits['eval'])} test={len(self.splits['test'])} ficheros>")

    def batch(self, batch_size: int, length: int, split: str = "train"):
        import numpy as np
        pool = self.splits[split]
        if not pool:
            raise ValueError(f"Split '{split}' vacío.")
        chosen = random.sample(pool, k=min(batch_size, len(pool)))
        seqs = [self._load_seq(f, length + 1) for f in chosen]
        seqs = [s for s in seqs if s is not None]
        if not seqs:
            raise IndexError("Secuencias demasiado cortas para el max_seq solicitado.")
        while len(seqs) < batch_size:
            seqs.append(random.choice(seqs))
        return np.stack(seqs[:batch_size])

    def slide_batch(self, batch_size: int, length: int, split: str = "train"):
        data = self.batch(batch_size, length, split)
        if self.is_compound:
            return data[:, :-1, :], data[:, 1:, :]
        return data[:, :-1], data[:, 1:]

    def _load_seq(self, path, max_len):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if len(data) < max_len:
            return None
        start = random.randrange(0, len(data) - max_len + 1)
        return data[start:start + max_len]

    def n_batches(self, batch_size: int, split: str = "train") -> int:
        return max(1, len(self.splits[split]) // batch_size)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO  (factoría: importa torch solo cuando se necesita)
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(vocab_size_or_sizes, is_compound: bool, fields,
                  dim: int, n_layer: int, n_head: int, max_seq: int, dropout: float):
    """
    Backbone GPT-style compartido (bloques causales pre-LN + posición aprendida)
    con una cabeza de entrada/salida escalar o una por campo, según is_compound.
    """
    import torch
    import torch.nn as nn

    class _CausalSelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            assert dim % n_head == 0, "dim debe ser divisible por n_head"
            self.n_head = n_head
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, mask):
            B, L, D = x.shape
            hd = D // self.n_head
            qkv = self.qkv(x).reshape(B, L, 3, self.n_head, hd).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            att = att.masked_fill(mask, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.drop(att)
            out = (att @ v).transpose(1, 2).reshape(B, L, D)
            return self.proj(out)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(dim)
            self.attn = _CausalSelfAttention()
            self.ln2 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(dropout),
            )

        def forward(self, x, mask):
            x = x + self.attn(self.ln1(x), mask)
            x = x + self.ff(self.ln2(x))
            return x

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, dim))
            nn.init.normal_(self.pos_emb, std=0.02)
            self.blocks = nn.ModuleList([_Block() for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, h):
            L = h.size(1)
            h = self.drop(h + self.pos_emb[:, :L, :])
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), diagonal=1)
            for blk in self.blocks:
                h = blk(h, mask)
            return self.ln_f(h)

    if not is_compound:
        class _ScalarTransformer(nn.Module):
            KIND = "scalar"

            def __init__(self):
                super().__init__()
                self.vocab_size = vocab_size_or_sizes
                self.tok_emb = nn.Embedding(self.vocab_size, dim)
                self.backbone = _Backbone()
                self.head = nn.Linear(dim, self.vocab_size)

            def forward(self, x):
                h = self.tok_emb(x.long())
                h = self.backbone(h)
                return self.head(h)

            def config_dict(self):
                return {"kind": "scalar", "vocab_size": self.vocab_size, "dim": dim,
                        "n_layer": n_layer, "n_head": n_head, "max_seq": max_seq}

        return _ScalarTransformer()

    class _CompoundTransformer(nn.Module):
        KIND = "compound"

        def __init__(self):
            super().__init__()
            self.fields = list(fields)
            self.vocab_sizes = dict(vocab_size_or_sizes)
            self.tok_embs = nn.ModuleDict({f: nn.Embedding(self.vocab_sizes[f], dim) for f in self.fields})
            self.backbone = _Backbone()
            self.heads = nn.ModuleDict({f: nn.Linear(dim, self.vocab_sizes[f]) for f in self.fields})

        def forward(self, x):
            h = sum(self.tok_embs[f](x[:, :, i].long()) for i, f in enumerate(self.fields))
            h = self.backbone(h)
            return {f: self.heads[f](h) for f in self.fields}

        def config_dict(self):
            return {"kind": "compound", "fields": self.fields, "vocab_sizes": self.vocab_sizes,
                    "dim": dim, "n_layer": n_layer, "n_head": n_head, "max_seq": max_seq}

    return _CompoundTransformer()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENADOR
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    CHECKPOINT_NAME = "checkpoint.pt"
    BEST_NAME       = "best_model.pt"
    HISTORY_NAME    = "train_history.json"
    CONFIG_NAME     = "model_config.json"

    def __init__(self, model, optimizer, model_dir: Path,
                 label_smooth: float = 0.1, patience: int = 20):
        self.model = model
        self.optimizer = optimizer
        self.model_dir = model_dir
        self.label_smooth = label_smooth
        self.patience = patience
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.best_val_loss = float("inf")
        self.no_improve = 0
        self.start_epoch = 0

    def _loss_and_acc(self, logits, target):
        import torch
        import torch.nn.functional as F
        if self.model.KIND == "scalar":
            V = logits.size(-1)
            loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1).long(),
                                    ignore_index=PAD_ID, label_smoothing=self.label_smooth)
            with torch.no_grad():
                pred = logits.argmax(-1)
                mask = target != PAD_ID
                acc = (pred[mask] == target[mask]).float().mean().item() if mask.any() else 0.0
            return loss, acc

        total_loss, accs = 0.0, []
        for i, f in enumerate(self.model.fields):
            V = logits[f].size(-1)
            tgt_f = target[:, :, i].long()
            mask = tgt_f != PAD_ID
            l = F.cross_entropy(logits[f].reshape(-1, V), tgt_f.reshape(-1),
                                 ignore_index=PAD_ID, label_smoothing=self.label_smooth)
            total_loss = total_loss + l
            with torch.no_grad():
                pred = logits[f].argmax(-1)
                accs.append((pred[mask] == tgt_f[mask]).float().mean().item() if mask.any() else 0.0)
        return total_loss / len(self.model.fields), sum(accs) / len(accs)

    def save_checkpoint(self, epoch, val_loss, is_best):
        import torch
        state = {"epoch": epoch, "model_state": self.model.state_dict(),
                  "optimizer_state": self.optimizer.state_dict(),
                  "best_val_loss": self.best_val_loss, "no_improve": self.no_improve,
                  "history": self.history}
        torch.save(state, self.model_dir / self.CHECKPOINT_NAME)
        if is_best:
            torch.save(state, self.model_dir / self.BEST_NAME)
        with open(self.model_dir / self.HISTORY_NAME, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self):
        import torch
        path = self.model_dir / self.CHECKPOINT_NAME
        if not path.exists():
            print("[train] No se encontró checkpoint — entrenando desde cero.")
            return
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.best_val_loss = state["best_val_loss"]
        self.no_improve = state["no_improve"]
        self.history = state["history"]
        self.start_epoch = state["epoch"] + 1
        print(f"[train] Reanudando desde época {self.start_epoch} (mejor val_loss={self.best_val_loss:.4f})")

    def fit(self, dataset, epochs: int, batch_size: int, max_seq: int, eval_batches: int = 4):
        import torch
        device = next(self.model.parameters()).device
        n_batches = dataset.n_batches(batch_size)

        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            t0, epoch_loss, n_done = time.time(), 0.0, 0
            for _ in range(n_batches):
                try:
                    bx, by = dataset.slide_batch(batch_size, max_seq)
                except (IndexError, ValueError):
                    continue
                bx = torch.from_numpy(bx).to(device)
                by = torch.from_numpy(by).to(device)
                self.optimizer.zero_grad()
                logits = self.model(bx)
                loss, _ = self._loss_and_acc(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_done += 1

            val_loss, val_acc = self._evaluate(dataset, batch_size, max_seq, eval_batches)
            avg_loss = epoch_loss / max(n_done, 1)
            elapsed = time.time() - t0
            self.history["train_loss"].append(avg_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            print(f"[train] época {epoch + 1:4d}/{epochs}  train_loss={avg_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  {elapsed:.1f}s")

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss, self.no_improve = val_loss, 0
            else:
                self.no_improve += 1
            self.save_checkpoint(epoch, val_loss, is_best)

            if self.patience > 0 and self.no_improve >= self.patience:
                print(f"[train] early stopping tras {self.no_improve} épocas sin mejora")
                break

    def _evaluate(self, dataset, batch_size, max_seq, n_batches=4):
        import torch
        self.model.eval()
        device = next(self.model.parameters()).device
        total_loss = total_acc = 0.0
        done = 0
        with torch.no_grad():
            for _ in range(n_batches):
                try:
                    bx, by = dataset.slide_batch(batch_size, max_seq, split="eval")
                except (IndexError, ValueError):
                    continue
                bx = torch.from_numpy(bx).to(device)
                by = torch.from_numpy(by).to(device)
                logits = self.model(bx)
                loss, acc = self._loss_and_acc(logits, by)
                total_loss += loss.item()
                total_acc += acc
                done += 1
        done = max(done, 1)
        return total_loss / done, total_acc / done


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: prepare
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scheme_cls = SCHEMES[args.scheme]
    scheme = scheme_cls()

    midi_files = sorted(list(input_dir.glob("*.mid")) + list(input_dir.glob("*.midi")))
    if not midi_files:
        print(f"[prepare] No se encontraron archivos MIDI en {input_dir}")
        sys.exit(1)

    print(f"[prepare] {len(midi_files)} archivos MIDI  |  esquema: {args.scheme}")

    seqs, ok, errors = [], 0, 0
    for path in midi_files:
        try:
            notes, tpb = load_notes(str(path))
            tokens = scheme.tokenize(notes, tpb)
            if len(tokens) < 4:
                continue
            seqs.append((path.stem, tokens))
            ok += 1
        except Exception as e:
            errors += 1
            print(f"  [{path.stem}] ERROR: {e}")

    if not seqs:
        print("[prepare] Ningún fichero se pudo tokenizar.")
        sys.exit(1)

    svocab = SchemeVocab(args.scheme, scheme.is_compound, scheme.FIELDS)
    for _, tokens in seqs:
        for tok in tokens:
            svocab.add(tok)

    with open(output_dir / "vocab.json", "w") as f:
        json.dump(svocab.to_json(), f, indent=2, ensure_ascii=False)

    import numpy as np
    for name, tokens in seqs:
        if scheme.is_compound:
            ids = np.array([svocab.encode(tok) for tok in tokens], dtype=np.int32)
        else:
            ids = np.array([svocab.encode(tok) for tok in tokens], dtype=np.int32)
        with open(output_dir / f"{name}.pickle", "wb") as f:
            pickle.dump(ids, f)

    sizes = svocab.sizes()
    vocab_desc = (f"{len(scheme.FIELDS)} campos: " + ", ".join(f"{f}={sizes[f]}" for f in scheme.FIELDS)
                  if scheme.is_compound else f"{sizes} tokens")
    print()
    print("═" * 60)
    print("  RESUMEN PREPARE")
    print("═" * 60)
    print(f"  Ficheros tokenizados : {ok}")
    print(f"  Errores              : {errors}")
    print(f"  Vocabulario          : {vocab_desc}")
    print(f"  Salida               : {output_dir}/")
    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    import torch

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    with open(data_dir / "vocab.json") as f:
        svocab = SchemeVocab.from_json(json.load(f))

    dataset = TokenDataset(data_dir, is_compound=svocab.is_compound, seed=args.seed)
    print(f"[train] {dataset}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] dispositivo: {device}")

    model = _build_model(svocab.sizes(), svocab.is_compound, svocab.fields,
                          dim=args.dim, n_layer=args.layers, n_head=args.heads,
                          max_seq=args.max_seq, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] modelo {model.KIND}: {n_params:,} parámetros")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    trainer = Trainer(model, optimizer, model_dir, label_smooth=args.label_smooth, patience=args.patience)

    cfg = model.config_dict()
    cfg["scheme"] = svocab.scheme_name
    with open(model_dir / Trainer.CONFIG_NAME, "w") as f:
        json.dump(cfg, f, indent=2)
    with open(model_dir / "vocab.json", "w") as f:
        json.dump(svocab.to_json(), f, indent=2, ensure_ascii=False)

    if args.resume:
        trainer.load_checkpoint()

    trainer.fit(dataset, epochs=args.epochs, batch_size=args.batch_size, max_seq=args.max_seq)
    print(f"[train] entrenamiento completado. Modelo en {model_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: compose
# ══════════════════════════════════════════════════════════════════════════════

def cmd_compose(args):
    import torch
    import torch.nn.functional as F

    model_dir = Path(args.model_dir)
    with open(model_dir / Trainer.CONFIG_NAME) as f:
        cfg = json.load(f)
    with open(model_dir / "vocab.json") as f:
        svocab = SchemeVocab.from_json(json.load(f))
    scheme = SCHEMES[svocab.scheme_name]()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_arg = svocab.sizes() if svocab.is_compound else svocab.sizes()
    model = _build_model(vocab_arg, svocab.is_compound, svocab.fields,
                          dim=cfg["dim"], n_layer=cfg["n_layer"], n_head=cfg["n_head"],
                          max_seq=cfg["max_seq"], dropout=0.0).to(device)

    ckpt_name = Trainer.BEST_NAME if (model_dir / Trainer.BEST_NAME).exists() else Trainer.CHECKPOINT_NAME
    state = torch.load(model_dir / ckpt_name, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[compose] modelo cargado ({n_params:,} parámetros, esquema {svocab.scheme_name})")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    n_fields = len(svocab.fields) if svocab.is_compound else 1
    eos_row = [EOS_ID] * n_fields

    if args.prime:
        prime_notes, prime_tpb = load_notes(args.prime)
        prime_tokens = scheme.tokenize(prime_notes, prime_tpb)
        if args.prime_length:
            prime_tokens = prime_tokens[:args.prime_length]

        if svocab.is_compound:
            prime_ids = [list(svocab.encode(tok)) for tok in prime_tokens]
            unknown = sum(1 for ids in prime_ids if PAD_ID in ids)
        else:
            prime_ids = [svocab.encode(tok) for tok in prime_tokens]
            unknown = sum(1 for i in prime_ids if i == PAD_ID)

        cap = cfg["max_seq"] - 1  # dejar sitio para al menos un token generado
        if len(prime_ids) > cap:
            print(f"[compose] prime de {len(prime_ids)} tokens recortado a los últimos {cap} (max_seq={cfg['max_seq']})")
            prime_ids = prime_ids[-cap:]

        seq = ([[BOS_ID] * n_fields] if svocab.is_compound else [BOS_ID]) + prime_ids
        print(f"[compose] priming con {args.prime} → {len(prime_ids)} tokens"
              + (f"  ({unknown} fuera del vocabulario entrenado, se tratan como padding)" if unknown else ""))
    else:
        seq = [[BOS_ID] * n_fields] if svocab.is_compound else [BOS_ID]

    max_seq = cfg["max_seq"]
    with torch.no_grad():
        for _ in range(args.length):
            ctx = seq[-max_seq:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)
            if svocab.is_compound:
                nxt = []
                for f in svocab.fields:
                    lg = logits[f][0, -1, :] / max(args.temperature, 1e-8)
                    p = F.softmax(lg, dim=-1)
                    nxt.append(torch.multinomial(p, 1).item())
                seq.append(nxt)
                if nxt == eos_row:
                    break
            else:
                lg = logits[0, -1, :] / max(args.temperature, 1e-8)
                p = F.softmax(lg, dim=-1)
                nxt = torch.multinomial(p, 1).item()
                seq.append(nxt)
                if nxt == EOS_ID:
                    break

    body = seq[1:]
    if svocab.is_compound:
        tokens = []
        for ids in body:
            if ids == eos_row:
                break
            tok = {f: _coerce_compound_value(svocab.per_field[f].decode(i))
                   for f, i in zip(svocab.fields, ids)}
            tokens.append(tok)
    else:
        tokens = []
        for i in body:
            if i == EOS_ID:
                break
            tokens.append(svocab.flat.decode(i))

    notes = scheme.detokenize(tokens, args.ticks_per_beat)
    notes_to_midi(notes, args.ticks_per_beat, args.output)
    if args.prime:
        n_prime = len(prime_ids)
        print(f"[compose] {n_prime} tokens de prime + {len(tokens) - n_prime} generados, "
              f"{len(notes)} notas → {args.output}")
    else:
        print(f"[compose] {len(tokens)} tokens generados, {len(notes)} notas → {args.output}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: infill  (solo esquema MMM -- infilling por pista completa)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_infill(args):
    """
    Regenera o añade una pista completa de un MIDI existente, usando las demás
    pistas como contexto. Solo tiene sentido con un modelo entrenado sobre
    --scheme MMM, cuya gramática ya serializa cada pista como un bloque
    autocontenido (Track_Start_<label> ... Track_End).

    Como el Transformer es autoregresivo (cada token solo ve lo que va ANTES en
    la secuencia), los bloques de las pistas que se conservan se colocan TODOS
    antes del bloque a generar -- reordenados respecto a su posición original
    en el fichero si hace falta -- para que actúen como contexto completo.
    El orden de los bloques no afecta a la música resultante: cada uno lleva
    su propio Track_Start_<label>, no depende de su posición.
    """
    import torch
    import torch.nn.functional as F

    model_dir = Path(args.model_dir)
    with open(model_dir / Trainer.CONFIG_NAME) as f:
        cfg = json.load(f)
    with open(model_dir / "vocab.json") as f:
        svocab = SchemeVocab.from_json(json.load(f))

    if svocab.scheme_name != "MMM":
        print(f"[infill] este comando solo funciona con modelos --scheme MMM "
              f"(modelo actual: {svocab.scheme_name})")
        sys.exit(1)

    scheme = SCHEMES["MMM"]()
    base = SCHEMES[scheme.base_name]()  # TSD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(svocab.sizes(), False, None, dim=cfg["dim"], n_layer=cfg["n_layer"],
                          n_head=cfg["n_head"], max_seq=cfg["max_seq"], dropout=0.0).to(device)
    ckpt_name = Trainer.BEST_NAME if (model_dir / Trainer.BEST_NAME).exists() else Trainer.CHECKPOINT_NAME
    state = torch.load(model_dir / ckpt_name, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    print(f"[infill] modelo cargado ({sum(p.numel() for p in model.parameters()):,} parámetros)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    notes, tpb = load_notes(args.input)
    by_track = {}
    for n in notes:
        by_track.setdefault(n.track, []).append(n)

    target_track = args.track
    kept_tracks = sorted(t for t in by_track if t != target_track)

    if by_track.get(target_track):
        tn = by_track[target_track]
        is_drum = any(n.is_drum for n in tn)
        prog = next((n.program for n in tn if not n.is_drum), 0)
        print(f"[infill] regenerando pista {target_track} existente (se descarta su contenido actual)")
    elif args.drums:
        is_drum, prog = True, 0
        print(f"[infill] añadiendo pista {target_track} nueva (batería)")
    elif args.program is not None:
        is_drum, prog = False, args.program
        print(f"[infill] añadiendo pista {target_track} nueva (programa {prog})")
    else:
        print("[infill] la pista indicada no existe en --input: especifica --program N o --drums")
        sys.exit(1)

    label = "drums" if is_drum else prog

    # contexto = bloques de las pistas conservadas (en cualquier orden) + el
    # Track_Start de la pista a generar, todo ANTES de lo que el modelo genera
    context_tokens = []
    for ti in kept_tracks:
        tn = by_track[ti]
        t_is_drum = any(n.is_drum for n in tn)
        t_prog = next((n.program for n in tn if not n.is_drum), 0)
        t_label = "drums" if t_is_drum else t_prog
        context_tokens.append(f"Track_Start_{t_label}")
        context_tokens += base.tokenize(tn, tpb)
        context_tokens.append("Track_End")
    context_tokens.append(f"Track_Start_{label}")

    context_ids = [svocab.encode(tok) for tok in context_tokens]
    unknown = sum(1 for i in context_ids if i == PAD_ID)
    max_seq = cfg["max_seq"]
    if len(context_ids) > max_seq - 1:
        print(f"[infill] contexto de {len(context_ids)} tokens recortado a los últimos "
              f"{max_seq - 1} (max_seq={max_seq}) -- puede perder pistas conservadas más antiguas")
        context_ids = context_ids[-(max_seq - 1):]
    if unknown:
        print(f"[infill] aviso: {unknown} tokens del contexto no estaban en el vocabulario entrenado")

    seq = [BOS_ID] + context_ids
    generated = []
    with torch.no_grad():
        for _ in range(args.max_length):
            ctx = seq[-max_seq:]
            x = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(x)
            lg = logits[0, -1, :] / max(args.temperature, 1e-8)
            p = F.softmax(lg, dim=-1)
            nxt = torch.multinomial(p, 1).item()
            if nxt == EOS_ID:
                break
            tok_str = svocab.flat.decode(nxt)
            if tok_str == "Track_End":
                break
            seq.append(nxt)
            generated.append(tok_str)
        else:
            print(f"[infill] aviso: se alcanzó --max-length ({args.max_length}) sin que "
                  f"el modelo emitiera Track_End -- la pista puede quedar cortada")

    new_notes = base.detokenize(generated, tpb)
    for n in new_notes:
        n.track, n.program, n.is_drum = target_track, prog, is_drum

    final_notes = new_notes + [n for ti in kept_tracks for n in by_track[ti]]
    notes_to_midi(final_notes, args.ticks_per_beat, args.output)
    print(f"[infill] pista {target_track} ({label}): {len(generated)} tokens generados, "
          f"{len(new_notes)} notas nuevas  |  {len(kept_tracks)} pista(s) conservada(s) intacta(s) → {args.output}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    if args.data_dir:
        print(f"\n── Dataset: {args.data_dir} ──")
        data_dir = Path(args.data_dir)
        try:
            with open(data_dir / "vocab.json") as f:
                svocab = SchemeVocab.from_json(json.load(f))
            ds = TokenDataset(data_dir, is_compound=svocab.is_compound)
            print(f"  {ds}")
            print(f"  esquema: {svocab.scheme_name}  ({'compuesto' if svocab.is_compound else 'escalar'})")
            sizes = svocab.sizes()
            if svocab.is_compound:
                for f in svocab.fields:
                    print(f"    campo {f:<16} {sizes[f]} tokens")
            else:
                print(f"    vocabulario: {sizes} tokens")
            for split, files in ds.splits.items():
                lens = []
                for p in files[:50]:
                    with open(p, "rb") as fh:
                        lens.append(len(pickle.load(fh)))
                avg = sum(lens) // max(len(lens), 1)
                print(f"    {split:5s}: {len(files):4d} ficheros, ~{avg} tokens/fichero (muestra)")
        except Exception as e:
            print(f"  ERROR: {e}")

    if args.model_dir:
        print(f"\n── Modelo: {args.model_dir} ──")
        model_dir = Path(args.model_dir)
        try:
            with open(model_dir / Trainer.CONFIG_NAME) as f:
                cfg = json.load(f)
            print("  Config:")
            for k, v in cfg.items():
                print(f"    {k}: {v}")

            ckpt_path = model_dir / Trainer.CHECKPOINT_NAME
            if ckpt_path.exists():
                import torch
                ckpt = torch.load(ckpt_path, map_location="cpu")
                print("  Checkpoint:")
                print(f"    época:       {ckpt.get('epoch', '?')}")
                print(f"    best_val_loss: {ckpt.get('best_val_loss', '?')}")
                hist = ckpt.get("history", {})
                if hist.get("val_loss"):
                    tail = hist["val_loss"][-5:]
                    print(f"    val_loss (últimas): {['%.4f' % v for v in tail]}")
            else:
                print("  (sin checkpoint guardado aún)")
        except FileNotFoundError as e:
            print(f"  {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not args.data_dir and not args.model_dir:
        print("Indica --data-dir y/o --model-dir para inspeccionar.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="token_composer",
        description="Transformer autoregresivo sobre las 9 tokenizaciones de miditok_algoritmos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("prepare", help="corpus MIDI → tokens + vocab.json + pickles de ids")
    p.add_argument("--input-dir", required=True, metavar="DIR")
    p.add_argument("--output-dir", required=True, metavar="DIR")
    p.add_argument("--scheme", required=True, choices=list(SCHEMES))
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="entrena el Transformer")
    p.add_argument("--data-dir", required=True, metavar="DIR")
    p.add_argument("--model-dir", required=True, metavar="DIR")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--max-seq", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--label-smooth", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=20, help="early stopping (0 = desactivado)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("compose", help="genera un MIDI nuevo desde un checkpoint")
    p.add_argument("--model-dir", required=True, metavar="DIR")
    p.add_argument("--output", required=True, metavar="FILE")
    p.add_argument("--length", type=int, default=512, help="tokens a generar tras el prime (si lo hay)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ticks-per-beat", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prime", default=None, metavar="FILE.mid",
        help="MIDI de referencia: tokeniza sus primeros tokens y continúa desde ahí "
             "(transferencia de estilo aproximada por 'priming')")
    p.add_argument("--prime-length", type=int, default=None,
        help="nº máx. de tokens del --prime a usar (por defecto, todos los que quepan en max_seq)")
    p.set_defaults(func=cmd_compose)

    p = sub.add_parser("infill", help="regenera o añade una pista completa (solo modelos --scheme MMM)")
    p.add_argument("--model-dir", required=True, metavar="DIR")
    p.add_argument("--input", required=True, metavar="FILE.mid")
    p.add_argument("--track", type=int, required=True,
        help="índice de pista (0-based, según el orden de pistas del --input) a regenerar o añadir")
    p.add_argument("--program", type=int, default=None,
        help="programa MIDI para una pista nueva (solo si --track no existe ya en --input)")
    p.add_argument("--drums", action="store_true", help="la pista nueva es de batería")
    p.add_argument("--output", required=True, metavar="FILE")
    p.add_argument("--max-length", type=int, default=256,
        help="tope de tokens generados antes de forzar el corte si el modelo no emite Track_End")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--ticks-per-beat", type=int, default=480)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_infill)

    p = sub.add_parser("inspect", help="diagnóstico de dataset y/o checkpoint")
    p.add_argument("--data-dir", default=None, metavar="DIR")
    p.add_argument("--model-dir", default=None, metavar="DIR")
    p.set_defaults(func=cmd_inspect)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
