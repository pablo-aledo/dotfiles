#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      GRAMMAR COMPOSER  v1.1                                  ║
║         Motor de composición dirigido por gramáticas formales                ║
║                                                                              ║
║  DOS MODOS DE USO:                                                           ║
║                                                                              ║
║  ── MODO COMPOSE (por defecto) ────────────────────────────────────────────║
║  Combina un MIDI de entrada con una gramática textual para generar un       ║
║  nuevo MIDI. La gramática define una estructura jerárquica que puede        ║
║  referenciar compases del MIDI original o notas ABC directas.               ║
║                                                                              ║
║  ── MODO LEARN ─────────────────────────────────────────────────────────── ║
║  Analiza un MIDI y extrae automáticamente su gramática usando el            ║
║  algoritmo Re-Pair (simplificación de Sequitur). El resultado se guarda     ║
║  como fichero de gramática listo para usarse con el modo compose.           ║
║  Cada regla generada referencia compases del MIDI original, de modo que     ║
║  la gramática aprendida es directamente reproducible.                       ║
║                                                                              ║
║  FORMATO DE GRAMÁTICA:                                                       ║
║    · La primera línea es la raíz: lista de etiquetas separadas por espacio  ║
║    · Cada línea posterior: <label> : <elementos...>                          ║
║                                                                              ║
║  TIPOS DE ELEMENTOS en una regla:                                            ║
║    · Nota ABC  — nota individual con duración: C4:1  E4:0.5  G#4:2         ║
║                  (nombre+octava:duración_en_negras). La duración admite     ║
║                  enteros, decimales y fracciones exactas: G4:1/3 es una     ║
║                  corchea de tresillo (3 seguidas = 1 negra, sin deriva).    ║
║                  Sufijo & (C4:2&): suena a la vez que el siguiente elemento y║
║                  no avanza el cursor → acordes y polifonía: C4:2& E4:2& G4:2 ║
║    · z:D      — silencio de D negras (avanza el cursor sin sonar): z:1  z:1/3║
║    · Número    — compás N del MIDI de entrada (base 1)                      ║
║    · M-N       — rango de compases M a N del MIDI de entrada (inclusivos)  ║
║    · [label]   — referencia recursiva a otra regla de la gramática          ║
║    · +N / -N   — modifica el OFFSET GLOBAL de transposición en N semitonos  ║
║                  (acumulativo). Afecta a todas las notas posteriores en el  ║
║                  orden de expansión (compases del MIDI y notas ABC), también║
║                  fuera de la regla donde aparece. Empieza en 0. Se puede    ║
║                  usar también en la raíz. No afecta al canal 10 (batería).  ║
║                  Ej:  motivo: 1-2 +2 1-2 -2     (2.ª vez dos semitonos más) ║
║    · 0         — resetea el offset global a 0 (no es un compás: los        ║
║                  compases empiezan en 1).                                    ║
║                                                                              ║
║  EJEMPLO DE GRAMÁTICA (generada o escrita a mano):                          ║
║    intro verse chorus verse chorus outro                                     ║
║    intro: 1-4                                                                ║
║    verse: [motivo_a] [motivo_b] [motivo_a] 9                                ║
║    chorus: C4:1 E4:1 G4:1 C5:2 [cadencia]                                  ║
║    motivo_a: 5-6                                                             ║
║    motivo_b: G3:0.5 A3:0.5 B3:0.5 C4:0.5                                   ║
║    cadencia: 10-12                                                           ║
║    outro: [cadencia] C4:4                                                    ║
║                                                                              ║
║  USO — MODO COMPOSE:                                                         ║
║    python grammar_composer.py entrada.mid gramatica.txt                     ║
║    python grammar_composer.py entrada.mid gramatica.txt -o salida.mid       ║
║    python grammar_composer.py entrada.mid gramatica.txt --tempo 120         ║
║    python grammar_composer.py entrada.mid gramatica.txt --channel 0         ║
║    python grammar_composer.py entrada.mid gramatica.txt --report            ║
║    python grammar_composer.py entrada.mid gramatica.txt --verbose           ║
║                                                                              ║
║  USO — MODO LEARN:                                                           ║
║    python grammar_composer.py --learn entrada.mid                           ║
║    python grammar_composer.py --learn entrada.mid -o gramatica.txt         ║
║    python grammar_composer.py --learn entrada.mid --min-pair 3              ║
║    python grammar_composer.py --learn entrada.mid --max-rules 20            ║
║    python grammar_composer.py --learn entrada.mid --report --verbose        ║
║                                                                              ║
║  USO — MODO LEARN-GESTURES (codebook de gestos transponibles):               ║
║    python grammar_composer.py --learn-gestures entrada.mid                   ║
║    python grammar_composer.py --learn-gestures entrada.mid -o gestos.txt     ║
║    python grammar_composer.py --learn-gestures entrada.mid --quantize 1/12   ║
║    python grammar_composer.py entrada.mid gestos.txt   (recompone la obra)   ║
║    Aprende grupos de notas (con su ritmo) que se repiten a distinta altura y ║
║    escribe una gramática autocontenida: un gesto gN por grupo y una regla    ║
║    `song` que los coloca con el offset global (+n / -n / 0). Al terminar     ║
║    recompone la gramática y comprueba que coincide con el original.          ║
║    No aprende velocidades (usa --velocity). Usa --min-pair y --max-rules.    ║
║    --quantize FRAC  Rejilla en negras (1/4, 1/12…) para MIDIs interpretados. ║
║                                                                              ║
║  OPCIONES COMUNES:                                                           ║
║    --track N          Track del MIDI de entrada a usar (default: auto)      ║
║    --report           Mostrar informe detallado                              ║
║    --verbose          Traza paso a paso                                      ║
║                                                                              ║
║  OPCIONES MODO COMPOSE:                                                      ║
║    --out / -o FILE    Fichero MIDI de salida (default: entrada.grammar.mid) ║
║    --tempo BPM        Tempo de salida (default: detectado del MIDI)         ║
║    --channel N        Canal MIDI para notas ABC (default: 0)                ║
║    --velocity N       Velocidad para notas ABC (default: 80)                ║
║    --tpb N            Ticks por negra en salida (default: como entrada)     ║
║                                                                              ║
║  OPCIONES MODO LEARN:                                                        ║
║    --learn            Activar modo de aprendizaje de gramática               ║
║    --out / -o FILE    Fichero de gramática de salida (default: midi.grammar.txt)║
║    --min-pair N       Frecuencia mínima de un par para crear regla (def: 2) ║
║    --max-rules N      Número máximo de reglas a generar (def: ilimitado)    ║
║    --merge-bars       Fusionar compases consecutivos repetidos en rangos     ║
║                                                                              ║
║  NOTAS ABC MIDI soportadas:                                                  ║
║    Nombre: C D E F G A B  (con # o b para alteraciones: C#, Eb)             ║
║  Octava:  -1 a 9 (octava MIDI estándar: C4 = nota 60, C-1 = nota 0)          ║
║    Duración: número decimal en negras (1 = negra, 0.5 = corchea, 4 = redonda)║
║                                                                              ║
║  DEPENDENCIAS: mido                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import re
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

VERSION = "1.1"
MAX_RECURSION_DEPTH = 64   # protección frente a gramáticas con ciclos

NOTE_NAMES = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

ABC_NOTE_RE = re.compile(
    r'^([A-Ga-g])([#b]?)(-?\d):(\d+/\d+|[\d.]+)(&?)$'   # C#4:0.5  eb3:2  G4:1/3  C4:1&
)
REST_RE    = re.compile(r'^[zZ]:(\d+/\d+|[\d.]+)$')   # e.g.  z:1  z:1/3  (silencio)
RANGE_RE   = re.compile(r'^(\d+)-(\d+)$')   # e.g.  3-7
NUMBER_RE  = re.compile(r'^\d+$')            # e.g.  5
LABEL_RE   = re.compile(r'^\[([^\]]+)\]$')  # e.g.  [verso]
SHIFT_RE   = re.compile(r'^([+-])(\d+)$')    # e.g.  +2  -12  (offset global)

PERCUSSION_CHANNEL = 9   # canal 10 GM: no se transpone
RESET_TOKEN = "0"        # token que resetea el offset global a 0 (no es un compás)


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MidiNote:
    """Nota MIDI lista para escribir en un track (tiempo relativo al cursor)."""
    pitch:    int
    velocity: int
    duration_ticks: int
    channel:  int = 0


@dataclass
class ExpansionReport:
    """Registro de la expansión de la gramática."""
    root_sequence:   list[str]               = field(default_factory=list)
    rules_expanded:  list[tuple[str, str]]   = field(default_factory=list)  # (label, tipo)
    bars_used:       list[int]               = field(default_factory=list)
    abc_notes_added: int                     = 0
    total_notes:     int                     = 0
    warnings:        list[str]               = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# PARSEO DE GRAMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

def parse_grammar(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """
    Parsea el fichero de gramática.

    Devuelve:
        root    — lista de etiquetas de la primera línea (raíz)
        rules   — diccionario {label: [token, token, …]}
    """
    lines = [l.rstrip() for l in text.splitlines() if l.strip() and not l.strip().startswith('#')]

    if not lines:
        raise ValueError("El fichero de gramática está vacío.")

    # Primera línea: raíz — puede contener etiquetas, números de compás y rangos M-N
    root = lines[0].split()

    rules: dict[str, list[str]] = {}
    for lineno, line in enumerate(lines[1:], start=2):
        if ':' not in line:
            raise ValueError(f"Línea {lineno}: formato inválido (falta ':') → {line!r}")
        label, _, body = line.partition(':')
        label = label.strip()
        tokens = body.split()
        if not tokens:
            raise ValueError(f"Línea {lineno}: regla '{label}' no tiene cuerpo.")
        rules[label] = tokens

    # Verificar que las etiquetas de la raíz que no son números/rangos tienen regla
    for r in root:
        if RANGE_RE.match(r) or NUMBER_RE.match(r) or SHIFT_RE.match(r):
            continue   # compás, rango u offset global: válidos en la raíz
        if LABEL_RE.match(r):
            # Token entre corchetes en la raíz: extraer nombre
            label_name = LABEL_RE.match(r).group(1)
            if label_name not in rules:
                raise ValueError(f"La etiqueta raíz '[{label_name}]' no tiene regla definida.")
            continue
        if r not in rules:
            raise ValueError(f"La etiqueta raíz '{r}' no tiene regla definida.")

    return root, rules


def grammar_uses_bars(root: list[str], rules: dict[str, list[str]]) -> bool:
    """True si la raíz o alguna regla referencia compases del MIDI (N o M-N)."""
    for tok in root:
        if tok != RESET_TOKEN and (RANGE_RE.match(tok) or NUMBER_RE.match(tok)):
            return True
    for body in rules.values():
        for elem in tokenize_rule(body):
            if elem['type'] in ('bar', 'range'):
                return True
    return False


def _parse_duration(text: str, tok: str) -> Fraction:
    """
    Duración en negras como Fraction exacta. Acepta enteros (2), decimales
    (0.5) y fracciones (1/3, 2/3, 3/2). Con fracciones los tresillos y otros
    grupos irregulares no acumulan error de redondeo.
    """
    try:
        dur = Fraction(text)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Duración inválida en nota ABC: {tok!r}")
    if dur <= 0:
        raise ValueError(f"La duración debe ser positiva: {tok!r}")
    return dur


def tokenize_rule(tokens: list[str]) -> list[dict]:
    """
    Clasifica cada token de una regla en su tipo:
        {'type': 'note',  'name': ..., 'alter': ..., 'octave': ..., 'dur': ...}
        {'type': 'bar',   'n': int}
        {'type': 'range', 'm': int, 'n': int}
        {'type': 'ref',   'label': str}
        {'type': 'shift', 'n': int}      # offset global, con signo
        {'type': 'reset'}                # token "0"
        {'type': 'rest',  'dur': Fraction}   # silencio  z:dur
    Las notas llevan además 'sync' (True si el token acaba en '&').
    """
    parsed = []
    for tok in tokens:
        if tok == RESET_TOKEN:
            parsed.append({'type': 'reset'})
            continue
        m = SHIFT_RE.match(tok)
        if m:
            n = int(m.group(2))
            parsed.append({'type': 'shift', 'n': -n if m.group(1) == '-' else n})
            continue

        m = ABC_NOTE_RE.match(tok)
        if m:
            parsed.append({
                'type':   'note',
                'name':   m.group(1).upper(),
                'alter':  m.group(2),
                'octave': int(m.group(3)),
                'dur':    _parse_duration(m.group(4), tok),
                'sync':   m.group(5) == '&',   # True → el cursor NO avanza
            })
            continue

        m = REST_RE.match(tok)
        if m:
            parsed.append({'type': 'rest', 'dur': _parse_duration(m.group(1), tok)})
            continue

        m = RANGE_RE.match(tok)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                lo, hi = hi, lo
            parsed.append({'type': 'range', 'm': lo, 'n': hi})
            continue

        if NUMBER_RE.match(tok):
            parsed.append({'type': 'bar', 'n': int(tok)})
            continue

        m = LABEL_RE.match(tok)
        if m:
            parsed.append({'type': 'ref', 'label': m.group(1)})
            continue

        raise ValueError(f"Token no reconocido en gramática: {tok!r}")

    return parsed


# ═══════════════════════════════════════════════════════════════════════════════
# LECTURA DE COMPASES DEL MIDI DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def get_tempo(mid: MidiFile) -> int:
    """Devuelve el primer tempo encontrado (µs/negra) o 500 000 (120 BPM)."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500_000


def choose_main_track(mid: MidiFile) -> int:
    """
    Elige el track con más mensajes note_on como track principal.
    Ignora el track 0 si contiene sólo meta-mensajes (típico en Type 1).
    """
    best_idx, best_count = 0, -1
    for i, track in enumerate(mid.tracks):
        count = sum(1 for m in track if m.type == 'note_on' and m.velocity > 0)
        if count > best_count:
            best_count, best_idx = count, i
    return best_idx


def get_bar_length_ticks(mid: MidiFile) -> 'Callable[[int], int]':
    """
    Devuelve una función bar_length(bar_n) que calcula la duración en ticks
    del compás bar_n (base 1), teniendo en cuenta cambios de firma de compás.

    Lee las TimeSignature del track 0 (donde suelen vivir los meta-mensajes).
    Si no hay ninguna, asume 4/4.
    """
    tpb = mid.ticks_per_beat
    time_sigs: list[tuple[int,int,int]] = [(0, 4, 4)]  # (abs_tick, num, denom)
    abs_t = 0
    for msg in mid.tracks[0]:
        abs_t += msg.time
        if msg.type == 'time_signature':
            time_sigs.append((abs_t, msg.numerator, msg.denominator))
    time_sigs.sort(key=lambda x: x[0])

    def bar_ticks_at_tick(tick: int) -> int:
        num, denom = 4, 4
        for ts_tick, ts_num, ts_denom in time_sigs:
            if ts_tick <= tick:
                num, denom = ts_num, ts_denom
        return int(tpb * num * 4 / denom)

    def bar_start(bar_n: int) -> int:
        cursor = 0
        for b in range(1, bar_n):
            cursor += bar_ticks_at_tick(cursor)
        return cursor

    def bar_length(bar_n: int) -> int:
        start = bar_start(bar_n)
        return bar_ticks_at_tick(start)

    return bar_length


def extract_bars(mid: MidiFile, track_idx: int) -> dict[int, list[MidiNote]]:
    """
    Segmenta el track en compases (base 1).

    Devuelve  {bar_number: [MidiNote, …]}  usando el tiempo de firma de compás
    del fichero. Si no hay TimeSignature, asume 4/4.

    La duración de cada nota se calcula a partir de note_off / note_on-vel0.
    """
    tpb = mid.ticks_per_beat

    # Recolectar eventos del track con tiempo absoluto
    track  = mid.tracks[track_idx]
    abs_t  = 0
    events = []
    time_sigs = [(0, 4, 4)]   # (abs_tick, num, denom)
    for msg in track:
        abs_t += msg.time
        events.append((abs_t, msg))
        if msg.type == 'time_signature':
            time_sigs.append((abs_t, msg.numerator, msg.denominator))

    # También buscar TimeSignature en track 0
    if track_idx != 0:
        abs_t0 = 0
        for msg in mid.tracks[0]:
            abs_t0 += msg.time
            if msg.type == 'time_signature':
                time_sigs.append((abs_t0, msg.numerator, msg.denominator))
    time_sigs.sort(key=lambda x: x[0])

    def bar_ticks_at(tick: int) -> int:
        """Duración en ticks de un compás en el momento 'tick'."""
        num, denom = 4, 4
        for ts_tick, ts_num, ts_denom in time_sigs:
            if ts_tick <= tick:
                num, denom = ts_num, ts_denom
        beats_per_bar = num * 4 / denom
        return int(tpb * beats_per_bar)

    def tick_to_bar(tick: int) -> int:
        """Compás (base 1) correspondiente al tick absoluto."""
        bar = 1
        cursor = 0
        while True:
            blen = bar_ticks_at(cursor)
            if cursor + blen > tick:
                return bar
            cursor += blen
            bar    += 1

    def bar_start_tick(bar_n: int) -> int:
        cursor = 0
        for b in range(1, bar_n):
            cursor += bar_ticks_at(cursor)
        return cursor

    # Construir notas con duración usando pares note_on / note_off
    active: dict[tuple[int,int], list[int]] = {}   # (channel, pitch) → [abs_start, …]
    raw_notes: list[tuple[int,int,int,int,int]] = []  # (start, end, pitch, vel, ch)

    for abs_t, msg in events:
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (msg.channel, msg.note)
            active.setdefault(key, []).append(abs_t)
        elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active and active[key]:
                start = active[key].pop(0)
                raw_notes.append((start, abs_t, msg.note,
                                  64,            # velocidad nominal
                                  msg.channel))

    # Necesitamos también capturar la velocidad real del note_on
    # Segunda pasada más precisa
    active2: dict[tuple[int,int], list[tuple[int,int]]] = {}  # (ch,note) → [(abs_t, vel)]
    raw_notes2: list[tuple[int,int,int,int,int]] = []

    for abs_t, msg in events:
        if msg.type == 'note_on' and msg.velocity > 0:
            key = (msg.channel, msg.note)
            active2.setdefault(key, []).append((abs_t, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in active2 and active2[key]:
                start_t, vel = active2[key].pop(0)
                raw_notes2.append((start_t, abs_t, msg.note, vel, msg.channel))

    # Agrupar por compás
    bars: dict[int, list[MidiNote]] = {}
    for (start_t, end_t, pitch, vel, ch) in raw_notes2:
        bar_n    = tick_to_bar(start_t)
        bar_st   = bar_start_tick(bar_n)
        dur_tick = max(1, end_t - start_t)
        note = MidiNote(
            pitch          = pitch,
            velocity       = vel,
            duration_ticks = dur_tick,
            channel        = ch,
        )
        # offset dentro del compás como campo extra (no en dataclass, usamos tupla)
        bars.setdefault(bar_n, [])
        bars[bar_n].append((start_t - bar_st, note))   # (offset, MidiNote)

    # Ordenar las notas dentro de cada compás por offset
    for k in bars:
        bars[k].sort(key=lambda x: x[0])

    return bars


def extract_all_tracks(
    mid: MidiFile,
) -> dict[int, dict[int, list[tuple[int, 'MidiNote']]]]:
    """
    Extrae los compases de TODOS los tracks del MIDI que contengan notas.

    Devuelve  {track_idx: {bar_n: [(offset, MidiNote), …]}, …}

    Los tracks sin notas (p. ej. el track 0 de meta-mensajes en Type 1)
    se omiten del resultado pero sus TimeSignature y tempo sí se tienen en
    cuenta a través de extract_bars.
    """
    result: dict[int, dict] = {}
    for i, track in enumerate(mid.tracks):
        has_notes = any(
            m.type == 'note_on' and m.velocity > 0 for m in track
        )
        if not has_notes:
            continue
        bars = extract_bars(mid, i)
        if bars:
            result[i] = bars
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE EXPANSIÓN DE LA GRAMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

def abc_to_pitch(name: str, alter: str, octave: int) -> int:
    """Convierte nombre+alteración+octava a número MIDI. C4 = 60."""
    base   = NOTE_NAMES[name.upper()]
    alter_v = {'#': 1, 'b': -1, '': 0}[alter]
    return (octave + 1) * 12 + base + alter_v


class GrammarExpander:
    def __init__(
        self,
        rules:        dict[str, list[str]],
        bars:         dict[int, list[tuple[int, MidiNote]]],
        tpb:          int,
        bar_length_fn: 'Callable[[int], int] | None' = None,
        channel:      int   = 0,
        velocity:     int   = 80,
        verbose:      bool  = False,
    ):
        self.rules          = rules
        self.bars           = bars
        self.tpb            = tpb
        self.bar_length_fn  = bar_length_fn   # bar_n → duración métrica en ticks
        self.channel        = channel
        self.velocity       = velocity
        self.verbose        = verbose
        self.report         = ExpansionReport()
        self.pitch_offset   = 0   # offset global en semitonos (token +N / -N)

    # ──────────────────────────────────────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────────────────────────────────────

    def expand_root(self, root: list[str]) -> list[tuple[int, MidiNote]]:
        """
        Expande la secuencia raíz y devuelve la lista global de notas como
        [(offset_ticks_absolute, MidiNote), …] con offsets acumulados.

        La raíz puede contener:
          · nombres de regla (sin corchetes)   → se expanden recursivamente
          · números de compás "N"              → se extraen del MIDI de entrada
          · rangos "M-N"                       → ídem
          · referencias "[label]"              → se expanden como reglas
        """
        self.report.root_sequence = root
        all_notes: list[tuple[int, MidiNote]] = []
        cursor = 0

        self.pitch_offset = 0
        for token in root:
            # Determinar tipo de token de la raíz
            m_shift = SHIFT_RE.match(token)
            m_range = RANGE_RE.match(token)
            m_num   = NUMBER_RE.match(token)
            m_label = LABEL_RE.match(token)

            if token == RESET_TOKEN:
                self._reset_shift()
                continue   # no produce notas ni consume tiempo
            if m_shift:
                self._apply_shift(int(m_shift.group(2)) * (-1 if m_shift.group(1) == '-' else 1))
                continue   # no produce notas ni consume tiempo
            if m_range:
                lo, hi = int(m_range.group(1)), int(m_range.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                notes = self._get_bars(lo, hi)
            elif m_num:
                notes = self._get_bars(int(token), int(token))
            elif m_label:
                notes = self._expand_label(m_label.group(1), depth=0)
            else:
                # Nombre de regla directo (sin corchetes, generado por learn)
                notes = self._expand_label(token, depth=0)

            for (off, note) in notes:
                # El cursor puede ser una Fraction (notas ABC con fracciones):
                # se redondea una sola vez al tick final, sin acumular error.
                all_notes.append((round(cursor + off), note))
            cursor += self._token_duration(token)

        self.report.total_notes = len(all_notes)
        return all_notes

    # ──────────────────────────────────────────────────────────────────────────
    # Expansión recursiva de etiquetas
    # ──────────────────────────────────────────────────────────────────────────

    def _expand_label(self, label: str, depth: int) -> list[tuple[int, MidiNote]]:
        if depth > MAX_RECURSION_DEPTH:
            msg = f"Profundidad máxima de recursión alcanzada en '{label}'. ¿Ciclo en gramática?"
            self.report.warnings.append(msg)
            print(f"  [AVISO] {msg}", file=sys.stderr)
            return []

        if label not in self.rules:
            msg = f"Etiqueta '{label}' no definida en la gramática."
            self.report.warnings.append(msg)
            print(f"  [AVISO] {msg}", file=sys.stderr)
            return []

        tokens   = self.rules[label]
        parsed   = tokenize_rule(tokens)
        result   = []
        cursor   = 0

        for elem in parsed:
            if elem['type'] == 'note':
                notes = self._make_abc_note(elem)
                self.report.abc_notes_added += len(notes)
                tipo = 'nota-abc'
            elif elem['type'] == 'bar':
                notes = self._get_bars(elem['n'], elem['n'])
                tipo  = f"compás-{elem['n']}"
            elif elem['type'] == 'range':
                notes = self._get_bars(elem['m'], elem['n'])
                tipo  = f"compases-{elem['m']}-{elem['n']}"
            elif elem['type'] == 'ref':
                notes = self._expand_label(elem['label'], depth + 1)
                tipo  = f"ref-{elem['label']}"
            elif elem['type'] == 'rest':
                notes = []
                tipo  = 'silencio'
            elif elem['type'] == 'reset':
                self._reset_shift()
                notes = []
                tipo  = 'offset-reset'
            elif elem['type'] == 'shift':
                self._apply_shift(elem['n'])
                notes = []
                tipo  = f"offset{elem['n']:+d}"
            else:
                notes = []
                tipo  = 'desconocido'

            self.report.rules_expanded.append((label, tipo))

            if self.verbose:
                print(f"  {'  ' * depth}[{label}] → {tipo}: {len(notes)} notas")

            for (off, note) in notes:
                result.append((cursor + off, note))

            cursor += self._elem_duration(elem)

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Nota ABC → MidiNote
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_shift(self, n: int) -> None:
        """Acumula n semitonos en el offset global."""
        self.pitch_offset += n
        if self.verbose:
            print(f"  [offset] {n:+d} → offset global = {self.pitch_offset:+d}")

    def _reset_shift(self) -> None:
        """Vuelve el offset global a 0 (token '0')."""
        self.pitch_offset = 0
        if self.verbose:
            print("  [offset] reset → offset global = +0")

    def _transpose(self, note: MidiNote) -> MidiNote:
        """
        Devuelve la nota con el offset global aplicado (copia; nunca muta la
        nota original, que puede reutilizarse en otros compases). La batería
        (canal 10) no se transpone.
        """
        if self.pitch_offset == 0 or note.channel == PERCUSSION_CHANNEL:
            return note
        return MidiNote(
            pitch          = max(0, min(127, note.pitch + self.pitch_offset)),
            velocity       = note.velocity,
            duration_ticks = note.duration_ticks,
            channel        = note.channel,
        )

    def _make_abc_note(self, elem: dict) -> list[tuple[int, MidiNote]]:
        pitch = abc_to_pitch(elem['name'], elem['alter'], elem['octave'])
        pitch += self.pitch_offset
        dur   = round(self.tpb * elem['dur'])
        note  = MidiNote(
            pitch          = max(0, min(127, pitch)),
            velocity       = self.velocity,
            duration_ticks = max(1, dur),
            channel        = self.channel,
        )
        return [(0, note)]

    # ──────────────────────────────────────────────────────────────────────────
    # Extracción de compases del MIDI original
    # ──────────────────────────────────────────────────────────────────────────

    def _bar_dur(self, bar_n: int) -> int:
        """Duración métrica en ticks del compás bar_n."""
        if self.bar_length_fn is not None:
            return self.bar_length_fn(bar_n)
        return self.tpb * 4   # fallback 4/4

    def _bars_dur(self, bar_from: int, bar_to: int) -> int:
        """Suma de duraciones métricas de bar_from..bar_to inclusive."""
        return sum(self._bar_dur(b) for b in range(bar_from, bar_to + 1))

    def _token_duration(self, token: str) -> int:
        """
        Duración total en ticks de un token de la raíz: rango, número de compás
        o nombre de regla. Siempre basada en duración métrica.
        """
        m_range = RANGE_RE.match(token)
        m_num   = NUMBER_RE.match(token)
        m_label = LABEL_RE.match(token)
        if m_range:
            lo, hi = int(m_range.group(1)), int(m_range.group(2))
            if lo > hi: lo, hi = hi, lo
            return self._bars_dur(lo, hi)
        if m_num:
            return self._bar_dur(int(token))
        if m_label:   # "[regla]" en la raíz: misma duración que "regla"
            return self._rule_duration(m_label.group(1))
        # Regla: sumar la duración de sus tokens constituyentes
        return self._rule_duration(token)

    def _elem_duration(self, elem: dict) -> int:
        """Duración en ticks de un elemento ya parseado de una regla."""
        return self._elem_duration_seen(elem, frozenset())

    def _rule_duration(self, label: str, _seen: frozenset = frozenset()) -> int:
        """Duración total de una regla, sumando sus elementos."""
        if label in _seen or label not in self.rules:
            return 0
        seen2 = _seen | {label}
        total = 0
        for elem in tokenize_rule(self.rules[label]):
            total += self._elem_duration_seen(elem, seen2)
        return total

    def _elem_duration_seen(self, elem: dict, seen: frozenset) -> int:
        if elem['type'] == 'bar':
            return self._bar_dur(elem['n'])
        if elem['type'] == 'range':
            return self._bars_dur(elem['m'], elem['n'])
        if elem['type'] == 'ref':
            return self._rule_duration(elem['label'], seen)
        if elem['type'] == 'note':
            if elem.get('sync'):
                return 0                    # simultánea con el siguiente elemento
            return self.tpb * elem['dur']   # Fraction exacta: sin deriva
        if elem['type'] == 'rest':
            return self.tpb * elem['dur']
        return 0

    def _get_bars(self, bar_from: int, bar_to: int) -> list[tuple[int, MidiNote]]:
        result: list[tuple[int, MidiNote]] = []
        cursor = 0

        for bar_n in range(bar_from, bar_to + 1):
            # Duración métrica del compás: siempre avanzamos exactamente esto,
            # independientemente de cuántas notas haya (o si el compás está vacío).
            if self.bar_length_fn is not None:
                bar_dur = self.bar_length_fn(bar_n)
            else:
                bar_dur = self.tpb * 4   # fallback 4/4

            if bar_n not in self.bars:
                msg = f"Compás {bar_n} no existe en el MIDI de entrada (o está vacío)."
                if msg not in self.report.warnings:
                    self.report.warnings.append(msg)
                    print(f"  [AVISO] {msg}", file=sys.stderr)
                cursor += bar_dur
                continue

            self.report.bars_used.append(bar_n)
            bar_notes = self.bars[bar_n]

            for (off, note) in bar_notes:
                result.append((cursor + off, self._transpose(note)))

            # Avanzar siempre por la duración métrica del compás, no por la
            # última nota. Esto garantiza que todos los tracks avanzan igual.
            cursor += bar_dur

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# ESCRITURA DEL MIDI DE SALIDA
# ═══════════════════════════════════════════════════════════════════════════════

def _notes_list_to_track(
    notes:   list[tuple[int, 'MidiNote']],
    track:   'MidiTrack',
    tempo:   int | None,
    name:    str | None = None,
) -> None:
    """
    Rellena un MidiTrack con los eventos note_on/note_off de `notes`.
    Si `tempo` no es None, inserta set_tempo al inicio (solo en el primer track).
    """
    if tempo is not None:
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))
    if name:
        track.append(MetaMessage('track_name', name=name, time=0))

    events: list[tuple[int, str, 'Message']] = []
    for (abs_off, note) in notes:
        on  = Message('note_on',  channel=note.channel,
                      note=note.pitch, velocity=note.velocity, time=0)
        off = Message('note_off', channel=note.channel,
                      note=note.pitch, velocity=0,             time=0)
        events.append((abs_off,                       'on',  on))
        events.append((abs_off + note.duration_ticks, 'off', off))

    events.sort(key=lambda x: (x[0], 0 if x[1] == 'off' else 1))

    prev_tick = 0
    for (abs_tick, _, msg) in events:
        msg.time  = abs_tick - prev_tick
        track.append(msg)
        prev_tick = abs_tick

    track.append(MetaMessage('end_of_track', time=0))


def notes_to_midi(
    notes:    list[tuple[int, 'MidiNote']],
    tpb:      int,
    tempo:    int,
    out_path: Path,
) -> None:
    """
    Escribe la lista de notas en un fichero MIDI tipo 0 (un único track).
    Usado cuando la expansión produce una lista plana de notas sin información
    de track de origen (p. ej. gramáticas manuales con notas ABC).
    """
    mid = MidiFile(type=0, ticks_per_beat=tpb)
    trk = MidiTrack()
    mid.tracks.append(trk)
    _notes_list_to_track(notes, trk, tempo=tempo, name='GrammarComposer')
    mid.save(str(out_path))


def multitrack_to_midi(
    tracks_notes: dict[int, list[tuple[int, 'MidiNote']]],
    tpb:          int,
    tempo:        int,
    out_path:     Path,
) -> None:
    """
    Escribe un MIDI tipo 1 con un track por cada entrada de `tracks_notes`.

    tracks_notes  — {track_idx_original: [(abs_tick, MidiNote), …]}

    El primer track (índice más bajo) recibe el meta-mensaje set_tempo.
    Los canales MIDI originales de cada nota se preservan tal cual.
    """
    mid = MidiFile(type=1, ticks_per_beat=tpb)

    for i, (orig_idx, notes) in enumerate(sorted(tracks_notes.items())):
        trk = MidiTrack()
        mid.tracks.append(trk)
        _notes_list_to_track(
            notes,
            trk,
            tempo = tempo if i == 0 else None,
            name  = f'Track{orig_idx}',
        )

    mid.save(str(out_path))


# ═══════════════════════════════════════════════════════════════════════════════
# INFORME
# ═══════════════════════════════════════════════════════════════════════════════

def print_report(report: ExpansionReport, grammar_path: Path, midi_path: Path) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║            GRAMMAR COMPOSER — Informe de expansión       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  MIDI de entrada : {midi_path}")
    print(f"  Gramática       : {grammar_path}")
    print()
    print(f"  Raíz            : {' '.join(report.root_sequence)}")
    print()
    print(f"  Reglas expandidas ({len(report.rules_expanded)}):")
    from collections import Counter
    cnt = Counter(f"{label}→{tipo}" for label, tipo in report.rules_expanded)
    for key, n in cnt.most_common():
        print(f"    {key:40s}  ×{n}")
    print()
    unique_bars = sorted(set(report.bars_used))
    print(f"  Compases del MIDI usados: {unique_bars}")
    print(f"  Notas ABC añadidas      : {report.abc_notes_added}")
    print(f"  Total notas en salida   : {report.total_notes}")
    if report.warnings:
        print()
        print(f"  Advertencias ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    ⚠  {w}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MODO LEARN — EXTRACCIÓN DE GRAMÁTICA DESDE MIDI (Re-Pair / Sequitur)
# ═══════════════════════════════════════════════════════════════════════════════

def bar_fingerprint(notes: list) -> tuple:
    """
    Genera una firma hashable del contenido de un compás para detect repeticiones
    exactas. Dos compases son idénticos si y solo si tienen exactamente las mismas
    notas con el mismo pitch, velocidad, duración y offset dentro del compás.

    No se normaliza ni el offset ni la velocidad: cualquier diferencia entre
    compases produce una firma distinta, garantizando que Re-Pair solo agrupe
    compases que son musicalmente bit-perfect idénticos.
    """
    if not notes:
        return ()
    # Ordenar por offset luego por pitch para orden canónico determinista
    return tuple(
        (off, n.pitch, n.velocity, n.duration_ticks)
        for off, n in sorted(notes, key=lambda x: (x[0], x[1].pitch))
    )


def bar_similarity(fp_a: tuple, fp_b: tuple) -> float:
    """
    Calcula la similitud entre dos firmas de compás como un valor en [0, 1].

    Algoritmo: para cada nota de fp_a busca la nota más cercana en fp_b
    que coincida en pitch y cuyo offset esté dentro de una ventana de
    tolerancia (por defecto ±1 semitono de pitch y offsets no comparados
    directamente — se usa solo pitch y duración relativa).

    Más concretamente: se representa cada firma como un multiset de
    (pitch, dur_cuantizado) y se calcula el ratio de intersección sobre
    la unión (Jaccard sobre multisets), que es 1.0 si son idénticas y 0.0
    si no tienen ninguna nota en común.

    La cuantización de duración agrupa notas cuya duración difiere en menos
    de un 20% (se redondea al múltiplo de 60 ticks más cercano), lo que
    tolera ligeras variaciones de swing o rubato.
    """
    def to_multiset(fp: tuple) -> dict:
        ms: dict = {}
        for item in fp:
            if len(item) == 4:
                off, pitch, vel, dur = item
            else:
                # firma combinada: ignorar track_idx exterior
                return {}
            # Cuantizar duración al múltiplo de 60 más cercano (≈semicorchea a 480tpb)
            q_dur = max(60, round(dur / 60) * 60)
            key = (pitch, q_dur)
            ms[key] = ms.get(key, 0) + 1
        return ms

    ms_a = to_multiset(fp_a)
    ms_b = to_multiset(fp_b)

    if not ms_a and not ms_b:
        return 1.0   # dos compases vacíos son idénticos
    if not ms_a or not ms_b:
        return 0.0

    # Intersección de multisets
    intersection = sum(min(ms_a.get(k, 0), ms_b.get(k, 0)) for k in ms_a)
    union        = sum(ms_a.values()) + sum(ms_b.values()) - intersection
    return intersection / union if union > 0 else 1.0


def _fuzzy_assign_tokens(
    combined_fps: list[tuple],
    threshold: float,
) -> list[int]:
    """
    Asigna tokens a una lista de firmas combinadas usando similitud fuzzy.

    Cada firma nueva se compara con todos los tokens canónicos ya vistos.
    Si la similitud con alguno supera `threshold`, se le asigna ese token.
    Si no, se crea un token nuevo.

    La similitud entre dos firmas combinadas (una por track) es el promedio
    de las similitudes por track.

    Devuelve la lista de tokens asignados (misma longitud que combined_fps).
    """
    canonical_fps: list[tuple] = []   # lista de firmas canónicas por token
    token_of: list[int] = []

    for fp in combined_fps:
        best_token = -1
        best_sim   = threshold - 1e-9   # tiene que superar el threshold

        for tok_idx, canon_fp in enumerate(canonical_fps):
            # fp y canon_fp son tuplas de (track_i, bar_fp_tuple)
            # Calcular similitud promedio por track
            tracks_a = {ti: bfp for ti, bfp in fp}
            tracks_b = {ti: bfp for ti, bfp in canon_fp}
            all_tracks = set(tracks_a) | set(tracks_b)
            if not all_tracks:
                sim = 1.0
            else:
                sims = []
                for ti in all_tracks:
                    bfp_a = tracks_a.get(ti, ())
                    bfp_b = tracks_b.get(ti, ())
                    sims.append(bar_similarity(bfp_a, bfp_b))
                sim = sum(sims) / len(sims)

            if sim > best_sim:
                best_sim   = sim
                best_token = tok_idx

        if best_token == -1:
            best_token = len(canonical_fps)
            canonical_fps.append(fp)

        token_of.append(best_token)

    return token_of


def extract_bar_sequence(
    mid: MidiFile,
    track_idx: int,
    fuzzy_threshold: float = 1.0,
) -> tuple[list[int], dict[int, int]]:
    """
    Construye la secuencia de tokens para Re-Pair y el mapa de traducción,
    considerando el contenido de TODOS los tracks con notas simultáneamente.

    Con fuzzy_threshold=1.0 (por defecto) dos compases son idénticos solo
    si son bit-perfect iguales en todos los tracks.

    Con fuzzy_threshold<1.0 (modo --fuzzy) dos compases se consideran
    equivalentes si su similitud promedio por track supera el threshold.
    La similitud se mide como Jaccard sobre multisets de (pitch, dur_cuantizado),
    lo que tolera variaciones de velocidad y pequeñas diferencias rítmicas.

    Devuelve:
        sequence      — lista de tokens (int), uno por compás
        token_to_bar  — {token: bar_number} para serializar la gramática
    """
    # Extraer compases de todos los tracks con notas
    all_bars: dict[int, dict[int, list]] = {}
    for i, track in enumerate(mid.tracks):
        has_notes = any(m.type == 'note_on' and m.velocity > 0 for m in track)
        if not has_notes:
            continue
        bars_i = extract_bars(mid, i)
        if bars_i:
            all_bars[i] = bars_i

    if not all_bars:
        return [], {}

    total_bars = max(max(b.keys()) for b in all_bars.values())

    # Construir lista de firmas combinadas, una por compás
    combined_fps: list[tuple] = []
    for bar_n in range(1, total_bars + 1):
        combined_fp = tuple(
            (ti, bar_fingerprint(bars_i.get(bar_n, [])))
            for ti, bars_i in sorted(all_bars.items())
        )
        combined_fps.append(combined_fp)

    # Asignar tokens
    if fuzzy_threshold >= 1.0:
        # Modo exacto: hash directo
        fingerprint_to_token: dict[tuple, int] = {}
        next_token = 0
        sequence: list[int] = []
        for fp in combined_fps:
            if fp not in fingerprint_to_token:
                fingerprint_to_token[fp] = next_token
                next_token += 1
            sequence.append(fingerprint_to_token[fp])
    else:
        # Modo fuzzy: comparación por similitud
        sequence = _fuzzy_assign_tokens(combined_fps, fuzzy_threshold)

    # token_to_bar: primer compás con cada token
    token_to_bar: dict[int, int] = {}
    for bar_idx, tok in enumerate(sequence):
        bar_n = bar_idx + 1
        if tok not in token_to_bar:
            token_to_bar[tok] = bar_n

    return sequence, token_to_bar


def _expand_grammar_rule_to_tokens(
    rule_tokens: list[str],
    bar_to_token: dict[int, int],
) -> list[int] | None:
    """
    Expande una regla de gramática (lista de tokens string como '1', '3-5', '[R1]')
    a una secuencia de tokens opacos de Re-Pair.

    Solo se pueden expandir reglas que contengan exclusivamente referencias a
    compases (números o rangos). Las referencias a otras reglas ([Rx]) y las
    notas ABC no son expandibles aquí porque necesitaríamos expandirlas
    recursivamente y aún no tenemos los tokens de esas reglas.

    Devuelve la lista de tokens int, o None si la regla contiene elementos
    no expandibles en este nivel.
    """
    import re
    RANGE_RE_ = re.compile(r'^(\d+)-(\d+)$')
    NUM_RE_   = re.compile(r'^\d+$')
    REF_RE_   = re.compile(r'^\[(\w+)\]$')

    result: list[int] = []
    for tok in rule_tokens:
        if NUM_RE_.match(tok):
            bar_n = int(tok)
            if bar_n not in bar_to_token:
                return None   # compás fuera del MIDI
            result.append(bar_to_token[bar_n])
        elif RANGE_RE_.match(tok):
            m = RANGE_RE_.match(tok)
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi: lo, hi = hi, lo
            for bar_n in range(lo, hi + 1):
                if bar_n not in bar_to_token:
                    return None
                result.append(bar_to_token[bar_n])
        elif REF_RE_.match(tok):
            return None   # referencia a otra regla: no expandible en este nivel
        else:
            return None   # nota ABC u otro: no aplicable
    return result if result else None


def apply_seed_grammar(
    sequence:          list[int],
    seed_rules:        dict[str, list[str]],
    token_to_bar:      dict[int, int],
    bar_to_token_full: dict[int, int] | None = None,
    verbose:           bool = False,
) -> tuple[list, dict[str, list], int]:
    """
    Aplica una gramática semilla (seed) a la secuencia de tokens antes de
    ejecutar Re-Pair, de la misma forma que SequiturWithCustomRules en mscz2vec.

    Proceso:
      1. Para cada regla de la semilla, expandir su cuerpo a tokens opacos.
         Solo se procesan reglas cuyos cuerpos son completamente expandibles
         (solo números de compás y rangos, sin referencias a otras reglas).
         Las reglas recursivas (que referencian otras reglas semilla) se resuelven
         en un segundo paso tras expandir las hojas.
      2. Sustituir en la secuencia las ocurrencias de cada patrón expandido
         por el nombre de la regla semilla (las más largas primero).
      3. Devolver la secuencia modificada, el dict de reglas semilla expandidas
         (en tokens opacos), y el primer ID numérico libre para Re-Pair.

    Las reglas semilla se preservan con sus nombres originales; Re-Pair usará
    identificadores R{N} con N empezando desde el primer entero libre tras los
    nombres de las reglas semilla (para evitar colisiones).

    Parámetros:
        sequence           — lista de tokens opacos (salida de extract_bar_sequence)
        seed_rules         — {nombre: [token_str, …]} tal como lo devuelve parse_grammar
        token_to_bar       — {token_opaco: bar_number} (primer compás de cada token)
        bar_to_token_full  — {bar_number: token_opaco} para TODOS los compases;
                             si None se construye solo desde token_to_bar (incompleto
                             para rangos con compases no-canónicos)
        verbose            — traza el proceso

    Devuelve:
        modified_seq      — secuencia con reglas semilla aplicadas
        expanded_rules    — {nombre: [token_opaco, …]} para reglas expandidas
        next_repaint_id   — primer entero N para que Re-Pair use R{N}, R{N+1}, …
    """
    # Mapa bar_number → token_opaco para todos los compases
    if bar_to_token_full is not None:
        bar_to_token = bar_to_token_full
    else:
        bar_to_token = {bar: tok for tok, bar in token_to_bar.items()}

    # ── Paso 1: expandir reglas de la semilla en dos sub-pasos:
    #   a) Hojas: reglas con solo números/rangos → lista de tokens opacos
    #   b) Recursivas: reglas con referencias a otras reglas semilla →
    #      lista mixta [token_opaco | str_nombre_regla], preservando la
    #      referencia para que la serialización final muestre [NombreRegla]
    #      en lugar de expandir todo a números de compás.
    import re
    REF_RE_ = re.compile(r'^\[(\w+)\]$')

    expanded: dict[str, list] = {}   # nombre → [tokens_opacos y/o str refs]
    unresolved = dict(seed_rules)

    max_passes = len(seed_rules) + 1
    for _ in range(max_passes):
        if not unresolved:
            break
        progress = False
        for name, body_tokens in list(unresolved.items()):
            resolved_body: list = []
            ok = True
            for tok in body_tokens:
                m = REF_RE_.match(tok)
                if m:
                    ref_name = m.group(1)
                    if ref_name in expanded:
                        # Para la sustitución en la secuencia necesitamos
                        # los tokens opacos; guardamos la referencia por nombre
                        # pero también necesitamos poder sustituir en la secuencia.
                        # Solución: resolved_body lleva tokens opacos para
                        # la parte expandida, pero el cuerpo ALMACENADO en
                        # expanded preserva la referencia como string.
                        resolved_body.append(('ref', ref_name, expanded[ref_name]))
                    else:
                        ok = False
                        break
                else:
                    resolved_body.append(('tok', tok))

            if not ok:
                continue

            # Construir versión plana (solo ints) para sustitución en secuencia
            flat: list[int] = []
            stored: list = []   # mixta: ints + str refs para serialización
            for item in resolved_body:
                if item[0] == 'ref':
                    _, ref_name, ref_tokens = item
                    # Para sustituir en la secuencia: tokens planos
                    for t in ref_tokens:
                        if isinstance(t, int):
                            flat.append(t)
                        # str en ref_tokens (ref encadenada): expandir recursivamente
                        # (esto es raro y no lo manejamos en profundidad)
                    # Para almacenar: nombre de referencia como string
                    stored.append(ref_name)
                else:
                    _, tok_str = item
                    bar_ns: list[str] = []
                    import re as re2
                    RANGE2 = re2.compile(r'(\d+)-(\d+)')
                    NUM2   = re2.compile(r'\d+')
                    rm = RANGE2.match(tok_str)
                    if rm:
                        lo, hi = int(rm.group(1)), int(rm.group(2))
                        if lo > hi: lo, hi = hi, lo
                        for bn in range(lo, hi+1):
                            if bn not in bar_to_token:
                                flat = None; break
                            flat.append(bar_to_token[bn]) if flat is not None else None
                            stored.append(bar_to_token[bn])
                    elif NUM2.fullmatch(tok_str):
                        bn = int(tok_str)
                        if bn not in bar_to_token:
                            flat = None
                        else:
                            if flat is not None: flat.append(bar_to_token[bn])
                            stored.append(bar_to_token[bn])
                    else:
                        flat = None  # nota ABC u otro: no expandible
                    if flat is None:
                        break

            if flat is None or not flat:
                continue

            # flat = tokens opacos para sustitución en secuencia
            # stored = lista mixta (ints + str refs) para preservar semántica
            expanded[name] = flat          # usado para sustitución
            expanded[name + '.__stored__'] = stored  # usado para serialización
            del unresolved[name]
            progress = True
            if verbose:
                bars = [token_to_bar.get(t, t) if isinstance(t, int) else t
                        for t in stored]
                print(f"  [seed] regla '{name}' → {bars}")

        if not progress:
            break

    if unresolved and verbose:
        print(f"  [seed] reglas no expandibles (se ignorarán): {list(unresolved.keys())}")

    # Separar reglas reales de metadatos .__stored__
    real_rules  = {k: v for k, v in expanded.items() if not k.endswith('.__stored__')}
    stored_map  = {k[:-len('.__stored__')]: v
                   for k, v in expanded.items() if k.endswith('.__stored__')}

    if not real_rules:
        return list(sequence), {}, 1

    # ── Paso 2: sustituir en la secuencia (más largas primero)
    current_seq = list(sequence)
    sorted_rules = sorted(real_rules.items(), key=lambda x: len(x[1]), reverse=True)

    for rule_name, pattern in sorted_rules:
        if not pattern:
            continue
        plen = len(pattern)
        new_seq: list = []
        i = 0
        while i < len(current_seq):
            if (i + plen <= len(current_seq)
                    and current_seq[i:i+plen] == pattern):
                new_seq.append(rule_name)
                i += plen
            else:
                new_seq.append(current_seq[i])
                i += 1
        current_seq = new_seq
        if verbose:
            n_subs = sum(1 for t in current_seq if t == rule_name)
            print(f"  [seed] '{rule_name}' sustituido {n_subs} veces en la secuencia")

    # ── Paso 3: calcular el primer ID libre para Re-Pair
    import re
    numeric_ids = []
    for name in real_rules:
        m = re.match(r'^R(\d+)$', name)
        if m:
            numeric_ids.append(int(m.group(1)))
    next_id = max(numeric_ids, default=0) + 1

    # ── Paso 4: construir dict de reglas para Re-Pair usando stored_map
    # donde está disponible (preserva referencias a otras reglas semilla)
    rules_for_repaint = {}
    for name, flat_tokens in real_rules.items():
        stored = stored_map.get(name)
        rules_for_repaint[name] = stored if stored is not None else flat_tokens

    return current_seq, rules_for_repaint, next_id


def repaint_learn(
    sequence:     list,
    min_pair:     int = 2,
    max_rules:    int | None = None,
    seed_rules:   dict[str, list] | None = None,
    first_rule_id: int = 1,
    verbose:      bool = False,
) -> tuple[dict[str, list], list]:
    """
    Algoritmo Re-Pair sobre la secuencia de símbolos (tokens opacos o referencias
    a reglas). Las reglas semilla ya aplicadas se pasan en seed_rules para que
    queden incluidas en el resultado final.

    Parámetros:
        sequence      — secuencia ya con sustituciones semilla aplicadas
        min_pair      — frecuencia mínima para crear una nueva regla
        max_rules     — límite total de reglas (incluyendo semilla)
        seed_rules    — {nombre: [tokens]} reglas semilla ya aplicadas
        first_rule_id — primer N para generar R{N} (evita colisiones con semilla)
        verbose       — traza

    Devuelve:
        rules   — seed_rules ∪ reglas_aprendidas
        root    — secuencia raíz tras todas las sustituciones
    """
    from collections import Counter

    rules: dict[str, list] = dict(seed_rules) if seed_rules else {}
    next_id = [first_rule_id]

    def new_name() -> str:
        # Saltar nombres que ya existen (por si seed_rules usa R{N})
        while f"R{next_id[0]}" in rules:
            next_id[0] += 1
        name = f"R{next_id[0]}"
        next_id[0] += 1
        return name

    current = list(sequence)

    while True:
        # Límite de reglas
        if max_rules is not None and len(rules) >= max_rules:
            break

        # Contar pares adyacentes
        pairs: Counter = Counter()
        for i in range(len(current) - 1):
            pairs[(current[i], current[i + 1])] += 1

        if not pairs:
            break

        best_pair, count = pairs.most_common(1)[0]
        if count < min_pair:
            break

        # Crear nueva regla
        rule_name = new_name()
        rules[rule_name] = list(best_pair)

        if verbose:
            print(f"  [learn] nueva regla {rule_name} = {list(best_pair)}  (×{count})")

        # Sustituir todas las ocurrencias del par en la secuencia
        new_seq: list = []
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


def _compress_ranges(tokens: list) -> list[str]:
    """
    Comprime secuencias de enteros estrictamente ascendentes y consecutivos
    en rangos "M-N". Las referencias a reglas se envuelven en [corchetes].

    IMPORTANTE: solo se comprime un tramo si todos sus valores son distintos
    y cada uno es exactamente el anterior + 1. En cuanto hay una repetición
    o un salto, el tramo se corta.

    Ejemplos:
        [1, 2, 3, 'R1', 4]  →  ['1-3', '[R1]', '4']
        [1, 2, 1, 2]        →  ['1', '2', '1', '2']   (no comprimir: hay repetición)
        [1, 2, 3, 5, 6]     →  ['1-3', '5-6']          (dos tramos separados)
    """
    result: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(tok, str):
            result.append(f"[{tok}]")
            i += 1
        else:
            # Intentar extender tramo consecutivo ascendente
            j = i + 1
            while (j < len(tokens)
                   and isinstance(tokens[j], int)
                   and tokens[j] == tokens[j - 1] + 1):
                j += 1
            span = tokens[i:j]
            if len(span) == 1:
                result.append(str(span[0]))
            else:
                result.append(f"{span[0]}-{span[-1]}")
            i = j
    return result


def grammar_to_text(rules: dict[str, list], root: list) -> str:
    """
    Serializa la gramática aprendida al formato esperado por el modo compose:

        R1 R1 R2 R2 R1            ← raíz: etiquetas sin corchetes / rangos de compás
        R1 : 1-2                  ← reglas hoja: sólo números de compás
        R2 : 3-4
        R3 : [R1] [R2]            ← reglas internas: referencias con corchetes

    IMPORTANTE: en la raíz los nombres de regla se escriben sin corchetes
    (son etiquetas, no tokens de cuerpo); sólo en el cuerpo de las reglas
    las referencias llevan corchetes.
    """
    lines: list[str] = []

    # ── Raíz: nombres de regla directos (sin corchetes); enteros consecutivos
    # únicos como rangos. Usamos _compress_ranges y quitamos los corchetes
    # de los nombres de regla (en la raíz van sin corchetes).
    raw_root_tokens = _compress_ranges(root)
    root_parts = [t[1:-1] if (t.startswith('[') and t.endswith(']')) else t
                  for t in raw_root_tokens]
    lines.append(" ".join(root_parts))

    # ── Ordenar reglas: hojas primero, luego nodos internos, por ID numérico
    def rule_sort_key(name: str) -> tuple:
        body = rules[name]
        has_refs = any(isinstance(s, str) for s in body)
        num = int(name[1:]) if name[1:].isdigit() else 0
        return (has_refs, num)

    for name in sorted(rules.keys(), key=rule_sort_key):
        body_tokens = _compress_ranges(rules[name])   # refs llevan corchetes aquí
        lines.append(f"{name} : {' '.join(body_tokens)}")

    return "\n".join(lines)


def _translate_tokens(
    obj: list,
    token_to_bar: dict[int, int],
) -> list:
    """
    Sustituye tokens opacos (ints) por números de compás reales en una lista
    que puede contener ints y strings (nombres de regla). Recursivo en
    el sentido de que opera sobre listas planas; las reglas se traducen
    una a una desde learn_grammar.
    """
    return [token_to_bar[t] if isinstance(t, int) else t for t in obj]


def learn_grammar(
    mid:              MidiFile,
    track_idx:        int,
    min_pair:         int   = 2,
    max_rules:        int | None = None,
    merge_bars:       bool  = False,
    fuzzy_threshold:  float = 1.0,
    seed_grammar:     str | None = None,
    verbose:          bool  = False,
) -> tuple[str, dict, list]:
    """
    Pipeline completo de aprendizaje:
        1. Extrae la secuencia de tokens por contenido + mapa token→compás.
        2. (opcional) Aplica gramática semilla: sustituye patrones conocidos.
        3. Re-Pair sobre la secuencia resultante.
        4. Traduce tokens opacos a números de compás fuente.
        5. Serializa al formato de texto compatible con el modo compose.

    Con seed_grammar (texto en el mismo formato que una gramática normal) se
    definen reglas iniciales que Re-Pair respetará. Las reglas de la semilla
    se sustituyen en la secuencia antes de que Re-Pair arranque, de modo que
    Re-Pair solo aprende la estructura que la semilla no cubre. La raíz de la
    gramática semilla se ignora — solo se usan sus definiciones de reglas.

    Con fuzzy_threshold < 1.0 se activa la comparación difusa de compases.

    Devuelve (texto_gramática, rules_dict_traducido, root_list_traducida).
    """
    fuzzy_str = f", fuzzy={fuzzy_threshold:.2f}" if fuzzy_threshold < 1.0 else ""
    print(f"  Extrayendo secuencia de compases (todos los tracks{fuzzy_str}) …")
    seq, token_to_bar = extract_bar_sequence(mid, track_idx,
                                             fuzzy_threshold=fuzzy_threshold)
    if not seq:
        raise ValueError("El track no contiene notas. Comprueba el índice de track.")

    n_unique = len(token_to_bar)
    print(f"  → {len(seq)} compases, {n_unique} tipos únicos de contenido")

    # ── Aplicar gramática semilla (si se proporcionó)
    seed_rules_tok: dict[str, list] = {}
    first_id = 1
    working_seq = seq

    if seed_grammar:
        print(f"  Aplicando gramática semilla …")
        try:
            _seed_root, seed_rules_str = parse_grammar(seed_grammar)
        except ValueError as e:
            raise ValueError(f"Error en la gramática semilla: {e}") from e

        # Construir mapa completo bar_n → token_opaco desde la secuencia
        # (token_to_bar solo tiene el primer compás de cada tipo)
        bar_to_token_full = {bar_n: seq[bar_n - 1]
                             for bar_n in range(1, len(seq) + 1)}

        working_seq, seed_rules_tok, first_id = apply_seed_grammar(
            seq, seed_rules_str, token_to_bar,
            bar_to_token_full=bar_to_token_full,
            verbose=verbose,
        )
        n_seed = len(seed_rules_tok)
        n_subs = sum(1 for t in working_seq if isinstance(t, str))
        print(f"  → {n_seed} reglas semilla aplicadas, "
              f"{n_subs} sustituciones en la secuencia")

    print(f"  Aplicando Re-Pair (min_pair={min_pair}"
          + (f", max_rules={max_rules}" if max_rules else "")
          + (f", seed={len(seed_rules_tok)} reglas" if seed_rules_tok else "")
          + ") …")
    rules_tok, root_tok = repaint_learn(
        working_seq,
        min_pair      = min_pair,
        max_rules     = max_rules,
        seed_rules    = seed_rules_tok,
        first_rule_id = first_id,
        verbose       = verbose,
    )
    print(f"  → {len(rules_tok)} reglas totales, raíz de {len(root_tok)} símbolos")

    # Traducir tokens opacos → números de compás reales
    # Las reglas semilla ya están en tokens opacos; las de Re-Pair también.
    rules_bar = {name: _translate_tokens(body, token_to_bar)
                 for name, body in rules_tok.items()}
    root_bar  = _translate_tokens(root_tok, token_to_bar)

    text = grammar_to_text(rules_bar, root_bar)
    return text, rules_bar, root_bar


def print_learn_report(rules: dict, root: list, out_path: Path) -> None:
    """Imprime un resumen del proceso de aprendizaje."""
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          GRAMMAR COMPOSER — Informe modo learn            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Fichero de gramática : {out_path}")
    print()
    print(f"  Reglas generadas     : {len(rules)}")

    # Estadísticas de las reglas
    leaf_rules  = {k: v for k, v in rules.items()
                   if all(isinstance(s, int) for s in v)}
    inner_rules = {k: v for k, v in rules.items()
                   if any(isinstance(s, str) for s in v)}

    print(f"    · Reglas hoja (solo compases)     : {len(leaf_rules)}")
    print(f"    · Reglas internas (con sub-reglas): {len(inner_rules)}")
    print()
    print(f"  Raíz ({len(root)} símbolos):")

    # Mostrar raíz con rangos comprimidos
    root_tokens = _compress_ranges(root)
    # Agrupar en líneas de ~70 caracteres
    line, lines = "", []
    for tok in root_tokens:
        if len(line) + len(tok) + 1 > 70:
            lines.append(line)
            line = tok
        else:
            line = (line + " " + tok).lstrip()
    if line:
        lines.append(line)
    for l in lines:
        print(f"    {l}")
    print()

    # Mostrar las primeras reglas
    MAX_SHOW = 10
    print(f"  Primeras {min(MAX_SHOW, len(rules))} reglas:")
    for i, (name, body) in enumerate(rules.items()):
        if i >= MAX_SHOW:
            print(f"    … ({len(rules) - MAX_SHOW} más)")
            break
        body_str = " ".join(_compress_ranges(body))
        print(f"    {name:6s} : {body_str}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MODO LEARN-GESTURES — CODEBOOK DE GESTOS INVARIANTES A LA TRANSPOSICIÓN
# ═══════════════════════════════════════════════════════════════════════════════
#
# Idea: el MIDI se trocea en "cortes" (todas las notas que empiezan a la vez).
# Cada corte es (forma, avance): la forma son las alturas relativas a su nota
# más grave y las duraciones; el avance es el tiempo hasta el siguiente corte
# (así los silencios quedan incluidos). Dos cortes con la misma forma son el
# mismo símbolo aunque estén a distinta altura.
#
# Sobre esa secuencia se aplica Re-Pair, pero un par adyacente solo cuenta como
# "el mismo" si además el intervalo entre sus notas graves es idéntico. Cada
# regla creada es un GESTO: un grupo de cortes (ritmo + intervalos) que aparece
# varias veces a alturas distintas. Las reglas usadas una sola vez se
# reabsorben (no son un gesto repetido).
#
# La gramática resultante es autocontenida (solo notas ABC): el cuerpo de cada
# gesto se escribe a su altura canónica (la de su primera aparición) y la regla
# raíz `song` lo coloca con el offset global:  +7 [g1] -3 [g1] 0 [g2] …

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def pitch_to_abc_name(pitch: int) -> str:
    """Número MIDI → nombre ABC con octava (60 → C4, 0 → C-1)."""
    return f"{PITCH_NAMES[pitch % 12]}{pitch // 12 - 1}"


def extract_note_events(
    mid:            MidiFile,
    track_idx:      int,
    quantize_ticks: int | None = None,
) -> tuple[list[tuple[int, int, int]], list[int], int]:
    """
    Devuelve ([(onset, pitch, dur_ticks), …] ordenadas, canales usados,
    velocidad media). Con quantize_ticks, onsets y finales se redondean a esa
    rejilla (la duración mínima es un paso de rejilla).
    """
    abs_t  = 0
    active: dict[tuple[int, int], list[tuple[int, int]]] = {}
    events: list[tuple[int, int, int]] = []
    channels: set[int] = set()
    vels: list[int] = []

    for msg in mid.tracks[track_idx]:
        abs_t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            active.setdefault((msg.channel, msg.note), []).append((abs_t, msg.velocity))
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if active.get(key):
                start, vel = active[key].pop(0)
                events.append((start, msg.note, max(1, abs_t - start)))
                channels.add(msg.channel)
                vels.append(vel)

    if quantize_ticks and quantize_ticks > 1:
        g = quantize_ticks
        quantized = []
        for on, pitch, dur in events:
            on_q  = round(on / g) * g
            end_q = round((on + dur) / g) * g
            quantized.append((on_q, pitch, max(g, end_q - on_q)))
        events = quantized

    events.sort()
    mean_vel = round(sum(vels) / len(vels)) if vels else 80
    return events, sorted(channels), mean_vel


def build_gesture_sequence(events: list[tuple[int, int, int]]):
    """
    Convierte las notas en la secuencia de cortes.

    Devuelve (terms, ids, refs, first_onset):
        terms[tid] = {'shape': ((dp, dur), …), 'adv': ticks, 'rho': altura_ref}
        ids[k]     = tipo de terminal del corte k
        refs[k]    = altura de su nota más grave (el "ancla" del corte)
    """
    by_onset: dict[int, list[tuple[int, int]]] = {}
    for on, pitch, dur in events:
        by_onset.setdefault(on, []).append((pitch, dur))
    onsets = sorted(by_onset)

    term_ids: dict[tuple, int] = {}
    terms: list[dict] = []
    ids:   list[int]  = []
    refs:  list[int]  = []
    for k, on in enumerate(onsets):
        notes = sorted(by_onset[on])
        ref   = notes[0][0]
        shape = tuple((p - ref, d) for p, d in notes)
        adv   = (onsets[k + 1] - on) if k + 1 < len(onsets) else max(d for _, d in notes)
        key   = (shape, adv)
        tid   = term_ids.get(key)
        if tid is None:
            tid = len(terms)
            term_ids[key] = tid
            terms.append({'shape': shape, 'adv': adv, 'rho': ref})
        ids.append(tid)
        refs.append(ref)
    return terms, ids, refs, onsets[0]


def gesture_repair(
    ids:       list[int],
    refs:      list[int],
    rho_of,
    first_id:  int,
    min_pair:  int = 2,
    max_rules: int | None = None,
    verbose:   bool = False,
) -> tuple[dict[int, dict], list[int], list[int]]:
    """
    Re-Pair invariante a la transposición. Un par (A, B) es "el mismo" si
    coinciden los dos símbolos y el intervalo entre sus anclas (refs[i+1]-refs[i]).

    Cada regla nueva: {'body': [(hijo, ref_en_el_marco_de_la_regla), …],
                       'rho': ancla canónica de la regla}.
    El marco de una regla es el de su primera aparición: su primer hijo está a
    su altura canónica y el resto a ancla + intervalo.
    """
    nts: dict[int, dict] = {}
    next_id = first_id

    while max_rules is None or len(nts) < max_rules:
        n = len(ids)
        counts: dict[tuple, int] = {}
        last:   dict[tuple, int] = {}
        for i in range(n - 1):
            key = (ids[i], ids[i + 1], refs[i + 1] - refs[i])
            if i - last.get(key, -2) >= 2:      # no solapar (A A A → un solo par)
                counts[key] = counts.get(key, 0) + 1
                last[key]   = i
        if not counts:
            break
        best = max(counts, key=counts.get)
        if counts[best] < min_pair:
            break

        a, b, delta = best
        rho_a = rho_of(a, nts)
        nts[next_id] = {'body': [(a, rho_a), (b, rho_a + delta)], 'rho': rho_a}
        if verbose:
            print(f"  [gesto #{next_id}] ×{counts[best]}  ({a}, {b}, Δ{delta:+d})")

        new_ids:  list[int] = []
        new_refs: list[int] = []
        i = 0
        while i < n:
            if (i < n - 1 and ids[i] == a and ids[i + 1] == b
                    and refs[i + 1] - refs[i] == delta):
                new_ids.append(next_id)
                new_refs.append(refs[i])
                i += 2
            else:
                new_ids.append(ids[i])
                new_refs.append(refs[i])
                i += 1
        ids, refs = new_ids, new_refs
        next_id  += 1

    return nts, ids, refs


def _finalize_gestures(
    nts:  dict[int, dict],
    ids:  list[int],
    refs: list[int],
) -> tuple[dict[int, dict], list[tuple[int, int]]]:
    """
    Reabsorbe las reglas usadas una sola vez (no son gestos repetidos) y
    aplana sus cuerpos. Devuelve (gestos_conservados, secuencia_raíz).
    El aplanado es iterativo: Re-Pair genera cadenas de cientos de niveles.
    """
    usage: dict[int, int] = {}
    for i in ids:
        if i in nts:
            usage[i] = usage.get(i, 0) + 1
    for nt in nts.values():
        for child, _ in nt['body']:
            if child in nts:
                usage[child] = usage.get(child, 0) + 1
    keep = {nid for nid in nts if usage.get(nid, 0) >= 2}

    def flat(items: list[tuple[int, int]], delta: int) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        stack = [(iter(items), delta)]
        while stack:
            it, d = stack[-1]
            for (i, r) in it:
                r2 = r + d
                if i in nts and i not in keep:
                    stack.append((iter(nts[i]['body']), r2 - nts[i]['rho']))
                    break
                out.append((i, r2))
            else:
                stack.pop()
        return out

    kept = {nid: {'body': flat(nts[nid]['body'], 0), 'rho': nts[nid]['rho']}
            for nid in sorted(keep)}
    root = flat(list(zip(ids, refs)), 0)
    return kept, root


def _slice_tokens(
    written: list[int],
    shape:   tuple,
    adv:     int,
    tpb:     int,
) -> list[str]:
    """
    Tokens ABC de un corte. Las notas llevan '&' (simultáneas con la siguiente);
    la última avanza el cursor. Si ninguna nota dura exactamente `adv`, el
    avance lo pone un silencio  z:adv.
    """
    notes   = [(written[k], shape[k][1]) for k in range(len(shape))]
    carrier = None
    for k in range(len(notes) - 1, -1, -1):
        if notes[k][1] == adv:
            carrier = k
            break
    toks = [f"{pitch_to_abc_name(p)}:{Fraction(d, tpb)}&"
            for k, (p, d) in enumerate(notes) if k != carrier]
    if carrier is not None:
        p, d = notes[carrier]
        toks.append(f"{pitch_to_abc_name(p)}:{Fraction(d, tpb)}")
    else:
        toks.append(f"z:{Fraction(adv, tpb)}")
    return toks


def _emit_gesture_sequence(
    items:    list[tuple[int, int]],
    terms:    list[dict],
    nt_info:  dict[int, dict],
    tpb:      int,
    absolute: bool,
) -> tuple[list[str], int]:
    """
    Serializa una secuencia [(símbolo, ref)] siguiendo el offset global.
    absolute=True (raíz): se permite el token "0". En el cuerpo de un gesto el
    offset de entrada es desconocido, así que solo se usan desplazamientos
    relativos. Devuelve (tokens, desplazamiento_neto_al_acabar).
    """
    n_terms = len(terms)
    tokens: list[str] = []
    c = 0

    def shift_to(target: int) -> None:
        nonlocal c
        if target == c:
            return
        tokens.append(RESET_TOKEN if (absolute and target == 0) else f"{target - c:+d}")
        c = target

    for (sym, ref) in items:
        if sym < n_terms:
            t       = terms[sym]
            shape   = t['shape']
            written = [ref + dp - c for dp, _ in shape]
            if min(written) < 0 or max(written) > 127:
                # fuera del rango escribible: ir a la altura canónica del corte
                shift_to(ref - t['rho'])
                written = [t['rho'] + dp for dp, _ in shape]
            tokens.extend(_slice_tokens(written, shape, t['adv'], tpb))
        else:
            info = nt_info[sym]
            shift_to(ref - info['rho'])
            tokens.append(f"[{info['name']}]")
            c += info['net']
    return tokens, c


def verify_gesture_grammar(
    grammar_text: str,
    events:       list[tuple[int, int, int]],
    tpb:          int,
) -> tuple[bool, int, int, list[str]]:
    """Expande la gramática y la compara con las notas originales (onset, altura, duración)."""
    root, rules = parse_grammar(grammar_text)
    expander = GrammarExpander(rules=rules, bars={}, tpb=tpb)
    got = sorted((on, n.pitch, n.duration_ticks) for on, n in expander.expand_root(root))
    want = sorted(events)
    return got == want, len(got), len(want), list(expander.report.warnings)


def learn_gestures(
    mid:       MidiFile,
    track_idx: int,
    min_pair:  int = 2,
    max_rules: int | None = None,
    quantize:  Fraction | None = None,
    verbose:   bool = False,
) -> tuple[str, dict]:
    """
    Aprende el codebook de gestos de un track y devuelve (texto_gramática, stats).
    """
    tpb = mid.ticks_per_beat
    q_ticks = None
    if quantize is not None:
        q_ticks = round(tpb * quantize)
        if q_ticks < 1:
            raise ValueError(f"--quantize {quantize} es menor que 1 tick (tpb={tpb}).")

    events, channels, mean_vel = extract_note_events(mid, track_idx, q_ticks)
    if not events:
        raise ValueError("El track no contiene notas. Comprueba el índice de track.")

    terms, ids, refs, first_onset = build_gesture_sequence(events)
    n_slices = len(ids)
    n_terms  = len(terms)
    print(f"  → {len(events)} notas, {n_slices} cortes, {n_terms} formas distintas")

    def rho_of(sym: int, nts: dict) -> int:
        return terms[sym]['rho'] if sym < n_terms else nts[sym]['rho']

    print(f"  Aplicando Re-Pair invariante a la transposición (min_pair={min_pair}"
          + (f", max_rules={max_rules}" if max_rules else "") + ") …")
    nts, rids, rrefs = gesture_repair(ids, refs, rho_of, first_id=n_terms,
                                      min_pair=min_pair, max_rules=max_rules,
                                      verbose=verbose)
    kept, root_items = _finalize_gestures(nts, rids, rrefs)
    print(f"  → {len(nts)} reglas de Re-Pair, {len(kept)} gestos tras reabsorber "
          f"los de un solo uso")

    # ── Nombres y serialización (hijos antes que padres: ids ascendentes)
    nt_info: dict[int, dict] = {}
    for k, nid in enumerate(sorted(kept), start=1):
        nt_info[nid] = {'name': f"g{k}", 'rho': kept[nid]['body'][0][1], 'net': 0}
    body_tokens: dict[int, list[str]] = {}
    for nid in sorted(kept):
        toks, net = _emit_gesture_sequence(kept[nid]['body'], terms, nt_info, tpb, absolute=False)
        body_tokens[nid]       = toks
        nt_info[nid]['net']    = net

    song, _ = _emit_gesture_sequence(root_items, terms, nt_info, tpb, absolute=True)
    if first_onset > 0:
        song.insert(0, f"z:{Fraction(first_onset, tpb)}")

    # ── Estadísticas por gesto
    n_slices_of: dict[int, int] = {}
    for nid in sorted(kept):
        n_slices_of[nid] = sum(1 if s < n_terms else n_slices_of[s] for s, _ in kept[nid]['body'])
    uses: dict[int, int] = {nid: 0 for nid in kept}
    for s, _ in root_items:
        if s in kept:
            uses[s] += 1
    for nid in sorted(kept, reverse=True):
        for s, _ in kept[nid]['body']:
            if s in kept:
                uses[s] += uses[nid]
    covered = sum(n_slices_of[s] for s, _ in root_items if s in kept)

    naive_tokens = sum(len(_slice_tokens([0] * len(terms[s]['shape']), terms[s]['shape'],
                                         terms[s]['adv'], tpb)) for s in ids)
    grammar_tokens = len(song) + sum(len(t) for t in body_tokens.values())

    # ── Texto de la gramática
    lines = [
        "# Gramática de gestos — grammar_composer --learn-gestures",
        f"# Fuente: track {track_idx}, tpb={tpb}, {len(events)} notas, {n_slices} cortes"
        + (f", cuantizado a {quantize} negra" if quantize is not None else ""),
        "# Cada gesto gN es un grupo de notas (con su ritmo) repetido a distintas alturas;",
        "# la regla `song` los coloca con el offset global (+n / -n / 0).",
        "# Las notas con '&' suenan a la vez que el siguiente elemento; z:d es un silencio.",
        f"# Recomponer: python grammar_composer.py <original.mid> <esta gramática>"
        f" --velocity {mean_vel}"
        + (f" --channel {channels[0]}" if len(channels) == 1 and channels[0] != 0 else ""),
        "song",
    ]
    for nid in sorted(kept):
        info = nt_info[nid]
        lines.append(f"# {info['name']}: {n_slices_of[nid]} cortes, {uses[nid]} apariciones")
        lines.append(f"{info['name']} : {' '.join(body_tokens[nid])}")
    lines.append(f"song : {' '.join(song)}")
    text = "\n".join(lines) + "\n"

    ok, n_got, n_want, warns = verify_gesture_grammar(text, events, tpb)

    stats = {
        'notes': len(events), 'slices': n_slices, 'terminal_shapes': n_terms,
        'gestures': len(kept), 'covered': covered,
        'naive_tokens': naive_tokens, 'grammar_tokens': grammar_tokens,
        'roundtrip_ok': ok, 'notes_out': n_got, 'warnings': warns,
        'info': {nid: {'name': nt_info[nid]['name'], 'slices': n_slices_of[nid],
                       'uses': uses[nid]} for nid in kept},
        'channels': channels, 'mean_vel': mean_vel,
    }
    return text, stats


def print_gesture_report(stats: dict, out_path: Path) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        GRAMMAR COMPOSER — Informe learn-gestures          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Fichero de gramática : {out_path}")
    print(f"  Notas / cortes       : {stats['notes']} / {stats['slices']}")
    print(f"  Formas de corte      : {stats['terminal_shapes']} distintas "
          f"(invariantes a la transposición)")
    print(f"  Gestos aprendidos    : {stats['gestures']}")
    pct = 100 * stats['covered'] / max(1, stats['slices'])
    print(f"  Cortes dentro de un gesto repetido : {stats['covered']} ({pct:.0f} %)")
    print(f"  Tokens: {stats['naive_tokens']} (notas planas) → "
          f"{stats['grammar_tokens']} (gramática)")
    top = sorted(stats['info'].values(), key=lambda g: -(g['slices'] * g['uses']))[:10]
    if top:
        print("  Gestos que más cubren:")
        for g in top:
            print(f"    {g['name']:6s} {g['slices']:3d} cortes × {g['uses']} apariciones")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='grammar_composer',
        description=(
            'Genera MIDI a partir de una gramática formal y un MIDI de entrada '
            '(modo compose), o extrae la gramática de un MIDI (modo learn).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Modo learn (flag)
    p.add_argument('--learn', action='store_true',
                   help='Activar modo learn: extrae la gramática del MIDI dado')
    p.add_argument('--learn-gestures', action='store_true',
                   help=('Activar modo learn-gestures: aprende un codebook de gestos '
                         '(grupos de notas repetidos a distinta altura) y escribe una '
                         'gramática autocontenida que los recompone con +n/-n/0. '
                         'Usa --min-pair, --max-rules y --quantize.'))

    # ── Argumentos posicionales (flexibles según modo)
    p.add_argument('midi',    type=Path,
                   help='Fichero MIDI de entrada')
    p.add_argument('grammar', type=Path, nargs='?', default=None,
                   help='[modo compose] Fichero de gramática (.txt)')

    # ── Salida (válida en ambos modos)
    p.add_argument('-o', '--out', type=Path, default=None,
                   help='Fichero de salida: MIDI (compose) o gramática .txt (learn)')

    # ── Opciones modo compose
    compose = p.add_argument_group('Opciones modo compose')
    compose.add_argument('--tempo',    type=int, default=None,
                         help='Tempo BPM (default: detectado del MIDI)')
    compose.add_argument('--channel',  type=int, default=0,
                         help='Canal MIDI para notas ABC (default: 0)')
    compose.add_argument('--velocity', type=int, default=80,
                         help='Velocidad para notas ABC (default: 80)')
    compose.add_argument('--tpb',      type=int, default=None,
                         help='Ticks por negra en salida (default: igual que entrada)')

    # ── Opciones modo learn
    learn = p.add_argument_group('Opciones modo learn')
    learn.add_argument('--min-pair',  type=int, default=2,
                       help='Frecuencia mínima de un par para crear regla (default: 2)')
    learn.add_argument('--max-rules', type=int, default=None,
                       help='Número máximo de reglas a generar (default: ilimitado)')
    learn.add_argument('--merge-bars', action='store_true',
                       help='Fusionar compases consecutivos en rangos (en la raíz)')
    learn.add_argument('--seed-grammar', type=Path, default=None, metavar='FILE',
                       help=(
                           'Gramática semilla (.txt) cuyas reglas se aplican antes '
                           'de Re-Pair. Las reglas definidas en el fichero se '
                           'sustituyen en la secuencia de compases aprendida; '
                           'Re-Pair solo aprende la estructura restante. '
                           'La raíz de la gramática semilla se ignora.'
                       ))
    learn.add_argument('--fuzzy', type=float, default=None, metavar='THRESHOLD',
                       help=(
                           'Activar comparación difusa de compases. THRESHOLD es '
                           'la similitud mínima (0.0–1.0) para considerar dos compases '
                           'equivalentes. Valores típicos: 0.85–0.95. '
                           'Sin esta opción se usa comparación exacta (equivalente a 1.0).'
                       ))

    gest = p.add_argument_group('Opciones modo learn-gestures')
    gest.add_argument('--quantize', type=str, default=None, metavar='FRAC',
                      help=('Cuantizar onsets y duraciones a una rejilla (en negras) '
                            'antes de aprender: 1/4, 1/12, 0.25… Necesario con MIDIs '
                            'interpretados a mano; con ello la recomposición ya no es '
                            'idéntica al original, sino al original cuantizado.'))

    # ── Opciones comunes
    p.add_argument('--track',   type=int, default=None,
                   help='Índice del track del MIDI a usar (default: auto)')
    p.add_argument('--report',  action='store_true',
                   help='Mostrar informe detallado')
    p.add_argument('--verbose', action='store_true',
                   help='Traza paso a paso')
    return p


def main() -> None:
    parser = build_argument_parser()
    args   = parser.parse_args()

    # ── Validar fichero MIDI
    if not args.midi.exists():
        parser.error(f"Fichero MIDI no encontrado: {args.midi}")

    # ── Carga del MIDI (común a ambos modos)
    print(f"[1/?] Cargando MIDI: {args.midi} …")
    try:
        mid = MidiFile(str(args.midi))
    except Exception as e:
        sys.exit(f"  Error al leer el MIDI: {e}")

    t_idx = args.track if args.track is not None else choose_main_track(mid)
    print(f"  → {len(mid.tracks)} track(s), tpb={mid.ticks_per_beat}, "
          f"track principal={t_idx}")

    if args.learn and args.learn_gestures:
        parser.error("--learn y --learn-gestures son modos distintos; usa uno solo.")

    # ════════════════════════════════════════════════════════════════
    # MODO LEARN-GESTURES
    # ════════════════════════════════════════════════════════════════
    if args.learn_gestures:
        out_path = args.out or args.midi.with_suffix('.gestures.txt')
        try:
            quantize = Fraction(args.quantize) if args.quantize else None
        except (ValueError, ZeroDivisionError):
            parser.error(f"--quantize inválido: {args.quantize!r} (usa 1/4, 0.25…)")

        print("[2/4] Modo LEARN-GESTURES — aprendiendo el codebook de gestos …")
        try:
            grammar_text, stats = learn_gestures(
                mid       = mid,
                track_idx = t_idx,
                min_pair  = args.min_pair,
                max_rules = args.max_rules,
                quantize  = quantize,
                verbose   = args.verbose,
            )
        except Exception as e:
            sys.exit(f"  Error durante el aprendizaje de gestos: {e}")

        print(f"[3/4] Guardando gramática: {out_path} …")
        try:
            out_path.write_text(grammar_text, encoding='utf-8')
        except Exception as e:
            sys.exit(f"  Error al escribir la gramática: {e}")
        print(f"  → Fichero guardado: {out_path}")

        print("[4/4] Verificación: recomponer la gramática y comparar con el original …")
        if stats['roundtrip_ok']:
            print(f"  → OK: las {stats['notes_out']} notas coinciden exactamente "
                  f"(altura, inicio y duración)"
                  + (" con el original cuantizado." if quantize is not None else "."))
        else:
            print(f"  → ✗ NO coincide: {stats['notes_out']} notas recompuestas "
                  f"frente a {stats['notes']} originales.")
            for w in stats['warnings']:
                print(f"    ⚠  {w}")
        if len(stats['channels']) > 1:
            print(f"  [AVISO] El track usa {len(stats['channels'])} canales MIDI; "
                  f"la recomposición los unifica en uno (--channel).")
        print("  [AVISO] La velocidad no se aprende: la recomposición usa --velocity "
              f"(media del original: {stats['mean_vel']}).")

        if args.report:
            print_gesture_report(stats, out_path)

        print("\n✓ Grammar Composer (learn-gestures) finalizado correctamente.\n")
        if not stats['roundtrip_ok']:
            sys.exit(1)
        return

    # ════════════════════════════════════════════════════════════════
    # MODO LEARN
    # ════════════════════════════════════════════════════════════════
    if args.learn:
        out_path = args.out or args.midi.with_suffix('.grammar.txt')

        print(f"[2/4] Modo LEARN — extrayendo gramática …")
        try:
            # Leer gramática semilla si se proporcionó
            seed_grammar_text = None
            if args.seed_grammar is not None:
                if not args.seed_grammar.exists():
                    sys.exit(f"  Fichero de gramática semilla no encontrado: {args.seed_grammar}")
                seed_grammar_text = args.seed_grammar.read_text(encoding='utf-8')

            grammar_text, rules, root = learn_grammar(
                mid              = mid,
                track_idx        = t_idx,
                min_pair         = args.min_pair,
                max_rules        = args.max_rules,
                merge_bars       = args.merge_bars,
                fuzzy_threshold  = args.fuzzy if args.fuzzy is not None else 1.0,
                seed_grammar     = seed_grammar_text,
                verbose          = args.verbose,
            )
        except Exception as e:
            sys.exit(f"  Error durante el aprendizaje: {e}")

        print(f"[3/4] Guardando gramática: {out_path} …")
        try:
            out_path.write_text(grammar_text, encoding='utf-8')
        except Exception as e:
            sys.exit(f"  Error al escribir la gramática: {e}")
        print(f"  → Fichero guardado: {out_path}")

        if args.report:
            print_learn_report(rules, root, out_path)

        print("\n✓ Grammar Composer (learn) finalizado correctamente.\n")
        return

    # ════════════════════════════════════════════════════════════════
    # MODO COMPOSE
    # ════════════════════════════════════════════════════════════════
    if args.grammar is None:
        parser.error("En modo compose debes proporcionar un fichero de gramática, "
                     "o usar --learn para extraer la gramática del MIDI.")
    if not args.grammar.exists():
        parser.error(f"Fichero de gramática no encontrado: {args.grammar}")

    tpb   = args.tpb or mid.ticks_per_beat
    tempo = (int(60_000_000 / args.tempo) if args.tempo else get_tempo(mid))

    print(f"  → tempo={round(60_000_000/tempo)} BPM, tpb={tpb}")

    # ── Extracción de compases (todos los tracks con notas)
    print(f"[2/5] Extrayendo compases de todos los tracks …")
    all_bars = extract_all_tracks(mid)
    n_tracks = len(all_bars)
    n_bars   = max((max(b.keys()) for b in all_bars.values()), default=0)
    print(f"  → {n_tracks} track(s) con notas, {n_bars} compases máx.")
    # El track principal sigue siendo t_idx (usado en learn y para el informe)
    bars = all_bars.get(t_idx, extract_bars(mid, t_idx))

    # ── Parseo de la gramática
    print(f"[3/5] Parseando gramática: {args.grammar} …")
    try:
        grammar_text = args.grammar.read_text(encoding='utf-8')
        root, rules  = parse_grammar(grammar_text)
    except (ValueError, UnicodeDecodeError) as e:
        sys.exit(f"  Error en la gramática: {e}")

    print(f"  → Raíz: {root}")
    print(f"  → Reglas: {list(rules.keys())}")

    # Gramática autocontenida (solo notas ABC): no depende de los compases del
    # MIDI, así que se expande una sola vez en lugar de una por cada track.
    if not grammar_uses_bars(root, rules) and n_tracks > 1:
        print(f"  → Gramática autocontenida (sin compases del MIDI): se expande "
              f"una sola vez, no una por cada uno de los {n_tracks} tracks.")
        all_bars = {t_idx: {}}
        n_tracks = 1

    # ── Expansión de la gramática (un expander por track)
    print(f"[4/5] Expandiendo gramática ({n_tracks} track(s)) …")
    tracks_notes: dict[int, list] = {}
    report = None
    bar_length_fn = get_bar_length_ticks(mid)
    for orig_idx, trk_bars in sorted(all_bars.items()):
        expander = GrammarExpander(
            rules          = rules,
            bars           = trk_bars,
            tpb            = tpb,
            bar_length_fn  = bar_length_fn,
            channel        = args.channel,
            velocity       = args.velocity,
            verbose        = args.verbose,
        )
        try:
            notes = expander.expand_root(root)
        except Exception as e:
            sys.exit(f"  Error durante la expansión del track {orig_idx}: {e}")
        tracks_notes[orig_idx] = notes
        if orig_idx == t_idx:
            report = expander.report

    total_notes = sum(len(n) for n in tracks_notes.values())
    if total_notes == 0:
        print("  [AVISO] La expansión no produjo ninguna nota. "
              "Revisa la gramática y los compases del MIDI.")
    print(f"  → {total_notes} notas generadas en {n_tracks} track(s)")

    # ── Escritura del MIDI de salida
    out_path = args.out or args.midi.with_suffix('.grammar.mid')
    print(f"[5/5] Escribiendo MIDI de salida: {out_path} …")
    try:
        if n_tracks == 1:
            notes_to_midi(list(tracks_notes.values())[0], tpb, tempo, out_path)
        else:
            multitrack_to_midi(tracks_notes, tpb, tempo, out_path)
    except Exception as e:
        sys.exit(f"  Error al escribir el MIDI: {e}")

    print(f"  → Fichero guardado: {out_path}")

    if args.report and report is not None:
        print_report(report, args.grammar, args.midi)

    print("\n✓ Grammar Composer finalizado correctamente.\n")


if __name__ == '__main__':
    main()
