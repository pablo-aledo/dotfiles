#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║ expressive_injector.py                                               ║
# ║                                                                      ║
# ║ Complementario a expressive_resources.py: introduce un recurso       ║
# ║ expresivo (rítmico, melódico o armónico) en un compás concreto de un ║
# ║ MIDI de piano con manos separadas (RH/LH), modificando las notas     ║
# ║ existentes en ese compás para materializarlo, y escribe un nuevo     ║
# ║ fichero .mid con el resultado.                                       ║
# ║                                                                      ║
# ║ Recursos soportados (misma definición que expressive_resources.py): ║
# ║   sincopa       - adelanta el ataque de una nota a una posición      ║
# ║                    débil y la sostiene sobre el pulso fuerte         ║
# ║                    siguiente, vaciándolo de ataque propio            ║
# ║   contratiempo  - traslada un ataque al "y" del tiempo, dejando el   ║
# ║                    pulso fuerte anterior en silencio (sin ligadura)  ║
# ║   hemiola       - reagrupa los ataques de un compás ternario simple  ║
# ║                    o compuesto según la subdivisión contraria        ║
# ║                    (p.ej. 3+3 -> 2+2+2), extendiéndose a los         ║
# ║                    compases siguientes si hace falta más recorrido   ║
# ║   apoyatura     - antepone una nota ajena a la armonía de LH en      ║
# ║                    tiempo fuerte que resuelve por grado conjunto a   ║
# ║                    la nota original (requiere mano RH y armonía LH)  ║
# ║   floreo        - sustituye el ataque de una nota por un grupo       ║
# ║                    ornamental breve (mordente, grupeto o trino)      ║
# ║                                                                      ║
# ║ USO:                                                                 ║
# ║   python3 expressive_injector.py list-resources                     ║
# ║   python3 expressive_injector.py info pieza.mid                     ║
# ║   python3 expressive_injector.py inject pieza.mid 12 sincopa        ║
# ║   python3 expressive_injector.py inject pieza.mid 12 contratiempo   ║
# ║       --hand LH --beat 2 --out pieza_mod.mid                        ║
# ║   python3 expressive_injector.py inject pieza.mid 8 hemiola         ║
# ║   python3 expressive_injector.py inject pieza.mid 4 apoyatura       ║
# ║       --beat 1 --direccion arriba                                    ║
# ║   python3 expressive_injector.py inject pieza.mid 6 floreo          ║
# ║       --tipo grupeto                                                 ║
# ║                                                                      ║
# ║ Opciones de "inject":                                                ║
# ║   --hand RH|LH        mano objetivo (por defecto RH; apoyatura       ║
# ║                        siempre usa RH con armonía de LH)             ║
# ║   --beat N            pulso objetivo dentro del compás (1-indexado); ║
# ║                        si se omite, se usa el primer pulso de esa    ║
# ║                        mano con una nota atacando justo ahí          ║
# ║   --tipo T            solo floreo: mordente|grupeto|trino            ║
# ║   --direccion D       solo apoyatura: arriba|abajo (auto si se omite)║
# ║   --out FICHERO       ruta de salida (por defecto se deriva del      ║
# ║                        nombre de entrada, el recurso y el compás)    ║
# ║                                                                      ║
# ║ Cada subcomando de inyección requiere que ya exista, en el compás y  ║
# ║ mano indicados, al menos una nota que sirva de punto de partida      ║
# ║ (el "material" sobre el que se moldea el recurso); no compone notas  ║
# ║ desde cero. Si no encuentra una candidata adecuada, informa por qué  ║
# ║ y sugiere ajustar --beat o --hand.                                   ║
# ║                                                                      ║
# ║ Dependencias: mido, numpy                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

import argparse
import sys
from dataclasses import dataclass

import numpy as np

try:
    import mido
except ImportError:
    print("Este script necesita 'mido'. Instala con: pip install mido --break-system-packages",
          file=sys.stderr)
    sys.exit(1)

# ── Colores ANSI ─────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

RESOURCE_COLOR = {
    "sincopa": C.YELLOW,
    "contratiempo": C.CYAN,
    "hemiola": C.MAGENTA,
    "apoyatura": C.GREEN,
    "floreo": C.BLUE,
}

RESOURCE_DESCRIPTIONS = {
    "sincopa": "Desplazamiento del acento a un tiempo o subdivisión débil, "
               "sosteniendo la nota sobre el tiempo fuerte siguiente y "
               "suprimiendo su ataque esperado.",
    "contratiempo": "Ataque en la subdivisión débil ('y' del tiempo) "
                     "precedido de silencio en el tiempo fuerte, sin "
                     "ligadura que lo sostenga desde antes.",
    "hemiola": "Reagrupación temporal de los pulsos del compás (p.ej. "
               "3+3 reinterpretado como 2+2+2), creando ambigüedad "
               "métrica momentánea.",
    "apoyatura": "Nota ajena a la armonía sonando en tiempo fuerte que "
                 "resuelve por grado conjunto a una nota del acorde.",
    "floreo": "Grupo ornamental de notas muy breves (mordente, trino o "
              "grupeto) alrededor de una nota principal.",
}

ALL_RESOURCES = list(RESOURCE_DESCRIPTIONS.keys())

# ── Modelo de datos ──────────────────────────────────────────────────────

@dataclass
class Note:
    pitch: int
    start_tick: int
    end_tick: int
    velocity: int
    channel: int
    hand: str          # "RH" | "LH"
    track_idx: int

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick


@dataclass
class TimeSigChange:
    tick: int
    numerator: int
    denominator: int


# ── Carga de MIDI, separación de manos y eventos no-nota ─────────────────

def load_notes(path: str):
    """Igual que en expressive_resources.py, pero además conserva todo
    mensaje que no sea nota (meta-eventos, control_change, etc.) con su
    tick absoluto y su pista de origen, para poder reconstruir el fichero
    tras modificar las notas."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat

    time_sigs = [TimeSigChange(0, 4, 4)]
    notes = []
    other_events = {}  # track_idx -> [(abs_tick, msg_sin_time)]

    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        open_notes = {}  # (pitch, channel) -> (start_tick, velocity, channel)
        for msg in track:
            abs_tick += msg.time
            if msg.type == "time_signature":
                time_sigs.append(TimeSigChange(abs_tick, msg.numerator, msg.denominator))
                other_events.setdefault(track_idx, []).append((abs_tick, msg.copy(time=0)))
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.note, msg.channel)] = (abs_tick, msg.velocity, msg.channel)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in open_notes:
                    start_tick, vel, ch = open_notes.pop(key)
                    notes.append(Note(msg.note, start_tick, abs_tick, vel, ch, "?", track_idx))
            elif msg.type == "end_of_track":
                continue  # se regenera al reconstruir
            else:
                other_events.setdefault(track_idx, []).append((abs_tick, msg.copy(time=0)))

    time_sigs = sorted(time_sigs, key=lambda t: t.tick)
    dedup = [time_sigs[0]]
    for ts in time_sigs[1:]:
        if ts.tick == dedup[-1].tick:
            dedup[-1] = ts
        else:
            dedup.append(ts)
    time_sigs = dedup

    notes.sort(key=lambda n: n.start_tick)
    return tpb, time_sigs, notes, mid, other_events


def assign_hands(mid: "mido.MidiFile", notes):
    """Asigna RH/LH. Primero por nombre de pista, si no por registro.
    (idéntico al criterio de expressive_resources.py)"""
    track_names = {}
    for idx, track in enumerate(mid.tracks):
        name = ""
        for msg in track:
            if msg.type == "track_name":
                name = msg.name.lower()
                break
        track_names[idx] = name

    rh_kw = ["right", "derecha", " rh", "rh ", "treble", "melody", "melodia", "soprano"]
    lh_kw = ["left", "izquierda", " lh", "lh ", "bass", "bajo", "acomp"]

    hand_by_track = {}
    for idx, name in track_names.items():
        if any(k in name for k in rh_kw):
            hand_by_track[idx] = "RH"
        elif any(k in name for k in lh_kw):
            hand_by_track[idx] = "LH"

    track_indices = sorted(set(n.track_idx for n in notes))

    if len(hand_by_track) < 2 and len(track_indices) >= 2:
        by_track_pitch = {}
        for n in notes:
            by_track_pitch.setdefault(n.track_idx, []).append(n.pitch)
        medians = {t: float(np.median(p)) for t, p in by_track_pitch.items()}
        ordered = sorted(track_indices, key=lambda t: -medians.get(t, 60))
        if len(ordered) >= 2:
            hand_by_track = {ordered[0]: "RH", ordered[1]: "LH"}
            for t in ordered[2:]:
                hand_by_track[t] = "RH" if medians[t] >= 60 else "LH"

    if len(hand_by_track) < 2:
        pitches = [n.pitch for n in notes]
        split = float(np.median(pitches)) if pitches else 60
        for n in notes:
            n.hand = "RH" if n.pitch >= split else "LH"
        return notes

    for n in notes:
        n.hand = hand_by_track.get(n.track_idx, "RH" if n.pitch >= 60 else "LH")
    return notes


# ── Rejilla métrica (idéntica a expressive_resources.py, + bar_info) ─────

class MetricGrid:
    def __init__(self, tpb, time_sigs, last_tick):
        self.tpb = tpb
        self.time_sigs = time_sigs
        self._build(last_tick)

    def _bar_len_ticks(self, numerator, denominator):
        return int(round(numerator * (4.0 / denominator) * self.tpb))

    def _build(self, last_tick):
        bars = []
        cur_tick = 0
        bar_num = 1
        ts_idx = 0
        cur_ts = self.time_sigs[0]
        next_change_tick = (self.time_sigs[1].tick if len(self.time_sigs) > 1 else None)
        while cur_tick <= last_tick + self._bar_len_ticks(cur_ts.numerator, cur_ts.denominator):
            if next_change_tick is not None and cur_tick >= next_change_tick:
                ts_idx += 1
                cur_ts = self.time_sigs[ts_idx]
                next_change_tick = (self.time_sigs[ts_idx + 1].tick
                                     if ts_idx + 1 < len(self.time_sigs) else None)
            bars.append((cur_tick, cur_ts.numerator, cur_ts.denominator, bar_num))
            cur_tick += self._bar_len_ticks(cur_ts.numerator, cur_ts.denominator)
            bar_num += 1
        self.bars = bars

    def locate(self, tick):
        idx = 0
        for i, (bt, num, den, bn) in enumerate(self.bars):
            if bt <= tick:
                idx = i
            else:
                break
        bar_start, num, den, bar_num = self.bars[idx]
        bar_len = self._bar_len_ticks(num, den)
        beat_pos = (tick - bar_start) / self.tpb
        return bar_num, num, den, beat_pos, bar_start, bar_len

    def bar_info(self, bar_num):
        """(bar_start_tick, numerator, denominator, bar_len_ticks) para un
        número de compás 1-indexado, o None si está fuera de rango."""
        idx = bar_num - 1
        if idx < 0 or idx >= len(self.bars):
            return None
        bar_start, num, den, bn = self.bars[idx]
        return bar_start, num, den, self._bar_len_ticks(num, den)

    def is_compound(self, numerator, denominator):
        return denominator == 8 and numerator % 3 == 0 and numerator > 3

    def beat_label(self, numerator, denominator, beat_pos_quarters):
        quarters_per_beat = 4.0 / denominator
        if self.is_compound(numerator, denominator):
            quarters_per_beat = 1.5
        beat_idx = beat_pos_quarters / quarters_per_beat
        return f"{beat_idx + 1:.2f}"


# ── Utilidades armónicas (idénticas a expressive_resources.py) ───────────

def pitch_classes(pitches):
    return set(p % 12 for p in pitches)


def notes_sounding_at(notes, tick, exclude=None):
    return [n for n in notes if n.start_tick <= tick < n.end_tick and n is not exclude]


NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def midi_name(pitch: int) -> str:
    octave = pitch // 12 - 1
    return f"{NOTE_NAMES[pitch % 12]}{octave}"


# ── Helpers comunes de los inyectores ─────────────────────────────────────

def quarters_per_beat_of(grid, num, den):
    return 1.5 if grid.is_compound(num, den) else (4.0 / den)


def beat_attacks_in_bar(hand_notes, bar_start, bar_len, quarters_per_beat, tpb, tol=0.08):
    """Notas de una mano que atacan (con tolerancia `tol`, en fracción de
    pulso) justo sobre un pulso dentro del compás [bar_start, bar_start+bar_len).
    Devuelve [(beat_idx_1indexed, Note), ...] ordenado por beat_idx."""
    out = []
    for n in hand_notes:
        if not (bar_start <= n.start_tick < bar_start + bar_len):
            continue
        rel = (n.start_tick - bar_start) / tpb
        beat_idx = round(rel / quarters_per_beat) + 1
        if abs(rel - (beat_idx - 1) * quarters_per_beat) < tol * quarters_per_beat:
            out.append((beat_idx, n))
    return sorted(out, key=lambda c: c[0])


def pick_candidate(candidates, beat):
    if beat is not None:
        candidates = [c for c in candidates if c[0] == beat]
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: c[0])[0]


# ── Inyectores ────────────────────────────────────────────────────────────

def inject_sincopa(notes, grid, bar_num, hand, beat=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    beat_ticks = int(round(qpb * grid.tpb))
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)

    candidates = beat_attacks_in_bar(hand_notes, bar_start, bar_len, qpb, grid.tpb)
    picked = pick_candidate(candidates, beat)
    if not picked:
        return None, ("No hay ninguna nota de esa mano atacando justo sobre un pulso "
                       "en ese compás (necesaria como punto de partida para la síncopa). "
                       "Prueba otra mano con --hand o fija el pulso con --beat.")
    beat_idx, target = picked
    beat_tick = target.start_tick
    half = max(1, beat_ticks // 2)
    new_start = max(0, beat_tick - half)

    for m in hand_notes:
        if m is target:
            continue
        if m.start_tick <= new_start < m.end_tick:
            m.end_tick = new_start

    orig_pitch = target.pitch
    target.start_tick = new_start
    msg = (f"Compás {bar_num}, mano {hand}: {midi_name(orig_pitch)} adelanta su ataque, "
           f"del tiempo {beat_idx} a una posición débil justo antes, y queda sostenida "
           f"sobre el tiempo {beat_idx} sin nuevo ataque ahí -> síncopa.")
    return target, msg


def inject_contratiempo(notes, grid, bar_num, hand, beat=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    beat_ticks = int(round(qpb * grid.tpb))
    half = max(1, beat_ticks // 2)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)

    candidates = beat_attacks_in_bar(hand_notes, bar_start, bar_len, qpb, grid.tpb)
    picked = pick_candidate(candidates, beat)
    if not picked:
        return None, ("No hay ninguna nota de esa mano atacando justo sobre un pulso "
                       "en ese compás (necesaria para convertirla en contratiempo). "
                       "Prueba otra mano con --hand o fija el pulso con --beat.")
    beat_idx, target = picked
    beat_tick = target.start_tick
    new_start = beat_tick + half
    max_end = beat_tick + beat_ticks
    new_end = min(target.end_tick, max_end)
    if new_end <= new_start:
        new_end = new_start + max(1, half // 2)

    # el pulso fuerte debe quedar en silencio: recorta cualquier nota que
    # llegara ligada desde antes
    for m in hand_notes:
        if m is target:
            continue
        if m.start_tick < beat_tick <= m.end_tick:
            m.end_tick = beat_tick

    orig_pitch = target.pitch
    target.start_tick = new_start
    target.end_tick = new_end
    msg = (f"Compás {bar_num}, mano {hand}: {midi_name(orig_pitch)} se traslada del "
           f"tiempo {beat_idx} al 'y' de ese tiempo, dejando el pulso fuerte en "
           f"silencio y sin sostenerse sobre el siguiente pulso -> contratiempo.")
    return target, msg


def inject_hemiola(notes, grid, bar_num, hand):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    is_compound = grid.is_compound(num, den)
    if not (is_compound or num == 3):
        return None, (f"La hemiola requiere un compás ternario simple (3/x) o "
                       f"compuesto (6/8, 9/8, 12/8); el compás {bar_num} es {num}/{den}.")

    hemiola_unit = 1.0 if is_compound else 1.5
    unit_ticks = int(round(hemiola_unit * grid.tpb))
    needed_ticks = 4 * unit_ticks

    span_start = bar_start
    span_end = bar_start + bar_len
    last_bar = bar_num
    while span_end - span_start < needed_ticks:
        nxt = grid.bar_info(last_bar + 1)
        if nxt and nxt[1] == num and nxt[2] == den:
            span_end += nxt[3]
            last_bar += 1
        else:
            break

    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    in_span = [n for n in hand_notes if span_start <= n.start_tick < span_end]
    if not in_span:
        return None, ("No hay ninguna nota de esa mano en ese compás (ni en los "
                       "siguientes que comparten compás) para tomar como referencia "
                       "de alturas. Prueba otra mano con --hand.")
    ref_pitches = [n.pitch for n in in_span]
    ref_vel = int(round(np.mean([n.velocity for n in in_span])))
    ref_channel = in_span[0].channel
    ref_track = in_span[0].track_idx

    for n in list(notes):
        if n.hand != hand:
            continue
        if n.start_tick < span_end and n.end_tick > span_start:
            if n.start_tick < span_start:
                n.end_tick = min(n.end_tick, span_start)
            else:
                notes.remove(n)

    new_notes = []
    tick = span_start
    i = 0
    while tick < span_end:
        pitch = ref_pitches[i % len(ref_pitches)]
        end = min(tick + unit_ticks, span_end)
        nn = Note(pitch=pitch, start_tick=tick, end_tick=end, velocity=ref_vel,
                  channel=ref_channel, hand=hand, track_idx=ref_track)
        notes.append(nn)
        new_notes.append(nn)
        tick += unit_ticks
        i += 1

    bars_txt = f"{bar_num}" if last_bar == bar_num else f"{bar_num}-{last_bar}"
    msg = (f"Compases {bars_txt}, mano {hand}: se han regenerado {len(new_notes)} "
           f"ataques equiespaciados cada {hemiola_unit:g} negra(s), en lugar de la "
           f"subdivisión natural de {num}/{den} -> hemiola.")
    return new_notes, msg


def inject_apoyatura(notes, grid, bar_num, hand, beat=None, direccion=None):
    if hand != "RH":
        return None, "La apoyatura se construye en RH contra la armonía de LH; usa --hand RH."
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)

    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    if not lh_notes:
        return None, "No hay notas en LH que sirvan de armonía de referencia."

    candidates = beat_attacks_in_bar(rh_notes, bar_start, bar_len, qpb, grid.tpb, tol=0.15)
    picked = pick_candidate(candidates, beat)
    if not picked:
        return None, ("No hay ninguna nota RH atacando justo sobre un pulso en ese "
                       "compás. Prueba otro compás o fija el pulso con --beat.")
    beat_idx, target = picked

    harmony = notes_sounding_at(lh_notes, target.start_tick)
    if not harmony:
        return None, ("No suena ninguna nota de LH en el instante de esa nota; no hay "
                       "armonía contra la que construir la disonancia.")
    harmony_pcs = pitch_classes([h.pitch for h in harmony])

    up, down = target.pitch + 1, target.pitch - 1
    order = [down, up] if direccion == "abajo" else [up, down]
    neighbor = next((c for c in order if c % 12 not in harmony_pcs), None)
    if neighbor is None:
        order2 = [target.pitch + 2, target.pitch - 2]
        neighbor = next((c for c in order2 if c % 12 not in harmony_pcs), None)
    if neighbor is None:
        return None, ("Todas las notas vecinas de esa nota pertenecen a la armonía de "
                       "LH en ese instante; no se puede construir una disonancia clara. "
                       "Prueba otro compás/pulso.")

    dur = target.duration_ticks
    orn_dur = max(1, min(dur // 2, int(round(qpb * grid.tpb * 0.4))))
    if orn_dur >= dur:
        orn_dur = max(1, dur - 1)

    orig_pitch = target.pitch
    appog = Note(pitch=neighbor, start_tick=target.start_tick,
                 end_tick=target.start_tick + orn_dur, velocity=target.velocity,
                 channel=target.channel, hand="RH", track_idx=target.track_idx)
    target.start_tick = target.start_tick + orn_dur
    notes.append(appog)

    direction_txt = "descendente" if orig_pitch < neighbor else "ascendente"
    msg = (f"Compás {bar_num}, tiempo {beat_idx}: se antepone {midi_name(neighbor)} "
           f"(ajena a la armonía de LH) que resuelve {direction_txt} por grado "
           f"conjunto a {midi_name(orig_pitch)} -> apoyatura.")
    return appog, msg


def inject_floreo(notes, grid, bar_num, hand, beat=None, tipo="mordente"):
    if tipo not in ("mordente", "grupeto", "trino"):
        return None, f"Tipo de floreo desconocido: {tipo} (usa mordente, grupeto o trino)."
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    beat_ticks = int(round(qpb * grid.tpb))
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)

    in_bar = [n for n in hand_notes if bar_start <= n.start_tick < bar_start + bar_len]
    if beat is not None:
        filtered = []
        for n in in_bar:
            rel = (n.start_tick - bar_start) / grid.tpb
            beat_idx = round(rel / qpb) + 1
            if beat_idx == beat:
                filtered.append(n)
        in_bar = filtered
    in_bar = [n for n in in_bar if n.duration_ticks >= beat_ticks * 0.3]
    if not in_bar:
        return None, ("No hay ninguna nota de esa mano en ese compás con duración "
                       "suficiente para alojar un floreo. Prueba otro compás/pulso.")
    target = in_bar[0]
    orig_pitch = target.pitch

    step = 2 if tipo == "grupeto" else 1
    upper, lower = orig_pitch + step, orig_pitch - step
    grace_dur = max(1, int(round(beat_ticks * 0.14)))

    if tipo == "mordente":
        seq, durs = [upper, orig_pitch], [grace_dur, grace_dur]
    elif tipo == "trino":
        seq, durs = [orig_pitch, upper, orig_pitch, upper], [grace_dur] * 4
    else:  # grupeto
        seq, durs = [upper, orig_pitch, lower, orig_pitch], [grace_dur] * 4

    total_orn = sum(durs)
    if total_orn >= target.duration_ticks:
        scale = (target.duration_ticks * 0.7) / total_orn if total_orn else 0
        durs = [max(1, int(d * scale)) for d in durs]
        total_orn = sum(durs)
        if total_orn >= target.duration_ticks:
            return None, ("La nota elegida es demasiado breve para alojar el floreo. "
                           "Prueba otro compás/pulso.")

    remaining = target.duration_ticks - total_orn
    tick = target.start_tick
    new_notes = []
    for pitch, d in zip(seq, durs):
        nn = Note(pitch=pitch, start_tick=tick, end_tick=tick + d, velocity=target.velocity,
                  channel=target.channel, hand=hand, track_idx=target.track_idx)
        notes.append(nn)
        new_notes.append(nn)
        tick += d
    target.start_tick = tick
    target.end_tick = tick + remaining

    msg = (f"Compás {bar_num}, mano {hand}: el ataque de {midi_name(orig_pitch)} se "
           f"sustituye por un {tipo} ({', '.join(midi_name(p) for p in seq)}) seguido "
           f"de la nota principal sostenida -> floreo.")
    return new_notes + [target], msg


INJECTORS = {
    "sincopa": inject_sincopa,
    "contratiempo": inject_contratiempo,
    "hemiola": inject_hemiola,
    "apoyatura": inject_apoyatura,
    "floreo": inject_floreo,
}

# ── Reconstrucción y escritura del MIDI ───────────────────────────────────

def rebuild_and_save(mid, tpb, notes, other_events, out_path):
    """Reconstruye cada pista a partir de los eventos no-nota originales y
    la lista de notas final (tras la inyección), y guarda un nuevo .mid."""
    new_mid = mido.MidiFile(ticks_per_beat=tpb)

    notes_by_track = {}
    for n in notes:
        notes_by_track.setdefault(n.track_idx, []).append(n)

    for t_idx in range(len(mid.tracks)):
        events = []  # (abs_tick, prioridad, msg)  prioridad: note_off < otros < note_on
        for abs_tick, msg in other_events.get(t_idx, []):
            events.append((abs_tick, 1, msg))
        for n in notes_by_track.get(t_idx, []):
            if n.end_tick <= n.start_tick:
                continue
            events.append((n.start_tick, 2, mido.Message(
                "note_on", note=n.pitch, velocity=max(1, min(127, n.velocity)),
                channel=n.channel, time=0)))
            events.append((n.end_tick, 0, mido.Message(
                "note_off", note=n.pitch, velocity=0, channel=n.channel, time=0)))

        events.sort(key=lambda e: (e[0], e[1]))
        track = mido.MidiTrack()
        last_tick = 0
        for abs_tick, _, msg in events:
            delta = max(0, abs_tick - last_tick)
            track.append(msg.copy(time=delta))
            last_tick = abs_tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        new_mid.tracks.append(track)

    new_mid.save(out_path)


def default_output_name(path, recurso, compas):
    base = path[:-4] if path.lower().endswith(".mid") else path
    return f"{base}_{recurso}_c{compas}.mid"


# ── Subcomandos ───────────────────────────────────────────────────────────

def cmd_info(args):
    tpb, time_sigs, notes, mid, _ = load_notes(args.midi)
    notes = assign_hands(mid, notes)
    last_tick = max((n.end_tick for n in notes), default=0)
    grid = MetricGrid(tpb, time_sigs, last_tick)

    rh = [n for n in notes if n.hand == "RH"]
    lh = [n for n in notes if n.hand == "LH"]

    print(f"{C.BOLD}Fichero:{C.RESET} {args.midi}")
    print(f"Pistas: {len(mid.tracks)}   Ticks/negra: {tpb}")
    print(f"Compases de tiempo: " +
          ", ".join(f"{ts.numerator}/{ts.denominator}@tick{ts.tick}" for ts in time_sigs))
    print(f"Notas totales: {len(notes)}  (RH: {len(rh)}, LH: {len(lh)})")
    print(f"Compases estimados: {grid.bars[-1][3] if grid.bars else 0}")


def cmd_list_resources(args):
    for name, desc in RESOURCE_DESCRIPTIONS.items():
        color = RESOURCE_COLOR.get(name, "")
        print(f"{C.BOLD}{color}{name}{C.RESET}")
        print(f"  {desc}")
        print()


def cmd_inject(args):
    tpb, time_sigs, notes, mid, other_events = load_notes(args.midi)
    notes = assign_hands(mid, notes)
    last_tick = max((n.end_tick for n in notes), default=0)
    # margen extra: permite inyectar hemiolas que se extiendan más allá del
    # final de la pieza, o compases pedidos justo en el límite
    grid = MetricGrid(tpb, time_sigs, last_tick + tpb * 32)

    hand = args.hand or "RH"
    if args.recurso == "apoyatura":
        hand = "RH"

    kwargs = {"beat": args.beat} if args.recurso in ("sincopa", "contratiempo", "floreo") else {}
    if args.recurso == "apoyatura":
        kwargs = {"beat": args.beat, "direccion": args.direccion}
    if args.recurso == "floreo":
        kwargs["tipo"] = args.tipo

    fn = INJECTORS[args.recurso]
    if args.recurso == "hemiola":
        result, msg = fn(notes, grid, args.compas, hand)
    else:
        result, msg = fn(notes, grid, args.compas, hand, **kwargs)

    if result is None:
        print(f"{C.RED}✗{C.RESET} {msg}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or default_output_name(args.midi, args.recurso, args.compas)
    rebuild_and_save(mid, tpb, notes, other_events, out_path)

    color = RESOURCE_COLOR.get(args.recurso, "")
    print(f"{C.GREEN}✓{C.RESET} {C.BOLD}{color}[{args.recurso.upper()}]{C.RESET} {msg}")
    print(f"Guardado en: {C.BOLD}{out_path}{C.RESET}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inyecta un recurso expresivo en un compás concreto de un MIDI "
                    "de piano con manos separadas. Complementario a "
                    "expressive_resources.py.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Muestra metadatos básicos del MIDI")
    p_info.add_argument("midi")
    p_info.set_defaults(func=cmd_info)

    p_list = sub.add_parser("list-resources", help="Lista los recursos disponibles")
    p_list.set_defaults(func=cmd_list_resources)

    p_inject = sub.add_parser("inject", help="Inyecta un recurso en un compás y guarda un nuevo MIDI")
    p_inject.add_argument("midi")
    p_inject.add_argument("compas", type=int, help="Número de compás (1-indexado)")
    p_inject.add_argument("recurso", choices=ALL_RESOURCES)
    p_inject.add_argument("--hand", choices=["RH", "LH"], default=None,
                           help="Mano objetivo (por defecto RH)")
    p_inject.add_argument("--beat", type=int, default=None,
                           help="Pulso objetivo dentro del compás (1-indexado)")
    p_inject.add_argument("--tipo", choices=["mordente", "grupeto", "trino"], default="mordente",
                           help="Solo floreo: tipo de adorno")
    p_inject.add_argument("--direccion", choices=["arriba", "abajo"], default=None,
                           help="Solo apoyatura: dirección de la nota vecina")
    p_inject.add_argument("--out", default=None, help="Ruta del MIDI de salida")
    p_inject.set_defaults(func=cmd_inject)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
