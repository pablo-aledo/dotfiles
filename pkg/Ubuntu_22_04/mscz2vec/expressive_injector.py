#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║ expressive_injector.py                                               ║
# ║                                                                      ║
# ║ Complementario a expressive_resources.py: introduce un recurso       ║
# ║ expresivo (rítmico, melódico, armónico o dinámico) en un compás      ║
# ║ concreto de un MIDI de piano con manos separadas (RH/LH), modificando║
# ║ las notas (o el tempo) existentes para materializarlo, y escribe un  ║
# ║ nuevo fichero .mid con el resultado.                                 ║
# ║                                                                      ║
# ║ Recursos soportados (usa 'list-resources' para el detalle):          ║
# ║   Rítmicos:   sincopa, contratiempo, hemiola, rubato*, agogica*,     ║
# ║               polirritmia                                            ║
# ║   Melódicos:  apoyatura, floreo, anticipacion, escapada,             ║
# ║               nota_de_paso_cromatica                                 ║
# ║   Armónicos:  acorde_disminuido, dominante_secundaria,               ║
# ║               acorde_prestado, sexta_napolitana, pedal_armonico,     ║
# ║               retardo                                                ║
# ║   Dinámicos:  sforzando, silencio_expresivo, crescendo_dirigido      ║
# ║   (* rubato y agogica no tocan notas: insertan una curva de          ║
# ║   mensajes set_tempo en el compás indicado)                          ║
# ║                                                                      ║
# ║ USO:                                                                 ║
# ║   python3 expressive_injector.py list-resources                     ║
# ║   python3 expressive_injector.py info pieza.mid                     ║
# ║   python3 expressive_injector.py inject pieza.mid <compas> <recurso> ║
# ║       [opciones]                                                     ║
# ║                                                                      ║
# ║ Ejemplos:                                                             ║
# ║   ... 12 sincopa --beat 2                                            ║
# ║   ... 12 contratiempo --hand LH --beat 2 --out salida.mid            ║
# ║   ... 8 hemiola                                                      ║
# ║   ... 4 apoyatura --beat 1 --direccion arriba                        ║
# ║   ... 6 floreo --tipo grupeto                                        ║
# ║   ... 5 anticipacion                                                  ║
# ║   ... 7 escapada                                                     ║
# ║   ... 3 nota_de_paso_cromatica                                       ║
# ║   ... 2 acorde_disminuido --septima                                  ║
# ║   ... 9 dominante_secundaria                                         ║
# ║   ... 9 acorde_prestado                                              ║
# ║   ... 4 sexta_napolitana                                             ║
# ║   ... 6 pedal_armonico --run 4                                       ║
# ║   ... 10 retardo --hand RH                                           ║
# ║   ... 2 sforzando --beat 3                                           ║
# ║   ... 5 silencio_expresivo                                           ║
# ║   ... 3 crescendo_dirigido --run 5                                   ║
# ║   ... 4 polirritmia --patron 3:2                                     ║
# ║   ... 6 rubato --direccion-tempo ritardando                          ║
# ║   ... 6 agogica --direccion-tempo alargamiento                       ║
# ║                                                                      ║
# ║ Opciones de "inject" (según el recurso):                             ║
# ║   --hand RH|LH        mano objetivo (por defecto RH; varios recursos ║
# ║                        fuerzan una mano concreta, ver list-resources)║
# ║   --beat N            pulso objetivo dentro del compás (1-indexado); ║
# ║                        si se omite, se usa el primer candidato       ║
# ║   --tipo T            solo floreo: mordente|grupeto|trino            ║
# ║   --direccion D       solo apoyatura: arriba|abajo (auto si se omite)║
# ║   --direccion-tempo D solo rubato/agogica: acelerando|ritardando     ║
# ║   --septima           solo acorde_disminuido: séptima en vez de tríada║
# ║   --patron A:B        solo polirritmia: proporción RH:LH (p.ej. 3:2) ║
# ║   --run N             solo pedal_armonico / crescendo_dirigido       ║
# ║   --out FICHERO       ruta de salida (por defecto se deriva del      ║
# ║                        nombre de entrada, el recurso y el compás)    ║
# ║                                                                      ║
# ║ Cada inyector toma notas ya existentes en el compás (y mano) como    ║
# ║ punto de partida y las transforma para materializar el recurso; no   ║
# ║ compone música desde cero. Si no encuentra una candidata adecuada,   ║
# ║ informa por qué y sugiere ajustar --beat, --hand o el compás.        ║
# ║                                                                      ║
# ║ Dependencias: mido, numpy                                            ║
# ╚══════════════════════════════════════════════════════════════════════╝

import argparse
import bisect
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
    "rubato": "\033[91m",
    "agogica": "\033[93m",
    "polirritmia": "\033[95m",
    "anticipacion": "\033[92m",
    "escapada": "\033[94m",
    "nota_de_paso_cromatica": "\033[96m",
    "acorde_disminuido": C.RED,
    "dominante_secundaria": C.WHITE,
    "acorde_prestado": "\033[97m",
    "sexta_napolitana": "\033[35m",
    "pedal_armonico": "\033[34m",
    "retardo": "\033[32m",
    "sforzando": "\033[31m",
    "silencio_expresivo": "\033[90m",
    "crescendo_dirigido": "\033[33m",
}

RESOURCE_DESCRIPTIONS = {
    # --- Rítmicos ---
    "sincopa": "Desplazamiento del acento a un tiempo o subdivisión débil, "
               "sosteniendo la nota sobre el tiempo fuerte siguiente y "
               "suprimiendo su ataque esperado.",
    "contratiempo": "Ataque en la subdivisión débil ('y' del tiempo) "
                     "precedido de silencio en el tiempo fuerte, sin "
                     "ligadura que lo sostenga desde antes.",
    "hemiola": "Reagrupación temporal de los pulsos del compás (p.ej. "
               "3+3 reinterpretado como 2+2+2), creando ambigüedad "
               "métrica momentánea.",
    "rubato": "Cambio de tempo continuado (accelerando/ritardando) a lo "
              "largo de varios eventos. Se codifica insertando una curva "
              "de mensajes set_tempo (mano '—', no toca notas).",
    "agogica": "Pequeño respiro de tempo puntual que se recupera de "
               "inmediato, para realzar una llegada. Igual que rubato, "
               "se codifica como mensajes set_tempo, sin tocar notas.",
    "polirritmia": "Dos subdivisiones distintas y regulares del mismo "
                    "pulso sonando a la vez en cada mano (3 contra 2, "
                    "4 contra 3, etc.).",
    # --- Melódicos ---
    "apoyatura": "Nota ajena a la armonía sonando en tiempo fuerte que "
                 "resuelve por grado conjunto a una nota del acorde.",
    "floreo": "Grupo ornamental de notas muy breves (mordente, trino o "
              "grupeto) alrededor de una nota principal.",
    "anticipacion": "Nota que suena justo antes de que la armonía cambie, "
                     "adelantando una clase de altura del acorde que está "
                     "por llegar.",
    "escapada": "Nota ajena alcanzada por grado conjunto desde una nota "
                "del acorde y abandonada con un salto en dirección "
                "opuesta hacia otra nota del acorde.",
    "nota_de_paso_cromatica": "Nota ajena que conecta por grado conjunto, "
                               "en la misma dirección y con al menos un "
                               "semitono, dos notas del acorde.",
    # --- Armónicos ---
    "acorde_disminuido": "Acorde (tríada o séptima) formado enteramente "
                          "por terceras menores apiladas, intrínsecamente "
                          "inestable.",
    "dominante_secundaria": "Acorde con tercera mayor y tritono (sonoridad "
                             "de dominante) ajeno a la tonalidad estimada, "
                             "que resuelve por círculo de quintas. Mano LH.",
    "acorde_prestado": "Acorde con alguna nota ajena a la tonalidad "
                        "estimada, sin tritono, sugiriendo un préstamo de "
                        "la tonalidad paralela.",
    "sexta_napolitana": "Tríada mayor sobre el segundo grado descendido "
                         "en primera inversión, respecto a la tonalidad "
                         "estimada. Mano LH.",
    "pedal_armonico": "Nota grave sostenida o repetida sin cambiar "
                       "mientras la armonía superior se mueve por encima. "
                       "Mano LH.",
    "retardo": "Nota consonante preparada, ligada sobre un cambio de "
               "armonía en el que pasa a ser disonante, que resuelve "
               "después por grado conjunto (suspensión).",
    # --- Dinámicos / articulatorios ---
    "sforzando": "Nota o acorde con una velocidad muy por encima de la "
                 "media local de esa mano: un acento súbito y puntual.",
    "silencio_expresivo": "Silencio notablemente más largo de lo habitual "
                           "justo antes de una entrada, respecto al hueco "
                           "medio reciente de esa mano.",
    "crescendo_dirigido": "Racha de ataques consecutivos con velocidad "
                           "creciente en una misma mano, dirigiendo la "
                           "frase hacia el último.",
}

ALL_RESOURCES = list(RESOURCE_DESCRIPTIONS.keys())

# Recursos que fuerzan una mano concreta (coincide con lo que exige el
# detector correspondiente en expressive_resources.py)
FORCED_HAND = {
    "apoyatura": "RH",
    "anticipacion": "RH",
    "escapada": "RH",
    "nota_de_paso_cromatica": "RH",
    "dominante_secundaria": "LH",
    "sexta_napolitana": "LH",
    "pedal_armonico": "LH",
}

# Recursos que aceptan --beat para elegir el pulso objetivo
BEAT_RESOURCES = {
    "sincopa", "contratiempo", "floreo", "apoyatura", "anticipacion",
    "escapada", "nota_de_paso_cromatica", "acorde_disminuido",
    "acorde_prestado", "dominante_secundaria", "sexta_napolitana",
    "retardo", "sforzando", "silencio_expresivo", "crescendo_dirigido",
}

# Recursos armónicos que necesitan la tonalidad estimada (AnalysisContext)
NEEDS_KEY = {"dominante_secundaria", "acorde_prestado", "sexta_napolitana"}

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


@dataclass
class AnalysisContext:
    key_tonic: int            # clase de altura 0-11
    key_mode: str              # "mayor" | "menor"
    scale_pcs: set              # clases de altura diatónicas a la tonalidad estimada


# ── Carga de MIDI, separación de manos y eventos no-nota ─────────────────

def load_notes(path: str):
    """Igual que en expressive_resources.py, pero además conserva todo
    mensaje que no sea nota (meta-eventos incluido set_tempo,
    control_change, etc.) con su tick absoluto y su pista de origen, para
    poder reconstruir el fichero tras modificar las notas o el tempo."""
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


# ── Utilidades armónicas y de agrupación (idénticas a expressive_resources.py) ─

def pitch_classes(pitches):
    return set(p % 12 for p in pitches)


def notes_sounding_at(notes, tick, exclude=None):
    return [n for n in notes if n.start_tick <= tick < n.end_tick and n is not exclude]


def group_by_onset(hand_notes):
    """Agrupa las notas que atacan exactamente en el mismo instante
    (acordes) en un único evento. Devuelve [(start_tick, pitches, end_tick)]."""
    groups = {}
    for n in hand_notes:
        groups.setdefault(n.start_tick, []).append(n)
    onsets = []
    for tick in sorted(groups.keys()):
        ns = groups[tick]
        pitches = sorted(set(n.pitch for n in ns))
        end_tick = max(n.end_tick for n in ns)
        onsets.append((tick, pitches, end_tick))
    return onsets


def describe_pitches(pitches):
    names = [midi_name(p) for p in sorted(pitches)]
    label = "-".join(names)
    subject = "La nota" if len(names) == 1 else "El acorde"
    return f"{subject} {label}", label


NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def midi_name(pitch: int) -> str:
    octave = pitch // 12 - 1
    return f"{NOTE_NAMES[pitch % 12]}{octave}"


# ── Estimación de tonalidad (Krumhansl-Schmuckler simplificado) ──────────

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(notes):
    weights = np.zeros(12)
    for n in notes:
        weights[n.pitch % 12] += max(n.duration_ticks, 1)
    if weights.sum() == 0 or weights.std() == 0:
        return 0, "mayor", set(range(12))

    best_score, best = -1e9, (0, "mayor")
    for tonic in range(12):
        rotated = np.roll(weights, -tonic)
        for profile, mode in ((KRUMHANSL_MAJOR, "mayor"), (KRUMHANSL_MINOR, "menor")):
            corr = np.corrcoef(rotated, profile)[0, 1]
            if np.isnan(corr):
                continue
            if corr > best_score:
                best_score, best = corr, (tonic, mode)

    tonic, mode = best
    if mode == "mayor":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else:
        intervals = [0, 2, 3, 5, 7, 8, 10, 11]
    scale_pcs = set((tonic + i) % 12 for i in intervals)
    return tonic, mode, scale_pcs


# ── Helpers comunes de los inyectores ─────────────────────────────────────

def quarters_per_beat_of(grid, num, den):
    return 1.5 if grid.is_compound(num, den) else (4.0 / den)


def beat_attacks_in_bar(hand_notes, bar_start, bar_len, quarters_per_beat, tpb, tol=0.08):
    """Notas que atacan (con tolerancia `tol`, en fracción de pulso) justo
    sobre un pulso dentro de [bar_start, bar_start+bar_len). Devuelve
    [(beat_idx_1indexed, Note), ...] ordenado por beat_idx."""
    out = []
    for n in hand_notes:
        if not (bar_start <= n.start_tick < bar_start + bar_len):
            continue
        rel = (n.start_tick - bar_start) / tpb
        beat_idx = round(rel / quarters_per_beat) + 1
        if abs(rel - (beat_idx - 1) * quarters_per_beat) < tol * quarters_per_beat:
            out.append((beat_idx, n))
    return sorted(out, key=lambda c: c[0])


def notes_in_bar_at_beat(hand_notes, bar_start, bar_len, quarters_per_beat, tpb, beat):
    """Notas de una mano cuyo ataque cae dentro del compás, filtradas por
    número de pulso (1-indexado) si `beat` no es None."""
    out = []
    for n in hand_notes:
        if not (bar_start <= n.start_tick < bar_start + bar_len):
            continue
        if beat is not None:
            rel = (n.start_tick - bar_start) / tpb
            beat_idx = round(rel / quarters_per_beat) + 1
            if beat_idx != beat:
                continue
        out.append(n)
    return out


def pick_candidate(candidates, beat):
    if beat is not None:
        candidates = [c for c in candidates if c[0] == beat]
    if not candidates:
        return None
    return sorted(candidates, key=lambda c: c[0])[0]


def regenerate_span(notes, hand, span_start, span_end, n_onsets):
    """Elimina las notas de esa mano dentro de [span_start, span_end) y
    las sustituye por n_onsets ataques equiespaciados, reutilizando
    cíclicamente las alturas/velocidad/canal/pista de las notas que había
    ahí (o de la nota más próxima anterior si no había ninguna). Devuelve
    la lista de notas nuevas, o None si no hay ninguna referencia posible."""
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    in_span = [n for n in hand_notes if span_start <= n.start_tick < span_end]
    if not in_span:
        prior = [n for n in hand_notes if n.start_tick < span_end]
        if not prior:
            return None
        in_span = prior[-1:]
    ref_pitches = [n.pitch for n in in_span] or [60]
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

    unit = (span_end - span_start) / n_onsets
    new_notes = []
    for i in range(n_onsets):
        start = int(round(span_start + i * unit))
        end = int(round(span_start + (i + 1) * unit))
        pitch = ref_pitches[i % len(ref_pitches)]
        nn = Note(pitch=pitch, start_tick=start, end_tick=end, velocity=ref_vel,
                  channel=ref_channel, hand=hand, track_idx=ref_track)
        notes.append(nn)
        new_notes.append(nn)
    return new_notes


# ── Inyectores: rítmicos ──────────────────────────────────────────────────

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
    # el ataque puede ser un acorde: hay que mover TODAS las notas
    # simultáneas, no solo la elegida, o el resto quedaría anclada y
    # rompería la ligadura sobre el tiempo fuerte
    group = [m for m in hand_notes if m.start_tick == beat_tick]
    half = max(1, beat_ticks // 2)
    new_start = max(0, beat_tick - half)

    for m in hand_notes:
        if m in group:
            continue
        if m.start_tick <= new_start < m.end_tick:
            m.end_tick = new_start

    orig_pitch = target.pitch
    subject, _ = describe_pitches([m.pitch for m in group])
    for m in group:
        m.start_tick = new_start
    msg = (f"Compás {bar_num}, mano {hand}: {subject} adelanta su ataque, "
           f"del tiempo {beat_idx} a una posición débil justo antes, y queda sostenida "
           f"sobre el tiempo {beat_idx} sin nuevo ataque ahí -> síncopa.")
    return group, msg


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
    # el ataque puede ser un acorde: hay que trasladar TODAS las notas
    # simultáneas, o el resto seguiría sonando en el tiempo fuerte y no
    # quedaría en silencio
    group = [m for m in hand_notes if m.start_tick == beat_tick]
    new_start = beat_tick + half
    max_end = beat_tick + beat_ticks

    for m in hand_notes:
        if m in group:
            continue
        if m.start_tick < beat_tick <= m.end_tick:
            m.end_tick = beat_tick

    orig_pitch = target.pitch
    subject, _ = describe_pitches([m.pitch for m in group])
    for m in group:
        new_end = min(m.end_tick, max_end)
        if new_end <= new_start:
            new_end = new_start + max(1, half // 2)
        m.start_tick = new_start
        m.end_tick = new_end
    msg = (f"Compás {bar_num}, mano {hand}: {subject} se traslada del "
           f"tiempo {beat_idx} al 'y' de ese tiempo, dejando el pulso fuerte en "
           f"silencio y sin sostenerse sobre el siguiente pulso -> contratiempo.")
    return group, msg


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

    new_notes = regenerate_span(notes, hand, span_start, span_end, max(4, (span_end - span_start) // unit_ticks))
    if new_notes is None:
        return None, ("No hay ninguna nota de esa mano (ni antes ni en ese compás) que "
                       "sirva de referencia de altura. Prueba otra mano con --hand.")

    bars_txt = f"{bar_num}" if last_bar == bar_num else f"{bar_num}-{last_bar}"
    msg = (f"Compases {bars_txt}, mano {hand}: se han regenerado {len(new_notes)} "
           f"ataques equiespaciados cada {hemiola_unit:g} negra(s), en lugar de la "
           f"subdivisión natural de {num}/{den} -> hemiola.")
    return new_notes, msg


def find_tempo_track(other_events, mid):
    for idx in range(len(mid.tracks)):
        if any(m.type == "set_tempo" for _, m in other_events.get(idx, [])):
            return idx
    return 0


def tempo_before(other_events, track_idx, tick, default=500000):
    best, best_tick = default, -1
    for t, m in other_events.get(track_idx, []):
        if m.type == "set_tempo" and t <= tick and t > best_tick:
            best, best_tick = m.tempo, t
    return best


def inject_rubato(other_events, mid, grid, bar_num, direccion="ritardando",
                   n_events=4, change_ratio=0.22):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    track_idx = find_tempo_track(other_events, mid)
    base_tempo = tempo_before(other_events, track_idx, bar_start)

    other_events[track_idx] = [(t, m) for t, m in other_events.get(track_idx, [])
                                if not (m.type == "set_tempo" and bar_start <= t < bar_start + bar_len)]
    step = bar_len / max(1, n_events - 1)
    tempos = []
    for i in range(n_events):
        frac = i / (n_events - 1)
        factor = (1 - change_ratio * frac) if direccion == "acelerando" else (1 + change_ratio * frac)
        tempo = max(20000, int(round(base_tempo * factor)))
        tick = int(round(bar_start + step * i))
        other_events.setdefault(track_idx, []).append((tick, mido.MetaMessage("set_tempo", tempo=tempo, time=0)))
        tempos.append((tick, tempo))

    bpm1, bpm2 = 60_000_000 / tempos[0][1], 60_000_000 / tempos[-1][1]
    msg = (f"Compás {bar_num}: se codifica una curva de tempo continuada de "
           f"{bpm1:.0f} a {bpm2:.0f} BPM a lo largo de {n_events} cambios "
           f"({direccion}) -> rubato.")
    return tempos, msg


def inject_agogica(other_events, mid, grid, bar_num, kind="alargamiento", deviation_ratio=0.09):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    track_idx = find_tempo_track(other_events, mid)
    base_tempo = tempo_before(other_events, track_idx, bar_start)
    mid_tick = bar_start + bar_len // 2

    other_events[track_idx] = [(t, m) for t, m in other_events.get(track_idx, [])
                                if not (m.type == "set_tempo" and bar_start <= t < bar_start + bar_len)]
    factor = (1 + deviation_ratio) if kind == "alargamiento" else (1 - deviation_ratio)
    spike_tempo = max(20000, int(round(base_tempo * factor)))
    events = [(bar_start, base_tempo), (mid_tick, spike_tempo), (bar_start + bar_len - 1, base_tempo)]
    for t, tp in events:
        other_events.setdefault(track_idx, []).append((t, mido.MetaMessage("set_tempo", tempo=tp, time=0)))

    msg = (f"Compás {bar_num}: pequeño {kind} puntual del tempo que se "
           f"recupera de inmediato -> agógica.")
    return events, msg


RATIO_PAIRS = {(3, 2): "3 contra 2", (2, 3): "2 contra 3",
               (4, 3): "4 contra 3", (3, 4): "3 contra 4"}


def inject_polirritmia(notes, grid, bar_num, hand=None, patron=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info

    if patron:
        try:
            a, b = (int(x) for x in patron.split(":"))
        except Exception:
            return None, "Formato de --patron inválido; usa a:b, p.ej. 3:2."
    else:
        a, b = 3, 2
    if (a, b) not in RATIO_PAIRS:
        return None, ("Patrón no soportado; usa uno de: " +
                       ", ".join(f"{x}:{y}" for (x, y) in RATIO_PAIRS))

    rh_new = regenerate_span(notes, "RH", bar_start, bar_start + bar_len, a)
    lh_new = regenerate_span(notes, "LH", bar_start, bar_start + bar_len, b)
    if rh_new is None or lh_new is None:
        return None, ("Hace falta al menos una nota previa en RH y en LH (en ese "
                       "compás o antes) para tomar como referencia de altura/timbre.")

    msg = (f"Compás {bar_num}: RH regenerada con {a} ataques regulares y LH con {b}, "
           f"formando {RATIO_PAIRS[(a, b)]} -> polirritmia.")
    return rh_new + lh_new, msg


# ── Inyectores: melódicos ─────────────────────────────────────────────────

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

    in_bar = notes_in_bar_at_beat(hand_notes, bar_start, bar_len, qpb, grid.tpb, beat)
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
            return None, "La nota elegida es demasiado breve para alojar el floreo. Prueba otro compás/pulso."

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


def inject_anticipacion(notes, grid, bar_num, hand, beat=None, max_lead_frac=0.3):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    beat_ticks = int(round(qpb * grid.tpb))
    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    lh_onsets = group_by_onset(lh_notes)
    if not lh_onsets:
        return None, "No hay notas en LH cuya armonía se pueda anticipar."

    candidates = notes_in_bar_at_beat(rh_notes, bar_start, bar_len, qpb, grid.tpb, beat)
    if not candidates:
        return None, "No hay ninguna nota RH en ese compás/pulso."
    target = candidates[-1]

    lh_ticks = [t for t, _, _ in lh_onsets]
    idx = bisect.bisect_right(lh_ticks, target.start_tick)
    if idx >= len(lh_onsets):
        return None, "No hay ningún acorde de LH después de esa nota al que anticipar."
    next_tick, next_pitches, _ = lh_onsets[idx]

    max_gap = max(1, int(round(beat_ticks * max_lead_frac)))
    desired_start = max(0, next_tick - max(1, max_gap // 2))
    for m in rh_notes:
        if m is target:
            continue
        if m.start_tick <= desired_start < m.end_tick:
            m.end_tick = desired_start
    target.start_tick = desired_start
    if target.end_tick <= target.start_tick:
        target.end_tick = target.start_tick + 1

    current_harmony = notes_sounding_at(lh_notes, target.start_tick)
    current_pcs = pitch_classes([h.pitch for h in current_harmony]) if current_harmony else set()
    next_pcs = pitch_classes(next_pitches)
    foreign_pcs = [pc for pc in next_pcs if pc not in current_pcs]
    if not foreign_pcs:
        return None, ("La armonía que llega no aporta ninguna clase de altura distinta "
                       "de la actual; no se puede anticipar con claridad. Prueba otro compás.")

    orig_pitch = target.pitch
    best_pitch = min((pc + 12 * k for pc in foreign_pcs for k in range(orig_pitch // 12 - 1, orig_pitch // 12 + 2)),
                      key=lambda p: abs(p - orig_pitch))
    target.pitch = best_pitch

    bar_n, _, _, _, _, _ = grid.locate(target.start_tick)
    msg = (f"Compás {bar_n}, mano RH: {midi_name(best_pitch)} se adelanta "
           f"{(next_tick - target.start_tick) / grid.tpb:.2f} negra(s) antes de que "
           f"LH cambie a un acorde que ya la contiene -> anticipación.")
    return target, msg


def inject_escapada(notes, grid, bar_num, hand, beat=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    if len(rh_notes) < 3 or not lh_notes:
        return None, "Hacen falta al menos 3 notas en RH y notas en LH."

    idx_candidates = []
    for i in range(1, len(rh_notes) - 1):
        n1 = rh_notes[i]
        if not (bar_start <= n1.start_tick < bar_start + bar_len):
            continue
        if beat is not None:
            rel = (n1.start_tick - bar_start) / grid.tpb
            b_idx = round(rel / qpb) + 1
            if b_idx != beat:
                continue
        idx_candidates.append(i)
    if not idx_candidates:
        return None, "No hay una nota RH con vecinas a ambos lados en ese compás/pulso."

    for i in idx_candidates:
        n0, n1, n2 = rh_notes[i - 1], rh_notes[i], rh_notes[i + 1]

        harmony1 = notes_sounding_at(lh_notes, n1.start_tick)
        if not harmony1:
            continue
        pcs1 = pitch_classes([h.pitch for h in harmony1])

        step1 = None
        for cand_step in (1, 2, -1, -2):
            if (n0.pitch + cand_step) % 12 not in pcs1:
                step1 = cand_step
                break
        if step1 is None:
            continue

        harmony2 = notes_sounding_at(lh_notes, n2.start_tick) or harmony1
        pcs2 = pitch_classes([h.pitch for h in harmony2])
        direction2 = -1 if step1 > 0 else 1
        best = None
        for semis in range(3, 16):
            cand = (n0.pitch + step1) + direction2 * semis
            if cand % 12 in pcs2:
                best = cand
                break
        if best is None:
            continue

        n1.pitch = n0.pitch + step1
        n2.pitch = best
        bar_n, _, _, _, _, _ = grid.locate(n1.start_tick)
        msg = (f"Compás {bar_n}: tras {midi_name(n0.pitch)}, {midi_name(n1.pitch)} "
               f"(ajena a LH) se alcanza por grado conjunto y salta en dirección "
               f"opuesta a {midi_name(n2.pitch)} (nota del acorde) -> escapada.")
        return [n1, n2], msg

    return None, ("No se encontró, en ese compás/pulso, una combinación válida de "
                  "vecina ajena + salto en dirección opuesta hacia el acorde; "
                  "prueba otro compás/pulso.")


def inject_nota_de_paso_cromatica(notes, grid, bar_num, hand, beat=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    if len(rh_notes) < 3 or not lh_notes:
        return None, "Hacen falta al menos 3 notas en RH y notas en LH."

    idx_candidates = []
    for i in range(1, len(rh_notes) - 1):
        n1 = rh_notes[i]
        if not (bar_start <= n1.start_tick < bar_start + bar_len):
            continue
        if beat is not None:
            rel = (n1.start_tick - bar_start) / grid.tpb
            b_idx = round(rel / qpb) + 1
            if b_idx != beat:
                continue
        idx_candidates.append(i)
    if not idx_candidates:
        return None, "No hay una nota RH con vecinas a ambos lados en ese compás/pulso."

    for i in idx_candidates:
        n0, n1, n2 = rh_notes[i - 1], rh_notes[i], rh_notes[i + 1]

        harmony1 = notes_sounding_at(lh_notes, n1.start_tick)
        if not harmony1:
            continue
        pcs1 = pitch_classes([h.pitch for h in harmony1])
        harmony0 = notes_sounding_at(lh_notes, n0.start_tick) or harmony1
        pcs0 = pitch_classes([h.pitch for h in harmony0])
        if n0.pitch % 12 not in pcs0:
            continue

        for direction in (1, -1):
            cand1 = n0.pitch + direction
            if cand1 % 12 in pcs1:
                continue
            harmony2 = notes_sounding_at(lh_notes, n2.start_tick) or harmony1
            pcs2 = pitch_classes([h.pitch for h in harmony2])
            cand2 = None
            for step2 in (1, 2):
                c2 = cand1 + direction * step2
                if c2 % 12 in pcs2:
                    cand2 = c2
                    break
            if cand2 is not None:
                n1.pitch = cand1
                n2.pitch = cand2
                direction_txt = "ascendente" if direction > 0 else "descendente"
                bar_n, _, _, _, _, _ = grid.locate(n1.start_tick)
                msg = (f"Compás {bar_n}: entre {midi_name(n0.pitch)} y "
                       f"{midi_name(n2.pitch)} (del acorde), {midi_name(n1.pitch)} "
                       f"pasa por grado conjunto {direction_txt} fuera de la "
                       f"armonía -> nota de paso cromática.")
                return [n1, n2], msg
    return None, ("No se encontró, en ese compás/pulso, una nota RH anterior que ya "
                  "sea del acorde de LH junto con una vecina cromática y su "
                  "resolución; prueba otro compás/pulso.")


# ── Inyectores: armónicos ─────────────────────────────────────────────────

def inject_acorde_disminuido(notes, grid, bar_num, hand, beat=None, septima=False):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    candidates = beat_attacks_in_bar(hand_notes, bar_start, bar_len, qpb, grid.tpb)
    picked = pick_candidate(candidates, beat)
    if not picked:
        return None, "No hay ninguna nota de esa mano atacando justo sobre un pulso en ese compás."
    beat_idx, target = picked
    root = target.pitch
    intervals = (0, 3, 6, 9) if septima else (0, 3, 6)

    for m in list(notes):
        if m is target:
            continue
        if m.hand == hand and m.start_tick == target.start_tick:
            notes.remove(m)

    new_notes = [target]
    for iv in intervals[1:]:
        nn = Note(pitch=root + iv, start_tick=target.start_tick, end_tick=target.end_tick,
                  velocity=target.velocity, channel=target.channel, hand=hand, track_idx=target.track_idx)
        notes.append(nn)
        new_notes.append(nn)

    shape_name = "séptima disminuida" if septima else "tríada disminuida"
    msg = (f"Compás {bar_num}, mano {hand}: se construye una {shape_name} sobre "
           f"{midi_name(root)} apilando terceras menores -> acorde disminuido.")
    return new_notes, msg


def inject_dominante_secundaria(notes, grid, ctx, bar_num, hand, beat=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    lh_notes = sorted([n for n in notes if n.hand == "LH"], key=lambda n: n.start_tick)
    candidates = beat_attacks_in_bar(lh_notes, bar_start, bar_len, qpb, grid.tpb)
    picked = pick_candidate(candidates, beat)
    if not picked:
        return None, "No hay ningún acorde de LH atacando justo sobre un pulso en ese compás."
    beat_idx, anchor = picked
    onset_tick = anchor.start_tick
    onset_end = max(n.end_tick for n in lh_notes if n.start_tick == onset_tick)

    later = [n for n in lh_notes if n.start_tick > onset_tick]
    if not later:
        return None, "No hay ningún acorde de LH después de ese pulso al que resolver."
    next_tick = later[0].start_tick
    next_group = [n for n in lh_notes if n.start_tick == next_tick]
    next_root_pitch = min(n.pitch for n in next_group)
    next_root = next_root_pitch % 12

    root_pc = (next_root - 5) % 12
    base_octave = (anchor.pitch // 12) * 12
    root_pitch = base_octave + root_pc
    while root_pitch - anchor.pitch > 6:
        root_pitch -= 12
    while anchor.pitch - root_pitch > 6:
        root_pitch += 12

    for m in list(notes):
        if m.hand == "LH" and m.start_tick == onset_tick:
            notes.remove(m)
    new_notes = []
    for iv in (0, 4, 7, 10):
        nn = Note(pitch=root_pitch + iv, start_tick=onset_tick, end_tick=onset_end,
                  velocity=anchor.velocity, channel=anchor.channel, hand="LH", track_idx=anchor.track_idx)
        notes.append(nn)
        new_notes.append(nn)

    msg = (f"Compás {bar_num}: se sustituye el acorde de LH por una séptima de "
           f"dominante sobre {midi_name(root_pitch)}, ajena a la tonalidad estimada "
           f"({NOTE_NAMES[ctx.key_tonic]} {ctx.key_mode}), que resuelve una 4a justa "
           f"después hacia {midi_name(next_root_pitch)} -> dominante secundaria.")
    return new_notes, msg


def inject_acorde_prestado(notes, grid, ctx, bar_num, hand, beat=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    candidates = beat_attacks_in_bar(hand_notes, bar_start, bar_len, qpb, grid.tpb)
    picked = pick_candidate(candidates, beat)
    if not picked:
        return None, "No hay ninguna nota de esa mano atacando justo sobre un pulso en ese compás."
    beat_idx, target = picked
    root = target.pitch

    shapes = [(3, 7), (4, 7)] if ctx.key_mode == "mayor" else [(4, 7), (3, 7)]
    chosen = None
    for thirds, fifth in shapes:
        pcs = {root % 12, (root + thirds) % 12, (root + fifth) % 12}
        if pcs - ctx.scale_pcs:
            chosen = (thirds, fifth)
            break
    if chosen is None:
        return None, ("No se ha encontrado, a partir de esa nota, una tríada con "
                       "alguna clase de altura ajena a la tonalidad estimada.")
    thirds, fifth = chosen

    for m in list(notes):
        if m is target:
            continue
        if m.hand == hand and m.start_tick == target.start_tick:
            notes.remove(m)
    new_notes = [target]
    for iv in (thirds, fifth):
        nn = Note(pitch=root + iv, start_tick=target.start_tick, end_tick=target.end_tick,
                  velocity=target.velocity, channel=target.channel, hand=hand, track_idx=target.track_idx)
        notes.append(nn)
        new_notes.append(nn)

    msg = (f"Compás {bar_num}, mano {hand}: se construye una tríada sobre "
           f"{midi_name(root)} con al menos una nota ajena a la tonalidad estimada "
           f"({NOTE_NAMES[ctx.key_tonic]} {ctx.key_mode}) -> acorde prestado.")
    return new_notes, msg


def inject_sexta_napolitana(notes, grid, ctx, bar_num, hand, beat=None):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    lh_notes = sorted([n for n in notes if n.hand == "LH"], key=lambda n: n.start_tick)
    candidates = beat_attacks_in_bar(lh_notes, bar_start, bar_len, qpb, grid.tpb)
    picked = pick_candidate(candidates, beat)
    if not picked:
        return None, "No hay ningún acorde de LH atacando justo sobre un pulso en ese compás."
    beat_idx, anchor = picked
    onset_tick = anchor.start_tick
    onset_end = max(n.end_tick for n in lh_notes if n.start_tick == onset_tick)

    napolitan_root = (ctx.key_tonic + 1) % 12
    third_pc = (napolitan_root + 4) % 12
    fifth_pc = (napolitan_root + 7) % 12

    def pitch_at_or_above(base_pitch, pc):
        return base_pitch + ((pc - base_pitch) % 12)

    bass_pitch = anchor.pitch - ((anchor.pitch - third_pc) % 12)
    root_pitch = pitch_at_or_above(bass_pitch + 1, napolitan_root)
    fifth_pitch = pitch_at_or_above(root_pitch + 1, fifth_pc)

    for m in list(notes):
        if m.hand == "LH" and m.start_tick == onset_tick:
            notes.remove(m)
    new_notes = []
    for p in (bass_pitch, root_pitch, fifth_pitch):
        nn = Note(pitch=p, start_tick=onset_tick, end_tick=onset_end, velocity=anchor.velocity,
                  channel=anchor.channel, hand="LH", track_idx=anchor.track_idx)
        notes.append(nn)
        new_notes.append(nn)

    msg = (f"Compás {bar_num}: se construye una tríada mayor sobre {midi_name(root_pitch)} "
           f"(segundo grado descendido) en primera inversión, con {midi_name(bass_pitch)} "
           f"en el bajo -> sexta napolitana (respecto a {NOTE_NAMES[ctx.key_tonic]} "
           f"{ctx.key_mode}).")
    return new_notes, msg


def inject_pedal_armonico(notes, grid, bar_num, hand, run=4):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    run = max(3, run)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    onset_ticks = sorted(set(n.start_tick for n in hand_notes if n.start_tick >= bar_start))
    if len(onset_ticks) < run:
        return None, (f"No hay al menos {run} ataques de esa mano a partir de ese "
                       f"compás para sostener el pedal (usa --run para pedir menos).")
    chosen_ticks = onset_ticks[:run]
    groups = [[n for n in hand_notes if n.start_tick == t] for t in chosen_ticks]
    pedal_pitch = min(n.pitch for n in groups[0])

    upper_pcs = set()
    for g in groups:
        bass_note = min(g, key=lambda n: n.pitch)
        bass_note.pitch = pedal_pitch
        for n in g:
            if n is not bass_note:
                upper_pcs.add(n.pitch % 12)
    if len(upper_pcs) < 2:
        for i, g in enumerate(groups):
            uppers = [n for n in g if n.pitch != pedal_pitch]
            if uppers:
                uppers[0].pitch += (i % 3) - 1
                if uppers[0].pitch == pedal_pitch:
                    uppers[0].pitch += 1

    bar_n, _, _, _, _, _ = grid.locate(chosen_ticks[0])
    changed = [n for g in groups for n in g]
    msg = (f"Compás {bar_n}, mano {hand}: el bajo {midi_name(pedal_pitch)} se "
           f"repite sin cambiar durante {run} ataques mientras las voces "
           f"superiores se mueven -> pedal armónico.")
    return changed, msg


def inject_retardo(notes, grid, bar_num, hand, beat=None, max_step=2):
    other = "LH" if hand == "RH" else "RH"
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    other_notes = sorted([n for n in notes if n.hand == other], key=lambda n: n.start_tick)
    if not other_notes:
        return None, f"No hay notas en {other} para construir el cambio de armonía."

    in_bar = notes_in_bar_at_beat(hand_notes, bar_start, bar_len, qpb, grid.tpb, beat)
    if not in_bar:
        return None, f"No hay ninguna nota de {hand} en ese compás/pulso."
    n = in_bar[0]

    other_onsets = group_by_onset(other_notes)
    other_ticks = [t for t, _, _ in other_onsets]
    change_idx = bisect.bisect_right(other_ticks, n.start_tick)
    if change_idx >= len(other_onsets):
        return None, f"No hay ningún ataque de {other} después del inicio de esa nota."
    change_tick, change_pitches, _ = other_onsets[change_idx]

    tie_amount = max(1, int(round(0.15 * qpb * grid.tpb)))
    new_end = max(n.end_tick, change_tick + tie_amount)

    # al ligar n sobre el cambio de armonía puede tragarse notas intermedias
    # de la misma mano: se eliminan (pero nunca notas que compartan el
    # mismo ataque que n, que son compañeras de acorde, no intermedias), y
    # la resolución pasa a ser la primera nota de esa mano que quede
    # después del nuevo final de n
    for m in list(hand_notes):
        if m is n:
            continue
        if n.start_tick < m.start_tick < new_end:
            hand_notes.remove(m)
            notes.remove(m)
    n.end_tick = new_end

    later = [m for m in hand_notes if m.start_tick > n.start_tick and m.start_tick >= n.end_tick]
    if not later:
        return None, (f"Esa nota de {hand} no tiene, tras ligarla sobre el cambio de "
                       f"armonía, ninguna nota siguiente en la misma mano para resolver.")
    nxt = later[0]

    harmony_before = notes_sounding_at(other_notes, n.start_tick)
    if not harmony_before:
        return None, f"No suena ninguna nota de {other} en el instante en que empieza esta nota."
    pcs_before = pitch_classes([h.pitch for h in harmony_before])

    pcs_after = pitch_classes(change_pitches)
    harmony_res = notes_sounding_at(other_notes, nxt.start_tick)
    pcs_res = pitch_classes([h.pitch for h in harmony_res]) if harmony_res else pcs_after

    # busca conjuntamente una nota de preparación (consonante con pcs_before,
    # disonante con pcs_after) y un paso de resolución hacia pcs_res; probar
    # todas las combinaciones da muchas más posibilidades cuando la mano de
    # apoyo es poco densa (p.ej. una melodía monofónica con una sola clase
    # de altura sonando en cada instante)
    orig_pitch = n.pitch
    solution = None
    pcs_before_sorted = sorted(pcs_before, key=lambda pc: min((pc - orig_pitch) % 12, (orig_pitch - pc) % 12))
    for pc_before in pcs_before_sorted:
        cand_n = orig_pitch - ((orig_pitch - pc_before) % 12)
        if cand_n % 12 in pcs_after:
            continue  # seguiría siendo consonante tras el cambio: no hay disonancia que resolver
        for cand_step in (1, 2, -1, -2):
            cand_res = cand_n + cand_step
            if cand_res % 12 in pcs_res:
                solution = (cand_n, cand_res)
                break
        if solution:
            break
    if solution is None:
        return None, "No se encontró una resolución por grado conjunto dentro del acorde siguiente."
    n.pitch, nxt.pitch = solution

    if n.pitch % 12 in pcs_after:
        change_group = [m for m in other_notes if m.start_tick == change_tick]
        clash = next((m for m in change_group if m.pitch % 12 == n.pitch % 12), None)
        if clash is not None:
            clash.pitch += 1

    bar_c, _, _, _, _, _ = grid.locate(change_tick)
    direction_txt = "descendente" if nxt.pitch < n.pitch else "ascendente"
    msg = (f"Compás {bar_c}: {midi_name(n.pitch)} ({hand}) queda preparada como "
           f"consonancia y ligada sobre el cambio de armonía de {other}, "
           f"convirtiéndose en disonancia que resuelve {direction_txt} a "
           f"{midi_name(nxt.pitch)} -> retardo (suspensión).")
    return [n, nxt], msg


# ── Inyectores: dinámicos / articulatorios ────────────────────────────────

def inject_sforzando(notes, grid, bar_num, hand, beat=None, min_diff=26, window=6):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    in_bar = notes_in_bar_at_beat(hand_notes, bar_start, bar_len, qpb, grid.tpb, beat)
    if not in_bar:
        return None, f"No hay ninguna nota de {hand} en ese compás/pulso."
    target = in_bar[0]
    idx = hand_notes.index(target)
    lo, hi = max(0, idx - window // 2), min(len(hand_notes), idx + window // 2 + 1)
    local = [m.velocity for k, m in enumerate(hand_notes[lo:hi], start=lo) if k != idx]
    avg_local = sum(local) / len(local) if local else target.velocity

    orig_vel = target.velocity
    new_vel = min(127, int(round(avg_local + min_diff + 4)))
    target.velocity = new_vel

    bar_n, _, _, _, _, _ = grid.locate(target.start_tick)
    msg = (f"Compás {bar_n}, mano {hand}: {midi_name(target.pitch)} sube de "
           f"velocidad {orig_vel} a {new_vel}, muy por encima de la media "
           f"local ({avg_local:.0f}) -> sforzando.")
    return target, msg


def inject_silencio_expresivo(notes, grid, bar_num, hand, beat=None, min_ratio=2.2, min_gap_beats=0.75):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    qpb = quarters_per_beat_of(grid, num, den)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    onsets = group_by_onset(hand_notes)
    if len(onsets) < 4:
        return None, f"Hacen falta al menos 4 ataques de {hand} en toda la pieza."

    in_bar_idx = []
    for i, (t, _, _) in enumerate(onsets):
        if i < 1 or not (bar_start <= t < bar_start + bar_len):
            continue
        if beat is not None:
            rel = (t - bar_start) / grid.tpb
            b_idx = round(rel / qpb) + 1
            if b_idx != beat:
                continue
        in_bar_idx.append(i)
    if not in_bar_idx:
        return None, f"No hay ningún ataque de {hand} (con uno anterior) en ese compás/pulso."
    i = in_bar_idx[0]

    prev_tick = onsets[i - 1][0]
    prev_end = max(m.end_tick for m in hand_notes if m.start_tick == prev_tick)
    window = []
    for k in range(max(1, i - 4), i):
        k_prev_end = max(m.end_tick for m in hand_notes if m.start_tick == onsets[k - 1][0])
        window.append(onsets[k][0] - k_prev_end)
    avg = (sum(window) / len(window)) if window and sum(window) > 0 else grid.tpb * 0.25
    needed_gap = max(int(round(avg * min_ratio)) + 1, int(round(min_gap_beats * grid.tpb)) + 1)

    target_tick = onsets[i][0]
    existing_gap = target_tick - prev_end
    delta = needed_gap - existing_gap
    if delta > 0:
        for n in hand_notes:
            if n.start_tick >= target_tick:
                n.start_tick += delta
                n.end_tick += delta

    onset_notes = [n for n in hand_notes if n.start_tick == target_tick + max(0, delta)]
    new_start = onset_notes[0].start_tick
    gap = new_start - prev_end
    bar_n, _, _, _, _, _ = grid.locate(new_start)
    subject, _ = describe_pitches([n.pitch for n in onset_notes])
    msg = (f"Compás {bar_n}, mano {hand}: antes de {subject} se abre un "
           f"silencio de {gap / grid.tpb:.2f} negra(s), muy por encima del "
           f"hueco medio reciente ({avg / grid.tpb:.2f}) -> silencio expresivo.")
    return onset_notes, msg


def inject_crescendo_dirigido(notes, grid, bar_num, hand, beat=None, run=5, total_increase=26, base=55):
    info = grid.bar_info(bar_num)
    if not info:
        return None, "Ese número de compás está fuera del rango del MIDI."
    bar_start, num, den, bar_len = info
    run = max(4, run)
    qpb = quarters_per_beat_of(grid, num, den)
    hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
    onsets = group_by_onset(hand_notes)

    start_idx = None
    for i, (t, _, _) in enumerate(onsets):
        if not (bar_start <= t < bar_start + bar_len):
            continue
        if beat is not None:
            rel = (t - bar_start) / grid.tpb
            b_idx = round(rel / qpb) + 1
            if b_idx != beat:
                continue
        start_idx = i
        break
    if start_idx is None:
        return None, f"No hay ningún ataque de {hand} en ese compás/pulso."
    end_idx = min(len(onsets), start_idx + run)
    if end_idx - start_idx < run:
        return None, (f"No hay suficientes ataques de {hand} a partir de ese punto "
                       f"(hacen falta {run}; usa --run para pedir menos).")
    chosen = onsets[start_idx:end_idx]
    n_steps = len(chosen)
    changed = []
    for k, (t, _, _) in enumerate(chosen):
        vel = min(127, base + int(round(total_increase * k / (n_steps - 1))))
        for n in hand_notes:
            if n.start_tick == t:
                n.velocity = vel
                changed.append(n)

    bar_n, _, _, _, _, _ = grid.locate(chosen[0][0])
    msg = (f"Compás {bar_n}, mano {hand}: a lo largo de {n_steps} ataques "
           f"consecutivos la velocidad sube de {base} a "
           f"{min(127, base + total_increase)}, dirigiendo la frase hacia "
           f"el último -> crescendo dirigido.")
    return changed, msg


INJECTORS = {
    "sincopa": inject_sincopa,
    "contratiempo": inject_contratiempo,
    "hemiola": inject_hemiola,
    "polirritmia": inject_polirritmia,
    "apoyatura": inject_apoyatura,
    "floreo": inject_floreo,
    "anticipacion": inject_anticipacion,
    "escapada": inject_escapada,
    "nota_de_paso_cromatica": inject_nota_de_paso_cromatica,
    "acorde_disminuido": inject_acorde_disminuido,
    "dominante_secundaria": inject_dominante_secundaria,
    "acorde_prestado": inject_acorde_prestado,
    "sexta_napolitana": inject_sexta_napolitana,
    "pedal_armonico": inject_pedal_armonico,
    "retardo": inject_retardo,
    "sforzando": inject_sforzando,
    "silencio_expresivo": inject_silencio_expresivo,
    "crescendo_dirigido": inject_crescendo_dirigido,
    # "rubato" y "agogica" se gestionan aparte (tocan tempo, no notas)
}

# ── Reconstrucción y escritura del MIDI ───────────────────────────────────

def rebuild_and_save(mid, tpb, notes, other_events, out_path):
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
    key_tonic, key_mode, _ = estimate_key(notes)

    rh = [n for n in notes if n.hand == "RH"]
    lh = [n for n in notes if n.hand == "LH"]

    print(f"{C.BOLD}Fichero:{C.RESET} {args.midi}")
    print(f"Pistas: {len(mid.tracks)}   Ticks/negra: {tpb}")
    print(f"Compases de tiempo: " +
          ", ".join(f"{ts.numerator}/{ts.denominator}@tick{ts.tick}" for ts in time_sigs))
    print(f"Tonalidad estimada: {NOTE_NAMES[key_tonic]} {key_mode}")
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
    # margen extra: permite inyectar recursos que se extiendan más allá del
    # final de la pieza, o compases pedidos justo en el límite
    grid = MetricGrid(tpb, time_sigs, last_tick + tpb * 32)

    if args.recurso in ("rubato", "agogica"):
        if args.recurso == "rubato":
            direccion = args.direccion_tempo or "ritardando"
            result, msg = inject_rubato(other_events, mid, grid, args.compas, direccion=direccion)
        else:
            kind = "alargamiento" if (args.direccion_tempo or "ritardando") == "ritardando" else "apresuramiento"
            result, msg = inject_agogica(other_events, mid, grid, args.compas, kind=kind)
        if result is None:
            print(f"{C.RED}✗{C.RESET} {msg}", file=sys.stderr)
            sys.exit(1)
        out_path = args.out or default_output_name(args.midi, args.recurso, args.compas)
        rebuild_and_save(mid, tpb, notes, other_events, out_path)
        color = RESOURCE_COLOR.get(args.recurso, "")
        print(f"{C.GREEN}✓{C.RESET} {C.BOLD}{color}[{args.recurso.upper()}]{C.RESET} {msg}")
        print(f"Guardado en: {C.BOLD}{out_path}{C.RESET}")
        return

    hand = FORCED_HAND.get(args.recurso, args.hand or "RH")

    kwargs = {}
    if args.recurso in BEAT_RESOURCES:
        kwargs["beat"] = args.beat
    if args.recurso == "floreo":
        kwargs["tipo"] = args.tipo
    if args.recurso == "apoyatura":
        kwargs["direccion"] = args.direccion
    if args.recurso == "acorde_disminuido":
        kwargs["septima"] = args.septima
    if args.recurso == "polirritmia":
        kwargs["patron"] = args.patron
    if args.recurso == "pedal_armonico":
        kwargs["run"] = args.run or 4
    if args.recurso == "crescendo_dirigido":
        kwargs["run"] = args.run or 5

    fn = INJECTORS[args.recurso]
    if args.recurso in NEEDS_KEY:
        key_tonic, key_mode, scale_pcs = estimate_key(notes)
        ctx = AnalysisContext(key_tonic=key_tonic, key_mode=key_mode, scale_pcs=scale_pcs)
        result, msg = fn(notes, grid, ctx, args.compas, hand, **kwargs)
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
                           help="Mano objetivo (por defecto RH; algunos recursos fuerzan una mano)")
    p_inject.add_argument("--beat", type=int, default=None,
                           help="Pulso objetivo dentro del compás (1-indexado)")
    p_inject.add_argument("--tipo", choices=["mordente", "grupeto", "trino"], default="mordente",
                           help="Solo floreo: tipo de adorno")
    p_inject.add_argument("--direccion", choices=["arriba", "abajo"], default=None,
                           help="Solo apoyatura: dirección de la nota vecina")
    p_inject.add_argument("--direccion-tempo", dest="direccion_tempo",
                           choices=["acelerando", "ritardando"], default=None,
                           help="Solo rubato/agogica: dirección del cambio de tempo")
    p_inject.add_argument("--septima", action="store_true",
                           help="Solo acorde_disminuido: séptima disminuida en vez de tríada")
    p_inject.add_argument("--patron", default=None,
                           help="Solo polirritmia: proporción RH:LH, p.ej. 3:2 (por defecto 3:2)")
    p_inject.add_argument("--run", type=int, default=None,
                           help="Solo pedal_armonico (ataques a sostener) y crescendo_dirigido "
                                "(ataques de la racha)")
    p_inject.add_argument("--out", default=None, help="Ruta del MIDI de salida")
    p_inject.set_defaults(func=cmd_inject)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
