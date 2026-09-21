#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║ expressive_resources.py                                              ║
# ║                                                                      ║
# ║ Detecta recursos expresivos (rítmicos, melódicos, armónicos y        ║
# ║ dinámicos) en un MIDI de piano con separación estándar de mano       ║
# ║ derecha (RH) / mano izquierda (LH), e indica por cada hallazgo:      ║
# ║ recurso, compás, intención expresiva y explicación en lenguaje       ║
# ║ natural.                                                              ║
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
# ║   (* rubato y agogica requieren una curva de tempo codificada en     ║
# ║   el MIDI; sin mensajes set_tempo no detectarán nada)                ║
# ║                                                                      ║
# ║ USO:                                                                 ║
# ║   python3 expressive_resources.py info pieza.mid                    ║
# ║   python3 expressive_resources.py list-resources                    ║
# ║   python3 expressive_resources.py analyze pieza.mid                 ║
# ║   python3 expressive_resources.py analyze pieza.mid --resources      ║
# ║       hemiola,sincopa --json                                        ║
# ║                                                                      ║
# ║ Entrada: un .mid de piano a 2 pistas (o 2 canales) con mano          ║
# ║ derecha e izquierda separadas. Si no puede identificarlas por el     ║
# ║ nombre de pista, separa por registro (mediana de altura). La         ║
# ║ tonalidad (para los recursos armónicos) se estima automáticamente    ║
# ║ mediante correlación de Krumhansl-Schmuckler.                        ║
# ║                                                                      ║
# ║ Dependencias: mido, numpy                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

import argparse
import bisect
import json
import sys
from dataclasses import dataclass, field, asdict
from fractions import Fraction

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
              "largo de varios eventos. Requiere una curva de tempo "
              "codificada en el MIDI (mensajes set_tempo).",
    "agogica": "Pequeño respiro de tempo puntual que se recupera de "
               "inmediato, para realzar una llegada. Igual que rubato, "
               "requiere una curva de tempo en el MIDI.",
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
                             "que resuelve por círculo de quintas.",
    "acorde_prestado": "Acorde con alguna nota ajena a la tonalidad "
                        "estimada, sin tritono, sugiriendo un préstamo de "
                        "la tonalidad paralela.",
    "sexta_napolitana": "Tríada mayor sobre el segundo grado descendido "
                         "en primera inversión, respecto a la tonalidad "
                         "estimada.",
    "pedal_armonico": "Nota grave sostenida o repetida sin cambiar "
                       "mientras la armonía superior se mueve por encima.",
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

# ── Modelo de datos ──────────────────────────────────────────────────────

@dataclass
class Note:
    pitch: int
    start_tick: int
    end_tick: int
    velocity: int
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
class Finding:
    resource: str
    hand: str
    compas: int
    tiempo: str          # posición dentro del compás en notación legible
    intencion: str
    explicacion: str
    confianza: float     # 0-1, heurística


# ── Carga de MIDI y separación de manos ──────────────────────────────────

def load_notes(path: str):
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat

    time_sigs = [TimeSigChange(0, 4, 4)]  # default 4/4 hasta que se indique otra
    tempo_changes = []  # (abs_tick, tempo_us_por_negra)
    notes = []

    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        open_notes = {}  # (pitch) -> (start_tick, velocity)
        for msg in track:
            abs_tick += msg.time
            if msg.type == "time_signature":
                time_sigs.append(TimeSigChange(abs_tick, msg.numerator, msg.denominator))
            elif msg.type == "set_tempo":
                tempo_changes.append((abs_tick, msg.tempo))
            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.note, msg.channel)
                if key in open_notes:
                    # retrigger sin note_off explícito (frecuente en piezas
                    # reales, p.ej. bajos repetidos con pedal): se cierra la
                    # nota anterior aquí mismo en vez de perderla en
                    # silencio, y se abre la nueva
                    prev_start, prev_vel = open_notes.pop(key)
                    if abs_tick > prev_start:
                        notes.append(Note(msg.note, prev_start, abs_tick, prev_vel, "?", track_idx))
                open_notes[key] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.note, msg.channel)
                if key in open_notes:
                    start_tick, vel = open_notes.pop(key)
                    notes.append(Note(msg.note, start_tick, abs_tick, vel, "?", track_idx))

    time_sigs = sorted(time_sigs, key=lambda t: t.tick)
    # si hay varios cambios en el mismo tick (p.ej. el 4/4 por defecto y un
    # time_signature real leído del fichero en tick 0), se queda el último
    dedup = [time_sigs[0]]
    for ts in time_sigs[1:]:
        if ts.tick == dedup[-1].tick:
            dedup[-1] = ts
        else:
            dedup.append(ts)
    time_sigs = dedup

    tempo_changes.sort(key=lambda t: t[0])
    notes.sort(key=lambda n: n.start_tick)
    return tpb, time_sigs, tempo_changes, notes, mid


def assign_hands(mid: "mido.MidiFile", notes):
    """Asigna RH/LH. Primero por nombre de pista, si no por registro."""
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
        # fallback: pista de menor índice con notas -> RH, siguiente -> LH,
        # pero solo si el registro medio de la primera es más agudo
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
        # último recurso: separar por pitch mediano global (C4=60)
        pitches = [n.pitch for n in notes]
        split = float(np.median(pitches)) if pitches else 60
        for n in notes:
            n.hand = "RH" if n.pitch >= split else "LH"
        return notes

    for n in notes:
        n.hand = hand_by_track.get(n.track_idx, "RH" if n.pitch >= 60 else "LH")
    return notes


# ── Rejilla métrica ───────────────────────────────────────────────────────

class MetricGrid:
    """Convierte ticks absolutos en (compás, posición-en-beats-de-negra,
    pulso-fuerte-mas-cercano) teniendo en cuenta cambios de compás."""

    def __init__(self, tpb, time_sigs, last_tick):
        self.tpb = tpb
        self.time_sigs = time_sigs
        self.bar_starts = []  # lista de (tick_inicio, numerator, denominator, bar_number_inicial)
        self._build(last_tick)

    def _bar_len_ticks(self, numerator, denominator):
        # longitud de compás en ticks = numerator * (4/denominator) * tpb
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
        """Devuelve (bar_number, numerator, denominator, beat_pos_negras,
        bar_start_tick, bar_len_ticks)."""
        lo, hi = 0, len(self.bars) - 1
        idx = 0
        for i, (bt, num, den, bn) in enumerate(self.bars):
            if bt <= tick:
                idx = i
            else:
                break
        bar_start, num, den, bar_num = self.bars[idx]
        bar_len = self._bar_len_ticks(num, den)
        beat_pos = (tick - bar_start) / self.tpb  # en negras
        return bar_num, num, den, beat_pos, bar_start, bar_len

    def is_compound(self, numerator, denominator):
        return denominator == 8 and numerator % 3 == 0 and numerator > 3

    def beat_label(self, numerator, denominator, beat_pos_quarters):
        """Etiqueta legible tipo '2.5' (tiempo 2, mitad) para mostrar."""
        quarters_per_beat = 4.0 / denominator
        if self.is_compound(numerator, denominator):
            quarters_per_beat = 1.5
        beat_idx = beat_pos_quarters / quarters_per_beat
        return f"{beat_idx + 1:.2f}"


# ── Utilidades armónicas ─────────────────────────────────────────────────

def pitch_classes(pitches):
    return set(p % 12 for p in pitches)


def notes_sounding_at(notes, tick, exclude=None):
    return [n for n in notes if n.start_tick <= tick < n.end_tick and n is not exclude]


def group_by_onset(hand_notes):
    """Agrupa las notas de una mano que atacan exactamente en el mismo
    instante (acordes) en un único evento. Devuelve una lista ordenada de
    (start_tick, pitches_ordenados, end_tick_maximo)."""
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
    """('La nota Sol4', 'Sol4') o ('El acorde Do1-Sol1-Do2', 'Do1-Sol1-Do2')."""
    names = [midi_name(p) for p in sorted(pitches)]
    label = "-".join(names)
    subject = "La nota" if len(names) == 1 else "El acorde"
    return f"{subject} {label}", label


# ── Estimación de tonalidad (Krumhansl-Schmuckler simplificado) ──────────

KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                             2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                             2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(notes):
    """Estima tónica y modo correlacionando el histograma de clases de
    altura (ponderado por duración) con los perfiles tonales de
    Krumhansl-Schmuckler. Devuelve (tonica_pc, modo, clases_diatonicas)."""
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
        # menor natural + sensible elevada (7º grado de la menor armónica),
        # muy frecuente, para no marcarla como "ajena" por defecto
        intervals = [0, 2, 3, 5, 7, 8, 10, 11]
    scale_pcs = set((tonic + i) % 12 for i in intervals)
    return tonic, mode, scale_pcs


@dataclass
class AnalysisContext:
    """Información compartida entre detectores, calculada una sola vez."""
    tempo_changes: list       # [(tick, microsegundos_por_negra), ...]
    key_tonic: int            # clase de altura 0-11
    key_mode: str             # "mayor" | "menor"
    scale_pcs: set            # clases de altura diatónicas a la tonalidad estimada


# ── Detectores ────────────────────────────────────────────────────────────

def detect_sincopa(notes, grid, ctx, min_off_frac=0.2):
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        for start_tick, pitches, end_tick in group_by_onset(hand_notes):
            bar_num, num, den, beat_pos, bar_start, bar_len = grid.locate(start_tick)
            quarters_per_beat = 1.5 if grid.is_compound(num, den) else (4.0 / den)
            offset_within_beat = (beat_pos % quarters_per_beat) / quarters_per_beat
            # nota que empieza claramente fuera del pulso (no en el 0)
            if offset_within_beat < min_off_frac or offset_within_beat > (1 - min_off_frac):
                continue
            # ¿se sostiene sobre el siguiente pulso fuerte?
            next_pulse_beatpos = (beat_pos // quarters_per_beat + 1) * quarters_per_beat
            next_pulse_tick = bar_start + int(round(next_pulse_beatpos * grid.tpb))
            if next_pulse_tick < end_tick and next_pulse_tick > start_tick:
                subject, _ = describe_pitches(pitches)
                findings.append(Finding(
                    resource="sincopa",
                    hand=hand,
                    compas=bar_num,
                    tiempo=grid.beat_label(num, den, beat_pos),
                    intencion="Generar tensión rítmica anticipando el acento y "
                              "vaciando el tiempo fuerte de un ataque propio.",
                    explicacion=(f"{subject} entra en el tiempo "
                                 f"{grid.beat_label(num, den, beat_pos)} (posición débil) "
                                 f"y se prolonga sobre el siguiente pulso fuerte, que "
                                 f"queda sin ataque nuevo."),
                    confianza=0.65,
                ))
    return findings


def detect_contratiempo(notes, grid, ctx, min_off_frac=0.2):
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        for start_tick, pitches, end_tick in group_by_onset(hand_notes):
            bar_num, num, den, beat_pos, bar_start, bar_len = grid.locate(start_tick)
            quarters_per_beat = 1.5 if grid.is_compound(num, den) else (4.0 / den)
            offset_within_beat = (beat_pos % quarters_per_beat) / quarters_per_beat
            if offset_within_beat < min_off_frac or offset_within_beat > (1 - min_off_frac):
                continue
            # el pulso fuerte inmediatamente anterior debe estar en silencio en esta mano
            prev_pulse_beatpos = (beat_pos // quarters_per_beat) * quarters_per_beat
            prev_pulse_tick = bar_start + int(round(prev_pulse_beatpos * grid.tpb))
            sounding_before = [m for m in hand_notes
                                if m.start_tick <= prev_pulse_tick < m.end_tick]
            if sounding_before:
                continue  # hay ligadura desde antes -> no es contratiempo puro
            # el contratiempo no se sostiene sobre el siguiente pulso fuerte
            # (si lo hiciera, es más propiamente una síncopa)
            next_pulse_beatpos = (beat_pos // quarters_per_beat + 1) * quarters_per_beat
            next_pulse_tick = bar_start + int(round(next_pulse_beatpos * grid.tpb))
            if end_tick > next_pulse_tick:
                continue
            subject, _ = describe_pitches(pitches)
            findings.append(Finding(
                resource="contratiempo",
                hand=hand,
                compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Dar impulso hacia adelante dejando el tiempo fuerte "
                          "vacío y atacando solo en el 'y'.",
                explicacion=(f"{subject} ataca en el tiempo "
                             f"{grid.beat_label(num, den, beat_pos)}, tras un silencio "
                             f"en el pulso fuerte anterior de esta mano (sin ligadura "
                             f"que lo sostenga)."),
                confianza=0.6,
            ))
    return findings


def detect_hemiola(notes, grid, ctx, min_run=4):
    """Heurística: busca, dentro de compases simples ternarios (3/x) o
    compuestos (6/8, 9/8, 12/8), una racha de al menos `min_run` ataques
    consecutivos en una mano espaciados de forma regular pero que NO
    coincide con la subdivisión natural del compás (agrupación 2 en vez
    de 3, o viceversa)."""
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        onsets = group_by_onset(hand_notes)  # colapsa acordes en un único ataque
        if len(onsets) < min_run:
            continue
        iois = []
        for i in range(len(onsets) - 1):
            t1, _, _ = onsets[i]
            t2, _, _ = onsets[i + 1]
            bar_num, num, den, beat_pos, bar_start, bar_len = grid.locate(t1)
            if num % 3 != 0 and num != 3:
                iois.append(None)
                continue
            ioi_beats = (t2 - t1) / grid.tpb
            iois.append((ioi_beats, bar_num, num, den, beat_pos))

        i = 0
        while i < len(iois):
            if iois[i] is None:
                i += 1
                continue
            base_ioi, bar_num0, num0, den0, _ = iois[i]
            natural_unit = 1.5 if grid.is_compound(num0, den0) else 1.0
            hemiola_unit = 1.0 if grid.is_compound(num0, den0) else (1.5 if num0 == 3 else None)
            if hemiola_unit is None or base_ioi <= 0:
                i += 1
                continue
            run_start = i
            j = i
            while (j < len(iois) and iois[j] is not None
                   and abs(iois[j][0] - hemiola_unit) < 0.15):
                j += 1
            run_len = j - run_start
            if run_len >= min_run - 1:
                _, bstart, numS, denS, beatS = iois[run_start]
                findings.append(Finding(
                    resource="hemiola",
                    hand=hand,
                    compas=bstart,
                    tiempo=grid.beat_label(numS, denS, beatS),
                    intencion="Crear ambigüedad métrica reagrupando los pulsos "
                              "del compás, generando un efecto de ensanchamiento "
                              "o contracción del tiempo antes de reafirmar el "
                              "compás original.",
                    explicacion=(f"Durante {run_len + 1} ataques consecutivos, "
                                 f"esta mano acentúa cada {hemiola_unit:g} negra(s) "
                                 f"en lugar de la subdivisión natural del compás "
                                 f"({num0}/{den0}), sugiriendo una métrica alternativa."),
                    confianza=0.45,
                ))
            i = max(j, i + 1)
    return findings


def detect_apoyatura(notes, grid, ctx, max_ioi_semitone=2):
    """Requiere ambas manos. Para cada nota de RH en tiempo relativamente
    fuerte, comprueba si su clase de altura no pertenece al conjunto de
    clases sonando en LH en ese instante, y si la nota siguiente en RH
    resuelve por grado conjunto a una clase presente en la armonía."""
    findings = []
    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    if not lh_notes:
        return findings

    for i in range(len(rh_notes) - 1):
        n, nxt = rh_notes[i], rh_notes[i + 1]
        bar_num, num, den, beat_pos, bar_start, bar_len = grid.locate(n.start_tick)
        quarters_per_beat = 1.5 if grid.is_compound(num, den) else (4.0 / den)
        offset_within_beat = (beat_pos % quarters_per_beat) / quarters_per_beat
        is_strongish = offset_within_beat < 0.15  # cae en un pulso, no en su mitad

        harmony = notes_sounding_at(lh_notes, n.start_tick)
        if not harmony:
            continue
        harmony_pcs = pitch_classes([h.pitch for h in harmony])
        if n.pitch % 12 in harmony_pcs:
            continue  # es nota del acorde, no disonancia

        interval = abs(nxt.pitch - n.pitch)
        resolves_stepwise = 0 < interval <= max_ioi_semitone
        next_harmony = notes_sounding_at(lh_notes, nxt.start_tick)
        next_pcs = pitch_classes([h.pitch for h in next_harmony]) if next_harmony else harmony_pcs
        resolves_into_chord = (nxt.pitch % 12) in next_pcs

        if resolves_stepwise and resolves_into_chord and is_strongish:
            direction = "descendente" if nxt.pitch < n.pitch else "ascendente"
            findings.append(Finding(
                resource="apoyatura",
                hand="RH",
                compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Retrasar y realzar la llegada a la nota del acorde "
                          "mediante una disonancia preparatoria en tiempo fuerte.",
                explicacion=(f"La nota {midi_name(n.pitch)} suena en tiempo "
                             f"{grid.beat_label(num, den, beat_pos)} sin pertenecer "
                             f"a la armonía de la mano izquierda, y resuelve por "
                             f"grado conjunto {direction} a {midi_name(nxt.pitch)}, "
                             f"que sí es nota del acorde."),
                confianza=0.55,
            ))
    return findings


def detect_floreo(notes, grid, ctx, grace_frac=0.22, max_run=6):
    """Detecta grupos de notas muy breves (relativo al pulso) en torno a
    una nota principal: mordentes (1 vecino), grupetos (2-4 notas),
    trinos (alternancia rápida de 2 alturas)."""
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
        i = 0
        while i < len(hand_notes):
            n = hand_notes[i]
            bar_num, num, den, beat_pos, bar_start, bar_len = grid.locate(n.start_tick)
            quarters_per_beat = 1.5 if grid.is_compound(num, den) else (4.0 / den)
            beat_ticks = quarters_per_beat * grid.tpb
            dur_frac = n.duration_ticks / beat_ticks if beat_ticks else 1

            if dur_frac > grace_frac:
                i += 1
                continue

            # agrupa notas cortas consecutivas y cercanas en altura
            j = i
            group = [n]
            while j + 1 < len(hand_notes) and (j - i) < max_run:
                m = hand_notes[j + 1]
                gap = m.start_tick - hand_notes[j].end_tick
                dur_frac_m = m.duration_ticks / beat_ticks if beat_ticks else 1
                close_pitch = abs(m.pitch - hand_notes[j].pitch) <= 4
                if dur_frac_m <= grace_frac and gap <= beat_ticks * 0.35 and close_pitch:
                    group.append(m)
                    j += 1
                else:
                    break

            if len(group) >= 2:
                pitches = [g.pitch for g in group]
                if len(set(pitches)) == 2 and len(group) >= 4:
                    kind = "trino"
                    intencion = "Sostener y brillar la nota principal mediante " \
                                "alternancia rápida con su vecina superior."
                elif len(group) <= 2:
                    kind = "mordente"
                    intencion = "Dar un breve impulso ornamental a la nota " \
                                "principal sin alterar su función armónica."
                else:
                    kind = "grupeto"
                    intencion = "Rodear melódicamente la nota principal para " \
                                "suavizar su llegada."
                findings.append(Finding(
                    resource="floreo",
                    hand=hand,
                    compas=bar_num,
                    tiempo=grid.beat_label(num, den, beat_pos),
                    intencion=intencion,
                    explicacion=(f"Grupo de {len(group)} notas muy breves "
                                 f"({', '.join(midi_name(p) for p in pitches)}) "
                                 f"identificado como {kind} en torno al tiempo "
                                 f"{grid.beat_label(num, den, beat_pos)}."),
                    confianza=0.5,
                ))
                i = j + 1
            else:
                i += 1
    return findings


DIM_TRIAD = frozenset({0, 3, 6})
DIM7 = frozenset({0, 3, 6, 9})
NAPOLITAN_TRIAD = frozenset({0, 4, 7})
RATIO_PAIRS = {(3, 2): "3 contra 2", (2, 3): "2 contra 3",
               (4, 3): "4 contra 3", (3, 4): "3 contra 4"}


def chord_matches_shape(pcs, shape):
    """Si el conjunto de clases de altura `pcs` coincide (en alguna
    transposición) con `shape`, devuelve la clase de altura raíz."""
    if len(pcs) != len(shape):
        return None
    for root in pcs:
        if frozenset((p - root) % 12 for p in pcs) == shape:
            return root
    return None


# --- Rítmicos ---------------------------------------------------------

def detect_rubato(notes, grid, ctx, min_events=3, min_change_ratio=0.06):
    """Requiere que el MIDI codifique una curva de tempo (mensajes
    set_tempo); en un MIDI cuantizado sin automatización de tempo no
    detectará nada."""
    findings = []
    tempos = sorted(ctx.tempo_changes, key=lambda t: t[0])
    if len(tempos) < min_events:
        return findings
    i = 0
    while i < len(tempos) - 1:
        direction = None
        j = i
        while j + 1 < len(tempos):
            t1, t2 = tempos[j][1], tempos[j + 1][1]
            if t2 == t1:
                break
            d = "accel" if t2 < t1 else "rit"
            if direction is None:
                direction = d
            elif d != direction:
                break
            j += 1
        run_len = j - i + 1
        if run_len >= min_events:
            first_tempo, last_tempo = tempos[i][1], tempos[j][1]
            change_ratio = abs(last_tempo - first_tempo) / first_tempo
            if change_ratio >= min_change_ratio:
                bar_num, num, den, beat_pos, _, _ = grid.locate(tempos[i][0])
                bpm1, bpm2 = 60_000_000 / first_tempo, 60_000_000 / last_tempo
                kind = "accelerando" if direction == "accel" else "ritardando"
                findings.append(Finding(
                    resource="rubato", hand="—", compas=bar_num,
                    tiempo=grid.beat_label(num, den, beat_pos),
                    intencion="Dar flexibilidad expresiva al tiempo, estirando "
                              "o comprimiendo el pulso de forma continuada.",
                    explicacion=(f"El tempo varía de forma continuada de "
                                 f"{bpm1:.0f} a {bpm2:.0f} BPM a lo largo de "
                                 f"{run_len} cambios ({kind}), a diferencia de "
                                 f"un cambio de tempo puntual."),
                    confianza=0.5,
                ))
        i = j + 1 if j > i else i + 1
    return findings


def detect_agogica(notes, grid, ctx, deviation_ratio=0.04):
    """Igual que rubato, requiere una curva de tempo codificada en el
    MIDI; busca un 'respiro' de tempo puntual que se recupera enseguida,
    a diferencia del cambio sostenido del rubato."""
    findings = []
    tempos = sorted(ctx.tempo_changes, key=lambda t: t[0])
    if len(tempos) < 3:
        return findings
    for i in range(1, len(tempos) - 1):
        prev_t, cur_t, next_t = tempos[i - 1][1], tempos[i][1], tempos[i + 1][1]
        dev_in = (cur_t - prev_t) / prev_t
        dev_out = (next_t - cur_t) / cur_t
        recovers = abs(next_t - prev_t) / prev_t < deviation_ratio * 0.6
        if abs(dev_in) >= deviation_ratio and (dev_in * dev_out) < 0 and recovers:
            bar_num, num, den, beat_pos, _, _ = grid.locate(tempos[i][0])
            kind = "alargamiento" if cur_t > prev_t else "apresuramiento"
            findings.append(Finding(
                resource="agogica", hand="—", compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Realzar un instante puntual (una llegada o una "
                          "nota expresiva) con un pequeño respiro de tempo "
                          "que se recupera de inmediato.",
                explicacion=(f"El tempo sufre un {kind} puntual en este punto "
                             f"y vuelve al pulso original casi enseguida, a "
                             f"diferencia de un cambio de tempo sostenido."),
                confianza=0.4,
            ))
    return findings


def detect_polirritmia(notes, grid, ctx, tolerance=0.18):
    """Compara, compás a compás, si cada mano ataca con una subdivisión
    regular pero distinta (p.ej. 3 ataques regulares en RH frente a 2 en
    LH), la definición habitual de poliritmia a nivel de compás."""
    findings = []
    rh_onsets = [t for t, _, _ in group_by_onset([n for n in notes if n.hand == "RH"])]
    lh_onsets = [t for t, _, _ in group_by_onset([n for n in notes if n.hand == "LH"])]
    if not rh_onsets or not lh_onsets:
        return findings

    def evenly_spaced(ts):
        if len(ts) < 2:
            return False
        diffs = np.diff(sorted(ts))
        if diffs.min() <= 0:
            return False
        return (diffs.std() / diffs.mean()) < tolerance

    for bar_start, num, den, bar_num in grid.bars:
        bar_len = grid._bar_len_ticks(num, den)
        w_start, w_end = bar_start, bar_start + bar_len
        rh_in = [t for t in rh_onsets if w_start <= t < w_end]
        lh_in = [t for t in lh_onsets if w_start <= t < w_end]
        if len(rh_in) < 2 or len(lh_in) < 2:
            continue
        if not (evenly_spaced(rh_in) and evenly_spaced(lh_in)):
            continue
        ratio = (len(rh_in), len(lh_in))
        if ratio in RATIO_PAIRS:
            findings.append(Finding(
                resource="polirritmia", hand="RH/LH", compas=bar_num,
                tiempo="1.00",
                intencion="Superponer dos subdivisiones simultáneas del "
                          "mismo pulso para crear una textura rítmica "
                          "compleja.",
                explicacion=(f"En este compás, la mano derecha ataca "
                             f"{len(rh_in)} veces de forma regular "
                             f"mientras la izquierda ataca {len(lh_in)}, "
                             f"formando una relación de "
                             f"{RATIO_PAIRS[ratio]}."),
                confianza=0.5,
            ))
    return findings


# --- Melódicos ----------------------------------------------------------

def detect_anticipacion(notes, grid, ctx, max_lead_frac=0.3):
    findings = []
    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    lh_onsets = group_by_onset(lh_notes)
    if not lh_onsets:
        return findings
    lh_onset_ticks = [t for t, _, _ in lh_onsets]

    for n in rh_notes:
        idx = bisect.bisect_right(lh_onset_ticks, n.start_tick)
        if idx >= len(lh_onsets):
            continue
        next_tick, next_pitches, _ = lh_onsets[idx]
        gap = next_tick - n.start_tick
        if gap <= 0:
            continue
        bar_num, num, den, beat_pos, _, _ = grid.locate(n.start_tick)
        quarters_per_beat = 1.5 if grid.is_compound(num, den) else (4.0 / den)
        beat_ticks = quarters_per_beat * grid.tpb
        if gap > beat_ticks * max_lead_frac:
            continue
        current_harmony = notes_sounding_at(lh_notes, n.start_tick)
        current_pcs = pitch_classes([h.pitch for h in current_harmony]) if current_harmony else set()
        next_pcs = pitch_classes(next_pitches)
        if (n.pitch % 12) in current_pcs or (n.pitch % 12) not in next_pcs:
            continue
        findings.append(Finding(
            resource="anticipacion", hand="RH", compas=bar_num,
            tiempo=grid.beat_label(num, den, beat_pos),
            intencion="Adelantar melódicamente una nota de la armonía que "
                      "está a punto de llegar, creando expectación antes "
                      "del cambio armónico real.",
            explicacion=(f"La nota {midi_name(n.pitch)} suena "
                         f"{gap / grid.tpb:.2f} negra(s) antes de que la "
                         f"mano izquierda cambie a un acorde que sí la "
                         f"contiene, anticipando esa armonía."),
            confianza=0.45,
        ))
    return findings


def detect_escapada(notes, grid, ctx, step_max=2, leap_min=3):
    findings = []
    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    if len(rh_notes) < 3 or not lh_notes:
        return findings
    for i in range(len(rh_notes) - 2):
        n0, n1, n2 = rh_notes[i], rh_notes[i + 1], rh_notes[i + 2]
        harmony1 = notes_sounding_at(lh_notes, n1.start_tick)
        if not harmony1:
            continue
        pcs1 = pitch_classes([h.pitch for h in harmony1])
        if n1.pitch % 12 in pcs1:
            continue
        step1, step2 = n1.pitch - n0.pitch, n2.pitch - n1.pitch
        if not (0 < abs(step1) <= step_max):
            continue
        if not (abs(step2) >= leap_min):
            continue
        if (step1 > 0) == (step2 > 0):
            continue  # debe cambiar de dirección
        harmony2 = notes_sounding_at(lh_notes, n2.start_tick) or harmony1
        pcs2 = pitch_classes([h.pitch for h in harmony2])
        if n2.pitch % 12 not in pcs2:
            continue
        bar_num, num, den, beat_pos, _, _ = grid.locate(n1.start_tick)
        findings.append(Finding(
            resource="escapada", hand="RH", compas=bar_num,
            tiempo=grid.beat_label(num, den, beat_pos),
            intencion="Dar color melódico momentáneo abandonando la nota "
                      "del acorde por grado conjunto y saltando en "
                      "dirección opuesta a la siguiente nota del acorde.",
            explicacion=(f"Tras {midi_name(n0.pitch)}, la melodía se mueve "
                         f"por grado conjunto a {midi_name(n1.pitch)} (nota "
                         f"ajena a la armonía) y de ahí salta a "
                         f"{midi_name(n2.pitch)}, nota del acorde, en "
                         f"dirección contraria."),
            confianza=0.45,
        ))
    return findings


def detect_nota_de_paso_cromatica(notes, grid, ctx):
    findings = []
    rh_notes = sorted([n for n in notes if n.hand == "RH"], key=lambda n: n.start_tick)
    lh_notes = [n for n in notes if n.hand == "LH"]
    if len(rh_notes) < 3 or not lh_notes:
        return findings
    for i in range(len(rh_notes) - 2):
        n0, n1, n2 = rh_notes[i], rh_notes[i + 1], rh_notes[i + 2]
        step1, step2 = n1.pitch - n0.pitch, n2.pitch - n1.pitch
        if not (0 < abs(step1) <= 2 and 0 < abs(step2) <= 2):
            continue
        if (step1 > 0) != (step2 > 0):
            continue  # misma dirección -> nota de paso, no bordadura
        if not (abs(step1) == 1 or abs(step2) == 1):
            continue  # exige al menos un semitono (cromatismo)
        harmony1 = notes_sounding_at(lh_notes, n1.start_tick)
        if not harmony1:
            continue
        if n1.pitch % 12 in pitch_classes([h.pitch for h in harmony1]):
            continue
        harmony0 = notes_sounding_at(lh_notes, n0.start_tick) or harmony1
        harmony2 = notes_sounding_at(lh_notes, n2.start_tick) or harmony1
        if (n0.pitch % 12) not in pitch_classes([h.pitch for h in harmony0]):
            continue
        if (n2.pitch % 12) not in pitch_classes([h.pitch for h in harmony2]):
            continue
        bar_num, num, den, beat_pos, _, _ = grid.locate(n1.start_tick)
        direction = "ascendente" if step1 > 0 else "descendente"
        findings.append(Finding(
            resource="nota_de_paso_cromatica", hand="RH", compas=bar_num,
            tiempo=grid.beat_label(num, den, beat_pos),
            intencion="Conectar dos notas del acorde por movimiento "
                      "cromático, suavizando y coloreando el paso "
                      "melódico.",
            explicacion=(f"Entre {midi_name(n0.pitch)} y {midi_name(n2.pitch)} "
                         f"(ambas del acorde), {midi_name(n1.pitch)} pasa "
                         f"por grado conjunto {direction} incluyendo al "
                         f"menos un semitono, fuera de la armonía."),
            confianza=0.4,
        ))
    return findings


# --- Armónicos ------------------------------------------------------------

def detect_acorde_disminuido(notes, grid, ctx):
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        for start_tick, pitches, end_tick in group_by_onset(hand_notes):
            pcs = set(p % 12 for p in pitches)
            if len(pcs) < 3:
                continue
            root, shape_name = None, None
            if len(pcs) == 3:
                root = chord_matches_shape(pcs, DIM_TRIAD)
                shape_name = "tríada disminuida"
            elif len(pcs) == 4:
                root = chord_matches_shape(pcs, DIM7)
                shape_name = "séptima disminuida"
            if root is None:
                continue
            bar_num, num, den, beat_pos, _, _ = grid.locate(start_tick)
            subject, _ = describe_pitches(pitches)
            findings.append(Finding(
                resource="acorde_disminuido", hand=hand, compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Generar inestabilidad armónica con un acorde "
                          "simétrico de terceras menores que exige "
                          "resolución.",
                explicacion=(f"{subject} forma una {shape_name} sobre "
                             f"{midi_name(root)}, un acorde inestable que "
                             f"suele resolver por semitono."),
                confianza=0.55,
            ))
    return findings


def detect_dominante_secundaria(notes, grid, ctx):
    findings = []
    lh_notes = [n for n in notes if n.hand == "LH"]
    chords = []
    for t, pitches, end in group_by_onset(lh_notes):
        pcs = set(p % 12 for p in pitches)
        bass_pc = min(pitches) % 12
        root = None
        if len(pcs) >= 3:
            for r in pcs:
                rel = set((p - r) % 12 for p in pcs)
                if 4 in rel and 10 in rel:  # 3a mayor + tritono -> sonoridad de dominante
                    root = r
                    break
        chords.append((t, pitches, pcs, root, bass_pc))

    for i in range(len(chords) - 1):
        t, pitches, pcs, root, bass_pc = chords[i]
        if root is None or pcs.issubset(ctx.scale_pcs):
            continue  # totalmente diatónico -> es la dominante normal, no "secundaria"
        _, _, nxt_pcs, nxt_root, nxt_bass_pc = chords[i + 1]
        target_root = nxt_root if nxt_root is not None else nxt_bass_pc
        resolves = (target_root - root) % 12 == 5  # 4a justa ascendente
        if resolves:
            bar_num, num, den, beat_pos, _, _ = grid.locate(t)
            subject, _ = describe_pitches(pitches)
            findings.append(Finding(
                resource="dominante_secundaria", hand="LH", compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Tonicizar momentáneamente un grado distinto de "
                          "la tónica, introduciendo tensión cromática que "
                          "resuelve por círculo de quintas.",
                explicacion=(f"{subject} contiene la tercera mayor y el "
                             f"tritono característicos de un acorde de "
                             f"dominante sobre {midi_name(root)}, ajeno a "
                             f"la tonalidad estimada "
                             f"({NOTE_NAMES[ctx.key_tonic]} {ctx.key_mode}), "
                             f"y resuelve una 4a justa después."),
                confianza=0.4,
            ))
    return findings


def detect_acorde_prestado(notes, grid, ctx):
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        for t, pitches, end in group_by_onset(hand_notes):
            pcs = set(p % 12 for p in pitches)
            if len(pcs) < 3:
                continue
            foreign = pcs - ctx.scale_pcs
            if not foreign:
                continue
            has_tritone = any((a - b) % 12 == 6 for a in pcs for b in pcs if a != b)
            if has_tritone:
                continue  # eso se clasifica aparte como dominante secundaria
            bar_num, num, den, beat_pos, _, _ = grid.locate(t)
            subject, _ = describe_pitches(pitches)
            foreign_names = ", ".join(NOTE_NAMES[p] for p in sorted(foreign))
            findings.append(Finding(
                resource="acorde_prestado", hand=hand, compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Tomar prestado un color armónico de la "
                          "tonalidad paralela (mayor/menor), oscureciendo "
                          "o iluminando momentáneamente la armonía.",
                explicacion=(f"{subject} incluye {foreign_names}, ajena(s) "
                             f"a la tonalidad estimada "
                             f"({NOTE_NAMES[ctx.key_tonic]} {ctx.key_mode}), "
                             f"lo que sugiere un acorde prestado de la "
                             f"tonalidad paralela."),
                confianza=0.35,
            ))
    return findings


def detect_sexta_napolitana(notes, grid, ctx):
    findings = []
    lh_notes = [n for n in notes if n.hand == "LH"]
    napolitan_root = (ctx.key_tonic + 1) % 12
    for t, pitches, end in group_by_onset(lh_notes):
        pcs = set(p % 12 for p in pitches)
        if len(pcs) != 3:
            continue
        if frozenset((p - napolitan_root) % 12 for p in pcs) != NAPOLITAN_TRIAD:
            continue
        bass_pitch = min(pitches)
        if bass_pitch % 12 != (napolitan_root + 4) % 12:
            continue  # exige primera inversión (6a)
        bar_num, num, den, beat_pos, _, _ = grid.locate(t)
        subject, _ = describe_pitches(pitches)
        findings.append(Finding(
            resource="sexta_napolitana", hand="LH", compas=bar_num,
            tiempo=grid.beat_label(num, den, beat_pos),
            intencion="Oscurecer momentáneamente la armonía con el color "
                      "característico de la sexta napolitana antes de "
                      "resolver hacia la dominante o la tónica.",
            explicacion=(f"{subject} forma una tríada mayor sobre "
                         f"{midi_name(napolitan_root)} (segundo grado "
                         f"descendido) en primera inversión, el patrón "
                         f"clásico de la sexta napolitana respecto a la "
                         f"tonalidad estimada ({NOTE_NAMES[ctx.key_tonic]} "
                         f"{ctx.key_mode})."),
            confianza=0.35,
        ))
    return findings


def detect_pedal_armonico(notes, grid, ctx, min_run=3):
    findings = []
    lh_notes = [n for n in notes if n.hand == "LH"]
    onsets = group_by_onset(lh_notes)
    if len(onsets) < min_run:
        return findings
    i = 0
    while i < len(onsets):
        base_bass_pc = min(onsets[i][1]) % 12
        j = i
        while j < len(onsets) and min(onsets[j][1]) % 12 == base_bass_pc:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            upper_pcs = set()
            for k in range(i, j):
                bass_k = min(onsets[k][1])
                upper_pcs |= set(p % 12 for p in onsets[k][1] if p != bass_k)
            if len(upper_pcs) >= 2:  # algo se mueve por encima, no es un acorde estático
                t0, pitches0, _ = onsets[i]
                bar_num, num, den, beat_pos, _, _ = grid.locate(t0)
                findings.append(Finding(
                    resource="pedal_armonico", hand="LH", compas=bar_num,
                    tiempo=grid.beat_label(num, den, beat_pos),
                    intencion="Crear tensión creciente sosteniendo un bajo "
                              "fijo mientras la armonía superior se mueve "
                              "y se aleja de él.",
                    explicacion=(f"El bajo {midi_name(min(pitches0))} se "
                                 f"repite sin cambiar durante {run_len} "
                                 f"ataques consecutivos mientras las voces "
                                 f"superiores cambian de altura."),
                    confianza=0.5,
                ))
        i = max(j, i + 1)
    return findings


def detect_retardo(notes, grid, ctx, max_step=2):
    """Suspensión: nota preparada como consonancia, ligada sobre un "
    cambio de armonía en el que pasa a ser disonante, y que resuelve
    después por grado conjunto."""
    findings = []
    for hand in ("RH", "LH"):
        other = "LH" if hand == "RH" else "RH"
        hand_notes = sorted([n for n in notes if n.hand == hand], key=lambda n: n.start_tick)
        other_notes = [n for n in notes if n.hand == other]
        other_onsets = group_by_onset(other_notes)
        other_ticks = [t for t, _, _ in other_onsets]
        if not other_onsets:
            continue

        for idx, n in enumerate(hand_notes):
            harmony_before = notes_sounding_at(other_notes, n.start_tick)
            if not harmony_before:
                continue
            if n.pitch % 12 not in pitch_classes([h.pitch for h in harmony_before]):
                continue  # debe estar preparada como consonancia

            change_idx = bisect.bisect_right(other_ticks, n.start_tick)
            if change_idx >= len(other_onsets):
                continue
            change_tick, change_pitches, _ = other_onsets[change_idx]
            if not (n.start_tick < change_tick < n.end_tick):
                continue  # debe estar ligada sobre el cambio de armonía

            pcs_after = pitch_classes(change_pitches)
            if n.pitch % 12 in pcs_after:
                continue  # sigue siendo consonante: no hay retardo

            # la siguiente nota de esta mano cuyo ataque cae en o después del
            # final (ya extendido) de n; no basta con "el siguiente índice",
            # porque n puede formar parte de un acorde (varias notas con el
            # mismo start_tick) y el índice siguiente sería una compañera de
            # acorde simultánea, no la resolución real
            later = [m for m in hand_notes if m.start_tick >= n.end_tick]
            if not later:
                continue
            nxt = later[0]
            interval = abs(nxt.pitch - n.pitch)
            if not (0 < interval <= max_step):
                continue
            harmony_res = notes_sounding_at(other_notes, nxt.start_tick)
            pcs_res = pitch_classes([h.pitch for h in harmony_res]) if harmony_res else pcs_after
            if nxt.pitch % 12 not in pcs_res:
                continue

            bar_num, num, den, beat_pos, _, _ = grid.locate(change_tick)
            direction = "descendente" if nxt.pitch < n.pitch else "ascendente"
            findings.append(Finding(
                resource="retardo", hand=hand, compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Retrasar la llegada de una nota del acorde "
                          "manteniendo sonando la nota anterior, ya "
                          "preparada, hasta convertirse en disonancia que "
                          "resuelve por grado conjunto.",
                explicacion=(f"{midi_name(n.pitch)} suena de forma "
                             f"consonante y se sostiene sobre el cambio de "
                             f"armonía, quedando en disonancia; resuelve "
                             f"{direction} a {midi_name(nxt.pitch)}."),
                confianza=0.45,
            ))
    return findings


# --- Dinámicos / articulatorios ------------------------------------------

def detect_sforzando(notes, grid, ctx, window=6, min_diff=22):
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        groups = {}
        for n in hand_notes:
            groups.setdefault(n.start_tick, []).append(n)
        events = [(t, max(m.velocity for m in groups[t]), [m.pitch for m in groups[t]])
                  for t in sorted(groups.keys())]
        vels = [e[1] for e in events]
        for i, (t, vel, pitches) in enumerate(events):
            lo, hi = max(0, i - window // 2), min(len(events), i + window // 2 + 1)
            local = vels[lo:hi]
            avg_local = (sum(local) - vel) / max(len(local) - 1, 1)
            if vel - avg_local >= min_diff:
                bar_num, num, den, beat_pos, _, _ = grid.locate(t)
                subject, _ = describe_pitches(pitches)
                findings.append(Finding(
                    resource="sforzando", hand=hand, compas=bar_num,
                    tiempo=grid.beat_label(num, den, beat_pos),
                    intencion="Marcar un acento súbito y puntual muy por "
                              "encima de la dinámica circundante.",
                    explicacion=(f"{subject} ataca con velocidad {vel}, muy "
                                 f"por encima de la media local de "
                                 f"{avg_local:.0f} en esta mano."),
                    confianza=0.5,
                ))
    return findings


def detect_silencio_expresivo(notes, grid, ctx, min_ratio=1.8, min_gap_beats=0.5):
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        onsets = group_by_onset(hand_notes)
        if len(onsets) < 4:
            continue
        gaps = []
        for i in range(1, len(onsets)):
            prev_end = max(m.end_tick for m in hand_notes if m.start_tick == onsets[i - 1][0])
            gaps.append(onsets[i][0] - prev_end)
        for i in range(1, len(gaps)):
            window = gaps[max(0, i - 4):i]
            if not window:
                continue
            avg = sum(window) / len(window)
            if avg <= 0:
                continue
            gap = gaps[i]
            if gap / grid.tpb < min_gap_beats or gap / avg < min_ratio:
                continue
            t, pitches, _ = onsets[i + 1]
            bar_num, num, den, beat_pos, _, _ = grid.locate(t)
            subject, _ = describe_pitches(pitches)
            findings.append(Finding(
                resource="silencio_expresivo", hand=hand, compas=bar_num,
                tiempo=grid.beat_label(num, den, beat_pos),
                intencion="Crear expectación mediante un silencio "
                          "notablemente más largo de lo habitual justo "
                          "antes de una entrada importante.",
                explicacion=(f"Antes de {subject}, esta mano guarda un "
                             f"silencio de {gap / grid.tpb:.2f} negra(s), "
                             f"muy por encima del hueco medio reciente "
                             f"({avg / grid.tpb:.2f})."),
                confianza=0.4,
            ))
    return findings


def detect_crescendo_dirigido(notes, grid, ctx, min_run=4, min_total_increase=18):
    findings = []
    for hand in ("RH", "LH"):
        hand_notes = [n for n in notes if n.hand == hand]
        groups = {}
        for n in hand_notes:
            groups.setdefault(n.start_tick, []).append(n)
        events = [(t, max(m.velocity for m in groups[t])) for t in sorted(groups.keys())]
        i = 0
        while i < len(events) - 1:
            j = i
            while j + 1 < len(events) and events[j + 1][1] >= events[j][1]:
                j += 1
            run_len = j - i + 1
            total_increase = events[j][1] - events[i][1]
            if run_len >= min_run and total_increase >= min_total_increase:
                bar_num, num, den, beat_pos, _, _ = grid.locate(events[i][0])
                findings.append(Finding(
                    resource="crescendo_dirigido", hand=hand, compas=bar_num,
                    tiempo=grid.beat_label(num, den, beat_pos),
                    intencion="Dirigir la energía de una frase hacia un "
                              "punto de llegada mediante un aumento "
                              "progresivo de intensidad.",
                    explicacion=(f"A lo largo de {run_len} ataques "
                                 f"consecutivos en esta mano, la velocidad "
                                 f"sube de {events[i][1]} a {events[j][1]}, "
                                 f"dirigiendo la frase hacia el último."),
                    confianza=0.45,
                ))
                i = j + 1
            else:
                i += 1
    return findings


DETECTORS = {
    "sincopa": detect_sincopa,
    "contratiempo": detect_contratiempo,
    "hemiola": detect_hemiola,
    "apoyatura": detect_apoyatura,
    "floreo": detect_floreo,
    "rubato": detect_rubato,
    "agogica": detect_agogica,
    "polirritmia": detect_polirritmia,
    "anticipacion": detect_anticipacion,
    "escapada": detect_escapada,
    "nota_de_paso_cromatica": detect_nota_de_paso_cromatica,
    "acorde_disminuido": detect_acorde_disminuido,
    "dominante_secundaria": detect_dominante_secundaria,
    "acorde_prestado": detect_acorde_prestado,
    "sexta_napolitana": detect_sexta_napolitana,
    "pedal_armonico": detect_pedal_armonico,
    "retardo": detect_retardo,
    "sforzando": detect_sforzando,
    "silencio_expresivo": detect_silencio_expresivo,
    "crescendo_dirigido": detect_crescendo_dirigido,
}

# ── Utilidades de presentación ────────────────────────────────────────────

NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


def midi_name(pitch: int) -> str:
    octave = pitch // 12 - 1
    return f"{NOTE_NAMES[pitch % 12]}{octave}"


def print_report(findings, use_color=True):
    if not findings:
        print("No se detectaron recursos con los umbrales actuales.")
        return
    findings = sorted(findings, key=lambda f: (f.compas, f.resource))
    for f in findings:
        color = RESOURCE_COLOR.get(f.resource, "") if use_color else ""
        reset = C.RESET if use_color else ""
        bold = C.BOLD if use_color else ""
        dim = C.DIM if use_color else ""
        print(f"{bold}{color}[{f.resource.upper()}]{reset} "
              f"compás {f.compas}, tiempo {f.tiempo} ({f.hand})  "
              f"{dim}(confianza {f.confianza:.2f}){reset}")
        print(f"  {C.BOLD if use_color else ''}Intención:{reset} {f.intencion}")
        print(f"  {C.BOLD if use_color else ''}Explicación:{reset} {f.explicacion}")
        print()


# ── Subcomandos ───────────────────────────────────────────────────────────

def cmd_info(args):
    tpb, time_sigs, tempo_changes, notes, mid = load_notes(args.midi)
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
    print(f"Cambios de tempo codificados: {len(tempo_changes)}")
    print(f"Tonalidad estimada: {NOTE_NAMES[key_tonic]} {key_mode}")
    print(f"Notas totales: {len(notes)}  (RH: {len(rh)}, LH: {len(lh)})")
    print(f"Compases estimados: {grid.bars[-1][3] if grid.bars else 0}")


def cmd_list_resources(args):
    for name, desc in RESOURCE_DESCRIPTIONS.items():
        color = RESOURCE_COLOR.get(name, "")
        print(f"{C.BOLD}{color}{name}{C.RESET}")
        print(f"  {desc}")
        print()


def cmd_analyze(args):
    tpb, time_sigs, tempo_changes, notes, mid = load_notes(args.midi)
    notes = assign_hands(mid, notes)
    last_tick = max((n.end_tick for n in notes), default=0)
    grid = MetricGrid(tpb, time_sigs, last_tick)
    key_tonic, key_mode, scale_pcs = estimate_key(notes)
    ctx = AnalysisContext(tempo_changes=tempo_changes, key_tonic=key_tonic,
                           key_mode=key_mode, scale_pcs=scale_pcs)

    requested = [r.strip() for r in args.resources.split(",")] if args.resources else ALL_RESOURCES
    unknown = [r for r in requested if r not in DETECTORS]
    if unknown:
        print(f"Recurso(s) desconocido(s): {', '.join(unknown)}. "
              f"Disponibles: {', '.join(ALL_RESOURCES)}", file=sys.stderr)
        sys.exit(1)

    all_findings = []
    for r in requested:
        all_findings.extend(DETECTORS[r](notes, grid, ctx))

    if args.json:
        print(json.dumps([asdict(f) for f in all_findings], ensure_ascii=False, indent=2))
    else:
        print_report(all_findings, use_color=not args.no_color)
        print(f"{C.DIM}Total: {len(all_findings)} hallazgo(s) sobre "
              f"{grid.bars[-1][3] if grid.bars else 0} compases.{C.RESET}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detecta recursos expresivos (rítmicos, melódicos, "
                    "armónicos) en un MIDI de piano con manos separadas.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Muestra metadatos básicos del MIDI")
    p_info.add_argument("midi")
    p_info.set_defaults(func=cmd_info)

    p_list = sub.add_parser("list-resources", help="Lista los recursos detectables")
    p_list.set_defaults(func=cmd_list_resources)

    p_analyze = sub.add_parser("analyze", help="Analiza el MIDI y reporta hallazgos")
    p_analyze.add_argument("midi")
    p_analyze.add_argument("--resources", default=None,
                            help=f"Lista separada por comas de: {', '.join(ALL_RESOURCES)} "
                                 f"(por defecto, todos)")
    p_analyze.add_argument("--json", action="store_true", help="Salida en JSON")
    p_analyze.add_argument("--no-color", action="store_true", help="Desactiva color ANSI")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
