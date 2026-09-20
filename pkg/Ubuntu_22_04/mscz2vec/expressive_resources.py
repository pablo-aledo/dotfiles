#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║ expressive_resources.py                                              ║
# ║                                                                      ║
# ║ Detecta recursos expresivos (rítmicos, melódicos y armónicos) en un  ║
# ║ MIDI de piano con separación estándar de mano derecha (RH) / mano    ║
# ║ izquierda (LH), e indica por cada hallazgo: recurso, compás,         ║
# ║ intención expresiva y explicación en lenguaje natural.               ║
# ║                                                                      ║
# ║ Recursos soportados:                                                 ║
# ║   sincopa       - acento desplazado a tiempo débil, sostenido sobre  ║
# ║                    el tiempo fuerte siguiente                        ║
# ║   contratiempo  - ataque en el "y" del tiempo tras un silencio en    ║
# ║                    el tiempo fuerte (sin ligadura)                   ║
# ║   hemiola       - reagrupación métrica (p.ej. 3+3 -> 2+2+2)          ║
# ║   apoyatura     - nota extraña a la armonía en tiempo fuerte que      ║
# ║                    resuelve por grado conjunto                       ║
# ║   floreo        - ornamentación rápida (mordente, trino, grupeto)    ║
# ║                    alrededor de una nota principal                   ║
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
# ║ nombre de pista, separa por registro (mediana de altura).            ║
# ║                                                                      ║
# ║ Dependencias: mido, numpy                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

import argparse
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
    notes = []

    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        open_notes = {}  # (pitch) -> (start_tick, velocity)
        for msg in track:
            abs_tick += msg.time
            if msg.type == "time_signature":
                time_sigs.append(TimeSigChange(abs_tick, msg.numerator, msg.denominator))
            elif msg.type == "note_on" and msg.velocity > 0:
                open_notes[(msg.note, msg.channel)] = (abs_tick, msg.velocity)
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

    notes.sort(key=lambda n: n.start_tick)
    return tpb, time_sigs, notes, mid


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


# ── Detectores ────────────────────────────────────────────────────────────

def detect_sincopa(notes, grid, min_off_frac=0.2):
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


def detect_contratiempo(notes, grid, min_off_frac=0.2):
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


def detect_hemiola(notes, grid, min_run=4):
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


def detect_apoyatura(notes, grid, max_ioi_semitone=2):
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


def detect_floreo(notes, grid, grace_frac=0.22, max_run=6):
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


DETECTORS = {
    "sincopa": detect_sincopa,
    "contratiempo": detect_contratiempo,
    "hemiola": detect_hemiola,
    "apoyatura": detect_apoyatura,
    "floreo": detect_floreo,
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
    tpb, time_sigs, notes, mid = load_notes(args.midi)
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


def cmd_analyze(args):
    tpb, time_sigs, notes, mid = load_notes(args.midi)
    notes = assign_hands(mid, notes)
    last_tick = max((n.end_tick for n in notes), default=0)
    grid = MetricGrid(tpb, time_sigs, last_tick)

    requested = [r.strip() for r in args.resources.split(",")] if args.resources else ALL_RESOURCES
    unknown = [r for r in requested if r not in DETECTORS]
    if unknown:
        print(f"Recurso(s) desconocido(s): {', '.join(unknown)}. "
              f"Disponibles: {', '.join(ALL_RESOURCES)}", file=sys.stderr)
        sys.exit(1)

    all_findings = []
    for r in requested:
        all_findings.extend(DETECTORS[r](notes, grid))

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
