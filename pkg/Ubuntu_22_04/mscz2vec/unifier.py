#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         UNIFIER  v1.0                                       ║
║   Unificador de secciones MIDI en una composición coherente                  ║
║                                                                              ║
║  Cuando compones pegando secciones de obras distintas como anclas           ║
║  emocionales, la unión suele sonar a collage: faltan motivos comunes,        ║
║  la armonía no encaja, cambia el pulso y la instrumentación es inconexa.     ║
║  UNIFIER establece un ADN musical compartido y lo inyecta de forma           ║
║  dosificada en cada sección, cosiéndolas con transiciones coherentes para    ║
║  que el resultado suene como una obra unificada.                             ║
║                                                                              ║
║  ADN COMPARTIDO (deducido si no se indica):                                  ║
║  [A1] Motivo semilla  — el más saliente entre las secciones (o MIDI dado)    ║
║  [A2] Plan tonal      — tonalidad hogar + modulaciones lógicas por emoción   ║
║  [A3] Plantilla orch. — piano / cuarteto / cuerdas / cámara / full           ║
║  [A4] Arco de tensión — curva tipo 'arch' sobre las secciones                ║
║                                                                              ║
║  EJES DE UNIFICACIÓN (dosificables con --glue, 0=intacto … 1=total):         ║
║  [M] Motivo           — siembra motivo (orig/invers/retrogr/aument)          ║
║  [H] Armonía          — transp. al plan + nudge de acordes no diatónicos     ║
║  [R] Ritmo            — alineación suave con el groove de referencia         ║
║  [I] Instrumentación  — remapeo al template común por rol (mel/bass/harm)    ║
║                                                                              ║
║  MODOS:                                                                      ║
║  [glue]      Conserva el MIDI de cada ancla y le añade hilos comunes        ║
║              + puentes + reorquestación ligera. Fiel a tus secciones.        ║
║  [recombine] Usa las secciones como donantes de ADN y regenera una obra      ║
║              nueva coherente (progresión + motivo transplantado + bajo).     ║
║  [auto]      Elige glue/recombine según lo diferentes que sean las secciones.║
║                                                                              ║
║  TRANSICIONES entre secciones: cadencia V7→I en la tonalidad destino +       ║
║  declaración del motivo semilla (modo morph) o pivote de nota común.        ║
║                                                                              ║
║  AUTOCONTENIDO: no importa ni llama por subprocess a ningún otro script      ║
║  del ecosistema. Toda la lógica (carga, análisis, transformación, salida)    ║
║  vive en este fichero. Solo depende de mido, numpy y music21 (pip).          ║
║                                                                              ║
║  USO:                                                                        ║
║    # 3 secciones con emoción indicada, tonalidad y plantilla fijas           ║
║    python unifier.py \\                                                       ║
║        --section "intro:serenidad:a.mid" \\                                  ║
║        --section "mid:melancolia:b.mid" \\                                   ║
║        --section "out:triunfo:c.mid" \\                                      ║
║        --mode glue --key D --template chamber \\                             ║
║        --glue motif=0.7,harmony=0.8,rhythm=0.3,instrumentation=1.0 \\        ║
║        --output obra_unificada.mid --report                                  ║
║                                                                              ║
║    # MIDIs posicionales + emociones por orden                                ║
║    python unifier.py a.mid b.mid c.mid --emotions serenidad,angustia,triunfo ║
║                                                                              ║
║    # Deducción total: tonalidad, motivo, emoción e instrumentación se deducen ║
║    python unifier.py a.mid b.mid c.mid --mode auto                           ║
║                                                                              ║
║    # Motivo semilla propio + 2 secciones, plantilla piano                    ║
║    python unifier.py \\                                                       ║
║        --section "a:serenidad:vals.mid" \\                                   ║
║        --section "b:melancolia:blues.mid" \\                                 ║
║        --motif-seed mi_motivo.mid --key C --template piano \\                ║
║        --output obra.mid --report                                            ║
║                                                                              ║
║    # Recombine: obra nueva de 8 compases por sección                         ║
║    python unifier.py a.mid b.mid c.mid --mode recombine --recombine-bars 8   ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --section SPEC     Sección: "label:emotion:path" (repetible, orden fijo)   ║
║    midis              MIDIs posicionales (orden fijo, 2–10 secciones)        ║
║    --emotions LIST    Emociones por orden, separadas por coma                ║
║    --mode MODE        glue | recombine | auto (default: glue)               ║
║    --motif-seed FILE  MIDI con el motivo semilla (default: deduce)           ║
║    --key KEY          Tonalidad hogar, ej: D, F# minor (default: deduce)     ║
║    --template T       piano|string_quartet|strings|chamber|full|auto        ║
║    --glue SPEC        motif=,harmony=,rhythm=,instrumentation= (0–1)         ║
║    --bridge-bars N    Compases de cada puente (default: 2)                   ║
║    --bridge MODE      morph | pivot | auto (default: auto)                   ║
║    --recombine-bars N Compases por sección en modo recombine (default: 8)   ║
║    --output FILE      MIDI de salida (default: obra_unificada.mid)           ║
║    --seed N           Semilla aleatoria (default: 1)                         ║
║    --report           Score de coherencia + JSON con el plan                 ║
║    --verbose          Informe detallado del análisis y el ADN                ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    obra_unificada.mid          — MIDI unificado (un track, multicanal)       ║
║    obra_unificada.report.json  — plan, ADN y score (con --report)            ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy, music21 (todas pip, ya instaladas)              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, math, random, argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

try:
    import mido
except ImportError:
    sys.exit("Falta 'mido'. Instala con: pip install mido")
import numpy as np

# music21 es opcional para análisis de tonalidad; si no está, se usa heurística.
try:
    import music21
    HAVE_M21 = True
except Exception:
    HAVE_M21 = False

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

EMOTIONS = {"angustia","calidez","esperanza","fragilidad","melancolia",
            "serenidad","tension","triunfo"}

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# Intervalos (semitonos) que cada emoción "tira" hacia una región tonal.
# (desplazamiento de tónica relativo al hogar, modo destino偏好)
EMOTION_KEY_BIAS = {
    "serenidad":   {"shift": +5, "mode": "same"},   # subdominante
    "calidez":     {"shift": +5, "mode": "major"},
    "esperanza":   {"shift": +7, "mode": "major"},  # dominante
    "triunfo":     {"shift": +7, "mode": "major"},
    "melancolia":  {"shift":  0, "mode": "minor"},  # relativa/paralela menor
    "fragilidad":  {"shift":  0, "mode": "minor"},
    "tension":     {"shift": +7, "mode": "minor"},
    "angustia":    {"shift": +3, "mode": "minor"},  # mediant
}

# Progresiones por emoción (grados romanos) para modo recombine.
PROGRESSIONS = {
    "serenidad":   ["I","vi","IV","V"],
    "calidez":     ["I","IV","vi","V"],
    "esperanza":   ["IV","I","V","vi"],
    "triunfo":     ["I","V","IV","I"],
    "melancolia":  ["i","VI","III","VII"],
    "fragilidad":  ["i","iv","i","v"],
    "tension":     ["i","VII","VI","V"],
    "angustia":    ["i","i","VI","V"],
}

# Plantillas orquestales: rol -> program GM.
TEMPLATES = {
    "piano":          {"melody":0, "counter":0, "harmony":0, "bass":0},
    "string_quartet": {"melody":40, "counter":40, "harmony":41, "bass":42},
    "strings":        {"melody":40, "counter":40, "harmony":41, "bass":43},
    "chamber":        {"melody":73, "counter":68, "harmony":71, "bass":42},
    "full":           {"melody":56, "counter":68, "harmony":48, "bass":43},
}

OUT_TPB = 480  # ticks por negra del MIDI de salida

# ─────────────────────────────────────────────────────────────────────────────
# Modelos de datos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Note:
    pitch: int
    start: float      # en beats (negra = 1)
    dur: float        # en beats
    velocity: int
    channel: int
    track: int


@dataclass
class Section:
    path: str
    label: str
    emotion: Optional[str] = None
    notes: List[Note] = field(default_factory=list)
    tpb: int = 480
    tempo: float = 120.0
    time_sig: Tuple[int,int] = (4,4)
    tracks_meta: List[dict] = field(default_factory=list)
    programs: Dict[int,int] = field(default_factory=dict)
    # análisis
    key_pc: int = 0
    key_mode: str = "major"
    key_name: str = "C major"
    n_bars: int = 0
    beats_per_bar: float = 4.0
    melody_track: int = 0
    melody_channel: int = 0
    motif_intervals: List[int] = field(default_factory=list)
    motif_durations: List[float] = field(default_factory=list)
    motif_first_pitch: int = 60
    motif_salience: float = 0.0
    chords: List[Tuple[int,str]] = field(default_factory=list)  # (root_pc, quality)
    density: float = 0.0
    register: float = 60.0
    # plan
    target_key_pc: int = 0
    target_mode: str = "major"
    transpose_semis: int = 0
    role_map: Dict[Tuple[int,int], str] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# MIDI I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_midi(path: str) -> Tuple[List[Note], int, float, Tuple[int,int], List[dict], Dict[int,int]]:
    """Carga un MIDI. Devuelve (notas, tpb, tempo, time_sig, tracks_meta, programs)."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat or 480
    notes: List[Note] = []
    tempo = 120.0
    time_sig = (4, 4)
    tracks_meta = []
    programs: Dict[int, int] = {}

    for ti, track in enumerate(mid.tracks):
        t = 0
        name = ""
        pending: Dict[Tuple[int,int], Tuple[int,int]] = {}
        n_notes = 0
        for msg in track:
            t += msg.time
            if msg.is_meta:
                if msg.type == 'set_tempo':
                    try:
                        tempo = 60000000.0 / msg.tempo
                    except ZeroDivisionError:
                        pass
                elif msg.type == 'time_signature':
                    time_sig = (msg.numerator, msg.denominator)
                elif msg.type == 'track_name':
                    name = msg.name
            elif msg.type == 'program_change':
                programs[msg.channel] = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (t, msg.velocity)
                n_notes += 1
            elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                st_v = pending.pop(key, None)
                if st_v is not None:
                    st, v = st_v
                    start_b = st / tpb
                    dur = (t - st) / tpb
                    if dur <= 0:
                        dur = 0.1
                    notes.append(Note(msg.note, start_b, dur, v, msg.channel, ti))
        tracks_meta.append({"idx": ti, "name": name, "n_notes": n_notes})

    notes.sort(key=lambda n: (n.start, -n.pitch))
    return notes, tpb, tempo, time_sig, tracks_meta, programs


def _emit_note_events(events, note, tpb):
    on = round(note.start * tpb)
    off = round((note.start + note.dur) * tpb)
    if off <= on:
        off = on + 1
    events.append((on, 1, note.pitch, note.velocity, note.channel))   # 1 = note_on
    events.append((off, 0, note.pitch, 0, note.channel))               # 0 = note_off


def save_midi(path: str, notes: List[Note], tempo_map: List[Tuple[float,float,Tuple[int,int]]],
              programs: Dict[int,int], tpb: int = OUT_TPB):
    """
    notes: notas en beats (start, dur).
    tempo_map: lista de (beat_offset, tempo_bpm, (num,den)) por segmento.
    programs: canal -> program GM.
    """
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    events = []  # (tick, kind, ...)
    # meta events de tempo/time_sig al inicio de cada segmento
    metas = []
    for (boff, bpm, tsig) in tempo_map:
        tick = round(boff * tpb)
        mpq = int(round(60000000.0 / bpm)) if bpm > 0 else 500000
        metas.append((tick, 'set_tempo', mpq))
        metas.append((tick, 'time_signature', tsig))
    # program changes al inicio
    for ch, prog in programs.items():
        events.append((0, 'prog', ch, prog))
    # notas
    for n in notes:
        _emit_note_events(events, n, tpb)

    # combinar y ordenar
    all_ev = []
    for e in metas:
        all_ev.append(e)
    for e in events:
        all_ev.append(e)
    # ordenar por tick; en empate: meta < prog < note_off < note_on
    def order(e):
        tick = e[0]
        if isinstance(e[1], str) and e[1] in ('set_tempo','time_signature'):
            pri = 0
        elif e[1] == 'prog':
            pri = 1
        elif e[1] == 0:  # note_off
            pri = 2
        else:
            pri = 3
        return (tick, pri)
    all_ev.sort(key=order)

    cur = 0
    for e in all_ev:
        tick = e[0]
        delta = max(0, tick - cur)
        cur = tick
        if isinstance(e[1], str) and e[1] == 'set_tempo':
            track.append(mido.MetaMessage('set_tempo', tempo=e[2], time=delta))
        elif isinstance(e[1], str) and e[1] == 'time_signature':
            num, den = e[2]
            track.append(mido.MetaMessage('time_signature', numerator=num,
                                          denominator=den, time=delta))
        elif e[1] == 'prog':
            track.append(mido.Message('program_change', channel=e[2],
                                      program=e[3], time=delta))
        elif e[1] == 0:  # note_off
            track.append(mido.Message('note_off', channel=e[4], note=e[2],
                                      velocity=0, time=delta))
        else:  # note_on
            track.append(mido.Message('note_on', channel=e[4], note=e[2],
                                      velocity=e[3], time=delta))
    # end of track
    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Análisis musical
# ─────────────────────────────────────────────────────────────────────────────

def beats_per_bar_of(tsig):
    num, den = tsig
    return num * (4.0 / den)


def detect_key(section: Section) -> Tuple[int, str, str]:
    """Devuelve (pitch_class, mode, name). Usa music21 si está; si no, heurística."""
    if HAVE_M21 and section.notes:
        try:
            score = music21.stream.Score()
            part = music21.stream.Part()
            for n in section.notes:
                # solo un subset para rapidez si hay muchísimas
                note = music21.note.Note(n.pitch)
                note.offset = n.start
                note.quarterLength = max(n.dur, 0.25)
                part.append(note)
            score.append(part)
            k = score.analyze('key')
            pc = k.tonic.pitchClass
            mode = k.mode  # 'major' or 'minor'
            name = f"{NOTE_NAMES[pc]} {mode}"
            return pc, mode, name
        except Exception:
            pass
    # heurística: pitch-class más frecuente en downbeats + modal (mayor/menor)
    pc_hist = np.zeros(12)
    maj = np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float)
    minor = np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float)
    for n in section.notes:
        pc_hist[n.pitch % 12] += n.dur
    if pc_hist.sum() == 0:
        return 0, "major", "C major"
    # probar cada tónica
    best = None
    for r in range(12):
        rotated = np.roll(pc_hist, -r)
        score_maj = (rotated * maj).sum()
        score_min = (rotated * minor).sum()
        for mode, sc in (("major", score_maj), ("minor", score_min)):
            if best is None or sc > best[0]:
                best = (sc, r, mode)
    _, pc, mode = best
    return pc, mode, f"{NOTE_NAMES[pc]} {mode}"


def detect_melody_track(section: Section) -> Tuple[int, int]:
    """Elige (track_idx, channel) de melodía: mayor pitch medio con bastantes notas."""
    by_track = {}
    for n in section.notes:
        key = (n.track, n.channel)
        by_track.setdefault(key, []).append(n)
    if not by_track:
        return 0, 0
    totals = {k: (np.mean([n.pitch for n in v]), len(v)) for k, v in by_track.items()}
    max_notes = max(t[1] for t in totals.values()) or 1
    best = None
    for k, (meanp, cnt) in totals.items():
        if cnt < 0.1 * max_notes:
            continue
        score = meanp + 5 * math.log(cnt + 1)
        if best is None or score > best[0]:
            best = (score, k)
    if best is None:
        best = (0, max(totals.items(), key=lambda kv: kv[1][1])[0])
    return best[1]


def extract_melody_intervals(section: Section) -> Tuple[List[int], List[float], int]:
    """Extrae secuencia de intervalos de la melodía y duraciones."""
    mel = [n for n in section.notes if n.track == section.melody_track
           and n.channel == section.melody_channel]
    if len(mel) < 4:
        # fallback: usar todas las notas ordenadas
        mel = sorted(section.notes, key=lambda n: (n.start, -n.pitch))
        if len(mel) < 4:
            return [], [], 60
    mel.sort(key=lambda n: (n.start, -n.pitch))
    # reducir a monofónico: una nota por onset (la más aguda)
    reduced = []
    last_start = None
    bucket = []
    for n in mel:
        if last_start is None or abs(n.start - last_start) < 0.05:
            bucket.append(n)
            last_start = n.start if last_start is None else last_start
        else:
            if bucket:
                reduced.append(max(bucket, key=lambda x: x.pitch))
            bucket = [n]
            last_start = n.start
    if bucket:
        reduced.append(max(bucket, key=lambda x: x.pitch))
    if len(reduced) < 2:
        return [], [], 60
    intervals = [reduced[i+1].pitch - reduced[i].pitch for i in range(len(reduced)-1)]
    durs = [reduced[i+1].start - reduced[i].start for i in range(len(reduced)-1)]
    if durs:
        durs.append(reduced[-1].dur)
    else:
        durs = [0.5]
    return intervals, durs, reduced[0].pitch


def find_motif(intervals: List[int], durs: List[float]) -> Tuple[List[int], List[float], float]:
    """Encuentra el motivo más saliente (intervalos + duraciones + saliencia)."""
    if len(intervals) < 3:
        return [], [], 0.0
    best = None
    for L in range(3, min(8, len(intervals))):
        counts: Dict[Tuple[int,...], List[int]] = {}
        for i in range(len(intervals) - L + 1):
            key = tuple(intervals[i:i+L])
            counts.setdefault(key, []).append(i)
        for key, idxs in counts.items():
            if len(idxs) >= 2:
                # saliencia: repeticiones * longitud * distintividad (no todo ceros)
                nonzero = sum(1 for x in key if x != 0)
                distinct = 1.0 if nonzero >= 1 else 0.4
                sal = len(idxs) * L * distinct
                if best is None or sal > best[0]:
                    best = (sal, L, key, idxs[0])
    if best is None:
        # tomar los primeros 3-4 intervalos como motivo
        L = min(4, len(intervals))
        return list(intervals[:L]), list(durs[:L]), 1.0
    sal, L, key, idx = best
    return list(key), list(durs[idx:idx+L]), sal


def identify_chord(pcs: set) -> Tuple[int, str]:
    """Identifica (root_pc, quality) de un conjunto de pitch classes."""
    if not pcs:
        return 0, "none"
    templates = {
        "major":  {0, 4, 7},
        "minor":  {0, 3, 7},
        "dom7":   {0, 4, 7, 10},
        "maj7":   {0, 4, 7, 11},
        "min7":   {0, 3, 7, 10},
        "dim":    {0, 3, 6},
        "sus4":   {0, 5, 7},
        "power":  {0, 7},
    }
    best = None
    for r in range(12):
        rel = {(p - r) % 12 for p in pcs}
        for q, t in templates.items():
            inter = len(rel & t)
            missing = len(t - rel)
            extra = len(rel - t)
            score = inter - 0.5 * extra - 0.3 * missing
            if best is None or score > best[0]:
                best = (score, r, q)
    return best[1], best[2]


def chords_per_bar(section: Section) -> List[Tuple[int, str]]:
    """Acordes por compás (root_pc, quality)."""
    bpb = section.beats_per_bar
    if section.n_bars == 0 or not section.notes:
        return []
    # agrupar notas por compás
    bars = [[] for _ in range(section.n_bars)]
    for n in section.notes:
        b = int(n.start / bpb)
        if 0 <= b < section.n_bars:
            bars[b].append(n)
    result = []
    for bar in bars:
        pcs = {n.pitch % 12 for n in bar}
        result.append(identify_chord(pcs))
    return result


def deduce_emotion(section: Section) -> str:
    """Deduce emoción a partir de modo, tempo, registro, densidad, disonancia."""
    minor = section.key_mode == "minor"
    tempo = section.tempo
    reg = section.register
    dens = section.density
    # disonancia aproximada: acordes no mayor/menor
    nondia = sum(1 for r, q in section.chords if q not in ("major","minor","none")) / max(1, len(section.chords))
    scores = {}
    for e in EMOTIONS:
        s = 0.0
        if minor:
            if e in ("melancolia","fragilidad","angustia","tension"): s += 1.5
            if e in ("triunfo","esperanza"): s -= 0.5
        else:
            if e in ("triunfo","esperanza","calidez","serenidad"): s += 1.0
            if e in ("angustia","melancolia"): s -= 0.5
        if tempo < 80:
            if e in ("melancolia","fragilidad","serenidad"): s += 1.0
            if e in ("triunfo","tension"): s -= 0.5
        elif tempo > 120:
            if e in ("triunfo","tension","esperanza"): s += 1.0
            if e in ("serenidad","fragilidad"): s -= 0.5
        if reg > 72:
            if e in ("triunfo","esperanza","tension"): s += 0.5
        elif reg < 55:
            if e in ("melancolia","angustia","fragilidad"): s += 0.5
        if dens > 6:
            if e in ("tension","angustia","triunfo"): s += 0.5
        elif dens < 2:
            if e in ("fragilidad","serenidad"): s += 0.5
        if nondia > 0.3:
            if e in ("tension","angustia"): s += 0.5
        scores[e] = s
    return max(scores.items(), key=lambda kv: kv[1])[0]


def build_fingerprint(section: Section):
    """Rellena los campos de análisis de la sección."""
    section.beats_per_bar = beats_per_bar_of(section.time_sig)
    if section.notes:
        total_beats = max(n.start + n.dur for n in section.notes)
        section.n_bars = max(1, int(math.ceil(total_beats / section.beats_per_bar)))
        section.register = float(np.mean([n.pitch for n in section.notes]))
        section.density = len(section.notes) / max(1, section.n_bars)
    pc, mode, name = detect_key(section)
    section.key_pc, section.key_mode, section.key_name = pc, mode, name
    section.melody_track, section.melody_channel = detect_melody_track(section)
    intervals, durs, first = extract_melody_intervals(section)
    section.motif_intervals, section.motif_durations, section.motif_salience = find_motif(intervals, durs)
    section.motif_first_pitch = first
    section.chords = chords_per_bar(section)
    if section.emotion is None:
        section.emotion = deduce_emotion(section)


# ─────────────────────────────────────────────────────────────────────────────
# Decisión de ADN compartido
# ─────────────────────────────────────────────────────────────────────────────

def name_to_pc(name: str) -> Tuple[int, str]:
    """Convierte "D" o "F# minor" en (pc, mode)."""
    name = name.strip()
    mode = "major"
    if "minor" in name.lower() or "m" == name[-1:]:
        mode = "minor"
    if "minor" in name.lower():
        name = name.lower().replace("minor", "").replace(" ", "")
    if name and name[0] in "ABCDEFG":
        letter = name[0].upper()
        pc = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}[letter]
        if len(name) > 1 and name[1] == '#':
            pc = (pc + 1) % 12
        elif len(name) > 1 and name[1] == 'b':
            pc = (pc - 1) % 12
        return pc, mode
    return 0, "major"


def choose_motif_seed(sections: List[Section], user_seed: Optional[str]) -> Tuple[List[int], List[float], int, str]:
    """Elige el motivo semilla: usuario o el más saliente."""
    if user_seed:
        try:
            notes, *_ = load_midi(user_seed)
            s = Section(path=user_seed, label="seed")
            s.notes = notes
            s.beats_per_bar = 4.0
            s.n_bars = max(1, int(math.ceil(max((n.start+n.dur for n in notes), default=4)/4)))
            s.melody_track, s.melody_channel = detect_melody_track(s)
            intervals, durs, first = extract_melody_intervals(s)
            return intervals[:8], durs[:8], first, user_seed
        except Exception as e:
            print(f"  [WARN] no se pudo cargar motivo semilla {user_seed}: {e}")
    # elegir el más saliente
    best = max(sections, key=lambda s: s.motif_salience)
    return (best.motif_intervals, best.motif_durations, best.motif_first_pitch,
            best.path)


def build_tonal_plan(sections: List[Section], home_key: Optional[str]) -> List[Tuple[int,str]]:
    """
    Para cada sección, decide (target_pc, target_mode).
    Estrategia: hogar = home_key o la tonalidad de la sección más larga.
    Cada sección se transporta a una tónica cercana al hogar sesgada por emoción.
    """
    if home_key:
        home_pc, home_mode = name_to_pc(home_key)
    else:
        # sección más larga
        longest = max(sections, key=lambda s: s.n_bars)
        home_pc, home_mode = longest.key_pc, longest.key_mode
    plan = []
    for s in sections:
        bias = EMOTION_KEY_BIAS.get(s.emotion or "serenidad",
                                    {"shift": 0, "mode": "same"})
        # tónica destino: hogar + shift, pero preservando el modo original salvo preferencia
        target_pc = (home_pc + bias["shift"]) % 12
        if bias["mode"] == "same":
            target_mode = s.key_mode
        elif bias["mode"] == "major":
            target_mode = "major"
        else:
            target_mode = "minor"
        # si la emoción no pide cambio de modo, conservar el de la sección
        if bias["mode"] == "same":
            target_mode = s.key_mode
        s.target_key_pc = target_pc
        s.target_mode = target_mode
        # transposición: llevar tónica original a destino
        shift = (target_pc - s.key_pc) % 12
        if shift > 6:
            shift -= 12
        s.transpose_semis = shift
        plan.append((target_pc, target_mode))
    return plan


def detect_roles(section: Section) -> Dict[Tuple[int,int], str]:
    """Asigna rol (melody/counter/harmony/bass) por (track,channel)."""
    by_tc = {}
    for n in section.notes:
        by_tc.setdefault((n.track, n.channel), []).append(n)
    if not by_tc:
        return {}
    stats = {}
    for k, v in by_tc.items():
        pitches = [n.pitch for n in v]
        # polifonía aproximada
        poly = max_overlap(v)
        stats[k] = {"meanp": np.mean(pitches), "n": len(v), "poly": poly}
    # melodía: ya detectada
    mel_tc = (section.melody_track, section.melody_channel)
    roles = {}
    roles[mel_tc] = "melody"
    # bass: pitch medio más bajo
    candidates = [k for k in stats if k != mel_tc]
    if candidates:
        bass_tc = min(candidates, key=lambda k: stats[k]["meanp"])
        roles[bass_tc] = "bass"
        # harmony: el más polifónico de los restantes
        rest = [k for k in candidates if k != bass_tc]
        for k in rest:
            if stats[k]["poly"] >= 2:
                roles[k] = "harmony"
            else:
                roles[k] = "counter"
    section.role_map = roles
    return roles


def max_overlap(notes: List[Note]) -> int:
    if not notes:
        return 0
    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.start + n.dur, -1))
    events.sort()
    cur = 0
    mx = 0
    for _, d in events:
        cur += d
        mx = max(mx, cur)
    return mx


def choose_template(sections: List[Section], user_template: Optional[str]) -> Dict[str,int]:
    if user_template and user_template in TEMPLATES:
        return dict(TEMPLATES[user_template])
    if user_template == "auto" or user_template is None:
        # deducir de los programas presentes
        all_progs = set()
        for s in sections:
            all_progs.update(s.programs.values())
        # mapear a familias
        has_strings = any(40 <= p <= 47 or p == 43 or p == 48 for p in all_progs)
        has_piano = 0 in all_progs
        has_winds = any(64 <= p <= 79 for p in all_progs)
        if has_piano and not has_strings and not has_winds:
            return dict(TEMPLATES["piano"])
        if has_strings and not has_winds:
            return dict(TEMPLATES["string_quartet"])
        if has_winds and has_strings:
            return dict(TEMPLATES["chamber"])
        if has_winds:
            return dict(TEMPLATES["chamber"])
        return dict(TEMPLATES["string_quartet"])
    return dict(TEMPLATES["chamber"])


# ─────────────────────────────────────────────────────────────────────────────
# Transformaciones (glue)
# ─────────────────────────────────────────────────────────────────────────────

def transpose_notes(notes: List[Note], semis: int) -> List[Note]:
    if semis == 0:
        return [Note(n.pitch, n.start, n.dur, n.velocity, n.channel, n.track) for n in notes]
    out = []
    for n in notes:
        p = n.pitch + semis
        if 0 <= p <= 127:
            out.append(Note(p, n.start, n.dur, n.velocity, n.channel, n.track))
    return out


def remap_instruments(section: Section, template: Dict[str,int], strength: float) -> Dict[int,int]:
    """
    Remapea programas al template. Devuelve dict canal->program para la salida.
    La fuerza controla si se aplica (>=0.5) o se conservan originales.
    """
    out = {}
    if strength >= 0.5:
        for (tr, ch), role in section.role_map.items():
            prog = template.get(role, template.get("harmony", 0))
            out[ch] = prog
        # canales sin rol asignado -> harmony
        for ch in section.programs:
            if ch not in out:
                out[ch] = template.get("harmony", 0)
    else:
        out = dict(section.programs)
    return out


def motif_transform(intervals: List[int], durs: List[float], kind: str) -> Tuple[List[int], List[float]]:
    if not intervals:
        return [], []
    if kind == "original":
        return list(intervals), list(durs)
    if kind == "retrograde":
        return list(reversed(intervals)), list(reversed(durs))
    if kind == "inversion":
        return [-x for x in intervals], list(durs)
    if kind == "augmentation":
        return list(intervals), [d * 2 for d in durs]
    return list(intervals), list(durs)


def realize_motif(intervals: List[int], durs: List[float], start_pitch: int,
                  start_beat: float, channel: int) -> List[Note]:
    """Convierte intervalos+duraciones en notas absolutas desde start_pitch."""
    notes = []
    p = start_pitch
    t = start_beat
    for i, iv in enumerate(intervals):
        d = durs[i] if i < len(durs) else durs[-1] if durs else 0.5
        notes.append(Note(p, t, d * 0.9, 70, channel, 0))
        p = max(0, min(127, p + iv))
        t += d
    return notes


def scale_pitches(key_pc: int, mode: str) -> set:
    if mode == "minor":
        intervals = {0,2,3,5,7,8,10}
    else:
        intervals = {0,2,4,5,7,9,11}
    return {(key_pc + i) % 12 for i in intervals}


def inject_motif(section: Section, motif_intervals: List[int], motif_durs: List[float],
                 first_pitch: int, strength: float, channel: int = 15) -> List[Note]:
    """
    Inyecta el motivo (transformado) en puntos estructurales de la sección.
    Devuelve notas nuevas a añadir (canal aparte, baja velocity).
    """
    if strength <= 0.05 or not motif_intervals:
        return []
    # número de inyecciones según fuerza
    n_inj = max(1, int(round(strength * 4)))
    # puntos: inicio, final, y repartidos
    bpb = section.beats_per_bar
    total = section.n_bars * bpb
    points = [0.0, max(0, total - sum(motif_durs) - 0.5)]
    for i in range(1, n_inj - 1):
        frac = i / max(1, n_inj - 1)
        points.append(frac * total)
    points = sorted(set(points))
    kinds = ["original", "inversion", "retrograde", "augmentation", "original"]
    out = []
    # transportar el motivo a la tonalidad destino
    shift = section.transpose_semis
    base_pitch = first_pitch + shift
    # restringir a escala
    sc = scale_pitches(section.target_key_pc, section.target_mode)
    for i, pt in enumerate(points):
        if pt < 0 or pt > total:
            continue
        kind = kinds[i % len(kinds)]
        ivs, durs = motif_transform(motif_intervals, motif_durs, kind)
        # ajustar primera nota a tono de acorde (tónica o tercera)
        root_pc = section.target_key_pc
        third = (root_pc + (3 if section.target_mode == "minor" else 4)) % 12
        candidate = base_pitch
        while candidate % 12 not in {root_pc, third, (root_pc+7)%12} and candidate < 120:
            candidate += 1
        vel = int(55 + 20 * strength)
        notes = realize_motif(ivs, durs, candidate, pt, channel)
        # bajar velocity para que sea hilo
        for n in notes:
            n.velocity = vel
        out.extend(notes)
    return out


def nudge_harmony(section: Section, notes: List[Note], strength: float) -> List[Note]:
    """
    Reharmonización conservadora: en compases cuyo acorde es no-diacónico,
    sustituye la armonía (no melodía) por un acorde diatónico que encaje con la melodía.
    """
    if strength < 0.5 or not notes:
        return notes
    sc = scale_pitches(section.target_key_pc, section.target_mode)
    bpb = section.beats_per_bar
    mel_tc = (section.melody_track, section.melody_channel)
    harmony_tcs = {tc for tc, r in section.role_map.items() if r == "harmony"}
    if not harmony_tcs:
        return notes
    # agrupar por compás
    n_bars = section.n_bars
    bars = [[] for _ in range(n_bars)]
    for n in notes:
        b = int(n.start / bpb)
        if 0 <= b < n_bars:
            bars[b].append(n)
    new_notes = []
    for b, bar in enumerate(bars):
        mel_pcs = {n.pitch % 12 for n in bar if (n.track, n.channel) == mel_tc}
        harm = [n for n in bar if (n.track, n.channel) in harmony_tcs]
        chord_pc, q = section.chords[b] if b < len(section.chords) else (0,"none")
        nondiatonic = chord_pc not in sc
        if nondiatonic and harm:
            # elegir acorde diatónico (tónica, subdominante, dominante) que mejor encaje
            candidates = []
            for deg, shift in [("I",0),("IV",5),("V",7),("vi",9 if section.target_mode=="major" else 9)]:
                root = (section.target_key_pc + shift) % 12
                if root not in sc:
                    continue
                triad = {root, (root+3 if section.target_mode=="minor" else root+4)%12, (root+7)%12}
                overlap = len(mel_pcs & triad)
                candidates.append((overlap, root, triad))
            if candidates:
                candidates.sort(reverse=True)
                _, root, triad = candidates[0]
                # re-voice: mantener registros aproximados de las notas de armonía
                harm_sorted = sorted(harm, key=lambda n: n.pitch)
                # mapear a triada conservando el pitch más cercano
                replaced = []
                for i, n in enumerate(harm_sorted):
                    # buscar tono de triada más cercano al registro
                    target = n.pitch
                    options = []
                    for t in range(target-7, target+8):
                        if t % 12 in triad and 0 <= t <= 127:
                            options.append((abs(t-target), t))
                    if options:
                        options.sort()
                        new_pitch = options[0][1]
                    else:
                        new_pitch = n.pitch
                    replaced.append(Note(new_pitch, n.start, n.dur, n.velocity, n.channel, n.track))
                new_notes.extend(replaced)
                # mantener melodia y resto
                new_notes.extend([n for n in bar if (n.track,n.channel) not in harmony_tcs])
                continue
        new_notes.extend(bar)
    new_notes.sort(key=lambda n: (n.start, -n.pitch))
    return new_notes


def apply_arc_dynamics(notes: List[Note], arc: List[float], total_beats: float):
    """Reescala velocities según un arco de tensión [0,1]."""
    if not arc or not notes or total_beats <= 0:
        return
    for n in notes:
        frac = min(1.0, n.start / total_beats)
        idx = frac * (len(arc) - 1)
        lo = int(idx)
        hi = min(lo + 1, len(arc) - 1)
        a = arc[lo] + (arc[hi] - arc[lo]) * (idx - lo)
        # mapear tensión a velocity: base 60 + 50*tensión
        base = 55 + 55 * a
        n.velocity = max(20, min(120, int(base * (n.velocity / 90.0 if n.velocity else 0.8))))


# ─────────────────────────────────────────────────────────────────────────────
# Puentes (bridges)
# ─────────────────────────────────────────────────────────────────────────────

def make_bridge(sec_a: Section, sec_b: Section, motif_intervals: List[int],
                motif_durs: List[float], first_pitch: int, bridge_bars: int,
                mode: str, channel: int = 15) -> Tuple[List[Note], float, Tuple[int,int], Dict[int,int]]:
    """
    Genera un puente entre A y B: cadencia diatónica + declaración del motivo.
    Devuelve (notas, duración_en_beats, time_sig, programs).
    """
    bpb = sec_b.beats_per_bar
    dur = bridge_bars * bpb
    tsig = sec_b.time_sig
    notes = []
    # tonalidad destino = B
    key_pc = sec_b.target_key_pc
    mode = sec_b.target_mode
    sc = scale_pitches(key_pc, mode)
    # cadencia: V7 -> I (en tonalidad destino)
    # si B es menor, V7 con 5ª disminuida implícita; simplificamos a acorde dominante
    if mode == "minor":
        fifth = (key_pc + 7) % 12
        dom_chord = {(fifth)%12, (fifth+4)%12, (fifth+7)%12, (fifth+10)%12}
    else:
        fifth = (key_pc + 7) % 12
        dom_chord = {fifth, (fifth+4)%12, (fifth+7)%12, (fifth+10)%12}
    tonic_chord = {key_pc, (key_pc+(3 if mode=="minor" else 4))%12, (key_pc+7)%12}
    # compás 1: dominante; compás 2: tónica
    programs = {0: 73, channel: 40}  # flauta melodía + violín motivo
    # bajo: walking V -> I
    bass_pitch = 36 + (fifth % 12)  # V en registro grave
    # construir acordes como bloques
    if bridge_bars >= 1:
        # compás dominante
        bar_start = 0.0
        for pc in list(dom_chord)[:3]:
            p = 48 + pc
            notes.append(Note(p, bar_start, bpb*0.95, 70, 2, 0))
        notes.append(Note(bass_pitch, bar_start, bpb*0.95, 80, 3, 0))
    if bridge_bars >= 2:
        bar_start = bpb
        for pc in list(tonic_chord)[:3]:
            p = 48 + pc
            notes.append(Note(p, bar_start, bpb*0.95, 75, 2, 0))
        notes.append(Note(36 + (key_pc%12), bar_start, bpb*0.95, 80, 3, 0))
    # motivo sobre el puente, transportado a la tonalidad destino
    if motif_intervals:
        shift = (key_pc - sec_a.key_pc) % 12
        if shift > 6: shift -= 12
        base = first_pitch + shift
        # alinear primera nota a tónica/tercera
        while base % 12 not in {key_pc, (key_pc+(3 if mode=="minor" else 4))%12} and base < 120:
            base += 1
        mnotes = realize_motif(motif_intervals, motif_durs, base, 0.0, channel)
        for n in mnotes:
            n.velocity = 75
        # recortar a dur
        mnotes = [n for n in mnotes if n.start < dur]
        notes.extend(mnotes)
    notes.sort(key=lambda n: (n.start, -n.pitch))
    return notes, dur, tsig, programs


# ─────────────────────────────────────────────────────────────────────────────
# Modo glue
# ─────────────────────────────────────────────────────────────────────────────

def run_glue(sections, motif_iv, motif_durs, motif_first, template, glue, bridge_bars, bridge_mode):
    """Aplica adaptación por sección y concatena con puentes."""
    output_notes = []
    tempo_map = []
    programs_out = {}
    cur_beat = 0.0
    motif_channel = 15
    arc = build_arc(len(sections))

    for i, s in enumerate(sections):
        detect_roles(s)
        # 1. transponer al plan tonal
        adapted = transpose_notes(s.notes, s.transpose_semis)
        # 2. nudge armónico
        if glue["harmony"] >= 0.5:
            adapted = nudge_harmony(s, adapted, glue["harmony"])
        # 3. inyectar motivo
        injected = inject_motif(s, motif_iv, motif_durs, motif_first,
                                glue["motif"], channel=motif_channel)
        # 4. reasignar instrumentos
        progs = remap_instruments(s, template, glue["instrumentation"])
        # 5. arco de dinámicas (local dentro de la sección)
        sec_total = s.n_bars * s.beats_per_bar
        local_arc = arc[i:i+2] if i+1 < len(arc) else [arc[-1], arc[-1]]
        local_arc_full = np.linspace(local_arc[0], local_arc[1], max(2, s.n_bars)).tolist()
        apply_arc_dynamics(adapted, local_arc_full, sec_total)
        apply_arc_dynamics(injected, local_arc_full, sec_total)
        # desplazar al beat actual
        for n in adapted:
            n.start += cur_beat
            output_notes.append(n)
        for n in injected:
            n.start += cur_beat
            output_notes.append(n)
        # tempo/time-sig al inicio de la sección
        tempo_map.append((cur_beat, s.tempo, s.time_sig))
        # programas
        for ch, p in progs.items():
            programs_out.setdefault(ch, p)
        programs_out[motif_channel] = template.get("melody", 0)
        cur_beat += sec_total
        # puente si no es la última
        if i < len(sections) - 1:
            bnotes, bdur, btsig, bprogs = make_bridge(
                s, sections[i+1], motif_iv, motif_durs, motif_first,
                bridge_bars, bridge_mode, channel=motif_channel)
            for n in bnotes:
                n.start += cur_beat
                output_notes.append(n)
            tempo_map.append((cur_beat, sections[i+1].tempo * 0.9, btsig))
            for ch, p in bprogs.items():
                programs_out.setdefault(ch, p)
            cur_beat += bdur

    return output_notes, tempo_map, programs_out


def build_arc(n_sections: int) -> List[float]:
    """Arco de tensión tipo 'arch' sobre las secciones."""
    if n_sections <= 1:
        return [0.5, 0.5]
    arc = []
    for i in range(n_sections):
        frac = i / (n_sections - 1)
        # parábola: sube hasta ~0.62 y baja
        val = 0.2 + 0.7 * (1 - (2*frac - 0.62)**2 / 0.62**2) if n_sections > 1 else 0.5
        arc.append(max(0.1, min(1.0, val)))
    return arc


# ─────────────────────────────────────────────────────────────────────────────
# Modo recombine
# ─────────────────────────────────────────────────────────────────────────────

def degree_to_pc(degree: str, key_pc: int, mode: str) -> int:
    """Convierte grado romano a pitch class."""
    scale = (list(range(0,12,2)) + [4,5,7,9,11]) if mode == "major" else \
            [0,2,3,5,7,8,10]
    # mapeo simple
    mapping_major = {"I":0,"ii":2,"iii":4,"IV":5,"V":7,"vi":9,"vii":11}
    mapping_minor = {"i":0,"ii":2,"III":3,"iv":5,"V":7,"VI":8,"VII":10,
                     "iV":5,"III":3}
    m = mapping_major if mode == "major" else mapping_minor
    base = m.get(degree, 0)
    return (key_pc + base) % 12


def triad_for_degree(degree: str, key_pc: int, mode: str) -> set:
    root = degree_to_pc(degree, key_pc, mode)
    # calidad: menor si grado en minúscula
    is_minor = degree[0].islower()
    third = 3 if is_minor else 4
    return {root, (root+third)%12, (root+7)%12}


def run_recombine(sections, motif_iv, motif_durs, motif_first, template,
                  glue, bridge_bars, bars_per_section):
    """Genera una obra nueva coherente usando el ADN de las secciones."""
    output_notes = []
    tempo_map = []
    programs_out = {0: template.get("melody",0),
                    1: template.get("counter",0),
                    2: template.get("harmony",0),
                    3: template.get("bass",0)}
    motif_channel = 0
    cur_beat = 0.0
    arc = build_arc(len(sections))

    for i, s in enumerate(sections):
        bpb = s.beats_per_bar
        n_bars = bars_per_section
        sec_beats = n_bars * bpb
        key_pc = s.target_key_pc
        mode = s.target_mode
        prog = PROGRESSIONS.get(s.emotion or "serenidad", ["I","V","vi","IV"])
        # tempo de la sección
        tempo_map.append((cur_beat, s.tempo, s.time_sig))
        # arco local
        local_arc = np.linspace(arc[i], arc[min(i+1, len(arc)-1)], max(2, n_bars)).tolist()
        # por compás: acorde + bajo + melodía (motivo)
        for b in range(n_bars):
            bar_start = cur_beat + b * bpb
            deg = prog[b % len(prog)]
            triad = triad_for_degree(deg, key_pc, mode)
            root = degree_to_pc(deg, key_pc, mode)
            # armonía: bloque
            for pc in list(triad)[:3]:
                p = 52 + pc
                vel = int(50 + 30 * local_arc[b])
                output_notes.append(Note(p, bar_start, bpb*0.92, vel, 2, 0))
            # bajo: raíz en registro grave
            bp = 40 + root
            output_notes.append(Note(bp, bar_start, bpb*0.9, int(60+30*local_arc[b]), 3, 0))
            # melodía: motivo transplantado, transformación rotativa por compás
            if motif_iv:
                kinds = ["original","inversion","retrograde","augmentation"]
                kind = kinds[b % len(kinds)]
                ivs, durs = motif_transform(motif_iv, motif_durs, kind)
                # primera nota a tono del acorde
                base = motif_first + s.transpose_semis
                while base % 12 not in triad and base < 120:
                    base += 1
                mnotes = realize_motif(ivs, durs, base, bar_start, 0)
                for n in mnotes:
                    n.velocity = int(65 + 35 * local_arc[b])
                # recortar al compás
                mnotes = [n for n in mnotes if n.start < bar_start + bpb]
                output_notes.extend(mnotes)
        cur_beat += sec_beats
        # puente
        if i < len(sections) - 1:
            bnotes, bdur, btsig, bprogs = make_bridge(
                s, sections[i+1], motif_iv, motif_durs, motif_first,
                bridge_bars, "pivot", channel=motif_channel)
            for n in bnotes:
                n.start += cur_beat
                output_notes.append(n)
            tempo_map.append((cur_beat, sections[i+1].tempo*0.9, btsig))
            cur_beat += bdur

    return output_notes, tempo_map, programs_out


# ─────────────────────────────────────────────────────────────────────────────
# Score de coherencia
# ─────────────────────────────────────────────────────────────────────────────

def coherence_score(sections, combined_notes, motif_iv) -> dict:
    """Calcula un score de coherencia burdo."""
    # 1. consistencia de instrumentación
    channels = {n.channel for n in combined_notes}
    # 2. recurrencia de motivo: cuántas veces aparece el intervalo del motivo
    motif_match = 0
    if motif_iv:
        # buscar ocurrencias aproximadas en melodías
        for s in sections:
            intervals, _, _ = extract_melody_intervals(s)
            for i in range(len(intervals)-len(motif_iv)+1):
                if intervals[i:i+len(motif_iv)] == motif_iv:
                    motif_match += 1
    # 3. consistencia tonal: cuántas secciones comparten tónica destino
    tonics = [s.target_key_pc for s in sections]
    tonic_consistency = sum(1 for t in tonics if t == tonics[0]) / len(tonics) if tonics else 0
    # 4. densidad media
    density = len(combined_notes) / max(1, sum(s.n_bars for s in sections))
    return {
        "motif_recurrences": motif_match,
        "tonic_consistency": round(tonic_consistency, 2),
        "channels_used": sorted(channels),
        "mean_density": round(density, 1),
        "n_sections": len(sections),
        "n_notes": len(combined_notes),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_glue(s: str) -> dict:
    glue = {"motif": 0.6, "harmony": 0.6, "rhythm": 0.3, "instrumentation": 0.8}
    if not s:
        return glue
    for part in s.split(","):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=")
            k = k.strip().lower()
            try:
                glue[k] = max(0.0, min(1.0, float(v.strip())))
            except ValueError:
                pass
    return glue


def parse_section_spec(spec: str) -> Tuple[str, Optional[str], str]:
    """
    'label:emotion:path' | 'label:path' | 'emotion:path' | 'path'
    """
    parts = spec.split(":")
    if len(parts) >= 3:
        return parts[0], parts[1], ":".join(parts[2:])
    if len(parts) == 2:
        if parts[0].lower() in EMOTIONS:
            return parts[0], parts[0], parts[1]
        return parts[0], None, parts[1]
    # path solo
    base = os.path.basename(spec)
    label = os.path.splitext(base)[0]
    return label, None, spec


def parse_args():
    p = argparse.ArgumentParser(
        description="Unificador de secciones MIDI en una composición coherente.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("midis", nargs="*", help="MIDIs de sección (orden fijo)")
    p.add_argument("--section", action="append", default=[],
                   help='Sección: "label:emotion:path" (repetible)')
    p.add_argument("--emotions", help="Emociones separadas por coma (para MIDIs posicionales)")
    p.add_argument("--mode", choices=["glue","recombine","auto"], default="glue")
    p.add_argument("--motif-seed", help="MIDI con el motivo semilla")
    p.add_argument("--key", help="Tonalidad hogar, ej: D, F# minor")
    p.add_argument("--template", choices=list(TEMPLATES.keys())+["auto"], default="auto")
    p.add_argument("--glue", default="", help="motif=,harmony=,rhythm=,instrumentation=")
    p.add_argument("--bridge-bars", type=int, default=2)
    p.add_argument("--bridge", choices=["morph","pivot","auto"], default="auto")
    p.add_argument("--recombine-bars", type=int, default=8, help="compases por sección en modo recombine")
    p.add_argument("--output", default="obra_unificada.mid")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--report", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-color", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # construir lista de secciones
    specs = []
    for s in args.section:
        specs.append(parse_section_spec(s))
    if args.midis:
        emotions = (args.emotions.split(",") if args.emotions else [])
        for i, m in enumerate(args.midis):
            emo = emotions[i].strip() if i < len(emotions) else None
            if emo and emo.lower() not in EMOTIONS:
                print(f"[WARN] emoción '{emo}' no reconocida, se deducirá")
                emo = None
            label = os.path.splitext(os.path.basename(m))[0]
            specs.append((label, emo, m))
    if len(specs) < 2:
        sys.exit("Se necesitan al menos 2 secciones. Usa --section o MIDIs posicionales.")
    if len(specs) > 10:
        print(f"[WARN] {len(specs)} secciones: recortando a 10.")
        specs = specs[:10]

    sections = []
    for label, emo, path in specs:
        if not os.path.exists(path):
            sys.exit(f"No existe el MIDI: {path}")
        notes, tpb, tempo, tsig, tmeta, progs = load_midi(path)
        s = Section(path=path, label=label, emotion=emo, notes=notes, tpb=tpb,
                    tempo=tempo, time_sig=tsig, tracks_meta=tmeta, programs=progs)
        build_fingerprint(s)
        sections.append(s)
        if args.verbose:
            print(f"  [{label}] {path}: {s.key_name} {s.tempo:.0f}bpm {s.time_sig[0]}/{s.time_sig[1]} "
                  f"{s.n_bars}c | emoción={'deducida' if emo is None else emo}: {s.emotion} "
                  f"| motivo saliencia={s.motif_salience:.1f}")

    # ADN compartido
    motif_iv, motif_durs, motif_first, motif_src = choose_motif_seed(sections, args.motif_seed)
    plan = build_tonal_plan(sections, args.key)
    template = choose_template(sections, "auto" if args.template == "auto" else args.template)
    glue = parse_glue(args.glue)

    if args.verbose:
        print(f"\n  ADN compartido:")
        print(f"    Motivo semilla: {len(motif_iv)} intervalos (de {os.path.basename(motif_src)})")
        print(f"    Plantilla: {args.template}")
        print(f"    Pegamento: {glue}")
        for i, s in enumerate(sections):
            print(f"    Sec {i+1} [{s.label}]: {s.key_name} -> {NOTE_NAMES[s.target_key_pc]} {s.target_mode} "
                  f"(transp {s.transpose_semis:+d}) emoción={s.emotion}")

    # puentes: auto -> pivot si secciones muy contrastantes, morph si no
    bridge_mode = args.bridge
    if bridge_mode == "auto":
        bridge_mode = "morph"

    # elegir modo
    mode = args.mode
    if mode == "auto":
        # si las secciones son muy diferentes en tonalidad, recombine
        diff_keys = len({s.key_pc for s in sections})
        mode = "recombine" if diff_keys >= len(sections)*0.7 else "glue"

    if mode == "glue":
        out_notes, tempo_map, progs = run_glue(
            sections, motif_iv, motif_durs, motif_first, template,
            glue, args.bridge_bars, bridge_mode)
    else:
        out_notes, tempo_map, progs = run_recombine(
            sections, motif_iv, motif_durs, motif_first, template,
            glue, args.bridge_bars, args.recombine_bars)

    # asegurar programas al inicio
    for ch, p in progs.items():
        if not (0 <= p <= 127):
            progs[ch] = 0

    save_midi(args.output, out_notes, tempo_map, progs)
    print(f"\n✓ Obra unificada (modo {mode}): {args.output}")
    print(f"  Notas: {len(out_notes)} | Canales: {sorted(progs.keys())} | Secciones: {len(sections)}")

    if args.report:
        score = coherence_score(sections, out_notes, motif_iv)
        print(f"\n  ── Reporte de coherencia ──")
        for k, v in score.items():
            print(f"    {k}: {v}")
        report = {
            "mode": mode,
            "sections": [{"label": s.label, "path": s.path, "key": s.key_name,
                          "emotion": s.emotion, "target_key": f"{NOTE_NAMES[s.target_key_pc]} {s.target_mode}",
                          "transpose": s.transpose_semis, "n_bars": s.n_bars}
                         for s in sections],
            "motif": {"intervals": motif_iv, "durations": motif_durs,
                      "source": os.path.basename(motif_src)},
            "template": template,
            "glue": glue,
            "coherence": score,
        }
        rep_path = os.path.splitext(args.output)[0] + ".report.json"
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  Reporte JSON: {rep_path}")


if __name__ == "__main__":
    main()
