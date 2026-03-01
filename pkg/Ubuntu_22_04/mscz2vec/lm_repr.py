"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_REPR  v1.0  —  Representación compacta MIDI ↔ Tensor                   ║
║  Módulo base del sistema de aprendizaje latente multi-instrumento           ║
║                                                                              ║
║  Convierte MIDIs polifónicos multi-pista en tensores compactos              ║
║  [T, 68] y reconstruye MIDIs desde esos tensores.                           ║
║                                                                              ║
║  ESTRUCTURA DEL TENSOR (68 dims por frame de 500ms):                        ║
║    Cuerdas   (dims  0-16): chroma×12, registro×2, densidad, dinámica, onset ║
║    Maderas   (dims 17-33): ídem                                              ║
║    Metales   (dims 34-50): ídem                                              ║
║    Percusión (dims 51-67): ídem (pitch relativo para timbal, energía resto) ║
║                                                                              ║
║  USO COMO SCRIPT (test y visualización):                                    ║
║    python lm_repr.py archivo.mid                                             ║
║    python lm_repr.py archivo.mid --plot         # visualización completa    ║
║    python lm_repr.py archivo.mid --roundtrip    # test reconstrucción       ║
║    python lm_repr.py archivo.mid --info         # estadísticas del tensor   ║
║    python lm_repr.py *.mid --batch-check        # validar corpus completo   ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install mido numpy                                                    ║
║    pip install matplotlib  (solo para --plot)                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import math
import argparse
import numpy as np
from collections import defaultdict

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

FRAME_MS   = 500          # duración de cada frame en ms
TICKS_OUT  = 480          # ticks/beat en MIDI de salida
DEFAULT_BPM = 120

# Familias y sus instrumentos GM (program numbers 0-based)
# Cuerdas: pizzicato, arco, spiccato GM programs
# Redefinidas para reflejar corpus real (pop/jazz/orquestal mixto)
FAMILY_GM = {
    'keys':       list(range(0, 24)),                         # piano, organo, cromaticos
    'strings':    list(range(24, 32)) + list(range(40, 56)), # guitarras + cuerdas orq + coro
    'bass':       list(range(32, 40)),                        # todos los bajos
    'winds':      list(range(56, 80)),                        # metales + saxos + maderas
    'percussion': [],                                         # canal 9 GM siempre
}

# Mapeo instrumento orchestrator.py -> familia del modelo
INSTR_TO_FAMILY = {
    'piano': 'keys', 'electric_piano': 'keys', 'organ': 'keys',
    'harpsichord': 'keys', 'celesta': 'keys', 'vibraphone': 'keys',
    'violin1': 'strings', 'violin2': 'strings', 'viola': 'strings',
    'cello': 'strings', 'contrabass': 'strings',
    'guitar': 'strings', 'nylon_guitar': 'strings', 'steel_guitar': 'strings',
    'string_ensemble': 'strings', 'choir': 'strings',
    'bass_guitar': 'bass', 'electric_bass': 'bass', 'acoustic_bass': 'bass',
    'fretless_bass': 'bass', 'synth_bass': 'bass',
    'flute': 'winds', 'oboe': 'winds', 'clarinet': 'winds', 'bassoon': 'winds',
    'horn': 'winds', 'trumpet': 'winds', 'trombone': 'winds', 'tuba': 'winds',
    'alto_sax': 'winds', 'tenor_sax': 'winds', 'soprano_sax': 'winds',
    'baritone_sax': 'winds', 'brass_section': 'winds',
    'timpani': 'percussion', 'drums': 'percussion', 'drum_kit': 'percussion',
}

FAMILIES = ['keys', 'strings', 'bass', 'winds', 'percussion']
N_FAMILIES = len(FAMILIES)

# Rangos idiomáticos por instrumento (del orchestrator)
INSTRUMENT_RANGES = {
    'violin1':    (55, 96), 'violin2':    (55, 91),
    'viola':      (48, 84), 'cello':      (36, 76), 'contrabass': (28, 60),
    'flute':      (60, 96), 'oboe':       (58, 91),
    'clarinet':   (50, 94), 'bassoon':    (34, 75),
    'horn':       (34, 77), 'trumpet':    (52, 82),
    'trombone':   (34, 72), 'tuba':       (28, 58),
    'timpani':    (41, 65),
}

# Rangos de familia completos (unión de todos sus instrumentos)
FAMILY_RANGES = {
    'keys':       (21, 108),  # piano completo
    'strings':    (28, 96),   # bajo guitarra hasta violin agudo
    'bass':       (28, 67),   # bajos electricos y acusticos
    'winds':      (34, 96),   # tuba hasta piccolo
    'percussion': (35, 81),   # rango GM percusion
}

# Notas GM de percusión más comunes
PERC_NOTES_GM = {
    35: 'kick2', 36: 'kick', 38: 'snare', 39: 'clap',
    40: 'snare2', 41: 'lowtom', 42: 'hihat_c', 43: 'hitom',
    44: 'hihat_p', 45: 'midtom', 46: 'hihat_o', 47: 'midtom2',
    48: 'hightom', 49: 'crash', 50: 'hightom2', 51: 'ride',
}

DIMS_PER_FAMILY = 17   # 12 chroma + 2 registro + 1 densidad + 1 dinámica + 1 onset
TENSOR_DIMS     = N_FAMILIES * DIMS_PER_FAMILY   # 68

# Índices dentro de cada bloque de familia (offsets relativos al base)
IDX_CHROMA_START = 0
IDX_CHROMA_END   = 12
IDX_REG_MID = 12
IDX_REG_RNG = 13
IDX_DENSITY = 14
IDX_DYNAMIC = 15
IDX_ONSET   = 16


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE FAMILIA DESDE PISTA MIDI
# ══════════════════════════════════════════════════════════════════════════════

def detect_family_of_track(track, channel_programs: dict) -> str:
    """
    Dado un track mido y el mapa {channel: program}, devuelve la familia.
    Usa heurísticas en orden de prioridad:
      1. Canal 9 → percusión
      2. Program number → familia GM
      3. Nombre del track → coincidencia textual
      4. Fallback: strings
    """
    # Recopilar canales usados en este track
    channels_used = set()
    prog_used = set()
    track_name = ""

    for msg in track:
        if hasattr(msg, 'channel'):
            channels_used.add(msg.channel)
        if msg.type == 'track_name':
            track_name = msg.name.lower()
        if msg.type == 'program_change':
            prog_used.add(msg.program)

    # Canal 9 → percusión GM
    if 9 in channels_used:
        return 'percussion'

    # Por program number del track
    for p in prog_used:
        for fam, progs in FAMILY_GM.items():
            if p in progs:
                return fam

    # Por program en canal (del mapa global)
    for ch in channels_used:
        p = channel_programs.get(ch, 0)
        for fam, progs in FAMILY_GM.items():
            if p in progs:
                return fam

    # Por nombre del track
    keywords = {
        'strings':   ['violin', 'viola', 'cello', 'bass', 'string', 'cuerd'],
        'woodwinds': ['flute', 'oboe', 'clarinet', 'bassoon', 'madera',
                      'flauta', 'clarinete', 'fagot'],
        'brass':     ['horn', 'trumpet', 'trombone', 'tuba', 'metal',
                      'trompa', 'trompeta', 'tromb'],
        'percussion':['drum', 'perc', 'timpani', 'timbal', 'cymbal', 'snare'],
    }
    for fam, kws in keywords.items():
        if any(k in track_name for k in kws):
            return fam

    return 'strings'   # fallback


# ══════════════════════════════════════════════════════════════════════════════
#  PARSE MIDI → EVENTOS ABSOLUTOS
# ══════════════════════════════════════════════════════════════════════════════

class NoteEvent:
    __slots__ = ('time_sec', 'duration_sec', 'pitch', 'velocity', 'family', 'channel')

    def __init__(self, time_sec, duration_sec, pitch, velocity, family, channel):
        self.time_sec    = time_sec
        self.duration_sec = duration_sec
        self.pitch       = pitch
        self.velocity    = velocity
        self.family      = family
        self.channel     = channel


def midi_to_note_events(file_path: str) -> tuple[list[NoteEvent], float, float]:
    """
    Lee un MIDI y devuelve (lista_de_NoteEvent, duracion_total_seg, bpm_medio).
    Maneja tempo changes, tracks múltiples, y asignación de familia.
    Retorna ([], 0, 120) si el fichero es inválido.
    """
    try:
        mid = MidiFile(file_path)
    except Exception:
        return [], 0.0, DEFAULT_BPM

    tpb = mid.ticks_per_beat

    # ── Mapa global de tempo (ticks absolutos → µs/beat) ─────────────────────
    # Necesitamos el tempo_map para convertir ticks a segundos correctamente.
    # Lo construimos desde el track 0 (o el primero que tenga tempo).
    tempo_map = [(0, 500000)]   # (tick_abs, µs_per_beat)
    abs_tick = 0
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'set_tempo':
                tempo_map.append((abs_tick, msg.tempo))
        break   # solo el primer track para tempo (convención MIDI type 1)

    tempo_map.sort(key=lambda x: x[0])

    def ticks_to_sec(tick_abs: int) -> float:
        sec = 0.0
        prev_tick, prev_tempo = tempo_map[0]
        for i in range(1, len(tempo_map)):
            t_tick, t_tempo = tempo_map[i]
            if tick_abs <= t_tick:
                break
            delta = t_tick - prev_tick
            sec += delta * prev_tempo / (tpb * 1_000_000)
            prev_tick, prev_tempo = t_tick, t_tempo
        remaining = tick_abs - prev_tick
        sec += remaining * prev_tempo / (tpb * 1_000_000)
        return sec

    # ── Mapa global de program por canal ─────────────────────────────────────
    channel_programs: dict[int, int] = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'program_change':
                channel_programs[msg.channel] = msg.program

    # ── Extraer notas por track ───────────────────────────────────────────────
    events: list[NoteEvent] = []

    for track in mid.tracks:
        family = detect_family_of_track(track, channel_programs)

        abs_tick = 0
        open_notes: dict[tuple[int,int], tuple[int, int]] = {}  # (ch, pitch) → (tick_on, vel)

        for msg in track:
            abs_tick += msg.time

            if msg.type == 'program_change':
                channel_programs[msg.channel] = msg.program
                # re-detectar familia si cambia el program
                new_fam = 'percussion' if msg.channel == 9 else None
                if new_fam is None:
                    for fam, progs in FAMILY_GM.items():
                        if msg.program in progs:
                            new_fam = fam
                            break
                if new_fam:
                    family = new_fam

            elif msg.type == 'note_on' and msg.velocity > 0:
                key = (msg.channel, msg.note)
                open_notes[key] = (abs_tick, msg.velocity)

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in open_notes:
                    tick_on, vel = open_notes.pop(key)
                    t_start = ticks_to_sec(tick_on)
                    t_end   = ticks_to_sec(abs_tick)
                    dur     = max(t_end - t_start, 0.05)
                    fam     = 'percussion' if msg.channel == 9 else family
                    events.append(NoteEvent(t_start, dur, msg.note, vel, fam, msg.channel))

        # Cerrar notas que quedaron abiertas (MIDI sin note_off final)
        for (ch, pitch), (tick_on, vel) in open_notes.items():
            t_start = ticks_to_sec(tick_on)
            t_end   = ticks_to_sec(abs_tick)
            dur     = max(t_end - t_start, 0.05)
            fam     = 'percussion' if ch == 9 else family
            events.append(NoteEvent(t_start, dur, pitch, vel, fam, ch))

    if not events:
        return [], 0.0, DEFAULT_BPM

    total_sec = max(e.time_sec + e.duration_sec for e in events)

    # BPM medio a partir del tempo_map
    bpm_sum, bpm_count = 0, 0
    for _, t in tempo_map:
        bpm_sum += 60_000_000 / t
        bpm_count += 1
    bpm_mean = bpm_sum / bpm_count if bpm_count else DEFAULT_BPM

    return events, total_sec, bpm_mean


# ══════════════════════════════════════════════════════════════════════════════
#  EVENTOS → TENSOR [T, 68]
# ══════════════════════════════════════════════════════════════════════════════

def events_to_tensor(events: list[NoteEvent], total_sec: float,
                     frame_ms: int = FRAME_MS) -> np.ndarray:
    """
    Convierte lista de NoteEvent en tensor numpy float32 [T, 68].
    T = ceil(total_sec * 1000 / frame_ms)
    """
    frame_sec = frame_ms / 1000.0
    n_frames  = max(1, math.ceil(total_sec / frame_sec))

    tensor = np.zeros((n_frames, TENSOR_DIMS), dtype=np.float32)

    # Para detección de onsets, llevamos el frame del onset anterior por familia
    last_onset_frame = {f: -999 for f in FAMILIES}

    # Acumular notas por frame y familia
    # Usamos dict: (frame, family) → lista de (pitch, velocity)
    frame_fam_notes: dict[tuple[int,str], list[tuple[int,int]]] = defaultdict(list)
    frame_fam_onsets: dict[tuple[int,str], int] = defaultdict(int)

    for ev in events:
        f_start = int(ev.time_sec / frame_sec)
        f_end   = int((ev.time_sec + ev.duration_sec) / frame_sec)
        fam     = ev.family

        for f in range(f_start, min(f_end + 1, n_frames)):
            frame_fam_notes[(f, fam)].append((ev.pitch, ev.velocity))

        # onset: solo en el frame donde empieza
        if f_start < n_frames:
            frame_fam_onsets[(f_start, fam)] += 1

    # Rellenar tensor
    for (f, fam), notes in frame_fam_notes.items():
        fam_idx = FAMILIES.index(fam)
        base    = fam_idx * DIMS_PER_FAMILY

        pitches    = np.array([p for p, v in notes], dtype=np.float32)
        velocities = np.array([v for p, v in notes], dtype=np.float32)
        n          = len(notes)

        # Chroma (12 dims): suma de velocidades normalizadas por pitch class
        chroma = np.zeros(12, dtype=np.float32)
        for p, v in notes:
            chroma[p % 12] += v / 127.0
        if chroma.max() > 0:
            chroma /= chroma.max()   # normalizar a [0,1]
        tensor[f, base + IDX_CHROMA_START : base + IDX_CHROMA_END] = chroma

        # Registro: pitch medio normalizado [0,1] y rango normalizado [0,1]
        fam_lo, fam_hi = FAMILY_RANGES[fam]
        pitch_span = max(fam_hi - fam_lo, 1)
        pitch_mid  = (pitches.mean() - fam_lo) / pitch_span
        pitch_rng  = (pitches.max() - pitches.min()) / pitch_span
        tensor[f, base + IDX_REG_MID] = float(np.clip(pitch_mid, 0, 1))
        tensor[f, base + IDX_REG_RNG] = float(np.clip(pitch_rng, 0, 1))

        # Densidad: notas activas / máximo razonable (8 simultáneas)
        tensor[f, base + IDX_DENSITY] = float(np.clip(n / 8.0, 0, 1))

        # Dinámica: velocidad media normalizada
        tensor[f, base + IDX_DYNAMIC] = float(velocities.mean() / 127.0)

    # Onsets
    for (f, fam), count in frame_fam_onsets.items():
        fam_idx = FAMILIES.index(fam)
        base    = fam_idx * DIMS_PER_FAMILY
        tensor[f, base + IDX_ONSET] = float(np.clip(count / 4.0, 0, 1))

    return tensor


# ══════════════════════════════════════════════════════════════════════════════
#  TENSOR → MIDI
# ══════════════════════════════════════════════════════════════════════════════

def tensor_to_midi(tensor: np.ndarray, bpm: float,
                   instruments: list[dict] | None = None,
                   frame_ms: int = FRAME_MS,
                   output_path: str = 'reconstructed.mid') -> MidiFile:
    """
    Convierte tensor [T, 68] en MidiFile multi-pista.

    instruments: lista de dicts del orchestrator.py:
        [{'name': 'violin1', 'role': 'melody'}, {'name': 'cello', 'role': 'bass'}, ...]
        Si es None, genera una pista genérica por familia activa.

    La conversión es determinista: del chroma + registro + densidad
    se infieren las notas más probables respetando INSTRUMENT_RANGES.
    """
    mid = MidiFile(type=1, ticks_per_beat=TICKS_OUT)
    tempo_us = int(60_000_000 / max(bpm, 20))
    frame_ticks = int(TICKS_OUT * bpm * frame_ms / 60_000)

    # Track de tempo
    tempo_track = MidiTrack()
    tempo_track.append(MetaMessage('set_tempo', tempo=tempo_us, time=0))
    mid.tracks.append(tempo_track)

    # Determinar qué instrumentos generar por familia
    if instruments is None:
        # Un instrumento representativo por familia
        default_instrs = {
            'keys':       {'name': 'piano',     'role': 'accompaniment', 'program': 0},
            'strings':    {'name': 'violin1',   'role': 'melody',        'program': 40},
            'bass':       {'name': 'bass_guitar','role': 'bass',          'program': 33},
            'winds':      {'name': 'trumpet',   'role': 'melody',        'program': 56},
            'percussion': {'name': 'drums',     'role': 'percussion',    'program': 0},
        }
        instr_by_family = {
            fam: [cfg] for fam, cfg in default_instrs.items()
        }
    else:
        instr_by_family = defaultdict(list)
        for instr in instruments:
            name = instr.get('name', '')
            fam  = INSTR_TO_FAMILY.get(name, 'strings')
            instr_by_family[fam].append(instr)

    # Programa GM por instrumento
    INSTR_PROGRAM = {
        'piano': 0, 'electric_piano': 4, 'organ': 19, 'harpsichord': 6,
        'vibraphone': 11, 'celesta': 8,
        'violin1': 40, 'violin2': 40, 'viola': 41, 'cello': 42, 'contrabass': 43,
        'string_ensemble': 48, 'choir': 52,
        'guitar': 25, 'nylon_guitar': 24, 'steel_guitar': 25,
        'bass_guitar': 33, 'electric_bass': 33, 'acoustic_bass': 32,
        'fretless_bass': 35, 'synth_bass': 38,
        'flute': 73, 'oboe': 68, 'clarinet': 71, 'bassoon': 70,
        'horn': 60, 'trumpet': 56, 'trombone': 57, 'tuba': 58,
        'alto_sax': 65, 'tenor_sax': 66, 'soprano_sax': 64,
        'baritone_sax': 67, 'brass_section': 61,
        'timpani': 47, 'drums': 0, 'drum_kit': 0,
    }

    n_frames = tensor.shape[0]

    for fam_idx, fam in enumerate(FAMILIES):
        instrs = instr_by_family.get(fam, [])
        if not instrs:
            continue

        base = fam_idx * DIMS_PER_FAMILY

        for ch_offset, instr in enumerate(instrs):
            name    = instr.get('name', fam)
            role    = instr.get('role', 'melody')
            program = INSTR_PROGRAM.get(name, 40)
            channel = fam_idx * 4 + ch_offset   # canal MIDI único
            if channel == 9:
                channel = 10   # evitar canal de percusión GM salvo para perc

            if fam == 'percussion':
                channel = 9
                program = None

            # Rango del instrumento
            lo, hi = INSTRUMENT_RANGES.get(name, FAMILY_RANGES.get(fam, (36, 84)))

            track = MidiTrack()
            track.append(MetaMessage('track_name', name=name, time=0))
            if program is not None:
                track.append(Message('program_change', channel=channel,
                                     program=program, time=0))

            prev_tick  = 0
            active_notes: list[tuple[int,int]] = []  # (pitch, tick_off)

            for f in range(n_frames):
                frame_start_tick = f * frame_ticks

                # Cerrar notas que han terminado
                still_active = []
                for pitch, tick_off in active_notes:
                    if tick_off <= frame_start_tick:
                        dt = tick_off - prev_tick
                        track.append(Message('note_off', channel=channel,
                                             note=pitch, velocity=0, time=max(0,dt)))
                        prev_tick = tick_off
                    else:
                        still_active.append((pitch, tick_off))
                active_notes = still_active

                onset = tensor[f, base + IDX_ONSET]
                if onset < 0.05:
                    continue   # sin onset en este frame

                density  = tensor[f, base + IDX_DENSITY]
                dynamic  = tensor[f, base + IDX_DYNAMIC]
                reg_mid  = tensor[f, base + IDX_REG_MID]
                chroma   = tensor[f, base + IDX_CHROMA_START : base + IDX_CHROMA_END]

                # Cuántas notas generar
                n_notes = max(1, int(round(density * 4)))

                # Registro objetivo
                fam_lo, fam_hi = FAMILY_RANGES[fam]
                target_pitch = int(fam_lo + reg_mid * (fam_hi - fam_lo))
                target_pitch = int(np.clip(target_pitch, lo, hi))

                # Seleccionar pitch classes del chroma
                chroma_norm = chroma / (chroma.sum() + 1e-8)
                selected_pcs = _sample_pitch_classes(chroma_norm, n_notes, role)

                # Construir notas en el registro objetivo
                notes_to_play = _pcs_to_pitches(selected_pcs, target_pitch, lo, hi)

                velocity = int(np.clip(dynamic * 127, 20, 120))
                note_dur_ticks = max(frame_ticks // 2,
                                     int(frame_ticks * (0.4 + density * 0.5)))

                dt = frame_start_tick - prev_tick
                for i, pitch in enumerate(notes_to_play):
                    track.append(Message('note_on', channel=channel,
                                         note=pitch, velocity=velocity,
                                         time=dt if i == 0 else 0))
                    active_notes.append((pitch, frame_start_tick + note_dur_ticks))
                    prev_tick = frame_start_tick

            # Cerrar notas restantes al final
            last_tick = n_frames * frame_ticks
            for pitch, tick_off in active_notes:
                dt = max(0, tick_off - prev_tick)
                track.append(Message('note_off', channel=channel,
                                     note=pitch, velocity=0, time=dt))
                prev_tick = tick_off
            track.append(MetaMessage('end_of_track', time=0))
            mid.tracks.append(track)

    mid.save(output_path)
    return mid


def _sample_pitch_classes(chroma_norm: np.ndarray, n: int, role: str) -> list[int]:
    """
    Muestrea n pitch classes ponderados por chroma.
    Para 'melody': toma el PC más saliente.
    Para 'bass': pesos hacia PCs bajos del chroma.
    Para 'accompaniment'/'pad': distribución más amplia.
    """
    if chroma_norm.sum() < 1e-8:
        # Sin información: usar escala mayor de Do
        scale = [0, 2, 4, 5, 7, 9, 11]
        return [scale[i % len(scale)] for i in range(n)]

    if role == 'melody':
        # Tomar los n PCs más probables, sin repetir
        pcs = np.argsort(chroma_norm)[::-1][:n].tolist()
    elif role == 'bass':
        # PC más fuerte para bajo + opcionalmente la quinta
        top_pc = int(np.argmax(chroma_norm))
        pcs = [top_pc, (top_pc + 7) % 12][:n]
    else:
        # Muestreo estocástico pero determinista (seed por frame)
        rng = np.random.default_rng(int(chroma_norm.sum() * 1e6) % 2**32)
        pcs = rng.choice(12, size=n, p=chroma_norm, replace=True).tolist()
        pcs = list(dict.fromkeys(pcs))[:n]   # deduplicar manteniendo orden

    return pcs


def _pcs_to_pitches(pcs: list[int], target_pitch: int,
                    lo: int, hi: int) -> list[int]:
    """
    Convierte pitch classes en notas MIDI dentro del rango [lo, hi],
    centradas alrededor de target_pitch.
    """
    pitches = []
    for pc in pcs:
        # Octava más cercana a target_pitch
        base = (target_pitch // 12) * 12 + pc
        candidates = [base - 12, base, base + 12]
        best = min(candidates, key=lambda p: (abs(p - target_pitch), p))
        best = int(np.clip(best, lo, hi))
        if best not in pitches:
            pitches.append(best)
    return pitches


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA: funciones de conveniencia
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_tensor(file_path: str,
                   frame_ms: int = FRAME_MS,
                   max_frames: int = 256) -> tuple[np.ndarray, float] | tuple[None, None]:
    """
    Función principal de encoding. Convierte un fichero MIDI en tensor.

    Retorna:
        (tensor [T, 68], bpm)  donde T <= max_frames
        (None, None)           si el fichero es inválido o vacío

    El tensor se normaliza a max_frames truncando o con padding de ceros.
    Esto garantiza tensores de tamaño uniforme para el modelo.
    """
    events, total_sec, bpm = midi_to_note_events(file_path)
    if not events:
        return None, None

    tensor = events_to_tensor(events, total_sec, frame_ms)

    # Normalizar longitud
    T = tensor.shape[0]
    if T >= max_frames:
        tensor = tensor[:max_frames]
    else:
        pad = np.zeros((max_frames - T, TENSOR_DIMS), dtype=np.float32)
        tensor = np.concatenate([tensor, pad], axis=0)

    return tensor, bpm


def tensor_to_midi_file(tensor: np.ndarray, bpm: float,
                        instruments: list[dict] | None = None,
                        output_path: str = 'output.mid',
                        frame_ms: int = FRAME_MS) -> str:
    """
    Función principal de decoding. Genera MIDI desde tensor.

    instruments: lista de dicts del orchestrator compatible:
        [{'name': 'violin1', 'role': 'melody'}, ...]

    Retorna la ruta del fichero generado.
    """
    # Recortar padding (frames de ceros al final)
    mask = tensor.any(axis=1)
    if mask.any():
        last = int(np.where(mask)[0][-1]) + 1
        tensor = tensor[:last]

    tensor_to_midi(tensor, bpm, instruments, frame_ms, output_path)
    return output_path


def get_tensor_stats(tensor: np.ndarray) -> dict:
    """
    Estadísticas descriptivas del tensor para diagnóstico y logging.
    Útil para entender qué información se capturó antes de entrenar.
    """
    stats = {
        'n_frames':    tensor.shape[0],
        'total_sec':   tensor.shape[0] * FRAME_MS / 1000,
        'families':    {},
        'global': {
            'activity':    float(tensor.any(axis=1).mean()),  # % frames con contenido
            'mean':        float(tensor.mean()),
            'std':         float(tensor.std()),
            'nonzero_pct': float((tensor > 0).mean()),
        }
    }

    for fam_idx, fam in enumerate(FAMILIES):
        base = fam_idx * DIMS_PER_FAMILY
        block = tensor[:, base:base + DIMS_PER_FAMILY]

        chroma = block[:, IDX_CHROMA_START:IDX_CHROMA_END]          # [T, 12]
        active_frames = block[:, IDX_DENSITY] > 0.01

        # PC más frecuente
        chroma_sum = chroma.sum(axis=0)
        top_pc = int(np.argmax(chroma_sum)) if chroma_sum.sum() > 0 else -1

        NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        stats['families'][fam] = {
            'active_pct':  float(active_frames.mean()),
            'mean_density':float(block[:, IDX_DENSITY].mean()),
            'mean_dynamic':float(block[:, IDX_DYNAMIC].mean()),
            'mean_reg_mid':float(block[:, IDX_REG_MID].mean()),
            'top_pc':      NOTE_NAMES[top_pc] if top_pc >= 0 else 'N/A',
            'onset_rate':  float(block[:, IDX_ONSET].mean()),
        }

    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH CHECK: validar corpus completo
# ══════════════════════════════════════════════════════════════════════════════

def batch_check(midi_files: list[str], frame_ms: int = FRAME_MS,
                max_frames: int = 256) -> dict:
    """
    Procesa una lista de MIDIs e informa estadísticas del corpus.
    Útil para validar que los datos de entrenamiento son correctos.
    """
    ok, failed = [], []
    all_bpms, all_active = [], []
    family_coverage = defaultdict(int)

    total = len(midi_files)
    for i, f in enumerate(midi_files):
        tensor, bpm = midi_to_tensor(f, frame_ms, max_frames)
        if tensor is None:
            failed.append(f)
            continue
        ok.append(f)
        all_bpms.append(bpm)
        all_active.append(float(tensor.any(axis=1).mean()))

        for fam_idx, fam in enumerate(FAMILIES):
            base = fam_idx * DIMS_PER_FAMILY
            if tensor[:, base + IDX_DENSITY].max() > 0.01:
                family_coverage[fam] += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Procesados: {i+1}/{total}  OK: {len(ok)}  Fallidos: {len(failed)}",
                  end='\r')

    print()
    return {
        'total':           total,
        'valid':           len(ok),
        'failed':          len(failed),
        'failed_files':    failed[:20],   # primeros 20
        'bpm_mean':        float(np.mean(all_bpms)) if all_bpms else 0,
        'bpm_std':         float(np.std(all_bpms))  if all_bpms else 0,
        'activity_mean':   float(np.mean(all_active)) if all_active else 0,
        'family_coverage': dict(family_coverage),
    }



# ══════════════════════════════════════════════════════════════════════════════
#  SURVEY DE PROGRAMS: diagnostico de deteccion de familias
# ══════════════════════════════════════════════════════════════════════════════

def survey_programs(midi_files: list[str]) -> dict:
    """
    Recorre los MIDIs y devuelve estadisticas de program numbers usados,
    numero de pistas por fichero, y canales activos.
    Util para calibrar la deteccion de familia antes de entrenar.
    """
    from collections import Counter

    prog_counter   = Counter()
    tracks_counts  = []
    channel_counts = Counter()

    GM_NAMES = {
        0:'Acoustic Grand Piano', 1:'Bright Acoustic Piano',
        2:'Electric Grand Piano', 4:'Electric Piano 1', 5:'Electric Piano 2',
        6:'Harpsichord', 7:'Clavinet', 8:'Celesta', 11:'Vibraphone',
        13:'Xylophone', 14:'Tubular Bells', 16:'Drawbar Organ',
        19:'Church Organ', 24:'Nylon Guitar', 25:'Steel Guitar',
        26:'Jazz Guitar', 29:'Overdriven Guitar',
        32:'Acoustic Bass', 33:'Electric Bass(finger)', 34:'Electric Bass(pick)',
        35:'Fretless Bass', 38:'Synth Bass 1',
        40:'Violin', 41:'Viola', 42:'Cello', 43:'Contrabass',
        44:'Tremolo Strings', 45:'Pizzicato Strings', 46:'Orchestral Harp',
        48:'String Ensemble 1', 49:'String Ensemble 2',
        50:'Synth Strings 1', 51:'Synth Strings 2',
        52:'Choir Aahs', 54:'Synth Voice', 55:'Orchestra Hit',
        56:'Trumpet', 57:'Trombone', 58:'Tuba', 59:'Muted Trumpet',
        60:'French Horn', 61:'Brass Section', 62:'Synth Brass 1',
        64:'Soprano Sax', 65:'Alto Sax', 66:'Tenor Sax', 67:'Baritone Sax',
        68:'Oboe', 69:'English Horn', 70:'Bassoon', 71:'Clarinet',
        72:'Piccolo', 73:'Flute', 74:'Recorder', 75:'Pan Flute',
        104:'Sitar', 105:'Banjo', 107:'Koto',
    }

    total = len(midi_files)
    for i, f in enumerate(midi_files):
        try:
            mid = mido.MidiFile(f)
        except Exception:
            continue

        progs_in_file = set()
        tracks_counts.append(len(mid.tracks))

        for track in mid.tracks:
            for msg in track:
                if msg.type == 'program_change':
                    progs_in_file.add(msg.program)
                    if hasattr(msg, 'channel'):
                        channel_counts[msg.channel] += 1
                elif msg.type == 'note_on' and hasattr(msg, 'channel'):
                    channel_counts[msg.channel] += 1

        for p in progs_in_file:
            prog_counter[p] += 1

        if (i+1) % 100 == 0 or (i+1) == total:
            print(f"  Analizando programs: {i+1}/{total}", end='\r')

    print()

    return {
        'prog_counter':  prog_counter,
        'gm_names':      GM_NAMES,
        'tracks_mean':   float(np.mean(tracks_counts)) if tracks_counts else 0,
        'tracks_max':    int(np.max(tracks_counts))    if tracks_counts else 0,
        'tracks_hist':   dict(Counter(tracks_counts)),
        'channel_counts':dict(channel_counts.most_common(16)),
        'has_perc_ch9':  channel_counts.get(9, 0),
    }

# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def plot_tensor(tensor: np.ndarray, title: str = '', bpm: float = 120.0):
    """
    Visualiza el tensor [T, 68] como heatmap por familia.
    Requiere matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("pip install matplotlib para visualización.")
        return

    n_frames = tensor.shape[0]
    time_axis = np.arange(n_frames) * FRAME_MS / 1000.0   # segundos

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(title or 'Tensor de representación MIDI', fontsize=14)

    gs = gridspec.GridSpec(N_FAMILIES + 1, 2,
                           height_ratios=[3]*N_FAMILIES + [2],
                           hspace=0.45, wspace=0.3)

    colors = {'strings': '#4e79a7', 'woodwinds': '#f28e2b',
               'brass': '#e15759', 'percussion': '#76b7b2'}

    for fam_idx, fam in enumerate(FAMILIES):
        base  = fam_idx * DIMS_PER_FAMILY
        block = tensor[:, base:base + DIMS_PER_FAMILY]

        # Heatmap del chroma
        ax_chroma = fig.add_subplot(gs[fam_idx, 0])
        chroma_data = block[:, IDX_CHROMA_START:IDX_CHROMA_END].T   # [12, T]
        NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        im = ax_chroma.imshow(chroma_data, aspect='auto', origin='lower',
                               cmap='Blues', vmin=0, vmax=1,
                               extent=[0, time_axis[-1], 0, 12])
        ax_chroma.set_yticks(range(12))
        ax_chroma.set_yticklabels(NOTE_NAMES, fontsize=7)
        ax_chroma.set_title(f'{fam.capitalize()} — Chroma', fontsize=9)
        ax_chroma.set_xlabel('Tiempo (s)', fontsize=8)
        plt.colorbar(im, ax=ax_chroma, fraction=0.03)

        # Curvas de densidad / dinámica / onset
        ax_curves = fig.add_subplot(gs[fam_idx, 1])
        c = colors[fam]
        ax_curves.fill_between(time_axis, block[:, IDX_DENSITY],
                                alpha=0.4, color=c, label='Densidad')
        ax_curves.plot(time_axis, block[:, IDX_DYNAMIC],
                       color=c, lw=1.2, label='Dinámica')
        ax_curves.plot(time_axis, block[:, IDX_ONSET],
                       color='gray', lw=0.8, alpha=0.6, label='Onset')
        ax_curves.plot(time_axis, block[:, IDX_REG_MID],
                       color='black', lw=0.8, ls='--', alpha=0.5, label='Registro')
        ax_curves.set_ylim(0, 1)
        ax_curves.set_title(f'{fam.capitalize()} — Curvas', fontsize=9)
        ax_curves.set_xlabel('Tiempo (s)', fontsize=8)
        ax_curves.legend(fontsize=7, loc='upper right')

    # Actividad global (heatmap completo)
    ax_global = fig.add_subplot(gs[N_FAMILIES, :])
    im2 = ax_global.imshow(tensor.T, aspect='auto', origin='lower',
                            cmap='viridis', vmin=0, vmax=1,
                            extent=[0, time_axis[-1], 0, TENSOR_DIMS])
    ax_global.set_title('Tensor completo [68 dims]', fontsize=9)
    ax_global.set_xlabel('Tiempo (s)', fontsize=8)
    ax_global.set_ylabel('Dim', fontsize=8)
    plt.colorbar(im2, ax=ax_global, fraction=0.01)

    # Líneas separadoras de familia
    for fam_idx in range(1, N_FAMILIES):
        ax_global.axhline(fam_idx * DIMS_PER_FAMILY, color='white',
                          lw=0.8, ls='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / TEST
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='LM_REPR: test y visualización de la representación tensor MIDI')
    parser.add_argument('midi', nargs='+', help='Fichero(s) MIDI a procesar')
    parser.add_argument('--plot',        action='store_true',
                        help='Visualizar tensor con matplotlib')
    parser.add_argument('--roundtrip',   action='store_true',
                        help='Test de reconstrucción: MIDI → tensor → MIDI')
    parser.add_argument('--info',        action='store_true',
                        help='Mostrar estadísticas detalladas del tensor')
    parser.add_argument('--batch-check', action='store_true',
                        help='Validar corpus completo (para multiples MIDIs)')
    parser.add_argument('--survey-programs', action='store_true',
                        help='Analizar program numbers del corpus para calibrar deteccion de familia')
    parser.add_argument('--frame-ms',    type=int, default=FRAME_MS,
                        help=f'Duración de frame en ms (default: {FRAME_MS})')
    parser.add_argument('--max-frames',  type=int, default=256,
                        help='Longitud máxima del tensor (default: 256)')
    parser.add_argument('--output',      default='reconstructed.mid',
                        help='Fichero de salida para --roundtrip')
    args = parser.parse_args()

    # ── Batch check ──────────────────────────────────────────────────────────
    if args.batch_check:
        print(f"\nValidando corpus: {len(args.midi)} ficheros...")
        results = batch_check(args.midi, args.frame_ms, args.max_frames)
        print("\n═══ RESUMEN DEL CORPUS ═══")
        print(f"  Total:         {results['total']}")
        print(f"  Válidos:       {results['valid']}  "
              f"({100*results['valid']/max(results['total'],1):.1f}%)")
        print(f"  Fallidos:      {results['failed']}")
        print(f"  BPM medio:     {results['bpm_mean']:.1f} ± {results['bpm_std']:.1f}")
        print(f"  Actividad med: {results['activity_mean']:.3f}")
        print(f"  Cobertura por familia:")
        for fam, count in results['family_coverage'].items():
            pct = 100 * count / max(results['valid'], 1)
            print(f"    {fam:<12}: {count:>5}  ({pct:.1f}%)")
        if results['failed_files']:
            print(f"\n  Primeros fallidos:")
            for f in results['failed_files'][:10]:
                print(f"    {f}")
        return

    # ── Survey de programs ──────────────────────────────────────────────────
    if args.survey_programs:
        print(f"\nAnalizando programs en {len(args.midi)} ficheros...")
        res = survey_programs(args.midi)
        print("\n══ PROGRAMS MAS FRECUENTES (top 30) ══")
        print(f"  {'Program':>7}  {'Nombre GM':<28}  {'MIDIs':>6}  {'%':>5}")
        print(f"  {'-'*7}  {'-'*28}  {'-'*6}  {'-'*5}")
        total_valid = sum(res['prog_counter'].values()) or 1
        for prog, count in res['prog_counter'].most_common(30):
            name = res['gm_names'].get(prog, f'Program {prog}')
            pct  = 100 * count / len(args.midi)
            fam  = 'strings'
            if prog in range(68,80): fam = 'woodwinds'
            elif prog in range(56,68): fam = 'brass'
            elif prog in range(0,16): fam = 'keys'
            elif prog in range(24,40): fam = 'guitar/bass'
            print(f"  {prog:>7}  {name:<28}  {count:>6}  {pct:>5.1f}%  [{fam}]")
        print(f"\n══ PISTAS POR FICHERO ══")
        print(f"  Media: {res['tracks_mean']:.1f}   Max: {res['tracks_max']}")
        hist = sorted(res['tracks_hist'].items())
        for n_tracks, count in hist[:15]:
            bar = '█' * min(count // 5 + 1, 40)
            print(f"  {n_tracks:>2} pistas: {count:>4}  {bar}")
        print(f"\n══ CANALES ACTIVOS ══")
        for ch, count in sorted(res['channel_counts'].items())[:16]:
            marker = '  ← PERCUSION GM' if ch == 9 else ''
            print(f"  Canal {ch:>2}: {count:>6} eventos{marker}")
        print(f"\n  Ficheros con canal 9 (percusion GM): {res['has_perc_ch9']}")
        print(f"\n  CONSEJO: si ves programas 0-39 dominando, considera ampliar")
        print(f"  FAMILY_GM['strings'] para incluir ensembles (48,49) y arpa (46).")
        return

    # ── Procesamiento individual ──────────────────────────────────────────────
    midi_file = args.midi[0]
    if not os.path.isfile(midi_file):
        print(f"ERROR: no se encuentra {midi_file}")
        sys.exit(1)

    print(f"\nProcesando: {midi_file}")
    tensor, bpm = midi_to_tensor(midi_file, args.frame_ms, args.max_frames)

    if tensor is None:
        print("ERROR: no se pudo extraer información del MIDI.")
        sys.exit(1)

    T, D = tensor.shape
    print(f"  Tensor:    [{T}, {D}]")
    print(f"  Duración:  {T * args.frame_ms / 1000:.1f}s  ({T} frames × {args.frame_ms}ms)")
    print(f"  BPM:       {bpm:.1f}")
    print(f"  Actividad: {tensor.any(axis=1).mean()*100:.1f}% frames con contenido")
    print(f"  Memoria:   {tensor.nbytes / 1024:.1f} KB")

    if args.info:
        stats = get_tensor_stats(tensor)
        print("\n═══ ESTADÍSTICAS ═══")
        print(f"  Actividad global:  {stats['global']['activity']*100:.1f}%")
        print(f"  No-cero global:    {stats['global']['nonzero_pct']*100:.1f}%")
        print(f"\n  Por familia:")
        for fam, s in stats['families'].items():
            print(f"  ┌─ {fam}")
            print(f"  │  Activo:    {s['active_pct']*100:5.1f}%   "
                  f"PC dominante: {s['top_pc']}")
            print(f"  │  Densidad:  {s['mean_density']:.3f}   "
                  f"Dinámica: {s['mean_dynamic']:.3f}")
            print(f"  └  Registro:  {s['mean_reg_mid']:.3f}   "
                  f"Onsets:   {s['onset_rate']:.3f}")

    if args.roundtrip:
        out = args.output
        print(f"\nRoundtrip → {out}")
        tensor_to_midi_file(tensor, bpm, output_path=out,
                            frame_ms=args.frame_ms)
        print(f"  Guardado: {out}")
        print(f"  (Reproduce con un reproductor MIDI para evaluar la fidelidad)")

    if args.plot:
        title = f"{os.path.basename(midi_file)}  [{T}×{D}]  {bpm:.0f} BPM"
        plot_tensor(tensor, title=title, bpm=bpm)


if __name__ == '__main__':
    main()
