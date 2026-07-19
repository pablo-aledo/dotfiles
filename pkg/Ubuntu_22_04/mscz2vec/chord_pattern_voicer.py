#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CHORD PATTERN VOICER  v1.0                             ║
║      Reescribe el patrón rítmico de la pista de armonía de un MIDI          ║
║                                                                              ║
║  Toma un MIDI con varias pistas, detecta automáticamente cuál es la         ║
║  pista de armonía (la más polifónica, por acordes) y reescribe SOLO esa     ║
║  pista alterando la forma de "tocar" cada acorde: en vez de bloques         ║
║  simultáneos, distribuye las mismas notas en el tiempo según un patrón      ║
║  configurable (bombo-caja, bajo-alberti, arpegios, bloque, o uno propio).   ║
║  El resto de pistas (melodía, bajo, batería…) se conservan intactas.        ║
║                                                                              ║
║  PATRONES DISPONIBLES (--pattern):                                          ║
║    block            — todas las notas del acorde a la vez (sin alterar)     ║
║    bombo-caja        — 1º la nota más grave (tónica/bajo), 2º el resto      ║
║                        del acorde junto (imita bombo→caja)                  ║
║    alberti           — bajo, agudo, medio, agudo (bajo-alberti clásico)     ║
║    arpeggio-up        — arpegio ascendente, nota a nota                     ║
║    arpeggio-down       — arpegio descendente, nota a nota                    ║
║    arpeggio-updown     — arpegio ascendente y luego descendente             ║
║    custom            — patrón definido a mano con --custom-pattern         ║
║                                                                              ║
║  ROLES para --custom-pattern (lista separada por comas):                    ║
║    bajo/bass/root    nota más grave del acorde                              ║
║    agudo/top/alto    nota más aguda del acorde                              ║
║    medio/mid         nota central del acorde                                ║
║    resto/rest/caja   todas las notas salvo la más grave, juntas             ║
║    todas/all/block   todas las notas del acorde a la vez                    ║
║    0, 1, 2, -1…      índice de nota ordenada de grave a aguda (soporta -1)  ║
║    [a,b,…]           combina varios roles/índices en un solo paso          ║
║                        SIMULTÁNEO, p.ej. "[0,1]" o "[bajo,medio]"           ║
║                                                                              ║
║  VELOCIDAD DE REPETICIÓN (--reps):                                          ║
║    Nº de veces que el patrón completo se repite dentro de la duración de    ║
║    cada acorde detectado (típicamente un compás). --reps 1 = una vez por    ║
║    compás; --reps 2 = dos veces por compás (el doble de rápido).           ║
║                                                                              ║
║  PATRONES POR RANGO DE COMPASES (--segments):                               ║
║    Permite usar un patrón/velocidad distintos en distintas zonas de la      ║
║    pieza. Cada segmento se pasa como un argumento independiente con el      ║
║    formato:                                                                  ║
║        INICIO-FIN:pattern=NOMBRE[,reps=N][,gate=F][,accent=F][,custom=…]   ║
║    · INICIO y FIN son compases 1-based; FIN es EXCLUSIVO, así que rangos    ║
║      consecutivos como "1-10" y "10-20" no se solapan ni dejan huecos.      ║
║    · Los campos reps/gate/accent/octave son opcionales: si se omiten se     ║
║      usan los valores globales (--reps/--gate/--accent/--octave).           ║
║    · Para pattern=custom, los roles van en "custom=" separados por "|"      ║
║      (no por comas, que ya se usan para separar los campos): p.ej.          ║
║      custom=bajo|agudo|resto|medio                                          ║
║    · Los compases no cubiertos por ningún segmento usan --pattern/--reps/   ║
║      --gate/--accent globales como valores por defecto.                     ║
║                                                                              ║
║  USO:                                                                        ║
║    python chord_pattern_voicer.py cancion.mid                               ║
║    python chord_pattern_voicer.py cancion.mid --pattern bombo-caja          ║
║    python chord_pattern_voicer.py cancion.mid --pattern alberti --reps 2    ║
║    python chord_pattern_voicer.py cancion.mid --pattern arpeggio-up --reps 2 --gate 0.6 ║
║    python chord_pattern_voicer.py cancion.mid --pattern custom              ║
║        --custom-pattern "bajo,agudo,resto,medio"                            ║
║    python chord_pattern_voicer.py cancion.mid --pattern custom              ║
║        --custom-pattern "[0,1],1,2,0"   (bajo+medio a la vez, luego medio,  ║
║                                           agudo, bajo)                       ║
║    python chord_pattern_voicer.py cancion.mid --track 2 --verbose           ║
║    python chord_pattern_voicer.py cancion.mid --list-patterns               ║
║                                                                              ║
║    # Patrón distinto por tramos de la pieza (el ejemplo del usuario):       ║
║    python chord_pattern_voicer.py cancion.mid --segments                    ║
║        "1-10:pattern=bombo-caja,reps=1"                                     ║
║        "10-20:pattern=alberti,reps=2"                                       ║
║        "20-30:pattern=custom,reps=2,custom=bajo|agudo|resto|medio"          ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --track N          Fuerza el índice (0-based) de la pista de armonía.    ║
║                        Por defecto se autodetecta la pista más polifónica.  ║
║    --pattern P         Patrón rítmico a aplicar por defecto (bombo-caja)     ║
║    --custom-pattern S  Roles separados por comas (con --pattern custom).    ║
║                        Usá "[a,b,…]" para tocar varios roles/índices a la  ║
║                        vez en un mismo paso, p.ej. "[0,1],1,2,0"           ║
║    --reps N            Repeticiones del patrón por acorde (default: 1)      ║
║    --gate F            Fracción sonante de cada paso, 0-1 (default: 0.85)   ║
║                        F bajo = más staccato/separado; F=1 = legato total   ║
║    --accent F          Multiplica la velocity del primer golpe de cada      ║
║                        repetición (default: 1.0 = sin acento)               ║
║    --octave N          Transporta TODAS las notas del acorde N octavas      ║
║                        (default: 0). Positivo = sube, negativo = baja,      ║
║                        p.ej. --octave -1 baja una octava, --octave 2 sube   ║
║                        dos. Las notas resultantes se recortan al rango      ║
║                        MIDI válido (0-127). También se puede fijar por      ║
║                        tramo con octave=N dentro de --segments.             ║
║    --segments S [S…]  Patrón/velocidad distintos por rango de compases      ║
║                        (ver "PATRONES POR RANGO DE COMPASES" arriba)        ║
║    --min-notes N       Nº mínimo de notas simultáneas para considerar una   ║
║                        pista polifónica en la autodetección (default: 2)    ║
║    --out FILE          Ruta del MIDI de salida                              ║
║    --out-dir DIR       Carpeta de salida (default: junto al MIDI de entrada)║
║    --list-patterns     Lista los patrones disponibles y sale                ║
║    --report            Guarda un JSON con el análisis completo              ║
║    --verbose           Informe detallado por stdout                         ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    <base>.voiced.mid          MIDI con la pista de armonía reescrita       ║
║    <base>.voiced_report.json  con --report                                 ║
║                                                                              ║
║  DEPENDENCIAS: mido                                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import argparse
import traceback
from collections import defaultdict

import mido


# ═══════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DRUM_CHANNEL = 9  # canal MIDI 10 (0-based) reservado convencionalmente a batería

NAMED_PATTERNS = {
    'block':            ['todas'],
    'bombo-caja':        ['bajo', 'resto'],
    'alberti':           ['bajo', 'agudo', 'medio', 'agudo'],
    # arpeggio-*: se generan dinámicamente según el nº de notas de cada acorde
    'arpeggio-up':        None,
    'arpeggio-down':      None,
    'arpeggio-updown':    None,
    'custom':            None,
}

ROLE_ALIASES = {
    'bajo': 'bass', 'root': 'bass', 'raiz': 'bass', 'bass': 'bass',
    'agudo': 'top', 'alto': 'top', 'top': 'top',
    'medio': 'mid', 'mid': 'mid', 'middle': 'mid',
    'resto': 'rest', 'caja': 'rest', 'rest': 'rest',
    'todas': 'all', 'all': 'all', 'block': 'all',
}


def pitch_name(p):
    return f"{PITCH_NAMES[p % 12]}{p // 12 - 1}"


# ═══════════════════════════════════════════════════════════
# COMPASES Y SEGMENTOS (--segments)
# ═══════════════════════════════════════════════════════════

def detect_time_signature(mid):
    """Busca el primer mensaje time_signature en cualquier pista. Default 4/4."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                return msg.numerator, msg.denominator
    return 4, 4


def ticks_per_bar(tpb, numerator, denominator):
    return max(1, int(round(tpb * numerator * 4 / denominator)))


def bar_of_tick(tick, tpb_bar):
    """Compás 1-based que contiene el tick dado."""
    return tick // tpb_bar + 1


def parse_segment_spec(spec):
    """
    Parsea un segmento con formato:
      INICIO-FIN:pattern=NOMBRE[,reps=N][,gate=F][,accent=F][,custom=a|b|c]
    FIN es exclusivo. Devuelve un dict.
    """
    if ':' not in spec:
        raise ValueError(f"Segmento mal formado (falta ':'): «{spec}»")
    bar_part, field_part = spec.split(':', 1)

    if '-' not in bar_part:
        raise ValueError(f"Segmento mal formado (rango de compases sin '-'): «{spec}»")
    start_s, end_s = bar_part.split('-', 1)
    try:
        bar_start, bar_end = int(start_s), int(end_s)
    except ValueError:
        raise ValueError(f"Rango de compases no numérico en: «{spec}»")
    if bar_start < 1 or bar_end <= bar_start:
        raise ValueError(f"Rango de compases inválido (FIN debe ser > INICIO >= 1): «{spec}»")

    fields = {}
    for chunk in field_part.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '=' not in chunk:
            raise ValueError(f"Campo sin '=' en segmento «{spec}»: «{chunk}»")
        k, v = chunk.split('=', 1)
        fields[k.strip().lower()] = v.strip()

    if 'pattern' not in fields:
        raise ValueError(f"Segmento «{spec}» no especifica pattern=…")
    pattern = fields['pattern']
    if pattern not in NAMED_PATTERNS:
        raise ValueError(f"Patrón desconocido «{pattern}» en segmento «{spec}». "
                          f"Opciones: {', '.join(NAMED_PATTERNS.keys())}")

    custom_pattern = None
    if pattern == 'custom':
        if 'custom' not in fields:
            raise ValueError(f"Segmento «{spec}» usa pattern=custom pero no define custom=roles "
                              f"(separados por '|')")
        custom_pattern = fields['custom'].replace('|', ',')

    seg = {
        'bar_start': bar_start,
        'bar_end': bar_end,
        'pattern': pattern,
        'custom_pattern': custom_pattern,
        'reps': int(fields['reps']) if 'reps' in fields else None,
        'gate': float(fields['gate']) if 'gate' in fields else None,
        'accent': float(fields['accent']) if 'accent' in fields else None,
        'octave': int(fields['octave']) if 'octave' in fields else None,
        'raw': spec,
    }
    return seg


def parse_segments(spec_list):
    segments = [parse_segment_spec(s) for s in spec_list]
    segments.sort(key=lambda s: s['bar_start'])
    # aviso de solapes (no bloqueante: se aplica el primero en orden de aparición)
    for i in range(len(segments) - 1):
        a, b = segments[i], segments[i + 1]
        if a['bar_end'] > b['bar_start']:
            print(f"[AVISO] Segmentos solapados: «{a['raw']}» y «{b['raw']}» "
                  f"comparten compases {b['bar_start']}-{a['bar_end']-1}. "
                  f"Se usará el primero definido para esos compases.")
    return segments


def resolve_segment_for_bar(segments, bar_num, defaults):
    """
    Devuelve la config efectiva (pattern, custom_pattern, reps, gate, accent) para
    el compás `bar_num`: el primer segmento cuyo rango lo cubra, o los valores
    globales por defecto si ningún segmento lo cubre.
    """
    for seg in segments:
        if seg['bar_start'] <= bar_num < seg['bar_end']:
            return {
                'pattern': seg['pattern'],
                'custom_pattern': seg['custom_pattern'],
                'reps': seg['reps'] if seg['reps'] is not None else defaults['reps'],
                'gate': seg['gate'] if seg['gate'] is not None else defaults['gate'],
                'accent': seg['accent'] if seg['accent'] is not None else defaults['accent'],
                'octave': seg['octave'] if seg['octave'] is not None else defaults['octave'],
            }
    return dict(defaults)


# ═══════════════════════════════════════════════════════════
# LECTURA DE MIDI Y DETECCIÓN DE LA PISTA DE ARMONÍA
# ═══════════════════════════════════════════════════════════

def extract_track_notes(track, tpb):
    """
    Extrae las notas de una pista (en ticks absolutos) y también la lista de
    mensajes que NO son note_on/note_off (con su tick absoluto), para poder
    reconstruir la pista más tarde conservando program_change, track_name, etc.
    Devuelve: notes (list of dict start,end,pitch,vel,channel), other_msgs (list of (abs_tick, msg))
    """
    notes = []
    other_msgs = []
    pending = {}
    abs_t = 0
    for msg in track:
        abs_t += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
        elif msg.type in ('note_off', 'note_on'):
            key = (msg.channel, msg.note)
            if key in pending:
                on_t, vel = pending.pop(key)
                notes.append({
                    'start': on_t, 'end': abs_t, 'pitch': msg.note,
                    'vel': vel, 'channel': msg.channel,
                })
        else:
            other_msgs.append((abs_t, msg))
    # notas que quedaron sin note_off explícito (MIDI mal formado): cerrarlas al final
    for (ch, note), (on_t, vel) in pending.items():
        notes.append({'start': on_t, 'end': abs_t, 'pitch': note, 'vel': vel, 'channel': ch})
    notes.sort(key=lambda n: n['start'])
    return notes, other_msgs


def polyphony_score(notes, tpb, window_beats=0.12):
    """
    Agrupa notas por onset cercano y devuelve (avg_group_size, n_groups, n_notes).
    Un avg_group_size alto indica una pista de acordes (armonía).
    """
    if not notes:
        return 0.0, 0, 0
    window = window_beats * tpb
    groups = []
    i = 0
    ordered = sorted(notes, key=lambda n: n['start'])
    while i < len(ordered):
        t0 = ordered[i]['start']
        j = i
        while j < len(ordered) and ordered[j]['start'] - t0 <= window:
            j += 1
        groups.append(j - i)
        i = j
    avg = sum(groups) / len(groups)
    return avg, len(groups), len(ordered)


def detect_harmony_track(mid, min_notes=2, verbose=False):
    """
    Recorre todas las pistas del MIDI y elige la más polifónica (probable
    armonía/acordes), ignorando el canal de batería. Devuelve:
      idx, notes, other_msgs, candidates_info (para depuración/reporte)
    """
    candidates = []
    for idx, track in enumerate(mid.tracks):
        notes, other_msgs = extract_track_notes(track, mid.ticks_per_beat)
        # descartar notas de batería (canal 9) del cómputo de polifonía
        mel_notes = [n for n in notes if n['channel'] != DRUM_CHANNEL]
        if not mel_notes:
            continue
        avg_poly, n_groups, n_notes = polyphony_score(mel_notes, mid.ticks_per_beat)
        track_name = None
        for msg in track:
            if msg.type == 'track_name':
                track_name = msg.name
                break
        candidates.append({
            'idx': idx, 'name': track_name, 'avg_poly': round(avg_poly, 3),
            'n_notes': n_notes, 'n_groups': n_groups, 'is_drum': False,
        })

    if not candidates:
        raise RuntimeError("No se encontraron notas melódicas/armónicas en el MIDI.")

    if verbose:
        print("    Pistas candidatas:")
        for c in candidates:
            label = c['name'] or '(sin nombre)'
            print(f"      #{c['idx']:<2} {label:<20} notas={c['n_notes']:<5} "
                  f"acordes~{c['n_groups']:<5} polifonía media={c['avg_poly']}")

    polyphonic = [c for c in candidates if c['avg_poly'] >= min_notes]
    pool = polyphonic if polyphonic else candidates
    best = max(pool, key=lambda c: (c['avg_poly'], c['n_notes']))

    if verbose:
        origin = "por polifonía" if polyphonic else "fallback (ninguna pista alcanzó min-notes)"
        print(f"    → Pista de armonía elegida: #{best['idx']} ({origin})")

    return best['idx'], candidates


# ═══════════════════════════════════════════════════════════
# AGRUPACIÓN DE NOTAS EN ACORDES
# ═══════════════════════════════════════════════════════════

def group_chords(notes, tpb, window_beats=0.12):
    """
    Agrupa las notas de la pista de armonía en acordes por onset cercano.
    Devuelve list of dict: start (tick), dur (tick), notes=[(pitch,vel,channel), …] (grave→agudo)
    """
    if not notes:
        return []
    window = window_beats * tpb
    ordered = sorted(notes, key=lambda n: n['start'])

    raw_groups = []
    i = 0
    while i < len(ordered):
        t0 = ordered[i]['start']
        group = []
        j = i
        while j < len(ordered) and ordered[j]['start'] - t0 <= window:
            group.append(ordered[j])
            j += 1
        raw_groups.append(group)
        i = j

    chords = []
    for group in raw_groups:
        start = min(n['start'] for n in group)
        # duración provisional = duración mínima del grupo (se recalcula abajo)
        dur = min(n['end'] - n['start'] for n in group)
        gnotes = sorted(group, key=lambda n: n['pitch'])
        chords.append({
            'start': start, 'dur': dur,
            'notes': [(n['pitch'], n['vel'], n['channel']) for n in gnotes],
        })

    # cada acorde dura hasta el siguiente onset (así el patrón ocupa el compás completo)
    for k in range(len(chords) - 1):
        chords[k]['dur'] = max(1, chords[k + 1]['start'] - chords[k]['start'])

    return chords


# ═══════════════════════════════════════════════════════════
# MOTOR DE PATRONES
# ═══════════════════════════════════════════════════════════

def resolve_role(token, notes):
    """
    Resuelve un token de rol (o índice) a una sublista de notas (pitch,vel,channel)
    dentro del acorde `notes` (ya ordenado grave→agudo).

    Soporta tokens compuestos entre corchetes, p.ej. "[0,1]" o "[bajo,medio]",
    que combinan varios roles/índices en un único paso simultáneo. Las notas
    resultantes se deduplican (por pitch+canal) preservando el orden de
    aparición.
    """
    token = token.strip()
    if token.startswith('[') and token.endswith(']'):
        inner = token[1:-1]
        combined = []
        seen = set()
        for sub in inner.split(','):
            sub = sub.strip()
            if not sub:
                continue
            for note in resolve_role(sub, notes):
                key = (note[0], note[2])  # (pitch, channel)
                if key not in seen:
                    seen.add(key)
                    combined.append(note)
        return combined if combined else [notes[0]]

    n = len(notes)
    t = ROLE_ALIASES.get(token.lower(), token.lower())
    if t == 'bass':
        return [notes[0]]
    if t == 'top':
        return [notes[-1]]
    if t == 'mid':
        return [notes[n // 2]] if n >= 3 else [notes[-1]]
    if t == 'rest':
        return notes[1:] if n > 1 else [notes[0]]
    if t == 'all':
        return list(notes)
    try:
        idx = int(t) % n
        return [notes[idx]]
    except ValueError:
        return [notes[0]]  # token desconocido → fallback a la más grave


def split_custom_pattern(custom_pattern):
    """
    Separa --custom-pattern en tokens por comas, respetando los corchetes
    "[...]" para que las comas dentro de ellos no partan un token simultáneo
    (p.ej. "[0,1],1,2,0" → ["[0,1]", "1", "2", "0"]).
    """
    tokens = []
    current = ''
    depth = 0
    for ch in custom_pattern:
        if ch == '[':
            depth += 1
            current += ch
        elif ch == ']':
            depth = max(0, depth - 1)
            current += ch
        elif ch == ',' and depth == 0:
            tokens.append(current)
            current = ''
        else:
            current += ch
    if current.strip():
        tokens.append(current)
    return [tok.strip() for tok in tokens if tok.strip()]


def build_tokens(pattern_name, custom_pattern, n_notes):
    """Devuelve la lista de tokens de rol para un acorde de n_notes notas."""
    if pattern_name == 'custom':
        if not custom_pattern:
            raise ValueError("--pattern custom requiere --custom-pattern \"rol1,rol2,…\"")
        return split_custom_pattern(custom_pattern)
    if pattern_name == 'arpeggio-up':
        return [str(i) for i in range(n_notes)]
    if pattern_name == 'arpeggio-down':
        return [str(i) for i in range(n_notes - 1, -1, -1)]
    if pattern_name == 'arpeggio-updown':
        up = list(range(n_notes))
        down = list(range(n_notes - 2, 0, -1)) if n_notes > 2 else []
        return [str(i) for i in up + down]
    return list(NAMED_PATTERNS[pattern_name])


def transpose_notes(notes, octave):
    """
    Transporta una lista de notas (pitch,vel,channel) `octave` octavas
    (positivo = sube, negativo = baja), recortando el pitch resultante al
    rango MIDI válido (0-127).
    """
    if not octave:
        return notes
    shift = octave * 12
    return [(max(0, min(127, pitch + shift)), vel, channel)
            for (pitch, vel, channel) in notes]


def render_chord(chord, pattern_name, custom_pattern, reps, gate, accent, octave=0):
    """
    Aplica el patrón rítmico a un acorde y devuelve una lista de eventos:
      (start_tick, end_tick, pitch, velocity, channel)
    `octave` transporta todas las notas del acorde N octavas antes de
    aplicar el patrón (positivo = sube, negativo = baja).
    """
    notes = transpose_notes(chord['notes'], octave)
    tokens = build_tokens(pattern_name, custom_pattern, len(notes))
    if not tokens:
        return []

    cycle_len = max(1, chord['dur'] // reps)
    events = []
    for rep in range(reps):
        cycle_start = chord['start'] + rep * cycle_len
        step_len = max(1, cycle_len // len(tokens))
        for step_idx, token in enumerate(tokens):
            step_start = cycle_start + step_idx * step_len
            step_dur = max(1, int(step_len * gate))
            step_notes = resolve_role(token, notes)
            for (pitch, vel, channel) in step_notes:
                v = vel
                if step_idx == 0 and accent != 1.0:
                    v = max(1, min(127, round(vel * accent)))
                events.append((step_start, step_start + step_dur, pitch, v, channel))
    return events


def render_all_chords(chords, tpb_bar, segments, defaults, verbose=False):
    """
    Renderiza todos los acordes, resolviendo para cada uno (según el compás en
    el que empieza) qué patrón/reps/gate/accent aplicar: el de un --segments
    que lo cubra, o los valores globales por defecto.
    Devuelve (events, assignment_log) donde assignment_log documenta qué
    configuración se usó para cada acorde (para --verbose / --report).
    """
    events = []
    assignment_log = []
    for chord in chords:
        bar_num = bar_of_tick(chord['start'], tpb_bar)
        cfg = resolve_segment_for_bar(segments, bar_num, defaults)
        events.extend(render_chord(
            chord, cfg['pattern'], cfg['custom_pattern'], cfg['reps'], cfg['gate'], cfg['accent'],
            cfg['octave']
        ))
        assignment_log.append({
            'bar': bar_num, 'start_tick': chord['start'],
            'pattern': cfg['pattern'], 'reps': cfg['reps'],
            'gate': cfg['gate'], 'accent': cfg['accent'], 'octave': cfg['octave'],
        })
        if verbose:
            print(f"      compás {bar_num:<4} → pattern={cfg['pattern']:<12} "
                  f"reps={cfg['reps']}  gate={cfg['gate']}  accent={cfg['accent']}  "
                  f"octave={cfg['octave']:+d}")
    return events, assignment_log


# ═══════════════════════════════════════════════════════════
# RECONSTRUCCIÓN DE LA PISTA MIDI
# ═══════════════════════════════════════════════════════════

def build_new_track(other_msgs, note_events):
    """
    Reconstruye una mido.MidiTrack fusionando los mensajes originales que no
    son notas (program_change, track_name, control_change, …) con los nuevos
    eventos de nota, todo ordenado por tick absoluto y convertido a delta-time.
    """
    merged = []  # (abs_tick, priority, msg)

    for abs_t, msg in other_msgs:
        if msg.type == 'end_of_track':
            continue
        merged.append((abs_t, 1, msg))

    for (start, end, pitch, vel, channel) in note_events:
        merged.append((start, 2, mido.Message('note_on', channel=channel, note=pitch, velocity=vel, time=0)))
        merged.append((end, 0, mido.Message('note_off', channel=channel, note=pitch, velocity=0, time=0)))

    merged.sort(key=lambda x: (x[0], x[1]))

    track = mido.MidiTrack()
    prev_t = 0
    for abs_t, _prio, msg in merged:
        delta = max(0, abs_t - prev_t)
        track.append(msg.copy(time=delta))
        prev_t = abs_t
    track.append(mido.MetaMessage('end_of_track', time=0))
    return track


# ═══════════════════════════════════════════════════════════
# ORQUESTACIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════

def voice_chords(input_midi, track_override=None, pattern='bombo-caja', custom_pattern=None,
                  reps=1, gate=0.85, accent=1.0, octave=0, segments_spec=None, min_notes=2,
                  out_path=None, out_dir=None, report=False, verbose=False):

    print(f"\n{'═'*64}")
    print(f"  CHORD PATTERN VOICER")
    print(f"{'═'*64}")
    print(f"  Entrada  : {input_midi}")

    try:
        mid = mido.MidiFile(input_midi)
    except Exception as e:
        print(f"[ERROR] No se pudo abrir el MIDI: {e}")
        return {}

    tpb = mid.ticks_per_beat or 480
    ts_num, ts_den = detect_time_signature(mid)
    tpb_bar = ticks_per_bar(tpb, ts_num, ts_den)
    print(f"  Ticks/beat: {tpb}   Compás: {ts_num}/{ts_den}   "
          f"Pistas: {len(mid.tracks)}   Tipo: {mid.type}")

    # ── Segmentos por rango de compases ────────────────────────────────────────
    segments = []
    if segments_spec:
        try:
            segments = parse_segments(segments_spec)
        except ValueError as e:
            print(f"[ERROR] --segments: {e}")
            return {}
        print(f"\n  Segmentos configurados ({len(segments)}):")
        for seg in segments:
            extra = f"  custom={seg['custom_pattern']}" if seg['custom_pattern'] else ""
            octave_str = f"  octave={seg['octave']:+d}" if seg['octave'] is not None else ""
            print(f"    compases {seg['bar_start']}-{seg['bar_end']-1}: "
                  f"pattern={seg['pattern']}"
                  f"{'  reps=' + str(seg['reps']) if seg['reps'] is not None else ''}"
                  f"{'  gate=' + str(seg['gate']) if seg['gate'] is not None else ''}"
                  f"{'  accent=' + str(seg['accent']) if seg['accent'] is not None else ''}"
                  f"{octave_str}"
                  f"{extra}")
        print(f"    (resto de compases) → pattern={pattern}  reps={reps}  "
              f"gate={gate}  accent={accent}  octave={octave:+d}")

    defaults = {'pattern': pattern, 'custom_pattern': custom_pattern,
                'reps': reps, 'gate': gate, 'accent': accent, 'octave': octave}

    # ── Detección / selección de pista de armonía ─────────────────────────────
    print("\n  [1/3] Detectando pista de armonía…")
    try:
        if track_override is not None:
            if not (0 <= track_override < len(mid.tracks)):
                raise ValueError(f"--track {track_override} fuera de rango (0-{len(mid.tracks)-1})")
            harm_idx = track_override
            _, candidates = detect_harmony_track(mid, min_notes=min_notes, verbose=verbose)
            print(f"    → Pista de armonía forzada: #{harm_idx}")
        else:
            harm_idx, candidates = detect_harmony_track(mid, min_notes=min_notes, verbose=verbose)
    except Exception as e:
        print(f"[ERROR] Detección de pista: {e}")
        return {}

    notes, other_msgs = extract_track_notes(mid.tracks[harm_idx], tpb)
    if not notes:
        print(f"[ERROR] La pista #{harm_idx} no contiene notas.")
        return {}

    chords = group_chords(notes, tpb)
    print(f"    Notas en pista de armonía : {len(notes)}")
    print(f"    Acordes detectados        : {len(chords)}")
    if verbose:
        for c in chords[:8]:
            names = [pitch_name(p) for p, _, _ in c['notes']]
            bar_num = bar_of_tick(c['start'], tpb_bar)
            print(f"      compás={bar_num:<4} tick={c['start']:<6} dur={c['dur']:<5} notas={names}")
        if len(chords) > 8:
            print(f"      … ({len(chords) - 8} más)")

    # ── Aplicación del patrón ──────────────────────────────────────────────────
    print(f"\n  [2/3] Aplicando patrón(es)…")
    try:
        note_events, assignment_log = render_all_chords(
            chords, tpb_bar, segments, defaults, verbose=verbose
        )
    except Exception as e:
        print(f"[ERROR] Aplicando patrón: {e}")
        if verbose:
            traceback.print_exc()
        return {}
    print(f"    Eventos de nota generados : {len(note_events)}")

    # ── Reconstrucción y escritura ─────────────────────────────────────────────
    print("\n  [3/3] Reescribiendo MIDI…")
    new_track = build_new_track(other_msgs, note_events)
    mid.tracks[harm_idx] = new_track

    stem = os.path.splitext(os.path.basename(input_midi))[0]
    if out_path:
        final_out = out_path
    else:
        directory = out_dir or os.path.dirname(os.path.abspath(input_midi))
        final_out = os.path.join(directory, f"{stem}.voiced.mid")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(final_out)), exist_ok=True)
        mid.save(final_out)
        print(f"    → {final_out}")
    except Exception as e:
        print(f"[ERROR] Escritura MIDI: {e}")
        if verbose:
            traceback.print_exc()
        return {}

    report_data = {
        'input': input_midi,
        'output': final_out,
        'ticks_per_beat': tpb,
        'time_signature': [ts_num, ts_den],
        'ticks_per_bar': tpb_bar,
        'harmony_track_index': harm_idx,
        'track_candidates': candidates,
        'n_chords': len(chords),
        'default_pattern': pattern,
        'default_custom_pattern': custom_pattern,
        'default_reps': reps,
        'default_gate': gate,
        'default_accent': accent,
        'default_octave': octave,
        'segments': [{k: v for k, v in seg.items()} for seg in segments],
        'n_note_events': len(note_events),
        'chord_assignment': assignment_log,
        'chords_preview': [
            {'start_tick': c['start'], 'dur_tick': c['dur'],
             'notes': [pitch_name(p) for p, _, _ in c['notes']]}
            for c in chords[:20]
        ],
    }

    if report:
        report_path = os.path.join(os.path.dirname(os.path.abspath(final_out)),
                                    f"{stem}.voiced_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\n  Reporte: {report_path}")

    print(f"\n{'═'*64}")
    print(f"  Completado.")
    print(f"{'═'*64}\n")

    return report_data


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def print_pattern_help():
    print("\nPatrones disponibles (--pattern):\n")
    print("  block             todas las notas del acorde a la vez")
    print("  bombo-caja         1º la nota más grave, 2º el resto del acorde junto")
    print("  alberti            bajo, agudo, medio, agudo (bajo-alberti clásico)")
    print("  arpeggio-up         arpegio ascendente nota a nota")
    print("  arpeggio-down       arpegio descendente nota a nota")
    print("  arpeggio-updown     arpegio ascendente y luego descendente")
    print("  custom             patrón propio vía --custom-pattern \"rol1,rol2,…\"\n")
    print("Roles válidos para --custom-pattern:")
    print("  bajo/bass/root · agudo/top/alto · medio/mid · resto/rest/caja ·")
    print("  todas/all/block · índices numéricos (0, 1, 2, -1, …)\n")


def build_parser():
    p = argparse.ArgumentParser(
        description='CHORD PATTERN VOICER — Reescribe el patrón rítmico de la pista de armonía de un MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python chord_pattern_voicer.py cancion.mid
  python chord_pattern_voicer.py cancion.mid --pattern bombo-caja
  python chord_pattern_voicer.py cancion.mid --pattern alberti --reps 2
  python chord_pattern_voicer.py cancion.mid --pattern arpeggio-up --reps 2 --gate 0.6
  python chord_pattern_voicer.py cancion.mid --pattern custom --custom-pattern "bajo,agudo,resto,medio"
  python chord_pattern_voicer.py cancion.mid --track 2 --verbose --report
  python chord_pattern_voicer.py cancion.mid --pattern arpeggio-up --octave -1
  python chord_pattern_voicer.py cancion.mid --pattern alberti --octave 2

  # Patrón distinto por tramos de la pieza:
  python chord_pattern_voicer.py cancion.mid --segments \\
      "1-10:pattern=bombo-caja,reps=1" \\
      "10-20:pattern=alberti,reps=2" \\
      "20-30:pattern=custom,reps=2,custom=bajo|agudo|resto|medio"

  python chord_pattern_voicer.py --list-patterns
        """
    )
    p.add_argument('input_midi', nargs='?', help='MIDI de entrada con varias pistas')
    p.add_argument('--track', type=int, default=None, metavar='N',
                   help='Fuerza el índice (0-based) de la pista de armonía')
    p.add_argument('--pattern', default='bombo-caja',
                   choices=list(NAMED_PATTERNS.keys()),
                   help='Patrón rítmico por defecto (default: bombo-caja); '
                        'se usa donde --segments no cubra')
    p.add_argument('--custom-pattern', default=None, metavar='ROLES',
                   help='Roles separados por comas, para --pattern custom '
                        '(ej: "bajo,agudo,resto,medio"). Usá "[a,b]" para '
                        'roles/índices simultáneos (ej: "[0,1],1,2,0")')
    p.add_argument('--reps', type=int, default=1, metavar='N',
                   help='Repeticiones del patrón por acorde/compás (default: 1)')
    p.add_argument('--gate', type=float, default=0.85, metavar='F',
                   help='Fracción sonante de cada paso, 0-1 (default: 0.85)')
    p.add_argument('--accent', type=float, default=1.0, metavar='F',
                   help='Multiplicador de velocity del primer golpe de cada '
                        'repetición (default: 1.0 = sin acento)')
    p.add_argument('--octave', type=int, default=0, metavar='N',
                   help='Transporta todas las notas del acorde N octavas '
                        '(default: 0). Positivo sube, negativo baja, '
                        'ej: --octave -1, --octave 2. También se puede fijar '
                        'por tramo con octave=N dentro de --segments')
    p.add_argument('--segments', nargs='+', default=None, metavar='SPEC',
                   help='Patrón/velocidad distintos por rango de compases. '
                        'Cada SPEC: "INICIO-FIN:pattern=NOMBRE[,reps=N][,gate=F]'
                        '[,accent=F][,custom=a|b|c]" con FIN exclusivo. '
                        'Ej: "1-10:pattern=bombo-caja,reps=1"')
    p.add_argument('--min-notes', type=int, default=2, metavar='N',
                   help='Mínimo de notas simultáneas para considerar polifónica '
                        'una pista en la autodetección (default: 2)')
    p.add_argument('--out', default=None, metavar='FILE',
                   help='Ruta del MIDI de salida (default: <input>.voiced.mid)')
    p.add_argument('--out-dir', default=None, metavar='DIR',
                   help='Carpeta de salida (default: junto al MIDI de entrada)')
    p.add_argument('--list-patterns', action='store_true',
                   help='Lista los patrones disponibles y sale')
    p.add_argument('--report', action='store_true',
                   help='Guarda un JSON con el análisis completo')
    p.add_argument('--verbose', action='store_true',
                   help='Informe detallado por stdout')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list_patterns:
        print_pattern_help()
        return

    if not args.input_midi:
        parser.print_help()
        return

    if not os.path.isfile(args.input_midi):
        print(f"[ERROR] No se encuentra el archivo: {args.input_midi}")
        sys.exit(1)

    if not (0.0 < args.gate <= 1.0):
        print(f"[ERROR] --gate debe estar en (0, 1]. Recibido: {args.gate}")
        sys.exit(1)

    if args.reps < 1:
        print(f"[ERROR] --reps debe ser >= 1. Recibido: {args.reps}")
        sys.exit(1)

    if args.pattern == 'custom' and not args.custom_pattern:
        print("[ERROR] --pattern custom requiere --custom-pattern \"rol1,rol2,…\"")
        sys.exit(1)

    voice_chords(
        input_midi=args.input_midi,
        track_override=args.track,
        pattern=args.pattern,
        custom_pattern=args.custom_pattern,
        reps=args.reps,
        gate=args.gate,
        accent=args.accent,
        octave=args.octave,
        segments_spec=args.segments,
        min_notes=args.min_notes,
        out_path=args.out,
        out_dir=args.out_dir,
        report=args.report,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
