#!/usr/bin/env python3
"""
midi_transpose.py — MIDI Transpose: transposición de un MIDI COMPLETO sin tocar lo que no debe tocarse
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
La suite no tenía una transposición de fichero completo: part_extractor --transpose
trabaja por partes (hay que trocear y volver a unir con midi_merge) y otras
herramientas (unifier, midi_dna_unified) transponen solo como efecto colateral de un
plan tonal. Para llevar una obra de Re menor a Re mayor, o de Do a Mi♭, bastaba un
comando que no existía. Esta herramienta lo es, y mueve únicamente lo que es música:

  · NO transpone el canal 9 (percusión GM): un bombo desplazado es otro instrumento.
  · NO transpone los keyswitches (notas < E1 de duración mínima, ≈2 ticks, que emiten
    p. ej. piano_to_orchestra.py y las librerías de samples): moverlos rompe el cambio
    de articulación. Mismo criterio que playability_auditor.py.
  · Reescribe los meta-eventos key_signature a la nueva tonalidad.
  · Falla (rc=2) en lugar de producir notas fuera de 0..127, salvo con --fold.
  · No cambia duraciones, velocidades, tempos, CC ni ninguna otra cosa.

  info     Pistas, tesitura y tonalidad estimada (Krumhansl-Schmuckler ponderada por
           duración, sin percusión ni keyswitches)
  shift    Transpone ±N semitonos
  to-key   Lleva la tónica de la obra a otra tonalidad por el camino más corto
           (detecta la tonalidad de origen, o se indica con --from)
  fit      Busca el desplazamiento (±--max-shift) que deja más notas dentro de una
           tesitura LO-HI; en empate, el de menor valor absoluto
  selftest Comprueba el flujo con un MIDI sintético

La detección de tonalidad es una estimación por perfiles: en obras con modulaciones
o muy cromáticas conviene confirmarla con key_consensus.py y pasar --from.
to-key mueve la tónica, no el modo: de Re menor a «D» (Re mayor) usa --key D con el modo
mayor, pero NO convierte las terceras; avisa cuando el modo pedido difiere del detectado.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  info    midi [--json]
  shift   midi --semitones N [-o SALIDA.mid] [--tracks N ...] [--exclude-channels CH ...]
               [--fold] [--dry-run] [--json]
  to-key  midi --key TONALIDAD [--from TONALIDAD] [--direction nearest|up|down]
               [-o SALIDA.mid] [--tracks N ...] [--exclude-channels CH ...]
               [--fold] [--dry-run] [--json]
  fit     midi --range LO-HI [--max-shift 12] [-o SALIDA.mid] [--tracks N ...]
               [--exclude-channels CH ...] [--fold] [--dry-run] [--strict] [--json]
  selftest

Tonalidades: Dm, D, F#m, Bb, «D minor», «Eb mayor»… (m/min/minor/menor = menor).
Tesituras: nombres (C3-C6, con C4 = 60) o números MIDI (48-84).
--tracks usa los índices que muestra «info». Códigos de salida: 0 correcto ·
2 argumento/lectura no válidos o notas fuera de 0..127 · 3 (solo fit --strict) si tras
el mejor desplazamiento quedan notas fuera de la tesitura.

EJEMPLOS
  python midi_transpose.py info obra.mid
  python midi_transpose.py shift obra.mid --semitones -3 -o obra_m3.mid
  python midi_transpose.py to-key obra.mid --key D -o obra_Dmayor.mid
  python midi_transpose.py to-key obra.mid --from Dm --key Gm --direction up --dry-run
  python midi_transpose.py fit melodia.mid --range G3-C6 --dry-run

@params v1
[info]
midi        file  io=in ext=mid required  # MIDI a inspeccionar
--json      bool  default=false  # Salida en JSON
[shift]
midi        file  io=in ext=mid required  # MIDI de entrada
--semitones int   required min=-60 max=60 unit=semitonos alias=-s  # Desplazamiento
--output    file  io=out ext=mid alias=-o  # Salida (default: <entrada>_t+N.mid)
--tracks    list  of=int  # Índices de pista a transponer (default: todas)
--exclude-channels list of=int default=9  # Canales que no se transponen (percusión)
--fold      bool  default=false  # Plegar por octavas las notas que se salgan de 0..127
--dry-run   bool  default=false  # Solo informa, no escribe
--json      bool  default=false  # Salida en JSON
[to-key]
midi        file  io=in ext=mid required  # MIDI de entrada
--key       str   required  # Tonalidad de destino (Dm, D, F#m, Bb…)
--from      str   # Tonalidad de origen (default: estimada)
--direction enum  default=nearest values=nearest|up|down  # Sentido del desplazamiento
--output    file  io=out ext=mid alias=-o  # Salida (default: <entrada>_<tonalidad>.mid)
--tracks    list  of=int  # Índices de pista a transponer (default: todas)
--exclude-channels list of=int default=9  # Canales que no se transponen (percusión)
--fold      bool  default=false  # Plegar por octavas las notas que se salgan de 0..127
--dry-run   bool  default=false  # Solo informa, no escribe
--json      bool  default=false  # Salida en JSON
[fit]
midi        file  io=in ext=mid required  # MIDI de entrada
--range     str   required format="LO-HI"  # Tesitura objetivo (C3-C6 o 48-84)
--max-shift int   default=12 min=0 max=36 unit=semitonos  # Desplazamiento máximo a probar
--output    file  io=out ext=mid alias=-o  # Salida (default: <entrada>_fit+N.mid)
--tracks    list  of=int  # Índices de pista a considerar y transponer (default: todas)
--exclude-channels list of=int default=9  # Canales que no se transponen (percusión)
--fold      bool  default=false  # Plegar por octavas las notas que se salgan de 0..127
--dry-run   bool  default=false  # Solo informa, no escribe
--strict    bool  default=false  # rc=3 si quedan notas fuera de la tesitura
--json      bool  default=false  # Salida en JSON
@end
"""
import argparse
import json
import math
import os
import re
import sys
import tempfile

import mido

VERSION = "1.0"
DRUM_CH = 9
KS_PITCH_MAX = 28            # < E1: por debajo de cualquier instrumento (cf. playability_auditor)

PC_MAJ = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
PC_MIN = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KRUM_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KRUM_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_KEY_RE = re.compile(r"^\s*([A-Ga-g])\s*([#b♯♭]?)\s*(m|min|minor|menor|M|maj|major|mayor)?\s*$")
_NOTE_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)(-?\d+)$")


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre."""
    if isinstance(path, (str, bytes, os.PathLike)):
        d = os.path.dirname(os.path.abspath(os.fsdecode(path)))
        if d:
            os.makedirs(d, exist_ok=True)
    return path


def _open_midi(path):
    """Lectura estricta; si falla por datos fuera de rango, reintenta con clip=True
    y avisa; si tampoco, error con pista hacia midi_doctor."""
    try:
        return mido.MidiFile(path)
    except FileNotFoundError:
        raise
    except (OSError, ValueError, EOFError, IndexError, KeyError) as e1:
        try:
            mid = mido.MidiFile(path, clip=True)
        except Exception as e2:
            raise ValueError(f"MIDI ilegible ({e1!r}; con clip=True: {e2!r}). "
                             f"Pruebe: python midi_doctor.py fix {path}") from e2
        print(f"[AVISO] {path}: MIDI con datos fuera de rango ({e1}); leído con clip=True. "
              f"Repárelo con: python midi_doctor.py fix {path}", file=sys.stderr)
        return mid


# ══════════════════════════════════════════════════════════════════════════════
#  TONALIDADES Y NOTAS
# ══════════════════════════════════════════════════════════════════════════════

def parse_key(txt):
    """'Dm', 'F#m', 'Bb', 'D minor' → (pc, 'maj'|'min')."""
    m = _KEY_RE.match(str(txt))
    if not m:
        raise ValueError(f"tonalidad no reconocida: {txt!r} (ejemplos: Dm, D, F#m, Bb)")
    letter, acc, mode = m.groups()
    pc = _LETTER[letter.upper()] + (1 if acc in ("#", "♯") else -1 if acc in ("b", "♭") else 0)
    minor = mode in ("m", "min", "minor", "menor")
    return pc % 12, "min" if minor else "maj"


def key_name(pc, mode):
    return (PC_MIN[pc % 12] + "m") if mode == "min" else PC_MAJ[pc % 12]


def key_label(pc, mode):
    return f"{key_name(pc, mode)} ({'menor' if mode == 'min' else 'mayor'})"


def note_name(p):
    return f"{SHARPS[p % 12]}{p // 12 - 1}"


def parse_pitch(txt):
    txt = str(txt).strip()
    if re.fullmatch(r"-?\d+", txt):
        return int(txt)
    m = _NOTE_RE.match(txt)
    if not m:
        raise ValueError(f"nota no reconocida: {txt!r} (ejemplos: C3, F#4, 60)")
    letter, acc, octv = m.groups()
    pc = _LETTER[letter.upper()] + (1 if acc in ("#", "♯") else -1 if acc in ("b", "♭") else 0)
    return (int(octv) + 1) * 12 + pc


def parse_range(txt):
    parts = str(txt).split("-", 1) if re.match(r"^[A-Ga-g]", str(txt)) is None else \
        re.split(r"-(?=[A-Ga-g0-9])", str(txt), maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"tesitura no válida: {txt!r} (formato LO-HI, p. ej. C3-C6 o 48-84)")
    lo, hi = parse_pitch(parts[0]), parse_pitch(parts[1])
    if lo > hi:
        raise ValueError(f"tesitura invertida: {txt!r}")
    return lo, hi


def _shift_key_str(key, semitones):
    pc, mode = parse_key(key)
    return key_name((pc + semitones) % 12, mode)


# ══════════════════════════════════════════════════════════════════════════════
#  ESCANEO DE NOTAS
# ══════════════════════════════════════════════════════════════════════════════

def _scan_track(tr, tpb, exclude_ch):
    """→ (notes, ks). notes = [i_on, i_off|None, canal, pitch, inicio, fin];
    ks = índices de mensajes que son keyswitch (nota < E1 de duración mínima)."""
    t = 0
    open_ = {}
    notes = []
    for i, m in enumerate(tr):
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            open_.setdefault((m.channel, m.note), []).append((i, t))
        elif m.type == "note_off" or (m.type == "note_on" and m.velocity == 0):
            q = open_.get((m.channel, m.note))
            if q:
                i0, t0 = q.pop(0)
                notes.append([i0, i, m.channel, m.note, t0, t])
    for (ch, p), q in open_.items():
        for i0, t0 in q:
            notes.append([i0, None, ch, p, t0, t])
    ks_max = max(4, tpb // 32)
    ks = set()
    for n in notes:
        if n[2] not in exclude_ch and n[3] < KS_PITCH_MAX and (n[5] - n[4]) <= ks_max:
            ks.add(n[0])
            if n[1] is not None:
                ks.add(n[1])
    return notes, ks


def eligible_notes(mid, tracks=None, exclude_ch=(DRUM_CH,)):
    """Notas musicales: sin percusión, sin keyswitches, solo de las pistas elegidas."""
    out = []
    for ti, tr in enumerate(mid.tracks):
        if tracks is not None and ti not in tracks:
            continue
        notes, ks = _scan_track(tr, mid.ticks_per_beat, exclude_ch)
        for i0, _i1, ch, p, s, e in notes:
            if ch in exclude_ch or i0 in ks:
                continue
            out.append((ti, ch, p, s, e))
    return out


def _pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else 0.0


def detect_key(mid, tracks=None, exclude_ch=(DRUM_CH,)):
    """Krumhansl-Schmuckler ponderado por duración → lista [(r, pc, modo)] ordenada, o None."""
    hist = [0.0] * 12
    for _ti, _ch, p, s, e in eligible_notes(mid, tracks, exclude_ch):
        hist[p % 12] += max(1, e - s)
    if not sum(hist):
        return None
    sc = []
    for tonic in range(12):
        rot = [hist[(tonic + k) % 12] for k in range(12)]
        sc.append((_pearson(rot, KRUM_MAJ), tonic, "maj"))
        sc.append((_pearson(rot, KRUM_MIN), tonic, "min"))
    sc.sort(reverse=True)
    return sc


def _confidence(sc):
    if not sc or len(sc) < 2:
        return "n/d"
    r1, r2 = sc[0][0], sc[1][0]
    if r1 >= 0.75 and r1 - r2 >= 0.05:
        return "alta"
    if r1 >= 0.6:
        return "media"
    return "baja"


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSPOSICIÓN
# ══════════════════════════════════════════════════════════════════════════════

def transpose_midi(mid, semitones, tracks=None, exclude_ch=(DRUM_CH,), fold=False):
    """Devuelve (MidiFile nuevo, estadísticas). Lanza ValueError si hay notas fuera de
    0..127 y no se pidió fold."""
    out = mido.MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
    st = {"moved": 0, "drums": 0, "keyswitches": 0, "folded": 0, "overflow": 0,
          "key_signatures": 0, "tracks_untouched": 0}
    for ti, tr in enumerate(mid.tracks):
        nt = mido.MidiTrack()
        selected = tracks is None or ti in tracks
        if not selected:
            st["tracks_untouched"] += 1
        _notes, ks = _scan_track(tr, mid.ticks_per_beat, exclude_ch)
        for i, m in enumerate(tr):
            if m.type in ("note_on", "note_off", "polytouch") and selected \
                    and m.channel not in exclude_ch and i not in ks and semitones:
                new = m.note + semitones
                if not 0 <= new <= 127:
                    if fold:
                        while new < 0:
                            new += 12
                        while new > 127:
                            new -= 12
                        if m.type == "note_on" and m.velocity > 0:
                            st["folded"] += 1
                    else:
                        if m.type == "note_on" and m.velocity > 0:
                            st["overflow"] += 1
                        new = min(127, max(0, new))
                if m.type == "note_on" and m.velocity > 0:
                    st["moved"] += 1
                nt.append(m.copy(note=new))
                continue
            if m.type == "note_on" and m.velocity > 0:
                if m.channel in exclude_ch:
                    st["drums"] += 1
                elif i in ks:
                    st["keyswitches"] += 1
            if m.type == "key_signature" and tracks is None and semitones % 12:
                try:
                    nt.append(m.copy(key=_shift_key_str(m.key, semitones)))
                    st["key_signatures"] += 1
                    continue
                except ValueError:
                    pass
            nt.append(m.copy())
        out.tracks.append(nt)
    if st["overflow"]:
        raise ValueError(f"{st['overflow']} nota(s) saldrían de 0..127 con {semitones:+d} semitonos; "
                         f"use --fold para plegarlas por octavas o reduzca el desplazamiento")
    return out, st


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def _track_name(tr, idx):
    for m in tr:
        if m.type == "track_name" and m.name.strip():
            return m.name.strip()
    return f"pista {idx}"


def _exclude(a):
    return tuple(a.exclude_channels) if getattr(a, "exclude_channels", None) is not None else (DRUM_CH,)


def _tracks(a):
    return set(a.tracks) if getattr(a, "tracks", None) else None


def _default_out(path, tag):
    base, _ = os.path.splitext(path)
    return f"{base}_{tag}.mid"


def _finish_shift(a, mid, semitones, tag, extra=None):
    """Común a shift / to-key / fit: aplica, informa y escribe (salvo --dry-run)."""
    ex, tr = _exclude(a), _tracks(a)
    res = dict(extra or {})
    res.update({"input": a.midi, "semitones": semitones})
    if a.dry_run:
        pitches = [p for _t, _c, p, _s, _e in eligible_notes(mid, tr, ex)]
        over = sum(1 for p in pitches if not 0 <= p + semitones <= 127)
        res.update({"dry_run": True, "notes": len(pitches), "out_of_midi_range": over})
        res["output"] = None
    else:
        new, st = transpose_midi(mid, semitones, tr, ex, a.fold)
        out = a.output or _default_out(a.midi, tag)
        new.save(_ensure_parent(out))
        res.update({"dry_run": False, "output": out, **st})
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    print(f"  Desplazamiento: {semitones:+d} semitonos" + ("   (dry-run: no se escribe)" if a.dry_run else ""))
    if a.dry_run:
        print(f"  Notas afectadas: {res['notes']}   fuera de 0..127: {res['out_of_midi_range']}")
    else:
        print(f"  Notas transpuestas: {res['moved']}   percusión intacta: {res['drums']}   "
              f"keyswitches intactos: {res['keyswitches']}")
        if res["key_signatures"]:
            print(f"  key_signature reescritos: {res['key_signatures']}")
        if res["folded"]:
            print(f"  Notas plegadas por octavas (--fold): {res['folded']}")
        print(f"  Guardado en: {res['output']}")
    return 0


def cmd_info(a):
    mid = _open_midi(a.midi)
    ex = (DRUM_CH,)
    rows = []
    for ti, tr in enumerate(mid.tracks):
        notes, ks = _scan_track(tr, mid.ticks_per_beat, ex)
        real = [n for n in notes if n[0] not in ks]
        chans = sorted({n[2] for n in notes})
        if not notes:
            continue
        ps = [n[3] for n in real if n[2] not in ex] or [n[3] for n in real]
        rows.append({"track": ti, "name": _track_name(tr, ti), "channels": chans,
                     "notes": len(real), "keyswitches": len(ks) // 2 if ks else 0,
                     "percussion": DRUM_CH in chans,
                     "lowest": note_name(min(ps)) if ps else None,
                     "highest": note_name(max(ps)) if ps else None})
    sc = detect_key(mid)
    key = None
    if sc:
        key = {"best": key_label(sc[0][1], sc[0][2]), "r": round(sc[0][0], 3),
               "second": key_label(sc[1][1], sc[1][2]), "r2": round(sc[1][0], 3),
               "confidence": _confidence(sc)}
    sigs = []
    for tr in mid.tracks:
        for m in tr:
            if m.type == "key_signature" and m.key not in sigs:
                sigs.append(m.key)
    if a.json:
        print(json.dumps({"input": a.midi, "tracks": rows, "key_estimate": key,
                          "key_signatures": sigs}, ensure_ascii=False, indent=2))
        return 0
    print(f"\n  MIDI TRANSPOSE v{VERSION} · info · {os.path.basename(a.midi)}")
    print(f"  {'pista':>5}  {'nombre':26} {'canales':10} {'notas':>6}  tesitura")
    for r in rows:
        tes = f"{r['lowest']}–{r['highest']}" if r["lowest"] else "—"
        tag = "  (percusión: no se transpone)" if r["percussion"] else ""
        ksn = f"  +{r['keyswitches']} keyswitch" if r["keyswitches"] else ""
        print(f"  {r['track']:>5}  {r['name'][:26]:26} {','.join(map(str, r['channels'])):10} "
              f"{r['notes']:>6}  {tes}{tag}{ksn}")
    if key:
        print(f"\n  Tonalidad estimada: {key['best']}  (r={key['r']}, confianza {key['confidence']}; "
              f"2ª: {key['second']} r={key['r2']})")
    else:
        print("\n  Tonalidad estimada: n/d (sin notas melódicas)")
    print(f"  key_signature en el fichero: {', '.join(sigs) if sigs else 'ninguno'}")
    return 0


def cmd_shift(a):
    mid = _open_midi(a.midi)
    s = a.semitones
    tag = f"t{s:+d}"
    return _finish_shift(a, mid, s, tag)


def cmd_to_key(a):
    mid = _open_midi(a.midi)
    ex, tr = _exclude(a), _tracks(a)
    t_pc, t_mode = parse_key(a.key)
    if a.from_key:
        s_pc, s_mode = parse_key(a.from_key)
        conf, how = "indicada", "indicada con --from"
    else:
        sc = detect_key(mid, tr, ex)
        if not sc:
            raise ValueError("no hay notas melódicas para estimar la tonalidad; use --from")
        _r, s_pc, s_mode = sc[0]
        conf = _confidence(sc)
        how = f"estimada (r={sc[0][0]:.2f}, confianza {conf})"
        if conf == "baja":
            print("[AVISO] tonalidad de origen poco fiable; confírmela con key_consensus.py "
                  "y repita con --from", file=sys.stderr)
    d = (t_pc - s_pc) % 12
    if a.direction == "up":
        shift = d
    elif a.direction == "down":
        shift = d - 12 if d else 0
    else:
        shift = d if d <= 6 else d - 12
    if s_mode != t_mode and not a.json:
        print(f"[AVISO] el modo difiere ({key_label(s_pc, s_mode)} → {key_label(t_pc, t_mode)}): "
              f"solo se mueve la tónica, las terceras no cambian", file=sys.stderr)
    if not a.json:
        print(f"\n  MIDI TRANSPOSE v{VERSION} · to-key")
        print(f"  Origen: {key_label(s_pc, s_mode)} ({how})   Destino: {key_label(t_pc, t_mode)}")
    extra = {"from": key_label(s_pc, s_mode), "to": key_label(t_pc, t_mode),
             "from_source": how, "mode_changed": s_mode != t_mode}
    return _finish_shift(a, mid, shift, key_name(t_pc, t_mode), extra)


def cmd_fit(a):
    mid = _open_midi(a.midi)
    lo, hi = parse_range(a.range)
    ex, tr = _exclude(a), _tracks(a)
    pitches = [p for _t, _c, p, _s, _e in eligible_notes(mid, tr, ex)]
    if not pitches:
        raise ValueError("no hay notas melódicas que ajustar")
    table = []
    for s in range(-a.max_shift, a.max_shift + 1):
        fuera = sum(1 for p in pitches if not lo <= p + s <= hi)
        table.append((fuera, abs(s), s))
    table.sort()
    best_out, _abs, best = table[0]
    if not a.json:
        print(f"\n  MIDI TRANSPOSE v{VERSION} · fit {note_name(lo)}–{note_name(hi)}")
        print(f"  {len(pitches)} notas · tesitura actual {note_name(min(pitches))}–{note_name(max(pitches))}")
        print("  Mejores desplazamientos (fuera de tesitura):")
        for fuera, _a2, s in table[:5]:
            print(f"    {s:+3d} st  →  {fuera:>4} fuera ({100 * fuera / len(pitches):.1f} %)")
    extra = {"range": [note_name(lo), note_name(hi)], "best_shift": best,
             "out_of_range_after": best_out, "notes": len(pitches)}
    if best == 0 and not a.dry_run:
        if not a.json:
            print("  Ya está en la mejor posición: no se escribe ningún fichero.")
        else:
            print(json.dumps({"input": a.midi, "semitones": 0, "output": None, **extra},
                             ensure_ascii=False, indent=2))
        return 3 if (a.strict and best_out) else 0
    rc = _finish_shift(a, mid, best, f"fit{best:+d}", extra)
    if a.strict and best_out:
        print(f"[AVISO] quedan {best_out} nota(s) fuera de la tesitura", file=sys.stderr)
        return 3
    return rc


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic(path):
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("key_signature", key="Dm"))
    meta.append(mido.MetaMessage("set_tempo", tempo=500000))
    mid.tracks.append(meta)
    mel = mido.MidiTrack()
    mel.append(mido.MetaMessage("track_name", name="mel"))
    mel.append(mido.Message("note_on", channel=0, note=21, velocity=100, time=0))       # keyswitch
    mel.append(mido.Message("note_off", channel=0, note=21, velocity=0, time=2))
    for p in [62, 64, 65, 67, 69, 70, 73, 74] * 2 + [62]:                               # Re menor
        mel.append(mido.Message("note_on", channel=0, note=p, velocity=80, time=0))
        mel.append(mido.Message("note_off", channel=0, note=p, velocity=0, time=480))
    mid.tracks.append(mel)
    dr = mido.MidiTrack()
    dr.append(mido.MetaMessage("track_name", name="drums"))
    for p in (36, 38, 42):
        dr.append(mido.Message("note_on", channel=9, note=p, velocity=90, time=0))
        dr.append(mido.Message("note_off", channel=9, note=p, velocity=0, time=240))
    mid.tracks.append(dr)
    mid.save(path)
    return path


def _pitch_list(mid):
    out = []
    for tr in mid.tracks:
        for m in tr:
            if m.type == "note_on" and m.velocity > 0:
                out.append((m.channel, m.note))
    return out


def cmd_selftest(a):
    d = tempfile.mkdtemp()
    src = _synthetic(os.path.join(d, "s.mid"))
    mid = mido.MidiFile(src)
    res = []

    sc = detect_key(mid)
    res.append(("detecta Re menor sin contar percusión ni keyswitch",
                sc and (sc[0][1], sc[0][2]) == (2, "min")))

    new, st = transpose_midi(mid, 2)
    before, after = _pitch_list(mid), _pitch_list(new)
    mel_ok = all(b[1] + 2 == a2[1] for b, a2 in zip(before, after) if b[0] == 0 and b[1] != 21)
    res.append(("shift +2 mueve la melodía", mel_ok))
    res.append(("percusión (canal 9) intacta",
                [x for x in after if x[0] == 9] == [x for x in before if x[0] == 9] and st["drums"] == 3))
    res.append(("keyswitch intacto", (0, 21) in after and st["keyswitches"] == 1))
    ks = [m.key for tr in new.tracks for m in tr if m.type == "key_signature"]
    res.append(("key_signature Dm → Em", ks == ["Em"]))

    back, _ = transpose_midi(new, -2)
    res.append(("ida y vuelta restituye las notas", _pitch_list(back) == before))

    hi = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    tr.append(mido.Message("note_on", note=120, velocity=80, time=0))
    tr.append(mido.Message("note_off", note=120, velocity=0, time=480))
    hi.tracks.append(tr)
    try:
        transpose_midi(hi, 12)
        err = False
    except ValueError:
        err = True
    res.append(("desbordar 0..127 lanza error", err))
    folded, stf = transpose_midi(hi, 12, fold=True)
    res.append(("--fold pliega por octavas", folded.tracks[0][0].note == 120 and stf["folded"] == 1))

    res.append(("parse_key", parse_key("F#m") == (6, "min") and parse_key("Bb") == (10, "maj")
                and parse_key("D minor") == (2, "min")))
    res.append(("parse_range", parse_range("C3-C6") == (48, 84) and parse_range("48-84") == (48, 84)))

    ok = all(r for _n, r in res)
    for n, r in res:
        print(f"  {'✔' if r else '✘'} {n}")
    print("\n  SELFTEST OK" if ok else "\n  SELFTEST FALLA")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Transposición de un MIDI completo (sin percusión ni keyswitches)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info"); p.add_argument("midi"); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("shift"); p.add_argument("midi")
    p.add_argument("-o", "--output")
    p.add_argument("--tracks", type=int, nargs="+")
    p.add_argument("--exclude-channels", type=int, nargs="+", default=[DRUM_CH])
    p.add_argument("--fold", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("-s", "--semitones", type=int, required=True)
    p.set_defaults(fn=cmd_shift)
    p = sub.add_parser("to-key"); p.add_argument("midi")
    p.add_argument("-o", "--output")
    p.add_argument("--tracks", type=int, nargs="+")
    p.add_argument("--exclude-channels", type=int, nargs="+", default=[DRUM_CH])
    p.add_argument("--fold", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--key", required=True)
    p.add_argument("--from", dest="from_key")
    p.add_argument("--direction", default="nearest", choices=("nearest", "up", "down"))
    p.set_defaults(fn=cmd_to_key)
    p = sub.add_parser("fit"); p.add_argument("midi")
    p.add_argument("-o", "--output")
    p.add_argument("--tracks", type=int, nargs="+")
    p.add_argument("--exclude-channels", type=int, nargs="+", default=[DRUM_CH])
    p.add_argument("--fold", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--range", required=True)
    p.add_argument("--max-shift", type=int, default=12)
    p.add_argument("--strict", action="store_true")
    p.set_defaults(fn=cmd_fit)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)

    a = ap.parse_args()
    try:
        sys.exit(a.fn(a))
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e.filename}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
