#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ORCHESTRAL COLORIST  v1.0                                ║
║              Color tímbrico emocional para obras MIDI                       ║
║                                                                              ║
║  orchestrator asigna instrumentos de forma mecánica/idiomática; esta        ║
║  herramienta lo hace por SEMÁNTICA EMOCIONAL: cada emoción define una       ║
║  paleta (instrumento por rol, registro diana y rango dinámico) extraída     ║
║  de la práctica orquestal — melancolía → violonchelo en registro medio,     ║
║  fragilidad → flauta sola pp con arpa, angustia → registros extremos y      ║
║  trémolo de cuerda, triunfo → metales ff...                                 ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] ROLES      — detección por pista: melodía / bajo / armonía /           ║
║                   contrapunto / percusión (canal 10)                         ║
║  [2] PALETA     — instrumento GM + registro + dinámica según --emotion      ║
║  [3] APLICACIÓN — program change, transposición de octavas hacia el         ║
║                   registro diana DENTRO del rango físico (catálogo de       ║
║                   playability_auditor) y reescalado de velocidades          ║
║  [4] INFORME    — tabla de asignaciones + escritura                          ║
║                                                                              ║
║  EMOCIONES: melancolia · calidez · fragilidad · angustia · esperanza        ║
║             serenidad · tension · triunfo                                    ║
║                                                                              ║
║  USO:                                                                        ║
║    python orchestral_colorist.py obra.mid --emotion melancolia              ║
║    python orchestral_colorist.py obra.mid --emotion triunfo --output t.mid  ║
║    python orchestral_colorist.py obra.mid --emotion angustia --keep 0       ║
║    python orchestral_colorist.py --list-emotions                            ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi               MIDI de entrada                                        ║
║    --emotion E        Emoción diana (obligatoria salvo --list-emotions)     ║
║    --output FILE      MIDI de salida (default: <obra>_<emocion>.mid)        ║
║    --keep N...        Pistas a no tocar (índices)                            ║
║    --no-register      No transponer hacia el registro diana                  ║
║    --no-dynamics      No reescalar velocidades                               ║
║    --report           Solo informe de roles y paleta, sin escribir MIDI      ║
║    --list-emotions    Catálogo de paletas y salir                            ║
║    --no-color         Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from orchestral_colorist import colorize_midi, EMOTION_PALETTES          ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio,             ║
║                E/S MIDI + catálogo de rangos)                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    from playability_auditor import (read_midi, write_midi, MidiEvent,
                                     extract_notes, pitch_name, INSTRUMENT_DB,
                                     _write_vlq)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PALETAS EMOCIONALES
#  rol → (perfil_de_playability_auditor, program_GM, centro_de_registro,
#         (vel_min, vel_max), nombre_para_la_pista)
# ══════════════════════════════════════════════════════════════════════════════

EMOTION_PALETTES = {
    "melancolia": {
        "desc": "violonchelo cantabile, corno inglés, cuerdas graves, piano",
        "melodia":     ("violonchelo", 42, 55, (42, 72), "Violonchelo solo"),
        "contrapunto": ("corno_ingles", 69, 64, (38, 64), "Corno inglés"),
        "armonia":     ("cuerdas", 48, 52, (30, 56), "Cuerdas graves"),
        "bajo":        ("contrabajo", 43, 38, (34, 60), "Contrabajo"),
    },
    "calidez": {
        "desc": "trompa lírica, clarinete, colchón de cuerdas",
        "melodia":     ("trompa", 60, 60, (52, 84), "Trompa"),
        "contrapunto": ("clarinete", 71, 62, (46, 74), "Clarinete"),
        "armonia":     ("cuerdas", 48, 55, (40, 66), "Cuerdas"),
        "bajo":        ("violonchelo", 42, 43, (44, 70), "Violonchelos"),
    },
    "fragilidad": {
        "desc": "flauta sola pp, arpa, pizzicati apenas audibles",
        "melodia":     ("flauta", 73, 74, (24, 50), "Flauta sola"),
        "contrapunto": ("celesta", 8, 79, (20, 42), "Celesta"),
        "armonia":     ("arpa", 46, 62, (22, 46), "Arpa"),
        "bajo":        ("violonchelo", 45, 45, (20, 44), "Pizzicato"),
    },
    "angustia": {
        "desc": "violín agudo, trémolo de cuerda, metales graves",
        "melodia":     ("violin", 40, 84, (72, 112), "Violín agudo"),
        "contrapunto": ("trombon", 57, 50, (66, 100), "Trombón"),
        "armonia":     ("cuerdas", 44, 64, (58, 92), "Cuerdas (trémolo)"),
        "bajo":        ("contrabajo", 43, 33, (70, 104), "Contrabajos"),
    },
    "esperanza": {
        "desc": "oboe luminoso, flauta en contrapunto, cuerdas medias",
        "melodia":     ("oboe", 68, 72, (52, 84), "Oboe"),
        "contrapunto": ("flauta", 73, 79, (46, 76), "Flauta"),
        "armonia":     ("cuerdas", 48, 58, (42, 68), "Cuerdas"),
        "bajo":        ("violonchelo", 42, 45, (44, 70), "Violonchelos"),
    },
    "serenidad": {
        "desc": "clarinete chalumeau, arpa, graves en pp",
        "melodia":     ("clarinete", 71, 58, (34, 58), "Clarinete (chalumeau)"),
        "contrapunto": ("flauta", 73, 70, (28, 50), "Flauta"),
        "armonia":     ("arpa", 46, 60, (26, 48), "Arpa"),
        "bajo":        ("contrabajo", 43, 36, (26, 50), "Contrabajo"),
    },
    "tension": {
        "desc": "trompeta incisiva, cuerdas altas, tuba como amenaza",
        "melodia":     ("trompeta", 56, 74, (66, 100), "Trompeta"),
        "contrapunto": ("trompa", 60, 58, (60, 92), "Trompas"),
        "armonia":     ("cuerdas", 48, 70, (56, 88), "Cuerdas altas"),
        "bajo":        ("tuba", 58, 38, (62, 96), "Tuba"),
    },
    "triunfo": {
        "desc": "metales plenos ff, percusión, cuerdas en bloque",
        "melodia":     ("trompeta", 56, 72, (88, 120), "Trompetas"),
        "contrapunto": ("trompa", 60, 62, (82, 112), "Trompas"),
        "armonia":     ("viento_metal_gen", 61, 58, (78, 108), "Metales"),
        "bajo":        ("trombon", 57, 46, (84, 116), "Trombones y tuba"),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE ROLES
# ══════════════════════════════════════════════════════════════════════════════

def detect_roles(tracks_notes, mid) -> dict:
    """idx → rol (melodia | bajo | armonia | contrapunto | percusion)."""
    roles = {}
    stats = {}
    for idx, ns in tracks_notes.items():
        if 9 in mid.tracks[idx].channels:
            roles[idx] = "percusion"
            continue
        mean_p = sum(n.pitch for n in ns) / len(ns)
        total = sum(n.end - n.start for n in ns)
        span = max(n.end for n in ns) - min(n.start for n in ns) or 1
        poly = total / span
        stats[idx] = (mean_p, poly)
    rest = list(stats)
    if rest:
        bass = min(rest, key=lambda i: stats[i][0])
        roles[bass] = "bajo"
        rest.remove(bass)
    if rest:
        mel = max(rest, key=lambda i: stats[i][0])
        roles[mel] = "melodia"
        rest.remove(mel)
    if rest:
        harm = max(rest, key=lambda i: stats[i][1])
        roles[harm] = "armonia"
        rest.remove(harm)
    for i in rest:
        roles[i] = "contrapunto"
    return roles


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _set_program(trk, channel, program):
    trk.events = [e for e in trk.events
                  if not (e.kind == "channel" and e.data
                          and (e.data[0] & 0xF0) == 0xC0)]
    trk.events.insert(0, MidiEvent(abs=0, kind="channel", channel=channel,
                                   data=bytes([0xC0 | channel, program])))
    trk.programs = [program]


def _set_name(trk, name):
    payload = name.encode("latin-1", errors="replace")
    ev = MidiEvent(abs=0, kind="meta", meta_type=0x03,
                   data=bytes([0xFF, 0x03]) + _write_vlq(len(payload)) + payload)
    trk.events = [e for e in trk.events if e.meta_type != 0x03]
    trk.events.insert(0, ev)
    trk.name = name


def _rebuild(trk, notes):
    others = [e for e in trk.events if e.kind not in ("note_on", "note_off")]
    evs = list(others)
    for n in notes:
        evs.append(MidiEvent(abs=n.start, kind="note_on", channel=n.channel,
                             pitch=n.pitch, vel=n.vel))
        evs.append(MidiEvent(abs=n.end, kind="note_off", channel=n.channel,
                             pitch=n.pitch, vel=0))
    trk.events = sorted(evs, key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))


def colorize_midi(path: str, emotion: str, out_path: Optional[str] = None,
                  keep: Optional[List[int]] = None, register: bool = True,
                  dynamics: bool = True, report_only: bool = False) -> dict:
    """API pública. Devuelve la tabla de asignaciones aplicada."""
    if emotion not in EMOTION_PALETTES:
        raise ValueError(f"emoción desconocida: {emotion} "
                         f"(usa --list-emotions)")
    palette = EMOTION_PALETTES[emotion]
    keep = set(keep or [])
    mid = read_midi(path)
    tracks_notes = {i: extract_notes(t) for i, t in enumerate(mid.tracks)}
    tracks_notes = {i: ns for i, ns in tracks_notes.items() if ns}
    if not tracks_notes:
        raise ValueError("el MIDI no contiene notas")
    roles = detect_roles(tracks_notes, mid)

    table = []
    for idx, ns in tracks_notes.items():
        role = roles[idx]
        if idx in keep or role == "percusion" or role not in palette:
            table.append({"track": idx, "old": mid.tracks[idx].name or f"Pista {idx}",
                          "role": role, "new": "(sin cambios)", "shift": 0})
            continue
        prof_key, program, center, (vlo, vhi), new_name = palette[role]
        prof = INSTRUMENT_DB[prof_key]
        lo, hi = prof["range"]

        shift = 0
        if register:
            mean_p = sum(n.pitch for n in ns) / len(ns)
            shift = int(round((center - mean_p) / 12)) * 12
        vels = [n.vel for n in ns]
        vmin, vmax = min(vels), max(vels)
        for n in ns:
            p = n.pitch + shift
            while p < lo:
                p += 12
            while p > hi:
                p -= 12
            n.pitch = p
            if dynamics:
                x = (n.vel - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                n.vel = max(1, min(127, int(round(vlo + x * (vhi - vlo)))))
        table.append({"track": idx, "old": mid.tracks[idx].name or f"Pista {idx}",
                      "role": role, "new": new_name, "program": program,
                      "profile": prof_key, "shift": shift,
                      "dyn": (vlo, vhi)})
        if not report_only:
            ch = ns[0].channel
            _set_program(mid.tracks[idx], ch, program)
            _set_name(mid.tracks[idx], new_name)
            _rebuild(mid.tracks[idx], ns)

    out = None
    if not report_only:
        out = out_path or str(Path(path).with_name(
            f"{Path(path).stem}_{emotion}.mid"))
        write_midi(mid, out)
    return {"emotion": emotion, "desc": palette["desc"],
            "table": table, "output": out}


def list_emotions():
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{B}Catálogo de paletas emocionales{R}")
    print(f"{'─' * 78}")
    for emo, pal in EMOTION_PALETTES.items():
        print(f"\n  {B}{emo}{R} — {pal['desc']}")
        for role in ("melodia", "contrapunto", "armonia", "bajo"):
            prof_key, prog, center, dyn, name = pal[role]
            print(f"    {role:<12} {name:<24} GM {prog:<3}  "
                  f"registro ≈ {pitch_name(center):<4}  vel {dyn[0]}–{dyn[1]}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="orchestral_colorist.py",
                                 description="Color tímbrico emocional para MIDIs.")
    ap.add_argument("midi", nargs="?")
    ap.add_argument("--emotion", choices=sorted(EMOTION_PALETTES))
    ap.add_argument("--output")
    ap.add_argument("--keep", type=int, nargs="+", default=[])
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--list-emotions", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False
    if args.list_emotions:
        list_emotions()
        return 0
    if not args.midi or not args.emotion:
        ap.error("indica MIDI y --emotion (o usa --list-emotions)")

    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}")
    print(f"  {B}ORCHESTRAL COLORIST v{VERSION}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")
    try:
        res = colorize_midi(args.midi, args.emotion, args.output,
                            keep=args.keep, register=not args.no_register,
                            dynamics=not args.no_dynamics,
                            report_only=args.report)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Emoción: {B}{res['emotion']}{R} — {res['desc']}\n")
    print(f"  {'pista':<7}{'antes':<22}{'rol':<14}{'después':<26}{'8ªs':<5}dinámica")
    for row in res["table"]:
        shift = f"{row['shift'] // 12:+d}" if row["shift"] else "—"
        dyn = f"{row['dyn'][0]}–{row['dyn'][1]}" if "dyn" in row else "—"
        print(f"  [{row['track']}]{'':<4}{row['old']:<22}{row['role']:<14}"
              f"{row['new']:<26}{shift:<5}{dyn}")
    if res["output"]:
        print(f"\n  {_c('GRN')}MIDI coloreado: {res['output']}{R}")
    else:
        print(f"\n  {G}(solo informe){R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
