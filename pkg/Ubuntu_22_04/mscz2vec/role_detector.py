#!/usr/bin/env python3
"""
role_detector.py — Role Detector: detector de roles de pista (melodía, contramelodía,
bajo, armonía, percusión) para MIDIs multipista
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
quality_scorer.py y midi2plan.py llevaban cada uno su propia heurística de roles
("melodía = pista con el pitch medio más alto") y ambas fallaban igual: una pista
decorativa de 6 notas en el agudo se llevaba la melodía y las líneas solistas reales
(viola, violín) acababan etiquetadas como "armonía". Esta herramienta centraliza el
cálculo, lo EXPLICA (qué rasgos pesan en cada decisión) y da una CONFIANZA por pista,
de modo que el resto del ecosistema pueda llamarla por subprocess (JSON) en lugar de
reimplementarla.

RASGOS POR PISTA (todos calculados del contenido, el nombre solo añade una pista):
  registro       pitch medio y rango; posición relativa entre las pistas
  actividad      notas respecto a la pista más activa
  linealidad     1 − fracción de ataques con otra nota sonando (1 = monofónica)
  suavidad       proporción de saltos ≤ 5 semitonos en la línea superior
  cobertura      fracción de compases con notas
  sostenimiento  duración media de nota (en pulsos)
  nombre / GM    palabras clave (bajo, bass, pad, cuerdas…) y programa GM

ROLES:  melody · countermelody · bass · harmony · percussion

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  detect    Roles de todas las pistas (tabla o --json); admite --roles-file
  explain   Rasgos y razones de una pista concreta (por índice o nombre)
  override  Fija roles a mano y los guarda en <midi>.roles.json
  selftest  Comprueba el detector con un MIDI sintético

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  detect    midi  [--json RUTA|-] [--roles-file F] [--no-color]
  explain   midi  PISTA  (índice 0-based de pista MIDI o parte del nombre)
  override  midi  --set N=ROL [N=ROL ...] [--roles-file F]
  selftest

EJEMPLOS
  python role_detector.py detect obra.mid
  python role_detector.py detect obra.mid --json -          # para otras herramientas
  python role_detector.py explain obra.mid viola
  python role_detector.py override obra.mid --set 1=melody 3=harmony
  python role_detector.py detect obra.mid                   # aplica obra.mid.roles.json

El fichero de roles es {"roles": {"<índice de pista>": "<rol>"}} y tiene prioridad
sobre la heurística (confianza 1.0).
"""
import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict

import mido
import numpy as np

def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    if isinstance(path, (str, bytes, os.PathLike)):
        d = os.path.dirname(os.path.abspath(os.fsdecode(path)))
        if d:
            os.makedirs(d, exist_ok=True)
    return path


VERSION = "1.0"
ROLES = ("melody", "countermelody", "bass", "harmony", "percussion")
BASS_PROGRAMS = set(range(32, 40))               # GM bajos
ENSEMBLE_PROGRAMS = set(range(48, 56)) | set(range(88, 96))   # cuerdas/coros/pads
DRUM_CH = 9

NAME_BASS = ("bajo", "bass", "contrabajo", "double bass", "tuba", "cb")
NAME_HARM = ("pad", "cuerdas", "strings", "ensemble", "sección", "seccion", "section",
             "coro", "choir", "tutti", "arpa", "harp", "piano", "organ", "órgano")
NAME_MEL = ("solista", "solo", "melod", "lead", "voz", "voice")
NAME_DRUM = ("perc", "drum", "bateria", "batería", "timbal", "caja")


def _c(code):
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "GRN": "\033[92m",
            "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}.get(code, "")


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA
# ══════════════════════════════════════════════════════════════════════════════

def load_tracks(path):
    """→ (tracks, tpb, ticks_per_bar). Cada pista: dict con notas (start,end,pitch,vel)."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    ts = (4, 4)
    tracks = []
    for idx, tr in enumerate(mid.tracks):
        t = 0
        open_ = defaultdict(list)
        notes, name, prog, chans = [], "", None, set()
        for m in tr:
            t += m.time
            if m.type == "track_name" and not name:
                name = m.name
            elif m.type == "time_signature" and not tracks and t == 0:
                ts = (m.numerator, m.denominator)
            elif m.type == "program_change" and prog is None:
                prog = m.program
            elif m.type == "note_on" and m.velocity > 0:
                open_[(m.channel, m.note)].append((t, m.velocity))
                chans.add(m.channel)
            elif m.type in ("note_off", "note_on"):
                q = open_.get((m.channel, m.note))
                if q:
                    s, v = q.pop(0)
                    notes.append((s, max(t, s + 1), m.note, v))
        notes.sort()
        tracks.append({"index": idx, "name": name, "program": prog,
                       "channels": sorted(chans), "notes": notes})
    tpbar = int(tpb * 4 * ts[0] / ts[1])
    return tracks, tpb, tpbar


def concurrency(notes):
    """(fracción de ataques con otra nota ya sonando, máximo simultáneo)."""
    ev = []
    for s, e, p, v in notes:
        ev.append((s, 1, e))
        ev.append((e, -1, e))
    ev.sort(key=lambda x: (x[0], x[1]))
    act, multi, ons, kmax = 0, 0, 0, 0
    for t, k, _ in ev:
        if k == 1:
            act += 1
            ons += 1
            kmax = max(kmax, act)
            if act > 1:
                multi += 1
        else:
            act -= 1
    return (multi / ons if ons else 0.0), kmax


def features(tr, tpb, tpbar, total_bars):
    notes = tr["notes"]
    n = len(notes)
    f = {"n_notes": n}
    if not n:
        return f
    pitches = np.array([x[2] for x in notes])
    chord, kmax = concurrency(notes)
    # línea superior: nota más aguda por ataque
    top = {}
    for s, e, p, v in notes:
        top[s] = max(top.get(s, 0), p)
    seq = [top[k] for k in sorted(top)]
    leaps = np.abs(np.diff(seq)) if len(seq) > 1 else np.array([0])
    bars = {s // tpbar for s, _, _, _ in notes}
    f.update({
        "mean_pitch": float(pitches.mean()),
        "pitch_min": int(pitches.min()), "pitch_max": int(pitches.max()),
        "chordiness": float(chord), "max_poly": int(kmax),
        "linearity": float(1.0 - chord),
        "smoothness": float((leaps <= 5).mean()),
        "coverage": float(len(bars) / max(1, total_bars)),
        "mean_dur_beats": float(np.mean([(e - s) / tpb for s, e, _, _ in notes])),
    })
    return f


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def detect(path, forced=None):
    tracks, tpb, tpbar = load_tracks(path)
    last = max((x[1] for t in tracks for x in t["notes"]), default=0)
    total_bars = max(1, -(-last // tpbar))
    forced = forced or {}
    feats = {t["index"]: features(t, tpb, tpbar, total_bars) for t in tracks}
    live = [t for t in tracks if t["notes"]]
    out = {}

    def setrole(i, role, conf, reasons):
        out[i] = {"role": role, "confidence": round(float(conf), 2), "reasons": reasons}

    # 0 · forzados
    for i, r in forced.items():
        if i in feats and feats[i]["n_notes"]:
            setrole(i, r, 1.0, ["rol fijado manualmente"])

    # 1 · percusión
    for t in live:
        i = t["index"]
        if i in out:
            continue
        low = t["name"].lower()
        if t["channels"] and set(t["channels"]) == {DRUM_CH}:
            setrole(i, "percussion", 0.98, ["canal 10 (GM percusión)"])
        elif any(k in low for k in NAME_DRUM) and feats[i]["mean_pitch"] < 90 \
                and feats[i]["max_poly"] >= 1 and t["program"] in (None, 0) \
                and DRUM_CH in t["channels"]:
            setrole(i, "percussion", 0.9, ["nombre + canal de percusión"])

    pool = [t for t in live if t["index"] not in out]
    if pool:
        max_n = max(feats[t["index"]]["n_notes"] for t in pool)
        pm = [feats[t["index"]]["mean_pitch"] for t in pool]
        lo, hi = min(pm), max(pm)

        def rel_reg(i):
            return (feats[i]["mean_pitch"] - lo) / (hi - lo) if hi > lo else 0.5

        # 2 · bajo: registro grave absoluto (o programa/nombre de bajo) y no acórdico
        bass_score = {}
        for t in pool:
            i, f, low = t["index"], feats[t["index"]], t["name"].lower()
            s, why = 0.0, []
            if f["mean_pitch"] <= 52:
                s += 0.5 * min(1.0, (52 - f["mean_pitch"]) / 16 + 0.4)
                why.append(f"registro grave (pitch medio {f['mean_pitch']:.0f})")
            if t["program"] in BASS_PROGRAMS:
                s += 0.3
                why.append("programa GM de bajo")
            if any(k in low.split() or k == low for k in NAME_BASS) or "contrabajo" in low \
                    or "bass" in low:
                s += 0.2
                why.append("nombre de bajo")
            s += 0.2 * f["linearity"]
            bass_score[i] = (s, why)
        bass = [i for i, (s, w) in bass_score.items()
                if s >= 0.55 and feats[i]["mean_pitch"] <= 55]
        for i in bass:
            s, why = bass_score[i]
            setrole(i, "bass", min(0.95, s), why)

        # 3 · melodía / contramelodía / armonía entre el resto
        rest = [t for t in pool if t["index"] not in out]
        mel_score = {}
        for t in rest:
            i, f, low = t["index"], feats[t["index"]], t["name"].lower()
            reg = rel_reg(i)
            act = f["n_notes"] / max_n
            s = (0.25 * reg + 0.25 * min(1.0, act * 2) + 0.20 * f["linearity"]
                 + 0.15 * f["coverage"] + 0.15 * f["smoothness"])
            why = [f"registro relativo {reg:.2f}", f"actividad {act:.2f}",
                   f"linealidad {f['linearity']:.2f}", f"cobertura {f['coverage']:.2f}"]
            if f["n_notes"] < 0.08 * max_n:
                s *= 0.4
                why.append("muy pocas notas (<8 % de la pista más activa)")
            if any(k in low for k in NAME_MEL):
                s += 0.08
                why.append("nombre de solista/melodía")
            if any(k in low for k in NAME_HARM) or t["program"] in ENSEMBLE_PROGRAMS:
                s -= 0.12
                why.append("nombre/programa de sección-armonía")
            if f["chordiness"] > 0.5:
                s -= 0.1
                why.append(f"muy acórdica ({f['chordiness']:.0%})")
            mel_score[i] = (s, why)
        if mel_score:
            ranked = sorted(mel_score, key=lambda i: -mel_score[i][0])
            best = ranked[0]
            best_s = mel_score[best][0]
            second_s = mel_score[ranked[1]][0] if len(ranked) > 1 else 0.0
            conf = max(0.2, min(0.95, 0.5 + (best_s - second_s) * 2))
            setrole(best, "melody", conf, mel_score[best][1])
            for i in ranked[1:]:
                s, why = mel_score[i]
                f = feats[i]
                if s >= 0.70 * best_s and f["linearity"] >= 0.7 and f["n_notes"] >= 0.2 * max_n:
                    setrole(i, "countermelody", 0.5 + min(0.3, s - 0.5 * best_s), why)
                else:
                    hs = 0.5 + 0.3 * f["chordiness"] + 0.2 * min(1, f["mean_dur_beats"] / 3)
                    setrole(i, "harmony", min(0.9, hs),
                            [f"acórdica {f['chordiness']:.0%}",
                             f"dur. media {f['mean_dur_beats']:.1f} pulsos",
                             f"puntuación de melodía baja ({s:.2f} < {best_s:.2f})"])
    for t in tracks:
        if t["index"] not in out and not t["notes"]:
            setrole(t["index"], "harmony", 0.0, ["pista sin notas"])
    res = []
    for t in tracks:
        i = t["index"]
        res.append({"index": i, "name": t["name"], "n_notes": feats[i]["n_notes"],
                    **out[i], "features": {k: (round(v, 3) if isinstance(v, float) else v)
                                           for k, v in feats[i].items()}})
    return {"tool": "role_detector", "version": VERSION, "file": os.path.basename(path),
            "bars": total_bars, "tracks": res}


def roles_path(midi, given=None):
    return given or midi + ".roles.json"


def read_forced(path):
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return {int(k): v for k, v in d.get("roles", {}).items() if v in ROLES}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_detect(a):
    forced = read_forced(roles_path(a.midi, a.roles_file))
    rep = detect(a.midi, forced)
    if a.json:
        txt = json.dumps(rep, ensure_ascii=False, indent=1)
        if a.json == "-":
            print(txt)
            return 0
        open(_ensure_parent(a.json), "w", encoding="utf-8").write(txt)
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n  {B}ROLE DETECTOR v{VERSION} · {rep['file']}{R}   ({rep['bars']} compases)")
    if forced:
        print(f"  {G}roles fijados a mano: {forced}{R}")
    print(f"  {G}{'#':>2} {'pista':<26}{'notas':>6}  {'rol':<14}{'conf':>5}{R}")
    low = []
    for t in rep["tracks"]:
        col = _c("GRN") if t["confidence"] >= 0.7 else _c("YEL") if t["confidence"] >= 0.45 else _c("RED")
        print(f"  {t['index']:>2} {t['name'][:25]:<26}{t['n_notes']:>6}  {t['role']:<14}{col}{t['confidence']:>5.2f}{R}")
        if t["confidence"] < 0.45 and t["n_notes"]:
            low.append(t["index"])
    mels = [t for t in rep["tracks"] if t["role"] == "melody"]
    if not mels:
        print(f"\n  {_c('YEL')}⚠ ninguna pista asignada como melodía{R}")
    if low:
        print(f"\n  {_c('YEL')}⚠ confianza baja en las pistas {low}: use 'explain' o 'override'{R}")
    print()
    return 0


def _find_track(rep, key):
    if key.isdigit():
        for t in rep["tracks"]:
            if t["index"] == int(key):
                return t
    k = key.lower()
    for t in rep["tracks"]:
        if k in t["name"].lower():
            return t
    return None


def cmd_explain(a):
    rep = detect(a.midi, read_forced(roles_path(a.midi, a.roles_file)))
    t = _find_track(rep, a.pista)
    if not t:
        print(f"[ERROR] pista '{a.pista}' no encontrada", file=sys.stderr)
        return 2
    print(f"\n  Pista {t['index']} «{t['name']}» → {_c('B')}{t['role']}{_c('R')} (confianza {t['confidence']:.2f})")
    for r in t["reasons"]:
        print(f"    · {r}")
    print("  Rasgos:")
    for k, v in t["features"].items():
        print(f"    {k:<16} {v}")
    print()
    return 0


def cmd_override(a):
    p = roles_path(a.midi, a.roles_file)
    cur = {}
    if os.path.exists(p):
        cur = json.load(open(p, encoding="utf-8")).get("roles", {})
    for kv in a.set:
        if "=" not in kv:
            print(f"[ERROR] '{kv}': use N=ROL", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        if not k.isdigit() or v not in ROLES:
            print(f"[ERROR] '{kv}': índice entero y rol en {ROLES}", file=sys.stderr)
            return 2
        cur[k] = v
    json.dump({"roles": cur}, open(_ensure_parent(p), "w", encoding="utf-8"), indent=1)
    print(f"  ✓ {p}: {cur}")
    return 0


def _synthetic(path):
    """Melodía solista (viola), bajo, acordes sostenidos y un adorno de 4 notas agudas."""
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    mid.tracks.append(meta)

    def mk(name, prog, chan, ev):
        t = mido.MidiTrack()
        t.append(mido.MetaMessage("track_name", name=name))
        t.append(mido.Message("program_change", program=prog, channel=chan))
        items = []
        for s, d, p in ev:
            items += [(s, 1, p), (s + d, 0, p)]
        items.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for tt, on, p in items:
            t.append(mido.Message("note_on", note=p, velocity=80 if on else 0, channel=chan,
                                  time=tt - last))
            last = tt
        mid.tracks.append(t)

    B = 1920
    mel = [(b * B + k * 480, 480, 62 + (k * 2 + b) % 7) for b in range(16) for k in range(4)]
    bas = [(b * B, B, 38 + (b % 4)) for b in range(16)]
    chords = [(b * B, B, p) for b in range(16) for p in (50, 54, 57)]
    deco = [(b * B * 4, 480, 90 + b) for b in range(4)]
    mk("Viola solista", 41, 0, mel)
    mk("Contrabajo", 43, 1, bas)
    mk("Cuerdas", 48, 2, chords)
    mk("Adorno", 40, 3, deco)
    mid.save(_ensure_parent(path))


def cmd_selftest(a):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "s.mid")
    _synthetic(p)
    rep = detect(p)
    roles = {t["name"]: t["role"] for t in rep["tracks"] if t["n_notes"]}
    exp = {"Viola solista": "melody", "Contrabajo": "bass", "Cuerdas": "harmony"}
    ok = all(roles.get(k) == v for k, v in exp.items()) and roles.get("Adorno") != "melody"
    print(f"  roles: {roles}")
    rep2 = detect(p, {4: "melody"})
    ok2 = {t["name"]: t["role"] for t in rep2["tracks"]}["Adorno"] == "melody"
    print(f"  {'✓' if ok else '✗'} detección heurística   {'✓' if ok2 else '✗'} override")
    return 0 if ok and ok2 else 1


def main():
    ap = argparse.ArgumentParser(description="Detector de roles de pista en MIDIs multipista")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("detect")
    p.add_argument("midi")
    p.add_argument("--json")
    p.add_argument("--roles-file")
    p.set_defaults(fn=cmd_detect)
    p = sub.add_parser("explain")
    p.add_argument("midi")
    p.add_argument("pista")
    p.add_argument("--roles-file")
    p.set_defaults(fn=cmd_explain)
    p = sub.add_parser("override")
    p.add_argument("midi")
    p.add_argument("--set", nargs="+", required=True)
    p.add_argument("--roles-file")
    p.set_defaults(fn=cmd_override)
    p = sub.add_parser("selftest")
    p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    try:
        sys.exit(a.fn(a))
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e.filename}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
