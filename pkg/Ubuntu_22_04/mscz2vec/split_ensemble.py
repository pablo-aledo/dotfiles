#!/usr/bin/env python3
"""
split_ensemble.py — Split Ensemble: divide una pista de sección (acordes) en voces
monofónicas con conducción de voces
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Una "sección" (cuerdas, metales, coro) suele salir de las herramientas como UNA pista
con acordes de 3-4 notas. Para importarla en un DAW con una librería de samples, para
generar partes individuales (part_extractor) o para que playability_auditor la trate
como instrumentos reales, hace falta repartirla en VOCES monofónicas: Voz 1 (más aguda)
… Voz N. Ninguna herramienta lo hacía; part_extractor separa por pista, no por voz.

ALGORITMO (barrido temporal, notas y duraciones se conservan):
  [1] Se recorren las notas por ataque. En cada ataque se resuelve la asignación de las
      notas nuevas a las voces LIBRES minimizando el salto de registro respecto a la
      última nota de cada voz (coste por permutaciones; sin dependencias externas).
  [2] Las voces conservan su orden por registro cuando la textura es de acordes
      (la nota más aguda de un ataque va a la voz más aguda de las libres).
  [3] Si hay más notas simultáneas que voces (--voices menor que el máximo), la voz más
      cercana CORTA su nota anterior (se informa de cuántas).
  [4] Las notas que ya suenan no se tocan: una voz ocupada no recibe notas nuevas.

SALIDA: un MIDI con una pista por voz ("<nombre> · V1", "· V2"…), mismo canal y
programa que la pista original, con tempo/compás copiados. Con --programs se puede
asignar un programa GM distinto a cada voz (p. ej. 40 41 42 43 → Vln, Vln, Vla, Vc).

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  info      Notas simultáneas máximas y voces necesarias por pista
  split     Divide una pista (o todas las polifónicas con --all) en voces
  check     Verifica un MIDI dividido: cada pista monofónica y notas conservadas
  selftest  Prueba con acordes sintéticos

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  info   midi
  split  midi  [--track N|NOMBRE | --all] [--voices N] [--programs P ...]
               [--keep-others] [-o SALIDA.mid]
  check  original.mid dividido.mid [--track N|NOMBRE]
  selftest

EJEMPLOS
  python split_ensemble.py info obra.mid
  python split_ensemble.py split obra.mid --track "Cuerdas" --voices 4 -o cuerdas_4v.mid
  python split_ensemble.py split obra.mid --all --keep-others -o obra_voces.mid
  python split_ensemble.py check obra.mid obra_voces.mid --track Cuerdas
"""
import argparse
import itertools
import os
import sys
import tempfile
from collections import defaultdict

import mido

def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    if isinstance(path, (str, bytes, os.PathLike)):
        d = os.path.dirname(os.path.abspath(os.fsdecode(path)))
        if d:
            os.makedirs(d, exist_ok=True)
    return path


VERSION = "1.0"


def _c(code):
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "GRN": "\033[92m",
            "YEL": "\033[93m", "RED": "\033[91m"}.get(code, "")


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA
# ══════════════════════════════════════════════════════════════════════════════

def read_tracks(path):
    mid = mido.MidiFile(path)
    out = []
    for idx, tr in enumerate(mid.tracks):
        t, open_, notes, name, prog, chan = 0, defaultdict(list), [], "", None, None
        for m in tr:
            t += m.time
            if m.type == "track_name" and not name:
                name = m.name
            elif m.type == "program_change" and prog is None:
                prog, chan = m.program, m.channel
            elif m.type == "note_on" and m.velocity > 0:
                open_[(m.channel, m.note)].append((t, m.velocity))
                chan = m.channel if chan is None else chan
            elif m.type in ("note_off", "note_on"):
                q = open_.get((m.channel, m.note))
                if q:
                    s, v = q.pop(0)
                    notes.append([s, max(t, s + 1), m.note, v])
        notes.sort(key=lambda n: (n[0], -n[2]))
        out.append({"index": idx, "name": name, "program": prog, "channel": chan or 0,
                    "notes": notes})
    return mid, out


def max_poly(notes):
    ev = []
    for s, e, *_ in notes:
        ev += [(s, 1), (e, -1)]
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, k in ev:
        cur += k
        best = max(best, cur)
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  DIVISIÓN EN VOCES
# ══════════════════════════════════════════════════════════════════════════════

def split_notes(notes, n_voices):
    """→ (voces, recortadas, omitidas). Cada voz: lista de [start,end,pitch,vel].
    recortadas = notas previas cortadas para dejar sitio; omitidas = notas nuevas que
    no caben (más simultáneas que voces y ninguna voz que robar: se descarta la más grave)."""
    voices = [[] for _ in range(n_voices)]
    last_pitch = [None] * n_voices
    trimmed = dropped = 0
    by_start = defaultdict(list)
    for n in notes:
        by_start[n[0]].append([n[0], n[1], n[2], n[3]])
    for start in sorted(by_start):
        new = sorted(by_start[start], key=lambda n: -n[2])        # aguda → grave
        # voces libres (su última nota ya terminó)
        free = [i for i in range(n_voices)
                if not voices[i] or voices[i][-1][1] <= start]
        if len(new) > len(free):
            # robar voces: se corta la que suena con el pitch más cercano
            busy = [i for i in range(n_voices) if i not in free]
            need = len(new) - len(free)
            ref = sorted(p[2] for p in new)
            busy.sort(key=lambda i: min(abs(voices[i][-1][2] - r) for r in ref))
            for i in busy[:need]:
                if voices[i][-1][1] > start:
                    voices[i][-1][1] = start
                    if voices[i][-1][1] <= voices[i][-1][0]:
                        voices[i].pop()
                    trimmed += 1
                free.append(i)
        free.sort()
        if len(new) > len(free):
            dropped += len(new) - len(free)
            new = new[:len(free)]
        # asignación de coste mínimo: notas nuevas → voces libres
        best, assign = None, None
        cand = free if len(free) <= 6 else free[:6]
        for combo in itertools.permutations(cand, len(new)):
            if any(i > j for i, j in zip(combo, combo[1:])) and len(new) > 1:
                order_pen = 3.0       # preferir el orden agudo→grave de las voces
            else:
                order_pen = 0.0
            cost = order_pen
            for note, v in zip(new, combo):
                lp = last_pitch[v]
                cost += 0 if lp is None else abs(note[2] - lp)
                if lp is None:                          # primera nota: afinidad de registro
                    cost += abs(v - (len(cand) - 1) * 0) * 0.01
            if best is None or cost < best:
                best, assign = cost, combo
        for note, v in zip(new, assign):
            voices[v].append(note)
            last_pitch[v] = note[2]
    return voices, trimmed, dropped


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA
# ══════════════════════════════════════════════════════════════════════════════

def make_track(name, program, channel, notes):
    t = mido.MidiTrack()
    t.append(mido.MetaMessage("track_name", name=name))
    if program is not None:
        t.append(mido.Message("program_change", program=program, channel=channel, time=0))
    ev = []
    for s, e, p, v in notes:
        ev += [(s, 1, p, v), (e, 0, p, 0)]
    ev.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for tt, on, p, v in ev:
        t.append(mido.Message("note_on", note=p, velocity=v if on else 0,
                              channel=channel, time=tt - last))
        last = tt
    return t


def pick_tracks(tracks, sel):
    if sel is None:
        return []
    if sel.isdigit():
        return [t for t in tracks if t["index"] == int(sel)]
    return [t for t in tracks if sel.lower() in t["name"].lower() and t["notes"]]


def cmd_info(a):
    _, tracks = read_tracks(a.midi)
    print(f"\n  {_c('B')}SPLIT ENSEMBLE v{VERSION} · info · {os.path.basename(a.midi)}{_c('R')}")
    print(f"  {_c('G')}{'#':>2} {'pista':<28}{'notas':>6}{'máx. simult.':>14}  voces necesarias{_c('R')}")
    for t in tracks:
        if not t["notes"]:
            continue
        k = max_poly(t["notes"])
        tag = f"{_c('YEL')}{k} (dividir){_c('R')}" if k > 1 else f"{_c('GRN')}1 (ya monofónica){_c('R')}"
        print(f"  {t['index']:>2} {t['name'][:27]:<28}{len(t['notes']):>6}{k:>14}  {tag}")
    print()
    return 0


def cmd_split(a):
    mid, tracks = read_tracks(a.midi)
    if a.all:
        sel = [t for t in tracks if t["notes"] and max_poly(t["notes"]) > 1]
    else:
        sel = pick_tracks(tracks, a.track)
    if not sel:
        print("[ERROR] ninguna pista seleccionada (use --track N|NOMBRE o --all)", file=sys.stderr)
        return 2
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=1)
    # pista 0: meta global (la del original, que contiene tempo/compás) sin notas
    seln = {t["index"] for t in sel}
    for t in tracks:
        if t["index"] in seln:
            continue
        if a.keep_others or not t["notes"]:
            out.tracks.append(mid.tracks[t["index"]])
    total_trim = 0
    for t in sel:
        k = a.voices or max_poly(t["notes"])
        voices, trimmed, dropped = split_notes(t["notes"], k)
        total_trim += trimmed
        print(f"  {t['name'] or t['index']}: {len(t['notes'])} notas → {k} voces "
              f"({', '.join(str(len(v)) for v in voices)} notas)"
              + (f"  {_c('YEL')}[{trimmed} recortada(s)]{_c('R')}" if trimmed else "")
              + (f"  {_c('RED')}[{dropped} OMITIDA(s): faltan voces, use --voices {max_poly(t['notes'])}]{_c('R')}" if dropped else ""))
        for i, v in enumerate(voices):
            prog = a.programs[i] if a.programs and i < len(a.programs) else t["program"]
            out.tracks.append(make_track(f"{t['name']} · V{i + 1}", prog, t["channel"], v))
    path = a.output or os.path.splitext(a.midi)[0] + "_voces.mid"
    out.save(_ensure_parent(path))
    print(f"  → {path}")
    return 0


def cmd_check(a):
    _, orig = read_tracks(a.original)
    _, new = read_tracks(a.split)
    sel = pick_tracks(orig, a.track) if a.track else [t for t in orig if t["notes"]]
    ok = True
    for t in sel:
        voices = [v for v in new if v["name"].startswith(t["name"] + " · V")]
        if not voices:
            continue
        mono = all(max_poly(v["notes"]) <= 1 for v in voices)
        n_in = len(t["notes"])
        n_out = sum(len(v["notes"]) for v in voices)
        p_in = sorted((n[0], n[2]) for n in t["notes"])
        p_out = sorted((n[0], n[2]) for v in voices for n in v["notes"])
        same = p_in == p_out
        print(f"  {t['name']}: {len(voices)} voces · monofónicas {'✓' if mono else '✗'} · "
              f"notas {n_in}→{n_out} · ataques/pitches idénticos {'✓' if same else '✗ (hay recortes/omisiones)'}")
        ok &= mono
    return 0 if ok else 1


def cmd_selftest(a):
    d = tempfile.mkdtemp()
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=500000))
    mid.tracks.append(meta)
    chords = [[62, 65, 69, 74], [62, 65, 70, 74], [60, 64, 67, 72], [59, 62, 67, 71]]
    notes = []
    for i, c in enumerate(chords * 2):
        for p in c:
            notes.append([i * 1920, i * 1920 + 1800, p, 80])
    # nota larga solapada con el siguiente acorde (textura no homorrítmica)
    notes.append([1920 * 8, 1920 * 8 + 3000, 50, 70])
    notes.append([1920 * 8 + 960, 1920 * 8 + 1800, 57, 70])
    mid.tracks.append(make_track("Cuerdas", 48, 0, notes))
    src = os.path.join(d, "s.mid")
    mid.save(src)
    res = []
    voices, trimmed, dropped = split_notes(notes, max_poly(notes))
    res.append(("máx. simultáneas = 4", max_poly(notes) == 4))
    res.append(("todas las voces monofónicas", all(max_poly(v) <= 1 for v in voices)))
    res.append(("no se pierde ninguna nota", sum(len(v) for v in voices) == len(notes) and trimmed == 0))
    top = voices[0]
    res.append(("V1 = línea aguda", all(n[2] >= 71 for n in top[:8])))
    v2, tr2, dr2 = split_notes(notes, 3)
    res.append(("con 3 voces: recorta/omite, avisa y sigue monofónico",
                (tr2 + dr2) > 0 and all(max_poly(v) <= 1 for v in v2)))
    # CLI end-to-end
    class A:  # noqa
        pass
    a2 = A()
    a2.midi, a2.all, a2.track, a2.voices, a2.programs = src, False, "cuerdas", None, [40, 41, 42, 43]
    a2.keep_others, a2.output = False, os.path.join(d, "o.mid")
    rc = cmd_split(a2)
    a3 = A()
    a3.original, a3.split, a3.track = src, a2.output, "Cuerdas"
    res.append(("split + check CLI", rc == 0 and cmd_check(a3) == 0))
    _, back = read_tracks(a2.output)
    progs = [t["program"] for t in back if t["name"].startswith("Cuerdas · V")]
    res.append(("programas por voz", progs == [40, 41, 42, 43]))
    for n, v in res:
        print(f"  {'✓' if v else '✗'} {n}")
    return 0 if all(v for _, v in res) else 1


def main():
    ap = argparse.ArgumentParser(description="Divide pistas de sección en voces monofónicas")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info"); p.add_argument("midi"); p.set_defaults(fn=cmd_info)
    p = sub.add_parser("split"); p.add_argument("midi")
    p.add_argument("--track"); p.add_argument("--all", action="store_true")
    p.add_argument("--voices", type=int); p.add_argument("--programs", type=int, nargs="+")
    p.add_argument("--keep-others", action="store_true"); p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_split)
    p = sub.add_parser("check"); p.add_argument("original"); p.add_argument("split")
    p.add_argument("--track"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    try:
        sys.exit(a.fn(a))
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e.filename}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
