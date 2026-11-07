#!/usr/bin/env python3
"""
ostinato_detector.py — Ostinato Detector: aísla una voz (por defecto el bajo)
y encuentra células que se repiten sin cambiar: pedal, ostinato melódico o
rítmico, con su periodo, dónde empieza/acaba y cuántas veces se repite
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Detectores de «motivos repetidos» (Re-Pair, n-gramas) priorizan el
acompañamiento y no distinguen un ostinato grave de un motivo melódico. Aquí
la voz se elige explícitamente (bajo, melodía, pista concreta o cada pista) y
se busca una PERIODICIDAD: para cada periodo candidato P (en semicorcheas)
se compara la voz consigo misma desplazada P posiciones y se toma el tramo
más largo con pocas discrepancias (suma máxima de aciertos − penalización de
fallos). Se elige el periodo más corto que explica el tramo. Se informan
tres tipos: «pedal» (una sola altura repetida), «ostinato» (varias alturas)
y «ritmico» (mismo ritmo con alturas cambiantes, p. ej. un marcato sobre
armonías distintas). También da el mapa de CAPAS (qué voz entra en qué
compás) para estudiar acumulación.
Autocontenida: solo mido.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  detect   Ostinatos de la voz elegida (célula, periodo, tramo, repeticiones)
  layers   Mapa de capas: entrada de cada voz por compás y su ostinato
  export   Guarda la célula del mejor ostinato como MIDI

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  detect
    midi MIDI
    --voice VOICE               bass|melody|each|track:N   (defecto bass)
    --top TOP                   ostinatos a informar (defecto 3)
    --min-reps MIN_REPS         repeticiones mínimas (defecto 3)
    --min-bars MIN_BARS         duración mínima del tramo en compases (defecto 2)
    --max-period MAX_PERIOD     periodo máximo en semicorcheas (defecto 64)
    --tol TOL                   tolerancia de fallos 0-0.5 (defecto 0.1)
    --drums                     incluye el canal 10
    --json JSON
    --no-color

  layers
    midi MIDI
    --min-notes MIN_NOTES       ignora voces con menos notas (defecto 8)
    --json JSON
    --no-color

  export
    midi MIDI
    --out OUT
    --voice VOICE               (defecto bass)
    --repeat REPEAT             veces que se escribe la célula (defecto 4)

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[detect]
midi         file  io=in ext=mid required
--voice      str   default=bass
--top        int   default=3 min=1
--min-reps   int   default=3 min=2
--min-bars   float default=2 min=0.25 unit=compases
--max-period int   default=64 min=2 unit=semicorcheas
--tol        float default=0.1 min=0 max=0.5
--drums      bool  default=false
--json       file  io=out ext=json
--no-color   bool  default=false
[layers]
midi         file  io=in ext=mid required
--min-notes  int   default=8 min=1
--json       file  io=out ext=json
--no-color   bool  default=false
[export]
midi         file  io=in ext=mid required
--out        file  io=out ext=mid required
--voice      str   default=bass
--repeat     int   default=4 min=1
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python ostinato_detector.py detect obra.mid
    python ostinato_detector.py detect obra.mid --voice each --top 2
    python ostinato_detector.py detect obra.mid --voice track:2 --min-reps 4
    python ostinato_detector.py layers obra.mid
    python ostinato_detector.py export obra.mid --out celula.mid --repeat 8

  SELECCIÓN DE VOZ:
    bass       la pista/canal de registro medio más grave (≥ 8 notas),
               monofonizada a la nota más grave por posición
    melody     la de registro medio más agudo
    track:N    la pista N (la de «layers»)
    each       todas las voces con suficientes notas

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 2 fichero ilegible

  API:  from ostinato_detector import load_voices, find_ostinatos

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  mido
"""

import argparse
import json
import math
import os
import sys

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def _ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pname(p):
    return f"{NOTE[p % 12]}{p // 12 - 1}"


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA Y SELECCIÓN DE VOCES
# ══════════════════════════════════════════════════════════════════════════════

def load_voices(path, drums=False, min_notes=8):
    """Devuelve (meta, voices). Una voz = dict(id,label,notes=[(s,e,p)],median)."""
    mid = mido.MidiFile(path, clip=True)
    tpb = mid.ticks_per_beat
    tsigs, tempos = [], []
    voices = {}
    for ti, tr in enumerate(mid.tracks):
        t = 0
        name = ""
        active = {}
        for m in tr:
            t += m.time
            if m.type == "track_name":
                name = m.name
            elif m.type == "time_signature":
                tsigs.append((t, m.numerator, m.denominator))
            elif m.type == "set_tempo":
                tempos.append((t, m.tempo))
            elif m.type == "note_on" and m.velocity > 0:
                active.setdefault((m.channel, m.note), []).append(t)
            elif m.type in ("note_off", "note_on"):
                st = active.get((m.channel, m.note))
                if st:
                    s = st.pop(0)
                    if drums or m.channel != 9:
                        key = (ti, m.channel)
                        v = voices.setdefault(key, {"track": ti, "channel": m.channel,
                                                    "name": name, "notes": []})
                        v["notes"].append((s, max(t, s + 1), m.note))
        for (ch, p), st in active.items():
            for s in st:
                if drums or ch != 9:
                    v = voices.setdefault((ti, ch), {"track": ti, "channel": ch,
                                                     "name": name, "notes": []})
                    v["notes"].append((s, s + tpb, p))
    out = []
    for i, (key, v) in enumerate(sorted(voices.items())):
        if len(v["notes"]) < min_notes:
            continue
        v["notes"].sort()
        ps = sorted(n[2] for n in v["notes"])
        v["median"] = ps[len(ps) // 2]
        v["id"] = len(out)
        v["label"] = f"{v['name'] or 'pista ' + str(v['track'])}/ch{v['channel'] + 1}"
        out.append(v)
    tsigs.sort()
    num, den = (tsigs[0][1], tsigs[0][2]) if tsigs else (4, 4)
    meta = {"tpb": tpb, "num": num, "den": den,
            "bar_ticks": int(tpb * 4 * num / den),
            "tempo": tempos[0][1] if tempos else 500000}
    return meta, out


def select_voices(voices, spec):
    if not voices:
        return []
    if spec == "each":
        return list(voices)
    if spec == "melody":
        return [max(voices, key=lambda v: v["median"])]
    if spec.startswith("track:"):
        n = int(spec.split(":")[1])
        sel = [v for v in voices if v["track"] == n]
        return sel
    return [min(voices, key=lambda v: v["median"])]      # bass


def to_slots(voice, tpb, mono="low"):
    """Voz -> lista indexada por semicorchea con la altura que ataca (o -1)."""
    q = max(1, tpb // 4)
    best = {}
    for s, e, p in voice["notes"]:
        k = int(round(s / q))
        cur = best.get(k)
        if cur is None or (p < cur if mono == "low" else p > cur):
            best[k] = p
    n = (max(best) + 1) if best else 0
    seq = [-1] * n
    for k, p in best.items():
        seq[k] = p
    return seq


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE PERIODICIDAD
# ══════════════════════════════════════════════════════════════════════════════

def _best_segment(seq, P, rhythmic, tol):
    """Kadane sobre la comparación seq[i]==seq[i+P].
    Devuelve (score, start, end_exclusive, matches, mismatches)."""
    N = len(seq)
    if N <= P:
        return 0, 0, 0, 0, 0
    w = max(1.0, (1.0 - tol) / max(tol, 1e-6))           # penalización de un fallo
    best = (0.0, 0, 0, 0, 0)
    cur, cs, cm, cx = 0.0, 0, 0, 0
    for i in range(N - P):
        a, b = seq[i], seq[i + P]
        if a == -1 and b == -1:
            continue
        ok = (a != -1 and b != -1) if rhythmic else (a == b)
        if cur <= 0:
            cur, cs, cm, cx = 0.0, i, 0, 0
        if ok:
            cur += 1.0
            cm += 1
        else:
            cur -= w
            cx += 1
        if cur > best[0]:
            best = (cur, cs, i + P + 1, cm, cx)
    return best


def _trim_to_events(seq, start, end):
    """Ajusta el tramo a la primera/última nota real."""
    while start < end and seq[start] == -1:
        start += 1
    while end > start and seq[end - 1] == -1:
        end -= 1
    return start, end


def _refine_run(seq, s, e, P, rhythmic, min_sim=0.75):
    """Divide el tramo en repeticiones y devuelve la racha homogénea más larga
    (evita fusionar dos ostinatos consecutivos con el mismo periodo)."""
    nrep = (e - s) // P
    if nrep < 2:
        return s, e
    reps = [seq[s + k * P:s + (k + 1) * P] for k in range(nrep)]

    def sim(a, b):
        idx = [i for i in range(P) if a[i] != -1 or b[i] != -1]
        if not idx:
            return 1.0
        if rhythmic:
            ok = sum(1 for i in idx if (a[i] != -1) == (b[i] != -1))
        else:
            ok = sum(1 for i in idx if a[i] == b[i])
        return ok / len(idx)

    runs, st = [], 0
    for k in range(1, nrep):
        if sim(reps[k - 1], reps[k]) < min_sim:
            runs.append((st, k))
            st = k
    runs.append((st, nrep))
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return s + a * P, (s + b * P) if b < nrep else e


def find_ostinatos(seq, slots_per_bar, top=3, min_reps=3, max_period=64, tol=0.1, min_bars=2.0):
    """Encuentra hasta `top` ostinatos disjuntos en una secuencia de slots."""
    seq = list(seq)
    found = []
    for _ in range(top):
        cands = []
        for rhythmic in (False, True):
            for P in range(2, min(max_period, len(seq) // max(2, min_reps)) + 1):
                sc, s, e, m, x = _best_segment(seq, P, rhythmic, tol)
                if m == 0:
                    continue
                s, e = _trim_to_events(seq, s, e)
                s, e = _refine_run(seq, s, e, P, rhythmic)
                s, e = _trim_to_events(seq, s, e)
                if e - s < P * min_reps or (e - s) < min_bars * slots_per_bar:
                    continue
                reps = (e - s) / P
                cell = seq[s:s + P]
                onsets = [v for v in cell if v != -1]
                if not onsets:
                    continue
                cands.append({"period": P, "start": s, "end": e, "reps": reps, "score": sc,
                              "matches": m, "mismatches": x, "rhythmic": rhythmic})
        if not cands:
            break
        # prioridad: tramo exacto > rítmico; mayor cobertura; periodo más corto entre los equivalentes
        def key(c):
            return (not c["rhythmic"], c["score"])
        top_c = max(cands, key=key)
        # el periodo más corto que explique casi lo mismo y solape con el mejor
        near = [c for c in cands if c["rhythmic"] == top_c["rhythmic"]
                and c["score"] >= 0.85 * top_c["score"]
                and c["start"] < top_c["end"] and c["end"] > top_c["start"]]
        ch = min(near, key=lambda c: (c["period"], -c["score"]))
        s, e, P = ch["start"], ch["end"], ch["period"]
        # fase: la célula empieza en la primera nota del tramo
        cell = seq[s:s + P]
        pitches = [v for v in cell if v != -1]
        if ch["rhythmic"]:
            kind = "ritmico"
        elif len(set(pitches)) == 1:
            kind = "pedal"
        else:
            kind = "ostinato"
        ch.update(kind=kind, cell=cell,
                  coverage=(e - s) / max(1, len(seq)))
        # ¿hay un ritmo más corto que se repite sobre las mismas alturas cambiantes?
        if not ch["rhythmic"]:
            rh = [c for c in cands if c["rhythmic"] and c["period"] < P
                  and (min(c["end"], e) - max(c["start"], s)) >= 0.8 * (e - s)]
            if rh:
                r0 = min(rh, key=lambda c: c["period"])
                ch["rhythm_period"] = r0["period"]
                ch["rhythm_cell"] = "".join("x" if v != -1 else "." for v in seq[r0["start"]:r0["start"] + r0["period"]])
        found.append(ch)
        for i in range(s, e):                 # enmascarar y buscar el siguiente
            seq[i] = -1
    return found


def cell_text(cell):
    """Texto de la célula: alturas con su posición en la rejilla de semicorcheas."""
    ons = [(i, p) for i, p in enumerate(cell) if p != -1]
    rhythm = "".join("x" if v != -1 else "." for v in cell)
    return rhythm, " ".join(f"{pname(p)}@{i}" for i, p in ons)


def describe(osti, meta, voice_label):
    spb = meta["bar_ticks"] // max(1, meta["tpb"] // 4)       # semicorcheas por compás
    rows = []
    for o in osti:
        rhythm, notes = cell_text(o["cell"])
        rows.append({
            "voice": voice_label, "kind": o["kind"],
            "period_16ths": o["period"], "period_beats": round(o["period"] / 4.0, 2),
            "period_bars": round(o["period"] / spb, 3),
            "start_bar": o["start"] // spb + 1, "end_bar": int(math.ceil(o["end"] / spb)),
            "reps": round(o["reps"], 1), "coverage": round(o["coverage"], 3),
            "mismatches": o["mismatches"], "rhythm": rhythm, "cell": notes,
            "rhythm_period_16ths": o.get("rhythm_period"), "rhythm_cell": o.get("rhythm_cell"),
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def _load(path, drums=False, min_notes=8):
    try:
        return load_voices(path, drums=drums, min_notes=min_notes)
    except Exception as e:  # noqa: BLE001
        print(f"Ilegible: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def cmd_detect(args):
    r = _load(args.midi, drums=args.drums)
    if r is None:
        return 2
    meta, voices = r
    sel = select_voices(voices, args.voice)
    if not sel:
        print("No hay voces con notas suficientes para esa selección.")
        return 0
    spb = meta["bar_ticks"] // max(1, meta["tpb"] // 4)
    mono = "high" if args.voice == "melody" else "low"
    allrows = []
    print(f"\n{_c('B')}{os.path.basename(args.midi)}{_c('R')}  "
          f"({meta['num']}/{meta['den']}, {spb} semicorcheas/compás)")
    for v in sel:
        seq = to_slots(v, meta["tpb"], mono)
        osti = find_ostinatos(seq, spb, args.top, args.min_reps, args.max_period, args.tol, args.min_bars)
        rows = describe(osti, meta, v["label"])
        allrows += rows
        print(f"\n  Voz: {_c('CYA')}{v['label']}{_c('R')}  ({len(v['notes'])} notas, registro medio {pname(v['median'])})")
        if not rows:
            print("    sin ostinato detectable con estos parámetros")
        for i, o in enumerate(rows, 1):
            print(f"    #{i} {_c('B')}{o['kind']:<8}{_c('R')} periodo {o['period_beats']} pulsos "
                  f"({o['period_bars']} comp.)  compases {o['start_bar']}–{o['end_bar']}  "
                  f"×{o['reps']}  cobertura {o['coverage']:.0%}  fallos {o['mismatches']}")
            print(f"       ritmo {o['rhythm']}   {_c('G')}{o['cell']}{_c('R')}")
            if o.get("rhythm_cell"):
                print(f"       ritmo base repetido cada {o['rhythm_period_16ths'] / 4} pulsos: {o['rhythm_cell']}")
    print()
    if args.json:
        _ensure_parent(args.json)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(allrows, f, ensure_ascii=False, indent=1)
    return 0


def cmd_layers(args):
    r = _load(args.midi, min_notes=args.min_notes)
    if r is None:
        return 2
    meta, voices = r
    if not voices:
        print("Sin voces.")
        return 0
    spb = meta["bar_ticks"] // max(1, meta["tpb"] // 4)
    last = max(n[1] for v in voices for n in v["notes"])
    nbars = int(math.ceil(last / meta["bar_ticks"]))
    out = []
    print(f"\n{_c('B')}{os.path.basename(args.midi)}{_c('R')}  ({nbars} compases)\n")
    shades = " ░▒▓█"
    width = min(nbars, 96)
    scale = nbars / width
    for v in sorted(voices, key=lambda v: v["median"]):
        dens = [0] * nbars
        for s, _e, _p in v["notes"]:
            b = min(nbars - 1, s // meta["bar_ticks"])
            dens[b] += 1
        mx = max(dens) or 1
        line = ""
        for c in range(width):
            b0, b1 = int(c * scale), max(int(c * scale) + 1, int((c + 1) * scale))
            d = max(dens[b0:b1])
            line += shades[0] if d == 0 else shades[min(4, 1 + int(3 * d / mx))]
        entry = next(i for i, d in enumerate(dens) if d) + 1
        seq = to_slots(v, meta["tpb"], "low")
        osti = describe(find_ostinatos(seq, spb, 1, 3, 64, 0.1), meta, v["label"])
        o = osti[0] if osti else None
        tag = (f"{o['kind']} c.{o['start_bar']}–{o['end_bar']} ×{o['reps']}" if o else "—")
        print(f"  {v['label'][:20]:<20} {pname(v['median']):<4} entra c.{entry:<3} |{line}| {tag}")
        out.append({"voice": v["label"], "track": v["track"], "median": pname(v["median"]),
                    "entry_bar": entry, "ostinato": o})
    entries = sorted(set(x["entry_bar"] for x in out))
    print(f"\n  Entradas de capa en los compases: {', '.join(map(str, entries))}"
          f"   ({len(entries)} entradas distintas → {'acumulación' if len(entries) >= 3 else 'poca acumulación'})\n")
    if args.json:
        _ensure_parent(args.json)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    return 0


def cmd_export(args):
    r = _load(args.midi)
    if r is None:
        return 2
    meta, voices = r
    sel = select_voices(voices, args.voice)
    if not sel:
        print("Sin voz para exportar.", file=sys.stderr)
        return 1
    v = sel[0]
    spb = meta["bar_ticks"] // max(1, meta["tpb"] // 4)
    seq = to_slots(v, meta["tpb"], "high" if args.voice == "melody" else "low")
    osti = find_ostinatos(seq, spb, 1)
    if not osti:
        print("No se detectó ostinato que exportar.", file=sys.stderr)
        return 1
    o = osti[0]
    cell = o["cell"]
    q = max(1, meta["tpb"] // 4)
    # duración de cada nota = hasta el siguiente ataque de la célula (o fin de periodo)
    ons = [i for i, p in enumerate(cell) if p != -1]
    mid = mido.MidiFile(ticks_per_beat=meta["tpb"])
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=meta["tempo"]))
    tr.append(mido.MetaMessage("time_signature", numerator=meta["num"], denominator=meta["den"]))
    tr.append(mido.MetaMessage("track_name", name=f"ostinato {o['kind']}"))
    events = []
    for rep in range(args.repeat):
        base = rep * o["period"]
        for j, i in enumerate(ons):
            nxt = ons[j + 1] if j + 1 < len(ons) else o["period"]
            s = (base + i) * q
            e = (base + nxt) * q
            events.append((s, 1, cell[i]))
            events.append((max(s + 1, e - max(1, q // 4)), 0, cell[i]))
    events.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for t, on, p in events:
        tr.append(mido.Message("note_on" if on else "note_off", note=p,
                               velocity=80 if on else 0, time=t - last))
        last = t
    _ensure_parent(args.out)
    mid.save(args.out)
    rhythm, notes = cell_text(cell)
    print(f"Célula ({o['kind']}, periodo {o['period'] / 4} pulsos) ×{args.repeat} → {args.out}")
    print(f"  {rhythm}   {notes}")
    return 0


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="ostinato_detector.py",
                                 description="Detecta y exporta ostinatos (bajo, melodía o cualquier voz).")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    d = sub.add_parser("detect", help="Ostinatos de la voz elegida")
    d.add_argument("midi")
    d.add_argument("--voice", default="bass")
    d.add_argument("--top", type=int, default=3)
    d.add_argument("--min-reps", type=int, default=3)
    d.add_argument("--min-bars", type=float, default=2.0)
    d.add_argument("--max-period", type=int, default=64)
    d.add_argument("--tol", type=float, default=0.1)
    d.add_argument("--drums", action="store_true")
    d.add_argument("--json")
    d.add_argument("--no-color", action="store_true")
    d.set_defaults(fn=cmd_detect)

    l = sub.add_parser("layers", help="Mapa de capas por compás")
    l.add_argument("midi")
    l.add_argument("--min-notes", type=int, default=8)
    l.add_argument("--json")
    l.add_argument("--no-color", action="store_true")
    l.set_defaults(fn=cmd_layers)

    e = sub.add_parser("export", help="Exporta la célula del mejor ostinato")
    e.add_argument("midi")
    e.add_argument("--out", required=True)
    e.add_argument("--voice", default="bass")
    e.add_argument("--repeat", type=int, default=4)
    e.set_defaults(fn=cmd_export)

    args = ap.parse_args(argv)
    if getattr(args, "no_color", False):
        _USE_COLOR = False
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
