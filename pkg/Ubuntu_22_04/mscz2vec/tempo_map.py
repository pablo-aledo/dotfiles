#!/usr/bin/env python3
"""
tempo_map.py — Tempo Map: visualizar y editar el mapa de tempo de un MIDI existente
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
tempo_designer.py DISEÑA una agógica desde cero (terrazas, rampas, calderones) y
performer aplica rubato por frase, pero nada permitía MIRAR el mapa de tempo de un
MIDI ya generado (p. ej. los 150 cambios que produce generar_midi.py por
interpolación lineal) ni retocarlo puntualmente sin volver a editar el YAML. Esta
herramienta trabaja sobre el MIDI final:

  show     Tabla por compás (BPM al inicio, mín/máx dentro del compás) y gráfico
           ASCII de la curva; detecta saltos bruscos
  edit     Retoques puntuales sobre el mapa existente, sin tocar las notas:
             --set C:BPM          fija el tempo desde el compás C
             --scale S:E=F        multiplica el tempo de los compases S..E por F
             --rubato S:E=A       rubato: ± A (fracción) con forma --shape
                                  (arch = se frena en el centro · ritard = se frena al final ·
                                   accel = se acelera al final · hold = frena todo el tramo)
             --fermata C=F        calderón: tempo del compás C dividido por F
  smooth   Suaviza el mapa con media móvil de N compases (elimina escalones)
  export   Vuelca el mapa a JSON / CSV / fragmento YAML (compás: BPM)
  selftest Comprueba con un MIDI sintético que las notas no cambian

El editor NO mueve ni modifica ninguna nota: solo sustituye los meta-eventos
set_tempo (todos pasan a la pista 0). Los compases se numeran desde 1 y se calculan
con la primera time_signature (compases con cambios de compás: se usa el mapa
completo de time_signature).

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  show    midi [--width 60] [--bars S:E] [--no-table]
  edit    midi [-o SALIDA.mid] [--set C:BPM ...] [--scale S:E=F ...]
               [--rubato S:E=A ...] [--shape arch|ritard|accel|hold]
               [--fermata C=F ...] [--resolution beat|bar]
  smooth  midi [-o SALIDA.mid] [--window 3] [--resolution beat|bar]
  export  midi [--json F] [--csv F] [--yaml F]
  selftest

EJEMPLOS
  python tempo_map.py show obra.mid
  python tempo_map.py edit obra.mid --rubato 40:46=0.12 --shape ritard -o obra_rub.mid
  python tempo_map.py edit obra.mid --fermata 146=3 --set 1:50 -o obra2.mid
  python tempo_map.py smooth obra.mid --window 5 -o suave.mid
  python tempo_map.py export obra.mid --csv tempo.csv
"""
import argparse
import bisect
import csv
import json
import math
import os
import sys
import tempfile

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
SPARK = "▁▂▃▄▅▆▇█"


def _c(code):
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "GRN": "\033[92m",
            "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}.get(code, "")


# ══════════════════════════════════════════════════════════════════════════════
#  MAPAS (tempo y compás)
# ══════════════════════════════════════════════════════════════════════════════

class Score:
    def __init__(self, path):
        self.path = path
        self.mid = mido.MidiFile(path)
        self.tpb = self.mid.ticks_per_beat
        tempos, sigs, end = {}, {}, 0
        for tr in self.mid.tracks:
            t = 0
            for m in tr:
                t += m.time
                if m.type == "set_tempo":
                    tempos[t] = m.tempo
                elif m.type == "time_signature":
                    sigs[t] = (m.numerator, m.denominator)
            end = max(end, t)
        self.end = end
        self.tempos = sorted(tempos.items()) or [(0, 500000)]
        if self.tempos[0][0] != 0:
            self.tempos.insert(0, (0, 500000))
        self.sigs = sorted(sigs.items()) or [(0, (4, 4))]
        if self.sigs[0][0] != 0:
            self.sigs.insert(0, (0, (4, 4)))
        # inicio de cada compás
        self.bar_starts = [0]
        t = 0
        while t < end:
            i = bisect.bisect_right([s[0] for s in self.sigs], t) - 1
            n, d = self.sigs[i][1]
            t += int(self.tpb * 4 * n / d)
            self.bar_starts.append(t)
        self.n_bars = max(1, len(self.bar_starts) - 1)

    def bar_ticks(self, bar):                 # compás 1-based → (start, end)
        bar = max(1, min(bar, self.n_bars))
        return self.bar_starts[bar - 1], self.bar_starts[bar]

    def bar_of(self, tick):
        return max(1, min(self.n_bars, bisect.bisect_right(self.bar_starts, tick)))

    def us_at(self, tick, tempos=None):
        tempos = tempos or self.tempos
        i = bisect.bisect_right([t for t, _ in tempos], tick) - 1
        return tempos[max(i, 0)][1]

    def bpm_at(self, tick, tempos=None):
        return 60_000_000 / self.us_at(tick, tempos)

    def bar_table(self):
        rows = []
        for b in range(1, self.n_bars + 1):
            s, e = self.bar_ticks(b)
            inside = [60_000_000 / u for t, u in self.tempos if s <= t < e]
            first = self.bpm_at(s)
            allv = [first] + inside
            rows.append({"bar": b, "bpm": round(first, 2),
                         "min": round(min(allv), 2), "max": round(max(allv), 2),
                         "changes": len(inside)})
        return rows


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA (sustituir set_tempo conservando todo lo demás)
# ══════════════════════════════════════════════════════════════════════════════

def write_with_tempos(sc, tempos, path):
    """Escribe el MIDI con un nuevo mapa de tempo [(tick, us)] en la pista 0.
    Las notas y demás eventos conservan su tick absoluto."""
    mid = mido.MidiFile(type=sc.mid.type, ticks_per_beat=sc.tpb)
    tempos = _dedupe(sorted(tempos))
    for ti, tr in enumerate(sc.mid.tracks):
        ev, t = [], 0
        for m in tr:
            t += m.time
            if m.type == "end_of_track" or m.type == "set_tempo":
                continue
            ev.append((t, 1, m.copy(time=0)))
        end_t = t
        if ti == 0:
            ev += [(tk, 0, mido.MetaMessage("set_tempo", tempo=int(us), time=0)) for tk, us in tempos]
            end_t = max([end_t] + [tk for tk, _ in tempos])
        ev.sort(key=lambda x: (x[0], x[1]))
        out, last = mido.MidiTrack(), 0
        for tk, _, m in ev:
            out.append(m.copy(time=tk - last))
            last = tk
        out.append(mido.MetaMessage("end_of_track", time=max(0, end_t - last)))
        mid.tracks.append(out)
    mid.save(_ensure_parent(path))


def _dedupe(tempos):
    out = []
    for tk, us in tempos:
        us = max(1, int(round(us)))
        if out and out[-1][0] == tk:
            out[-1] = (tk, us)
        elif not out or out[-1][1] != us:
            out.append((tk, us))
    if not out or out[0][0] != 0:
        out.insert(0, (0, 500000))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  EDICIÓN
# ══════════════════════════════════════════════════════════════════════════════

def parse_range(s):
    a, b = s.split(":")
    return int(a), int(b)


def shape_factor(shape, t, amount):
    """Factor de tempo (BPM) para t∈[0,1]. amount>0 frena."""
    if shape == "arch":
        return 1 - amount * math.sin(math.pi * t)
    if shape == "ritard":
        return 1 - amount * (t ** 2)
    if shape == "accel":
        return 1 + amount * (t ** 2)
    if shape == "hold":
        return 1 - amount
    raise ValueError(f"forma desconocida: {shape}")


def build_multiplier(sc, scales, rubatos, shape, fermatas):
    """Devuelve g(tick) → factor multiplicativo del BPM efectivo, y los límites tocados."""
    spans = []                                        # (t0, t1, fn)
    for rng, f in scales:
        s, e = parse_range(rng)
        t0, t1 = sc.bar_ticks(s)[0], sc.bar_ticks(e)[1]
        spans.append((t0, t1, (lambda f: lambda t: f)(f)))
    for rng, a in rubatos:
        s, e = parse_range(rng)
        t0, t1 = sc.bar_ticks(s)[0], sc.bar_ticks(e)[1]
        spans.append((t0, t1, (lambda t0, t1, a: lambda t: shape_factor(
            shape, (t - t0) / max(1, t1 - t0), a))(t0, t1, a)))
    for bar, f in fermatas:
        t0, t1 = sc.bar_ticks(bar)
        spans.append((t0, t1, (lambda f: lambda t: 1.0 / f)(f)))

    def g(tick):
        r = 1.0
        for t0, t1, fn in spans:
            if t0 <= tick < t1:
                r *= fn(tick)
        return r
    return g, spans


def apply_edits(sc, sets, scales, rubatos, shape, fermatas, resolution="beat"):
    """→ nuevo mapa [(tick, us)]. Conserva el mapa original fuera de las zonas editadas."""
    g, spans = build_multiplier(sc, scales, rubatos, shape, fermatas)
    # 1) puntos de muestreo: eventos originales + rejilla dentro de los tramos editados
    pts = {t for t, _ in sc.tempos}
    step = sc.tpb if resolution == "beat" else None
    for t0, t1, _ in spans:
        if step:
            tt = t0
            while tt < t1:
                pts.add(tt)
                tt += step
        else:
            for b in range(sc.bar_of(t0), sc.bar_of(max(t0, t1 - 1)) + 1):
                pts.add(sc.bar_ticks(b)[0])
        pts.add(t0)
        pts.add(t1)                                   # restituye el mapa original al salir
    base = {t: sc.us_at(t) for t in pts}
    new = {t: base[t] / g(t) for t in pts}            # menos BPM = más microsegundos/negra
    # 2) --set: fija el tempo desde el compás C (hasta el siguiente evento propio)
    for bar, bpm in sets:
        t = sc.bar_ticks(bar)[0]
        new[t] = 60_000_000 / bpm
    return sorted(new.items())


def smooth_map(sc, window, resolution="beat"):
    pts = []
    step = sc.tpb if resolution == "beat" else None
    t = 0
    while t < sc.end:
        pts.append(t)
        t += step or max(1, sc.bar_ticks(sc.bar_of(t))[1] - sc.bar_ticks(sc.bar_of(t))[0])
    bpms = [sc.bpm_at(t) for t in pts]
    # ventana en compases → nº de puntos
    per_bar = max(1, len(pts) // max(1, sc.n_bars))
    k = max(1, int(window * per_bar))
    half = k // 2
    out = []
    for i, t in enumerate(pts):
        lo, hi = max(0, i - half), min(len(pts), i + half + 1)
        out.append((t, 60_000_000 / (sum(bpms[lo:hi]) / (hi - lo))))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def spark_line(vals, width):
    n = len(vals)
    cols = []
    for x in range(width):
        i0, i1 = int(x * n / width), max(int((x + 1) * n / width), int(x * n / width) + 1)
        cols.append(sum(vals[i0:i1]) / (i1 - i0))
    lo, hi = min(cols), max(cols)
    if hi - lo < 1e-9:
        return SPARK[3] * width
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in cols)


def cmd_show(a):
    sc = Score(a.midi)
    rows = sc.bar_table()
    s, e = (parse_range(a.bars) if a.bars else (1, sc.n_bars))
    sel = [r for r in rows if s <= r["bar"] <= e]
    B, R, G = _c("B"), _c("R"), _c("G")
    bpms = [r["bpm"] for r in rows]
    print(f"\n  {B}TEMPO MAP v{VERSION} · {os.path.basename(a.midi)}{R}   "
          f"{sc.n_bars} compases · {len(sc.tempos)} eventos de tempo · "
          f"{min(bpms):.0f}–{max(bpms):.0f} BPM")
    print(f"  {G}{spark_line([r['bpm'] for r in sel], a.width)}{R}")
    print(f"  {G}cc {sel[0]['bar']}{' ' * max(1, a.width - 8 - len(str(sel[-1]['bar'])))}cc {sel[-1]['bar']}{R}")
    jumps = [(r0["bar"], r1["bar"], r0["bpm"], r1["bpm"]) for r0, r1 in zip(rows, rows[1:])
             if abs(r1["bpm"] - r0["bpm"]) > 0.25 * r0["bpm"]]
    if not a.no_table:
        print(f"\n  {G}{'cc':>4} {'BPM':>7} {'mín':>7} {'máx':>7}  cambios{R}")
        prev = None
        for r in sel:
            if prev is not None and r["bpm"] == prev and r["changes"] == 0 and a.compact:
                continue
            print(f"  {r['bar']:>4} {r['bpm']:>7.1f} {r['min']:>7.1f} {r['max']:>7.1f}  "
                  f"{r['changes'] or ''}")
            prev = r["bpm"]
    if jumps:
        print(f"\n  {_c('YEL')}⚠ {len(jumps)} salto(s) brusco(s) (>25 %): "
              + ", ".join(f"cc{x}→{y} ({p:.0f}→{q:.0f} BPM)" for x, y, p, q in jumps[:6]) + R)
    print()
    return 0


def _outpath(a, suffix):
    return a.output or os.path.splitext(a.midi)[0] + suffix + ".mid"


def cmd_edit(a):
    sc = Score(a.midi)
    sets = []
    for x in a.set or []:
        c, b = x.split(":")
        sets.append((int(c), float(b)))
    scales = []
    for x in a.scale or []:
        r, f = x.split("=")
        scales.append((r, float(f)))
    rubs = []
    for x in a.rubato or []:
        r, am = x.split("=")
        rubs.append((r, float(am)))
    ferm = []
    for x in a.fermata or []:
        c, f = x.split("=")
        ferm.append((int(c), float(f)))
    if not (sets or scales or rubs or ferm):
        print("[ERROR] nada que editar: use --set, --scale, --rubato o --fermata", file=sys.stderr)
        return 2
    for _, f in scales:
        if f <= 0:
            print("[ERROR] --scale necesita un factor > 0", file=sys.stderr)
            return 2
    for _, f in ferm:
        if f <= 0:
            print("[ERROR] --fermata necesita un factor > 0", file=sys.stderr)
            return 2
    tempos = apply_edits(sc, sets, scales, rubs, a.shape, ferm, a.resolution)
    path = _outpath(a, "_tempo")
    write_with_tempos(sc, tempos, path)
    n = Score(path)
    print(f"  ✓ {path}: {len(sc.tempos)} → {len(n.tempos)} eventos de tempo "
          f"({min(r['bpm'] for r in n.bar_table()):.0f}–{max(r['bpm'] for r in n.bar_table()):.0f} BPM)")
    return 0


def cmd_smooth(a):
    sc = Score(a.midi)
    tempos = smooth_map(sc, a.window, a.resolution)
    path = _outpath(a, "_suave")
    write_with_tempos(sc, tempos, path)
    n = Score(path)
    print(f"  ✓ {path}: {len(sc.tempos)} → {len(n.tempos)} eventos de tempo")
    return 0


def cmd_export(a):
    sc = Score(a.midi)
    rows = sc.bar_table()
    if a.json:
        json.dump({"file": os.path.basename(a.midi), "tpb": sc.tpb, "bars": rows,
                   "events": [{"tick": t, "bpm": round(60_000_000 / u, 3)} for t, u in sc.tempos]},
                  open(_ensure_parent(a.json), "w", encoding="utf-8"), indent=1)
        print(f"  → {a.json}")
    if a.csv:
        with open(_ensure_parent(a.csv), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["bar", "bpm", "min", "max", "changes"])
            for r in rows:
                w.writerow([r["bar"], r["bpm"], r["min"], r["max"], r["changes"]])
        print(f"  → {a.csv}")
    if a.yaml:
        with open(_ensure_parent(a.yaml), "w", encoding="utf-8") as fh:
            fh.write("# compás: BPM al inicio del compás\ntempo_por_compas:\n")
            for r in rows:
                fh.write(f"  {r['bar']}: {r['bpm']}\n")
        print(f"  → {a.yaml}")
    if not (a.json or a.csv or a.yaml):
        for r in rows:
            print(f"{r['bar']},{r['bpm']}")
    return 0


def _notes(path):
    out = []
    for tr in mido.MidiFile(path).tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type in ("note_on", "note_off"):
                out.append((t, m.type, m.note, m.velocity))
    return sorted(out)


def cmd_selftest(a):
    d = tempfile.mkdtemp()
    src = os.path.join(d, "s.mid")
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    for b in range(16):
        meta.append(mido.MetaMessage("set_tempo", tempo=int(60e6 / (60 + 3 * b)),
                                     time=0 if b == 0 else 1920))
    mid.tracks.append(meta)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name="m"))
    for i in range(64):
        tr.append(mido.Message("note_on", note=60 + i % 12, velocity=80, time=0 if i == 0 else 0))
        tr.append(mido.Message("note_off", note=60 + i % 12, velocity=0, time=480))
    mid.tracks.append(tr)
    mid.save(src)
    sc = Score(src)
    res = [("16 compases / 16 eventos", sc.n_bars == 16 and len(sc.tempos) == 16)]
    ref = _notes(src)

    def run(**kw):
        class A:  # noqa
            pass
        o = A()
        o.midi, o.output, o.resolution, o.shape = src, os.path.join(d, "o.mid"), "beat", "arch"
        o.set = o.scale = o.rubato = o.fermata = None
        for k, v in kw.items():
            setattr(o, k, v)
        assert cmd_edit(o) == 0
        return Score(o.output), o.output

    n, p = run(rubato=["5:8=0.2"])
    res.append(("rubato arch: notas intactas", _notes(p) == ref))
    mid_bpm = n.bpm_at(n.bar_ticks(6)[0] + 960)
    orig_bpm = sc.bpm_at(n.bar_ticks(6)[0] + 960)
    res.append(("rubato arch frena en el centro", mid_bpm < orig_bpm * 0.9))
    res.append(("fuera del tramo no cambia", abs(n.bpm_at(n.bar_ticks(10)[0]) - sc.bpm_at(n.bar_ticks(10)[0])) < 0.05
                and abs(n.bpm_at(n.bar_ticks(2)[0]) - sc.bpm_at(n.bar_ticks(2)[0])) < 0.05))
    n, p = run(scale=["3:4=0.5"], rubato=None)
    res.append(("scale 0.5 divide el BPM",
                abs(n.bpm_at(n.bar_ticks(3)[0]) - sc.bpm_at(n.bar_ticks(3)[0]) * 0.5) < 0.1))
    n, p = run(fermata=["9=3"])
    res.append(("calderón: compás 9 a 1/3", abs(n.bpm_at(n.bar_ticks(9)[0]) - sc.bpm_at(n.bar_ticks(9)[0]) / 3) < 0.1
                and abs(n.bpm_at(n.bar_ticks(10)[0]) - sc.bpm_at(n.bar_ticks(10)[0])) < 0.05))
    n, p = run(set=["1:100"])
    res.append(("set 1:100", abs(n.bpm_at(0) - 100) < 0.01))

    class S:  # noqa
        pass
    s = S()
    s.midi, s.output, s.window, s.resolution = src, os.path.join(d, "s2.mid"), 3, "beat"
    cmd_smooth(s)
    res.append(("smooth: notas intactas", _notes(s.output) == ref))
    L0, L1 = mido.MidiFile(src).length, mido.MidiFile(s.output).length
    res.append(("smooth: duración total dentro de ±5 %", abs(L1 - L0) / L0 < 0.05))
    for k, v in res:
        print(f"  {'✓' if v else '✗'} {k}")
    return 0 if all(v for _, v in res) else 1


def main():
    ap = argparse.ArgumentParser(description="Visualiza y edita el mapa de tempo de un MIDI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show"); p.add_argument("midi"); p.add_argument("--width", type=int, default=60)
    p.add_argument("--bars"); p.add_argument("--no-table", action="store_true")
    p.add_argument("--compact", action="store_true", help="omite compases con tempo igual al anterior")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("edit"); p.add_argument("midi"); p.add_argument("-o", "--output")
    p.add_argument("--set", nargs="+"); p.add_argument("--scale", nargs="+")
    p.add_argument("--rubato", nargs="+"); p.add_argument("--fermata", nargs="+")
    p.add_argument("--shape", default="arch", choices=("arch", "ritard", "accel", "hold"))
    p.add_argument("--resolution", default="beat", choices=("beat", "bar"))
    p.set_defaults(fn=cmd_edit)
    p = sub.add_parser("smooth"); p.add_argument("midi"); p.add_argument("-o", "--output")
    p.add_argument("--window", type=float, default=3)
    p.add_argument("--resolution", default="beat", choices=("beat", "bar"))
    p.set_defaults(fn=cmd_smooth)
    p = sub.add_parser("export"); p.add_argument("midi")
    p.add_argument("--json"); p.add_argument("--csv"); p.add_argument("--yaml")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    try:
        sys.exit(a.fn(a))
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e.filename}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, KeyError) as e:
        print(f"[ERROR] argumento no válido: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
