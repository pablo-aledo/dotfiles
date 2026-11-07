#!/usr/bin/env python3
"""
yaml_orquestar.py — YAML Orquestar: comprobación de orquestación y estadísticas reales
de una partitura YAML (o de un MIDI) ANTES de entregar la obra
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
El flujo YAML → MIDI (generar_midi.py) → auditoría (playability_auditor.py) obligaba a
ciclos de «generar, auditar, corregir» con 14 pistas decididas a mano, y las
estadísticas de generar_midi.py (--estadisticas / --preview) cuentan estructura y
acordes, no las notas realmente escritas por pista (con ciertos YAML informan 0 notas).
Esta herramienta cierra ese hueco sin duplicar lógica (integración por subprocess,
como el resto del ecosistema): materializa el YAML en un MIDI TEMPORAL (nunca ensucia
tu carpeta de salida) y sobre él ofrece:

  check     Rangos, respiración, polifonía y demás comprobaciones del auditor,
            traducidas a las pistas y compases del YAML, más avisos de pistas
            vacías. Código de salida 1 si hay errores (apto para scripts/CI).
  suggest   Para cada pista, qué instrumentos encajan MEJOR con su registro real y
            su polifonía, y si el programa GM asignado es una mala elección.
  stats     Notas, rango, pitch medio, polifonía máxima y mapa de densidad por
            compás de cada pista (rejilla de texto), sin abrir un DAW.
  selftest  Verifica el flujo con una partitura sintética.

Acepta indistintamente un .yaml/.yml (se genera el MIDI temporal con
generar_midi.py --completa) o un .mid ya generado.

Requiere, en la misma carpeta (o --suite-dir): generar_midi.py y, para `check`,
playability_auditor.py.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  check    ENTRADA [--instrument N=INSTR ...] [--profile estandar|estricto|tolerante]
                   [--max-issues 5] [--suite-dir DIR]
  suggest  ENTRADA [--top 3] [--json F] [--suite-dir DIR]
  stats    ENTRADA [--cols 64] [--json F] [--suite-dir DIR]
  selftest

EJEMPLOS
  python yaml_orquestar.py check partitura.yaml
  python yaml_orquestar.py check obra.mid --instrument 7=cuerdas 8=viento_metal_gen
  python yaml_orquestar.py suggest partitura.yaml --top 3
  python yaml_orquestar.py stats partitura.yaml --cols 80
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import mido

VERSION = "1.0"
HERE = Path(__file__).resolve().parent
GLYPHS = " ·░▒▓█"

# instrumento: (clave, nombre, rango_físico, zona_cómoda, monofónico)
INSTR = [
    ("violin", "Violín", (55, 103), (55, 96), False),
    ("viola", "Viola", (48, 88), (48, 81), False),
    ("violonchelo", "Violonchelo", (36, 81), (36, 76), False),
    ("contrabajo", "Contrabajo", (28, 67), (28, 60), True),
    ("flauta", "Flauta", (60, 96), (62, 91), True),
    ("oboe", "Oboe", (58, 91), (60, 86), True),
    ("clarinete", "Clarinete", (50, 91), (52, 86), True),
    ("fagot", "Fagot", (34, 72), (36, 67), True),
    ("trompa", "Trompa", (34, 77), (41, 72), True),
    ("trompeta", "Trompeta", (54, 82), (58, 79), True),
    ("trombon", "Trombón", (40, 72), (41, 67), True),
    ("tuba", "Tuba", (28, 58), (30, 53), True),
    ("arpa", "Arpa", (24, 103), (31, 96), False),
    ("piano", "Piano", (21, 108), (28, 103), False),
    ("cuerdas", "Sección de cuerdas", (36, 96), (40, 88), False),
    ("metales", "Sección de metales", (35, 86), (41, 79), False),
]
# programa GM (0-indexado) → clave
GM = {40: "violin", 41: "viola", 42: "violonchelo", 43: "contrabajo", 48: "cuerdas",
      49: "cuerdas", 73: "flauta", 68: "oboe", 71: "clarinete", 70: "fagot",
      60: "trompa", 56: "trompeta", 57: "trombon", 58: "tuba", 46: "arpa", 0: "piano",
      1: "piano", 59: "trompeta", 61: "metales", 45: "cuerdas", 44: "cuerdas"}


def _c(code):
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "GRN": "\033[92m",
            "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}.get(code, "")


# ══════════════════════════════════════════════════════════════════════════════
#  MATERIALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def materialize(entrada, suite_dir):
    """→ (ruta_midi, directorio_temporal|None, log). Para .yaml genera un MIDI temporal."""
    p = Path(entrada)
    if not p.exists():
        raise FileNotFoundError(entrada)
    if p.suffix.lower() in (".mid", ".midi"):
        return str(p), None, ""
    gm = Path(suite_dir) / "generar_midi.py"
    if not gm.exists():
        raise FileNotFoundError(str(gm))
    tmp = tempfile.mkdtemp(prefix="yaml_orq_")
    r = subprocess.run([sys.executable, str(gm), "--yaml", str(p.resolve()),
                        "--output", tmp + "/", "--completa"],
                       capture_output=True, text=True, cwd=str(suite_dir))
    found = sorted(glob.glob(os.path.join(tmp, "*_COMPLETA.mid"))) or \
        sorted(glob.glob(os.path.join(tmp, "*.mid")))
    if not found:
        raise RuntimeError("generar_midi.py no produjo ningún MIDI:\n" + (r.stdout + r.stderr)[-1500:])
    return found[0], tmp, r.stdout + r.stderr


def read_tracks(path):
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    ts = (4, 4)
    out = []
    for idx, tr in enumerate(mid.tracks):
        t, open_, notes, name, prog = 0, defaultdict(list), [], "", None
        drum = False
        for m in tr:
            t += m.time
            if m.type == "track_name" and not name:
                name = m.name
            elif m.type == "time_signature" and idx == 0 and t == 0:
                ts = (m.numerator, m.denominator)
            elif m.type == "program_change" and prog is None:
                prog = m.program
            elif m.type == "note_on" and m.velocity > 0:
                open_[(m.channel, m.note)].append(t)
                drum = drum or m.channel == 9
            elif m.type in ("note_off", "note_on"):
                q = open_.get((m.channel, m.note))
                if q:
                    s = q.pop(0)
                    notes.append((s, max(t, s + 1), m.note))
        out.append({"index": idx, "name": name, "program": prog, "drum": drum,
                    "notes": sorted(notes)})
    tpbar = max(1, int(tpb * 4 * ts[0] / ts[1]))
    end = max((n[1] for t in out for n in t["notes"]), default=0)
    return out, tpb, tpbar, max(1, -(-end // tpbar))


def max_poly(notes):
    ev = []
    for s, e, _ in notes:
        ev += [(s, 1), (e, -1)]
    ev.sort(key=lambda x: (x[0], x[1]))
    cur = best = 0
    for _, k in ev:
        cur += k
        best = max(best, cur)
    return best


def nn(p):
    return ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"][p % 12] + str(p // 12 - 1)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECK
# ══════════════════════════════════════════════════════════════════════════════

def cmd_check(a):
    path, tmp, _ = materialize(a.entrada, a.suite_dir)
    aud = Path(a.suite_dir) / "playability_auditor.py"
    if not aud.exists():
        print(f"[ERROR] no se encuentra {aud}", file=sys.stderr)
        return 2
    jpath = os.path.join(tmp or tempfile.mkdtemp(), "aud.json")
    cmd = [sys.executable, str(aud), path, "--quiet", "--json", jpath, "--profile", a.profile]
    if a.instrument:
        cmd += ["--instrument"] + a.instrument
    subprocess.run(cmd, capture_output=True, text=True, cwd=str(a.suite_dir))
    rep = json.load(open(jpath, encoding="utf-8"))
    tracks, tpb, tpbar, nbars = read_tracks(path)
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n  {B}YAML ORQUESTAR v{VERSION} · check · {os.path.basename(a.entrada)}{R}   "
          f"({nbars} compases, {sum(1 for t in tracks if t['notes'])} pistas con notas)")
    tot_err = 0
    empty = [t for t in tracks if not t["notes"] and t["name"] and t["index"] > 0]
    for t in rep["tracks"]:
        errs = [i for i in t["issues"] if i["severity"] == "ERROR"]
        warns = [i for i in t["issues"] if i["severity"] != "ERROR"]
        tot_err += len(errs)
        col = _c("GRN") if not errs else _c("RED")
        extra = f"  {_c('YEL')}ℹ {t['auto_note'][:90]}…{R}" if t.get("auto_note") else ""
        print(f"\n  {col}[{t['index']}]{R} {t['name']} · {t['instrument_es']} · {t['n_notes']} notas · "
              f"{t['score']:.2f}  errores {len(errs)}  avisos {len(warns)}{extra}")
        by = defaultdict(list)
        for i in errs:
            by[i["check"]].append(i)
        for chk, items in by.items():
            bars = sorted({i["bar"] for i in items if i.get("bar")})
            print(f"      ✗ {chk}: {len(items)}× · cc {', '.join(map(str, bars[:a.max_issues]))}"
                  f"{' …' if len(bars) > a.max_issues else ''} — {items[0]['msg']}")
    for t in empty:
        print(f"\n  {_c('YEL')}⚠ pista «{t['name']}» sin notas (¿capa sin asignar en el YAML?){R}")
    print(f"\n  {B}GLOBAL{R}  tocabilidad {rep['global_score']:.2f}   errores {tot_err}\n")
    return 1 if tot_err else 0


# ══════════════════════════════════════════════════════════════════════════════
#  SUGGEST
# ══════════════════════════════════════════════════════════════════════════════

def fit(notes, inst):
    _, _, rng, sweet, mono = inst
    n = len(notes)
    in_sweet = sum(1 for _, _, p in notes if sweet[0] <= p <= sweet[1]) / n
    in_range = sum(1 for _, _, p in notes if rng[0] <= p <= rng[1]) / n
    return in_sweet, in_range


def suggest_for(track, top):
    notes = track["notes"]
    poly = max_poly(notes)
    mean_p = sum(p for _, _, p in notes) / len(notes)
    cur = GM.get(track.get("program"))
    rows = []
    for inst in INSTR:
        if inst[0] in ("piano", "arpa") and poly < 3 and cur != inst[0]:
            continue                                  # abarcan casi todo: no informan en líneas
        sw, rg = fit(notes, inst)
        lo, hi = inst[3]
        center = 1 - min(1.0, abs(mean_p - (lo + hi) / 2) / ((hi - lo) / 2))
        score = 0.55 * sw + 0.25 * rg + 0.20 * center
        if inst[4] and poly > 1:                      # monofónico con acordes
            score *= 0.4
        if inst[0] in ("violin", "viola", "violonchelo") and poly > 2:
            score *= 0.6                              # un solista de cuerda no hace acordes de 3+
        if inst[0] in ("cuerdas", "metales") and poly < 3:
            score *= 0.9                              # las secciones se reservan para acordes
        rows.append((score, inst[0], inst[1], sw, rg))
    rows.sort(reverse=True)
    return poly, rows[:top], {r[1]: r for r in rows}


def cmd_suggest(a):
    path, tmp, _ = materialize(a.entrada, a.suite_dir)
    tracks, tpb, tpbar, nbars = read_tracks(path)
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n  {B}YAML ORQUESTAR v{VERSION} · suggest · {os.path.basename(a.entrada)}{R}")
    out = []
    for t in tracks:
        if not t["notes"] or t["drum"]:
            continue
        ps = [p for _, _, p in t["notes"]]
        poly, best, allr = suggest_for(t, a.top)
        cur = GM.get(t["program"])
        cur_row = allr.get(cur)
        flag = ""
        if cur_row and best and best[0][1] != cur and best[0][0] - cur_row[0] > 0.12:
            flag = f"  {_c('YEL')}⚠ el programa asignado encaja peor ({cur_row[0]:.2f}){R}"
        if t["program"] is not None and cur is None:
            flag += f"  {G}(programa GM {t['program']} no catalogado){R}"
        print(f"\n  [{t['index']}] {t['name']}  {G}{nn(min(ps))}–{nn(max(ps))} · polifonía máx. {poly} · "
              f"{len(ps)} notas · programa {t['program']}{R}{flag}")
        for score, key, nm, sw, rg in best:
            mark = "→" if key == cur else " "
            print(f"     {mark} {nm:<20} encaje {score:.2f}  (en zona cómoda {sw:.0%}, en rango {rg:.0%})")
        out.append({"track": t["index"], "name": t["name"], "program": t["program"],
                    "current": cur, "max_poly": poly,
                    "suggestions": [{"instrument": k, "score": round(s, 3)} for s, k, *_ in best]})
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  STATS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_stats(a):
    path, tmp, _ = materialize(a.entrada, a.suite_dir)
    tracks, tpb, tpbar, nbars = read_tracks(path)
    B, R, G = _c("B"), _c("R"), _c("G")
    cols = min(a.cols, nbars)
    print(f"\n  {B}YAML ORQUESTAR v{VERSION} · stats · {os.path.basename(a.entrada)}{R}   "
          f"{nbars} compases · {sum(len(t['notes']) for t in tracks)} notas · 1 columna = "
          f"{nbars / cols:.1f} cc")
    print(f"  {G}{'#':>2} {'pista':<24}{'notas':>6} {'rango':<11}{'medio':<6}{'pol':>3} {'cob':>4}  densidad por compás{R}")
    out = []
    for t in tracks:
        if not t["notes"]:
            continue
        per_bar = [0] * nbars
        for s, e, p in t["notes"]:
            per_bar[min(s // tpbar, nbars - 1)] += 1
        ps = [p for _, _, p in t["notes"]]
        cov = sum(1 for x in per_bar if x) / nbars
        mx = max(per_bar) or 1
        line = ""
        for c in range(cols):
            i0, i1 = int(c * nbars / cols), max(int((c + 1) * nbars / cols), int(c * nbars / cols) + 1)
            v = max(per_bar[i0:i1])
            line += GLYPHS[0] if v == 0 else GLYPHS[1 + min(4, int(v / mx * 4.999))]
        print(f"  {t['index']:>2} {t['name'][:23]:<24}{len(ps):>6} {nn(min(ps)) + '–' + nn(max(ps)):<11}"
              f"{nn(int(round(sum(ps) / len(ps)))):<6}{max_poly(t['notes']):>3} {cov:>4.0%}  {line}")
        out.append({"track": t["index"], "name": t["name"], "notes": len(ps), "min": min(ps),
                    "max": max(ps), "mean": round(sum(ps) / len(ps), 1),
                    "max_poly": max_poly(t["notes"]), "coverage": round(cov, 3),
                    "per_bar": per_bar})
    tot = [0] * nbars
    for t in tracks:
        for s, e, p in t["notes"]:
            tot[min(s // tpbar, nbars - 1)] += 1
    mx = max(tot) or 1
    line = ""
    for c in range(cols):
        i0, i1 = int(c * nbars / cols), max(int((c + 1) * nbars / cols), int(c * nbars / cols) + 1)
        v = max(tot[i0:i1])
        line += GLYPHS[0] if v == 0 else GLYPHS[1 + min(4, int(v / mx * 4.999))]
    print(f"  {G}{'':>2} {'TOTAL':<24}{sum(tot):>6} {'':<11}{'':<6}{'':>3} {'':>4}  {R}{line}")
    silent = [i + 1 for i, v in enumerate(tot) if v == 0]
    if silent:
        print(f"\n  {G}compases sin ninguna nota: {silent[:20]}{' …' if len(silent) > 20 else ''}{R}")
    if a.json:
        json.dump(out, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print()
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  SELFTEST
# ══════════════════════════════════════════════════════════════════════════════

def _synthetic_midi(path, bad_cello=False):
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    meta.append(mido.MetaMessage("set_tempo", tempo=500000))
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
            t.append(mido.Message("note_on", note=p, velocity=80 if on else 0, channel=chan, time=tt - last))
            last = tt
        mid.tracks.append(t)
    B = 1920
    mk("Violin", 40, 0, [(b * B + k * 480, 480, 69 + (k % 3)) for b in range(8) for k in range(4)])
    cello_base = 28 if bad_cello else 40               # E1 está por debajo del cello (C2)
    mk("Violoncello", 42, 1, [(b * B, B, cello_base + b % 3) for b in range(8)])
    mk("Cuerdas", 48, 2, [(b * B, B, p) for b in range(8) for p in (55, 59, 64)])
    mk("Vacia", 71, 3, [])
    mid.save(path)


def cmd_selftest(a):
    d = tempfile.mkdtemp()
    ok_p, bad_p = os.path.join(d, "ok.mid"), os.path.join(d, "bad.mid")
    _synthetic_midi(ok_p)
    _synthetic_midi(bad_p, bad_cello=True)
    res = []

    class A:  # noqa
        pass

    def args(p, **kw):
        o = A()
        o.entrada, o.suite_dir = p, str(HERE)
        o.profile, o.instrument, o.max_issues, o.top, o.json, o.cols = "estandar", None, 5, 3, None, 40
        for k, v in kw.items():
            setattr(o, k, v)
        return o
    has_aud = (HERE / "playability_auditor.py").exists()
    if has_aud:
        res.append(("check: partitura correcta → código 0", cmd_check(args(ok_p)) == 0))
        res.append(("check: cello demasiado grave → código 1", cmd_check(args(bad_p)) == 1))
    else:
        print("  (playability_auditor.py no está en la carpeta: se omite `check`)")
    res.append(("stats", cmd_stats(args(ok_p)) == 0))
    tracks, *_ = read_tracks(bad_p)
    cello = [t for t in tracks if t["name"] == "Violoncello"][0]
    _, best, _ = suggest_for(cello, 3)
    res.append(("suggest: el registro de la pista «cello» (E1–G1) NO encaja con violonchelo",
                best[0][1] != "violonchelo"))
    _, best, _ = suggest_for(cello, 2)                # E1–G1 → solo contrabajo/tuba tienen sentido
    res.append(("suggest: registro E1–G1 → contrabajo/tuba primero", best[0][1] in ("contrabajo", "tuba")))
    hi = {"program": None, "notes": [(i * 480, i * 480 + 480, 90 + i % 4) for i in range(20)]}
    _, best, _ = suggest_for(hi, 2)                   # F#6–A6 → violín/flauta
    res.append(("suggest: registro muy agudo → violín/flauta primero", best[0][1] in ("violin", "flauta")))
    ch = [t for t in tracks if t["name"] == "Cuerdas"][0]
    _, best, _ = suggest_for(ch, 3)
    res.append(("suggest: acordes → secciones/teclado, no monofónicos",
                best[0][1] in ("cuerdas", "piano", "arpa", "metales", "viola", "violin")))
    res.append(("suggest CLI", cmd_suggest(args(ok_p)) == 0))
    for k, v in res:
        print(f"  {'✓' if v else '✗'} {k}")
    return 0 if all(v for _, v in res) else 1


def main():
    ap = argparse.ArgumentParser(description="Comprobación de orquestación y estadísticas de un YAML/MIDI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("entrada")
        p.add_argument("--suite-dir", default=str(HERE))

    p = sub.add_parser("check"); common(p)
    p.add_argument("--instrument", nargs="+"); p.add_argument("--max-issues", type=int, default=5)
    p.add_argument("--profile", default="estandar", choices=("estandar", "estricto", "tolerante"))
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("suggest"); common(p); p.add_argument("--top", type=int, default=3)
    p.add_argument("--json"); p.set_defaults(fn=cmd_suggest)
    p = sub.add_parser("stats"); common(p); p.add_argument("--cols", type=int, default=64)
    p.add_argument("--json"); p.set_defaults(fn=cmd_stats)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    try:
        sys.exit(a.fn(a))
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
