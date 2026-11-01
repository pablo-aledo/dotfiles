#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                          MIDI METER  v1.0                                    ║
║          Inferencia y reparación del compás real de un MIDI                  ║
║                                                                              ║
║  Muchos MIDIs declaran un compás erróneo ("1/4", "1/8", varios cambios       ║
║  espurios) o ninguno. Como casi todas las herramientas cuentan compases      ║
║  a partir de ese metadato (harmonic_analyzer, form_analyzer, section_score,  ║
║  bass_line_composer, midi_slice, stitcher...), un compás falso distorsiona   ║
║  todo el flujo. Esta herramienta lo INFIERE de la música (periodicidad de    ║
║  los ataques, acentos por duración/registro grave/percusión y repetición     ║
║  entre compases) y puede reescribir el metadato.                             ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    infer obra.mid          Ranking de hipótesis con puntuación y confianza   ║
║    fix   obra.mid -o ok.mid  Reescribe el compás si el declarado es dudoso   ║
║    scan  carpeta/          Informe de un lote: declarado vs inferido         ║
║    selftest                Comprobaciones internas                           ║
║                                                                              ║
║  EJEMPLOS:                                                                   ║
║    python midi_meter.py infer obra.mid                                       ║
║    python midi_meter.py infer obra.mid --json -                              ║
║    python midi_meter.py fix obra.mid -o obra_ok.mid                          ║
║    python midi_meter.py fix obra.mid --inplace --min-conf 0.7                ║
║    python midi_meter.py scan midis/ --csv informe.csv --only-suspect         ║
║                                                                              ║
║  QUÉ SE INFIERE:                                                             ║
║    · La LONGITUD DEL COMPÁS en negras (2, 3, 4, 5 o 6) y el compás más       ║
║      probable con esa longitud (2/4, 3/4, 4/4, 5/4, 6/4).                    ║
║    · 3/4 y 6/8 (y 2/4 y 4/4) comparten longitud o múltiplo: se consideran    ║
║      COMPATIBLES y `fix` no cambia un 6/8 declarado por un 3/4.              ║
║    · Una estimación (experimental) del desplazamiento del primer tiempo      ║
║      fuerte (--phase muestra la fase en negras; anacrusis).                  ║
║                                                                              ║
║  CRITERIO DE `fix` (solo escribe si hay motivo):                             ║
║    · compás ausente, implausible (numerador 1, duración < 1.5 negras...),    ║
║      con varios cambios inestables, o                                        ║
║    · distinto del inferido con confianza ≥ --min-conf (default 0.75) y no    ║
║      compatible. Con --force siempre escribe el inferido.                    ║
║    · Reemplaza TODOS los time_signature por uno solo al inicio. No mueve     ║
║      ninguna nota.                                                           ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --candidates 2/4,3/4,4/4   Restringe las hipótesis                        ║
║    --min-bars N              Mínimo de compases para inferir (default 4)     ║
║    --json FILE|-             Salida JSON                                     ║
║    --no-drums                Ignora la percusión en el análisis              ║
║                                                                              ║
║  LIMITACIONES: asume un compás único para toda la obra (mezclas de           ║
║  compases se resuelven por mayoría); 2/4 vs 4/4 es ambiguo por naturaleza    ║
║  y se resuelve con una preferencia por 4/4; usa --candidates para forzar.    ║
║  · Mide la periodicidad en TICKS: si el ticks_per_beat del fichero es        ║
║    incoherente con la música (p.ej. tpb=72 con tresillos), el resultado      ║
║    describe el compás tal como lo verán las demás herramientas (5/4), no el  ║
║    de la partitura original.                                                 ║
║  · Shuffle/swing en 4/4 puede dar 6/4 (sensación de 12/8) con confianza      ║
║    media; `fix` solo reescribe si la confianza supera --min-conf.            ║
║  · Con pocos compases (< --min-bars) se niega a inferir.                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path

try:
    import mido
    import numpy as np
except ImportError:                                   # pragma: no cover
    sys.exit("ERROR: requiere mido y numpy (pip install mido numpy)")

VERSION = "1.0"
G = 12                      # pasos de rejilla por negra (admite binario y ternario)

# nombre, longitud en negras, prior
CANDS = {
    "2/4": (2.0, 0.00),
    "3/4": (3.0, 0.00),
    "4/4": (4.0, 0.06),
    "5/4": (5.0, -0.09),
    "6/4": (6.0, -0.05),
}
COMPATIBLE = {"3/4": {"6/8", "3/4", "6/4"}, "6/4": {"3/4", "6/8", "12/8", "6/4"},
              "4/4": {"4/4", "2/4", "2/2", "12/8"}, "2/4": {"2/4", "4/4", "2/2"},
              "5/4": {"5/4", "5/8", "10/8"}}


# ─────────────────────────────────────────────────────────────────────────────
#  Lectura
# ─────────────────────────────────────────────────────────────────────────────

def read_midi(path):
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat or 480
    ons = []
    declared = []
    for trk in mid.tracks:
        t = 0
        pend = {}
        for m in trk:
            t += m.time
            if m.type == "time_signature":
                declared.append((t / tpb, m.numerator, m.denominator))
            elif m.type == "note_on" and m.velocity > 0:
                pend.setdefault((m.channel, m.note), []).append((t, m.velocity))
            elif m.type in ("note_off", "note_on"):
                lst = pend.get((m.channel, m.note))
                if lst:
                    t0, v = lst.pop(0)
                    ons.append((t0 / tpb, max(t - t0, 1) / tpb, m.note, v, m.channel == 9))
    declared.sort()
    return ons, declared, tpb


def accent_curve(ons, use_drums=True):
    ons = [o for o in ons if use_drums or not o[4]]
    if not ons:
        return None
    T = int(math.ceil(max(o[0] + o[1] for o in ons) * G)) + G
    a = np.zeros(T)
    for b, d, p, v, drum in ons:
        i = int(round(b * G))
        if i >= T:
            continue
        if drum:
            w = {35: 1.6, 36: 1.6, 38: 1.0, 40: 1.0}.get(p, 0.35)
        else:
            w = 0.6 + min(d, 4.0) * 0.35 + (0.4 if p < 50 else 0.0)
        a[i] += w * (v / 100.0)
    return a


# ─────────────────────────────────────────────────────────────────────────────
#  Puntuación
# ─────────────────────────────────────────────────────────────────────────────

def _bars_matrix(sm, L, s):
    n = int(round(L * G))
    nb = (len(sm) - s) // n
    if nb < 3:
        return None
    return np.stack([sm[s + i * n: s + (i + 1) * n] for i in range(nb)])


def score_meter(a, L, phases="auto"):
    """(score, fase_en_negras, sim_consecutiva, prominencia_1er_tiempo)."""
    sm = a.copy()
    sm[1:] += 0.5 * a[:-1]
    sm[:-1] += 0.5 * a[1:]
    best = None
    n_ph = int(round(L * 2))
    for ph in range(n_ph):
        s = ph * (G // 2)
        M = _bars_matrix(sm, L, s)
        if M is None:
            continue
        nb = M.shape[0]
        nr = np.linalg.norm(M, axis=1) + 1e-9
        sim = float(np.mean([(M[i] @ M[i + 1]) / (nr[i] * nr[i + 1]) for i in range(nb - 1)]))
        prof = M.sum(axis=0)
        down = prof[:3].sum() / 3.0
        beats = [prof[int(k * G): int(k * G) + 3].sum() / 3.0 for k in range(1, int(L))]
        other = float(np.mean(beats)) if beats else 1e-9
        prom = down / (down + other + 1e-9)
        sc = 0.5 * sim + 0.5 * prom
        if best is None or sc > best[0]:
            best = (sc, ph * 0.5, sim, prom)
    return best


def hier_bonus(a, phase):
    """4/4: el 3.er pulso pesa más que el 2.º y el 4.º."""
    sm = a.copy()
    sm[1:] += 0.5 * a[:-1]
    sm[:-1] += 0.5 * a[1:]
    M = _bars_matrix(sm, 4.0, int(round(phase * G)))
    if M is None:
        return 0.0
    P = M.sum(axis=0)

    def pos(x):
        i = int(round(x * G))
        return P[max(i - 1, 0): i + 2].sum() / 3.0
    r = pos(2.0) / max(0.5 * (pos(1.0) + pos(3.0)), 1e-9)
    return 0.04 * float(np.clip(math.log2(max(r, 1e-3)), 0.0, 2.0))


def infer_meter(path, candidates=None, min_bars=4, use_drums=True):
    ons, declared, tpb = read_midi(path)
    a = accent_curve(ons, use_drums)
    info = {"file": str(path), "tool": "midi_meter", "version": VERSION,
            "declared": _fmt_declared(declared), "n_declared": len(declared)}
    if a is None:
        info.update(ok=False, reason="sin notas")
        return info
    names = candidates or list(CANDS)
    total_beats = max(o[0] + o[1] for o in ons)
    if total_beats < min_bars * 3.0:
        info.update(ok=False, reason=f"material insuficiente ({total_beats:.0f} negras, "
                                     f"se piden ≥ {min_bars * 3})")
        return info
    decl_prior = _declared_prior(declared, total_beats)
    rows = []
    for name in names:
        if name not in CANDS:
            raise ValueError(f"candidato no soportado: {name} (admite {', '.join(CANDS)})")
        L, prior = CANDS[name]
        if len(a) < L * G * max(min_bars, 3):
            continue
        r = score_meter(a, L)
        if r is None:
            continue
        sc, ph, sim, prom = r
        adj = sc + prior + decl_prior.get(name, 0.0)
        if name == "4/4":
            adj += hier_bonus(a, ph)
        rows.append({"meter": name, "bar_beats": L, "score": round(sc, 3),
                     "adjusted": round(adj, 3), "phase_beats": ph,
                     "similarity": round(sim, 3), "downbeat": round(prom, 3)})
    if not rows:
        info.update(ok=False, reason=f"material insuficiente (< {min_bars} compases)")
        return info
    rows.sort(key=lambda r: -r["adjusted"])
    best, second = rows[0], rows[1] if len(rows) > 1 else None
    margin = best["adjusted"] - (second["adjusted"] if second else best["adjusted"] - 0.1)
    fit = min(1.0, max(0.0, (best["score"] - 0.35) / 0.45))
    conf = round(float(np.clip(0.35 + 4.0 * margin, 0.0, 1.0) * (0.5 + 0.5 * fit)), 3)
    info.update(ok=True, inferred=best["meter"], bar_beats=best["bar_beats"],
                confidence=conf, phase_beats=best["phase_beats"], ranking=rows,
                n_bars=int(round(max(o[0] + o[1] for o in ons) / best["bar_beats"])))
    info["status"], info["reason"] = judge(info)
    return info


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnóstico del metadato declarado
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_declared(declared):
    if not declared:
        return None
    seen = []
    for _, n, d in declared:
        s = f"{n}/{d}"
        if s not in seen:
            seen.append(s)
    return "+".join(seen)


def _declared_prior(declared, total_beats, bonus=0.04):
    """Bonus a la hipótesis del compás declarado plausible que cubre más duración
    (el metadato parcialmente bueno también es evidencia)."""
    dur = {}
    for i, (t, n, d) in enumerate(declared):
        end = declared[i + 1][0] if i + 1 < len(declared) else total_beats
        if declared_plausible(n, d) and end > t:
            dur[(n, d)] = dur.get((n, d), 0.0) + (end - t)
    if not dur:
        return {}
    (n, d), _ = max(dur.items(), key=lambda kv: kv[1])
    key = f"{n}/{d}"
    for cand, fam in COMPATIBLE.items():
        if key == cand or key in fam:
            return {cand: bonus}
    return {}


def declared_plausible(num, den):
    if den not in (2, 4, 8, 16) or num < 2:
        return False
    return 1.5 <= num * 4.0 / den <= 8.0


def judge(info):
    """Devuelve (estado, motivo): ok | missing | implausible | unstable | mismatch | uncertain."""
    inferred = info["inferred"]
    conf = info["confidence"]
    decl = info["declared"]
    if not decl:
        return "missing", "el MIDI no declara compás"
    parts = decl.split("+")
    first = parts[0]
    num, den = (int(x) for x in first.split("/"))
    if not declared_plausible(num, den):
        return "implausible", f"compás declarado implausible ({first})"
    if len(parts) > 1 and info["n_declared"] > 2:
        return "unstable", f"{info['n_declared']} cambios de compás declarados ({decl})"
    if first == inferred or first in COMPATIBLE.get(inferred, ()):
        return "ok", f"declarado {first} compatible con el inferido {inferred}"
    L_decl = num * 4.0 / den
    L_inf = info["bar_beats"]
    if abs(L_decl - L_inf) < 1e-9:
        return "ok", f"declarado {first} con la misma longitud que {inferred}"
    return ("mismatch" if conf >= 0.5 else "uncertain",
            f"declarado {first}, inferido {inferred} (confianza {conf:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
#  Reparación
# ─────────────────────────────────────────────────────────────────────────────

def write_fixed(src, dst, num, den):
    mid = mido.MidiFile(src)
    for trk in mid.tracks:
        acc = 0
        keep = []
        for m in trk:
            if m.type == "time_signature":
                acc += m.time          # conserva el tiempo absoluto de los demás
                continue
            keep.append(m.copy(time=m.time + acc))
            acc = 0
        if acc and keep:
            keep[-1] = keep[-1].copy(time=keep[-1].time + acc)
        trk[:] = keep
    target = next((t for t in mid.tracks if any(m.type == "set_tempo" for m in t)),
                  mid.tracks[0] if mid.tracks else None)
    if target is None:
        raise ValueError("el MIDI no tiene pistas")
    ins = mido.MetaMessage("time_signature", numerator=num, denominator=den,
                           clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0)
    # justo después de los metas iniciales a tick 0
    idx = 0
    while idx < len(target) and target[idx].time == 0 and target[idx].is_meta:
        idx += 1
    target.insert(idx, ins)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    mid.save(dst)


def should_fix(info, min_conf, force):
    if not info.get("ok"):
        return False, info.get("reason", "sin inferencia")
    if force:
        return True, "--force"
    st = info["status"]
    if st in ("missing", "implausible", "unstable"):
        if info["confidence"] >= min(min_conf, 0.3):
            return True, info["reason"]
        return False, f"{info['reason']}, pero confianza baja ({info['confidence']:.2f})"
    if st == "mismatch" and info["confidence"] >= min_conf:
        return True, info["reason"]
    return False, info["reason"]


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def print_info(info, show_phase=False):
    print(f"\n  MIDI METER v{VERSION} · {info['file']}")
    if not info.get("ok"):
        print(f"  Sin inferencia: {info.get('reason')}")
        return
    print(f"  Declarado : {info['declared'] or '(ninguno)'}")
    print(f"  Inferido  : {info['inferred']}  (confianza {info['confidence']:.2f}, "
          f"~{info['n_bars']} compases)")
    print(f"  Estado    : {info['status']} — {info['reason']}")
    print(f"\n  {'hipótesis':<10}{'ajust.':>8}{'score':>8}{'simil.':>8}{'1er t.':>8}"
          + ("  fase" if show_phase else ""))
    for r in info["ranking"]:
        print(f"  {r['meter']:<10}{r['adjusted']:>8.3f}{r['score']:>8.3f}"
              f"{r['similarity']:>8.3f}{r['downbeat']:>8.3f}"
              + (f"  {r['phase_beats']:>4}" if show_phase else ""))


def _cands(a):
    return [c.strip() for c in a.candidates.split(",")] if getattr(a, "candidates", None) else None


def _dump_json(obj, dest):
    text = json.dumps(obj, ensure_ascii=False, indent=1)
    if dest == "-":
        print(text)
    else:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(text, encoding="utf-8")
        print(f"  JSON: {dest}", file=sys.stderr)


def cmd_infer(a):
    info = infer_meter(a.midi, _cands(a), a.min_bars, not a.no_drums)
    if a.json == "-":
        _dump_json(info, "-")
    else:
        print_info(info, a.phase)
        if a.json:
            _dump_json(info, a.json)
    return 0 if info.get("ok") else 1


def cmd_fix(a):
    info = infer_meter(a.midi, _cands(a), a.min_bars, not a.no_drums)
    do, why = should_fix(info, a.min_conf, a.force)
    print_info(info)
    if not do:
        print(f"\n  Sin cambios: {why}")
        return 0 if info.get("ok") else 1
    num, den = (int(x) for x in info["inferred"].split("/"))
    if a.inplace:
        dst = a.midi
    else:
        dst = a.output or str(Path(a.midi).with_name(Path(a.midi).stem + "_meter.mid"))
    write_fixed(a.midi, dst, num, den)
    print(f"\n  Compás reescrito → {info['inferred']}  ({why})\n  → {dst}")
    return 0


def cmd_scan(a):
    root = Path(a.dir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".mid", ".midi"))
    if a.limit:
        files = files[: a.limit]
    rows = []
    for p in files:
        try:
            info = infer_meter(str(p), _cands(a), a.min_bars, not a.no_drums)
        except Exception as e:                        # MIDI corrupto
            info = {"file": str(p), "ok": False, "reason": f"error: {e}", "declared": None}
        rows.append(info)
    shown = [r for r in rows if not a.only_suspect or r.get("status") not in ("ok", None)]
    print(f"\n  {'archivo':<46}{'declarado':<14}{'inferido':<9}{'conf':>5}  estado")
    for r in shown:
        name = os.path.relpath(r["file"], root)
        if len(name) > 45:
            name = "…" + name[-44:]
        if r.get("ok"):
            print(f"  {name:<46}{(r['declared'] or '-')[:13]:<14}{r['inferred']:<9}"
                  f"{r['confidence']:>5.2f}  {r['status']}")
        else:
            print(f"  {name:<46}{(r.get('declared') or '-')[:13]:<14}{'?':<9}{'':>5}  {r.get('reason')}")
    cnt = {}
    for r in rows:
        k = r.get("status", "sin-inferencia") if r.get("ok") else "sin-inferencia"
        cnt[k] = cnt.get(k, 0) + 1
    print("\n  Resumen: " + "  ".join(f"{k}={v}" for k, v in sorted(cnt.items())) + f"  (total {len(rows)})")
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["file", "declared", "inferred", "confidence", "status", "reason"])
            for r in rows:
                w.writerow([r["file"], r.get("declared") or "", r.get("inferred", ""),
                            r.get("confidence", ""), r.get("status", ""), r.get("reason", "")])
        print(f"  CSV: {a.csv}")
    if a.json:
        _dump_json(rows, a.json)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Selftest
# ─────────────────────────────────────────────────────────────────────────────

def _synth(path, beats_per_bar, n_bars=16, ts=None, tpb=480, drums=False, comp=False):
    """Pieza sintética: bajo en el 1er tiempo, acorde largo, melodía por negras
    con acento y un patrón que se repite cada compás (con variación suave)."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=500000))
    if ts:
        tr.append(mido.MetaMessage("time_signature", numerator=ts[0], denominator=ts[1]))
    ev = []
    for b in range(n_bars):
        t0 = b * beats_per_bar
        ev.append((t0, 36 + (0, 5, 7, 0)[b % 4], 1.0 + 0.5 * (beats_per_bar > 3), 95, 0))
        for k in range(int(beats_per_bar)):
            ev.append((t0 + k, 64 + ((b + k) % 3) * 2, 0.9 if k else 0.9, 80 if k else 100, 0))
        if drums:
            ev.append((t0, 36, 0.25, 100, 9))
            ev.append((t0 + beats_per_bar / 2, 38, 0.25, 90, 9))
    flat = []
    for t, p, d, v, ch in ev:
        flat.append((int(t * tpb), 1, p, v, ch))
        flat.append((int((t + d) * tpb), 0, p, 0, ch))
    flat.sort()
    last = 0
    for t, on, p, v, ch in flat:
        tr.append(mido.Message("note_on", note=p, velocity=v if on else 0, channel=ch, time=t - last))
        last = t
    mid.save(path)


def cmd_selftest(a):
    ok = True

    def check(c, msg):
        nonlocal ok
        print(("  ✔ " if c else "  ✘ ") + msg)
        ok &= bool(c)

    with tempfile.TemporaryDirectory() as d:
        for bpb, expect in ((3, "3/4"), (4, "4/4"), (5, "5/4")):
            p = os.path.join(d, f"s{bpb}.mid")
            _synth(p, bpb)
            r = infer_meter(p)
            check(r.get("ok") and r["inferred"] == expect, f"sintético {expect} → {r.get('inferred')}")
        p = os.path.join(d, "bad.mid")
        _synth(p, 3, ts=(1, 4))
        r = infer_meter(p)
        check(r["status"] == "implausible", f"1/4 declarado con música en 3/4 → {r['status']}")
        out = os.path.join(d, "fixed.mid")
        do, _ = should_fix(r, 0.75, False)
        check(do, "fix decide reparar el 1/4")
        write_fixed(p, out, 3, 4)
        r2 = infer_meter(out)
        check(r2["declared"] == "3/4" and r2["status"] == "ok", f"tras fix declara {r2['declared']} → {r2['status']}")
        n0 = sum(1 for t in mido.MidiFile(p).tracks for m in t if m.type == "note_on")
        n1 = sum(1 for t in mido.MidiFile(out).tracks for m in t if m.type == "note_on")
        check(n0 == n1, "fix conserva todas las notas")
        p = os.path.join(d, "none.mid")
        _synth(p, 4)
        r = infer_meter(p)
        check(r["status"] == "missing", "sin compás declarado → missing")
        p = os.path.join(d, "ok.mid")
        _synth(p, 4, ts=(4, 4))
        r = infer_meter(p)
        check(r["status"] == "ok" and not should_fix(r, 0.75, False)[0], "4/4 correcto: fix no toca")
        p = os.path.join(d, "tiny.mid")
        _synth(p, 4, n_bars=2)
        check(not infer_meter(p).get("ok"), "material insuficiente se detecta")
    print("  SELFTEST OK" if ok else "  SELFTEST FALLÓ")
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(prog="midi_meter.py",
                                 description="Inferencia y reparación del compás de un MIDI")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--candidates", default=None)
        p.add_argument("--min-bars", type=int, default=4)
        p.add_argument("--no-drums", action="store_true")

    p = sp.add_parser("infer", help="Ranking de hipótesis de compás")
    p.add_argument("midi")
    common(p)
    p.add_argument("--json", default=None, metavar="FILE|-")
    p.add_argument("--phase", action="store_true", help="Muestra la fase estimada del 1er tiempo")
    p.set_defaults(fn=cmd_infer)

    p = sp.add_parser("fix", help="Reescribe el metadato de compás si es dudoso")
    p.add_argument("midi")
    common(p)
    p.add_argument("-o", "--output", default=None)
    p.add_argument("--inplace", action="store_true")
    p.add_argument("--min-conf", type=float, default=0.75)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_fix)

    p = sp.add_parser("scan", help="Informe de un lote")
    p.add_argument("dir")
    common(p)
    p.add_argument("--csv", default=None)
    p.add_argument("--json", default=None, metavar="FILE|-")
    p.add_argument("--only-suspect", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_scan)

    p = sp.add_parser("selftest", help="Comprobaciones internas")
    p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, OSError, EOFError, KeyError, IndexError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
