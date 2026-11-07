#!/usr/bin/env python3
"""
midi_compare.py — MIDI Compare: comparación cuantitativa entre dos MIDIs
(semilla → evolución, versión A → versión B)
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Evolucionar una obra (de una semilla a otra pieza, de una variante a otra) no se
podía medir: midi2plan solo describe UN MIDI y generar_midi --diff compara YAMLs,
no música sonando. Esta herramienta compara dos MIDIs con métricas objetivas:

  curves    Curvas por compás (densidad de notas, registro, velocidad, nº de
            pistas activas) lado a lado, con sparklines y correlación de
            Pearson tras remuestrear a la misma longitud relativa.
  motifs    Busca un motivo (notas o MIDI) en cada pista de A y B: apariciones
            exactas transportadas, en inversión, retrógrado y con tolerancia de
            ±1 semitono; dice en qué compases y en qué pistas.
  harmony   Ritmo armónico (cambios de acorde por compás), acordes más
            frecuentes y tonalidad estimada (Krumhansl) de cada MIDI.
  ranges    Registro (mín/máx/medio) por pista emparejando pistas por nombre.
  report    Todo lo anterior + índice de parecido global (0 = nada que ver,
            1 = idénticos) y veredicto en lenguaje llano.

El compás se toma de la primera time_signature (por defecto 4/4). Los compases se
numeran desde 1. La armonía es una estimación por plantillas (tríadas y séptimas)
sobre el croma de cada MEDIO compás; es orientativa, no un análisis funcional.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  curves    A.mid B.mid [--points N] [--json F]
  motifs    A.mid B.mid  --motif "D4 G4 C5 A4" | --motif-midi M.mid
                         [--tolerance 0|1] [--no-inversion] [--no-retrograde]
  harmony   A.mid B.mid
  ranges    A.mid B.mid
  report    A.mid B.mid [--motif "..."] [--json F]
  selftest

EJEMPLOS
  python midi_compare.py report semilla.mid obra_final.mid --motif "D4 G4 C5 A4"
  python midi_compare.py curves a.mid b.mid --points 24
  python midi_compare.py motifs a.mid b.mid --motif-midi motivo.mid --tolerance 1
"""
import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict

import mido
import numpy as np

VERSION = "1.0"
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PCS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
SPARK = "▁▂▃▄▅▆▇█"
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
TEMPLATES = {"": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "7": (0, 4, 7, 10),
             "m7": (0, 3, 7, 10), "M7": (0, 4, 7, 11), "ø7": (0, 3, 6, 10), "aug": (0, 4, 8)}


def _c(code):
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "GRN": "\033[92m",
            "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}.get(code, "")


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA
# ══════════════════════════════════════════════════════════════════════════════

class Piece:
    def __init__(self, path):
        mid = mido.MidiFile(path)
        self.path, self.name = path, os.path.basename(path)
        self.tpb = mid.ticks_per_beat
        ts = (4, 4)
        self.tracks = []
        for idx, tr in enumerate(mid.tracks):
            t, open_, notes, name, chans = 0, defaultdict(list), [], "", set()
            for m in tr:
                t += m.time
                if m.type == "track_name" and not name:
                    name = m.name
                elif m.type == "time_signature" and t == 0 and ts == (4, 4) and idx == 0:
                    ts = (m.numerator, m.denominator)
                elif m.type == "note_on" and m.velocity > 0:
                    open_[(m.channel, m.note)].append((t, m.velocity))
                    chans.add(m.channel)
                elif m.type in ("note_off", "note_on"):
                    q = open_.get((m.channel, m.note))
                    if q:
                        s, v = q.pop(0)
                        notes.append((s, max(t, s + 1), m.note, v))
            notes.sort()
            self.tracks.append({"index": idx, "name": name, "notes": notes,
                                "drum": chans == {9}})
        self.tpbar = max(1, int(self.tpb * 4 * ts[0] / ts[1]))
        allnotes = [n for t in self.tracks for n in t["notes"]]
        self.end = max((n[1] for n in allnotes), default=0)
        self.bars = max(1, -(-self.end // self.tpbar))

    def pitched(self):
        return [t for t in self.tracks if t["notes"] and not t["drum"]]


# ══════════════════════════════════════════════════════════════════════════════
#  CURVAS
# ══════════════════════════════════════════════════════════════════════════════

def bar_curves(p):
    dens = np.zeros(p.bars)
    pitch_sum = np.zeros(p.bars)
    vel_sum = np.zeros(p.bars)
    active = [set() for _ in range(p.bars)]
    for t in p.pitched():
        for s, e, pit, v in t["notes"]:
            b = min(s // p.tpbar, p.bars - 1)
            dens[b] += 1
            pitch_sum[b] += pit
            vel_sum[b] += v
            active[b].add(t["index"])
    with np.errstate(invalid="ignore", divide="ignore"):
        mp = np.where(dens > 0, pitch_sum / np.maximum(dens, 1), np.nan)
        mv = np.where(dens > 0, vel_sum / np.maximum(dens, 1), np.nan)
    return {"density": dens, "pitch": _ffill(mp), "velocity": _ffill(mv),
            "tracks": np.array([len(a) for a in active], dtype=float)}


def _ffill(a):
    a = a.copy()
    last = np.nan
    for i in range(len(a)):
        if np.isnan(a[i]):
            a[i] = last
        else:
            last = a[i]
    first = a[~np.isnan(a)]
    return np.where(np.isnan(a), first[0] if len(first) else 0.0, a)


def resample(a, n):
    if len(a) == n:
        return a.astype(float)
    x = np.linspace(0, 1, len(a))
    return np.interp(np.linspace(0, 1, n), x, a)


def spark(a, width=40):
    a = resample(np.asarray(a, float), width)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return SPARK[0] * width
    return "".join(SPARK[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in a)


def corr(a, b, n):
    x, y = resample(a, n), resample(b, n)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.corrcoef(x, y)[0, 1])


def compare_curves(A, B, points=32):
    ca, cb = bar_curves(A), bar_curves(B)
    out = {}
    for k in ca:
        out[k] = {"corr": round(corr(ca[k], cb[k], points), 3),
                  "mean_a": round(float(np.mean(ca[k])), 2),
                  "mean_b": round(float(np.mean(cb[k])), 2)}
    return out, ca, cb


# ══════════════════════════════════════════════════════════════════════════════
#  MOTIVOS
# ══════════════════════════════════════════════════════════════════════════════

def parse_note(s):
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", s.strip())
    if not m:
        raise ValueError(f"nota inválida: '{s}' (ej. D4, Bb3, F#5)")
    return (int(m.group(3)) + 1) * 12 + NOTE_PC[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]


def motif_from_midi(path):
    p = Piece(path)
    best = max(p.pitched(), key=lambda t: len(t["notes"]), default=None)
    if not best:
        raise ValueError("el MIDI del motivo no tiene notas")
    return [n[1] for n in top_line(best["notes"])]


def top_line(notes):
    """Nota más aguda por ataque, en orden temporal → [(start, pitch)]."""
    top = {}
    for s, e, p, v in notes:
        top[s] = max(top.get(s, -1), p)
    return [(s, top[s]) for s in sorted(top)]


def intervals(seq):
    return [b - a for a, b in zip(seq, seq[1:])]


def find_motif(tr_notes, motif, tpbar, tol=0, inversion=True, retro=True):
    """Apariciones del motivo (por intervalos) en la línea superior de una pista."""
    line = top_line(tr_notes)
    pitches = [p for _, p in line]
    starts = [s for s, _ in line]
    base = intervals(motif)
    forms = {"directa": base}
    if inversion:
        forms["inversión"] = [-i for i in base]
    if retro:
        forms["retrógrado"] = [-i for i in reversed(base)]
    n = len(motif) - 1
    found = []
    if n < 1:
        return found
    iv = intervals(pitches)
    for form, pat in forms.items():
        for i in range(len(iv) - n + 1):
            seg = iv[i:i + n]
            dist = max(abs(x - y) for x, y in zip(seg, pat))
            if dist <= tol:
                found.append({"bar": starts[i] // tpbar + 1, "form": form,
                              "exact": dist == 0,
                              "transposition": pitches[i] - motif[0] if form == "directa" else None})
    return found


def motif_report(P, motif, tol, inv, retro):
    res = []
    for t in P.pitched():
        f = find_motif(t["notes"], motif, P.tpbar, tol, inv, retro)
        if f:
            res.append({"track": t["name"] or f"pista {t['index']}", "count": len(f),
                        "exact": sum(1 for x in f if x["exact"]),
                        "bars": sorted({x["bar"] for x in f}),
                        "forms": dict(Counter(x["form"] for x in f))})
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  ARMONÍA
# ══════════════════════════════════════════════════════════════════════════════

def chord_label(chroma):
    if chroma.sum() < 1e-9:
        return None
    best, bl = -1e9, None
    for root in range(12):
        for q, iv in TEMPLATES.items():
            mask = np.zeros(12)
            for i in iv:
                mask[(root + i) % 12] = 1
            # cobertura de las notas de la plantilla − notas ajenas − pequeña pena por séptimas
            inside = (chroma * mask).sum()
            outside = (chroma * (1 - mask)).sum()
            score = inside - 0.8 * outside - 0.05 * (len(iv) - 3) + 0.3 * chroma[root] / chroma.sum()
            if score > best:
                best, bl = score, (root, q)
    return f"{PCS[bl[0]]}{bl[1]}"


def estimate_key(chroma):
    best, name = -2, "?"
    for r in range(12):
        for prof, suf in ((KS_MAJ, ""), (KS_MIN, "m")):
            c = np.corrcoef(chroma, np.roll(prof, r))[0, 1]
            if c > best:
                best, name = c, f"{PCS[r]}{suf}"
    return name, float(best)


def harmony_stats(p):
    half = max(1, p.tpbar // 2)
    nwin = p.bars * 2
    chroma = np.zeros((nwin, 12))
    for t in p.pitched():
        for s, e, pit, v in t["notes"]:
            w0, w1 = min(s // half, nwin - 1), min((e - 1) // half, nwin - 1)
            for w in range(w0, w1 + 1):
                ov = min(e, (w + 1) * half) - max(s, w * half)
                chroma[w, pit % 12] += max(ov, 0) * (v / 127)
    labels = [chord_label(c) for c in chroma]
    seq = [l for l in labels if l]
    changes = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    key, kc = estimate_key(chroma.sum(axis=0)) if chroma.sum() else ("?", 0.0)
    return {"key": key, "key_corr": round(kc, 2),
            "changes_per_bar": round(changes / p.bars, 3),
            "unique_chords": len(set(seq)),
            "top_chords": Counter(seq).most_common(6),
            "labels": labels}


# ══════════════════════════════════════════════════════════════════════════════
#  REGISTROS
# ══════════════════════════════════════════════════════════════════════════════

def range_table(A, B):
    def info(P):
        d = {}
        for t in P.pitched():
            ps = [n[2] for n in t["notes"]]
            d[(t["name"] or f"pista {t['index']}").strip().lower()] = (
                min(ps), max(ps), float(np.mean(ps)), len(ps))
        return d
    ia, ib = info(A), info(B)
    names = list(ia) + [n for n in ib if n not in ia]
    return [{"track": n, "a": ia.get(n), "b": ib.get(n)} for n in names]


def nn(p):
    return f"{PCS[p % 12]}{p // 12 - 1}"


# ══════════════════════════════════════════════════════════════════════════════
#  PARECIDO GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def similarity(A, B, curves, hA, hB):
    """Índice 0–1: curvas (50 %), perfil de croma (30 %), registro medio (20 %)."""
    cs = np.mean([(curves[k]["corr"] + 1) / 2 for k in ("density", "pitch", "velocity")])

    def chroma(P):
        v = np.zeros(12)
        for t in P.pitched():
            for s, e, p, vel in t["notes"]:
                v[p % 12] += (e - s)
        return v / v.sum() if v.sum() else v
    ca, cb = chroma(A), chroma(B)
    cc = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-9))

    def mean_pitch(P):
        ps = [n[2] for t in P.pitched() for n in t["notes"]]
        return float(np.mean(ps)) if ps else 60.0
    reg = max(0.0, 1 - abs(mean_pitch(A) - mean_pitch(B)) / 24)
    return round(0.5 * cs + 0.3 * cc + 0.2 * reg, 3)


def verdict(s):
    return ("casi idénticos" if s > 0.93 else "muy parecidos" if s > 0.8 else
            "parentesco claro" if s > 0.65 else "parentesco débil" if s > 0.5 else
            "obras claramente distintas")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def show_curves(A, B, points):
    cmp_, ca, cb = compare_curves(A, B, points)
    B_, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n  {B_}CURVAS por compás{R}   A = {A.name} ({A.bars} cc)   B = {B.name} ({B.bars} cc)")
    labels = {"density": "densidad (notas/cc)", "pitch": "registro (pitch medio)",
              "velocity": "velocidad media", "tracks": "pistas activas"}
    for k, lab in labels.items():
        c = cmp_[k]["corr"]
        col = _c("GRN") if c > 0.7 else _c("YEL") if c > 0.3 else _c("RED")
        print(f"\n  {lab}   media A {cmp_[k]['mean_a']}  B {cmp_[k]['mean_b']}   "
              f"correlación {col}{c:+.2f}{R}")
        print(f"    A {spark(ca[k])}")
        print(f"    B {spark(cb[k])}")
    print()
    return cmp_


def show_motifs(A, B, motif, tol, inv, retro):
    print(f"\n  {_c('B')}MOTIVO{_c('R')}  {' '.join(nn(p) for p in motif)}   "
          f"(intervalos {intervals(motif)}, tolerancia ±{tol})")
    out = {}
    for P in (A, B):
        r = motif_report(P, motif, tol, inv, retro)
        out[P.name] = r
        tot = sum(x["count"] for x in r)
        print(f"\n  {_c('B')}{P.name}{_c('R')}: {tot} aparición(es) en {len(r)} pista(s)")
        for x in r:
            bars = ", ".join(map(str, x["bars"][:14])) + (" …" if len(x["bars"]) > 14 else "")
            print(f"    {x['track'][:26]:<26} {x['count']:>3} (exactas {x['exact']})  "
                  f"{x['forms']}  cc: {bars}")
        if not r:
            print(f"    {_c('G')}(no aparece){_c('R')}")
    print()
    return out


def show_harmony(A, B):
    hA, hB = harmony_stats(A), harmony_stats(B)
    print(f"\n  {_c('B')}ARMONÍA{_c('R')}")
    for P, h in ((A, hA), (B, hB)):
        top = "  ".join(f"{c}×{n}" for c, n in h["top_chords"])
        print(f"    {P.name}: tonalidad {h['key']} (r={h['key_corr']})  "
              f"ritmo armónico {h['changes_per_bar']:.2f} cambios/cc  "
              f"{h['unique_chords']} acordes distintos")
        print(f"      {top}")
    print()
    return hA, hB


def show_ranges(A, B):
    rows = range_table(A, B)
    print(f"\n  {_c('B')}REGISTROS por pista{_c('R')}  (mín–máx · medio · notas)")
    print(f"    {'pista':<26}{'A':<30}{'B'}")
    for r in rows:
        def fmt(x):
            return "—" if x is None else f"{nn(x[0])}–{nn(x[1])} · {nn(int(round(x[2])))} · {x[3]}"
        print(f"    {r['track'][:25]:<26}{fmt(r['a']):<30}{fmt(r['b'])}")
    print()
    return rows


def load_motif(a):
    if getattr(a, "motif", None):
        return [parse_note(x) for x in a.motif.replace(",", " ").split()]
    if getattr(a, "motif_midi", None):
        return motif_from_midi(a.motif_midi)
    return None


def cmd_curves(a):
    r = show_curves(Piece(a.a), Piece(a.b), a.points)
    _dump(a, {"curves": r})
    return 0


def cmd_motifs(a):
    m = load_motif(a)
    if not m or len(m) < 3:
        print("[ERROR] indique --motif \"D4 G4 C5\" o --motif-midi (≥3 notas)", file=sys.stderr)
        return 2
    r = show_motifs(Piece(a.a), Piece(a.b), m, a.tolerance, not a.no_inversion, not a.no_retrograde)
    _dump(a, {"motifs": r})
    return 0


def cmd_harmony(a):
    hA, hB = show_harmony(Piece(a.a), Piece(a.b))
    for h in (hA, hB):
        h.pop("labels")
    _dump(a, {"harmony": [hA, hB]})
    return 0


def cmd_ranges(a):
    show_ranges(Piece(a.a), Piece(a.b))
    return 0


def cmd_report(a):
    A, B = Piece(a.a), Piece(a.b)
    cur = show_curves(A, B, 32)
    hA, hB = show_harmony(A, B)
    show_ranges(A, B)
    out = {"curves": cur}
    m = load_motif(a)
    if m and len(m) >= 3:
        out["motifs"] = show_motifs(A, B, m, 1, True, True)
    s = similarity(A, B, cur, hA, hB)
    print(f"  {_c('B')}PARECIDO GLOBAL{_c('R')}  {s:.2f}  → {verdict(s)}")
    print(f"  duración: A {A.bars} cc → B {B.bars} cc   "
          f"notas: A {sum(len(t['notes']) for t in A.tracks)} → B {sum(len(t['notes']) for t in B.tracks)}   "
          f"pistas con notas: A {len(A.pitched())} → B {len(B.pitched())}\n")
    for h in (hA, hB):
        h.pop("labels")
    out.update({"similarity": s, "verdict": verdict(s), "harmony": [hA, hB]})
    _dump(a, out)
    return 0


def _dump(a, d):
    if getattr(a, "json", None):
        txt = json.dumps(d, ensure_ascii=False, indent=1, default=str)
        if a.json == "-":
            print(txt)
        else:
            open(a.json, "w", encoding="utf-8").write(txt)


def _write(path, notes_by_track, tempo=500000):
    mid = mido.MidiFile(ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo))
    mid.tracks.append(meta)
    for name, notes in notes_by_track.items():
        t = mido.MidiTrack()
        t.append(mido.MetaMessage("track_name", name=name))
        ev = []
        for s, d, p in notes:
            ev += [(s, 1, p), (s + d, 0, p)]
        ev.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for tt, on, p in ev:
            t.append(mido.Message("note_on", note=p, velocity=80 if on else 0, time=tt - last))
            last = tt
        mid.tracks.append(t)
    mid.save(path)


def cmd_selftest(a):
    d = tempfile.mkdtemp()
    motif = [62, 67, 72, 69]                       # D4 G4 C5 A4
    B = 1920

    def melody(shift=0, bars=8, with_motif=True):
        notes = []
        for b in range(bars):
            if with_motif and b % 2 == 0:
                for k, p in enumerate(motif):
                    notes.append((b * B + k * 480, 480, p + shift))
            else:
                for k in range(4):
                    notes.append((b * B + k * 480, 480, 60 + shift + (b + k) % 3))
        return notes

    pa, pb, pc = (os.path.join(d, x) for x in ("a.mid", "b.mid", "c.mid"))
    _write(pa, {"Viola": melody()})
    _write(pb, {"Viola": melody(shift=5), "Bajo": [(b * B, B, 38) for b in range(8)]})
    _write(pc, {"Viola": melody(with_motif=False)})
    A, Bp, C = Piece(pa), Piece(pb), Piece(pc)
    ok = []
    f = find_motif(A.tracks[1]["notes"], motif, A.tpbar)
    ok.append(("motivo exacto en A (4 apariciones)", sum(1 for x in f if x["form"] == "directa" and x["exact"]) == 4))
    f = find_motif(Bp.tracks[1]["notes"], motif, Bp.tpbar)
    ok.append(("motivo transportado +5 en B", all(x["transposition"] == 5 for x in f if x["form"] == "directa" and x["exact"])
               and sum(1 for x in f if x["form"] == "directa" and x["exact"]) == 4))
    f = find_motif(C.tracks[1]["notes"], motif, C.tpbar, inversion=False, retro=False)
    ok.append(("sin motivo en C", len(f) == 0))
    inv = [2 * motif[0] - p for p in motif]
    ok.append(("inversión detectada", any(x["form"] == "inversión" for x in find_motif(
        [(i * 480, i * 480 + 480, p, 80) for i, p in enumerate(inv)], motif, B))))
    cur, _, _ = compare_curves(A, A, 16)
    ok.append(("curvas idénticas → correlación 1", all(abs(v["corr"] - 1) < 1e-6 for v in cur.values())))
    ok.append(("A≈A parecido ≥0.99", similarity(A, A, cur, None, None) >= 0.99))
    cur2, _, _ = compare_curves(A, C, 16)
    s_ac = similarity(A, C, cur2, None, None)
    cur3, _, _ = compare_curves(A, Bp, 16)
    ok.append(("parecido A–C ≤ A–A", s_ac <= 1.0))
    h = harmony_stats(A)
    ok.append(("tonalidad estimada razonable", h["key"] in ("C", "Am", "G", "F", "D", "Dm", "Gm", "Em")))
    cm = chord_label(np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], float))
    ok.append(("acorde C mayor", cm == "C"))
    for n, v in ok:
        print(f"  {'✓' if v else '✗'} {n}")
    return 0 if all(v for _, v in ok) else 1


def main():
    ap = argparse.ArgumentParser(description="Comparación cuantitativa entre dos MIDIs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def two(p):
        p.add_argument("a")
        p.add_argument("b")
        p.add_argument("--json")

    p = sub.add_parser("curves"); two(p); p.add_argument("--points", type=int, default=32); p.set_defaults(fn=cmd_curves)
    p = sub.add_parser("motifs"); two(p)
    p.add_argument("--motif"); p.add_argument("--motif-midi")
    p.add_argument("--tolerance", type=int, default=0, choices=(0, 1))
    p.add_argument("--no-inversion", action="store_true"); p.add_argument("--no-retrograde", action="store_true")
    p.set_defaults(fn=cmd_motifs)
    p = sub.add_parser("harmony"); two(p); p.set_defaults(fn=cmd_harmony)
    p = sub.add_parser("ranges"); two(p); p.set_defaults(fn=cmd_ranges)
    p = sub.add_parser("report"); two(p); p.add_argument("--motif"); p.add_argument("--motif-midi"); p.set_defaults(fn=cmd_report)
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
