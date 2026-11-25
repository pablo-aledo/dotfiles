#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POEMA SINFÓNICO CON LEITMOTIFS VIVOS
====================================

Entrada : un guion (story.json) con personajes, escenas y estados emocionales.
Salida  : MIDI orquestal + audio + línea temporal + informe del rastreador.

Cadena de módulos (equivalentes a los programas de la taxonomía):
  narrator            -> carga del guion y plan de escenas
  motif bank          -> un leitmotif por personaje (en GRADOS de escala)
  theme_transformer   -> estado emocional  => recetas de transformación
  leitmotif_tracker   -> mide si cada variante sigue siendo reconocible
                         y la repara si no; verifica leyendo el MIDI final
  melody_harmonizer   -> armonía "melodía primero" (Viterbi por compases)
  orchestral_colorist -> timbre fijo por personaje + acompañamiento
  silence_designer    -> silencios en los giros dramáticos + ritardandi
"""
import argparse, json, math, os, random, subprocess, sys, shutil
from dataclasses import dataclass, replace
from typing import List, Optional

import mido

PPQ = 480

# ──────────────────────────────────────────────────────────────────────────
# 1. TEORÍA BÁSICA
# ──────────────────────────────────────────────────────────────────────────
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MODES = {
    "major": MAJOR,
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
}
PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
      "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

# programa GM -> (nota mínima, máxima, centro cómodo)
INSTR = {
    60: (41, 77, 60), 56: (55, 82, 72), 57: (40, 72, 52), 58: (28, 58, 40),
    68: (58, 91, 74), 69: (52, 81, 66), 73: (60, 96, 80), 40: (55, 100, 78),
    42: (36, 76, 52), 71: (50, 94, 68), 70: (34, 75, 52), 48: (40, 96, 66),
    43: (28, 55, 40), 46: (36, 96, 66), 47: (36, 55, 45), 52: (48, 84, 64),
}


def deg_semis(scale, deg, alt):
    """Semitonos de un grado (puede ser negativo). `alt` es relativo al modo
    MAYOR y es idempotente: alt=-1 baja el grado solo si el modo no lo tiene
    ya bajado; alt=+1 lo sube solo si el modo no lo tiene ya subido."""
    base, m = scale[deg % 7], MAJOR[deg % 7]
    if alt < 0:
        s = min(base, m + alt)
    elif alt > 0:
        s = max(base, m + alt)
    else:
        s = base
    return 12 * (deg // 7) + s


# ──────────────────────────────────────────────────────────────────────────
# 2. MOTIVOS Y TRANSFORMACIONES
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class N:
    deg: Optional[int]      # grado de escala (0 = tónica); None = silencio
    dur: float              # en negras
    alt: int = 0            # inflexión cromática relativa al modo mayor
    acc: float = 0.0        # acento 0..1


# plantillas por perfil: lista de (grado, duración, alteración)
PROFILES = {
    "heroico": [
        [(0, 1.5, 0), (4, .5, 0), (3, 1, 0), (2, 1, 0), (4, 2, 0)],
        [(0, 1, 0), (0, .5, 0), (2, .5, 0), (4, 2, 0), (3, 1, 0), (2, 1, 0)],
    ],
    "lirico": [
        [(4, 2, 0), (5, 1, 0), (4, 1, 0), (2, 1.5, 0), (3, .5, 0), (1, 2, 0)],
        [(2, 2, 0), (4, 1, 0), (3, 1, 0), (5, 2, 0), (4, 2, 0)],
    ],
    "sombrio": [
        [(0, .5, 0), (1, .5, -1), (0, .5, 0), (3, .5, 1), (2, 1, 0), (0, 2, 0)],
        [(4, .5, 0), (3, .5, 1), (4, .5, 0), (1, .5, -1), (0, 1, 0), (0, 2, 0)],
    ],
}


def make_motif(profile, rng, explicit=None):
    tpl = explicit if explicit else rng.choice(PROFILES[profile])
    m = [N(d, du, al) for d, du, al in tpl]
    longest = max(range(len(m)), key=lambda i: m[i].dur)
    m[0] = replace(m[0], acc=1.0)
    if longest != 0:
        m[longest] = replace(m[longest], acc=0.6)
    return m


def op_augment(m, k=2.0): return [replace(n, dur=n.dur * k) for n in m]
def op_diminish(m, k=2.0): return [replace(n, dur=n.dur / k) for n in m]
def op_retro(m): return list(reversed(m))
def op_transpose(m, steps=0): return [replace(n, deg=None if n.deg is None else n.deg + steps) for n in m]
def op_purify(m): return [replace(n, alt=0) for n in m]


def op_fragment(m, n=3, part="head"):
    p = [x for x in m if x.deg is not None]
    return p[:n] if part == "head" else p[-n:]


def op_invert(m):
    piv = next(x.deg for x in m if x.deg is not None)
    return [n if n.deg is None else replace(n, deg=2 * piv - n.deg, alt=-n.alt) for n in m]


def op_flat(m, degs=(2, 5, 6)):
    return [replace(n, alt=-1) if (n.deg is not None and n.alt == 0 and n.deg % 7 in degs) else n for n in m]


def op_syncopate(m):
    last = m[-1]
    return [N(None, .5)] + list(m[:-1]) + [replace(last, dur=max(.25, last.dur - .5))]


def op_ornament(m):
    out, done = [], False
    for n in m:
        if not done and n.deg is not None and n.dur >= 1:
            out += [replace(n, dur=.25), replace(n, deg=n.deg + 1, alt=0, dur=.25, acc=0),
                    replace(n, dur=n.dur - .5, acc=0)]
            done = True
        else:
            out.append(n)
    return out


STMT_OPS = {"augment": op_augment, "diminish": op_diminish, "retro": op_retro,
            "transpose": op_transpose, "purify": op_purify, "fragment": op_fragment,
            "invert": op_invert, "flat": op_flat, "syncopate": op_syncopate,
            "ornament": op_ornament}
# "octave" y "sequence" se tratan aparte (registro y nivel de frase)


@dataclass
class Variation:
    statements: list
    octave: int = 0


def apply_ops(motif, ops):
    cur, octave, seq = list(motif), 0, None
    for name, kw in ops:
        if name == "octave":
            octave += kw.get("k", 0)
        elif name == "sequence":
            seq = kw
        else:
            cur = STMT_OPS[name](cur, **kw)
    stmts = [cur]
    if seq:
        for r in range(1, seq.get("reps", 2)):
            stmts.append(op_transpose(cur, steps=seq.get("step", 1) * r))
    return Variation(stmts, octave)


# estado emocional -> receta (ordenada de más a menos esencial)
STATES = {
    "sereno":      dict(ops=[], vel=64, art=.95, sustain=True),
    "inquieto":    dict(ops=[("syncopate", {}), ("flat", {})], vel=70, art=.9),
    "amenazante":  dict(ops=[("octave", {"k": -1}), ("diminish", {"k": 1.5}),
                             ("sequence", {"reps": 3, "step": 0})], vel=88, art=.7),
    "herido":      dict(ops=[("fragment", {"n": 3}), ("augment", {"k": 1.5}),
                             ("transpose", {"steps": -2})], vel=54, art=1.0, sustain=True),
    "traicionado": dict(ops=[("invert", {}), ("flat", {}), ("syncopate", {})], vel=84, art=.85),
    "desafiante":  dict(ops=[("diminish", {"k": 2}), ("sequence", {"reps": 3, "step": 1})], vel=100, art=.8),
    "triunfal":    dict(ops=[("ornament", {}), ("transpose", {"steps": 2}), ("octave", {"k": 1}),
                             ("sequence", {"reps": 2, "step": 2})], vel=104, art=.9),
    "desolado":    dict(ops=[("fragment", {"n": 2}), ("augment", {"k": 2.5}),
                             ("transpose", {"steps": -3}), ("octave", {"k": -1})], vel=42, art=1.0, sustain=True),
    "esperanza":   dict(ops=[("transpose", {"steps": 2}), ("ornament", {})], vel=66, art=.95, sustain=True),
    "redimido":    dict(ops=[("purify", {}), ("augment", {"k": 1.5}), ("ornament", {})],
                        vel=78, art=1.0, sustain=True),
}
STATE_COLORS = {
    "sereno": "#8fbf8f", "inquieto": "#d6c35a", "amenazante": "#7a3b8f", "herido": "#c97b63",
    "traicionado": "#b3262b", "desafiante": "#e07a1f", "triunfal": "#e5b800",
    "desolado": "#5b7aa6", "esperanza": "#6fc3d6", "redimido": "#f2e6c9",
}


# ──────────────────────────────────────────────────────────────────────────
# 3. LEITMOTIF TRACKER: reconocibilidad de una variante
# ──────────────────────────────────────────────────────────────────────────
def events_of(m):
    t, ev = 0.0, []
    for n in m:
        if n.deg is not None:
            ev.append((t, n.dur, n.deg))
        t += n.dur
    return ev


def _intervals(ev): return [b[2] - a[2] for a, b in zip(ev, ev[1:])]
def _parsons(iv): return [1 if x > 0 else -1 if x < 0 else 0 for x in iv]


def _lev_sim(a, b):
    if not a and not b:
        return 1.0
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (x != y))
    return 1 - dp[-1] / max(len(a), len(b))


def _onset_sim(a, b, cells=32):
    def grid(ev):
        t0 = ev[0][0]
        span = max(e[0] + e[1] for e in ev) - t0
        return sorted(round((e[0] - t0) / span * cells) for e in ev)
    ga, gb = grid(a), grid(b)
    used, hit = set(), 0
    for x in ga:
        for j, y in enumerate(gb):
            if j not in used and abs(x - y) <= 1:
                used.add(j); hit += 1
                break
    return 2 * hit / (len(ga) + len(gb))


def _pair_score(ea, eb):
    ia, ib = _intervals(ea), _intervals(eb)
    if ia and ib:
        c = _lev_sim(_parsons(ia), _parsons(ib))
        i = (1 - sum(min(abs(x - y), 4) / 4 for x, y in zip(ia, ib)) / len(ia)) if len(ia) == len(ib) else c * .8
    else:
        c = i = 0.0
    r = _onset_sim(ea, eb)
    return .4 * max(c, r) + .2 * c + .2 * i + .2 * r


def recognizability(orig, var):
    """0..1. Compara en espacio de grados (independiente de tonalidad, modo y
    octava). Acepta la variante tal cual, invertida o en retrógrado (peso .75)
    y penaliza los fragmentos cortos."""
    A, B = events_of(orig), events_of(var)
    if not B:
        return 0.0
    tot = max(e[0] + e[1] for e in B)
    inv = [(t, d, -g) for t, d, g in B]
    retro = sorted([(tot - (t + d), d, g) for t, d, g in B])
    views = [(1.0, B), (.75, inv), (.75, retro)]
    p, q = len(A), len(B)
    cover = min(1.0, (q / p) ** .6)
    best = 0.0
    for w, vb in views:
        wins = [A[k:k + q] for k in range(p - q + 1)] if q <= p else [A]
        for wa in wins:
            best = max(best, w * _pair_score(wa, vb))
    return round(best * cover, 3)


def build_variation(motif, ops, thr):
    """Aplica la receta; si la variante cae bajo el umbral, retira la
    transformación cuya eliminación más mejora la reconocibilidad."""
    ops, dropped = list(ops), []
    while True:
        var = apply_ops(motif, ops)
        score = recognizability(motif, var.statements[0])
        cand = [i for i, (nm, _) in enumerate(ops) if nm not in ("octave", "sequence")]
        if score >= thr or not cand:
            return var, score, ops, dropped
        scored = [(recognizability(motif, apply_ops(motif, ops[:i] + ops[i + 1:]).statements[0]), i) for i in cand]
        _, k = max(scored)
        dropped.append(ops[k][0])
        ops.pop(k)


# ──────────────────────────────────────────────────────────────────────────
# 4. ARMONÍA "MELODÍA PRIMERO"
# ──────────────────────────────────────────────────────────────────────────
FUNC = {0: "T", 1: "S", 2: "T", 3: "S", 4: "D", 5: "T", 6: "D"}
TRANS = {("T", "T"): .3, ("T", "S"): .6, ("T", "D"): .45, ("S", "D"): .85, ("S", "T"): .35,
         ("S", "S"): .25, ("D", "T"): .95, ("D", "S"): .1, ("D", "D"): .2}


def build_chords(tonic, scale, tension):
    out = []

    def add(name, root, pcs, func, deg=-1):
        key = (root, tuple(sorted(pcs)))
        if all((c["root"], tuple(sorted(c["pcs"]))) != key for c in out):
            out.append(dict(name=name, root=root, pcs=list(pcs), func=func, deg=deg))

    for d in range(7):
        r = (tonic + scale[d]) % 12
        t = (tonic + scale[(d + 2) % 7]) % 12
        f = (tonic + scale[(d + 4) % 7]) % 12
        third, fifth = (t - r) % 12, (f - r) % 12
        nm = ROMAN[d] if third == 4 else ROMAN[d].lower()
        if fifth == 6:
            nm = ROMAN[d].lower() + "°"
        pcs = [r, t, f]
        if FUNC[d] == "D" and tension >= .5:
            pcs.append((r + (10 if third == 4 else 9 if tension >= .8 else 10)) % 12)
        add(nm, r, pcs, FUNC[d], d)
    minorish = scale[2] == 3
    if minorish:
        r = (tonic + 7) % 12
        add("V", r, [r, (r + 4) % 12, (r + 7) % 12] + ([(r + 10) % 12] if tension >= .5 else []), "D", 4)
        r = (tonic + 11) % 12
        add("vii°", r, [r, (r + 3) % 12, (r + 6) % 12] + ([(r + 9) % 12] if tension >= .5 else []), "D", 6)
    if tension >= .55:
        r = (tonic + 1) % 12
        add("bII", r, [r, (r + 4) % 12, (r + 7) % 12], "S")
    if not minorish and tension >= .6:
        r = (tonic + 5) % 12
        add("iv", r, [r, (r + 3) % 12, (r + 7) % 12], "S")
        r = (tonic + 8) % 12
        add("bVI", r, [r, (r + 4) % 12, (r + 7) % 12], "S")
    return out


def harmonize(bar_notes, bars, tonic, scale, tension, ending, bar_beats):
    cands = build_chords(tonic, scale, tension)
    spc = {(tonic + s) % 12 for s in scale}

    def allowed(b):
        c = cands
        if bars < 2:
            return c
        if ending == "authentic":
            if b == bars - 1: c = [x for x in cands if x["deg"] == 0 and x["root"] == tonic]
            if b == bars - 2: c = [x for x in cands if x["func"] == "D"]
        elif ending == "plagal":
            if b == bars - 1: c = [x for x in cands if x["deg"] == 0 and x["root"] == tonic]
            if b == bars - 2: c = [x for x in cands if x["root"] == (tonic + 5) % 12 and x["func"] == "S"]
        elif ending == "half":
            if b == bars - 1: c = [x for x in cands if x["root"] == (tonic + 7) % 12 and x["func"] == "D"]
        elif ending == "deceptive":
            if b == bars - 2: c = [x for x in cands if x["root"] == (tonic + 7) % 12 and x["func"] == "D"]
            if b == bars - 1: c = [x for x in cands if x["root"] == (tonic + scale[5]) % 12 and x["func"] == "T"]
        c = c or cands
        if any(x["name"] == "V" for x in c):   # en menor, las cadencias usan V mayor
            c = [x for x in c if x["name"] != "v"]
        return c

    def cover(b, c):
        notes = bar_notes[b]
        if not notes:
            return 0.0
        num = den = 0.0
        for rel, dur, pc in notes:
            strong = rel < 1e-6 or abs(rel - bar_beats / 2) < 1e-6
            w = dur * (2 if strong else 1)
            v = 1.0 if pc in c["pcs"] else (-.25 if pc in spc else -.6)
            num += w * v; den += w
        return num / den

    score = [dict() for _ in range(bars)]
    back = [dict() for _ in range(bars)]
    for b in range(bars):
        for ci, c in enumerate(allowed(b)):
            base = 1.6 * cover(b, c)
            if b == 0:
                score[b][c["name"] + str(c["root"])] = (base + (1.0 if c["deg"] == 0 and c["root"] == tonic else 0), c, None)
                continue
            best = None
            for k, (sp, pc_, _) in score[b - 1].items():
                tr = TRANS[(pc_["func"], c["func"])] * .8 - (.15 if pc_["root"] == c["root"] and pc_["name"] == c["name"] else 0)
                v = sp + base + tr
                if best is None or v > best[0]:
                    best = (v, k)
            score[b][c["name"] + str(c["root"])] = (best[0], c, best[1])
    k = max(score[-1], key=lambda x: score[-1][x][0])
    seq = []
    for b in range(bars - 1, -1, -1):
        _, c, prev = score[b][k]
        seq.append(c)
        k = prev
    return list(reversed(seq))


def voice_pad(pcs, prev, lo=52, hi=76):
    import itertools
    opts = [[p for p in range(lo, hi + 1) if p % 12 == pc] for pc in pcs]
    best, bv = None, None
    for combo in itertools.product(*opts):
        s = sorted(combo)
        if s[-1] - s[0] > 19 and len(pcs) > 3:
            continue
        cost = sum(abs(a - b) for a, b in zip(s, sorted(prev)[:len(s)] + [sum(prev) // len(prev)] * 4))
        if len(set(s)) == len(s) and (best is None or cost < best):
            best, bv = cost, s
    return bv or sorted(prev)


# ──────────────────────────────────────────────────────────────────────────
# 5. COMPOSICIÓN DE UNA ESCENA
# ──────────────────────────────────────────────────────────────────────────
def place(pitches, lo, hi, center, octave=0):
    med = sorted(pitches)[len(pitches) // 2]
    K = round((center - med) / 12)
    out = [p + 12 * (K + octave) for p in pitches]
    while max(out) > hi and min(out) - 12 >= lo - 6:
        out = [p - 12 for p in out]
    while min(out) < lo and max(out) + 12 <= hi + 6:
        out = [p + 12 for p in out]
    return out


def statement_len(stmt): return sum(n.dur for n in stmt)


def schedule(scene, cast, bar_beats, bars):
    total, tension = bars * bar_beats, scene["tension"]
    tex = scene.get("texture", "alternar")
    spans = [sum(statement_len(s) for s in c["var"].statements) for c in cast]
    pl = []
    if tex == "canon":
        delay = scene.get("canon_delay_bars", 2)
        for i, c in enumerate(cast):
            period = math.ceil(spans[i] / bar_beats) * bar_beats
            t = i * delay * bar_beats
            while t + spans[i] <= total:
                pl.append((i, t)); t += period
    elif tex == "stretto":
        off = [i * 1.5 for i in range(len(cast))]
        slot = max(math.ceil((spans[i] + off[i]) / bar_beats) for i in range(len(cast)))
        t = 0.0
        while t + max(spans[i] + off[i] for i in range(len(cast))) <= total:
            for i in range(len(cast)):
                pl.append((i, t + off[i]))
            t += slot * bar_beats
    else:
        gap = 1 if tension < .6 else 0
        others = list(range(1, len(cast)))
        t, k = scene.get("entry_delay_bars", 0) * bar_beats, 0
        while True:
            i = 0 if (k % 2 == 0 or not others) else others[(k // 2) % len(others)]
            if t + spans[i] > total:
                break
            pl.append((i, t))
            t += (math.ceil(spans[i] / bar_beats) + gap) * bar_beats
            k += 1
    return sorted(pl, key=lambda x: x[1]), spans


def compose_scene(si, scene, story, chars, voices, g0, rng, thr, log):
    tonic = PC[scene["key"]]
    scale = MODES[scene["mode"]]
    num, den = scene.get("meter", [4, 4])
    bar_beats = num * 4 / den
    bars, tension = scene["bars"], scene["tension"]
    cast = []
    for who, state in scene["cast"]:
        st = STATES[state]
        var, sc, ops_used, dropped = build_variation(chars[who]["motif"], st["ops"], thr)
        cast.append(dict(who=who, state=state, var=var, score=sc, ops=[o[0] for o in ops_used],
                         dropped=dropped, st=st))
    placements, spans = schedule(scene, cast, bar_beats, bars)

    # --- realizar melodías ------------------------------------------------
    plc = []
    for ci, t0 in placements:
        c = cast[ci]; ch = chars[c["who"]]; v = ch["voice"]
        raw, tt, groups = [], t0, []
        for stmt in c["var"].statements:
            ev, t = [], tt
            for n in stmt:
                if n.deg is not None:
                    ev.append([t, n.dur, tonic + deg_semis(scale, n.deg, n.alt), n.acc])
                t += n.dur
            groups.append(ev); tt = t
        pitches = place([e[2] for ev in groups for e in ev], v["lo"], v["hi"], v["center"], c["var"].octave)
        k = 0
        for ev in groups:
            for e in ev:
                e[2] = pitches[k]; k += 1
        flat = [e for ev in groups for e in ev]
        if c["st"].get("sustain") and scene.get("texture", "alternar") == "alternar":
            end_bar = math.ceil((flat[-1][0] + flat[-1][1]) / bar_beats) * bar_beats
            hold = min(end_bar, bars * bar_beats) - .15 - flat[-1][0]
            if hold > flat[-1][1]:
                flat[-1][1] = hold
        base = c["st"]["vel"] + 22 * (tension - .5)
        out = []
        for e in flat:
            vel = int(max(25, min(120, base + 8 * e[3] + (4 if e[1] >= 2 else 0))))
            out.append([e[0], e[1] * (1.0 if e[1] > 2.5 else c["st"]["art"]), e[2], vel])
        plc.append(dict(ci=ci, start=t0, events=out, n_first=len(groups[0]),
                        end=max(e[0] + e[1] for e in out)))

    # --- armonía melodía-primero -----------------------------------------
    bar_notes = [[] for _ in range(bars)]
    for p in plc:
        for st_, du, pi, _ in p["events"]:
            b = int(st_ // bar_beats)
            if b >= bars: continue
            bar_notes[b].append((st_ - b * bar_beats, min(du, bar_beats), pi % 12))
            rest, bb = st_ + du - (b + 1) * bar_beats, b + 1
            while rest > 1e-6 and bb < bars:
                bar_notes[bb].append((0.0 + 0.01, min(rest, bar_beats), pi % 12)); rest -= bar_beats; bb += 1
    chords = harmonize(bar_notes, bars, tonic, scale, tension, scene.get("ending", "authentic"), bar_beats)

    # --- ajuste a acordes en pulsos fuertes (solo escenas tutti) ----------
    if scene.get("snap"):
        for p in plc:
            for e in p["events"]:
                b = min(int(e[0] // bar_beats), bars - 1)
                rel = e[0] - b * bar_beats
                if (rel < 1e-6 or abs(rel - bar_beats / 2) < 1e-6) and e[1] >= 1 and e[2] % 12 not in chords[b]["pcs"]:
                    for sh in (-1, 1, -2, 2):
                        if (e[2] + sh) % 12 in chords[b]["pcs"]:
                            e[2] += sh; break

    # --- volcar melodías a las voces + dobles ------------------------------
    apps = []
    for p in plc:
        c = cast[p["ci"]]; ch = chars[c["who"]]; v = ch["voice"]
        for e in p["events"]:
            v["events"].append([g0 + e[0], e[1], e[2], e[3]])
        want_double = ch.get("double") and (tension >= .75 or c["state"] in ("redimido", "triunfal") or scene.get("snap"))
        if want_double:
            dv = ch["dvoice"]
            off = 12 * round((dv["center"] - v["center"]) / 12)
            for e in p["events"]:
                pi = e[2] + off
                while pi > dv["hi"]: pi -= 12
                while pi < dv["lo"]: pi += 12
                dv["events"].append([g0 + e[0], e[1], pi, int(e[3] * .8)])
        first = p["events"][:p["n_first"]]
        apps.append(dict(character=c["who"], scene=si, scene_title=scene["title"], state=c["state"],
                         start_beat=g0 + p["start"], bar_in_scene=int(p["start"] // bar_beats) + 1,
                         transformations=c["ops"], dropped=c["dropped"], recognizability=c["score"],
                         degrees=[n.deg for n in c["var"].statements[0] if n.deg is not None],
                         n_statements=len(c["var"].statements),
                         expected_pitches=[e[2] for e in first],
                         first_onsets=[g0 + e[0] for e in first], span=p["end"] - p["start"]))

    # --- acompañamiento ----------------------------------------------------
    pad, bass, harp, timp, choir, drums = (voices[k] for k in ("pad", "bass", "harp", "timp", "choir", "drums"))
    pad_level = scene.get("pad_level", 1.0)
    prev = [57, 62, 66]
    for b, ch_ in enumerate(chords):
        t = g0 + b * bar_beats
        v = voice_pad(ch_["pcs"], prev); prev = v
        if pad_level > 0:
            for pi in v:
                pad["events"].append([t, bar_beats * .98, pi, int((28 + 38 * tension) * pad_level)])
            if scene.get("choir"):
                for pi in v:
                    choir["events"].append([t, bar_beats * .98, pi, int(34 + 24 * tension)])
        r = ch_["root"]
        rb = next(p_ for p_ in range(36, 48) if p_ % 12 == r)
        bass["events"].append([t, bar_beats * (.5 if tension > .5 and bar_beats >= 4 else .98), rb, int(52 + 30 * tension)])
        if tension > .5 and bar_beats >= 4:
            fifth = rb + 7 if rb + 7 < 52 else rb - 5
            bass["events"].append([t + bar_beats / 2, bar_beats * .48, fifth, int(48 + 30 * tension)])
        if tension < .45 and pad_level > 0:
            arp = sorted(set(x for x in range(48, 85) if x % 12 in ch_["pcs"][:3]))
            idx = [0, 1, 2, 3, 4, 3, 2, 1]
            steps = int(bar_beats / .5)
            for s in range(steps):
                pi = arp[min(idx[s % 8] + 2, len(arp) - 1)]
                harp["events"].append([t + s * .5, .5, pi, int(38 + 22 * tension)])
        if tension >= .6:
            tp = next(p_ for p_ in range(38, 51) if p_ % 12 == r)
            if tension >= .85:
                for s in range(int(bar_beats * 2)):
                    timp["events"].append([t + s * .5, .45, tp, int(60 + 40 * tension + (10 if s == 0 else 0))])
            else:
                timp["events"].append([t, 1.0, tp, int(60 + 40 * tension)])
    crash = list(scene.get("crash", [0] if tension >= .9 else []))
    if scene.get("final") and "last" not in crash:
        crash.append(bars - 1)
    for b in crash:
        drums["events"].append([g0 + b * bar_beats, 2.0, 49, 100])
    return dict(bars=bars, bar_beats=bar_beats, chords=[c["name"] for c in chords], apps=apps,
                cast=[(c["who"], c["state"]) for c in cast], tonic=tonic)


# ──────────────────────────────────────────────────────────────────────────
# 6. MIDI, VERIFICACIÓN, AUDIO, GRÁFICO, INFORME
# ──────────────────────────────────────────────────────────────────────────
def write_midi(path, voices, tempo_events, ts_events, markers, rng):
    mid = mido.MidiFile(ticks_per_beat=PPQ, type=1)
    cond = mido.MidiTrack(); mid.tracks.append(cond)
    ev = [(0, 0, mido.MetaMessage("track_name", name="Conductor"))]
    for b, bpm in tempo_events:
        ev.append((round(b * PPQ), 1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm))))
    for b, n, d in ts_events:
        ev.append((round(b * PPQ), 1, mido.MetaMessage("time_signature", numerator=n, denominator=d)))
    for b, txt in markers:
        ev.append((round(b * PPQ), 2, mido.MetaMessage("marker", text=txt)))
    ev.sort(key=lambda x: (x[0], x[1]))
    last = 0
    for tk, _, m in ev:
        m.time = tk - last; last = tk; cond.append(m)
    for name, v in voices.items():
        if not v["events"]:
            continue
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        if v["kind"] != "drums":
            tr.append(mido.Message("program_change", channel=v["channel"], program=v["program"], time=0))
        msgs = []
        for st_, du, pi, vel in v["events"]:
            on = round(st_ * PPQ); off = round((st_ + du) * PPQ)
            if v["kind"] in ("lead", "double"):
                on = max(0, on + rng.randint(-6, 6)); vel = max(1, min(127, int(vel) + rng.randint(-4, 4)))
            off = max(off, on + 20)
            msgs.append((on, 1, mido.Message("note_on", channel=v["channel"], note=int(pi), velocity=int(vel))))
            msgs.append((off, 0, mido.Message("note_off", channel=v["channel"], note=int(pi), velocity=0)))
        msgs.sort(key=lambda x: (x[0], x[1]))
        last = 0
        for tk, _, m in msgs:
            m.time = tk - last; last = tk; tr.append(m)
    mid.save(path)


def verify_midi(path, apps, voices_by_char):
    """Releer el MIDI final y comprobar que cada aparición suena como se registró."""
    mid = mido.MidiFile(path)
    onsets = {}
    for tr in mid.tracks:
        tk, name = 0, None
        for m in tr:
            tk += m.time
            if m.type == "track_name": name = m.name
            if m.type == "note_on" and m.velocity > 0:
                onsets.setdefault(name, []).append((tk, m.note))
    ok = 0
    for a in apps:
        name = voices_by_char[a["character"]]
        t0 = round(a["first_onsets"][0] * PPQ) - 30
        t1 = round(a["first_onsets"][-1] * PPQ) + 30
        got = [p for tk, p in sorted(onsets.get(name, [])) if t0 <= tk <= t1]
        a["verified"] = (got == a["expected_pitches"])
        ok += a["verified"]
    return ok, len(apps)


def render_audio(mid_path, out_dir, base):
    sfs = [p for p in ("/usr/share/sounds/sf2/default-GM.sf2", "/usr/share/sounds/sf2/TimGM6mb.sf2") if os.path.exists(p)]
    if not sfs or not shutil.which("fluidsynth"):
        return None
    wav = os.path.join(out_dir, base + ".wav")
    subprocess.run(["fluidsynth", "-ni", "-g", "0.9", "-r", "44100", "-F", wav, sfs[0], mid_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    if not os.path.exists(wav):
        return None
    mp3 = os.path.join(out_dir, base + ".mp3")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-af", "loudnorm=I=-18:TP=-1.5:LRA=14", "-codec:a", "libmp3lame", "-q:a", "3", mp3])
    if r.returncode == 0 and os.path.exists(mp3):
        os.remove(wav)
        return mp3
    return wav


def plot_timeline(path, story, scenes_info, chars, apps, scene_starts, scene_ends, tension_pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(3, 1, figsize=(17, 10), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 3, 1.7]))
    for a_ in ax:
        for s, e in zip(scene_starts, scene_ends):
            a_.axvline(s, color="#999", lw=.6, ls=":")
    # 1) tensión
    xs, ys = zip(*tension_pts)
    ax[0].fill_between(xs, ys, color="#b3262b", alpha=.25); ax[0].plot(xs, ys, color="#b3262b", lw=2)
    ax[0].set_ylim(0, 1.3); ax[0].set_ylabel("tensión")
    for i, (s, e) in enumerate(zip(scene_starts, scene_ends)):
        ax[0].text((s + e) / 2, 1.27, f"{i + 1}. {story['scenes'][i]['title']}", ha="center", va="top", fontsize=8, wrap=True)
    ax[0].set_title(story["title"], fontsize=13, weight="bold")
    # 2) piano roll
    for name, ch in chars.items():
        v = ch["voice"]
        for st_, du, pi, _ in v["events"]:
            ax[1].broken_barh([(st_, du)], (pi - .4, .8), color=ch["color"], lw=0)
    ax[1].set_ylabel("altura MIDI (voces de los motivos)")
    ax[1].legend(handles=[Patch(color=c["color"], label=f"{n} ({c['profile']})") for n, c in chars.items()],
                 loc="upper right", fontsize=8, ncol=3)
    # 3) apariciones por estado
    names = list(chars)
    for a in apps:
        y = names.index(a["character"])
        ax[2].barh(y, a["span"], left=a["start_beat"], height=.7, color=STATE_COLORS[a["state"]],
                   edgecolor="k", lw=.4, alpha=.45 + .55 * a["recognizability"])
        if a["span"] > 3:
            ax[2].text(a["start_beat"] + a["span"] / 2, y, f"{a['recognizability']:.2f}", ha="center", va="center", fontsize=6)
    ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names)
    used = sorted({a["state"] for a in apps})
    ax[2].legend(handles=[Patch(color=STATE_COLORS[s], label=s) for s in used], loc="upper center",
                 bbox_to_anchor=(.5, -.18), ncol=len(used), fontsize=8, title="estado emocional · el número es la reconocibilidad del motivo", title_fontsize=8)
    ax[2].set_xlabel("tiempo (negras)")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def motif_str(m):
    return " ".join("·" if n.deg is None else f"{n.deg}{'♭' if n.alt < 0 else '♯' if n.alt > 0 else ''}({n.dur:g})" for n in m)


def write_report(path, story, chars, scenes_info, apps, pauses, verified, audio):
    L = [f"# {story['title']}", "",
         "Informe del *leitmotif tracker* generado automáticamente.", "",
         "## Personajes y motivos originales", "",
         "Notación: grado de escala (duración en negras); ♭/♯ = inflexión cromática.", "",
         "| Personaje | Perfil | Instrumento (GM) | Motivo |", "|---|---|---|---|"]
    for n, c in chars.items():
        L.append(f"| {n} | {c['profile']} | {c['program']} | `{motif_str(c['motif'])}` |")
    L += ["", "## Escenas", "", "| # | Escena | Tonalidad | Compás | Tempo | Tensión | Textura | Armonía (acordes por compás) |", "|---|---|---|---|---|---|---|---|"]
    for i, (sc, inf) in enumerate(zip(story["scenes"], scenes_info)):
        L.append(f"| {i + 1} | {sc['title']} | {sc['key']} {sc['mode']} | {sc.get('meter', [4, 4])[0]}/{sc.get('meter', [4, 4])[1]} | {sc['bpm']} | {sc['tension']} | {sc.get('texture', 'alternar')} | {' '.join(inf['chords'])} |")
    L += ["", "## Biografía de cada motivo", ""]
    for n in chars:
        L += [f"### {n}", "", "| Escena | Cmp | Estado | Transformaciones aplicadas | Retiradas por reparación | Reconocibilidad | Verificado en MIDI |", "|---|---|---|---|---|---|---|"]
        for a in [x for x in apps if x["character"] == n]:
            L.append(f"| {a['scene'] + 1}. {a['scene_title']} | {a['bar_in_scene']} | {a['state']} | {', '.join(a['transformations']) or '—'} | {', '.join(a['dropped']) or '—'} | {a['recognizability']:.2f} | {'✓' if a.get('verified') else '✗'} |")
        L.append("")
    sc_all = [a["recognizability"] for a in apps]
    rep = [a for a in apps if a["dropped"]]
    L += ["## Resumen del rastreador", "",
          f"- Apariciones de leitmotifs: **{len(apps)}**",
          f"- Reconocibilidad media: **{sum(sc_all) / len(sc_all):.2f}** (mínima {min(sc_all):.2f})",
          f"- Variantes reparadas automáticamente: **{len(rep)}**",
          f"- Verificación leyendo el MIDI final: **{verified[0]}/{verified[1]}** apariciones suenan exactamente como se registraron", ""]
    L += ["## Silencios y giros dramáticos", ""]
    if pauses:
        for p in pauses:
            L.append(f"- Tras la escena {p['after'] + 1} («{story['scenes'][p['after']]['title']}»): silencio de **{p['beats']:g} negras** ({p['why']}).")
    else:
        L.append("- (ninguno)")
    L += ["", "## Archivos", "",
          "- `poema_sinfonico.mid` — partitura MIDI (marcadores de escena y de cada aparición de leitmotif en la pista de dirección)",
          f"- `{os.path.basename(audio)}` — render con soundfont GM" if audio else "- (sin render de audio)",
          "- `timeline.png` — tensión, piano roll por personaje y estado de cada motivo",
          "- `leitmotifs.json` — registro completo de apariciones",
          "- `story.json` — guion editable; vuelve a ejecutar `poema_sinfonico.py` para recomponer", ""]
    open(path, "w", encoding="utf-8").write("\n".join(L))


# ──────────────────────────────────────────────────────────────────────────
# 7. GUION POR DEFECTO
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_STORY = {
    "title": "La sombra y el caballero — poema sinfónico",
    "seed": 7,
    "recognition_threshold": 0.6,
    "characters": {
        "Aldric": {"profile": "heroico", "program": 60, "double": {"program": 56}, "color": "#d98e04"},
        "Mira":   {"profile": "lirico", "program": 68, "double": {"program": 40}, "color": "#c2418a"},
        "Vorn":   {"profile": "sombrio", "program": 57, "double": {"program": 58}, "color": "#4b3f9e"},
    },
    "scenes": [
        {"title": "Amanecer en el reino", "key": "D", "mode": "major", "bpm": 76, "meter": [4, 4], "bars": 12,
         "tension": 0.20, "texture": "alternar", "ending": "authentic",
         "cast": [["Aldric", "sereno"], ["Mira", "sereno"]]},
        {"title": "Llega la sombra", "key": "D", "mode": "minor", "bpm": 84, "meter": [4, 4], "bars": 12,
         "tension": 0.55, "texture": "alternar", "ending": "half",
         "cast": [["Vorn", "amenazante"], ["Aldric", "inquieto"]]},
        {"title": "La traición", "key": "B", "mode": "phrygian", "bpm": 96, "meter": [4, 4], "bars": 10,
         "tension": 0.80, "texture": "alternar", "ending": "deceptive",
         "cast": [["Aldric", "traicionado"], ["Vorn", "triunfal"]]},
        {"title": "Tormenta (el duelo)", "key": "D", "mode": "minor", "bpm": 132, "meter": [6, 8], "bars": 12,
         "tension": 0.95, "texture": "stretto", "ending": "half", "crash": [0, 6],
         "cast": [["Aldric", "desafiante"], ["Vorn", "amenazante"]]},
        {"title": "Silencio", "key": "D", "mode": "dorian", "bpm": 52, "meter": [4, 4], "bars": 6,
         "tension": 0.12, "texture": "alternar", "ending": "open", "pad_level": 0.5, "pause_after": 4,
         "cast": [["Aldric", "herido"], ["Mira", "desolado"]]},
        {"title": "Lamento de Mira", "key": "G", "mode": "minor", "bpm": 58, "meter": [4, 4], "bars": 10,
         "tension": 0.30, "texture": "alternar", "ending": "half",
         "cast": [["Mira", "desolado"], ["Aldric", "herido"]]},
        {"title": "Redención", "key": "D", "mode": "major", "bpm": 72, "meter": [4, 4], "bars": 16,
         "tension": 0.70, "texture": "canon", "canon_delay_bars": 2, "ending": "plagal", "snap": True,
         "choir": True, "final": True,
         "cast": [["Aldric", "redimido"], ["Mira", "redimido"], ["Vorn", "redimido"]]},
    ],
}


# ──────────────────────────────────────────────────────────────────────────
# 8. PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--story", default=None, help="guion JSON (por defecto, el incluido)")
    ap.add_argument("--out", default=".", help="carpeta de salida")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--write-story", metavar="FILE", help="volcar el guion por defecto y salir")
    a = ap.parse_args()
    if a.write_story:
        json.dump(DEFAULT_STORY, open(a.write_story, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("guion escrito en", a.write_story); return
    story = json.load(open(a.story, encoding="utf-8")) if a.story else DEFAULT_STORY
    os.makedirs(a.out, exist_ok=True)
    seed = a.seed if a.seed is not None else story.get("seed", 1)
    rng = random.Random(seed)
    thr = story.get("recognition_threshold", .5)

    # voces ------------------------------------------------------------------
    voices, chan = {}, [0]

    def new_voice(name, kind, program, rng_=None):
        c = chan[0]
        if kind == "drums":
            ch = 9
        else:
            if c == 9: c += 1
            ch = c; chan[0] = c + 1
        lo, hi, ce = INSTR.get(program, (48, 84, 66))
        v = dict(name=name, kind=kind, program=program, channel=ch, lo=lo, hi=hi, center=ce, events=[])
        voices[name] = v
        return v

    chars = {}
    for name, cfg in story["characters"].items():
        motif = make_motif(cfg["profile"], rng, cfg.get("motif"))
        ch = dict(profile=cfg["profile"], program=cfg["program"], motif=motif, color=cfg.get("color", "#555"),
                  double=cfg.get("double"))
        ch["voice"] = new_voice(name, "lead", cfg["program"])
        if cfg.get("double"):
            ch["dvoice"] = new_voice(name + " (doble)", "double", cfg["double"]["program"])
        chars[name] = ch
    new_voice("pad", "acc", 48); new_voice("bass", "acc", 43); new_voice("harp", "acc", 46)
    new_voice("timp", "acc", 47); new_voice("choir", "acc", 52); new_voice("drums", "drums", 0)

    # giros dramáticos y silencios -------------------------------------------
    sc_list = story["scenes"]
    pauses, pause_after = [], [s.get("pause_after", 0) for s in sc_list]
    for i in range(1, len(sc_list)):
        d = sc_list[i]["tension"] - sc_list[i - 1]["tension"]
        if abs(d) >= .3:
            auto = round(1 + 4 * abs(d), 1)
            if auto > pause_after[i - 1]:
                pause_after[i - 1] = auto
                pauses.append(dict(after=i - 1, beats=auto, why=f"giro dramático: la tensión pasa de {sc_list[i - 1]['tension']:.2f} a {sc_list[i]['tension']:.2f}"))
            else:
                pauses.append(dict(after=i - 1, beats=pause_after[i - 1], why="silencio indicado en el guion"))
    for i, s in enumerate(sc_list):
        if s.get("pause_after") and not any(p["after"] == i for p in pauses):
            pauses.append(dict(after=i, beats=s["pause_after"], why="silencio indicado en el guion"))
    pauses.sort(key=lambda p: p["after"])

    # componer ---------------------------------------------------------------
    g = 0.0
    tempo, ts, markers, scenes_info = [], [], [], []
    starts, ends, tension_pts, apps = [], [], [], []
    for si, sc in enumerate(sc_list):
        num, den = sc.get("meter", [4, 4])
        info = compose_scene(si, sc, story, chars, voices, g, rng, thr, None)
        scenes_info.append(info)
        bb, bars = info["bar_beats"], info["bars"]
        starts.append(g); ends.append(g + bars * bb)
        tension_pts += [(g, sc["tension"]), (g + bars * bb, sc["tension"])]
        ts.append((g, num, den))
        markers.append((g, f"{si + 1}. {sc['title']} [{sc['key']} {sc['mode']}]"))
        # tempo + ritardando final
        tempo.append((g, sc["bpm"]))
        nxt = sc_list[si + 1]["bpm"] if si + 1 < len(sc_list) else None
        target = sc["bpm"] * .8 if (pause_after[si] > 0 or nxt is None) else nxt
        rit_beats = min(2, bars) * bb
        for k in range(1, 7):
            frac = (k / 6) ** 1.5
            tempo.append((g + bars * bb - rit_beats + rit_beats * (k - 1) / 6, sc["bpm"] + (target - sc["bpm"]) * frac))
        for x in info["apps"]:
            markers.append((x["start_beat"], f"{x['character']}:{x['state']} ({x['recognizability']:.2f})"))
        apps += info["apps"]
        g += bars * bb + pause_after[si]
    # quitar tempos duplicados en el mismo pulso (conservar el último)
    tdict = {}
    for b, bpm in tempo:
        tdict[round(b, 4)] = bpm
    tempo = sorted(tdict.items())

    mid_path = os.path.join(a.out, "poema_sinfonico.mid")
    write_midi(mid_path, voices, tempo, ts, markers, rng)
    verified = verify_midi(mid_path, apps, {n: n for n in chars})

    audio = None if a.no_audio else render_audio(mid_path, a.out, "poema_sinfonico")
    plot_timeline(os.path.join(a.out, "timeline.png"), story, scenes_info, chars, apps, starts, ends, tension_pts)
    log = [{k: v for k, v in x.items() if k not in ("expected_pitches", "first_onsets")} for x in apps]
    json.dump(dict(title=story["title"], seed=seed, threshold=thr, appearances=log, pauses=pauses),
              open(os.path.join(a.out, "leitmotifs.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_report(os.path.join(a.out, "informe.md"), story, chars, scenes_info, apps, pauses, verified, audio)
    sc = [x["recognizability"] for x in apps]
    print(f"escenas={len(sc_list)} apariciones={len(apps)} reconocibilidad media={sum(sc) / len(sc):.2f} "
          f"mín={min(sc):.2f} reparadas={sum(1 for x in apps if x['dropped'])} verificadas={verified[0]}/{verified[1]}")
    print(f"duración total ≈ {g:.0f} negras · audio: {audio}")


if __name__ == "__main__":
    main()

# {
#   "title": "La sombra y el caballero — poema sinfónico",
#   "seed": 7,
#   "recognition_threshold": 0.6,
#   "characters": {
#     "Aldric": {
#       "profile": "heroico",
#       "program": 60,
#       "double": {
#         "program": 56
#       },
#       "color": "#d98e04"
#     },
#     "Mira": {
#       "profile": "lirico",
#       "program": 68,
#       "double": {
#         "program": 40
#       },
#       "color": "#c2418a"
#     },
#     "Vorn": {
#       "profile": "sombrio",
#       "program": 57,
#       "double": {
#         "program": 58
#       },
#       "color": "#4b3f9e"
#     }
#   },
#   "scenes": [
#     {
#       "title": "Amanecer en el reino",
#       "key": "D",
#       "mode": "major",
#       "bpm": 76,
#       "meter": [
#         4,
#         4
#       ],
#       "bars": 12,
#       "tension": 0.2,
#       "texture": "alternar",
#       "ending": "authentic",
#       "cast": [
#         [
#           "Aldric",
#           "sereno"
#         ],
#         [
#           "Mira",
#           "sereno"
#         ]
#       ]
#     },
#     {
#       "title": "Llega la sombra",
#       "key": "D",
#       "mode": "minor",
#       "bpm": 84,
#       "meter": [
#         4,
#         4
#       ],
#       "bars": 12,
#       "tension": 0.55,
#       "texture": "alternar",
#       "ending": "half",
#       "cast": [
#         [
#           "Vorn",
#           "amenazante"
#         ],
#         [
#           "Aldric",
#           "inquieto"
#         ]
#       ]
#     },
#     {
#       "title": "La traición",
#       "key": "B",
#       "mode": "phrygian",
#       "bpm": 96,
#       "meter": [
#         4,
#         4
#       ],
#       "bars": 10,
#       "tension": 0.8,
#       "texture": "alternar",
#       "ending": "deceptive",
#       "cast": [
#         [
#           "Aldric",
#           "traicionado"
#         ],
#         [
#           "Vorn",
#           "triunfal"
#         ]
#       ]
#     },
#     {
#       "title": "Tormenta (el duelo)",
#       "key": "D",
#       "mode": "minor",
#       "bpm": 132,
#       "meter": [
#         6,
#         8
#       ],
#       "bars": 12,
#       "tension": 0.95,
#       "texture": "stretto",
#       "ending": "half",
#       "crash": [
#         0,
#         6
#       ],
#       "cast": [
#         [
#           "Aldric",
#           "desafiante"
#         ],
#         [
#           "Vorn",
#           "amenazante"
#         ]
#       ]
#     },
#     {
#       "title": "Silencio",
#       "key": "D",
#       "mode": "dorian",
#       "bpm": 52,
#       "meter": [
#         4,
#         4
#       ],
#       "bars": 6,
#       "tension": 0.12,
#       "texture": "alternar",
#       "ending": "open",
#       "pad_level": 0.5,
#       "pause_after": 4,
#       "cast": [
#         [
#           "Aldric",
#           "herido"
#         ],
#         [
#           "Mira",
#           "desolado"
#         ]
#       ]
#     },
#     {
#       "title": "Lamento de Mira",
#       "key": "G",
#       "mode": "minor",
#       "bpm": 58,
#       "meter": [
#         4,
#         4
#       ],
#       "bars": 10,
#       "tension": 0.3,
#       "texture": "alternar",
#       "ending": "half",
#       "cast": [
#         [
#           "Mira",
#           "desolado"
#         ],
#         [
#           "Aldric",
#           "herido"
#         ]
#       ]
#     },
#     {
#       "title": "Redención",
#       "key": "D",
#       "mode": "major",
#       "bpm": 72,
#       "meter": [
#         4,
#         4
#       ],
#       "bars": 16,
#       "tension": 0.7,
#       "texture": "canon",
#       "canon_delay_bars": 2,
#       "ending": "plagal",
#       "snap": true,
#       "choir": true,
#       "final": true,
#       "cast": [
#         [
#           "Aldric",
#           "redimido"
#         ],
#         [
#           "Mira",
#           "redimido"
#         ],
#         [
#           "Vorn",
#           "redimido"
#         ]
#       ]
#     }
#   ]
# }
