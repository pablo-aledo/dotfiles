#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  grammar_sampler.py — Generación por muestreo de ELEMENTOS DE GRAMÁTICA  ║
# ║                                                                          ║
# ║  Idea: en vez de muestrear nota a nota, se aprende una gramática Re-Pair ║
# ║  sobre la secuencia de COMPASES de un corpus (cada compás, normalizado a ║
# ║  la tónica, es un símbolo terminal) y se muestrea a nivel de ELEMENTO    ║
# ║  (compás suelto o regla de 2..N compases = frase/motivo recurrente).     ║
# ║  Se espera mejor fraseo y estructura que con muestreo de notas, porque   ║
# ║  cada elemento trae consigo repetición, ritmo y cadencia ya "resueltos". ║
# ║                                                                          ║
# ║  Pipeline:                                                               ║
# ║    MIDI → melodía (skyline) → cuantización → compases → tokens           ║
# ║    (grado_relativo_a_la_tónica, duración)  [tónica por Krumhansl]        ║
# ║    → Re-Pair sobre IDs de compás (sin cruzar piezas)                     ║
# ║    → elementos + estadísticas (ocurrencias, posición en la frase,        ║
# ║      ¿final de pieza?, intervalo de frontera, rasgos, bigramas)          ║
# ║    → muestreo: softmax( Σ w_i · puntuación_i / T ) sobre los elementos   ║
# ║      que caben en lo que queda; salida MIDI y/o ABC                      ║
# ║                                                                          ║
# ║  Subcomandos:                                                            ║
# ║    synth      corpus MIDI sintético con estructura conocida (--share:    ║
# ║               0 = cada pieza con material propio, 1 = muy idiomático)    ║
# ║    learn      MIDIs → modelo .json (compases normalizados por modo y     ║
# ║               compás); la gramática se reconstruye al cargar             ║
# ║    info       estadísticas de la gramática: reglas, cobertura, top       ║
# ║    generate   muestrea piezas (--sampler element|bar|note) → .mid/.abc   ║
# ║    eval       métricas de estructura de MIDIs frente al corpus           ║
# ║    bench      comparativa de samplers sobre corpus sintético (tablas;    ║
# ║               suites main|sharing|size|meter|ablate|temp|form|ambig|     ║
# ║               novelty|robust|all)                                        ║
# ║    selftest   pruebas de integridad (MIDI, Re-Pair, validez, semilla…)   ║
# ║                                                                          ║
# ║  Ejemplos:                                                               ║
# ║    ./grammar_sampler.py synth -o corpus/ -n 80 --share 0.8               ║
# ║    ./grammar_sampler.py learn corpus/ -o modelo.json                     ║
# ║    ./grammar_sampler.py info modelo.json --top 10                        ║
# ║    ./grammar_sampler.py generate modelo.json -o pieza.mid --abc pieza.abc║
# ║         --bars 32 --seed 3 --form AABB --max-elem 8 --temp 0.8           ║
# ║    ./grammar_sampler.py generate modelo.json -o base.mid --sampler note  ║
# ║    ./grammar_sampler.py eval modelo.json pieza.mid base.mid              ║
# ║    ./grammar_sampler.py bench --suite main --pieces 40                   ║
# ║    ./grammar_sampler.py bench --suite all --judge abc_predictor_v2.py    ║
# ║                                                                          ║
# ║  Samplers:                                                               ║
# ║    element  grafo de elementos de la gramática (reglas + compases)       ║
# ║    bar      igual pero solo compases sueltos (ablación: sin gramática)   ║
# ║    note     n-gram Witten-Bell sobre (grado, duración), con compases     ║
# ║             exactos y cadencia en la tónica (línea base a nivel de nota) ║
# ║                                                                          ║
# ║  Mandos de generate (ver --help):                                        ║
# ║    --max-elem N    tamaño máx. de elemento, en compases (1 = compases)   ║
# ║    --inventory repair|ngram|bars    --align (reglas sin cruzar frases)   ║
# ║    --w succ=,bound=,compat=,pos=,occ=,len=,fin=  pesos de la puntuación  ║
# ║      · succ↓ → más novedad y menos estructura (bench --suite novelty)    ║
# ║      · pos=0 hunde cadencias y fraseo: es el término más importante      ║
# ║    --temp T  --form AABA --vary p  --tonic Bb  --tempo 120  -n piezas    ║
# ║                                                                          ║
# ║  Limitaciones: monofónico (skyline), sin anacrusa, ligaduras entre       ║
# ║  compases se parten en notas repetidas, sin tresillos (rejilla fija).    ║
# ║  Solo numpy. Con --judge usa score_piece() de abc_predictor_v2.py.       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import argparse
import glob
import importlib.util
import json
import math
import os
import random
import struct
import sys
import tempfile
import time
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

# ── ANSI ────────────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def bold(s):   return _c("1", s)
def dim(s):    return _c("2", s)
def green(s):  return _c("32", s)
def yellow(s): return _c("33", s)
def red(s):    return _c("31", s)
def cyan(s):   return _c("36", s)

def die(msg):
    print(red(f"error: {msg}"), file=sys.stderr)
    sys.exit(1)

def need(cond, msg):
    if not cond:
        die(msg)

REST = 99                                   # "altura" de un silencio en los tokens
SCALES = {"maj": (0, 2, 4, 5, 7, 9, 11), "min": (0, 2, 3, 5, 7, 8, 10)}
THIRD = {"maj": 4, "min": 3}
KK_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
FORMAT = "grammar_sampler/1"

# ═══════════════════════════════════════════════════════════════════════════
#  MIDI (lector y escritor SMF mínimos, sin dependencias)
# ═══════════════════════════════════════════════════════════════════════════
def _read_vlq(data, i):
    v = 0
    while True:
        b = data[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i

def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))

def read_midi(path, track=None):
    """→ dict(ppq, qpm, ts, key, notes[(onset_beats, dur_beats, pitch, vel)], tracks=n_notas_por_pista)."""
    with open(path, "rb") as fh:
        data = fh.read()
    need(data[:4] == b"MThd", f"{path}: no es un fichero MIDI")
    hlen = struct.unpack(">I", data[4:8])[0]
    _fmt, ntr, div = struct.unpack(">HHH", data[8:14])
    need(not div & 0x8000, f"{path}: división SMPTE no soportada")
    i = 8 + hlen
    tempo = ts = key = None
    tracks = []
    for _ in range(ntr):
        if data[i:i + 4] != b"MTrk":
            break
        n = struct.unpack(">I", data[i + 4:i + 8])[0]
        j, end = i + 8, i + 8 + n
        i = end
        t, run, active, notes = 0, None, {}, []
        while j < end:
            dt, j = _read_vlq(data, j)
            t += dt
            st = data[j]
            if st == 0xFF:
                mt = data[j + 1]
                ln, j2 = _read_vlq(data, j + 2)
                body = data[j2:j2 + ln]
                j = j2 + ln
                if mt == 0x51 and tempo is None and ln == 3:
                    tempo = int.from_bytes(body, "big")
                elif mt == 0x58 and ts is None and ln >= 2:
                    ts = (body[0], 1 << body[1])
                elif mt == 0x59 and key is None and ln == 2:
                    key = (struct.unpack("b", body[:1])[0], body[1])
                elif mt == 0x2F:
                    break
                continue
            if st in (0xF0, 0xF7):
                ln, j2 = _read_vlq(data, j + 1)
                j = j2 + ln
                continue
            if st & 0x80:
                run = st
                j += 1
            need(run is not None, f"{path}: MIDI corrupto (running status)")
            kind, ch = run & 0xF0, run & 0x0F
            if kind in (0xC0, 0xD0):
                j += 1
                continue
            d1, d2 = data[j], data[j + 1]
            j += 2
            if kind == 0x90 and d2 > 0:
                active.setdefault((ch, d1), []).append((t, d2))
            elif kind == 0x80 or (kind == 0x90 and d2 == 0):
                lst = active.get((ch, d1))
                if lst:
                    s, v = lst.pop(0)
                    if ch != 9:
                        notes.append((s / div, max(t - s, 1) / div, d1, v))
        tracks.append(sorted(notes))
    need(tracks, f"{path}: sin pistas")
    if track is None:
        track = max(range(len(tracks)), key=lambda k: len(tracks[k]))
    need(0 <= track < len(tracks), f"{path}: la pista {track} no existe")
    return {"ppq": div, "qpm": 60e6 / tempo if tempo else 120.0, "ts": ts or (4, 4), "key": key,
            "notes": tracks[track], "tracks": [len(x) for x in tracks]}

def write_midi(path, notes, ppq=480, qpm=120.0, ts=(4, 4), key=None, title=None):
    """notes: [(onset_beats, dur_beats, pitch, vel)] → SMF formato 0."""
    ev = [(0, 0, b"\xff\x51\x03" + int(round(60e6 / qpm)).to_bytes(3, "big")),
          (0, 0, b"\xff\x58\x04" + bytes([ts[0], int(round(math.log2(ts[1]))), 24, 8]))]
    if key is not None:
        ev.append((0, 0, b"\xff\x59\x02" + struct.pack("bB", key[0], key[1])))
    if title:
        tb = title.encode("latin-1", "replace")
        ev.append((0, 0, b"\xff\x03" + _vlq(len(tb)) + tb))
    for on, dur, p, v in notes:
        s = int(round(on * ppq))
        e = max(int(round((on + dur) * ppq)), s + 1)
        ev.append((s, 2, bytes([0x90, p, v])))
        ev.append((e, 1, bytes([0x80, p, 0])))
    ev.sort(key=lambda x: (x[0], x[1]))
    out, last = bytearray(), 0
    for t, _, b in ev:
        out += _vlq(t - last) + b
        last = t
    out += b"\x00\xff\x2f\x00"
    with open(path, "wb") as fh:
        fh.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, ppq) + b"MTrk" + struct.pack(">I", len(out)) + bytes(out))

# ═══════════════════════════════════════════════════════════════════════════
#  ANÁLISIS: melodía → compases → tokens (grado relativo a la tónica, duración)
# ═══════════════════════════════════════════════════════════════════════════
def melody_line(notes, grid, legato=0.3, min_rest=2):
    """Skyline monofónico cuantizado: [(inicio_u, fin_u, altura)] sin solapes (u = unidades de rejilla).
    Los huecos de articulación (≤ max(1/4 pulso, legato·duración) o < min_rest unidades) se rellenan; los silencios reales se conservan."""
    q = sorted(([int(round(on / grid)), on, on + dur, p] for on, dur, p, *_ in notes), key=lambda x: (x[0], -x[3]))
    sky = []
    for it in q:
        if not sky or sky[-1][0] != it[0]:
            sky.append(it)
    for k in range(len(sky) - 1):
        gap = sky[k + 1][1] - sky[k][2]
        if 0 < gap <= max(0.25, legato * (sky[k][2] - sky[k][1])):
            sky[k][2] = sky[k + 1][1]
    out = []
    for k, (s, on, end, p) in enumerate(sky):
        e = max(int(round(end / grid)), s + 1)
        if k + 1 < len(sky):
            e = min(e, sky[k + 1][0])
            if 0 < sky[k + 1][0] - e < min_rest:
                e = sky[k + 1][0]
        out.append((s, e, p))
    return out

def estimate_key(line, final_bonus=0.5):
    """Tónica (pc) y modo por correlación con perfiles de Krumhansl (+ bonus a la nota final)."""
    h = np.zeros(12)
    for s, e, p in line:
        h[p % 12] += e - s
    best = (-9.0, 0, "maj")
    for mode, prof in (("maj", KK_MAJ), ("min", KK_MIN)):
        for t in range(12):
            r = float(np.corrcoef(h, np.roll(prof, t))[0, 1]) if h.std() > 0 else 0.0
            if line[-1][2] % 12 == t:
                r += final_bonus
            if line[0][2] % 12 == t:
                r += 0.05
            if r > best[0]:
                best = (r, t, mode)
    return best[1], best[2]

def split_bars(line, U, base):
    """→ (compases como tuplas de tokens (rel, unidades), nº de notas partidas por barra)."""
    total = line[-1][1]
    nb = -(-total // U)
    bars = [[] for _ in range(nb)]
    split = 0

    def put(a, b, rel):
        nonlocal split
        while a < b:
            bi = a // U
            end = min(b, (bi + 1) * U)
            bars[bi].append((rel, end - a))
            if end < b and rel != REST:
                split += 1
            a = end
    cur = 0
    for s, e, p in line:
        if s > cur:
            put(cur, s, REST)
        put(s, e, p - base)
        cur = e
    if cur < nb * U:
        put(cur, nb * U, REST)
    return [tuple(b) for b in bars], split

def tune_from_midi(path, grid=0.25, anchor="final", track=None, legato=0.3, force_mode=None, force_tonic=None, min_rest=2):
    """Analiza un MIDI. Devuelve dict(name, mode, tonic_pc, num, den, qpm, bars, base, split) o None."""
    m = read_midi(path, track)
    num, den = m["ts"]
    bar_beats = 4.0 * num / den
    U = int(round(bar_beats / grid))
    if abs(U * grid - bar_beats) > 1e-6 or U < 2:
        return None
    line = melody_line(m["notes"], grid, legato, min_rest)
    if len(line) < 4:
        return None
    tonic, mode = estimate_key(line)
    if force_mode:                                    # modo impuesto (eval --part); tónica = --tonic o la nota final
        mode, tonic = force_mode, (line[-1][2] % 12 if force_tonic is None else force_tonic)
    last = line[-1][2]
    if anchor == "final" and last % 12 == tonic:
        base = last
    else:
        med = float(np.median([p for _, _, p in line]))
        base = tonic + 12 * int(round((med - tonic) / 12))
    bars, split = split_bars(line, U, base)
    return {"name": os.path.basename(path), "mode": mode, "tonic_pc": tonic, "num": num, "den": den,
            "qpm": m["qpm"], "bars": bars, "base": base, "split": split}

# ═══════════════════════════════════════════════════════════════════════════
#  MODELO: particiones homogéneas (modo, compás) con vocabulario de compases
# ═══════════════════════════════════════════════════════════════════════════
class Part:
    """Conjunto de piezas del mismo modo y compás. Cada compás distinto es un símbolo (ID)."""
    def __init__(self, mode, num, den, grid):
        self.mode, self.num, self.den, self.grid = mode, num, den, grid
        self.U = int(round(4.0 * num / den / grid))
        self.vocab, self.vid, self.tunes = [], {}, []

    @property
    def label(self):
        return f"{self.mode}-{self.num}/{self.den}"

    def bar_id(self, bar):
        k = self.vid.get(bar)
        if k is None:
            k = self.vid[bar] = len(self.vocab)
            self.vocab.append(bar)
        return k

    def add(self, name, bars, qpm=120.0, tonic_pc=0):
        self.tunes.append({"name": name, "bars": [self.bar_id(b) for b in bars], "qpm": qpm, "tonic_pc": tonic_pc})

    def tune_bars(self, k):
        return [self.vocab[i] for i in self.tunes[k]["bars"]]

    def subset(self, idx):
        p = Part(self.mode, self.num, self.den, self.grid)
        for k in idx:
            t = self.tunes[k]
            p.add(t["name"], self.tune_bars(k), t["qpm"], t["tonic_pc"])
        return p

def build_parts(tunes, grid):
    parts = {}
    for t in tunes:
        key = (t["mode"], t["num"], t["den"])
        if key not in parts:
            parts[key] = Part(t["mode"], t["num"], t["den"], grid)
        parts[key].add(t["name"], t["bars"], t["qpm"], t["tonic_pc"])
    return {p.label: p for p in parts.values()}

def save_model(path, parts, grid):
    obj = {"format": FORMAT, "grid": grid, "parts": {}}
    for lab, p in parts.items():
        obj["parts"][lab] = {"mode": p.mode, "num": p.num, "den": p.den,
                             "vocab": [[list(t) for t in b] for b in p.vocab], "tunes": p.tunes}
    with open(path, "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))

def load_model(path):
    with open(path) as fh:
        obj = json.load(fh)
    need(obj.get("format") == FORMAT, f"{path}: no es un modelo de grammar_sampler")
    parts = {}
    for lab, d in obj["parts"].items():
        p = Part(d["mode"], d["num"], d["den"], obj["grid"])
        for b in d["vocab"]:
            p.bar_id(tuple((int(r), int(u)) for r, u in b))
        p.tunes = d["tunes"]
        parts[lab] = p
    return parts, obj["grid"]

def pick_part(parts, label=None):
    need(parts, "el modelo está vacío")
    if label:
        need(label in parts, f"partición {label!r} inexistente (hay: {', '.join(parts)})")
        return parts[label]
    return max(parts.values(), key=lambda p: sum(len(t["bars"]) for t in p.tunes))

# ═══════════════════════════════════════════════════════════════════════════
#  GRAMÁTICA: Re-Pair sobre IDs de compás + elementos + estadísticas
# ═══════════════════════════════════════════════════════════════════════════
def repair(seqs, V, min_pair=2, max_rules=None):
    """Re-Pair sobre secuencias de enteros 0..V-1 (una por pieza; los pares no cruzan piezas).
    → (reglas {símbolo: (a, b)} con símbolos ≥ V, secuencias parseadas)."""
    cur = []
    for s in seqs:
        cur.extend(s)
        cur.append(-1)
    cur = np.array(cur, dtype=np.int64)
    rules, banned, nxt = {}, [], V
    while max_rules is None or len(rules) < max_rules:
        a, b = cur[:-1], cur[1:]
        ok = (a >= 0) & (b >= 0)
        for x, y in banned:
            ok &= ~((a == x) & (b == y))
        if not ok.any():
            break
        M = int(cur.max()) + 2
        uniq, cnt = np.unique(a[ok] * M + b[ok], return_counts=True)
        j = int(np.argmax(cnt))                       # empate → par menor (determinista)
        if cnt[j] < min_pair:
            break
        ka, kb = int(uniq[j] // M), int(uniq[j] % M)
        sel, last = [], -2
        for p in np.nonzero(ok & (a == ka) & (b == kb))[0].tolist():
            if p > last + 1:                          # sin solapes (aaa → un solo par)
                sel.append(p)
                last = p
        if len(sel) < min_pair:
            banned.append((ka, kb))
            continue
        sel = np.array(sel)
        cur[sel] = nxt
        keep = np.ones(len(cur), bool)
        keep[sel + 1] = False
        cur = cur[keep]
        rules[nxt] = (ka, kb)
        nxt += 1
    parsed, row = [], []
    for v in cur.tolist():
        if v == -1:
            parsed.append(row)
            row = []
        else:
            row.append(v)
    return rules, parsed

def expand_rules(rules, V):
    exp = {}
    for s in sorted(rules):                           # los hijos siempre tienen ID menor
        a, b = rules[s]
        exp[s] = ([a] if a < V else exp[a]) + ([b] if b < V else exp[b])
    return exp

def bar_features(bar, U):
    """Vector de compatibilidad de un compás (clases de altura, intervalos, densidad, registro…)."""
    pc, pit, rest = np.zeros(12), [], 0
    for rel, u in bar:
        if rel == REST:
            rest += u
        else:
            pc[rel % 12] += u
            pit.append(rel)
    ih = np.zeros(5)
    for x, y in zip(pit, pit[1:]):
        d = abs(y - x)
        ih[0 if d == 0 else 1 if d <= 2 else 2 if d <= 4 else 3 if d <= 7 else 4] += 1
    v = np.concatenate([pc / pc.sum() if pc.sum() else pc, 0.7 * ih / ih.sum() if ih.sum() else ih,
                        [min(1.0, len(pit) / (U / 2.0)), rest / U,
                         0.5 * (np.mean(pit) / 12.0 if pit else 0.0), 0.5 * ((max(pit) - min(pit)) / 12.0 if pit else 0.0)]])
    n = np.linalg.norm(v)
    return v / n if n else v

class Grammar:
    """Gramática de un Part. Elementos = compases sueltos + (según `inventory`) secuencias de compases:
       repair → reglas de Re-Pair de ≤ max_elem compases | ngram → toda secuencia repetida | bars → solo compases."""
    def __init__(self, part, pb=4, min_pair=2, max_rules=600, max_elem=8, inventory="repair", align=False, posmode="bar"):
        self.part, self.pb, self.U, self.align = part, pb, part.U, align
        V = self.V = len(part.vocab)
        seqs = [t["bars"] for t in part.tunes]
        need(V > 0 and seqs, "corpus vacío")
        self.rules, self.parsed = {}, seqs
        cand = set()
        if inventory == "repair":
            # align=True: Re-Pair por frases (separador cada pb compases) → ninguna regla cruza una frontera de frase
            chunks, owner = [], []
            for ti, s in enumerate(seqs):
                for i in range(0, len(s), pb) if align else [0]:
                    chunks.append(s[i:i + pb] if align else s)
                    owner.append(ti)
            self.rules, pch = repair(chunks, V, min_pair, max_rules)
            self.parsed = [[] for _ in seqs]
            for o, pc in zip(owner, pch):
                self.parsed[o].extend(pc)
            exp = expand_rules(self.rules, V)
            cand = {tuple(e) for e in exp.values() if len(e) <= max_elem}
        elif inventory == "ngram":
            cnt = Counter()
            for s in seqs:
                for n in range(2, max_elem + 1):
                    for i in range(len(s) - n + 1):
                        if not align or i // pb == (i + n - 1) // pb:
                            cnt[tuple(s[i:i + n])] += 1
            cand = {k for k, c in cnt.items() if c >= min_pair}
        else:
            need(inventory == "bars", f"inventario desconocido: {inventory}")
        singles = {(b,) for s in seqs for b in s}
        self.elems = sorted(singles | cand, key=lambda t: (len(t), t))
        E = self.E = len(self.elems)
        by_len = defaultdict(dict)
        for k, e in enumerate(self.elems):
            by_len[len(e)][e] = k
        occ, pos, st0, fin = np.zeros(E), np.zeros((E, pb)), np.zeros(E), np.zeros(E)
        for s in seqs:                                # estadísticas de posición por ocurrencia real
            L = len(s)
            for n, d in by_len.items():
                for i in range(L - n + 1):
                    k = d.get(tuple(s[i:i + n]))
                    if k is not None:
                        occ[k] += 1
                        pos[k, i % pb] += 1
                        st0[k] += (i == 0)
                        fin[k] += (i + n == L)
        a = 0.5
        self.E_n = np.array([len(e) for e in self.elems])
        self.E_head = np.array([e[0] for e in self.elems])
        self.E_tail = np.array([e[-1] for e in self.elems])
        self.occ = occ
        self.occ_lp = np.log(np.maximum(occ, 1.0))
        self.len_lp = np.log(self.E_n.astype(float))
        # posición en la frase / inicio / final: por ELEMENTO ("elem") o compuesta con las estadísticas de sus
        # COMPASES ("bar": suma de log P(pos | compás), así un solo compás mal colocado se castiga entero;
        # el inicio usa el compás cabeza y el final el compás cola).
        # Los elementos raros tienen pocos datos propios; sus compases constituyentes tienen muchos.
        self.pos_lp = np.log((pos + a) / (occ[:, None] + a * pb))
        self.st0_lp = np.log((st0 + a) / (occ + 2 * a))
        self.fin_lp = np.log((fin + a) / (occ + 2 * a))
        self.nfin_lp = np.log((occ - fin + a) / (occ + 2 * a))
        self.pos0_lp = self.st0_lp
        if posmode == "bar":
            bocc, bpos, bst0, bfin = np.zeros(V), np.zeros((V, pb)), np.zeros(V), np.zeros(V)
            for s in seqs:
                for i, b in enumerate(s):
                    bocc[b] += 1
                    bpos[b, i % pb] += 1
                    bst0[b] += (i == 0)
                    bfin[b] += (i == len(s) - 1)
            bpos_lp = np.log((bpos + a) / (bocc[:, None] + a * pb))
            maxn = int(self.E_n.max())
            pad = np.full((E, maxn), -1)
            for k, e in enumerate(self.elems):
                pad[k, :len(e)] = e
            pl = np.zeros((E, pb))
            for p in range(pb):
                for i in range(maxn):
                    m = pad[:, i] >= 0
                    pl[m, p] += bpos_lp[pad[m, i], (p + i) % pb]
            self.pos_lp = pl
            self.st0_lp = np.log((bst0 + a) / (bocc + 2 * a))[self.E_head]
            self.fin_lp = np.log((bfin + a) / (bocc + 2 * a))[self.E_tail]
            self.nfin_lp = np.log((bocc - bfin + a) / (bocc + 2 * a))[self.E_tail]
            self.pos0_lp = self.st0_lp + self.pos_lp[:, 0]
        else:
            need(posmode == "elem", f"posmode desconocido: {posmode}")
        # bordes y rasgos por compás
        head, tail, lastp = np.full(V, np.nan), np.full(V, np.nan), np.full(V, np.nan)
        for i, b in enumerate(part.vocab):
            if b[0][0] != REST:
                head[i] = b[0][0]
            if b[-1][0] != REST:
                tail[i] = b[-1][0]
            pit = [r for r, _ in b if r != REST]
            if pit:
                lastp[i] = pit[-1]
        self.tail = tail
        self.E_headp = head[self.E_head]
        lt = lastp[self.E_tail]
        self.E_lastrel = lt                              # última altura exacta (NaN si el elemento acaba sin notas)
        self.E_lastpc = np.where(np.isnan(lt), -1, np.mod(np.nan_to_num(lt), 12)).astype(int)
        self.F = np.array([bar_features(b, self.U) for b in part.vocab])
        # bigramas de compás (backoff Witten-Bell a la frecuencia de compás) e intervalos de frontera
        uni, self.follow = np.zeros(V), defaultdict(Counter)
        ih, nrest, nb = np.zeros(25), 0, 0
        for s in seqs:
            for i, b in enumerate(s):
                uni[b] += 1
                if i + 1 < len(s):
                    self.follow[b][s[i + 1]] += 1
                    nb += 1
                    if np.isnan(tail[b]) or np.isnan(head[s[i + 1]]):
                        nrest += 1
                    else:
                        ih[int(np.clip(head[s[i + 1]] - tail[b], -12, 12)) + 12] += 1
        self.Pu = (uni + 1.0) / (uni.sum() + V)
        self.cnt_prev = {b: sum(c.values()) for b, c in self.follow.items()}
        r = min(0.98, max(0.02, nrest / nb)) if nb else 0.5
        self.rest_lp = math.log(r)
        self.pit_lp = np.log((1 - r) * (ih + 0.5) / (ih.sum() + 12.5))

    # puntuaciones vectoriales sobre todos los elementos, dado el compás previo
    def succ_vec(self, prev):
        fol = self.follow.get(prev)
        c, u = self.cnt_prev.get(prev, 0), (len(fol) if fol else 0)
        pv = (u / (c + u) if c + u else 1.0) * self.Pu
        if fol:
            for b, k in fol.items():
                pv[b] += k / (c + u)
        return pv[self.E_head]

    def bound_vec(self, prev):
        tp = self.tail[prev]
        if np.isnan(tp):
            return np.zeros(self.E)
        iv = self.E_headp - tp
        lp = np.full(self.E, self.rest_lp)
        m = ~np.isnan(iv)
        lp[m] = self.pit_lp[np.clip(iv[m], -12, 12).astype(int) + 12]
        return lp

    def compat_vec(self, prev):
        return (self.F @ self.F[prev])[self.E_head]

    def describe(self, top=8):
        cov = Counter()
        tot = sum(len(t["bars"]) for t in self.part.tunes)
        exp = expand_rules(self.rules, self.V)
        multi = sum(len(exp[x]) for s in self.parsed for x in s if x >= self.V)
        lens = Counter(int(n) for n in self.E_n)
        out = [f"compases distintos: {self.V}   piezas: {len(self.part.tunes)}   compases totales: {tot}",
               f"reglas Re-Pair: {len(self.rules)}   elementos muestreables: {self.E}   por nº de compases: "
               + ", ".join(f"{n}:{c}" for n, c in sorted(lens.items())),
               f"cobertura de reglas (compases del corpus dentro de una regla del parseo): {multi / max(tot, 1):.1%}"]
        order = np.argsort(-(self.occ * (self.E_n - 1 + 0.01)))[:top]
        out.append(f"{'×':>5} {'n':>2}  contenido (compás → tokens grado:unidades, '-' = silencio)")
        for k in order:
            if self.E_n[k] < 2 and self.occ[k] < 2:
                continue
            e = self.elems[k]
            out.append(f"{int(self.occ[k]):5d} {len(e):2d}  " + " | ".join(
                " ".join(("-" if r == REST else str(r)) + ":" + str(u) for r, u in self.part.vocab[b]) for b in e[:3])
                + (" …" if len(e) > 3 else ""))
        return "\n".join(out)

# ═══════════════════════════════════════════════════════════════════════════
#  SAMPLERS
# ═══════════════════════════════════════════════════════════════════════════
DEF_W = {"succ": 1.0, "bound": 1.0, "compat": 1.0, "pos": 1.0, "occ": 1.0, "len": 0.0, "fin": 1.0}
COMPAT_SCALE = 3.0

class ElementSampler:
    """Muestrea elementos de la gramática: softmax(Σ w·puntuación / T) sobre los que caben en lo que queda."""
    kind = "element"
    def __init__(self, G, temp=1.0, w=None):
        self.G, self.temp, self.trace = G, temp, []
        self.w = dict(DEF_W)
        if w:
            self.w.update(w)

    def extend(self, prefix, nbars, rng, final=False):
        G, w = self.G, self.w
        cur = [G.part.vid.get(b, -1) for b in prefix]
        out, rem = [], nbars
        while rem > 0:
            prev = cur[-1] if cur and cur[-1] >= 0 else None
            sc = np.zeros(G.E)
            if prev is not None:
                if w["succ"]:
                    sc += w["succ"] * np.log(G.succ_vec(prev))
                if w["bound"]:
                    sc += w["bound"] * G.bound_vec(prev)
                if w["compat"]:
                    sc += w["compat"] * COMPAT_SCALE * G.compat_vec(prev)
            if w["pos"]:
                sc += w["pos"] * (G.pos0_lp if not cur else G.pos_lp[:, len(cur) % G.pb])
            if w["occ"]:
                sc += w["occ"] * G.occ_lp
            if w["len"]:
                sc += w["len"] * G.len_lp
            if w["fin"]:
                sc += w["fin"] * np.where((G.E_n == rem) & final, G.fin_lp, G.nfin_lp)
            mask = G.E_n <= rem
            if final:                                    # cierre en la tónica EXACTA (grado 0), si no en su clase de altura
                for ok in (G.E_lastrel == 0, G.E_lastpc == 0):
                    m2 = mask & ((G.E_n != rem) | ok)
                    if m2.any() and (m2 & (G.E_n == rem)).any():
                        mask = m2
                        break
            z = np.where(mask, sc / max(self.temp, 1e-6), -np.inf)
            p = np.exp(z - z.max())
            p /= p.sum()
            k = int(rng.choice(G.E, p=p))
            e = G.elems[k]
            cur.extend(e)
            out.extend(e)
            self.trace.append(len(e))
            rem -= len(e)
        return [G.part.vocab[i] for i in out]

class BarSampler(ElementSampler):
    """Ablación: el mismo grafo pero solo con compases sueltos (sin reglas)."""
    kind = "bar"

class NoteSampler:
    """Línea base: n-gram Witten-Bell sobre tokens (grado, duración) + prior de posición métrica.
    Solo se permiten tokens que caben en el compás; en el último compás la nota final es la tónica."""
    kind = "note"
    def __init__(self, part, order=3, temp=1.0, w_pos=1.0):
        self.part, self.U, self.order, self.temp, self.w_pos = part, part.U, order, temp, w_pos
        self.cnt = [defaultdict(Counter) for _ in range(order + 1)]
        self.pos_u = defaultdict(Counter)
        toks = set()
        for k in range(len(part.tunes)):
            seq = []
            for bar in part.tune_bars(k):
                off = 0
                for t in bar:
                    seq.append(t)
                    self.pos_u[off][t[1]] += 1
                    off += t[1]
            toks.update(seq)
            for i, t in enumerate(seq):
                for kk in range(order + 1):
                    if i - kk >= 0:
                        self.cnt[kk][tuple(seq[i - kk:i])][t] += 1
        self.tokens = sorted(toks)
        self.tot = [{c: sum(v.values()) for c, v in d.items()} for d in self.cnt]
        self.Du = len({u for t in toks for u in (t[1],)})

    def _probs(self, ctx, cands):
        p = {t: 1.0 / (len(self.tokens) + 1) for t in cands}
        for k in range(0, self.order + 1):
            if k > len(ctx):
                break
            c = tuple(ctx[len(ctx) - k:]) if k else ()
            ctr = self.cnt[k].get(c)
            if not ctr:
                continue
            tot, u = self.tot[k][c], len(ctr)
            for t in cands:
                p[t] = (ctr.get(t, 0) + u * p[t]) / (tot + u)
        return p

    def extend(self, prefix, nbars, rng, final=False):
        U = self.U
        ctx = []
        for b in prefix[-3:]:
            ctx.extend(b)
        ctx = ctx[-self.order:] if self.order else []
        out = []
        for bi in range(nbars):
            bar, off = [], 0
            lastbar = final and bi == nbars - 1
            while off < U:
                rem = U - off
                cands = [t for t in self.tokens if t[1] <= rem]
                if lastbar:                              # la nota final es la tónica exacta (grado 0), si no su clase de altura
                    for okf in (lambda t: t[0] == 0, lambda t: t[0] != REST and t[0] % 12 == 0):
                        c2 = [t for t in cands if t[1] != rem or okf(t)]
                        if any(t[1] == rem for t in c2):
                            cands = c2
                            break
                if not cands:
                    tok = (REST, rem)
                else:
                    p = self._probs(tuple(ctx), cands)
                    pu = self.pos_u.get(off, Counter())
                    tp = sum(pu.values())
                    lp = np.array([math.log(p[t]) + self.w_pos * math.log((pu.get(t[1], 0) + 0.5) / (tp + 0.5 * self.Du))
                                   for t in cands]) / max(self.temp, 1e-6)
                    q = np.exp(lp - lp.max())
                    q /= q.sum()
                    tok = cands[int(rng.choice(len(cands), p=q))]
                bar.append(tok)
                off += tok[1]
                if self.order:
                    ctx = (ctx + [tok])[-self.order:]
            out.append(tuple(bar))
        return out

def compose(sampler, nbars, rng, sec_bars=8, form="none", vary=0.0):
    """Pieza de `nbars` compases. Con forma (AABA…), cada letra repetida COPIA la primera aparición de su sección;
    la copia final (o con prob. `vary`) re-muestrea el último compás (final alternativo / cadencia de cierre)."""
    if not form or form.lower() == "none":
        return sampler.extend([], nbars, rng, final=True)
    pat = form.upper()
    nsec = -(-nbars // sec_bars)
    bars, first = [], {}
    for s in range(nsec):
        letter, n, last = pat[s % len(pat)], min(sec_bars, nbars - len(bars)), s == nsec - 1
        if letter not in first:
            first[letter] = len(bars)
            bars += sampler.extend(bars, n, rng, final=last)
            continue
        cp = list(bars[first[letter]:first[letter] + n])
        if len(cp) < n:
            cp += sampler.extend(bars + cp, n - len(cp), rng, final=last)
        elif last or rng.random() < vary:
            cp = cp[:-1] + sampler.extend(bars + cp[:-1], 1, rng, final=last)
        bars += cp
    return bars

# ═══════════════════════════════════════════════════════════════════════════
#  SALIDA: compases → notas → MIDI / ABC
# ═══════════════════════════════════════════════════════════════════════════
NAT = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJ_FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
MAJ_NAMES = {0: "C", 7: "G", 2: "D", 9: "A", 4: "E", 11: "B", 6: "F#", 1: "Db", 8: "Ab", 3: "Eb", 10: "Bb", 5: "F"}
MIN_NAMES = {9: "A", 4: "E", 11: "B", 6: "F#", 1: "C#", 8: "G#", 0: "C", 7: "G", 2: "D", 5: "F", 10: "Bb", 3: "D#"}
PC_NAMES = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7,
            "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}

def parse_tonic(s):
    s = str(s).strip().upper()
    if s.lstrip("-").isdigit():
        return int(s) % 12
    need(s in PC_NAMES, f"tónica no reconocida: {s!r} (ej. C, F#, Bb, 7)")
    return PC_NAMES[s]

def key_fifths(tonic_pc, mode):
    return MAJ_FIFTHS[tonic_pc if mode == "maj" else (tonic_pc + 3) % 12]

def abc_key(tonic_pc, mode):
    return MAJ_NAMES[tonic_pc] if mode == "maj" else MIN_NAMES[tonic_pc] + "m"

def base_midi(tonic_pc):
    return 60 + tonic_pc - (12 if tonic_pc > 6 else 0)

def bars_to_notes(bars, U, grid, base):
    notes = []
    for bi, bar in enumerate(bars):
        off = 0
        for rel, u in bar:
            if rel != REST:
                notes.append(((bi * U + off) * grid, u * grid, base + rel, 88 if off == 0 else 74))
            off += u
    return notes

def write_piece_midi(path, bars, part, tonic_pc, qpm, title=None):
    key = (key_fifths(tonic_pc, part.mode), 0 if part.mode == "maj" else 1)
    write_midi(path, bars_to_notes(bars, part.U, part.grid, base_midi(tonic_pc)), 480, qpm, (part.num, part.den), key, title)

def bars_to_abc(bars, part, tonic_pc, qpm=120.0, title="grammar_sampler", pb=4):
    g = part.grid
    need(g in (1.0, 0.5, 0.25, 0.125), "ABC solo con rejilla 1, 1/2, 1/4 o 1/8 de pulso")
    f = key_fifths(tonic_pc, part.mode)
    acck = {c: 0 for c in "CDEFGAB"}
    for c in ("FCGDAEB"[:f] if f > 0 else "BEADGCF"[:-f]):
        acck[c] = 1 if f > 0 else -1
    diat = {(NAT[c] + acck[c]) % 12: (c, acck[c]) for c in "CDEFGAB"}
    natural = {v: k for k, v in NAT.items()}
    chrom = {1: ("C", 1), 3: ("D", 1), 6: ("F", 1), 8: ("G", 1), 10: ("A", 1)} if f >= 0 else \
            {1: ("D", -1), 3: ("E", -1), 6: ("G", -1), 8: ("A", -1), 10: ("B", -1)}
    base = base_midi(tonic_pc)
    lines, cur = [], []
    for bi, bar in enumerate(bars):
        st, toks = {}, []
        for rel, u in bar:
            ln = str(u) if u != 1 else ""
            if rel == REST:
                toks.append("z" + ln)
                continue
            p = base + rel
            pc = p % 12
            letter, acc = diat.get(pc) or ((natural[pc], 0) if pc in natural else chrom[pc])
            oct_ = (p - acc) // 12 - 1
            cur_acc = st.get((letter, oct_), acck[letter])
            sym = "" if acc == cur_acc else ("^" if acc == 1 else "_" if acc == -1 else "=")
            st[(letter, oct_)] = acc
            name = letter if oct_ <= 4 else letter.lower()
            name += "," * (4 - oct_) if oct_ < 4 else "'" * (oct_ - 5) if oct_ > 5 else ""
            toks.append(sym + name + ln)
        cur.append(" ".join(toks))
        if (bi + 1) % pb == 0 or bi == len(bars) - 1:
            lines.append(" | ".join(cur) + (" |]" if bi == len(bars) - 1 else " |"))
            cur = []
    hdr = [f"X:1", f"T:{title}", f"M:{part.num}/{part.den}", f"L:1/{int(round(4 / g))}",
           f"Q:1/4={int(round(qpm))}", f"K:{abc_key(tonic_pc, part.mode)}"]
    return "\n".join(hdr + lines) + "\n"

# ═══════════════════════════════════════════════════════════════════════════
#  MÉTRICAS (todas sobre compases = tuplas de tokens (rel, unidades))
# ═══════════════════════════════════════════════════════════════════════════
def _js(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        k = a > 0
        return float(np.sum(a[k] * np.log2(a[k] / b[k])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def _hists(bars):
    pc, ih, du, pit = np.zeros(12), np.zeros(25), Counter(), []
    for b in bars:
        for r, u in b:
            du[u] += 1
            if r != REST:
                pc[r % 12] += u
                pit.append(r)
    for x, y in zip(pit, pit[1:]):
        ih[int(np.clip(y - x, -12, 12)) + 12] += 1
    return pc, ih, du

class Judge:
    """n-gram de orden 2 sobre tokens (Witten-Bell); bits/token como medida de 'idiomaticidad'."""
    def __init__(self, part):
        self.c = [defaultdict(Counter) for _ in range(3)]
        toks = set()
        for k in range(len(part.tunes)):
            seq = [t for b in part.tune_bars(k) for t in b]
            toks.update(seq)
            for i, t in enumerate(seq):
                for kk in range(3):
                    if i - kk >= 0:
                        self.c[kk][tuple(seq[i - kk:i])][t] += 1
        self.V = len(toks) + 1
        self.tot = [{c: sum(v.values()) for c, v in d.items()} for d in self.c]

    def bpn(self, flat):
        bits = []
        for i, t in enumerate(flat):
            p = 1.0 / self.V
            for kk in range(3):
                if i - kk < 0:
                    break
                ctx = tuple(flat[i - kk:i])
                ctr = self.c[kk].get(ctx)
                if ctr:
                    u = len(ctr)
                    p = (ctr.get(t, 0) + u * p) / (self.tot[kk][ctx] + u)
            bits.append(-math.log2(p))
        return float(np.mean(bits)) if bits else 0.0

class Ref:
    """Contexto de evaluación construido con el corpus de ENTRENAMIENTO."""
    def __init__(self, part, pb=4):
        self.part, self.pb, self.U, self.mode = part, pb, part.U, part.mode
        self.ng8, self.barset = set(), set(part.vocab)
        allbars = []
        for k in range(len(part.tunes)):
            bars = part.tune_bars(k)
            allbars += bars
            flat = [t for b in bars for t in b]
            for i in range(len(flat) - 7):
                self.ng8.add(tuple(flat[i:i + 8]))
        self.pc, self.ih, self.du = _hists(allbars)
        self.win = {4: set(), 8: set()}                   # ventanas deslizantes de 4 y 8 compases (IDs de compás)
        for t in part.tunes:
            for n in self.win:
                for i in range(len(t["bars"]) - n + 1):
                    self.win[n].add(tuple(t["bars"][i:i + n]))
        self.judge = Judge(part)

def _bar_sets(bar):
    on, mel, rhy = 0, set(), set()
    for rel, u in bar:
        if rel != REST:
            mel.add((on, rel))
            rhy.add(on)
        on += u
    return mel, rhy

def _jac(x, y):
    return 1.0 if not x and not y else len(x & y) / len(x | y)

def bar_sim(A, B):
    return 0.5 * _jac(A[0], B[0]) + 0.5 * _jac(A[1], B[1])

METRIC_KEYS = ["copy", "div", "S2", "S4", "S8", "per", "phr", "nov8", "nov4b", "nov8b", "barnov", "step", "leap", "junc", "rng",
               "scale", "ends", "cad", "jsPC", "jsINT", "jsDUR", "bpn"]
DIST_KEYS = ["copy", "S4", "S8", "per", "phr", "nov8", "nov4b", "nov8b", "step", "leap", "junc", "scale", "ends", "cad", "jsPC", "jsINT", "bpn"]
DIST_FLOOR = {"bpn": 0.3, "rng": 2.0, "jsPC": 0.02, "jsINT": 0.02, "jsDUR": 0.02}

def piece_metrics(bars, ref):
    U, pb, nb = ref.U, ref.pb, len(bars)
    flat = [t for b in bars for t in b]
    m = {}
    seen, rep = set(), 0
    for b in bars:
        rep += b in seen
        seen.add(b)
    m["copy"], m["div"] = rep / max(nb - 1, 1), len(seen) / nb
    sets = [_bar_sets(b) for b in bars]
    S = {}
    for l in range(1, 9):
        v = [bar_sim(sets[i], sets[i - l]) for i in range(l, nb)]
        S[l] = float(np.mean(v)) if v else 0.0
    m["S2"], m["S4"], m["S8"] = S[2], S[4], S[8]
    m["per"] = float(np.mean([S[4], S[8]]) - np.mean([S[3], S[5], S[6], S[7]]))
    nw, rp = nb // pb, 0
    for j in range(1, nw):
        best = max(float(np.mean([bar_sim(sets[j * pb + q], sets[jj * pb + q]) for q in range(pb)])) for jj in range(j))
        rp += best >= 0.6
    m["phr"] = rp / max(nw - 1, 1)
    cov = np.zeros(len(flat), bool)
    for i in range(len(flat) - 7):
        if tuple(flat[i:i + 8]) in ref.ng8:
            cov[i:i + 8] = True
    m["nov8"] = 1.0 - float(cov.mean()) if len(flat) else 0.0
    m["barnov"] = float(np.mean([b not in ref.barset for b in bars]))
    ids = [ref.part.vid.get(b, -1) for b in bars]
    for n, key in ((4, "nov4b"), (8, "nov8b")):
        ws = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
        m[key] = float(np.mean([w not in ref.win[n] for w in ws])) if ws else 1.0
    ps = [r for r, u in flat if r != REST]
    iv = [y - x for x, y in zip(ps, ps[1:])]
    m["step"] = float(np.mean([abs(x) <= 2 for x in iv])) if iv else 0.0
    m["leap"] = float(np.mean([abs(x) > 7 for x in iv])) if iv else 0.0
    m["rng"] = float(max(ps) - min(ps)) if ps else 0.0
    jl = [abs(bars[i + 1][0][0] - bars[i][-1][0]) > 7 for i in range(nb - 1)
          if bars[i][-1][0] != REST and bars[i + 1][0][0] != REST]
    m["junc"] = float(np.mean(jl)) if jl else 0.0
    sc = set(SCALES[ref.mode])
    pu = sum(u for r, u in flat if r != REST)
    m["scale"] = sum(u for r, u in flat if r != REST and r % 12 in sc) / max(pu, 1)
    m["ends"] = float(bool(ps) and ps[-1] % 12 == 0)
    stable, ends = {0, 7, THIRD[ref.mode]}, []
    for i in range(pb - 1, nb, pb):
        pit = [r for r, _ in bars[i] if r != REST]
        if pit:
            ends.append(pit[-1] % 12 in stable)
    m["cad"] = float(np.mean(ends)) if ends else 0.0
    pc, ih, du = _hists(bars)
    m["jsPC"], m["jsINT"] = _js(pc, ref.pc), _js(ih, ref.ih)
    keys = sorted(set(du) | set(ref.du))
    m["jsDUR"] = _js([du.get(k, 0) for k in keys], [ref.du.get(k, 0) for k in keys])
    m["bpn"] = ref.judge.bpn(flat)
    return m

def ref_stats(test_bars_list, ref):
    rows = [piece_metrics(b, ref) for b in test_bars_list]
    return {k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows]))) for k in METRIC_KEYS}

def dist_to_ref(m, rs):
    """Distancia media (en desviaciones del conjunto de referencia, con suelo y tope 3) a las piezas reales."""
    z = []
    for k in DIST_KEYS:
        mu, sd = rs[k]
        z.append(min(3.0, abs(m[k] - mu) / max(sd, DIST_FLOOR.get(k, 0.05))))
    return float(np.mean(z))

# ═══════════════════════════════════════════════════════════════════════════
#  CORPUS SINTÉTICO con estructura conocida
#    compás = plantilla (ritmo, grados de la escala) · frase = 4 compases (cierre en V/III/II o en I)
#    parte = antecedente + consecuente (8 compases) · pieza = forma de 4 partes (AABB, AABA, ABAC…)
#    --share = prob. de tomar cada frase de la biblioteca de la familia (material idiomático compartido)
#              en vez de crear una frase nueva y única
#    --bar-reuse = prob. de que los compases m/c de una frase salgan de la biblioteca de compases (baja → compases
#              únicos por frase; alta → los mismos compases aparecen en muchas frases distintas = continuación ambigua)
# ═══════════════════════════════════════════════════════════════════════════
RHY = {(4, 4): [(4, 4, 4, 4), (6, 2, 4, 4), (4, 4, 8), (2, 2, 4, 4, 4), (8, 4, 2, 2), (4, 2, 2, 4, 2, 2), (6, 2, 6, 2),
                (2, 2, 2, 2, 4, 4), (4, 4, 4, 2, 2), (8, 8), (6, 2, 4, 2, 2), (3, 1, 4, 4, 4), (4, 2, 2, 2, 2, 4), (2, 2, 4, 2, 2, 4)],
       (3, 4): [(4, 4, 4), (8, 4), (6, 2, 4), (4, 2, 2, 4), (2, 2, 2, 2, 4), (4, 8), (6, 2, 2, 2), (2, 2, 4, 4), (12,), (4, 4, 2, 2)],
       (6, 8): [(2, 2, 2, 2, 2, 2), (6, 6), (4, 2, 4, 2), (2, 2, 2, 6), (6, 2, 2, 2), (4, 2, 2, 4), (2, 4, 2, 4), (4, 2, 2, 2, 2),
                (8, 4), (2, 2, 4, 2, 2)]}
CAD_FULL = {(4, 4): [(4, 4, 8), (4, 2, 2, 8), (6, 2, 8), (8, 8)], (3, 4): [(4, 4, 4), (4, 8), (8, 4), (6, 2, 4)],
            (6, 8): [(6, 6), (4, 2, 6), (2, 4, 6), (8, 4)]}
CAD_HALF = {(4, 4): [(4, 4, 8), (4, 2, 2, 8), (4, 4, 6, -2), (2, 2, 4, 6, -2)], (3, 4): [(4, 4, 4), (4, 6, -2), (8, 4)],
            (6, 8): [(6, 6), (4, 2, 6), (6, 4, -2)]}
STEPS = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
STEP_W = [.02, .03, .09, .28, .10, .28, .09, .03, .02]

def _walk(rng, k, d0, lo=-4, hi=9):
    ds = [d0]
    for _ in range(k - 1):
        s = rng.choices(STEPS, STEP_W)[0]
        d = ds[-1] + s
        if d < lo or d > hi:
            d = ds[-1] - s
        ds.append(min(hi, max(lo, d)))
    return ds

class Family:
    def __init__(self, rng, mode, meter, n_bars=24, n_phr=8, reuse=0.5):
        self.rng, self.mode, self.meter, self.scale, self.reuse = rng, mode, meter, SCALES[mode], reuse
        self.rh = rng.sample(RHY[meter], min(6, len(RHY[meter])))
        self.bars = [self.new_bar() for _ in range(n_bars)]
        self.ante_lib = [self.new_phrase("ante") for _ in range(n_phr)]
        self.cons_lib = [self.new_phrase("cons", base=a) for a in self.ante_lib]

    def new_bar(self):
        rh = self.rng.choice(self.rh)
        return (rh, tuple(_walk(self.rng, sum(r > 0 for r in rh), self.rng.choice([0, 1, 2, 3, 4]))))

    def new_cad(self, kind):
        end = 0 if kind == "full" else self.rng.choices([4, 1, 2], [.7, .15, .15])[0]
        rh = self.rng.choice((CAD_FULL if kind == "full" else CAD_HALF)[self.meter])
        ds = _walk(self.rng, sum(r > 0 for r in rh), end + self.rng.choice([-2, -1, 1, 2]))
        ds[-1] = end
        return (rh, tuple(ds))

    def new_phrase(self, kind, base=None):
        m = base[0][0] if base else self.rng.choice(self.bars) if self.rng.random() < self.reuse else self.new_bar()
        c = self.rng.choice(self.bars) if self.rng.random() < self.reuse else self.new_bar()
        return [(m, 0), (c, 0), (m, self.rng.choice([0, 0, 1, -1])), (self.new_cad("half" if kind == "ante" else "full"), 0)]

    def render(self, tm, shift=0):
        rh, ds = tm
        out, j = [], 0
        for r in rh:
            if r < 0:
                out.append((REST, -r))
            else:
                d = ds[j] + shift
                out.append((12 * (d // 7) + self.scale[d % 7], r))
                j += 1
        return tuple(out)

def synth_corpus(n, seed=0, share=0.8, modes=("maj",), meters=((4, 4),), forms=("AABB", "AABA", "ABAC", "ABCB", "ABAB"),
                 vary=0.3, lib_bars=24, lib_phr=8, bar_reuse=0.5):
    """→ lista de piezas dict(name, bars, mode, num, den, qpm, tonic_pc, form, family)."""
    rng = random.Random(seed)
    fams = [Family(rng, md, mt, lib_bars, lib_phr, bar_reuse) for md in modes for mt in meters]
    tunes = []
    for i in range(n):
        fam = fams[i % len(fams)]
        form = rng.choice(forms)
        parts = {}
        for L in sorted(set(form)):
            ante = rng.choice(fam.ante_lib) if rng.random() < share else fam.new_phrase("ante")
            cons = rng.choice(fam.cons_lib) if rng.random() < share else fam.new_phrase("cons", base=ante)
            parts[L] = (ante, cons)
        bars, seen = [], set()
        for L in form:
            ante, cons = parts[L]
            a = [fam.render(tm, sh) for tm, sh in ante]
            c = [fam.render(tm, sh) for tm, sh in cons]
            if L in seen and rng.random() < vary:
                c[-1] = fam.render(fam.new_cad("full"))
            bars += a + c
            seen.add(L)
        tpc = rng.randrange(12)
        tunes.append({"name": f"synth_{i:04d}.mid", "bars": bars, "mode": fam.mode, "num": fam.meter[0], "den": fam.meter[1],
                      "qpm": float(rng.choice([96, 104, 112, 120, 132, 144])), "tonic_pc": tpc, "form": form,
                      "family": fams.index(fam)})
    return tunes

def humanize_notes(notes, part, base, rng, jitter=0.04, accomp=True, nbars=0):
    """Simula un MIDI 'real': micro-desajustes, articulación (80-100% de la duración), velocidades variables
    y un acompañamiento grave (raíz + quinta al inicio de cada compás) en la MISMA pista."""
    out = []
    for on, dur, p, v in notes:
        out.append((max(0.0, on + rng.gauss(0, jitter)), dur * rng.uniform(0.8, 1.0), p, rng.randint(55, 105)))
    if accomp:
        bb = 4.0 * part.num / part.den
        for b in range(nbars):
            out.append((b * bb, bb * 0.9, base - 24, 60))
            out.append((b * bb, bb * 0.9, base - 17, 55))
    return out

def write_synth(tunes, outdir, grid=0.25, humanize=0.0, accomp=False, seed=0):
    os.makedirs(outdir, exist_ok=True)
    rng, truth = random.Random(seed), {}
    for t in tunes:
        part = Part(t["mode"], t["num"], t["den"], grid)
        base = base_midi(t["tonic_pc"])
        notes = bars_to_notes(t["bars"], part.U, grid, base)
        if humanize > 0 or accomp:
            notes = humanize_notes(notes, part, base, rng, humanize, accomp, len(t["bars"]))
        key = (key_fifths(t["tonic_pc"], t["mode"]), 0 if t["mode"] == "maj" else 1)
        write_midi(os.path.join(outdir, t["name"]), notes, 480, t["qpm"], (t["num"], t["den"]), key, t["name"])
        truth[t["name"]] = {k: t[k] for k in ("mode", "num", "den", "tonic_pc", "form", "family", "qpm")}
    with open(os.path.join(outdir, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=1)

def analyse_dir(paths, grid=0.25, anchor="final", track=None, legato=0.3, force_mode=None, force_tonic=None, min_rest=2):
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.mid*"), recursive=True)) if os.path.isdir(p) else [p]
    tunes, skipped = [], 0
    for f in files:
        try:
            t = tune_from_midi(f, grid, anchor, track, legato, force_mode, force_tonic, min_rest)
        except (OSError, ValueError, IndexError, struct.error) as e:
            print(yellow(f"  aviso: {f}: {e}"), file=sys.stderr)
            t = None
        if t is None:
            skipped += 1
        else:
            tunes.append(t)
    return tunes, skipped

# ═══════════════════════════════════════════════════════════════════════════
#  SPECS DE SAMPLER  ("nombre:clave=valor,..."), BANCO DE PRUEBAS
# ═══════════════════════════════════════════════════════════════════════════
def _val(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v

def parse_spec(s):
    name, _, rest = s.partition(":")
    kv = {}
    for it in filter(None, rest.split(",")):
        k, _, v = it.partition("=")
        kv[k.strip()] = _val(v.strip())
    return name.strip(), kv

def parse_weights(s):
    w = {}
    for it in filter(None, (s or "").split(",")):
        k, _, v = it.partition("=")
        need(k.strip() in DEF_W, f"peso desconocido {k!r} (usa {', '.join(DEF_W)})")
        w[k.strip()] = float(v)
    return w

def make_sampler(part, spec, pb=4, gcache=None):
    """spec → (sampler, form, vary). Claves: T, form, vary | note: ord, wpos | element/bar: max, inv, minpair, w<peso>."""
    name, kv = parse_spec(spec)
    need(name in ("element", "bar", "note"), f"sampler desconocido: {name!r} (element|bar|note)")
    form, T = str(kv.pop("form", "none")), float(kv.pop("T", 1.0))
    vary = float(kv.pop("vary", 0.3 if form != "none" else 0.0))
    if name == "note":
        s = NoteSampler(part, int(kv.pop("ord", 3)), T, float(kv.pop("wpos", 1.0)))
    else:
        inv, mx, mp = str(kv.pop("inv", "bars" if name == "bar" else "repair")), int(kv.pop("max", 8)), int(kv.pop("minpair", 2))
        al, pm = bool(int(kv.pop("align", 0))), str(kv.pop("pos", "bar"))
        key = (id(part), inv, mx, mp, pb, al, pm)
        if gcache is not None and key in gcache:
            G = gcache[key]
        else:
            G = Grammar(part, pb, mp, 600, mx, inv, al, pm)
            if gcache is not None:
                gcache[key] = G
        w = {k[1:]: float(kv.pop(k)) for k in [k for k in kv if k.startswith("w")]}
        for k in w:
            need(k in DEF_W, f"peso desconocido en {spec!r}: w{k}")
        s = (BarSampler if name == "bar" else ElementSampler)(G, T, w)
    need(not kv, f"opciones desconocidas en {spec!r}: {', '.join(kv)}")
    return s, form, vary

def make_judge(path):
    """score_piece() de abc_predictor_v2.py como evaluador externo e independiente de estructura."""
    sp = importlib.util.spec_from_file_location("abc_pred_judge", path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    def f(bars, part):
        try:
            return float(mod.score_piece(bars_to_abc(bars, part, 0))["total"])
        except (SystemExit, Exception):
            return float("nan")
    f.mod = mod
    return f

def learn_quality(part, G, pb):
    """¿Las reglas cubren las frases realmente compartidas? Ventanas alineadas de pb compases que aparecen ≥2 veces."""
    wc = Counter()
    for t in part.tunes:
        s = t["bars"]
        for i in range(0, len(s) - pb + 1, pb):
            wc[tuple(s[i:i + pb])] += 1
    exp = expand_rules(G.rules, G.V)
    hit = tot_shared = tot_cov = 0
    for t, par in zip(part.tunes, G.parsed):
        s = t["bars"]
        shared, cov = np.zeros(len(s), bool), np.zeros(len(s), bool)
        for i in range(0, len(s) - pb + 1, pb):
            if wc[tuple(s[i:i + pb])] >= 2:
                shared[i:i + pb] = True
        i = 0
        for x in par:
            n = len(exp[x]) if x >= G.V else 1
            if n >= 2:
                cov[i:i + n] = True
            i += n
        hit += int((shared & cov).sum())
        tot_shared += int(shared.sum())
        tot_cov += int(cov.sum())
    return {"recall": hit / max(tot_shared, 1), "precision": hit / max(tot_cov, 1), "shared": tot_shared,
            "rules": len(G.rules), "elems": G.E, "vocab": G.V, "tunes": len(part.tunes)}

def eval_condition(corpus_kw, specs, pieces=30, reps=2, seed=100, nbars=32, sec_bars=8, pb=4, judge=None,
                   humanize=0.0, accomp=False, grid=0.25):
    acc = {s: [] for s in specs}
    refrows, lq, keyerr, ntun, barok, nbtot = defaultdict(list), [], 0, 0, 0, 0
    for r in range(reps):
        tunes = synth_corpus(seed=seed + r, **corpus_kw)
        with tempfile.TemporaryDirectory() as d:
            write_synth(tunes, d, grid, humanize, accomp, seed + r)
            an, _ = analyse_dir([d], grid)
        truth = {t["name"]: t for t in tunes}
        for a_ in an:
            t = truth[a_["name"]]
            keyerr += not (a_["mode"] == t["mode"] and a_["tonic_pc"] == t["tonic_pc"])
            ntun += 1
            if len(a_["bars"]) == len(t["bars"]):
                barok += sum(x == y for x, y in zip(a_["bars"], t["bars"]))
            nbtot += len(t["bars"])
        part = pick_part(build_parts(an, grid))
        idx = list(range(len(part.tunes)))
        ntest = max(2, len(idx) // 5)
        train, tests = part.subset(idx[:-ntest]), [part.tune_bars(i) for i in idx[-ntest:]]
        ref = Ref(train, pb)
        rs = ref_stats(tests, ref)
        gcache = {}
        for spec in specs:
            sampler, form, vary = make_sampler(train, spec, pb, gcache)
            for k in range(pieces):
                rng = np.random.default_rng(seed * 1000 + r * 100 + k)
                bars = compose(sampler, nbars, rng, sec_bars, form, vary)
                m = piece_metrics(bars, ref)
                m["dist"] = dist_to_ref(m, rs)
                if judge:
                    m["abc"] = judge(bars, train)
                acc[spec].append(m)
        rr = random.Random(seed + r)
        for t in tests:
            m = piece_metrics(t, ref)
            m["dist"] = dist_to_ref(m, rs)
            if judge:
                m["abc"] = judge(t, train)
            refrows["real (test)"].append(m)
        for k in range(pieces):
            for nm, bars in (("compases al azar", [rr.choice(train.vocab) for _ in range(nbars)]),
                             ("pieza del corpus", train.tune_bars(rr.randrange(len(train.tunes))))):
                m = piece_metrics(bars, ref)
                m["dist"] = dist_to_ref(m, rs)
                if judge:
                    m["abc"] = judge(bars, train)
                refrows[nm].append(m)
        lq.append(learn_quality(train, Grammar(train, pb, 2, 600, 8, "repair", True), pb))
    info = {"key_errors": keyerr, "tunes": ntun, "bar_fidelity": barok / max(nbtot, 1),
            "learn": {k: float(np.mean([q[k] for q in lq])) for k in lq[0]}}
    return acc, dict(refrows), info

TABLE_COLS = ["copy", "S4", "S8", "per", "phr", "nov8", "nov4b", "nov8b", "step", "junc", "cad", "bpn"]

def summarize(rows):
    out = {}
    for name, ms in rows.items():
        mean = {k: float(np.mean([m[k] for m in ms])) for k in METRIC_KEYS + ["dist"]}
        dd = [m["dist"] for m in ms]
        mean["se"] = float(np.std(dd) / math.sqrt(len(dd)))
        ab = [m["abc"] for m in ms if "abc" in m and not math.isnan(m["abc"])]
        mean["abc"] = float(np.mean(ab)) if ab else float("nan")
        out[name] = mean
    return out

def print_table(title, summ, refs=None, cols=TABLE_COLS, with_abc=False):
    print(bold(f"\n{title}"))
    hdr = f"{'sampler':30s}" + "".join(f"{c:>6s}" for c in cols) + f"{'dist':>7s}{'±':>6s}" + (f"{'abc':>6s}" if with_abc else "")
    print(dim(hdr))
    def line(name, m, ref=False):
        s = f"{name[:30]:30s}" + "".join(f"{m[c]:6.2f}" for c in cols) + f"{m['dist']:7.2f}{m['se']:6.2f}"
        if with_abc:
            s += f"{m['abc']:6.2f}"
        print(dim(s) if ref else s)
    for k, m in (refs or {}).items():
        line(k, m, True)
    best = min(m["dist"] for m in summ.values()) if summ else 0
    for k, m in summ.items():
        s = f"{k[:30]:30s}" + "".join(f"{m[c]:6.2f}" for c in cols) + f"{m['dist']:7.2f}{m['se']:6.2f}" + (f"{m['abc']:6.2f}" if with_abc else "")
        print(green(s) if m["dist"] == best else s)

MAIN_SPECS = ["note:ord=1", "note:ord=2", "note:ord=3", "note:ord=4", "bar", "element:max=2", "element:max=4", "element:max=8",
              "element:max=4,align=1", "note:ord=3,form=AABB", "bar:form=AABB", "element:max=8,form=AABB", "element:max=4,align=1,form=AABB"]

def run_suite(name, a, judge):
    kw0 = dict(n=a.tunes, share=a.share)
    common = dict(pieces=a.pieces, reps=a.reps, seed=a.seed, nbars=a.bars, judge=judge, sec_bars=a.section_bars, pb=a.phrase_bars)
    results = {}
    def one(title, corpus_kw, specs, **extra):
        acc, refs, info = eval_condition(corpus_kw, specs, **{**common, **extra})
        summ, rsum = summarize(acc), summarize(refs)
        print_table(title, summ, rsum, with_abc=bool(judge))
        L = info["learn"]
        print(dim(f"  corpus: {info['tunes']} piezas · tonalidad mal detectada: {info['key_errors']} · compases idénticos tras MIDI→análisis: "
                  f"{info['bar_fidelity']:.1%} · Re-Pair: {L['rules']:.0f} reglas, {L['elems']:.0f} elementos, "
                  f"recall de frases compartidas {L['recall']:.2f}, precisión {L['precision']:.2f}"))
        results[title] = {"samplers": summ, "refs": rsum, "info": info}
    if name in ("main", "all"):
        one(f"[main] share={a.share} n={a.tunes} (4/4 mayor)", kw0, MAIN_SPECS)
    if name in ("sharing", "all"):
        for sh in (0.0, 0.3, 0.6, 0.9):
            one(f"[sharing] share={sh}", {**kw0, "share": sh},
                ["note:ord=3", "bar", "element:max=8", "note:ord=3,form=AABB", "element:max=8,form=AABB"])
    if name in ("size", "all"):
        for n in (12, 30, 60, 120, 300):
            one(f"[size] n={n} tunes", {**kw0, "n": n}, ["note:ord=3", "bar", "element:max=8", "element:max=8,inv=ngram"])
    if name in ("meter", "all"):
        for md, mt in (("maj", (4, 4)), ("min", (4, 4)), ("maj", (3, 4)), ("maj", (6, 8)), ("min", (6, 8))):
            one(f"[meter] {md} {mt[0]}/{mt[1]}", {**kw0, "modes": (md,), "meters": (mt,)}, ["note:ord=3", "bar", "element:max=8"])
    if name in ("ablate", "all"):
        base = "element:max=8"
        specs = [base] + [f"{base},w{k}=0" for k in ("succ", "bound", "compat", "pos", "occ", "fin")] + \
                [f"{base},wlen=0.5", f"{base},wlen=1.0", f"{base},inv=ngram", f"{base},inv=bars", f"{base},minpair=3", "note:ord=3,wpos=0"]
        one(f"[ablate] componentes de la puntuación (share={a.share})", kw0, specs)
    if name in ("temp", "all"):
        one(f"[temp] temperatura", kw0, [f"note:ord=3,T={t}" for t in (0.4, 0.7, 1.0, 1.5)] + [f"element:max=8,T={t}" for t in (0.4, 0.7, 1.0, 1.5)])
    if name in ("form", "all"):
        specs = []
        for f in ("none", "AABB", "AABA", "ABAC"):
            specs += [f"note:ord=3,form={f}", f"element:max=8,form={f}"]
        specs += ["note:ord=3,form=AABA,vary=0", "element:max=8,form=AABA,vary=0", "note:ord=3,form=AABA,vary=0.8", "element:max=8,form=AABA,vary=0.8"]
        one("[form] forma por secciones (copia con final alternativo)", kw0, specs)
    if name in ("ambig", "all"):
        one("[ambig] pocos compases muy reutilizados en muchas frases (continuación ambigua: donde las reglas deberían ayudar)",
            {**kw0, "share": 0.9, "lib_bars": 6, "lib_phr": 40, "bar_reuse": 0.95},
            ["note:ord=3", "bar", "element:max=2", "element:max=4", "element:max=8", "element:max=8,inv=ngram",
             "element:max=4,align=1", "element:max=8,align=1", "element:max=8,inv=ngram,align=1",
             "element:max=4,pos=elem", "element:max=8,pos=elem"])
    if name in ("novelty", "all"):
        one(f"[novelty] curva novedad ↔ estructura al aflojar el peso del sucesor (share={a.share})", kw0,
            ["note:ord=3"] + [f"bar:wsucc={x}" for x in (0.0, 0.25, 0.5, 1.0)] + [f"element:max=4,align=1,wsucc={x}" for x in (0.0, 0.25, 0.5, 0.75, 1.0)]
            + ["element:max=4,align=1,wsucc=0.5,T=1.3"])
    if name in ("robust", "all"):
        one("[robust] MIDI 'real': jitter σ=0.04, articulación 80-100%, acompañamiento grave en la misma pista", kw0,
            ["note:ord=3", "bar", "element:max=8", "element:max=8,form=AABB"], humanize=0.04, accomp=True)
    return results

def cmd_bench(a):
    judge = make_judge(a.judge) if a.judge else None
    need(a.suite in ("main", "sharing", "size", "meter", "ablate", "temp", "form", "ambig", "novelty", "robust", "all"), "suite desconocida")
    t0 = time.time()
    res = run_suite(a.suite, a, judge)
    print(dim(f"\n(dist = distancia media a las piezas reales de test en desviaciones típicas, menor = mejor; "
              f"{a.pieces} piezas × {a.reps} corpus por fila; {time.time() - t0:.0f}s)"))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)

# ═══════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ═══════════════════════════════════════════════════════════════════════════
def _meters(s):
    out = []
    for m in s.split(","):
        n, _, d = m.partition("/")
        out.append((int(n), int(d)))
    return tuple(out)

def cmd_synth(a):
    tunes = synth_corpus(a.n, a.seed, a.share, tuple(a.modes.split(",")), _meters(a.meters), tuple(a.forms.split(",")),
                         a.vary, a.lib_bars, a.lib_phrases, a.bar_reuse)
    write_synth(tunes, a.out, a.grid, a.humanize, a.accomp, a.seed)
    c = Counter((t["mode"], f"{t['num']}/{t['den']}") for t in tunes)
    print(green(f"{len(tunes)} MIDIs en {a.out}/") + dim("  (+ truth.json)  ") + ", ".join(f"{m} {t}: {n}" for (m, t), n in sorted(c.items())))

def cmd_learn(a):
    tunes, skipped = analyse_dir(a.inputs, a.grid, a.anchor, a.track, a.legato, None, None, a.min_rest)
    need(tunes, "ningún MIDI utilizable")
    parts = build_parts(tunes, a.grid)
    save_model(a.out, parts, a.grid)
    split = sum(t["split"] for t in tunes)
    print(green(f"{len(tunes)} piezas → {a.out}") + dim(f"   (omitidas: {skipped}; notas partidas por barra: {split})"))
    for lab, p in sorted(parts.items()):
        nb = sum(len(t["bars"]) for t in p.tunes)
        print(f"  {lab:9s} {len(p.tunes):4d} piezas  {nb:6d} compases  {len(p.vocab):6d} distintos  "
              f"({1 - len(p.vocab) / max(nb, 1):.0%} compases repetidos)")

def cmd_info(a):
    parts, _ = load_model(a.model)
    p = pick_part(parts, a.part)
    print(bold(f"partición {p.label}") + dim(f"  (disponibles: {', '.join(sorted(parts))})"))
    print(Grammar(p, a.phrase_bars, a.min_pair, a.max_rules, a.max_elem, a.inventory, a.align, a.pos_mode).describe(a.top))

def _build_sampler(a, part):
    if a.sampler == "note":
        return NoteSampler(part, a.order, a.temp)
    G = Grammar(part, a.phrase_bars, a.min_pair, a.max_rules, a.max_elem, "bars" if a.sampler == "bar" else a.inventory, a.align, a.pos_mode)
    return (BarSampler if a.sampler == "bar" else ElementSampler)(G, a.temp, parse_weights(a.w))

def cmd_generate(a):
    parts, _ = load_model(a.model)
    part = pick_part(parts, a.part)
    sampler = _build_sampler(a, part)
    ref = Ref(part, a.phrase_bars)
    tonic = parse_tonic(a.tonic)
    qpm = a.tempo or float(np.median([t["qpm"] for t in part.tunes]))
    root, ext = os.path.splitext(a.out)
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        if hasattr(sampler, "trace"):
            sampler.trace.clear()
        bars = compose(sampler, a.bars, rng, a.section_bars, a.form, a.vary)
        path = a.out if a.n == 1 else f"{root}_{i:02d}{ext}"
        write_piece_midi(path, bars, part, tonic, qpm, title=os.path.basename(path))
        if a.abc:
            ar, ae = os.path.splitext(a.abc)
            with open(a.abc if a.n == 1 else f"{ar}_{i:02d}{ae}", "w") as fh:
                fh.write(bars_to_abc(bars, part, tonic, qpm, os.path.basename(path), a.phrase_bars))
        m = piece_metrics(bars, ref)
        tr = f"  elementos: {'+'.join(map(str, sampler.trace))}" if getattr(sampler, "trace", None) else ""
        print(green(path) + dim(f"  {len(bars)} compases ({part.label}, tónica {a.tonic})  copy {m['copy']:.2f}  S4 {m['S4']:.2f}  "
                                f"phr {m['phr']:.2f}  nov8 {m['nov8']:.2f}  ends {int(m['ends'])}{tr}"))

def cmd_eval(a):
    parts, grid = load_model(a.model)
    fm = None
    if a.part:
        need(a.part in parts, f"partición {a.part!r} inexistente (hay: {', '.join(parts)})")
        fm = parts[a.part].mode
    tunes, skipped = analyse_dir(a.inputs, grid, "final", None, a.legato, fm, parse_tonic(a.tonic) if a.tonic else None)
    need(tunes, "ningún MIDI utilizable")
    refs, rows = {}, []
    for t in tunes:
        lab = a.part or f"{t['mode']}-{t['num']}/{t['den']}"
        p = parts.get(lab) or pick_part(parts)
        if lab not in parts:
            print(yellow(f"  {t['name']}: partición {lab} no está en el modelo; se usa {p.label}"))
        if p.label not in refs:
            refs[p.label] = Ref(p, a.phrase_bars)
        rows.append((t["name"], piece_metrics(t["bars"], refs[p.label])))
    cols = TABLE_COLS + ["ends"]
    print(dim(f"{'pieza':30s}" + "".join(f"{c:>6s}" for c in cols)))
    for n, m in rows:
        print(f"{n[:30]:30s}" + "".join(f"{m[c]:6.2f}" for c in cols))
    if len(rows) > 1:
        print(dim(f"{'media':30s}" + "".join(f"{np.mean([m[c] for _, m in rows]):6.2f}" for c in cols)))
    print(dim("nov8 = fracción de notas NO cubiertas por 8-gramas del corpus (1 = totalmente nuevo; 0 = copia)."))

def cmd_selftest(a):
    res = []
    def check(name, ok, detail=""):
        res.append(bool(ok))
        print(("  " + green("PASS") if ok else "  " + red("FAIL")) + f"  {name}" + (dim(f"  [{detail}]") if detail else ""))
    judge = None
    for cand in ([a.judge] if a.judge else []) + ["abc_predictor_v2.py", os.path.join(os.path.dirname(os.path.abspath(__file__)), "abc_predictor_v2.py")]:
        if cand and os.path.exists(cand):
            judge = make_judge(cand)
            break
    print(bold("selftest") + dim(f"  (abc_predictor_v2: {'sí' if judge else 'no encontrado → pruebas ABC omitidas'})"))
    tmp = tempfile.mkdtemp()
    # 1 MIDI ida y vuelta
    notes = [(0.0, 1.0, 60, 90), (1.0, 0.5, 62, 80), (1.5, 2.5, 64, 70), (4.0, 1.0, 67, 100)]
    write_midi(os.path.join(tmp, "a.mid"), notes, 480, 133.0, (6, 8), (2, 1), "t")
    m = read_midi(os.path.join(tmp, "a.mid"))
    check("MIDI: escritura/lectura exacta", [(round(o, 4), round(d, 4), p, v) for o, d, p, v in m["notes"]] == [(round(o, 4), round(d, 4), p, v) for o, d, p, v in notes]
          and m["ts"] == (6, 8) and abs(m["qpm"] - 133.0) < 0.01 and m["key"] == (2, 1))
    # 2 skyline + legato
    line = melody_line([(0, 1, 72, 80), (0, 1, 48, 80), (1, 0.8, 74, 80), (2, 1, 76, 80), (3.5, 0.5, 77, 80)], 0.25)
    check("skyline: elige la voz aguda y rellena articulación (hueco 0.2) pero no silencios (0.5)",
          line == [(0, 4, 72), (4, 8, 74), (8, 12, 76), (14, 16, 77)], str(line))
    # 3 fidelidad MIDI→compases, limpia y humanizada
    tunes = synth_corpus(120, seed=11, share=0.8, modes=("maj", "min"), meters=((4, 4), (3, 4), (6, 8)))
    for hum, acc, thr_k, thr_b in ((0.0, False, 1.0, 0.999), (0.04, True, 0.98, 0.90)):
        d = os.path.join(tmp, f"s{hum}")
        write_synth(tunes, d, 0.25, hum, acc, 1)
        an, _ = analyse_dir([d])
        kok = np.mean([x["mode"] == t["mode"] and x["tonic_pc"] == t["tonic_pc"] for x, t in zip(an, tunes)])
        bok = np.mean([x == y for x, t in zip(an, tunes) for x, y in zip(x["bars"], t["bars"])]) if all(len(x["bars"]) == len(t["bars"]) for x, t in zip(an, tunes)) else 0
        check(f"análisis MIDI→compases (humanize={hum}, acomp={acc}): tonalidad ≥{thr_k:.0%}, compases idénticos ≥{thr_b:.1%}",
              kok >= thr_k and bok >= thr_b, f"tonalidad {kok:.1%}, compases {bok:.1%}")
    # 4 Re-Pair
    seqs = [[0, 1, 0, 1, 2], [0, 1, 2, 0, 1, 2], [3, 3, 3, 3, 3], [], [4]]
    rules, parsed = repair(seqs, 5)
    exp = expand_rules(rules, 5)
    back = [[b for x in row for b in ([x] if x < 5 else exp[x])] for row in parsed]
    check("Re-Pair: la expansión reproduce las secuencias (incl. vacías y aaaaa)", back == seqs and len(parsed) == len(seqs), f"{len(rules)} reglas")
    check("Re-Pair: determinista", repair(seqs, 5) == (rules, parsed))
    p0 = Part("maj", 4, 4, 0.25)
    for t in tunes[:40]:
        if (t["mode"], t["num"], t["den"]) == ("maj", 4, 4):
            p0.add(t["name"], t["bars"])
    G = Grammar(p0, 4, 2, 600, 8, "repair")
    ex = expand_rules(G.rules, G.V)
    check("Re-Pair sobre corpus: parseo sin pérdida y sin cruzar piezas",
          all([b for x in par for b in ([x] if x < G.V else ex[x])] == t["bars"] for par, t in zip(G.parsed, p0.tunes)), G.describe().splitlines()[1])
    # 5 generación: validez, final, semilla, longitudes raras
    part = pick_part(build_parts(analyse_dir([os.path.join(tmp, "s0.0")])[0], 0.25))
    ok_valid = ok_end = ok_len = True
    specs = ["note:ord=3", "note:ord=0", "bar", "element:max=8", "element:max=8,inv=ngram", "element:max=3,T=0.01", "element:max=8,T=5",
             "element:max=8,form=AABA", "note:ord=3,form=ABAC,vary=1"]
    for sp in specs:
        s, form, vary = make_sampler(part, sp)
        for nb in (1, 3, 5, 16, 32, 33):
            bars = compose(s, nb, np.random.default_rng(nb), 8, form, vary)
            ok_len &= len(bars) == nb
            ok_valid &= all(sum(u for _, u in b) == part.U for b in bars)
            pit = [r for b in bars for r, _ in b if r != REST]
            ok_end &= bool(pit) and pit[-1] % 12 == 0
    check("generación: nº de compases exacto (1,3,5,16,32,33) para todos los samplers/formas", ok_len)
    check("generación: todos los compases suman exactamente la duración del compás", ok_valid)
    check("generación: la última nota es la tónica (clase de altura 0)", ok_end)
    s, form, vary = make_sampler(part, "element:max=8,form=AABB")
    b1 = compose(s, 32, np.random.default_rng(5), 8, form, vary)
    b2 = compose(s, 32, np.random.default_rng(5), 8, form, vary)
    b3 = compose(s, 32, np.random.default_rng(6), 8, form, vary)
    check("generación: misma semilla → misma pieza; otra semilla → otra pieza", b1 == b2 and b1 != b3)
    check("forma AABB: la sección 2 copia la 1 (salvo el último compás) y la 4 la 3", b1[8:15] == b1[0:7] and b1[24:31] == b1[16:23])
    # 6 MIDI de salida
    pth = os.path.join(tmp, "o.mid")
    write_piece_midi(pth, b1, part, 5, 120.0)
    back = read_midi(pth)["notes"]
    exp_notes = bars_to_notes(b1, part.U, part.grid, base_midi(5))
    check("MIDI generado: las notas leídas coinciden con las muestreadas", len(back) == len(exp_notes) and
          all(abs(x[0] - y[0]) < 1e-3 and abs(x[1] - y[1]) < 1e-3 and x[2] == y[2] for x, y in zip(back, exp_notes)))
    rt_ok = rt_n = 0
    for sp in ("note:ord=3", "element:max=8", "bar"):
        s, form, vary = make_sampler(part, sp)
        for sd in range(6):
            bars = compose(s, 32, np.random.default_rng(sd), 8, form, vary)
            q = os.path.join(tmp, "rt.mid")
            write_piece_midi(q, bars, part, 7, 120.0)
            back = tune_from_midi(q, 0.25, "final", None, 0.3, part.mode, 7)
            rt_n += 1
            rt_ok += back is not None and list(back["bars"]) == list(bars)
    check("ida y vuelta: generar → MIDI → re-analizar (modo/tónica conocidos) reproduce los mismos compases", rt_ok == rt_n, f"{rt_ok}/{rt_n}")
    # 7 ABC (spelling de 24 tonalidades con alteraciones al azar)
    if judge:
        rr, okabc = random.Random(1), True
        for mode in ("maj", "min"):
            pp = Part(mode, 4, 4, 0.25)
            for tpc in range(12):
                bars = [tuple((rr.randint(-14, 16), 2) for _ in range(8)) for _ in range(30)]
                pc = judge.mod.parse_abc(bars_to_abc(bars, pp, tpc))
                got = [n[2] for n in pc["notes"]]
                want = [base_midi(tpc) + r for b in bars for r, _ in b]
                if got != want:
                    okabc = False
                    print(red(f"    ABC distinto en {mode} tónica {tpc}: {sum(g != w for g, w in zip(got, want))} notas"))
        check("ABC: 24 tonalidades × 30 compases con alteraciones aleatorias se parsean a las mismas alturas", okabc)
        abc = bars_to_abc(b1, part, 7)
        pc = judge.mod.parse_abc(abc)
        check("ABC de una pieza generada: duración total = nº de compases × duración de compás",
              abs(pc["total"] - 32 * pc["bar_beats"]) < 1e-6, f"{pc['total']} beats")
    # 8 corpus degenerados y mixtos
    for nt in (1, 2, 3):
        ps = build_parts(analyse_dir([write_one(tmp, tunes[:nt], f"d{nt}")])[0], 0.25)
        pp = pick_part(ps)
        ok = True
        for sp in ("note:ord=3", "bar", "element:max=8"):
            s, f, v = make_sampler(pp, sp)
            bb = compose(s, 16, np.random.default_rng(0), 8, f, v)
            ok &= len(bb) == 16 and all(sum(u for _, u in b) == pp.U for b in bb)
        check(f"corpus mínimo de {nt} pieza(s): todos los samplers generan piezas válidas", ok)
    nr = synth_corpus(30, seed=2, share=0.0, vary=0.0, forms=("ABCD",))
    pn = Part("maj", 4, 4, 0.25)
    for t in nr:
        if (t["mode"], t["num"]) == ("maj", 4):
            pn.add(t["name"], t["bars"])
    Gn = Grammar(pn, 4, 2, 600, 8, "repair")
    check("corpus sin repeticiones entre piezas: las reglas son internas a cada pieza y no hay NaN en la puntuación",
          not np.isnan(Gn.pos_lp).any() and not np.isnan(Gn.fin_lp).any(), f"{len(Gn.rules)} reglas")
    mixed = build_parts(analyse_dir([os.path.join(tmp, "s0.0")])[0], 0.25)
    check("corpus mixto: una partición por (modo, compás) y cada una genera", len(mixed) == 6 and all(
        len(compose(make_sampler(p, "element:max=8")[0], 8, np.random.default_rng(1))) == 8 for p in mixed.values()), ", ".join(sorted(mixed)))
    mp = os.path.join(tmp, "m.json")
    save_model(mp, mixed, 0.25)
    back_parts, g = load_model(mp)
    check("modelo .json: guardar/cargar conserva vocabulario y piezas", g == 0.25 and all(
        back_parts[k].vocab == mixed[k].vocab and back_parts[k].tunes == mixed[k].tunes for k in mixed))
    print(bold(f"\n{sum(res)}/{len(res)} pruebas superadas") if all(res) else red(f"\n{len(res) - sum(res)} FALLO(S)"))
    sys.exit(0 if all(res) else 1)

def write_one(tmp, tunes, name):
    d = os.path.join(tmp, name)
    write_synth(tunes, d)
    return d

def main():
    ap = argparse.ArgumentParser(description="Generación musical por muestreo de elementos de gramática (ver cabecera del fichero).")
    sp = ap.add_subparsers(dest="cmd", required=True)
    def common_g(p):
        p.add_argument("--phrase-bars", type=int, default=4, help="compases por frase (posición métrica de los elementos)")
        p.add_argument("--section-bars", type=int, default=8, help="compases por sección para --form")
        p.add_argument("--max-elem", type=int, default=8, help="máx. compases de un elemento")
        p.add_argument("--min-pair", type=int, default=2, help="mín. repeticiones para crear una regla")
        p.add_argument("--max-rules", type=int, default=600)
        p.add_argument("--inventory", choices=("repair", "ngram", "bars"), default="repair")
        p.add_argument("--align", action="store_true", help="Re-Pair por frases: ninguna regla cruza una frontera de frase")
        p.add_argument("--pos-mode", choices=("bar", "elem"), default="bar", help="estadística de posición: compuesta por compás (bar) o por elemento (elem)")
    p = sp.add_parser("synth", help="corpus MIDI sintético")
    p.add_argument("-o", "--out", required=True); p.add_argument("-n", type=int, default=80)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--share", type=float, default=0.8)
    p.add_argument("--modes", default="maj"); p.add_argument("--meters", default="4/4")
    p.add_argument("--forms", default="AABB,AABA,ABAC,ABCB,ABAB"); p.add_argument("--vary", type=float, default=0.3)
    p.add_argument("--lib-bars", type=int, default=24); p.add_argument("--lib-phrases", type=int, default=8)
    p.add_argument("--bar-reuse", type=float, default=0.5, help="prob. de reutilizar compases de la biblioteca dentro de las frases")
    p.add_argument("--humanize", type=float, default=0.0, help="σ del jitter en pulsos (+articulación 80-100%%)")
    p.add_argument("--accomp", action="store_true", help="añade acompañamiento grave en la misma pista")
    p.add_argument("--grid", type=float, default=0.25)
    p.set_defaults(f=cmd_synth)
    p = sp.add_parser("learn", help="MIDIs → modelo")
    p.add_argument("inputs", nargs="+"); p.add_argument("-o", "--out", required=True)
    p.add_argument("--grid", type=float, default=0.25, help="rejilla en pulsos (0.25 = semicorchea)")
    p.add_argument("--anchor", choices=("final", "median"), default="final", help="referencia de octava: nota final o mediana")
    p.add_argument("--track", type=int, default=None); p.add_argument("--legato", type=float, default=0.3,
                   help="huecos de articulación (fracción de la duración) que se rellenan")
    p.add_argument("--min-rest", type=int, default=2, help="silencios más cortos que N unidades de rejilla se absorben en la nota previa")
    p.set_defaults(f=cmd_learn)
    p = sp.add_parser("info", help="gramática del modelo")
    p.add_argument("model"); p.add_argument("--part"); p.add_argument("--top", type=int, default=8); common_g(p)
    p.set_defaults(f=cmd_info)
    p = sp.add_parser("generate", help="muestrear piezas")
    p.add_argument("model"); p.add_argument("-o", "--out", required=True); p.add_argument("--abc")
    p.add_argument("--sampler", choices=("element", "bar", "note"), default="element")
    p.add_argument("-n", type=int, default=1); p.add_argument("--bars", type=int, default=32)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--order", type=int, default=3, help="orden del n-gram (sampler note)")
    p.add_argument("--w", default="", help="pesos: succ,bound,compat,pos,occ,len,fin  (ej. succ=0.5,len=0.5)")
    p.add_argument("--form", default="none", help="forma de secciones, ej. AABA (cada letra repetida copia la primera)")
    p.add_argument("--vary", type=float, default=0.3, help="prob. de final alternativo en copias de sección")
    p.add_argument("--tonic", default="C"); p.add_argument("--tempo", type=float, default=0.0); p.add_argument("--part")
    common_g(p); p.set_defaults(f=cmd_generate)
    p = sp.add_parser("eval", help="métricas de MIDIs frente al corpus del modelo")
    p.add_argument("model"); p.add_argument("inputs", nargs="+"); p.add_argument("--phrase-bars", type=int, default=4)
    p.add_argument("--legato", type=float, default=0.3)
    p.add_argument("--part", help="fuerza la partición (modo-compás), p. ej. maj-4/4; sin ella la tonalidad se detecta sola y puede fallar en piezas poco tonales")
    p.add_argument("--tonic", help="con --part: tónica de las piezas (por defecto, su nota final)")
    p.set_defaults(f=cmd_eval)
    p = sp.add_parser("bench", help="comparativa de samplers sobre corpus sintético")
    p.add_argument("--suite", default="main", help="main|sharing|size|meter|ablate|temp|form|ambig|novelty|robust|all")
    p.add_argument("--pieces", type=int, default=30); p.add_argument("--reps", type=int, default=2)
    p.add_argument("--tunes", type=int, default=120); p.add_argument("--share", type=float, default=0.8)
    p.add_argument("--bars", type=int, default=32); p.add_argument("--seed", type=int, default=100)
    p.add_argument("--section-bars", type=int, default=8); p.add_argument("--phrase-bars", type=int, default=4)
    p.add_argument("--judge", help="ruta a abc_predictor_v2.py: añade la columna abc (score_piece)")
    p.add_argument("--json"); p.set_defaults(f=cmd_bench)
    p = sp.add_parser("selftest", help="pruebas de integridad")
    p.add_argument("--judge"); p.set_defaults(f=cmd_selftest)
    a = ap.parse_args()
    a.f(a)

if __name__ == "__main__":
    main()
