#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        FORM ANALYZER  v1.0                                   ║
║          Forma musical de un MIDI: secciones, repeticiones y etiquetas       ║
║                                                                              ║
║  harvester segmenta por novedad y section_score puntúa coherencia, pero      ║
║  ninguna herramienta dice QUÉ FORMA tiene una pieza (A A' B A'), qué         ║
║  compases son repetición literal o variada de otros, ni permite comparar     ║
║  la forma de una reconstrucción con la del original. Esta herramienta lo     ║
║  hace con una matriz de autosimilitud por compás.                            ║
║                                                                              ║
║  MÉTODO:                                                                     ║
║  [1] COMPASES  — divide el MIDI en compases reales (respeta cambios de       ║
║                  compás; 6/8 y 12/8 se cuentan bien)                         ║
║  [2] RASGOS    — por compás: clases de altura, bajo, rejilla rítmica y       ║
║                  contorno de la voz superior                                 ║
║  [3] SIMILITUD — matriz compás×compás (0-1)                                  ║
║  [4] REPETICIONES — diagonales de alta similitud = material que vuelve       ║
║  [5] FRONTERAS — curva de novedad (kernel de Foote) + bordes de repetición   ║
║  [6] ETIQUETAS — A, A', B ...: misma letra = misma sección; prima = variada  ║
║  [7] FORMA     — clasifica (binaria, ternaria, AABA, rondó, estrófica ...)   ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    detect    Analiza un MIDI e imprime/exporta la forma                      ║
║    plan      Convierte la forma en una estructura reutilizable por otras     ║
║              herramientas (JSON, o comando de song_architect)                ║
║    compare   Compara la forma de dos MIDIs (fidelidad estructural 0-1)       ║
║    selftest  Genera MIDIs sintéticos con forma conocida y verifica           ║
║                                                                              ║
║  USO:                                                                        ║
║    python form_analyzer.py detect obra.mid                                   ║
║    python form_analyzer.py detect obra.mid --json forma.json --ssm           ║
║    python form_analyzer.py detect obra.mid --min-bars 8 --sens 0.7           ║
║    python form_analyzer.py detect obra.mid --timesig 6/8 --track 1 2         ║
║    python form_analyzer.py plan obra.mid --target song_architect             ║
║    python form_analyzer.py plan obra.mid --target json --output plan.json    ║
║    python form_analyzer.py compare original.mid reconstruccion.mid           ║
║    python form_analyzer.py selftest                                          ║
║                                                                              ║
║  OPCIONES DE ANÁLISIS (detect, plan, compare):                               ║
║    --min-bars N      Longitud mínima de sección (default: según longitud, 2-8) ║
║    --sens X          Sensibilidad de fronteras 0-1 (default: 0.5; mayor =    ║
║                      más secciones)                                          ║
║    --same-thr X     Similitud para "misma sección" (default: 0.86)           ║
║    --var-thr X      Similitud para "sección variada" A' (default: 0.70)      ║
║    --snap N          Ajusta fronteras a múltiplos de N compases (default: 0) ║
║    --timesig N/D     Fuerza el compás (si el MIDI lo declara mal)            ║
║    --track N [N..]   Usa solo estas pistas (default: todas salvo percusión)  ║
║                                                                              ║
║  OPCIONES DE SALIDA:                                                         ║
║    --json FILE       (detect) Escribe el análisis completo ('-' = stdout)    ║
║    --ssm             (detect) Dibuja la matriz de autosimilitud en ASCII     ║
║    --output FILE     (plan) Escribe la estructura                            ║
║    --target T        (plan) json | song_architect                            ║
║                                                                              ║
║  LIMITACIONES: heurístico. Funciona mejor con MIDIs cuantizados y de forma   ║
║  clara; con material homogéneo (preludios de arpegios) devuelve pocas        ║
║  secciones. Las etiquetas son relativas (A es la primera sección, no "el     ║
║  tema"). Un MIDI con compás mal declarado da compases erróneos: usar         ║
║  --timesig. No distingue repetición escrita de repetición con signos.        ║
║                                                                              ║
║  DEPENDENCIAS: ninguna (solo stdlib)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import re
import json
import math
import argparse
import tempfile
from bisect import bisect_right
from pathlib import Path
from collections import defaultdict

VERSION = "1.0"
_USE_COLOR = sys.stdout.isatty()


def _c(code):
    if not _USE_COLOR:
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "CYA": "\033[96m",
            "YEL": "\033[93m", "RED": "\033[91m", "GRN": "\033[92m"}.get(code, "")


# ══════════════════════════════════════════════════════════════════════════════
#  Lectura de MIDI (autocontenida)
# ══════════════════════════════════════════════════════════════════════════════

class Note:
    __slots__ = ("start", "end", "pitch", "vel", "ch", "trk")

    def __init__(self, start, end, pitch, vel, ch, trk):
        self.start, self.end, self.pitch, self.vel, self.ch, self.trk = start, end, pitch, vel, ch, trk


def _vlq(buf, i):
    v = 0
    while True:
        b = buf[i]
        i += 1
        v = (v << 7) | (b & 0x7F)
        if not b & 0x80:
            return v, i


def read_midi(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b"MThd":
        raise ValueError(f"{path}: no es un archivo MIDI")
    n_tracks = int.from_bytes(raw[10:12], "big")
    div = int.from_bytes(raw[12:14], "big")
    if div & 0x8000:
        raise ValueError("división SMPTE no soportada")
    tempos, sigs, notes, names = [], [], [], {}
    i = 14
    for ti in range(n_tracks):
        if raw[i:i + 4] != b"MTrk":
            break
        ln = int.from_bytes(raw[i + 4:i + 8], "big")
        ch = raw[i + 8:i + 8 + ln]
        i += 8 + ln
        t, j, status = 0, 0, 0
        open_n = {}
        while j < len(ch):
            d, j = _vlq(ch, j)
            t += d
            if j >= len(ch):
                break
            b0 = ch[j]
            if b0 & 0x80:
                status = b0
                j += 1
            if status == 0xFF:
                mt = ch[j]
                j += 1
                ml, j = _vlq(ch, j)
                md = ch[j:j + ml]
                j += ml
                if mt == 0x51 and ml == 3:
                    tempos.append((t, int.from_bytes(md, "big")))
                elif mt == 0x58 and ml >= 2:
                    sigs.append((t, md[0], 2 ** md[1]))
                elif mt == 0x03 and ti not in names:
                    names[ti] = md.decode("latin-1", "replace").strip()
            elif status in (0xF0, 0xF7):
                ml, j = _vlq(ch, j)
                j += ml
            else:
                hi, c = status & 0xF0, status & 0x0F
                if hi in (0xC0, 0xD0):
                    j += 1
                else:
                    a, b = ch[j], ch[j + 1]
                    j += 2
                    if hi == 0x90 and b > 0:
                        k = (c, a)
                        if k in open_n:                 # re-disparo sin off
                            s, v = open_n.pop(k)
                            if t > s:
                                notes.append(Note(s, t, a, v, c, ti))
                        open_n[k] = (t, b)
                    elif hi == 0x80 or (hi == 0x90 and b == 0):
                        k = (c, a)
                        if k in open_n:
                            s, v = open_n.pop(k)
                            if t > s:
                                notes.append(Note(s, t, a, v, c, ti))
        for (c, a), (s, v) in open_n.items():           # notas sin cerrar
            if t > s:
                notes.append(Note(s, t, a, v, c, ti))
    notes.sort(key=lambda n: (n.start, n.pitch))
    tempos.sort()
    sigs.sort()
    return {"tpb": div, "tempos": tempos, "sigs": sigs, "notes": notes, "names": names,
            "n_tracks": n_tracks}


# ══════════════════════════════════════════════════════════════════════════════
#  Compases
# ══════════════════════════════════════════════════════════════════════════════

def build_bars(mid, notes, force_sig=None):
    """Lista de (tick_ini, tick_fin, num, den) hasta el final de las notas."""
    tpb = mid["tpb"]
    end = max(n.end for n in notes)
    sigs = [(0, force_sig[0], force_sig[1])] if force_sig else list(mid["sigs"])
    if not sigs or sigs[0][0] > 0:
        sigs.insert(0, (0, 4, 4))
    bars = []
    t = 0
    for si, (st, num, den) in enumerate(sigs):
        nxt = sigs[si + 1][0] if si + 1 < len(sigs) else None
        tpc = max(1, tpb * 4 * num // den)
        if t < st:
            t = st
        while True:
            if t >= end and nxt is None:
                break
            if nxt is not None and t >= nxt:
                break
            bars.append((t, t + tpc, num, den))
            t += tpc
            if t >= end and nxt is None:
                break
    return bars


# ══════════════════════════════════════════════════════════════════════════════
#  Rasgos por compás
# ══════════════════════════════════════════════════════════════════════════════

_KS_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else None


def _cos(a, b):
    if a is None or b is None:
        return 1.0 if a is None and b is None else 0.0
    return sum(x * y for x, y in zip(a, b))


def estimate_key(pc):
    if sum(pc) <= 0:
        return None
    best, bs = None, -9
    for tonic in range(12):
        for mode, prof in (("major", _KS_MAJ), ("minor", _KS_MIN)):
            rot = [prof[(i - tonic) % 12] for i in range(12)]
            ma, mb = sum(pc) / 12, sum(rot) / 12
            num = sum((pc[i] - ma) * (rot[i] - mb) for i in range(12))
            da = math.sqrt(sum((x - ma) ** 2 for x in pc))
            db = math.sqrt(sum((x - mb) ** 2 for x in rot))
            if da < 1e-12:
                continue
            r = num / (da * db)
            if r > bs:
                best, bs = (tonic, mode), r
    return f"{_NAMES[best[0]]} {best[1]}" if best else None


GRID = 16


def bar_features(bars, notes):
    n = len(bars)
    starts = [b[0] for b in bars]
    pc = [[0.0] * 12 for _ in range(n)]
    bass = [[0.0] * 12 for _ in range(n)]
    onset = [[0.0] * GRID for _ in range(n)]
    top = [dict() for _ in range(n)]            # onset_tick -> pitch máximo
    low = [dict() for _ in range(n)]
    cnt = [0] * n
    vel = [0.0] * n
    reg = [0.0] * n
    for nt in notes:
        b0 = bisect_right(starts, nt.start) - 1
        if b0 < 0:
            continue
        # duración repartida por compases solapados
        bi = b0
        while bi < n and bars[bi][0] < nt.end:
            lo, hi = max(nt.start, bars[bi][0]), min(nt.end, bars[bi][1])
            if hi > lo:
                pc[bi][nt.pitch % 12] += (hi - lo)
            bi += 1
        if b0 < n and nt.start < bars[b0][1]:
            tpc = bars[b0][1] - bars[b0][0]
            pos = (nt.start - bars[b0][0]) * GRID / tpc        # reparto suave entre 2 casillas (tolera rubato)
            s0 = min(GRID - 1, int(pos))
            frac = pos - int(pos)
            onset[b0][s0] += 1.0 - frac * 0.5
            onset[b0][min(GRID - 1, s0 + 1)] += frac * 0.5
            cnt[b0] += 1
            vel[b0] += nt.vel
            reg[b0] += nt.pitch
            k = nt.start
            if k not in top[b0] or nt.pitch > top[b0][k]:
                top[b0][k] = nt.pitch
            if k not in low[b0] or nt.pitch < low[b0][k]:
                low[b0][k] = nt.pitch
    contour = []
    for i in range(n):
        seq = [top[i][k] for k in sorted(top[i])]
        iv = []
        for a, b in zip(seq, seq[1:]):
            d = b - a
            iv.append(0 if d == 0 else (1 if d > 0 else -1) * min(3, 1 + abs(d) // 3))
        contour.append(set(zip(iv, iv[1:])) | set((x,) for x in iv))
        for k, p in low[i].items():
            bass[i][p % 12] += 1.0
    feats = []
    for i in range(n):
        feats.append({
            "pc": _norm(pc[i]), "bass": _norm(bass[i]), "onset": _norm(onset[i]),
            "contour": contour[i], "empty": cnt[i] == 0,
            "count": cnt[i], "vel": (vel[i] / cnt[i]) if cnt[i] else 0.0,
            "reg": (reg[i] / cnt[i]) if cnt[i] else 0.0, "pc_raw": pc[i],
        })
    return feats


def _jac(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def similarity_matrix(feats):
    n = len(feats)
    S = [[0.0] * n for _ in range(n)]
    for i in range(n):
        fi = feats[i]
        S[i][i] = 1.0
        for j in range(i + 1, n):
            fj = feats[j]
            if fi["empty"] and fj["empty"]:
                s = 1.0
            elif fi["empty"] or fj["empty"]:
                s = 0.0
            else:
                s = (0.40 * _cos(fi["pc"], fj["pc"]) + 0.20 * _cos(fi["bass"], fj["bass"])
                     + 0.25 * _cos(fi["onset"], fj["onset"]) + 0.15 * _jac(fi["contour"], fj["contour"]))
            S[i][j] = S[j][i] = s
    return S


# ══════════════════════════════════════════════════════════════════════════════
#  Repeticiones, fronteras, etiquetas
# ══════════════════════════════════════════════════════════════════════════════

def find_repeats(S, thr=0.82, min_len=2):
    """Diagonales con S>=thr: lista de (i, j, L, media), i<j."""
    n = len(S)
    runs = []
    for k in range(1, n - min_len + 1):
        i = 0
        while i < n - k:
            if S[i][i + k] >= thr:
                j = i
                tot = 0.0
                gap = 0
                last_good = i
                while j < n - k and (S[j][j + k] >= thr or (gap == 0 and S[j][j + k] >= thr - 0.12
                                                            and j + 1 < n - k and S[j + 1][j + 1 + k] >= thr)):
                    if S[j][j + k] >= thr:
                        last_good = j
                        gap = 0
                    else:
                        gap = 1
                    tot += S[j][j + k]
                    j += 1
                L = last_good - i + 1
                if L >= min_len:
                    m = sum(S[x][x + k] for x in range(i, i + L)) / L
                    runs.append((i, i + k, L, m))
                i = max(j, i + 1)
            else:
                i += 1
    return runs


def novelty_curve(S, w):
    n = len(S)
    nov = [0.0] * (n + 1)
    for b in range(1, n):                      # frontera entre compás b-1 y b
        tot = 0.0
        cnt = 0
        for a in range(1, w + 1):
            for c in range(1, w + 1):
                p, q = b - a, b - c
                r, s = b + a - 1, b + c - 1
                if p < 0 or q < 0 or r >= n or s >= n:
                    continue
                tot += S[p][q] + S[r][s] - S[p][s] - S[r][q]
                cnt += 1
        nov[b] = tot / cnt if cnt else 0.0
    return nov


def pick_boundaries(S, runs, min_bars, sens, snap=0):
    n = len(S)
    if n < 2 * min_bars:
        return []
    w = max(2, min(8, min_bars))
    nov = novelty_curve(S, w)
    mx = max(nov) if max(nov) > 1e-9 else 1.0
    nn = [max(0.0, x) / mx for x in nov]
    edge = [0.0] * (n + 1)
    for (i, j, L, m) in runs:
        wgt = L * m
        for b in (i, i + L, j, j + L):
            if 0 < b < n:
                edge[b] += wgt
    em = max(edge) if max(edge) > 0 else 1.0
    edge_n = [e / em for e in edge]
    tau_nov = 0.30 - 0.20 * sens           # contraste mínimo absoluto del kernel
    tau_rel = 0.55 - 0.30 * sens           # y relativo al mayor pico de novedad
    cands = []
    for b in range(1, n):
        is_nov = nov[b] >= tau_nov and nn[b] >= tau_rel and nov[b] >= nov[b - 1] and nov[b] >= nov[min(n, b + 1)]
        is_rep = edge[b] >= 1.5 * min_bars and edge_n[b] >= 0.30
        if is_nov or is_rep:
            cands.append((max(nn[b] if is_nov else 0.0, edge_n[b] if is_rep else 0.0), b))
    cands.sort(key=lambda x: (-x[0], x[1]))
    chosen = []
    for _sc, b in cands:
        if b < min_bars or n - b < min_bars:
            continue
        if all(abs(b - c) >= min_bars for c in chosen):
            chosen.append(b)
    chosen.sort()
    if snap and snap > 1:
        snapped = []
        for b in chosen:
            sb = int(round(b / snap)) * snap
            if sb < min_bars or n - sb < min_bars:
                sb = b
            if not snapped or sb - snapped[-1] >= min_bars:
                snapped.append(sb)
        chosen = snapped
    return chosen


def sec_similarity(S, a, b):
    (a0, a1), (b0, b1) = a, b
    la, lb = a1 - a0, b1 - b0
    m = min(la, lb)
    if m <= 0:
        return 0.0
    s = sum(S[a0 + t][b0 + t] for t in range(m)) / m
    return s * (m / max(la, lb)) ** 0.5


def label_sections(S, spans, same_thr, var_thr):
    letters = []        # por grupo: lista de (span, variant_id)
    out = []
    for sp in spans:
        best = (-1, None, None)
        for g, members in enumerate(letters):
            for (msp, vid) in members:
                s = sec_similarity(S, sp, msp)
                if s > best[0]:
                    best = (s, g, vid)
        if best[0] >= var_thr:
            g = best[1]
            if best[0] >= same_thr:
                vid = best[2]
            else:
                vid = max(v for _, v in letters[g]) + 1
            letters[g].append((sp, vid))
        else:
            g = len(letters)
            letters.append([(sp, 0)])
            vid = 0
        out.append((g, vid, round(max(best[0], 0.0), 3)))
    names = []
    for g, vid, s in out:
        base = chr(ord("A") + g) if g < 26 else f"S{g}"
        names.append(base + "'" * vid)
    return names, [o[0] for o in out], [o[2] for o in out]


def classify_form(groups, n_bars, spans):
    seq = "".join(chr(ord("A") + g) if g < 26 else "?" for g in groups)
    k = len(seq)
    if k == 1:
        return seq, "una sola sección (sin forma detectable / material homogéneo)"
    uniq = len(set(seq))
    if uniq == k:
        return seq, "durchkomponiert (sin repeticiones de secciones)"
    if uniq == 1:
        return seq, "estrófica / variaciones sobre una misma sección"
    table = {
        "AB": "binaria", "AABB": "binaria con repeticiones (||: A :||: B :||, danza barroca)",
        "AAB": "binaria con repetición inicial", "ABB": "binaria con repetición final",
        "ABA": "ternaria (A B A)", "AABA": "AABA (forma canción de 32 compases)",
        "ABAB": "estrófica alterna (verso-estribillo)", "ABAC": "ABAC (periodo/forma canción)",
        "ABACA": "rondó (ABACA)", "ABACABA": "rondó (ABACABA)", "AABBA": "AABBA (ternaria con repeticiones)",
        "ABCA": "ABCA (retorno final)", "ABCB": "ABCB (balada)", "ABAA": "ABAA",
    }
    # forma colapsando repeticiones consecutivas
    col = seq[0]
    for ch in seq[1:]:
        if ch != col[-1]:
            col += ch
    if seq in table:
        return seq, table[seq]
    if col in table and col != seq:
        return seq, f"{table[col]} con secciones repetidas literalmente"
    if k >= 5 and col.startswith("A") and col.count("A") >= 3 and uniq >= 3:
        return seq, "tipo rondó (A reaparece varias veces entre episodios)"
    if col in ("ABAB", "ABABAB"):
        return seq, "estrófica alterna (verso-estribillo)"
    return seq, "forma libre con secciones que retornan"


# ══════════════════════════════════════════════════════════════════════════════
#  Análisis completo
# ══════════════════════════════════════════════════════════════════════════════

def analyze(path, min_bars=None, sens=0.5, same_thr=0.86, var_thr=0.70, snap=0,
            timesig=None, tracks=None, want_ssm=False):
    mid = read_midi(path)
    notes = [n for n in mid["notes"] if n.ch != 9]
    if tracks:
        notes = [n for n in notes if n.trk in tracks]
    if not notes:
        raise ValueError("el MIDI (o las pistas elegidas) no contiene notas melódicas")
    force = None
    if timesig:
        m = re.fullmatch(r"(\d+)/(\d+)", timesig)
        if not m:
            raise ValueError("--timesig debe tener la forma N/D, p. ej. 6/8")
        force = (int(m.group(1)), int(m.group(2)))
    bars = build_bars(mid, notes, force)
    n = len(bars)
    if min_bars is None:
        min_bars = 2 if n < 12 else (4 if n < 64 else (6 if n < 128 else 8))
    feats = bar_features(bars, notes)
    S = similarity_matrix(feats)
    off = [S[i][j] for i in range(n) for j in range(n) if i != j]
    mean_off = sum(off) / len(off) if off else 1.0
    runs = find_repeats(S, thr=max(0.82, min(0.96, mean_off + 0.12)), min_len=max(2, min_bars // 2)) if n >= 4 else []
    cuts = pick_boundaries(S, runs, min_bars, sens, snap) if n >= 4 else []
    edges = [0] + cuts + [n]
    spans = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    labels, groups, conf = label_sections(S, spans, same_thr, var_thr)
    form_seq, form_name = classify_form(groups, n, spans)

    # cobertura de repetición: compases que repiten material anterior
    rep = set()
    for (i, j, L, m) in runs:
        if L >= max(2, min_bars // 2):
            rep.update(range(j, j + L))
    # estadísticas por sección
    sections = []
    seen = {}
    for idx, ((a, b), lab, g, cf) in enumerate(zip(spans, labels, groups, conf)):
        pc = [0.0] * 12
        for t in range(a, b):
            for p in range(12):
                pc[p] += feats[t]["pc_raw"][p]
        cnts = [feats[t]["count"] for t in range(a, b)]
        vels = [feats[t]["vel"] for t in range(a, b) if feats[t]["count"]]
        regs = [feats[t]["reg"] for t in range(a, b) if feats[t]["count"]]
        first = seen.get(lab, idx + 1)
        seen.setdefault(lab, idx + 1)
        sections.append({
            "index": idx + 1, "label": lab, "group": chr(ord("A") + g) if g < 26 else str(g),
            "start_bar": a + 1, "end_bar": b, "n_bars": b - a,
            "key": estimate_key(pc), "notes_per_bar": round(sum(cnts) / max(1, len(cnts)), 1),
            "velocity": round(sum(vels) / len(vels), 1) if vels else 0,
            "register": round(sum(regs) / len(regs), 1) if regs else 0,
            "similarity_to_group": cf, "repeat_of_section": first,
            "bars_repeating_earlier": sum(1 for t in range(a, b) if t in rep),
        })
    res = {
        "file": str(path), "tool": "form_analyzer", "version": VERSION,
        "n_bars": n, "time_signature": f"{bars[0][2]}/{bars[0][3]}",
        "time_signature_changes": len({(b[2], b[3]) for b in bars}) > 1,
        "tpb": mid["tpb"], "n_notes": len(notes),
        "params": {"min_bars": min_bars, "sens": sens, "same_thr": same_thr, "var_thr": var_thr, "snap": snap},
        "homogeneity": round(mean_off, 3),
        "boundaries_bar": [c + 1 for c in cuts],
        "sections": sections,
        "form": " ".join(labels), "form_letters": form_seq, "form_name": form_name,
        "repetition_coverage": round(len(rep) / n, 3) if n else 0.0,
        "repeats": [{"bars": [i + 1, i + L], "again_at": [j + 1, j + L], "lag": j - i, "similarity": round(m, 3)}
                    for (i, j, L, m) in sorted(runs, key=lambda r: -r[2] * r[3])[:40]],
        "notes": [],
    }
    if mean_off > 0.9:
        res["notes"].append("material muy homogéneo (autosimilitud media %.2f): las fronteras son poco fiables" % mean_off)
    if n < 8:
        res["notes"].append("pieza muy corta: forma poco informativa")
    if len(sections) == 1 and res["repetition_coverage"] < 0.05 and n >= 16:
        res["notes"].append("sin repeticiones ni contrastes detectables: puede ser un MIDI interpretado (no cuantizado), "
                            "un compás mal declarado (prueba --timesig) o una pieza sin retornos; "
                            "también puedes bajar --var-thr o subir --sens")
    if any(s["n_bars"] > 0.6 * n for s in sections) and len(sections) > 1:
        res["notes"].append("una sección ocupa más del 60 %% de la obra; prueba --sens mayor o --min-bars menor")
    if want_ssm:
        res["_S"] = S
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  Presentación
# ══════════════════════════════════════════════════════════════════════════════

def render_ssm(S, width=64):
    n = len(S)
    step = max(1, math.ceil(n / width))
    m = (n + step - 1) // step
    shades = " .:-=+*#%@"
    lines = []
    for bi in range(m):
        row = []
        for bj in range(m):
            vals = [S[i][j] for i in range(bi * step, min(n, (bi + 1) * step))
                    for j in range(bj * step, min(n, (bj + 1) * step))]
            v = sum(vals) / len(vals)
            row.append(shades[min(9, int(max(0.0, (v - 0.4) / 0.6) * 9.99))])
        lines.append("".join(row))
    return lines, step


def print_report(r, show_ssm=False):
    print(f"\n{'═' * 78}")
    print(f"  {_c('B')}FORM ANALYZER v{VERSION}  ·  {r['file']}{_c('R')}")
    print(f"{'═' * 78}")
    print(f"  Compases: {r['n_bars']}  ·  compás {r['time_signature']}"
          + (" (con cambios)" if r["time_signature_changes"] else "")
          + f"  ·  notas: {r['n_notes']}")
    print(f"  {_c('B')}Forma:{_c('R')} {_c('GRN')}{r['form']}{_c('R')}")
    print(f"  Tipo : {r['form_name']}")
    print(f"  Repetición: {r['repetition_coverage'] * 100:.0f} % de los compases repite material anterior "
          f"· homogeneidad {r['homogeneity']:.2f}")
    print(f"\n  {_c('G')} #  sec    compases     n   tonalidad      notas/c  vel  repite{_c('R')}")
    for s in r["sections"]:
        rep = f"sec {s['repeat_of_section']}" if s["repeat_of_section"] != s["index"] else "—"
        print(f"  {s['index']:>2}  {s['label']:<5} c.{s['start_bar']:>3}–{s['end_bar']:<4} {s['n_bars']:>3}   "
              f"{(s['key'] or '—'):<13}  {s['notes_per_bar']:>6}  {s['velocity']:>4.0f}  {rep}")
    # línea de tiempo
    total = r["n_bars"]
    scale = min(1.0, 70.0 / max(1, total))
    tl = ""
    for s in r["sections"]:
        w = max(len(s["label"]) + 2, int(round(s["n_bars"] * scale)))
        tl += ("[" + s["label"]).ljust(w - 1, "─") + "]"
    print(f"\n  {tl}")
    for n in r["notes"]:
        print(f"  {_c('YEL')}⚠ {n}{_c('R')}")
    if show_ssm and "_S" in r:
        lines, step = render_ssm(r["_S"])
        print(f"\n  {_c('B')}Matriz de autosimilitud{_c('R')} (1 carácter = {step} compás/es)")
        for l in lines:
            print("  " + l)
    print(f"{'═' * 78}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  plan
# ══════════════════════════════════════════════════════════════════════════════

def _roles(r):
    """Asigna roles de canción (intro/verso/estribillo/puente/solo/outro) a las secciones."""
    secs = r["sections"]
    k = len(secs)
    groups = defaultdict(list)
    for s in secs:
        groups[s["group"]].append(s)
    avg = sum(s["n_bars"] for s in secs) / k
    roles = {}
    if k == 1:
        roles[1] = "verse1"
        return roles
    # intro / outro: secciones únicas y cortas en los extremos
    if len(groups[secs[0]["group"]]) == 1 and secs[0]["n_bars"] <= 0.75 * avg and k > 2:
        roles[1] = "intro"
    if len(groups[secs[-1]["group"]]) == 1 and secs[-1]["n_bars"] <= 0.75 * avg and k > 2:
        roles[k] = "outro"
    rest = [s for s in secs if s["index"] not in roles]
    rep_groups = sorted((g for g in groups if len([s for s in groups[g] if s["index"] not in roles]) > 1),
                        key=lambda g: (-sum(s["velocity"] + s["notes_per_bar"] for s in groups[g]) / len(groups[g])))
    if rep_groups:
        chorus_g = rep_groups[0]
        verse_g = rep_groups[1] if len(rep_groups) > 1 else None
        v_count = 0
        for s in rest:
            if s["group"] == chorus_g:
                roles[s["index"]] = "chorus"
            elif verse_g and s["group"] == verse_g:
                v_count += 1
                roles[s["index"]] = "verse1" if v_count == 1 else "verse2"
        # secciones únicas restantes: puente / solo
        for s in rest:
            if s["index"] in roles:
                continue
            roles[s["index"]] = "bridge" if "bridge" not in roles.values() else "solo"
        if verse_g is None and not any(v.startswith("verse") for v in roles.values()):
            # solo hay un grupo repetido: la primera aparición hace de verso
            first = next(s for s in rest if s["group"] == chorus_g)
            roles[first["index"]] = "verse1"
    else:
        for i, s in enumerate(rest):
            roles[s["index"]] = ["verse1", "chorus", "bridge", "solo", "verse2"][min(i, 4)]
    return roles


def build_plan(r, target):
    roles = _roles(r)
    sections = []
    for s in r["sections"]:
        sections.append({"index": s["index"], "label": s["label"], "role": roles.get(s["index"]),
                         "start_bar": s["start_bar"], "n_bars": s["n_bars"], "key": s["key"],
                         "repeat_of_section": s["repeat_of_section"],
                         "fragment": f"--fragment-start {s['start_bar']} --fragment-end {s['end_bar']}"})
    plan = {"source": r["file"], "n_bars": r["n_bars"], "time_signature": r["time_signature"],
            "form": r["form"], "form_name": r["form_name"], "sections": sections}
    if target == "song_architect":
        canon = ["intro", "verse1", "prechorus", "chorus", "verse2", "bridge", "solo", "outro"]
        used = [x for x in canon if x in set(roles.values())]
        lens = sorted(s["n_bars"] for s in r["sections"])
        med = lens[len(lens) // 2]
        cmd = f"python song_architect.py melodia.mid --sections {' '.join(used)} --bars-per-section {med}"
        plan["song_architect"] = {
            "command": cmd, "sections": used, "bars_per_section": med,
            "warning": ("song_architect genera cada sección UNA vez y en orden canónico "
                        "(intro, verse1, prechorus, chorus, verse2, bridge, solo, outro); "
                        "las repeticiones y el orden real de la forma %s no se reproducen." % r["form"]),
        }
    return plan


# ══════════════════════════════════════════════════════════════════════════════
#  compare
# ══════════════════════════════════════════════════════════════════════════════

def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _canon(groups):
    m, out = {}, []
    for g in groups:
        if g not in m:
            m[g] = chr(ord("A") + len(m))
        out.append(m[g])
    return "".join(out)


def compare_forms(ra, rb):
    ga = [s["group"] for s in ra["sections"]]
    gb = [s["group"] for s in rb["sections"]]
    ca, cb = _canon(ga), _canon(gb)
    label_sim = 1 - _lev(ca, cb) / max(len(ca), len(cb), 1)
    ba = [s["start_bar"] / ra["n_bars"] for s in ra["sections"][1:]]
    bb = [s["start_bar"] / rb["n_bars"] for s in rb["sections"][1:]]
    if not ba and not bb:
        bsim = 1.0
    elif not ba or not bb:
        bsim = 0.0
    else:
        d1 = sum(min(abs(x - y) for y in bb) for x in ba) / len(ba)
        d2 = sum(min(abs(x - y) for y in ba) for x in bb) / len(bb)
        bsim = max(0.0, 1 - (d1 + d2))
    rsim = 1 - abs(ra["repetition_coverage"] - rb["repetition_coverage"])
    lratio = min(ra["n_bars"], rb["n_bars"]) / max(ra["n_bars"], rb["n_bars"])
    score = 0.45 * label_sim + 0.25 * bsim + 0.15 * rsim + 0.15 * lratio
    return {"score": round(score, 3), "form_a": ra["form"], "form_b": rb["form"],
            "canonical_a": ca, "canonical_b": cb,
            "components": {"label_sequence": round(label_sim, 3), "boundary_positions": round(bsim, 3),
                           "repetition_coverage": round(rsim, 3), "length_ratio": round(lratio, 3)},
            "bars_a": ra["n_bars"], "bars_b": rb["n_bars"]}


# ══════════════════════════════════════════════════════════════════════════════
#  selftest (MIDIs sintéticos con forma conocida)
# ══════════════════════════════════════════════════════════════════════════════

def _vlq_w(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    return bytes(reversed(out))


def _write_midi(path, events, tpb=480, sig=(4, 4)):
    """events: lista (tick, pitch, dur, vel). Una sola pista."""
    ev = []
    for t, p, d, v in events:
        ev.append((t, 1, p, v))
        ev.append((t + d, 0, p, 0))
    ev.sort(key=lambda e: (e[0], e[1]))
    tr = bytearray()
    den_pow = int(math.log2(sig[1]))
    tr += b"\x00\xff\x58\x04" + bytes([sig[0], den_pow, 24, 8])
    tr += b"\x00\xff\x51\x03" + (500000).to_bytes(3, "big")
    last = 0
    for t, on, p, v in ev:
        tr += _vlq_w(t - last) + bytes([0x90 if on else 0x80, p, v if on else 0])
        last = t
    tr += b"\x00\xff\x2f\x00"
    Path(path).write_bytes(b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big")
                           + tpb.to_bytes(2, "big") + b"MTrk" + len(tr).to_bytes(4, "big") + bytes(tr))


def _section_events(kind, bar0, n_bars, bar_ticks, vel=80):
    """Sección sintética: cada compás tiene contenido propio (determinista por tipo y
    número de compás), de modo que dos apariciones de la misma sección son idénticas."""
    import random
    scales = {"A": [60, 62, 64, 65, 67, 69, 71, 72], "B": [54, 56, 58, 59, 61, 63, 65, 66],
              "C": [63, 65, 67, 68, 70, 72, 74, 75]}
    rhythms = {"A": [4, 4, 4, 4], "B": [2, 2, 2, 2, 2, 2, 2, 2], "C": [8, 8]}
    sc, rh = scales[kind], rhythms[kind]
    ev = []
    for b in range(n_bars):
        rnd = random.Random(ord(kind) * 1000 + b)
        t0 = (bar0 + b) * bar_ticks
        unit = bar_ticks // sum(rh) if sum(rh) else bar_ticks
        pos = 0
        for r in rh:
            d = r * unit
            p = rnd.choice(sc)
            ev.append((t0 + pos, p, max(1, d - 10), vel))
            pos += d
        ev.append((t0, rnd.choice(sc) - 24, max(1, bar_ticks - 20), vel - 20))     # bajo
    return ev


def cmd_selftest(args):
    ok_all = True
    cases = [("AABA", (4, 4), 4, 8), ("ABAB", (4, 4), 4, 8), ("AB", (3, 4), 3, 8), ("ABA", (6, 8), 3, 8),
             ("AABAC", (4, 4), 4, 8)]
    with tempfile.TemporaryDirectory() as td:
        for form, sig, beats_q, nb in cases:
            tpb = 480
            bar_ticks = tpb * 4 * sig[0] // sig[1]
            ev = []
            for si, ch in enumerate(form):
                ev += _section_events(ch, si * nb, nb, bar_ticks)
            p = os.path.join(td, f"t_{form}_{sig[0]}{sig[1]}.mid")
            _write_midi(p, ev, tpb, sig)
            r = analyze(p, min_bars=4)
            got = _canon([s["group"] for s in r["sections"]])
            want = _canon(list(form))
            n_ok = r["n_bars"] == nb * len(form)
            ok = (got == want) and n_ok
            ok_all &= ok
            print(f"  {'✔' if ok else '✘'} forma real {form:<6} compás {sig[0]}/{sig[1]}  →  "
                  f"detectada {r['form']:<22} compases {r['n_bars']}/{nb * len(form)}")
        # comparación de una forma consigo misma y con otra distinta
        pa = os.path.join(td, "a.mid")
        pb = os.path.join(td, "b.mid")
        bt = 480 * 4
        _write_midi(pa, sum((_section_events(ch, i * 8, 8, bt) for i, ch in enumerate("AABA")), []))
        _write_midi(pb, sum((_section_events(ch, i * 8, 8, bt) for i, ch in enumerate("ABCD"[:3] if False else "ABC")), []))
        same = compare_forms(analyze(pa), analyze(pa))["score"]
        diff = compare_forms(analyze(pa), analyze(pb))["score"]
        ok = same > 0.99 and diff < same
        ok_all &= ok
        print(f"  {'✔' if ok else '✘'} compare: consigo mismo {same:.2f}, AABA vs ABC {diff:.2f}")
    print(f"\n  {'SELFTEST OK' if ok_all else 'SELFTEST CON FALLOS'}")
    return 0 if ok_all else 1


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _add_analysis_args(p):
    p.add_argument("--min-bars", type=int, default=None, help="Longitud mínima de sección (default: 2 si <12 compases, 4 si <64, 6 si <128, 8 si más)")
    p.add_argument("--sens", type=float, default=0.5, help="Sensibilidad de fronteras 0-1 (default 0.5)")
    p.add_argument("--same-thr", type=float, default=0.86, help="Similitud de 'misma sección' (default 0.86)")
    p.add_argument("--var-thr", type=float, default=0.70, help="Similitud de 'sección variada' (default 0.70)")
    p.add_argument("--snap", type=int, default=0, help="Ajusta fronteras a múltiplos de N compases")
    p.add_argument("--timesig", default=None, help="Fuerza el compás, p. ej. 6/8")
    p.add_argument("--track", type=int, nargs="+", default=None, help="Pistas a usar (default: todas salvo percusión)")
    p.add_argument("--no-color", action="store_true")


def _run(path, a, want_ssm=False):
    return analyze(path, min_bars=a.min_bars, sens=max(0.0, min(1.0, a.sens)), same_thr=a.same_thr,
                   var_thr=a.var_thr, snap=a.snap, timesig=a.timesig, tracks=a.track, want_ssm=want_ssm)


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="form_analyzer.py",
                                 description="Forma musical: secciones, repeticiones y etiquetas.")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("detect", help="Analiza la forma de un MIDI")
    p.add_argument("midi")
    _add_analysis_args(p)
    p.add_argument("--json", default=None, help="Escribe el análisis completo ('-' = stdout)")
    p.add_argument("--ssm", action="store_true", help="Dibuja la matriz de autosimilitud")
    p = sp.add_parser("plan", help="Estructura reutilizable por otras herramientas")
    p.add_argument("midi")
    _add_analysis_args(p)
    p.add_argument("--target", choices=["json", "song_architect"], default="json")
    p.add_argument("--output", default=None, help="Escribe la estructura en este fichero")
    p = sp.add_parser("compare", help="Fidelidad estructural entre dos MIDIs")
    p.add_argument("midi_a")
    p.add_argument("midi_b")
    _add_analysis_args(p)
    p.add_argument("--json", default=None, help="Escribe el resultado ('-' = stdout)")
    sp.add_parser("selftest", help="Verifica con MIDIs sintéticos de forma conocida")
    args = ap.parse_args(argv)
    if getattr(args, "no_color", False):
        _USE_COLOR = False

    try:
        if args.cmd == "selftest":
            return cmd_selftest(args)
        if args.cmd == "detect":
            r = _run(args.midi, args, want_ssm=args.ssm)
            if args.json == "-":
                d = {k: v for k, v in r.items() if k != "_S"}
                print(json.dumps(d, ensure_ascii=False, indent=1))
                return 0
            print_report(r, show_ssm=args.ssm)
            if args.json:
                d = {k: v for k, v in r.items() if k != "_S"}
                Path(args.json).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  JSON: {args.json}")
            return 0
        if args.cmd == "plan":
            r = _run(args.midi, args)
            plan = build_plan(r, args.target)
            txt = json.dumps(plan, ensure_ascii=False, indent=1)
            if args.output:
                Path(args.output).write_text(txt, encoding="utf-8")
                print(f"  Estructura escrita en {args.output}   forma: {r['form']}")
            else:
                print(txt)
            return 0
        if args.cmd == "compare":
            ra, rb = _run(args.midi_a, args), _run(args.midi_b, args)
            res = compare_forms(ra, rb)
            if args.json == "-":
                print(json.dumps(res, ensure_ascii=False, indent=1))
                return 0
            print(f"\n  {_c('B')}Fidelidad estructural: {res['score']:.2f}{_c('R')}")
            print(f"  A: {res['form_a']}  ({res['bars_a']} compases)")
            print(f"  B: {res['form_b']}  ({res['bars_b']} compases)")
            for k, v in res["components"].items():
                print(f"    {k:<22} {v:.2f}")
            if args.json:
                Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print()
            return 0
    except (ValueError, OSError, IndexError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
