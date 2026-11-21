#!/usr/bin/env python3
"""
midi_index.py — Indexa miles de MIDIs y los consulta sin conexión: por
ejemplo, por forma de trayectoria, por progresión, por rasgos y por
«sensaciones» (versión 1.0).

──────────────────────────────────────────────────
Fundamento
──────────────────────────────────────────────────
Un índice sirve si lo que guarda vale para consultas que aún no sabes
formular. Por eso se guardan MEDIDAS CRUDAS, versionadas y válidas para
cualquier estilo, y las «sensaciones» se calculan al consultar:

  · Rasgos agnósticos al estilo (no dependen de adivinar melodía/bajo):
    tonales (perfil de clases de altura relativo a la tónica), rítmicos
    (fase en el pulso, IOI, duraciones, tempo en notas/s), de textura
    (polifonía, batería), de registro, melódicos (histograma de intervalos
    del perfil superior), armónicos (acordes por plantilla, grados,
    movimiento de raíz INDEPENDIENTE de la tonalidad, cadencias, bucle),
    de dinámica y de trayectoria (variación, tendencia, periodicidad,
    reinicios por compás).
  · Series por compás (para consultar formas) y ventanas de forma de 4/8/16
    compases (vector de 8 puntos, centrado y escalado: invariante a
    transposición y a amplitud).
  · Gestos de gesture_lab (rampas, dientes de sierra, bucles, pedales…).
  · Cada fila lleva la versión del extractor y el hash del contenido: si
    mejoras el extractor, `index` recalcula solo lo desactualizado; los
    duplicados exactos se detectan y no se cuentan dos veces.

Las sensaciones (energía, brillo, tensión, estabilidad, repetición,
insistencia, ambigüedad, movimiento, apertura) son combinaciones de z-scores
de rasgos y se expresan como PERCENTIL dentro de TU corpus: son hipótesis
heurísticas sin calibrar. Para calibrarlas con tu oído: `tag` + `similar --tag`
o, mejor, `calibrate TAG`: aprende una regresión logística (L2, clases
balanceadas) con tus piezas etiquetadas, la guarda en la base y la ofrece como
una sensación más (`find --feel NOMBRE>0.8`). Informa del AUC en validación
cruzada, de los rasgos que más pesan y de piezas candidatas a revisar.

Compatibilidad: los rasgos se guardan por nombre. Añadir rasgos nuevos no
invalida lo indexado (las piezas antiguas quedan sin ese rasgo hasta
reindexar). Si cambia el SIGNIFICADO de uno existente, sube EXTRACTOR y
`index` recalculará solo esas filas.

Limitaciones: compás y tempo iniciales (cambios de compás se cuentan, no se
siguen); acordes por plantilla sobre ventanas de media/una barra (no hay
inversiones ni extensiones >7.ª); sin timbre; los MIDIs expresivos sin
cuadrícula degradan los rasgos rítmicos (la fracción `rhy_other` lo señala).

──────────────────────────────────────────────────
Modos de uso
──────────────────────────────────────────────────
  midi_index.py index  CARPETA... --db corpus.db [--jobs 4] [--force]
  midi_index.py info   --db corpus.db [--errors]
  midi_index.py show   ARCHIVO --db corpus.db [--json]
  midi_index.py similar ARCHIVO|--tag T --db corpus.db [--groups g1,g2]
                       [--exclude g] [--weights rhythm=2] [--invariant]
  midi_index.py find   --db corpus.db [--range f:min:max] [--gesture ...]
                       [--progression "i bVI bIII bVII" [--anchor]]
                       [--feel tension>0.7] [--tag T] [--sort f --desc]
  midi_index.py shape  --db corpus.db --from ARCHIVO --feature reg_mean
                       --bars 1-8   |  --values 0,1,2,3,0,1,2,3
                       [--amp-tol 0.5] [--sign] [--per-file 1]
  midi_index.py tag    add|rm|ls ...
  midi_index.py calibrate TAG --db corpus.db [--neg T2] [--name N]
  midi_index.py export --db corpus.db --out rasgos.csv
  midi_index.py selftest

──────────────────────────────────────────────────
Ejemplos
──────────────────────────────────────────────────
  midi_index.py index ~/midis --db corpus.db --jobs 4
  midi_index.py find --db corpus.db --gesture "ramp_reset:reg_mean>=0.6" \\
      --feel "tension>0.6" --progression "i bVI bIII bVII"
  midi_index.py shape --db corpus.db --from arcane.mid --feature reg_mean \\
      --bars 1-8 --per-file 1 -n 15
  midi_index.py similar arcane.mid --db corpus.db --groups tonal,gesture

──────────────────────────────────────────────────
Dependencias
──────────────────────────────────────────────────
  numpy, mido; gesture_lab.py en el mismo directorio (series y gestos).
  Códigos de salida: 0 ok, 1 error de datos/consulta, 2 uso/entorno.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
import time
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VERSION = "1.0"
EXTRACTOR = "1.1"
NAN = float("nan")
ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI",
         "bVII", "VII"]
_ROMAN_DEG = {"I": 0, "II": 2, "III": 4, "IV": 5, "V": 7, "VI": 9, "VII": 11}
MINORQ = ("m", "m7", "d", "o")      # 'o' = semidisminuida
# (calidad, intervalos, tercera/quinta/séptima esperadas)
TEMPLATES = [
    ("M", (0, 4, 7)), ("m", (0, 3, 7)), ("d", (0, 3, 6)), ("a", (0, 4, 8)),
    ("5", (0, 7)), ("7", (0, 4, 7, 10)), ("M7", (0, 4, 7, 11)),
    ("m7", (0, 3, 7, 10)), ("o", (0, 3, 6, 10)),
]
QUALS = [t[0] for t in TEMPLATES] + ["x"]
GST_SHAPES = ["ramp", "ramp_reset", "fall", "fall_reset", "plateau",
              "constant", "periodic", "loop", "tonal_wander", "open_fifths",
              "micro_cycle", "pedal", "cell_lock", "burst"]
TRJ_SERIES = ["all_pitch", "low_pitch", "all_onsets", "rough", "span"]
# series de ventana y amplitud mínima (en sus unidades) para guardarlas
WIN_SERIES = {"all_pitch": 2.0, "low_pitch": 2.0, "high_pitch": 2.0,
              "all_onsets": 0.5, "rough": 0.1, "span": 3.0, "vel": 8.0,
              "reg_mean": 2.0, "reg_max": 2.0, "bass_low": 2.0}
WIN_LENGTHS = (4, 8, 16)
SEC_W = 16              # compases por sección para gestos locales
SEC_MIN_BARS = 25       # por debajo, la pieza entera es una sola sección
SEC_MAX = 48            # máximo de secciones analizadas por pieza
K = 8

GROUPS = {"ton": "tonal", "rhy": "rhythm", "tex": "texture", "reg": "register",
          "mel": "melodic", "har": "harmony", "dyn": "dynamics",
          "trj": "trajectory", "gst": "gesture", "sec": "sections"}

FEELS = {
    "energy": [("rhy_onsets_per_sec", 1), ("dyn_mean", .5), ("tex_poly_mean", .5),
               ("rhy_16th", .5), ("tex_drum_density", .3)],
    "brightness": [("reg_mean", 1), ("ton_minor", -1), ("ton_offscale", .2)],
    "tension": [("har_rough", 1), ("har_chrom", 1), ("ton_offscale", 1),
                ("ton_r", -1), ("har_q_d", .5), ("har_q_x", 1)],
    "stability": [("ton_r", 1), ("har_tonic", 1), ("har_final_tonic", .5),
                  ("har_cad_auth", .5), ("har_loop", .5)],
    "repetition": [("har_loop", 1), ("mel_recur", 1), ("trj_all_pitch_per", .5),
                   ("gst_loop", .5), ("gst_periodic", .5)],
    "insistence": [("gst_ramp_reset", 1), ("gst_fall_reset", 1),
                   ("gst_periodic", .5), ("gst_micro_cycle", .5),
                   ("trj_all_pitch_reset", 1)],
    "ambiguity": [("ton_ambig", -1), ("gst_tonal_wander", 1), ("ton_changes", 1),
                  ("gst_open_fifths", .5)],
    "motion": [("mel_abs_mean", 1), ("mel_turn", .5), ("rhy_onbeat", -.5)],
    "openness": [("gst_open_fifths", 1), ("har_q_5", 1), ("tex_poly_mean", -.3)],
}


# ══════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════
def _np():
    try:
        import numpy as np
        return np
    except ImportError:
        sys.stderr.write("Error: falta numpy (pip install numpy)\n")
        sys.exit(2)


def _gl():
    try:
        import gesture_lab as gl
        return gl
    except ImportError as e:
        sys.stderr.write("Error: no se puede importar gesture_lab.py "
                         "(debe estar junto a este script): %s\n" % e)
        sys.exit(2)


def _ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.isdir(d):
        os.makedirs(d)


def _fnum(v, nd=4):
    if v is None:
        return None
    v = float(v)
    if math.isnan(v) or math.isinf(v):
        return None
    return round(v, nd)


def group_of(name):
    return GROUPS.get(name.split("_", 1)[0], "other")


def roman(deg, q):
    s = ROMAN[deg % 12]
    if q in MINORQ:
        s = s.lower()
    suf = {"d": "°", "a": "+", "7": "7", "M7": "M7", "m7": "7", "o": "ø",
           "5": "5", "x": "?"}.get(q, "")
    return s + suf


def parse_chord_token(tok):
    """'bVI' → (8, 'maj'); 'iv' → (5, 'min'). None si no es válido."""
    m = re.match(r"^([b#]?)(VII|VI|IV|V|III|II|I|vii|vi|iv|v|iii|ii|i)(.*)$", tok)
    if not m:
        return None
    acc, num, rest = m.groups()
    deg = _ROMAN_DEG[num.upper()] + (-1 if acc == "b" else 1 if acc == "#" else 0)
    minor = num.islower() or "°" in rest or "ø" in rest
    return deg % 12, ("min" if minor else "maj")


def qclass(q):
    return "min" if q in MINORQ else "maj"


# ══════════════════════════════════════════════════
#  EXTRACTOR
# ══════════════════════════════════════════════════
def _midi_extras(path):
    """(drum_notes, drum_onsets, n_tempo, n_ts, bpm_medio, duración_s, ticks)."""
    import mido
    mid = mido.MidiFile(path)
    tpb = float(mid.ticks_per_beat)
    tempos, ts_ch, drums, maxtick = [], 0, [], 0
    for tr in mid.tracks:
        tick = 0
        for msg in tr:
            tick += msg.time
            if msg.is_meta:
                if msg.type == "set_tempo":
                    tempos.append((tick, msg.tempo))
                elif msg.type == "time_signature":
                    ts_ch += 1
            elif msg.type == "note_on" and msg.velocity > 0 and msg.channel == 9:
                drums.append(tick / tpb)
        maxtick = max(maxtick, tick)
    tempos.sort()
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, 500000))
    secs = 0.0
    for i, (tk, tp) in enumerate(tempos):
        nxt = tempos[i + 1][0] if i + 1 < len(tempos) else maxtick
        if nxt > tk:
            secs += (nxt - tk) / tpb * tp / 1e6
    return drums, len(set(t for _, t in tempos)) - 1, max(0, ts_ch - 1), secs


def _entropy(counts):
    tot = float(sum(counts))
    if tot <= 0:
        return NAN
    return -sum(c / tot * math.log(c / tot, 2) for c in counts if c > 0)


def _pc_rough(m):
    """Aspereza de un perfil de clases de altura (ponderada por masa)."""
    w = {1: 1.0, 2: 0.35, 6: 0.6, 10: 0.35, 11: 1.0}
    tot = sum(m)
    if tot <= 0:
        return NAN
    r = 0.0
    for i in range(12):
        if m[i] <= 0:
            continue
        for j in range(i + 1, 12):
            if m[j] > 0:
                r += m[i] * m[j] * w.get(j - i, 0.0)
    return 2.0 * r / (tot * tot)


def _classify_chord(m, bass_pc):
    """(raíz, calidad, puntuación) por plantilla; None si silencio; calidad 'x' si no hay ajuste."""
    tot = float(sum(m))
    if tot <= 1e-9:
        return None
    best, bs = None, -9.0
    for r in range(12):
        for q, iv in TEMPLATES:
            ins = sum(m[(r + i) % 12] for i in iv) / tot
            out = 1.0 - ins
            sc = ins - 0.6 * out
            miss = 0
            for i in iv:
                if m[(r + i) % 12] / tot < 0.04:
                    miss += 1
            if q in ("7", "M7", "m7", "o"):
                if m[(r + iv[3]) % 12] / tot < 0.10:
                    continue
                sc -= 0.05
            if q == "5":
                sc -= 0.08
            sc -= 0.25 * miss
            if bass_pc == r:
                sc += 0.12
            if sc > bs:
                bs, best = sc, (r, q)
    if best is None or bs < 0.30:
        return (bass_pc if bass_pc is not None else 0, "x", bs)
    return best + (bs,)


def _autocorr_peak(x, lo=2, hi=16):
    np = _np()
    a = np.asarray(x, dtype=float)
    n = len(a)
    if n < 8 or a.std() < 1e-9:
        return 0.0, 0
    a = a - a.mean()
    best, bl = 0.0, 0
    for lag in range(lo, min(hi, n // 2) + 1):
        c = float(np.dot(a[:-lag], a[lag:]) / (np.dot(a, a) + 1e-12))
        if c > best + 0.03 or (bl == 0 and c > best):
            best, bl = c, lag
    return max(0.0, best), bl


def _reset_rate(x):
    n = len(x)
    if n < 6:
        return 0.0
    rng = max(x) - min(x)
    if rng < 1e-9:
        return 0.0
    cnt = 0
    for i in range(3, n):
        if x[i] - x[i - 1] < -0.4 * rng and x[i - 1] - x[i - 4 if i >= 4 else 0] > 0.25 * rng:
            cnt += 1
    return cnt * 8.0 / n


def _shape_vec(win, gl):
    v, amp = gl.shape_vector([float(a) for a in win])
    return v, amp


def _slice_piece(p, gl, b0, b1):
    """Sub-pieza con los compases [b0, b1) re-ubicados desde 0 (o None)."""
    import copy
    q = copy.copy(p)
    t0, t1 = b0 * p.bar_beats, b1 * p.bar_beats

    def cut(lst):
        return [(n[0] - t0, n[1], n[2], n[3]) for n in lst
                if t0 - 1e-9 <= n[0] < t1 - 1e-9]
    q.mel, q.bass_all, q.harm_all = cut(p.mel), cut(p.bass_all), cut(p.harm_all)
    if len(q.mel) < 8:
        return None
    vs = []
    for v in p.voices:
        nt = cut(v["notes"])
        vs.append(dict(v, notes=nt,
                       mean=gl._mean([n[2] for n in nt]) if nt else 0.0))
    q.voices = vs
    q._bass_on = [n[0] for n in q.bass_all]
    q._bass_maxdur = max([n[1] for n in q.bass_all] or [0.0])
    q.nbars = b1 - b0
    hist = [0.0] * 12
    for n in q.mel + q.harm_all:
        hist[n[2] % 12] += max(n[1], 0.05)
    ke = gl._key_from_hist(hist)
    if ke:
        (r, t, m), (r2, t2, m2) = ke
        q.key = dict(tonic_pc=t, tonic=gl._pcname(t), mode=m, r=r,
                     second=dict(tonic=gl._pcname(t2), mode=m2, r=r2))
    return q


def _section_ranges(nb):
    if nb < SEC_MIN_BARS:
        return []
    stride = max(SEC_W // 2, int(math.ceil((nb - SEC_W) / float(SEC_MAX - 1))))
    starts = list(range(0, nb - SEC_W + 1, stride))
    if starts[-1] != nb - SEC_W:
        starts.append(nb - SEC_W)
    return [(s, s + SEC_W) for s in starts]


def _section_gestures(p, gl, glob):
    """Gestos hallados en ventanas de SEC_W compases, con compases absolutos.
    Se descartan los que repiten uno global y los solapados (>50 %) de la
    misma forma y rasgo, conservando el de mayor puntuación."""
    out = []
    for (a, b) in _section_ranges(p.nbars):
        q = _slice_piece(p, gl, a, b)
        if q is None:
            continue
        try:
            an = gl.analyze_piece(q)
        except Exception:  # noqa
            continue
        for g in an["gestures"]:
            g = dict(g)
            bars = g["bars"] or [1, b - a]
            g["bars"] = [bars[0] + a, bars[1] + a]
            g["text"] = "c.%d-%d: %s" % (g["bars"][0], g["bars"][1], g["text"])
            out.append(g)
    out.sort(key=lambda g: -g["score"])
    keep = []
    gkeys = set((g["shape"], g["feature"], tuple(g["bars"] or ())) for g in glob)
    for g in out:
        if (g["shape"], g["feature"], tuple(g["bars"])) in gkeys:
            continue
        dup = False
        for k in keep:
            if k["shape"] == g["shape"] and k["feature"] == g["feature"]:
                lo = max(k["bars"][0], g["bars"][0])
                hi = min(k["bars"][1], g["bars"][1])
                span = min(k["bars"][1] - k["bars"][0],
                           g["bars"][1] - g["bars"][0]) + 1
                if hi - lo + 1 > 0.5 * span:
                    dup = True
                    break
        if not dup:
            keep.append(g)
    return keep[:120]


def extract(path, opts=None):
    """Devuelve dict(meta, feats, series, gestures, chords, windows)."""
    np = _np()
    gl = _gl()
    opts = opts or {}
    p = gl.load_piece(path, min_notes=opts.get("min_notes", 8))
    drums, n_tempo, n_ts, dur_s = _midi_extras(path)
    N = sorted(n for v in p.voices for n in v["notes"])
    if len(N) > opts.get("max_notes", 400000):
        raise ValueError("demasiadas notas (%d)" % len(N))
    on = np.array([n[0] for n in N])
    du = np.array([max(n[1], 0.02) for n in N])
    pi = np.array([n[2] for n in N])
    ve = np.array([n[3] for n in N])
    nb, bb = p.nbars, p.bar_beats
    end_beat = float((on + du).max())
    total_beats = max(end_beat, 1e-6)
    bpm_avg = 60.0 * total_beats / dur_s if dur_s > 0 else p.tempo_bpm
    key = p.key
    tonic, minor = key["tonic_pc"], key["mode"] == "minor"
    F = {}

    # ── tonal ─────────────────────────────────────
    h = np.zeros(12)
    np.add.at(h, (pi % 12).astype(int), np.maximum(du, 0.05))
    hr = np.roll(h, -tonic)
    hr = hr / hr.sum()
    for i in range(12):
        F["ton_pc_rel_%d" % i] = float(hr[i])
    scale = {0, 2, 3, 5, 7, 8, 10, 11} if minor else {0, 2, 4, 5, 7, 9, 11}
    F["ton_offscale"] = float(sum(hr[i] for i in range(12) if i not in scale))
    F["ton_minor"] = 1.0 if minor else 0.0
    F["ton_r"] = float(key["r"])
    sec = key.get("second")
    F["ton_r2"] = float(sec["r"]) if sec else NAN
    F["ton_ambig"] = F["ton_r"] - F["ton_r2"] if sec else NAN
    F["ton_pc_entropy"] = float(_entropy(list(h)) / math.log(12, 2))

    # ── rítmico ───────────────────────────────────
    key48 = np.round(on * 48).astype(np.int64)
    uniq, cnt = np.unique(key48, return_counts=True)
    ou = uniq / 48.0
    F["rhy_onsets_per_beat"] = len(ou) / total_beats
    F["rhy_onsets_per_sec"] = len(ou) / dur_s if dur_s > 0 else NAN
    F["rhy_tempo_log"] = math.log(max(bpm_avg, 1.0) / 120.0, 2)
    fr = ou % 1.0
    on_b = (fr < 0.06) | (fr > 0.94)
    e8 = np.abs(fr - 0.5) < 0.06
    s16 = (np.abs(fr - 0.25) < 0.06) | (np.abs(fr - 0.75) < 0.06)
    trp = np.zeros(len(fr), dtype=bool)
    for c3 in (1 / 6.0, 1 / 3.0, 2 / 3.0, 5 / 6.0):
        trp |= np.abs(fr - c3) < 0.04
    n_o = float(len(ou))
    F["rhy_onbeat"] = float(on_b.sum() / n_o)
    F["rhy_8th"] = float((e8 & ~on_b).sum() / n_o)
    F["rhy_16th"] = float((s16 & ~on_b & ~e8).sum() / n_o)
    F["rhy_triplet"] = float((trp & ~on_b & ~e8 & ~s16).sum() / n_o)
    F["rhy_other"] = float(1.0 - F["rhy_onbeat"] - F["rhy_8th"] - F["rhy_16th"]
                           - F["rhy_triplet"])
    F["rhy_downbeat"] = float((np.abs(ou % bb) < 0.06).sum() / n_o)
    if len(ou) > 2:
        ioi = np.round(np.diff(ou) * 12).astype(int)
        F["rhy_ioi_entropy"] = float(_entropy(list(Counter(ioi.tolist()).values())))
    else:
        F["rhy_ioi_entropy"] = NAN
    edges = [0.2, 0.45, 0.9, 1.8, 3.6]
    bins = np.digitize(du, edges)
    for i in range(6):
        F["rhy_dur_%d" % i] = float((bins == i).sum() / len(du))
    leg = []
    for v in p.voices:
        nt = v["notes"]
        for a, b in zip(nt, nt[1:]):
            gap = b[0] - a[0]
            if gap > 1e-3:
                leg.append(min(1.5, a[1] / gap))
    F["rhy_legato"] = float(np.median(leg)) if leg else NAN
    F["rhy_meter"] = float(bb)
    F["rhy_compound"] = 1.0 if (p.time_signature[1] == 8 and
                                p.time_signature[0] in (6, 9, 12)) else 0.0
    F["rhy_changes"] = float(n_tempo + n_ts)

    # ── textura ───────────────────────────────────
    t_ev = np.concatenate([on, on + du])
    d_ev = np.concatenate([np.ones(len(on)), -np.ones(len(on))])
    o = np.lexsort((d_ev, t_ev))
    t_s, c_s = t_ev[o], np.cumsum(d_ev[o])
    seg = np.diff(t_s)
    lvl = c_s[:-1]
    act = seg[lvl > 0].sum()
    F["tex_poly_mean"] = float((seg * lvl).sum() / act) if act > 0 else NAN
    F["tex_poly_max"] = float(lvl.max()) if len(lvl) else NAN
    F["tex_poly_ge3"] = float(seg[lvl >= 3].sum() / act) if act > 0 else NAN
    F["tex_rest"] = float(max(0.0, 1.0 - act / total_beats))
    F["tex_voices"] = float(sum(1 for v in p.voices if len(v["notes"]) >= 8))
    F["tex_chordal"] = float((cnt >= 3).sum() / n_o)
    F["tex_drums"] = 1.0 if drums else 0.0
    F["tex_drum_density"] = len(set(round(d * 48) for d in drums)) / total_beats \
        if drums else 0.0

    # ── registro ──────────────────────────────────
    F["reg_mean"] = float(pi.mean())
    F["reg_std"] = float(pi.std())
    F["reg_lo"] = float(np.percentile(pi, 5))
    F["reg_hi"] = float(np.percentile(pi, 95))
    F["reg_range"] = F["reg_hi"] - F["reg_lo"]
    sky = gl._skyline(N, "max")
    skl = gl._skyline(N, "min")
    F["reg_top_mean"] = float(np.mean([n[2] for n in sky]))
    F["reg_low_mean"] = float(np.mean([n[2] for n in skl]))

    # ── melódico (perfil superior de todas las notas) ──
    sp = [n[2] for n in sky]
    ivs = [b - a for a, b in zip(sp, sp[1:])]
    names = ["mel_int_0", "mel_up_1", "mel_up_2", "mel_up_3", "mel_up_4",
             "mel_dn_1", "mel_dn_2", "mel_dn_3", "mel_dn_4"]
    if len(ivs) >= 6:
        def cls(v):
            a = abs(v)
            if a == 0:
                return 0
            k = 1 if a <= 2 else 2 if a <= 4 else 3 if a <= 7 else 4
            return k if v > 0 else k + 4
        c = Counter(cls(v) for v in ivs)
        for i, nm in enumerate(names):
            F[nm] = c.get(i, 0) / float(len(ivs))
        nz = [v for v in ivs if v]
        F["mel_abs_mean"] = float(np.mean([abs(v) for v in ivs]))
        F["mel_up_share"] = (sum(1 for v in nz if v > 0) / float(len(nz))
                             if nz else NAN)
        sg = [1 if v > 0 else -1 for v in nz]
        F["mel_turn"] = (sum(1 for a, b in zip(sg, sg[1:]) if a != b)
                         / float(len(sg) - 1) if len(sg) > 1 else NAN)
        grams = [tuple(ivs[i:i + 4]) for i in range(len(ivs) - 3)]
        gc = Counter(grams)
        F["mel_recur"] = (sum(c2 for c2 in gc.values() if c2 >= 2)
                          / float(len(grams)) if grams else NAN)
        F["mel_range"] = float(np.percentile(sp, 95) - np.percentile(sp, 5))
    else:
        for nm in names + ["mel_abs_mean", "mel_up_share", "mel_turn",
                           "mel_recur", "mel_range"]:
            F[nm] = NAN

    # ── dinámica ──────────────────────────────────
    F["dyn_mean"] = float(ve.mean())
    F["dyn_std"] = float(ve.std())
    F["dyn_const"] = float(Counter(ve.tolist()).most_common(1)[0][1] / len(ve))

    # ── armonía por segmentos ─────────────────────
    seg_len = bb / 2.0 if bb in (4.0, 6.0, 8.0) else bb
    nseg = int(math.ceil(end_beat / seg_len - 1e-9))
    order = np.argsort(on, kind="stable")
    on_s, du_s, pi_s = on[order], np.minimum(du[order], 16.0), pi[order]
    segs_per_bar = max(1, int(round(bb / seg_len)))
    segdata, rough_seg, off_seg = [], [], []
    for si in range(nseg):
        s_, e_ = si * seg_len, (si + 1) * seg_len
        lo_i = np.searchsorted(on_s, s_ - 16.0, "left")
        hi_i = np.searchsorted(on_s, e_, "left")
        o_, d_, p_ = on_s[lo_i:hi_i], du_s[lo_i:hi_i], pi_s[lo_i:hi_i]
        ov = np.minimum(o_ + d_, e_) - np.maximum(o_, s_)
        k = ov > 1e-6
        if not k.any():
            segdata.append(None)
            rough_seg.append(NAN)
            off_seg.append(NAN)
            continue
        ov, p_ = ov[k], p_[k]
        m = np.zeros(12)
        np.add.at(m, (p_ % 12).astype(int), ov)
        segdata.append((m, int(p_.min())))
        rough_seg.append(_pc_rough(list(m)))
        m_rel = np.roll(m, -tonic)
        off_seg.append(float(sum(m_rel[i] for i in range(12) if i not in scale)
                             / m.sum()))

    def _cls(parts):
        parts = [x for x in parts if x is not None]
        if not parts:
            return None
        m = sum(x[0] for x in parts)
        low = min(x[1] for x in parts)
        return _classify_chord(list(m), low % 12)

    chords_raw = []          # (pos_en_compases, raíz, calidad, puntuación)
    for b0 in range(0, nseg, segs_per_bar):
        grp = segdata[b0:b0 + segs_per_bar]
        pos = b0 * seg_len / bb
        whole = _cls(grp)
        if whole is None:
            chords_raw.append(None)
            continue
        if segs_per_bar == 2 and len(grp) == 2:
            h1, h2 = _cls(grp[:1]), _cls(grp[1:])
            # el compás se divide solo si no encaja en un único acorde y las
            # dos mitades sí encajan en acordes distintos
            if (whole[2] < 0.6 and h1 and h2 and h1[2] >= 0.5 and h2[2] >= 0.5
                    and (h1[0], h1[1]) != (h2[0], h2[1])):
                chords_raw.append((pos,) + h1)
                chords_raw.append((pos + 0.5,) + h2)
                continue
        chords_raw.append((pos,) + whole)
    seq = []
    for c in chords_raw:
        if c is None:
            continue
        if not seq or (seq[-1][1], seq[-1][2]) != (c[1], c[2]):
            seq.append(c)
    n_all = len([c for c in chords_raw if c is not None])
    F["har_chord_rate"] = len(seq) / float(nb)
    F["har_none"] = 1.0 - n_all / float(max(1, len(chords_raw)))
    qc = Counter(c[2] for c in seq)
    for q in QUALS:
        F["har_q_%s" % q] = qc.get(q, 0) / float(len(seq)) if seq else NAN
    rmc = Counter((b[1] - a[1]) % 12 for a, b in zip(seq, seq[1:]))
    for i in range(12):
        F["har_rm_%d" % i] = rmc.get(i, 0) / float(len(seq) - 1) \
            if len(seq) > 1 else NAN
    if seq:
        rel = [(c[1] - tonic) % 12 for c in seq]
        F["har_diat"] = sum(1 for d in rel if d in scale) / float(len(rel))
        F["har_chrom"] = 1.0 - F["har_diat"]
        F["har_tonic"] = sum(1 for d in rel if d == 0) / float(len(rel))
        F["har_dom"] = sum(1 for d in rel if d == 7) / float(len(rel))
        F["har_sub"] = sum(1 for d in rel if d == 5) / float(len(rel))
        F["har_final_tonic"] = 1.0 if rel[-1] == 0 else 0.0
        last2 = rel[-2:] if len(rel) >= 2 else []
        F["har_cad_auth"] = 1.0 if last2 == [7, 0] else 0.0
        F["har_cad_plag"] = 1.0 if last2 == [5, 0] else 0.0
        F["har_cad_half"] = 1.0 if rel[-1] == 7 else 0.0
        F["har_unique"] = len(set((c[1], c[2]) for c in seq)) / float(len(seq))
    else:
        for nm in ("har_diat", "har_chrom", "har_tonic", "har_dom", "har_sub",
                   "har_final_tonic", "har_cad_auth", "har_cad_plag",
                   "har_cad_half", "har_unique"):
            F[nm] = NAN
    rs = [x for x in rough_seg if not math.isnan(x)]
    F["har_rough"] = float(np.mean(rs)) if rs else NAN
    # bucle armónico (secuencia de acordes sin repeticiones consecutivas)
    keys = [(c[1], c[2]) for c in seq]
    loop_sc, loop_p = 0.0, 0
    if len(keys) >= 8:
        res = []
        for per in range(2, min(8, len(keys) // 2) + 1):
            mt = sum(1 for i in range(len(keys) - per) if keys[i] == keys[i + per])
            res.append((mt / float(len(keys) - per), per))
        top = max(r[0] for r in res)
        for sc, per in res:
            if sc >= top - 0.03:
                loop_sc, loop_p = sc, per
                break
        if loop_sc < 0.6:
            loop_sc, loop_p = 0.0, 0
    F["har_loop"] = loop_sc
    F["har_loop_len"] = float(loop_p)

    # ── series por compás (agnósticas) ───────────
    S = {k: [] for k in ("all_onsets", "all_pitch", "low_pitch", "high_pitch",
                         "span", "rough", "offscale", "vel")}
    bar_idx = np.minimum(nb - 1, (on / bb + 1e-9).astype(int))
    for b in range(nb):
        k = bar_idx == b
        if not k.any():
            for kk in S:
                S[kk].append(None)
            continue
        pp = pi[k]
        S["all_onsets"].append(len(np.unique(key48[k])) / bb)
        S["all_pitch"].append(float(pp.mean()))
        S["low_pitch"].append(float(pp.min()))
        S["high_pitch"].append(float(pp.max()))
        S["span"].append(float(pp.max() - pp.min()))
        S["vel"].append(float(ve[k].mean()))
        rr = [x for x in rough_seg[b * segs_per_bar:(b + 1) * segs_per_bar]
              if not math.isnan(x)]
        S["rough"].append(float(np.mean(rr)) if rr else None)
        oo = [x for x in off_seg[b * segs_per_bar:(b + 1) * segs_per_bar]
              if not math.isnan(x)]
        S["offscale"].append(float(np.mean(oo)) if oo else None)
    series = {}
    for k, v in S.items():
        f = gl._fill(v)
        if f is not None:
            series[k] = f

    # ── analizador de gestos (capa de trayectoria) ─
    an = gl.analyze_piece(p)
    for k, v in an["series"].items():
        if k in ("reg_mean", "reg_max", "reg_min", "bass_low", "density",
                 "key_r", "clash", "third_ratio"):
            series[k] = v
    gest = an["gestures"]
    best = {}
    for g in gest:
        sh = g["shape"] if g["shape"] in GST_SHAPES else "other"
        best[sh] = max(best.get(sh, 0.0), g["score"])
    for sh in GST_SHAPES:
        F["gst_%s" % sh] = best.get(sh, 0.0)
    F["gst_n"] = float(len(gest))
    sec_g = _section_gestures(p, gl, gest)
    sbest = {}
    for g in sec_g:
        sh = g["shape"] if g["shape"] in GST_SHAPES else "other"
        sbest[sh] = max(sbest.get(sh, 0.0), g["score"])
    for sh in GST_SHAPES:      # en piezas cortas la sección es la pieza entera
        F["sec_%s" % sh] = sbest.get(sh, 0.0) if p.nbars >= SEC_MIN_BARS \
            else best.get(sh, 0.0)
    F["sec_n"] = float(len(sec_g)) if p.nbars >= SEC_MIN_BARS \
        else float(len(gest))
    kc = an["categorical"].get("key_col")
    F["ton_changes"] = (sum(1 for a, b in zip(kc, kc[1:]) if a != b)
                        / float(max(1, len(kc) - 1))) if kc else NAN

    # ── trayectoria ───────────────────────────────
    for s in TRJ_SERIES:
        x = series.get(s)
        if x is None or len(x) < 4:
            for suf in ("std", "delta", "trend", "per", "lag", "reset"):
                F["trj_%s_%s" % (s, suf)] = NAN
            continue
        a = np.asarray(x, dtype=float)
        F["trj_%s_std" % s] = float(a.std())
        F["trj_%s_delta" % s] = float(np.abs(np.diff(a)).mean())
        sl = np.polyfit(np.arange(len(a)), a, 1)[0]
        F["trj_%s_trend" % s] = float(sl * len(a) / (a.std() + 1e-9)) \
            if a.std() > 1e-9 else 0.0
        pk, lg = _autocorr_peak(a)
        F["trj_%s_per" % s] = pk
        F["trj_%s_lag" % s] = float(lg)
        F["trj_%s_reset" % s] = _reset_rate(list(a))

    # ── ventanas de forma ─────────────────────────
    windows = []
    for s, min_amp in WIN_SERIES.items():
        x = series.get(s)
        if x is None:
            continue
        for L in WIN_LENGTHS:
            if len(x) < L:
                continue
            for st in range(0, len(x) - L + 1, max(1, L // 2)):
                w = x[st:st + L]
                v, amp = _shape_vec(w, gl)
                if amp >= min_amp:
                    windows.append((s, st, L, float(amp), v))

    chords = [[round(c[0], 3), int(c[1]), c[2]] for c in seq]
    deg1 = " ".join(roman((c[1] - tonic) % 12, c[2]) for c in seq)
    sec_t = None
    if sec:
        sec_t = gl._name_to_pc(sec["tonic"]) if hasattr(gl, "_name_to_pc") else None
    deg2 = " ".join(roman((c[1] - sec_t) % 12, c[2]) for c in seq) \
        if sec_t is not None else ""
    meta = dict(
        bars=nb, tempo=round(bpm_avg, 2), ts_num=p.time_signature[0],
        ts_den=p.time_signature[1], notes=len(N), voices=len(p.voices),
        duration_s=round(dur_s, 2), tonic=key["tonic"], mode=key["mode"],
        key_r=round(key["r"], 3),
        key2="%s %s" % (sec["tonic"], sec["mode"]) if sec else "",
        has_drums=1 if drums else 0, prog=deg1, prog2=deg2)
    top = [dict(g, scope="global")
           for g in sorted(gest, key=lambda g: -g["score"])[:60]]
    top += [dict(g, scope="section") for g in sec_g]
    return dict(meta=meta, feats=F, series=series, gestures=top,
                chords=chords, windows=windows)


# ══════════════════════════════════════════════════
#  BASE DE DATOS
# ══════════════════════════════════════════════════
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, name TEXT, sha1 TEXT, size INTEGER,
  mtime REAL, status TEXT, error TEXT, dup_of INTEGER, extractor TEXT,
  indexed_at REAL, bars INTEGER, tempo REAL, ts_num INTEGER, ts_den INTEGER,
  notes INTEGER, voices INTEGER, duration_s REAL, tonic TEXT, mode TEXT,
  key_r REAL, key2 TEXT, has_drums INTEGER, prog TEXT, prog2 TEXT);
CREATE INDEX IF NOT EXISTS ix_files_sha ON files(sha1);
CREATE TABLE IF NOT EXISTS feats(file_id INTEGER, name TEXT, value REAL);
CREATE INDEX IF NOT EXISTS ix_feats ON feats(name, value);
CREATE INDEX IF NOT EXISTS ix_feats_f ON feats(file_id);
CREATE TABLE IF NOT EXISTS series(file_id INTEGER PRIMARY KEY, data BLOB);
CREATE TABLE IF NOT EXISTS gestures(file_id INTEGER, shape TEXT, feature TEXT,
  score REAL, b0 INTEGER, b1 INTEGER, text TEXT, params TEXT,
  scope TEXT DEFAULT 'global');
CREATE TABLE IF NOT EXISTS models(name TEXT PRIMARY KEY, tag TEXT, neg TEXT,
  data TEXT, created REAL);
CREATE INDEX IF NOT EXISTS ix_gest ON gestures(shape, score);
CREATE INDEX IF NOT EXISTS ix_gest_f ON gestures(file_id);
CREATE TABLE IF NOT EXISTS chords(file_id INTEGER PRIMARY KEY, seq TEXT);
CREATE TABLE IF NOT EXISTS windows(file_id INTEGER, feature TEXT, start INTEGER,
  length INTEGER, amp REAL, vec BLOB);
CREATE INDEX IF NOT EXISTS ix_win ON windows(feature, length);
CREATE INDEX IF NOT EXISTS ix_win_f ON windows(file_id);
CREATE TABLE IF NOT EXISTS tags(tag TEXT, file_id INTEGER,
  PRIMARY KEY(tag, file_id));
"""

META_COLS = ["bars", "tempo", "ts_num", "ts_den", "notes", "voices",
             "duration_s", "tonic", "mode", "key_r", "key2", "has_drums",
             "prog", "prog2"]


def connect(path, create=False):
    if not create and not os.path.exists(path):
        sys.stderr.write("Error: no existe la base de datos %s "
                         "(créala con `index`)\n" % path)
        sys.exit(1)
    if create:
        _ensure_parent(path)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA journal_mode=WAL")
    except sqlite3.DatabaseError:
        pass
    con.executescript(SCHEMA)
    cols = [r[1] for r in con.execute("PRAGMA table_info(gestures)")]
    if "scope" not in cols:          # bases creadas con extractor 1.0
        con.execute("ALTER TABLE gestures ADD COLUMN scope TEXT DEFAULT 'global'")
    con.execute("INSERT OR REPLACE INTO meta VALUES('schema','2')")
    con.commit()
    return con


def _sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _clear(con, fid):
    for t in ("feats", "series", "gestures", "chords", "windows"):
        con.execute("DELETE FROM %s WHERE file_id=?" % t, (fid,))


def _upsert_file(con, path, st, sha, status, error=None, dup_of=None,
                 meta=None):
    row = con.execute("SELECT id FROM files WHERE path=?", (path,)).fetchone()
    cols = dict(path=path, name=os.path.basename(path), sha1=sha, size=st.st_size,
                mtime=st.st_mtime, status=status, error=error, dup_of=dup_of,
                extractor=EXTRACTOR, indexed_at=time.time())
    for c in META_COLS:
        cols[c] = (meta or {}).get(c)
    if row:
        fid = row["id"]
        _clear(con, fid)
        sets = ",".join("%s=?" % c for c in cols)
        con.execute("UPDATE files SET %s WHERE id=?" % sets,
                    list(cols.values()) + [fid])
    else:
        names = ",".join(cols)
        qs = ",".join("?" * len(cols))
        cur = con.execute("INSERT INTO files(%s) VALUES(%s)" % (names, qs),
                          list(cols.values()))
        fid = cur.lastrowid
    return fid


def store_result(con, path, st, sha, res):
    np = _np()
    fid = _upsert_file(con, path, st, sha, "ok", meta=res["meta"])
    con.executemany("INSERT INTO feats VALUES(?,?,?)",
                    [(fid, k, float(v)) for k, v in res["feats"].items()
                     if v is not None and not math.isnan(v) and not math.isinf(v)])
    ser = {k: [round(x, 3) for x in v] for k, v in res["series"].items()}
    con.execute("INSERT INTO series VALUES(?,?)",
                (fid, zlib.compress(json.dumps(ser).encode("utf-8"))))
    con.executemany(
        "INSERT INTO gestures VALUES(?,?,?,?,?,?,?,?,?)",
        [(fid, g["shape"], g["feature"], g["score"],
          g["bars"][0] if g.get("bars") else None,
          g["bars"][1] if g.get("bars") else None, g["text"],
          json.dumps(g["params"]), g.get("scope", "global"))
         for g in res["gestures"]])
    con.execute("INSERT INTO chords VALUES(?,?)",
                (fid, json.dumps(res["chords"])))
    con.executemany(
        "INSERT INTO windows VALUES(?,?,?,?,?,?)",
        [(fid, s, st_, L, amp, np.asarray(v, dtype=np.float32).tobytes())
         for (s, st_, L, amp, v) in res["windows"]])
    return fid


def _work(args):
    path, opts = args
    try:
        return path, extract(path, opts), None
    except Exception as e:  # noqa
        return path, None, "%s: %s" % (type(e).__name__, e)


# ══════════════════════════════════════════════════
#  CORPUS EN MEMORIA (matriz de rasgos, z-scores, sensaciones)
# ══════════════════════════════════════════════════
class Corpus(object):
    def __init__(self, con):
        np = _np()
        self.con = con
        rows = con.execute("SELECT * FROM files WHERE status='ok' "
                           "ORDER BY id").fetchall()
        self.files = {r["id"]: r for r in rows}
        self.ids = [r["id"] for r in rows]
        self.pos = {i: k for k, i in enumerate(self.ids)}
        data = con.execute(
            "SELECT x.file_id, x.name, x.value FROM feats x "
            "JOIN files f ON f.id=x.file_id WHERE f.status='ok'").fetchall()
        self.names = sorted(set(r[1] for r in data))
        self.col = {n: j for j, n in enumerate(self.names)}
        X = np.full((len(self.ids), len(self.names)), np.nan)
        for fid, nm, v in data:
            X[self.pos[fid], self.col[nm]] = v
        self.X = X
        if len(self.ids):
            with np.errstate(all="ignore"):
                self.mu = np.nanmean(X, axis=0)
                self.sd = np.nanstd(X, axis=0)
            self.mu = np.where(np.isnan(self.mu), 0.0, self.mu)
        else:
            self.mu = np.zeros(len(self.names))
            self.sd = np.ones(len(self.names))
        self.sd = np.where(np.isnan(self.sd) | (self.sd < 1e-6), 1.0, self.sd)
        self.Z = self.zvec(X)
        self.models = {}
        for r in con.execute("SELECT name, data FROM models"):
            self.models[r["name"]] = json.loads(r["data"])
        self.feel_raw = {k: self._score(k, X, self.Z)
                         for k in list(FEELS) + list(self.models)}

    def zvec(self, X):
        np = _np()
        Z = (X - self.mu) / self.sd
        Z = np.clip(Z, -4, 4)
        return np.nan_to_num(Z, nan=0.0)

    def vec_from_feats(self, feats):
        np = _np()
        x = np.full(len(self.names), np.nan)
        for k, v in feats.items():
            j = self.col.get(k)
            if j is not None and v is not None and not math.isnan(v):
                x[j] = v
        return x

    def _score(self, name, X, Z):
        """Puntuación cruda (n,) de una sensación o modelo para las filas X/Z."""
        np = _np()
        if name in FEELS:
            sc = np.zeros(Z.shape[0])
            for f, w in FEELS[name]:
                j = self.col.get(f)
                if j is not None:
                    sc = sc + w * Z[:, j]
            return sc
        m = self.models[name]
        sc = np.full(X.shape[0], float(m["b"]))
        for k, nm in enumerate(m["names"]):
            j = self.col.get(nm)
            if j is None:
                continue
            z = (X[:, j] - m["mu"][k]) / m["sd"][k]
            z = np.nan_to_num(np.clip(z, -4, 4), nan=0.0)
            sc = sc + m["w"][k] * z
        return sc

    def feel_pct(self, raw, name):
        """Percentil (0–1) de un valor crudo dentro del corpus."""
        np = _np()
        ref = self.feel_raw[name]
        if len(ref) < 2:
            return 0.5
        return float((ref < raw - 1e-12).sum() + 0.5 * (np.abs(ref - raw) <= 1e-12).sum()) / len(ref)

    def feels_of_row(self, k):
        return {n: self.feel_pct(float(self.feel_raw[n][k]), n)
                for n in self.feel_raw}

    def feels_of_vec(self, x):
        X = x.reshape(1, -1)
        Z = self.zvec(X)
        return {n: self.feel_pct(float(self._score(n, X, Z)[0]), n)
                for n in self.feel_raw}


def resolve_file(con, ref):
    """Fila de files por id, ruta o nombre (último que coincida)."""
    if ref.isdigit():
        r = con.execute("SELECT * FROM files WHERE id=?", (int(ref),)).fetchone()
        if r:
            return r
    for q, a in (("path=?", ref), ("path=?", os.path.abspath(ref)),
                 ("name=?", os.path.basename(ref))):
        r = con.execute("SELECT * FROM files WHERE %s ORDER BY id DESC" % q,
                        (a,)).fetchone()
        if r:
            return r
    return None


def _load_series(con, fid):
    r = con.execute("SELECT data FROM series WHERE file_id=?", (fid,)).fetchone()
    return json.loads(zlib.decompress(r[0]).decode("utf-8")) if r else {}


# ══════════════════════════════════════════════════
#  SALIDA
# ══════════════════════════════════════════════════
def _table(headers, rows, maxw=34):
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    rows = [[c if len(c) <= maxw else c[:maxw - 1] + "…" for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
         for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    print("  ".join("─" * w[i] for i in range(len(headers))))
    for r in rows:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))


def _keystr(r):
    return "%s %s" % (r["tonic"], "m" if r["mode"] == "minor" else "M") \
        if r["tonic"] else "?"


def _fmt(v, nd=2):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) \
        else ("%.*f" % (nd, v))


def _col_value(C, k, name):
    """Valor de un rasgo o sensación de la fila k del corpus."""
    if name in C.feel_raw:
        return C.feel_pct(float(C.feel_raw[name][k]), name)
    j = C.col.get(name)
    return None if j is None else C.X[k, j]


# ══════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════
def cmd_index(args):
    gl = _gl()
    files = []
    for f in gl.collect_midis(args.paths):
        if os.path.isfile(f):
            files.append(os.path.abspath(f))
        else:
            sys.stderr.write("aviso: no existe %s\n" % f)
    files = sorted(set(files))
    if not files:
        sys.stderr.write("Error: no hay MIDIs en las rutas indicadas\n")
        return 1
    con = connect(args.db, create=True)
    t0 = time.time()
    seen = {}
    for r in con.execute("SELECT id, sha1 FROM files WHERE status='ok'"):
        seen[r["sha1"]] = r["id"]
    exist = {r["path"]: r for r in con.execute("SELECT * FROM files")}
    todo, skipped, dups, bad = [], 0, 0, 0
    for f in files:
        try:
            st = os.stat(f)
        except OSError as e:
            sys.stderr.write("aviso: %s (%s)\n" % (f, e))
            continue
        r = exist.get(f)
        if (r and not args.force and r["size"] == st.st_size and
                abs((r["mtime"] or 0) - st.st_mtime) < 1e-3 and
                r["extractor"] == EXTRACTOR):
            skipped += 1
            continue
        try:
            sha = _sha1(f)
        except OSError as e:
            sys.stderr.write("aviso: %s (%s)\n" % (f, e))
            continue
        if (r and not args.force and r["sha1"] == sha and
                r["extractor"] == EXTRACTOR):
            con.execute("UPDATE files SET mtime=?, size=? WHERE id=?",
                        (st.st_mtime, st.st_size, r["id"]))
            skipped += 1
            continue
        other = seen.get(sha)
        if other is not None and (not r or r["id"] != other):
            _upsert_file(con, f, st, sha, "dup", dup_of=other)
            dups += 1
            continue
        todo.append((f, st, sha))
        seen[sha] = -1
    con.commit()
    print("%d archivos: %d sin cambios, %d duplicados, %d a procesar"
          % (len(files), skipped, dups, len(todo)))
    opts = dict(min_notes=args.min_notes, max_notes=args.max_notes)
    jobs = max(1, args.jobs)
    ok = 0
    t_last = time.time()

    def handle(path, st, sha, res, err):
        nonlocal ok, bad, t_last
        if err:
            _upsert_file(con, path, st, sha, "error", error=err)
            bad += 1
            sys.stderr.write("  error: %s → %s\n" % (os.path.basename(path), err))
        else:
            fid = store_result(con, path, st, sha, res)
            seen[sha] = fid
            ok += 1
        if (ok + bad) % 25 == 0:
            con.commit()
        if time.time() - t_last > 5:
            t_last = time.time()
            print("  %d/%d …" % (ok + bad, len(todo)))

    info = {f: (st, sha) for f, st, sha in todo}
    if jobs == 1 or len(todo) < 2:
        for f, st, sha in todo:
            path, res, err = _work((f, opts))
            handle(path, st, sha, res, err)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            step = jobs * 8
            for i in range(0, len(todo), step):
                chunk = [(f, opts) for f, _, _ in todo[i:i + step]]
                for path, res, err in ex.map(_work, chunk):
                    st, sha = info[path]
                    handle(path, st, sha, res, err)
    if args.prune:
        gone = [r["id"] for r in con.execute("SELECT id, path FROM files")
                if not os.path.exists(r["path"])]
        for fid in gone:
            _clear(con, fid)
            con.execute("DELETE FROM tags WHERE file_id=?", (fid,))
            con.execute("DELETE FROM files WHERE id=?", (fid,))
        if gone:
            print("eliminados %d archivos que ya no existen" % len(gone))
    con.execute("INSERT OR REPLACE INTO meta VALUES('extractor',?)", (EXTRACTOR,))
    con.commit()
    print("listo: %d indexados, %d con error, %.1f s" % (ok, bad, time.time() - t0))
    return 0


def cmd_info(args):
    con = connect(args.db)
    n = lambda q: con.execute(q).fetchone()[0]
    print("Base: %s (%.1f MB)" % (args.db, os.path.getsize(args.db) / 1e6))
    print("Archivos: %d  ok=%d  duplicados=%d  con error=%d"
          % (n("SELECT COUNT(*) FROM files"),
             n("SELECT COUNT(*) FROM files WHERE status='ok'"),
             n("SELECT COUNT(*) FROM files WHERE status='dup'"),
             n("SELECT COUNT(*) FROM files WHERE status='error'")))
    old = n("SELECT COUNT(*) FROM files WHERE extractor!='%s'" % EXTRACTOR)
    if old:
        print("  %d archivos con extractor antiguo: ejecuta `index` para recalcular"
              % old)
    if n("SELECT COUNT(*) FROM files WHERE status='ok'") == 0:
        return 0
    print("Rasgos por archivo: %.0f | gestos: %d | ventanas: %d | acordes: %d"
          % (n("SELECT COUNT(*)*1.0/COUNT(DISTINCT file_id) FROM feats"),
             n("SELECT COUNT(*) FROM gestures"),
             n("SELECT COUNT(*) FROM windows"),
             n("SELECT COUNT(*) FROM chords")))
    print("\nTonalidad (modo): " + ", ".join(
        "%s=%d" % (r[0], r[1]) for r in con.execute(
            "SELECT mode, COUNT(*) FROM files WHERE status='ok' GROUP BY mode")))
    print("Compás: " + ", ".join(
        "%d/%d=%d" % (r[0], r[1], r[2]) for r in con.execute(
            "SELECT ts_num, ts_den, COUNT(*) FROM files WHERE status='ok' "
            "GROUP BY ts_num, ts_den ORDER BY COUNT(*) DESC LIMIT 6")))
    q = con.execute("SELECT MIN(tempo), AVG(tempo), MAX(tempo), MIN(bars), "
                    "AVG(bars), MAX(bars) FROM files WHERE status='ok'").fetchone()
    print("Tempo (min/med/max): %.0f / %.0f / %.0f   Compases: %d / %.0f / %d"
          % tuple(q))
    print("Con batería: %d" % n("SELECT COUNT(*) FROM files WHERE has_drums=1"))
    print("\nGestos más frecuentes (score ≥ 0.6):")
    for r in con.execute("SELECT shape, COUNT(DISTINCT file_id) c FROM gestures "
                         "WHERE score>=0.6 GROUP BY shape ORDER BY c DESC LIMIT 8"):
        print("  %-14s en %d archivos" % (r[0], r[1]))
    tg = con.execute("SELECT tag, COUNT(*) FROM tags GROUP BY tag").fetchall()
    if tg:
        print("\nEtiquetas: " + ", ".join("%s(%d)" % (r[0], r[1]) for r in tg))
    errs = con.execute("SELECT path, error FROM files WHERE status='error' "
                       "ORDER BY id").fetchall()
    if errs:
        print("\nErrores%s:" % ("" if args.errors else " (usa --errors para todos)"))
        for r in errs[:None if args.errors else 5]:
            print("  %s → %s" % (os.path.basename(r["path"]), r["error"]))
    return 0


def _feat_dict_for(con, ref, args_note=True):
    """(fila|None, feats, meta, chords, series) para una referencia que puede
    estar indexada o ser un MIDI nuevo (se extrae al vuelo)."""
    r = resolve_file(con, ref)
    if r is not None and r["status"] == "ok":
        feats = {x[0]: x[1] for x in con.execute(
            "SELECT name, value FROM feats WHERE file_id=?", (r["id"],))}
        ch = con.execute("SELECT seq FROM chords WHERE file_id=?",
                         (r["id"],)).fetchone()
        return r, feats, json.loads(ch[0]) if ch else [], _load_series(con, r["id"])
    if os.path.isfile(ref):
        res = extract(ref)
        return None, res["feats"], res["chords"], res["series"], res
    return None


def cmd_show(args):
    con = connect(args.db)
    got = _feat_dict_for(con, args.file)
    if got is None:
        sys.stderr.write("Error: %s no está indexado ni existe como archivo\n"
                         % args.file)
        return 1
    r, feats, chords, series = got[:4]
    res = got[4] if len(got) > 4 else None
    C = Corpus(con)
    x = C.vec_from_feats(feats)
    feels = C.feels_of_vec(x) if len(C.ids) else {}
    meta = dict(r) if r is not None else res["meta"]
    gest = ([dict(g) for g in con.execute(
        "SELECT * FROM gestures WHERE file_id=? ORDER BY score DESC LIMIT 12",
        (r["id"],))] if r is not None else
        [dict(shape=g["shape"], feature=g["feature"], score=g["score"],
              text=g["text"]) for g in res["gestures"][:12]])
    if args.json:
        print(json.dumps(dict(meta={k: meta[k] for k in META_COLS},
                              feelings={k: _fnum(v) for k, v in feels.items()},
                              features={k: _fnum(v) for k, v in feats.items()},
                              gestures=gest, chords=chords), indent=1,
                         ensure_ascii=False))
        return 0
    gl = _gl()
    print("%s%s" % (meta.get("name") or os.path.basename(args.file),
                    "" if r is not None else "  (no indexado: extraído al vuelo)"))
    print("  %s compases, %s/%s, %.0f bpm, %s notas, %s voces, clave %s %s "
          "(r=%.2f; 2.ª: %s)" % (
              meta["bars"], meta["ts_num"], meta["ts_den"], meta["tempo"],
              meta["notes"], meta["voices"], meta["tonic"], meta["mode"],
              meta["key_r"], meta["key2"] or "—"))
    if feels:
        small = "  (corpus pequeño: percentiles poco fiables)" if len(C.ids) < 8 else ""
        print("\nSensaciones (percentil en el corpus)%s" % small)
        for k, v in feels.items():
            print("  %-11s %s %3d%%" % (k, "█" * int(v * 20) + "·" * (20 - int(v * 20)),
                                       round(v * 100)))
    print("\nProgresión (grados, clave 1): %s" % (meta["prog"] or "—"))
    if meta["prog2"]:
        print("Progresión (grados, clave 2): %s" % meta["prog2"])
    if gest:
        print("\nGestos:")
        for g in gest:
            print("  %.2f  %s" % (g["score"], g["text"]))
    if series:
        print("\nSeries:")
        for k in sorted(series):
            v = series[k]
            print("  %-11s %s  [%s … %s]" % (k, gl._spark(v[:60]), _fmt(min(v)),
                                             _fmt(max(v))))
    if args.features:
        print("\nRasgos:")
        for g in GROUPS.values():
            items = [(k, v) for k, v in sorted(feats.items()) if group_of(k) == g]
            if items:
                print("  [%s]" % g)
                for k, v in items:
                    print("    %-26s %s" % (k, _fmt(v, 3)))
    return 0


def _weights(C, args):
    np = _np()
    sel = set(s.strip() for s in args.groups.split(",")) if args.groups else None
    ex = set(s.strip() for s in args.exclude.split(",")) if args.exclude else set()
    if args.invariant:
        ex.add("register")
    gw = {g: 1.0 for g in GROUPS.values()}
    if args.weights:
        for kv in args.weights.split(","):
            k, v = kv.split("=")
            gw[k.strip()] = float(v)
    cols = {}
    for j, nm in enumerate(C.names):
        g = group_of(nm)
        if (sel is not None and g not in sel) or g in ex or g == "other":
            continue
        cols.setdefault(g, []).append(j)
    w = np.zeros(len(C.names))
    for g, js in cols.items():
        for j in js:
            w[j] = gw.get(g, 1.0) / math.sqrt(len(js))
    return w, cols


def cmd_similar(args):
    np = _np()
    con = connect(args.db)
    C = Corpus(con)
    if not C.ids:
        sys.stderr.write("Error: el índice no tiene piezas\n")
        return 1
    exclude = set()
    if args.tag:
        tagged = [r[0] for r in con.execute(
            "SELECT file_id FROM tags WHERE tag=?", (args.tag,))]
        tagged = [i for i in tagged if i in C.pos]
        if not tagged:
            sys.stderr.write("Error: la etiqueta %r no tiene piezas\n" % args.tag)
            return 1
        zq = np.mean([C.Z[C.pos[i]] for i in tagged], axis=0)
        title = "etiqueta %s (%d piezas)" % (args.tag, len(tagged))
        if not args.include_tagged:
            exclude = set(tagged)
    else:
        if not args.file:
            sys.stderr.write("Error: indica ARCHIVO o --tag\n")
            return 2
        got = _feat_dict_for(con, args.file)
        if got is None:
            sys.stderr.write("Error: %s no está indexado ni existe\n" % args.file)
            return 1
        r = got[0]
        zq = C.zvec(C.vec_from_feats(got[1]))
        title = args.file
        if r is not None:
            exclude.add(r["id"])
    w, cols = _weights(C, args)
    if not w.any():
        sys.stderr.write("Error: ningún grupo seleccionado (grupos: %s)\n"
                         % ", ".join(sorted(set(GROUPS.values()))))
        return 2
    D = C.Z - zq
    dist = np.sqrt(((D * w) ** 2).sum(axis=1) / (w ** 2).sum())
    order = np.argsort(dist)
    print("Más parecidas a %s  [grupos: %s]" % (title, ", ".join(sorted(cols))))
    rows = []
    for k in order:
        fid = C.ids[k]
        if fid in exclude:
            continue
        f = C.files[fid]
        gd = {g: math.sqrt(float((D[k, js] ** 2).mean())) for g, js in cols.items()}
        close = ", ".join(g for g, _ in sorted(gd.items(), key=lambda t: t[1])[:3])
        rows.append([fid, f["name"], f["bars"], _keystr(f), "%.0f" % f["tempo"],
                     "%.3f" % (1.0 / (1.0 + dist[k])), close])
        if len(rows) >= args.n:
            break
    _table(["id", "archivo", "comp", "clave", "bpm", "similitud",
            "coinciden en"], rows)
    return 0


def _parse_cmp(s):
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*(>=|<=|>|<|=)\s*(-?[0-9.]+)\s*$", s)
    if not m:
        raise ValueError("expresión inválida %r (use nombre>valor)" % s)
    return m.group(1), m.group(2), float(m.group(3))


def _cmp(a, op, b):
    return {">": a > b, "<": a < b, ">=": a >= b, "<=": a <= b,
            "=": abs(a - b) < 1e-9}[op]


def _prog_match(chords, qry, strict, tonics=None):
    """Posiciones donde la subsecuencia coincide por movimiento de raíz.
    Con `tonics` (lista de clases de altura) exige además que el primer acorde
    caiga en el grado pedido respecto a alguna de esas tónicas."""
    n, m = len(chords), len(qry)
    out = []
    for i in range(n - m + 1):
        ok = True
        if tonics is not None and not any(
                (chords[i][1] - t) % 12 == qry[0][0] for t in tonics):
            continue
        for j in range(1, m):
            d_file = (chords[i + j][1] - chords[i + j - 1][1]) % 12
            d_q = (qry[j][0] - qry[j - 1][0]) % 12
            if d_file != d_q:
                ok = False
                break
        if ok and strict:
            ok = all(qclass(chords[i + j][2]) == qry[j][1] for j in range(m))
        if ok:
            out.append(i)
    return out


def cmd_find(args):
    gl = _gl()
    con = connect(args.db)
    C = Corpus(con)
    if not C.ids:
        sys.stderr.write("Error: el índice no tiene piezas\n")
        return 1
    ids = set(C.ids)
    notes = {}
    try:
        for rg in args.range or []:
            parts = rg.split(":")
            if len(parts) != 3:
                raise ValueError("--range espera nombre:min:max (use '-' como "
                                 "abierto)")
            nm, lo, hi = parts
            lo = -1e18 if lo in ("", "-") else float(lo)
            hi = 1e18 if hi in ("", "-") else float(hi)
            if nm not in C.col:
                raise ValueError("rasgo desconocido %r" % nm)
            ids &= set(r[0] for r in con.execute(
                "SELECT file_id FROM feats WHERE name=? AND value BETWEEN ? AND ?",
                (nm, lo, hi)))
        for gs in args.gesture or []:
            m = re.match(r"^([A-Za-z_:]+?)(?::([A-Za-z_]+))?(?:>=?([0-9.]+))?$",
                         gs.replace(" ", ""))
            if not m:
                raise ValueError("--gesture inválido %r (forma[:rasgo][>=score])" % gs)
            sh, ft, sc = m.group(1), m.group(2), float(m.group(3) or 0.5)
            q = ("SELECT file_id, score, b0, b1, scope FROM gestures "
                 "WHERE shape=? AND score>=?")
            a = [sh, sc]
            if ft:
                q += " AND feature=?"
                a.append(ft)
            got, where = {}, {}
            for r in con.execute(q + " ORDER BY score, (scope='global')", a):
                got[r[0]] = r[1]
                where[r[0]] = (" c.%d-%d" % (r[2], r[3])) if r[2] else ""
            if not got:
                known = sorted(r[0] for r in con.execute(
                    "SELECT DISTINCT shape FROM gestures"))
                sys.stderr.write("aviso: ningún gesto %r (formas: %s)\n"
                                 % (sh, ", ".join(known)))
            ids &= set(got)
            for k, v in got.items():
                notes.setdefault(k, []).append("%s %.2f%s" % (sh, v, where[k]))
        for fe in args.feel or []:
            nm, op, val = _parse_cmp(fe)
            if nm not in C.feel_raw:
                raise ValueError("sensación desconocida %r (%s)"
                                 % (nm, ", ".join(C.feel_raw)))
            ids &= set(C.ids[k] for k in range(len(C.ids))
                       if _cmp(C.feel_pct(float(C.feel_raw[nm][k]), nm), op, val))
        if args.tag:
            ids &= set(r[0] for r in con.execute(
                "SELECT file_id FROM tags WHERE tag=?", (args.tag,)))
        if args.text:
            ids &= set(r[0] for r in con.execute(
                "SELECT id FROM files WHERE path LIKE ?", ("%" + args.text + "%",)))
        if args.progression:
            toks = args.progression.split()
            qry = []
            for t in toks:
                pc = parse_chord_token(t)
                if pc is None:
                    raise ValueError("acorde inválido %r en --progression "
                                     "(ej.: i bVI bIII bVII)" % t)
                qry.append(pc)
            if len(qry) < 2:
                raise ValueError("--progression necesita al menos 2 acordes")
            hit = set()
            for r in con.execute("SELECT file_id, seq FROM chords"):
                if r[0] not in ids:
                    continue
                tonics = None
                if args.anchor:
                    fr = C.files[r[0]]
                    tonics = [gl._name_to_pc(fr["tonic"])]
                    if fr["key2"]:
                        tonics.append(gl._name_to_pc(fr["key2"].split()[0]))
                pos = _prog_match(json.loads(r[1]), qry, args.strict_quality,
                                  tonics)
                if pos:
                    hit.add(r[0])
                    notes.setdefault(r[0], []).append(
                        "prog en c.%d" % (json.loads(r[1])[pos[0]][0] + 1))
            ids &= hit
    except ValueError as e:
        sys.stderr.write("Error: %s\n" % e)
        return 2
    sel = [k for k in range(len(C.ids)) if C.ids[k] in ids]
    if args.sort:
        sel.sort(key=lambda k: (_col_value(C, k, args.sort) is None,
                                -(_col_value(C, k, args.sort) or 0)
                                if args.desc else (_col_value(C, k, args.sort) or 0)))
    show = [s for s in (args.show or "").split(",") if s]
    rows = []
    for k in sel[:args.n]:
        f = C.files[C.ids[k]]
        rows.append([C.ids[k], f["name"], f["bars"], _keystr(f),
                     "%.0f" % f["tempo"]] +
                    [_fmt(_col_value(C, k, s)) for s in show] +
                    ["; ".join(notes.get(C.ids[k], []))])
    print("%d coincidencias de %d piezas%s" % (len(sel), len(C.ids),
                                               "" if len(sel) <= args.n else
                                               " (mostrando %d)" % args.n))
    if rows:
        _table(["id", "archivo", "comp", "clave", "bpm"] + show + ["detalle"], rows)
    return 0


def cmd_shape(args):
    np = _np()
    gl = _gl()
    con = connect(args.db)
    if args.values:
        try:
            win = [float(v) for v in args.values.split(",")]
        except ValueError:
            sys.stderr.write("Error: --values espera números separados por comas\n")
            return 2
        feat = args.feature or "reg_mean"
    else:
        if not args.from_file:
            sys.stderr.write("Error: indica --from ARCHIVO --feature F [--bars a-b] "
                             "o --values ...\n")
            return 2
        feat = args.feature
        if not feat:
            sys.stderr.write("Error: falta --feature\n")
            return 2
        got = _feat_dict_for(con, args.from_file)
        if got is None:
            sys.stderr.write("Error: %s no existe\n" % args.from_file)
            return 1
        ser = got[3]
        if feat not in ser:
            sys.stderr.write("Error: la serie %r no existe (disponibles: %s)\n"
                             % (feat, ", ".join(sorted(ser))))
            return 1
        win = ser[feat]
        if args.bars:
            m = re.match(r"^(\d+)-(\d+)$", args.bars)
            if not m or int(m.group(1)) < 1 or int(m.group(2)) < int(m.group(1)):
                sys.stderr.write("Error: --bars espera a-b (1 ≤ a ≤ b)\n")
                return 2
            win = win[int(m.group(1)) - 1:int(m.group(2))]
    if len(win) < 3:
        sys.stderr.write("Error: la ventana necesita al menos 3 valores\n")
        return 2
    qv, qamp = gl.shape_vector(win)
    if qamp < 1e-9:
        sys.stderr.write("Error: la ventana es plana: no define una forma\n")
        return 1
    q = np.array(qv)
    print("Forma de consulta (%s, %d compases, amplitud %.2f): %s"
          % (feat, len(win), qamp, gl._spark(list(q))))
    sql = ("SELECT w.file_id, w.start, w.length, w.amp, w.vec FROM windows w "
           "JOIN files f ON f.id=w.file_id WHERE f.status='ok' AND w.feature=?")
    a = [feat]
    if args.length:
        sql += " AND w.length=?"
        a.append(args.length)
    best = {}
    allr = []
    rws = con.execute(sql, a).fetchall()
    if rws:
        V = np.frombuffer(b"".join(r[4] for r in rws),
                          dtype=np.float32).reshape(len(rws), -1).astype(float)
        sims = np.maximum(0.0, 1.0 - np.sqrt(((V - q) ** 2).mean(axis=1)) / 0.5)
        for r, v, sim in zip(rws, V, sims):
            if r[3] < args.min_amp or sim < args.min_sim:
                continue
            if args.amp_tol and not (qamp * args.amp_tol <= r[3] <= qamp / args.amp_tol):
                continue
            if args.sign and ((v[-1] - v[0]) * (q[-1] - q[0]) < 0):
                continue
            allr.append((float(sim), r[0], r[1], r[2], r[3]))
    allr.sort(key=lambda t: -t[0])
    rows = []
    for sim, fid, st, L, amp in allr:
        if args.per_file and best.get(fid, 0) >= args.per_file:
            continue
        best[fid] = best.get(fid, 0) + 1
        f = con.execute("SELECT name, tonic, mode FROM files WHERE id=?",
                        (fid,)).fetchone()
        rows.append([fid, f["name"], "%d-%d" % (st + 1, st + L), "%.2f" % sim,
                     "%.1f" % amp])
        if len(rows) >= args.n:
            break
    if not rows:
        print("Sin ventanas parecidas (baja --min-sim o amplía --length).")
        return 0
    _table(["id", "archivo", "compases", "similitud", "amplitud"], rows)
    return 0


def cmd_tag(args):
    con = connect(args.db)
    if args.action == "ls":
        q = "SELECT t.tag, f.id, f.name FROM tags t JOIN files f ON f.id=t.file_id"
        a = []
        if args.tag:
            q += " WHERE t.tag=?"
            a.append(args.tag)
        rows = con.execute(q + " ORDER BY t.tag, f.id", a).fetchall()
        if not rows:
            print("(sin etiquetas)")
        for r in rows:
            print("%-14s %5d  %s" % (r[0], r[1], r[2]))
        return 0
    if not args.tag:
        sys.stderr.write("Error: falta el nombre de la etiqueta\n")
        return 2
    n = 0
    if args.action == "rm" and not args.files:
        n = con.execute("DELETE FROM tags WHERE tag=?", (args.tag,)).rowcount
    else:
        for ref in args.files:
            r = resolve_file(con, ref)
            if r is None or r["status"] != "ok":
                sys.stderr.write("aviso: %s no está indexado como pieza válida\n" % ref)
                continue
            if args.action == "add":
                con.execute("INSERT OR IGNORE INTO tags VALUES(?,?)",
                            (args.tag, r["id"]))
            else:
                con.execute("DELETE FROM tags WHERE tag=? AND file_id=?",
                            (args.tag, r["id"]))
            n += 1
    con.commit()
    print("%s: %d piezas %s" % (args.tag, n, "añadidas" if args.action == "add"
                                else "quitadas"))
    return 0


def _auc(pos, neg):
    """Área bajo la curva ROC por rangos (empates con rango medio)."""
    np = _np()
    if len(pos) == 0 or len(neg) == 0:
        return NAN
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    r = np.arange(1, len(v) + 1, dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        r[i:j + 1] = (i + j + 2) / 2.0
        i = j + 1
    ranks = np.empty(len(v))
    ranks[order] = r
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _fit_logistic(X, y, sw, lam, iters=40):
    """Regresión logística L2 ponderada (Newton). Devuelve (w, b)."""
    np = _np()
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    R = np.eye(d + 1) * lam
    R[d, d] = 0.0
    wb = np.zeros(d + 1)
    for _ in range(iters):
        z = np.clip(Xb @ wb, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = Xb.T @ (sw * (y - p)) - R @ wb
        H = (Xb.T * (sw * p * (1 - p) + 1e-6)) @ Xb + R
        step = np.linalg.solve(H, g)
        wb = wb + step
        if np.abs(step).max() < 1e-7:
            break
    return wb[:d], float(wb[d])


def _balanced(y):
    np = _np()
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return np.where(y > 0.5, len(y) / (2.0 * n1), len(y) / (2.0 * n0))


def _print_model(m, top):
    order = sorted(range(len(m["names"])), key=lambda k: -abs(m["w"][k]))[:top]
    rows = []
    for k in order:
        if abs(m["w"][k]) < 1e-6:
            continue
        rows.append(["%+.2f" % m["w"][k], m["names"][k], group_of(m["names"][k]),
                     _fmt(m["pos_mean"][k], 3), _fmt(m["rest_mean"][k], 3)])
    _table(["peso", "rasgo", "grupo", "media_pos", "media_resto"], rows, maxw=30)


def cmd_calibrate(args):
    np = _np()
    con = connect(args.db)
    if args.list:
        rows = con.execute("SELECT name, tag, neg, data, created FROM models "
                           "ORDER BY created").fetchall()
        if not rows:
            print("(sin modelos; entrena uno con `calibrate TAG`)")
        for r in rows:
            m = json.loads(r["data"])
            print("%-14s etiqueta=%-12s neg=%-10s pos=%d neg=%d AUC_cv=%s  %s" % (
                r["name"], r["tag"], r["neg"] or "(resto)", m["n_pos"], m["n_neg"],
                _fmt(m["cv_auc"]), time.strftime("%Y-%m-%d", time.localtime(r["created"]))))
        return 0
    if args.show or args.delete:
        nm = args.show or args.delete
        r = con.execute("SELECT data FROM models WHERE name=?", (nm,)).fetchone()
        if r is None:
            sys.stderr.write("Error: no existe el modelo %r\n" % nm)
            return 1
        if args.delete:
            con.execute("DELETE FROM models WHERE name=?", (nm,))
            con.commit()
            print("modelo %s eliminado" % nm)
            return 0
        _print_model(json.loads(r["data"]), args.top)
        return 0
    if not args.tag:
        sys.stderr.write("Error: indica la ETIQUETA positiva (o --list/--show/--delete)\n")
        return 2
    name = args.name or args.tag
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name) or name in FEELS:
        sys.stderr.write("Error: nombre de modelo inválido o reservado: %r\n" % name)
        return 2
    C = Corpus(con)
    tagged = lambda t: [C.pos[r[0]] for r in con.execute(
        "SELECT file_id FROM tags WHERE tag=?", (t,)) if r[0] in C.pos]
    ip = tagged(args.tag)
    if len(ip) < 3:
        sys.stderr.write("Error: la etiqueta %r tiene %d piezas indexadas; hacen "
                         "falta al menos 3\n" % (args.tag, len(ip)))
        return 1
    if args.neg:
        ineg = tagged(args.neg)
    else:
        sp = set(ip)
        ineg = [k for k in range(len(C.ids)) if k not in sp]
    if len(ineg) < 3:
        sys.stderr.write("Error: hay solo %d piezas negativas\n" % len(ineg))
        return 1
    sel = set(x.strip() for x in args.groups.split(",")) if args.groups else None
    ex = set(x.strip() for x in args.exclude.split(",")) if args.exclude else set()
    cols = [j for j, nm in enumerate(C.names)
            if group_of(nm) != "other" and (sel is None or group_of(nm) in sel)
            and group_of(nm) not in ex]
    if not cols:
        sys.stderr.write("Error: ningún grupo seleccionado\n")
        return 2
    idx = np.array(ip + ineg)
    y = np.array([1.0] * len(ip) + [0.0] * len(ineg))
    Xa = C.Z[idx][:, cols]
    w, b = _fit_logistic(Xa, y, _balanced(y), args.lam)
    # validación cruzada (k pliegues sobre positivas y negativas)
    k = max(2, min(5, len(ip)))
    rs = np.random.RandomState(0)
    fold = np.zeros(len(y), dtype=int)
    for grp in (np.where(y > 0.5)[0], np.where(y < 0.5)[0]):
        perm = rs.permutation(grp)
        fold[perm] = np.arange(len(perm)) % k
    aucs = []
    for f in range(k):
        tr, te = fold != f, fold == f
        if y[tr].sum() < 1 or y[te].sum() < 1 or (y[te] < 0.5).sum() < 1:
            continue
        wf, bf = _fit_logistic(Xa[tr], y[tr], _balanced(y[tr]), args.lam)
        sc = Xa[te] @ wf + bf
        aucs.append(_auc(sc[y[te] > 0.5], sc[y[te] < 0.5]))
    cv = float(np.mean(aucs)) if aucs else NAN
    pm = np.nanmean(C.X[ip][:, cols], axis=0)
    rm = np.nanmean(C.X[ineg][:, cols], axis=0)
    model = dict(names=[C.names[j] for j in cols], w=[float(v) for v in w], b=b,
                 mu=[float(v) for v in C.mu[cols]], sd=[float(v) for v in C.sd[cols]],
                 n_pos=len(ip), n_neg=len(ineg), cv_auc=_fnum(cv), lam=args.lam,
                 groups=sorted(set(group_of(C.names[j]) for j in cols)),
                 pos_mean=[_fnum(v) for v in pm], rest_mean=[_fnum(v) for v in rm])
    con.execute("INSERT OR REPLACE INTO models VALUES(?,?,?,?,?)",
                (name, args.tag, args.neg, json.dumps(model), time.time()))
    con.commit()
    print("Modelo %r: %d positivas (%s) frente a %d %s; %d rasgos; λ=%g"
          % (name, len(ip), args.tag, len(ineg),
             "de %s" % args.neg if args.neg else "del resto del corpus (incluye sin etiquetar)",
             len(cols), args.lam))
    print("AUC en validación cruzada (%d pliegues): %s" % (len(aucs), _fmt(cv)))
    if len(ip) < 8:
        print("aviso: con menos de 8 positivas el AUC es solo orientativo")
    if not math.isnan(cv) and cv < 0.7:
        print("aviso: el modelo no separa bien; revisa las etiquetas, usa --groups "
              "para acotar o sube --lam")
    print("\nRasgos que más pesan:")
    _print_model(model, args.top)
    allsc = C.Z[:, cols] @ w + b
    excl = set(ip) | (set(ineg) if args.neg else set())
    cand = [k2 for k2 in np.argsort(-allsc) if k2 not in excl][:args.candidates]
    if cand:
        print("\nCandidatas a revisar (no etiquetadas, mejor puntuadas):")
        _table(["id", "archivo", "prob"],
               [[C.ids[k2], C.files[C.ids[k2]]["name"],
                 "%.2f" % (1.0 / (1.0 + math.exp(-float(allsc[k2]))))] for k2 in cand])
    print("\nÚsalo con: find --feel \"%s>0.8\"   (percentil dentro del corpus)" % name)
    return 0


def cmd_export(args):
    con = connect(args.db)
    C = Corpus(con)
    _ensure_parent(args.out)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "path"] + META_COLS[:-2] + C.names + list(C.feel_raw))
        for k, fid in enumerate(C.ids):
            f = C.files[fid]
            w.writerow([fid, f["path"]] + [f[c] for c in META_COLS[:-2]] +
                       [_fnum(v) if not math.isnan(v) else "" for v in C.X[k]] +
                       [_fnum(C.feel_pct(float(C.feel_raw[n][k]), n))
                        for n in C.feel_raw])
    print("exportadas %d filas × %d rasgos → %s" % (len(C.ids), len(C.names),
                                                     args.out))
    return 0


# ══════════════════════════════════════════════════
#  AUTOTEST
# ══════════════════════════════════════════════════
def _mk_midi(path, notes, ts=(4, 4), tempo=120, drums=False):
    import mido
    tpb = 480
    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("time_signature", numerator=ts[0], denominator=ts[1]))
    tr.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
    ev = []
    for on, dur, p, v, ch in notes:
        ev.append((int(on * tpb), 1, p, v, ch))
        ev.append((int((on + dur) * tpb), 0, p, 0, ch))
    ev.sort(key=lambda e: (e[0], e[1]))
    t = 0
    for tk, kind, p, v, ch in ev:
        tr.append(mido.Message("note_on", note=p, velocity=v, channel=ch, time=tk - t))
        t = tk
    mid.tracks.append(tr)
    mid.save(path)


def cmd_selftest(args):
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="midi_index_")
    res = []

    def check(name, cond):
        res.append((name, bool(cond)))
        print("  %s %s" % ("OK " if cond else "FALLA", name))

    try:
        # 1: I IV V I (do mayor) con bajo y melodía, 4/4, 16 compases
        prog = [(0, [0, 4, 7]), (5, [0, 4, 7]), (7, [0, 4, 7]), (0, [0, 4, 7])] * 4
        n1 = []
        for i, (r, tr) in enumerate(prog):
            n1.append((i * 4, 4, 36 + r, 80, 1))
            for t in tr:
                n1.append((i * 4, 4, 48 + r + t, 60, 2))
            for k in range(4):
                n1.append((i * 4 + k, 1, 72 + [0, 4, 7, 4][k] + r % 12 - (12 if r > 6 else 0),
                           90, 0))
        _mk_midi(os.path.join(d, "a_cadencia.mid"), n1)
        # 2: bucle menor Am F C G (i bVI bIII bVII) con ritmo de corcheas
        pm = [(9, [0, 3, 7]), (5, [0, 4, 7]), (0, [0, 4, 7]), (7, [0, 4, 7])] * 4
        n2 = []
        for i, (r, tr) in enumerate(pm):
            for e in range(8):
                n2.append((i * 4 + e * 0.5, 0.5, 36 + r, 85, 1))
            for t in tr:
                n2.append((i * 4, 4, 48 + r + t, 60, 2))
            for k in range(4):
                n2.append((i * 4 + k, 0.75, 69 + [0, 3, 7, 5][(k + i) % 4], 95, 0))
        _mk_midi(os.path.join(d, "b_menor.mid"), n2, tempo=100)
        # 3: vals 3/4 melodía
        n3 = [(i * 1.5, 1.5, 60 + [0, 2, 4, 5, 7][i % 5], 80, 0) for i in range(45)]
        _mk_midi(os.path.join(d, "c_vals.mid"), n3, ts=(3, 4), tempo=150)
        shutil.copy(os.path.join(d, "a_cadencia.mid"), os.path.join(d, "a_copia.mid"))
        with open(os.path.join(d, "z_roto.mid"), "wb") as f:
            f.write(b"no es midi")
        db = os.path.join(d, "t.db")
        import io
        import contextlib

        def run(argv):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                try:
                    rc = main(argv)
                except SystemExit as e:
                    rc = e.code if isinstance(e.code, int) else 1
            return rc, buf.getvalue()

        rc, out = run(["index", d, "--db", db, "--jobs", "1"])
        check("index termina con código 0", rc == 0)
        con = connect(db)
        st = {r[0]: r[1] for r in con.execute("SELECT status, COUNT(*) FROM files GROUP BY status")}
        check("3 ok, 1 duplicado y 1 error", st == {"ok": 3, "dup": 1, "error": 1})
        r = con.execute("SELECT prog FROM files WHERE name='a_cadencia.mid'").fetchone()
        check("progresión I IV V I detectada (%s)" % (r["prog"][:20] if r else "?"),
              r is not None and r["prog"].startswith("I IV V I"))
        r = con.execute("SELECT prog FROM files WHERE name='b_menor.mid'").fetchone()
        check("bucle menor en grados (%s)" % (r["prog"][:22] if r else "?"),
              r is not None and r["prog"].split()[:4] in (["i", "bVI", "bIII", "bVII"],
                                                          ["vi", "IV", "I", "V"]))
        con.close()
        rc, out = run(["index", d, "--db", db, "--jobs", "1"])
        check("reindexar no recalcula nada", "0 a procesar" in out)
        rc, out = run(["find", "--db", db, "--progression", "I IV V"])
        check("find --progression I IV V encuentra a_cadencia",
              rc == 0 and "a_cadencia" in out and "c_vals" not in out)
        rc, out = run(["find", "--db", db, "--progression", "i bVI bIII bVII"])
        check("find --progression menor encuentra b_menor",
              rc == 0 and "b_menor" in out)
        rc, out = run(["find", "--db", db, "--range", "rhy_meter:3:3"])
        check("find --range rhy_meter:3:3 → solo el vals",
              rc == 0 and "c_vals" in out and "a_cadencia" not in out)
        rc, out = run(["find", "--db", db, "--range", "no_existe:0:1"])
        check("rasgo inexistente → código 2", rc == 2)
        rc, out = run(["find", "--db", db, "--feel", "tension>0.1", "--sort",
                       "tension", "--desc"])
        check("find --feel funciona", rc == 0 and "coincidencias" in out)
        rc, out = run(["similar", os.path.join(d, "a_cadencia.mid"), "--db", db])
        check("similar lista resultados", rc == 0 and "similitud" in out)
        rc, out = run(["similar", os.path.join(d, "a_cadencia.mid"), "--db", db,
                       "--groups", "inventado"])
        check("similar con grupo inválido → código 2", rc == 2)
        rc, out = run(["shape", "--db", db, "--values", "0,1,2,3,4,5,6,7",
                       "--feature", "all_pitch", "--min-sim", "0"])
        check("shape --values no falla", rc in (0, 1))
        rc, out = run(["shape", "--db", db, "--values", "3,3,3,3"])
        check("shape con ventana plana → código 1", rc == 1)
        rc, out = run(["tag", "add", "calma", "c_vals.mid", "--db", db])
        rc2, out2 = run(["similar", "--tag", "calma", "--db", db, "--include-tagged"])
        check("tag + similar --tag", rc == 0 and rc2 == 0 and "c_vals" in out2)
        rc, out = run(["show", "b_menor.mid", "--db", db, "--json"])
        ok = rc == 0
        try:
            j = json.loads(out)
            ok = ok and "tension" in j["feelings"] and j["features"]["rhy_meter"] == 4
        except Exception:  # noqa
            ok = False
        check("show --json válido", ok)
        csvp = os.path.join(d, "x.csv")
        rc, out = run(["export", "--db", db, "--out", csvp])
        with open(csvp) as fh:
            nrows = sum(1 for _ in fh)
        check("export: cabecera + 3 filas", rc == 0 and nrows == 4)
        rc, out = run(["info", "--db", db])
        check("info ejecuta", rc == 0 and "duplicados=1" in out)
        rc, out = run(["find", "--db", os.path.join(d, "nada.db")])
        check("base inexistente → código 1", rc == 1)
        # ── secciones y calibración ──────────────────
        import random
        d2 = os.path.join(d, "cal")
        os.makedirs(d2)
        rg = random.Random(5)
        for i in range(8):                   # P: bucle de acordes + melodía en rampa
            sh = i + 1
            nn = []
            for b in range(16):
                r = [0, 5, 7, 7][b % 4] + sh
                nn.append((b * 4, 4, 36 + r % 12, 80, 1))
                for k in range(4):
                    nn.append((b * 4 + k, 1, 60 + sh % 12 + (b % 8) * 2 + [0, 4, 7, 4][k], 85, 0))
            _mk_midi(os.path.join(d2, "p%d.mid" % i), nn)
        for i in range(8):                   # N: notas al azar
            nn = [(k * 0.5, rg.choice([0.25, 0.5, 1.0]), rg.randint(40, 90),
                   rg.randint(40, 110), 0) for k in range(130)]
            _mk_midi(os.path.join(d2, "n%d.mid" % i), nn, tempo=rg.randint(80, 150))
        mel = []
        for b in range(48):                  # 24 compases estáticos + 24 con sierra
            top = 72 if b < 24 else 58 + ((b - 24) % 8) * 3
            mel.append((b * 4, 4, 36 + [0, 5, 7, 7][b % 4], 80, 1))
            mel += [(b * 4 + k, 1, top + [0, 4, 7, 4][k], 85, 0) for k in range(4)]
        _mk_midi(os.path.join(d2, "mitad.mid"), mel)
        db2 = os.path.join(d, "c.db")
        rc, out = run(["index", d2, "--db", db2, "--jobs", "1"])
        check("index de 17 piezas de calibración (%s)" % out.splitlines()[0][:40],
              rc == 0 and "17 a procesar" in out)
        rc, out = run(["find", "--db", db2, "--gesture", "ramp_reset>=0.5", "--text", "mitad"])
        check("gesto por sección localiza compases (%s)" % out.strip().splitlines()[-1][-30:],
              rc == 0 and "mitad.mid" in out and " c." in out)
        rc, out = run(["tag", "add", "pos", "p0.mid", "p1.mid", "p2.mid", "p3.mid", "--db", db2])
        rc, out = run(["calibrate", "pos", "--db", db2])
        check("calibrate entrena y muestra AUC", rc == 0 and "AUC en validación" in out)
        rc, out = run(["calibrate", "--list", "--db", db2])
        check("calibrate --list muestra el modelo", rc == 0 and "pos" in out and "AUC_cv" in out)
        con2 = connect(db2)
        C2 = Corpus(con2)
        pk = [k for k, i in enumerate(C2.ids)
              if C2.files[i]["name"] in ("p4.mid", "p5.mid", "p6.mid", "p7.mid")]
        nk = [k for k, i in enumerate(C2.ids) if C2.files[i]["name"].startswith("n")]
        sp_ = [C2.feel_pct(float(C2.feel_raw["pos"][k]), "pos") for k in pk]
        sn_ = [C2.feel_pct(float(C2.feel_raw["pos"][k]), "pos") for k in nk]
        check("el modelo puntúa más alto las positivas no vistas (%.2f vs %.2f)"
              % (sum(sp_) / len(sp_), sum(sn_) / len(sn_)),
              sum(sp_) / len(sp_) > sum(sn_) / len(sn_) + 0.2)
        con2.close()
        rc, out = run(["find", "--db", db2, "--feel", "pos>0.7"])
        check("find --feel con el modelo propio", rc == 0 and "p4.mid" in out)
        rc, out = run(["calibrate", "--db", db2, "inexistente"])
        check("calibrate con etiqueta vacía → código 1", rc == 1)
        rc, out = run(["calibrate", "--db", db2, "pos", "--name", "tension"])
        check("nombre reservado → código 2", rc == 2)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    bad = [n for n, ok in res if not ok]
    print("\nselftest: %d/%d" % (len(res) - len(bad), len(res)))
    return 0 if not bad else 1


# ══════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="midi_index.py",
        description="Indexa MIDIs y consulta rasgos, formas, progresiones y "
                    "sensaciones sin conexión.")
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    def db(sp):
        sp.add_argument("--db", required=True, help="base SQLite del índice")

    sp = sub.add_parser("index", help="indexa MIDIs (incremental)")
    sp.add_argument("paths", nargs="+", help="archivos, carpetas o patrones")
    db(sp)
    sp.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    sp.add_argument("--force", action="store_true", help="recalcula todo")
    sp.add_argument("--prune", action="store_true",
                    help="borra del índice archivos que ya no existen")
    sp.add_argument("--min-notes", type=int, default=8)
    sp.add_argument("--max-notes", type=int, default=400000)
    sp.set_defaults(fn=cmd_index)

    sp = sub.add_parser("info", help="resumen del corpus")
    db(sp)
    sp.add_argument("--errors", action="store_true")
    sp.set_defaults(fn=cmd_info)

    sp = sub.add_parser("show", help="ficha de una pieza")
    sp.add_argument("file")
    db(sp)
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--features", action="store_true", help="lista todos los rasgos")
    sp.set_defaults(fn=cmd_show)

    sp = sub.add_parser("similar", help="piezas parecidas a una pieza o etiqueta")
    sp.add_argument("file", nargs="?")
    db(sp)
    sp.add_argument("--tag")
    sp.add_argument("--include-tagged", action="store_true")
    sp.add_argument("--groups", help="solo estos grupos: " + ",".join(GROUPS.values()))
    sp.add_argument("--exclude", help="grupos a ignorar")
    sp.add_argument("--invariant", action="store_true",
                    help="ignora el registro absoluto (compara piezas transportadas)")
    sp.add_argument("--weights", help="p. ej. rhythm=2,harmony=0.5")
    sp.add_argument("-n", type=int, default=10)
    sp.set_defaults(fn=cmd_similar)

    sp = sub.add_parser("find", help="filtra por rasgos, gestos, progresión, sensación")
    db(sp)
    sp.add_argument("--range", action="append", help="rasgo:min:max ('-' abierto)")
    sp.add_argument("--gesture", action="append", help="forma[:rasgo][>=score]")
    sp.add_argument("--progression", help='grados, p. ej. "i bVI bIII bVII"')
    sp.add_argument("--anchor", action="store_true",
                    help="el primer acorde debe caer en ese grado de la tonalidad "
                         "(clave 1 o 2); sin esto solo cuenta el movimiento de raíz")
    sp.add_argument("--strict-quality", action="store_true",
                    help="exige también mayor/menor en cada acorde")
    sp.add_argument("--feel", action="append", help="sensación>percentil (0–1)")
    sp.add_argument("--tag")
    sp.add_argument("--text", help="subcadena de la ruta")
    sp.add_argument("--sort", help="rasgo o sensación por el que ordenar")
    sp.add_argument("--desc", action="store_true")
    sp.add_argument("--show", help="columnas extra (rasgos o sensaciones)")
    sp.add_argument("-n", type=int, default=30)
    sp.set_defaults(fn=cmd_find)

    sp = sub.add_parser("shape", help="busca ventanas con la misma forma")
    db(sp)
    sp.add_argument("--from", dest="from_file")
    sp.add_argument("--feature", help="serie (p. ej. reg_mean, all_pitch, rough)")
    sp.add_argument("--bars", help="a-b (compases, desde 1)")
    sp.add_argument("--values", help="valores separados por comas")
    sp.add_argument("--length", type=int, choices=WIN_LENGTHS)
    sp.add_argument("--min-amp", type=float, default=0.0)
    sp.add_argument("--min-sim", type=float, default=0.6)
    sp.add_argument("--amp-tol", type=float, default=0.0,
                    help="0–1: exige amplitud entre q·tol y q/tol (p. ej. 0.5 = "
                         "mitad a doble); por defecto la forma no mira la amplitud")
    sp.add_argument("--sign", action="store_true",
                    help="exige el mismo sentido neto (sube/baja)")
    sp.add_argument("--per-file", type=int, default=0, help="máx. ventanas por pieza")
    sp.add_argument("-n", type=int, default=20)
    sp.set_defaults(fn=cmd_shape)

    sp = sub.add_parser("tag", help="etiquetas propias para calibrar sensaciones")
    sp.add_argument("action", choices=["add", "rm", "ls"])
    sp.add_argument("tag", nargs="?")
    sp.add_argument("files", nargs="*")
    db(sp)
    sp.set_defaults(fn=cmd_tag)

    sp = sub.add_parser("calibrate", help="aprende una sensación propia "
                                          "a partir de piezas etiquetadas")
    sp.add_argument("tag", nargs="?", help="etiqueta con las piezas positivas")
    db(sp)
    sp.add_argument("--neg", help="etiqueta de negativas (por defecto: el resto)")
    sp.add_argument("--name", help="nombre del modelo (por defecto, la etiqueta)")
    sp.add_argument("--groups", help="solo estos grupos de rasgos")
    sp.add_argument("--exclude", help="grupos a ignorar")
    sp.add_argument("--lam", type=float, default=3.0,
                    help="regularización L2 (más alta = más conservador)")
    sp.add_argument("--top", type=int, default=12)
    sp.add_argument("--candidates", type=int, default=10)
    sp.add_argument("--list", action="store_true")
    sp.add_argument("--show", metavar="MODELO")
    sp.add_argument("--delete", metavar="MODELO")
    sp.set_defaults(fn=cmd_calibrate)

    sp = sub.add_parser("export", help="CSV con todos los rasgos")
    db(sp)
    sp.add_argument("--out", required=True)
    sp.set_defaults(fn=cmd_export)

    sp = sub.add_parser("selftest", help="pruebas internas")
    sp.set_defaults(fn=cmd_selftest)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (IOError, OSError, ValueError, sqlite3.Error) as e:
        sys.stderr.write("Error: %s\n" % e)
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("interrumpido\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
