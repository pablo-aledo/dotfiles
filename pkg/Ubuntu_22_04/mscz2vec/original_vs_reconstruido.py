#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ORIGINAL_VS_RECONSTRUIDO v2.0 — Modelo de decisión «original / reconstructed»║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Dado un MIDI devuelve la etiqueta («original» | «reconstructed») y la       ║
║  probabilidad P(reconstructed). Pensado como DISCRIMINADOR de un flujo       ║
║  antagónico: las herramientas de la suite reconstruyen, éste las delata.     ║
║                                                                              ║
║  SUBCOMANDOS                                                                 ║
║    features   Extrae y cachea el vector de rasgos de uno o varios MIDIs      ║
║    train      Entrena desde cero o continúa un modelo (--init MODELO)        ║
║    eval       Validación cruzada AGRUPADA por pieza (+ ablación/importancia) ║
║    predict    Clasifica MIDIs con un modelo entrenado                        ║
║    info       Resumen de un modelo (arquitectura, rasgos, historial)         ║
║                                                                              ║
║  EJEMPLOS                                                                    ║
║    python3 original_vs_reconstruido.py train --orig 'midis/*.mid' \\          ║
║        --recon 'out/midis_finales/*.mid' --out modelo.npz                    ║
║    python3 original_vs_reconstruido.py train --init modelo.npz --epochs 200  ║
║        --orig ... --recon ... --out modelo2.npz        # continuar/mejorar   ║
║    python3 original_vs_reconstruido.py eval  --orig ... --recon ... --cv 5   ║
║        --compare --importance                                                ║
║    python3 original_vs_reconstruido.py predict cancion.mid --model modelo.npz║
║                                                                              ║
║  DISEÑO (v2.0: criterios MUSICALES, independientes del origen)               ║
║   • Modo por defecto  --features musical. Canonicalización: solo se ven      ║
║     (inicio, fin, altura) de notas no percusivas, en rejilla de semicorchea. ║
║     Se ignoran metadatos, pistas, canales, programas, controladores, tempo,  ║
║     velocidades y percusión → no puede delatar «qué herramienta lo escribió».║
║   • Rasgos (84): ritmo (síncopa, repetición de patrones de compás), melodía  ║
║     (saltos y su recuperación, clímax, entropía condicional), bajo (movim.   ║
║     fundamental), armonía (estabilidad tonal, disonancia fuerte/débil,       ║
║     ajuste a tríadas/séptimas, cadencia y tónica final), conducción de voces ║
║     (paralelas, directo/contrario), textura y forma (autosimilitud por       ║
║     compás a distintos retardos, repetición de bloques, compresibilidad).    ║
║   • Invariancia por ENTRENAMIENTO: --augment K añade K copias de cada        ║
║     fichero con humanización sintética (jitter, rubato, duraciones) para     ║
║     que el modelo no use el «toque» del intérprete ni la cuantización.       ║
║   • --features forense: modo antiguo (huella del pipeline: metadatos,        ║
║     expresión). Útil para saber SI un MIDI salió de un flujo conocido, NO    ║
║     para juzgar música.                                                      ║
║   • Decisión por fichero = media de logits de sus ventanas (8 compases).     ║
║   • Modelo: conjunto de MLP (numpy, Adam, dropout, L2, pesos por fichero y   ║
║     clase). --layers vacío = logística. Escala con --layers/--epochs/        ║
║     --batch/--ensemble/--window-bars/--augment sin tocar código.             ║
║   • Calibración de Platt OOF. Evaluación: CV agrupada por pieza y           ║
║     --loro (deja fuera cada familia de generador = origen desconocido).      ║
║                                                                              ║
║  DEPENDENCIAS: numpy, mido (las de la suite). Sin torch ni scikit-learn.     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import argparse, glob, hashlib, json, math, os, re, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # corrcoef con varianza 0 → se tratan como 0
np.seterr(all="ignore")

try:
    import mido
except ImportError:  # pragma: no cover
    sys.exit("Falta 'mido' (pip install mido)")

VERSION = "2.0"
LABELS = ("original", "reconstructed")
GROUPS = ["meta", "timing", "velocity", "pitch", "texture", "repetition", "harmony", "drums"]
_KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_MAJ_SC = {0, 2, 4, 5, 7, 9, 11}
_MIN_SC = {0, 2, 3, 5, 7, 8, 10}


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path

# ═════════════════════════ colores ═════════════════════════
class _C:
    on = sys.stdout.isatty()
    @staticmethod
    def c(s, k):
        return f"\033[{k}m{s}\033[0m" if _C.on else str(s)

# ═════════════════════════ lectura de MIDI ═════════════════════════
def read_midi(path):
    """→ dict con notas (start_b, end_b, pitch, vel, ch), batería y metadatos."""
    mf = mido.MidiFile(path)
    tpb = float(mf.ticks_per_beat)
    notes, drums = [], []
    cc, tempos, ts, progs = {}, [], 0, set()
    pb = 0; keysig = 0; names = 0; texts = 0; off_exp = 0; off_v0 = 0; eot = 0; bpb = None
    last_end = 0.0; chans = set(); end_tick = 0
    for tr in mf.tracks:
        t = 0; open_ = {}
        if any(m.type == "track_name" for m in tr): names += 1
        for m in tr:
            t += m.time
            ty = m.type
            if ty == "note_on" and m.velocity > 0:
                open_.setdefault((m.channel, m.note), []).append((t, m.velocity)); chans.add(m.channel)
            elif ty == "note_off" or (ty == "note_on" and m.velocity == 0):
                if ty == "note_off": off_exp += 1
                else: off_v0 += 1
                st = open_.get((m.channel, m.note))
                if st:
                    s, v = st.pop(0)
                    rec = (s / tpb, t / tpb, m.note, v, m.channel)
                    (drums if m.channel == 9 else notes).append(rec)
            elif ty == "control_change":
                cc[m.control] = cc.get(m.control, 0) + 1
            elif ty == "pitchwheel": pb += 1
            elif ty == "program_change": progs.add((m.channel, m.program))
            elif ty == "set_tempo": tempos.append((t / tpb, mido.tempo2bpm(m.tempo)))
            elif ty == "time_signature":
                ts += 1
                if bpb is None: bpb = m.numerator * 4.0 / m.denominator
            elif ty == "key_signature": keysig += 1
            elif ty in ("text", "lyrics", "marker", "cue_marker"): texts += 1
            elif ty == "end_of_track": eot += 1
        end_tick = max(end_tick, t)
    notes.sort(); drums.sort()
    return dict(notes=np.array(notes, float).reshape(-1, 5), drums=np.array(drums, float).reshape(-1, 5),
                tpb=tpb, fmt=mf.type, n_tracks=len(mf.tracks), cc=cc, pb=pb, tempos=tempos, ts=ts, keysig=keysig,
                progs=progs, names=names, texts=texts, off_exp=off_exp, off_v0=off_v0, chans=chans,
                end_b=end_tick / tpb, eot=eot, bpb=bpb or 4.0)

# ═════════════════════════ utilidades de rasgos ═════════════════════════
def _ent(x):
    x = np.asarray(x, float); s = x.sum()
    if s <= 0: return 0.0
    p = x[x > 0] / s
    return float(-(p * np.log2(p)).sum())

def _ent_vals(v, nb=None):
    if len(v) == 0: return 0.0
    _, c = np.unique(v, return_counts=True); return _ent(c)

def _dist_grid(f, g):
    r = np.mod(f, g); return np.minimum(r, g - r)

def _key_fit(pch):
    """mejor correlación Krumhansl, modo(+1 mayor/-1 menor), tónica."""
    best = (-2.0, 1, 0)
    if pch.sum() <= 0 or pch.std() == 0: return best
    for tn in range(12):
        for mode, prof in ((1, _KS_MAJ), (-1, _KS_MIN)):
            r = np.corrcoef(pch, np.roll(prof, tn))[0, 1]
            if r > best[0]: best = (r, mode, tn)
    return best

def _autocorr(x, lag):
    x = np.asarray(x, float)
    if len(x) <= lag + 2 or x.std() == 0: return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

def _hist_frac(v, edges):
    h, _ = np.histogram(v, bins=edges); s = h.sum()
    return (h / s) if s else np.zeros(len(h))

# ═════════════════════════ rasgos GLOBALES (fichero) ═════════════════════════
def global_features(d):
    F = {}
    n_b = max(d["end_b"], 1.0)
    tem = np.array([b for _, b in d["tempos"]]) if d["tempos"] else np.array([120.0])
    F["meta_log2_tpb"] = math.log2(d["tpb"])
    F["meta_format"] = d["fmt"]
    F["meta_log_tracks"] = math.log1p(d["n_tracks"])
    F["meta_log_channels"] = math.log1p(len(d["chans"]))
    F["meta_log_tempo_events"] = math.log1p(len(d["tempos"]))
    F["meta_log_tempo_mean"] = math.log(max(tem.mean(), 1.0))
    F["meta_tempo_cv"] = float(tem.std() / tem.mean()) if tem.mean() else 0.0
    F["meta_ts_events"] = math.log1p(d["ts"])
    F["meta_keysig"] = float(d["keysig"] > 0)
    F["meta_n_programs"] = math.log1p(len(d["progs"]))
    for k in (64, 1, 11, 7, 10, 91, 93):
        F[f"meta_cc{k}_rate"] = math.log1p(1000.0 * d["cc"].get(k, 0) / n_b)
    F["meta_cc_other"] = math.log1p(1000.0 * sum(v for k, v in d["cc"].items() if k not in (64, 1, 11, 7, 10, 91, 93)) / n_b)
    F["meta_pitchbend_rate"] = math.log1p(1000.0 * d["pb"] / n_b)
    F["meta_names_frac"] = d["names"] / max(d["n_tracks"], 1)
    F["meta_log_texts"] = math.log1p(d["texts"])
    tot_off = d["off_exp"] + d["off_v0"]
    F["meta_explicit_off_frac"] = d["off_exp"] / tot_off if tot_off else 0.5
    F["meta_log_len_beats"] = math.log(n_b)
    F["meta_eot_per_track"] = d["eot"] / max(d["n_tracks"], 1)
    nn = d["notes"]
    F["meta_first_onset"] = math.log1p(float(nn[:, 0].min())) if len(nn) else 0.0
    F["meta_tail_gap"] = math.log1p(max(0.0, d["end_b"] - float(nn[:, 1].max()))) if len(nn) else 0.0
    return F

# ═════════════════════════ rasgos de VENTANA ═════════════════════════
def window_features(N, D, t0, t1):
    """N, D: notas y batería de la ventana [t0,t1). Devuelve dict ordenado."""
    F = {}
    W = t1 - t0
    st = N[:, 0] - t0; en = N[:, 1] - t0; pit = N[:, 2]; vel = N[:, 3]; ch = N[:, 4]
    dur = np.maximum(en - st, 0)
    n = len(N)
    frac = np.mod(st, 1.0)
    # ── timing ──
    d4 = _dist_grid(st, 0.25); d3 = _dist_grid(st, 1 / 3.0); dd = np.minimum(d4, d3)
    F["timing_dev_grid_mean"] = float(dd.mean())
    F["timing_dev16_mean"] = float(d4.mean())
    F["timing_dev16_std"] = float(d4.std())
    F["timing_on16_frac"] = float((d4 < 0.004).mean())
    F["timing_on_triplet_frac"] = float((d3 < 0.004).mean())
    F["timing_on_any_frac"] = float((dd < 0.004).mean())
    F["timing_unique_pos_ratio"] = len(np.unique(np.round(frac * 96))) / max(n, 1)
    F["timing_pos_entropy"] = _ent_vals(np.round(frac * 24).astype(int) % 24)
    F["timing_on_beat_frac"] = float((_dist_grid(st, 1.0) < 0.004).mean())
    F["timing_on_half_frac"] = float((_dist_grid(st, 0.5) < 0.004).mean())
    ons = np.unique(np.round(st * 96) / 96)
    ioi = np.diff(ons)
    if len(ioi) > 1:
        F["timing_ioi_unique_ratio"] = len(np.unique(np.round(ioi * 96))) / len(ioi)
        F["timing_ioi_entropy"] = _ent_vals(np.round(ioi * 96).astype(int))
        F["timing_ioi_cv"] = float(ioi.std() / ioi.mean()) if ioi.mean() else 0.0
        F["timing_ioi_mean"] = float(ioi.mean())
    else:
        F["timing_ioi_unique_ratio"] = F["timing_ioi_entropy"] = F["timing_ioi_cv"] = F["timing_ioi_mean"] = 0.0
    F["timing_dur_unique_ratio"] = len(np.unique(np.round(dur * 96))) / max(n, 1)
    F["timing_dur_entropy"] = _ent_vals(np.round(dur * 96).astype(int))
    F["timing_dur_mean"] = float(dur.mean())
    F["timing_dur_std"] = float(dur.std())
    F["timing_dur_on_grid"] = float((_dist_grid(dur, 0.25) < 0.004).mean())
    if len(ons) > 2:
        mx = np.array([dur[np.abs(st - o) < 1 / 96.0].max() for o in ons[:-1]])
        leg = mx / np.maximum(ioi, 1e-6)
        F["timing_legato_med"] = float(np.clip(np.median(leg), 0, 2))
        F["timing_legato_touch"] = float((np.abs(leg - 1) < 0.02).mean())
        F["timing_legato_overlap"] = float((leg > 1.02).mean())
        F["timing_legato_short"] = float((leg < 0.5).mean())
    else:
        F["timing_legato_med"] = F["timing_legato_touch"] = F["timing_legato_overlap"] = F["timing_legato_short"] = 0.0
    # ── velocity ──
    F["velocity_mean"] = float(vel.mean()) / 127
    F["velocity_std"] = float(vel.std()) / 127
    F["velocity_range"] = float(vel.max() - vel.min()) / 127
    F["velocity_unique"] = math.log1p(len(np.unique(vel)))
    F["velocity_entropy"] = _ent_vals(vel.astype(int))
    F["velocity_mod5"] = float((np.mod(vel, 5) == 0).mean())
    F["velocity_mod8"] = float((np.mod(vel, 8) == 0).mean())
    F["velocity_lag1"] = _autocorr(vel, 1)
    onb = (_dist_grid(st, 1.0) < 0.004).astype(float)
    F["velocity_beat_corr"] = float(np.corrcoef(vel, onb)[0, 1]) if vel.std() > 0 and onb.std() > 0 else 0.0
    g = {}
    for s_, v_ in zip(np.round(st * 96), vel): g.setdefault(s_, []).append(v_)
    sds = [np.std(v) for v in g.values() if len(v) > 1]
    F["velocity_chord_std"] = float(np.mean(sds)) / 127 if sds else 0.0
    # ── pitch ──
    pch = np.bincount((pit.astype(int)) % 12, weights=dur + 1e-3, minlength=12)
    kr, kmode, ktn = _key_fit(pch)
    F["pitch_range"] = float(pit.max() - pit.min()) / 60
    F["pitch_mean"] = float(pit.mean()) / 127
    F["pitch_std"] = float(pit.std()) / 24
    F["pitch_pc_entropy"] = _ent(pch)
    sp = np.sort(pch / pch.sum())[::-1]
    for i in range(12): F[f"pitch_pc_sorted{i}"] = float(sp[i])
    F["pitch_key_r"] = float(kr); F["pitch_key_mode"] = float(kmode)
    tops = np.array([pit[np.abs(st - o) < 1 / 96.0].max() for o in ons])
    bots = np.array([pit[np.abs(st - o) < 1 / 96.0].min() for o in ons])
    for nm, v in (("top", tops), ("bot", bots)):
        if len(v) > 2:
            iv = np.diff(v)
            h = _hist_frac(np.abs(iv), [0, 0.5, 1.5, 2.5, 4.5, 7.5, 12.5, 200])
            for i, x in enumerate(h): F[f"pitch_{nm}_int{i}"] = float(x)
            F[f"pitch_{nm}_dir"] = float(np.sign(iv).mean())
            F[f"pitch_{nm}_int_entropy"] = _ent_vals(iv.astype(int))
        else:
            for i in range(7): F[f"pitch_{nm}_int{i}"] = 0.0
            F[f"pitch_{nm}_dir"] = F[f"pitch_{nm}_int_entropy"] = 0.0
    F["pitch_low_frac"] = float((pit < 48).mean()); F["pitch_high_frac"] = float((pit > 84).mean())
    F["pitch_spread_mean"] = float((tops - bots).mean()) / 48
    # ── texture ──
    F["texture_log_notes"] = math.log1p(n)
    cnt = np.bincount(np.clip(st.astype(int), 0, int(W) - 1), minlength=int(W))
    F["texture_density_mean"] = float(cnt.mean())
    F["texture_density_std"] = float(cnt.std())
    F["texture_density_lag4"] = _autocorr(cnt, 4)
    gs = np.array([np.sum(np.abs(st - o) < 1 / 96.0) for o in ons])
    F["texture_chord_size_mean"] = float(gs.mean())
    F["texture_simul_frac"] = float((gs > 1).mean())
    poly = np.array([np.sum((st <= o + 1e-6) & (en > o + 1e-6)) for o in ons])
    F["texture_poly_mean"] = float(poly.mean()); F["texture_poly_std"] = float(poly.std()); F["texture_poly_max"] = float(poly.max())
    cov = np.zeros(int(W) * 4 + 1, bool)
    for a, b in zip(st, en):
        cov[max(int(a * 4), 0):max(min(int(math.ceil(b * 4)), len(cov)), 0)] = True
    F["texture_rest_frac"] = float(1 - cov[:-1].mean())
    F["texture_long_frac"] = float((dur > 2).mean())
    F["texture_channels"] = float(len(np.unique(ch)))
    F["texture_barline_cross"] = float(np.mean(np.floor(st / 4) != np.floor(np.maximum(en - 1e-6, 0) / 4)))
    # ── repetition ──
    nb = max(int(W // 4), 1)
    bh, rh = [], []
    for k in range(nb):
        m = (st >= 4 * k) & (st < 4 * k + 4)
        bh.append(tuple(sorted(set((pit[m].astype(int) % 12).tolist()))))
        rh.append(tuple(sorted(set(np.round((st[m] - 4 * k) * 4).astype(int).tolist()))))
    def rep(h):
        seen, r = set(), 0
        for x in h:
            if x in seen and x: r += 1
            seen.add(x)
        return r / max(len(h) - 1, 1)
    F["repetition_bar_pc"] = rep(bh)
    F["repetition_bar_rhythm"] = rep(rh)
    if len(tops) > 6:
        iv = tuple(np.diff(tops).astype(int)); grams = [iv[i:i + 4] for i in range(len(iv) - 3)]
        _, c = np.unique(np.array(grams), axis=0, return_counts=True)
        F["repetition_ngram4"] = float((c[c > 1]).sum() / len(grams))
    else:
        F["repetition_ngram4"] = 0.0
    F["repetition_density_lag8"] = _autocorr(cnt, 8)
    # ── harmony ──
    bsets = []
    for b in range(int(W)):
        m = (st <= b + 1e-6) & (en > b + 1e-6)
        bsets.append(frozenset((pit[m].astype(int) % 12).tolist()))
    jd = [1 - len(a & b) / len(a | b) for a, b in zip(bsets[:-1], bsets[1:]) if (a | b)]
    F["harmony_change_rate"] = float(np.mean(jd)) if jd else 0.0
    sc = _MAJ_SC if kmode == 1 else _MIN_SC
    inkey = np.mean([((int(p) - ktn) % 12) in sc for p in pit])
    F["harmony_in_key"] = float(inkey)
    tpl = []
    for r in range(12):
        for iv_ in ((0, 4, 7), (0, 3, 7), (0, 4, 7, 10), (0, 3, 7, 10), (0, 4, 7, 11), (0, 3, 6)):
            tpl.append(frozenset((r + i) % 12 for i in iv_))
    ne = [s for s in bsets if len(s) >= 3]
    F["harmony_triad_frac"] = float(np.mean([any(s <= t for t in tpl) for s in ne])) if ne else 0.0
    F["harmony_pcset_size"] = float(np.mean([len(s) for s in bsets]))
    # ── drums ──
    if len(D):
        ds = D[:, 0] - t0
        F["drums_present"] = 1.0; F["drums_density"] = len(D) / W
        F["drums_unique"] = float(len(np.unique(D[:, 2])))
        F["drums_vel_std"] = float(D[:, 3].std()) / 127
        F["drums_on16"] = float((_dist_grid(ds, 0.25) < 0.004).mean())
    else:
        F["drums_present"] = F["drums_density"] = F["drums_unique"] = F["drums_vel_std"] = F["drums_on16"] = 0.0
    return F

def _group_of(name):
    return name.split("_", 1)[0]

def file_windows(d, window=32.0, stride=None, max_windows=40, min_notes=8):
    """→ (matriz [n_ventanas × n_rasgos], nombres)."""
    stride = stride or window
    G = global_features(d)
    N, D = d["notes"], d["drums"]
    rows, names = [], None
    if len(N) == 0:
        return np.zeros((0, 0)), []
    t_end = float(N[:, 0].max()) + 1e-6
    starts = np.arange(0.0, max(t_end - window * 0.5, 1.0), stride)
    if len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    for t0 in starts:
        t1 = t0 + window
        m = (N[:, 0] >= t0) & (N[:, 0] < t1)
        if m.sum() < min_notes: continue
        md = (D[:, 0] >= t0) & (D[:, 0] < t1) if len(D) else np.zeros(0, bool)
        F = window_features(N[m], D[md] if len(D) else D, t0, t1)
        F.update(G)
        if names is None: names = list(F.keys())
        rows.append([F[k] for k in names])
    if not rows: return np.zeros((0, 0)), []
    X = np.nan_to_num(np.array(rows, float), nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


# ═════════════════════════ MODO «musical»: rasgos independientes del origen ═════════════════════════
# Canonicalización: solo (inicio, fin, altura) de las notas NO percusivas, ajustados a una rejilla de
# 1/4 de negra (+ perturbaciones de entrenamiento). Se descartan metadatos, pistas, canales, controladores, velocidades, tempo y percusión:
# humanizar, cuantizar, reordenar pistas o cambiar de resolución NO cambia nada de lo que ve el modelo.
LAT = 4                                          # unidades de rejilla por negra (semicorchea). Tresillos se aproximan.
MGROUPS = ["rhythm", "melody", "bass", "harmony", "voice", "texture", "form"]
_DISS = {1, 6, 11}

def canon(d):
    N = d["notes"]
    if len(N) == 0: return None
    st = np.round(N[:, 0] * LAT).astype(np.int64); en = np.maximum(np.round(N[:, 1] * LAT).astype(np.int64), st + 1)
    pit = N[:, 2].astype(np.int64)
    _, ix = np.unique(np.stack([st, pit], 1), axis=0, return_index=True); ix.sort()
    return st[ix], en[ix], pit[ix]

def _tb(s, p, e):
    """voz superior/inferior por instante de ataque."""
    o = np.argsort(s, kind="stable"); s, p, e = s[o], p[o], e[o]
    f = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    return s[f], np.maximum.reduceat(p, f), np.minimum.reduceat(p, f), np.diff(np.r_[f, len(s)])

def _cent(seq):
    """entropía condicional H(x_t | x_{t-1}) con estimador plug-in."""
    if len(seq) < 4: return 0.0
    a = np.asarray(seq); pairs = {}
    for x, y in zip(a[:-1], a[1:]): pairs.setdefault(x, []).append(y)
    n = len(a) - 1
    return float(sum(len(v) / n * _ent_vals(np.array(v)) for v in pairs.values()))

def _ratio_zlib(tokens):
    import zlib
    if len(tokens) < 8: return 1.0
    b = bytes(int(t) & 255 for t in tokens)
    return len(zlib.compress(b, 9)) / max(len(b), 1)

def _bars(on, barL, nb):
    return [tuple(sorted(set(((on[(on >= k * barL) & (on < (k + 1) * barL)] - k * barL) // (LAT // 4)).tolist()))) for k in range(nb)]

def _jac(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0

def mus_window(st, en, pit, t0, barL):
    F = {}
    s = st - t0; e = en - t0; n = len(s)
    on, top, bot, gs = _tb(s, pit, e)
    W = int(max(e.max(), 1)); nb = max(int(math.ceil(W / barL)), 1)
    # ── rhythm ──
    pos = on % LAT
    F["rhythm_offbeat_frac"] = float((pos != 0).mean())
    F["rhythm_halfbeat_frac"] = float((pos == LAT // 2).mean())
    F["rhythm_16th_frac"] = float(np.isin(pos, [LAT // 4, 3 * LAT // 4]).mean())
    F["rhythm_barpos_entropy"] = _ent_vals((on % barL) // (LAT // 4))
    F["rhythm_downbeat_frac"] = float((on % barL == 0).mean())
    ioi = np.diff(on)
    F["rhythm_ioi_entropy"] = _ent_vals(ioi) if len(ioi) else 0.0
    F["rhythm_ioi_unique"] = len(np.unique(ioi)) / max(len(ioi), 1)
    F["rhythm_ioi_cv"] = float(ioi.std() / ioi.mean()) if len(ioi) > 1 and ioi.mean() else 0.0
    dur = e - s
    F["rhythm_dur_entropy"] = _ent_vals(dur); F["rhythm_dur_mean"] = float(dur.mean() / LAT)
    F["rhythm_long_frac"] = float((dur > 2 * LAT).mean())
    F["rhythm_tie_frac"] = float(np.mean((s // barL) != ((e - 1) // barL)))
    bp = _bars(on, barL, nb)
    F["rhythm_bar_distinct"] = len(set(bp)) / nb
    for lag in (1, 2, 4):
        F[f"rhythm_bar_sim_lag{lag}"] = float(np.mean([_jac(bp[k], bp[k - lag]) for k in range(lag, nb)])) if nb > lag else 0.0
    # ── melody (voz superior) ──
    iv = np.diff(top)
    if len(iv) > 3:
        a = np.abs(iv)
        F["melody_step_frac"] = float(((a >= 1) & (a <= 2)).mean()); F["melody_leap_frac"] = float((a > 4).mean())
        F["melody_rep_frac"] = float((a == 0).mean()); F["melody_bigleap_frac"] = float((a > 9).mean())
        lp = np.nonzero(a[:-1] > 4)[0]
        F["melody_leap_recover"] = float(np.mean([(np.sign(iv[i + 1]) == -np.sign(iv[i])) and a[i + 1] <= 2 for i in lp])) if len(lp) else 0.5
        sg = np.sign(iv[iv != 0]); F["melody_dir_change"] = float((sg[1:] != sg[:-1]).mean()) if len(sg) > 1 else 0.0
        F["melody_int_entropy"] = _ent_vals(np.clip(iv, -12, 12)); F["melody_cond_entropy"] = _cent(np.clip(iv, -12, 12))
        F["melody_mean_abs_int"] = float(a.mean())
    else:
        for k in ("step_frac", "leap_frac", "rep_frac", "bigleap_frac", "leap_recover", "dir_change", "int_entropy", "cond_entropy", "mean_abs_int"): F["melody_" + k] = 0.0
    F["melody_range"] = float(top.max() - top.min()) / 36.0
    F["melody_climax_pos"] = float(np.argmax(top)) / max(len(top) - 1, 1)
    F["melody_climax_unique"] = float((top == top.max()).mean())
    F["melody_pc_entropy"] = _ent(np.bincount(top % 12, minlength=12))
    # ── bass ──
    ib = np.abs(np.diff(bot)) % 12 if len(bot) > 3 else np.zeros(0, int)
    if len(ib):
        F["bass_unison"] = float((ib == 0).mean()); F["bass_step"] = float(np.isin(ib, [1, 2, 10, 11]).mean())
        F["bass_third"] = float(np.isin(ib, [3, 4, 8, 9]).mean()); F["bass_fourth_fifth"] = float(np.isin(ib, [5, 7]).mean())
        F["bass_tritone"] = float((ib == 6).mean()); F["bass_cond_entropy"] = _cent(np.clip(np.diff(bot), -12, 12))
    else:
        for k in ("unison", "step", "third", "fourth_fifth", "tritone", "cond_entropy"): F["bass_" + k] = 0.0
    F["bass_low_frac"] = float((bot < 48).mean()); F["bass_pc_entropy"] = _ent(np.bincount(bot % 12, minlength=12))
    # ── harmony ──
    pch = np.bincount(pit % 12, weights=dur + 1e-3, minlength=12)
    kr, kmode, ktn = _key_fit(pch); sc = _MAJ_SC if kmode == 1 else _MIN_SC
    F["harmony_key_r"] = float(kr)
    F["harmony_in_key"] = float(np.mean([((int(p) - ktn) % 12) in sc for p in pit])); F["harmony_pc_entropy"] = _ent(pch)
    sets = []
    for b in range(0, W, LAT):
        m = (s <= b) & (e > b); sets.append(frozenset((pit[m] % 12).tolist()))
    jd = [1 - len(a_ & b_) / len(a_ | b_) for a_, b_ in zip(sets[:-1], sets[1:]) if (a_ | b_)]
    F["harmony_change_rate"] = float(np.mean(jd)) if jd else 0.0
    tpl = [frozenset((r + i) % 12 for i in iv_) for r in range(12) for iv_ in ((0, 4, 7), (0, 3, 7), (0, 4, 7, 10), (0, 3, 7, 10), (0, 4, 7, 11), (0, 3, 6), (0, 3, 6, 9))]
    ne = [x for x in sets if len(x) >= 3]
    F["harmony_triad_fit"] = float(np.mean([any(x <= t for t in tpl) for x in ne])) if ne else 0.0
    F["harmony_pcset_size"] = float(np.mean([len(x) for x in sets])); F["harmony_pcset_distinct"] = len(set(sets)) / max(len(sets), 1)
    dstr = dweak = nstr = nwk = 0
    sel = on if len(on) <= 70 else on[np.linspace(0, len(on) - 1, 70).astype(int)]
    for o in sel:
        m = (s <= o) & (e > o); ps = np.unique(pit[m])
        if len(ps) < 2: continue
        ic = np.abs(ps[:, None] - ps[None, :])[np.triu_indices(len(ps), 1)] % 12
        nd = np.isin(ic, list(_DISS)).sum()
        if o % LAT == 0: dstr += nd; nstr += len(ic)
        else: dweak += nd; nwk += len(ic)
    F["harmony_diss_strong"] = dstr / nstr if nstr else 0.0; F["harmony_diss_weak"] = dweak / nwk if nwk else 0.0
    F["harmony_diss_gap"] = F["harmony_diss_weak"] - F["harmony_diss_strong"]
    # ── voice leading ──
    pol = top > bot
    if pol.sum() > 3:
        dt, db = np.diff(top), np.diff(bot); vi = (top - bot)
        ok = pol[:-1] & pol[1:]; mv = ok & ((dt != 0) | (db != 0))
        if mv.sum():
            sim = mv & (np.sign(dt) == np.sign(db)) & (dt != 0); con = mv & (np.sign(dt) == -np.sign(db)) & (dt != 0) & (db != 0)
            obl = mv & ((dt == 0) ^ (db == 0))
            par = sim & np.isin(vi[1:] % 12, [0, 7]) & np.isin(vi[:-1] % 12, [0, 7]) & (vi[1:] % 12 == vi[:-1] % 12)
            F["voice_similar"] = float(sim.sum() / mv.sum()); F["voice_contrary"] = float(con.sum() / mv.sum())
            F["voice_oblique"] = float(obl.sum() / mv.sum()); F["voice_parallel_perfect"] = float(par.sum() / mv.sum())
        else:
            F["voice_similar"] = F["voice_contrary"] = F["voice_oblique"] = F["voice_parallel_perfect"] = 0.0
        F["voice_spacing_mean"] = float(vi[pol].mean()) / 24.0; F["voice_spacing_std"] = float(vi[pol].std()) / 12.0
    else:
        for k in ("similar", "contrary", "oblique", "parallel_perfect", "spacing_mean", "spacing_std"): F["voice_" + k] = 0.0
    # ── texture ──
    poly = np.array([np.sum((s <= o) & (e > o)) for o in sel])
    F["texture_poly_mean"] = float(poly.mean()); F["texture_poly_std"] = float(poly.std()); F["texture_poly_max"] = float(poly.max())
    F["texture_simul_frac"] = float((gs > 1).mean()); F["texture_chord_size"] = float(gs.mean())
    cnt = np.bincount(np.clip(s // barL, 0, nb - 1), minlength=nb)
    F["texture_bar_density_cv"] = float(cnt.std() / cnt.mean()) if cnt.mean() else 0.0
    F["texture_log_notes"] = math.log1p(n)
    cov = np.zeros(W // (LAT // 4) + 1, bool)
    for a_, b_ in zip(s, e): cov[a_ // (LAT // 4): -(-b_ // (LAT // 4))] = True
    F["texture_rest_frac"] = float(1 - cov[:-1].mean())
    # ── form (local) ──
    bpc = [tuple(sorted(set((pit[(s >= k * barL) & (s < (k + 1) * barL)] % 12).tolist()))) for k in range(nb)]
    seen, r = set(), 0
    for x in bpc:
        if x in seen and x: r += 1
        seen.add(x)
    F["form_bar_pc_repeat"] = r / max(nb - 1, 1)
    if len(iv) > 6:
        g4 = [tuple(iv[i:i + 4]) for i in range(len(iv) - 3)]; from collections import Counter
        c = Counter(g4); F["form_ngram4_repeat"] = sum(v for v in c.values() if v > 1) / len(g4)
    else: F["form_ngram4_repeat"] = 0.0
    tok = np.concatenate([np.clip(iv, -12, 12) + 12, np.clip(ioi // (LAT // 4), 0, 30) + 40]) if len(iv) and len(ioi) else np.zeros(0)
    F["form_compress_ratio"] = _ratio_zlib(np.ravel(np.column_stack([np.clip(iv, -12, 12) + 12, np.clip(ioi // (LAT // 4), 0, 30) + 40])) if len(iv) == len(ioi) and len(iv) else tok)
    return F

def mus_file(st, en, pit, barL, t_first):
    """rasgos ESTRUCTURALES de la pieza completa (constantes entre sus ventanas)."""
    F = {}
    s = st - t_first; e = en - t_first
    on, top, bot, gs = _tb(s, pit, e)
    nbars = max(int(math.ceil(e.max() / barL)), 1)
    F["form_log_bars"] = math.log(nbars)
    pch = np.bincount(pit % 12, weights=(e - s) + 1e-3, minlength=12); kr, kmode, ktn = _key_fit(pch)
    F["harmony_global_key_r"] = float(kr)
    win = barL * 8; tn = []
    for t0 in range(0, int(e.max()), win):
        m = (s >= t0) & (s < t0 + win)
        if m.sum() > 10:
            h = np.bincount(pit[m] % 12, weights=(e[m] - s[m]) + 1e-3, minlength=12); tn.append((_key_fit(h)[2], _key_fit(h)[1]))
    F["harmony_key_stability"] = float(np.mean([t == (ktn, kmode) for t in tn])) if tn else 1.0
    F["harmony_key_changes"] = float(sum(a_ != b_ for a_, b_ in zip(tn[:-1], tn[1:]))) / max(len(tn) - 1, 1) if len(tn) > 1 else 0.0
    F["harmony_start_tonic"] = float(bot[0] % 12 == ktn); F["harmony_end_tonic_bass"] = float(bot[-1] % 12 == ktn)
    F["harmony_end_tonic_top"] = float((top[-1] - ktn) % 12 in (0, 4, 7, 3)); F["harmony_end_tonic_exact"] = float((top[-1] - ktn) % 12 == 0)
    last = (e[s == on[-1]]).max() - on[-1]
    F["form_last_long"] = float(min(last / LAT, 8.0)) / 8.0
    dens = np.bincount(np.clip(s // barL, 0, nbars - 1), minlength=nbars).astype(float)
    F["form_density_cv"] = float(dens.std() / dens.mean()) if dens.mean() else 0.0
    F["form_density_peak_pos"] = float(np.argmax(dens)) / max(nbars - 1, 1)
    F["form_climax_pos"] = float(np.argmax(top)) / max(len(top) - 1, 1); F["form_climax_unique_global"] = float((top == top.max()).mean())
    vecs = []
    for k in range(min(nbars, 160)):
        m = (s >= k * barL) & (s < (k + 1) * barL)
        v = np.zeros(barL // (LAT // 4) + 12)
        v[((s[m] - k * barL) // (LAT // 4)).clip(0, barL // (LAT // 4) - 1)] = 1; np.add.at(v, barL // (LAT // 4) + (pit[m] % 12), 1)
        vecs.append(v)
    V = np.array(vecs); nv = np.linalg.norm(V, axis=1, keepdims=True); V = V / np.maximum(nv, 1e-9)
    S = V @ V.T; nbv = len(V)
    for lag in (1, 2, 4, 8):
        F[f"form_selfsim_lag{lag}"] = float(np.mean(np.diag(S, lag))) if nbv > lag + 1 else 0.0
    F["form_selfsim_far_max"] = float(max([np.mean(np.diag(S, l)) for l in range(4, min(nbv - 2, 40))] or [0.0]))
    blk = [tuple(np.round(V[k:k + 4].sum(0), 1).tolist()) for k in range(0, nbv - 3, 4)]
    F["form_block4_repeat"] = float(1 - len(set(blk)) / len(blk)) if blk else 0.0
    ioi = np.diff(on) // (LAT // 4); iv = np.diff(top)
    if len(iv) > 8:
        F["form_global_compress"] = _ratio_zlib(np.ravel(np.column_stack([np.clip(iv, -12, 12) + 12, np.clip(ioi, 0, 30) + 40])))
    else: F["form_global_compress"] = 1.0
    return F

def file_windows_musical(d, window_bars=8, max_windows=40, min_notes=8):
    c = canon(d)
    if c is None: return np.zeros((0, 0)), []
    st, en, pit = c
    bpb = d.get("bpb") or 4.0; barL = max(int(round(bpb * LAT)), LAT)
    t_first = (int(st.min()) // barL) * barL
    G = mus_file(st, en, pit, barL, t_first)
    win = barL * window_bars; rows = []; names = None
    s0 = st - t_first
    starts = np.arange(0, max(int(s0.max()) + 1 - win // 2, 1), win)
    if len(starts) > max_windows: starts = starts[np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    for t0 in starts:
        m = (s0 >= t0) & (s0 < t0 + win)
        if m.sum() < min_notes: continue
        try:
            F = mus_window(s0[m], en[m] - t_first, pit[m], int(t0), barL)
        except Exception:
            continue
        F.update(G)
        if names is None: names = list(F.keys())
        rows.append([F.get(k, 0.0) for k in names])
    if not rows: return np.zeros((0, 0)), []
    return np.nan_to_num(np.array(rows, float), nan=0.0, posinf=0.0, neginf=0.0), names


def perturb_notes(N, rs, strength=1.0):
    """Humanización sintética (jitter, rubato suave, escalado de duraciones): solo para ENTRENAR invariancia."""
    N = N.copy()
    if len(N) == 0: return N
    n = len(N); A = rs.uniform(0.03, 0.14) * strength; P = rs.uniform(6, 28); ph = rs.uniform(0, 6.28)
    warp = lambda t: A * np.sin(2 * np.pi * t / P + ph)
    j = rs.normal(0, 0.045 * strength, n)
    st = N[:, 0] + j + warp(N[:, 0])
    dur = np.maximum(N[:, 1] - N[:, 0], 0.02) * rs.uniform(0.6, 1.5) * np.exp(rs.normal(0, 0.15 * strength, n))
    en = st + np.maximum(dur, 0.03)
    N[:, 0], N[:, 1] = np.maximum(st, 0), np.maximum(en, 0.03)
    N[:, 3] = np.clip(N[:, 3] + rs.normal(0, 8, n), 1, 127)
    return N

# ═════════════════════════ negativos SINTÉTICOS (corrupciones musicales de un original) ═════════════════════════
# Enseñan «coherencia musical» (forma, tonalidad, melodía) sin depender de qué receta generó el MIDI.
# Cada corrupción parte de un ORIGINAL y comparte su gid → la CV agrupada nunca mezcla un original y sus corrupciones.
SYNTH_KINDS = ["shuffle_bars", "seg_transpose", "key_break", "pitch_noise", "loop_block"]

def _is_real(d):
    """True para ficheros reales (ni copia perturbada ni negativo sintético): los únicos que puntúan en las métricas."""
    return not d["aug"] and not d.get("kind")

def _same_notes(A, B):
    if A.shape != B.shape: return False
    ka = A[np.lexsort((A[:, 2], A[:, 0]))][:, [0, 2]]; kb = B[np.lexsort((B[:, 2], B[:, 0]))][:, [0, 2]]
    return bool(np.allclose(ka, kb))

def _shift_segments(N, b, ub, seg_lo, seg_hi, shifts, rs, cumulative=False):
    """Parte los compases en segmentos de seg_lo..seg_hi compases y transpone cada uno (el primero queda intacto)."""
    out = N.copy(); i = 0; acc = 0; first = True
    while i < len(ub):
        seg = ub[i:i + int(rs.randint(seg_lo, seg_hi + 1))]; i += len(seg)
        if first: first = False; continue
        sh = int(rs.choice(shifts))
        if cumulative: acc += sh; sh = (acc + 6) % 12 - 6
        m = np.isin(b, seg)
        out[m, 2] = np.clip(out[m, 2] + sh, 0, 127)
    return out

def corrupt_notes(N0, rs, kind, bpb=4.0):
    """Devuelve las notas corrompidas (mismo formato que d['notes']) o None si la pieza no admite esa corrupción."""
    if len(N0) < 16: return None
    N = N0.copy()
    b = np.floor(N[:, 0] / bpb + 1e-6).astype(int); ub = np.unique(b)
    out = None
    if kind == "shuffle_bars":          # bloques de 1-4 compases en orden barajado: rompe la forma, conserva el material local
        ok = [c for c in (1, 2, 4) if len(ub) // c >= 2]
        if not ok: return None
        blk = int(rs.choice(ok)); nbk = int(math.ceil(len(ub) / blk))
        blocks = [ub[i * blk:(i + 1) * blk] for i in range(nbk)]
        order = rs.permutation(nbk)
        if np.array_equal(order, np.arange(nbk)): order = np.roll(order, 1)
        new = np.concatenate([blocks[q] for q in order])              # el compás new[k] pasa a la ranura ub[k]
        slot = np.empty_like(ub); slot[np.searchsorted(ub, new)] = ub
        d = (slot[np.searchsorted(ub, b)] - b) * bpb
        N[:, 0] += d; N[:, 1] += d; out = N
    elif kind == "seg_transpose":       # segmentos cortos (1-3 compases) transpuestos de forma independiente
        if len(ub) < 4: return None
        out = _shift_segments(N, b, ub, 1, 3, [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6], rs)
    elif kind == "key_break":           # «modulaciones» cromáticas/tritono acumuladas cada 4-8 compases (toda ventana de 8 compases cruza una)
        if len(ub) < 8: return None
        out = _shift_segments(N, b, ub, 4, 8, [1, 6, 11], rs, cumulative=True)
    elif kind == "pitch_noise":         # 15-40 % de las notas desplazadas ±1..9 semitonos
        sel = rs.rand(len(N)) < rs.uniform(0.15, 0.4)
        off = rs.randint(-9, 10, len(N)); off[off == 0] = 1
        N[sel, 2] = np.clip(N[sel, 2] + off[sel], N[:, 2].min(), N[:, 2].max()); out = N
    elif kind == "loop_block":          # un bloque de 1-4 compases repetido en bucle durante toda la pieza
        ok = [c for c in (1, 2, 4) if len(ub) >= 3 * c]
        if not ok: return None
        blk = int(rs.choice(ok)); s0 = int(ub[int(rs.randint(0, len(ub) - blk + 1))])
        m = (b >= s0) & (b < s0 + blk)
        if m.sum() < 4: return None
        first, last = int(ub[0]), int(ub[-1]); parts = []
        for r in range(int(math.ceil((last + 1 - first) / blk))):
            P = N[m].copy(); dl = (first + r * blk - s0) * bpb
            P[:, 0] += dl; P[:, 1] += dl; parts.append(P)
        out = np.vstack(parts); out = out[out[:, 0] < (last + 1) * bpb]
    else:
        raise ValueError(f"corrupción desconocida: {kind}")
    if out is None or _same_notes(N0, out): return None
    return out[np.lexsort((out[:, 2], out[:, 0]))]

# ═════════════════════════ caché de rasgos ═════════════════════════
def cached_windows(path, window, stride, max_windows, cache_dir, features="musical", aug=0, synth=None):
    """synth = (tipo, n) → ventanas de una corrupción sintética del fichero (ver corrupt_notes); X vacío si no aplica."""
    key = hashlib.md5((f"{aug}|{os.path.abspath(path)}|{os.path.getsize(path)}|{int(os.path.getmtime(path))}|{window}|{stride}|{max_windows}|{features}|{VERSION}"
                       + (f"|synth:{synth[0]}:{synth[1]}" if synth else "")).encode()).hexdigest()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cp = os.path.join(cache_dir, key + ".npz")
        if os.path.exists(cp):
            z = np.load(cp, allow_pickle=True)
            return z["X"], list(z["names"])
    d = read_midi(path)
    if aug:
        d["notes"] = perturb_notes(d["notes"], np.random.RandomState(int(hashlib.md5(f"{path}{aug}".encode()).hexdigest()[:8], 16)))
    if synth:
        rs_s = np.random.RandomState(int(hashlib.md5(f"{path}|{synth[0]}|{synth[1]}".encode()).hexdigest()[:8], 16))
        cn = corrupt_notes(d["notes"], rs_s, synth[0], d.get("bpb") or 4.0)
        if cn is None:
            X, names = np.zeros((0, 0)), []
            if cache_dir: np.savez_compressed(cp, X=X, names=np.array(names))
            return X, names
        d["notes"] = cn
    if features == "musical":
        X, names = file_windows_musical(d, int(round(window)), max_windows)
    else:
        X, names = file_windows(d, window, stride, max_windows)
    if cache_dir:
        np.savez_compressed(cp, X=X, names=np.array(names))
    return X, names

# ═════════════════════════ conjunto de datos ═════════════════════════
def _expand(patterns):
    out = []
    for p in patterns or []:
        g = sorted(glob.glob(p)) if any(c in p for c in "*?[") else ([p] if os.path.exists(p) else [])
        if os.path.isdir(p): g = sorted(glob.glob(os.path.join(p, "*.mid")))
        out += g
    return out

def _piece_id(path, label, orig_ids=None):
    """ID de pieza. Original: su nombre sin extensión. Reconstrucción `NN_ID_receta.mid`:
    ID = el nombre de un original conocido (admite IDs alfanuméricos y con '_');
    si no hay originales conocidos, ID numérico (`^\\d+_(\\d+)_`)."""
    b = os.path.splitext(os.path.basename(path))[0]
    if label == 1:
        m = re.match(r"^\d+_(.+)$", b)
        if m and orig_ids:
            rest = m.group(1)
            best = max((o for o in orig_ids if rest.startswith(o + "_")), key=len, default=None)
            if best is not None:
                return best
        m = re.match(r"^\d+_(\d+)_", b)
        return m.group(1) if m else b
    return b

def build_dataset(args, need_labels=True):
    items = []
    for lab, pats in ((0, args.orig), (1, args.recon)):
        for p in _expand(pats):
            items.append((p, lab))
    if not items: sys.exit("No hay MIDIs: usa --orig/--recon")
    t0 = time.time(); ds = []
    names = None
    K = getattr(args, "augment", 0) if args.features == "musical" else 0
    SK = getattr(args, "synth", 0) or 0
    if SK and args.features != "musical":
        print("  [AVISO] --synth solo aplica a --features musical; se ignora.", file=sys.stderr); SK = 0
    kinds = getattr(args, "synth_kinds", None) or SYNTH_KINDS; n_skip = 0
    orig_ids = {os.path.splitext(os.path.basename(p))[0] for p, lab in items if lab == 0}
    for i, (p, lab) in enumerate(items):
        for ag in range(K + 1):
            try:
                X, nm = cached_windows(p, (args.window_bars if args.features == "musical" else args.window), args.stride, args.max_windows, args.cache, args.features, ag)
            except Exception as e:
                if ag == 0: print(f"  [AVISO] {p}: {e}", file=sys.stderr)
                continue
            if len(X) == 0:
                if ag == 0: print(f"  [AVISO] {p}: sin ventanas útiles", file=sys.stderr)
                continue
            names = names or nm
            ds.append(dict(path=p, y=lab, gid=_piece_id(p, lab, orig_ids), X=X, aug=ag, kind=""))
        if lab == 0 and SK:      # K negativos sintéticos por original, repartiendo los tipos en rotación
            for j in range(SK):
                kind = kinds[j % len(kinds)]
                try:
                    X, nm = cached_windows(p, args.window_bars, args.stride, args.max_windows, args.cache, args.features, 0, (kind, j))
                except Exception as e:
                    print(f"  [AVISO] sintético {kind} de {p}: {e}", file=sys.stderr); continue
                if len(X) == 0: n_skip += 1; continue
                names = names or nm
                ds.append(dict(path=p, y=1, gid=_piece_id(p, 0), X=X, aug=0, kind=kind))
        if args.verbose and (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(items)} ficheros ({time.time()-t0:.0f}s)", file=sys.stderr)
    if n_skip: print(f"  [AVISO] {n_skip} negativos sintéticos omitidos (pieza demasiado corta/simple para esa corrupción)", file=sys.stderr)
    ids_o = {d["gid"] for d in ds if d["y"] == 0}; ids_r = {d["gid"] for d in ds if d["y"] == 1 and not d["kind"]}
    for d in ds: d["paired"] = d["gid"] in ids_o and d["gid"] in ids_r
    _unp = sorted(g for g in ids_r if g not in ids_o)
    if _unp:
        print(f"  [AVISO] {len(_unp)} reconstruccion(es) sin original emparejado (p. ej. {_unp[0]!r}): "
              "el nombre debe ser NN_<ID>_<receta>.mid con <ID> = nombre del original sin .mid. "
              "Sin pareja no se agrupan en la CV.", file=sys.stderr)
    return ds, names

def groups_in(names):
    return [g for g in GROUPS + MGROUPS if any(_group_of(n) == g for n in names)]

def drop_columns(names, drop_groups):
    keep = [i for i, n in enumerate(names) if _group_of(n) not in set(drop_groups or [])]
    return keep

# ═════════════════════════ MLP en numpy ═════════════════════════
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

class MLP:
    """Perceptrón multicapa (ReLU, dropout, L2, Adam). layers=[] → regresión logística."""
    def __init__(self, d_in, layers, seed=0, dropout=0.2, l2=1e-3):
        self.sizes = [d_in] + list(layers) + [1]
        self.dropout, self.l2 = dropout, l2
        r = np.random.RandomState(seed)
        self.W = [r.randn(a, b) * math.sqrt(2.0 / a) for a, b in zip(self.sizes[:-1], self.sizes[1:])]
        self.b = [np.zeros(b) for b in self.sizes[1:]]
        self.m = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.v = [np.zeros_like(w) for w in self.W] + [np.zeros_like(b) for b in self.b]
        self.t = 0
        self.rng = np.random.RandomState(seed + 1000)

    def forward(self, X, train=False):
        acts, masks = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(z, 0)
                if train and self.dropout > 0:
                    mk = (self.rng.rand(*h.shape) > self.dropout) / (1 - self.dropout)
                    h = h * mk; masks.append(mk)
                else: masks.append(None)
            else:
                h = z
            acts.append(h)
        return acts, masks

    def logit(self, X):
        return self.forward(X)[0][-1][:, 0]

    def step(self, X, y, w, lr, b1=0.9, b2=0.999, eps=1e-8):
        acts, masks = self.forward(X, True)
        z = acts[-1][:, 0]; p = _sig(z)
        g = ((p - y) * w / max(w.sum(), 1e-9))[:, None]
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ g + self.l2 * self.W[i]; gb[i] = g.sum(0)
            if i > 0:
                g = g @ self.W[i].T
                if masks[i - 1] is not None: g = g * masks[i - 1]
                g = g * (acts[i] > 0)
        self.t += 1
        params = self.W + self.b; grads = gW + gb
        for k, (P, G) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * G
            self.v[k] = b2 * self.v[k] + (1 - b2) * G * G
            P -= lr * (self.m[k] / (1 - b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - b2 ** self.t)) + eps)
        loss = -np.sum(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))) / max(w.sum(), 1e-9)
        return loss

    def fit(self, X, y, w, epochs, batch, lr, verbose=False):
        n = len(X); hist = []
        for ep in range(epochs):
            idx = self.rng.permutation(n); tot = 0.0
            for s in range(0, n, batch):
                j = idx[s:s + batch]
                tot += self.step(X[j], y[j], w[j], lr) * len(j)
            hist.append(tot / n)
            if verbose and (ep + 1) % max(epochs // 5, 1) == 0:
                print(f"    época {ep+1}/{epochs}  pérdida {hist[-1]:.4f}", file=sys.stderr)
        return hist

    def state(self):
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        d.update({f"m{i}": m for i, m in enumerate(self.m)}); d.update({f"v{i}": v for i, v in enumerate(self.v)})
        d["t"] = np.array(self.t)
        return d

    @classmethod
    def from_state(cls, z, prefix, sizes, dropout, l2, seed=0):
        o = cls(sizes[0], sizes[1:-1], seed, dropout, l2)
        nl = len(o.W)
        o.W = [z[f"{prefix}W{i}"].copy() for i in range(nl)]; o.b = [z[f"{prefix}b{i}"].copy() for i in range(nl)]
        o.m = [z[f"{prefix}m{i}"].copy() for i in range(nl)] + [z[f"{prefix}m{nl+i}"].copy() for i in range(nl)]
        o.v = [z[f"{prefix}v{i}"].copy() for i in range(nl)] + [z[f"{prefix}v{nl+i}"].copy() for i in range(nl)]
        o.t = int(z[f"{prefix}t"])
        return o

# ═════════════════════════ modelo (conjunto + normalizador + Platt) ═════════════════════════
class Detector:
    def __init__(self, names, keep, layers, dropout, l2, window, stride, max_windows, features="musical",
                 agg="mean", agg_q=0.25, agg_temp=1.0, unc_nets=0.0, unc_win=0.0):
        self.names = names; self.keep = keep; self.features = features
        self.agg, self.agg_q, self.agg_temp, self.unc_nets, self.unc_win = agg, agg_q, agg_temp, unc_nets, unc_win
        self.layers = list(layers); self.dropout = dropout; self.l2 = l2
        self.window, self.stride, self.max_windows = window, stride, max_windows
        self.mu = self.sd = None; self.nets = []; self.platt = (1.0, 0.0)
        self.history = []   # [{fecha, epochs, n_files, ...}]

    # --- normalización robusta ---
    def fit_norm(self, X):
        Z = X[:, self.keep]
        self.mu = np.median(Z, 0); q = np.percentile(Z, [10, 90], 0)
        self.sd = np.maximum((q[1] - q[0]) / 2.563, Z.std(0) * 0.25 + 1e-6)

    def norm(self, X):
        return np.clip((X[:, self.keep] - self.mu) / self.sd, -6, 6)

    def _net_logits(self, X):
        Z = self.norm(X)
        return np.array([n.logit(Z) for n in self.nets])          # [redes × ventanas]

    def window_logits(self, X):
        return self._net_logits(X).mean(0)

    def file_score(self, X):
        """Puntuación de la pieza (>0 ⇒ «reconstructed», mayor = peor).
        agg='worst': media del q·100 % de ventanas más «reconstructed» (cola mala; una ventana horrible no se diluye).
        agg='lse'  : soft-max suave (log-mean-exp con temperatura agg_temp). agg='mean': media clásica.
        + unc_nets·(desacuerdo medio entre redes) + unc_win·(desviación entre ventanas) como penalización de incertidumbre."""
        L = self._net_logits(X); wl = L.mean(0)
        if self.agg == "worst":
            k = max(1, int(math.ceil(self.agg_q * len(wl)))); s = float(np.sort(wl)[-k:].mean())
        elif self.agg == "lse":
            t = max(self.agg_temp, 1e-6); s = float((np.logaddexp.reduce(t * wl) - math.log(len(wl))) / t)
        else:
            s = float(wl.mean())
        if self.unc_nets and L.shape[0] > 1: s += self.unc_nets * float(L.std(0).mean())
        if self.unc_win and len(wl) > 1: s += self.unc_win * float(wl.std())
        return s

    def prob(self, score):
        a, b = self.platt
        return float(_sig(a * score + b))

    # --- guardar / cargar ---
    def save(self, path):
        meta = dict(version=VERSION, names=self.names, keep=[int(k) for k in self.keep], layers=self.layers, dropout=self.dropout,
                    l2=self.l2, window=self.window, stride=self.stride, max_windows=self.max_windows, features=self.features, platt=list(self.platt),
                    agg=self.agg, agg_q=self.agg_q, agg_temp=self.agg_temp, unc_nets=self.unc_nets, unc_win=self.unc_win,
                    n_nets=len(self.nets), history=self.history, labels=list(LABELS))
        arr = {"meta": np.array(json.dumps(meta)), "mu": self.mu, "sd": self.sd}
        for i, n in enumerate(self.nets):
            arr.update({f"n{i}_{k}": v for k, v in n.state().items()})
        np.savez_compressed(_ensure_parent(path), **arr)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        o = cls(m["names"], m["keep"], m["layers"], m["dropout"], m["l2"], m["window"], m["stride"], m["max_windows"], m.get("features", "forense"),
                agg=m.get("agg", "mean"), agg_q=m.get("agg_q", 0.25), agg_temp=m.get("agg_temp", 1.0),
                unc_nets=m.get("unc_nets", 0.0), unc_win=m.get("unc_win", 0.0))
        o.mu, o.sd = z["mu"], z["sd"]; o.platt = tuple(m["platt"]); o.history = m.get("history", [])
        sizes = [len(m["keep"])] + m["layers"] + [1]
        o.nets = [MLP.from_state(z, f"n{i}_", sizes, m["dropout"], m["l2"], seed=i) for i in range(m["n_nets"])]
        return o

def _stack(ds, idx):
    X = np.vstack([ds[i]["X"] for i in idx])
    y = np.concatenate([np.full(len(ds[i]["X"]), ds[i]["y"], float) for i in idx])
    f = np.concatenate([np.full(len(ds[i]["X"]), k) for k, i in enumerate(idx)])
    return X, y, f

def _weights(y, f, n_files, kd=None, synth_w=0.35):
    """pesos: cada FICHERO pesa igual dentro de su clase y las dos clases pesan igual.
    Dentro de la clase «reconstructed», los negativos reales (recetas) suman 1-synth_w y los sintéticos synth_w,
    repartido a partes iguales entre TIPOS de corrupción (así ningún tipo domina por tener más ficheros)."""
    cnt = np.bincount(f.astype(int), minlength=n_files).astype(float)
    w0 = 1.0 / cnt[f.astype(int)]; w = w0.copy()
    kd = np.full(len(y), "") if kd is None else np.asarray(kd)
    m0 = y == 0
    if m0.any(): w[m0] *= 0.5 / w0[m0].sum()
    m1 = y == 1
    if m1.any():
        real = m1 & (kd == ""); kinds = sorted(set(kd[m1 & (kd != "")].tolist()))
        if real.any() and kinds: s_real, s_syn = 0.5 * (1 - synth_w), 0.5 * synth_w
        elif real.any(): s_real, s_syn = 0.5, 0.0
        else: s_real, s_syn = 0.0, 0.5
        if real.any(): w[real] *= s_real / w0[real].sum()
        for k in kinds:
            mk = m1 & (kd == k); w[mk] *= (s_syn / len(kinds)) / w0[mk].sum()
    return w * len(y)

def train_detector(ds, names, args, det=None, idx=None, verbose=False):
    """Entrena (o continúa) un Detector sobre ds[idx]."""
    idx = list(range(len(ds))) if idx is None else list(idx)
    X, y, f = _stack(ds, idx)
    kd = np.array([ds[i]["kind"] for i in idx for _ in range(len(ds[i]["X"]))])
    w = _weights(y, f, len(idx), kd, getattr(args, "synth_weight", 0.35))
    fresh = det is None
    if fresh:
        keep = drop_columns(names, args.drop_groups)
        det = Detector(names, keep, args.layers, args.dropout, args.l2, (args.window_bars if args.features == "musical" else args.window), args.stride, args.max_windows, args.features,
                       agg=getattr(args, "agg", "worst"), agg_q=getattr(args, "agg_q", 0.25), agg_temp=getattr(args, "agg_temp", 1.0),
                       unc_nets=getattr(args, "unc_nets", 0.0), unc_win=getattr(args, "unc_win", 0.0))
        det.fit_norm(X)
        det.nets = [MLP(len(keep), args.layers, seed=args.seed + 17 * k, dropout=args.dropout, l2=args.l2) for k in range(args.ensemble)]
    Z = det.norm(X)
    rs = np.random.RandomState(args.seed)
    last = []
    for k, net in enumerate(det.nets):
        if args.bootstrap and fresh:        # cada red ve un remuestreo de FICHEROS (diversidad)
            files = rs.choice(len(idx), len(idx), replace=True)
            sel = np.concatenate([np.where(f == q)[0] for q in files])
        else:
            sel = np.arange(len(y))
        h = net.fit(Z[sel], y[sel], w[sel], args.epochs, args.batch, args.lr, verbose and k == 0)
        last.append(h[-1])
    det.history.append(dict(epochs=args.epochs, batch=args.batch, lr=args.lr, files=len(idx), windows=int(len(y)),
                            final_loss=float(np.mean(last)), fresh=fresh, when=time.strftime("%Y-%m-%d %H:%M:%S")))
    return det

def predict_files(det, ds, idx):
    return np.array([det.file_score(ds[i]["X"]) for i in idx])

# ═════════════════════════ métricas ═════════════════════════
def auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    # empates: rangos medios
    allv = np.concatenate([pos, neg]); order = np.argsort(allv, kind="mergesort"); sr = allv[order]
    ranks = np.empty(len(allv)); i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

def fit_platt(s, y, iters=100):
    """Platt scaling p=σ(a·s+b) por Newton con retroceso (objetivos suavizados de Lin et al.)."""
    s = np.clip(np.asarray(s, float), -40, 40); y = np.asarray(y, float)
    npos, nneg = max(y.sum(), 1), max((1 - y).sum(), 1)
    t = np.where(y == 1, (npos + 1) / (npos + 2), 1 / (nneg + 2))
    def nll(a, b):
        z = a * s + b
        return float(np.sum(np.logaddexp(0, z) - t * z))
    a, b = 1.0, 0.0; cur = nll(a, b)
    for _ in range(iters):
        p = _sig(a * s + b); g = p - t; wv = p * (1 - p) + 1e-9
        ga, gb = (g * s).sum(), g.sum()
        haa, hab, hbb = (wv * s * s).sum() + 1e-6, (wv * s).sum(), wv.sum() + 1e-6
        det_ = haa * hbb - hab * hab
        if abs(det_) < 1e-12: break
        da, db = (hbb * ga - hab * gb) / det_, (haa * gb - hab * ga) / det_
        step = 1.0
        while step > 1e-6 and nll(a - step * da, b - step * db) > cur + 1e-12: step *= 0.5
        if step <= 1e-6: break
        a -= step * da; b -= step * db; new = nll(a, b)
        if abs(cur - new) < 1e-10: cur = new; break
        cur = new
    return float(a), float(b)

def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p); pr = (p >= 0.5).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    ll = float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))
    bins = np.linspace(0, 1, 6); ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi + (1e-9 if hi == 1 else 0))
        if m.any(): ece += m.mean() * abs(p[m].mean() - y[m].mean())
    rec = tp / max(tp + fn, 1); spe = tn / max(tn + fp, 1)
    return dict(acc=(tp + tn) / max(len(y), 1), bal_acc=(rec + spe) / 2, auc=auc(y, p), logloss=ll, brier=float(np.mean((p - y) ** 2)),
                ece=float(ece), tp=tp, tn=tn, fp=fp, fn=fn, recall_recon=rec, specificity=spe)

# ═════════════════════════ validación cruzada agrupada ═════════════════════════
def grouped_folds(ds, k, seed, paired_only=False):
    paired = sorted({d["gid"] for d in ds if (d["paired"] or not paired_only)})
    rs = np.random.RandomState(seed); rs.shuffle(paired)
    k = len(paired) if k <= 0 else min(k, len(paired))     # k<=0 → leave-one-piece-out
    return [set(paired[i::k]) for i in range(k)]

def run_cv(ds, names, args, groups_imp=False, log=True):
    folds = grouped_folds(ds, args.cv, args.seed, getattr(args, "paired_only", False))
    n = len(ds); oof = np.full(n, np.nan); oof_s = np.full(n, np.nan); imp = {g: [] for g in groups_in(names)}
    po = getattr(args, "paired_only", False)
    for fi, test_g in enumerate(folds):
        te = [i for i, d in enumerate(ds) if d["gid"] in test_g and _is_real(d) and (d["paired"] or not po)]
        ts = [i for i, d in enumerate(ds) if d["gid"] in test_g and d.get("kind") and (d["paired"] or not po)]   # sintéticos: solo diagnóstico
        tr = [i for i, d in enumerate(ds) if d["gid"] not in test_g]
        det = train_detector(ds, names, args, idx=tr)
        base = predict_files(det, ds, te); oof[te] = base
        if ts: oof_s[ts] = predict_files(det, ds, ts)
        if groups_imp:
            yb = np.array([ds[i]["y"] for i in te]); b_auc = auc(yb, base)
            rs = np.random.RandomState(args.seed + fi)
            for g in groups_in(names):
                cols = [j for j, c in enumerate(det.keep) if _group_of(names[c]) == g]
                if not cols: continue
                sc = []
                for rep in range(2):
                    sc_files = []
                    for i in te:      # sustituye el grupo por el de OTRO fichero de test al azar
                        j = te[rs.randint(len(te))]; Xj = ds[j]["X"]
                        Xp = ds[i]["X"].copy()
                        for c in cols:
                            Xp[:, det.keep[c]] = Xj[rs.randint(len(Xj), size=len(Xp)), det.keep[c]]
                        sc_files.append(det.file_score(Xp))
                    sc.append(auc(yb, np.array(sc_files)))
                imp[g].append(b_auc - float(np.mean(sc)))
        if log:
            print(f"    pliegue {fi+1}/{len(folds)}: {len(te)} ficheros de test, AUC {auc([ds[i]['y'] for i in te], base):.3f}", file=sys.stderr)
    mask = ~np.isnan(oof)
    y = np.array([d["y"] for d in ds])[mask]; s = oof[mask]
    a, b = fit_platt(s, y); p = _sig(a * s + b)
    mr, mc = metrics(y, _sig(s)), metrics(y, p)
    mr["auc"] = mc["auc"] = auc(y, s)                          # AUC sobre la puntuación bruta (invariante a la calibración)
    res = dict(n_files=int(mask.sum()), raw=mr, calibrated=mc, platt=(a, b))
    # acierto por pareja: ¿P(recon) de la reconstrucción > P(recon) del original?
    byg = {}
    for i in np.where(mask)[0]:
        byg.setdefault(ds[i]["gid"], {})[ds[i]["y"]] = oof[i]
    pr = [v[1] > v[0] for v in byg.values() if 0 in v and 1 in v]
    res["pair_acc"] = float(np.mean(pr)) if pr else float("nan"); res["n_pairs"] = len(pr)
    res["imp"] = {g: (float(np.mean(v)), float(np.std(v))) for g, v in imp.items() if v}
    res["oof"] = {ds[i]["path"]: float(oof[i]) for i in np.where(mask)[0]}
    # diagnóstico de sintéticos (fuera de pliegue): AUC original-vs-tipo y % detectado con la Platt de los datos reales
    orig_sc = oof[mask & (np.array([d["y"] for d in ds]) == 0)]; syn = {}
    for k in sorted({d["kind"] for d in ds if d.get("kind")}):
        ix = [i for i in np.where(~np.isnan(oof_s))[0] if ds[i]["kind"] == k]
        if not ix or not len(orig_sc): continue
        sk = oof_s[ix]
        syn[k] = (auc(np.r_[np.zeros(len(orig_sc)), np.ones(len(sk))], np.r_[orig_sc, sk]), float(np.mean(_sig(a * sk + b) >= 0.5)), len(ix))
    res["synth"] = syn
    return res


def _recipe(path):
    m = re.match(r"^\d+_\d+_(.+)\.mid$", os.path.basename(path)); return m.group(1) if m else None

def run_loro(ds, names, args, log=True):
    """Leave-one-recipe-out: cada familia de generador se evalúa con un modelo que NO la ha visto
    (ni sus piezas). Los originales sin pareja se reparten entre los pliegues."""
    rec = sorted({_recipe(d["path"]) for d in ds if d["y"] == 1 and _recipe(d["path"])})
    gid_rec = {d["gid"]: _recipe(d["path"]) for d in ds if d["y"] == 1 and _recipe(d["path"])}
    unp = sorted({d["gid"] for d in ds if d["gid"] not in gid_rec}); rs = np.random.RandomState(args.seed); rs.shuffle(unp)
    fold_of = {g: r for g, r in gid_rec.items()}
    for i, g in enumerate(unp): fold_of[g] = rec[i % len(rec)]
    oof = np.full(len(ds), np.nan)
    for r in rec:
        te = [i for i, d in enumerate(ds) if fold_of.get(d["gid"]) == r and _is_real(d)]
        tr = [i for i, d in enumerate(ds) if fold_of.get(d["gid"]) != r]
        det = train_detector(ds, names, args, idx=tr); oof[te] = predict_files(det, ds, te)
        if log:
            yt = np.array([ds[i]["y"] for i in te]); print(f"    receta «{r}»: {int((yt==1).sum())} recon + {int((yt==0).sum())} originales fuera", file=sys.stderr)
    mask = ~np.isnan(oof); y = np.array([d["y"] for d in ds])[mask]; sc = oof[mask]
    a, b = fit_platt(sc, y); out = dict(platt=(a, b), auc=auc(y, sc), m=metrics(y, _sig(a * sc + b)), per={})
    for r in rec:
        ix = [i for i in np.where(mask)[0] if ds[i]["y"] == 1 and _recipe(ds[i]["path"]) == r]
        out["per"][r] = (float(np.mean([_sig(a * oof[i] + b) >= 0.5 for i in ix])), len(ix))
    paired_o = [i for i in np.where(mask)[0] if ds[i]["y"] == 0 and ds[i]["paired"]]
    unp_o = [i for i in np.where(mask)[0] if ds[i]["y"] == 0 and not ds[i]["paired"]]
    out["spec_paired"] = float(np.mean([_sig(a * oof[i] + b) < 0.5 for i in paired_o])) if paired_o else float("nan")
    out["spec_unpaired"] = float(np.mean([_sig(a * oof[i] + b) < 0.5 for i in unp_o])) if unp_o else float("nan")
    out["n_unpaired"] = len(unp_o)
    return out

# ═════════════════════════ subcomandos ═════════════════════════
def _common_data_args(p):
    p.add_argument("--orig", nargs="*", default=[], help="MIDIs originales (rutas/globs/dirs)")
    p.add_argument("--recon", nargs="*", default=[], help="MIDIs reconstruidos (nombre NN_ID_receta.mid → agrupa con el original ID.mid; ID = nombre del original sin extensión)")
    p.add_argument("--features", choices=["musical", "forense"], default="musical",
                   help="musical (def): rasgos canonicalizados, independientes del origen · forense: huella del pipeline (metadatos, expresión)")
    p.add_argument("--window-bars", type=int, default=8, help="[musical] Ventana en compases")
    p.add_argument("--augment", type=int, default=0, help="[musical] Copias perturbadas (humanización sintética) por fichero para entrenar invariancia")
    p.add_argument("--synth", type=int, default=0, help="[musical] Negativos sintéticos por original (corrupciones, tipos en rotación). 0 = desactivado")
    p.add_argument("--synth-kinds", nargs="*", default=None, choices=SYNTH_KINDS, help="Tipos de corrupción a usar (def: todos)")
    p.add_argument("--window", type=float, default=32.0, help="[forense] Ventana en negras (def 32)")
    p.add_argument("--stride", type=float, default=None, help="Salto entre ventanas (def = window)")
    p.add_argument("--max-windows", type=int, default=40, help="Máx. ventanas por fichero")
    p.add_argument("--cache", default=".cache_ovr", help="Directorio de caché de rasgos ('' = sin caché)")
    p.add_argument("--drop-groups", nargs="*", default=[], choices=GROUPS + MGROUPS, help="Excluir grupos de rasgos (ablación)")
    p.add_argument("--verbose", "-v", action="store_true")

def _common_model_args(p):
    p.add_argument("--layers", type=int, nargs="*", default=[96, 48], help="Capas ocultas (vacío = logística). Def: 96 48")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--ensemble", type=int, default=5, help="Nº de redes del conjunto")
    p.add_argument("--bootstrap", action="store_true", help="Cada red entrena con un remuestreo de ficheros")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--synth-weight", type=float, default=0.35, help="Peso de los negativos sintéticos dentro de la clase «reconstructed» (repartido por tipo)")
    p.add_argument("--agg", choices=["mean", "worst", "lse"], default="worst", help="Agregación de ventanas por pieza: worst = media de la cola peor (def) · lse = soft-max · mean")
    p.add_argument("--agg-q", type=float, default=0.25, help="[worst] Fracción de ventanas peores que cuentan")
    p.add_argument("--agg-temp", type=float, default=1.0, help="[lse] Temperatura del soft-max")
    p.add_argument("--unc-nets", type=float, default=0.0, help="Penalización por desacuerdo entre redes del conjunto")
    p.add_argument("--unc-win", type=float, default=0.0, help="Penalización por dispersión entre ventanas")

def cmd_features(a):
    ds, names = build_dataset(a)
    print(f"{len(ds)} ficheros, {sum(len(d['X']) for d in ds)} ventanas, {len(names)} rasgos")
    cnt = {g: sum(1 for n in names if _group_of(n) == g) for g in groups_in(names)}
    print("rasgos por grupo:", ", ".join(f"{g}={c}" for g, c in cnt.items()))
    if a.json:
        json.dump(dict(names=names, files=[dict(path=d["path"], y=d["y"], gid=d["gid"], windows=len(d["X"]),
                                                 mean=d["X"].mean(0).round(5).tolist()) for d in ds]), open(_ensure_parent(a.json), "w"))
        print("→", a.json)

def cmd_train(a):
    ds, names = build_dataset(a)
    no, nr = sum(d["y"] == 0 for d in ds), sum(d["y"] == 1 for d in ds)
    print(f"Datos: {sum(d['y']==0 and not d['aug'] for d in ds)} originales + {sum(d['y']==1 and _is_real(d) for d in ds)} reconstruidos (+{sum(bool(d['aug']) for d in ds)} copias perturbadas, +{sum(bool(d['kind']) for d in ds)} negativos sintéticos) · {sum(len(d['X']) for d in ds)} ventanas · {len(names)} rasgos")
    det = None
    if a.init:
        det = Detector.load(a.init)
        if det.names != names:
            sys.exit("El modelo --init usa otros rasgos/ventana; no se puede continuar con estos datos.")
        print(f"Continuando {a.init}: {len(det.nets)} redes, capas {det.layers}, {sum(h['epochs'] for h in det.history)} épocas previas")
        if a.layers != [96, 48] and a.layers != det.layers:
            print("  [AVISO] --layers se ignora al continuar un modelo (la arquitectura es la del modelo).", file=sys.stderr)
    a.drop_groups = a.drop_groups if not det else [g for g in groups_in(names) if g not in {_group_of(names[k]) for k in det.keep}]
    t0 = time.time()
    if a.oof_calibration and (det is None or a.recalibrate):
        print(f"Calibración Platt por validación cruzada agrupada ({a.cv} pliegues)…")
        cva = argparse.Namespace(**vars(a)); cva.init = None
        r = run_cv(ds, names, cva, log=a.verbose)
        platt = r["platt"]
        print(f"  AUC OOF {r['raw']['auc']:.3f} · acc {r['raw']['acc']:.3f} · Platt a={platt[0]:.3f} b={platt[1]:.3f}")
    else:
        platt = det.platt if det else (1.0, 0.0)
    det = train_detector(ds, names, a, det=det, verbose=a.verbose)
    det.platt = platt
    det.save(_ensure_parent(a.out))
    idx = list(range(len(ds)))
    idx = [i for i, d in enumerate(ds) if _is_real(d)]
    s = predict_files(det, ds, idx); y = np.array([ds[i]["y"] for i in idx]); p = _sig(platt[0] * s + platt[1])
    m = metrics(y, p)
    print(f"Modelo guardado: {a.out}  ({time.time()-t0:.0f}s)  · redes {len(det.nets)} · capas {det.layers}")
    print(f"Ajuste sobre los propios datos de entrenamiento (OPTIMISTA, usa 'eval' para el real): acc {m['acc']:.3f} · AUC {m['auc']:.3f}")

def _fmt(m):
    return (f"acc {m['acc']:.3f}  acc.bal {m['bal_acc']:.3f}  AUC {m['auc']:.3f}  logloss {m['logloss']:.3f}  "
            f"Brier {m['brier']:.3f}  ECE {m['ece']:.3f}   [TP {m['tp']} TN {m['tn']} FP {m['fp']} FN {m['fn']}]")

def cmd_eval(a):
    ds, names = build_dataset(a)
    npair = len({d["gid"] for d in ds if d["paired"]})
    print(f"Datos: {sum(1 for d in ds if _is_real(d))} ficheros (+{sum(1 for d in ds if d['aug'])} copias perturbadas, +{sum(1 for d in ds if d['kind'])} sintéticos) ({npair} piezas emparejadas) · {len(names)} rasgos · CV agrupada k={a.cv} × {a.repeats}")
    if a.loro:
        r = run_loro(ds, names, a, log=a.verbose)
        print(f"\n▶ LEAVE-ONE-RECIPE-OUT (generador desconocido) · {len(r['per'])} familias · features={a.features}")
        print("   " + _fmt(r["m"]).replace(f"AUC {r['m']['auc']:.3f}", f"AUC {r['auc']:.3f}"))
        print(f"   especificidad: originales emparejados {r['spec_paired']:.3f} · originales sin pareja ({r['n_unpaired']}) {r['spec_unpaired']:.3f}")
        print("   sensibilidad por familia no vista:  " + "  ".join(f"{k} {v[0]:.2f}(n={v[1]})" for k, v in r["per"].items()))
        if a.json: json.dump(dict(loro=r), open(_ensure_parent(a.json), "w"), indent=1, default=float)
        return
    configs = [("principal", dict())]
    if a.compare:
        configs = [("logística (sin capas ocultas)", dict(layers=[], ensemble=1)),
                   ("MLP pequeña [32]", dict(layers=[32], ensemble=3)),
                   ("MLP principal", dict()),
                   ("MLP grande [192,96,48]", dict(layers=[192, 96, 48], ensemble=5, epochs=a.epochs))]
    if a.ablate:
        configs += [(f"sin grupo «{g}»", dict(drop_groups=[g])) for g in groups_in(names)]
    out = []
    for name, over in configs:
        accs = []; last = None; imp0 = None
        for r in range(a.repeats):
            b = argparse.Namespace(**vars(a)); b.seed = a.seed + 101 * r
            for k, v in over.items(): setattr(b, k, v)
            last = run_cv(ds, names, b, groups_imp=(a.importance and name in ("principal", "MLP principal") and r == 0), log=a.verbose)
            accs.append(last)
            if r == 0: imp0 = last.get("imp")
        keys = ["acc", "bal_acc", "auc", "logloss", "brier", "ece"]
        mean = {k: float(np.mean([x["calibrated"][k] for x in accs])) for k in keys}
        mean.update({k: int(round(np.mean([x["calibrated"][k] for x in accs]))) for k in ("tp", "tn", "fp", "fn")})
        sd_auc = float(np.std([x["calibrated"]["auc"] for x in accs])); pair = float(np.mean([x["pair_acc"] for x in accs]))
        raw_auc = float(np.mean([x["raw"]["auc"] for x in accs]))
        print(f"\n▶ {name}   capas={over.get('layers', a.layers)} ens={over.get('ensemble', a.ensemble)}")
        print("   " + _fmt(mean) + f"   (±{sd_auc:.3f} AUC entre repeticiones)")
        print(f"   acierto por pareja (la reconstrucción puntúa más que su original): {pair:.3f} sobre {accs[0]['n_pairs']} parejas")
        if accs[0].get("synth"):
            print("   negativos sintéticos fuera de pliegue (AUC original-vs-tipo · % detectado con la calibración real):")
            for k, (sa, sr, sn) in accs[0]["synth"].items():
                print(f"     {k:<14s} AUC {sa:.3f} · detectado {sr:.2f} (n={sn})")
        if imp0:
            print("   importancia por grupo (caída de AUC al permutar el grupo):")
            for g, (mu, sd) in sorted(imp0.items(), key=lambda kv: -kv[1][0]):
                print(f"     {g:<11s} {mu:+.3f} ± {sd:.3f}")
        out.append(dict(name=name, mean=mean, auc_sd=sd_auc, pair_acc=pair, raw_auc=raw_auc, imp=imp0, platt=last["platt"]))
        if name in ("principal", "MLP principal"):
            res_last = last
    if a.json:
        json.dump(out, open(_ensure_parent(a.json), "w"), indent=1); print("\n→", a.json)
    if a.errors and 'res_last' in locals():
        print("\nErrores del modelo principal (puntuación OOF, >0 ⇒ «reconstructed»):")
        for pth, s in res_last["oof"].items():
            lab = 1 if re.match(r"^\d+_\d+_", os.path.basename(pth)) else 0
            if (s > 0) != (lab == 1): print(f"   {'recon' if lab else 'orig ':5s}  score {s:+.2f}  {os.path.basename(pth)}")

def cmd_predict(a):
    det = Detector.load(a.model)
    files = _expand(a.files)
    if not files: sys.exit("Sin ficheros que clasificar")
    res = []
    for p in files:
        try:
            X, nm = cached_windows(p, det.window, det.stride, det.max_windows, a.cache, det.features)
            if len(X) == 0: raise ValueError("sin ventanas útiles (¿pocas notas?)")
            if nm != det.names: raise ValueError("rasgos incompatibles con el modelo")
            wl = det.window_logits(X); s = det.file_score(X); pr = det.prob(s)
            res.append(dict(file=p, label=LABELS[int(pr >= 0.5)], p_reconstructed=round(pr, 4), p_original=round(1 - pr, 4), calibrated=bool(tuple(det.platt) != (1.0, 0.0)),
                            score=round(s, 3), score_mean=round(float(wl.mean()), 3), windows=int(len(X)), window_agreement=round(float(np.mean((wl > 0) == (s > 0))), 3)))
        except Exception as e:
            res.append(dict(file=p, error=str(e)))
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1)); return
    for r in res:
        if "error" in r: print(f"{r['file']}: ERROR {r['error']}"); continue
        col = "31" if r["label"] == "reconstructed" else "32"
        print(f"{_C.c(r['label'].upper(), col):<24s} P(reconstructed)={r['p_reconstructed']:.3f}  P(original)={r['p_original']:.3f}  "
              f"[{r['windows']} ventanas, acuerdo {r['window_agreement']:.2f}]  {r['file']}")

def cmd_info(a):
    det = Detector.load(a.model)
    print(f"Modelo {a.model} · versión {VERSION}")
    print(f"  Rasgos usados: {len(det.keep)}/{len(det.names)} · ventana {det.window:g} {'compases' if det.features == 'musical' else 'negras'} · modo {det.features} · máx {det.max_windows} ventanas/fichero")
    gs = {}
    for k in det.keep: gs[_group_of(det.names[k])] = gs.get(_group_of(det.names[k]), 0) + 1
    print("  Por grupo: " + ", ".join(f"{g}={c}" for g, c in gs.items()))
    npar = sum(w.size + b.size for n in det.nets for w, b in zip(n.W, n.b))
    print(f"  Conjunto: {len(det.nets)} redes · capas ocultas {det.layers} · {npar:,} parámetros · dropout {det.dropout} · L2 {det.l2}")
    print(f"  Calibración Platt: a={det.platt[0]:.3f} b={det.platt[1]:.3f}")
    print(f"  Agregación por pieza: {det.agg}" + (f" (q={det.agg_q})" if det.agg == "worst" else f" (T={det.agg_temp})" if det.agg == "lse" else "")
          + f" · penal. redes {det.unc_nets} · penal. ventanas {det.unc_win}")
    for i, h in enumerate(det.history):
        print(f"  [{i}] {h['when']}  {'desde cero' if h['fresh'] else 'continuación'}  épocas {h['epochs']}  lote {h['batch']}  lr {h['lr']}  "
              f"ficheros {h['files']}  ventanas {h['windows']}  pérdida final {h['final_loss']:.4f}")

def main():
    ap = argparse.ArgumentParser(prog="original_vs_reconstruido.py", description="Discriminador original / reconstructed para MIDIs (numpy + mido).")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("features", help="Extrae y cachea rasgos"); _common_data_args(p); p.add_argument("--json"); p.set_defaults(f=cmd_features)
    p = sp.add_parser("train", help="Entrena / continúa"); _common_data_args(p); _common_model_args(p)
    p.add_argument("--out", required=True, help="Modelo de salida (.npz)")
    p.add_argument("--init", help="Modelo previo desde el que continuar")
    p.add_argument("--cv", type=int, default=5, help="Pliegues para la calibración OOF")
    p.add_argument("--no-oof-calibration", dest="oof_calibration", action="store_false")
    p.add_argument("--recalibrate", action="store_true", help="Recalibrar Platt (OOF desde cero) al continuar")
    p.set_defaults(f=cmd_train)
    p = sp.add_parser("eval", help="Validación cruzada agrupada por pieza"); _common_data_args(p); _common_model_args(p)
    p.add_argument("--cv", type=int, default=5); p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--loo-piece", dest="cv", action="store_const", const=0, help="Leave-one-piece-out: un pliegue por pieza (equivale a --cv <nº de piezas>); puntuaciones OOF por fichero en --json ('oof')")
    p.add_argument("--paired-only", action="store_true", help="Solo evalúa las piezas con original+reconstrucción")
    p.add_argument("--loro", action="store_true", help="Leave-one-recipe-out: deja fuera cada familia de generador (origen desconocido)")
    p.add_argument("--compare", action="store_true", help="Compara logística / MLP pequeña / principal / grande")
    p.add_argument("--ablate", action="store_true", help="Ablación por grupo de rasgos")
    p.add_argument("--importance", action="store_true", help="Importancia por permutación de grupos")
    p.add_argument("--errors", action="store_true", help="Lista los ficheros mal clasificados")
    p.add_argument("--json"); p.set_defaults(f=cmd_eval)
    p = sp.add_parser("predict", help="Clasifica MIDIs"); p.add_argument("files", nargs="+"); p.add_argument("--model", required=True)
    p.add_argument("--cache", default=".cache_ovr"); p.add_argument("--json", action="store_true"); p.set_defaults(f=cmd_predict)
    p = sp.add_parser("info", help="Resumen del modelo"); p.add_argument("--model", required=True); p.set_defaults(f=cmd_info)
    a = ap.parse_args()
    if getattr(a, "cache", None) == "": a.cache = None
    a.f(a)

if __name__ == "__main__":
    main()
