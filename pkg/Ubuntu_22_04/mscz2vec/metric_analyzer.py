#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      METRIC ANALYZER  v1.0                                  ║
║        Análisis de la estructura métrica y la síncopa de una obra MIDI      ║
║                                                                              ║
║  La suite de análisis del ecosistema mide altura, armonía, emoción, forma   ║
║  y tocabilidad (analyzer, quality_scorer, section_score, arc_supervisor,    ║
║  playability_auditor), pero ninguna herramienta standalone diagnostica el   ║
║  eje RÍTMICO-MÉTRICO: cuánto sincopa la obra, dónde pelea contra el pulso,   ║
║  cómo se agrupa en hipercompases y si su ritmo es plano o variado. Esta      ║
║  herramienta cubre ese hueco. midi_pianoroll_analyzer muestra densidad y    ║
║  tensión por compás pero no la jerarquía métrica ni la síncopa.             ║
║                                                                              ║
║  MÉTRICAS CALCULADAS:                                                        ║
║    síncopa (LHL)   — nota en posición débil sostenida sobre una fuerte      ║
║                      (Longuet-Higgins & Lee), ponderada por peso métrico    ║
║    densidad        — onsets por compás                                       ║
║    fuerza downbeat — proporción de onsets sobre el tiempo 1                 ║
║    nPVI            — variabilidad rítmica de los IOI (plano ↔ irregular)    ║
║    perfil métrico  — histograma de onsets por posición dentro del compás    ║
║    hipermetro      — periodicidad de la acentuación (grupos de 2/4/8)       ║
║    estabilidad     — cambios de compás/tempo a lo largo de la obra          ║
║                                                                              ║
║  MODOS DE SALIDA:                                                            ║
║    (por defecto)   — informe de texto con sparklines por compás             ║
║    --json FILE     — vuelca todas las métricas a un sidecar JSON            ║
║    --csv FILE      — tabla por compás (compás, densidad, síncopa, downbeat) ║
║                                                                              ║
║  USO:                                                                        ║
║    python metric_analyzer.py obra.mid                                        ║
║    python metric_analyzer.py obra.mid --subdiv 4                            ║
║    python metric_analyzer.py obra.mid --track 0                             ║
║    python metric_analyzer.py obra.mid --json obra.metric.json --csv m.csv   ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi              MIDI de entrada                                         ║
║    --track N         Analizar solo la pista N (default: todas agregadas)    ║
║    --subdiv N        Subdivisiones por tiempo para la rejilla (default: 4)  ║
║    --json FILE       Escribe un sidecar JSON con todas las métricas         ║
║    --csv FILE        Escribe una tabla CSV por compás                        ║
║    --max-bars N      Limita la tabla impresa a N compases (default: 32)     ║
║    --no-color        Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from metric_analyzer import analyze_meter                                 ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np

try:
    from playability_auditor import read_midi, TimeContext, extract_notes
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_SPARK = "▁▂▃▄▅▆▇█"


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def sparkline(v) -> str:
    v = np.asarray(v, dtype=float)
    if v.size == 0:
        return ""
    if v.max() <= v.min():
        return _SPARK[0] * len(v)
    x = (v - v.min()) / (v.max() - v.min())
    return "".join(_SPARK[min(7, int(round(t * 7)))] for t in x)


# ══════════════════════════════════════════════════════════════════════════════
#  JERARQUÍA MÉTRICA
# ══════════════════════════════════════════════════════════════════════════════

def metric_weights(num: int, subdiv: int) -> np.ndarray:
    """Peso métrico por slot (mayor = más fuerte), estilo Lerdahl-Jackendoff.
    L = num*subdiv slots por compás. El tiempo 1 recibe el peso máximo."""
    L = max(1, num * subdiv)
    levels = [L, subdiv]              # compás entero + cada tiempo
    step = subdiv
    while step % 2 == 0 and step > 1:
        step //= 2
        levels.append(step)
    levels.append(1)
    W = np.zeros(L)
    for i in range(L):
        W[i] = sum(1 for s in set(levels) if i % s == 0)
    return W


def _onset_grid(onsets_ticks, tc: TimeContext, bar: int, num: int,
                subdiv: int) -> np.ndarray:
    """Rejilla binaria de onsets del compás (1-based). Longitud num*subdiv."""
    L = num * subdiv
    t0, t1 = tc.bar_range_ticks(bar, bar)
    bar_len = max(1, t1 - t0)
    slot_ticks = bar_len / L
    grid = np.zeros(L, dtype=int)
    for t in onsets_ticks:
        if t0 <= t < t1:
            slot = int(round((t - t0) / slot_ticks)) % L
            grid[slot] = 1
    return grid


def bar_syncopation(grid: np.ndarray, W: np.ndarray) -> float:
    """Síncopa del compás (Longuet-Higgins & Lee): una nota en un slot débil
    que suena por encima de un slot métricamente más fuerte y vacío."""
    L = len(grid)
    onset_slots = [i for i in range(L) if grid[i]]
    if not onset_slots:
        return 0.0
    total = 0.0
    for k, i in enumerate(onset_slots):
        # la nota suena hasta el siguiente onset (o fin de compás)
        nxt = onset_slots[k + 1] if k + 1 < len(onset_slots) else L
        span = range(i + 1, nxt)
        if not span:
            continue
        max_w = max((W[m] for m in span), default=W[i])
        if max_w > W[i]:
            total += max_w - W[i]
    return float(total)


def npvi(iois: List[float]) -> float:
    """Normalized Pairwise Variability Index de los IOI (0=plano)."""
    d = [x for x in iois if x > 0]
    if len(d) < 2:
        return 0.0
    s = 0.0
    for a, b in zip(d[:-1], d[1:]):
        denom = (a + b) / 2
        if denom > 0:
            s += abs(a - b) / denom
    return 100.0 * s / (len(d) - 1)


def hypermeter(density: np.ndarray) -> Dict:
    """Periodicidad de la densidad por compás vía autocorrelación (lags 2/4/8)."""
    x = np.asarray(density, dtype=float)
    if len(x) < 4:
        return {"period": None, "strength": 0.0}
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 1e-9:
        return {"period": None, "strength": 0.0}
    best_lag, best_val = None, 0.0
    for lag in (2, 3, 4, 8):
        if lag < len(x):
            ac = np.dot(x[:-lag], x[lag:]) / denom
            if ac > best_val:
                best_val, best_lag = ac, lag
    return {"period": best_lag, "strength": round(float(best_val), 3)}


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def analyze_meter(path: str, track: Optional[int] = None,
                  subdiv: int = 4) -> dict:
    """API pública. Devuelve un diccionario con todas las métricas."""
    mid = read_midi(path)
    tc = TimeContext(mid)

    # recolección de onsets (agregados o por pista)
    all_onsets: List[int] = []
    durations: List[int] = []
    if track is not None:
        if not (0 <= track < len(mid.tracks)):
            raise ValueError(f"pista {track} fuera de rango (0..{len(mid.tracks)-1})")
        src_tracks = [mid.tracks[track]]
    else:
        src_tracks = mid.tracks
    for trk in src_tracks:
        for n in extract_notes(trk):
            all_onsets.append(n.start)
            durations.append(n.end - n.start)
    if not all_onsets:
        raise ValueError("el MIDI (o la pista indicada) no contiene notas")

    last = max(n_end for n_end in
               (n.end for trk in src_tracks for n in extract_notes(trk)))
    n_bars = max(1, tc.bar(last - 1))

    # compás base para la rejilla (usa el primer time signature)
    num = mid.timesig_map[0][1]
    W = metric_weights(num, subdiv)
    w_max = float(W.max()) if W.size else 1.0

    per_bar_sync = np.zeros(n_bars)
    per_bar_dens = np.zeros(n_bars)
    per_bar_down = np.zeros(n_bars)     # 1 si hay onset en el tiempo 1
    profile = np.zeros(num * subdiv)     # histograma de posición métrica
    onset_ticks = sorted(all_onsets)

    for bar in range(1, n_bars + 1):
        grid = _onset_grid(onset_ticks, tc, bar, num, subdiv)
        per_bar_dens[bar - 1] = grid.sum()
        per_bar_down[bar - 1] = 1.0 if grid[0] else 0.0
        profile += grid
        raw = bar_syncopation(grid, W)
        # normalizada por el nº de onsets del compás y el peso máximo
        n_on = max(1, grid.sum())
        per_bar_sync[bar - 1] = raw / (n_on * w_max)

    # IOI globales (en beats) para nPVI
    iois = np.diff(onset_ticks) / tc.tpb
    npvi_val = npvi(list(iois))

    # métricas globales
    total_onsets = len(onset_ticks)
    downbeat_strength = float(per_bar_down.mean())
    global_sync = float(per_bar_sync.mean())
    hyper = hypermeter(per_bar_dens)
    ts_changes = len(mid.timesig_map) - 1
    tempo_changes = len(mid.tempo_map) - 1

    # perfil métrico normalizado
    prof_norm = (profile / profile.sum()).tolist() if profile.sum() else profile.tolist()

    # etiqueta cualitativa de síncopa
    if global_sync < 0.05:
        sync_label = "muy recto"
    elif global_sync < 0.15:
        sync_label = "levemente sincopado"
    elif global_sync < 0.30:
        sync_label = "sincopado"
    else:
        sync_label = "muy sincopado"

    return {
        "file": path,
        "n_bars": int(n_bars),
        "time_signature": f"{num}/{mid.timesig_map[0][2]}",
        "subdiv": subdiv,
        "total_onsets": total_onsets,
        "global_syncopation": round(global_sync, 4),
        "syncopation_label": sync_label,
        "downbeat_strength": round(downbeat_strength, 4),
        "npvi": round(npvi_val, 2),
        "hypermeter": hyper,
        "ts_changes": ts_changes,
        "tempo_changes": tempo_changes,
        "metric_profile": [round(x, 4) for x in prof_norm],
        "metric_weights": [int(x) for x in W.tolist()],
        "per_bar": {
            "syncopation": [round(float(x), 4) for x in per_bar_sync],
            "density": [int(x) for x in per_bar_dens],
            "downbeat": [int(x) for x in per_bar_down],
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SALIDAS
# ══════════════════════════════════════════════════════════════════════════════

def _write_csv(st: dict, path: str):
    rows = ["compas,densidad,sincopa,downbeat"]
    pb = st["per_bar"]
    for i in range(st["n_bars"]):
        rows.append(f"{i+1},{pb['density'][i]},{pb['syncopation'][i]:.4f},"
                    f"{pb['downbeat'][i]}")
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="metric_analyzer.py",
        description="Análisis de estructura métrica y síncopa de un MIDI.")
    ap.add_argument("midi")
    ap.add_argument("--track", type=int)
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--json")
    ap.add_argument("--csv")
    ap.add_argument("--max-bars", type=int, default=32)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    scope = f"pista {args.track}" if args.track is not None else "todas las pistas"
    print(f"  {B}METRIC ANALYZER v{VERSION}  ·  {args.midi}  ({scope}){R}")
    print(f"{'═' * 78}")

    try:
        st = analyze_meter(args.midi, track=args.track, subdiv=args.subdiv)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    hyp = st["hypermeter"]
    hyp_txt = (f"grupos de {hyp['period']} (fuerza {hyp['strength']:.2f})"
               if hyp["period"] else "sin periodicidad clara")
    print(f"  Compás: {st['time_signature']}   compases: {st['n_bars']}   "
          f"onsets: {st['total_onsets']}   rejilla: 1/{st['subdiv']} de tiempo")
    print(f"  Síncopa global : {C}{st['global_syncopation']:.3f}{R}  "
          f"({st['syncopation_label']})")
    print(f"  Fuerza downbeat: {st['downbeat_strength']:.2f}   "
          f"nPVI: {st['npvi']:.1f}   hipermetro: {hyp_txt}")
    if st["ts_changes"] or st["tempo_changes"]:
        print(f"  Estabilidad    : {st['ts_changes']} cambios de compás, "
              f"{st['tempo_changes']} cambios de tempo")

    # perfil métrico (una barra por posición dentro del compás)
    prof = np.array(st["metric_profile"])
    print(f"  Perfil métrico : {sparkline(prof)}  (t.1 … fin del compás)")

    # sparklines por compás
    dens = np.array(st["per_bar"]["density"])
    sync = np.array(st["per_bar"]["syncopation"])
    print(f"  densidad/comp. : {sparkline(dens)}")
    print(f"  síncopa/comp.  : {sparkline(sync)}")

    # tabla por compás (limitada)
    nb = min(st["n_bars"], args.max_bars)
    print(f"\n  {G}compás   densidad   síncopa   downbeat{R}")
    pb = st["per_bar"]
    for i in range(nb):
        flag = "•" if pb["downbeat"][i] else " "
        bar_sync = pb["syncopation"][i]
        mark = _c("RED") if bar_sync >= 0.30 else (_c("YEL") if bar_sync >= 0.15 else "")
        print(f"  {i+1:>5}   {pb['density'][i]:>8}   "
              f"{mark}{bar_sync:>7.3f}{R}   {flag:>7}")
    if st["n_bars"] > nb:
        print(f"  … ({st['n_bars'] - nb} compases más; usa --csv para la tabla completa)")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"version": VERSION, **st}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n  {C}JSON: {args.json}{R}")
    if args.csv:
        _write_csv(st, args.csv)
        print(f"  {C}CSV: {args.csv}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
