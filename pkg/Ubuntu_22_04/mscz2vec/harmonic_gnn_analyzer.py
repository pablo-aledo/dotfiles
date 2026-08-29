#!/usr/bin/env python3
"""
║                    HARMONIC GNN ANALYZER  v1.0                              ║
║   Análisis armónico por grafo de notas (simultaneidad + sucesión)           ║
║                                                                              ║
║  Complementa a harmonic_analyzer.py. Ese analiza compás a compás con        ║
║  plantillas de acordes sobre un histograma de alturas "plano" (una sola     ║
║  secuencia lineal). Esta herramienta, en cambio, modela la pieza como un    ║
║  GRAFO DE NOTAS con dos tipos de arista — como describe la Parte III.10     ║
║  de la teoría de GNNs aplicada a música:                                    ║
║                                                                              ║
║    · arista de SIMULTANEIDAD — dos notas suenan a la vez (verticalidad,     ║
║      "esto es un acorde")                                                   ║
║    · arista de SUCESIÓN       — una nota sigue a otra en la misma voz       ║
║      (horizontalidad, "esto es una línea melódica")                         ║
║                                                                              ║
║  Un histograma plano por compás no distingue "cuatro notas simultáneas      ║
║  = un acorde" de "cuatro notas en sucesión rápida = un arpegio o una        ║
║  floritura melódica que pasa por esas alturas sin ser ellas el acorde".     ║
║  El grafo sí: al propagar información entre vecinos (varias rondas de       ║
║  "message passing", igual que una capa de convolución de grafo) el         ║
║  contexto armónico de cada nota emerge de mezclar lo que sueña con ella     ║
║  Y lo que la rodea en el tiempo, no solo lo que cae dentro de una ventana   ║
║  temporal arbitraria.                                                       ║
║                                                                              ║
║  Efecto práctico: mejor desambiguación de compases con notas de adorno,     ║
║  arpegios o texturas no-en-bloque, y una nueva capacidad que                ║
║  harmonic_analyzer.py no ofrece — clasificar CADA NOTA individual como      ║
║  "estructural" (nota del acorde) u "ornamental" (nota de paso/bordadura/    ║
║  apoyatura), útil como entrada previa a schenkerian_reducer.py.             ║
║                                                                              ║
║  MODOS:                                                                     ║
║    graph           — construye el grafo de notas y lo exporta (JSON)       ║
║    analyze          — análisis armónico compás a compás vía GNN            ║
║    classify-notes   — nota a nota: estructural vs. ornamental              ║
║    compare          — GNN (con contexto) vs. baseline (solo el compás)     ║
║                                                                              ║
║  PIPELINE (analyze):                                                        ║
║  [1] NOTAS        — extrae notas del MIDI (pitch, inicio/fin en tiempos)    ║
║  [2] GRAFO         — construye aristas de simultaneidad y de sucesión       ║
║  [3] PROPAGACIÓN   — K rondas de message passing sobre croma 12-dim         ║
║  [4] AGREGACIÓN    — combina los vectores de nota por compás                ║
║  [5] TONALIDAD     — Krumhansl-Schmuckler sobre el croma global             ║
║  [6] PLANTILLAS    — empareja cada compás con la plantilla de acorde más    ║
║                      cercana (tríadas y tétradas, 12 raíces)                 ║
║  [7] ROMANOS       — numeral + función (T/S/D) respecto a la tonalidad      ║
║  [8] INFORME       — tabla por compás + JSON opcional                       ║
║                                                                              ║
║  USO:                                                                        ║
║    python harmonic_gnn_analyzer.py analyze obra.mid                         ║
║    python harmonic_gnn_analyzer.py analyze obra.mid --key Cmaj              ║
║    python harmonic_gnn_analyzer.py analyze obra.mid --layers 3 --explain    ║
║    python harmonic_gnn_analyzer.py analyze obra.mid --json obra.gnn.json    ║
║    python harmonic_gnn_analyzer.py graph obra.mid --export-graph obra.g.json║
║    python harmonic_gnn_analyzer.py classify-notes obra.mid                  ║
║    python harmonic_gnn_analyzer.py classify-notes obra.mid --explain        ║
║    python harmonic_gnn_analyzer.py compare obra.mid                         ║
║    python harmonic_gnn_analyzer.py compare obra.mid --only-changed          ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi                 MIDI de entrada                                     ║
║    --key KEY             Fuerza la tonalidad (p.ej. Cmaj, Am, F#min)        ║
║    --beats-per-bar N     Pulsos por compás (default: 4)                     ║
║    --window N            Compases por ventana de análisis (default: 1)      ║
║    --layers K            Rondas de propagación en el grafo (default: 2)     ║
║    --w-self F            Peso del propio nodo en cada ronda (default: 0.34) ║
║    --w-simul F           Peso de vecinos por simultaneidad (default: 0.44)  ║
║    --w-succ F            Peso de vecinos por sucesión (default: 0.22)       ║
║    --track N             Analiza solo la pista N (default: todas)           ║
║    --explain             Añade una explicación por compás/nota              ║
║    --json FILE           Escribe un sidecar JSON con el análisis completo   ║
║    --export-graph FILE   (modo graph) Escribe el grafo en JSON              ║
║    --only-changed        (modo compare) Solo muestra compases en desacuerdo ║
║    --max-bars N          Limita la tabla impresa (default: 64)              ║
║    --no-color            Desactivar colores ANSI                            ║
║    --quiet               Salida mínima (útil en scripts)                    ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from harmonic_gnn_analyzer import (                                      ║
║        extract_notes, build_graph, propagate, analyze_harmony_gnn,          ║
║        classify_notes,                                                      ║
║    )                                                                        ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    (stdout)                 tabla por compás/nota + informe                 ║
║    --json FILE              análisis completo en JSON                       ║
║    --export-graph FILE      nodos + aristas del grafo en JSON               ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    Siempre:   mido, numpy                                                    ║
║    Integración conceptual: harmonic_analyzer.py (mismo repertorio de         ║
║    plantillas/numerales), schenkerian_reducer.py (consume classify-notes)    ║
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mido

# ─────────────────────────────────────────────────────────────────────────
#  Colores ANSI (desactivables con --no-color)
# ─────────────────────────────────────────────────────────────────────────

class _C:
    ON = True
    HDR = "\033[1;36m"
    DIM = "\033[2m"
    OK = "\033[1;32m"
    WARN = "\033[1;33m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def wrap(cls, s, code):
        return f"{code}{s}{cls.RESET}" if cls.ON else s


PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Perfiles de Krumhansl-Schmuckler (mayor / menor)
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Plantillas de acorde: pitch-classes relativas a la raíz, y símbolo/calidad
CHORD_TEMPLATES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "hdim7": [0, 3, 6, 10],
    "dim7": [0, 3, 6, 9],
}

# Grados diatónicos mayor/menor -> numeral romano + función (T/S/D)
DIATONIC_MAJOR = {0: ("I", "T"), 2: ("ii", "S"), 4: ("iii", "T"), 5: ("IV", "S"),
                  7: ("V", "D"), 9: ("vi", "T"), 11: ("vii°", "D")}
DIATONIC_MINOR = {0: ("i", "T"), 2: ("ii°", "S"), 3: ("III", "T"), 5: ("iv", "S"),
                  7: ("v", "D"), 8: ("VI", "T"), 10: ("VII", "D")}


# ─────────────────────────────────────────────────────────────────────────
#  Estructuras
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Note:
    idx: int
    pitch: int
    start: float          # en pulsos (beats)
    end: float             # en pulsos (beats)
    track: int
    velocity: int

    @property
    def dur(self) -> float:
        return max(self.end - self.start, 1e-6)

    @property
    def pc(self) -> int:
        return self.pitch % 12

    @property
    def bar(self) -> int:
        return None  # se asigna externamente según beats_per_bar


@dataclass
class Graph:
    notes: list
    edges_simul: list = field(default_factory=list)   # lista de (i, j, peso)
    edges_succ: list = field(default_factory=list)     # lista de (i, j, peso)


# ─────────────────────────────────────────────────────────────────────────
#  [1] Extracción de notas
# ─────────────────────────────────────────────────────────────────────────

def extract_notes(path: str, track_filter: Optional[int] = None) -> list:
    """Lee un MIDI y devuelve una lista de Note con tiempos en pulsos (beats).
    Ignora tempo/microsegundos: trabaja en unidades de ticks_per_beat, que es
    la unidad natural para razonar sobre compases y métrica."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat or 480
    notes = []
    idx = 0
    for track_no, track in enumerate(mid.tracks):
        if track_filter is not None and track_no != track_filter:
            continue
        t = 0
        active = {}  # (pitch) -> (start_tick, velocity)  [por pista, monofónico-safe]
        for msg in track:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(msg.note, []).append((t, msg.velocity))
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                stack = active.get(msg.note)
                if stack:
                    start_tick, vel = stack.pop(0)
                    notes.append(Note(
                        idx=idx, pitch=msg.note,
                        start=start_tick / tpb, end=max(t, start_tick + 1) / tpb,
                        track=track_no, velocity=vel,
                    ))
                    idx += 1
    notes.sort(key=lambda n: (n.start, n.pitch))
    for i, n in enumerate(notes):
        n.idx = i
    return notes


# ─────────────────────────────────────────────────────────────────────────
#  [2] Construcción del grafo
# ─────────────────────────────────────────────────────────────────────────

def build_graph(notes: list) -> Graph:
    """Construye dos tipos de arista:
      - SIMULTANEIDAD: notas cuyos intervalos [start, end) se solapan
        (peso = fracción de solape respecto a la nota más corta)
      - SUCESIÓN: dentro de cada pista, cada nota conecta con la siguiente
        nota (por tiempo de inicio) de esa misma pista (peso = 1, decayendo
        si hay silencio entre ambas)
    """
    g = Graph(notes=notes)
    n = len(notes)

    # --- simultaneidad: barrido por eventos (mucho más rápido que O(n^2) puro
    #     para MIDIs largos, pero para piezas de tutorial n^2 ya es instantáneo)
    for i in range(n):
        ni = notes[i]
        for j in range(i + 1, n):
            nj = notes[j]
            if nj.start >= ni.end:
                break  # ordenado por start: ya no hay más solapes posibles
            overlap = min(ni.end, nj.end) - max(ni.start, nj.start)
            if overlap > 1e-6:
                shorter = min(ni.dur, nj.dur)
                w = float(np.clip(overlap / shorter, 0.0, 1.0))
                g.edges_simul.append((i, j, w))
                g.edges_simul.append((j, i, w))

    # --- sucesión: cadena cronológica por pista
    by_track = defaultdict(list)
    for note in notes:
        by_track[note.track].append(note.idx)
    for track_no, idxs in by_track.items():
        idxs.sort(key=lambda k: notes[k].start)
        for a, b in zip(idxs, idxs[1:]):
            gap = max(0.0, notes[b].start - notes[a].end)
            w = 1.0 / (1.0 + gap)  # decae con el silencio entre notas
            g.edges_succ.append((a, b, w))
            g.edges_succ.append((b, a, w))

    return g


def _adjacency(n: int, edges: list) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float64)
    for i, j, w in edges:
        A[i, j] += w
    return A


def _row_normalize(A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return A / deg


# ─────────────────────────────────────────────────────────────────────────
#  [3] Propagación (message passing sobre croma 12-dim)
# ─────────────────────────────────────────────────────────────────────────

def propagate(g: Graph, layers: int = 2, w_self: float = 0.34,
              w_simul: float = 0.44, w_succ: float = 0.22) -> np.ndarray:
    """Cada nota-nodo empieza como un one-hot de su clase de altura (croma,
    12-dim) ponderado por su duración. En cada ronda, el vector de cada nodo
    se mezcla con el promedio (normalizado por grado) de sus vecinos de
    simultaneidad y de sucesión, más su propio vector — exactamente el
    mismo patrón de la CapaGNN de la Parte III.10: transformar + agregar por
    tipo de arista + combinar. Aquí la 'transformación' es la identidad
    (se mantiene en la base de croma para que el resultado siga siendo
    interpretable como distribución de alturas), y lo que aporta valor es
    la agregación estructurada por tipo de arista.
    """
    n = len(g.notes)
    H = np.zeros((n, 12), dtype=np.float64)
    for note in g.notes:
        H[note.idx, note.pc] += note.dur
    row_sums = H.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    H = H / row_sums

    A_simul = _row_normalize(_adjacency(n, g.edges_simul))
    A_succ = _row_normalize(_adjacency(n, g.edges_succ))

    total_w = w_self + w_simul + w_succ
    w_self, w_simul, w_succ = w_self / total_w, w_simul / total_w, w_succ / total_w

    for _ in range(max(layers, 0)):
        msg_simul = A_simul @ H
        msg_succ = A_succ @ H
        H = w_self * H + w_simul * msg_simul + w_succ * msg_succ
        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        H = H / row_sums  # renormaliza a distribución de probabilidad

    return H  # (n_notas, 12)


# ─────────────────────────────────────────────────────────────────────────
#  Utilidades de teoría musical compartidas
# ─────────────────────────────────────────────────────────────────────────

def estimate_key(chroma_global: np.ndarray) -> tuple:
    """Krumhansl-Schmuckler: correlaciona el croma agregado de toda la pieza
    contra los 24 perfiles (12 raíces x mayor/menor) y devuelve el mejor."""
    best = None
    for root in range(12):
        maj = np.roll(KS_MAJOR, root)
        minr = np.roll(KS_MINOR, root)
        c_maj = np.corrcoef(chroma_global, maj)[0, 1]
        c_min = np.corrcoef(chroma_global, minr)[0, 1]
        for corr, mode in ((c_maj, "major"), (c_min, "minor")):
            if best is None or corr > best[0]:
                best = (corr, root, mode)
    _, root, mode = best
    name = PITCH_NAMES[root] + ("maj" if mode == "major" else "min")
    return root, mode, name


def parse_key_arg(key_str: str) -> tuple:
    key_str = key_str.strip()
    mode = "minor" if key_str.lower().endswith(("min", "m")) and not key_str.lower().endswith("maj") else "major"
    base = key_str
    for suf in ("maj", "min", "Maj", "Min", "MAJ", "MIN", "m", "M"):
        if base.endswith(suf) and len(base) > len(suf):
            base = base[: -len(suf)]
            break
    base = base.strip()
    name_to_pc = {n: i for i, n in enumerate(PITCH_NAMES)}
    aliases = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
    base = aliases.get(base, base)
    if base not in name_to_pc:
        raise ValueError(f"Tonalidad no reconocida: {key_str!r}")
    return name_to_pc[base], mode, key_str


def match_chord_template(chroma: np.ndarray) -> tuple:
    """Empareja un vector de croma (12,) con la plantilla de acorde más
    cercana (coseno) entre 12 raíces x 9 calidades. Devuelve
    (root_pc, quality, score, chord_tones_pc)."""
    best = None
    for root in range(12):
        for quality, intervals in CHORD_TEMPLATES.items():
            tpl = np.zeros(12)
            for iv in intervals:
                tpl[(root + iv) % 12] = 1.0
            tpl = tpl / tpl.sum()
            denom = (np.linalg.norm(chroma) * np.linalg.norm(tpl))
            score = float(chroma @ tpl / denom) if denom > 0 else 0.0
            if best is None or score > best[0]:
                best = (score, root, quality, [(root + iv) % 12 for iv in intervals])
    score, root, quality, tones = best
    return root, quality, score, tones


def roman_numeral(chord_root_pc: int, quality: str, key_root_pc: int, key_mode: str) -> tuple:
    degree = (chord_root_pc - key_root_pc) % 12
    table = DIATONIC_MAJOR if key_mode == "major" else DIATONIC_MINOR
    if degree in table:
        numeral, func = table[degree]
        if quality in ("maj7", "min7", "hdim7", "dim7", "dom7"):
            numeral = numeral + "7"
        return numeral, func, True
    # cromático / dominante secundario: expresarlo como V/x si es dom7 o maj
    if quality in ("maj", "dom7"):
        target_degree = (degree + 7) % 12
        table_here = table
        if target_degree in table_here:
            target_numeral, _ = table_here[target_degree]
            return f"V{'7' if quality=='dom7' else ''}/{target_numeral}", "D", False
    return f"chr({PITCH_NAMES[chord_root_pc]}{'' if quality=='maj' else quality})", "?", False


CHORD_SYMBOL_SUFFIX = {
    "maj": "", "min": "m", "dim": "°", "aug": "+", "dom7": "7",
    "maj7": "maj7", "min7": "m7", "hdim7": "ø7", "dim7": "°7",
}


def chord_symbol(root_pc: int, quality: str) -> str:
    return PITCH_NAMES[root_pc] + CHORD_SYMBOL_SUFFIX[quality]


# ─────────────────────────────────────────────────────────────────────────
#  Segmentación en compases / ventanas
# ─────────────────────────────────────────────────────────────────────────

def bar_index(note: Note, beats_per_bar: float) -> int:
    return int(note.start // beats_per_bar)


def make_windows(notes: list, beats_per_bar: float, window: int) -> list:
    """Devuelve lista de (win_start_bar, win_end_bar_exclusive, [idx notas])."""
    if not notes:
        return []
    max_bar = max(bar_index(n, beats_per_bar) for n in notes) + 1
    wins = []
    for wb in range(0, max_bar, window):
        we = wb + window
        idxs = [n.idx for n in notes
                if wb <= bar_index(n, beats_per_bar) < we]
        wins.append((wb, we, idxs))
    return wins


def aggregate_chroma(H: np.ndarray, idxs: list) -> np.ndarray:
    if not idxs:
        return np.ones(12) / 12
    v = H[idxs].sum(axis=0)
    s = v.sum()
    return v / s if s > 0 else np.ones(12) / 12


# ─────────────────────────────────────────────────────────────────────────
#  [4]-[8] Análisis armónico completo
# ─────────────────────────────────────────────────────────────────────────

def analyze_harmony_gnn(notes: list, g: Graph, H: np.ndarray, beats_per_bar: float,
                         window: int, key_arg: Optional[str] = None) -> dict:
    if not notes:
        return {"key": None, "bars": []}

    global_chroma = aggregate_chroma(H, list(range(len(notes))))
    if key_arg:
        key_root, key_mode, key_label = parse_key_arg(key_arg)
    else:
        key_root, key_mode, key_label = estimate_key(global_chroma)
        key_label = PITCH_NAMES[key_root] + ("maj" if key_mode == "major" else "min")

    wins = make_windows(notes, beats_per_bar, window)
    bars = []
    for wb, we, idxs in wins:
        chroma = aggregate_chroma(H, idxs)
        root, quality, score, tones = match_chord_template(chroma)
        numeral, func, diatonic = roman_numeral(root, quality, key_root, key_mode)
        bars.append({
            "bar_start": wb, "bar_end": we,
            "chord": chord_symbol(root, quality),
            "root_pc": root, "quality": quality,
            "roman": numeral, "function": func, "diatonic": diatonic,
            "confidence": round(score, 3),
            "n_notes": len(idxs),
        })
    return {
        "key": key_label, "key_root_pc": key_root, "key_mode": key_mode,
        "beats_per_bar": beats_per_bar, "window": window, "bars": bars,
    }


def baseline_bar_chroma(notes: list, beats_per_bar: float, window: int) -> list:
    """Croma 'plano' por compás, sin ningún paso de grafo/propagación —
    la línea base equivalente a lo que hace un analizador solo-secuencial."""
    wins = make_windows(notes, beats_per_bar, window)
    out = []
    for wb, we, idxs in wins:
        v = np.zeros(12)
        for i in idxs:
            v[notes[i].pc] += notes[i].dur
        s = v.sum()
        out.append(v / s if s > 0 else np.ones(12) / 12)
    return out


def classify_notes(notes: list, analysis: dict, H: np.ndarray, beats_per_bar: float) -> list:
    """Nota a nota: ¿su clase de altura pertenece al acorde detectado en el
    compás donde suena? Si sí -> 'estructural'; si no -> 'ornamental'
    (nota de paso / bordadura / apoyatura / anticipación...). Además reporta
    la 'sorpresa' de la nota respecto a su propio vecindario en el grafo
    (cuánto diverge H[nota] de su propio pitch-class puro): una nota muy
    'sorprendida' por su contexto es un candidato fuerte a no-chord-tone."""
    bars_by_range = analysis["bars"]

    def bar_for(note):
        for b in bars_by_range:
            if b["bar_start"] <= bar_index(note, beats_per_bar) < b["bar_end"]:
                return b
        return None

    out = []
    for note in notes:
        b = bar_for(note)
        if b is None:
            continue
        # tonos del acorde reconstruidos directamente de la plantilla ya elegida
        # (raíz + calidad) para el compás de esta nota
        chord_tones = {(b["root_pc"] + iv) % 12 for iv in CHORD_TEMPLATES[b["quality"]]}
        own = np.eye(12)[note.pc]
        own_h = H[note.idx]
        surprise = float(1.0 - (own @ own_h) / (np.linalg.norm(own_h) + 1e-9))
        role = "estructural" if note.pc in chord_tones else "ornamental"
        out.append({
            "idx": note.idx, "pitch": note.pitch, "pitch_class": PITCH_NAMES[note.pc],
            "start": round(note.start, 3), "track": note.track,
            "bar": bar_index(note, beats_per_bar),
            "chord": b["chord"], "role": role,
            "surprise": round(surprise, 3),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Presentación en terminal
# ─────────────────────────────────────────────────────────────────────────

def _hr(title, width=70):
    print(_C.wrap(f"── {title} " + "─" * max(0, width - len(title) - 4), _C.HDR))


def print_analyze(analysis: dict, explain: bool, max_bars: int):
    key_label = analysis["key"]
    print(_C.wrap(f"Tonalidad estimada: {key_label}", _C.BOLD))
    _hr("ANÁLISIS ARMÓNICO (GNN de grafo de notas)")
    print(f"{'compás':>8}  {'acorde':<8} {'numeral':<12} {'func':<5} {'conf.':>6}  notas")
    for b in analysis["bars"][:max_bars]:
        rango = f"{b['bar_start']}" if b["bar_end"] - b["bar_start"] == 1 else f"{b['bar_start']}-{b['bar_end']-1}"
        conf_col = _C.OK if b["confidence"] >= 0.7 else (_C.WARN if b["confidence"] >= 0.45 else _C.DIM)
        conf_str = f"{b['confidence']:.3f}"
        print(f"{rango:>8}  {b['chord']:<8} {b['roman']:<12} {b['function']:<5} "
              f"{_C.wrap(conf_str, conf_col):>6}  {b['n_notes']}")
        if explain:
            diat = "diatónico" if b["diatonic"] else "cromático / préstamo o secundario"
            print(_C.wrap(f"           → {diat}, similitud con la plantilla de "
                           f"{b['chord']} = {b['confidence']:.3f}", _C.DIM))
    if len(analysis["bars"]) > max_bars:
        print(_C.wrap(f"... ({len(analysis['bars']) - max_bars} compases más, usa --max-bars)", _C.DIM))


def print_classify(rows: list, explain: bool, max_bars: int):
    _hr("CLASIFICACIÓN DE NOTAS (estructural vs. ornamental)")
    print(f"{'compás':>6} {'inicio':>7} {'pista':>5} {'nota':<5} {'acorde':<7} "
          f"{'rol':<12} {'sorpresa':>8}")
    shown_bars = set()
    for r in rows:
        if len(shown_bars) >= max_bars and r["bar"] not in shown_bars:
            continue
        shown_bars.add(r["bar"])
        role_col = _C.OK if r["role"] == "estructural" else _C.WARN
        print(f"{r['bar']:>6} {r['start']:>7.2f} {r['track']:>5} {r['pitch_class']:<5} "
              f"{r['chord']:<7} {_C.wrap(r['role'], role_col):<12} {r['surprise']:>8.3f}")
        if explain and r["role"] == "ornamental":
            print(_C.wrap(f"        → {r['pitch_class']} no pertenece a {r['chord']}: "
                           f"probable nota de paso/bordadura/apoyatura", _C.DIM))


def print_compare(base_rows: list, gnn_rows: list, only_changed: bool):
    _hr("COMPARACIÓN: baseline (solo compás) vs. GNN (con contexto de grafo)")
    print(f"{'compás':>8}  {'baseline':<10} {'GNN':<10}  {'¿coincide?':<10}")
    n_diff = 0
    for a, b in zip(base_rows, gnn_rows):
        same = a["chord"] == b["chord"]
        if not same:
            n_diff += 1
        if only_changed and same:
            continue
        rango = f"{a['bar_start']}" if a["bar_end"] - a["bar_start"] == 1 else f"{a['bar_start']}-{a['bar_end']-1}"
        mark = _C.wrap("✓", _C.OK) if same else _C.wrap("✗ difiere", _C.WARN)
        print(f"{rango:>8}  {a['chord']:<10} {b['chord']:<10}  {mark}")
    total = len(gnn_rows)
    print()
    print(f"Compases en desacuerdo: {n_diff}/{total} "
          f"({(100*n_diff/total) if total else 0:.1f}%)")
    if n_diff:
        print(_C.wrap("En esos compases, el contexto de vecindario (grafo) cambió la "
                       "decisión respecto a mirar solo las notas de ese compás — suele "
                       "ocurrir con arpegios, notas de paso rápidas o texturas dispersas.",
                       _C.DIM))


# ─────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────

def _common_args(p):
    p.add_argument("midi", help="Fichero MIDI de entrada")
    p.add_argument("--key", help="Fuerza la tonalidad (p.ej. Cmaj, Am, F#min)")
    p.add_argument("--beats-per-bar", type=float, default=4.0)
    p.add_argument("--window", type=int, default=1, help="Compases por ventana de análisis")
    p.add_argument("--layers", type=int, default=2, help="Rondas de propagación en el grafo")
    p.add_argument("--w-self", type=float, default=0.34)
    p.add_argument("--w-simul", type=float, default=0.44)
    p.add_argument("--w-succ", type=float, default=0.22)
    p.add_argument("--track", type=int, default=None)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--json", default=None)
    p.add_argument("--max-bars", type=int, default=64)
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--quiet", action="store_true")


def cmd_graph(args):
    notes = extract_notes(args.midi, args.track)
    g = build_graph(notes)
    if not args.quiet:
        _hr("GRAFO DE NOTAS")
        print(f"Notas (nodos):           {len(notes)}")
        print(f"Aristas de simultaneidad: {len(g.edges_simul)//2} (no dirigidas)")
        print(f"Aristas de sucesión:      {len(g.edges_succ)//2} (no dirigidas)")
        pistas = sorted(set(n.track for n in notes))
        print(f"Pistas involucradas:      {pistas}")
    if args.export_graph:
        payload = {
            "nodes": [{"idx": n.idx, "pitch": n.pitch, "pitch_class": PITCH_NAMES[n.pc],
                       "start": n.start, "end": n.end, "track": n.track,
                       "velocity": n.velocity} for n in notes],
            "edges_simultaneity": [{"a": i, "b": j, "w": w} for i, j, w in g.edges_simul if i < j],
            "edges_succession": [{"a": i, "b": j, "w": w} for i, j, w in g.edges_succ if i < j],
        }
        with open(args.export_graph, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nGrafo exportado a {args.export_graph}")


def cmd_analyze(args):
    notes = extract_notes(args.midi, args.track)
    if not notes:
        print(_C.wrap("No se encontraron notas en el MIDI.", _C.WARN))
        return
    g = build_graph(notes)
    H = propagate(g, args.layers, args.w_self, args.w_simul, args.w_succ)
    analysis = analyze_harmony_gnn(notes, g, H, args.beats_per_bar, args.window, args.key)
    if not args.quiet:
        print_analyze(analysis, args.explain, args.max_bars)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nJSON escrito en {args.json}")


def cmd_classify_notes(args):
    notes = extract_notes(args.midi, args.track)
    if not notes:
        print(_C.wrap("No se encontraron notas en el MIDI.", _C.WARN))
        return
    g = build_graph(notes)
    H = propagate(g, args.layers, args.w_self, args.w_simul, args.w_succ)
    analysis = analyze_harmony_gnn(notes, g, H, args.beats_per_bar, args.window, args.key)
    rows = classify_notes(notes, analysis, H, args.beats_per_bar)
    if not args.quiet:
        print_classify(rows, args.explain, args.max_bars)
        n_struct = sum(1 for r in rows if r["role"] == "estructural")
        print(f"\nEstructurales: {n_struct}/{len(rows)}  ·  "
              f"Ornamentales: {len(rows)-n_struct}/{len(rows)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nJSON escrito en {args.json}")


def cmd_compare(args):
    notes = extract_notes(args.midi, args.track)
    if not notes:
        print(_C.wrap("No se encontraron notas en el MIDI.", _C.WARN))
        return
    g = build_graph(notes)

    H_gnn = propagate(g, args.layers, args.w_self, args.w_simul, args.w_succ)
    gnn_analysis = analyze_harmony_gnn(notes, g, H_gnn, args.beats_per_bar, args.window, args.key)

    H_flat = propagate(g, 0, 1.0, 0.0, 0.0)  # layers=0 -> sin propagación (baseline)
    base_analysis = analyze_harmony_gnn(notes, g, H_flat, args.beats_per_bar, args.window, args.key)

    if not args.quiet:
        print_compare(base_analysis["bars"], gnn_analysis["bars"], args.only_changed)
    if args.json:
        payload = {"baseline": base_analysis, "gnn": gnn_analysis}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        if not args.quiet:
            print(f"\nJSON escrito en {args.json}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="harmonic_gnn_analyzer.py",
        description="Análisis armónico por grafo de notas (simultaneidad + sucesión)",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_graph = sub.add_parser("graph", help="Construye y exporta el grafo de notas")
    _common_args(p_graph)
    p_graph.add_argument("--export-graph", default=None)
    p_graph.set_defaults(func=cmd_graph)

    p_analyze = sub.add_parser("analyze", help="Análisis armónico compás a compás vía GNN")
    _common_args(p_analyze)
    p_analyze.set_defaults(func=cmd_analyze)

    p_classify = sub.add_parser("classify-notes", help="Nota a nota: estructural vs. ornamental")
    _common_args(p_classify)
    p_classify.set_defaults(func=cmd_classify_notes)

    p_compare = sub.add_parser("compare", help="GNN (con contexto) vs. baseline (solo el compás)")
    _common_args(p_compare)
    p_compare.add_argument("--only-changed", action="store_true")
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    _C.ON = not args.no_color
    args.func(args)


if __name__ == "__main__":
    main()
