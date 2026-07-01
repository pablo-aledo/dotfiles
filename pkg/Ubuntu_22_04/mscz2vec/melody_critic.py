#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       MELODY CRITIC  v1.0                                   ║
║        Crítica constructiva de una melodía propia (didáctica)               ║
║                                                                              ║
║  quality_scorer PUNTÚA una obra pero no ENSEÑA; esta herramienta analiza    ║
║  una melodía tuya y devuelve comentario pedagógico: qué está bien y qué      ║
║  mejorar (contorno, ámbito, saltos, motion por grados, notas repetidas,     ║
║  colocación del clímax, equilibrio de frases antecedente/consecuente).      ║
║  Pensada para quien está aprendiendo a componer, no para rankear.           ║
║                                                                              ║
║  MÉTRICAS:                                                                   ║
║    ámbito        — distancia tónica-aguda; ideal cantábile ≈ 8ª–10ª         ║
║    grados vs saltos — proporción de segundas frente a saltos (>3st)         ║
║    saltos sueltos — saltos grandes NO compensados por grado contrario       ║
║    clímax        — dónde cae la nota más alta (ideal ≈ 60–75% de la frase)  ║
║    cambios de dirección — sinuosidad del contorno (ni recto ni errático)    ║
║    notas repetidas — proporción de repeticiones (estatismo)                 ║
║    frases         — nº y equilibrio de frases (separadas por silencios)     ║
║                                                                              ║
║  USO:                                                                        ║
║    python melody_critic.py melodia.mid                                       ║
║    python melody_critic.py obra.mid --track 0                               ║
║    python melody_critic.py melodia.mid --json critica.json                  ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi              MIDI de entrada (melodía)                              ║
║    --track N         Pista a analizar (default: la más aguda con notas)     ║
║    --json FILE       Escribe un sidecar JSON con métricas y comentarios      ║
║    --no-color        Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from melody_critic import critique_melody                                 ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List

import numpy as np

try:
    from playability_auditor import read_midi, TimeContext, extract_notes, pitch_name
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
        return _SPARK[3] * len(v)
    x = (v - v.min()) / (v.max() - v.min())
    return "".join(_SPARK[min(7, int(round(t * 7)))] for t in x)


# ══════════════════════════════════════════════════════════════════════════════
#  SELECCIÓN DE PISTA MELÓDICA
# ══════════════════════════════════════════════════════════════════════════════

def _pick_melody_track(mid) -> int:
    best, best_avg = 0, -1.0
    for i, trk in enumerate(mid.tracks):
        ns = extract_notes(trk)
        if len(ns) < 3:
            continue
        avg = sum(n.pitch for n in ns) / len(ns)
        if avg > best_avg:
            best, best_avg = i, avg
    return best


def _monophonic(notes) -> List:
    """Reduce a línea monofónica: en cada onset, la nota más aguda (skyline)."""
    by_start = {}
    for n in notes:
        by_start.setdefault(n.start, []).append(n)
    line = []
    for st in sorted(by_start):
        line.append(max(by_start[st], key=lambda n: n.pitch))
    return line


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def critique_melody(path: str, track: Optional[int] = None) -> dict:
    """API pública. Devuelve métricas + comentarios (elogios / sugerencias)."""
    mid = read_midi(path)
    tc = TimeContext(mid)
    tidx = track if track is not None else _pick_melody_track(mid)
    if not (0 <= tidx < len(mid.tracks)):
        raise ValueError(f"pista {tidx} fuera de rango")
    notes = _monophonic(extract_notes(mid.tracks[tidx]))
    if len(notes) < 4:
        raise ValueError("la pista tiene muy pocas notas para criticar una melodía")

    pitches = np.array([n.pitch for n in notes])
    intervals = np.diff(pitches)
    n = len(notes)

    # ámbito
    ambit = int(pitches.max() - pitches.min())
    # grados vs saltos
    steps = int(np.sum(np.abs(intervals) <= 2))
    leaps = int(np.sum(np.abs(intervals) > 2))
    big_leaps = int(np.sum(np.abs(intervals) >= 7))
    step_ratio = steps / max(1, len(intervals))
    # saltos sueltos (salto grande no compensado por grado contrario)
    unresolved = 0
    for i in range(len(intervals) - 1):
        if abs(intervals[i]) >= 5:
            comp = intervals[i + 1]
            if not (np.sign(comp) == -np.sign(intervals[i]) and abs(comp) <= 2):
                unresolved += 1
    # clímax
    peak_idx = int(np.argmax(pitches))
    peak_pos = peak_idx / max(1, n - 1)
    peak_unique = int(np.sum(pitches == pitches.max()))
    # cambios de dirección
    dirs = np.sign(intervals[intervals != 0])
    dir_changes = int(np.sum(dirs[1:] != dirs[:-1])) if len(dirs) > 1 else 0
    dir_ratio = dir_changes / max(1, len(dirs) - 1)
    # notas repetidas
    repeats = int(np.sum(intervals == 0))
    repeat_ratio = repeats / max(1, len(intervals))
    # frases por silencios (gap > 1 negra)
    gap = tc.tpb
    phrase_bounds = [i for i in range(1, n) if notes[i].start - notes[i - 1].end >= gap]
    n_phrases = len(phrase_bounds) + 1
    phrase_lens = np.diff([0] + phrase_bounds + [n])
    phrase_balance = (float(phrase_lens.min() / phrase_lens.max())
                      if len(phrase_lens) > 1 else 1.0)

    # ── comentario ──
    elogios, sugerencias = [], []

    if 7 <= ambit <= 17:
        elogios.append(f"Ámbito cantábile ({ambit} semitonos): fácil de cantar y recordar.")
    elif ambit < 5:
        sugerencias.append(f"El ámbito es muy estrecho ({ambit} st): prueba a ampliar el "
                           "registro para dar más recorrido a la melodía.")
    elif ambit > 21:
        sugerencias.append(f"El ámbito es muy amplio ({ambit} st, más de 1½ 8ª): puede "
                           "costar de cantar/tocar; considera acotarlo.")

    if 0.55 <= step_ratio <= 0.85:
        elogios.append(f"Buen equilibrio grados/saltos ({step_ratio:.0%} por grado): "
                       "fluye pero no es plana.")
    elif step_ratio > 0.9:
        sugerencias.append("Casi todo va por grados conjuntos: introduce algún salto "
                           "expresivo (4ª, 5ª, 6ª) para crear interés.")
    elif step_ratio < 0.4:
        sugerencias.append("Predominan los saltos: añade movimiento por grados para que "
                           "la línea sea más cantábile y conecte las notas.")

    if unresolved == 0 and leaps > 0:
        elogios.append("Todos los saltos grandes se compensan con paso contrario: "
                       "conducción melódica muy sólida.")
    elif unresolved > 0:
        sugerencias.append(f"{unresolved} salto(s) grande(s) sin compensar: tras un salto "
                           "amplio, suele sonar mejor moverse por grado en dirección contraria.")

    if 0.55 <= peak_pos <= 0.8 and peak_unique <= 2:
        elogios.append(f"Clímax bien colocado (≈{peak_pos:.0%} de la frase) y único: "
                       "da una curva dramática clara.")
    elif peak_unique > 3:
        sugerencias.append(f"La nota más alta aparece {peak_unique} veces: reservarla para "
                           "un solo momento haría el clímax más impactante.")
    elif peak_pos < 0.3:
        sugerencias.append("El punto culminante llega muy pronto: retrasarlo hacia el último "
                           "tercio suele dar más tensión narrativa.")

    if 0.25 <= dir_ratio <= 0.6:
        elogios.append("El contorno alterna subidas y bajadas con naturalidad.")
    elif dir_ratio > 0.7:
        sugerencias.append("El contorno cambia de dirección casi en cada nota (zigzag): "
                           "prueba tramos más largos en una misma dirección.")
    elif dir_ratio < 0.15:
        sugerencias.append("La línea es muy unidireccional: alterna más subidas y bajadas.")

    if repeat_ratio > 0.35:
        sugerencias.append(f"Muchas notas repetidas ({repeat_ratio:.0%}): variar la altura "
                           "evita que suene estática.")

    if n_phrases >= 2 and phrase_balance >= 0.6:
        elogios.append(f"{n_phrases} frases de longitud equilibrada: buena sintaxis "
                       "antecedente/consecuente.")
    elif n_phrases >= 2 and phrase_balance < 0.4:
        sugerencias.append("Las frases son muy desiguales en longitud: equilibrarlas "
                           "(p.ej. 4+4 compases) da sensación de pregunta/respuesta.")
    elif n_phrases == 1:
        sugerencias.append("Parece una sola frase larga: dividirla con una respiración "
                           "crearía estructura de pregunta/respuesta.")

    score = min(1.0, 0.25 + 0.12 * len(elogios) - 0.05 * len(sugerencias))
    return {
        "file": path, "track": tidx, "n_notes": n,
        "range_semitones": ambit,
        "range": [pitch_name(int(pitches.min())), pitch_name(int(pitches.max()))],
        "step_ratio": round(step_ratio, 3),
        "leaps": leaps, "big_leaps": big_leaps, "unresolved_leaps": unresolved,
        "peak_position": round(peak_pos, 3), "peak_note": pitch_name(int(pitches.max())),
        "peak_occurrences": peak_unique,
        "direction_change_ratio": round(dir_ratio, 3),
        "repeat_ratio": round(repeat_ratio, 3),
        "n_phrases": n_phrases, "phrase_balance": round(phrase_balance, 3),
        "contour": [int(p) for p in pitches],
        "elogios": elogios, "sugerencias": sugerencias,
        "overall": round(max(0.0, score), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="melody_critic.py",
        description="Crítica constructiva de una melodía.")
    ap.add_argument("midi")
    ap.add_argument("--track", type=int)
    ap.add_argument("--json")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    print(f"  {B}MELODY CRITIC v{VERSION}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")
    try:
        st = critique_melody(args.midi, track=args.track)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Pista {st['track']}   notas: {st['n_notes']}   "
          f"ámbito: {st['range'][0]}–{st['range'][1]} ({st['range_semitones']} st)")
    print(f"  contorno: {sparkline(st['contour'])}")
    print(f"  grados: {st['step_ratio']:.0%}   saltos grandes: {st['big_leaps']}   "
          f"clímax≈{st['peak_position']:.0%} ({st['peak_note']})   "
          f"frases: {st['n_phrases']}")
    print(f"  valoración global: {C}{st['overall']:.2f}{R}")

    if st["elogios"]:
        print(f"\n  {_c('GRN')}Lo que funciona:{R}")
        for e in st["elogios"]:
            print(f"    ✓ {e}")
    if st["sugerencias"]:
        print(f"\n  {_c('YEL')}Para mejorar:{R}")
        for s in st["sugerencias"]:
            print(f"    → {s}")
    if not st["sugerencias"]:
        print(f"\n  {G}(sin sugerencias: la línea está muy bien construida){R}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"version": VERSION, **st}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n  {C}JSON: {args.json}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
