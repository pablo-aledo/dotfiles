#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      HARMONY CHECKER  v1.0                                  ║
║        Corrector de conducción de voces a cuatro voces (SATB)               ║
║                                                                              ║
║  voice_leader GENERA el voicing SATB correcto; esta herramienta hace lo      ║
║  contrario: toma TU intento de armonización a cuatro voces y marca los       ║
║  errores con explicación pedagógica, como un profesor corrigiendo deberes   ║
║  de armonía. No reescribe nada: señala y explica.                            ║
║                                                                              ║
║  ENTRADA:                                                                    ║
║    · 4 pistas (Soprano/Alto/Tenor/Bajo, se ordenan por altura), o           ║
║    · 1 pista con acordes de 4 notas por tiempo (se separan las voces)       ║
║                                                                              ║
║  ERRORES DETECTADOS:                                                         ║
║    5ª/8ª paralelas   — dos voces se mueven en paralelo a 5ª justa u 8ª      ║
║    5ª/8ª directas    — llegada a 5ª/8ª por movimiento directo en extremos    ║
║    cruce de voces    — una voz inferior sube por encima de la superior      ║
║    solapamiento      — una voz invade el registro que dejó su vecina        ║
║    espaciado         — hueco > 8ª entre voces superiores adyacentes         ║
║    sensible sin resolver — la sensible en el bajo/soprano no sube a tónica  ║
║    salto excesivo    — salto melódico > 6ª en una voz (aviso)               ║
║                                                                              ║
║  USO:                                                                        ║
║    python harmony_checker.py coral.mid                                       ║
║    python harmony_checker.py coral.mid --key Cmaj                           ║
║    python harmony_checker.py coral.mid --grid 1 --json errores.json         ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi              MIDI SATB de entrada                                    ║
║    --key KEY         Tonalidad para comprobar la sensible (p.ej. Cmaj, Am)  ║
║    --grid BEATS      Rejilla de agrupación de acordes en tiempos (default:1)║
║    --json FILE       Escribe un sidecar JSON con la lista de errores         ║
║    --no-color        Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from harmony_checker import check_harmony                                 ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Tuple

try:
    from playability_auditor import read_midi, TimeContext, extract_notes, pitch_name
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m",
           "GRN": "\033[92m", "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
VOICES = ["S", "A", "T", "B"]


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def _parse_key(s: str) -> Tuple[int, str]:
    letter = s[0].upper()
    i, pc = 1, _PC[s[0].upper()]
    while i < len(s) and s[i] in "#b♯♭sf":
        pc += 1 if s[i] in "#♯s" else -1
        i += 1
    mode = "min" if s[i:].lower() in ("m", "min", "minor", "menor") else "maj"
    return pc % 12, mode


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE ACORDES SATB
# ══════════════════════════════════════════════════════════════════════════════

def _extract_chords(mid, tc: TimeContext, grid_beats: float) -> List[List[int]]:
    """Devuelve una lista de acordes; cada acorde es [S,A,T,B] (altura descendente)."""
    note_tracks = [t for t in mid.tracks if extract_notes(t)]
    step = max(1, int(round(grid_beats * tc.tpb)))
    last = max((n.end for t in note_tracks for n in extract_notes(t)), default=0)

    chords = []
    t = 0
    while t < last:
        sounding = []
        for trk in note_tracks:
            for n in extract_notes(trk):
                if n.start <= t < n.end:
                    sounding.append(n.pitch)
        if sounding:
            uniq = sorted(set(sounding), reverse=True)[:4]
            chords.append(uniq)
        t += step
    # colapsa acordes repetidos consecutivos (misma verticalidad)
    collapsed = []
    for ch in chords:
        if not collapsed or collapsed[-1] != ch:
            collapsed.append(ch)
    return collapsed


# ══════════════════════════════════════════════════════════════════════════════
#  COMPROBACIONES
# ══════════════════════════════════════════════════════════════════════════════

def _interval_class(a: int, b: int) -> int:
    return abs(a - b) % 12


def check_pair(prev: List[int], cur: List[int], idx: int) -> List[dict]:
    """Errores entre dos acordes consecutivos (índice del segundo, 1-based)."""
    issues = []
    nv = min(len(prev), len(cur))
    # paralelas y directas
    for i in range(nv):
        for j in range(i + 1, nv):
            if i >= len(prev) or j >= len(prev):
                continue
            m1a, m1b = prev[i], prev[j]
            m2a, m2b = cur[i], cur[j]
            ic1 = _interval_class(m1a, m1b)
            ic2 = _interval_class(m2a, m2b)
            moved_i = m2a != m1a
            moved_j = m2b != m1b
            same_dir = (m2a - m1a) * (m2b - m1b) > 0
            vi, vj = VOICES[i] if i < 4 else f"v{i}", VOICES[j] if j < 4 else f"v{j}"
            if moved_i and moved_j and same_dir and ic1 == ic2 and ic2 in (0, 7):
                kind = "8ª/unísono" if ic2 == 0 else "5ª justa"
                issues.append(dict(
                    chord=idx, severity="ERROR", type="paralelas",
                    voices=f"{vi}-{vj}",
                    msg=f"{kind}s paralelas entre {vi} y {vj}",
                    why="dos voces que se mueven manteniendo la misma 5ª u 8ª "
                        "suenan como una sola voz y debilitan la independencia."))
            elif (i == 0 and j == nv - 1) and same_dir and ic2 in (0, 7) \
                    and moved_i and moved_j and abs(m2a - m1a) > 2:
                kind = "8ª" if ic2 == 0 else "5ª"
                issues.append(dict(
                    chord=idx, severity="AVISO", type="directas",
                    voices=f"{vi}-{vj}",
                    msg=f"{kind} directa en las voces extremas ({vi}-{vj})",
                    why="llegar a una 5ª/8ª por movimiento directo con salto en la voz "
                        "superior expone la consonancia perfecta; mejor por grado o "
                        "movimiento contrario."))
    # cruce y solapamiento
    for i in range(nv - 1):
        if cur[i] < cur[i + 1]:
            issues.append(dict(
                chord=idx, severity="ERROR", type="cruce",
                voices=f"{VOICES[i]}-{VOICES[i+1]}",
                msg=f"cruce de voces: {VOICES[i]} por debajo de {VOICES[i+1]}",
                why="las voces deben mantener su orden S≥A≥T≥B para que cada línea "
                    "sea audible por separado."))
        if i < len(prev) and cur[i] < prev[i + 1] - 0 and prev[i + 1] > cur[i]:
            # solapamiento: la voz i baja por debajo de donde estaba la i+1
            if cur[i] < prev[i + 1]:
                issues.append(dict(
                    chord=idx, severity="AVISO", type="solapamiento",
                    voices=f"{VOICES[i]}-{VOICES[i+1]}",
                    msg=f"solapamiento: {VOICES[i]} invade el registro previo de {VOICES[i+1]}",
                    why="una voz no debería moverse a una altura que la voz vecina "
                        "acababa de ocupar; confunde la percepción de las líneas."))
    return issues


def check_spacing(chord: List[int], idx: int) -> List[dict]:
    issues = []
    for i in range(min(2, len(chord) - 1)):        # S-A y A-T
        if chord[i] - chord[i + 1] > 12:
            issues.append(dict(
                chord=idx, severity="AVISO", type="espaciado",
                voices=f"{VOICES[i]}-{VOICES[i+1]}",
                msg=f"hueco > 8ª entre {VOICES[i]} y {VOICES[i+1]}",
                why="entre voces superiores adyacentes conviene no pasar de una 8ª "
                    "para mantener un bloque sonoro compacto."))
    return issues


def check_leaps(prev: List[int], cur: List[int], idx: int) -> List[dict]:
    issues = []
    for i in range(min(len(prev), len(cur))):
        leap = abs(cur[i] - prev[i])
        if leap > 9:
            issues.append(dict(
                chord=idx, severity="AVISO", type="salto",
                voices=VOICES[i] if i < 4 else f"v{i}",
                msg=f"salto de {leap} st en {VOICES[i] if i<4 else i} (> 6ª)",
                why="saltos melódicos amplios son difíciles de cantar; en coral se "
                    "prefieren grados y saltos de consonancia pequeños."))
    return issues


def check_leading_tone(prev: List[int], cur: List[int], idx: int,
                       tonic_pc: int, mode: str) -> List[dict]:
    """La sensible en voces extremas debe resolver ascendiendo a la tónica."""
    issues = []
    lead_pc = (tonic_pc - 1) % 12
    for i in (0, len(cur) - 1):                     # soprano y bajo
        if i >= len(prev):
            continue
        if prev[i] % 12 == lead_pc:
            resolves = (cur[i] - prev[i]) == 1 and cur[i] % 12 == tonic_pc
            if not resolves and cur[i] != prev[i]:
                v = VOICES[i] if i < 4 else f"v{i}"
                issues.append(dict(
                    chord=idx, severity="AVISO", type="sensible",
                    voices=v,
                    msg=f"sensible sin resolver en {v} ({pitch_name(prev[i])})",
                    why="la sensible en una voz extrema tiende a subir un semitono a "
                        "la tónica; no hacerlo deja la tensión sin resolver."))
    return issues


# ══════════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def check_harmony(path: str, key: Optional[str] = None,
                  grid_beats: float = 1.0) -> dict:
    """API pública. Devuelve la lista de errores de conducción de voces."""
    mid = read_midi(path)
    tc = TimeContext(mid)
    chords = _extract_chords(mid, tc, grid_beats)
    if len(chords) < 2:
        raise ValueError("se necesitan al menos dos acordes para comprobar la conducción")

    tonic_pc, mode = _parse_key(key) if key else (None, None)

    issues: List[dict] = []
    four_voice = sum(1 for ch in chords if len(ch) >= 4)
    for k in range(len(chords)):
        issues += check_spacing(chords[k], k + 1)
        if k > 0:
            issues += check_pair(chords[k - 1], chords[k], k + 1)
            issues += check_leaps(chords[k - 1], chords[k], k + 1)
            if tonic_pc is not None:
                issues += check_leading_tone(chords[k - 1], chords[k], k + 1,
                                             tonic_pc, mode)

    n_err = sum(1 for i in issues if i["severity"] == "ERROR")
    n_warn = sum(1 for i in issues if i["severity"] == "AVISO")
    return {
        "file": path, "n_chords": len(chords),
        "voices_detected": max((len(ch) for ch in chords), default=0),
        "four_voice_chords": four_voice,
        "chords": [[pitch_name(p) for p in ch] for ch in chords],
        "errors": n_err, "warnings": n_warn,
        "issues": issues,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="harmony_checker.py",
        description="Corrector de conducción de voces SATB.")
    ap.add_argument("midi")
    ap.add_argument("--key")
    ap.add_argument("--grid", type=float, default=1.0)
    ap.add_argument("--json")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    B, R, G, C = _c("B"), _c("R"), _c("G"), _c("CYA")
    print(f"\n{'═' * 78}")
    print(f"  {B}HARMONY CHECKER v{VERSION}  ·  {args.midi}{R}")
    print(f"{'═' * 78}")
    try:
        st = check_harmony(args.midi, key=args.key, grid_beats=args.grid)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Acordes: {st['n_chords']}   voces: {st['voices_detected']}   "
          f"acordes a 4 voces: {st['four_voice_chords']}")
    if st["voices_detected"] < 4:
        print(f"  {_c('YEL')}(aviso: se detectaron menos de 4 voces; algunas "
              f"comprobaciones no aplican){R}")
    print(f"  {_c('RED')}errores: {st['errors']}{R}   "
          f"{_c('YEL')}avisos: {st['warnings']}{R}")

    if st["issues"]:
        print(f"\n  {G}acorde  tipo           voces   detalle{R}")
        for it in st["issues"]:
            col = _c("RED") if it["severity"] == "ERROR" else _c("YEL")
            print(f"  {it['chord']:>5}   {col}{it['type']:<12}{R}   "
                  f"{it['voices']:<6}  {it['msg']}")
            print(f"          {G}{it['why']}{R}")
    else:
        print(f"\n  {_c('GRN')}Sin errores de conducción de voces. ¡Buen trabajo!{R}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"version": VERSION, **st}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n  {C}JSON: {args.json}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
