#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       MELODY SELECTOR  v1.0                                  ║
║        Cierra el bucle generar → criticar → elegir para melodías             ║
║                                                                              ║
║  Los generadores (melody_conditioned_v5 --candidates, bass_line_composer,    ║
║  melody_generator…) eligen "el mejor" con su PROPIO score interno. Esta      ║
║  herramienta puntúa los candidatos con el crítico oficial del ecosistema     ║
║  (melody_critic.py) y se queda con el mejor; con `loop` automatiza todo el   ║
║  ciclo (generar N candidatos con semillas distintas, criticar, elegir).      ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    rank   Puntúa varios MIDIs y los ordena (tabla + JSON opcional)           ║
║    pick   rank + copia el mejor MIDI a --output                              ║
║    loop   Ejecuta un comando generador N veces (una semilla por vuelta),     ║
║           critica todo lo que produce y se queda con el mejor                ║
║                                                                              ║
║  USO:                                                                        ║
║    python melody_selector.py rank cand_*.mid                                 ║
║    python melody_selector.py rank cand_*.mid --track 1 --json ranking.json   ║
║    python melody_selector.py pick mel.mid mel_v2.mid mel_v3.mid \\            ║
║        --output mejor.mid                                                    ║
║    python melody_selector.py loop --n 12 --stop-at 0.55 --output mejor.mid \\ ║
║        --cmd "python melody_conditioned_v5.py --engine motif --key C \\       ║
║               --bars 16 --seed {seed} --candidates 3 --output {out}"         ║
║                                                                              ║
║  OPCIONES COMUNES:                                                           ║
║    --track N        Pista a criticar (default: la más aguda con notas)       ║
║    --critic FILE    melody_critic.py (default: junto a este script / cwd)    ║
║    --min-notes N    Descarta candidatos con menos de N notas (default: 4)    ║
║    --jobs N         Críticas en paralelo (default: 4)                        ║
║    --json FILE      Informe JSON (métricas de cada candidato + ganador)      ║
║    --no-color       Sin colores ANSI                                         ║
║                                                                              ║
║  OPCIONES DE loop:                                                           ║
║    --cmd TEXTO      Comando generador. Marcadores: {seed} {out} {i}.         ║
║                     {out} es un .mid dentro de una carpeta propia por        ║
║                     vuelta; se critican TODOS los .mid que esa carpeta       ║
║                     contenga (p. ej. out.mid, out_v2.mid, out_v3.mid)        ║
║    --n N            Nº de vueltas (default: 8)                               ║
║    --seed-start S   Primera semilla (default: 1; vuelta i usa S+i)           ║
║    --stop-at F      Parar en cuanto un candidato llegue a F (0-1)            ║
║    --workdir DIR    Carpeta de trabajo (default: ./selector_work)            ║
║    --keep           No borrar los candidatos descartados                     ║
║    --cwd DIR        Directorio desde el que se ejecuta --cmd (default: .)    ║
║    --timeout S      Timeout por vuelta (default: 300)                        ║
║                                                                              ║
║  PUNTUACIÓN:                                                                 ║
║    score = "overall" de melody_critic (0-1). Los empates se deshacen por     ║
║    menos saltos sin compensar y, luego, por más notas.                       ║
║                                                                              ║
║  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso/entorno · 2 ningún candidato válido║
║                                                                              ║
║  DEPENDENCIAS: stdlib · melody_critic.py (se invoca por subprocess)          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k: str) -> str:
    return _COLORS.get(k, "") if _USE_COLOR else ""


# ══════════════════════════════════════════════════════════════════════════════
#  CRÍTICA (subprocess a melody_critic.py)
# ══════════════════════════════════════════════════════════════════════════════

def find_critic(explicit: Optional[str] = None) -> str:
    cands = [explicit] if explicit else [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "melody_critic.py"),
        os.path.join(os.getcwd(), "melody_critic.py"),
    ]
    for p in cands:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "no se encuentra melody_critic.py (usa --critic RUTA o colócalo junto a "
        "melody_selector.py)")


def critique(path: str, critic: str, track: Optional[int] = None,
             timeout: int = 120) -> Dict:
    """Devuelve el JSON de melody_critic para `path`, o {'error': ...}."""
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="selcrit_")
    os.close(fd)
    try:
        cmd = [sys.executable, critic, path, "--json", tmp, "--no-color"]
        if track is not None:
            cmd += ["--track", str(track)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            msg = (p.stderr or p.stdout or "").strip().splitlines()
            return {"file": path, "error": msg[-1] if msg else f"rc={p.returncode}"}
        with open(tmp, encoding="utf-8") as fh:
            d = json.load(fh)
        d["file"] = path
        return d
    except subprocess.TimeoutExpired:
        return {"file": path, "error": f"timeout ({timeout}s)"}
    except Exception as e:  # noqa: BLE001
        return {"file": path, "error": str(e)}
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def sort_key(d: Dict):
    """Mayor es mejor. Los fallos/descartados van al final."""
    if d.get("error") or d.get("overall") is None:
        return (0, 0.0, 0.0, 0.0)
    return (1, float(d["overall"]), -float(d.get("unresolved_leaps", 0)),
            float(d.get("n_notes", 0)))


def rank_files(files: List[str], critic: str, track: Optional[int],
               min_notes: int, jobs: int) -> List[Dict]:
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        results = list(ex.map(lambda f: critique(f, critic, track), files))
    for d in results:
        if not d.get("error") and d.get("n_notes", 0) < min_notes:
            d["error"] = f"solo {d.get('n_notes', 0)} notas (< --min-notes {min_notes})"
    results.sort(key=sort_key, reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def print_table(results: List[Dict], title: str = "RANKING") -> None:
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}\n  {B}{title}{R}\n{'═' * 78}")
    print(f"  {G}{'#':>2}  {'score':>5}  {'notas':>5}  {'ámbito':>6}  {'grados':>6}  "
          f"{'saltos±':>7}  {'clímax':>6}  fichero{R}")
    for i, d in enumerate(results, 1):
        name = os.path.relpath(d["file"]) if os.path.exists(d["file"]) else d["file"]
        if d.get("error") or d.get("overall") is None:
            print(f"  {i:>2}  {_c('RED')}{'—':>5}{R}  {d.get('error', 'sin score')}  "
                  f"{G}{name}{R}")
            continue
        col = _c("GRN") if i == 1 else ""
        print(f"  {i:>2}  {col}{d['overall']:>5.2f}{R}  {d['n_notes']:>5}  "
              f"{d['range_semitones']:>4}st  {d['step_ratio']:>6.0%}  "
              f"{d['unresolved_leaps']:>7}  {d['peak_position']:>6.0%}  {name}")
    print(f"{'═' * 78}")


def write_report(path: str, results: List[Dict], winner: Optional[Dict],
                 extra: Optional[Dict] = None) -> None:
    keep = ("file", "overall", "n_notes", "range_semitones", "step_ratio",
            "big_leaps", "unresolved_leaps", "peak_position", "n_phrases",
            "error", "sugerencias")
    doc = {
        "version": VERSION,
        "winner": winner["file"] if winner else None,
        "winner_score": winner.get("overall") if winner else None,
        "candidates": [{k: d[k] for k in keep if k in d} for d in results],
    }
    if extra:
        doc.update(extra)
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"  {_c('CYA')}JSON: {path}{_c('R')}")


def best_of(results: List[Dict]) -> Optional[Dict]:
    if results and not results[0].get("error") and results[0].get("overall") is not None:
        return results[0]
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def expand_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
        files += [h for h in hits if os.path.isfile(h)]
    return list(dict.fromkeys(files))


def cmd_rank(a) -> int:
    files = expand_inputs(a.midis)
    if not files:
        print("[ERROR] no se encontró ningún MIDI de entrada", file=sys.stderr)
        return 1
    critic = find_critic(a.critic)
    res = rank_files(files, critic, a.track, a.min_notes, a.jobs)
    print_table(res)
    w = best_of(res)
    if w:
        print(f"  {_c('B')}Ganador:{_c('R')} {w['file']}  (score {w['overall']:.2f})\n")
    if a.json:
        write_report(a.json, res, w)
    return 0 if w else 2


def cmd_pick(a) -> int:
    files = expand_inputs(a.midis)
    if not files:
        print("[ERROR] no se encontró ningún MIDI de entrada", file=sys.stderr)
        return 1
    critic = find_critic(a.critic)
    res = rank_files(files, critic, a.track, a.min_notes, a.jobs)
    print_table(res, "SELECCIÓN")
    w = best_of(res)
    if not w:
        print("[ERROR] ningún candidato válido", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    shutil.copyfile(w["file"], a.output)
    print(f"  {_c('GRN')}✔ {w['file']} → {a.output}{_c('R')}  (score {w['overall']:.2f})\n")
    if a.json:
        write_report(a.json, res, w)
    return 0


def _render_cmd(template: str, seed: int, out: str, i: int) -> List[str]:
    toks = shlex.split(template)
    out_toks = []
    for t in toks:
        t = t.replace("{seed}", str(seed)).replace("{out}", out).replace("{i}", str(i))
        out_toks.append(t)
    if out_toks and out_toks[0] in ("python", "python3"):
        out_toks[0] = sys.executable
    return out_toks


def cmd_loop(a) -> int:
    if "{out}" not in a.cmd:
        print("[ERROR] --cmd debe contener el marcador {out}", file=sys.stderr)
        return 1
    critic = find_critic(a.critic)
    work = os.path.abspath(a.workdir)
    os.makedirs(work, exist_ok=True)
    all_results: List[Dict] = []
    first_score: Optional[float] = None
    stopped_at = None
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}\n  {B}MELODY SELECTOR · loop{R}  ({a.n} vueltas, "
          f"semillas {a.seed_start}…{a.seed_start + a.n - 1})\n{'═' * 78}")

    for i in range(a.n):
        seed = a.seed_start + i
        cdir = os.path.join(work, f"cand_{i:03d}_seed{seed}")
        os.makedirs(cdir, exist_ok=True)
        out = os.path.join(cdir, "out.mid")
        cmd = _render_cmd(a.cmd, seed, out, i)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=a.cwd, capture_output=True, text=True,
                               timeout=a.timeout)
            rc, log = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            rc, log = 124, f"timeout ({a.timeout}s)"
        except Exception as e:  # noqa: BLE001
            rc, log = 1, str(e)
        Path(cdir, "generator.log").write_text(
            "$ " + " ".join(shlex.quote(x) for x in cmd) + "\n\n" + log, encoding="utf-8")
        produced = sorted(glob.glob(os.path.join(cdir, "*.mid")))
        if not produced:
            print(f"  {_c('RED')}vuelta {i + 1}/{a.n} seed={seed}: sin MIDI "
                  f"(rc={rc}){R}  → {cdir}/generator.log")
            continue
        res = rank_files(produced, critic, a.track, a.min_notes, a.jobs)
        for d in res:
            d["seed"] = seed
        all_results += res
        top = best_of(res)
        sc = top["overall"] if top else None
        if first_score is None and sc is not None:
            first_score = sc
        print(f"  vuelta {i + 1:>2}/{a.n} seed={seed:<4} {len(produced)} cand.  "
              f"mejor={'—' if sc is None else f'{sc:.2f}'}  {G}({time.time() - t0:.1f}s){R}")
        if a.stop_at is not None and sc is not None and sc >= a.stop_at:
            stopped_at = i + 1
            print(f"  {_c('GRN')}→ umbral {a.stop_at:.2f} alcanzado, se detiene{R}")
            break

    all_results.sort(key=sort_key, reverse=True)
    print_table(all_results[:15], "MEJORES CANDIDATOS (top 15)")
    w = best_of(all_results)
    if not w:
        print("[ERROR] ningún candidato válido", file=sys.stderr)
        return 2
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    shutil.copyfile(w["file"], a.output)
    gain = "" if first_score is None else f"  (primer candidato: {first_score:.2f} → +{w['overall'] - first_score:.2f})"
    print(f"  {_c('GRN')}✔ ganador → {a.output}{R}  score {w['overall']:.2f}{gain}\n")
    if a.json:
        write_report(a.json, all_results, w,
                     {"rounds": a.n, "stopped_at_round": stopped_at,
                      "first_candidate_score": first_score})
    if not a.keep:
        win_dir = os.path.dirname(w["file"])
        for d in glob.glob(os.path.join(work, "cand_*")):
            if os.path.abspath(d) != os.path.abspath(win_dir):
                shutil.rmtree(d, ignore_errors=True)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="melody_selector.py",
        description="Selecciona la mejor melodía entre candidatos usando melody_critic.")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    def common(p):
        p.add_argument("--track", type=int, default=None)
        p.add_argument("--critic", default=None, help="ruta a melody_critic.py")
        p.add_argument("--min-notes", type=int, default=4)
        p.add_argument("--jobs", type=int, default=4)
        p.add_argument("--json", default=None)
        p.add_argument("--no-color", action="store_true")

    r = sub.add_parser("rank", help="puntúa y ordena MIDIs")
    r.add_argument("midis", nargs="+")
    common(r)
    r.set_defaults(fn=cmd_rank)

    k = sub.add_parser("pick", help="rank + copia el mejor")
    k.add_argument("midis", nargs="+")
    k.add_argument("--output", "-o", required=True)
    common(k)
    k.set_defaults(fn=cmd_pick)

    l = sub.add_parser("loop", help="generar N veces, criticar y elegir")
    l.add_argument("--cmd", required=True, help="comando generador con {seed} y {out}")
    l.add_argument("--output", "-o", required=True)
    l.add_argument("--n", type=int, default=8)
    l.add_argument("--seed-start", type=int, default=1)
    l.add_argument("--stop-at", type=float, default=None)
    l.add_argument("--workdir", default="selector_work")
    l.add_argument("--keep", action="store_true")
    l.add_argument("--cwd", default=None)
    l.add_argument("--timeout", type=int, default=300)
    common(l)
    l.set_defaults(fn=cmd_loop)
    return ap


def main(argv=None) -> int:
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if a.no_color:
        _USE_COLOR = False
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[interrumpido]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
