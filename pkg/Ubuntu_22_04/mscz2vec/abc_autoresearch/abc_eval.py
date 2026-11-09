"""
abc_eval.py — Evaluación musical compartida por abc_train.py,
abc_train_diffusion.py y abc_conditioned.py.

Pipeline:  muestras ABC  →  MIDI (abc2midi)  →  evaluador seleccionable.

EVALUADORES (backends)
----------------------
  preference   preference_trainer.py eval --json   escala 0-1
               Modelo de preferencias musicales personal (necesita un modelo
               entrenado: prefs.prefs.json, prefs.linear.json u otro tipo).
  quality      quality_scorer.py --quiet           escala 0-1
               Scorer automático multidimensional (formal, coherence, arc,
               melodic...). No necesita entrenamiento previo.

SELECCIÓN
---------
  1. Variable de entorno  ABC_EVAL_BACKEND=quality|preference   (prioridad)
  2. Variable EVAL_BACKEND del script de entrenamiento
  Elige UNO al empezar una sesión y no lo cambies entre experimentos: las
  escalas son distintas y los combined_score dejarían de ser comparables.

SALIDA NORMALIZADA
------------------
  reward_score = pref_score / PREF_SCALE   (backend preference; PREF_SCALE = 1.0)
  reward_score = quality_score             (backend quality)
  combined_score = (1 - w) * (1 / val_bpb) + w * reward_score

  OJO: preference_trainer.py devuelve `score` en [0, 1] (sigmoide del modelo; sus
  valoraciones 1-5 se mapean con (r-1)/4), NO en 0-5 como asumían las versiones
  anteriores de este proyecto, que dividían por 5. Por eso los combined_score con
  backend preference de sesiones antiguas no son comparables con los actuales.

LOCALIZACIÓN DE LOS EVALUADORES
-------------------------------
  preference_trainer.py y quality_scorer.py se buscan, por este orden, en:
  la ruta indicada (PREF_TRAINER_PATH / QUALITY_SCORER_PATH o las variables
  ABC_PREF_TRAINER / ABC_QUALITY_SCORER), el directorio de trabajo, la carpeta
  de este proyecto y su carpeta padre (donde viven en mscz2vec/).
  El modelo de preferencias (PREF_MODEL) y el corpus (PREF_CORPUS) se buscan
  en esos mismos tres directorios.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

BACKENDS = ("preference", "quality")
PREF_SCALE = 1.0           # preference_trainer.py eval devuelve score en [0, 1]
# (sufijo del fichero de modelo, --model-type). `prefs.json` es el formato legado
# (<base>.prefs.json) que preference_trainer.py eval exige que exista; los tipados
# se guardan como <base>.<tipo>.json (no <base>.prefs.<tipo>.json).
_PREF_FILES = (("prefs.json", None), ("linear.json", None), ("neural.json", "neural"),
               ("kernel.json", "kernel"), ("gp.json", "gp"), ("svm.json", "svm"))


# ---------------------------------------------------------------------------
# Selección del backend y localización de ficheros
# ---------------------------------------------------------------------------

def resolve_backend(default="preference"):
    """ABC_EVAL_BACKEND (si existe) tiene prioridad sobre `default`."""
    name = (os.environ.get("ABC_EVAL_BACKEND") or default or "").strip().lower()
    if name not in BACKENDS:
        raise ValueError(
            f"EVAL_BACKEND/ABC_EVAL_BACKEND = {name!r} no es válido; "
            f"opciones: {', '.join(BACKENDS)}")
    return name


def _search_dirs():
    dirs, seen = [], set()
    for d in (Path.cwd(), HERE, HERE.parent):
        if d not in seen:
            seen.add(d)
            dirs.append(d)
    return dirs


def find_file(name, override=None, env_var=None):
    """Devuelve la ruta de `name` (override > variable de entorno > búsqueda) o None."""
    for cand in (override, os.environ.get(env_var) if env_var else None):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file():
                return p
    for d in _search_dirs():
        p = d / name
        if p.is_file():
            return p
    return None


def _find_prefs_model(pref_model):
    """Devuelve (base_para_--model, model_type|None, fichero) o None.

    Busca <base>.prefs.json (legado), <base>.linear.json y los modelos tipados
    <base>.{neural,kernel,gp,svm}.json en cwd, carpeta del proyecto y carpeta
    padre. Para los tipos distintos del lineal se pasa --model-type.
    Nota: `preference_trainer.py eval` comprueba la existencia de
    <base>.prefs.json; si solo hay ficheros tipados, su mensaje de error se
    muestra tal cual."""
    base_path = Path(pref_model).expanduser()
    bases = [base_path] if base_path.is_absolute() else [d / base_path for d in _search_dirs()]
    for base in bases:
        for suffix, mtype in _PREF_FILES:
            f = Path(f"{base}.{suffix}")
            if f.is_file():
                arg = pref_model if base == base_path else str(base)
                return arg, mtype, str(f)
    return None


# ---------------------------------------------------------------------------
# ABC → MIDI
# ---------------------------------------------------------------------------

def convert_abc_to_midi(abc_text, midi_path):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".abc",
                                     delete=False, encoding="utf-8") as f:
        f.write(abc_text)
        abc_path = f.name
    try:
        result = subprocess.run(
            ["abc2midi", abc_path, "-o", midi_path],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0 and Path(midi_path).exists()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        try:
            os.unlink(abc_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Evaluadores
# ---------------------------------------------------------------------------

def _tail(text, n=3):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return " | ".join(lines[-n:])


def _score_preference(midi_files, trainer_path, pref_model, pref_corpus, timeout):
    """Media (0-1) de los `score` de preference_trainer.py eval --json, o None."""
    trainer = find_file("preference_trainer.py", trainer_path, "ABC_PREF_TRAINER")
    if trainer is None:
        print("  [eval:preference] No se encuentra preference_trainer.py "
              "(busqué en: " + ", ".join(str(d) for d in _search_dirs()) + ")")
        return None
    found = _find_prefs_model(pref_model)
    if found is None:
        print(f"  [eval:preference] No se encontró {pref_model}.prefs.json — saltando")
        return None
    model_arg, model_type, model_file = found

    cmd = [sys.executable, str(trainer), "eval", *midi_files, "--model", model_arg, "--json"]
    if model_type:
        cmd += ["--model-type", model_type]
    if pref_corpus:
        corpus = find_file(Path(pref_corpus).name, pref_corpus) if not Path(pref_corpus).is_file() \
            else Path(pref_corpus)
        if corpus is not None:
            cmd += ["--corpus", str(corpus)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  [eval:preference] Error: {e}")
        return None
    scores = _parse_pref_json(r.stdout)
    if scores:
        return sum(scores) / len(scores)
    print(f"  [eval:preference] Sin puntuaciones (rc={r.returncode}): {_tail(r.stderr or r.stdout)}")
    return None


def _parse_pref_json(out):
    """`preference_trainer.py eval --json` imprime, tras el texto de resumen, una
    LISTA JSON de entradas {path, score, pref_score, percentile, ...}. Devuelve
    los `score` válidos (0..PREF_SCALE) de todas las entradas."""
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"(?m)^\[", out))):
        try:
            data, _ = dec.raw_decode(out[m.start():])
        except ValueError:
            continue
        if isinstance(data, list):
            return [e["score"] for e in data
                    if isinstance(e, dict) and isinstance(e.get("score"), (int, float))
                    and 0.0 <= e["score"] <= PREF_SCALE]
    return []


_QUIET_LINE = re.compile(r"^\s*(\d+(?:\.\d+)?)(?:\s|$)")


def _score_quality(midi_files, scorer_path, extra_args, timeout):
    """Media (0-1) de quality_scorer.py --quiet, o None.

    --quiet imprime «0.xxxx» (un fichero) o «0.xxxx  ruta» por fichero."""
    scorer = find_file("quality_scorer.py", scorer_path, "ABC_QUALITY_SCORER")
    if scorer is None:
        print("  [eval:quality] No se encuentra quality_scorer.py "
              "(busqué en: " + ", ".join(str(d) for d in _search_dirs()) + ")")
        return None
    cmd = [sys.executable, str(scorer), *midi_files, "--quiet", *extra_args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"  [eval:quality] Error: {e}")
        return None
    scores = []
    for line in r.stdout.splitlines():
        m = _QUIET_LINE.match(line)
        if m:
            v = float(m.group(1))
            if 0.0 <= v <= 1.0:
                scores.append(v)
    if scores:
        return sum(scores) / len(scores)
    print(f"  [eval:quality] Sin puntuaciones (rc={r.returncode}): {_tail(r.stderr or r.stdout)}")
    return None


def score_midi_files(midi_files, backend, *,
                     pref_trainer_path=None, pref_model="prefs", pref_corpus=None,
                     quality_scorer_path=None, quality_extra_args=(), timeout=None):
    """Puntúa MIDIs con el backend elegido.

    Devuelve {"backend", "reward_score" (0-1 | None), "pref_score" (0-1 | None),
              "quality_score" (0-1 | None)}. Solo se rellena el del backend usado.
    """
    res = {"backend": backend, "reward_score": None,
           "pref_score": None, "quality_score": None}
    if not midi_files:
        return res
    if backend == "preference":
        raw = _score_preference(midi_files, pref_trainer_path, pref_model, pref_corpus,
                                timeout or 60)
        if raw is not None:
            res["pref_score"] = round(raw, 4)
            res["reward_score"] = round(raw / PREF_SCALE, 4)
    elif backend == "quality":
        raw = _score_quality(midi_files, quality_scorer_path, list(quality_extra_args or ()),
                             timeout or max(60, 20 * len(midi_files)))
        if raw is not None:
            res["quality_score"] = round(raw, 4)
            res["reward_score"] = round(raw, 4)
    else:
        raise ValueError(f"backend desconocido: {backend!r}")
    return res


def evaluate_samples(sampler, is_valid, n_samples, backend, **score_kwargs):
    """Genera `n_samples` piezas con `sampler()` (→ texto ABC), filtra las
    válidas, las convierte a MIDI y las puntúa.

    Devuelve el dict de score_midi_files() más "validity_ratio" y "n_valid"."""
    print(f"\n  [eval:{backend}] Generando {n_samples} muestras ABC...")
    valid = []
    for _ in range(n_samples):
        text = sampler()
        if is_valid(text):
            valid.append(text)
    n_valid = len(valid)
    ratio = round(n_valid / max(1, n_samples), 3)
    print(f"  [eval:{backend}] {n_valid}/{n_samples} válidas ({100 * n_valid // max(1, n_samples)}%)")

    res = {"backend": backend, "reward_score": None, "pref_score": None,
           "quality_score": None, "validity_ratio": ratio, "n_valid": n_valid}
    if n_valid == 0:
        return res

    tmpdir = tempfile.mkdtemp(prefix="abc_eval_")
    midi_files = []
    try:
        for i, abc_text in enumerate(valid):
            midi_path = os.path.join(tmpdir, f"sample_{i:03d}.mid")
            if convert_abc_to_midi(abc_text, midi_path):
                midi_files.append(midi_path)
        print(f"  [eval:{backend}] {len(midi_files)}/{n_valid} convertidos a MIDI")
        res.update(score_midi_files(midi_files, backend, **score_kwargs))
        res["validity_ratio"] = ratio
    finally:
        for f in midi_files:
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
    return res


# ---------------------------------------------------------------------------
# Métrica compuesta y resumen en consola
# ---------------------------------------------------------------------------

def combined_score(val_bpb, reward_score, weight):
    """(1 - w) * (1/val_bpb) + w * reward_score, o None si no hay reward."""
    if reward_score is None:
        return None
    return (1 - weight) * (1.0 / max(val_bpb, 1e-6)) + weight * reward_score


def summary_lines(backend, result):
    """Líneas del resumen final, idénticas en los tres scripts de entrenamiento.

    Mantiene `pref_score:` (compatibilidad con greps y results.tsv antiguos) y
    añade `eval_backend:`, `reward_score:` y `quality_score:`."""
    def fmt(v, d=4):
        return f"{v:.{d}f}" if v is not None else "N/A"
    r = result or {}
    return [
        f"{'eval_backend:':<18}{backend}",
        f"{'reward_score:':<18}{fmt(r.get('reward_score'))}",
        f"{'pref_score:':<18}{fmt(r.get('pref_score'))}",
        f"{'quality_score:':<18}{fmt(r.get('quality_score'))}",
    ]
