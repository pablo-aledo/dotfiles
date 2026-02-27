#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          RUNNER  v1.0                                        ║
║         Ejecutor de pipelines de composición                                 ║
║                                                                              ║
║  Lee un plan .yaml (de theorist.py o narrator.py) y ejecuta cada paso       ║
║  del pipeline en orden, con logging, gestión de errores, puntos de pausa    ║
║  interactivos y capacidad de reanudar pipelines interrumpidos.               ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] CARGA     — lee el .yaml y detecta su formato (theorist / narrator)    ║
║  [2] VALIDACIÓN— verifica que los scripts existen y los inputs están listos ║
║  [3] EJECUCIÓN — lanza cada paso, registra stdout/stderr, mide duración     ║
║  [4] CHECKPOINT— guarda estado tras cada paso para poder reanudar           ║
║  [5] INFORME   — resumen de lo ejecutado con tiempos y archivos generados   ║
║                                                                              ║
║  USO:                                                                        ║
║    python runner.py obra_plan.yaml                                           ║
║    python runner.py obra_plan.yaml --pause-after tension_designer           ║
║    python runner.py obra_plan.yaml --from midi_dna_unified                  ║
║    python runner.py obra_plan.yaml --only midi_dna_unified reharmonizer     ║
║    python runner.py obra_plan.yaml --until orchestrator                     ║
║    python runner.py obra_plan.yaml --dry-run                                ║
║    python runner.py obra_plan.yaml --resume                                 ║
║    python runner.py --status obra_plan.yaml                                 ║
║    python runner.py --list-steps obra_plan.yaml                             ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    plan              Archivo .yaml del pipeline (theorist o narrator)        ║
║    --from STEP       Empezar desde este paso (saltar anteriores)             ║
║    --until STEP      Ejecutar hasta este paso (inclusive)                    ║
║    --only STEP...    Ejecutar solo estos pasos específicos                   ║
║    --pause-after S   Pausar y pedir confirmación tras cada paso S            ║
║    --pause-all       Pausar tras cada paso                                   ║
║    --resume          Reanudar desde el último checkpoint guardado            ║
║    --retry-failed    Reintentar solo los pasos que fallaron                  ║
║    --dry-run         Mostrar los comandos sin ejecutar nada                  ║
║    --status          Mostrar el estado del último run de este plan           ║
║    --list-steps      Listar los pasos del plan sin ejecutar                  ║
║    --output-dir DIR  Directorio de salida (default: junto al plan)           ║
║    --log FILE        Archivo de log adicional (default: <plan>.runner.log)   ║
║    --no-log          No escribir archivo de log                              ║
║    --timeout N       Timeout por paso en segundos (default: 300)            ║
║    --verbose         Mostrar stdout/stderr de cada script en tiempo real     ║
║    --strict          Abortar al primer fallo (default: continuar y reportar) ║
║                                                                              ║
║  FORMATOS DE PLAN SOPORTADOS:                                                ║
║    theorist  — plan con steps.cmd (comandos CLI directos)                   ║
║    narrator  — plan con sections + pipeline_steps estructurados             ║
║    manual    — lista de steps con name+cmd libres                            ║
║                                                                              ║
║  ARCHIVOS GENERADOS:                                                         ║
║    <plan>.runner.log      — log completo de la ejecución                    ║
║    <plan>.runner.json     — checkpoint de estado (para --resume)            ║
║                                                                              ║
║  DEPENDENCIAS: ninguna (solo stdlib)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import time
import re
import textwrap
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import OrderedDict

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0"

# Colores ANSI (desactivados si no hay terminal)
def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

_COLOR = _supports_color()

def _c(code: str, text: str) -> str:
    if not _COLOR:
        return text
    codes = {
        "green":  "\033[92m", "red":    "\033[91m", "yellow": "\033[93m",
        "blue":   "\033[94m", "cyan":   "\033[96m", "gray":   "\033[90m",
        "bold":   "\033[1m",  "reset":  "\033[0m",
    }
    return f"{codes.get(code, '')}{text}{codes['reset']}"

# Estados de paso
ST_PENDING  = "pending"
ST_RUNNING  = "running"
ST_OK       = "ok"
ST_FAILED   = "failed"
ST_SKIPPED  = "skipped"
ST_TIMEOUT  = "timeout"

STATUS_ICON = {
    ST_PENDING: "○",
    ST_RUNNING: "◎",
    ST_OK:      "✓",
    ST_FAILED:  "✗",
    ST_SKIPPED: "⊘",
    ST_TIMEOUT: "⏱",
}

STATUS_COLOR = {
    ST_PENDING: "gray",
    ST_RUNNING: "cyan",
    ST_OK:      "green",
    ST_FAILED:  "red",
    ST_SKIPPED: "yellow",
    ST_TIMEOUT: "yellow",
}


# ══════════════════════════════════════════════════════════════════════════════
#  PARSEO DE YAML
#  Usa PyYAML si está disponible; fallback a parser mínimo de stdlib.
# ══════════════════════════════════════════════════════════════════════════════

try:
    import yaml as _yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _yaml_parse(text: str) -> dict:
    """Parsea YAML usando pyyaml o fallback stdlib."""
    if _HAS_YAML:
        result = _yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    # Fallback muy simple: solo para planes sin anidamiento complejo
    return _yaml_parse_simple(text)


def _yaml_parse_simple(text: str) -> dict:
    """Parser YAML mínimo de emergencia (sin pyyaml)."""
    result: dict = {}
    current_list_key = None
    current_item: dict | None = None
    list_indent = -1

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if stripped.startswith("- "):
            rest = stripped[2:].strip()
            if current_list_key is None:
                continue
            if ":" in rest:
                current_item = {}
                _simple_kv(rest, current_item)
                result[current_list_key].append(current_item)
            else:
                result[current_list_key].append(rest)
                current_item = None
        elif ":" in stripped and not stripped.startswith("-"):
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if indent == 0:
                if val == "":
                    result[key] = []
                    current_list_key = key
                    current_item = None
                    list_indent = indent
                else:
                    result[key] = _simple_cast(val)
                    current_list_key = None
            elif current_item is not None:
                current_item[key] = _simple_cast(val)

    return result


def _simple_kv(text: str, target: dict):
    key, _, val = text.partition(":")
    k = key.strip()
    v = val.strip()
    if k:
        target[k] = _simple_cast(v)


def _simple_cast(val: str):
    if val in ("true", "True"):  return True
    if val in ("false", "False"): return False
    if val in ("null", "None", "~", ""): return None
    try: return int(val)
    except ValueError: pass
    try: return float(val)
    except ValueError: pass
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    return val


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZACIÓN DE PLAN
#  Convierte cualquier formato de .yaml a una lista uniforme de PipelineStep
# ══════════════════════════════════════════════════════════════════════════════

class PipelineStep:
    """Representa un paso normalizado del pipeline."""

    def __init__(self, name: str, cmd: list[str],
                 output_files: list[str] | None = None,
                 input_files: list[str] | None = None,
                 label: str | None = None,
                 optional: bool = False):
        self.name         = name
        self.cmd          = cmd           # lista de tokens
        self.output_files = output_files or []
        self.input_files  = input_files or []
        self.label        = label or name
        self.optional     = optional

        # Estado de ejecución (rellenado durante el run)
        self.status       = ST_PENDING
        self.returncode   = None
        self.stdout       = ""
        self.stderr       = ""
        self.duration_s   = 0.0
        self.started_at   = None
        self.finished_at  = None
        self.error_msg    = ""

    def cmd_str(self) -> str:
        """Comando como string legible."""
        return " ".join(str(t) for t in self.cmd)

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "label":        self.label,
            "cmd":          self.cmd_str(),
            "status":       self.status,
            "returncode":   self.returncode,
            "duration_s":   round(self.duration_s, 2),
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "output_files": self.output_files,
            "error_msg":    self.error_msg,
        }


def _tokenize_cmd(cmd_str: str) -> list[str]:
    """
    Convierte un comando CLI string en lista de tokens.
    Respeta comillas dobles y simples.
    Elimina artefactos de continuacion YAML (\ solitarios).

    Caso especial: los valores de curvas --mt-* tienen la forma
    "BAR:VAL, BAR:VAL, ..." con coma+espacio interna. Cuando el YAML ya
    consumio las comillas externas el split por espacios parte estos valores
    en tokens sueltos (ej: "0:sparse," y "16:dense"). Esta funcion los reune.
    """
    cleaned = re.sub(r'\s*\\\s*', ' ', cmd_str).strip()
    tokens = []
    current = ""
    in_quote = None
    for ch in cleaned:
        if ch in ('"', "'") and in_quote is None:
            in_quote = ch
        elif ch == in_quote:
            in_quote = None
        elif ch == " " and in_quote is None:
            if current:
                tokens.append(current)
                current = ""
        else:
            current += ch
    if current:
        tokens.append(current)
    tokens = [t for t in tokens if t]

    # Reparar fragmentos de curvas MT partidos por coma+espacio.
    # Patron: flag --mt-* seguido de tokens con forma "N:valor" o "N:valor,"
    # También maneja el caso donde el LLM omite el prefijo de compás en el
    # primer token (ej: "0.0, 16:0.0" en lugar de "0:0.0, 16:0.0")
    _BAR_VAL   = re.compile(r'^\d+:[^\s,]+,?$')   # "0:sparse," o "16:0.5"
    _BARE_VAL  = re.compile(r'^[^\-][^\s,]*,?$')  # valor suelto sin prefijo N:
    _MT_FLAG   = re.compile(r'^--mt-')

    def _is_mt_value(tok):
        """Token que puede ser parte de una curva MT (con o sin prefijo de compás)."""
        return bool(_BAR_VAL.match(tok) or _BARE_VAL.match(tok))

    repaired = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if repaired and _MT_FLAG.match(repaired[-1]) and _is_mt_value(tok):
            # Normalizar: si el primer fragmento no tiene N:, añadir "0:"
            if not _BAR_VAL.match(tok):
                tok = '0:' + tok.lstrip(',')
            parts = [tok]
            while parts[-1].endswith(',') and i + 1 < len(tokens):
                i += 1
                next_tok = tokens[i]
                # Normalizar siguientes tokens sin prefijo también
                if _BARE_VAL.match(next_tok) and not _BAR_VAL.match(next_tok):
                    # No podemos inferir el compás correcto para los siguientes,
                    # así que los dejamos tal cual (el LLM debería ponerlos bien)
                    pass
                parts.append(next_tok)
            repaired.append(', '.join(p.rstrip(',') for p in parts))
        else:
            repaired.append(tok)
        i += 1

    return repaired
def _normalize_cmd_tokens(tokens: list[str]) -> list[str]:
    """
    Reemplaza 'python' por sys.executable para garantizar el intérprete correcto.
    """
    result = []
    for t in tokens:
        if t == "python" or t == "python3":
            result.append(sys.executable)
        else:
            result.append(t)
    return result


def _extract_output_from_cmd(tokens: list[str]) -> list[str]:
    """Extrae el valor de --output de la lista de tokens."""
    outputs = []
    for i, t in enumerate(tokens):
        if t in ("--output", "-o") and i + 1 < len(tokens):
            outputs.append(tokens[i + 1])
    return outputs


def normalize_theorist_plan(raw: dict, plan_dir: Path) -> list[PipelineStep]:
    """
    Normaliza un plan de formato theorist (steps con cmd: directo).
    """
    steps_raw = raw.get("steps", [])
    if not steps_raw:
        return []

    steps = []
    for s in steps_raw:
        if not isinstance(s, dict):
            continue
        name    = str(s.get("name", f"step_{len(steps)+1}"))
        cmd_raw = str(s.get("cmd", ""))
        output  = s.get("output", None)

        tokens = _normalize_cmd_tokens(_tokenize_cmd(cmd_raw))
        if not tokens:
            continue

        # Resolver rutas relativas al directorio del plan
        resolved = _resolve_script_tokens(tokens, plan_dir)

        out_files = [str(plan_dir / output)] if output else _extract_output_from_cmd(resolved)

        steps.append(PipelineStep(
            name=name,
            cmd=resolved,
            output_files=out_files,
            label=s.get("label", name),
        ))

    return steps


def normalize_narrator_plan(raw: dict, plan_dir: Path) -> list[PipelineStep]:
    """
    Normaliza un plan de formato narrator (sections + pipeline_steps).
    """
    steps = []
    arc   = raw.get("arc", "obra")
    key   = raw.get("key", "C")
    tempo = raw.get("tempo", 120)
    bars  = raw.get("bars", 32)

    section_list = raw.get("sections", [])
    pipeline     = raw.get("pipeline_steps", raw.get("pipeline", []))

    # Si hay pipeline_steps con estructura completa (de narrator.py)
    if pipeline and isinstance(pipeline[0], dict) and "tool" in pipeline[0]:
        for ps in pipeline:
            tool    = str(ps.get("tool", "midi_dna_unified.py"))
            label   = str(ps.get("label", tool))
            step_n  = ps.get("step", len(steps)+1)
            name    = f"step_{step_n:02d}_{tool.replace('.py','')}"
            output  = ps.get("output", "")
            inputs  = ps.get("inputs", [])
            flags   = ps.get("flags", {})

            script  = _find_script(tool, plan_dir)
            if not script:
                script = str(plan_dir / tool)

            cmd = [sys.executable, script]

            # Añadir inputs como argumentos posicionales
            for inp in inputs:
                cmd.append(str(plan_dir / inp))

            # Añadir flags
            for flag, val in flags.items():
                if val is True:
                    cmd.append(flag)
                elif val is not None and val is not False:
                    cmd += [flag, str(val)]

            if output and "--output" not in cmd:
                cmd += ["--output", str(plan_dir / output)]

            out_files = [str(plan_dir / output)] if output else []
            inp_files = [str(plan_dir / f) for f in inputs]

            steps.append(PipelineStep(
                name=name, cmd=cmd,
                output_files=out_files,
                input_files=inp_files,
                label=label,
            ))
        return steps

    # Si hay sections (lista de secciones) con pipeline inline (de theorist._plan.json)
    if pipeline and isinstance(pipeline[0], dict) and "step" in pipeline[0]:
        for ps in pipeline:
            step_name = str(ps.get("step", "step"))
            params    = ps.get("params", {})
            script    = _find_script(f"{step_name}.py", plan_dir)
            if not script:
                script = str(plan_dir / f"{step_name}.py")

            cmd = [sys.executable, script]
            for flag, val in params.items():
                if val is None:
                    continue
                if isinstance(val, bool):
                    if val:
                        cmd.append(flag)
                else:
                    cmd += [flag, str(val)]

            out_files = _extract_output_from_cmd(cmd)
            steps.append(PipelineStep(
                name=step_name, cmd=cmd,
                output_files=out_files,
                label=step_name,
            ))
        return steps

    # Fallback: generar steps desde secciones con midi_dna_unified
    for i, sec in enumerate(section_list):
        sec_id    = str(sec.get("id", f"sec_{i+1}"))
        sec_bars  = int(sec.get("bars", bars))
        sec_key   = str(sec.get("key", key))
        sec_label = str(sec.get("label", sec_id))
        out_file  = str(plan_dir / f"seccion_{i+1:02d}_{sec_id}.mid")

        script = _find_script("midi_dna_unified.py", plan_dir)
        if not script:
            script = str(plan_dir / "midi_dna_unified.py")

        cmd = [
            sys.executable, script,
            "--key", sec_key,
            "--bars", str(sec_bars),
            "--tempo", str(tempo),
            "--mode", "auto",
            "--export-fingerprint",
            "--output", out_file,
        ]

        steps.append(PipelineStep(
            name=f"midi_dna_{sec_id}",
            cmd=cmd,
            output_files=[out_file],
            label=f"Generar sección {sec_label}",
        ))

    # Paso de stitcher si hay más de una sección
    if len(section_list) > 1:
        final_out  = str(plan_dir / f"obra_{arc}_final.mid")
        fp_inputs  = [str(plan_dir / f"seccion_{i+1:02d}_{sec.get('id','s')}.fingerprint.json")
                      for i, sec in enumerate(section_list)]
        script = _find_script("stitcher.py", plan_dir)
        if not script:
            script = str(plan_dir / "stitcher.py")
        cmd = [sys.executable, script] + fp_inputs + ["--output", final_out]
        steps.append(PipelineStep(
            name="stitcher",
            cmd=cmd,
            output_files=[final_out],
            input_files=fp_inputs,
            label="Ensamblar secciones",
        ))
    else:
        final_out = str(plan_dir / f"seccion_01_{section_list[0].get('id','s')}.mid") if section_list else "obra.mid"

    # Orchestrator
    orch_out = str(plan_dir / f"obra_{arc}_orquestada.mid")
    script   = _find_script("orchestrator.py", plan_dir)
    if not script:
        script = str(plan_dir / "orchestrator.py")
    steps.append(PipelineStep(
        name="orchestrator",
        cmd=[sys.executable, script, final_out, "--template", "chamber",
             "--output", orch_out],
        output_files=[orch_out],
        input_files=[final_out],
        label="Orquestar",
    ))

    return steps


def load_plan(yaml_path: str) -> tuple[list[PipelineStep], dict, str]:
    """
    Carga y normaliza un plan YAML.
    Retorna (steps, raw_dict, formato_detectado).
    Formatos: 'theorist', 'narrator', 'manual'
    """
    plan_path = Path(yaml_path).resolve()
    plan_dir  = plan_path.parent

    with open(plan_path, "r", encoding="utf-8") as f:
        text = f.read()

    raw = _yaml_parse(text)

    # Detectar formato
    if "steps" in raw and raw["steps"] and isinstance(raw["steps"][0], dict) and "cmd" in str(raw["steps"][0]):
        fmt = "theorist"
        steps = normalize_theorist_plan(raw, plan_dir)
    elif "sections" in raw or "pipeline_steps" in raw:
        fmt = "narrator"
        steps = normalize_narrator_plan(raw, plan_dir)
    elif "steps" in raw:
        # formato manual genérico
        fmt = "manual"
        steps = normalize_theorist_plan(raw, plan_dir)
    else:
        fmt = "unknown"
        steps = []

    return steps, raw, fmt


def load_narrator_json(json_path: str) -> tuple[list[PipelineStep], dict, str]:
    """Carga un _plan.json generado por theorist/narrator (formato JSON)."""
    plan_path = Path(json_path).resolve()
    plan_dir  = plan_path.parent

    with open(plan_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    steps = normalize_narrator_plan(raw, plan_dir)
    return steps, raw, "narrator_json"


# ══════════════════════════════════════════════════════════════════════════════
#  LOCALIZACIÓN DE SCRIPTS
# ══════════════════════════════════════════════════════════════════════════════

def _find_script(name: str, plan_dir: Path) -> str | None:
    """Busca un script en plan_dir, CWD y PATH."""
    candidates = [
        plan_dir / name,
        Path.cwd() / name,
        Path(__file__).parent / name,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which(name)
    return found


def _resolve_script_tokens(tokens: list[str], plan_dir: Path) -> list[str]:
    """
    Para cada token que parece un script .py, intenta resolverlo a ruta absoluta.
    """
    resolved = []
    for i, t in enumerate(tokens):
        if t.endswith(".py") and not t.startswith("/") and not t.startswith("."):
            script = _find_script(t, plan_dir)
            resolved.append(script if script else t)
        elif t.endswith(".py") and (t.startswith("./") or not t.startswith("/")):
            candidate = plan_dir / t
            resolved.append(str(candidate) if candidate.exists() else t)
        else:
            resolved.append(t)
    return resolved


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

class Checkpoint:
    """Persiste el estado del pipeline para reanudar ejecuciones interrumpidas."""

    def __init__(self, plan_path: str):
        self.path = str(Path(plan_path).with_suffix(".runner.json"))
        self.data: dict = {}

    def load(self) -> dict:
        if Path(self.path).exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}
        return self.data

    def save(self, run_info: dict, steps: list[PipelineStep]):
        self.data = {
            "version":    VERSION,
            "plan":       run_info.get("plan_path", ""),
            "started_at": run_info.get("started_at", ""),
            "last_saved": datetime.now().isoformat(),
            "steps": [s.to_dict() for s in steps],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def restore_into(self, steps: list[PipelineStep]) -> int:
        """
        Restaura el estado desde checkpoint en los steps.
        Retorna el índice del primer paso no completado.
        """
        saved = {s["name"]: s for s in self.data.get("steps", [])}
        resume_from = 0
        for i, step in enumerate(steps):
            if step.name in saved:
                s = saved[step.name]
                step.status      = s.get("status", ST_PENDING)
                step.returncode  = s.get("returncode")
                step.duration_s  = s.get("duration_s", 0.0)
                step.started_at  = s.get("started_at")
                step.finished_at = s.get("finished_at")
                step.error_msg   = s.get("error_msg", "")
                if step.status == ST_OK:
                    resume_from = i + 1
        return resume_from

    def clear(self):
        if Path(self.path).exists():
            Path(self.path).unlink()


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class RunLogger:
    """Escribe log en archivo y opcionalmente en stdout."""

    def __init__(self, log_path: str | None, verbose: bool = False):
        self.log_path = log_path
        self.verbose  = verbose
        self._file    = None
        if log_path:
            try:
                self._file = open(log_path, "w", encoding="utf-8")
            except Exception as e:
                print(f"  [warn] No se pudo abrir log {log_path}: {e}")

    def write(self, text: str, echo: bool = True):
        """Escribe en log y opcionalmente en stdout."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {text}"
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()
        if echo:
            print(text)

    def write_raw(self, text: str):
        """Escribe texto sin timestamp (para stdout/stderr de subprocesos)."""
        if self._file:
            self._file.write(text)
            self._file.flush()
        if self.verbose:
            print(text, end="")

    def close(self):
        if self._file:
            self._file.close()
            self._file = None


# ══════════════════════════════════════════════════════════════════════════════
#  EJECUCIÓN DE PASO
# ══════════════════════════════════════════════════════════════════════════════

def _spinner_thread(stop_event: threading.Event, step_name: str):
    """Spinner visual mientras el proceso corre."""
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i = 0
    while not stop_event.is_set():
        frame = frames[i % len(frames)]
        print(f"\r  {_c('cyan', frame)} {step_name} ...", end="", flush=True)
        time.sleep(0.1)
        i += 1
    print("\r" + " " * (len(step_name) + 12) + "\r", end="", flush=True)


def execute_step(step: PipelineStep, timeout: int, verbose: bool,
                 logger: RunLogger, work_dir: Path) -> bool:
    """
    Ejecuta un paso del pipeline.
    Retorna True si tuvo éxito (returncode == 0 y outputs existen si se declararon).
    """
    step.status     = ST_RUNNING
    step.started_at = datetime.now().isoformat()
    start_time      = time.time()

    logger.write(f"  Ejecutando: {step.label}", echo=False)

    import os as _os
    cmd = list(step.cmd)
    script_name = Path(cmd[1]).name if len(cmd) > 1 else ""

    if script_name == "tension_designer.py":
        if "--no-gui" not in cmd:
            cmd.append("--no-gui")
        if "--auto-generate" not in cmd and "--preset" not in cmd:
            cmd += ["--preset", "arch"]
        # Eliminar argumentos posicionales que sean ficheros inexistentes
        # (evita que tension_designer falle buscando obra.mid si no se genero)
        import os as _os2
        cmd = [t for t in cmd
               if t.startswith('-') or not t.endswith('.mid')
               or _os2.path.exists(t)]

    logger.write(f"  CMD: {' '.join(str(t) for t in cmd)}", echo=False)

    _env = _os.environ.copy()
    _env["PYTHONWARNINGS"] = "ignore"

    # Spinner en modo no-verbose
    stop_spinner = threading.Event()
    if not verbose and _COLOR:
        t = threading.Thread(target=_spinner_thread,
                             args=(stop_spinner, step.label), daemon=True)
        t.start()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(work_dir),
            env=_env,
        )

        stdout_lines = []
        stderr_lines = []

        # Leer output en tiempo real si verbose
        if verbose:
            def _read_stream(stream, store, prefix):
                for line in stream:
                    store.append(line)
                    logger.write_raw(f"    {prefix} {line}")
            t_out = threading.Thread(target=_read_stream,
                                     args=(proc.stdout, stdout_lines, "│"))
            t_err = threading.Thread(target=_read_stream,
                                     args=(proc.stderr, stderr_lines, "│"))
            t_out.start(); t_err.start()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                step.status    = ST_TIMEOUT
                step.error_msg = f"Timeout tras {timeout}s"
                return False
            t_out.join(); t_err.join()
        else:
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=timeout)
                stdout_lines = stdout_raw.splitlines(keepends=True)
                stderr_lines = stderr_raw.splitlines(keepends=True)
                logger.write_raw("".join(stdout_lines))
                logger.write_raw("".join(stderr_lines))
            except subprocess.TimeoutExpired:
                proc.kill()
                step.status    = ST_TIMEOUT
                step.error_msg = f"Timeout tras {timeout}s"
                return False

    except FileNotFoundError as e:
        step.status    = ST_FAILED
        step.error_msg = f"Script no encontrado: {e}"
        return False
    except Exception as e:
        step.status    = ST_FAILED
        step.error_msg = str(e)
        return False
    finally:
        stop_spinner.set()

    step.returncode  = proc.returncode
    step.stdout      = "".join(stdout_lines)
    step.stderr      = "".join(stderr_lines)
    step.duration_s  = time.time() - start_time
    step.finished_at = datetime.now().isoformat()

    # Determinar éxito
    ok = (proc.returncode == 0)

    # Verificar outputs declarados
    missing_outputs = []
    if ok and step.output_files:
        for out in step.output_files:
            if not Path(out).exists():
                missing_outputs.append(out)
        if missing_outputs:
            ok = False
            step.error_msg = (
                f"returncode=0 pero outputs no encontrados: "
                f"{', '.join(Path(f).name for f in missing_outputs)}"
            )

    if ok:
        step.status = ST_OK
    else:
        step.status = ST_FAILED
        if not step.error_msg and step.stderr:
            # Extraer última línea de error significativa
            err_lines = [l.strip() for l in step.stderr.splitlines() if l.strip()]
            if err_lines:
                step.error_msg = err_lines[-1][:120]

    return ok


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class PipelineRunner:
    """Gestiona la ejecución completa del pipeline."""

    def __init__(self, steps: list[PipelineStep], plan_path: str,
                 raw_plan: dict, fmt: str, args):
        self.steps      = steps
        self.plan_path  = plan_path
        self.raw_plan   = raw_plan
        self.fmt        = fmt
        self.args       = args
        self.work_dir   = Path(plan_path).parent
        self.checkpoint = Checkpoint(plan_path)

        log_path = None if args.no_log else (
            args.log or str(Path(plan_path).with_suffix(".runner.log"))
        )
        self.logger = RunLogger(log_path, verbose=args.verbose)
        self.log_path = log_path

        self.started_at = None
        self.run_info   = {"plan_path": plan_path}

    def _filter_steps(self) -> list[tuple[int, PipelineStep]]:
        """
        Retorna los pasos a ejecutar según --from, --until, --only.
        """
        indexed = list(enumerate(self.steps))

        if self.args.only:
            only_set = set(self.args.only)
            return [(i, s) for i, s in indexed if s.name in only_set]

        from_idx  = 0
        until_idx = len(self.steps) - 1

        if self.args.from_step:
            matches = [i for i, s in indexed if s.name == self.args.from_step]
            if matches:
                from_idx = matches[0]
            else:
                print(f"  [warn] --from '{self.args.from_step}' no encontrado. "
                      f"Pasos disponibles: {[s.name for s in self.steps]}")

        if self.args.until:
            matches = [i for i, s in indexed if s.name == self.args.until]
            if matches:
                until_idx = matches[0]
            else:
                print(f"  [warn] --until '{self.args.until}' no encontrado.")

        return [(i, s) for i, s in indexed if from_idx <= i <= until_idx]

    def _should_pause(self, step_name: str) -> bool:
        if self.args.pause_all:
            return True
        if self.args.pause_after and step_name in self.args.pause_after:
            return True
        return False

    def _ask_continue(self, step: PipelineStep) -> bool:
        """Pausa interactiva. Retorna True para continuar, False para abortar."""
        print()
        print(f"  {'─'*58}")
        print(f"  Paso completado: {_c('green', step.label)}")
        if step.output_files:
            for f in step.output_files:
                if Path(f).exists():
                    size = Path(f).stat().st_size
                    print(f"    ✓ {Path(f).name}  ({size//1024} KB)")
        print(f"  {'─'*58}")
        try:
            resp = input("  [Enter=continuar, s=saltar siguiente, q=salir] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if resp == "q":
            return False
        return True

    def print_plan(self):
        """Imprime la lista de pasos del plan."""
        print(f"\n{'═'*62}")
        obra_key  = self.raw_plan.get("obra", {})
        if isinstance(obra_key, dict):
            arc   = obra_key.get("arc", self.raw_plan.get("arc", "—"))
            bars  = obra_key.get("bars", self.raw_plan.get("bars", "—"))
            key   = obra_key.get("key",  self.raw_plan.get("key", "—"))
            tempo = obra_key.get("tempo",self.raw_plan.get("tempo","—"))
        else:
            arc   = self.raw_plan.get("arc", "—")
            bars  = self.raw_plan.get("bars", "—")
            key   = self.raw_plan.get("key", "—")
            tempo = self.raw_plan.get("tempo", "—")

        print(f"  RUNNER v{VERSION}  —  {Path(self.plan_path).name}")
        print(f"  Formato: {self.fmt}   Arc: {arc}   "
              f"{bars} bars   {key}   {tempo} BPM")
        print(f"{'═'*62}")
        print(f"  {'#':>2}  {'Paso':<28}  {'Script':<26}")
        print(f"  {'─'*2}  {'─'*28}  {'─'*26}")
        for i, s in enumerate(self.steps, 1):
            script = Path(s.cmd[1]).name if len(s.cmd) > 1 else "—"
            print(f"  {i:>2}  {s.label:<28}  {script:<26}")
        print()

    def run(self) -> bool:
        """Ejecuta el pipeline. Retorna True si todos los pasos tuvieron éxito."""
        self.started_at       = datetime.now()
        self.run_info["started_at"] = self.started_at.isoformat()

        to_run = self._filter_steps()

        if not to_run:
            print("  [warn] No hay pasos que ejecutar con los filtros indicados.")
            return True

        # Marcar pasos saltados
        run_indices = {i for i, _ in to_run}
        for i, s in enumerate(self.steps):
            if i not in run_indices and s.status == ST_PENDING:
                s.status = ST_SKIPPED

        # Reanudar desde checkpoint si se pidió
        if self.args.resume:
            ckpt = self.checkpoint.load()
            if ckpt:
                resume_idx = self.checkpoint.restore_into(self.steps)
                to_run = [(i, s) for i, s in to_run if i >= resume_idx and s.status != ST_OK]
                print(f"  [resume] Reanudando desde el paso {resume_idx + 1}")
            else:
                print("  [resume] No hay checkpoint guardado. Ejecutando desde el inicio.")

        # Si es retry-failed, cargar checkpoint y filtrar solo los fallidos
        if self.args.retry_failed:
            ckpt = self.checkpoint.load()
            if ckpt:
                self.checkpoint.restore_into(self.steps)
            to_run = [(i, s) for i, s in to_run if s.status in (ST_FAILED, ST_TIMEOUT)]
            if not to_run:
                print("  No hay pasos fallidos que reintentar.")
                return True
            # Reset status de los que se van a reintentar
            for _, s in to_run:
                s.status = ST_PENDING

        # ── Cabecera ──────────────────────────────────────────────────────────
        self.logger.write(f"\n{'═'*62}")
        self.logger.write(f"  RUNNER v{VERSION}  —  {Path(self.plan_path).name}")
        self.logger.write(f"  Inicio: {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.write(f"  Pasos a ejecutar: {len(to_run)}")
        self.logger.write(f"{'═'*62}")

        n_ok      = 0
        n_fail    = 0
        aborted   = False

        for idx, (step_i, step) in enumerate(to_run):
            total = len(to_run)
            self.logger.write(
                f"\n  [{idx+1}/{total}] {_c('bold', step.label)}"
                f"  ({step.name})"
            )

            # Verificar inputs declarados
            missing = [f for f in step.input_files if not Path(f).exists()]
            if missing:
                missing_names = ', '.join(Path(f).name for f in missing)
                self.logger.write(
                    f"  {_c('red', STATUS_ICON[ST_FAILED])} Inputs no encontrados: "
                    f"{missing_names}"
                )
                self.logger.write(
                    "     (un paso anterior fallo y no genero los ficheros necesarios)"
                )
                step.status    = ST_FAILED
                step.error_msg = f"FileNotFoundError: {missing_names}"
                n_fail += 1
                self.checkpoint.save(self.run_info, self.steps)
                if self.args.strict:
                    aborted = True
                    break
                continue

            # Verificar que el script existe
            script_path = step.cmd[1] if len(step.cmd) > 1 else ""
            if script_path and not Path(script_path).exists():
                step.status    = ST_FAILED
                step.error_msg = f"Script no encontrado: {Path(script_path).name}"
                self.logger.write(
                    f"  {_c('red', STATUS_ICON[ST_FAILED])} {step.error_msg}"
                )
                if step.optional:
                    self.logger.write("     (paso opcional, continuando)")
                    n_fail += 1
                    continue
                n_fail += 1
                self.checkpoint.save(self.run_info, self.steps)
                if self.args.strict:
                    aborted = True
                    break
                continue

            # Ejecutar
            ok = execute_step(
                step, self.args.timeout, self.args.verbose,
                self.logger, self.work_dir
            )

            dur = f"{step.duration_s:.1f}s"
            if ok:
                n_ok += 1
                self.logger.write(
                    f"  {_c('green', STATUS_ICON[ST_OK])} {step.label}"
                    f"  {_c('gray', dur)}"
                )
                for out in step.output_files:
                    if Path(out).exists():
                        size = Path(out).stat().st_size
                        self.logger.write(
                            f"    {_c('gray', '→')} {Path(out).name}"
                            f"  {_c('gray', f'({size//1024} KB)')}"
                        )
            else:
                n_fail += 1
                self.logger.write(
                    f"  {_c('red', STATUS_ICON[ST_FAILED])} {step.label}"
                    f"  {_c('gray', dur)}"
                )
                if step.error_msg:
                    self.logger.write(
                        f"    {_c('red', '→')} {step.error_msg}"
                    )
                if step.status == ST_TIMEOUT:
                    self.logger.write(
                        f"    {_c('yellow', '→')} Timeout tras {self.args.timeout}s"
                    )

            # Guardar checkpoint tras cada paso
            self.checkpoint.save(self.run_info, self.steps)

            # Pausa interactiva
            if ok and self._should_pause(step.name):
                if not self._ask_continue(step):
                    aborted = True
                    break

            # Abortar en strict mode
            if not ok and self.args.strict and not step.optional:
                self.logger.write(
                    f"\n  {_c('red', 'Abortado')} (--strict): fallo en '{step.name}'"
                )
                aborted = True
                break

        # ── Resumen final ─────────────────────────────────────────────────────
        self._print_summary(n_ok, n_fail, aborted)
        self.logger.close()

        return n_fail == 0 and not aborted

    def _print_summary(self, n_ok: int, n_fail: int, aborted: bool):
        elapsed = datetime.now() - self.started_at
        elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))

        self.logger.write(f"\n{'═'*62}")
        self.logger.write(f"  Resumen del pipeline")
        self.logger.write(f"  {'─'*58}")

        # Estado de cada paso
        for s in self.steps:
            icon  = _c(STATUS_COLOR[s.status], STATUS_ICON[s.status])
            dur   = f"  {s.duration_s:.1f}s" if s.duration_s else ""
            self.logger.write(f"  {icon}  {s.label:<40}{_c('gray', dur)}")
            if s.status == ST_FAILED and s.error_msg:
                err_short = s.error_msg[:70]
                self.logger.write(f"      {_c('red', '→')} {err_short}")

        self.logger.write(f"  {'─'*58}")

        if aborted:
            status_str = _c("yellow", "ABORTADO")
        elif n_fail == 0:
            status_str = _c("green", "COMPLETADO")
        else:
            status_str = _c("yellow", f"COMPLETADO CON {n_fail} ERROR(ES)")

        self.logger.write(f"  Estado:  {status_str}")
        self.logger.write(f"  Exitosos: {n_ok}   Fallidos: {n_fail}")
        self.logger.write(f"  Tiempo total: {elapsed_str}")

        # Archivos generados
        generated = []
        for s in self.steps:
            if s.status == ST_OK:
                for f in s.output_files:
                    if Path(f).exists():
                        generated.append(f)
        if generated:
            self.logger.write(f"\n  Archivos generados:")
            for f in generated:
                size = Path(f).stat().st_size
                self.logger.write(
                    f"    ◆ {Path(f).name}  {_c('gray', f'({size//1024} KB)')}"
                )

        if self.log_path:
            self.logger.write(f"\n  Log completo: {self.log_path}")

        # Tip de resume si hay fallos
        if n_fail > 0 and not aborted:
            self.logger.write(
                f"\n  Para reintentar los pasos fallidos:"
            )
            self.logger.write(
                f"    python runner.py {Path(self.plan_path).name} --retry-failed"
            )

        self.logger.write(f"{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  DRY RUN
# ══════════════════════════════════════════════════════════════════════════════

def dry_run(steps: list[PipelineStep], raw: dict, fmt: str, plan_path: str):
    """Imprime los comandos sin ejecutar nada."""
    print(f"\n{'═'*62}")
    print(f"  RUNNER v{VERSION}  —  Dry run: {Path(plan_path).name}")
    print(f"  Formato: {fmt}   Pasos: {len(steps)}")
    print(f"{'═'*62}")

    for i, s in enumerate(steps, 1):
        script = Path(s.cmd[1]).name if len(s.cmd) > 1 else "?"
        print(f"\n  [{i}] {_c('bold', s.label)}")
        print(f"  Script: {script}")
        # Imprimir comando formateado
        cmd_str = s.cmd_str()
        if len(cmd_str) > 90:
            tokens = s.cmd[:]
            lines  = [f"    {sys.executable} {Path(tokens[1]).name if len(tokens) > 1 else ''}"]
            chunk  = []
            for t in tokens[2:]:
                chunk.append(t)
                if t.startswith("--") and len(chunk) > 2:
                    lines.append("      " + " ".join(chunk[:-1]))
                    chunk = [t]
            if chunk:
                lines.append("      " + " ".join(chunk))
            print(" \\\n".join(lines))
        else:
            print(f"    {cmd_str}")
        if s.output_files:
            print(f"  Salida:  {', '.join(Path(f).name for f in s.output_files)}")
        if s.input_files:
            print(f"  Entrada: {', '.join(Path(f).name for f in s.input_files)}")

    print(f"\n{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STATUS
# ══════════════════════════════════════════════════════════════════════════════

def print_status(plan_path: str):
    """Muestra el estado del último run desde el checkpoint."""
    ckpt_path = str(Path(plan_path).with_suffix(".runner.json"))
    if not Path(ckpt_path).exists():
        print(f"\n  Sin checkpoint para: {Path(plan_path).name}")
        print(f"  (ejecuta el pipeline al menos una vez)")
        return

    with open(ckpt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n{'═'*62}")
    print(f"  Estado del pipeline: {Path(plan_path).name}")
    print(f"  Iniciado:    {data.get('started_at', '—')[:16].replace('T',' ')}")
    print(f"  Guardado:    {data.get('last_saved', '—')[:16].replace('T',' ')}")
    print(f"{'═'*62}")

    steps = data.get("steps", [])
    for s in steps:
        status = s.get("status", ST_PENDING)
        icon   = _c(STATUS_COLOR[status], STATUS_ICON[status])
        dur    = f"  {s.get('duration_s', 0):.1f}s" if s.get("duration_s") else ""
        print(f"  {icon}  {s.get('label', s.get('name', '?')):<40}{_c('gray', dur)}")
        if status == ST_FAILED and s.get("error_msg"):
            print(f"      {_c('red', '→')} {s['error_msg'][:70]}")

    n_ok   = sum(1 for s in steps if s.get("status") == ST_OK)
    n_fail = sum(1 for s in steps if s.get("status") == ST_FAILED)
    print(f"\n  Exitosos: {n_ok}   Fallidos: {n_fail}   Total: {len(steps)}")
    print(f"{'═'*62}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="runner.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            RUNNER v{VERSION} — Ejecutor de pipelines de composición
            ──────────────────────────────────────────────────────────
            Lee un plan .yaml generado por theorist.py o narrator.py
            y ejecuta cada paso del pipeline en orden.

            Ejemplos:
              python runner.py obra_plan.yaml
              python runner.py obra_plan.yaml --dry-run
              python runner.py obra_plan.yaml --from midi_dna_unified
              python runner.py obra_plan.yaml --pause-all
              python runner.py obra_plan.yaml --resume
              python runner.py --status obra_plan.yaml
        """),
    )

    parser.add_argument("plan", nargs="?", default=None,
                        help="Archivo .yaml o _plan.json del pipeline")

    # Filtros de ejecución
    parser.add_argument("--from", dest="from_step", default=None, metavar="STEP",
                        help="Empezar desde este paso (nombre)")
    parser.add_argument("--until", default=None, metavar="STEP",
                        help="Ejecutar hasta este paso (inclusive)")
    parser.add_argument("--only", nargs="+", default=None, metavar="STEP",
                        help="Ejecutar solo estos pasos")

    # Pausa interactiva
    parser.add_argument("--pause-after", nargs="+", default=None, metavar="STEP",
                        help="Pausar tras estos pasos")
    parser.add_argument("--pause-all", action="store_true",
                        help="Pausar tras cada paso")

    # Reanudación
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde el último checkpoint")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Reintentar solo los pasos fallidos")

    # Modos especiales
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar comandos sin ejecutar")
    parser.add_argument("--status", action="store_true",
                        help="Mostrar estado del último run")
    parser.add_argument("--list-steps", action="store_true",
                        help="Listar pasos del plan sin ejecutar")

    # Salida y logging
    parser.add_argument("--output-dir", default=None,
                        help="Directorio de salida (default: junto al plan)")
    parser.add_argument("--log", default=None,
                        help="Archivo de log (default: <plan>.runner.log)")
    parser.add_argument("--no-log", action="store_true",
                        help="No escribir archivo de log")

    # Control de ejecución
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout por paso en segundos (default: 300)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mostrar stdout/stderr en tiempo real")
    parser.add_argument("--strict", action="store_true",
                        help="Abortar al primer fallo")

    args = parser.parse_args()

    # ── Sin plan: mostrar ayuda ───────────────────────────────────────────────
    if not args.plan:
        parser.print_help()
        print("\n  Ejemplos rápidos:")
        print("    python runner.py obra_plan.yaml")
        print("    python runner.py obra_plan.yaml --dry-run")
        print("    python runner.py obra_plan.yaml --resume")
        sys.exit(0)

    plan_path = str(Path(args.plan).resolve())
    if not Path(plan_path).exists():
        print(f"ERROR: no existe el archivo: {args.plan}")
        sys.exit(1)

    # ── Solo mostrar status ───────────────────────────────────────────────────
    if args.status:
        print_status(plan_path)
        return

    # ── Cargar el plan ────────────────────────────────────────────────────────
    # Detectar formato por extensión
    suffix = Path(plan_path).suffix.lower()
    if suffix == ".json":
        try:
            steps, raw, fmt = load_narrator_json(plan_path)
        except Exception as e:
            print(f"ERROR cargando {plan_path}: {e}")
            sys.exit(1)
    else:
        try:
            steps, raw, fmt = load_plan(plan_path)
        except Exception as e:
            print(f"ERROR cargando {plan_path}: {e}")
            sys.exit(1)

    if not steps:
        print(f"  [warn] No se encontraron pasos en el plan ({fmt}).")
        print(f"         Revisa el formato del archivo: {plan_path}")
        sys.exit(1)

    # ── Solo listar pasos ─────────────────────────────────────────────────────
    if args.list_steps:
        runner = PipelineRunner(steps, plan_path, raw, fmt, args)
        runner.print_plan()
        return

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        dry_run(steps, raw, fmt, plan_path)
        return

    # ── Ejecutar pipeline ─────────────────────────────────────────────────────
    runner = PipelineRunner(steps, plan_path, raw, fmt, args)
    runner.print_plan()

    try:
        success = runner.run()
    except KeyboardInterrupt:
        print(f"\n\n  {_c('yellow', 'Interrumpido por el usuario.')}")
        print(f"  El estado se guardó en el checkpoint.")
        print(f"  Para reanudar: python runner.py {Path(plan_path).name} --resume")
        sys.exit(130)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
