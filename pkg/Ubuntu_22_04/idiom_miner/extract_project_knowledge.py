#!/usr/bin/env python3
"""
extract_project_knowledge.py
=============================

Extrae conocimiento de un proyecto de software en DOS FASES y lo deja
como texto plano/Markdown consultable offline, sin depender de un LLM
para el trabajo diario posterior.

FASE 1 - EXTRACCION ESTATICA (sin LLM, gratis, repetible cuantas veces quieras)
    - Arbol de archivos
    - Indice de simbolos (via `ctags` si esta instalado; si no, regex de respaldo)
    - Grafo de dependencias (imports/includes por lenguaje)
    - Historial git por archivo (si el proyecto es un repo git)
    - Deteccion heuristica de funciones complejas / puntos de entrada

FASE 2 - SINTESIS CON LLM (una unica pasada, con cache para poder reanudar)
    - Resumen por archivo
    - Mapa semantico del proyecto
    - Vision de arquitectura
    - Convenciones detectadas
    - Explicacion de algoritmos dificiles
    - Base de conocimiento consultable (preguntas y respuestas)
    - Snippets especificos del proyecto (formato UltiSnips)
    - Checklist de revision de codigo
    - Casos de uso tipicos

REQUISITOS
    - Python 3.8+ (solo libreria estandar, sin dependencias externas)
    - Opcional: `universal-ctags` en PATH para un indice de simbolos mejor
    - Opcional: `git` en PATH para historial por archivo
    - Para la fase 2: clave de API del proveedor elegido
        * Anthropic: variable de entorno ANTHROPIC_API_KEY
        * OpenAI:    variable de entorno OPENAI_API_KEY

USO BASICO
    # Solo extraccion estatica, sin llamadas a ningun LLM ni red:
    python3 extract_project_knowledge.py --root . --no-llm

    # Extraccion completa con Anthropic (proveedor por defecto):
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 extract_project_knowledge.py --root .

    # Extraccion completa con OpenAI:
    export OPENAI_API_KEY=sk-...
    python3 extract_project_knowledge.py --root . --provider openai

    # Modelo explicito (sobrescribe el default del proveedor):
    python3 extract_project_knowledge.py --root . --provider openai --model gpt-4o-mini

    # Reanudar tras un corte (usa la cache en .project-knowledge/.cache.json):
    python3 extract_project_knowledge.py --root .

SALIDA
    Todo se escribe en <output>/ (por defecto .project-knowledge/) como
    ficheros .txt/.md planos, pensados para grep/fzf/telescope, no para
    ser parseados por programas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from collections import defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = ".project-knowledge"

DEFAULT_IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", "dist", "build", "target", ".mypy_cache", ".pytest_cache",
    ".idea", ".vscode", "vendor", ".next", ".nuxt", "coverage",
    ".project-knowledge",
}

LANGUAGE_BY_EXT = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".cs": "csharp", ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "shell", ".lua": "lua", ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell", ".ml": "ocaml", ".sql": "sql", ".vim": "vimscript",
}

# regex de imports/dependencias por lenguaje (heuristico, no un parser real)
IMPORT_PATTERNS = {
    "python": [r"^\s*import\s+([\w\.]+)", r"^\s*from\s+([\w\.]+)\s+import"],
    "javascript": [r"require\(['\"]([^'\"]+)['\"]\)",
                   r"from\s+['\"]([^'\"]+)['\"]"],
    "typescript": [r"require\(['\"]([^'\"]+)['\"]\)",
                    r"from\s+['\"]([^'\"]+)['\"]"],
    "java": [r"^\s*import\s+([\w\.]+);"],
    "go": [r"^\s*\"([\w\./-]+)\"\s*$"],
    "rust": [r"^\s*use\s+([\w:]+)"],
    "c": [r'#include\s*[<"]([^>"]+)[>"]'],
    "cpp": [r'#include\s*[<"]([^>"]+)[>"]'],
    "ruby": [r"^\s*require(?:_relative)?\s+['\"]([^'\"]+)['\"]"],
    "php": [r"^\s*(?:require|include)(?:_once)?\s*\(?['\"]([^'\"]+)['\"]"],
    "csharp": [r"^\s*using\s+([\w\.]+);"],
}

# regex de respaldo para simbolos si no hay ctags (funcion/clase, aproximado)
FALLBACK_SYMBOL_PATTERNS = {
    "python": [(r"^\s*def\s+(\w+)\s*\(", "function"),
               (r"^\s*class\s+(\w+)\s*[:\(]", "class")],
    "javascript": [(r"function\s+(\w+)\s*\(", "function"),
                    (r"class\s+(\w+)", "class"),
                    (r"const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function")],
    "typescript": [(r"function\s+(\w+)\s*\(", "function"),
                    (r"class\s+(\w+)", "class"),
                    (r"interface\s+(\w+)", "interface")],
    "java": [(r"(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(", "method"),
              (r"class\s+(\w+)", "class")],
    "go": [(r"^func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(", "function"),
            (r"^type\s+(\w+)\s+struct", "struct")],
    "rust": [(r"^\s*fn\s+(\w+)\s*\(", "function"),
              (r"^\s*struct\s+(\w+)", "struct"),
              (r"^\s*enum\s+(\w+)", "enum")],
    "c": [(r"^\w[\w\s\*]*?(\w+)\s*\([^;]*\)\s*\{", "function")],
    "cpp": [(r"^\w[\w\s\*:<>]*?(\w+)\s*\([^;]*\)\s*\{", "function"),
             (r"class\s+(\w+)", "class")],
    "ruby": [(r"^\s*def\s+(\w+)", "method"),
              (r"^\s*class\s+(\w+)", "class")],
}

ENTRYPOINT_HINTS = [
    "main.py", "__main__.py", "app.py", "manage.py", "index.js",
    "index.ts", "main.go", "main.rs", "Main.java", "main.c", "main.cpp",
    "server.js", "server.py", "cli.py", "wsgi.py", "asgi.py",
]

MAX_FILE_BYTES_FOR_LLM = 20_000       # ~5k tokens, tope por fichero enviado
COMPLEXITY_LINE_THRESHOLD = 45        # funciones mas largas que esto -> candidatas
MAX_COMPLEX_FUNCTIONS = 25            # tope de explicaciones de algoritmos
MAX_FILES_FOR_SUMMARY = 400           # tope de seguridad de llamadas LLM
CONVENTIONS_SAMPLE_SIZE = 10          # ficheros representativos para convenciones

ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

OPENAI_MODEL_DEFAULT = "gpt-4o"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

# variable de entorno por defecto donde se busca la clave, segun proveedor
DEFAULT_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
DEFAULT_MODEL_BY_PROVIDER = {
    "anthropic": ANTHROPIC_MODEL_DEFAULT,
    "openai": OPENAI_MODEL_DEFAULT,
}


# --------------------------------------------------------------------------
# Utilidades generales
# --------------------------------------------------------------------------

def sh(cmd: list, cwd: str = ".") -> str:
    """Ejecuta un comando y devuelve stdout, o "" si falla."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=60
        )
        return result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def is_git_repo(root: str) -> bool:
    return sh(["git", "rev-parse", "--is-inside-work-tree"], cwd=root).strip() == "true"


def has_ctags() -> bool:
    return sh(["ctags", "--version"]) != ""


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_EXT.get(path.suffix.lower(), "")


def read_text(path: Path, limit: int = None) -> str:
    try:
        data = path.read_bytes()
        if limit:
            data = data[:limit]
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "0"


def is_probably_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:1024]
        return b"\x00" in chunk
    except OSError:
        return True


# --------------------------------------------------------------------------
# Descubrimiento de ficheros
# --------------------------------------------------------------------------

def list_files(root: str) -> list:
    """Lista ficheros de texto del proyecto, usando `git ls-files` si es
    posible (respeta .gitignore automaticamente) y si no, un walk manual."""
    root_path = Path(root).resolve()

    if is_git_repo(root):
        out = sh(["git", "ls-files"], cwd=root)
        files = [root_path / f for f in out.splitlines() if f.strip()]
    else:
        files = []
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORE_DIRS
                            and not d.startswith(".")]
            for fn in filenames:
                files.append(Path(dirpath) / fn)

    result = []
    for f in files:
        if not f.is_file():
            continue
        if any(part in DEFAULT_IGNORE_DIRS for part in f.parts):
            continue
        if is_probably_binary(f):
            continue
        result.append(f)
    return sorted(set(result))


# --------------------------------------------------------------------------
# FASE 1a: arbol de ficheros y estadisticas basicas
# --------------------------------------------------------------------------

def build_file_tree_text(files: list, root: Path) -> str:
    lines = ["ARBOL DE FICHEROS", "=" * 60, ""]
    for f in files:
        rel = f.relative_to(root)
        lines.append(str(rel))
    return "\n".join(lines) + "\n"


def compute_file_stats(files: list, root: Path) -> dict:
    """Devuelve {ruta_relativa: {lang, loc, bytes, hash}}"""
    stats = {}
    for f in files:
        rel = str(f.relative_to(root))
        text = read_text(f)
        stats[rel] = {
            "path": f,
            "lang": detect_language(f),
            "loc": text.count("\n") + 1 if text else 0,
            "bytes": f.stat().st_size,
            "hash": file_hash(f),
        }
    return stats


# --------------------------------------------------------------------------
# FASE 1b: indice de simbolos
# --------------------------------------------------------------------------

def run_ctags_symbols(files: list, root: Path) -> list:
    """Usa universal-ctags si esta disponible. Devuelve lista de dicts:
    {file, name, kind, line, end, scope, signature}"""
    if not has_ctags():
        return []

    file_list_path = root / ".ctags_filelist.tmp"
    try:
        file_list_path.write_text("\n".join(str(f) for f in files))
        out = sh([
            "ctags", "-L", str(file_list_path), "--output-format=json",
            "--fields=+n+e+S", "-f", "-",
        ], cwd=str(root))
    finally:
        if file_list_path.exists():
            file_list_path.unlink()

    symbols = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            tag = json.loads(line)
        except json.JSONDecodeError:
            continue
        if tag.get("_type") != "tag":
            continue
        try:
            rel = str(Path(tag.get("path", "")).resolve().relative_to(root))
        except ValueError:
            rel = tag.get("path", "")
        symbols.append({
            "file": rel,
            "name": tag.get("name", ""),
            "kind": tag.get("kind", ""),
            "line": tag.get("line", 0),
            "end": tag.get("end", 0),
            "scope": tag.get("scope", ""),
            "signature": tag.get("signature", ""),
        })
    return symbols


def run_fallback_symbols(stats: dict, root: Path) -> list:
    """Extraccion de simbolos por regex cuando no hay ctags."""
    symbols = []
    for rel, meta in stats.items():
        lang = meta["lang"]
        patterns = FALLBACK_SYMBOL_PATTERNS.get(lang)
        if not patterns:
            continue
        text = read_text(meta["path"])
        for i, line in enumerate(text.splitlines(), start=1):
            for pattern, kind in patterns:
                m = re.search(pattern, line)
                if m:
                    symbols.append({
                        "file": rel, "name": m.group(1), "kind": kind,
                        "line": i, "end": 0, "scope": "", "signature": "",
                    })
    return symbols


def build_symbol_index_text(symbols: list) -> str:
    lines = ["INDICE DE SIMBOLOS (por fichero)", "=" * 60, ""]
    by_file = defaultdict(list)
    for s in symbols:
        by_file[s["file"]].append(s)

    for rel in sorted(by_file):
        lines.append(f"\n--- {rel} ---")
        for s in sorted(by_file[rel], key=lambda x: x["line"]):
            loc = f"L{s['line']}"
            if s.get("end"):
                loc += f"-{s['end']}"
            scope = f" (en {s['scope']})" if s.get("scope") else ""
            sig = f" {s['signature']}" if s.get("signature") else ""
            lines.append(f"  [{loc}] {s['kind']}: {s['name']}{scope}{sig}")

    lines.append("\n\nINDICE DE SIMBOLOS (alfabetico, global)")
    lines.append("=" * 60)
    by_name = defaultdict(list)
    for s in symbols:
        by_name[s["name"]].append(s)
    for name in sorted(by_name, key=str.lower):
        locs = ", ".join(f"{s['file']}:{s['line']}" for s in by_name[name])
        lines.append(f"{name}: {locs}")

    return "\n".join(lines) + "\n"


def flag_complex_functions(symbols: list) -> list:
    """Funciones/metodos con mucho span de lineas -> candidatas a
    explicacion de algoritmo."""
    candidates = []
    for s in symbols:
        if s["kind"] not in ("function", "method"):
            continue
        span = (s["end"] - s["line"]) if s.get("end") else 0
        if span >= COMPLEXITY_LINE_THRESHOLD:
            candidates.append((span, s))
    candidates.sort(key=lambda x: -x[0])
    return [s for _, s in candidates[:MAX_COMPLEX_FUNCTIONS]]


# --------------------------------------------------------------------------
# FASE 1c: grafo de dependencias
# --------------------------------------------------------------------------

def extract_imports(text: str, lang: str) -> list:
    patterns = IMPORT_PATTERNS.get(lang, [])
    found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.MULTILINE):
            found.append(m.group(1))
    return sorted(set(found))


def build_dependency_graph(stats: dict) -> dict:
    graph = {}
    for rel, meta in stats.items():
        if not meta["lang"]:
            continue
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        deps = extract_imports(text, meta["lang"])
        if deps:
            graph[rel] = deps
    return graph


def build_dependency_graph_text(graph: dict) -> str:
    lines = ["GRAFO DE DEPENDENCIAS (imports/includes detectados por fichero)",
              "=" * 60, ""]
    for rel in sorted(graph):
        lines.append(f"\n{rel}")
        for dep in graph[rel]:
            lines.append(f"  -> {dep}")

    # tabla inversa: modulo -> quien lo usa
    reverse = defaultdict(list)
    for rel, deps in graph.items():
        for dep in deps:
            reverse[dep].append(rel)

    lines.append("\n\nUSO INVERSO (modulo -> ficheros que lo importan)")
    lines.append("=" * 60)
    for dep in sorted(reverse):
        users = ", ".join(reverse[dep])
        lines.append(f"{dep}: {users}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# FASE 1d: historial git por fichero
# --------------------------------------------------------------------------

def git_file_history_text(files: list, root: Path) -> str:
    if not is_git_repo(str(root)):
        return "HISTORIAL GIT\n" + "=" * 60 + "\n\n(No es un repositorio git; fase omitida)\n"

    lines = ["HISTORIAL GIT POR FICHERO", "=" * 60, ""]
    for f in files:
        rel = f.relative_to(root)
        commit_count = sh(["git", "rev-list", "--count", "HEAD", "--", str(rel)], cwd=str(root)).strip()
        last_commit = sh(["git", "log", "-1", "--format=%ad|%an|%s", "--date=short", "--", str(rel)], cwd=str(root)).strip()
        if not commit_count:
            continue
        lines.append(f"{rel}")
        lines.append(f"  commits: {commit_count}")
        if last_commit:
            date, author, subject = (last_commit.split("|", 2) + ["", "", ""])[:3]
            lines.append(f"  ultimo cambio: {date} por {author} - {subject}")
        lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# FASE 1e: puntos de entrada
# --------------------------------------------------------------------------

def detect_entrypoints(stats: dict) -> list:
    found = []
    for rel, meta in stats.items():
        base = Path(rel).name
        if base in ENTRYPOINT_HINTS:
            found.append(rel)
            continue
        if meta["lang"] == "python":
            text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
            if "__name__" in text and "__main__" in text:
                found.append(rel)
    return sorted(set(found))


# --------------------------------------------------------------------------
# Cache de resultados LLM (para poder reanudar sin repetir llamadas)
# --------------------------------------------------------------------------

class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value):
        self.data[key] = value
        self._save()

    def _save(self):
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------
# Llamada al LLM (Anthropic API, sin dependencias externas)
# --------------------------------------------------------------------------

def _call_anthropic(system: str, user: str, model: str, api_key: str,
                     max_tokens: int, retries: int) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_API_URL, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
                return "\n".join(parts).strip()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            return f"[ERROR LLM (anthropic): {e.code} - {detail[:300]}]"
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(3)
    return f"[ERROR LLM (anthropic) tras {retries} intentos: {last_err}]"


def _call_openai(system: str, user: str, model: str, api_key: str,
                  max_tokens: int, retries: int) -> str:
    body = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENAI_API_URL, data=body, method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if not choices:
                    return "[ERROR LLM (openai): respuesta sin 'choices']"
                return (choices[0].get("message", {}).get("content") or "").strip()
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            return f"[ERROR LLM (openai): {e.code} - {detail[:300]}]"
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(3)
    return f"[ERROR LLM (openai) tras {retries} intentos: {last_err}]"


_ACTIVE_PROVIDER = {"name": "anthropic"}  # fijado una vez en main()


def set_active_provider(provider: str):
    _ACTIVE_PROVIDER["name"] = provider


def call_llm(system: str, user: str, model: str, api_key: str,
             max_tokens: int = 2000, retries: int = 3, provider: str = None) -> str:
    """Despachador unico usado por todas las fases de sintesis.
    Si no se pasa `provider` explicitamente, usa el fijado en main()
    via set_active_provider() (--provider anthropic|openai)."""
    provider = provider or _ACTIVE_PROVIDER["name"]
    if provider == "openai":
        return _call_openai(system, user, model, api_key, max_tokens, retries)
    if provider == "anthropic":
        return _call_anthropic(system, user, model, api_key, max_tokens, retries)
    raise ValueError(f"Proveedor desconocido: {provider}")


def chunk_join(items: list, sep: str, max_chars: int) -> str:
    out, total = [], 0
    for item in items:
        if total + len(item) > max_chars:
            break
        out.append(item)
        total += len(item)
    return sep.join(out)


# --------------------------------------------------------------------------
# FASE 2: sintesis con LLM
# --------------------------------------------------------------------------

def phase_file_summaries(stats: dict, cache: Cache, model: str, api_key: str) -> dict:
    summaries = {}
    items = list(stats.items())[:MAX_FILES_FOR_SUMMARY]
    total = len(items)
    for i, (rel, meta) in enumerate(items, 1):
        cache_key = f"summary::{_ACTIVE_PROVIDER['name']}::{rel}::{meta['hash']}"
        cached = cache.get(cache_key)
        if cached is not None:
            summaries[rel] = cached
            continue

        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        if not text.strip():
            continue

        print(f"  [{i}/{total}] resumiendo {rel}")
        system = (
            "Eres un ingeniero de software analizando un fichero de codigo "
            "para dejar documentacion util a otros desarrolladores. "
            "Responde en texto plano, sin markdown, de forma concisa."
        )
        user = (
            f"Fichero: {rel}\nLenguaje: {meta['lang'] or 'desconocido'}\n\n"
            "Contenido:\n```\n" + text + "\n```\n\n"
            "Escribe:\n"
            "1) Resumen (2-4 frases) de que hace este fichero.\n"
            "2) Responsabilidades clave (lista).\n"
            "3) Dependencias/colaboradores notables (que usa o de que depende).\n"
            "4) Notas de complejidad o puntos delicados, si los hay."
        )
        result = call_llm(system, user, model, api_key, max_tokens=500)
        summaries[rel] = result
        cache.set(cache_key, result)

    return summaries


def phase_semantic_map(summaries: dict, model: str, api_key: str) -> str:
    joined = "\n\n".join(f"### {rel}\n{s}" for rel, s in summaries.items())
    joined = chunk_join(joined.split("\n\n"), "\n\n", 60_000)
    system = (
        "Eres un arquitecto de software. A partir de resumenes por fichero de "
        "un proyecto, produces un mapa semantico jerarquico: modulos/paquetes, "
        "su responsabilidad y como se relacionan entre si. Texto plano en "
        "castellano, con indentacion para mostrar jerarquia."
    )
    user = "Resumenes por fichero:\n\n" + joined + "\n\nGenera el mapa semantico del proyecto."
    return call_llm(system, user, model, api_key, max_tokens=2000)


def phase_architecture(summaries: dict, dep_graph: dict, entrypoints: list,
                         model: str, api_key: str) -> str:
    joined = chunk_join(
        [f"### {rel}\n{s}" for rel, s in summaries.items()], "\n\n", 50_000
    )
    deps_text = "\n".join(f"{k} -> {', '.join(v)}" for k, v in list(dep_graph.items())[:200])
    system = (
        "Eres un arquitecto de software. Produce una vision de arquitectura "
        "en Markdown claro para un README tecnico, en castellano."
    )
    user = (
        f"Puntos de entrada detectados: {entrypoints}\n\n"
        f"Dependencias (muestra):\n{deps_text}\n\n"
        f"Resumenes por fichero:\n{joined}\n\n"
        "Escribe una vision de arquitectura que cubra: capas/componentes "
        "principales, flujo de datos tipico, patrones de diseno detectados, "
        "servicios/integraciones externas, y puntos de entrada."
    )
    return call_llm(system, user, model, api_key, max_tokens=2500)


def phase_conventions(stats: dict, model: str, api_key: str) -> str:
    # elige una muestra representativa: los ficheros mas grandes en distintos lenguajes
    by_lang = defaultdict(list)
    for rel, meta in stats.items():
        if meta["lang"]:
            by_lang[meta["lang"]].append((rel, meta))

    sample = []
    for lang, items in by_lang.items():
        items.sort(key=lambda x: -x[1]["loc"])
        sample.extend(items[:max(1, CONVENTIONS_SAMPLE_SIZE // max(1, len(by_lang)))])
    sample = sample[:CONVENTIONS_SAMPLE_SIZE]

    blocks = []
    for rel, meta in sample:
        text = read_text(meta["path"], limit=8000)
        blocks.append(f"### {rel} ({meta['lang']})\n```\n{text}\n```")

    system = (
        "Eres un revisor de codigo senior. A partir de una muestra de ficheros "
        "representativos, infieres las convenciones reales del proyecto (no "
        "las ideales). Responde en Markdown, en castellano."
    )
    user = (
        "Muestra de ficheros:\n\n" + "\n\n".join(blocks) + "\n\n"
        "Documenta las convenciones observadas: estilo de nombres, "
        "organizacion de carpetas/modulos, manejo de errores, logging, "
        "patrones de testing, formato/lint aparente, comentarios/docstrings."
    )
    return call_llm(system, user, model, api_key, max_tokens=2000)


def phase_algorithm_explanations(complex_functions: list, root: Path,
                                    model: str, api_key: str, cache: Cache) -> str:
    if not complex_functions:
        return "No se detectaron funciones suficientemente largas/complejas segun el umbral configurado.\n"

    out = []
    for s in complex_functions:
        path = root / s["file"]
        text = read_text(path)
        lines = text.splitlines()
        start = max(0, s["line"] - 1)
        end = s["end"] if s.get("end") else min(len(lines), start + 80)
        snippet = "\n".join(lines[start:end])

        cache_key = f"algo::{_ACTIVE_PROVIDER['name']}::{s['file']}::{s['name']}::{s['line']}::{file_hash(path)}"
        cached = cache.get(cache_key)
        if cached is not None:
            out.append(f"## {s['file']} :: {s['name']} (L{s['line']}-{end})\n\n{cached}\n")
            continue

        system = (
            "Eres un ingeniero explicando un algoritmo complejo a otro "
            "desarrollador que no lo ha visto antes. Se claro, en castellano, "
            "sin rodeos."
        )
        user = (
            f"Funcion `{s['name']}` en {s['file']}:\n```\n{snippet}\n```\n\n"
            "Explica: que hace paso a paso, complejidad aproximada (Big-O si "
            "aplica), casos limite que maneja, y posibles puntos fragiles."
        )
        result = call_llm(system, user, model, api_key, max_tokens=700)
        cache.set(cache_key, result)
        out.append(f"## {s['file']} :: {s['name']} (L{s['line']}-{end})\n\n{result}\n")

    return "\n".join(out)


def phase_knowledge_base(architecture: str, conventions: str, semantic_map: str,
                            model: str, api_key: str) -> str:
    system = (
        "Generas una base de conocimiento consultable, en formato pregunta y "
        "respuesta, para que un desarrollador nuevo en el proyecto encuentre "
        "respuestas rapidas por busqueda de texto. Castellano, Markdown."
    )
    user = (
        f"Arquitectura:\n{architecture}\n\nConvenciones:\n{conventions}\n\n"
        f"Mapa semantico:\n{semantic_map}\n\n"
        "Genera entre 15 y 30 preguntas frecuentes de un desarrollador nuevo "
        "(del tipo '¿Donde se maneja X?', '¿Como añado un Y?', '¿Que pasa "
        "cuando ocurre Z?') con respuestas concisas basadas SOLO en la "
        "informacion dada. Formato: '### Pregunta' seguido de la respuesta."
    )
    return call_llm(system, user, model, api_key, max_tokens=3000)


def phase_snippets(conventions: str, stats: dict, model: str, api_key: str) -> str:
    langs = sorted({m["lang"] for m in stats.values() if m["lang"]},
                    key=lambda l: -sum(1 for m in stats.values() if m["lang"] == l))
    main_langs = langs[:3]

    system = (
        "Generas snippets de codigo en formato UltiSnips a partir de las "
        "convenciones reales de un proyecto (no snippets genericos de "
        "libro de texto). Cada snippet debe reflejar el estilo observado."
    )
    user = (
        f"Convenciones del proyecto:\n{conventions}\n\n"
        f"Lenguajes principales: {main_langs}\n\n"
        "Genera snippets utiles y especificos del proyecto (plantilla de "
        "funcion, manejo de errores tipico, esqueleto de test, etc.) en "
        "formato UltiSnips:\n"
        "snippet trigger \"descripcion\"\n<cuerpo>\nendsnippet\n\n"
        "Agrupa por lenguaje con un comentario '# --- lenguaje ---' antes de cada grupo."
    )
    return call_llm(system, user, model, api_key, max_tokens=2000)


def phase_review_checklist(architecture: str, conventions: str, model: str, api_key: str) -> str:
    system = (
        "Generas un checklist de revision de codigo (pull request) especifico "
        "para este proyecto, no generico. Castellano, Markdown con casillas."
    )
    user = (
        f"Arquitectura:\n{architecture}\n\nConvenciones:\n{conventions}\n\n"
        "Genera un checklist de revision de PR en formato '- [ ] item', "
        "agrupado en secciones (correctitud, estilo/convenciones, "
        "seguridad, rendimiento, tests, documentacion) con items concretos "
        "derivados de las convenciones y arquitectura reales de este proyecto."
    )
    return call_llm(system, user, model, api_key, max_tokens=1500)


def phase_typical_cases(architecture: str, entrypoints: list, symbols: list,
                          model: str, api_key: str) -> str:
    key_symbols = ", ".join(sorted({s["name"] for s in symbols
                                     if s["kind"] in ("function", "method")})[:150])
    system = (
        "Describes flujos de ejecucion tipicos de un proyecto para ayudar a "
        "un desarrollador a orientarse rapido. Castellano, texto claro."
    )
    user = (
        f"Puntos de entrada: {entrypoints}\n\n"
        f"Arquitectura:\n{architecture}\n\n"
        f"Simbolos clave disponibles: {key_symbols}\n\n"
        "Describe entre 5 y 10 'casos tipicos' de uso/ejecucion del sistema "
        "(por ejemplo: 'caso tipico: llega una peticion X -> pasa por Y -> "
        "termina en Z'), citando ficheros/funciones reales cuando sea posible."
    )
    return call_llm(system, user, model, api_key, max_tokens=1800)


# --------------------------------------------------------------------------
# Orquestacion
# --------------------------------------------------------------------------

def write(output_dir: Path, name: str, content: str):
    path = output_dir / name
    path.write_text(content, encoding="utf-8")
    print(f"  escrito: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".", help="raiz del proyecto")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="carpeta de salida")
    parser.add_argument("--no-llm", action="store_true",
                          help="solo extraccion estatica, sin llamadas a ningun LLM ni red")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                          help="proveedor de LLM a usar en la fase 2 (default: anthropic)")
    parser.add_argument("--model", default=None,
                          help="modelo a usar. Si se omite, se usa el default del proveedor "
                               "(claude-sonnet-5 para anthropic, gpt-4o para openai)")
    parser.add_argument("--api-key-env", default=None,
                          help="variable de entorno con la clave de API. Si se omite, se usa "
                               "ANTHROPIC_API_KEY o OPENAI_API_KEY segun --provider")
    parser.add_argument("--yes", action="store_true", help="no pedir confirmacion antes de llamar al LLM")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL_BY_PROVIDER[args.provider]
    api_key_env = args.api_key_env or DEFAULT_API_KEY_ENV[args.provider]
    set_active_provider(args.provider)

    root = Path(args.root).resolve()
    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analizando proyecto en: {root}")
    files = list_files(str(root))
    print(f"Ficheros de texto detectados: {len(files)}")

    # ---- FASE 1: estatica ----
    print("\n[Fase 1] Extraccion estatica...")
    stats = compute_file_stats(files, root)

    write(output_dir, "00_arbol_ficheros.txt", build_file_tree_text(files, root))

    if has_ctags():
        print("  usando ctags para el indice de simbolos")
        symbols = run_ctags_symbols(files, root)
    else:
        print("  ctags no encontrado, usando extraccion por regex (menos precisa)")
        symbols = run_fallback_symbols(stats, root)
    write(output_dir, "01_indice_simbolos.txt", build_symbol_index_text(symbols))

    dep_graph = build_dependency_graph(stats)
    write(output_dir, "02_grafo_dependencias.txt", build_dependency_graph_text(dep_graph))

    write(output_dir, "03_historial_git.txt", git_file_history_text(files, root))

    entrypoints = detect_entrypoints(stats)
    complex_functions = flag_complex_functions(symbols)

    resumen_estatico = (
        "RESUMEN DE LA EXTRACCION ESTATICA\n" + "=" * 60 + "\n\n"
        f"Ficheros analizados: {len(files)}\n"
        f"Simbolos indexados: {len(symbols)}\n"
        f"Ficheros con dependencias detectadas: {len(dep_graph)}\n"
        f"Puntos de entrada detectados: {entrypoints}\n"
        f"Funciones complejas candidatas a explicacion: {len(complex_functions)}\n"
    )
    write(output_dir, "04_resumen_extraccion.txt", resumen_estatico)

    if args.no_llm:
        print("\n--no-llm activo: fase de sintesis omitida. Listo.")
        return

    api_key = os.environ.get(api_key_env)
    if not api_key:
        print(f"\nNo se encontro la variable de entorno {api_key_env}. "
              "Ejecuta con --no-llm o exporta la clave de API.")
        sys.exit(1)

    n_calls_estimate = len(stats) + 6 + min(len(complex_functions), MAX_COMPLEX_FUNCTIONS)
    print(f"\n[Fase 2] Sintesis con LLM. Llamadas estimadas: ~{n_calls_estimate} "
          f"(proveedor: {args.provider}, modelo: {model})")
    if not args.yes:
        resp = input("¿Continuar con las llamadas al LLM? [s/N]: ").strip().lower()
        if resp != "s":
            print("Cancelado. La extraccion estatica ya quedo guardada.")
            return

    cache = Cache(output_dir / ".cache.json")

    print("\n  Generando resumenes por fichero...")
    summaries = phase_file_summaries(stats, cache, model, api_key)
    write(output_dir, "05_resumenes_por_fichero.txt",
          "\n\n".join(f"### {rel}\n{s}" for rel, s in summaries.items()))

    print("  Generando mapa semantico...")
    semantic_map = phase_semantic_map(summaries, model, api_key)
    write(output_dir, "06_mapa_semantico.md", semantic_map)

    print("  Generando vision de arquitectura...")
    architecture = phase_architecture(summaries, dep_graph, entrypoints, model, api_key)
    write(output_dir, "07_arquitectura.md", architecture)

    print("  Detectando convenciones...")
    conventions = phase_conventions(stats, model, api_key)
    write(output_dir, "08_convenciones.md", conventions)

    print("  Explicando algoritmos complejos...")
    algo_explanations = phase_algorithm_explanations(complex_functions, root, model, api_key, cache)
    write(output_dir, "09_explicacion_algoritmos.md", algo_explanations)

    print("  Generando base de conocimiento...")
    kb = phase_knowledge_base(architecture, conventions, semantic_map, model, api_key)
    write(output_dir, "10_base_conocimiento.md", kb)

    print("  Generando snippets del proyecto...")
    snippets = phase_snippets(conventions, stats, model, api_key)
    write(output_dir, "11_snippets.snippets", snippets)

    print("  Generando checklist de revision...")
    checklist = phase_review_checklist(architecture, conventions, model, api_key)
    write(output_dir, "12_checklist_revision.md", checklist)

    print("  Generando casos tipicos...")
    typical_cases = phase_typical_cases(architecture, entrypoints, symbols, model, api_key)
    write(output_dir, "13_casos_tipicos.md", typical_cases)

    print(f"\nListo. Todo el conocimiento esta en: {output_dir}")


if __name__ == "__main__":
    main()
