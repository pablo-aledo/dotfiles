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
    - [v2] Indice de subcomandos CLI (argparse/click/clap) por fichero
    - [v2] Grafo de llamadas aproximado (que funcion llama a que funcion)
    - [v2] TODO/FIXME/HACK/XXX con contexto y autor (deuda tecnica)
    - [v2] Mapa de configuracion: env vars, flags CLI y ficheros de config
    - [v2] Cobertura de tests por convencion de nombres (que NO tiene test)
    - [v2] "God files": ranking por tamano/simbolos/complejidad aproximada
    - [v2] Superficie publica vs privada por fichero
    - [v2] Dependencias declaradas (manifest) vs realmente importadas
    - [v2] Glosario de dominio (terminos no genericos mas frecuentes)
    - [v2] Linea de tiempo de commits agregada por tipo/mes (sin LLM)
    - [v2] Logica de negocio vs. infraestructura/cross-cutting (categorias
      editables en INFRA_CATEGORIES: logging, telemetria, red, persistencia,
      auth, serializacion...), con vista por fichero y vista invertida por
      categoria
    - [v2] Glosario de acronimos (expansion detectada en el propio codigo o
      en el diccionario COMMON_ACRONYMS editable) y terminos de dominio con
      contexto de primera aparicion
    - [v2] Integracion nativa con vim: fichero `tags` (formato ctags, rutas
      absolutas) junto al resto de la salida generada (<output>/vim/tags,
      registrado automaticamente en 'tags' al fuentear project_nav.vim),
      quickfix lists (.qf) navegables con :cnext/:cprev para TODOs/god
      files/tests faltantes/funciones complejas/logica-vs-infra,
      symbols.tsv para fzf, y un project_nav.vim que ata comandos a todo
      ello (carpeta <output>/vim/)
    - [v2] Busqueda semantica de funciones, opcional (--semantic-index):
      Opcion A, descripcion en lenguaje natural por funcion (LLM, cacheada)
      + fzf (:ProjSemantic), 100% offline en el momento de buscar; Opcion
      B, embeddings reales por funcion (solo --provider openai) + busqueda
      por similitud coseno (:ProjSemanticVec <consulta>), shell-out a
      `--semantic-query` desde vim (una unica llamada de red por consulta,
      para embeber el texto buscado)

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
    - [v2] Definicion opcional de acronimos/terminos que la Fase 1 dejo
      pendientes por falta de expansion en el propio codigo

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
import math
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

# --------------------------------------------------------------------------
# [v2] Patrones para los nuevos extractores estaticos
# --------------------------------------------------------------------------

# Definicion de subcomandos CLI (argparse / click / clap) por lenguaje
CLI_SUBCOMMAND_PATTERNS = {
    "python": [
        (r"add_parser\(\s*['\"](\w[\w-]*)['\"]", "argparse"),
        (r"@click\.command\(\s*(?:name=)?['\"]?(\w[\w-]*)?['\"]?\s*\)", "click"),
        (r"^\s*def\s+(cmd_\w+)\s*\(", "cmd_*"),
    ],
    "rust": [
        (r"#\[command\(\s*name\s*=\s*\"(\w[\w-]*)\"", "clap"),
        (r"Subcommand::(\w+)", "clap-enum"),
        (r"^\s*(\w+)\s*\{[^}]*\}\s*,?\s*//\s*subcommand", "clap-variant"),
    ],
}
CLI_HELP_PATTERN = re.compile(r"add_argument\(\s*['\"](-{1,2}[\w-]+)['\"](?:[^)]*help\s*=\s*['\"]([^'\"]*)['\"])?", re.DOTALL)
CLI_ARGPARSE_DESC = re.compile(r"add_parser\(\s*['\"](\w[\w-]*)['\"][^)]*help\s*=\s*['\"]([^'\"]*)['\"]", re.DOTALL)

# Marcadores de deuda tecnica
TODO_PATTERN = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b[:\s]?(.*)")

# Variables de entorno / flags de configuracion por lenguaje
ENV_VAR_PATTERNS = {
    "python": [r"os\.environ(?:\.get)?\(?\[?['\"](\w+)['\"]", r"os\.getenv\(\s*['\"](\w+)['\"]"],
    "rust": [r"env::var\(\s*\"(\w+)\"", r"std::env::var\(\s*\"(\w+)\""],
    "javascript": [r"process\.env\.(\w+)", r"process\.env\[['\"](\w+)['\"]\]"],
    "typescript": [r"process\.env\.(\w+)", r"process\.env\[['\"](\w+)['\"]\]"],
    "go": [r"os\.Getenv\(\s*\"(\w+)\"\)"],
    "ruby": [r"ENV\[['\"](\w+)['\"]\]"],
}
CLI_FLAG_PATTERNS = {
    "python": [r"add_argument\(\s*['\"](-{1,2}[\w-]+)['\"]"],
    "rust": [r"#\[arg\([^)]*\)\]\s*\n?\s*(?:pub\s+)?(\w+)\s*:", r"long\s*=\s*\"([\w-]+)\""],
}
CONFIG_FILE_NAME_HINTS = [
    ".env", ".env.example", ".env.sample", "config.toml", "config.yaml",
    "config.yml", "config.json", "settings.py", "settings.toml",
    "pyproject.toml", "Cargo.toml", "package.json", "docker-compose.yml",
    "docker-compose.yaml", "Dockerfile", ".flaskenv",
]

# Ficheros de test por convencion de nombre, por lenguaje
TEST_NAME_PATTERNS = [
    (re.compile(r"^test_(.+)\.py$"), "{0}.py"),
    (re.compile(r"^(.+)_test\.py$"), "{0}.py"),
    (re.compile(r"^(.+)\.test\.[jt]sx?$"), "{0}"),
    (re.compile(r"^(.+)\.spec\.[jt]sx?$"), "{0}"),
    (re.compile(r"^(.+)_test\.go$"), "{0}.go"),
]
TEST_DIR_HINTS = {"tests", "test", "__tests__", "spec"}

# Palabras clave de control de flujo, para complejidad ciclomatica aproximada
CONTROL_FLOW_KEYWORDS = {
    "python": [r"\bif\b", r"\belif\b", r"\bfor\b", r"\bwhile\b", r"\bexcept\b", r"\bwith\b", r"\band\b", r"\bor\b"],
    "javascript": [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bcatch\b", r"\bcase\b", r"\b\&\&\b", r"\b\|\|\b"],
    "typescript": [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bcatch\b", r"\bcase\b", r"\b\&\&\b", r"\b\|\|\b"],
    "rust": [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bmatch\b", r"\bloop\b", r"\b\&\&\b", r"\b\|\|\b"],
    "go": [r"\bif\b", r"\bfor\b", r"\bswitch\b", r"\bcase\b", r"\b\&\&\b", r"\b\|\|\b"],
    "c": [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bswitch\b", r"\bcase\b", r"\b\&\&\b", r"\b\|\|\b"],
    "cpp": [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bswitch\b", r"\bcase\b", r"\b\&\&\b", r"\b\|\|\b"],
    "java": [r"\bif\b", r"\bfor\b", r"\bwhile\b", r"\bswitch\b", r"\bcase\b", r"\bcatch\b", r"\b\&\&\b", r"\b\|\|\b"],
    "ruby": [r"\bif\b", r"\bunless\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b", r"\bwhen\b"],
}

# Manifiestos de dependencias declaradas, por lenguaje/ecosistema
MANIFEST_FILES = [
    "requirements.txt", "pyproject.toml", "Pipfile", "setup.py",
    "Cargo.toml", "package.json", "go.mod", "Gemfile",
]

# Stopwords genericas de programacion a excluir del glosario de dominio
GLOSSARY_STOPWORDS = {
    "get", "set", "run", "main", "init", "new", "build", "make", "create",
    "delete", "remove", "update", "load", "save", "read", "write", "parse",
    "check", "validate", "process", "handle", "helper", "util", "utils",
    "test", "tests", "data", "value", "item", "items", "list", "dict",
    "config", "default", "base", "impl", "type", "types", "self", "args",
    "kwargs", "func", "function", "method", "class", "obj", "object",
    "index", "idx", "tmp", "temp", "result", "results", "output", "input",
    "file", "files", "path", "name", "str", "int", "bool", "true", "false",
    "none", "null", "error", "errors", "exception", "log", "logger",
    "print", "return", "start", "end", "size", "count", "num", "number",
    "cmd", "cli", "app", "core", "common", "shared", "manager", "service",
}

# --------------------------------------------------------------------------
# [v2] Logica de negocio vs. infraestructura/cross-cutting concerns
#
# >>> EDITA ESTO LIBREMENTE <<<
# Este diccionario es deliberadamente editable a mano: cada categoria tiene
# patrones regex por lenguaje (mas una clave especial "any" que se aplica
# sin importar el lenguaje del fichero). Añade, quita o ajusta categorias
# segun el stack real de tu proyecto — cuanto mas especifico el patron,
# mejor la senal. No es un analisis semantico: es deteccion de "vocabulario
# de infraestructura" por coincidencia de texto.
# --------------------------------------------------------------------------
INFRA_CATEGORIES = {
    "logging": {
        "patterns": {
            "python": [r"\blogging\.\w+\(", r"\blogger\.\w+\("],
            "rust": [r"\btracing::\w+!", r"\blog::\w+!"],
            "javascript": [r"console\.(?:log|warn|error|info|debug)\("],
            "typescript": [r"console\.(?:log|warn|error|info|debug)\("],
            "go": [r"\blog\.\w+\("],
            "java": [r"\bLoggerFactory\b", r"\.log\("],
            "any": [r"\bwinston\b", r"\bloguru\b"],
        },
    },
    "telemetria": {
        "patterns": {
            "any": [r"opentelemetry", r"\bprometheus\b", r"\bstatsd\b",
                     r"\bdatadog\b", r"\bmetrics?\.\w+\(", r"\bsentry\b"],
        },
    },
    "red_comunicaciones": {
        "patterns": {
            "python": [r"\brequests\.\w+\(", r"\burllib\.request\b",
                        r"\bhttpx\.\w+\(", r"\bsocket\.\w+\(", r"\baiohttp\b"],
            "rust": [r"\breqwest::", r"\btokio::net::", r"\bhyper::", r"\btonic::"],
            "javascript": [r"\bfetch\(", r"\baxios\.\w+\(", r"\bXMLHttpRequest\b"],
            "typescript": [r"\bfetch\(", r"\baxios\.\w+\(", r"\bXMLHttpRequest\b"],
            "go": [r"\bnet/http\b", r"\bhttp\.\w+\("],
            "any": [r"\bgrpc\b", r"\bwebsocket\b"],
        },
    },
    "persistencia": {
        "patterns": {
            "python": [r"\bsqlalchemy\b", r"\bsession\.query\(", r"\bcursor\.execute\(",
                        r"\bredis\.\w+\(", r"\bboto3\b", r"\bpymongo\b"],
            "rust": [r"\bsqlx::", r"\bdiesel::", r"\brusqlite::"],
            "javascript": [r"\bmongoose\.\w+\(", r"\bprisma\.\w+\.", r"\bknex\("],
            "typescript": [r"\bmongoose\.\w+\(", r"\bprisma\.\w+\.", r"\bknex\("],
            "go": [r"\bdatabase/sql\b", r"\bgorm\.\w+\("],
            "any": [r"\bSELECT\s+.+\s+FROM\b", r"\bINSERT\s+INTO\b"],
        },
    },
    "auth": {
        "patterns": {
            "any": [r"\bjwt\b", r"\boauth\b", r"\bbcrypt\b", r"\bpasslib\b",
                     r"['\"]Authorization['\"]", r"\bhash_password\b", r"\bverify_password\b"],
        },
    },
    "serializacion": {
        "patterns": {
            "python": [r"\bjson\.(?:dumps|loads)\(", r"\bpickle\.\w+\(", r"\byaml\.(?:safe_)?load\("],
            "rust": [r"\bserde\b", r"\bbincode::"],
            "javascript": [r"JSON\.(?:stringify|parse)\("],
            "typescript": [r"JSON\.(?:stringify|parse)\("],
            "go": [r"\bencoding/json\b"],
            "any": [r"\bprotobuf\b", r"\bmsgpack\b"],
        },
    },
}

# --------------------------------------------------------------------------
# [v2] Glosario: acronimos y terminos de dominio
#
# >>> EDITA ESTO LIBREMENTE <<<
# Diccionario de acronimos genericos de ML/audio/sistemas usado como
# fallback cuando el propio codigo no trae la expansion. Anade las siglas
# de tu propio stack para que dejen de aparecer como "pendientes".
# --------------------------------------------------------------------------
COMMON_ACRONYMS = {
    "API": "Application Programming Interface",
    "CLI": "Command Line Interface",
    "GAN": "Generative Adversarial Network",
    "VAE": "Variational Autoencoder",
    "RNN": "Recurrent Neural Network",
    "LSTM": "Long Short-Term Memory",
    "CNN": "Convolutional Neural Network",
    "FFT": "Fast Fourier Transform",
    "STFT": "Short-Time Fourier Transform",
    "DDPM": "Denoising Diffusion Probabilistic Model",
    "DDIM": "Denoising Diffusion Implicit Model",
    "JSON": "JavaScript Object Notation",
    "YAML": "YAML Ain't Markup Language",
    "TOML": "Tom's Obvious Minimal Language",
    "JWT": "JSON Web Token",
    "ORM": "Object-Relational Mapping",
    "SQL": "Structured Query Language",
    "HTTP": "HyperText Transfer Protocol",
    "HTTPS": "HTTP Secure",
    "URL": "Uniform Resource Locator",
    "URI": "Uniform Resource Identifier",
    "REST": "REpresentational State Transfer",
    "GRPC": "Google Remote Procedure Call",
    "CPU": "Central Processing Unit",
    "GPU": "Graphics Processing Unit",
    "OS": "Operating System",
    "ML": "Machine Learning",
    "AI": "Artificial Intelligence",
    "LLM": "Large Language Model",
    "NLP": "Natural Language Processing",
    "MIDI": "Musical Instrument Digital Interface",
    "UUID": "Universally Unique Identifier",
    "CSV": "Comma-Separated Values",
    "XML": "eXtensible Markup Language",
    "SDK": "Software Development Kit",
    "CI": "Continuous Integration",
    "CD": "Continuous Deployment/Delivery",
}

# Marcadores/palabras que no queremos tratar como acronimos de dominio.
# Incluye marcadores de deuda tecnica y palabras cortas en castellano que
# a veces se escriben en mayusculas por enfasis (falsos positivos reales
# detectados en pruebas, p.ej. "esto NO es...").
ACRONYM_EXCLUDE = {
    "TODO", "FIXME", "HACK", "XXX",
    "NO", "SI", "ES", "LA", "EL", "UN", "DE", "EN", "MAS", "SOLO", "ASI", "AQUI",
}

ACRONYM_TOKEN_RE = re.compile(r"\b[A-Z]{2,6}\b")
# "QKD (Quantum Key Distribution)"
EXPANSION_ACRONYM_FIRST_RE = re.compile(r"\b([A-Z]{2,6})\s*\(([A-Z][A-Za-z0-9\-/ ]{3,60})\)")
# "Quantum Key Distribution (QKD)"
EXPANSION_WORDS_FIRST_RE = re.compile(
    r"\b((?:[A-Z][a-zA-Z0-9]*\s+){1,6}[A-Z][a-zA-Z0-9]*)\s*\(([A-Z]{2,6})\)"
)

MAX_DOMAIN_TERMS_FOR_CONTEXT = 60

ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-5"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

OPENAI_MODEL_DEFAULT = "gpt-4o"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_BATCH_SIZE = 96   # limite prudente por llamada a /v1/embeddings

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
# [v2] FASE 1f: indice de subcomandos CLI (argparse/click/clap)
# --------------------------------------------------------------------------

def index_cli_commands(stats: dict) -> dict:
    """Devuelve {rel: {"subcommands": [(name, origen)], "help": {name: texto},
    "flags": [flag, ...]}} usando regex sobre patrones habituales de
    argparse/click (Python) y clap (Rust)."""
    out = {}
    for rel, meta in stats.items():
        lang = meta["lang"]
        if lang not in CLI_SUBCOMMAND_PATTERNS and lang not in CLI_FLAG_PATTERNS:
            continue
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        if not text:
            continue

        subcommands = []
        for pattern, source in CLI_SUBCOMMAND_PATTERNS.get(lang, []):
            for m in re.finditer(pattern, text):
                name = m.group(1)
                if name:
                    subcommands.append((name, source))

        help_map = {}
        for m in CLI_ARGPARSE_DESC.finditer(text):
            help_map[m.group(1)] = m.group(2)

        flags = []
        for pattern in CLI_FLAG_PATTERNS.get(lang, []):
            for m in re.finditer(pattern, text):
                flags.append(m.group(1))

        if subcommands or flags:
            out[rel] = {
                "subcommands": sorted(set(subcommands)),
                "help": help_map,
                "flags": sorted(set(flags)),
            }
    return out


def build_cli_index_text(cli_index: dict) -> str:
    lines = ["INDICE DE SUBCOMANDOS CLI (argparse/click/clap detectados por regex)",
              "=" * 60, ""]
    if not cli_index:
        lines.append("(No se detectaron subcomandos CLI reconocibles)")
        return "\n".join(lines) + "\n"
    for rel in sorted(cli_index):
        info = cli_index[rel]
        lines.append(f"\n--- {rel} ---")
        for name, source in info["subcommands"]:
            help_txt = info["help"].get(name, "")
            help_part = f" - {help_txt}" if help_txt else ""
            lines.append(f"  comando: {name} [{source}]{help_part}")
        if info["flags"]:
            lines.append(f"  flags detectados: {', '.join(info['flags'])}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1g: grafo de llamadas aproximado (funcion -> funcion)
# --------------------------------------------------------------------------

def build_call_graph(symbols: list, stats: dict) -> dict:
    """Aproximacion por regex: para cada funcion/metodo conocido, busca
    llamadas a otros simbolos conocidos dentro de su rango de lineas.
    No es un analisis semantico real (no resuelve shadowing, overloads,
    etc.) pero da una primera foto util del flujo de llamadas."""
    known_names = {s["name"] for s in symbols if s["kind"] in ("function", "method")}
    by_file = defaultdict(list)
    for s in symbols:
        if s["kind"] in ("function", "method"):
            by_file[s["file"]].append(s)

    graph = defaultdict(set)
    for rel, funcs in by_file.items():
        meta = stats.get(rel)
        if not meta:
            continue
        text = read_text(meta["path"])
        lines = text.splitlines()
        funcs_sorted = sorted(funcs, key=lambda x: x["line"])
        for i, f in enumerate(funcs_sorted):
            start = max(0, f["line"] - 1)
            if f.get("end"):
                end = f["end"]
            elif i + 1 < len(funcs_sorted):
                end = funcs_sorted[i + 1]["line"] - 1
            else:
                end = len(lines)
            body = "\n".join(lines[start:end])
            caller_key = f"{rel}::{f['name']}"
            for m in re.finditer(r"\b(\w+)\s*\(", body):
                called = m.group(1)
                if called in known_names and called != f["name"]:
                    graph[caller_key].add(called)
    return graph


def build_call_graph_text(graph: dict) -> str:
    lines = ["GRAFO DE LLAMADAS APROXIMADO (regex, puede tener falsos positivos)",
              "=" * 60, ""]
    if not graph:
        lines.append("(Sin resultados; requiere indice de simbolos con funciones/metodos)")
        return "\n".join(lines) + "\n"
    for caller in sorted(graph):
        callees = ", ".join(sorted(graph[caller]))
        lines.append(f"{caller} -> {callees}")

    reverse = defaultdict(set)
    for caller, callees in graph.items():
        for c in callees:
            reverse[c].add(caller)
    lines.append("\n\nUSO INVERSO (funcion -> quien la llama)")
    lines.append("=" * 60)
    for callee in sorted(reverse):
        callers = ", ".join(sorted(reverse[callee]))
        lines.append(f"{callee}: {callers}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1h: TODO/FIXME/HACK/XXX con contexto y autor
# --------------------------------------------------------------------------

def _git_blame_map(root: Path, rel: str) -> dict:
    """Devuelve {linea: (autor, fecha)} via `git blame --line-porcelain`,
    o {} si no aplica/falla."""
    out = sh(["git", "blame", "--line-porcelain", "--", rel], cwd=str(root))
    if not out:
        return {}
    mapping = {}
    current_line = None
    author = None
    date = None
    for line in out.splitlines():
        m = re.match(r"^[0-9a-f]{40}\s+\d+\s+(\d+)", line)
        if m:
            current_line = int(m.group(1))
            author, date = None, None
        elif line.startswith("author "):
            author = line[len("author "):]
        elif line.startswith("author-time "):
            try:
                ts = int(line[len("author-time "):])
                date = time.strftime("%Y-%m-%d", time.localtime(ts))
            except ValueError:
                date = ""
        elif line.startswith("\t") and current_line is not None:
            if current_line not in mapping and author:
                mapping[current_line] = (author, date or "")
    return mapping


def extract_todos(files: list, root: Path) -> list:
    use_git = is_git_repo(str(root))
    todos = []
    for f in files:
        rel = str(f.relative_to(root))
        text = read_text(f, limit=300_000)
        if not text or not TODO_PATTERN.search(text):
            continue
        blame_map = _git_blame_map(root, rel) if use_git else {}
        for i, line in enumerate(text.splitlines(), start=1):
            m = TODO_PATTERN.search(line)
            if m:
                author, date = blame_map.get(i, ("", ""))
                todos.append({
                    "file": rel, "line": i, "marker": m.group(1),
                    "text": m.group(2).strip()[:200],
                    "author": author, "date": date,
                })
    return todos


def build_todos_text(todos: list) -> str:
    lines = ["TODO / FIXME / HACK / XXX (deuda tecnica)", "=" * 60, ""]
    if not todos:
        lines.append("(No se encontraron marcadores)")
        return "\n".join(lines) + "\n"
    by_marker = defaultdict(list)
    for t in todos:
        by_marker[t["marker"]].append(t)
    for marker in ["FIXME", "HACK", "XXX", "TODO"]:
        items = by_marker.get(marker, [])
        if not items:
            continue
        lines.append(f"\n### {marker} ({len(items)})")
        for t in items:
            who = f" [{t['author']}, {t['date']}]" if t["author"] else ""
            lines.append(f"  {t['file']}:{t['line']}{who} - {t['text']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1i: mapa de configuracion (env vars, flags CLI, ficheros de config)
# --------------------------------------------------------------------------

def build_config_map(stats: dict) -> dict:
    env_vars = defaultdict(set)
    cli_flags = defaultdict(set)
    config_files = []
    for rel, meta in stats.items():
        lang = meta["lang"]
        base = Path(rel).name
        if base in CONFIG_FILE_NAME_HINTS or base.startswith(".env"):
            config_files.append(rel)

        env_patterns = ENV_VAR_PATTERNS.get(lang, [])
        flag_patterns = CLI_FLAG_PATTERNS.get(lang, [])
        if not env_patterns and not flag_patterns:
            continue
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        for pattern in env_patterns:
            for m in re.finditer(pattern, text):
                env_vars[m.group(1)].add(rel)
        for pattern in flag_patterns:
            for m in re.finditer(pattern, text):
                cli_flags[m.group(1)].add(rel)

    return {
        "env_vars": env_vars,
        "cli_flags": cli_flags,
        "config_files": sorted(set(config_files)),
    }


def build_config_map_text(config_map: dict) -> str:
    lines = ["MAPA DE CONFIGURACION (env vars, flags CLI, ficheros de config)",
              "=" * 60, ""]
    lines.append("Ficheros de configuracion detectados:")
    for f in config_map["config_files"]:
        lines.append(f"  - {f}")
    if not config_map["config_files"]:
        lines.append("  (ninguno)")

    lines.append("\nVariables de entorno leidas en el codigo:")
    for var in sorted(config_map["env_vars"]):
        files = ", ".join(sorted(config_map["env_vars"][var]))
        lines.append(f"  {var}: {files}")
    if not config_map["env_vars"]:
        lines.append("  (ninguna detectada)")

    lines.append("\nFlags de CLI detectados (add_argument / #[arg]):")
    for flag in sorted(config_map["cli_flags"]):
        files = ", ".join(sorted(config_map["cli_flags"][flag]))
        lines.append(f"  {flag}: {files}")
    if not config_map["cli_flags"]:
        lines.append("  (ninguno detectado)")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1j: cobertura de tests por convencion de nombres
# --------------------------------------------------------------------------

def build_test_coverage(stats: dict) -> dict:
    """Empareja ficheros de test con su fichero fuente por convencion de
    nombre (test_x.py <-> x.py, x.test.ts <-> x.ts, etc). Heuristico."""
    test_map = {}
    all_rels = set(stats.keys())
    for rel in stats:
        base = Path(rel).name
        for pattern, target_tpl in TEST_NAME_PATTERNS:
            m = pattern.match(base)
            if m:
                target_name = target_tpl.format(m.group(1))
                for candidate in all_rels:
                    if Path(candidate).name == target_name:
                        test_map[candidate] = rel
                break

    untested = []
    for rel, meta in stats.items():
        if rel in test_map:
            continue
        base = Path(rel).name
        if any(part in TEST_DIR_HINTS for part in Path(rel).parts):
            continue
        if base.startswith("test_") or "_test." in base or ".test." in base or ".spec." in base:
            continue
        if meta["lang"] == "rust":
            text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
            if "#[cfg(test)]" in text or "#[test]" in text:
                continue
        if meta["lang"] in ("python", "javascript", "typescript", "go", "rust",
                              "java", "ruby") and meta["loc"] > 5:
            untested.append(rel)

    return {"tested": test_map, "untested": sorted(untested)}


def build_test_coverage_text(coverage: dict) -> str:
    lines = ["COBERTURA DE TESTS POR CONVENCION DE NOMBRE (heuristico)", "=" * 60, ""]
    lines.append(f"Ficheros con test asociado detectado: {len(coverage['tested'])}")
    for src, test in sorted(coverage["tested"].items()):
        lines.append(f"  {src}  <-  {test}")

    lines.append(f"\nCandidatos SIN test asociado detectado ({len(coverage['untested'])}):")
    lines.append("(heuristico por nombre de fichero; puede haber falsos positivos)")
    for rel in coverage["untested"]:
        lines.append(f"  - {rel}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1k: "god files" - ranking de ficheros pesados/candidatos a refactor
# --------------------------------------------------------------------------

def build_god_files(stats: dict, symbols: list) -> list:
    """Ranking heuristico: LOC + numero de simbolos + complejidad ciclomatica
    aproximada (conteo de palabras clave de control de flujo por regex)."""
    symbol_count = defaultdict(int)
    for s in symbols:
        symbol_count[s["file"]] += 1

    ranked = []
    for rel, meta in stats.items():
        if meta["loc"] <= 0:
            continue
        keywords = CONTROL_FLOW_KEYWORDS.get(meta["lang"])
        complexity = 0
        if keywords:
            text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
            for kw in keywords:
                complexity += len(re.findall(kw, text))
        score = meta["loc"] + symbol_count.get(rel, 0) * 5 + complexity * 2
        ranked.append({
            "file": rel, "loc": meta["loc"], "symbols": symbol_count.get(rel, 0),
            "complexity_aprox": complexity, "score": score,
        })
    ranked.sort(key=lambda x: -x["score"])
    return ranked


def build_god_files_text(ranked: list) -> str:
    lines = ["\"GOD FILES\": RANKING DE FICHEROS PESADOS (candidatos a refactor)",
              "=" * 60,
              "Score = LOC + simbolos*5 + complejidad_aprox*2 (heuristico, no una metrica formal)",
              ""]
    for r in ranked[:40]:
        lines.append(f"  {r['file']}: score={r['score']} "
                      f"(loc={r['loc']}, simbolos={r['symbols']}, complejidad~{r['complexity_aprox']})")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1l: superficie publica vs privada por fichero
# --------------------------------------------------------------------------

def build_public_surface(stats: dict, symbols: list) -> dict:
    by_file = defaultdict(list)
    for s in symbols:
        by_file[s["file"]].append(s)

    surface = {}
    for rel, meta in stats.items():
        lang = meta["lang"]
        if lang == "python":
            text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
            all_match = re.search(r"__all__\s*=\s*\[([^\]]*)\]", text, re.DOTALL)
            explicit = set(re.findall(r"['\"](\w+)['\"]", all_match.group(1))) if all_match else set()
            names = [s["name"] for s in by_file.get(rel, [])
                     if s["kind"] in ("function", "class", "method")]
            if explicit:
                # si hay __all__ explicito, ese es el contrato real: todo lo
                # demas (con o sin prefijo _) es interno por convencion.
                public_syms = [n for n in names if n in explicit]
                private_syms = [n for n in names if n not in explicit]
            else:
                public_syms = [n for n in names if not n.startswith("_")]
                private_syms = [n for n in names if n.startswith("_")]
            if public_syms or private_syms:
                surface[rel] = {"public": sorted(set(public_syms)), "private": sorted(set(private_syms))}
        elif lang == "rust":
            text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
            pub_items = re.findall(
                r"^\s*pub(?:\([^)]*\))?\s+(?:fn|struct|enum|trait|const|mod)\s+(\w+)",
                text, re.MULTILINE)
            all_syms = [s["name"] for s in by_file.get(rel, [])]
            private_syms = [n for n in all_syms if n not in pub_items]
            if pub_items or private_syms:
                surface[rel] = {"public": sorted(set(pub_items)), "private": sorted(set(private_syms))}
    return surface


def build_public_surface_text(surface: dict) -> str:
    lines = ["SUPERFICIE PUBLICA VS PRIVADA POR FICHERO (heuristico, python/rust)",
              "=" * 60, ""]
    if not surface:
        lines.append("(Sin resultados; solo se analizan ficheros python/rust)")
        return "\n".join(lines) + "\n"
    for rel in sorted(surface):
        info = surface[rel]
        lines.append(f"\n--- {rel} ---")
        lines.append(f"  publico: {', '.join(info['public']) or '(ninguno detectado)'}")
        lines.append(f"  privado/interno: {', '.join(info['private']) or '(ninguno)'}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1m: dependencias declaradas (manifest) vs realmente importadas
# --------------------------------------------------------------------------

def compute_fanin(stats: dict, dep_graph: dict) -> dict:
    """Estimacion heuristica de cuantos ficheros importan cada fichero del
    proyecto, cruzando el import string contra el nombre/ruta del fichero.
    No resuelve imports de verdad; sirve solo para rankear candidatos."""
    reverse_counts = defaultdict(int)
    for deps in dep_graph.values():
        for dep in deps:
            reverse_counts[dep] += 1

    fanin = {}
    for rel in stats:
        stem = Path(rel).stem
        dotted = str(Path(rel).with_suffix("")).replace(os.sep, ".")
        count = reverse_counts.get(stem, 0) + reverse_counts.get(dotted, 0)
        if count == 0:
            count = sum(v for k, v in reverse_counts.items() if k.endswith(stem))
        if count:
            fanin[rel] = count
    return fanin


def build_deps_check(stats: dict, dep_graph: dict) -> dict:
    declared = defaultdict(set)
    for rel, meta in stats.items():
        base = Path(rel).name
        if base not in MANIFEST_FILES:
            continue
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        names = set()
        if base in ("requirements.txt", "Pipfile"):
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                m = re.match(r"^([A-Za-z0-9_.\-]+)", line)
                if m:
                    names.add(m.group(1).lower())
        elif base == "pyproject.toml":
            for m in re.finditer(r'dependencies\s*=\s*\[(.*?)\]', text, re.DOTALL):
                names.update(n.lower() for n in re.findall(r'"([A-Za-z0-9_.\-]+)', m.group(1)))
        elif base == "Cargo.toml":
            in_deps = False
            for line in text.splitlines():
                if re.match(r"^\[.*dependencies.*\]", line):
                    in_deps = True
                    continue
                if line.startswith("[") and in_deps:
                    in_deps = False
                if in_deps:
                    m = re.match(r"^\s*([A-Za-z0-9_\-]+)\s*=", line)
                    if m:
                        names.add(m.group(1).lower())
        elif base == "package.json":
            try:
                data = json.loads(text)
                for key in ("dependencies", "devDependencies"):
                    names.update(n.lower() for n in data.get(key, {}).keys())
            except json.JSONDecodeError:
                pass
        elif base == "go.mod":
            for m in re.finditer(r"^\s*([\w\.\-/]+)\s+v[\d.]+", text, re.MULTILINE):
                names.add(m.group(1).lower())
        elif base == "Gemfile":
            for m in re.finditer(r"gem\s+['\"]([\w\-]+)['\"]", text):
                names.add(m.group(1).lower())
        if names:
            declared[rel] = names

    used = set()
    for rel, deps in dep_graph.items():
        for dep in deps:
            top = dep.split(".")[0].split("::")[0].split("/")[0].strip('"<>')
            if top:
                used.add(top.lower())

    all_declared = set()
    for names in declared.values():
        all_declared.update(names)

    declared_not_used = sorted(n for n in all_declared
                                 if n not in used and n.replace("-", "_") not in used)
    used_not_declared = sorted(n for n in used
                                 if n not in all_declared and len(n) > 1)

    return {
        "declared_by_manifest": {k: sorted(v) for k, v in declared.items()},
        "declared_not_used": declared_not_used,
        "used_not_declared": used_not_declared,
    }


def build_deps_check_text(check: dict) -> str:
    lines = ["DEPENDENCIAS DECLARADAS (manifest) VS REALMENTE IMPORTADAS", "=" * 60, ""]
    lines.append("Nota: heuristico. 'usadas pero no declaradas' incluira previsiblemente")
    lines.append("modulos de la libreria estandar e imports relativos (falsos positivos).")

    lines.append("\n\nDeclaradas por manifest:")
    for rel, names in check["declared_by_manifest"].items():
        lines.append(f"\n--- {rel} ---")
        for n in names:
            lines.append(f"  - {n}")
    if not check["declared_by_manifest"]:
        lines.append("  (no se encontro ningun manifest de dependencias conocido)")

    lines.append("\n\nDeclaradas pero NO detectadas en ningun import:")
    for n in check["declared_not_used"]:
        lines.append(f"  - {n}")
    if not check["declared_not_used"]:
        lines.append("  (ninguna)")

    lines.append("\n\nUsadas en imports pero NO declaradas en ningun manifest (revisar falsos positivos):")
    for n in check["used_not_declared"][:150]:
        lines.append(f"  - {n}")

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1n: glosario de dominio (terminos no genericos mas frecuentes)
# --------------------------------------------------------------------------

def build_glossary(symbols: list) -> list:
    """Cuenta tokens no genericos extraidos de nombres de simbolos (separa
    snake_case y camelCase), excluyendo vocabulario generico de programacion."""
    freq = defaultdict(int)
    for s in symbols:
        parts = s["name"].split("_")
        expanded = []
        for p in parts:
            expanded.extend(re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", p))
        for tok in expanded:
            tok = tok.lower()
            if len(tok) < 3 or tok in GLOSSARY_STOPWORDS or tok.isdigit():
                continue
            freq[tok] += 1
    return sorted(freq.items(), key=lambda x: -x[1])


def build_glossary_text(ranked: list) -> str:
    lines = ["GLOSARIO DE DOMINIO (terminos no genericos mas frecuentes en simbolos)",
              "=" * 60, ""]
    if not ranked:
        lines.append("(Sin resultados; se necesita indice de simbolos)")
        return "\n".join(lines) + "\n"
    for term, count in ranked[:100]:
        lines.append(f"  {term}: {count}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1o: linea de tiempo de commits agregada (sin LLM)
# --------------------------------------------------------------------------

def build_commit_timeline_text(root: Path) -> str:
    if not is_git_repo(str(root)):
        return "LINEA DE TIEMPO DE COMMITS\n" + "=" * 60 + "\n\n(No es un repositorio git; fase omitida)\n"

    out = sh(["git", "log", "--format=%ad|%s", "--date=format:%Y-%m"], cwd=str(root))
    by_month = defaultdict(list)
    by_type = defaultdict(int)
    conv_re = re.compile(r"^(\w+)(\([\w\-\.]+\))?!?:\s*(.*)")
    for line in out.splitlines():
        if "|" not in line:
            continue
        month, subject = line.split("|", 1)
        by_month[month].append(subject)
        m = conv_re.match(subject)
        by_type[m.group(1).lower() if m else "(sin prefijo convencional)"] += 1

    lines = ["LINEA DE TIEMPO DE COMMITS (agregada, sin LLM)", "=" * 60, ""]
    lines.append("Por tipo de commit (prefijo estilo conventional commits):")
    for t, count in sorted(by_type.items(), key=lambda x: -x[1]):
        lines.append(f"  {t}: {count}")

    lines.append("\n\nPor mes (numero de commits y ejemplos):")
    for month in sorted(by_month):
        subs = by_month[month]
        lines.append(f"\n{month} ({len(subs)} commits)")
        for s in subs[:5]:
            lines.append(f"  - {s}")
        if len(subs) > 5:
            lines.append(f"  ... y {len(subs) - 5} mas")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1r: integracion con vim (tags, quickfix, fzf, loader .vim)
#
# Estos ficheros no son para leer, son para que vim los consuma
# directamente: navegacion de simbolos sin plugins (tags), saltar entre
# hallazgos con :cnext/:cprev (quickfix), y busqueda difusa de simbolos
# (fzf) si el usuario tiene fzf.vim instalado.
# --------------------------------------------------------------------------

VIM_KIND_MAP = {
    "function": "f", "method": "f", "class": "c", "struct": "s",
    "enum": "g", "enumerator": "e", "interface": "i", "variable": "v",
    "member": "m", "macro": "d", "typedef": "t", "namespace": "n",
    "module": "n", "constant": "v", "field": "m",
}

VIM_LOADER_TEMPLATE = r'''" Generado automaticamente por extract_project_knowledge_v2.py (v2)
" Fuentea este fichero para tener comandos de navegacion quickfix listos:
"   :source /ruta/a/vim/project_nav.vim
"
" El fichero `tags` (junto a este script) se registra automaticamente en
" 'tags' al fuentear este fichero, asi que Ctrl-]/:tag funcionan sin tocar
" tu vimrc. Usa rutas absolutas por dentro (vive fuera de la raiz del
" proyecto), asi que sigue resolviendo bien aunque abras vim desde otro sitio.

let s:qf_dir = expand('<sfile>:p:h')
execute 'set tags+=' . fnameescape(s:qf_dir . '/tags')

command! ProjTodos    execute 'cfile ' . s:qf_dir . '/todos.qf'             | copen
command! ProjGodFiles execute 'cfile ' . s:qf_dir . '/god_files.qf'         | copen
command! ProjUntested execute 'cfile ' . s:qf_dir . '/untested.qf'         | copen
command! ProjComplex  execute 'cfile ' . s:qf_dir . '/complex_functions.qf' | copen
command! ProjBusiness execute 'cfile ' . s:qf_dir . '/business_mixed.qf'   | copen
command! ProjAll      execute 'cfile ' . s:qf_dir . '/combined.qf'         | copen

" Filtra la lista combinada por categoria (usa el plugin estandar
" cfilter; si no esta cargado, hace :packadd cfilter primero).
" Uso: :ProjFilter auth
command! -nargs=1 ProjFilter call s:ProjFilterCombined(<q-args>)
function! s:ProjFilterCombined(pattern)
  execute 'cfile ' . s:qf_dir . '/combined.qf'
  try
    packadd! cfilter
    execute 'Cfilter /\[' . a:pattern . '\]/'
  catch
    echo 'Plugin cfilter no disponible; mostrando la lista sin filtrar.'
  endtry
  copen
endfunction

" Busqueda semantica vectorial (Opcion B): embebe la consulta via la API
" de embeddings de OpenAI (unica llamada de red, no una por resultado) y
" ordena por similitud coseno contra el indice local generado con
" --semantic-index. Requiere OPENAI_API_KEY en el entorno.
" Uso: :ProjSemanticVec reconciliacion de bits entre dos claves
command! -nargs=1 ProjSemanticVec call s:ProjSemanticVecSearch(<q-args>)
function! s:ProjSemanticVecSearch(query)
  let l:cmd = 'python3 ' . shellescape('__PY_SCRIPT__')
        \ . ' --root ' . shellescape('__PROJ_ROOT__')
        \ . ' --output ' . shellescape('__OUTPUT_DIRNAME__')
        \ . ' --provider openai --embeddings-model ' . shellescape('__EMB_MODEL__')
        \ . ' --semantic-query ' . shellescape(a:query)
  let l:results = systemlist(l:cmd)
  if empty(l:results)
    echo 'Sin resultados (o el indice de embeddings no existe: genera con --semantic-index).'
    return
  endif
  call setqflist([], ' ', {'title': 'ProjSemanticVec: ' . a:query, 'lines': l:results})
  copen
endfunction

if exists(':FZF')
  function! s:ProjSymbolSink(line)
    let l:parts = split(a:line, "\t")
    if len(l:parts) < 2
      return
    endif
    execute 'edit ' . l:parts[0]
    execute l:parts[1]
  endfunction

  function! ProjSymbolsFzf()
    let l:opts = {}
    let l:opts.source = 'tail -n +2 ' . s:qf_dir . '/symbols.tsv'
    let l:opts.sink = function('s:ProjSymbolSink')
    let l:opts.options = ['--delimiter=\t', '--with-nth=3,4,1,2', '--prompt=Symbols> ']
    call fzf#run(fzf#wrap(l:opts))
  endfunction

  command! ProjSymbols call ProjSymbolsFzf()

  " Busqueda semantica por texto (Opcion A): fzf sobre descripciones en
  " lenguaje natural generadas una vez por el LLM. 100% offline en el
  " momento de buscar (el LLM ya hizo su trabajo al indexar).
  function! ProjSemanticFzf()
    let l:file = s:qf_dir . '/semantic_functions.tsv'
    if !filereadable(l:file)
      echo 'No existe semantic_functions.tsv (genera con --semantic-index).'
      return
    endif
    let l:opts = {}
    let l:opts.source = 'tail -n +2 ' . l:file
    let l:opts.sink = function('s:ProjSymbolSink')
    let l:opts.options = ['--delimiter=\t', '--with-nth=4,3,1,2', '--prompt=Semantic> ']
    call fzf#run(fzf#wrap(l:opts))
  endfunction

  command! ProjSemantic call ProjSemanticFzf()
endif

echo 'project_nav.vim cargado (tags registrado automaticamente): :ProjTodos :ProjGodFiles :ProjUntested :ProjComplex :ProjBusiness :ProjAll :ProjFilter <cat> :ProjSemantic :ProjSemanticVec <consulta>'
'''


def write_vim_tags_file(symbols: list, root: Path, tags_dir: Path) -> Path:
    """Genera un fichero `tags` (formato ctags extendido, el que lee vim de
    forma nativa con Ctrl-]/:tag) a partir del indice de simbolos ya
    calculado — venga de ctags real o del indice de respaldo por regex.
    Vive junto al resto de ficheros generados (tags_dir), no en la raiz
    del proyecto analizado; por eso las rutas se guardan en absoluto (si
    fueran relativas, vim las buscaria relativas a tags_dir, no a root)."""
    lines = [
        "!_TAG_FILE_FORMAT\t2\t/extended format/",
        "!_TAG_FILE_SORTED\t1\t/0=unsorted, 1=sorted, 2=foldcase/",
    ]
    entries = []
    for s in symbols:
        name = s.get("name")
        if not name:
            continue
        line_no = s.get("line") or 1
        kind = VIM_KIND_MAP.get(s.get("kind", ""), (s.get("kind") or "?")[:1])
        file_abs = str((root / s["file"]).resolve()).replace(os.sep, "/")
        entries.append((name, file_abs, line_no, kind))

    entries.sort(key=lambda e: e[0])
    for name, file_abs, line_no, kind in entries:
        lines.append(f'{name}\t{file_abs}\t{line_no};"\t{kind}')

    tags_path = tags_dir / "tags"
    tags_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tags_path


def write_symbols_tsv(symbols: list, path: Path) -> None:
    """Lista plana file/line/kind/name/signature para fzf u otros
    selectores difusos (funciona incluso sin ctags instalado)."""
    lines = ["file\tline\tkind\tname\tsignature"]
    for s in sorted(symbols, key=lambda s: (s["file"], s.get("line", 0))):
        sig = (s.get("signature") or "").replace("\t", " ")
        lines.append(f"{s['file']}\t{s.get('line', 1)}\t{s['kind']}\t{s['name']}\t{sig}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def qf_todos(todos: list) -> list:
    return [(t["file"], t["line"], f"[{t['marker']}] {t['text']}") for t in todos]


def qf_god_files(god_files: list) -> list:
    return [(r["file"], 1,
              f"god-file score={r['score']} loc={r['loc']} "
              f"simbolos={r['symbols']} complejidad~{r['complexity_aprox']}")
            for r in god_files]


def qf_untested(test_coverage: dict) -> list:
    return [(rel, 1, "sin test asociado (heuristico)") for rel in test_coverage["untested"]]


def qf_complex_functions(complex_functions: list) -> list:
    return [(s["file"], s["line"], f"funcion compleja: {s['name']} (hasta L{s.get('end') or '?'})")
            for s in complex_functions]


def qf_business_flagged(business_report: list) -> list:
    return [(r["file"], 1, f"logica-vs-infra: {r['tag']} (dominante={r['dominant_category']})")
            for r in business_report
            if r["tag"] not in ("logica_de_negocio (candidato)", "indeterminado")]


def write_quickfix_file(path: Path, entries: list) -> None:
    lines = [f"{f}:{ln}:{msg}" for f, ln, msg in entries]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_combined_quickfix(path: Path, categorized: dict) -> None:
    lines = []
    for cat, entries in categorized.items():
        for f, ln, msg in entries:
            lines.append(f"{f}:{ln}:[{cat}] {msg}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_vim_integration(output_dir: Path, root: Path, symbols: list, todos: list,
                             god_files: list, test_coverage: dict, complex_functions: list,
                             business_report: list,
                             embeddings_model: str = DEFAULT_EMBEDDING_MODEL) -> dict:
    """Genera todo el paquete de integracion con vim: tags junto al resto de
    la salida, quickfix lists (por categoria + combinada), symbols.tsv para
    fzf, y un .vim que ata comandos a todo ello (incluyendo, si se generan
    despues con --semantic-index, busqueda semantica por texto y por
    embeddings). Devuelve un resumen para el informe de extraccion."""
    vim_dir = output_dir / "vim"
    vim_dir.mkdir(parents=True, exist_ok=True)

    tags_path = write_vim_tags_file(symbols, root, vim_dir)
    write_symbols_tsv(symbols, vim_dir / "symbols.tsv")

    categorized = {
        "todo": qf_todos(todos),
        "god-file": qf_god_files(god_files),
        "untested": qf_untested(test_coverage),
        "complex": qf_complex_functions(complex_functions),
        "business": qf_business_flagged(business_report),
    }
    write_quickfix_file(vim_dir / "todos.qf", categorized["todo"])
    write_quickfix_file(vim_dir / "god_files.qf", categorized["god-file"])
    write_quickfix_file(vim_dir / "untested.qf", categorized["untested"])
    write_quickfix_file(vim_dir / "complex_functions.qf", categorized["complex"])
    write_quickfix_file(vim_dir / "business_mixed.qf", categorized["business"])
    write_combined_quickfix(vim_dir / "combined.qf", categorized)

    loader = (VIM_LOADER_TEMPLATE
              .replace("__PY_SCRIPT__", str(Path(__file__).resolve()))
              .replace("__PROJ_ROOT__", str(root.resolve()))
              .replace("__OUTPUT_DIRNAME__", output_dir.name)
              .replace("__EMB_MODEL__", embeddings_model))
    (vim_dir / "project_nav.vim").write_text(loader, encoding="utf-8")

    return {
        "tags_path": tags_path,
        "vim_dir": vim_dir,
        "n_symbols": len(symbols),
        "n_entries_combined": sum(len(v) for v in categorized.values()),
    }






def compute_infra_density(stats: dict) -> dict:
    """rel -> {categoria: nº de coincidencias} usando INFRA_CATEGORIES."""
    result = {}
    for rel, meta in stats.items():
        lang = meta["lang"]
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        if not text:
            continue
        counts = {}
        for cat, cfg in INFRA_CATEGORIES.items():
            patterns = list(cfg["patterns"].get(lang, [])) + list(cfg["patterns"].get("any", []))
            if not patterns:
                continue
            total = sum(len(re.findall(p, text, re.IGNORECASE)) for p in patterns)
            if total:
                counts[cat] = total
        if counts:
            result[rel] = counts
    return result


def compute_business_logic_report(stats: dict, symbols: list, infra_by_file: dict,
                                     domain_terms: set) -> list:
    """Heuristico por fichero: cuanto vocabulario de dominio (glosario) tiene
    frente a cuanta densidad de infraestructura, para aproximar donde vive
    la logica de negocio real frente a plumbing/cross-cutting concerns."""
    by_file = defaultdict(list)
    for s in symbols:
        by_file[s["file"]].append(s["name"])

    report = []
    for rel, meta in stats.items():
        loc = max(meta["loc"], 1)
        tokens = []
        for name in by_file.get(rel, []):
            for part in name.split("_"):
                tokens.extend(re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", part))
        tokens = [t.lower() for t in tokens if len(t) >= 3]
        domain_hits = sum(1 for t in tokens if t in domain_terms)
        domain_density = round(domain_hits / loc * 100, 2)

        infra_counts = infra_by_file.get(rel, {})
        infra_total = sum(infra_counts.values())
        infra_density = round(infra_total / loc * 100, 2)

        if infra_counts:
            top_cat, top_count = max(infra_counts.items(), key=lambda x: x[1])
            others_total = infra_total - top_count
            if others_total == 0 or top_count >= 2 * max(others_total, 1):
                dominant = top_cat
            else:
                top3 = sorted(infra_counts.items(), key=lambda x: -x[1])[:3]
                dominant = "mixto(" + ", ".join(c for c, _ in top3) + ")"
        else:
            dominant = "(sin senal de infraestructura)"

        if infra_density < 1.0 and domain_hits > 0:
            tag = "logica_de_negocio (candidato)"
        elif infra_counts and dominant in infra_counts:
            tag = dominant
        elif infra_counts:
            tag = "mixto"
        else:
            tag = "indeterminado"

        report.append({
            "file": rel, "loc": meta["loc"], "domain_hits": domain_hits,
            "domain_density_100loc": domain_density, "infra_total": infra_total,
            "infra_density_100loc": infra_density, "infra_breakdown": infra_counts,
            "dominant_category": dominant, "tag": tag,
        })

    report.sort(key=lambda x: (-x["domain_density_100loc"], x["infra_density_100loc"]))
    return report


def build_business_logic_text(report: list) -> str:
    lines = ["LOGICA DE NEGOCIO VS INFRAESTRUCTURA (heuristico, por fichero)",
              "=" * 60,
              "Categorias de infraestructura editables en INFRA_CATEGORIES, arriba del script.",
              "Densidad = coincidencias por cada 100 lineas de codigo (LOC).",
              "Esto NO es un analisis semantico: es deteccion por vocabulario/patrones.", ""]
    for r in report:
        lines.append(f"\n--- {r['file']} (loc={r['loc']}) ---")
        lines.append(f"  etiqueta: {r['tag']}")
        lines.append(f"  vocabulario de dominio: {r['domain_density_100loc']}/100loc "
                      f"({r['domain_hits']} hits)")
        lines.append(f"  infraestructura total: {r['infra_density_100loc']}/100loc "
                      f"({r['infra_total']} hits)")
        lines.append(f"  categoria dominante: {r['dominant_category']}")
        if r["infra_breakdown"]:
            breakdown = ", ".join(f"{k}={v}" for k, v in
                                    sorted(r["infra_breakdown"].items(), key=lambda x: -x[1]))
            lines.append(f"  desglose: {breakdown}")
    return "\n".join(lines) + "\n"


def build_infra_by_category_text(infra_by_file: dict) -> str:
    by_cat = defaultdict(list)
    for rel, counts in infra_by_file.items():
        for cat, count in counts.items():
            by_cat[cat].append((rel, count))

    lines = ["INFRAESTRUCTURA AGRUPADA POR CATEGORIA (vista invertida)", "=" * 60, ""]
    if not by_cat:
        lines.append("(No se detecto ninguna categoria de INFRA_CATEGORIES en el proyecto)")
        return "\n".join(lines) + "\n"
    for cat in sorted(by_cat, key=lambda c: -sum(n for _, n in by_cat[c])):
        items = sorted(by_cat[cat], key=lambda x: -x[1])
        lines.append(f"\n### {cat} ({len(items)} ficheros)")
        for rel, count in items:
            lines.append(f"  {rel}: {count} coincidencias")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# [v2] FASE 1q: glosario de acronimos y terminos de dominio (estatico)
# --------------------------------------------------------------------------

def collect_acronyms(stats: dict) -> dict:
    """acronimo -> nº de apariciones en todo el proyecto."""
    counts = defaultdict(int)
    for meta in stats.values():
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        if not text:
            continue
        for m in ACRONYM_TOKEN_RE.finditer(text):
            token = m.group(0)
            if token in ACRONYM_EXCLUDE:
                continue
            counts[token] += 1
    return counts


def extract_incode_expansions(stats: dict) -> dict:
    """acronimo -> (expansion, fichero) detectado por patrones tipo
    'QKD (Quantum Key Distribution)' o 'Quantum Key Distribution (QKD)'."""
    found = {}
    for rel, meta in stats.items():
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        if not text:
            continue
        for m in EXPANSION_ACRONYM_FIRST_RE.finditer(text):
            acronym, expansion = m.group(1), m.group(2).strip()
            if acronym in ACRONYM_EXCLUDE or acronym in found:
                continue
            found[acronym] = (expansion, rel)
        for m in EXPANSION_WORDS_FIRST_RE.finditer(text):
            phrase, acronym = m.group(1).strip(), m.group(2)
            if acronym in ACRONYM_EXCLUDE or acronym in found:
                continue
            words = [w for w in re.split(r"\s+", phrase) if w]
            initials = "".join(w[0] for w in words).upper()
            if initials == acronym.upper():
                found[acronym] = (phrase, rel)
    return found


def build_acronym_glossary(stats: dict) -> list:
    counts = collect_acronyms(stats)
    incode = extract_incode_expansions(stats)
    entries = []
    for acronym, count in sorted(counts.items(), key=lambda x: -x[1]):
        if acronym in incode:
            expansion, source = incode[acronym]
            entries.append({"term": acronym, "count": count,
                              "expansion": expansion, "source": f"codigo ({source})"})
        elif acronym in COMMON_ACRONYMS:
            entries.append({"term": acronym, "count": count,
                              "expansion": COMMON_ACRONYMS[acronym], "source": "diccionario comun"})
        else:
            entries.append({"term": acronym, "count": count, "expansion": None, "source": None})
    return entries


def find_first_context(term: str, stats: dict):
    """Primera aparicion de `term` en el proyecto; prefiere lineas que
    parecen comentario/docstring. Devuelve (fichero, linea, snippet) o None."""
    pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    comment_hit, any_hit = None, None
    for rel, meta in stats.items():
        text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
        if not text or not pattern.search(text):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                snippet = line.strip()[:160]
                if any_hit is None:
                    any_hit = (rel, i, snippet)
                if re.search(r'(#|//|/\*|"""|\'\'\')', line):
                    comment_hit = (rel, i, snippet)
                    break
        if comment_hit:
            break
    return comment_hit or any_hit


def build_domain_terms_context(glossary_ranked: list, stats: dict,
                                  limit: int = MAX_DOMAIN_TERMS_FOR_CONTEXT) -> list:
    out = []
    for term, count in glossary_ranked[:limit]:
        out.append({"term": term, "count": count, "context": find_first_context(term, stats)})
    return out


def build_glossary_definitions_text(acronym_entries: list, domain_entries: list) -> str:
    lines = [
        "GLOSARIO: ACRONIMOS Y TERMINOS DE DOMINIO", "=" * 60,
        "Acronimos: expansion encontrada en el propio codigo, o en el",
        "diccionario COMMON_ACRONYMS editable (arriba del script), o",
        "'pendiente' si no se encontro ninguna.",
        "Terminos de dominio: como no son acronimos, no se puede derivar su",
        "significado por regex. Se muestra el contexto de primera aparicion",
        "(no es una definicion). Si ejecutaste la Fase 2 con LLM, revisa",
        "tambien el fichero de definiciones generado (si aplica).", "",
    ]

    lines.append("\n## ACRONIMOS\n")
    for e in acronym_entries:
        if e["expansion"]:
            lines.append(f"  {e['term']} ({e['count']}x) = {e['expansion']}  [{e['source']}]")
        else:
            lines.append(f"  {e['term']} ({e['count']}x) = (sin expansion encontrada - pendiente)")

    lines.append("\n\n## TERMINOS DE DOMINIO (no acronimos)\n")
    for e in domain_entries:
        if e["context"]:
            rel, ln, snippet = e["context"]
            lines.append(f"  {e['term']} ({e['count']}x): {rel}:{ln} -> \"{snippet}\"")
        else:
            lines.append(f"  {e['term']} ({e['count']}x): (sin contexto de comentario encontrado)")

    return "\n".join(lines) + "\n"




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


def select_key_files(stats: dict, god_files: list, entrypoints: list,
                       business_report: list, fanin: dict,
                       n: int = CONVENTIONS_SAMPLE_SIZE) -> list:
    """Elige una muestra representativa combinando señales ya calculadas en
    Fase 1, en vez de simplemente coger los ficheros mas grandes por
    lenguaje: puntos de entrada, ficheros complejos ("god files"),
    candidatos a logica de negocio pura, y ficheros muy importados
    (fan-in). El objetivo es que el LLM vea codigo core real, no solo
    codigo largo."""
    chosen, seen = [], set()

    def add(rel):
        if rel in stats and rel not in seen:
            seen.add(rel)
            chosen.append(rel)

    quota = max(1, n // 4)
    for rel in entrypoints[:quota]:
        add(rel)
    for r in god_files[:quota]:
        add(r["file"])
    for r in business_report:
        if r["tag"] == "logica_de_negocio (candidato)":
            add(r["file"])
        if len(chosen) >= 3 * quota:
            break
    for rel, _ in sorted(fanin.items(), key=lambda x: -x[1])[:quota]:
        add(rel)

    # relleno si aun queda hueco: resto de god_files por orden de score
    if len(chosen) < n:
        for r in god_files:
            add(r["file"])
            if len(chosen) >= n:
                break

    return [(rel, stats[rel]) for rel in chosen[:n]]


def phase_conventions_and_patterns(stats: dict, god_files: list, entrypoints: list,
                                      business_report: list, fanin: dict,
                                      model: str, api_key: str) -> str:
    sample = select_key_files(stats, god_files, entrypoints, business_report, fanin)

    blocks = []
    for rel, meta in sample:
        text = read_text(meta["path"], limit=8000)
        blocks.append(f"### {rel} ({meta['lang']})\n```\n{text}\n```")

    system = (
        "Eres un revisor de codigo senior. A partir de una muestra de "
        "ficheros representativos (elegidos por ser puntos de entrada, "
        "ficheros centrales/complejos, candidatos a logica de negocio pura, "
        "o muy importados por otros ficheros), infieres las convenciones "
        "reales del proyecto y los patrones de diseno realmente presentes. "
        "No fuerces la deteccion de patrones: si no ves ninguno claro, dilo "
        "explicitamente. Cuando identifiques un patron, cita el fichero y, "
        "si puedes, la clase/funcion concreta que lo implementa. Responde "
        "en Markdown, en castellano."
    )
    user = (
        "Muestra de ficheros representativos:\n\n" + "\n\n".join(blocks) + "\n\n"
        "Documenta dos cosas por separado:\n\n"
        "1. CONVENCIONES: estilo de nombres, organizacion de carpetas/modulos, "
        "manejo de errores, logging, patrones de testing, formato/lint "
        "aparente, estilo de comentarios/docstrings.\n\n"
        "2. PATRONES DE DISENO: que patrones de diseno (GoF u otros: "
        "Factory, Singleton, Observer, Strategy, Builder, Adapter, "
        "Repository, Command, Visitor, Decorator, dependency injection, "
        "etc.) estan realmente presentes en esta muestra, con cita de "
        "fichero/clase. Si algo parece un patron pero no estas seguro, "
        "marcalo como 'posible' en vez de afirmarlo."
    )
    return call_llm(system, user, model, api_key, max_tokens=2500)





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


def phase_glossary_definitions(acronym_entries: list, domain_entries: list,
                                  model: str, api_key: str, cache: Cache) -> str:
    """Sub-paso opcional de Fase 2: define SOLO lo que la Fase 1 no pudo
    resolver por si sola (acronimos sin expansion en el codigo/diccionario
    comun, y terminos de dominio sin expansion posible por definicion)."""
    pending_acronyms = [e for e in acronym_entries if not e["expansion"]]
    pending_domain = domain_entries

    if not pending_acronyms and not pending_domain:
        return "No hay terminos pendientes: todo se resolvio en la fase estatica.\n"

    terms_signature = ",".join(sorted(
        [e["term"] for e in pending_acronyms] + [e["term"] for e in pending_domain]))
    cache_key = ("glossary_defs::" + _ACTIVE_PROVIDER["name"] + "::" +
                  hashlib.sha1(terms_signature.encode()).hexdigest())
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    lines_in = []
    for e in pending_acronyms:
        lines_in.append(f"- ACRONIMO: {e['term']} (aparece {e['count']} veces; "
                          "sin expansion encontrada en el codigo)")
    for e in pending_domain:
        if e["context"]:
            rel, ln, snippet = e["context"]
            lines_in.append(f"- TERMINO: {e['term']} (aparece {e['count']} veces). "
                              f"Contexto: \"{snippet}\" ({rel}:{ln})")
        else:
            lines_in.append(f"- TERMINO: {e['term']} (aparece {e['count']} veces; "
                              "sin contexto de comentario encontrado)")

    system = (
        "Eres un ingeniero senior escribiendo un glosario tecnico para un "
        "desarrollador nuevo en el proyecto. Para cada entrada da una "
        "definicion breve (1-2 frases) en castellano. Usa el contexto dado "
        "si existe; si no, usa tu conocimiento general del dominio (musica, "
        "audio, machine learning, criptografia, sistemas, etc). Si de "
        "verdad no puedes inferir un significado razonable, responde "
        "exactamente: 'sin definicion clara disponible'. No inventes "
        "hechos especificos del proyecto que no se puedan inferir del "
        "contexto dado."
    )
    user = ("Terminos a definir:\n\n" + "\n".join(lines_in) +
             "\n\nResponde en formato '### termino' seguido de la definicion, uno por uno.")
    result = call_llm(system, user, model, api_key, max_tokens=2500)
    cache.set(cache_key, result)
    return result


# --------------------------------------------------------------------------
# [v2] FASE 2 (opcional): busqueda semantica de funciones
#
# Opcion A (siempre disponible con --semantic-index): una frase en lenguaje
# natural por funcion, generada una vez por el LLM y cacheada. La busqueda
# en si es texto plano via fzf, 100% offline.
#
# Opcion B (solo con --provider openai): ademas de la frase, un embedding
# real por funcion. La busqueda en si SI necesita una llamada de red (para
# embeber la consulta), pero es una unica llamada barata, no una llamada
# por resultado.
# --------------------------------------------------------------------------

def phase_function_semantic_index(stats: dict, symbols: list, cache: Cache,
                                     model: str, api_key: str) -> list:
    """Una frase por funcion, pensada para busqueda semantica (no repite el
    nombre, describe el proposito). Un LLM call por fichero (no por
    funcion), para que el coste escale con nº de ficheros, no de funciones."""
    by_file = defaultdict(list)
    for s in symbols:
        if s["kind"] in ("function", "method"):
            by_file[s["file"]].append(s)

    results = []
    items = list(by_file.items())[:MAX_FILES_FOR_SUMMARY]
    total = len(items)
    for i, (rel, funcs) in enumerate(items, 1):
        meta = stats.get(rel)
        if not meta:
            continue
        cache_key = f"funcsem::{_ACTIVE_PROVIDER['name']}::{rel}::{meta['hash']}"
        cached = cache.get(cache_key)
        if cached is not None:
            desc_map = json.loads(cached)
        else:
            text = read_text(meta["path"], limit=MAX_FILE_BYTES_FOR_LLM)
            if not text.strip():
                continue
            print(f"  [{i}/{total}] indexando funciones de {rel}")
            names = sorted({f["name"] for f in funcs})
            system = (
                "Eres un ingeniero de software. Para cada funcion/metodo listada, "
                "escribe UNA frase corta (en castellano) que describa que hace, "
                "pensada para busqueda semantica: usa palabras que alguien "
                "buscaria por significado, no te limites a repetir el nombre. "
                "Responde EXACTAMENTE una linea por funcion, formato "
                "'nombre_funcion: descripcion'. No incluyas funciones fuera de "
                "la lista dada. No anadas nada mas, ni encabezados ni markdown."
            )
            user = (
                f"Fichero: {rel}\n\nFunciones a describir: {', '.join(names)}\n\n"
                "Contenido del fichero:\n```\n" + text + "\n```"
            )
            raw = call_llm(system, user, model, api_key, max_tokens=1500)
            desc_map = {}
            for line in raw.splitlines():
                m = re.match(r"\s*[-*]?\s*([A-Za-z_]\w*)\s*:\s*(.+)", line)
                if m and m.group(1) in names:
                    desc_map[m.group(1)] = m.group(2).strip()
            cache.set(cache_key, json.dumps(desc_map, ensure_ascii=False))

        for f in funcs:
            desc = desc_map.get(f["name"])
            if desc:
                results.append({"file": f["file"], "line": f["line"],
                                  "name": f["name"], "description": desc})
    return results


def write_semantic_functions_tsv(entries: list, path: Path) -> None:
    lines = ["file\tline\tname\tdescription"]
    for e in entries:
        desc = e["description"].replace("\t", " ").replace("\n", " ")
        lines.append(f"{e['file']}\t{e['line']}\t{e['name']}\t{desc}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _call_openai_embeddings(texts: list, model: str, api_key: str, retries: int = 3) -> list:
    body = json.dumps({"model": model, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_EMBEDDINGS_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in items]
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                detail = str(e)
            raise RuntimeError(f"Error de la API de embeddings: {e.code} - {detail[:300]}")
        except urllib.error.URLError as e:
            last_err = e
            time.sleep(3)
    raise RuntimeError(f"No se pudo contactar la API de embeddings tras {retries} intentos: {last_err}")


def phase_semantic_embeddings(semantic_entries: list, model: str, api_key: str,
                                 provider: str, cache: Cache) -> list:
    """Embedding real por funcion, a partir de su descripcion (Opcion A).
    Requiere --provider openai: la API publica de Anthropic no ofrece un
    endpoint de embeddings."""
    if provider != "openai":
        print("  [semantic-index] los embeddings reales requieren --provider openai; "
              "se omite (el indice de descripciones + fzf sigue disponible).")
        return []

    out = []
    for i in range(0, len(semantic_entries), EMBEDDING_BATCH_SIZE):
        batch = semantic_entries[i:i + EMBEDDING_BATCH_SIZE]
        cache_keys = [f"embed::{model}::{e['file']}::{e['name']}::{e['line']}::{e['description']}"
                       for e in batch]
        vectors = [None] * len(batch)
        to_fetch_idx = []
        for idx, ck in enumerate(cache_keys):
            cached = cache.get(ck)
            if cached is not None:
                vectors[idx] = json.loads(cached)
            else:
                to_fetch_idx.append(idx)

        if to_fetch_idx:
            texts = [f"{batch[idx]['name']}: {batch[idx]['description']}" for idx in to_fetch_idx]
            print(f"  [semantic-index] embebiendo lote {i // EMBEDDING_BATCH_SIZE + 1} "
                  f"({len(texts)} funciones)...")
            fetched = _call_openai_embeddings(texts, model, api_key)
            for idx, vec in zip(to_fetch_idx, fetched):
                vectors[idx] = vec
                cache.set(cache_keys[idx], json.dumps(vec))

        for e, vec in zip(batch, vectors):
            out.append({**e, "vector": vec})
    return out


def write_embeddings_jsonl(entries: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({
                "file": e["file"], "line": e["line"], "name": e["name"],
                "description": e["description"], "vector": e["vector"],
            }, ensure_ascii=False) + "\n")


def load_embeddings_jsonl(path: Path) -> list:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def run_semantic_query(output_dir: Path, query: str, provider: str,
                          embeddings_model: str, api_key: str, top_k: int) -> None:
    """Modo standalone (--semantic-query): NO ejecuta la extraccion. Carga
    el indice de embeddings ya generado, embebe la consulta (unica llamada
    de red), y saca resultados en formato quickfix por stdout — pensado
    para :cexpr systemlist(...) desde vim."""
    emb_path = output_dir / "vim" / "embeddings.jsonl"
    if not emb_path.exists():
        print(f"[ERROR] No existe {emb_path}. Genera el indice antes con --semantic-index.",
              file=sys.stderr)
        sys.exit(1)
    if provider != "openai":
        print("[ERROR] La busqueda semantica por vectores requiere --provider openai.",
              file=sys.stderr)
        sys.exit(1)

    entries = load_embeddings_jsonl(emb_path)
    if not entries:
        print("[ERROR] El indice de embeddings esta vacio.", file=sys.stderr)
        sys.exit(1)

    query_vec = _call_openai_embeddings([query], embeddings_model, api_key)[0]
    scored = sorted(
        ((cosine_similarity(query_vec, e["vector"]), e) for e in entries),
        key=lambda x: -x[0],
    )
    for score, e in scored[:top_k]:
        desc = e["description"].replace("\n", " ")
        print(f"{e['file']}:{e['line']}:[{score:.3f}] {e['name']} - {desc}")




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
    parser.add_argument("--semantic-index", action="store_true",
                          help="ademas de la Fase 2, genera un indice de descripciones "
                               "semanticas por funcion (fzf, Opcion A) y, si --provider "
                               "openai, tambien embeddings reales (Opcion B)")
    parser.add_argument("--semantic-query", metavar="TEXTO", default=None,
                          help="modo standalone: NO ejecuta la extraccion. Busca TEXTO en "
                               "el indice de embeddings ya generado con --semantic-index y "
                               "devuelve resultados en formato quickfix por stdout (para "
                               ":ProjSemanticVec desde vim)")
    parser.add_argument("--embeddings-model", default=DEFAULT_EMBEDDING_MODEL,
                          help=f"modelo de embeddings de OpenAI (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--top-k", type=int, default=15,
                          help="numero de resultados para --semantic-query (default: 15)")
    args = parser.parse_args()

    model = args.model or DEFAULT_MODEL_BY_PROVIDER[args.provider]
    api_key_env = args.api_key_env or DEFAULT_API_KEY_ENV[args.provider]
    set_active_provider(args.provider)

    root = Path(args.root).resolve()
    output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.semantic_query:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            print(f"[ERROR] No se encontro la variable de entorno {api_key_env}.", file=sys.stderr)
            sys.exit(1)
        run_semantic_query(output_dir, args.semantic_query, args.provider,
                             args.embeddings_model, api_key, args.top_k)
        return

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

    print("  Indexando subcomandos CLI...")
    cli_index = index_cli_commands(stats)
    write(output_dir, "05_indice_cli.txt", build_cli_index_text(cli_index))

    print("  Construyendo grafo de llamadas aproximado...")
    call_graph = build_call_graph(symbols, stats)
    write(output_dir, "06_grafo_llamadas.txt", build_call_graph_text(call_graph))

    print("  Extrayendo TODO/FIXME/HACK/XXX...")
    todos = extract_todos(files, root)
    write(output_dir, "07_todos_deuda_tecnica.txt", build_todos_text(todos))

    print("  Construyendo mapa de configuracion...")
    config_map = build_config_map(stats)
    write(output_dir, "08_mapa_configuracion.txt", build_config_map_text(config_map))

    print("  Calculando cobertura de tests por convencion de nombres...")
    test_coverage = build_test_coverage(stats)
    write(output_dir, "09_cobertura_tests.txt", build_test_coverage_text(test_coverage))

    print("  Rankeando 'god files'...")
    god_files = build_god_files(stats, symbols)
    write(output_dir, "10_god_files.txt", build_god_files_text(god_files))

    print("  Detectando superficie publica/privada...")
    public_surface = build_public_surface(stats, symbols)
    write(output_dir, "11_superficie_publica.txt", build_public_surface_text(public_surface))

    print("  Comparando dependencias declaradas vs usadas...")
    deps_check = build_deps_check(stats, dep_graph)
    write(output_dir, "12_dependencias_declaradas_vs_usadas.txt", build_deps_check_text(deps_check))

    print("  Construyendo glosario de dominio...")
    glossary = build_glossary(symbols)
    write(output_dir, "13_glosario_dominio.txt", build_glossary_text(glossary))

    print("  Agregando linea de tiempo de commits...")
    write(output_dir, "14_linea_tiempo_commits.txt", build_commit_timeline_text(root))

    print("  Analizando logica de negocio vs infraestructura...")
    infra_by_file = compute_infra_density(stats)
    domain_terms = {t for t, _ in glossary}
    business_report = compute_business_logic_report(stats, symbols, infra_by_file, domain_terms)
    write(output_dir, "15_logica_negocio_vs_infraestructura.txt", build_business_logic_text(business_report))
    write(output_dir, "16_infraestructura_por_categoria.txt", build_infra_by_category_text(infra_by_file))

    n_business_candidates = sum(1 for r in business_report if r["tag"] == "logica_de_negocio (candidato)")

    print("  Construyendo glosario de acronimos y terminos de dominio...")
    acronym_entries = build_acronym_glossary(stats)
    domain_entries = build_domain_terms_context(glossary, stats)
    write(output_dir, "17_glosario_acronimos_y_terminos.txt",
          build_glossary_definitions_text(acronym_entries, domain_entries))

    n_acronyms_pending = sum(1 for e in acronym_entries if not e["expansion"])

    print("  Generando integracion con vim (tags, quickfix, fzf)...")
    vim_info = write_vim_integration(output_dir, root, symbols, todos, god_files,
                                        test_coverage, complex_functions, business_report,
                                        embeddings_model=args.embeddings_model)
    print(f"    tags: {vim_info['tags_path']}")
    print(f"    quickfix/fzf: {vim_info['vim_dir']}")

    resumen_estatico = (
        "RESUMEN DE LA EXTRACCION ESTATICA\n" + "=" * 60 + "\n\n"
        f"Ficheros analizados: {len(files)}\n"
        f"Simbolos indexados: {len(symbols)}\n"
        f"Ficheros con dependencias detectadas: {len(dep_graph)}\n"
        f"Puntos de entrada detectados: {entrypoints}\n"
        f"Funciones complejas candidatas a explicacion: {len(complex_functions)}\n"
        f"Ficheros con subcomandos CLI detectados: {len(cli_index)}\n"
        f"Funciones con llamadas resueltas en el grafo de llamadas: {len(call_graph)}\n"
        f"Marcadores TODO/FIXME/HACK/XXX encontrados: {len(todos)}\n"
        f"Variables de entorno detectadas: {len(config_map['env_vars'])}\n"
        f"Ficheros de config detectados: {len(config_map['config_files'])}\n"
        f"Ficheros con test asociado: {len(test_coverage['tested'])}\n"
        f"Ficheros sin test asociado (candidatos): {len(test_coverage['untested'])}\n"
        f"Ficheros con superficie publica/privada detectada: {len(public_surface)}\n"
        f"Dependencias declaradas pero no usadas: {len(deps_check['declared_not_used'])}\n"
        f"Terminos de glosario de dominio detectados: {len(glossary)}\n"
        f"Ficheros con senal de infraestructura detectada: {len(infra_by_file)}\n"
        f"Ficheros candidatos a logica de negocio pura: {n_business_candidates}\n"
        f"Acronimos detectados: {len(acronym_entries)} ({n_acronyms_pending} pendientes de definir)\n"
        f"Fichero tags de vim generado en: {vim_info['tags_path']}\n"
        f"Entradas en quickfix combinado (vim/combined.qf): {vim_info['n_entries_combined']}\n"
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

    n_calls_estimate = len(stats) + 7 + min(len(complex_functions), MAX_COMPLEX_FUNCTIONS)
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
    write(output_dir, "18_resumenes_por_fichero.txt",
          "\n\n".join(f"### {rel}\n{s}" for rel, s in summaries.items()))

    print("  Generando mapa semantico...")
    semantic_map = phase_semantic_map(summaries, model, api_key)
    write(output_dir, "19_mapa_semantico.md", semantic_map)

    print("  Generando vision de arquitectura...")
    architecture = phase_architecture(summaries, dep_graph, entrypoints, model, api_key)
    write(output_dir, "20_arquitectura.md", architecture)

    print("  Detectando convenciones y patrones de diseno...")
    conventions = phase_conventions_and_patterns(
        stats, god_files, entrypoints, business_report, compute_fanin(stats, dep_graph),
        model, api_key)
    write(output_dir, "21_convenciones_y_patrones.md", conventions)

    print("  Explicando algoritmos complejos...")
    algo_explanations = phase_algorithm_explanations(complex_functions, root, model, api_key, cache)
    write(output_dir, "22_explicacion_algoritmos.md", algo_explanations)

    print("  Generando base de conocimiento...")
    kb = phase_knowledge_base(architecture, conventions, semantic_map, model, api_key)
    write(output_dir, "23_base_conocimiento.md", kb)

    print("  Generando snippets del proyecto...")
    snippets = phase_snippets(conventions, stats, model, api_key)
    write(output_dir, "24_snippets.snippets", snippets)

    print("  Generando checklist de revision...")
    checklist = phase_review_checklist(architecture, conventions, model, api_key)
    write(output_dir, "25_checklist_revision.md", checklist)

    print("  Generando casos tipicos...")
    typical_cases = phase_typical_cases(architecture, entrypoints, symbols, model, api_key)
    write(output_dir, "26_casos_tipicos.md", typical_cases)

    print("  Definiendo acronimos/terminos pendientes del glosario...")
    glossary_defs = phase_glossary_definitions(acronym_entries, domain_entries, model, api_key, cache)
    write(output_dir, "27_glosario_definiciones.md", glossary_defs)

    if args.semantic_index:
        print("\n[Fase 2 opcional] Indice de busqueda semantica de funciones...")
        semantic_entries = phase_function_semantic_index(stats, symbols, cache, model, api_key)
        write_semantic_functions_tsv(semantic_entries, vim_info["vim_dir"] / "semantic_functions.tsv")
        print(f"    descripciones: {vim_info['vim_dir'] / 'semantic_functions.tsv'} "
              f"({len(semantic_entries)} funciones) -> usable con :ProjSemantic")

        embedding_entries = phase_semantic_embeddings(
            semantic_entries, args.embeddings_model, api_key, args.provider, cache)
        if embedding_entries:
            write_embeddings_jsonl(embedding_entries, vim_info["vim_dir"] / "embeddings.jsonl")
            print(f"    embeddings: {vim_info['vim_dir'] / 'embeddings.jsonl'} "
                  f"({len(embedding_entries)} vectores) -> usable con :ProjSemanticVec")

    print(f"\nListo. Todo el conocimiento esta en: {output_dir}")


if __name__ == "__main__":
    main()
