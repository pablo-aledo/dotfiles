#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          SUITE INDEX  v1.1                                   ║
║        Catálogo consultable y linter de las herramientas del ecosistema      ║
║                                                                              ║
║  La suite tiene más de 150 ficheros .py autocontenidos. Quien los usa (una   ║
║  persona o un agente) tiene que adivinar qué herramienta hace qué, con qué   ║
║  flags y qué produce, y los errores típicos son afirmar que una opción no    ║
║  existe cuando sí existe, o confundir versiones (fingering_v3 ... v6).       ║
║  Esta herramienta lee las cabeceras y los argparse SIN EJECUTAR nada (solo   ║
║  análisis estático con ast) y construye un índice buscable. Además detecta   ║
║  inconsistencias entre herramientas (lint).                                  ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    build   Escanea el directorio y escribe el índice JSON                    ║
║    list    Lista las herramientas (filtro por categoría o texto)             ║
║    find    Busca herramientas por intención ("rearmonizar", "bajo")          ║
║    show    Ficha completa de una herramienta (flags, subcomandos, usos)      ║
║    lint    Detecta inconsistencias entre herramientas y entre doc y código   ║
║    stats   Resumen global del ecosistema                                     ║
║    examples Valida los ejemplos de las cabeceras (--run: los ejecuta)        ║
║    smoke    --help de cada herramienta y flujos mínimos con asserts (--deep) ║
║    weights  Pesos, módulos, programas y red que necesita cada herramienta    ║
║                                                                              ║
║  USO:                                                                        ║
║    python suite_index.py build [--dir DIR] [--out suite_index.json]          ║
║    python suite_index.py list [--category harmony] [--grep texto]            ║
║    python suite_index.py find rearmonizar acordes [--top 8]                  ║
║    python suite_index.py show drummer [--json]                               ║
║    python suite_index.py show drummer --flag style                           ║
║    python suite_index.py lint [--check output-flag style-vocab] [--json]     ║
║    python suite_index.py stats                                               ║
║    python suite_index.py examples [tool ...] [--run] [--all] [--timeout 20]  ║
║    python suite_index.py smoke [tool ...] [--subs] [--deep | --deep-only]    ║
║    python suite_index.py weights [tool ...] [--all] [--network]              ║
║                                                                              ║
║  OPCIONES COMUNES:                                                           ║
║    --dir DIR       Directorio de la suite (default: el de este script)       ║
║    --index FILE    Caché del índice (default: <dir>/.suite_index.json)       ║
║    --rebuild       Ignorar la caché y volver a escanear                      ║
║    --json          Salida en JSON (build/list/find/show/lint/stats)          ║
║                                                                              ║
║  CHECKS DE LINT:                                                             ║
║    output-flag   Herramientas con entrada y sin ninguna opción de salida     ║
║    style-vocab   Vocabularios --style distintos entre herramientas           ║
║    exit-findings Herramientas que devuelven rc!=0 al encontrar hallazgos     ║
║    exit-contract Herramientas con rc=1 por resultado sin documentar 0/1/2/3 ║
║    output-dir    Herramientas que escriben ficheros sin crear el directorio  ║
║    doc-vs-code   Flags citados en la cabecera que el argparse no define      ║
║    undocumented  Flags definidos que la cabecera no menciona                 ║
║    versions      Familias de versiones (x, x_v2, x_v3 ...) y su última       ║
║    syntax        Ficheros que no se pueden analizar                          ║
║    doc-vs-sub    Flags citados para un subcomando que este no define         ║
║    example-args  Ejemplos de cabecera con subcomando/flag/valor inválido     ║
║    missing-refs  Ficheros .py citados en cabeceras que no existen            ║
║    counts        Documentos que afirman otro nº de herramientas              ║
║                                                                              ║
║  LIMITACIONES: build/list/find/show/lint son análisis estático (ast): no ven ║
║  flags dinámicos (bucles, sys.argv a mano) ni choices calculadas al ejecutar.║
║  examples --run, smoke y weights sí ejecutan código de las herramientas      ║
║  (siempre en directorios temporales). Las categorías son orientativas.       ║
║                                                                              ║
║  DEPENDENCIAS: ninguna (solo stdlib)                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import ast
import sys
import warnings
import os
import re
import json
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict, OrderedDict

VERSION = "1.1"
INDEX_SCHEMA = 2

_USE_COLOR = sys.stdout.isatty()


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _c(code):
    if not _USE_COLOR:
        return ""
    return {"B": "\033[1m", "R": "\033[0m", "G": "\033[2m", "CYA": "\033[96m",
            "YEL": "\033[93m", "RED": "\033[91m", "GRN": "\033[92m"}.get(code, "")


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidades de texto
# ══════════════════════════════════════════════════════════════════════════════

_BOX = "═║╔╗╚╝╠╣╦╩╬─│┌┐└┘├┤┬┴┼━┃"


def _fold(s: str) -> str:
    """Minúsculas sin acentos."""
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def _tokens(s: str):
    return [t for t in re.split(r"[^a-z0-9]+", _fold(s)) if len(t) > 1]


def _strip_box(line: str) -> str:
    return line.strip().strip(_BOX).strip()


# Palabras vacías mínimas (es/en) para la búsqueda
_STOP = set("""de la el los las un una y o en con para por del al que se su sus es
the of and to for in on with a an is are from this that como entre sobre mas""".split())

# Sinónimos muy básicos para la búsqueda por intención
_SYN = {
    "rearmonizar": ["reharmonizer", "reharmonize", "reharmonization", "rearmonizacion"],
    "acordes": ["chord", "chords", "armonia", "harmony", "progresion"],
    "bajo": ["bass"], "bateria": ["drum", "drums", "percusion", "drummer"],
    "melodia": ["melody"], "ritmo": ["rhythm", "metric"], "forma": ["form", "section", "structure", "estructura"],
    "digitacion": ["fingering"], "variacion": ["variation", "variations"],
    "fidelidad": ["fidelity", "similarity", "parecido"], "calidad": ["quality", "score"],
    "cadencia": ["cadence", "cadences"], "orquestar": ["orchestrator", "orchestration", "orquestacion"],
    "coral": ["chorale", "satb", "voice", "voces"], "analizar": ["analyzer", "analysis", "analisis"],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Extracción estática (ast)
# ══════════════════════════════════════════════════════════════════════════════

def _lit(node, env):
    """Evalúa un nodo como literal; resuelve nombres de módulo simples."""
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return env.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Call) and node.args and isinstance(node.func, ast.Name):
        if node.func.id in ("list", "tuple", "sorted", "set"):
            inner = _lit(node.args[0], env)
            if inner is _UNKNOWN:
                return _UNKNOWN
            if isinstance(inner, dict):
                inner = list(inner.keys())
            try:
                return sorted(inner) if node.func.id == "sorted" else list(inner)
            except TypeError:
                return _UNKNOWN
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
            and node.func.attr == "keys" and isinstance(node.func.value, ast.Name):
        d = env.get(node.func.value.id, _UNKNOWN)
        if isinstance(d, dict):
            return list(d.keys())
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _lit(node.left, env), _lit(node.right, env)
        if isinstance(a, list) and isinstance(b, list):
            return a + b
    return _UNKNOWN


class _Unknown:
    def __repr__(self):
        return "<dinámico>"


_UNKNOWN = _Unknown()


def _module_env(tree):
    """Asignaciones literales a nivel de módulo (para resolver choices=NOMBRE)."""
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            v = _lit(node.value, env)
            if v is not _UNKNOWN:
                env[node.targets[0].id] = v
    return env


def _extract_cli(tree, env):
    """Devuelve (flags, subcommands, dynamic) de los add_argument/add_parser."""
    scope = {}                  # variable -> nombre de subcomando ('' = global)
    subowner = {}               # variable de add_subparsers -> subcomando que la posee
    sub_help = OrderedDict()
    calls = []
    dynamic = False

    nodes = sorted((n for n in ast.walk(tree) if hasattr(n, "lineno")),
                   key=lambda n: (n.lineno, getattr(n, "col_offset", 0)))
    for n in nodes:
        # x = <algo>.add_parser('nombre', help=...)  /  g = p.add_argument_group()
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Attribute) \
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            attr = n.value.func.attr
            tgt = n.targets[0].id
            if attr == "add_subparsers":
                base = n.value.func.value
                subowner[tgt] = scope.get(base.id, "") if isinstance(base, ast.Name) else ""
            elif attr == "add_parser" and n.value.args:
                name = _lit(n.value.args[0], env)
                if isinstance(name, str):
                    bv = n.value.func.value
                    owner = subowner.get(bv.id, "") if isinstance(bv, ast.Name) else ""
                    name = (owner + " " + name).strip()      # subcomandos anidados: "config show"
                    scope[tgt] = name
                    h = ""
                    for kw in n.value.keywords:
                        if kw.arg == "help":
                            hv = _lit(kw.value, env)
                            h = hv if isinstance(hv, str) else ""
                    sub_help[name] = h
            elif attr in ("add_argument_group", "add_mutually_exclusive_group"):
                base = n.value.func.value
                if isinstance(base, ast.Name):
                    scope[tgt] = scope.get(base.id, "")
        # with-as no es habitual; se ignora
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr == "add_argument":
                base = n.func.value
                sc = scope.get(base.id, "") if isinstance(base, ast.Name) else ""
                calls.append((sc, n))
            elif n.func.attr == "add_parser" and not isinstance(getattr(n, "_x", None), str):
                # add_parser sin asignación: registrar nombre igualmente
                if n.args:
                    name = _lit(n.args[0], env)
                    if isinstance(name, str):
                        bv = n.func.value
                        owner = subowner.get(bv.id, "") if isinstance(bv, ast.Name) else ""
                        name = (owner + " " + name).strip()
                        if name not in sub_help:
                            sub_help[name] = ""

    flags = []
    for sc, n in calls:
        opts = []
        for a in n.args:
            v = _lit(a, env)
            if isinstance(v, str):
                opts.append(v)
            else:
                dynamic = True
        if not opts:
            continue
        kw = {k.arg: k.value for k in n.keywords if k.arg}
        info = {"opts": opts, "sub": sc,
                "positional": not opts[0].startswith("-")}
        for key in ("default", "type", "choices", "help", "action", "nargs", "metavar", "required", "dest"):
            if key in kw:
                if key == "type":
                    t = kw[key]
                    info["type"] = t.id if isinstance(t, ast.Name) else \
                        (t.attr if isinstance(t, ast.Attribute) else "?")
                    continue
                v = _lit(kw[key], env)
                if v is _UNKNOWN:
                    info[key] = None
                    info.setdefault("dynamic", []).append(key)
                else:
                    info[key] = v
        flags.append(info)
    return flags, list(sub_help.items()), dynamic


def _exit_patterns(src: str):
    """Heurística de códigos de salida."""
    pats = []
    if re.search(r"sys\.exit\(\s*0\s+if\s+.+?\s+else\s+1\s*\)", src):
        pats.append("rc=1 si hay hallazgos (sys.exit(0 if ok else 1))")
    if re.search(r"sys\.exit\(\s*main\(\)\s*\)", src):
        pats.append("rc = retorno de main()")
    return pats


_CATEGORIES = [
    ("analysis",   "análisis",   "analyz analis analysis detect metric profile diagnos audit corpus explor"),
    ("harmony",    "armonía",    "chord acorde harmon armon cadence cadenc progression progresion reharm tonic modulat"),
    ("melody",     "melodía",    "melod motif motivo phrase frase theme tema lyric contour"),
    ("rhythm",     "ritmo",      "rhythm ritmo drum bateria percus groove bass bajo tempo meter metr"),
    ("generation", "generación", "generat compos genera diffusion gan latent transformer remi token neural model"),
    ("transform",  "transformación", "transform variation variac mutat style morph blend simplif reduc adapt expand"),
    ("expression", "expresión",  "perform express rubato dynamic climax tension arc emotion humaniz"),
    ("evaluation", "evaluación", "quality score critic fidelity evaluat rank"),
    ("orchestration", "orquestación", "orchestr instrument stem separat synth audio piano guitar part"),
    ("structure",  "estructura", "section form structure song architect harvest segment stitch unif concat merge align"),
    ("education",  "educación",  "etude ear train tutor practice fingering digit teach"),
]


def _categorize(name: str, text: str):
    hay = _fold(name + " " + text)
    best, best_n = None, 0
    for key, _label, kws in _CATEGORIES:
        n = sum(hay.count(k) for k in kws.split())
        # el nombre pesa más
        n += 3 * sum(1 for k in kws.split() if k in _fold(name))
        if n > best_n:
            best, best_n = key, n
    return best or "other"


_OUT_RE = re.compile(r"^--?(output|out|o|out-?dir|outdir|out-?json|export-.*|json|report|save.*|"
                     r"out-?midi|output-?dir|output-?name|output-?prefix|prefix|dest.*|write.*)$")


def _is_output_opt(opt: str) -> bool:
    return bool(_OUT_RE.match(opt))


def parse_tool(path: Path):
    """Analiza un fichero .py y devuelve su ficha."""
    src = path.read_text(encoding="utf-8", errors="replace")
    name = path.stem
    rec = {"name": name, "file": path.name, "size": path.stat().st_size,
           "mtime": path.stat().st_mtime}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError as e:
        rec.update({"error": f"SyntaxError línea {e.lineno}: {e.msg}", "flags": [],
                    "subcommands": [], "purpose": "", "usage": [], "category": "other"})
        return rec

    doc = ast.get_docstring(tree, clean=False) or ""
    lines = [_strip_box(l) for l in doc.splitlines()]
    lines = [l for l in lines if l]
    title = lines[0] if lines else name
    m = re.search(r"\bv(\d+(?:\.\d+)*)\b", " ".join(lines[:3]), re.I)
    version = m.group(1) if m else None
    if not version:
        env0 = _module_env(tree)
        v = env0.get("VERSION")
        version = str(v) if isinstance(v, (str, int, float)) else None

    # propósito: primeras líneas tras el título hasta la primera sección en mayúsculas
    purpose = []
    for l in lines[1:]:
        if re.match(r"^(USO|USAGE|FLUJO|OPCIONES|SUBCOMANDOS|DEPENDENCIAS|EJEMPLOS?|MODOS?)\b", l):
            break
        purpose.append(l)
        if len(" ".join(purpose)) > 420:
            break
    purpose_txt = " ".join(purpose).strip()

    usage = []
    pat = re.compile(r"\bpython3?\s+" + re.escape(path.name))
    for l in doc.splitlines():
        ls = _strip_box(l)
        if pat.search(ls):
            usage.append(ls)

    env = _module_env(tree)
    flags, subs, dynamic = _extract_cli(tree, env)

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".")[0])
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    external = sorted(i for i in imports if i not in stdlib and i != name)
    local = sorted(i for i in external if (path.parent / f"{i}.py").exists())
    third = sorted(i for i in external if i not in local)

    rec.update({
        "title": title, "version": version, "purpose": purpose_txt,
        "usage": usage[:14], "flags": flags, "subcommands": subs,
        "dynamic_cli": dynamic, "has_argparse": bool(flags or "argparse" in imports),
        "exit_patterns": _exit_patterns(src),
        "exit_documented": "contrato de la suite" in src,
        "writes_files": bool(re.search(r"\.save\(|open\([^)]*[\"'][wax]b?[\"']|write_text\(|write_bytes\(", src)),
        "outdir_ok": bool(re.search(r"_ensure_parent|makedirs|\.mkdir\(", src)),
        "local_deps": local, "third_party": third,
        "category": _categorize(name, title + " " + purpose_txt),
        "doc_flags": sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9\-]*)", doc))),
        "doc_text": " ".join(lines)[:2500],
    })
    return rec


# ══════════════════════════════════════════════════════════════════════════════
#  Índice (con caché por mtime/size)
# ══════════════════════════════════════════════════════════════════════════════

def load_index(root: Path, index_path: Path, rebuild=False, quiet=True):
    cache = {}
    if index_path.exists() and not rebuild:
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            if raw.get("schema") == INDEX_SCHEMA:
                cache = {t["file"]: t for t in raw.get("tools", [])}
        except Exception:
            cache = {}
    tools = []
    n_new = 0
    for p in sorted(root.glob("*.py")):
        if p.name == Path(__file__).name and False:
            continue
        old = cache.get(p.name)
        st = p.stat()
        if old and abs(old.get("mtime", 0) - st.st_mtime) < 1e-6 and old.get("size") == st.st_size:
            tools.append(old)
        else:
            tools.append(parse_tool(p))
            n_new += 1
    idx = {"schema": INDEX_SCHEMA, "dir": str(root), "n_tools": len(tools), "tools": tools}
    if n_new or not index_path.exists():
        try:
            _ensure_parent(index_path).write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass          # directorio de solo lectura: se trabaja en memoria
    return idx


def _find_tool(idx, name):
    n = name[:-3] if name.endswith(".py") else name
    for t in idx["tools"]:
        if t["name"] == n:
            return t
    # coincidencia aproximada
    cands = [t for t in idx["tools"] if n in t["name"]]
    return cands[0] if len(cands) == 1 else None


# ══════════════════════════════════════════════════════════════════════════════
#  Subcomandos
# ══════════════════════════════════════════════════════════════════════════════

def _short(s, n=100):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def cmd_build(args, idx):
    out = Path(args.out) if args.out else None
    if out:
        _ensure_parent(out).write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
    errs = [t for t in idx["tools"] if t.get("error")]
    if args.json:
        print(json.dumps({"n_tools": idx["n_tools"], "errors": [t["file"] for t in errs],
                          "out": str(out) if out else None}, ensure_ascii=False))
        return 0
    print(f"{_c('B')}Índice:{_c('R')} {idx['n_tools']} herramientas en {idx['dir']}")
    n_flags = sum(len(t.get("flags", [])) for t in idx["tools"])
    print(f"  flags extraídos: {n_flags}   ficheros con error de sintaxis: {len(errs)}")
    if out:
        print(f"  Escrito: {out}")
    return 0


def cmd_list(args, idx):
    rows = idx["tools"]
    if args.category:
        rows = [t for t in rows if t.get("category") == args.category]
    if args.grep:
        g = _fold(args.grep)
        rows = [t for t in rows if g in _fold(t["name"] + " " + t.get("title", "") + " " + t.get("purpose", ""))]
    if args.json:
        print(json.dumps([{k: t.get(k) for k in ("name", "version", "category", "title")} for t in rows],
                         ensure_ascii=False, indent=1))
        return 0
    for t in rows:
        v = f"v{t['version']}" if t.get("version") else "  "
        print(f"  {t['name']:<34} {v:<6} {t.get('category', ''):<14} {_short(t.get('title', ''), 60)}")
    print(f"\n  {len(rows)} herramientas")
    return 0


def _expand_query(words):
    toks = []
    for w in words:
        for t in _tokens(w):
            if t in _STOP:
                continue
            toks.append(t)
            toks.extend(_SYN.get(t, []))
    return toks


def score_tool(t, qtoks):
    name = _fold(t["name"])
    title = _fold(t.get("title", ""))
    purpose = _fold(t.get("purpose", ""))
    usage = _fold(" ".join(t.get("usage", [])))
    flag_txt = _fold(" ".join(" ".join(f["opts"]) + " " + str(f.get("help") or "")
                              for f in t.get("flags", [])))
    doc = _fold(t.get("doc_text", ""))
    s = 0.0
    for q in qtoks:
        if q in name:
            s += 6
        if q in title:
            s += 4
        if q in purpose:
            s += 2.5
        if q in usage:
            s += 1.5
        if q in flag_txt:
            s += 1
        if q in doc:
            s += 0.5
    return s


def cmd_find(args, idx):
    qtoks = _expand_query(args.query)
    if not qtoks:
        print("[ERROR] consulta vacía", file=sys.stderr)
        return 1
    scored = [(score_tool(t, qtoks), t) for t in idx["tools"]]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], x[1]["name"]))
    scored = scored[:args.top]
    # marcar la última versión de cada familia
    fam = _families(idx)
    if args.json:
        print(json.dumps([{"name": t["name"], "score": round(s, 1), "title": t.get("title"),
                           "purpose": _short(t.get("purpose", ""), 200)} for s, t in scored],
                         ensure_ascii=False, indent=1))
        return 0
    if not scored:
        print("  (sin resultados)")
        return 0
    for s, t in scored:
        base = _family_base(t["name"])
        latest = fam.get(base, {}).get("latest")
        note = f"  {_c('YEL')}[hay versión más reciente: {latest}]{_c('R')}" if latest and latest != t["name"] else ""
        print(f"  {_c('B')}{t['name']:<32}{_c('R')} {s:>5.1f}  {_short(t.get('title', ''), 54)}{note}")
        if t.get("purpose"):
            print(f"      {_c('G')}{_short(t['purpose'], 150)}{_c('R')}")
        if t.get("usage"):
            print(f"      {_c('CYA')}{_short(t['usage'][0], 120)}{_c('R')}")
    return 0


def _fmt_flag(f):
    opts = ", ".join(f["opts"])
    extra = []
    if f.get("choices"):
        ch = f["choices"]
        extra.append("choices=" + ("{" + ",".join(map(str, ch)) + "}" if isinstance(ch, (list, tuple)) else str(ch)))
    if f.get("default") is not None and not f.get("positional"):
        extra.append(f"default={f['default']}")
    if f.get("type"):
        extra.append(f["type"])
    if f.get("action"):
        extra.append(f["action"])
    if f.get("nargs"):
        extra.append(f"nargs={f['nargs']}")
    if f.get("dynamic"):
        extra.append("(" + ",".join(f["dynamic"]) + " dinámico)")
    return opts, "  ".join(extra), _short(f.get("help") or "", 90)


def cmd_show(args, idx):
    t = _find_tool(idx, args.tool)
    if not t:
        cands = [x["name"] for x in idx["tools"] if _fold(args.tool) in _fold(x["name"])]
        print(f"[ERROR] herramienta '{args.tool}' no encontrada."
              + (f" ¿Quizá: {', '.join(cands[:8])}?" if cands else ""), file=sys.stderr)
        return 1
    if args.flag:
        q = args.flag.lstrip("-")
        hits = [f for f in t["flags"] if any(o.lstrip("-") == q for o in f["opts"])]
        if args.json:
            print(json.dumps({"tool": t["name"], "flag": args.flag, "found": bool(hits), "defs": hits},
                             ensure_ascii=False, indent=1))
            return 0 if hits else 2
        if not hits:
            similar = sorted({o for f in t["flags"] for o in f["opts"] if q[:4] in o})
            print(f"  {t['name']} NO define --{q}." + (f" Parecidas: {', '.join(similar)}" if similar else ""))
            if t.get("dynamic_cli"):
                print("  (aviso: este fichero construye parte de su CLI dinámicamente; el resultado no es concluyente)")
            return 2
        for f in hits:
            o, ex, h = _fmt_flag(f)
            sub = f" [subcomando {f['sub']}]" if f["sub"] else ""
            print(f"  {t['name']}{sub}: {o}  {ex}\n      {h}")
        return 0
    if args.json:
        d = {k: v for k, v in t.items() if k not in ("doc_text", "mtime")}
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return 0
    v = f" v{t['version']}" if t.get("version") else ""
    print(f"\n{_c('B')}{t['name']}{v}{_c('R')}  ·  {t.get('category')}  ·  {t['size'] // 1024} KB")
    print(f"  {t.get('title', '')}")
    if t.get("error"):
        print(f"  {_c('RED')}{t['error']}{_c('R')}")
    if t.get("purpose"):
        import textwrap
        print("\n" + textwrap.fill(t["purpose"], 78, initial_indent="  ", subsequent_indent="  "))
    if t.get("usage"):
        print(f"\n{_c('B')}Uso (de la cabecera){_c('R')}")
        for u in t["usage"]:
            print(f"  {u}")
    if t.get("subcommands"):
        print(f"\n{_c('B')}Subcomandos{_c('R')}")
        for n, h in t["subcommands"]:
            print(f"  {n:<16} {_short(h, 80)}")
    pos = [f for f in t["flags"] if f["positional"]]
    opts = [f for f in t["flags"] if not f["positional"]]
    if pos:
        print(f"\n{_c('B')}Argumentos posicionales{_c('R')}")
        for f in pos:
            o, ex, h = _fmt_flag(f)
            sub = f"[{f['sub']}] " if f["sub"] else ""
            print(f"  {sub}{o:<20} {ex}  {h}")
    if opts:
        print(f"\n{_c('B')}Opciones ({len(opts)}){_c('R')}")
        for f in opts:
            o, ex, h = _fmt_flag(f)
            sub = f"[{f['sub']}] " if f["sub"] else ""
            print(f"  {sub}{o:<30} {ex}")
            if h:
                print(f"      {_c('G')}{h}{_c('R')}")
    outs = [o for f in opts for o in f["opts"] if _is_output_opt(o)]
    print(f"\n{_c('B')}Salida{_c('R')}  " + (", ".join(outs) if outs else "(sin opción de salida explícita)"))
    if t.get("exit_patterns"):
        print(f"{_c('B')}Código de salida{_c('R')}  " + "; ".join(t["exit_patterns"]))
    if t.get("third_party"):
        print(f"{_c('B')}Dependencias externas{_c('R')}  " + ", ".join(t["third_party"]))
    if t.get("local_deps"):
        print(f"{_c('B')}Usa otras herramientas{_c('R')}  " + ", ".join(t["local_deps"]))
    if t.get("dynamic_cli"):
        print(f"{_c('YEL')}Aviso: parte del CLI se construye dinámicamente; la lista puede estar incompleta.{_c('R')}")
    fam = _families(idx).get(_family_base(t["name"]))
    if fam and len(fam["members"]) > 1:
        print(f"{_c('B')}Familia{_c('R')}  " + " → ".join(fam["members"]) + f"   (última: {fam['latest']})")
    print()
    return 0


# ── familias de versiones ─────────────────────────────────────────────────────

_VER_RE = re.compile(r"^(.*?)(?:_?v(\d+))?(?:_(\d+))?$")


def _family_base(name):
    m = re.match(r"^(.*?)(?:_v?\d+)+$", name)
    return m.group(1) if m else name


def _family_rank(name):
    nums = re.findall(r"(\d+)", name[len(_family_base(name)):])
    return tuple(int(n) for n in nums)


def _families(idx):
    fams = defaultdict(list)
    for t in idx["tools"]:
        fams[_family_base(t["name"])].append(t["name"])
    out = {}
    for base, members in fams.items():
        members = sorted(members, key=lambda n: (_family_rank(n), n))
        out[base] = {"members": members, "latest": members[-1]}
    return out


# ── lint ──────────────────────────────────────────────────────────────────────

def lint(idx, checks):
    findings = []

    def add(check, tool, sev, msg, extra=None):
        d = {"check": check, "tool": tool, "severity": sev, "message": msg}
        if extra:
            d["extra"] = extra
        findings.append(d)

    tools = idx["tools"]

    if "syntax" in checks:
        for t in tools:
            if t.get("error"):
                add("syntax", t["name"], "error", t["error"])

    if "output-flag" in checks:
        for t in tools:
            if t.get("error") or not t.get("has_argparse"):
                continue
            has_in = any(f["positional"] for f in t["flags"])
            has_out = any(_is_output_opt(o) for f in t["flags"] for o in f["opts"])
            if has_in and not has_out and t["flags"]:
                add("output-flag", t["name"], "info",
                    "recibe entrada pero no define ninguna opción de salida (--output/--out-dir/--json/--export-*); "
                    "¿escribe junto a la entrada o solo a stdout?")

    if "style-vocab" in checks:
        vocab = {}
        for t in tools:
            for f in t.get("flags", []):
                if "--style" in f["opts"] and isinstance(f.get("choices"), (list, tuple)):
                    key = (t["name"], f["sub"])
                    vocab[key] = set(map(str, f["choices"]))
        if vocab:
            allv = defaultdict(list)
            for (tn, sub), vs in vocab.items():
                for v in vs:
                    allv[v].append(tn + (f":{sub}" if sub else ""))
            names = sorted(vocab)
            for (tn, sub) in names:
                vs = vocab[(tn, sub)]
                shared = {v for v in vs if len(set(x.split(":")[0] for x in allv[v])) > 1}
                add("style-vocab", tn + (f":{sub}" if sub else ""), "info",
                    f"--style acepta {len(vs)} valores ({len(shared)} compartidos con otras herramientas)",
                    {"values": sorted(vs), "shared": sorted(shared)})
            # valores 'populares' que faltan en alguna herramienta
            count = defaultdict(int)
            for (tn, sub), vs in vocab.items():
                for v in vs:
                    count[v] += 1
            common = [v for v, n in count.items() if n >= 3]
            for (tn, sub), vs in vocab.items():
                miss = sorted(c for c in common if c not in vs)
                if miss:
                    add("style-vocab", tn + (f":{sub}" if sub else ""), "warn",
                        f"no acepta valores de --style usados por ≥3 herramientas: {', '.join(miss)}")

    if "exit-findings" in checks:
        for t in tools:
            if t.get("exit_documented"):
                continue            # ya documenta el contrato de códigos de salida
            for p in t.get("exit_patterns", []):
                if p.startswith("rc=1 si hay hallazgos"):
                    add("exit-findings", t["name"], "info",
                        "devuelve rc=1 cuando encuentra hallazgos (no solo en errores): en pipelines con "
                        "`set -e` o runner --strict la herramienta 'falla' haciendo su trabajo; "
                        "conviene documentarlo en la cabecera")

    if "output-dir" in checks:
        # Contrato: toda herramienta que escribe ficheros crea el directorio padre
        # (helper _ensure_parent o makedirs/mkdir); si no, falla con FileNotFoundError.
        for t in tools:
            if t.get("error") or not t.get("writes_files"):
                continue
            if not t.get("outdir_ok"):
                add("output-dir", t["name"], "warn",
                    "escribe ficheros pero no crea el directorio de salida "
                    "(usa _ensure_parent(ruta) o os.makedirs(..., exist_ok=True))")

    if "exit-contract" in checks:
        # Contrato: los hallazgos no son un fallo. Quien termina con rc!=0 según un
        # booleano de resultado debe documentar los códigos 0/1/2/3 en su --help.
        for t in tools:
            if t.get("error") or t.get("exit_documented"):
                continue
            if t.get("exit_patterns") and any(p.startswith("rc=1 si hay hallazgos")
                                              for p in t["exit_patterns"]):
                add("exit-contract", t["name"], "warn",
                    "termina con rc=1 según un resultado y no documenta los códigos de "
                    "salida del contrato (0 ok · 1 error · 2 uso · 3 hallazgos con --strict)")

    if "doc-vs-code" in checks:
        everywhere = {o for tt in tools for f in tt.get("flags", []) for o in f["opts"]}
        for t in tools:
            if t.get("error") or not t["flags"] or t.get("dynamic_cli"):
                continue
            defined = {o for f in t["flags"] for o in f["opts"] if o.startswith("--")}
            # ignorar dest típicos de ayuda
            defined.add("--help")
            missing = []
            for d in t.get("doc_flags", []):
                if d.endswith("-") or d in defined:       # prefijos tipo --mt-* no son flags
                    continue
                if d.startswith("--no-") and ("--" + d[5:]) in defined:
                    continue                                # --no-x documenta la negación de --x
                if d in everywhere:
                    continue                                # referencia al flag de OTRA herramienta
                missing.append(d)
            if missing:
                add("doc-vs-code", t["name"], "warn",
                    "la cabecera cita flags que no existen en ninguna herramienta (¿deriva doc/código?): " + ", ".join(sorted(set(missing))[:10]),
                    {"flags": sorted(set(missing))})

    if "undocumented" in checks:
        for t in tools:
            if t.get("error") or not t["flags"] or not t.get("doc_flags"):
                continue
            doc_f = set(t["doc_flags"])
            und = [o for f in t["flags"] for o in f["opts"][:1]
                   if o.startswith("--") and o not in doc_f and f["sub"] == ""]
            frac = len(und) / max(1, sum(1 for f in t["flags"] if not f["positional"]))
            if und and frac > 0.5 and len(und) >= 4:
                add("undocumented", t["name"], "info",
                    f"{len(und)} flags definidos no aparecen en la cabecera", {"flags": und[:15]})

    if "versions" in checks:
        for base, fam in sorted(_families(idx).items()):
            if len(fam["members"]) > 1:
                add("versions", base, "info",
                    "familia de versiones: " + " → ".join(fam["members"]) + f"  (última: {fam['latest']})")

    findings.extend(lint_docs_extra(idx, checks))

    sev_order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: (sev_order[f["severity"]], f["check"], f["tool"]))
    return findings


ALL_CHECKS = ["syntax", "output-flag", "output-dir", "style-vocab", "exit-findings",
              "exit-contract", "doc-vs-code",
              "undocumented", "versions", "doc-vs-sub", "example-args", "missing-refs", "counts"]


def cmd_lint(args, idx):
    checks = args.check or ALL_CHECKS
    bad = [c for c in checks if c not in ALL_CHECKS]
    if bad:
        print(f"[ERROR] checks desconocidos: {', '.join(bad)}. Disponibles: {', '.join(ALL_CHECKS)}",
              file=sys.stderr)
        return 1
    findings = lint(idx, set(checks))
    if args.severity:
        order = {"error": 0, "warn": 1, "info": 2}
        findings = [f for f in findings if order[f["severity"]] <= order[args.severity]]
    if args.json:
        print(json.dumps(findings, ensure_ascii=False, indent=1))
    else:
        cur = None
        for f in findings:
            if f["check"] != cur:
                cur = f["check"]
                print(f"\n{_c('B')}[{cur}]{_c('R')}")
            col = {"error": "RED", "warn": "YEL", "info": "G"}[f["severity"]]
            print(f"  {_c(col)}{f['severity']:<5}{_c('R')} {f['tool']:<30} {_short(f['message'], 150)}")
        print(f"\n  {len(findings)} hallazgos "
              f"({sum(f['severity'] == 'error' for f in findings)} error, "
              f"{sum(f['severity'] == 'warn' for f in findings)} warn, "
              f"{sum(f['severity'] == 'info' for f in findings)} info)")
    # rc: lint informativo por defecto; --strict falla con errores/avisos
    if args.strict and any(f["severity"] in ("error", "warn") for f in findings):
        return 1
    return 0


def cmd_stats(args, idx):
    tools = idx["tools"]
    cats = defaultdict(int)
    for t in tools:
        cats[t.get("category", "other")] += 1
    n_flags = sum(len(t.get("flags", [])) for t in tools)
    third = defaultdict(int)
    for t in tools:
        for d in t.get("third_party", []):
            third[d] += 1
    fams = {b: f for b, f in _families(idx).items() if len(f["members"]) > 1}
    d = {"tools": len(tools), "flags": n_flags, "total_kb": sum(t["size"] for t in tools) // 1024,
         "categories": dict(sorted(cats.items(), key=lambda x: -x[1])),
         "third_party_top": dict(sorted(third.items(), key=lambda x: -x[1])[:12]),
         "version_families": len(fams),
         "with_subcommands": sum(1 for t in tools if t.get("subcommands")),
         "without_argparse": [t["name"] for t in tools if not t.get("has_argparse")][:20]}
    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return 0
    print(f"\n{_c('B')}Ecosistema{_c('R')}: {d['tools']} herramientas · {d['total_kb']} KB · {d['flags']} flags")
    print(f"  con subcomandos: {d['with_subcommands']} · familias de versiones: {d['version_families']}")
    print(f"\n{_c('B')}Por categoría{_c('R')}")
    for k, n in d["categories"].items():
        print(f"  {k:<16} {n:>4} {'█' * min(60, n)}")
    print(f"\n{_c('B')}Dependencias externas más usadas{_c('R')}")
    for k, n in d["third_party_top"].items():
        print(f"  {k:<16} {n:>4}")
    if d["without_argparse"]:
        print(f"\n{_c('B')}Sin argparse detectado{_c('R')}: {', '.join(d['without_argparse'])}")
    print()
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  EJEMPLOS DE CABECERA: extracción, validación estática y ejecución
# ══════════════════════════════════════════════════════════════════════════════

_PLACEHOLDER_RE = re.compile(r"[<>\[\]|{}]|\.\.\.|…|\$\(")
_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9\-]*)")


def _doc_lines(path: Path):
    """Líneas del docstring del módulo, sin marcos de caja."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        return []
    doc = ast.get_docstring(tree, clean=False) or ""
    return [_strip_box(l) for l in doc.splitlines()]


def header_examples(path: Path):
    """Comandos de ejemplo de la cabecera (une las continuaciones con '\\')."""
    lines = _doc_lines(path)
    rx = re.compile(r"^(?:\$\s*)?(?:python3?\s+)?(?:\S*/)?" + re.escape(path.name) +
                    r"(?:\s+(.*))?$")
    out, i = [], 0
    while i < len(lines):
        m = rx.match(lines[i])
        if m:
            cmd = (m.group(1) or "").strip()
            while cmd.endswith("\\") and i + 1 < len(lines):
                i += 1
                cmd = cmd[:-1].rstrip() + " " + lines[i].strip()
            cmd = cmd.rstrip("\\").strip()
            nxt = lines[i + 1].lower() if i + 1 < len(lines) else ""
            # un comentario inmediato "(nota: ... no existe ...)" marca un contraejemplo
            if cmd and cmd[0] not in ":(" and not re.search(r"no existe|no soporta|no admite", nxt):
                out.append(cmd)
        i += 1
    return out


def _example_tokens(cmd: str):
    """Tokens del ejemplo o None si es una plantilla con marcadores."""
    import shlex
    if _PLACEHOLDER_RE.search(cmd):
        return None
    try:
        return shlex.split(cmd, comments=True)
    except ValueError:
        return None


def _flag_table(tool, sub):
    """opt -> ficha para el subcomando dado (+ flags globales)."""
    valid = {}
    for f in tool.get("flags", []):
        if f["positional"]:
            continue
        if f["sub"] in ("", sub, sub.split(" ")[0]):
            for o in f["opts"]:
                valid[o] = f
    return valid


def _takes_value(f):
    return not str(f.get("action") or "").startswith(("store_true", "store_false", "count", "help", "version"))


def validate_example(tool, toks):
    """Comprueba un ejemplo contra el argparse extraído. Devuelve lista de problemas."""
    if tool.get("dynamic_cli") or tool.get("error"):
        return []
    issues = []
    toks = list(toks)
    subs = {n for n, _ in tool.get("subcommands", [])}
    sub = ""
    if subs and len(toks) > 1 and " ".join(toks[:2]) in subs:
        sub = " ".join(toks[:2])
        toks = toks[2:]
    elif subs and toks and not toks[0].startswith("-"):
        if toks[0] in subs:
            sub = toks.pop(0)
        elif re.match(r"^[a-z][a-z0-9_\-]*$", toks[0]):
            return [f"subcomando desconocido '{toks[0]}' (hay: {', '.join(sorted(subs))})"]
    valid = _flag_table(tool, sub)
    seen = set()
    i = 0
    while i < len(toks):
        t = toks[i]
        i += 1
        if not (t.startswith("--") and len(t) > 2):
            continue
        name, eq, val = t.partition("=")
        if name == "--help":
            continue
        f = valid.get(name)
        if f is None:
            cands = [o for o in valid if o.startswith(name)]
            if len(cands) == 1:
                f = valid[cands[0]]
            else:
                where = sorted({g["sub"] for g in tool.get("flags", []) if name in g["opts"] and g["sub"]})
                hint = f" (sí existe en: {', '.join(where)})" if where else ""
                issues.append(f"flag {name} no existe en '{sub or 'global'}'{hint}")
                continue
        seen.add(id(f))
        if not _takes_value(f):
            continue
        values = [val] if eq else []
        if not eq:
            nargs = f.get("nargs")
            n = 10 ** 6 if nargs in ("+", "*") else (nargs if isinstance(nargs, int) else 1)
            while len(values) < n and i < len(toks) and (not toks[i].startswith("-") or re.match(r"^-\d", toks[i])):
                values.append(toks[i])
                i += 1
        ch = f.get("choices")
        if isinstance(ch, (list, tuple)) and ch:
            ok = {str(c) for c in ch}
            for v in values:
                if v not in ok:
                    issues.append(f"valor '{v}' no válido para {name} (admite: {', '.join(sorted(ok))})")
                    break
    return issues


def header_sub_flags(path: Path, subs):
    """{subcomando: {flags citados}} desde los bloques 'sub:  --a --b ...' de la cabecera."""
    res, cur = defaultdict(set), None
    usage_rx = re.compile(r"^(?:python3?\s|\S+\.py\b)")
    for l in _doc_lines(path):
        if not l:
            cur = None
            continue
        m = re.match(r"^([a-z][a-z0-9_\-]*)\s*:\s*(.*)$", l)
        if m and m.group(1) in subs:
            cur = m.group(1)
            res[cur].update(_FLAG_RE.findall(m.group(2)))
        elif m:
            cur = None
        elif cur and "--" in l and not usage_rx.match(l):
            res[cur].update(_FLAG_RE.findall(l))
        else:
            cur = None
    return res


_REF_IGNORE = {"setup.py", "__init__.py", "__main__.py", "main.py", "conftest.py"}


def lint_docs_extra(idx, checks):
    """Checks añadidos en v1.1: doc-vs-sub, example-args, missing-refs, counts."""
    out = []
    root = Path(idx["dir"])
    tools = idx["tools"]
    names = {t["file"] for t in tools} | {p.name for p in root.glob("*.py")}

    def add(check, tool, sev, msg, extra=None):
        d = {"check": check, "tool": tool, "severity": sev, "message": msg}
        if extra:
            d["extra"] = extra
        out.append(d)

    for t in tools:
        if t.get("error"):
            continue
        path = root / t["file"]
        if "doc-vs-sub" in checks and t.get("subcommands") and not t.get("dynamic_cli"):
            subs = {n for n, _ in t["subcommands"]}
            for sub, cited in sorted(header_sub_flags(path, subs).items()):
                valid = set(_flag_table(t, sub)) | {"--help"}
                miss = sorted(c for c in cited if c not in valid and
                              not (c.startswith("--no-") and "--" + c[5:] in valid) and
                              not any(v.startswith(c) for v in valid))
                if miss:
                    add("doc-vs-sub", t["name"], "warn",
                        f"la cabecera cita para '{sub}' flags que ese subcomando no define: {', '.join(miss)}",
                        {"sub": sub, "flags": miss})
        if "example-args" in checks:
            for cmd in header_examples(path):
                toks = _example_tokens(cmd)
                if toks is None:
                    continue
                for msg in validate_example(t, toks):
                    add("example-args", t["name"], "warn", f"{msg}   ← {_short(cmd, 90)}", {"example": cmd})
        if "missing-refs" in checks:
            text = " ".join(_doc_lines(path))
            refs = {r for r in re.findall(r"\b([A-Za-z][A-Za-z0-9_]*\.py)\b", text)
                    if r not in names and r not in _REF_IGNORE and r != t["file"]}
            if refs:
                add("missing-refs", t["name"], "info",
                    "la cabecera cita ficheros .py que no están en la suite: " + ", ".join(sorted(refs)),
                    {"refs": sorted(refs)})

    if "counts" in checks:
        n = len(tools)
        rx = re.compile(r"\b(\d{2,3})\s+(?:herramientas|tools|scripts)\b", re.I)
        cands = []
        for base, depth in ((root, 0), (root / "academia", 3)):
            if not base.is_dir():
                continue
            for p in sorted(base.rglob("*") if depth else base.glob("*")):
                if p.is_file() and p.suffix.lower() in (".md", ".txt", ".html") and not p.name.startswith(".") \
                        and p.stat().st_size < 3_000_000 and len(p.relative_to(base).parts) <= max(depth, 1):
                    cands.append(p)
        for p in cands:
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            claimed = sorted({int(x) for x in rx.findall(txt) if abs(int(x) - n) <= 60 and int(x) != n})
            if claimed:
                add("counts", str(p.relative_to(root)), "info",
                    f"afirma {', '.join(map(str, claimed))} herramientas; el índice encuentra {n}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Fixtures MIDI sintéticos (SMF escrito a mano: solo stdlib)
# ══════════════════════════════════════════════════════════════════════════════

def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _smf_track(events):
    events = sorted(events, key=lambda e: (e[0], 0 if (e[1][0] & 0xF0) == 0x80 else 1))
    data, last = b"", 0
    for tick, msg in events:
        data += _vlq(tick - last) + bytes(msg)
        last = tick
    data += b"\x00\xff\x2f\x00"
    return b"MTrk" + len(data).to_bytes(4, "big") + data


def make_fixture_midi(path: Path, variant=0, bars=8, parts=("mel", "bas", "cho", "drm")):
    """MIDI multipista en 4/4 (melodía, bajo, acordes, batería según parts); variant cambia tonalidad y tempo."""
    import random
    rnd = random.Random(variant)
    tpb = 480
    bar = tpb * 4
    shift = (variant * 2) % 12
    prog = [(0, [0, 4, 7]), (9, [0, 3, 7]), (5, [0, 4, 7]), (7, [0, 4, 7])]   # I vi IV V
    tempo = int(60_000_000 / (96 + 6 * variant))
    meta = [(0, [0xFF, 0x51, 0x03, (tempo >> 16) & 255, (tempo >> 8) & 255, tempo & 255]),
            (0, [0xFF, 0x58, 0x04, 4, 2, 24, 8])]
    mel, bas, cho, drm = [], [], [], []

    def note(ev, ch, p, t0, dur, vel):
        ev.append((t0, [0x90 | ch, p, vel]))
        ev.append((t0 + dur, [0x80 | ch, p, 0]))

    for b in range(bars):
        root, triad = prog[b % 4]
        t = b * bar
        for k in range(8):
            deg = rnd.choice(triad + [2, 9, 11] if k % 4 == 3 else triad)
            note(mel, 0, 72 + shift + root % 12 // 2 + (deg % 12) - (6 if deg > 7 else 0), t + k * tpb // 2, tpb // 2 - 10, 82)
        note(bas, 1, 36 + shift + root, t, 2 * tpb - 10, 90)
        note(bas, 1, 43 + shift + root, t + 2 * tpb, 2 * tpb - 10, 84)
        for iv in triad:
            note(cho, 2, 55 + shift + root + iv, t, bar - 20, 64)
        for k in range(8):
            note(drm, 9, 42, t + k * tpb // 2, 40, 60)
        for k in (0, 2):
            note(drm, 9, 36, t + k * tpb, 60, 90)
        for k in (1, 3):
            note(drm, 9, 38, t + k * tpb, 60, 85)
    spec = [("mel", "Melody", 0, 0, mel), ("bas", "Bass", 33, 1, bas),
            ("cho", "Chords", 48, 2, cho), ("drm", "Drums", None, 9, drm)]
    tracks = [_smf_track(meta)]
    for key, name, prog, ch, ev in spec:
        if key not in parts:
            continue
        head = [(0, [0xFF, 0x03, len(name)] + list(name.encode()))]
        if prog is not None:
            head.append((0, [0xC0 | ch, prog]))
        tracks.append(_smf_track(head + ev))
    hdr = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + tpb.to_bytes(2, "big")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hdr + b"".join(tracks))
    return path


def _fixture_dir(d: Path, n=6, start=0, parts=("mel", "bas", "cho", "drm")):
    d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        make_fixture_midi(d / f"f{i}.mid", start + i, parts=parts)
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  Ejecución controlada de subprocesos
# ══════════════════════════════════════════════════════════════════════════════

_NOFILE_RE = re.compile(r"No such file|FileNotFoundError|no existe|not found|No se encontr|no se encuentra|cannot find|Errno 2|ejecuta antes|run .* first|modelo no encontrado",
                        re.I)


def _run(argv, cwd, timeout=20, mem_gb=None):
    """Ejecuta argv; devuelve (estado, rc, salida). estado: ok|timeout|crash|error."""
    import subprocess
    kw = {}
    if os.name == "posix":
        kw["start_new_session"] = True
        if mem_gb:
            def _lim():
                try:
                    import resource
                    b = int(mem_gb * 1024 ** 3)
                    resource.setrlimit(resource.RLIMIT_AS, (b, b))
                except Exception:
                    pass
            kw["preexec_fn"] = _lim
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"), PYTHONWARNINGS="ignore")
    try:
        p = subprocess.Popen(argv, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, errors="replace", env=env, **kw)
    except OSError as e:
        return "error", 127, str(e)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(p.pid, 9)
            else:
                p.kill()
        except Exception:
            pass
        out, _ = p.communicate()
        return "timeout", None, out or ""
    out = out or ""
    if p.returncode == 0:
        return "ok", 0, out
    if "Traceback (most recent call last)" in out or "MemoryError" in out or "ArrayMemoryError" in out:
        return "crash", p.returncode, out
    return "error", p.returncode, out


def _last_line(out, n=1):
    ls = [l.strip() for l in (out or "").splitlines() if l.strip()]
    return " | ".join(ls[-n:])[:220] if ls else ""


def _missing_module(out):
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", out or "")
    if m:
        return m.group(1).split(".")[0]
    m = re.search(r"ImportError: .*?'([^']+)'", out or "")
    return m.group(1) if m else None


def _select_tools(idx, names):
    if not names:
        return [t for t in idx["tools"] if not t.get("error")]
    out = []
    for n in names:
        t = _find_tool(idx, n)
        if t is None:
            print(f"[ERROR] herramienta desconocida: {n}", file=sys.stderr)
            continue
        out.append(t)
    return out


def _pool_map(fn, items, jobs):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        return list(ex.map(fn, items))


# ══════════════════════════════════════════════════════════════════════════════
#  Subcomando examples
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_example(tool, toks, cwd: Path, parts=("mel", "cho")):
    """Crea en cwd los ficheros .mid/directorios de entrada que el ejemplo menciona."""
    k = 0
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else ""
        is_out = prev.startswith("-") and _is_output_opt(prev.split("=")[0])
        if t.lower().endswith((".mid", ".midi")) and not is_out and "*" not in t and not t.startswith("-"):
            make_fixture_midi(cwd / t, k, parts=parts)
            k += 1
        elif ("*" in t or t.endswith("/")) and not t.startswith("-"):
            d = cwd / (os.path.dirname(t) if "*" in t else t)
            if "*" in t and not os.path.dirname(t):
                d = cwd
            if not is_out:
                _fixture_dir(d, 6, parts=parts)
    return toks


def _example_job(job):
    tool, cmd, root, timeout, run = job
    toks = _example_tokens(cmd)
    rec = {"tool": tool["name"], "cmd": cmd, "status": "plantilla", "detail": ""}
    if toks is None:
        return rec
    issues = validate_example(tool, toks)
    if issues:
        rec.update(status="invalido", detail="; ".join(issues))
        return rec
    rec["status"] = "valido"
    if not run:
        return rec
    import tempfile
    for parts in (("mel", "cho"), ("mel", "bas", "cho", "drm")):     # 2.º intento: con batería
        with tempfile.TemporaryDirectory(prefix="si_ex_") as td:
            cwd = Path(td)
            _prepare_example(tool, toks, cwd, parts)
            st, rc, out = _run([sys.executable, str(root / tool["file"])] + toks, cwd, timeout)
        if st != "error" or _NOFILE_RE.search(out):
            break
    if st == "ok":
        rec["status"] = "ok"
    elif st == "timeout":
        rec["status"] = "timeout"
    elif _NOFILE_RE.search(out) and st != "crash":
        rec.update(status="omitido", detail="necesita un fichero de entrada que no se puede fabricar: " + _last_line(out))
    elif st == "crash":
        rec.update(status="crash", detail=_last_line(out, 2))
    else:
        rec.update(status="error", detail=f"rc={rc}: " + _last_line(out))
    return rec


def cmd_examples(args, idx):
    root = Path(idx["dir"])
    tools = _select_tools(idx, args.tool)
    jobs = []
    for t in tools:
        for cmd in header_examples(root / t["file"]):
            jobs.append((t, cmd, root, args.timeout, args.run))
    results = _pool_map(_example_job, jobs, args.jobs if args.run else 1)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
    else:
        order = {"crash": 0, "invalido": 1, "error": 2, "timeout": 3, "omitido": 4, "plantilla": 5, "valido": 6, "ok": 7}
        shown = [r for r in results if args.all or r["status"] in ("crash", "invalido", "error")]
        for r in sorted(shown, key=lambda r: (order[r["status"]], r["tool"])):
            col = {"crash": "RED", "invalido": "RED", "error": "YEL"}.get(r["status"], "G")
            print(f"  {_c(col)}{r['status']:<9}{_c('R')} {r['tool']:<28} {_short(r['cmd'], 70)}")
            if r["detail"]:
                print(f"            {_c('G')}{_short(r['detail'], 170)}{_c('R')}")
        cnt = defaultdict(int)
        for r in results:
            cnt[r["status"]] += 1
        print(f"\n  {len(results)} ejemplos en {len(tools)} herramientas: " +
              ", ".join(f"{k}={v}" for k, v in sorted(cnt.items(), key=lambda x: order[x[0]])))
        if not args.run:
            print("  (validación estática; usa --run para ejecutarlos con MIDIs sintéticos)")
    bad = sum(1 for r in results if r["status"] in ("crash", "invalido"))
    return 1 if (args.strict and bad) else 0


# ══════════════════════════════════════════════════════════════════════════════
#  Subcomando smoke
# ══════════════════════════════════════════════════════════════════════════════

def _help_job(job):
    tool, root, sub, timeout = job
    argv = [sys.executable, str(root / tool["file"])] + (sub.split() if sub else []) + ["--help"]
    st, rc, out = _run(argv, Path(tempfile_dir()), timeout)
    label = tool["name"] + (f" {sub}" if sub else "")
    rec = {"tool": label, "status": st, "detail": ""}
    if st == "ok":
        if not out.strip():
            rec.update(status="error", detail="--help no imprime nada")
        return rec
    if st == "timeout":
        rec["detail"] = "--help tarda más de %ss (¿importa módulos pesados?)" % timeout
        return rec
    mod = _missing_module(out)
    if mod:
        rec.update(status="dependencia", detail=f"falta el módulo '{mod}'")
    else:
        rec["detail"] = _last_line(out, 2)
    return rec


_TMP_DIR = None


def tempfile_dir():
    global _TMP_DIR
    if _TMP_DIR is None:
        import tempfile
        _TMP_DIR = tempfile.mkdtemp(prefix="si_smoke_")
    return _TMP_DIR


# ── recetas 'deep': flujos mínimos con asserts, derivados de bugs reales ──────
# Cada paso: args (con {tool} {corpus} {A} {B} {work} {other} {mid}), cwd (work|other),
# rc esperado, files (globs relativos al cwd), json [(fichero, clave, valor)], stdout_re.

DEEP_RECIPES = [
    {"tool": "musegan", "needs": ["torch", "numpy"],
     "why": "forward del discriminador con los datos por defecto (4 compases)",
     "steps": [
         {"args": ["prepare", "--input-dir", "{corpus}", "--output", "d.npz", "--bars", "4", "--resolution", "12"], "files": ["d.npz"]},
         {"args": ["train", "--data", "d.npz", "--model-dir", "m", "--steps", "2", "--batch-size", "8",
                   "--latent-dim", "64", "--n-dis", "2", "--cpu"], "files": ["m/checkpoint.pt"]},
         {"args": ["generate", "--model-dir", "m", "--output-dir", "g", "--n", "1", "--cpu"], "files": ["g/*"]},
     ]},
    {"tool": "remi_diffusion_v2", "needs": ["torch", "numpy"],
     "why": "un modelo entrenado con --model-dir relativo debe cargarse desde otro directorio",
     "steps": [
         {"args": ["train", "{corpus}", "--model-dir", "runs/m", "--small", "--epochs", "1", "--seed", "1"],
          "files": ["runs/m/model_config.json"], "timeout": 600},
         {"args": ["generate", "--model-dir", "{work}/runs/m", "--bars", "4", "--steps", "4", "--output", "g.mid"],
          "cwd": "other", "files": ["g.mid"]},
     ]},
    {"tool": "midinet_composer", "needs": ["torch", "numpy", "mido"],
     "why": "el rango --pitch-lo/--pitch-hi de prepare debe llegar al modelo",
     "steps": [
         {"args": ["prepare", "--input-dir", "{corpus}", "--output-dir", "data", "--pitch-lo", "36", "--pitch-hi", "84"]},
         {"args": ["train", "--data-dir", "data", "--model-dir", "model", "--epochs", "1", "--nz", "32",
                   "--base-ch", "32", "--cpu"], "json": [("model/meta.json", "pitch_lo", 36)], "timeout": 300},
     ]},
    {"tool": "cycle_gan_style_transfer_v3", "needs": ["numpy", "sklearn", "music21"],
     "why": "entrenamiento simbólico con valores por defecto (dimensión 61440) sin agotar la memoria",
     "steps": [
         {"args": ["train", "--domain-a", "{A}", "--domain-b", "{B}", "--model", "cg.pkl", "--iters", "2"],
          "files": ["cg.pkl"], "mem_gb": 6, "timeout": 300},
         {"args": ["transform", "{A}/f0.mid", "--model", "cg.pkl", "--output", "t.mid"], "files": ["t.mid"]},
     ]},
    {"tool": "chord_copilot", "needs": ["mido"],
     "why": "render debe poder usar el modelo entrenado con build-model",
     "steps": [
         {"args": ["build-model", "--corpus", "{corpus}", "--order", "2", "--out", "m.json"], "files": ["m.json"]},
         {"args": ["render", "--context", "Am F C", "--model", "m.json", "--out", "p.mid"], "files": ["p.mid"]},
     ]},
    {"tool": "mutator", "needs": ["mido"],
     "why": "el ejemplo de su propia cabecera ('urgencia') debe cambiar el tempo",
     "steps": [
         {"args": ["{mid}", "añade urgencia en el clímax", "--no-llm", "--output", "m.mid"],
          "files": ["m.mid"], "stdout_re": r"Tempo"},
     ]},
    {"tool": "theme_transformer", "needs": ["mido"],
     "why": "--out-dir debe recibir los ficheros de salida",
     "steps": [
         {"args": ["{mid}", "--style", "whole_tone", "--out-dir", "th/"], "files": ["th/*.mid"]},
     ]},
]


def _run_recipe(rec, root: Path, keep=False):
    import glob
    import importlib.util
    import tempfile
    res = {"tool": rec["tool"], "status": "ok", "detail": "", "why": rec["why"]}
    miss = [m for m in rec.get("needs", []) if importlib.util.find_spec(m) is None]
    if miss:
        res.update(status="omitido", detail="faltan módulos: " + ", ".join(miss))
        return res
    tool_path = root / (rec["tool"] + ".py")
    if not tool_path.exists():
        res.update(status="omitido", detail="no existe " + tool_path.name)
        return res
    base = Path(tempfile.mkdtemp(prefix=f"si_deep_{rec['tool']}_"))
    try:
        ctx = {"tool": str(tool_path), "corpus": str(_fixture_dir(base / "corpus", 8)),
               "A": str(_fixture_dir(base / "A", 4, 0)), "B": str(_fixture_dir(base / "B", 4, 5)),
               "work": str(base / "work"), "other": str(base / "other"),
               "mid": str(make_fixture_midi(base / "one.mid", 1))}
        (base / "work").mkdir()
        (base / "other").mkdir()
        for n, st in enumerate(rec["steps"], 1):
            argv = [sys.executable, str(tool_path)] + [a.format(**ctx) for a in st["args"]]
            cwd = Path(ctx[st.get("cwd", "work")])
            state, rc, out = _run(argv, cwd, st.get("timeout", 240), st.get("mem_gb"))
            tag = f"paso {n} ({st['args'][0] if not st['args'][0].startswith('{') else 'run'})"
            want = st.get("rc", 0)
            if state == "timeout":
                res.update(status="timeout", detail=f"{tag}: excede el tiempo"); return res
            if (rc or 0) != want:
                res.update(status="fallo", detail=f"{tag}: rc={rc}: " + _last_line(out, 2)); return res
            for g in st.get("files", []):
                if not glob.glob(str(cwd / g)):
                    res.update(status="fallo", detail=f"{tag}: no se generó {g}"); return res
            for fn, key, val in st.get("json", []):
                try:
                    got = json.loads((cwd / fn).read_text())[key]
                except Exception as e:
                    res.update(status="fallo", detail=f"{tag}: no se pudo leer {fn}[{key}]: {e}"); return res
                if got != val:
                    res.update(status="fallo", detail=f"{tag}: {fn}[{key}]={got!r}, esperado {val!r}"); return res
            if st.get("stdout_re") and not re.search(st["stdout_re"], out):
                res.update(status="fallo", detail=f"{tag}: la salida no contiene /{st['stdout_re']}/"); return res
        return res
    finally:
        if not keep:
            import shutil
            shutil.rmtree(base, ignore_errors=True)


def cmd_smoke(args, idx):
    root = Path(idx["dir"])
    results = []
    tempfile_dir()
    if not args.deep_only:
        tools = _select_tools(idx, args.tool)
        jobs = []
        for t in tools:
            jobs.append((t, root, None, args.timeout))
            if args.subs:
                for n, _h in t.get("subcommands", []):
                    jobs.append((t, root, n, args.timeout))
        results = _pool_map(_help_job, jobs, args.jobs)
    deep = []
    if args.deep or args.deep_only:
        recs = [r for r in DEEP_RECIPES if not args.tool or any(r["tool"] == (_find_tool(idx, n) or {}).get("name") for n in args.tool)]
        deep = [_run_recipe(r, root, args.keep) for r in recs]
    if args.json:
        print(json.dumps({"help": results, "deep": deep}, ensure_ascii=False, indent=1))
    else:
        order = {"crash": 0, "fallo": 0, "error": 1, "dependencia": 2, "timeout": 3, "omitido": 4, "ok": 5}
        if results:
            bad = [r for r in results if r["status"] != "ok"]
            for r in sorted(bad, key=lambda r: (order.get(r["status"], 9), r["tool"])):
                col = {"crash": "RED", "error": "RED", "dependencia": "YEL"}.get(r["status"], "G")
                print(f"  {_c(col)}{r['status']:<11}{_c('R')} {r['tool']:<34} {_short(r['detail'], 120)}")
            cnt = defaultdict(int)
            for r in results:
                cnt[r["status"]] += 1
            print(f"\n  --help: {len(results)} comprobaciones: " + ", ".join(f"{k}={v}" for k, v in sorted(cnt.items(), key=lambda x: order.get(x[0], 9))))
        if deep:
            print(f"\n{_c('B')}Flujos deep{_c('R')}")
            for r in deep:
                col = {"ok": "GRN", "fallo": "RED", "omitido": "G", "timeout": "YEL"}[r["status"]]
                print(f"  {_c(col)}{r['status']:<8}{_c('R')} {r['tool']:<30} {r['why']}")
                if r["detail"]:
                    print(f"           {_c('G')}{_short(r['detail'], 170)}{_c('R')}")
    bad = sum(1 for r in results if r["status"] in ("crash", "error")) + sum(1 for r in deep if r["status"] == "fallo")
    return 1 if (args.strict and bad) else 0


# ══════════════════════════════════════════════════════════════════════════════
#  Subcomando weights: dependencias externas (pesos, red, programas, módulos)
# ══════════════════════════════════════════════════════════════════════════════

_NET_PATTERNS = [
    (r"from_pretrained\s*\(", "descarga HuggingFace (from_pretrained)"),
    (r"hf_hub_download|snapshot_download", "descarga HuggingFace Hub"),
    (r"urlretrieve|urlopen\s*\(|requests\.(?:get|post)\s*\(|httpx\.", "peticiones HTTP"),
    (r"\b(?:HF_TOKEN|HUGGINGFACE_HUB_TOKEN)\b", "token HuggingFace"),
    (r"\b(?:ANTHROPIC_API_KEY|OPENAI_API_KEY|GOOGLE_API_KEY)\b", "clave de API"),
    (r"import anthropic|anthropic\.Anthropic", "API de Anthropic"),
]
_WEIGHT_EXT = (".onnx", ".safetensors", ".sf2", ".sf3", ".ckpt", ".tflite", ".h5")
_PROGRAMS = ("fluidsynth", "ffmpeg", "musescore", "mscore", "lilypond", "timidity", "sox", "ffprobe")


def _optional_imports(tree):
    """Módulos importados solo dentro de try/except (opcionales) y los obligatorios."""
    opt, req = set(), set()

    def walk(node, in_try):
        for ch in ast.iter_child_nodes(node):
            t = in_try or isinstance(ch, ast.Try)
            if isinstance(ch, ast.Import):
                for a in ch.names:
                    (opt if t else req).add(a.name.split(".")[0])
            elif isinstance(ch, ast.ImportFrom) and ch.module and ch.level == 0:
                (opt if t else req).add(ch.module.split(".")[0])
            walk(ch, t)

    walk(tree, False)
    return opt - req, req


def scan_external(path: Path, root: Path):
    import importlib.util
    import shutil
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(src)
    except SyntaxError:
        return None
    stdlib = set(getattr(sys, "stdlib_module_names", set()))
    local = {p.stem for p in root.glob("*.py")}
    opt, req = _optional_imports(tree)
    items = []
    for mod in sorted((opt | req) - stdlib - local - {"__future__"}):
        ok = importlib.util.find_spec(mod) is not None
        items.append({"kind": "módulo", "name": mod, "optional": mod in opt, "ok": ok})
    weights = defaultdict(set)
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) < 200 \
                and n.value.lower().endswith(_WEIGHT_EXT) and not re.search(r"[{}=\s]", n.value) \
                and not os.path.basename(n.value).startswith("."):
            weights[os.path.splitext(n.value)[1].lower()].add(n.value)
    for ext, ws in sorted(weights.items()):
        # varias rutas con la misma extensión son alternativas: basta con que exista una
        def _exists(w):
            cands = [Path(w)] if os.path.isabs(w) else [root / w, root / "models" / w, root / "weights" / w, Path.cwd() / w]
            return any(c.exists() for c in cands)
        ws = sorted(ws)
        items.append({"kind": "pesos", "name": " | ".join(ws), "optional": False, "ok": any(_exists(w) for w in ws)})
    for pat, label in _NET_PATTERNS:
        if re.search(pat, src):
            items.append({"kind": "red", "name": label, "optional": False, "ok": None})
    progs = set(re.findall(r"shutil\.which\(\s*['\"]([^'\"]+)['\"]", src))
    for p in _PROGRAMS:
        if re.search(r"subprocess\.\w+\(\s*\[\s*['\"]" + p + r"['\"]", src):
            progs.add(p)
    for p in sorted(progs):
        items.append({"kind": "programa", "name": p, "optional": False, "ok": shutil.which(p) is not None})
    return items


def cmd_weights(args, idx):
    root = Path(idx["dir"])
    rows = []
    for t in _select_tools(idx, args.tool):
        if t["file"] == Path(__file__).name:
            continue
        items = scan_external(root / t["file"], root)
        if items:
            rows.append({"tool": t["name"], "items": items})
    for r in rows:
        r["blocked"] = [i for i in r["items"] if i["ok"] is False and not i["optional"]]
        r["optional_missing"] = [i for i in r["items"] if i["ok"] is False and i["optional"]]
        r["network"] = [i for i in r["items"] if i["kind"] == "red"]
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0
    shown = [r for r in rows if args.all or r["blocked"] or (args.network and r["network"])]
    for r in sorted(shown, key=lambda r: (-len(r["blocked"]), r["tool"])):
        print(f"\n{_c('B')}{r['tool']}{_c('R')}")
        for i in r["items"]:
            if i["ok"] is True and not args.all:
                continue
            if i["ok"] is None:
                mark = f"{_c('YEL')}red  {_c('R')}"
            elif i["ok"]:
                mark = f"{_c('GRN')}ok   {_c('R')}"
            elif i["optional"]:
                mark = f"{_c('G')}opc. {_c('R')}"
            else:
                mark = f"{_c('RED')}falta{_c('R')}"
            print(f"  {mark} {i['kind']:<9} {i['name']}")
    blocked = [r for r in rows if r["blocked"]]
    net = [r for r in rows if r["network"]]
    print(f"\n  {len(idx['tools'])} herramientas · {len(blocked)} con algo obligatorio que falta aquí · "
          f"{len(net)} tocan red/tokens/APIs")
    if not args.all:
        print("  (usa --all para ver también lo que sí está disponible, --network para añadir las que usan red)")
    return 1 if (args.strict and blocked) else 0


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    ap = argparse.ArgumentParser(prog="suite_index.py",
                                 description="Catálogo y linter de las herramientas de la suite.", epilog='Códigos de salida (contrato de la suite): 0 = ejecución correcta, aunque haya hallazgos; 1 = error de ejecución o de entrada; 2 = uso incorrecto o entrada ilegible; 3 = hallazgos o puerta no superada (con --strict, o en los modos cuyo fin es validar).')
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default=None, help="Directorio de la suite (default: el de este script)")
    common.add_argument("--index", default=None, help="Caché del índice (default: <dir>/.suite_index.json)")
    common.add_argument("--rebuild", action="store_true", help="Ignorar la caché")
    common.add_argument("--json", action="store_true", help="Salida en JSON")
    common.add_argument("--no-color", action="store_true")
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("build", parents=[common], help="Escanea y escribe el índice JSON")
    p.add_argument("--out", default=None, help="Escribe además el índice (indentado) en este fichero")
    p = sp.add_parser("list", parents=[common], help="Lista las herramientas")
    p.add_argument("--category", choices=[c[0] for c in _CATEGORIES] + ["other"])
    p.add_argument("--grep", help="Filtra por texto en nombre/título/propósito")
    p = sp.add_parser("find", parents=[common], help="Busca por intención")
    p.add_argument("query", nargs="+")
    p.add_argument("--top", type=int, default=8)
    p = sp.add_parser("show", parents=[common], help="Ficha de una herramienta")
    p.add_argument("tool")
    p.add_argument("--flag", help="Comprueba si define este flag (rc=2 si no existe)")
    p = sp.add_parser("lint", parents=[common], help="Inconsistencias entre herramientas")
    p.add_argument("--check", nargs="+", help=f"Checks a ejecutar ({', '.join(ALL_CHECKS)})")
    p.add_argument("--severity", choices=["error", "warn", "info"], help="Severidad mínima a mostrar")
    p.add_argument("--strict", action="store_true", help="rc=1 si hay errores o avisos")
    sp.add_parser("stats", parents=[common], help="Resumen del ecosistema")
    p = sp.add_parser("examples", parents=[common], help="Valida (y con --run ejecuta) los ejemplos de las cabeceras")
    p.add_argument("tool", nargs="*", help="Herramientas a revisar (default: todas)")
    p.add_argument("--run", action="store_true", help="Ejecuta los ejemplos en un directorio temporal con MIDIs sintéticos")
    p.add_argument("--all", action="store_true", help="Muestra también los ejemplos correctos")
    p.add_argument("--timeout", type=int, default=20, help="Segundos por ejemplo con --run (default: 20)")
    p.add_argument("--jobs", type=int, default=4, help="Ejemplos en paralelo con --run (default: 4)")
    p.add_argument("--strict", action="store_true", help="rc=1 si hay ejemplos inválidos o que revientan")
    p = sp.add_parser("smoke", parents=[common], help="Humo: --help de cada herramienta y flujos mínimos (--deep)")
    p.add_argument("tool", nargs="*", help="Herramientas a revisar (default: todas)")
    p.add_argument("--subs", action="store_true", help="Prueba también `<sub> --help` de cada subcomando")
    p.add_argument("--deep", action="store_true", help="Añade los flujos mínimos con asserts (entrenan 1-2 épocas)")
    p.add_argument("--deep-only", action="store_true", dest="deep_only", help="Solo los flujos deep")
    p.add_argument("--keep", action="store_true", help="Conserva los directorios temporales de los flujos deep")
    p.add_argument("--timeout", type=int, default=30, help="Segundos por --help (default: 30)")
    p.add_argument("--jobs", type=int, default=4, help="Comprobaciones en paralelo (default: 4)")
    p.add_argument("--strict", action="store_true", help="rc=1 si algo revienta o falla")
    p = sp.add_parser("weights", parents=[common], help="Pesos, módulos, programas y red que necesita cada herramienta")
    p.add_argument("tool", nargs="*", help="Herramientas a revisar (default: todas)")
    p.add_argument("--all", action="store_true", help="Muestra también lo que sí está disponible")
    p.add_argument("--network", action="store_true", help="Incluye las herramientas que usan red/tokens/APIs")
    p.add_argument("--strict", action="store_true", help="rc=1 si falta algo obligatorio")
    return ap


def main(argv=None):
    global _USE_COLOR
    args = build_parser().parse_args(argv)
    if args.no_color:
        _USE_COLOR = False
    root = Path(args.dir).resolve() if args.dir else Path(__file__).resolve().parent
    if not root.is_dir():
        print(f"[ERROR] no existe el directorio {root}", file=sys.stderr)
        return 1
    index_path = Path(args.index) if args.index else root / ".suite_index.json"
    idx = load_index(root, index_path, rebuild=args.rebuild)
    fn = {"build": cmd_build, "list": cmd_list, "find": cmd_find,
          "show": cmd_show, "lint": cmd_lint, "stats": cmd_stats,
          "examples": cmd_examples, "smoke": cmd_smoke, "weights": cmd_weights}[args.cmd]
    try:
        return fn(args, idx)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
