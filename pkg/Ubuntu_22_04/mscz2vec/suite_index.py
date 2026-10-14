#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          SUITE INDEX  v1.0                                   ║
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
║                                                                              ║
║  USO:                                                                        ║
║    python suite_index.py build [--dir DIR] [--out suite_index.json]          ║
║    python suite_index.py list [--category harmony] [--grep texto]            ║
║    python suite_index.py find rearmonizar acordes [--top 8]                  ║
║    python suite_index.py show drummer [--json]                               ║
║    python suite_index.py show drummer --flag style                           ║
║    python suite_index.py lint [--check output-flag style-vocab] [--json]     ║
║    python suite_index.py stats                                               ║
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
║    doc-vs-code   Flags citados en la cabecera que el argparse no define      ║
║    undocumented  Flags definidos que la cabecera no menciona                 ║
║    versions      Familias de versiones (x, x_v2, x_v3 ...) y su última       ║
║    syntax        Ficheros que no se pueden analizar                          ║
║                                                                              ║
║  LIMITACIONES: análisis estático. Los flags creados dinámicamente (bucles,   ║
║  sys.argv a mano) o con choices calculadas en tiempo de ejecución no se      ║
║  ven; en ese caso la ficha lo indica. Las categorías se infieren por         ║
║  palabras clave y son orientativas.                                          ║
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

VERSION = "1.0"
INDEX_SCHEMA = 1

_USE_COLOR = sys.stdout.isatty()


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
            if attr == "add_parser" and n.value.args:
                name = _lit(n.value.args[0], env)
                if isinstance(name, str):
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
                    if isinstance(name, str) and name not in sub_help:
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
            index_path.write_text(json.dumps(idx, ensure_ascii=False), encoding="utf-8")
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
        out.write_text(json.dumps(idx, ensure_ascii=False, indent=1), encoding="utf-8")
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
            for p in t.get("exit_patterns", []):
                if p.startswith("rc=1 si hay hallazgos"):
                    add("exit-findings", t["name"], "info",
                        "devuelve rc=1 cuando encuentra hallazgos (no solo en errores): en pipelines con "
                        "`set -e` o runner --strict la herramienta 'falla' haciendo su trabajo; "
                        "conviene documentarlo en la cabecera")

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

    sev_order = {"error": 0, "warn": 1, "info": 2}
    findings.sort(key=lambda f: (sev_order[f["severity"]], f["check"], f["tool"]))
    return findings


ALL_CHECKS = ["syntax", "output-flag", "style-vocab", "exit-findings", "doc-vs-code",
              "undocumented", "versions"]


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
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    ap = argparse.ArgumentParser(prog="suite_index.py",
                                 description="Catálogo y linter de las herramientas de la suite.")
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
          "show": cmd_show, "lint": cmd_lint, "stats": cmd_stats}[args.cmd]
    try:
        return fn(args, idx)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
