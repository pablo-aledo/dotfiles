#!/usr/bin/env python3
"""organize_versions.py — Fase 1 de la limpieza de versiones (mutopia).

Qué hace (en este orden):
  1. Mueve las versiones superadas a  old/  (no borra nada).
  2. Renombra la versión vigente de cada herramienta quitándole el sufijo _vN.
  3. Repara las referencias cruzadas en los scripts que se quedan (imports,
     prog=, usos en docstrings, registro de pruebas de suite_index.py...).
  4. Verifica la sintaxis y que ningún script importe un módulo que ya no existe.
  5. Escribe  old/MOVED.tsv  (qué fue adónde)  y  refs_report.txt  (cambios de
     referencias y líneas que conviene revisar a mano).

Por defecto es una SIMULACIÓN: no toca nada. Añade --apply para ejecutar.
Antes de aplicar hace una copia de seguridad .tgz de todos los .py.

Uso:
    python organize_versions.py [DIR]                # simulación
    python organize_versions.py [DIR] --apply        # ejecutar
    python organize_versions.py [DIR] --apply --no-backup

Solo usa la biblioteca estándar.
"""
import argparse
import ast
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Plan de la fase 1.
#   new       nombre final (sin .py)
#   keep      nombre actual de la versión que se queda
#   base_old  nombre que tendrá en old/ el fichero base sin sufijo (None = no hay)
#   olds      versiones con sufijo que se mueven a old/ con su nombre actual
# Quedan FUERA (decisiones pendientes): melody_conditioned, diffusion_composer,
# combined_simplifier*.
# ─────────────────────────────────────────────────────────────────────────────
FAMILIES = [
    dict(new="fingering", keep="fingering_v6", base_old="fingering_v1",
         olds=["fingering_v2", "fingering_v3", "fingering_v4", "fingering_v5"]),
    dict(new="jev_composer", keep="jev_composer_v4", base_old="jev_composer_v1",
         olds=["jev_composer_v2", "jev_composer_v3"]),
    dict(new="piano_duo_analyzer", keep="piano_duo_analyzer_v5",
         base_old="piano_duo_analyzer_v1",
         olds=["piano_duo_analyzer_v3", "piano_duo_analyzer_v4"]),
    dict(new="unifier", keep="unifier_v3", base_old="unifier_v1",
         olds=["unifier_v2"]),
    dict(new="grammar_composer", keep="grammar_composer_v2",
         base_old="grammar_composer_v1", olds=[]),
    dict(new="grammar_transformer", keep="grammar_transformer_v3",
         base_old="grammar_transformer_v1", olds=[]),
    dict(new="genopatch", keep="genopatch_v2", base_old="genopatch_v1", olds=[]),
    dict(new="remi_diffusion", keep="remi_diffusion_v2",
         base_old="remi_diffusion_v1", olds=[]),
    # El docstring del base de cycle_gan dice «versión 2.0» -> se archiva como _v2.
    dict(new="cycle_gan_style_transfer", keep="cycle_gan_style_transfer_v3",
         base_old="cycle_gan_style_transfer_v2", olds=[]),
    # Solo existe la v3: únicamente se renombra.
    dict(new="genetic_style_transfer", keep="genetic_style_transfer_v3",
         base_old=None, olds=[]),
]

# Sustituciones literales previas (nombres de funciones y textos que la regex
# general no resolvería bien). fichero_final -> [(antes, despues)]
EXTRA = {
    "genopatch_flow.py": [("_import_genopatch_v2", "_import_genopatch")],
    "suite_index.py": [("(fingering_v3 ... v6)",
                        "(las distintas versiones de fingering)")],
}

OLD_DIR = "old"


def build_tables():
    tok = {}                       # token versionado -> (familia, 'keep'|'old')
    for fam in FAMILIES:
        tok[fam["keep"]] = (fam, "keep")
        for o in fam["olds"]:
            tok[o] = (fam, "old")
    return tok


TOK = build_tables()
TOK_RE = re.compile(r"\b(" + "|".join(sorted(TOK, key=len, reverse=True)) + r")\b")
BASE_RE = re.compile(
    r"\b(" + "|".join(f["new"] for f in FAMILIES) + r")\.py\b")
RESIDUAL_RE = re.compile(r"(?<![A-Za-z0-9])v[1-9]\b")
# Una cita al fichero base solo se marca si la línea habla de versiones
VERSION_HINT_RE = re.compile(r"(?i)(?<![A-Za-z0-9])v\d|_v\d|versi[oó]n|versiones|anterior|previa|\bbase\b")


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def read(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return fh.read()


def write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


# ─────────────────────────────────────────────────────────────────────────────
def plan(root: Path):
    """Devuelve (moves_old, renames, errores)."""
    errors = []
    moves_old = []     # (src, dst en old/)
    renames = []       # (src, dst)
    names = {p.name for p in root.glob("*.py")}
    old = root / OLD_DIR
    for fam in FAMILIES:
        keep_f, new_f = fam["keep"] + ".py", fam["new"] + ".py"
        if keep_f not in names:
            errors.append(f"falta {keep_f}")
        renames.append((root / keep_f, root / new_f))
        for o in fam["olds"]:
            if o + ".py" not in names:
                errors.append(f"falta {o}.py")
            moves_old.append((root / (o + ".py"), old / (o + ".py")))
        if fam["base_old"]:
            if new_f not in names:
                errors.append(f"falta {new_f} (base sin sufijo)")
            moves_old.append((root / new_f, old / (fam["base_old"] + ".py")))
        elif new_f in names:
            errors.append(f"{new_f} ya existe y chocaría con el renombrado")
    for _, dst in moves_old:
        if dst.exists():
            errors.append(f"ya existe {dst.relative_to(root)}")
    return moves_old, renames, errors


def rewrite_text(text, final_name, original_name):
    """Aplica las sustituciones a un fichero. Devuelve (texto, eventos, revisar)."""
    final_stem = final_name[:-3]
    own = next((f for f in FAMILIES if f["new"] == final_stem), None)
    events, review = [], []
    out_lines = []
    extras = EXTRA.get(final_name, [])
    extras_hit = {a: False for a, _ in extras}

    for lineno, line in enumerate(text.splitlines(keepends=True), 1):
        orig = line
        for a, b in extras:
            if a in line:
                line = line.replace(a, b)
                extras_hit[a] = True

        n_sub = 0
        flagged = []

        def cb(m):
            nonlocal n_sub
            token = m.group(1)
            fam, role = TOK[token]
            n_sub += 1
            if role == "keep":
                return fam["new"]                       # su propio nuevo nombre
            if own is fam:                              # versión vieja citada en la vigente
                if m.string[m.end():m.end() + 3] == ".py":
                    flagged.append(f"cita {token}.py -> ahora en old/")
                    return f"{OLD_DIR}/{token}"
                flagged.append(f"cita {token} (versión archivada en old/)")
                return token
            flagged.append(f"cita la versión {token} -> ahora {fam['new']}")
            return fam["new"]                           # otro fichero: apunta al vigente

        line = TOK_RE.sub(cb, line)

        # En el fichero vigente, citas del fichero base sin sufijo (ahora en old/)
        if own is not None and original_name == own["keep"] + ".py":
            for m in BASE_RE.finditer(orig):
                if (m.group(1) == own["new"] and own["base_old"]
                        and VERSION_HINT_RE.search(orig)):
                    flagged.append(
                        f"cita {own['new']}.py: ¿versión base (ahora old/{own['base_old']}.py) "
                        f"o la herramienta en general?")
                    break

        if line != orig:
            events.append((lineno, orig.rstrip("\n"), line.rstrip("\n")))
        why = list(flagged)
        if flagged:                      # solo se revisan líneas con citas a otras versiones
            if n_sub >= 2 and line != orig:
                why.append("varias sustituciones en la misma línea")
            if RESIDUAL_RE.search(line):
                why.append("queda un «vN» suelto en la línea")
        if why:
            review.append((lineno, why, orig.rstrip("\n"), line.rstrip("\n")))
        out_lines.append(line)

    for a, hit in extras_hit.items():
        if not hit:
            print(f"AVISO: no se encontró «{a}» en {final_name}", file=sys.stderr)
    return "".join(out_lines), events, review


def short(s, n=150):
    s = s.strip()
    return s if len(s) <= n else s[:n - 1] + "…"


def main():
    ap = argparse.ArgumentParser(description="Fase 1: archivar versiones viejas en old/, "
                                             "renombrar las vigentes y reparar referencias.")
    ap.add_argument("dir", nargs="?", default=".", help="carpeta con los scripts (defecto: .)")
    ap.add_argument("--apply", action="store_true", help="ejecutar (sin esto, solo simula)")
    ap.add_argument("--no-backup", action="store_true", help="no crear copia de seguridad .tgz")
    ap.add_argument("--report", default="refs_report.txt", help="fichero del informe de referencias")
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        die(f"{root} no es una carpeta")

    moves_old, renames, errors = plan(root)
    if errors:
        already = (root / OLD_DIR / "MOVED.tsv").exists()
        for e in errors:
            print(f"  · {e}")
        die("comprobaciones previas fallidas" +
            (" (¿ya se aplicó antes? existe old/MOVED.tsv)" if already else ""))

    moved_src = {s for s, _ in moves_old}
    rename_map = {s: d for s, d in renames}
    self_path = Path(__file__).resolve()      # este script no debe reescribirse a sí mismo

    # 1) Reescribir referencias EN MEMORIA, sobre el conjunto de ficheros que se quedan
    texts, all_events, all_review = {}, {}, {}
    for p in sorted(root.glob("*.py")):
        if p in moved_src or p.resolve() == self_path:
            continue
        # el base de cada familia se mueve (está en moved_src); el resto se procesa
        final = rename_map.get(p, p)
        new_text, events, review = rewrite_text(read(p), final.name, p.name)
        texts[final] = new_text
        if events:
            all_events[final.name] = events
        if review:
            all_review[final.name] = review

    # 2) Verificación previa de sintaxis y de imports huérfanos (sobre los textos nuevos)
    removed_stems = {s.stem for s in moved_src} | {s.stem for s in rename_map}
    # un stem sigue existiendo si es el nombre final de alguna renombrada
    still_exist = {d.stem for d in rename_map.values()}
    orphan_stems = removed_stems - still_exist
    problems = []
    for dst, txt in texts.items():
        try:
            tree = ast.parse(txt)
        except SyntaxError as e:
            problems.append(f"{dst.name}: error de sintaxis tras reescribir ({e})")
            continue
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module.split(".")[0]]
            for m in mods:
                if m in orphan_stems:
                    problems.append(f"{dst.name}:{node.lineno}: importa «{m}», que ya no existirá")
    if problems:
        for pr in problems:
            print("  ·", pr)
        die("la verificación previa ha fallado; no se toca nada")

    # 3) Informe
    lines = []
    lines.append(f"{'APLICADO' if args.apply else 'SIMULACIÓN'} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"== Se mueven a {OLD_DIR}/ ({len(moves_old)})")
    for s, d in moves_old:
        lines.append(f"   {s.name:42s} -> {d.relative_to(root)}")
    lines.append("")
    lines.append(f"== Se renombran ({len(renames)})")
    for s, d in renames:
        lines.append(f"   {s.name:42s} -> {d.name}")
    lines.append("")
    n_ev = sum(len(v) for v in all_events.values())
    lines.append(f"== Líneas con referencias reescritas: {n_ev} en {len(all_events)} ficheros")
    for f, evs in sorted(all_events.items()):
        lines.append(f"   {f}: {len(evs)}")
    lines.append("")
    n_rev = sum(len(v) for v in all_review.values())
    lines.append(f"== REVISAR A MANO ({n_rev} líneas)")
    for f, rv in sorted(all_review.items()):
        for lineno, why, before, after in rv:
            lines.append(f"-- {f}:{lineno}: {'; '.join(why)}")
            lines.append(f"     antes: {short(before)}")
            if before != after:
                lines.append(f"     ahora: {short(after)}")
    report = "\n".join(lines) + "\n"
    print(report)

    if not args.apply:
        print("Simulación terminada. Nada ha cambiado. Usa --apply para ejecutar.")
        return

    # 4) Copia de seguridad
    if not args.no_backup:
        ts = time.strftime("%y%m%d-%H%M%S")
        bk = root.parent / f"{root.name}_pre_fase1_{ts}.tgz"
        try:
            with tarfile.open(bk, "w:gz") as tf:
                for p in sorted(root.glob("*.py")):
                    tf.add(p, arcname=p.name)
        except OSError as e:
            die(f"no pude crear la copia de seguridad en {bk} ({e}); usa --no-backup si no la quieres")
        print(f"Copia de seguridad: {bk}")

    # 5) Mover a old/, renombrar y escribir los textos nuevos
    (root / OLD_DIR).mkdir(exist_ok=True)
    tsv = ["original\tdestino\trol"]
    for s, d in moves_old:
        shutil.move(str(s), str(d))
        tsv.append(f"{s.name}\t{d.relative_to(root)}\tarchivado")
    for s, d in renames:
        shutil.move(str(s), str(d))
        tsv.append(f"{s.name}\t{d.name}\trenombrado")
    for dst, txt in texts.items():
        if read(dst) != txt:
            write(dst, txt)
    write(root / OLD_DIR / "MOVED.tsv", "\n".join(tsv) + "\n")
    write(root / args.report, report)
    print(f"Hecho. Informe: {root / args.report}  |  Mapa: {root / OLD_DIR / 'MOVED.tsv'}")
    print("Para ejecutar un script archivado: PYTHONPATH=. python old/<nombre>.py ...")


if __name__ == "__main__":
    main()
