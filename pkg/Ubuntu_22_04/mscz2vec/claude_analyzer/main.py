#!/usr/bin/env python3
"""
main.py — punto de entrada CLI para midi_analyzer v12.0
════════════════════════════════════════════════════════

Salidas independientes, activables por separado o juntas:

  --essay-output  pieza.txt    → ensayo narrativo (solo texto)
  --html-output   pieza.html   → informe estadístico (tablas + gráficas)
  --yaml-output   pieza.yaml   → exportación estructurada (machine-readable)
  --output        informe.txt  → informe completo de métricas (texto plano)
  --essay-only                 → solo el ensayo, sin métricas (excluyente)

Si no se especifica ninguna salida, imprime el ensayo por stdout.

Uso:
    python main.py pieza.mid
    python main.py pieza.mid --essay-output analisis.txt
    python main.py pieza.mid --html-output  informe.html
    python main.py pieza.mid --yaml-output  datos.yaml
    python main.py pieza.mid --output informe.txt
    python main.py pieza.mid --print-report
    python main.py pieza.mid --essay-only --essay-output ensayo.txt
    python main.py pieza.mid --essay-output analisis.txt \\
                              --html-output  informe.html \\
                              --yaml-output  datos.yaml
    python main.py pieza.mid --all-outputs pieza
    python main.py pieza.mid --sections 8 --essay-output analisis.txt
"""

import sys
import os
import argparse

# ── Imports ──────────────────────────────────────────────────────
# Funciona como paquete (python -m claude_analyzer.main) y como script
# (python claude_analyzer/main.py), sea cual sea el nombre de la carpeta.
try:
    # Como paquete:  python -m claude_analyzer.main   /   import claude_analyzer.main
    from .core import run_analysis, build_text_report
    from .renderers.essay import render_essay
    from .renderers.html_report import render_html
    from .renderers.yaml_export import render_yaml
except ImportError:
    # Como script:  python claude_analyzer/main.py  (core y renderers desde su carpeta)
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from core import run_analysis, build_text_report
    from renderers.essay import render_essay
    from renderers.html_report import render_html
    from renderers.yaml_export import render_yaml


# ── CLI ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='midi_analyzer',
        description=(
            'Analizador musical y emocional avanzado v12.0\n'
            'Genera hasta tres salidas diferenciadas desde un archivo MIDI.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        'filepath',
        help='Ruta al archivo .mid o .midi',
    )
    p.add_argument(
        '--sections', type=int, default=6,
        help='Número de secciones para el análisis estático (default: 6)',
    )

    # ── Output targets ──────────────────────────────────────────
    out = p.add_argument_group('salidas')
    out.add_argument(
        '--essay-output', metavar='ARCHIVO.txt',
        help='Guardar ensayo narrativo en un archivo .txt',
    )
    out.add_argument(
        '--html-output', metavar='ARCHIVO.html',
        help='Guardar informe estadístico interactivo en un archivo .html',
    )
    out.add_argument(
        '--yaml-output', metavar='ARCHIVO.yaml',
        help='Guardar exportación estructurada en un archivo .yaml',
    )
    out.add_argument(
        '--all-outputs', metavar='BASE',
        help=(
            'Generar las tres salidas con el mismo nombre base. '
            'Ejemplo: --all-outputs mi_pieza genera '
            'mi_pieza.txt, mi_pieza.html, mi_pieza.yaml'
        ),
    )

    # ── Stdout options ──────────────────────────────────────────
    out.add_argument(
        '--print-essay', action='store_true',
        help='Imprimir ensayo por stdout (además de guardar si se especifica)',
    )
    out.add_argument(
        '--print-yaml', action='store_true',
        help='Imprimir YAML por stdout',
    )
    out.add_argument(
        '--output', metavar='ARCHIVO.txt',
        help='Guardar el informe completo de métricas en texto plano '
             '(equivale al --output de claude_analyzer.py)',
    )
    out.add_argument(
        '--print-report', action='store_true',
        help='Imprimir el informe completo de métricas por stdout',
    )
    out.add_argument(
        '--essay-only', action='store_true',
        help='Solo el ensayo narrativo, sin métricas: se imprime por stdout y, '
             'con --essay-output, también se guarda. Incompatible con '
             '--output, --print-report, --html-output, --yaml-output, '
             '--print-yaml y --all-outputs',
    )

    return p


def _save(path: str, content: str, label: str) -> None:
    try:
        _d = os.path.dirname(path)
        if _d:
            os.makedirs(_d, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        size_kb = len(content.encode('utf-8')) / 1024
        print(f"  ✅  {label} guardado en: {path}  ({size_kb:.1f} KB)")
    except OSError as e:
        print(f"  ❌  Error guardando {label}: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # ── --essay-only es excluyente ──────────────────────────────
    if args.essay_only:
        _conf = [n for n, v in (
            ('--output', args.output), ('--print-report', args.print_report),
            ('--html-output', args.html_output), ('--yaml-output', args.yaml_output),
            ('--print-yaml', args.print_yaml), ('--all-outputs', args.all_outputs),
        ) if v]
        if _conf:
            parser.error('--essay-only es incompatible con: ' + ', '.join(_conf))

    # ── Expand --all-outputs ─────────────────────────────────────
    if args.all_outputs:
        base = args.all_outputs
        if not args.essay_output:
            args.essay_output = f"{base}.txt"
        if not args.html_output:
            args.html_output  = f"{base}.html"
        if not args.yaml_output:
            args.yaml_output  = f"{base}.yaml"

    # ── Check at least one output is requested ───────────────────
    any_file_output = any([
        args.essay_output, args.html_output, args.yaml_output,
        args.output,
    ])
    any_stdout = (args.print_essay or args.print_yaml or args.print_report
                  or args.essay_only)

    # Default: print essay to stdout if nothing else requested
    default_stdout = not any_file_output and not any_stdout

    # ── Check input file ─────────────────────────────────────────
    if not os.path.isfile(args.filepath):
        print(f"❌  Archivo no encontrado: {args.filepath}", file=sys.stderr)
        sys.exit(1)

    # ── Run analysis ─────────────────────────────────────────────
    print(f"\n🎵  Analizando: {os.path.basename(args.filepath)}")
    print(f"    Secciones: {args.sections}\n")

    results = run_analysis(args.filepath, n_sections=args.sections)

    if not results or not isinstance(results, dict):
        print("⚠️  El análisis no devolvió resultados.", file=sys.stderr)
        sys.exit(1)

    # ── Determine which renderers to run ─────────────────────────
    need_essay = bool(args.essay_output or args.print_essay or default_stdout
                      or args.essay_only)
    need_html  = bool(args.html_output)
    need_yaml  = bool(args.yaml_output or args.print_yaml)
    need_report = bool(args.output or args.print_report)

    print()

    # ── Essay renderer ────────────────────────────────────────────
    if need_essay:
        print("  📝  Generando ensayo narrativo...")
        essay = render_essay(results)
        if args.essay_output:
            _save(args.essay_output, essay, "ensayo narrativo")
        if args.print_essay or args.essay_only or default_stdout:
            print()
            print(essay)

    # ── Informe completo de métricas (texto plano) ──────────────
    if need_report:
        print("  📑  Generando informe completo de métricas...")
        report = build_text_report(results)
        if args.output:
            _save(args.output, report, "informe de métricas")
        if args.print_report:
            print()
            print(report)

    # ── HTML renderer ─────────────────────────────────────────────
    if need_html:
        print("  📊  Generando informe HTML...")
        html = render_html(results)
        _save(args.html_output, html, "informe HTML")

    # ── YAML renderer ─────────────────────────────────────────────
    if need_yaml:
        print("  📋  Generando exportación YAML...")
        yaml_str = render_yaml(results)
        if args.yaml_output:
            _save(args.yaml_output, yaml_str, "exportación YAML")
        if args.print_yaml:
            print()
            print(yaml_str)

    print()


if __name__ == '__main__':
    main()
