#!/usr/bin/env python3
"""
main.py — punto de entrada CLI para midi_analyzer v12.0
════════════════════════════════════════════════════════

Tres salidas independientes, activables por separado o juntas:

  --essay-output  pieza.txt    → ensayo narrativo (solo texto)
  --html-output   pieza.html   → informe estadístico (tablas + gráficas)
  --yaml-output   pieza.yaml   → exportación estructurada (machine-readable)

Si no se especifica ninguna salida, imprime el ensayo por stdout.

Uso:
    python main.py pieza.mid
    python main.py pieza.mid --essay-output analisis.txt
    python main.py pieza.mid --html-output  informe.html
    python main.py pieza.mid --yaml-output  datos.yaml
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
# Permitir ejecución tanto desde la carpeta midi_analyzer/
# como desde su directorio padre.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)
if os.path.dirname(_here) not in sys.path:
    sys.path.insert(0, os.path.dirname(_here))

try:
    from midi_analyzer.core import run_analysis
    from midi_analyzer.renderers.essay import render_essay
    from midi_analyzer.renderers.html_report import render_html
    from midi_analyzer.renderers.yaml_export import render_yaml
except ImportError:
    # Fallback: mismo directorio
    from core import run_analysis
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

    return p


def _save(path: str, content: str, label: str) -> None:
    try:
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
    ])
    any_stdout = args.print_essay or args.print_yaml

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
    need_essay = bool(args.essay_output or args.print_essay or default_stdout)
    need_html  = bool(args.html_output)
    need_yaml  = bool(args.yaml_output or args.print_yaml)

    print()

    # ── Essay renderer ────────────────────────────────────────────
    if need_essay:
        print("  📝  Generando ensayo narrativo...")
        essay = render_essay(results)
        if args.essay_output:
            _save(args.essay_output, essay, "ensayo narrativo")
        if args.print_essay or default_stdout:
            print()
            print(essay)

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
