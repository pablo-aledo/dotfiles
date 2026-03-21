# midi_analyzer/__init__.py
"""
midi_analyzer v12.0
════════════════════
Analizador musical y emocional avanzado para archivos MIDI.

Tres salidas diferenciadas:
  - Ensayo narrativo (.txt)  — prosa interpretativa
  - Informe estadístico (.html) — tablas y gráficas interactivas
  - Exportación estructurada (.yaml) — machine-readable

Uso básico:
    from midi_analyzer.core import run_analysis
    from midi_analyzer.renderers.essay import render_essay
    from midi_analyzer.renderers.html_report import render_html
    from midi_analyzer.renderers.yaml_export import render_yaml

    results = run_analysis("pieza.mid")
    essay = render_essay(results)
    html  = render_html(results)
    yaml  = render_yaml(results)
"""
