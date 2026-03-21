# midi_analyzer/renderers/__init__.py
"""
Tres renderers independientes que consumen el dict de run_analysis().
Cada uno tiene su propio propósito y audiencia:

  essay.py       → prosa narrativa (.txt)       — lector humano
  html_report.py → tablas y gráficas (.html)    — analista / visualización
  yaml_export.py → schema estructurado (.yaml)  — otro programa / pipeline
"""
