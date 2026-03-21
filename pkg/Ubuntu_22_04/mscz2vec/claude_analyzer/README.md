# midi_analyzer v12.0

Analizador musical y emocional avanzado para archivos MIDI.

## Instalación

```bash
pip install mido
```

## Uso

```bash
# Las tres salidas de una vez (recomendado)
python main.py pieza.mid --all-outputs pieza

# O individualmente:
python main.py pieza.mid --essay-output analisis.txt
python main.py pieza.mid --html-output  informe.html
python main.py pieza.mid --yaml-output  datos.yaml

# Solo por stdout (ensayo):
python main.py pieza.mid

# Con más secciones de análisis:
python main.py pieza.mid --sections 8 --all-outputs pieza
```

## Estructura

```
midi_analyzer/
├── main.py              ← punto de entrada CLI
├── core.py              ← motor de análisis (~81 análisis)
├── requirements.txt
└── renderers/
    ├── essay.py         ← ensayo narrativo (.txt)
    ├── html_report.py   ← informe estadístico (.html)
    └── yaml_export.py   ← exportación estructurada (.yaml)
```

## Salidas

| Archivo | Contenido | Audiencia |
|---------|-----------|-----------|
| `.txt`  | Ensayo narrativo en 10 secciones, prosa libre | Lector humano |
| `.html` | Gráficas interactivas (Chart.js), tablas, SSM, métricas | Analista / visualización |
| `.yaml` | Schema estructurado tipado, curvas como arrays | Pipeline / otro programa |

## Requisitos

- Python 3.9+
- `mido` (única dependencia externa)
