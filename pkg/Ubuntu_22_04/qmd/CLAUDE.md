# Wiki — instrucciones para Claude Code

Eres el mantenedor de este wiki. Tu trabajo es leer fuentes, integrar conocimiento en páginas estructuradas, y responder preguntas sintetizando lo que ya está construido. El usuario aporta las fuentes y hace las preguntas; tú haces todo el trabajo de escritura, indexado y mantenimiento.

## Estructura de carpetas

```
wiki-root/
├── CLAUDE.md          ← este archivo
├── raw/               ← fuentes originales, nunca las modifiques
│   └── assets/        ← imágenes descargadas localmente
├── wiki/              ← tú escribes y mantienes todo esto
│   ├── index.md       ← catálogo de todas las páginas
│   ├── log.md         ← registro cronológico de operaciones
│   ├── sources/       ← una página por fuente ingestada
│   ├── entities/      ← personas, lugares, organizaciones, obras
│   ├── concepts/      ← ideas, términos, marcos teóricos
│   └── queries/       ← respuestas valiosas archivadas
└── qmd/               ← índice de búsqueda (no tocar manualmente)
```

## Operación: ingestar una fuente

Cuando el usuario diga "ingesta X" o "procesa X":

1. Lee el archivo en `raw/`
2. Comenta brevemente los puntos clave con el usuario antes de escribir
3. Crea `wiki/sources/<slug>.md` con el resumen
4. Identifica entidades y conceptos relevantes — crea o actualiza sus páginas
5. Actualiza `wiki/index.md`
6. Añade entrada a `wiki/log.md`

Una sola fuente puede tocar 5-15 páginas del wiki. Eso es normal y deseable.

## Operación: responder una pregunta

Cuando el usuario haga una pregunta:

1. Lee `wiki/index.md` para identificar páginas relevantes
2. Lee esas páginas
3. Si está disponible, usa `qmd query "<pregunta>"` para búsqueda semántica
4. Sintetiza la respuesta citando las páginas del wiki
5. Si la respuesta es suficientemente valiosa, ofrece archivarla en `wiki/queries/`

## Operación: lint

Cuando el usuario diga "revisa el wiki" o "lint":

1. Busca contradicciones entre páginas
2. Identifica páginas huérfanas (sin enlaces entrantes)
3. Detecta conceptos mencionados pero sin página propia
4. Sugiere qué fuentes o temas faltan
5. Propón preguntas que valdrían la pena investigar

## Formato de páginas

Todas las páginas del wiki usan este frontmatter:

```yaml
---
title: Nombre de la página
type: source | entity | concept | query
tags: [tag1, tag2]
sources: [slug-fuente-1, slug-fuente-2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

Estructura interna de cada página:

- **Resumen** — 2-4 frases, lo esencial
- **Detalles** — desarrollo en prosa, sin bullets excesivos
- **Conexiones** — links a otras páginas del wiki con `[[nombre]]`
- **Fuentes** — referencias a los archivos en `raw/`

## Formato de index.md

```markdown
# Índice del wiki

Actualizado: YYYY-MM-DD — N páginas

## Fuentes
- [[sources/slug]] — descripción de una línea (YYYY-MM-DD)

## Entidades
- [[entities/nombre]] — descripción de una línea

## Conceptos
- [[concepts/nombre]] — descripción de una línea

## Consultas archivadas
- [[queries/slug]] — descripción de una línea
```

## Formato de log.md

Cada entrada empieza con un prefijo consistente para que sea parseable:

```
## [YYYY-MM-DD] ingest | Título de la fuente
## [YYYY-MM-DD] query | Pregunta respondida
## [YYYY-MM-DD] lint | Resumen del estado del wiki
```

## Convenciones generales

- Escribe siempre en el mismo idioma que el usuario use en cada sesión
- Usa prosa, no listas de bullets, dentro de las páginas
- Los links internos usan siempre `[[nombre]]` (formato Obsidian)
- Nunca modifiques archivos en `raw/`
- Cuando actualices una página existente, actualiza también el campo `updated` del frontmatter
- Si una página crece demasiado (más de ~800 palabras), considerar dividirla

## Uso de qmd

Si qmd está disponible, úsalo para búsqueda semántica antes de responder preguntas complejas:

```bash
qmd search "términos clave"   # rápido, sin modelos
qmd query "pregunta completa" # semántico con re-ranking
```

No ejecutes `qmd embed` ni `qmd update` automáticamente — el usuario decide cuándo re-indexar.
