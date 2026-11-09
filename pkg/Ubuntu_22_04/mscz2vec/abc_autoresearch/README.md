# abc_autoresearch

Investigación autónoma de modelos de lenguaje para generación de música en formato ABC,
guiada opcionalmente por un **evaluador musical seleccionable** (`preference_trainer.py`
o `quality_scorer.py`).

Basado en [autoresearch](https://github.com/karpathy/autoresearch) de Karpathy, adaptado
a corpus ABC. Este directorio unifica los backups `abc_autoresearch.backup.260620-030133`
(junio) y `abc_autoresearch.backup.260905-051828` (septiembre); lo sustituido está en `old/`.

---

## Requisitos

- Python 3.10+
- [uv](https://astral.sh/uv) (gestor de paquetes)
- `abc2midi` (para fase 2): `sudo apt install abcmidi`
- Claude Code (para el bucle autónomo de experimentación)
- GPU opcional — funciona en CPU
- Para la fase 2, `preference_trainer.py` y/o `quality_scorer.py` (viven en la carpeta
  padre, `mscz2vec/`; ver «Evaluador musical»)

Instalación de dependencias Python (la hace `uv` en el primer `uv run`; las declara
`pyproject.toml`, que ya está en esta carpeta):

```bash
uv sync                          # dependencias base
uv sync --extra conditioned      # + sentence-transformers, solo para abc_conditioned.py
```

---

## Ficheros del proyecto

| Fichero | Descripción |
|---|---|
| `abc_prepare.py` | Corpus y tokenizador BPE. **No modificar.** |
| `abc_model.py` | GPT autorregresivo, `generate_abc_sample`, `is_valid_abc`. Lo comparten `abc_train.py` y `generate.py`. |
| `abc_model_diffusion.py` | Transformer bidireccional de difusión discreta y su muestreador. |
| `abc_eval.py` | **Evaluación musical compartida** (ABC → MIDI → evaluador seleccionable). |
| `abc_train.py` | Entrenamiento del GPT. **El agente lo modifica.** |
| `abc_train_diffusion.py` | Entrenamiento de la variante de difusión (intercambiable con `abc_train.py`). |
| `generate.py` | Generación desde un checkpoint; detecta la arquitectura (`gpt` / `diffusion`). |
| `abc_conditioned.py` | Producto aparte: GPT condicionado por lyrics (`prepare`, `train`, `generate`, `inspect`). |
| `abc_program.md` | Instrucciones del agente para `abc_train.py`. |
| `abc_conditioned_program.md` | Instrucciones del agente para `abc_conditioned.py`. |
| `old/` | Versiones sustituidas: `generate.py`, `generate_patched.py`, `abc_program_v1.md`, `abc_program_v2.md`, `README.md` y `pyproject.toml` anteriores. |

Dependencias entre módulos:

```
abc_prepare ──┐
abc_model ────┼─→ abc_train ───────────┐
              ├─→ abc_model_diffusion ─┼─→ abc_train_diffusion
abc_eval ─────┴────────────────────────┘     (los tres entrenadores usan abc_eval)
generate.py → abc_model, abc_model_diffusion (sin importar ningún entrenador)
```

---

## Evaluador musical (seleccionable)

En fase 2 cada experimento genera muestras ABC, las convierte a MIDI con `abc2midi` y las
puntúa con uno de estos evaluadores:

| `EVAL_BACKEND` | Herramienta | Escala | Requisito |
|---|---|---|---|
| `preference` (defecto) | `preference_trainer.py eval --json` | 0-1 | modelo entrenado (`prefs.prefs.json`, `prefs.linear.json` u otro `prefs.<tipo>.json`) |
| `quality` | `quality_scorer.py --quiet` | 0-1 | ninguno |

Ambos devuelven un valor de 0 a 1, que es el `reward_score` (`pref_score` o `quality_score`). La métrica
compuesta es `combined_score = (1 - PREF_WEIGHT) * (1/val_bpb) + PREF_WEIGHT * reward_score`.

**Selección**, por este orden de prioridad:

```bash
ABC_EVAL_BACKEND=quality uv run abc_train.py        # variable de entorno
# o editar EVAL_BACKEND = "quality" en abc_train.py / abc_train_diffusion.py / abc_conditioned.py
```

**Elige un evaluador por sesión y no lo cambies entre experimentos**: las escalas son
distintas y los `combined_score` dejarían de ser comparables (por eso `results.tsv` lleva
una columna `eval_backend`).

**Dónde busca los ficheros** (`abc_eval.py`): el directorio de trabajo, esta carpeta y su
carpeta padre. Con esta carpeta dentro de `mscz2vec/`, encuentra `preference_trainer.py`,
`quality_scorer.py`, `prefs.prefs.json` y `corpus.npz` en `mscz2vec/` sin configurar nada.
Se pueden fijar a mano con `PREF_TRAINER_PATH`, `QUALITY_SCORER_PATH` (o las variables
`ABC_PREF_TRAINER` y `ABC_QUALITY_SCORER`). Si el backend `quality` necesita opciones
(p. ej. `--fast`), van en `QUALITY_EXTRA_ARGS`.

Resumen que imprimen los tres entrenadores (los campos antiguos se mantienen):

```
phase:            2
val_bpb:          1.182000
eval_backend:     preference
reward_score:     0.4800     ← 0-1, MAYOR es mejor
pref_score:       0.4800     ← solo con backend preference (0-1)
quality_score:    N/A        ← solo con backend quality (0-1)
validity_ratio:   0.600
combined_score:   0.736200
```

---

## Preparación del corpus

Coloca tus ficheros `.abc` en una carpeta (p. ej. `./mnt/`):

```bash
python abc_prepare.py --abc-dir ./mnt/
```

Esto crea en `~/.cache/autoresearch-abc/`:
- `data/train.txt` y `data/val.txt` — corpus dividido 95/5
- `tokenizer/tokenizer.pkl` — tokenizador BPE (vocab 512)

Solo hay que ejecutarlo una vez.

---

## Flujo de trabajo

Dos fases controladas por la variable `PHASE` en `abc_train.py`.

### Fase 1 — Solo lenguaje (`PHASE = 1`)

El modelo aprende la sintaxis ABC optimizando únicamente `val_bpb` (bits por byte, menor es
mejor). No se evalúa calidad musical.

**Condición de avance:** `validity_ratio >= 0.5` en dos experimentos consecutivos (fracción
de muestras generadas que son ABC sintácticamente válido).

### Fase 2 — Lenguaje + calidad musical (`PHASE = 2`)

Al final de cada experimento:

1. Genera `PREF_EVAL_SAMPLES` piezas ABC
2. Filtra las sintácticamente válidas
3. Las convierte a MIDI con `abc2midi`
4. Las puntúa con el evaluador activo (`EVAL_BACKEND`)
5. Calcula `combined_score`

**Métrica de decisión en fase 2:** `combined_score` — mayor es mejor.

---

## Ejecución manual (sin agente)

```bash
git init
git add abc_prepare.py abc_model.py abc_eval.py abc_train.py abc_program.md
git commit -m "baseline"

uv run abc_train.py                              # GPT autorregresivo
uv run abc_train_diffusion.py                    # variante de difusión
ABC_TIME_BUDGET=20 uv run abc_train.py           # smoke test rápido (20 s de entrenamiento)
```

Extraer métricas de un log:

```bash
grep "^phase:\|^val_bpb:\|^eval_backend:\|^reward_score:\|^validity_ratio:\|^combined_score:" run.log
```

---

## Ejecución autónoma con Claude Code

### Preparar el evaluador (solo fase 2)

- Backend `preference`: entrenar el modelo de preferencias (desde `mscz2vec/`):

  ```bash
  python preference_trainer.py index ./mnt/ --extended-features
  python preference_trainer.py train corpus.npz --rounds 30 --mode contrast \
      --soundfont /ruta/a/TimbreOfHeaven.sf2
  ```

  Anota ~30 pares de piezas (20-30 min). Genera `prefs.prefs.json`.
- Backend `quality`: no hace falta preparar nada.

### Arrancar el agente

```bash
cd /ruta/a/mscz2vec/abc_autoresearch
claude
```

Prompt inicial:

```
Lee abc_program.md y arranquemos una nueva sesión. Haz el setup primero.
```

El agente crea una rama `autoresearch/<tag>`, inicializa `results.tsv` y ejecuta el bucle
de experimentación de forma indefinida: modifica hiperparámetros, hace commits, registra
resultados y avanza de fase cuando se cumple la condición. Para `abc_conditioned.py` el
prompt es el mismo pero con `abc_conditioned_program.md`.

**Alcance del agente.** Según `abc_program.md`, el agente solo modifica `abc_train.py`
(hiperparámetros, optimizador y bucle). El código de la arquitectura (`abc_model.py`), la
evaluación (`abc_eval.py`) y la generación quedan fuera de su alcance. Si quieres que el
agente pueda experimentar también con la arquitectura, añade `abc_model.py` a los ficheros
que puede modificar en la sección Setup de `abc_program.md`.

### Seguimiento mientras corre

```bash
cat results.tsv          # tabla de resultados acumulados
tail -f run.log          # log del experimento en curso
git log --oneline        # historial de experimentos
```

---

## Generar y escuchar música

Al terminar cada run se guarda `model_checkpoint.pt`. `generate.py` lee el campo `arch`
del checkpoint (los de `abc_train.py` no lo tienen y se tratan como GPT), así que sirve
igual para el GPT y para el modelo de difusión.

```bash
uv run generate.py --play                    # generar 1 pieza y reproducirla
uv run generate.py --n 5 --only-valid        # 5 piezas, solo las válidas
uv run generate.py --n 3 --temp 0.7          # más predecible
```

Las piezas se guardan en `./generated/` como `.abc` y `.mid`.

| Opción | Defecto | Descripción |
|---|---|---|
| `--checkpoint` | `model_checkpoint.pt` | Ruta al checkpoint |
| `--n` | `1` | Número de piezas a generar |
| `--temp` | `0.85` | Temperatura de muestreo |
| `--max-tokens` | `256` | Longitud máxima en tokens |
| `--only-valid` | — | Solo guardar piezas ABC válidas |
| `--out-dir` | `./generated` | Directorio de salida |
| `--play` | — | Reproducir la primera pieza válida |
| `--no-midi` | — | No convertir a MIDI |

---

## Hiperparámetros principales (`abc_train.py`)

| Variable | Defecto | Descripción |
|---|---|---|
| `PHASE` | `1` | Fase activa (1 o 2) |
| `EVAL_BACKEND` | `"preference"` | Evaluador de la fase 2 (`preference` / `quality`); el usuario lo fija |
| `DEPTH` | `4` | Número de capas transformer |
| `ASPECT_RATIO` | `24` | model_dim = DEPTH × ASPECT_RATIO |
| `HEAD_DIM` | `32` | Dimensión por cabeza de atención |
| `TOTAL_BATCH_SIZE` | `1024` | Tokens por paso de optimización |
| `DEVICE_BATCH_SIZE` | `2` | Batch size por forward pass |
| `LR` | `1e-3` | Learning rate AdamW |
| `PREF_WEIGHT` | `0.3` | Peso de `reward_score` en `combined_score` |
| `PREF_TEMPERATURE` | `0.85` | Temperatura de muestreo ABC en la evaluación |
| `PREF_EVAL_SAMPLES` | `20` | Muestras a evaluar en fase 2 |

**Restricción importante:** `TOTAL_BATCH_SIZE` debe ser múltiplo de
`DEVICE_BATCH_SIZE × MAX_SEQ_LEN`. Con los defaults: múltiplo de `2 × 512 = 1024`.
Valores válidos: 1024, 2048, 4096, etc.

---

## Notas para CPU (sin GPU)

Usa `F.scaled_dot_product_attention` de PyTorch en lugar de Flash Attention 3, y AdamW en
lugar de Muon. Funciona en cualquier máquina. Referencia orientativa (T14 Gen 3, Intel
integrado, con `DEPTH=2` y `ASPECT_RATIO=32`): ~32 ms por paso, ~8500 pasos en 5 minutos,
~8,5 M tokens por experimento. Empieza con modelos pequeños y sube gradualmente.

---

## Estructura de results.tsv

```
commit  phase  val_bpb  eval_backend  reward_score  validity_ratio  combined_score  memory_gb  status  description
```

Ejemplo (backend `preference`, `PREF_WEIGHT = 0.3`):

```
a1b2c3d  1  3.091  preference  N/A    N/A    N/A    —  keep     baseline
d4e5f6g  2  2.690  preference  N/A    0.420  N/A    —  keep     diagnóstico validity
g7h8i9j  2  2.600  preference  0.480  0.580  0.413  —  keep     FASE 2 activada
```

---

## Cambios respecto a los backups (unificación)

**Estructura.** Nuevos `abc_eval.py` (evaluación compartida) y `abc_model_diffusion.py`
(modelo de difusión extraído del entrenador); `generate_patched.py` pasa a ser `generate.py`
(sin importar ningún entrenador); `abc_program.md` y `abc_program_v2.md` se fusionan en un
único `abc_program.md`; `abc_train.py` admite `ABC_TIME_BUDGET` como el de difusión.

**Correcciones en la evaluación con `preference_trainer.py`** (afectan a resultados antiguos):

1. *Escala.* `eval` devuelve `score` en [0, 1] (sigmoide del modelo; sus valoraciones 1-5 se
   mapean con `(r-1)/4`). El código anterior lo trataba como 0-5 y dividía por 5, de modo que
   el reward quedaba comprimido a 0-0,2. Los `combined_score` antiguos con backend `preference`
   no son comparables con los actuales.
2. *Lectura del JSON.* `eval --json` imprime una **lista** de entradas; el código anterior leía
   solo el último objeto y promediaba todos sus números, incluido `percentile` (0-100) cuando
   había corpus. Ahora se promedia el `score` de todas las muestras.
3. *Nombre del modelo.* Los modelos tipados se guardan como `prefs.linear.json`,
   `prefs.neural.json`... (no `prefs.prefs.linear.json`, que es lo que se buscaba).
4. *`abc_conditioned.py`* llamaba a `eval` sin `--model` ni `--json` y leía texto libre; ahora
   usa la misma ruta que `abc_train.py` (incluido `--corpus` si existe `corpus.npz`).

**Otras correcciones.** `abc_conditioned.py` hacía `clip_grad_norm_(params, ...)` con `params`
sin definir (`NameError` en el primer paso de `train`); `abc_program.md` proponía ajustar
`WINDOW_PATTERN` y `MATRIX_LR`, que no existen en `abc_train.py`; la tabla de hiperparámetros
del README no coincidía con los valores reales; `pyproject.toml` declaraba dependencias que
ningún `.py` importa (`kernels`, `matplotlib`, `pandas`, `pyarrow`, `requests`) y no declaraba
`sentence-transformers`.
