# autoresearch-abc

Investigación autónoma de modelos de lenguaje para generación de música en formato ABC,
con guía opcional de un **evaluador musical seleccionable**: `preference_trainer.py`
(modelo de preferencias personal) o `quality_scorer.py` (scorer automático). Ver `abc_eval.py`.

El proceso tiene **dos fases explícitas** controladas por la variable `PHASE` en `abc_train.py`.

---

## Setup

Para iniciar un experimento, trabaja con el usuario para:

1. **Acordar un tag de sesión**: proponer un tag basado en la fecha (e.g. `abc_may5`).
   La rama `autoresearch/<tag>` no debe existir — es una sesión nueva.
2. **Crear la rama**: `git checkout -b autoresearch/<tag>` desde master.
3. **Leer los ficheros en scope**:
   - `abc_train.py` — hiperparámetros, optimizador, bucle de entrenamiento. **Tú lo modificas**.
   - `abc_program.md` — estas instrucciones. **No modificar**.
   - `abc_prepare.py` — corpus y tokenizador. **No modificar**.
   - `abc_model.py`, `abc_eval.py` — modelo, evaluación y generación compartidos
     (también los usan `abc_train_diffusion.py` y `generate.py`). **No modificar**.
4. **Verificar datos**: comprobar que `~/.cache/autoresearch-abc/` contiene `data/train.txt`,
   `data/val.txt` y `tokenizer/tokenizer.pkl`. Si no, decirle al usuario:
   ```
   python abc_prepare.py --abc-dir ./abc/
   ```
5. **Verificar fase inicial**: confirmar que `PHASE = 1` en `abc_train.py`. Si no, corregirlo.
6. **Anotar el evaluador** (informativo, no bloqueante): leer `EVAL_BACKEND` en `abc_train.py`
   (o `ABC_EVAL_BACKEND` en el entorno, que tiene prioridad).
   - `preference`: necesita un modelo entrenado (`prefs.prefs.json`, `prefs.linear.json` u otro `prefs.<tipo>.json`).
   - `quality`: no necesita entrenamiento.
   **El evaluador lo elige el usuario. No lo cambies durante la sesión**: las escalas son
   distintas (ambas van de 0 a 1, pero miden cosas diferentes) y los `combined_score`
   dejarían de ser comparables.
7. **Inicializar results.tsv** con solo la cabecera.
8. **Confirmar y arrancar**.

---

## Las dos fases

### FASE 1 — Solo val_bpb (`PHASE = 1`)

**Objetivo**: que el modelo aprenda la sintaxis ABC y genere piezas válidas.

- La variable `PHASE = 1` está activa en `abc_train.py`.
- No se generan muestras ni se evalúa calidad musical — la evaluación al final es
  únicamente `val_bpb` (igual que el autoresearch original).
- **Métrica de decisión**: `val_bpb` — MENOR es mejor.
- **Condición de avance a fase 2**: cuando en **dos experimentos consecutivos**
  `validity_ratio >= 0.5` (ver más abajo cómo medirlo).

Para medir `validity_ratio` en fase 1 sin activar el pipeline completo, puedes
añadir ocasionalmente un experimento de diagnóstico cambiando temporalmente
`PHASE = 2`. Con backend `preference` y sin modelo de preferencias, `reward_score` saldrá
`N/A` pero `validity_ratio` se calculará (rápido). Con backend `quality` también se
ejecutará el scorer sobre las muestras válidas (más lento). Regresa a `PHASE = 1` para el
experimento siguiente si la condición de avance no se cumple aún.

### FASE 2 — val_bpb + reward_score (`PHASE = 2`)

**Objetivo**: que el modelo genere música que además puntúe bien con el evaluador elegido.

- Cambiar `PHASE = 2` en `abc_train.py` cuando se cumpla la condición de avance.
- Al terminar el entrenamiento, el script genera `PREF_EVAL_SAMPLES` piezas ABC,
  las convierte a MIDI con `abc2midi`, y las evalúa con el backend activo:
  `preference_trainer.py eval --json` o `quality_scorer.py --quiet`.
- **Métrica de decisión**: `combined_score` — MAYOR es mejor.
- **Requisitos**: backend `preference` → debe existir el modelo de preferencias; si no,
  `reward_score` saldrá `N/A` y debes usar solo `val_bpb` hasta que el usuario lo entrene.
  Backend `quality` → solo requiere que `quality_scorer.py` sea localizable (se busca en el
  directorio de trabajo, en este proyecto y en su carpeta padre).

---

## Métricas por fase

**Fase 1** — resumen al final del run:
```
phase:            1
val_bpb:          0.XXXXXX   ← MENOR es mejor
eval_backend:     preference
reward_score:     N/A
pref_score:       N/A
quality_score:    N/A
validity_ratio:   N/A
combined_score:   N/A
```

**Fase 2** — resumen al final del run (ejemplo con backend `preference`):
```
phase:            2
val_bpb:          0.XXXXXX   ← MENOR es mejor (calidad lingüística)
eval_backend:     preference
reward_score:     0.XXXX     ← 0-1, MAYOR es mejor (calidad musical, normalizada)
pref_score:       0.XXXX     ← 0-1, solo con backend preference (igual a reward_score)
quality_score:    N/A        ← 0-1, solo con backend quality (reward_score = quality_score)
validity_ratio:   0.XXX      ← fracción de muestras ABC válidas generadas
combined_score:   X.XXXXXX   ← métrica compuesta, MAYOR es mejor
```

**Notas sobre combined_score:**
- `combined_score = (1 - PREF_WEIGHT) * (1/val_bpb) + PREF_WEIGHT * reward_score`
- Un `reward_score` alto con `val_bpb` muy alto es sospechoso — puede ser degeneración.
- Con backend `preference` y un modelo con pocos ejemplos (< 20), considera `PREF_WEIGHT = 0.1`.

---

## Extracción de resultados

```bash
grep "^phase:\|^val_bpb:\|^eval_backend:\|^reward_score:\|^validity_ratio:\|^combined_score:" run.log
```

Si el grep está vacío → crash. Ver stack trace:
```bash
tail -n 50 run.log
```

---

## Registro en results.tsv

Columnas (tab-separated, NO comas):

```
commit  phase  val_bpb  eval_backend  reward_score  validity_ratio  combined_score  memory_gb  status  description
```

Ejemplo de sesión completa con transición de fase (backend `preference`):

```
commit   phase  val_bpb   eval_backend  reward_score  validity_ratio  combined_score  memory_gb  status   description
a1b2c3d  1      1.420000  preference    N/A           N/A             N/A             2.1        keep     baseline fase1
b2c3d4e  1      1.310000  preference    N/A           N/A             N/A             2.1        keep     depth=6
c3d4e5f  1      1.250000  preference    N/A           N/A             N/A             2.1        keep     LR x0.5
d4e5f6g  2      1.230000  preference    N/A           0.420           N/A             2.1        keep     diagnóstico validity
e5f6g7h  1      1.210000  preference    N/A           N/A             N/A             2.1        keep     batch más pequeño
f6g7h8i  2      1.195000  preference    N/A           0.560           N/A             2.1        keep     diagnóstico: validity>=0.5 primera vez
g7h8i9j  2      1.188000  preference    N/A           0.580           N/A             2.2        keep     diagnóstico: validity>=0.5 segunda vez → avanzar a fase 2
h8i9j0k  2      1.182000  preference    0.480         0.600           0.736           2.2        keep     FASE 2 activada
i9j0k1l  2      1.175000  preference    0.530         0.630           0.755           2.2        keep     temperatura 0.75
j0k1l2m  2      1.180000  preference    0.440         0.580           0.725           2.1        discard  LR x2, reward_score baja
```

---

## Bucle de experimentos

BUCLE INFINITO:

1. Ver estado git: rama/commit actual y fase activa (`grep "^PHASE" abc_train.py`)
2. Modificar `abc_train.py` con la idea experimental
3. `git commit`
4. Ejecutar: `uv run abc_train.py > run.log 2>&1`
5. Extraer métricas: `grep "^phase:\|^val_bpb:\|^eval_backend:\|^reward_score:\|^validity_ratio:\|^combined_score:" run.log`
6. Si grep vacío → crash. Intentar fix. Si no es fácil → log "crash", continuar.
7. Registrar en results.tsv
8. **Regla de avance de fase**: si estás en fase 1 y acabas de ver `validity_ratio >= 0.5`
   por segunda vez consecutiva (en experimentos de diagnóstico) → cambiar `PHASE = 2`
   en `abc_train.py` para el siguiente experimento. Anotar el cambio en la descripción.
9. **Regla de keep/discard**:
   - Fase 1 (experimentos normales): `val_bpb` mejoró → keep; si no → discard + `git reset --hard HEAD~1`
   - Fase 1 (diagnósticos de validity): siempre keep si no crashea; son informativos
   - Fase 2: `combined_score` mejoró → keep; si no → discard + `git reset --hard HEAD~1`

**Timeout**: si un run supera 10 minutos → kill, tratar como crash. (Con backend `quality`
la evaluación de fase 2 añade tiempo: el scorer es más lento que `preference_trainer.py`.)

**NUNCA PARAR**: una vez iniciado el bucle, no preguntar al usuario si continuar.
El usuario puede estar durmiendo. Continúa indefinidamente hasta que te interrumpan.

---

## Variante de difusión

`abc_train_diffusion.py` es intercambiable con `abc_train.py` (mismos datos, misma salida por
consola, mismo checkpoint más los campos `arch` y `mask_token_id`). Úsala solo si el usuario
lo pide para la sesión. Su `val_bpb` es un **proxy** (denoising con la fila 100 % enmascarada),
no una verosimilitud autorregresiva: no es comparable directamente con el del GPT. Si alternas
entre ambas arquitecturas, anótalo en `description` y no compares sus `val_bpb` entre sí.

---

## Ideas de experimentos por fase

### Fase 1 — Mejorar val_bpb

- **Baseline primero**: ejecutar sin cambios para establecer la referencia.
- Buscar el `DEPTH` óptimo: probar 3, 4, 5, 6, 8
- Ajustar `TOTAL_BATCH_SIZE`: con corpus ABC pequeño, `2**13` o `2**14` puede
  dar más pasos de gradiente y convergencia más rápida
- Explorar `HEAD_DIM`: 32 o 64 (el modelo es pequeño, no necesita 128)
- Ajustar `LR` entre 3e-4 y 3e-3
- Experimentar con `WARMDOWN_RATIO`: con pocos datos, más cooldown puede ayudar

**Cuándo hacer diagnósticos de validity** (con `PHASE = 2`):
- Cada 5-8 experimentos de fase 1 para ver si el modelo ya genera ABC válido
- Si `validity_ratio < 0.2`: el modelo no ha visto suficientes épocas o es demasiado pequeño
- Si `validity_ratio` entre 0.2–0.5: la estructura básica emerge, optimizar val_bpb
- Si `validity_ratio >= 0.5` dos veces seguidas: pasar permanentemente a fase 2

### Fase 2 — Mejorar combined_score

- **Primera ejecución**: establecer baseline de fase 2 con los mejores parámetros de fase 1
- Ajustar `PREF_TEMPERATURE`: 0.6–0.7 para coherencia, 0.85–1.0 para variedad
- Subir `PREF_EVAL_SAMPLES` a 50 para estimación más robusta del reward_score
  (con backend `quality` cada muestra extra alarga el run)
- Explorar `PREF_WEIGHT`: 0.2 si la señal del evaluador es débil, 0.4–0.5 si es fuerte
- Continuar mejorando val_bpb — sigue siendo el 70% de la métrica con PREF_WEIGHT=0.3
- Probar distintos seeds (`torch.manual_seed`) si el modelo parece atascado
