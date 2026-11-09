# abc_conditioned — Investigación autónoma

Generación de melodías ABC condicionadas a lyrics (texto).
Motor: GPT (RoPE + QK-Norm + Value Embeddings) con `emotion_proj` (MiniLM → d_model).

El proceso tiene **dos fases explícitas** controladas por `PHASE` en `abc_conditioned.py`.

---

## Setup

Al iniciar una sesión de investigación:

1. **Acordar un tag de sesión** con el usuario, basado en la fecha (ej. `cond_jun20`).
   La rama `autoresearch/<tag>` no debe existir — es una sesión nueva.
2. **Crear la rama**: `git checkout -b autoresearch/<tag>` desde master.
3. **Leer los ficheros en scope**:
   - `abc_conditioned.py` — todo en un solo fichero. **Tú lo modificas**.
   - `abc_conditioned_program.md` — estas instrucciones. **No modificar**.
4. **Verificar datos BPE**: comprobar que `~/.cache/autoresearch-abc/` contiene
   `data/train.txt`, `data/val.txt` y `tokenizer/tokenizer.pkl`.
   Si no: `python abc_conditioned.py prepare --abc-dir ./corpus/`
5. **Verificar fase inicial**: `grep "^PHASE" abc_conditioned.py` → debe ser `PHASE = 1`.
6. **Anotar el evaluador**: leer `EVAL_BACKEND` en `abc_conditioned.py` (o `ABC_EVAL_BACKEND`
   en el entorno, que tiene prioridad): `preference` (`preference_trainer.py`, escala 0-1,
   necesita modelo entrenado) o `quality` (`quality_scorer.py`, escala 0-1). **Lo elige el
   usuario; no lo cambies durante la sesión** (las escalas no son comparables). Ver `abc_eval.py`.
7. **Inicializar `results.tsv`** con solo la cabecera (ver formato más abajo).
8. **Confirmar y arrancar el bucle**.

---

## Ficheros en scope — qué modificar y qué no

### Variables que el agente modifica en `abc_conditioned.py`

Bloque marcado con `← EL AGENTE MODIFICA ESTO`:

```python
PHASE = 1                       # ← EL AGENTE MODIFICA ESTO
PREF_TEMPERATURE   = 0.9        # ← EL AGENTE MODIFICA ESTO
PREF_EVAL_SAMPLES  = 20         # ← EL AGENTE MODIFICA ESTO
PREF_WEIGHT        = 0.3        # ← EL AGENTE MODIFICA ESTO
PREF_TOKENS_PER_PHRASE = 60     # ← EL AGENTE MODIFICA ESTO
PREF_N_PHRASES     = 4          # ← EL AGENTE MODIFICA ESTO
```

Y los argumentos por defecto en `cmd_train` / `build_parser` (hiperparámetros del modelo):
- `--epochs` (default: 60)
- `--batch-size` (default: 8)
- `--lr` (default: 3e-4)
- `--d-model` (default: 128)
- `--n-heads` (default: 4)
- `--n-layers` (default: 4)
- `--max-seq-len` (default: 512)
- `--label-smooth` (default: 0.1)

> El agente **cambia los valores por defecto** en `build_parser`, no pasa flags en CLI,
> para que git diff muestre exactamente qué cambió.

### Qué NO modificar

- El tokenizador BPE y el caché (`prepare`, `CACHE_DIR`, `train_tokenizer`).
- La función `evaluate_bpb` — es la métrica de referencia.
- La arquitectura `ConditionedGPT` y `ConditionedGPTConfig` — solo sus hiperparámetros.
- El pipeline de generación (`cmd_generate`) y la CLI.

---

## Las dos fases

### FASE 1 — Solo val_bpb (`PHASE = 1`)

**Objetivo**: que el modelo aprenda la sintaxis ABC y el condicionamiento básico
(embeddings de lyrics → notas) antes de introducir la señal del evaluador musical.

- No se generan muestras ni se llama a ningún evaluador.
- **Métrica de decisión**: `val_bpb` — MENOR es mejor.
- **Condición de avance a fase 2**: `validity_ratio >= 0.5` en **dos diagnósticos
  consecutivos** (ver abajo cómo hacer diagnósticos).

#### Diagnósticos de validity en fase 1

Para medir `validity_ratio` sin activar el pipeline completo:
1. Cambiar temporalmente `PHASE = 2` en `abc_conditioned.py`.
2. Ejecutar `train` normalmente.
3. El resumen imprimirá `validity_ratio` real (aunque `reward_score` pueda ser `N/A`
   si el evaluador no está disponible, p. ej. backend `preference` sin modelo entrenado).
4. **Volver a `PHASE = 1`** para el siguiente experimento si la condición no se cumple.

Frecuencia recomendada: cada 5-8 experimentos de fase 1.

### FASE 2 — val_bpb + reward_score (`PHASE = 2`)

**Objetivo**: mejorar la calidad musical (según el evaluador elegido) además de la lingüística.

- Al final del entrenamiento se generan `PREF_EVAL_SAMPLES` muestras ABC (libre,
  embedding cero), se convierten a MIDI con `abc2midi`, y se evalúan con el backend
  activo (`preference_trainer.py eval --json` o `quality_scorer.py --quiet`).
- **Métrica de decisión**: `combined_score` — MAYOR es mejor.

```
combined_score = (1 - PREF_WEIGHT) * (1 / val_bpb) + PREF_WEIGHT * reward_score
```

`reward_score` va de 0 a 1 con ambos backends: es `pref_score` con `preference` y
`quality_score` con `quality`.
Con `PREF_WEIGHT = 0.3`: 70% calidad lingüística, 30% calidad musical.

---

## Métricas — formato del resumen

Al final de cada `train` el script imprime:

```
phase:            1
val_bpb:          1.320000
eval_backend:     preference
reward_score:     N/A
pref_score:       N/A
quality_score:    N/A
validity_ratio:   N/A
combined_score:   N/A
n_songs:          1200
d_model:          128
n_layer:          4
n_head:           4
n_params:         2340864
mem_gb:           0.01
train_seconds:    240
```

En fase 2 con evaluación completa:

```
phase:            2
val_bpb:          1.185000
eval_backend:     preference
reward_score:     0.5700
pref_score:       2.8500
quality_score:    N/A
validity_ratio:   0.600
combined_score:   0.983400
n_songs:          1200
d_model:          256
n_layer:          6
n_head:           8
n_params:         8932352
mem_gb:           0.04
train_seconds:    480
```

### Extracción de métricas del log

```bash
grep "^phase:\|^val_bpb:\|^eval_backend:\|^reward_score:\|^validity_ratio:\|^combined_score:" run.log
```

Si el grep está vacío → crash. Ver stack trace:

```bash
tail -n 60 run.log
```

---

## Registro en results.tsv

Columnas (tab-separated, sin comas):

```
commit  phase  val_bpb  eval_backend  reward_score  validity_ratio  combined_score  mem_gb  status  description
```

Ejemplo de sesión completa:

```
commit   phase  val_bpb   eval_backend  reward_score  validity_ratio  combined_score  mem_gb  status   description
a1b2c3d  1      1.520000  preference    N/A           N/A             N/A             0.01   keep     baseline d_model=128 n_layer=4
b2c3d4e  1      1.410000  preference    N/A           N/A             N/A             0.01   keep     n_layer=6
c3d4e5f  1      1.350000  preference    N/A           N/A             N/A             0.02   keep     d_model=192 n_layer=6
d4e5f6g  2      1.340000  preference    N/A           0.380           N/A             0.02   keep     diagnostico validity: 0.38 < 0.5
e5f6g7h  1      1.310000  preference    N/A           N/A             N/A             0.02   keep     lr=1e-4 warmup mas largo
f6g7h8i  1      1.280000  preference    N/A           N/A             N/A             0.02   keep     label_smooth=0.05
g7h8i9j  2      1.275000  preference    N/A           0.530           N/A             0.02   keep     diagnostico validity: 0.53 >= 0.5 (1a vez)
h8i9j0k  2      1.270000  preference    N/A           0.560           N/A             0.02   keep     diagnostico validity: 0.56 >= 0.5 (2a vez) -> FASE 2
i9j0k1l  2      1.265000  preference    0.4800        0.570           0.697360        0.02   keep     FASE 2: baseline con prefs
j0k1l2m  2      1.258000  preference    0.5300        0.590           0.715439        0.02   keep     temperatura 0.85 en pref eval
k1l2m3n  2      1.262000  preference    0.4300        0.540           0.611197        0.02   discard  PREF_WEIGHT=0.5: combined_score baja
```

---

## Bucle de experimentos

**BUCLE INFINITO**:

1. Ver estado git: rama/commit actual y `grep "^PHASE" abc_conditioned.py`
2. Decidir el siguiente experimento (ver "Ideas" más abajo)
3. Modificar `abc_conditioned.py`
4. `git add abc_conditioned.py && git commit -m "<descripcion breve>"`
5. Ejecutar:
   ```bash
   python abc_conditioned.py train \
     --corpus ./corpus/ \
     --model-out conditioned.pt \
     > run.log 2>&1
   ```
6. Extraer métricas con grep (ver arriba)
7. Si grep vacío → crash → `tail -n 60 run.log` → intentar fix → si no es fácil,
   anotar "crash" en results.tsv y continuar con siguiente experimento
8. Registrar en results.tsv
9. **Regla de avance de fase**: si estás en fase 1 y el último diagnóstico de validity
   muestra `validity_ratio >= 0.5` por segunda vez consecutiva → cambiar `PHASE = 2`
   permanentemente en el siguiente experimento. Anotarlo en la descripción.
10. **Regla de keep/discard**:
    - Fase 1 normal: `val_bpb` mejoró → keep; si no → discard + `git reset --hard HEAD~1`
    - Fase 1 diagnóstico: keep siempre si no crashea (son informativos, no descartar)
    - Fase 2: `combined_score` mejoró → keep; si no → discard + `git reset --hard HEAD~1`
    - Si `reward_score = N/A` en fase 2 (evaluador no disponible, p. ej. backend `preference`
      sin modelo entrenado): usar solo `val_bpb` como criterio hasta que esté disponible.

**Timeout**: si un run supera 15 minutos → kill (`Ctrl-C` o `kill`), tratar como crash.

**NUNCA PARAR**: una vez iniciado el bucle, no preguntar al usuario si continuar.
El usuario puede estar durmiendo. Continúa indefinidamente hasta que te interrumpan.

---

## Ideas de experimentos por fase

### Fase 1 — Mejorar val_bpb

**Primera ejecución**: baseline sin cambios, para establecer referencia.

**Arquitectura del modelo** (modificar valores por defecto en `build_parser`):
- `--d-model`: probar 64, 128, 192, 256. Con corpus pequeño, mayor no siempre gana.
- `--n-layers`: probar 2, 4, 6, 8.
- `--n-heads`: mantener `n_embd % n_heads == 0`. Con d_model=128: 4 u 8 cabezas.
- Relación d_model/n_layers: preferir modelos más "anchos" (mayor d_model, menos capas)
  para corpus ABC pequeño; los modelos "profundos" pueden sobreajustar.

**Optimización**:
- `--lr`: rango útil 1e-4 a 5e-4. Bajar si la loss oscila; subir si converge lento.
- `--batch-size`: con corpus pequeño (< 500 canciones), batch=4 puede ser suficiente.
- `--label-smooth`: 0.05–0.15. Reduce sobreajuste a tokens frecuentes de ABC.
- `--epochs`: aumentar a 100–150 si val_bpb sigue bajando al final del run.

**Condicionamiento** (la novedad respecto a abc_train.py):
- `emotion_proj` ya usa dos capas Linear + GELU. Si val_bpb estanca, probar
  simplificar a una sola capa (`nn.Linear(cfg.emotion_dim, cfg.n_embd)`) para
  reducir parámetros innecesarios en corpus pequeño.
- Si las canciones del corpus tienen muy pocas `w:`, el condicionamiento aporta
  poco. Inspeccionar con `abc_conditioned.py inspect --corpus ./corpus/` para
  ver cuántos versos hay de media.

**Cuándo hacer diagnósticos de validity**:
- Cada 5-8 experimentos de fase 1.
- Si `validity_ratio < 0.2`: el modelo no ha visto suficientes épocas o es demasiado
  pequeño para el corpus dado.
- Si `validity_ratio` entre 0.2–0.5: la estructura ABC emerge; optimizar val_bpb.
- Si `validity_ratio >= 0.5` dos veces seguidas → pasar permanentemente a fase 2.

### Fase 2 — Mejorar combined_score

**Primera ejecución en fase 2**: baseline con mejores parámetros de fase 1.

**Calidad de muestras de evaluación**:
- `PREF_EVAL_SAMPLES`: subir a 30–50 para estimación más robusta del `reward_score`.
- `PREF_TOKENS_PER_PHRASE`: 40–80. Más tokens = piezas más largas pero más riesgo
  de degenerar. Empezar en 60.
- `PREF_N_PHRASES`: 3–6. Ajustar a la duración típica de las canciones del corpus.

**Temperatura de muestreo**:
- `PREF_TEMPERATURE`: 0.7 para más coherencia, 0.9–1.0 para más variedad.
  Un combined_score bajo con reward_score alto y validity_ratio baja sugiere
  temperatura muy alta (genera notas "creativas" pero inválidas en ABC).

**Peso del evaluador**:
- `PREF_WEIGHT`: 0.2 si la señal del evaluador es débil o ruidosa;
  0.4–0.5 si el evaluador es fiable y el corpus es rico en lyrics.
  No subir por encima de 0.5 hasta tener al menos 20 puntuaciones en results.tsv.

**Continuar mejorando val_bpb**: sigue siendo el 70% de combined_score con
PREF_WEIGHT=0.3. No abandonarlo en fase 2.

---

## Comandos de diagnóstico útiles

```bash
# Inspeccionar modelo y corpus
python abc_conditioned.py inspect \
    --model conditioned.pt \
    --corpus ./corpus/

# Generar muestra de prueba con lyrics
python abc_conditioned.py generate \
    --model conditioned.pt \
    --lyrics sample_lyrics.txt \
    --abc-out test_out.abc \
    --temperature 0.85

# Generar muestra libre
python abc_conditioned.py generate \
    --model conditioned.pt \
    --n-phrases 4 \
    --abc-out test_free.abc

# Convertir a MIDI manualmente
abc2midi test_out.abc -o test_out.mid
timidity test_out.mid   # o fluidsynth

# Ver ultimas lineas del log si hay crash
tail -n 60 run.log

# Comparar cambios del experimento actual con el anterior
git diff HEAD~1 abc_conditioned.py | grep "^[+-]" | grep -v "^---\|^+++"
```

---

## Notas sobre el condicionamiento

A diferencia de `abc_train.py`, este modelo tiene **dos señales de aprendizaje**:

1. **Señal lingüística** (val_bpb): ¿genera texto ABC sintácticamente correcto?
2. **Señal de condicionamiento**: ¿las notas generadas se parecen a las que aparecen
   en el corpus junto a ese tipo de lyrics? Esta señal es implícita — el modelo
   aprende la correlación lyrics→melodía a través del entrenamiento supervisado.

En fase 1, la convergencia puede ser más lenta que `abc_train.py` porque el modelo
tiene más parámetros (emotion_proj añade ~384×d_model×2 parámetros). Si el corpus
tiene pocas canciones con `w:`, el condicionamiento añade ruido más que señal —
en ese caso, reducir la tasa de aprendizaje de `emotion_proj` es una opción avanzada
(requeriría modificar el optimizer para separar grupos de parámetros).

La validez del ABC generado (`validity_ratio`) en este modelo puede ser inicialmente
peor que en `abc_train.py` puro porque el embedding emocional variable perturba
la distribución aprendida. Es normal que necesite más épocas para estabilizarse.
