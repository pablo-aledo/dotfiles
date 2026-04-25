#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   MELODY CONDITIONED  v4.0                                  ║
║       Generación de melodías condicionadas a progresión de acordes          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NUEVOS MOTORES en v4.0:                                                     ║
║    motif       — Transformaciones sobre un motivo semilla de 2-4 notas.    ║
║                  Repetición, secuencia, inversión, retrogradación,          ║
║                  aumentación/disminución y variación ornamental.            ║
║                  Produce coherencia temática interna. El motivo puede       ║
║                  venir del usuario (--motif) o generarse automáticamente.   ║
║    bayes       — N-gram con suavizado de Kneser-Ney entrenado sobre el     ║
║                  corpus. Siempre tiene una estimación razonada para          ║
║                  transiciones no vistas. Sin torch, sin GPU.                ║
║    tension_arc — Planificación de atrás hacia adelante. Decide primero      ║
║                  la nota de resolución de cada frase, luego la de máxima   ║
║                  tensión, y rellena el camino entre ambas. Narrativa        ║
║                  armónica explícita con destino audible.                    ║
║    interpolate — Interpolación lineal entre dos perfiles emocionales.       ║
║                  --alpha 0.0 = perfil A puro, --alpha 1.0 = perfil B.     ║
║                  Permite explorar el espacio entre flamenco y tanguero,    ║
║                  o cualquier otro par de perfiles.                          ║
║                                                                              ║
║  RESUMEN                                                                     ║
║  ───────                                                                     ║
║  Genera una melodía MIDI a partir de una progresión de acordes dada.        ║
║  A diferencia de un generador ex nihilo, cada nota se condiciona al         ║
║  acorde activo en ese momento: el motor sabe en todo instante qué acorde    ║
║  suena y toma decisiones de altura y ritmo en consecuencia.                 ║
║                                                                              ║
║  Cinco motores de distinta naturaleza: reglas armónicas (chord_guided),     ║
║  elaboración contrapuntística (grammar), búsqueda heurística (search),      ║
║  evolución (genetic) y aprendizaje desde corpus propio (lstm).              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MOTORES (--engine)                                                          ║
║  ──────────────────                                                          ║
║  chord_guided  Markov de 2º orden (intervalo × duración) condicionado       ║
║                nota a nota al acorde activo. Los pesos de cada pitch se     ║
║                calculan como: función_armónica × perfil_emocional ×         ║
║                tensión_actual. Penaliza avoid notes en tiempos fuertes.     ║
║                Sin entrenamiento previo. Resultado reproducible con --seed. ║
║                Con --corpus aprende transiciones desde MIDIs reales en      ║
║                lugar de usar tablas sintéticas.                             ║
║                                                                              ║
║  grammar       Elaboración Schenkerian en tres capas ancladas al acorde.    ║
║                Capa 1: selecciona el chord tone más cercano al valor de     ║
║                contorno como nota estructural de cada compás. Capa 2:       ║
║                rellena con notas de paso diatónicas entre estructuras.      ║
║                Capa 3: añade bordaduras y appoggiaturas en compases de      ║
║                alta tensión. Determinista: misma entrada → misma salida.    ║
║                Produce melodías más "cantables" que los otros motores.      ║
║                                                                              ║
║  search        Beam search A* (beam=5) sobre el espacio de pitches.         ║
║                Función de coste local: penaliza avoid notes en tiempo       ║
║                fuerte (+5.0), notas fuera de escala (+2.5), saltos >12      ║
║                semitonos, repetición (iv=0); premia chord tones (-1.5) y   ║
║                consonancias. Garantiza propiedades locales por nota.        ║
║                Más lento que chord_guided pero produce más coherencia       ║
║                armónica paso a paso.                                        ║
║                                                                              ║
║  genetic       Algoritmo genético. Población de 20 melodías evolucionada    ║
║                durante 30 generaciones. Fitness multicriterio: armonía      ║
║                por acorde activo (30%), contorno (20%), suavidad (20%),     ║
║                variedad rítmica (10%), rango (10%), clímax en proporción    ║
║                áurea ajustada al perfil (10%). Mejor para encontrar         ║
║                soluciones globalmente óptimas a costa de más tiempo CPU.    ║
║                                                                              ║
║  lstm          Red neuronal LSTM pequeña (~114K parámetros) entrenada       ║
║                sobre un corpus propio de MIDIs. Aprende el estilo           ║
║                melódico del corpus y lo aplica condicionado al acorde.      ║
║                Diseñado para CPU con poca RAM. Ver sección LSTM más abajo.  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FUENTES DE ACORDES                                                          ║
║  ──────────────────                                                          ║
║  --chords "Am:2 G:2 F:2 E7:2"                                               ║
║      Texto directo. Formato: NombreAcorde:beats. Si se omiten los beats,    ║
║      se asumen 4 (un compás en 4/4). Soporta: mayor (C), menor (Cm),       ║
║      séptimas (G7, Cmaj7, Am7), disminuidos (Bdim, Bdim7), aumentados      ║
║      (Caug), semidisminuidos (Bø), sus (Csus4, Csus2), extensiones         ║
║      (C9, Cadd9, C13) e inversiones ignoradas (C/E → C).                   ║
║                                                                              ║
║  --chords-json prog.chords.json                                              ║
║      JSON con lista de acordes. Formatos soportados:                        ║
║        ["Am:2", "G:2", "F:2"]            (lista de strings)                ║
║        [{"name":"Am","duration":2}, …]   (lista de objetos)                ║
║        {"chords": […]}                   (objeto con clave "chords")        ║
║                                                                              ║
║  --chords-midi fichero.mid                                                   ║
║      Extrae acordes del track de armonía de un MIDI. Por defecto usa        ║
║      track index 2, canal 1. Ajustable con --chords-midi-track,             ║
║      --chords-midi-channel y --chords-midi-window (ventana de análisis      ║
║      en beats, default 0.5). Usa template matching para identificar         ║
║      el acorde más probable en cada ventana.                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PERFILES EMOCIONALES (--profile)                                            ║
║  ─────────────────────────────────                                           ║
║  Cada perfil configura simultáneamente: modo de escala por defecto,         ║
║  rango de velocidades MIDI, pesos de intervalos para el Markov,             ║
║  multiplicadores de tipo de nota (CT/T/PT/NT), forma del contorno           ║
║  melódico, curva de tensión compás a compás y posición esperada del         ║
║  clímax. Todos son valores por defecto: cualquiera puede sobreescribirse    ║
║  con --mode, --contour, --tension-curve, --range.                           ║
║                                                                              ║
║  Perfil       Escala              Dinámica    Característica principal       ║
║  ──────────── ─────────────────── ─────────── ──────────────────────────    ║
║  heroic       major               f–ff        quintas y octavas, arco       ║
║  melancholic  minor               p–mf        descenso por semitono         ║
║  playful      major               mf–f        saltos cortos, ondulación     ║
║  tense        phrygian            mf–f        semitonos y tritono, ascend.  ║
║  serene       major               p–mf        grados conjuntos, arco suave  ║
║  mysterious   phrygian_dominant   pp–mp       tritonos, evita chord tones   ║
║  triumphant   major               f–ff        quintas, clímax al 85%        ║
║  elegiac      melodic_minor       pp–mf       6ª descendente, clímax 35%    ║
║  ecstatic     lydian              f–fff       #11, euforia desde el inicio  ║
║  brooding     locrian             ppp–mp      semitono orbita, peso oscuro  ║
║  pastoral     mixolydian          p–mf        grados conjuntos, sin drama   ║
║  agitated     chromatic           mf–f        caos, dinámica extrema        ║
║  hypnotic     pentatonic_minor    p–mp        repetición, tensión plana     ║
║  flamenco     phrygian_dominant   pp–fff      cadencia andaluza b2→1        ║
║  tanguero     minor               mf–f        4ª desc., micro-arcos frase   ║
║  custom       major               mf–f        sin sesgo, para uso manual    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CONDICIONAMIENTO EMOCIONAL                                                  ║
║  ──────────────────────────                                                  ║
║  El perfil no es decorativo: actúa en siete capas simultáneas.              ║
║                                                                              ║
║  1. Modo de escala       — universo de notas disponibles                    ║
║  2. Curva de tensión     — t_val[0,1] por compás; modula la intensidad      ║
║                            armónica a lo largo del tiempo                   ║
║  3. Pesos de intervalo   — distribución de saltos en el Markov              ║
║  4. Pesos CT/T/PT/NT     — qué tipo de nota se prefiere sobre cada acorde  ║
║  5. Contorno global      — forma macroscópica de la melodía (arch, desc…)  ║
║  6. Rango de velocidades — dinámica expresiva del instrumento               ║
║  7. Posición del clímax  — criterio de selección del mejor candidato        ║
║                                                                              ║
║  El parámetro t_val (tensión del compás) modula nota a nota:                ║
║    CT_weight *= (1.0 - t_val × 0.3)   ← menos chord tones a más tensión   ║
║    T_weight  *= (1.0 + t_val × 1.5)   ← más tensiones                      ║
║    NT_weight  = t_val × 1.2           ← no-armónicas solo con tensión alta  ║
║  Además, avoid notes en tiempo fuerte reciben ×0.05 independientemente.    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  POST-PROCESADO                                                              ║
║  ──────────────                                                              ║
║  --fix-avoid      Recorre la melodía generada y corrige avoid notes que     ║
║                   caen en tiempos fuertes. Usa nearest_admissible con       ║
║                   preservación de dirección de contorno. Respeta notas de   ║
║                   paso y bordaduras (is_passing_note / is_neighbor_note).   ║
║                                                                              ║
║  --ornaments O    Ornamentación post-generación. Tipos disponibles:         ║
║    appoggiatura — nota disonante (±1-2 st) antes de la nota real,          ║
║                   duración 25% + 75%. Solo en notas ≥ 0.75 beats.          ║
║    passing      — nota de paso escalonada hacia la siguiente nota,          ║
║                   divide la nota en 60%+20%+20%. Solo ≥ 1.0 beat.          ║
║    neighbor     — bordadura superior o inferior y vuelta, 50%+25%+25%.     ║
║    all          — aplica los tres según contexto.                           ║
║  --ornament-prob F  Probabilidad de ornamentar cada nota (default: 0.35).  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  LSTM — DESCRIPCIÓN TÉCNICA                                                  ║
║  ──────────────────────────                                                  ║
║  Problema: aprender el estilo melódico de un corpus propio de MIDIs         ║
║  y generalizarlo a progresiones de acordes nuevas.                          ║
║                                                                              ║
║  Representación — piano roll cuantizado a semicorchea:                      ║
║    Cada compás en 4/4 genera 16 pasos (1 paso = semicorchea = 0.25 beats). ║
║    Cada paso es un token entero en [0, 110]:                                ║
║      0–108  pitch MIDI (A0–C8)                                              ║
║      109    silencio                                                         ║
║      110    hold (la nota anterior continúa)                                ║
║    Ejemplo: una redonda Do4 (MIDI 60) en un compás 4/4 ocupa 16 tokens:    ║
║      [60, 110, 110, 110, 110, 110, 110, 110,                                ║
║       110, 110, 110, 110, 110, 110, 110, 110]                               ║
║                                                                              ║
║  Condicionamiento — chord_vector[16] por paso:                              ║
║    [0:12]  pitch classes activos en el acorde (0.0 / 1.0)                  ║
║    [12:16] función armónica one-hot: T=12, S=13, D=14, ?=15                ║
║    Se calcula estáticamente para toda la progresión antes de generar.      ║
║    No añade parámetros al modelo — es una feature de entrada.               ║
║                                                                              ║
║  Arquitectura (~114K parámetros con defaults):                              ║
║    Embedding(111, 32)        →  emb[B, T, 32]                               ║
║    concat(emb, chord_vec)    →  inp[B, T, 48]   (32 + 16)                  ║
║    LSTM(48 → 128, layers=1)  →  out[B, T, 128]                              ║
║    Linear(128 → 111)         →  logits[B, T, 111]                           ║
║                                                                              ║
║  Entrenamiento — truncated BPTT:                                             ║
║    Cada ventana de 64 pasos se divide en chunks de 32. El estado oculto    ║
║    se mantiene entre chunks pero el grafo de gradientes se corta            ║
║    (detach). Reduce la memoria de gradientes en ~2×.                        ║
║    Con batch=16, chunk=32, hidden=128: ~1MB de activaciones por chunk.      ║
║    Tiempo estimado por época (500 MIDIs augmentados): 3-8 min en CPU.      ║
║                                                                              ║
║  Augmentación del corpus:                                                    ║
║    · Transposición: cada MIDI se transpone ±5 semitonos → ×11 ejemplos.   ║
║      Las transposiciones que llevan notas fuera de [A0, C8] se descartan.  ║
║    · Ventanas solapadas: stride 16 pasos sobre secuencias de 64 pasos.     ║
║      Un MIDI de 32 compases (512 pasos) genera ~28 ventanas por tono.      ║
║    Resultado con 500 MIDIs: ~168K ventanas de entrenamiento.                ║
║                                                                              ║
║  Generación — autoregresiva con sesgo de contorno:                          ║
║    En cada paso, los logits se modifican antes del sampling:                ║
║    · Bias de contorno: penaliza pitches lejos del target de altura          ║
║      (proporcional a la distancia relativa en el rango).                   ║
║    · Bias de escala: penaliza notas fuera de la escala cuando t_val < 0.5. ║
║    · Penaliza hold en el primer tiempo del compás.                          ║
║    · Penaliza saltos > 12 semitonos desde el último pitch real.             ║
║    Sampling: temperature (suaviza/agudiza la distribución) + top-k          ║
║    (limita la búsqueda a los k tokens más probables).                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SCORING (selección del mejor candidato entre N)                             ║
║  ────────────────────────────────────────────────                            ║
║  score_melody devuelve un valor en [0, 1] como media ponderada de:          ║
║    15%  Consonancia con la escala (% notas en la escala)                    ║
║    20%  Armonía con el acorde activo (CT=1.0, T=0.7, PT=0.5, NT=0.1)       ║
║         Avoid notes en tiempo fuerte penalizadas ×0.2                       ║
║    15%  Suavidad (% intervalos ≤ 7 semitonos)                               ║
║    15%  Conformidad con el contorno del perfil                              ║
║    10%  Rango melódico razonable (óptimo: 12 semitonos)                     ║
║    10%  Variedad rítmica (nº de duraciones distintas)                       ║
║     8%  Arco dinámico (desviación estándar de velocidades)                  ║
║     7%  Posición del clímax (ajustada al perfil, p.ej. 0.618 para serene)  ║
║                                                                              ║
║  score_melody_vs_progression (disponible con --verbose) da además:          ║
║    · Score por acorde individual                                             ║
║    · Lista de colisiones (avoid notes en tiempo fuerte con su beat)         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  EJEMPLOS DE USO                                                             ║
║  ───────────────                                                             ║
║                                                                              ║
║  ── Motor chord_guided ──────────────────────────────────────────────────   ║
║                                                                              ║
║  # Básico: progresión en La menor, 8 compases, perfil melancólico           ║
║  python melody_conditioned_v3.py                                             ║
║      --chords "Am:2 G:2 F:2 E7:2" --key Am --bars 8                        ║
║      --profile melancholic                                                   ║
║                                                                              ║
║  # Con ornamentación y corrección de avoid notes                             ║
║  python melody_conditioned_v3.py                                             ║
║      --chords "Dm:2 C:2 Bb:2 A7:2" --key Dm --bars 8                       ║
║      --profile tanguero --ornaments all --ornament-prob 0.4 --fix-avoid     ║
║                                                                              ║
║  # Acordes desde MIDI, incluir armonía en la salida                         ║
║  python melody_conditioned_v3.py                                             ║
║      --chords-midi harmony.mid --key Am --bars 16                           ║
║      --profile flamenco --include-chords                                     ║
║                                                                              ║
║  # Con corpus real para el Markov                                            ║
║  python melody_conditioned_v3.py                                             ║
║      --corpus ./midis/ --chords "C:4 Am:4 F:4 G:4" --key C --bars 16       ║
║      --profile serene --candidates 5                                         ║
║                                                                              ║
║  # Curva de tensión manual (sube, clímax, baja) + contorno personalizado    ║
║  python melody_conditioned_v3.py                                             ║
║      --chords "Em:4 C:4 G:4 D:4" --key Em --bars 16                        ║
║      --tension-curve "0.2,0.3,0.5,0.7,0.9,0.9,0.7,0.5,0.4,0.3"            ║
║      --contour arch --profile heroic                                         ║
║                                                                              ║
║  ── Motor grammar ───────────────────────────────────────────────────────   ║
║                                                                              ║
║  # Elaboración Schenkerian — produce melodías más "clásicas"                 ║
║  python melody_conditioned_v3.py --engine grammar                            ║
║      --chords "C:4 Am:4 F:4 G:4" --key C --bars 8                          ║
║      --profile serene --ornaments passing neighbor                           ║
║                                                                              ║
║  # Modo frigio con contorno descendente                                      ║
║  python melody_conditioned_v3.py --engine grammar                            ║
║      --chords "Am:4 G:4 F:4 E:4" --key Am --mode phrygian --bars 8         ║
║      --profile mysterious --contour descending                               ║
║                                                                              ║
║  ── Motor search ────────────────────────────────────────────────────────   ║
║                                                                              ║
║  # A* con corrección de avoid notes — máxima coherencia armónica local      ║
║  python melody_conditioned_v3.py --engine search                             ║
║      --chords "Cm:2 Ab:2 Eb:2 Bb:2" --key Cm --bars 8                      ║
║      --profile brooding --fix-avoid                                          ║
║                                                                              ║
║  ── Motor genetic ───────────────────────────────────────────────────────   ║
║                                                                              ║
║  # Genético — mejor resultado global a costa de más tiempo                   ║
║  python melody_conditioned_v3.py --engine genetic                            ║
║      --chords "D:4 A:4 Bm:4 G:4" --key D --bars 8                          ║
║      --profile triumphant --candidates 5                                     ║
║                                                                              ║
║  ── Motor lstm — flujo completo ─────────────────────────────────────────   ║
║                                                                              ║
║  # Paso 1: Preprocesar corpus (solo hace falta una vez)                      ║
║  python melody_conditioned_v3.py --preprocess                                ║
║      --corpus ./midis/ --dataset corpus.npz --verbose                        ║
║                                                                              ║
║  # Paso 2: Entrenar                                                          ║
║  python melody_conditioned_v3.py --train                                     ║
║      --dataset corpus.npz --model-out lstm.pt                               ║
║      --epochs 30 --hidden 128 --batch-size 16 --verbose                     ║
║                                                                              ║
║  # Paso 3: Generar con el modelo entrenado                                   ║
║  python melody_conditioned_v3.py --engine lstm                               ║
║      --model lstm.pt                                                         ║
║      --chords "Am:2 G:2 F:2 E7:2" --key Am --bars 8                        ║
║      --profile melancholic --temperature 0.9 --top-k 10                     ║
║                                                                              ║
║  # Todo en un paso (preprocesa si no existe el .npz, entrena y genera)      ║
║  python melody_conditioned_v3.py --engine lstm --train                       ║
║      --corpus ./midis/ --dataset corpus.npz --model-out lstm.pt             ║
║      --chords "Am:2 G:2 F:2 E7:2" --key Am --bars 8                        ║
║                                                                              ║
║  # LSTM con ornamentación y fix-avoid post-generación                        ║
║  python melody_conditioned_v3.py --engine lstm                               ║
║      --model lstm.pt --chords "Dm:2 C:2 Bb:2 A7:2" --key Dm --bars 16      ║
║      --profile tanguero --ornaments passing --fix-avoid                      ║
║      --temperature 0.8 --top-k 15 --candidates 3                            ║
║                                                                              ║
║  # LSTM con modelo más grande (más expresivo, más lento de entrenar)         ║
║  python melody_conditioned_v3.py --train                                     ║
║      --dataset corpus.npz --model-out lstm_large.pt                         ║
║      --hidden 256 --d-embed 64 --num-layers 2 --epochs 50                   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OPCIONES COMPLETAS                                                          ║
║  ──────────────────                                                          ║
║  General:                                                                    ║
║    --engine E          chord_guided|grammar|search|genetic|lstm             ║
║    --key KEY           Tonalidad: C, Am, F#, Bb… (default: C)              ║
║    --mode MODE         Modo de escala (default: auto desde perfil)          ║
║    --bars N            Compases a generar (default: 16)                     ║
║    --tempo BPM         Tempo en BPM (default: 120)                          ║
║    --time-sig S        Compás: 4/4, 3/4, 6/8… (default: 4/4)             ║
║    --profile P         Estado emocional (default: serene)                   ║
║    --rhythm R          Perfil rítmico: flowing|march|syncopated|baroque|    ║
║                        jazz|sparse|dense (default: flowing)                 ║
║    --contour C         Contorno: auto|arch|ascending|descending|wave|       ║
║                        plateau|inverted|erratic|custom (default: auto)      ║
║    --contour-values V  Valores para --contour custom: "0,0.3,0.8,1,0.5"    ║
║    --tension-curve V   Curva de tensión compás a compás: "0.2,0.5,0.8,…"  ║
║    --range LOW HIGH    Rango MIDI (default: 60 84, Do4–Do6)                ║
║    --candidates N      Generar N candidatos y exportar el mejor (def: 3)   ║
║    --seed N            Semilla aleatoria para reproducibilidad (def: 42)    ║
║    --verbose           Informe detallado: config, scores, colisiones        ║
║    --dry-run           Mostrar parámetros resueltos sin generar             ║
║                                                                              ║
║  Fuentes de acordes:                                                         ║
║    --chords TEXT            "Am:2 G:2 F:2 E7:2"                             ║
║    --chords-json FILE        .chords.json                                    ║
║    --chords-midi FILE        MIDI con track de armonía                      ║
║    --chords-midi-track N     Índice del track de armonía (default: 2)       ║
║    --chords-midi-channel N   Canal MIDI de armonía (default: 1)             ║
║    --chords-midi-window F    Ventana de análisis en beats (default: 0.5)    ║
║                                                                              ║
║  Corpus (chord_guided + lstm):                                               ║
║    --corpus DIR        Directorio con MIDIs                                  ║
║    --max-files N       Máximo MIDIs a procesar (0=todos, default: 0)        ║
║                                                                              ║
║  LSTM específico:                                                             ║
║    --train             Entrenar antes de generar (o solo entrenar)           ║
║    --preprocess        Solo preprocesar corpus a .npz                        ║
║    --dataset FILE      Dataset preprocesado (default: corpus.npz)           ║
║    --model-out FILE    Guardar modelo entrenado (default: lstm.pt)           ║
║    --model FILE        Cargar modelo para generar                            ║
║    --epochs N          Épocas de entrenamiento (default: 30)                 ║
║    --hidden N          Tamaño estado oculto (default: 128)                  ║
║    --d-embed N         Dimensión del embedding (default: 32)                ║
║    --num-layers N      Capas LSTM: 1 para CPU, 2 para GPU (default: 1)     ║
║    --batch-size N      Batch size (default: 16)                             ║
║    --chunk-size N      Pasos por chunk BPTT (default: 32)                  ║
║    --lr F              Learning rate (default: 0.001)                       ║
║    --temperature F     Sampling: <1 determinista, >1 aleatorio (def: 1.0)  ║
║    --top-k N           Solo top-k tokens más probables (0=off, def: 10)    ║
║                                                                              ║
║  Post-procesado:                                                             ║
║    --ornaments O [O…]  appoggiatura|passing|neighbor|all                    ║
║    --ornament-prob F   Probabilidad por nota (default: 0.35)                ║
║    --fix-avoid         Corregir avoid notes en tiempos fuertes              ║
║                                                                              ║
║  Exportación:                                                                ║
║    --output FILE       Fichero MIDI de salida (def: melody_conditioned.mid) ║
║    --include-chords    Añadir track de acordes al MIDI de salida            ║
║    --export-motif      Exportar motivo semilla (.motif.mid, 2 compases)     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SALIDAS                                                                     ║
║  ───────                                                                     ║
║  melody_conditioned.mid        Melodía principal (mejor candidato)          ║
║  melody_conditioned_v2.mid     Segundo candidato (si --candidates ≥ 2)     ║
║  melody_conditioned_v3.mid     Tercer candidato  (si --candidates ≥ 3)     ║
║  melody_conditioned.melody.json  Notas, duraciones, score y detalle         ║
║  melody_conditioned.motif.mid  Motivo semilla (con --export-motif)          ║
║                                                                              ║
║  El JSON incluye score global y desglose por criterio:                      ║
║    consonance, harmony, smoothness, contour, range, variety, arc, climax   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DEPENDENCIAS                                                                ║
║  ────────────                                                                ║
║    pip install mido numpy          (siempre requerido)                       ║
║    pip install torch               (solo para --engine lstm)                 ║
║    pip install music21             (opcional: mejora detección de acordes)   ║
║                                                                              ║
║  Formato del corpus MIDI para LSTM:                                          ║
║    Track 0 — metadatos (tempo, time signature)                              ║
║    Track 1 — melodía  (canal 0)                                             ║
║    Track 2 — acordes  (canal 1)                                             ║
║  MIDIs sin track 2 o con menos de 2 pistas se descartan en el preproceso.  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import random
import argparse
import re
import heapq
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# ── Dependencias opcionales ───────────────────────────────────────────────────
try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("[WARN] mido no disponible: pip install mido")


try:
    from music21 import harmony, stream, midi as m21midi
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES  = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC  = {"Db": 1, "Eb": 3, "Fb": 4, "Gb": 6, "Ab": 8, "Bb": 10, "Cb": 11}

SCALE_INTERVALS: Dict[str, List[int]] = {
    "major":             [0, 2, 4, 5, 7, 9, 11],
    "minor":             [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":    [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":     [0, 2, 3, 5, 7, 9, 11],
    "dorian":            [0, 2, 3, 5, 7, 9, 10],
    "phrygian":          [0, 1, 3, 5, 7, 8, 10],
    "phrygian_dominant": [0, 1, 4, 5, 7, 8, 10],
    "lydian":            [0, 2, 4, 6, 7, 9, 11],
    "mixolydian":        [0, 2, 4, 5, 7, 9, 10],
    "locrian":           [0, 1, 3, 5, 6, 8, 10],
    "whole_tone":        [0, 2, 4, 6, 8, 10],
    "pentatonic_major":  [0, 2, 4, 7, 9],
    "pentatonic_minor":  [0, 3, 5, 7, 10],
    "blues":             [0, 3, 5, 6, 7, 10],
    "chromatic":         list(range(12)),
}

CHORD_INTERVALS: Dict[str, List[int]] = {
    "":      [0, 4, 7],
    "m":     [0, 3, 7],
    "7":     [0, 4, 7, 10],
    "M7":    [0, 4, 7, 11],
    "m7":    [0, 3, 7, 10],
    "dim":   [0, 3, 6],
    "dim7":  [0, 3, 6, 9],
    "hdim7": [0, 3, 6, 10],
    "aug":   [0, 4, 8],
    "sus4":  [0, 5, 7],
    "sus2":  [0, 2, 7],
    "9":     [0, 4, 7, 10, 14],
    "M9":    [0, 4, 7, 11, 14],
    "m9":    [0, 3, 7, 10, 14],
    "6":     [0, 4, 7, 9],
    "m6":    [0, 3, 7, 9],
    "add9":  [0, 4, 7, 14],
    "11":    [0, 4, 7, 10, 14, 17],
    "13":    [0, 4, 7, 10, 14, 21],
}

CHORD_TENSIONS: Dict[str, List[int]] = {
    "":      [14, 9, 11],
    "m":     [14, 10, 17],
    "7":     [14, 17, 21, 1],
    "M7":    [14, 9, 18],
    "m7":    [14, 17, 9],
    "dim":   [9, 15],
    "dim7":  [14, 2],
    "hdim7": [14, 17],
    "aug":   [14, 10],
    "sus4":  [14, 10],
    "sus2":  [17, 10],
}

# Notas a evitar (avoid notes) por calidad de acorde — intervalos desde raíz
AVOID_INTERVALS: Dict[str, List[int]] = {
    "":      [1, 6],
    "m":     [6, 11],
    "7":     [6],
    "M7":    [6],
    "m7":    [6],
    "dim":   [1, 8],
    "dim7":  [1],
    "hdim7": [1],
    "aug":   [1, 6],
    "sus4":  [],
    "sus2":  [],
}

HARMONIC_FUNCTION_WEIGHTS = {
    "T":  {"CT": 5.0, "T": 2.0, "PT": 2.5, "NT": 0.3},
    "S":  {"CT": 4.0, "T": 3.0, "PT": 3.0, "NT": 0.5},
    "D":  {"CT": 3.5, "T": 4.0, "PT": 2.5, "NT": 0.8},
    "?":  {"CT": 4.0, "T": 2.5, "PT": 2.5, "NT": 0.5},
}

EMOTIONAL_CHORD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "heroic":      {"CT": 1.3, "T": 0.8, "PT": 0.9, "NT": 0.4},
    "melancholic": {"CT": 1.0, "T": 1.5, "PT": 1.3, "NT": 0.6},
    "playful":     {"CT": 1.2, "T": 1.2, "PT": 1.4, "NT": 0.5},
    "tense":       {"CT": 0.8, "T": 2.0, "PT": 1.2, "NT": 1.0},
    "serene":      {"CT": 1.5, "T": 0.8, "PT": 1.0, "NT": 0.2},
    "mysterious":  {"CT": 0.7, "T": 1.8, "PT": 1.0, "NT": 1.2},
    "triumphant":  {"CT": 1.4, "T": 0.9, "PT": 0.8, "NT": 0.3},
    # ── Perfiles nuevos ───────────────────────────────────────────────────────
    "elegiac":     {"CT": 1.1, "T": 1.6, "PT": 1.4, "NT": 0.5},
    # Luto contenido: más tensiones que melancholic, notas de paso prominentes,
    # chord tones con peso medio (no huye de ellos, pero los rodea de color).
    "ecstatic":    {"CT": 0.9, "T": 2.2, "PT": 1.0, "NT": 0.8},
    # Euforia: tensiones muy altas, chord tones casi evitados — la melodía
    # flota sobre el acorde sin posarse en sus pilares.
    "brooding":    {"CT": 0.6, "T": 1.6, "PT": 1.1, "NT": 1.4},
    # Ensombrecido: más NT que cualquier otro perfil, chord tones muy bajos —
    # la melodía choca constantemente con el acorde de forma controlada.
    "pastoral":    {"CT": 1.6, "T": 0.6, "PT": 1.2, "NT": 0.1},
    # Campo abierto: chord tones dominantes, casi sin no-armónicas. Más
    # "puro" que serene — serene tiene T=0.8, aquí es 0.6.
    "agitated":    {"CT": 0.7, "T": 1.5, "PT": 1.0, "NT": 1.5},
    # Agitado: NT al mismo nivel que T, distribución plana — cualquier nota
    # puede aparecer; el caos es el mensaje.
    "hypnotic":    {"CT": 1.8, "T": 0.5, "PT": 0.8, "NT": 0.1},
    # Hipnótico: chord tones dominantes al máximo, casi sin variación
    # armónica — la melodía orbita una sola nota estructural.
    "flamenco":    {"CT": 1.2, "T": 1.4, "PT": 1.3, "NT": 0.7},
    # Flamenco: chord tones con peso respetable pero rodeados de tensiones
    # y cromatismo (NT=0.7 — el más alto entre los que usan CT normalmente).
    "tanguero":    {"CT": 1.1, "T": 1.3, "PT": 1.2, "NT": 0.6},
    # Tanguero: balance medio-alto en todo; el tango habita ese espacio entre
    # la claridad armónica y la ambigüedad expresiva.
    "custom":      {"CT": 1.0, "T": 1.0, "PT": 1.0, "NT": 0.5},
}

INTERVAL_WEIGHTS: Dict[str, Dict[int, float]] = {
    "heroic":      {0: 0.5, 2: 2.0, 3: 1.5, 4: 2.0, 5: 2.5, 7: 3.0, 12: 2.5,
                    -2: 2.0, -3: 1.0, -5: 1.5, -7: 1.5},
    "melancholic": {0: 0.3, 1: 2.5, 2: 3.0, 3: 2.0, -1: 3.0, -2: 3.5,
                    -3: 2.0, -5: 1.5, 5: 1.0, 7: 0.5},
    "playful":     {0: 0.5, 1: 1.5, 2: 2.5, 3: 2.0, 4: 2.0, 5: 1.5,
                    -1: 1.5, -2: 2.5, -3: 2.0, -4: 1.5, 7: 1.0, -7: 1.0},
    "tense":       {0: 0.5, 1: 3.0, 2: 2.0, 6: 2.0, -1: 3.0, -2: 2.0,
                    -6: 2.0, 11: 1.5, -11: 1.5, 3: 1.0, -3: 1.0},
    "serene":      {0: 1.0, 2: 3.0, 3: 2.0, 4: 2.5, 5: 2.0, -2: 3.0,
                    -3: 2.0, -4: 2.0, -5: 1.5, 7: 1.0, -7: 1.0},
    "mysterious":  {0: 1.0, 1: 2.0, 6: 2.5, -1: 2.0, -6: 2.5, 3: 1.5,
                    -3: 1.5, 8: 1.5, -8: 1.5, 2: 1.0},
    "triumphant":  {0: 0.3, 2: 2.0, 4: 2.5, 5: 2.0, 7: 3.5, 12: 2.0,
                    -2: 1.5, -3: 1.5, -5: 1.5, 3: 2.0},
    # ── Perfiles nuevos ───────────────────────────────────────────────────────
    "elegiac":     {0: 0.4, 2: 2.5, 3: 2.0, -2: 3.0, -3: 2.5, -5: 2.5,
                    9: 2.0, -9: 1.5, 1: 1.5, -1: 2.0, 5: 1.0},
    # 6ª descendente (−9) y 6ª ascendente (9) son característicos del lamento;
    # el semitono ascendente (1) aporta el dolor contenido; descenso por grado
    # dominante como en melancholic pero con más espacio entre notas.
    "ecstatic":    {0: 0.2, 4: 3.0, 5: 2.5, 7: 2.5, 11: 2.0, 12: 2.0,
                    -2: 1.5, -4: 2.0, -5: 1.5, 3: 1.5, 8: 1.5},
    # Saltos amplios ascendentes (7ª mayor, 8ª, 4ª, 5ª); poca repetición (0: 0.2);
    # la 7ª mayor ascendente (11) es el gesto de arrebato sin resolución.
    "brooding":    {0: 1.5, 1: 3.5, -1: 3.0, 2: 1.5, -2: 2.0, 6: 1.5,
                    -6: 2.0, 3: 1.0, -3: 1.5, -5: 1.5},
    # Semitono inferior dominante (−1: 3.0) y superior (1: 3.5) — la melodía
    # orbita una nota sin alejarse; tritono descendente (−6) para el peso oscuro;
    # repetición alta (0: 1.5) — el estancamiento es intencional.
    "pastoral":    {0: 1.2, 2: 3.5, 3: 3.0, 4: 2.0, 5: 2.5, -2: 3.5,
                    -3: 3.0, -4: 1.5, -5: 1.5, 7: 0.8},
    # Movimiento por grado conjunto y terceras puras; sin semitonos ni tritonos;
    # el campo abierto no necesita cromatismo. La quinta (7) aparece pero sin
    # protagonismo — es un horizonte, no un salto dramático.
    "agitated":    {0: 0.3, 1: 2.5, -1: 2.5, 2: 2.0, -2: 2.0, 3: 1.5,
                    -3: 1.5, 6: 2.0, -6: 2.0, 5: 1.5, -5: 1.5, 11: 1.5},
    # Distribución casi plana con ligero sesgo a semitonos y tritono;
    # la 7ª mayor ascendente (11) añade gestos bruscos; la repetición
    # casi eliminada (0: 0.3) — el agitado no se detiene nunca.
    "hypnotic":    {0: 3.0, 2: 2.5, -2: 2.5, 1: 1.0, -1: 1.0, 3: 1.5,
                    -3: 1.5, 5: 1.0, -5: 0.8},
    # Repetición dominante (0: 3.0) — la nota se repite deliberadamente;
    # grado conjunto como movimiento principal; sin saltos de ningún tipo;
    # la melodía hipnótica es casi estática.
    "flamenco":    {0: 0.4, 1: 3.5, -1: 3.0, 2: 2.0, -2: 2.5, 3: 1.5,
                    -3: 2.0, 5: 1.5, -5: 2.0, 4: 1.0, -4: 1.5},
    # Semitono superior dominante (1: 3.5) — la cadencia andaluza b2→1 es
    # el gesto flamenco por excelencia; semitono inferior también presente;
    # 4ª descendente (−5) para los giros de falseta; sin saltos de 8ª.
    "tanguero":    {0: 0.5, 2: 2.5, -2: 2.5, 3: 2.0, -3: 2.0, 5: 2.5,
                    -5: 3.0, 4: 1.5, -4: 2.0, 9: 2.0, -9: 2.5, 1: 1.5},
    # 4ª descendente (−5: 3.0) y 6ª (9/−9: 2.0/2.5) son los saltos del tango;
    # el semitono (1) aparece en los bordones cromáticos; grado conjunto
    # presente pero sin dominar — el tango alterna saltos y pasos.
    "custom":      {0: 1.0, 2: 2.5, -2: 2.5, 3: 1.5, -3: 1.5, 5: 1.5,
                    -5: 1.5, 7: 1.5, -7: 1.0},
}

PROFILE_CONTOUR: Dict[str, str] = {
    "heroic":      "arch",
    "melancholic": "descending",
    "playful":     "wave",
    "tense":       "ascending",
    "serene":      "arch",
    "mysterious":  "erratic",
    "triumphant":  "arch",
    "elegiac":     "descending",   # como melancholic pero con elevaciones internas
    "ecstatic":    "plateau",      # ya está arriba desde casi el principio
    "brooding":    "inverted",     # empieza alto, cae y se queda abajo
    "pastoral":    "plateau",      # plano y abierto, sin narrativa de tensión
    "agitated":    "erratic",      # sin tendencia — el caos no tiene forma
    "hypnotic":    "plateau",      # casi horizontal — el movimiento es mínimo
    "flamenco":    "arch",         # sube hacia el clímax y resuelve
    "tanguero":    "wave",         # micro-arcos de frase (2-4 compases)
    "custom":      "arch",
}

PROFILE_VELOCITY: Dict[str, Tuple[int, int]] = {
    "heroic":      (70, 110),
    "melancholic": (40, 75),
    "playful":     (60, 95),
    "tense":       (55, 100),
    "serene":      (40, 75),
    "mysterious":  (35, 70),
    "triumphant":  (75, 115),
    "elegiac":     (38, 72),    # más contenido que melancholic, rango estrecho
    "ecstatic":    (80, 120),   # el más fuerte — la euforia no se contiene
    "brooding":    (30, 65),    # el más suave — peso sin volumen
    "pastoral":    (45, 80),    # suave pero con más presencia que serene
    "agitated":    (50, 105),   # rango amplísimo — la dinámica es impredecible
    "hypnotic":    (45, 65),    # rango muy estrecho — variación mínima
    "flamenco":    (40, 115),   # del pianissimo al fortissimo — el flamenco es extremo
    "tanguero":    (50, 95),    # medio-alto; el tango es expresivo pero controlado
    "custom":      (50, 90),
}

PROFILE_TO_MODE: Dict[str, str] = {
    "heroic":      "major",
    "melancholic": "minor",
    "playful":     "major",
    "tense":       "phrygian",
    "serene":      "major",
    "mysterious":  "phrygian_dominant",
    "triumphant":  "major",
    "elegiac":     "melodic_minor",      # menor melódico: el lamento con 6ª y 7ª altas
    "ecstatic":    "lydian",             # #11 crea la sensación de flotación sin límites
    "brooding":    "locrian",            # el modo más oscuro: b2, b5, b7
    "pastoral":    "mixolydian",         # mayor con 7ª menor — abierto, sin tensión dominante
    "agitated":    "chromatic",          # todo el cromatismo disponible
    "hypnotic":    "pentatonic_minor",   # solo 5 notas — la repetición necesita poco material
    "flamenco":    "phrygian_dominant",  # el modo andaluz por excelencia
    "tanguero":    "minor",              # menor natural con cromatismo explícito en intervalos
    "custom":      "major",
}

# Posición del clímax por perfil (fracción de la obra)
PROFILE_CLIMAX_POS: Dict[str, float] = {
    "heroic":      0.75,
    "melancholic": 0.40,
    "playful":     0.618,
    "tense":       0.85,
    "serene":      0.618,
    "mysterious":  0.618,
    "triumphant":  0.85,
    "elegiac":     0.35,    # el punto de máximo dolor es temprano, luego resignación
    "ecstatic":    0.20,    # ya está en el clímax casi desde el principio
    "brooding":    0.15,    # el peso más oscuro al principio, sin recuperación
    "pastoral":    0.50,    # centro exacto — equilibrio sin narrativa dramática
    "agitated":    0.60,    # ligeramente post-centro — el caos tiene un punto de ebullición
    "hypnotic":    0.50,    # el clímax no importa — todo es igual de inmóvil
    "flamenco":    0.70,    # el duende llega en el último tercio
    "tanguero":    0.618,   # proporción áurea — el tango es formalmente elegante
    "custom":      0.618,
}

RHYTHM_DISTS: Dict[str, Dict[float, float]] = {
    "flowing":    {0.5: 3.0, 1.0: 4.0, 1.5: 1.5, 2.0: 1.5},
    "march":      {1.0: 5.0, 0.5: 2.0, 2.0: 1.5, 0.25: 1.0},
    "syncopated": {0.5: 3.5, 0.75: 3.0, 1.0: 2.5, 1.5: 2.0, 0.25: 2.0},
    "baroque":    {0.25: 2.0, 0.5: 3.0, 1.0: 3.0, 1.5: 1.5, 2.0: 1.0},
    "jazz":       {0.5: 3.0, 0.75: 3.5, 1.0: 2.5, 1.5: 2.0, 0.25: 1.5},
    "sparse":     {2.0: 3.5, 1.5: 2.0, 3.0: 2.0, 4.0: 1.5, 1.0: 2.0},
    "dense":      {0.25: 4.0, 0.5: 3.5, 0.75: 2.0, 1.0: 1.5},
}

# ── Constantes del piano roll LSTM ───────────────────────────────────────────
PR_STEPS_PER_BEAT  = 4          # semicorcheas por beat (resolución mínima)
PR_PITCH_MIN       = 21         # A0
PR_PITCH_MAX       = 108        # C8
PR_SILENCE         = 109        # token: silencio
PR_HOLD            = 110        # token: nota sostenida
PR_VOCAB           = 111        # tamaño del vocabulario (0-108 pitch + 109 + 110)
PR_CHORD_DIM       = 16         # 12 pitch-classes + 4 harm_func one-hot
PR_MAX_TRANSP      = 5          # transposición máxima en semitonos (±5)


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MelodyNote:
    pitch:    int
    duration: float
    velocity: int
    offset:   float

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12 if self.pitch >= 0 else -1

    @property
    def name(self) -> str:
        if self.pitch < 0:
            return "REST"
        oct_ = self.pitch // 12 - 1
        return f"{NOTE_NAMES[self.pitch % 12]}{oct_}"


@dataclass
class ChordEvent:
    name:        str
    root_pc:     int
    quality:     str
    chord_pcs:   List[int]
    tension_pcs: List[int]
    avoid_pcs:   List[int]
    duration:    float
    offset:      float
    harm_func:   str = "?"


@dataclass
class MelodyResult:
    notes:    List[MelodyNote]
    key:      str
    mode:     str
    bars:     int
    tempo:    int
    engine:   str
    profile:  str
    score:    float = 0.0
    score_detail: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "mode": self.mode, "bars": self.bars,
            "tempo": self.tempo, "engine": self.engine,
            "profile": self.profile, "score": round(self.score, 4),
            "score_detail": {k: round(v, 4) for k, v in self.score_detail.items()},
            "total_notes": len(self.notes),
            "notes": [
                {"pitch": n.pitch, "name": n.name,
                 "duration": n.duration, "velocity": n.velocity,
                 "offset": round(n.offset, 4)}
                for n in self.notes
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES TONALES
# ══════════════════════════════════════════════════════════════════════════════

def parse_note_name(s: str) -> int:
    s = s.strip()
    if s in ENHARMONIC:
        return ENHARMONIC[s]
    base = s[0].upper()
    if base not in NOTE_NAMES:
        return 0
    pc = NOTE_NAMES.index(base)
    for ch in s[1:]:
        if ch == "#":
            pc += 1
        elif ch == "b":
            pc -= 1
    return pc % 12


def parse_key(key_str: str) -> Tuple[int, str]:
    key_str = key_str.strip()
    mode = "major"
    for m in ["harmonic_minor", "melodic_minor", "phrygian_dominant",
              "dorian", "phrygian", "lydian", "mixolydian", "locrian",
              "whole_tone", "pentatonic_major", "pentatonic_minor",
              "blues", "chromatic"]:
        if m.replace("_", " ") in key_str.lower() or m in key_str.lower():
            mode = m
            break
    else:
        if "minor" in key_str.lower() or "menor" in key_str.lower():
            mode = "minor"
        elif "major" in key_str.lower() or "mayor" in key_str.lower():
            mode = "major"
        else:
            root_part = key_str.split()[0]
            if (len(root_part) >= 2
                    and root_part[-1].lower() == "m"
                    and root_part[-2] not in ("#", "b")):
                mode = "minor"

    root_str = key_str.split()[0]
    root_str = re.sub(r'm(ajor|inor)?$', '', root_str, flags=re.IGNORECASE).strip()
    root_pc  = parse_note_name(root_str)
    return root_pc, mode


def get_scale_pitches(root_pc: int, mode: str,
                      low: int = 48, high: int = 96) -> List[int]:
    intervals = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    result = []
    for oct_ in range(11):
        for iv in intervals:
            p = oct_ * 12 + root_pc + iv
            if low <= p <= high:
                result.append(p)
    return sorted(result)


def snap_to_scale(pitch: int, scale: List[int]) -> int:
    if not scale:
        return pitch
    return min(scale, key=lambda p: abs(p - pitch))


def weighted_choice(options: list, weights: list, rng: random.Random):
    total = sum(weights)
    if total <= 0:
        return rng.choice(options)
    r = rng.random() * total
    acc = 0.0
    for opt, w in zip(options, weights):
        acc += w
        if r <= acc:
            return opt
    return options[-1]


def build_contour_curve(shape: str, n: int,
                        custom: Optional[List[float]] = None) -> List[float]:
    t = np.linspace(0, 1, n)
    if shape == "arch":
        return list(np.sin(t * math.pi))
    elif shape == "ascending":
        return list(t)
    elif shape == "descending":
        return list(1.0 - t)
    elif shape == "wave":
        return list(0.5 + 0.5 * np.sin(t * 2 * math.pi))
    elif shape == "plateau":
        c = np.zeros(n)
        s, e = max(1, n // 5), n - max(1, n // 5)
        c[s:e] = 0.8
        c[:s]  = np.linspace(0, 0.8, s)
        c[e:]  = np.linspace(0.8, 0, n - e)
        return list(c)
    elif shape == "inverted":
        return list(1.0 - np.sin(t * math.pi))
    elif shape == "erratic":
        return list(np.random.default_rng(42).random(n))
    elif shape == "custom" and custom:
        return list(np.interp(np.linspace(0, 1, n),
                               np.linspace(0, 1, len(custom)), custom))
    return [0.5] * n


def resize_curve(curve: List[float], n: int) -> List[float]:
    if not curve:
        return [0.5] * n
    if len(curve) == n:
        return curve
    return list(np.interp(np.linspace(0, 1, n),
                           np.linspace(0, 1, len(curve)), curve))


def sample_duration(rhythm: str, beats_left: float,
                    rng: random.Random) -> float:
    dist = RHYTHM_DISTS.get(rhythm, RHYTHM_DISTS["flowing"])
    valid = {d: w for d, w in dist.items() if 0 < d <= beats_left}
    if not valid:
        return min(beats_left, 1.0)
    return weighted_choice(list(valid.keys()), list(valid.values()), rng)


# ══════════════════════════════════════════════════════════════════════════════
#  PARSING DE ACORDES
# ══════════════════════════════════════════════════════════════════════════════

def parse_chord_name(chord_str: str) -> Tuple[int, str]:
    chord_str = chord_str.strip()
    m = re.match(r'^([A-G][#b]?)(.*)', chord_str)
    if not m:
        return 0, ""
    root_str, quality = m.group(1), m.group(2)
    root_pc = parse_note_name(root_str)
    quality = quality.strip("/").strip()
    quality = re.sub(r'/[A-G][#b]?$', '', quality)
    aliases = {
        "maj7": "M7", "Maj7": "M7", "maj": "", "MAJ": "",
        "min7": "m7", "min": "m", "MIN": "m",
        "°": "dim", "ø": "hdim7", "ø7": "hdim7",
        "+": "aug", "sus": "sus4",
    }
    for alias, canonical in aliases.items():
        if quality == alias:
            quality = canonical
            break
    if quality not in CHORD_INTERVALS:
        for q in sorted(CHORD_INTERVALS.keys(), key=len, reverse=True):
            if quality.startswith(q):
                quality = q
                break
        else:
            quality = ""
    return root_pc, quality


def build_chord_event(chord_name: str, duration: float,
                      offset: float, root_pc_key: int,
                      mode: str) -> ChordEvent:
    root_pc, quality = parse_chord_name(chord_name)
    ivs = CHORD_INTERVALS.get(quality, CHORD_INTERVALS[""])
    chord_pcs   = [(root_pc + iv) % 12 for iv in ivs]
    tension_pcs = [(root_pc + iv) % 12 for iv in CHORD_TENSIONS.get(quality, [])]
    avoid_pcs   = [(root_pc + iv) % 12 for iv in AVOID_INTERVALS.get(quality, [])]

    scale_ivs   = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    degree      = (root_pc - root_pc_key) % 12
    harm_func   = "?"
    dom_degrees = {scale_ivs[4] if len(scale_ivs) > 4 else 7}
    sub_degrees = {scale_ivs[3] if len(scale_ivs) > 3 else 5,
                   scale_ivs[1] if len(scale_ivs) > 1 else 2}
    ton_degrees = {0, scale_ivs[2] if len(scale_ivs) > 2 else 4,
                   scale_ivs[5] if len(scale_ivs) > 5 else 9}
    if degree in ton_degrees:
        harm_func = "T"
    elif degree in dom_degrees:
        harm_func = "D"
    elif degree in sub_degrees:
        harm_func = "S"

    return ChordEvent(
        name=chord_name, root_pc=root_pc, quality=quality,
        chord_pcs=chord_pcs, tension_pcs=tension_pcs, avoid_pcs=avoid_pcs,
        duration=duration, offset=offset, harm_func=harm_func,
    )


def parse_chords_text(text: str, root_pc_key: int,
                      mode: str) -> List[ChordEvent]:
    tokens = text.strip().split()
    events = []
    offset = 0.0
    for tok in tokens:
        if ":" in tok:
            name, beats = tok.split(":", 1)
            dur = float(beats)
        else:
            name, dur = tok, 4.0
        ev = build_chord_event(name, dur, offset, root_pc_key, mode)
        events.append(ev)
        offset += dur
    return events


def parse_chords_json(path: str, root_pc_key: int,
                      mode: str) -> List[ChordEvent]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chords_raw = []
    if isinstance(data, list):
        chords_raw = data
    elif "chords" in data:
        chords_raw = data["chords"]
    elif "progression" in data:
        chords_raw = data["progression"]
    events = []
    offset = 0.0
    for item in chords_raw:
        if isinstance(item, str):
            name, dur = (item.split(":") + ["4.0"])[:2]
            dur = float(dur)
        elif isinstance(item, dict):
            name = item.get("name", item.get("chord", "C"))
            dur  = float(item.get("duration", item.get("beats", 4.0)))
        else:
            continue
        ev = build_chord_event(name.strip(), dur, offset, root_pc_key, mode)
        events.append(ev)
        offset += dur
    return events


def chord_events_to_text(events: List[ChordEvent]) -> str:
    return " ".join(f"{e.name}:{e.duration}" for e in events)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE ACORDES DESDE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _chord_pcs_to_name(pcs: List[int]) -> Tuple[str, int, str]:
    if not pcs:
        return "C", 0, ""
    pcs_set = set(pcs)
    best_name, best_root, best_quality = "C", 0, ""
    best_score = -1
    for root_pc in range(12):
        for quality, ivs in CHORD_INTERVALS.items():
            template = {(root_pc + iv) % 12 for iv in ivs}
            matches = len(pcs_set & template)
            extras  = len(pcs_set - template)
            missing = len(template - pcs_set)
            score   = matches * 2 - extras * 0.5 - missing * 1.0
            if score > best_score:
                best_score   = score
                best_root    = root_pc
                best_quality = quality
                best_name    = NOTE_NAMES[root_pc] + quality
    return best_name, best_root, best_quality


def extract_chords_from_midi(midi_path: str, root_pc_key: int,
                              mode: str,
                              harmony_track_idx: int = 2,
                              harmony_channel: int = 1,
                              min_window_beats: float = 0.5
                              ) -> List[ChordEvent]:
    if not MIDO_OK:
        raise RuntimeError("mido requerido: pip install mido")

    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat or 480

    target_track = None
    if harmony_track_idx < len(mid.tracks):
        target_track = mid.tracks[harmony_track_idx]
    else:
        for tr in mid.tracks:
            for msg in tr:
                if hasattr(msg, "channel") and msg.channel == harmony_channel:
                    target_track = tr
                    break
            if target_track:
                break

    if target_track is None:
        raise ValueError(f"No se encontró track de armonía en {midi_path}")

    notes: List[Tuple[float, float, int]] = []
    current_tick = 0
    active: Dict[int, float] = {}

    for msg in target_track:
        current_tick += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = current_tick
        elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                on_tick = active.pop(msg.note)
                notes.append((on_tick / tpb, current_tick / tpb, msg.note))

    if not notes:
        dur_total = max(4.0, sum(
            msg.time for msg in target_track if not msg.is_meta) / tpb)
        return [build_chord_event(NOTE_NAMES[root_pc_key], dur_total,
                                  0.0, root_pc_key, mode)]

    total_duration = max(off for _, off, _ in notes)
    n_windows = max(1, int(math.ceil(total_duration / min_window_beats)))
    window_pcs: List[List[int]] = [[] for _ in range(n_windows)]

    for on_q, off_q, pitch in notes:
        pc = pitch % 12
        win_start = int(on_q / min_window_beats)
        win_end   = max(win_start + 1, int(math.ceil(off_q / min_window_beats)))
        for w in range(win_start, min(win_end, n_windows)):
            if pc not in window_pcs[w]:
                window_pcs[w].append(pc)

    window_chords: List[Tuple[str, int, str]] = []
    for pcs in window_pcs:
        if pcs:
            window_chords.append(_chord_pcs_to_name(pcs))
        else:
            window_chords.append(window_chords[-1] if window_chords else ("C", 0, ""))

    events: List[ChordEvent] = []
    current_name, _, _ = window_chords[0]
    current_start  = 0.0
    current_length = min_window_beats

    for i in range(1, len(window_chords)):
        name, _, _ = window_chords[i]
        if name == current_name:
            current_length += min_window_beats
        else:
            events.append(build_chord_event(current_name, current_length,
                                             current_start, root_pc_key, mode))
            current_name   = name
            current_start += current_length
            current_length = min_window_beats

    events.append(build_chord_event(current_name, current_length,
                                     current_start, root_pc_key, mode))
    return events


# ══════════════════════════════════════════════════════════════════════════════
#  CLASIFICACIÓN ARMÓNICA Y CONTEXTUAL
# ══════════════════════════════════════════════════════════════════════════════

def get_chord_at(events: List[ChordEvent], offset: float) -> ChordEvent:
    for ev in reversed(events):
        if ev.offset <= offset:
            return ev
    return events[0] if events else ChordEvent(
        "C", 0, "", [0, 4, 7], [10, 14], [1, 6], 4.0, 0.0, "T")


def chord_note_type(pitch: int, chord: ChordEvent,
                    scale_pitches: List[int]) -> str:
    """CT | T | PT | NT — sin contexto."""
    pc = pitch % 12
    if pc in chord.chord_pcs:
        return "CT"
    if pc in chord.tension_pcs:
        return "T"
    if pitch in scale_pitches:
        return "PT"
    return "NT"


def is_avoid_note(pitch: int, chord: ChordEvent) -> bool:
    """True si la nota es avoid note para el acorde."""
    return pitch % 12 in chord.avoid_pcs


def is_strong_beat(beat_offset: float, beats_per_bar: float) -> bool:
    """True si el offset cae en tiempo fuerte (1º o mitad de compás)."""
    pos = beat_offset % beats_per_bar
    if pos < 0.25:
        return True
    if beats_per_bar >= 4 and abs(pos - beats_per_bar / 2) < 0.25:
        return True
    return False


def is_passing_note(notes: List[MelodyNote], idx: int,
                    beats_per_bar: float) -> bool:
    """
    Heurística: nota de paso si está en tiempo débil y las notas
    anterior y siguiente se mueven por grado conjunto en la misma dirección.
    """
    if idx == 0 or idx >= len(notes) - 1:
        return False
    prev_p = notes[idx - 1].pitch
    curr_p = notes[idx].pitch
    next_p = notes[idx + 1].pitch
    step_in  = curr_p - prev_p
    step_out = next_p - curr_p
    if abs(step_in) > 2 or abs(step_out) > 2:
        return False
    if step_in == 0 or step_out == 0:
        return False
    if (step_in > 0) != (step_out > 0):
        return False
    return not is_strong_beat(notes[idx].offset, beats_per_bar)


def is_neighbor_note(notes: List[MelodyNote], idx: int,
                     beats_per_bar: float) -> bool:
    """
    Heurística: bordadura si el intervalo de entrada y salida son ≤2 semitonos
    y la nota anterior y siguiente son cercanas; en tiempo débil.
    """
    if idx == 0 or idx >= len(notes) - 1:
        return False
    prev_p = notes[idx - 1].pitch
    curr_p = notes[idx].pitch
    next_p = notes[idx + 1].pitch
    if abs(curr_p - prev_p) > 2 or abs(next_p - curr_p) > 2:
        return False
    if abs(next_p - prev_p) > 2:
        return False
    return not is_strong_beat(notes[idx].offset, beats_per_bar)


def nearest_chord_tone(pitch: int, chord: ChordEvent,
                       direction_hint: int = 0) -> int:
    """Chord tone más cercano. direction_hint: +1 arriba, -1 abajo, 0 más cercano."""
    ivs = CHORD_INTERVALS.get(chord.quality, [0, 4, 7])
    candidates = []
    for octave in range(2, 8):
        for iv in ivs:
            candidates.append(chord.root_pc + iv + octave * 12)
    if not candidates:
        return pitch
    in_range = [c for c in candidates if abs(c - pitch) <= 12]
    pool = in_range if in_range else candidates
    if direction_hint == 0:
        return min(pool, key=lambda c: abs(c - pitch))
    elif direction_hint > 0:
        above = [c for c in pool if c >= pitch]
        return min(above, key=lambda c: c - pitch) if above else min(pool, key=lambda c: abs(c - pitch))
    else:
        below = [c for c in pool if c <= pitch]
        return max(below, key=lambda c: pitch - c) if below else min(pool, key=lambda c: abs(c - pitch))


def nearest_admissible(pitch: int, chord: ChordEvent,
                       scale_pitches: List[int],
                       direction_hint: int = 0) -> int:
    """
    Nota admisible más cercana (CT o T, no avoid).
    Preserva el contorno con direction_hint.
    """
    admissible_pcs = set(chord.chord_pcs) | set(chord.tension_pcs)
    candidates = []
    for octave in range(2, 8):
        for pc in admissible_pcs:
            p = pc + octave * 12
            if p in scale_pitches or pc in chord.chord_pcs:
                candidates.append(p)
    if not candidates:
        return nearest_chord_tone(pitch, chord, direction_hint)
    in_range = [c for c in candidates if abs(c - pitch) <= 12]
    pool = in_range if in_range else candidates
    if direction_hint == 0:
        return min(pool, key=lambda c: abs(c - pitch))
    elif direction_hint > 0:
        above = [c for c in pool if c >= pitch]
        return min(above, key=lambda c: c - pitch) if above else min(pool, key=lambda c: abs(c - pitch))
    else:
        below = [c for c in pool if c <= pitch]
        return max(below, key=lambda c: pitch - c) if below else min(pool, key=lambda c: abs(c - pitch))


def get_note_weights_for_chord(chord: ChordEvent, profile: str,
                                scale_pitches: List[int],
                                pitch_range: Tuple[int, int],
                                tension_val: float,
                                beat_offset: float,
                                beats_per_bar: float) -> Dict[int, float]:
    """
    Peso de cada pitch en el rango: función armónica × emoción × tensión.
    Nueva: penaliza avoid notes más fuerte en tiempos fuertes.
    """
    low, high = pitch_range
    func_weights = HARMONIC_FUNCTION_WEIGHTS.get(chord.harm_func,
                                                   HARMONIC_FUNCTION_WEIGHTS["?"])
    emo_mult = EMOTIONAL_CHORD_WEIGHTS.get(profile, EMOTIONAL_CHORD_WEIGHTS["custom"])
    strong   = is_strong_beat(beat_offset, beats_per_bar)

    tension_bonus = {
        "CT": 1.0 - tension_val * 0.3,
        "T":  1.0 + tension_val * 1.5,
        "PT": 1.0 + tension_val * 0.5,
        "NT": tension_val * 1.2,
    }

    weights: Dict[int, float] = {}
    for p in range(low, high + 1):
        ntype = chord_note_type(p, chord, scale_pitches)
        w = func_weights[ntype] * emo_mult[ntype] * max(0.01, tension_bonus[ntype])
        # Penalizar avoid notes en tiempo fuerte
        if is_avoid_note(p, chord) and strong:
            w *= 0.05
        weights[p] = w

    return weights


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESADO: CORRECCIÓN DE AVOID NOTES
# ══════════════════════════════════════════════════════════════════════════════

def fix_avoid_notes(notes: List[MelodyNote],
                    chord_events: List[ChordEvent],
                    scale_pitches: List[int],
                    beats_per_bar: float) -> List[MelodyNote]:
    """
    Corrige avoid notes en tiempos fuertes usando nearest_admissible
    con preservación de contorno (direction_hint desde contexto local).
    Notas de paso y bordaduras se respetan aunque sean avoid.
    """
    result = []
    for idx, n in enumerate(notes):
        chord = get_chord_at(chord_events, n.offset)
        if (is_avoid_note(n.pitch, chord)
                and is_strong_beat(n.offset, beats_per_bar)
                and not is_passing_note(notes, idx, beats_per_bar)
                and not is_neighbor_note(notes, idx, beats_per_bar)):
            # Determinar dirección de contorno desde contexto
            if idx > 0:
                delta = n.pitch - notes[idx - 1].pitch
                hint  = 1 if delta >= 0 else -1
            else:
                hint = 0
            new_pitch = nearest_admissible(n.pitch, chord, scale_pitches, hint)
            new_pitch = max(21, min(108, new_pitch))
            result.append(MelodyNote(new_pitch, n.duration, n.velocity, n.offset))
        else:
            result.append(n)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESADO: ORNAMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _passing_notes_between(p1: int, p2: int,
                            scale_pitches: List[int]) -> List[int]:
    direction = 1 if p2 > p1 else -1
    result = []
    for sp in scale_pitches:
        if direction == 1 and p1 < sp < p2:
            result.append(sp)
        elif direction == -1 and p2 < sp < p1:
            result.append(sp)
    return sorted(result, key=lambda x: direction * x)


def _neighbor_pitch(pitch: int, scale_pitches: List[int],
                    upper: bool = True) -> Optional[int]:
    if pitch not in scale_pitches:
        return None
    idx = scale_pitches.index(pitch)
    if upper and idx + 1 < len(scale_pitches):
        return scale_pitches[idx + 1]
    elif not upper and idx - 1 >= 0:
        return scale_pitches[idx - 1]
    return None


def apply_ornaments(notes: List[MelodyNote],
                    chord_events: List[ChordEvent],
                    scale_pitches: List[int],
                    beats_per_bar: float,
                    ornament_types: List[str],
                    ornament_prob: float = 0.35,
                    rng: Optional[random.Random] = None) -> List[MelodyNote]:
    """
    Ornamentación post-generación: appoggiatura, passing, neighbor.
    Solo se aplica a notas con duración suficiente (≥ 0.75 beats)
    que caigan en tiempos apropiados.
    """
    if not ornament_types or not notes:
        return notes
    if rng is None:
        rng = random.Random(42)
    if "all" in ornament_types:
        ornament_types = ["appoggiatura", "passing", "neighbor"]

    ornamented: List[MelodyNote] = []

    for idx, note in enumerate(notes):
        chord = get_chord_at(chord_events, note.offset)

        # Condiciones para ornamentar
        if (note.duration < 0.75
                or rng.random() > ornament_prob
                or chord_note_type(note.pitch, chord, scale_pitches) == "NT"):
            ornamented.append(note)
            continue

        chosen = rng.choice(ornament_types)
        replacement = None

        if chosen == "appoggiatura" and idx + 1 < len(notes):
            # Bordadura disonante en el primer tiempo → resolución al chord tone
            appog_pitch = nearest_chord_tone(note.pitch + rng.choice([-1, 1, -2, 2]),
                                              chord, direction_hint=0)
            if appog_pitch != note.pitch and abs(appog_pitch - note.pitch) <= 2:
                dur1 = note.duration * 0.25
                dur2 = note.duration * 0.75
                replacement = [
                    MelodyNote(appog_pitch, dur1, note.velocity - 5, note.offset),
                    MelodyNote(note.pitch, dur2, note.velocity, note.offset + dur1),
                ]

        elif chosen == "passing" and idx + 1 < len(notes):
            next_pitch = notes[idx + 1].pitch
            passing = _passing_notes_between(note.pitch, next_pitch, scale_pitches)
            if passing and note.duration >= 1.0:
                p = passing[0]
                dur1 = note.duration * 0.6
                dur2 = note.duration * 0.2
                dur3 = note.duration * 0.2
                replacement = [
                    MelodyNote(note.pitch, dur1, note.velocity, note.offset),
                    MelodyNote(p, dur2, note.velocity - 10, note.offset + dur1),
                    MelodyNote(next_pitch, dur3, note.velocity - 5,
                               note.offset + dur1 + dur2),
                ]
                # Marcar la siguiente nota como absorbida
                if idx + 1 < len(notes):
                    notes[idx + 1] = MelodyNote(
                        notes[idx + 1].pitch, 0.0,
                        notes[idx + 1].velocity, notes[idx + 1].offset)

        elif chosen == "neighbor":
            upper = rng.random() > 0.5
            nb = _neighbor_pitch(note.pitch, scale_pitches, upper)
            if nb is not None and note.duration >= 0.75:
                dur1 = note.duration * 0.5
                dur2 = note.duration * 0.25
                dur3 = note.duration * 0.25
                replacement = [
                    MelodyNote(note.pitch, dur1, note.velocity, note.offset),
                    MelodyNote(nb, dur2, note.velocity - 10, note.offset + dur1),
                    MelodyNote(note.pitch, dur3, note.velocity - 5,
                               note.offset + dur1 + dur2),
                ]

        if replacement:
            ornamented.extend(replacement)
        else:
            ornamented.append(note)

    # Filtrar notas con duración 0 (absorbidas) y reordenar
    ornamented = [n for n in ornamented if n.duration > 0.01]
    ornamented.sort(key=lambda n: n.offset)
    return ornamented


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 1: CHORD_GUIDED (Markov + corpus)
# ══════════════════════════════════════════════════════════════════════════════

class ChordGuidedEngine:
    """
    Markov de 2º orden condicionado a acordes.
    Opcionalmente entrenado desde corpus MIDI real (train_from_corpus).
    """

    def __init__(self, profile: str, rhythm: str, rng: random.Random):
        self.profile = profile
        self.rhythm  = rhythm
        self.rng     = rng
        self._markov: Dict[Tuple, List[Tuple]] = defaultdict(list)
        self._markov_trained = False

    def _dur_bucket(self, d: float) -> int:
        return 0 if d <= 0.5 else (1 if d <= 1.5 else 2)

    # ── Entrenamiento desde corpus ────────────────────────────────────────────

    def train_from_midi(self, midi_path: str) -> int:
        """Aprende transiciones desde un MIDI. Devuelve nº de transiciones."""
        if not MIDO_OK:
            return 0
        try:
            mid = mido.MidiFile(midi_path)
        except Exception:
            return 0
        notes = []
        tpb = mid.ticks_per_beat or 480
        for track in mid.tracks:
            current_time = 0
            pending: Dict[int, Tuple[int, int]] = {}
            for msg in track:
                current_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    pending[msg.note] = (current_time, msg.velocity)
                elif msg.type in ("note_off",) or (
                        msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in pending:
                        on_t, vel = pending.pop(msg.note)
                        dur_q = (current_time - on_t) / tpb
                        notes.append((on_t / tpb, msg.note, dur_q))
        notes.sort(key=lambda n: n[0])
        count = 0
        for i in range(len(notes) - 2):
            iv1 = notes[i+1][1] - notes[i][1]
            iv2 = notes[i+2][1] - notes[i+1][1]
            db1 = self._dur_bucket(notes[i][2])
            db2 = self._dur_bucket(notes[i+1][2])
            db3 = self._dur_bucket(notes[i+2][2])
            self._markov[(iv1, db1, db2)].append((iv2, db3))
            count += 1
        if count > 0:
            self._markov_trained = True
        return count

    def train_from_corpus(self, corpus_dir: str, verbose: bool = False):
        """Entrena desde todos los MIDIs de un directorio."""
        total = 0
        path  = Path(corpus_dir)
        files = list(path.rglob("*.mid")) + list(path.rglob("*.midi"))
        for f in files:
            n = self.train_from_midi(str(f))
            if verbose and n > 0:
                print(f"  [corpus] {f.name}: {n} transiciones")
            total += n
        if verbose:
            print(f"  [corpus] Total: {total} transiciones, {len(files)} archivos")
        self._markov_trained = (total > 0)

    def _build_markov_synthetic(self):
        """Tabla de Markov sintética desde el perfil si no hay corpus."""
        iw  = INTERVAL_WEIGHTS.get(self.profile, INTERVAL_WEIGHTS["serene"])
        ivs, iws = list(iw.keys()), list(iw.values())
        rd  = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])
        prev_db = 1
        for _ in range(300):
            iv  = weighted_choice(ivs, iws, self.rng)
            dur = weighted_choice(list(rd.keys()), list(rd.values()), self.rng)
            db  = self._dur_bucket(dur)
            niv = weighted_choice(ivs, iws, self.rng)
            ndb = self._dur_bucket(
                weighted_choice(list(rd.keys()), list(rd.values()), self.rng))
            self._markov[(iv, prev_db, db)].append((niv, ndb))
            prev_db = db
        self._markov_trained = True

    def _markov_next(self, prev_iv: int, prev_db: int,
                     cur_db: int, tension_val: float) -> int:
        if not self._markov_trained:
            self._build_markov_synthetic()
        key = (prev_iv, prev_db, cur_db)
        candidates = self._markov.get(key, [])
        if not candidates:
            for k, v in self._markov.items():
                if k[2] == cur_db:
                    candidates.extend(v)
        iw  = INTERVAL_WEIGHTS.get(self.profile, INTERVAL_WEIGHTS["serene"])
        ivs, iws = list(iw.keys()), list(iw.values())
        base_iv = (self.rng.choice(candidates)[0]
                   if candidates else weighted_choice(ivs, iws, self.rng))
        if tension_val > 0.6:
            tension_ivs = [iv for iv in ivs if abs(iv) >= 3]
            tension_iws = [iw[iv] for iv in tension_ivs]
            if tension_ivs and self.rng.random() < tension_val * 0.6:
                base_iv = weighted_choice(tension_ivs, tension_iws, self.rng)
        return base_iv

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        if not self._markov_trained:
            self._build_markov_synthetic()

        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        total_beats = n_bars * beats_per_bar

        first_chord = chord_events[0] if chord_events else None
        if first_chord and first_chord.chord_pcs:
            mid_range = (low + high) // 2
            start_candidates = [p for p in range(low, high + 1)
                                 if p % 12 in first_chord.chord_pcs]
            current_pitch = (min(start_candidates, key=lambda p: abs(p - mid_range))
                             if start_candidates
                             else snap_to_scale(mid_range, scale_pitches))
        else:
            current_pitch = snap_to_scale((low + high) // 2, scale_pitches)

        notes: List[MelodyNote] = []
        offset = 0.0
        prev_iv, prev_db = 0, 1
        bar_idx = 0

        while offset < total_beats - 0.01:
            bar_offset = 0.0
            t_val = tension[min(bar_idx, len(tension) - 1)]
            c_val = contour[min(bar_idx, len(contour) - 1)]

            while bar_offset < beats_per_bar - 0.01:
                beats_left = min(beats_per_bar - bar_offset, total_beats - offset)
                dur = sample_duration(self.rhythm, beats_left, self.rng)
                db  = self._dur_bucket(dur)
                cur_offset = offset + bar_offset
                chord = get_chord_at(chord_events, cur_offset)

                note_weights = get_note_weights_for_chord(
                    chord, self.profile, scale_pitches, pitch_range,
                    t_val, cur_offset, beats_per_bar)

                iv = self._markov_next(prev_iv, prev_db, db, t_val)

                cur_rel = (current_pitch - low) / max(high - low, 1)
                if cur_rel > c_val + 0.15 and iv > 0:
                    iv = -abs(iv)
                elif cur_rel < c_val - 0.15 and iv < 0:
                    iv = abs(iv)

                max_leap = int(3 + t_val * 9)
                if abs(iv) > max_leap:
                    iv = int(math.copysign(max_leap, iv))

                candidate = max(low, min(high, current_pitch + iv))
                neighborhood = range(max(low, candidate - 3),
                                     min(high, candidate + 3) + 1)
                nbr_weights = [note_weights.get(p, 0.01) for p in neighborhood]
                new_pitch = (weighted_choice(list(neighborhood), nbr_weights, self.rng)
                             if any(w > 0 for w in nbr_weights)
                             else snap_to_scale(candidate, scale_pitches))

                climax = math.sin((cur_offset / total_beats) * math.pi) * t_val
                vel = max(vel_min, min(vel_max,
                          int(vel_min + (vel_max - vel_min) * max(0.0, climax))))

                notes.append(MelodyNote(new_pitch, dur, vel, round(cur_offset, 4)))
                prev_iv, prev_db = new_pitch - current_pitch, db
                current_pitch = new_pitch
                bar_offset += dur

            offset += beats_per_bar
            bar_idx += 1

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 2: GRAMMAR (Schenkerian condicionado a acordes)
# ══════════════════════════════════════════════════════════════════════════════

class GrammarEngine:
    """
    Elaboración Schenkerian en tres capas anclada al acorde activo.

    Capa 1 — Estructura de frase: para cada compás elige el chord tone
             del acorde activo más cercano al valor objetivo de contorno.
    Capa 2 — Notas de paso: rellena el movimiento entre notas estructurales
             con notas de paso diatónicas.
    Capa 3 — Ornamentos: bordaduras o appoggiaturas en tiempos de alta tensión.
    """

    def __init__(self, root_pc: int, mode: str, rng: random.Random):
        self.root_pc = root_pc
        self.mode    = mode
        self.rng     = rng

    def _chord_tone_near_target(self, chord: ChordEvent,
                                 target_pitch: int,
                                 scale_pitches: List[int],
                                 pitch_range: Tuple[int, int]) -> int:
        """Chord tone del acorde activo más cercano al target."""
        low, high = pitch_range
        candidates = [
            p for p in range(low, high + 1)
            if p % 12 in chord.chord_pcs and p in scale_pitches
        ]
        if not candidates:
            candidates = [p for p in range(low, high + 1)
                          if p % 12 in chord.chord_pcs]
        if not candidates:
            return snap_to_scale(target_pitch, scale_pitches)
        return min(candidates, key=lambda p: abs(p - target_pitch))

    def _passing_notes(self, p1: int, p2: int,
                       scale_pitches: List[int]) -> List[int]:
        direction = 1 if p2 > p1 else -1
        result = [sp for sp in scale_pitches
                  if (direction == 1 and p1 < sp < p2)
                  or (direction == -1 and p2 < sp < p1)]
        return sorted(result, key=lambda x: direction * x)

    def _upper_neighbor(self, pitch: int, scale_pitches: List[int]) -> int:
        if pitch in scale_pitches:
            idx = scale_pitches.index(pitch)
            if idx + 1 < len(scale_pitches):
                return scale_pitches[idx + 1]
        return pitch + 1

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get("serene", (40, 75))
        total_beats = n_bars * beats_per_bar

        # ── Capa 1: nota estructural por compás (chord tone + contorno) ──────
        struct_pitches: List[int] = []
        for bar_idx in range(n_bars):
            c_val  = contour[min(bar_idx, len(contour) - 1)]
            target = low + int(c_val * (high - low))
            chord  = get_chord_at(chord_events, bar_idx * beats_per_bar)
            sp = self._chord_tone_near_target(chord, target, scale_pitches,
                                               pitch_range)
            struct_pitches.append(sp)

        # ── Capa 2: relleno compás a compás ──────────────────────────────────
        notes: List[MelodyNote] = []
        offset = 0.0

        for bar_idx in range(n_bars):
            struct = struct_pitches[bar_idx]
            next_s = struct_pitches[min(bar_idx + 1, n_bars - 1)]
            t_val  = tension[min(bar_idx, len(tension) - 1)]
            chord  = get_chord_at(chord_events, offset)

            vel_base = int(vel_min + (vel_max - vel_min) *
                           math.sin((offset / total_beats) * math.pi) * t_val)
            vel_base = max(vel_min, min(vel_max, vel_base))

            if t_val < 0.3:
                # Simple: nota larga
                notes.append(MelodyNote(struct, beats_per_bar,
                                        vel_base, round(offset, 4)))

            elif t_val < 0.6:
                # Medio: estructura + notas de paso
                passing = self._passing_notes(struct, next_s, scale_pitches)
                if passing and beats_per_bar >= 2.0:
                    dur1 = beats_per_bar * 0.5
                    notes.append(MelodyNote(struct, dur1, vel_base,
                                            round(offset, 4)))
                    n_pass = min(len(passing), int(beats_per_bar * 0.5 / 0.25))
                    dur_pass = (beats_per_bar * 0.5) / max(n_pass, 1)
                    for i, p in enumerate(passing[:n_pass]):
                        notes.append(MelodyNote(
                            p, dur_pass, vel_base - 8,
                            round(offset + dur1 + i * dur_pass, 4)))
                else:
                    dur1 = beats_per_bar * 0.6
                    dur2 = beats_per_bar * 0.4
                    notes.append(MelodyNote(struct, dur1, vel_base,
                                            round(offset, 4)))
                    notes.append(MelodyNote(next_s, dur2, vel_base - 5,
                                            round(offset + dur1, 4)))

            else:
                # Alta tensión: bordadura + estructura + notas de paso
                upper_nb = self._upper_neighbor(struct, scale_pitches)
                passing  = self._passing_notes(struct, next_s, scale_pitches)

                beat = 0.0
                # Appoggiatura
                if beats_per_bar >= 1.0:
                    notes.append(MelodyNote(upper_nb, 0.25,
                                            vel_base - 8, round(offset + beat, 4)))
                    beat += 0.25
                # Nota estructural
                dur_struct = min(0.5, beats_per_bar - beat - 0.01)
                notes.append(MelodyNote(struct, max(0.25, dur_struct),
                                        vel_base, round(offset + beat, 4)))
                beat += max(0.25, dur_struct)
                # Notas de paso
                for p in passing[:2]:
                    if beat < beats_per_bar - 0.01:
                        notes.append(MelodyNote(p, 0.25, vel_base - 12,
                                                round(offset + beat, 4)))
                        beat += 0.25
                # Relleno
                while beat < beats_per_bar - 0.01:
                    notes.append(MelodyNote(struct, 0.25, vel_base - 5,
                                            round(offset + beat, 4)))
                    beat += 0.25

            offset += beats_per_bar

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 3: SEARCH (Beam A* condicionado a acordes)
# ══════════════════════════════════════════════════════════════════════════════

class SearchEngine:
    """
    Beam search A* con función de coste armónico.
    Garantías locales por nota: penaliza avoid notes en tiempos fuertes,
    premia chord tones, penaliza saltos grandes y desviaciones de contorno.
    """

    def __init__(self, profile: str, rng: random.Random):
        self.profile = profile
        self.rng     = rng

    def _local_cost(self, from_pitch: int, to_pitch: int,
                    duration: float, offset: float, total_beats: float,
                    chord: ChordEvent, scale_pitches: List[int],
                    contour_val: float, tension_val: float,
                    pitch_range: Tuple[int, int],
                    beats_per_bar: float) -> float:
        low, high = pitch_range
        cost = 0.0
        iv   = abs(to_pitch - from_pitch)
        ntype = chord_note_type(to_pitch, chord, scale_pitches)
        strong = is_strong_beat(offset, beats_per_bar)

        # Nota fuera de escala
        if to_pitch not in scale_pitches:
            cost += 2.5

        # Avoid note en tiempo fuerte
        if is_avoid_note(to_pitch, chord) and strong:
            cost += 5.0

        # Premio por tipo armónico
        cost -= {"CT": 1.5, "T": 0.8, "PT": 0.3, "NT": 0.0}[ntype]

        # Fuera de rango
        if to_pitch < low or to_pitch > high:
            cost += 10.0

        # Salto muy grande
        if iv > 12:
            cost += (iv - 12) * 0.5
        if iv == 0:
            cost += 1.5

        # Consonancias (3ª, 5ª, 6ª) desde perfil
        iv_mod = iv % 12
        if iv_mod in (3, 4, 7, 8, 9):
            cost -= 0.3

        # Contorno
        rel_pos = (to_pitch - low) / max(high - low, 1)
        cost += abs(rel_pos - contour_val) * 0.8

        # Tensión: más NT permitida a mayor tensión
        if ntype == "NT" and tension_val < 0.4:
            cost += 2.0

        return cost

    def _heuristic(self, pitch: int, offset: float,
                   total_beats: float) -> float:
        remaining = 1.0 - offset / max(total_beats, 1)
        return remaining * 0.1

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        total_beats = n_bars * beats_per_bar
        beam_width  = 5

        start_pitch = snap_to_scale(low + (high - low) // 3, scale_pitches)
        beam: List[Tuple[float, float, int, List[MelodyNote]]] = [
            (0.0, 0.0, start_pitch, [])
        ]

        while beam:
            beam.sort(key=lambda x: x[0])
            best_cost, offset, cur_pitch, notes_so_far = beam[0]

            if offset >= total_beats - 0.01:
                return notes_so_far

            bar_idx = int(offset // beats_per_bar)
            c_val   = contour[min(bar_idx, len(contour) - 1)]
            t_val   = tension[min(bar_idx, len(tension) - 1)]
            chord   = get_chord_at(chord_events, offset)

            beats_left_in_bar = beats_per_bar - (offset % beats_per_bar)
            dur_options = [d for d in [0.25, 0.5, 1.0, 1.5, 2.0]
                           if d <= min(beats_left_in_bar,
                                       total_beats - offset) + 0.01]
            if not dur_options:
                return notes_so_far

            new_beam = []
            seen = set()

            for dur in dur_options[:3]:
                for nxt_pitch in scale_pitches:
                    if not (low <= nxt_pitch <= high):
                        continue
                    key = (round(offset, 2), nxt_pitch, dur)
                    if key in seen:
                        continue
                    seen.add(key)

                    lc = self._local_cost(cur_pitch, nxt_pitch, dur, offset,
                                          total_beats, chord, scale_pitches,
                                          c_val, t_val, pitch_range, beats_per_bar)
                    h  = self._heuristic(nxt_pitch, offset + dur, total_beats)

                    climax = math.sin((offset / total_beats) * math.pi) * t_val
                    vel = max(vel_min, min(vel_max,
                              int(vel_min + (vel_max - vel_min) * climax)))

                    new_beam.append((
                        best_cost + lc + h,
                        offset + dur,
                        nxt_pitch,
                        notes_so_far + [MelodyNote(nxt_pitch, dur, vel,
                                                    round(offset, 4))],
                    ))

            new_beam.sort(key=lambda x: x[0])
            beam = new_beam[:beam_width]

        return beam[0][3] if beam else []


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 4: GENETIC (fitness armónico por acorde)
# ══════════════════════════════════════════════════════════════════════════════

class GeneticEngine:
    """
    Algoritmo genético con fitness que incluye:
    · Armonía por acorde activo (CT/T/PT/NT)
    · Clímax en posición de perfil (proporción áurea o ajustada)
    · Suavidad, contorno, variedad rítmica, rango
    """

    def __init__(self, profile: str, rhythm: str,
                 population_size: int = 20, generations: int = 30,
                 rng: Optional[random.Random] = None):
        self.profile    = profile
        self.rhythm     = rhythm
        self.pop_size   = population_size
        self.generations = generations
        self.rng        = rng or random.Random(42)

    def _random_individual(self, n_notes: int, n_pitches: int,
                            n_durs: int) -> List[Tuple[int, int]]:
        return [(self.rng.randint(0, n_pitches - 1),
                 self.rng.randint(0, n_durs - 1))
                for _ in range(n_notes)]

    def _fitness(self, individual: List[Tuple[int, int]],
                 scale_in_range: List[int], dur_options: List[float],
                 contour: List[float], tension: List[float],
                 beats_per_bar: float,
                 chord_events: List[ChordEvent]) -> float:
        pitches = [scale_in_range[min(idx, len(scale_in_range) - 1)]
                   for idx, _ in individual]
        durs    = [dur_options[min(idx, len(dur_options) - 1)]
                   for _, idx in individual]
        if len(pitches) < 2:
            return 0.0

        # Reconstruir offsets aproximados
        offsets = []
        off = 0.0
        for d in durs:
            offsets.append(off)
            off += d

        # 1. Armonía por acorde activo
        harmony_vals = []
        for p, o in zip(pitches, offsets):
            chord = get_chord_at(chord_events, o)
            ntype = chord_note_type(p, chord, scale_in_range + [p])
            harmony_vals.append({"CT": 1.0, "T": 0.7, "PT": 0.5, "NT": 0.1}[ntype])
        harmony_score = float(np.mean(harmony_vals)) if harmony_vals else 0.5

        # 2. Suavidad
        intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches) - 1)]
        big_leaps = sum(1 for iv in intervals if iv > 7)
        smoothness = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

        # 3. Contorno
        low, high = min(pitches), max(pitches)
        rng_val = high - low or 1
        contour_error = 0.0
        for i, p in enumerate(pitches):
            bar_idx = min(int(offsets[i] / beats_per_bar), len(contour) - 1)
            contour_error += abs((p - low) / rng_val - contour[bar_idx])
        contour_score = max(0.0, 1.0 - contour_error / len(pitches))

        # 4. Variedad rítmica
        variety = min(1.0, len(set(round(d, 2) for d in durs)) / 4)

        # 5. Rango (óptimo: 7-17 semitonos)
        rng_st = max(pitches) - min(pitches)
        range_score = min(1.0, max(0.0, (rng_st - 5) / 19))

        # 6. Posición del clímax según perfil
        climax_target = PROFILE_CLIMAX_POS.get(self.profile, 0.618)
        max_pos = pitches.index(max(pitches)) / max(len(pitches) - 1, 1)
        climax_score = 1.0 - abs(max_pos - climax_target)

        return (harmony_score  * 0.30 +
                smoothness     * 0.20 +
                contour_score  * 0.20 +
                variety        * 0.10 +
                range_score    * 0.10 +
                climax_score   * 0.10)

    def _crossover(self, p1: List[Tuple], p2: List[Tuple]) -> List[Tuple]:
        if len(p1) < 2:
            return deepcopy(p1)
        point = self.rng.randint(1, len(p1) - 1)
        return p1[:point] + p2[point:]

    def _mutate(self, individual: List[Tuple[int, int]],
                n_pitches: int, n_durs: int,
                rate: float = 0.1) -> List[Tuple[int, int]]:
        result = []
        for pitch_idx, dur_idx in individual:
            if self.rng.random() < rate:
                pitch_idx = max(0, min(n_pitches - 1,
                                       pitch_idx + self.rng.randint(-3, 3)))
            if self.rng.random() < rate * 0.5:
                dur_idx = self.rng.randint(0, n_durs - 1)
            result.append((pitch_idx, dur_idx))
        return result

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:
        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        total_beats = n_bars * beats_per_bar

        scale_in_range = [p for p in scale_pitches if low <= p <= high] or scale_pitches
        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])
        dur_options = sorted(rd.keys())

        avg_dur = sum(d * w for d, w in rd.items()) / max(sum(rd.values()), 1)
        n_notes = max(4, int(total_beats / avg_dur))

        population = [self._random_individual(n_notes, len(scale_in_range),
                                               len(dur_options))
                      for _ in range(self.pop_size)]

        for _ in range(self.generations):
            scored = [(self._fitness(ind, scale_in_range, dur_options,
                                     contour, tension, beats_per_bar,
                                     chord_events), ind)
                      for ind in population]
            scored.sort(key=lambda x: -x[0])
            elite   = [deepcopy(ind) for _, ind in scored[:2]]
            new_pop = list(elite)
            while len(new_pop) < self.pop_size:
                t1 = self.rng.choice(scored[:self.pop_size // 2])[1]
                t2 = self.rng.choice(scored[:self.pop_size // 2])[1]
                child = self._mutate(self._crossover(t1, t2),
                                     len(scale_in_range), len(dur_options))
                new_pop.append(child)
            population = new_pop

        best = max(population,
                   key=lambda ind: self._fitness(ind, scale_in_range, dur_options,
                                                  contour, tension, beats_per_bar,
                                                  chord_events))
        notes: List[MelodyNote] = []
        offset = 0.0
        for pitch_idx, dur_idx in best:
            if offset >= total_beats - 0.01:
                break
            p = scale_in_range[min(pitch_idx, len(scale_in_range) - 1)]
            d = min(dur_options[min(dur_idx, len(dur_options) - 1)],
                    total_beats - offset)
            bar_idx = int(offset // beats_per_bar)
            t_val   = tension[min(bar_idx, len(tension) - 1)]
            climax  = math.sin((offset / total_beats) * math.pi) * t_val
            vel = max(vel_min, min(vel_max,
                      int(vel_min + (vel_max - vel_min) * climax)))
            notes.append(MelodyNote(p, d, vel, round(offset, 4)))
            offset += d

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 5b: MOTIF (transformaciones temáticas)
# ══════════════════════════════════════════════════════════════════════════════

def parse_motif_text(text: str, root_pc: int,
                     scale_pitches: List[int]) -> List[MelodyNote]:
    """
    Parsea un motivo desde texto: "C4:0.5 E4:1 G4:0.5 E4:1"
    Formato: NombreNota:duracion_en_beats
    """
    PITCH_MAP = {"C":0,"D":2,"E":4,"F":5,"G":7,"A":9,"B":11}
    notes = []
    offset = 0.0
    for tok in text.strip().split():
        parts = tok.split(":")
        if len(parts) != 2:
            continue
        name_s, dur_s = parts
        # Parsear pitch
        base = name_s[0].upper()
        if base not in PITCH_MAP:
            continue
        pc = PITCH_MAP[base]
        rest = name_s[1:]
        oct_num = 4
        for i, ch in enumerate(rest):
            if ch == "#":
                pc += 1
            elif ch.lower() == "b":
                pc -= 1
            elif ch.isdigit():
                oct_num = int(rest[i:])
                break
        pitch = (oct_num + 1) * 12 + (pc % 12)
        dur   = float(dur_s)
        notes.append(MelodyNote(pitch=pitch, duration=dur,
                                velocity=80, offset=round(offset, 4)))
        offset += dur
    return notes


class MotifEngine:
    """
    Construye una melodía por elaboración de un motivo semilla.

    Transformaciones disponibles:
      identity      — repetición exacta
      sequence_up   — secuencia ascendente (motivo transpuesto +2 st en la escala)
      sequence_down — secuencia descendente (motivo transpuesto -2 st en la escala)
      inversion     — inversión de intervalos (sube↔baja)
      retrograde    — retrogradación (el motivo al revés)
      augmentation  — aumentación rítmica (duraciones ×2)
      diminution    — disminución rítmica (duraciones /2)
      variation     — variación: reemplaza una nota por el CT más cercano del acorde

    El motor elige qué transformación aplicar en cada semifrase (2 compases)
    según la tensión, la función armónica y el contorno objetivo.
    """

    # Probabilidad de cada transformación según tensión baja / alta
    TRANSFORMS_LOW = [
        "identity", "identity", "sequence_up", "sequence_down",
        "augmentation", "variation",
    ]
    TRANSFORMS_HIGH = [
        "inversion", "retrograde", "diminution", "sequence_up",
        "sequence_down", "variation", "variation",
    ]

    def __init__(self, profile: str, rng: random.Random,
                 motif: Optional[List[MelodyNote]] = None):
        self.profile = profile
        self.rng     = rng
        self.motif   = motif  # None = generar automáticamente

    # ── Transformaciones ──────────────────────────────────────────────────────

    def _snap_pitch(self, pitch: int, scale_pitches: List[int],
                    low: int, high: int) -> int:
        p = snap_to_scale(pitch, scale_pitches)
        while p < low and p + 12 <= high:
            p += 12
        while p > high and p - 12 >= low:
            p -= 12
        return max(low, min(high, p))

    def _transpose_by_scale_steps(self, motif: List[MelodyNote],
                                   steps: int,
                                   scale_pitches: List[int],
                                   low: int, high: int) -> List[MelodyNote]:
        """Transpone el motivo N grados de la escala arriba o abajo."""
        result = []
        for n in motif:
            if n.pitch in scale_pitches:
                idx = scale_pitches.index(n.pitch)
            else:
                idx = min(range(len(scale_pitches)),
                          key=lambda i: abs(scale_pitches[i] - n.pitch))
            new_idx = max(0, min(len(scale_pitches) - 1, idx + steps))
            new_pitch = scale_pitches[new_idx]
            new_pitch = self._snap_pitch(new_pitch, scale_pitches, low, high)
            result.append(MelodyNote(new_pitch, n.duration, n.velocity, 0.0))
        return result

    def _invert(self, motif: List[MelodyNote],
                scale_pitches: List[int],
                low: int, high: int) -> List[MelodyNote]:
        """Invierte los intervalos del motivo alrededor de la primera nota."""
        if not motif:
            return []
        pivot = motif[0].pitch
        result = [MelodyNote(motif[0].pitch, motif[0].duration,
                             motif[0].velocity, 0.0)]
        for n in motif[1:]:
            iv = n.pitch - pivot
            new_pitch = self._snap_pitch(pivot - iv, scale_pitches, low, high)
            result.append(MelodyNote(new_pitch, n.duration, n.velocity, 0.0))
        return result

    def _retrograde(self, motif: List[MelodyNote]) -> List[MelodyNote]:
        """Retrogradación: el motivo al revés (duraciones también)."""
        rev = list(reversed(motif))
        return [MelodyNote(n.pitch, n.duration, n.velocity, 0.0) for n in rev]

    def _augment(self, motif: List[MelodyNote],
                 factor: float = 2.0) -> List[MelodyNote]:
        """Aumentación rítmica: duraciones ×factor."""
        return [MelodyNote(n.pitch, n.duration * factor, n.velocity, 0.0)
                for n in motif]

    def _diminish(self, motif: List[MelodyNote],
                  factor: float = 0.5) -> List[MelodyNote]:
        """Disminución rítmica: duraciones ×factor."""
        return [MelodyNote(n.pitch, max(0.25, n.duration * factor),
                           n.velocity, 0.0)
                for n in motif]

    def _variation(self, motif: List[MelodyNote],
                   chord: ChordEvent,
                   scale_pitches: List[int],
                   low: int, high: int) -> List[MelodyNote]:
        """
        Variación ornamental: reemplaza una nota no-CT por el CT más cercano
        del acorde activo, y añade una nota de paso antes de la siguiente.
        """
        result = []
        for n in motif:
            pc = n.pitch % 12
            if pc not in chord.chord_pcs and len(result) == len(motif) // 2:
                new_pitch = nearest_chord_tone(n.pitch, chord)
                new_pitch = self._snap_pitch(new_pitch, scale_pitches, low, high)
                result.append(MelodyNote(new_pitch, n.duration,
                                         n.velocity, 0.0))
            else:
                result.append(MelodyNote(n.pitch, n.duration, n.velocity, 0.0))
        return result

    def _apply_transform(self, motif: List[MelodyNote],
                         transform: str,
                         chord: ChordEvent,
                         scale_pitches: List[int],
                         low: int, high: int) -> List[MelodyNote]:
        if transform == "identity":
            return [MelodyNote(n.pitch, n.duration, n.velocity, 0.0)
                    for n in motif]
        elif transform == "sequence_up":
            return self._transpose_by_scale_steps(motif, 2, scale_pitches, low, high)
        elif transform == "sequence_down":
            return self._transpose_by_scale_steps(motif, -2, scale_pitches, low, high)
        elif transform == "inversion":
            return self._invert(motif, scale_pitches, low, high)
        elif transform == "retrograde":
            return self._retrograde(motif)
        elif transform == "augmentation":
            return self._augment(motif)
        elif transform == "diminution":
            return self._diminish(motif)
        elif transform == "variation":
            return self._variation(motif, chord, scale_pitches, low, high)
        return [MelodyNote(n.pitch, n.duration, n.velocity, 0.0)
                for n in motif]

    def _generate_seed_motif(self, chord: ChordEvent,
                              scale_pitches: List[int],
                              low: int, high: int,
                              beats_per_bar: float) -> List[MelodyNote]:
        """Genera un motivo semilla de 3-4 notas armónicamente coherente."""
        # Primera nota: CT del acorde inicial cerca del centro del rango
        mid = (low + high) // 2
        candidates = [p for p in scale_pitches if p % 12 in chord.chord_pcs]
        p1 = min(candidates or scale_pitches, key=lambda p: abs(p - mid))

        # Segunda nota: un grado de escala hacia arriba o abajo
        idx = scale_pitches.index(p1) if p1 in scale_pitches else len(scale_pitches)//2
        step = self.rng.choice([-2, -1, 1, 2])
        p2_idx = max(0, min(len(scale_pitches)-1, idx + step))
        p2 = scale_pitches[p2_idx]

        # Tercera nota: CT del acorde una 3ª o 5ª más alta
        p3_candidates = [p for p in scale_pitches
                         if p % 12 in chord.chord_pcs and p > p1]
        p3 = (min(p3_candidates, key=lambda p: abs(p - (p1 + 5)))
              if p3_candidates else p2)

        # Cuarta nota: vuelta a p1 (cierra el motivo)
        p4 = p1

        # Duraciones: distribución cuaternaria dentro de un compás
        total = beats_per_bar
        dur_patterns = [
            [0.5, 0.5, 0.5, total - 1.5],
            [1.0, 0.5, 0.5, total - 2.0],
            [0.5, 1.0, total - 1.5],
            [1.0, 1.0, total - 2.0],
        ]
        durs = self.rng.choice(dur_patterns)

        pitches = ([p1, p2, p3, p4] if len(durs) == 4
                   else [p1, p2, p3])[:len(durs)]

        return [MelodyNote(p, d, 80, 0.0)
                for p, d in zip(pitches, durs)]

    def _fit_motif_to_duration(self, motif: List[MelodyNote],
                                target_beats: float) -> List[MelodyNote]:
        """Ajusta la duración total del motivo transformado al target."""
        total = sum(n.duration for n in motif)
        if total <= 0:
            return motif
        scale = target_beats / total
        result = []
        for n in motif:
            new_dur = max(0.25, round(n.duration * scale * 4) / 4)
            result.append(MelodyNote(n.pitch, new_dur, n.velocity, 0.0))
        # Ajustar el último para que sume exactamente target_beats
        actual = sum(n.duration for n in result)
        diff   = target_beats - actual
        if result and abs(diff) > 0.01:
            last = result[-1]
            result[-1] = MelodyNote(last.pitch,
                                     max(0.25, last.duration + diff),
                                     last.velocity, last.offset)
        return result

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:

        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        total_beats = n_bars * beats_per_bar

        # Generar o usar motivo semilla
        first_chord = get_chord_at(chord_events, 0.0)
        if self.motif:
            seed = self.motif
        else:
            seed = self._generate_seed_motif(first_chord, scale_pitches,
                                              low, high, beats_per_bar)

        notes: List[MelodyNote] = []
        offset = 0.0
        phrase_beats = beats_per_bar * 2  # semifrase de 2 compases
        current_motif = seed

        while offset < total_beats - 0.01:
            bar_idx = int(offset / beats_per_bar)
            t_val   = tension[min(bar_idx, len(tension) - 1)]
            c_val   = contour[min(bar_idx, len(contour) - 1)]
            chord   = get_chord_at(chord_events, offset)

            # Elegir transformación según tensión y contorno
            pool = self.TRANSFORMS_HIGH if t_val > 0.5 else self.TRANSFORMS_LOW

            # Forzar identidad en el primer y último compás para anclar
            if bar_idx == 0:
                transform = "identity"
            elif bar_idx >= n_bars - 2:
                transform = "identity"
            else:
                # Sesgar hacia secuencia ascendente si el contorno quiere subir
                cur_rel = (sum(n.pitch for n in current_motif) /
                           max(len(current_motif), 1) - low) / max(high - low, 1)
                if c_val > cur_rel + 0.2:
                    pool = ["sequence_up", "sequence_up"] + pool
                elif c_val < cur_rel - 0.2:
                    pool = ["sequence_down", "sequence_down"] + pool
                transform = self.rng.choice(pool)

            transformed = self._apply_transform(
                current_motif, transform, chord, scale_pitches, low, high)

            # Ajustar duración al espacio disponible
            remaining = total_beats - offset
            target    = min(phrase_beats, remaining)
            fitted    = self._fit_motif_to_duration(transformed, target)

            # Asignar offsets y velocidades
            vel_base = int(vel_min + (vel_max - vel_min) *
                           math.sin((offset / total_beats) * math.pi) * t_val)
            vel_base = max(vel_min, min(vel_max, vel_base))

            cur_off = offset
            for n in fitted:
                if cur_off >= total_beats - 0.01:
                    break
                vel = max(vel_min, min(vel_max,
                          vel_base + self.rng.randint(-8, 8)))
                notes.append(MelodyNote(
                    pitch    = max(low, min(high, n.pitch)),
                    duration = min(n.duration, total_beats - cur_off),
                    velocity = vel,
                    offset   = round(cur_off, 4),
                ))
                cur_off += n.duration

            # El motivo actual pasa a ser el transformado para la próxima frase
            # (con probabilidad 0.6 — a veces volvemos al original)
            if self.rng.random() < 0.6:
                current_motif = transformed
            else:
                current_motif = seed

            offset += target

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 6: BAYES (n-gram con suavizado Kneser-Ney)
# ══════════════════════════════════════════════════════════════════════════════

class BayesEngine:
    """
    Modelo n-gram (orden 3) con suavizado de Kneser-Ney entrenado sobre
    el corpus MIDI. A diferencia del Markov sintético de chord_guided,
    siempre tiene una estimación razonada para transiciones no vistas.

    El vocabulario es (pitch_class, dur_bucket, chord_func):
      pitch_class: 0-11
      dur_bucket: 0=corta(≤0.5), 1=media(≤1.5), 2=larga(>1.5)
      chord_func: T/S/D/?

    El suavizado Kneser-Ney estima p(w|context) como:
      max(c(context,w) - d, 0) / c(context)
        + lambda(context) * p_KN(w | context[1:])

    donde d=0.75 (descuento estándar) y p_KN es la distribución
    de continuación (número de contextos únicos en que aparece w).
    """

    DISCOUNT = 0.75
    ORDER    = 3  # trigramas

    def __init__(self, profile: str, rhythm: str, rng: random.Random):
        self.profile = profile
        self.rhythm  = rhythm
        self.rng     = rng
        # Conteos n-gram: {contexto_tuple: {token: count}}
        self._ngram:    Dict[tuple, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self._ngram_m1: Dict[tuple, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        self._ngram_m2: Dict[tuple, Dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
        # Conteos de continuación (KN): {token: set_of_unique_contexts}
        self._continuation: Dict[tuple, set] = defaultdict(set)
        self._trained = False
        self._vocab:   List[tuple] = []

    def _dur_bucket(self, d: float) -> int:
        return 0 if d <= 0.5 else (1 if d <= 1.5 else 2)

    def _func_code(self, hf: str) -> int:
        return {"T": 0, "S": 1, "D": 2, "?": 3}.get(hf, 3)

    def _note_to_token(self, pitch: int, dur: float,
                       chord: ChordEvent) -> tuple:
        return (pitch % 12, self._dur_bucket(dur), self._func_code(chord.harm_func))

    def train_from_midi(self, midi_path: str,
                        root_pc_key: int = 0,
                        mode: str = "major") -> int:
        """Aprende n-gramas desde un MIDI. Devuelve nº de tokens aprendidos."""
        if not MIDO_OK:
            return 0
        try:
            mid = mido.MidiFile(midi_path)
        except Exception:
            return 0

        tpb = mid.ticks_per_beat or 480
        notes_raw: List[Tuple[float, float, int]] = []
        current_tick = 0
        active: Dict[int, int] = {}

        track_idx = 1 if len(mid.tracks) > 1 else 0
        for msg in mid.tracks[track_idx]:
            current_tick += msg.time
            if hasattr(msg, "channel") and msg.channel != 0:
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = current_tick
            elif msg.type in ("note_off",) or (
                    msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    on_t = active.pop(msg.note)
                    dur_q = (current_tick - on_t) / tpb
                    notes_raw.append((on_t / tpb, dur_q, msg.note))

        if not notes_raw:
            return 0

        notes_raw.sort(key=lambda n: n[0])

        # Extraer acordes para el condicionamiento
        try:
            chord_events = extract_chords_from_midi(
                midi_path, root_pc_key, mode,
                harmony_track_idx=2 if len(mid.tracks) > 2 else 0)
        except Exception:
            # Fallback: un solo acorde tónico
            chord_events = [build_chord_event(
                NOTE_NAMES[root_pc_key], 9999.0, 0.0, root_pc_key, mode)]

        tokens = [self._note_to_token(p, d, get_chord_at(chord_events, o))
                  for o, d, p in notes_raw]

        if len(tokens) < self.ORDER:
            return 0

        # Acumular conteos
        for i in range(len(tokens)):
            tok = tokens[i]
            if i >= 1:
                ctx1 = (tokens[i-1],)
                self._ngram_m1[ctx1][tok] += 1
                self._continuation[tok].add(ctx1)
            if i >= 2:
                ctx2 = (tokens[i-2], tokens[i-1])
                self._ngram_m2[ctx2][tok] += 1
            if i >= self.ORDER - 1:
                ctx3 = tuple(tokens[i-self.ORDER+1:i])
                self._ngram[ctx3][tok] += 1

        self._trained = True
        return len(tokens)

    def train_from_corpus(self, corpus_dir: str,
                          verbose: bool = False):
        path  = Path(corpus_dir)
        files = list(path.rglob("*.mid")) + list(path.rglob("*.midi"))
        total = 0
        for f in files:
            n = self.train_from_midi(str(f))
            total += n
        # Construir vocabulario
        all_tokens: set = set()
        for ctx_dict in [self._ngram, self._ngram_m1, self._ngram_m2]:
            for ctx, succs in ctx_dict.items():
                all_tokens.update(succs.keys())
        self._vocab = list(all_tokens) if all_tokens else [
            (pc, db, f) for pc in range(12)
            for db in range(3) for f in range(4)]
        self._trained = bool(total)
        if verbose:
            print(f"  [bayes] corpus={len(files)} MIDIs  "
                  f"tokens={total}  vocab={len(self._vocab)}")

    def _build_synthetic(self):
        """Fallback sintético si no hay corpus."""
        iw = INTERVAL_WEIGHTS.get(self.profile, INTERVAL_WEIGHTS["serene"])
        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])
        rng = self.rng
        for _ in range(500):
            iv  = weighted_choice(list(iw.keys()), list(iw.values()), rng)
            pc  = iv % 12
            db  = self._dur_bucket(
                weighted_choice(list(rd.keys()), list(rd.values()), rng))
            tok = (pc, db, rng.randint(0, 3))
            ctx = ((pc, db, 0),)
            self._ngram_m1[ctx][tok] += 1
            self._continuation[tok].add(ctx)
        self._vocab = [(pc, db, f) for pc in range(12)
                       for db in range(3) for f in range(4)]
        self._trained = True

    def _kn_prob(self, token: tuple, context: tuple) -> float:
        """
        Probabilidad Kneser-Ney para token dado context (trigramas → bigramas → unigramas).
        """
        d = self.DISCOUNT

        # Nivel 3 (trigrama)
        if len(context) >= 2:
            ctx3 = context[-2:]
            c3   = self._ngram.get(ctx3, {})
            n3   = sum(c3.values())
            if n3 > 0:
                count = c3.get(token, 0)
                lam   = d * len(c3) / n3
                p_low = self._kn_prob(token, context[-1:])
                return max(count - d, 0) / n3 + lam * p_low

        # Nivel 2 (bigrama)
        if len(context) >= 1:
            ctx2 = context[-1:]
            c2   = self._ngram_m1.get(ctx2, {})
            n2   = sum(c2.values())
            if n2 > 0:
                count = c2.get(token, 0)
                lam   = d * len(c2) / n2
                p_low = self._kn_prob(token, ())
                return max(count - d, 0) / n2 + lam * p_low

        # Nivel 1 (unigrama de continuación)
        n_cont_tok = len(self._continuation.get(token, set()))
        n_cont_all = sum(len(v) for v in self._continuation.values()) or 1
        return max(n_cont_tok, 1) / n_cont_all

    def _sample_next(self, context: tuple,
                     chord: ChordEvent,
                     scale_pitches: List[int],
                     pitch_range: Tuple[int, int],
                     tension_val: float) -> Tuple[int, int]:
        """
        Muestrea el siguiente token. Devuelve (pitch_class, dur_bucket).
        El acorde activo sesga las probabilidades hacia chord tones.
        """
        if not self._vocab:
            return (0, 1)

        low, high = pitch_range
        scale_pcs = {p % 12 for p in scale_pitches}
        emo_weights = EMOTIONAL_CHORD_WEIGHTS.get(
            self.profile, EMOTIONAL_CHORD_WEIGHTS["custom"])

        probs = []
        for tok in self._vocab:
            pc, db, func = tok
            p_kn = self._kn_prob(tok, context)

            # Sesgo armónico: multiplicar por el peso del tipo de nota
            if pc in chord.chord_pcs:
                ntype = "CT"
            elif pc in chord.tension_pcs:
                ntype = "T"
            elif pc in scale_pcs:
                ntype = "PT"
            else:
                ntype = "NT"
            harm_mult = emo_weights[ntype]

            # Sesgo de tensión
            tension_mult = 1.0
            if ntype == "CT":
                tension_mult = 1.0 - tension_val * 0.3
            elif ntype == "T":
                tension_mult = 1.0 + tension_val * 1.5
            elif ntype == "NT":
                tension_mult = tension_val * 1.2

            probs.append(max(p_kn * harm_mult * max(tension_mult, 0.01), 1e-9))

        # Muestrear
        total = sum(probs)
        r = self.rng.random() * total
        acc = 0.0
        for tok, w in zip(self._vocab, probs):
            acc += w
            if r <= acc:
                return tok[0], tok[1]  # pc, dur_bucket
        return self._vocab[-1][0], self._vocab[-1][1]

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:

        if not self._trained:
            self._build_synthetic()

        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        total_beats = n_bars * beats_per_bar

        dur_from_bucket = {0: 0.5, 1: 1.0, 2: 2.0}
        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])

        # Pitch inicial
        first_chord = chord_events[0] if chord_events else None
        if first_chord and first_chord.chord_pcs:
            mid_range = (low + high) // 2
            cands = [p for p in range(low, high+1)
                     if p % 12 in first_chord.chord_pcs]
            current_pitch = (min(cands, key=lambda p: abs(p - mid_range))
                             if cands else snap_to_scale(mid_range, scale_pitches))
        else:
            current_pitch = snap_to_scale((low + high) // 2, scale_pitches)

        notes: List[MelodyNote] = []
        context: tuple = ()
        offset = 0.0

        while offset < total_beats - 0.01:
            bar_idx = int(offset / beats_per_bar)
            t_val   = tension[min(bar_idx, len(tension) - 1)]
            c_val   = contour[min(bar_idx, len(contour) - 1)]
            chord   = get_chord_at(chord_events, offset)

            # Muestrear siguiente token
            pc, db = self._sample_next(context, chord, scale_pitches,
                                        pitch_range, t_val)

            # Convertir pitch_class a pitch absoluto, guiado por contorno
            target_pitch = low + c_val * (high - low)
            octave_candidates = [p for p in range(low, high + 1)
                                  if p % 12 == pc]
            if octave_candidates:
                new_pitch = min(octave_candidates,
                                key=lambda p: abs(p - target_pitch))
            else:
                new_pitch = snap_to_scale(current_pitch, scale_pitches)

            # Duración desde el bucket, con verificación de límites
            beats_left = total_beats - offset
            base_dur = dur_from_bucket.get(db, 1.0)
            # Ajustar a la distribución rítmica del perfil
            valid_durs = {d: w for d, w in rd.items()
                          if abs(d - base_dur) <= 0.75 and d <= beats_left}
            if valid_durs:
                dur = weighted_choice(list(valid_durs.keys()),
                                      list(valid_durs.values()), self.rng)
            else:
                dur = min(base_dur, beats_left)
            dur = max(0.25, dur)

            # Velocidad
            climax = math.sin((offset / total_beats) * math.pi) * t_val
            vel = max(vel_min, min(vel_max,
                      int(vel_min + (vel_max - vel_min) * max(0.0, climax))))

            notes.append(MelodyNote(new_pitch, dur, vel, round(offset, 4)))

            # Actualizar contexto (ventana deslizante de ORDER-1 tokens)
            func_code = {"T":0,"S":1,"D":2,"?":3}.get(chord.harm_func, 3)
            new_tok = (pc, self._dur_bucket(dur), func_code)
            context = (context + (new_tok,))[-(self.ORDER - 1):]

            current_pitch = new_pitch
            offset += dur

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 7: TENSION_ARC (planificación de atrás hacia adelante)
# ══════════════════════════════════════════════════════════════════════════════

class TensionArcEngine:
    """
    Planificación de la melodía de atrás hacia adelante.

    Estructura por frase (cada phrase_bars compases):
      1. Planificar: elegir nota de resolución (chord tone de la tónica o
         del acorde de cierre), nota de clímax (tensión máxima), y punto
         de máxima tensión (PROFILE_CLIMAX_POS de la frase).
      2. Relleno: generar el camino de la apertura al clímax y del clímax
         a la resolución usando movimiento escalonado + saltos armónicos.

    La resolución siempre es un chord tone del acorde activo al final de
    la frase. El clímax tiende a ser la 7ª, 9ª, o una nota cromática
    de tensión real.
    """

    def __init__(self, profile: str, rhythm: str,
                 rng: random.Random,
                 phrase_bars: int = 4):
        self.profile     = profile
        self.rhythm      = rhythm
        self.rng         = rng
        self.phrase_bars = phrase_bars

    def _resolution_pitch(self, chord: ChordEvent,
                           scale_pitches: List[int],
                           low: int, high: int,
                           target_height: float) -> int:
        """
        Nota de resolución: chord tone del acorde, cerca de target_height
        (valor relativo 0-1 en el rango).
        """
        target = low + target_height * (high - low)
        cands  = [p for p in scale_pitches
                  if p % 12 in chord.chord_pcs and low <= p <= high]
        if not cands:
            cands = [p for p in scale_pitches if low <= p <= high]
        return min(cands, key=lambda p: abs(p - target))

    def _climax_pitch(self, chord: ChordEvent,
                      scale_pitches: List[int],
                      low: int, high: int,
                      resolution: int,
                      t_val: float) -> int:
        """
        Nota de clímax: tensión o nota cromática, por encima de la resolución.
        """
        # Preferir tensiones del acorde
        tension_cands = [p for p in range(low, high + 1)
                         if p % 12 in chord.tension_pcs and p > resolution]
        if tension_cands and self.rng.random() < 0.6:
            return min(tension_cands, key=lambda p: p - resolution)

        # Fallback: nota escalonada más alta disponible
        above = [p for p in scale_pitches if p > resolution and p <= high]
        if above:
            # A mayor tensión, saltar más arriba
            n_steps = max(1, int(t_val * len(above)))
            n_steps = min(n_steps, len(above) - 1)
            return above[min(n_steps, len(above) - 1)]

        return min(high, resolution + 2)

    def _fill_path(self, start: int, end: int,
                   n_beats: float, chord: ChordEvent,
                   scale_pitches: List[int],
                   low: int, high: int,
                   t_val: float) -> List[Tuple[int, float]]:
        """
        Rellena el camino de start a end en n_beats.
        Devuelve lista de (pitch, duration).
        """
        if n_beats <= 0.25:
            return [(end, n_beats)]

        direction = 1 if end >= start else -1
        interval  = abs(end - start)

        # Calcular cuántas notas caben
        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])
        avg_dur = (sum(d * w for d, w in rd.items()) /
                   max(sum(rd.values()), 1))
        n_notes = max(2, int(n_beats / avg_dur))

        # Generar pitches intermedios escalonados hacia el destino
        path_pcs: List[int] = []
        # Notas en la dirección correcta desde start hasta end
        if direction > 0:
            candidates = [p for p in scale_pitches
                          if start < p <= end]
        else:
            candidates = [p for p in scale_pitches
                          if end <= p < start]

        # Añadir chord tones como waypoints importantes
        chord_on_path = [p for p in candidates if p % 12 in chord.chord_pcs]

        if interval <= 2:
            path_pcs = [start, end]
        elif chord_on_path:
            path_pcs = [start] + chord_on_path + [end]
        else:
            step = max(1, interval // n_notes)
            path_pcs = list(range(start, end + direction, direction * step))
            path_pcs = [p for p in path_pcs if low <= p <= high]
            if path_pcs and path_pcs[-1] != end:
                path_pcs.append(end)
            if not path_pcs:
                path_pcs = [start, end]

        # Snappear a escala
        path_pcs = [snap_to_scale(p, scale_pitches) for p in path_pcs]
        path_pcs = sorted(set(path_pcs), key=path_pcs.index)
        if not path_pcs:
            path_pcs = [start, end]

        # Distribuir duración uniformemente
        n = len(path_pcs)
        base_dur = n_beats / n
        # Cuantizar a grid
        grid = 0.25
        dur  = max(grid, round(base_dur / grid) * grid)

        result = []
        remaining = n_beats
        for i, p in enumerate(path_pcs):
            if remaining <= 0.01:
                break
            d = dur if i < n - 1 else max(grid, remaining)
            result.append((p, min(d, remaining)))
            remaining -= d

        return result if result else [(end, n_beats)]

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:

        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        total_beats     = n_bars * beats_per_bar
        phrase_beats    = self.phrase_bars * beats_per_bar
        climax_pos_frac = PROFILE_CLIMAX_POS.get(self.profile, 0.618)

        notes:  List[MelodyNote] = []
        offset = 0.0

        while offset < total_beats - 0.01:
            phrase_end  = min(offset + phrase_beats, total_beats)
            phrase_dur  = phrase_end - offset
            bar_idx     = int(offset / beats_per_bar)

            # Tensión media de la frase
            t_vals  = [tension[min(bar_idx + b, len(tension) - 1)]
                       for b in range(self.phrase_bars)]
            t_mean  = float(np.mean(t_vals))
            t_peak  = max(t_vals)

            # Contorno al inicio, clímax y final de la frase
            c_start = contour[min(bar_idx, len(contour) - 1)]
            c_climax = min(1.0, c_start + t_mean * 0.4)
            c_end    = max(0.0, c_start - t_mean * 0.2)

            # Acordes al inicio y al final de la frase
            chord_start  = get_chord_at(chord_events, offset)
            chord_end    = get_chord_at(chord_events, phrase_end - 0.01)

            # ── Planificación ─────────────────────────────────────────────────
            pitch_start  = self._resolution_pitch(
                chord_start, scale_pitches, low, high, c_start)
            pitch_end    = self._resolution_pitch(
                chord_end, scale_pitches, low, high, c_end)
            pitch_climax = self._climax_pitch(
                chord_start, scale_pitches, low, high,
                max(pitch_start, pitch_end), t_peak)

            # Posición del clímax dentro de la frase
            climax_offset = offset + phrase_dur * climax_pos_frac

            # ── Relleno: apertura → clímax ────────────────────────────────────
            pre_dur   = climax_offset - offset
            post_dur  = phrase_end - climax_offset

            pre_path  = self._fill_path(pitch_start, pitch_climax,
                                         pre_dur, chord_start,
                                         scale_pitches, low, high, t_mean)
            post_path = self._fill_path(pitch_climax, pitch_end,
                                         post_dur, chord_end,
                                         scale_pitches, low, high, t_mean)

            # ── Convertir a MelodyNote ─────────────────────────────────────────
            cur_off = offset
            for path, is_climax_zone in [(pre_path, False), (post_path, True)]:
                for pitch, dur in path:
                    if cur_off >= total_beats - 0.01:
                        break
                    bar_now = int(cur_off / beats_per_bar)
                    t_now   = tension[min(bar_now, len(tension) - 1)]
                    frac    = cur_off / total_beats
                    climax  = math.sin(frac * math.pi) * t_now
                    vel = max(vel_min, min(vel_max,
                              int(vel_min + (vel_max - vel_min) *
                                  max(0.0, climax))))
                    # El clímax suena más fuerte
                    if is_climax_zone and pitch == pitch_climax:
                        vel = min(vel_max, vel + 10)

                    notes.append(MelodyNote(
                        pitch    = max(low, min(high, pitch)),
                        duration = min(dur, total_beats - cur_off),
                        velocity = vel,
                        offset   = round(cur_off, 4),
                    ))
                    cur_off += dur

            offset = phrase_end

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR 8: INTERPOLATE (interpolación entre dos perfiles)
# ══════════════════════════════════════════════════════════════════════════════

def _interpolate_dicts(a: Dict, b: Dict, alpha: float) -> Dict:
    """Interpola linealmente dos diccionarios numéricos."""
    keys = set(a) | set(b)
    return {k: (1 - alpha) * a.get(k, 0.0) + alpha * b.get(k, 0.0)
            for k in keys}


def _build_interpolated_profile(profile_a: str, profile_b: str,
                                 alpha: float) -> Dict[str, Any]:
    """
    Construye un perfil interpolado entre profile_a (alpha=0) y profile_b (alpha=1).
    Devuelve un dict con todos los parámetros interpolados.
    """
    def lerp(va, vb):
        return (1 - alpha) * va + alpha * vb

    # Velocidades
    vmin_a, vmax_a = PROFILE_VELOCITY.get(profile_a, (50, 90))
    vmin_b, vmax_b = PROFILE_VELOCITY.get(profile_b, (50, 90))
    vel = (int(lerp(vmin_a, vmin_b)), int(lerp(vmax_a, vmax_b)))

    # Pesos de intervalo
    iw_a = INTERVAL_WEIGHTS.get(profile_a, INTERVAL_WEIGHTS["custom"])
    iw_b = INTERVAL_WEIGHTS.get(profile_b, INTERVAL_WEIGHTS["custom"])
    iw   = _interpolate_dicts(iw_a, iw_b, alpha)

    # Pesos armónicos emocionales
    ew_a = EMOTIONAL_CHORD_WEIGHTS.get(profile_a, EMOTIONAL_CHORD_WEIGHTS["custom"])
    ew_b = EMOTIONAL_CHORD_WEIGHTS.get(profile_b, EMOTIONAL_CHORD_WEIGHTS["custom"])
    ew   = _interpolate_dicts(ew_a, ew_b, alpha)

    # Posición del clímax
    climax = lerp(PROFILE_CLIMAX_POS.get(profile_a, 0.618),
                  PROFILE_CLIMAX_POS.get(profile_b, 0.618))

    return {
        "velocity":       vel,
        "interval_weights": iw,
        "emotional_weights": ew,
        "climax_pos":     climax,
        "profile_a":      profile_a,
        "profile_b":      profile_b,
        "alpha":          alpha,
    }


class InterpolateEngine:
    """
    Interpolación entre dos perfiles emocionales.

    alpha=0.0 → profile_a puro
    alpha=1.0 → profile_b puro
    alpha=0.5 → mezcla equitativa

    Internamente usa un ChordGuidedEngine con los parámetros del
    perfil interpolado inyectados en las tablas de pesos.
    La curva de tensión también se interpola compás a compás.
    """

    def __init__(self, profile_a: str, profile_b: str,
                 alpha: float, rhythm: str,
                 rng: random.Random):
        self.profile_a = profile_a
        self.profile_b = profile_b
        self.alpha     = max(0.0, min(1.0, alpha))
        self.rhythm    = rhythm
        self.rng       = rng
        self._iprofile = _build_interpolated_profile(profile_a, profile_b, alpha)

    def _interp_tension(self, n_bars: int) -> List[float]:
        """Interpola las curvas de tensión de ambos perfiles."""
        t = np.linspace(0, 1, n_bars)

        def curve(profile: str) -> np.ndarray:
            curves = {
                "heroic":      0.3 + 0.6 * np.sin(t * math.pi),
                "melancholic": 0.5 - 0.3 * t,
                "tense":       np.clip(0.4 + 0.5 * t, 0, 1),
                "playful":     0.4 + 0.2 * np.sin(t * 4 * math.pi),
                "serene":      0.2 + 0.15 * np.sin(t * 2 * math.pi),
                "mysterious":  0.3 + 0.4 * np.sin(t * math.pi),
                "triumphant":  np.clip(0.5*t + 0.5*np.sin(t*math.pi), 0, 1),
                "elegiac":     0.55 - 0.25*t + 0.15*np.sin(t*3*math.pi),
                "ecstatic":    np.clip(0.75 + 0.2*np.sin(t*2*math.pi), 0, 1),
                "brooding":    np.clip(0.7 - 0.1*t + 0.05*np.sin(t*5*math.pi), 0, 1),
                "pastoral":    0.15 + 0.1*np.sin(t*2*math.pi),
                "agitated":    np.clip(0.5 + 0.4*np.abs(np.sin(t*7*math.pi)), 0, 1),
                "hypnotic":    np.full(n_bars, 0.25),
                "flamenco":    np.clip(0.35 + 0.5*np.sin(t*math.pi)
                                       + 0.15*np.sin(t*5*math.pi), 0, 1),
                "tanguero":    0.4 + 0.35*np.sin(t*4*math.pi),
            }
            return curves.get(profile, np.full(n_bars, 0.4))

        ca = curve(self.profile_a)
        cb = curve(self.profile_b)
        return list((1 - self.alpha) * np.asarray(ca) +
                    self.alpha * np.asarray(cb))

    def generate(self, chord_events: List[ChordEvent],
                 scale_pitches: List[int], n_bars: int,
                 beats_per_bar: float, contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:

        ip = self._iprofile
        low, high = pitch_range
        vel_min, vel_max = ip["velocity"]
        total_beats = n_bars * beats_per_bar
        iw = ip["interval_weights"]
        ew = ip["emotional_weights"]
        climax_pos = ip["climax_pos"]

        # Curva de tensión interpolada (salvo que el usuario la haya dado)
        interp_tension = self._interp_tension(n_bars)
        # Mezclar con la curva dada (si el usuario la proporcionó)
        effective_tension = [
            0.5 * t + 0.5 * it
            for t, it in zip(tension, interp_tension)
        ]

        # Usar ChordGuidedEngine como motor base, inyectando los parámetros
        # interpolados mediante monkey-patching local de las tablas
        ivs = list(iw.keys())
        iws = list(iw.values())

        # Markov sintético con los pesos interpolados
        dur_buckets = {0: 0.5, 1: 1.0, 2: 2.0}
        rd = RHYTHM_DISTS.get(self.rhythm, RHYTHM_DISTS["flowing"])

        # Pitch inicial
        first_chord = chord_events[0] if chord_events else None
        if first_chord and first_chord.chord_pcs:
            mid_range = (low + high) // 2
            cands = [p for p in range(low, high+1)
                     if p % 12 in first_chord.chord_pcs]
            current_pitch = (min(cands, key=lambda p: abs(p - mid_range))
                             if cands else snap_to_scale(mid_range, scale_pitches))
        else:
            current_pitch = snap_to_scale((low + high) // 2, scale_pitches)

        notes: List[MelodyNote] = []
        offset = 0.0
        prev_iv = 0

        while offset < total_beats - 0.01:
            bar_idx = int(offset / beats_per_bar)
            t_val   = effective_tension[min(bar_idx, len(effective_tension) - 1)]
            c_val   = contour[min(bar_idx, len(contour) - 1)]
            chord   = get_chord_at(chord_events, offset)

            # Duración
            beats_left = total_beats - offset
            beats_in_bar = beats_per_bar - (offset % beats_per_bar)
            valid_durs = {d: w for d, w in rd.items()
                          if d <= min(beats_in_bar, beats_left) + 0.01}
            if not valid_durs:
                break
            dur = weighted_choice(list(valid_durs.keys()),
                                  list(valid_durs.values()), self.rng)

            # Intervalo desde pesos interpolados
            tension_ivs = [iv for iv in ivs if abs(iv) >= 3] if t_val > 0.6 else []
            if tension_ivs and self.rng.random() < t_val * 0.5:
                iv = weighted_choice(tension_ivs,
                                     [iw[i] for i in tension_ivs], self.rng)
            else:
                iv = weighted_choice(ivs, iws, self.rng)

            # Corrección de contorno
            cur_rel = (current_pitch - low) / max(high - low, 1)
            if cur_rel > c_val + 0.15 and iv > 0:
                iv = -abs(iv)
            elif cur_rel < c_val - 0.15 and iv < 0:
                iv = abs(iv)

            max_leap = int(3 + t_val * 9)
            if abs(iv) > max_leap:
                iv = int(math.copysign(max_leap, iv))

            candidate = max(low, min(high, current_pitch + iv))

            # Pesos armónicos interpolados
            scale_pcs = {p % 12 for p in scale_pitches}
            func_w = HARMONIC_FUNCTION_WEIGHTS.get(chord.harm_func,
                                                     HARMONIC_FUNCTION_WEIGHTS["?"])
            nbr = range(max(low, candidate - 3), min(high, candidate + 3) + 1)
            nbr_weights = []
            for p in nbr:
                pc = p % 12
                if pc in chord.chord_pcs:
                    ntype = "CT"
                elif pc in chord.tension_pcs:
                    ntype = "T"
                elif pc in scale_pcs:
                    ntype = "PT"
                else:
                    ntype = "NT"
                w = func_w[ntype] * ew.get(ntype, 1.0)
                nbr_weights.append(max(w, 0.01))

            new_pitch = (weighted_choice(list(nbr), nbr_weights, self.rng)
                         if any(w > 0 for w in nbr_weights)
                         else snap_to_scale(candidate, scale_pitches))

            # Velocidad
            frac   = offset / total_beats
            climax = math.sin(frac * math.pi) * t_val
            vel = max(vel_min, min(vel_max,
                      int(vel_min + (vel_max - vel_min) * max(0.0, climax))))

            notes.append(MelodyNote(new_pitch, dur, vel, round(offset, 4)))
            prev_iv = new_pitch - current_pitch
            current_pitch = new_pitch
            offset += dur

        return notes


# ══════════════════════════════════════════════════════════════════════════════
#  BANCO DE EJEMPLOS FEW-SHOT
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM — PREPROCESADO DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def _chord_vector(chord: ChordEvent) -> np.ndarray:
    """
    Construye el vector de condicionamiento armónico para un ChordEvent.
    Devuelve float32[16]:
      [0:12]  — pitch classes activos (0.0 / 1.0)
      [12:16] — función armónica one-hot (T / S / D / ?)
    """
    vec = np.zeros(PR_CHORD_DIM, dtype=np.float32)
    for pc in chord.chord_pcs:
        vec[pc % 12] = 1.0
    func_idx = {"T": 12, "S": 13, "D": 14, "?": 15}.get(chord.harm_func, 15)
    vec[func_idx] = 1.0
    return vec


def _midi_to_piano_roll(midi_path: str,
                         beats_per_bar: float = 4.0,
                         root_pc_key: int = 0,
                         mode: str = "major",
                         harmony_track_idx: int = 2,
                         melody_track_idx: int = 1,
                         melody_channel: int = 0
                         ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Convierte un MIDI del corpus a (piano_roll, chord_matrix):

      piano_roll:   int16[T]  — token por paso de semicorchea
      chord_matrix: float32[T, 16] — chord_vector por paso

    Devuelve None si el MIDI no tiene el formato esperado.
    """
    if not MIDO_OK:
        return None
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return None

    if len(mid.tracks) < 3:
        return None

    tpb = mid.ticks_per_beat or 480
    steps_per_beat = PR_STEPS_PER_BEAT
    ticks_per_step = tpb / steps_per_beat  # ticks por semicorchea

    # ── Leer notas de melodía ─────────────────────────────────────────────────
    mel_notes: List[Tuple[int, int, int]] = []  # (step_on, step_off, pitch)
    current_tick = 0
    active: Dict[int, int] = {}

    for msg in mid.tracks[melody_track_idx]:
        current_tick += msg.time
        if hasattr(msg, "channel") and msg.channel != melody_channel:
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = current_tick
        elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0):
            if msg.note in active:
                on_tick  = active.pop(msg.note)
                step_on  = int(on_tick / ticks_per_step)
                step_off = max(step_on + 1,
                               int(current_tick / ticks_per_step))
                if PR_PITCH_MIN <= msg.note <= PR_PITCH_MAX:
                    mel_notes.append((step_on, step_off, msg.note))

    if not mel_notes:
        return None

    total_steps = max(off for _, off, _ in mel_notes)
    if total_steps < steps_per_beat * beats_per_bar:
        return None  # demasiado corto

    # ── Construir piano roll ──────────────────────────────────────────────────
    piano_roll = np.full(total_steps, PR_SILENCE, dtype=np.int16)

    mel_notes.sort(key=lambda n: n[0])
    for step_on, step_off, pitch in mel_notes:
        if step_on >= total_steps:
            continue
        piano_roll[step_on] = pitch
        for s in range(step_on + 1, min(step_off, total_steps)):
            # Solo marca hold si no hay otra nota que empieza en este paso
            if piano_roll[s] == PR_SILENCE:
                piano_roll[s] = PR_HOLD

    # ── Extraer acordes y construir chord_matrix ──────────────────────────────
    try:
        chord_events = extract_chords_from_midi(
            midi_path, root_pc_key, mode,
            harmony_track_idx=harmony_track_idx,
            harmony_channel=1,
            min_window_beats=0.5,
        )
    except Exception:
        return None

    if not chord_events:
        return None

    chord_matrix = np.zeros((total_steps, PR_CHORD_DIM), dtype=np.float32)
    for step in range(total_steps):
        beat_offset = step / steps_per_beat
        chord = get_chord_at(chord_events, beat_offset)
        chord_matrix[step] = _chord_vector(chord)

    return piano_roll, chord_matrix


def _transpose_roll(piano_roll: np.ndarray,
                    semitones: int) -> Optional[np.ndarray]:
    """
    Transpone un piano roll. Devuelve None si alguna nota queda
    fuera del rango [PR_PITCH_MIN, PR_PITCH_MAX].
    """
    result = piano_roll.copy()
    for i, tok in enumerate(result):
        if 0 <= tok <= 127:
            new_pitch = tok + semitones
            if not (PR_PITCH_MIN <= new_pitch <= PR_PITCH_MAX):
                return None  # nota fuera de rango — descartar transposición
            result[i] = new_pitch
    return result


def preprocess_corpus(corpus_dir: str,
                      output_path: str = "corpus.npz",
                      beats_per_bar: float = 4.0,
                      max_files: int = 0,
                      verbose: bool = False) -> int:
    """
    Preprocesa el corpus de MIDIs y guarda el dataset en un fichero .npz.

    Para cada MIDI:
      1. Convierte a piano roll + chord_matrix.
      2. Aplica transposición ±PR_MAX_TRANSP semitonos.
      3. Extrae ventanas de 64 pasos con stride 16.

    El .npz contiene dos arrays:
      'rolls'  : int16[N, 64]   — ventanas de piano roll
      'chords' : float32[N, 64, 16] — chord_matrix por ventana

    Devuelve el número de ventanas generadas.
    """
    if not MIDO_OK:
        raise RuntimeError("mido requerido: pip install mido")

    path  = Path(corpus_dir)
    files = sorted(list(path.rglob("*.mid")) + list(path.rglob("*.midi")))
    if max_files > 0:
        files = files[:max_files]

    window_size = 64
    stride      = 16

    all_rolls:  List[np.ndarray] = []
    all_chords: List[np.ndarray] = []

    ok = skip = 0
    for f in files:
        result = _midi_to_piano_roll(str(f), beats_per_bar=beats_per_bar)
        if result is None:
            skip += 1
            continue

        piano_roll, chord_matrix = result

        # Generar transposiciones
        transpositions = [0] + list(range(-PR_MAX_TRANSP, 0)) + \
                         list(range(1, PR_MAX_TRANSP + 1))

        for semitones in transpositions:
            if semitones == 0:
                roll = piano_roll
            else:
                roll = _transpose_roll(piano_roll, semitones)
                if roll is None:
                    continue

            # La chord_matrix no cambia con la transposición (es relativa al acorde)
            n_steps = len(roll)
            if n_steps < window_size:
                continue

            # Extraer ventanas solapadas
            for start in range(0, n_steps - window_size + 1, stride):
                w_roll   = roll[start:start + window_size]
                w_chords = chord_matrix[start:start + window_size]
                all_rolls.append(w_roll)
                all_chords.append(w_chords)

        ok += 1
        if verbose and ok % 50 == 0:
            print(f"  [preprocess] {ok}/{len(files)} MIDIs  "
                  f"ventanas={len(all_rolls)}")

    if not all_rolls:
        raise ValueError("No se generaron ventanas. Revisa el corpus.")

    rolls_arr  = np.stack(all_rolls,  axis=0).astype(np.int16)
    chords_arr = np.stack(all_chords, axis=0).astype(np.float32)

    np.savez_compressed(output_path,
                        rolls=rolls_arr,
                        chords=chords_arr)

    if verbose:
        print(f"  [preprocess] MIDIs ok={ok} skip={skip}")
        print(f"  [preprocess] Ventanas={len(all_rolls)}  "
              f"shape rolls={rolls_arr.shape}  "
              f"chords={chords_arr.shape}")
        print(f"  [preprocess] Guardado en: {output_path}")

    return len(all_rolls)


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM — ARQUITECTURA
# ══════════════════════════════════════════════════════════════════════════════

class MelodyLSTM(nn.Module if TORCH_OK else object):
    """
    LSTM ligero para generación de melodía condicionada a acordes.

    Arquitectura:
      Embedding(PR_VOCAB, d_embed)         → emb[B, T, d_embed]
      concat(emb, chord_vec[B, T, 16])     → inp[B, T, d_embed+16]
      LSTM(d_embed+16, hidden, num_layers) → out[B, T, hidden]
      Linear(hidden, PR_VOCAB)             → logits[B, T, PR_VOCAB]

    En entrenamiento: teacher forcing completo sobre chunks de seq_len pasos.
    En inferencia: autoregresivo paso a paso, manteniendo el estado oculto.

    Parámetros por defecto (hidden=128, d_embed=32, layers=1): ~114K
    """

    def __init__(self, hidden: int = 128, d_embed: int = 32,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.hidden     = hidden
        self.num_layers = num_layers
        self.embedding  = nn.Embedding(PR_VOCAB, d_embed)
        self.lstm       = nn.LSTM(
            input_size  = d_embed + PR_CHORD_DIM,
            hidden_size = hidden,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden, PR_VOCAB)

    def forward(self, tokens: "torch.Tensor",
                chord_vecs: "torch.Tensor",
                hidden: Optional[Tuple] = None
                ) -> Tuple["torch.Tensor", Tuple]:
        """
        tokens:     int64[B, T]
        chord_vecs: float32[B, T, 16]
        →  logits: float32[B, T, PR_VOCAB]
           hidden: nuevo estado oculto
        """
        emb  = self.embedding(tokens)              # (B, T, d_embed)
        inp  = torch.cat([emb, chord_vecs], dim=-1)  # (B, T, d_embed+16)
        out, hidden = self.lstm(inp, hidden)       # (B, T, hidden)
        logits = self.out_proj(out)                # (B, T, PR_VOCAB)
        return logits, hidden

    def init_hidden(self, batch_size: int,
                    device: "torch.device") -> Tuple:
        """Inicializa el estado oculto a ceros."""
        zeros = torch.zeros(self.num_layers, batch_size,
                            self.hidden, device=device)
        return (zeros, zeros)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_lstm(model: "MelodyLSTM", path: str,
              hparams: Dict[str, Any]):
    """Guarda el modelo y sus hiperparámetros."""
    torch.save({"state_dict": model.state_dict(),
                "hparams": hparams}, path)


def load_lstm(path: str) -> "MelodyLSTM":
    """Carga un modelo LSTM guardado con save_lstm."""
    if not TORCH_OK:
        raise RuntimeError("torch requerido: pip install torch")
    ckpt   = torch.load(path, map_location="cpu", weights_only=False)
    hp     = ckpt["hparams"]
    model  = MelodyLSTM(hidden=hp["hidden"], d_embed=hp["d_embed"],
                        num_layers=hp["num_layers"], dropout=hp.get("dropout", 0.1))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM — ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(dataset_path: str,
               model_out: str = "lstm.pt",
               epochs: int = 30,
               hidden: int = 128,
               d_embed: int = 32,
               num_layers: int = 1,
               dropout: float = 0.1,
               batch_size: int = 16,
               chunk_size: int = 32,
               lr: float = 0.001,
               seed: int = 42,
               verbose: bool = False) -> "MelodyLSTM":
    """
    Entrena la LSTM sobre el dataset preprocesado.

    Estrategia de memoria (truncated BPTT):
      - Cada ventana de 64 pasos se divide en chunks de chunk_size.
      - El estado oculto se mantiene entre chunks pero el grafo de
        gradientes se corta (detach) al inicio de cada chunk.
      - Esto reduce la memoria de gradientes en ~64/chunk_size veces.

    Con batch_size=16, chunk_size=32, hidden=128:
      Tensor de activaciones por chunk: 16×32×128 × 4 bytes ≈ 1MB.
      Perfectamente viable en CPU con 4GB de RAM.
    """
    if not TORCH_OK:
        raise RuntimeError("torch requerido: pip install torch")

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")  # diseñado explícitamente para CPU

    # ── Cargar dataset ────────────────────────────────────────────────────────
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

    data     = np.load(dataset_path)
    rolls    = data["rolls"]    # int16[N, 64]
    chords   = data["chords"]   # float32[N, 64, 16]
    n_total  = len(rolls)

    if verbose:
        print(f"  [train] Dataset: {n_total} ventanas  "
              f"shape={rolls.shape}")

    # Split 90/10
    idx      = np.random.permutation(n_total)
    n_val    = max(1, n_total // 10)
    idx_train, idx_val = idx[n_val:], idx[:n_val]

    # ── Modelo ────────────────────────────────────────────────────────────────
    model = MelodyLSTM(hidden=hidden, d_embed=d_embed,
                       num_layers=num_layers, dropout=dropout).to(device)
    if verbose:
        print(f"  [train] Parámetros: {model.count_params():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    win_len = rolls.shape[1]  # 64

    def run_epoch(indices: np.ndarray,
                  train: bool) -> float:
        """Ejecuta una época sobre el subconjunto dado."""
        model.train(train)
        np.random.shuffle(indices)
        total_loss = 0.0
        n_batches  = 0

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            if len(batch_idx) == 0:
                continue

            # Cargar batch
            b_rolls  = torch.tensor(rolls[batch_idx],
                                    dtype=torch.long,   device=device)  # (B, 64)
            b_chords = torch.tensor(chords[batch_idx],
                                    dtype=torch.float32, device=device)  # (B, 64, 16)
            B = b_rolls.size(0)

            # Inicializar estado oculto para el batch completo
            h = model.init_hidden(B, device)

            batch_loss = 0.0
            n_chunks   = 0

            # Truncated BPTT: iterar sobre chunks
            for c_start in range(0, win_len - 1, chunk_size):
                c_end = min(c_start + chunk_size, win_len - 1)

                # Desconectar gradientes del estado anterior
                h = (h[0].detach(), h[1].detach())

                inp_tok = b_rolls[:, c_start:c_end]       # (B, chunk)
                tgt_tok = b_rolls[:, c_start+1:c_end+1]   # (B, chunk)
                c_vecs  = b_chords[:, c_start:c_end]      # (B, chunk, 16)

                with torch.set_grad_enabled(train):
                    logits, h = model(inp_tok, c_vecs, h)
                    # logits: (B, chunk, PR_VOCAB)
                    loss = criterion(
                        logits.reshape(-1, PR_VOCAB),
                        tgt_tok.reshape(-1),
                    )

                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                batch_loss += loss.item()
                n_chunks   += 1

            total_loss += batch_loss / max(n_chunks, 1)
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    best_val  = float("inf")
    best_path = model_out

    print(f"[train] Entrenando LSTM  epochs={epochs}  "
          f"hidden={hidden}  d_embed={d_embed}  layers={num_layers}")
    print(f"        batch={batch_size}  chunk={chunk_size}  lr={lr}")
    print(f"        train={len(idx_train)}  val={len(idx_val)} ventanas")

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(idx_train, train=True)
        val_loss   = run_epoch(idx_val,   train=False)
        scheduler.step(val_loss)

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            hparams  = {"hidden": hidden, "d_embed": d_embed,
                        "num_layers": num_layers, "dropout": dropout}
            save_lstm(model, best_path, hparams)
            marker = "  ✓ guardado"

        if verbose or epoch % 5 == 0 or epoch == epochs:
            print(f"  época {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}{marker}")

    print(f"[train] Completado. Mejor val_loss={best_val:.4f}  → {best_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM — MOTOR DE GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════════

class LSTMEngine:
    """
    Generación de melodía condicionada con la LSTM entrenada.

    Proceso:
      1. Construye la chord_matrix para toda la progresión dada.
      2. Genera el piano roll paso a paso (autoregresivo), pasando
         el chord_vector del paso actual como condicionamiento.
      3. Convierte el piano roll a MelodyNote.
      4. Aplica snap_to_scale y ajuste de rango post-generación.

    Sampling:
      temperature > 1 → más aleatorio
      temperature < 1 → más determinista
      top_k > 0       → solo los top_k tokens más probables
    """

    def __init__(self, model: "MelodyLSTM", profile: str,
                 rng: random.Random,
                 temperature: float = 1.0, top_k: int = 10):
        self.model       = model
        self.profile     = profile
        self.rng         = rng
        self.temperature = max(temperature, 1e-3)
        self.top_k       = top_k

    def _build_chord_matrix(self,
                             chord_events: List[ChordEvent],
                             n_steps: int,
                             steps_per_beat: int) -> "torch.Tensor":
        """
        Construye chord_matrix float32[1, n_steps, 16] para la inferencia.
        """
        mat = np.zeros((n_steps, PR_CHORD_DIM), dtype=np.float32)
        for step in range(n_steps):
            beat_offset = step / steps_per_beat
            chord = get_chord_at(chord_events, beat_offset)
            mat[step] = _chord_vector(chord)
        return torch.tensor(mat, dtype=torch.float32).unsqueeze(0)  # (1, T, 16)

    def _sample(self, logits: "torch.Tensor") -> int:
        """Muestrea un token desde logits[PR_VOCAB]."""
        logits = logits / self.temperature
        if self.top_k > 0:
            # Enmascara todo excepto los top_k valores
            top_vals, _ = torch.topk(logits, min(self.top_k, logits.size(-1)))
            threshold   = top_vals[-1]
            logits      = logits.masked_fill(logits < threshold, float("-inf"))
        probs = torch.softmax(logits, dim=-1)
        # Usar numpy para reproducibilidad con la semilla del RNG propio
        probs_np = probs.detach().cpu().numpy()
        # Normalizar por si hay NaN o suma ≠ 1 por precisión
        probs_np = np.clip(probs_np, 0, None)
        s = probs_np.sum()
        if s <= 0:
            return PR_SILENCE
        probs_np /= s
        return int(self.rng.choices(range(PR_VOCAB), weights=probs_np)[0])

    def _piano_roll_to_notes(self,
                              roll: List[int],
                              steps_per_beat: float,
                              beats_per_bar: float,
                              vel_min: int, vel_max: int,
                              total_beats: float) -> List[MelodyNote]:
        """
        Convierte una lista de tokens de piano roll a MelodyNote.

        Agrupa tokens hold con su nota de inicio, calcula duraciones
        y asigna velocidades basadas en la posición temporal.
        """
        notes: List[MelodyNote] = []
        i = 0
        while i < len(roll):
            tok = roll[i]
            if tok == PR_SILENCE or tok == PR_HOLD:
                i += 1
                continue
            # Inicio de nota real: contar duración (hold continuos)
            pitch = int(tok)
            j = i + 1
            while j < len(roll) and roll[j] == PR_HOLD:
                j += 1
            duration_steps = j - i
            offset_beats   = i / steps_per_beat
            duration_beats = duration_steps / steps_per_beat

            if offset_beats >= total_beats:
                break

            # Velocidad dinámica: arco sinusoidal según posición
            frac   = offset_beats / max(total_beats, 1)
            climax = math.sin(frac * math.pi)
            vel    = int(vel_min + (vel_max - vel_min) * max(0.0, climax))
            vel    = max(vel_min, min(vel_max, vel))

            notes.append(MelodyNote(
                pitch=pitch,
                duration=round(min(duration_beats,
                                   total_beats - offset_beats), 4),
                velocity=vel,
                offset=round(offset_beats, 4),
            ))
            i = j

        return notes

    def generate(self,
                 chord_events: List[ChordEvent],
                 scale_pitches: List[int],
                 n_bars: int,
                 beats_per_bar: float,
                 contour: List[float],
                 tension: List[float],
                 pitch_range: Tuple[int, int]) -> List[MelodyNote]:

        if not TORCH_OK:
            raise RuntimeError("torch requerido: pip install torch")

        low, high = pitch_range
        vel_min, vel_max = PROFILE_VELOCITY.get(self.profile, (50, 90))
        steps_per_beat = PR_STEPS_PER_BEAT
        total_steps    = int(n_bars * beats_per_bar * steps_per_beat)
        total_beats    = n_bars * beats_per_bar

        # Chord matrix para toda la secuencia
        chord_mat = self._build_chord_matrix(chord_events, total_steps,
                                             steps_per_beat)  # (1, T, 16)

        self.model.eval()
        roll: List[int] = []

        with torch.no_grad():
            h = self.model.init_hidden(1, torch.device("cpu"))

            # Primer token: chord tone del primer acorde, cerca del centro del rango
            first_chord = chord_events[0] if chord_events else None
            if first_chord and first_chord.chord_pcs:
                mid_range = (low + high) // 2
                candidates = [p for p in range(low, high + 1)
                              if p % 12 in first_chord.chord_pcs]
                if candidates:
                    cur_tok = min(candidates, key=lambda p: abs(p - mid_range))
                else:
                    cur_tok = snap_to_scale(mid_range, scale_pitches)
            else:
                cur_tok = snap_to_scale((low + high) // 2, scale_pitches)

            cur_tok = max(PR_PITCH_MIN, min(PR_PITCH_MAX, cur_tok))
            roll.append(cur_tok)

            for step in range(1, total_steps):
                inp_tok = torch.tensor([[cur_tok]], dtype=torch.long)  # (1,1)
                c_vec   = chord_mat[:, step - 1:step, :]               # (1,1,16)

                logits, h = self.model(inp_tok, c_vec, h)
                logits = logits[0, 0]  # (PR_VOCAB,)

                # Condicionamiento por contorno: sesgar hacia zona alta/baja
                bar_idx = int((step / steps_per_beat) // beats_per_bar)
                c_val   = contour[min(bar_idx, len(contour) - 1)]
                t_val   = tension[min(bar_idx, len(tension) - 1)]
                target_pitch = low + c_val * (high - low)

                # Suavizar el bias: notas lejanas del target pierden peso
                pitch_bias = torch.zeros(PR_VOCAB)
                for p in range(PR_PITCH_MIN, PR_PITCH_MAX + 1):
                    dist = abs(p - target_pitch) / max(high - low, 1)
                    pitch_bias[p] = -dist * 2.0  # penalización proporcional

                # Mayor tensión → más tolerancia a notas no-escala
                if t_val < 0.5:
                    # Penalizar pitches fuera de la escala
                    scale_set = {p % 12 for p in scale_pitches}
                    for p in range(PR_PITCH_MIN, PR_PITCH_MAX + 1):
                        if p % 12 not in scale_set:
                            pitch_bias[p] -= 3.0 * (1 - t_val * 2)

                logits = logits + pitch_bias

                # Evitar hold al principio de compás (tiempo fuerte)
                step_in_bar = step % int(beats_per_bar * steps_per_beat)
                if step_in_bar == 0:
                    logits[PR_HOLD] = float("-inf")

                # Evitar saltos muy grandes del último pitch real
                if cur_tok not in (PR_SILENCE, PR_HOLD):
                    for p in range(PR_PITCH_MIN, PR_PITCH_MAX + 1):
                        if abs(p - cur_tok) > 12:
                            logits[p] -= 2.0

                cur_tok = self._sample(logits)
                roll.append(cur_tok)

                # Actualizar cur_tok de referencia para el sesgo de salto
                if cur_tok not in (PR_SILENCE, PR_HOLD):
                    pass  # cur_tok ya es el nuevo pitch real

        # Convertir roll a notas
        notes = self._piano_roll_to_notes(
            roll, steps_per_beat, beats_per_bar,
            vel_min, vel_max, total_beats)

        # Snap a escala y ajuste de rango
        processed: List[MelodyNote] = []
        for n in notes:
            p = n.pitch
            if p % 12 not in {s % 12 for s in scale_pitches}:
                p = snap_to_scale(p, scale_pitches)
            while p < low and p + 12 <= high:
                p += 12
            while p > high and p - 12 >= low:
                p -= 12
            p = max(low, min(high, p))
            processed.append(MelodyNote(p, n.duration, n.velocity, n.offset))

        return processed


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING EXTENDIDO (8 criterios)
# ══════════════════════════════════════════════════════════════════════════════

def score_melody(notes: List[MelodyNote],
                 chord_events: List[ChordEvent],
                 scale_pitches: List[int],
                 contour: List[float],
                 beats_per_bar: float,
                 profile: str) -> Tuple[float, Dict[str, float]]:
    """
    Puntuación multicriterio [0, 1].
    Devuelve (score_global, detalle_por_criterio).

    Criterios:
      1. Consonancia con la escala           (15%)
      2. Armonía por acorde activo           (20%)
      3. Suavidad de movimiento              (15%)
      4. Conformidad con el contorno         (15%)
      5. Rango melódico razonable            (10%)
      6. Variedad rítmica                    (10%)
      7. Arco dinámico (std velocidades)     ( 8%)
      8. Posición del clímax (perfil)        ( 7%)
    """
    if not notes:
        return 0.0, {}

    pitches   = [n.pitch for n in notes]
    durs      = [n.duration for n in notes]
    vels      = [n.velocity for n in notes]
    scale_set = {p % 12 for p in scale_pitches}

    # 1. Consonancia
    consonance = sum(1 for p in pitches if p % 12 in scale_set) / len(pitches)

    # 2. Armonía por acorde activo (penaliza avoid en tiempos fuertes)
    chord_vals = []
    for n in notes:
        chord = get_chord_at(chord_events, n.offset)
        ntype = chord_note_type(n.pitch, chord, scale_pitches)
        base  = {"CT": 1.0, "T": 0.7, "PT": 0.5, "NT": 0.1}[ntype]
        if is_avoid_note(n.pitch, chord) and is_strong_beat(n.offset, beats_per_bar):
            base *= 0.2
        chord_vals.append(base)
    harmony_score = float(np.mean(chord_vals)) if chord_vals else 0.5

    # 3. Suavidad
    intervals  = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches) - 1)]
    big_leaps  = sum(1 for iv in intervals if iv > 7)
    smoothness = max(0.0, 1.0 - big_leaps / max(len(intervals), 1))

    # 4. Contorno
    low, high = min(pitches), max(pitches)
    rng_v = high - low or 1
    c_errors = []
    for n in notes:
        bar_idx = int(n.offset // beats_per_bar)
        c_val   = contour[min(bar_idx, len(contour) - 1)]
        c_errors.append(abs((n.pitch - low) / rng_v - c_val))
    contour_score = max(0.0, 1.0 - float(np.mean(c_errors)))

    # 5. Rango melódico (óptimo: 7-17 semitonos)
    rng_st   = max(pitches) - min(pitches)
    range_sc = min(1.0, max(0.0, 1 - abs(rng_st - 12) / 12))

    # 6. Variedad rítmica
    variety = min(1.0, len(set(round(d, 2) for d in durs)) / 5)

    # 7. Arco dinámico
    vel_std   = float(np.std(vels)) if len(vels) >= 4 else 5.0
    arc_score = min(1.0, vel_std / 15.0)

    # 8. Posición del clímax
    climax_target = PROFILE_CLIMAX_POS.get(profile, 0.618)
    climax_pos    = pitches.index(max(pitches)) / max(len(pitches) - 1, 1)
    climax_score  = max(0.0, 1.0 - abs(climax_pos - climax_target))

    detail = {
        "consonance":    consonance,
        "harmony":       harmony_score,
        "smoothness":    smoothness,
        "contour":       contour_score,
        "range":         range_sc,
        "variety":       variety,
        "arc":           arc_score,
        "climax":        climax_score,
    }
    total = (consonance    * 0.15 +
             harmony_score * 0.20 +
             smoothness    * 0.15 +
             contour_score * 0.15 +
             range_sc      * 0.10 +
             variety       * 0.10 +
             arc_score     * 0.08 +
             climax_score  * 0.07)

    return total, detail


def score_melody_vs_progression(notes: List[MelodyNote],
                                 chord_events: List[ChordEvent],
                                 scale_pitches: List[int],
                                 beats_per_bar: float
                                 ) -> Dict[str, Any]:
    """
    Análisis de compatibilidad por acorde: score por segmento + lista de colisiones.
    Útil como informe post-generación.
    """
    scores_per_chord = []
    collisions = []

    for i, ev in enumerate(chord_events):
        end = (chord_events[i + 1].offset
               if i + 1 < len(chord_events)
               else ev.offset + ev.duration)
        mel_in_window = [n for n in notes
                         if ev.offset <= n.offset < end]
        if not mel_in_window:
            continue
        chord_vals = []
        for n in mel_in_window:
            ntype = chord_note_type(n.pitch, ev, scale_pitches)
            chord_vals.append({"CT": 1.0, "T": 0.7, "PT": 0.5, "NT": 0.1}[ntype])
            if is_avoid_note(n.pitch, ev) and is_strong_beat(n.offset, beats_per_bar):
                collisions.append({
                    "offset": round(n.offset, 3),
                    "note":   n.name,
                    "chord":  ev.name,
                })
        scores_per_chord.append({
            "chord": ev.name,
            "start": round(ev.offset, 2),
            "score": round(float(np.mean(chord_vals)), 4),
        })

    global_score = (float(np.mean([s["score"] for s in scores_per_chord]))
                    if scores_per_chord else 0.0)
    return {
        "global_score": round(global_score, 4),
        "by_chord":     scores_per_chord,
        "collisions":   collisions,
    }


def extract_motif(notes: List[MelodyNote], beats_per_bar: float,
                  motif_bars: int = 2) -> List[MelodyNote]:
    """Extrae las primeras motif_bars de la melodía como motivo semilla."""
    threshold = motif_bars * beats_per_bar
    return [n for n in notes if n.offset < threshold]


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT MIDI
# ══════════════════════════════════════════════════════════════════════════════

def notes_to_midi(notes: List[MelodyNote], chord_events: List[ChordEvent],
                  tempo_bpm: int, output_path: str,
                  ticks_per_beat: int = 480,
                  include_chords: bool = False,
                  source_midi_path: Optional[str] = None,
                  source_chord_track_idx: int = 2):
    """
    Exporta la melodía a MIDI (track 0 meta, track 1 melodía, track 2 acordes).
    Si include_chords=True y hay fuente MIDI, copia el track original verbatim.
    """
    if not MIDO_OK:
        print("[ERROR] mido no disponible.")
        return

    mid   = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo = int(60_000_000 / max(tempo_bpm, 1))

    def q_ticks(q: float) -> int:
        return int(q * ticks_per_beat)

    # Track 0: metadatos
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    # Track 1: melodía
    mel = mido.MidiTrack()
    mid.tracks.append(mel)
    mel.append(mido.MetaMessage("track_name", name="Melody", time=0))
    mel.append(mido.Message("program_change", channel=0, program=73, time=0))

    evts = []
    for n in notes:
        if n.pitch < 0:
            continue
        evts.append((q_ticks(n.offset),            "note_on",  n.pitch, n.velocity))
        evts.append((q_ticks(n.offset + n.duration), "note_off", n.pitch, 0))
    evts.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
    cur = 0
    for abs_t, mtype, note, vel in evts:
        delta = max(0, abs_t - cur)
        mel.append(mido.Message(mtype, channel=0, note=note,
                                velocity=vel, time=delta))
        cur = abs_t

    # Track 2: acordes (opcional)
    if include_chords:
        copied = False
        if source_midi_path and os.path.exists(source_midi_path):
            try:
                src     = mido.MidiFile(source_midi_path)
                src_tpb = src.ticks_per_beat or 480
                if source_chord_track_idx < len(src.tracks):
                    ratio = ticks_per_beat / src_tpb
                    acc   = mido.MidiTrack()
                    mid.tracks.append(acc)
                    acc.append(mido.MetaMessage("track_name", name="Chords", time=0))
                    for msg in src.tracks[source_chord_track_idx]:
                        if not msg.is_meta:
                            acc.append(msg.copy(time=int(round(msg.time * ratio))))
                    copied = True
            except Exception as e:
                print(f"[WARN] No se pudo copiar track de acordes: {e}")

        if not copied and chord_events:
            acc = mido.MidiTrack()
            mid.tracks.append(acc)
            acc.append(mido.MetaMessage("track_name", name="Chords", time=0))
            acc.append(mido.Message("program_change", channel=1, program=0, time=0))
            chord_evts = []
            for ev in chord_events:
                pitches_oct3 = [ev.root_pc + 48] + [
                    ev.root_pc + 48 + iv
                    for iv in CHORD_INTERVALS.get(ev.quality, [0, 4, 7])
                ]
                on_t  = q_ticks(ev.offset)
                off_t = q_ticks(ev.offset + ev.duration)
                for p in pitches_oct3:
                    if 0 <= p <= 127:
                        chord_evts.append((on_t,  "note_on",  p, 60))
                        chord_evts.append((off_t, "note_off", p, 0))
            chord_evts.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))
            cur = 0
            for abs_t, mtype, note, vel in chord_evts:
                delta = max(0, abs_t - cur)
                acc.append(mido.Message(mtype, channel=1, note=note,
                                        velocity=vel, time=delta))
                cur = abs_t

    mid.save(output_path)


def motif_to_midi(motif: List[MelodyNote], tempo_bpm: int,
                  output_path: str, ticks_per_beat: int = 480):
    """Exporta el motivo semilla como MIDI independiente."""
    notes_to_midi(motif, [], tempo_bpm, output_path,
                  ticks_per_beat=ticks_per_beat)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class ConditionedMelodyGenerator:
    """Coordina motores, condicionamiento emocional y post-procesado."""

    def __init__(self,
                 key: str = "C", mode: str = "auto",
                 bars: int = 16, tempo: int = 120,
                 time_sig: str = "4/4",
                 engine: str = "chord_guided",
                 profile: str = "serene",
                 rhythm: str = "flowing",
                 contour: str = "auto",
                 custom_contour: Optional[List[float]] = None,
                 tension_curve: Optional[List[float]] = None,
                 pitch_low: int = 60, pitch_high: int = 84,
                 chord_events: Optional[List[ChordEvent]] = None,
                 corpus_dir: Optional[str] = None,
                 ornament_types: Optional[List[str]] = None,
                 ornament_prob: float = 0.35,
                 fix_avoid: bool = False,
                 lstm_model: Optional["MelodyLSTM"] = None,
                 lstm_temperature: float = 1.0,
                 lstm_top_k: int = 10,
                 # motif
                 motif_notes: Optional[List[MelodyNote]] = None,
                 # interpolate
                 interp_profile_b: str = "tanguero",
                 interp_alpha: float = 0.5,
                 # bayes (usa corpus_dir)
                 # tension_arc
                 phrase_bars: int = 4,
                 seed: int = 42, verbose: bool = False):

        self.key          = key
        self.bars         = bars
        self.tempo        = tempo
        self.engine       = engine
        self.profile      = profile
        self.rhythm       = rhythm
        self.verbose      = verbose
        self.chord_events = chord_events or []
        self.corpus_dir   = corpus_dir
        self.ornament_types = ornament_types or []
        self.ornament_prob  = ornament_prob
        self.fix_avoid      = fix_avoid
        self._lstm_model       = lstm_model
        self._lstm_temperature = lstm_temperature
        self._lstm_top_k       = lstm_top_k
        self._motif_notes      = motif_notes
        self._interp_profile_b = interp_profile_b
        self._interp_alpha     = interp_alpha
        self._phrase_bars      = phrase_bars

        num, den = time_sig.split("/")
        self.beats_per_bar = float(num)
        self.rng = random.Random(seed)
        np.random.seed(seed)

        self.root_pc, detected_mode = parse_key(key)
        self.mode = detected_mode if mode == "auto" else mode
        if self.mode == "auto":
            self.mode = PROFILE_TO_MODE.get(profile, "major")

        self.scale_pitches = get_scale_pitches(
            self.root_pc, self.mode, pitch_low - 12, pitch_high + 12)
        self.scale_pitches = [p for p in self.scale_pitches
                               if pitch_low <= p <= pitch_high]
        if not self.scale_pitches:
            self.scale_pitches = get_scale_pitches(
                self.root_pc, self.mode, pitch_low - 24, pitch_high + 24)

        self.pitch_range = (pitch_low, pitch_high)

        contour_shape = contour
        if contour_shape == "auto":
            contour_shape = PROFILE_CONTOUR.get(profile, "arch")
        self.contour_shape = contour_shape
        self.contour_curve = resize_curve(
            build_contour_curve(contour_shape, bars, custom_contour), bars)

        self.tension_curve = (resize_curve(tension_curve, bars)
                              if tension_curve
                              else self._default_tension(profile, bars))

        if verbose:
            self._print_config()

    def _default_tension(self, profile: str, n: int) -> List[float]:
        t = np.linspace(0, 1, n)
        curves = {
            "heroic":      0.3 + 0.6 * np.sin(t * math.pi),
            "melancholic": 0.5 - 0.3 * t,
            "tense":       np.clip(0.4 + 0.5 * t, 0, 1),
            "playful":     0.4 + 0.2 * np.sin(t * 4 * math.pi),
            "serene":      0.2 + 0.15 * np.sin(t * 2 * math.pi),
            "mysterious":  0.3 + 0.4 * np.sin(t * math.pi),
            "triumphant":  np.clip(0.5 * t + 0.5 * np.sin(t * math.pi), 0, 1),
            # ── Perfiles nuevos ───────────────────────────────────────────────
            "elegiac":     0.55 - 0.25 * t + 0.15 * np.sin(t * 3 * math.pi),
            # Parte alta (0.55), desciende lentamente con ondulaciones internas
            # que representan los momentos de elevación antes de la caída final.
            "ecstatic":    np.clip(0.75 + 0.2 * np.sin(t * 2 * math.pi), 0, 1),
            # Alta desde el principio (0.75) con oscilaciones rápidas —
            # la euforia no crece, ya está ahí, solo pulsa.
            "brooding":    np.clip(0.7 - 0.1 * t + 0.05 * np.sin(t * 5 * math.pi), 0, 1),
            # Alta y casi plana (0.7), con microoscilaciones — el peso oscuro
            # no fluctúa mucho, solo pesa.
            "pastoral":    0.15 + 0.1 * np.sin(t * 2 * math.pi),
            # Muy baja (0.15) con oscilaciones suaves — el campo es tranquilo,
            # sin acumulación de tensión en ningún momento.
            "agitated":    np.clip(0.5 + 0.4 * np.abs(np.sin(t * 7 * math.pi)), 0, 1),
            # Oscilaciones rápidas e irregulares en la zona media-alta —
            # la agitación no tiene arco, solo espasmos.
            "hypnotic":    np.full(n, 0.25),
            # Completamente plana — la hipnosis no tiene narrativa de tensión,
            # es un estado sostenido sin principio ni fin.
            "flamenco":    np.clip(0.35 + 0.5 * np.sin(t * math.pi)
                                   + 0.15 * np.sin(t * 5 * math.pi), 0, 1),
            # Arco principal (sube al clímax) con micro-oscilaciones rápidas
            # superpuestas — el duende surge en picos dentro de la frase.
            "tanguero":    0.4 + 0.35 * np.sin(t * 4 * math.pi),
            # Cuatro ciclos completos en la obra — cada frase de tango tiene
            # su propio mini-arco de tensión-resolución.
        }
        return list(curves.get(profile, np.full(n, 0.4)))

    def _print_config(self):
        scale_names = [NOTE_NAMES[(self.root_pc + iv) % 12]
                       for iv in SCALE_INTERVALS.get(self.mode, [])]
        print(f"┌─ MELODY CONDITIONED v2 ─────────────────────────────")
        print(f"│  Motor     : {self.engine}")
        print(f"│  Tonalidad : {self.key} {self.mode}")
        print(f"│  Escala    : {' '.join(scale_names)}")
        print(f"│  Compases  : {self.bars}  Tempo: {self.tempo} BPM")
        print(f"│  Perfil    : {self.profile}  Contorno: {self.contour_shape}")
        print(f"│  Acordes   : {len(self.chord_events)} eventos")
        if self.chord_events:
            print(f"│  Progresión: {chord_events_to_text(self.chord_events[:8])}")
        if self.ornament_types:
            print(f"│  Ornamentos: {', '.join(self.ornament_types)}  "
                  f"prob={self.ornament_prob:.2f}")
        if self.fix_avoid:
            print(f"│  fix-avoid : activado")
        print(f"└─────────────────────────────────────────────────────")

    def generate_once(self, seed_offset: int = 0) -> MelodyResult:
        rng = random.Random(self.rng.randint(0, 2**31) + seed_offset)

        if self.engine == "chord_guided":
            eng = ChordGuidedEngine(self.profile, self.rhythm, rng)
            if self.corpus_dir:
                eng.train_from_corpus(self.corpus_dir, self.verbose)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "grammar":
            eng = GrammarEngine(self.root_pc, self.mode, rng)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "search":
            eng = SearchEngine(self.profile, rng)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "genetic":
            eng = GeneticEngine(self.profile, self.rhythm,
                                population_size=20, generations=30, rng=rng)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "lstm":
            if self._lstm_model is None:
                raise ValueError(
                    "Motor 'lstm' requiere --model con un modelo entrenado "
                    "o --train para entrenar primero.")
            eng = LSTMEngine(self._lstm_model, self.profile, rng,
                             self._lstm_temperature, self._lstm_top_k)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "motif":
            eng = MotifEngine(self.profile, rng, motif=self._motif_notes)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "bayes":
            eng = BayesEngine(self.profile, self.rhythm, rng)
            if self.corpus_dir:
                eng.train_from_corpus(self.corpus_dir, self.verbose)
            else:
                eng._build_synthetic()
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "tension_arc":
            eng = TensionArcEngine(self.profile, self.rhythm, rng,
                                   phrase_bars=self._phrase_bars)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        elif self.engine == "interpolate":
            eng = InterpolateEngine(self.profile, self._interp_profile_b,
                                    self._interp_alpha, self.rhythm, rng)
            notes = eng.generate(
                self.chord_events, self.scale_pitches,
                self.bars, self.beats_per_bar,
                self.contour_curve, self.tension_curve, self.pitch_range)

        else:
            raise ValueError(
                f"Motor desconocido: {self.engine}. "
                f"Usa: chord_guided | grammar | search | genetic | lstm | "
                f"motif | bayes | tension_arc | interpolate")

        # ── Post-procesado ────────────────────────────────────────────────────
        if self.fix_avoid:
            notes = fix_avoid_notes(notes, self.chord_events,
                                    self.scale_pitches, self.beats_per_bar)

        if self.ornament_types:
            notes = apply_ornaments(notes, self.chord_events, self.scale_pitches,
                                    self.beats_per_bar, self.ornament_types,
                                    self.ornament_prob, rng)

        sc, detail = score_melody(notes, self.chord_events, self.scale_pitches,
                                   self.contour_curve, self.beats_per_bar,
                                   self.profile)

        return MelodyResult(
            notes=notes, key=self.key, mode=self.mode,
            bars=self.bars, tempo=self.tempo,
            engine=self.engine, profile=self.profile,
            score=sc, score_detail=detail,
            metadata={"contour": self.contour_shape,
                      "rhythm": self.rhythm,
                      "chord_progression": chord_events_to_text(self.chord_events),
                      "ornaments": self.ornament_types,
                      "fix_avoid": self.fix_avoid},
        )

    def generate_candidates(self, n: int = 3) -> List[MelodyResult]:
        candidates = []
        for i in range(n):
            r = self.generate_once(seed_offset=i * 1337)
            candidates.append(r)
            if self.verbose:
                print(f"  [candidato {i+1}/{n}] score={r.score:.3f}  "
                      f"notas={len(r.notes)}  "
                      f"harmony={r.score_detail.get('harmony', 0):.2f}  "
                      f"climax={r.score_detail.get('climax', 0):.2f}")
        return sorted(candidates, key=lambda r: -r.score)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MELODY CONDITIONED v2 — Generación condicionada a acordes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--engine", default="chord_guided",
                   choices=["chord_guided", "grammar", "search", "genetic",
                            "lstm", "motif", "bayes", "tension_arc", "interpolate"])

    # ── Modos especiales ──────────────────────────────────────────────────────
    p.add_argument("--train",      action="store_true",
                   help="Entrenar la LSTM sobre el corpus antes de generar")
    p.add_argument("--preprocess", action="store_true",
                   help="Solo preprocesar el corpus a .npz, sin entrenar ni generar")

    # ── Motif ─────────────────────────────────────────────────────────────────
    p.add_argument("--motif", type=str, default=None,
                   help="Motivo semilla para --engine motif: \"C4:0.5 E4:1 G4:0.5\"")

    # ── Interpolate ───────────────────────────────────────────────────────────
    p.add_argument("--profile-b", type=str, default="tanguero",
                   choices=list(PROFILE_VELOCITY.keys()),
                   help="Segundo perfil para --engine interpolate (default: tanguero)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Interpolación: 0.0=perfil A, 1.0=perfil B (default: 0.5)")

    # ── Tension arc ───────────────────────────────────────────────────────────
    p.add_argument("--phrase-bars", type=int, default=4,
                   help="Compases por frase en tension_arc (default: 4)")

    p.add_argument("--key",      default="C")
    p.add_argument("--mode",     default="auto",
                   choices=list(SCALE_INTERVALS.keys()) + ["auto"])
    p.add_argument("--bars",     type=int,   default=16)
    p.add_argument("--tempo",    type=int,   default=120)
    p.add_argument("--time-sig", default="4/4")

    p.add_argument("--profile", default="serene",
                   choices=list(PROFILE_VELOCITY.keys()))
    p.add_argument("--rhythm",  default="flowing",
                   choices=list(RHYTHM_DISTS.keys()))
    p.add_argument("--contour", default="auto",
                   choices=["auto", "arch", "ascending", "descending",
                            "wave", "plateau", "inverted", "erratic", "custom"])
    p.add_argument("--contour-values", type=str, default=None)
    p.add_argument("--tension-curve",  type=str, default=None)

    p.add_argument("--range", nargs=2, type=int, default=[60, 84],
                   metavar=("LOW", "HIGH"))

    p.add_argument("--chords",      type=str, default=None)
    p.add_argument("--chords-json", type=str, default=None)
    p.add_argument("--chords-midi", type=str, default=None)
    p.add_argument("--chords-midi-track",   type=int, default=2)
    p.add_argument("--chords-midi-channel", type=int, default=1)
    p.add_argument("--chords-midi-window",  type=float, default=0.5)

    p.add_argument("--corpus", type=str, default=None,
                   help="MIDIs para entrenar Markov (chord_guided) o LSTM")
    p.add_argument("--dataset",   type=str, default="corpus.npz",
                   help="Fichero .npz preprocesado (default: corpus.npz)")
    p.add_argument("--model-out", type=str, default="lstm.pt",
                   help="Guardar modelo LSTM entrenado (default: lstm.pt)")
    p.add_argument("--model",     type=str, default=None,
                   help="Cargar modelo LSTM para generar")
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--d-embed",    type=int,   default=32)
    p.add_argument("--num-layers", type=int,   default=1)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--chunk-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=0.001)
    p.add_argument("--max-files",  type=int,   default=0,
                   help="Máximo MIDIs a cargar en preprocess (0=todos)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperatura de sampling LSTM (default: 1.0)")
    p.add_argument("--top-k",      type=int,   default=10,
                   help="Top-k sampling LSTM (0=desactivado, default: 10)")

    p.add_argument("--ornaments", nargs="+", default=[],
                   choices=["appoggiatura", "passing", "neighbor", "all"],
                   help="Tipos de ornamento post-generación")
    p.add_argument("--ornament-prob", type=float, default=0.35,
                   help="Probabilidad de ornamentación por nota (default: 0.35)")
    p.add_argument("--fix-avoid", action="store_true",
                   help="Corregir avoid notes en tiempos fuertes")

    p.add_argument("--output",         type=str, default="melody_conditioned.mid")
    p.add_argument("--include-chords", action="store_true")
    p.add_argument("--export-motif",   action="store_true",
                   help="Exportar también el motivo semilla (.motif.mid)")
    p.add_argument("--candidates",     type=int, default=3)

    p.add_argument("--seed",    type=int,  default=42)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Modo preprocesado puro ────────────────────────────────────────────────
    if args.preprocess:
        if not args.corpus:
            print("[ERROR] --corpus requerido para --preprocess")
            sys.exit(1)
        if not TORCH_OK:
            print("[ERROR] torch requerido para LSTM: pip install torch")
            sys.exit(1)
        print(f"[preprocess] Corpus: {args.corpus}")
        print(f"[preprocess] Salida: {args.dataset}")
        n = preprocess_corpus(
            corpus_dir  = args.corpus,
            output_path = args.dataset,
            max_files   = args.max_files,
            verbose     = args.verbose,
        )
        print(f"[preprocess] {n} ventanas generadas → {args.dataset}")
        sys.exit(0)

    root_pc, detected_mode = parse_key(args.key)
    mode = detected_mode if args.mode == "auto" else args.mode
    if mode == "auto":
        mode = PROFILE_TO_MODE.get(args.profile, "major")

    # ── Modo entrenamiento LSTM ───────────────────────────────────────────────
    lstm_model = None
    if args.train or args.engine == "lstm":
        if not TORCH_OK:
            print("[ERROR] torch requerido para LSTM: pip install torch")
            sys.exit(1)

    if args.train:
        # Si no existe el dataset, preprocesar primero
        if not os.path.exists(args.dataset):
            if not args.corpus:
                print(f"[ERROR] Dataset no encontrado ({args.dataset}) "
                      f"y --corpus no especificado.")
                sys.exit(1)
            print(f"[train] Dataset no encontrado. Preprocesando corpus…")
            preprocess_corpus(
                corpus_dir  = args.corpus,
                output_path = args.dataset,
                max_files   = args.max_files,
                verbose     = args.verbose,
            )

        lstm_model = train_lstm(
            dataset_path = args.dataset,
            model_out    = args.model_out,
            epochs       = args.epochs,
            hidden       = args.hidden,
            d_embed      = args.d_embed,
            num_layers   = args.num_layers,
            dropout      = 0.1,
            batch_size   = args.batch_size,
            chunk_size   = args.chunk_size,
            lr           = args.lr,
            seed         = args.seed,
            verbose      = args.verbose,
        )
        # Si solo se pidió entrenar (sin generar), salir
        if args.engine != "lstm":
            print("[train] Entrenamiento completado.")
            sys.exit(0)

    # ── Cargar modelo LSTM si es necesario ────────────────────────────────────
    if args.engine == "lstm" and lstm_model is None:
        model_path = args.model or args.model_out
        if not os.path.exists(model_path):
            print(f"[ERROR] Modelo LSTM no encontrado: {model_path}")
            print("        Usa --train para entrenar primero.")
            sys.exit(1)
        print(f"[lstm] Cargando modelo: {model_path}")
        lstm_model = load_lstm(model_path)

    # ── Resolver acordes ──────────────────────────────────────────────────────
    chord_events: List[ChordEvent] = []

    if args.chords_midi:
        if not os.path.exists(args.chords_midi):
            print(f"[ERROR] No se encuentra: {args.chords_midi}")
            sys.exit(1)
        print(f"[acordes] Extrayendo desde MIDI: {args.chords_midi}")
        chord_events = extract_chords_from_midi(
            midi_path         = args.chords_midi,
            root_pc_key       = root_pc,
            mode              = mode,
            harmony_track_idx = args.chords_midi_track,
            harmony_channel   = args.chords_midi_channel,
            min_window_beats  = args.chords_midi_window,
        )
        if args.verbose:
            print(f"  → {len(chord_events)} acordes extraídos")
            for ev in chord_events[:8]:
                print(f"    {ev.offset:.1f}  {ev.name}  ({ev.harm_func})")

    elif args.chords_json:
        if not os.path.exists(args.chords_json):
            print(f"[ERROR] No se encuentra: {args.chords_json}")
            sys.exit(1)
        chord_events = parse_chords_json(args.chords_json, root_pc, mode)

    elif args.chords:
        chord_events = parse_chords_text(args.chords, root_pc, mode)

    if not chord_events:
        scale = SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
        fallback = [0, 4, 5, 3] if len(scale) > 5 else [0, 4, 0, 4]
        print("[WARN] Sin acordes. Usando I-V-VI-IV por defecto.")
        text_parts = [f"{NOTE_NAMES[(root_pc + scale[min(d, len(scale)-1)]) % 12]}:4"
                      for d in fallback]
        chord_events = parse_chords_text(" ".join(text_parts), root_pc, mode)

    # ── Curvas ────────────────────────────────────────────────────────────────
    tension_curve = None
    if args.tension_curve:
        try:
            tension_curve = [float(v) for v in args.tension_curve.split(",")]
        except ValueError:
            print("[WARN] --tension-curve inválida.")

    custom_contour = None
    if args.contour == "custom" and args.contour_values:
        try:
            custom_contour = [float(v) for v in args.contour_values.split(",")]
        except ValueError:
            print("[WARN] --contour-values inválido.")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n── PARÁMETROS (dry-run) ─────────────────────────────")
        print(f"  engine   = {args.engine}")
        print(f"  key      = {args.key}  mode = {mode}")
        print(f"  bars     = {args.bars}  tempo = {args.tempo}")
        print(f"  profile  = {args.profile}  rhythm = {args.rhythm}")
        print(f"  contour  = {args.contour}")
        print(f"  range    = {args.range[0]}-{args.range[1]}")
        print(f"  chords   = {chord_events_to_text(chord_events)}")
        print(f"  ornaments = {args.ornaments}  prob={args.ornament_prob}")
        print(f"  fix-avoid = {args.fix_avoid}")
        if args.engine == "lstm":
            print(f"  temperature = {args.temperature}  top_k = {args.top_k}")
        if args.corpus:
            print(f"  corpus   = {args.corpus}")
        print("─────────────────────────────────────────────────────")
        sys.exit(0)

    # ── Parsear motivo si se proporcionó ──────────────────────────────────────
    motif_notes = None
    if args.motif:
        root_pc_tmp, _ = parse_key(args.key)
        scale_tmp = get_scale_pitches(root_pc_tmp, mode, 48, 96)
        motif_notes = parse_motif_text(args.motif, root_pc_tmp, scale_tmp)
        if not motif_notes:
            print(f"[WARN] --motif no pudo parsearse: '{args.motif}'")

    # ── Generar ───────────────────────────────────────────────────────────────
    gen = ConditionedMelodyGenerator(
        key            = args.key,
        mode           = args.mode,
        bars           = args.bars,
        tempo          = args.tempo,
        time_sig       = args.time_sig,
        engine         = args.engine,
        profile        = args.profile,
        rhythm         = args.rhythm,
        contour        = args.contour,
        custom_contour = custom_contour,
        tension_curve  = tension_curve,
        pitch_low      = args.range[0],
        pitch_high     = args.range[1],
        chord_events   = chord_events,
        corpus_dir     = args.corpus,
        ornament_types = args.ornaments,
        ornament_prob  = args.ornament_prob,
        fix_avoid      = args.fix_avoid,
        lstm_model     = lstm_model,
        lstm_temperature = args.temperature,
        lstm_top_k     = args.top_k,
        motif_notes    = motif_notes,
        interp_profile_b = args.profile_b,
        interp_alpha   = args.alpha,
        phrase_bars    = args.phrase_bars,
        seed           = args.seed,
        verbose        = args.verbose,
    )

    print(f"\nGenerando {args.candidates} candidato(s) con motor '{args.engine}'…")
    candidates = gen.generate_candidates(args.candidates)

    # ── Exportar ──────────────────────────────────────────────────────────────
    base = args.output.replace(".mid", "")
    output_paths = []

    for i, res in enumerate(candidates):
        suffix = "" if i == 0 else f"_v{i+1}"
        out_p  = f"{base}{suffix}.mid"
        notes_to_midi(
            res.notes, chord_events, res.tempo, out_p,
            include_chords=args.include_chords,
            source_midi_path=args.chords_midi,
            source_chord_track_idx=args.chords_midi_track,
        )
        output_paths.append(out_p)
        tag = "★ MEJOR" if i == 0 else f"  v{i+1}"
        print(f"  {tag}  score={res.score:.3f}  notas={len(res.notes)}  → {out_p}")

    if not candidates:
        return

    best = candidates[0]

    # ── Motivo semilla ────────────────────────────────────────────────────────
    if args.export_motif:
        motif = extract_motif(best.notes, gen.beats_per_bar, motif_bars=2)
        motif_path = f"{base}.motif.mid"
        motif_to_midi(motif, best.tempo, motif_path)
        print(f"[motif]  {motif_path}  ({len(motif)} notas)")

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_path = f"{base}.melody.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(best.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"[json]   {json_path}")

    # ── Análisis de compatibilidad ────────────────────────────────────────────
    if args.verbose:
        analysis = score_melody_vs_progression(
            best.notes, chord_events, gen.scale_pitches, gen.beats_per_bar)
        print(f"\n[análisis] score_global={analysis['global_score']:.3f}  "
              f"colisiones={len(analysis['collisions'])}")
        for s in analysis["by_chord"]:
            print(f"  {s['chord']:8s} @ {s['start']:5.1f}b  score={s['score']:.3f}")
        if analysis["collisions"]:
            print(f"  Colisiones:")
            for c in analysis["collisions"]:
                print(f"    beat={c['offset']:.2f}  {c['note']} sobre {c['chord']}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    prog_preview = chord_events_to_text(chord_events[:6])
    if len(chord_events) > 6:
        prog_preview += " …"
    print(f"\n╔═ RESUMEN ════════════════════════════════════════════╗")
    print(f"║  Motor     : {best.engine}")
    print(f"║  Tonalidad : {best.key} {best.mode}")
    print(f"║  Compases  : {best.bars}  Tempo: {best.tempo} BPM")
    print(f"║  Perfil    : {best.profile}")
    print(f"║  Notas     : {len(best.notes)}")
    print(f"║  Score     : {best.score:.3f}  "
          f"(harm={best.score_detail.get('harmony',0):.2f}  "
          f"contour={best.score_detail.get('contour',0):.2f}  "
          f"climax={best.score_detail.get('climax',0):.2f})")
    print(f"║  Progresión: {prog_preview}")
    print(f"╚══════════════════════════════════════════════════════╝")
    print(f"  Salida: {output_paths[0]}")


if __name__ == "__main__":
    main()
