#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              CYCLE-GAN STYLE TRANSFER  v3.1                                 ║
║       Transferencia de estilo bidireccional MIDI — arquitectura híbrida     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CORRECCIONES v3.1 (sobre v3.0)                                             ║
║                                                                              ║
║  Bug 1  — _ticks_per_bar eliminada. Devolvía tpb*4 (4/4 hardcodeado) y    ║
║           no se usaba en ningún sitio; fuente potencial de bugs futuros.   ║
║                                                                              ║
║  Bug 2  — _rolls_to_midi asumía 4/4. ticks_tick calculaba (tpb*4)/res en  ║
║           lugar de (tpb*ts_num)/res. Ahora lee cfg['ts_num'] (default 4). ║
║           _midi_to_rolls devuelve (rolls, ts_num) y todos los llamadores  ║
║           inyectan ts_num en el cfg antes de llamar a _rolls_to_midi.     ║
║           cmd_round_trip también lee ts_num desde _infer_time_sig_from_midi║
║           Afectaba directamente a piezas en 3/4 (vals criollo).            ║
║                                                                              ║
║  Bug 3  — cmd_cycle sin guarda HAS_DNA. Llamaba a UnifiedDNA, build_midi  ║
║           y run_cycle_gan_transform sin comprobar si el ecosistema DNA      ║
║           estaba disponible. Ahora tiene fallback que mide la pérdida de   ║
║           ciclo en feature space y reporta posiciones D_A/D_B sin MIDI.   ║
║                                                                              ║
║  Bug 4  — threshold_pct=99.0 en fallback simbólico. El default del CLI es ║
║           99.0 (pensado para backend neural con rolls continuos). El        ║
║           fallback simbólico usa rolls binarizados donde 97.0 es más        ║
║           apropiado. Comentario explícito añadido.                          ║
║                                                                              ║
║  Bug 5  — MidiRollDataset cargaba todos los .npz en RAM en __init__. Con  ║
║           corpus grandes agotaba memoria. Ahora es lazy: solo lee           ║
║           metadatos en __init__ y carga arrays en __getitem__ con          ║
║           context manager (np.load como ctx manager libera memoria).        ║
║                                                                              ║
║  Bug 6  — _nearest_neighbor_pairs era O(n²) en Python puro. Sustituido    ║
║           por sklearn.metrics.pairwise_distances cuando sklearn está         ║
║           disponible (vectorizado, 10-100× más rápido con corpus grandes). ║
║           Fallback Python puro mantenido para el caso sin sklearn.          ║
║                                                                              ║
║  Bug 7  — _extract_palette_from_midi podía asignar canal 9 a roles no-    ║
║           percusión. used_channels ahora inicializa con {9} reservado y    ║
║           el contador ch_counter salta el 9 implícitamente al no estar     ║
║           disponible. La percusión GM siempre obtiene el canal 9.          ║
║                                                                              ║
║  Bug 8  — cmd_round_trip ignoraba --resolution del usuario al cargar cfg  ║
║           desde model_dir. Ahora hace update() del cfg cargado y luego    ║
║           restaura la resolución del usuario (tiene prioridad).             ║
║                                                                              ║
║  Bug 9  — _prepare_one_midi no guardaba ts_num en los metadatos del .npz. ║
║           MidiRollDataset.__getitem__ no tenía acceso al compás real.      ║
║           Ahora ts_num se serializa en meta y está disponible para uso     ║
║           futuro en el DataLoader neural.                                   ║
║                                                                              ║
║  Bug 10 — _cmd_transform_neural procesaba barra a barra (n_bars forward   ║
║           passes). Ahora construye un batch completo (n_bars, n_roles,     ║
║           res, n_pitch) y hace un único forward pass. Para 32 compases     ║
║           con GPU esto es ~32× más rápido; en CPU también mejora por       ║
║           reducir overhead de llamadas.                                     ║
║                                                                              ║
║  Bug 11 — extract_raw_melody seleccionaba el canal con pitch_mean más alto ║
║           (sesgo hacia soprano en SATB). Nuevo criterio combinado:         ║
║           0.5×pitch_medio + 0.3×rango_pitch + 0.2×(1−densidad_relativa).  ║
║           La melodía tiende a estar en el registro agudo, tener buen        ║
║           rango interválico y no ser el stream más denso.                  ║
║                                                                              ║
║  Bug 12 — PCA silenciosamente recortaba n_components sin avisar. Con       ║
║           corpus de 5 muestras y --pca-dim 32 el recorte era a 9 sin       ║
║           ningún mensaje. Ahora emite un AVISO explícito con la causa.     ║
║                                                                              ║
║  Bug 13 — torch.load sin weights_only emite DeprecationWarning en          ║
║           PyTorch >= 2.0. Añadido weights_only=False explícito en          ║
║           _load_neural_models y CycleTrainer.load_checkpoint.              ║
║                                                                              ║
║                                                                              ║
║  CONCEPTO                                                                    ║
║    Implementa el framework CycleGAN (Zhu et al. 2017) sobre MIDI en dos    ║
║    modos seleccionables en tiempo de ejecución mediante --backend:          ║
║                                                                              ║
║    SIMBÓLICO (default, sin GPU)                                             ║
║      G, F : Ridge regression o KNN en el espacio de piano roll centroide   ║
║             (centroide temporal por rol, opcionalmente reducido con PCA).   ║
║      D_A, D_B : distancia de Mahalanobis al centroide del corpus.          ║
║      Entrenamiento iterativo alternado con pares de vecinos más cercanos    ║
║      y refinamiento con datos pseudo-ciclados.                              ║
║      Modelo serializado en un único .pkl.                                   ║
║                                                                              ║
║    NEURAL (requiere torch)                                                   ║
║      G_A2B, G_B2A : ResNet de N bloques (default 9).                       ║
║        Stem(conv7+IN+ReLU) → Down×2 → ResBlock×N → Up×2 → Head(conv7+σ)  ║
║        ResBlock: ReflectionPad + Conv + IN + ReLU + Dropout(0.05) + skip   ║
║      D_A, D_B : PatchGAN 4 capas (~70×70 campo receptivo).                 ║
║      Loss: LSGAN (MSE vs 1/0) + λ_cycle×L1 + λ_identity×L1               ║
║      Estabilización: ReplayBuffer 50 muestras para el discriminador.        ║
║      LR scheduling: decay lineal en la segunda mitad del entrenamiento.     ║
║      Early stopping por val_cycle. Checkpoint del mejor val_cycle.         ║
║      Modelo serializado en model_config.json + checkpoint.pt/best_model.pt ║
║                                                                              ║
║    DATOS (ambos backends)                                                    ║
║      MIDI → RoleAssigner → PianoRollConverter → rolls por rol              ║
║      5 roles: melody, counterpoint, accompaniment, bass, percussion         ║
║      Resolución configurable (default 96 ticks/compás).                    ║
║      Rango de pitch recortable centrado en Do central (MIDI 60).           ║
║                                                                              ║
║    GENERACIÓN (backend simbólico)                                            ║
║      El vector transformado se proyecta de vuelta a parámetros DNA via     ║
║      roll_vector_to_dna_overrides (densidad rítmica, registro de pitch,    ║
║      complejidad armónica, tensión por pitch-classes disonantes).           ║
║      La melodía original se escala, se snapa a la nueva tonalidad y se     ║
║      mezcla con melodía generada según --intensity.                         ║
║      Pipeline completo: melodía + acompañamiento + bajo + contrapunto +    ║
║      percusión + ornamentación + humanización.                              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  COMANDOS                                                                    ║
║                                                                              ║
║  prepare    MIDI corpus → archivos .npz (solo necesario para --backend      ║
║             neural). Procesa en paralelo con ThreadPoolExecutor.            ║
║                                                                              ║
║  train      Aprende G y F desde dos corpus. Acepta MIDIs directamente      ║
║             (simbólico) o carpetas de .npz (neural).                       ║
║                                                                              ║
║  transform  Aplica G (AtoB) o F (BtoA) a un MIDI. El simbólico genera     ║
║             un MIDI completo vía DNA; el neural devuelve un roll directo.  ║
║                                                                              ║
║  cycle      Aplica G y luego F al mismo MIDI, genera ambos MIDIs           ║
║             intermedios y reporta ‖F(G(a)) − a‖₂ en feature space.        ║
║                                                                              ║
║  analyze    Posición de un MIDI en el espacio A/B: distancias D_A/D_B,    ║
║             probabilidades de pertenencia, pérdida de ciclo puntual y      ║
║             los primeros 10 componentes del vector transformado.            ║
║                                                                              ║
║  style-corpus  Aplica el generador a todo un corpus y reporta densidad     ║
║             media de activación. Detecta colapso de modo (>0.3) o falta   ║
║             de convergencia (<0.005). Guarda JSON con estadísticas.        ║
║                                                                              ║
║  round-trip MIDI → piano roll → MIDI sin modelo. Diagnóstico del parser:   ║
║             verifica que la extracción y conversión son correctas antes de  ║
║             entrenar. Acepta config de model_dir si existe.                 ║
║                                                                              ║
║  inspect    Inspecciona archivos .npz (shapes, densidad por rol) y/o       ║
║             el historial de pérdida de un entrenamiento neural             ║
║             (history.json) y su model_config.json.                         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  REFERENCIA DE OPCIONES                                                      ║
║                                                                              ║
║  prepare                                                                     ║
║    --input-dir  DIR     Carpeta con MIDIs fuente            [requerido]     ║
║    --output-dir DIR     Carpeta destino de .npz             [requerido]     ║
║    --resolution N       Ticks por compás              [default: 96]         ║
║    --window-bars N      Compases por ventana           [default: 4]         ║
║    --disable-roles ROL+ Excluir roles del roll                              ║
║    --pitch-range N      Notas centradas en MIDI 60 (e.g. 48, 64, 128)     ║
║                                                                              ║
║  train  (simbólico)                                                          ║
║    --backend symbolic   [default]                                            ║
║    --domain-a DIR       Carpeta MIDIs dominio A             [requerido]     ║
║    --domain-b DIR       Carpeta MIDIs dominio B             [requerido]     ║
║    --model FILE         Salida .pkl                  [default: cycle_gan_v3.pkl]║
║    --resolution N       Ticks/compás                  [default: 96]         ║
║    --pitch-range N      Notas en el roll              [default: 128]        ║
║    --disable-roles ROL+ Excluir roles                                       ║
║    --pca-dim N          Reducción PCA (0=sin PCA)     [default: 0]         ║
║    --solver STR         ridge | knn                   [default: ridge]      ║
║    --alpha F            Regularización Ridge          [default: 1.0]        ║
║    --k-neighbors N      Vecinos para KNN              [default: 5]          ║
║    --lambda-cycle F     Peso pérdida de ciclo         [default: 10.0]       ║
║    --lambda-identity F  Peso pérdida de identidad     [default: 5.0]        ║
║    --iters N            Iteraciones alternadas        [default: 100]        ║
║    --verbose            Mostrar pérdidas por iteración                      ║
║                                                                              ║
║  train  (neural)                                                             ║
║    --backend neural                                                          ║
║    --domain-a DIR       Carpeta .npz dominio A              [requerido]     ║
║    --domain-b DIR       Carpeta .npz dominio B              [requerido]     ║
║    --model-dir DIR      Directorio del modelo        [default: model_neural] ║
║    --epochs N           Épocas de entrenamiento       [default: 200]        ║
║    --batch-size N       Batch size                    [default: 4]          ║
║    --lr F               Learning rate Adam            [default: 2e-4]       ║
║    --n-res-blocks N     Bloques ResNet en G           [default: 9]          ║
║    --base-ch N          Canales base G y D            [default: 64]         ║
║    --lambda-cycle F     Peso L1 ciclo                 [default: 10.0]       ║
║    --lambda-identity F  Peso L1 identidad             [default: 5.0]        ║
║    --patience N         Early stopping                [default: 40]         ║
║    --resume             Reanudar desde checkpoint                           ║
║                                                                              ║
║  transform                                                                   ║
║    input                MIDI de entrada                     [requerido]     ║
║    --backend STR        symbolic | neural              [default: symbolic]  ║
║    --model FILE         Modelo simbólico .pkl   (symbolic)                  ║
║    --model-dir DIR      Directorio modelo        (neural)                   ║
║    --direction STR      AtoB | BtoA                   [default: AtoB]       ║
║    --intensity F        0.0=sin cambio, 1.0=total     [default: 1.0]       ║
║    --output FILE        MIDI de salida          [default: <input>_cganv3_*] ║
║    --bars N             Compases de salida      [default: auto desde input] ║
║    --candidates N       Candidatos a evaluar (simbólico) [default: 3]      ║
║    --no-percussion      No generar percusión (simbólico)                    ║
║    --export-fingerprint Exportar fingerprint DNA (simbólico)                ║
║    --seed N             Semilla aleatoria              [default: 42]        ║
║    --bpm F              BPM del MIDI de salida (neural) [default: 120.0]   ║
║    --threshold F        Umbral de binarización fijo (neural)                ║
║    --threshold-pct F    Percentil para umbral adaptativo [default: 99.0]   ║
║    --palette FILE       JSON con paleta de instrumentos (neural)            ║
║    --verbose            Detalle del proceso                                 ║
║                                                                              ║
║  cycle                                                                       ║
║    input                MIDI de entrada                     [requerido]     ║
║    --model FILE         Modelo simbólico .pkl               [requerido]     ║
║    --direction STR      Primera dirección AtoB | BtoA  [default: AtoB]     ║
║    --intensity F        Intensidad de la transformación [default: 1.0]     ║
║    --output-ab FILE     Salida tras G              [default: <input>_ab.mid]║
║    --output-aba FILE    Salida tras F∘G            [default: <input>_aba.mid]║
║    --bars N             Compases de salida                                  ║
║    --candidates N       Candidatos a evaluar          [default: 2]          ║
║    --no-percussion      Sin percusión                                       ║
║    --seed N             Semilla aleatoria              [default: 42]        ║
║    --verbose                                                                 ║
║                                                                              ║
║  analyze                                                                     ║
║    input                MIDI a analizar                     [requerido]     ║
║    --model FILE         Modelo simbólico .pkl               [requerido]     ║
║    --verbose                                                                 ║
║                                                                              ║
║  style-corpus                                                                ║
║    --input-dir DIR      Carpeta con MIDIs a analizar        [requerido]     ║
║    --backend STR        symbolic | neural              [default: symbolic]  ║
║    --model FILE         Modelo simbólico .pkl   (symbolic)                  ║
║    --model-dir DIR      Directorio modelo        (neural)                   ║
║    --direction STR      a2b | b2a                      [default: a2b]       ║
║    --output FILE        JSON de estadísticas    [default: corpus_stats_*.json]║
║                                                                              ║
║  round-trip                                                                  ║
║    --input FILE         MIDI de entrada                     [requerido]     ║
║    --model-dir DIR      Config opcional desde model_dir                     ║
║    --resolution N       Ticks/compás                  [default: 48]         ║
║    --disable-roles ROL+ Excluir roles                                       ║
║    --output FILE        MIDI de salida           [default: roundtrip.mid]   ║
║    --bpm F              BPM de salida             [default: 120.0]          ║
║                                                                              ║
║  inspect                                                                     ║
║    --what npz|loss_curve [+]  Qué inspeccionar    [default: npz]            ║
║    --data-dir DIR       Carpeta con .npz          (para --what npz)         ║
║    --model-dir DIR      Directorio del modelo     (para --what loss_curve)  ║
║    --file FILE          .npz específico a inspeccionar                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FLUJOS DE USO COMPLETOS                                                     ║
║                                                                              ║
║  ── Backend simbólico (CPU, rápido) ──────────────────────────────────────  ║
║                                                                              ║
║  # Entrenamiento mínimo (corpus de MIDIs directamente)                      ║
║  python cycle_gan_style_transfer_v3.py train                                ║
║      --domain-a corpus/tango/  --domain-b corpus/epico/                    ║
║      --model tango2epico.pkl                                                ║
║                                                                              ║
║  # Con reducción de dimensión y solo melodía+bajo                           ║
║  python cycle_gan_style_transfer_v3.py train                                ║
║      --domain-a corpus/tango/  --domain-b corpus/epico/                    ║
║      --model tango2epico.pkl  --resolution 16  --pitch-range 48            ║
║      --pca-dim 64  --disable-roles counterpoint accompaniment percussion   ║
║      --lambda-cycle 10  --lambda-identity 0.5  --iters 200  --verbose      ║
║                                                                              ║
║  # Transferir con intensidad parcial                                         ║
║  python cycle_gan_style_transfer_v3.py transform entrada.mid                ║
║      --model tango2epico.pkl  --direction AtoB  --intensity 0.7            ║
║      --output resultado.mid  --candidates 5  --seed 123                    ║
║                                                                              ║
║  # Dirección inversa (épico → tango)                                        ║
║  python cycle_gan_style_transfer_v3.py transform entrada.mid                ║
║      --model tango2epico.pkl  --direction BtoA  --output inverso.mid       ║
║                                                                              ║
║  # Verificar consistencia de ciclo                                           ║
║  python cycle_gan_style_transfer_v3.py cycle entrada.mid                    ║
║      --model tango2epico.pkl  --direction AtoB                              ║
║      --output-ab ab.mid  --output-aba aba.mid  --verbose                   ║
║                                                                              ║
║  # Analizar posición de un MIDI en el espacio A/B                           ║
║  python cycle_gan_style_transfer_v3.py analyze entrada.mid                  ║
║      --model tango2epico.pkl                                                ║
║                                                                              ║
║  # Estadísticas del generador sobre un corpus                               ║
║  python cycle_gan_style_transfer_v3.py style-corpus                         ║
║      --input-dir corpus/test/  --model tango2epico.pkl  --direction a2b    ║
║      --output stats.json                                                    ║
║                                                                              ║
║  ── Backend neural (CPU lento / GPU recomendada) ─────────────────────────  ║
║                                                                              ║
║  # 1. Preparar corpus (una vez)                                             ║
║  python cycle_gan_style_transfer_v3.py prepare                              ║
║      --input-dir corpus/tango/  --output-dir data/tango/                   ║
║      --resolution 48  --pitch-range 64  --window-bars 4                    ║
║  python cycle_gan_style_transfer_v3.py prepare                              ║
║      --input-dir corpus/epico/  --output-dir data/epico/                   ║
║      --resolution 48  --pitch-range 64  --window-bars 4                    ║
║                                                                              ║
║  # 2. Diagnosticar el parser antes de entrenar                              ║
║  python cycle_gan_style_transfer_v3.py round-trip                           ║
║      --input corpus/tango/pieza.mid  --output prueba_rt.mid  --bpm 120     ║
║                                                                              ║
║  # 3. Inspeccionar .npz generados                                           ║
║  python cycle_gan_style_transfer_v3.py inspect                              ║
║      --what npz  --data-dir data/tango/  --file pieza.npz                  ║
║                                                                              ║
║  # 4. Entrenar neural (n-res-blocks 6 = más rápido en CPU)                 ║
║  python cycle_gan_style_transfer_v3.py train  --backend neural              ║
║      --domain-a data/tango/  --domain-b data/epico/                        ║
║      --model-dir model_t2e/  --epochs 200  --batch-size 4                  ║
║      --n-res-blocks 6  --base-ch 32  --lr 2e-4  --patience 40              ║
║      --lambda-cycle 10  --lambda-identity 0.5                               ║
║                                                                              ║
║  # 4b. Reanudar entrenamiento interrumpido                                  ║
║  python cycle_gan_style_transfer_v3.py train  --backend neural              ║
║      --domain-a data/tango/  --domain-b data/epico/                        ║
║      --model-dir model_t2e/  --epochs 200  --resume                        ║
║                                                                              ║
║  # 5. Inspeccionar curva de pérdida                                         ║
║  python cycle_gan_style_transfer_v3.py inspect                              ║
║      --what loss_curve  --model-dir model_t2e/                              ║
║                                                                              ║
║  # 6. Transferir (neural)                                                   ║
║  python cycle_gan_style_transfer_v3.py transform entrada.mid                ║
║      --backend neural  --model-dir model_t2e/  --direction AtoB            ║
║      --output resultado_neural.mid  --bpm 120  --threshold-pct 97          ║
║                                                                              ║
║  # 7. Verificar densidad de salida sobre corpus                             ║
║  python cycle_gan_style_transfer_v3.py style-corpus                         ║
║      --backend neural  --model-dir model_t2e/                               ║
║      --input-dir corpus/test/  --direction a2b  --output stats.json        ║
║                                                                              ║
║  ── Diagnóstico de MIDI vacío (neural) ───────────────────────────────────  ║
║                                                                              ║
║  Si transform neural devuelve un MIDI vacío, el diag del compás 0 muestra  ║
║  mean/max/umbral/notas_activas. Soluciones en orden de preferencia:        ║
║    1. Bajar --threshold-pct (99 → 97 → 95)                                 ║
║    2. Fijar --threshold 0.3  (umbral absoluto)                              ║
║    3. Entrenar más épocas o con más datos                                   ║
║    4. Ejecutar round-trip para verificar que el corpus no está vacío        ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CLASES PRINCIPALES                                                          ║
║    RoleAssigner         Asigna roles (melody/cp/acc/bass/perc) por perfil   ║
║    PianoRollConverter   MIDI notes → (n_bars, resolution, 128)              ║
║    MidiRollDataset      Dataset de ventanas .npz para DataLoader neural     ║
║    SymbolicDiscriminator Distancia Mahalanobis al centroide del corpus      ║
║    FeatureMapper        G o F simbólico: Ridge/KNN en feature space        ║
║    CycleGANSymbolic     Modelo simbólico completo (G,F,D_A,D_B,PCA)       ║
║    ReplayBuffer         Buffer de 50 muestras para estabilizar D neural    ║
║    CycleTrainer         Bucle entrenamiento neural completo con callbacks   ║
║                                                                              ║
║  ARCHIVOS DEL MODELO NEURAL                                                  ║
║    model_config.json    Hiperparámetros y config de datos                   ║
║    checkpoint.pt        Último checkpoint (para --resume)                   ║
║    best_model.pt        Mejor checkpoint según val_cycle                    ║
║    history.json         Historial de pérdidas por época                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DEPENDENCIAS                                                                ║
║    Obligatorias : mido, numpy, midi_dna_unified.py (mismo directorio)      ║
║    Simbólico    : scikit-learn  (sklearn.linear_model, .decomposition)     ║
║    Neural       : torch  (torch.nn, torch.nn.functional, torch.nn.utils)   ║
║    Opcional DNA : music21  (para _snap_to_scale, key handling)             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import glob
import json
import pickle
import argparse
import random
import copy
import math
import textwrap
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ── Ecosistema DNA ────────────────────────────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import midi_dna_unified as dna_mod
    from midi_dna_unified import (
        UnifiedDNA, EmotionalController, FormGenerator,
        MarkovMelody, GrooveMap,
        generate_accompaniment, generate_bass, generate_counterpoint,
        generate_percussion, add_ornamentation, humanize,
        humanize_with_swing, build_midi, score_candidate,
        _snap_to_scale, _get_scale_midi, _get_scale_pcs,
        _quarter_to_ticks, INSTRUMENT_RANGES,
    )
    from music21 import pitch as m21pitch, key as m21key
    HAS_DNA = True
except ImportError as e:
    HAS_DNA = False
    print(f"AVISO: midi_dna_unified no disponible ({e}). "
          "Backend simbólico y generación DNA desactivados.")

try:
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("AVISO: scikit-learn no encontrado. Backend simbólico limitado.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils as nn_utils
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import mido


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

ROLES = ['melody', 'counterpoint', 'accompaniment', 'bass', 'percussion']

GM_ROLE_HINTS = {
    43: 'bass', 42: 'bass', 58: 'bass', 70: 'bass',
    73: 'melody', 72: 'melody', 56: 'melody', 40: 'melody',
    68: 'counterpoint', 71: 'counterpoint', 41: 'counterpoint',
    48: 'accompaniment', 49: 'accompaniment',
    19: 'accompaniment', 52: 'accompaniment',
    88: 'accompaniment', 89: 'accompaniment',
}

PITCH_CLASSES       = 128
MIDI_CENTER         = 60
TICKS_PER_BAR       = 96   # FIX: 48 era demasiado bajo; notas < 1 beat generaban rolls vacíos
WINDOW_BARS_DEFAULT = 4

DEFAULT_PALETTE = {
    'melody':        {'program': 0,  'channel': 0, 'velocity': 80},
    'counterpoint':  {'program': 40, 'channel': 1, 'velocity': 70},
    'accompaniment': {'program': 48, 'channel': 2, 'velocity': 65},
    'bass':          {'program': 43, 'channel': 3, 'velocity': 75},
    'percussion':    {'program': 0,  'channel': 9, 'velocity': 90},
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE PITCH Y TIEMPO
# ══════════════════════════════════════════════════════════════════════════════

def _pitch_range(n):
    if n is None:
        return None
    half = n // 2
    lo   = max(0,   MIDI_CENTER - half)
    hi   = min(127, lo + n - 1)
    lo   = max(0, hi - n + 1)
    return (lo, hi)

def _crop_pitch(roll, pitch_lo, pitch_hi):
    return roll[..., pitch_lo: pitch_hi + 1]

def _pad_pitch(roll, pitch_lo, n_full=128):
    n_crop     = roll.shape[-1]
    suffix     = n_full - pitch_lo - n_crop
    pad_widths = [(0, 0)] * (roll.ndim - 1) + [(pitch_lo, suffix)]
    return np.pad(roll, pad_widths, mode='constant')

def _fmt_time(seconds):
    s = int(seconds)
    if s >= 3600: return f"{s//3600}h {(s%3600)//60}m {s%60:02d}s"
    if s >= 60:   return f"{s//60}m {s%60:02d}s"
    return f"{s}s"


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE NOTAS POR STREAM
# ══════════════════════════════════════════════════════════════════════════════

def _extract_note_lists(mid):
    """→ {(track_idx, channel): [(start_tick, end_tick, note, vel, prog)]}"""
    active = {}
    result = {}
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        prog     = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == 'program_change':
                prog = msg.program
            if msg.type in ('note_on', 'note_off'):
                ch  = msg.channel
                key = (ti, ch, msg.note)
                on  = msg.type == 'note_on' and msg.velocity > 0
                if on:
                    active[key] = (abs_tick, msg.velocity, prog)
                else:
                    if key in active:
                        st, vel, pr = active.pop(key)
                        result.setdefault((ti, ch), []).append(
                            (st, abs_tick, msg.note, vel, pr))
    return result

# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNADOR DE ROLES
# ══════════════════════════════════════════════════════════════════════════════

class RoleAssigner:
    def assign(self, mid):
        note_lists = _extract_note_lists(mid)
        if not note_lists:
            return {}
        return self._resolve_roles(self._build_profiles(note_lists, mid))

    def _build_profiles(self, note_lists, mid):
        tpb_raw   = mid.ticks_per_beat
        total_dur = max(n[1] for notes in note_lists.values() for n in notes) if note_lists else 1
        profiles  = []
        for (ti, ch), notes in note_lists.items():
            if not notes:
                continue
            pitches     = [n[2] for n in notes]
            program     = notes[0][4]
            pitch_mean  = sum(pitches) / len(pitches)
            pitch_range = max(pitches) - min(pitches)
            density     = len(notes) / max(total_dur / tpb_raw, 1)
            polyphony   = self._mean_polyphony(notes)
            profiles.append({
                'key': (ti, ch), 'channel': ch, 'program': program,
                'pitch_mean': pitch_mean, 'pitch_range': pitch_range,
                'density': density, 'polyphony': polyphony,
            })
        return profiles

    @staticmethod
    def _mean_polyphony(notes):
        if len(notes) < 2:
            return 1.0
        events = []
        for (st, en, *_) in notes:
            events.append((st, 1)); events.append((en, -1))
        events.sort()
        current = 0; samples = []
        for _, delta in events:
            current += delta; samples.append(max(current, 0))
        return sum(samples) / len(samples) if samples else 1.0

    def _resolve_roles(self, profiles):
        if not profiles:
            return {}
        assigned   = {}
        unassigned = []
        for p in profiles:
            if p['channel'] == 9:
                if 'percussion' not in assigned:
                    assigned['percussion'] = p['key']
            else:
                unassigned.append(p)

        remaining_roles = [r for r in ROLES if r != 'percussion']
        if len(unassigned) == 1:
            p  = unassigned[0]; pm = p['pitch_mean']
            if pm >= 60:   role = 'melody'
            elif pm >= 52: role = 'counterpoint'
            elif pm >= 44: role = 'accompaniment'
            else:          role = 'bass'
            assigned[role] = p['key']
            return assigned
        if not unassigned:
            return assigned

        def norm(lst, key):
            vals = [p[key] for p in lst]; lo, hi = min(vals), max(vals)
            span = hi - lo or 1
            return {p['key']: (p[key] - lo) / span for p in lst}

        n_pm   = norm(unassigned, 'pitch_mean')
        n_pr   = norm(unassigned, 'pitch_range')
        n_poly = norm(unassigned, 'polyphony')
        n_dens = norm(unassigned, 'density')

        def score(p, role):
            k = p['key']
            hint = 0.25 if GM_ROLE_HINTS.get(p['program']) == role else 0.0
            if role == 'melody':
                return 0.40*n_pm[k] + 0.35*n_pr[k] + 0.15*(1-n_poly[k]) + hint
            elif role == 'counterpoint':
                return 0.30*(1-abs(n_pm[k]-0.65)) + 0.25*n_pr[k] + 0.20*(1-n_poly[k]) + hint
            elif role == 'accompaniment':
                return 0.40*n_poly[k] + 0.25*(1-abs(n_pm[k]-0.50)) + 0.15*n_dens[k] + hint
            elif role == 'bass':
                return 0.50*(1-n_pm[k]) + 0.25*(1-n_pr[k]) + hint
            return 0.0

        pairs = [(score(p, r), r, p['key']) for p in unassigned for r in remaining_roles]
        pairs.sort(key=lambda x: -x[0])
        taken_keys = set(); taken_roles = set()
        for sc, role, key in pairs:
            if role not in taken_roles and key not in taken_keys:
                assigned[role] = key; taken_roles.add(role); taken_keys.add(key)
        return assigned


# ══════════════════════════════════════════════════════════════════════════════
#  PIANO ROLL CONVERTER
# ══════════════════════════════════════════════════════════════════════════════

class PianoRollConverter:
    def __init__(self, resolution=TICKS_PER_BAR, window_bars=WINDOW_BARS_DEFAULT):
        self.resolution  = resolution
        self.window_bars = window_bars

    def notes_to_roll(self, notes, tpbar, n_bars):
        """[(start_tick, end_tick, note, vel, prog)] → (n_bars, resolution, 128)

        tpbar : ticks por compás (ya calculado por el llamador, p.ej. tpb * ts_num).
                NO se multiplica internamente por 4; el llamador es responsable de
                pasar el valor correcto según el compás real del MIDI.
        """
        roll               = np.zeros((n_bars, self.resolution, PITCH_CLASSES), dtype=np.float32)
        ticks_per_internal = tpbar / self.resolution          # FIX: era tpbar*4/res
        for start, end, pitch, vel, prog in notes:
            bar_s  = int(start / tpbar)                       # FIX: era start/(tpbar*4)
            bar_e  = int(end   / tpbar)                       # FIX: era end/(tpbar*4)
            tick_s = int((start % tpbar) / ticks_per_internal) # FIX: era (start%(tpbar*4))/…
            tick_e = int((end   % tpbar) / ticks_per_internal) # FIX: era (end%(tpbar*4))/…
            if bar_s >= n_bars: continue
            if bar_s == bar_e:
                roll[bar_s, tick_s:min(tick_e + 1, self.resolution), pitch] = 1.0
            else:
                roll[bar_s, tick_s:, pitch] = 1.0
                for b in range(bar_s + 1, min(bar_e, n_bars)):
                    roll[b, :, pitch] = 1.0
                if bar_e < n_bars:
                    roll[bar_e, :max(tick_e, 1), pitch] = 1.0
        return roll

    def roll_to_windows(self, roll):
        """(n_bars, res, pitch) → (n_windows, window_bars, res, pitch)"""
        n_bars = roll.shape[0]
        if n_bars < self.window_bars:
            return np.zeros((0, self.window_bars, self.resolution, PITCH_CLASSES),
                            dtype=np.float32)
        return np.stack([roll[i: i + self.window_bars]
                         for i in range(n_bars - self.window_bars + 1)], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  UMBRAL ADAPTATIVO Y CONVERSIÓN ROLLS → MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _adaptive_threshold(roll, percentile=99.0):
    """
    Binarización robusta:
    · >90% celdas cerca de 0 → percentil sobre toda la distribución.
    · Si no → percentil sobre valores activos (>0.001).
    """
    flat           = roll.flatten()
    frac_near_zero = float((flat < 0.01).mean())
    if frac_near_zero > 0.90:
        return max(float(np.percentile(flat, percentile)), 1e-4)
    nonzero = flat[flat > 0.001]
    if len(nonzero) == 0:
        return 0.5
    return float(np.percentile(nonzero, percentile))


def _rolls_to_midi(bars_per_role, cfg, palette, output_path,
                   bpm=120.0, threshold=None, threshold_pct=99.0):
    """
    Convierte {rol: (n_bars, res, n_pitch)} a un fichero MIDI.
    Respeta el compás real del MIDI fuente vía cfg['ts_num'] (default 4).
    """
    resolution = cfg['resolution']
    ts_num     = cfg.get('ts_num', 4)          # FIX: usar compás real, no asumir 4/4
    tpb        = 480
    ticks_tick = (tpb * ts_num) / resolution   # FIX: era (tpb*4)/resolution
    pitch_lo   = cfg.get('pitch_lo', 0)
    pitch_hi   = cfg.get('pitch_hi', 127)
    do_expand  = (pitch_lo, pitch_hi) != (0, 127)

    mid = mido.MidiFile(ticks_per_beat=tpb)
    t0  = mido.MidiTrack()
    t0.append(mido.MetaMessage('set_tempo', tempo=int(60_000_000 / bpm), time=0))
    mid.tracks.append(t0)

    n_notes_total = 0

    for role in cfg['roles']:
        if role not in bars_per_role:
            continue
        roll = bars_per_role[role]
        if do_expand:
            roll = _pad_pitch(roll, pitch_lo, n_full=128)

        thr    = threshold if threshold is not None else _adaptive_threshold(roll, threshold_pct)
        pal    = palette.get(role, {})
        prog   = int(pal.get('program', 0))
        ch     = int(pal.get('channel', 0))
        vel    = int(pal.get('velocity', 80))
        binary = (roll > thr).astype(np.float32)

        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message('program_change', program=prog, channel=ch, time=0))

        events          = []
        n_bars_r, res_r, _ = binary.shape
        for bar in range(n_bars_r):
            for tick in range(res_r):
                abs_t = int((bar * res_r + tick) * ticks_tick)
                for pitch in range(128):
                    cur  = binary[bar, tick, pitch] > 0
                    prev = (binary[bar, tick-1, pitch] > 0 if tick > 0
                            else (binary[bar-1, -1, pitch] > 0 if bar > 0 else False))
                    if cur and not prev:
                        events.append((abs_t, 'on',  pitch)); n_notes_total += 1
                    elif not cur and prev:
                        events.append((abs_t, 'off', pitch))

        events.sort(key=lambda e: e[0])
        prev_t = 0
        for abs_t, etype, pitch in events:
            delta = abs_t - prev_t
            if etype == 'on':
                track.append(mido.Message('note_on',  note=pitch, velocity=vel,
                                          channel=ch, time=delta))
            else:
                track.append(mido.Message('note_off', note=pitch, velocity=0,
                                          channel=ch, time=delta))
            prev_t = abs_t

    mid.save(output_path)
    return n_notes_total


def _extract_palette_from_midi(midi_path, role_map):
    """
    Extrae la paleta real (program, channel, velocity_mean) del MIDI fuente
    para cada rol asignado por RoleAssigner. Devuelve dict compatible con
    _rolls_to_midi.
    """
    mid        = mido.MidiFile(midi_path)
    note_lists = _extract_note_lists(mid)

    # Mapa (track_idx, channel) → program
    prog_map = {}
    for ti, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'program_change':
                prog_map[(ti, msg.channel)] = msg.program

    # Canales ya usados para evitar colisiones en la salida
    # FIX: ch_counter arranca en 0 y siempre salta el canal 9 (percusión GM)
    used_channels = {9}   # reservar el 9 de entrada; se libera solo para percusión real
    palette       = {}
    ch_counter    = 0

    for role, stream_key in role_map.items():
        ti, ch_orig = stream_key
        prog  = prog_map.get(stream_key, DEFAULT_PALETTE.get(role, {}).get('program', 0))
        # Velocidad media de las notas de este stream
        notes = note_lists.get(stream_key, [])
        vel   = int(sum(n[3] for n in notes) / len(notes)) if notes else 80
        vel   = max(40, min(110, vel))

        if ch_orig == 9:
            # Percusión: siempre canal 9
            out_ch = 9
            used_channels.add(9)
        elif ch_orig not in used_channels:
            out_ch = ch_orig
            used_channels.add(out_ch)
        else:
            # Canal original ya ocupado: asignar el siguiente libre (nunca el 9)
            while ch_counter in used_channels:
                ch_counter += 1
            out_ch = ch_counter
            used_channels.add(out_ch)
            ch_counter += 1

        palette[role] = {'program': prog, 'channel': out_ch, 'velocity': vel}

    return palette


def _infer_bpm_from_midi(midi_path):
    """Lee el primer set_tempo del MIDI y devuelve BPM."""
    try:
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return round(60_000_000 / msg.tempo, 2)
    except Exception:
        pass
    return 120.0


def _infer_time_sig_from_midi(midi_path):
    """Lee el primer time_signature y devuelve beats_per_bar."""
    try:
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'time_signature':
                    return msg.numerator, msg.denominator
    except Exception:
        pass
    return 4, 4


def _midi_to_rolls(midi_path, cfg):
    """MIDI → {rol: (n_bars, resolution, n_pitch)}.
    Usa el compás real del MIDI (time_signature) en lugar de asumir 4/4."""
    mid         = mido.MidiFile(midi_path)
    resolution  = cfg['resolution']
    window_bars = cfg.get('window_bars', WINDOW_BARS_DEFAULT)
    note_lists  = _extract_note_lists(mid)
    if not note_lists:
        raise ValueError(f"No se encontraron notas en {midi_path}")

    tpb = mid.ticks_per_beat

    # Detectar compás real para calcular ticks por compás
    ts_num = 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                ts_num = msg.numerator
                break

    tpbar      = tpb * ts_num          # ticks por compás según el compás real
    max_tick   = max((n[1] for nl in note_lists.values() for n in nl), default=0)
    total_bars = max(1, int(max_tick / tpbar) + 1)

    active_roles = cfg.get('roles', ROLES)
    pitch_lo     = cfg.get('pitch_lo', 0)
    pitch_hi     = cfg.get('pitch_hi', 127)
    do_crop      = (pitch_lo, pitch_hi) != (0, 127)

    role_map = RoleAssigner().assign(mid)
    conv     = PianoRollConverter(resolution=resolution, window_bars=window_bars)
    rolls    = {}
    for role, stream_key in role_map.items():
        if role not in active_roles:
            continue
        roll = conv.notes_to_roll(note_lists[stream_key], tpbar, total_bars)
        if do_crop:
            roll = _crop_pitch(roll, pitch_lo, pitch_hi)
        rolls[role] = roll
    # ts_num se pasa al llamador via cfg modificado para que _rolls_to_midi
    # calcule ticks_tick correctamente. El llamador debe añadirlo al cfg.
    return rolls, ts_num


def _load_palette(palette_path, cfg):
    if palette_path is None:
        return DEFAULT_PALETTE
    try:
        with open(palette_path) as f:
            return json.load(f)
    except Exception:
        return DEFAULT_PALETTE


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PARA BACKEND NEURAL
# ══════════════════════════════════════════════════════════════════════════════

class MidiRollDataset:
    """Dataset de ventanas de piano roll leídas desde .npz.

    Lazy loading: los arrays se cargan solo cuando se accede a __getitem__,
    no en __init__. Esto evita agotar RAM con corpus grandes.
    """

    def __init__(self, data_dir, roles=None):
        self.roles   = roles or ROLES
        self.samples = []   # (path, window_idx, meta)
        self.n_pitch = None

        for path in sorted(Path(data_dir).glob('*.npz')):
            try:
                # Solo leer metadatos en __init__, no los arrays
                with np.load(str(path), allow_pickle=True) as data:
                    meta = json.loads(str(data['meta_json'][0]))
                if self.n_pitch is None:
                    self.n_pitch = meta.get('n_pitch', PITCH_CLASSES)
                for i in range(meta['n_windows']):
                    self.samples.append((str(path), i, meta))
            except Exception:
                continue

        if self.n_pitch is None:
            self.n_pitch = PITCH_CLASSES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, widx, meta = self.samples[idx]
        resolution = meta['resolution']
        # ts_num guardado por _prepare_one_midi; disponible para uso futuro
        x_parts    = []
        # FIX: lazy load — abrir el npz solo al acceder al item
        with np.load(path, allow_pickle=True) as data:
            for role in self.roles:
                key = f'roll_{role}'
                if key in data:
                    x_parts.append(data[key][widx][-1].copy())
                else:
                    x_parts.append(np.zeros((resolution, self.n_pitch), dtype=np.float32))
        if HAS_TORCH:
            return torch.tensor(np.stack(x_parts, axis=0))
        return np.stack(x_parts, axis=0)


def _collate_fn(batch):
    if HAS_TORCH:
        return torch.stack(batch)
    return np.stack(batch)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE MELODÍA (para motor DNA)
# ══════════════════════════════════════════════════════════════════════════════

def extract_raw_melody(midi_path, verbose=False):
    """→ ([(offset_beats, pitch, dur_beats, vel)], tempo_bpm, (ts_num, ts_den))"""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb      = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4
    notes_by_channel = {}
    pending          = {}

    for track in mid.tracks:
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            abs_beats  = abs_ticks / tpb
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_beats, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key_ = (msg.channel, msg.note)
                if key_ in pending:
                    onset, vel = pending.pop(key_)
                    dur = max(0.1, abs_beats - onset)
                    if msg.channel != 9:
                        notes_by_channel.setdefault(msg.channel, []).append(
                            (onset, msg.note, dur, vel))

    tempo_bpm = 60_000_000 / max(tempo_us, 1)
    if not notes_by_channel:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    # FIX: criterio combinado en lugar de solo pitch_mean.
    # Pondera: pitch_mean (0.5) + pitch_range (0.3) + densidad_inversa (0.2).
    # La melodía tiende a estar en el registro agudo, tener buen rango y
    # no ser el canal más denso (ese suele ser el acompañamiento).
    all_notes_flat = [(p, d) for notes in notes_by_channel.values()
                      for _, p, d, _ in notes]
    if all_notes_flat:
        global_max_pitch = max(p for p, _ in all_notes_flat)
        global_density   = sum(1 for _ in all_notes_flat)
    else:
        global_max_pitch = 127
        global_density   = 1

    def _melody_score(ch):
        notes = notes_by_channel[ch]
        if not notes: return 0.0
        pitches = [p for _, p, _, _ in notes]
        pm    = sum(pitches) / len(pitches) / 127.0          # pitch medio norm.
        pr    = (max(pitches) - min(pitches)) / 127.0        # rango norm.
        dens  = len(notes) / max(global_density, 1)          # densidad relativa
        return 0.5 * pm + 0.3 * pr + 0.2 * (1.0 - dens)     # mayor pitch, más rango, menos denso

    melody_ch = max(notes_by_channel, key=_melody_score)
    melody = sorted(notes_by_channel[melody_ch], key=lambda x: x[0])
    if verbose:
        print(f"    Melodía: {len(melody)} notas | ch={melody_ch} | "
              f"{tempo_bpm:.1f} BPM | {ts_num}/{ts_den}")
    return melody, tempo_bpm, (ts_num, ts_den)


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZACIÓN MIDI → PIANO ROLL CENTROIDE (backend simbólico)
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_roll_vector(midi_path, resolution=TICKS_PER_BAR,
                        active_roles=None, pitch_lo=None, pitch_hi=None,
                        verbose=False):
    """
    Centroide temporal por rol concatenado y aplanado.
    Shape: (n_roles × resolution × n_pitch,)
    """
    if active_roles is None:
        active_roles = ROLES
    n_pitch = (pitch_hi - pitch_lo + 1) if pitch_lo is not None else PITCH_CLASSES

    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        raise RuntimeError(f"No se encontraron notas en {midi_path}")

    tpb_raw   = mid.ticks_per_beat
    # FIX: leer el compás real en lugar de asumir siempre 4/4
    ts_num = 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                ts_num = msg.numerator
                break
    tpbar     = tpb_raw * ts_num
    all_ticks = max((n[1] for nl in note_lists.values() for n in nl), default=0)
    n_bars    = max(1, int(all_ticks / tpbar) + 1)

    role_map = RoleAssigner().assign(mid)
    conv     = PianoRollConverter(resolution=resolution)

    centroid_parts = []
    active_mask    = []   # FIX: registrar qué roles tienen datos reales
    for role in active_roles:
        stream_key = role_map.get(role)
        if stream_key and stream_key in note_lists:
            roll     = conv.notes_to_roll(note_lists[stream_key], tpbar, n_bars)
            if pitch_lo is not None:
                roll = _crop_pitch(roll, pitch_lo, pitch_hi)
            centroid = roll.mean(axis=0)
            has_data = True
        else:
            centroid = np.zeros((resolution, n_pitch), dtype=np.float32)
            has_data = False
        centroid_parts.append(centroid)
        active_mask.append(has_data)

    vec = np.concatenate([c.flatten() for c in centroid_parts]).astype(np.float32)
    if verbose:
        n_active = sum(active_mask)
        print(f"    Roll: {n_bars} compases | {n_active}/{len(active_roles)} roles | "
              f"vector {len(vec)}D")
    return vec


def roll_vector_dim(n_roles, resolution, n_pitch):
    return n_roles * resolution * n_pitch


# ══════════════════════════════════════════════════════════════════════════════
#  DISCRIMINADORES SIMBÓLICOS (Mahalanobis)
# ══════════════════════════════════════════════════════════════════════════════

class SymbolicDiscriminator:
    def __init__(self, name):
        self.name = name; self.centroid = None
        self.inv_cov = None; self.scaler = None

    def fit(self, features):
        if HAS_SKLEARN:
            self.scaler = StandardScaler()
            fs = self.scaler.fit_transform(features)
        else:
            fs = features
        self.centroid = np.mean(fs, axis=0)
        try:
            self.inv_cov = np.linalg.pinv(np.cov(fs.T))
        except Exception:
            self.inv_cov = None

    def score(self, feat):
        if self.centroid is None: return 1.0
        if HAS_SKLEARN and self.scaler:
            f = self.scaler.transform(feat.reshape(1, -1)).flatten()
        else:
            f = feat
        diff = f - self.centroid
        if self.inv_cov is not None:
            return float(np.sqrt(np.maximum(diff @ self.inv_cov @ diff, 0.0)))
        return float(np.linalg.norm(diff))

    def domain_probability(self, feat):
        return float(np.exp(-self.score(feat) * 0.5))


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES SIMBÓLICOS G y F  (Ridge / KNN)
# ══════════════════════════════════════════════════════════════════════════════

class FeatureMapper:
    def __init__(self, name, solver='ridge', alpha=1.0, k=5):
        self.name = name; self.solver = solver
        self.alpha = alpha; self.k = k
        self.model = None; self.scaler_in = None; self.scaler_out = None

    def fit(self, X, Y):
        if not HAS_SKLEARN:
            self._mean_dst = np.mean(Y, axis=0)
            self._mean_src = np.mean(X, axis=0)
            return
        self.scaler_in  = StandardScaler().fit(X)
        self.scaler_out = StandardScaler().fit(Y)
        Xs = self.scaler_in.transform(X)
        Ys = self.scaler_out.transform(Y)
        self.model = (KNeighborsRegressor(n_neighbors=min(self.k, len(X)))
                      if self.solver == 'knn' else Ridge(alpha=self.alpha))
        self.model.fit(Xs, Ys)

    def transform(self, feat):
        if not HAS_SKLEARN or self.model is None:
            if hasattr(self, '_mean_dst'):
                return np.clip(feat + (self._mean_dst - self._mean_src), 0.0, 1.0)
            return feat.copy()
        f_in  = self.scaler_in.transform(feat.reshape(1, -1))
        f_out = self.model.predict(f_in)[0]
        return self.scaler_out.inverse_transform(f_out.reshape(1, -1))[0]

    def cycle_loss(self, feats_a, mapper_back):
        return float(np.mean([np.linalg.norm(mapper_back.transform(self.transform(f)) - f)
                               for f in feats_a]))

    def identity_loss(self, feats_dst):
        return float(np.mean([np.linalg.norm(self.transform(f) - f) for f in feats_dst]))


def _nearest_neighbor_pairs(src, dst):
    """
    Para cada vector en src, devuelve el índice del más cercano en dst.
    Usa sklearn.metrics.pairwise_distances cuando está disponible (O(n·m) vectorizado),
    con fallback a Python puro para el caso sin sklearn.
    """
    if HAS_SKLEARN:
        from sklearn.metrics import pairwise_distances
        dmat = pairwise_distances(src, dst, metric='euclidean')
        return dmat.argmin(axis=1).tolist()
    # Fallback Python puro
    return [int(np.argmin([np.linalg.norm(f - g) for g in dst])) for f in src]


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO SIMBÓLICO  (G, F, D_A, D_B + PCA)
# ══════════════════════════════════════════════════════════════════════════════

class CycleGANSymbolic:
    """Backend simbólico: piano roll centroide + PCA + Ridge/KNN."""

    VERSION = "3.0-symbolic"

    def __init__(self, solver='ridge', alpha=1.0, k_neighbors=5,
                 lambda_cycle=10.0, lambda_identity=5.0,
                 resolution=TICKS_PER_BAR, active_roles=None,
                 pitch_lo=None, pitch_hi=None, pca_dim=0):

        self.solver          = solver
        self.alpha           = alpha
        self.k_neighbors     = k_neighbors
        self.lambda_cycle    = lambda_cycle
        self.lambda_identity = lambda_identity
        self.resolution      = resolution
        self.active_roles    = active_roles or ROLES
        self.pitch_lo        = pitch_lo
        self.pitch_hi        = pitch_hi
        self.pca_dim         = pca_dim

        n_pitch      = (pitch_hi - pitch_lo + 1) if pitch_lo is not None else PITCH_CLASSES
        self.raw_dim = roll_vector_dim(len(self.active_roles), resolution, n_pitch)

        self.G   = FeatureMapper("G_AtoB", solver, alpha, k_neighbors)
        self.F   = FeatureMapper("F_BtoA", solver, alpha, k_neighbors)
        self.D_A = SymbolicDiscriminator("D_A")
        self.D_B = SymbolicDiscriminator("D_B")
        self.pca = None

        self.losses_history = []
        self.domain_a_name  = "A"
        self.domain_b_name  = "B"

    # ── Vectorización ─────────────────────────────────────────────────────

    def midi_to_vec(self, midi_path, verbose=False):
        vec = midi_to_roll_vector(midi_path, self.resolution,
                                  self.active_roles, self.pitch_lo,
                                  self.pitch_hi, verbose)
        if self.pca is not None:
            vec = self.pca.transform(vec.reshape(1, -1))[0]
        return vec

    # ── Entrenamiento ─────────────────────────────────────────────────────

    def fit(self, feats_a, feats_b, iters=100, verbose=False):
        if len(feats_a) == 0 or len(feats_b) == 0:
            raise ValueError("Se necesitan MIDIs en ambos dominios.")

        if self.pca_dim > 0 and HAS_SKLEARN:
            all_vecs     = np.vstack([feats_a, feats_b])
            n_components = min(self.pca_dim, all_vecs.shape[0], all_vecs.shape[1])
            # FIX: warning explícito si PCA se recorta por corpus pequeño
            if n_components < self.pca_dim:
                print(f"  AVISO: pca_dim={self.pca_dim} recortado a {n_components} "
                      f"(limitado por n_samples={all_vecs.shape[0]} o dim={all_vecs.shape[1]}). "
                      f"Amplía el corpus o reduce --pca-dim.")
            self.pca = PCA(n_components=n_components)
            self.pca.fit(all_vecs)
            feats_a = self.pca.transform(feats_a)
            feats_b = self.pca.transform(feats_b)
            if verbose:
                var = self.pca.explained_variance_ratio_.sum()
                print(f"  PCA: {n_components} componentes | varianza explicada: {var:.1%}")
        else:
            self.pca = None

        self.D_A.fit(feats_a)
        self.D_B.fit(feats_b)

        pairs_ab = _nearest_neighbor_pairs(feats_a, feats_b)
        pairs_ba = _nearest_neighbor_pairs(feats_b, feats_a)
        self.G.fit(feats_a, np.array([feats_b[j] for j in pairs_ab]))
        self.F.fit(feats_b, np.array([feats_a[j] for j in pairs_ba]))

        if verbose:
            print(f"  {'Iter':>4}  {'CycLoss_A':>10}  {'CycLoss_B':>10}  "
                  f"{'IdLoss_G':>9}  {'IdLoss_F':>9}  {'Total':>10}")

        for it in range(iters):
            lc_a  = self.G.cycle_loss(feats_a, self.F)
            lc_b  = self.F.cycle_loss(feats_b, self.G)
            li_g  = self.G.identity_loss(feats_b)
            li_f  = self.F.identity_loss(feats_a)
            total = self.lambda_cycle * (lc_a + lc_b) + self.lambda_identity * (li_g + li_f)

            self.losses_history.append({
                'iter': it, 'cycle_a': lc_a, 'cycle_b': lc_b,
                'identity_g': li_g, 'identity_f': li_f, 'total': total
            })
            if verbose and (it % max(1, iters // 20) == 0 or it == iters - 1):
                print(f"  {it:>4}  {lc_a:>10.4f}  {lc_b:>10.4f}  "
                      f"{li_g:>9.4f}  {li_f:>9.4f}  {total:>10.4f}")

            # Refinamiento iterativo alternado con datos pseudo-ciclados
            G_out = np.array([self.G.transform(f) for f in feats_a])
            self.G.fit(
                np.vstack([feats_a, [self.F.transform(f) for f in feats_b]]),
                np.vstack([G_out, feats_b]))

            F_out = np.array([self.F.transform(f) for f in feats_b])
            self.F.fit(
                np.vstack([feats_b, [self.G.transform(f) for f in feats_a]]),
                np.vstack([F_out, feats_a]))

        if verbose:
            print(f"\n  Pérdida final total: {total:.4f}")

    # ── Inferencia ────────────────────────────────────────────────────────

    def map_features(self, feat, direction='AtoB'):
        if direction == 'AtoB': return self.G.transform(feat)
        if direction == 'BtoA': return self.F.transform(feat)
        raise ValueError(f"direction debe ser 'AtoB' o 'BtoA'")

    def discriminate(self, feat):
        return {
            'D_A_dist': self.D_A.score(feat),           'D_B_dist': self.D_B.score(feat),
            'D_A_prob': self.D_A.domain_probability(feat), 'D_B_prob': self.D_B.domain_probability(feat),
        }

    # ── Serialización ─────────────────────────────────────────────────────

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Modelo simbólico guardado → {path}")

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, CycleGANSymbolic):
            raise TypeError("El fichero no contiene un CycleGANSymbolic.")
        return obj


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITECTURA NEURAL  (ResNet-9 + PatchGAN)
# ══════════════════════════════════════════════════════════════════════════════

def _build_generator(n_roles, resolution, n_pitch, n_res_blocks=9, base_ch=64):
    """
    Generador ResNet para piano rolls.
    Entrada/Salida: (B, N_ROLES, resolution, n_pitch)
    Stem(conv7) → Down×2 → ResBlock×N → Up×2 → Head(conv7+Sigmoid)
    """
    if not HAS_TORCH:
        raise ImportError("torch requerido para backend neural.")

    class _ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.net = nn.Sequential(
                nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3),
                nn.InstanceNorm2d(ch), nn.ReLU(inplace=True), nn.Dropout2d(0.05),
                nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch),
            )
        def forward(self, x): return x + self.net(x)

    class _Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.ReflectionPad2d(3), nn.Conv2d(n_roles, base_ch, 7),
                nn.InstanceNorm2d(base_ch), nn.ReLU(inplace=True),
            )
            self.down = nn.Sequential(
                nn.Conv2d(base_ch,     base_ch*2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(base_ch*2), nn.ReLU(inplace=True),
                nn.Conv2d(base_ch*2,   base_ch*4, 3, stride=2, padding=1),
                nn.InstanceNorm2d(base_ch*4), nn.ReLU(inplace=True),
            )
            self.res  = nn.Sequential(*[_ResBlock(base_ch*4) for _ in range(n_res_blocks)])
            self.up   = nn.Sequential(
                nn.ConvTranspose2d(base_ch*4, base_ch*2, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(base_ch*2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(base_ch*2, base_ch,   3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(base_ch), nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(
                nn.ReflectionPad2d(3), nn.Conv2d(base_ch, n_roles, 7), nn.Sigmoid(),
            )
        def forward(self, x):
            return self.head(self.up(self.res(self.down(self.stem(x)))))

    return _Generator()


def _build_discriminator(n_roles, base_ch=64):
    """
    PatchGAN ~70×70 campo receptivo.
    Entrada: (B, N_ROLES, resolution, n_pitch)  →  (B, 1, H', W')
    """
    if not HAS_TORCH:
        raise ImportError("torch requerido para backend neural.")

    def _block(ic, oc, norm=True):
        layers = [nn.Conv2d(ic, oc, 4, stride=2, padding=1)]
        if norm: layers.append(nn.InstanceNorm2d(oc))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    return nn.Sequential(
        *_block(n_roles,     base_ch,    norm=False),
        *_block(base_ch,     base_ch*2),
        *_block(base_ch*2,   base_ch*4),
        nn.Conv2d(base_ch*4, base_ch*8, 4, stride=1, padding=1),
        nn.InstanceNorm2d(base_ch*8), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(base_ch*8, 1, 4, stride=1, padding=1),
    )


class ReplayBuffer:
    """Almacena los últimos max_size ejemplos generados para estabilizar D."""
    def __init__(self, max_size=50):
        self.max_size = max_size; self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element); to_return.append(element)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = element
                    to_return.append(tmp)
                else:
                    to_return.append(element)
        return torch.cat(to_return, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENADOR NEURAL
# ══════════════════════════════════════════════════════════════════════════════

class CycleTrainer:
    CONFIG_NAME     = 'model_config.json'
    CHECKPOINT_NAME = 'checkpoint.pt'
    BEST_NAME       = 'best_model.pt'
    HISTORY_NAME    = 'history.json'

    def __init__(self, G_A2B, G_B2A, D_A, D_B,
                 opt_G, opt_D_A, opt_D_B,
                 model_dir, patience=40,
                 lambda_cycle=10.0, lambda_ident=0.5):
        self.G_A2B = G_A2B; self.G_B2A = G_B2A
        self.D_A   = D_A;   self.D_B   = D_B
        self.opt_G = opt_G; self.opt_D_A = opt_D_A; self.opt_D_B = opt_D_B
        self.model_dir  = Path(model_dir)
        self.patience   = patience
        self.λ_cycle    = lambda_cycle
        self.λ_ident    = lambda_ident
        self.history    = {'train_G': [], 'train_D': [], 'val_cycle': []}
        self.best_loss  = float('inf')
        self.no_improve = 0
        self.start_epoch = 0

    def save_checkpoint(self, epoch, val_loss, is_best):
        state = {
            'epoch': epoch,
            'G_A2B': self.G_A2B.state_dict(), 'G_B2A': self.G_B2A.state_dict(),
            'D_A':   self.D_A.state_dict(),   'D_B':   self.D_B.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D_A': self.opt_D_A.state_dict(), 'opt_D_B': self.opt_D_B.state_dict(),
            'best_loss': self.best_loss, 'no_improve': self.no_improve,
            'history': self.history,
        }
        torch.save(state, self.model_dir / self.CHECKPOINT_NAME)
        if is_best:
            torch.save(state, self.model_dir / self.BEST_NAME)
        with open(self.model_dir / self.HISTORY_NAME, 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_checkpoint(self):
        path = self.model_dir / self.CHECKPOINT_NAME
        if not path.exists():
            print("[train] Entrenando desde cero."); return
        state = torch.load(path, map_location='cpu', weights_only=False)  # FIX: explícito para torch>=2.0
        self.G_A2B.load_state_dict(state['G_A2B']); self.G_B2A.load_state_dict(state['G_B2A'])
        self.D_A.load_state_dict(state['D_A']);   self.D_B.load_state_dict(state['D_B'])
        self.opt_G.load_state_dict(state['opt_G'])
        self.opt_D_A.load_state_dict(state['opt_D_A']); self.opt_D_B.load_state_dict(state['opt_D_B'])
        self.best_loss = state['best_loss']; self.no_improve = state['no_improve']
        self.history   = state['history'];  self.start_epoch = state['epoch'] + 1
        print(f"[train] Reanudando desde época {self.start_epoch} (mejor val={self.best_loss:.4f})")

    def _gan_loss(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return F.mse_loss(pred, target)

    def _train_epoch(self, loader_A, loader_B, buf_A, buf_B, device):
        self.G_A2B.train(); self.G_B2A.train()
        self.D_A.train();   self.D_B.train()
        sum_G = sum_D = n_batches = 0
        iter_B = iter(loader_B)

        for real_A in loader_A:
            try:
                real_B = next(iter_B)
            except StopIteration:
                iter_B = iter(loader_B); real_B = next(iter_B)

            real_A = real_A.to(device); real_B = real_B.to(device)

            # ── Generadores ──────────────────────────────────────────────
            self.opt_G.zero_grad()
            fake_B = self.G_A2B(real_A); fake_A = self.G_B2A(real_B)
            loss_G = (self._gan_loss(self.D_B(fake_B), True)
                    + self._gan_loss(self.D_A(fake_A), True)
                    + self.λ_cycle * (F.l1_loss(self.G_B2A(fake_B), real_A)
                                    + F.l1_loss(self.G_A2B(fake_A), real_B))
                    + self.λ_ident * (F.l1_loss(self.G_A2B(real_B), real_B)
                                    + F.l1_loss(self.G_B2A(real_A), real_A)))
            loss_G.backward()
            nn_utils.clip_grad_norm_(
                list(self.G_A2B.parameters()) + list(self.G_B2A.parameters()), 1.0)
            self.opt_G.step()

            # ── Discriminadores ───────────────────────────────────────────
            self.opt_D_A.zero_grad()
            loss_D_A = 0.5 * (self._gan_loss(self.D_A(real_A), True)
                             + self._gan_loss(self.D_A(buf_A.push_and_pop(fake_A.detach())), False))
            loss_D_A.backward(); self.opt_D_A.step()

            self.opt_D_B.zero_grad()
            loss_D_B = 0.5 * (self._gan_loss(self.D_B(real_B), True)
                             + self._gan_loss(self.D_B(buf_B.push_and_pop(fake_B.detach())), False))
            loss_D_B.backward(); self.opt_D_B.step()

            sum_G += loss_G.item(); sum_D += (loss_D_A + loss_D_B).item(); n_batches += 1

        n_batches = max(n_batches, 1)
        return sum_G / n_batches, sum_D / n_batches

    def _val_cycle_loss(self, loader_A, loader_B, device):
        self.G_A2B.eval(); self.G_B2A.eval()
        total = n = 0
        with torch.no_grad():
            for real_A in loader_A:
                real_A = real_A.to(device)
                total += F.l1_loss(self.G_B2A(self.G_A2B(real_A)), real_A).item()
                n += 1
                if n >= 20: break
        return total / max(n, 1)

    def train(self, loader_A_tr, loader_B_tr, loader_A_val, loader_B_val,
              n_epochs, resume=False):
        import time
        import torch.optim.lr_scheduler as sched

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for m in (self.G_A2B, self.G_B2A, self.D_A, self.D_B):
            m.to(device)

        buf_A = ReplayBuffer(); buf_B = ReplayBuffer()

        def _lr_lambda(epoch):
            ds = n_epochs // 2
            return 1.0 if epoch < ds else 1.0 - (epoch - ds) / max(n_epochs - ds, 1)

        sch_G   = sched.LambdaLR(self.opt_G,   _lr_lambda)
        sch_D_A = sched.LambdaLR(self.opt_D_A, _lr_lambda)
        sch_D_B = sched.LambdaLR(self.opt_D_B, _lr_lambda)

        if resume:
            self.load_checkpoint()

        print(f"\n{'─'*64}")
        print(f"  CycleGAN neural: {n_epochs} épocas  |  dispositivo: {device}")
        print(f"  λ_cycle={self.λ_cycle}  λ_identity={self.λ_ident}")
        print(f"{'─'*64}\n")

        train_start = time.time()
        bar_w = 30

        for epoch in range(self.start_epoch, n_epochs):
            t0 = time.time()
            if epoch == self.start_epoch and resume:
                for _ in range(self.start_epoch):
                    sch_G.step(); sch_D_A.step(); sch_D_B.step()

            loss_G, loss_D = self._train_epoch(loader_A_tr, loader_B_tr, buf_A, buf_B, device)
            val_cycle      = self._val_cycle_loss(loader_A_val, loader_B_val, device)

            sch_G.step(); sch_D_A.step(); sch_D_B.step()

            self.history['train_G'].append(round(loss_G, 5))
            self.history['train_D'].append(round(loss_D, 5))
            self.history['val_cycle'].append(round(val_cycle, 5))

            is_best = val_cycle < self.best_loss
            if is_best:
                self.best_loss = val_cycle; self.no_improve = 0
            else:
                self.no_improve += 1

            self.save_checkpoint(epoch, val_cycle, is_best)

            elapsed  = time.time() - t0
            progress = int((epoch + 1) / n_epochs * bar_w)
            bar      = '█' * progress + '░' * (bar_w - progress)
            best_str = ' ◀ mejor' if is_best else ''
            stop_str = (f'  [sin mejora {self.no_improve}/{self.patience}]'
                        if self.no_improve > 0 else '')
            print(f"  Época {epoch+1:4d}/{n_epochs}  │{bar}│  "
                  f"G={loss_G:.4f}  D={loss_D:.4f}  val_cycle={val_cycle:.4f}"
                  f"  {_fmt_time(elapsed)}/ép{best_str}{stop_str}")

            if self.no_improve >= self.patience:
                print(f"\n  Early stopping tras {epoch+1} épocas.")
                break

        print(f"\n{'─'*64}")
        print(f"  Completado en {_fmt_time(time.time() - train_start)}.")
        print(f"  Mejor val_cycle : {self.best_loss:.4f}")
        print(f"  Modelos en      : {self.model_dir}")
        print(f"{'─'*64}\n")


def _load_neural_models(model_dir):
    cfg_path   = Path(model_dir) / CycleTrainer.CONFIG_NAME
    model_path = Path(model_dir) / CycleTrainer.BEST_NAME
    if not cfg_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Modelo neural no encontrado en {model_dir}. ¿Has ejecutado train --backend neural?")
    with open(cfg_path) as f:
        cfg = json.load(f)
    G_A2B = _build_generator(cfg['n_roles'], cfg['resolution'], cfg['n_pitch'],
                              cfg.get('n_res_blocks', 9), cfg.get('base_ch', 64))
    G_B2A = _build_generator(cfg['n_roles'], cfg['resolution'], cfg['n_pitch'],
                              cfg.get('n_res_blocks', 9), cfg.get('base_ch', 64))
    state = torch.load(str(model_path), map_location='cpu', weights_only=False)  # FIX: explícito para torch>=2.0
    G_A2B.load_state_dict(state['G_A2B']); G_B2A.load_state_dict(state['G_B2A'])
    G_A2B.eval(); G_B2A.eval()
    return G_A2B, G_B2A, cfg


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN ROLL TRANSFORMADO → PARÁMETROS DNA (backend simbólico)
# ══════════════════════════════════════════════════════════════════════════════

def roll_vector_to_dna_overrides(src_vec, dst_vec, source_dna, model, intensity=1.0):
    """
    Deriva overrides de parámetros DNA desde el desplazamiento en el espacio de roll:
      · Densidad rítmica (slot melody vs src)
      · Registro medio de pitch (centro de masa del slot melody)
      · Complejidad armónica (ratio acc/melody)
      · Tensión emocional (pitch-classes disonantes {1,2,6,10,11})
    Solo sobreescribe los campos que el roll indica que deben cambiar.
    """
    out       = copy.deepcopy(source_dna)
    n_pitch   = (model.pitch_hi - model.pitch_lo + 1) if model.pitch_lo is not None else PITCH_CLASSES
    slot_size = model.resolution * n_pitch

    def _get_raw(vec):
        if model.pca is not None:
            return model.pca.inverse_transform(vec.reshape(1, -1))[0]
        return vec

    src_raw = _get_raw(src_vec)
    dst_raw = _get_raw(dst_vec)
    interp  = src_raw * (1.0 - intensity) + dst_raw * intensity

    def _slot(vec, role_name):
        if role_name not in model.active_roles: return None
        idx = model.active_roles.index(role_name)
        return vec[idx * slot_size: (idx + 1) * slot_size].reshape(model.resolution, n_pitch)

    mel_slot = _slot(interp, 'melody')
    acc_slot = _slot(interp, 'accompaniment')

    # ── Densidad rítmica ──────────────────────────────────────────────────
    if mel_slot is not None:
        src_mel = _slot(src_raw, 'melody')
        if src_mel is not None:
            out.harmony_complexity = float(np.clip(
                getattr(source_dna, 'harmony_complexity', 0.3)
                + (float(mel_slot.mean()) - float(src_mel.mean())),
                0.0, 1.0))

    # ── Registro medio de pitch ───────────────────────────────────────────
    if mel_slot is not None and mel_slot.sum() > 0:
        pitch_lo_eff = model.pitch_lo if model.pitch_lo is not None else 0
        mean_idx     = float(np.average(np.arange(n_pitch),
                                        weights=mel_slot.sum(axis=0) + 1e-9))
        src_mel2 = _slot(src_raw, 'melody')
        if src_mel2 is not None and src_mel2.sum() > 0:
            src_mean_idx = float(np.average(np.arange(n_pitch),
                                            weights=src_mel2.sum(axis=0) + 1e-9))
            pitch_delta  = (mean_idx - src_mean_idx) * intensity
            if abs(pitch_delta) > 1.0:
                contour = list(getattr(source_dna, 'pitch_contour', []) or [])
                if contour:
                    shift = int(round(pitch_delta))
                    out.pitch_contour = [int(np.clip(p + shift, 21, 108)) for p in contour]

    # ── Complejidad armónica acc/melody ───────────────────────────────────
    if acc_slot is not None and mel_slot is not None:
        ratio   = float(acc_slot.mean()) / (float(mel_slot.mean()) + 1e-9)
        src_acc = _slot(src_raw, 'accompaniment'); src_mel3 = _slot(src_raw, 'melody')
        if src_acc is not None and src_mel3 is not None:
            src_ratio = float(src_acc.mean()) / (float(src_mel3.mean()) + 1e-9)
            out.harmony_complexity = float(np.clip(
                out.harmony_complexity + (ratio - src_ratio) * intensity * 0.3, 0.0, 1.0))

    # ── Tensión emocional (pitch-classes disonantes) ──────────────────────
    if mel_slot is not None:
        pitch_lo_eff  = model.pitch_lo if model.pitch_lo is not None else 0
        DISSONANT_PCS = {1, 2, 6, 10, 11}
        pc_profile    = np.zeros(12, dtype=np.float32)
        for pc in range(12):
            idxs = [i for i in range(n_pitch) if (pitch_lo_eff + i) % 12 == pc]
            if idxs: pc_profile[pc] = mel_slot[:, idxs].mean()
        diss_frac = sum(pc_profile[pc] for pc in DISSONANT_PCS) / (pc_profile.sum() + 1e-9)

        src_mel4 = _slot(src_raw, 'melody')
        if src_mel4 is not None:
            src_pc = np.zeros(12, dtype=np.float32)
            for pc in range(12):
                idxs = [i for i in range(n_pitch) if (pitch_lo_eff + i) % 12 == pc]
                if idxs: src_pc[pc] = src_mel4[:, idxs].mean()
            src_diss      = sum(src_pc[pc] for pc in DISSONANT_PCS) / (src_pc.sum() + 1e-9)
            delta_tension = (diss_frac - src_diss) * intensity
            tc            = list(getattr(source_dna, 'tension_curve', [0.5]) or [0.5])
            out.tension_curve = [float(np.clip(t + delta_tension, 0.0, 1.0)) for t in tc]

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE CORPUS (backend simbólico)
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus_rolls(directory, resolution=TICKS_PER_BAR, active_roles=None,
                      pitch_lo=None, pitch_hi=None, verbose=False):
    midi_files = (glob.glob(os.path.join(directory, "*.mid")) +
                  glob.glob(os.path.join(directory, "*.midi")))
    if not midi_files:
        raise RuntimeError(f"No se encontraron MIDIs en {directory}")

    vecs = []
    for path in midi_files:
        try:
            v = midi_to_roll_vector(path, resolution, active_roles or ROLES,
                                    pitch_lo, pitch_hi, verbose=False)
            vecs.append(v)
            if verbose: print(f"  ✓ {os.path.basename(path)}")
        except Exception as e:
            print(f"  ✗ {os.path.basename(path)}: {e}")

    if not vecs:
        raise RuntimeError(f"No se pudo procesar ningún MIDI en {directory}")

    arr = np.array(vecs, dtype=np.float32)
    print(f"  {len(arr)} MIDIs cargados desde '{directory}' | vector dim = {arr.shape[1]}")
    return arr


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE GENERACIÓN DNA (compartido por ambos backends)
# ══════════════════════════════════════════════════════════════════════════════

def _snap_melody_to_key(melody, key_obj):
    return [(o, _snap_to_scale(p, key_obj), d, v) for o, p, d, v in melody]


def run_cycle_gan_transform(source_dna, transformed_dna, original_melody,
                            intensity=1.0, n_bars=16, candidates=3,
                            seed=42, verbose=False):
    random.seed(seed); np.random.seed(seed)

    target_key = transformed_dna.key_obj or source_dna.key_obj
    tempo_bpm  = transformed_dna.tempo_bpm
    time_sig   = transformed_dna.time_sig or source_dna.time_sig
    bpb        = time_sig[0]

    prog_s = source_dna.harmony_prog or [('I', 2.0)]
    prog_t = transformed_dna.harmony_prog or [('I', 2.0)]
    n_prog = max(len(prog_s), len(prog_t))
    prog_s = (prog_s * (n_prog // max(len(prog_s), 1) + 1))[:n_prog]
    prog_t = (prog_t * (n_prog // max(len(prog_t), 1) + 1))[:n_prog]
    mixed_prog = [(ft, dt) if random.random() < intensity else (fs, ds)
                  for (fs, ds), (ft, dt) in zip(prog_s, prog_t)]

    ec = EmotionalController(
        tension_curve   = transformed_dna.tension_curve   or [0.5],
        arousal_curve   = transformed_dna.arousal_curve   or [0.0],
        valence_curve   = transformed_dna.valence_curve   or [0.0],
        stability_curve = getattr(transformed_dna, 'stability_curve', None) or [0.7],
        activity_curve  = getattr(transformed_dna, 'activity_curve',  None) or [0.5],
        emotional_arc_label = getattr(transformed_dna, 'emotional_arc_label', ''),
        n_bars = n_bars,
    )
    fg = FormGenerator(
        form_string       = getattr(transformed_dna, 'form_string',       'AABA'),
        section_map       = getattr(transformed_dna, 'section_map',       {}),
        phrase_lengths    = getattr(transformed_dna, 'phrase_lengths',    [4]),
        cadence_positions = getattr(transformed_dna, 'cadence_positions', []),
        n_bars_out        = n_bars,
    )

    total_beats = n_bars * bpb
    if original_melody:
        t_in   = max(o + d for o, _, d, _ in original_melody)
        scale  = total_beats / max(t_in, 1e-9)
        scaled = [(o * scale, p, d * scale, v) for o, p, d, v in original_melody]
        scaled = _snap_melody_to_key(scaled, target_key)
    else:
        scaled = []

    groove = (transformed_dna.groove_map
              if getattr(transformed_dna, 'groove_map', None)
              and transformed_dna.groove_map.trained else None)

    best_score, best_result = -1.0, None

    for ci in range(max(1, candidates)):
        random.seed(seed + ci * 17); np.random.seed(seed + ci * 17)

        generated = dna_mod._generate_melody_with_modulation(
            h_prog              = mixed_prog,
            target_key          = target_key,
            r_pat               = transformed_dna.rhythm_pattern or source_dna.rhythm_pattern,
            contour             = transformed_dna.pitch_contour  or source_dna.pitch_contour,
            reg                 = source_dna.pitch_register,
            motif               = source_dna.motif_intervals,
            n_bars              = n_bars,
            ec                  = ec,
            fg                  = fg,
            bpb                 = bpb,
            rhythm_strength     = intensity,
            markov              = transformed_dna.markov,
            seq_phrases         = getattr(transformed_dna, 'sequitur_phrases', []),
            melody_mode         = 'markov',
            surprise_rate       = 0.08,
            use_motif_coherence = True,
            use_tension_markov  = True,
        )

        if scaled and generated:
            n = min(len(scaled), len(generated))
            alpha = 1.0 - intensity
            gen_s = sorted(generated, key=lambda x: x[0])
            src_s = sorted(scaled,    key=lambda x: x[0])
            mel   = []
            for i in range(n):
                oa, pa, da, va = src_s[i]; ob, pb, db, vb = gen_s[i]
                mp = _snap_to_scale(int(round(pa * alpha + pb * (1 - alpha))), target_key)
                mel.append((oa, mp, da * alpha + db * (1 - alpha),
                            int(va * alpha + vb * (1 - alpha))))
        elif scaled:
            mel = scaled
        else:
            mel = generated

        mel = add_ornamentation(mel, target_key, getattr(transformed_dna, 'style', 'generic'))
        if groove: mel = humanize(mel, groove, bpb)

        acc  = generate_accompaniment(mixed_prog, target_key, n_bars, ec, fg, bpb,
                                      groove_map=groove,
                                      harmony_complexity=transformed_dna.harmony_complexity)
        bass = generate_bass(mixed_prog, target_key, n_bars, bpb, groove)
        cp   = generate_counterpoint(mel, mixed_prog, target_key, n_bars, bpb, ec)

        sc = score_candidate(mel, acc, target_key)
        if verbose: print(f"    Candidato {ci+1}/{candidates}: score={sc:.3f}")
        if sc > best_score: best_score, best_result = sc, (mel, acc, bass, cp)

    mel, acc, bass, cp = best_result
    perc = generate_percussion(
        transformed_dna.rhythm_grid, transformed_dna.rhythm_accent_grid,
        n_bars, bpb, groove_map=groove,
        style=getattr(transformed_dna, 'style', 'generic'))

    return mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARAR CORPUS (.npz, backend neural)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_one_midi(args_tuple):
    (midi_path, output_dir, resolution, window_bars,
     active_roles, pitch_lo, pitch_hi) = args_tuple

    stem    = midi_path.stem
    stats   = {r: 0 for r in ROLES}
    stats.update({'files_ok': 0, 'files_skipped': 0, 'total_windows': 0})
    n_pitch = (pitch_hi - pitch_lo + 1) if pitch_lo is not None else PITCH_CLASSES

    try:
        mid = mido.MidiFile(str(midi_path))
    except Exception as e:
        return stem, f"ERROR al cargar: {e}", None, stats

    note_lists = _extract_note_lists(mid)
    if not note_lists:
        stats['files_skipped'] = 1
        return stem, "sin notas — omitido", None, stats

    role_assignment = RoleAssigner().assign(mid)
    if not role_assignment:
        stats['files_skipped'] = 1
        return stem, "sin asignación de roles — omitido", None, stats

    tpb_raw    = mid.ticks_per_beat
    # FIX: leer compás real en lugar de asumir 4/4
    ts_num = 4
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                ts_num = msg.numerator
                break
    tpbar_real = tpb_raw * ts_num
    all_ticks  = max((n[1] for notes in note_lists.values() for n in notes), default=0)
    total_bars = max(1, int(all_ticks / tpbar_real) + 1)
    converter  = PianoRollConverter(resolution=resolution, window_bars=window_bars)

    role_rolls  = {}
    roles_found = []
    for role, key in role_assignment.items():
        if role not in active_roles: continue
        notes = note_lists.get(key, [])
        if not notes: continue
        roll = converter.notes_to_roll(notes, tpbar_real, total_bars)
        if pitch_lo is not None:
            roll = _crop_pitch(roll, pitch_lo, pitch_hi)
        role_rolls[role] = roll
        roles_found.append(role); stats[role] = 1

    if not role_rolls:
        stats['files_skipped'] = 1
        return stem, "no se pudo construir ningún piano roll — omitido", None, stats

    role_windows = {}; min_windows = None
    for role, roll in role_rolls.items():
        windows = converter.roll_to_windows(roll)
        if windows.shape[0] == 0: continue
        role_windows[role] = windows
        min_windows = (windows.shape[0] if min_windows is None
                       else min(min_windows, windows.shape[0]))

    if not min_windows:
        stats['files_skipped'] = 1
        return stem, f"demasiado corto ({total_bars} compases) — omitido", None, stats

    save_dict = {f'roll_{r}': role_windows[r][:min_windows] for r in role_windows}
    meta = {
        'source': stem, 'resolution': resolution, 'window_bars': window_bars,
        'total_bars': total_bars, 'n_windows': min_windows,
        'roles': roles_found, 'tpbar': tpbar_real,   # FIX: era tpb_raw (= tpb*4, asumía 4/4)
        'ts_num': ts_num,          # FIX: guardar compás real para que _rolls_to_midi lo use
        'pitch_lo': pitch_lo if pitch_lo is not None else 0,
        'pitch_hi': pitch_hi if pitch_hi is not None else 127,
        'n_pitch':  n_pitch,
    }
    save_dict['meta_json'] = np.array([json.dumps(meta)])
    np.savez_compressed(str(Path(output_dir) / f"{stem}.npz"), **save_dict)

    stats['files_ok'] = 1; stats['total_windows'] = min_windows
    return (stem,
            f"OK  ({total_bars} compases, {min_windows} ventanas, "
            f"roles: {', '.join(roles_found)})",
            True, stats)


def cmd_prepare(args):
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    disabled     = set(getattr(args, 'disable_roles', None) or [])
    active_roles = [r for r in ROLES if r not in disabled]
    pr           = _pitch_range(getattr(args, 'pitch_range', None))
    pitch_lo     = pr[0] if pr else None
    pitch_hi     = pr[1] if pr else None

    if pr:
        print(f"[prepare] Rango de pitch: {args.pitch_range} notas (MIDI {pitch_lo}–{pitch_hi})")
    if disabled:
        print(f"[prepare] Roles deshabilitados: {', '.join(sorted(disabled))}")

    midi_files = sorted(list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))
    if not midi_files:
        print(f"[prepare] No se encontraron MIDI en {input_dir}"); sys.exit(1)

    print(f"[prepare] {len(midi_files)} archivos MIDI → {output_dir}\n")

    job_args    = [(p, output_dir, args.resolution, args.window_bars,
                    active_roles, pitch_lo, pitch_hi) for p in midi_files]
    stats_total = {r: 0 for r in ROLES}
    stats_total.update({'files_ok': 0, 'files_skipped': 0, 'total_windows': 0})

    with ThreadPoolExecutor() as ex:
        futures = {ex.submit(_prepare_one_midi, a): a[0] for a in job_args}
        for fut in as_completed(futures):
            stem, msg, ok, partial = fut.result()
            print(f"  {'[OK]' if ok else '[--]'} {stem:40s}  {msg}")
            for k in stats_total: stats_total[k] += partial.get(k, 0)

    print(f"\n[prepare] Archivos OK       : {stats_total['files_ok']}")
    print(f"[prepare] Archivos omitidos : {stats_total['files_skipped']}")
    print(f"[prepare] Ventanas totales  : {stats_total['total_windows']}")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_roll_args(args):
    active_roles = [r for r in ROLES
                    if r not in set(getattr(args, 'disable_roles', None) or [])]
    pr           = _pitch_range(getattr(args, 'pitch_range', None))
    pitch_lo     = pr[0] if pr else None
    pitch_hi     = pr[1] if pr else None
    return active_roles, pitch_lo, pitch_hi, getattr(args, 'resolution', TICKS_PER_BAR)


def _compute_n_bars(args, original_melody, source_dna):
    if args.bars: return args.bars
    bpb = source_dna.time_sig[0]
    if original_melody:
        total_beats = max(o + d for o, _, d, _ in original_melody)
        return max(4, (int(math.ceil(total_beats / bpb)) // 4) * 4)
    return 16


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    backend = getattr(args, 'backend', 'symbolic')
    print("═" * 65)
    print(f"  CYCLE-GAN STYLE TRANSFER v3 — TRAIN  [{backend}]")
    print("═" * 65)
    if backend == 'neural':
        if not HAS_TORCH:
            print("ERROR: torch no disponible. Usa --backend symbolic."); sys.exit(1)
        _cmd_train_neural(args)
    else:
        _cmd_train_symbolic(args)


def _cmd_train_symbolic(args):
    active_roles, pitch_lo, pitch_hi, resolution = _parse_roll_args(args)
    pca_dim = getattr(args, 'pca_dim', 0)

    print(f"  Dominio A    : {args.domain_a}")
    print(f"  Dominio B    : {args.domain_b}")
    print(f"  Modelo       : {args.model}")
    print(f"  Resolución   : {resolution}  Pitch: {args.pitch_range or 128}  PCA: {pca_dim or 'no'}")
    print(f"  λ_ciclo={args.lambda_cycle}  λ_identity={args.lambda_identity}  iters={args.iters}")

    print("\n[1/3] Cargando corpus A…")
    feats_a = load_corpus_rolls(args.domain_a, resolution, active_roles,
                                pitch_lo, pitch_hi, args.verbose)
    print("\n[2/3] Cargando corpus B…")
    feats_b = load_corpus_rolls(args.domain_b, resolution, active_roles,
                                pitch_lo, pitch_hi, args.verbose)

    print("\n[3/3] Entrenando CycleGAN simbólico…")
    model = CycleGANSymbolic(
        solver=args.solver, alpha=args.alpha, k_neighbors=args.k_neighbors,
        lambda_cycle=args.lambda_cycle, lambda_identity=args.lambda_identity,
        resolution=resolution, active_roles=active_roles,
        pitch_lo=pitch_lo, pitch_hi=pitch_hi, pca_dim=pca_dim,
    )
    model.domain_a_name = os.path.basename(args.domain_a.rstrip('/'))
    model.domain_b_name = os.path.basename(args.domain_b.rstrip('/'))
    model.fit(feats_a, feats_b, iters=args.iters, verbose=args.verbose)
    model.save(args.model)

    if model.losses_history:
        last = model.losses_history[-1]
        print(f"\n  Pérdida ciclo A/B : {last['cycle_a']:.4f} / {last['cycle_b']:.4f}")
        print(f"  Pérdida ident G/F : {last['identity_g']:.4f} / {last['identity_f']:.4f}")

    print("═" * 65)


def _cmd_train_neural(args):
    from torch.utils.data import DataLoader, random_split

    dir_A     = Path(args.domain_a)
    dir_B     = Path(args.domain_b)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    disabled     = set(getattr(args, 'disable_roles', None) or [])
    active_roles = [r for r in ROLES if r not in disabled]

    print(f"  Corpus A: {dir_A}  |  Corpus B: {dir_B}")
    print(f"  model_dir: {model_dir}  épocas: {args.epochs}  batch: {args.batch_size}")

    ds_A = MidiRollDataset(str(dir_A), roles=active_roles)
    ds_B = MidiRollDataset(str(dir_B), roles=active_roles)

    if len(ds_A) == 0 or len(ds_B) == 0:
        print("ERROR: algún dataset está vacío. Ejecuta primero: prepare")
        sys.exit(1)

    sample     = ds_A[0]
    n_roles    = sample.shape[0]
    resolution = ds_A.samples[0][2]['resolution']
    n_pitch    = ds_A.n_pitch

    print(f"  n_roles={n_roles}  resolution={resolution}  n_pitch={n_pitch}")
    print(f"  Corpus A: {len(ds_A)} ventanas  |  Corpus B: {len(ds_B)} ventanas")

    def _split(ds):
        n_val = max(1, int(len(ds) * 0.1))
        return random_split(ds, [len(ds) - n_val, n_val])

    tr_A, val_A = _split(ds_A); tr_B, val_B = _split(ds_B)
    kw = dict(collate_fn=_collate_fn, num_workers=0, pin_memory=False)
    loader_A_tr  = DataLoader(tr_A,  batch_size=args.batch_size, shuffle=True,  **kw)
    loader_B_tr  = DataLoader(tr_B,  batch_size=args.batch_size, shuffle=True,  **kw)
    loader_A_val = DataLoader(val_A, batch_size=args.batch_size, shuffle=False, **kw)
    loader_B_val = DataLoader(val_B, batch_size=args.batch_size, shuffle=False, **kw)

    n_res_blocks = getattr(args, 'n_res_blocks', 9)
    base_ch      = getattr(args, 'base_ch', 64)
    lr           = getattr(args, 'lr', 2e-4)

    G_A2B = _build_generator(n_roles, resolution, n_pitch, n_res_blocks, base_ch)
    G_B2A = _build_generator(n_roles, resolution, n_pitch, n_res_blocks, base_ch)
    D_A   = _build_discriminator(n_roles, base_ch)
    D_B   = _build_discriminator(n_roles, base_ch)

    opt_G   = torch.optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()),
                               lr=lr, betas=(0.5, 0.999))
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

    _npz  = sorted(dir_A.glob('*.npz'))[0]
    _meta = json.loads(str(np.load(str(_npz), allow_pickle=True)['meta_json'][0]))
    cfg = {
        'n_roles': n_roles, 'roles': active_roles,
        'resolution': resolution, 'n_pitch': n_pitch,
        'pitch_lo': _meta.get('pitch_lo', 0), 'pitch_hi': _meta.get('pitch_hi', 127),
        'window_bars': _meta.get('window_bars', WINDOW_BARS_DEFAULT),
        'base_ch': base_ch, 'n_res_blocks': n_res_blocks,
        'lambda_cycle': args.lambda_cycle, 'lambda_identity': args.lambda_identity,
        'model_version': 'cyclegan_v3_neural',
    }
    with open(model_dir / CycleTrainer.CONFIG_NAME, 'w') as f:
        json.dump(cfg, f, indent=2)

    trainer = CycleTrainer(
        G_A2B, G_B2A, D_A, D_B, opt_G, opt_D_A, opt_D_B,
        model_dir,
        patience=getattr(args, 'patience', 40),
        lambda_cycle=args.lambda_cycle,
        lambda_ident=args.lambda_identity,
    )
    trainer.train(loader_A_tr, loader_B_tr, loader_A_val, loader_B_val,
                  args.epochs, resume=getattr(args, 'resume', False))


# ── TRANSFORM ─────────────────────────────────────────────────────────────────

def cmd_transform(args):
    backend = getattr(args, 'backend', 'symbolic')
    print("═" * 65)
    print(f"  CYCLE-GAN STYLE TRANSFER v3 — TRANSFORM  [{backend}]")
    print("═" * 65)
    if backend == 'neural':
        _cmd_transform_neural(args)
    else:
        _cmd_transform_symbolic(args)


def _cmd_transform_symbolic(args):
    model = CycleGANSymbolic.load(args.model)

    # ── Vectorización ──────────────────────────────────────────────────────
    print("\n[1/4] Convirtiendo a piano roll centroide…")
    feat_src = model.midi_to_vec(args.input, verbose=args.verbose)
    print(f"  ✓ {len(feat_src)}D" + (f" (PCA {model.pca_dim}D)" if model.pca else ""))

    print("\n[2/4] Aplicando CycleGAN simbólico…")
    feat_dst    = model.map_features(feat_src, direction=args.direction)
    feat_interp = feat_src * (1 - args.intensity) + feat_dst * args.intensity
    disc        = model.discriminate(feat_interp)
    if args.verbose:
        print(f"  D_A prob: {disc['D_A_prob']:.3f}  |  D_B prob: {disc['D_B_prob']:.3f}")

    if HAS_DNA:
        # ── Pipeline completo vía DNA ──────────────────────────────────────
        print("\n[3/4] Extrayendo melodía…")
        original_melody, _, _ = extract_raw_melody(args.input, verbose=args.verbose)
        print(f"  ✓ {len(original_melody)} notas")

        print("\n[4/4] Derivando parámetros DNA y generando MIDI…")
        source_dna = UnifiedDNA(args.input)
        source_dna.extract(verbose=args.verbose)
        transformed_dna = roll_vector_to_dna_overrides(
            feat_src, feat_interp, source_dna, model, intensity=args.intensity)

        n_bars = _compute_n_bars(args, original_melody, source_dna)
        print(f"       {n_bars} compases, {args.candidates} candidatos…")

        mel, acc, bass, cp, perc, target_key, tempo_bpm, time_sig, fg = run_cycle_gan_transform(
            source_dna, transformed_dna, original_melody,
            intensity=args.intensity, n_bars=n_bars,
            candidates=args.candidates, seed=args.seed, verbose=args.verbose)

        out_path = args.output or (
            os.path.splitext(args.input)[0] + f"_cganv3_{args.direction}.mid")
        build_midi(mel, acc, bass, cp, target_key, tempo_bpm, time_sig, n_bars,
                   form_gen=fg, output_path=out_path,
                   percussion_notes=None if args.no_percussion else perc)

        if args.export_fingerprint:
            fp = dna_mod.extract_fingerprint(
                mel, bass, acc, target_key, tempo_bpm, n_bars, time_sig,
                fg, transformed_dna, out_path)
            print(f"  → Fingerprint: {dna_mod.export_fingerprint(fp, out_path)}")

        print("\n" + "═" * 65)
        print(f"  Salida    : {out_path}")
        print(f"  Tonalidad : {target_key.tonic.name} {target_key.mode}  |  {tempo_bpm:.0f} BPM")
        print(f"  D_A prob  : {disc['D_A_prob']:.3f}  |  D_B prob: {disc['D_B_prob']:.3f}")
        print("═" * 65)

    else:
        # ── FIX: Fallback roll-directo cuando midi_dna_unified no está disponible ──
        # Pipeline: MIDI → rolls → modular amplitud por ratio de energía por rol → MIDI
        print("\n  [AVISO] midi_dna_unified no disponible; usando pipeline de rolls directo.")
        print("\n[3/4] Leyendo piano rolls del MIDI fuente…")

        cfg = {
            'resolution': model.resolution,
            'roles':      model.active_roles,
            'pitch_lo':   model.pitch_lo,
            'pitch_hi':   model.pitch_hi,
            'window_bars': WINDOW_BARS_DEFAULT,
        }
        rolls, ts_num_src = _midi_to_rolls(args.input, cfg)
        if not rolls:
            print("  ERROR: no se encontraron notas en el MIDI."); sys.exit(1)
        cfg['ts_num'] = ts_num_src   # propagar compás para _rolls_to_midi

        bpm    = _infer_bpm_from_midi(args.input)
        n_bars = min(r.shape[0] for r in rolls.values())
        print(f"  ✓ {n_bars} compases | {list(rolls.keys())} | {bpm:.0f} BPM")

        print("\n[4/4] Aplicando transformación por energía de rol…")
        # Reconstruir vectores raw desde PCA para calcular ratios por segmento de rol
        n_pitch  = (model.pitch_hi - model.pitch_lo + 1) if model.pitch_lo is not None else PITCH_CLASSES
        seg_dim  = model.resolution * n_pitch

        if model.pca is not None:
            vec_raw_src = model.pca.inverse_transform(feat_src.reshape(1, -1))[0]
            vec_raw_dst = model.pca.inverse_transform(feat_interp.reshape(1, -1))[0]
        else:
            vec_raw_src = feat_src
            vec_raw_dst = feat_interp

        # FIX: enmascarar roles ausentes (energía src ~0) para evitar ratios degenerados
        transformed_rolls = {}
        for i, role in enumerate(model.active_roles):
            if role not in rolls:
                continue
            s      = i * seg_dim
            e      = s + seg_dim
            e_src  = float(np.abs(vec_raw_src[s:e]).mean())
            e_dst  = float(np.abs(vec_raw_dst[s:e]).mean())
            # Solo aplicar ratio si el rol tiene energía real en el src
            if e_src < 1e-7:
                ratio = 1.0          # rol ausente en src: no modificar
            else:
                ratio = float(np.clip(e_dst / (e_src + 1e-9), 0.25, 4.0))
            if args.verbose:
                print(f"  {role}: e_src={e_src:.5f}  e_dst={e_dst:.5f}  ratio={ratio:.3f}")
            transformed_rolls[role] = np.clip(rolls[role] * ratio, 0.0, 1.0)

        mid_src  = mido.MidiFile(args.input)
        role_map = RoleAssigner().assign(mid_src)
        palette  = _extract_palette_from_midi(args.input, role_map)

        out_path = args.output or (
            os.path.splitext(args.input)[0] + f"_cganv3_{args.direction}.mid")
        threshold_pct = getattr(args, 'threshold_pct', 97.0)  # FIX: 97 más adecuado para rolls simbólicos
        n_notes = _rolls_to_midi(transformed_rolls, cfg, palette, out_path,
                                 bpm=bpm, threshold_pct=threshold_pct)

        print("\n" + "═" * 65)
        print(f"  Salida    : {out_path}  ({n_notes} notas)")
        print(f"  D_A prob  : {disc['D_A_prob']:.3f}  |  D_B prob: {disc['D_B_prob']:.3f}")
        if n_notes == 0:
            print("  ⚠  MIDI vacío — prueba --threshold-pct 95")
        print("═" * 65)


def _cmd_transform_neural(args):
    if not HAS_TORCH:
        print("ERROR: torch no disponible."); sys.exit(1)

    model_dir = Path(args.model_dir)
    G_A2B, G_B2A, cfg = _load_neural_models(model_dir)
    generator  = G_A2B if args.direction == 'AtoB' else G_B2A
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator.to(device)
    palette    = _load_palette(getattr(args, 'palette', None), cfg)

    print(f"\n  {args.input}  ({'A→B' if args.direction == 'AtoB' else 'B→A'})")
    rolls, ts_num_src = _midi_to_rolls(args.input, cfg)
    if not rolls:
        print("ERROR: no se encontraron notas."); sys.exit(1)
    cfg['ts_num'] = ts_num_src

    role_list  = cfg['roles']
    n_roles    = cfg['n_roles']
    resolution = cfg['resolution']
    n_pitch    = cfg.get('n_pitch', 128)
    n_bars     = min(r.shape[0] for r in rolls.values())
    threshold_pct = getattr(args, 'threshold_pct', 99.0)

    print(f"  {n_bars} compases  ·  {n_roles} roles")

    # FIX: construir un batch completo (n_bars, n_roles, res, n_pitch) y hacer
    # un único forward pass en lugar de n_bars pasadas individuales (mucho más rápido)
    x_all = np.zeros((n_bars, n_roles, resolution, n_pitch), dtype=np.float32)
    for ridx, role in enumerate(role_list):
        if role in rolls:
            x_all[:, ridx] = rolls[role][:n_bars]

    x_t = torch.tensor(x_all).to(device)
    with torch.no_grad():
        y_all = generator(x_t).cpu().numpy()   # (n_bars, n_roles, res, n_pitch)

    # Diagnóstico en el primer compás
    y0 = y_all[0]
    thr   = _adaptive_threshold(y0, threshold_pct)
    n_act = int((y0 > thr).sum())
    print(f"\n  [diag] Compás 0: mean={y0.mean():.4f}  max={y0.max():.4f}  "
          f"umbral={thr:.4f}  notas_activas={n_act}")
    if n_act == 0:
        print("         ⚠  sin notas activas — prueba --threshold-pct 97")

    bars_per_role = {}
    for ridx, role in enumerate(role_list):
        bars_per_role[role] = y_all[:, ridx]   # (n_bars, res, n_pitch)

    out_path = args.output or (
        os.path.splitext(args.input)[0] + f"_neural_{args.direction}.mid")
    n_notes  = _rolls_to_midi(bars_per_role, cfg, palette, out_path,
                              bpm=getattr(args, 'bpm', 120.0),
                              threshold=getattr(args, 'threshold', None),
                              threshold_pct=threshold_pct)
    print(f"\n  Guardado: {out_path}  ({n_notes} notas, {n_bars} compases)")
    if n_notes == 0:
        print("  ⚠  MIDI vacío — ajusta --threshold-pct (prueba 95.0 ó 97.0)")


# ── CYCLE ─────────────────────────────────────────────────────────────────────

def cmd_cycle(args):
    print("═" * 65)
    print("  CYCLE-GAN STYLE TRANSFER v3 — CYCLE CONSISTENCY")
    print("═" * 65)

    if not HAS_DNA:
        # Fallback: medir consistencia de ciclo solo en feature space, sin generar MIDI
        print("\n  [AVISO] midi_dna_unified no disponible. "
              "Calculando pérdida de ciclo en feature space únicamente.")
        model    = CycleGANSymbolic.load(args.model)
        feat_src = model.midi_to_vec(args.input, verbose=args.verbose)
        dir1     = args.direction
        dir2     = 'BtoA' if dir1 == 'AtoB' else 'AtoB'
        feat_ab  = model.map_features(feat_src, direction=dir1)
        feat_aba = model.map_features(feat_ab,  direction=dir2)
        cycle_loss = float(np.linalg.norm(feat_aba - feat_src))
        print(f"\n  Pérdida de ciclo ‖F(G(a)) − a‖₂ : {cycle_loss:.4f}")
        disc_src = model.discriminate(feat_src)
        disc_ab  = model.discriminate(feat_ab)
        print(f"  Fuente   → D_A={disc_src['D_A_prob']:.3f}  D_B={disc_src['D_B_prob']:.3f}")
        print(f"  Tras G   → D_A={disc_ab['D_A_prob']:.3f}  D_B={disc_ab['D_B_prob']:.3f}")
        print("═" * 65)
        return

    model = CycleGANSymbolic.load(args.model)
    original_melody, _, _ = extract_raw_melody(args.input, verbose=args.verbose)
    feat_src = model.midi_to_vec(args.input, verbose=args.verbose)
    source_dna = UnifiedDNA(args.input); source_dna.extract(verbose=args.verbose)

    dir1 = args.direction; dir2 = 'BtoA' if dir1 == 'AtoB' else 'AtoB'

    print(f"\n[1/4] Aplicando {dir1} (G)…")
    feat_ab   = model.map_features(feat_src, direction=dir1)
    feat_ab_i = feat_src * (1 - args.intensity) + feat_ab * args.intensity
    dna_ab    = roll_vector_to_dna_overrides(feat_src, feat_ab_i, source_dna, model,
                                             intensity=args.intensity)

    n_bars = _compute_n_bars(args, original_melody, source_dna)

    print(f"\n[2/4] Generando MIDI intermedio ({n_bars} compases)…")
    mel_ab, acc_ab, bass_ab, cp_ab, perc_ab, tk_ab, tmp_ab, ts_ab, fg_ab = \
        run_cycle_gan_transform(source_dna, dna_ab, original_melody,
                                intensity=args.intensity, n_bars=n_bars,
                                candidates=args.candidates, seed=args.seed,
                                verbose=args.verbose)
    out_ab = args.output_ab or (os.path.splitext(args.input)[0] + "_ab.mid")
    build_midi(mel_ab, acc_ab, bass_ab, cp_ab, tk_ab, tmp_ab, ts_ab, n_bars,
               form_gen=fg_ab, output_path=out_ab,
               percussion_notes=None if args.no_percussion else perc_ab)
    print(f"  ✓ {out_ab}")

    print(f"\n[3/4] Aplicando {dir2} (F)…")
    feat_ab_real = model.midi_to_vec(out_ab, verbose=False)
    feat_aba     = model.map_features(feat_ab_real, direction=dir2)
    feat_aba_i   = feat_ab_real * (1 - args.intensity) + feat_aba * args.intensity
    dna_ab_file  = UnifiedDNA(out_ab); dna_ab_file.extract(verbose=False)
    dna_aba      = roll_vector_to_dna_overrides(feat_ab_real, feat_aba_i, dna_ab_file, model,
                                               intensity=args.intensity)
    mel_aba_src, _, _ = extract_raw_melody(out_ab, verbose=False)

    print(f"\n[4/4] Generando MIDI reconstruido ({n_bars} compases)…")
    mel_aba, acc_aba, bass_aba, cp_aba, perc_aba, tk_aba, tmp_aba, ts_aba, fg_aba = \
        run_cycle_gan_transform(dna_ab_file, dna_aba, mel_aba_src,
                                intensity=args.intensity, n_bars=n_bars,
                                candidates=args.candidates, seed=args.seed + 100,
                                verbose=args.verbose)
    out_aba = args.output_aba or (os.path.splitext(args.input)[0] + "_aba.mid")
    build_midi(mel_aba, acc_aba, bass_aba, cp_aba, tk_aba, tmp_aba, ts_aba, n_bars,
               form_gen=fg_aba, output_path=out_aba,
               percussion_notes=None if args.no_percussion else perc_aba)

    cycle_loss = float(np.linalg.norm(model.map_features(feat_ab_real, dir2) - feat_src))

    print("\n" + "═" * 65)
    print(f"  Salida G    : {out_ab}")
    print(f"  Salida F∘G  : {out_aba}")
    print(f"  ‖F(G(a)) - a‖₂ : {cycle_loss:.4f}")
    print("═" * 65)


# ── ANALYZE ───────────────────────────────────────────────────────────────────

def cmd_analyze(args):
    print("═" * 65)
    print("  CYCLE-GAN STYLE TRANSFER v3 — ANALYZE")
    print("═" * 65)
    model   = CycleGANSymbolic.load(args.model)
    feat    = model.midi_to_vec(args.input, verbose=args.verbose)
    disc    = model.discriminate(feat)
    feat_ab = model.map_features(feat, 'AtoB')
    feat_ba = model.map_features(feat, 'BtoA')

    print(f"\n  Dominio A ({model.domain_a_name}):")
    print(f"    Distancia : {disc['D_A_dist']:.4f}  |  Prob: {disc['D_A_prob']:.4f}")
    print(f"\n  Dominio B ({model.domain_b_name}):")
    print(f"    Distancia : {disc['D_B_dist']:.4f}  |  Prob: {disc['D_B_prob']:.4f}")
    total = disc['D_A_prob'] + disc['D_B_prob'] + 1e-9
    print(f"\n  Posición: {disc['D_A_prob']/total*100:.1f}% A / {disc['D_B_prob']/total*100:.1f}% B")
    print(f"  Espacio: {'PCA '+str(model.pca_dim)+'D' if model.pca else 'roll '+str(len(feat))+'D'}")
    n_show = min(10, len(feat))
    print(f"\n  Vector (primeros {n_show}):")
    for i in range(n_show):
        print(f"    [{i:>3}]  src={feat[i]:.4f}  A→B={feat_ab[i]:.4f}  B→A={feat_ba[i]:.4f}")
    feat_aba = model.map_features(feat_ab, 'BtoA')
    print(f"\n  Pérdida ciclo (A→B→A): {float(np.linalg.norm(feat_aba - feat)):.4f}")
    print("═" * 65)


# ── STYLE-CORPUS ──────────────────────────────────────────────────────────────

def cmd_style_corpus(args):
    backend   = getattr(args, 'backend', 'symbolic')
    input_dir = Path(args.input_dir)
    direction = getattr(args, 'direction', 'a2b')
    dir_label = 'A→B' if direction == 'a2b' else 'B→A'
    midi_files = sorted(list(input_dir.glob('*.mid')) + list(input_dir.glob('*.midi')))

    if not midi_files:
        print(f"[style-corpus] No se encontraron MIDIs en {input_dir}"); sys.exit(1)

    print(f"[style-corpus] {len(midi_files)} archivos  ·  {dir_label}  [{backend}]\n")
    densities = []; skipped = 0

    if backend == 'neural':
        if not HAS_TORCH:
            print("ERROR: torch no disponible."); sys.exit(1)
        G_A2B, G_B2A, cfg = _load_neural_models(Path(args.model_dir))
        generator = G_A2B if direction == 'a2b' else G_B2A
        device    = 'cuda' if torch.cuda.is_available() else 'cpu'
        generator.to(device)
        n_roles = cfg['n_roles']; role_list = cfg['roles']
        resolution = cfg['resolution']; n_pitch = cfg.get('n_pitch', 128)

        for midi_path in midi_files:
            try:
                rolls, ts_num_src  = _midi_to_rolls(str(midi_path), cfg)
                cfg['ts_num'] = ts_num_src
                n_bars = min(r.shape[0] for r in rolls.values())
                x_np   = np.zeros((n_bars, n_roles, resolution, n_pitch), dtype=np.float32)
                for ridx, role in enumerate(role_list):
                    if role in rolls: x_np[:, ridx] = rolls[role][:n_bars]
                x_t = torch.tensor(x_np).to(device)
                with torch.no_grad():
                    y_np = generator(x_t).cpu().numpy()
                thr  = _adaptive_threshold(y_np)
                dens = float((y_np > thr).mean())
                densities.append(dens)
                print(f"  [OK] {midi_path.stem:40s}  densidad_out={dens:.4f}")
            except Exception as e:
                print(f"  [SKIP] {midi_path.stem} — {e}"); skipped += 1
    else:
        model = CycleGANSymbolic.load(args.model)
        direction_ = 'AtoB' if direction == 'a2b' else 'BtoA'
        for midi_path in midi_files:
            try:
                feat_src = model.midi_to_vec(str(midi_path))
                feat_dst = model.map_features(feat_src, direction=direction_)
                disc     = model.discriminate(feat_dst)
                dens     = disc['D_B_prob'] if direction == 'a2b' else disc['D_A_prob']
                densities.append(dens)
                print(f"  [OK] {midi_path.stem:40s}  D_dst_prob={dens:.4f}")
            except Exception as e:
                print(f"  [SKIP] {midi_path.stem} — {e}"); skipped += 1

    if not densities:
        print("[style-corpus] Sin resultados."); sys.exit(1)

    mean_d = float(np.mean(densities)); std_d = float(np.std(densities))
    out_path = args.output or f"corpus_stats_{input_dir.stem}_{direction}.json"
    with open(out_path, 'w') as f:
        json.dump({'n_files': len(densities), 'n_skipped': skipped,
                   'density_mean': mean_d, 'density_std': std_d,
                   'densities': densities, 'direction': direction}, f, indent=2)

    print(f"\n[style-corpus] Procesados: {len(densities)}  Omitidos: {skipped}")
    print(f"[style-corpus] Densidad media: {mean_d:.4f} ± {std_d:.4f}")
    if mean_d < 0.005: print("[style-corpus] ⚠  densidad muy baja — generador puede no haber convergido")
    if mean_d > 0.3:   print("[style-corpus] ⚠  densidad muy alta — posible colapso de modo")
    print(f"[style-corpus] → {out_path}")


# ── ROUND-TRIP ────────────────────────────────────────────────────────────────

def cmd_round_trip(args):
    """
    MIDI → piano roll → MIDI sin modelo (diagnóstico del parser).
    Preserva los programas, canales y velocidades del MIDI original.
    """
    cfg_rt = {
        'resolution':  args.resolution,
        'window_bars': WINDOW_BARS_DEFAULT,
        'roles':       [r for r in ROLES if r not in (args.disable_roles or [])],
        'n_pitch':     PITCH_CLASSES,
        'pitch_lo':    0,
        'pitch_hi':    127,
    }
    model_dir = getattr(args, 'model_dir', None)
    if model_dir:
        try:
            cfg_path = Path(model_dir) / CycleTrainer.CONFIG_NAME
            if cfg_path.exists():
                with open(cfg_path) as f:
                    loaded = json.load(f)
                # FIX: el --resolution del usuario tiene prioridad sobre el del model_dir
                user_resolution = args.resolution
                cfg_rt.update(loaded)
                cfg_rt['resolution'] = user_resolution
                print(f"[round-trip] Config desde {model_dir} (resolución del usuario: {user_resolution})")
        except Exception:
            pass

    # BPM y compás desde el fuente si no se especifica
    bpm = args.bpm if args.bpm != 120.0 else _infer_bpm_from_midi(args.input)
    ts_num, _ = _infer_time_sig_from_midi(args.input)
    cfg_rt['ts_num'] = ts_num   # FIX: propagar compás real para ticks correctos

    rolls, ts_num_src = _midi_to_rolls(args.input, cfg_rt)
    cfg_rt['ts_num']  = ts_num_src   # refinar con el valor real del MIDI (puede diferir)
    n_bars   = min(r.shape[0] for r in rolls.values())
    print(f"[round-trip] {n_bars} compases  ·  roles: {list(rolls.keys())}  ·  {bpm} BPM")

    # Paleta extraída del MIDI original
    mid_src  = mido.MidiFile(args.input)
    role_map = RoleAssigner().assign(mid_src)
    palette  = _extract_palette_from_midi(args.input, role_map)
    print(f"[round-trip] Paleta: { {r: (v['program'], v['channel']) for r,v in palette.items()} }")

    n_notes = _rolls_to_midi(rolls, cfg_rt, palette, args.output, bpm=bpm)
    print(f"[round-trip] → {args.output}  ({n_notes} notas)")
    if n_notes == 0:
        print("[round-trip] ⚠  MIDI vacío — posible problema en el parser")


# ── INSPECT ───────────────────────────────────────────────────────────────────

def cmd_inspect(args):
    if 'npz' in args.what and args.data_dir:
        data_dir  = Path(args.data_dir)
        npz_files = sorted(data_dir.glob('*.npz'))
        if not npz_files:
            print(f"[inspect] Sin .npz en {data_dir}")
        else:
            target = getattr(args, 'npz_file', None) or npz_files[0].name
            path   = data_dir / target
            if not path.exists():
                print(f"[inspect] No encontrado: {path}")
            else:
                data = np.load(str(path), allow_pickle=True)
                meta = json.loads(str(data['meta_json'][0]))
                print(f"[inspect] {path.name}")
                for k in ('resolution', 'window_bars', 'total_bars', 'n_windows', 'roles', 'n_pitch'):
                    print(f"  {k:20s}: {meta.get(k, '?')}")
                for role in meta['roles']:
                    key = f'roll_{role}'
                    if key in data:
                        arr = data[key]
                        print(f"    {role:20s}: shape={arr.shape}  densidad={float(arr.mean()):.5f}")

    if 'loss_curve' in args.what and args.model_dir:
        hist_path = Path(args.model_dir) / CycleTrainer.HISTORY_NAME
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)
            print(f"\n[inspect] Historial ({len(hist.get('train_G',[]))} épocas)")
            for key, vals in hist.items():
                if vals:
                    print(f"  {key:15s}: último={vals[-1]:.4f}  min={min(vals):.4f}")
        else:
            print(f"[inspect] Sin historial en {args.model_dir}")

    if args.model_dir:
        cfg_path = Path(args.model_dir) / CycleTrainer.CONFIG_NAME
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            print(f"\n[inspect] Config ({cfg_path})")
            for k, v in cfg.items():
                print(f"  {k:20s}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='cycle_gan_style_transfer_v3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="CYCLE-GAN STYLE TRANSFER v3.0 — simbólico + neural",
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── prepare ───────────────────────────────────────────────────────────────
    p_pr = sub.add_parser('prepare', help='MIDI corpus → .npz (backend neural)')
    p_pr.add_argument('--input-dir',    required=True)
    p_pr.add_argument('--output-dir',   required=True)
    p_pr.add_argument('--resolution',   type=int, default=TICKS_PER_BAR)
    p_pr.add_argument('--window-bars',  type=int, default=WINDOW_BARS_DEFAULT, dest='window_bars')
    p_pr.add_argument('--disable-roles', nargs='+', choices=ROLES, default=[], dest='disable_roles')
    p_pr.add_argument('--pitch-range',  type=int, default=None, dest='pitch_range')
    p_pr.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p_tr = sub.add_parser('train', help='Entrenar el modelo CycleGAN')
    p_tr.add_argument('--backend',        default='symbolic', choices=['symbolic', 'neural'])
    p_tr.add_argument('--domain-a',       default=None, dest='domain_a')
    p_tr.add_argument('--domain-b',       default=None, dest='domain_b')
    # Simbólico
    p_tr.add_argument('--model',          default='cycle_gan_v3.pkl')
    p_tr.add_argument('--resolution',     type=int,   default=TICKS_PER_BAR)
    p_tr.add_argument('--pitch-range',    type=int,   default=None, dest='pitch_range')
    p_tr.add_argument('--disable-roles',  nargs='+',  choices=ROLES, default=[], dest='disable_roles')
    p_tr.add_argument('--pca-dim',        type=int,   default=0, dest='pca_dim')
    p_tr.add_argument('--solver',         default='ridge', choices=['ridge', 'knn'])
    p_tr.add_argument('--alpha',          type=float, default=1.0)
    p_tr.add_argument('--k-neighbors',    type=int,   default=5, dest='k_neighbors')
    p_tr.add_argument('--lambda-cycle',   type=float, default=10.0, dest='lambda_cycle')
    p_tr.add_argument('--lambda-identity',type=float, default=5.0,  dest='lambda_identity')
    p_tr.add_argument('--iters',          type=int,   default=100)
    # Neural
    p_tr.add_argument('--model-dir',      default='model_neural', dest='model_dir')
    p_tr.add_argument('--epochs',         type=int,   default=200)
    p_tr.add_argument('--batch-size',     type=int,   default=4, dest='batch_size')
    p_tr.add_argument('--lr',             type=float, default=2e-4)
    p_tr.add_argument('--n-res-blocks',   type=int,   default=9, dest='n_res_blocks')
    p_tr.add_argument('--base-ch',        type=int,   default=64, dest='base_ch')
    p_tr.add_argument('--patience',       type=int,   default=40)
    p_tr.add_argument('--resume',         action='store_true')
    p_tr.add_argument('--verbose',        action='store_true')
    p_tr.set_defaults(func=cmd_train)

    # ── transform ─────────────────────────────────────────────────────────────
    p_tf = sub.add_parser('transform', help='Transferir estilo a un MIDI')
    p_tf.add_argument('input')
    p_tf.add_argument('--backend',            default='symbolic', choices=['symbolic', 'neural'])
    p_tf.add_argument('--model',              default=None)
    p_tf.add_argument('--model-dir',          default=None, dest='model_dir')
    p_tf.add_argument('--direction',          default='AtoB', choices=['AtoB', 'BtoA'])
    p_tf.add_argument('--intensity',          type=float, default=1.0)
    p_tf.add_argument('--output',             default=None)
    p_tf.add_argument('--bars',               type=int,   default=None)
    p_tf.add_argument('--candidates',         type=int,   default=3)
    p_tf.add_argument('--no-percussion',      action='store_true', dest='no_percussion')
    p_tf.add_argument('--export-fingerprint', action='store_true', dest='export_fingerprint')
    p_tf.add_argument('--seed',               type=int,   default=42)
    p_tf.add_argument('--bpm',                type=float, default=120.0)
    p_tf.add_argument('--threshold',          type=float, default=None)
    p_tf.add_argument('--threshold-pct',      type=float, default=99.0, dest='threshold_pct')
    p_tf.add_argument('--palette',            default=None)
    p_tf.add_argument('--verbose',            action='store_true')
    p_tf.set_defaults(func=cmd_transform)

    # ── cycle ─────────────────────────────────────────────────────────────────
    p_cy = sub.add_parser('cycle', help='Verificar consistencia de ciclo A→B→A')
    p_cy.add_argument('input')
    p_cy.add_argument('--model',         required=True)
    p_cy.add_argument('--backend',       default='symbolic', choices=['symbolic', 'neural'])
    p_cy.add_argument('--direction',     default='AtoB', choices=['AtoB', 'BtoA'])
    p_cy.add_argument('--intensity',     type=float, default=1.0)
    p_cy.add_argument('--output-ab',     default=None, dest='output_ab')
    p_cy.add_argument('--output-aba',    default=None, dest='output_aba')
    p_cy.add_argument('--bars',          type=int,   default=None)
    p_cy.add_argument('--candidates',    type=int,   default=2)
    p_cy.add_argument('--no-percussion', action='store_true', dest='no_percussion')
    p_cy.add_argument('--seed',          type=int,   default=42)
    p_cy.add_argument('--verbose',       action='store_true')
    p_cy.set_defaults(func=cmd_cycle)

    # ── analyze ───────────────────────────────────────────────────────────────
    p_an = sub.add_parser('analyze', help='Posición de un MIDI en el espacio A/B')
    p_an.add_argument('input')
    p_an.add_argument('--model',   required=True)
    p_an.add_argument('--backend', default='symbolic', choices=['symbolic', 'neural'])
    p_an.add_argument('--verbose', action='store_true')
    p_an.set_defaults(func=cmd_analyze)

    # ── style-corpus ──────────────────────────────────────────────────────────
    p_sc = sub.add_parser('style-corpus', help='Estadísticas de densidad del generador')
    p_sc.add_argument('--input-dir',  required=True)
    p_sc.add_argument('--backend',    default='symbolic', choices=['symbolic', 'neural'])
    p_sc.add_argument('--model',      default=None)
    p_sc.add_argument('--model-dir',  default=None, dest='model_dir')
    p_sc.add_argument('--direction',  default='a2b', choices=['a2b', 'b2a'])
    p_sc.add_argument('--output',     default=None)
    p_sc.set_defaults(func=cmd_style_corpus)

    # ── round-trip ────────────────────────────────────────────────────────────
    p_rt = sub.add_parser('round-trip', help='MIDI → roll → MIDI sin modelo')
    p_rt.add_argument('--input',         required=True)
    p_rt.add_argument('--model-dir',     default=None, dest='model_dir')
    p_rt.add_argument('--resolution',    type=int, default=TICKS_PER_BAR)
    p_rt.add_argument('--disable-roles', nargs='+', choices=ROLES, default=[], dest='disable_roles')
    p_rt.add_argument('--output',        default='roundtrip.mid')
    p_rt.add_argument('--bpm',           type=float, default=120.0)
    p_rt.set_defaults(func=cmd_round_trip)

    # ── inspect ───────────────────────────────────────────────────────────────
    p_ins = sub.add_parser('inspect', help='.npz / curvas de pérdida / config')
    p_ins.add_argument('--what',      nargs='+', choices=['npz', 'loss_curve'],
                       default=['npz'])
    p_ins.add_argument('--data-dir',  default=None, dest='data_dir')
    p_ins.add_argument('--model-dir', default=None, dest='model_dir')
    p_ins.add_argument('--file',      default=None, dest='npz_file')
    p_ins.set_defaults(func=cmd_inspect)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
