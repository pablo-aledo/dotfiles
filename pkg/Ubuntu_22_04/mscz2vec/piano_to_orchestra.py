"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     PIANO TO ORCHESTRA  v2.0                                 ║
║          Capa frontal de análisis y voicing para orchestrator.py             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PROPÓSITO                                                                   ║
║  ─────────                                                                   ║
║  Recibe un MIDI de piano (melodía + armonía en 1 o 2 pistas) y genera:      ║
║    [1] MIDI multitrack con cuatro voces orquestales separadas:               ║
║          Melody / Counterpoint / Accompaniment / Bass                        ║
║    [2] Un fingerprint JSON por sección, compatibles con orchestrator.py      ║
║    [3] Llamada automática a orchestrator.py (opcional)                       ║
║                                                                              ║
║  PIPELINE INTERNO                                                            ║
║  ────────────────                                                            ║
║  piano.mid                                                                   ║
║    → MIDILoader          Carga notas, tempo, ticks_per_beat; hash MD5        ║
║    → MelodyExtractor     Separa melodía / bajo / voces interiores            ║
║    → HarmonicAnalyzer    Tonalidad, secciones, acordes, curva de tensión     ║
║    → VoicingEngine       Distribuye 4 voces por sección (3 backends)        ║
║    → CounterpointEngine  Genera voz de contrapunto real nota-a-nota          ║
║    → TrackSplitter       Construye las 4 pistas con material real            ║
║    → FingerprintGenerator  Serializa análisis a JSON por sección             ║
║    → (HumanReviewLoop)   Revisión interactiva de candidatos (opcional)       ║
║    → MIDIWriter          Escribe MIDI tipo-1 + fingerprints                  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  MÓDULOS Y FUNCIONALIDADES                                                   ║
║  ─────────────────────────                                                   ║
║                                                                              ║
║  MelodyExtractor                                                             ║
║    Separación melodía/bajo por análisis de contorno melódico real.           ║
║    Cada nota se puntúa por cinco factores combinados:                        ║
║      · Continuidad de contorno: penaliza saltos > octava entre ataques       ║
║        consecutivos; bonifica grado conjunto (≤ 2 semitones)                ║
║      · Peso métrico: tiempo 1 > tiempo 3 > tiempos 2,4 > subdivisiones      ║
║      · Velocidad relativa: la melodía tiende a sonar más fuerte              ║
║      · Duración: notas muy largas en el agudo = pedal, no melodía           ║
║      · Posición en cluster: nota más aguda del grupo tiene ventaja           ║
║    La nota más grave de cada instante siempre va al bajo. Las intermedias   ║
║    van a inner_notes y alimentan la pista Accompaniment.                     ║
║                                                                              ║
║  HarmonicAnalyzer                                                            ║
║    · Tonalidad global y local por Krumhansl-Schmuckler (correlación con      ║
║      perfiles de alturas mayor/menor)                                        ║
║    · Detección de secciones por distancia coseno entre vectores chroma       ║
║      en ventanas deslizantes de 2 compases (paso = 1 compás):               ║
║        – Suavizado gaussiano + find_peaks (scipy) para cortes naturales      ║
║        – --sections N fuerza exactamente N secciones                         ║
║        – --min-bars N descarta secciones menores de N compases              ║
║        – Cortes alineados al compás completo más cercano (múltiplo tpb×4)  ║
║    · Identificación de acorde por template matching con 13 tipos:            ║
║        maj min dim aug maj7 min7 dom7 dim7 hdim7 sus2 sus4 maj6 min6        ║
║    · Curva de tensión de 16 puntos por sección, combinando:                 ║
║        – Grado funcional (I=0.0 … VII=0.9, notas cromáticas=0.85)          ║
║        – Disonancia intervalar (ratio de intervalos disonantes en el acorde)║
║        – Registro (notas agudas = mayor tensión)                            ║
║        – Densidad (notas por tiempo)                                         ║
║    · Clasificación de arco emocional: neutral / rise / fall / arch / high   ║
║    · Cálculo de harmony_complexity (variedad de clases de altura, 0–1)      ║
║                                                                              ║
║  VoicingEngine — tres backends intercambiables                               ║
║    rules   Muestreo aleatorio de notas del acorde en rangos SATB +          ║
║            scoring por: espaciado de Rimski-Kórsakov (bajo-tenor ≥ 8ª,     ║
║            tenor-alto ≥ 4ª, alto-soprano ≥ 2ª), consonancias, duplicación  ║
║    greedy  Construcción voraz voz-a-voz desde el bajo. En tensión alta      ║
║            desplaza el registro hacia el agudo. Garantiza espaciado mínimo  ║
║    ml      Genera candidatos con 'rules' y los rankea con un                 ║
║            GradientBoostingClassifier entrenado desde preferences.jsonl.    ║
║            Features: intervalos entre voces, registro medio, tensión, arco  ║
║    Los tres backends devuelven listas (voices, score) ordenadas por calidad ║
║                                                                              ║
║  CounterpointEngine                                                          ║
║    Genera la voz de contrapunto (alto) sincronizada nota-a-nota con la      ║
║    melodía. Implementa contrapunto de primera especie adaptado:              ║
║      · Movimiento contrario a la melodía como primera prioridad             ║
║      · Prohibición de quintas y octavas paralelas con soprano               ║
║      · Resolución obligatoria de la sensible hacia la tónica                ║
║      · Preferencia por terceras y sextas (consonancias "cálidas")           ║
║      · Movimiento por grado conjunto; saltos > quinta penalizados           ║
║      · Rango estricto de alto: F3–D5 (MIDI 53–74)                          ║
║      · Sin unísonos con la melodía                                          ║
║    Verificado: sin quintas paralelas, ≥ 50% movimiento contrario,           ║
║    mayoría de intervalos consonantes.                                        ║
║                                                                              ║
║  HumanReviewLoop + PreferenceTrainer                                         ║
║    · --review muestra N candidatos por sección con nombre de notas,         ║
║      score y contexto armónico; registra la elección en preferences.jsonl   ║
║    · preferences.jsonl acumula: voicing elegido, candidatos, scores,        ║
║      tonalidad, acorde, arco, tensión, hash del MIDI, timestamp             ║
║    · --train entrena GradientBoostingClassifier (100 árboles, depth=3)     ║
║      con StandardScaler; guarda model + scaler en voicing_ranker.pkl        ║
║    · --stats muestra distribución de tonalidades, arcos, índices elegidos   ║
║      y tensión media acumulada                                               ║
║                                                                              ║
║  FingerprintGenerator                                                        ║
║    Produce un JSON por sección con el formato exacto que consume             ║
║    orchestrator.py:                                                          ║
║      meta: key_tonic, key_mode, tempo_bpm, n_bars,                         ║
║            emotional_arc, harmony_complexity, section_label                 ║
║      tension_curve: mean, peak, entry, exit, peak_bar                       ║
║      tension_curve_full: lista de 16 floats en [0, 1]                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INTEGRACIÓN CON ORCHESTRATOR.PY                                             ║
║  ───────────────────────────────                                             ║
║  piano_to_orchestra.py genera exactamente lo que orchestrator.py espera:    ║
║    · MIDI tipo-1 con pistas nombradas Melody/Counterpoint/Accompaniment/Bass ║
║    · Fingerprints con campos meta, tension_curve y tension_curve_full        ║
║  Con --run-orchestrator el pipeline completo es un solo comando.             ║
║  Sin él, el script imprime el comando exacto para ejecutar manualmente.      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  EJEMPLOS DE USO                                                             ║
║  ───────────────                                                             ║
║                                                                              ║
║  ① Uso básico — reglas, secciones automáticas:                              ║
║       python piano_to_orchestra.py pieza.mid                                 ║
║                                                                              ║
║  ② Voicing greedy, 4 secciones forzadas, mínimo 2 barras:                  ║
║       python piano_to_orchestra.py pieza.mid                                  ║
║           --voicing greedy --sections 4 --min-bars 2                        ║
║                                                                              ║
║  ③ Debug completo para inspeccionar el análisis armónico:                   ║
║       python piano_to_orchestra.py pieza.mid --debug --verbose               ║
║       # Muestra por sección: tonalidad local, acorde, arco,                 ║
║       #   tensión media, barras y voicing elegido                            ║
║                                                                              ║
║  ④ Pipeline completo → orchestrator.py con orquesta de cámara:             ║
║       python piano_to_orchestra.py pieza.mid                                  ║
║           --run-orchestrator --template chamber --library nucleus            ║
║       # Genera pieza_split.mid + fingerprints + llama a orchestrator.py     ║
║       # Resultado final: pieza_split_orquestado.mid listo para FL Studio    ║
║                                                                              ║
║  ⑤ Orquesta completa con biblioteca Metropolis:                             ║
║       python piano_to_orchestra.py pieza.mid                                  ║
║           --run-orchestrator --template full --library metropolis  ║
║           --sections 6 --min-bars 2 --voicing greedy                        ║
║                                                                              ║
║  ⑥ Acumular preferencias personales (loop de aprendizaje):                 ║
║       python piano_to_orchestra.py pieza.mid                                  ║
║           --candidates 4 --review --voicing greedy                          ║
║       # Para cada sección muestra 4 candidatos con notas SATB y score.      ║
║       # Elige con 1-4, Enter (= mejor), o 's' para saltar.                 ║
║       # Cada elección se guarda en preferences.jsonl.                       ║
║                                                                              ║
║  ⑦ Entrenar el ranker ML tras acumular ~10 preferencias:                   ║
║       python piano_to_orchestra.py --train                                   ║
║       # Entrena GradientBoostingClassifier sobre preferences.jsonl           ║
║       # Guarda model + scaler en voicing_ranker.pkl                         ║
║                                                                              ║
║  ⑧ Usar el ranker entrenado para voicing personalizado:                     ║
║       python piano_to_orchestra.py pieza.mid --voicing ml                   ║
║       # Genera candidatos con 'rules' y los rankea con tu modelo            ║
║       # Si voicing_ranker.pkl no existe, cae a 'rules' automáticamente      ║
║                                                                              ║
║  ⑨ Ver estadísticas de preferencias acumuladas:                             ║
║       python piano_to_orchestra.py --stats                                   ║
║       # Muestra: total de entradas, distribución de tonalidades,            ║
║       #   arcos emocionales, índices elegidos y tensión media               ║
║                                                                              ║
║  ⑩ Loop completo de experimentación (días sucesivos):                       ║
║       # Día 1: acumula preferencias con rules                                ║
║       python piano_to_orchestra.py pieza.mid --candidates 3 --review        ║
║       # Día 2: entrena el modelo                                             ║
║       python piano_to_orchestra.py --train                                   ║
║       # Día 3: usa el modelo para generar y comparar con greedy             ║
║       python piano_to_orchestra.py pieza.mid --voicing ml   -o ml_out       ║
║       python piano_to_orchestra.py pieza.mid --voicing greedy -o greedy_out ║
║       # Escucha ambos y sigue acumulando preferencias con --review           ║
║                                                                              ║
║  ⑪ Pasar fingerprints a orchestrator.py manualmente:                        ║
║       python piano_to_orchestra.py pieza.mid --sections 3                    ║
║       python orchestrator.py pieza_split.mid                                  ║
║           pieza_split.secA.fingerprint.json                                   ║
║           pieza_split.secB.fingerprint.json                                   ║
║           pieza_split.secC.fingerprint.json                                   ║
║           --template full --library nucleus --no-perc                        ║
║                                                                              ║
║  ⑫ Usar modelo y fingerprints personalizados:                               ║
║       python piano_to_orchestra.py pieza.mid                                  ║
║           --voicing ml --model mis_preferencias.pkl                           ║
║           --prefs mis_preferencias.jsonl                                      ║
║           --output pieza_v3 --sections 5 --min-bars 2                       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OPCIONES COMPLETAS                                                          ║
║  ──────────────────                                                          ║
║                                                                              ║
║  Entrada / salida                                                            ║
║    midi                 MIDI de piano de entrada (posicional)                ║
║    --output  BASE       Nombre base de salida (default: <input>_split)       ║
║                         Genera BASE.mid + BASE.secX.fingerprint.json        ║
║                                                                              ║
║  Análisis                                                                    ║
║    --sections  N|auto   Número exacto de secciones, o auto (default: auto)  ║
║    --min-bars  N        Mínimo de compases por sección (default: 2)         ║
║    --debug              Muestra análisis armónico detallado por sección      ║
║    --verbose            Muestra transposiciones, voicings, conteos de notas  ║
║                                                                              ║
║  Voicing                                                                     ║
║    --voicing   rules|greedy|ml   Backend de voicing (default: rules)        ║
║    --candidates  N      Generar N candidatos por sección (default: 1)       ║
║    --review             Revisión humana interactiva; activa --candidates ≥ 3 ║
║    --model     FILE     Archivo del modelo ML (default: voicing_ranker.pkl) ║
║    --prefs     FILE     Archivo de preferencias (default: preferences.jsonl) ║
║                                                                              ║
║  Integración con orchestrator.py                                             ║
║    --run-orchestrator   Llamar a orchestrator.py tras generar el split       ║
║    --template  X        Plantilla orquestal: chamber|full|strings_only      ║
║                           chamber      = cuerdas + maderas + 2 trompas      ║
║                           full         = orquesta completa                   ║
║                           strings_only = solo cuerdas                        ║
║    --library   X        Library de samples: nucleus|metropolis|generic       ║
║                           nucleus      = Audio Imperia Nucleus               ║
║                           metropolis   = Orchestral Tools Metropolis Ark 1   ║
║                           generic      = GM / sin keyswitches                ║
║    --orchestrator-path  Ruta a orchestrator.py (default: ./orchestrator.py) ║
║                                                                              ║
║  Modos especiales (sin MIDI de entrada)                                      ║
║    --train              Entrenar ranker ML desde preferences.jsonl y salir   ║
║    --stats              Mostrar estadísticas de preferencias y salir         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ARCHIVOS GENERADOS                                                          ║
║  ──────────────────                                                          ║
║    BASE.mid                      MIDI tipo-1, una pista por voz             ║
║    BASE.secA.fingerprint.json    Fingerprint sección A                      ║
║    BASE.secB.fingerprint.json    Fingerprint sección B  (uno por sección)   ║
║    preferences.jsonl             Preferencias acumuladas (append)           ║
║    voicing_ranker.pkl            Modelo ML entrenado (model + scaler)       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DEPENDENCIAS                                                                ║
║    pip install mido numpy scipy scikit-learn                                 ║
║      mido         lectura/escritura MIDI                                     ║
║      numpy        álgebra lineal, estadística                                ║
║      scipy        gaussian_filter1d, find_peaks (detección de secciones)    ║
║      scikit-learn GradientBoostingClassifier, StandardScaler (backend ml)   ║
║    scipy y scikit-learn son opcionales: el script funciona sin ellos         ║
║    (secciones por división uniforme, backend ml cae a rules)                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import argparse
import random
import copy
import hashlib
import subprocess
import pickle
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido"); sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy"); sys.exit(1)

try:
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES  = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
ENHARMONICS = {'Db':'C#','Eb':'D#','Fb':'E','Gb':'F#','Ab':'G#','Bb':'A#','Cb':'B'}

# Plantillas de acordes: (nombre, intervalos desde raíz)
CHORD_TEMPLATES = [
    ('maj',    [0, 4, 7]),
    ('min',    [0, 3, 7]),
    ('dim',    [0, 3, 6]),
    ('aug',    [0, 4, 8]),
    ('maj7',   [0, 4, 7, 11]),
    ('min7',   [0, 3, 7, 10]),
    ('dom7',   [0, 4, 7, 10]),
    ('dim7',   [0, 3, 6, 9]),
    ('hdim7',  [0, 3, 6, 10]),
    ('sus4',   [0, 5, 7]),
    ('sus2',   [0, 2, 7]),
    ('maj6',   [0, 4, 7, 9]),
    ('min6',   [0, 3, 7, 9]),
]

# Tonalidades y sus grados (para análisis funcional)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # natural

# Tensión por grado en tonalidad mayor (0=tónica, 6=sensible)
DEGREE_TENSION = {
    0: 0.0,   # I
    1: 0.6,   # II
    2: 0.4,   # III
    3: 0.3,   # IV
    4: 0.8,   # V
    5: 0.3,   # VI
    6: 0.9,   # VII
}

# Tensión por tipo de acorde
CHORD_TENSION = {
    'maj': 0.1, 'min': 0.2, 'dim': 0.7, 'aug': 0.6,
    'maj7': 0.2, 'min7': 0.3, 'dom7': 0.6,
    'dim7': 0.8, 'hdim7': 0.7,
    'sus4': 0.4, 'sus2': 0.3,
    'maj6': 0.2, 'min6': 0.4,
}

# Intervalos consonantes / disonantes
CONSONANCES = {0, 3, 4, 5, 7, 8, 9}   # unísono, 3ª, 4ª, 5ª, 6ª, 8ª
DISSONANCES = {1, 2, 6, 10, 11}         # 2ª, 7ª, tritono

# Reglas de espaciado mínimo entre voces (semitones) según registro
# Rimski-Kórsakov: graves más separados, agudos más juntos
MIN_SPACING = {
    'bass_tenor':  12,   # Entre bajo y tenor: al menos octava
    'tenor_alto':   5,   # Entre tenor y alto: al menos cuarta
    'alto_soprano': 2,   # Entre alto y soprano: al menos segunda
}


# ══════════════════════════════════════════════════════════════════════════════
#  MIDI LOADER
# ══════════════════════════════════════════════════════════════════════════════

class MIDILoader:
    """Carga un MIDI de piano y separa notas por mano/pista."""

    def __init__(self, path, verbose=False):
        self.path    = path
        self.verbose = verbose
        self.mid     = MidiFile(path)
        self.tpb     = self.mid.ticks_per_beat
        self.tempo   = 500000  # 120 BPM default

    def load(self):
        """
        Devuelve:
          all_notes: [(abs_tick, pitch, dur_ticks, vel, track_idx), ...]
          tempo: microsegundos por beat
          tpb: ticks per beat
        """
        all_notes = []

        for ti, track in enumerate(self.mid.tracks):
            abs_tick = 0
            note_ons = {}
            for msg in track:
                abs_tick += msg.time
                if msg.type == 'set_tempo':
                    self.tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    note_ons[msg.note] = (abs_tick, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in note_ons:
                        t_on, vel = note_ons.pop(msg.note)
                        dur = abs_tick - t_on
                        if dur > 0:
                            all_notes.append((t_on, msg.note, dur, vel, ti))

        all_notes.sort(key=lambda x: x[0])

        if self.verbose:
            pitches = [n[1] for n in all_notes]
            print(f"  Cargadas {len(all_notes)} notas | "
                  f"rango: {NOTE_NAMES[min(pitches)%12]}{min(pitches)//12-1} – "
                  f"{NOTE_NAMES[max(pitches)%12]}{max(pitches)//12-1}")

        return all_notes, self.tempo, self.tpb

    def midi_hash(self):
        with open(self.path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════════════════════════
#  MELODY EXTRACTOR  (mejora A)
#  Separación melodía/bajo por análisis de contorno melódico real.
#
#  Criterios para identificar la melodía:
#    1. Nota más aguda en cada instante (base)
#    2. Continuidad de contorno: penaliza saltos bruscos entre candidatos
#    3. Inter-Onset Interval (IOI): la melodía tiende a tener IOI más cortos
#       (más activa) o duración coherente, no notas pedal largas en el agudo
#    4. Peso métrico: notas en tiempos fuertes (pulso) ponderan más
#    5. Bajo: nota más grave en cada instante, con continuidad propia
# ══════════════════════════════════════════════════════════════════════════════

class MelodyExtractor:
    """
    Extrae melodía, bajo e interiores de un conjunto de notas simultáneas
    usando análisis de contorno melódico, IOI y peso métrico.
    """

    def __init__(self, tpb, beats_per_bar=4):
        self.tpb          = tpb
        self.beats_per_bar = beats_per_bar
        # Pesos métricos: tiempo 1 > tiempo 3 > tiempos 2,4 > subdivisiones
        self._beat_weights = {0: 1.0, 2: 0.7, 1: 0.4, 3: 0.4}

    def _metric_weight(self, abs_tick):
        """Peso métrico de un tick dentro del compás [0,1]."""
        beat_pos  = (abs_tick / self.tpb) % self.beats_per_bar
        beat_int  = int(beat_pos)
        sub_frac  = beat_pos - beat_int
        w = self._beat_weights.get(beat_int, 0.3)
        return w * (1.0 - sub_frac * 0.3)  # leve decaimiento dentro del tiempo

    def _ioi(self, notes_sorted):
        """
        Calcula el IOI (inter-onset interval) promedio de una lista de notas.
        Lista de (t, p, d, v, ti) ordenada por t.
        """
        if len(notes_sorted) < 2:
            return self.tpb * 4  # silencio largo → no es melodía activa
        iois = [notes_sorted[i+1][0] - notes_sorted[i][0]
                for i in range(len(notes_sorted)-1)]
        return float(np.mean(iois))

    def _contour_score(self, candidate_pitch, prev_pitch):
        """
        Penaliza saltos melódicos grandes (> octava) y bonifica grado conjunto.
        """
        if prev_pitch is None:
            return 1.0
        interval = abs(candidate_pitch - prev_pitch)
        if interval == 0:
            return 0.9   # unísono: repetición, aceptable
        elif interval <= 2:
            return 1.0   # grado conjunto: ideal
        elif interval <= 5:
            return 0.85  # tercera/cuarta
        elif interval <= 7:
            return 0.75  # quinta/sexta
        elif interval <= 12:
            return 0.55  # séptima/octava
        else:
            return 0.25  # salto grande: muy penalizado

    def extract(self, notes):
        """
        Separa notes en (melody, bass, inner).
        Cada grupo: lista de (abs_tick, pitch, dur, vel, track_idx).
        """
        if not notes:
            return [], [], []

        # Agrupar por onset (tick de ataque)
        time_groups = defaultdict(list)
        for n in notes:
            time_groups[n[0]].append(n)

        sorted_ticks = sorted(time_groups.keys())

        melody_notes = []
        bass_notes   = []
        inner_notes  = []

        prev_mel_pitch  = None
        prev_bass_pitch = None

        # Primera pasada: candidatos por grupo de tiempo
        for tick in sorted_ticks:
            group = sorted(time_groups[tick], key=lambda x: x[1])  # grave → agudo

            if len(group) == 1:
                n = group[0]
                # Nota sola: ¿es melodía o bajo?
                # Si está por encima de la mediana global → melodía; si no → bajo
                all_pitches = [nn[1] for nn in notes]
                median_p    = np.median(all_pitches)
                mw          = self._metric_weight(tick)
                cscore      = self._contour_score(n[1], prev_mel_pitch)
                if n[1] >= median_p and cscore * mw > 0.3:
                    melody_notes.append(n)
                    prev_mel_pitch = n[1]
                else:
                    bass_notes.append(n)
                    prev_bass_pitch = n[1]
                continue

            # Bajo: siempre la nota más grave
            bass_candidate = group[0]
            bass_notes.append(bass_candidate)
            if prev_bass_pitch is not None:
                # Continuidad de bajo: si el salto es > 2 octavas, re-anclar
                if abs(bass_candidate[1] - prev_bass_pitch) > 24:
                    prev_bass_pitch = bass_candidate[1]
            prev_bass_pitch = bass_candidate[1]

            # Melodía: escoger entre las notas agudas del grupo
            # Candidatos: todas las notas excepto el bajo
            upper = group[1:]  # ya ordenadas grave→agudo

            if len(upper) == 1:
                melody_notes.append(upper[0])
                prev_mel_pitch = upper[0][1]
                continue

            # Puntuar cada candidato agudo como melodía
            best_score  = -1
            best_note   = upper[-1]  # fallback: la más aguda

            for n in upper:
                pitch = n[1]
                vel   = n[3]
                dur   = n[2]

                # Factores
                mw      = self._metric_weight(tick)
                cscore  = self._contour_score(pitch, prev_mel_pitch)
                # Velocidad normalizada (la melodía suele sonar más fuerte)
                vel_norm = vel / 127.0
                # Duración: notas muy largas en el agudo son pedal, no melodía
                dur_beats = dur / self.tpb
                dur_score = 1.0 if dur_beats <= 2.0 else max(0.4, 1.0 - (dur_beats-2)*0.1)
                # Posición en el grupo: más aguda = más probable melodía
                pos_score = (upper.index(n) + 1) / len(upper)

                total = (cscore * 0.35 + mw * 0.25 + vel_norm * 0.20
                         + pos_score * 0.15 + dur_score * 0.05)

                if total > best_score:
                    best_score = total
                    best_note  = n

            melody_notes.append(best_note)
            prev_mel_pitch = best_note[1]

            # El resto → interiores
            for n in upper:
                if n is not best_note:
                    inner_notes.append(n)

        return melody_notes, bass_notes, inner_notes


# ══════════════════════════════════════════════════════════════════════════════
#  COUNTERPOINT ENGINE  (mejora B)
#  Genera una voz de contrapunto real para la pista Counterpoint.
#
#  Reglas implementadas:
#    1. Movimiento preferentemente contrario a la melodía
#    2. Evitar quintas y octavas paralelas con soprano
#    3. Resolver la sensible hacia la tónica cuando aparece
#    4. Preferir movimiento por grado conjunto (≤ 2 semitones)
#    5. Mantenerse en rango de alto (F3–D5, MIDI 53–74)
#    6. Notas siempre del acorde de la sección (consonancias armónicas)
# ══════════════════════════════════════════════════════════════════════════════

class CounterpointEngine:
    """
    Genera contrapunto de primera especie adaptado para voz de alto orquestal.
    Opera nota-a-nota sincronizado con la melodía.
    """

    ALTO_LO = 53   # F3
    ALTO_HI = 74   # D5

    # Intervalos prohibidos con soprano (clase de intervalo mod 12)
    PARALLEL_FORBIDDEN = {0, 7}   # octavas y quintas paralelas

    def __init__(self, tpb):
        self.tpb = tpb

    def _chord_pcs(self, section):
        """Clases de altura del acorde de la sección."""
        root      = section['chord_root']
        ctype     = section['chord_type']
        intervals = dict(CHORD_TEMPLATES).get(ctype, [0, 4, 7])
        return [(root + i) % 12 for i in intervals]

    def _scale_pcs(self, section):
        """Clases de altura de la escala de la sección."""
        root  = section['key_root']
        scale = MAJOR_SCALE if section['key_mode'] == 'major' else MINOR_SCALE
        return [(root + i) % 12 for i in scale]

    def _leading_tone(self, section):
        """Clase de altura de la sensible en la tonalidad."""
        root  = section['key_root']
        scale = MAJOR_SCALE if section['key_mode'] == 'major' else MINOR_SCALE
        # Sensible = 7º grado (índice 6 en escala mayor)
        return (root + scale[6]) % 12 if len(scale) > 6 else (root + 11) % 12

    def _tonic_pc(self, section):
        return section['key_root'] % 12

    def _candidates_for_tick(self, section, mel_pitch, prev_alto, prev_mel):
        """
        Genera lista de pitches candidatos para el alto en este instante.
        Ordena por coste (movimiento contrario, grado conjunto, consonancia).
        """
        chord_pcs = self._chord_pcs(section)
        scale_pcs = self._scale_pcs(section)
        leading   = self._leading_tone(section)
        tonic     = self._tonic_pc(section)

        # Movimiento de la melodía (si hay anterior)
        mel_direction = 0
        if prev_mel is not None:
            mel_direction = np.sign(mel_pitch - prev_mel)

        candidates = []
        for p in range(self.ALTO_LO, self.ALTO_HI + 1):
            pc = p % 12

            # Solo notas del acorde (o la escala si el acorde es pequeño)
            if pc not in chord_pcs and pc not in scale_pcs:
                continue

            cost = 0.0

            # 1. Movimiento contrario a melodía (deseable)
            if prev_alto is not None:
                alto_move = np.sign(p - prev_alto)
                if mel_direction != 0 and alto_move == -mel_direction:
                    cost -= 2.0   # contrario: muy bueno
                elif alto_move == 0:
                    cost -= 0.5   # nota tenida: aceptable
                elif alto_move == mel_direction:
                    cost += 1.5   # movimiento directo: malo

                # 2. Preferir grado conjunto
                interval = abs(p - prev_alto)
                if interval <= 2:
                    cost -= 1.0
                elif interval <= 4:
                    cost -= 0.3
                elif interval > 7:
                    cost += 1.0   # salto > quinta: penalizar

                # 3. Quintas/octavas paralelas con melodía
                if prev_mel is not None:
                    prev_interval = abs(prev_mel - prev_alto) % 12
                    curr_interval = abs(mel_pitch - p) % 12
                    if (prev_interval in self.PARALLEL_FORBIDDEN
                            and curr_interval == prev_interval
                            and alto_move == mel_direction):
                        cost += 5.0   # prohibido

            # 4. Resolver sensible hacia tónica
            if prev_alto is not None and (prev_alto % 12) == leading:
                if pc == tonic:
                    cost -= 3.0   # resolución correcta
                else:
                    cost += 2.0   # no resolver la sensible: malo

            # 5. Evitar unísono con melodía
            if p == mel_pitch:
                cost += 2.0

            # 6. Preferir terceras y sextas con soprano (consonancias agradables)
            iv = abs(mel_pitch - p) % 12
            if iv in {3, 4, 8, 9}:   # terceras y sextas
                cost -= 0.8
            elif iv in {0, 7}:         # octava/quinta: no tan interesante
                cost += 0.3

            # 7. Nota del acorde > nota de la escala
            if pc in chord_pcs:
                cost -= 0.5

            candidates.append((p, cost))

        candidates.sort(key=lambda x: x[1])
        return candidates

    def generate(self, melody_notes, section, s_tick, e_tick):
        """
        Genera notas de contrapunto sincronizadas con melody_notes.
        Devuelve lista de (abs_tick, pitch, dur_ticks, vel).
        """
        if not melody_notes:
            # Sin melodía: generar relleno armónico simple
            return self._harmonic_fill(section, s_tick, e_tick)

        result     = []
        prev_alto  = None
        prev_mel   = None

        for i, (t, mel_p, dur, vel, _) in enumerate(melody_notes):
            if t < s_tick or t >= e_tick:
                continue

            candidates = self._candidates_for_tick(section, mel_p, prev_alto, prev_mel)

            if not candidates:
                # Fallback: tercera por debajo de la melodía en el rango
                fb = max(self.ALTO_LO, min(self.ALTO_HI, mel_p - 4))
                chosen = fb
            else:
                chosen = candidates[0][0]

            # Dinámica: ligeramente más suave que la melodía
            alto_vel = max(40, int(vel * 0.80))

            result.append((t, chosen, dur, alto_vel))
            prev_alto = chosen
            prev_mel  = mel_p

        return result

    def _harmonic_fill(self, section, s_tick, e_tick):
        """Relleno armónico cuando no hay melodía de referencia."""
        chord_pcs = self._chord_pcs(section)
        tension   = section['tension_curve']['mean']
        # Elegir una nota del acorde en rango de alto
        target = 62  # D4 como centro
        best_p, best_d = target, 999
        for pc in chord_pcs:
            for oct in range(3, 6):
                p = (oct + 1) * 12 + pc
                if self.ALTO_LO <= p <= self.ALTO_HI and abs(p - target) < best_d:
                    best_d = abs(p - target)
                    best_p = p

        beats_per_note = 4 if tension < 0.5 else 2
        note_dur = int(beats_per_note * self.tpb)
        result   = []
        t = s_tick
        while t < e_tick:
            dur = min(note_dur, e_tick - t)
            result.append((t, best_p, dur, 60))
            t += note_dur
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  HARMONIC ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class HarmonicAnalyzer:
    """
    Analiza un conjunto de notas MIDI y detecta:
      - Tonalidad global (tónica + modo)
      - Segmentos/secciones
      - Acorde por segmento
      - Curva de tensión
      - Melodía vs bajos
    """

    def __init__(self, notes, tpb, tempo, n_sections='auto', min_bars=2,
                 verbose=False, debug=False):
        self.notes      = notes   # [(abs_tick, pitch, dur, vel, ti), ...]
        self.tpb        = tpb
        self.tempo      = tempo
        self.n_sections = n_sections
        self.min_bars   = min_bars
        self.verbose    = verbose
        self.debug      = debug

    # ── Utilidades ──────────────────────────────────────────────────────────

    def _ticks_to_beats(self, t): return t / self.tpb
    def _beats_to_ticks(self, b): return int(b * self.tpb)

    def _pitch_class(self, pitch): return pitch % 12

    def _chroma_vector(self, notes):
        """Vector de 12 dimensiones con duración acumulada por clase de altura."""
        chroma = np.zeros(12)
        for t, p, d, v, _ in notes:
            chroma[p % 12] += d * (v / 127.0)
        total = chroma.sum()
        return chroma / total if total > 0 else chroma

    # ── Detección de tonalidad (Krumhansl-Schmuckler simplificado) ──────────

    def detect_key(self, notes=None):
        """Devuelve (tónica_int, modo_str) con tónica en [0,11]."""
        if notes is None:
            notes = self.notes
        chroma = self._chroma_vector(notes)

        # Perfiles de Krumhansl-Schmuckler
        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

        best_r, best_root, best_mode = -np.inf, 0, 'major'
        for root in range(12):
            rotated = np.roll(chroma, -root)
            r_maj = np.corrcoef(rotated, major_profile)[0,1]
            r_min = np.corrcoef(rotated, minor_profile)[0,1]
            if r_maj > best_r:
                best_r, best_root, best_mode = r_maj, root, 'major'
            if r_min > best_r:
                best_r, best_root, best_mode = r_min, root, 'minor'

        return best_root, best_mode

    # ── Detección de secciones (mejora C) ───────────────────────────────────

    def detect_sections(self, min_bars=2):
        """
        Detecta secciones por cambios de densidad armónica.
        v2.0:
          - --sections N fuerza exactamente N secciones
          - Alineación de cortes a barras completas (múltiplos de tpb*4)
          - min_bars: mínimo de compases por sección (evita secciones triviales)
        Devuelve lista de (start_tick, end_tick, notes_in_section).
        """
        if not self.notes:
            return []

        total_ticks   = max(t + d for t, p, d, v, _ in self.notes)
        bar_ticks     = self.tpb * 4   # asumimos 4/4
        min_sec_ticks = bar_ticks * max(1, min_bars)

        window = bar_ticks * 2    # ventana de 2 compases
        step   = bar_ticks        # paso de 1 compás (alineado a barras)

        # Densidad y cambio armónico por ventana
        positions = []
        chromas   = []
        t = 0
        while t < total_ticks:
            win_notes = [(nt,p,d,v,ti) for nt,p,d,v,ti in self.notes
                         if nt >= t and nt < t + window]
            positions.append(t)
            chromas.append(self._chroma_vector(win_notes) if win_notes else np.zeros(12))
            t += step

        if len(chromas) < 2:
            return [(0, total_ticks, self.notes)]

        # Distancia coseno entre ventanas consecutivas
        diffs = []
        for i in range(1, len(chromas)):
            a, b = chromas[i-1], chromas[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0 and nb > 0:
                diffs.append(1 - np.dot(a,b) / (na*nb))
            else:
                diffs.append(0.0)

        diffs_arr = np.array(diffs)

        # Número de cortes objetivo
        if self.n_sections == 'auto':
            n_target = max(1, min(8, len(diffs_arr) // 4))
        else:
            n_target = max(1, int(self.n_sections) - 1)  # N secciones = N-1 cortes

        # Picos de cambio armónico, respetando distancia mínima entre cortes
        min_distance = max(1, min_sec_ticks // step)

        if SCIPY_OK and len(diffs_arr) > 3:
            smooth = gaussian_filter1d(diffs_arr, sigma=1.5)
            peaks, _ = find_peaks(smooth, distance=min_distance)
            if len(peaks) > n_target:
                peak_vals = smooth[peaks]
                top_idx   = np.argsort(peak_vals)[-n_target:]
                peaks     = sorted(peaks[top_idx])
            elif len(peaks) < n_target and len(diffs_arr) >= n_target:
                # Pocos picos naturales: forzar divisiones uniformes
                # pero alineadas a barras
                total_bars   = int(total_ticks / bar_ticks)
                bars_per_sec = max(min_bars, total_bars // (n_target + 1))
                peaks = [int(i * bars_per_sec * bar_ticks / step)
                         for i in range(1, n_target + 1)
                         if i * bars_per_sec < total_bars]
        else:
            total_bars   = int(total_ticks / bar_ticks)
            bars_per_sec = max(min_bars, total_bars // (n_target + 1))
            peaks = [int(i * bars_per_sec * bar_ticks / step)
                     for i in range(1, n_target + 1)
                     if i * bars_per_sec < total_bars]

        # Construir boundaries alineados a barras
        raw_boundaries = [positions[min(p + 1, len(positions)-1)] for p in peaks]
        # Alinear cada boundary al inicio del compás más cercano
        aligned = []
        for b in raw_boundaries:
            aligned_b = round(b / bar_ticks) * bar_ticks
            aligned_b = max(bar_ticks, min(total_ticks - bar_ticks, aligned_b))
            aligned.append(aligned_b)

        boundaries = sorted(set([0] + aligned + [total_ticks]))

        # Filtrar secciones demasiado cortas
        filtered = [boundaries[0]]
        for b in boundaries[1:]:
            if b - filtered[-1] >= min_sec_ticks or b == total_ticks:
                filtered.append(b)
        boundaries = filtered
        if boundaries[-1] != total_ticks:
            boundaries[-1] = total_ticks

        sections = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i+1]
            sec_notes = [(t,p,d,v,ti) for t,p,d,v,ti in self.notes if t >= s and t < e]
            sections.append((s, e, sec_notes))

        if self.verbose:
            print(f"  Secciones detectadas: {len(sections)} "
                  f"(objetivo={n_target+1}, min_bars={min_bars})")

        return sections

    # ── Identificación de acorde ─────────────────────────────────────────────

    def identify_chord(self, notes):
        """
        Identifica el acorde más probable en un conjunto de notas.
        Devuelve (root_int, chord_type_str, score).
        """
        if not notes:
            return 0, 'maj', 0.0

        chroma = self._chroma_vector(notes)
        best_score, best_root, best_type = -1, 0, 'maj'

        for root in range(12):
            for name, intervals in CHORD_TEMPLATES:
                score = sum(chroma[(root + i) % 12] for i in intervals)
                # Penalizar notas fuera del acorde
                chord_pcs = set((root + i) % 12 for i in intervals)
                penalty   = sum(chroma[pc] for pc in range(12) if pc not in chord_pcs) * 0.3
                score -= penalty
                if score > best_score:
                    best_score, best_root, best_type = score, root, name

        return best_root, best_type, best_score

    # ── Separación melodía / bajo ────────────────────────────────────────────

    def split_melody_bass(self, notes):
        """
        Separa notas en melodía, bajo e interiores.
        v2.0: delega en MelodyExtractor (análisis de contorno real).
        Devuelve (melody_notes, bass_notes, inner_notes).
        """
        if not notes:
            return [], [], []
        return MelodyExtractor(self.tpb).extract(notes)

    # ── Curva de tensión ─────────────────────────────────────────────────────

    def compute_tension_curve(self, notes, key_root, key_mode, n_points=16):
        """
        Calcula curva de tensión (lista de floats 0-1) para un conjunto de notas.
        Combina: disonancia intervalar, grado funcional, registro y densidad.
        """
        if not notes:
            return [0.5] * n_points

        total_dur = max(t + d for t,p,d,v,_ in notes) - min(t for t,p,d,v,_ in notes)
        if total_dur <= 0:
            return [0.5] * n_points

        start = min(t for t,p,d,v,_ in notes)
        curve = []

        scale = MAJOR_SCALE if key_mode == 'major' else MINOR_SCALE

        for i in range(n_points):
            pos_start = start + (i / n_points) * total_dur
            pos_end   = start + ((i+1) / n_points) * total_dur
            seg_notes = [(t,p,d,v,ti) for t,p,d,v,ti in notes
                         if t >= pos_start and t < pos_end]

            if not seg_notes:
                curve.append(curve[-1] if curve else 0.3)
                continue

            # 1. Tensión por grado funcional
            pitches = [p for _,p,_,_,_ in seg_notes]
            pcs     = [p % 12 for p in pitches]
            degree_t = []
            for pc in pcs:
                rel = (pc - key_root) % 12
                # Encontrar grado más cercano
                if rel in scale:
                    deg = scale.index(rel)
                    degree_t.append(DEGREE_TENSION.get(deg, 0.5))
                else:
                    degree_t.append(0.85)  # nota cromática = alta tensión
            tension_deg = np.mean(degree_t) if degree_t else 0.5

            # 2. Tensión por disonancia intervalar (todos los pares)
            interval_t = 0.5
            if len(pitches) >= 2:
                intervals = []
                for a in range(len(pitches)):
                    for b in range(a+1, len(pitches)):
                        intervals.append(abs(pitches[a]-pitches[b]) % 12)
                dissonant = sum(1 for iv in intervals if iv in DISSONANCES)
                interval_t = dissonant / len(intervals) if intervals else 0.5

            # 3. Tensión por registro (agudo = más tenso)
            mean_pitch = np.mean(pitches)
            register_t = np.clip((mean_pitch - 48) / 48, 0, 1)

            # 4. Densidad (notas por beat)
            beats_seg  = (pos_end - pos_start) / self.tpb
            density    = min(len(seg_notes) / max(beats_seg, 1) / 4, 1.0)

            # Combinar
            tension = (tension_deg * 0.4 + interval_t * 0.3 +
                       register_t * 0.15 + density * 0.15)
            curve.append(float(np.clip(tension, 0, 1)))

        return curve

    # ── Análisis completo ────────────────────────────────────────────────────

    def analyze(self):
        """
        Ejecuta el análisis completo.
        Devuelve lista de dicts por sección, con toda la info necesaria.
        """
        key_root, key_mode = self.detect_key()
        key_name = NOTE_NAMES[key_root]

        if self.verbose:
            print(f"  Tonalidad global: {key_name} {key_mode}")

        sections_raw = self.detect_sections(min_bars=self.min_bars)
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        results  = []

        for i, (s_tick, e_tick, sec_notes) in enumerate(sections_raw):
            label = letters[i] if i < 26 else letters[i//26-1]+letters[i%26]

            # Tonalidad local
            loc_root, loc_mode = self.detect_key(sec_notes) if sec_notes else (key_root, key_mode)

            # Acorde principal
            chord_root, chord_type, chord_score = self.identify_chord(sec_notes)

            # Curva de tensión
            tension_full = self.compute_tension_curve(sec_notes, loc_root, loc_mode)
            t_mean  = float(np.mean(tension_full))
            t_peak  = float(np.max(tension_full))
            t_entry = tension_full[0]
            t_exit  = tension_full[-1]
            peak_bar = int(np.argmax(tension_full))

            # Arco emocional
            if t_peak > 0.7 and peak_bar > len(tension_full) // 3:
                arc = 'arch'
            elif t_exit > t_entry + 0.2:
                arc = 'rise'
            elif t_exit < t_entry - 0.2:
                arc = 'fall'
            elif t_mean > 0.6:
                arc = 'high'
            else:
                arc = 'neutral'

            # Separar melodía/bajo
            melody_notes, bass_notes, inner_notes = self.split_melody_bass(sec_notes)

            # Harmony complexity: variedad de PCs
            unique_pcs = len(set(p % 12 for _,p,_,_,_ in sec_notes)) if sec_notes else 0
            harm_complexity = min(unique_pcs / 8.0, 1.0)

            # Bars aproximados
            sec_beats = (e_tick - s_tick) / self.tpb
            n_bars    = max(1, int(round(sec_beats / 4)))

            bpm = round(60_000_000 / self.tempo, 1)

            result = {
                'label':      label,
                'start_tick': s_tick,
                'end_tick':   e_tick,
                'notes':      sec_notes,
                'melody_notes':  melody_notes,
                'bass_notes':    bass_notes,
                'inner_notes':   inner_notes,
                'key_root':   loc_root,
                'key_mode':   loc_mode,
                'key_name':   NOTE_NAMES[loc_root],
                'chord_root': chord_root,
                'chord_type': chord_type,
                'tension_full': tension_full,
                'tension_curve': {
                    'mean':     t_mean,
                    'peak':     t_peak,
                    'entry':    t_entry,
                    'exit':     t_exit,
                    'peak_bar': peak_bar,
                },
                'arc':        arc,
                'n_bars':     n_bars,
                'bpm':        bpm,
                'harm_complexity': harm_complexity,
            }
            results.append(result)

            if self.debug:
                ch_name = NOTE_NAMES[chord_root] + chord_type
                print(f"  [{label}] ticks {s_tick}-{e_tick} | "
                      f"key={NOTE_NAMES[loc_root]}{loc_mode[:3]} | "
                      f"chord={ch_name} | arc={arc} | "
                      f"tension_mean={t_mean:.2f} | bars={n_bars}")

        return key_root, key_mode, results


# ══════════════════════════════════════════════════════════════════════════════
#  VOICING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class VoicingEngine:
    """
    Genera voicings para cada sección.
    Soporta backends: rules, greedy, ml.
    """

    VOICE_RANGES = {
        'soprano': (60, 84),   # C4–C6
        'alto':    (53, 74),   # F3–D5
        'tenor':   (48, 69),   # C3–A4
        'bass':    (36, 60),   # C2–C4
    }

    def __init__(self, backend='rules', prefs_path='preferences.jsonl',
                 model_path='voicing_ranker.pkl', verbose=False):
        self.backend    = backend
        self.prefs_path = prefs_path
        self.model_path = model_path
        self.verbose    = verbose
        self.model      = None
        self.scaler     = None
        if backend == 'ml':
            self._load_model()

    # ── Scoring de voicings ─────────────────────────────────────────────────

    def _score_voicing(self, voices):
        """
        Puntúa un voicing de 4 voces [bajo, tenor, alto, soprano].
        Mayor score = mejor voicing.
        Aplica: Rimski spacing, paralelismos, sensible, duplicación.
        """
        if len(voices) < 2:
            return 0.0

        score = 1.0
        v = sorted(voices)  # bajo → soprano

        # 1. Regla de Rimski: espaciado mínimo entre voces graves
        if len(v) >= 2:
            gap_bt = v[1] - v[0]  # bajo-tenor
            if gap_bt < MIN_SPACING['bass_tenor']:
                score -= 0.4 * (MIN_SPACING['bass_tenor'] - gap_bt) / 12
        if len(v) >= 3:
            gap_ta = v[2] - v[1]  # tenor-alto
            if gap_ta < MIN_SPACING['tenor_alto']:
                score -= 0.2 * (MIN_SPACING['tenor_alto'] - gap_ta) / 5
        if len(v) >= 4:
            gap_as = v[3] - v[2]  # alto-soprano
            if gap_as < MIN_SPACING['alto_soprano']:
                score -= 0.1

        # 2. Penalizar intervalos muy amplios (> 2 octavas entre voces adyacentes)
        for i in range(len(v)-1):
            if v[i+1] - v[i] > 24:
                score -= 0.3

        # 3. Bonificar consonancias
        for i in range(len(v)):
            for j in range(i+1, len(v)):
                iv = (v[j] - v[i]) % 12
                if iv in CONSONANCES:
                    score += 0.05

        # 4. Penalizar duplicación de sensible (tono 11 en escala mayor = si en Do)
        # (simplificado: nota a semitono de la octava)
        pcs = [p % 12 for p in v]
        if pcs.count(pcs[-1]) > 1:  # soprano duplicada
            score -= 0.1

        return max(score, 0.0)

    def _voicing_features(self, voices, section):
        """Extrae features numéricas para el ranker ML."""
        v = sorted(voices)
        feats = []
        # Intervalos entre voces adyacentes
        for i in range(len(v)-1):
            feats.append(v[i+1] - v[i])
        while len(feats) < 3:
            feats.append(0)
        # Registro medio
        feats.append(np.mean(v))
        # Tensión de la sección
        feats.append(section['tension_curve']['mean'])
        # Arco (codificado)
        arc_map = {'neutral':0,'rise':1,'fall':2,'arch':3,'high':4}
        feats.append(arc_map.get(section['arc'], 0))
        return feats

    # ── Generación de candidatos ─────────────────────────────────────────────

    def _chord_pitches(self, root, chord_type):
        """Genera pitches de un acorde en distintas octavas."""
        intervals = dict(CHORD_TEMPLATES).get(chord_type, [0,4,7])
        pcs = [(root + i) % 12 for i in intervals]
        # Generar en 3 octavas (C2–C6)
        pitches = []
        for octave in range(2, 6):
            for pc in pcs:
                p = (octave + 1) * 12 + pc
                if 28 <= p <= 88:
                    pitches.append(p)
        return pitches

    def _generate_candidates_rules(self, section, n=5):
        """Genera N voicings usando reglas de espaciado."""
        root  = section['chord_root']
        ctype = section['chord_type']
        pitches = self._chord_pitches(root, ctype)

        bass_range     = [p for p in pitches if 36 <= p <= 60]
        tenor_range    = [p for p in pitches if 48 <= p <= 69]
        alto_range     = [p for p in pitches if 53 <= p <= 74]
        soprano_range  = [p for p in pitches if 60 <= p <= 84]

        if not bass_range:    bass_range = [pitches[0]] if pitches else [48]
        if not soprano_range: soprano_range = [pitches[-1]] if pitches else [72]
        if not tenor_range:   tenor_range = bass_range
        if not alto_range:    alto_range  = soprano_range

        candidates = []
        attempts   = 0
        while len(candidates) < n * 3 and attempts < 200:
            attempts += 1
            b = random.choice(bass_range)
            t = random.choice(tenor_range)
            a = random.choice(alto_range)
            s = random.choice(soprano_range)
            voices = sorted([b, t, a, s])
            # Filtro básico: no cruce de voces
            if voices[0] < voices[1] < voices[2] < voices[3]:
                score = self._score_voicing(voices)
                candidates.append((voices, score))

        # Ordenar y devolver top N únicos
        candidates.sort(key=lambda x: -x[1])
        seen = set()
        unique = []
        for v, sc in candidates:
            key = tuple(v)
            if key not in seen:
                seen.add(key)
                unique.append((v, sc))
            if len(unique) >= n:
                break

        # Completar si faltan
        while len(unique) < n:
            fallback = [48, 55, 62, 67]
            unique.append((fallback, 0.5))

        return unique

    def _generate_candidates_greedy(self, section, n=5):
        """
        Genera candidatos por optimización voraz:
        fija el bajo, luego elige tenor, alto, soprano minimizando coste.
        """
        root  = section['chord_root']
        ctype = section['chord_type']
        pitches = self._chord_pitches(root, ctype)
        tension = section['tension_curve']['mean']

        # En tensión alta → voicings más abiertos y agudos
        register_offset = int(tension * 12)

        bass_candidates = sorted([p for p in pitches if 36 <= p+register_offset//2 <= 60])
        if not bass_candidates:
            bass_candidates = [48]

        candidates = []
        for bass in bass_candidates[:3]:
            # Tenor: al menos una octava sobre el bajo, mismo PC o quinta
            tenor_options = [p for p in pitches
                             if p > bass + MIN_SPACING['bass_tenor']
                             and p <= bass + 24
                             and 48 <= p <= 69]
            if not tenor_options:
                tenor_options = [bass + 12]
            for tenor in tenor_options[:2]:
                alto_options = [p for p in pitches
                                if p > tenor + MIN_SPACING['tenor_alto']
                                and p <= tenor + 16
                                and 53 <= p <= 74]
                if not alto_options:
                    alto_options = [tenor + 7]
                for alto in alto_options[:2]:
                    sop_options = [p for p in pitches
                                   if p > alto + MIN_SPACING['alto_soprano']
                                   and p <= alto + 14
                                   and 60 <= p <= 84]
                    if not sop_options:
                        sop_options = [alto + 5]
                    for sop in sop_options[:2]:
                        v = [bass, tenor, alto, sop]
                        sc = self._score_voicing(v)
                        candidates.append((v, sc))

        candidates.sort(key=lambda x: -x[1])
        seen = set()
        unique = []
        for v, sc in candidates:
            k = tuple(sorted(v))
            if k not in seen:
                seen.add(k)
                unique.append((v, sc))
            if len(unique) >= n:
                break

        while len(unique) < n:
            unique.append(([48, 55, 62, 67], 0.5))

        return unique

    def _load_model(self):
        if os.path.exists(self.model_path) and SKLEARN_OK:
            try:
                with open(self.model_path, 'rb') as f:
                    saved = pickle.load(f)
                self.model  = saved['model']
                self.scaler = saved['scaler']
                print(f"  Modelo ML cargado: {self.model_path}")
            except Exception as e:
                print(f"  ⚠ No se pudo cargar el modelo ML: {e}")
                self.backend = 'rules'
        else:
            print(f"  ⚠ Modelo ML no encontrado ({self.model_path}). Usando 'rules'.")
            self.backend = 'rules'

    def _generate_candidates_ml(self, section, n=5):
        """Genera candidatos con rules y los rankea con el modelo ML."""
        candidates = self._generate_candidates_rules(section, n=n*3)
        if self.model is None:
            return candidates[:n]

        scored = []
        for v, _ in candidates:
            feats = self._voicing_features(v, section)
            feat_arr = self.scaler.transform([feats])
            prob = self.model.predict_proba(feat_arr)[0][1]  # prob de "elegido"
            scored.append((v, float(prob)))

        scored.sort(key=lambda x: -x[1])
        return scored[:n]

    # ── API pública ─────────────────────────────────────────────────────────

    def generate(self, section, n_candidates=1):
        """
        Genera voicings para una sección.
        Devuelve lista de (voices, score).
        """
        if self.backend == 'greedy':
            cands = self._generate_candidates_greedy(section, n=max(n_candidates, 5))
        elif self.backend == 'ml':
            cands = self._generate_candidates_ml(section, n=max(n_candidates, 5))
        else:  # rules
            cands = self._generate_candidates_rules(section, n=max(n_candidates, 5))

        return cands[:n_candidates] if n_candidates > 0 else cands


# ══════════════════════════════════════════════════════════════════════════════
#  TRACK SPLITTER
# ══════════════════════════════════════════════════════════════════════════════

class TrackSplitter:
    """
    Toma el análisis de secciones y los voicings elegidos,
    y construye las pistas Melody/Counterpoint/Accompaniment/Bass
    compatibles con orchestrator.py.
    """

    def __init__(self, tpb, verbose=False):
        self.tpb     = tpb
        self.verbose = verbose

    def build_tracks(self, sections_analysis, chosen_voicings):
        """
        sections_analysis: lista de dicts del HarmonicAnalyzer
        chosen_voicings:   lista de [bass, tenor, alto, soprano] por sección

        Devuelve dict:
          {'Melody': [(t,p,d,v),...], 'Counterpoint':..., 'Accompaniment':..., 'Bass':...}
        """
        tracks = {name: [] for name in ['Melody', 'Counterpoint', 'Accompaniment', 'Bass']}
        cp_engine = CounterpointEngine(self.tpb)

        for sec, voicing in zip(sections_analysis, chosen_voicings):
            s_tick = sec['start_tick']
            e_tick = sec['end_tick']
            melody_notes = sec['melody_notes']
            bass_notes   = sec['bass_notes']
            inner_notes  = sec['inner_notes']

            bass_pitch, tenor_pitch, alto_pitch, soprano_pitch = voicing

            # ── Melody: notas originales de melodía ajustadas al rango de soprano
            if melody_notes:
                for t, p, d, v, _ in melody_notes:
                    # Transponer al rango soprano (60–84) conservando el PC
                    new_p = p
                    while new_p < 60: new_p += 12
                    while new_p > 84: new_p -= 12
                    tracks['Melody'].append((t, new_p, d, v))
            else:
                dur = e_tick - s_tick
                tracks['Melody'].append((s_tick, soprano_pitch, dur, 80))

            # ── Counterpoint: contrapunto real con CounterpointEngine (mejora B)
            cp_notes = cp_engine.generate(melody_notes, sec, s_tick, e_tick)
            tracks['Counterpoint'].extend(cp_notes)

            # ── Accompaniment: notas interiores originales o relleno armónico
            if inner_notes:
                for t, p, d, v, _ in inner_notes:
                    # Rango de tenor (48–69)
                    new_p = p
                    while new_p < 48: new_p += 12
                    while new_p > 69: new_p -= 12
                    tracks['Accompaniment'].append((t, new_p, d, v))
            else:
                self._fill_voice(tracks['Accompaniment'], sec, tenor_pitch, s_tick, e_tick)

            # ── Bass: notas originales de bajo ajustadas al rango de bajo
            if bass_notes:
                for t, p, d, v, _ in bass_notes:
                    new_p = p
                    while new_p < 36: new_p += 12
                    while new_p > 60: new_p -= 12
                    tracks['Bass'].append((t, new_p, d, v))
            else:
                dur = e_tick - s_tick
                tracks['Bass'].append((s_tick, bass_pitch, dur, 75))

        # Ordenar todas las pistas por tiempo
        for name in tracks:
            tracks[name].sort(key=lambda x: x[0])

        return tracks

    def _fill_voice(self, track, section, base_pitch, s_tick, e_tick):
        """Rellena una voz interior con notas del acorde según tensión."""
        root   = section['chord_root']
        ctype  = section['chord_type']
        intervals = dict(CHORD_TEMPLATES).get(ctype, [0,4,7])
        pcs    = [(root + i) % 12 for i in intervals]
        tension = section['tension_curve']['mean']

        # Densidad según tensión: tensión alta → más notas
        beats_per_note = 4 if tension < 0.4 else (2 if tension < 0.7 else 1)
        note_dur = int(beats_per_note * self.tpb)

        t = s_tick
        while t < e_tick:
            # Elegir PC del acorde más cercano al base_pitch
            best_p = base_pitch
            best_d = 999
            for pc in pcs:
                for oct in range(2, 6):
                    p = (oct+1)*12 + pc
                    if abs(p - base_pitch) < best_d and 36 <= p <= 84:
                        best_d = abs(p - base_pitch)
                        best_p = p
            dur = min(note_dur, e_tick - t)
            vel = int(60 + tension * 40)
            track.append((t, best_p, dur, vel))
            t += note_dur


# ══════════════════════════════════════════════════════════════════════════════
#  FINGERPRINT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class FingerprintGenerator:
    """
    Genera fingerprints JSON por sección compatibles con orchestrator.py.
    Incluye todos los campos que orchestrator.py accede directamente:
      meta, tension_curve, tension_curve_full,
      stitching_hints, entry, exit
    """

    # Traducción acorde → grado romano (simplificado, tonalidad mayor)
    _MAJOR_ROMAN = {0:'I', 1:'bII', 2:'II', 3:'bIII', 4:'III',
                    5:'IV', 6:'bV', 7:'V', 8:'bVI', 9:'VI',
                    10:'bVII', 11:'VII'}
    _MINOR_ROMAN = {0:'i', 1:'bII', 2:'ii°', 3:'III', 4:'iv',
                    5:'iv', 6:'bVI', 7:'V', 8:'VI', 9:'VI',
                    10:'VII', 11:'vii°'}

    def _chord_roman(self, section):
        """Devuelve el grado romano del acorde principal de la sección."""
        key_root   = section['key_root']
        chord_root = section['chord_root']
        chord_type = section['chord_type']
        mode       = section['key_mode']
        rel        = (chord_root - key_root) % 12
        table      = self._MINOR_ROMAN if mode == 'minor' else self._MAJOR_ROMAN
        roman      = table.get(rel, 'I')
        # Añadir calidad al símbolo
        if chord_type in ('dim', 'dim7', 'hdim7') and '°' not in roman:
            roman += '°'
        elif chord_type in ('dom7', 'maj7', 'min7'):
            roman += '7'
        return roman

    def _openness(self, section):
        """
        Calcula 'openness': qué tan abierta/inconclusa queda la sección.
        Alta si la tensión de salida es mayor que la de entrada, o si el
        arco es 'rise'. Baja si termina en tónica con tensión descendente.
        """
        tc  = section['tension_curve']
        arc = section['arc']
        t_exit  = tc['exit']
        t_entry = tc['entry']
        # Apertura base desde tensión de salida
        openness = t_exit * 0.6
        # Arco ascendente = más abierto
        if arc == 'rise':
            openness += 0.3
        elif arc == 'fall':
            openness -= 0.2
        elif arc == 'arch':
            # Clímax ya pasó, resuelto
            openness -= 0.1
        # Si la tensión sube del inicio al final, más abierto
        if t_exit > t_entry + 0.15:
            openness += 0.15
        return float(max(0.0, min(1.0, openness)))

    def generate(self, section):
        """Devuelve un dict fingerprint completo para una sección."""
        tc          = section['tension_curve']
        chord_roman = self._chord_roman(section)
        openness    = self._openness(section)

        return {
            'meta': {
                'key_tonic':          section['key_name'],
                'key_mode':           section['key_mode'],
                'tempo_bpm':          section['bpm'],
                'n_bars':             section['n_bars'],
                'emotional_arc':      section['arc'],
                'harmony_complexity': section['harm_complexity'],
                'section_label':      section['label'],
                'syncopation':        0.3,   # valor neutro; sin análisis rítmico
            },
            'tension_curve': {
                'mean':     tc['mean'],
                'peak':     tc['peak'],
                'entry':    tc['entry'],
                'exit':     tc['exit'],
                'peak_bar': tc['peak_bar'],
            },
            'tension_curve_full': section['tension_full'],
            # Acorde de entrada y salida de la sección (para sugerencias de orquestación)
            'entry': {
                'chord_roman': chord_roman,
                'tension':     tc['entry'],
            },
            'exit': {
                'chord_roman': chord_roman,   # simplificado: mismo acorde
                'tension':     tc['exit'],
            },
            # Apertura hacia la siguiente sección (usado por generate_orchestral_percussion)
            'stitching_hints': {
                'openness':    openness,
                'arc':         section['arc'],
                'tension_exit': tc['exit'],
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
#  HUMAN REVIEW LOOP
# ══════════════════════════════════════════════════════════════════════════════

class HumanReviewLoop:
    """
    Presenta candidatos al usuario y registra su elección en preferences.jsonl.
    """

    def __init__(self, prefs_path='preferences.jsonl'):
        self.prefs_path = prefs_path

    def _voice_str(self, voices):
        return ' | '.join(
            f"{name}={NOTE_NAMES[p%12]}{p//12-1}"
            for name, p in zip(['B','T','A','S'], sorted(voices))
        )

    def choose(self, section, candidates, input_hash):
        """
        Muestra candidatos y pide elección.
        Devuelve (voices_elegidas, idx_elegido).
        """
        print(f"\n  ── Sección {section['label']} "
              f"[{NOTE_NAMES[section['key_root']]}{section['key_mode'][:3]} | "
              f"{NOTE_NAMES[section['chord_root']]}{section['chord_type']} | "
              f"arc={section['arc']} | tension={section['tension_curve']['mean']:.2f}] ──")

        for i, (v, sc) in enumerate(candidates):
            print(f"    [{i+1}] {self._voice_str(v)}  (score={sc:.3f})")

        while True:
            try:
                raw = input(f"    Elige [1-{len(candidates)}] (Enter=1, s=saltar): ").strip()
                if raw == '' or raw == '1':
                    chosen_idx = 0
                    break
                elif raw.lower() == 's':
                    return candidates[0][0], 0
                else:
                    idx = int(raw) - 1
                    if 0 <= idx < len(candidates):
                        chosen_idx = idx
                        break
            except (ValueError, KeyboardInterrupt):
                chosen_idx = 0
                break

        chosen_voices = candidates[chosen_idx][0]
        self._save(section, candidates, chosen_idx, input_hash)
        return chosen_voices, chosen_idx

    def _save(self, section, candidates, chosen_idx, input_hash):
        entry = {
            'input_hash': input_hash,
            'section':    section['label'],
            'key':        NOTE_NAMES[section['key_root']] + section['key_mode'][:3],
            'chord':      NOTE_NAMES[section['chord_root']] + section['chord_type'],
            'arc':        section['arc'],
            'tension':    section['tension_curve']['mean'],
            'candidates': [v for v,_ in candidates],
            'scores':     [sc for _,sc in candidates],
            'chosen_idx': chosen_idx,
            'chosen':     candidates[chosen_idx][0],
            'timestamp':  datetime.now().isoformat(),
        }
        with open(self.prefs_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')


# ══════════════════════════════════════════════════════════════════════════════
#  PREFERENCE TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class PreferenceTrainer:
    """
    Entrena un ranker (GradientBoosting) desde preferences.jsonl.
    """

    def __init__(self, prefs_path='preferences.jsonl',
                 model_path='voicing_ranker.pkl'):
        self.prefs_path = prefs_path
        self.model_path = model_path

    def load_preferences(self):
        if not os.path.exists(self.prefs_path):
            return []
        prefs = []
        with open(self.prefs_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    prefs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
        return prefs

    def _extract_features(self, voicing, pref):
        """Extrae features de un voicing en contexto."""
        v = sorted(voicing)
        feats = []
        for i in range(len(v)-1):
            feats.append(v[i+1] - v[i])
        while len(feats) < 3:
            feats.append(0)
        feats.append(np.mean(v))
        feats.append(pref.get('tension', 0.5))
        arc_map = {'neutral':0,'rise':1,'fall':2,'arch':3,'high':4}
        feats.append(arc_map.get(pref.get('arc','neutral'), 0))
        return feats

    def train(self):
        if not SKLEARN_OK:
            print("ERROR: scikit-learn no disponible. pip install scikit-learn")
            return False

        prefs = self.load_preferences()
        if len(prefs) < 10:
            print(f"  Pocas preferencias ({len(prefs)}). Se recomiendan al menos 10.")
            if len(prefs) == 0:
                return False

        X, y = [], []
        for pref in prefs:
            chosen_idx = pref.get('chosen_idx', 0)
            for i, voicing in enumerate(pref.get('candidates', [])):
                feats = self._extract_features(voicing, pref)
                X.append(feats)
                y.append(1 if i == chosen_idx else 0)

        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                           random_state=42)
        model.fit(X_scaled, y)

        with open(self.model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': scaler}, f)

        pos_rate = y.mean()
        print(f"  Modelo entrenado: {len(prefs)} preferencias | "
              f"{len(X)} ejemplos | tasa_positiva={pos_rate:.2f}")
        print(f"  Guardado en: {self.model_path}")
        return True

    def stats(self):
        prefs = self.load_preferences()
        if not prefs:
            print("  No hay preferencias acumuladas.")
            return

        print(f"\n  ── Estadísticas de preferencias ──")
        print(f"  Total:      {len(prefs)} entradas")

        keys   = Counter(p.get('key','?') for p in prefs)
        arcs   = Counter(p.get('arc','?') for p in prefs)
        chosen = Counter(p.get('chosen_idx',0) for p in prefs)

        print(f"  Tonalidades: {dict(keys.most_common(5))}")
        print(f"  Arcos:       {dict(arcs)}")
        print(f"  Índice elegido: {dict(sorted(chosen.items()))}")

        tensions = [p.get('tension',0.5) for p in prefs]
        print(f"  Tensión media: {np.mean(tensions):.2f} ± {np.std(tensions):.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  MIDI WRITER
# ══════════════════════════════════════════════════════════════════════════════

class MIDIWriter:
    """
    Escribe el MIDI multitrack de salida (compatible con orchestrator.py)
    y los fingerprints JSON.
    """

    TRACK_PROGRAMS = {
        'Melody':        40,  # Violin
        'Counterpoint':  40,
        'Accompaniment': 42,  # Cello
        'Bass':          43,  # Contrabass
    }

    def __init__(self, tpb, tempo):
        self.tpb   = tpb
        self.tempo = tempo

    def write_midi(self, tracks_dict, sections_analysis, output_path):
        """
        Escribe MIDI tipo 1 con una pista por rol.
        Inserta marcadores de sección en la pista 0.
        """
        mid = MidiFile(type=1, ticks_per_beat=self.tpb)

        # Pista 0: tempo + marcadores de sección
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)
        tempo_track.name = 'Tempo'
        tempo_track.append(MetaMessage('set_tempo', tempo=self.tempo, time=0))

        # Marcadores de sección
        prev_tick = 0
        for sec in sections_analysis:
            t = sec['start_tick']
            delta = t - prev_tick
            tempo_track.append(MetaMessage('marker',
                                           text=f"[{sec['label']}]",
                                           time=delta))
            prev_tick = t

        # Pistas de instrumentos
        track_order = ['Melody', 'Counterpoint', 'Accompaniment', 'Bass']
        for ch, name in enumerate(track_order):
            notes = tracks_dict.get(name, [])
            track = MidiTrack()
            mid.tracks.append(track)
            track.name = name

            prog = self.TRACK_PROGRAMS.get(name, 40)
            track.append(Message('program_change', channel=ch,
                                 program=prog, time=0))

            # Convertir notas absolutas → delta
            events = []
            for t, p, d, v in notes:
                events.append((t,     'on',  p, v))
                events.append((t + d, 'off', p, 0))
            events.sort(key=lambda x: (x[0], 0 if x[1]=='off' else 1))

            prev = 0
            for tick, etype, pitch, vel in events:
                delta = tick - prev
                prev  = tick
                if etype == 'on':
                    track.append(Message('note_on',  channel=ch,
                                         note=pitch, velocity=vel, time=delta))
                else:
                    track.append(Message('note_off', channel=ch,
                                         note=pitch, velocity=0,  time=delta))

        mid.save(output_path)
        return output_path

    def write_fingerprints(self, fps_data, base_path):
        """Escribe un fingerprint JSON por sección."""
        paths = []
        for fp, label in fps_data:
            p = f"{base_path}.sec{label}.fingerprint.json"
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(fp, f, indent=2, ensure_ascii=False)
            paths.append(p)
        return paths


# ══════════════════════════════════════════════════════════════════════════════
#  INSTRUMENT ASSIGNER
#  Convierte las cuatro pistas de voz (Melody/Counterpoint/Accompaniment/Bass)
#  en un MIDI orquestal final con instrumentos reales, keyswitches, CC1/CC11
#  y percusión. No depende de orchestrator.py.
# ══════════════════════════════════════════════════════════════════════════════

# ── Rangos idiomáticos (MIDI note numbers) ───────────────────────────────────

INSTR_RANGES = {
    'violin1':    (55, 96),   # G3–C7
    'violin2':    (55, 91),   # G3–G6
    'viola':      (48, 84),   # C3–C6
    'cello':      (36, 76),   # C2–E5
    'contrabass': (28, 60),   # E1–C4
    'flute':      (60, 96),   # C4–C7
    'oboe':       (58, 91),   # Bb3–G6
    'clarinet':   (50, 94),   # D3–Bb6
    'bassoon':    (34, 75),   # Bb1–Eb5
    'horn':       (34, 77),   # Bb1–F5
    'trumpet':    (52, 82),   # E3–Bb5
    'trombone':   (34, 72),   # Bb1–C5
}

INSTR_SWEET = {
    'violin1':    (62, 88),
    'violin2':    (60, 84),
    'viola':      (55, 79),
    'cello':      (43, 72),
    'contrabass': (33, 52),
    'flute':      (65, 90),
    'oboe':       (62, 86),
    'clarinet':   (55, 86),
    'bassoon':    (38, 67),
    'horn':       (40, 70),
    'trumpet':    (56, 77),
    'trombone':   (40, 67),
}

INSTR_FAMILY = {
    'violin1': 'strings', 'violin2': 'strings', 'viola': 'strings',
    'cello': 'strings', 'contrabass': 'strings',
    'flute': 'winds', 'oboe': 'winds', 'clarinet': 'winds', 'bassoon': 'winds',
    'horn': 'brass', 'trumpet': 'brass', 'trombone': 'brass',
}

GM_PROGRAMS = {
    'violin1': 40, 'violin2': 40, 'viola': 41, 'cello': 42, 'contrabass': 43,
    'flute': 73, 'oboe': 68, 'clarinet': 71, 'bassoon': 70,
    'horn': 60, 'trumpet': 56, 'trombone': 57,
    'timpani': 47,
}

INSTR_DISPLAY = {
    'violin1': 'Violin I', 'violin2': 'Violin II', 'viola': 'Viola',
    'cello': 'Cello', 'contrabass': 'Contrabajo',
    'flute': 'Flauta', 'oboe': 'Oboe', 'clarinet': 'Clarinete', 'bassoon': 'Fagot',
    'horn': 'Trompa', 'trumpet': 'Trompeta', 'trombone': 'Trombón',
}

# ── Plantillas ────────────────────────────────────────────────────────────────
# source: pista de voz de la que toma el material
# role:   melody | counterpoint | accompaniment | bass | pad | melody_double | bass_double
# section_filter: lista de labels de sección; None = todas
# octave_shift: desplazamiento en octavas (entero)

ORCH_TEMPLATES = {
    'strings_only': [
        {'name': 'violin1',    'source': 'Melody',        'role': 'melody',        'ch': 0},
        {'name': 'violin2',    'source': 'Counterpoint',  'role': 'counterpoint',  'ch': 1},
        {'name': 'viola',      'source': 'Accompaniment', 'role': 'accompaniment', 'ch': 2},
        {'name': 'cello',      'source': 'Bass',          'role': 'bass',          'ch': 3},
        {'name': 'contrabass', 'source': 'Bass',          'role': 'bass',          'ch': 4, 'octave_shift': -1},
    ],
    'chamber': [
        {'name': 'violin1',    'source': 'Melody',        'role': 'melody',        'ch': 0},
        {'name': 'violin2',    'source': 'Counterpoint',  'role': 'counterpoint',  'ch': 1},
        {'name': 'viola',      'source': 'Accompaniment', 'role': 'accompaniment', 'ch': 2},
        {'name': 'cello',      'source': 'Bass',          'role': 'bass',          'ch': 3},
        {'name': 'contrabass', 'source': 'Bass',          'role': 'bass',          'ch': 4, 'octave_shift': -1},
        {'name': 'oboe',       'source': 'Melody',        'role': 'melody_double', 'ch': 5, 'section_filter': 'first_last'},
        {'name': 'clarinet',   'source': 'Counterpoint',  'role': 'counterpoint',  'ch': 6, 'section_filter': 'middle'},
        {'name': 'bassoon',    'source': 'Bass',          'role': 'bass_double',   'ch': 7, 'section_filter': 'middle'},
        {'name': 'horn',       'source': 'Accompaniment', 'role': 'pad',           'ch': 8, 'section_filter': 'climax'},
    ],
    'full': [
        {'name': 'violin1',    'source': 'Melody',        'role': 'melody',        'ch': 0},
        {'name': 'violin2',    'source': 'Counterpoint',  'role': 'counterpoint',  'ch': 1},
        {'name': 'viola',      'source': 'Accompaniment', 'role': 'accompaniment', 'ch': 2},
        {'name': 'cello',      'source': 'Bass',          'role': 'bass',          'ch': 3},
        {'name': 'contrabass', 'source': 'Bass',          'role': 'bass',          'ch': 4, 'octave_shift': -1},
        {'name': 'flute',      'source': 'Melody',        'role': 'melody_double', 'ch': 5, 'section_filter': 'outer'},
        {'name': 'oboe',       'source': 'Melody',        'role': 'melody_double', 'ch': 6, 'section_filter': 'first'},
        {'name': 'clarinet',   'source': 'Counterpoint',  'role': 'counterpoint',  'ch': 7},
        {'name': 'bassoon',    'source': 'Bass',          'role': 'bass_double',   'ch': 8},
        {'name': 'horn',       'source': 'Accompaniment', 'role': 'pad',           'ch': 10, 'section_filter': 'high_tension'},
        {'name': 'trumpet',    'source': 'Melody',        'role': 'melody_double', 'ch': 11, 'section_filter': 'peak'},
        {'name': 'trombone',   'source': 'Accompaniment', 'role': 'pad',           'ch': 12, 'section_filter': 'high_tension'},
    ],
}

# ── Keyswitches ───────────────────────────────────────────────────────────────

KS_TABLES = {
    'nucleus': {
        'strings': {'legato': 21, 'sustain': 22, 'tremolo': 23,
                    'spiccato': 24, 'pizzicato': 25, 'portato': 22},
        'winds':   {'legato': 21, 'sustain': 22, 'staccato': 23, 'portato': 22},
        'brass':   {'legato': 21, 'sustain': 22, 'staccato': 23, 'portato': 22},
    },
    'metropolis': {
        'strings': {'legato': 21, 'sustain': 22, 'tremolo': 23,
                    'spiccato': 24, 'col_legno': 25, 'portato': 22, 'pizzicato': -1},
        'winds':   {'sustain': 21, 'staccato': 22, 'portato': 21, 'legato': -1},
        'brass':   {'legato': 21, 'sustain': 22, 'marcato_l': 23,
                    'marcato_s': 24, 'staccato': 25, 'swell': 26, 'portato': 22},
    },
    'generic': {
        'strings': {'legato': -1, 'sustain': -1, 'spiccato': -1,
                    'tremolo': -1, 'pizzicato': -1, 'portato': -1},
        'winds':   {'legato': -1, 'sustain': -1, 'staccato': -1, 'portato': -1},
        'brass':   {'legato': -1, 'sustain': -1, 'staccato': -1, 'portato': -1},
    },
}

# Umbrales de duración en beats para clasificar articulación
ART_THR = {'legato': 2.0, 'sustain': 0.75, 'portato': 0.4}

PERC_GM = {'kick': 36, 'snare': 38, 'crash': 49, 'sus_cym': 51, 'tamtam': 54}


class InstrumentAssigner:
    """
    Toma las cuatro pistas de voz y las distribuye a instrumentos reales,
    aplicando rangos idiomáticos, keyswitches, CC1/CC11 y percusión.
    Genera directamente el MIDI orquestal final.
    """

    def __init__(self, tpb, tempo, template='chamber', library='nucleus',
                 no_perc=False, no_ks=False, no_cc=False,
                 humanize=0.1, verbose=False):
        self.tpb      = tpb
        self.tempo    = tempo
        self.template = ORCH_TEMPLATES.get(template, ORCH_TEMPLATES['chamber'])
        self.library  = library
        self.ks_table = KS_TABLES.get(library, KS_TABLES['generic'])
        self.no_perc  = no_perc
        self.no_ks    = no_ks
        self.no_cc    = no_cc
        self.humanize = humanize
        self.verbose  = verbose

    # ── Utilidades ───────────────────────────────────────────────────────────

    def _beats(self, ticks):
        return ticks / self.tpb

    def _fit(self, pitch, name):
        """Transpone por octavas hasta que la nota cabe en el rango idiomático."""
        lo, hi   = INSTR_RANGES.get(name, (48, 84))
        sw_lo, sw_hi = INSTR_SWEET.get(name, (lo+5, hi-5))
        if lo <= pitch <= hi:
            return pitch
        for direction in [1, -1]:
            p = pitch
            for _ in range(4):
                p += 12 * direction
                if lo <= p <= hi:
                    if sw_lo <= p <= sw_hi:
                        return p
            p = pitch
            for _ in range(4):
                p -= 12 * direction
                if lo <= p <= hi:
                    return p
        return max(lo, min(hi, pitch))

    def _section_for(self, tick, sections):
        for s in sections:
            end = s['end_tick'] if s['end_tick'] else float('inf')
            if s['start_tick'] <= tick < end:
                return s
        return sections[-1]

    def _tension_at(self, tick, section):
        """Interpola la tensión dentro de una sección desde tension_full."""
        s, e   = section['start_tick'], section.get('end_tick')
        if not e:
            return section['tension_curve']['mean']
        dur    = max(e - s, 1)
        pos    = np.clip((tick - s) / dur, 0, 1)
        curve  = section.get('tension_full', [section['tension_curve']['mean']] * 16)
        idx    = min(int(pos * len(curve)), len(curve) - 1)
        return float(curve[idx])

    def _resolve_section_filter(self, filt, sections):
        """
        Convierte filtros semánticos a listas de labels concretos.
        Garantiza que siempre devuelve al menos 1 sección cuando hay material.
        """
        if filt is None:
            return None
        if isinstance(filt, list):
            return filt
        labels = [s['label'] for s in sections]
        n = len(labels)

        if filt == 'first':
            return [labels[0]]
        if filt == 'last':
            return [labels[-1]]
        if filt == 'first_last':
            return [labels[0], labels[-1]] if n > 1 else [labels[0]]
        if filt == 'outer':
            # Primera + última; con 3+ secciones añade también la penúltima
            result = {labels[0], labels[-1]}
            if n >= 3:
                result.add(labels[-2])
            return list(result)
        if filt == 'middle':
            return labels if n <= 2 else labels[1:-1]

        # 'high_tension': umbral bajo (0.45) para que siempre active algo
        if filt in ('high_tension', 'climax'):
            result = [s['label'] for s in sections
                      if s['tension_curve']['mean'] > 0.45
                      or s.get('arc', 'neutral') in ('arch', 'high', 'rise')]
            if not result:
                # Fallback: sección de mayor tensión
                result = [max(sections,
                              key=lambda s: s['tension_curve']['mean'])['label']]
            return result

        # 'peak': solo la sección de mayor tensión (para trompeta)
        if filt == 'peak':
            return [max(sections,
                        key=lambda s: s['tension_curve']['peak'])['label']]

        return None

    # ── Clasificación de articulación ────────────────────────────────────────

    def _articulation(self, note, prev_note, next_note, tension, family, role):
        t_on, pitch, dur_ticks, vel = note
        dur_beats = self._beats(dur_ticks)
        gap_beats = self._beats(t_on - (prev_note[0] + prev_note[2])) if prev_note else 1.0
        high = tension > 0.65
        low  = tension < 0.30

        if family == 'strings':
            if role in ('melody', 'melody_double') and dur_beats >= ART_THR['legato'] and gap_beats < 0.25:
                return 'legato'
            if role in ('melody', 'melody_double') and dur_beats >= ART_THR['sustain']:
                return 'sustain'
            if high and dur_beats < ART_THR['portato']:
                return 'spiccato'
            if role == 'accompaniment' and high and dur_beats >= 1.0:
                return 'tremolo'
            # Pizzicato: solo notas cortas en acompañamiento de baja tensión
            if role == 'accompaniment' and low and dur_beats < ART_THR['sustain']:
                return 'pizzicato'
            if dur_beats >= ART_THR['legato']:
                return 'legato'
            if dur_beats >= ART_THR['portato']:
                return 'portato'
            return 'spiccato'

        if family == 'winds':
            if dur_beats >= ART_THR['legato'] and gap_beats < 0.2:
                return 'legato'
            if dur_beats >= ART_THR['sustain']:
                return 'sustain'
            if dur_beats >= ART_THR['portato']:
                return 'portato'
            return 'staccato'

        if family == 'brass':
            if dur_beats >= ART_THR['legato']:
                return 'marcato_l' if (high and dur_beats >= 2.0) else 'legato'
            if dur_beats >= ART_THR['sustain']:
                return 'sustain'
            return 'staccato'

        return 'sustain'

    # ── Procesado de un instrumento ──────────────────────────────────────────

    def _process(self, raw_notes, instr_cfg, sections, fingerprints):
        """
        Aplica rango, KS, CC y humanize a las notas de un instrumento.
        Devuelve lista de eventos (abs_tick, type, *data).
        """
        name    = instr_cfg['name']
        role    = instr_cfg.get('role', 'melody')
        family  = INSTR_FAMILY.get(name, 'strings')
        octsh   = instr_cfg.get('octave_shift', 0)
        filt    = self._resolve_section_filter(
                      instr_cfg.get('section_filter'), sections)
        ks_map  = self.ks_table.get(family, {})

        # Filtrar por sección
        if filt is not None:
            keep = []
            for s in sections:
                if s['label'] in filt:
                    st = s['start_tick']
                    en = s['end_tick'] if s['end_tick'] else float('inf')
                    keep += [(t,p,d,v) for t,p,d,v in raw_notes if st <= t < en]
            raw_notes = sorted(keep, key=lambda x: x[0])

        if not raw_notes:
            return []

        # Desplazamiento de octava
        if octsh:
            raw_notes = [(t, p + octsh*12, d, v) for t,p,d,v in raw_notes]

        events     = []
        prev_art   = None
        prev_note  = None
        cc1_next   = raw_notes[0][0]
        last_cc1   = -1
        cc1_step   = self.tpb // 2   # cada corchea

        # CC inicial
        if not self.no_cc:
            init_sec = self._section_for(raw_notes[0][0], sections)
            init_t   = init_sec['tension_curve'].get('entry', 0.4)
            init_cc1 = int(np.clip(init_t * 100 + 15, 20, 115))
            events.append((raw_notes[0][0], 'cc', 1,  init_cc1))
            events.append((raw_notes[0][0], 'cc', 11, 100))
            last_cc1 = init_cc1

        for i, note in enumerate(raw_notes):
            t_on, pitch, dur_ticks, vel = note
            next_n = raw_notes[i+1] if i+1 < len(raw_notes) else None

            # Humanize
            if self.humanize > 0:
                jit = int(self.humanize * self.tpb * 0.04)
                if jit > 0:
                    t_on = max(0, t_on + random.randint(-jit, jit))

            # Rango idiomático
            pitch = self._fit(pitch, name)

            # Tensión
            sec     = self._section_for(t_on, sections)
            tension = self._tension_at(t_on, sec)

            # Articulación y KS
            art = self._articulation(note, prev_note, next_n, tension, family, role)
            if not self.no_ks and art != prev_art:
                ks = ks_map.get(art, ks_map.get('sustain', -1))
                if ks >= 0:
                    events.append((max(0, t_on - 5), 'ks', ks, 100))
                prev_art = art

            # CC1 periódico
            if not self.no_cc and t_on >= cc1_next:
                cc1 = int(np.clip(tension * 100 + 15, 15, 120))
                if last_cc1 >= 0:
                    cc1 = int(np.clip(cc1, last_cc1 - 15, last_cc1 + 15))
                if cc1 != last_cc1:
                    events.append((t_on, 'cc', 1, cc1))
                    last_cc1 = cc1
                cc1_next = t_on + cc1_step

            # CC11 swell en notas largas
            if not self.no_cc and self._beats(dur_ticks) >= 1.5:
                mid_t  = t_on + dur_ticks // 2
                end_t  = t_on + int(dur_ticks * 0.85)
                cc11_p = min(127, int(vel * 1.1 + tension * 20))
                cc11_e = max(40,  int(vel * 0.8))
                events.append((mid_t, 'cc', 11, cc11_p))
                events.append((end_t, 'cc', 11, cc11_e))

            # Velocidad efectiva
            if family in ('strings', 'brass') and art in ('legato', 'sustain', 'tremolo'):
                eff_vel = 80
            else:
                eff_vel = int(np.clip(vel * (0.7 + tension * 0.5), 30, 127))

            events.append((t_on, 'note', pitch, eff_vel, dur_ticks))
            prev_note = note

        return events

    # ── Percusión ────────────────────────────────────────────────────────────

    def _percussion(self, sections, fingerprints):
        if self.no_perc:
            return [], []

        tpb         = self.tpb
        ticks_bar   = tpb * 4
        timpani_evs = []
        gm_evs      = []
        pc_map      = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
                       'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}

        for sec_i, sec in enumerate(sections):
            fp      = fingerprints[sec_i] if sec_i < len(fingerprints) else {}
            arc     = sec.get('arc', 'neutral')
            tc      = sec['tension_curve']
            t_entry = tc.get('entry', 0.4)
            t_exit  = tc.get('exit', 0.4)
            t_peak  = tc.get('peak', 0.6)
            t_mean  = tc.get('mean', 0.5)
            n_bars  = sec.get('n_bars', 4)
            s_tick  = sec['start_tick']

            key_name = sec.get('key_name', 'C')
            key_pc   = pc_map.get(key_name, 0)
            # Tónica y dominante del timbal
            tonic_t = max(41, min(65, (3)*12 + key_pc))     # octava 2
            dom_t   = max(41, min(65, tonic_t + 7))

            for bar in range(n_bars):
                bar_tick = s_tick + bar * ticks_bar
                pos      = bar / max(n_bars - 1, 1)

                # Tensión interpolada por barra
                if arc == 'arch':
                    tension = t_entry*(1-pos) + t_exit*pos
                    tension = max(tension, 2*t_peak*pos*(1-pos))
                elif arc == 'rise':
                    tension = t_entry + (t_peak - t_entry) * pos
                elif arc == 'fall':
                    tension = t_peak - (t_peak - t_exit) * pos
                else:
                    tension = t_mean
                tension = float(np.clip(tension, 0, 1))
                high = tension > 0.65
                mid  = 0.35 <= tension <= 0.65
                low  = tension < 0.35

                # Timbal
                if arc in ('arch', 'rise') and high:
                    for beat in range(4):
                        t  = bar_tick + beat * tpb
                        pt = tonic_t if beat % 2 == 0 else dom_t
                        vt = int(np.clip(85 + tension*30, 85, 115))
                        timpani_evs.append((t, pt, tpb-5, vt))
                elif arc in ('arch', 'rise') and mid:
                    for beat in [0, 2]:
                        t  = bar_tick + beat * tpb
                        vt = int(np.clip(65 + tension*25, 60, 95))
                        timpani_evs.append((t, tonic_t, tpb-5, vt))
                elif arc == 'neutral' and mid and bar % 2 == 0:
                    vt = int(70 + tension*15)
                    timpani_evs.append((bar_tick, tonic_t, tpb-5, vt))
                elif arc == 'fall' and low and bar % 2 == 0:
                    vt = int(np.clip(45 + tension*20, 40, 70))
                    timpani_evs.append((bar_tick, tonic_t, tpb*2-5, vt))

                # Percusión GM
                if arc in ('rise', 'arch') and high:
                    for beat in range(4):
                        t = bar_tick + beat * tpb
                        if beat % 2 == 0:
                            gm_evs.append((t, PERC_GM['kick'],  tpb//2, int(np.clip(90+tension*25,90,115))))
                        else:
                            gm_evs.append((t, PERC_GM['crash'], tpb-5,  int(np.clip(75+tension*20,70,100))))
                elif arc in ('arch',) and mid:
                    gm_evs.append((bar_tick,          PERC_GM['kick'],  tpb//2, 75))
                    gm_evs.append((bar_tick + 2*tpb,  PERC_GM['snare'], tpb//2, 65))
                elif arc == 'fall' and low and bar % 4 == 0 and tension > 0.2:
                    gm_evs.append((bar_tick, PERC_GM['sus_cym'], tpb*4, 45))

                # Tam-tam: inicio de sección con alta apertura
                sh = fp.get('stitching_hints', {})
                if bar == 0 and sh.get('openness', 0) > 0.7:
                    gm_evs.append((bar_tick, PERC_GM['tamtam'], tpb*6, 85))

        return (sorted(timpani_evs, key=lambda x: x[0]),
                sorted(gm_evs,     key=lambda x: x[0]))

    # ── Escritura del MIDI final ──────────────────────────────────────────────

    def _events_to_track(self, events, channel, name, program):
        """Convierte lista de eventos a MidiTrack."""
        if not events:
            return None
        trk = MidiTrack()
        trk.name = name
        trk.append(MetaMessage('set_tempo', tempo=int(self.tempo), time=0))
        trk.append(Message('program_change', channel=channel,
                            program=program, time=0))

        raw = []
        for ev in events:
            tick, etype = ev[0], ev[1]
            if etype == 'note':
                _, _, pitch, vel, dur = ev
                pitch = max(0, min(127, pitch))
                vel   = max(1, min(127, vel))
                raw.append((tick,       3, Message('note_on',  channel=channel, note=pitch, velocity=vel, time=0)))
                raw.append((tick + dur, 0, Message('note_off', channel=channel, note=pitch, velocity=0,   time=0)))
            elif etype == 'cc':
                _, _, cc_num, val = ev
                raw.append((tick, 1, Message('control_change', channel=channel, control=cc_num, value=max(0,min(127,val)), time=0)))
            elif etype == 'ks':
                _, _, ks_note, vel = ev
                ks_note = max(0, min(127, ks_note))
                raw.append((tick,     2, Message('note_on',  channel=channel, note=ks_note, velocity=vel, time=0)))
                raw.append((tick + 2, 0, Message('note_off', channel=channel, note=ks_note, velocity=0,   time=0)))

        raw.sort(key=lambda x: (x[0], x[1]))
        prev = 0
        for abs_tick, _, msg in raw:
            delta = max(0, abs_tick - prev)
            prev  = abs_tick
            trk.append(msg.copy(time=delta))
        trk.append(MetaMessage('end_of_track', time=0))
        return trk

    def write(self, tracks_dict, sections, fingerprints, output_path):
        """
        Genera el MIDI orquestal final.
        tracks_dict: {'Melody':..., 'Counterpoint':..., 'Accompaniment':..., 'Bass':...}
        sections:    lista de dicts del HarmonicAnalyzer
        fingerprints: lista de dicts de FingerprintGenerator (uno por sección)
        """
        mid = MidiFile(type=1, ticks_per_beat=self.tpb)

        # Pista 0: tempo + marcadores
        t0 = MidiTrack(); mid.tracks.append(t0)
        t0.name = 'Tempo'
        t0.append(MetaMessage('set_tempo', tempo=int(self.tempo), time=0))
        prev = 0
        for sec in sections:
            t0.append(MetaMessage('marker', text=f"[{sec['label']}]",
                                  time=sec['start_tick'] - prev))
            prev = sec['start_tick']

        # Instrumentos
        n_notes = {}
        for cfg in self.template:
            name   = cfg['name']
            source = cfg['source']
            ch     = cfg['ch']

            raw = [(t,p,d,v) for t,p,d,v in tracks_dict.get(source, [])]
            events = self._process(raw, cfg, sections, fingerprints)
            if not events:
                continue

            prog = GM_PROGRAMS.get(name, 40)
            disp = INSTR_DISPLAY.get(name, name)
            trk  = self._events_to_track(events, ch, disp, prog)
            if trk:
                mid.tracks.append(trk)
                note_count = sum(1 for e in events if e[1] == 'note')
                ks_count   = sum(1 for e in events if e[1] == 'ks')
                cc_count   = sum(1 for e in events if e[1] == 'cc')
                n_notes[disp] = (note_count, ks_count, cc_count)
                if self.verbose:
                    print(f"  [{disp:<16}] {note_count:>4} notas  "
                          f"{ks_count:>3} KS  {cc_count:>4} CC")

        # Percusión
        timpani_evs, gm_evs = self._percussion(sections, fingerprints)
        if timpani_evs:
            trk = self._events_to_track(
                [(t,'note',p,v,d) for t,p,d,v in timpani_evs],
                15, 'Timbal', GM_PROGRAMS['timpani'])
            if trk: mid.tracks.append(trk)
        if gm_evs:
            trk = self._events_to_track(
                [(t,'note',n,v,d) for t,n,d,v in gm_evs],
                9, 'Percusión', 0)
            if trk: mid.tracks.append(trk)

        mid.save(output_path)
        return n_notes


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Piano MIDI → MIDI orquestal completo (sin orchestrator.py)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('midi', nargs='?', help='MIDI de piano de entrada')

    # Análisis
    parser.add_argument('--voicing',    default='rules',
                        choices=['rules','greedy','ml'],
                        help='Backend de voicing (default: rules)')
    parser.add_argument('--candidates', type=int, default=1,
                        help='Candidatos por sección para revisión (default: 1)')
    parser.add_argument('--review',     action='store_true',
                        help='Revisión humana interactiva de candidatos')
    parser.add_argument('--sections',   default='auto',
                        help='Número exacto de secciones (default: auto)')
    parser.add_argument('--min-bars',   type=int, default=2,
                        help='Mínimo de compases por sección (default: 2)')

    # Orquestación
    parser.add_argument('--template',   default='chamber',
                        choices=list(ORCH_TEMPLATES.keys()),
                        help='Plantilla orquestal (default: chamber)')
    parser.add_argument('--library',    default='nucleus',
                        choices=list(KS_TABLES.keys()),
                        help='Library de samples / keyswitches (default: nucleus)')
    parser.add_argument('--no-perc',    action='store_true',
                        help='Sin percusión orquestal')
    parser.add_argument('--no-ks',      action='store_true',
                        help='Sin keyswitches')
    parser.add_argument('--no-cc',      action='store_true',
                        help='Sin CC1/CC11')
    parser.add_argument('--humanize',   type=float, default=0.1,
                        help='Micro-jitter de timing 0.0–1.0 (default: 0.1)')

    # Salida
    parser.add_argument('--output',     default=None,
                        help='Nombre base de salida (default: <input>_orch)')
    parser.add_argument('--split',      action='store_true',
                        help='Exportar también el MIDI de voces (Melody/CP/Acc/Bass)')
    parser.add_argument('--fingerprints', action='store_true',
                        help='Exportar fingerprints JSON por sección')

    # ML y preferencias
    parser.add_argument('--prefs',      default='preferences.jsonl')
    parser.add_argument('--model',      default='voicing_ranker.pkl')
    parser.add_argument('--train',      action='store_true',
                        help='Entrenar ranker ML y salir')
    parser.add_argument('--stats',      action='store_true',
                        help='Mostrar estadísticas de preferencias y salir')

    # Debug
    parser.add_argument('--debug',      action='store_true')
    parser.add_argument('--verbose',    action='store_true')

    args = parser.parse_args()

    print("╔══════════════════════════════════════════╗")
    print("║     PIANO TO ORCHESTRA  v2.1             ║")
    print("╚══════════════════════════════════════════╝\n")

    # ── Modos especiales ──────────────────────────────────────────────────────

    if args.train:
        PreferenceTrainer(args.prefs, args.model).train()
        return
    if args.stats:
        PreferenceTrainer(args.prefs, args.model).stats()
        return
    if not args.midi:
        parser.print_help()
        return
    if not os.path.exists(args.midi):
        print(f"ERROR: No se encuentra {args.midi}"); sys.exit(1)

    base = args.output or Path(args.midi).stem + '_orch'

    print(f"  Entrada:   {args.midi}")
    print(f"  Salida:    {base}.mid")
    print(f"  Plantilla: {args.template}  |  Library: {args.library}")
    print(f"  Voicing:   {args.voicing}   |  Secciones: {args.sections}")

    # ── 1. Cargar MIDI ────────────────────────────────────────────────────────
    print(f"\n  [1/5] Cargando MIDI...")
    loader    = MIDILoader(args.midi, verbose=args.verbose)
    notes, tempo, tpb = loader.load()
    midi_hash = loader.midi_hash()
    print(f"  {len(notes)} notas | {round(60_000_000/tempo,1)} BPM | tpb={tpb}")

    # ── 2. Análisis armónico ──────────────────────────────────────────────────
    print(f"\n  [2/5] Analizando armonía...")
    analyzer = HarmonicAnalyzer(notes, tpb, tempo,
                                 n_sections=args.sections,
                                 min_bars=args.min_bars,
                                 verbose=args.verbose,
                                 debug=args.debug)
    key_root, key_mode, sections = analyzer.analyze()
    print(f"  Tonalidad: {NOTE_NAMES[key_root]} {key_mode} | {len(sections)} secciones")

    # ── 3. Voicing ───────────────────────────────────────────────────────────
    print(f"\n  [3/5] Generando voicings ({args.voicing})...")
    engine   = VoicingEngine(backend=args.voicing, prefs_path=args.prefs,
                              model_path=args.model, verbose=args.verbose)
    review   = HumanReviewLoop(args.prefs) if args.review else None
    n_cands  = max(args.candidates, 3 if args.review else 1)
    chosen_voicings = []

    for sec in sections:
        candidates = engine.generate(sec, n_candidates=n_cands)
        if args.review and review:
            chosen, _ = review.choose(sec, candidates, midi_hash)
        else:
            chosen = candidates[0][0]
        chosen_voicings.append(chosen)
        if args.verbose:
            v = chosen
            print(f"  [{sec['label']}] B={NOTE_NAMES[v[0]%12]}{v[0]//12-1} "
                  f"T={NOTE_NAMES[v[1]%12]}{v[1]//12-1} "
                  f"A={NOTE_NAMES[v[2]%12]}{v[2]//12-1} "
                  f"S={NOTE_NAMES[v[3]%12]}{v[3]//12-1}")

    # ── 4. Pistas de voces ───────────────────────────────────────────────────
    print(f"\n  [4/5] Construyendo voces...")
    splitter = TrackSplitter(tpb, verbose=args.verbose)
    tracks   = splitter.build_tracks(sections, chosen_voicings)
    for name, nts in tracks.items():
        print(f"  {name:<15} {len(nts):>4} notas")

    # ── 5. Orquestar y escribir ───────────────────────────────────────────────
    print(f"\n  [5/5] Orquestando → {base}.mid ...")
    fp_gen      = FingerprintGenerator()
    fingerprints = [fp_gen.generate(sec) for sec in sections]

    assigner = InstrumentAssigner(
        tpb=tpb, tempo=tempo,
        template=args.template,
        library=args.library,
        no_perc=args.no_perc,
        no_ks=args.no_ks,
        no_cc=args.no_cc,
        humanize=args.humanize,
        verbose=args.verbose,
    )

    out_orch = base + '.mid'
    n_notes  = assigner.write(tracks, sections, fingerprints, out_orch)
    print(f"\n  Instrumentos generados:")
    for disp, (nn, nks, ncc) in n_notes.items():
        print(f"  {disp:<18} {nn:>4} notas  {nks:>3} KS  {ncc:>4} CC")

    # ── Exportaciones opcionales ──────────────────────────────────────────────

    if args.split:
        out_split = base + '_voices.mid'
        writer    = MIDIWriter(tpb, tempo)
        writer.write_midi(tracks, sections, out_split)
        print(f"\n  Voces exportadas: {out_split}")

    if args.fingerprints:
        writer   = MIDIWriter(tpb, tempo)
        fps_data = [(fp, sec['label']) for fp, sec in zip(fingerprints, sections)]
        fp_paths = writer.write_fingerprints(fps_data, base)
        print(f"\n  Fingerprints exportados:")
        for p in fp_paths:
            print(f"    {p}")

    print(f"\n{'═'*55}")
    print(f"  Secciones:   {len(sections)}")
    instr_list = [INSTR_DISPLAY.get(c['name'], c['name'])
                  for c in ORCH_TEMPLATES.get(args.template, [])]
    print(f"  Instrumentos: {', '.join(instr_list)}")
    print(f"  Salida:      {out_orch}")
    print(f"{'═'*55}")

    print(f"\n  Reproducir:")
    print(f"  timidity -x 'soundfont ~/sf2/timbres_of_heaven.sf2' {out_orch}")


if __name__ == '__main__':
    main()

