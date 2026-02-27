#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     COUNTEREXAMPLE  v1.0                                     ║
║         Composición paramétrica por contraejemplo                            ║
║                                                                              ║
║  Dado un MIDI deseado (lo que te gusta), un MIDI rechazado (lo que no te    ║
║  gusta) y un MIDI de referencia (material genético de base), calcula el     ║
║  vector diferencial musical entre deseo y rechazo y usa ese diferencial     ║
║  como función de fitness dirigida en un algoritmo genético: los candidatos  ║
║  que se acercan al deseado y se alejan del rechazado son seleccionados.     ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] Vectorización de los tres MIDIs (17 dimensiones perceptuales)          ║
║  [2] Cálculo del vector diferencial: Δ = deseado − rechazado                ║
║  [3] Proyección del vector diferencial sobre el MIDI de referencia          ║
║  [4] Construcción de la función de fitness dirigida:                         ║
║        fitness = sim(candidato, deseado) − λ·sim(candidato, rechazado)      ║
║        + α·score_musical(candidato)                                          ║
║  [5] Evolución genética de la referencia guiada por ese fitness             ║
║  [6] Exportación de los mejores candidatos con informe de distancias        ║
║                                                                              ║
║  DIMENSIONES VECTORIALES (17D):                                              ║
║    [0]  pitch_mean        — altura media de las notas                        ║
║    [1]  pitch_std         — variabilidad de altura                           ║
║    [2]  pitch_range       — rango melódico total                             ║
║    [3]  interval_mean     — salto medio entre notas (abs)                    ║
║    [4]  interval_std      — variabilidad de los saltos                       ║
║    [5]  density           — notas por beat                                   ║
║    [6]  rhythm_variety    — diversidad de duraciones                         ║
║    [7]  syncopation       — ratio de síncopa                                 ║
║    [8]  velocity_mean     — dinámica media                                   ║
║    [9]  velocity_std      — variabilidad dinámica                            ║
║    [10] tension_mean      — tensión armónica media                           ║
║    [11] tension_peak      — pico de tensión                                  ║
║    [12] contour_asc       — proporción de intervalos ascendentes             ║
║    [13] large_leaps       — proporción de saltos > 5 semitonos              ║
║    [14] n_tracks          — número de pistas (textura)                       ║
║    [15] n_instruments     — riqueza instrumental                             ║
║    [16] climax_position   — posición relativa del clímax dinámico            ║
║                                                                              ║
║  USO:                                                                        ║
║    python counterexample.py deseado.mid rechazado.mid referencia.mid        ║
║    python counterexample.py good.mid bad.mid ref.mid --generations 8        ║
║    python counterexample.py good.mid bad.mid ref.mid --repulsion 1.5        ║
║    python counterexample.py good.mid bad.mid ref.mid --top 5 --listen       ║
║    python counterexample.py good.mid bad.mid ref.mid --report               ║
║    python counterexample.py good.mid bad.mid ref.mid --dry-run              ║
║    python counterexample.py good.mid bad.mid ref.mid \\                     ║
║        --generations 10 --population 20 --output-dir ./contra_out           ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --generations N     Generaciones del algoritmo genético (default: 6)     ║
║    --population N      Tamaño de la población (default: 12)                 ║
║    --elite N           Individuos de élite (default: 2)                     ║
║    --mutation-rate F   Tasa de mutación 0-1 (default: 0.35)                ║
║    --repulsion F       Peso λ de la repulsión del rechazado (default: 1.2)  ║
║    --attraction F      Peso de la atracción al deseado (default: 1.0)      ║
║    --musical-weight F  Peso α del score musical base (default: 0.3)         ║
║    --bars N            Compases por individuo (default: desde referencia)   ║
║    --top N             Mejores candidatos a exportar (default: 3)           ║
║    --output-dir DIR    Carpeta de salida (default: ./contra_out)            ║
║    --mixer PATH        Ruta a midi_dna_unified.py (default: auto)           ║
║    --report            Exportar informe JSON con análisis vectorial         ║
║    --plot              Visualizar espacio vectorial UMAP (requiere umap)    ║
║    --listen             Reproducir los mejores al final (requiere pygame)   ║
║    --play-seconds N    Segundos de reproducción por MIDI (default: 10)     ║
║    --seed N            Semilla aleatoria (default: 42)                      ║
║    --verbose           Informe detallado de cada generación                 ║
║    --dry-run           Solo calcular y mostrar el diferencial, sin generar  ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    contra_top01_fit0.832.mid  — mejor candidato                             ║
║    contra_top02_fit0.791.mid  — segundo mejor                               ║
║    contra_analysis.json       — análisis vectorial completo (con --report)  ║
║    contra_umap.png            — visualización (con --plot)                  ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy + evolver.py + midi_dna_unified.py               ║
║  Opcional: umap-learn, matplotlib (para --plot)                             ║
║            pygame (para --listen)                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import copy
import random
import shutil
import time
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCIAS DEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    sys.path.insert(0, SCRIPT_DIR)
    import midi_dna_unified as dna_mod
    from midi_dna_unified import UnifiedDNA, score_candidate
    DNA_OK = True
except ImportError:
    DNA_OK = False
    print("[AVISO] midi_dna_unified.py no encontrado. Algunas funciones estarán limitadas.")

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("ERROR: pip install mido")
    sys.exit(1)

try:
    import evolver as evolver_mod
    EVOLVER_OK = True
except ImportError:
    EVOLVER_OK = False
    print("[AVISO] evolver.py no encontrado. El modo genético no estará disponible.")


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZACIÓN  (17 dimensiones perceptuales, sin dependencias de music21)
# ══════════════════════════════════════════════════════════════════════════════

DIM_NAMES = [
    "pitch_mean",       # 0
    "pitch_std",        # 1
    "pitch_range",      # 2
    "interval_mean",    # 3
    "interval_std",     # 4
    "density",          # 5
    "rhythm_variety",   # 6
    "syncopation",      # 7
    "velocity_mean",    # 8
    "velocity_std",     # 9
    "tension_mean",     # 10
    "tension_peak",     # 11
    "contour_asc",      # 12
    "large_leaps",      # 13
    "n_tracks",         # 14
    "n_instruments",    # 15
    "climax_position",  # 16
]

N_DIMS = len(DIM_NAMES)


def _tension_of_interval(semitones):
    """Tensión perceptual aproximada de un intervalo (0=unísono, 1=muy tenso)."""
    TENSION = {0: 0.0, 1: 0.9, 2: 0.5, 3: 0.3, 4: 0.2, 5: 0.15,
               6: 1.0, 7: 0.1, 8: 0.35, 9: 0.25, 10: 0.6, 11: 0.8}
    return TENSION.get(semitones % 12, 0.5)


def vectorize_midi(file_path):
    """
    Extrae el vector perceptual 17D de un MIDI.
    Devuelve (vector np.ndarray[17], metadata dict) o (None, None) si falla.
    """
    try:
        mid = mido.MidiFile(file_path)
    except Exception as e:
        print(f"  [ERROR] No se pudo leer {file_path}: {e}")
        return None, None

    tpb = mid.ticks_per_beat or 480
    notes = []          # (start_beat, pitch, dur_beats, velocity)
    instruments = set()
    n_tracks = len(mid.tracks)

    for track in mid.tracks:
        current_program = 0
        abs_tick = 0
        active = {}  # (channel, pitch) → (start_beat, velocity)
        for msg in track:
            abs_tick += msg.time
            beat = abs_tick / tpb
            if msg.type == 'program_change':
                current_program = msg.program
                instruments.add(current_program)
            elif msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (beat, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start_b, vel = active.pop(key)
                    dur = beat - start_b
                    if dur < 0.01:
                        dur = 0.25
                    notes.append((start_b, msg.note, dur, vel))

    if not notes:
        print(f"  [AVISO] {file_path}: sin notas detectadas.")
        return None, None

    pitches   = np.array([n[1] for n in notes], dtype=float)
    durs      = np.array([n[2] for n in notes], dtype=float)
    vels      = np.array([n[3] for n in notes], dtype=float)
    starts    = np.array([n[0] for n in notes], dtype=float)

    # Duración total de la pieza en beats
    total_beats = max(starts + durs) if len(notes) > 0 else 1.0

    # Intervalos sucesivos
    sorted_idx = np.argsort(starts)
    sorted_pitches = pitches[sorted_idx]
    diffs = np.diff(sorted_pitches)
    abs_diffs = np.abs(diffs)

    # [0] pitch_mean
    pitch_mean = float(np.mean(pitches))
    # [1] pitch_std
    pitch_std = float(np.std(pitches))
    # [2] pitch_range
    pitch_range = float(np.max(pitches) - np.min(pitches)) / 48.0  # normalizado a 0-1 (4 octavas)
    # [3] interval_mean
    interval_mean = float(np.mean(abs_diffs)) / 12.0 if len(diffs) > 0 else 0.0
    # [4] interval_std
    interval_std = float(np.std(abs_diffs)) / 12.0 if len(diffs) > 0 else 0.0
    # [5] density: notas por beat
    density = min(1.0, len(notes) / max(total_beats, 1.0) / 4.0)
    # [6] rhythm_variety: diversidad de duraciones cuantizadas
    q_durs = [round(d * 4) / 4.0 for d in durs]
    rhythm_variety = min(1.0, len(set(q_durs)) / 8.0)
    # [7] syncopation: proporción de notas que empiezan en tiempos débiles
    beat_phases = starts % 1.0
    syncopation = float(np.mean((beat_phases > 0.1) & (beat_phases < 0.9)))
    # [8] velocity_mean
    velocity_mean = float(np.mean(vels)) / 127.0
    # [9] velocity_std
    velocity_std = float(np.std(vels)) / 64.0
    # [10] tension_mean: media de tensión interválica
    if len(diffs) > 0:
        tension_vals = [_tension_of_interval(int(abs(d))) for d in diffs]
        tension_mean = float(np.mean(tension_vals))
        tension_peak = float(np.max(tension_vals))
    else:
        tension_mean = 0.0
        tension_peak = 0.0
    # [12] contour_asc
    contour_asc = float(np.sum(diffs > 0) / len(diffs)) if len(diffs) > 0 else 0.5
    # [13] large_leaps
    large_leaps = float(np.sum(abs_diffs > 5) / len(diffs)) if len(diffs) > 0 else 0.0
    # [14] n_tracks normalizado
    n_tracks_norm = min(1.0, n_tracks / 8.0)
    # [15] n_instruments normalizado
    n_instr_norm = min(1.0, len(instruments) / 8.0)
    # [16] climax_position: posición relativa del beat con mayor velocidad
    if len(vels) > 1:
        climax_idx = int(np.argmax(vels))
        climax_position = float(starts[climax_idx]) / max(total_beats, 1.0)
    else:
        climax_position = 0.5

    vector = np.array([
        pitch_mean / 127.0,   # [0] normalizado
        pitch_std / 20.0,     # [1]
        pitch_range,          # [2]
        interval_mean,        # [3]
        interval_std,         # [4]
        density,              # [5]
        rhythm_variety,       # [6]
        syncopation,          # [7]
        velocity_mean,        # [8]
        velocity_std,         # [9]
        tension_mean,         # [10]
        tension_peak,         # [11]
        contour_asc,          # [12]
        large_leaps,          # [13]
        n_tracks_norm,        # [14]
        n_instr_norm,         # [15]
        climax_position,      # [16]
    ], dtype=float)

    metadata = {
        'n_notes': len(notes),
        'total_beats': float(total_beats),
        'n_tracks': n_tracks,
        'n_instruments': len(instruments),
        'tempo_bpm': _detect_tempo_from_midi(mid),
    }

    return vector, metadata


def _detect_tempo_from_midi(mid):
    """Extrae el BPM del mensaje set_tempo."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return round(60_000_000 / max(msg.tempo, 1), 1)
    return 120.0


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DEL DIFERENCIAL
# ══════════════════════════════════════════════════════════════════════════════

def compute_differential(vec_desired, vec_rejected):
    """
    Calcula el vector diferencial Δ = deseado − rechazado.
    Devuelve también una máscara de dimensiones significativas.
    """
    delta = vec_desired - vec_rejected
    # Dimensiones con diferencia significativa (> 0.1 en espacio normalizado)
    significant_mask = np.abs(delta) > 0.08
    return delta, significant_mask


def cosine_similarity(a, b):
    """Similitud coseno entre dos vectores."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def directional_fitness(vec_candidate, vec_desired, vec_rejected,
                         attraction=1.0, repulsion=1.2, musical_score=0.0,
                         musical_weight=0.3):
    """
    Función de fitness dirigida por contraejemplo.

    fitness = attraction · sim(candidato, deseado)
            − repulsion · sim(candidato, rechazado)
            + musical_weight · musical_score
    """
    sim_desired  = cosine_similarity(vec_candidate, vec_desired)
    sim_rejected = cosine_similarity(vec_candidate, vec_rejected)

    raw_fitness = (attraction  * sim_desired
                 - repulsion   * sim_rejected
                 + musical_weight * musical_score)

    # Normalizar a [0, 1] aproximadamente
    # El rango teórico es [-repulsion, attraction + musical_weight]
    max_val = attraction + musical_weight
    min_val = -repulsion
    normalized = (raw_fitness - min_val) / max((max_val - min_val), 1e-9)
    return float(np.clip(normalized, 0.0, 1.0)), sim_desired, sim_rejected


def describe_differential(delta, significant_mask, verbose=True):
    """Genera un informe legible del vector diferencial."""
    lines = []
    lines.append("\n  ── ANÁLISIS DEL DIFERENCIAL ─────────────────────────────────────")
    lines.append(f"  Dimensiones totales  : {N_DIMS}")
    lines.append(f"  Dimensiones activas  : {int(significant_mask.sum())}  (|Δ| > 0.08)")
    lines.append("")
    lines.append("  Dimensión               Δ       Dirección")
    lines.append("  " + "─" * 52)

    # Ordenar por magnitud descendente
    order = np.argsort(-np.abs(delta))
    for i in order:
        if not significant_mask[i]:
            continue
        d = delta[i]
        direction = "▲ más" if d > 0 else "▼ menos"
        bar = "█" * int(abs(d) * 20)
        lines.append(f"  {DIM_NAMES[i]:<22} {d:+.3f}   {direction} en el deseado  {bar}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR GENÉTICO CON FITNESS DIRIGIDO
# ══════════════════════════════════════════════════════════════════════════════

def _score_midi_for_fitness(midi_path, vec_desired, vec_rejected,
                             attraction, repulsion, musical_weight,
                             key_obj=None):
    """
    Vectoriza un MIDI generado y calcula su fitness dirigido.
    Combina el score musical de evolver con la distancia vectorial.
    """
    vec, meta = vectorize_midi(midi_path)
    if vec is None:
        return 0.0, None, 0.0, 0.0

    # Score musical base desde midi_dna_unified (si disponible)
    musical_sc = 0.0
    if DNA_OK and key_obj:
        try:
            mid = mido.MidiFile(midi_path)
            tpb = mid.ticks_per_beat or 480
            melody_notes = []
            for track in mid.tracks:
                abs_t = 0
                active = {}
                for msg in track:
                    abs_t += msg.time
                    b = abs_t / tpb
                    if msg.type == 'note_on' and msg.velocity > 0:
                        active[(msg.channel, msg.note)] = (b, msg.velocity)
                    elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                        k = (msg.channel, msg.note)
                        if k in active:
                            s, v = active.pop(k)
                            if msg.channel == 0:
                                melody_notes.append((s, msg.note, b - s, v))
            if melody_notes:
                musical_sc = float(score_candidate(melody_notes, [], key_obj))
        except Exception:
            pass

    fitness, sim_d, sim_r = directional_fitness(
        vec, vec_desired, vec_rejected,
        attraction=attraction, repulsion=repulsion,
        musical_score=musical_sc, musical_weight=musical_weight
    )
    return fitness, vec, sim_d, sim_r


class CounterexampleEvolver:
    """
    Motor de evolución genética guiado por la función de fitness por contraejemplo.
    Extiende la lógica de evolver.py inyectando la función de fitness dirigida.
    """
    def __init__(self, source_dnas, params_base, vec_desired, vec_rejected,
                 output_dir, mixer_script,
                 population_size=12, elite_size=2,
                 mutation_rate=0.35, crossover_rate=0.6,
                 n_generations=6, top_n=3,
                 attraction=1.0, repulsion=1.2, musical_weight=0.3,
                 verbose=False):

        if not EVOLVER_OK:
            raise ImportError("evolver.py no encontrado. Colócalo en el mismo directorio.")

        self.source_dnas    = source_dnas
        self.params_base    = params_base
        self.vec_desired    = vec_desired
        self.vec_rejected   = vec_rejected
        self.output_dir     = output_dir
        self.mixer_script   = mixer_script
        self.pop_size       = population_size
        self.elite_size     = elite_size
        self.mut_rate       = mutation_rate
        self.cross_rate     = crossover_rate
        self.n_gen          = n_generations
        self.top_n          = top_n
        self.attraction     = attraction
        self.repulsion      = repulsion
        self.musical_weight = musical_weight
        self.verbose        = verbose
        self.history        = []   # lista de dicts por generación
        self.best_ever      = []   # (fitness, Individual, vec, sim_d, sim_r)
        os.makedirs(output_dir, exist_ok=True)

        # Extraer key_obj del primer DNA fuente
        self.key_obj = getattr(source_dnas[0], 'key_obj', None)

    # ── Población inicial ─────────────────────────────────────────────────────

    def _seed_population(self):
        pop = []
        for i in range(self.pop_size):
            dnas = [copy.deepcopy(random.choice(self.source_dnas))
                    for _ in range(max(1, len(self.source_dnas)))]
            params = copy.deepcopy(self.params_base)
            params['seed'] = self.params_base.get('seed', 42) + i * 7
            if random.random() < 0.3:
                params['mode'] = random.choice([
                    'harmony_melody', 'rhythm_melody', 'full_blend', 'emotion'])
            ind = evolver_mod.Individual(dnas, params)
            ind.generation = 0
            pop.append(ind)
        return pop

    # ── Crossover y mutación (delegados a evolver) ────────────────────────────

    def _crossover(self, parent_a, parent_b):
        child = parent_a.clone()
        if random.random() < self.cross_rate:
            op = random.choice(['contour', 'harmony', 'rhythm'])
            for i in range(min(len(child.dna_list), len(parent_b.dna_list))):
                da = child.dna_list[i]
                db = parent_b.dna_list[i]
                if op == 'contour':
                    child.dna_list[i] = evolver_mod.crossover_contour(da, db)
                elif op == 'harmony':
                    child.dna_list[i] = evolver_mod.crossover_harmony(da, db)
                elif op == 'rhythm':
                    child.dna_list[i] = evolver_mod.crossover_rhythm(da, db)
        return child

    def _mutate(self, individual):
        ind = individual.clone()
        n_bars = ind.params.get('bars', 16)
        if random.random() < self.mut_rate:
            for i in range(len(ind.dna_list)):
                op = random.choice(['melody', 'rhythm', 'key', 'emotion', 'none'])
                if op == 'melody':
                    ind.dna_list[i] = evolver_mod.mutate_melody(ind.dna_list[i], rate=0.15)
                elif op == 'rhythm':
                    ind.dna_list[i] = evolver_mod.mutate_rhythm(ind.dna_list[i], rate=0.2)
                elif op == 'key':
                    ind.dna_list[i] = evolver_mod.mutate_key(ind.dna_list[i])
                elif op == 'emotion':
                    ind.dna_list[i] = evolver_mod.mutate_emotional_curve(ind.dna_list[i])
        if random.random() < self.mut_rate * 0.5:
            ind.params = evolver_mod.mutate_tempo(ind.params)
        if random.random() < self.mut_rate * 0.4:
            ind.params = evolver_mod.mutate_params_mt(ind.params, n_bars)
        ind.params['seed'] = random.randint(1, 99999)
        return ind

    # ── Evaluación con fitness dirigido ───────────────────────────────────────

    def _evaluate_individual(self, ind):
        """
        Genera el MIDI del individuo y calcula su fitness dirigido.
        Devuelve (fitness, midi_path, vec, sim_desired, sim_rejected).
        """
        # Generar MIDI usando la misma lógica que evolver
        gen = ind.generation
        uid = abs(hash(frozenset(str(ind.params).encode()))) % 100000
        out_path = os.path.join(self.output_dir, f"gen{gen:02d}_{uid:05d}.mid")

        temp_midis = []
        for i, dna in enumerate(ind.dna_list):
            tmp = os.path.join(self.output_dir, f"_tmp_gen{gen}_{uid}_src{i}.mid")
            evolver_mod._write_dna_as_midi(dna, tmp)
            temp_midis.append(tmp)

        if not temp_midis:
            return 0.0, None, None, 0.0, 0.0

        p = ind.params
        cmd = [sys.executable, self.mixer_script] + temp_midis
        cmd += ['--output', out_path]
        cmd += ['--bars', str(p.get('bars', 16))]
        cmd += ['--mode', p.get('mode', 'auto')]
        cmd += ['--seed', str(p.get('seed', 42))]
        cmd += ['--surprise', str(p.get('surprise', 0.08))]
        cmd += ['--rhythm_strength', str(p.get('rhythm_strength', 1.0))]
        if p.get('tempo'):
            cmd += ['--tempo', str(p['tempo'])]
        if p.get('no_percussion'):
            cmd += ['--no-percussion']
        for mt_key in ['mt_density', 'mt_harmony_complexity', 'mt_register', 'mt_swing']:
            val = p.get(mt_key)
            if val:
                cmd += ['--' + mt_key.replace('_', '-'), val]

        if self.verbose:
            print(f"    CMD: {' '.join(cmd[:6])} … [{len(cmd)} args]")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                if self.verbose:
                    print(f"    ⚠ Error: {result.stderr[-200:]}")
                for t in temp_midis:
                    try: os.remove(t)
                    except: pass
                return 0.0, None, None, 0.0, 0.0
        except (subprocess.TimeoutExpired, Exception) as e:
            if self.verbose:
                print(f"    ⚠ {type(e).__name__}: {e}")
            for t in temp_midis:
                try: os.remove(t)
                except: pass
            return 0.0, None, None, 0.0, 0.0

        for t in temp_midis:
            try: os.remove(t)
            except: pass

        if not os.path.exists(out_path):
            return 0.0, None, None, 0.0, 0.0

        fitness, vec, sim_d, sim_r = _score_midi_for_fitness(
            out_path, self.vec_desired, self.vec_rejected,
            self.attraction, self.repulsion, self.musical_weight, self.key_obj
        )
        return fitness, out_path, vec, sim_d, sim_r

    def _evaluate_population(self, population, generation):
        print(f"  Evaluando {len(population)} individuos (gen {generation})…")
        results = []
        for i, ind in enumerate(population):
            ind.generation = generation
            fitness, path, vec, sim_d, sim_r = self._evaluate_individual(ind)
            ind.fitness    = fitness
            ind.midi_path  = path
            # Guardar metadatos extra en el individuo
            ind._vec     = vec
            ind._sim_d   = sim_d
            ind._sim_r   = sim_r
            status = f"✓ {fitness:.3f} (→{sim_d:.2f} ←{sim_r:.2f})" if path else "✗ error"
            print(f"    [{i+1:2d}/{len(population)}] {status}", end='\r')
            results.append((fitness, sim_d, sim_r))
        print()
        return results

    def _tournament_select(self, population, k=3):
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda ind: ind.fitness)

    def _update_best(self, population):
        for ind in population:
            if ind.midi_path and os.path.exists(ind.midi_path):
                self.best_ever.append((
                    ind.fitness, copy.copy(ind),
                    getattr(ind, '_vec', None),
                    getattr(ind, '_sim_d', 0.0),
                    getattr(ind, '_sim_r', 0.0),
                ))
        self.best_ever.sort(key=lambda x: -x[0])
        seen = set()
        deduped = []
        for entry in self.best_ever:
            path = entry[1].midi_path
            if path not in seen:
                seen.add(path)
                deduped.append(entry)
        self.best_ever = deduped[:self.top_n * 3]

    # ── Loop principal ────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "═" * 68)
        print("  COUNTEREXAMPLE v1.0  —  Composición por contraejemplo")
        print("═" * 68)
        print(f"  Población     : {self.pop_size}  |  Generaciones: {self.n_gen}")
        print(f"  Élite         : {self.elite_size}  |  Mut.rate: {self.mut_rate:.2f}")
        print(f"  Atracción     : {self.attraction:.2f}  |  Repulsión: {self.repulsion:.2f}")
        print(f"  Peso musical  : {self.musical_weight:.2f}")
        print(f"  Salida        : {self.output_dir}")

        population = self._seed_population()
        self._evaluate_population(population, generation=0)
        population.sort(key=lambda ind: -ind.fitness)
        self._update_best(population)

        fitnesses = [ind.fitness for ind in population]
        print(f"\n  Gen 0: mejor={population[0].fitness:.3f}  "
              f"media={np.mean(fitnesses):.3f}")

        for gen in range(1, self.n_gen + 1):
            print(f"\n  ─── Generación {gen}/{self.n_gen} ───")
            elite = [ind.clone() for ind in population[:self.elite_size]]
            for e in elite:
                e.generation = gen

            new_pop = list(elite)
            while len(new_pop) < self.pop_size:
                pa = self._tournament_select(population)
                pb = self._tournament_select(population)
                child = self._crossover(pa, pb)
                child = self._mutate(child)
                child.generation = gen
                new_pop.append(child)

            results = self._evaluate_population(new_pop, gen)
            new_pop.sort(key=lambda ind: -ind.fitness)
            population = new_pop

            fitnesses = [ind.fitness for ind in population]
            sim_ds    = [getattr(ind, '_sim_d', 0.0) for ind in population]
            print(f"  Gen {gen}: mejor={population[0].fitness:.3f}  "
                  f"media={np.mean(fitnesses):.3f}  "
                  f"sim_deseado_best={population[0]._sim_d:.3f}")

            self.history.append({
                'generation': gen,
                'best_fitness': float(population[0].fitness),
                'mean_fitness': float(np.mean(fitnesses)),
                'best_sim_desired': float(population[0]._sim_d),
                'best_sim_rejected': float(population[0]._sim_r),
            })
            self._update_best(population)

        return population


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def export_results(evolver_obj, top_n, output_dir):
    """Copia los mejores MIDIs al directorio de salida con nombre descriptivo."""
    top = evolver_obj.best_ever[:top_n]
    exported = []
    print(f"\n  ══ TOP {top_n} CANDIDATOS ══")
    for rank, (fitness, ind, vec, sim_d, sim_r) in enumerate(top, 1):
        if not ind.midi_path or not os.path.exists(ind.midi_path):
            continue
        dest = os.path.join(output_dir, f"contra_top{rank:02d}_fit{fitness:.3f}.mid")
        shutil.copy2(ind.midi_path, dest)
        exported.append((dest, fitness, sim_d, sim_r))
        print(f"  #{rank}  fitness={fitness:.4f}  "
              f"sim_deseado={sim_d:.3f}  sim_rechazado={sim_r:.3f}")
        print(f"      gen={ind.generation}  "
              f"modo={ind.params.get('mode','?')}  "
              f"seed={ind.params.get('seed','?')}")
        print(f"      → {dest}")
    return exported


def save_report(args, vec_desired, vec_rejected, vec_ref, delta, significant_mask,
                evolver_obj, exported_files, output_dir):
    """Guarda un JSON con el análisis vectorial y el historial de evolución."""
    report = {
        'inputs': {
            'desired':  args.desired,
            'rejected': args.rejected,
            'reference': args.reference,
        },
        'params': {
            'generations':    args.generations,
            'population':     args.population,
            'elite':          args.elite,
            'mutation_rate':  args.mutation_rate,
            'repulsion':      args.repulsion,
            'attraction':     args.attraction,
            'musical_weight': args.musical_weight,
            'bars':           args.bars,
        },
        'vectors': {
            'dimensions': DIM_NAMES,
            'desired':    vec_desired.tolist(),
            'rejected':   vec_rejected.tolist(),
            'reference':  vec_ref.tolist(),
            'delta':      delta.tolist(),
            'significant_dims': [
                {'name': DIM_NAMES[i], 'delta': float(delta[i])}
                for i in np.argsort(-np.abs(delta))
                if significant_mask[i]
            ],
        },
        'evolution_history': evolver_obj.history if hasattr(evolver_obj, 'history') else [],
        'results': [
            {
                'rank': i + 1,
                'path': path,
                'fitness': float(fitness),
                'sim_desired': float(sim_d),
                'sim_rejected': float(sim_r),
            }
            for i, (path, fitness, sim_d, sim_r) in enumerate(exported_files)
        ],
    }
    rpt_path = os.path.join(output_dir, 'contra_analysis.json')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  → Informe guardado: {rpt_path}")
    return rpt_path


def plot_umap(vec_desired, vec_rejected, vec_ref, evolver_obj, output_dir):
    """Visualiza el espacio vectorial usando UMAP."""
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  [AVISO] umap-learn o matplotlib no disponibles. Instala con: pip install umap-learn matplotlib")
        return

    # Recolectar vectores de todos los candidatos generados
    all_vecs, all_labels, all_fitness = [], [], []

    for fitness, ind, vec, sim_d, sim_r in evolver_obj.best_ever:
        if vec is not None:
            all_vecs.append(vec)
            all_labels.append('candidato')
            all_fitness.append(fitness)

    if len(all_vecs) < 3:
        print("  [AVISO] Pocos vectores para UMAP. Omitiendo visualización.")
        return

    # Añadir los MIDIs de referencia
    special_vecs   = [vec_desired, vec_rejected, vec_ref]
    special_labels = ['deseado', 'rechazado', 'referencia']
    all_vecs_arr = np.array(all_vecs + special_vecs)

    reducer = umap.UMAP(n_neighbors=min(10, len(all_vecs_arr) - 1),
                        min_dist=0.1, random_state=42)
    emb = reducer.fit_transform(all_vecs_arr)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#1a1a1a')

    # Candidatos coloreados por fitness
    n_cand = len(all_vecs)
    sc = ax.scatter(emb[:n_cand, 0], emb[:n_cand, 1],
                    c=all_fitness, cmap='plasma', s=40, alpha=0.7,
                    label='candidatos', zorder=2)
    plt.colorbar(sc, ax=ax, label='fitness')

    # Puntos especiales
    colors_spec = ['#00ff88', '#ff4444', '#4488ff']
    markers_spec = ['★', '✖', '●']
    for i, (label, color) in enumerate(zip(special_labels, colors_spec)):
        xi, yi = emb[n_cand + i, 0], emb[n_cand + i, 1]
        ax.scatter(xi, yi, c=color, s=200, zorder=5, edgecolors='white', linewidths=1.5)
        ax.annotate(label, (xi, yi), textcoords='offset points', xytext=(8, 8),
                    color=color, fontsize=11, fontweight='bold')

    ax.set_title('Espacio vectorial  —  Composición por contraejemplo',
                 color='white', fontsize=13, pad=12)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

    legend_patches = [
        mpatches.Patch(color='#00ff88', label='deseado'),
        mpatches.Patch(color='#ff4444', label='rechazado'),
        mpatches.Patch(color='#4488ff', label='referencia'),
    ]
    ax.legend(handles=legend_patches, facecolor='#222222', labelcolor='white')

    out_path = os.path.join(output_dir, 'contra_umap.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"\n  → Visualización guardada: {out_path}")


def play_results(midi_paths, play_seconds=10):
    """Reproduce los mejores resultados."""
    try:
        import pygame
        pygame.init()
        pygame.mixer.init()
        for path in midi_paths:
            if not os.path.exists(path):
                continue
            print(f"\n  ▶ {os.path.basename(path)}")
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            t0 = time.time()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                if time.time() - t0 >= play_seconds:
                    pygame.mixer.music.stop()
                    break
        pygame.quit()
    except Exception as e:
        print(f"  ⚠ Reproducción no disponible: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='COUNTEREXAMPLE — Composición paramétrica por contraejemplo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    # Argumentos posicionales
    parser.add_argument('desired',   help='MIDI deseado (lo que te gusta)')
    parser.add_argument('rejected',  help='MIDI rechazado (lo que no te gusta)')
    parser.add_argument('reference', help='MIDI de referencia (material base para evolución)')

    # Fitness
    parser.add_argument('--repulsion',      type=float, default=1.2,
                        help='Peso λ de repulsión del rechazado (default: 1.2)')
    parser.add_argument('--attraction',     type=float, default=1.0,
                        help='Peso de atracción al deseado (default: 1.0)')
    parser.add_argument('--musical-weight', type=float, default=0.3,
                        help='Peso del score musical base (default: 0.3)')

    # Evolución
    parser.add_argument('--generations',    type=int,   default=6,
                        help='Generaciones (default: 6)')
    parser.add_argument('--population',     type=int,   default=12,
                        help='Tamaño de población (default: 12)')
    parser.add_argument('--elite',          type=int,   default=2,
                        help='Individuos de élite (default: 2)')
    parser.add_argument('--mutation-rate',  type=float, default=0.35,
                        help='Tasa de mutación (default: 0.35)')
    parser.add_argument('--bars',           type=int,   default=None,
                        help='Compases por individuo (default: desde referencia)')

    # Salida
    parser.add_argument('--top',            type=int,   default=3,
                        help='Mejores a exportar (default: 3)')
    parser.add_argument('--output-dir',     default='./contra_out',
                        help='Carpeta de salida (default: ./contra_out)')
    parser.add_argument('--mixer',          default=None,
                        help='Ruta a midi_dna_unified.py (default: auto)')
    parser.add_argument('--report',         action='store_true',
                        help='Exportar informe JSON')
    parser.add_argument('--plot',           action='store_true',
                        help='Visualizar espacio vectorial UMAP')
    parser.add_argument('--listen',         action='store_true',
                        help='Reproducir los mejores al final')
    parser.add_argument('--play-seconds',   type=int,   default=10,
                        help='Segundos de reproducción por MIDI (default: 10)')
    parser.add_argument('--seed',           type=int,   default=42,
                        help='Semilla aleatoria (default: 42)')
    parser.add_argument('--verbose',        action='store_true',
                        help='Informe detallado')
    parser.add_argument('--dry-run',        action='store_true',
                        help='Solo calcular diferencial, sin generar MIDIs')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Validar archivos de entrada ───────────────────────────────────────────
    for label, path in [('deseado', args.desired),
                         ('rechazado', args.rejected),
                         ('referencia', args.reference)]:
        if not os.path.exists(path):
            print(f"ERROR: No se encontró el MIDI {label}: {path}")
            sys.exit(1)

    # ── Localizar mixer ───────────────────────────────────────────────────────
    mixer_script = args.mixer
    if not mixer_script:
        candidates = [
            os.path.join(SCRIPT_DIR, 'midi_dna_unified.py'),
            'midi_dna_unified.py',
        ]
        for c in candidates:
            if os.path.exists(c):
                mixer_script = c
                break
    if not mixer_script or not os.path.exists(mixer_script):
        if not args.dry_run:
            print("ERROR: No se encontró midi_dna_unified.py. Usa --mixer para especificarlo.")
            sys.exit(1)

    # ══════════════════════════════════════════════════════════════════════════
    #  PASO 1: Vectorización
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 68)
    print("  COUNTEREXAMPLE v1.0  —  Composición paramétrica por contraejemplo")
    print("═" * 68)
    print("\n  [1/4] Vectorizando MIDIs…")

    vec_desired, meta_d = vectorize_midi(args.desired)
    vec_rejected, meta_r = vectorize_midi(args.rejected)
    vec_ref, meta_ref = vectorize_midi(args.reference)

    for label, vec in [('deseado', vec_desired),
                        ('rechazado', vec_rejected),
                        ('referencia', vec_ref)]:
        if vec is None:
            print(f"ERROR: No se pudo vectorizar el MIDI {label}.")
            sys.exit(1)

    print(f"    ✓ deseado   : {meta_d['n_notes']} notas, "
          f"{meta_d['n_tracks']} pistas, {meta_d['tempo_bpm']} BPM")
    print(f"    ✓ rechazado : {meta_r['n_notes']} notas, "
          f"{meta_r['n_tracks']} pistas, {meta_r['tempo_bpm']} BPM")
    print(f"    ✓ referencia: {meta_ref['n_notes']} notas, "
          f"{meta_ref['n_tracks']} pistas, {meta_ref['tempo_bpm']} BPM")

    # ══════════════════════════════════════════════════════════════════════════
    #  PASO 2: Cálculo del diferencial
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  [2/4] Calculando vector diferencial…")
    delta, significant_mask = compute_differential(vec_desired, vec_rejected)

    print(describe_differential(delta, significant_mask))

    # Similitud inicial de la referencia con deseado/rechazado
    sim_ref_d = cosine_similarity(vec_ref, vec_desired)
    sim_ref_r = cosine_similarity(vec_ref, vec_rejected)
    fit_ref, _, _ = directional_fitness(
        vec_ref, vec_desired, vec_rejected,
        args.attraction, args.repulsion, 0.0, 0.0)
    print(f"\n  Referencia — sim_deseado={sim_ref_d:.3f}  "
          f"sim_rechazado={sim_ref_r:.3f}  fitness_base={fit_ref:.3f}")

    if args.dry_run:
        print("\n  [--dry-run] Análisis completado. Sin generación de MIDIs.")
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════════
    #  PASO 3: Extracción de ADN del MIDI de referencia
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  [3/4] Extrayendo ADN del MIDI de referencia…")
    if not DNA_OK:
        print("ERROR: midi_dna_unified.py no disponible. No se puede continuar.")
        sys.exit(1)
    if not EVOLVER_OK:
        print("ERROR: evolver.py no disponible. No se puede continuar.")
        sys.exit(1)

    ref_dna = dna_mod.UnifiedDNA(args.reference)
    if not ref_dna.extract(verbose=args.verbose):
        print("ERROR: No se pudo extraer ADN del MIDI de referencia.")
        sys.exit(1)
    print(f"    ✓ ADN extraído: {ref_dna.key_obj.tonic.name} {ref_dna.key_obj.mode}  "
          f"{ref_dna.tempo_bpm:.0f} BPM")

    # Inferir número de compases si no se especificó
    n_bars = args.bars
    if n_bars is None:
        # Estimación desde el ADN: forma musical
        n_bars = max(8, len(ref_dna.phrase_lengths) * 4
                     if ref_dna.phrase_lengths else 16)
        print(f"    ✓ Compases inferidos: {n_bars}")

    params_base = {
        'mode': 'auto',
        'bars': n_bars,
        'tempo': ref_dna.tempo_bpm,
        'surprise': 0.08,
        'rhythm_strength': 1.0,
        'seed': args.seed,
        'no_percussion': True,
    }

    # ══════════════════════════════════════════════════════════════════════════
    #  PASO 4: Evolución guiada
    # ══════════════════════════════════════════════════════════════════════════
    print("\n  [4/4] Iniciando evolución genética dirigida…")
    os.makedirs(args.output_dir, exist_ok=True)

    ce = CounterexampleEvolver(
        source_dnas    = [ref_dna],
        params_base    = params_base,
        vec_desired    = vec_desired,
        vec_rejected   = vec_rejected,
        output_dir     = args.output_dir,
        mixer_script   = mixer_script,
        population_size = args.population,
        elite_size     = args.elite,
        mutation_rate  = args.mutation_rate,
        crossover_rate = 0.6,
        n_generations  = args.generations,
        top_n          = args.top,
        attraction     = args.attraction,
        repulsion      = args.repulsion,
        musical_weight = args.musical_weight,
        verbose        = args.verbose,
    )
    ce.run()

    # ── Exportar resultados ───────────────────────────────────────────────────
    exported = export_results(ce, args.top, args.output_dir)

    if args.report:
        save_report(args, vec_desired, vec_rejected, vec_ref,
                    delta, significant_mask, ce, exported, args.output_dir)

    if args.plot:
        plot_umap(vec_desired, vec_rejected, vec_ref, ce, args.output_dir)

    # ── Limpiar archivos temporales ───────────────────────────────────────────
    for f in os.listdir(args.output_dir):
        if f.startswith('_tmp_') or (f.startswith('gen') and not f.startswith('contra')):
            try:
                os.remove(os.path.join(args.output_dir, f))
            except Exception:
                pass

    if args.listen and exported:
        play_results([path for path, *_ in exported], args.play_seconds)

    print(f"\n  ✓ Completado. {len(exported)} MIDI(s) en: {args.output_dir}\n")


if __name__ == '__main__':
    main()
