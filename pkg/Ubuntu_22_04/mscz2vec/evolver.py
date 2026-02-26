"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        EVOLVER  v1.0                                         ║
║          Motor de evolución genética de secciones MIDI                       ║
║                                                                              ║
║  Toma N MIDIs como «genes», genera una población de variantes por            ║
║  cruce y mutación, las puntúa automáticamente con el scoring de              ║
║  midi_dna_unified y selecciona las mejores generación a generación.          ║
║                                                                              ║
║  OPERADORES GENÉTICOS:                                                       ║
║  [X1] Cruce de contorno melódico entre dos padres                            ║
║  [X2] Cruce de progresión armónica                                           ║
║  [X3] Cruce de patrón rítmico                                                ║
║  [M1] Mutación de intervalo: desplaza notas aleatorias ±1-3 semitonos        ║
║  [M2] Mutación rítmica: escala duraciones de notas aleatorias                ║
║  [M3] Mutación de tempo: ±5-15 BPM                                           ║
║  [M4] Mutación de tonalidad: transpone al relativo o dominante               ║
║  [M5] Mutación emocional: sesgos en las curvas de tensión/arousal            ║
║                                                                              ║
║  FITNESS:                                                                    ║
║  - score_candidate() de midi_dna_unified (consonancia, variedad, arco)      ║
║  - Penalización de clones (diversidad de población)                          ║
║  - Bonus por coherencia motívica entre generaciones                          ║
║                                                                              ║
║  USO:                                                                        ║
║    python evolver.py fuente1.mid fuente2.mid [más MIDIs]                    ║
║    python evolver.py *.mid --generations 10 --population 20                 ║
║    python evolver.py a.mid b.mid --generations 5 --elite 3                  ║
║    python evolver.py a.mid b.mid --mode full_blend --bars 32                ║
║    python evolver.py a.mid b.mid --generations 8 --output-dir ./evolved     ║
║    python evolver.py a.mid b.mid --listen --play-seconds 8                  ║
║    python evolver.py a.mid b.mid --seed 99 --verbose                        ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --generations N   Número de generaciones (default: 5)                     ║
║    --population N    Tamaño de la población por generación (default: 10)     ║
║    --elite N         Individuos de élite que pasan directamente (default: 2) ║
║    --mutation-rate F Probabilidad de mutación por individuo 0-1 (default:   ║
║                      0.3)                                                    ║
║    --crossover-rate F Probabilidad de cruce 0-1 (default: 0.6)              ║
║    --bars N          Compases de cada individuo (default: 16)                ║
║    --mode MODE       Modo de mezcla de midi_dna_unified (default: auto)      ║
║    --output-dir DIR  Carpeta de salida (default: ./evolver_out)              ║
║    --save-all        Guardar todos los individuos, no solo los mejores       ║
║    --top N           Cuántos mejores individuos exportar (default: 3)        ║
║    --listen          Reproducir los mejores al final (requiere pygame)       ║
║    --play-seconds N  Segundos de reproducción por MIDI (default: 10)        ║
║    --no-percussion   No generar pista de percusión                           ║
║    --seed N          Semilla aleatoria (default: 42)                         ║
║    --verbose         Informe detallado de cada generación                    ║
║                                                                              ║
║  DEPENDENCIAS: las mismas que midi_dna_unified.py                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import random
import copy
import subprocess
import time
import tempfile
import shutil
from collections import defaultdict

import numpy as np

# ── Importar módulos de midi_dna_unified ──────────────────────────────────────
# Asumimos que está en el mismo directorio (o en el PATH).
# Importamos solo lo necesario para no reescribir código ya existente.
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import midi_dna_unified as dna_mod
except ImportError:
    print("ERROR: midi_dna_unified.py no encontrado en el mismo directorio.")
    sys.exit(1)

import mido


# ══════════════════════════════════════════════════════════════════════════════
#  REPRESENTACIÓN DE UN INDIVIDUO
# ══════════════════════════════════════════════════════════════════════════════

class Individual:
    """
    Un individuo de la población genética.
    Contiene los «genes» (ADN extraído de los MIDIs fuente)
    y los parámetros de generación.
    """
    def __init__(self, dna_list, params):
        """
        dna_list : lista de UnifiedDNA (copiados del padre, posiblemente mutados)
        params   : dict con parámetros de generación
                   {mode, bars, key, tempo, emotion_src, form_src,
                    rhythm_strength, surprise, seed,
                    mt_density, mt_harmony_complexity, mt_register,
                    mt_swing, mt_emotion_morph}
        """
        self.dna_list = dna_list
        self.params   = params
        self.fitness  = 0.0
        self.midi_path = None   # ruta al MIDI generado
        self.generation = 0

    def clone(self):
        """Copia profunda del individuo."""
        new_dna = [copy.deepcopy(d) for d in self.dna_list]
        return Individual(new_dna, copy.deepcopy(self.params))


# ══════════════════════════════════════════════════════════════════════════════
#  OPERADORES GENÉTICOS
# ══════════════════════════════════════════════════════════════════════════════

def crossover_contour(dna_a, dna_b):
    """[X1] Cruce del contorno melódico entre dos ADNs padre."""
    child = copy.deepcopy(dna_a)
    contour_a = list(dna_a.pitch_contour) or [0]
    contour_b = list(dna_b.pitch_contour) or [0]
    # Asegurar misma longitud
    n = max(len(contour_a), len(contour_b))
    contour_a = (contour_a * (n // len(contour_a) + 1))[:n]
    contour_b = (contour_b * (n // len(contour_b) + 1))[:n]
    # Punto de cruce aleatorio
    cp = random.randint(1, n - 1)
    child.pitch_contour = contour_a[:cp] + contour_b[cp:]
    # Recalcular secuencia de pitch desde el nuevo contorno
    if child.pitch_sequence:
        start = child.pitch_sequence[0]
        child.pitch_sequence = [start]
        for step in child.pitch_contour:
            child.pitch_sequence.append(child.pitch_sequence[-1] + step)
    return child


def crossover_harmony(dna_a, dna_b):
    """[X2] Cruce de progresión armónica."""
    child = copy.deepcopy(dna_a)
    prog_a = list(dna_a.harmony_prog) or [('I', 2.0)]
    prog_b = list(dna_b.harmony_prog) or [('I', 2.0)]
    n = max(len(prog_a), len(prog_b))
    prog_a = (prog_a * (n // len(prog_a) + 1))[:n]
    prog_b = (prog_b * (n // len(prog_b) + 1))[:n]
    cp = random.randint(1, n - 1)
    child.harmony_prog = prog_a[:cp] + prog_b[cp:]
    child.harmony_functions = [f for f, _ in child.harmony_prog]
    return child


def crossover_rhythm(dna_a, dna_b):
    """[X3] Cruce del patrón rítmico."""
    child = copy.deepcopy(dna_a)
    pat_a = list(dna_a.rhythm_pattern) or [[]]
    pat_b = list(dna_b.rhythm_pattern) or [[]]
    n = max(len(pat_a), len(pat_b))
    pat_a = (pat_a * (n // len(pat_a) + 1))[:n]
    pat_b = (pat_b * (n // len(pat_b) + 1))[:n]
    # Intercalar compases de cada padre
    child.rhythm_pattern = [pat_a[i] if i % 2 == 0 else pat_b[i] for i in range(n)]
    # Rhythm grid: promedio ponderado
    ga = np.array(dna_a.rhythm_grid) if len(dna_a.rhythm_grid) == 16 else np.ones(16) / 16
    gb = np.array(dna_b.rhythm_grid) if len(dna_b.rhythm_grid) == 16 else np.ones(16) / 16
    alpha = random.uniform(0.3, 0.7)
    child.rhythm_grid = (ga * alpha + gb * (1 - alpha)).tolist()
    return child


def mutate_melody(dna, rate=0.2):
    """[M1] Mutación de intervalo: desplaza notas del contorno ±1-3 semitonos."""
    child = copy.deepcopy(dna)
    contour = list(child.pitch_contour)
    for i in range(len(contour)):
        if random.random() < rate:
            shift = random.choice([-3, -2, -1, 1, 2, 3])
            contour[i] += shift
    child.pitch_contour = contour
    # Actualizar secuencia
    if child.pitch_sequence and contour:
        start = child.pitch_sequence[0]
        child.pitch_sequence = [start]
        for step in contour:
            child.pitch_sequence.append(child.pitch_sequence[-1] + step)
    # Re-entrenamos Markov con la nueva secuencia
    try:
        child.markov = dna_mod.MarkovMelody(order=2)
        child.markov.train(
            [child.pitch_sequence[i+1] - child.pitch_sequence[i]
             for i in range(len(child.pitch_sequence) - 1)],
            child.durations
        )
    except Exception:
        pass
    return child


def mutate_rhythm(dna, rate=0.25):
    """[M2] Mutación rítmica: escala duraciones de eventos aleatorios."""
    child = copy.deepcopy(dna)
    new_pattern = []
    for bar in child.rhythm_pattern:
        new_bar = []
        for event in bar:
            if len(event) >= 4 and random.random() < rate:
                offset, dur, accent, syn = event[0], event[1], event[2], event[3]
                scale = random.choice([0.5, 0.75, 1.5, 2.0])
                new_dur = max(0.1, dur * scale)
                new_bar.append((offset, new_dur, accent, syn))
            else:
                new_bar.append(event)
        new_pattern.append(new_bar)
    child.rhythm_pattern = new_pattern
    return child


def mutate_tempo(params, delta_range=(5, 15)):
    """[M3] Mutación de tempo: ±delta BPM."""
    p = copy.deepcopy(params)
    delta = random.randint(*delta_range) * random.choice([-1, 1])
    current = p.get('tempo', 120.0)
    p['tempo'] = max(60.0, min(200.0, current + delta))
    return p


def mutate_key(dna):
    """[M4] Mutación de tonalidad: transpone al relativo o dominante."""
    child = copy.deepcopy(dna)
    mode = random.choice(['relative', 'dominant'])
    try:
        if mode == 'relative':
            new_key = dna_mod._get_relative_key(child.key_obj)
        else:
            new_key = dna_mod._get_dominant_key(child.key_obj)
        # Transponer secuencia de pitch
        from music21 import pitch as m21pitch
        src_tonic = m21pitch.Pitch(child.key_obj.tonic.name).pitchClass
        tgt_tonic = m21pitch.Pitch(new_key.tonic.name).pitchClass
        shift = (tgt_tonic - src_tonic)
        if shift > 6:  shift -= 12
        if shift < -6: shift += 12
        child.pitch_sequence = [p + shift for p in child.pitch_sequence]
        child.key_obj = new_key
    except Exception:
        pass
    return child


def mutate_emotional_curve(dna, curve_name='tension_curve', noise_sigma=0.1):
    """[M5] Mutación emocional: añade ruido gaussiano suave a una curva."""
    child = copy.deepcopy(dna)
    curve = getattr(child, curve_name, [])
    if curve:
        arr = np.array(curve, dtype=float)
        arr += np.random.normal(0, noise_sigma, size=arr.shape)
        arr = np.clip(arr, 0.0, 1.0)
        setattr(child, curve_name, arr.tolist())
    return child


def mutate_params_mt(params, n_bars):
    """Muta las curvas de mutación temporal (mt_*) de los parámetros."""
    p = copy.deepcopy(params)
    options = [
        ('mt_density',            '0:sparse, {h}:dense, {n}:medium'),
        ('mt_harmony_complexity', '0:simple, {h}:extended, {n}:diatonic'),
        ('mt_register',           '0:mid-low, {h}:high, {n}:mid'),
        ('mt_swing',              '0:0.0, {h}:0.8, {n}:0.0'),
    ]
    # Seleccionar una curva al azar y randomizarla
    key, template = random.choice(options)
    h = random.randint(n_bars // 4, 3 * n_bars // 4)
    curve_str = template.format(h=h, n=n_bars)
    p[key] = curve_str
    return p


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUACIÓN DE FITNESS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_individual(individual, output_dir, mixer_script, verbose=False):
    """
    Genera el MIDI del individuo llamando a midi_dna_unified.py como subproceso
    y luego lo lee para calcular el fitness con score_candidate().
    Retorna el fitness score (float) y la ruta al MIDI generado.
    """
    p = individual.params
    gen = individual.generation
    uid = abs(hash(frozenset(str(p).encode())))  % 100000

    out_path = os.path.join(output_dir, f"gen{gen:02d}_{uid:05d}.mid")

    # Guardar los ADNs mutados como MIDIs temporales para pasarlos al mixer
    temp_midis = []
    for i, dna in enumerate(individual.dna_list):
        tmp = os.path.join(output_dir, f"_tmp_gen{gen}_{uid}_src{i}.mid")
        _write_dna_as_midi(dna, tmp)
        temp_midis.append(tmp)

    if not temp_midis:
        return 0.0, None

    # Construir comando
    cmd = [sys.executable, mixer_script] + temp_midis
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
    # Curvas de mutación temporal
    for mt_key in ['mt_density', 'mt_harmony_complexity', 'mt_register', 'mt_swing']:
        val = p.get(mt_key)
        if val:
            flag = '--' + mt_key.replace('_', '-')
            cmd += [flag, val]

    if verbose:
        print(f"    CMD: {' '.join(cmd[:8])} … [{len(cmd)} args]")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            if verbose:
                print(f"    ⚠ Error en generación: {result.stderr[-300:]}")
            return 0.0, None
    except subprocess.TimeoutExpired:
        if verbose:
            print("    ⚠ Timeout en generación")
        return 0.0, None
    except Exception as e:
        if verbose:
            print(f"    ⚠ Excepción: {e}")
        return 0.0, None

    # Limpiar MIDIs temporales
    for tmp in temp_midis:
        try:
            os.remove(tmp)
        except Exception:
            pass

    # Calcular fitness: leer el MIDI generado y usar score_candidate()
    fitness = _score_midi_file(out_path, individual.dna_list[0].key_obj)
    return fitness, out_path


def _score_midi_file(midi_path, key_obj):
    """Lee un MIDI y calcula el fitness con score_candidate()."""
    try:
        mid = mido.MidiFile(midi_path)
    except Exception:
        return 0.0

    TICKS = mid.ticks_per_beat or 480
    notes = []
    for track in mid.tracks:
        abs_ticks = 0
        pending = {}
        for msg in track:
            abs_ticks += msg.time
            abs_beats = abs_ticks / TICKS
            if msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_beats, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                key_ = (msg.channel, msg.note)
                if key_ in pending:
                    onset, vel = pending.pop(key_)
                    dur = abs_beats - onset
                    if dur < 0.01:
                        dur = 0.25
                    # Solo canal 0 (melodía principal)
                    if msg.channel == 0:
                        notes.append((onset, msg.note, dur, vel))

    if not notes:
        return 0.0

    # Usar score_candidate de midi_dna_unified
    try:
        score = dna_mod.score_candidate(notes, [], key_obj)
        # Bonus por número de notas (densidad mínima)
        density_bonus = min(len(notes) / 64.0, 0.1)
        return float(score + density_bonus)
    except Exception:
        return 0.0


def _write_dna_as_midi(dna, output_path):
    """
    Escribe el ADN mutado como un MIDI provisional para pasarlo al mixer.
    Usa la secuencia de pitch + duraciones extraídas del ADN.
    """
    TICKS = 480
    bpm = getattr(dna, 'tempo_bpm', 120.0)
    us_per_beat = int(60_000_000 / max(bpm, 1))
    ts_num, ts_den = getattr(dna, 'time_sig', (4, 4))

    mid = mido.MidiFile(type=0, ticks_per_beat=TICKS)
    trk = mido.MidiTrack()
    mid.tracks.append(trk)

    trk.append(mido.MetaMessage('set_tempo', tempo=us_per_beat, time=0))
    trk.append(mido.MetaMessage('time_signature',
                                 numerator=ts_num, denominator=ts_den,
                                 clocks_per_click=24, notated_32nd_notes_per_beat=8,
                                 time=0))
    trk.append(mido.Message('program_change', channel=0, program=0, time=0))

    pitches = getattr(dna, 'pitch_sequence', []) or []
    durations = getattr(dna, 'durations', []) or []

    # Sincronizar longitudes
    n = min(len(pitches), len(durations), 128)
    if n == 0:
        # Sin datos: escribir escala de do mayor
        pitches = [60, 62, 64, 65, 67, 69, 71, 72]
        durations = [0.5] * len(pitches)
        n = len(pitches)

    events = []
    cursor = 0.0
    for i in range(n):
        p = max(21, min(108, int(pitches[i])))
        d = max(0.1, float(durations[i]))
        t_on  = int(cursor * TICKS)
        t_off = int((cursor + d * 0.9) * TICKS)
        events.append((t_on, 'on', p, 80))
        events.append((t_off, 'off', p, 0))
        cursor += d

    events.sort(key=lambda e: (e[0], 0 if e[1] == 'off' else 1))
    prev = 0
    for tick, kind, note, vel in events:
        delta = max(0, tick - prev)
        prev = tick
        msg_type = 'note_on' if kind == 'on' else 'note_off'
        trk.append(mido.Message(msg_type, channel=0, note=note, velocity=vel, time=delta))

    trk.append(mido.MetaMessage('end_of_track', time=0))

    try:
        mid.save(output_path)
    except Exception as e:
        print(f"    ⚠ No se pudo guardar MIDI temporal: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  EVOLUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

class GeneticEvolver:
    def __init__(self, source_dnas, params_base, output_dir,
                 mixer_script, population_size=10, elite_size=2,
                 mutation_rate=0.3, crossover_rate=0.6,
                 n_generations=5, top_n=3, verbose=False):
        self.source_dnas   = source_dnas
        self.params_base   = params_base
        self.output_dir    = output_dir
        self.mixer_script  = mixer_script
        self.pop_size      = population_size
        self.elite_size    = elite_size
        self.mut_rate      = mutation_rate
        self.cross_rate    = crossover_rate
        self.n_gen         = n_generations
        self.top_n         = top_n
        self.verbose       = verbose
        self.history       = []   # (generation, fitness_list)
        self.best_ever     = []   # lista de (fitness, Individual)
        os.makedirs(output_dir, exist_ok=True)

    # ── Población inicial ─────────────────────────────────────────────────────

    def _seed_population(self):
        """Genera la población inicial con variaciones de los parámetros base."""
        pop = []
        n = self.pop_size
        for i in range(n):
            # Asignar padres: selección aleatoria de los ADNs fuente
            dnas = [copy.deepcopy(random.choice(self.source_dnas))
                    for _ in range(max(1, len(self.source_dnas)))]
            params = copy.deepcopy(self.params_base)
            # Pequeña variación de semilla para diversidad
            params['seed'] = self.params_base.get('seed', 42) + i * 7
            # Variar modo aleatoriamente en el 30% de la población inicial
            if random.random() < 0.3:
                params['mode'] = random.choice(['harmony_melody', 'rhythm_melody', 'full_blend', 'emotion'])
            ind = Individual(dnas, params)
            ind.generation = 0
            pop.append(ind)
        return pop

    # ── Cruce ─────────────────────────────────────────────────────────────────

    def _crossover(self, parent_a, parent_b):
        """Genera un hijo por cruce de dos padres."""
        child = parent_a.clone()
        if random.random() < self.cross_rate:
            op = random.choice(['contour', 'harmony', 'rhythm'])
            for i in range(min(len(child.dna_list), len(parent_b.dna_list))):
                dna_a = child.dna_list[i]
                dna_b = parent_b.dna_list[i]
                if op == 'contour':
                    child.dna_list[i] = crossover_contour(dna_a, dna_b)
                elif op == 'harmony':
                    child.dna_list[i] = crossover_harmony(dna_a, dna_b)
                elif op == 'rhythm':
                    child.dna_list[i] = crossover_rhythm(dna_a, dna_b)
        return child

    # ── Mutación ─────────────────────────────────────────────────────────────

    def _mutate(self, individual):
        """Aplica mutaciones aleatorias al individuo."""
        ind = individual.clone()
        n_bars = ind.params.get('bars', 16)

        if random.random() < self.mut_rate:
            for i in range(len(ind.dna_list)):
                op = random.choice(['melody', 'rhythm', 'key', 'emotion', 'none'])
                if op == 'melody':
                    ind.dna_list[i] = mutate_melody(ind.dna_list[i], rate=0.15)
                elif op == 'rhythm':
                    ind.dna_list[i] = mutate_rhythm(ind.dna_list[i], rate=0.2)
                elif op == 'key':
                    ind.dna_list[i] = mutate_key(ind.dna_list[i])
                elif op == 'emotion':
                    ind.dna_list[i] = mutate_emotional_curve(ind.dna_list[i])

        if random.random() < self.mut_rate * 0.5:
            ind.params = mutate_tempo(ind.params)

        if random.random() < self.mut_rate * 0.4:
            ind.params = mutate_params_mt(ind.params, n_bars)

        # Variar semilla para que la generación sea diferente
        ind.params['seed'] = random.randint(1, 99999)
        return ind

    # ── Selección ─────────────────────────────────────────────────────────────

    def _tournament_select(self, population, k=3):
        """Selección por torneo de k individuos."""
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda ind: ind.fitness)

    # ── Evaluación de la población ────────────────────────────────────────────

    def _evaluate_population(self, population, generation):
        """Evalúa todos los individuos de una generación."""
        print(f"  Evaluando {len(population)} individuos (gen {generation})…")
        for i, ind in enumerate(population):
            ind.generation = generation
            fitness, path = evaluate_individual(
                ind, self.output_dir, self.mixer_script, self.verbose)
            ind.fitness  = fitness
            ind.midi_path = path
            status = f"✓ {fitness:.3f}" if path else "✗ error"
            print(f"    [{i+1:2d}/{len(population)}] {status}", end='\r')
        print()  # nueva línea

    # ── Loop principal ────────────────────────────────────────────────────────

    def run(self):
        """Ejecuta el algoritmo genético completo."""
        print("\n" + "═" * 65)
        print("  EVOLVER v1.0  —  Evolución genética de secciones MIDI")
        print("═" * 65)
        print(f"  Fuentes      : {len(self.source_dnas)} ADNs")
        print(f"  Población    : {self.pop_size}  |  Generaciones: {self.n_gen}")
        print(f"  Élite        : {self.elite_size}  |  Mut.rate: {self.mut_rate:.2f}  |  Cross.rate: {self.cross_rate:.2f}")
        print(f"  Compases     : {self.params_base.get('bars', 16)}")
        print(f"  Directorio   : {self.output_dir}")

        # Generación 0: población inicial
        population = self._seed_population()
        self._evaluate_population(population, generation=0)
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        print(f"\n  Gen 0: mejor={population[0].fitness:.3f}  media={np.mean([i.fitness for i in population]):.3f}")

        # Actualizar best_ever
        self._update_best(population)

        for gen in range(1, self.n_gen + 1):
            print(f"\n  ─── Generación {gen}/{self.n_gen} ───")

            # Élite: los mejores pasan directamente
            elite = [ind.clone() for ind in population[:self.elite_size]]
            for e in elite:
                e.generation = gen

            # Nueva población
            new_pop = list(elite)
            while len(new_pop) < self.pop_size:
                pa = self._tournament_select(population)
                pb = self._tournament_select(population)
                child = self._crossover(pa, pb)
                child = self._mutate(child)
                child.generation = gen
                new_pop.append(child)

            # Re-evaluar todos (incluyendo élite con nueva semilla)
            self._evaluate_population(new_pop, gen)
            new_pop.sort(key=lambda ind: ind.fitness, reverse=True)
            population = new_pop

            fitnesses = [ind.fitness for ind in population]
            print(f"  Gen {gen}: mejor={population[0].fitness:.3f}  "
                  f"media={np.mean(fitnesses):.3f}  "
                  f"min={min(fitnesses):.3f}")
            self.history.append((gen, fitnesses))
            self._update_best(population)

        return population

    def _update_best(self, population):
        """Mantiene la lista de los mejores individuos entre generaciones."""
        for ind in population:
            if ind.midi_path and os.path.exists(ind.midi_path):
                self.best_ever.append((ind.fitness, copy.copy(ind)))
        self.best_ever.sort(key=lambda x: -x[0])
        # Deduplicar por path y mantener solo top_n * 3
        seen = set()
        deduped = []
        for f, ind in self.best_ever:
            if ind.midi_path not in seen:
                seen.add(ind.midi_path)
                deduped.append((f, ind))
        self.best_ever = deduped[:self.top_n * 3]


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORTACIÓN DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════

def export_top_results(evolver, top_n, output_dir):
    """Copia los mejores MIDIs a la carpeta de salida con nombre descriptivo."""
    top = evolver.best_ever[:top_n]
    exported = []
    print(f"\n  ═══ TOP {top_n} RESULTADOS ═══")
    for rank, (fitness, ind) in enumerate(top, 1):
        if not ind.midi_path or not os.path.exists(ind.midi_path):
            continue
        dest = os.path.join(output_dir, f"evolved_top{rank:02d}_fit{fitness:.3f}.mid")
        shutil.copy2(ind.midi_path, dest)
        exported.append(dest)
        print(f"  #{rank}  fitness={fitness:.4f}  gen={ind.generation}")
        print(f"       modo={ind.params.get('mode','?')}  "
              f"tempo={ind.params.get('tempo',120.0):.0f}  "
              f"seed={ind.params.get('seed','?')}")
        print(f"       → {dest}")
    return exported


def save_evolution_report(evolver, output_dir):
    """Guarda un JSON con el historial de evolución."""
    report = {
        'generations': evolver.n_gen,
        'population_size': evolver.pop_size,
        'elite_size': evolver.elite_size,
        'mutation_rate': evolver.mut_rate,
        'crossover_rate': evolver.cross_rate,
        'history': [
            {'generation': g, 'max_fitness': max(fs), 'mean_fitness': float(np.mean(fs))}
            for g, fs in evolver.history
        ],
        'top_results': [
            {'rank': i+1, 'fitness': f, 'generation': ind.generation,
             'midi': ind.midi_path, 'params': {
                 k: v for k, v in ind.params.items()
                 if not k.startswith('_')
             }}
            for i, (f, ind) in enumerate(evolver.best_ever[:evolver.top_n])
        ]
    }
    rpt_path = os.path.join(output_dir, 'evolution_report.json')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  → Informe guardado: {rpt_path}")
    return rpt_path


def play_top_results(midi_paths, play_seconds=10):
    """Reproduce los mejores MIDIs usando pygame."""
    try:
        import pygame
        pygame.init()
        pygame.mixer.init()
        for path in midi_paths:
            if not os.path.exists(path):
                continue
            print(f"\n  ▶ Reproduciendo: {os.path.basename(path)}")
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
        print(f"  ⚠ No se pudo reproducir: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='EVOLVER — Motor de evolución genética de secciones MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('inputs', nargs='+', help='MIDIs fuente (genes)')
    parser.add_argument('--generations',   type=int,   default=5,   help='Número de generaciones (default: 5)')
    parser.add_argument('--population',    type=int,   default=10,  help='Tamaño de la población (default: 10)')
    parser.add_argument('--elite',         type=int,   default=2,   help='Individuos de élite (default: 2)')
    parser.add_argument('--mutation-rate', type=float, default=0.3, help='Tasa de mutación 0-1 (default: 0.3)')
    parser.add_argument('--crossover-rate',type=float, default=0.6, help='Tasa de cruce 0-1 (default: 0.6)')
    parser.add_argument('--bars',          type=int,   default=16,  help='Compases por individuo (default: 16)')
    parser.add_argument('--mode',          default='auto',
                        choices=['auto','rhythm_melody','harmony_melody','full_blend',
                                 'custom','mosaic','energy','emotion'],
                        help='Modo de mezcla (default: auto)')
    parser.add_argument('--output-dir',    default='./evolver_out', help='Carpeta de salida')
    parser.add_argument('--save-all',      action='store_true',     help='Guardar todos los individuos')
    parser.add_argument('--top',           type=int,   default=3,   help='Mejores a exportar (default: 3)')
    parser.add_argument('--listen',        action='store_true',     help='Reproducir los mejores al final')
    parser.add_argument('--play-seconds',  type=int,   default=10,  help='Segundos de reproducción (default: 10)')
    parser.add_argument('--seed',          type=int,   default=42,  help='Semilla aleatoria (default: 42)')
    parser.add_argument('--verbose',       action='store_true',     help='Informe detallado')
    parser.add_argument('--no-percussion', action='store_true',     help='No generar pista de percusión')
    parser.add_argument('--mixer',         default=None,
                        help='Ruta al script midi_dna_unified.py (default: detecta automáticamente)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Localizar midi_dna_unified.py
    mixer_script = args.mixer
    if not mixer_script:
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'midi_dna_unified.py'),
            'midi_dna_unified.py',
        ]
        for c in candidates:
            if os.path.exists(c):
                mixer_script = c
                break
    if not mixer_script or not os.path.exists(mixer_script):
        print("ERROR: No se encontró midi_dna_unified.py. Usa --mixer para especificar la ruta.")
        sys.exit(1)

    # Validar ficheros de entrada
    midi_paths = [p for p in args.inputs if os.path.exists(p)]
    if not midi_paths:
        print("ERROR: No se encontraron ficheros MIDI válidos.")
        sys.exit(1)

    # Extraer ADNs de los MIDIs fuente
    print(f"\nExtrayendo ADN de {len(midi_paths)} fuente(s)…")
    source_dnas = []
    for path in midi_paths:
        print(f"  ▶ {os.path.basename(path)}")
        dna = dna_mod.UnifiedDNA(path)
        if dna.extract(verbose=args.verbose):
            source_dnas.append(dna)
        else:
            print(f"    ⚠ No se pudo extraer ADN de {path}")

    if not source_dnas:
        print("ERROR: No se pudo extraer ADN de ningún fichero.")
        sys.exit(1)

    print(f"  ✓ {len(source_dnas)} ADNs extraídos")

    # Parámetros base para la generación
    params_base = {
        'mode': args.mode,
        'bars': args.bars,
        'tempo': source_dnas[0].tempo_bpm,
        'surprise': 0.08,
        'rhythm_strength': 1.0,
        'seed': args.seed,
        'no_percussion': args.no_percussion,
    }

    # Crear el evolucionador
    evolver = GeneticEvolver(
        source_dnas    = source_dnas,
        params_base    = params_base,
        output_dir     = args.output_dir,
        mixer_script   = mixer_script,
        population_size= args.population,
        elite_size     = args.elite,
        mutation_rate  = args.mutation_rate,
        crossover_rate = args.crossover_rate,
        n_generations  = args.generations,
        top_n          = args.top,
        verbose        = args.verbose,
    )

    # Ejecutar evolución
    final_pop = evolver.run()

    # Exportar resultados
    exported = export_top_results(evolver, args.top, args.output_dir)
    save_evolution_report(evolver, args.output_dir)

    # Limpiar MIDIs intermedios si no se pidió guardar todos
    if not args.save_all:
        for fname in os.listdir(args.output_dir):
            if fname.startswith('gen') and fname.endswith('.mid'):
                full = os.path.join(args.output_dir, fname)
                if full not in exported:
                    try:
                        os.remove(full)
                    except Exception:
                        pass

    print(f"\n  ✓ Evolución completa. {len(exported)} MIDIs en: {args.output_dir}")

    if args.listen and exported:
        print("\n  ▶ Reproduciendo los mejores resultados…")
        play_top_results(exported, args.play_seconds)


if __name__ == '__main__':
    main()
