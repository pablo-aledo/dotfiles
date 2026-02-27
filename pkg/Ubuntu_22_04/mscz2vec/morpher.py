#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MORPHER  v1.0                                        ║
║         Morphing gradual entre dos obras MIDI                                ║
║                                                                              ║
║  Dado A.mid y B.mid, genera N pasos intermedios donde el ADN de A           ║
║  se transforma gradualmente en B. A diferencia de style_transfer            ║
║  (transferencia puntual), el morpher produce una secuencia continua          ║
║  que va de A a B a través de estados intermedios.                            ║
║                                                                              ║
║  DIMENSIONES DE MORPHING (todas controlables individualmente):               ║
║    [P] Pitch/Melodía  — contorno melódico y secuencia de notas              ║
║    [R] Ritmo          — patrón rítmico y groove                             ║
║    [H] Armonía        — progresión y complejidad armónica                   ║
║    [E] Emoción        — curvas de tensión/arousal/valencia                  ║
║    [D] Dinámica       — envolvente de velocidades                           ║
║    [T] Tempo          — velocidad de la pieza                               ║
║                                                                              ║
║  MODOS DE INTERPOLACIÓN:                                                     ║
║    linear    — interpolación lineal A→B                                     ║
║    sigmoid   — aceleración en el centro, suave en extremos                 ║
║    exponential — cambio lento al principio, rápido al final                ║
║    sinusoidal — A→B→A (ida y vuelta)                                       ║
║    step      — cambio abrupto en el centro                                  ║
║                                                                              ║
║  USO:                                                                        ║
║    python morpher.py A.mid B.mid                                            ║
║    python morpher.py A.mid B.mid --steps 8                                  ║
║    python morpher.py A.mid B.mid --steps 10 --mode sigmoid                  ║
║    python morpher.py A.mid B.mid --steps 6 --dims pitch rhythm              ║
║    python morpher.py A.mid B.mid --steps 5 --catalog                        ║
║    python morpher.py A.mid B.mid --steps 8 --plot                           ║
║    python morpher.py A.mid B.mid --steps 6 --listen                        ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    --steps N       Número de MIDIs intermedios (default: 6; incluye A y B) ║
║    --mode MODE     Modo de interpolación (default: sigmoid)                 ║
║    --dims DIMS     Dimensiones a interpolar (default: todas)                ║
║    --bars N        Compases por paso (default: auto desde A)                ║
║    --catalog       Generar un MIDI único que concatena todos los pasos      ║
║    --plot          Visualizar curvas de morphing                             ║
║    --listen        Reproducir secuencia al terminar (requiere pygame)       ║
║    --play-seconds N Segundos por paso en reproducción (default: 8)         ║
║    --output-dir DIR Carpeta de salida (default: ./morph_out)               ║
║    --prefix NAME   Prefijo de los archivos de salida (default: morph)      ║
║    --verbose       Informe detallado                                        ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    morph_000_A.mid          — copia del original A                          ║
║    morph_001_t0.17.mid      — paso intermedio (alpha=0.17)                  ║
║    ...                                                                      ║
║    morph_00N_B.mid          — copia del original B                          ║
║    morph_catalog.mid        — concatenación (con --catalog)                 ║
║    morph_curves.json        — datos de las curvas de morphing               ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy, midi_dna_unified.py                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import copy
import shutil
import time
import threading
import tempfile

import numpy as np

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    MIDO_OK = True
except ImportError:
    MIDO_OK = False
    print("ERROR: pip install mido")
    sys.exit(1)

try:
    import pygame
    pygame.init()
    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import midi_dna_unified as dna_mod
    from midi_dna_unified import UnifiedDNA
    DNA_OK = True
except ImportError:
    DNA_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE INTERPOLACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _interp_alpha(t: float, mode: str) -> float:
    """
    Dado t en [0,1], retorna alpha en [0,1] según el modo de interpolación.
    alpha=0 → todo A,  alpha=1 → todo B.
    """
    t = float(np.clip(t, 0, 1))
    if mode == 'linear':
        return t
    elif mode == 'sigmoid':
        # Sigmoide centrada en 0.5
        k = 10.0
        return float(1 / (1 + np.exp(-k * (t - 0.5))))
    elif mode == 'exponential':
        return float(t ** 2)
    elif mode == 'sinusoidal':
        # Va de A a B y vuelve a A
        return float(0.5 - 0.5 * np.cos(np.pi * t))
    elif mode == 'step':
        return 0.0 if t < 0.5 else 1.0
    return t


def lerp(a, b, alpha: float):
    """Interpolación lineal entre dos valores escalares o arrays."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) * (1 - alpha) + float(b) * alpha
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    if len(a_arr) != len(b_arr):
        # Resamplear al mismo tamaño
        n = max(len(a_arr), len(b_arr))
        a_arr = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a_arr)), a_arr)
        b_arr = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(b_arr)), b_arr)
    return (a_arr * (1 - alpha) + b_arr * alpha).tolist()


def lerp_int_list(a: list, b: list, alpha: float) -> list:
    """Interpolación lineal de listas de enteros (notas MIDI)."""
    result = lerp(a, b, alpha)
    return [int(round(x)) for x in result]


def lerp_harmony(prog_a: list, prog_b: list, alpha: float) -> list:
    """Interpolación de progresiones armónicas: mezcla figuras romanas."""
    n = max(len(prog_a), len(prog_b))
    prog_a_ext = (prog_a * (n // max(len(prog_a), 1) + 1))[:n]
    prog_b_ext = (prog_b * (n // max(len(prog_b), 1) + 1))[:n]
    result = []
    for i in range(n):
        fig_a, dur_a = prog_a_ext[i] if isinstance(prog_a_ext[i], tuple) else (prog_a_ext[i], 2.0)
        fig_b, dur_b = prog_b_ext[i] if isinstance(prog_b_ext[i], tuple) else (prog_b_ext[i], 2.0)
        dur = lerp(dur_a, dur_b, alpha)
        # Para la figura: usa A si alpha < 0.5, B si alpha >= 0.5,
        # con probabilidad proporcional a alpha
        if np.random.random() < alpha:
            result.append((fig_b, dur))
        else:
            result.append((fig_a, dur))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACTOR DE DNA SIMPLE (si midi_dna_unified no está disponible)
# ══════════════════════════════════════════════════════════════════════════════

class SimpleDNA:
    """ADN musical extraído directamente de un MIDI sin depender de UnifiedDNA."""
    def __init__(self, path: str):
        self.path      = path
        self.name      = os.path.basename(path)
        self.notes     = []       # [(pitch, velocity, start_tick, dur_ticks)]
        self.tempo     = 500000   # microsegundos por beat (= 120 BPM)
        self.ticks_per_beat = 480
        self.time_sig  = (4, 4)
        self.n_bars    = 0

    def extract(self) -> bool:
        if not MIDO_OK:
            return False
        try:
            mid = MidiFile(self.path)
            self.ticks_per_beat = mid.ticks_per_beat
            abs_time = 0
            active_notes = {}
            self.notes = []

            for msg in mido.merge_tracks(mid.tracks):
                abs_time += msg.time
                if msg.type == 'set_tempo':
                    self.tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[(msg.channel, msg.note)] = (msg.velocity, abs_time)
                elif msg.type in ('note_off', 'note_on') and \
                     (msg.velocity == 0 or msg.type == 'note_off'):
                    key = (msg.channel, msg.note)
                    if key in active_notes:
                        vel, start = active_notes.pop(key)
                        dur = abs_time - start
                        self.notes.append((msg.note, vel, start, max(dur, 1)))

            # Estimar n_bars
            if self.notes:
                last_tick = max(n[2] + n[3] for n in self.notes)
                beats_per_bar = self.time_sig[0]
                total_beats = last_tick / self.ticks_per_beat
                self.n_bars = max(1, int(total_beats / beats_per_bar))

            return bool(self.notes)
        except Exception as e:
            print(f"  ERROR extrayendo {self.path}: {e}")
            return False


# ══════════════════════════════════════════════════════════════════════════════
#  MORPHING DNA
# ══════════════════════════════════════════════════════════════════════════════

class MorphedDNA:
    """
    Representa un estado intermedio entre dos DNAs (A y B) a alpha dado.
    Puede generar el MIDI correspondiente.
    """
    def __init__(self, dna_a, dna_b, alpha: float, dims: list, n_bars: int = None):
        self.alpha  = alpha
        self.dims   = dims
        self.n_bars = n_bars

        # Copiar A como base
        self.tempo          = _interp_attr(dna_a, dna_b, 'tempo_bpm' if DNA_OK else 'tempo',
                                            alpha, 'tempo' in dims)
        self.pitch_sequence = _interp_list_attr(dna_a, dna_b, 'pitch_sequence',
                                                  alpha, 'pitch' in dims)
        self.durations      = _interp_list_attr(dna_a, dna_b, 'durations',
                                                  alpha, 'rhythm' in dims)
        self.tension_curve  = _interp_list_attr(dna_a, dna_b, 'tension_curve',
                                                  alpha, 'emotion' in dims)
        self.velocity_base  = _interp_attr(dna_a, dna_b,
                                            'dynamics_mean' if DNA_OK else 'velocity_base',
                                            alpha, 'dynamics' in dims)
        # Armonía
        if DNA_OK and 'harmony' in dims:
            prog_a = getattr(dna_a, 'harmony_prog', [('I', 2.0)])
            prog_b = getattr(dna_b, 'harmony_prog', [('I', 2.0)])
            self.harmony_prog = lerp_harmony(prog_a, prog_b, alpha)
        elif DNA_OK:
            self.harmony_prog = getattr(dna_a, 'harmony_prog', [('I', 2.0)])
        else:
            self.harmony_prog = [('I', 2.0)]

        # DNA original para generación completa
        self._dna_a = dna_a
        self._dna_b = dna_b

    def write_midi(self, path: str) -> bool:
        """Genera el MIDI para este estado interpolado."""
        if DNA_OK:
            return self._write_via_dna(path)
        return self._write_direct(path)

    def _write_direct(self, path: str) -> bool:
        """Escritura directa interpolando notas y tiempos."""
        a, b = self._dna_a, self._dna_b
        alpha = self.alpha

        notes_a = [(n, v, s, d) for n, v, s, d in a.notes] if hasattr(a, 'notes') else []
        notes_b = [(n, v, s, d) for n, v, s, d in b.notes] if hasattr(b, 'notes') else []

        if not notes_a and not notes_b:
            return False

        # Normalizar ambas listas al mismo número de notas
        n = max(len(notes_a), len(notes_b))
        if not notes_a:
            notes_a = notes_b
        if not notes_b:
            notes_b = notes_a

        def extend_notes(nl, target):
            if len(nl) == 0:
                return [(60, 64, 0, 480)] * target
            rep = (nl * (target // len(nl) + 1))[:target]
            return rep

        notes_a = extend_notes(notes_a, n)
        notes_b = extend_notes(notes_b, n)

        tpb = a.ticks_per_beat if hasattr(a, 'ticks_per_beat') else 480
        tempo_a = a.tempo if hasattr(a, 'tempo') else 500000
        tempo_b = b.tempo if hasattr(b, 'tempo') else 500000
        tempo   = int(lerp(tempo_a, tempo_b, alpha))

        # Limitar a n_bars
        n_bars = self.n_bars or max(a.n_bars if hasattr(a,'n_bars') else 8,
                                     b.n_bars if hasattr(b,'n_bars') else 8)
        max_ticks = n_bars * a.time_sig[0] * tpb if hasattr(a,'time_sig') else n_bars * 4 * tpb

        mid   = MidiFile(ticks_per_beat=tpb)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        events = []
        for i in range(n):
            na, va, sa, da = notes_a[i]
            nb, vb, sb, db = notes_b[i]
            note = int(round(lerp(na, nb, alpha)))
            vel  = int(round(lerp(va, vb, alpha)))
            start= int(round(lerp(sa, sb, alpha)))
            dur  = int(round(lerp(da, db, alpha)))

            note  = int(np.clip(note, 21, 108))
            vel   = int(np.clip(vel, 10, 120))
            start = max(0, start)
            dur   = max(1, dur)

            if start >= max_ticks:
                continue

            events.append(('on',  start,       note, vel))
            events.append(('off', start + dur, note, 0))

        events.sort(key=lambda x: x[1])

        prev_tick = 0
        for ev_type, tick, note, vel in events:
            delta = tick - prev_tick
            if ev_type == 'on':
                track.append(Message('note_on',  note=note, velocity=vel,  time=delta))
            else:
                track.append(Message('note_off', note=note, velocity=0,    time=delta))
            prev_tick = tick

        try:
            mid.save(path)
            return True
        except Exception as e:
            print(f"  ERROR guardando {path}: {e}")
            return False

    def _write_via_dna(self, path: str) -> bool:
        """Generación completa usando midi_dna_unified."""
        try:
            # Crear un DNA sintético con atributos interpolados
            synth_dna = copy.deepcopy(self._dna_a)

            if 'pitch' in self.dims and self.pitch_sequence:
                synth_dna.pitch_sequence = lerp_int_list(
                    self._dna_a.pitch_sequence or [60],
                    self._dna_b.pitch_sequence or [60],
                    self.alpha)

            if 'rhythm' in self.dims and self.durations:
                synth_dna.durations = lerp(
                    self._dna_a.durations or [1.0],
                    self._dna_b.durations or [1.0],
                    self.alpha)

            if 'emotion' in self.dims:
                for curve in ['tension_curve','arousal_curve','valence_curve','activity_curve']:
                    ca = getattr(self._dna_a, curve, [0.5])
                    cb = getattr(self._dna_b, curve, [0.5])
                    setattr(synth_dna, curve, lerp(ca, cb, self.alpha))

            if 'harmony' in self.dims:
                synth_dna.harmony_prog = self.harmony_prog

            if 'dynamics' in self.dims:
                synth_dna.dynamics_mean = int(lerp(
                    self._dna_a.dynamics_mean, self._dna_b.dynamics_mean, self.alpha))

            if 'tempo' in self.dims:
                synth_dna.tempo_bpm = float(lerp(
                    self._dna_a.tempo_bpm, self._dna_b.tempo_bpm, self.alpha))

            n_bars = self.n_bars or max(8, self._dna_a.n_bars if hasattr(self._dna_a,'n_bars') else 8)

            try:
                from midi_dna_unified import generate_midi_from_dna
                generate_midi_from_dna(synth_dna, n_bars, path, verbose=False)
                return True
            except Exception:
                pass

            # Fallback: escribir directamente desde pitch_sequence
            return self._write_from_sequence(synth_dna, path, n_bars)

        except Exception as e:
            print(f"  ERROR _write_via_dna: {e}")
            return False

    def _write_from_sequence(self, dna, path: str, n_bars: int) -> bool:
        """Fallback: escribe MIDI simple desde pitch_sequence y durations."""
        tpb = 480
        tempo = int(mido.bpm2tempo(getattr(dna, 'tempo_bpm', 120)))
        mid   = MidiFile(ticks_per_beat=tpb)
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage('set_tempo', tempo=tempo, time=0))

        notes = getattr(dna, 'pitch_sequence', [60, 62, 64, 65, 67])
        durs  = getattr(dna, 'durations', [1.0] * len(notes))
        vel_base = getattr(dna, 'dynamics_mean', 72)

        max_events = n_bars * 4
        for i in range(max_events):
            idx   = i % len(notes)
            note  = int(np.clip(notes[idx], 21, 108))
            dur_b = float(durs[idx % len(durs)])
            dur_t = int(dur_b * tpb)
            vel   = int(np.clip(vel_base + np.random.randint(-10, 10), 20, 120))
            track.append(Message('note_on',  note=note, velocity=vel, time=0))
            track.append(Message('note_off', note=note, velocity=0,   time=dur_t))

        try:
            mid.save(path)
            return True
        except Exception as e:
            print(f"  ERROR write_from_sequence: {e}")
            return False


# Helpers para interpolación de atributos de DNA

def _interp_attr(dna_a, dna_b, attr: str, alpha: float, do_interp: bool):
    a_val = getattr(dna_a, attr, None)
    b_val = getattr(dna_b, attr, None)
    if a_val is None:
        return b_val
    if b_val is None or not do_interp:
        return a_val
    return lerp(a_val, b_val, alpha)


def _interp_list_attr(dna_a, dna_b, attr: str, alpha: float, do_interp: bool):
    a_val = getattr(dna_a, attr, [])
    b_val = getattr(dna_b, attr, [])
    if not a_val:
        return b_val
    if not b_val or not do_interp:
        return a_val
    return lerp(a_val, b_val, alpha)


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DE MORPHING
# ══════════════════════════════════════════════════════════════════════════════

ALL_DIMS = ['pitch', 'rhythm', 'harmony', 'emotion', 'dynamics', 'tempo']


def run_morph(path_a: str, path_b: str, args) -> list:
    """
    Ejecuta el morphing y devuelve la lista de paths generados.
    """
    steps       = args.steps
    mode        = args.mode
    dims        = args.dims or ALL_DIMS
    n_bars      = args.bars
    output_dir  = args.output_dir
    prefix      = args.prefix
    verbose     = args.verbose

    os.makedirs(output_dir, exist_ok=True)

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print(f"║  MORPHER v1.0 — {steps} pasos | modo: {mode:<26}║")
    print(f"║  A: {os.path.basename(path_a)[:52]:<52}║")
    print(f"║  B: {os.path.basename(path_b)[:52]:<52}║")
    print(f"║  Dims: {', '.join(dims):<55}║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    # Extraer DNAs
    print("  Extrayendo DNA de A...")
    if DNA_OK:
        dna_a = UnifiedDNA(path_a)
        dna_a.extract(verbose=verbose)
    else:
        dna_a = SimpleDNA(path_a)
        dna_a.extract()

    print("  Extrayendo DNA de B...")
    if DNA_OK:
        dna_b = UnifiedDNA(path_b)
        dna_b.extract(verbose=verbose)
    else:
        dna_b = SimpleDNA(path_b)
        dna_b.extract()

    # Determinar n_bars
    if n_bars is None:
        bars_a = getattr(dna_a, 'n_bars', 8) or 8
        bars_b = getattr(dna_b, 'n_bars', 8) or 8
        n_bars = max(bars_a, bars_b)
        if verbose:
            print(f"  n_bars auto: {n_bars} (A={bars_a}, B={bars_b})")

    # Generar alphas según modo
    t_values = np.linspace(0, 1, steps)
    alphas   = [_interp_alpha(t, mode) for t in t_values]

    output_paths = []
    morph_data   = []

    for i, (t, alpha) in enumerate(zip(t_values, alphas)):
        is_a = (i == 0)
        is_b = (i == steps - 1)

        if is_a:
            fname  = os.path.join(output_dir, f"{prefix}_{i:03d}_A.mid")
            label  = "A (original)"
            shutil.copy(path_a, fname)
            success = True
        elif is_b:
            fname  = os.path.join(output_dir, f"{prefix}_{i:03d}_B.mid")
            label  = "B (original)"
            shutil.copy(path_b, fname)
            success = True
        else:
            fname  = os.path.join(output_dir, f"{prefix}_{i:03d}_t{alpha:.2f}.mid")
            label  = f"t={t:.2f} α={alpha:.2f}"
            morphed = MorphedDNA(dna_a, dna_b, alpha, dims, n_bars)
            success = morphed.write_midi(fname)

        if success:
            output_paths.append(fname)
            morph_data.append({'step': i, 't': float(t), 'alpha': float(alpha),
                                'file': fname, 'label': label})
            print(f"  [{i+1:02d}/{steps}] {label} → {os.path.basename(fname)}")
        else:
            print(f"  [{i+1:02d}/{steps}] ERROR generando {fname}")

    # Guardar catálogo de datos
    curves_path = os.path.join(output_dir, f"{prefix}_curves.json")
    with open(curves_path, 'w') as f:
        json.dump({'steps': morph_data, 'dims': dims, 'mode': mode,
                   'n_bars': n_bars, 'source_a': path_a, 'source_b': path_b},
                  f, indent=2)
    print(f"\n  Datos guardados: {curves_path}")

    # Catálogo MIDI
    if args.catalog:
        cat_path = os.path.join(output_dir, f"{prefix}_catalog.mid")
        _build_catalog(output_paths, cat_path)
        print(f"  Catálogo MIDI: {cat_path}")

    # Plot
    if args.plot and MATPLOTLIB_OK:
        _plot_morph(morph_data, dna_a, dna_b, alphas, dims)

    # Reproducir
    if args.listen and PYGAME_OK:
        _listen_morph(output_paths, getattr(args, 'play_seconds', 8))

    return output_paths


def _build_catalog(midi_paths: list, output_path: str):
    """Concatena todos los MIDIs en uno."""
    if not MIDO_OK or not midi_paths:
        return
    try:
        catalog = MidiFile()
        tpb = 480
        track = MidiTrack()
        catalog.tracks.append(track)
        catalog.ticks_per_beat = tpb

        for midi_path in midi_paths:
            try:
                mid = MidiFile(midi_path)
                for msg in mido.merge_tracks(mid.tracks):
                    if not msg.is_meta:
                        track.append(msg.copy())
                    elif msg.type == 'set_tempo':
                        track.append(msg.copy())
                # Silencio breve entre secciones
                track.append(Message('note_off', note=60, velocity=0, time=tpb))
            except Exception as e:
                print(f"  [catalog] Error con {midi_path}: {e}")

        track.append(MetaMessage('end_of_track', time=0))
        catalog.save(output_path)
    except Exception as e:
        print(f"  ERROR construyendo catálogo: {e}")


def _plot_morph(morph_data: list, dna_a, dna_b, alphas: list, dims: list):
    """Visualiza las curvas de morphing."""
    fig, axes = plt.subplots(len(dims), 1, figsize=(14, 2.5 * len(dims)))
    fig.patch.set_facecolor('#0d1117')
    if len(dims) == 1:
        axes = [axes]

    steps = len(morph_data)
    xs    = [d['alpha'] for d in morph_data]

    curve_attrs = {
        'pitch':    ('pitch_sequence', 'Pitch medio', lambda v: np.mean(v) if v else 60),
        'rhythm':   ('durations',      'Duración media', lambda v: np.mean(v) if v else 1.0),
        'harmony':  ('harmony_complexity', 'Complejidad armónica', lambda v: v),
        'emotion':  ('tension_curve',  'Tensión media', lambda v: np.mean(v) if v else 0.5),
        'dynamics': ('dynamics_mean',  'Dinámica media', lambda v: v),
        'tempo':    ('tempo_bpm',       'Tempo (BPM)',   lambda v: v),
    }

    for ax, dim in zip(axes, dims):
        ax.set_facecolor('#161b22')
        if dim in curve_attrs:
            attr, label, extract = curve_attrs[dim]
            try:
                val_a = extract(getattr(dna_a, attr, None) or 0)
                val_b = extract(getattr(dna_b, attr, None) or 0)
                ys_linear  = [lerp(val_a, val_b, a) for a in alphas]
                ys_actual  = [lerp(val_a, val_b, _interp_alpha(a, 'sigmoid')) for a in alphas]
                ax.plot(xs, ys_linear, '--', color='#444', linewidth=1, label='lineal')
                ax.plot(xs, ys_actual, color='#3b82f6', linewidth=2, label='actual')
                ax.axhline(val_a, color='#22c55e', linestyle=':', alpha=0.5, linewidth=1)
                ax.axhline(val_b, color='#f97316', linestyle=':', alpha=0.5, linewidth=1)
                ax.set_ylabel(label, color='#8b949e', fontsize=8)
            except Exception:
                pass
        ax.set_xlim(-0.05, 1.05)
        ax.tick_params(colors='#8b949e', labelsize=7)
        ax.spines[:].set_color('#30363d')
        ax.grid(True, color='#21262d', linewidth=0.5)
        ax.legend(fontsize=7, facecolor='#21262d', labelcolor='white')

    fig.suptitle('MORPHER — Curvas de interpolación (verde=A, naranja=B)',
                 color='white', fontsize=11)
    plt.tight_layout()
    plt.show()


def _listen_morph(midi_paths: list, play_seconds: int = 8):
    """Reproduce la secuencia de MIDIs."""
    if not PYGAME_OK:
        return
    print(f"\n  Reproduciendo secuencia ({len(midi_paths)} pasos)...")
    for i, path in enumerate(midi_paths):
        print(f"  [{i+1}/{len(midi_paths)}] {os.path.basename(path)}")
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            start = time.time()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
                if time.time() - start >= play_seconds:
                    pygame.mixer.music.stop()
                    break
        except Exception as e:
            print(f"    [play] {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE Y MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='MORPHER v1.0 — Morphing gradual entre dos obras MIDI')
    p.add_argument('source_a', help='MIDI fuente A')
    p.add_argument('source_b', help='MIDI fuente B')
    p.add_argument('--steps',        type=int,   default=6,
                   help='Número total de pasos incluyendo A y B (default: 6)')
    p.add_argument('--mode',         default='sigmoid',
                   choices=['linear', 'sigmoid', 'exponential', 'sinusoidal', 'step'])
    p.add_argument('--dims',         nargs='+', default=None,
                   choices=ALL_DIMS,
                   help='Dimensiones a interpolar (default: todas)')
    p.add_argument('--bars',         type=int,   default=None)
    p.add_argument('--catalog',      action='store_true')
    p.add_argument('--plot',         action='store_true')
    p.add_argument('--listen',       action='store_true')
    p.add_argument('--play-seconds', type=int,   default=8, dest='play_seconds')
    p.add_argument('--output-dir',   default='./morph_out', dest='output_dir')
    p.add_argument('--prefix',       default='morph')
    p.add_argument('--verbose',      action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.source_a):
        print(f"ERROR: {args.source_a} no encontrado.")
        sys.exit(1)
    if not os.path.exists(args.source_b):
        print(f"ERROR: {args.source_b} no encontrado.")
        sys.exit(1)
    if args.steps < 2:
        print("ERROR: --steps debe ser al menos 2.")
        sys.exit(1)

    output_paths = run_morph(args.source_a, args.source_b, args)
    print(f"\n  ✓ Morphing completado. {len(output_paths)} archivos en: {args.output_dir}")


if __name__ == '__main__':
    main()
