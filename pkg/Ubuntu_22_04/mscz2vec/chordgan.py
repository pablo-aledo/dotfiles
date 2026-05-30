#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CHORDGAN  v1.0                                       ║
║       Transferencia de estilo MIDI condicionada por chroma (sin TF)          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CONCEPTO                                                                    ║
║    Reimplementación del paper «ChordGAN» (Lu & Dubnov, UCSD 2021) sin       ║
║    TensorFlow. Sustituye la GAN original por un generador condicional        ║
║    basado en Ridge/KNN que aprende a mapear vectores chroma → piano rolls.  ║
║                                                                              ║
║    El condicionamiento es idéntico al original:                              ║
║      • El generador recibe el chroma del MIDI a transformar (12D por paso)  ║
║      • Produce un piano roll con el «estilo» aprendido del corpus A          ║
║    Además se añade pérdida de ciclo (F(G(x)) ≈ x) mediante un segundo       ║
║    mapper inverso, igual que cycle_gan_style_transfer.py.                   ║
║                                                                              ║
║  REPRESENTACIÓN                                                              ║
║    Piano roll: notas MIDI 24-101 (78 notas, C1–D#7), binarizado             ║
║    Canal de ritmo: onset detection (78D extra) → vector de 156D             ║
║    Segmentación: ventanas de num_steps pasos a fs=8 step/s                 ║
║    Chroma: 12 clases por ventana, extraído del propio piano roll            ║
║                                                                              ║
║  COMANDOS                                                                    ║
║    train      — Aprende el mapeo chroma→roll de un corpus de MIDIs          ║
║    transfer   — Aplica el estilo aprendido a un MIDI nuevo                   ║
║    analyze    — Estadísticas de chroma de un corpus o MIDI concreto         ║
║    info       — Muestra los metadatos de un modelo .cgan.pkl                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO — TRAIN                                                                 ║
║                                                                              ║
║    python chordgan.py train --corpus midis_pop/ --model pop.cgan.pkl        ║
║                                                                              ║
║    python chordgan.py train                                                  ║
║        --corpus  midis_pop/                                                  ║
║        --model   pop.cgan.pkl                                                ║
║        --steps   4                                                           ║
║        --solver  ridge                                                       ║
║        --alpha   1.0                                                         ║
║        --lambda-cycle 10.0                                                   ║
║        --iters   50                                                          ║
║        --verbose                                                             ║
║                                                                              ║
║    Opciones de train:                                                        ║
║      --corpus DIR       Carpeta con MIDIs del corpus a aprender             ║
║      --model  FILE      Ruta del modelo a guardar [default: chordgan.pkl]   ║
║      --steps  N         Pasos por ventana de entrenamiento [default: 4]     ║
║      --solver STR       ridge | knn [default: ridge]                        ║
║      --alpha  F         Regularización Ridge [default: 1.0]                 ║
║      --k      N         Vecinos para KNN [default: 5]                       ║
║      --lambda-cycle F   Peso de la pérdida de ciclo [default: 10.0]        ║
║      --iters  N         Iteraciones de refinamiento [default: 50]           ║
║      --seed   N         Semilla aleatoria [default: 42]                     ║
║      --verbose          Mostrar pérdidas por iteración                      ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO — TRANSFER                                                              ║
║                                                                              ║
║    python chordgan.py transfer entrada.mid                                   ║
║        --model pop.cgan.pkl --output resultado.mid                           ║
║                                                                              ║
║    python chordgan.py transfer entrada.mid                                   ║
║        --model pop.cgan.pkl                                                  ║
║        --intensity 0.7                                                       ║
║        --bpm 120                                                             ║
║        --output resultado.mid                                                ║
║        --verbose                                                             ║
║                                                                              ║
║    Opciones de transfer:                                                     ║
║      --model     FILE   Modelo entrenado (.cgan.pkl)                        ║
║      --output    FILE   MIDI de salida [default: chordgan_out.mid]          ║
║      --intensity F      0=sin cambio, 1=transformación total [default: 1.0] ║
║      --bpm       F      Tempo del MIDI de salida [default: 120.0]           ║
║      --program   N      Programa GM del instrumento [default: 0 = piano]    ║
║      --threshold F      Umbral de activación del roll [default: 0.35]       ║
║      --seed      N      Semilla [default: 42]                               ║
║      --verbose          Detalle del proceso                                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO — ANALYZE                                                               ║
║                                                                              ║
║    python chordgan.py analyze --input midis_pop/           # corpus          ║
║    python chordgan.py analyze --input cancion.mid          # fichero         ║
║    python chordgan.py analyze --input midis_pop/ --plot    # con ASCII plot  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO — INFO                                                                  ║
║                                                                              ║
║    python chordgan.py info --model pop.cgan.pkl                              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS: mido, numpy, scikit-learn                                     ║
║  OPCIONALES:   pretty_midi (para info extendida de corpus)                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import glob
import os
import pickle
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

# ── Imports opcionales ────────────────────────────────────────────────────────
try:
    import mido
except ImportError:
    print("ERROR: mido no instalado. Ejecuta: pip install mido")
    sys.exit(1)

try:
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    print("ERROR: scikit-learn no instalado. Ejecuta: pip install scikit-learn")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

MIDI_LO   = 24    # C1
MIDI_HI   = 102   # D#7 (excluido, 78 notas)
N_NOTES   = MIDI_HI - MIDI_LO          # 78
ROLL_DIM  = N_NOTES * 2                # notas + canal de ritmo = 156
CHROMA_DIM = 12
FS        = 8     # steps por segundo (igual que el paper original)

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSIÓN MIDI ↔ PIANO ROLL
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_roll(midi_path: str, fs: int = FS) -> np.ndarray:
    """
    Convierte un fichero MIDI en un piano roll binario con canal de ritmo.
    Retorna array (T, ROLL_DIM) = (T, 156).
    La primera mitad [0:78] son notas activas, la segunda [78:156] son onsets.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo leer {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo = 500_000  # 120 BPM por defecto

    # Acumular eventos de nota con tiempo absoluto en segundos
    events = []  # (time_sec, note, velocity)
    for track in mid.tracks:
        abs_ticks = 0
        abs_sec   = 0.0
        prev_ticks = 0
        cur_tempo  = tempo
        for msg in track:
            abs_ticks += msg.time
            # Convertir delta ticks a segundos
            abs_sec += mido.tick2second(msg.time, tpb, cur_tempo)
            if msg.type == 'set_tempo':
                cur_tempo = msg.tempo
            elif msg.type in ('note_on', 'note_off'):
                vel = msg.velocity if msg.type == 'note_on' else 0
                events.append((abs_sec, msg.note, vel))

    if not events:
        return np.zeros((1, ROLL_DIM), dtype=np.float32)

    # Calcular duración total y número de frames
    max_time = max(e[0] for e in events) + 0.5
    n_frames = max(1, int(np.ceil(max_time * fs)))

    notes_roll  = np.zeros((N_NOTES, n_frames), dtype=np.float32)
    rhythm_roll = np.zeros((N_NOTES, n_frames), dtype=np.float32)
    active = {}  # note -> (frame_start, frame_on)

    for t_sec, note, vel in sorted(events):
        if note < MIDI_LO or note >= MIDI_HI:
            continue
        idx = note - MIDI_LO
        frame = int(t_sec * fs)
        frame = min(frame, n_frames - 1)

        if vel > 0:
            active[note] = frame
            if frame < n_frames:
                rhythm_roll[idx, frame] = 1.0
        else:
            if note in active:
                f0 = active.pop(note)
                for f in range(f0, min(frame + 1, n_frames)):
                    notes_roll[idx, f] = 1.0

    # Combinar: (notas, T) + (ritmo, T) → (T, 156)
    combined = np.concatenate([notes_roll, rhythm_roll], axis=0)  # (156, T)
    return combined.T.astype(np.float32)  # (T, 156)


def roll_to_midi(roll: np.ndarray, output_path: str,
                 bpm: float = 120.0, program: int = 0,
                 fs: int = FS, threshold: float = 0.35) -> None:
    """
    Convierte un piano roll (T, ROLL_DIM) en un fichero MIDI.
    Usa el canal de notas [0:78]; el canal de ritmo se ignora en la salida.
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo_us = int(60_000_000 / bpm)
    tpb = 480
    mid.ticks_per_beat = tpb

    track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
    track.append(mido.Message('program_change', program=program, time=0))

    notes_roll = (roll[:, :N_NOTES] >= threshold).astype(np.uint8)  # (T, 78)

    # Detectar eventos note_on / note_off por cambios
    padded = np.pad(notes_roll, [(1, 1), (0, 0)], 'constant')
    diff   = np.diff(padded.astype(np.int8), axis=0)  # (T+1, 78)

    events = []  # (frame, note_midi, type)
    for frame, note_idx in zip(*np.where(diff == 1)):
        events.append((int(frame), note_idx + MIDI_LO, 'on'))
    for frame, note_idx in zip(*np.where(diff == -1)):
        events.append((int(frame), note_idx + MIDI_LO, 'off'))

    events.sort(key=lambda e: (e[0], 0 if e[2] == 'off' else 1))

    ticks_per_step = int(tpb * bpm / 60.0 / fs)
    prev_tick = 0

    for frame, note, etype in events:
        tick = frame * ticks_per_step
        delta = tick - prev_tick
        vel   = 80 if etype == 'on' else 0
        track.append(mido.Message('note_on', note=note, velocity=vel, time=max(0, delta)))
        prev_tick = tick

    # Silencio final de 1 compás
    track.append(mido.MetaMessage('end_of_track', time=tpb * 4))
    mid.save(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE CHROMA
# ══════════════════════════════════════════════════════════════════════════════

def roll_to_chroma(roll: np.ndarray) -> np.ndarray:
    """
    Extrae chroma (T, 12) del piano roll (T, ROLL_DIM).
    Cada clase de pitch acumula la presencia de todas las octavas.
    """
    notes = roll[:, :N_NOTES]  # (T, 78)
    chroma = np.zeros((notes.shape[0], CHROMA_DIM), dtype=np.float32)
    for i in range(N_NOTES):
        midi_note = i + MIDI_LO
        pc = midi_note % 12
        chroma[:, pc] += notes[:, i]
    # Normalizar por frame
    norm = chroma.sum(axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    chroma /= norm
    return chroma


def chroma_global(roll: np.ndarray) -> np.ndarray:
    """Chroma global promediado sobre toda la duración (12,)."""
    return roll_to_chroma(roll).mean(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTACIÓN EN VENTANAS
# ══════════════════════════════════════════════════════════════════════════════

def segment_roll(roll: np.ndarray, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide el roll (T, 156) en ventanas de num_steps pasos.
    Retorna:
      X_roll   (N, num_steps * ROLL_DIM)   — ventanas aplanadas
      X_chroma (N, num_steps * CHROMA_DIM) — chroma de cada ventana aplanado
    """
    T = roll.shape[0]
    n_windows = T // num_steps
    if n_windows == 0:
        return np.zeros((0, num_steps * ROLL_DIM)), np.zeros((0, num_steps * CHROMA_DIM))

    roll_trunc   = roll[:n_windows * num_steps]
    chroma_full  = roll_to_chroma(roll_trunc)

    roll_w   = roll_trunc.reshape(n_windows, num_steps * ROLL_DIM)
    chroma_w = chroma_full.reshape(n_windows, num_steps * CHROMA_DIM)
    return roll_w.astype(np.float32), chroma_w.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def load_corpus(directory: str, num_steps: int,
                verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga todos los MIDIs de *directory* y los segmenta.
    Retorna (X_roll, X_chroma) con todas las ventanas concatenadas.
    """
    patterns = [
        os.path.join(directory, '*.mid'),
        os.path.join(directory, '*.midi'),
        os.path.join(directory, '**/*.mid'),
        os.path.join(directory, '**/*.midi'),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))

    if not files:
        print(f"  AVISO: no se encontraron MIDIs en {directory}")
        return np.zeros((0, num_steps * ROLL_DIM)), np.zeros((0, num_steps * CHROMA_DIM))

    all_rolls, all_chromas = [], []
    ok = 0
    for f in files:
        try:
            roll = midi_to_roll(f)
            rw, cw = segment_roll(roll, num_steps)
            if rw.shape[0] > 0:
                all_rolls.append(rw)
                all_chromas.append(cw)
                ok += 1
                if verbose:
                    print(f"    ✓ {os.path.basename(f):40s}  {rw.shape[0]} ventanas")
        except Exception as e:
            if verbose:
                print(f"    ✗ {os.path.basename(f):40s}  {e}")

    print(f"  {ok}/{len(files)} MIDIs cargados")
    if not all_rolls:
        return np.zeros((0, num_steps * ROLL_DIM)), np.zeros((0, num_steps * CHROMA_DIM))

    X_roll   = np.concatenate(all_rolls,   axis=0)
    X_chroma = np.concatenate(all_chromas, axis=0)
    print(f"  Total ventanas: {X_roll.shape[0]}  |  dim roll: {X_roll.shape[1]}  |  dim chroma: {X_chroma.shape[1]}")
    return X_roll, X_chroma


# ══════════════════════════════════════════════════════════════════════════════
#  MAPPER (generador condicional)
# ══════════════════════════════════════════════════════════════════════════════

def _build_mapper(solver: str, alpha: float, k: int):
    """Construye el regresor base según el solver elegido."""
    if solver == 'knn':
        return KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)
    return Ridge(alpha=alpha, fit_intercept=True)


class ChordGANModel:
    """
    Modelo chroma-condicionado con pérdida de ciclo.

    G: chroma → roll  (generador directo)
    F: roll  → chroma (inverso aproximado para pérdida de ciclo)
    """

    def __init__(self, solver='ridge', alpha=1.0, k=5,
                 lambda_cycle=10.0, num_steps=4, iters=50):
        self.solver       = solver
        self.alpha        = alpha
        self.k            = k
        self.lambda_cycle = lambda_cycle
        self.num_steps    = num_steps
        self.iters        = iters

        self.G  = None   # chroma → roll
        self.F  = None   # roll   → chroma
        self.scaler_c = StandardScaler()
        self.scaler_r = StandardScaler()

        self.corpus_name    = ''
        self.n_train_files  = 0
        self.n_train_wins   = 0
        self.losses_history = []

    # ── Entrenamiento ─────────────────────────────────────────────────────────

    def fit(self, X_roll: np.ndarray, X_chroma: np.ndarray,
            verbose: bool = False) -> None:
        """
        Entrena G y F con refinamiento iterativo y pérdida de ciclo.
        X_roll   (N, num_steps * ROLL_DIM)
        X_chroma (N, num_steps * CHROMA_DIM)
        """
        Xc = self.scaler_c.fit_transform(X_chroma)
        Xr = self.scaler_r.fit_transform(X_roll)

        # Entrenamiento inicial
        self.G = _build_mapper(self.solver, self.alpha, self.k)
        self.F = _build_mapper(self.solver, self.alpha, self.k)
        self.G.fit(Xc, Xr)
        self.F.fit(Xr, Xc)

        if verbose:
            print(f"  {'Iter':>5}  {'cyc_G':>10}  {'cyc_F':>10}")

        # Refinamiento iterativo con pérdida de ciclo
        for it in range(self.iters):
            G_pred = self.G.predict(Xc)   # roll sintético
            F_pred = self.F.predict(Xr)   # chroma sintético

            # Pérdida de ciclo: cuánto se aleja la reconstrucción
            cyc_G = np.mean((self.F.predict(G_pred) - Xc) ** 2)
            cyc_F = np.mean((self.G.predict(F_pred) - Xr) ** 2)

            self.losses_history.append({'iter': it, 'cyc_G': float(cyc_G), 'cyc_F': float(cyc_F)})

            if verbose and (it % max(1, self.iters // 10) == 0 or it == self.iters - 1):
                print(f"  {it:>5}  {cyc_G:>10.4f}  {cyc_F:>10.4f}")

            # Reentrenar con residuos ponderados por la pérdida de ciclo
            lc = self.lambda_cycle
            # Augmentar X_chroma con los residuos de ciclo como regularización implícita
            F_of_G = self.F.predict(G_pred)
            residual_c = Xc - F_of_G
            Xc_aug = Xc + (lc / (lc + 1.0)) * residual_c

            G_of_F = self.G.predict(F_pred)
            residual_r = Xr - G_of_F
            Xr_aug = Xr + (lc / (lc + 1.0)) * residual_r

            self.G = _build_mapper(self.solver, self.alpha, self.k)
            self.F = _build_mapper(self.solver, self.alpha, self.k)
            self.G.fit(Xc_aug, Xr)
            self.F.fit(Xr_aug, Xc)

    # ── Transformación ────────────────────────────────────────────────────────

    def transform(self, roll: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """
        Aplica el estilo aprendido a un piano roll (T, ROLL_DIM).
        intensity: 0=original, 1=completamente transformado.
        Retorna piano roll transformado (T, ROLL_DIM).
        """
        T = roll.shape[0]
        rw, cw = segment_roll(roll, self.num_steps)
        if rw.shape[0] == 0:
            return roll

        Xc = self.scaler_c.transform(cw)
        G_pred_scaled = self.G.predict(Xc)          # (N, roll_dim_scaled)
        G_pred = self.scaler_r.inverse_transform(G_pred_scaled)
        G_pred = np.clip(G_pred, 0.0, 1.0)

        # Mezcla por intensidad
        if intensity < 1.0:
            G_pred = intensity * G_pred + (1.0 - intensity) * rw

        # Reensamblar a (T, ROLL_DIM)
        n_windows = rw.shape[0]
        out_roll = G_pred.reshape(n_windows * self.num_steps, ROLL_DIM)

        # Rellenar si el roll original era más largo
        if T > out_roll.shape[0]:
            pad = np.zeros((T - out_roll.shape[0], ROLL_DIM), dtype=np.float32)
            out_roll = np.concatenate([out_roll, pad], axis=0)
        return out_roll[:T]

    # ── Serialización ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"  Modelo guardado: {path}")

    @staticmethod
    def load(path: str) -> 'ChordGANModel':
        with open(path, 'rb') as f:
            model = pickle.load(f)
        if not isinstance(model, ChordGANModel):
            raise ValueError(f"{path} no contiene un ChordGANModel válido.")
        return model


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    print("═" * 65)
    print("  CHORDGAN v1.0 — TRAIN")
    print("═" * 65)
    print(f"  Corpus       : {args.corpus}")
    print(f"  Modelo       : {args.model}")
    print(f"  Steps/ventana: {args.steps}")
    print(f"  Solver       : {args.solver}")
    print(f"  Alpha (Ridge): {args.alpha}")
    print(f"  K (KNN)      : {args.k}")
    print(f"  λ ciclo      : {args.lambda_cycle}")
    print(f"  Iteraciones  : {args.iters}")
    print(f"  Semilla      : {args.seed}")
    print()

    np.random.seed(args.seed)

    if not os.path.isdir(args.corpus):
        print(f"ERROR: {args.corpus} no es un directorio válido.")
        sys.exit(1)

    print("[1/3] Cargando corpus…")
    t0 = time.time()
    X_roll, X_chroma = load_corpus(args.corpus, args.steps, verbose=args.verbose)
    if X_roll.shape[0] == 0:
        print("ERROR: corpus vacío, no se puede entrenar.")
        sys.exit(1)

    print(f"\n[2/3] Entrenando modelo (solver={args.solver}, iters={args.iters})…")
    model = ChordGANModel(
        solver       = args.solver,
        alpha        = args.alpha,
        k            = args.k,
        lambda_cycle = args.lambda_cycle,
        num_steps    = args.steps,
        iters        = args.iters,
    )
    model.corpus_name   = os.path.basename(args.corpus.rstrip('/'))
    model.n_train_files = X_roll.shape[0]
    model.n_train_wins  = X_roll.shape[0]
    model.fit(X_roll, X_chroma, verbose=args.verbose)

    print(f"\n[3/3] Guardando modelo…")
    model.save(args.model)

    if model.losses_history:
        last = model.losses_history[-1]
        print(f"\n  Pérdida ciclo G (final) : {last['cyc_G']:.4f}")
        print(f"  Pérdida ciclo F (final) : {last['cyc_F']:.4f}")

    elapsed = time.time() - t0
    print(f"\n  Tiempo total: {elapsed:.1f}s")
    print("═" * 65)
    print(f"  Modelo listo: {args.model}")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: transfer
# ══════════════════════════════════════════════════════════════════════════════

def cmd_transfer(args):
    print("═" * 65)
    print("  CHORDGAN v1.0 — TRANSFER")
    print("═" * 65)
    print(f"  Entrada    : {args.input}")
    print(f"  Modelo     : {args.model}")
    print(f"  Salida     : {args.output}")
    print(f"  Intensidad : {args.intensity}")
    print(f"  BPM        : {args.bpm}")
    print(f"  Programa   : {args.program} (GM)")
    print(f"  Umbral     : {args.threshold}")
    print()

    np.random.seed(args.seed)

    # Cargar modelo
    if not os.path.isfile(args.model):
        print(f"ERROR: modelo no encontrado: {args.model}")
        sys.exit(1)

    print("[1/3] Cargando modelo…")
    model = ChordGANModel.load(args.model)
    print(f"  Corpus origen : {model.corpus_name}")
    print(f"  Steps/ventana : {model.num_steps}")
    print(f"  Ventanas entrenamiento: {model.n_train_wins}")

    # Cargar MIDI de entrada
    print("\n[2/3] Procesando MIDI de entrada…")
    if not os.path.isfile(args.input):
        print(f"ERROR: fichero no encontrado: {args.input}")
        sys.exit(1)

    roll = midi_to_roll(args.input)
    print(f"  Frames totales: {roll.shape[0]}")

    chroma_in = chroma_global(roll)
    dominant_pc = int(np.argmax(chroma_in))
    print(f"  Clase de pitch dominante: {NOTE_NAMES[dominant_pc]}")

    if args.verbose:
        rw, _ = segment_roll(roll, model.num_steps)
        print(f"  Ventanas generables: {rw.shape[0]}")

    # Transformar
    print("\n[3/3] Aplicando transferencia de estilo…")
    out_roll = model.transform(roll, intensity=args.intensity)

    roll_to_midi(out_roll, args.output,
                 bpm=args.bpm,
                 program=args.program,
                 threshold=args.threshold)

    # Estadísticas de salida
    if args.verbose:
        active_frames = (out_roll[:, :N_NOTES] >= args.threshold).any(axis=1).sum()
        total_frames  = out_roll.shape[0]
        density = active_frames / max(1, total_frames)
        print(f"  Frames con actividad: {active_frames}/{total_frames}  ({density:.1%})")

    print("\n" + "═" * 65)
    print(f"  Resultado guardado: {args.output}")
    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: analyze
# ══════════════════════════════════════════════════════════════════════════════

def _analyze_roll(roll: np.ndarray, label: str, plot: bool) -> dict:
    """Analiza un piano roll y devuelve estadísticas."""
    chroma = chroma_global(roll)
    dominant_pc  = int(np.argmax(chroma))
    notes_active = (roll[:, :N_NOTES] > 0).sum(axis=1)  # notas activas por frame
    density = (notes_active > 0).mean()

    stats = {
        'label'       : label,
        'frames'      : roll.shape[0],
        'density'     : float(density),
        'avg_polyphony': float(notes_active[notes_active > 0].mean()) if density > 0 else 0.0,
        'dominant_pc' : NOTE_NAMES[dominant_pc],
        'chroma'      : chroma.tolist(),
    }

    print(f"\n  ── {label}")
    print(f"     Frames        : {stats['frames']}")
    print(f"     Densidad      : {stats['density']:.1%}")
    print(f"     Polifonía prom: {stats['avg_polyphony']:.1f} notas/frame")
    print(f"     PC dominante  : {stats['dominant_pc']}")

    if plot:
        # ASCII bar chart del perfil chroma
        print("     Perfil chroma :")
        max_v = max(chroma) if max(chroma) > 0 else 1.0
        for i, (name, val) in enumerate(zip(NOTE_NAMES, chroma)):
            bar_len = int(val / max_v * 30)
            bar = '█' * bar_len
            print(f"     {name:>3}  {bar:<30}  {val:.3f}")

    return stats


def cmd_analyze(args):
    print("═" * 65)
    print("  CHORDGAN v1.0 — ANALYZE")
    print("═" * 65)
    print(f"  Entrada: {args.input}")
    print()

    target = args.input

    if os.path.isfile(target) and target.lower().endswith(('.mid', '.midi')):
        roll = midi_to_roll(target)
        _analyze_roll(roll, os.path.basename(target), args.plot)

    elif os.path.isdir(target):
        patterns = [
            os.path.join(target, '*.mid'),
            os.path.join(target, '*.midi'),
            os.path.join(target, '**/*.mid'),
            os.path.join(target, '**/*.midi'),
        ]
        files = []
        for p in patterns:
            files.extend(glob.glob(p, recursive=True))
        files = sorted(set(files))

        if not files:
            print(f"  AVISO: no se encontraron MIDIs en {target}")
            sys.exit(1)

        print(f"  {len(files)} MIDIs encontrados")
        all_chroma = []
        for f in files:
            try:
                roll = midi_to_roll(f)
                stats = _analyze_roll(roll, os.path.basename(f), args.plot)
                all_chroma.append(stats['chroma'])
            except Exception as e:
                print(f"  ✗ {os.path.basename(f)}: {e}")

        if all_chroma:
            centroid = np.mean(all_chroma, axis=0)
            dom_pc   = int(np.argmax(centroid))
            print(f"\n  ── CENTROIDE DEL CORPUS")
            print(f"     PC dominante del corpus: {NOTE_NAMES[dom_pc]}")
            if args.plot:
                print("     Perfil chroma promedio:")
                max_v = max(centroid) if max(centroid) > 0 else 1.0
                for name, val in zip(NOTE_NAMES, centroid):
                    bar_len = int(val / max_v * 30)
                    print(f"     {name:>3}  {'█' * bar_len:<30}  {val:.3f}")
    else:
        print(f"ERROR: {target} no es un fichero MIDI ni un directorio válido.")
        sys.exit(1)

    print("\n" + "═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: info
# ══════════════════════════════════════════════════════════════════════════════

def cmd_info(args):
    print("═" * 65)
    print("  CHORDGAN v1.0 — INFO")
    print("═" * 65)

    if not os.path.isfile(args.model):
        print(f"ERROR: {args.model} no encontrado.")
        sys.exit(1)

    try:
        model = ChordGANModel.load(args.model)
    except Exception as e:
        print(f"ERROR cargando modelo: {e}")
        sys.exit(1)

    size_kb = os.path.getsize(args.model) / 1024
    print(f"  Fichero          : {args.model}  ({size_kb:.1f} KB)")
    print(f"  Corpus origen    : {model.corpus_name}")
    print(f"  Ventanas entreno : {model.n_train_wins}")
    print(f"  Steps/ventana    : {model.num_steps}")
    print(f"  Solver           : {model.solver}")
    print(f"  Alpha (Ridge)    : {model.alpha}")
    print(f"  K (KNN)          : {model.k}")
    print(f"  λ ciclo          : {model.lambda_cycle}")
    print(f"  Iteraciones      : {model.iters}")
    print(f"  Dim chroma input : {model.num_steps * CHROMA_DIM}")
    print(f"  Dim roll output  : {model.num_steps * ROLL_DIM}")

    if model.losses_history:
        last  = model.losses_history[-1]
        first = model.losses_history[0]
        print(f"\n  Pérdida ciclo G  : {first['cyc_G']:.4f} → {last['cyc_G']:.4f}")
        print(f"  Pérdida ciclo F  : {first['cyc_F']:.4f} → {last['cyc_F']:.4f}")

    print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='chordgan',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            CHORDGAN v1.0 — Transferencia de estilo MIDI condicionada por chroma

            Flujo típico:
              1. analyze   — explorar el corpus antes de entrenar
              2. train     — aprender el estilo de un corpus
              3. transfer  — aplicar el estilo a un MIDI concreto
        """),
    )

    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── train ──────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Aprende el estilo de un corpus de MIDIs')
    p.add_argument('--corpus',        required=True,  metavar='DIR',
                   help='Carpeta con los MIDIs del corpus')
    p.add_argument('--model',         default='chordgan.pkl', metavar='FILE',
                   help='Ruta del modelo a guardar [default: chordgan.pkl]')
    p.add_argument('--steps',         type=int,   default=4,   metavar='N',
                   help='Pasos por ventana de entrenamiento [default: 4]')
    p.add_argument('--solver',        default='ridge', choices=['ridge', 'knn'],
                   help='Regresor base [default: ridge]')
    p.add_argument('--alpha',         type=float, default=1.0, metavar='F',
                   help='Regularización Ridge [default: 1.0]')
    p.add_argument('--k',             type=int,   default=5,   metavar='N',
                   help='Vecinos para KNN [default: 5]')
    p.add_argument('--lambda-cycle',  type=float, default=10.0, metavar='F',
                   help='Peso pérdida de ciclo [default: 10.0]')
    p.add_argument('--iters',         type=int,   default=50,  metavar='N',
                   help='Iteraciones de refinamiento [default: 50]')
    p.add_argument('--seed',          type=int,   default=42,  metavar='N')
    p.add_argument('--verbose',       action='store_true')
    p.set_defaults(func=cmd_train)

    # ── transfer ───────────────────────────────────────────────────────────────
    p = sub.add_parser('transfer', help='Aplica el estilo aprendido a un MIDI')
    p.add_argument('input',           metavar='FILE',
                   help='MIDI de entrada a transformar')
    p.add_argument('--model',         required=True, metavar='FILE',
                   help='Modelo entrenado (.pkl)')
    p.add_argument('--output',        default='chordgan_out.mid', metavar='FILE',
                   help='MIDI de salida [default: chordgan_out.mid]')
    p.add_argument('--intensity',     type=float, default=1.0, metavar='F',
                   help='Intensidad de la transferencia 0-1 [default: 1.0]')
    p.add_argument('--bpm',           type=float, default=120.0, metavar='F',
                   help='Tempo del MIDI de salida [default: 120.0]')
    p.add_argument('--program',       type=int,   default=0,   metavar='N',
                   help='Programa GM del instrumento [default: 0 = piano]')
    p.add_argument('--threshold',     type=float, default=0.35, metavar='F',
                   help='Umbral de activación del roll [default: 0.35]')
    p.add_argument('--seed',          type=int,   default=42,  metavar='N')
    p.add_argument('--verbose',       action='store_true')
    p.set_defaults(func=cmd_transfer)

    # ── analyze ────────────────────────────────────────────────────────────────
    p = sub.add_parser('analyze', help='Estadísticas de chroma de un MIDI o corpus')
    p.add_argument('--input',         required=True, metavar='PATH',
                   help='Fichero MIDI o carpeta con MIDIs')
    p.add_argument('--plot',          action='store_true',
                   help='Mostrar perfil chroma en ASCII')
    p.set_defaults(func=cmd_analyze)

    # ── info ───────────────────────────────────────────────────────────────────
    p = sub.add_parser('info', help='Muestra los metadatos de un modelo .pkl')
    p.add_argument('--model',         required=True, metavar='FILE',
                   help='Modelo entrenado (.pkl)')
    p.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
