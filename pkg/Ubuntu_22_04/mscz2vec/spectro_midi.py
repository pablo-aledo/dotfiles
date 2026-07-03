#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SPECTRO MIDI  v1.0                                    ║
║  Transcripción automática de espectrogramas NPZ → MIDI                       ║
║  CNN + Transformer · agnóstico de instrumento · bucle autónomo               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUBCOMANDOS                                                                 ║
║    prepare      MIDI corpus / generación procedimental → NPZ de entrenamiento║
║    train        Entrena el modelo sobre el corpus preparado                  ║
║    transcribe   NPZ / WAV → MIDI  (inferencia)                               ║
║    autotrain    Bucle autónomo: genera → entrena → evalúa → itera            ║
║    eval         Evalúa MIDI predicho vs referencia (F1, precisión, recall)   ║
║    inspect      Diagnóstico del modelo y del corpus                          ║
║    melody       NPZ/WAV → MIDI monofónico  (pico espectral, SIN ML)         ║
║    harmony      NPZ/WAV → MIDI de n voces  (picos espectrales, SIN ML)      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARQUITECTURA                                                                ║
║    Input  : espectrograma STFT  [frames × bins]  (NPZ de audio_lab.py)      ║
║    CNN    : extrae patrones armónicos locales por frame                      ║
║    Transformer encoder: captura contexto temporal (onsets, duraciones)       ║
║    Salida : activación sigmoid  [frames × 128]  (una celda por nota MIDI)   ║
║                                                                              ║
║  TAMAÑOS DE MODELO  (--model-size)                                           ║
║    small  : CNN-2 + 2 capas TR   ~2M params  CPU viable (~30min/epoch)      ║
║    medium : CNN-3 + 4 capas TR  ~15M params  GPU recomendada                ║
║    large  : CNN-4 + 8 capas TR  ~50M params  GPU necesaria                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  COHERENCIA DE FRASE  (--use-phrase-prior)                                   ║
║    Durante la inferencia, el sistema detecta segmentos espectralmente        ║
║    similares en la misma pieza (similitud coseno sobre frames STFT).         ║
║    Ante notas con probabilidad similar, se favorece la de frases conocidas.  ║
║    El corpus de frases se acumula en phrase_corpus/ y crece con autotrain.   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GENERACIÓN PROCEDIMENTAL DE MIDIS  (prepare --procedural)                  ║
║    --proc-mode  notes   notas individuales (todas las MIDI 21-108)           ║
║                 chords  tríadas y séptimas con inversiones                   ║
║                 phrases progresiones de 4-8 acordes (I-IV-V-I, ii-V-I…)    ║
║                 all     los tres modos combinados                            ║
║    --humanize   0.0-1.0 variación de velocidad y timing (0=perfecto)        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CURRICULUM LEARNING  (autotrain --curriculum)                               ║
║    Nivel 1: notas individuales, síntesis pura                                ║
║    Nivel 2: acordes de 2 notas, timbre simple                               ║
║    Nivel 3: acordes de 3-4 notas, timbre con armónicos                      ║
║    Nivel 4: frases con progresiones, humanización ligera                     ║
║    Nivel 5: frases complejas, humanización completa, audio real              ║
║    Sube de nivel cuando F1 ≥ --curriculum-threshold (default: 0.80)         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODOS SIN ML  (melody / harmony)                                           ║
║    No usan modelo ni entrenamiento: en cada frame localizan el/los bin(s)   ║
║    de mayor intensidad del espectrograma dentro del rango de piano y        ║
║    sostienen la(s) nota(s) resultante(s) mientras sigan siendo dominantes.   ║
║    melody   → 1 bin dominante  → línea monofónica                           ║
║    harmony  → n bins dominantes (--notes, típ. 3=tríada / 4=acorde 4 notas) ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS  numpy  torch  mido  soundfile  Pillow                        ║
║                audio_lab.py  (en el mismo directorio o en PATH)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  EJEMPLOS                                                                    ║
║    # Preparar corpus desde MIDIs existentes                                  ║
║    python spectro_midi.py prepare --midi-dir midis/ --out-dir corpus/       ║
║                                                                              ║
║    # Preparar corpus procedimental (acordes con humanización)                ║
║    python spectro_midi.py prepare --procedural --proc-mode chords \         ║
║        --humanize 0.3 --out-dir corpus/                                      ║
║                                                                              ║
║    # Entrenar modelo pequeño en CPU                                          ║
║    python spectro_midi.py train --corpus corpus/ --model-size small \       ║
║        --epochs 50 --model-dir model/                                        ║
║                                                                              ║
║    # Entrenar modelo grande en GPU                                           ║
║    python spectro_midi.py train --corpus corpus/ --model-size large \       ║
║        --epochs 200 --batch-size 16 --model-dir model_large/                ║
║                                                                              ║
║    # Transcribir un NPZ                                                      ║
║    python spectro_midi.py transcribe audio.npz --model-dir model/ \         ║
║        --use-phrase-prior --output audio.mid                                 ║
║                                                                              ║
║    # Bucle autónomo con curriculum                                           ║
║    python spectro_midi.py autotrain --model-dir model/ \                    ║
║        --midi-dir midis/ --cycles 20 --curriculum                            ║
║        --use-phrase-prior                                                    ║
║                                                                              ║
║    # Evaluar transcripción                                                   ║
║    python spectro_midi.py eval --predicted out.mid --reference orig.mid     ║
║                                                                              ║
║    # Melodía monofónica sin ML (pico espectral dominante)                    ║
║    python spectro_midi.py melody audio.npz --output melodia.mid            ║
║                                                                              ║
║    # Acompañamiento en tríadas sin ML (3 bins dominantes por frame)          ║
║    python spectro_midi.py harmony audio.wav --notes 3 -o acordes.mid       ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── lazy imports ──────────────────────────────────────────────────────────────

def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        sys.exit("✗  torch no encontrado. pip install torch")

def _import_mido():
    try:
        import mido
        return mido
    except ImportError:
        sys.exit("✗  mido no encontrado. pip install mido")

def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("✗  soundfile no encontrado. pip install soundfile")

def _global_tempo_map(mid, bpm_override: Optional[float] = None):
    """Fusiona los eventos set_tempo de TODAS las pistas en un único mapa
    [(tick_abs, tempo_us), ...] ordenado por tick.

    Necesario porque en un MIDI formato 1 el tempo vive en la pista
    conductora (pista 0) y las notas en otras pistas; reiniciar el tempo a
    120 bpm al empezar cada pista ignora por completo el tempo real (p.ej.
    el escrito por tempo_designer.py) en cuanto hay más de una pista.
    """
    default_tempo = int(60_000_000 / bpm_override) if bpm_override else 500000
    changes = [(0, default_tempo)]
    if not bpm_override:
        for track in mid.tracks:
            tick_abs = 0
            for msg in track:
                tick_abs += msg.time
                if msg.type == "set_tempo":
                    changes.append((tick_abs, msg.tempo))
    changes.sort(key=lambda c: c[0])
    merged = []
    for tick, tempo in changes:
        if merged and merged[-1][0] == tick:
            merged[-1] = (tick, tempo)
        else:
            merged.append((tick, tempo))
    return merged

def _tick_to_seconds_fn(mido_mod, tempo_map, tpb: int):
    """Devuelve una función tick_abs -> segundos usando el mapa de tempo global."""
    import bisect
    ticks = [t for t, _ in tempo_map]
    breakpoints = []                      # (tick, tiempo_acumulado_s, tempo_us)
    acc = 0.0
    prev_tick, prev_tempo = tempo_map[0]
    breakpoints.append((prev_tick, 0.0, prev_tempo))
    for tick, tempo in tempo_map[1:]:
        acc += mido_mod.tick2second(tick - prev_tick, tpb, prev_tempo)
        breakpoints.append((tick, acc, tempo))
        prev_tick, prev_tempo = tick, tempo

    def tick_to_seconds(tick_abs: int) -> float:
        idx = bisect.bisect_right(ticks, tick_abs) - 1
        idx = max(0, idx)
        bt, bs, btempo = breakpoints[idx]
        return bs + mido_mod.tick2second(tick_abs - bt, tpb, btempo)

    return tick_to_seconds

# ── constantes ────────────────────────────────────────────────────────────────

MIDI_MIN      = 21     # A0
MIDI_MAX      = 108    # C8
N_NOTES       = MIDI_MAX - MIDI_MIN + 1   # 88
SR            = 44100
STFT_WINDOW   = 4096
STFT_HOP      = 0.25
N_BINS        = STFT_WINDOW // 2 + 1      # 1025
FRAME_MS      = STFT_WINDOW * STFT_HOP / SR * 1000   # ≈ 11.6ms

# Progresiones de acordes (grados relativos en semitones desde la tónica)
CHORD_TYPES = {
    "maj":   [0, 4, 7],
    "min":   [0, 3, 7],
    "maj7":  [0, 4, 7, 11],
    "min7":  [0, 3, 7, 10],
    "dom7":  [0, 4, 7, 10],
    "dim":   [0, 3, 6],
    "aug":   [0, 4, 8],
    "sus2":  [0, 2, 7],
    "sus4":  [0, 5, 7],
}

PROGRESSIONS = [
    ["maj", "maj", "min", "maj"],          # I-I-vi-IV
    ["maj", "dom7", "maj", "maj"],         # I-V7-I-I
    ["maj", "min", "dom7", "maj"],         # I-ii-V-I
    ["min", "maj", "dom7", "maj"],         # vi-IV-V-I
    ["maj", "min7", "dom7", "maj"],        # I-ii7-V7-I
    ["min", "dim", "maj", "dom7"],         # i-vii°-I-V7
]


# ══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN PROCEDIMENTAL DE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _humanize(value: int, amount: float, lo: int, hi: int) -> int:
    if amount <= 0:
        return value
    delta = int(random.gauss(0, amount * 10))
    return max(lo, min(hi, value + delta))

def _humanize_time(ticks: int, amount: float, tpb: int) -> int:
    if amount <= 0:
        return ticks
    jitter = int(random.gauss(0, amount * tpb * 0.05))
    return max(0, ticks + jitter)

def _make_midi_notes(notes: List[int], duration_ticks: int, velocity: int,
                     tpb: int, humanize: float) -> "mido.MidiFile":
    mido = _import_mido()
    mid   = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    for note in notes:
        vel = _humanize(velocity, humanize, 40, 127)
        t0  = _humanize_time(0, humanize, tpb)
        track.append(mido.Message("note_on",  note=note, velocity=vel,   time=t0))
    dur = _humanize_time(duration_ticks, humanize, tpb)
    for i, note in enumerate(notes):
        t_off = dur if i == 0 else 0
        track.append(mido.Message("note_off", note=note, velocity=0, time=t_off))

    return mid

def generate_single_notes(out_dir: Path, humanize: float = 0.0,
                           duration_beats: float = 1.5) -> List[Path]:
    """Una nota por fichero, todas las notas del rango de piano."""
    mido  = _import_mido()
    tpb   = 480
    dur   = int(tpb * duration_beats)
    paths = []
    for note in range(MIDI_MIN, MIDI_MAX + 1):
        mid  = _make_midi_notes([note], dur, 80, tpb, humanize)
        path = out_dir / f"note_{note:03d}.mid"
        mid.save(str(path))
        paths.append(path)
    return paths

def generate_chords(out_dir: Path, humanize: float = 0.0,
                    duration_beats: float = 2.0) -> List[Path]:
    """Tríadas y séptimas en todas las inversiones y todas las tónicas."""
    mido  = _import_mido()
    tpb   = 480
    dur   = int(tpb * duration_beats)
    paths = []
    for root in range(48, 73):   # C3 a C5
        for cname, intervals in CHORD_TYPES.items():
            notes = [root + i for i in intervals if root + i <= MIDI_MAX]
            if len(notes) < 2:
                continue
            # Posición fundamental
            mid  = _make_midi_notes(notes, dur, 75, tpb, humanize)
            path = out_dir / f"chord_{root}_{cname}_root.mid"
            mid.save(str(path))
            paths.append(path)
            # Primera inversión
            inv1  = notes[1:] + [notes[0] + 12]
            inv1  = [n for n in inv1 if n <= MIDI_MAX]
            mid   = _make_midi_notes(inv1, dur, 72, tpb, humanize)
            path  = out_dir / f"chord_{root}_{cname}_inv1.mid"
            mid.save(str(path))
            paths.append(path)
    return paths

def generate_phrases(out_dir: Path, humanize: float = 0.0,
                     beats_per_chord: float = 2.0) -> List[Path]:
    """Progresiones de acordes de 4-8 acordes."""
    mido  = _import_mido()
    tpb   = 480
    dur   = int(tpb * beats_per_chord)
    paths = []
    for prog_idx, prog in enumerate(PROGRESSIONS):
        for root in range(48, 73, 3):   # cada 3 semitonos para no generar demasiado
            mid   = mido.MidiFile(ticks_per_beat=tpb)
            track = mido.MidiTrack()
            mid.tracks.append(track)
            track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

            offsets = [0, 5, 7, 9]   # I, IV, V, vi relativos
            for chord_idx, ctype in enumerate(prog):
                off       = offsets[chord_idx % len(offsets)]
                chord_root = root + off
                intervals  = CHORD_TYPES[ctype]
                notes      = [chord_root + i for i in intervals
                              if MIDI_MIN <= chord_root + i <= MIDI_MAX]
                if not notes:
                    continue
                vel = _humanize(72, humanize, 50, 100)
                t0  = _humanize_time(0, humanize, tpb)
                for note in notes:
                    track.append(mido.Message("note_on",  note=note,
                                              velocity=vel, time=t0))
                    t0 = 0
                d = _humanize_time(dur, humanize, tpb)
                for i, note in enumerate(notes):
                    track.append(mido.Message("note_off", note=note,
                                              velocity=0, time=d if i == 0 else 0))

            path = out_dir / f"phrase_{prog_idx}_{root}.mid"
            mid.save(str(path))
            paths.append(path)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSIÓN MIDI → NPZ (via audio_lab.py)
# ══════════════════════════════════════════════════════════════════════════════

def _find_audio_lab() -> str:
    """Localiza audio_lab.py en el directorio del script o en PATH."""
    candidates = [
        Path(__file__).parent / "audio_lab.py",
        Path("audio_lab.py"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Intentar en PATH
    import shutil
    found = shutil.which("audio_lab.py")
    if found:
        return found
    sys.exit("✗  audio_lab.py no encontrado. Debe estar en el mismo directorio.")

def midi_to_npz(midi_path: Path, npz_path: Path,
                instrument_map: Optional[str] = None,
                window: int = STFT_WINDOW,
                hop: float = STFT_HOP) -> bool:
    """
    MIDI → WAV (via audio_lab.py play-midi) → NPZ (via audio_lab.py spectrogram).
    Devuelve True si tuvo éxito.
    """
    audio_lab = _find_audio_lab()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        cmd_midi = [sys.executable, audio_lab, "play-midi", str(midi_path),
                    "--output", wav_path]
        if instrument_map:
            cmd_midi += ["--instrument-map", instrument_map]
        r = subprocess.run(cmd_midi, capture_output=True, text=True)
        if r.returncode != 0:
            return False

        cmd_spec = [sys.executable, audio_lab, "spectrogram", wav_path,
                    "--window", str(window), "--hop", str(hop),
                    "--output", str(npz_path)]
        r = subprocess.run(cmd_spec, capture_output=True, text=True)
        return r.returncode == 0
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

def wav_to_npz(wav_path: Path, npz_path: Path,
               window: int = STFT_WINDOW, hop: float = STFT_HOP) -> bool:
    """WAV → NPZ via audio_lab.py spectrogram."""
    audio_lab = _find_audio_lab()
    cmd = [sys.executable, audio_lab, "spectrogram", str(wav_path),
           "--window", str(window), "--hop", str(hop), "--output", str(npz_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


# ══════════════════════════════════════════════════════════════════════════════
# CONVERSIÓN MIDI → PIANO ROLL (ground truth)
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_pianoroll(midi_path: str, n_frames: int,
                      sr: int = SR, window: int = STFT_WINDOW,
                      hop: float = STFT_HOP) -> np.ndarray:
    """
    MIDI → piano roll binario  [n_frames × 128].
    Cada celda = 1 si la nota MIDI está activa en ese frame.
    """
    mido    = _import_mido()
    mid     = mido.MidiFile(str(midi_path))
    tpb     = mid.ticks_per_beat
    roll    = np.zeros((n_frames, 128), dtype=np.float32)
    hop_s   = window * hop / sr   # segundos por frame

    tempo_map = _global_tempo_map(mid)
    tick_to_seconds = _tick_to_seconds_fn(mido, tempo_map, tpb)

    for track in mid.tracks:
        tick_abs = 0
        active: Dict[int, float] = {}   # note → time_on_s

        for msg in track:
            tick_abs += msg.time
            time_s = tick_to_seconds(tick_abs)

            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = time_s
            elif msg.type in ("note_off",) or \
                 (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    t_on  = active.pop(msg.note)
                    f_on  = int(t_on / hop_s)
                    f_off = int(time_s / hop_s)
                    f_on  = max(0, min(f_on,  n_frames - 1))
                    f_off = max(f_on + 1, min(f_off, n_frames))
                    roll[f_on:f_off, msg.note] = 1.0

        # Notas sin note_off
        for note, t_on in active.items():
            f_on  = int(t_on / hop_s)
            f_on  = max(0, min(f_on, n_frames - 1))
            roll[f_on:, note] = 1.0

    return roll


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS DE FRASES (prior de frase)
# ══════════════════════════════════════════════════════════════════════════════

class PhraseCorpus:
    """
    Acumula vectores espectrales de frases conocidas.
    Durante la inferencia, busca los K frames más similares al frame actual
    y devuelve sus activaciones MIDI como prior blando.
    """
    def __init__(self, corpus_dir: Path, k: int = 5, sim_threshold: float = 0.85):
        self.dir       = corpus_dir
        self.k         = k
        self.threshold = sim_threshold
        corpus_dir.mkdir(parents=True, exist_ok=True)
        self._keys:   List[np.ndarray] = []   # [n × N_BINS]
        self._values: List[np.ndarray] = []   # [n × 128]
        self._keys_mat:   Optional[np.ndarray] = None   # caché de np.stack(_keys)
        self._values_mat: Optional[np.ndarray] = None   # caché de np.stack(_values)
        self._load()

    def _load(self):
        idx_path = self.dir / "index.npz"
        if idx_path.exists():
            data          = np.load(str(idx_path))
            self._keys    = [data["keys"][i]   for i in range(len(data["keys"]))]
            self._values  = [data["values"][i] for i in range(len(data["values"]))]

    def save(self):
        if not self._keys:
            return
        idx_path = self.dir / "index.npz"
        np.savez_compressed(str(idx_path),
                            keys=np.stack(self._keys),
                            values=np.stack(self._values))

    def add(self, spec_frame: np.ndarray, midi_frame: np.ndarray):
        """Añade un par (frame espectral, activación MIDI)."""
        key = spec_frame / (np.linalg.norm(spec_frame) + 1e-10)
        self._keys.append(key.astype(np.float32))
        self._values.append(midi_frame.astype(np.float32))
        self._keys_mat = None
        self._values_mat = None

    def query(self, spec_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Devuelve un prior suave [128] promediando las K activaciones más similares,
        o None si ninguna supera el umbral de similitud.
        """
        if not self._keys:
            return None
        q = spec_frame / (np.linalg.norm(spec_frame) + 1e-10)
        if self._keys_mat is None:          # se invalida en add(); evita re-apilar
            self._keys_mat = np.stack(self._keys)      # [N × N_BINS]
            self._values_mat = np.stack(self._values)  # [N × 128]
        keys = self._keys_mat
        sims = keys @ q               # [N]
        top_k = np.argsort(sims)[::-1][:self.k]
        best_sim = sims[top_k[0]]
        if best_sim < self.threshold:
            return None
        weights = sims[top_k]
        weights = np.maximum(weights, 0)
        if weights.sum() < 1e-8:
            return None
        weights /= weights.sum()
        prior = (weights[:, None] * self._values_mat[top_k]).sum(axis=0)
        return prior.astype(np.float32)

    def populate_from_corpus(self, corpus_dir: Path, max_entries: int = 50000):
        """Rellena el índice desde los NPZ y pianorolls del corpus."""
        added = 0
        for npz_path in sorted(corpus_dir.glob("*.npz")):
            if "_roll" in npz_path.stem:
                continue
            roll_path = npz_path.with_name(npz_path.stem + "_roll.npz")
            if not roll_path.exists():
                continue
            try:
                spec = np.load(str(npz_path))["magnitudes"]
                roll = np.load(str(roll_path))["roll"]
                n    = min(spec.shape[0], roll.shape[0])
                for i in range(n):
                    self.add(spec[i], roll[i])
                    added += 1
                    if added >= max_entries:
                        break
            except Exception:
                continue
            if added >= max_entries:
                break
        self.save()
        return added


# ══════════════════════════════════════════════════════════════════════════════
# MODELO: CNN + TRANSFORMER
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "small":  {"cnn_layers": 2, "d_model": 128,  "n_heads": 4,  "n_tr": 2, "dropout": 0.1},
    "medium": {"cnn_layers": 3, "d_model": 256,  "n_heads": 8,  "n_tr": 4, "dropout": 0.1},
    "large":  {"cnn_layers": 4, "d_model": 512,  "n_heads": 8,  "n_tr": 8, "dropout": 0.15},
}

def _build_model(model_size: str = "small"):
    torch = _import_torch()
    nn    = torch.nn
    cfg   = MODEL_CONFIGS[model_size]

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            pe  = torch.zeros(max_len, d_model)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))  # [1 × max_len × d_model]

        def forward(self, x):
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)

    class SpectroMidiModel(nn.Module):
        def __init__(self):
            super().__init__()
            d   = cfg["d_model"]
            nl  = cfg["cnn_layers"]
            drop = cfg["dropout"]

            # CNN: extrae patrones armónicos locales
            # Input: [B × 1 × T × N_BINS]
            layers = []
            in_ch  = 1
            out_ch = 32
            for i in range(nl):
                out_ch = 32 * (2 ** min(i, 3))
                layers += [
                    nn.Conv2d(in_ch, out_ch,
                              kernel_size=(3, 7), padding=(1, 3)),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    nn.Conv2d(out_ch, out_ch,
                              kernel_size=(3, 5), padding=(1, 2)),
                    nn.BatchNorm2d(out_ch),
                    nn.GELU(),
                    # Pool solo en frecuencia, no en tiempo
                    nn.MaxPool2d(kernel_size=(1, 2)),
                ]
                in_ch = out_ch
            self.cnn = nn.Sequential(*layers)

            # Calcular tamaño de salida CNN en la dimensión de frecuencia
            bins_out = N_BINS
            for _ in range(nl):
                bins_out = bins_out // 2
            cnn_out_dim = out_ch * bins_out

            # Proyección a d_model
            self.proj = nn.Sequential(
                nn.Linear(cnn_out_dim, d),
                nn.LayerNorm(d),
                nn.GELU(),
                nn.Dropout(drop),
            )

            # Positional encoding
            self.pos_enc = PositionalEncoding(d, dropout=drop)

            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg["n_heads"],
                dim_feedforward=d * 4,
                dropout=drop, batch_first=True,
                norm_first=True,       # pre-norm más estable
            )
            self.transformer = nn.TransformerEncoder(
                enc_layer, num_layers=cfg["n_tr"])

            # Cabeza de salida: [B × T × 128] sigmoid
            self.head = nn.Sequential(
                nn.Linear(d, d // 2),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(d // 2, N_NOTES),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            x: [B × T × N_BINS]  magnitudes STFT normalizadas
            → [B × T × 128]  probabilidades de activación por nota
            """
            B, T, F = x.shape
            # CNN espera [B × C × T × F]
            h = x.unsqueeze(1)                              # [B×1×T×F]
            h = self.cnn(h)                                 # [B×C'×T×F']
            _, C, T2, F2 = h.shape
            # Colapsar canales y frecuencia → [B×T×(C*F)]
            h = h.permute(0, 2, 1, 3).reshape(B, T2, C * F2)
            h = self.proj(h)                                # [B×T×d]
            h = self.pos_enc(h)
            h = self.transformer(h)                         # [B×T×d]
            return self.head(h)                             # [B×T×128]

        def predict_proba(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.sigmoid(self.forward(x))

    return SpectroMidiModel()


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

def _load_corpus_pairs(corpus_dir: Path,
                       max_frames: int = 256,
                       augment: bool = True
                       ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Carga pares (spec [T×N_BINS], roll [T×128]) desde el corpus.
    Divide en segmentos de max_frames con solapamiento del 50%.
    """
    pairs = []
    for npz_path in sorted(corpus_dir.glob("*.npz")):
        if "_roll" in npz_path.stem:
            continue
        roll_path = npz_path.with_name(npz_path.stem + "_roll.npz")
        if not roll_path.exists():
            continue
        try:
            spec = np.load(str(npz_path))["magnitudes"].astype(np.float32)
            roll = np.load(str(roll_path))["roll"].astype(np.float32)
        except Exception:
            continue

        n = min(spec.shape[0], roll.shape[0])
        if n < 8:
            continue

        # Normalizar magnitudes log a [0,1]
        spec = np.log1p(spec)
        spec = spec / (spec.max() + 1e-8)

        # Segmentar con solapamiento
        step = max_frames // 2
        for start in range(0, n - max_frames + 1, step):
            s = spec[start:start + max_frames]
            r = roll[start:start + max_frames, MIDI_MIN:MIDI_MAX + 1]
            pairs.append((s, r))

        # Augmentación: transponer ±2 semitonos en el piano roll
        if augment:
            for shift in [-2, -1, 1, 2]:
                r_shift = np.zeros_like(roll[:n, MIDI_MIN:MIDI_MAX + 1])
                src_lo = max(0, -shift)
                src_hi = N_NOTES - max(0, shift)
                dst_lo = max(0, shift)
                dst_hi = N_NOTES - max(0, -shift)
                r_shift[:, dst_lo:dst_hi] = roll[:n, MIDI_MIN:MIDI_MAX + 1][:, src_lo:src_hi]
                step2 = max_frames // 2
                for start in range(0, n - max_frames + 1, step2):
                    s = spec[start:start + max_frames]
                    r2 = r_shift[start:start + max_frames]
                    pairs.append((s, r2))

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# LOSS: weighted BCE + onset loss
# ══════════════════════════════════════════════════════════════════════════════

def _compute_loss(logits: "torch.Tensor", targets: "torch.Tensor",
                  pos_weight: float = 10.0) -> "torch.Tensor":
    """
    BCE ponderada: las celdas activas tienen peso pos_weight para compensar
    el desequilibrio clase positiva (~5%) vs negativa (~95%).
    + Onset loss: penaliza errores en los frames de inicio de nota.
    """
    torch = _import_torch()
    nn    = torch.nn.functional

    # BCE ponderada
    pw  = torch.full_like(targets, pos_weight) * targets + (1 - targets)
    bce = nn.binary_cross_entropy_with_logits(logits, targets,
                                               weight=pw, reduction="mean")

    # Onset loss: diferencia entre frames consecutivos (positivo = onset)
    onsets_t = torch.clamp(targets[:, 1:] - targets[:, :-1], min=0)
    onsets_p = torch.sigmoid(logits[:, 1:]) - torch.sigmoid(logits[:, :-1])
    onset_l  = nn.mse_loss(onsets_p * onsets_t, onsets_t * onsets_t)

    return bce + 0.5 * onset_l


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCIA: activaciones → MIDI
# ══════════════════════════════════════════════════════════════════════════════

def activations_to_midi(proba: np.ndarray,
                        threshold: float = 0.5,
                        hop_s: float = STFT_WINDOW * STFT_HOP / SR,
                        min_duration_frames: int = 2,
                        phrase_corpus: Optional[PhraseCorpus] = None,
                        spec: Optional[np.ndarray] = None,
                        prior_weight: float = 0.3) -> "mido.MidiFile":
    """
    Convierte la matriz de probabilidades [T×88] en un fichero MIDI.
    
    prior_weight: peso del prior de frase en la decisión final  ∈ [0,1].
    Si phrase_corpus y spec se proporcionan, se mezcla el prior con proba.
    """
    mido = _import_mido()
    T, _  = proba.shape
    proba = proba.copy()   # no mutar el array del llamador al mezclar el prior

    # Aplicar prior de frase si está disponible
    if phrase_corpus is not None and spec is not None:
        for t in range(T):
            prior = phrase_corpus.query(spec[t])
            if prior is not None:
                # prior tiene 128 notas, recortar a rango piano
                prior_piano = prior[MIDI_MIN:MIDI_MAX + 1]
                proba[t]    = (1 - prior_weight) * proba[t] + \
                              prior_weight * prior_piano

    # Binarizar con histeresis suave: activar con threshold, desactivar con threshold*0.7
    active   = proba >= threshold
    active_h = np.zeros_like(active)
    state    = np.zeros(N_NOTES, dtype=bool)
    for t in range(T):
        for n in range(N_NOTES):
            if not state[n] and active[t, n]:
                state[n] = True
            elif state[n] and proba[t, n] < threshold * 0.7:
                state[n] = False
            active_h[t, n] = state[n]

    # Extraer eventos note_on/off
    tpb      = 480
    hop_ticks = int(hop_s * 2 * tpb)   # frames → ticks
    mid      = mido.MidiFile(ticks_per_beat=tpb)
    track    = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    events = []
    for n in range(N_NOTES):
        midi_note = n + MIDI_MIN
        col       = active_h[:, n]
        # Detectar transiciones
        starts = np.where(np.diff(np.concatenate([[0], col.astype(int)])) == 1)[0]
        ends   = np.where(np.diff(np.concatenate([col.astype(int), [0]])) == -1)[0]
        for s, e in zip(starts, ends):
            if e - s < min_duration_frames:
                continue
            # Velocidad: media de la probabilidad durante la nota
            vel = int(np.clip(proba[s:e, n].mean() * 127, 40, 110))
            events.append((s * hop_ticks, "on",  midi_note, vel))
            events.append((e * hop_ticks, "off", midi_note, 0))

    events.sort(key=lambda x: x[0])
    prev_tick = 0
    for tick, etype, note, vel in events:
        delta = tick - prev_tick
        if etype == "on":
            track.append(mido.Message("note_on",  note=note,
                                      velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=note,
                                      velocity=0,  time=delta))
        prev_tick = tick

    return mid


# ══════════════════════════════════════════════════════════════════════════════
# PEAK PICKING SIN ML  (modos melody / harmony)
# ══════════════════════════════════════════════════════════════════════════════
#
# Heurística directa sobre el espectrograma, sin modelo entrenado ni corpus:
# en cada frame se localizan los n bins de mayor intensidad dentro del rango
# de piano, se convierten a nota MIDI y se sostienen mientras sigan siendo
# los bins dominantes de ese frame. melody = harmony con n=1.

def _bin_to_freq(bin_idx: np.ndarray, n_bins: int, sr: int = SR) -> np.ndarray:
    """Índice de bin FFT → frecuencia en Hz.
    Asume ventana real de tamaño 2*(n_bins-1) (coherente con N_BINS = window//2+1)."""
    window = 2 * (n_bins - 1)
    return bin_idx.astype(np.float64) * sr / window

def _freq_to_midi(freq: np.ndarray) -> np.ndarray:
    """Frecuencia en Hz → número de nota MIDI (float; el redondeo lo hace el llamador)."""
    freq = np.maximum(freq, 1e-6)
    return 69.0 + 12.0 * np.log2(freq / 440.0)

def _note_to_bin(note: int, n_bins: int, sr: int = SR) -> int:
    """Nota MIDI → índice de bin FFT más cercano (inversa de _bin_to_freq)."""
    window = 2 * (n_bins - 1)
    freq   = 440.0 * (2.0 ** ((note - 69) / 12.0))
    return int(np.clip(round(freq * window / sr), 0, n_bins - 1))

def _mode_filter(seq: np.ndarray, window: int) -> np.ndarray:
    """Filtro de moda de ventana deslizante: suaviza saltos de 1-2 frames
    (bin flickering) sin necesitar dependencias extra (solo numpy)."""
    if window <= 1:
        return seq
    n     = len(seq)
    half  = window // 2
    out   = seq.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        vals, counts = np.unique(seq[lo:hi], return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out

def _peak_track(mags: np.ndarray, n_voices: int, note_min: int, note_max: int,
                sr: int = SR, silence_thresh: float = 0.0,
                smooth_frames: int = 1) -> np.ndarray:
    """
    Para cada frame localiza las `n_voices` notas MIDI de mayor intensidad
    (colapsando bins adyacentes que caen en la misma nota) y devuelve una
    matriz [T × n_voices] de números de nota MIDI (-1 = silencio / sin
    suficientes picos). La voz 0 es siempre la más intensa de ese frame, la
    voz 1 la segunda, etc. — ordenadas por intensidad, no por altura.

    melody  → n_voices=1
    harmony → n_voices=n  (típicamente 3 = tríada, 4 = acorde de 4 notas)
    """
    T, n_bins = mags.shape
    freqs  = _bin_to_freq(np.arange(n_bins), n_bins, sr)
    midi_r = np.round(_freq_to_midi(freqs)).astype(int)

    # Bins cuya nota cae dentro del rango de piano considerado (evita DC / armónicos fuera de rango)
    valid_bins = np.where((midi_r >= note_min) & (midi_r <= note_max))[0]
    if len(valid_bins) == 0:
        raise ValueError("Rango de notas vacío: revisa --note-min/--note-max")

    peak_global = mags.max() + 1e-12
    out = np.full((T, n_voices), -1, dtype=int)

    for t in range(T):
        frame = mags[t, valid_bins]
        if frame.max() < silence_thresh * peak_global:
            continue   # frame por debajo del umbral de silencio: todas las voces a -1

        order = valid_bins[np.argsort(frame)[::-1]]   # bins válidos, intensidad descendente

        # Colapsar bins adyacentes que caen en la misma nota MIDI: nos quedamos
        # con el más intenso de cada nota (el orden ya viene descendente).
        seen_notes: Dict[int, float] = {}
        for b in order:
            note = int(midi_r[b])
            if note not in seen_notes:
                seen_notes[note] = float(mags[t, b])

        top_notes = sorted(seen_notes.items(), key=lambda kv: kv[1], reverse=True)
        for v in range(min(n_voices, len(top_notes))):
            out[t, v] = top_notes[v][0]

    if smooth_frames > 1:
        for v in range(n_voices):
            out[:, v] = _mode_filter(out[:, v], smooth_frames)

    return out

def _voice_track_to_events(track: np.ndarray, min_frames: int) -> List[Tuple[int, int, int]]:
    """[T] de notas MIDI (-1=silencio) → lista de eventos (start_frame, end_frame, note),
    fusionando frames consecutivos con la misma nota y descartando los más
    cortos que min_frames."""
    events: List[Tuple[int, int, int]] = []
    T = len(track)
    t = 0
    while t < T:
        note = track[t]
        if note < 0:
            t += 1
            continue
        start = t
        while t < T and track[t] == note:
            t += 1
        if t - start >= min_frames:
            events.append((start, t, int(note)))
    return events

def peaks_to_midi(mags: np.ndarray, n_voices: int, note_min: int = MIDI_MIN,
                  note_max: int = MIDI_MAX, sr: int = SR,
                  hop_s: float = STFT_WINDOW * STFT_HOP / SR,
                  min_frames: int = 2, silence_thresh: float = 0.0,
                  smooth_frames: int = 1,
                  velocity: Optional[int] = None) -> "mido.MidiFile":
    """
    Convierte un espectrograma [T × N_BINS] en un MidiFile mediante selección
    directa de picos espectrales (sin ML, sin modelo entrenado).
    n_voices=1 → melody (monofónico) · n_voices>1 → harmony (acorde de n notas).

    velocity: si es None, se calcula automáticamente a partir de la intensidad
    media del bin durante la nota (proporcional al pico global, entre 40-110);
    si se indica un entero, se usa como velocidad fija para todas las notas.
    """
    mido = _import_mido()
    n_bins = mags.shape[1]
    voice_tracks = _peak_track(mags, n_voices, note_min, note_max, sr,
                               silence_thresh, smooth_frames)
    peak_global = mags.max() + 1e-12

    tpb       = 480
    hop_ticks = int(hop_s * 2 * tpb)
    mid       = mido.MidiFile(ticks_per_beat=tpb)
    track     = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))

    events = []
    for v in range(n_voices):
        for start, end, note in _voice_track_to_events(voice_tracks[:, v], min_frames):
            if velocity is not None:
                vel = int(np.clip(velocity, 1, 127))
            else:
                bin_idx = _note_to_bin(note, n_bins, sr)
                mag_avg = mags[start:end, bin_idx].mean()
                vel     = int(np.clip(mag_avg / peak_global * 127, 40, 110))
            events.append((start * hop_ticks, "on",  note, vel))
            events.append((end   * hop_ticks, "off", note, 0))

    # Los "off" van antes que los "on" en el mismo tick para no dejar notas colgadas
    events.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))
    prev_tick = 0
    for tick, etype, note, vel in events:
        delta = max(0, tick - prev_tick)
        if etype == "on":
            track.append(mido.Message("note_on",  note=note, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=note, velocity=0,  time=delta))
        prev_tick = tick

    return mid


# ══════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_midi(pred_path: str, ref_path: str,
                  onset_tolerance_ms: float = 50.0) -> Dict:
    """
    Compara MIDI predicho vs referencia a nivel de nota.
    Métrica: precision, recall, F1 con ventana de onset ±onset_tolerance_ms.
    También calcula F1 a nivel de frame (más permisivo).
    """
    mido = _import_mido()
    hop_s = STFT_WINDOW * STFT_HOP / SR

    def midi_to_notes(path):
        """→ lista de (time_s, note, duration_s)"""
        mid     = mido.MidiFile(path)
        tpb     = mid.ticks_per_beat
        notes   = []
        tempo_map = _global_tempo_map(mid)
        tick_to_seconds = _tick_to_seconds_fn(mido, tempo_map, tpb)
        for track in mid.tracks:
            tick_abs = 0
            active   = {}
            for msg in track:
                tick_abs += msg.time
                time_s = tick_to_seconds(tick_abs)
                if msg.type == "note_on" and msg.velocity > 0:
                    active[msg.note] = time_s
                elif msg.type == "note_off" or \
                     (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in active:
                        t0 = active.pop(msg.note)
                        notes.append((t0, msg.note, time_s - t0))
        return notes

    pred_notes = midi_to_notes(pred_path)
    ref_notes  = midi_to_notes(ref_path)
    tol        = onset_tolerance_ms / 1000.0

    # Note-level F1
    matched_ref  = set()
    matched_pred = set()
    for i, (t_p, n_p, _) in enumerate(pred_notes):
        for j, (t_r, n_r, _) in enumerate(ref_notes):
            if j in matched_ref:
                continue
            if n_p == n_r and abs(t_p - t_r) <= tol:
                matched_ref.add(j)
                matched_pred.add(i)
                break

    tp  = len(matched_pred)
    fp  = len(pred_notes) - tp
    fn  = len(ref_notes)  - len(matched_ref)
    prec  = tp / (tp + fp + 1e-8)
    rec   = tp / (tp + fn + 1e-8)
    f1    = 2 * prec * rec / (prec + rec + 1e-8)

    # Frame-level F1 (100ms frames)
    frame_s   = 0.1
    dur       = max((max(t + d for t, _, d in ref_notes) if ref_notes else 1),
                    (max(t + d for t, _, d in pred_notes) if pred_notes else 1))
    n_frames  = int(dur / frame_s) + 1

    def roll(notes, nf):
        r = np.zeros((nf, 128), dtype=bool)
        for t, n, d in notes:
            f0 = int(t / frame_s)
            f1 = min(int((t + d) / frame_s) + 1, nf)
            r[f0:f1, n] = True
        return r

    r_pred = roll(pred_notes, n_frames)
    r_ref  = roll(ref_notes,  n_frames)
    tp_f   = int((r_pred & r_ref).sum())
    fp_f   = int((r_pred & ~r_ref).sum())
    fn_f   = int((~r_pred & r_ref).sum())
    prec_f = tp_f / (tp_f + fp_f + 1e-8)
    rec_f  = tp_f / (tp_f + fn_f + 1e-8)
    f1_f   = 2 * prec_f * rec_f / (prec_f + rec_f + 1e-8)

    return {
        "note_precision": round(prec, 4),
        "note_recall":    round(rec, 4),
        "note_f1":        round(f1, 4),
        "frame_precision": round(prec_f, 4),
        "frame_recall":    round(rec_f, 4),
        "frame_f1":        round(f1_f, 4),
        "n_pred":  len(pred_notes),
        "n_ref":   len(ref_notes),
        "n_match": tp,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CURRICULUM
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumState:
    level:    int   = 1
    best_f1:  float = 0.0
    patience: int   = 0

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump({"level": self.level, "best_f1": self.best_f1,
                       "patience": self.patience}, f)

    @classmethod
    def load(cls, path: Path) -> "CurriculumState":
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            return cls(**d)
        return cls()

CURRICULUM_CONFIGS = {
    1: {"proc_mode": "notes",   "humanize": 0.0, "model_epochs": 20,
        "description": "notas individuales, síntesis pura"},
    2: {"proc_mode": "chords",  "humanize": 0.0, "model_epochs": 20,
        "description": "acordes de 2 notas, timbre simple"},
    3: {"proc_mode": "chords",  "humanize": 0.2, "model_epochs": 30,
        "description": "acordes de 3-4 notas, humanización ligera"},
    4: {"proc_mode": "phrases", "humanize": 0.3, "model_epochs": 40,
        "description": "frases con progresiones"},
    5: {"proc_mode": "all",     "humanize": 0.5, "model_epochs": 50,
        "description": "all + humanización completa"},
}


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    """MIDI corpus / generación procedimental → pares (NPZ espectrograma, NPZ pianoroll)."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_paths: List[Path] = []

    if args.procedural:
        proc_dir = out_dir / "proc_midi"
        proc_dir.mkdir(exist_ok=True)
        mode = args.proc_mode
        h    = args.humanize
        print(f"  [prepare] Generando MIDIs procedurales  modo={mode}  humanize={h}")

        if mode in ("notes", "all"):
            p = generate_single_notes(proc_dir, h)
            midi_paths += p
            print(f"  ✓  {len(p)} notas individuales")
        if mode in ("chords", "all"):
            p = generate_chords(proc_dir, h)
            midi_paths += p
            print(f"  ✓  {len(p)} acordes")
        if mode in ("phrases", "all"):
            p = generate_phrases(proc_dir, h)
            midi_paths += p
            print(f"  ✓  {len(p)} frases")

    if args.midi_dir:
        for ext in ("*.mid", "*.midi"):
            midi_paths += list(Path(args.midi_dir).rglob(ext))
        print(f"  [prepare] {len(midi_paths)} MIDIs desde {args.midi_dir}")

    if not midi_paths:
        sys.exit("✗  Sin MIDIs. Usa --midi-dir o --procedural.")

    ok = 0
    skip = 0
    for i, midi_path in enumerate(midi_paths):
        npz_path  = out_dir / (midi_path.stem + ".npz")
        roll_path = out_dir / (midi_path.stem + "_roll.npz")

        if npz_path.exists() and roll_path.exists() and not args.force:
            skip += 1
            continue

        sys.stdout.write(f"\r  [{i+1}/{len(midi_paths)}] {midi_path.name[:40]:<40}")
        sys.stdout.flush()

        success = midi_to_npz(midi_path, npz_path,
                              instrument_map=args.instrument_map,
                              window=args.window, hop=args.hop)
        if not success:
            continue

        # Cargar NPZ para saber n_frames
        try:
            data     = np.load(str(npz_path))
            n_frames = data["magnitudes"].shape[0]
        except Exception:
            continue

        roll = midi_to_pianoroll(str(midi_path), n_frames,
                                 window=args.window, hop=args.hop)
        np.savez_compressed(str(roll_path), roll=roll.astype(np.float32))
        ok += 1

    print(f"\n  ✓  {ok} pares generados  |  {skip} ya existían  |  corpus → {out_dir}")

    if args.use_phrase_prior:
        pc = PhraseCorpus(Path(args.phrase_corpus_dir))
        n  = pc.populate_from_corpus(out_dir)
        pc.save()
        print(f"  ✓  Corpus de frases: {n} entradas → {args.phrase_corpus_dir}")


def cmd_train(args):
    """Entrena el modelo CNN+Transformer sobre el corpus preparado."""
    torch = _import_torch()
    nn    = torch.nn

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    print(f"\n● spectro_midi  →  train")
    print(f"  Dispositivo: {device}  |  modelo: {args.model_size}")

    model = _build_model(args.model_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parámetros: {n_params:,}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "model.pt"
    meta_path = model_dir / "meta.json"

    # Cargar checkpoint si existe
    start_epoch = 0
    best_loss   = float("inf")
    if ckpt_path.exists() and not args.reset:
        ck = torch.load(str(ckpt_path), map_location=device)
        model.load_state_dict(ck["model"])
        start_epoch = ck.get("epoch", 0)
        best_loss   = ck.get("best_loss", float("inf"))
        print(f"  Reanudando desde epoch {start_epoch}  (best_loss={best_loss:.4f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Cargar datos
    print(f"  Cargando corpus desde {args.corpus}...")
    pairs = _load_corpus_pairs(Path(args.corpus),
                               max_frames=args.max_frames,
                               augment=not args.no_augment)
    if not pairs:
        sys.exit("✗  Corpus vacío. Ejecuta primero: prepare")
    print(f"  {len(pairs)} segmentos de entrenamiento")

    # Split train/val 90/10
    random.shuffle(pairs)
    n_val   = max(1, len(pairs) // 10)
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]

    def make_batch(subset, batch_size):
        random.shuffle(subset)
        for i in range(0, len(subset), batch_size):
            batch = subset[i:i + batch_size]
            specs = np.stack([p[0] for p in batch])
            rolls = np.stack([p[1] for p in batch])
            yield (torch.tensor(specs, dtype=torch.float32).to(device),
                   torch.tensor(rolls, dtype=torch.float32).to(device))

    patience_count = 0
    log_path       = model_dir / "train_log.jsonl"

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        losses = []
        for specs, rolls in make_batch(train_p, args.batch_size):
            optimizer.zero_grad()
            logits = model(specs)
            loss   = _compute_loss(logits, rolls, pos_weight=args.pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # Validación
        model.eval()
        val_losses = []
        with torch.no_grad():
            for specs, rolls in make_batch(val_p, args.batch_size * 2):
                logits = model(specs)
                val_losses.append(_compute_loss(logits, rolls,
                                                pos_weight=args.pos_weight).item())

        tr_loss  = float(np.mean(losses))
        val_loss = float(np.mean(val_losses)) if val_losses else tr_loss
        lr_now   = optimizer.param_groups[0]["lr"]

        print(f"  epoch {epoch+1:4d}/{start_epoch+args.epochs}  "
              f"loss={tr_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

        # Log
        with open(log_path, "a") as f:
            f.write(json.dumps({"epoch": epoch + 1, "train_loss": tr_loss,
                                 "val_loss": val_loss}) + "\n")

        # Checkpoint
        if val_loss < best_loss:
            best_loss      = val_loss
            patience_count = 0
            torch.save({"model": model.state_dict(), "epoch": epoch + 1,
                        "best_loss": best_loss, "model_size": args.model_size,
                        "window": args.window, "hop": args.hop}, str(ckpt_path))
            print(f"  ✓  Checkpoint guardado (best_loss={best_loss:.4f})")
        else:
            patience_count += 1
            if args.patience and patience_count >= args.patience:
                print(f"  ⚠  Early stopping (patience={args.patience})")
                break

    # Guardar metadatos
    with open(meta_path, "w") as f:
        json.dump({"model_size": args.model_size, "window": args.window,
                   "hop": args.hop, "best_loss": best_loss}, f, indent=2)
    print(f"  ✓  Entrenamiento completado  →  {model_dir}")


def cmd_transcribe(args):
    """NPZ o WAV → MIDI."""
    torch = _import_torch()

    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / "model.pt"
    meta_path = model_dir / "meta.json"
    if not ckpt_path.exists():
        sys.exit(f"✗  Modelo no encontrado: {ckpt_path}")

    meta       = json.load(open(meta_path)) if meta_path.exists() else {}
    model_size = meta.get("model_size", "small")
    window     = meta.get("window", STFT_WINDOW)
    hop        = meta.get("hop", STFT_HOP)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    model  = _build_model(model_size).to(device)
    ck     = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"\n● spectro_midi  →  transcribe  [{model_size} @ {device}]")

    # Cargar espectrograma
    src = Path(args.input)
    if src.suffix.lower() in (".wav", ".flac", ".ogg"):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            npz_tmp = f.name
        ok = wav_to_npz(src, Path(npz_tmp), window, hop)
        if not ok:
            sys.exit("✗  Error convirtiendo WAV → NPZ")
        npz_path = npz_tmp
    else:
        npz_path = str(src)

    data = np.load(npz_path)
    mags = data["magnitudes"].astype(np.float32)

    # Normalizar
    mags_log = np.log1p(mags)
    mags_log = mags_log / (mags_log.max() + 1e-8)

    # Inferencia por segmentos (evita OOM en secuencias largas)
    seg_size = 512
    probas   = []
    with torch.no_grad():
        for start in range(0, len(mags_log), seg_size):
            seg  = mags_log[start:start + seg_size]
            x    = torch.tensor(seg[None], dtype=torch.float32).to(device)
            prob = model.predict_proba(x)[0].cpu().numpy()
            probas.append(prob)
    proba = np.concatenate(probas, axis=0)   # [T × 128 → recortar a rango piano]
    proba_piano = proba   # modelo emite directamente N_NOTES (88) notas del rango piano

    # Prior de frase
    phrase_corpus = None
    if args.use_phrase_prior:
        pc_dir = Path(args.phrase_corpus_dir)
        if pc_dir.exists():
            phrase_corpus = PhraseCorpus(pc_dir, k=5, sim_threshold=0.80)
            print(f"  [prior] Corpus de frases: {len(phrase_corpus._keys)} entradas")
        else:
            print(f"  ⚠  Corpus de frases no encontrado: {pc_dir}")

    hop_s = window * hop / SR
    mid   = activations_to_midi(
        proba_piano, threshold=args.threshold,
        hop_s=hop_s, min_duration_frames=args.min_frames,
        phrase_corpus=phrase_corpus,
        spec=mags_log if phrase_corpus else None,
        prior_weight=args.prior_weight,
    )

    out_path = args.output or src.stem + "_transcribed.mid"
    mid.save(out_path)

    n_notes = sum(1 for t in mid.tracks for m in t if m.type == "note_on"
                  and m.velocity > 0)
    print(f"  ✓  MIDI → {out_path}  ({n_notes} notas, threshold={args.threshold})")

    if isinstance(npz_path, str) and npz_path.endswith(".npz") and \
       npz_path != str(src):
        os.unlink(npz_path)


def _load_spectrogram(src: Path, window: int, hop: float) -> np.ndarray:
    """Carga un espectrograma [T × N_BINS] desde NPZ o desde WAV/FLAC/OGG
    (convirtiendo vía audio_lab.py a un NPZ temporal)."""
    if src.suffix.lower() in (".wav", ".flac", ".ogg"):
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            npz_tmp = f.name
        try:
            ok = wav_to_npz(src, Path(npz_tmp), window, hop)
            if not ok:
                sys.exit("✗  Error convirtiendo WAV → NPZ")
            data = np.load(npz_tmp)
            return data["magnitudes"].astype(np.float32)
        finally:
            if os.path.exists(npz_tmp):
                os.unlink(npz_tmp)
    data = np.load(str(src))
    return data["magnitudes"].astype(np.float32)


def _cmd_peak_track(args, n_voices: int, mode_name: str):
    """Implementación común de los modos melody/harmony: pico(s) espectral(es)
    dominante(s) por frame → nota(s) MIDI. No usa machine learning."""
    src  = Path(args.input)
    mags = _load_spectrogram(src, args.window, args.hop)

    hop_s = args.window * args.hop / SR
    mid = peaks_to_midi(
        mags, n_voices=n_voices,
        note_min=args.note_min, note_max=args.note_max, sr=SR, hop_s=hop_s,
        min_frames=args.min_frames, silence_thresh=args.silence_thresh,
        smooth_frames=args.smooth, velocity=args.velocity,
    )

    out_path = args.output or f"{src.stem}_{mode_name}.mid"
    mid.save(out_path)

    n_notes = sum(1 for t in mid.tracks for m in t if m.type == "note_on"
                  and m.velocity > 0)
    voces = "1 voz (melodía)" if n_voices == 1 else f"{n_voices} voces (acorde)"
    print(f"  ✓  MIDI → {out_path}  ({n_notes} notas, {voces}, "
          f"rango={args.note_min}-{args.note_max}, sin ML)")


def cmd_melody(args):
    """Modo melody: en cada frame, el bin de mayor intensidad → 1 nota
    sostenida hasta que el bin dominante cambia. Sin machine learning."""
    _cmd_peak_track(args, n_voices=1, mode_name="melody")


def cmd_harmony(args):
    """Modo harmony: igual que melody pero con los n bins de mayor
    intensidad por frame (--notes), típicamente 3 = tríada o 4 = acorde de
    cuatro notas. Sin machine learning."""
    _cmd_peak_track(args, n_voices=args.notes, mode_name="harmony")


def cmd_autotrain(args):
    """Bucle autónomo: genera datos → entrena → evalúa → itera."""
    model_dir  = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = model_dir / "corpus_auto"
    corpus_dir.mkdir(exist_ok=True)
    curr_path  = model_dir / "curriculum.json"

    curr = CurriculumState.load(curr_path) if args.curriculum else CurriculumState()
    print(f"\n● spectro_midi  →  autotrain")
    print(f"  Ciclos: {args.cycles}  |  curriculum: {args.curriculum}  "
          f"|  nivel actual: {curr.level}")

    for cycle in range(args.cycles):
        print(f"\n  ── Ciclo {cycle+1}/{args.cycles}  "
              f"(nivel curriculum: {curr.level}) ──")

        # Determinar configuración del ciclo
        if args.curriculum:
            cfg = CURRICULUM_CONFIGS.get(curr.level, CURRICULUM_CONFIGS[5])
            print(f"  [curriculum] {cfg['description']}")
        else:
            cfg = {"proc_mode": args.proc_mode, "humanize": args.humanize,
                   "model_epochs": args.epochs_per_cycle}

        # Generar datos procedurales
        cycle_dir = corpus_dir / f"level_{curr.level}"
        cycle_dir.mkdir(exist_ok=True)

        midi_paths: List[Path] = []
        if args.midi_dir:
            for ext in ("*.mid", "*.midi"):
                midi_paths += list(Path(args.midi_dir).rglob(ext))

        h    = cfg["humanize"]
        mode = cfg["proc_mode"]
        proc_dir = cycle_dir / "midi"
        proc_dir.mkdir(exist_ok=True)

        if mode in ("notes", "all") or not midi_paths:
            midi_paths += generate_single_notes(proc_dir, h)
        if mode in ("chords", "all"):
            midi_paths += generate_chords(proc_dir, h)
        if mode in ("phrases", "all"):
            midi_paths += generate_phrases(proc_dir, h)

        print(f"  [datos] {len(midi_paths)} MIDIs")

        # Convertir a NPZ
        ok = 0
        for mp in midi_paths[:args.max_per_cycle]:
            np_ = cycle_dir / (mp.stem + ".npz")
            rp  = cycle_dir / (mp.stem + "_roll.npz")
            if np_.exists() and rp.exists():
                ok += 1
                continue
            if midi_to_npz(mp, np_, window=STFT_WINDOW, hop=STFT_HOP):
                data     = np.load(str(np_))
                n_frames = data["magnitudes"].shape[0]
                roll     = midi_to_pianoroll(str(mp), n_frames)
                np.savez_compressed(str(rp), roll=roll.astype(np.float32))
                ok += 1
        print(f"  [datos] {ok} pares NPZ listos")

        # Entrenar
        train_args = argparse.Namespace(
            corpus       = str(cycle_dir),
            model_dir    = args.model_dir,
            model_size   = args.model_size,
            epochs       = cfg["model_epochs"],
            batch_size   = args.batch_size,
            lr           = args.lr,
            patience     = args.patience,
            pos_weight   = 10.0,
            max_frames   = 256,
            no_augment   = False,
            reset        = False,
            cpu          = args.cpu,
            window       = STFT_WINDOW,
            hop          = STFT_HOP,
        )
        cmd_train(train_args)

        # Evaluar sobre un subconjunto
        midi_eval = [mp for mp in midi_paths[:20]
                     if (cycle_dir / (mp.stem + ".npz")).exists()]
        f1_scores = []
        for mp in midi_eval[:10]:
            npz_path = cycle_dir / (mp.stem + ".npz")
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                pred_path = f.name
            try:
                tr_args = argparse.Namespace(
                    input            = str(npz_path),
                    model_dir        = args.model_dir,
                    output           = pred_path,
                    threshold        = 0.5,
                    min_frames       = 2,
                    use_phrase_prior = False,
                    phrase_corpus_dir = str(model_dir / "phrase_corpus"),
                    prior_weight     = 0.3,
                    cpu              = args.cpu,
                )
                cmd_transcribe(tr_args)
                metrics = evaluate_midi(pred_path, str(mp))
                f1_scores.append(metrics["frame_f1"])
            except Exception as e:
                pass
            finally:
                if os.path.exists(pred_path):
                    os.unlink(pred_path)

        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        print(f"  [eval] F1 medio (frame): {mean_f1:.4f}  "
              f"(sobre {len(f1_scores)} piezas)")

        # Actualizar corpus de frases
        if args.use_phrase_prior:
            pc = PhraseCorpus(model_dir / "phrase_corpus")
            n  = pc.populate_from_corpus(cycle_dir, max_entries=10000)
            pc.save()
            print(f"  [prior] Corpus de frases actualizado: {n} entradas nuevas")

        # Curriculum: subir de nivel si F1 ≥ threshold
        if args.curriculum:
            if mean_f1 >= args.curriculum_threshold:
                curr.patience = 0
                if curr.level < max(CURRICULUM_CONFIGS.keys()):
                    curr.level  += 1
                    curr.best_f1 = mean_f1
                    print(f"  ✓  Subiendo a nivel {curr.level}  "
                          f"(F1={mean_f1:.3f} ≥ {args.curriculum_threshold})")
                else:
                    print(f"  ✓  Nivel máximo alcanzado  (F1={mean_f1:.3f})")
            else:
                curr.patience += 1
                print(f"  [curriculum] F1={mean_f1:.3f} < {args.curriculum_threshold}  "
                      f"patience={curr.patience}/{args.curriculum_patience}")
                if curr.patience >= args.curriculum_patience and curr.level > 1:
                    curr.level    = max(1, curr.level - 1)
                    curr.patience = 0
                    print(f"  ⚠  Bajando a nivel {curr.level}")
            curr.save(curr_path)

        # Log del ciclo
        log_entry = {"cycle": cycle + 1, "level": curr.level,
                     "f1": mean_f1, "n_pairs": ok}
        with open(model_dir / "autotrain_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    print(f"\n  ✓  Autotrain completado  →  {args.model_dir}")


def cmd_eval(args):
    """Evalúa MIDI predicho vs referencia."""
    print(f"\n● spectro_midi  →  eval")
    metrics = evaluate_midi(args.predicted, args.reference,
                            onset_tolerance_ms=args.onset_tol)
    print(f"\n  Nota-level:")
    print(f"    Precision : {metrics['note_precision']:.4f}")
    print(f"    Recall    : {metrics['note_recall']:.4f}")
    print(f"    F1        : {metrics['note_f1']:.4f}")
    print(f"    Notas pred: {metrics['n_pred']}  |  ref: {metrics['n_ref']}  "
          f"|  coincidencias: {metrics['n_match']}")
    print(f"\n  Frame-level (ventana 100ms):")
    print(f"    Precision : {metrics['frame_precision']:.4f}")
    print(f"    Recall    : {metrics['frame_recall']:.4f}")
    print(f"    F1        : {metrics['frame_f1']:.4f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n  ✓  Métricas → {args.output}")


def cmd_inspect(args):
    """Diagnóstico del modelo y del corpus."""
    torch = _import_torch()
    print(f"\n● spectro_midi  →  inspect")

    model_dir = Path(args.model_dir)
    ckpt_path = model_dir / "model.pt"
    meta_path = model_dir / "meta.json"

    if ckpt_path.exists():
        meta       = json.load(open(meta_path)) if meta_path.exists() else {}
        model_size = meta.get("model_size", "small")
        model      = _build_model(model_size)
        ck         = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ck["model"])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Modelo: {model_size}  |  {n_params:,} params")
        print(f"  Epoch: {ck.get('epoch', '?')}  |  best_loss: {ck.get('best_loss', '?'):.4f}")
        cfg = MODEL_CONFIGS[model_size]
        print(f"  CNN layers: {cfg['cnn_layers']}  |  d_model: {cfg['d_model']}  "
              f"|  Transformer: {cfg['n_tr']} capas × {cfg['n_heads']} heads")
    else:
        print(f"  ⚠  Sin modelo entrenado en {model_dir}")

    if args.corpus:
        corpus_dir = Path(args.corpus)
        npzs  = list(corpus_dir.glob("*.npz"))
        specs  = [f for f in npzs if "_roll" not in f.stem]
        rolls  = [f for f in npzs if "_roll" in f.stem]
        pairs  = len([s for s in specs
                      if (corpus_dir / (s.stem + "_roll.npz")).exists()])
        print(f"\n  Corpus: {corpus_dir}")
        print(f"  NPZ espectrogramas : {len(specs)}")
        print(f"  NPZ piano rolls    : {len(rolls)}")
        print(f"  Pares completos    : {pairs}")

        if specs:
            sample = np.load(str(specs[0]))["magnitudes"]
            hop_s  = STFT_WINDOW * STFT_HOP / SR
            print(f"  Ejemplo: {sample.shape[0]} frames "
                  f"({sample.shape[0]*hop_s:.1f}s) × {sample.shape[1]} bins")

    log_path = model_dir / "autotrain_log.jsonl"
    if log_path.exists():
        lines = log_path.read_text().strip().split("\n")
        print(f"\n  Historial autotrain ({len(lines)} ciclos):")
        for line in lines[-5:]:
            d = json.loads(line)
            print(f"    ciclo {d['cycle']:3d}  nivel={d['level']}  "
                  f"F1={d['f1']:.4f}  pares={d['n_pairs']}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="spectro_midi",
        description="Transcripción automática espectrograma → MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser("prepare", help="MIDI / procedimental → corpus NPZ")
    p.add_argument("--midi-dir",  default=None,
                   help="Directorio con ficheros MIDI de entrenamiento")
    p.add_argument("--out-dir",   default="corpus", help="Directorio de salida")
    p.add_argument("--procedural", action="store_true",
                   help="Generar MIDIs proceduralmente (notas, acordes, frases)")
    p.add_argument("--proc-mode", default="all",
                   choices=["notes","chords","phrases","all"],
                   help="Tipo de MIDIs procedurales (default: all)")
    p.add_argument("--humanize",  type=float, default=0.0,
                   help="Variación de velocidad/timing ∈ [0,1] (default: 0)")
    p.add_argument("--instrument-map", default=None,
                   help="JSON de instrumentos para audio_lab.py play-midi")
    p.add_argument("--window",    type=int,   default=STFT_WINDOW)
    p.add_argument("--hop",       type=float, default=STFT_HOP)
    p.add_argument("--force",     action="store_true",
                   help="Regenerar aunque ya existan los NPZ")
    p.add_argument("--use-phrase-prior", action="store_true",
                   help="Poblar corpus de frases tras preparar")
    p.add_argument("--phrase-corpus-dir", default="phrase_corpus")
    p.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Entrena el modelo CNN+Transformer")
    p.add_argument("--corpus",     required=True, help="Directorio del corpus")
    p.add_argument("--model-dir",  default="model")
    p.add_argument("--model-size", default="small",
                   choices=["small","medium","large"])
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early stopping (0 = desactivado)")
    p.add_argument("--pos-weight",  type=float, default=10.0,
                   help="Peso de las celdas activas en el BCE (default: 10)")
    p.add_argument("--max-frames",  type=int,   default=256,
                   help="Longitud de segmento de entrenamiento en frames (default: 256)")
    p.add_argument("--no-augment",  action="store_true",
                   help="Desactivar augmentación por transposición")
    p.add_argument("--reset",       action="store_true",
                   help="Ignorar checkpoint existente y entrenar desde cero")
    p.add_argument("--cpu",         action="store_true",
                   help="Forzar CPU aunque haya GPU disponible")
    p.add_argument("--window",      type=int,   default=STFT_WINDOW)
    p.add_argument("--hop",         type=float, default=STFT_HOP)
    p.set_defaults(func=cmd_train)

    # ── transcribe ────────────────────────────────────────────────────────────
    p = sub.add_parser("transcribe", help="NPZ / WAV → MIDI")
    p.add_argument("input", help="Fichero NPZ o WAV/FLAC/OGG")
    p.add_argument("--model-dir",  default="model")
    p.add_argument("--output",     "-o", default=None)
    p.add_argument("--threshold",  type=float, default=0.5,
                   help="Umbral de activación ∈ [0,1] (default: 0.5)")
    p.add_argument("--min-frames", type=int,   default=2,
                   help="Duración mínima de nota en frames (default: 2)")
    p.add_argument("--use-phrase-prior", action="store_true")
    p.add_argument("--phrase-corpus-dir", default="phrase_corpus")
    p.add_argument("--prior-weight", type=float, default=0.3,
                   help="Peso del prior de frase ∈ [0,1] (default: 0.3)")
    p.add_argument("--cpu",  action="store_true")
    p.set_defaults(func=cmd_transcribe)

    # ── melody (sin ML) ──────────────────────────────────────────────────────
    p = sub.add_parser("melody",
                       help="NPZ/WAV → MIDI monofónico (pico espectral dominante, sin ML)")
    p.add_argument("input", help="Fichero NPZ o WAV/FLAC/OGG")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--note-min", type=int, default=MIDI_MIN,
                   help=f"Nota MIDI mínima a considerar (default: {MIDI_MIN}, A0)")
    p.add_argument("--note-max", type=int, default=MIDI_MAX,
                   help=f"Nota MIDI máxima a considerar (default: {MIDI_MAX}, C8)")
    p.add_argument("--min-frames", type=int, default=2,
                   help="Duración mínima de nota en frames (default: 2)")
    p.add_argument("--smooth", type=int, default=1,
                   help="Ventana del filtro de moda para suavizar saltos de bin "
                        "(default: 1 = sin suavizado)")
    p.add_argument("--silence-thresh", type=float, default=0.0,
                   help="Umbral de silencio relativo al pico global ∈ [0,1] "
                        "(default: 0 = desactivado)")
    p.add_argument("--velocity", type=int, default=None,
                   help="Velocidad MIDI fija (default: auto, proporcional a la intensidad)")
    p.add_argument("--window", type=int,   default=STFT_WINDOW)
    p.add_argument("--hop",    type=float, default=STFT_HOP)
    p.set_defaults(func=cmd_melody)

    # ── harmony (sin ML) ─────────────────────────────────────────────────────
    p = sub.add_parser("harmony",
                       help="NPZ/WAV → MIDI de n voces (n picos espectrales dominantes, sin ML)")
    p.add_argument("input", help="Fichero NPZ o WAV/FLAC/OGG")
    p.add_argument("--notes", type=int, default=3,
                   help="Número de voces/notas simultáneas: 3=tríada, 4=acorde "
                        "de 4 notas... (default: 3)")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--note-min", type=int, default=MIDI_MIN,
                   help=f"Nota MIDI mínima a considerar (default: {MIDI_MIN}, A0)")
    p.add_argument("--note-max", type=int, default=MIDI_MAX,
                   help=f"Nota MIDI máxima a considerar (default: {MIDI_MAX}, C8)")
    p.add_argument("--min-frames", type=int, default=2,
                   help="Duración mínima de nota en frames (default: 2)")
    p.add_argument("--smooth", type=int, default=1,
                   help="Ventana del filtro de moda para suavizar saltos de bin "
                        "(default: 1 = sin suavizado)")
    p.add_argument("--silence-thresh", type=float, default=0.0,
                   help="Umbral de silencio relativo al pico global ∈ [0,1] "
                        "(default: 0 = desactivado)")
    p.add_argument("--velocity", type=int, default=None,
                   help="Velocidad MIDI fija (default: auto, proporcional a la intensidad)")
    p.add_argument("--window", type=int,   default=STFT_WINDOW)
    p.add_argument("--hop",    type=float, default=STFT_HOP)
    p.set_defaults(func=cmd_harmony)

    # ── autotrain ─────────────────────────────────────────────────────────────
    p = sub.add_parser("autotrain", help="Bucle autónomo genera→entrena→evalúa")
    p.add_argument("--model-dir",   default="model")
    p.add_argument("--midi-dir",    default=None,
                   help="MIDIs de referencia adicionales al procedimental")
    p.add_argument("--cycles",      type=int,   default=10)
    p.add_argument("--model-size",  default="small", choices=["small","medium","large"])
    p.add_argument("--batch-size",  type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--epochs-per-cycle", type=int, default=20)
    p.add_argument("--proc-mode",   default="all",
                   choices=["notes","chords","phrases","all"])
    p.add_argument("--humanize",    type=float, default=0.0)
    p.add_argument("--max-per-cycle", type=int, default=500,
                   help="MIDIs máximos por ciclo (default: 500)")
    p.add_argument("--curriculum",  action="store_true",
                   help="Activar curriculum learning (complejidad progresiva)")
    p.add_argument("--curriculum-threshold", type=float, default=0.80,
                   help="F1 mínimo para subir de nivel (default: 0.80)")
    p.add_argument("--curriculum-patience", type=int, default=3,
                   help="Ciclos sin mejora antes de bajar nivel (default: 3)")
    p.add_argument("--use-phrase-prior", action="store_true")
    p.add_argument("--phrase-corpus-dir", default="phrase_corpus")
    p.add_argument("--cpu", action="store_true")
    p.set_defaults(func=cmd_autotrain)

    # ── eval ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("eval", help="Evalúa MIDI predicho vs referencia")
    p.add_argument("--predicted",  required=True)
    p.add_argument("--reference",  required=True)
    p.add_argument("--onset-tol",  type=float, default=50.0,
                   help="Tolerancia de onset en ms (default: 50)")
    p.add_argument("--output",     "-o", default=None,
                   help="Guardar métricas en JSON")
    p.set_defaults(func=cmd_eval)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser("inspect", help="Diagnóstico del modelo y corpus")
    p.add_argument("--model-dir", default="model")
    p.add_argument("--corpus",    default=None)
    p.set_defaults(func=cmd_inspect)

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    print(f"\n● spectro_midi  →  {args.command}")
    args.func(args)
    print()


if __name__ == "__main__":
    main()
