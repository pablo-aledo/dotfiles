#!/usr/bin/env python3
# encoding: utf-8
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                                                                          ║
# ║   P I T C H   T R A N S C R I B E R                                     ║
# ║   ─────────────────────────────────                                    ║
# ║                                                                          ║
# ║   Transcriptor automatico de audio a MIDI (Automatic Music              ║
# ║   Transcription). Toma un fichero de audio (wav/mp3/flac/ogg/...) y     ║
# ║   produce un fichero MIDI polifonico con pitch bends, usando la red     ║
# ║   neuronal ligera "Basic Pitch" (Spotify Audio Intelligence Lab,        ║
# ║   ICASSP 2022) ya entrenada y exportada a ONNX.                         ║
# ║                                                                          ║
# ║   Adaptacion de github.com/spotify/basic-pitch al estilo "mutopia":     ║
# ║   un unico fichero autocontenido, sin imports cruzados a otros          ║
# ║   scripts, con subcomandos argparse sencillos. El pipeline de           ║
# ║   entrenamiento, la abstraccion multi-runtime (TF/CoreML/TFLite) y      ║
# ║   los cargadores de datasets del proyecto original se han descartado:   ║
# ║   aqui solo se hace inferencia, con onnxruntime como unico backend.     ║
# ║                                                                          ║
# ║   USO                                                                   ║
# ║   ───                                                                   ║
# ║     transcribir un fichero de audio a MIDI:                            ║
# ║       $ python pitch_transcriber.py transcribe entrada.wav salida/     ║
# ║                                                                          ║
# ║     ademas del MIDI, guardar posteriogramas crudos, notas en CSV y      ║
# ║     una version renderizada a wav del MIDI resultante:                 ║
# ║       $ python pitch_transcriber.py transcribe entrada.wav salida/ \   ║
# ║             --save-model-outputs --save-notes --sonify                 ║
# ║                                                                          ║
# ║     renderizar a wav un MIDI ya existente (sin pasar por el modelo):   ║
# ║       $ python pitch_transcriber.py sonify cancion.mid cancion.wav     ║
# ║                                                                          ║
# ║     comprobar que el modelo carga y ver sus dimensiones de E/S:        ║
# ║       $ python pitch_transcriber.py info                               ║
# ║                                                                          ║
# ║   DEPENDENCIAS                                                          ║
# ║   ────────────                                                         ║
# ║     numpy, scipy, librosa, pretty_midi, onnxruntime                    ║
# ║                                                                          ║
# ║   MODELO                                                                ║
# ║   ──────                                                                ║
# ║     Este script espera encontrar el fichero de pesos "nmp.onnx"        ║
# ║     (el modelo ICASSP-2022 de Basic Pitch, ~228 KB) en el mismo         ║
# ║     directorio que este script. Alternativamente, indica su ruta       ║
# ║     con --model o la variable de entorno PITCH_TRANSCRIBER_MODEL.      ║
# ║     El fichero puede obtenerse de:                                     ║
# ║       github.com/spotify/basic-pitch/tree/main/basic_pitch/            ║
# ║             saved_models/icassp_2022/nmp.onnx                          ║
# ║                                                                          ║
# ║   Basado en Basic Pitch (c) 2022 Spotify AB, licencia Apache 2.0.       ║
# ║   Paper: "A Lightweight Instrument-Agnostic Model for Polyphonic        ║
# ║   Note Transcription and Multipitch Estimation" (ICASSP 2022).          ║
# ║                                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.signal
from scipy.io import wavfile

# ──────────────────────────────────────────────────────────────────────────
# Colores ANSI (estilo mutopia)
# ──────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def info(msg: str) -> None:
    print(f"{CYAN}[info]{RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{GREEN}[ok]{RESET}   {msg}")


def warn(msg: str) -> None:
    print(f"{YELLOW}[warn]{RESET} {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    print(f"{RED}[error]{RESET} {msg}", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────
# Constantes del modelo (portadas de basic_pitch/constants.py)
# ──────────────────────────────────────────────────────────────────────────

SEMITONES_PER_OCTAVE = 12
FFT_HOP = 256
NOTES_BINS_PER_SEMITONE = 1
CONTOURS_BINS_PER_SEMITONE = 3
ANNOTATIONS_BASE_FREQUENCY = 27.5  # nota mas grave del piano (A0)
ANNOTATIONS_N_SEMITONES = 88  # numero de teclas de un piano
AUDIO_SAMPLE_RATE = 22050
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE  # 264

AUDIO_WINDOW_LENGTH = 2  # segundos por ventana de entrada al modelo
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP  # 86
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH  # 172
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844

# constantes de note_creation.py
MIDI_OFFSET = 21
N_PITCH_BEND_TICKS = 8192
MAX_FREQ_IDX = 87
ENERGY_TOLERANCE = 11
MAGIC_ALIGNMENT_OFFSET = 0.0018
MIDI_VELOCITY_SCALE = 127
PITCH_BEND_SCALE = 4096

# valores por defecto de la CLI (idénticos a los del basic-pitch original)
DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 127.7
DEFAULT_MIDI_TEMPO = 120.0
DEFAULT_SONIFICATION_SAMPLERATE = 44100
DEFAULT_OVERLAPPING_FRAMES = 30

# nombres de tensores del grafo ONNX exportado (icassp_2022/nmp.onnx).
# Si se usa un .onnx re-exportado con otros nombres, ajustar aqui.
ONNX_INPUT_NAME = "serving_default_input_2:0"
ONNX_OUTPUT_NAMES = {
    "note": "StatefulPartitionedCall:1",
    "onset": "StatefulPartitionedCall:2",
    "contour": "StatefulPartitionedCall:0",
}

DEFAULT_MODEL_FILENAME = "nmp.onnx"
MODEL_PATH_ENV_VAR = "PITCH_TRANSCRIBER_MODEL"


def _freq_bins(bins_per_semitone: int, base_frequency: float, n_semitones: int) -> np.ndarray:
    d = 2.0 ** (1.0 / (SEMITONES_PER_OCTAVE * bins_per_semitone))
    return base_frequency * d ** np.arange(bins_per_semitone * n_semitones)


FREQ_BINS_NOTES = _freq_bins(NOTES_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)
FREQ_BINS_CONTOURS = _freq_bins(CONTOURS_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)


# ──────────────────────────────────────────────────────────────────────────
# Carga perezosa de dependencias pesadas / opcionales
# ──────────────────────────────────────────────────────────────────────────
#
# librosa, pretty_midi y onnxruntime son importaciones relativamente lentas.
# Se cargan bajo demanda para que "pitch_transcriber.py info --help" y demas
# ayudas de argparse respondan al instante, siguiendo la convencion mutopia
# de arranque rapido.


def _import_librosa() -> Any:
    try:
        import librosa

        return librosa
    except ImportError as e:
        fail("Este comando necesita 'librosa' (pip install librosa).")
        raise SystemExit(1) from e


def _import_pretty_midi() -> Any:
    try:
        import pretty_midi

        return pretty_midi
    except ImportError as e:
        fail("Este comando necesita 'pretty_midi' (pip install pretty_midi).")
        raise SystemExit(1) from e


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime

        return onnxruntime
    except ImportError as e:
        fail("Este comando necesita 'onnxruntime' (pip install onnxruntime).")
        raise SystemExit(1) from e


# ──────────────────────────────────────────────────────────────────────────
# Resolucion y carga del modelo ONNX
# ──────────────────────────────────────────────────────────────────────────


def resolve_model_path(explicit_path: Optional[str]) -> pathlib.Path:
    """Decide que fichero de pesos usar, en orden de prioridad:
    1) --model explicito, 2) variable de entorno, 3) junto al script.
    """
    if explicit_path:
        path = pathlib.Path(explicit_path)
        if not path.is_file():
            fail(f"No se encuentra el modelo indicado con --model: {path}")
            raise SystemExit(1)
        return path

    env_path = os.environ.get(MODEL_PATH_ENV_VAR)
    if env_path:
        path = pathlib.Path(env_path)
        if not path.is_file():
            fail(f"No se encuentra el modelo indicado por {MODEL_PATH_ENV_VAR}: {path}")
            raise SystemExit(1)
        return path

    default_path = pathlib.Path(__file__).resolve().parent / DEFAULT_MODEL_FILENAME
    if default_path.is_file():
        return default_path

    fail(
        f"No se encuentra el fichero de pesos '{DEFAULT_MODEL_FILENAME}'.\n"
        f"        Colocalo junto a este script, o indica su ruta con --model,\n"
        f"        o exporta {MODEL_PATH_ENV_VAR}=/ruta/a/nmp.onnx"
    )
    raise SystemExit(1)


class PitchModel:
    """Envoltorio fino sobre una sesion onnxruntime del modelo Basic Pitch."""

    def __init__(self, model_path: Union[pathlib.Path, str]):
        ort = _import_onnxruntime()
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        try:
            self.session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as e:
            fail(f"No se pudo cargar el modelo ONNX en {model_path}: {e}")
            raise SystemExit(1) from e

        input_names = {i.name for i in self.session.get_inputs()}
        output_names = {o.name for o in self.session.get_outputs()}
        if ONNX_INPUT_NAME not in input_names or not set(ONNX_OUTPUT_NAMES.values()) <= output_names:
            fail(
                "El fichero ONNX no tiene los nombres de tensor esperados.\n"
                f"        Entrada esperada: {ONNX_INPUT_NAME!r} (encontrada: {sorted(input_names)})\n"
                f"        Salidas esperadas: {sorted(ONNX_OUTPUT_NAMES.values())} "
                f"(encontradas: {sorted(output_names)})\n"
                "        Si es un re-export distinto del modelo, ajusta ONNX_INPUT_NAME / "
                "ONNX_OUTPUT_NAMES en la cabecera del script."
            )
            raise SystemExit(1)

    def predict(self, audio_windowed: np.ndarray) -> Dict[str, np.ndarray]:
        """audio_windowed: array (n_ventanas, AUDIO_N_SAMPLES, 1) float32."""
        keys = ["note", "onset", "contour"]
        run_names = [ONNX_OUTPUT_NAMES[k] for k in keys]
        outputs = self.session.run(run_names, {ONNX_INPUT_NAME: audio_windowed})
        return dict(zip(keys, outputs))

    def describe(self) -> str:
        lines = []
        for i in self.session.get_inputs():
            lines.append(f"  entrada  {i.name:<32} shape={i.shape} dtype={i.type}")
        for o in self.session.get_outputs():
            lines.append(f"  salida   {o.name:<32} shape={o.shape} dtype={o.type}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Carga y ventaneo de audio (portado de inference.py)
# ──────────────────────────────────────────────────────────────────────────


def window_audio_file(audio_original: np.ndarray, hop_size: int) -> Iterable[Tuple[np.ndarray, Dict[str, float]]]:
    """Trocea una señal de audio en ventanas de longitud fija AUDIO_N_SAMPLES."""
    for i in range(0, audio_original.shape[0], hop_size):
        window = audio_original[i : i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, pad_width=[[0, AUDIO_N_SAMPLES - len(window)]])
        t_start = float(i) / AUDIO_SAMPLE_RATE
        window_time = {"start": t_start, "end": t_start + (AUDIO_N_SAMPLES / AUDIO_SAMPLE_RATE)}
        yield np.expand_dims(window, axis=-1), window_time


def get_audio_input(
    audio_path: Union[pathlib.Path, str], overlap_len: int, hop_size: int
) -> Iterable[Tuple[np.ndarray, Dict[str, float], int]]:
    """Lee un fichero de audio (mono, remuestreado a AUDIO_SAMPLE_RATE) y lo ventanea."""
    assert overlap_len % 2 == 0, f"overlap_len debe ser par, se recibio {overlap_len}"

    librosa = _import_librosa()
    audio_original, _ = librosa.load(str(audio_path), sr=AUDIO_SAMPLE_RATE, mono=True)

    original_length = audio_original.shape[0]
    audio_original = np.concatenate([np.zeros((int(overlap_len / 2),), dtype=np.float32), audio_original])
    for window, window_time in window_audio_file(audio_original, hop_size):
        yield np.expand_dims(window, axis=0), window_time, original_length


def unwrap_output(
    output: np.ndarray, audio_original_length: int, n_overlapping_frames: int, hop_size: int
) -> Optional[np.ndarray]:
    """Concatena las predicciones por ventana (con overlap) en una unica matriz temporal."""
    if len(output.shape) != 3:
        return None

    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        output = output[:, n_olap:-n_olap, :]

    output_shape = output.shape
    unwrapped_output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])

    n_expected_windows = audio_original_length / hop_size
    n_frames_per_window = (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS) - n_overlapping_frames
    return unwrapped_output[: int(n_expected_windows * n_frames_per_window), :]


def run_inference(audio_path: Union[pathlib.Path, str], model: PitchModel) -> Dict[str, np.ndarray]:
    """Ejecuta el modelo sobre todo el fichero de audio y devuelve los posteriogramas."""
    n_overlapping_frames = DEFAULT_OVERLAPPING_FRAMES
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    raw: Dict[str, List[np.ndarray]] = {"note": [], "onset": [], "contour": []}
    audio_original_length = 0
    for audio_windowed, _window_time, audio_original_length in get_audio_input(audio_path, overlap_len, hop_size):
        prediction = model.predict(audio_windowed.astype(np.float32))
        for k, v in prediction.items():
            raw[k].append(v)

    return {
        k: unwrap_output(np.concatenate(v), audio_original_length, n_overlapping_frames, hop_size)
        for k, v in raw.items()
    }


# ──────────────────────────────────────────────────────────────────────────
# Postprocesado: posteriogramas -> eventos de nota -> MIDI
# (portado de note_creation.py)
# ──────────────────────────────────────────────────────────────────────────


def constrain_frequency(
    onsets: np.ndarray, frames: np.ndarray, max_freq: Optional[float], min_freq: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    librosa = _import_librosa()
    n_freqs = onsets.shape[1]
    min_freq_idx = 0
    max_freq_idx = n_freqs

    if min_freq is not None:
        min_freq_idx = int(np.round(librosa.hz_to_midi(min_freq) - MIDI_OFFSET))
    if max_freq is not None:
        max_freq_idx = int(np.round(librosa.hz_to_midi(max_freq) - MIDI_OFFSET))

    onsets[:, :min_freq_idx] = 0
    frames[:, :min_freq_idx] = 0
    onsets[:, max_freq_idx:] = 0
    frames[:, max_freq_idx:] = 0
    return onsets, frames


def get_infered_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    if frames.shape[0] == 0:
        # Audio de entrada mas corto que un hop de analisis: no hay frames que
        # procesar. El original revienta aqui con "zero-size array to
        # reduction operation maximum which has no identity"; devolvemos un
        # array vacio en su lugar para poder seguir sin notas.
        return onsets
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)

    return np.max([onsets, frame_diff], axis=0)


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: Optional[float],
    min_freq: Optional[float],
    melodia_trick: bool = True,
    energy_tol: int = ENERGY_TOLERANCE,
) -> List[Tuple[int, int, int, float]]:
    n_frames = frames.shape[0]
    if n_frames == 0:
        # Audio de entrada mas corto que un hop de analisis (~1.6 s por
        # defecto): no hay ningun frame de posteriograma que decodificar.
        return []

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]
    onset_freq_idx = onset_idx[1][::-1]

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    note_events = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start_idx >= n_frames - 1:
            continue

        i = note_start_idx + 1
        k = 0
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        i -= k

        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0

        amplitude = np.mean(frames[note_start_idx:i, freq_idx])
        note_events.append((note_start_idx, i, freq_idx + MIDI_OFFSET, amplitude))

    if melodia_trick:
        energy_shape = remaining_energy.shape

        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k

            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k

            assert i_start >= 0, f"{i_start}"
            assert i_end < n_frames

            if i_end - i_start <= min_note_len:
                continue

            amplitude = np.mean(frames[i_start:i_end, freq_idx])
            note_events.append((i_start, i_end, freq_idx + MIDI_OFFSET, amplitude))

    return note_events


def midi_pitch_to_contour_bin(pitch_midi: int) -> float:
    librosa = _import_librosa()
    pitch_hz = librosa.midi_to_hz(pitch_midi)
    return 12.0 * CONTOURS_BINS_PER_SEMITONE * np.log2(pitch_hz / ANNOTATIONS_BASE_FREQUENCY)


def get_pitch_bends(
    contours: np.ndarray, note_events: List[Tuple[int, int, int, float]], n_bins_tolerance: int = 25
) -> List[Tuple[int, int, int, float, Optional[List[int]]]]:
    window_length = n_bins_tolerance * 2 + 1
    freq_gaussian = scipy.signal.windows.gaussian(window_length, std=5)
    note_events_with_pitch_bends = []
    for start_idx, end_idx, pitch_midi, amplitude in note_events:
        freq_idx = int(np.round(midi_pitch_to_contour_bin(pitch_midi)))
        freq_start_idx = np.max([freq_idx - n_bins_tolerance, 0])
        freq_end_idx = np.min([N_FREQ_BINS_CONTOURS, freq_idx + n_bins_tolerance + 1])

        pitch_bend_submatrix = (
            contours[start_idx:end_idx, freq_start_idx:freq_end_idx]
            * freq_gaussian[
                np.max([0, n_bins_tolerance - freq_idx]) : window_length
                - np.max([0, freq_idx - (N_FREQ_BINS_CONTOURS - n_bins_tolerance - 1)])
            ]
        )
        pb_shift = n_bins_tolerance - np.max([0, n_bins_tolerance - freq_idx])

        bends: Optional[List[int]] = list(np.argmax(pitch_bend_submatrix, axis=1) - pb_shift)
        note_events_with_pitch_bends.append((start_idx, end_idx, pitch_midi, amplitude, bends))
    return note_events_with_pitch_bends


def drop_overlapping_pitch_bends(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
) -> List[Tuple[float, float, int, float, Optional[List[int]]]]:
    note_events = sorted(note_events_with_pitch_bends)
    for i in range(len(note_events) - 1):
        for j in range(i + 1, len(note_events)):
            if note_events[j][0] >= note_events[i][1]:
                break
            note_events[i] = note_events[i][:-1] + (None,)
            note_events[j] = note_events[j][:-1] + (None,)
    return note_events


def model_frames_to_time(n_frames: int) -> np.ndarray:
    librosa = _import_librosa()
    original_times = librosa.core.frames_to_time(np.arange(n_frames), sr=AUDIO_SAMPLE_RATE, hop_length=FFT_HOP)
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (
        ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)
    ) + MAGIC_ALIGNMENT_OFFSET
    return original_times - (window_offset * window_numbers)


def note_events_to_midi(
    note_events_with_pitch_bends: List[Tuple[float, float, int, float, Optional[List[int]]]],
    multiple_pitch_bends: bool = False,
    midi_tempo: float = DEFAULT_MIDI_TEMPO,
) -> Any:
    pretty_midi = _import_pretty_midi()
    mid = pretty_midi.PrettyMIDI(initial_tempo=midi_tempo)
    if not multiple_pitch_bends:
        note_events_with_pitch_bends = drop_overlapping_pitch_bends(note_events_with_pitch_bends)

    piano_program = pretty_midi.instrument_name_to_program("Electric Piano 1")
    instruments: DefaultDict[int, Any] = defaultdict(lambda: pretty_midi.Instrument(program=piano_program))
    for start_time, end_time, note_number, amplitude, pitch_bend in note_events_with_pitch_bends:
        instrument = instruments[note_number] if multiple_pitch_bends else instruments[0]
        note = pretty_midi.Note(
            velocity=int(np.round(MIDI_VELOCITY_SCALE * amplitude)),
            pitch=note_number,
            start=start_time,
            end=end_time,
        )
        instrument.notes.append(note)
        if not pitch_bend:
            continue
        pitch_bend_times = np.linspace(start_time, end_time, len(pitch_bend))
        pitch_bend_midi_ticks = np.round(np.array(pitch_bend) * PITCH_BEND_SCALE / CONTOURS_BINS_PER_SEMITONE).astype(
            int
        )
        pitch_bend_midi_ticks[pitch_bend_midi_ticks > N_PITCH_BEND_TICKS - 1] = N_PITCH_BEND_TICKS - 1
        pitch_bend_midi_ticks[pitch_bend_midi_ticks < -N_PITCH_BEND_TICKS] = -N_PITCH_BEND_TICKS
        for pb_time, pb_midi in zip(pitch_bend_times, pitch_bend_midi_ticks):
            instrument.pitch_bends.append(pretty_midi.PitchBend(pb_midi, pb_time))
    mid.instruments.extend(instruments.values())
    return mid


def model_output_to_notes(
    output: Dict[str, np.ndarray],
    onset_thresh: float,
    frame_thresh: float,
    infer_onsets: bool = True,
    min_note_len: int = 11,
    min_freq: Optional[float] = None,
    max_freq: Optional[float] = None,
    include_pitch_bends: bool = True,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    midi_tempo: float = DEFAULT_MIDI_TEMPO,
) -> Tuple[Any, List[Tuple[float, float, int, float, Optional[List[int]]]]]:
    frames = output["note"]
    onsets = output["onset"]
    contours = output["contour"]

    estimated_notes = output_to_notes_polyphonic(
        frames,
        onsets,
        onset_thresh=onset_thresh,
        frame_thresh=frame_thresh,
        infer_onsets=infer_onsets,
        min_note_len=min_note_len,
        min_freq=min_freq,
        max_freq=max_freq,
        melodia_trick=melodia_trick,
    )
    if include_pitch_bends:
        estimated_notes_with_pitch_bend = get_pitch_bends(contours, estimated_notes)
    else:
        estimated_notes_with_pitch_bend = [(n[0], n[1], n[2], n[3], None) for n in estimated_notes]

    times_s = model_frames_to_time(contours.shape[0])
    estimated_notes_time_seconds = [
        (times_s[n[0]], times_s[n[1]], n[2], n[3], n[4]) for n in estimated_notes_with_pitch_bend
    ]

    midi = note_events_to_midi(estimated_notes_time_seconds, multiple_pitch_bends, midi_tempo)
    return midi, estimated_notes_time_seconds


def sonify_midi(midi: Any, save_path: Union[pathlib.Path, str], sr: int = DEFAULT_SONIFICATION_SAMPLERATE) -> None:
    """Renderiza un objeto pretty_midi.PrettyMIDI a un wav (sintesis aditiva simple)."""
    y = midi.synthesize(sr)
    wavfile.write(str(save_path), sr, y)


def save_note_events(
    note_events: List[Tuple[float, float, int, float, Optional[List[int]]]],
    save_path: Union[pathlib.Path, str],
) -> None:
    with open(save_path, "w", newline="") as fhandle:
        writer = csv.writer(fhandle, delimiter=",")
        writer.writerow(["start_time_s", "end_time_s", "pitch_midi", "velocity", "pitch_bend"])
        for start_time, end_time, note_number, amplitude, pitch_bend in note_events:
            row = [start_time, end_time, note_number, int(np.round(MIDI_VELOCITY_SCALE * amplitude))]
            if pitch_bend:
                row.extend(pitch_bend)
            writer.writerow(row)


# ──────────────────────────────────────────────────────────────────────────
# Orquestacion de alto nivel
# ──────────────────────────────────────────────────────────────────────────


def transcribe_file(
    audio_path: Union[pathlib.Path, str],
    model: PitchModel,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_length_ms: float = DEFAULT_MINIMUM_NOTE_LENGTH_MS,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = None,
    multiple_pitch_bends: bool = False,
    melodia_trick: bool = True,
    midi_tempo: float = DEFAULT_MIDI_TEMPO,
) -> Tuple[Dict[str, np.ndarray], Any, List[Tuple[float, float, int, float, Optional[List[int]]]]]:
    """Pipeline completo: audio en disco -> (posteriogramas, MIDI, eventos de nota)."""
    model_output = run_inference(audio_path, model)
    min_note_len = int(np.round(minimum_note_length_ms / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = model_output_to_notes(
        model_output,
        onset_thresh=onset_threshold,
        frame_thresh=frame_threshold,
        min_note_len=min_note_len,
        min_freq=minimum_frequency,
        max_freq=maximum_frequency,
        multiple_pitch_bends=multiple_pitch_bends,
        melodia_trick=melodia_trick,
        midi_tempo=midi_tempo,
    )
    return model_output, midi_data, note_events


def build_output_path(audio_path: Union[pathlib.Path, str], output_directory: pathlib.Path, suffix: str) -> pathlib.Path:
    basename = pathlib.Path(audio_path).stem
    return output_directory / f"{basename}_transcribed.{suffix}"


# ──────────────────────────────────────────────────────────────────────────
# Subcomandos CLI
# ──────────────────────────────────────────────────────────────────────────


def cmd_transcribe(args: argparse.Namespace) -> int:
    model_path = resolve_model_path(args.model)
    output_dir = pathlib.Path(args.output_dir)
    if not output_dir.is_dir():
        fail(f"El directorio de salida no existe: {output_dir}")
        return 1

    info(f"Cargando modelo desde {model_path}")
    model = PitchModel(model_path)

    exit_code = 0
    for audio_path in args.audio_files:
        audio_path = pathlib.Path(audio_path)
        if not audio_path.is_file():
            fail(f"No existe el fichero de audio: {audio_path}")
            exit_code = 1
            continue

        info(f"Transcribiendo {audio_path} ...")
        try:
            model_output, midi_data, note_events = transcribe_file(
                audio_path,
                model,
                onset_threshold=args.onset_threshold,
                frame_threshold=args.frame_threshold,
                minimum_note_length_ms=args.min_note_length,
                minimum_frequency=args.min_freq,
                maximum_frequency=args.max_freq,
                multiple_pitch_bends=args.multiple_pitch_bends,
                melodia_trick=not args.no_melodia_trick,
                midi_tempo=args.midi_tempo,
            )
        except Exception as e:
            fail(f"Fallo al transcribir {audio_path}: {e}")
            exit_code = 1
            continue

        if model_output["note"].shape[0] == 0:
            warn(
                f"{audio_path} es mas corto que una ventana de analisis "
                f"(~{(AUDIO_N_SAMPLES - DEFAULT_OVERLAPPING_FRAMES * FFT_HOP) / AUDIO_SAMPLE_RATE:.2f} s); "
                "no se ha podido extraer ningun frame. Se generara un MIDI vacio."
            )

        midi_path = build_output_path(audio_path, output_dir, "mid")
        midi_data.write(str(midi_path))
        ok(f"MIDI guardado en {midi_path}  ({len(note_events)} notas)")

        if args.save_model_outputs:
            npz_path = build_output_path(audio_path, output_dir, "npz")
            np.savez(npz_path, **model_output)
            ok(f"Posteriogramas guardados en {npz_path}")

        if args.save_notes:
            csv_path = build_output_path(audio_path, output_dir, "csv")
            save_note_events(note_events, csv_path)
            ok(f"Eventos de nota guardados en {csv_path}")

        if args.sonify:
            wav_path = build_output_path(audio_path, output_dir, "wav")
            sonify_midi(midi_data, wav_path, sr=args.sonification_samplerate)
            ok(f"Sonificacion guardada en {wav_path}")

    return exit_code


def cmd_sonify(args: argparse.Namespace) -> int:
    pretty_midi = _import_pretty_midi()
    midi_path = pathlib.Path(args.midi_file)
    if not midi_path.is_file():
        fail(f"No existe el fichero MIDI: {midi_path}")
        return 1

    info(f"Renderizando {midi_path} a audio...")
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    sonify_midi(midi, args.output_wav, sr=args.samplerate)
    ok(f"Guardado en {args.output_wav}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    print(f"{BOLD}pitch_transcriber{RESET} — Basic Pitch (ICASSP 2022) sobre onnxruntime")
    print(f"  sample rate de entrada : {AUDIO_SAMPLE_RATE} Hz")
    print(f"  ventana de audio       : {AUDIO_WINDOW_LENGTH} s ({AUDIO_N_SAMPLES} muestras)")
    print(f"  fps de anotacion       : {ANNOTATIONS_FPS}")
    print(f"  rango de notas         : {ANNOTATIONS_N_SEMITONES} semitonos desde {ANNOTATIONS_BASE_FREQUENCY} Hz")
    print()
    try:
        model_path = resolve_model_path(args.model)
    except SystemExit:
        return 1
    info(f"Cargando modelo de prueba desde {model_path} ...")
    model = PitchModel(model_path)
    ok("Modelo cargado correctamente.")
    print(model.describe())
    return 0


# ──────────────────────────────────────────────────────────────────────────
# main / argparse
# ──────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pitch_transcriber.py",
        description="Transcriptor de audio a MIDI basado en Basic Pitch (Spotify), adaptado a estilo mutopia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_transcribe = sub.add_parser("transcribe", help="Transcribe uno o mas ficheros de audio a MIDI")
    p_transcribe.add_argument("audio_files", nargs="+", help="Ficheros de audio de entrada (wav, mp3, flac, ...)")
    p_transcribe.add_argument("output_dir", help="Directorio donde guardar los resultados")
    p_transcribe.add_argument("--model", default=None, help="Ruta al fichero .onnx del modelo")
    p_transcribe.add_argument("--onset-threshold", type=float, default=DEFAULT_ONSET_THRESHOLD)
    p_transcribe.add_argument("--frame-threshold", type=float, default=DEFAULT_FRAME_THRESHOLD)
    p_transcribe.add_argument("--min-note-length", type=float, default=DEFAULT_MINIMUM_NOTE_LENGTH_MS,
                               help="Duracion minima de nota en milisegundos")
    p_transcribe.add_argument("--min-freq", type=float, default=None, help="Frecuencia minima en Hz")
    p_transcribe.add_argument("--max-freq", type=float, default=None, help="Frecuencia maxima en Hz")
    p_transcribe.add_argument("--multiple-pitch-bends", action="store_true",
                               help="Permite pitch bends en notas superpuestas (un instrumento MIDI por altura)")
    p_transcribe.add_argument("--no-melodia-trick", action="store_true",
                               help="Desactiva el post-procesado 'melodia trick'")
    p_transcribe.add_argument("--midi-tempo", type=float, default=DEFAULT_MIDI_TEMPO)
    p_transcribe.add_argument("--save-model-outputs", action="store_true", help="Guarda tambien los posteriogramas en .npz")
    p_transcribe.add_argument("--save-notes", action="store_true", help="Guarda tambien los eventos de nota en .csv")
    p_transcribe.add_argument("--sonify", action="store_true", help="Renderiza tambien el MIDI resultante a .wav")
    p_transcribe.add_argument("--sonification-samplerate", type=int, default=DEFAULT_SONIFICATION_SAMPLERATE)
    p_transcribe.set_defaults(func=cmd_transcribe)

    p_sonify = sub.add_parser("sonify", help="Renderiza un MIDI existente a wav (sin pasar por el modelo)")
    p_sonify.add_argument("midi_file", help="Fichero MIDI de entrada")
    p_sonify.add_argument("output_wav", help="Fichero wav de salida")
    p_sonify.add_argument("--samplerate", type=int, default=DEFAULT_SONIFICATION_SAMPLERATE)
    p_sonify.set_defaults(func=cmd_sonify)

    p_info = sub.add_parser("info", help="Comprueba que el modelo carga y muestra sus dimensiones de E/S")
    p_info.add_argument("--model", default=None, help="Ruta al fichero .onnx del modelo")
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
