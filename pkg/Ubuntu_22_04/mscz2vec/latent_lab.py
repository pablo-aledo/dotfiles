#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         LATENT LAB  v1.0                                     ║
║  Codec neuronal de audio y puentes espectrograma↔latente — fichero único     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUBCOMANDOS                                                                 ║
║    train-coder                  WAVs → entrena codec wav↔latente → coder.pt  ║
║    wav-to-latent                WAV → latentes .npz [+ --png]                ║
║    latent-to-wav                .npz / PNG latente → WAV (decoder)           ║
║    latent-to-png                .npz latente → PNG escala de grises          ║
║    png-to-latent                PNG → .npz latente                           ║
║    spectrogram-to-latent-train_step1  WAVs + coder.pt → pares PNG editables  ║
║                                        espectrograma/latente (sin entrenar)  ║
║    spectrogram-to-latent-train_step2  pares PNG → mapper.pt, --method        ║
║                                        {pix2pix,retrieval,causal,            ║
║                                        causal-adv} (ver ARQUITECTURA)        ║
║    spectrogram-to-latent        .npz STFT (audio_lab) → .npz latente,        ║
║                                  --method {auto,pix2pix,greedy,viterbi,      ║
║                                  causal,causal-adv}                          ║
║    latent-to-spectrogram        .npz/PNG latente → .npz STFT [+ --wav],      ║
║                                  mismo --method que spectrogram-to-latent    ║
║    spectrogram-to-png            NPZ STFT (audio_lab) → PNG escala de grises ║
║    png-to-spectrogram            PNG → NPZ STFT (audio_lab)                  ║
║    train-pca                    NPZ latente(s)/espectrograma(s) → pca.npz    ║
║    intermediate-to-pca          NPZ intermedio + pca.npz → coords PCA .npz   ║
║    pca-to-intermediate          coords PCA .npz/PNG → NPZ intermedio         ║
║    info                         Diagnóstico de WAV, NPZ, PNG o .pt           ║
║    download-pretrained          Descarga y verifica un DAC oficial pre-      ║
║                                  entrenado, listo para usar como --coder     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS  numpy  soundfile  torch  Pillow                               ║
║                descript-audio-codec (opcional, solo para --coder *.pth)      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARQUITECTURA                                                                ║
║                                                                              ║
║  El codec es un autoencoder convolucional 1-D inspirado en el Descript       ║
║  Audio Codec (DAC): encoder con convoluciones con stride (activación         ║
║  Snake), espacio latente CONTINUO de D componentes por frame, y decoder      ║
║  espejo con convoluciones transpuestas. Sin cuantización RVQ: el latente     ║
║  continuo se presta a ser editado como imagen en escala de grises.           ║
║                                                                              ║
║  hop = producto de los strides (default 2·4·8·8 = 512 muestras/frame).       ║
║  A 44100 Hz eso son ~86 frames latentes por segundo.                         ║
║                                                                              ║
║  MODELOS PRE-ENTRENADOS (recomendado en vez de train-coder desde cero)       ║
║  wav-to-latent y latent-to-wav aceptan como --coder un checkpoint OFICIAL    ║
║  de Descript Audio Codec (fichero .pth, p.ej. el weights.pth publicado en    ║
║  github.com/descriptinc/descript-audio-codec/releases). Se detecta por la    ║
║  extensión y se usa el encoder/decoder continuos de DAC (D=1024, hop=512,    ║
║  76.7M parámetros, 44.1kHz) sin necesitar entrenar nada. Requiere el         ║
║  paquete opcional: pip install descript-audio-codec                          ║
║    python latent_lab.py wav-to-latent cancion.wav \\                         ║
║           --coder dac_weights.pth --png -o cancion_latent.npz                ║
║    python latent_lab.py latent-to-wav cancion_latent.png \\                  ║
║           --coder dac_weights.pth -o resultado.wav                           ║
║  El binario TSAC (Bellard) con sus modelos .bin es otra alternativa          ║
║  pre-entrenada independiente, usable directamente por su cuenta (no se       ║
║  integra en este script, ver tsac-*/readme.txt).                             ║
║                                                                              ║
║  El mapper espectrograma↔latente se construye en DOS PASOS, y el paso 2      ║
║  admite TRES MÉTODOS alternativos, seleccionables por línea de comandos:     ║
║                                                                              ║
║  step1: por cada WAV, calcula el espectrograma STFT (audio_lab) y el         ║
║  latente del coder, los ALINEA en el tiempo y los vuelca como PNG en         ║
║  escala de grises (uno por representación) + manifest.json con los           ║
║  metadatos (sr, window, hop, D, lat_min/max). NO entrena nada. Un humano     ║
║  puede editar esos PNG a mano (limpiar ruido, recortar, corregir) antes      ║
║  de continuar. Es COMÚN a los tres métodos del paso 2.                       ║
║                                                                              ║
║  step2 --method pix2pix (default, compatible con versiones anteriores)       ║
║  Entrena DOS redes con el enfoque pix2pix (Isola et al. 2017): un            ║
║  generador U-Net con skip-connections + un discriminador PatchGAN, para      ║
║  cada dirección (espectrograma→latente y latente→espectrograma), con         ║
║  pérdida L1 + adversarial. Las imágenes se trocean en parches temporales     ║
║  de anchura --tile para el entrenamiento; en inferencia se recomponen        ║
║  con solape y ventana de Hann. Generaliza a entradas nunca vistas, pero      ║
║  tiende al promedio de las texturas plausibles (blur) cuando el sketch       ║
║  de entrada es ambiguo — ver más abajo, MÉTODOS DE RECUPERACIÓN.             ║
║                                                                              ║
║  step2 --method retrieval (sin red neuronal — indexa, no entrena)            ║
║  Concatena todos los pares frame a frame en un banco de búsqueda             ║
║  (S_bank/Z_bank). No hay optimización: el corpus ES el modelo. El            ║
║  mismo banco sirve para los dos métodos de búsqueda siguientes —             ║
║  greedy y viterbi eligen el ALGORITMO de búsqueda en INFERENCIA              ║
║  (spectrogram-to-latent --method), no en este paso.                          ║
║                                                                              ║
║  MÉTODOS DE RECUPERACIÓN (spectrogram-to-latent --method greedy|viterbi)     ║
║                                                                              ║
║  Frente a pix2pix (regresión → tiende al promedio de casos ambiguos),        ║
║  greedy y viterbi seleccionan frames REALES del corpus, sin promediar        ║
║  nunca — a costa de generalizar peor que una red a sketches muy              ║
║  distintos de lo visto en entrenamiento (sobreajuste consciente).            ║
║                                                                              ║
║  greedy: para cada frame de entrada, el vecino más cercano del banco         ║
║  (target cost = distancia euclídea), de forma independiente. Rápido,         ║
║  pero sin garantía de continuidad entre frames consecutivos.                 ║
║                                                                              ║
║  viterbi: programación dinámica sobre los --topk candidatos por frame,       ║
║  minimizando target cost + --join-weight · join cost (continuidad en         ║
║  el espacio de salida entre frames consecutivos). Es la técnica de           ║
║  unit-selection clásica de síntesis concatenativa de voz (Hunt &             ║
║  Black 1996) aplicada a frames latentes. greedy es el caso particular        ║
║  topk=1 (sin coste de unión).                                                ║
║                                                                              ║
║  MÉTODOS AUTORREGRESIVOS (spectrogram-to-latent --method causal|causal-adv)  ║
║                                                                              ║
║  step2 --method causal: entrena CausalGenerator (WaveNet-lite), que          ║
║  predice la columna latente t a partir de las columnas latentes              ║
║  ANTERIORES (convoluciones causales dilatadas, campo receptivo =             ║
║  2**n_layers ≥ --context frames) más un condicionamiento del                 ║
║  espectrograma COMPLETO (no causal — en generación offline el                ║
║  espectrograma entero ya está disponible). L1 puro + teacher forcing         ║
║  FIJO (siempre se alimenta el latente real desplazado, nunca la              ║
║  propia predicción, durante el entrenamiento). En INFERENCIA sí es           ║
║  autorregresivo de verdad, frame a frame, alimentado de sus propias          ║
║  predicciones (no hay latente real disponible fuera de entrenamiento)        ║
║  — expuesto por tanto al exposure bias clásico de los modelos                ║
║  autorregresivos, y al mismo promedio de texturas ambiguas que               ║
║  pix2pix, al ser también regresión L1 pura.                                  ║
║                                                                              ║
║  step2 --method causal-adv: añade dos mitigaciones sobre lo anterior:        ║
║    · scheduled sampling (Bengio et al. 2015): con probabilidad               ║
║      creciente --ss-final-prob a lo largo del entrenamiento, la              ║
║      historia se sustituye por la propia predicción del modelo (de           ║
║      una primera pasada, sin gradiente) en vez del latente real —            ║
║      simplificación en DOS pasadas paralelas, no rollout secuencial          ║
║      completo (ver docstring de _train_step2_causal_adv).                    ║
║    · pérdida adversarial (CausalDiscriminator1D, PatchGAN 1-D                ║
║      condicionado al espectrograma): penaliza que la secuencia               ║
║      generada no tenga la estadística de una secuencia real,                 ║
║      empujando hacia UNA textura plausible en vez del promedio.              ║
║                                                                              ║
║  Ambos métodos SOLO entrenan la dirección espectrograma→latente              ║
║  (latent-to-spectrogram --method causal falla con error explícito) —         ║
║  ver conversación de diseño: esta dirección no hacía falta aquí.             ║
║  Generación con ventana acotada al campo receptivo del modelo                ║
║  (O(T·campo_receptivo), no O(T²) con toda la historia completa).             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PIPELINE NPZ ↔ PNG (latentes)                                               ║
║                                                                              ║
║  El PNG es una imagen en escala de grises donde:                             ║
║    · Eje X  = tiempo  (frame 0 a N-1, izquierda → derecha)                   ║
║    · Eje Y  = componente latente  (componente 0 abajo, D-1 arriba)           ║
║    · Valor  = valor del latente, normalizado por componente al rango         ║
║               [lat_min, lat_max] calculado sobre el corpus de entrenamiento  ║
║                                                                              ║
║  Los metadatos necesarios para la reconversión (sr, hop, lat_min, lat_max,   ║
║  orientación) se guardan en chunks tEXt del PNG y en un sidecar .ll.json     ║
║  (fallback si el editor de imagen elimina los metadatos al exportar).        ║
║                                                                              ║
║  Flujo típico de edición creativa:                                           ║
║    train-coder corpus/*.wav -o coder.pt                                      ║
║    wav-to-latent cancion.wav --coder coder.pt --png -o cancion.npz           ║
║    [editar cancion.png en GIMP/Photoshop: borrar, pintar, clonar…]           ║
║    latent-to-wav cancion_editada.png --coder coder.pt -o resultado.wav       ║
║                                                                              ║
║  Interoperabilidad con audio_lab.py: los NPZ de espectrograma que consume    ║
║  y produce este programa (claves magnitudes/phases/freqs/sample_rate/        ║
║  window_size/hop_ratio) son 100% compatibles con audio_lab spectrogram,      ║
║  reconstruct, edit-spectrum y npz-to-png.                                    ║
║                                                                              ║
║  PIPELINE NPZ ↔ PNG (espectrogramas) — par gemelo del anterior               ║
║                                                                              ║
║  spectrogram-to-png / png-to-spectrogram hacen para el espectrograma STFT    ║
║  lo mismo que latent-to-png / png-to-latent para el latente: mismo esquema   ║
║  de PNG editable (bin 0 abajo), pero con la normalización propia de          ║
║  magnitudes STFT (mags_to_norm/norm_to_mags: pico del fichero + rango dB     ║
║  con --db-floor), NO min/max por componente como en el latente. Metadatos    ║
║  (sr, window, hop_ratio, db_floor, orientación) en chunks tEXt + sidecar     ║
║  .ll.json, igual que en el par de latentes. El NPZ de salida es 100%         ║
║  compatible con audio_lab y con spectrogram-to-latent.                       ║
║                                                                              ║
║  EJEMPLOS                                                                    ║
║    python latent_lab.py train-coder corpus/*.wav --steps 4000 -o coder.pt    ║
║    python latent_lab.py wav-to-latent voz.wav --coder coder.pt --png         ║
║    python latent_lab.py latent-to-wav voz.png --coder coder.pt -o out.wav    ║
║    python latent_lab.py spectrogram-to-latent-train_step1 corpus/*.wav \\    ║
║           --coder coder.pt -o pairs/                                         ║
║    [editar los PNG en pairs/ a mano, opcional]                               ║
║    python latent_lab.py spectrogram-to-latent-train_step2 pairs/ \\          ║
║           -o mapper.pt                                                       ║
║    python latent_lab.py spectrogram-to-latent-train_step2 pairs/ \\          ║
║           --method retrieval -o mapper_retrieval.pt                          ║
║    python latent_lab.py spectrogram-to-latent voz_stft.npz \\                ║
║           --mapper mapper_retrieval.pt --method greedy -o voz_lat.npz        ║
║    python latent_lab.py spectrogram-to-latent voz_stft.npz \\                ║
║           --mapper mapper_retrieval.pt --method viterbi --topk 8 \\          ║
║           --join-weight 1.0 -o voz_lat.npz                                   ║
║    python latent_lab.py spectrogram-to-latent-train_step2 pairs/ \\          ║
║           --method causal --context 32 -o mapper_causal.pt                   ║
║    python latent_lab.py spectrogram-to-latent-train_step2 pairs/ \\          ║
║           --method causal-adv --context 32 --ss-final-prob 0.5 \\            ║
║           -o mapper_causal_adv.pt                                            ║
║    python latent_lab.py spectrogram-to-latent voz_stft.npz \\                ║
║           --mapper mapper_causal_adv.pt -o voz_lat.npz                       ║
║    python latent_lab.py spectrogram-to-latent voz_stft.npz \\                ║
║           --mapper mapper.pt -o voz_lat.npz                                  ║
║    python latent_lab.py latent-to-spectrogram voz_lat.npz \\                 ║
║           --mapper mapper.pt --wav voz_gl.wav                                ║
║    python latent_lab.py train-pca corpus_latents/*.npz -o pca.npz            ║
║    python latent_lab.py intermediate-to-pca voz_lat.npz --pca pca.npz        ║
║           --png -o voz_pca.npz                                               ║
║    python latent_lab.py pca-to-intermediate voz_pca.npz -o voz_lat2.npz      ║
║    python latent_lab.py info coder.pt                                        ║
║    python latent_lab.py spectrogram-to-png voz_stft.npz -o voz_spec.png      ║
║    [editar voz_spec.png en GIMP/Photoshop]                                   ║
║    python latent_lab.py png-to-spectrogram voz_spec.png -o voz_stft2.npz     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── lazy imports ──────────────────────────────────────────────────────────────
def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("✗  soundfile no encontrado. Instala con: pip install soundfile")

def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        sys.exit("✗  torch no encontrado. Instala con: pip install torch")

def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        sys.exit("✗  Pillow no encontrado. Instala con: pip install Pillow")

def _import_dac():
    try:
        import dac
        return dac
    except ImportError:
        sys.exit("✗  descript-audio-codec no encontrado (necesario para coders .pth "
                 "pre-entrenados oficiales). Instala con: pip install descript-audio-codec")

# ── modelos oficiales DAC descargables (github.com/descriptinc/descript-audio-codec) ──
# clave "variante" → (url, sample_rate, descripción)
PRETRAINED_DAC_MODELS = {
    "44khz-8kbps": (
        "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth",
        44100, "Modelo universal 44.1kHz @ 8kbps (~90x compresión). El usado por defecto."),
    "24khz-8kbps": (
        "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.4/weights_24khz.pth",
        24000, "Modelo 24kHz @ 8kbps."),
    "16khz-8kbps": (
        "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth",
        16000, "Modelo 16kHz @ 8kbps (voz)."),
    "44khz-16kbps": (
        "https://github.com/descriptinc/descript-audio-codec/releases/download/1.0.0/weights_44khz_16kbps.pth",
        44100, "Modelo 44.1kHz @ 16kbps, mayor calidad/bitrate."),
}

# ── constantes ────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 44100
DEFAULT_STRIDES  = (2, 4, 8, 8)     # hop = 512 muestras/frame latente
DEFAULT_LATENT   = 32               # componentes del espacio latente
DEFAULT_BASE_CH  = 32               # canales de la primera conv del encoder
DEFAULT_DB_FLOOR = -80.0            # piso dB para normalizar espectrogramas
PIX2PIX_CANVAS_H = 256               # altura fija del lienzo pix2pix (bins/D se
                                      # reescalan a esta altura para entrenar/inferir)

# ══════════════════════════════════════════════════════════════════════════════
# AUDIO I/O
# ══════════════════════════════════════════════════════════════════════════════

def read_wav(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Devuelve (audio_mono_float32, sample_rate). Remuestrea (lineal) si se pide."""
    sf = _import_soundfile()
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]
    if target_sr and sr != target_sr:
        n_out   = int(len(audio) * target_sr / sr)
        idx_new = np.linspace(0, len(audio) - 1, n_out)
        audio   = np.interp(idx_new, np.arange(len(audio)), audio).astype(np.float32)
        print(f"  ⚠  {Path(path).name}: remuestreado {sr} → {target_sr} Hz (lineal)")
        sr = target_sr
    return audio, sr

def write_wav(path: str, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf = _import_soundfile()
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio, sr, subtype="PCM_24")
    print(f"  ✓  WAV escrito → {path}  ({len(audio)/sr:.2f}s, {sr}Hz)")

# ══════════════════════════════════════════════════════════════════════════════
# STFT (compatible con audio_lab.py)
# ══════════════════════════════════════════════════════════════════════════════

def stft_analyse(audio: np.ndarray, sr: int,
                 window_size: int = 4096,
                 hop_ratio: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT con ventana Hann. Devuelve (mags [frames×bins], fases, freqs_hz)."""
    hop    = max(1, int(window_size * hop_ratio))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / window_size)
    n_pad  = window_size // 2
    padded = np.pad(audio, (n_pad, n_pad + window_size), mode="constant")
    frames = (len(padded) - window_size) // hop + 1

    n_bins = window_size // 2 + 1
    mags   = np.zeros((frames, n_bins), dtype=np.float32)
    phases = np.zeros((frames, n_bins), dtype=np.float32)
    for i in range(frames):
        start = i * hop
        chunk = padded[start : start + window_size] * window
        X     = np.fft.rfft(chunk, n=window_size)
        mags[i]   = np.abs(X)
        phases[i] = np.angle(X)
    freqs = np.fft.rfftfreq(window_size, 1.0 / sr)
    return mags, phases, freqs

def _phase_vocoder_init(mags: np.ndarray, hop: int, n_fft: int) -> np.ndarray:
    """Inicialización de fase por acumulación de la frecuencia central de cada
    bin (truco clásico de phase vocoder / "identity phase locking"), en vez de
    fase aleatoria. Da a Griffin-Lim un punto de partida ya coherente entre
    frames consecutivos (cada bin avanza su fase exactamente lo que le
    correspondería a una senoide pura en su frecuencia central), lo que suele
    converger a un resultado menos "metálico" que partir de ruido puro,
    especialmente con pocas iteraciones."""
    frames, n_bins = mags.shape
    bin_advance = np.arange(n_bins) * (2 * np.pi * hop / n_fft)   # rad/hop por bin
    phases = np.outer(np.arange(frames), bin_advance)
    return np.mod(phases, 2 * np.pi)

def griffin_lim(mags: np.ndarray, hop_ratio: float,
                n_iter: int = 32, sr: int = SAMPLE_RATE,
                phase_init: str = "random") -> np.ndarray:
    """Griffin-Lim iterativo: estima fases coherentes a partir de magnitudes.

    phase_init:
      - "random"    (default, comportamiento histórico): fase inicial
        uniforme aleatoria (semilla fija, determinista).
      - "propagate": fase inicial por acumulación de frecuencia central por
        bin (ver _phase_vocoder_init) en vez de ruido — punto de partida más
        informado, normalmente converge algo mejor con pocas iteraciones.
    """
    n_bins  = mags.shape[1]
    n_fft   = (n_bins - 1) * 2
    hop     = max(1, int(n_fft * hop_ratio))
    frames  = mags.shape[0]
    win     = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)

    if phase_init == "propagate":
        phases = _phase_vocoder_init(mags, hop, n_fft)
    else:
        rng    = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, mags.shape)

    def _istft(ph: np.ndarray) -> np.ndarray:
        out_len = (frames - 1) * hop + n_fft
        out     = np.zeros(out_len, dtype=np.float64)
        norm    = np.zeros(out_len, dtype=np.float64)
        for i in range(frames):
            X     = mags[i] * np.exp(1j * ph[i])
            chunk = np.fft.irfft(X, n=n_fft)[:n_fft] * win
            start = i * hop
            out[start:start + n_fft]  += chunk
            norm[start:start + n_fft] += win ** 2
        nz = norm > 1e-10
        out[nz] /= norm[nz]
        return out

    def _stft_phases(audio: np.ndarray) -> np.ndarray:
        n_pad  = n_fft // 2
        padded = np.pad(audio, (n_pad, n_pad + n_fft))
        ph     = np.zeros(mags.shape, dtype=np.float64)
        for i in range(frames):
            start = i * hop
            chunk = padded[start:start + n_fft] * win
            ph[i] = np.angle(np.fft.rfft(chunk, n=n_fft))
        return ph

    for _ in range(n_iter):
        audio  = _istft(phases)
        phases = _stft_phases(audio)
    audio = _istft(phases)
    n_pad = n_fft // 2
    return audio[n_pad : n_pad + len(audio) - n_fft].astype(np.float32)

# ── normalización de espectrogramas (log dB → [0,1]) ─────────────────────────

def mags_to_norm(mags: np.ndarray, db_floor: float = DEFAULT_DB_FLOOR) -> np.ndarray:
    """Magnitudes STFT → [0,1] en escala log, normalizadas al pico del fichero."""
    ref = mags.max() + 1e-10
    db  = 20 * np.log10(mags / ref + 1e-10)
    db  = np.clip(db, db_floor, 0.0)
    return ((db - db_floor) / (-db_floor)).astype(np.float32)

def norm_to_mags(norm: np.ndarray, db_floor: float = DEFAULT_DB_FLOOR,
                 mag_max: float = 1.0) -> np.ndarray:
    """[0,1] log-normalizado → magnitudes STFT lineales."""
    norm = np.clip(norm, 0.0, 1.0)
    db   = norm * (-db_floor) + db_floor
    mags = (10 ** (db / 20.0)) * mag_max
    mags[norm < 1.0 / 255.0] = 0.0     # casi-negro → silencio exacto
    return mags.astype(np.float32)

def _resample_frames(arr: np.ndarray, n_out: int) -> np.ndarray:
    """Interpola linealmente [frames × dims] al nuevo número de frames."""
    n_in = arr.shape[0]
    if n_in == n_out:
        return arr.copy()
    old = np.arange(n_in, dtype=np.float64)
    new = np.linspace(0, n_in - 1, n_out)
    out = np.empty((n_out, arr.shape[1]), dtype=arr.dtype)
    for d in range(arr.shape[1]):
        out[:, d] = np.interp(new, old, arr[:, d])
    return out

def _resize_image(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Redimensiona una imagen float32 2-D [H×W] a otra resolución (bilineal)."""
    Image = _import_pil()
    h_out, w_out = out_hw
    if arr.shape == (h_out, w_out):
        return arr.astype(np.float32).copy()
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    img = img.resize((w_out, h_out), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# MODELOS TORCH (definidos lazy dentro de una factoría, torch es opcional)
# ══════════════════════════════════════════════════════════════════════════════

_TORCH_ZOO: Dict[str, object] = {}

def _torch_zoo() -> Dict[str, object]:
    """Importa torch una sola vez y define/cachea las clases de los modelos."""
    if _TORCH_ZOO:
        return _TORCH_ZOO
    torch = _import_torch()
    nn    = torch.nn

    class Snake(nn.Module):
        """Activación Snake (x + sin²(αx)/α), como en DAC/BigVGAN."""
        def __init__(self, channels: int):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        def forward(self, x):
            return x + torch.sin(self.alpha * x) ** 2 / (self.alpha + 1e-9)

    def _enc_channels(base: int, strides) -> List[int]:
        chans, c = [base], base
        for _ in strides:
            c = min(c * 2, 512)
            chans.append(c)
        return chans

    class Encoder(nn.Module):
        """wav [B,1,T] → latente [B,D,T/hop]."""
        def __init__(self, latent_dim: int, strides, base: int):
            super().__init__()
            chans  = _enc_channels(base, strides)
            layers = [nn.Conv1d(1, chans[0], 7, padding=3)]
            for i, s in enumerate(strides):
                layers += [Snake(chans[i]),
                           nn.Conv1d(chans[i], chans[i + 1], 2 * s,
                                     stride=s, padding=s // 2)]
            layers += [Snake(chans[-1]),
                       nn.Conv1d(chans[-1], latent_dim, 3, padding=1)]
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class Decoder(nn.Module):
        """latente [B,D,F] → wav [B,1,F·hop]."""
        def __init__(self, latent_dim: int, strides, base: int):
            super().__init__()
            chans  = _enc_channels(base, strides)   # mismo perfil, en espejo
            layers = [nn.Conv1d(latent_dim, chans[-1], 7, padding=3)]
            for i, s in reversed(list(enumerate(strides))):
                layers += [Snake(chans[i + 1]),
                           nn.ConvTranspose1d(chans[i + 1], chans[i], 2 * s,
                                              stride=s, padding=s // 2)]
            layers += [Snake(chans[0]),
                       nn.Conv1d(chans[0], 1, 7, padding=3),
                       nn.Tanh()]
            self.net = nn.Sequential(*layers)
        def forward(self, z):
            return self.net(z)

    class ResBlock1D(nn.Module):
        """Bloque residual con convoluciones dilatadas (MRF simplificado de
        HiFiGAN): varias dilataciones en paralelo sobre residuales sucesivos
        para ampliar el contexto sin perder resolución temporal."""
        def __init__(self, channels: int, kernel: int = 3, dilations=(1, 3, 5)):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Conv1d(channels, channels, kernel, dilation=d,
                          padding=(kernel - 1) * d // 2)
                for d in dilations])
        def forward(self, x):
            for conv in self.convs:
                x = x + conv(nn.functional.leaky_relu(x, 0.1))
            return x

    class LiteVocoder(nn.Module):
        """
        Vocoder neuronal ligero: magnitud STFT log-normalizada [B,n_bins,F]
        → forma de onda [B,1,F·hop]. Alternativa opcional a Griffin-Lim.

        Arquitectura estilo HiFiGAN simplificado (sin discriminador/GAN,
        entrenado solo con pérdida L1 de onda + STFT multi-escala — más
        barato de entrenar en CPU que la receta adversarial completa):
        proyección de bins a canales ocultos, seguida de tantas etapas de
        sobremuestreo transpuesto como haga falta para que el producto de
        sus factores sea igual al hop del STFT, cada una con un ResBlock1D.
        """
        def __init__(self, n_bins: int, hidden: int = 256,
                     upsample_rates=(8, 8, 4, 4)):
            super().__init__()
            self.hop = int(np.prod(upsample_rates))
            self.in_proj = nn.Conv1d(n_bins, hidden, 7, padding=3)
            ups, ch = [], hidden
            for r in upsample_rates:
                out_ch = max(ch // 2, 32)
                pad    = r // 2 + (r % 2)
                ups.append(nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.ConvTranspose1d(ch, out_ch, 2 * r, stride=r,
                                       padding=pad, output_padding=r % 2),
                    ResBlock1D(out_ch),
                ))
                ch = out_ch
            self.ups = nn.ModuleList(ups)
            self.out_proj = nn.Sequential(
                nn.LeakyReLU(0.1), nn.Conv1d(ch, 1, 7, padding=3), nn.Tanh())
        def forward(self, mag):        # mag [B, n_bins, F] ∈ [0,1]
            x = self.in_proj(mag)
            for up in self.ups:
                x = up(x)
            return self.out_proj(x)    # [B, 1, ≈F·hop]

    class FrameMapper(nn.Module):
        """Traductor frame-a-frame con contexto temporal (conv 1-D sobre frames)."""
        def __init__(self, in_dim: int, out_dim: int,
                     hidden: int = 256, kernel: int = 9):
            super().__init__()
            pad = kernel // 2
            self.net = nn.Sequential(
                nn.Conv1d(in_dim, hidden, kernel, padding=pad), nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel, padding=pad), nn.GELU(),
                nn.Conv1d(hidden, out_dim, 1),
            )
        def forward(self, x):        # [B, in_dim, F] → [B, out_dim, F]
            return self.net(x)

    # ── pix2pix: U-Net generador + PatchGAN discriminador (imagen→imagen) ──────
    # Se usan para spectrogram-to-latent-train_step2: ambas direcciones
    # (espectrograma→latente y latente→espectrograma) se tratan como imágenes
    # de 1 canal [tiempo × dim] y se traducen con la receta pix2pix (Isola et al.).

    class UNetDown(nn.Module):
        def __init__(self, in_c, out_c, norm=True):
            super().__init__()
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1, bias=not norm)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class UNetUp(nn.Module):
        def __init__(self, in_c, out_c, dropout=0.0):
            super().__init__()
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(out_c), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)
        def forward(self, x, skip):
            x = self.net(x)
            # recorta/rellena por si la skip-connection difiere en 1px
            dh = skip.shape[-2] - x.shape[-2]
            dw = skip.shape[-1] - x.shape[-1]
            if dh != 0 or dw != 0:
                x = nn.functional.pad(x, (0, max(dw, 0), 0, max(dh, 0)))
                x = x[..., :skip.shape[-2], :skip.shape[-1]]
            return torch.cat([x, skip], dim=1)

    class UNetGenerator(nn.Module):
        """
        Generador pix2pix (U-Net de 6 niveles) para traducir una imagen
        [B,1,H,W] en otra [B,1,H_out,W_out] (mismo H×W internamente; el
        remuestreo a la resolución temporal destino se hace fuera, antes/
        después de invocar la red).
        """
        def __init__(self, base: int = 64):
            super().__init__()
            self.d1 = UNetDown(1,      base,     norm=False)
            self.d2 = UNetDown(base,   base * 2)
            self.d3 = UNetDown(base*2, base * 4)
            self.d4 = UNetDown(base*4, base * 8)
            self.d5 = UNetDown(base*8, base * 8)
            self.u1 = UNetUp(base*8,       base * 8, dropout=0.5)
            self.u2 = UNetUp(base*8 + base*8, base * 4)
            self.u3 = UNetUp(base*4 + base*4, base * 2)
            self.u4 = UNetUp(base*2 + base*2, base)
            self.final = nn.Sequential(
                nn.ConvTranspose2d(base + base, 1, 4, stride=2, padding=1),
                nn.Sigmoid(),   # salida en [0,1], como las imágenes normalizadas
            )
        def forward(self, x):
            d1 = self.d1(x)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            d4 = self.d4(d3)
            d5 = self.d5(d4)
            u1 = self.u1(d5, d4)
            u2 = self.u2(u1, d3)
            u3 = self.u3(u2, d2)
            u4 = self.u4(u3, d1)
            out = self.final(u4)
            # ajuste fino de tamaño (las convs con stride pueden perder ±1px)
            if out.shape[-2:] != x.shape[-2:]:
                out = nn.functional.interpolate(out, size=x.shape[-2:],
                                                mode="bilinear", align_corners=False)
            return out

    class PatchDiscriminator(nn.Module):
        """PatchGAN: clasifica parches locales de la imagen como reales/falsos.
        Entrada: concatenación canal-wise de (condición, imagen) → [B,2,H,W]."""
        def __init__(self, base: int = 64):
            super().__init__()
            def block(in_c, out_c, norm=True, stride=2):
                layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1,
                                    bias=not norm)]
                if norm:
                    layers.append(nn.InstanceNorm2d(out_c))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            self.net = nn.Sequential(
                *block(2,        base,   norm=False),
                *block(base,     base*2),
                *block(base*2,   base*4),
                *block(base*4,   base*8, stride=1),
                nn.Conv2d(base*8, 1, 4, stride=1, padding=1),
            )
        def forward(self, cond, img):
            x = torch.cat([cond, img], dim=1)
            return self.net(x)

    # ── causal: WaveNet-lite autorregresivo (columna t ← columnas < t) ──────────
    # Predice la columna latente t a partir de las columnas latentes ANTERIORES
    # (convoluciones causales dilatadas, campo receptivo ≈ 2^n_layers) más un
    # condicionamiento del espectrograma COMPLETO (no causal: en generación
    # offline el espectrograma entero ya está disponible, pasado y futuro).
    # El llamante desplaza el latente de entrada una posición (spec_fn/
    # _causal_generate) para que la predicción en t nunca vea el Z[t] real.

    class CausalConv1d(nn.Module):
        """Conv1d causal: solo mira al pasado (padding a la izquierda)."""
        def __init__(self, in_c, out_c, kernel=2, dilation=1):
            super().__init__()
            self.pad  = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_c, out_c, kernel, dilation=dilation)
        def forward(self, x):
            x = nn.functional.pad(x, (self.pad, 0))
            return self.conv(x)

    class CausalResBlock(nn.Module):
        """Bloque residual con convolución causal dilatada, estilo WaveNet."""
        def __init__(self, channels, dilation):
            super().__init__()
            self.conv = CausalConv1d(channels, channels, kernel=2, dilation=dilation)
            self.proj = nn.Conv1d(channels, channels, 1)
            self.act  = nn.GELU()
        def forward(self, x):
            y = self.act(self.conv(x))
            y = self.proj(y)
            return x + y

    class CausalGenerator(nn.Module):
        """
        Traductor autorregresivo espectrograma→latente: predice la columna
        latente t a partir de las columnas latentes t-1..t-context (pila de
        CausalResBlock con dilatación 1,2,4,...) sumadas a un condicionamiento
        del espectrograma completo (spec_enc, NO causal — se puede precomputar
        una sola vez para todo el clip y reutilizar en cada paso de generación,
        ver _causal_generate). n_layers se elige para que el campo receptivo
        (2**n_layers) cubra al menos --context frames.
        """
        def __init__(self, n_bins, D, hidden=64, context=32):
            super().__init__()
            n_layers = max(1, math.ceil(math.log2(max(context, 1) + 1)))
            self.in_proj  = nn.Conv1d(D, hidden, 1)
            self.spec_enc = nn.Sequential(
                nn.Conv1d(n_bins, hidden, 5, padding=2), nn.GELU(),
                nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2), nn.GELU(),
                nn.Conv1d(hidden, hidden, 1),
            )
            self.blocks = nn.ModuleList(
                [CausalResBlock(hidden, 2 ** i) for i in range(n_layers)])
            self.out_proj = nn.Conv1d(hidden, D, 1)
            self.n_layers = n_layers
            self.receptive_field = 2 ** n_layers

        def encode_spec(self, S_cond):
            """S_cond [B,bins,T] → spec_feat [B,hidden,T]. No causal: usa todo
            el clip. Se precomputa una vez y se reutiliza en cada paso t."""
            return self.spec_enc(S_cond)

        def forward(self, Z_shifted, spec_feat):
            """Z_shifted [B,D,T] (desplazado 1 frame por el llamante) +
            spec_feat [B,hidden,T] (de encode_spec) → Zn [B,D,T] ∈ [0,1]."""
            h = self.in_proj(Z_shifted) + spec_feat
            for blk in self.blocks:
                h = blk(h)
            return torch.sigmoid(self.out_proj(h))

    class CausalDiscriminator1D(nn.Module):
        """PatchGAN 1-D: juzga parches temporales de la secuencia latente
        como reales/falsos, condicionado al espectrograma — mismo espíritu
        que PatchDiscriminator, pero sobre [B,canal,T] en vez de [B,canal,H,W]."""
        def __init__(self, n_bins, D, base=64):
            super().__init__()
            def block(in_c, out_c, stride=2, norm=True):
                layers = [nn.Conv1d(in_c, out_c, 4, stride=stride, padding=1,
                                    bias=not norm)]
                if norm:
                    layers.append(nn.InstanceNorm1d(out_c))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers
            self.net = nn.Sequential(
                *block(n_bins + D, base,   norm=False),
                *block(base,       base*2),
                *block(base*2,     base*4),
                *block(base*4,     base*8, stride=1),
                nn.Conv1d(base*8, 1, 4, stride=1, padding=1),
            )
        def forward(self, S_cond, Z_seq):
            x = torch.cat([S_cond, Z_seq], dim=1)
            return self.net(x)

    _TORCH_ZOO.update(torch=torch, nn=nn, Snake=Snake,
                      Encoder=Encoder, Decoder=Decoder, FrameMapper=FrameMapper,
                      UNetGenerator=UNetGenerator, PatchDiscriminator=PatchDiscriminator,
                      CausalGenerator=CausalGenerator,
                      CausalDiscriminator1D=CausalDiscriminator1D,
                      ResBlock1D=ResBlock1D, LiteVocoder=LiteVocoder)
    return _TORCH_ZOO

def _get_device(name: str):
    zoo, torch = _torch_zoo(), _torch_zoo()["torch"]
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  [device] {name}")
    return torch.device(name)

def _multiscale_stft_loss(torch, x, y):
    """Pérdida STFT multi-escala (L1 sobre magnitud lineal y log)."""
    loss = 0.0
    for w in (512, 1024, 2048):
        win = torch.hann_window(w, device=x.device)
        X = torch.stft(x, w, w // 4, window=win, return_complex=True).abs()
        Y = torch.stft(y, w, w // 4, window=win, return_complex=True).abs()
        loss = loss + (X - Y).abs().mean() \
                    + (torch.log(X + 1e-5) - torch.log(Y + 1e-5)).abs().mean()
    return loss / 3.0

def _torch_load(path: str):
    torch = _torch_zoo()["torch"]
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTS: CODER Y MAPPER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Coder:
    encoder:    object
    decoder:    object
    sr:         int
    latent_dim: int
    strides:    Tuple[int, ...]
    hop:        int
    lat_min:    np.ndarray     # [D] estadísticas del corpus de entrenamiento
    lat_max:    np.ndarray     # [D]
    device:     object

def _load_pretrained_dac_coder(path: str, device_name: str = "auto") -> "Coder":
    """
    Carga un checkpoint OFICIAL de Descript Audio Codec (.pth, p.ej. el
    'weights.pth' publicado en descriptinc/descript-audio-codec) y lo expone
    como un Coder de latent_lab. Se usa el encoder/decoder CONTINUOS del DAC
    (la representación antes de la cuantización RVQ), que es la que mejor se
    presta a edición como imagen; la cuantización discreta de DAC no se usa.

    latent_dim aquí es 1024 (mucho mayor que nuestro coder propio) porque es
    la anchura real del cuello de botella de DAC antes de RVQ.
    """
    dac    = _import_dac()
    torch  = _torch_zoo()["torch"]
    device = _get_device(device_name)
    model  = dac.DAC.load(path)
    model.to(device).eval()
    hop = int(model.hop_length)
    lat_dim = int(model.latent_dim)
    # DAC no publica min/max del corpus; usamos un rango simétrico amplio
    # calibrado empíricamente sobre el propio modelo (ver README de resultados).
    lat_min = np.full(lat_dim, -8.0, dtype=np.float32)
    lat_max = np.full(lat_dim,  8.0, dtype=np.float32)
    print(f"  [coder] {path}: DAC oficial pre-entrenado  D={lat_dim}  hop={hop}  "
          f"sr={model.sample_rate}  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M parámetros)")
    return Coder(model.encoder, model.decoder, int(model.sample_rate), lat_dim,
                 tuple(), hop, lat_min, lat_max, device)

def load_coder(path: str, device_name: str = "auto") -> Coder:
    if Path(path).suffix.lower() == ".pth":
        return _load_pretrained_dac_coder(path, device_name)
    zoo    = _torch_zoo()
    ckpt   = _torch_load(path)
    if ckpt.get("kind") != "latent_lab_coder":
        sys.exit(f"✗  {path} no es un checkpoint de train-coder ni un .pth de DAC oficial")
    device  = _get_device(device_name)
    strides = tuple(ckpt["strides"])
    enc = zoo["Encoder"](ckpt["latent_dim"], strides, ckpt["base_channels"])
    dec = zoo["Decoder"](ckpt["latent_dim"], strides, ckpt["base_channels"])
    enc.load_state_dict(ckpt["encoder"]);  enc.to(device).eval()
    dec.load_state_dict(ckpt["decoder"]);  dec.to(device).eval()
    print(f"  [coder] {path}: D={ckpt['latent_dim']}  hop={ckpt['hop']}  "
          f"sr={ckpt['sr']}  strides={strides}")
    return Coder(enc, dec, int(ckpt["sr"]), int(ckpt["latent_dim"]), strides,
                 int(ckpt["hop"]),
                 np.array(ckpt["lat_min"], dtype=np.float32),
                 np.array(ckpt["lat_max"], dtype=np.float32),
                 device)

def _download_file(url: str, dest: Path) -> None:
    """Descarga con barra de progreso simple (solo stdlib, sin requests)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "latent_lab/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=30) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done  = 0
        chunk = 1 << 20   # 1 MiB
        with open(tmp, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"\r  [download] {done/1e6:6.1f}/{total/1e6:.1f} MB "
                          f"({pct:5.1f}%)", end="", flush=True)
                else:
                    print(f"\r  [download] {done/1e6:6.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)

def cmd_download_pretrained(args):
    """
    Descarga un checkpoint OFICIAL de Descript Audio Codec desde GitHub
    Releases y lo deja listo para usarse como --coder en wav-to-latent /
    latent-to-wav / spectrogram-to-latent-train, exactamente como cualquier
    otro .pth.
    """
    variant = args.variant
    if variant not in PRETRAINED_DAC_MODELS:
        opciones = ", ".join(PRETRAINED_DAC_MODELS)
        sys.exit(f"✗  Variante desconocida: {variant!r}. Opciones: {opciones}")

    url, sr, desc = PRETRAINED_DAC_MODELS[variant]
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dac_{variant}.pth"

    print(f"  [download-pretrained] {variant}  (sr={sr}Hz)")
    print(f"  [download-pretrained] {desc}")

    if out_path.exists() and not args.force:
        print(f"  ✓  Ya existe: {out_path}  (usa --force para re-descargar)")
    else:
        print(f"  [download-pretrained] Descargando desde {url}")
        try:
            _download_file(url, out_path)
        except Exception as e:
            sys.exit(f"✗  Fallo al descargar: {e}")
        print(f"  ✓  Descargado → {out_path}  "
              f"({out_path.stat().st_size/1e6:.1f} MB)")

    if args.no_verify:
        print(f"  [download-pretrained] Verificación omitida (--no-verify)")
        return

    print(f"  [download-pretrained] Verificando que el checkpoint carga y funciona...")
    dac    = _import_dac()
    model  = dac.DAC.load(str(out_path))
    n      = sum(p.numel() for p in model.parameters())
    print(f"  ✓  Carga OK: sample_rate={model.sample_rate}  hop_length={model.hop_length}  "
          f"latent_dim={model.latent_dim}  ({n/1e6:.1f}M parámetros)")

    # smoke-test funcional: encode+decode de una señal sintética corta
    torch = _torch_zoo()["torch"]
    model.eval()
    with torch.no_grad():
        t = np.linspace(0, 1.0, model.sample_rate, dtype=np.float32)
        x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        xt = torch.from_numpy(x).view(1, 1, -1)
        xt = model.preprocess(xt, model.sample_rate)
        y  = model.decoder(model.encoder(xt))
    # tolerancia de tamaño: el decoder de DAC puede producir unas pocas
    # muestras menos/más que la entrada según el redondeo interno de sus
    # convoluciones transpuestas; una diferencia de hasta un hop es normal.
    len_diff = abs(y.shape[-1] - xt.shape[-1])
    ok = len_diff <= model.hop_length and bool(torch.isfinite(y).all())
    print(f"  {'✓' if ok else '✗'}  Smoke-test encode→decode: "
          f"{'OK' if ok else 'FALLO'}  (salida {tuple(y.shape)})")
    print()
    print(f"  Listo para usar:")
    print(f"    python3 latent_lab.py wav-to-latent tu_audio.wav "
          f"--coder {out_path} --png -o latente.npz")
    print(f"    python3 latent_lab.py latent-to-wav latente.npz "
          f"--coder {out_path} -o resultado.wav")

def encode_wav(coder: Coder, audio: np.ndarray) -> np.ndarray:
    """WAV mono float32 → latentes [frames × D]."""
    torch = _torch_zoo()["torch"]
    pad   = (-len(audio)) % coder.hop
    if pad:
        audio = np.pad(audio, (0, pad))
    with torch.no_grad():
        x = torch.from_numpy(audio).to(coder.device).view(1, 1, -1)
        z = coder.encoder(x)[0]                      # [D, F]
    return z.cpu().numpy().T.astype(np.float32)     # [F, D]

def decode_latents(coder: Coder, latents: np.ndarray) -> np.ndarray:
    """Latentes [frames × D] → WAV mono float32."""
    torch = _torch_zoo()["torch"]
    with torch.no_grad():
        z = torch.from_numpy(latents.T.astype(np.float32)) \
                 .to(coder.device).unsqueeze(0)      # [1, D, F]
        y = coder.decoder(z)[0, 0]
    return y.cpu().numpy().astype(np.float32)

def _factorize_hop(hop: int) -> List[int]:
    """Descompone hop en factores de sobremuestreo razonables para
    LiteVocoder (mayor a menor), probando primero factores "bonitos" para
    convoluciones transpuestas. Si no encuentra una descomposición exacta con
    esos factores, recurre a la factorización prima simple."""
    preferred = [8, 6, 5, 4, 3, 2]
    factors, remaining = [], hop
    changed = True
    while remaining > 1 and changed:
        changed = False
        for f in preferred:
            if remaining % f == 0:
                factors.append(f)
                remaining //= f
                changed = True
                break
    if remaining > 1:      # resto no descomponible con los factores preferidos
        factors.append(remaining)
    return factors

def load_vocoder(path: str, device_name: str = "auto"):
    """Carga un checkpoint de train-vocoder (LiteVocoder)."""
    zoo  = _torch_zoo()
    ckpt = _torch_load(path)
    if ckpt.get("kind") != "latent_lab_vocoder":
        sys.exit(f"✗  {path} no es un checkpoint de train-vocoder")
    device = _get_device(device_name)
    voc = zoo["LiteVocoder"](ckpt["n_bins"], ckpt["hidden"],
                             tuple(ckpt["upsample_rates"]))
    voc.load_state_dict(ckpt["state_dict"])
    voc.to(device).eval()
    print(f"  [vocoder] {path}: {ckpt['n_bins']} bins  hop={ckpt['hop']}  "
          f"sr={ckpt['sr']}  ({sum(p.numel() for p in voc.parameters())/1e6:.2f}M parámetros)")
    return voc, ckpt

def vocoder_synthesize(voc, norm_mags: np.ndarray) -> np.ndarray:
    """Magnitud log-normalizada [frames × bins] ∈ [0,1] → forma de onda,
    usando un LiteVocoder ya cargado (en el dispositivo correspondiente)."""
    torch  = _torch_zoo()["torch"]
    device = next(voc.parameters()).device
    with torch.no_grad():
        mag = torch.from_numpy(norm_mags.T[None]).to(device).float()  # [1,bins,F]
        y   = voc(mag)[0, 0].cpu().numpy()
    return y.astype(np.float32)

def latents_normalize(latents: np.ndarray, lat_min: np.ndarray,
                      lat_max: np.ndarray) -> np.ndarray:
    """[frames×D] → [0,1] por componente según estadísticas del corpus."""
    span = np.maximum(lat_max - lat_min, 1e-6)
    return np.clip((latents - lat_min) / span, 0.0, 1.0).astype(np.float32)

def latents_denormalize(norm: np.ndarray, lat_min: np.ndarray,
                        lat_max: np.ndarray) -> np.ndarray:
    span = np.maximum(lat_max - lat_min, 1e-6)
    return (norm * span + lat_min).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# NPZ / PNG DE LATENTES
# ══════════════════════════════════════════════════════════════════════════════

def save_latents_npz(path: str, latents: np.ndarray, sr: int, hop: int,
                     lat_min: np.ndarray, lat_max: np.ndarray) -> None:
    np.savez_compressed(path,
                        latents=latents.astype(np.float32),
                        sample_rate=np.array(sr),
                        hop_length=np.array(hop),
                        lat_min=lat_min.astype(np.float32),
                        lat_max=lat_max.astype(np.float32))
    F, D = latents.shape
    print(f"  ✓  NPZ latente → {path}  ({F} frames × {D} componentes, "
          f"~{F * hop / sr:.2f}s)")

def load_latents_npz(path: str):
    data = np.load(path)
    if "latents" not in data.files:
        kind = "espectrograma STFT (audio_lab)" if "magnitudes" in data.files \
               else "desconocido"
        sys.exit(f"✗  {path} no contiene latentes (NPZ detectado: {kind}). "
                 f"¿Querías usar spectrogram-to-latent primero?")
    return (data["latents"].astype(np.float32), int(data["sample_rate"]),
            int(data["hop_length"]),
            data["lat_min"].astype(np.float32), data["lat_max"].astype(np.float32))

def latents_to_png(latents: np.ndarray, sr: int, hop: int,
                   lat_min: np.ndarray, lat_max: np.ndarray,
                   out_path: str, flip_y: bool = True) -> None:
    """
    Latentes [frames × D] → PNG escala de grises [D(alto) × frames(ancho)].
    Componente 0 abajo (como la frecuencia 0 en audio_lab), salvo flip_y=False.
    Metadatos en chunks tEXt + sidecar .ll.json (fallback anti-GIMP).
    """
    Image = _import_pil()
    from PIL import PngImagePlugin

    norm = latents_normalize(latents, lat_min, lat_max)   # [F, D] ∈ [0,1]
    img_data = norm.T                                     # [D × F]
    if flip_y:
        img_data = img_data[::-1, :]
    pixels = (img_data * 255).clip(0, 255).astype(np.uint8)
    img    = Image.fromarray(pixels, mode="L")

    F, D = latents.shape
    fields = {
        "latent_lab_sr":       str(sr),
        "latent_lab_hop":      str(hop),
        "latent_lab_n_frames": str(F),
        "latent_lab_n_comp":   str(D),
        "latent_lab_flip_y":   "1" if flip_y else "0",
        "latent_lab_lat_min":  json.dumps([round(float(v), 6) for v in lat_min]),
        "latent_lab_lat_max":  json.dumps([round(float(v), 6) for v in lat_max]),
    }
    meta = PngImagePlugin.PngInfo()
    for k, v in fields.items():
        meta.add_text(k, v)

    sidecar = Path(out_path).with_suffix(".ll.json")
    with open(sidecar, "w") as f:
        json.dump(fields, f, indent=2)

    img.save(out_path, format="PNG", pnginfo=meta)
    print(f"  ✓  PNG → {out_path}  ({F}×{D}px)")
    print(f"     Metadatos embebidos: sr={sr} hop={hop} D={D}")
    print(f"     Sidecar JSON → {sidecar}  (fallback si el editor elimina tEXt)")

def png_to_latents(path: str, coder: Optional[Coder] = None):
    """
    PNG escala de grises → (latentes [frames×D], sr, hop, lat_min, lat_max).
    Lee metadatos tEXt; si faltan, prueba el sidecar .ll.json; si tampoco,
    usa las estadísticas del coder pasado (obligatorio en ese caso).
    """
    Image  = _import_pil()
    img    = Image.open(path).convert("L")
    pixels = np.array(img, dtype=np.float64)            # [H × W]
    meta   = dict(img.info)

    sidecar = Path(path).with_suffix(".ll.json")
    if "latent_lab_sr" not in meta and sidecar.exists():
        with open(sidecar) as f:
            meta.update(json.load(f))
        print(f"  [png-to-latent] Metadatos leídos desde sidecar: {sidecar.name}")

    if "latent_lab_sr" in meta:
        sr      = int(float(meta["latent_lab_sr"]))
        hop     = int(float(meta["latent_lab_hop"]))
        flip_y  = str(meta.get("latent_lab_flip_y", "1")) in ("1", "True", "true")
        lat_min = np.array(json.loads(meta["latent_lab_lat_min"]), dtype=np.float32)
        lat_max = np.array(json.loads(meta["latent_lab_lat_max"]), dtype=np.float32)
    elif coder is not None:
        print(f"  ⚠  PNG sin metadatos ni sidecar — usando estadísticas del coder")
        sr, hop, flip_y = coder.sr, coder.hop, True
        lat_min, lat_max = coder.lat_min, coder.lat_max
    else:
        sys.exit("✗  PNG sin metadatos ni sidecar y sin --coder para deducirlos")

    H, W = pixels.shape
    D    = len(lat_min)
    if H != D and W == D:
        pixels = pixels.T
        H, W   = pixels.shape
        print(f"  ⚠  PNG rotado detectado (W coincidía con D={D}) — transpuesto")
    if H != D:
        sys.exit(f"✗  Altura del PNG ({H}) ≠ componentes latentes ({D})")

    if flip_y:
        pixels = pixels[::-1, :]
    norm    = (pixels.T / 255.0).astype(np.float32)     # [frames × D]
    latents = latents_denormalize(norm, lat_min, lat_max)
    print(f"  [png-to-latent] {W}×{H}px → {W} frames × {H} componentes")
    return latents, sr, hop, lat_min, lat_max

def _load_latents_any(path: str, coder: Optional[Coder] = None):
    """Carga latentes desde .npz o .png indistintamente."""
    ext = Path(path).suffix.lower()
    if ext == ".npz":
        return load_latents_npz(path)
    elif ext == ".png":
        return png_to_latents(path, coder)
    else:
        sys.exit(f"✗  Formato no soportado: {ext!r}  (usa .npz o .png)")

# ══════════════════════════════════════════════════════════════════════════════
# CORPUS DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def _load_corpus(paths: List[str], sr: int) -> List[np.ndarray]:
    corpus = []
    for p in paths:
        audio, _ = read_wav(p, target_sr=sr)
        if len(audio) < sr // 10:
            print(f"  ⚠  {Path(p).name}: <0.1s, ignorado")
            continue
        corpus.append(audio)
        print(f"  [corpus] {Path(p).name}: {len(audio)/sr:.1f}s")
    if not corpus:
        sys.exit("✗  Ningún fichero de audio utilizable")
    total = sum(len(a) for a in corpus) / sr
    print(f"  [corpus] {len(corpus)} ficheros, {total:.1f}s en total")
    return corpus

def _random_crop_batch(torch, corpus: List[np.ndarray], batch: int,
                       segment: int, rng: np.random.Generator, device):
    """Batch [B, 1, segment] de recortes aleatorios del corpus."""
    out = np.zeros((batch, 1, segment), dtype=np.float32)
    for b in range(batch):
        a = corpus[rng.integers(len(corpus))]
        if len(a) <= segment:
            out[b, 0, :len(a)] = a
        else:
            start = rng.integers(len(a) - segment)
            out[b, 0] = a[start : start + segment]
    return torch.from_numpy(out).to(device)

def _fix_wav_length(torch, y, target_len: int):
    """Recorta o rellena con ceros [B,1,T] a exactamente target_len muestras
    (las ConvTranspose1d del vocoder pueden desviarse en unas pocas muestras
    del objetivo exacto, igual que ya ocurre con las skip-connections de
    UNetUp más arriba)."""
    T = y.shape[-1]
    if T == target_len:
        return y
    if T > target_len:
        return y[..., :target_len]
    return torch.nn.functional.pad(y, (0, target_len - T))

def _latent_stats(coder_like, corpus: List[np.ndarray],
                  hop: int, device, max_seconds: float = 60.0,
                  sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """min/max por componente sobre el corpus (ampliado un 5% de margen)."""
    torch  = _torch_zoo()["torch"]
    mins, maxs = [], []
    with torch.no_grad():
        for a in corpus:
            a = a[: int(max_seconds * sr)]
            pad = (-len(a)) % hop
            if pad:
                a = np.pad(a, (0, pad))
            x = torch.from_numpy(a).to(device).view(1, 1, -1)
            z = coder_like(x)[0].cpu().numpy()      # [D, F]
            mins.append(z.min(axis=1))
            maxs.append(z.max(axis=1))
    lat_min = np.min(mins, axis=0)
    lat_max = np.max(maxs, axis=0)
    margin  = 0.05 * np.maximum(lat_max - lat_min, 1e-6)
    return (lat_min - margin).astype(np.float32), (lat_max + margin).astype(np.float32)

# ══════════════════════════════════════════════════════════════════════════════
# COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train_coder(args):
    """WAVs → entrena el autoencoder wav↔latente → coder.pt"""
    zoo    = _torch_zoo()
    torch  = zoo["torch"]
    device = _get_device(args.device)
    sr     = args.sr

    strides = tuple(args.strides)
    if any(s % 2 for s in strides):
        sys.exit("✗  Todos los strides deben ser pares (ej: 2 4 8 8)")
    hop = int(np.prod(strides))
    if args.segment % hop:
        sys.exit(f"✗  --segment debe ser múltiplo del hop ({hop})")

    print(f"  [train-coder] D={args.latent_dim}  strides={strides}  hop={hop}  "
          f"({sr/hop:.1f} frames/s)")
    corpus = _load_corpus(args.inputs, sr)

    enc = zoo["Encoder"](args.latent_dim, strides, args.base_channels).to(device)
    dec = zoo["Decoder"](args.latent_dim, strides, args.base_channels).to(device)
    n_params = sum(p.numel() for p in enc.parameters()) \
             + sum(p.numel() for p in dec.parameters())
    print(f"  [train-coder] {n_params/1e6:.2f}M parámetros  "
          f"batch={args.batch}  segment={args.segment}  steps={args.steps}")

    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=args.lr, betas=(0.8, 0.99))
    rng = np.random.default_rng(args.seed)

    enc.train(); dec.train()
    for step in range(1, args.steps + 1):
        x = _random_crop_batch(torch, corpus, args.batch, args.segment, rng, device)
        y = dec(enc(x))
        loss_wav  = (x - y).abs().mean()
        loss_stft = _multiscale_stft_loss(torch, x[:, 0], y[:, 0])
        loss      = loss_wav + loss_stft
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(dec.parameters()), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f"  [train-coder] step {step:>6}/{args.steps}  "
                  f"loss={loss.item():.4f}  (wav={loss_wav.item():.4f}  "
                  f"stft={loss_stft.item():.4f})")

    enc.eval(); dec.eval()
    print(f"  [train-coder] Calculando estadísticas del espacio latente...")
    lat_min, lat_max = _latent_stats(enc, corpus, hop, device, sr=sr)
    print(f"  [train-coder] lat_min∈[{lat_min.min():.3f},{lat_min.max():.3f}]  "
          f"lat_max∈[{lat_max.min():.3f},{lat_max.max():.3f}]")

    out_path = args.output or "coder.pt"
    torch.save({
        "kind":          "latent_lab_coder",
        "sr":            sr,
        "latent_dim":    args.latent_dim,
        "strides":       list(strides),
        "base_channels": args.base_channels,
        "hop":           hop,
        "lat_min":       [float(v) for v in lat_min],
        "lat_max":       [float(v) for v in lat_max],
        "encoder":       enc.state_dict(),
        "decoder":       dec.state_dict(),
    }, out_path)
    print(f"  ✓  Coder → {out_path}")


def cmd_train_vocoder(args):
    """
    WAVs → entrena un vocoder neuronal ligero (LiteVocoder) → vocoder.pt.

    Alternativa OPCIONAL a Griffin-Lim para reconstruir audio a partir de
    magnitud STFT en latent-to-spectrogram / pca-to-intermediate (--vocoder).
    Si nunca entrenas uno, el comportamiento por defecto de esos comandos no
    cambia: siguen usando Griffin-Lim.
    """
    zoo    = _torch_zoo()
    torch  = zoo["torch"]
    device = _get_device(args.device)
    sr     = args.sr
    window, hop_ratio = args.window, args.hop
    hop    = max(1, int(window * hop_ratio))
    n_bins = window // 2 + 1

    rates = list(args.upsample_rates) if args.upsample_rates else _factorize_hop(hop)
    prod  = int(np.prod(rates))
    if prod != hop:
        sys.exit(f"✗  --upsample-rates {rates} multiplican {prod} ≠ hop STFT "
                 f"{hop} (window×hop-ratio). Ajusta manualmente o cambia "
                 f"--window/--hop para que factoricen bien.")

    print(f"  [train-vocoder] window={window} hop={hop} ({n_bins} bins)  "
          f"upsample_rates={rates}")
    corpus = _load_corpus(args.inputs, sr)

    voc = zoo["LiteVocoder"](n_bins, args.hidden, tuple(rates)).to(device)
    n_params = sum(p.numel() for p in voc.parameters())
    print(f"  [train-vocoder] {n_params/1e6:.2f}M parámetros  batch={args.batch}  "
          f"frames={args.frames}  steps={args.steps}")

    opt     = torch.optim.Adam(voc.parameters(), lr=args.lr, betas=(0.8, 0.99))
    rng     = np.random.default_rng(args.seed)
    segment = args.frames * hop

    voc.train()
    for step in range(1, args.steps + 1):
        x = _random_crop_batch(torch, corpus, args.batch, segment, rng, device)
        mags_np = np.stack([
            mags_to_norm(stft_analyse(x[b, 0].cpu().numpy(), sr,
                                      window, hop_ratio)[0], args.db_floor)
            for b in range(args.batch)])                      # [B, frames, bins]
        mag = torch.from_numpy(mags_np.transpose(0, 2, 1)).to(device)  # [B,bins,F]
        y   = voc(mag)
        y   = _fix_wav_length(torch, y, x.shape[-1])
        loss_wav  = (x - y).abs().mean()
        loss_stft = _multiscale_stft_loss(torch, x[:, 0], y[:, 0])
        loss      = loss_wav + loss_stft
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(voc.parameters(), 1.0)
        opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f"  [train-vocoder] step {step:>6}/{args.steps}  "
                  f"loss={loss.item():.4f}  (wav={loss_wav.item():.4f}  "
                  f"stft={loss_stft.item():.4f})")

    voc.eval()
    out_path = args.output or "vocoder.pt"
    torch.save({
        "kind":            "latent_lab_vocoder",
        "sr":              sr,
        "window":          window,
        "hop_ratio":       hop_ratio,
        "hop":             hop,
        "n_bins":          n_bins,
        "hidden":          args.hidden,
        "upsample_rates":  rates,
        "db_floor":        args.db_floor,
        "state_dict":      voc.state_dict(),
    }, out_path)
    print(f"  ✓  Vocoder → {out_path}")
    print(f"     Úsalo con: --vocoder {out_path} en latent-to-spectrogram --wav")


def cmd_wav_to_latent(args):
    """WAV → representación tiempo × componentes latentes (.npz [+ PNG])."""
    coder    = load_coder(args.coder, args.device)
    audio, _ = read_wav(args.input, target_sr=coder.sr)
    print(f"  [wav-to-latent] {len(audio)/coder.sr:.2f}s → encoder...")
    latents = encode_wav(coder, audio)

    out_path = args.output or Path(args.input).stem + "_latent.npz"
    save_latents_npz(out_path, latents, coder.sr, coder.hop,
                     coder.lat_min, coder.lat_max)
    if args.png:
        png_path = str(Path(out_path).with_suffix(".png"))
        latents_to_png(latents, coder.sr, coder.hop,
                       coder.lat_min, coder.lat_max, png_path,
                       flip_y=not args.no_flip_y)


def cmd_latent_to_wav(args):
    """NPZ o PNG de latentes → WAV vía decoder."""
    coder = load_coder(args.coder, args.device)
    latents, sr, hop, _, _ = _load_latents_any(args.input, coder)
    if latents.shape[1] != coder.latent_dim:
        sys.exit(f"✗  El fichero tiene {latents.shape[1]} componentes pero el "
                 f"coder espera {coder.latent_dim}")
    if hop != coder.hop:
        print(f"  ⚠  hop del fichero ({hop}) ≠ hop del coder ({coder.hop}) — "
              f"se usa el del coder")
    print(f"  [latent-to-wav] {latents.shape[0]} frames × "
          f"{latents.shape[1]} componentes → decoder...")
    audio = decode_latents(coder, latents)
    if args.normalize:
        peak = np.abs(audio).max()
        if peak > 1e-9:
            audio = (audio / peak * 0.85).astype(np.float32)
            print(f"  [latent-to-wav] Normalizado a 0.85 peak")
    out_path = args.output or Path(args.input).stem + "_decoded.wav"
    write_wav(out_path, audio, coder.sr)


def cmd_latent_to_png(args):
    """NPZ latente → PNG escala de grises."""
    latents, sr, hop, lat_min, lat_max = load_latents_npz(args.input)
    out_path = args.output or Path(args.input).stem + ".png"
    latents_to_png(latents, sr, hop, lat_min, lat_max, out_path,
                   flip_y=not args.no_flip_y)


def cmd_png_to_latent(args):
    """PNG escala de grises → NPZ latente."""
    coder = load_coder(args.coder, args.device) if args.coder else None
    latents, sr, hop, lat_min, lat_max = png_to_latents(args.input, coder)
    out_path = args.output or Path(args.input).stem + ".npz"
    save_latents_npz(out_path, latents, sr, hop, lat_min, lat_max)


def _save_gray_png(path, arr01: np.ndarray) -> None:
    """Guarda un array float32 [0,1] de forma [H×W] como PNG 8-bit escala de grises.
    Fila 0 = arriba de la imagen; se aplica flip vertical para que el índice 0
    (bin de frecuencia 0, o componente latente 0) quede abajo, como en audio_lab.
    """
    Image = _import_pil()
    pixels = (np.clip(arr01[::-1, :], 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(str(path))

def _load_gray_png(path) -> np.ndarray:
    """Inverso de _save_gray_png: PNG → float32 [0,1] [H×W], reponiendo el flip."""
    Image = _import_pil()
    img = Image.open(str(path)).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[::-1, :].copy()


def cmd_spec_latent_train_step1(args):
    """
    WAVs + coder.pt → vuelca a disco, SIN entrenar, los pares alineados
    (espectrograma, latente) como PNG editables + manifest.json.

    Tras ejecutar este paso, un humano puede abrir los PNG en un editor de
    imagen (GIMP, Photoshop…) y modificarlos — por ejemplo limpiar ruido,
    recortar silencios o corregir errores de alineado — antes de lanzar
    spectrogram-to-latent-train_step2, que entrena sobre lo que encuentre en
    el directorio en ese momento.
    """
    coder  = load_coder(args.coder, args.device)
    sr     = coder.sr
    window, hop_ratio = args.window, args.hop
    n_bins = window // 2 + 1
    print(f"  [step1] STFT window={window} hop={hop_ratio} ({n_bins} bins)  ↔  "
          f"latente D={coder.latent_dim} hop={coder.hop}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_meta = []
    for path in args.inputs:
        stem = Path(path).stem
        audio, _ = read_wav(path, target_sr=sr)
        mags, _, _ = stft_analyse(audio, sr, window, hop_ratio)
        S  = mags_to_norm(mags, args.db_floor)                     # [Fs × bins]
        Z  = encode_wav(coder, audio)                              # [Fz × D]
        Zn = latents_normalize(Z, coder.lat_min, coder.lat_max)    # → [0,1]
        S_al = _resample_frames(S, Zn.shape[0])                    # alinear tiempo

        spec_png   = out_dir / f"{stem}_spectrogram.png"
        latent_png = out_dir / f"{stem}_latent.png"
        _save_gray_png(spec_png,   S_al.T)     # [bins × frames]
        _save_gray_png(latent_png, Zn.T)       # [D × frames]

        files_meta.append({"stem": stem, "n_frames": int(Zn.shape[0])})
        print(f"  [step1] {stem}: {S.shape[0]} frames STFT → {Zn.shape[0]} frames "
              f"latentes (alineados)  →  {spec_png.name}  +  {latent_png.name}")

    manifest = {
        "kind":        "latent_lab_pairs",
        "sr":          sr,
        "window":      window,
        "hop_ratio":   hop_ratio,
        "db_floor":    args.db_floor,
        "n_bins":      n_bins,
        "latent_dim":  coder.latent_dim,
        "coder_hop":   coder.hop,
        "lat_min":     [float(v) for v in coder.lat_min],
        "lat_max":     [float(v) for v in coder.lat_max],
        "files":       files_meta,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓  {len(files_meta)} pares → {out_dir}/  (manifest.json)")
    print(f"     Puedes editar los PNG en {out_dir}/ y luego ejecutar:")
    print(f"     python3 latent_lab.py spectrogram-to-latent-train_step2 {out_dir} -o mapper.pt")


def _load_pairs(pairs_dir: Path):
    """Lee manifest.json + los PNG (posiblemente editados a mano) del directorio.
    Devuelve (manifest, lista de (spec_img [bins×W], latent_img [D×W])).
    Las dimensiones W de cada par se toman de los PNG reales en disco, no del
    manifest, precisamente para permitir que la edición humana cambie el
    contenido libremente. Si la altura no coincide con bins/D se reescala con
    aviso (p.ej. si el editor recortó/redimensionó el lienzo).
    """
    manifest_path = pairs_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"✗  No se encuentra {manifest_path} — ¿ejecutaste antes step1?")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("kind") != "latent_lab_pairs":
        sys.exit(f"✗  {manifest_path} no es un manifest de spectrogram-to-latent-train_step1")

    n_bins, D = manifest["n_bins"], manifest["latent_dim"]
    pairs = []
    for entry in manifest["files"]:
        stem = entry["stem"]
        spec_png   = pairs_dir / f"{stem}_spectrogram.png"
        latent_png = pairs_dir / f"{stem}_latent.png"
        if not spec_png.exists() or not latent_png.exists():
            print(f"  ⚠  {stem}: PNG no encontrado, se omite (¿borrado a mano?)")
            continue
        S_img = _load_gray_png(spec_png)
        Z_img = _load_gray_png(latent_png)
        if S_img.shape[0] != n_bins:
            print(f"  ⚠  {stem}_spectrogram.png: altura {S_img.shape[0]} ≠ {n_bins} "
                  f"bins esperados — reescalado")
            S_img = _resize_image(S_img, (n_bins, S_img.shape[1]))
        if Z_img.shape[0] != D:
            print(f"  ⚠  {stem}_latent.png: altura {Z_img.shape[0]} ≠ {D} "
                  f"componentes esperados — reescalado")
            Z_img = _resize_image(Z_img, (D, Z_img.shape[1]))
        if S_img.shape[1] != Z_img.shape[1]:
            # el humano pudo recortar/ampliar solo uno de los dos PNG: realinear
            w = min(S_img.shape[1], Z_img.shape[1])
            print(f"  ⚠  {stem}: anchura espectrograma/latente distinta "
                  f"({S_img.shape[1]} vs {Z_img.shape[1]}) — recortado a {w} frames")
            S_img, Z_img = S_img[:, :w], Z_img[:, :w]
        pairs.append((S_img, Z_img))
        print(f"  [step2] par cargado: {stem}  ({S_img.shape[1]} frames)")
    if not pairs:
        sys.exit(f"✗  Ningún par válido en {pairs_dir}")
    return manifest, pairs


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ COMÚN DE MAPPERS ESPECTROGRAMA↔LATENTE
# ══════════════════════════════════════════════════════════════════════════════
#
# Los tres métodos (pix2pix, greedy, viterbi) son intercambiables porque todos
# se cargan con _load_mapper() y devuelven el mismo contrato (_MapperHandle,
# más abajo): un objeto con .ckpt (metadatos: sr/window/hop_ratio/db_floor/
# n_bins/latent_dim/coder_hop/lat_min/lat_max) y dos métodos
#   spec_to_latent(S_norm [Fs×bins] ∈[0,1]) -> Zn [Fz×D] ∈[0,1]
#   latent_to_spec(Zn [Fz×D] ∈[0,1])        -> S_norm [Fs×bins] ∈[0,1]
# cmd_spectrogram_to_latent / cmd_latent_to_spectrogram llaman solo a esta
# interfaz y no necesitan saber qué método hay detrás.
#
#   · pix2pix    mapper.pt con redes neuronales (gen_s2l/gen_l2s state_dict).
#                Un único generador cubre todo el espacio de entrada.
#   · retrieval  mapper.pt con un banco de pares frame a frame (S_bank/Z_bank),
#                sin red neuronal: el corpus ES el modelo. greedy y viterbi
#                comparten el MISMO banco/checkpoint (kind=
#                "latent_lab_mapper_retrieval") — solo difieren en el
#                algoritmo de búsqueda, elegido en inferencia con --method.
#
# spectrogram-to-latent-train_step2 --method {pix2pix,retrieval} construye el
# checkpoint; spectrogram-to-latent/latent-to-spectrogram --method
# {auto,pix2pix,greedy,viterbi} eligen cómo usarlo.

def _pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Distancia euclídea al cuadrado entre cada fila de A [n×d] y cada fila de
    B [m×d] → [n×m], vía el truco |a|²+|b|²-2·a·b (una multiplicación de
    matrices con BLAS en vez de un bucle n×m de restas).
    """
    a2 = np.sum(A * A, axis=1, keepdims=True)            # [n,1]
    b2 = np.sum(B * B, axis=1, keepdims=True).T           # [1,m]
    cross = A @ B.T                                        # [n,m]
    return np.maximum(a2 + b2 - 2.0 * cross, 0.0)


def _retrieve_greedy(query: np.ndarray, bank_query: np.ndarray,
                     bank_target: np.ndarray) -> np.ndarray:
    """
    Recuperación por vecino más cercano, frame a frame e independiente
    (sin coste de unión entre frames consecutivos — es Viterbi con K=1).
    query [F×d] se compara contra bank_query [N×d] (target cost); se
    devuelve la fila correspondiente de bank_target [N×d'] para cada frame.
    """
    d2  = _pairwise_sqdist(query, bank_query)              # [F × N]
    idx = np.argmin(d2, axis=1)
    print(f"  [retrieval:greedy] {query.shape[0]} frames buscados "
          f"sobre banco de {bank_query.shape[0]}")
    return bank_target[idx]


def _retrieve_viterbi(query: np.ndarray, bank_query: np.ndarray,
                      bank_target: np.ndarray, topk: int,
                      join_weight: float) -> np.ndarray:
    """
    Recuperación por programación dinámica (Viterbi) sobre los `topk`
    candidatos más parecidos por frame: minimiza a la vez
      Σ target_cost(frame_f, candidato_f)              — parecido al query
      Σ join_cost(candidato_{f-1}, candidato_f) · join_weight  — continuidad
    en vez de elegir cada frame de forma independiente (como greedy), lo que
    evita saltos bruscos entre unidades consecutivas del resultado. El coste
    de unión se mide en el espacio de SALIDA (bank_target), porque lo que
    importa es que la secuencia final quede continua, no el query.
    """
    F = query.shape[0]
    N = bank_query.shape[0]
    K = max(1, min(topk, N))

    d2 = _pairwise_sqdist(query, bank_query)                # [F × N] target cost
    if K < N:
        part = np.argpartition(d2, K - 1, axis=1)[:, :K]    # [F × K] candidatos
    else:
        part = np.tile(np.arange(N), (F, 1))
    rows  = np.arange(F)[:, None]
    order = np.argsort(d2[rows, part], axis=1)
    cand  = part[rows, order]                                # [F × K] ordenados por coste
    tgt_cost = d2[rows, cand]                                # [F × K]

    dp   = np.zeros((F, K), dtype=np.float64)
    back = np.zeros((F, K), dtype=np.int32)
    dp[0] = tgt_cost[0]

    for f in range(1, F):
        prev_target = bank_target[cand[f - 1]]               # [K × d']
        cur_target  = bank_target[cand[f]]                   # [K × d']
        join  = _pairwise_sqdist(cur_target, prev_target)     # [K(actual) × K(previo)]
        total = dp[f - 1][None, :] + join_weight * join       # [K × K]
        back[f] = np.argmin(total, axis=1)
        dp[f]   = tgt_cost[f] + total[np.arange(K), back[f]]

    path = np.zeros(F, dtype=np.int32)
    path[F - 1] = np.argmin(dp[F - 1])
    for f in range(F - 1, 0, -1):
        path[f - 1] = back[f][path[f]]

    chosen = cand[np.arange(F), path]
    print(f"  [retrieval:viterbi] {F} frames  topk={K}  banco={N}  "
          f"join_weight={join_weight}")
    return bank_target[chosen]


def _train_step2_retrieval(args, manifest, pairs, n_bins, D):
    """
    Construye el banco de recuperación (--method retrieval): concatena todos
    los pares frame a frame en S_bank [N×bins] / Z_bank [N×D], sin canvas_h
    (no hace falta: no hay CNN, cada búsqueda usa la resolución nativa). No
    hay entrenamiento — es indexado puro. track_id/frame_idx quedan
    guardados por si en el futuro se quiere premiar continuar dentro de la
    misma pista real en vez de saltar entre pistas distintas.
    """
    S_list, Z_list, track_id, frame_idx = [], [], [], []
    for ti, (S_img, Z_img) in enumerate(pairs):
        W = S_img.shape[1]
        S_list.append(S_img.T.astype(np.float32))      # [W × bins]
        Z_list.append(Z_img.T.astype(np.float32))      # [W × D]
        track_id.append(np.full(W, ti, dtype=np.int32))
        frame_idx.append(np.arange(W, dtype=np.int32))

    S_bank    = np.concatenate(S_list, axis=0)
    Z_bank    = np.concatenate(Z_list, axis=0)
    track_id  = np.concatenate(track_id)
    frame_idx = np.concatenate(frame_idx)

    zoo   = _torch_zoo()
    torch = zoo["torch"]
    out_path = args.output or "mapper.pt"
    torch.save({
        "kind":       "latent_lab_mapper_retrieval",
        "sr":         manifest["sr"],
        "window":     manifest["window"],
        "hop_ratio":  manifest["hop_ratio"],
        "db_floor":   manifest["db_floor"],
        "n_bins":     n_bins,
        "latent_dim": D,
        "coder_hop":  manifest["coder_hop"],
        "lat_min":    manifest["lat_min"],
        "lat_max":    manifest["lat_max"],
        "S_bank":     S_bank,
        "Z_bank":     Z_bank,
        "track_id":   track_id,
        "frame_idx":  frame_idx,
    }, out_path)
    print(f"  [step2] retrieval: banco de {S_bank.shape[0]} frames  "
          f"({len(pairs)} pistas)  bins={n_bins}  D={D}")
    print(f"  ✓  Mapper (retrieval, greedy+viterbi comparten este banco) → {out_path}")


def _sample_causal_tiles(pairs, n: int, tile: int, n_bins: int, D: int, rng):
    """
    Recorta `n` parches temporales [n_bins×tile]/[D×tile] de los pares (en
    resolución NATIVA, sin canvas — el modelo causal trata bins/componentes
    como canales, igual que FrameMapper, no como una imagen 2-D).
    """
    S_b = np.zeros((n, n_bins, tile), dtype=np.float32)
    Z_b = np.zeros((n, D, tile), dtype=np.float32)
    for b in range(n):
        S_img, Z_img = pairs[rng.integers(len(pairs))]     # [bins×W] / [D×W]
        w = S_img.shape[1]
        if w <= tile:
            S_b[b, :, :w] = S_img
            Z_b[b, :, :w] = Z_img
        else:
            start = rng.integers(w - tile + 1)
            S_b[b] = S_img[:, start:start + tile]
            Z_b[b] = Z_img[:, start:start + tile]
    return S_b, Z_b

def _causal_generate(gen, torch, S_cond_full, D: int, device,
                     log_every: int = 200) -> np.ndarray:
    """
    Genera Zn [D×T] frame a frame, autorregresivamente: en inferencia no
    existe el latente real, así que cada paso se alimenta de sus propias
    predicciones anteriores (a diferencia del entrenamiento con teacher
    forcing). spec_feat se precomputa UNA vez para todo el clip (no es
    causal). Cada paso solo reevalúa la red sobre la ventana
    [t-receptive_field, t] — fuera de ahí el campo receptivo del modelo no
    puede ver nada — acotando el coste a O(T·receptive_field) en vez de
    O(T²) con toda la historia completa.
    """
    T  = S_cond_full.shape[-1]
    rf = gen.receptive_field
    with torch.no_grad():
        spec_feat_full = gen.encode_spec(S_cond_full)         # [1,hidden,T], una vez
        Z_gen = torch.zeros(1, D, T, device=device)
        for t in range(T):
            start = max(0, t - rf)
            L = t - start + 1
            Z_hist_win = torch.zeros(1, D, L, device=device)
            if start > 0:
                Z_hist_win[:, :, 0] = Z_gen[:, :, start - 1]
            if L > 1:
                Z_hist_win[:, :, 1:] = Z_gen[:, :, start:t]
            spec_win = spec_feat_full[:, :, start:t + 1]
            pred = gen(Z_hist_win, spec_win)
            Z_gen[:, :, t] = pred[:, :, -1]
            if log_every and (t % log_every == 0 or t == T - 1):
                print(f"  [causal] generando frame {t + 1}/{T}", end="\r")
    if log_every:
        print()
    return Z_gen[0].cpu().numpy()                              # [D × T]

def _train_step2_causal(args, manifest, pairs, n_bins: int, D: int):
    """
    --method causal (versión simple): entrena CausalGenerator con L1 puro y
    teacher forcing FIJO (siempre se alimenta el latente REAL desplazado una
    posición, nunca la propia predicción del modelo) — sin scheduled
    sampling ni pérdida adversarial. Es la versión más directa: rápida de
    entrenar (una sola pasada hacia delante por paso, totalmente paralela
    en el tiempo gracias a las convoluciones causales), pero expuesta al
    "exposure bias" clásico de los modelos autorregresivos (en inferencia
    el error de cada frame se compone sobre el siguiente, porque ya no hay
    latente real con el que corregir el rumbo) y a la misma tendencia al
    "promedio de texturas ambiguas" que pix2pix, al ser también regresión
    L1 pura. Ver --method causal-adv para las mitigaciones de ambos.
    """
    zoo = _torch_zoo(); torch, nn = zoo["torch"], zoo["nn"]
    device = _get_device(args.device)
    gen = zoo["CausalGenerator"](n_bins, D, hidden=args.hidden,
                                 context=args.context).to(device)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"  [step2] causal (L1 + teacher forcing fijo)  {n_params/1e6:.2f}M "
          f"parámetros  campo_receptivo≈{gen.receptive_field} frames "
          f"({gen.n_layers} capas dilatadas)  tile={args.tile}  "
          f"batch={args.batch}  epochs={args.epochs}")

    opt = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.9, 0.999))
    rng = np.random.default_rng(args.seed)
    total_frames = sum(S.shape[1] for S, _ in pairs)
    steps_per_epoch = max(1, total_frames // (args.batch * args.tile) + 1)
    total_steps = args.epochs * steps_per_epoch
    print(f"  [step2] ~{steps_per_epoch} steps/epoch → {total_steps} steps totales")

    gen.train()
    for step in range(1, total_steps + 1):
        S_np, Z_np = _sample_causal_tiles(pairs, args.batch, args.tile, n_bins, D, rng)
        S = torch.from_numpy(S_np).to(device)
        Z = torch.from_numpy(Z_np).to(device)
        Z_shifted = torch.zeros_like(Z)
        Z_shifted[:, :, 1:] = Z[:, :, :-1]

        spec_feat = gen.encode_spec(S)
        pred = gen(Z_shifted, spec_feat)
        loss = (pred - Z).abs().mean()

        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1 or step % args.log_every == 0:
            print(f"  [step2] step {step:>6}/{total_steps}  L1={loss.item():.4f}")

    out_path = args.output or "mapper.pt"
    torch.save({
        "kind":        "latent_lab_mapper_causal",
        "sr":          manifest["sr"],
        "window":      manifest["window"],
        "hop_ratio":   manifest["hop_ratio"],
        "db_floor":    manifest["db_floor"],
        "n_bins":      n_bins,
        "latent_dim":  D,
        "coder_hop":   manifest["coder_hop"],
        "lat_min":     manifest["lat_min"],
        "lat_max":     manifest["lat_max"],
        "hidden":      args.hidden,
        "context":     args.context,
        "adversarial": False,
        "gen":         gen.state_dict(),
    }, out_path)
    print(f"  ✓  Mapper (causal, L1 + teacher forcing fijo) → {out_path}")

def _train_step2_causal_adv(args, manifest, pairs, n_bins: int, D: int):
    """
    --method causal-adv: como causal, pero con dos mitigaciones explícitas
    de los dos problemas que discutimos:

    1) SCHEDULED SAMPLING (Bengio et al. 2015), contra el exposure bias:
       en vez de alimentar SIEMPRE el latente real desplazado, con
       probabilidad ss_prob (creciente a lo largo del entrenamiento, hasta
       --ss-final-prob) se sustituye cada frame de la historia por la
       propia predicción del modelo (calculada en una primera pasada,
       desconectada del grafo). AVISO — simplificación deliberada: esto NO
       es un rollout secuencial completo (frame a frame, como en un RNN),
       que sería el scheduled sampling "de libro" pero perdería el
       paralelismo temporal de las convoluciones causales (pasaría de
       O(1) a O(T) pasadas por paso de entrenamiento). Aquí se hace en DOS
       pasadas paralelas: (1) predicción con historia real, (2) mezcla
       historia real / propia predicción frame a frame, y el gradiente
       solo fluye por la pasada (2). Expone al modelo a su propio error
       de un paso sin pagar el coste de un rollout completo.

    2) PÉRDIDA ADVERSARIAL (CausalDiscriminator1D, PatchGAN 1-D
       condicionado al espectrograma), contra el "promedio de texturas
       ambiguas": igual que en pix2pix, el discriminador penaliza que la
       secuencia generada no tenga la estadística local de una secuencia
       latente real, empujando hacia UNA textura plausible concreta en
       vez del compromiso promedio entre todas — ver conversación previa
       sobre pix2pix y mean-seeking bajo L1 puro.
    """
    zoo = _torch_zoo(); torch, nn = zoo["torch"], zoo["nn"]
    device = _get_device(args.device)
    gen  = zoo["CausalGenerator"](n_bins, D, hidden=args.hidden,
                                  context=args.context).to(device)
    disc = zoo["CausalDiscriminator1D"](n_bins, D, base=args.disc_base).to(device)
    n_params = (sum(p.numel() for p in gen.parameters())
               + sum(p.numel() for p in disc.parameters()))
    print(f"  [step2] causal-adv (scheduled sampling + adversarial)  "
          f"{n_params/1e6:.2f}M parámetros  campo_receptivo≈{gen.receptive_field} "
          f"frames  tile={args.tile}  batch={args.batch}  epochs={args.epochs}  "
          f"ss_final_prob={args.ss_final_prob}  ss_warmup={args.ss_warmup}")

    opt_g = torch.optim.Adam(gen.parameters(),  lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    total_frames = sum(S.shape[1] for S, _ in pairs)
    steps_per_epoch = max(1, total_frames // (args.batch * args.tile) + 1)
    total_steps = args.epochs * steps_per_epoch
    print(f"  [step2] ~{steps_per_epoch} steps/epoch → {total_steps} steps totales")

    gen.train(); disc.train()
    for step in range(1, total_steps + 1):
        ss_prob = args.ss_final_prob * min(
            1.0, step / max(1, total_steps * args.ss_warmup))

        S_np, Z_np = _sample_causal_tiles(pairs, args.batch, args.tile, n_bins, D, rng)
        S = torch.from_numpy(S_np).to(device)
        Z = torch.from_numpy(Z_np).to(device)
        Z_shifted_real = torch.zeros_like(Z)
        Z_shifted_real[:, :, 1:] = Z[:, :, :-1]

        spec_feat = gen.encode_spec(S)

        if ss_prob > 0:
            with torch.no_grad():
                pred_pass1 = gen(Z_shifted_real, spec_feat)
                pred_shifted = torch.zeros_like(Z)
                pred_shifted[:, :, 1:] = pred_pass1[:, :, :-1]
            mask = (torch.rand(Z.shape[0], 1, Z.shape[2], device=device) < ss_prob).float()
            Z_hist = mask * pred_shifted + (1 - mask) * Z_shifted_real
        else:
            Z_hist = Z_shifted_real

        # ── discriminador ──
        with torch.no_grad():
            fake = gen(Z_hist, spec_feat)
        pred_real = disc(S, Z)
        pred_fake = disc(S, fake)
        loss_d = 0.5 * (bce(pred_real, torch.ones_like(pred_real))
                        + bce(pred_fake, torch.zeros_like(pred_fake)))
        opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        # ── generador ──
        fake = gen(Z_hist, spec_feat)
        pred_fake = disc(S, fake)
        loss_adv = bce(pred_fake, torch.ones_like(pred_fake))
        loss_l1  = (fake - Z).abs().mean()
        loss_g   = loss_adv + args.l1_weight * loss_l1
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if step == 1 or step % args.log_every == 0:
            print(f"  [step2] step {step:>6}/{total_steps}  ss_prob={ss_prob:.2f}  "
                  f"D={loss_d.item():.3f}  G_adv={loss_adv.item():.3f}  "
                  f"L1={loss_l1.item():.4f}")

    out_path = args.output or "mapper.pt"
    torch.save({
        "kind":        "latent_lab_mapper_causal",
        "sr":          manifest["sr"],
        "window":      manifest["window"],
        "hop_ratio":   manifest["hop_ratio"],
        "db_floor":    manifest["db_floor"],
        "n_bins":      n_bins,
        "latent_dim":  D,
        "coder_hop":   manifest["coder_hop"],
        "lat_min":     manifest["lat_min"],
        "lat_max":     manifest["lat_max"],
        "hidden":      args.hidden,
        "context":     args.context,
        "disc_base":   args.disc_base,
        "adversarial": True,
        "gen":         gen.state_dict(),
        "disc":        disc.state_dict(),
    }, out_path)
    print(f"  ✓  Mapper (causal-adv: scheduled sampling + adversarial) → {out_path}")

def cmd_spec_latent_train_step2(args):
    """
    Directorio de pares (de step1, posiblemente editados a mano) → construye
    el traductor espectrograma↔latente con el método elegido por --method:
      · pix2pix    entrena U-Net generador + PatchGAN discriminador (L1 +
                    adversarial), como en las versiones anteriores del
                    programa — mapper.pt con redes neuronales.
      · retrieval  indexa los pares frame a frame en un banco de búsqueda,
                    sin entrenar nada — mapper.pt con el corpus. greedy y
                    viterbi (elegidos luego en spectrogram-to-latent
                    --method) reutilizan el MISMO banco/checkpoint.
      · causal     CausalGenerator autorregresivo (WaveNet-lite), L1 puro
                    + teacher forcing fijo. Solo entrena la dirección
                    espectrograma→latente (ver latent-to-spectrogram).
      · causal-adv como causal, + scheduled sampling (contra exposure
                    bias) + pérdida adversarial (contra el promedio de
                    texturas ambiguas de la regresión L1 pura).
    """
    pairs_dir = Path(args.pairs_dir)
    manifest, pairs = _load_pairs(pairs_dir)
    n_bins, D = manifest["n_bins"], manifest["latent_dim"]

    if args.method == "retrieval":
        _train_step2_retrieval(args, manifest, pairs, n_bins, D)
        return
    if args.method == "causal":
        _train_step2_causal(args, manifest, pairs, n_bins, D)
        return
    if args.method == "causal-adv":
        _train_step2_causal_adv(args, manifest, pairs, n_bins, D)
        return

    zoo   = _torch_zoo()
    torch, nn = zoo["torch"], zoo["nn"]
    device = _get_device(args.device)

    gen_s2l  = zoo["UNetGenerator"](args.gen_base).to(device)
    disc_s2l = zoo["PatchDiscriminator"](args.disc_base).to(device)
    gen_l2s  = zoo["UNetGenerator"](args.gen_base).to(device)
    disc_l2s = zoo["PatchDiscriminator"](args.disc_base).to(device)
    n_params = sum(p.numel() for m in (gen_s2l, disc_s2l, gen_l2s, disc_l2s)
                   for p in m.parameters())
    print(f"  [step2] pix2pix  {n_params/1e6:.2f}M parámetros totales  "
          f"tile={args.tile}  batch={args.batch}  epochs={args.epochs}")

    opt_g = torch.optim.Adam(list(gen_s2l.parameters()) + list(gen_l2s.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(list(disc_s2l.parameters()) + list(disc_l2s.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    tile = args.tile

    canvas_h = PIX2PIX_CANVAS_H
    # ── pre-escalar TODOS los pares a la misma altura de lienzo (canvas_h) ──
    # para que espectrograma (n_bins filas) y latente (D filas) puedan
    # concatenarse en el discriminador y compartir arquitectura de generador.
    pairs_canvas = [(_resize_image(S_img, (canvas_h, S_img.shape[1])),
                     _resize_image(Z_img, (canvas_h, Z_img.shape[1])))
                    for S_img, Z_img in pairs]

    def _sample_tiles(n):
        """Recorta n parches temporales aleatorios [canvas_h × tile] de los pares
        (ya reescalados a canvas_h filas)."""
        S_b = np.zeros((n, 1, canvas_h, tile), dtype=np.float32)
        Z_b = np.zeros((n, 1, canvas_h, tile), dtype=np.float32)
        for b in range(n):
            S_img, Z_img = pairs_canvas[rng.integers(len(pairs_canvas))]
            w = S_img.shape[1]
            if w <= tile:
                S_b[b, 0, :, :w] = S_img
                Z_b[b, 0, :, :w] = Z_img
            else:
                start = rng.integers(w - tile)
                S_b[b, 0] = S_img[:, start:start + tile]
                Z_b[b, 0] = Z_img[:, start:start + tile]
        return (torch.from_numpy(S_b).to(device), torch.from_numpy(Z_b).to(device))

    steps_per_epoch = max(1, sum(p[0].shape[1] for p in pairs) // (args.batch * tile) + 1)
    total_steps = args.epochs * steps_per_epoch
    print(f"  [step2] ~{steps_per_epoch} steps/epoch  →  {total_steps} steps totales")

    def _gan_step(gen, disc, cond, target, opt_g_step, opt_d_step):
        """Un paso pix2pix completo (D real/falso + G adversarial+L1) para una
        dirección (cond→target). Devuelve (loss_d, loss_g, loss_l1)."""
        # ── discriminador ──
        with torch.no_grad():
            fake = gen(cond)
        pred_real = disc(cond, target)
        pred_fake = disc(cond, fake)
        loss_d = 0.5 * (bce(pred_real, torch.ones_like(pred_real)) +
                        bce(pred_fake, torch.zeros_like(pred_fake)))
        opt_d_step.zero_grad(); loss_d.backward(); opt_d_step.step()

        # ── generador ──
        fake = gen(cond)
        pred_fake = disc(cond, fake)
        loss_g_adv = bce(pred_fake, torch.ones_like(pred_fake))
        loss_l1    = (fake - target).abs().mean()
        loss_g     = loss_g_adv + args.l1_weight * loss_l1
        opt_g_step.zero_grad(); loss_g.backward(); opt_g_step.step()
        return loss_d.item(), loss_g_adv.item(), loss_l1.item()

    gen_s2l.train(); disc_s2l.train(); gen_l2s.train(); disc_l2s.train()
    for step in range(1, total_steps + 1):
        S, Z = _sample_tiles(args.batch)
        d1, g1, l1_1 = _gan_step(gen_s2l, disc_s2l, S, Z, opt_g, opt_d)
        d2, g2, l1_2 = _gan_step(gen_l2s, disc_l2s, Z, S, opt_g, opt_d)
        if step == 1 or step % args.log_every == 0:
            print(f"  [step2] step {step:>6}/{total_steps}  "
                  f"spec→lat[D={d1:.3f} G={g1:.3f} L1={l1_1:.4f}]  "
                  f"lat→spec[D={d2:.3f} G={g2:.3f} L1={l1_2:.4f}]")

    out_path = args.output or "mapper.pt"
    torch.save({
        "kind":        "latent_lab_mapper_pix2pix",
        "sr":          manifest["sr"],
        "window":      manifest["window"],
        "hop_ratio":   manifest["hop_ratio"],
        "db_floor":    manifest["db_floor"],
        "n_bins":      n_bins,
        "latent_dim":  D,
        "coder_hop":   manifest["coder_hop"],
        "gen_base":    args.gen_base,
        "disc_base":   args.disc_base,
        "tile":        tile,
        "canvas_h":    canvas_h,
        "lat_min":     manifest["lat_min"],
        "lat_max":     manifest["lat_max"],
        "gen_s2l":     gen_s2l.state_dict(),
        "gen_l2s":     gen_l2s.state_dict(),
        "disc_s2l":    disc_s2l.state_dict(),
        "disc_l2s":    disc_l2s.state_dict(),
    }, out_path)
    print(f"  ✓  Mapper (pix2pix) → {out_path}")


def _run_generator_tiled(gen, torch, img: np.ndarray, tile: int, device) -> np.ndarray:
    """
    Aplica un UNetGenerator sobre una imagen [H×W] de anchura arbitraria,
    troceando en parches de anchura `tile` (el tamaño con el que se entrenó)
    con solape del 25% y mezcla lineal en las zonas de solape, para evitar
    discontinuidades entre parches consecutivos.
    """
    H, W = img.shape
    if W <= tile:
        x = torch.from_numpy(np.ascontiguousarray(img)[None, None]).to(device)
        with torch.no_grad():
            y = gen(x)[0, 0].cpu().numpy()
        return y
    hop = max(1, tile * 3 // 4)
    out  = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((1, W), dtype=np.float32)
    win  = np.hanning(tile).astype(np.float32)[None, :]
    win  = np.clip(win, 0.1, 1.0)   # evita ceros exactos en los bordes
    pos = 0
    with torch.no_grad():
        while True:
            start = min(pos, W - tile)
            chunk = np.ascontiguousarray(img[:, start:start + tile])
            x = torch.from_numpy(chunk[None, None]).to(device)
            y = gen(x)[0, 0].cpu().numpy()
            out[:, start:start + tile]  += y * win
            wsum[:, start:start + tile] += win
            if start + tile >= W:
                break
            pos += hop
    wsum[wsum < 1e-6] = 1.0
    return out / wsum


def _load_mapper(path: str, device_name: str, method: str = "auto",
                 topk: int = 8, join_weight: float = 1.0):
    """
    Carga un checkpoint de mapper y devuelve un objeto con dos métodos
    uniformes — spec_to_latent(S_norm)->Zn y latent_to_spec(Zn)->S_norm —
    que funcionan igual sea el mapper legado (FrameMapper 1-D), el pix2pix
    (UNetGenerator 2-D) o el de retrieval (greedy/viterbi sobre un banco de
    pares), para que cmd_spectrogram_to_latent y cmd_latent_to_spectrogram
    no necesiten saber cuál es cuál. `method` solo se usa (y es obligatorio
    resolverlo) cuando el checkpoint es de retrieval — "auto" implica
    "viterbi" en ese caso; para pix2pix/legado se ignora salvo que se pida
    explícitamente un método incompatible, que es un error de uso.
    """
    zoo  = _torch_zoo()
    torch = zoo["torch"]
    ckpt = _torch_load(path)
    kind = ckpt.get("kind")
    device = _get_device(device_name)

    class _MapperHandle:
        def __init__(self, ckpt, spec_fn, lat_fn):
            self.ckpt = ckpt
            self._spec_fn = spec_fn
            self._lat_fn  = lat_fn
        def spec_to_latent(self, S_norm: np.ndarray) -> np.ndarray:
            return self._spec_fn(S_norm)
        def latent_to_spec(self, Z_norm: np.ndarray) -> np.ndarray:
            return self._lat_fn(Z_norm)

    if kind == "latent_lab_mapper":
        if method not in (None, "auto", "pix2pix"):
            sys.exit(f"✗  --method {method} no es válido para este checkpoint "
                     f"(FrameMapper legado) — usa pix2pix o auto")
        spec2lat = zoo["FrameMapper"](ckpt["n_bins"], ckpt["latent_dim"],
                                      ckpt["hidden"], ckpt["kernel"])
        lat2spec = zoo["FrameMapper"](ckpt["latent_dim"], ckpt["n_bins"],
                                      ckpt["hidden"], ckpt["kernel"])
        spec2lat.load_state_dict(ckpt["spec2lat"]);  spec2lat.to(device).eval()
        lat2spec.load_state_dict(ckpt["lat2spec"]);  lat2spec.to(device).eval()
        print(f"  [mapper] {path}: (FrameMapper) window={ckpt['window']} "
              f"hop={ckpt['hop_ratio']} ↔ D={ckpt['latent_dim']} coder_hop={ckpt['coder_hop']}")

        def spec_fn(S_norm):   # S_norm [Fs × bins] → Zn [Fz × D]
            spec_hop = max(1, int(ckpt["window"] * ckpt["hop_ratio"]))
            n_lat = max(1, round(S_norm.shape[0] * spec_hop / ckpt["coder_hop"]))
            S_al = _resample_frames(S_norm, n_lat)
            with torch.no_grad():
                x = torch.from_numpy(S_al.T[None]).to(device)
                Zn = spec2lat(x)[0].cpu().numpy().T
            return np.clip(Zn, 0.0, 1.0)

        def lat_fn(Z_norm):    # Z_norm [Fz × D] → S_norm [Fs × bins]
            spec_hop = max(1, int(ckpt["window"] * ckpt["hop_ratio"]))
            n_spec = max(1, round(Z_norm.shape[0] * ckpt["coder_hop"] / spec_hop))
            with torch.no_grad():
                z = torch.from_numpy(Z_norm.T[None]).to(device)
                S = lat2spec(z)[0].cpu().numpy().T
            S = np.clip(S, 0.0, 1.0)
            return _resample_frames(S, n_spec)

        return _MapperHandle(ckpt, spec_fn, lat_fn)

    elif kind == "latent_lab_mapper_pix2pix":
        if method not in (None, "auto", "pix2pix"):
            sys.exit(f"✗  --method {method} no es válido para este checkpoint "
                     f"(pix2pix) — usa pix2pix o auto")
        gen_s2l = zoo["UNetGenerator"](ckpt["gen_base"])
        gen_l2s = zoo["UNetGenerator"](ckpt["gen_base"])
        gen_s2l.load_state_dict(ckpt["gen_s2l"]);  gen_s2l.to(device).eval()
        gen_l2s.load_state_dict(ckpt["gen_l2s"]);  gen_l2s.to(device).eval()
        tile     = ckpt["tile"]
        canvas_h = ckpt.get("canvas_h", PIX2PIX_CANVAS_H)
        n_bins_m = ckpt["n_bins"]
        D_m      = ckpt["latent_dim"]
        print(f"  [mapper] {path}: (pix2pix U-Net) window={ckpt['window']} "
              f"hop={ckpt['hop_ratio']} ↔ D={ckpt['latent_dim']} coder_hop={ckpt['coder_hop']} "
              f"tile={tile}  canvas_h={canvas_h}")

        def spec_fn(S_norm):   # S_norm [Fs × bins] → Zn [Fz × D]
            spec_hop = max(1, int(ckpt["window"] * ckpt["hop_ratio"]))
            n_lat = max(1, round(S_norm.shape[0] * spec_hop / ckpt["coder_hop"]))
            S_al  = _resample_frames(S_norm, n_lat).T                    # [bins × F]
            S_can = _resize_image(S_al, (canvas_h, S_al.shape[1]))       # → lienzo
            Zn_can = _run_generator_tiled(gen_s2l, torch, S_can, tile, device)
            Zn = _resize_image(Zn_can, (D_m, Zn_can.shape[1]))           # → D real
            return np.clip(Zn.T, 0.0, 1.0)                               # [F × D]

        def lat_fn(Z_norm):    # Z_norm [Fz × D] → S_norm [Fs × bins]
            spec_hop = max(1, int(ckpt["window"] * ckpt["hop_ratio"]))
            n_spec = max(1, round(Z_norm.shape[0] * ckpt["coder_hop"] / spec_hop))
            Z_img = Z_norm.T                                             # [D × F]
            Z_can = _resize_image(Z_img, (canvas_h, Z_img.shape[1]))     # → lienzo
            S_can = _run_generator_tiled(gen_l2s, torch, Z_can, tile, device)
            S_img = _resize_image(S_can, (n_bins_m, S_can.shape[1]))     # → bins reales
            S = np.clip(S_img.T, 0.0, 1.0)                               # [F × bins]
            return _resample_frames(S, n_spec)

        return _MapperHandle(ckpt, spec_fn, lat_fn)

    elif kind == "latent_lab_mapper_retrieval":
        resolved = method if method not in (None, "auto") else "viterbi"
        if resolved not in ("greedy", "viterbi"):
            sys.exit(f"✗  --method {resolved} no es válido para este checkpoint "
                     f"(retrieval) — usa greedy, viterbi o auto")
        S_bank = ckpt["S_bank"]      # [N × bins] normalizado [0,1]
        Z_bank = ckpt["Z_bank"]      # [N × D]    normalizado [0,1]
        print(f"  [mapper] {path}: (retrieval, método={resolved}) "
              f"banco={S_bank.shape[0]} frames  window={ckpt['window']} "
              f"hop={ckpt['hop_ratio']} ↔ D={ckpt['latent_dim']} coder_hop={ckpt['coder_hop']}"
              + (f"  topk={topk}  join_weight={join_weight}" if resolved == "viterbi" else ""))

        def spec_fn(S_norm):   # S_norm [Fs × bins] → Zn [Fz × D]
            spec_hop = max(1, int(ckpt["window"] * ckpt["hop_ratio"]))
            n_lat = max(1, round(S_norm.shape[0] * spec_hop / ckpt["coder_hop"]))
            S_al = _resample_frames(S_norm, n_lat)                       # [Fz × bins]
            if resolved == "greedy":
                Zn = _retrieve_greedy(S_al, S_bank, Z_bank)
            else:
                Zn = _retrieve_viterbi(S_al, S_bank, Z_bank, topk, join_weight)
            return np.clip(Zn, 0.0, 1.0)

        def lat_fn(Z_norm):    # Z_norm [Fz × D] → S_norm [Fs × bins]
            spec_hop = max(1, int(ckpt["window"] * ckpt["hop_ratio"]))
            n_spec = max(1, round(Z_norm.shape[0] * ckpt["coder_hop"] / spec_hop))
            if resolved == "greedy":
                S = _retrieve_greedy(Z_norm, Z_bank, S_bank)
            else:
                S = _retrieve_viterbi(Z_norm, Z_bank, S_bank, topk, join_weight)
            S = np.clip(S, 0.0, 1.0)
            return _resample_frames(S, n_spec)

        return _MapperHandle(ckpt, spec_fn, lat_fn)

    elif kind == "latent_lab_mapper_causal":
        if method not in (None, "auto", "causal", "causal-adv"):
            sys.exit(f"✗  --method {method} no es válido para este checkpoint "
                     f"(causal) — usa causal, causal-adv o auto")
        gen = zoo["CausalGenerator"](ckpt["n_bins"], ckpt["latent_dim"],
                                     hidden=ckpt["hidden"], context=ckpt["context"])
        gen.load_state_dict(ckpt["gen"]);  gen.to(device).eval()
        tag = "causal-adv" if ckpt.get("adversarial") else "causal"
        print(f"  [mapper] {path}: ({tag}, autorregresivo) window={ckpt['window']} "
              f"hop={ckpt['hop_ratio']} ↔ D={ckpt['latent_dim']} coder_hop={ckpt['coder_hop']} "
              f"campo_receptivo≈{gen.receptive_field} frames")

        def spec_fn(S_norm):   # S_norm [Fs × bins] → Zn [Fz × D]
            spec_hop = max(1, int(ckpt["window"] * ckpt["hop_ratio"]))
            n_lat = max(1, round(S_norm.shape[0] * spec_hop / ckpt["coder_hop"]))
            S_al = _resample_frames(S_norm, n_lat)                    # [Fz × bins]
            S_t  = torch.from_numpy(S_al.T[None].astype(np.float32)).to(device)
            Zn = _causal_generate(gen, torch, S_t, ckpt["latent_dim"], device)
            return np.clip(Zn.T, 0.0, 1.0)                            # [Fz × D]

        def lat_fn(Z_norm):
            sys.exit("✗  latent-to-spectrogram no está soportado para mappers "
                     "causales: causal/causal-adv solo entrenan la dirección "
                     "espectrograma→latente (ver conversación de diseño) — "
                     "usa pix2pix o retrieval para latente→espectrograma")

        return _MapperHandle(ckpt, spec_fn, lat_fn)

    else:
        sys.exit(f"✗  {path} no es un checkpoint de spectrogram-to-latent-train"
                 f"(_step2 ni de la versión legada)")


def cmd_spectrogram_to_latent(args):
    """NPZ de espectrograma (audio_lab) → NPZ latente [+ PNG]."""
    mapper = _load_mapper(args.mapper, args.device, args.method,
                          args.topk, args.join_weight)
    ckpt   = mapper.ckpt

    data   = np.load(args.input)
    if "magnitudes" not in data.files:
        kind = "latente" if "latents" in data.files else "desconocido"
        sys.exit(f"✗  {args.input} no es un NPZ de espectrograma (detectado: {kind}). "
                 f"¿Querías usar audio_lab spectrogram primero?")
    mags   = data["magnitudes"].astype(np.float64)
    sr     = int(data["sample_rate"])
    window = int(data["window_size"])
    hop_r  = float(data["hop_ratio"])
    if window != ckpt["window"] or abs(hop_r - ckpt["hop_ratio"]) > 1e-9:
        sys.exit(f"✗  El NPZ es window={window} hop={hop_r} pero el mapper fue "
                 f"entrenado con window={ckpt['window']} hop={ckpt['hop_ratio']}")

    S = mags_to_norm(mags, ckpt["db_floor"])                     # [Fs × bins]
    Zn = mapper.spec_to_latent(S)                                # [Fz × D] ∈ [0,1]
    print(f"  [spectrogram-to-latent] {S.shape[0]} frames STFT → "
          f"{Zn.shape[0]} frames latentes")

    lat_min = np.array(ckpt["lat_min"], dtype=np.float32)
    lat_max = np.array(ckpt["lat_max"], dtype=np.float32)
    latents = latents_denormalize(Zn, lat_min, lat_max)

    out_path = args.output or Path(args.input).stem + "_latent.npz"
    save_latents_npz(out_path, latents, sr, ckpt["coder_hop"], lat_min, lat_max)
    if args.png:
        png_path = str(Path(out_path).with_suffix(".png"))
        latents_to_png(latents, sr, ckpt["coder_hop"], lat_min, lat_max, png_path)


def cmd_latent_to_spectrogram(args):
    """NPZ/PNG latente → NPZ de espectrograma compatible con audio_lab [+ WAV]."""
    mapper = _load_mapper(args.mapper, args.device, args.method,
                          args.topk, args.join_weight)
    ckpt   = mapper.ckpt

    lat_min = np.array(ckpt["lat_min"], dtype=np.float32)
    lat_max = np.array(ckpt["lat_max"], dtype=np.float32)
    latents, sr, hop, _, _ = _load_latents_any(args.input)
    if latents.shape[1] != ckpt["latent_dim"]:
        sys.exit(f"✗  El fichero tiene {latents.shape[1]} componentes pero el "
                 f"mapper espera {ckpt['latent_dim']}")
    Zn = latents_normalize(latents, lat_min, lat_max)

    S = mapper.latent_to_spec(Zn)                                # [Fs × bins] ∈ [0,1]
    window, hop_r = ckpt["window"], ckpt["hop_ratio"]
    spec_hop      = max(1, int(window * hop_r))
    n_spec_frames = S.shape[0]
    print(f"  [latent-to-spectrogram] {latents.shape[0]} frames latentes → "
          f"{n_spec_frames} frames STFT")

    mags   = norm_to_mags(S, ckpt["db_floor"], args.mag_max)
    phases = np.zeros_like(mags)
    freqs  = np.fft.rfftfreq(window, 1.0 / ckpt["sr"]).astype(np.float32)

    out_path = args.output or Path(args.input).stem + "_stft.npz"
    np.savez_compressed(out_path,
                        magnitudes=mags, phases=phases, freqs=freqs,
                        sample_rate=np.array(ckpt["sr"]),
                        window_size=np.array(window),
                        hop_ratio=np.array(hop_r))
    dur = n_spec_frames * spec_hop / ckpt["sr"]
    print(f"  ✓  NPZ STFT → {out_path}  ({n_spec_frames} frames × "
          f"{mags.shape[1]} bins, ~{dur:.2f}s)")
    print(f"     Fase: cero — compatible con audio_lab reconstruct (Griffin-Lim)")

    if args.wav:
        if args.vocoder:
            print(f"  [latent-to-spectrogram] vocoder neuronal ({args.vocoder})...")
            voc, vckpt = load_vocoder(args.vocoder, args.device)
            if vckpt["n_bins"] != S.shape[1]:
                sys.exit(f"✗  El vocoder espera {vckpt['n_bins']} bins pero el "
                         f"espectrograma generado tiene {S.shape[1]}")
            audio = vocoder_synthesize(voc, S)
        else:
            print(f"  [latent-to-spectrogram] Griffin-Lim {args.gl_iters} iter "
                  f"(phase_init={args.gl_phase_init})...")
            audio = griffin_lim(mags, hop_r, n_iter=args.gl_iters, sr=ckpt["sr"],
                                phase_init=args.gl_phase_init)
        peak  = np.abs(audio).max()
        if peak > 1e-9:
            audio = (audio / peak * 0.85).astype(np.float32)
        write_wav(args.wav, audio, ckpt["sr"])

# ══════════════════════════════════════════════════════════════════════════════
# NPZ / PNG DE ESPECTROGRAMAS — par gemelo de "NPZ / PNG DE LATENTES"
# ══════════════════════════════════════════════════════════════════════════════
#
# Mismo esquema de imagen editable que latents_to_png/png_to_latents (PNG en
# escala de grises, bin 0 abajo, metadatos en chunks tEXt + sidecar .ll.json),
# pero con la normalización propia de magnitudes STFT (mags_to_norm/norm_to_mags:
# pico del fichero + rango dB con db_floor) en vez de min/max por componente.
# Así spectrogram-to-latent (que solo acepta NPZ) puede alimentarse de un PNG
# editado a mano sin pasar por ningún script externo: basta con
# png-to-spectrogram antes.

def spectrogram_to_png(mags: np.ndarray, sr: int, window: int, hop_ratio: float,
                       db_floor: float, out_path: str, flip_y: bool = True) -> None:
    """
    Magnitudes STFT [frames × bins] → PNG escala de grises [bins(alto) ×
    frames(ancho)]. Bin 0 (grave) abajo, salvo flip_y=False. Metadatos en
    chunks tEXt + sidecar .ll.json (fallback anti-GIMP), igual que
    latents_to_png pero con mags_to_norm/db_floor en vez de lat_min/lat_max.
    """
    Image = _import_pil()
    from PIL import PngImagePlugin

    norm = mags_to_norm(mags, db_floor)          # [F, bins] ∈ [0,1]
    img_data = norm.T                            # [bins × F]
    if flip_y:
        img_data = img_data[::-1, :]
    pixels = (img_data * 255).clip(0, 255).astype(np.uint8)
    img    = Image.fromarray(pixels, mode="L")

    F, n_bins = mags.shape
    fields = {
        "latent_lab_kind":      "spectrogram",
        "latent_lab_sr":        str(sr),
        "latent_lab_window":    str(window),
        "latent_lab_hop_ratio": str(hop_ratio),
        "latent_lab_db_floor":  str(db_floor),
        "latent_lab_n_frames":  str(F),
        "latent_lab_n_bins":    str(n_bins),
        "latent_lab_flip_y":    "1" if flip_y else "0",
    }
    meta = PngImagePlugin.PngInfo()
    for k, v in fields.items():
        meta.add_text(k, v)

    sidecar = Path(out_path).with_suffix(".ll.json")
    with open(sidecar, "w") as f:
        json.dump(fields, f, indent=2)

    img.save(out_path, format="PNG", pnginfo=meta)
    print(f"  ✓  PNG → {out_path}  ({F}×{n_bins}px)")
    print(f"     Metadatos embebidos: sr={sr} window={window} "
          f"hop_ratio={hop_ratio} db_floor={db_floor}")
    print(f"     Sidecar JSON → {sidecar}  (fallback si el editor elimina tEXt)")

def png_to_spectrogram(path: str, sr: Optional[int] = None,
                       window: Optional[int] = None,
                       hop_ratio: Optional[float] = None,
                       db_floor: Optional[float] = None,
                       mag_max: float = 1.0):
    """
    Inverso de spectrogram_to_png: PNG escala de grises → magnitudes STFT
    [frames × bins]. Lee metadatos tEXt; si faltan, prueba el sidecar
    .ll.json; si tampoco, usa los overrides sr/window/hop_ratio/db_floor
    (obligatorios en ese caso — no hay equivalente al --coder de png-to-latent
    porque estos parámetros son de análisis STFT, no de un modelo entrenado).
    """
    Image  = _import_pil()
    img    = Image.open(path).convert("L")
    pixels = np.array(img, dtype=np.float64)            # [H × W]
    meta   = dict(img.info)

    sidecar = Path(path).with_suffix(".ll.json")
    if "latent_lab_sr" not in meta and sidecar.exists():
        with open(sidecar) as f:
            meta.update(json.load(f))
        print(f"  [png-to-spectrogram] Metadatos leídos desde sidecar: {sidecar.name}")

    def _get(key, override, cast):
        if override is not None:
            return override
        if key in meta:
            return cast(meta[key])
        sys.exit(f"✗  PNG sin metadato {key!r} (ni sidecar) — pásalo con la "
                 f"opción correspondiente (--sr/--window/--hop/--db-floor)")

    sr_v       = _get("latent_lab_sr",        sr,        lambda v: int(float(v)))
    window_v   = _get("latent_lab_window",    window,    lambda v: int(float(v)))
    hop_r_v    = _get("latent_lab_hop_ratio", hop_ratio, float)
    db_floor_v = _get("latent_lab_db_floor",  db_floor,  float)
    flip_y     = str(meta.get("latent_lab_flip_y", "1")) in ("1", "True", "true")

    H, W = pixels.shape
    n_bins_expected = window_v // 2 + 1
    if H != n_bins_expected:
        print(f"  ⚠  Altura del PNG ({H}) ≠ bins esperados para window={window_v} "
              f"({n_bins_expected}) — se usa la altura del PNG tal cual")

    if flip_y:
        pixels = pixels[::-1, :]
    norm = (pixels.T / 255.0).astype(np.float32)          # [frames × bins]
    mags = norm_to_mags(norm, db_floor_v, mag_max)
    print(f"  [png-to-spectrogram] {W}×{H}px → {W} frames × {H} bins")
    return mags, sr_v, window_v, hop_r_v, db_floor_v

def cmd_spectrogram_to_png(args):
    """NPZ de espectrograma (audio_lab) → PNG escala de grises editable."""
    data = np.load(args.input)
    if "magnitudes" not in data.files:
        kind = "latente" if "latents" in data.files else "desconocido"
        sys.exit(f"✗  {args.input} no es un NPZ de espectrograma (detectado: "
                 f"{kind}). ¿Querías usar latent-to-png?")
    mags   = data["magnitudes"].astype(np.float64)
    sr     = int(data["sample_rate"])
    window = int(data["window_size"])
    hop_r  = float(data["hop_ratio"])

    out_path = args.output or Path(args.input).stem + ".png"
    spectrogram_to_png(mags, sr, window, hop_r, args.db_floor, out_path,
                       flip_y=not args.no_flip_y)

def cmd_png_to_spectrogram(args):
    """PNG escala de grises → NPZ de espectrograma compatible con audio_lab."""
    mags, sr, window, hop_r, db_floor = png_to_spectrogram(
        args.input, sr=args.sr, window=args.window, hop_ratio=args.hop,
        db_floor=args.db_floor, mag_max=args.mag_max)
    phases = np.zeros_like(mags)
    freqs  = np.fft.rfftfreq(window, 1.0 / sr).astype(np.float32)

    out_path = args.output or Path(args.input).stem + "_stft.npz"
    np.savez_compressed(out_path,
                        magnitudes=mags, phases=phases, freqs=freqs,
                        sample_rate=np.array(sr),
                        window_size=np.array(window),
                        hop_ratio=np.array(hop_r))
    spec_hop = max(1, int(window * hop_r))
    dur = mags.shape[0] * spec_hop / sr
    print(f"  ✓  NPZ STFT → {out_path}  ({mags.shape[0]} frames × "
          f"{mags.shape[1]} bins, ~{dur:.2f}s)")
    print(f"     Fase: cero — compatible con audio_lab reconstruct (Griffin-Lim) "
          f"y con spectrogram-to-latent")

# ══════════════════════════════════════════════════════════════════════════════
# PCA SOBRE REPRESENTACIONES INTERMEDIAS (latentes o espectrogramas)
# ══════════════════════════════════════════════════════════════════════════════
#
# Un modelo PCA se entrena sobre el conjunto de vectores frame-a-frame de una
# o varias representaciones intermedias (latentes .npz de wav-to-latent /
# spectrogram-to-latent, o espectrogramas STFT .npz de audio_lab / este mismo
# programa). intermediate-to-pca proyecta una representación en las coordenadas
# de sus componentes principales; pca-to-intermediate aplica la transformación
# inversa para recuperar (una aproximación de) la representación original.

PCA_KIND = "latent_lab_pca"

def _load_intermediate_any(path: str):
    """
    Carga una representación intermedia (latente o espectrograma STFT) desde
    .npz o .png y la normaliza a un formato común: (vectors [frames × dim],
    kind, meta) donde meta contiene todo lo necesario para reconstruir el NPZ
    original tras aplicar la inversa de PCA.

    Formatos .png soportados:
      • PNG con metadatos embebidos de wav-to-latent --png / latent-to-png
        (latent_lab_sr, latent_lab_lat_min… en chunks tEXt o sidecar .ll.json)
      • PNG "crudo" de spectrogram-to-latent-train_step1 (sin metadatos propios),
        resuelto vía el manifest.json presente en el mismo directorio.
    """
    ext = Path(path).suffix.lower()
    if ext == ".png":
        return _load_intermediate_png(path)
    if ext != ".npz":
        sys.exit(f"✗  Formato no soportado: {ext!r}  (usa .npz o .png)")

    data = np.load(path)
    if "latents" in data.files:
        latents, sr, hop, lat_min, lat_max = load_latents_npz(path)
        meta = {"sr": sr, "hop": hop,
                "lat_min": lat_min.tolist(), "lat_max": lat_max.tolist()}
        return latents, "latent", meta
    elif "magnitudes" in data.files:
        mags   = data["magnitudes"].astype(np.float32)
        sr     = int(data["sample_rate"])
        window = int(data["window_size"])
        hop_r  = float(data["hop_ratio"])
        freqs  = data["freqs"].astype(np.float32)
        meta = {"sr": sr, "window": window, "hop_ratio": hop_r,
                "freqs": freqs.tolist()}
        return mags, "spectrogram", meta
    else:
        sys.exit(f"✗  {path} no es un NPZ de latente ni de espectrograma reconocido")


def _load_intermediate_png(path: str):
    """Resuelve un .png de representación intermedia (ver _load_intermediate_any)."""
    Image = _import_pil()
    img = Image.open(str(path))
    has_own_meta = any(str(k).startswith("latent_lab") for k in img.info)
    sidecar = Path(path).with_suffix(".ll.json")

    if has_own_meta or sidecar.exists():
        # PNG estilo latents_to_png (wav-to-latent --png, latent-to-png…)
        latents, sr, hop, lat_min, lat_max = png_to_latents(path)
        meta = {"sr": sr, "hop": hop,
                "lat_min": lat_min.tolist(), "lat_max": lat_max.tolist()}
        return latents, "latent", meta

    # PNG "crudo" de spectrogram-to-latent-train_step1: buscar manifest.json
    # en el mismo directorio y localizar la entrada correspondiente.
    pairs_dir = Path(path).parent
    manifest_path = pairs_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"✗  {path} no tiene metadatos propios ni manifest.json en su "
                 f"directorio — no se puede interpretar")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("kind") != "latent_lab_pairs":
        sys.exit(f"✗  {manifest_path} no es un manifest de spectrogram-to-latent-train_step1")

    stem = Path(path).stem
    is_latent = stem.endswith("_latent")
    is_spec   = stem.endswith("_spectrogram")
    if not (is_latent or is_spec):
        sys.exit(f"✗  {path}: nombre no reconocido (se esperaba *_latent.png o "
                 f"*_spectrogram.png, como las que genera step1)")

    img_arr = _load_gray_png(path)             # [H × W] ∈ [0,1], H=dim, W=frames
    if is_latent:
        D = manifest["latent_dim"]
        if img_arr.shape[0] != D:
            sys.exit(f"✗  Altura del PNG ({img_arr.shape[0]}) ≠ latent_dim ({D})")
        lat_min = np.array(manifest["lat_min"], dtype=np.float32)
        lat_max = np.array(manifest["lat_max"], dtype=np.float32)
        latents = latents_denormalize(img_arr.T, lat_min, lat_max)   # [F × D]
        meta = {"sr": manifest["sr"], "hop": manifest["coder_hop"],
                "lat_min": lat_min.tolist(), "lat_max": lat_max.tolist()}
        print(f"  [pca] {path}: PNG crudo de step1 (latente), "
              f"metadatos vía {manifest_path.name}")
        return latents, "latent", meta
    else:
        n_bins = manifest["n_bins"]
        if img_arr.shape[0] != n_bins:
            sys.exit(f"✗  Altura del PNG ({img_arr.shape[0]}) ≠ n_bins ({n_bins})")
        mags = norm_to_mags(img_arr.T, manifest["db_floor"])         # [F × bins]
        freqs = np.fft.rfftfreq(manifest["window"], 1.0 / manifest["sr"]).astype(np.float32)
        meta = {"sr": manifest["sr"], "window": manifest["window"],
                "hop_ratio": manifest["hop_ratio"], "freqs": freqs.tolist()}
        print(f"  [pca] {path}: PNG crudo de step1 (espectrograma), "
              f"metadatos vía {manifest_path.name}")
        return mags, "spectrogram", meta

def _stack_vectors(paths: List[str]):
    """Carga varios NPZ de representación intermedia y concatena sus vectores
    frame-a-frame. Todos deben ser del mismo tipo (latente o espectrograma) y
    la misma dimensionalidad."""
    all_vecs, kind0, meta0 = None, None, None
    chunks = []
    for p in paths:
        vecs, kind, meta = _load_intermediate_any(p)
        if kind0 is None:
            kind0, meta0 = kind, meta
        elif kind != kind0:
            sys.exit(f"✗  {p} es {kind!r} pero los ficheros anteriores eran {kind0!r} "
                     f"— no se puede mezclar latentes y espectrogramas en el mismo PCA")
        elif vecs.shape[1] != chunks[0].shape[1]:
            sys.exit(f"✗  {p} tiene dimensión {vecs.shape[1]} pero los ficheros "
                     f"anteriores tenían {chunks[0].shape[1]}")
        chunks.append(vecs)
        print(f"  [train-pca] {Path(p).name}: {vecs.shape[0]} frames × {vecs.shape[1]} dim  ({kind})")
    return np.concatenate(chunks, axis=0), kind0, meta0


def cmd_train_pca(args):
    """Uno o varios NPZ de representación intermedia → modelo PCA (pca.npz)."""
    vectors, kind, meta = _stack_vectors(args.inputs)
    n_samples, n_features = vectors.shape
    print(f"  [train-pca] corpus total: {n_samples} vectores × {n_features} dim  ({kind})")

    k = args.n_components
    if k is None:
        k = min(n_samples, n_features)
    k = min(k, n_samples, n_features)

    mean = vectors.mean(axis=0)
    centered = (vectors - mean).astype(np.float64)

    print(f"  [train-pca] SVD sobre {n_samples}×{n_features} (k={k})…")
    # SVD económico: para n_samples >> n_features (caso típico aquí) es más
    # rápido diagonalizar la matriz de covarianza (n_features×n_features).
    if n_features <= n_samples:
        cov = (centered.T @ centered) / max(1, n_samples - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1][:k]
        components = eigvecs[:, order].T                     # [k, n_features]
        explained_var = eigvals[order]
    else:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:k]                                  # [k, n_features]
        explained_var = (S[:k] ** 2) / max(1, n_samples - 1)

    total_var = centered.var(axis=0, ddof=1).sum()
    ratio = explained_var / (total_var + 1e-12)
    cum_ratio = np.cumsum(ratio)
    print(f"  [train-pca] varianza explicada por los primeros {min(k,10)} componentes: "
          f"{np.round(ratio[:10], 4)}")
    print(f"  [train-pca] varianza acumulada @k={k}: {cum_ratio[-1]*100:.2f}%")

    out_path = args.output or "pca.npz"
    save_kwargs = dict(
        kind=np.array(PCA_KIND),
        input_kind=np.array(kind),
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        explained_variance=explained_var.astype(np.float32),
        explained_variance_ratio=ratio.astype(np.float32),
        n_samples_seen=np.array(n_samples),
    )
    if kind == "latent":
        save_kwargs.update(
            sr=np.array(meta["sr"]), hop=np.array(meta["hop"]),
            lat_min=np.array(meta["lat_min"], dtype=np.float32),
            lat_max=np.array(meta["lat_max"], dtype=np.float32))
    else:  # spectrogram
        save_kwargs.update(
            sr=np.array(meta["sr"]), window=np.array(meta["window"]),
            hop_ratio=np.array(meta["hop_ratio"]),
            freqs=np.array(meta["freqs"], dtype=np.float32))
    np.savez_compressed(out_path, **save_kwargs)
    print(f"  ✓  Modelo PCA → {out_path}  ({kind}, {n_features}→{k} componentes, "
          f"{cum_ratio[-1]*100:.1f}% varianza explicada)")


def _load_pca_model(path: str):
    data = np.load(path, allow_pickle=False)
    if str(data.get("kind", "")) != PCA_KIND and        not (data.files and "components" in data.files and "mean" in data.files):
        sys.exit(f"✗  {path} no es un checkpoint de train-pca")
    model = {
        "input_kind": str(data["input_kind"]),
        "mean":       data["mean"].astype(np.float32),
        "components": data["components"].astype(np.float32),   # [k, n_features]
        "explained_variance_ratio": data["explained_variance_ratio"].astype(np.float32),
        "sr":         int(data["sr"]),
    }
    if model["input_kind"] == "latent":
        model["hop"]     = int(data["hop"])
        model["lat_min"] = data["lat_min"].astype(np.float32)
        model["lat_max"] = data["lat_max"].astype(np.float32)
    else:
        model["window"]  = int(data["window"])
        model["hop_ratio"] = float(data["hop_ratio"])
        model["freqs"]   = data["freqs"].astype(np.float32)
    k, n_features = model["components"].shape
    print(f"  [pca] {path}: {model['input_kind']}  {n_features}→{k} componentes  "
          f"({model['explained_variance_ratio'].sum()*100:.1f}% varianza explicada)")
    return model


def _save_pca_coords_npz(path: str, coords: np.ndarray, model: dict) -> None:
    """Guarda las coordenadas PCA [frames × k] junto con todo lo necesario
    para invertir la transformación después, en un NPZ autocontenido."""
    save_kwargs = dict(
        kind=np.array("latent_lab_pca_coords"),
        input_kind=np.array(model["input_kind"]),
        coords=coords.astype(np.float32),
        mean=model["mean"],
        components=model["components"],
        sr=np.array(model["sr"]),
    )
    if model["input_kind"] == "latent":
        save_kwargs.update(hop=np.array(model["hop"]),
                           lat_min=model["lat_min"], lat_max=model["lat_max"])
    else:
        save_kwargs.update(window=np.array(model["window"]),
                           hop_ratio=np.array(model["hop_ratio"]),
                           freqs=model["freqs"])
    np.savez_compressed(path, **save_kwargs)
    F, k = coords.shape
    print(f"  ✓  NPZ PCA → {path}  ({F} frames × {k} componentes PCA)")


def cmd_intermediate_to_pca(args):
    """Representación intermedia (.npz latente o espectrograma) + modelo PCA
    → coordenadas PCA (.npz [+ PNG])."""
    model = _load_pca_model(args.pca)
    vectors, kind, meta = _load_intermediate_any(args.input)
    if kind != model["input_kind"]:
        sys.exit(f"✗  {args.input} es {kind!r} pero el modelo PCA se entrenó sobre "
                 f"{model['input_kind']!r}")
    n_features_model = model["components"].shape[1]
    if vectors.shape[1] != n_features_model:
        sys.exit(f"✗  {args.input} tiene dimensión {vectors.shape[1]} pero el modelo "
                 f"PCA espera {n_features_model}")

    coords = (vectors - model["mean"]) @ model["components"].T   # [frames, k]
    k = coords.shape[1]
    print(f"  [intermediate-to-pca] {vectors.shape[0]} frames × {vectors.shape[1]} dim "
          f"→ {vectors.shape[0]} frames × {k} componentes PCA")

    out_path = args.output or Path(args.input).stem + "_pca.npz"
    _save_pca_coords_npz(out_path, coords, model)

    if args.png:
        # Reutiliza el pipeline PNG de latentes: las coordenadas PCA se tratan
        # como un latente genérico de k componentes, normalizado por el rango
        # observado en esta proyección concreta (min/max por componente).
        pca_min = coords.min(axis=0)
        pca_max = coords.max(axis=0)
        png_path = str(Path(out_path).with_suffix(".png"))
        hop_for_png = model.get("hop") or max(1, int(model.get("window", 4096) *
                                                     model.get("hop_ratio", 0.25)))
        latents_to_png(coords, model["sr"], hop_for_png, pca_min, pca_max, png_path)


def cmd_pca_to_intermediate(args):
    """Coordenadas PCA (.npz [o .png]) → representación intermedia original
    (.npz latente o espectrograma), aplicando la inversa de PCA."""
    ext = Path(args.input).suffix.lower()
    if ext == ".png":
        # PNG genérico de coordenadas PCA: requiere --pca para recuperar mean/
        # components/metadatos, ya que el PNG solo guarda min/max de esta proyección.
        if not args.pca:
            sys.exit("✗  --pca es obligatorio para invertir un PNG de coordenadas PCA")
        model = _load_pca_model(args.pca)
        coords, sr, hop, pca_min, pca_max = png_to_latents(args.input)
    elif ext == ".npz":
        data = np.load(args.input)
        if str(data.get("kind", "")) != "latent_lab_pca_coords":
            sys.exit(f"✗  {args.input} no es un NPZ de coordenadas PCA "
                     f"(usa intermediate-to-pca primero)")
        coords = data["coords"].astype(np.float32)
        model = {
            "input_kind": str(data["input_kind"]),
            "mean":       data["mean"].astype(np.float32),
            "components": data["components"].astype(np.float32),
            "sr":         int(data["sr"]),
        }
        if model["input_kind"] == "latent":
            model["hop"] = int(data["hop"])
            model["lat_min"] = data["lat_min"].astype(np.float32)
            model["lat_max"] = data["lat_max"].astype(np.float32)
        else:
            model["window"] = int(data["window"])
            model["hop_ratio"] = float(data["hop_ratio"])
            model["freqs"] = data["freqs"].astype(np.float32)
    else:
        sys.exit(f"✗  Formato no soportado: {ext!r}  (usa .npz o .png)")

    vectors = coords @ model["components"] + model["mean"]     # [frames, n_features]
    print(f"  [pca-to-intermediate] {coords.shape[0]} frames × {coords.shape[1]} "
          f"componentes PCA → {vectors.shape[0]} frames × {vectors.shape[1]} dim "
          f"({model['input_kind']})")

    if model["input_kind"] == "latent":
        out_path = args.output or Path(args.input).stem + "_latent.npz"
        save_latents_npz(out_path, vectors, model["sr"], model["hop"],
                         model["lat_min"], model["lat_max"])
        if args.png:
            png_path = str(Path(out_path).with_suffix(".png"))
            latents_to_png(vectors, model["sr"], model["hop"],
                           model["lat_min"], model["lat_max"], png_path)
    else:
        mags   = np.clip(vectors, 0.0, None).astype(np.float32)
        phases = np.zeros_like(mags)
        out_path = args.output or Path(args.input).stem + "_stft.npz"
        np.savez_compressed(out_path,
                            magnitudes=mags, phases=phases, freqs=model["freqs"],
                            sample_rate=np.array(model["sr"]),
                            window_size=np.array(model["window"]),
                            hop_ratio=np.array(model["hop_ratio"]))
        n_frames = mags.shape[0]
        spec_hop = max(1, int(model["window"] * model["hop_ratio"]))
        dur = n_frames * spec_hop / model["sr"]
        print(f"  ✓  NPZ STFT → {out_path}  ({n_frames} frames × {mags.shape[1]} bins, "
              f"~{dur:.2f}s)")
        print(f"     Fase: cero — compatible con audio_lab reconstruct (Griffin-Lim)")
        if args.wav:
            if args.vocoder:
                print(f"  [pca-to-intermediate] vocoder neuronal ({args.vocoder})...")
                voc, vckpt = load_vocoder(args.vocoder, args.device)
                if vckpt["n_bins"] != mags.shape[1]:
                    sys.exit(f"✗  El vocoder espera {vckpt['n_bins']} bins pero "
                             f"el espectrograma reconstruido tiene {mags.shape[1]}")
                norm  = mags_to_norm(mags, vckpt["db_floor"])
                audio = vocoder_synthesize(voc, norm)
            else:
                print(f"  [pca-to-intermediate] Griffin-Lim {args.gl_iters} iter "
                      f"(phase_init={args.gl_phase_init})…")
                audio = griffin_lim(mags, model["hop_ratio"], n_iter=args.gl_iters,
                                    sr=model["sr"], phase_init=args.gl_phase_init)
            peak  = np.abs(audio).max()
            if peak > 1e-9:
                audio = (audio / peak * 0.85).astype(np.float32)
            write_wav(args.wav, audio, model["sr"])




def cmd_info(args):
    """Diagnóstico rápido de WAV / NPZ / PNG / checkpoint .pt"""
    path = Path(args.input)
    ext  = path.suffix.lower()

    if ext == ".wav":
        audio, sr = read_wav(str(path))
        rms  = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.abs(audio).max())
        print(f"\n  WAV: {path.name}")
        print(f"  Duración     : {len(audio)/sr:.3f}s")
        print(f"  Sample rate  : {sr}Hz")
        print(f"  RMS          : {rms:.5f}  ({20*math.log10(rms+1e-12):.1f} dBFS)")
        print(f"  Peak         : {peak:.5f}")
    elif ext == ".npz":
        data = np.load(str(path))
        pca_kind = str(data["kind"]) if "kind" in data.files else ""
        if pca_kind == PCA_KIND:
            kind = f"modelo PCA ({data['input_kind']})"
        elif pca_kind == "latent_lab_pca_coords":
            kind = f"coordenadas PCA ({data['input_kind']})"
        elif "latents" in data.files:
            kind = "latente"
        elif "magnitudes" in data.files:
            kind = "STFT (audio_lab)"
        else:
            kind = "desconocido"
        print(f"\n  NPZ ({kind}): {path.name}")
        for k in data.files:
            v = data[k]
            print(f"  {k:<16}: shape={v.shape}  dtype={v.dtype}")
        if "latents" in data.files:
            F, D = data["latents"].shape
            sr, hop = int(data["sample_rate"]), int(data["hop_length"])
            print(f"  Duración     : ~{F*hop/sr:.2f}s  ({sr/hop:.1f} frames/s)")
        elif pca_kind == PCA_KIND:
            k, n_features = data["components"].shape
            ratio = float(data["explained_variance_ratio"].sum())
            print(f"  Componentes  : {n_features} → {k}")
            print(f"  Varianza expl.: {ratio*100:.2f}%")
        elif pca_kind == "latent_lab_pca_coords":
            F, k = data["coords"].shape
            print(f"  Coordenadas  : {F} frames × {k} componentes PCA")
    elif ext == ".png":
        Image = _import_pil()
        img   = Image.open(str(path))
        meta  = {k: v for k, v in img.info.items() if k.startswith("latent_lab")}
        print(f"\n  PNG: {path.name}")
        print(f"  Tamaño       : {img.size[0]}×{img.size[1]}px  modo={img.mode}")
        if meta:
            for k, v in meta.items():
                print(f"  {k:<22}: {v}")
        else:
            sidecar = path.with_suffix(".ll.json")
            print(f"  Metadatos    : ✗ sin tEXt de latent_lab"
                  + (f"  (sidecar: {sidecar.name} ✓)" if sidecar.exists() else ""))
    elif ext == ".pt":
        _torch_zoo()
        ckpt = _torch_load(str(path))
        kind = ckpt.get("kind", "desconocido")
        print(f"\n  Checkpoint: {path.name}  ({kind})")
        for k, v in ckpt.items():
            if k in ("encoder", "decoder", "spec2lat", "lat2spec",
                     "gen_s2l", "gen_l2s", "disc_s2l", "disc_l2s",
                     "gen", "disc"):
                n = sum(int(np.prod(t.shape)) for t in v.values())
                print(f"  {k:<14}: state_dict  ({n/1e6:.2f}M parámetros)")
            elif isinstance(v, np.ndarray):
                print(f"  {k:<14}: array shape={v.shape}  dtype={v.dtype}")
            elif isinstance(v, list) and len(v) > 8:
                print(f"  {k:<14}: [{len(v)} valores]  "
                      f"min={min(v):.3f} max={max(v):.3f}")
            else:
                print(f"  {k:<14}: {v}")
    elif ext == ".pth":
        dac = _import_dac()
        model = dac.DAC.load(str(path))
        n = sum(p.numel() for p in model.parameters())
        print(f"\n  Checkpoint DAC oficial pre-entrenado: {path.name}")
        print(f"  sample_rate  : {model.sample_rate}")
        print(f"  hop_length   : {model.hop_length}")
        print(f"  latent_dim   : {model.latent_dim}  (continuo, antes de RVQ)")
        print(f"  n_codebooks  : {getattr(model, 'n_codebooks', '?')}")
        print(f"  parámetros   : {n/1e6:.1f}M")
        print(f"  Uso en latent_lab: --coder {path} en wav-to-latent / latent-to-wav")
    elif ext == ".json":
        with open(str(path)) as f:
            d = json.load(f)
        if d.get("kind") == "latent_lab_pairs":
            print(f"\n  Manifest de pares (spectrogram-to-latent-train_step1): {path.name}")
            print(f"  sr           : {d['sr']}")
            print(f"  window/hop   : {d['window']} / {d['hop_ratio']}")
            print(f"  n_bins       : {d['n_bins']}")
            print(f"  latent_dim   : {d['latent_dim']}")
            print(f"  coder_hop    : {d['coder_hop']}")
            print(f"  ficheros     : {len(d['files'])}")
            for entry in d["files"]:
                print(f"    {entry['stem']:<30} {entry['n_frames']} frames")
            print(f"  Uso: python3 latent_lab.py spectrogram-to-latent-train_step2 "
                  f"{path.parent} -o mapper.pt")
        else:
            print(f"\n  JSON: {path.name}")
            for k, v in d.items():
                if isinstance(v, list):
                    print(f"  {k:<16}: [{len(v)} elementos]")
                else:
                    print(f"  {k:<16}: {v}")
    else:
        print(f"  ✗  Formato no reconocido: {ext}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def _add_device(p):
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Dispositivo torch (default: auto)")

def main():
    parser = argparse.ArgumentParser(
        prog="latent_lab",
        description="Codec neuronal de audio y puentes espectrograma↔latente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ── train-coder ───────────────────────────────────────────────────────────
    p = sub.add_parser("train-coder",
                       help="WAVs → entrena el codec wav↔latente → coder.pt")
    p.add_argument("inputs", nargs="+", help="Ficheros WAV de entrenamiento")
    p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT,
                   help=f"Componentes del espacio latente (default: {DEFAULT_LATENT})")
    p.add_argument("--strides", nargs="+", type=int, default=list(DEFAULT_STRIDES),
                   help="Strides del encoder; su producto es el hop "
                        f"(default: {' '.join(map(str, DEFAULT_STRIDES))} → hop 512)")
    p.add_argument("--base-channels", type=int, default=DEFAULT_BASE_CH,
                   help=f"Canales de la primera conv (default: {DEFAULT_BASE_CH})")
    p.add_argument("--sr", type=int, default=SAMPLE_RATE,
                   help=f"Sample rate de trabajo (default: {SAMPLE_RATE})")
    p.add_argument("--steps", type=int, default=2000,
                   help="Pasos de entrenamiento (default: 2000)")
    p.add_argument("--batch", type=int, default=16,
                   help="Tamaño de batch (default: 16)")
    p.add_argument("--segment", type=int, default=32768,
                   help="Muestras por recorte, múltiplo del hop (default: 32768)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate Adam (default: 1e-4)")
    p.add_argument("--seed", type=int, default=0, help="Semilla RNG (default: 0)")
    p.add_argument("--log-every", type=int, default=50,
                   help="Imprimir pérdida cada N pasos (default: 50)")
    _add_device(p)
    p.add_argument("--output", "-o", default="coder.pt",
                   help="Checkpoint de salida (default: coder.pt)")
    p.set_defaults(func=cmd_train_coder)

    # ── train-vocoder ─────────────────────────────────────────────────────────
    p = sub.add_parser("train-vocoder",
                       help="WAVs → entrena un vocoder neuronal ligero "
                            "(alternativa opcional a Griffin-Lim) → vocoder.pt")
    p.add_argument("inputs", nargs="+", help="Ficheros WAV de entrenamiento")
    p.add_argument("--sr", type=int, default=SAMPLE_RATE,
                   help=f"Sample rate de trabajo (default: {SAMPLE_RATE})")
    p.add_argument("--window", type=int, default=4096,
                   help="Tamaño de ventana STFT, debe coincidir con el usado "
                        "para generar los espectrogramas que luego alimentarás "
                        "al vocoder (default: 4096)")
    p.add_argument("--hop", type=float, default=0.25,
                   help="hop_ratio STFT, ídem (default: 0.25 → hop=1024 con "
                        "--window 4096)")
    p.add_argument("--db-floor", type=float, default=DEFAULT_DB_FLOOR,
                   help=f"Piso dB de normalización (default: {DEFAULT_DB_FLOOR})")
    p.add_argument("--upsample-rates", nargs="+", type=int, default=None,
                   help="Factores de sobremuestreo del vocoder; su producto "
                        "debe ser igual al hop STFT (window×hop-ratio). Si se "
                        "omite, se calculan automáticamente (default: auto)")
    p.add_argument("--hidden", type=int, default=256,
                   help="Canales ocultos iniciales (default: 256)")
    p.add_argument("--steps", type=int, default=2000,
                   help="Pasos de entrenamiento (default: 2000)")
    p.add_argument("--batch", type=int, default=8,
                   help="Tamaño de batch (default: 8)")
    p.add_argument("--frames", type=int, default=32,
                   help="Frames STFT por recorte de entrenamiento. Debe ser "
                        "suficiente para que frames×hop supere ~2048 muestras "
                        "(lo que exige la pérdida STFT multi-escala interna); "
                        "con los valores por defecto 32 ya cumple de sobra "
                        "(default: 32)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Learning rate Adam (default: 2e-4)")
    p.add_argument("--seed", type=int, default=0, help="Semilla RNG (default: 0)")
    p.add_argument("--log-every", type=int, default=50,
                   help="Imprimir pérdida cada N pasos (default: 50)")
    _add_device(p)
    p.add_argument("--output", "-o", default="vocoder.pt",
                   help="Checkpoint de salida (default: vocoder.pt)")
    p.set_defaults(func=cmd_train_vocoder)

    # ── wav-to-latent ─────────────────────────────────────────────────────────
    p = sub.add_parser("wav-to-latent",
                       help="WAV → latentes tiempo×componentes .npz [+ PNG]")
    p.add_argument("input", help="Fichero WAV")
    p.add_argument("--coder", required=True, help="Checkpoint coder.pt")
    p.add_argument("--png", action="store_true",
                   help="Genera también PNG editable (mismo nombre, .png)")
    p.add_argument("--no-flip-y", action="store_true",
                   help="No invertir eje Y del PNG (componente 0 queda arriba)")
    _add_device(p)
    p.add_argument("--output", "-o", default=None, help="NPZ de salida")
    p.set_defaults(func=cmd_wav_to_latent)

    # ── latent-to-wav ─────────────────────────────────────────────────────────
    p = sub.add_parser("latent-to-wav",
                       help="NPZ / PNG de latentes → WAV (decoder)")
    p.add_argument("input", help="Fichero .npz o .png de latentes")
    p.add_argument("--coder", required=True, help="Checkpoint coder.pt")
    p.add_argument("--normalize", action="store_true",
                   help="Normalizar el WAV de salida a 0.85 peak")
    _add_device(p)
    p.add_argument("--output", "-o", default=None, help="WAV de salida")
    p.set_defaults(func=cmd_latent_to_wav)

    # ── latent-to-png ─────────────────────────────────────────────────────────
    p = sub.add_parser("latent-to-png",
                       help="NPZ latente → PNG escala de grises")
    p.add_argument("input", help="Fichero .npz de latentes")
    p.add_argument("--no-flip-y", action="store_true",
                   help="No invertir eje Y (componente 0 queda arriba)")
    p.add_argument("--output", "-o", default=None, help="PNG de salida")
    p.set_defaults(func=cmd_latent_to_png)

    # ── png-to-latent ─────────────────────────────────────────────────────────
    p = sub.add_parser("png-to-latent",
                       help="PNG escala de grises → NPZ latente")
    p.add_argument("input", help="Fichero PNG (idealmente de wav-to-latent --png)")
    p.add_argument("--coder", default=None,
                   help="coder.pt para deducir metadatos si el PNG no los tiene")
    _add_device(p)
    p.add_argument("--output", "-o", default=None, help="NPZ de salida")
    p.set_defaults(func=cmd_png_to_latent)

    # ── spectrogram-to-latent-train_step1 ─────────────────────────────────────
    p = sub.add_parser("spectrogram-to-latent-train_step1",
                       help="WAVs + coder.pt → vuelca pares espectrograma/latente "
                            "como PNG editables (sin entrenar)")
    p.add_argument("inputs", nargs="+", help="Ficheros WAV de entrenamiento")
    p.add_argument("--coder", required=True, help="Checkpoint coder.pt")
    p.add_argument("--window", type=int, default=4096,
                   help="Ventana STFT en muestras, como audio_lab (default: 4096)")
    p.add_argument("--hop", type=float, default=0.25,
                   help="Hop ratio STFT ∈ (0,1] (default: 0.25)")
    p.add_argument("--db-floor", type=float, default=DEFAULT_DB_FLOOR,
                   help=f"Piso dB al normalizar magnitudes (default: {DEFAULT_DB_FLOOR})")
    _add_device(p)
    p.add_argument("--output-dir", "-o", default="pairs",
                   help="Carpeta donde escribir los pares PNG + manifest.json "
                        "(default: pairs/)")
    p.set_defaults(func=cmd_spec_latent_train_step1)

    # ── spectrogram-to-latent-train_step2 ─────────────────────────────────────
    p = sub.add_parser("spectrogram-to-latent-train_step2",
                       help="Carpeta de pares (de step1, editable) → construye "
                            "el traductor espectrograma↔latente → mapper.pt")
    p.add_argument("pairs_dir", help="Carpeta generada por step1 (con manifest.json)")
    p.add_argument("--method", default="pix2pix",
                   choices=["pix2pix", "retrieval", "causal", "causal-adv"],
                   help="pix2pix: U-Net+PatchGAN, imagen completa de una vez. "
                        "retrieval: indexa un banco de pares frame a frame, "
                        "sin entrenar nada (usable luego con --method greedy/"
                        "viterbi). causal: red autorregresiva (columna t ← "
                        "columnas <t), L1 + teacher forcing fijo. causal-adv: "
                        "como causal, + scheduled sampling + adversarial "
                        "(default: pix2pix)")
    p.add_argument("--tile", type=int, default=256,
                   help="[pix2pix/causal/causal-adv] Anchura temporal (en frames) "
                        "de los parches de entrenamiento; en pix2pix debe ser "
                        "múltiplo de 32 (default: 256)")
    p.add_argument("--gen-base", type=int, default=64,
                   help="[pix2pix] Canales base del generador U-Net (default: 64)")
    p.add_argument("--disc-base", type=int, default=64,
                   help="[pix2pix/causal-adv] Canales base del discriminador "
                        "PatchGAN (default: 64)")
    p.add_argument("--l1-weight", type=float, default=100.0,
                   help="[pix2pix/causal-adv] Peso de la pérdida L1 frente a la "
                        "adversarial, como en el paper pix2pix (default: 100.0)")
    p.add_argument("--hidden", type=int, default=64,
                   help="[causal/causal-adv] Canales ocultos de CausalGenerator "
                        "(default: 64)")
    p.add_argument("--context", type=int, default=32,
                   help="[causal/causal-adv] Nº de frames latentes anteriores a "
                        "tener en cuenta; determina el nº de capas causales "
                        "dilatadas (campo receptivo = 2**n_layers ≥ context) "
                        "(default: 32)")
    p.add_argument("--ss-final-prob", type=float, default=0.5,
                   help="[causal-adv] Probabilidad final de scheduled sampling "
                        "(sustituir historia real por la propia predicción del "
                        "modelo); 0 = equivalente a teacher forcing fijo "
                        "(default: 0.5)")
    p.add_argument("--ss-warmup", type=float, default=0.5,
                   help="[causal-adv] Fracción del entrenamiento (0-1) sobre la "
                        "que ss_prob crece linealmente de 0 a --ss-final-prob "
                        "(default: 0.5)")
    p.add_argument("--epochs", type=int, default=50,
                   help="[pix2pix/causal/causal-adv] Epochs sobre el corpus de "
                        "pares (default: 50)")
    p.add_argument("--batch", type=int, default=4,
                   help="[pix2pix/causal/causal-adv] Tamaño de batch de parches "
                        "(default: 4)")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="[pix2pix/causal/causal-adv] Learning rate Adam "
                        "(default: 2e-4)")
    p.add_argument("--seed", type=int, default=0,
                   help="Semilla RNG (default: 0)")
    p.add_argument("--log-every", type=int, default=50,
                   help="Imprimir pérdida cada N pasos (default: 50)")
    _add_device(p)
    p.add_argument("--output", "-o", default="mapper.pt",
                   help="Checkpoint de salida (default: mapper.pt)")
    p.set_defaults(func=cmd_spec_latent_train_step2)

    # ── spectrogram-to-latent ─────────────────────────────────────────────────
    p = sub.add_parser("spectrogram-to-latent",
                       help="NPZ STFT (audio_lab) → NPZ latente [+ PNG]")
    p.add_argument("input", help="Fichero .npz de espectrograma (audio_lab spectrogram)")
    p.add_argument("--mapper", required=True, help="Checkpoint mapper.pt "
                   "(pix2pix o retrieval, ver spectrogram-to-latent-train_step2)")
    p.add_argument("--method", default="auto",
                   choices=["auto", "pix2pix", "greedy", "viterbi",
                            "causal", "causal-adv"],
                   help="Método de traducción. auto: detecta pix2pix/legado/"
                        "causal del checkpoint, o usa viterbi si es un mapper "
                        "de retrieval. greedy/viterbi requieren --method "
                        "retrieval en step2 (default: auto)")
    p.add_argument("--topk", type=int, default=8,
                   help="[viterbi] Nº de candidatos por frame considerados en la "
                        "búsqueda (default: 8)")
    p.add_argument("--join-weight", type=float, default=1.0,
                   help="[viterbi] Peso del coste de unión/continuidad frente al "
                        "de coincidencia con el query (default: 1.0; 0 = greedy "
                        "de facto, ignora continuidad)")
    p.add_argument("--png", action="store_true",
                   help="Genera también PNG editable del latente")
    _add_device(p)
    p.add_argument("--output", "-o", default=None, help="NPZ latente de salida")
    p.set_defaults(func=cmd_spectrogram_to_latent)

    # ── latent-to-spectrogram ─────────────────────────────────────────────────
    p = sub.add_parser("latent-to-spectrogram",
                       help="NPZ/PNG latente → NPZ STFT compatible audio_lab [+ WAV]")
    p.add_argument("input", help="Fichero .npz o .png de latentes")
    p.add_argument("--mapper", required=True, help="Checkpoint mapper.pt "
                   "(pix2pix o retrieval, ver spectrogram-to-latent-train_step2)")
    p.add_argument("--method", default="auto",
                   choices=["auto", "pix2pix", "greedy", "viterbi",
                            "causal", "causal-adv"],
                   help="Método de traducción, igual que en spectrogram-to-latent "
                        "(causal/causal-adv no soportan esta dirección — ver "
                        "--help de spectrogram-to-latent-train_step2) "
                        "(default: auto)")
    p.add_argument("--topk", type=int, default=8,
                   help="[viterbi] Nº de candidatos por frame (default: 8)")
    p.add_argument("--join-weight", type=float, default=1.0,
                   help="[viterbi] Peso del coste de unión/continuidad (default: 1.0)")
    p.add_argument("--mag-max", type=float, default=1.0,
                   help="Escala absoluta de las magnitudes de salida (default: 1.0; "
                        "reconstruct de audio_lab normaliza automáticamente)")
    p.add_argument("--wav", default=None,
                   help="Si se especifica, reconstruye también un WAV "
                        "(Griffin-Lim por defecto, o --vocoder)")
    p.add_argument("--gl-iters", type=int, default=32,
                   help="Iteraciones de Griffin-Lim para --wav (default: 32)")
    p.add_argument("--gl-phase-init", default="random",
                   choices=["random", "propagate"],
                   help="Inicialización de fase de Griffin-Lim (default: random, "
                        "comportamiento histórico; 'propagate' usa fase de "
                        "phase-vocoder como punto de partida — ver train-vocoder "
                        "--help para la alternativa neuronal completa)")
    p.add_argument("--vocoder", default=None,
                   help="Checkpoint de train-vocoder: si se indica, reemplaza "
                        "Griffin-Lim por un vocoder neuronal ligero para --wav "
                        "(default: ninguno, se usa Griffin-Lim como hasta ahora)")
    _add_device(p)
    p.add_argument("--output", "-o", default=None, help="NPZ STFT de salida")
    p.set_defaults(func=cmd_latent_to_spectrogram)

    # ── spectrogram-to-png ────────────────────────────────────────────────────
    p = sub.add_parser("spectrogram-to-png",
                       help="NPZ de espectrograma (audio_lab) → PNG escala de "
                            "grises editable (par gemelo de latent-to-png)")
    p.add_argument("input", help="Fichero .npz de espectrograma (audio_lab spectrogram)")
    p.add_argument("--db-floor", type=float, default=DEFAULT_DB_FLOOR,
                   help=f"Piso dB al normalizar magnitudes (default: {DEFAULT_DB_FLOOR})")
    p.add_argument("--no-flip-y", action="store_true",
                   help="No invertir eje Y del PNG (bin 0 queda arriba)")
    p.add_argument("--output", "-o", default=None, help="PNG de salida")
    p.set_defaults(func=cmd_spectrogram_to_png)

    # ── png-to-spectrogram ────────────────────────────────────────────────────
    p = sub.add_parser("png-to-spectrogram",
                       help="PNG escala de grises → NPZ de espectrograma "
                            "compatible audio_lab (par gemelo de png-to-latent)")
    p.add_argument("input", help="Fichero PNG (idealmente de spectrogram-to-png, "
                                  "o editado a mano)")
    p.add_argument("--sr", type=int, default=None,
                   help="Sample rate, si el PNG no trae metadatos")
    p.add_argument("--window", type=int, default=None,
                   help="Ventana STFT en muestras, si el PNG no trae metadatos")
    p.add_argument("--hop", type=float, default=None,
                   help="Hop ratio STFT, si el PNG no trae metadatos")
    p.add_argument("--db-floor", type=float, default=None,
                   help="Piso dB, si el PNG no trae metadatos")
    p.add_argument("--mag-max", type=float, default=1.0,
                   help="Escala absoluta de las magnitudes de salida (default: 1.0)")
    p.add_argument("--output", "-o", default=None, help="NPZ de salida")
    p.set_defaults(func=cmd_png_to_spectrogram)

    # ── train-pca ───────────────────────────────────────────────────────────────────────────
    p = sub.add_parser("train-pca",
                       help="NPZ/PNG latente(s)/espectrograma(s) → entrena modelo PCA → pca.npz")
    p.add_argument("inputs", nargs="+",
                   help="Ficheros .npz o .png (todos latentes o todos espectrogramas, "
                        "misma dimensión)")
    p.add_argument("--n-components", type=int, default=None,
                   help="Número de componentes principales a conservar "
                        "(default: todos los posibles, min(n_muestras, dim))")
    p.add_argument("--output", "-o", default="pca.npz",
                   help="Modelo PCA de salida (default: pca.npz)")
    p.set_defaults(func=cmd_train_pca)

    # ── intermediate-to-pca ───────────────────────────────────────────────
    p = sub.add_parser("intermediate-to-pca",
                       help="NPZ/PNG latente/espectrograma + modelo PCA → coordenadas PCA .npz [+ PNG]")
    p.add_argument("input", help="Fichero .npz o .png de representación intermedia")
    p.add_argument("--pca", required=True, help="Modelo PCA (de train-pca)")
    p.add_argument("--png", action="store_true",
                   help="Genera también PNG editable de las coordenadas PCA")
    p.add_argument("--output", "-o", default=None, help="NPZ de coordenadas PCA de salida")
    p.set_defaults(func=cmd_intermediate_to_pca)

    # ── pca-to-intermediate ───────────────────────────────────────────────
    p = sub.add_parser("pca-to-intermediate",
                       help="Coordenadas PCA .npz/.png → representación intermedia original "
                            "(latente o espectrograma) [+ WAV]")
    p.add_argument("input", help="Fichero .npz (o .png, con --pca) de coordenadas PCA")
    p.add_argument("--pca", default=None,
                   help="Modelo PCA (obligatorio si input es .png; opcional para .npz)")
    p.add_argument("--png", action="store_true",
                   help="Si la salida es un latente, genera también su PNG")
    p.add_argument("--wav", default=None,
                   help="Si la salida es un espectrograma, reconstruye también un WAV "
                        "(Griffin-Lim por defecto, o --vocoder)")
    p.add_argument("--gl-iters", type=int, default=32,
                   help="Iteraciones de Griffin-Lim para --wav (default: 32)")
    p.add_argument("--gl-phase-init", default="random",
                   choices=["random", "propagate"],
                   help="Inicialización de fase de Griffin-Lim (default: random, "
                        "comportamiento histórico; 'propagate' usa fase de "
                        "phase-vocoder como punto de partida)")
    p.add_argument("--vocoder", default=None,
                   help="Checkpoint de train-vocoder: si se indica, reemplaza "
                        "Griffin-Lim por un vocoder neuronal para --wav (default: "
                        "ninguno, se usa Griffin-Lim como hasta ahora)")
    _add_device(p)
    p.add_argument("--output", "-o", default=None, help="NPZ de salida")
    p.set_defaults(func=cmd_pca_to_intermediate)

    # ── info ──────────────────────────────────────────────────────────────────
    p = sub.add_parser("info", help="Diagnóstico rápido de WAV / NPZ / PNG / .pt")
    p.add_argument("input", help="Fichero a inspeccionar")
    p.set_defaults(func=cmd_info)

    # ── download-pretrained ──────────────────────────────────────────────────
    p = sub.add_parser("download-pretrained",
                       help="Descarga un DAC oficial pre-entrenado y lo verifica",
                       formatter_class=argparse.RawDescriptionHelpFormatter,
                       description=
                       "Variantes disponibles:\n" +
                       "\n".join(f"  {k:<14} sr={v[1]:<6} {v[2]}"
                                 for k, v in PRETRAINED_DAC_MODELS.items()))
    p.add_argument("--variant", default="44khz-8kbps",
                   choices=list(PRETRAINED_DAC_MODELS),
                   help="Variante a descargar (default: 44khz-8kbps, la usada "
                        "en los ejemplos de este programa)")
    p.add_argument("--output-dir", default="pretrained",
                   help="Carpeta de destino (default: pretrained/)")
    p.add_argument("--force", action="store_true",
                   help="Re-descargar aunque ya exista el fichero")
    p.add_argument("--no-verify", action="store_true",
                   help="No cargar el checkpoint tras descargarlo (más rápido, "
                        "pero sin garantía de que funcione)")
    p.set_defaults(func=cmd_download_pretrained)

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    print(f"\n● latent_lab  →  {args.command}")
    args.func(args)
    print()


if __name__ == "__main__":
    main()
