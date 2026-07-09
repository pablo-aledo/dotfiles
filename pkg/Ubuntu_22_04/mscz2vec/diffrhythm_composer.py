#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               DIFFRHYTHM COMPOSER  v1.2                                      ║
║   Generación de canciones completas mediante difusión latente                ║
║   (wrapper unificado, autocontenido, sobre DiffRhythm)                       ║
║                                                                              ║
║  ARQUITECTURA (DiffRhythm, ASLP-LAB):                                        ║
║    VAE de audio (downsampling 2048x)  →  espacio latente 64-dim              ║
║    DiT  (transformer Llama, ~1.1B params)  →  denoiser                       ║
║    CFM (Conditional Flow Matching, Euler, 32 pasos)  +  CFG (strength 4)      ║
║    MuQ-MuLan (embedding 512-dim)  →  style prompt (audio o texto)             ║
║    G2P (espeak-ng, multilingüe)  →  letras alineadas a frames                ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    prepare     — descarga pesos (DiT, VAE, MuQ) a ./pretrained                ║
║    inspect     — diagnostica config, dependencias, device y pesos            ║
║    lrc-token   — tokeniza un .lrc y muestra alineación letra→frame            ║
║    style       — extrae embedding de estilo (.json) de wav o texto            ║
║    infer       — genera canción: lrc + (ref-audio | prompt de texto)          ║
║    edit        — edita segmentos de una canción existente (infilling)         ║
║    round-trip  — VAE encode→decode sobre un wav (diagnóstico del codec)       ║
║                                                                              ║
║  DEPENDENCIAS (Python):                                                       ║
║    torch, torchaudio, torchdiffeq, transformers==4.49, x-transformers,       ║
║    einops, librosa, mutagen, huggingface_hub, muq, numpy                      ║
║  DEPENDENCIAS (sistema):                                                      ║
║    espeak-ng  (libespeak-ng)  — g2p fonético                                 ║
║  PAQUETES DEL PROYECTO (importados, no inlineados):                          ║
║    model  (DiT, CFM),  g2p  (fonemizador),  thirdparty/LangSegment           ║
║                                                                              ║
║  MODO DE USO:                                                                 ║
║    # 1) Preparar pesos (una sola vez, ~5.5 GB)                                ║
║    python diffrhythm_composer.py prepare --variant base                      ║
║    python diffrhythm_composer.py prepare --variant full                      ║
║    python diffrhythm_composer.py prepare --only vae                          ║
║    python diffrhythm_composer.py prepare --only muq                          ║
║    python diffrhythm_composer.py prepare --dry-run        # sin descargar    ║
║                                                                              ║
║    # 2) Diagnosticar entorno                                                  ║
║    python diffrhythm_composer.py inspect                                      ║
║    python diffrhythm_composer.py inspect --variant full                      ║
║                                                                              ║
║    # 3) Tokenizar letras (sin modelo, sólo g2p)                               ║
║    python diffrhythm_composer.py lrc-token --lrc infer/example/eg_cn_full.lrc║
║    python diffrhythm_composer.py lrc-token --lrc infer/example/eg_en_full.lrc --audio-length 285
║                                                                              ║
║    # 4) Extraer embedding de estilo reutilizable                              ║
║    python diffrhythm_composer.py style --ref-prompt "folk, acoustic guitar" --output style.json
║    python diffrhythm_composer.py style --ref-audio infer/example/eg_en.mp3 --output en.json
║                                                                              ║
║    # 5) Generar canción desde texto (prompt de estilo)                         ║
║    python diffrhythm_composer.py infer \\                                      ║
║        --lrc infer/example/eg_cn_full.lrc \\                                  ║
║        --ref-prompt "folk, acoustic guitar, harmonica, touching." \\          ║
║        --audio-length 95 --chunked --output results/cn.wav                    ║
║                                                                              ║
║    # 5b) Generar canción desde audio de referencia                             ║
║    python diffrhythm_composer.py infer \\                                      ║
║        --lrc infer/example/eg_en_full.lrc \\                                  ║
║        --ref-audio infer/example/eg_en.mp3 \\                                ║
║        --audio-length 177 --chunked --output results/en.wav                   ║
║                                                                              ║
║    # 6) Editar segmentos de una canción existente                             ║
║    python diffrhythm_composer.py edit \\                                       ║
║        --lrc infer/example/edit_en.lrc --ref-song infer/example/edit_en.mp3 \\║
║        --edit-segments "[[10.0,20.0]]" --output results/edit.wav              ║
║                                                                              ║
║    # 7) Diagnóstico del codec VAE                                             ║
║    python diffrhythm_composer.py round-trip --input infer/example/eg_en.mp3 \\║
║        --output results/rt.wav                                                ║
║                                                                              ║
║  NOTAS:                                                                      ║
║    • DiffRhythm-base requiere ≥8 GB VRAM con --chunked.                       ║
║    • Sin CUDA/MPS se usa CPU (muy lento para infer/edit).                     ║
║    • --steps y --cfg permiten acortar la difusión (más rápido, peor calidad).  ║
║    • Los .lrc se generan con la herramienta en el HuggingFace Space.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio
from einops import rearrange

# ──────────────────────────────────────────────────────────────────────────────
#  Rutas y entorno
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = PROJECT_ROOT / "pretrained"
CONFIG_PATH = PROJECT_ROOT / "config" / "diffrhythm-1b.json"
G2P_VOCAB_PATH = PROJECT_ROOT / "g2p" / "g2p" / "vocab.json"
VOCAL_NEG_PATH = PROJECT_ROOT / "infer" / "example" / "vocal.npy"

REPO_BY_VARIANT = {
    "base": "ASLP-lab/DiffRhythm-1_2",
    "full": "ASLP-lab/DiffRhythm-1_2-full",
}
VAE_REPO = "ASLP-lab/DiffRhythm-vae"
MUQ_REPO = "OpenMuQ/MuQ-MuLan-large"

SAMPLING_RATE = 44100
DOWNSAMPLE_RATE = 2048
IO_CHANNELS = 2


def _setup_paths() -> None:
    """Asegura que los import del paquete del proyecto funcionen desde cualquier cwd."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(PROJECT_ROOT)


def _auto_espeak_env() -> None:
    """Configura PHONEMIZER_ESPEAK_LIBRARY si no está puesto y detecta la lib."""
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = [
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        "/opt/homebrew/Cellar/espeak-ng/1.52.0/lib/libespeak-ng.dylib",
    ]
    for c in candidates:
        if Path(c).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = c
            return


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio_file(path: str):
    """Carga audio de forma robusta: torchaudio → soundfile → librosa."""
    try:
        return torchaudio.load(path)
    except Exception as e_ta:
        try:
            import soundfile as sf
            data, sr = sf.read(path, always_2d=True)
            t = torch.from_numpy(data).float().T  # [ch, samples]
            return t, sr
        except Exception as e_sf:
            import librosa
            data, sr = librosa.load(path, sr=None, mono=False)
            t = torch.from_numpy(data).float()
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t, sr


def save_audio(path: str, waveform: torch.Tensor, sample_rate: int = SAMPLING_RATE) -> None:
    """Guarda audio de forma robusta, evitando el bug de torchaudio+torchcodec
    con tensores int16 (que distorsiona el rango y produce ruido a escala completa).

    Recibe waveform float en [-1, 1] (cualquier shape [n], [ch, n] o [b, ch, n])
    y lo escribe como PCM_16 vía soundfile (fallback torchaudio float32).
    """
    w = waveform.detach().cpu().float()
    # reducir a [ch, n]
    if w.dim() == 1:
        w = w.unsqueeze(0)
    elif w.dim() == 3:
        w = w.squeeze(0)
    # peak-normalizar y clampar por seguridad
    peak = w.abs().max()
    if peak > 1e-8:
        w = w / peak
    w = w.clamp(-1, 1)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(out), w.numpy().T, sample_rate, subtype="PCM_16")
    except Exception:
        # fallback: torchaudio con float32 (sin int16, que rompe con torchcodec)
        torchaudio.save(str(out), w, sample_rate=sample_rate)


def max_frames_for(audio_length: int) -> int:
    if audio_length == 95:
        return 2048
    if 95 < audio_length <= 285:
        return 6144
    raise ValueError(
        f"audio_length inválido: {audio_length}. "
        "Soportados: 95 o cualquier valor entre 96 y 285 (inclusive)."
    )


# ──────────────────────────────────────────────────────────────────────────────
#  Utilidades de audio (inlineadas de infer/infer_utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_audio(y, target_dbfs=0):
    max_amp = torch.max(torch.abs(y))
    target_amp = 10.0 ** (target_dbfs / 20.0)
    return y * (target_amp / max_amp)


def set_audio_channels(audio, target_channels):
    if target_channels == 1:
        return audio.mean(1, keepdim=True)
    if target_channels == 2:
        if audio.shape[1] == 1:
            return audio.repeat(1, 2, 1)
        if audio.shape[1] > 2:
            return audio[:, :2, :]
    return audio


class PadCrop(torch.nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if not self.randomize else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, : min(s, self.n_samples)] = signal[:, start:end]
        return output


def prepare_audio(audio, in_sr, target_sr, target_length, target_channels, device):
    audio = audio.to(device)
    if in_sr != target_sr:
        audio = torchaudio.functional.Resample(in_sr, target_sr).to(device)(audio)
    if target_length is not None:
        audio = PadCrop(target_length, randomize=False)(audio)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0).unsqueeze(0)
    elif audio.dim() == 2:
        audio = audio.unsqueeze(0)
    return set_audio_channels(audio, target_channels)


def vae_sample(mean, scale):
    stdev = torch.nn.functional.softplus(scale) + 1e-4
    var = stdev * stdev
    logvar = torch.log(var)
    latents = torch.randn_like(mean) * stdev + mean
    kl = (mean * mean + var - logvar - 1).sum(1).mean()
    return latents, kl


def decode_audio(latents, vae_model, chunked=False, overlap=32, chunk_size=128):
    downsampling_ratio = DOWNSAMPLE_RATE
    io_channels = IO_CHANNELS
    if not chunked:
        return vae_model.decode_export(latents)
    hop = chunk_size - overlap
    total = latents.shape[2]
    batch = latents.shape[0]
    chunks, i = [], 0
    for i in range(0, total - chunk_size + 1, hop):
        chunks.append(latents[:, :, i : i + chunk_size])
    if i + chunk_size != total:
        chunks.append(latents[:, :, -chunk_size:])
    chunks = torch.stack(chunks)
    num = chunks.shape[0]
    spl = downsampling_ratio
    y_size = total * spl
    y_final = torch.zeros((batch, io_channels, y_size)).to(latents.device)
    for i in range(num):
        y_chunk = vae_model.decode_export(chunks[i, :])
        if i == num - 1:
            t_end = y_size
            t_start = t_end - y_chunk.shape[2]
        else:
            t_start = i * hop * spl
            t_end = t_start + chunk_size * spl
        ol = (overlap // 2) * spl
        cs, ce = 0, y_chunk.shape[2]
        if i > 0:
            t_start += ol
            cs += ol
        if i < num - 1:
            t_end -= ol
            ce -= ol
        y_final[:, :, t_start:t_end] = y_chunk[:, :, cs:ce]
    return y_final


def encode_audio(audio, vae_model, chunked=False, overlap=32, chunk_size=128):
    downsampling_ratio = DOWNSAMPLE_RATE
    latent_dim = 128
    if not chunked:
        return vae_model.encode_export(audio)
    spl = downsampling_ratio
    total = audio.shape[2]
    batch = audio.shape[0]
    chunk_size *= spl
    overlap *= spl
    hop = chunk_size - overlap
    chunks, i = [], 0
    for i in range(0, total - chunk_size + 1, hop):
        chunks.append(audio[:, :, i : i + chunk_size])
    if i + chunk_size != total:
        chunks.append(audio[:, :, -chunk_size:])
    chunks = torch.stack(chunks)
    num = chunks.shape[0]
    y_size = total // spl
    y_final = torch.zeros((batch, latent_dim, y_size)).to(audio.device)
    for i in range(num):
        y_chunk = vae_model.encode_export(chunks[i, :])
        if i == num - 1:
            t_end = y_size
            t_start = t_end - y_chunk.shape[2]
        else:
            t_start = i * hop // spl
            t_end = t_start + chunk_size // spl
        ol = overlap // spl // 2
        cs, ce = 0, y_chunk.shape[2]
        if i > 0:
            t_start += ol
            cs += ol
        if i < num - 1:
            t_end -= ol
            ce -= ol
        y_final[:, :, t_start:t_end] = y_chunk[:, :, cs:ce]
    return y_final


# ──────────────────────────────────────────────────────────────────────────────
#  Letras: parseo + tokenización G2P (inlineado de infer/infer_utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def parse_lyrics(lyrics: str):
    out = []
    lyrics = lyrics.strip()
    for line in lyrics.split("\n"):
        try:
            time, lyric = line[1:9], line[10:]
            mins, secs = time.split(":")
            secs = int(mins) * 60 + float(secs)
            out.append((secs, lyric.strip()))
        except Exception:
            continue
    return out


class CNENTokenizer:
    def __init__(self):
        with open(G2P_VOCAB_PATH, "r", encoding="utf-8") as f:
            self.phone2id = json.load(f)["vocab"]
        self.id2phone = {v: k for (k, v) in self.phone2id.items()}
        from g2p.g2p_generation import chn_eng_g2p

        self.tokenizer = chn_eng_g2p

    def encode(self, text):
        _, token = self.tokenizer(text)
        return [x + 1 for x in token]

    def decode(self, token):
        return "|".join([self.id2phone[x - 1] for x in token])


def get_lrc_token(max_frames, text, tokenizer, max_secs, device):
    comma_id, period_id = 1, 2
    lrc_with_time = parse_lyrics(text)
    lrc_with_time = [(t, tokenizer.encode(line)) for (t, line) in lrc_with_time]
    lrc_with_time = [(t, line) for (t, line) in lrc_with_time if t < max_secs]
    if max_frames == 2048 and len(lrc_with_time) >= 1:
        lrc_with_time = lrc_with_time[:-1]

    end_frame = max_frames if max_frames == 2048 else int(max_secs * (SAMPLING_RATE / DOWNSAMPLE_RATE))
    end_frame = min(end_frame, max_frames)

    normalized_duration = end_frame / max_frames
    lrc = torch.zeros((end_frame,), dtype=torch.long)
    last_end = 0
    for time_start, line in lrc_with_time:
        tokens = [tok if tok != period_id else comma_id for tok in line] + [period_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        n = tokens.shape[0]
        gt_frame_start = int(time_start * SAMPLING_RATE / DOWNSAMPLE_RATE)
        frame_start = max(gt_frame_start, last_end)
        frame_len = min(n, end_frame - frame_start)
        lrc[frame_start : frame_start + frame_len] = tokens[:frame_len]
        last_end = frame_start + frame_len

    lrc_emb = lrc.unsqueeze(0).to(device)
    nstart = torch.tensor(0.0).unsqueeze(0).to(device).half()
    ndur = torch.tensor(normalized_duration).unsqueeze(0).to(device).half()
    return lrc_emb, nstart, end_frame, ndur


# ──────────────────────────────────────────────────────────────────────────────
#  Style prompt (MuQ-MuLan) — inlineado de infer/infer_utils.py
# ──────────────────────────────────────────────────────────────────────────────

def get_negative_style_prompt(device):
    vocal = torch.from_numpy(np.load(VOCAL_NEG_PATH)).to(device).half()
    return vocal


@torch.no_grad()
def get_style_prompt(model, wav_path=None, prompt=None):
    mulan = model
    if prompt is not None:
        return mulan(texts=prompt).half()
    import librosa
    from mutagen.mp3 import MP3

    ext = os.path.splitext(wav_path)[-1].lower()
    if ext == ".mp3":
        audio_len = MP3(wav_path).info.length
    elif ext in (".wav", ".flac"):
        audio_len = librosa.get_duration(path=wav_path)
    else:
        raise ValueError(f"Formato no soportado: {ext}")
    if audio_len < 10:
        print(f"  ! audio {wav_path} dura {audio_len:.2f}s (se recomiendan ≥10s)")
    assert audio_len >= 10, "el audio de referencia debe durar al menos 10s"
    mid = audio_len // 2
    start = mid - 5
    wav, _ = librosa.load(wav_path, sr=24000, offset=start, duration=10)
    wav = torch.tensor(wav).unsqueeze(0).to(model.device)
    with torch.no_grad():
        emb = mulan(wavs=wav)
    return emb.half()


# ──────────────────────────────────────────────────────────────────────────────
#  Latente de referencia / edición (inlineado de infer/infer_utils.py)
# ──────────────────────────────────────────────────────────────────────────────

def get_reference_latent(device, max_frames, edit, pred_segments, ref_song, vae_model):
    if edit:
        input_audio, in_sr = load_audio_file(ref_song)
        input_audio = prepare_audio(
            input_audio, in_sr=in_sr, target_sr=SAMPLING_RATE,
            target_length=None, target_channels=IO_CHANNELS, device=device,
        )
        input_audio = normalize_audio(input_audio, -6)
        with torch.no_grad():
            latent = encode_audio(input_audio, vae_model, chunked=True)  # [b d t]
            mean, scale = latent.chunk(2, dim=1)
            prompt, _ = vae_sample(mean, scale)
            prompt = prompt.transpose(1, 2)  # [b t d]
        segs = json.loads(pred_segments)
        pred_frames = []
        for st, et in segs:
            sf = 0 if st == -1 else int(st * SAMPLING_RATE / DOWNSAMPLE_RATE)
            ef = max_frames if et == -1 else int(et * SAMPLING_RATE / DOWNSAMPLE_RATE)
            pred_frames.append((sf, ef))
        return prompt, pred_frames
    else:
        prompt = torch.zeros(1, max_frames, 64).to(device)
        return prompt, [(0, max_frames)]


# ──────────────────────────────────────────────────────────────────────────────
#  Carga de modelos
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(model, ckpt_path, device, dtype="fp16"):
    if dtype == "fp16":
        model = model.half()
    elif dtype == "bf16":
        model = model.bfloat16()
    else:
        model = model.float()
    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file
        # los safetensors son un state_dict plano → envolver para extraer model_state_dict
        checkpoint = {"model_state_dict": load_file(ckpt_path)}
    else:
        # los .pt de DiffRhythm YA son {"model_state_dict": {...}} (y a veces ema_model_state_dict,
        # step, initted). NO hay que envolverlos otra vez.
        checkpoint = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    # ser robusto: si bajo la clave model_state_dict hay un dict con model_state_dict anidado,
    # descender un nivel (compat con checkpoints que envuelven dos veces).
    msd = checkpoint.get("model_state_dict", checkpoint)
    if isinstance(msd, dict) and "model_state_dict" in msd:
        msd = msd["model_state_dict"]
    missing, unexpected = model.load_state_dict(msd, strict=False)
    n_loaded = len(msd) - len(unexpected)
    print(f"  load_checkpoint: {n_loaded}/{len(msd)} tensores cargados "
          f"({len(missing)} missing, {len(unexpected)} unexpected)")
    return model.to(device)


def prepare_model(max_frames, device, variant="base", dtype="fp16"):
    from model import DiT, CFM

    repo_id = REPO_BY_VARIANT[variant]
    dit_ckpt = os.path.join(
        PRETRAINED_DIR, f"models--{repo_id.replace('/', '--')}", "snapshots"
    )
    # resolver vía hf_hub_download (usa caché, no redescarga)
    from huggingface_hub import hf_hub_download
    dit_ckpt_path = hf_hub_download(
        repo_id=repo_id, filename="cfm_model.pt", cache_dir=str(PRETRAINED_DIR)
    )
    with open(CONFIG_PATH) as f:
        model_config = json.load(f)
    cfm = CFM(
        transformer=DiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames,
    ).to(device)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, dtype=dtype)

    tokenizer = CNENTokenizer()

    from muq import MuQMuLan
    muq = MuQMuLan.from_pretrained(MUQ_REPO, cache_dir=str(PRETRAINED_DIR)).to(device).eval()

    vae_ckpt_path = hf_hub_download(
        repo_id=VAE_REPO, filename="vae_model.pt", cache_dir=str(PRETRAINED_DIR)
    )
    vae = torch.jit.load(vae_ckpt_path, map_location="cpu").to(device)

    return cfm, tokenizer, muq, vae


# ──────────────────────────────────────────────────────────────────────────────
#  Bucle de inferencia (inlineado de infer/infer.py)
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(cfm_model, vae_model, cond, text, duration, style_prompt,
                  negative_style_prompt, start_time, pred_frames,
                  batch_infer_num, song_duration, chunked=False,
                  steps=32, cfg_strength=4.0):
    # Sincronizar dtypes: las funciones de extracción devuelven .half() por defecto,
    # pero el modelo puede estar en fp32 (CPU) o bf16. Todo al dtype del modelo.
    mdt = next(cfm_model.parameters()).dtype
    cond = cond.to(mdt)
    style_prompt = style_prompt.to(mdt)
    negative_style_prompt = negative_style_prompt.to(mdt)
    start_time = start_time.to(mdt)
    song_duration = song_duration.to(mdt)
    with torch.inference_mode():
        latents, _ = cfm_model.sample(
            cond=cond, text=text, duration=duration,
            style_prompt=style_prompt, max_duration=duration,
            song_duration=song_duration,
            negative_style_prompt=negative_style_prompt,
            steps=steps, cfg_strength=cfg_strength,
            start_time=start_time, latent_pred_segments=pred_frames,
            batch_infer_num=batch_infer_num,
        )
        outputs = []
        for latent in latents:
            latent = latent.to(torch.float32).transpose(1, 2)  # [b d t]
            output = decode_audio(latent, vae_model, chunked=chunked)
            output = rearrange(output, "b d n -> d (b n)")
            # Peak-normalizar y clampar; la conversión final a PCM_16 la hace save_audio
            output = output.to(torch.float32)
            peak = output.abs().max()
            if peak > 1e-8:
                output = output / peak
            output = output.clamp(-1, 1)
            outputs.append(output)
        return outputs


# ──────────────────────────────────────────────────────────────────────────────
#  Comandos
# ──────────────────────────────────────────────────────────────────────────────

def cmd_prepare(args):
    """Descarga pesos de HuggingFace a ./pretrained."""
    from huggingface_hub import hf_hub_download, snapshot_download, HfApi

    api = HfApi()
    targets = []  # (repo, filename, repo_label)
    only = args.only
    if only:
        if only == "dit":
            targets.append((REPO_BY_VARIANT[args.variant], "cfm_model.pt", "DiT"))
        elif only == "vae":
            targets.append((VAE_REPO, "vae_model.pt", "VAE"))
        elif only == "muq":
            targets.append((MUQ_REPO, "__snapshot__", "MuQ"))
        else:
            print(f"  --only inválido: {only} (usa dit|vae|muq)")
            return 1
    else:
        targets = [
            (REPO_BY_VARIANT[args.variant], "cfm_model.pt", "DiT"),
            (VAE_REPO, "vae_model.pt", "VAE"),
            (MUQ_REPO, "__snapshot__", "MuQ"),
        ]

    print(f"prepare: variant={args.variant} dry_run={args.dry_run}")
    PRETRAINED_DIR.mkdir(exist_ok=True)
    for repo, fn, label in targets:
        try:
            if args.dry_run:
                info = api.get_paths_info(repo, [fn]) if fn != "__snapshot__" else None
                size = info[0].size if info else "n/a"
                files = api.list_repo_files(repo) if fn == "__snapshot__" else [fn]
                print(f"  [{label}] {repo}  ({', '.join(files)})  "
                      f"{round(size/1e6,1) if isinstance(size,(int,float)) else size} MB (dry-run)")
            else:
                if fn == "__snapshot__":
                    snapshot_download(repo_id=repo, cache_dir=str(PRETRAINED_DIR))
                else:
                    hf_hub_download(repo_id=repo, filename=fn, cache_dir=str(PRETRAINED_DIR))
                print(f"  [{label}] {repo} → OK")
        except Exception as e:
            print(f"  [{label}] {repo} → ERROR: {e}")
    print(f"prepare: completo.  pretrained_dir={PRETRAINED_DIR}")
    return 0


def cmd_inspect(args):
    """Diagnostica configuración, dependencias, dispositivo y pesos."""
    print("═" * 64)
    print("  DIFFRHYTHM COMPOSER — INSPECT")
    print("═" * 64)
    # dispositivo
    dev = pick_device()
    print(f"device           : {dev}")
    if dev == "cuda":
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM total      : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    # config
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    print(f"config           : {CONFIG_PATH.name}")
    print(f"  model_type      : {cfg.get('model_type')}")
    m = cfg["model"]
    print(f"  dim/depth/heads : {m['dim']}/{m['depth']}/{m['heads']}")
    print(f"  mel_dim         : {m['mel_dim']}")
    print(f"  text_num_embeds : {m['text_num_embeds']}")
    # parámetros (sin construir pesos, sólo la arquitectura)
    try:
        from model import DiT
        for mf, label in [(2048, "base (95s)"), (6144, "full (285s)")]:
            net = DiT(**cfg["model"], max_frames=mf)
            n = sum(p.numel() for p in net.parameters())
            print(f"  params [{label}] : {n/1e6:.1f} M")
            del net
    except Exception as e:
        print(f"  params          : no se pudo construir el modelo ({e})")
    # variant
    print(f"variant          : {args.variant}  → {REPO_BY_VARIANT[args.variant]}")
    # dependencias python
    deps = ["torch", "torchaudio", "torchdiffeq", "transformers",
            "x_transformers", "einops", "librosa", "mutagen",
            "huggingface_hub", "muq", "phonemizer", "onnxruntime", "numpy"]
    print("deps (python)    :")
    for d in deps:
        try:
            mod = __import__(d)
            print(f"  {d:<18} OK  {getattr(mod, '__version__', '')}")
        except Exception as e:
            print(f"  {d:<18} FALTA  ({e.__class__.__name__})")
    # espeak
    _auto_espeak_env()
    espeak_lib = os.environ.get("PHONEMIZER_ESPEAK_LIBRARY")
    print(f"espeak-ng lib    : {espeak_lib or 'NO DETECTADO (g2p fallará)'}")
    # pesos descargados
    print(f"pretrained dir   : {PRETRAINED_DIR}  "
          f"({'existe' if PRETRAINED_DIR.exists() else 'NO CREADO'})")
    if PRETRAINED_DIR.exists():
        for label, repo in [("DiT", REPO_BY_VARIANT[args.variant]),
                            ("VAE", VAE_REPO), ("MuQ", MUQ_REPO)]:
            d = PRETRAINED_DIR / f"models--{repo.replace('/', '--')}"
            ok = d.exists()
            print(f"  {label:<6} {'✔' if ok else '✗'}  {repo}")
    print("═" * 64)
    return 0


def cmd_lrc_token(args):
    """Tokeniza un .lrc y muestra alineación letra→frame."""
    _auto_espeak_env()
    device = pick_device()
    max_frames = max_frames_for(args.audio_length)
    max_secs = max_frames / (SAMPLING_RATE / DOWNSAMPLE_RATE)

    with open(args.lrc, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CNENTokenizer()
    raw = parse_lyrics(text)
    print(f"lrc: {args.lrc}")
    print(f"audio_length={args.audio_length}s  max_frames={max_frames}  max_secs={max_secs:.1f}")
    print(f"líneas parseadas: {len(raw)}")
    print("─" * 72)
    print(f"{'t (s)':>7}  {'frame':>6}  {'#tok':>4}  texto / fonemas")
    print("─" * 72)
    for t, line in raw:
        if t >= max_secs:
            print(f"{t:>7.2f}  {'>max':>6}  {0:>4}  (descartada)  {line[:40]}")
            continue
        ph, tok = tokenizer.tokenizer(line)
        frame = int(t * SAMPLING_RATE / DOWNSAMPLE_RATE)
        print(f"{t:>7.2f}  {frame:>6}  {len(tok):>4}  {line[:30]}")
        if args.verbose:
            print(f"          fonemas: {ph}")
    print("─" * 72)
    lrc_emb, nstart, end_frame, ndur = get_lrc_token(max_frames, text, tokenizer, max_secs, device)
    print(f"end_frame (duración latente): {end_frame}")
    n_nonzero = int((lrc_emb != 0).sum().item())
    print(f"tokens totales colocados   : {n_nonzero}")
    print(f"normalized_duration        : {float(ndur):.4f}")
    if args.verbose:
        ids = lrc_emb.squeeze(0).tolist()
        nz = [(i, ids[i]) for i in range(len(ids)) if ids[i] != 0]
        print(f"posiciones no-cero (idx,token): {nz[:20]}{' ...' if len(nz)>20 else ''}")
    return 0


def cmd_style(args):
    """Extrae embedding de estilo (MuQ-MuLan) y lo guarda como .json."""
    device = pick_device()
    from muq import MuQMuLan
    print(f"style: cargando MuQ-MuLan ...")
    muq = MuQMuLan.from_pretrained(MUQ_REPO, cache_dir=str(PRETRAINED_DIR)).to(device).eval()
    if args.ref_prompt:
        print(f"  mode=text  prompt={args.ref_prompt!r}")
        emb = get_style_prompt(muq, prompt=args.ref_prompt)
    else:
        print(f"  mode=audio ref={args.ref_audio}")
        emb = get_style_prompt(muq, wav_path=args.ref_audio)
    emb_f32 = emb.float().cpu().numpy()
    out = {
        "source": args.ref_prompt or args.ref_audio,
        "dim": int(emb_f32.shape[-1]),
        "embedding": emb_f32.tolist(),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  embedding shape = {emb_f32.shape}  → {out_path}")
    return 0


def cmd_infer(args):
    """Genera una canción a partir de lrc + estilo."""
    assert args.ref_prompt or args.ref_audio, "indica --ref-prompt o --ref-audio"
    assert not (args.ref_prompt and args.ref_audio), "sólo uno de los dos"
    device = pick_device()
    max_frames = max_frames_for(args.audio_length)
    max_secs = max_frames / (SAMPLING_RATE / DOWNSAMPLE_RATE)

    print(f"infer: variant={args.variant} audio_length={args.audio_length}s "
          f"max_frames={max_frames} device={device} dtype={args.dtype}")
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device, variant=args.variant, dtype=args.dtype)

    with open(args.lrc, "r", encoding="utf-8") as f:
        lrc = f.read()
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc, tokenizer, max_secs, device)

    if args.ref_audio:
        style_prompt = get_style_prompt(muq, wav_path=args.ref_audio)
    else:
        style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)
    negative_style_prompt = get_negative_style_prompt(device)

    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, False, None, None, vae)

    t0 = time.time()
    songs = run_inference(
        cfm_model=cfm, vae_model=vae, cond=latent_prompt, text=lrc_prompt,
        duration=end_frame, style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt, start_time=start_time,
        pred_frames=pred_frames, batch_infer_num=args.batch_infer_num,
        song_duration=song_duration, chunked=args.chunked,
        steps=args.steps, cfg_strength=args.cfg,
    )
    dt = time.time() - t0
    print(f"infer cost {dt:.2f}s  ({args.steps} pasos × CFG)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    chosen = random.sample(songs, 1)[0]
    save_audio(str(out), chosen, SAMPLING_RATE)
    dur = chosen.shape[-1] / SAMPLING_RATE
    print(f"  → {out}  ({tuple(chosen.shape)}, {dur:.1f}s)")
    return 0


def cmd_edit(args):
    """Edita segmentos de una canción existente (infilling latente)."""
    assert args.ref_song and args.edit_segments, (
        "edit requiere --ref-song y --edit-segments")
    device = pick_device()
    max_frames = max_frames_for(args.audio_length)
    max_secs = max_frames / (SAMPLING_RATE / DOWNSAMPLE_RATE)

    print(f"edit: variant={args.variant} audio_length={args.audio_length}s "
          f"segments={args.edit_segments} device={device}")
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device, variant=args.variant, dtype=args.dtype)

    with open(args.lrc, "r", encoding="utf-8") as f:
        lrc = f.read()
    lrc_prompt, start_time, end_frame, song_duration = get_lrc_token(
        max_frames, lrc, tokenizer, max_secs, device)

    if args.ref_prompt:
        style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)
    elif args.ref_audio:
        style_prompt = get_style_prompt(muq, wav_path=args.ref_audio)
    else:
        # sin estilo explícito: usar el negativo neutro también como positivo
        style_prompt = get_negative_style_prompt(device)
    negative_style_prompt = get_negative_style_prompt(device)

    latent_prompt, pred_frames = get_reference_latent(
        device, max_frames, True, args.edit_segments, args.ref_song, vae)

    t0 = time.time()
    songs = run_inference(
        cfm_model=cfm, vae_model=vae, cond=latent_prompt, text=lrc_prompt,
        duration=end_frame, style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt, start_time=start_time,
        pred_frames=pred_frames, batch_infer_num=args.batch_infer_num,
        song_duration=song_duration, chunked=args.chunked,
        steps=args.steps, cfg_strength=args.cfg,
    )
    dt = time.time() - t0
    print(f"edit cost {dt:.2f}s")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    chosen = random.sample(songs, 1)[0]
    save_audio(str(out), chosen, SAMPLING_RATE)
    dur = chosen.shape[-1] / SAMPLING_RATE
    print(f"  → {out}  ({tuple(chosen.shape)}, {dur:.1f}s)")
    return 0


def cmd_round_trip(args):
    """VAE encode→decode sobre un wav (diagnóstico del codec)."""
    device = pick_device()
    from huggingface_hub import hf_hub_download
    print(f"round-trip: device={device} input={args.input}")
    vae_ckpt = hf_hub_download(
        repo_id=VAE_REPO, filename="vae_model.pt", cache_dir=str(PRETRAINED_DIR))
    vae = torch.jit.load(vae_ckpt, map_location="cpu").to(device)

    audio, in_sr = load_audio_file(args.input)
    audio = prepare_audio(audio, in_sr=in_sr, target_sr=SAMPLING_RATE,
                          target_length=None, target_channels=IO_CHANNELS, device=device)
    audio = normalize_audio(audio, -6)
    print(f"  audio shape={tuple(audio.shape)} sr={SAMPLING_RATE}")
    t0 = time.time()
    with torch.no_grad():
        latent = encode_audio(audio, vae, chunked=args.chunked)  # [b d t]
        mean, scale = latent.chunk(2, dim=1)
        z, _ = vae_sample(mean, scale)
        recon = vae.decode_export(z)
    dt = time.time() - t0
    print(f"  latent shape={tuple(latent.shape)}  z={tuple(z.shape)}")
    print(f"  recon shape={tuple(recon.shape)}  cost={dt:.2f}s")
    recon = rearrange(recon, "b d n -> d (b n)")
    out = Path(args.output)
    save_audio(str(out), recon.float(), SAMPLING_RATE)
    dur = recon.shape[-1] / SAMPLING_RATE
    print(f"  → {out}  ({dur:.1f}s)")
    return 0


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        prog="diffrhythm_composer",
        description="Generación de canciones completas mediante difusión latente (wrapper de DiffRhythm)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")
    sub.required = True

    # prepare
    pp = sub.add_parser("prepare", help="descarga pesos (DiT, VAE, MuQ) a ./pretrained")
    pp.add_argument("--variant", choices=["base", "full"], default="base")
    pp.add_argument("--only", choices=["dit", "vae", "muq"], default=None,
                    help="descargar sólo un componente")
    pp.add_argument("--dry-run", action="store_true",
                    help="resolver URLs/tamaños sin descargar")
    pp.set_defaults(func=cmd_prepare)

    # inspect
    pi = sub.add_parser("inspect", help="diagnostica config, deps, device y pesos")
    pi.add_argument("--variant", choices=["base", "full"], default="base")
    pi.set_defaults(func=cmd_inspect)

    # lrc-token
    pl = sub.add_parser("lrc-token", help="tokeniza un .lrc y muestra alineación")
    pl.add_argument("--lrc", required=True, metavar="FILE")
    pl.add_argument("--audio-length", type=int, default=95, metavar="SEC",
                    help="95 (base) o 96–285 (full)")
    pl.add_argument("--verbose", action="store_true")
    pl.set_defaults(func=cmd_lrc_token)

    # style
    ps = sub.add_parser("style", help="extrae embedding de estilo → .json")
    gstyle = ps.add_mutually_exclusive_group(required=True)
    gstyle.add_argument("--ref-prompt", default=None, metavar="TEXT")
    gstyle.add_argument("--ref-audio", default=None, metavar="FILE")
    ps.add_argument("--output", default="style.json", metavar="FILE")
    ps.set_defaults(func=cmd_style)

    # infer
    pf = sub.add_parser("infer", help="genera canción: lrc + estilo",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        description=textwrap.dedent("""\
        Genera una canción completa condicionada en letras (.lrc) y estilo.

        Estilo:  --ref-prompt "texto descriptivo"  O  --ref-audio referencia.wav
        Modos de longitud:  95s (base, max_frames 2048)  o  96–285s (full, 6144)

        Ejemplos:
          infer --lrc cancion.lrc --ref-prompt "folk, acoustic guitar" --audio-length 95
          infer --lrc cancion.lrc --ref-audio ref.mp3 --audio-length 177 --chunked
        """))
    pf.add_argument("--lrc", required=True, metavar="FILE")
    g = pf.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref-prompt", default=None, metavar="TEXT")
    g.add_argument("--ref-audio", default=None, metavar="FILE")
    pf.add_argument("--variant", choices=["base", "full"], default="base")
    pf.add_argument("--audio-length", type=int, default=95, metavar="SEC")
    pf.add_argument("--output", default="results/output.wav", metavar="FILE")
    pf.add_argument("--chunked", action="store_true", help="decodificación VAE por chunks (≤8GB VRAM)")
    pf.add_argument("--batch-infer-num", type=int, default=1, metavar="N")
    pf.add_argument("--steps", type=int, default=32, help="pasos Euler (menos = más rápido)")
    pf.add_argument("--cfg", type=float, default=4.0, help="classifier-free guidance strength")
    pf.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="precisión (CPU: usar fp32)")
    pf.set_defaults(func=cmd_infer)

    # edit
    pe = sub.add_parser("edit", help="edita segmentos de una canción existente",
                        formatter_class=argparse.RawDescriptionHelpFormatter,
                        description=textwrap.dedent("""\
        Edición por infilling latente: conserva el audio fuera de los segmentos
        y regenera sólo lo indicado en --edit-segments.

        El --ref-song debe durar ≥ audio-length (en segundos), para que su
        latente alcance max_frames (95s → 2048 frames; 285s → 6144).
        Se recorta si es más largo.

        Formato de --edit-segments: '[[start1,end1],[start2,end2],...]'
        Usa -1 para inicio/fin del audio: '[[-1,25],[50.0,-1]]'

        Ejemplo:
          edit --lrc edit.lrc --ref-song obra.mp3 --edit-segments '[[10.0,20.0]]'
        """))
    pe.add_argument("--lrc", required=True, metavar="FILE")
    pe.add_argument("--ref-song", required=True, metavar="FILE",
                    help="canción original a editar")
    pe.add_argument("--edit-segments", required=True, metavar="JSON",
                    help='"[[start,end],...]" (-1 = inicio/fin del audio)')
    ge = pe.add_mutually_exclusive_group(required=False)
    ge.add_argument("--ref-prompt", default=None, metavar="TEXT")
    ge.add_argument("--ref-audio", default=None, metavar="FILE")
    pe.add_argument("--variant", choices=["base", "full"], default="base")
    pe.add_argument("--audio-length", type=int, default=95, metavar="SEC")
    pe.add_argument("--output", default="results/edit.wav", metavar="FILE")
    pe.add_argument("--chunked", action="store_true")
    pe.add_argument("--batch-infer-num", type=int, default=1, metavar="N")
    pe.add_argument("--steps", type=int, default=32)
    pe.add_argument("--cfg", type=float, default=4.0)
    pe.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    pe.set_defaults(func=cmd_edit)

    # round-trip
    pr = sub.add_parser("round-trip", help="VAE encode→decode (diagnóstico del codec)")
    pr.add_argument("--input", required=True, metavar="FILE")
    pr.add_argument("--output", default="results/roundtrip.wav", metavar="FILE")
    pr.add_argument("--chunked", action="store_true")
    pr.set_defaults(func=cmd_round_trip)

    return p


def main(argv=None):
    _setup_paths()
    _auto_espeak_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
