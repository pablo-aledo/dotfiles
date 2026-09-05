#!/usr/bin/env python3
# ==============================================================================
#  ddsp_synth.py
# ------------------------------------------------------------------------------
#  Self-contained port of Google Magenta's DDSP (Differentiable Digital Signal
#  Processing) library: harmonic + filtered-noise synthesis, a learnable
#  reverb, a full trainable autoencoder (encoder -> latent z -> decoder ->
#  synths -> reverb), and the CLI glue to drive all of it from one file.
#
#  USAGE
#  -----
#    ddsp_synth.py synth-harmonic  --f0 220 --duration 2.0 --out tone.wav
#    ddsp_synth.py synth-noise     --duration 2.0 --out noise.wav
#    ddsp_synth.py reverb          --in tone.wav --out tone_verb.wav
#    ddsp_synth.py analyze         --in voice.wav --out voice_features.npz
#    ddsp_synth.py train           --data ./corpus_dir --out model.pt --epochs 200
#    ddsp_synth.py resynth         --model model.pt --in voice.wav --out recon.wav
#    ddsp_synth.py infer           --model model.pt --f0 440 --duration 3.0 --out gen.wav
#    ddsp_synth.py selftest        --tmp /tmp/ddsp_selftest
#
#  Every subcommand works on plain WAV files (mono, resampled internally to
#  --sample-rate, default 16000 Hz, matching the original DDSP examples).
#
#  DEPENDENCIES
#  ------------
#    Baseline (always required): numpy, scipy, soundfile
#    Only for train/resynth/infer/selftest: torch  (imported lazily)
#
#  DESIGN NOTES (differences from the upstream TensorFlow/Gin codebase)
#  ----------------------------------------------------------------------------
#   * Harmonic synth: sample-wise phase accumulation (cumsum of angular
#     frequency), identical algorithm to core.oscillator_bank, reimplemented
#     in torch/numpy.
#   * Filtered-noise synth: reimplemented as frequency-domain filtering of
#     white noise via STFT/ISTFT (multiply each noise STFT frame by the
#     target magnitude envelope, upsampled to match frame count/bin count),
#     rather than the frame-wise FIR overlap-add convolution used upstream.
#     Same idea (time-varying spectral shaping of noise), simpler code.
#   * Reverb: FFT convolution with either a loaded/synthesized impulse
#     response (DSP-only `reverb` subcommand) or a learnable IR parameter
#     trained jointly with the autoencoder.
#   * f0/loudness extraction: lightweight autocorrelation pitch tracker +
#     A-weighted log-power loudness, no CREPE/external pretrained model
#     dependency (kept self-contained).
#   * Multi-scale spectral loss: L1 on magnitude + log-magnitude STFTs at
#     several FFT sizes, largest scale defaults to 4096 (the window size
#     that fixed spectral leakage in spectro_midi.py/audio_lab.py).
# ==============================================================================

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------------------
# Lazy imports
# ------------------------------------------------------------------------------

def _sf():
    import soundfile as sf
    return sf


def _torch():
    import torch
    return torch


def _torch_nn():
    import torch.nn as nn
    return nn


def _torch_f():
    import torch.nn.functional as F
    return F


# ------------------------------------------------------------------------------
# Constants / defaults
# ------------------------------------------------------------------------------

DEFAULT_SR = 16000
DEFAULT_N_HARMONICS = 60
DEFAULT_N_NOISE_BANDS = 65
DEFAULT_NOISE_WINDOW = 257          # frequency bins target for filtered noise (odd)
DEFAULT_Z_DIM = 16
DEFAULT_HIDDEN = 256
DEFAULT_FRAME_HOP = 64              # control-rate hop in samples (250 Hz @ 16kHz)
DEFAULT_FRAME_SIZE = 1024
DEFAULT_SEGMENT_SECONDS = 4.0
SPECTRAL_LOSS_FFT_SIZES = (4096, 2048, 1024, 512, 256, 128, 64)


# ==============================================================================
# I/O helpers
# ==============================================================================

def load_wav(path, sample_rate=DEFAULT_SR):
    if not Path(path).is_file():
        print(f"error: input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    sf = _sf()
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != sample_rate:
        audio = resample_numpy(audio, sr, sample_rate)
    return audio.astype(np.float32)


def save_wav(path, audio, sample_rate=DEFAULT_SR):
    sf = _sf()
    audio = np.asarray(audio, dtype=np.float32)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.98
    sf.write(str(path), audio, sample_rate)


def resample_numpy(audio, sr_in, sr_out):
    if sr_in == sr_out:
        return audio
    n_out = int(round(len(audio) * sr_out / sr_in))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ==============================================================================
# Core scalar / control-rate math (numpy, used by both DSP-only CLI paths and
# feature extraction; torch equivalents defined further down for the model)
# ==============================================================================

def exp_sigmoid_np(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    return max_value * (1.0 / (1.0 + np.exp(-x))) ** math.log(exponent) + threshold


def hz_to_midi(f0_hz):
    f0_hz = np.maximum(f0_hz, 1e-7)
    return 12.0 * (np.log2(f0_hz) - np.log2(440.0)) + 69.0


def upsample_frames_np(frames, n_samples):
    """Linearly upsample a [n_frames] or [n_frames, C] control signal to
    [n_samples] / [n_samples, C]."""
    frames = np.asarray(frames, dtype=np.float32)
    n_frames = frames.shape[0]
    if n_frames == n_samples:
        return frames
    x_old = np.linspace(0.0, 1.0, num=n_frames, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=n_samples, endpoint=True)
    if frames.ndim == 1:
        return np.interp(x_new, x_old, frames).astype(np.float32)
    out = np.stack(
        [np.interp(x_new, x_old, frames[:, c]) for c in range(frames.shape[1])],
        axis=-1,
    )
    return out.astype(np.float32)


# ==============================================================================
# DSP-only synths (numpy) — used directly by the synth-harmonic / synth-noise /
# reverb subcommands, no torch / gradients involved.
# ==============================================================================

def harmonic_synth_np(f0_hz, amplitudes, harmonic_distribution, n_samples,
                       sample_rate=DEFAULT_SR):
    """f0_hz, amplitudes: [n_frames]; harmonic_distribution: [n_frames, n_harm].
    Returns audio [n_samples]."""
    n_harm = harmonic_distribution.shape[-1]
    f0_env = upsample_frames_np(f0_hz, n_samples)
    amp_env = upsample_frames_np(amplitudes, n_samples)
    hd_env = upsample_frames_np(harmonic_distribution, n_samples)  # [n_samples, n_harm]

    harmonic_numbers = np.arange(1, n_harm + 1, dtype=np.float32)
    freq_envelopes = f0_env[:, None] * harmonic_numbers[None, :]   # [n_samples, n_harm]
    harm_amp_envelopes = hd_env * amp_env[:, None]

    # Zero out anything above Nyquist.
    harm_amp_envelopes = np.where(
        freq_envelopes >= sample_rate / 2.0, 0.0, harm_amp_envelopes
    )

    omegas = freq_envelopes * (2.0 * np.pi) / sample_rate
    phases = np.cumsum(omegas, axis=0)
    audio = np.sum(harm_amp_envelopes * np.sin(phases), axis=-1)
    return audio.astype(np.float32)


def filtered_noise_synth_np(magnitudes, n_samples, sample_rate=DEFAULT_SR,
                             n_fft=1024):
    """magnitudes: [n_frames, n_bands], strictly positive. Returns audio
    [n_samples] by spectrally shaping white noise frame-by-frame."""
    rng = np.random.default_rng()
    noise = rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)

    hop = n_fft // 4
    window = np.hanning(n_fft).astype(np.float32)
    n_bins = n_fft // 2 + 1

    # Upsample the (coarse) band magnitudes to n_bins frequency bins.
    n_frames_in = magnitudes.shape[0]
    band_x = np.linspace(0.0, 1.0, num=magnitudes.shape[1], endpoint=True)
    bin_x = np.linspace(0.0, 1.0, num=n_bins, endpoint=True)
    mag_bins = np.stack(
        [np.interp(bin_x, band_x, magnitudes[f]) for f in range(n_frames_in)],
        axis=0,
    )  # [n_frames_in, n_bins]

    n_stft_frames = 1 + (n_samples - n_fft) // hop
    n_stft_frames = max(n_stft_frames, 1)
    frame_x = np.linspace(0.0, 1.0, num=n_frames_in, endpoint=True)
    frame_x_new = np.linspace(0.0, 1.0, num=n_stft_frames, endpoint=True)
    mag_stft = np.stack(
        [np.interp(frame_x_new, frame_x, mag_bins[:, b]) for b in range(n_bins)],
        axis=-1,
    )  # [n_stft_frames, n_bins]

    out = np.zeros(n_samples, dtype=np.float32)
    win_sum = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_stft_frames):
        start = i * hop
        end = start + n_fft
        if end > n_samples:
            chunk = np.zeros(n_fft, dtype=np.float32)
            chunk[: n_samples - start] = noise[start:n_samples]
        else:
            chunk = noise[start:end]
        spec = np.fft.rfft(chunk * window)
        spec *= mag_stft[i]
        filtered = np.fft.irfft(spec, n=n_fft).astype(np.float32) * window
        seg_end = min(end, n_samples)
        out[start:seg_end] += filtered[: seg_end - start]
        win_sum[start:seg_end] += window[: seg_end - start] ** 2
    win_sum = np.maximum(win_sum, 1e-8)
    return (out / win_sum * window.sum() / n_fft * 4).astype(np.float32)


def synth_reverb_ir(duration_s, decay, sample_rate=DEFAULT_SR):
    n = int(duration_s * sample_rate)
    rng = np.random.default_rng()
    noise = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    t = np.arange(n) / sample_rate
    env = np.exp(-decay * t).astype(np.float32)
    ir = noise * env
    ir[0] = 1.0  # direct path
    return ir


def apply_reverb_np(audio, ir, wet=0.5):
    wet_signal = np.convolve(audio, ir, mode="full")[: len(audio)]
    peak = np.max(np.abs(wet_signal)) + 1e-8
    wet_signal = wet_signal / peak * (np.max(np.abs(audio)) + 1e-8)
    return ((1 - wet) * audio + wet * wet_signal).astype(np.float32)


# ==============================================================================
# Feature extraction: f0 (autocorrelation) + A-weighted loudness
# ==============================================================================

def _a_weighting_db(freqs_hz):
    f = np.maximum(freqs_hz, 1e-6)
    f2 = f ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / (
        (f2 + 20.6 ** 2)
        * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
        * (f2 + 12194.0 ** 2)
    )
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(ra, 1e-12)) + 2.0
    return db


def extract_loudness(audio, sample_rate=DEFAULT_SR, frame_size=DEFAULT_FRAME_SIZE,
                      hop=DEFAULT_FRAME_HOP):
    n_samples = len(audio)
    n_frames = max(1, (n_samples - frame_size) // hop + 1)
    window = np.hanning(frame_size).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    weight_db = _a_weighting_db(freqs)
    loudness = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        chunk = audio[s : s + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        spec = np.fft.rfft(chunk * window)
        power_db = 10.0 * np.log10(np.maximum(np.abs(spec) ** 2, 1e-10))
        weighted = power_db + weight_db
        loudness[i] = 10.0 * np.log10(np.mean(10.0 ** (weighted / 10.0)) + 1e-10)
    return loudness


def extract_f0(audio, sample_rate=DEFAULT_SR, frame_size=DEFAULT_FRAME_SIZE,
                hop=DEFAULT_FRAME_HOP, fmin=50.0, fmax=2000.0,
                voicing_threshold=0.35):
    """Simple per-frame autocorrelation pitch tracker. Frames whose
    normalized autocorrelation peak (peak / zero-lag energy) falls below
    voicing_threshold are treated as unvoiced (f0=0) — without this, noisy
    or unvoiced frames report a spurious pitch from whatever weak periodicity
    the autocorrelation happens to find."""
    n_samples = len(audio)
    n_frames = max(1, (n_samples - frame_size) // hop + 1)
    window = np.hanning(frame_size).astype(np.float32)
    lag_min = int(sample_rate / fmax)
    lag_max = min(int(sample_rate / fmin), frame_size - 1)
    f0 = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        chunk = audio[s : s + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        chunk = (chunk - chunk.mean()) * window
        if np.max(np.abs(chunk)) < 1e-6:
            f0[i] = 0.0
            continue
        corr = np.correlate(chunk, chunk, mode="full")[frame_size - 1 :]
        corr = corr[: lag_max + 1]
        if lag_max <= lag_min or len(corr) <= lag_min:
            f0[i] = 0.0
            continue
        segment = corr[lag_min:lag_max]
        if segment.size == 0 or np.max(segment) <= 0:
            f0[i] = 0.0
            continue
        zero_lag_energy = corr[0] if corr[0] > 0 else 1e-8
        if np.max(segment) / zero_lag_energy < voicing_threshold:
            f0[i] = 0.0
            continue
        peak_lag = lag_min + int(np.argmax(segment))
        # Parabolic interpolation for sub-sample precision.
        if 0 < peak_lag < len(corr) - 1:
            a, b, c = corr[peak_lag - 1], corr[peak_lag], corr[peak_lag + 1]
            denom = a - 2 * b + c
            shift = 0.5 * (a - c) / denom if abs(denom) > 1e-8 else 0.0
            peak_lag = peak_lag + float(np.clip(shift, -1, 1))
        f0[i] = sample_rate / peak_lag if peak_lag > 0 else 0.0
    return f0


def extract_mel(audio, sample_rate=DEFAULT_SR, frame_size=DEFAULT_FRAME_SIZE,
                 hop=DEFAULT_FRAME_HOP, n_mels=40):
    """Lightweight log-mel-ish spectrogram (triangular filterbank on the
    linear power spectrum) used purely as the z-encoder's input features."""
    n_samples = len(audio)
    n_frames = max(1, (n_samples - frame_size) // hop + 1)
    window = np.hanning(frame_size).astype(np.float32)
    n_bins = frame_size // 2 + 1
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min, mel_max = hz_to_mel(20.0), hz_to_mel(sample_rate / 2.0)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_bins - 1) * hz_pts / (sample_rate / 2.0)).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_bins - 1)

    fbank = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, min(center, n_bins)):
            fbank[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, min(right, n_bins)):
            fbank[m - 1, k] = (right - k) / max(right - center, 1)

    mel_frames = np.zeros((n_frames, n_mels), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        chunk = audio[s : s + frame_size]
        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))
        power = np.abs(np.fft.rfft(chunk * window)) ** 2
        mel_energy = fbank @ power
        mel_frames[i] = np.log(mel_energy + 1e-6)
    return mel_frames.astype(np.float32)


# ==============================================================================
# Torch (differentiable) versions of the same primitives, for the model
# ==============================================================================

def exp_sigmoid_torch(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    torch = _torch()
    return max_value * torch.sigmoid(x) ** math.log(exponent) + threshold


def upsample_frames_torch(x, n_samples):
    """x: [batch, n_frames, C] -> [batch, n_samples, C] via linear interp."""
    F = _torch_f()
    x = x.transpose(1, 2)  # [batch, C, n_frames]
    x = F.interpolate(x, size=n_samples, mode="linear", align_corners=True)
    return x.transpose(1, 2)  # [batch, n_samples, C]


def harmonic_synth_torch(f0_hz, amplitudes, harmonic_distribution, n_samples,
                          sample_rate=DEFAULT_SR):
    """f0_hz, amplitudes: [batch, n_frames, 1]; harmonic_distribution:
    [batch, n_frames, n_harm]. Returns [batch, n_samples]."""
    torch = _torch()
    n_harm = harmonic_distribution.shape[-1]
    f0_env = upsample_frames_torch(f0_hz, n_samples)             # [b, n, 1]
    amp_env = upsample_frames_torch(amplitudes, n_samples)       # [b, n, 1]
    hd_env = upsample_frames_torch(harmonic_distribution, n_samples)  # [b, n, H]

    harmonic_numbers = torch.arange(
        1, n_harm + 1, dtype=f0_env.dtype, device=f0_env.device
    )
    freq_envelopes = f0_env * harmonic_numbers[None, None, :]    # [b, n, H]
    harm_amp_envelopes = hd_env * amp_env

    mask = (freq_envelopes < sample_rate / 2.0).to(f0_env.dtype)
    harm_amp_envelopes = harm_amp_envelopes * mask

    omegas = freq_envelopes * (2.0 * math.pi) / sample_rate
    phases = torch.cumsum(omegas, dim=1)
    audio = torch.sum(harm_amp_envelopes * torch.sin(phases), dim=-1)
    return audio


def _interp_last_axis(x, size_out):
    """Linearly interpolate the last axis of a [b, n_frames, C_in] tensor to
    [b, n_frames, size_out]."""
    F = _torch_f()
    b, n_frames, c_in = x.shape
    x = x.reshape(b * n_frames, 1, c_in)  # treat as [N, channels=1, length=c_in]
    x = F.interpolate(x, size=size_out, mode="linear", align_corners=True)
    return x.reshape(b, n_frames, size_out)


def _resize_frames(x, n_frames_out):
    """[b, n_frames_in, C] -> [b, n_frames_out, C]."""
    return upsample_frames_torch(x, n_frames_out)


def filtered_noise_synth_torch(magnitudes, n_samples, n_fft=1024):
    """magnitudes: [batch, n_frames, n_bands] -> [batch, n_samples]. Shapes
    white noise in the frequency domain using the (upsampled) magnitude
    envelope, via STFT / ISTFT."""
    torch = _torch()
    device = magnitudes.device
    batch = magnitudes.shape[0]
    n_bins = n_fft // 2 + 1
    hop = n_fft // 4

    mag_bins = _interp_last_axis(magnitudes, n_bins)  # [b, n_frames, n_bins]

    noise = torch.rand(batch, n_samples, device=device) * 2 - 1
    window = torch.hann_window(n_fft, device=device)
    stft = torch.stft(
        noise, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
        return_complex=True, center=True,
    )  # [b, n_bins, T]
    n_stft_frames = stft.shape[-1]
    mag_t = _resize_frames(mag_bins, n_stft_frames).transpose(1, 2)  # [b, n_bins, T]
    shaped = stft * mag_t
    audio = torch.istft(
        shaped, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window,
        length=n_samples, center=True,
    )
    return audio


def make_reverb_ir_torch(ir_length, decay, device):
    torch = _torch()
    t = torch.arange(ir_length, device=device, dtype=torch.float32)
    env = torch.exp(-decay * t / DEFAULT_SR)
    noise = torch.rand(ir_length, device=device) * 2 - 1
    ir = noise * env
    return ir


def apply_reverb_torch(audio, ir, wet=1.0):
    """audio: [batch, n_samples], ir: [ir_len] (shared) or [batch, ir_len].
    FFT convolution, wet-mixed with the dry signal."""
    torch = _torch()
    batch, n = audio.shape
    if ir.dim() == 1:
        ir = ir.unsqueeze(0).expand(batch, -1)
    ir_len = ir.shape[-1]
    n_fft = 1
    while n_fft < n + ir_len - 1:
        n_fft *= 2
    A = torch.fft.rfft(audio, n=n_fft)
    H = torch.fft.rfft(ir, n=n_fft)
    wet_signal = torch.fft.irfft(A * H, n=n_fft)[:, :n]
    dry_peak = audio.abs().amax(dim=-1, keepdim=True) + 1e-8
    wet_peak = wet_signal.abs().amax(dim=-1, keepdim=True) + 1e-8
    wet_signal = wet_signal / wet_peak * dry_peak
    return (1 - wet) * audio + wet * wet_signal


def multiscale_spectral_loss(audio, target, fft_sizes=SPECTRAL_LOSS_FFT_SIZES):
    torch = _torch()
    total = torch.zeros((), device=audio.device)
    n_terms = 0
    for n_fft in fft_sizes:
        if n_fft > audio.shape[-1]:
            continue
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=audio.device)
        a_stft = torch.stft(audio, n_fft=n_fft, hop_length=hop, window=window,
                             return_complex=True, center=True)
        t_stft = torch.stft(target, n_fft=n_fft, hop_length=hop, window=window,
                             return_complex=True, center=True)
        a_mag = a_stft.abs()
        t_mag = t_stft.abs()
        lin_loss = (a_mag - t_mag).abs().mean()
        log_loss = (torch.log(a_mag + 1e-5) - torch.log(t_mag + 1e-5)).abs().mean()
        total = total + lin_loss + log_loss
        n_terms += 1
    return total / max(n_terms, 1)


# ==============================================================================
# Model: encoder (mel -> z) + decoder (f0, loudness, z -> synth controls)
# ==============================================================================

def _build_model_classes():
    """Defined lazily so importing this file never requires torch unless a
    torch-backed subcommand is actually invoked."""
    torch = _torch()
    nn = _torch_nn()

    class ZEncoder(nn.Module):
        def __init__(self, n_mels, z_dim, hidden):
            super().__init__()
            self.gru = nn.GRU(n_mels, hidden, batch_first=True)
            self.proj = nn.Linear(hidden, z_dim)

        def forward(self, mel):  # [b, n_frames, n_mels]
            out, _ = self.gru(mel)
            return self.proj(out)  # [b, n_frames, z_dim]

    class Decoder(nn.Module):
        def __init__(self, z_dim, hidden, n_harmonics, n_noise_bands):
            super().__init__()

            def mlp(in_dim):
                return nn.Sequential(
                    nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2), nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.LayerNorm(hidden),
                )

            self.mlp_f0 = mlp(1)
            self.mlp_loud = mlp(1)
            self.mlp_z = mlp(z_dim)
            self.gru = nn.GRU(hidden * 3, hidden, batch_first=True)
            self.out_mlp = mlp(hidden)
            self.amp_head = nn.Linear(hidden, 1)
            self.hd_head = nn.Linear(hidden, n_harmonics)
            self.noise_head = nn.Linear(hidden, n_noise_bands)

        def forward(self, f0_scaled, loud_scaled, z):
            hf0 = self.mlp_f0(f0_scaled)
            hloud = self.mlp_loud(loud_scaled)
            hz = self.mlp_z(z)
            h = torch.cat([hf0, hloud, hz], dim=-1)
            h, _ = self.gru(h)
            h = self.out_mlp(h)
            amplitudes = self.amp_head(h)
            harmonic_distribution = self.hd_head(h)
            noise_magnitudes = self.noise_head(h)
            return amplitudes, harmonic_distribution, noise_magnitudes

    class DDSPAutoencoder(nn.Module):
        def __init__(self, sample_rate=DEFAULT_SR, n_harmonics=DEFAULT_N_HARMONICS,
                     n_noise_bands=DEFAULT_N_NOISE_BANDS, z_dim=DEFAULT_Z_DIM,
                     hidden=DEFAULT_HIDDEN, n_mels=40, noise_n_fft=1024,
                     reverb_length=8000, reverb_decay=8.0):
            super().__init__()
            self.sample_rate = sample_rate
            self.n_harmonics = n_harmonics
            self.n_noise_bands = n_noise_bands
            self.z_dim = z_dim
            self.n_mels = n_mels
            self.noise_n_fft = noise_n_fft
            self.reverb_length = reverb_length
            self.reverb_decay = reverb_decay

            self.encoder = ZEncoder(n_mels, z_dim, hidden)
            self.decoder = Decoder(z_dim, hidden, n_harmonics, n_noise_bands)
            init_ir = make_reverb_ir_torch(reverb_length, reverb_decay, "cpu")
            self.reverb_ir = nn.Parameter(init_ir)

        def decode_and_synthesize(self, f0_hz, loudness_db, z, n_samples):
            f0_scaled = hz_to_midi_torch(f0_hz) / 127.0
            loud_scaled = (loudness_db + 80.0) / 80.0
            amplitudes, harmonic_distribution, noise_magnitudes = self.decoder(
                f0_scaled, loud_scaled, z
            )
            amplitudes = exp_sigmoid_torch(amplitudes)
            harmonic_distribution = exp_sigmoid_torch(harmonic_distribution)
            harmonic_distribution = harmonic_distribution / (
                harmonic_distribution.sum(dim=-1, keepdim=True) + 1e-7
            )
            noise_magnitudes = exp_sigmoid_torch(noise_magnitudes - 5.0)

            harmonic_audio = harmonic_synth_torch(
                f0_hz, amplitudes, harmonic_distribution, n_samples, self.sample_rate
            )
            noise_audio = filtered_noise_synth_torch(
                noise_magnitudes, n_samples, n_fft=self.noise_n_fft
            )
            dry = harmonic_audio + noise_audio
            wet = apply_reverb_torch(dry, self.reverb_ir, wet=1.0)
            return wet, dict(amplitudes=amplitudes,
                              harmonic_distribution=harmonic_distribution,
                              noise_magnitudes=noise_magnitudes)

        def forward(self, mel, f0_hz, loudness_db, n_samples):
            z = self.encoder(mel)
            z = _resize_frames(z, f0_hz.shape[1])
            audio, controls = self.decode_and_synthesize(f0_hz, loudness_db, z, n_samples)
            return audio, controls

    return ZEncoder, Decoder, DDSPAutoencoder


def hz_to_midi_torch(f0_hz):
    torch = _torch()
    f0_hz = torch.clamp(f0_hz, min=1e-7)
    return 12.0 * (torch.log2(f0_hz) - math.log2(440.0)) + 69.0


# ==============================================================================
# Dataset preparation (numpy features -> torch tensors)
# ==============================================================================

def prepare_example(audio, sample_rate, segment_samples, frame_size, hop, n_mels):
    f0 = extract_f0(audio, sample_rate, frame_size, hop)
    loudness = extract_loudness(audio, sample_rate, frame_size, hop)
    mel = extract_mel(audio, sample_rate, frame_size, hop, n_mels)
    n_frames = min(len(f0), len(loudness), len(mel))
    return (
        audio[:segment_samples],
        f0[:n_frames],
        loudness[:n_frames],
        mel[:n_frames],
    )


def chunk_audio(audio, segment_samples):
    n_full = len(audio) // segment_samples
    chunks = []
    for i in range(max(n_full, 0)):
        chunks.append(audio[i * segment_samples : (i + 1) * segment_samples])
    if not chunks and len(audio) > 0:
        pad = np.zeros(segment_samples, dtype=np.float32)
        pad[: len(audio)] = audio
        chunks.append(pad)
    return chunks


# ==============================================================================
# Subcommands
# ==============================================================================

def cmd_synth_harmonic(args):
    n_samples = int(args.duration * args.sample_rate)
    n_frames = max(4, int(args.duration * 50))
    f0 = np.full(n_frames, args.f0, dtype=np.float32)
    if args.vibrato > 0:
        t = np.linspace(0, args.duration, n_frames)
        f0 = f0 * (1.0 + args.vibrato * np.sin(2 * np.pi * 5.0 * t))
    amplitudes = np.full(n_frames, args.amplitude, dtype=np.float32)
    n_harm = args.n_harmonics
    harmonic_distribution = np.array(
        [1.0 / (h ** 1.2) for h in range(1, n_harm + 1)], dtype=np.float32
    )
    harmonic_distribution = np.tile(harmonic_distribution, (n_frames, 1))
    harmonic_distribution /= harmonic_distribution.sum(axis=-1, keepdims=True)

    audio = harmonic_synth_np(f0, amplitudes, harmonic_distribution, n_samples,
                               args.sample_rate)
    save_wav(args.out, audio, args.sample_rate)
    print(f"synth-harmonic: wrote {args.out} ({args.duration:.2f}s @ {args.sample_rate}Hz, "
          f"f0={args.f0}Hz, {n_harm} harmonics)")


def cmd_synth_noise(args):
    n_samples = int(args.duration * args.sample_rate)
    n_frames = max(4, int(args.duration * 50))
    n_bands = args.n_bands
    if args.shape == "white":
        magnitudes = np.ones((n_frames, n_bands), dtype=np.float32)
    elif args.shape == "lowpass":
        magnitudes = np.tile(
            np.linspace(1.0, 0.02, n_bands, dtype=np.float32), (n_frames, 1)
        )
    elif args.shape == "highpass":
        magnitudes = np.tile(
            np.linspace(0.02, 1.0, n_bands, dtype=np.float32), (n_frames, 1)
        )
    else:
        raise ValueError(f"unknown shape {args.shape}")
    audio = filtered_noise_synth_np(magnitudes, n_samples, args.sample_rate)
    save_wav(args.out, audio, args.sample_rate)
    print(f"synth-noise: wrote {args.out} ({args.duration:.2f}s, shape={args.shape})")


def cmd_reverb(args):
    audio = load_wav(args.inp, args.sample_rate)
    if args.ir:
        ir = load_wav(args.ir, args.sample_rate)
    else:
        ir = synth_reverb_ir(args.ir_duration, args.decay, args.sample_rate)
    out = apply_reverb_np(audio, ir, wet=args.wet)
    save_wav(args.out, out, args.sample_rate)
    print(f"reverb: wrote {args.out} (wet={args.wet}, decay={args.decay})")


def cmd_analyze(args):
    audio = load_wav(args.inp, args.sample_rate)
    f0 = extract_f0(audio, args.sample_rate, args.frame_size, args.hop)
    loudness = extract_loudness(audio, args.sample_rate, args.frame_size, args.hop)
    np.savez(args.out, f0=f0, loudness=loudness, sample_rate=args.sample_rate,
             frame_size=args.frame_size, hop=args.hop)
    voiced = f0 > 0
    f0_mean = float(f0[voiced].mean()) if voiced.any() else 0.0
    print(f"analyze: wrote {args.out} ({len(f0)} frames, mean f0={f0_mean:.1f}Hz, "
          f"loudness range=[{loudness.min():.1f}, {loudness.max():.1f}]dB)")


def cmd_train(args):
    torch = _torch()
    _, _, DDSPAutoencoder = _build_model_classes()

    data_dir = Path(args.data)
    wav_paths = sorted(list(data_dir.glob("*.wav")))
    if not wav_paths:
        print(f"train: no .wav files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    segment_samples = int(DEFAULT_SEGMENT_SECONDS * args.sample_rate)
    examples = []
    for p in wav_paths:
        audio = load_wav(p, args.sample_rate)
        for chunk in chunk_audio(audio, segment_samples):
            examples.append(chunk)
    print(f"train: {len(wav_paths)} files -> {len(examples)} segments of "
          f"{DEFAULT_SEGMENT_SECONDS}s")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    model = DDSPAutoencoder(sample_rate=args.sample_rate,
                             n_harmonics=args.n_harmonics,
                             n_noise_bands=args.n_noise_bands,
                             z_dim=args.z_dim, hidden=args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    features = []
    for audio in examples:
        f0 = extract_f0(audio, args.sample_rate, DEFAULT_FRAME_SIZE, DEFAULT_FRAME_HOP)
        loud = extract_loudness(audio, args.sample_rate, DEFAULT_FRAME_SIZE, DEFAULT_FRAME_HOP)
        mel = extract_mel(audio, args.sample_rate, DEFAULT_FRAME_SIZE, DEFAULT_FRAME_HOP, 40)
        n_frames = min(len(f0), len(loud), len(mel))
        features.append((audio, f0[:n_frames], loud[:n_frames], mel[:n_frames]))

    losses = []
    for epoch in range(args.epochs):
        np.random.shuffle(features)
        epoch_loss = 0.0
        for i in range(0, len(features), args.batch_size):
            batch = features[i : i + args.batch_size]
            if not batch:
                continue
            audio_b = torch.tensor(np.stack([b[0] for b in batch]), device=device)
            f0_b = torch.tensor(np.stack([b[1] for b in batch]), device=device).unsqueeze(-1)
            loud_b = torch.tensor(np.stack([b[2] for b in batch]), device=device).unsqueeze(-1)
            mel_b = torch.tensor(np.stack([b[3] for b in batch]), device=device)

            opt.zero_grad()
            recon, _ = model(mel_b, f0_b, loud_b, segment_samples)
            loss = multiscale_spectral_loss(recon, audio_b)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
        epoch_loss /= max(1, math.ceil(len(features) / args.batch_size))
        losses.append(epoch_loss)
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}/{args.epochs}  spectral_loss={epoch_loss:.4f}")

    hparams = dict(sample_rate=args.sample_rate, n_harmonics=args.n_harmonics,
                    n_noise_bands=args.n_noise_bands, z_dim=args.z_dim,
                    hidden=args.hidden, n_mels=40, segment_samples=segment_samples,
                    frame_size=DEFAULT_FRAME_SIZE, hop=DEFAULT_FRAME_HOP)
    torch.save({"state_dict": model.state_dict(), "hparams": hparams}, args.out)
    with open(str(args.out) + ".json", "w") as f:
        json.dump({"hparams": hparams, "loss_curve": losses}, f, indent=2)
    print(f"train: wrote checkpoint {args.out} (final loss={losses[-1]:.4f})")
    return losses


def _load_model(model_path):
    torch = _torch()
    _, _, DDSPAutoencoder = _build_model_classes()
    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    h = ckpt["hparams"]
    model = DDSPAutoencoder(sample_rate=h["sample_rate"], n_harmonics=h["n_harmonics"],
                             n_noise_bands=h["n_noise_bands"], z_dim=h["z_dim"],
                             hidden=h["hidden"], n_mels=h.get("n_mels", 40))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, h


def cmd_resynth(args):
    torch = _torch()
    model, h = _load_model(args.model)
    audio = load_wav(args.inp, h["sample_rate"])
    seg = h["segment_samples"]
    chunks = chunk_audio(audio, seg)
    out_chunks = []
    with torch.no_grad():
        for chunk in chunks:
            f0 = extract_f0(chunk, h["sample_rate"], h["frame_size"], h["hop"])
            loud = extract_loudness(chunk, h["sample_rate"], h["frame_size"], h["hop"])
            mel = extract_mel(chunk, h["sample_rate"], h["frame_size"], h["hop"], h.get("n_mels", 40))
            n_frames = min(len(f0), len(loud), len(mel))
            f0_t = torch.tensor(f0[:n_frames]).float().unsqueeze(0).unsqueeze(-1)
            loud_t = torch.tensor(loud[:n_frames]).float().unsqueeze(0).unsqueeze(-1)
            mel_t = torch.tensor(mel[:n_frames]).float().unsqueeze(0)
            recon, _ = model(mel_t, f0_t, loud_t, seg)
            out_chunks.append(recon.squeeze(0).numpy())
    out = np.concatenate(out_chunks)[: len(audio)]
    save_wav(args.out, out, h["sample_rate"])
    print(f"resynth: wrote {args.out} ({len(chunks)} segment(s))")


def cmd_infer(args):
    torch = _torch()
    model, h = _load_model(args.model)
    n_samples = int(args.duration * h["sample_rate"])
    n_frames = max(4, int(n_samples / h["hop"]))
    f0 = np.full(n_frames, args.f0, dtype=np.float32)
    loud = np.full(n_frames, args.loudness, dtype=np.float32)
    z_dim = h["z_dim"]
    z = np.zeros((n_frames, z_dim), dtype=np.float32)
    if args.z_file:
        z_loaded = np.load(args.z_file)["z"]
        z = upsample_frames_np(z_loaded, n_frames) if z_loaded.shape[0] != n_frames else z_loaded

    with torch.no_grad():
        f0_t = torch.tensor(f0).float().unsqueeze(0).unsqueeze(-1)
        loud_t = torch.tensor(loud).float().unsqueeze(0).unsqueeze(-1)
        z_t = torch.tensor(z).float().unsqueeze(0)
        audio, _ = model.decode_and_synthesize(f0_t, loud_t, z_t, n_samples)
    save_wav(args.out, audio.squeeze(0).numpy(), h["sample_rate"])
    print(f"infer: wrote {args.out} ({args.duration:.2f}s, f0={args.f0}Hz, "
          f"loudness={args.loudness}dB)")


def cmd_selftest(args):
    _run_selftest(Path(args.tmp))


# ==============================================================================
# CLI
# ==============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        prog="ddsp_synth.py",
        description="Self-contained DDSP-style synthesis + trainable autoencoder.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("synth-harmonic", help="Synthesize a harmonic tone.")
    sp.add_argument("--f0", type=float, default=220.0)
    sp.add_argument("--amplitude", type=float, default=0.5)
    sp.add_argument("--duration", type=float, default=2.0)
    sp.add_argument("--n-harmonics", type=int, default=DEFAULT_N_HARMONICS)
    sp.add_argument("--vibrato", type=float, default=0.0, help="relative depth, e.g. 0.01")
    sp.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_synth_harmonic)

    sp = sub.add_parser("synth-noise", help="Synthesize filtered noise.")
    sp.add_argument("--duration", type=float, default=2.0)
    sp.add_argument("--n-bands", type=int, default=DEFAULT_N_NOISE_BANDS)
    sp.add_argument("--shape", choices=["white", "lowpass", "highpass"], default="white")
    sp.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_synth_noise)

    sp = sub.add_parser("reverb", help="Apply reverb to a WAV file.")
    sp.add_argument("--in", dest="inp", required=True)
    sp.add_argument("--ir", default=None, help="Optional WAV impulse response.")
    sp.add_argument("--ir-duration", type=float, default=2.0)
    sp.add_argument("--decay", type=float, default=4.0)
    sp.add_argument("--wet", type=float, default=0.4)
    sp.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_reverb)

    sp = sub.add_parser("analyze", help="Extract f0 + loudness from a WAV.")
    sp.add_argument("--in", dest="inp", required=True)
    sp.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE)
    sp.add_argument("--hop", type=int, default=DEFAULT_FRAME_HOP)
    sp.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_analyze)

    sp = sub.add_parser("train", help="Train the DDSP autoencoder on a WAV corpus.")
    sp.add_argument("--data", required=True, help="Directory of .wav files.")
    sp.add_argument("--epochs", type=int, default=200)
    sp.add_argument("--batch-size", type=int, default=4)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--n-harmonics", type=int, default=DEFAULT_N_HARMONICS)
    sp.add_argument("--n-noise-bands", type=int, default=DEFAULT_N_NOISE_BANDS)
    sp.add_argument("--z-dim", type=int, default=DEFAULT_Z_DIM)
    sp.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    sp.add_argument("--sample-rate", type=int, default=DEFAULT_SR)
    sp.add_argument("--device", default="auto")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("resynth", help="Autoencode (analysis-resynthesis) a WAV file.")
    sp.add_argument("--model", required=True)
    sp.add_argument("--in", dest="inp", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_resynth)

    sp = sub.add_parser("infer", help="Generate audio from f0/loudness only (decoder path).")
    sp.add_argument("--model", required=True)
    sp.add_argument("--f0", type=float, default=440.0)
    sp.add_argument("--loudness", type=float, default=-20.0)
    sp.add_argument("--duration", type=float, default=2.0)
    sp.add_argument("--z-file", default=None, help="Optional .npz with a 'z' array.")
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_infer)

    sp = sub.add_parser("selftest", help="Run an internal end-to-end sanity check.")
    sp.add_argument("--tmp", default="/tmp/ddsp_synth_selftest")
    sp.set_defaults(func=cmd_selftest)

    return p


# ==============================================================================
# Built-in selftest (also used by the exhaustive test pass below)
# ==============================================================================

def _run_selftest(tmp_dir: Path):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[selftest] working dir: {tmp_dir}")

    # 1. Harmonic synth sanity: energy at the fundamental.
    sr = DEFAULT_SR
    f0 = np.full(50, 220.0, dtype=np.float32)
    amp = np.full(50, 0.5, dtype=np.float32)
    hd = np.zeros((50, 10), dtype=np.float32)
    hd[:, 0] = 1.0
    audio = harmonic_synth_np(f0, amp, hd, int(1.0 * sr), sr)
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    peak_freq = freqs[np.argmax(spec)]
    assert abs(peak_freq - 220.0) < 5.0, f"harmonic synth peak at {peak_freq}, expected ~220"
    print(f"[selftest] harmonic synth OK (peak={peak_freq:.1f}Hz)")

    # 2. Filtered noise: lowpass should have less high-frequency energy than highpass.
    n_samples = int(1.0 * sr)
    mag_low = np.tile(np.linspace(1.0, 0.02, 20, dtype=np.float32), (20, 1))
    mag_high = np.tile(np.linspace(0.02, 1.0, 20, dtype=np.float32), (20, 1))
    low = filtered_noise_synth_np(mag_low, n_samples, sr)
    high = filtered_noise_synth_np(mag_high, n_samples, sr)

    def hf_energy(x):
        spec = np.abs(np.fft.rfft(x))
        freqs = np.fft.rfftfreq(len(x), 1 / sr)
        return spec[freqs > sr / 4].sum() / (spec.sum() + 1e-8)

    assert hf_energy(high) > hf_energy(low), "highpass noise should have more HF energy than lowpass"
    print(f"[selftest] filtered noise OK (hf_low={hf_energy(low):.3f}, hf_high={hf_energy(high):.3f})")

    # 3. Reverb: adds late energy (tail) beyond the dry signal's length-equivalent decay.
    dry = np.zeros(sr, dtype=np.float32)
    dry[:100] = 1.0  # impulse-like transient
    ir = synth_reverb_ir(1.0, 4.0, sr)
    wet = apply_reverb_np(dry, ir, wet=0.8)
    tail_energy = np.sum(wet[sr // 2 :] ** 2)
    assert tail_energy > 0.0, "reverb produced no tail energy"
    print(f"[selftest] reverb OK (tail_energy={tail_energy:.4f})")

    # 4. f0 extraction round-trip on a synthetic tone.
    test_f0 = 330.0
    t = np.arange(int(1.0 * sr)) / sr
    tone = 0.8 * np.sin(2 * np.pi * test_f0 * t).astype(np.float32)
    f0_est = extract_f0(tone, sr)
    voiced = f0_est[f0_est > 0]
    mean_est = float(voiced.mean()) if len(voiced) else 0.0
    assert abs(mean_est - test_f0) < 3.0, f"f0 estimate {mean_est}, expected ~{test_f0}"
    print(f"[selftest] f0 extraction OK (estimated {mean_est:.2f}Hz vs {test_f0}Hz)")

    # 5. Loudness extraction: louder signal -> higher loudness.
    quiet = 0.01 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    loud = 0.9 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    l_quiet = extract_loudness(quiet, sr).mean()
    l_loud = extract_loudness(loud, sr).mean()
    assert l_loud > l_quiet, "louder signal should have higher extracted loudness"
    print(f"[selftest] loudness extraction OK ({l_quiet:.1f}dB -> {l_loud:.1f}dB)")

    # 6. End-to-end: train briefly on synthetic tones and check loss decreases.
    torch = _torch()
    corpus_dir = tmp_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for i, f in enumerate([220, 330, 440]):
        t = np.arange(int(DEFAULT_SEGMENT_SECONDS * sr)) / sr
        sig = 0.4 * np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(len(t))
        save_wav(corpus_dir / f"tone_{i}.wav", sig.astype(np.float32), sr)

    class Args:
        pass

    a = Args()
    a.data = str(corpus_dir)
    a.epochs = 6
    a.batch_size = 2
    a.lr = 1e-3
    a.n_harmonics = 20
    a.n_noise_bands = 20
    a.z_dim = 8
    a.hidden = 64
    a.sample_rate = sr
    a.device = "cpu"
    a.out = str(tmp_dir / "selftest_model.pt")
    losses = cmd_train(a)
    assert losses[-1] < losses[0], f"training loss did not decrease: {losses}"
    print(f"[selftest] training OK (loss {losses[0]:.3f} -> {losses[-1]:.3f})")

    # 7. resynth + infer round trip using the freshly trained checkpoint.
    test_wav = corpus_dir / "tone_0.wav"
    recon_path = tmp_dir / "recon.wav"

    class RArgs:
        pass

    ra = RArgs()
    ra.model = a.out
    ra.inp = str(test_wav)
    ra.out = str(recon_path)
    cmd_resynth(ra)
    recon = load_wav(recon_path, sr)
    assert np.isfinite(recon).all() and np.abs(recon).max() > 0, "resynth produced silence/NaN"
    print(f"[selftest] resynth OK (wrote {recon_path}, peak={np.abs(recon).max():.3f})")

    infer_path = tmp_dir / "infer.wav"

    class IArgs:
        pass

    ia = IArgs()
    ia.model = a.out
    ia.f0 = 440.0
    ia.loudness = -15.0
    ia.duration = 1.0
    ia.z_file = None
    ia.out = str(infer_path)
    cmd_infer(ia)
    gen = load_wav(infer_path, sr)
    assert np.isfinite(gen).all() and np.abs(gen).max() > 0, "infer produced silence/NaN"
    print(f"[selftest] infer OK (wrote {infer_path}, peak={np.abs(gen).max():.3f})")

    print("[selftest] ALL CHECKS PASSED")


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
