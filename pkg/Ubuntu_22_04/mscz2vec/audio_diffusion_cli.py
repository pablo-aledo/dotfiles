#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       AUDIO DIFFUSION CLI  v1                                ║
║         Generación y transformación de audio mediante difusión               ║
║         sobre espectrogramas Mel (wrapper sobre audio-diffusion)             ║
║                                                                              ║
║  ARQUITECTURA:                                                               ║
║    Audio → Espectrograma Mel (imagen) → DDPM/DDIM → Audio                   ║
║    Modelos pre-entrenados en HuggingFace Hub (teticio/audio-diffusion-*)     ║
║    Latent Audio Diffusion opcional (VAE + DDIM, más rápido)                  ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    generate     — Genera audio nuevo desde ruido                             ║
║    variate      — Genera variación de un audio existente                     ║
║    interpolate  — Morphing esférico entre dos audios en espacio latente      ║
║    outpaint     — Extiende un audio preservando inicio y/o final             ║
║    encode       — Audio → z_latent.json (vector de ruido latente)            ║
║    info         — Muestra información del modelo y parámetros Mel            ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    torch, diffusers>=0.12.0, librosa, soundfile, numpy, Pillow              ║
║    Instalar: pip install audiodiffusion diffusers librosa soundfile          ║
╚══════════════════════════════════════════════════════════════════════════════╝

# ── Generar audio desde ruido (modelo por defecto) ────────────────────────────
python audio_diffusion_cli.py generate --output salida.wav

# Generar con modelo específico y más pasos (más calidad):
python audio_diffusion_cli.py generate \\
    --model teticio/audio-diffusion-breaks-256 \\
    --steps 100 --seed 42 \\
    --output breaks_nuevo.wav

# Generar con modelo latente (mucho más rápido, ~50 pasos):
python audio_diffusion_cli.py generate \\
    --model teticio/latent-audio-diffusion-ddim-256 \\
    --steps 50 --eta 0.0 \\
    --output latente.wav

# ── Variar un audio existente ─────────────────────────────────────────────────
# strength=0.0 → copia exacta   strength=1.0 → completamente libre
python audio_diffusion_cli.py variate \\
    --input original.wav \\
    --strength 0.5 --steps 50 \\
    --output variacion.wav

# Varias variaciones del mismo archivo:
python audio_diffusion_cli.py variate \\
    --input original.wav --strength 0.3 \\
    --count 4 --output-dir variaciones/

# ── Interpolar entre dos audios ───────────────────────────────────────────────
# Genera N pasos de morphing esférico (slerp) entre A y B:
python audio_diffusion_cli.py interpolate \\
    --input-a audio_a.wav --input-b audio_b.wav \\
    --steps 8 \\
    --output-dir morphing/

# Interpolación con alpha fijo (un único punto intermedio):
python audio_diffusion_cli.py interpolate \\
    --input-a audio_a.wav --input-b audio_b.wav \\
    --alpha 0.5 \\
    --output mezcla.wav

# ── Extender un audio (out-painting) ─────────────────────────────────────────
# Preservar los primeros 2 segundos y generar el resto:
python audio_diffusion_cli.py outpaint \\
    --input original.wav \\
    --mask-start 2.0 \\
    --output extendido.wav

# Preservar inicio y final, generar la parte central:
python audio_diffusion_cli.py outpaint \\
    --input original.wav \\
    --mask-start 1.0 --mask-end 1.0 \\
    --output relleno.wav

# ── Encode: audio → z_latent.json ────────────────────────────────────────────
# Requiere modelo DDIM (el encode es una inversión determinista):
python audio_diffusion_cli.py encode \\
    --model teticio/audio-diffusion-ddim-256 \\
    --input original.wav \\
    --output z_original.json

# Usar el vector guardado para generar variaciones reproducibles:
python audio_diffusion_cli.py variate \\
    --input original.wav --latent z_original.json \\
    --strength 0.4 --output variacion_reproducible.wav

# ── Información del modelo ────────────────────────────────────────────────────
python audio_diffusion_cli.py info --model teticio/audio-diffusion-256
python audio_diffusion_cli.py info --model teticio/latent-audio-diffusion-ddim-256

"""

import argparse
import json
import os
import sys
import textwrap
from math import acos, sin
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL   = 'teticio/audio-diffusion-256'
DEFAULT_STEPS   = 1000   # DDPM — se ajusta automáticamente a 50 para DDIM
DEFAULT_ETA     = 0.0
DEFAULT_SR      = 22050

# Modelos conocidos: nombre → scheduler esperado
KNOWN_MODELS = {
    'teticio/audio-diffusion-256':                   'ddpm',
    'teticio/audio-diffusion-breaks-256':            'ddpm',
    'teticio/audio-diffusion-instrumental-hiphop-256': 'ddpm',
    'teticio/audio-diffusion-ddim-256':              'ddim',
    'teticio/latent-audio-diffusion-256':            'ddpm',
    'teticio/latent-audio-diffusion-ddim-256':       'ddim',
    'teticio/conditional-latent-audio-diffusion-512': 'ddim',
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════════════════════

def _load_pipeline(model_id: str):
    """Carga el AudioDiffusionPipeline desde HuggingFace Hub o directorio local."""
    try:
        from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline
    except ImportError:
        print('[error] No se encontró audiodiffusion. Instala con:')
        print('        pip install audiodiffusion diffusers librosa soundfile')
        sys.exit(1)

    print(f'[load] Cargando modelo: {model_id}')
    try:
        pipe = AudioDiffusionPipeline.from_pretrained(model_id)
    except Exception as e:
        print(f'[error] No se pudo cargar el modelo: {e}')
        sys.exit(1)

    try:
        import torch
        if torch.cuda.is_available():
            pipe = pipe.to('cuda')
            print('[load] Usando CUDA')
        else:
            print('[load] Usando CPU (puede ser lento)')
    except Exception:
        pass

    return pipe


def _default_steps(pipe) -> int:
    """Devuelve el número de pasos recomendado según el scheduler del pipeline."""
    return pipe.get_default_steps()


def _save_audio(audio, sample_rate: int, path: str):
    """Guarda un array numpy como archivo de audio .wav."""
    import numpy as np
    try:
        import soundfile as sf
        sf.write(path, audio, sample_rate)
    except ImportError:
        # Fallback: scipy
        try:
            from scipy.io import wavfile
            audio_int = (audio * 32767).astype(np.int16)
            wavfile.write(path, sample_rate, audio_int)
        except ImportError:
            print('[error] Instala soundfile o scipy para guardar audio:')
            print('        pip install soundfile')
            sys.exit(1)


def _load_audio(path: str, sample_rate: int = DEFAULT_SR):
    """Carga un archivo de audio como array numpy mono."""
    try:
        import librosa
        audio, sr = librosa.load(path, mono=True, sr=sample_rate)
        return audio, sr
    except ImportError:
        print('[error] librosa no está instalado: pip install librosa')
        sys.exit(1)
    except Exception as e:
        print(f'[error] No se pudo cargar {path}: {e}')
        sys.exit(1)


def _image_to_noise(pipe, image, steps: int = 50):
    """
    Invierte una imagen de espectrograma a su tensor de ruido latente.
    Solo funciona con schedulers DDIM (la inversión es determinista).
    """
    from diffusers import DDIMScheduler
    if not isinstance(pipe.scheduler, DDIMScheduler):
        raise ValueError(
            'El encode DDIM requiere un modelo con DDIMScheduler. '
            'Usa --model teticio/audio-diffusion-ddim-256'
        )
    noise = pipe.encode([image], steps=steps)
    return noise


def _slerp(x0, x1, alpha: float):
    """Interpolación esférica (slerp) entre dos tensores de ruido."""
    import torch
    from math import acos, sin
    theta = acos(
        torch.dot(torch.flatten(x0), torch.flatten(x1))
        / torch.norm(x0) / torch.norm(x1)
    )
    if theta < 1e-6:
        return (1 - alpha) * x0 + alpha * x1
    return (sin((1 - alpha) * theta) * x0 + sin(alpha * theta) * x1) / sin(theta)


def _is_latent_model(pipe) -> bool:
    """Devuelve True si el pipeline usa un VAE (latent diffusion)."""
    return pipe.vqvae is not None


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: generate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate(args):
    import torch
    import numpy as np

    pipe  = _load_pipeline(args.model)
    steps = args.steps or _default_steps(pipe)

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)
        print(f'[generate] Seed: {args.seed}')

    # Cargar ruido latente desde fichero si se proporciona
    noise = None
    if args.latent:
        with open(args.latent) as f:
            data = json.load(f)
        import torch
        noise = torch.tensor(data['noise'])
        print(f'[generate] Ruido cargado desde {args.latent}')

    print(f'[generate] Modelo   : {args.model}')
    print(f'[generate] Pasos    : {steps}  eta={args.eta}')
    latent_str = ' (latent)' if _is_latent_model(pipe) else ''
    print(f'[generate] Tipo     : {"DDIM" if args.eta == 0.0 else "DDPM"}{latent_str}')

    # Directorio de salida para --count > 1
    if args.count > 1:
        out_dir = Path(args.output_dir or Path(args.output).parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.output).stem if args.output != 'output.wav' else 'generated'
    else:
        out_dir = None

    generated = []
    for i in range(args.count):
        images, (sample_rate, audios) = pipe(
            batch_size=1,
            steps=steps,
            generator=generator,
            eta=args.eta,
            noise=noise,
            return_dict=False,
        )
        audio = audios[0]

        if args.count == 1:
            out_path = args.output
        else:
            out_path = str(out_dir / f'{stem}_{i+1:03d}.wav')

        _save_audio(audio, sample_rate, out_path)
        duration = len(audio) / sample_rate
        print(f'[generate] [{i+1}/{args.count}] {out_path}  '
              f'({duration:.2f}s, {sample_rate}Hz)')
        generated.append(out_path)

    if args.count > 1:
        print(f'\n[generate] {args.count} archivos guardados en {out_dir}')
    else:
        print(f'[generate] Guardado en {args.output}')


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: variate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_variate(args):
    import torch
    import numpy as np

    pipe  = _load_pipeline(args.model)
    steps = args.steps or _default_steps(pipe)

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    # Cargar ruido latente externo si se proporciona
    noise = None
    if args.latent:
        with open(args.latent) as f:
            data = json.load(f)
        noise = torch.tensor(data['noise'])
        print(f'[variate] Ruido cargado desde {args.latent}')

    print(f'[variate] Input    : {args.input}')
    print(f'[variate] Modelo   : {args.model}')
    print(f'[variate] Strength : {args.strength}  (0=copia, 1=libre)')
    print(f'[variate] Pasos    : {steps}  eta={args.eta}')

    # start_step: cuántos pasos saltar desde el final (strength fracción del total)
    start_step = int(args.strength * steps)

    # Directorio de salida para --count > 1
    if args.count > 1:
        out_dir = Path(args.output_dir or 'variaciones')
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.input).stem
    else:
        out_dir = None

    for i in range(args.count):
        images, (sample_rate, audios) = pipe(
            audio_file=args.input,
            slice=args.slice,
            start_step=start_step,
            steps=steps,
            generator=generator,
            eta=args.eta,
            noise=noise,
            return_dict=False,
        )
        audio = audios[0]

        if args.count == 1:
            out_path = args.output
        else:
            out_path = str(out_dir / f'{stem}_var{i+1:03d}_s{args.strength}.wav')

        _save_audio(audio, sample_rate, out_path)
        duration = len(audio) / sample_rate
        print(f'[variate] [{i+1}/{args.count}] {out_path}  ({duration:.2f}s)')

    if args.count > 1:
        print(f'\n[variate] {args.count} variaciones guardadas en {out_dir}')
    else:
        print(f'[variate] Guardado en {args.output}')


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: interpolate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_interpolate(args):
    import torch
    import numpy as np

    pipe  = _load_pipeline(args.model)
    steps = args.steps or _default_steps(pipe)

    print(f'[interpolate] A      : {args.input_a}')
    print(f'[interpolate] B      : {args.input_b}')
    print(f'[interpolate] Modelo : {args.model}')
    print(f'[interpolate] Pasos  : {steps}  eta={args.eta}')

    # ── Encode de ambos audios (requiere DDIM) ────────────────────────────
    print('[interpolate] Encodando A ...')
    images_a, _ = pipe(audio_file=args.input_a, start_step=0,
                        steps=steps, return_dict=False)
    noise_a = _image_to_noise(pipe, images_a[0], steps=steps)

    print('[interpolate] Encodando B ...')
    images_b, _ = pipe(audio_file=args.input_b, start_step=0,
                        steps=steps, return_dict=False)
    noise_b = _image_to_noise(pipe, images_b[0], steps=steps)

    # ── Generar puntos de interpolación ──────────────────────────────────
    if args.alpha is not None:
        # Un único punto intermedio
        alphas   = [args.alpha]
        use_dir  = False
    else:
        # N pasos uniformes entre A y B
        n        = args.steps_interp
        alphas   = [i / (n - 1) for i in range(n)]
        use_dir  = True

    if use_dir:
        out_dir = Path(args.output_dir or 'interpolacion')
        out_dir.mkdir(parents=True, exist_ok=True)
        stem_a = Path(args.input_a).stem
        stem_b = Path(args.input_b).stem
        print(f'[interpolate] Generando {len(alphas)} pasos → {out_dir}')
    else:
        print(f'[interpolate] Generando alpha={args.alpha:.3f}')

    for idx, alpha in enumerate(alphas):
        noise_interp = _slerp(noise_a, noise_b, alpha)

        _, (sample_rate, audios) = pipe(
            noise=noise_interp,
            steps=steps,
            eta=args.eta,
            return_dict=False,
        )
        audio = audios[0]

        if use_dir:
            out_path = str(out_dir / f'{stem_a}_to_{stem_b}_{idx+1:03d}_a{alpha:.2f}.wav')
        else:
            out_path = args.output

        _save_audio(audio, sample_rate, out_path)
        duration = len(audio) / sample_rate
        print(f'[interpolate] [{idx+1}/{len(alphas)}] alpha={alpha:.3f}  '
              f'{out_path}  ({duration:.2f}s)')

    if use_dir:
        print(f'\n[interpolate] {len(alphas)} archivos guardados en {out_dir}')
    else:
        print(f'[interpolate] Guardado en {args.output}')


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: outpaint
# ══════════════════════════════════════════════════════════════════════════════

def cmd_outpaint(args):
    import torch

    pipe  = _load_pipeline(args.model)
    steps = args.steps or _default_steps(pipe)

    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    print(f'[outpaint] Input      : {args.input}')
    print(f'[outpaint] Modelo     : {args.model}')
    print(f'[outpaint] Mask start : {args.mask_start}s')
    print(f'[outpaint] Mask end   : {args.mask_end}s')
    print(f'[outpaint] Pasos      : {steps}  eta={args.eta}')

    images, (sample_rate, audios) = pipe(
        audio_file=args.input,
        slice=args.slice,
        start_step=0,
        steps=steps,
        generator=generator,
        mask_start_secs=args.mask_start,
        mask_end_secs=args.mask_end,
        eta=args.eta,
        return_dict=False,
    )
    audio = audios[0]

    _save_audio(audio, sample_rate, args.output)
    duration = len(audio) / sample_rate
    print(f'[outpaint] Guardado en {args.output}  ({duration:.2f}s, {sample_rate}Hz)')


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: encode
# ══════════════════════════════════════════════════════════════════════════════

def cmd_encode(args):
    import torch
    import numpy as np

    pipe  = _load_pipeline(args.model)
    steps = args.steps or _default_steps(pipe)

    # Verificar que el modelo es DDIM antes de intentar el encode
    from diffusers import DDIMScheduler
    if not isinstance(pipe.scheduler, DDIMScheduler):
        scheduler_type = type(pipe.scheduler).__name__
        print(f'[encode] ERROR: el encode requiere un modelo DDIM.')
        print(f'         El modelo cargado usa {scheduler_type}.')
        print(f'         Usa: --model teticio/audio-diffusion-ddim-256')
        print(f'              o: --model teticio/latent-audio-diffusion-ddim-256')
        sys.exit(1)

    print(f'[encode] Input  : {args.input}')
    print(f'[encode] Modelo : {args.model}')
    print(f'[encode] Pasos  : {steps}')

    # Generar la imagen de espectrograma del audio de entrada
    print('[encode] Convirtiendo audio → espectrograma ...')
    images, (sample_rate, _) = pipe(
        audio_file=args.input,
        slice=args.slice,
        start_step=0,
        steps=steps,
        return_dict=False,
    )

    # Invertir DDIM: imagen → ruido latente
    print('[encode] Invirtiendo DDIM (imagen → ruido) ...')
    noise = _image_to_noise(pipe, images[0], steps=steps)

    out_path = args.output or (Path(args.input).stem + '.latent.json')
    payload = {
        'source':      args.input,
        'model':       args.model,
        'slice':       args.slice,
        'ddim_steps':  steps,
        'sample_rate': sample_rate,
        'noise_shape': list(noise.shape),
        'noise':       noise.cpu().numpy().tolist(),
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'[encode] Ruido latente guardado en {out_path}')
    print(f'[encode] Shape: {list(noise.shape)}')
    print()
    print('  Usa este archivo con:')
    print(f'    variate  --latent {out_path} --input {args.input}')
    print(f'    generate --latent {out_path}')
    print(f'    interpolate --input-a {args.input} --input-b otro.wav')


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: info
# ══════════════════════════════════════════════════════════════════════════════

def cmd_info(args):
    pipe = _load_pipeline(args.model)

    from diffusers import DDIMScheduler, DDPMScheduler

    scheduler_type = 'DDIM' if isinstance(pipe.scheduler, DDIMScheduler) else 'DDPM'
    default_steps  = pipe.get_default_steps()
    is_latent      = _is_latent_model(pipe)

    # Parámetros del espectrograma Mel
    mel = pipe.mel
    sr        = mel.get_sample_rate()
    hop       = mel.hop_length
    x_res     = mel.x_res
    y_res     = mel.y_res
    n_fft     = mel.n_fft
    top_db    = mel.top_db
    n_iter    = mel.n_iter
    slice_sec = x_res * hop / sr

    # Parámetros de la U-Net
    try:
        unet        = pipe.unet
        sample_size = unet.sample_size
        in_ch       = unet.in_channels
        unet_params = sum(p.numel() for p in unet.parameters()) / 1e6
    except Exception:
        sample_size = '?'
        in_ch       = '?'
        unet_params = 0.0

    print()
    print('═' * 60)
    print(f'  MODELO: {args.model}')
    print('═' * 60)
    print(f'  Scheduler     : {scheduler_type}  (pasos por defecto: {default_steps})')
    print(f'  Latent (VAE)  : {"Sí" if is_latent else "No"}')
    print(f'  U-Net size    : {sample_size}  in_channels={in_ch}')
    if unet_params > 0:
        print(f'  Parámetros    : {unet_params:.1f}M')
    print()
    print('  Espectrograma Mel:')
    print(f'    Resolución   : {x_res} × {y_res}  (tiempo × frecuencia)')
    print(f'    Sample rate  : {sr} Hz')
    print(f'    Hop length   : {hop}')
    print(f'    n_fft        : {n_fft}')
    print(f'    top_db       : {top_db}')
    print(f'    n_iter       : {n_iter}  (Griffin-Lim inversión)')
    print(f'    Duración/slice: {slice_sec:.2f}s  '
          f'({x_res} ticks × {hop} / {sr}Hz)')
    print()
    print('  Comandos recomendados:')
    if scheduler_type == 'DDIM':
        print(f'    generate   --model {args.model} --steps 50 --eta 0.0')
        print(f'    encode     --model {args.model} --input audio.wav')
        print(f'    interpolate --model {args.model} --input-a a.wav --input-b b.wav')
    else:
        print(f'    generate   --model {args.model} --steps 100')
        print(f'    variate    --model {args.model} --input audio.wav --strength 0.5')
    print('═' * 60)
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='audio_diffusion_cli',
        description='Generación y transformación de audio mediante difusión sobre espectrogramas Mel',
    )
    sub = parser.add_subparsers(dest='command')
    sub.required = True

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = sub.add_parser('generate',
        help='Genera audio nuevo desde ruido gaussiano',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Genera audio nuevo mediante el proceso de denoising completo.
            Sin --input: generación completamente libre desde ruido.
            Con --latent: parte de un vector de ruido previamente codificado.

            Modelos recomendados:
              teticio/audio-diffusion-256                 (DDPM, 1000 pasos)
              teticio/audio-diffusion-ddim-256            (DDIM, 50 pasos)
              teticio/latent-audio-diffusion-ddim-256     (Latent DDIM, más rápido)

            Ejemplos:
              generate --output nuevo.wav
              generate --model teticio/audio-diffusion-ddim-256 --steps 50 --output nuevo.wav
              generate --seed 42 --count 4 --output-dir lote/
        """))
    p_gen.add_argument('--model',      default=DEFAULT_MODEL, metavar='ID_O_DIR',
        help=f'Modelo HuggingFace Hub o directorio local (default: {DEFAULT_MODEL})')
    p_gen.add_argument('--steps',      type=int, default=None, metavar='INT',
        help='Pasos de denoising (default: 50 DDIM / 1000 DDPM según modelo)')
    p_gen.add_argument('--eta',        type=float, default=DEFAULT_ETA, metavar='FLOAT',
        help='Estocasticidad DDIM: 0.0=determinista, 1.0=DDPM completo (default: 0.0)')
    p_gen.add_argument('--seed',       type=int, default=None, metavar='INT',
        help='Semilla aleatoria para reproducibilidad')
    p_gen.add_argument('--latent',     default=None, metavar='FILE',
        help='Fichero .json con vector de ruido latente (de encode)')
    p_gen.add_argument('--count',      type=int, default=1, metavar='INT',
        help='Número de audios a generar (default: 1)')
    p_gen.add_argument('--output',     default='output.wav', metavar='FILE')
    p_gen.add_argument('--output-dir', default=None, metavar='DIR',
        dest='output_dir',
        help='Directorio de salida cuando --count > 1')
    p_gen.set_defaults(func=cmd_generate)

    # ── variate ───────────────────────────────────────────────────────────────
    p_var = sub.add_parser('variate',
        help='Genera una variación de un audio existente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Parte del espectrograma del audio de entrada, añade ruido hasta
            el nivel definido por --strength, y regenera desde ahí.

            strength=0.0 → copia casi exacta del input
            strength=0.5 → variación moderada (recomendado para exploración)
            strength=1.0 → generación completamente libre (ignora el input)

            Ejemplos:
              variate --input loop.wav --strength 0.4 --output variacion.wav
              variate --input loop.wav --strength 0.3 --count 8 --output-dir variaciones/
        """))
    p_var.add_argument('--input',      required=True, metavar='FILE',
        help='Audio de referencia (.wav, .mp3, .flac)')
    p_var.add_argument('--model',      default=DEFAULT_MODEL, metavar='ID_O_DIR')
    p_var.add_argument('--strength',   type=float, default=0.5, metavar='FLOAT',
        help='Distancia respecto al input: 0.0=copia, 1.0=libre (default: 0.5)')
    p_var.add_argument('--steps',      type=int, default=None, metavar='INT',
        help='Pasos de denoising (default: según modelo)')
    p_var.add_argument('--eta',        type=float, default=DEFAULT_ETA, metavar='FLOAT',
        help='Estocasticidad DDIM: 0.0=determinista, 1.0=DDPM (default: 0.0)')
    p_var.add_argument('--slice',      type=int, default=0, metavar='INT',
        help='Slice del audio a procesar cuando el archivo es más largo (default: 0)')
    p_var.add_argument('--seed',       type=int, default=None, metavar='INT')
    p_var.add_argument('--latent',     default=None, metavar='FILE',
        help='Fichero .json de ruido latente para variaciones reproducibles')
    p_var.add_argument('--count',      type=int, default=1, metavar='INT',
        help='Número de variaciones a generar (default: 1)')
    p_var.add_argument('--output',     default='output_variate.wav', metavar='FILE')
    p_var.add_argument('--output-dir', default=None, metavar='DIR',
        dest='output_dir',
        help='Directorio de salida cuando --count > 1')
    p_var.set_defaults(func=cmd_variate)

    # ── interpolate ───────────────────────────────────────────────────────────
    p_int = sub.add_parser('interpolate',
        help='Morphing esférico (slerp) entre dos audios en espacio latente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Codifica dos audios a sus vectores de ruido latente (inversión DDIM)
            e interpola esféricamente entre ellos.

            Requiere un modelo DDIM (la inversión es determinista).
            Recomendado: teticio/audio-diffusion-ddim-256

            Modos:
              --steps-interp N   genera N audios de A→B (guarda en --output-dir)
              --alpha FLOAT      un único punto a alpha entre A y B (guarda en --output)

            Ejemplos:
              interpolate --input-a a.wav --input-b b.wav --steps-interp 8 --output-dir morph/
              interpolate --input-a a.wav --input-b b.wav --alpha 0.5 --output mezcla.wav
        """))
    p_int.add_argument('--input-a',     required=True, metavar='FILE',
        dest='input_a', help='Audio origen A')
    p_int.add_argument('--input-b',     required=True, metavar='FILE',
        dest='input_b', help='Audio destino B')
    p_int.add_argument('--model',       default='teticio/audio-diffusion-ddim-256',
        metavar='ID_O_DIR',
        help='Modelo DDIM (default: teticio/audio-diffusion-ddim-256)')
    p_int.add_argument('--steps',       type=int, default=None, metavar='INT',
        help='Pasos DDIM para encode y decode (default: 50)')
    p_int.add_argument('--eta',         type=float, default=0.0, metavar='FLOAT')
    p_int.add_argument('--alpha',       type=float, default=None, metavar='FLOAT',
        help='Alpha fijo [0,1]: 0.0=A, 1.0=B. Si se indica, genera un único punto.')
    p_int.add_argument('--steps-interp', type=int, default=8, metavar='INT',
        dest='steps_interp',
        help='Número de pasos de interpolación A→B (default: 8). '
             'Ignorado si --alpha está presente.')
    p_int.add_argument('--output',      default='output_interp.wav', metavar='FILE',
        help='Archivo de salida cuando se usa --alpha')
    p_int.add_argument('--output-dir',  default=None, metavar='DIR',
        dest='output_dir',
        help='Directorio de salida para la secuencia de interpolación')
    p_int.set_defaults(func=cmd_interpolate)

    # ── outpaint ──────────────────────────────────────────────────────────────
    p_out = sub.add_parser('outpaint',
        help='Extiende un audio preservando inicio y/o final',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Aplica un proceso de denoising parcial sobre el espectrograma:
            los segmentos enmascarados se preservan exactamente, y el
            resto se regenera de forma coherente con el contexto.

            mask-start/end definen cuántos segundos del inicio/final se
            conservan sin cambios. La parte no enmascarada se genera
            libremente condicionada al resto del espectrograma.

            Ejemplos:
              outpaint --input loop.wav --mask-start 1.5 --output extendido.wav
              outpaint --input loop.wav --mask-start 1.0 --mask-end 1.0 --output relleno.wav
        """))
    p_out.add_argument('--input',      required=True, metavar='FILE')
    p_out.add_argument('--model',      default=DEFAULT_MODEL, metavar='ID_O_DIR')
    p_out.add_argument('--mask-start', type=float, default=0.0, metavar='SECS',
        dest='mask_start',
        help='Segundos de audio a preservar al inicio (default: 0.0)')
    p_out.add_argument('--mask-end',   type=float, default=0.0, metavar='SECS',
        dest='mask_end',
        help='Segundos de audio a preservar al final (default: 0.0)')
    p_out.add_argument('--steps',      type=int, default=None, metavar='INT')
    p_out.add_argument('--eta',        type=float, default=DEFAULT_ETA, metavar='FLOAT')
    p_out.add_argument('--slice',      type=int, default=0, metavar='INT')
    p_out.add_argument('--seed',       type=int, default=None, metavar='INT')
    p_out.add_argument('--output',     default='output_outpaint.wav', metavar='FILE')
    p_out.set_defaults(func=cmd_outpaint)

    # ── encode ────────────────────────────────────────────────────────────────
    p_enc = sub.add_parser('encode',
        help='Audio → z_latent.json mediante inversión DDIM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Invierte el proceso DDIM para recuperar el vector de ruido latente
            que correspondería al audio de entrada. El resultado se guarda como
            JSON y puede usarse con generate, variate e interpolate.

            Requiere un modelo DDIM.
            Recomendado: teticio/audio-diffusion-ddim-256

            Ejemplo:
              encode --input loop.wav --output z_loop.json
              variate --input loop.wav --latent z_loop.json --strength 0.3
        """))
    p_enc.add_argument('--input',   required=True, metavar='FILE')
    p_enc.add_argument('--model',   default='teticio/audio-diffusion-ddim-256',
        metavar='ID_O_DIR')
    p_enc.add_argument('--steps',   type=int, default=None, metavar='INT',
        help='Pasos DDIM de inversión (default: 50)')
    p_enc.add_argument('--slice',   type=int, default=0, metavar='INT',
        help='Slice del audio a codificar (default: 0)')
    p_enc.add_argument('--output',  default=None, metavar='FILE',
        help='Ruta del .json de salida (default: <stem>.latent.json)')
    p_enc.set_defaults(func=cmd_encode)

    # ── info ──────────────────────────────────────────────────────────────────
    p_inf = sub.add_parser('info',
        help='Información del modelo: scheduler, parámetros Mel, comandos recomendados')
    p_inf.add_argument('--model', default=DEFAULT_MODEL, metavar='ID_O_DIR')
    p_inf.set_defaults(func=cmd_info)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
