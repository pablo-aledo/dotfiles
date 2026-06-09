#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           AUDIOLM  v1.0                                      ║
║         Modelado de lenguaje sobre audio — wrapper CLI para audiolm-pytorch  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  CONCEPTO                                                                    ║
║    AudioLM (Google Research, 2022) trata la generación de audio como un     ║
║    problema de predicción de tokens, igual que un LLM con texto.            ║
║    El pipeline tiene tres etapas encadenadas:                               ║
║                                                                              ║
║    Audio bruto                                                               ║
║      │                                                                       ║
║      ├─[1] SemanticTransformer  ← tokens semánticos (HuBERT/vq-wav2vec)    ║
║      │       captura estructura de alto nivel: prosodia, contenido          ║
║      │                                                                       ║
║      ├─[2] CoarseTransformer   ← tokens acústicos gruesos (SoundStream)    ║
║      │       captura timbre, envolvente, estructura armónica                ║
║      │                                                                       ║
║      └─[3] FineTransformer     ← tokens acústicos finos                    ║
║              captura detalles de alta frecuencia                            ║
║                                                                              ║
║    Cada transformer se entrena por separado; la generación los encadena.   ║
║    El codec (SoundStream o EnCodec) actúa como tokenizador de audio.       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUBCOMANDOS                                                                 ║
║                                                                              ║
║    train-codec      Entrenar SoundStream sobre un corpus de audio           ║
║    train-semantic   Entrenar el SemanticTransformer (etapa 1)               ║
║    train-coarse     Entrenar el CoarseTransformer (etapa 2)                 ║
║    train-fine       Entrenar el FineTransformer (etapa 3)                   ║
║    generate         Generar audio con los 3 modelos entrenados              ║
║    continue         Continuar audio desde un fichero de referencia (prompt) ║
║    info             Inspeccionar un checkpoint guardado                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FLUJO TÍPICO DE USO                                                         ║
║                                                                              ║
║  # 1. Entrenar el codec neuronal (tokenizador de audio)                     ║
║  python audiolm.py train-codec                                               ║
║      --audio-dir ./corpus/                                                   ║
║      --checkpoint codec.pt                                                   ║
║      --steps 100000                                                          ║
║                                                                              ║
║  # 2. Etapa semántica (requiere HuBERT preentrenado)                        ║
║  python audiolm.py train-semantic                                            ║
║      --audio-dir ./corpus/                                                   ║
║      --hubert-ckpt hubert_base_ls960.pt                                     ║
║      --hubert-quantizer hubert_base_ls960_L9_km500.bin                      ║
║      --checkpoint semantic.pt                                                ║
║      --steps 100000                                                          ║
║                                                                              ║
║  # 3. Transformer grueso                                                    ║
║  python audiolm.py train-coarse                                              ║
║      --audio-dir ./corpus/                                                   ║
║      --codec-ckpt codec.pt                                                  ║
║      --hubert-ckpt hubert_base_ls960.pt                                     ║
║      --hubert-quantizer hubert_base_ls960_L9_km500.bin                      ║
║      --checkpoint coarse.pt                                                  ║
║      --steps 100000                                                          ║
║                                                                              ║
║  # 4. Transformer fino                                                      ║
║  python audiolm.py train-fine                                                ║
║      --audio-dir ./corpus/                                                   ║
║      --codec-ckpt codec.pt                                                  ║
║      --checkpoint fine.pt                                                    ║
║      --steps 100000                                                          ║
║                                                                              ║
║  # 5. Generar audio                                                         ║
║  python audiolm.py generate                                                  ║
║      --codec-ckpt codec.pt                                                  ║
║      --semantic-ckpt semantic.pt                                             ║
║      --coarse-ckpt coarse.pt                                                 ║
║      --fine-ckpt fine.pt                                                     ║
║      --hubert-ckpt hubert_base_ls960.pt                                     ║
║      --hubert-quantizer hubert_base_ls960_L9_km500.bin                      ║
║      --output generado.wav                                                   ║
║                                                                              ║
║  # 5b. Generar con texto (si los transformers fueron entrenados con T5)     ║
║  python audiolm.py generate  [mismos flags]  --text "lluvia sobre el mar"  ║
║                                                                              ║
║  # 5c. Continuar desde un audio de referencia (audio prompting)             ║
║  python audiolm.py continue referencia.wav  [mismos flags]                  ║
║                                                                              ║
║  # 6. Inspeccionar un checkpoint                                             ║
║  python audiolm.py info codec.pt                                             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OPCIONES COMUNES (disponibles en todos los subcomandos de entrenamiento)    ║
║    --audio-dir DIR        Carpeta con ficheros .wav/.flac/.mp3              ║
║    --checkpoint FILE      Dónde guardar / desde dónde cargar el modelo      ║
║    --results-dir DIR      Carpeta para samples y checkpoints [./results]    ║
║    --steps N              Pasos de entrenamiento [100000]                   ║
║    --batch-size N         Tamaño de batch [4]                               ║
║    --grad-accum N         Pasos de acumulación de gradiente [8]             ║
║    --lr F                 Learning rate [3e-4]                              ║
║    --seconds F            Duración máxima de cada clip en segundos [2.0]   ║
║    --save-every N         Guardar checkpoint cada N pasos [1000]           ║
║    --sample-every N       Generar muestra cada N pasos [500]               ║
║    --wandb                Activar logging con Weights & Biases              ║
║    --seed N               Semilla aleatoria [42]                            ║
║    --verbose              Mostrar configuración completa antes de lanzar    ║
║    --dry-run              Construir modelos y mostrar config sin entrenar   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ARQUITECTURA POR DEFECTO (compatible entre las 4 etapas de entrenamiento)  ║
║    codec:    codebook_size=4096, rq_num_quantizers=12, target_sample_hz=16k ║
║              num_coarse_quantizers=3  (→ fine usa los 9 restantes)          ║
║    semantic: dim=1024, depth=6, num_semantic_tokens=500                     ║
║    coarse:   dim=1024, depth=6, num_coarse_quantizers=3                    ║
║    fine:     dim=1024, depth=6, num_fine_quantizers=9                      ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install audiolm-pytorch                                               ║
║    pip install fairseq   # para HuBERT / vq-wav2vec                         ║
║                                                                              ║
║  DESCARGA DE PESOS HuBERT:                                                  ║
║    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt          ║
║    wget https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0"

# Arquitectura por defecto — coherente entre las 4 etapas
DEFAULT_CODEBOOK_SIZE       = 4096
DEFAULT_RQ_NUM_QUANTIZERS   = 12       # cuantizadores totales del codec
DEFAULT_NUM_COARSE_Q        = 3        # cuantizadores gruesos (etapas 2 y 3)
DEFAULT_NUM_SEMANTIC_TOKENS = 500      # clusters KMeans de HuBERT
DEFAULT_TARGET_SAMPLE_HZ    = 16000
DEFAULT_DIM                 = 1024
DEFAULT_DEPTH               = 6

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        sys.exit("[ERROR] PyTorch no encontrado. Instala con: pip install torch torchaudio")

def _import_audiolm():
    try:
        import audiolm_pytorch
        return audiolm_pytorch
    except ImportError:
        sys.exit("[ERROR] audiolm-pytorch no encontrado. Instala con: pip install audiolm-pytorch")

def _print_config(title, cfg: dict):
    """Imprime configuración en tabla alineada al estilo de las herramientas."""
    width = 76
    print(f"╔{'═' * width}╗")
    print(f"║  {title:<{width - 2}}║")
    print(f"╠{'═' * width}╣")
    for k, v in cfg.items():
        line = f"  {k:<30} {str(v)}"
        print(f"║{line:<{width}}║")
    print(f"╚{'═' * width}╝")
    print()

def _save_wav(tensor, path: str, sample_rate: int):
    """Guarda tensor de audio (1, T) o (T,) como WAV."""
    import torchaudio
    import torch
    t = tensor.detach().cpu()
    if t.ndim == 1:
        t = t.unsqueeze(0)
    if t.ndim == 3:
        t = t.squeeze(0)
    torchaudio.save(path, t, sample_rate)

def _load_hubert(hubert_ckpt: str, hubert_quantizer: str):
    """Construye y devuelve un HubertWithKmeans."""
    alm = _import_audiolm()
    wav2vec = alm.HubertWithKmeans(
        checkpoint_path=hubert_ckpt,
        kmeans_path=hubert_quantizer,
    )
    return wav2vec

def _build_soundstream(args, num_coarse_q: int = None):
    """Construye SoundStream con la arquitectura por defecto o los args dados."""
    alm = _import_audiolm()
    ncoarse = num_coarse_q if num_coarse_q is not None else args.num_coarse_q
    soundstream = alm.SoundStream(
        codebook_size        = args.codebook_size,
        rq_num_quantizers    = args.rq_num_quantizers,
        rq_groups            = args.rq_groups,
        use_lookup_free_quantizer = args.use_lfq,
        target_sample_hz     = args.sample_hz,
        attn_window_size     = args.attn_window,
        attn_depth           = args.attn_depth,
    )
    return soundstream

def _load_soundstream_from_ckpt(ckpt_path: str, args):
    """Carga SoundStream desde checkpoint; reconstruye la arquitectura."""
    alm = _import_audiolm()
    soundstream = _build_soundstream(args)
    soundstream.load(ckpt_path)
    soundstream.eval()
    return soundstream

# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: train-codec
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train_codec(args):
    """
    Entrena SoundStream sobre un corpus de audio.
    Es el primer paso obligatorio (a menos que uses EnCodec preentrenado).
    """
    torch = _import_torch()
    alm   = _import_audiolm()

    cfg = {
        "audio_dir":           args.audio_dir,
        "checkpoint":          args.checkpoint,
        "results_dir":         args.results_dir,
        "steps":               args.steps,
        "batch_size":          args.batch_size,
        "grad_accum":          args.grad_accum,
        "lr":                  args.lr,
        "seconds":             args.seconds,
        "codebook_size":       args.codebook_size,
        "rq_num_quantizers":   args.rq_num_quantizers,
        "rq_groups":           args.rq_groups,
        "num_coarse_q":        args.num_coarse_q,
        "target_sample_hz":    args.sample_hz,
        "use_lfq":             args.use_lfq,
        "attn_window":         args.attn_window,
        "attn_depth":          args.attn_depth,
        "save_every":          args.save_every,
        "sample_every":        args.sample_every,
        "wandb":               args.wandb,
        "seed":                args.seed,
    }

    if args.verbose or args.dry_run:
        _print_config("TRAIN-CODEC — Configuración", cfg)

    if args.dry_run:
        print("[dry-run] Construcción de SoundStream simulada. No se entrena.")
        return

    # Construir modelo
    soundstream = _build_soundstream(args)

    print(f"[train-codec] SoundStream listo.")
    print(f"  Parámetros: {sum(p.numel() for p in soundstream.parameters()):,}")
    print(f"  Corpus:     {args.audio_dir}")
    print()

    trainer = alm.SoundStreamTrainer(
        soundstream,
        folder                = args.audio_dir,
        num_train_steps       = args.steps,
        batch_size            = args.batch_size,
        grad_accum_every      = args.grad_accum,
        lr                    = args.lr,
        data_max_length_seconds = args.seconds,
        results_folder        = args.results_dir,
        save_results_every    = args.sample_every,
        save_model_every      = args.save_every,
        random_split_seed     = args.seed,
        use_wandb_tracking    = args.wandb,
        force_clear_prev_results = False,
    )

    trainer.train()

    # Exportar checkpoint final con nombre indicado por el usuario
    # El trainer guarda como soundstream.{paso}.pt — buscar el más reciente
    import shutil
    results = Path(args.results_dir)
    candidates = sorted(results.glob("soundstream.*.pt"),
                        key=lambda p: int(p.stem.split(".")[-1]))
    if candidates:
        ckpt_src = candidates[-1]
        shutil.copy(ckpt_src, args.checkpoint)
        print(f"\n[train-codec] Checkpoint guardado en: {args.checkpoint}  (desde {ckpt_src.name})")
    else:
        print(f"\n[train-codec] Entrenamiento completado. Checkpoints en: {args.results_dir}/")

# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: train-semantic
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train_semantic(args):
    """
    Entrena el SemanticTransformer (etapa 1).
    Requiere HuBERT preentrenado (--hubert-ckpt y --hubert-quantizer).
    No necesita el codec — trabaja solo con tokens semánticos.
    """
    torch = _import_torch()
    alm   = _import_audiolm()

    cfg = {
        "audio_dir":          args.audio_dir,
        "hubert_ckpt":        args.hubert_ckpt,
        "hubert_quantizer":   args.hubert_quantizer,
        "checkpoint":         args.checkpoint,
        "results_dir":        args.results_dir,
        "steps":              args.steps,
        "batch_size":         args.batch_size,
        "grad_accum":         args.grad_accum,
        "lr":                 args.lr,
        "seconds":            args.seconds,
        "num_semantic_tokens": args.num_semantic_tokens,
        "dim":                args.dim,
        "depth":              args.depth,
        "heads":              args.heads,
        "flash_attn":         args.flash_attn,
        "has_condition":      args.has_condition,
        "save_every":         args.save_every,
        "sample_every":       args.sample_every,
        "wandb":              args.wandb,
        "seed":               args.seed,
    }

    if args.verbose or args.dry_run:
        _print_config("TRAIN-SEMANTIC — Configuración", cfg)

    if args.dry_run:
        print("[dry-run] Construcción simulada. No se entrena.")
        return

    # HuBERT + KMeans
    print(f"[train-semantic] Cargando HuBERT desde {args.hubert_ckpt} ...")
    wav2vec = _load_hubert(args.hubert_ckpt, args.hubert_quantizer)

    # SemanticTransformer
    semantic_transformer = alm.SemanticTransformer(
        dim                  = args.dim,
        depth                = args.depth,
        heads                = args.heads,
        num_semantic_tokens  = args.num_semantic_tokens,
        has_condition        = args.has_condition,
        flash_attn           = args.flash_attn,
    )

    print(f"[train-semantic] SemanticTransformer listo.")
    print(f"  Parámetros: {sum(p.numel() for p in semantic_transformer.parameters()):,}")
    print(f"  Tokens semánticos: {args.num_semantic_tokens}")
    print()

    trainer = alm.SemanticTransformerTrainer(
        wav2vec              = wav2vec,
        transformer          = semantic_transformer,
        folder               = args.audio_dir,
        num_train_steps      = args.steps,
        batch_size           = args.batch_size,
        grad_accum_every     = args.grad_accum,
        lr                   = args.lr,
        data_max_length_seconds = args.seconds,
        results_folder       = args.results_dir,
        save_results_every   = args.sample_every,
        save_model_every     = args.save_every,
        random_split_seed    = args.seed,
        use_wandb_tracking   = args.wandb,
        force_clear_prev_results = False,
    )

    trainer.train()
    print(f"\n[train-semantic] Entrenamiento completado. "
          f"Checkpoints en: {args.results_dir}/")

# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: train-coarse
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train_coarse(args):
    """
    Entrena el CoarseTransformer (etapa 2).
    Requiere el codec entrenado (--codec-ckpt) y HuBERT.
    Aprende a generar tokens acústicos gruesos condicionados en tokens semánticos.
    """
    torch = _import_torch()
    alm   = _import_audiolm()

    cfg = {
        "audio_dir":          args.audio_dir,
        "codec_ckpt":         args.codec_ckpt,
        "hubert_ckpt":        args.hubert_ckpt,
        "hubert_quantizer":   args.hubert_quantizer,
        "checkpoint":         args.checkpoint,
        "results_dir":        args.results_dir,
        "steps":              args.steps,
        "batch_size":         args.batch_size,
        "grad_accum":         args.grad_accum,
        "lr":                 args.lr,
        "seconds":            args.seconds,
        "codebook_size":      args.codebook_size,
        "num_coarse_q":       args.num_coarse_q,
        "num_semantic_tokens": args.num_semantic_tokens,
        "dim":                args.dim,
        "depth":              args.depth,
        "heads":              args.heads,
        "flash_attn":         args.flash_attn,
        "has_condition":      args.has_condition,
        "save_every":         args.save_every,
        "sample_every":       args.sample_every,
        "wandb":              args.wandb,
        "seed":               args.seed,
    }

    if args.verbose or args.dry_run:
        _print_config("TRAIN-COARSE — Configuración", cfg)

    if args.dry_run:
        print("[dry-run] Construcción simulada. No se entrena.")
        return

    # Codec
    print(f"[train-coarse] Cargando codec desde {args.codec_ckpt} ...")
    soundstream = _build_soundstream(args)
    soundstream.load(args.codec_ckpt)

    # HuBERT
    print(f"[train-coarse] Cargando HuBERT desde {args.hubert_ckpt} ...")
    wav2vec = _load_hubert(args.hubert_ckpt, args.hubert_quantizer)

    # CoarseTransformer
    coarse_transformer = alm.CoarseTransformer(
        codebook_size          = args.codebook_size,
        num_coarse_quantizers  = args.num_coarse_q,
        num_semantic_tokens    = args.num_semantic_tokens,
        dim                    = args.dim,
        depth                  = args.depth,
        heads                  = args.heads,
        has_condition          = args.has_condition,
        flash_attn             = args.flash_attn,
    )

    print(f"[train-coarse] CoarseTransformer listo.")
    print(f"  Parámetros: {sum(p.numel() for p in coarse_transformer.parameters()):,}")
    print()

    trainer = alm.CoarseTransformerTrainer(
        transformer          = coarse_transformer,
        codec                = soundstream,
        wav2vec              = wav2vec,
        folder               = args.audio_dir,
        num_train_steps      = args.steps,
        batch_size           = args.batch_size,
        grad_accum_every     = args.grad_accum,
        lr                   = args.lr,
        data_max_length_seconds = args.seconds,
        results_folder       = args.results_dir,
        save_results_every   = args.sample_every,
        save_model_every     = args.save_every,
        random_split_seed    = args.seed,
        use_wandb_tracking   = args.wandb,
        force_clear_prev_results = False,
    )

    trainer.train()
    print(f"\n[train-coarse] Entrenamiento completado. "
          f"Checkpoints en: {args.results_dir}/")

# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: train-fine
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train_fine(args):
    """
    Entrena el FineTransformer (etapa 3).
    Solo necesita el codec (no HuBERT): refina los cuantizadores finos
    condicionado en los tokens gruesos ya generados.
    """
    torch = _import_torch()
    alm   = _import_audiolm()

    num_fine_q = args.rq_num_quantizers - args.num_coarse_q

    cfg = {
        "audio_dir":          args.audio_dir,
        "codec_ckpt":         args.codec_ckpt,
        "checkpoint":         args.checkpoint,
        "results_dir":        args.results_dir,
        "steps":              args.steps,
        "batch_size":         args.batch_size,
        "grad_accum":         args.grad_accum,
        "lr":                 args.lr,
        "seconds":            args.seconds,
        "codebook_size":      args.codebook_size,
        "num_coarse_q":       args.num_coarse_q,
        "num_fine_q":         num_fine_q,
        "dim":                args.dim,
        "depth":              args.depth,
        "heads":              args.heads,
        "flash_attn":         args.flash_attn,
        "has_condition":      args.has_condition,
        "save_every":         args.save_every,
        "sample_every":       args.sample_every,
        "wandb":              args.wandb,
        "seed":               args.seed,
    }

    if args.verbose or args.dry_run:
        _print_config("TRAIN-FINE — Configuración", cfg)

    if args.dry_run:
        print("[dry-run] Construcción simulada. No se entrena.")
        return

    # Codec
    print(f"[train-fine] Cargando codec desde {args.codec_ckpt} ...")
    soundstream = _build_soundstream(args)
    soundstream.load(args.codec_ckpt)

    # FineTransformer
    fine_transformer = alm.FineTransformer(
        codebook_size         = args.codebook_size,
        num_coarse_quantizers = args.num_coarse_q,
        num_fine_quantizers   = num_fine_q,
        dim                   = args.dim,
        depth                 = args.depth,
        heads                 = args.heads,
        has_condition         = args.has_condition,
        flash_attn            = args.flash_attn,
    )

    print(f"[train-fine] FineTransformer listo.")
    print(f"  Parámetros: {sum(p.numel() for p in fine_transformer.parameters()):,}")
    print(f"  Cuantizadores finos: {num_fine_q}  (de {args.rq_num_quantizers} totales)")
    print()

    trainer = alm.FineTransformerTrainer(
        transformer          = fine_transformer,
        codec                = soundstream,
        folder               = args.audio_dir,
        num_train_steps      = args.steps,
        batch_size           = args.batch_size,
        grad_accum_every     = args.grad_accum,
        lr                   = args.lr,
        data_max_length_seconds = args.seconds,
        results_folder       = args.results_dir,
        save_results_every   = args.sample_every,
        save_model_every     = args.save_every,
        random_split_seed    = args.seed,
        use_wandb_tracking   = args.wandb,
        force_clear_prev_results = False,
    )

    trainer.train()
    print(f"\n[train-fine] Entrenamiento completado. "
          f"Checkpoints en: {args.results_dir}/")

# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: generate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate(args):
    """
    Genera audio nuevo encadenando los tres transformers entrenados.

    Si se pasa --text, los transformers deben haber sido entrenados con
    --has-condition.  Si se pasa --prompt, el modelo continúa ese audio.
    """
    torch = _import_torch()
    alm   = _import_audiolm()

    num_fine_q = args.rq_num_quantizers - args.num_coarse_q

    cfg = {
        "codec_ckpt":         args.codec_ckpt,
        "semantic_ckpt":      args.semantic_ckpt,
        "coarse_ckpt":        args.coarse_ckpt,
        "fine_ckpt":          args.fine_ckpt,
        "hubert_ckpt":        args.hubert_ckpt,
        "hubert_quantizer":   args.hubert_quantizer,
        "use_encodec":        args.use_encodec,
        "text":               args.text or "(ninguno)",
        "prompt":             args.prompt or "(ninguno)",
        "output":             args.output,
        "batch_size":         args.batch_size,
        "max_length":         args.max_length,
        "codebook_size":      args.codebook_size,
        "rq_num_quantizers":  args.rq_num_quantizers,
        "num_coarse_q":       args.num_coarse_q,
        "num_semantic_tokens": args.num_semantic_tokens,
        "dim":                args.dim,
        "depth":              args.depth,
        "heads":              args.heads,
        "seed":               args.seed,
    }

    if args.verbose or args.dry_run:
        _print_config("GENERATE — Configuración", cfg)

    if args.dry_run:
        print("[dry-run] Pipeline de generación simulado. No se genera audio.")
        return

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[generate] CUDA no disponible — usando CPU (lento)\n")

    # ── Codec ─────────────────────────────────────────────────────────────────
    if args.use_encodec:
        print("[generate] Usando EnCodec preentrenado (24 kHz) ...")
        codec = alm.EncodecWrapper()
    else:
        print(f"[generate] Cargando SoundStream desde {args.codec_ckpt} ...")
        codec = _build_soundstream(args)
        codec.load(args.codec_ckpt)
    codec.eval()

    # ── HuBERT ────────────────────────────────────────────────────────────────
    print(f"[generate] Cargando HuBERT ...")
    wav2vec = _load_hubert(args.hubert_ckpt, args.hubert_quantizer)

    # ── SemanticTransformer ───────────────────────────────────────────────────
    print(f"[generate] Cargando SemanticTransformer desde {args.semantic_ckpt} ...")
    semantic_transformer = alm.SemanticTransformer(
        dim                  = args.dim,
        depth                = args.depth,
        heads                = args.heads,
        num_semantic_tokens  = args.num_semantic_tokens,
        has_condition        = bool(args.text),
        flash_attn           = args.flash_attn,
    )
    semantic_transformer.load(args.semantic_ckpt)

    # ── CoarseTransformer ─────────────────────────────────────────────────────
    print(f"[generate] Cargando CoarseTransformer desde {args.coarse_ckpt} ...")
    coarse_transformer = alm.CoarseTransformer(
        codebook_size          = args.codebook_size,
        num_coarse_quantizers  = args.num_coarse_q,
        num_semantic_tokens    = args.num_semantic_tokens,
        dim                    = args.dim,
        depth                  = args.depth,
        heads                  = args.heads,
        has_condition          = bool(args.text),
        flash_attn             = args.flash_attn,
    )
    coarse_transformer.load(args.coarse_ckpt)

    # ── FineTransformer ───────────────────────────────────────────────────────
    print(f"[generate] Cargando FineTransformer desde {args.fine_ckpt} ...")
    fine_transformer = alm.FineTransformer(
        codebook_size         = args.codebook_size,
        num_coarse_quantizers = args.num_coarse_q,
        num_fine_quantizers   = num_fine_q,
        dim                   = args.dim,
        depth                 = args.depth,
        heads                 = args.heads,
        has_condition         = bool(args.text),
        flash_attn            = args.flash_attn,
    )
    fine_transformer.load(args.fine_ckpt)

    # ── AudioLM ───────────────────────────────────────────────────────────────
    audiolm = alm.AudioLM(
        wav2vec              = wav2vec,
        codec                = codec,
        semantic_transformer = semantic_transformer,
        coarse_transformer   = coarse_transformer,
        fine_transformer     = fine_transformer,
    )

    print()
    print(f"[generate] Generando {args.batch_size} muestra(s), max_length={args.max_length} ...")

    generate_kwargs = dict(
        batch_size  = args.batch_size,
        max_length  = args.max_length,
    )
    if args.text:
        generate_kwargs["text"] = [args.text] * args.batch_size
        print(f"  Texto:  \"{args.text}\"")
    if args.prompt:
        generate_kwargs["prime_wave_path"] = args.prompt
        print(f"  Prompt: {args.prompt}")

    generated = audiolm(**generate_kwargs)  # (B, T)

    # Guardar
    sample_hz = codec.target_sample_hz if not args.use_encodec else 24000
    output_path = Path(args.output)

    if args.batch_size == 1:
        _save_wav(generated[0], str(output_path), sample_hz)
        print(f"\n[generate] Guardado: {output_path}")
    else:
        stem   = output_path.stem
        suffix = output_path.suffix or ".wav"
        parent = output_path.parent
        for i, wave in enumerate(generated):
            p = parent / f"{stem}_{i:02d}{suffix}"
            _save_wav(wave, str(p), sample_hz)
            print(f"  Guardado: {p}")
        print(f"\n[generate] {args.batch_size} ficheros generados.")

# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: continue  (alias de generate con --prompt obligatorio)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_continue(args):
    """
    Continúa un audio de referencia (audio prompting).
    Es exactamente generate con --prompt ya fijado como argumento posicional.
    """
    # El argumento posicional 'input' se mapea a args.prompt internamente
    args.prompt = args.input
    if not args.text:
        args.text = None
    cmd_generate(args)

# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: info
# ══════════════════════════════════════════════════════════════════════════════

def cmd_info(args):
    """
    Inspecciona un checkpoint .pt guardado por cualquiera de los trainers.
    Muestra versión, paso de entrenamiento, claves del estado y config si está disponible.
    """
    torch = _import_torch()
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        sys.exit(f"[ERROR] No existe el fichero: {ckpt_path}")

    print(f"[info] Cargando {ckpt_path} ...")
    pkg = torch.load(str(ckpt_path), map_location="cpu")

    print()
    print(f"  Fichero:  {ckpt_path}")
    print(f"  Tamaño:   {ckpt_path.stat().st_size / 1e6:.1f} MB")
    print()

    # Claves de primer nivel
    print("  Claves en el checkpoint:")
    for k, v in pkg.items():
        if isinstance(v, dict):
            print(f"    {k:<30} dict ({len(v)} entradas)")
        elif isinstance(v, str):
            print(f"    {k:<30} \"{v}\"")
        else:
            print(f"    {k:<30} {type(v).__name__}")

    # Versión y paso
    if "version" in pkg:
        print(f"\n  Versión audiolm-pytorch: {pkg['version']}")
    if "steps" in pkg:
        print(f"  Paso de entrenamiento:   {pkg['steps']}")

    # Contar parámetros si hay state_dict
    model_key = next((k for k in ("model", "ema_model") if k in pkg), None)
    if model_key:
        state = pkg[model_key]
        n_params = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
        print(f"  Parámetros ({model_key}):    {n_params:,}")
        if args.verbose:
            print(f"\n  Tensores en '{model_key}':")
            for name, tensor in list(state.items())[:30]:
                print(f"    {name:<60} {list(tensor.shape)}")
            if len(state) > 30:
                print(f"    ... y {len(state) - 30} más")

    # Config serializada (SoundStream la guarda en _configs)
    if "config" in pkg or "_configs" in pkg:
        config_key = "config" if "config" in pkg else "_configs"
    if config_key in pkg:
        import pickle
        try:
            config = pickle.loads(pkg[config_key])
            print("\n  Configuración del modelo (SoundStream):")
            for k, v in config.items():
                if k not in ("self", "__class__"):
                    print(f"    {k:<35} {v}")
        except Exception:
            print("  [_configs presente pero no deserializable]")

    print()

# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENTOS COMUNES (shared argument groups)
# ══════════════════════════════════════════════════════════════════════════════

def _add_training_args(p):
    """Argumentos comunes a todos los subcomandos de entrenamiento."""
    g = p.add_argument_group("entrenamiento")
    g.add_argument("--audio-dir",    required=True,
                   help="Carpeta con ficheros .wav / .flac / .mp3")
    g.add_argument("--checkpoint",   default="model.pt",
                   help="Ruta del checkpoint a guardar [model.pt]")
    g.add_argument("--results-dir",  default="./results",
                   help="Carpeta para samples y checkpoints periódicos [./results]")
    g.add_argument("--steps",        type=int,   default=100_000,
                   help="Pasos de entrenamiento [100000]")
    g.add_argument("--batch-size",   type=int,   default=4,
                   help="Tamaño de batch [4]")
    g.add_argument("--grad-accum",   type=int,   default=8,
                   help="Pasos de acumulación de gradiente (batch efectivo = batch×accum) [8]")
    g.add_argument("--lr",           type=float, default=3e-4,
                   help="Learning rate [3e-4]")
    g.add_argument("--seconds",      type=float, default=2.0,
                   help="Duración máxima de cada clip de entrenamiento en segundos [2.0]")
    g.add_argument("--save-every",   type=int,   default=1000,
                   help="Guardar checkpoint cada N pasos [1000]")
    g.add_argument("--sample-every", type=int,   default=500,
                   help="Generar muestra de validación cada N pasos [500]")
    g.add_argument("--wandb",        action="store_true",
                   help="Activar logging con Weights & Biases")
    g.add_argument("--seed",         type=int,   default=42,
                   help="Semilla aleatoria [42]")
    g.add_argument("--verbose",      action="store_true",
                   help="Mostrar configuración completa antes de entrenar")
    g.add_argument("--dry-run",      action="store_true",
                   help="Construir modelos y mostrar config sin entrenar")


def _add_codec_arch_args(p):
    """Arquitectura del codec (compartida entre train-codec, train-coarse, train-fine, generate)."""
    g = p.add_argument_group("arquitectura del codec")
    g.add_argument("--codebook-size",      type=int,   default=DEFAULT_CODEBOOK_SIZE,
                   help=f"Tamaño del codebook [{DEFAULT_CODEBOOK_SIZE}]")
    g.add_argument("--rq-num-quantizers",  type=int,   default=DEFAULT_RQ_NUM_QUANTIZERS,
                   help=f"Número total de cuantizadores residuales [{DEFAULT_RQ_NUM_QUANTIZERS}]")
    g.add_argument("--num-coarse-q",       type=int,   default=DEFAULT_NUM_COARSE_Q,
                   dest="num_coarse_q",
                   help=f"Cuantizadores gruesos (fine usa el resto) [{DEFAULT_NUM_COARSE_Q}]")
    g.add_argument("--rq-groups",          type=int,   default=1,
                   help="Grupos para multi-head RVQ [1]")
    g.add_argument("--use-lfq",            action="store_true",
                   help="Usar Lookup-Free Quantizer en vez de RVQ estándar")
    g.add_argument("--sample-hz",          type=int,   default=DEFAULT_TARGET_SAMPLE_HZ,
                   help=f"Frecuencia de muestreo objetivo [Hz] [{DEFAULT_TARGET_SAMPLE_HZ}]")
    g.add_argument("--attn-window",        type=int,   default=128,
                   help="Ventana de atención local en el bottleneck [128]")
    g.add_argument("--attn-depth",         type=int,   default=2,
                   help="Capas de atención local en el bottleneck [2]")


def _add_transformer_arch_args(p):
    """Arquitectura del transformer (compartida entre train-semantic/coarse/fine y generate)."""
    g = p.add_argument_group("arquitectura del transformer")
    g.add_argument("--dim",                type=int,   default=DEFAULT_DIM,
                   help=f"Dimensión del transformer [{DEFAULT_DIM}]")
    g.add_argument("--depth",              type=int,   default=DEFAULT_DEPTH,
                   help=f"Número de capas del transformer [{DEFAULT_DEPTH}]")
    g.add_argument("--heads",              type=int,   default=8,
                   help="Número de cabezas de atención [8]")
    g.add_argument("--flash-attn",         action="store_true",
                   help="Usar Flash Attention (requiere GPU Ampere+)")
    g.add_argument("--has-condition",      action="store_true",
                   help="Entrenar con conditioning de texto (T5)")
    g.add_argument("--num-semantic-tokens",type=int,   default=DEFAULT_NUM_SEMANTIC_TOKENS,
                   help=f"Número de tokens semánticos (= clusters KMeans de HuBERT) [{DEFAULT_NUM_SEMANTIC_TOKENS}]")


def _add_hubert_args(p, required=True):
    """Argumentos para HuBERT (compartidos entre train-semantic, train-coarse y generate)."""
    g = p.add_argument_group("HuBERT (tokenizador semántico)")
    g.add_argument("--hubert-ckpt",       required=required,
                   help="Ruta al checkpoint de HuBERT (hubert_base_ls960.pt)")
    g.add_argument("--hubert-quantizer",  required=required,
                   help="Ruta al quantizador KMeans (hubert_base_ls960_L9_km500.bin)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="audiolm.py",
        description="AudioLM v{} — Generación de audio por modelado de lenguaje".format(VERSION),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train-codec ───────────────────────────────────────────────────────────
    p_codec = sub.add_parser(
        "train-codec",
        help="Entrenar SoundStream (tokenizador neuronal de audio)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Etapa 0: entrena el codec neuronal SoundStream sobre un corpus de audio.\n"
            "Es el primer paso obligatorio. Genera el checkpoint que usarán\n"
            "train-coarse, train-fine y generate."
        ),
    )
    _add_training_args(p_codec)
    _add_codec_arch_args(p_codec)

    # ── train-semantic ────────────────────────────────────────────────────────
    p_sem = sub.add_parser(
        "train-semantic",
        help="Entrenar SemanticTransformer (etapa 1 del pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Etapa 1: entrena el SemanticTransformer sobre tokens semánticos\n"
            "extraídos con HuBERT. No necesita el codec."
        ),
    )
    _add_training_args(p_sem)
    _add_hubert_args(p_sem)
    _add_transformer_arch_args(p_sem)

    # ── train-coarse ──────────────────────────────────────────────────────────
    p_coarse = sub.add_parser(
        "train-coarse",
        help="Entrenar CoarseTransformer (etapa 2 del pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Etapa 2: entrena el CoarseTransformer. Requiere el codec entrenado\n"
            "y HuBERT para extraer tokens semánticos durante el entrenamiento."
        ),
    )
    _add_training_args(p_coarse)
    _add_codec_arch_args(p_coarse)
    _add_hubert_args(p_coarse)
    _add_transformer_arch_args(p_coarse)
    p_coarse.add_argument("--codec-ckpt", required=True,
                          help="Checkpoint del codec (de train-codec)")

    # ── train-fine ────────────────────────────────────────────────────────────
    p_fine = sub.add_parser(
        "train-fine",
        help="Entrenar FineTransformer (etapa 3 del pipeline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Etapa 3: entrena el FineTransformer. Solo necesita el codec;\n"
            "aprende a refinar los cuantizadores finos sin HuBERT."
        ),
    )
    _add_training_args(p_fine)
    _add_codec_arch_args(p_fine)
    _add_transformer_arch_args(p_fine)
    p_fine.add_argument("--codec-ckpt", required=True,
                        help="Checkpoint del codec (de train-codec)")

    # ── generate ──────────────────────────────────────────────────────────────
    p_gen = sub.add_parser(
        "generate",
        help="Generar audio con los tres modelos entrenados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Encadena SemanticTransformer → CoarseTransformer → FineTransformer\n"
            "para generar audio. Opcionalmente condicionado por texto (--text)\n"
            "o continuando desde un audio de referencia (--prompt)."
        ),
    )
    g_ckpt = p_gen.add_argument_group("checkpoints")
    g_ckpt.add_argument("--codec-ckpt",    required=False, default=None,
                        help="Checkpoint de SoundStream (omitir si --use-encodec)")
    g_ckpt.add_argument("--semantic-ckpt", required=True,
                        help="Checkpoint del SemanticTransformer")
    g_ckpt.add_argument("--coarse-ckpt",   required=True,
                        help="Checkpoint del CoarseTransformer")
    g_ckpt.add_argument("--fine-ckpt",     required=True,
                        help="Checkpoint del FineTransformer")
    _add_hubert_args(p_gen, required=True)
    _add_codec_arch_args(p_gen)
    _add_transformer_arch_args(p_gen)
    g_gen = p_gen.add_argument_group("generación")
    g_gen.add_argument("--text",        default=None,
                       help="Descripción textual del audio a generar (requiere --has-condition)")
    g_gen.add_argument("--prompt",      default=None, metavar="AUDIO",
                       help="Fichero de audio de referencia para continuar (audio prompting)")
    g_gen.add_argument("--output",      default="generated.wav",
                       help="Fichero WAV de salida [generated.wav]")
    g_gen.add_argument("--batch-size",  type=int,   default=1,
                       help="Número de muestras a generar en paralelo [1]")
    g_gen.add_argument("--max-length",  type=int,   default=2048,
                       help="Longitud máxima de la secuencia generada en tokens [2048]")
    g_gen.add_argument("--use-encodec", action="store_true",
                       help="Usar EnCodec preentrenado (24 kHz) en lugar de SoundStream")
    g_gen.add_argument("--seed",        type=int,   default=42,
                       help="Semilla aleatoria [42]")
    g_gen.add_argument("--verbose",     action="store_true")
    g_gen.add_argument("--dry-run",     action="store_true",
                       help="Mostrar configuración sin generar audio")

    # ── continue ──────────────────────────────────────────────────────────────
    p_cont = sub.add_parser(
        "continue",
        help="Continuar audio desde un fichero de referencia (audio prompting)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Versión directa de generate con un audio de referencia obligatorio.\n"
            "El modelo escucha los primeros segundos de 'input' y los continúa."
        ),
    )
    p_cont.add_argument("input", help="Fichero de audio de referencia (.wav)")
    g_cont_ckpt = p_cont.add_argument_group("checkpoints")
    g_cont_ckpt.add_argument("--codec-ckpt",    required=False, default=None)
    g_cont_ckpt.add_argument("--semantic-ckpt", required=True)
    g_cont_ckpt.add_argument("--coarse-ckpt",   required=True)
    g_cont_ckpt.add_argument("--fine-ckpt",     required=True)
    _add_hubert_args(p_cont, required=True)
    _add_codec_arch_args(p_cont)
    _add_transformer_arch_args(p_cont)
    g_cont_gen = p_cont.add_argument_group("generación")
    g_cont_gen.add_argument("--text",        default=None)
    g_cont_gen.add_argument("--output",      default="continued.wav")
    g_cont_gen.add_argument("--batch-size",  type=int,   default=1)
    g_cont_gen.add_argument("--max-length",  type=int,   default=2048)
    g_cont_gen.add_argument("--use-encodec", action="store_true")
    g_cont_gen.add_argument("--seed",        type=int,   default=42)
    g_cont_gen.add_argument("--verbose",     action="store_true")
    g_cont_gen.add_argument("--dry-run",     action="store_true")

    # ── info ──────────────────────────────────────────────────────────────────
    p_info = sub.add_parser(
        "info",
        help="Inspeccionar un checkpoint guardado",
        description="Muestra metadatos, parámetros y configuración de un checkpoint .pt.",
    )
    p_info.add_argument("checkpoint", help="Ruta al fichero .pt a inspeccionar")
    p_info.add_argument("--verbose", action="store_true",
                        help="Listar todos los tensores del state_dict")

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()

    {
        "train-codec":    cmd_train_codec,
        "train-semantic": cmd_train_semantic,
        "train-coarse":   cmd_train_coarse,
        "train-fine":     cmd_train_fine,
        "generate":       cmd_generate,
        "continue":       cmd_continue,
        "info":           cmd_info,
    }[args.command](args)


if __name__ == "__main__":
    main()
