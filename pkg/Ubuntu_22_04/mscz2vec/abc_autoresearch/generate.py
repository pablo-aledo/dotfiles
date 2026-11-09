"""
Genera piezas ABC a partir de un checkpoint guardado por abc_train.py (GPT
autorregresivo) o por abc_train_diffusion.py (difusión). La arquitectura se
detecta con el campo "arch" del checkpoint; los de abc_train.py no lo
tienen y se tratan como GPT.

Uso:
    uv run generate.py                     # genera 1 pieza y la reproduce
    uv run generate.py --n 5              # genera 5 piezas
    uv run generate.py --n 10 --only-valid # solo guarda las válidas
    uv run generate.py --temp 0.7         # temperatura más baja (más predecible)
    uv run generate.py --play             # reproduce la primera pieza válida
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from abc_model import GPT, GPTConfig
from abc_model import generate_abc_sample as generate_abc_sample_gpt
from abc_model import is_valid_abc
from abc_prepare import Tokenizer


def load_model(checkpoint_path: str):
    if not Path(checkpoint_path).exists():
        print(f"ERROR: no se encontró {checkpoint_path}", file=sys.stderr)
        print("Ejecuta primero: uv run abc_train.py", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["config"]
    arch = ckpt.get("arch", "gpt")  # checkpoints de abc_train.py no tienen "arch" -> gpt
    tokenizer = Tokenizer.from_directory()

    if arch == "diffusion":
        from abc_model_diffusion import DiffusionGPT, DiffusionConfig
        from abc_model_diffusion import generate_abc_sample as generate_abc_sample_diffusion
        config = DiffusionConfig(
            sequence_len  = cfg["sequence_len"],
            vocab_size    = cfg["vocab_size"],
            n_layer       = cfg["n_layer"],
            n_head        = cfg["n_head"],
            n_embd        = cfg["n_embd"],
            mask_token_id = cfg["mask_token_id"],
        )
        model = DiffusionGPT(config)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        sampler = generate_abc_sample_diffusion
    else:
        config = GPTConfig(
            sequence_len   = cfg["sequence_len"],
            vocab_size     = cfg["vocab_size"],
            n_layer        = cfg["n_layer"],
            n_head         = cfg["n_head"],
            n_embd         = cfg["n_embd"],
            window_pattern = cfg.get("window_pattern", "L"),
        )
        model = GPT(config)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        sampler = generate_abc_sample_gpt

    print(f"Modelo: arch={arch} depth={cfg['n_layer']} dim={cfg['n_embd']} "
          f"val_bpb={ckpt.get('val_bpb', '?')} step={ckpt.get('step', '?')}")
    return model, tokenizer, sampler


def abc_to_midi(abc_text: str, output_path: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".abc",
                                    delete=False, encoding="utf-8") as f:
        f.write(abc_text)
        abc_path = f.name
    try:
        result = subprocess.run(
            ["abc2midi", abc_path, "-o", output_path],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0 and Path(output_path).exists()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    finally:
        try: os.unlink(abc_path)
        except OSError: pass


def play_midi(midi_path: str):
    for player in ["timidity", "fluidsynth"]:
        if subprocess.run(["which", player], capture_output=True).returncode == 0:
            print(f"Reproduciendo con {player}...")
            subprocess.run([player, midi_path])
            return
    print("No se encontró timidity ni fluidsynth. Instala uno:")
    print("  sudo apt install timidity")


def main():
    parser = argparse.ArgumentParser(description="Genera música ABC desde checkpoint")
    parser.add_argument("--checkpoint", default="model_checkpoint.pt")
    parser.add_argument("--n",          type=int,   default=1,    help="Piezas a generar")
    parser.add_argument("--temp",       type=float, default=0.85, help="Temperatura")
    parser.add_argument("--max-tokens", type=int,   default=256,  help="Tokens máximos")
    parser.add_argument("--only-valid", action="store_true",      help="Solo piezas válidas")
    parser.add_argument("--out-dir",    default="./generated",    help="Directorio de salida")
    parser.add_argument("--play",       action="store_true",      help="Reproducir primera válida")
    parser.add_argument("--no-midi",    action="store_true",      help="No convertir a MIDI")
    args = parser.parse_args()

    model, tokenizer, generate_abc_sample = load_model(args.checkpoint)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    n_generated  = 0
    n_valid      = 0
    first_midi   = None
    max_attempts = args.n * 5 if args.only_valid else args.n
    attempt      = 0

    while n_generated < args.n and attempt < max_attempts:
        attempt += 1
        abc_text = generate_abc_sample(model, tokenizer,
                                        max_new_tokens=args.max_tokens,
                                        temperature=args.temp)
        valid = is_valid_abc(abc_text)
        if args.only_valid and not valid:
            continue

        n_generated += 1
        if valid:
            n_valid += 1

        print(f"\n{'='*60}")
        print(f"Pieza {n_generated}  {'✓ válida' if valid else '✗ inválida'}")
        print('='*60)
        print(abc_text)

        (out_dir / f"piece_{n_generated:03d}.abc").write_text(abc_text, encoding="utf-8")

        if not args.no_midi and valid:
            midi_path = str(out_dir / f"piece_{n_generated:03d}.mid")
            if abc_to_midi(abc_text, midi_path):
                print(f"MIDI: {midi_path}")
                if first_midi is None:
                    first_midi = midi_path
            else:
                print("abc2midi falló")

    print(f"\nGeneradas: {n_generated} | Válidas: {n_valid} | Guardadas en: {out_dir}/")

    if args.play and first_midi:
        play_midi(first_midi)
    elif args.play:
        print("No hay MIDI válido para reproducir.")


if __name__ == "__main__":
    main()
