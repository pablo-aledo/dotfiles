"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_GENERATE2  v1.0  —  Generación MIDI con PianoRollVAE                  ║
║                                                                              ║
║  Versión adaptada a lm_model2 + lm_repr2 (piano roll [T, 360]).            ║
║                                                                              ║
║  MODOS DE USO:                                                               ║
║                                                                              ║
║  1. Reconstrucción:                                                          ║
║     python lm_generate2.py --model best.pt --ref a.mid --mode reconstruct  ║
║                                                                              ║
║  2. Interpolación entre dos piezas:                                         ║
║     python lm_generate2.py --model best.pt \                                ║
║         --ref a.mid b.mid --alpha 0.5 --mode interpolate                   ║
║                                                                              ║
║  3. Combinación ponderada de N referencias:                                 ║
║     python lm_generate2.py --model best.pt \                                ║
║         --ref a.mid b.mid c.mid --weights 0.5 0.3 0.2 --mode combine      ║
║                                                                              ║
║  4. Exploración: variaciones con ruido en z:                                ║
║     python lm_generate2.py --model best.pt --ref a.mid \                   ║
║         --mode explore --n-variations 4 --noise 0.3                        ║
║                                                                              ║
║  5. Sweep de interpolación:                                                 ║
║     python lm_generate2.py --model best.pt --ref a.mid b.mid \             ║
║         --mode sweep --n-variations 7                                       ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch numpy mido                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import argparse
import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from lm_repr2 import (midi_to_roll, roll_to_midi,
                           get_roll_stats, TENSOR_DIMS2,
                           N_PITCHES, N_FAMILIES, PITCH_LO)
    from lm_repr  import FAMILIES, INSTR_TO_FAMILY
    from lm_model2 import load_model2, PianoRollVAE
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODER DE REFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

BPM_MIN = 40.0
BPM_MAX = 240.0

def _norm_bpm(bpm: float) -> float:
    return float(np.clip((bpm - BPM_MIN) / (BPM_MAX - BPM_MIN), 0, 1))


def encode_midi(model: PianoRollVAE, midi_path: str,
                frame_ms: int = 125, max_frames: int = 256
                ) -> tuple[torch.Tensor, float] | tuple[None, None]:
    """
    Carga un MIDI, lo convierte a piano roll y lo codifica al espacio latente.
    Retorna (z [1, Z], bpm) o (None, None) si falla.
    """
    roll, bpm = midi_to_roll(midi_path, frame_ms=frame_ms, max_frames=max_frames)
    if roll is None:
        print(f"  ERROR: no se pudo procesar {midi_path}")
        return None, None

    x     = torch.from_numpy(roll).unsqueeze(0)               # [1, T, 360]
    bpm_t = torch.tensor([_norm_bpm(bpm)], dtype=torch.float32)

    z = model.encode(x, bpm_t)    # [1, Z]
    return z, bpm


def encode_midis(model: PianoRollVAE, midi_paths: list[str],
                 frame_ms: int = 125, max_frames: int = 256
                 ) -> tuple[list[torch.Tensor], list[float]]:
    """Codifica múltiples MIDIs. Retorna (zs, bpms) — solo los válidos."""
    zs, bpms = [], []
    for path in midi_paths:
        z, bpm = encode_midi(model, path, frame_ms, max_frames)
        if z is not None:
            zs.append(z)
            bpms.append(bpm)
            print(f"  Codificado: {os.path.basename(path)}  "
                  f"BPM={bpm:.0f}  ||z||={z.norm().item():.2f}")
    return zs, bpms


# ══════════════════════════════════════════════════════════════════════════════
#  DECODIFICACIÓN A MIDI
# ══════════════════════════════════════════════════════════════════════════════

def decode_to_midi(model: PianoRollVAE,
                   z: torch.Tensor,
                   bpm: float,
                   output_path: str,
                   instruments: list[dict] | None = None,
                   T: int = 256,
                   frame_ms: int = 125) -> str:
    """
    Decodifica z a piano roll y genera un fichero MIDI.
    """
    bpm_t  = torch.tensor([_norm_bpm(bpm)], dtype=torch.float32)
    roll   = model.decode(z, bpm_t, T)           # [1, T, 360]
    roll_np = roll.squeeze(0).numpy()             # [T, 360]

    path = roll_to_midi(roll_np, bpm, instruments, output_path,
                        frame_ms=frame_ms,
                        min_peak=0.30, valley_ratio=0.40, max_note_frames=24)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MODOS DE GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def mode_reconstruct(model, args, instruments):
    if not args.ref:
        print("ERROR: --mode reconstruct requiere exactamente 1 --ref")
        return

    ref = args.ref[0]
    print(f"\nReconstruyendo: {ref}")
    z, bpm = encode_midi(model, ref, args.frame_ms, args.max_frames)
    if z is None:
        return

    bpm_out = args.bpm or bpm
    out = args.output or _auto_name(ref, 'reconstruct')
    decode_to_midi(model, z, bpm_out, out, instruments, args.max_frames, args.frame_ms)
    print(f"  → {out}  (BPM={bpm_out:.0f})")
    _print_stats(out, args.frame_ms, args.max_frames)


def mode_interpolate(model, args, instruments):
    if len(args.ref) < 2:
        print("ERROR: --mode interpolate requiere 2 --ref")
        return

    print(f"\nInterpolando: {os.path.basename(args.ref[0])} ←→ "
          f"{os.path.basename(args.ref[1])}  (α={args.alpha})")

    zs, bpms = encode_midis(model, args.ref[:2], args.frame_ms, args.max_frames)
    if len(zs) < 2:
        return

    z_interp = (1 - args.alpha) * zs[0] + args.alpha * zs[1]
    bpm_out   = args.bpm or ((1 - args.alpha) * bpms[0] + args.alpha * bpms[1])
    out = args.output or _auto_name(args.ref[0], f'interp_{args.alpha:.2f}')
    decode_to_midi(model, z_interp, bpm_out, out, instruments, args.max_frames, args.frame_ms)
    print(f"  → {out}  (BPM={bpm_out:.0f})")
    _print_stats(out, args.frame_ms, args.max_frames)


def mode_combine(model, args, instruments):
    if not args.ref:
        print("ERROR: --mode combine requiere al menos 1 --ref")
        return

    print(f"\nCombinando {len(args.ref)} referencias:")
    zs, bpms = encode_midis(model, args.ref, args.frame_ms, args.max_frames)
    if not zs:
        return

    if args.weights:
        w = np.array(args.weights[:len(zs)], dtype=np.float32)
    else:
        w = np.ones(len(zs), dtype=np.float32)
    w = w / w.sum()

    print(f"  Pesos normalizados: {[f'{wi:.3f}' for wi in w]}")

    z_combined = sum(float(wi) * zi for wi, zi in zip(w, zs))
    bpm_out    = args.bpm or float(np.dot(w, bpms))
    out = args.output or _auto_name(args.ref[0], 'combined')
    decode_to_midi(model, z_combined, bpm_out, out, instruments, args.max_frames, args.frame_ms)
    print(f"  → {out}  (BPM={bpm_out:.0f})")
    _print_stats(out, args.frame_ms, args.max_frames)


def mode_explore(model, args, instruments):
    if not args.ref:
        print("ERROR: --mode explore requiere 1 --ref")
        return

    ref = args.ref[0]
    print(f"\nExploring variations: {os.path.basename(ref)}  "
          f"noise={args.noise}  n={args.n_variations}")

    z, bpm = encode_midi(model, ref, args.frame_ms, args.max_frames)
    if z is None:
        return

    bpm_out = args.bpm or bpm
    base    = os.path.splitext(args.output or ref)[0]

    rng = np.random.default_rng(args.seed)
    for i in range(args.n_variations):
        noise = torch.from_numpy(
            rng.normal(0, args.noise, z.shape).astype(np.float32))
        z_var = z + noise
        out   = f"{base}_var{i+1:02d}.mid"
        decode_to_midi(model, z_var, bpm_out, out, instruments, args.max_frames, args.frame_ms)
        dist = float(noise.norm())
        print(f"  var{i+1:02d}: {out}  ||noise||={dist:.2f}")

    out0 = f"{base}_var00_original.mid"
    decode_to_midi(model, z, bpm_out, out0, instruments, args.max_frames, args.frame_ms)
    print(f"\n  Reconstrucción original: {out0}")


def mode_sweep(model, args, instruments):
    if len(args.ref) < 2:
        print("ERROR: --mode sweep requiere 2 --ref")
        return

    n = args.n_variations or 5
    print(f"\nSweep de {n} pasos: {os.path.basename(args.ref[0])} → "
          f"{os.path.basename(args.ref[1])}")

    zs, bpms = encode_midis(model, args.ref[:2], args.frame_ms, args.max_frames)
    if len(zs) < 2:
        return

    base = os.path.splitext(args.output or args.ref[0])[0]
    for i in range(n):
        alpha   = i / (n - 1) if n > 1 else 0.5
        z_step  = (1 - alpha) * zs[0] + alpha * zs[1]
        bpm_out = args.bpm or ((1 - alpha) * bpms[0] + alpha * bpms[1])
        out     = f"{base}_sweep{i+1:02d}_a{alpha:.2f}.mid"
        decode_to_midi(model, z_step, bpm_out, out, instruments, args.max_frames, args.frame_ms)
        print(f"  step{i+1:02d} α={alpha:.2f}: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _auto_name(ref: str, suffix: str) -> str:
    base = os.path.splitext(os.path.basename(ref))[0]
    return f"{base}_{suffix}.mid"


def _print_stats(midi_path: str, frame_ms: int = 125, max_frames: int = 256):
    """Muestra estadísticas del piano roll del MIDI generado."""
    roll, bpm = midi_to_roll(midi_path, frame_ms=frame_ms, max_frames=max_frames)
    if roll is None:
        return
    stats = get_roll_stats(roll)
    print(f"  Estadísticas del output (BPM={bpm:.0f}):")
    print(f"  {'Familia':<12}  {'Activo':>7}  {'Top nota':>8}  {'Densidad':>9}")
    for fam, s in stats['families'].items():
        bar = '█' * int(s['active_pct'] * 10)
        print(f"  {fam:<12}  {s['active_pct']*100:6.1f}%  "
              f"{s['top_name']:>8}  {s['density']:>9.2f}  {bar}")


def parse_instruments(instr_list: list[str]) -> list[dict] | None:
    """
    Convierte ['violin1', 'cello', 'trumpet'] a formato de orchestrator.py.
    """
    if not instr_list:
        return None

    ROLE_MAP = {
        'keys':       'accompaniment',
        'strings':    'melody',
        'bass':       'bass',
        'winds':      'melody',
        'percussion': 'percussion',
    }

    result = []
    for name in instr_list:
        family = INSTR_TO_FAMILY.get(name)
        if family is None:
            print(f"  AVISO: instrumento '{name}' no reconocido, usando role=melody")
            family = 'strings'
        role = ROLE_MAP.get(family, 'accompaniment')
        result.append({'name': name, 'role': role})
        print(f"  Instrumento: {name:<16} familia={family:<12} role={role}")

    return result


def load_instruments_json(path: str) -> list[dict] | None:
    try:
        with open(path) as f:
            data = json.load(f)
        if 'instruments' in data:
            return data['instruments']
        return data if isinstance(data, list) else None
    except Exception as e:
        print(f"  ERROR cargando JSON: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='LM_GENERATE2: generación MIDI con PianoRollVAE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Reconstruir una pieza
  python lm_generate2.py --model best.pt --ref cancion.mid

  # Mezcla 70/30 de dos piezas
  python lm_generate2.py --model best.pt --ref a.mid b.mid \\
      --weights 0.7 0.3 --instruments violin1 cello trumpet

  # 5 variaciones (exploración del espacio latente)
  python lm_generate2.py --model best.pt --ref a.mid \\
      --mode explore --n-variations 5 --noise 0.3

  # Sweep de interpolación en 7 pasos
  python lm_generate2.py --model best.pt --ref a.mid b.mid \\
      --mode sweep --n-variations 7
        """
    )

    # Modelo
    parser.add_argument('--model', required=True,
                        help='Ruta del checkpoint .pt')

    # Referencias
    parser.add_argument('--ref', nargs='+', default=[],
                        help='Fichero(s) MIDI de referencia')

    # Modo
    parser.add_argument('--mode', default='combine',
                        choices=['reconstruct', 'interpolate', 'combine',
                                 'explore', 'sweep'])

    # Parámetros de generación
    parser.add_argument('--alpha',        type=float, default=0.5)
    parser.add_argument('--weights',      type=float, nargs='+', default=None)
    parser.add_argument('--noise',        type=float, default=0.3)
    parser.add_argument('--n-variations', type=int,   default=4)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--bpm',          type=float, default=None)

    # Instrumentos
    parser.add_argument('--instruments', nargs='+', default=None)
    parser.add_argument('--palette',     default=None,
                        help='JSON con paleta de instrumentos')

    # Output
    parser.add_argument('--output',     default=None)
    parser.add_argument('--max-frames', type=int, default=256)
    parser.add_argument('--frame-ms',   type=int, default=125,
                        help='Frame ms (debe coincidir con el entrenamiento, default: 125)')

    # Info
    parser.add_argument('--list-instruments', action='store_true')

    args = parser.parse_args()

    if args.list_instruments:
        print("\nInstrumentos disponibles:")
        from collections import defaultdict
        by_family = defaultdict(list)
        for name, fam in sorted(INSTR_TO_FAMILY.items()):
            by_family[fam].append(name)
        for fam in FAMILIES:
            print(f"\n  [{fam}]")
            for name in sorted(by_family[fam]):
                print(f"    {name}")
        return

    # Cargar modelo
    print(f"\nCargando modelo: {args.model}")
    model, extra = load_model2(args.model)
    ep = extra.get('epoch', '?')
    vl = extra.get('val_loss', 0)
    print(f"  Época: {ep}  val_loss: {vl:.4f}")
    print(f"  Latent dim: {model.latent_dim}")
    print(f"  Input dims: {model.cfg.get('tensor_dims', TENSOR_DIMS2)}  (piano roll)")

    # Instrumentos
    instruments = None
    if args.palette:
        print(f"\nCargando paleta: {args.palette}")
        instruments = load_instruments_json(args.palette)
    elif args.instruments:
        print(f"\nPaleta de instrumentos:")
        instruments = parse_instruments(args.instruments)

    # Ejecutar modo
    dispatch = {
        'reconstruct': mode_reconstruct,
        'interpolate': mode_interpolate,
        'combine':     mode_combine,
        'explore':     mode_explore,
        'sweep':       mode_sweep,
    }
    dispatch[args.mode](model, args, instruments)


if __name__ == '__main__':
    main()
