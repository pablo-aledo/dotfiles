"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_GENERATE  v1.0  —  Generación MIDI con el autoencoder latente          ║
║                                                                              ║
║  El modelo es un autoencoder determinista: el espacio latente no sigue      ║
║  N(0,1) pero tiene estructura continua útil. Piezas similares están cerca. ║
║                                                                              ║
║  MODOS DE USO:                                                               ║
║                                                                              ║
║  1. Reconstrucción (test de fidelidad):                                     ║
║     python lm_generate.py --model best.pt --ref a.mid --mode reconstruct   ║
║                                                                              ║
║  2. Interpolación entre dos piezas:                                         ║
║     python lm_generate.py --model best.pt \                                 ║
║         --ref a.mid b.mid --alpha 0.5 --mode interpolate                   ║
║                                                                              ║
║  3. Combinación ponderada de N referencias:                                 ║
║     python lm_generate.py --model best.pt \                                 ║
║         --ref a.mid b.mid c.mid --weights 0.5 0.3 0.2 --mode combine      ║
║                                                                              ║
║  4. Exploración: variaciones con ruido en z:                                ║
║     python lm_generate.py --model best.pt --ref a.mid \                    ║
║         --mode explore --n-variations 4 --noise 0.3                        ║
║                                                                              ║
║  5. Paleta de instrumentos personalizada (compatible con orchestrator.py):  ║
║     python lm_generate.py --model best.pt --ref a.mid \                    ║
║         --instruments violin1 cello trumpet flute --mode reconstruct       ║
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
    from lm_repr  import (midi_to_tensor, tensor_to_midi_file,
                           get_tensor_stats, FAMILIES, INSTR_TO_FAMILY,
                           DIMS_PER_FAMILY, IDX_DENSITY)
    from lm_model import load_model, MultiInstrumentVAE
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODER DE REFERENCIA
# ══════════════════════════════════════════════════════════════════════════════

def encode_midi(model: MultiInstrumentVAE, midi_path: str,
                frame_ms: int = 500, max_frames: int = 256
                ) -> tuple[torch.Tensor, float] | tuple[None, None]:
    """
    Carga un MIDI y lo codifica al espacio latente.
    Retorna (z [1, Z], bpm) o (None, None) si falla.
    """
    tensor, bpm = midi_to_tensor(midi_path, frame_ms=frame_ms,
                                 max_frames=max_frames)
    if tensor is None:
        print(f"  ERROR: no se pudo procesar {midi_path}")
        return None, None

    # Rellenar a max_frames si es más corto
    if tensor.shape[0] < max_frames:
        pad = np.zeros((max_frames - tensor.shape[0], tensor.shape[1]),
                       dtype=np.float32)
        tensor = np.concatenate([tensor, pad], axis=0)

    x   = torch.from_numpy(tensor).unsqueeze(0)       # [1, T, D]
    bpm_t = torch.tensor([bpm / 200.0], dtype=torch.float32)  # normalizado

    z = model.encode(x, bpm_t)    # [1, Z]
    return z, bpm


def encode_midis(model: MultiInstrumentVAE, midi_paths: list[str],
                 frame_ms: int = 500, max_frames: int = 256
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

def decode_to_midi(model: MultiInstrumentVAE,
                   z: torch.Tensor,
                   bpm: float,
                   output_path: str,
                   instruments: list[dict] | None = None,
                   T: int = 256) -> str:
    """
    Decodifica z a tensor y genera un fichero MIDI.

    instruments: lista de dicts con 'name' y 'role', como acepta orchestrator.py.
                 Si None, usa los instrumentos por defecto de cada familia.
    """
    bpm_t = torch.tensor([bpm / 200.0], dtype=torch.float32)
    tensor = model.decode(z, bpm_t, T)           # [1, T, D]
    tensor_np = tensor.squeeze(0).numpy()         # [T, D]

    path = tensor_to_midi_file(tensor_np, bpm, instruments, output_path)
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  MODOS DE GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def mode_reconstruct(model, args, instruments):
    """Encodea una referencia y la reconstruye."""
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
    decode_to_midi(model, z, bpm_out, out, instruments, args.max_frames)
    print(f"  → {out}  (BPM={bpm_out:.0f})")
    _print_stats(out)


def mode_interpolate(model, args, instruments):
    """Interpola entre dos referencias con alpha."""
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
    decode_to_midi(model, z_interp, bpm_out, out, instruments, args.max_frames)
    print(f"  → {out}  (BPM={bpm_out:.0f})")
    _print_stats(out)


def mode_combine(model, args, instruments):
    """Combina N referencias con pesos opcionales."""
    if not args.ref:
        print("ERROR: --mode combine requiere al menos 1 --ref")
        return

    print(f"\nCombinando {len(args.ref)} referencias:")
    zs, bpms = encode_midis(model, args.ref, args.frame_ms, args.max_frames)
    if not zs:
        return

    # Pesos
    if args.weights:
        w = np.array(args.weights[:len(zs)], dtype=np.float32)
    else:
        w = np.ones(len(zs), dtype=np.float32)
    w = w / w.sum()

    print(f"  Pesos normalizados: {[f'{wi:.3f}' for wi in w]}")

    z_combined = sum(float(wi) * zi for wi, zi in zip(w, zs))
    bpm_out    = args.bpm or float(np.dot(w, bpms))
    out = args.output or _auto_name(args.ref[0], 'combined')
    decode_to_midi(model, z_combined, bpm_out, out, instruments, args.max_frames)
    print(f"  → {out}  (BPM={bpm_out:.0f})")
    _print_stats(out)


def mode_explore(model, args, instruments):
    """Genera variaciones añadiendo ruido gaussiano a z."""
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
        decode_to_midi(model, z_var, bpm_out, out, instruments, args.max_frames)
        dist = float(noise.norm())
        print(f"  var{i+1:02d}: {out}  ||noise||={dist:.2f}")

    print(f"\n  Reconstrucción original también disponible:")
    out0 = f"{base}_var00_original.mid"
    decode_to_midi(model, z, bpm_out, out0, instruments, args.max_frames)
    print(f"  → {out0}")


def mode_sweep(model, args, instruments):
    """
    Genera una secuencia de interpolaciones entre 2 referencias.
    Útil para explorar el espacio entre dos piezas visualmente.
    """
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
        decode_to_midi(model, z_step, bpm_out, out, instruments, args.max_frames)
        print(f"  step{i+1:02d} α={alpha:.2f}: {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _auto_name(ref: str, suffix: str) -> str:
    base = os.path.splitext(os.path.basename(ref))[0]
    return f"{base}_{suffix}.mid"


def _print_stats(midi_path: str):
    """Muestra estadísticas del tensor del MIDI generado."""
    tensor, bpm = midi_to_tensor(midi_path)
    if tensor is None:
        return
    stats = get_tensor_stats(tensor)
    print(f"  Estadísticas del output (BPM={bpm:.0f}):")
    print(f"  {'Familia':<12}  {'Activo':>7}  {'Top PC':>6}")
    for fam, s in stats['families'].items():
        bar = '█' * int(s['active_pct'] * 10)
        print(f"  {fam:<12}  {s['active_pct']*100:6.1f}%  "
              f"{s['top_pc']:>6}  {bar}")


def parse_instruments(instr_list: list[str]) -> list[dict] | None:
    """
    Convierte ['violin1', 'cello', 'trumpet'] a formato de orchestrator.py.
    Asigna roles automáticamente según la familia.
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
    """Carga paleta de instrumentos desde JSON (formato orchestrator.py)."""
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
        description='LM_GENERATE: generación MIDI con el autoencoder latente',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Reconstruir una pieza
  python lm_generate.py --model best.pt --ref cancion.mid

  # Mezcla 70/30 de dos piezas con paleta personalizada
  python lm_generate.py --model best.pt --ref a.mid b.mid \\
      --weights 0.7 0.3 --instruments violin1 cello trumpet

  # 5 variaciones de una pieza (exploración del espacio latente)
  python lm_generate.py --model best.pt --ref a.mid \\
      --mode explore --n-variations 5 --noise 0.3

  # Sweep de interpolación en 7 pasos
  python lm_generate.py --model best.pt --ref a.mid b.mid \\
      --mode sweep --n-variations 7
        """
    )

    # Modelo
    parser.add_argument('--model', required=True,
                        help='Ruta del checkpoint .pt (ej: ./checkpoints4/best.pt)')

    # Referencias
    parser.add_argument('--ref', nargs='+', default=[],
                        help='Fichero(s) MIDI de referencia')

    # Modo
    parser.add_argument('--mode', default='combine',
                        choices=['reconstruct', 'interpolate', 'combine',
                                 'explore', 'sweep'],
                        help='Modo de generación (default: combine)')

    # Parámetros de generación
    parser.add_argument('--alpha',   type=float, default=0.5,
                        help='Alpha para interpolación [0=ref1, 1=ref2] (default: 0.5)')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Pesos para combinación (se normalizan a suma 1)')
    parser.add_argument('--noise',   type=float, default=0.3,
                        help='Magnitud del ruido para --mode explore (default: 0.3)')
    parser.add_argument('--n-variations', type=int, default=4,
                        help='Número de variaciones para explore/sweep (default: 4)')
    parser.add_argument('--seed',    type=int, default=42,
                        help='Semilla para ruido aleatorio (default: 42)')
    parser.add_argument('--bpm',     type=float, default=None,
                        help='BPM del output (default: inferido de las referencias)')

    # Instrumentos
    parser.add_argument('--instruments', nargs='+', default=None,
                        help='Lista de instrumentos del output (ej: violin1 cello trumpet)')
    parser.add_argument('--palette', default=None,
                        help='JSON con paleta de instrumentos (formato orchestrator.py)')

    # Output
    parser.add_argument('--output', default=None,
                        help='Ruta del MIDI de salida (default: auto-generado)')
    parser.add_argument('--max-frames', type=int, default=256)
    parser.add_argument('--frame-ms',   type=int, default=500)

    # Info
    parser.add_argument('--list-instruments', action='store_true',
                        help='Listar instrumentos disponibles y salir')

    args = parser.parse_args()

    # Listar instrumentos disponibles
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
    model, extra = load_model(args.model)
    ep = extra.get('epoch', '?')
    vl = extra.get('val_loss', 0)
    print(f"  Época: {ep}  val_loss: {vl:.4f}")
    print(f"  Latent dim: {model.latent_dim}")

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
