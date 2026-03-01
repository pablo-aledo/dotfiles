"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_DATASET2  v1.0  —  Dataset PyTorch para piano roll exacto              ║
║                                                                              ║
║  Usa lm_repr2 (piano roll [T, 360]) en lugar de lm_repr (estadísticas).    ║
║  Permite reconstrucción fiel nota a nota.                                   ║
║                                                                              ║
║  AUGMENTACIONES adaptadas al piano roll:                                    ║
║    [A] Transposición: desplazamiento circular en el eje de pitch            ║
║    [B] Stretch temporal: interpolación nearest-neighbor (no mezcla notas)  ║
║    [C] Dropout de familia: silencia bloques completos de 72 pitches         ║
║    [D] Velocity jitter: escala la velocity ×U[0.8, 1.2] por familia        ║
║    [E] Inversión temporal                                                   ║
║                                                                              ║
║  USO:                                                                        ║
║    python lm_dataset2.py --midi-dir ./midis                                 ║
║    python lm_dataset2.py --midi-dir ./midis --profile --show-augment       ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch numpy mido                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from lm_repr2 import (
        midi_to_roll, roll_to_midi, get_roll_stats,
        TENSOR_DIMS2, N_PITCHES, PITCH_LO, PITCH_HI,
        fam_pitch_idx,
    )
    from lm_repr import FAMILIES, N_FAMILIES
except ImportError as e:
    print(f"ERROR: {e}\n  lm_repr.py y lm_repr2.py deben estar en el mismo directorio.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHÉ
# ══════════════════════════════════════════════════════════════════════════════

def build_cache2(midi_files: list[str], cache_path: str,
                 max_frames: int = 256, frame_ms: int = 125,
                 verbose: bool = True) -> str:
    """
    Procesa MIDIs → piano rolls y guarda en .npz.
    Ejecutar una vez antes de entrenar.
    """
    rolls, bpms, paths = [], [], []
    total = len(midi_files)

    if verbose:
        print(f"  Construyendo caché piano roll: {total} MIDIs → {cache_path}")

    for i, f in enumerate(midi_files):
        roll, bpm = midi_to_roll(f, frame_ms=frame_ms, max_frames=max_frames)
        if roll is not None:
            rolls.append(roll)
            bpms.append(bpm)
            paths.append(f)
        if verbose and ((i + 1) % 100 == 0 or (i + 1) == total):
            print(f"  {i+1}/{total}  válidos: {len(rolls)}", end='\r')

    if verbose:
        print()

    rolls_arr = np.stack(rolls, axis=0).astype(np.float32)   # [N, T, D]
    bpms_arr  = np.array(bpms, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(cache_path,
                        rolls=rolls_arr,
                        bpms=bpms_arr,
                        paths=np.array(paths))

    if verbose:
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  Caché: {len(rolls)} piano rolls {rolls_arr.shape}  →  {size_mb:.1f} MB")
    return cache_path


def load_cache2(cache_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(cache_path, allow_pickle=True)
    return (data['rolls'].astype(np.float32),
            data['bpms'].astype(np.float32),
            list(data['paths']))


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTACIONES PARA PIANO ROLL
# ══════════════════════════════════════════════════════════════════════════════

def aug_transpose(roll: np.ndarray, semitones: int) -> np.ndarray:
    """
    [A] Transpone ±semitones desplazando el eje de pitch dentro de cada familia.
    Las notas que salen del rango se descartan (no se envuelven — evita
    que una nota de percusión acabe en keys).
    """
    if semitones == 0:
        return roll
    out = np.zeros_like(roll)
    for fi in range(N_FAMILIES):
        base = fi * N_PITCHES
        src  = roll[:, base : base + N_PITCHES]   # [T, 72]
        if semitones > 0:
            out[:, base + semitones : base + N_PITCHES] = src[:, : N_PITCHES - semitones]
        else:
            s = -semitones
            out[:, base : base + N_PITCHES - s] = src[:, s:]
    return out


def aug_time_stretch(roll: np.ndarray, factor: float) -> np.ndarray:
    """
    [B] Stretch temporal usando nearest-neighbor (no mezcla velocidades de notas).
    """
    if abs(factor - 1.0) < 0.01:
        return roll
    T, D = roll.shape
    new_T = max(1, int(T * factor))
    # Índices nearest-neighbor
    src_idx = np.clip((np.arange(T) / factor).astype(int), 0, new_T - 1)
    # Crear versión estirada/comprimida de new_T frames y recortar/pad a T
    stretched = np.zeros((new_T, D), dtype=np.float32)
    for f_new in range(new_T):
        f_src = int(f_new / factor)
        if f_src < T:
            stretched[f_new] = roll[f_src]
    if new_T >= T:
        return stretched[:T]
    else:
        pad = np.zeros((T - new_T, D), dtype=np.float32)
        return np.concatenate([stretched, pad], axis=0)


def aug_family_dropout(roll: np.ndarray, p: float = 0.15,
                       rng: np.random.Generator | None = None) -> np.ndarray:
    """
    [C] Silencia familias completas con probabilidad p.
    Nunca silencia todas las familias activas.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = roll.copy()
    # Familias que tienen contenido
    active_fams = [fi for fi in range(N_FAMILIES)
                   if roll[:, fi*N_PITCHES:(fi+1)*N_PITCHES].max() > 0.05]
    if len(active_fams) <= 1:
        return out   # solo 1 familia activa — no silenciar
    silenced = 0
    for fi in range(N_FAMILIES):
        if silenced >= len(active_fams) - 1:
            break
        if rng.random() < p:
            out[:, fi*N_PITCHES : (fi+1)*N_PITCHES] = 0.0
            silenced += 1
    return out


def aug_velocity_jitter(roll: np.ndarray, scale_range: tuple = (0.8, 1.2),
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """
    [D] Escala la velocity de cada familia por un factor aleatorio.
    Solo afecta celdas activas — los ceros siguen siendo cero.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = roll.copy()
    for fi in range(N_FAMILIES):
        base  = fi * N_PITCHES
        scale = rng.uniform(*scale_range)
        block = out[:, base : base + N_PITCHES]
        out[:, base : base + N_PITCHES] = np.clip(block * scale, 0.0, 1.0)
    return out


def aug_time_reverse(roll: np.ndarray) -> np.ndarray:
    """[E] Inversión temporal."""
    return roll[::-1].copy()


def apply_augmentation2(roll: np.ndarray,
                         aug_prob:        float = 0.5,
                         transpose_range: int   = 4,
                         stretch_range:   tuple = (0.9, 1.1),
                         dropout_p:       float = 0.15,
                         vel_jitter:      tuple = (0.8, 1.2),
                         reverse_p:       float = 0.05,
                         seed: int | None = None) -> np.ndarray:
    """
    Pipeline completo. Con prob aug_prob aplica augmentación,
    con prob (1-aug_prob) devuelve el roll limpio.
    """
    rng = np.random.default_rng(seed)

    # Mixup: 50% de muestras pasan limpias
    if rng.random() > aug_prob:
        return roll

    # [A] Transposición
    semitones = int(rng.integers(-transpose_range, transpose_range + 1))
    roll = aug_transpose(roll, semitones)

    # [B] Stretch temporal (70% de las veces)
    if rng.random() < 0.7:
        factor = float(rng.uniform(*stretch_range))
        roll = aug_time_stretch(roll, factor)

    # [C] Family dropout
    roll = aug_family_dropout(roll, p=dropout_p, rng=rng)

    # [D] Velocity jitter (80% de las veces)
    if rng.random() < 0.8:
        roll = aug_velocity_jitter(roll, vel_jitter, rng=rng)

    # [E] Inversión temporal (baja probabilidad)
    if rng.random() < reverse_p:
        roll = aug_time_reverse(roll)

    return roll


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MidiRollDataset(Dataset):
    """
    Dataset PyTorch sobre piano rolls [T, 360].

    Cada ítem:
        'roll':  FloatTensor [T, 360]  — piano roll normalizado [0,1]
        'bpm':   FloatTensor []         — BPM normalizado [0,1]
        'path':  str
        'idx':   int
    """

    BPM_MIN = 40.0
    BPM_MAX = 240.0

    def __init__(self, rolls: np.ndarray, bpms: np.ndarray,
                 paths: list[str], mode: str = 'train',
                 aug_config: dict | None = None):
        assert rolls.shape[0] == len(bpms) == len(paths)
        assert mode in ('train', 'val', 'test')
        self.rolls  = rolls
        self.bpms   = bpms
        self.paths  = paths
        self.mode   = mode
        self.aug    = aug_config or {}
        self.n      = len(rolls)
        self._family_activity = self._compute_activity()

    @classmethod
    def from_cache(cls, cache_path: str, mode: str = 'train',
                   val_split: float = 0.1, seed: int = 42,
                   aug_config: dict | None = None) -> 'MidiRollDataset':
        rolls, bpms, paths = load_cache2(cache_path)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(rolls))
        n_val = max(1, int(len(rolls) * val_split))
        idx   = idx[n_val:] if mode == 'train' else idx[:n_val]
        return cls(rolls[idx], bpms[idx], [paths[i] for i in idx],
                   mode, aug_config)

    @classmethod
    def from_midi_dir(cls, midi_dir: str, mode: str = 'train',
                      max_frames: int = 256, frame_ms: int = 125,
                      cache_path: str | None = None,
                      val_split: float = 0.1, seed: int = 42,
                      aug_config: dict | None = None) -> 'MidiRollDataset':
        midi_files = sorted([
            str(p) for p in Path(midi_dir).rglob('*.mid')
        ] + [str(p) for p in Path(midi_dir).rglob('*.midi')])

        if not midi_files:
            raise ValueError(f"No se encontraron MIDIs en {midi_dir}")

        if cache_path is None:
            cache_path = os.path.join(midi_dir, 'lm_cache2.npz')

        if not os.path.exists(cache_path):
            build_cache2(midi_files, cache_path, max_frames, frame_ms)

        return cls.from_cache(cache_path, mode, val_split, seed, aug_config)

    def _compute_activity(self) -> dict:
        activity = {}
        for fi, fam in enumerate(FAMILIES):
            base  = fi * N_PITCHES
            block = self.rolls[:, :, base : base + N_PITCHES]
            active = (block.max(axis=(1, 2)) > 0.05).mean()
            activity[fam] = float(active)
        return activity

    def normalize_bpm(self, bpm: float) -> float:
        return float(np.clip((bpm - self.BPM_MIN) / (self.BPM_MAX - self.BPM_MIN), 0, 1))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        roll = self.rolls[idx].copy()
        bpm  = float(self.bpms[idx])

        if self.mode == 'train':
            seed = int(time.time() * 1000) % (2**31) ^ (idx * 2654435761)
            roll = apply_augmentation2(
                roll,
                aug_prob        = self.aug.get('aug_prob', 0.5),
                transpose_range = self.aug.get('transpose_range', 4),
                stretch_range   = self.aug.get('stretch_range', (0.9, 1.1)),
                dropout_p       = self.aug.get('dropout_p', 0.15),
                vel_jitter      = self.aug.get('vel_jitter', (0.8, 1.2)),
                reverse_p       = self.aug.get('reverse_p', 0.05),
                seed            = seed,
            )

        return {
            'roll': torch.from_numpy(roll.copy()),
            'bpm':  torch.tensor(self.normalize_bpm(bpm), dtype=torch.float32),
            'path': self.paths[idx],
            'idx':  idx,
        }

    def get_info(self) -> str:
        T, D = self.rolls.shape[1], self.rolls.shape[2]
        lines = [
            f"MidiRollDataset [{self.mode}]",
            f"  Muestras:   {self.n}",
            f"  Tensor:     [{T}, {D}]  (T×F×P = {T}×{N_FAMILIES}×{N_PITCHES})",
            f"  BPM:        {self.bpms.min():.0f} – {self.bpms.max():.0f}"
            f"  (media {self.bpms.mean():.0f})",
            f"  Sparsidad:  {(self.rolls < 0.05).mean()*100:.1f}% ceros",
            f"  Augment:    {'sí' if self.mode == 'train' else 'no'}",
            f"  Actividad por familia:",
        ]
        for fam, act in self._family_activity.items():
            bar = '█' * int(act * 20)
            lines.append(f"    {fam:<12} {act*100:5.1f}%  {bar}")
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  COLLATE Y DATALOADER
# ══════════════════════════════════════════════════════════════════════════════

def collate_fn2(batch: list[dict]) -> dict:
    return {
        'roll':  torch.stack([b['roll'] for b in batch]),
        'bpm':   torch.stack([b['bpm']  for b in batch]),
        'paths': [b['path'] for b in batch],
        'idxs':  [b['idx']  for b in batch],
    }


def get_dataloader2(dataset: MidiRollDataset,
                    batch_size: int = 16,
                    shuffle: bool | None = None,
                    num_workers: int = 0) -> DataLoader:
    if shuffle is None:
        shuffle = (dataset.mode == 'train')
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn2,
        pin_memory  = False,
        drop_last   = (dataset.mode == 'train'),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='LM_DATASET2: dataset piano roll para VAE')
    parser.add_argument('--midi-dir',     required=True)
    parser.add_argument('--cache',        default=None)
    parser.add_argument('--rebuild-cache',action='store_true')
    parser.add_argument('--batch-size',   type=int, default=16)
    parser.add_argument('--max-frames',   type=int, default=256)
    parser.add_argument('--frame-ms',     type=int, default=125)
    parser.add_argument('--val-split',    type=float, default=0.1)
    parser.add_argument('--profile',      action='store_true')
    parser.add_argument('--show-augment', action='store_true')
    args = parser.parse_args()

    cache = args.cache or os.path.join(args.midi_dir, 'lm_cache2.npz')
    if args.rebuild_cache and os.path.exists(cache):
        os.remove(cache)
        print(f"  Caché eliminada: {cache}")

    print("\nCargando dataset...")
    t0 = time.time()
    ds_train = MidiRollDataset.from_midi_dir(
        args.midi_dir, mode='train',
        max_frames=args.max_frames, frame_ms=args.frame_ms,
        cache_path=cache, val_split=args.val_split)
    ds_val = MidiRollDataset.from_midi_dir(
        args.midi_dir, mode='val',
        max_frames=args.max_frames, frame_ms=args.frame_ms,
        cache_path=cache, val_split=args.val_split)
    print(f"  Carga: {time.time()-t0:.2f}s\n")
    print(ds_train.get_info())
    print()
    print(ds_val.get_info())

    item = ds_train[0]
    print(f"\n  Ítem[0]: roll={tuple(item['roll'].shape)}  "
          f"bpm={item['bpm']:.3f}  "
          f"path={os.path.basename(item['path'])}")

    if args.show_augment:
        print("\n  Efecto de augmentación (8 variantes del ítem 0):")
        orig = ds_train.rolls[0]
        print(f"  {'':12} {'activas':>8} {'vel_med':>8} {'transp':>7}")
        orig_act = (orig > 0.05).mean()
        print(f"  {'original':12} {orig_act*100:7.2f}%  {orig[orig>0.05].mean() if orig_act > 0 else 0:8.3f}")
        for s in range(8):
            seed = s * 12345
            aug  = apply_augmentation2(orig.copy(), seed=seed)
            act  = (aug > 0.05).mean()
            vel  = aug[aug > 0.05].mean() if act > 0 else 0
            print(f"  aug_{s:<7}  {act*100:7.2f}%  {vel:8.3f}")

    if args.profile:
        print(f"\n  Benchmark DataLoader (batch={args.batch_size})...")
        loader = get_dataloader2(ds_train, batch_size=args.batch_size)
        n_batches = min(20, len(loader))
        t0 = time.time()
        for i, batch in enumerate(loader):
            if i >= n_batches: break
        elapsed = time.time() - t0
        sps = n_batches * args.batch_size / elapsed
        print(f"  {n_batches} batches en {elapsed:.2f}s  →  {sps:.0f} muestras/s")
        print(f"  Batch roll shape: {batch['roll'].shape}")
        print(f"  1 epoch ({len(ds_train)} muestras): ≈ {len(ds_train)/sps:.1f}s")


if __name__ == '__main__':
    main()
