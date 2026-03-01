"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_DATASET  v1.0  —  Dataset PyTorch para el modelo latente                ║
║                                                                              ║
║  Carga MIDIs, los convierte a tensores via lm_repr, aplica augmentation     ║
║  y los sirve al modelo en batches. Diseñado para entrenamiento en CPU.      ║
║                                                                              ║
║  AUGMENTACIONES:                                                             ║
║    [A] Transposición cromática aleatoria ±6 semitonos (en espacio chroma)  ║
║    [B] Stretch temporal ×0.75 – ×1.25 (interpolación de frames)            ║
║    [C] Dropout de familia: silencia familias completas con p=0.2            ║
║    [D] Ruido gaussiano suave sobre registro y dinámica                      ║
║    [E] Inversión temporal (el modelo aprende invarianza temporal)           ║
║                                                                              ║
║  USO STANDALONE (test del dataset):                                          ║
║    python lm_dataset.py --midi-dir ./midis                                  ║
║    python lm_dataset.py --midi-dir ./midis --profile  # benchmark de carga ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch numpy mido                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import random
import argparse
import numpy as np
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

# lm_repr debe estar en el mismo directorio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from lm_repr import (
        midi_to_tensor, TENSOR_DIMS, DIMS_PER_FAMILY,
        FAMILIES, N_FAMILIES, IDX_CHROMA_START, IDX_CHROMA_END,
        IDX_REG_MID, IDX_REG_RNG, IDX_DENSITY, IDX_DYNAMIC, IDX_ONSET,
    )
except ImportError:
    print("ERROR: lm_repr.py debe estar en el mismo directorio.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CACHÉ EN DISCO: evita re-procesar MIDIs en cada epoch
# ══════════════════════════════════════════════════════════════════════════════

def build_cache(midi_files: list[str], cache_path: str,
                max_frames: int = 256, frame_ms: int = 500,
                verbose: bool = True) -> str:
    """
    Procesa todos los MIDIs y guarda tensores + BPMs en un .npz.
    Solo hay que ejecutarlo una vez antes de entrenar.

    Retorna la ruta del fichero .npz creado.
    """
    tensors, bpms, paths = [], [], []
    total = len(midi_files)

    if verbose:
        print(f"  Construyendo caché: {total} MIDIs → {cache_path}")

    for i, f in enumerate(midi_files):
        tensor, bpm = midi_to_tensor(f, frame_ms=frame_ms, max_frames=max_frames)
        if tensor is not None:
            tensors.append(tensor)
            bpms.append(bpm)
            paths.append(f)
        if verbose and ((i + 1) % 100 == 0 or (i + 1) == total):
            print(f"  {i+1}/{total}  válidos: {len(tensors)}", end='\r')

    if verbose:
        print()

    tensors_arr = np.stack(tensors, axis=0).astype(np.float32)  # [N, T, D]
    bpms_arr    = np.array(bpms, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(cache_path,
                        tensors=tensors_arr,
                        bpms=bpms_arr,
                        paths=np.array(paths))

    if verbose:
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  Caché guardada: {len(tensors)} tensores, "
              f"{tensors_arr.shape}  →  {size_mb:.1f} MB")
    return cache_path


def load_cache(cache_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Carga el .npz de caché. Retorna (tensors [N,T,D], bpms [N], paths [N])."""
    data = np.load(cache_path, allow_pickle=True)
    return (data['tensors'].astype(np.float32),
            data['bpms'].astype(np.float32),
            list(data['paths']))


# ══════════════════════════════════════════════════════════════════════════════
#  AUGMENTACIONES
# ══════════════════════════════════════════════════════════════════════════════

def augment_transpose(tensor: np.ndarray, semitones: int) -> np.ndarray:
    """
    [A] Transpone ±semitones en espacio chroma (rotación circular).
    No cambia registro ni densidad — solo el contenido armónico.
    semitones puede ser negativo.
    """
    if semitones == 0:
        return tensor
    out = tensor.copy()
    for fam_idx in range(N_FAMILIES):
        base = fam_idx * DIMS_PER_FAMILY
        chroma = out[:, base + IDX_CHROMA_START : base + IDX_CHROMA_END]
        out[:, base + IDX_CHROMA_START : base + IDX_CHROMA_END] = np.roll(
            chroma, semitones, axis=1)
    return out


def augment_time_stretch(tensor: np.ndarray, factor: float) -> np.ndarray:
    """
    [B] Estira o comprime temporalmente interpolando frames.
    factor > 1 → más lento (más frames), factor < 1 → más rápido.
    El resultado se recorta/rellena a la longitud original.
    """
    if abs(factor - 1.0) < 0.01:
        return tensor
    T, D = tensor.shape
    new_T = max(1, int(T * factor))
    # Interpolación lineal a lo largo del eje temporal
    old_idx = np.linspace(0, T - 1, new_T)
    stretched = np.zeros((new_T, D), dtype=np.float32)
    for d in range(D):
        stretched[:, d] = np.interp(old_idx, np.arange(T), tensor[:, d])
    # Ajustar a T original
    if new_T >= T:
        return stretched[:T]
    else:
        pad = np.zeros((T - new_T, D), dtype=np.float32)
        return np.concatenate([stretched, pad], axis=0)


def augment_family_dropout(tensor: np.ndarray, p: float = 0.2,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """
    [C] Silencia familias completas con probabilidad p.
    Crucial para que el modelo aprenda a generar familias ausentes
    cuando la paleta de inferencia no las incluye todas.
    Nunca silencia todas las familias a la vez.
    """
    if rng is None:
        rng = np.random.default_rng()
    out = tensor.copy()
    silenced = 0
    for fam_idx in range(N_FAMILIES):
        if silenced >= N_FAMILIES - 1:
            break   # siempre deja al menos una familia activa
        if rng.random() < p:
            base = fam_idx * DIMS_PER_FAMILY
            out[:, base : base + DIMS_PER_FAMILY] = 0.0
            silenced += 1
    return out


def augment_noise(tensor: np.ndarray, sigma: float = 0.02,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """
    [D] Ruido gaussiano suave sobre registro y dinámica.
    No toca chroma (evita alterar la armonía).
    """
    if rng is None:
        rng = np.random.default_rng()
    out = tensor.copy()
    for fam_idx in range(N_FAMILIES):
        base = fam_idx * DIMS_PER_FAMILY
        for dim in [IDX_REG_MID, IDX_REG_RNG, IDX_DYNAMIC]:
            noise = rng.normal(0, sigma, size=tensor.shape[0]).astype(np.float32)
            out[:, base + dim] = np.clip(out[:, base + dim] + noise, 0.0, 1.0)
    return out


def augment_time_reverse(tensor: np.ndarray) -> np.ndarray:
    """[E] Invierte el orden temporal del tensor."""
    return tensor[::-1].copy()


def apply_augmentation(tensor: np.ndarray,
                        transpose_range: int = 6,
                        stretch_range: tuple[float,float] = (0.8, 1.2),
                        family_dropout_p: float = 0.2,
                        noise_sigma: float = 0.02,
                        time_reverse_p: float = 0.1,
                        seed: int | None = None) -> np.ndarray:
    """
    Aplica el pipeline completo de augmentación con probabilidades razonables.
    Llamado por el Dataset en cada __getitem__.
    """
    rng = np.random.default_rng(seed)

    # [A] Transposición: siempre, en rango ±transpose_range
    semitones = int(rng.integers(-transpose_range, transpose_range + 1))
    tensor = augment_transpose(tensor, semitones)

    # [B] Stretch temporal: 70% de las veces
    if rng.random() < 0.7:
        factor = float(rng.uniform(*stretch_range))
        tensor = augment_time_stretch(tensor, factor)

    # [C] Family dropout: siempre activo durante entrenamiento
    tensor = augment_family_dropout(tensor, p=family_dropout_p, rng=rng)

    # [D] Ruido: 80% de las veces
    if rng.random() < 0.8:
        tensor = augment_noise(tensor, sigma=noise_sigma, rng=rng)

    # [E] Inversión temporal: con baja probabilidad
    if rng.random() < time_reverse_p:
        tensor = augment_time_reverse(tensor)

    return tensor


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MidiLatentDataset(Dataset):
    """
    Dataset PyTorch para el VAE latente multi-instrumento.

    Cada ítem devuelve un dict:
        'tensor':   FloatTensor [T, D]   — representación compacta del MIDI
        'bpm':      FloatTensor []        — BPM normalizado [0,1] (60→0, 200→1)
        'path':     str                   — ruta del fichero original
        'idx':      int                   — índice en el dataset

    En modo train aplica augmentación aleatoria por ítem.
    En modo val/test devuelve el tensor limpio.

    Uso recomendado:
        dataset = MidiLatentDataset.from_cache('corpus.npz', mode='train')
        loader  = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    """

    BPM_MIN = 40.0
    BPM_MAX = 240.0

    def __init__(self,
                 tensors: np.ndarray,
                 bpms:    np.ndarray,
                 paths:   list[str],
                 mode:    str = 'train',
                 augment_config: dict | None = None):
        """
        tensors: [N, T, D] float32
        bpms:    [N]       float32
        paths:   [N]       str
        mode:    'train' | 'val' | 'test'
        augment_config: dict con parámetros de augmentación (None = defaults)
        """
        assert tensors.shape[0] == len(bpms) == len(paths), "Dimensiones inconsistentes"
        assert mode in ('train', 'val', 'test')

        self.tensors = tensors
        self.bpms    = bpms
        self.paths   = paths
        self.mode    = mode
        self.aug_cfg = augment_config or {}

        # Estadísticas del dataset para logging
        self.n_samples  = len(tensors)
        self.tensor_shape = tensors.shape[1:]   # (T, D)

        # Calcular actividad por familia para diagnóstico
        self._family_activity = self._compute_family_activity()

    @classmethod
    def from_cache(cls, cache_path: str, mode: str = 'train',
                   val_split: float = 0.1, seed: int = 42,
                   augment_config: dict | None = None) -> 'MidiLatentDataset':
        """
        Carga desde .npz y divide train/val automáticamente.
        Usa seed para reproducibilidad del split.
        """
        tensors, bpms, paths = load_cache(cache_path)
        N = len(tensors)

        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)
        n_val = max(1, int(N * val_split))

        if mode == 'train':
            idx = idx[n_val:]
        else:
            idx = idx[:n_val]

        return cls(tensors[idx], bpms[idx],
                   [paths[i] for i in idx], mode, augment_config)

    @classmethod
    def from_midi_dir(cls, midi_dir: str, mode: str = 'train',
                      max_frames: int = 256, frame_ms: int = 500,
                      cache_path: str | None = None,
                      val_split: float = 0.1, seed: int = 42,
                      augment_config: dict | None = None) -> 'MidiLatentDataset':
        """
        Construye el dataset directamente desde un directorio de MIDIs.
        Si cache_path existe lo carga; si no, lo construye y guarda.
        """
        midi_files = sorted([
            str(p) for p in Path(midi_dir).rglob('*.mid')
        ] + [
            str(p) for p in Path(midi_dir).rglob('*.midi')
        ])

        if not midi_files:
            raise ValueError(f"No se encontraron MIDIs en {midi_dir}")

        if cache_path is None:
            cache_path = os.path.join(midi_dir, 'lm_cache.npz')

        if not os.path.exists(cache_path):
            print(f"  Cache no encontrada. Construyendo desde {len(midi_files)} MIDIs...")
            build_cache(midi_files, cache_path, max_frames, frame_ms, verbose=True)

        return cls.from_cache(cache_path, mode, val_split, seed, augment_config)

    def _compute_family_activity(self) -> dict:
        """Fracción de muestras donde cada familia tiene contenido."""
        activity = {}
        for fam_idx, fam in enumerate(FAMILIES):
            base = fam_idx * DIMS_PER_FAMILY
            # Activo si la densidad media supera umbral
            density_col = self.tensors[:, :, base + IDX_DENSITY]
            active = (density_col.max(axis=1) > 0.01).mean()
            activity[fam] = float(active)
        return activity

    def normalize_bpm(self, bpm: float) -> float:
        """BPM → [0, 1] con clamp."""
        return float(np.clip((bpm - self.BPM_MIN) / (self.BPM_MAX - self.BPM_MIN), 0, 1))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        tensor = self.tensors[idx].copy()   # [T, D]
        bpm    = float(self.bpms[idx])
        path   = self.paths[idx]

        if self.mode == 'train':
            aug_seed = int(time.time() * 1000) % (2**31) ^ (idx * 2654435761)
            # Mixup de augmentacion: 50% de muestras pasan limpias.
            # Critico para que el encoder vea la distribucion real de MIDIs.
            rng_mix = np.random.default_rng(aug_seed % (2**31))
            if rng_mix.random() > self.aug_cfg.get('aug_prob', 0.5):
                tensor = apply_augmentation(
                    tensor,
                    transpose_range    = self.aug_cfg.get('transpose_range', 3),
                    stretch_range      = self.aug_cfg.get('stretch_range', (0.9, 1.1)),
                    family_dropout_p   = self.aug_cfg.get('family_dropout_p', 0.1),
                    noise_sigma        = self.aug_cfg.get('noise_sigma', 0.01),
                    time_reverse_p     = self.aug_cfg.get('time_reverse_p', 0.05),
                    seed               = aug_seed,
                )

        return {
            'tensor': torch.from_numpy(tensor.copy()),     # [T, D]
            'bpm':    torch.tensor(self.normalize_bpm(bpm), dtype=torch.float32),
            'path':   path,
            'idx':    idx,
        }

    def get_info(self) -> str:
        """Resumen legible del dataset."""
        lines = [
            f"MidiLatentDataset [{self.mode}]",
            f"  Muestras:   {self.n_samples}",
            f"  Tensor:     {self.tensor_shape}  (T×D = {self.tensor_shape[0]}×{self.tensor_shape[1]})",
            f"  BPM:        {self.bpms.min():.0f} – {self.bpms.max():.0f}  "
            f"(media {self.bpms.mean():.0f})",
            f"  Augment:    {'sí' if self.mode == 'train' else 'no'}",
            f"",
            f"  Actividad por familia:",
        ]
        for fam, act in self._family_activity.items():
            bar = '█' * int(act * 20)
            lines.append(f"    {fam:<12} {act*100:5.1f}%  {bar}")
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  COLLATE: cómo apilar batches
# ══════════════════════════════════════════════════════════════════════════════

def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function para DataLoader.
    Apila tensores y BPMs; paths e índices como listas.
    """
    return {
        'tensor': torch.stack([b['tensor'] for b in batch]),   # [B, T, D]
        'bpm':    torch.stack([b['bpm']    for b in batch]),   # [B]
        'paths':  [b['path'] for b in batch],
        'idxs':   [b['idx']  for b in batch],
    }


def get_dataloader(dataset: MidiLatentDataset,
                   batch_size: int = 32,
                   shuffle: bool | None = None,
                   num_workers: int = 0) -> DataLoader:
    """
    Construye un DataLoader con configuración apropiada para CPU.
    num_workers=0 es más rápido en CPU con datasets pequeños.
    """
    if shuffle is None:
        shuffle = (dataset.mode == 'train')
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        num_workers = num_workers,
        collate_fn  = collate_fn,
        pin_memory  = False,   # solo útil con GPU
        drop_last   = (dataset.mode == 'train'),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / TEST
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='LM_DATASET: test y benchmark del dataset PyTorch')
    parser.add_argument('--midi-dir',    required=True,
                        help='Directorio con ficheros .mid')
    parser.add_argument('--cache',       default=None,
                        help='Ruta del .npz de caché (default: midi_dir/lm_cache.npz)')
    parser.add_argument('--rebuild-cache', action='store_true',
                        help='Forzar reconstrucción de la caché aunque exista')
    parser.add_argument('--batch-size',  type=int, default=16)
    parser.add_argument('--max-frames',  type=int, default=256)
    parser.add_argument('--frame-ms',    type=int, default=500)
    parser.add_argument('--val-split',   type=float, default=0.1)
    parser.add_argument('--profile',     action='store_true',
                        help='Benchmark de velocidad de carga')
    parser.add_argument('--show-augment', action='store_true',
                        help='Mostrar efecto de augmentación en el primer ítem')
    args = parser.parse_args()

    cache_path = args.cache or os.path.join(args.midi_dir, 'lm_cache.npz')

    # Reconstruir caché si se pide
    if args.rebuild_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"  Caché eliminada: {cache_path}")

    # Construir datasets
    print("\nCargando dataset de entrenamiento...")
    t0 = time.time()
    ds_train = MidiLatentDataset.from_midi_dir(
        args.midi_dir, mode='train',
        max_frames=args.max_frames, frame_ms=args.frame_ms,
        cache_path=cache_path, val_split=args.val_split)
    ds_val = MidiLatentDataset.from_midi_dir(
        args.midi_dir, mode='val',
        max_frames=args.max_frames, frame_ms=args.frame_ms,
        cache_path=cache_path, val_split=args.val_split)
    print(f"  Carga: {time.time()-t0:.2f}s\n")

    print(ds_train.get_info())
    print()
    print(ds_val.get_info())
    print()

    # Test de un ítem
    item = ds_train[0]
    print(f"  Ítem[0]: tensor={tuple(item['tensor'].shape)}  "
          f"bpm={item['bpm']:.3f}  path={os.path.basename(item['path'])}")

    # Mostrar efecto de augmentación
    if args.show_augment:
        print("\n  Efecto de augmentación (10 variantes del ítem 0):")
        orig = ds_train.tensors[0]
        print(f"  {'':12} {'mean':>7} {'std':>7} {'nonzero':>9}")
        print(f"  {'original':12} {orig.mean():7.4f} {orig.std():7.4f} "
              f"{(orig>0).mean()*100:8.1f}%")
        for seed in range(10):
            aug = apply_augmentation(orig.copy(), seed=seed)
            print(f"  aug_{seed:<7}  {aug.mean():7.4f} {aug.std():7.4f} "
                  f"{(aug>0).mean()*100:8.1f}%")

    # Benchmark de DataLoader
    if args.profile:
        print(f"\n  Benchmark DataLoader (batch={args.batch_size})...")
        loader = get_dataloader(ds_train, batch_size=args.batch_size)
        n_batches = min(20, len(loader))
        t0 = time.time()
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
        elapsed = time.time() - t0
        samples_per_sec = n_batches * args.batch_size / elapsed
        print(f"  {n_batches} batches en {elapsed:.2f}s  "
              f"→  {samples_per_sec:.0f} muestras/s")
        print(f"  Batch tensor shape: {batch['tensor'].shape}")
        print(f"  Batch BPM shape:    {batch['bpm'].shape}")
        print(f"  1 epoch completa ({len(ds_train)} muestras): "
              f"≈ {len(ds_train)/samples_per_sec:.1f}s")


if __name__ == '__main__':
    main()
