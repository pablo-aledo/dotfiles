"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_TRAIN2  v1.0  —  Loop de entrenamiento del VAE piano roll              ║
║                                                                              ║
║  Entrena PianoRollVAE ([T,360]) sobre tu corpus de MIDIs.                  ║
║  Diseñado para CPU: batch pequeño, checkpoints frecuentes, reanudable.     ║
║                                                                              ║
║  USO:                                                                        ║
║    # Entrenamiento básico                                                   ║
║    python lm_train2.py --midi-dir ./midis                                   ║
║                                                                              ║
║    # Con opciones                                                            ║
║    python lm_train.py --midi-dir ./midis                                \   ║
║        --epochs 300 --batch-size 16 --latent-dim 64                     \   ║
║        --checkpoint-dir ./checkpoints --save-every 25                       ║
║                                                                              ║
║    # Reanudar entrenamiento interrumpido                                    ║
║    python lm_train2.py --midi-dir ./midis --resume ./checkpoints2/last.pt  ║
║                                                                              ║
║    # Solo validar un checkpoint existente                                   ║
║    python lm_train2.py --midi-dir ./midis --eval-only ./checkpoints/best.pt║
║                                                                              ║
║  SALIDAS:                                                                    ║
║    checkpoints/best2.pt     — mejor modelo según val loss                  ║
║    checkpoints/last.pt      — último checkpoint (para reanudar)            ║
║    checkpoints/epochNNN.pt  — snapshots periódicos                         ║
║    training_log.csv         — loss por época (recon, KL, beta, total)      ║
║    training_curves.png      — gráfica de curvas (si matplotlib disponible) ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch numpy mido                                              ║
║    pip install matplotlib  (opcional, para curvas)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import csv
import time
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from lm_repr    import FAMILIES
    from lm_repr2   import TENSOR_DIMS2
    from lm_dataset2 import (MidiRollDataset, get_dataloader2,
                              build_cache2, load_cache2)
    from lm_model2  import (PianoRollVAE, RollVAELoss,
                             DEFAULT_CFG2, save_model2, load_model2)
except ImportError as e:
    print(f"ERROR: {e}\n  lm_repr.py, lm_dataset.py y lm_model.py deben estar en el mismo directorio.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """
    Registra métricas por época en CSV y muestra progreso en consola.
    Formato de log compatible con cualquier hoja de cálculo.
    """

    COLS = ['epoch', 'train_loss', 'train_recon', 'train_kl',
            'val_loss', 'val_recon', 'val_kl', 'beta', 'lr', 'epoch_sec']

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.rows: list[dict] = []
        self.best_val_loss = float('inf')

        # Escribir cabecera si el fichero no existe
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.COLS).writeheader()

    def log(self, metrics: dict):
        """Añade una fila al CSV y la muestra en consola."""
        row = {k: metrics.get(k, 0.0) for k in self.COLS}
        self.rows.append(row)

        # Escribir al CSV inmediatamente (por si se interrumpe)
        with open(self.log_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.COLS).writerow(row)

        # Marcar si es el mejor val_loss
        is_best = row['val_loss'] < self.best_val_loss
        if is_best:
            self.best_val_loss = row['val_loss']

        # Consola
        ep    = int(row['epoch'])
        t_l   = row['train_loss']
        v_l   = row['val_loss']
        recon = row['val_recon']
        kl    = row['val_kl']
        beta  = row['beta']
        lr    = row['lr']
        sec   = row['epoch_sec']
        best  = ' ★' if is_best else ''
        print(f"  Ep {ep:>4}  "
              f"train={t_l:.4f}  val={v_l:.4f}  "
              f"recon={recon:.4f}  KL={kl:.3f}  "
              f"β={beta:.3f}  lr={lr:.1e}  "
              f"{sec:.0f}s{best}")

        return is_best

    def load_existing(self):
        """Carga log existente (para continuar desde resume)."""
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append({k: float(v) if k != 'epoch' else int(float(v))
                                   for k, v in row.items()})
        if self.rows:
            val_losses = [r['val_loss'] for r in self.rows]
            self.best_val_loss = min(val_losses)

    def plot(self, out_path: str):
        """Genera gráfica de curvas de loss. Requiere matplotlib."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("  (matplotlib no disponible — skip plot)")
            return

        if not self.rows:
            return

        epochs    = [r['epoch']      for r in self.rows]
        train_l   = [r['train_loss'] for r in self.rows]
        val_l     = [r['val_loss']   for r in self.rows]
        recon_l   = [r['val_recon']  for r in self.rows]
        kl_l      = [r['val_kl']     for r in self.rows]
        beta_l    = [r['beta']       for r in self.rows]
        lr_l      = [r['lr']         for r in self.rows]

        fig = plt.figure(figsize=(14, 9))
        fig.suptitle('Curvas de entrenamiento — PianoRollVAE', fontsize=13)
        gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

        # Loss total
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, train_l, label='Train', color='#4e79a7')
        ax1.plot(epochs, val_l,   label='Val',   color='#e15759')
        best_ep = epochs[int(np.argmin(val_l))]
        ax1.axvline(best_ep, color='#e15759', ls='--', alpha=0.5,
                    label=f'Mejor val (ep {best_ep})')
        ax1.set_title('Loss total')
        ax1.set_xlabel('Época')
        ax1.legend(fontsize=8)
        ax1.set_yscale('log')

        # Reconstrucción vs KL
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, recon_l, label='Recon (val)', color='#59a14f')
        ax2_r = ax2.twinx()
        ax2_r.plot(epochs, kl_l, label='KL (val)', color='#f28e2b', ls='--')
        ax2.set_title('Reconstrucción vs KL divergence')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('BCE recon', color='#59a14f')
        ax2_r.set_ylabel('KL', color='#f28e2b')
        lines1, labs1 = ax2.get_legend_handles_labels()
        lines2, labs2 = ax2_r.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

        # Beta annealing
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(epochs, beta_l, color='#b07aa1')
        ax3.set_title('KL Weight (β annealing)')
        ax3.set_xlabel('Época')
        ax3.set_ylim(0, 1.1)

        # Learning rate
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.semilogy(epochs, lr_l, color='#76b7b2')
        ax4.set_title('Learning rate')
        ax4.set_xlabel('Época')

        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  Curvas guardadas: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  LOOP DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model: PianoRollVAE,
                loader,
                optimizer: torch.optim.Optimizer,
                loss_fn: RollVAELoss,
                grad_clip: float = 1.0) -> dict:
    """Una época de entrenamiento. Retorna dict con métricas medias."""
    model.train()
    totals = {'loss': 0., 'recon': 0., 'kl': 0.}
    n_batches = 0

    for batch in loader:
        x   = batch['roll']   # [B, T, D]
        bpm = batch['bpm']       # [B]

        x_rec, mu, logvar = model(x, bpm)
        loss, comps = loss_fn(x_rec, x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: estabiliza entrenamiento en CPU con GRU
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k in totals:
            totals[k] += comps[k]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model: PianoRollVAE,
              loader,
              loss_fn: RollVAELoss) -> dict:
    """Una época de validación sin gradientes."""
    model.eval()
    totals = {'loss': 0., 'recon': 0., 'kl': 0.}
    n_batches = 0

    for batch in loader:
        x   = batch['roll']
        bpm = batch['bpm']
        x_rec, mu, logvar = model(x, bpm)
        _, comps = loss_fn(x_rec, x, mu, logvar)
        for k in totals:
            totals[k] += comps[k]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    cache_path = args.cache or os.path.join(args.midi_dir, 'lm_cache2.npz')
    log_path   = os.path.join(args.checkpoint_dir, 'training_log.csv')
    plot_path  = os.path.join(args.checkpoint_dir, 'training_curves.png')
    best_path  = os.path.join(args.checkpoint_dir, 'best.pt')
    last_path  = os.path.join(args.checkpoint_dir, 'last.pt')

    print("\n" + "═"*65)
    print("  LM_TRAIN — PianoRollVAE")
    print("═"*65)

    # ── Datos ────────────────────────────────────────────────────────────────
    print("\n[1/4] Cargando datos...")
    if args.rebuild_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    aug_cfg = {
        'transpose_range': args.transpose_range,
        'dropout_p':       args.dropout_p,
        'reverse_p':       args.reverse_p,
        'aug_prob':        args.aug_prob,
    }

    ds_train = MidiRollDataset.from_midi_dir(
        args.midi_dir, mode='train',
        max_frames=args.max_frames, frame_ms=args.frame_ms,
        cache_path=cache_path, val_split=args.val_split,
        aug_config=aug_cfg)

    ds_val = MidiRollDataset.from_midi_dir(
        args.midi_dir, mode='val',
        max_frames=args.max_frames, frame_ms=args.frame_ms,
        cache_path=cache_path, val_split=args.val_split)

    loader_train = get_dataloader2(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val   = get_dataloader2(ds_val,   batch_size=args.batch_size, shuffle=False)

    print(f"  Train: {len(ds_train)} muestras  ({len(loader_train)} batches)")
    print(f"  Val:   {len(ds_val)}  muestras  ({len(loader_val)} batches)")
    print(f"  Familias activas train:")
    for fam, act in ds_train._family_activity.items():
        print(f"    {fam:<12} {act*100:.0f}%")

    # ── Modelo ───────────────────────────────────────────────────────────────
    print("\n[2/4] Inicializando modelo...")
    cfg = {
        **DEFAULT_CFG2,          # incluye kl_free_bits=1.0, w_density=5.0
        'latent_dim':     args.latent_dim,
        'gru_hidden':     args.gru_hidden,
        'gru_layers':     args.gru_layers,
        'beta_start':     args.beta_start,
        'beta_end':       args.beta_end,
        'beta_warmup':    args.beta_warmup,
        'beta_epochs':    args.beta_epochs,
        'lr':             args.lr,
        'max_frames':     args.max_frames,
        'weight_decay':   args.weight_decay,
        'tensor_dims':    TENSOR_DIMS2,
    }

    start_epoch = 0

    if args.resume:
        print(f"  Reanudando desde: {args.resume}")
        model, extra = load_model2(args.resume)
        start_epoch  = extra.get('epoch', 0) + 1
        print(f"  Época de inicio: {start_epoch}")
    else:
        model = PianoRollVAE(cfg)

    params = model.count_parameters()
    print(f"  Parámetros: {params['total']:,}  ({params['total']*4/1024/1024:.1f} MB)")
    print(f"  Latent dim: {cfg['latent_dim']}")

    # ── Optimizador y scheduler ───────────────────────────────────────────────
    optimizer = Adam(model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)

    if args.lr_schedule == 'cosine':
        # T_max = epocas restantes para que el cosine cubra todo el entrenamiento
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs - start_epoch, 1),
            eta_min=args.lr * 0.05)   # eta_min mas alto: no bajar tanto el LR
    elif args.lr_schedule == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=15, min_lr=args.lr * 0.01)
    else:
        scheduler = None

    # Restaurar estado del optimizador si se reanuda
    if args.resume:
        try:
            _, extra = load_model2(args.resume)
            if 'optimizer' in extra:
                optimizer.load_state_dict(extra['optimizer'])
        except Exception:
            pass

    loss_fn = RollVAELoss(cfg)
    logger  = TrainingLogger(log_path)
    if args.resume:
        logger.load_existing()

    # ── Estimación de tiempo ──────────────────────────────────────────────────
    print("\n[3/4] Estimando velocidad...")
    model.train()
    t0 = time.time()
    for i, batch in enumerate(loader_train):
        if i >= 3:
            break
        x_b, bpm_b = batch['roll'], batch['bpm']
        x_r, mu_b, lv_b = model(x_b, bpm_b)
        l, _ = loss_fn(x_r, x_b, mu_b, lv_b)
        optimizer.zero_grad(); l.backward(); optimizer.step()
    sec_3 = time.time() - t0
    sec_per_epoch = sec_3 / 3 * len(loader_train)
    remaining_epochs = args.epochs - start_epoch
    total_sec = sec_per_epoch * remaining_epochs
    h, m = divmod(int(total_sec), 3600)
    m //= 60
    print(f"  ~{sec_per_epoch:.0f}s/época  →  {remaining_epochs} épocas restantes  "
          f"≈ {h}h {m}min total")

    # ── Loop principal ────────────────────────────────────────────────────────
    print(f"\n[4/4] Entrenando {args.epochs} épocas (inicio: {start_epoch})...")
    print(f"  {'Ep':>4}  {'train':>8}  {'val':>8}  {'recon':>8}  "
          f"{'KL':>6}  {'β':>5}  {'lr':>8}  {'s':>4}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*6}  {'-'*5}  {'-'*8}  {'-'*4}")

    for epoch in range(start_epoch, args.epochs):
        t_ep = time.time()

        # Actualizar β
        loss_fn.update_beta(epoch)

        # Train + Val
        train_m = train_epoch(model, loader_train, optimizer, loss_fn,
                              grad_clip=args.grad_clip)
        val_m   = val_epoch(model, loader_val, loss_fn)

        # Learning rate step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if args.lr_schedule == 'plateau':
                scheduler.step(val_m['loss'])
            else:
                scheduler.step()

        ep_sec = time.time() - t_ep

        # Log
        metrics = {
            'epoch':      epoch,
            'train_loss': train_m['loss'],
            'train_recon':train_m['recon'],
            'train_kl':   train_m['kl'],
            'val_loss':   val_m['loss'],
            'val_recon':  val_m['recon'],
            'val_kl':     val_m['kl'],
            'beta':       loss_fn.current_beta,
            'lr':         current_lr,
            'epoch_sec':  ep_sec,
        }
        is_best = logger.log(metrics)

        # Guardar checkpoints
        extra = {
            'epoch':     epoch,
            'optimizer': optimizer.state_dict(),
            'val_loss':  val_m['loss'],
            'cfg':       cfg,
        }
        save_model2(model, last_path, extra)

        if is_best:
            save_model2(model, best_path, extra)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            snap_path = os.path.join(args.checkpoint_dir, f'epoch{epoch+1:04d}.pt')
            save_model2(model, snap_path, extra)

        # Plot periódico
        if args.plot_every > 0 and (epoch + 1) % args.plot_every == 0:
            logger.plot(plot_path)

        # Early stopping
        if args.early_stop > 0:
            recent = [r['val_loss'] for r in logger.rows[-args.early_stop:]]
            if len(recent) == args.early_stop and min(recent) >= logger.best_val_loss:
                print(f"\n  Early stopping: val_loss no mejora en {args.early_stop} épocas.")
                break

    # ── Resumen final ─────────────────────────────────────────────────────────
    logger.plot(plot_path)

    best_row = min(logger.rows, key=lambda r: r['val_loss'])
    print("\n" + "═"*65)
    print("  RESUMEN FINAL")
    print("═"*65)
    print(f"  Épocas entrenadas:  {len(logger.rows)}")
    print(f"  Mejor val_loss:     {best_row['val_loss']:.4f}  "
          f"(época {int(best_row['epoch'])})")
    print(f"  Mejor recon:        {best_row['val_recon']:.4f}")
    print(f"  Mejor KL:           {best_row['val_kl']:.4f}")
    print(f"  Checkpoint best:    {best_path}")
    print(f"  Log CSV:            {log_path}")
    print(f"  Curvas:             {plot_path}")
    print("═"*65)


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL ONLY: validar un checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def eval_only(args):
    """Carga un checkpoint y calcula métricas sobre el conjunto de validación."""
    print(f"\nEvaluando: {args.eval_only}")
    model, extra = load_model2(args.eval_only)
    print(f"  Checkpoint epoch: {extra.get('epoch', '?')}")
    print(f"  Val loss guardada: {extra.get('val_loss', '?'):.4f}")

    cache_path = args.cache or os.path.join(args.midi_dir, 'lm_cache2.npz')
    ds_val = MidiRollDataset.from_midi_dir(
        args.midi_dir, mode='val',
        max_frames=args.max_frames, frame_ms=args.frame_ms,
        cache_path=cache_path, val_split=args.val_split)
    loader_val = get_dataloader2(ds_val, batch_size=args.batch_size, shuffle=False)

    loss_fn = RollVAELoss(model.cfg)
    loss_fn.update_beta(9999)   # beta máximo para evaluación

    val_m = val_epoch(model, loader_val, loss_fn)
    print(f"\n  Métricas de validación (β=1.0):")
    print(f"    Loss total:  {val_m['loss']:.4f}")
    print(f"    Recon:       {val_m['recon']:.4f}")
    print(f"    KL:          {val_m['kl']:.4f}")

    # Muestrar reconstrucción de un ejemplo
    print(f"\n  Test de reconstrucción (1 muestra):")
    batch = next(iter(loader_val))
    x   = batch['roll'][:1]
    bpm = batch['bpm'][:1]
    with torch.no_grad():
        x_rec, mu, logvar = model(x, bpm)

    x_np   = x[0].numpy()
    rec_np = x_rec[0].numpy()
    mse    = float(((x_np - rec_np)**2).mean())
    corr   = float(np.corrcoef(x_np.flatten(), rec_np.flatten())[0, 1])
    print(f"    MSE:         {mse:.4f}")
    print(f"    Correlación: {corr:.4f}")
    print(f"    Fichero:     {batch['paths'][0]}")

    # Actividad por familia en la reconstrucción
    print(f"\n  Actividad por familia (original vs reconstruido):")
    from lm_repr2 import N_PITCHES as DIMS_PER_FAMILY, fam_pitch_idx
    IDX_DENSITY = N_PITCHES // 2  # aproximación para display
    from lm_repr2 import N_PITCHES as NP2
    print(f"  {'Familia':<12}  {'Original':>10}  {'Recon':>10}")
    for fi, fam in enumerate(FAMILIES):
        base = fi * NP2
        orig_act = float((x_np[:, base:base+NP2] > 0.05).mean())
        rec_act  = float((rec_np[:, base:base+NP2] > 0.05).mean())
        print(f"  {fam:<12}  {orig_act:>10.3f}  {rec_act:>10.3f}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='LM_TRAIN: entrenamiento del VAE latente multi-instrumento')

    # Datos
    parser.add_argument('--midi-dir',      required=True,
                        help='Directorio con ficheros .mid')
    parser.add_argument('--cache',         default=None,
                        help='Ruta del .npz de caché (default: midi_dir/lm_cache.npz)')
    parser.add_argument('--rebuild-cache', action='store_true')
    parser.add_argument('--max-frames',    type=int, default=256)
    parser.add_argument('--frame-ms',      type=int, default=125,
                        help='Ms por frame (default 125 para piano roll)')
    parser.add_argument('--val-split',     type=float, default=0.1)

    # Arquitectura
    parser.add_argument('--latent-dim',    type=int,   default=64)
    parser.add_argument('--gru-hidden',    type=int,   default=256)
    parser.add_argument('--gru-layers',    type=int,   default=2)

    # Entrenamiento
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--batch-size',    type=int,   default=16)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--weight-decay',  type=float, default=1e-5)
    parser.add_argument('--grad-clip',     type=float, default=1.0,
                        help='Gradient clipping (0 = desactivado)')
    parser.add_argument('--lr-schedule',   default='cosine',
                        choices=['cosine', 'plateau', 'none'])

    # KL annealing
    parser.add_argument('--beta-start',    type=float, default=0.0)
    parser.add_argument('--beta-end',      type=float, default=0.1,
                        help='KL weight maximo (default 0.1, muy conservador)')
    parser.add_argument('--beta-warmup',   type=int,   default=30,
                        help='Epocas con beta=0 antes de empezar annealing (default 30)')
    parser.add_argument('--beta-epochs',   type=int,   default=100,
                        help='Epocas de annealing tras el warmup (default 100)')

    # Augmentación
    parser.add_argument('--transpose-range',type=int,  default=6)
    parser.add_argument('--dropout-p',      type=float,default=0.2)
    parser.add_argument('--noise-sigma',    type=float,default=0.02)
    parser.add_argument('--reverse-p',      type=float,default=0.05)
    parser.add_argument('--aug-prob',        type=float,default=0.5,
                        help='Prob de aplicar augmentacion por muestra (default 0.5)')

    # Checkpoints
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--save-every',     type=int,  default=50,
                        help='Guardar snapshot cada N épocas (0 = desactivado)')
    parser.add_argument('--plot-every',     type=int,  default=25,
                        help='Generar plot cada N épocas (0 = solo al final)')
    parser.add_argument('--early-stop',     type=int,  default=30,
                        help='Parar si val_loss no mejora en N epocas (default: 30)')

    # Modos especiales
    parser.add_argument('--resume',    default=None,
                        help='Ruta de checkpoint para reanudar entrenamiento')
    parser.add_argument('--eval-only', default=None,
                        help='Evaluar checkpoint sin entrenar')

    args = parser.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
