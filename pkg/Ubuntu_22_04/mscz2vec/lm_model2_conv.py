"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_MODEL2  v1.0  —  VAE para piano roll [T, 360]                         ║
║                                                                              ║
║  Idéntico a lm_model.py en arquitectura (CNN+GRU) pero adaptado a:         ║
║    - Input [T, 360] en lugar de [T, 85]                                    ║
║    - Loss ponderada: errores en notas activas penalizados ×pos_weight       ║
║      Compensa el 98.9% de esparsidad del piano roll.                        ║
║    - Latent dim 128 (más capacidad para 360 dims de input)                 ║
║                                                                              ║
║  USO:                                                                        ║
║    python lm_model2.py --summary                                            ║
║    python lm_model2.py --benchmark                                          ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch numpy                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from lm_repr2 import TENSOR_DIMS2, N_PITCHES, N_FAMILIES
    from lm_repr  import FAMILIES
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  HIPERPARÁMETROS POR DEFECTO
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CFG2 = {
    # Arquitectura
    'tensor_dims':    TENSOR_DIMS2,          # 360
    'latent_dim':     128,
    'cnn_channels':   [256, 512, 512],
    'cnn_kernels':    [5, 3, 3],
    'gru_hidden':     512,
    'gru_layers':     2,
    'gru_dropout':    0.3,
    'dec_gru_hidden': 512,
    'dec_gru_layers': 2,
    'bpm_emb_dim':    16,
    'max_frames':     256,

    # Loss
    # pos_weight=20: con 99% de ceros, cada nota activa pesa ~17x más que un cero.
    # Valor teórico óptimo sería 99 (balanceo perfecto) pero desestabiliza.
    'pos_weight':     20.0,
    # beta muy bajo: este modelo es un autoencoder para interpolación,
    # no un generador VAE puro. Beta alto destruye la reconstrucción.
    'beta_start':     0.0,
    'beta_end':       0.02,   # era 0.1 — demasiado agresivo para piano roll
    'beta_warmup':    80,     # era 30 — esperar a que recon baje a ~0.09
    'beta_epochs':    80,     # annealing gradual tras el warmup
    'kl_free_bits':   4.0,    # era 2.0 — evita colapso del espacio latente
    'kl_floor':       0.001,  # era 0.005 — menos presión KL residual

    # Optimizador
    'lr':             3e-4,
    'weight_decay':   1e-5,
}


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class RollEncoder(nn.Module):
    """
    Encoder: [B, T, 360] → μ, σ  ∈  ℝ^latent_dim

    CNN 1D temporal (reduce T÷2) → GRU bidireccional → μ, logvar
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D     = cfg['tensor_dims']
        ch    = cfg['cnn_channels']
        ks    = cfg['cnn_kernels']
        gru_h = cfg['gru_hidden']
        gru_l = cfg['gru_layers']
        gru_d = cfg['gru_dropout']
        z_dim = cfg['latent_dim']
        bpm_d = cfg['bpm_emb_dim']

        self.bpm_proj = nn.Sequential(
            nn.Linear(1, bpm_d),
            nn.Tanh(),
        )

        # CNN: procesa [B, D+bpm_d, T]
        cnn_drop = gru_d * 0.5
        layers = []
        in_ch  = D + bpm_d
        for i, (out_ch, k) in enumerate(zip(ch, ks)):
            stride = 2 if i == 0 else 1
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k,
                          stride=stride, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(cnn_drop),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # GRU bidireccional
        self.gru = nn.GRU(
            input_size    = ch[-1],
            hidden_size   = gru_h,
            num_layers    = gru_l,
            batch_first   = True,
            bidirectional = True,
            dropout       = gru_d if gru_l > 1 else 0.0,
        )

        gru_out = gru_h * 2
        self.fc_mu     = nn.Linear(gru_out, z_dim)
        self.fc_logvar = nn.Linear(gru_out, z_dim)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)

    def forward(self, x: torch.Tensor, bpm: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        bpm_emb = self.bpm_proj(bpm.unsqueeze(-1))           # [B, bpm_d]
        bpm_exp = bpm_emb.unsqueeze(1).expand(-1, T, -1)     # [B, T, bpm_d]
        x = torch.cat([x, bpm_exp], dim=-1)                  # [B, T, D+bpm_d]
        x = x.permute(0, 2, 1)                               # [B, C, T]
        x = self.cnn(x)                                       # [B, ch[-1], T']
        x = x.permute(0, 2, 1)                               # [B, T', ch[-1]]
        _, h_n = self.gru(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)            # [B, gru_h*2]
        mu     = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10.0, 2.0)
        return mu, logvar


# ══════════════════════════════════════════════════════════════════════════════
#  DECODER
# ══════════════════════════════════════════════════════════════════════════════

class RollDecoder(nn.Module):
    """
    Decoder: z [B, Z] → piano roll [B, T, 360]

    z → GRU → CNN transpuesta → skip(z) → sigmoid
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D     = cfg['tensor_dims']
        z_dim = cfg['latent_dim']
        gru_h = cfg['dec_gru_hidden']
        gru_l = cfg['dec_gru_layers']
        gru_d = cfg['gru_dropout']
        ch    = list(reversed(cfg['cnn_channels']))
        ks    = list(reversed(cfg['cnn_kernels']))
        bpm_d = cfg['bpm_emb_dim']

        self.bpm_proj = nn.Sequential(nn.Linear(1, bpm_d), nn.Tanh())
        self.z_proj   = nn.Sequential(
            nn.Linear(z_dim + bpm_d, gru_h * gru_l),
            nn.Tanh(),
        )

        self.gru = nn.GRU(
            input_size  = z_dim + bpm_d,
            hidden_size = gru_h,
            num_layers  = gru_l,
            batch_first = True,
            dropout     = gru_d if gru_l > 1 else 0.0,
        )

        layers = []
        in_ch  = gru_h
        for i, (out_ch, k) in enumerate(zip(ch, ks)):
            if i == len(ch) - 1:
                layers += [
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k,
                                       stride=2, padding=k//2, output_padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                ]
            else:
                layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k//2),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                ]
            in_ch = out_ch
        self.cnn      = nn.Sequential(*layers)
        self.out_proj = nn.Conv1d(in_ch, D, kernel_size=1)
        self.skip_proj = nn.Linear(z_dim, D)   # skip connection directa z→D

        self.gru_layers = gru_l
        self.gru_hidden = gru_h
        self.z_dim      = z_dim
        self.bpm_d      = bpm_d

    def forward(self, z: torch.Tensor, bpm: torch.Tensor,
                T: int) -> torch.Tensor:
        B = z.shape[0]
        bpm_emb = self.bpm_proj(bpm.unsqueeze(-1))
        z_bpm   = torch.cat([z, bpm_emb], dim=-1)

        h0 = self.z_proj(z_bpm).view(B, self.gru_layers, self.gru_hidden)
        h0 = h0.permute(1, 0, 2).contiguous()

        T_gru = max(1, T // 2)
        inp   = z_bpm.unsqueeze(1).expand(-1, T_gru, -1)
        out, _ = self.gru(inp, h0)

        x = out.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.out_proj(x)
        x = x.permute(0, 2, 1)   # [B, T', D]

        # Ajustar longitud
        T_out = x.shape[1]
        if T_out > T:
            x = x[:, :T, :]
        elif T_out < T:
            x = torch.cat([x, torch.zeros(B, T - T_out, x.shape[2],
                                           device=x.device)], dim=1)

        # Skip connection: z influye directamente en cada frame
        skip = self.skip_proj(z).unsqueeze(1).expand(-1, T, -1)
        x = x + skip

        return torch.sigmoid(x)


# ══════════════════════════════════════════════════════════════════════════════
#  VAE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class PianoRollVAE(nn.Module):
    """
    VAE para piano roll multi-familia.

    Métodos clave:
        forward(roll, bpm)   → roll_rec, mu, logvar    (train)
        encode(roll, bpm)    → z                        (inference)
        decode(z, bpm, T)    → roll_rec                 (inference)
        interpolate(z1, z2, alpha, bpm, T)
        combine_refs(zs, weights)
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg     = {**DEFAULT_CFG2, **(cfg or {})}
        self.encoder = RollEncoder(self.cfg)
        self.decoder = RollDecoder(self.cfg)
        self.latent_dim = self.cfg['latent_dim']

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        if training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, roll: torch.Tensor, bpm: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = roll.shape[1]
        mu, logvar = self.encoder(roll, bpm)
        z = self.reparameterize(mu, logvar, self.training)
        roll_rec = self.decoder(z, bpm, T)
        return roll_rec, mu, logvar

    @torch.no_grad()
    def encode(self, roll: torch.Tensor, bpm: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, _ = self.encoder(roll, bpm)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor, bpm: torch.Tensor,
               T: int | None = None) -> torch.Tensor:
        self.eval()
        return self.decoder(z, bpm, T or self.cfg['max_frames'])

    @torch.no_grad()
    def interpolate(self, z1, z2, alpha, bpm, T=None):
        self.eval()
        return self.decode((1-alpha)*z1 + alpha*z2, bpm, T)

    @torch.no_grad()
    def combine_refs(self, zs, weights=None):
        if weights is None:
            weights = [1.0] * len(zs)
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()
        stack = torch.stack([z.squeeze(0) if z.dim() > 1 else z
                             for z in zs], dim=0)
        return (w.unsqueeze(-1) * stack).sum(0, keepdim=True)

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        enc   = sum(p.numel() for p in self.encoder.parameters())
        dec   = sum(p.numel() for p in self.decoder.parameters())
        return {'total': total, 'encoder': enc, 'decoder': dec}

    def summary(self) -> str:
        p = self.count_parameters()
        return '\n'.join([
            "PianoRollVAE",
            f"  Input:       [B, {self.cfg['max_frames']}, {self.cfg['tensor_dims']}]",
            f"  Latent dim:  {self.cfg['latent_dim']}",
            f"  CNN channels:{self.cfg['cnn_channels']}",
            f"  GRU enc:     {self.cfg['gru_layers']}L × {self.cfg['gru_hidden']}H (bidir)",
            f"  GRU dec:     {self.cfg['dec_gru_layers']}L × {self.cfg['dec_gru_hidden']}H",
            f"  pos_weight:  {self.cfg['pos_weight']}x  (penalización notas activas)",
            f"",
            f"  Parámetros:",
            f"    Encoder: {p['encoder']:>10,}",
            f"    Decoder: {p['decoder']:>10,}",
            f"    Total:   {p['total']:>10,}  ({p['total']*4/1024/1024:.1f} MB)",
        ])


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS PONDERADA PARA PIANO ROLL SPARSE
# ══════════════════════════════════════════════════════════════════════════════

class RollVAELoss(nn.Module):
    """
    Loss para piano roll sparse:

    Reconstrucción: BCE ponderada — las celdas con nota activa pesan
    pos_weight veces más que los silencios. Con 98.9% de ceros y
    pos_weight=10, el modelo no puede minimizar la loss prediciendo
    todo-cero (como haría con BCE uniforme).

    KL: con free bits y floor, igual que en lm_model.py.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.pos_weight   = cfg.get('pos_weight',   10.0)
        self.beta_start   = cfg.get('beta_start',    0.0)
        self.beta_end     = cfg.get('beta_end',      0.1)
        self.beta_warmup  = cfg.get('beta_warmup',   30)
        self.beta_epochs  = cfg.get('beta_epochs',  100)
        self.kl_free_bits = cfg.get('kl_free_bits',  2.0)
        self.kl_floor     = cfg.get('kl_floor',     0.005)
        self.current_beta = 0.0

        # Peso por celda: pos_weight donde hay nota, 1.0 donde no
        # Se aplica durante forward como tensor dinámico (depende del target)

    def update_beta(self, epoch: int):
        if epoch < self.beta_warmup:
            self.current_beta = 0.0
        else:
            prog = min(1.0, (epoch - self.beta_warmup) / max(self.beta_epochs, 1))
            self.current_beta = self.beta_start + prog * (self.beta_end - self.beta_start)

    def forward(self, roll_rec: torch.Tensor, roll: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        """
        roll_rec, roll: [B, T, D]
        mu, logvar:     [B, Z]
        """
        B, T, D = roll.shape

        # Máscara de peso: pos_weight donde la nota es activa, 1.0 donde no
        weight = torch.ones_like(roll)
        weight[roll > 0.05] = self.pos_weight

        # BCE ponderada elemento a elemento
        roll_c   = torch.clamp(roll_rec, 1e-7, 1 - 1e-7)
        bce_elem = -(roll * torch.log(roll_c) +
                     (1 - roll) * torch.log(1 - roll_c))
        recon = (bce_elem * weight).sum() / (B * T * D)

        # KL con free bits
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_free    = torch.clamp(kl_per_dim, min=self.kl_free_bits)
        kl         = kl_free.mean()
        kl_raw     = kl_per_dim.mean()

        loss = recon + self.current_beta * kl + self.kl_floor * kl_raw

        return loss, {
            'loss':  loss.item(),
            'recon': recon.item(),
            'kl':    kl_raw.item(),
            'beta':  self.current_beta,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_model2(model: PianoRollVAE, path: str, extra: dict | None = None):
    torch.save({
        'cfg':        model.cfg,
        'state_dict': model.state_dict(),
        'extra':      extra or {},
    }, path)


def load_model2(path: str, map_location: str = 'cpu'
                ) -> tuple[PianoRollVAE, dict]:
    ckpt  = torch.load(path, map_location=map_location, weights_only=False)
    model = PianoRollVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt.get('extra', {})


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='LM_MODEL2: test PianoRollVAE')
    parser.add_argument('--summary',    action='store_true')
    parser.add_argument('--benchmark',  action='store_true')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--T',          type=int, default=256)
    args = parser.parse_args()

    model   = PianoRollVAE()
    loss_fn = RollVAELoss(DEFAULT_CFG2)

    print(model.summary())
    print()

    B, T, D = args.batch_size, args.T, TENSOR_DIMS2
    roll = torch.rand(B, T, D) * 0.3   # sparse sintético
    roll[roll < 0.2] = 0.0              # ~80% ceros
    bpm  = torch.rand(B)

    print(f"Forward pass ({B}×{T}×{D})...")
    t0 = time.time()
    roll_rec, mu, logvar = model(roll, bpm)
    ms = (time.time() - t0) * 1000
    print(f"  roll_rec: {tuple(roll_rec.shape)}")
    print(f"  Tiempo:   {ms:.1f} ms")
    assert roll_rec.shape == (B, T, D)
    assert 0 <= roll_rec.min() and roll_rec.max() <= 1

    loss_fn.update_beta(epoch=50)
    loss, comps = loss_fn(roll_rec, roll, mu, logvar)
    print(f"\nLoss (beta={comps['beta']:.3f}, pos_weight={loss_fn.pos_weight}):")
    print(f"  Total: {comps['loss']:.4f}")
    print(f"  Recon: {comps['recon']:.4f}")
    print(f"  KL:    {comps['kl']:.4f}")

    z      = model.encode(roll, bpm)
    rec    = model.decode(z, bpm, T)
    interp = model.interpolate(z[:1], z[1:2], 0.5, bpm[:1], T)
    combo  = model.combine_refs([z[:1], z[1:2]], [0.7, 0.3])
    print(f"\nEncode:    z={tuple(z.shape)}")
    print(f"Decode:    {tuple(rec.shape)}")
    print(f"Interpolate: {tuple(interp.shape)}")
    print(f"Combine:   {tuple(combo.shape)}")

    if args.benchmark:
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        model.train()
        n = 20
        t0 = time.time()
        for _ in range(n):
            r = torch.rand(B, T, D) * 0.3
            r[r < 0.2] = 0.0
            b = torch.rand(B)
            r_rec, mu_b, lv_b = model(r, b)
            loss_b, _ = loss_fn(r_rec, r, mu_b, lv_b)
            optimizer.zero_grad(); loss_b.backward(); optimizer.step()
        elapsed = time.time() - t0
        ms_iter = elapsed / n * 1000
        sps     = n * B / elapsed
        n_train = 644
        iters_ep = n_train // B
        sec_ep   = iters_ep * ms_iter / 1000
        print(f"\nBenchmark: {ms_iter:.0f} ms/iter  ({sps:.0f} muestras/s)")
        print(f"  ~{sec_ep:.0f}s/época  ({sec_ep/60:.1f} min)")
        print(f"  100 épocas → {100*sec_ep/3600:.1f} h")

    print("\n✓ Todos los tests pasados")


if __name__ == '__main__':
    main()
