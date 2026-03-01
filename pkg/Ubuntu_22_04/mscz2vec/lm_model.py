"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LM_MODEL  v1.0  —  VAE multi-instrumento CNN+GRU                          ║
║                                                                              ║
║  Arquitectura:                                                               ║
║    Encoder:     1D CNN temporal  →  GRU bidireccional  →  μ, σ (z ∈ ℝ^64) ║
║    Decoder:     z + BPM emb  →  GRU  →  1D CNN  →  tensor [T, D]          ║
║    Bottleneck:  VAE con reparameterization trick y KL annealing             ║
║                                                                              ║
║  Diseño para CPU:                                                            ║
║    ~2.5M parámetros, entrenamiento ~3-5 min/epoch en i7/i9 moderno         ║
║    Inferencia: < 1 segundo para generar una pieza                           ║
║                                                                              ║
║  USO STANDALONE (test de arquitectura):                                      ║
║    python lm_model.py                      # forward pass test              ║
║    python lm_model.py --summary            # resumen de parámetros         ║
║    python lm_model.py --benchmark          # velocidad forward/backward     ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch numpy                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import math
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
    from lm_repr import TENSOR_DIMS, DIMS_PER_FAMILY, FAMILIES, N_FAMILIES
except ImportError:
    print("ERROR: lm_repr.py debe estar en el mismo directorio.")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  HIPERPARÁMETROS POR DEFECTO
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CFG = {
    # Arquitectura
    'latent_dim':     64,    # dimensión del espacio latente z
    'cnn_channels':   [128, 256, 256],  # canales de cada capa CNN
    'cnn_kernels':    [5, 3, 3],        # kernel sizes CNN
    'gru_hidden':     256,   # hidden size del GRU
    'gru_layers':     2,     # capas GRU
    'dec_gru_hidden': 256,   # hidden size GRU decoder
    'dec_gru_layers': 2,
    'bpm_emb_dim':    16,    # embedding de BPM (input al decoder)
    'gru_dropout':    0.3,   # dropout en GRU (regularizacion contra overfitting)

    # Entrenamiento
    'beta_start':     0.0,   # KL weight inicial
    'beta_end':       0.1,   # KL weight final — muy conservador
    'beta_warmup':    30,    # epocas con beta=0 antes de empezar annealing
    'beta_epochs':    100,   # epocas de annealing tras el warmup
    'kl_free_bits':   2.0,   # free bits: no penalizar KL por debajo de este valor
    'kl_floor':       0.005, # KL minimo siempre activo (evita que KL llegue a 9)
    'w_density':      5.0,   # peso extra en loss para dims de densidad y onset
    'lr':             3e-4,
    'weight_decay':   1e-5,
    'max_frames':     256,
    'tensor_dims':    TENSOR_DIMS,
}


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODER: CNN temporal + GRU bidireccional → μ, σ
# ══════════════════════════════════════════════════════════════════════════════

class MidiEncoder(nn.Module):
    """
    Encoder del VAE.

    Input:  x  [B, T, D]   — tensor de representación MIDI
            bpm [B]         — BPM normalizado [0,1]
    Output: mu  [B, latent_dim]
            logvar [B, latent_dim]
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D   = cfg['tensor_dims']
        ch  = cfg['cnn_channels']
        ks  = cfg['cnn_kernels']
        gru_h = cfg['gru_hidden']
        gru_l = cfg['gru_layers']
        z_dim = cfg['latent_dim']
        bpm_d = cfg['bpm_emb_dim']

        # BPM embedding: escalar → vector
        self.bpm_proj = nn.Sequential(
            nn.Linear(1, bpm_d),
            nn.Tanh(),
        )

        # CNN 1D temporal: [B, D+bpm_d, T] → [B, ch[-1], T//stride]
        # Usamos padding='same' para mantener longitud y stride=2 en primera capa
        cnn_dropout = cfg.get('gru_dropout', 0.3) * 0.5  # mitad del GRU dropout
        cnn_layers = []
        in_ch = D + bpm_d
        for i, (out_ch, k) in enumerate(zip(ch, ks)):
            stride = 2 if i == 0 else 1
            pad    = k // 2
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(cnn_dropout),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # GRU bidireccional: captura estructura global
        gru_drop = cfg.get('gru_dropout', 0.3)
        self.gru = nn.GRU(
            input_size  = ch[-1],
            hidden_size = gru_h,
            num_layers  = gru_l,
            batch_first = True,
            bidirectional = True,
            dropout     = gru_drop if gru_l > 1 else 0.0,
        )

        # Proyección a μ y log σ²
        gru_out_dim = gru_h * 2   # bidireccional
        self.fc_mu     = nn.Linear(gru_out_dim, z_dim)
        self.fc_logvar = nn.Linear(gru_out_dim, z_dim)

        # Inicialización conservadora de logvar
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.constant_(self.fc_logvar.bias, -2.0)   # σ inicial pequeña

    def forward(self, x: torch.Tensor, bpm: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x:   [B, T, D]
        bpm: [B]
        → mu, logvar: [B, latent_dim]
        """
        B, T, D = x.shape

        # Concatenar BPM embedding a cada frame
        bpm_emb = self.bpm_proj(bpm.unsqueeze(-1))         # [B, bpm_d]
        bpm_exp = bpm_emb.unsqueeze(1).expand(-1, T, -1)   # [B, T, bpm_d]
        x = torch.cat([x, bpm_exp], dim=-1)                # [B, T, D+bpm_d]

        # CNN: necesita [B, C, T]
        x = x.permute(0, 2, 1)   # [B, D+bpm_d, T]
        x = self.cnn(x)           # [B, ch[-1], T']
        x = x.permute(0, 2, 1)   # [B, T', ch[-1]]

        # GRU
        _, h_n = self.gru(x)      # h_n: [num_layers*2, B, gru_h]
        # Tomar último layer, concatenar ambas direcciones
        h_fwd = h_n[-2]           # [B, gru_h]
        h_bwd = h_n[-1]           # [B, gru_h]
        h = torch.cat([h_fwd, h_bwd], dim=-1)   # [B, gru_h*2]

        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, -10.0, 2.0)   # estabilidad numérica

        return mu, logvar


# ══════════════════════════════════════════════════════════════════════════════
#  DECODER: z + BPM → GRU → CNN → tensor [T, D]
# ══════════════════════════════════════════════════════════════════════════════

class MidiDecoder(nn.Module):
    """
    Decoder del VAE.

    Input:  z   [B, latent_dim]  — punto en espacio latente
            bpm [B]               — BPM normalizado
            T   int               — longitud de secuencia a generar
    Output: x_rec [B, T, D]      — tensor reconstruido, valores en (0,1)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D     = cfg['tensor_dims']
        z_dim = cfg['latent_dim']
        gru_h = cfg['dec_gru_hidden']
        gru_l = cfg['dec_gru_layers']
        bpm_d = cfg['bpm_emb_dim']
        ch    = list(reversed(cfg['cnn_channels']))   # mirror del encoder
        ks    = list(reversed(cfg['cnn_kernels']))

        self.bpm_proj = nn.Sequential(
            nn.Linear(1, bpm_d),
            nn.Tanh(),
        )

        # Proyectar z + bpm a hidden state inicial del GRU
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim + bpm_d, gru_h * gru_l),
            nn.Tanh(),
        )

        # GRU decoder: genera secuencia frame a frame (unidireccional)
        gru_drop = cfg.get('gru_dropout', 0.3)
        self.gru = nn.GRU(
            input_size  = z_dim + bpm_d,
            hidden_size = gru_h,
            num_layers  = gru_l,
            batch_first = True,
            dropout     = gru_drop if gru_l > 1 else 0.0,
        )

        # CNN transpuesta para subir resolución temporal (mirror de encoder)
        cnn_layers = []
        in_ch = gru_h
        for i, (out_ch, k) in enumerate(zip(ch, ks)):
            if i == len(ch) - 1:
                # Última capa: upsampling ×2 (mirror del stride=2 del encoder)
                cnn_layers += [
                    nn.ConvTranspose1d(in_ch, out_ch, kernel_size=k,
                                       stride=2, padding=k//2, output_padding=1),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                ]
            else:
                pad = k // 2
                cnn_layers += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        # Proyeccion final a D dimensiones
        self.out_proj = nn.Conv1d(in_ch, D, kernel_size=1)

        # Skip connection: z se proyecta directamente a D y se suma al output.
        # Esto garantiza que el decoder siempre tiene acceso directo a z,
        # evitando que aprenda a ignorarlo y generar el promedio del corpus.
        self.skip_proj = nn.Linear(z_dim, D)

        self.gru_layers = gru_l
        self.gru_hidden = gru_h
        self.z_dim      = z_dim
        self.bpm_d      = bpm_d

    def forward(self, z: torch.Tensor, bpm: torch.Tensor,
                T: int) -> torch.Tensor:
        """
        z:   [B, latent_dim]
        bpm: [B]
        T:   longitud deseada del output
        → x_rec: [B, T, D]
        """
        B = z.shape[0]

        bpm_emb = self.bpm_proj(bpm.unsqueeze(-1))   # [B, bpm_d]
        z_bpm   = torch.cat([z, bpm_emb], dim=-1)    # [B, z_dim+bpm_d]

        # Estado inicial del GRU desde z
        h0 = self.z_proj(z_bpm)                              # [B, gru_h*layers]
        h0 = h0.view(B, self.gru_layers, self.gru_hidden)
        h0 = h0.permute(1, 0, 2).contiguous()                # [layers, B, gru_h]

        # Construir T frames de input (z_bpm repetido)
        # T_gru = T//2 porque la CNN luego hace ×2
        T_gru = max(1, T // 2)
        inp = z_bpm.unsqueeze(1).expand(-1, T_gru, -1)  # [B, T_gru, z_dim+bpm_d]

        out, _ = self.gru(inp, h0)    # [B, T_gru, gru_h]

        # CNN: necesita [B, C, T]
        x = out.permute(0, 2, 1)      # [B, gru_h, T_gru]
        x = self.cnn(x)               # [B, ch[0], T_gru*2] (upsampling en última capa)
        x = self.out_proj(x)          # [B, D, T']
        x = x.permute(0, 2, 1)        # [B, T', D]

        # Ajustar longitud exacta a T (recortar o rellenar)
        T_out = x.shape[1]
        if T_out > T:
            x = x[:, :T, :]
        elif T_out < T:
            pad = torch.zeros(B, T - T_out, x.shape[2], device=x.device)
            x = torch.cat([x, pad], dim=1)

        # Skip connection: z proyectado se suma (broadcast a todos los frames)
        skip = self.skip_proj(z)                    # [B, D]
        skip = skip.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]
        x = x + skip

        return torch.sigmoid(x)   # valores en (0,1)


# ══════════════════════════════════════════════════════════════════════════════
#  VAE COMPLETO
# ══════════════════════════════════════════════════════════════════════════════

class MultiInstrumentVAE(nn.Module):
    """
    VAE multi-instrumento completo.

    Combina encoder y decoder con reparameterization trick.
    Expone los métodos clave para entrenamiento e inferencia.

    Métodos principales:
        forward(x, bpm)           → x_rec, mu, logvar    (entrenamiento)
        encode(x, bpm)            → z                     (inferencia)
        decode(z, bpm, T)         → x_rec                 (inferencia)
        encode_midi_files(paths)  → z                     (desde ficheros MIDI)
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.encoder = MidiEncoder(self.cfg)
        self.decoder = MidiDecoder(self.cfg)
        self.latent_dim = self.cfg['latent_dim']

    # ── Reparameterization trick ──────────────────────────────────────────────
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor,
                       training: bool = True) -> torch.Tensor:
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # en eval: usar la media directamente

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor, bpm: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:   [B, T, D]
        bpm: [B]
        → x_rec [B, T, D], mu [B, Z], logvar [B, Z]
        """
        T = x.shape[1]
        mu, logvar = self.encoder(x, bpm)
        z = self.reparameterize(mu, logvar, self.training)
        x_rec = self.decoder(z, bpm, T)
        return x_rec, mu, logvar

    # ── Encode / Decode ───────────────────────────────────────────────────────
    @torch.no_grad()
    def encode(self, x: torch.Tensor, bpm: torch.Tensor) -> torch.Tensor:
        """x [B,T,D], bpm [B] → z [B, latent_dim]  (modo eval, sin grad)"""
        self.eval()
        mu, _ = self.encoder(x, bpm)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor, bpm: torch.Tensor,
               T: int | None = None) -> torch.Tensor:
        """z [B,Z], bpm [B] → x_rec [B,T,D]  (modo eval, sin grad)"""
        self.eval()
        T = T or self.cfg['max_frames']
        return self.decoder(z, bpm, T)

    # ── Interpolación en espacio latente ──────────────────────────────────────
    @torch.no_grad()
    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor,
                    alpha: float, bpm: torch.Tensor,
                    T: int | None = None) -> torch.Tensor:
        """
        Interpola entre dos puntos del espacio latente.
        alpha=0 → z1, alpha=1 → z2
        """
        z = (1 - alpha) * z1 + alpha * z2
        return self.decode(z, bpm, T)

    # ── Combinar múltiples referencias ────────────────────────────────────────
    @torch.no_grad()
    def combine_refs(self, zs: list[torch.Tensor],
                     weights: list[float] | None = None) -> torch.Tensor:
        """
        Combina N vectores latentes con pesos opcionales.
        zs: lista de tensores [1, Z] o [Z]
        weights: lista de floats (se normalizan a suma 1)
        → z_combined [1, Z]
        """
        if weights is None:
            weights = [1.0] * len(zs)
        w = torch.tensor(weights, dtype=torch.float32)
        w = w / w.sum()

        zs_stack = torch.stack([z.squeeze(0) if z.dim() > 1 else z
                                for z in zs], dim=0)   # [N, Z]
        z_comb = (w.unsqueeze(-1) * zs_stack).sum(dim=0, keepdim=True)  # [1, Z]
        return z_comb

    # ── Info del modelo ───────────────────────────────────────────────────────
    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        enc   = sum(p.numel() for p in self.encoder.parameters())
        dec   = sum(p.numel() for p in self.decoder.parameters())
        return {
            'total':   total,
            'encoder': enc,
            'decoder': dec,
        }

    def summary(self) -> str:
        params = self.count_parameters()
        lines = [
            "MultiInstrumentVAE",
            f"  Tensor input:   [B, {self.cfg['max_frames']}, {self.cfg['tensor_dims']}]",
            f"  Latent dim:     {self.cfg['latent_dim']}",
            f"  CNN channels:   {self.cfg['cnn_channels']}",
            f"  CNN kernels:    {self.cfg['cnn_kernels']}",
            f"  GRU enc:        {self.cfg['gru_layers']}L × {self.cfg['gru_hidden']}H (bidir)",
            f"  GRU dec:        {self.cfg['dec_gru_layers']}L × {self.cfg['dec_gru_hidden']}H",
            f"  BPM emb dim:    {self.cfg['bpm_emb_dim']}",
            f"",
            f"  Parámetros:",
            f"    Encoder:  {params['encoder']:>9,}",
            f"    Decoder:  {params['decoder']:>9,}",
            f"    Total:    {params['total']:>9,}  "
            f"({params['total']*4/1024/1024:.1f} MB en float32)",
        ]
        return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIÓN DE PÉRDIDA
# ══════════════════════════════════════════════════════════════════════════════

class VAELoss(nn.Module):
    """
    Loss del VAE mejorada: Reconstruccion ponderada + beta·KL con free bits.

    Mejoras sobre v1.0:
    - Pesos por tipo de dimension: densidad y onset pesan mas que chroma.
      Esto fuerza al modelo a aprender CUAL familia esta activa (estructura)
      antes de aprender QUE notas suenan (contenido).
    - Free bits: no se penaliza KL por debajo de kl_free_bits nats por dim.
      Evita el colapso del posterior sin necesitar annealing muy lento.
    - beta_end conservador (0.5): el modelo retiene mas informacion especifica
      de cada pieza en el espacio latente.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.beta_start   = cfg.get('beta_start',   0.0)
        self.beta_end     = cfg.get('beta_end',      0.1)
        self.beta_warmup  = cfg.get('beta_warmup',   30)
        self.beta_epochs  = cfg.get('beta_epochs',   100)
        self.kl_free_bits = cfg.get('kl_free_bits',  2.0)
        self.kl_floor     = cfg.get('kl_floor',      0.005)
        self.w_density    = cfg.get('w_density',     5.0)
        self.current_beta = self.beta_start

        # Construir vector de pesos [D] para la loss de reconstruccion.
        # dims de densidad (14) y onset (16) de cada familia pesan mas.
        # chroma (0-11), registro (12-13) y dinamica (15) pesan 1.0.
        w = torch.ones(TENSOR_DIMS)
        for fi in range(N_FAMILIES):
            base = fi * DIMS_PER_FAMILY
            w[base + 14] = self.w_density   # densidad
            w[base + 16] = self.w_density   # onset
        # Normalizar para que la loss total sea comparable a BCE estandar
        w = w / w.mean()
        self.register_buffer('dim_weights', w)   # [D]

    def update_beta(self, epoch: int):
        """Llamar al inicio de cada epoca.
        Fase 1 (0..beta_warmup): beta=0, solo reconstruccion.
        Fase 2 (beta_warmup..beta_warmup+beta_epochs): annealing lineal.
        Fase 3 (despues): beta=beta_end constante.
        """
        warmup = getattr(self, 'beta_warmup', 0)
        if epoch < warmup:
            self.current_beta = 0.0
        else:
            progress = min(1.0, (epoch - warmup) / max(self.beta_epochs, 1))
            self.current_beta = self.beta_start + progress * (self.beta_end - self.beta_start)

    def forward(self, x_rec: torch.Tensor, x: torch.Tensor,
                mu: torch.Tensor, logvar: torch.Tensor,
                ) -> tuple[torch.Tensor, dict]:
        """
        x_rec, x: [B, T, D]
        mu, logvar: [B, Z]
        -> loss_total, dict con componentes
        """
        B, T, D = x.shape

        # Reconstruccion ponderada: BCE con pesos por tipo de dim
        x_rec_c = torch.clamp(x_rec, 1e-7, 1 - 1e-7)
        # BCE elemento a elemento: [B, T, D]
        bce_elem = -(x * torch.log(x_rec_c) + (1 - x) * torch.log(1 - x_rec_c))
        # Aplicar pesos de dim: [D] broadcast sobre [B, T, D]
        bce_weighted = bce_elem * self.dim_weights.unsqueeze(0).unsqueeze(0)
        recon = bce_weighted.sum() / (B * T * D)

        # KL con free bits: max(kl_per_dim, free_bits) promediado
        # Cada dimension del espacio latente contribuye independientemente.
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, Z]
        kl_free    = torch.clamp(kl_per_dim, min=self.kl_free_bits)
        kl = kl_free.mean()

        # kl_floor: penalizacion KL minima siempre activa, incluso con beta=0.
        # Evita que el encoder aprenda un espacio latente sin ninguna regularizacion
        # durante el warmup, lo que causaria KL=9+ y colapso al iniciar annealing.
        kl_raw = kl_per_dim.mean()   # KL real sin free bits
        loss = recon + self.current_beta * kl + self.kl_floor * kl_raw

        return loss, {
            'loss':  loss.item(),
            'recon': recon.item(),
            'kl':    kl_per_dim.mean().item(),   # KL real (sin free bits) para logging
            'beta':  self.current_beta,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model: MultiInstrumentVAE, path: str,
               extra: dict | None = None):
    """Guarda modelo + configuración + estado extra (epoch, losses...)."""
    torch.save({
        'cfg':        model.cfg,
        'state_dict': model.state_dict(),
        'extra':      extra or {},
    }, path)


def load_model(path: str, map_location: str = 'cpu') -> tuple[MultiInstrumentVAE, dict]:
    """
    Carga modelo desde checkpoint.
    Retorna (model, extra_dict).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = MultiInstrumentVAE(ckpt['cfg'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt.get('extra', {})


# ══════════════════════════════════════════════════════════════════════════════
#  CLI / TEST
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='LM_MODEL: test de arquitectura y benchmarks')
    parser.add_argument('--summary',   action='store_true',
                        help='Mostrar resumen de parámetros')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark de velocidad forward/backward')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--T',          type=int, default=256,
                        help='Longitud de secuencia (frames)')
    parser.add_argument('--latent-dim', type=int, default=64)
    args = parser.parse_args()

    cfg = {**DEFAULT_CFG, 'latent_dim': args.latent_dim, 'max_frames': args.T}
    model = MultiInstrumentVAE(cfg)

    if args.summary or not args.benchmark:
        print(model.summary())
        print()

    B, T, D = args.batch_size, args.T, TENSOR_DIMS
    x   = torch.rand(B, T, D)
    bpm = torch.rand(B)

    # Test forward
    print(f"Forward pass ({B}×{T}×{D})...")
    t0 = time.time()
    x_rec, mu, logvar = model(x, bpm)
    fwd_ms = (time.time() - t0) * 1000
    print(f"  x_rec shape: {tuple(x_rec.shape)}")
    print(f"  mu shape:    {tuple(mu.shape)}")
    print(f"  Tiempo:      {fwd_ms:.1f} ms")
    assert x_rec.shape == (B, T, D), f"Shape incorrecta: {x_rec.shape}"
    assert x_rec.min() >= 0 and x_rec.max() <= 1, "Valores fuera de [0,1]"

    # Test loss
    loss_fn = VAELoss(cfg)
    loss_fn.update_beta(epoch=25)
    loss, components = loss_fn(x_rec, x, mu, logvar)
    print(f"\nLoss (beta={components['beta']:.2f}):")
    print(f"  Total: {components['loss']:.4f}")
    print(f"  Recon: {components['recon']:.4f}")
    print(f"  KL:    {components['kl']:.4f}")

    # Test encode/decode
    z = model.encode(x, bpm)
    print(f"\nEncode: z shape = {tuple(z.shape)}")
    x_dec = model.decode(z, bpm, T)
    print(f"Decode: x_dec shape = {tuple(x_dec.shape)}")

    # Test interpolación
    z1 = model.encode(x[:1], bpm[:1])
    z2 = model.encode(x[1:2], bpm[1:2])
    x_interp = model.interpolate(z1, z2, alpha=0.5, bpm=bpm[:1], T=T)
    print(f"Interpolate: shape = {tuple(x_interp.shape)}")

    # Test combinación de referencias
    z_combo = model.combine_refs([z1, z2], weights=[0.7, 0.3])
    print(f"Combine refs: shape = {tuple(z_combo.shape)}")

    if args.benchmark:
        print(f"\nBenchmark ({args.benchmark} = True)...")
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        model.train()

        n_iters = 20
        t0 = time.time()
        for _ in range(n_iters):
            x_b   = torch.rand(B, T, D)
            bpm_b = torch.rand(B)
            x_rec_b, mu_b, logvar_b = model(x_b, bpm_b)
            loss_b, _ = loss_fn(x_rec_b, x_b, mu_b, logvar_b)
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()
        elapsed = time.time() - t0

        ms_per_iter = elapsed / n_iters * 1000
        samples_per_sec = n_iters * B / elapsed
        print(f"  {n_iters} iters: {elapsed:.2f}s")
        print(f"  {ms_per_iter:.0f} ms/iter  ({samples_per_sec:.0f} muestras/s)")

        # Estimar tiempo de epoch
        n_train = 450   # ~90% de 532 MIDIs válidos
        iters_per_epoch = n_train // B
        sec_per_epoch = iters_per_epoch * ms_per_iter / 1000
        print(f"\n  Con {n_train} muestras, batch={B}:")
        print(f"  ~{iters_per_epoch} iters/epoch → {sec_per_epoch:.0f}s/epoch "
              f"({sec_per_epoch/60:.1f} min)")
        print(f"  100 épocas → {100*sec_per_epoch/3600:.1f} horas")
        print(f"  300 épocas → {300*sec_per_epoch/3600:.1f} horas")

    print("\n✓ Todos los tests pasados")


if __name__ == '__main__':
    main()
