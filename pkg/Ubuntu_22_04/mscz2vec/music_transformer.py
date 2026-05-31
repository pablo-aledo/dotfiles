#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     MUSIC TRANSFORMER  v1.0                                  ║
║   Generación de música con estructura a largo plazo (Huang et al., 2019)     ║
║                                                                              ║
║  Transformer autoregresivo decoder-only con atención relativa (Srel).        ║
║  Reduce la complejidad espacial de O(N²D) a O(ND) mediante el truco          ║
║  de skewing, permitiendo secuencias de hasta 2048 eventos MIDI.              ║
║                                                                              ║
║  REPRESENTACIÓN:                                                             ║
║    388 eventos: note-on(128) + note-off(128) + velocity(32) + time(100)      ║
║    + 3 tokens especiales: PAD=388, SOS=389, EOS=390  → vocab=391             ║
║                                                                              ║
║  DEPENDENCIA EXTERNA:                                                        ║
║    git clone https://github.com/jason9693/midi-neural-processor.git          ║
║    mv midi-neural-processor midi_processor                                   ║
║    (debe estar en el mismo directorio que este fichero)                      ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    preprocess   — MIDI corpus → pickles de eventos                           ║
║    train        — Entrenar el modelo (desde cero o retomando)                ║
║    generate     — Generar MIDI desde un checkpoint                           ║
║    inspect      — Diagnóstico: dataset, checkpoint, config                   ║
║                                                                              ║
║  FLUJO TÍPICO:                                                               ║
║    # 1. Preparar datos                                                       ║
║    python music_transformer.py preprocess --input midis/ --output data/      ║
║                                                                              ║
║    # 2. Entrenar                                                              ║
║    python music_transformer.py train --data data/ --model-dir runs/exp1/     ║
║                                                                              ║
║    # 3. Generar                                                               ║
║    python music_transformer.py generate --model-dir runs/exp1/ \             ║
║        --output generated.mid --length 1024                                  ║
║                                                                              ║
║    # Retomar entrenamiento                                                    ║
║    python music_transformer.py train --data data/ --model-dir runs/exp1/ \   ║
║        --resume                                                              ║
║                                                                              ║
║    # Generar con prime (primeros 200 eventos de un MIDI existente)           ║
║    python music_transformer.py generate --model-dir runs/exp1/ \             ║
║        --prime midis/tema.mid --prime-events 200 --length 1024               ║
║                                                                              ║
║  OPCIONES PRINCIPALES (train):                                               ║
║    --epochs N        Épocas de entrenamiento (default: 300)                  ║
║    --batch-size N    Tamaño de batch (default: 4)                            ║
║    --max-seq N       Longitud máxima de secuencia (default: 2048)            ║
║    --layers N        Número de capas Transformer (default: 6)                ║
║    --dim N           Dimensión de embeddings (default: 256)                  ║
║    --heads N         Cabezas de atención (default: 4)                        ║
║    --dropout F       Dropout (default: 0.1)                                  ║
║    --label-smooth F  Label smoothing (default: 0.1)                          ║
║    --warmup N        Pasos de warmup del scheduler (default: 4000)           ║
║    --eval-every N    Evaluar cada N batches (default: 100)                   ║
║    --resume          Retomar desde último checkpoint                         ║
║                                                                              ║
║  OPCIONES PRINCIPALES (generate):                                            ║
║    --length N        Eventos a generar (default: 1024)                       ║
║    --prime FILE      MIDI de arranque (prime)                                ║
║    --prime-events N  Eventos del prime a usar (default: 100)                 ║
║    --temperature F   Temperatura de muestreo (default: 1.0)                  ║
║    --threshold-len N Ventana deslizante en generación (default: 512)         ║
║    --seed N          Semilla aleatoria (default: 42)                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import math
import time
import random
import pickle
import argparse
import datetime
import json

import numpy as np

# ── Dependencia: midi-neural-processor ───────────────────────────────────────
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from midi_processor.processor import encode_midi, decode_midi
    _MIDI_PROCESSOR_OK = True
except ImportError:
    _MIDI_PROCESSOR_OK = False

# ── PyTorch ───────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torch.distributions as dist_lib
    _TORCH_OK = True
except ImportError:
    # Stub mínimo para que las definiciones de clase no exploten en import-time.
    # Los comandos que necesitan torch llaman a _require_torch() antes de usarlo.
    class _TorchStub:
        Module = object
        def __getattr__(self, name):
            raise ImportError("PyTorch no disponible. Instala con: pip install torch")
    nn = _TorchStub()
    _TORCH_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

EVENT_DIM   = 388
PAD_TOKEN   = EVENT_DIM          # 388
SOS_TOKEN   = EVENT_DIM + 1      # 389
EOS_TOKEN   = EVENT_DIM + 2      # 390
VOCAB_SIZE  = EVENT_DIM + 3      # 391

CHECKPOINT_FNAME = 'checkpoint.pt'
CONFIG_FNAME     = 'config.json'


# ══════════════════════════════════════════════════════════════════════════════
#  CAPAS — atención relativa y bloques Transformer
# ══════════════════════════════════════════════════════════════════════════════

def _sinusoid(max_seq: int, dim: int) -> np.ndarray:
    """Embeddings posicionales sinusoidales, shape (1, max_seq, dim)."""
    return np.array([[
        [
            math.sin(
                pos * math.exp(-math.log(10000) * i / dim) *
                math.exp(math.log(10000) / dim * (i % 2)) +
                0.5 * math.pi * (i % 2)
            )
            for i in range(dim)
        ]
        for pos in range(max_seq)
    ]])


class DynamicPositionEmbedding(nn.Module):
    """Embedding posicional sinusoidal aplicado dinámicamente."""

    def __init__(self, dim: int, max_seq: int = 2048):
        super().__init__()
        pe = torch.from_numpy(_sinusoid(max_seq, dim)).float()
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class RelativeGlobalAttention(nn.Module):
    """
    Atención multi-cabeza con embeddings relativos de posición.
    Implementa el truco de skewing del paper Music Transformer (Huang 2019)
    para reducir la memoria de O(N²D) a O(ND).

    Corrección respecto al original: E tiene requires_grad=True para que
    los embeddings relativos se aprendan durante el entrenamiento.
    """

    def __init__(self, heads: int = 4, dim: int = 256, max_seq: int = 2048):
        super().__init__()
        self.h   = heads
        self.d   = dim
        self.dh  = dim // heads
        self.max_seq = max_seq

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

        # E: embeddings relativos — requires_grad=True (corregido del original)
        self.E = nn.Parameter(torch.randn(max_seq, self.dh))

    def forward(self, inputs, mask=None):
        q, k, v = inputs

        def project(x, W):
            x = W(x)
            x = x.view(x.size(0), x.size(1), self.h, self.dh)
            return x.permute(0, 2, 1, 3)   # (B, h, L, dh)

        q = project(q, self.Wq)
        k = project(k, self.Wk)
        v = project(v, self.Wv)

        len_q = q.size(2)
        len_k = k.size(2)

        # Srel: término de atención relativa via skewing
        E = self.E[max(0, self.max_seq - len_q):, :]     # (len_q, dh)
        QE  = torch.einsum('bhld,md->bhlm', q, E)        # (B, h, len_q, len_q)
        QE  = self._qe_masking(QE)
        Srel = self._skewing(QE, len_k)

        # Atención estándar + Srel
        logits = torch.matmul(q, k.permute(0, 1, 3, 2)) + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits = logits + (mask.to(torch.int64) * -1e9).to(logits.dtype)

        w = F.softmax(logits, dim=-1)
        out = torch.matmul(w, v)                          # (B, h, L, dh)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(out.size(0), -1, self.d)
        return self.fc(out), w

    @staticmethod
    def _qe_masking(qe: torch.Tensor) -> torch.Tensor:
        """Máscara causal para QE."""
        L = qe.size(-1)
        S = qe.size(-2)
        lengths = torch.arange(L - 1, L - S - 1, -1, device=qe.device)
        idx = torch.arange(L, device=qe.device).unsqueeze(0)
        mask = idx >= lengths.unsqueeze(1)    # (S, L)
        return qe * mask.to(qe.dtype)

    @staticmethod
    def _skewing(tensor: torch.Tensor, len_k: int) -> torch.Tensor:
        """Operación de skewing: convierte QE en Srel."""
        padded   = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
        reshaped = padded.view(padded.size(0), padded.size(1),
                               padded.size(-1), padded.size(-2))
        Srel = reshaped[:, :, 1:, :]
        len_q = Srel.size(2)
        if len_k > len_q:
            Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, len_k - len_q])
        elif len_k < len_q:
            Srel = Srel[:, :, :, :len_k]
        return Srel


class TransformerLayer(nn.Module):
    """
    Capa Transformer con atención relativa.
    Pre-LN: LayerNorm antes de la atención para mayor estabilidad.
    FFN: dim → dim//2 → dim (como en el original).
    """

    def __init__(self, dim: int, heads: int, dropout: float, max_seq: int):
        super().__init__()
        self.rga = RelativeGlobalAttention(heads=heads, dim=dim, max_seq=max_seq)

        self.ffn_pre = nn.Linear(dim, dim // 2)
        self.ffn_suf = nn.Linear(dim // 2, dim)

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None):
        attn_out, w = self.rga([x, x, x], mask)
        x = self.norm1(x + self.drop1(attn_out))

        ffn = F.relu(self.ffn_pre(x))
        ffn = self.drop2(self.ffn_suf(ffn))
        x = self.norm2(x + ffn)
        return x, w


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class MusicTransformer(nn.Module):
    """
    Decoder-only Transformer con atención relativa para generación de MIDI.

    Durante el entrenamiento: forward(x) → logits (B, L, V).
    Durante la inferencia:    generate(prime, length) → lista de eventos.
    """

    def __init__(self, dim: int = 256, vocab_size: int = VOCAB_SIZE,
                 num_layers: int = 6, max_seq: int = 2048,
                 heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dim       = dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq   = max_seq

        self.embedding  = nn.Embedding(vocab_size, dim)
        self.pos_enc    = DynamicPositionEmbedding(dim, max_seq=max_seq)
        self.dropout    = nn.Dropout(dropout)
        self.layers     = nn.ModuleList([
            TransformerLayer(dim, heads, dropout, max_seq)
            for _ in range(num_layers)
        ])
        self.fc_out     = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor):
        """
        x: (B, L) enteros — devuelve logits (B, L, V) y pesos de atención.
        """
        mask = _look_ahead_mask(x.size(1), x.device)

        h = self.embedding(x.long()) * math.sqrt(self.dim)
        h = self.pos_enc(h)
        h = self.dropout(h)

        weights = []
        for layer in self.layers:
            h, w = layer(h, mask)
            weights.append(w)

        logits = self.fc_out(h)
        return logits, weights

    @torch.no_grad()
    def generate(self, prime: torch.Tensor, length: int = 1024,
                 temperature: float = 1.0,
                 threshold_len: int = 512) -> list:
        """
        Generación autoregresiva con ventana deslizante.

        prime:         (1, P) tensor de eventos de arranque
        length:        número de eventos nuevos a generar
        temperature:   >1 más aleatorio, <1 más determinista
        threshold_len: tamaño máximo de ventana de contexto
        """
        self.eval()
        decode_buf = prime.clone()
        result     = prime.clone()

        for _ in range(length):
            # Ventana deslizante si el buffer supera el umbral
            ctx = decode_buf[:, -threshold_len:] if decode_buf.size(1) > threshold_len else decode_buf

            # Corrección del original: se pasa la máscara correctamente
            mask = _look_ahead_mask(ctx.size(1), ctx.device)
            h = self.embedding(ctx.long()) * math.sqrt(self.dim)
            h = self.pos_enc(h)
            for layer in self.layers:
                h, _ = layer(h, mask)
            logits = self.fc_out(h)         # (1, L, V)

            # Muestreo del último token con temperatura
            last_logits = logits[:, -1, :] / max(temperature, 1e-8)
            probs  = F.softmax(last_logits, dim=-1)
            next_t = dist_lib.Categorical(probs=probs).sample().unsqueeze(0)  # (1,1)

            decode_buf = torch.cat([decode_buf, next_t], dim=1)
            result     = torch.cat([result,     next_t], dim=1)

        return result[0].cpu().tolist()

    def config_dict(self) -> dict:
        return {
            'dim':        self.dim,
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'max_seq':    self.max_seq,
            'heads':      len(self.layers[0].rga.Wq.weight) if self.layers else 4,
        }

    @classmethod
    def from_config(cls, cfg: dict, dropout: float = 0.0) -> 'MusicTransformer':
        return cls(
            dim=cfg['dim'],
            vocab_size=cfg.get('vocab_size', VOCAB_SIZE),
            num_layers=cfg['num_layers'],
            max_seq=cfg['max_seq'],
            heads=cfg.get('heads', cfg['dim'] // 64),
            dropout=dropout,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE MASCARADO
# ══════════════════════════════════════════════════════════════════════════════

def _look_ahead_mask(size: int, device) -> torch.Tensor:
    """Máscara causal triangular superior (True = enmascarado)."""
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, L, L)


def _sequence_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) < lengths.unsqueeze(1)


# ══════════════════════════════════════════════════════════════════════════════
#  PÉRDIDA Y SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy con label smoothing (Szegedy et al., 2016).
    Ignora tokens PAD en el cálculo de la pérdida.
    """

    def __init__(self, smoothing: float = 0.1, vocab_size: int = VOCAB_SIZE,
                 ignore_index: int = PAD_TOKEN):
        super().__init__()
        assert 0.0 <= smoothing <= 1.0
        self.smoothing    = smoothing
        self.vocab_size   = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, L, V)
        target: (B, L)
        """
        B, L, V = logits.shape
        logits  = logits.view(B * L, V)
        target  = target.view(B * L)

        mask = (target != self.ignore_index)
        n    = mask.sum().clamp(min=1)

        q      = F.one_hot(target.long().clamp(0, V - 1), V).float()
        q_soft = (1 - self.smoothing) * q + self.smoothing / V
        q_soft = q_soft * mask.unsqueeze(1).float()

        log_p  = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        loss   = -(q_soft * log_p).sum() / n
        return loss


class NoamScheduler:
    """
    Scheduler de tasa de aprendizaje Noam (Attention is All You Need).
    Sube linealmente durante warmup_steps, luego decae como step^(-0.5).
    """

    def __init__(self, optimizer, dim: int, warmup_steps: int = 4000):
        self.opt           = optimizer
        self.dim           = dim
        self.warmup_steps  = warmup_steps
        self._step         = 0

    def step(self):
        self._step += 1
        lr = self._rate()
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        self.opt.step()

    def _rate(self, step=None) -> float:
        s = step or self._step
        return self.dim ** (-0.5) * min(s ** (-0.5), s * self.warmup_steps ** (-1.5))

    @property
    def lr(self) -> float:
        return self._rate()


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

class MidiDataset:
    """
    Carga pickles de eventos MIDI y sirve batches slide_seq2seq:
        x = seq[:-1],  y = seq[1:]
    El split train/eval/test se hace tras barajar los ficheros.
    """

    def __init__(self, data_dir: str, seed: int = 42):
        files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith('.pickle')
        ])
        if not files:
            raise FileNotFoundError(f"No se encontraron ficheros .pickle en {data_dir}")

        rng = random.Random(seed)
        rng.shuffle(files)

        n = len(files)
        self.splits = {
            'train': files[:int(n * 0.8)],
            'eval':  files[int(n * 0.8):int(n * 0.9)],
            'test':  files[int(n * 0.9):],
        }

    def __repr__(self):
        return (f"<MidiDataset train={len(self.splits['train'])} "
                f"eval={len(self.splits['eval'])} "
                f"test={len(self.splits['test'])} ficheros>")

    def batch(self, batch_size: int, length: int, split: str = 'train') -> np.ndarray:
        pool = self.splits[split]
        if not pool:
            raise ValueError(f"Split '{split}' vacío.")
        chosen = random.sample(pool, k=min(batch_size, len(pool)))
        seqs   = [self._load_seq(f, length + 1) for f in chosen]
        # Descarta los que resultaron demasiado cortos
        seqs = [s for s in seqs if s is not None]
        if not seqs:
            raise IndexError("Secuencias demasiado cortas para el max_seq solicitado.")
        while len(seqs) < batch_size:
            seqs.append(random.choice(seqs))
        return np.array(seqs[:batch_size])

    def slide_batch(self, batch_size: int, length: int,
                    split: str = 'train') -> tuple:
        """Devuelve (x, y) con x=seq[:-1], y=seq[1:]."""
        data = self.batch(batch_size, length, split)
        return data[:, :-1], data[:, 1:]

    def _load_seq(self, path: str, max_len: int):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if len(data) < max_len:
            return None
        start = random.randrange(0, len(data) - max_len + 1)
        return np.array(data[start:start + max_len], dtype=np.int32)

    def n_batches(self, batch_size: int, split: str = 'train') -> int:
        return max(1, len(self.splits[split]) // batch_size)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(model_dir: str, model: 'MusicTransformer',
                     scheduler: NoamScheduler, epoch: int,
                     best_loss: float, cfg: dict):
    os.makedirs(model_dir, exist_ok=True)
    torch.save({
        'epoch':      epoch,
        'best_loss':  best_loss,
        'step':       scheduler._step,
        'model':      model.state_dict(),
        'optimizer':  scheduler.opt.state_dict(),
    }, os.path.join(model_dir, CHECKPOINT_FNAME))
    with open(os.path.join(model_dir, CONFIG_FNAME), 'w') as f:
        json.dump(cfg, f, indent=2)


def _load_checkpoint(model_dir: str, model: 'MusicTransformer',
                     scheduler: NoamScheduler = None) -> dict:
    ckpt_path = os.path.join(model_dir, CHECKPOINT_FNAME)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No se encontró checkpoint en {model_dir}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if scheduler is not None:
        scheduler.opt.load_state_dict(ckpt['optimizer'])
        scheduler._step = ckpt.get('step', 0)
    return ckpt


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, CONFIG_FNAME)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No se encontró {CONFIG_FNAME} en {model_dir}")
    with open(cfg_path) as f:
        return json.load(f)


def _get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: preprocess
# ══════════════════════════════════════════════════════════════════════════════

def cmd_preprocess(args):
    _require_midi_processor()

    midi_exts = ('.mid', '.midi')
    midi_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(args.input)
        for f in files if f.lower().endswith(midi_exts)
    ]
    if not midi_files:
        print(f"ERROR: no se encontraron ficheros MIDI en {args.input}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    ok = errors = 0

    for path in midi_files:
        name = os.path.splitext(os.path.basename(path))[0]
        out  = os.path.join(args.output, name + '.pickle')
        try:
            data = encode_midi(path)
            with open(out, 'wb') as f:
                pickle.dump(data, f)
            ok += 1
            if args.verbose:
                print(f"  ✓  {path}  →  {len(data)} eventos")
        except KeyboardInterrupt:
            print("\nInterrumpido.")
            break
        except Exception as e:
            errors += 1
            print(f"  ✗  {path}: {e}")

    print(f"\nPreprocess completado: {ok} OK, {errors} errores → {args.output}/")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    _require_torch()
    device = _get_device()
    print(f"Dispositivo: {device}")

    # Dataset
    dataset = MidiDataset(args.data, seed=args.seed)
    print(dataset)

    # Config del modelo
    cfg = {
        'dim':        args.dim,
        'vocab_size': VOCAB_SIZE,
        'num_layers': args.layers,
        'max_seq':    args.max_seq,
        'heads':      args.heads,
    }

    # Modelo
    model = MusicTransformer(
        dim=args.dim, vocab_size=VOCAB_SIZE, num_layers=args.layers,
        max_seq=args.max_seq, heads=args.heads, dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, dim=args.dim, warmup_steps=args.warmup)
    criterion = SmoothCrossEntropyLoss(
        smoothing=args.label_smooth, vocab_size=VOCAB_SIZE, ignore_index=PAD_TOKEN
    )

    start_epoch = 0
    best_loss   = float('inf')

    # Retomar entrenamiento
    if args.resume:
        try:
            ckpt = _load_checkpoint(args.model_dir, model, scheduler)
            start_epoch = ckpt['epoch'] + 1
            best_loss   = ckpt.get('best_loss', best_loss)
            print(f"Retomando desde época {start_epoch} (step={scheduler._step})")
        except FileNotFoundError as e:
            print(f"AVISO: {e}. Iniciando desde cero.")

    os.makedirs(args.model_dir, exist_ok=True)
    n_batches = dataset.n_batches(args.batch_size)
    print(f"\nModelo: {sum(p.numel() for p in model.parameters()):,} parámetros")
    print(f"Épocas: {args.epochs}  |  Batches/época: {n_batches}  |  max_seq: {args.max_seq}\n")

    global_step = scheduler._step

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = epoch_acc = 0.0
        t0 = time.time()

        for b in range(n_batches):
            try:
                bx, by = dataset.slide_batch(args.batch_size, args.max_seq)
            except (IndexError, ValueError):
                continue

            bx = torch.from_numpy(bx).long().to(device)
            by = torch.from_numpy(by).long().to(device)

            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scheduler.step()

            # Accuracy
            with torch.no_grad():
                pred = logits.argmax(-1)
                mask = (by != PAD_TOKEN)
                acc  = (pred[mask] == by[mask]).float().mean().item() if mask.any() else 0.0

            epoch_loss += loss.item()
            epoch_acc  += acc
            global_step += 1

            # Evaluación periódica
            if (b + 1) % args.eval_every == 0:
                eval_loss, eval_acc = _evaluate(model, dataset, criterion, device,
                                                args.batch_size, args.max_seq)
                print(f"  [eval] loss={eval_loss:.4f}  acc={eval_acc:.4f}")
                model.train()

                if eval_loss < best_loss:
                    best_loss = eval_loss
                    _save_checkpoint(args.model_dir, model, scheduler,
                                     epoch, best_loss, cfg)

        elapsed   = time.time() - t0
        avg_loss  = epoch_loss / max(n_batches, 1)
        avg_acc   = epoch_acc  / max(n_batches, 1)
        print(f"Época {epoch+1:4d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  acc={avg_acc:.4f}  "
              f"lr={scheduler.lr:.2e}  {elapsed:.1f}s")

    # Checkpoint final
    _save_checkpoint(args.model_dir, model, scheduler, args.epochs - 1, best_loss, cfg)
    print(f"\nEntrenamiento completado. Modelo guardado en {args.model_dir}/")


def _evaluate(model, dataset, criterion, device, batch_size, max_seq,
              n_batches: int = 4) -> tuple:
    model.eval()
    total_loss = total_acc = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            try:
                bx, by = dataset.slide_batch(batch_size, max_seq, split='eval')
            except (IndexError, ValueError):
                continue
            bx = torch.from_numpy(bx).long().to(device)
            by = torch.from_numpy(by).long().to(device)

            logits, _ = model(bx)
            loss = criterion(logits, by)
            pred = logits.argmax(-1)
            mask = (by != PAD_TOKEN)
            acc  = (pred[mask] == by[mask]).float().mean().item() if mask.any() else 0.0
            total_loss += loss.item()
            total_acc  += acc

    return total_loss / n_batches, total_acc / n_batches


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: generate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate(args):
    _require_torch()
    _require_midi_processor()

    device = _get_device()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Cargar config y modelo
    cfg   = _load_config(args.model_dir)
    model = MusicTransformer.from_config(cfg, dropout=0.0).to(device)
    _load_checkpoint(args.model_dir, model)
    model.eval()
    print(f"Modelo cargado desde {args.model_dir}  "
          f"({sum(p.numel() for p in model.parameters()):,} parámetros)")

    # Prime: eventos de arranque
    if args.prime:
        print(f"Codificando prime: {args.prime}")
        prime_events = encode_midi(args.prime)[:args.prime_events]
        prime_tensor = torch.tensor([prime_events], dtype=torch.long, device=device)
        print(f"Prime: {len(prime_events)} eventos")
    else:
        prime_tensor = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
        print("Sin prime, generando desde SOS.")

    # Generar
    print(f"Generando {args.length} eventos (temperatura={args.temperature})…")
    t0     = time.time()
    result = model.generate(
        prime=prime_tensor,
        length=args.length,
        temperature=args.temperature,
        threshold_len=args.threshold_len,
    )
    elapsed = time.time() - t0
    print(f"Generados {len(result)} eventos en {elapsed:.1f}s")

    # Descartar tokens especiales al final
    result = [e for e in result if e < EVENT_DIM]

    # Guardar MIDI
    decode_midi(result, file_path=args.output)
    print(f"MIDI guardado en {args.output}")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    _require_torch()

    # Inspeccionar dataset
    if args.data:
        print(f"\n── Dataset: {args.data} ──")
        try:
            ds = MidiDataset(args.data)
            print(ds)
            sizes = []
            for split, files in ds.splits.items():
                total_events = 0
                for f in files:
                    with open(f, 'rb') as fh:
                        seq = pickle.load(fh)
                    total_events += len(seq)
                avg = total_events // max(len(files), 1)
                print(f"  {split:5s}: {len(files):4d} ficheros, "
                      f"~{avg} eventos/fichero, total {total_events:,}")
                sizes.extend([total_events])
        except Exception as e:
            print(f"  ERROR: {e}")

    # Inspeccionar checkpoint
    if args.model_dir:
        print(f"\n── Modelo: {args.model_dir} ──")
        try:
            cfg = _load_config(args.model_dir)
            print("  Config:")
            for k, v in cfg.items():
                print(f"    {k}: {v}")

            ckpt_path = os.path.join(args.model_dir, CHECKPOINT_FNAME)
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location='cpu')
                print(f"  Checkpoint:")
                print(f"    época:      {ckpt.get('epoch', '?')}")
                print(f"    step:       {ckpt.get('step', '?')}")
                print(f"    best_loss:  {ckpt.get('best_loss', '?'):.4f}"
                      if isinstance(ckpt.get('best_loss'), float) else
                      f"    best_loss:  {ckpt.get('best_loss', '?')}")
                model = MusicTransformer.from_config(cfg)
                model.load_state_dict(ckpt['model'])
                n_params = sum(p.numel() for p in model.parameters())
                n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"    parámetros: {n_params:,} totales, {n_train:,} entrenables")
            else:
                print("  (sin checkpoint guardado aún)")
        except FileNotFoundError as e:
            print(f"  {e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not args.data and not args.model_dir:
        print("Indica --data y/o --model-dir para inspeccionar.")


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDS DE DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def _require_midi_processor():
    if not _MIDI_PROCESSOR_OK:
        print("ERROR: midi_processor no disponible.")
        print("  git clone https://github.com/jason9693/midi-neural-processor.git")
        print("  mv midi-neural-processor midi_processor")
        sys.exit(1)


def _require_torch():
    if not _TORCH_OK:
        print("ERROR: PyTorch no disponible. Instala con:")
        print("  pip install torch")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='music_transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Music Transformer v1.0\n"
            "Generación de música con estructura a largo plazo.\n\n"
            "Flujo típico:\n"
            "  1. preprocess  — MIDI corpus → pickles de eventos\n"
            "  2. train       — entrenar el modelo\n"
            "  3. generate    — generar MIDI desde el checkpoint\n"
        ),
    )
    sub = parser.add_subparsers(dest='command', metavar='COMANDO')
    sub.required = True

    # ── preprocess ────────────────────────────────────────────────────────────
    p = sub.add_parser('preprocess', help='MIDI corpus → pickles de eventos')
    p.add_argument('--input',   required=True, metavar='DIR',
                   help='Directorio raíz con ficheros .mid/.midi')
    p.add_argument('--output',  required=True, metavar='DIR',
                   help='Directorio de salida para los .pickle')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(func=cmd_preprocess)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Entrenar el modelo')
    p.add_argument('--data',        required=True, metavar='DIR',
                   help='Directorio con los .pickle de preprocess')
    p.add_argument('--model-dir',   required=True, metavar='DIR',
                   help='Directorio para guardar checkpoints y config')
    p.add_argument('--epochs',      type=int,   default=300)
    p.add_argument('--batch-size',  type=int,   default=4)
    p.add_argument('--max-seq',     type=int,   default=2048,
                   help='Longitud máxima de secuencia (default: 2048)')
    p.add_argument('--layers',      type=int,   default=6,
                   help='Capas Transformer (default: 6)')
    p.add_argument('--dim',         type=int,   default=256,
                   help='Dimensión de embeddings (default: 256)')
    p.add_argument('--heads',       type=int,   default=4,
                   help='Cabezas de atención (default: 4)')
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--label-smooth', type=float, default=0.1, metavar='F')
    p.add_argument('--warmup',      type=int,   default=4000,
                   help='Pasos de warmup del scheduler Noam (default: 4000)')
    p.add_argument('--eval-every',  type=int,   default=100, metavar='N',
                   help='Evaluar cada N batches (default: 100)')
    p.add_argument('--resume',      action='store_true',
                   help='Retomar desde el último checkpoint')
    p.add_argument('--seed',        type=int,   default=42)
    p.set_defaults(func=cmd_train)

    # ── generate ──────────────────────────────────────────────────────────────
    p = sub.add_parser('generate', help='Generar MIDI desde un checkpoint')
    p.add_argument('--model-dir',     required=True, metavar='DIR')
    p.add_argument('--output',        default='generated.mid', metavar='FILE')
    p.add_argument('--length',        type=int,   default=1024,
                   help='Eventos a generar (default: 1024)')
    p.add_argument('--prime',         default=None, metavar='FILE',
                   help='MIDI de arranque (opcional)')
    p.add_argument('--prime-events',  type=int,   default=100, metavar='N',
                   help='Eventos del prime a usar (default: 100)')
    p.add_argument('--temperature',   type=float, default=1.0, metavar='F',
                   help='Temperatura de muestreo (default: 1.0)')
    p.add_argument('--threshold-len', type=int,   default=512, metavar='N',
                   help='Ventana deslizante en generación (default: 512)')
    p.add_argument('--seed',          type=int,   default=42)
    p.set_defaults(func=cmd_generate)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect', help='Diagnóstico de dataset y/o checkpoint')
    p.add_argument('--model-dir', default=None, metavar='DIR')
    p.add_argument('--data',      default=None, metavar='DIR',
                   help='Directorio de pickles a analizar')
    p.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
