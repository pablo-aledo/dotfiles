"""
abc_model_diffusion.py — Transformer bidireccional de difusión discreta
(denoising) sobre tokens ABC.

Extraído de abc_train_diffusion.py: igual que abc_model.py contiene el GPT
autorregresivo, este módulo contiene el modelo de difusión y su muestreador,
sin código de entrenamiento. Así generate.py (y cualquier otro consumidor)
puede reconstruir el modelo desde un checkpoint con arch="diffusion" sin
importar el script de entrenamiento.

Arquitectura inspirada en tiny-diffusion (github.com/nathan-barry/tiny-diffusion)
pero adaptada a tokens BPE (no caracteres) y al empaquetado por documentos ABC
de abc_prepare.py. Atención bidireccional + rotary + QK-norm + MLP ReLU².

Reutiliza de abc_model.py las piezas idénticas (norm, apply_rotary_emb, MLP,
is_valid_abc), que antes estaban copiadas en abc_train_diffusion.py.

Contrato público equivalente al de abc_model.GPT: init_weights(),
num_scaling_params(), setup_optimizer(), forward(idx, targets, reduction) y
config con .sequence_len/.vocab_size/.n_layer/.n_head/.n_embd; y
generate_abc_sample(model, tokenizer, max_new_tokens, temperature) → str.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc_model import MLP, apply_rotary_emb, is_valid_abc, norm

__all__ = ["DiffusionConfig", "BidirectionalSelfAttention", "Block", "DiffusionGPT",
           "generate_abc_sample", "is_valid_abc"]   # is_valid_abc se reexporta


@dataclass
class DiffusionConfig:
    sequence_len:   int = 512
    vocab_size:     int = 512
    n_layer:        int = 4
    n_head:         int = 4
    n_embd:         int = 128
    mask_token_id:  int = 0


class BidirectionalSelfAttention(nn.Module):
    """Idéntica a la atención de abc_model.py salvo is_causal=False: esta es
    LA diferencia arquitectónica que convierte el GPT en modelo de difusión."""

    def __init__(self, config):
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q    = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k    = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v    = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalSelfAttention(config)
        self.mlp  = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class DiffusionGPT(nn.Module):
    """Mismo contrato público que abc_model.GPT: init_weights(),
    num_scaling_params(), setup_optimizer(), forward(idx, targets, reduction),
    y config con .sequence_len/.vocab_size/.n_layer/.n_head/.n_embd."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.wte     = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        head_dim = config.n_embd // config.n_head
        self.rotary_seq_len = config.sequence_len * 2
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq      = 1.0 / (base ** (channel_range / head_dim))
        t     = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx, targets=None, reduction="mean", mask=None):
        B, T = idx.size()
        x = norm(self.wte(idx))
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        for block in self.blocks:
            x = block(x, (cos, sin))

        logits = self.lm_head(norm(x))

        if targets is None:
            return logits

        loss_flat = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none",
        )
        if mask is not None:
            mask_flat = mask.reshape(-1).float()
            if reduction == "mean":
                return (loss_flat * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
            return loss_flat * mask_flat

        return loss_flat.mean() if reduction == "mean" else loss_flat

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        emb   = self.wte.weight.numel()
        return {"total": total, "non_embedding": total - emb}

    def setup_optimizer(self, lr, weight_decay, betas):
        decay_params   = [p for n, p in self.named_parameters() if p.ndim >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.ndim < 2]
        return torch.optim.AdamW(
            [
                {"params": decay_params,   "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas,
        )


# ---------------------------------------------------------------------------
# Generación — decodificación paralela por confianza (un bloque de tamaño
# MAX_SEQ_LEN es suficiente: las piezas ABC son cortas). Mismo contrato de
# entrada/salida que abc_model.generate_abc_sample para que generate.py y
# el bucle de evaluación no necesiten distinguir la arquitectura.
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_abc_sample(model, tokenizer, max_new_tokens=256, temperature=0.85,
                         confidence_threshold=0.9, top_k=4):
    model.eval()
    mask_id = model.config.mask_token_id
    bos_id  = tokenizer.get_bos_token_id()
    T       = min(model.config.sequence_len, max_new_tokens + 1)

    stop_ids = set()
    for special in ["<|reserved_0|>", "<|reserved_1|>"]:
        try:
            stop_ids.add(tokenizer.enc.encode_single_token(special))
        except Exception:
            pass

    x = torch.full((1, T), mask_id, dtype=torch.long)
    x[0, 0] = bos_id
    masked = torch.zeros(1, T, dtype=torch.bool)
    masked[0, 1:] = True

    max_steps = T * 2  # cota de seguridad: nunca debería hacer falta más
    for _ in range(max_steps):
        if not masked.any():
            break
        logits = model(x)
        probs  = F.softmax(logits / temperature, dim=-1)
        top_k_probs, top_k_idx = torch.topk(probs, k=top_k, dim=-1)
        confidences = top_k_probs.sum(dim=-1)

        decode_mask = (confidences >= confidence_threshold) & masked
        if not decode_mask.any():
            masked_conf = torch.where(masked, confidences, torch.tensor(-float("inf")))
            decode_mask.view(-1)[masked_conf.argmax()] = True

        top_k_probs_norm = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        sampled_k = torch.multinomial(top_k_probs_norm.view(-1, top_k), 1).view(1, T)
        sampled   = torch.gather(top_k_idx, -1, sampled_k.unsqueeze(-1)).squeeze(-1)

        x = torch.where(decode_mask, sampled, x)
        masked = masked & ~decode_mask

    ids = x[0, 1:].tolist()  # excluir BOS
    for i, tok in enumerate(ids):
        if tok in stop_ids:
            ids = ids[:i]
            break
    return tokenizer.decode(ids)
