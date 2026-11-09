"""
Definición del modelo GPT y utilidades de generación compartidas entre
abc_train.py y generate.py.

No contiene código de entrenamiento ni de evaluación de preferencias.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from abc_prepare import MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# Arquitectura GPT — versión CPU
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len:   int = 512
    vocab_size:     int = 512
    n_layer:        int = 4
    n_head:         int = 4
    n_embd:         int = 128
    window_pattern: str = "L"


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Devuelve True si la capa debe tener Value Embedding (capas alternas)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q    = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k    = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v    = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
            if has_ve(layer_idx, config.n_layer) else None
        )

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        if ve is not None:
            ve   = ve.view(B, T, self.n_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v    = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp  = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h":   nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head       = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas    = nn.Parameter(torch.zeros(config.n_layer))

        head_dim = config.n_embd // config.n_head
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, config.n_embd)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })

        self.rotary_seq_len = config.sequence_len * 4
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            nn.init.uniform_(block.attn.c_q.weight, -s, s)
            nn.init.uniform_(block.attn.c_k.weight, -s, s)
            nn.init.uniform_(block.attn.c_v.weight, -s, s)
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)
        with torch.no_grad():
            self.resid_lambdas.fill_(1.0)
            self.x0_lambdas.fill_(0.1)
        for ve in self.value_embeds.values():
            nn.init.uniform_(ve.weight, -s, s)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq      = 1.0 / (base ** (channel_range / head_dim))
        t     = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        x  = self.transformer.wte(idx)
        x0 = x
        cos = self.cos[:, :T]
        sin = self.sin[:, :T]

        ve_cache = {
            i: self.value_embeds[str(i)](idx)
            for i in range(self.config.n_layer)
            if has_ve(i, self.config.n_layer)
        }

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, ve_cache.get(i), (cos, sin))

        logits = self.lm_head(norm(x))

        if targets is None:
            return logits

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction=reduction,
        )

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        emb   = self.transformer.wte.weight.numel()
        return {"total": total, "non_embedding": total - emb}

    def estimate_flops(self):
        L, E, T = self.config.n_layer, self.config.n_embd, self.config.sequence_len
        return L * (4 * T * E * E + 8 * E * E)

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
# Generación y validación
# ---------------------------------------------------------------------------

def generate_abc_sample(model, tokenizer, max_new_tokens=256, temperature=0.85):
    """
    Genera una pieza ABC por muestreo autoregresivo.
    Arranca con BOS y para al encontrar un token especial o alcanzar max_new_tokens.
    """
    model.eval()
    bos_id    = tokenizer.get_bos_token_id()
    ids       = [bos_id]
    input_ids = torch.tensor([ids], dtype=torch.long)

    stop_ids = set()
    for special in ["<|reserved_0|>", "<|reserved_1|>"]:
        try:
            stop_ids.add(tokenizer.enc.encode_single_token(special))
        except Exception:
            pass

    with torch.no_grad():
        for _ in range(max_new_tokens):
            ctx     = input_ids[:, -MAX_SEQ_LEN:]
            logits  = model(ctx)
            next_logits = logits[0, -1, :] / temperature
            probs   = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            if next_id in stop_ids:
                break
            ids.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]])], dim=1
            )

    return tokenizer.decode(ids[1:])  # excluir BOS


def is_valid_abc(text: str) -> bool:
    """Comprobación mínima de que el texto es ABC válido."""
    lines      = text.splitlines()
    has_header = any(line.startswith(("X:", "T:", "K:", "M:")) for line in lines)
    has_notes  = any(c in text for c in "ABCDEFGabcdefg")
    has_bar    = "|" in text
    return has_header and has_notes and has_bar
