"""
Entrenamiento autoresearch sobre corpus ABC — versión CPU (sin CUDA).

Dos fases controladas por la variable PHASE:

  PHASE = 1  — solo val_bpb. Sin evaluación de preferencias.
               Usar hasta que validity_ratio >= 0.5 en dos experimentos consecutivos.

  PHASE = 2  — val_bpb + reward_score combinados (evaluador seleccionable:
               preference_trainer.py o quality_scorer.py, ver abc_eval.py).
               Activar cuando se cumpla la condición de avance de fase.

Uso: uv run abc_train.py
"""

import gc
import math
import os
import time
from dataclasses import asdict

import torch

import abc_eval
from abc_model import GPT, GPTConfig, generate_abc_sample, is_valid_abc
from abc_prepare import (
    EVAL_TOKENS,
    MAX_SEQ_LEN,
    TIME_BUDGET as DEFAULT_TIME_BUDGET,
    Tokenizer,
    get_token_bytes,
)

# Permite acortar el presupuesto de entrenamiento (smoke tests) sin tocar
# abc_prepare.py, igual que abc_train_diffusion.py.
TIME_BUDGET = int(os.environ.get("ABC_TIME_BUDGET", DEFAULT_TIME_BUDGET))

# ---------------------------------------------------------------------------
# Fase de entrenamiento  ← EL AGENTE MODIFICA ESTO
# ---------------------------------------------------------------------------

# PHASE = 1 : solo val_bpb. Sin evaluación de preferencias.
# PHASE = 2 : val_bpb + reward_score. Pipeline completo de evaluación musical.
PHASE = 1

# ---------------------------------------------------------------------------
# Evaluación musical (solo relevante en PHASE = 2) — la lógica vive en abc_eval.py
# ---------------------------------------------------------------------------

# Evaluador. También se puede fijar con la variable de entorno ABC_EVAL_BACKEND
# (tiene prioridad), p. ej.:  ABC_EVAL_BACKEND=quality uv run abc_train.py
#   "preference" → preference_trainer.py   escala 0-1  (modelo de preferencias personal)
#   "quality"    → quality_scorer.py       escala 0-1  (scorer automático)
# Elegir UNO por sesión y NO cambiarlo entre experimentos (escalas no comparables).
EVAL_BACKEND        = "preference"

PREF_TRAINER_PATH   = None     # None = autodetectar (cwd, carpeta del proyecto, carpeta padre)
PREF_MODEL          = "prefs"  # solo backend "preference"
PREF_CORPUS         = "./corpus.npz"
QUALITY_SCORER_PATH = None     # None = autodetectar
QUALITY_EXTRA_ARGS  = []       # solo backend "quality"; p. ej. ["--fast"]
PREF_EVAL_SAMPLES   = 20
PREF_SAMPLE_LEN     = 256
PREF_TEMPERATURE    = 0.85
PREF_WEIGHT         = 0.3

# ---------------------------------------------------------------------------
# Hiperparámetros  ← EL AGENTE MODIFICA ESTA SECCIÓN
# ---------------------------------------------------------------------------

# Arquitectura
DEPTH        = 4    # capas transformer
HEAD_DIM     = 32   # dimensión por cabeza de atención
ASPECT_RATIO = 24   # model_dim = max(DEPTH * ASPECT_RATIO, HEAD_DIM) → 96

# Optimizador
LR             = 1e-3
WEIGHT_DECAY   = 0.1
ADAM_BETAS     = (0.9, 0.95)
WARMUP_RATIO   = 0.05
WARMDOWN_RATIO = 0.4
FINAL_LR_FRAC  = 0.0

# Batch
# TOTAL_BATCH_SIZE debe ser múltiplo de DEVICE_BATCH_SIZE * MAX_SEQ_LEN
TOTAL_BATCH_SIZE  = 2**10  # 1024 tokens/paso
DEVICE_BATCH_SIZE = 2      # tokens_per_step = 2 * 512 = 1024

# ---------------------------------------------------------------------------
# Dataloader CPU
# ---------------------------------------------------------------------------

def make_dataloader_cpu(tokenizer, B, T, split, buffer_size=500):
    from abc_prepare import SEP_TOKEN, _text_chunks
    assert split in ["train", "val"]
    row_capacity = T + 1
    bos_token    = tokenizer.get_bos_token_id()
    doc_buffer   = []
    chunk_iter   = _text_chunks(split, chunk_size=T * 3)
    epoch        = 1

    def refill():
        nonlocal epoch
        chunk = next(chunk_iter)
        docs  = [d.strip() for d in chunk.split(SEP_TOKEN) if d.strip()] or [chunk]
        doc_buffer.extend(tokenizer.encode(docs, prepend=bos_token))

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx, best_len = i, len(doc)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        yield row_buffer[:, :-1].clone(), row_buffer[:, 1:].clone(), epoch


# ---------------------------------------------------------------------------
# Evaluación BPB — CPU
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb_cpu(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cpu")
    val_loader  = make_dataloader_cpu(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps       = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats  = 0.0
    total_bytes = 0
    model.eval()
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        nbytes    = token_bytes[y.view(-1)]
        mask      = nbytes > 0
        total_nats  += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Evaluación musical (PHASE = 2) — la lógica vive en abc_eval.py
# ---------------------------------------------------------------------------

def run_music_evaluation(model, tokenizer, backend):
    """Genera muestras ABC, las pasa a MIDI y las puntúa con `backend`."""
    return abc_eval.evaluate_samples(
        lambda: generate_abc_sample(model, tokenizer,
                                    max_new_tokens=PREF_SAMPLE_LEN,
                                    temperature=PREF_TEMPERATURE),
        is_valid_abc, PREF_EVAL_SAMPLES, backend,
        pref_trainer_path=PREF_TRAINER_PATH, pref_model=PREF_MODEL,
        pref_corpus=PREF_CORPUS,
        quality_scorer_path=QUALITY_SCORER_PATH,
        quality_extra_args=QUALITY_EXTRA_ARGS,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_start = time.time()
    backend = abc_eval.resolve_backend(EVAL_BACKEND)   # falla pronto si no es válido
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    tokenizer  = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")

    base_dim  = DEPTH * ASPECT_RATIO
    model_dim = max(base_dim, HEAD_DIM)
    model_dim = ((model_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM

    config = GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=DEPTH, n_head=num_heads, n_embd=model_dim,
    )
    print(f"Model config: {asdict(config)}")

    model = GPT(config)
    model.init_weights()

    param_counts = model.num_scaling_params()
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    num_params = param_counts["total"]

    tokens_per_step = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
    assert TOTAL_BATCH_SIZE % tokens_per_step == 0, \
        f"TOTAL_BATCH_SIZE={TOTAL_BATCH_SIZE} no divisible por {tokens_per_step}"
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_step

    optimizer    = model.setup_optimizer(LR, WEIGHT_DECAY, ADAM_BETAS)
    train_loader = make_dataloader_cpu(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
    x, y, epoch  = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s | grad_accum_steps: {grad_accum_steps}")

    def get_lr(progress):
        if progress < WARMUP_RATIO:
            return LR * (progress / WARMUP_RATIO) if WARMUP_RATIO > 0 else LR
        elif progress < 1.0 - WARMDOWN_RATIO:
            return LR
        else:
            cooldown = (1.0 - progress) / WARMDOWN_RATIO
            return LR * (cooldown + (1 - cooldown) * FINAL_LR_FRAC)

    # ── Bucle de entrenamiento ────────────────────────────────────────────────

    t_start_training    = time.time()
    smooth_train_loss   = 0.0
    total_training_time = 0.0
    step                = 0

    while True:
        t0 = time.time()

        model.train()
        for _ in range(grad_accum_steps):
            loss = model(x, y) / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        train_loss_f = loss.item() * grad_accum_steps
        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            raise SystemExit(1)

        progress   = min(total_training_time / TIME_BUDGET, 1.0)
        current_lr = get_lr(progress)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        t1 = time.time()
        dt = t1 - t0
        if step > 5:
            total_training_time += dt

        ema_beta          = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased          = smooth_train_loss / (1 - ema_beta**(step + 1))
        remaining         = max(0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:04d} ({100*progress:.1f}%) | loss: {debiased:.4f} | "
            f"lr: {current_lr:.2e} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="", flush=True,
        )

        if step == 0:
            gc.collect()

        step += 1
        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    print()
    total_tokens = step * TOTAL_BATCH_SIZE

    # ── Evaluación final ──────────────────────────────────────────────────────

    model.eval()
    val_bpb = evaluate_bpb_cpu(model, tokenizer, DEVICE_BATCH_SIZE)

    eval_result = None
    if PHASE == 1:
        validity_ratio = combined_score = None
    else:
        print(f"\nEvaluando calidad musical (evaluador: {backend})...")
        eval_result    = run_music_evaluation(model, tokenizer, backend)
        validity_ratio = eval_result.get("validity_ratio", 0.0)
        combined_score = abc_eval.combined_score(
            val_bpb, eval_result.get("reward_score"), PREF_WEIGHT)

    # ── Guardar checkpoint ────────────────────────────────────────────────────

    torch.save({
        "model_state": model.state_dict(),
        "config":      asdict(config),
        "val_bpb":     val_bpb,
        "step":        step,
        "depth":       DEPTH,
        "model_dim":   model_dim,
        "head_dim":    HEAD_DIM,
        "vocab_size":  vocab_size,
    }, "model_checkpoint.pt")
    print("Checkpoint guardado: model_checkpoint.pt")

    # ── Resumen ───────────────────────────────────────────────────────────────

    def fmt(val, d=6):
        return f"{val:.{d}f}" if val is not None else "N/A"

    t_end = time.time()
    print("---")
    print(f"phase:            {PHASE}")
    print(f"val_bpb:          {val_bpb:.6f}")
    for _line in abc_eval.summary_lines(backend, eval_result):
        print(_line)
    print(f"validity_ratio:   {fmt(validity_ratio, 3)}")
    print(f"combined_score:   {fmt(combined_score, 6)}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.3f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.3f}")
    print(f"depth:            {DEPTH}")
    print(f"model_dim:        {model_dim}")
