"""
Preparación de datos para autoresearch-abc.
Lee una carpeta de ficheros .abc y entrena un tokenizador BPE sobre ellos.

Uso:
    python abc_prepare.py --abc-dir ./abc/
    python abc_prepare.py --abc-dir ./abc/ --vocab-size 512 --val-ratio 0.05

Los datos y el tokenizador se guardan en ~/.cache/autoresearch-abc/
"""

import os
import sys
import time
import math
import random
import argparse
import pickle
from pathlib import Path

import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constantes (fijas — no modificar)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN  = 512    # contexto más corto que el original: las piezas ABC son breves
TIME_BUDGET  = 300    # presupuesto de entrenamiento en segundos (5 min)
EVAL_TOKENS  = 200_000  # tokens de validación (corpus ABC es pequeño, 40M sería excesivo)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

CACHE_DIR      = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-abc")
DATA_DIR       = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR  = os.path.join(CACHE_DIR, "tokenizer")
VOCAB_SIZE     = 512   # ABC tiene vocabulario muy reducido; 512 cubre todo con margen

# Caracteres especiales
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN      = "<|reserved_0|>"  # inicio de pieza
SEP_TOKEN      = "<|reserved_1|>"  # separador entre piezas en el stream

# Patrón de split adaptado a ABC: no partir tokens naturales como "A2", "|:", "K:G"
# Preserva: encabezados (X:, T:, K:, M:, L:, Q:, P:), notas con duración, barras
SPLIT_PATTERN = r"""[A-Za-z][0-9]*|[|:]+|[<>^_=]|\d+|[\r\n]+|\s+|[^\s]"""

# ---------------------------------------------------------------------------
# Carga del corpus ABC
# ---------------------------------------------------------------------------

def load_abc_files(abc_dir: str) -> list[str]:
    """Carga todos los ficheros .abc del directorio (recursivo). Devuelve lista de textos."""
    abc_path = Path(abc_dir)
    if not abc_path.exists():
        print(f"ERROR: directorio no encontrado: {abc_dir}", file=sys.stderr)
        sys.exit(1)

    files = list(abc_path.rglob("*.abc")) + list(abc_path.rglob("*.ABC"))
    if not files:
        print(f"ERROR: no se encontraron ficheros .abc en {abc_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"ABC: encontrados {len(files)} ficheros en {abc_dir}")

    pieces = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                pieces.append(text)
        except Exception as e:
            print(f"  Advertencia: no se pudo leer {f}: {e}")

    print(f"ABC: {len(pieces)} piezas cargadas correctamente")
    return pieces


def split_train_val(pieces: list[str], val_ratio: float = 0.05, seed: int = 42):
    """Divide en train/val de forma reproducible."""
    rng = random.Random(seed)
    shuffled = pieces[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def prepare_data(abc_dir: str, val_ratio: float = 0.05):
    """Carga piezas, divide train/val y las guarda como ficheros de texto."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.txt")
    val_path   = os.path.join(DATA_DIR, "val.txt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Datos: ya preparados en {DATA_DIR}")
        return

    pieces = load_abc_files(abc_dir)
    train_pieces, val_pieces = split_train_val(pieces, val_ratio)

    print(f"Datos: {len(train_pieces)} train / {len(val_pieces)} val")

    sep = f"\n{SEP_TOKEN}\n"
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(sep.join(train_pieces))

    with open(val_path, "w", encoding="utf-8") as f:
        f.write(sep.join(val_pieces))

    # Estadísticas básicas
    total_chars = sum(len(p) for p in pieces)
    avg_len = total_chars / len(pieces) if pieces else 0
    print(f"Datos: {total_chars:,} chars totales, media {avg_len:.0f} chars/pieza")
    print(f"Datos: guardados en {DATA_DIR}")


# ---------------------------------------------------------------------------
# Entrenamiento del tokenizador
# ---------------------------------------------------------------------------

def text_iterator_for_tokenizer(max_chars: int = 5_000_000):
    """Itera sobre el corpus de entrenamiento para el tokenizador."""
    train_path = os.path.join(DATA_DIR, "train.txt")
    with open(train_path, encoding="utf-8") as f:
        content = f.read(max_chars)
    # Dividir en chunks de ~1000 chars para que rustbpe los procese bien
    chunk_size = 1000
    for i in range(0, len(content), chunk_size):
        yield content[i:i + chunk_size]


def train_tokenizer():
    """Entrena tokenizador BPE sobre el corpus ABC."""
    tokenizer_pkl    = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizador: ya entrenado en {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.txt")
    if not os.path.exists(train_path):
        print("Tokenizador: no hay datos. Ejecuta prepare_data primero.", file=sys.stderr)
        sys.exit(1)

    print("Tokenizador: entrenando BPE...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    vocab_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(
        text_iterator_for_tokenizer(),
        vocab_no_special,
        pattern=SPLIT_PATTERN
    )

    # Construir encoding tiktoken desde las fusiones aprendidas
    pattern           = tokenizer.get_pattern()
    mergeable_ranks   = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    tokens_offset     = len(mergeable_ranks)
    special_tokens    = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="abc_bpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizador: entrenado en {t1 - t0:.1f}s, vocab_size={enc.n_vocab}, guardado en {tokenizer_pkl}")

    # Tabla token → bytes (para cálculo de BPB)
    print("Tokenizador: construyendo tabla token_bytes...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        token_bytes_list.append(0 if token_str in special_set else len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)
    print(f"Tokenizador: token_bytes guardado en {token_bytes_path}")

    # Sanity check
    test = "X:1\nT:Test\nK:G\nM:4/4\n|:GABG|DEFD:|"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Roundtrip fallido: {test!r} -> {decoded!r}"
    print(f"Tokenizador: roundtrip OK ({len(encoded)} tokens para {len(test)} chars)")


# ---------------------------------------------------------------------------
# Utilidades de runtime (importadas por abc_train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Wrapper mínimo del tokenizador."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Tipo inválido: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _text_chunks(split: str, chunk_size: int = 512):
    """Iterador infinito sobre chunks del fichero train o val."""
    path = os.path.join(DATA_DIR, f"{split}.txt")
    assert os.path.exists(path), f"No existe {path}. Ejecuta abc_prepare.py primero."
    while True:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        # Dividir en trozos solapados de chunk_size
        for i in range(0, max(1, len(content) - chunk_size), chunk_size // 2):
            yield content[i:i + chunk_size]


def make_dataloader(tokenizer, B, T, split, buffer_size=500):
    """
    Dataloader con empaquetado best-fit, idéntico al original.
    Adaptado para leer texto plano en vez de parquet.
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    bos_token    = tokenizer.get_bos_token_id()
    doc_buffer   = []
    chunk_iter   = _text_chunks(split, chunk_size=T * 3)  # chunks generosos
    epoch        = 1

    def refill_buffer():
        nonlocal epoch
        chunk = next(chunk_iter)
        # Dividir en "documentos" por separador de pieza
        docs = [d.strip() for d in chunk.split(SEP_TOKEN) if d.strip()]
        if not docs:
            docs = [chunk]
        token_lists = tokenizer.encode(docs, prepend=bos_token)
        doc_buffer.extend(token_lists)
        if chunk.startswith(BOS_TOKEN):
            epoch += 1

    row_buffer  = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer  = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer  = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs  = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs      = gpu_buffer[:B * T].view(B, T)
    targets     = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos
                best_idx, best_len = -1, 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx, best_len = i, doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluación (NO MODIFICAR — métrica fija)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """Bits por byte. Igual que el original pero con corpus ABC."""
    token_bytes = get_token_bytes(device="cuda")
    val_loader  = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps       = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats  = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat    = y.view(-1)
        nbytes    = token_bytes[y_flat]
        mask      = nbytes > 0
        total_nats  += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara corpus ABC para autoresearch-abc")
    parser.add_argument("--abc-dir",   required=True,  help="Directorio con ficheros .abc")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Fracción de validación (def: 0.05)")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                        help=f"Tamaño del vocabulario BPE (def: {VOCAB_SIZE})")
    args = parser.parse_args()

    VOCAB_SIZE = args.vocab_size  # permite sobreescribir antes de entrenar

    print(f"Directorio cache: {CACHE_DIR}")
    print()

    prepare_data(args.abc_dir, val_ratio=args.val_ratio)
    print()

    train_tokenizer()
    print()
    print("¡Listo! Ahora puedes ejecutar: uv run abc_train.py")
