#!/usr/bin/env python3
"""
================================================================================
                      ABC CONDITIONED  v1.1
        Generacion de melodias ABC condicionada a texto (lyrics)

  Extiende el motor GPT de autoresearch-abc (abc_train.py / abc_model.py)
  anyadiendo condicionamiento por embeddings de texto. Soporta un bucle de
  mejora autonomo compatible con el framework autoresearch (abc_conditioned_
  program.md), con dos fases de entrenamiento y transfer learning desde un
  modelo base no condicionado.

================================================================================
  ARQUITECTURA
================================================================================

  Motor base: GPT pequenyo con RoPE, QK-Norm, Value Embeddings alternos y
  activacion ReGLU^2 en MLP (identico a abc_model.py de autoresearch-abc).

  Condicionamiento: emotion_proj (Linear -> GELU -> Linear, 384 -> d_model)
  proyecta el embedding del verso actual y se SUMA al embedding de token en
  cada posicion de la secuencia, antes de las capas de atencion. El embedding
  cambia verso a verso durante la generacion, modulando el "color" melodico
  de cada frase sin romper la continuidad autoregresiva.

  Embeddings de texto: paraphrase-multilingual-MiniLM-L12-v2 (384 dims).
  Multilingue: funciona en espanol, ingles y otras lenguas sin ajuste.

  Tokenizacion: BPE sobre el texto ABC crudo (compatible con abc_prepare.py,
  mismo cache en ~/.cache/autoresearch-abc/). El modelo aprende la sintaxis
  completa de ABC (cabeceras, barras, duraciones, ornamentos) de forma
  implicita. Canciones sin "w:" se incluyen con embedding cero: el modelo
  aprende a generar libre cuando no hay condicion.

================================================================================
  FASES DE ENTRENAMIENTO (bucle autoresearch)
================================================================================

  FASE 1  (PHASE = 1) — Solo val_bpb
    Objetivo: que el modelo aprenda la sintaxis ABC y el condicionamiento
    basico antes de introducir senyal de preferencias.
    Metrica: val_bpb (bits por byte) — MENOR es mejor.
    Avance:  validity_ratio >= 0.5 en dos diagnosticos consecutivos.

  FASE 2  (PHASE = 2) — val_bpb + reward_score
    Objetivo: que el modelo genere musica que ademas suene bien segun tus
    preferencias personales.
    Genera PREF_EVAL_SAMPLES muestras ABC, las convierte a MIDI con abc2midi
    y las evalua con el evaluador seleccionable (EVAL_BACKEND o ABC_EVAL_BACKEND):
    preference_trainer.py o quality_scorer.py (ambos en escala 0-1); ver abc_eval.py.
    Metrica: combined_score = (1-PREF_WEIGHT)*(1/val_bpb) + PREF_WEIGHT*reward_score
             (reward_score = pref_score con preference, quality_score con quality; ambos 0-1)
    El agente (Claude Code) lee abc_conditioned_program.md para el bucle
    completo de experimentos, registro en results.tsv y reglas de keep/discard.

================================================================================
  TRANSFER LEARNING DESDE MODELO BASE
================================================================================

  Flujo recomendado: entrenar primero con abc_train.py (modelo no condicionado)
  hasta que genere ABC coherente (validity_ratio >= 0.5), luego transferir los
  pesos al modelo condicionado con --init-from. Asi emotion_proj arranca con
  un transformer base ya formado y converge mucho mas rapido.

  REQUISITO: d_model, n_layers y n_heads deben coincidir entre el base y el
  condicionado. Si no coinciden, los pesos tienen formas distintas y no se
  transfieren (el log de "Ignorados del base" te avisa).

  Con --freeze-base-epochs N: congela los pesos transferidos durante las
  primeras N epocas para que emotion_proj aprenda a alinearse con el espacio
  del transformer antes de que este se mueva. Util con corpus de lyrics pequenyo.
  A partir de la epoca N+1 se descongela todo y el optimizer se reconstruye.

================================================================================
  TRES MODOS DE INICIALIZACION (train)
================================================================================

  1. Por defecto             — inicializacion aleatoria desde cero
  2. --resume                — reanudar checkpoint condicionado existente
  3. --init-from base.pt     — transferir pesos de modelo base (abc_train.py)
       [--freeze-base-epochs N]  congelar base N epocas (default: 0)

  Al usar --init-from, el log muestra:
    [train]   Pesos transferidos:  47
    [train]   Ignorados del base:  0
    [train]   Nuevos (init fresh): 6
    [train]     + emotion_proj.0.weight  [128, 384]
    [train]     + emotion_proj.0.bias    [128]
    ...

================================================================================
  COMANDOS
================================================================================

    prepare   Prepara corpus BPE desde ficheros .abc (una sola vez)
    train     Entrena el modelo condicionado
    generate  Genera ABC condicionado a un fichero de lyrics
    inspect   Muestra info del checkpoint y/o corpus

================================================================================
  EJEMPLOS
================================================================================

  # 1. Preparar corpus BPE (una sola vez, compatible con abc_prepare.py)
  python abc_conditioned.py prepare --abc-dir ./corpus/

  # 2a. Entrenar desde cero
  python abc_conditioned.py train \\
      --corpus ./corpus/ --epochs 60 --model-out conditioned.pt

  # 2b. Transferir desde modelo base de abc_train.py (RECOMENDADO)
  python abc_conditioned.py train \\
      --corpus ./corpus/ \\
      --init-from ~/.cache/autoresearch-abc/model.pt \\
      --freeze-base-epochs 10 \\
      --epochs 60 --d-model 256 --n-layers 6 --n-heads 8 \\
      --model-out conditioned.pt

  # 2c. Reanudar entrenamiento condicionado
  python abc_conditioned.py train \\
      --corpus ./corpus/ --resume --model-out conditioned.pt

  # 3a. Generar condicionado a una letra (una linea = un verso)
  python abc_conditioned.py generate \\
      --model conditioned.pt \\
      --lyrics letra.txt --abc-out salida.abc --key G --mode major

  # 3b. Generar libre (sin lyrics, embedding cero, n frases)
  python abc_conditioned.py generate \\
      --model conditioned.pt --n-phrases 4 --abc-out salida.abc

  # 4. Inspeccionar modelo y corpus
  python abc_conditioned.py inspect --model conditioned.pt --corpus ./corpus/

================================================================================
  FORMATO w: EN EL CORPUS ABC
================================================================================

    X:1
    T:Ejemplo
    M:4/4
    L:1/8
    K:C
    C D E F | G2 A2 |
    w:Ho-la_ mun-do mu-si-cal a-qui
    E2 F2 | G4 |
    w:es-te es un e-jem-plo

    Cada linea "w:" justo despues de una linea de melodia es UN VERSO.
    El texto (ignorando guiones '-', ligaduras '_' y espacios '*') se
    embede con MiniLM y se usa como condicion para las notas de esa frase.
    Canciones sin ninguna "w:" se incluyen con embedding cero.

================================================================================
  VARIABLES QUE EL AGENTE MODIFICA (bucle autoresearch)
================================================================================

    PHASE                    1 o 2
    PREF_TEMPERATURE         temperatura de muestreo en evaluacion
    PREF_EVAL_SAMPLES        numero de muestras para validity_ratio / reward_score
    PREF_WEIGHT              peso de reward_score en combined_score (0.0-1.0)
    PREF_TOKENS_PER_PHRASE   tokens BPE por frase en evaluacion
    PREF_N_PHRASES           frases por muestra en evaluacion

    EVAL_BACKEND (preference|quality) lo fija el usuario al empezar la sesion;
    el agente NO debe cambiarlo entre experimentos.

    Hiperparametros de arquitectura (valores por defecto en build_parser):
    --epochs, --batch-size, --lr, --d-model, --n-heads, --n-layers,
    --max-seq-len, --label-smooth

================================================================================
  RESUMEN ESTRUCTURADO (salida de train, parseado por el agente)
================================================================================

    phase:            1
    val_bpb:          1.320000
    eval_backend:     preference
    reward_score:     N/A      (0-1, MAYOR es mejor)
    pref_score:       N/A
    quality_score:    N/A
    validity_ratio:   N/A
    combined_score:   N/A
    n_songs:          1200
    d_model:          128
    n_layer:          4
    n_head:           4
    n_params:         2340864
    mem_gb:           0.01
    train_seconds:    240

    Extraer con:
    grep "^phase:\\|^val_bpb:\\|^eval_backend:\\|^reward_score:\\|^validity_ratio:\\|^combined_score:" run.log

================================================================================
  DEPENDENCIAS
================================================================================

    Siempre:      torch, mido
    Embeddings:   sentence-transformers
                    pip install sentence-transformers
                  (descarga paraphrase-multilingual-MiniLM-L12-v2 ~470MB
                   la primera vez; cache en ~/.cache/torch/sentence_transformers/)
    BPE:          rustbpe, tiktoken  (igual que abc_prepare.py)
    Evaluacion:   abc2midi (sudo apt install abcmidi)
                  preference_trainer.py o quality_scorer.py  (solo PHASE = 2; ver abc_eval.py)
================================================================================
"""

import argparse
import gc
import json
import math
import os
import pickle
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import abc_eval

# ==============================================================================
#  CONSTANTES GLOBALES
# ==============================================================================

VERSION = "1.0"

# BPE / tokenizador (compatible con abc_prepare.py)
CACHE_DIR      = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-abc")
DATA_DIR       = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR  = os.path.join(CACHE_DIR, "tokenizer")
VOCAB_SIZE     = 512
SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN      = "<|reserved_0|>"
SEP_TOKEN      = "<|reserved_1|>"
SPLIT_PATTERN  = r"""[A-Za-z][0-9]*|[|:]+|[<>^_=]|\d+|[\r\n]+|\s+|[^\s]"""

# Parametros de entrenamiento por defecto
MAX_SEQ_LEN   = 512
TIME_BUDGET   = 300   # segundos por experimento (solo si --time-budget)
EVAL_TOKENS   = 200_000

# Embeddings de texto
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM        = 384

# Token especial de separacion de verso dentro de la secuencia BPE
# Se reserva como "<|reserved_2|>" para no colisionar con BOS/SEP
PHRASE_TOKEN = "<|reserved_2|>"

# ==============================================================================
#  CONTROL DE FASE  ← EL AGENTE MODIFICA ESTAS VARIABLES
# ==============================================================================

# PHASE = 1 : solo val_bpb. Sin evaluacion de preferencias.
#             Usar hasta que validity_ratio >= 0.5 en dos diagnosticos consecutivos.
# PHASE = 2 : val_bpb + pref_score combinados. Pipeline completo.
PHASE = 1                       # ← EL AGENTE MODIFICA ESTO

# Temperatura de muestreo usada en evaluacion (fase 2)
PREF_TEMPERATURE   = 0.9        # ← EL AGENTE MODIFICA ESTO

# Numero de muestras generadas para evaluar validity_ratio y pref_score
PREF_EVAL_SAMPLES  = 20         # ← EL AGENTE MODIFICA ESTO

# Peso de la señal de preferencias en combined_score (0.0 = solo bpb, 1.0 = solo prefs)
PREF_WEIGHT        = 0.3        # ← EL AGENTE MODIFICA ESTO

# Tokens BPE por frase en la evaluacion de muestras
PREF_TOKENS_PER_PHRASE = 60     # ← EL AGENTE MODIFICA ESTO

# Numero de frases por muestra en la evaluacion (generacion libre, embedding cero)
PREF_N_PHRASES     = 4          # ← EL AGENTE MODIFICA ESTO

# Evaluador de la fase 2 (ver abc_eval.py). También: ABC_EVAL_BACKEND=quality|preference
#   "preference" → preference_trainer.py (escala 0-1)
#   "quality"    → quality_scorer.py     (escala 0-1)
# Elegir UNO por sesión y NO cambiarlo entre experimentos (escalas no comparables).
EVAL_BACKEND        = "preference"
PREF_TRAINER_PATH   = None       # None = autodetectar (cwd, carpeta del proyecto, carpeta padre)
PREF_MODEL          = "prefs"    # solo backend "preference"
PREF_CORPUS         = "./corpus.npz"
QUALITY_SCORER_PATH = None       # None = autodetectar
QUALITY_EXTRA_ARGS  = []         # solo backend "quality"; p. ej. ["--fast"]

# Escalas para la mascara tonal (usada en generate)
SCALE_INTERVALS: dict = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "chromatic":        list(range(12)),
}

KEY_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

# ==============================================================================
#  TORCH (import perezoso)
# ==============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

def _require_torch():
    if not _TORCH_OK:
        raise ImportError("PyTorch no disponible. Instala con: pip install torch")

# ==============================================================================
#  EMBEDDING DE TEXTO (sentence-transformers, import perezoso)
# ==============================================================================

_EMBED_MODEL = None

def _load_embed_model(device="cpu"):
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers no disponible.\n"
            "  pip install sentence-transformers"
        )
    print(f"[embed] Cargando '{EMBEDDING_MODEL_NAME}' (puede tardar la primera vez)...")
    _EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _EMBED_MODEL


def embed_texts(texts, device="cpu", batch_size=64):
    """Devuelve ndarray (N, EMBEDDING_DIM). Textos vacios -> vector cero."""
    import numpy as np
    model = _load_embed_model(device=device)
    non_empty = [i for i, t in enumerate(texts) if t and t.strip()]
    result = [None] * len(texts)
    if non_empty:
        vecs = model.encode(
            [texts[i] for i in non_empty],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for idx, vec in zip(non_empty, vecs):
            result[idx] = vec
    zero = np.zeros(EMBEDDING_DIM, dtype="float32")
    for i in range(len(result)):
        if result[i] is None:
            result[i] = zero
    return np.stack(result)

# ==============================================================================
#  PARSER ABC MINIMAL  (extrae tokens BPE + lyrics por verso)
# ==============================================================================

def _clean_w_text(raw):
    """Limpia una linea w: eliminando marcadores de alineacion silabica."""
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    raw = raw.replace("-", "").replace("_", " ").replace("*", "")
    return re.sub(r"\s+", " ", raw).strip()


def parse_abc_for_conditioned(path):
    """
    Parsea un fichero .abc devolviendo lista de canciones. Cada cancion es:
      {
        "raw_text": str,          # texto ABC completo (para BPE)
        "phrase_texts": [str],    # texto de cada verso (para embeddings)
        "phrase_spans": [(i,j)],  # posicion (caracter) de cada verso en raw_text
                                  # NOTA: no se usa para BPE, solo informativo
      }

    Estrategia:
      - raw_text es el texto completo del bloque X: SIN las lineas w:.
        Asi el tokenizador BPE opera exactamente igual que en abc_prepare.py.
      - phrase_texts recoge las lineas w: en orden, limpiadas.
      - Canciones sin ninguna w: tienen phrase_texts=[""]; se incluyen igual
        (el modelo aprende a generar libre cuando no hay condicion).
    """
    songs = []
    current_block = []

    def _flush(block):
        if not block:
            return
        raw_lines = []
        w_lines = []
        for ln in block:
            if re.match(r"^\s*w:", ln):
                w_lines.append(ln)
            else:
                raw_lines.append(ln)
        raw_text = "\n".join(raw_lines)
        phrase_texts = [_clean_w_text(w) for w in w_lines] if w_lines else [""]
        if raw_text.strip():
            songs.append({"raw_text": raw_text, "phrase_texts": phrase_texts})

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if re.match(r"^\s*X\s*:", line) and current_block:
                _flush(current_block)
                current_block = [line]
            elif line.strip() == "" and current_block:
                _flush(current_block)
                current_block = []
            else:
                current_block.append(line)
    _flush(current_block)
    return songs


def collect_corpus(corpus_path):
    """Recorre una carpeta/fichero .abc y devuelve (songs, files)."""
    abc_files = []
    if os.path.isdir(corpus_path):
        for root, _, files in os.walk(corpus_path):
            for fn in sorted(files):
                if fn.lower().endswith(".abc"):
                    abc_files.append(os.path.join(root, fn))
    else:
        abc_files = [corpus_path]

    all_songs = []
    for f in abc_files:
        try:
            all_songs.extend(parse_abc_for_conditioned(f))
        except Exception as e:
            print(f"[aviso] error parseando {f}: {e}", file=sys.stderr)
    return all_songs, abc_files

# ==============================================================================
#  BPE / TOKENIZADOR  (igual que abc_prepare.py, compatible con su cache)
# ==============================================================================

def prepare_corpus(abc_dir, val_ratio=0.05):
    """Carga .abc, divide train/val y guarda en CACHE_DIR."""
    os.makedirs(DATA_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.txt")
    val_path   = os.path.join(DATA_DIR, "val.txt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"[prepare] Datos ya preparados en {DATA_DIR}")
        return

    songs, files = collect_corpus(abc_dir)
    if not songs:
        print(f"[prepare] No se encontraron ficheros .abc en {abc_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[prepare] {len(files)} ficheros, {len(songs)} canciones")

    texts = [s["raw_text"] for s in songs]
    rng = random.Random(42)
    rng.shuffle(texts)
    n_val = max(1, int(len(texts) * val_ratio))
    train_texts, val_texts = texts[n_val:], texts[:n_val]

    sep = f"\n{SEP_TOKEN}\n"
    Path(train_path).write_text(sep.join(train_texts), encoding="utf-8")
    Path(val_path).write_text(sep.join(val_texts),   encoding="utf-8")
    print(f"[prepare] {len(train_texts)} train / {len(val_texts)} val -> {DATA_DIR}")


def train_tokenizer():
    """Entrena tokenizador BPE sobre train.txt (compatible con abc_prepare.py)."""
    import rustbpe, tiktoken

    tok_pkl  = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    tok_pt   = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if os.path.exists(tok_pkl) and os.path.exists(tok_pt):
        print(f"[prepare] Tokenizador ya existe en {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    train_path = os.path.join(DATA_DIR, "train.txt")
    if not os.path.exists(train_path):
        print("[prepare] Ejecuta 'prepare' antes de entrenar el tokenizador.", file=sys.stderr)
        sys.exit(1)

    print("[prepare] Entrenando BPE...")
    t0 = time.time()

    tokenizer = rustbpe.Tokenizer()
    content = Path(train_path).read_text(encoding="utf-8")[:5_000_000]
    chunk = 1000
    it = (content[i:i+chunk] for i in range(0, len(content), chunk))
    tokenizer.train_from_iterator(it, VOCAB_SIZE - len(SPECIAL_TOKENS), pattern=SPLIT_PATTERN)

    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    offset = len(mergeable_ranks)
    special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="abc_bpe",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    with open(tok_pkl, "wb") as f:
        pickle.dump(enc, f)

    # tabla token -> bytes (para calcular BPB)
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = [
        0 if enc.decode([i]) in special_set else len(enc.decode([i]).encode("utf-8"))
        for i in range(enc.n_vocab)
    ]
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), tok_pt)
    print(f"[prepare] BPE listo en {time.time()-t0:.1f}s, vocab={enc.n_vocab}")


def load_tokenizer():
    tok_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.exists(tok_pkl):
        print("[error] Tokenizador no encontrado. Ejecuta: prepare --abc-dir ...", file=sys.stderr)
        sys.exit(1)
    with open(tok_pkl, "rb") as f:
        enc = pickle.load(f)
    return enc


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _text_chunks(split: str, chunk_size: int = 512):
    """Iterador infinito de chunks del fichero train/val (igual que abc_prepare.py)."""
    path = os.path.join(DATA_DIR, f"{split}.txt")
    assert os.path.exists(path), f"No existe {path}"
    while True:
        content = Path(path).read_text(encoding="utf-8")
        for i in range(0, max(1, len(content) - chunk_size), chunk_size // 2):
            yield content[i:i + chunk_size]

# ==============================================================================
#  DATASET CONDICIONADO
#  Cada muestra: (token_ids, segment_embed_idx, phrase_texts)
#    token_ids:       secuencia BPE de longitud MAX_SEQ_LEN
#    segment_embed_idx: indice de frase (0-based) para cada posicion
#    phrase_texts:    lista de textos de los versos de esta cancion
# ==============================================================================

def build_conditioned_dataset(songs, enc, max_seq_len=MAX_SEQ_LEN):
    """
    Por cada cancion construye una muestra:
      - Tokeniza raw_text con el BPE del corpus (igual que abc_prepare.py).
      - Construye segment_embed_idx: array paralelo que indica, para cada
        posicion de la secuencia, que verso le corresponde.
      - Estrategia de asignacion: se divide la secuencia en N partes iguales
        (una por verso); cada posicion cae en la parte de su verso.
        Las canciones con un solo verso (o sin w:) asignan el indice 0
        a todas las posiciones (embedding cero si phrase_text=="").

    Devuelve lista de (token_ids, segment_embed_idx, phrase_texts).
    """
    bos_id  = enc.encode_single_token(BOS_TOKEN)
    dataset = []

    for song in songs:
        raw       = song["raw_text"]
        p_texts   = song["phrase_texts"]  # lista de versos
        n_phrases = len(p_texts)

        ids = [bos_id] + enc.encode_ordinary(raw)
        ids = ids[:max_seq_len]

        # Rellenar hasta max_seq_len con el token de padding (indice 0 del vocab,
        # que en tiktoken es el primer token BPE y normalmente no aparece en ABC).
        # Para evitar asumir un pad_id concreto usamos el mismo BOS como pad
        # (el loss se ignorara con la mascara de padding en train).
        pad_id = bos_id  # reutilizamos BOS como pad; se enmascara en la loss
        while len(ids) < max_seq_len:
            ids.append(pad_id)

        # Segment index: divido la secuencia en n_phrases bloques iguales
        T = max_seq_len
        seg_idx = [0] * T
        for pos in range(T):
            seg_idx[pos] = min(int(pos * n_phrases / T), n_phrases - 1)

        dataset.append((ids, seg_idx, p_texts))

    return dataset

# ==============================================================================
#  MODELO GPT CONDICIONADO
#  Extiende la arquitectura de abc_model.py anyadiendo emotion_proj.
# ==============================================================================

@dataclass
class ConditionedGPTConfig:
    sequence_len: int = 512
    vocab_size:   int = 512
    n_layer:      int = 4
    n_head:       int = 4
    n_embd:       int = 128
    emotion_dim:  int = EMBEDDING_DIM  # 384 por defecto (MiniLM)


def _norm(x):
    return F.rms_norm(x, (x.size(-1),))


def _has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def _apply_rotary(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class _CausalSelfAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        assert cfg.n_embd % cfg.n_head == 0
        self.c_q = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_k = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_v = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, cfg.n_head, bias=False)
            if _has_ve(layer_idx, cfg.n_layer) else None
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
        q, k = _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)
        q, k = _norm(q), _norm(k)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class _MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class _Block(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.attn = _CausalSelfAttention(cfg, layer_idx)
        self.mlp  = _MLP(cfg)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(_norm(x), ve, cos_sin)
        x = x + self.mlp(_norm(x))
        return x


class ConditionedGPT(nn.Module):
    """
    GPT de abc_model.py + emotion_proj para condicionamiento por texto.

    En el forward, segment_emotion (B, T, emotion_dim) se proyecta a d_model
    y se SUMA al embedding de nota + posicion en cada posicion, igual que en
    lyrics_melody.py. El resto de la arquitectura (RoPE, QK-Norm, Value
    Embeddings, residual lambdas) es identico a abc_model.py.
    """

    def __init__(self, cfg: ConditionedGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "h":   nn.ModuleList([_Block(cfg, i) for i in range(cfg.n_layer)]),
        })
        self.lm_head       = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(cfg.n_layer))
        self.x0_lambdas    = nn.Parameter(torch.zeros(cfg.n_layer))

        # Proyeccion de embedding emocional (384 -> n_embd)
        self.emotion_proj = nn.Sequential(
            nn.Linear(cfg.emotion_dim, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.n_embd),
        )

        # Value embeddings alternos (igual que abc_model.py)
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(cfg.vocab_size, cfg.n_embd)
            for i in range(cfg.n_layer) if _has_ve(i, cfg.n_layer)
        })

        head_dim = cfg.n_embd // cfg.n_head
        self.rotary_seq_len = cfg.sequence_len * 4
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        cr  = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv = 1.0 / (base ** (cr / head_dim))
        t   = torch.arange(seq_len, dtype=torch.float32)
        f   = torch.outer(t, inv)
        return f.cos()[None, :, None, :], f.sin()[None, :, None, :]

    def init_weights(self):
        nn.init.normal_(self.transformer.wte.weight, 0.0, 1.0)
        nn.init.normal_(self.lm_head.weight, 0.0, 0.001)
        s = 3 ** 0.5 * self.cfg.n_embd ** -0.5
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
        # Inicializacion pequenya para la proyeccion emocional
        for layer in self.emotion_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0.0, 0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, idx, segment_emotion, targets=None, reduction="mean"):
        """
        idx:             (B, T)  token ids
        segment_emotion: (B, T, emotion_dim)  embedding de verso por posicion
        targets:         (B, T)  ids objetivo para calcular la loss (opcional)
        """
        B, T = idx.size()
        cos  = self.cos[:, :T]
        sin  = self.sin[:, :T]

        # Embedding de nota + condicionamiento emocional
        x  = self.transformer.wte(idx)
        x  = x + self.emotion_proj(segment_emotion)   # <-- condicionamiento
        x0 = x

        ve_cache = {
            i: self.value_embeds[str(i)](idx)
            for i in range(self.cfg.n_layer) if _has_ve(i, self.cfg.n_layer)
        }
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            x = block(x, ve_cache.get(i), (cos, sin))

        logits = self.lm_head(_norm(x))
        if targets is None:
            return logits

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction=reduction,
        )

    def num_params(self):
        total = sum(p.numel() for p in self.parameters())
        emb   = self.transformer.wte.weight.numel()
        return {"total": total, "non_embedding": total - emb}

# ==============================================================================
#  DATALOADER CPU
#  Compatible con abc_prepare.py: usa el mismo train.txt / val.txt.
#  Para el condicionamiento, carga tambien los embeddings pre-calculados.
# ==============================================================================

def make_dataloader_cpu(enc, B, T, split, buffer_size=500):
    """
    Dataloader no condicionado: igual que abc_train.py (para BPB).
    Devuelve (x, y, epoch) sin segment embedding.
    """
    bos_id   = enc.encode_single_token(BOS_TOKEN)
    doc_buf  = []
    chunk_it = _text_chunks(split, chunk_size=T * 3)
    epoch    = 1

    def refill():
        nonlocal epoch
        chunk = next(chunk_it)
        docs  = [d.strip() for d in chunk.split(SEP_TOKEN) if d.strip()] or [chunk]
        doc_buf.extend(enc.encode_ordinary_batch(docs, num_threads=4))

    row_buf = torch.empty((B, T + 1), dtype=torch.long)
    while True:
        for row in range(B):
            pos = 0
            while pos < T + 1:
                while len(doc_buf) < buffer_size:
                    refill()
                remaining = T + 1 - pos
                best_i, best_l = -1, 0
                for i, doc in enumerate(doc_buf):
                    if len(doc) <= remaining and len(doc) > best_l:
                        best_i, best_l = i, len(doc)
                if best_i >= 0:
                    doc = doc_buf.pop(best_i)
                    row_buf[row, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    sh = min(range(len(doc_buf)), key=lambda i: len(doc_buf[i]))
                    doc = doc_buf.pop(sh)
                    row_buf[row, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        yield row_buf[:, :-1].clone(), row_buf[:, 1:].clone(), epoch

# ==============================================================================
#  EVALUACION BPB (sin condicionamiento, para seguimiento de calidad)
# ==============================================================================

@torch.no_grad()
def evaluate_bpb(model, enc, batch_size=2):
    token_bytes = get_token_bytes(device="cpu")
    val_loader  = make_dataloader_cpu(enc, batch_size, MAX_SEQ_LEN, "val")
    steps       = max(1, EVAL_TOKENS // (batch_size * MAX_SEQ_LEN))
    total_nats  = 0.0
    total_bytes = 0
    model.eval()
    import numpy as np
    zero_emb = torch.zeros(1, MAX_SEQ_LEN, EMBEDDING_DIM)
    for _ in range(steps):
        x, y, _ = next(val_loader)
        B = x.size(0)
        seg = zero_emb.expand(B, -1, -1)
        loss_flat = model(x, seg, y, reduction="none").view(-1)
        nbytes    = token_bytes[y.view(-1)]
        mask      = nbytes > 0
        total_nats  += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ==============================================================================
#  EVALUACION DE PREFERENCIAS  (solo PHASE == 2)
# ==============================================================================

def is_valid_abc(text):
    """Heuristica rapida: cabecera minima + al menos una nota + al menos una barra."""
    has_header = bool(re.search(r"^[XTMKL]:", text, re.MULTILINE))
    has_note   = bool(re.search(r"[A-Ga-gz]", text))
    has_bar    = "|" in text
    return has_header and has_note and has_bar


def _generate_sample(model, enc, cfg, device, n_phrases, tokens_per_phrase, temperature):
    """Genera una muestra ABC de n_phrases frases con embedding cero (generacion libre)."""
    import numpy as np
    bos_id = enc.encode_single_token(BOS_TOKEN)
    token_ids = [bos_id]
    zero_vec = np.zeros(cfg.emotion_dim, dtype="float32")

    model.eval()
    with torch.no_grad():
        for _ in range(n_phrases * tokens_per_phrase):
            window = token_ids[-cfg.sequence_len:]
            T      = len(window)
            seq    = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            seg    = torch.zeros(1, T, cfg.emotion_dim, device=device)
            logits = model(seq, seg)[0, -1, :]
            logits = logits / max(temperature, 1e-6)
            probs  = torch.softmax(logits, dim=-1)
            nxt    = torch.multinomial(probs, 1).item()
            token_ids.append(nxt)

    return enc.decode(token_ids[1:])


def evaluate_preferences(model, enc, cfg, device, lyrics_for_eval=None, backend=None):
    """
    Genera PREF_EVAL_SAMPLES muestras (generación libre, embedding cero), mide
    validity_ratio y las puntúa con el evaluador seleccionable de abc_eval.py:
    preference_trainer.py o quality_scorer.py (ambos devuelven 0-1).

    `lyrics_for_eval` se conserva por compatibilidad de firma; la generación
    sigue siendo libre.

    Devuelve dict con keys: backend, reward_score (0-1), pref_score, quality_score,
    validity_ratio, n_valid, n_total.
    """
    backend = abc_eval.resolve_backend(backend or EVAL_BACKEND)
    result = abc_eval.evaluate_samples(
        lambda: _generate_sample(
            model, enc, cfg, device,
            PREF_N_PHRASES, PREF_TOKENS_PER_PHRASE, PREF_TEMPERATURE,
        ),
        is_valid_abc, PREF_EVAL_SAMPLES, backend,
        pref_trainer_path=PREF_TRAINER_PATH, pref_model=PREF_MODEL,
        pref_corpus=PREF_CORPUS,
        quality_scorer_path=QUALITY_SCORER_PATH,
        quality_extra_args=QUALITY_EXTRA_ARGS,
    )
    result["n_total"] = PREF_EVAL_SAMPLES
    return result



# ==============================================================================
#  MODO PREPARE
# ==============================================================================

def cmd_prepare(args):
    prepare_corpus(args.abc_dir, val_ratio=args.val_ratio)
    _require_torch()
    train_tokenizer()
    print("[prepare] Listo. Ahora puedes ejecutar: train")

# ==============================================================================
#  MODO TRAIN
# ==============================================================================

def cmd_train(args):
    _require_torch()
    import numpy as np
    t0_train = time.time()
    backend = abc_eval.resolve_backend(EVAL_BACKEND)   # falla pronto si no es válido

    enc = load_tokenizer()
    vocab_size = enc.n_vocab
    print(f"[train] Vocab BPE: {vocab_size}")

    # Recoger corpus con lyrics
    print(f"[train] Leyendo corpus con lyrics desde {args.corpus} ...")
    all_songs, abc_files = collect_corpus(args.corpus)
    print(f"[train] {len(abc_files)} ficheros, {len(all_songs)} canciones")

    n_with_lyrics = sum(1 for s in all_songs if any(t for t in s["phrase_texts"]))
    print(f"[train] Canciones con al menos un verso: {n_with_lyrics}/{len(all_songs)}")

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[train] Device: {device}")

    # Dataset condicionado
    dataset = build_conditioned_dataset(all_songs, enc, max_seq_len=args.max_seq_len)
    print(f"[train] Muestras de entrenamiento: {len(dataset)}")

    # Pre-calcular embeddings por cancion (una vez, fuera del bucle)
    print("[train] Calculando embeddings de lyrics (puede tardar)...")
    song_embeddings = []  # lista de np.ndarray (n_phrases, 384)
    for _, _, p_texts in dataset:
        emb = embed_texts(p_texts, device=device)
        song_embeddings.append(emb)
    print(f"[train] Embeddings listos para {len(song_embeddings)} canciones.")

    # -------------------------------------------------------------------------
    # Configuracion del modelo — tres modos:
    #   1. --resume          : continuar entrenamiento condicionado (mismo .pt)
    #   2. --init-from <pt>  : transferir pesos de modelo base (abc_train.py)
    #   3. por defecto       : inicializacion aleatoria desde cero
    # -------------------------------------------------------------------------
    if args.resume and os.path.exists(args.model_out):
        # Modo 1: reanudar checkpoint condicionado
        print(f"[train] Reanudando desde {args.model_out} ...")
        ckpt_resume = torch.load(args.model_out, map_location="cpu")
        cfg = ConditionedGPTConfig(**ckpt_resume["config"])
        model = ConditionedGPT(cfg)
        model.load_state_dict(ckpt_resume["model_state"], strict=True)
        print("[train] Checkpoint condicionado cargado (strict).")

    elif args.init_from:
        # Modo 2: transferir desde modelo base no condicionado
        if not os.path.exists(args.init_from):
            print(f"[train] --init-from: fichero no encontrado: {args.init_from}",
                  file=sys.stderr)
            sys.exit(1)
        print(f"[train] Transfiriendo pesos desde modelo base: {args.init_from}")
        ckpt_base = torch.load(args.init_from, map_location="cpu")

        # La config del modelo condicionado la toma de los args (puede diferir
        # del base en emotion_dim, que el base no tiene).
        cfg = ConditionedGPTConfig(
            sequence_len = args.max_seq_len,
            vocab_size   = vocab_size,
            n_layer      = args.n_layers,
            n_head       = args.n_heads,
            n_embd       = args.d_model,
            emotion_dim  = EMBEDDING_DIM,
        )
        model = ConditionedGPT(cfg)
        model.init_weights()   # inicializar todo (incluido emotion_proj)

        # Cargar pesos compatibles con strict=False y loguear diferencias
        base_state  = ckpt_base.get("model_state", ckpt_base)
        cond_state  = model.state_dict()
        loaded, skipped_base, skipped_cond = [], [], []

        for k, v in base_state.items():
            if k in cond_state and cond_state[k].shape == v.shape:
                cond_state[k] = v
                loaded.append(k)
            else:
                skipped_base.append(k)

        for k in cond_state:
            if k not in base_state:
                skipped_cond.append(k)

        model.load_state_dict(cond_state, strict=True)

        print(f"[train]   Pesos transferidos:  {len(loaded)}")
        print(f"[train]   Ignorados del base:  {len(skipped_base)}"
              + (f" {skipped_base}" if skipped_base else ""))
        print(f"[train]   Nuevos (init fresh): {len(skipped_cond)}")
        for k in skipped_cond:
            print(f"[train]     + {k}  {list(cond_state[k].shape)}")

        # Congelar el transformer base las primeras epocas si se pide
        if args.freeze_base_epochs > 0:
            base_param_names = set(loaded)
            for name, param in model.named_parameters():
                if name in base_param_names:
                    param.requires_grad = False
            n_frozen = sum(1 for n, p in model.named_parameters() if not p.requires_grad)
            print(f"[train] Congelados {n_frozen} parametros del base "
                  f"durante {args.freeze_base_epochs} epocas.")

    else:
        # Modo 3: desde cero
        cfg = ConditionedGPTConfig(
            sequence_len = args.max_seq_len,
            vocab_size   = vocab_size,
            n_layer      = args.n_layers,
            n_head       = args.n_heads,
            n_embd       = args.d_model,
            emotion_dim  = EMBEDDING_DIM,
        )
        model = ConditionedGPT(cfg)
        model.init_weights()
        print("[train] Inicializacion aleatoria.")

    model.to(device)
    n_params = model.num_params()
    print(f"[train] Parametros: {n_params['total']:,} total  "
          f"({n_params['non_embedding']:,} sin embedding)")

    def _make_optimizer():
        trainable = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.1,
                                  betas=(0.9, 0.95))

    optimizer = _make_optimizer()

    # Arrays de entrenamiento
    note_ids_arr  = np.array([s[0] for s in dataset], dtype=np.int64)
    seg_idx_arr   = np.array([s[1] for s in dataset], dtype=np.int64)
    n_samples     = note_ids_arr.shape[0]
    batch_size    = min(args.batch_size, n_samples)
    T             = args.max_seq_len

    freeze_epochs = getattr(args, "freeze_base_epochs", 0)
    unfrozen      = (freeze_epochs == 0)   # ya descongelado si no hay freeze

    print(f"[train] Entrenando {args.epochs} epocas  "
          f"(d_model={cfg.n_embd}, layers={cfg.n_layer}, batch={batch_size}) ...")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):

        # Descongelar pesos del base al llegar a freeze_base_epochs
        if not unfrozen and epoch > freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = _make_optimizer()
            unfrozen  = True
            print(f"[train] Epoca {epoch}: pesos del base descongelados.")

        perm       = np.random.permutation(n_samples)
        total_loss = 0.0
        n_batches  = 0

        for i in range(0, n_samples, batch_size):
            idx  = perm[i:i + batch_size]
            B    = len(idx)

            note_ids = torch.tensor(note_ids_arr[idx], dtype=torch.long, device=device)
            seg_idx  = seg_idx_arr[idx]  # (B, T)

            # Construir tensor de segment embeddings (B, T, 384)
            seg_emb = np.zeros((B, T, EMBEDDING_DIM), dtype="float32")
            for b, song_idx in enumerate(idx):
                emb      = song_embeddings[song_idx]   # (n_phrases, 384)
                s_idx    = np.clip(seg_idx[b], 0, emb.shape[0] - 1)
                seg_emb[b] = emb[s_idx]
            seg_emb_t = torch.tensor(seg_emb, dtype=torch.float32, device=device)

            # Autoregresivo: inp = ids[:-1], tgt = ids[1:]
            inp_ids = note_ids[:, :-1]
            tgt_ids = note_ids[:, 1:]
            inp_seg = seg_emb_t[:, :-1, :]

            logits = model(inp_ids, inp_seg)
            # Ignorar el token BOS como objetivo de padding
            bos_id = enc.encode_single_token(BOS_TOKEN)
            loss   = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_ids.reshape(-1),
                ignore_index=bos_id,
                label_smoothing=args.label_smooth,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / max(1, n_batches)
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"[train] Epoca {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # -------------------------------------------------------------------------
    # Evaluar BPB final (embedding cero — comparable con abc_train.py)
    # -------------------------------------------------------------------------
    model.eval()
    val_bpb = evaluate_bpb(model, enc, batch_size=2)
    print(f"[train] val_bpb = {val_bpb:.6f}")

    # -------------------------------------------------------------------------
    # Fase 2: evaluacion de preferencias
    # -------------------------------------------------------------------------
    eval_result    = None
    validity_ratio = None
    combined_score = None

    if PHASE == 2:
        print(f"[train] Evaluador: {backend}")
        eval_result    = evaluate_preferences(model, enc, cfg, device, backend=backend)
        validity_ratio = eval_result["validity_ratio"]
        _cs = abc_eval.combined_score(val_bpb, eval_result["reward_score"], PREF_WEIGHT)
        combined_score = round(_cs, 6) if _cs is not None else None

    def _fmt(v, decimals=4):
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    # -------------------------------------------------------------------------
    # Guardar checkpoint con metricas
    # -------------------------------------------------------------------------
    t_elapsed = time.time() - t0_train
    n_params_total = model.num_params()["total"]

    ckpt = {
        "model_state":     model.state_dict(),
        "config":          asdict(cfg),
        "vocab_size":      vocab_size,
        "val_bpb":         val_bpb,
        "pref_score":      (eval_result or {}).get("pref_score"),
        "quality_score":   (eval_result or {}).get("quality_score"),
        "reward_score":    (eval_result or {}).get("reward_score"),
        "eval_backend":    backend,
        "validity_ratio":  validity_ratio,
        "combined_score":  combined_score,
        "phase":           PHASE,
        "version":         VERSION,
        "n_songs":         len(all_songs),
        "n_files":         len(abc_files),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "train_seconds":   round(t_elapsed, 1),
    }
    torch.save(ckpt, args.model_out)
    print(f"[train] Modelo guardado en {args.model_out} ({t_elapsed:.0f}s)")

    # -------------------------------------------------------------------------
    # RESUMEN ESTRUCTURADO — el agente parsea estas lineas exactas
    # -------------------------------------------------------------------------
    mem_gb = round(n_params_total * 4 / 1e9, 2)
    print()
    print(f"phase:            {PHASE}")
    print(f"val_bpb:          {_fmt(val_bpb, 6)}")
    for _line in abc_eval.summary_lines(backend, eval_result):
        print(_line)
    print(f"validity_ratio:   {_fmt(validity_ratio, 3)}")
    print(f"combined_score:   {_fmt(combined_score, 6)}")
    print(f"n_songs:          {len(all_songs)}")
    print(f"d_model:          {cfg.n_embd}")
    print(f"n_layer:          {cfg.n_layer}")
    print(f"n_head:           {cfg.n_head}")
    print(f"n_params:         {n_params_total}")
    print(f"mem_gb:           {mem_gb}")
    print(f"train_seconds:    {t_elapsed:.0f}")

# ==============================================================================
#  MODO GENERATE
# ==============================================================================

NOTE_RE = re.compile(
    r"(?P<acc>\^\^|\^|=|__|_)?"
    r"(?P<note>[A-Ga-g])"
    r"(?P<oct>[,']*)"
    r"(?P<dur>\d+/\d+|\d+|/\d+|/+)?"
)

def _parse_key_pc(key_str):
    m = re.match(r"\s*([A-G])([#b]?)", key_str)
    if not m:
        return 0
    return KEY_TO_PC.get(m.group(1) + m.group(2), KEY_TO_PC.get(m.group(1), 0))


def _build_scale_mask(vocab_size, tok_ids_with_pc, key_pc, mode):
    """
    Devuelve tensor bool (vocab_size,): True = token FUERA de la escala.
    tok_ids_with_pc: lista de (token_id, pitch_class) para todos los tokens
    que corresponden a notas MIDI.
    """
    scale_pcs = set((key_pc + iv) % 12 for iv in SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"]))
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tok_id, pc in tok_ids_with_pc:
        if pc not in scale_pcs:
            mask[tok_id] = True
    return mask


def lyrics_to_lines(text):
    return [l.strip() for l in text.splitlines() if l.strip()]


def cmd_generate(args):
    _require_torch()
    import numpy as np

    # --- Cargar modelo ---
    if not args.model or not os.path.exists(args.model):
        print(f"[generate] Checkpoint no encontrado: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"[generate] Cargando modelo desde {args.model} ...")
    ckpt = torch.load(args.model, map_location="cpu")
    cfg  = ConditionedGPTConfig(**ckpt["config"])
    model = ConditionedGPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[generate] Modelo: d_model={cfg.n_embd} layers={cfg.n_layer} "
          f"seq={cfg.sequence_len} val_bpb={ckpt.get('val_bpb','?'):.4f}")

    enc = load_tokenizer()
    bos_id = enc.encode_single_token(BOS_TOKEN)
    vocab_size = enc.n_vocab

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    model.to(device)

    # --- Lyrics: calcular embeddings ---
    if args.lyrics:
        with open(args.lyrics, "r", encoding="utf-8") as f:
            raw = f.read()
        line_texts = lyrics_to_lines(raw)
        if not line_texts:
            print("[generate] El fichero de lyrics esta vacio.", file=sys.stderr)
            sys.exit(1)
        print(f"[generate] {len(line_texts)} versos")
        print("[generate] Calculando embeddings de lyrics...")
        line_embeds = embed_texts(line_texts, device=device)  # (N, 384)
    else:
        # Generacion libre: n_phrases versos con embedding cero
        n = args.n_phrases
        line_texts  = [f"Verso {i+1}" for i in range(n)]
        line_embeds = np.zeros((n, EMBEDDING_DIM), dtype="float32")
        print(f"[generate] Modo libre ({n} frases, embedding cero)")

    n_lines = len(line_texts)

    # --- Estimacion de tokens por verso ---
    tokens_per_phrase = args.tokens_per_phrase
    total_max = tokens_per_phrase * n_lines + n_lines + 10

    # --- Mascara de escala (opcional) ---
    scale_mask = None
    if args.key and args.mode and args.mode != "chromatic":
        key_pc = _parse_key_pc(args.key)
        # No tenemos un vocabulario de notas estructurado como lyrics_melody.py
        # (aqui es BPE raw), asi que la mascara no se puede aplicar directamente.
        # En su lugar, se aplica un suavizado post-generacion si se especifica --key.
        print(f"[generate] Tonalidad: {args.key} {args.mode} "
              f"(la mascara de escala no se aplica sobre BPE crudo; "
              f"se anota en la cabecera K: del ABC de salida)")

    # --- Generacion autoregresiva ---
    print("[generate] Generando...")
    token_ids = [bos_id]
    phrase_token_counts = [0] * n_lines
    current_phrase = 0

    # Embedding del verso actual como tensor (1, 1, 384)
    def _get_seg_embed(phrase_idx):
        vec = line_embeds[phrase_idx]
        return torch.tensor(vec, dtype=torch.float32, device=device).view(1, 1, cfg.emotion_dim)

    with torch.no_grad():
        steps = 0
        while steps < total_max:
            # Ventana de contexto
            window = token_ids[-cfg.sequence_len:]
            T      = len(window)
            seq    = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)

            # Segment embedding: usamos el del verso actual para TODA la ventana.
            # Es la misma aproximacion que lyrics_melody.py generate_transformer:
            # el historial de notas da continuidad; el embedding cambia el "color"
            # emocional de las proximas notas.
            cur_phrase_clamped = min(current_phrase, n_lines - 1)
            seg_vec = line_embeds[cur_phrase_clamped]
            seg_emb = torch.tensor(seg_vec, dtype=torch.float32, device=device)
            seg_emb = seg_emb.view(1, 1, cfg.emotion_dim).expand(1, T, cfg.emotion_dim)

            logits     = model(seq, seg_emb)
            next_logits = logits[0, -1, :].clone()

            # Temperatura
            next_logits = next_logits / max(args.temperature, 1e-6)

            # Top-k
            if args.top_k > 0:
                v, _ = torch.topk(next_logits, min(args.top_k, next_logits.size(-1)))
                next_logits[next_logits < v[-1]] = float("-inf")

            probs   = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            token_ids.append(next_id)

            # Contar tokens del verso actual
            phrase_token_counts[cur_phrase_clamped] += 1

            # Avanzar verso si hemos generado suficientes tokens
            if phrase_token_counts[cur_phrase_clamped] >= tokens_per_phrase:
                current_phrase += 1
                if current_phrase >= n_lines:
                    break

            steps += 1

    # --- Decodificar a texto ABC ---
    generated_text = enc.decode(token_ids[1:])  # excluir BOS

    # --- Construir fichero ABC con cabecera y lyrics ---
    title = os.path.basename(args.lyrics) if args.lyrics else "generado"
    key   = args.key or "C"
    meter = args.meter or "4/4"

    abc_lines = [
        "X:1",
        f"T:{title}",
        f"M:{meter}",
        "L:1/8",
        f"K:{key}",
    ]

    # Intentamos distribuir el texto generado en N bloques (uno por verso)
    # y anyadir las lineas w: correspondientes.
    # El BPE no genera barras de forma garantizada, pero si hay '|' en el
    # output los usamos como delimitadores naturales de compas.
    raw_abc_body = generated_text.strip()

    # Dividir por compases (barras) para repartir los versos
    bars = [b.strip() for b in raw_abc_body.split("|") if b.strip()]
    bars_per_phrase = max(1, len(bars) // n_lines)

    phrase_idx = 0
    for i, bar in enumerate(bars):
        abc_lines.append(bar + " |")
        if (i + 1) % bars_per_phrase == 0 and phrase_idx < n_lines:
            abc_lines.append(f"w: {line_texts[phrase_idx]}")
            phrase_idx += 1

    # Versos que no tuvieron compas asociado
    while phrase_idx < n_lines:
        abc_lines.append(f"w: {line_texts[phrase_idx]}")
        phrase_idx += 1

    abc_output = "\n".join(abc_lines) + "\n"

    # --- Escribir salida ---
    if args.abc_out:
        Path(args.abc_out).write_text(abc_output, encoding="utf-8")
        print(f"[generate] ABC escrito en {args.abc_out}")
    else:
        print("\n" + "="*60)
        print(abc_output)

    # Convertir a MIDI si se pide
    if args.midi_out:
        try:
            import mido, tempfile, subprocess
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".abc",
                                              delete=False, encoding="utf-8")
            out_abc = args.abc_out or tmp.name
            if not args.abc_out:
                tmp.write(abc_output)
                tmp.close()
                out_abc = tmp.name
            result = subprocess.run(
                ["abc2midi", out_abc, "-o", args.midi_out],
                capture_output=True, timeout=15,
            )
            if result.returncode == 0:
                print(f"[generate] MIDI escrito en {args.midi_out}")
            else:
                print(f"[generate] abc2midi fallo: {result.stderr.decode()}", file=sys.stderr)
            if not args.abc_out:
                os.unlink(out_abc)
        except FileNotFoundError:
            print("[generate] abc2midi no encontrado. Instala: sudo apt install abcmidi",
                  file=sys.stderr)

# ==============================================================================
#  MODO INSPECT
# ==============================================================================

def cmd_inspect(args):
    if args.model:
        if not _TORCH_OK:
            print("[inspect] PyTorch no disponible.", file=sys.stderr)
        else:
            ckpt = torch.load(args.model, map_location="cpu")
            print(f"Modelo: {args.model}")
            print(f"  Version: {ckpt.get('version', '?')}")
            print(f"  val_bpb: {ckpt.get('val_bpb', '?')}")
            print(f"  Canciones de entrenamiento: {ckpt.get('n_songs', '?')}")
            print(f"  Ficheros fuente: {ckpt.get('n_files', '?')}")
            print(f"  Modelo de embeddings: {ckpt.get('embedding_model', EMBEDDING_MODEL_NAME)}")
            print(f"  Config:")
            for k, v in ckpt.get("config", {}).items():
                print(f"    {k}: {v}")

    if args.corpus:
        songs, files = collect_corpus(args.corpus)
        print(f"\nCorpus: {args.corpus}")
        print(f"  Ficheros .abc: {len(files)}")
        print(f"  Canciones: {len(songs)}")
        n_with_lyrics = sum(1 for s in songs if any(t for t in s["phrase_texts"]))
        print(f"  Canciones con w:: {n_with_lyrics}/{len(songs)}")
        if songs:
            n_phrases = [len(s["phrase_texts"]) for s in songs]
            print(f"  Versos por cancion: min={min(n_phrases)} max={max(n_phrases)} "
                  f"media={sum(n_phrases)/len(n_phrases):.1f}")
            all_texts = [t for s in songs for t in s["phrase_texts"] if t]
            print(f"  Versos con texto: {len(all_texts)}")
            if all_texts:
                lens = [len(t) for t in all_texts]
                print(f"  Longitud de verso: min={min(lens)} max={max(lens)} "
                      f"media={sum(lens)/len(lens):.1f} chars")

    if not args.model and not args.corpus:
        print("Indica --model y/o --corpus.")

# ==============================================================================
#  CLI
# ==============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Generacion de ABC condicionada a lyrics (GPT + MiniLM)",
    )
    p.add_argument("--version", action="version", version=f"abc_conditioned {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    # --- prepare ---
    pp = sub.add_parser("prepare", help="Preparar corpus BPE desde .abc")
    pp.add_argument("--abc-dir",   required=True, help="Carpeta con .abc")
    pp.add_argument("--val-ratio", type=float, default=0.05)
    pp.set_defaults(func=cmd_prepare)

    # --- train ---
    pt = sub.add_parser("train", help="Entrenar GPT condicionado")
    pt.add_argument("--corpus",      required=True,
                    help="Carpeta con .abc (usada para extraer lyrics + embeddings)")
    pt.add_argument("--model-out",   default="abc_conditioned.pt")
    pt.add_argument("--resume",      action="store_true",
                    help="Reanudar entrenamiento condicionado desde --model-out")
    pt.add_argument("--init-from",   default=None, metavar="PT",
                    help="Checkpoint de modelo base (abc_train.py) del que "
                         "transferir pesos. Incompatible con --resume.")
    pt.add_argument("--freeze-base-epochs", type=int, default=0, metavar="N",
                    help="Con --init-from: congelar pesos del base las primeras N epocas "
                         "para que emotion_proj aprenda sin perturbarlos. "
                         "0 = no congelar (default).")
    pt.add_argument("--epochs",      type=int,   default=60)
    pt.add_argument("--batch-size",  type=int,   default=8)
    pt.add_argument("--lr",          type=float, default=3e-4)
    pt.add_argument("--d-model",     type=int,   default=128)
    pt.add_argument("--n-heads",     type=int,   default=4)
    pt.add_argument("--n-layers",    type=int,   default=4)
    pt.add_argument("--max-seq-len", type=int,   default=MAX_SEQ_LEN)
    pt.add_argument("--label-smooth",type=float, default=0.1)
    pt.add_argument("--cpu",         action="store_true")
    pt.set_defaults(func=cmd_train)

    # --- generate ---
    pg = sub.add_parser("generate", help="Generar ABC condicionado a lyrics")
    pg.add_argument("--model",            required=True)
    pg.add_argument("--lyrics",           default=None,
                    help="Fichero de texto (una linea = un verso). "
                         "Si se omite, genera libre con --n-phrases versos.")
    pg.add_argument("--n-phrases",        type=int, default=4,
                    help="Numero de frases en generacion libre (default: 4)")
    pg.add_argument("--abc-out",          default=None,
                    help="Fichero .abc de salida (si se omite, imprime en stdout)")
    pg.add_argument("--midi-out",         default=None,
                    help="Fichero .mid de salida (requiere abc2midi)")
    pg.add_argument("--key",              default="C",
                    help="Tonalidad para la cabecera K: (ej: C, G, Eb, Am)")
    pg.add_argument("--mode",             default="major",
                    choices=list(SCALE_INTERVALS.keys()),
                    help="Modo de escala (solo informativo en la cabecera; "
                         "la mascara no se aplica sobre BPE crudo)")
    pg.add_argument("--meter",            default="4/4")
    pg.add_argument("--temperature",      type=float, default=0.9)
    pg.add_argument("--top-k",            type=int,   default=0)
    pg.add_argument("--tokens-per-phrase",type=int,   default=60,
                    help="Tokens BPE a generar por verso (default: 60)")
    pg.add_argument("--cpu",              action="store_true")
    pg.set_defaults(func=cmd_generate)

    # --- inspect ---
    pi = sub.add_parser("inspect", help="Informacion de un modelo o corpus")
    pi.add_argument("--model",  default=None)
    pi.add_argument("--corpus", default=None)
    pi.set_defaults(func=cmd_inspect)

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
