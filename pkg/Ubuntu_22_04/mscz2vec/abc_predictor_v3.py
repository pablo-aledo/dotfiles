#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  abc_predictor_v3.py — Predictor de música ABC + compresor aritmético    ║
# ║                                                                          ║
# ║  Idea (3Blue1Brown, "Compression is Intelligence"):                      ║
# ║    predictor  ≡  compresor.   bits ≈ Σ -log2 P_modelo(símbolo|contexto)  ║
# ║    entropía cruzada = bits/carácter (bpc)  → la métrica del modelo.      ║
# ║                                                                          ║
# ║  Subcomandos:                                                            ║
# ║    synth       genera un corpus ABC sintético (para pruebas)             ║
# ║    prepare     limpia/une ficheros .abc en un corpus de texto            ║
# ║    train       entrena un modelo (--backend ngram | torch)               ║
# ║    sweep       barrido orden/gamma/minlen del backend ngram              ║
# ║    bpc         bits/carácter sobre un fichero (vs gzip/bz2/xz)           ║
# ║    compress   / decompress   código aritmético con el modelo (sin pérd.) ║
# ║    surprise    sorpresa (bits) por carácter y por compás                 ║
# ║    generate    muestrea música ABC del modelo (--constrain: solo ABC     ║
# ║                válido: compases con la duración exacta, ligaduras, etc.) ║
# ║    info        datos del modelo                                          ║
# ║  Novedades v2 (el formato .abcp y las probabilidades NO cambian):        ║
# ║    generate  --chords --rhythms 1/2,1,2|auto:N --prior-strength S        ║
# ║              --form copy|AABA --curve arch|rise|a,b,c --style reel       ║
# ║              --best-of N --report --midi f.mid --expressive              ║
# ║    render      ABC → MIDI expresivo (humanización, rubato, acompañ.)     ║
# ║    rank        puntúa/ordena piezas ABC por estructura (+bpc)            ║
# ║    prepare --minimal --keep LETRAS   (conserva R: para --style)          ║
# ║                                                                          ║
# ║  Novedades v3 (gramática; el formato .abcp NO cambia):                   ║
# ║    generate  --grammar CORPUS.abc --grammar-bars N --p-grammar P         ║
# ║              --max-elem N --grammar-inventory repair|ngram|bars          ║
# ║              --grammar-align --grammar-temp T --grammar-w succ=0.5,pos=1 ║
# ║      Inyecta ELEMENTOS de una gramática Re-Pair sobre compases (frases   ║
# ║      recurrentes del corpus) en vez de muestrear carácter a carácter;    ║
# ║      el autómata y el predictor los procesan como cualquier carácter.    ║
# ║      Idea y banco de pruebas: grammar_sampler.py. Compatible con --form, ║
# ║      --curve, --best-of, --rhythms, --expressive...                      ║
# ║    grammar   estadísticas de la gramática de un corpus ABC               ║
# ║                                                                          ║
# ║  Backends:                                                               ║
# ║    ngram  Witten-Bell interpolado (orden 0..N) + match model. Solo       ║
# ║           numpy; ADAPTATIVO (aprende mientras comprime, como en el vídeo)║
# ║    torch  GRU a nivel de carácter (imports perezosos).                   ║
# ║                                                                          ║
# ║  AVISO: emisor y receptor deben calcular probabilidades IDÉNTICAS.       ║
# ║  Usa la misma máquina/versión de librerías en ambos extremos.            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import argparse
import bz2
import hashlib
import lzma
import math
import os
import pickle
import random
import re
import struct
import sys
import zlib
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

# ── ANSI ────────────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()
def _c(code, s):
    return f"\033[{code}m{s}\033[0m" if _TTY else s
def bold(s):  return _c("1", s)
def dim(s):   return _c("2", s)
def green(s): return _c("32", s)
def yellow(s): return _c("33", s)
def red(s):   return _c("31", s)
def cyan(s):  return _c("36", s)

def need(cond, msg):
    if not cond:
        die(msg)

def die(msg):
    print(red(f"error: {msg}"), file=sys.stderr)
    sys.exit(1)

MAGIC = b"ABP2"
HDR_LEN = 22                   # magic4 backend1 flag1 huella8 n4 crc32 4
MIN_BITS_PER_SYM = 0.0056       # cota inferior real del coste por símbolo (cuantización a 2^16)
FREQ_TOTAL = 1 << 16          # precisión de las frecuencias cuantizadas
BACKENDS = {"ngram": 1, "torch": 2}

# ═══════════════════════════════════════════════════════════════════════════
#  CORPUS ABC
# ═══════════════════════════════════════════════════════════════════════════
def split_tunes(text):
    """Divide un texto ABC en tunes (cada una empieza en 'X:')."""
    tunes, cur = [], []
    for line in text.splitlines():
        if line.startswith("X:") and cur:
            tunes.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        tunes.append("\n".join(cur))
    return [t for t in tunes if t.strip()]

def clean_tune(t, minimal, keep=""):
    """minimal: solo M: L: K: Q: R: + cuerpo (más las letras de `keep`). Las líneas R: (ritmo/estilo)
    se colocan justo antes de K: para que --style condicione al modelo con el mismo orden."""
    keepset = set("MLKQR") | set(keep.upper())
    out, rl = [], []
    for line in t.splitlines():
        s = line.rstrip()
        if not s or s.startswith("%"):
            continue
        if minimal and len(s) > 1 and s[1] == ":" and s[0].isalpha() and s[0] not in keepset:
            continue
        if s.startswith("R:"):
            rl.append(s)
            continue
        if s.startswith("K:") and rl:
            out += rl
            rl = []
        out.append(s)
    return "\n".join(out + rl) + "\n"

def collect_abc_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                files += [os.path.join(root, f) for f in sorted(fs)
                          if f.lower().endswith((".abc", ".txt"))]
        else:
            files.append(p)
    return files

def cmd_prepare(a):
    files = collect_abc_files(a.inputs)
    if not files:
        die("no hay ficheros de entrada")
    tunes = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            tunes += split_tunes(fh.read())
    out = "".join(clean_tune(t, a.minimal, a.keep or "") for t in tunes)
    out = out.encode("ascii", errors="replace").decode("ascii")   # vocab de 1 byte
    out = out.replace("\t", " ")
    out = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", "", out)                  # sin NUL ni controles
    with open(a.output, "w", encoding="ascii") as fh:
        fh.write(out)
    print(green(f"✓ {len(tunes)} tunes, {len(out):,} caracteres → {a.output}"))

# Corpus sintético (solo para pruebas; estilo folk simple)
def synth_tunes(n, seed=0):
    rng = random.Random(seed)
    scales = {"C": "CDEFGABcdefg", "G": "GABcdefga", "D": "DEFGABcdef",
              "Am": "ABcdefga", "Em": "EFGABcde"}
    tunes = []
    for i in range(n):
        key = rng.choice(list(scales))
        sc = scales[key]
        meter = rng.choice(["4/4", "6/8", "3/4"])
        per_bar = {"4/4": 8, "6/8": 6, "3/4": 6}[meter]

        def motif():
            pos = rng.randrange(len(sc))
            m = []
            for _ in range(per_bar):
                pos = max(0, min(len(sc) - 1, pos + rng.choice([-2, -1, -1, 0, 1, 1, 2])))
                m.append(sc[pos] + rng.choice(["", "", "", "2"]))
            return "".join(m)

        motifs = [motif() for _ in range(3)]
        bars = []
        for _ in range(8):
            bars.append(rng.choice(motifs) if rng.random() < 0.75 else motif())
        part = "|".join(bars[:4]) + "|" + "|".join(bars[4:])
        body = f"|:{part}:|\n|:{part}|{rng.choice(motifs)}|]\n"
        tunes.append(f"X:{i+1}\nT:Tune {i+1}\nM:{meter}\nL:1/8\nK:{key[0]}{'m' if key.endswith('m') else ''}\n{body}")
    return tunes

def cmd_synth(a):
    t = synth_tunes(a.n, a.seed)
    with open(a.output, "w") as fh:
        fh.write("\n".join(t))
    print(green(f"✓ {a.n} tunes sintéticos → {a.output}"))

# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTORES  (interfaz: probs() -> ndarray[256]; update(sym, learn=True))
# ═══════════════════════════════════════════════════════════════════════════
class NgramPredictor:
    """Witten-Bell interpolado de orden 0..N + match model (repeticiones largas).

    Todo determinista (numpy float64). Es adaptativo: update() actualiza
    contadores, de modo que el decodificador reproduce exactamente el aprendizaje
    del codificador (idea del vídeo: 'ajustar el modelo sobre la marcha').
    """
    name = "ngram"
    meta = None      # estadísticas del corpus (v2): len_hist, int_hist...; no afectan a probs()
    _log = None
    _mk = None

    def __init__(self, order=6, minlen=6, gamma=1.0):
        if not (0 <= order <= 32 and minlen >= 1 and gamma > 0):
            raise ValueError("parámetros inválidos: 0<=order<=32, minlen>=1, gamma>0")
        self.order, self.minlen, self.gamma = order, minlen, gamma
        self.tables = [dict() for _ in range(order + 1)]   # k -> {ctx: [counts, total]}
        self.hist = bytearray()
        self.hashidx = {}
        self.match_ptr = 0
        self.match_len = 0
        self.hits = [1] * 33
        self.tot = [2] * 33

    def probs(self):
        h, L = self.hist, len(self.hist)
        p = np.full(256, 1.0 / 256.0)
        for k in range(self.order + 1):
            if k > L:
                break
            e = self.tables[k].get(bytes(h[L - k:]))
            if e is None:
                break
            cnt, n = e
            lam = n / (n + self.gamma * len(cnt))
            v = np.zeros(256)
            for s, c in cnt.items():
                v[s] = c
            p = lam * (v / n) + (1.0 - lam) * p
        if self.match_len > 0:
            b = min(self.match_len, 32)
            w = min(0.98, self.hits[b] / self.tot[b])
            pred = h[self.match_ptr]
            p = p * (1.0 - w)
            p[pred] += w
        return p / p.sum()

    def update(self, sym, learn=True):
        h = self.hist
        L = len(h)
        if learn:
            for k in range(min(self.order, L) + 1):
                key = bytes(h[L - k:])
                e = self.tables[k].get(key)
                if e is None:
                    e = [{}, 0]
                    self.tables[k][key] = e
                e[0][sym] = e[0].get(sym, 0) + 1
                e[1] += 1
            if self.match_len > 0:
                b = min(self.match_len, 32)
                self.tot[b] += 1
                if h[self.match_ptr] == sym:
                    self.hits[b] += 1
        if self.match_len > 0 and h[self.match_ptr] == sym:
            self.match_len += 1
            self.match_ptr += 1
        else:
            self.match_len = 0
        h.append(sym)
        L += 1
        if L >= self.minlen:
            key = bytes(h[L - self.minlen:])
            if self.match_len == 0:
                cand = self.hashidx.get(key)
                if cand is not None:
                    l = 0
                    while l < 32 and cand - 1 - l >= 0 and h[cand - 1 - l] == h[L - 1 - l]:
                        l += 1
                    if l >= self.minlen:
                        self.match_ptr, self.match_len = cand, l
            if self._log is not None:
                self._log.append((key, self.hashidx.get(key)))
            self.hashidx[key] = L

    # ── marcar/rebobinar (solo válido mientras se actualiza con learn=False: generación) ──
    def mark(self):
        self._mk = (len(self.hist), self.match_ptr, self.match_len)
        self._log = []

    def rewind(self):
        n, ptr, ml = self._mk
        for key, old in reversed(self._log):
            if old is None:
                self.hashidx.pop(key, None)
            else:
                self.hashidx[key] = old
        del self.hist[n:]
        self.match_ptr, self.match_len = ptr, ml
        self._log = []

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop("_log", None)
        d.pop("_mk", None)
        return d

    def size_bytes(self):
        return len(pickle.dumps(self, protocol=4))


def _torch_build(cfg):
    import torch.nn as nn
    class GRULM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(256, cfg["emb"])
            self.rnn = nn.GRU(cfg["emb"], cfg["hidden"], cfg["layers"], batch_first=True)
            self.out = nn.Linear(cfg["hidden"], 256)
        def forward(self, x, h=None):
            y, h = self.rnn(self.emb(x), h)
            return self.out(y), h
    return GRULM()


class TorchPredictor:
    """GRU carácter a carácter. Estático (no aprende al comprimir).
    Símbolo 0 (NUL) = BOS: nunca aparece en ABC de texto."""
    name = "torch"

    def __init__(self, path):
        import torch
        torch.set_num_threads(1)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.torch = torch
        self.cfg = ck["cfg"]
        self.model = _torch_build(self.cfg)
        self.model.load_state_dict(ck["state"])
        self.model.eval()
        self.h = None
        self._logits = None
        self._step(0)

    def _step(self, tok):
        torch = self.torch
        with torch.no_grad():
            y, self.h = self.model(torch.tensor([[tok]]), self.h)
            self._logits = y[0, -1].double()

    def probs(self):
        p = self.torch.softmax(self._logits, 0).numpy().astype(np.float64)
        return p / p.sum()

    def update(self, sym, learn=True):
        self._step(sym)

    def size_bytes(self):
        return sum(p.numel() for p in self.model.parameters()) * 4


class _SafeUnpickler(pickle.Unpickler):
    """Solo admite NgramPredictor y bytearray: un .pkl ajeno no puede ejecutar código."""
    def find_class(self, module, name):
        if name == "NgramPredictor" and module in ("__main__", __name__, "abc_predictor", "abc_predictor_v3"):
            return NgramPredictor
        if (module, name) == ("builtins", "bytearray"):
            return bytearray
        raise pickle.UnpicklingError(f"clase no permitida en el modelo: {module}.{name}")

def load_predictor(path, backend=None):
    if backend is None:
        backend = "torch" if path.endswith(".pt") else "ngram"
    if backend == "torch":
        try:
            import torch  # noqa: F401
        except ImportError:
            die("el backend torch requiere PyTorch (pip install torch)")
        try:
            return TorchPredictor(path)
        except OSError:
            raise
        except Exception as e:
            die(f"{path}: no es un modelo torch válido ({e.__class__.__name__}: {e})")
    try:
        with open(path, "rb") as fh:
            obj = _SafeUnpickler(fh).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, ImportError,
            IndexError, KeyError, TypeError) as e:
        die(f"{path}: no es un modelo ngram válido ({e.__class__.__name__}: {e})")
    if not isinstance(obj, NgramPredictor):
        die(f"{path}: no es un modelo ngram válido")
    return obj

def fingerprint(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).digest()[:8]

# ═══════════════════════════════════════════════════════════════════════════
#  CODIFICADOR ARITMÉTICO ENTERO (Witten-Neal-Cleary, 32 bits)
#  Es la versión "con renormalización" que menciona el vídeo (Rissanen/Pascoe)
# ═══════════════════════════════════════════════════════════════════════════
_TOP = (1 << 32) - 1
_HALF = 1 << 31
_QTR = 1 << 30

def quantize(p):
    """Probabilidades → tabla acumulada entera, cada símbolo con freq ≥ 1."""
    f = np.floor(p * (FREQ_TOTAL - 256)).astype(np.int64) + 1
    cum = np.zeros(257, dtype=np.int64)
    np.cumsum(f, out=cum[1:])
    return cum

class Encoder:
    def __init__(self):
        self.low, self.high, self.pending = 0, _TOP, 0
        self.bits = []

    def _emit(self, b):
        self.bits.append(b)
        if self.pending:
            self.bits.extend([1 - b] * self.pending)
            self.pending = 0

    def encode(self, lo, hi, total):
        rng = self.high - self.low + 1
        self.high = self.low + rng * hi // total - 1
        self.low = self.low + rng * lo // total
        while True:
            if self.high < _HALF:
                self._emit(0)
            elif self.low >= _HALF:
                self._emit(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _QTR and self.high < _HALF + _QTR:
                self.pending += 1
                self.low -= _QTR
                self.high -= _QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        self.pending += 1
        self._emit(0 if self.low < _QTR else 1)
        bits = self.bits
        bits += [0] * ((-len(bits)) % 8)
        out = bytearray()
        for i in range(0, len(bits), 8):
            v = 0
            for b in bits[i:i + 8]:
                v = (v << 1) | b
            out.append(v)
        return bytes(out)

class Decoder:
    def __init__(self, data):
        self.data, self.pos = data, 0
        self.low, self.high = 0, _TOP
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self._bit()

    def _bit(self):
        i = self.pos >> 3
        b = 0
        if i < len(self.data):
            b = (self.data[i] >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return b

    def decode(self, cum):
        total = int(cum[-1])
        rng = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // rng
        s = int(np.searchsorted(cum, target, side="right")) - 1
        lo, hi = int(cum[s]), int(cum[s + 1])
        self.high = self.low + rng * hi // total - 1
        self.low = self.low + rng * lo // total
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.low -= _HALF
                self.high -= _HALF
                self.value -= _HALF
            elif self.low >= _QTR and self.high < _HALF + _QTR:
                self.low -= _QTR
                self.high -= _QTR
                self.value -= _QTR
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._bit()
        return s

def compress_bytes(data, pred, learn=True):
    enc = Encoder()
    for s in data:
        cum = quantize(pred.probs())
        enc.encode(int(cum[s]), int(cum[s + 1]), int(cum[-1]))
        pred.update(s, learn)
    return enc.finish()

def decompress_bytes(payload, n, pred, learn=True):
    dec = Decoder(payload)
    out = bytearray()
    for _ in range(n):
        s = dec.decode(quantize(pred.probs()))
        out.append(s)
        pred.update(s, learn)
    return bytes(out)

def score_bits(data, pred, learn=True):
    """Bits ideales (-log2 p) por símbolo, sin codificar."""
    bits = np.zeros(len(data))
    for i, s in enumerate(data):
        bits[i] = -np.log2(pred.probs()[s])
        pred.update(s, learn)
    return bits

# ═══════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ═══════════════════════════════════════════════════════════════════════════
def read_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()

_META_NOTE = re.compile(r"(?:\^\^|__|\^|_|=)?([A-Ga-gz])([,']*)(\d*)(/*)(\d*)")

def compute_meta(data):
    """Estadísticas del corpus para los priores v2: histograma de longitudes (en unidades de L:)
    y de intervalos diatónicos entre notas consecutivas. No tocan las probabilidades del modelo."""
    text = data.decode("ascii", errors="replace")
    lens, ints, nn, nt = Counter(), Counter(), 0, 0
    for tune in re.split(r"(?m)^[XK]:.*$", text):
        prev, seen = None, False
        for line in tune.split("\n"):
            if re.match(r"^[A-Za-z]:", line) or line.startswith("%"):
                continue
            line = re.sub(r'"[^"]*"|!\w+!|\{[^}]*\}|\[[A-Za-z]:[^\]]*\]', "", line)
            for m in _META_NOTE.finditer(line):
                letter, marks, num, sl, den = m.groups()
                if len(num) > 3 or len(den) > 3:
                    continue
                mult = Fraction(int(num) if num else 1)
                if sl:
                    mult = mult / (int(den) if den and int(den) else 2 ** len(sl))
                lens[str(mult)] += 1
                nn += 1
                seen = True
                if letter != "z":
                    d = deg_of(letter + marks)
                    if prev is not None:
                        ints[max(-14, min(14, d - prev))] += 1
                    prev = d
        nt += seen
    return {"v": 2, "len_hist": dict(lens), "int_hist": {str(k): v for k, v in ints.items()},
            "ntunes": nt, "nnotes": nn}

def cmd_train(a):
    data = read_bytes(a.corpus)
    if len(data) < 100:
        die("corpus demasiado pequeño")
    if a.backend == "ngram":
        need(0 <= a.order <= 32, "--order debe estar entre 0 y 32")
        need(a.gamma > 0, "--gamma debe ser > 0")
        need(a.minlen >= 1, "--minlen debe ser >= 1")
        pred = NgramPredictor(order=a.order, minlen=a.minlen, gamma=a.gamma)
        for s in data:
            pred.update(s, True)
        pred.meta = compute_meta(data)
        with open(a.output, "wb") as fh:
            pickle.dump(pred, fh, protocol=4)
        print(green(f"✓ ngram orden {a.order}: {len(data):,} car → {a.output} "
                    f"({os.path.getsize(a.output)/1024:.0f} KiB)"))
        return
    # ── torch ──
    need(min(a.emb, a.hidden, a.layers, a.seq, a.batch, a.steps, a.eval_every) >= 1,
         "--emb/--hidden/--layers/--seq/--batch/--steps/--eval-every deben ser >= 1")
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        die("backend torch requiere PyTorch (pip install torch)")
    dev = "cuda" if (a.device == "auto" and torch.cuda.is_available()) else (
        "cpu" if a.device == "auto" else a.device)
    arr = np.frombuffer(data, dtype=np.uint8)
    n_val = max(len(arr) // 20, a.seq + 2)
    if len(arr) - n_val < a.seq + 2:
        die("corpus demasiado pequeño para --seq")
    tr, va = arr[:-n_val], arr[-n_val:]
    cfg = dict(emb=a.emb, hidden=a.hidden, layers=a.layers)
    model = _torch_build(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def windows(src, idx):
        x = np.stack([np.concatenate([[0], src[i:i + a.seq - 1]]) for i in idx])
        y = np.stack([src[i:i + a.seq] for i in idx])
        return (torch.from_numpy(x.astype(np.int64)).to(dev),
                torch.from_numpy(y.astype(np.int64)).to(dev))

    def evaluate():
        model.eval()
        idx = list(range(0, len(va) - a.seq - 1, a.seq))[:64]
        if not idx:
            return float("nan")
        with torch.no_grad():
            x, y = windows(va, idx)
            lg, _ = model(x)
            loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        model.train()
        return loss.item() / np.log(2)

    best = float("inf")
    for step in range(1, a.steps + 1):
        idx = np.random.randint(0, len(tr) - a.seq - 1, size=a.batch)
        x, y = windows(tr, idx)
        lg, _ = model(x)
        loss = F.cross_entropy(lg.reshape(-1, 256), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate()
            print(f"  paso {step:6d}  train {loss.item()/np.log(2):.3f} bpc   val {v:.3f} bpc")
            if v < best:
                best = v
                torch.save({"cfg": cfg, "state": {k: t.cpu() for k, t in model.state_dict().items()}},
                           a.output)
    print(green(f"✓ mejor val {best:.3f} bpc → {a.output}"))

def _baselines(data):
    return {"gzip -9": len(zlib.compress(data, 9)),
            "bz2": len(bz2.compress(data, 9)),
            "xz": len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME))}

def cmd_bpc(a):
    data = read_bytes(a.data)
    need(len(data) > 0, f"{a.data}: el fichero está vacío")
    pred = load_predictor(a.model, a.backend)
    modes = [("adaptativo", True)] if a.backend_name(pred) == "torch" else \
        [("estático", False), ("adaptativo", True)]
    print(bold(f"{a.data}: {len(data):,} caracteres  ·  modelo {pred.name} "
               f"({pred.size_bytes()/1024:.0f} KiB)"))
    for label, learn in modes:
        p2 = load_predictor(a.model, a.backend)
        bits = score_bits(data, p2, learn)
        print(f"  {label:11s} {bits.sum()/len(data):6.3f} bpc  "
              f"({bits.sum()/8:,.0f} bytes ideales)")
    for k, v in _baselines(data).items():
        print(dim(f"  {k:11s} {v*8/len(data):6.3f} bpc  ({v:,} bytes)"))
    print(dim("  (el Premio Hutter además suma el tamaño del decodificador/modelo)"))

def cmd_sweep(a):
    """Barrido orden × gamma × minlen del backend ngram sobre train/test."""
    tr, te = read_bytes(a.train), read_bytes(a.test)
    need(len(tr) > 0 and len(te) > 0, "los ficheros de entrenamiento y test no pueden estar vacíos")
    need(all(0 <= o <= 32 for o in a.orders) and all(g > 0 for g in a.gammas) and all(m >= 1 for m in a.minlens),
         "valores inválidos: 0<=order<=32, gamma>0, minlen>=1")
    res = []
    for order in a.orders:
        for g in a.gammas:
            for ml in a.minlens:
                p = NgramPredictor(order=order, minlen=ml, gamma=g)
                for s in tr:
                    p.update(s, True)
                b = score_bits(te, p, learn=True).sum() / len(te)
                res.append((b, order, g, ml))
                print(f"  orden {order}  gamma {g:<4}  minlen {ml:<2}  {b:6.3f} bpc", flush=True)
    b, o, g, ml = min(res)
    print(green(f"✓ mejor: --order {o} --gamma {g} --minlen {ml}  ({b:.3f} bpc)"))

def cmd_compress(a):
    data = read_bytes(a.input)
    pred = load_predictor(a.model, a.backend)
    payload = compress_bytes(data, pred, learn=not a.static)
    flag = 1 if a.static else 0
    crc = zlib.crc32(data) & 0xFFFFFFFF
    hdr = (MAGIC + bytes([BACKENDS[pred.name], flag]) + fingerprint(a.model)
           + struct.pack(">II", len(data), crc))
    out = a.output or a.input + ".abcp"
    with open(out, "wb") as fh:
        fh.write(hdr + payload)
    tot = len(hdr) + len(payload)
    print(green(f"✓ {len(data):,} → {tot:,} bytes  ({tot*8/max(1,len(data)):.3f} bpc, "
                f"cabecera {len(hdr)} B)  → {out}"))

def cmd_decompress(a):
    blob = read_bytes(a.input)
    if len(blob) < HDR_LEN or blob[:4] != MAGIC:
        die("no es un fichero .abcp válido (formato ABP2)")
    flag, fp = blob[5], blob[6:14]
    n, crc = struct.unpack(">II", blob[14:22])
    payload = blob[HDR_LEN:]
    if n > (8 * len(payload) + 8) / MIN_BITS_PER_SYM:
        die("fichero corrupto: la longitud declarada es imposible para su tamaño")
    if fp != fingerprint(a.model):
        die("el modelo no coincide con el usado al comprimir")
    dest = a.output or (a.input[:-5] if a.input.endswith(".abcp") else a.input + ".out")
    if not a.output and os.path.exists(dest):
        die(f"{dest} ya existe; indica otro destino con -o para no sobrescribirlo")
    pred = load_predictor(a.model, a.backend)
    out = decompress_bytes(payload, n, pred, learn=(flag == 0))
    if zlib.crc32(out) & 0xFFFFFFFF != crc:
        die("fichero corrupto (suma de comprobación incorrecta); no se escribe la salida")
    with open(dest, "wb") as fh:
        fh.write(out)
    print(green(f"✓ {n:,} bytes → {dest}"))

def _heat(ch, bits):
    if not _TTY:
        return ch
    code = "2" if bits < 0.5 else "0" if bits < 2 else "33" if bits < 4 else "31;1"
    return f"\033[{code}m{ch}\033[0m"

def cmd_surprise(a):
    data = read_bytes(a.data)
    need(len(data) > 0, f"{a.data}: el fichero está vacío")
    pred = load_predictor(a.model, a.backend)
    bits = score_bits(data, pred, learn=not a.static)
    text = data.decode("ascii", errors="replace")
    print(bold(f"sorpresa media: {bits.mean():.3f} bits/car  ·  total {bits.sum()/8:,.0f} bytes"))
    # agrupar por compás ('|' como separador)
    bars, start = [], 0
    for i, ch in enumerate(text):
        if ch in "|\n":
            if i > start:
                bars.append((start, i))
            start = i + 1
    if start < len(text):
        bars.append((start, len(text)))
    stats = [(bits[s:e].sum(), s, e) for s, e in bars if e > s]
    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("bar,start,end,bits,bits_per_char,text\n")
            for n, (b, s, e) in enumerate(stats):
                fh.write(f'{n},{s},{e},{b:.2f},{b/(e-s):.3f},"{text[s:e]}"\n')
    print(bold(f"\nlos {a.top} compases más sorprendentes:"))
    for b, s, e in sorted(stats, reverse=True)[:a.top]:
        print(f"  {b:7.1f} bits  {b/(e-s):5.2f} b/c  {''.join(_heat(text[i], bits[i]) for i in range(s, e))}")
    print(bold(f"\nlos {a.top} más predecibles:"))
    for b, s, e in sorted(stats)[:a.top]:
        print(f"  {b:7.1f} bits  {b/(e-s):5.2f} b/c  {text[s:e]}")
    if a.heatmap:
        print(bold("\nmapa de calor:"))
        print("".join(_heat(text[i], bits[i]) for i in range(min(len(text), a.heatmap))))

# ═══════════════════════════════════════════════════════════════════════════
#  MUESTREO RESTRINGIDO (--constrain)
#  La probabilidad sigue siendo la del modelo; solo se descartan, en cada paso,
#  los caracteres que romperían la gramática ABC (duración del compás, ligaduras,
#  tresillos, ritmo roto, barras). Se renormaliza sobre los caracteres válidos.
#  Subconjunto soportado: notas con alteraciones/octavas/longitudes, silencios z,
#  ( ) ligaduras, (3 tresillos, > < ritmo roto, - ligadura de unión, . ~ adornos,
#  barras | || |] |: :| ::  y anacrusa. NO se generan: acordes "..", [ ], {..},
#  !..!, voltas 1/2, cabeceras en el cuerpo.
# ═══════════════════════════════════════════════════════════════════════════
_TOKENS = ("|", "||", "|]", "|:", ":|", "::", ":||")
_TOK_PREFIXES = {t[:i] for t in _TOKENS for i in range(1, len(t) + 1)}
_SECTION_TOKENS = {"||", "|]", "|:", ":|", "::", ":||"}   # tras ellos empieza sección
_ZERO_OK = {"||", "|]", ":|", "::", ":||"}                # permiten "|:" a principio de línea
_TUP_Q = {"2": 3, "3": 2, "4": 3, "5": 2, "6": 2, "7": 2, "8": 3, "9": 2}
_CAND = "ABCDEFGabcdefgz^_=,'0123456789/<>-()|:] \n.~\"#mijsu"

# Símbolos de acorde "Am7/E": gramática cerrada (raíz × calidad × bajo opcional)
_ROOTS = [l + a for l in "ABCDEFG" for a in ("", "#", "b")]
_QUALS = ["", "m", "7", "m7", "maj7", "dim", "aug", "sus4", "sus2", "6", "m6", "9", "dim7", "m7b5"]
CHORDS = {r + q + (("/" + b) if b else "") for r in _ROOTS for q in _QUALS for b in [""] + _ROOTS}
_CHORD_PREFIXES = {c[:i] for c in CHORDS for i in range(1, len(c) + 1)}

class AbcState:
    """Autómata del cuerpo ABC. step(ch) devuelve True si ch es válido y avanza
    el estado; si devuelve False el estado queda indefinido (probar sobre clone())."""

    def __init__(self, bar_len, unit, chords=False, palette=None, compound=False):
        self.bar_len, self.unit = bar_len, unit
        self.compound = compound                         # numerador múltiplo de 3 (abc2midi): (5 (7 (9 ocupan el tiempo de 3
        self.chords, self.chord = chords, None          # símbolos de acorde "..." (opcional)
        self.pal = None                                  # paleta rítmica: duraciones absolutas permitidas
        if palette:
            self.pal = frozenset(palette)
            g = unit / 4
            K = int(bar_len / g)
            tile = [False] * (K + 1)
            tile[0] = True
            G = sorted({int(d / g) for d in self.pal})
            for k in range(1, K + 1):
                tile[k] = any(k >= x and tile[k - x] for x in G)
            self.tile, self.g, self.pmin = tile, g, min(self.pal)
        self.deco_pending = self.note_deco = False
        self.tilde_pending = self.note_tilde = False
        self.need_pick, self.closing_comp = False, False   # tras un compás de cierre parcial debe venir la anacrusa
        self.origin, self.open_rep = "start", False        # destino de un ":|" y repetición abierta ("|:" sin cerrar)
        self.suf = ""                                    # sufijo de longitud escrito (dígitos, /)
        self.last_deg, self.last_int = None, None        # altura (grados diatónicos) y salto previos
        self.recent = ()                                 # últimos grados (para centro de registro)
        self.pos = Fraction(0)                 # posición dentro del compás (fracción de redonda)
        self.first_bar, self.pickup, self.sec_first = True, None, True
        self.last_token = None
        self.bt, self.bt_zero, self.bt_partial = "", False, False
        self.in_note, self.phase, self.pitch, self.is_rest = False, None, "", False
        self.n_start, self.n_base = Fraction(0), Fraction(0)
        self.num, self.den, self.slashes, self.has_slash, self.broken = "", "", 0, False, None
        self.next_factor = Fraction(1)
        self.tup_rem, self.tup_fac, self.in_tup = 0, Fraction(1), False
        self.npitch = 0                        # notas con altura ya cerradas
        self.paren, self.need_note, self.slur = False, False, 0
        self.tie = None
        self.prev = "\n"

    def clone(self):
        t = object.__new__(AbcState)
        t.__dict__ = self.__dict__.copy()
        return t

    @property
    def cap(self):
        """Longitud efectiva del compás en curso: la anacrusa si toca anacrusa, si no el compás."""
        return self.pickup if self.need_pick else self.bar_len

    # ── duración de la nota en curso ──
    def _dur(self):
        m = Fraction(int(self.num)) if self.num else Fraction(1)
        if self.has_slash:
            m = m / (int(self.den) if self.den else 2 ** self.slashes)
        v = self.n_base * m
        if self.broken == ">":
            v *= Fraction(3, 2)
        elif self.broken == "<":
            v *= Fraction(1, 2)
        return v

    def _min_next(self):
        """Duración mínima de lo que debe caber aún (siguiente nota + resto del tresillo)."""
        floor = self.unit / 2 if self.tilde_pending else 0        # "~" pendiente: la nota será >= media unidad
        if self.pal is not None:
            return max(self.pmin, floor)
        if self.tup_rem > 0:
            return self.tup_rem * self.unit * self.tup_fac * self.next_factor
        return max(self.unit * self.next_factor / 4, floor)

    def _reserve(self):
        r = Fraction(0)
        if self.tup_rem > 0:
            r += self.tup_rem * self.unit * self.tup_fac
        if self.broken == ">":
            r += self.unit * Fraction(1, 2)
        elif self.broken == "<":
            r += self.unit * Fraction(3, 2)
        return r

    def _min_dur(self):
        """Duración mínima a la que aún puede llegar la nota en curso con más sufijos."""
        if self.broken is not None:
            return self._dur()
        m = int(self.num) if self.num else 1
        if self.in_tup:
            return self.n_base * m
        if self.has_slash and self.den:
            f = Fraction(m, int(self.den))
        elif self.has_slash and self.slashes == 2:
            f = Fraction(1, 4)
        else:
            f = Fraction(m, 4) if (m % 2 == 1 and m < 4) else Fraction(m, 2)
        if self.note_tilde and f < Fraction(1, 2):
            f = Fraction(1, 2)                 # "~" no admite notas menores que media unidad
        return self.n_base * f

    def _pal_ok(self, d):
        """¿Una nota que acaba durando d es válida con la paleta? (está en ella y el resto del compás se puede teselar)"""
        if d not in self.pal or self.n_start + d > self.cap or (self.note_tilde and d * 2 < self.n_base):
            return False
        return self.first_bar or self.tile[int((self.cap - self.n_start - d) / self.g)]

    def _fits_pal_from(self, pos, d):
        return pos + d <= self.cap and (self.first_bar or self.tile[int((self.cap - pos - d) / self.g)])

    def _could_fit(self):
        if self.note_tilde and self._dur() * 2 < self.n_base:      # abc2midi: "~" no admite notas menores que media unidad
            return False
        if self.pal is not None:
            if self.in_tup or self.broken is not None:
                return self._pal_ok(self.n_base)
            return any(self._pal_ok(self.n_base * m) for m in _suf_mults(self.suf))
        return self.n_start + self._min_dur() + self._reserve() <= self.cap

    def _fits(self):
        if self.pal is not None and not self._pal_ok(self._dur()):
            return False
        return self.n_start + self._dur() + self._reserve() <= self.cap

    def _commit(self):
        if not self.in_note:
            return True
        if self.phase != "post":
            return False
        if self.tie is not None and self.pitch != self.tie:
            return False
        if not self._fits():
            return False
        newpos = self.n_start + self._dur()
        if self.tup_rem == 0 and (self.cap - newpos) % (self.unit / 4) != 0:
            return False
        self.pos = newpos
        if self.broken == ">":
            self.next_factor = Fraction(1, 2)
        elif self.broken == "<":
            self.next_factor = Fraction(3, 2)
        self.tie = None
        if not self.is_rest:
            self.npitch += 1
            d = deg_of(self.pitch)
            self.last_int = None if self.last_deg is None else d - self.last_deg
            self.last_deg = d
            self.recent = (self.recent + (d,))[-16:]
        self.in_note, self.phase, self.pitch, self.is_rest = False, None, "", False
        self.num, self.den, self.slashes, self.has_slash, self.broken = "", "", 0, False, None
        self.suf = ""
        return True

    def _set_pitch(self, p):
        self.pitch = p
        return self.tie is None or self.tie.startswith(p)

    def _begin_note(self):
        if not self._commit():
            return False
        if self.paren:
            if self.paren == "tup":           # sin ligaduras anidadas
                return False
            self.slur += 1
            self.paren = False
        self.need_note = False
        self.note_deco, self.deco_pending = self.deco_pending, False
        self.note_tilde, self.tilde_pending = self.tilde_pending, False
        return True

    def _init_dur(self):
        fac = Fraction(1)
        self.in_tup = self.tup_rem > 0 or self.next_factor != 1   # sin subdivisiones ni "<>"
        if self.tup_rem > 0:
            fac = self.tup_fac
            self.tup_rem -= 1
        fac *= self.next_factor
        self.next_factor = Fraction(1)
        self.n_base, self.n_start = self.unit * fac, self.pos
        self.num, self.den, self.slashes, self.has_slash, self.broken = "", "", 0, False, None
        self.suf = ""
        return self._could_fit()

    # ── API ──
    def step(self, ch):
        ok = self._step(ch)
        if ok:
            self.prev = ch
        return ok

    def _finish_token(self):
        t = self.bt
        self.last_token = t
        if t in _SECTION_TOKENS:
            self.sec_first = True
        comp, self.closing_comp = self.closing_comp, False
        if comp:
            self.need_pick = True                 # compás de cierre parcial → el siguiente es la anacrusa
        if t in (":|", ":||", "::"):
            self.open_rep = False
            self.origin = "pick" if comp else "full"
        if t in ("|:", "::"):
            self.open_rep = True
            if t == "|:":
                self.origin = "start" if self.first_bar else ("pick" if self.need_pick else "full")
        self.bt, self.bt_zero, self.bt_partial = "", False, False

    def _chord_step(self, ch):
        if ch == '"':
            if self.chord not in CHORDS:
                return False
            self.chord, self.need_note = None, True
            return True
        nb = self.chord + ch
        if nb in _CHORD_PREFIXES:
            self.chord = nb
            return True
        return False

    def _chord_start(self):
        if (not self.chords or self.paren or self.need_note or self.tie is not None
                or self.next_factor != 1 or self.tup_rem > 0):
            return False
        if not self._commit() or self.pos + self._min_next() > self.cap:
            return False
        self.chord = ""
        return True

    def _step(self, ch):
        if self.chord is not None:
            return self._chord_step(ch)
        if self.bt:
            if ch in "|:]" and (self.bt + ch) in _TOK_PREFIXES:
                nbt = self.bt + ch
                if self.bt_zero and nbt != "|:":
                    return False
                if nbt == "|:" and (self.open_rep or self.need_pick):
                    return False           # "|:" con otra repetición abierta, o antes de una anacrusa
                if self.bt_partial and nbt not in ("||", "|]"):
                    return False
                self.bt = nbt
                return True
            if self.bt not in _TOKENS:
                return False
            if self.bt_zero and self.bt != "|:":      # "|" suelta al principio de línea: no
                return False
            if self.bt_partial and self.bt not in ("||", "|]"):
                return False
            self._finish_token()
        return self._idle(ch)

    def _idle(self, ch):
        if ch == " ":
            if self.prev in (" ", "\n") or self.paren or self.need_note:
                return False
            if self.tup_rem > 0 and not self.in_note and self.prev.isdigit():
                return False
            return self._commit()
        if ch == "\n":
            if self.paren or self.need_note or self.tup_rem > 0 or self.prev not in "|:]":
                return False
            return self._commit() and self.next_factor == 1
        if ch in "|:":
            return self._start_bar(ch)
        if ch == '"':
            return self._chord_start()
        if ch == "(":
            if self.paren or self.need_note or self.tie is not None:
                return False
            if not self._commit() or self.pos + self._min_next() > self.cap:
                return False
            if self.slur > 0:                 # con ligadura abierta, "(" solo puede ser un tresillo
                if self.pal is not None or self.tup_rem > 0 or self.next_factor != 1 or self.pos + 2 * self.unit > self.cap:
                    return False
                self.paren = "tup"
            else:
                self.paren = True
            return True
        if ch.isdigit() and self.paren:
            if ch not in _TUP_Q or self.tup_rem > 0 or self.pal is not None:
                return False
            if self.next_factor != 1:
                return False
            tf = Fraction(3 if (self.compound and ch in "579") else _TUP_Q[ch], int(ch))
            if self.pos + int(ch) * self.unit * tf * self.next_factor > self.cap:
                return False
            self.tup_rem, self.tup_fac = int(ch), tf
            self.paren = False
            return True
        if ch == ")":
            if not (self.in_note and self.phase == "post" and self.slur > 0):
                return False
            if not self._commit():
                return False
            self.slur -= 1
            return True
        if ch in ".~":
            if self.paren or self.need_note or self.tie is not None or self.broken is not None or self.next_factor != 1:
                return False               # abc2midi no admite decoración junto a ritmo roto
            if not self._commit():
                return False
            self.tilde_pending = ch == "~"
            if self.pos + self._min_next() > self.cap:
                return False
            if self.tilde_pending and self.pal is not None and not any(
                    d * 2 >= self.unit and self._fits_pal_from(self.pos, d) for d in self.pal):
                return False                                 # paleta sin ninguna duración válida para una nota con "~"
            self.need_note = True
            self.deco_pending = True
            return True
        if ch in "^_=":
            return self._note_acc(ch)
        if ch in "ABCDEFGabcdefg":
            return self._note_letter(ch)
        if ch == "z":
            if not self._begin_note():
                return False
            self.in_note, self.phase, self.is_rest = True, "post", True
            return self._set_pitch("z") and self._init_dur()
        if self.in_note and self.phase == "post":
            return self._suffix(ch)
        return False

    def _note_acc(self, ch):
        if self.in_note and self.phase == "pre":
            np_ = self.pitch + ch
            return np_ in ("^", "^^", "_", "__", "=") and self._set_pitch(np_)
        if not self._commit() or self.pos + self._min_next() > self.cap:
            return False
        if not self._begin_note():
            return False
        self.in_note, self.phase, self.is_rest = True, "pre", False
        return self._set_pitch(ch)

    def _note_letter(self, ch):
        if self.in_note and self.phase == "pre":
            if not self._set_pitch(self.pitch + ch):
                return False
            self.phase = "post"
            return self._init_dur()
        if not self._begin_note():
            return False
        self.in_note, self.phase, self.is_rest = True, "post", False
        return self._set_pitch(ch) and self._init_dur()

    def _suffix(self, ch):
        if self.broken is not None:
            return False
        if ch in ",'":
            if self.is_rest or self.num or self.has_slash:
                return False
            letter = self.pitch.rstrip(",'")[-1]
            if ch == "," and not (letter.isupper() and "'" not in self.pitch):
                return False
            if ch == "'" and not (letter.islower() and "," not in self.pitch):
                return False
            if self.pitch.count(ch) >= 3:
                return False
            return self._set_pitch(self.pitch + ch)
        if self.tie is not None and self.pitch != self.tie:
            return False
        if ch.isdigit():
            if self.in_tup:
                return False
            if self.has_slash:
                nd = self.den + ch
                if self.slashes != 1 or nd not in ("2", "4"):
                    return False
                if self.num and (int(self.num) % 2 == 0 or int(self.num) >= int(nd)):
                    return False
                self.den = nd
                self.suf += ch
                return self._could_fit()
            else:
                nn = self.num + ch
                if nn.startswith("0") or int(nn) > 8:
                    return False
                self.num = nn
            self.suf += ch
            return self._could_fit()
        if ch == "/":
            if self.in_tup:
                return False
            if self.den != "" or self.slashes >= 2 or (self.slashes == 1 and self.num):
                return False
            if self.slashes == 1 and self.prev != "/":
                return False
            self.has_slash = True
            self.slashes += 1
            self.suf += "/"
            return self._could_fit()
        if ch in "<>":
            if self.in_tup or self.has_slash or self.num != "" or self.pal is not None or self.note_deco:
                return False               # abc2midi exige dos notas de igual longitud escrita; sin subdivisiones
            self.broken = ch
            return self._could_fit()
        if ch == "-":
            if self.is_rest or self.note_deco:   # abc2midi no entiende ".A-"
                return False
            pitch = self.pitch
            if not self._commit():
                return False
            self.tie = pitch
            return True
        return False

    def _start_bar(self, ch):
        if self.paren or self.need_note or self.tup_rem > 0 or self.next_factor != 1:
            return False
        if not self._commit() or self.next_factor != 1:   # el ritmo roto necesita su nota siguiente
            return False
        pos, bl = self.pos, self.bar_len
        if pos == 0:                       # barra al principio de línea (p. ej. "|:")
            if ch != "|" or self.prev != "\n":
                return False
            if not (self.last_token is None or self.last_token in _ZERO_OK):
                return False
            if self.open_rep or self.need_pick:    # una barra a principio de línea solo puede ser "|:"
                return False
            self.bt, self.bt_zero, self.bt_partial = "|", True, False
            return True
        full = pos == bl
        comp = self.pickup is not None and pos == bl - self.pickup and not full
        pick = self.need_pick and pos == self.pickup and not full
        first_partial = self.first_bar and pos < bl
        self.closing_comp = False
        if self.need_pick:                 # tras un cierre parcial: solo la anacrusa, con barra simple
            if not pick or ch == ":":
                return False
            partial = False
            self.need_pick = False
        elif ch == ":":                    # fin de repetición
            origin_pick = self.origin == "pick" or (self.origin == "start" and self.pickup is not None)
            if self.tie is not None or origin_pick or not full or not (self.open_rep or self.origin == "start"):
                return False               # ":|" sin "|:" solo vale una vez, desde el inicio               # volver a una anacrusa haría sonar un compás corto (abc2midi avisa)
            partial = False
        elif full or first_partial:
            if first_partial and self.open_rep:
                return False               # "|:" al inicio: el primer compás debe ser completo (si no, no se podría cerrar)
            partial = False
        elif comp:                         # compás de cierre de sección: exige || o |]
            partial = True
            self.closing_comp = True
        else:
            return False
        if self.first_bar:
            self.first_bar = False
            if pos < bl:
                self.pickup = pos
        self.sec_first = False
        self.pos = Fraction(0)
        self.bt, self.bt_zero, self.bt_partial = ch, False, partial
        return True

    def at_boundary(self):
        """¿Se puede terminar la pieza aquí?"""
        if (self.paren or self.need_note or self.tup_rem or self.tie is not None
                or self.next_factor != 1 or self.chord is not None):
            return False
        if self.bt and (self.bt not in _TOKENS or (self.bt_partial and self.bt not in ("||", "|]"))
                        or (self.bt_zero and self.bt != "|:")):
            return False
        op = self.open_rep                    # no terminar con una repetición abierta
        if self.bt in (":|", ":||"):
            op = False
        elif self.bt in ("|:", "::"):
            op = True
        if op:
            return False
        if self.in_note:
            if self.phase != "post" or self.broken is not None:
                return False
            if not self._fits():
                return False
            end = self.n_start + self._dur()
        else:
            end = self.pos
        if self.need_pick and not self.bt:     # a mitad de la anacrusa que sigue a un cierre parcial
            return end == 0
        ok = {self.bar_len, Fraction(0)}
        if self.pickup is not None:
            ok.add(self.bar_len - self.pickup)
        return end in ok


def deg_of(pitch):
    """Grado diatónico absoluto de una altura ABC ('^c,' → ...): 7*octava + índice de letra."""
    letters = pitch.lstrip("^_=")
    base = letters[0]
    return 7 * ((1 if base.islower() else 0) + letters.count("'") - letters.count(",")) + "CDEFGAB".index(base.upper())

_SUFTAB = {}
def _suf_mults(suf):
    """Multiplicadores de duración alcanzables extendiendo el sufijo `suf` (sin ritmo roto ni tresillos)."""
    if not _SUFTAB:
        V = {}
        d0 = AbcState(Fraction(10 ** 6), Fraction(1, 8))
        d0.step("C")
        def dfs(st, s):
            V[s] = st._dur() / st.n_base
            for ch in "0123456789/":
                c = st.clone()
                if c.step(ch):
                    dfs(c, s + ch)
        dfs(d0, "")
        for s_ in V:
            _SUFTAB[s_] = frozenset(m for t, m in V.items() if t.startswith(s_))
        _SUFTAB[None] = frozenset(V.values())
    return _SUFTAB.get(suf, frozenset())

def make_palette(spec, unit, bar_len, meta=None):
    """spec: "1/2,1,3/2,2,4" (en unidades de L:) o "auto:N" (las N más frecuentes del corpus)."""
    allm = _suf_mults(None)
    if spec.startswith("auto"):
        n = int(spec.split(":")[1]) if ":" in spec else 4
        hist = (meta or {}).get("len_hist")
        need(hist, "--rhythms auto necesita un modelo v2 entrenado con metadatos (reentrena) o una lista explícita")
        mult = [(c, Fraction(k)) for k, c in hist.items()]
        mult = [(c, m) for c, m in sorted(mult, key=lambda x: -x[0]) if m in allm][:n]
        fr = [m for _, m in mult]
    else:
        try:
            fr = [_parse_frac(x) for x in spec.split(",") if x.strip()]
        except (ValueError, ZeroDivisionError):
            die(f"--rhythms no interpretable: {spec!r} (ejemplo: 1/2,1,2,4)")
    g = unit / 4
    ok = []
    for m in fr:
        need(m > 0, "--rhythms: duraciones positivas")
        need(m in allm, f"--rhythms: {m} no es una longitud ABC expresable (usa n, n/2, 1/4, 3/4 con n≤8)")
        need((unit * m) % g == 0, f"--rhythms: {m} fuera de la rejilla")
        ok.append(unit * m)
    pal = set(ok)
    def tileable(P):
        K = int(bar_len / g)
        G = sorted({int(d / g) for d in P})
        t = [True] + [False] * K
        for k in range(1, K + 1):
            t[k] = any(k >= x and t[k - x] for x in G)
        return t[K]
    if spec.startswith("auto"):
        for extra in (Fraction(1), Fraction(1, 2), Fraction(1, 4), Fraction(2), Fraction(3, 4)):
            if pal and tileable(pal):
                break
            if extra in allm and (unit * extra) % g == 0:
                pal.add(unit * extra)
    need(pal, "--rhythms: paleta vacía")
    need(tileable(pal), f"--rhythms: ninguna combinación de {sorted(float(d / unit) for d in pal)} (en L:) llena un compás de {bar_len}")
    return pal

def _parse_frac(s):
    s = s.strip()
    if s in ("C", "C|"):
        return Fraction(1)
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    return Fraction(int(s))

def _constrained_header(a):
    prime = _prime_text(a.prime)
    lines = prime.split("\n")
    header, body, ptitle = None, prime, None
    for i, ln in enumerate(lines):
        if ln.startswith("K:"):
            hl = [l for l in lines[:i + 1] if not (l.startswith("X:") or l.startswith("T:"))]
            for l in lines[:i + 1]:
                if l.startswith("T:") and ptitle is None:
                    ptitle = l[2:].strip()
            header, body = "\n".join(hl) + "\n", "\n".join(lines[i + 1:])
            break
    if header is None and re.match(r"^[A-Za-z]:", prime):
        die("--prime con cabecera (X:, M:, L:...) necesita una línea K:")
    meter_s, unit_s = a.meter, a.unit
    style = (getattr(a, "style", None) or "").strip()
    if style and not meter_s and not (header and any(l.startswith("M:") for l in header.split("\n"))):
        meter_s = STYLE_METER.get(style.lower())
    if header is not None:
        for ln in header.split("\n"):
            if ln.startswith("M:") and not meter_s:
                meter_s = ln[2:].split("%")[0].strip()
            if ln.startswith("L:") and not unit_s:
                unit_s = ln[2:].split("%")[0].strip()
    try:
        bar_len = _parse_frac(meter_s or "4/4")
        unit = _parse_frac(unit_s) if unit_s else (Fraction(1, 16) if bar_len < Fraction(3, 4) else Fraction(1, 8))
    except (ValueError, ZeroDivisionError):
        die("M:/L: no interpretables (usa --meter 4/4 --unit 1/8)")
    need(bar_len > 0 and unit > 0, "M:/L: deben ser positivos")
    need(bar_len % (unit / 4) == 0,
         f"M:{bar_len} no es múltiplo de L:/4 ({unit / 4}): ninguna combinación de notas completaría un compás")
    if header is None:
        ustr = unit_s or ("1/16" if bar_len < Fraction(3, 4) else "1/8")
        header = f"M:{meter_s or '4/4'}\nL:{ustr}\nK:{a.key}\n"
    if style and not any(l.startswith("R:") for l in header.split("\n")):
        ls = header.rstrip("\n").split("\n")
        k = max(i for i, l in enumerate(ls) if l.startswith("K:"))
        header = "\n".join(ls[:k] + [f"R:{style}"] + ls[k:]) + "\n"
    return header, body, bar_len, unit, ptitle

def _rest_fill(st):
    """Silencio que completa el compás en curso (solo caminos de emergencia)."""
    t = st.clone()
    if t.bt or not t._commit() or t.pos == 0 or t.pos >= t.bar_len:
        return ""
    r = (t.bar_len - t.pos) / t.unit
    n, d = r.numerator, r.denominator
    if d == 1:
        return "z" if n == 1 else f"z{n}"
    if d in (2, 4, 8, 16) and n <= 16:
        return f"z{n}/{d}" if n > 1 else f"z/{d}"
    return ""

def _finish_text(text, st):
    text = text.rstrip()
    if st.paren or st.need_note:
        text = text[:-1]
    if st.in_note and st.phase == "pre":
        text = text.rstrip("^_=")
    if st.tie is not None or (st.in_note and False):
        text = re.sub(r"-(?=[\s|:\]]*$)", "", text)
    if not st.at_boundary():
        text += _rest_fill(st)
    text = text.rstrip()
    for _ in range(3):
        m = re.search(r"[|:\]]+$", text)
        if not m:
            return text + "|]"
        tok = m.group()
        if tok == "|:":
            text = text[:m.start()].rstrip()
            continue
        new = ":|" if tok in (":|", ":||", "::") else "|]"
        return text[:m.start()] + new
    return text

dead_end = [0]    # contador de callejones (diagnóstico)

# ═══════════════════════════════════════════════════════════════════════════
#  EXTENSIÓN: restricciones duras, priores suaves y controladores
#   Constraint  → allows(st, ch): veto extra sobre lo que ya permite AbcState
#   Prior       → apply(g, allowed, q): reescala q (masa conservada dentro de su grupo)
#   Controller  → before(g) puede forzar texto; temp_scale(g); after(g, ch, bits)
#  Se pasan al Generator en las listas constraints / priors / controllers.
# ═══════════════════════════════════════════════════════════════════════════
class Constraint:
    def allows(self, st, ch):
        return True

class Prior:
    def apply(self, g, allowed, q):
        pass

class Controller:
    at_start = False                  # True → before() también se llama antes del primer compás
    def stop(self, g):                # True → la pieza se cierra en el siguiente punto seguro
        return False
    def reset(self):
        pass
    def before(self, g):
        return None
    def temp_scale(self, g):
        return 1.0
    def after(self, g, ch, bits):
        pass

_LETTERS = "ABCDEFGabcdefg"
_DEF_INT = {0: 1.0, 1: 3.0, 2: 2.2, 3: 1.2, 4: 0.7, 5: 0.4, 6: 0.25, 7: 0.3}

def _note_ref(st):
    """(grado de referencia, intervalo previo) para una nota que empieza ahora."""
    if st.in_note and st.phase == "post" and not st.is_rest:
        d = deg_of(st.pitch)
        return d, (None if st.last_deg is None else d - st.last_deg)
    return st.last_deg, st.last_int

class PitchPrior(Prior):
    """Base: w(ref, prev_int, new, st) → factor ≥ 0 sobre la altura resultante.
    Letras: se reescalan conservando la masa total del grupo de letras (no cambia
    P(nota vs. barra vs. espacio), solo QUÉ nota). Marcas , ': razón w(nuevo)/w(viejo),
    de modo que el producto letra×marca equivale a w(altura final)."""
    def __init__(self, strength=1.0):
        self.s = strength

    def w(self, ref, pint, new, st):
        return 1.0

    def apply(self, g, allowed, q):
        st = g.st
        if self.s <= 0 or st.chord is not None:
            return
        ref, pint = _note_ref(st)
        letters = [o for o in allowed if chr(o) in _LETTERS]
        if letters and ref is not None:
            t0 = q[letters].sum()
            if t0 > 0:
                for o in letters:
                    q[o] *= max(self.w(ref, pint, deg_of(chr(o)), st), 1e-6) ** self.s
                t1 = q[letters].sum()
                if t1 > 0:
                    q[letters] *= t0 / t1
        if st.in_note and st.phase == "post" and not st.is_rest and st.last_deg is not None:
            old = deg_of(st.pitch)
            for ch in ",'":
                o = ord(ch)
                if q[o] > 0:
                    new = deg_of(st.pitch + ch)
                    r = (max(self.w(st.last_deg, st.last_int, new, st), 1e-6) /
                         max(self.w(st.last_deg, st.last_int, old, st), 1e-6)) ** self.s
                    q[o] *= min(3.0, max(0.05, r))

class IntervalPrior(PitchPrior):
    """Favorece los intervalos (en grados diatónicos) del corpus; sin meta, una curva por defecto."""
    def __init__(self, hist=None, strength=1.0, floor=0.03, maxint=14):
        super().__init__(strength)
        raw = {}
        for i in range(-maxint, maxint + 1):
            raw[i] = (hist.get(str(i), hist.get(i, 0)) + 0.5) if hist else _DEF_INT.get(abs(i), 0.08)
        m = max(raw.values())
        self.tab = {i: max(floor, v / m) for i, v in raw.items()}
        self.M = maxint

    def w(self, ref, pint, new, st):
        return self.tab[max(-self.M, min(self.M, new - ref))]

class ContourPrior(PitchPrior):
    """Tras un salto grande, preferir volver en dirección contraria y por grados; evitar notas repetidas en serie."""
    def w(self, ref, pint, new, st):
        i = new - ref
        f = 1.0
        if pint is not None:
            if abs(pint) >= 4:
                if i * pint > 0:
                    f *= 0.45
                elif i != 0 and abs(i) <= 2:
                    f *= 1.5
            if pint == 0 and i == 0:
                f *= 0.6
        return f

class RangePrior(PitchPrior):
    """Penaliza alejarse del centro del registro reciente más de `half` grados."""
    def __init__(self, strength=1.0, half=7):
        super().__init__(strength)
        self.half = half

    def w(self, ref, pint, new, st):
        c = sum(st.recent) / len(st.recent) if st.recent else ref
        return math.exp(-0.3 * max(0.0, abs(new - c) - self.half))

# ── variaciones de un compás copiado ──
_NOTE_RE = re.compile(r'("[^"]*")|((?:\^\^|__|\^|_|=)?)([A-Ga-g])([,\']*)')

def _spell(d):
    o, i = divmod(d, 7)
    L = "CDEFGAB"[i]
    return (L.lower() + "'" * (o - 1)) if o >= 1 else (L + "," * (-o))

def shift_bar(text, k):
    """Desplaza todas las notas k grados diatónicos (sin alteraciones)."""
    def rep(m):
        if m.group(1):
            return m.group(1)
        return _spell(deg_of(m.group(3) + m.group(4)) + k)
    return _NOTE_RE.sub(rep, text)

def end_change(text, k):
    """Cambia solo la última nota k grados (final alternativo)."""
    ms = [m for m in _NOTE_RE.finditer(text) if not m.group(1)]
    if not ms:
        return text
    m = ms[-1]
    return text[:m.start()] + _spell(deg_of(m.group(3) + m.group(4)) + k) + text[m.end():]

class CopyBarController(Controller):
    """Forma: al empezar un compás, con cierta probabilidad se COPIA uno anterior
    (opcionalmente variado) en vez de muestrearlo. Los caracteres copiados pasan
    por el autómata y por el predictor igual que los generados.
      mode 'copy'    : distancias típicas 1,2,4,8 (periodicidad)
      mode 'AABA'... : secciones de `phrase_bars` compases; cada letra repetida copia la 1ª aparición
    """
    def __init__(self, mode="copy", p_copy=0.35, vary=0.5, phrase_bars=4):
        self.mode, self.p, self.vary, self.pb = mode, p_copy, vary, phrase_bars

    def _source(self, g, k):
        bars = g.bars
        if self.mode == "copy":
            if k < 1 or g.rng.random() >= self.p:
                return None
            ds = [d for d in (1, 2, 4, 8) if d <= k]
            if g.rng.random() < 0.2 and k >= 3:
                ds.append(int(g.rng.integers(3, k + 1)))
            wts = np.array([{1: 0.15, 2: 0.35, 4: 0.35, 8: 0.15}.get(d, 0.1) for d in ds])
            d = ds[int(g.rng.choice(len(ds), p=wts / wts.sum()))]
            return k - d, False
        pat = self.mode
        sec, off = divmod(k, self.pb)
        cyc = sec // len(pat)
        letter = pat[sec % len(pat)]
        first = pat.index(letter)
        if first == sec % len(pat):
            return None
        src = (cyc * len(pat) + first) * self.pb + off
        if src >= len(bars):
            return None
        return src, off == self.pb - 1

    def before(self, g):
        if not g.just_barred:
            return None
        st = g.st
        if st.bt not in _TOKENS or st.bt_zero or st.bt_partial or st.pos != 0 or st.need_pick:
            return None
        k = len(g.bars)
        if k >= 2 and g.bars[-1] == g.bars[-2]:
            return None
        r = self._source(g, k)
        if r is None or r[0] >= len(g.bars):
            return None
        src, alt_end = r
        text = g.bars[src]
        cands = [text]
        if g.rng.random() < (1.0 if alt_end else self.vary) and "z" not in text.replace('"', ""):
            kk = int(g.rng.choice([-2, -1, 1, 2]))
            cands.insert(0, end_change(text, kk) if (alt_end or g.rng.random() < 0.5) else shift_bar(text, kk))
        best, bn = "", -1
        for c in cands:
            n = g.count_accepted(c)
            if n == len(c):
                return c + ""
            if n > bn:
                best, bn = c[:n], n
        return best or None

class SurpriseController(Controller):
    """Curva de sorpresa objetivo: la temperatura se ajusta para que los bits/carácter
    (según el modelo sin restringir) sigan base × curva(fase). Los primeros `warm`
    caracteres miden la base."""
    def __init__(self, curve="arch", gain=0.04, warm=80, alpha=0.06):
        self.curve, self.gain, self.warm, self.alpha = curve, gain, warm, alpha
        self.ema, self.base_sum, self.n, self.scale = None, 0.0, 0, 1.0

    def reset(self):
        self.ema, self.base_sum, self.n, self.scale = None, 0.0, 0, 1.0

    def f(self, ph):
        c = self.curve
        if c == "flat":
            return 1.0
        if c == "rise":
            return 0.7 + 0.7 * ph
        if c == "arch":
            v = 0.85 + 0.45 * math.exp(-(((ph - 0.65) / 0.2) ** 2))
            return v * (1 - 0.4 * min(1.0, max(0.0, (ph - 0.9) / 0.1)))
        xs = c if isinstance(c, list) else [float(x) for x in c.split(",")]
        if len(xs) == 1:
            return xs[0]
        t = ph * (len(xs) - 1)
        i = min(int(t), len(xs) - 2)
        return xs[i] * (1 - (t - i)) + xs[i + 1] * (t - i)

    def after(self, g, ch, bits):
        self.n += 1
        self.ema = bits if self.ema is None else self.ema * (1 - self.alpha) + bits * self.alpha
        if self.n <= self.warm:
            self.base_sum += bits
        elif self.base_sum > 0:
            base = self.base_sum / self.warm
            ph = min(1.0, (g.nprod) / max(1, g.chars))
            tgt = max(0.05, base * self.f(ph))
            err = max(-1.5, min(1.5, (tgt - self.ema) / tgt))
            self.scale = min(2.0, max(0.5, self.scale * math.exp(self.gain * err)))
        g.curve_log.append((g.nprod, self.scale))

    def temp_scale(self, g):
        return self.scale


# ═══════════════════════════════════════════════════════════════════════════
#  v3 — GRAMÁTICA: muestreo de ELEMENTOS (compases / reglas Re-Pair) como controlador
#
#  Se aprende del corpus ABC una gramática Re-Pair sobre la secuencia de COMPASES (cada compás,
#  normalizado a la tónica, es un símbolo: tokens (semitonos sobre la tónica, duración)). El
#  GrammarController, al empezar un compás/frase, muestrea un ELEMENTO (compás suelto o regla de
#  2..N compases) con la puntuación de grammar_sampler.py (sucesor, frontera, compatibilidad,
#  posición en la frase, ocurrencias, cierre) y lo inyecta como texto ABC. Ese texto pasa por el
#  autómata (duración exacta, restricciones) y por el predictor igual que cualquier carácter; lo
#  que el autómata no admita se trunca y el modelo de caracteres continúa. Si no hay controlador
#  que inyecte, el modelo sigue solo. Es compatible con --form, --curve, --best-of, --expressive...
# ═══════════════════════════════════════════════════════════════════════════
GS_REST = 99                                   # "altura" de un silencio en los tokens
GS_DEF_W = {"succ": 1.0, "bound": 1.0, "compat": 1.0, "pos": 1.0, "occ": 1.0, "len": 0.0, "fin": 1.0}
GS_COMPAT = 3.0

def gs_melody_line(notes, grid, legato=0.3, min_rest=2, exact=False):
    """Skyline monofónico cuantizado: [(inicio_u, fin_u, altura)] sin solapes (u = unidades de rejilla).
    Los huecos de articulación (≤ max(1/4 pulso, legato·duración) o < min_rest unidades) se rellenan; los silencios reales se conservan."""
    q = sorted(([int(round(on / grid)), on, on + dur, p] for on, dur, p, *_ in notes), key=lambda x: (x[0], -x[3]))
    sky = []
    for it in q:
        if not sky or sky[-1][0] != it[0]:
            sky.append(it)
    for k in range(0 if exact else len(sky) - 1):
        gap = sky[k + 1][1] - sky[k][2]
        if 0 < gap <= max(0.25, legato * (sky[k][2] - sky[k][1])):
            sky[k][2] = sky[k + 1][1]
    out = []
    for k, (s, on, end, p) in enumerate(sky):
        e = max(int(round(end / grid)), s + 1)
        if k + 1 < len(sky):
            e = min(e, sky[k + 1][0])
            if 0 < sky[k + 1][0] - e < min_rest:
                e = sky[k + 1][0]
        out.append((s, e, p))
    return out

def gs_split_bars(line, U, base):
    """→ (compases como tuplas de tokens (rel, unidades), nº de notas partidas por barra)."""
    total = line[-1][1]
    nb = -(-total // U)
    bars = [[] for _ in range(nb)]
    split = 0

    def put(a, b, rel):
        nonlocal split
        while a < b:
            bi = a // U
            end = min(b, (bi + 1) * U)
            bars[bi].append((rel, end - a))
            if end < b and rel != GS_REST:
                split += 1
            a = end
    cur = 0
    for s, e, p in line:
        if s > cur:
            put(cur, s, GS_REST)
        put(s, e, p - base)
        cur = e
    if cur < nb * U:
        put(cur, nb * U, GS_REST)
    return [tuple(b) for b in bars], split

def gs_repair(seqs, V, min_pair=2, max_rules=None):
    """Re-Pair sobre secuencias de enteros 0..V-1 (una por pieza; los pares no cruzan piezas).
    → (reglas {símbolo: (a, b)} con símbolos ≥ V, secuencias parseadas)."""
    cur = []
    for s in seqs:
        cur.extend(s)
        cur.append(-1)
    cur = np.array(cur, dtype=np.int64)
    rules, banned, nxt = {}, [], V
    while max_rules is None or len(rules) < max_rules:
        a, b = cur[:-1], cur[1:]
        ok = (a >= 0) & (b >= 0)
        for x, y in banned:
            ok &= ~((a == x) & (b == y))
        if not ok.any():
            break
        M = int(cur.max()) + 2
        uniq, cnt = np.unique(a[ok] * M + b[ok], return_counts=True)
        j = int(np.argmax(cnt))                       # empate → par menor (determinista)
        if cnt[j] < min_pair:
            break
        ka, kb = int(uniq[j] // M), int(uniq[j] % M)
        sel, last = [], -2
        for p in np.nonzero(ok & (a == ka) & (b == kb))[0].tolist():
            if p > last + 1:                          # sin solapes (aaa → un solo par)
                sel.append(p)
                last = p
        if len(sel) < min_pair:
            banned.append((ka, kb))
            continue
        sel = np.array(sel)
        cur[sel] = nxt
        keep = np.ones(len(cur), bool)
        keep[sel + 1] = False
        cur = cur[keep]
        rules[nxt] = (ka, kb)
        nxt += 1
    parsed, row = [], []
    for v in cur.tolist():
        if v == -1:
            parsed.append(row)
            row = []
        else:
            row.append(v)
    return rules, parsed

def gs_expand_rules(rules, V):
    exp = {}
    for s in sorted(rules):                           # los hijos siempre tienen ID menor
        a, b = rules[s]
        exp[s] = ([a] if a < V else exp[a]) + ([b] if b < V else exp[b])
    return exp

def gs_bar_features(bar, U):
    """Vector de compatibilidad de un compás (clases de altura, intervalos, densidad, registro…)."""
    pc, pit, rest = np.zeros(12), [], 0
    for rel, u in bar:
        if rel == GS_REST:
            rest += u
        else:
            pc[rel % 12] += u
            pit.append(rel)
    ih = np.zeros(5)
    for x, y in zip(pit, pit[1:]):
        d = abs(y - x)
        ih[0 if d == 0 else 1 if d <= 2 else 2 if d <= 4 else 3 if d <= 7 else 4] += 1
    v = np.concatenate([pc / pc.sum() if pc.sum() else pc, 0.7 * ih / ih.sum() if ih.sum() else ih,
                        [min(1.0, len(pit) / (U / 2.0)), rest / U,
                         0.5 * (np.mean(pit) / 12.0 if pit else 0.0), 0.5 * ((max(pit) - min(pit)) / 12.0 if pit else 0.0)]])
    n = np.linalg.norm(v)
    return v / n if n else v

class GsGrammar:
    """Gramática de un Part. Elementos = compases sueltos + (según `inventory`) secuencias de compases:
       repair → reglas de Re-Pair de ≤ max_elem compases | ngram → toda secuencia repetida | bars → solo compases."""
    def __init__(self, part, pb=4, min_pair=2, max_rules=600, max_elem=8, inventory="repair", align=False, posmode="bar"):
        self.part, self.pb, self.U, self.align = part, pb, part.U, align
        V = self.V = len(part.vocab)
        seqs = [t["bars"] for t in part.tunes]
        need(V > 0 and seqs, "corpus vacío")
        self.rules, self.parsed = {}, seqs
        cand = set()
        if inventory == "repair":
            # align=True: Re-Pair por frases (separador cada pb compases) → ninguna regla cruza una frontera de frase
            chunks, owner = [], []
            for ti, s in enumerate(seqs):
                for i in range(0, len(s), pb) if align else [0]:
                    chunks.append(s[i:i + pb] if align else s)
                    owner.append(ti)
            self.rules, pch = gs_repair(chunks, V, min_pair, max_rules)
            self.parsed = [[] for _ in seqs]
            for o, pc in zip(owner, pch):
                self.parsed[o].extend(pc)
            exp = gs_expand_rules(self.rules, V)
            cand = {tuple(e) for e in exp.values() if len(e) <= max_elem}
        elif inventory == "ngram":
            cnt = Counter()
            for s in seqs:
                for n in range(2, max_elem + 1):
                    for i in range(len(s) - n + 1):
                        if not align or i // pb == (i + n - 1) // pb:
                            cnt[tuple(s[i:i + n])] += 1
            cand = {k for k, c in cnt.items() if c >= min_pair}
        else:
            need(inventory == "bars", f"inventario desconocido: {inventory}")
        singles = {(b,) for s in seqs for b in s}
        self.elems = sorted(singles | cand, key=lambda t: (len(t), t))
        E = self.E = len(self.elems)
        by_len = defaultdict(dict)
        for k, e in enumerate(self.elems):
            by_len[len(e)][e] = k
        occ, pos, st0, fin = np.zeros(E), np.zeros((E, pb)), np.zeros(E), np.zeros(E)
        for s in seqs:                                # estadísticas de posición por ocurrencia real
            L = len(s)
            for n, d in by_len.items():
                for i in range(L - n + 1):
                    k = d.get(tuple(s[i:i + n]))
                    if k is not None:
                        occ[k] += 1
                        pos[k, i % pb] += 1
                        st0[k] += (i == 0)
                        fin[k] += (i + n == L)
        a = 0.5
        self.E_n = np.array([len(e) for e in self.elems])
        self.E_head = np.array([e[0] for e in self.elems])
        self.E_tail = np.array([e[-1] for e in self.elems])
        self.occ = occ
        self.occ_lp = np.log(np.maximum(occ, 1.0))
        self.len_lp = np.log(self.E_n.astype(float))
        # posición en la frase / inicio / final: por ELEMENTO ("elem") o compuesta con las estadísticas de sus
        # COMPASES ("bar": suma de log P(pos | compás), así un solo compás mal colocado se castiga entero;
        # el inicio usa el compás cabeza y el final el compás cola).
        # Los elementos raros tienen pocos datos propios; sus compases constituyentes tienen muchos.
        self.pos_lp = np.log((pos + a) / (occ[:, None] + a * pb))
        self.st0_lp = np.log((st0 + a) / (occ + 2 * a))
        self.fin_lp = np.log((fin + a) / (occ + 2 * a))
        self.nfin_lp = np.log((occ - fin + a) / (occ + 2 * a))
        self.pos0_lp = self.st0_lp
        if posmode == "bar":
            bocc, bpos, bst0, bfin = np.zeros(V), np.zeros((V, pb)), np.zeros(V), np.zeros(V)
            for s in seqs:
                for i, b in enumerate(s):
                    bocc[b] += 1
                    bpos[b, i % pb] += 1
                    bst0[b] += (i == 0)
                    bfin[b] += (i == len(s) - 1)
            bpos_lp = np.log((bpos + a) / (bocc[:, None] + a * pb))
            maxn = int(self.E_n.max())
            pad = np.full((E, maxn), -1)
            for k, e in enumerate(self.elems):
                pad[k, :len(e)] = e
            pl = np.zeros((E, pb))
            for p in range(pb):
                for i in range(maxn):
                    m = pad[:, i] >= 0
                    pl[m, p] += bpos_lp[pad[m, i], (p + i) % pb]
            self.pos_lp = pl
            self.st0_lp = np.log((bst0 + a) / (bocc + 2 * a))[self.E_head]
            self.fin_lp = np.log((bfin + a) / (bocc + 2 * a))[self.E_tail]
            self.nfin_lp = np.log((bocc - bfin + a) / (bocc + 2 * a))[self.E_tail]
            self.pos0_lp = self.st0_lp + self.pos_lp[:, 0]
        else:
            need(posmode == "elem", f"posmode desconocido: {posmode}")
        # bordes y rasgos por compás
        head, tail, lastp = np.full(V, np.nan), np.full(V, np.nan), np.full(V, np.nan)
        for i, b in enumerate(part.vocab):
            if b[0][0] != GS_REST:
                head[i] = b[0][0]
            if b[-1][0] != GS_REST:
                tail[i] = b[-1][0]
            pit = [r for r, _ in b if r != GS_REST]
            if pit:
                lastp[i] = pit[-1]
        self.tail = tail
        self.E_headp = head[self.E_head]
        lt = lastp[self.E_tail]
        self.E_lastrel = lt                              # última altura exacta (NaN si el elemento acaba sin notas)
        self.E_lastpc = np.where(np.isnan(lt), -1, np.mod(np.nan_to_num(lt), 12)).astype(int)
        self.F = np.array([gs_bar_features(b, self.U) for b in part.vocab])
        # bigramas de compás (backoff Witten-Bell a la frecuencia de compás) e intervalos de frontera
        uni, self.follow = np.zeros(V), defaultdict(Counter)
        ih, nrest, nb = np.zeros(25), 0, 0
        for s in seqs:
            for i, b in enumerate(s):
                uni[b] += 1
                if i + 1 < len(s):
                    self.follow[b][s[i + 1]] += 1
                    nb += 1
                    if np.isnan(tail[b]) or np.isnan(head[s[i + 1]]):
                        nrest += 1
                    else:
                        ih[int(np.clip(head[s[i + 1]] - tail[b], -12, 12)) + 12] += 1
        self.Pu = (uni + 1.0) / (uni.sum() + V)
        self.cnt_prev = {b: sum(c.values()) for b, c in self.follow.items()}
        r = min(0.98, max(0.02, nrest / nb)) if nb else 0.5
        self.rest_lp = math.log(r)
        self.pit_lp = np.log((1 - r) * (ih + 0.5) / (ih.sum() + 12.5))

    # puntuaciones vectoriales sobre todos los elementos, dado el compás previo
    def succ_vec(self, prev):
        fol = self.follow.get(prev)
        c, u = self.cnt_prev.get(prev, 0), (len(fol) if fol else 0)
        pv = (u / (c + u) if c + u else 1.0) * self.Pu
        if fol:
            for b, k in fol.items():
                pv[b] += k / (c + u)
        return pv[self.E_head]

    def bound_vec(self, prev):
        tp = self.tail[prev]
        if np.isnan(tp):
            return np.zeros(self.E)
        iv = self.E_headp - tp
        lp = np.full(self.E, self.rest_lp)
        m = ~np.isnan(iv)
        lp[m] = self.pit_lp[np.clip(iv[m], -12, 12).astype(int) + 12]
        return lp

    def compat_vec(self, prev):
        return (self.F @ self.F[prev])[self.E_head]

    def describe(self, top=8):
        cov = Counter()
        tot = sum(len(t["bars"]) for t in self.part.tunes)
        exp = gs_expand_rules(self.rules, self.V)
        multi = sum(len(exp[x]) for s in self.parsed for x in s if x >= self.V)
        lens = Counter(int(n) for n in self.E_n)
        out = [f"compases distintos: {self.V}   piezas: {len(self.part.tunes)}   compases totales: {tot}",
               f"reglas Re-Pair: {len(self.rules)}   elementos muestreables: {self.E}   por nº de compases: "
               + ", ".join(f"{n}:{c}" for n, c in sorted(lens.items())),
               f"cobertura de reglas (compases del corpus dentro de una regla del parseo): {multi / max(tot, 1):.1%}"]
        order = np.argsort(-(self.occ * (self.E_n - 1 + 0.01)))[:top]
        out.append(f"{'×':>5} {'n':>2}  contenido (compás → tokens grado:unidades, '-' = silencio)")
        for k in order:
            if self.E_n[k] < 2 and self.occ[k] < 2:
                continue
            e = self.elems[k]
            out.append(f"{int(self.occ[k]):5d} {len(e):2d}  " + " | ".join(
                " ".join(("-" if r == GS_REST else str(r)) + ":" + str(u) for r, u in self.part.vocab[b]) for b in e[:3])
                + (" …" if len(e) > 3 else ""))
        return "\n".join(out)


_GS_WRITE = {}
def gs_writable():
    """{multiplicador de L: sufijo más corto} que AbcState admite tras una nota (p. ej. 1:'', 2:'2', 1/2:'/2'; no 3/2 ni 12)."""
    if not _GS_WRITE:
        d0 = AbcState(Fraction(10 ** 6), Fraction(1, 8))
        d0.step("C")
        def dfs(st, s):
            m = st._dur() / st.n_base
            if m not in _GS_WRITE or len(s) < len(_GS_WRITE[m]):
                _GS_WRITE[m] = s
            for ch in "0123456789/":
                c = st.clone()
                if c.step(ch):
                    dfs(c, s + ch)
        dfs(d0, "")
    return _GS_WRITE

class GsPart:
    """Piezas del corpus con la misma tonalidad-modo y compás. Cada compás distinto es un símbolo."""
    def __init__(self, shift, num, den, grid):
        self.shift, self.num, self.den, self.grid = shift, num, den, grid
        self.U = int(round(4.0 * num / den / grid))
        self.vocab, self.vid, self.tunes, self.regs = [], {}, [], []

    @property
    def label(self):
        return f"modo{self.shift:+d} {self.num}/{self.den}"

    def bar_id(self, bar):
        k = self.vid.get(bar)
        if k is None:
            k = self.vid[bar] = len(self.vocab)
            self.vocab.append(bar)
        return k

    def add(self, name, bars):
        self.tunes.append({"name": name, "bars": [self.bar_id(b) for b in bars]})

    def base_for(self, tonic):
        """Altura MIDI de la tónica (grado 0) que reproduce el REGISTRO ABSOLUTO típico del corpus, sea cual sea la tonalidad."""
        if not self.regs:
            return tonic + 60
        ab = float(np.median([a for a, _ in self.regs]))
        rel = float(np.median([r for _, r in self.regs]))
        return tonic + 12 * int(round((ab - rel - tonic) / 12))

def gs_tonic_pc(kstr):
    m = re.match(r"\s*([A-G])([#b]?)", kstr)
    return (_BASE_PC[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)) % 12

def gs_load_corpus(text, grid=0.25):
    """Corpus ABC (varias piezas) → {(desplazamiento de modo, num, den): GsPart}."""
    parts, used, skipped = {}, 0, 0
    for i, src in enumerate(p for p in re.split(r"(?m)^(?=X:)", text) if p.strip()):
        mk = re.search(r"(?m)^K:\s*(.*)$", src)
        try:
            pk = parse_key(mk.group(1)) if mk else None
            pc = parse_abc(src)
        except (SystemExit, Exception):
            pk = pc = None
        if not pk or not pc or len(pc["notes"]) < 8:
            skipped += 1
            continue
        num, den = pc["meter"]
        bb = pc["bar_beats"]
        U = int(round(bb / grid))
        if abs(U * grid - bb) > 1e-6 or U < 2:
            skipped += 1
            continue
        st = pc["starts"]
        t0 = st[1] if len(st) > 1 and st[0] < -1e-9 else 0.0          # anacrusa: el primer compás completo marca el origen
        line = gs_melody_line([(t - t0, d, m) for t, d, m, *_ in pc["notes"] if t >= t0 - 1e-9], grid, exact=True, min_rest=0)
        if len(line) < 6:
            skipped += 1
            continue
        tonic = gs_tonic_pc(mk.group(1))
        last = line[-1][2]
        if last % 12 == tonic:
            base = last
        else:
            base = tonic + 12 * int(round((float(np.median([p for _, _, p in line])) - tonic) / 12))
        bars, _ = gs_split_bars(line, U, base)
        if len(bars) < 4:
            skipped += 1
            continue
        key = (pk[1], num, den)
        if key not in parts:
            parts[key] = GsPart(pk[1], num, den, grid)
        parts[key].add(f"tune{i}", bars)
        ab = float(np.median([p for _, _, p in line]))
        parts[key].regs.append((ab, ab - base))
        used += 1
    return parts, used, skipped

_GS_CACHE = {}
def gs_cached_corpus(path, grid=0.25):
    key = (os.path.abspath(path), os.path.getmtime(path), grid)
    if key not in _GS_CACHE:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            _GS_CACHE[key] = gs_load_corpus(fh.read(), grid)
    return _GS_CACHE[key]

class GsPicker:
    """Puntuación de elementos igual que ElementSampler de grammar_sampler.py, con el compás previo dado como tokens."""
    def __init__(self, G, temp=1.0, w=None):
        self.G, self.temp = G, temp
        self.w = dict(GS_DEF_W)
        if w:
            self.w.update(w)

    def pick(self, prev, k, rem, rng, final=True):
        """prev: tokens del compás anterior (None al principio); k: compases ya hechos; rem: compases que faltan."""
        G, w = self.G, self.w
        sc = np.zeros(G.E)
        if prev:
            pid = G.part.vid.get(prev, -1)
            pit = [r for r, _ in prev if r != GS_REST]
            tp = prev[-1][0] if prev[-1][0] != GS_REST else np.nan
            if w["succ"]:
                sc += w["succ"] * np.log(G.succ_vec(pid) if pid >= 0 else G.Pu[G.E_head])
            if w["bound"] and not np.isnan(tp):
                iv = G.E_headp - tp
                lp = np.full(G.E, G.rest_lp)
                m = ~np.isnan(iv)
                lp[m] = G.pit_lp[np.clip(iv[m], -12, 12).astype(int) + 12]
                sc += w["bound"] * lp
            if w["compat"]:
                f = G.F[pid] if pid >= 0 else gs_bar_features(prev, G.U)
                sc += w["compat"] * GS_COMPAT * (G.F @ f)[G.E_head]
        if w["pos"]:
            sc += w["pos"] * (G.pos0_lp if k == 0 else G.pos_lp[:, k % G.pb])
        if w["occ"]:
            sc += w["occ"] * G.occ_lp
        if w["len"]:
            sc += w["len"] * G.len_lp
        if w["fin"]:
            sc += w["fin"] * np.where((G.E_n == rem) & final, G.fin_lp, G.nfin_lp)
        mask = (G.E_n <= rem) & getattr(G, "E_valid", True)
        if final:
            for ok in (G.E_lastrel == 0, G.E_lastpc == 0):
                m2 = mask & ((G.E_n != rem) | ok)
                if m2.any() and (m2 & (G.E_n == rem)).any():
                    mask = m2
                    break
        z = np.where(mask, sc / max(self.temp, 1e-6), -np.inf)
        p = np.exp(z - z.max())
        p /= p.sum()
        return int(rng.choice(G.E, p=p))

class GrammarController(Controller):
    """Inyecta elementos de la gramática (ver arriba). target = nº de compases de la pieza (para cadencia y parada)."""
    at_start = True

    def __init__(self, part, header, bar_len, unit, target=32, p=1.0, temp=1.0, w=None, pb=4, max_elem=8,
                 inventory="repair", align=False, chords=False, palette=None):
        self.part, self.header, self.unit, self.target, self.p, self.grid = part, header, unit, target, p, part.grid
        self.G = GsGrammar(part, pb, 2, 600, max_elem, inventory, align)
        self.pick = GsPicker(self.G, temp, w)
        mk = re.search(r"(?m)^K:\s*(.*)$", header)
        pk = parse_key(mk.group(1)) if mk else None
        need(pk, "--grammar: la cabecera necesita una línea K: reconocible")
        self.fifths, self.tonic = pk[0], gs_tonic_pc(mk.group(1))
        self.base = part.base_for(self.tonic)
        self.U = part.U
        sig = key_sig(self.fifths)
        self.acck = {c: sig.get(c, 0) for c in "CDEFGAB"}
        nat = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        self.natural = {v: k for k, v in nat.items()}
        self.diat = {(nat[c] + self.acck[c]) % 12: (c, self.acck[c]) for c in "CDEFGAB"}
        self.chrom = {1: ("C", 1), 3: ("D", 1), 6: ("F", 1), 8: ("G", 1), 10: ("A", 1)} if self.fifths >= 0 else \
                     {1: ("D", -1), 3: ("E", -1), 6: ("G", -1), 8: ("A", -1), 10: ("B", -1)}
        mm = _GS_FIELD_RE.search(header)
        compound = bool(mm and int(mm.group(1)) % 3 == 0)
        ok = np.zeros(len(part.vocab), bool)
        for b, bar in enumerate(part.vocab):
            stt = AbcState(bar_len, unit, chords=chords, palette=palette, compound=compound)
            tx = self.text_of(bar)
            ok[b] = (tx is not None and any(r != GS_REST for r, _ in bar)      # sin compases enteros de silencio
                     and all(stt.step(c) for c in tx))
        self.G.E_valid = np.array([all(ok[b] for b in e) for e in self.G.elems])
        need(self.G.E_valid.any(), "--grammar: ningún compás del corpus se puede escribir con este L:/--rhythms (¿unidad L: demasiado grande?)")
        self.n_invalid = int((~ok).sum())
        if self.n_invalid:
            print(yellow(f"aviso: {self.n_invalid} de {len(ok)} compases del corpus no se pueden escribir con L:{unit}"
                         f"{' y la paleta rítmica' if palette else ''}; se excluyen del muestreo"), file=sys.stderr)
        self.reset()

    def reset(self):
        self.queue, self.expect, self.trace, self.injected, self.truncated = [], None, [], 0, 0

    def stop(self, g):
        return self.target > 0 and len(g.bars) >= self.target

    def claims(self, g):
        """El último compás siempre lo decide la gramática (cadencia en la tónica), aunque otro controlador (--form) copiase."""
        return self.p > 0 and self.target > 0 and len(g.bars) == self.target - 1

    # ── conversión texto ↔ tokens ──
    def tokens_of(self, text):
        try:
            pc = parse_abc(f"X:1\n{self.header}{text}|]\n")
        except (SystemExit, Exception):
            return None
        line = gs_melody_line([(t, d, m) for t, d, m, *_ in pc["notes"]], self.grid, exact=True, min_rest=0)
        if not line:
            return ((GS_REST, self.U),)
        bars, _ = gs_split_bars(line, self.U, self.base)
        return bars[0]

    def _lens(self, u):
        """Duración en unidades de L → lista de sufijos de longitud (None si no es escribible). Lo que el autómata no admite
        en una sola cifra/fracción (3/2, 12…) se parte en trozos que se escribirán ligados: 3/2 → 1+1/2, 12 → 8+4."""
        r = Fraction(u).limit_denominator(64) * Fraction(self.grid).limit_denominator(64) / 4 / self.unit
        tab = gs_writable()
        ms = sorted(tab, reverse=True)
        out = []
        while r > 0:
            m = next((x for x in ms if x <= r), None)
            if m is None:
                return None
            out.append(tab[m])
            r -= m
        return out

    def text_of(self, bar):
        st, toks = {}, []
        for rel, u in bar:
            lns = self._lens(u)
            if lns is None:
                return None
            if rel == GS_REST:
                toks.extend("z" + ln for ln in lns)
                continue
            p = self.base + rel
            pc = p % 12
            letter, acc = self.diat.get(pc) or ((self.natural[pc], 0) if pc in self.natural else self.chrom[pc])
            oc = (p - acc) // 12 - 1
            cur = st.get((letter, oc), self.acck[letter])
            sym = "" if acc == cur else ("^" if acc == 1 else "_" if acc == -1 else "=")
            st[(letter, oc)] = acc
            name = letter if oc <= 4 else letter.lower()
            name += "," * (4 - oc) if oc < 4 else "'" * (oc - 5) if oc > 5 else ""
            toks.append("-".join(sym + name + ln for ln in lns))      # el autómata exige repetir la alteración en la nota ligada
        return " ".join(toks)

    # ── ganchos del Generator ──
    def before(self, g):
        if not g.just_barred:
            return None
        st = g.st
        k = len(g.bars)
        if st.bt_zero or st.bt_partial or st.pos != 0 or st.need_pick or st.in_note:
            return None
        if k > 0 and st.bt not in _TOKENS:
            return None
        if k == 0 and st.npitch > 0:
            return None
        if self.queue and self.expect != k:                 # otro controlador intervino: abandonar la regla en curso
            self.queue = []
        if not self.queue:
            if k >= self.target or self.p <= 0 or (k < self.target - 1 and g.rng.random() >= self.p):
                return None
            prev = self.tokens_of(g.bars[-1]) if k > 0 else None
            if k > 0 and prev is None:
                return None
            e = self.pick.pick(prev, k, self.target - k, g.rng)
            self.trace.append(int(self.G.E_n[e]))
            self.queue = [self.text_of(self.part.vocab[b]) for b in self.G.elems[e]]
        text = self.queue.pop(0)
        n = g.count_accepted(text)
        if n < len(text):                                   # el autómata no lo admite entero: las notas completas que quepan
            self.truncated += 1
            self.queue = []
            keep = text[:n]
            keep = keep[:keep.rfind(" ") + 1]               # nunca a mitad de nota
            return keep or None
        self.injected += 1
        self.expect = k + 1
        return text

    def report_line(self):
        return (f"  gramática                 {self.injected} compases inyectados en {len(self.trace)} elementos "
                f"({'+'.join(map(str, self.trace)) or '-'}), {self.truncated} truncados")

_GS_FIELD_RE = re.compile(r"(?m)^M:\s*(\d+)/(\d+)")

def _make_grammar_controller(a, header, bar_len, unit, palette=None):
    need(os.path.isfile(a.grammar), f"--grammar: no existe el fichero {a.grammar!r}")
    need(a.grammar_bars >= 1, "--grammar-bars debe ser >= 1")
    need(0 <= a.p_grammar <= 1, "--p-grammar debe estar entre 0 y 1")
    parts, used, skipped = gs_cached_corpus(a.grammar)
    need(parts, f"--grammar: ninguna pieza utilizable en {a.grammar} (¿hay X:, M:, K:?)")
    mk = re.search(r"(?m)^K:\s*(.*)$", header)
    pk = parse_key(mk.group(1)) if mk else None
    need(pk, "--grammar: la cabecera necesita una línea K: reconocible")
    mm = _GS_FIELD_RE.search(header)
    num, den = (int(mm.group(1)), int(mm.group(2))) if mm else (bar_len.numerator, bar_len.denominator)
    part = parts.get((pk[1], num, den))
    need(part, f"--grammar: el corpus no tiene piezas con modo{pk[1]:+d} y compás {num}/{den} "
               f"(hay: {', '.join(p.label for p in parts.values())})")
    return GrammarController(part, header, bar_len, unit, a.grammar_bars, a.p_grammar, a.grammar_temp,
                             _gs_weights(a.grammar_w), a.grammar_phrase_bars, a.max_elem, a.grammar_inventory, a.grammar_align,
                             bool(getattr(a, "chords", False)), palette)

def _gs_weights(s):
    w = {}
    for it in filter(None, (s or "").split(",")):
        k, _, v = it.partition("=")
        need(k.strip() in GS_DEF_W, f"--grammar-w: peso desconocido {k!r} (usa {', '.join(GS_DEF_W)})")
        try:
            w[k.strip()] = float(v)
        except ValueError:
            die(f"--grammar-w: valor no numérico en {it!r}")
    return w

def cmd_grammar(a):
    """Estadísticas de la gramática de un corpus ABC."""
    need(os.path.isfile(a.corpus), f"no existe {a.corpus!r}")
    parts, used, skipped = gs_cached_corpus(a.corpus)
    need(parts, "ninguna pieza utilizable (¿hay X:, M:, K:?)")
    print(bold(f"{used} piezas utilizadas, {skipped} omitidas") + dim("  · particiones (modo, compás):"))
    for p in sorted(parts.values(), key=lambda q: -len(q.tunes)):
        nb = sum(len(t["bars"]) for t in p.tunes)
        print(f"  {p.label:16s} {len(p.tunes):5d} piezas {nb:7d} compases {len(p.vocab):6d} distintos")
    sel = None
    if a.key or a.meter:
        pk = parse_key(a.key or "C")
        mm = re.match(r"(\d+)/(\d+)", a.meter or "4/4")
        need(pk and mm, "--key/--meter no interpretables")
        sel = parts.get((pk[1], int(mm.group(1)), int(mm.group(2))))
        need(sel, "no hay esa partición en el corpus")
    sel = sel or max(parts.values(), key=lambda q: len(q.tunes))
    print(bold(f"\ngramática de {sel.label}"))
    print(GsGrammar(sel, a.grammar_phrase_bars, 2, 600, a.max_elem, a.grammar_inventory, a.grammar_align).describe(a.top))

class Generator:
    """Muestreo restringido con ganchos. Reutilizable: run(seed) puede llamarse varias veces
    (el predictor se rebobina a su estado tras el prime)."""
    def __init__(self, pred, header, body_prime, bar_len, unit, chars=400, temp=1.0,
                 chords=False, palette=None, constraints=(), priors=(), controllers=()):
        self.pred, self.header, self.body_prime = pred, header, body_prime
        self.bar_len, self.unit, self.chars, self.temp = bar_len, unit, chars, temp
        self.constraints, self.priors, self.controllers = list(constraints), list(priors), list(controllers)
        mm = re.search(r"(?m)^M:\s*(\d+)/(\d+)", header)
        compound = bool(mm and int(mm.group(1)) % 3 == 0)     # regla de abc2midi: numerador múltiplo de 3
        self.st0 = AbcState(bar_len, unit, chords=chords, palette=palette, compound=compound)
        for i, c in enumerate(body_prime):
            if not self.st0.step(c):
                die(f"--prime incompatible con --constrain en el carácter {i} ({c!r})")
        for s in (header + body_prime).encode("ascii", errors="replace"):
            pred.update(s, False)
        self.can_rewind = hasattr(pred, "mark")
        if self.can_rewind:
            pred.mark()
        self.cand = [(c, ord(c)) for c in _CAND if chords or c not in '"#mijsu']
        self.ran = 0

    # ── utilidades para los controladores ──
    def count_accepted(self, text):
        t = self.st.clone()
        for i, c in enumerate(text):
            if not t.step(c):
                return i
        return len(text)

    def _push(self, ch, bits=None):
        st = self.st
        # posición del compás ANTES de la barra, contando la nota pendiente (sin espacio previo a '|')
        pend = st.in_note and st.phase == "post"
        pos0 = (st.n_start + st._dur()) if pend else st.pos
        bt0 = st.bt
        ok = st.step(ch)
        assert ok
        o = ord(ch)
        self.out.append(ch)
        self.pred.update(o, False)
        if not bt0 and st.bt and ch in "|:" and pos0 > 0 and st.pos == 0:
            seg = "".join(self.out[self.seg_start:-1]).lstrip("|:] \n").strip()
            if pos0 == self.bar_len and seg and any(c in _LETTERS for c in seg):
                self.bars.append(seg)          # solo compases completos
            self.seg_start = len(self.out)
            self.just_barred = True
        if st.at_boundary() and st.npitch > 0:
            self.safe = (len(self.out), st.clone())

    def run(self, seed=None):
        if self.ran and not self.can_rewind:
            die("este predictor no admite repetir la generación (best-of) — usa el backend ngram")
        if self.ran:
            self.pred.rewind()
        self.ran += 1
        for c in self.controllers:
            c.reset()
        self.rng = np.random.default_rng(seed)
        self.st = self.st0.clone()
        self.out = list(self.body_prime)
        self.bars, self.seg_start = [], len(self.out)
        self.just_barred = any(getattr(c, "at_start", False) for c in self.controllers)
        self.safe, self.nprod, self.curve_log = None, 0, []
        self.dead = 0
        self.rej_sum = self.rej_bits = 0.0
        self.rej_n = self.rej_hi = 0
        self.rej_worst = []
        self.bits_sum = 0.0
        self.forced = 0
        self.T = self.temp
        p0 = len(self.out)
        extra = 0
        while True:
            self.nprod = len(self.out) - p0
            done = self.nprod >= self.chars or any(c.stop(self) for c in self.controllers)
            if done:
                if (self.safe is not None and self.safe[0] == len(self.out)) or extra >= 250:
                    break
                extra += 1
            st = self.st
            if self.just_barred:
                forced = None
                for c in sorted(self.controllers, key=lambda c: 0 if getattr(c, "claims", lambda g: False)(self) else 1):
                    forced = c.before(self)
                    if forced:
                        break
                self.just_barred = False
                if forced:
                    for ch in forced:
                        self._push(ch)
                        self.forced += 1
                    continue
            allowed = [o for c, o in self.cand
                       if st.clone().step(c) and all(k.allows(st, c) for k in self.constraints)]
            if allowed and done and st.open_rep:     # fase final: cerrar la repetición abierta
                if st.bt == ":" and ord("|") in allowed:
                    allowed = [ord("|")]
                elif ord(":") in allowed:
                    allowed = [ord(":")]
            if not allowed:
                self.dead += 1
                dead_end[0] += 1
                break
            p = self.pred.probs()
            rej = 1.0 - float(p[allowed].sum())
            self.rej_sum += rej
            self.rej_bits += -math.log2(max(1.0 - rej, 1e-12))
            self.rej_n += 1
            self.rej_hi += rej > 0.5
            ts = 1.0
            for c in self.controllers:
                ts *= c.temp_scale(self)
            if len(self.bars) >= 3 and self.bars[-1] == self.bars[-2] == self.bars[-3]:
                ts *= 1.8                      # rompebucles: 3 compases idénticos seguidos → más temperatura
            T = self.temp * ts
            self.T = T
            pt = p
            if T != 1.0:
                lp = np.log(np.maximum(p, 1e-300)) / T
                pt = np.exp(lp - lp.max())
            q = np.zeros(256)
            q[allowed] = pt[allowed]
            if q.sum() < 1e-12 * max(pt.sum(), 1e-300):
                q[allowed] = 1.0
            for pr in self.priors:
                pr.apply(self, allowed, q)
            q /= q.sum()
            o = int(self.rng.choice(256, p=q))
            bits = -math.log2(max(float(p[o]), 1e-12))
            self.bits_sum += bits
            if len(self.rej_worst) < 5 or rej > self.rej_worst[-1][0]:
                self.rej_worst.append((rej, self.nprod, chr(o), "".join(self.out[-14:])))
                self.rej_worst = sorted(self.rej_worst, reverse=True)[:5]
            self._push(chr(o))
            for c in self.controllers:
                c.after(self, chr(o), bits)
        if self.safe is None:
            die(f"no se pudo generar ninguna pieza válida con M:{self.bar_len} L:{self.unit} "
                f"(prueba otra --seed, otro --chars o un L: más pequeño)")
        out, st = self.out[:self.safe[0]], self.safe[1]
        body = "".join(out)
        if st.slur > 0:                           # cerrar ligaduras abiertas tras la última nota
            m = re.search(r"[\s|:\]]*$", body)
            body = body[:m.start()] + ")" * st.slur + body[m.start():]
        self.body = _finish_text(body, st)
        return self.body

    def stats(self):
        n = max(1, self.rej_n)
        return {"chars": len(self.body), "dead_ends": self.dead, "forced_chars": self.forced,
                "rejected_mass_mean": self.rej_sum / n, "distortion_bits_per_char": self.rej_bits / n,
                "steps_over_50pct": self.rej_hi, "worst": self.rej_worst,
                "bits_per_char_model": self.bits_sum / n, "bars_recorded": len(self.bars)}

    def report(self):
        s = self.stats()
        lines = [f"  masa rechazada media      {100 * s['rejected_mass_mean']:.1f}%  (probabilidad que el modelo daba a caracteres inválidos)",
                 f"  distorsión por la máscara {s['distortion_bits_per_char']:.3f} bits/car",
                 f"  pasos con >50% rechazado  {s['steps_over_50pct']}",
                 f"  sorpresa del modelo       {s['bits_per_char_model']:.2f} bits/car",
                 f"  caracteres copiados       {s['forced_chars']} ({len(self.bars)} compases completos registrados)"]
        lines += [c.report_line() for c in self.controllers if hasattr(c, "report_line")]
        for rej, pos, ch, ctx in s["worst"][:3]:
            lines.append(f"  peor: {100 * rej:.0f}% rechazado en car. {pos} (eligió {ch!r}) …{ctx!r}")
        return "\n".join(lines)

STYLE_METER = {"reel": "4/4", "hornpipe": "4/4", "march": "4/4", "polka": "2/4", "jig": "6/8",
               "slip jig": "9/8", "waltz": "3/4", "slide": "12/8", "strathspey": "4/4"}

def _build_generator(a, pred):
    header, body_prime, bar_len, unit, ptitle = _constrained_header(a)
    need(0 <= getattr(a, "p_copy", 0.35) <= 1 and 0 <= getattr(a, "vary", 0.5) <= 1, "--p-copy y --vary deben estar entre 0 y 1")
    need(getattr(a, "phrase_bars", 4) >= 1, "--phrase-bars debe ser >= 1")
    need(getattr(a, "curve_gain", 0.04) >= 0, "--curve-gain debe ser >= 0")
    meta = getattr(pred, "meta", None)
    palette = make_palette(a.rhythms, unit, bar_len, meta) if getattr(a, "rhythms", None) else None
    priors, controllers = [], []
    s = getattr(a, "prior_strength", 0.0) or 0.0
    if s > 0:
        priors += [IntervalPrior((meta or {}).get("int_hist"), s), ContourPrior(s), RangePrior(s)]
    form = getattr(a, "form", None)
    if form and form != "none":
        mode = "copy" if form == "copy" else form.upper()
        need(mode == "copy" or re.fullmatch("[A-Z]{2,8}", mode), "--form: 'copy', 'none' o un patrón tipo AABA")
        controllers.append(CopyBarController(mode, a.p_copy, a.vary, a.phrase_bars))
    gram = getattr(a, "grammar", None)
    if gram:
        controllers.append(_make_grammar_controller(a, header, bar_len, unit, palette))
    curve = getattr(a, "curve", None)
    if curve and curve != "none":
        try:
            SurpriseController(curve).f(0.5)
        except ValueError:
            die("--curve: none, flat, arch, rise o lista de multiplicadores (p. ej. 0.8,1,1.3,0.9)")
        controllers.append(SurpriseController(curve, a.curve_gain))
    g = Generator(pred, header, body_prime, bar_len, unit, chars=max(a.chars, 100 * a.grammar_bars) if gram else a.chars, temp=a.temp,
                  chords=bool(getattr(a, "chords", False)), palette=palette, priors=priors, controllers=controllers)
    return g, header, ptitle, bar_len, unit

def _compose_text(a, g, header, ptitle):
    return f"X:1\nT:{a.title or ptitle or 'Generada'}\n{header}{g.body}\n"

def _generate_constrained(a, pred):
    dead_end[0] = 0
    g, header, ptitle, bar_len, unit = _build_generator(a, pred)
    n = max(1, getattr(a, "best_of", 1) or 1)
    seed0 = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % (2 ** 31))
    best = None
    for i in range(n):
        if i > 0 and not g.can_rewind:
            pred = load_predictor(a.model, a.backend)
            g, header, ptitle, bar_len, unit = _build_generator(a, pred)
        g.run(seed0 + i)
        text = _compose_text(a, g, header, ptitle)
        if n > 1:
            sc = score_piece(text, getattr(a, "rank_by", "auto"), pred_meta=getattr(pred, "meta", None))
            print(dim(f"  candidato {i + 1}/{n} (semilla {seed0 + i}): puntuación {sc['total']:.3f}"), file=sys.stderr)
            if best is None or sc["total"] > best[0]:
                best = (sc["total"], text, g.report(), g.dead)
        else:
            best = (0, text, g.report(), g.dead)
    _, text, rep, dead = best
    if getattr(a, "report", False):
        print(bold("informe de la restricción:"), file=sys.stderr)
        print(rep, file=sys.stderr)
    if a.output:
        with open(a.output, "w") as fh:
            fh.write(text)
        print(green(f"✓ {len(text)} caracteres ({bar_len} por compás, L={unit}) → {a.output}"
                    + (" · terminó en un callejón, truncado al último punto seguro" if dead else "")))
    else:
        print(text, end="")
    if getattr(a, "midi", None):
        _render_text_to_midi(a, text)
    return text

def _prime_text(p):
    """El --prime es un fichero si existe, y si no, el propio texto."""
    if p and "\n" not in p and len(p) < 4096 and "\x00" not in p and os.path.isfile(p):
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return p or ""

def cmd_generate(a):
    need(a.temp > 0, "--temp debe ser > 0")
    a.chars = max(0, a.chars)
    if getattr(a, "expressive", False):
        _apply_expressive(a)
    if not a.constrain and (a.chords or a.rhythms or a.prior_strength > 0 or a.form not in (None, "none")
                            or a.curve not in (None, "none") or a.best_of > 1 or a.style or a.report
                            or getattr(a, "grammar", None)):
        a.constrain = True                    # estas opciones solo existen en modo restringido
    need(a.best_of >= 1, "--best-of debe ser >= 1")
    need(a.constrain or not a.midi, "--midi necesita --constrain (el ABC debe ser válido para interpretarse)")
    pred = load_predictor(a.model, a.backend)
    if a.constrain:
        return _generate_constrained(a, pred)
    rng = np.random.default_rng(a.seed)
    out = bytearray()
    prime_len = 0
    if a.prime:
        prime = _prime_text(a.prime).encode("utf-8", errors="replace")
        for s in prime:
            pred.update(s, False)
        out += prime
        prime_len = len(prime)
    while len(out) < a.chars + prime_len:
        p = pred.probs()
        if a.temp != 1.0:
            lp = np.log(p) / a.temp
            lp -= lp.max()
            p = np.exp(lp)
            p /= p.sum()
        s = int(rng.choice(256, p=p))
        if s == 0:
            continue
        out.append(s)
        pred.update(s, False)
    text = out.decode("ascii", errors="replace")
    if a.output:
        with open(a.output, "w") as fh:
            fh.write(text + "\n")
        print(green(f"✓ {len(text)} caracteres → {a.output}"))
    else:
        print(text)

def _add_grammar_args(g):
    g.add_argument("--grammar", metavar="CORPUS.abc", help="corpus ABC del que aprender la gramática de compases/frases (activa el controlador)")
    g.add_argument("--grammar-bars", type=int, default=32, help="nº de compases de la pieza (cadencia final y parada); --chars se ensancha solo")
    g.add_argument("--p-grammar", type=float, default=1.0, help="prob. de inyectar un elemento en cada frontera (1 = siempre)")
    g.add_argument("--grammar-temp", type=float, default=1.0, help="temperatura del muestreo de elementos")
    g.add_argument("--grammar-w", default="", help="pesos de la puntuación: succ,bound,compat,pos,occ,len,fin (ej. succ=0.5,len=0.3)")
    g.add_argument("--grammar-phrase-bars", type=int, default=4, help="compases por frase para la posición métrica de los elementos")
    g.add_argument("--max-elem", type=int, default=8, help="máx. compases de un elemento (1 = solo compases sueltos)")
    g.add_argument("--grammar-inventory", choices=("repair", "ngram", "bars"), default="repair")
    g.add_argument("--grammar-align", action="store_true", help="Re-Pair por frases: ninguna regla cruza una frontera de frase")

def cmd_info(a):
    pred = load_predictor(a.model, a.backend)
    print(bold(a.model))
    print(f"  backend      {pred.name}")
    print(f"  tamaño       {pred.size_bytes()/1024:.1f} KiB (serializado)")
    if pred.name == "ngram":
        print(f"  orden        {pred.order}   match mínimo {pred.minlen}")
        print(f"  historia     {len(pred.hist):,} caracteres de entrenamiento")
        for k, t in enumerate(pred.tables):
            print(f"  orden {k}: {len(t):,} contextos")
    else:
        print(f"  config       {pred.cfg}")

# ═══════════════════════════════════════════════════════════════════════════
#  INTERPRETACIÓN EXPRESIVA (render): ABC → MIDI
#  Analizador ABC indulgente (alteraciones y armadura, ligaduras de unión, tresillos,
#  ritmo roto, repeticiones, símbolos de acorde) + humanización + acompañamiento.
#  Escritor SMF propio (sin dependencias).
# ═══════════════════════════════════════════════════════════════════════════
_BASE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_FIFTHS_MAJOR = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F": -1}
_MODE_SHIFT = {"": 0, "maj": 0, "major": 0, "ion": 0, "m": -3, "min": -3, "minor": -3, "aeo": -3,
               "dor": -2, "dorian": -2, "mix": -1, "mixolydian": -1, "lyd": 1, "lydian": 1,
               "phr": -4, "phrygian": -4, "loc": -5, "locrian": -5}
_MODE_TONIC_DEG = {0: 0, -3: 5, -2: 1, -1: 4, 1: 3, -4: 2, -5: 6}
_CHORD_Q = {"": [0, 4, 7], "m": [0, 3, 7], "7": [0, 4, 7, 10], "m7": [0, 3, 7, 10], "maj7": [0, 4, 7, 11],
            "dim": [0, 3, 6], "aug": [0, 4, 8], "sus4": [0, 5, 7], "sus2": [0, 2, 7], "6": [0, 4, 7, 9],
            "m6": [0, 3, 7, 9], "9": [0, 4, 7, 10, 14], "dim7": [0, 3, 6, 9], "m7b5": [0, 3, 6, 10]}
_TUPQ = {2: 3, 3: 2, 4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 9: 2}

def parse_key(s):
    """'Gm', 'Bb', 'D dor', 'F#mix' → (fifths, shift de modo). None si no se reconoce."""
    m = re.match(r"\s*([A-G])([#b]?)\s*([A-Za-z]*)", s)
    if not m:
        return None
    letter, acc, mode = m.group(1), m.group(2), m.group(3).lower()
    sh = _MODE_SHIFT.get(mode, _MODE_SHIFT.get(mode[:3], 0))
    f = _FIFTHS_MAJOR[letter] + (7 if acc == "#" else -7 if acc == "b" else 0)
    return f + sh, sh

def key_sig(fifths):
    d = {}
    if fifths > 0:
        for l in "FCGDAEB"[:min(7, fifths)]:
            d[l] = 1
    elif fifths < 0:
        for l in "BEADGCF"[:min(7, -fifths)]:
            d[l] = -1
    return d

def chord_pcs(name):
    """'Am7/E' → (lista de pc con la fundamental primero, pc del bajo)."""
    m = re.match(r"^([A-G])([#b]?)([^/]*)(?:/([A-G])([#b]?))?$", name)
    if not m or m.group(3) not in _CHORD_Q:
        return None
    root = (_BASE_PC[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)) % 12
    pcs = [(root + i) % 12 for i in _CHORD_Q[m.group(3)]]
    bass = root
    if m.group(4):
        bass = (_BASE_PC[m.group(4)] + (1 if m.group(5) == "#" else -1 if m.group(5) == "b" else 0)) % 12
    return pcs, bass

_TOK = re.compile(r"""
  (?P<comment>%[^\n]*)
 |(?P<field>^[A-Za-z]:[^\n]*)
 |(?P<inline>\[[A-Za-z]:[^\]]*\])
 |(?P<chord>"[^"]*")
 |(?P<grace>\{[^}]*\})
 |(?P<deco>![^!\n]*!|\+[^+\n]*\+)
 |(?P<bar>:\|\||::|:\||\|\||\|\]|\|:|\[\||\|)
 |(?P<tup>\(\d(?::\d?(?::\d?)?)?)
 |(?P<grp>\[[^\]\n]*\](?P<gn>\d*)(?P<gs>/*)(?P<gd>\d*))
 |(?P<note>(?P<acc>[\^_=]*)(?P<l>[A-Ga-gzZxX])(?P<oct>[,']*)(?P<n>\d*)(?P<s>/*)(?P<d>\d*))
 |(?P<broken>[<>]+)
 |(?P<tie>-)
 |(?P<paren>[()])
 |(?P<stacc>\.)
 |(?P<other>.)
""", re.X | re.M | re.S)

def _len_mult(n, s, d):
    m = Fraction(int(n)) if n else Fraction(1)
    if s:
        den = int(d) if d and int(d) else 2 ** len(s)
        m = m / den
    return m

def parse_abc(text, tune=0):
    """Devuelve dict: notes[(t,dur,midi,slur,stacc)], chords[(t,name)], starts, meter, qpm, fifths, mode_shift, total, title."""
    parts = re.split(r"(?m)^(?=X:)", text)
    parts = [p for p in parts if p.strip()]
    need(parts, "no hay música en el fichero ABC")
    src = parts[min(tune, len(parts) - 1)]
    lines = src.split("\n")
    hdr = {}
    k = None
    for i, ln in enumerate(lines):
        if re.match(r"^[A-Za-z]:", ln):
            hdr.setdefault(ln[0], ln[2:].strip())
            if ln.startswith("K:"):
                k = i
                break
    body = "\n".join(lines[k + 1:]) if k is not None else "\n".join(lines)
    meter = (4, 4)
    mm = re.match(r"(\d+)\s*/\s*(\d+)", hdr.get("M", ""))
    if mm and int(mm.group(1)) > 0 and int(mm.group(2)) > 0:
        meter = (int(mm.group(1)), int(mm.group(2)))
    elif hdr.get("M", "").strip() in ("C", "C|"):
        meter = (4, 4) if hdr["M"].strip() == "C" else (2, 2)
    bar_beats = 4.0 * meter[0] / meter[1]
    L = Fraction(1, 16) if meter[0] / meter[1] < 0.75 else Fraction(1, 8)
    lm = re.match(r"(\d+)\s*/\s*(\d+)", hdr.get("L", ""))
    if lm and int(lm.group(1)) > 0 and int(lm.group(2)) > 0:
        L = Fraction(int(lm.group(1)), int(lm.group(2)))
    qpm = None
    def set_tempo(q):
        m = re.match(r'\s*(?:"[^"]*"\s*)?(?:(\d+)\s*/\s*(\d+)\s*=\s*)?(\d+)', q)
        if m and int(m.group(3)) > 0:
            beat = Fraction(int(m.group(1)), int(m.group(2))) if m.group(1) and int(m.group(2)) else Fraction(1, 4)
            return float(m.group(3)) * float(beat) * 4
        return None
    if "Q" in hdr:
        qpm = set_tempo(hdr["Q"])
    fifths, shift = 0, 0
    pk = parse_key(hdr.get("K", "C")) if hdr.get("K", "C").strip().lower() not in ("none", "hp") else None
    if pk:
        fifths, shift = pk
    items = []
    state = {"L": L, "fifths": fifths}
    for m in _TOK.finditer(body):
        g = m.lastgroup
        if g in ("comment", "deco", "grace", "other", "stacc", "paren", "tie", "broken", "tup", "chord", "bar", "grp", "note", "field", "inline"):
            pass
        if g == "field" or g == "inline":
            f = m.group().strip("[]")
            if f[0] in "KML" and f[1] == ":":
                items.append(("field", f[0], f[2:].strip()))
            elif f[0] == "Q":
                items.append(("field", "Q", f[2:].strip()))
        elif g == "chord":
            items.append(("chord", m.group()[1:-1]))
        elif g == "bar":
            items.append(("bar", m.group()))
        elif g == "tup":
            nums = [int(x) if x else None for x in m.group()[1:].split(":")]
            p = nums[0]
            q = nums[1] if len(nums) > 1 and nums[1] else (3 if (meter[1] == 8 and meter[0] % 3 == 0 and meter[0] >= 6 and p in (5, 7, 9)) else _TUPQ.get(p, 2))
            r = nums[2] if len(nums) > 2 and nums[2] else p
            items.append(("tup", p, q, r))
        elif g == "grp":
            inner = re.findall(r"([\^_=]*)([A-Ga-g])([,']*)(\d*)(/*)(\d*)", m.group()[1:m.group().index("]")])
            if inner:
                mult = max(_len_mult(n, s_, d) for _, _, _, n, s_, d in inner) * _len_mult(m.group("gn"), m.group("gs"), m.group("gd"))
                items.append(("grp", [(a_, l_, o_) for a_, l_, o_, *_ in inner], mult))
        elif g == "note":
            items.append(("note", m.group("acc"), m.group("l"), m.group("oct"),
                          _len_mult(m.group("n"), m.group("s"), m.group("d"))))
        elif g == "broken":
            items.append(("broken", m.group()[0], len(m.group())))
        elif g == "tie":
            items.append(("tie",))
        elif g == "paren":
            items.append(("paren", m.group()))
        elif g == "stacc":
            items.append(("stacc",))
    # expandir repeticiones
    ex, i, start, used = [], 0, 0, set()
    while i < len(items):
        it = items[i]
        ex.append(it)
        if it[0] == "bar":
            if it[1] in (":|", "::", ":||") and i not in used:
                used.add(i)
                i = start
                continue
            if it[1] in ("|:", "::"):
                start = i + 1
        i += 1
    notes, chords, bar_times = [], [], []
    t = 0.0
    acc_bar = {}
    tup_n, tup_f = 0, 1.0
    broken_next, slur, stacc_next = None, 0, False
    tie_idx = None
    cur = {"L": L, "fifths": fifths}
    ksig = key_sig(fifths)
    unit_beats = lambda: float(cur["L"]) * 4
    for it in ex:
        k0 = it[0]
        if k0 == "field":
            if it[1] == "K":
                pk = parse_key(it[2])
                if pk:
                    ksig = key_sig(pk[0])
            elif it[1] == "L":
                mm2 = re.match(r"(\d+)\s*/\s*(\d+)", it[2])
                if mm2 and int(mm2.group(1)) > 0 and int(mm2.group(2)) > 0:
                    cur["L"] = Fraction(int(mm2.group(1)), int(mm2.group(2)))
            elif it[1] == "Q":
                q2 = set_tempo(it[2])
                if q2 and qpm is None:
                    qpm = q2
        elif k0 == "chord":
            chords.append((t, it[1]))
        elif k0 == "bar":
            if not bar_times or t - bar_times[-1] > 1e-9:
                bar_times.append(t)
            acc_bar = {}
        elif k0 == "tup":
            tup_n, tup_f = it[3], it[2] / it[1]
        elif k0 == "broken":
            broken_next = (it[1], it[2])
            if notes and notes[-1][0] >= 0:
                # modifica la nota ya escrita
                nt, nd, npch, nsl, nst = notes[-1]
                f = 1 + (1 - 0.5 ** it[2]) if it[1] == ">" else 0.5 ** it[2]
                notes[-1] = (nt, nd * f, npch, nsl, nst)
                t += nd * f - nd
        elif k0 == "tie":
            tie_idx = len(notes) - 1 if notes else None
        elif k0 == "paren":
            slur += 1 if it[1] == "(" else -1
            slur = max(0, slur)
        elif k0 == "stacc":
            stacc_next = True
        elif k0 in ("note", "grp"):
            if k0 == "note":
                _, acc, l, octm, mult = it
                lett = l.upper()
                rest = l in "zZxX"
                dur = float(mult) * unit_beats()
            else:
                mult = it[2]
                dur = float(mult) * unit_beats()
                rest = False
            if tup_n > 0:
                dur *= tup_f
                tup_n -= 1
            if broken_next is not None:
                dir_, n_ = broken_next
                dur *= (0.5 ** n_) if dir_ == ">" else (1 + (1 - 0.5 ** n_))
                broken_next = None
            if rest:
                t += dur
                continue
            pitches = []
            for (acc_s, l_, o_) in ([(acc, l, octm)] if k0 == "note" else it[1]):
                base = l_.upper()
                octv = (5 if l_.islower() else 4) + o_.count("'") - o_.count(",")
                if acc_s:
                    a_ = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}.get(acc_s, 0)
                    acc_bar[(base, octv)] = a_
                else:
                    a_ = acc_bar.get((base, octv), ksig.get(base, 0))
                pitches.append(12 * (octv + 1) + _BASE_PC[base] + a_)
            for pch in pitches:
                merged = False
                if tie_idx is not None and tie_idx < len(notes):
                    nt, nd, npch, nsl, nst = notes[tie_idx]
                    if npch == pch and abs(nt + nd - t) < 1e-6:
                        notes[tie_idx] = (nt, nd + dur, npch, nsl, nst)
                        merged = True
                if not merged:
                    notes.append((t, dur, pch, slur > 0, stacc_next))
            tie_idx = None
            stacc_next = False
            t += dur
    # inicio de compases (con anacrusa)
    starts = [0.0]
    if bar_times and 1e-6 < bar_times[0] < bar_beats - 1e-6:
        starts = [bar_times[0] - bar_beats]
    starts += bar_times
    if qpm is None:
        qpm = 105.0 if (meter[1] == 8 and meter[0] % 3 == 0 and meter[0] >= 6) else 112.0
    return {"notes": notes, "chords": chords, "starts": starts, "meter": meter, "bar_beats": bar_beats,
            "qpm": qpm, "fifths": fifths, "shift": shift, "total": t, "title": hdr.get("T", ""), "L": L}

def metric_weight(pos, meter):
    """Peso métrico (0..1) de una posición (en negras) dentro del compás."""
    n, d = meter
    eps = 1e-6
    def near(x, step):
        return abs(x / step - round(x / step)) < eps * 10
    if d == 8 and n % 3 == 0 and n >= 6:
        if near(pos, 1.5 * 1):
            idx = int(round(pos / 1.5))
            return 1.0 if idx == 0 else (0.8 if (n == 12 and idx == 2) else 0.65)
        return 0.45 if near(pos, 0.5) else 0.3
    beat = 4.0 / d
    if near(pos, beat):
        idx = int(round(pos / beat))
        if idx == 0:
            return 1.0
        if n == 4 and idx == 2:
            return 0.8
        return 0.62
    if near(pos, beat / 2):
        return 0.42
    return 0.3

class Warp:
    """Rubato: relación (tiempo de partitura en negras) → (tiempo real en negras a tempo constante)."""
    def __init__(self, total, phrase_beats, amount=1.0, rng=None):
        dt = 1 / 32
        n = int((total + 8) / dt) + 2
        ts = np.arange(n) * dt
        sp = np.ones(n)
        if amount > 0:
            sp += amount * 0.03 * np.sin(2 * np.pi * ts / max(phrase_beats, 1.0))
            ph = np.mod(ts, phrase_beats)
            fall = np.clip((ph - (phrase_beats - 1.5)) / 1.5, 0, 1)
            sp *= 1 - amount * 0.10 * fall * fall * (3 - 2 * fall)
            fin = np.clip((ts - (total - 6.0)) / 6.0, 0, 1)
            sp *= 1 - amount * 0.28 * fin * fin
        self.ts = ts
        self.real = np.concatenate([[0], np.cumsum(dt / sp[:-1])])

    def __call__(self, t):
        return float(np.interp(t, self.ts, self.real))

def auto_chords(piece, rate="auto"):
    """Acordes diatónicos por Viterbi sobre las notas de la melodía → [(t0, t1, pcs, bass_pc)]."""
    meter, bb = piece["meter"], piece["bar_beats"]
    starts = piece["starts"]
    total = max(piece["total"], bb)
    seg_len = bb / 2 if (bb >= 4 and meter[1] != 8) else bb
    ends = list(starts[1:]) + [starts[-1] + bb]
    segs = []
    for b0, b1 in zip(starts, ends):
        t0 = b0
        while t0 < min(b1, total) - 1e-6:
            segs.append((t0, min(t0 + seg_len, b1)))
            t0 += seg_len
    if not segs or segs[-1][1] < total - 1e-6:
        t0 = segs[-1][1] if segs else 0.0
        while t0 < total - 1e-6:
            segs.append((t0, min(t0 + seg_len, total)))
            t0 += seg_len
    tonic_pc = (piece["fifths"] * 7) % 12 if True else 0
    maj = [0, 2, 4, 5, 7, 9, 11]
    scale = [(tonic_pc + x) % 12 for x in maj]
    tdeg = _MODE_TONIC_DEG.get(piece["shift"], 0)
    tri = [[scale[i % 7], scale[(i + 2) % 7], scale[(i + 4) % 7]] for i in range(7)]
    hist = np.zeros((len(segs), 12))
    for (t, d, p, _, _) in piece["notes"]:
        for si, (a_, b_) in enumerate(segs):
            ov = min(t + d, b_) - max(t, a_)
            if ov > 1e-9:
                pos = t - max((x for x in starts if x <= t + 1e-9), default=0.0)
                hist[si, p % 12] += ov * (0.5 + metric_weight(pos, meter))
    S = 7
    em = np.zeros((len(segs), S))
    for si in range(len(segs)):
        tot = hist[si].sum()
        if tot <= 0:
            continue
        for c in range(S):
            r, th, fi = tri[c]
            em[si, c] = (hist[si, r] * 1.0 + hist[si, th] * 0.85 + hist[si, fi] * 0.8
                         - 0.5 * (tot - hist[si, r] - hist[si, th] - hist[si, fi])) / tot
    good = {(4, 0): .7, (3, 4): .6, (1, 4): .7, (0, 3): .5, (0, 4): .5, (5, 3): .4, (5, 1): .4,
            (3, 0): .4, (4, 5): .4, (0, 5): .35, (2, 5): .4, (6, 0): .5, (0, 1): .3, (3, 1): .3}
    def trans(a_, b_):
        ra, rb = (a_ - tdeg) % 7, (b_ - tdeg) % 7
        if ra == rb:
            return 0.35
        v = good.get((ra, rb), 0.0)
        if rb == 6:
            v -= 0.3
        return v
    T = np.array([[trans(a_, b_) for b_ in range(S)] for a_ in range(S)])
    sc = 2.0 * em[0] + np.array([0.5 if c == tdeg else 0 for c in range(S)])
    bp = np.zeros((len(segs), S), dtype=int)
    for si in range(1, len(segs)):
        cand = sc[:, None] + T
        bp[si] = cand.argmax(0)
        sc = cand.max(0) + 2.0 * em[si]
    sc = sc + np.array([0.5 if c == tdeg else 0 for c in range(S)])
    c = int(sc.argmax())
    path = [c]
    for si in range(len(segs) - 1, 0, -1):
        c = int(bp[si][c])
        path.append(c)
    path.reverse()
    return [(a_, b_, tri[c][:], tri[c][0]) for (a_, b_), c in zip(segs, path)]

def explicit_chords(piece):
    ch = []
    for (t, name) in piece["chords"]:
        r = chord_pcs(name)
        if r:
            ch.append((t, r))
    if not ch:
        return []
    out = []
    for i, (t, (pcs, bass)) in enumerate(ch):
        t1 = ch[i + 1][0] if i + 1 < len(ch) else max(piece["total"], t + 1)
        t0 = 0.0 if i == 0 else t
        if t1 > t0 + 1e-6:
            out.append((t0, t1, pcs, bass))
    return out

def _voice(pcs, prev, lo=52, hi=69):
    """Voicing en [lo,hi] con el menor movimiento respecto al anterior."""
    best, bc = None, 1e9
    n = len(pcs)
    for inv in range(n):
        rot = pcs[inv:] + pcs[:inv]
        v, base = [], lo
        for pc in rot:
            x = base + ((pc - base) % 12)
            v.append(x)
            base = x + 1
        if max(v) > hi + 3:
            continue
        cost = 0 if prev is None else sum(abs(a_ - b_) for a_, b_ in zip(sorted(v), sorted(prev)[:len(v)])) + 3 * abs(len(v) - len(prev))
        if cost < bc:
            best, bc = v, cost
    if best is None:
        best = [lo + ((pc - lo) % 12) for pc in pcs]
    return best

def _bass(pc, near=43):
    x = near - 6 + ((pc - (near - 6)) % 12)
    return x

def accomp_pattern(meter, choice):
    n, d = meter
    if choice != "auto":
        return choice
    if d == 8 and n % 3 == 0 and n >= 6:
        return "jig"
    if (n, d) == (3, 4):
        return "waltz"
    if (n, d) in ((2, 4), (4, 4), (2, 2)):
        return "oompah"
    return "block"

def build_events(piece, o, rng):
    """Eventos [(pista, canal, inicio_negras, dur_negras, nota, vel)] ya humanizados (tiempo de partitura)."""
    meter, bb, starts = piece["meter"], piece["bar_beats"], piece["starts"]
    import bisect
    def bar_of(t):
        i = bisect.bisect_right(starts, t + 1e-9) - 1
        return max(0, i), t - starts[max(0, i)]
    human = o["human"]
    nb = len(starts)
    pb = max(1, o["phrase_bars"])
    ev = []
    pitches = [n[2] for n in piece["notes"]]
    centre = float(np.mean(pitches)) if pitches else 72
    for (t, d, p, slur, stac) in piece["notes"]:
        bi, pos = bar_of(t)
        w = metric_weight(pos, meter)
        ph = ((bi % pb) + pos / bb) / pb
        v = 76 + 24 * (w - 0.5) + 9 * math.sin(math.pi * min(1.0, ph)) + (p - centre) * 0.25
        v += rng.normal(0, 2.5) * human
        v = 76 + (v - 76) * (0.35 + 0.65 * human) if human < 1 else v
        # nota final de compás/frase algo más suave
        if abs((t + d) - (starts[bi + 1] if bi + 1 < nb else piece["total"])) < 1e-6 and (bi % pb) == pb - 1:
            v -= 6
        v = int(max(28, min(122, round(v))))
        off = 0.0
        if human > 0:
            off += rng.normal(0, 0.010) * human + (0.008 * (1 - w)) * human
        sw = o["swing"]
        if sw > 0 and not (meter[1] == 8 and meter[0] % 3 == 0) and abs(pos % 1 - 0.5) < 1e-6 and d <= 0.5 + 1e-6:
            off += sw * (1 / 6)
        gate = 0.5 if stac else (1.02 if slur else 0.93)
        if d <= 0.26:
            gate = min(gate, 0.88)
        end_phrase = (bi % pb) == pb - 1 and abs((t + d) - (starts[bi + 1] if bi + 1 < nb else piece["total"])) < 1e-6
        if end_phrase:
            gate = max(gate, 0.97)
        dd = max(0.04, d * gate - (0.03 if not slur else -0.01))
        ev.append((1, 0, max(0.0, t + off), dd, p, v))
    # acompañamiento
    pat = accomp_pattern(meter, o["accomp"])
    tl = []
    if o["accomp"] != "none":
        tl = explicit_chords(piece) or (auto_chords(piece) if o["auto_chords"] else [])
    prev_v = None
    for (t0, t1, pcs, bass) in tl:
        voicing = _voice(pcs, prev_v)
        prev_v = voicing
        step = 0.5 if pat in ("jig", "arp") else 1.0
        pts = [t0]
        x = math.ceil(t0 / step - 1e-9) * step
        if abs(x - t0) < 1e-6:
            x += step
        while x < t1 - 1e-6:
            pts.append(x)
            x += step
        if pat == "block":
            pts = [t0]
        for j, pt in enumerate(pts):
            nxt = pts[j + 1] if j + 1 < len(pts) else t1
            bi, pos = bar_of(pt)
            w = metric_weight(pos, meter)
            dur = max(0.1, (nxt - pt) * 0.92)
            jit = rng.normal(0, 0.008) * human
            vel = int(max(25, min(100, 50 + 12 * w + rng.normal(0, 2) * human)))
            on_beat = abs(pos - round(pos)) < 1e-6
            idx = int(round(pos))
            if pat == "block":
                for nn in voicing:
                    ev.append((2, 1, max(0, pt + jit), (t1 - t0) * 0.96, nn, vel - 6))
                ev.append((3, 2, max(0, pt + jit), (t1 - t0) * 0.96, _bass(bass), vel + 4))
            elif pat == "oompah":
                if on_beat and idx % 2 == 0:
                    root = _bass(bass if (idx // 2) % 2 == 0 else pcs[min(2, len(pcs) - 1)])
                    ev.append((3, 2, max(0, pt + jit), dur, root, vel + 6))
                else:
                    for nn in voicing:
                        ev.append((2, 1, max(0, pt + jit), min(dur, 0.8), nn, vel - 4))
            elif pat == "waltz":
                if on_beat and idx == 0:
                    ev.append((3, 2, max(0, pt + jit), dur, _bass(bass), vel + 6))
                else:
                    for nn in voicing:
                        ev.append((2, 1, max(0, pt + jit), min(dur, 0.8), nn, vel - 4))
            elif pat == "jig":
                e = int(round(pos / 0.5))
                if e % 3 == 0:
                    ev.append((3, 2, max(0, pt + jit), 1.3, _bass(bass if (e // 3) % 2 == 0 else pcs[min(2, len(pcs) - 1)]), vel + 6))
                else:
                    for nn in voicing:
                        ev.append((2, 1, max(0, pt + jit), 0.42, nn, vel - 6))
            else:   # arp
                cyc = [voicing[0] - 12 if False else _bass(bass, 48)] + voicing
                ev.append((2, 1, max(0, pt + jit), dur, cyc[j % len(cyc)], vel))
    # batería simple
    if o["drums"]:
        t = starts[0]
        total = piece["total"]
        compound = meter[1] == 8 and meter[0] % 3 == 0 and meter[0] >= 6
        i = 0
        while t < total - 1e-6:
            bi, pos = bar_of(t)
            if compound:
                eps_ = int(round(pos / 0.5))
                if eps_ % 3 == 0:
                    ev.append((4, 9, t, 0.2, 36 if eps_ in (0,) else 42, 80 if eps_ == 0 else 62))
                else:
                    ev.append((4, 9, t, 0.15, 42, 48))
            else:
                idx = int(round(pos / 0.5))
                if idx % 2 == 0:
                    beat = idx // 2
                    kind = 36 if beat == 0 or (meter[0] == 4 and beat == 2) else 37
                    ev.append((4, 9, t, 0.2, kind, 84 if beat == 0 else 66))
                ev.append((4, 9, t, 0.15, 42, 56 if idx % 2 == 0 else 42))
            t += 0.5
    return ev

def _vlq(n):
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))

def write_smf(path_or_none, tracks, ppq=480):
    """tracks: lista de listas de (tick, orden, bytes). Devuelve los bytes SMF formato 1."""
    chunks = []
    for evs in tracks:
        evs = sorted(evs, key=lambda e: (e[0], e[1]))
        data, last = bytearray(), 0
        for tick, _, msg in evs:
            data += _vlq(tick - last) + msg
            last = tick
        data += _vlq(0) + b"\xff\x2f\x00"
        chunks.append(b"MTrk" + struct.pack(">I", len(data)) + bytes(data))
    blob = b"MThd" + struct.pack(">IHHH", 6, 1, len(tracks), ppq) + b"".join(chunks)
    if path_or_none:
        with open(path_or_none, "wb") as fh:
            fh.write(blob)
    return blob

def parse_midi(blob):
    """Lector mínimo de SMF (para pruebas): → (ppq, [pistas: [(tick, tipo, canal, a, b)]]). Tipos: on, off, meta."""
    assert blob[:4] == b"MThd"
    fmt, ntr, ppq = struct.unpack(">HHH", blob[8:14])
    pos, tracks = 14, []
    for _ in range(ntr):
        assert blob[pos:pos + 4] == b"MTrk"
        ln = struct.unpack(">I", blob[pos + 4:pos + 8])[0]
        data, i, tick, run, evs = blob[pos + 8:pos + 8 + ln], 0, 0, None, []
        while i < len(data):
            v = 0
            while True:
                b = data[i]; i += 1
                v = (v << 7) | (b & 0x7F)
                if not b & 0x80:
                    break
            tick += v
            st = data[i]
            if st == 0xFF:
                ty, ln2 = data[i + 1], data[i + 2]
                evs.append((tick, "meta", ty, bytes(data[i + 3:i + 3 + ln2]), None))
                i += 3 + ln2
                continue
            if st & 0x80:
                run = st; i += 1
            else:
                st = run
            hi, ch = st & 0xF0, st & 0x0F
            if hi in (0xC0, 0xD0):
                evs.append((tick, "pc", ch, data[i], None)); i += 1
            else:
                a_, b_ = data[i], data[i + 1]; i += 2
                if hi == 0x90 and b_ > 0:
                    evs.append((tick, "on", ch, a_, b_))
                elif hi in (0x80, 0x90):
                    evs.append((tick, "off", ch, a_, b_))
                else:
                    evs.append((tick, "ctl", ch, a_, b_))
        tracks.append(evs)
        pos += 8 + ln
    return ppq, tracks

def render_piece(text, o, seed=0, tune=0):
    """ABC → bytes SMF expresivo. o: dict con tempo, human, swing, accomp, auto_chords, drums, rubato, programs, phrase_bars."""
    piece = parse_abc(text, tune)
    need(piece["notes"], "el ABC no contiene notas que interpretar")
    rng = np.random.default_rng(seed)
    meter, bb = piece["meter"], piece["bar_beats"]
    qpm = o["tempo"] or piece["qpm"]
    need(10 <= qpm <= 400, "--tempo fuera de rango (10..400)")
    ev = build_events(piece, o, rng)
    total = max(e[2] + e[3] for e in ev)
    warp = Warp(total, o["phrase_bars"] * bb, o["rubato"] if o["human"] > 0 else 0.0, rng)
    ppq = 480
    micro = int(round(60e6 / qpm))
    tempo_trk = [(0, 0, b"\xff\x03" + bytes([len(piece["title"][:60].encode("ascii", "replace"))]) + piece["title"][:60].encode("ascii", "replace")),
                 (0, 1, b"\xff\x51\x03" + micro.to_bytes(3, "big")),
                 (0, 2, b"\xff\x58\x04" + bytes([meter[0] & 0xFF, int(math.log2(meter[1])) if meter[1] & (meter[1] - 1) == 0 else 2, 24, 8]))]
    fs = max(-7, min(7, piece["fifths"]))
    tempo_trk.append((0, 3, b"\xff\x59\x02" + struct.pack("b", fs) + bytes([1 if piece["shift"] in (-3, -2, -4, -5) else 0])))
    trk = {1: [], 2: [], 3: [], 4: []}
    progs = {1: (0, o["program"]), 2: (1, o["accomp_program"]), 3: (2, o["bass_program"])}
    for k, (ch, pg) in progs.items():
        trk[k].append((0, 0, bytes([0xC0 | ch, pg & 0x7F])))
    last_off = {}                                    # (pista, canal, nota) → (lista, índice del off)
    for (k, ch, t, d, nn, vel) in sorted(ev, key=lambda e: (e[0], e[2], e[4])):
        nn = int(max(0, min(127, nn)))
        t0 = int(round(warp(t) * ppq))
        t1 = max(int(round(warp(t + d) * ppq)), t0 + 6)
        key = (k, ch, nn)
        if key in last_off:                           # misma altura solapada: se recorta la anterior
            lst, idx, on_prev = last_off[key]
            if lst[idx][0] > t0 - 1:
                if t0 - 1 <= on_prev:                 # mismo instante: desplazar un pelo el nuevo inicio
                    t0 = on_prev + 2
                    t1 = max(t1, t0 + 6)
                lst[idx] = (t0 - 1, 1, lst[idx][2])
        trk[k].append((t0, 2, bytes([0x90 | ch, nn, int(max(1, min(127, vel)))])))
        trk[k].append((t1, 1, bytes([0x80 | ch, nn, 0])))
        last_off[key] = (trk[k], len(trk[k]) - 1, t0)
    tracks = [tempo_trk] + [trk[k] for k in (1, 2, 3, 4) if trk[k] and (k == 1 or len(trk[k]) > (1 if k in progs else 0))]
    return write_smf(None, tracks, ppq)

def _add_render_args(g):
    g.add_argument("--tempo", type=float, help="negras por minuto (por defecto el Q: del ABC o ~110)")
    g.add_argument("--human", type=float, default=1.0, help="cantidad de humanización 0..1 (dinámica, micro-tiempos, rubato)")
    g.add_argument("--swing", type=float, default=0.0, help="0..1: retrasa las corcheas a contratiempo")
    g.add_argument("--no-rubato", action="store_true")
    g.add_argument("--accomp", default="auto", choices=["auto", "none", "block", "oompah", "waltz", "jig", "arp"],
                   help="patrón de acompañamiento (auto = según el compás); usa los acordes del ABC")
    g.add_argument("--auto-chords", action="store_true", help="armonizar por Viterbi (triadas diatónicas) si el ABC no trae acordes")
    g.add_argument("--drums", action="store_true", help="batería sencilla")
    g.add_argument("--program", type=int, default=73, help="instrumento GM de la melodía (0-127; 73=flauta)")
    g.add_argument("--accomp-program", type=int, default=24, help="instrumento GM del acompañamiento (24=guitarra)")
    g.add_argument("--bass-program", type=int, default=32, help="instrumento GM del bajo")

def _render_opts(a):
    for nm in ("human", "swing"):
        v = getattr(a, nm, None)
    need(0 <= a.human <= 1, "--human debe estar entre 0 y 1")
    need(0 <= a.swing <= 1, "--swing debe estar entre 0 y 1")
    for nm in ("program", "accomp_program", "bass_program"):
        need(0 <= getattr(a, nm) <= 127, f"--{nm.replace('_', '-')} debe estar entre 0 y 127")
    return {"tempo": a.tempo, "human": a.human, "swing": a.swing, "rubato": 0.0 if a.no_rubato else 1.0,
            "accomp": a.accomp, "auto_chords": a.auto_chords or getattr(a, "expressive", False),
            "drums": a.drums, "program": a.program, "accomp_program": a.accomp_program,
            "bass_program": a.bass_program, "phrase_bars": getattr(a, "phrase_bars", 4) or 4}

def _render_text_to_midi(a, text):
    blob = render_piece(text, _render_opts(a), seed=(a.seed or 0))
    with open(a.midi, "wb") as fh:
        fh.write(blob)
    print(green(f"✓ MIDI expresivo ({len(blob):,} bytes) → {a.midi}"), file=sys.stderr)

def cmd_render(a):
    with open(a.input, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    a.midi = a.output
    blob = render_piece(text, _render_opts(a), seed=a.seed, tune=a.tune)
    with open(a.output, "wb") as fh:
        fh.write(blob)
    print(green(f"✓ {len(blob):,} bytes → {a.output}"))


# ═══════════════════════════════════════════════════════════════════════════
#  MÉTRICAS DE ESTRUCTURA, PUNTUACIÓN (rank / --best-of) Y PRESET --expressive
# ═══════════════════════════════════════════════════════════════════════════
def piece_metrics(text, tune=0):
    """Medidas estructurales de una pieza ABC (sobre la melodía interpretada, repeticiones ya expandidas)."""
    pc = parse_abc(text, tune)
    notes = [n for n in pc["notes"]]
    if len(notes) < 2:
        return None
    ps = [n[2] for n in notes]
    iv = [b - a for a, b in zip(ps, ps[1:])]
    absi = [abs(x) for x in iv]
    starts = pc["starts"] + [pc["starts"][-1] + pc["bar_beats"]]
    import bisect
    bars = {}
    for (t, d, p, _, _) in notes:
        b = max(0, bisect.bisect_right(starts, t + 1e-9) - 1)
        rel = round(t - starts[b], 3)
        bars.setdefault(b, []).append((rel, round(d, 3), p))
    seq = [tuple(bars[b]) for b in sorted(bars)]
    seen, rep = set(), 0
    for s in seq:
        if s in seen:
            rep += 1
        seen.add(s)
    rseq = [tuple((r, d) for r, d, _ in s) for s in seq]
    rseen, rrep = set(), 0
    for s in rseq:
        if s in rseen:
            rrep += 1
        rseen.add(s)
    dh = Counter(round(n[1], 3) for n in notes)
    covered = sum(n[1] for n in notes)
    # tónica según el modo
    maj = [0, 2, 4, 5, 7, 9, 11]
    tonic_pc = ((pc["fifths"] * 7) + maj[_MODE_TONIC_DEG.get(pc["shift"], 0)]) % 12
    dd = Counter(max(-7, min(7, round(x / 1.7))) for x in iv)       # intervalo aprox. en grados
    return {"notes": len(notes), "bars": len(seq), "step": sum(a <= 2 for a in absi) / len(absi),
            "leap": sum(a > 7 for a in absi) / len(absi), "range": max(ps) - min(ps),
            "bar_rep": rep / max(1, len(seq)), "rhythm_rep": rrep / max(1, len(seq)),
            "distinct_len": len(dh), "coverage": covered / max(pc["total"], 1e-9),
            "ends_tonic": ps[-1] % 12 == tonic_pc, "last_len": notes[-1][1], "int_hist": dd}

_SC_W = {"step": .20, "leap": .15, "range": .10, "repet": .15, "rhythm": .15, "ending": .15, "rests": .10}

def score_piece(text, rank_by="auto", pred_meta=None, tune=0):
    m = piece_metrics(text, tune)
    if m is None:
        return {"total": 0.0, "parts": {}}
    # intervalos: distancia de variación total con el corpus (o curva por defecto)
    ref = {}
    h = (pred_meta or {}).get("int_hist")
    for i in range(-7, 8):
        ref[i] = (h.get(str(i), 0) + 0.5) if h else _DEF_INT.get(abs(i), 0.08)
    z = sum(ref.values())
    tot = sum(m["int_hist"].values()) or 1
    tv = 0.5 * sum(abs(m["int_hist"].get(i, 0) / tot - ref[i] / z) for i in range(-7, 8))
    parts = {
        "step": max(0.0, 1 - tv * 1.5),
        "leap": 1 - min(1.0, m["leap"] / 0.15),
        "range": 1 - min(1.0, max(0.0, (m["range"] - 17) / 12)),
        "repet": max(0.0, 1 - abs(m["bar_rep"] - 0.4) / 0.4),
        "rhythm": 1.0 if 2 <= m["distinct_len"] <= 5 else (0.3 if m["distinct_len"] < 2 else max(0.0, 1 - (m["distinct_len"] - 5) / 6)),
        "ending": (1.0 if m["ends_tonic"] else 0.4) * (1.0 if m["last_len"] >= 1 - 1e-9 else 0.75),
        "rests": min(1.0, m["coverage"] / 0.9),
    }
    if rank_by in (None, "auto"):
        w = _SC_W
    else:
        w = {}
        for part in rank_by.split(","):
            nm, _, val = part.partition("=")
            need(nm in _SC_W, f"--rank-by: criterio desconocido {nm!r} (usa {', '.join(_SC_W)} o auto)")
            w[nm] = float(val) if val else 1.0
    tw = sum(w.values())
    return {"total": sum(parts[k] * v for k, v in w.items()) / tw, "parts": parts, "metrics": m}

def cmd_rank(a):
    meta = None
    if a.model:
        pred = load_predictor(a.model, a.backend)
        meta = getattr(pred, "meta", None)
    rows = []
    for f in a.inputs:
        with open(f, "r", encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        tunes = [t for t in re.split(r"(?m)^(?=X:)", txt) if t.strip()]
        for i, t in enumerate(tunes):
            title = (re.search(r"(?m)^T:(.*)$", t) or [None, os.path.basename(f)])[1].strip()
            try:
                sc = score_piece(t, a.rank_by, meta)
            except SystemExit:
                continue
            if not sc["parts"]:
                continue
            bpc = None
            if a.model:
                p2 = load_predictor(a.model, a.backend)
                body = t.encode("ascii", errors="replace")
                bpc = float(score_bits(body, p2, False).mean()) if body else None
            rows.append((sc["total"], f"{os.path.basename(f)}#{i + 1}", title, sc, bpc))
    need(rows, "no se pudo puntuar ninguna pieza")
    rows.sort(key=lambda r: -r[0])
    print(bold(f"{'puntos':>7}  {'pieza':28s} " + " ".join(f"{k[:5]:>5}" for k in _SC_W) + ("   bpc" if a.model else "")))
    for tot, name, title, sc, bpc in rows[:a.top]:
        print(f"{tot:7.3f}  {(name + ' ' + title)[:28]:28s} " + " ".join(f"{sc['parts'][k]:5.2f}" for k in _SC_W)
              + (f"  {bpc:5.2f}" if bpc is not None else ""))
    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("pieza,titulo,total," + ",".join(_SC_W) + "\n")
            for tot, name, title, sc, bpc in rows:
                fh.write(f'{name},"{title}",{tot:.4f},' + ",".join(f"{sc['parts'][k]:.4f}" for k in _SC_W) + "\n")

def _apply_expressive(a):
    """Preset: lo que más ayudó en las pruebas de 'planitud'."""
    a.constrain = True
    a.chords = True
    if a.prior_strength <= 0:
        a.prior_strength = 1.0
    if a.form in (None, "none"):
        a.form = "copy"
    if a.curve in (None, "none"):
        a.curve = "arch"
    a.best_of = max(a.best_of, 4)

# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Predictor/compresor de música ABC (predictor ≡ compresor).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_model(p, need=True):
        p.add_argument("--model", required=need, help="modelo (.pkl ngram / .pt torch)")
        p.add_argument("--backend", choices=list(BACKENDS), default=None)

    p = sub.add_parser("synth", help="corpus ABC sintético")
    p.add_argument("-o", "--output", required=True); p.add_argument("-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("prepare", help="unir y limpiar .abc")
    p.add_argument("inputs", nargs="+"); p.add_argument("-o", "--output", required=True)
    p.add_argument("--minimal", action="store_true", help="solo M: L: K: Q: R: + cuerpo")
    p.add_argument("--keep", default="", metavar="LETRAS", help="con --minimal, campos extra a conservar (p. ej. CTZ)")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="entrenar")
    p.add_argument("corpus"); p.add_argument("-o", "--output", required=True)
    p.add_argument("--backend", choices=list(BACKENDS), default="ngram")
    p.add_argument("--order", type=int, default=4); p.add_argument("--minlen", type=int, default=6)
    p.add_argument("--gamma", type=float, default=2.0, help="suavizado Witten-Bell (mayor = más prudente)")
    p.add_argument("--emb", type=int, default=64); p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=2); p.add_argument("--seq", type=int, default=256)
    p.add_argument("--batch", type=int, default=32); p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--device", default="auto"); p.set_defaults(fn=cmd_train)

    p = sub.add_parser("sweep", help="barrido de hiperparámetros ngram (train vs test)")
    p.add_argument("train"); p.add_argument("test")
    p.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--gammas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--minlens", type=int, nargs="+", default=[6])
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("bpc", help="bits por carácter"); add_model(p)
    p.add_argument("data"); p.set_defaults(fn=cmd_bpc)

    p = sub.add_parser("compress", help="comprimir"); add_model(p)
    p.add_argument("input"); p.add_argument("-o", "--output")
    p.add_argument("--static", action="store_true", help="no aprender durante la compresión")
    p.set_defaults(fn=cmd_compress)

    p = sub.add_parser("decompress", help="descomprimir"); add_model(p)
    p.add_argument("input"); p.add_argument("-o", "--output"); p.set_defaults(fn=cmd_decompress)

    p = sub.add_parser("surprise", help="sorpresa por carácter/compás"); add_model(p)
    p.add_argument("data"); p.add_argument("--top", type=int, default=8)
    p.add_argument("--csv"); p.add_argument("--heatmap", type=int, default=0, metavar="N")
    p.add_argument("--static", action="store_true"); p.set_defaults(fn=cmd_surprise)

    p = sub.add_parser("generate", help="muestrear"); add_model(p)
    p.add_argument("--chars", type=int, default=400); p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--prime", help="texto o fichero ABC con el que arrancar")
    p.add_argument("--seed", type=int, default=None); p.add_argument("-o", "--output")
    p.add_argument("--constrain", action="store_true",
                   help="solo caracteres que mantengan el ABC válido (duración exacta por compás, ligaduras...)")
    p.add_argument("--meter", help="compás si el prime no trae cabecera (p. ej. 6/8)")
    p.add_argument("--unit", help="L: si el prime no trae cabecera (p. ej. 1/8)")
    p.add_argument("--key", default="G", help="K: si el prime no trae cabecera")
    p.add_argument("--title", default=None, help="T: de la salida con --constrain (por defecto el del prime)")
    g = p.add_argument_group("v2: estructura y expresividad (implican --constrain)")
    g.add_argument("--chords", action="store_true", help='permitir símbolos de acorde "Am7" (gramática cerrada)')
    g.add_argument("--rhythms", help='paleta rítmica en unidades de L: ("1/2,1,2,4") o auto:N (las N más frecuentes del corpus); sin tresillos ni ritmo roto')
    g.add_argument("--prior-strength", type=float, default=0.0, metavar="S",
                   help="prior suave de intervalos/contorno/registro (0=off, 1=normal)")
    g.add_argument("--form", default="none", help="none | copy | patrón de secciones (AABA) — copia compases con variación")
    g.add_argument("--p-copy", type=float, default=0.35); g.add_argument("--vary", type=float, default=0.5)
    g.add_argument("--phrase-bars", type=int, default=4)
    g.add_argument("--curve", default="none", help="curva de sorpresa objetivo: none|flat|arch|rise|a,b,c...")
    g.add_argument("--curve-gain", type=float, default=0.04)
    g.add_argument("--best-of", type=int, default=1, metavar="N", help="generar N y quedarse con la mejor puntuada")
    g.add_argument("--rank-by", default="auto", help="criterio de --best-of (ver 'rank --help')")
    g.add_argument("--style", help="reel, jig, waltz... (inserta R: y fija el compás por defecto)")
    g.add_argument("--report", action="store_true", help="informe de masa rechazada por la restricción (stderr)")
    g.add_argument("--expressive", action="store_true",
                   help="preset: acordes + priores + forma + curva arch + best-of 4 (+ MIDI expresivo si hay --midi)")
    g.add_argument("--midi", metavar="FILE", help="además, interpretar el ABC a MIDI expresivo")
    g3 = p.add_argument_group("v3: gramática (implica --constrain)")
    _add_grammar_args(g3)
    _add_render_args(g)
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("render", help="ABC → MIDI expresivo (humanización + acompañamiento)")
    p.add_argument("input"); p.add_argument("-o", "--output", required=True)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--tune", type=int, default=0, help="pieza (0-based) si el fichero tiene varias")
    p.add_argument("--phrase-bars", type=int, default=4)
    _add_render_args(p); p.set_defaults(fn=cmd_render, expressive=False)

    p = sub.add_parser("rank", help="puntuar/ordenar piezas ABC por estructura (y bpc con --model)")
    p.add_argument("inputs", nargs="+"); p.add_argument("--model"); p.add_argument("--backend", choices=list(BACKENDS), default=None)
    p.add_argument("--rank-by", default="auto", help="auto o lista de criterios: step,leap,range,repet,rhythm,ending,rests (con pesos a=2)")
    p.add_argument("--top", type=int, default=20); p.add_argument("--csv"); p.set_defaults(fn=cmd_rank)

    p = sub.add_parser("info", help="datos del modelo"); add_model(p); p.set_defaults(fn=cmd_info)

    p = sub.add_parser("grammar", help="estadísticas de la gramática de un corpus ABC")
    p.add_argument("corpus"); p.add_argument("--key", help="partición por modo (p. ej. G, Am, D dor); por defecto la mayor")
    p.add_argument("--meter", help="compás de la partición (p. ej. 6/8)"); p.add_argument("--top", type=int, default=8)
    p.add_argument("--grammar-phrase-bars", type=int, default=4)
    p.add_argument("--max-elem", type=int, default=8); p.add_argument("--grammar-inventory", choices=("repair", "ngram", "bars"), default="repair")
    p.add_argument("--grammar-align", action="store_true"); p.set_defaults(fn=cmd_grammar)

    a = ap.parse_args()
    # utilidad: nombre del backend de un predictor ya cargado
    a.backend_name = lambda pr: pr.name
    try:
        a.fn(a)
    except BrokenPipeError:                      # p. ej. "| head"
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except Exception:
            pass
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
    except OSError as e:
        die(f"{e.strerror or e}: {e.filename}" if e.filename else str(e))
    except ValueError as e:
        die(str(e))

if __name__ == "__main__":
    main()
