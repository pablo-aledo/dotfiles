#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  abc_predictor.py — Predictor de música ABC + compresor aritmético       ║
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
import os
import pickle
import random
import re
import struct
import sys
import zlib
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

def clean_tune(t, minimal):
    out = []
    for line in t.splitlines():
        s = line.rstrip()
        if not s or s.startswith("%"):
            continue
        if minimal and len(s) > 1 and s[1] == ":" and s[0].isalpha() and s[0] not in "MLKQ":
            continue                      # solo M: L: K: Q: + cuerpo
        out.append(s)
    return "\n".join(out) + "\n"

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
    out = "".join(clean_tune(t, a.minimal) for t in tunes)
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
            self.hashidx[key] = L

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
        if name == "NgramPredictor" and module in ("__main__", __name__, "abc_predictor"):
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
_CAND = "ABCDEFGabcdefgz^_=,'0123456789/<>-()|:] \n.~"

class AbcState:
    """Autómata del cuerpo ABC. step(ch) devuelve True si ch es válido y avanza
    el estado; si devuelve False el estado queda indefinido (probar sobre clone())."""

    def __init__(self, bar_len, unit):
        self.bar_len, self.unit = bar_len, unit
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
        if self.tup_rem > 0:
            return self.tup_rem * self.unit * self.tup_fac * self.next_factor
        return self.unit * self.next_factor / 4

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
        return self.n_base * f

    def _could_fit(self):
        return self.n_start + self._min_dur() + self._reserve() <= self.bar_len

    def _fits(self):
        return self.n_start + self._dur() + self._reserve() <= self.bar_len

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
        if self.tup_rem == 0 and (self.bar_len - newpos) % (self.unit / 4) != 0:
            return False
        self.pos = newpos
        if self.broken == ">":
            self.next_factor = Fraction(1, 2)
        elif self.broken == "<":
            self.next_factor = Fraction(3, 2)
        self.tie = None
        if not self.is_rest:
            self.npitch += 1
        self.in_note, self.phase, self.pitch, self.is_rest = False, None, "", False
        self.num, self.den, self.slashes, self.has_slash, self.broken = "", "", 0, False, None
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
        self.bt, self.bt_zero, self.bt_partial = "", False, False

    def _step(self, ch):
        if self.bt:
            if ch in "|:]" and (self.bt + ch) in _TOK_PREFIXES:
                nbt = self.bt + ch
                if self.bt_zero and nbt != "|:":
                    return False
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
        if ch == "(":
            if self.paren or self.need_note or self.tie is not None:
                return False
            if not self._commit() or self.pos + self._min_next() > self.bar_len:
                return False
            if self.slur > 0:                 # con ligadura abierta, "(" solo puede ser un tresillo
                if self.tup_rem > 0 or self.next_factor != 1 or self.pos + 2 * self.unit > self.bar_len:
                    return False
                self.paren = "tup"
            else:
                self.paren = True
            return True
        if ch.isdigit() and self.paren:
            if ch not in _TUP_Q or self.tup_rem > 0:
                return False
            if self.next_factor != 1:
                return False
            tf = Fraction(_TUP_Q[ch], int(ch))
            if self.pos + int(ch) * self.unit * tf * self.next_factor > self.bar_len:
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
            if self.paren or self.need_note or self.tie is not None:
                return False
            if not self._commit() or self.pos + self._min_next() > self.bar_len:
                return False
            self.need_note = True
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
        if not self._commit() or self.pos + self._min_next() > self.bar_len:
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
            else:
                nn = self.num + ch
                if nn.startswith("0") or int(nn) > 8:
                    return False
                self.num = nn
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
            return self._could_fit()
        if ch in "<>":
            if self.in_tup or self.has_slash:     # sin subdivisiones: se saldría de la rejilla
                return False
            self.broken = ch
            return self._could_fit()
        if ch == "-":
            if self.is_rest:
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
            self.bt, self.bt_zero, self.bt_partial = "|", True, False
            return True
        full = pos == bl
        comp = self.pickup is not None and pos == bl - self.pickup and not full
        pick = self.pickup is not None and self.sec_first and pos == self.pickup and not full
        first_partial = self.first_bar and pos < bl
        if ch == ":":                      # fin de repetición: compás completo o complemento
            if not (full or comp):
                return False
            partial = False
        elif full or first_partial or pick:
            partial = False
        elif comp:                         # compás de cierre de sección: exige || o |]
            partial = True
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
                or self.next_factor != 1):
            return False
        if self.bt and (self.bt not in _TOKENS or (self.bt_partial and self.bt not in ("||", "|]"))
                        or (self.bt_zero and self.bt != "|:")):
            return False
        if self.in_note:
            if self.phase != "post" or self.broken is not None:
                return False
            if not self._fits():
                return False
            end = self.n_start + self._dur()
        else:
            end = self.pos
        ok = {self.bar_len, Fraction(0)}
        if self.pickup is not None:
            ok.add(self.bar_len - self.pickup)
        return end in ok


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

def _generate_constrained(a, pred):
    dead_end[0] = 0
    header, body_prime, bar_len, unit, ptitle = _constrained_header(a)
    rng = np.random.default_rng(a.seed)
    for s in (header + body_prime).encode("ascii", errors="replace"):
        pred.update(s, False)
    st = AbcState(bar_len, unit)
    for i, c in enumerate(body_prime):
        if not st.step(c):
            die(f"--prime incompatible con --constrain en el carácter {i} ({c!r})")
    out = list(body_prime)
    cand = [(c, ord(c)) for c in _CAND]
    extra = 0
    safe = None
    while True:
        if len(out) - len(body_prime) >= a.chars:
            if (safe is not None and safe[0] == len(out)) or extra >= 250:
                break
            extra += 1
        allowed = [o for c, o in cand if st.clone().step(c)]
        if not allowed:
            dead_end[0] += 1
            break
        p = pred.probs()
        if a.temp != 1.0:
            lp = np.log(np.maximum(p, 1e-300)) / a.temp
            p = np.exp(lp - lp.max())
        q = np.zeros(256)
        q[allowed] = p[allowed]
        if q.sum() < 1e-12 * max(p.sum(), 1e-300):
            q[allowed] = 1.0
        q /= q.sum()
        o = int(rng.choice(256, p=q))
        st.step(chr(o))
        out.append(chr(o))
        pred.update(o, False)
        if st.at_boundary() and st.npitch > 0:
            safe = (len(out), st.clone())
    if safe is None:
        die(f"no se pudo generar ninguna pieza válida con M:{bar_len} L:{unit} "
            f"(prueba otra --seed, otro --chars o un L: más pequeño)")
    out, st = out[:safe[0]], safe[1]
    body = "".join(out)
    if st.slur > 0:                           # cerrar ligaduras abiertas tras la última nota
        m = re.search(r"[\s|:\]]*$", body)
        body = body[:m.start()] + ")" * st.slur + body[m.start():]
    body = _finish_text(body, st)
    text = f"X:1\nT:{a.title or ptitle or 'Generada'}\n{header}{body}\n"
    if a.output:
        with open(a.output, "w") as fh:
            fh.write(text)
        print(green(f"✓ {len(body)} caracteres de cuerpo ({bar_len} por compás, L={unit}) → {a.output}"
                    + (" · terminó en un callejón, truncado al último punto seguro" if dead_end[0] else "")))
    else:
        print(text, end="")

def _prime_text(p):
    """El --prime es un fichero si existe, y si no, el propio texto."""
    if p and "\n" not in p and len(p) < 4096 and "\x00" not in p and os.path.isfile(p):
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return p or ""

def cmd_generate(a):
    need(a.temp > 0, "--temp debe ser > 0")
    a.chars = max(0, a.chars)
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
    p.add_argument("--minimal", action="store_true", help="solo M: L: K: Q: + cuerpo")
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
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("info", help="datos del modelo"); add_model(p); p.set_defaults(fn=cmd_info)

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
