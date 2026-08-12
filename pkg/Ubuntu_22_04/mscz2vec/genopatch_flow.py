#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        GENOPATCH_FLOW  v0.2                                  ║
║  Inferencia amortizada de parámetros de síntesis vía flujos normalizadores   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUÉ HACE                                                                    ║
║    Alternativa a la búsqueda evolutiva de genopatch_v2.py: en vez de         ║
║    resolver el problema inverso (sonido → parámetros) desde cero en cada     ║
║    `match`, entrena UNA VEZ un flujo condicional p(θ|x) sobre pares          ║
║    sintéticos (θ~prior, x=render(θ)), y luego la inferencia es un solo       ║
║    forward pass — sin bucle evolutivo. Es "simulation-based inference"       ║
║    (SBI/neural posterior estimation) aplicado al simulador que ya es         ║
║    genopatch_v2.ENGINES.                                                     ║
║                                                                              ║
║    A diferencia del GA, que colapsa a UN punto arbitrario cuando el          ║
║    problema no es identificable (ver informe de refit: add_bright_brass,     ║
║    fundamental casi muda pero buen fitness), el flujo modela la              ║
║    distribución p(θ|x) completa — puede representar la cresta de             ║
║    soluciones equivalentes en vez de esconderla. Confirmado empíricamente    ║
║    en additive: 6 muestras del mismo objetivo recuperan la misma forma       ║
║    armónica normalizada con escala bruta libre, en vez de colapsar a una     ║
║    forma incorrecta como hacía el GA.                                        ║
║                                                                              ║
║  ARQUITECTURA                                                                ║
║    RealNVP condicional: una red pequeña ("embedder") reduce el vector de     ║
║    features de audio (~2052 dims, el mismo que usa --fitness spectral en     ║
║    genopatch_v2) a un embedding compacto (64 dims); 6 capas de               ║
║    acoplamiento afín (checkerboard mask) transforman θ normalizado           ║
║    (min-max a [-1,1] por parámetro) ↔ ruido base N(0,I), condicionadas en    ║
║    ese embedding. Entrenamiento por máxima verosimilitud (NLL), CPU-only,    ║
║    red pequeña (~450K parámetros) — nada que necesite GPU.                   ║
║                                                                              ║
║  ALCANCE — los 6 motores base de genopatch_v2.py (RAW_ENGINES)               ║
║    Entrenado SOLO sobre los parámetros base de cada motor (sin el wrapper    ║
║    de FX vibrato/unison/drive de genopatch_v2) — el patch resultante trae    ║
║    esos parámetros en su valor neutro (sin efecto). Entrenado a NOTA y       ║
║    DURACIÓN fijas por motor (ver CANONICAL / `info` del checkpoint):         ║
║                                                                              ║
║    karplus       nota 60, 1.5s  — calidad excelente (3 parámetros)           ║
║    additive      nota 60, 1.2s  — calidad excelente (mapeo casi lineal)      ║
║    wavetable     nota 60, 1.5s  — buena, el único caso donde el GA es        ║
║                  realmente competitivo con el flujo                          ║
║    noise         nota 60, 0.6s  — aceptable, alta varianza esperable         ║
║    subtractive   nota 60, 1.5s  — aceptable, ni flujo ni GA brillan          ║
║    fm2           nota 60, 1.5s  — NOTABLEMENTE PEOR que el resto —           ║
║                  paisaje θ→espectro no lineal, 6000 pares no bastan.         ║
║                  Usar con expectativas bajas, o como semilla del GA (el      ║
║                  híbrido ayuda algo pero no supera a la mejor muestra del    ║
║                  flujo sola — ver informe de testing).                       ║
║                                                                              ║
║    layered_fm_add NO tiene flujo propio — pendiente de una composición       ║
║    explícita desde los flujos de fm2+additive (fuera de alcance).            ║
║                                                                              ║
║  COLAPSO DE MODO SILENCIOSO — nota distinta a la canónica                    ║
║    Si el WAV objetivo tiene una nota distinta a la del checkpoint, el        ║
║    flujo puede fallar SIN avisar: en vez de degradar con gracia, todas       ║
║    las muestras colapsan al mismo punto exacto en los límites del rango      ║
║    (confirmado en karplus: la nota determina la longitud de su línea de      ║
║    retardo, así que cambia la estructura del motor por completo). `sample`   ║
║    detecta el patrón (muestras casi idénticas + pegadas a los límites) y     ║
║    avisa — pero NO es universal: en subtractive, la misma nota incorrecta    ║
║    degrada con diversidad real, sin disparar el aviso. Su silencio no es     ║
║    garantía de que el resultado sea bueno.                                   ║
║                                                                              ║
║  SUBCOMANDOS                                                                 ║
║    train    genera N pares sintéticos (prior de genopatch_v2._random_params, ║
║             mismo sesgo hacia el default que usa el GA) y entrena el flujo   ║
║    sample   WAV objetivo + checkpoint → N muestras de p(θ|x), como           ║
║             patch.json (mismo formato que genopatch_v2, cero fricción con    ║
║             `render`/`mutate`/`info` de ese script). Avisa si detecta        ║
║             colapso de modo.                                                 ║
║    info     inspecciona un checkpoint (motor, nota/duración de               ║
║             entrenamiento, dimensiones, nº de pares usados, val_nll)         ║
║                                                                              ║
║  USO                                                                         ║
║    genopatch_flow.py train --engine additive --n-samples 3200 \\             ║
║                       --epochs 150 --out flow_additive.pt                    ║
║    genopatch_flow.py sample target.wav --checkpoint flow_additive.pt \\      ║
║                       -n 8 --render --out-dir samples/                       ║
║    genopatch_flow.py sample target.wav --checkpoint flow_fm2.pt -n 20 \\     ║
║                       --out-dir samples/   # fm2: pide más muestras y        ║
║                                            # quédate con la mejor            ║
║    genopatch_flow.py info flow_additive.pt                                   ║
║                                                                              ║
║    # patch.json del flujo → seguir explorando con genopatch_v2.py:           ║
║    genopatch_v2.py render samples/sample_01.json --note E4 --out e4.wav      ║
║    genopatch_v2.py mutate samples/sample_01.json -n 8 --render \\            ║
║                    --out-dir variantes/                                      ║
║                                                                              ║
║  FORMATO patch.json — idéntico al de genopatch_v2.py, solo parámetros base   ║
║    {"engine": "additive", "note": 60, "duration": 1.2, "velocity": 0.85,     ║
║     "params": {"amp_h1": 1.0, "amp_h2": 0.62, ...}}                          ║
║                                                                              ║
║  DEPENDENCIAS  numpy  scipy  soundfile  torch (CPU)  genopatch_v2.py         ║
║                (debe estar en el mismo directorio o PYTHONPATH)              ║
║                                                                              ║
║  FINGERPRINT DE FEATURES                                                     ║
║    El checkpoint guarda _FEAT_N_FFT/_FEAT_HOP/pesos de vibrato de            ║
║    genopatch_v2 vigentes al entrenar. `sample` verifica que coincidan con    ║
║    los actuales antes de usar el checkpoint — si genopatch_v2.py cambia      ║
║    esos parámetros (ya ha pasado una vez, 1024→2048), un checkpoint viejo    ║
║    falla alto y claro en vez de dar muestras basura calladamente.            ║
║                                                                              ║
║  LIMITACIONES                                                                ║
║    · Nota/duración de entrenamiento fijas por motor — usar `sample` con      ║
║      un WAV de otra nota puede colapsar silenciosamente (ver arriba); con    ║
║      otra duración degrada más suave pero tampoco es gratis.                 ║
║    · No modela el wrapper de FX (vibrato/unison/drive) — los patches         ║
║      salen con esos parámetros en su default neutro.                         ║
║    · fm2 es notablemente peor que el resto — no lo trates como sustituto     ║
║      fiable de `genopatch_v2.py match` con ese motor todavía.                ║
║    · layered_fm_add no tiene checkpoint — no está en SUPPORTED_ENGINES.      ║
║    · El aviso de colapso de modo es una heurística (varianza casi nula +     ║
║      valor en el límite) — puede haber falsos negativos (colapso a un        ║
║      punto NO extremo) o falsos positivos (posterior genuinamente muy        ║
║      concentrada, p.ej. karplus con nota correcta y target limpio).          ║
║                                                                              ║
║  Módulo importable:                                                          ║
║    from genopatch_flow import train, train_from_dataset, \\                  ║
║                              sample_from_target, generate_dataset            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""



import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _import_genopatch_v2():
    try:
        import genopatch_v2 as gp
        return gp
    except ImportError:
        sys.path.insert(0, str(_script_dir()))
        try:
            import genopatch_v2 as gp
            return gp
        except ImportError:
            sys.exit("genopatch_v2.py no encontrado -- debe estar en el mismo "
                      "directorio o en PYTHONPATH.")


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        sys.exit("torch no encontrado. Instala con: pip install torch --break-system-packages")


def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("soundfile no encontrado. Instala con: pip install soundfile")


SUPPORTED_ENGINES = ("additive", "karplus", "subtractive", "wavetable", "noise", "fm2")

CANONICAL = {
    "additive":    {"note": 60, "duration": 1.2, "velocity": 0.85},
    "karplus":     {"note": 60, "duration": 1.5, "velocity": 0.85},
    "subtractive": {"note": 60, "duration": 1.5, "velocity": 0.85},
    "wavetable":   {"note": 60, "duration": 1.5, "velocity": 0.85},
    "noise":       {"note": 60, "duration": 0.6, "velocity": 0.85},
    "fm2":         {"note": 60, "duration": 1.5, "velocity": 0.85},
}


# =============================================================================
# section 1: synthetic dataset
# =============================================================================

def generate_dataset(gp, engine_name: str, n_samples: int, seed: int = 0
                       ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Genera n_samples pares (theta, x) para el motor RAW (sin FX) dado.
    theta se muestrea con el mismo prior sesgado que usa el GA
    (_random_params); x es el vector de features de
    _extract_internal_features sobre el audio renderizado a la
    nota/duracion canonicas del motor."""
    import random as _random
    engine = gp.RAW_ENGINES[engine_name]
    canon = CANONICAL[engine_name]
    specs = engine.params
    param_names = [p.name for p in specs]

    _random.seed(seed)
    np.random.seed(seed)

    thetas = np.zeros((n_samples, len(specs)), dtype=np.float64)
    feats = None
    for i in range(n_samples):
        params = gp._clip_params(gp._random_params(specs), specs)
        audio = engine.render(params, canon["note"], canon["duration"], canon["velocity"], 44100)
        feat = gp._extract_internal_features(audio, 44100)
        if feats is None:
            feats = np.zeros((n_samples, len(feat)), dtype=np.float64)
        feats[i] = feat
        thetas[i] = [params[name] for name in param_names]
    return thetas, feats, param_names


# =============================================================================
# section 2: conditional flow (RealNVP)
# =============================================================================

def _build_model(torch, nn, theta_dim: int, feat_dim: int,
                  embed_hidden: int = 128, embed_out: int = 64,
                  coupling_hidden: int = 128, n_coupling: int = 6):

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(feat_dim, embed_hidden), nn.ReLU(),
                nn.Linear(embed_hidden, embed_out), nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    class AffineCoupling(nn.Module):
        def __init__(self, dim, cond_dim, mask):
            super().__init__()
            self.register_buffer("mask", mask)
            self.net = nn.Sequential(
                nn.Linear(dim + cond_dim, coupling_hidden), nn.ReLU(),
                nn.Linear(coupling_hidden, coupling_hidden), nn.ReLU(),
                nn.Linear(coupling_hidden, dim * 2),
            )

        def forward(self, theta, cond):
            theta_masked = theta * self.mask
            st = self.net(torch.cat([theta_masked, cond], dim=-1))
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s) * (1 - self.mask)
            t = t * (1 - self.mask)
            theta_out = theta_masked + (1 - self.mask) * (theta * torch.exp(s) + t)
            log_det = s.sum(dim=-1)
            return theta_out, log_det

        def inverse(self, z, cond):
            z_masked = z * self.mask
            st = self.net(torch.cat([z_masked, cond], dim=-1))
            s, t = st.chunk(2, dim=-1)
            s = torch.tanh(s) * (1 - self.mask)
            t = t * (1 - self.mask)
            theta_out = z_masked + (1 - self.mask) * ((z - t) * torch.exp(-s))
            return theta_out

    class ConditionalFlow(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedder = Embedder()
            layers = []
            for i in range(n_coupling):
                mask = torch.zeros(theta_dim)
                mask[i % 2::2] = 1.0
                layers.append(AffineCoupling(theta_dim, embed_out, mask))
            self.layers = nn.ModuleList(layers)
            self.theta_dim = theta_dim

        def log_prob(self, theta, x):
            cond = self.embedder(x)
            z = theta
            log_det_total = torch.zeros(theta.shape[0], device=theta.device)
            for layer in self.layers:
                z, log_det = layer(z, cond)
                log_det_total = log_det_total + log_det
            base_log_prob = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.theta_dim * np.log(2 * np.pi)
            return base_log_prob + log_det_total

        @torch.no_grad()
        def sample(self, n, x):
            cond = self.embedder(x.expand(n, -1) if x.dim() == 1 else x)
            z = torch.randn(n, self.theta_dim)
            theta = z
            for layer in reversed(self.layers):
                theta = layer.inverse(theta, cond)
            return theta

    return ConditionalFlow()


# =============================================================================
# section 3: theta/x normalization
# =============================================================================

def _normalize_theta(thetas: np.ndarray, specs) -> np.ndarray:
    lo = np.array([p.lo for p in specs])
    hi = np.array([p.hi for p in specs])
    return 2.0 * (thetas - lo) / (hi - lo) - 1.0


def _denormalize_theta(theta_norm: np.ndarray, specs) -> np.ndarray:
    lo = np.array([p.lo for p in specs])
    hi = np.array([p.hi for p in specs])
    return lo + (theta_norm + 1.0) / 2.0 * (hi - lo)


def _feature_fingerprint(gp) -> dict:
    return {
        "feat_n_fft": gp._FEAT_N_FFT,
        "feat_hop": gp._FEAT_HOP,
        "vibrato_depth_weight": gp._VIBRATO_DEPTH_WEIGHT,
        "vibrato_rate_weight": gp._VIBRATO_RATE_WEIGHT,
    }


# =============================================================================
# section 4: train / sample
# =============================================================================

def train_from_dataset(gp, torch, nn, engine_name: str, thetas: np.ndarray, feats: np.ndarray,
                         param_names: List[str], epochs: int, out_path: str,
                         val_fraction: float = 0.15, batch_size: int = 128, lr: float = 1e-3,
                         seed: int = 0, verbose: bool = True) -> dict:
    """Entrena a partir de un dataset (theta, x) YA GENERADO -- separado de
    generate_dataset() para poder generar el dataset por trozos entre
    llamadas (la parte cara) y entrenar/ajustar hiperparametros aparte,
    sin repetir el renderizado."""
    engine = gp.RAW_ENGINES[engine_name]
    specs = engine.params
    n_samples = thetas.shape[0]

    theta_norm = _normalize_theta(thetas, specs)
    feat_mean = feats.mean(axis=0)
    feat_std = feats.std(axis=0) + 1e-6
    feat_norm = (feats - feat_mean) / feat_std

    n_val = max(1, int(n_samples * val_fraction))
    idx = np.random.RandomState(seed).permutation(n_samples)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    theta_t = torch.tensor(theta_norm, dtype=torch.float32)
    feat_t = torch.tensor(feat_norm, dtype=torch.float32)

    model = _build_model(torch, nn, theta_dim=len(specs), feat_dim=feats.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_theta = theta_t[val_idx]
    val_feat = feat_t[val_idx]

    history = []
    best_val_nll = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(epochs):
        perm = train_idx_t[torch.randperm(len(train_idx_t))]
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(perm), batch_size):
            batch_idx = perm[start:start + batch_size]
            th_batch = theta_t[batch_idx]
            fe_batch = feat_t[batch_idx]
            log_prob = model.log_prob(th_batch, fe_batch)
            loss = -log_prob.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            val_loss = -model.log_prob(val_theta, val_feat).mean().item()
        if val_loss < best_val_nll:
            best_val_nll = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_nll": epoch_loss / max(n_batches, 1),
                         "val_nll": val_loss})
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch+1:>4}/{epochs}  train_nll={epoch_loss/max(n_batches,1):.3f}"
                  f"  val_nll={val_loss:.3f}  (best={best_val_nll:.3f})")
    elapsed = time.time() - t0
    if verbose:
        print(f"  entrenamiento: {elapsed:.1f}s -- usando el checkpoint de mejor val_nll "
              f"(epoch {[h['epoch'] for h in history if h['val_nll']==best_val_nll][0]+1}), "
              f"no el de la última época")
    model.load_state_dict(best_state)

    checkpoint = {
        "engine": engine_name,
        "param_names": param_names,
        "param_bounds": [(p.lo, p.hi) for p in specs],
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "feat_dim": int(feats.shape[1]),
        "theta_dim": len(specs),
        "canonical": CANONICAL[engine_name],
        "fingerprint": _feature_fingerprint(gp),
        "n_samples": n_samples,
        "epochs": epochs,
        "final_train_nll": [h["train_nll"] for h in history if h["val_nll"] == best_val_nll][0],
        "final_val_nll": best_val_nll,
        "model_state": model.state_dict(),
    }
    torch.save(checkpoint, out_path)
    if verbose:
        print(f"  OK checkpoint -> {out_path}")
    return {"history": history, "checkpoint_path": out_path}


def train(engine_name: str, n_samples: int, epochs: int, out_path: str,
           val_fraction: float = 0.15, batch_size: int = 128, lr: float = 1e-3,
           seed: int = 0, verbose: bool = True) -> dict:
    if engine_name not in SUPPORTED_ENGINES:
        sys.exit(f"Motor no soportado por genopatch_flow (fase 1): {engine_name!r} "
                  f"-- usa uno de {SUPPORTED_ENGINES}")
    gp = _import_genopatch_v2()
    torch, nn = _import_torch()

    if verbose:
        print(f"  generando {n_samples} pares sinteticos ({engine_name})...")
    t0 = time.time()
    thetas, feats, param_names = generate_dataset(gp, engine_name, n_samples, seed=seed)
    if verbose:
        print(f"    listo en {time.time()-t0:.1f}s -- feature dim={feats.shape[1]}")

    return train_from_dataset(gp, torch, nn, engine_name, thetas, feats, param_names,
                                epochs, out_path, val_fraction=val_fraction,
                                batch_size=batch_size, lr=lr, seed=seed, verbose=verbose)


def _load_checkpoint(gp, torch, nn, path: str):
    if not Path(path).exists():
        sys.exit(f"checkpoint no encontrado: {path}")
    ckpt = torch.load(path, weights_only=False)
    current_fp = _feature_fingerprint(gp)
    if ckpt["fingerprint"] != current_fp:
        sys.exit(
            f"El checkpoint {path} se entreno con parametros de features distintos "
            f"a los actuales de genopatch_v2.py:\n"
            f"    checkpoint: {ckpt['fingerprint']}\n"
            f"    actual:     {current_fp}\n"
            f"  Reentrena el flujo.")
    model = _build_model(torch, nn, theta_dim=ckpt["theta_dim"], feat_dim=ckpt["feat_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return ckpt, model


def sample_from_target(target_wav: str, checkpoint_path: str, n: int, seed: int = 0
                         ) -> Tuple[List[Dict[str, float]], dict]:
    gp = _import_genopatch_v2()
    torch, nn = _import_torch()
    sf = _import_soundfile()

    ckpt, model = _load_checkpoint(gp, torch, nn, checkpoint_path)
    engine_name = ckpt["engine"]
    specs = gp.RAW_ENGINES[engine_name].params

    audio, sr = sf.read(target_wav, dtype="float64", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    feat = gp._extract_internal_features(audio, sr)
    if len(feat) != ckpt["feat_dim"]:
        sys.exit(f"Dimension de features del objetivo ({len(feat)}) no coincide "
                  f"con la del checkpoint ({ckpt['feat_dim']}).")

    feat_mean = np.array(ckpt["feat_mean"])
    feat_std = np.array(ckpt["feat_std"])
    feat_norm = (feat - feat_mean) / feat_std

    torch.manual_seed(seed)
    x_t = torch.tensor(feat_norm, dtype=torch.float32)
    theta_norm_samples = model.sample(n, x_t).numpy()
    theta_samples = _denormalize_theta(theta_norm_samples, specs)
    theta_samples = np.clip(theta_samples, [p.lo for p in specs], [p.hi for p in specs])

    param_names = ckpt["param_names"]
    results = [dict(zip(param_names, row)) for row in theta_samples]
    return results, ckpt


# =============================================================================
# CLI
# =============================================================================

def cmd_train(args):
    train(args.engine, args.n_samples, args.epochs, args.out,
          val_fraction=args.val_fraction, batch_size=args.batch_size, lr=args.lr,
          seed=args.seed, verbose=not args.quiet)


def _check_collapse_warning(results: List[Dict[str, float]], gp, engine_name: str) -> None:
    """Si las N muestras son casi idénticas Y varias caen justo en el límite
    de su rango, es la firma de un colapso de modo silencioso (visto en
    testing: input muy fuera de distribución -- típicamente nota distinta a
    la canónica -- hace que el flujo ignore z y devuelva siempre el mismo
    punto extremo). No es un error de programa, pero SÍ señal de que el
    resultado no es de fiar; se avisa en vez de fallar callado."""
    if len(results) < 2:
        return
    specs = {p.name: p for p in gp.RAW_ENGINES[engine_name].params}
    names = list(results[0].keys())
    n_at_bound = 0
    max_std_frac = 0.0
    for name in names:
        vals = np.array([r[name] for r in results])
        spec = specs[name]
        span = spec.hi - spec.lo
        std_frac = float(vals.std() / span) if span > 0 else 0.0
        max_std_frac = max(max_std_frac, std_frac)
        if std_frac < 0.005 and (abs(vals.mean() - spec.lo) < 0.01 * span
                                   or abs(vals.mean() - spec.hi) < 0.01 * span):
            n_at_bound += 1
    if max_std_frac < 0.01 and n_at_bound >= 1:
        print(f"  ADVERTENCIA: las {len(results)} muestras son casi idénticas y "
              f"{n_at_bound} parámetro(s) están pegados a su límite -- señal típica de "
              f"que el audio objetivo está muy fuera de lo que vio el entrenamiento "
              f"(la causa más probable, confirmada en testing: nota distinta a la "
              f"canónica de este checkpoint). Desconfía de este resultado.")


def cmd_sample(args):
    gp = _import_genopatch_v2()
    sf = _import_soundfile()
    results, ckpt = sample_from_target(args.target_wav, args.checkpoint, args.n, seed=args.seed)
    engine_name = ckpt["engine"]
    engine = gp.RAW_ENGINES[engine_name]
    canon = ckpt["canonical"]

    _check_collapse_warning(results, gp, engine_name)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, params in enumerate(results, 1):
        patch = gp.Patch(engine=engine_name, note=canon["note"], duration=canon["duration"],
                           velocity=canon["velocity"], params=params)
        base = f"sample_{i:02d}"
        gp.save_patch(patch, str(out_dir / f"{base}.json"))
        if args.render:
            audio = engine.render(params, canon["note"], canon["duration"], canon["velocity"], 44100)
            sf.write(str(out_dir / f"{base}.wav"), gp._normalize_peak(audio), 44100, subtype="FLOAT")
    suffix = " (+ WAV)" if args.render else ""
    print(f"  OK {len(results)} muestra(s){suffix} -> {out_dir}/")


def cmd_info(args):
    gp = _import_genopatch_v2()
    torch, nn = _import_torch()
    ckpt, _ = _load_checkpoint(gp, torch, nn, args.checkpoint)
    print(f"engine         {ckpt['engine']}")
    print(f"theta_dim      {ckpt['theta_dim']}")
    print(f"feat_dim       {ckpt['feat_dim']}")
    print(f"n_samples      {ckpt['n_samples']}")
    print(f"epochs         {ckpt['epochs']}")
    print(f"final_train_nll {ckpt['final_train_nll']:.3f}")
    print(f"final_val_nll   {ckpt['final_val_nll']:.3f}")
    print(f"canonical       nota={ckpt['canonical']['note']} "
          f"dur={ckpt['canonical']['duration']}s vel={ckpt['canonical']['velocity']}")
    print(f"fingerprint     {ckpt['fingerprint']}")
    print("param_names     " + ", ".join(ckpt["param_names"]))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genopatch_flow",
        description="Inferencia amortizada de parametros de sintesis via flujos "
                     "normalizadores condicionales (additive, karplus, subtractive, "
                     "wavetable, noise, fm2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("train", help="Genera dataset sintetico y entrena el flujo")
    p.add_argument("--engine", required=True, choices=list(SUPPORTED_ENGINES))
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("sample", help="WAV objetivo + checkpoint -> N patch.json")
    p.add_argument("target_wav")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("info", help="Inspecciona un checkpoint")
    p.add_argument("checkpoint")
    p.set_defaults(func=cmd_info)

    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
