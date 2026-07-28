#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MIX EVOLVER  v1.0                                   ║
║  Algoritmo genético: busca la mejor cadena de efectos para un WAV dado       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  QUÉ HACE                                                                    ║
║    Busca, mediante un algoritmo genético, la mejor `Chain` de efectos        ║
║    (tal como la define audio_effects.py) para un WAV concreto, puntuada      ║
║    por audio_reference_scorer.py contra un corpus de referencia. No          ║
║    implementa ningún efecto DSP propio: importa apply_chain de               ║
║    audio_effects.py para renderizar cada candidato, y load_manifest de ese   ║
║    mismo módulo para saber qué genes son válidos al mutar/inicializar.       ║
║                                                                              ║
║  ALCANCE                                                                     ║
║    Igual que audio_effects.py: opera sobre la mezcla completa ya             ║
║    renderizada (mono o estéreo, post-render), no sobre pistas individuales.  ║
║                                                                              ║
║  USO                                                                        ║
║    mix_evolver.py dry.wav --ref-cache ref.npz --out-dir results/            ║
║    mix_evolver.py dry.wav --ref-cache ref.npz --floor-cache floor.npz \\   ║
║                   --generations 60 --population 40                          ║
║    mix_evolver.py dry.wav --ref-cache ref.npz --top-n 5 --json log.json    ║
║    mix_evolver.py dry.wav --ref-cache ref.npz \\                           ║
║                   --effects-manifest custom_effects.json --fast             ║
║                                                                              ║
║  DEPENDENCIAS  numpy  (+ todas las de audio_effects.py y                    ║
║                 audio_reference_scorer.py, que se importan como             ║
║                 dependencias — este programa no las reimplementa)           ║
║                                                                              ║
║  LIMITACIONES (ver especificación §5)                                       ║
║    · No opera a nivel de pista individual — solo sobre la mezcla ya         ║
║      renderizada.                                                           ║
║    · Cada evaluación de fitness incluye una llamada real a                  ║
║      audio_reference_scorer.py; con poblaciones/generaciones grandes puede  ║
║      ser lento — --fast ayuda para iteración. Paralelizar la evaluación de  ║
║      la población es una mejora futura razonable, no implementada aquí.     ║
║                                                                              ║
║  Módulo importable:                                                         ║
║    from mix_evolver import evolve                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from audio_effects import (
    apply_chain, load_manifest, EffectGene, Chain,
    chain_to_dicts, save_chain,
)

# ══════════════════════════════════════════════════════════════════════════════
# ── lazy imports ────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _import_soundfile():
    try:
        import soundfile as sf
        return sf
    except ImportError:
        sys.exit("✗  soundfile no encontrado. Instala con: pip install soundfile")


def _import_scorer():
    """audio_reference_scorer.py es del mismo ecosistema (no un paquete pip).
    Se busca primero como módulo normal (mismo directorio / PYTHONPATH) y, si
    no aparece, junto a este fichero."""
    try:
        from audio_reference_scorer import score_against_corpus
        return score_against_corpus
    except ImportError:
        import importlib.util
        local = Path(__file__).resolve().parent / "audio_reference_scorer.py"
        if not local.exists():
            sys.exit("✗  audio_reference_scorer.py no encontrado (debe estar en el mismo "
                      "directorio que mix_evolver.py o en PYTHONPATH).")
        spec = importlib.util.spec_from_file_location("audio_reference_scorer", local)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.score_against_corpus


# ══════════════════════════════════════════════════════════════════════════════
# ── §2.7 estructuras de datos propias de este programa ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GenerationLog:
    generation: int
    best_score: float
    mean_score: float
    best_chain: Chain

    def to_json_dict(self) -> dict:
        return {
            "generation": self.generation,
            "best_score": self.best_score,
            "mean_score": self.mean_score,
            "best_chain": chain_to_dicts(self.best_chain),
        }


@dataclass
class EvolveResult:
    best_chain: Chain
    best_score: float
    best_wav_path: str
    top_n: List[Tuple[Chain, float, str]]
    history: List[GenerationLog]

    def to_json_dict(self) -> dict:
        return {
            "best_chain": chain_to_dicts(self.best_chain),
            "best_score": self.best_score,
            "best_wav_path": self.best_wav_path,
            "top_n": [
                {"chain": chain_to_dicts(chain), "score": score, "wav_path": wav_path}
                for chain, score, wav_path in self.top_n
            ],
            "history": [g.to_json_dict() for g in self.history],
        }


# ══════════════════════════════════════════════════════════════════════════════
# ── §2.4 operadores genéticos ───────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _random_param(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def _random_gene(rng: random.Random, manifest: dict) -> EffectGene:
    entry = rng.choice(manifest["effects"])
    params = {name: _random_param(rng, lo, hi) for name, (lo, hi) in entry["params"].items()}
    return EffectGene(effect=entry["name"], params=params)


def random_chain(rng: random.Random, manifest: dict) -> Chain:
    """Inicialización: longitud uniforme entre min/max_chain_length del
    manifest, efectos elegidos al azar (con reemplazo), parámetros muestreados
    uniformemente dentro de su rango declarado."""
    length = rng.randint(manifest["min_chain_length"], manifest["max_chain_length"])
    return [_random_gene(rng, manifest) for _ in range(length)]


def _manifest_param_ranges(manifest: dict) -> dict:
    return {entry["name"]: entry["params"] for entry in manifest["effects"]}


def tournament_select(rng: random.Random, population: List[Chain], scores: List[float],
                       tournament_size: int) -> Chain:
    idxs = rng.sample(range(len(population)), min(tournament_size, len(population)))
    winner_idx = max(idxs, key=lambda i: scores[i])
    return population[winner_idx]


def crossover(rng: random.Random, parent_a: Chain, parent_b: Chain, max_chain_length: int) -> Chain:
    """Punto de corte independiente en cada padre (listas de longitud
    distinta); el hijo toma la sub-lista inicial de un padre y la final del
    otro, truncando a max_chain_length si excede."""
    if not parent_a or not parent_b:
        child = list(parent_a) + list(parent_b)
    else:
        cut_a = rng.randint(0, len(parent_a))
        cut_b = rng.randint(0, len(parent_b))
        child = list(parent_a[:cut_a]) + list(parent_b[cut_b:])
    if not child:
        child = list(parent_a) or list(parent_b)
    return child[:max_chain_length]


def _mutate_perturb_param(rng: random.Random, chain: Chain, manifest: dict) -> Chain:
    if not chain:
        return chain
    chain = [EffectGene(g.effect, dict(g.params)) for g in chain]
    gene = rng.choice(chain)
    ranges = _manifest_param_ranges(manifest).get(gene.effect)
    if not ranges:
        return chain
    pname = rng.choice(list(gene.params.keys()))
    lo, hi = ranges[pname]
    span = hi - lo
    noise = rng.gauss(0, span * 0.1 if span else 0.1)
    gene.params[pname] = float(np.clip(gene.params[pname] + noise, lo, hi))
    return chain


def _mutate_add_effect(rng: random.Random, chain: Chain, manifest: dict) -> Chain:
    chain = list(chain)
    if len(chain) >= manifest["max_chain_length"]:
        return chain
    pos = rng.randint(0, len(chain))
    chain.insert(pos, _random_gene(rng, manifest))
    return chain


def _mutate_remove_effect(rng: random.Random, chain: Chain, manifest: dict) -> Chain:
    if len(chain) <= manifest["min_chain_length"]:
        return list(chain)
    chain = list(chain)
    del chain[rng.randrange(len(chain))]
    return chain


def _mutate_reorder(rng: random.Random, chain: Chain, manifest: dict) -> Chain:
    if len(chain) < 2:
        return list(chain)
    chain = list(chain)
    i = rng.randrange(len(chain) - 1)
    chain[i], chain[i + 1] = chain[i + 1], chain[i]
    return chain


def _mutate_replace_effect(rng: random.Random, chain: Chain, manifest: dict) -> Chain:
    if not chain:
        return chain
    chain = list(chain)
    i = rng.randrange(len(chain))
    chain[i] = _random_gene(rng, manifest)
    return chain


_MUTATION_OPERATORS = [
    ("perturb_param", _mutate_perturb_param, 0.4),
    ("add_effect", _mutate_add_effect, 0.15),
    ("remove_effect", _mutate_remove_effect, 0.15),
    ("reorder", _mutate_reorder, 0.15),
    ("replace_effect", _mutate_replace_effect, 0.15),
]


def mutate(rng: random.Random, chain: Chain, manifest: dict) -> Chain:
    """Un operador al azar, ponderado, por evento de mutación (§2.4)."""
    names, funcs, weights = zip(*_MUTATION_OPERATORS)
    op = rng.choices(funcs, weights=weights, k=1)[0]
    return op(rng, chain, manifest)


# ══════════════════════════════════════════════════════════════════════════════
# ── §2.5 función de fitness ──────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def write_temp_wav(audio: np.ndarray, sr: int, tmp_dir: str) -> str:
    sf = _import_soundfile()
    fd, path = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
    import os
    os.close(fd)
    sf.write(path, audio, sr, subtype="FLOAT")
    return path


def evaluate_genome(chain: Chain, dry_audio: np.ndarray, sr: int,
                     ref_cache: str, floor_cache: Optional[str], tmp_dir: str) -> Tuple[float, str]:
    """Aplica la cadena (audio_effects.apply_chain) y puntúa el resultado
    contra el corpus (audio_reference_scorer.score_against_corpus). Ni la
    aplicación de efectos ni la evaluación se reimplementan aquí — este
    programa solo orquesta ambas. Devuelve (score, wav_path) para poder
    reutilizar el WAV renderizado sin tener que reaplicar la cadena."""
    score_against_corpus = _import_scorer()
    processed = apply_chain(chain, dry_audio, sr)
    wav_path = write_temp_wav(processed, sr, tmp_dir)
    result = score_against_corpus(wav_path, ref_cache, floor_cache=floor_cache)
    score = (result.score_relative_to_floor if result.score_relative_to_floor is not None
             else result.score_absolute)
    if score is None or not np.isfinite(score):
        # Cadenas con parámetros extremos (p.ej. varios EQ apilados con ganancias
        # altas) pueden saturar la señal hasta producir NaN/inf en el backend de
        # features del scorer. No es un genoma válido: se trata como el peor
        # score posible (0.0) en vez de dejar que un NaN corrompa max()/mean()
        # y rompa la invariante de "mejora no decreciente" del GA.
        score = 0.0
    return score, wav_path


# ══════════════════════════════════════════════════════════════════════════════
# ── §2.6 bucle evolutivo ─────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def evolve(dry_wav: str, ref_cache: str, floor_cache: Optional[str], manifest_path: Optional[str],
           population_size: int = 30, generations: int = 40,
           mutation_rate: float = 0.3, elite_count: int = 2,
           patience: int = 8, tournament_size: int = 3, top_n: int = 3,
           seed: Optional[int] = None, verbose: bool = True) -> EvolveResult:
    """patience: si el mejor score no mejora en `patience` generaciones
    seguidas, detener antes de `generations`. Estructura estándar: población
    inicial → evaluar fitness de todos → mientras queden generaciones y no se
    agote patience: elitismo + selección por torneo + cruce + mutación →
    nueva población → evaluar → registrar mejor score de la generación."""
    sf = _import_soundfile()
    manifest = load_manifest(manifest_path)
    rng = random.Random(seed)

    dry_audio, sr = sf.read(dry_wav, dtype="float64", always_2d=False)

    tmp_dir = tempfile.mkdtemp(prefix="mix_evolver_")
    try:
        population = [random_chain(rng, manifest) for _ in range(population_size)]
        scores: List[float] = []
        wav_paths: List[str] = []
        for chain in population:
            score, wav_path = evaluate_genome(chain, dry_audio, sr, ref_cache, floor_cache, tmp_dir)
            scores.append(score)
            wav_paths.append(wav_path)

        history: List[GenerationLog] = []
        best_score = max(scores)
        best_idx = scores.index(best_score)
        best_chain = population[best_idx]
        best_wav_path = wav_paths[best_idx]
        stale_generations = 0

        history.append(GenerationLog(
            generation=0, best_score=best_score,
            mean_score=float(np.mean(scores)), best_chain=best_chain))
        if verbose:
            print(f"  gen 0   best={best_score:.4f}  mean={np.mean(scores):.4f}")

        gen = 0
        while gen < generations - 1 and stale_generations < patience:
            gen += 1
            ranked_idx = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
            new_population = [population[i] for i in ranked_idx[:elite_count]]

            while len(new_population) < population_size:
                parent_a = tournament_select(rng, population, scores, tournament_size)
                parent_b = tournament_select(rng, population, scores, tournament_size)
                child = crossover(rng, parent_a, parent_b, manifest["max_chain_length"])
                if rng.random() < mutation_rate:
                    child = mutate(rng, child, manifest)
                if len(child) < manifest["min_chain_length"]:
                    child = child + [_random_gene(rng, manifest)
                                      for _ in range(manifest["min_chain_length"] - len(child))]
                new_population.append(child)

            population = new_population
            scores = []
            wav_paths = []
            for chain in population:
                score, wav_path = evaluate_genome(chain, dry_audio, sr, ref_cache, floor_cache, tmp_dir)
                scores.append(score)
                wav_paths.append(wav_path)

            gen_best_score = max(scores)
            gen_best_idx = scores.index(gen_best_score)

            if gen_best_score > best_score:
                best_score = gen_best_score
                best_chain = population[gen_best_idx]
                best_wav_path = wav_paths[gen_best_idx]
                stale_generations = 0
            else:
                stale_generations += 1

            history.append(GenerationLog(
                generation=gen, best_score=gen_best_score,
                mean_score=float(np.mean(scores)), best_chain=population[gen_best_idx]))
            if verbose:
                print(f"  gen {gen}   best={gen_best_score:.4f}  mean={np.mean(scores):.4f}"
                      f"{'  (stale)' if stale_generations else ''}")

        ranked_idx = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        top_entries: List[Tuple[Chain, float, str]] = []
        seen_best = False
        for i in ranked_idx:
            if population[i] is best_chain and not seen_best:
                seen_best = True
            top_entries.append((population[i], scores[i], wav_paths[i]))
            if len(top_entries) >= top_n:
                break
        if not any(chain is best_chain for chain, _, _ in top_entries):
            top_entries = [(best_chain, best_score, best_wav_path)] + top_entries[: max(0, top_n - 1)]

        return EvolveResult(
            best_chain=best_chain, best_score=best_score, best_wav_path=best_wav_path,
            top_n=top_entries, history=history)
    finally:
        # los WAVs relevantes (best_wav_path, top_n) se copian a --out-dir antes
        # de llegar aquí en el flujo de la CLI; el directorio temporal completo
        # de la búsqueda se limpia siempre al terminar evolve().
        pass  # cleanup real ocurre en _cmd_evolve tras copiar lo necesario


# ══════════════════════════════════════════════════════════════════════════════
# ── CLI (§2.8) ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _cmd_evolve(args) -> None:
    manifest_path = args.effects_manifest
    population = args.population
    generations = args.generations
    if args.fast:
        population = min(population, 10)
        generations = min(generations, 10)

    t0 = time.time()
    result = evolve(
        dry_wav=args.dry_wav, ref_cache=args.ref_cache, floor_cache=args.floor_cache,
        manifest_path=manifest_path, population_size=population, generations=generations,
        mutation_rate=args.mutation_rate, elite_count=args.elite_count,
        patience=args.patience, tournament_size=args.tournament_size,
        top_n=args.top_n, seed=args.seed, verbose=not args.quiet,
    )
    elapsed = time.time() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_chain(result.best_chain, str(out_dir / "best_chain.json"))
    shutil.copy(result.best_wav_path, out_dir / "best.wav")

    top_dir = out_dir / "top_N"
    top_dir.mkdir(exist_ok=True)
    for rank, (chain, score, wav_path) in enumerate(result.top_n, start=1):
        save_chain(chain, str(top_dir / f"rank{rank}_chain.json"))
        shutil.copy(wav_path, top_dir / f"rank{rank}.wav")

    convergence = [g.to_json_dict() for g in result.history]
    (out_dir / "convergence_log.json").write_text(
        json.dumps(convergence, indent=2, ensure_ascii=False))

    if args.json:
        Path(args.json).write_text(json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False))

    if args.quiet:
        print(f"{result.best_score:.4f}")
    else:
        print(f"  ✓  Mejor score: {result.best_score:.4f}  ({elapsed:.1f}s, "
              f"{len(result.history)} generaciones)")
        print(f"  ✓  {out_dir / 'best_chain.json'}")
        print(f"  ✓  {out_dir / 'best.wav'}")
        print(f"  ✓  {top_dir}/ ({len(result.top_n)} genomas)")
        print(f"  ✓  {out_dir / 'convergence_log.json'}")


def main():
    parser = argparse.ArgumentParser(
        prog="mix_evolver",
        description="Busca, mediante algoritmo genético, la mejor cadena de efectos "
                     "(audio_effects.py) para un WAV, puntuada por "
                     "audio_reference_scorer.py contra un corpus de referencia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dry_wav", help="Audio de entrada sin procesar")
    parser.add_argument("--ref-cache", metavar="FILE", required=True,
                         help="Caché de corpus de audio_reference_scorer.py (obligatorio)")
    parser.add_argument("--floor-cache", metavar="FILE", default=None,
                         help="Calibración por suelo opcional")
    parser.add_argument("--effects-manifest", metavar="FILE", default=None,
                         help="Pasa a audio_effects.load_manifest; default el manifest de §1.3")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--mutation-rate", type=float, default=0.3)
    parser.add_argument("--elite-count", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--top-n", type=int, default=3,
                         help="Cuántos genomas finales exportar además del mejor (default 3)")
    parser.add_argument("--out-dir", metavar="DIR", default="results/")
    parser.add_argument("--json", metavar="FILE", default=None,
                         help="Log completo (EvolveResult serializado)")
    parser.add_argument("--fast", action="store_true",
                         help="Reduce población/generaciones para iteración rápida")
    parser.add_argument("--quiet", action="store_true",
                         help="Solo imprime el score final del mejor genoma")
    parser.add_argument("--seed", type=int, default=None,
                         help="Semilla del propio GA (selección/mutación/cruce); no confundir "
                              "con la semilla del reverb, que gestiona audio_effects.py")
    args = parser.parse_args()

    try:
        _cmd_evolve(args)
    except SystemExit:
        raise
    except FileNotFoundError as e:
        sys.exit(f"✗  Fichero no encontrado: {e.filename or e}")
    except Exception as e:
        if type(e).__name__ in ("LibsndfileError", "SoundFileError", "SoundFileRuntimeError"):
            sys.exit(f"✗  Error leyendo/escribiendo audio: {e}")
        raise


if __name__ == "__main__":
    main()
