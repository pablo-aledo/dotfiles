#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════╗
║ miditok_composer.py                                                       ║
║                                                                            ║
║ Tokenizador de música para modelos de secuencias (Transformer, etc).      ║
║ Convierte MIDI <-> tokens usando cualquiera de los esquemas clásicos      ║
║ (REMI, TSD, MIDI-Like, Structured, CPWord, Octuple, MuMIDI, MMM, PerTok), ║
║ entrena vocabularios sub-token (BPE / Unigram / WordPiece) sobre un       ║
║ corpus, e inspecciona MIDIs o tokenizadores ya entrenados.                ║
║                                                                            ║
║ Adaptación al estilo mutopia de MidiTok (Nathan Fradet, MIT License).     ║
║ Backend real: paquetes `miditok` + `symusic` (no se reimplementa la       ║
║ lógica de tokenización -- sería reinventar 9 papers distintos --, este    ║
║ fichero es una fachada CLI de fichero único sobre esa librería).          ║
║                                                                            ║
║ Uso:                                                                      ║
║   miditok_composer.py list-schemes                                       ║
║   miditok_composer.py info song.mid                                      ║
║   miditok_composer.py info tokenizer.json                                ║
║   miditok_composer.py tokenize song.mid -s REMI -o tokens.json           ║
║   miditok_composer.py tokenize song.mid -s REMI --tokenizer tok.json     ║
║   miditok_composer.py detokenize tokens.json --tokenizer tok.json \      ║
║       -o song_back.mid                                                   ║
║   miditok_composer.py train corpus/*.mid -s REMI --vocab-size 8000 \     ║
║       --model bpe -o tokenizer.json                                      ║
║                                                                            ║
║ Deps: pip install miditok symusic                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


class C:
    """Códigos ANSI para salida coloreada."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def ok(msg: str) -> None:
    print(f"{C.GREEN}✓{C.RESET} {msg}")


def info_line(msg: str) -> None:
    print(f"{C.CYAN}i{C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{C.YELLOW}!{C.RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{C.RED}✗ error:{C.RESET} {msg}", file=sys.stderr)
    sys.exit(1)


SCHEMES = ("REMI", "TSD", "MIDILike", "Structured", "CPWord", "Octuple", "MuMIDI", "MMM", "PerTok")

SCHEME_NOTES = {
    "REMI": "posición+bar explícitos; el más usado para generación con Transformer",
    "TSD": "similar a MIDI-Like pero con duración explícita en vez de note-off",
    "MIDILike": "note-on/note-off tal cual, secuencial",
    "Structured": "ritmo estrictamente periódico, gramática fija por posición",
    "CPWord": "tokens compuestos (varias familias por paso temporal, más compacto)",
    "Octuple": "8 atributos en paralelo por nota, secuencias muy cortas",
    "MuMIDI": "pensado para multitrack + letras (MuseGAN-like)",
    "MMM": "multi-track por bloques, bueno para inpainting/control por pista",
    "PerTok": "tokenización orientada a expresividad/microtiming (más reciente)",
}


def get_backend():
    """Importa miditok/symusic con un mensaje claro si faltan."""
    try:
        import miditok  # noqa: F401
        import symusic  # noqa: F401
    except ImportError as e:
        fail(
            f"falta una dependencia ({e.name}). Instala con:\n"
            f"    pip install miditok symusic"
        )
    import miditok

    return miditok


def build_tokenizer(miditok_mod, scheme: str, tokenizer_path: str | None, config_overrides: dict):
    """Construye o carga un tokenizador para el esquema pedido."""
    if tokenizer_path:
        cls = getattr(miditok_mod, scheme)
        return cls(params=Path(tokenizer_path))
    cls = getattr(miditok_mod, scheme)
    cfg = miditok_mod.TokenizerConfig(**config_overrides)
    return cls(cfg)


# --------------------------------------------------------------------------- #
# subcomandos
# --------------------------------------------------------------------------- #


def cmd_list_schemes(_args: argparse.Namespace) -> None:
    print(f"{C.BOLD}esquemas de tokenización disponibles:{C.RESET}")
    for name in SCHEMES:
        print(f"  {C.CYAN}{name:<12}{C.RESET} {SCHEME_NOTES[name]}")


def cmd_info(args: argparse.Namespace) -> None:
    mtk = get_backend()
    path = Path(args.path)
    if not path.exists():
        fail(f"no existe: {path}")

    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            fail(f"'{path}' no es JSON válido: {e}")

        if "_vocab_base" in data:  # tokenizador entrenado/guardado por miditok
            vocab = data.get("_vocab_base")
            vocab_size = len(vocab) if isinstance(vocab, dict) else "?"
            print(f"{C.BOLD}{path.name}{C.RESET} — tokenizador")
            info_line(f"esquema:       {data.get('tokenization', '?')}")
            info_line(f"tamaño vocab base: {vocab_size}")
            info_line(f"modelo BPE/subtoken guardado: {'sí' if data.get('_model') else 'no'}")
            info_line(f"miditok:       {data.get('miditok_version', '?')}  symusic: {data.get('symusic_version', '?')}")
        elif "tracks" in data:  # salida de este mismo `tokenize`
            n_tok = sum(len(t.get("tokens", [])) for t in data["tracks"])
            print(f"{C.BOLD}{path.name}{C.RESET} — tokens (salida de 'tokenize')")
            info_line(f"esquema:       {data.get('scheme', '?')}")
            info_line(f"origen:        {data.get('source', '?')}")
            info_line(f"pistas:        {len(data['tracks'])}")
            info_line(f"tokens totales:{n_tok}")
        else:
            warn(f"'{path}' es JSON pero no reconozco su formato")
        return

    # asumimos fichero de música (MIDI o abc)
    import symusic

    score = symusic.Score(str(path))
    dur_s = score.end() / score.ticks_per_quarter / (score.tempos[0].qpm / 60) if score.tempos else None
    print(f"{C.BOLD}{path.name}{C.RESET} — partitura")
    info_line(f"pistas:        {len(score.tracks)}")
    info_line(f"notas:         {sum(len(t.notes) for t in score.tracks)}")
    info_line(f"ticks/negra:   {score.ticks_per_quarter}")
    info_line(f"tempos:        {len(score.tempos)}")
    info_line(f"compases (sig):{len(score.time_signatures)}")
    for t in score.tracks[:8]:
        marca = "batería" if t.is_drum else f"programa {t.program}"
        print(f"    · {t.name or '(sin nombre)':<20} {marca:<14} {len(t.notes)} notas")
    if len(score.tracks) > 8:
        print(f"    … y {len(score.tracks) - 8} pistas más")


def cmd_tokenize(args: argparse.Namespace) -> None:
    mtk = get_backend()
    import symusic

    src = Path(args.midi)
    if not src.exists():
        fail(f"no existe: {src}")

    overrides = {}
    if args.num_velocities:
        overrides["num_velocities"] = args.num_velocities
    if args.use_chords:
        overrides["use_chords"] = True
    if args.use_programs:
        overrides["use_programs"] = True

    tokenizer = build_tokenizer(mtk, args.scheme, args.tokenizer, overrides)
    score = symusic.Score(str(src))
    tok_seq = tokenizer(score)

    # esquemas "one_token_stream" (p.ej. MuMIDI) devuelven un único TokSequence
    # para toda la pieza; el resto devuelve una lista, una por pista.
    one_stream = bool(getattr(tokenizer, "one_token_stream", False))
    seqs = [tok_seq] if one_stream else tok_seq
    out = {
        "scheme": args.scheme,
        "source": str(src),
        "one_token_stream": one_stream,
        "tracks": [
            {"tokens": s.tokens, "ids": s.ids} for s in seqs
        ],
    }

    dest = Path(args.output) if args.output else src.with_suffix(".tokens.json")
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    n_tok = sum(len(s.tokens) for s in seqs)
    ok(f"{n_tok} tokens ({len(seqs)} pista(s)) -> {dest}")


def cmd_detokenize(args: argparse.Namespace) -> None:
    mtk = get_backend()

    src = Path(args.tokens)
    if not src.exists():
        fail(f"no existe: {src}")
    data = json.loads(src.read_text())
    scheme = data.get("scheme") or args.scheme
    if not scheme:
        fail("el fichero de tokens no indica 'scheme'; pásalo con --scheme")

    tokenizer = build_tokenizer(mtk, scheme, args.tokenizer, {})

    # los tokenizadores "one_token_stream" (p.ej. MuMIDI) esperan los ids
    # de la única pista sin envolver en una lista de pistas; el resto
    # espera una lista con un elemento por pista.
    one_stream = bool(data.get("one_token_stream", getattr(tokenizer, "one_token_stream", False)))
    if one_stream:
        score = tokenizer(data["tracks"][0]["ids"])
    else:
        ids_per_track = [track["ids"] for track in data["tracks"]]
        score = tokenizer(ids_per_track)

    dest = Path(args.output) if args.output else src.with_suffix(".mid")
    score.dump_midi(str(dest))
    ok(f"reconstruido -> {dest}")


def cmd_train(args: argparse.Namespace) -> None:
    mtk = get_backend()

    files = []
    for pattern in args.midis:
        matched = glob.glob(pattern, recursive=True)
        files.extend(matched if matched else [pattern])
    files = [f for f in files if Path(f).exists()]
    if not files:
        fail("ningún fichero MIDI encontrado en los patrones dados")

    overrides = {}
    if args.num_velocities:
        overrides["num_velocities"] = args.num_velocities
    tokenizer = build_tokenizer(mtk, args.scheme, None, overrides)

    info_line(f"entrenando {args.model} sobre {len(files)} fichero(s), vocab_size={args.vocab_size}...")
    tokenizer.train(
        vocab_size=args.vocab_size,
        model=args.model,
        files_paths=[Path(f) for f in files],
    )

    dest = Path(args.output)
    tokenizer.save(dest)
    ok(f"vocabulario entrenado ({len(tokenizer.vocab)} tokens) -> {dest}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(
        prog="miditok_composer.py",
        description="Tokenización de música MIDI/abc para modelos de secuencias.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list-schemes", help="lista los esquemas de tokenización disponibles")
    sp.set_defaults(func=cmd_list_schemes)

    sp = sub.add_parser("info", help="inspecciona un MIDI o un tokenizador (.json)")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("tokenize", help="convierte un MIDI en tokens")
    sp.add_argument("midi")
    sp.add_argument("-s", "--scheme", default="REMI", choices=SCHEMES)
    sp.add_argument("--tokenizer", help="tokenizador ya entrenado (.json); si no, config por defecto")
    sp.add_argument("-o", "--output", help="fichero de salida (.json); por defecto <midi>.tokens.json")
    sp.add_argument("--num-velocities", type=int, default=None)
    sp.add_argument("--use-chords", action="store_true")
    sp.add_argument("--use-programs", action="store_true")
    sp.set_defaults(func=cmd_tokenize)

    sp = sub.add_parser("detokenize", help="reconstruye un MIDI a partir de tokens")
    sp.add_argument("tokens")
    sp.add_argument("--scheme", default=None, choices=SCHEMES, help="solo si el .json de tokens no lo indica")
    sp.add_argument("--tokenizer", help="tokenizador usado al generar los tokens (.json)")
    sp.add_argument("-o", "--output", help="fichero .mid de salida")
    sp.set_defaults(func=cmd_detokenize)

    sp = sub.add_parser("train", help="entrena un vocabulario sub-token (BPE/Unigram/WordPiece) sobre un corpus")
    sp.add_argument("midis", nargs="+", help="ficheros o patrones glob, p.ej. corpus/**/*.mid")
    sp.add_argument("-s", "--scheme", default="REMI", choices=SCHEMES)
    sp.add_argument("--vocab-size", type=int, default=8000)
    sp.add_argument("--model", default="BPE", choices=["BPE", "Unigram", "WordPiece"])
    sp.add_argument("--num-velocities", type=int, default=None)
    sp.add_argument("-o", "--output", default="tokenizer.json")
    sp.set_defaults(func=cmd_train)

    args = p.parse_args()
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as e:  # errores de miditok/symusic -> mensaje limpio, no traceback
        fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
