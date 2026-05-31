#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           CLARA  v2.0                                        ║
║              Generador de música con red neuronal AWD-LSTM                   ║
║                       (PyTorch 2.x + FastAI v2)                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  COMANDOS                                                                    ║
║                                                                              ║
║    encode      — MIDI corpus → archivos de texto codificados                 ║
║    split       — Distribuye los .txt codificados en train / test             ║
║    train       — Entrena el modelo AWD-LSTM generador                        ║
║    generate    — Genera piezas nuevas con un modelo entrenado                ║
║    critic-data — Crea datos para el crítico (real vs. generado)              ║
║    critic      — Entrena el clasificador real/generado (crítico)             ║
║    composer-data — Crea datos para el clasificador de compositores           ║
║    composer    — Entrena el clasificador de compositores                     ║
║    serve       — Lanza un servidor HTTP con playlist interactiva             ║
║                                                                              ║
║  DEPENDENCIAS                                                                ║
║    pip install torch fastai music21 mido numpy tqdm                         ║
║    sudo apt install fluidsynth mpg321 twolame                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FLUJO BÁSICO (datos de ejemplo incluidos)                                   ║
║                                                                              ║
║    # 1. Usar datos de ejemplo (ya codificados)                               ║
║    python clara.py split --example                                           ║
║                                                                              ║
║    # 2. Entrenar                                                             ║
║    python clara.py train --prefix mi_modelo                                  ║
║                                                                              ║
║    # 3. Generar                                                              ║
║    python clara.py generate --model mi_modelo --output mis_piezas            ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  FLUJO COMPLETO (MIDIs propios)                                              ║
║                                                                              ║
║    # 1. Codificar MIDIs                                                      ║
║    python clara.py encode                                                    ║
║    python clara.py encode --chamber                                          ║
║    python clara.py encode --composers bach,beethoven                         ║
║                                                                              ║
║    # 2. Construir train/test (notewise, piano solo, 62 notas, freq 12)       ║
║    python clara.py split                                                     ║
║    python clara.py split --chordwise                                         ║
║    python clara.py split --composers bach,mozart --sample 0.5                ║
║                                                                              ║
║    # 3. Entrenar generador                                                   ║
║    python clara.py train --prefix mi_modelo --epochs 5 --nl 4 --nh 600      ║
║    python clara.py train --prefix mi_modelo --load mi_modelo --level light   ║
║                                                                              ║
║    # 4. Generar piezas                                                       ║
║    python clara.py generate --model mi_modelo --output mis_piezas            ║
║    python clara.py generate --model notewise_generator --output demo \       ║
║        --size 2000 --random-freq 0.8 --trunc 3 --bs 8                       ║
║                                                                              ║
║    # 5. (Opcional) Entrenar crítico                                          ║
║    python clara.py critic-data --model mi_modelo                             ║
║    python clara.py critic --model mi_modelo                                  ║
║                                                                              ║
║    # 6. (Opcional) Entrenar clasificador de compositores                     ║
║    python clara.py composer-data --model mi_modelo                           ║
║    python clara.py composer --model mi_modelo                                ║
║                                                                              ║
║    # 7. Servidor de escucha                                                  ║
║    python clara.py serve --model mi_modelo                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ── Imports estándar ──────────────────────────────────────────────────────────
import argparse
import os
import random
import shutil
import sys
import threading
from glob import glob
from math import floor
from pathlib import Path
from functools import partial

import numpy as np
from tqdm import tqdm

# ── Imports opcionales (se comprueban en cada subcomando) ─────────────────────
def _require(pkg, pip_name=None):
    """Importa un paquete y aborta con mensaje claro si no está instalado."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        name = pip_name or pkg
        sys.exit(f"[clara] Dependencia no encontrada: '{pkg}'\n"
                 f"        Instala con:  pip install {name}")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — RUTAS
# ═════════════════════════════════════════════════════════════════════════════

def _paths():
    """Devuelve el diccionario canónico de rutas del proyecto."""
    base = Path(".")
    p = {
        "data":          base / "data",
        "composers_mid": base / "data" / "composers" / "midi",
        "notewise":      base / "data" / "composers" / "notewise",
        "chordwise":     base / "data" / "composers" / "chordwise",
        "train":         base / "data" / "train",
        "test":          base / "data" / "test",
        "example_train": base / "data" / "example_data" / "train",
        "example_test":  base / "data" / "example_data" / "test",
        "output":        base / "data" / "output",
        "models":        base / "models",
        "generator":     base / "models" / "generator",
        "critic_dir":    base / "models" / "critic",
        "composer_dir":  base / "models" / "composer",
        "critic_data":   base / "critic_data",
        "composer_data": base / "composer_data",
    }
    for v in p.values():
        v.mkdir(parents=True, exist_ok=True)
    return p


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CODIFICACIÓN MIDI → TEXTO  (encode)
# ═════════════════════════════════════════════════════════════════════════════

VIOLINLIKE = [
    "Violin", "Viola", "Cello", "Violincello", "Violoncello", "Flute",
    "Oboe", "Clarinet", "Recorder", "Voice", "Piccolo",
    "StringInstrument", "Bassoon", "Horn",
]
PIANOLIKE = ["Piano", "Harp", "Harpsichord", "Organ", ""]


def _assign_instrument(instr):
    s = str(instr)
    if s in PIANOLIKE:
        return 0
    if s in VIOLINLIKE:
        return 1
    print(f"  Advertencia: instrumento desconocido '{s}'")
    return -1


def _stream_to_chordwise(s, chamber, note_range, note_offset, sample_freq):
    music21 = _require("music21")
    n_instr = 2 if chamber else 1
    max_t = floor(s.duration.quarterLength * sample_freq) + 1
    arr = np.zeros((max_t, n_instr, note_range))

    notes = []
    instr_id = 0
    nf = music21.stream.filters.ClassFilter("Note")
    cf = music21.stream.filters.ClassFilter("Chord")

    for n in s.recurse().addFilter(nf):
        if chamber:
            instr_id = _assign_instrument(n.activeSite.getInstrument())
            if instr_id == -1:
                return []
        notes.append((
            n.pitch.midi - note_offset,
            floor(n.offset * sample_freq),
            floor(n.duration.quarterLength * sample_freq),
            instr_id,
        ))

    for c in s.recurse().addFilter(cf):
        if chamber:
            instr_id = _assign_instrument(c.activeSite.getInstrument())
            if instr_id == -1:
                return []
        for p in c.pitches:
            notes.append((
                p.midi - note_offset,
                floor(c.offset * sample_freq),
                floor(c.duration.quarterLength * sample_freq),
                instr_id,
            ))

    for pitch_raw, t_start, dur, iid in notes:
        pitch = pitch_raw
        while pitch < 0:
            pitch += 12
        while pitch >= note_range:
            pitch -= 12
        if iid == 1 and pitch < 22:
            while pitch < 22:
                pitch += 12
        arr[t_start, iid, pitch] = 1
        arr[t_start + 1:t_start + dur, iid, pitch] = 2

    label = {0: "p", 1: "v"}
    out = []
    for timestep in arr:
        for i in reversed(range(len(timestep))):
            out.append(label[i] + "".join(str(int(x)) for x in timestep[i]))
    return out


def _add_modulations(arr):
    note_range = len(arr[0]) - 1
    result = []
    for shift in range(12):
        for chord in arr:
            padded = "000000" + chord[1:] + "000000"
            result.append(chord[0] + padded[shift:shift + note_range])
    return result


def _chord_to_notewise(chords, sample_freq):
    tokens = []
    for j, chord in enumerate(chords):
        next_chord = ""
        for k in range(j + 1, len(chords)):
            if chords[k][0] == chord[0]:
                next_chord = chords[k]
                break
        prefix = chord[0]
        body = chord[1:]
        nb = next_chord[1:] if next_chord else ""
        for i, ch in enumerate(body):
            if ch == "0":
                continue
            note = prefix + str(i)
            if ch == "1":
                tokens.append(note)
            if not nb or nb[i] == "0":
                tokens.append("end" + note)
        if prefix == "p":
            tokens.append("wait")

    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "wait":
            count = 1
            while (count <= sample_freq * 2
                   and i + count < len(tokens)
                   and tokens[i + count] == "wait"):
                count += 1
            out.append("wait" + str(count))
            i += count
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def _encoding_dir(base, kind, note_range, sample_freq, chamber, composer):
    ensemble = "chamber" if chamber else "piano_solo"
    d = (base / kind / ensemble
         / f"note_range{note_range}" / f"sample_freq{sample_freq}" / composer)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _encode_piece(fpath, composer, chamber, sample_freqs, note_ranges,
                  note_offsets, cw_base, nw_base, replace):
    music21 = _require("music21")

    if not replace:
        all_exist = all(
            (_encoding_dir(nw_base, "notewise", nr, sf, chamber, composer)
             / (fpath.stem + ".txt")).exists()
            and
            (_encoding_dir(cw_base, "chordwise", nr, sf, chamber, composer)
             / (fpath.stem + ".txt")).exists()
            for sf in sample_freqs for nr in note_ranges
        )
        if all_exist:
            print("  Omitiendo (ya existe). Usa --replace para recodificar.")
            return

    mf = music21.midi.MidiFile()
    try:
        mf.open(str(fpath))
        mf.read()
        mf.close()
    except Exception:
        print("  Omitiendo: archivo MIDI con formato incorrecto")
        return

    if chamber and len(mf.tracks) == 1:
        print("  Omitiendo: se esperaba música de cámara pero solo hay 1 pista")
        return

    try:
        stream = music21.midi.translate.midiFileToStream(mf)
    except Exception:
        print("  Omitiendo: fallo en midiFileToStream")
        return

    fname_stem = fpath.stem + ".txt"
    for sf in sample_freqs:
        for nr in note_ranges:
            cw_arr = _stream_to_chordwise(stream, chamber, nr,
                                          note_offsets[nr], sf)
            if not cw_arr:
                print("  Omitiendo: instrumento desconocido")
                return
            cw_arr = _add_modulations(cw_arr)

            cw_dir = _encoding_dir(cw_base, "chordwise", nr, sf, chamber, composer)
            (cw_dir / fname_stem).write_text(" ".join(cw_arr))

            nw_str = _chord_to_notewise(cw_arr, sf)
            nw_dir = _encoding_dir(nw_base, "notewise", nr, sf, chamber, composer)
            (nw_dir / fname_stem).write_text(nw_str)

    print("  OK")


def cmd_encode(args):
    """MIDI corpus → archivos de texto codificados (notewise + chordwise)."""
    P = _paths()
    ensemble = "chamber" if args.chamber else "piano_solo"
    midi_base = P["composers_mid"] / ensemble

    if not midi_base.exists():
        sys.exit(f"[clara] No se encuentra la carpeta de MIDIs: {midi_base}")

    composers = (args.composers.split(",") if args.composers
                 else [d.name for d in midi_base.iterdir() if d.is_dir()])

    sample_freqs = [4, 12]
    note_ranges  = [38, 62]
    note_offsets = {38: 45, 62: 33}

    for composer in composers:
        midi_dir = midi_base / composer
        if not midi_dir.exists():
            print(f"[encode] Carpeta no encontrada, omitiendo: {midi_dir}")
            continue
        files = sorted(midi_dir.glob("*.mid"))
        print(f"\n[encode] {composer} — {len(files)} archivos")
        for j, fpath in enumerate(files, 1):
            print(f"  [{j}/{len(files)}] {fpath.name}")
            _encode_piece(fpath, composer, args.chamber,
                          sample_freqs, note_ranges, note_offsets,
                          P["chordwise"], P["notewise"], args.replace)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — SPLIT TRAIN/TEST  (split)
# ═════════════════════════════════════════════════════════════════════════════

def _change_rests(text):
    """Codificación especial de silencios para chordwise (mejora el entrenamiento)."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    notes = text.split(" ")
    if not notes:
        return text
    nr = len(notes[0]) - 1
    for i, tok in enumerate(notes):
        if tok and tok[1:] == "0" * nr:
            prev = sum(
                1 for j in range(max(0, i - 10), i)
                if notes[j] for c in notes[j][1:] if c == "1"
            )
            prev = min(prev, len(alphabet) - 1)
            notes[i] = tok[0] + alphabet[prev] * nr
    return " ".join(notes)


def _remove_duration(directory):
    """Chordwise: elimina marcas '2' (nota sostenida) y expande silencios."""
    for fpath in list(directory.glob("*.txt")):
        text = fpath.read_text()
        fpath.unlink()
        text = text.replace("2", "0")
        text = _change_rests(text)
        tokens = text.split(" ")
        piece = len(tokens) // 12
        for i in range(12):
            out = directory / f"{fpath.stem}___{i}.txt"
            out.write_text(" ".join(tokens[i * piece:(i + 1) * piece]))


def _remove_wait(directory):
    """Notewise: fusiona wait1 con la nota previa (marca 'eoc') y divide en 12."""
    for fpath in tqdm(list(directory.glob("*.txt")), desc="  remove_wait"):
        tokens = fpath.read_text().split(" ")
        fpath.unlink()
        i = 1
        while i < len(tokens):
            if (tokens[i][:4] == "wait"
                    and tokens[i - 1][:4] != "wait"
                    and tokens[i - 1][-3:] != "eoc"):
                tokens[i - 1] += "eoc"
                if tokens[i] == "wait1":
                    tokens[i] = ""
                else:
                    tokens[i] = "wait" + str(int(tokens[i][4:]) - 1)
            i += 1
        piece = len(tokens) // 12
        for k in range(12):
            out = directory / f"{fpath.stem}___{k}.txt"
            out.write_text(" ".join(tokens[k * piece:(k + 1) * piece]))


def cmd_split(args):
    """Construye las carpetas train / test a partir de los .txt codificados."""
    P = _paths()

    if args.example:
        print("[split] Copiando datos de ejemplo...")
        for dst, src in [(P["train"], P["example_train"]),
                         (P["test"],  P["example_test"])]:
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        print("[split] Listo. Carpetas train y test listas con datos de ejemplo.")
        return

    chordwise  = args.chordwise
    chamber    = args.chamber
    small_nr   = args.small_note_range
    ensemble   = "chamber" if chamber else "piano_solo"
    encoding   = "chordwise" if chordwise else "notewise"
    note_range = "note_range38" if small_nr else "note_range62"

    if args.sample_freq is None:
        sf = "sample_freq4" if chordwise else "sample_freq12"
    else:
        sf = f"sample_freq{args.sample_freq}"

    source = P["data"] / "composers" / encoding / ensemble / note_range / sf
    if not source.exists():
        sys.exit(f"[clara] Fuente no encontrada: {source}\n"
                 f"        Ejecuta primero:  python clara.py encode")

    composers = (args.composers.split(",") if args.composers
                 else [d.name for d in source.iterdir() if d.is_dir()])

    # Limpiar destinos
    for d in [P["train"], P["test"]]:
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)

    print(f"[split] Origen: {source}")
    for composer in composers:
        cdir = source / composer
        if not cdir.exists():
            print(f"  Compositor no encontrado, omitiendo: {composer}")
            continue
        files = list(cdir.glob("*.txt"))
        for f in files:
            if random.random() > args.sample:
                continue
            dst = P["test"] if random.random() < args.tt_split else P["train"]
            shutil.copy(f, dst)

    n_train = len(list(P["train"].glob("*.txt")))
    n_test  = len(list(P["test"].glob("*.txt")))
    print(f"[split] Copiados → train: {n_train}  test: {n_test}")

    if chordwise:
        print("[split] Chordwise: eliminando marcas de duración...")
        _remove_duration(P["train"])
        _remove_duration(P["test"])
    elif args.no_wait:
        print("[split] Notewise: procesando wait markers...")
        _remove_wait(P["train"])
        _remove_wait(P["test"])

    print("[split] Listo.")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ENTRENAMIENTO DEL GENERADOR  (train)
# ═════════════════════════════════════════════════════════════════════════════

def _music_tokenizer(x):
    return x.split(" ")


def _save_params(P, TEXT, md, bs, bptt, em_sz, nh, nl, prefix):
    import pickle
    d = dict(n_tok=md.nt, pad=md.pad_idx, bs=bs, bptt=bptt,
             em_sz=em_sz, nh=nh, nl=nl)
    (P["generator"] / f"{prefix}_params.pkl").write_bytes(pickle.dumps(d))
    (P["generator"] / f"{prefix}_text.pkl").write_bytes(pickle.dumps(TEXT))


def _load_params(P, model_name):
    import pickle
    params = pickle.loads((P["generator"] / f"{model_name}_params.pkl").read_bytes())
    TEXT   = pickle.loads((P["generator"] / f"{model_name}_text.pkl").read_bytes())
    return params, TEXT


def _train_and_save(learner, lr, epochs, fpath):
    print(f"\n  Entrenando → {fpath.name}  (lr={lr})")
    learner.fit_one_cycle(epochs, lr, wd=1e-6)
    import torch
    torch.save(learner.model.state_dict(), fpath)
    print(f"  Guardado → {fpath}")


def cmd_train(args):
    """Entrena el modelo AWD-LSTM generador."""
    # Imports pesados solo cuando se necesitan
    torch   = _require("torch")
    fastai  = _require("fastai")
    from fastai.text.all import (
        TextDataLoaders, language_model_learner, AWD_LSTM,
        Tokenizer, SpacyTokenizer,
    )
    # Usamos torchtext / fastai text API
    # FastAI v2 ofrece TextDataLoaders.from_folder para LM
    from fastai.text.all import TextDataLoaders, language_model_learner, AWD_LSTM
    import pickle

    P = _paths()

    train_dir = P["data"] / args.train_folder
    test_dir  = P["data"] / args.test_folder

    if len(list(train_dir.glob("*.txt"))) < 2:
        sys.exit(f"[train] No hay suficientes archivos en {train_dir}.\n"
                 f"        Ejecuta primero:  python clara.py split")
    if len(list(test_dir.glob("*.txt"))) < 2:
        sys.exit(f"[train] No hay suficientes archivos en {test_dir}.\n"
                 f"        Aumenta --tt-split o ejecuta split de nuevo.")

    # Parámetros del modelo
    bs, bptt   = args.bs, args.bptt
    em_sz, nh, nl = args.em_sz, args.nh, args.nl
    dm         = args.dropout

    # Si se carga un modelo previo, sus parámetros prevalecen
    if args.load:
        params, _ = _load_params(P, args.load)
        bptt, em_sz, nh, nl = params["bptt"], params["em_sz"], params["nh"], params["nl"]
        print(f"[train] Cargando hiperparámetros de '{args.load}'")

    print(f"[train] Construyendo vocabulario desde {P['data']}...")
    dls = TextDataLoaders.from_folder(
        P["data"],
        train=args.train_folder,
        valid=args.test_folder,
        tok_func=_music_tokenizer,
        bs=bs,
        seq_len=bptt,
        min_freq=args.min_freq,
        is_lm=True,
    )
    print(f"[train] Tamaño de vocabulario: {len(dls.vocab)}")

    # Guardar params/TEXT para uso posterior en generate / critic / composer
    # Simulamos el objeto 'md' con duck-typing mínimo
    class _FakeMD:
        nt = len(dls.vocab)
        pad_idx = dls.vocab.index("<pad>") if "<pad>" in dls.vocab else 1
    _save_params(P, dls.vocab, _FakeMD(), bs, bptt, em_sz, nh, nl, args.prefix)

    learn = language_model_learner(
        dls, AWD_LSTM,
        emb_sz=em_sz, n_hid=nh, n_layers=nl,
        drop_mult=dm,
        pretrained=False,
    )

    if args.load:
        model_file = P["generator"] / f"{args.load}_{args.level}.pth"
        learn.model.load_state_dict(torch.load(model_file, map_location="cpu"))
        print(f"[train] Pesos cargados desde {model_file}")

    levels  = ["light", "med", "full", "extra"]
    lrs_map = {"light": 3e-3, "med": 3e-4, "full": 3e-6, "extra": 3e-8}

    for level in levels:
        lr   = lrs_map[level]
        fout = P["generator"] / f"{args.prefix}_{level}.pth"
        _train_and_save(learn, lr, args.epochs, fout)

    print(f"\n[train] Completado. Modelos guardados en {P['generator']}")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — GENERACIÓN  (generate)
# ═════════════════════════════════════════════════════════════════════════════

def _load_lm(P, model_name, level, bs):
    """Carga un modelo de lenguaje preentrenado (FastAI v2)."""
    torch  = _require("torch")
    import pickle
    from fastai.text.all import language_model_learner, AWD_LSTM, TextDataLoaders

    params = pickle.loads((P["generator"] / f"{model_name}_params.pkl").read_bytes())
    vocab  = pickle.loads((P["generator"] / f"{model_name}_text.pkl").read_bytes())

    # Reconstruir learner vacío con la arquitectura correcta
    # (FastAI v2: necesitamos un dls mínimo; cargamos pesos manualmente)
    model_path = P["generator"] / f"{model_name}_{level}.pth"
    if not model_path.exists():
        sys.exit(f"[clara] Modelo no encontrado: {model_path}")

    from fastai.text.all import AWD_LSTM, get_language_model
    from fastai.text.models.awdlstm import AWD_LSTM as AWDLSTM
    # Reconstruir modelo con parámetros guardados
    config = dict(
        emb_sz=params["em_sz"],
        n_hid=params["nh"],
        n_layers=params["nl"],
        pad_token=params["pad"],
        hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5,
    )
    from fastai.text.all import AWD_LSTM
    model = get_language_model(AWD_LSTM, params["n_tok"], config=config)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    model[0].bs = bs
    return model, params, vocab


def _load_prompts(folder):
    """Carga todos los .txt de una carpeta como lista de strings."""
    return [f.read_text() for f in Path(folder).glob("*.txt")]


def _pick_prompts(prompts, bptt, bs):
    """Selecciona bs fragmentos aleatorios de longitud bptt."""
    result = []
    for _ in range(bs):
        for _ in range(100):
            src = random.choice(prompts).split(" ")
            if len(src) > bptt + 1:
                off = random.randint(0, len(src) - bptt - 1)
                result.append(" ".join(src[off:off + bptt]))
                break
        else:
            sys.exit("[generate] No se encontró ningún archivo de prompt "
                     f"con más de {bptt} tokens. Usa --bptt menor.")
    return result


def _generate_batch(model, vocab, params, prompts_text, num_words,
                    rand_freq, trunc, bs, bptt):
    """Genera bs piezas nuevas condicionadas en prompts_text."""
    torch = _require("torch")

    # Tokenizar prompts → índices
    def tok2idx(token):
        try:
            return vocab.index(token)
        except (ValueError, AttributeError):
            # vocab puede ser una lista o un objeto Vocab de fastai
            if hasattr(vocab, "o2i"):
                return vocab.o2i.get(token, 0)
            return 0

    def idx2tok(idx):
        try:
            return vocab[idx]
        except Exception:
            if hasattr(vocab, "itos"):
                return vocab.itos[idx]
            return "<unk>"

    s = [_music_tokenizer(p)[:bptt] for p in prompts_text]
    t = torch.LongTensor([[tok2idx(w) for w in row] for row in s])  # (bs, bptt)

    results = [""] * bs
    with torch.no_grad():
        model.reset()
        # Feed prompt
        for step in range(t.shape[1]):
            res, *_ = model(t[:, step].unsqueeze(0))

        # Generar
        print("[generate] Generando...")
        for _ in tqdm(range(num_words), desc="  tokens"):
            ps, n = res.topk(params["n_tok"])
            w = n[:, 0].clone()
            for j in range(bs):
                if random.random() < rand_freq:
                    ps_j = ps[j, :trunc]
                    r = torch.multinomial(ps_j.exp(), 1)
                    idx = r.item()
                    if idx != 0:
                        w[j] = n[j, idx]
            for j in range(bs):
                results[j] += idx2tok(w[j].item()) + " "
            res, *_ = model(w.unsqueeze(0))

    return prompts_text, results


def _write_midi_and_audio(text, fname, sample_freq, note_offset, out_dir, chordwise):
    """Convierte texto → MIDI → MP3/WAV."""
    music21 = _require("music21")
    stream = _string_to_stream(text, sample_freq, note_offset, chordwise)
    mid_path = out_dir / fname
    stream.write("midi", fp=str(mid_path))
    base = str(mid_path)[:-4]
    os.system(f"./data/mid2mp3.sh {base}.mid")
    os.system(f"mpg123 -w {base}.wav {base}.mp3")


def cmd_generate(args):
    """Genera piezas nuevas con un modelo entrenado."""
    P = _paths()

    model, params, vocab = _load_lm(P, args.model, args.level, args.bs)

    prompt_dir = (P["data"] / args.test_folder if args.use_test_prompt
                  else P["data"] / args.train_folder)
    prompts = _load_prompts(prompt_dir)
    if not prompts:
        sys.exit(f"[generate] Sin archivos de prompt en {prompt_dir}")

    bptt = args.prompt_size or params["bptt"]
    sample_freq = 4 if args.chordwise else 12
    if args.sample_freq:
        sample_freq = args.sample_freq
    note_offset = 45 if args.small_note_range else 33

    print(f"[generate] Modelo: {args.model} ({args.level})  "
          f"bs={args.bs}  tokens={args.size}  "
          f"random_freq={args.random_freq}  trunc={args.trunc}")

    chosen_prompts, results = _generate_batch(
        model, vocab, params, prompts_text=_pick_prompts(prompts, bptt, args.bs),
        num_words=args.size, rand_freq=args.random_freq,
        trunc=args.trunc, bs=args.bs, bptt=bptt,
    )

    out = P["output"] / args.output
    out.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(results):
        fname = str(i).zfill(2) + ".mid"
        (out / (str(i) + ".txt")).write_text(text)
        _write_midi_and_audio(text, fname, sample_freq, note_offset, out,
                              args.chordwise)

    for i, prompt in enumerate(chosen_prompts):
        fname = f"prompt{str(i).zfill(2)}.mid"
        _write_midi_and_audio(prompt, fname, sample_freq, note_offset, out,
                              args.chordwise)

    print(f"[generate] {len(results)} piezas guardadas en {out}")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — CONVERSIÓN TEXTO → STREAM MUSIC21
# (compartida por generate, critic-data y composer-data)
# ═════════════════════════════════════════════════════════════════════════════

def _string_to_stream(text, sample_freq, note_offset, chordwise):
    if chordwise:
        return _arr_to_stream_chordwise(text.split(" "), sample_freq, note_offset)
    return _arr_to_stream_notewise(text.split(" "), sample_freq, note_offset)


def _arr_to_stream_chordwise(score, sample_freq, note_offset):
    music21 = _require("music21")
    speed = 1.0 / sample_freq
    piano_notes, violin_notes = [], []
    for i, tok in enumerate(score):
        if not tok:
            continue
        for j in range(1, len(tok)):
            if tok[j] == "1":
                n = music21.note.Note(j + note_offset)
                n.duration = music21.duration.Duration(2 * speed)
                n.offset = i * speed
                (violin_notes if tok[0] == "v" else piano_notes).append(n)
    violin = music21.instrument.fromString("Violin")
    piano  = music21.instrument.fromString("Piano")
    violin_notes.insert(0, violin)
    piano_notes.insert(0, piano)
    return music21.stream.Stream([
        music21.stream.Stream(violin_notes),
        music21.stream.Stream(piano_notes),
    ])


def _arr_to_stream_notewise(score, sample_freq, note_offset):
    music21 = _require("music21")
    speed = 1.0 / sample_freq
    piano_notes, violin_notes = [], []
    time_offset = 0

    # Expandir octavas abreviadas
    i = 0
    while i < len(score):
        tok = score[i]
        if tok[:9] == "p_octave_":
            eoc = ""
            if tok[-3:] == "eoc":
                eoc = "eoc"
                tok = tok[:-3]
            base = tok[9:]
            score[i] = "p" + base
            score.insert(i + 1, "p" + str(int(base) + 12) + eoc)
            i += 1
        i += 1

    for i, tok in enumerate(score):
        if tok in ("", " ", "<eos>", "<unk>"):
            continue
        if tok[:3] == "end":
            if tok[-3:] == "eoc":
                time_offset += 1
            continue
        if tok[:4] == "wait":
            time_offset += int(tok[4:])
            continue

        duration = 1
        note_len = len(tok)
        for j in range(1, 200):
            if i + j >= len(score):
                break
            nxt = score[i + j]
            if nxt[:4] == "wait":
                duration += int(nxt[4:])
            if (nxt[:3 + note_len] == "end" + tok
                    or nxt[:note_len] == tok):
                break
            if nxt[-3:] == "eoc":
                duration += 1
        else:
            duration = 12

        add_wait = 0
        if tok[-3:] == "eoc":
            tok = tok[:-3]
            add_wait = 1

        try:
            n = music21.note.Note(int(tok[1:]) + note_offset)
            n.duration = music21.duration.Duration(duration * speed)
            n.offset = time_offset * speed
            (violin_notes if tok[0] == "v" else piano_notes).append(n)
        except Exception:
            pass

        time_offset += add_wait

    violin = music21.instrument.fromString("Violin")
    piano  = music21.instrument.fromString("Piano")
    violin_notes.insert(0, violin)
    piano_notes.insert(0, piano)
    return music21.stream.Stream([
        music21.stream.Stream(violin_notes),
        music21.stream.Stream(piano_notes),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — DATOS Y ENTRENAMIENTO DEL CRÍTICO  (critic-data / critic)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_critic_data(args):
    """Genera datos de entrenamiento para el crítico (piezas reales vs. generadas)."""
    P = _paths()

    model, params, vocab = _load_lm(P, args.model, args.level, args.bs)
    prompt_dir = (P["data"] / "test" if args.use_test_prompt
                  else P["data"] / "train")
    prompts = _load_prompts(prompt_dir)

    # Preparar carpetas
    for split in ["train", "test"]:
        for kind in ["real", "fake"]:
            d = P["critic_data"] / split / kind
            d.mkdir(parents=True, exist_ok=True)
            if args.replace:
                for f in d.glob("*.txt"):
                    f.unlink()

    bptt    = params["bptt"]
    n_iters = args.num // args.bs + 1

    for j in range(n_iters):
        print(f"[critic-data] Bloque {j + 1}/{n_iters}")
        chosen, results = _generate_batch(
            model, vocab, params,
            prompts_text=_pick_prompts(prompts, bptt, args.bs),
            num_words=args.size,
            rand_freq=random.random(),
            trunc=random.randint(1, 10),
            bs=args.bs,
            bptt=bptt,
        )
        results_tok = [r.split(" ") for r in results]
        n_mini = args.size // bptt

        for i in range(args.bs):
            split = "test" if random.random() < args.tt_split else "train"
            for mini in range(n_mini):
                fname = f"{args.prefix}{j}_{i}_{mini}.txt"
                (P["critic_data"] / split / "fake" / fname).write_text(
                    " ".join(results_tok[i][mini * bptt:(mini + 1) * bptt])
                )

        real_prompts = _pick_prompts(prompts, bptt, args.bs * n_mini)
        for i, rp in enumerate(real_prompts):
            split = "test" if random.random() < args.tt_split else "train"
            fname = f"{args.prefix}real_{j}_{i}.txt"
            (P["critic_data"] / split / "real" / fname).write_text(rp)

    print(f"[critic-data] Listo → {P['critic_data']}")


def cmd_critic(args):
    """Entrena el clasificador binario real / generado (crítico)."""
    torch  = _require("torch")
    import pickle
    from fastai.text.all import (
        TextDataLoaders, text_classifier_learner, AWD_LSTM,
    )

    P = _paths()
    params = pickle.loads((P["generator"] / f"{args.model}_params.pkl").read_bytes())
    vocab  = pickle.loads((P["generator"] / f"{args.model}_text.pkl").read_bytes())

    print("[critic] Cargando datos...")
    dls = TextDataLoaders.from_folder(
        P["critic_data"],
        valid_pct=params.get("tt_split", 0.1),
        tok_func=_music_tokenizer,
        bs=args.bs,
        seq_len=params["bptt"],
        vocab=vocab,
    )

    learn = text_classifier_learner(
        dls, AWD_LSTM,
        emb_sz=params["em_sz"],
        n_hid=params["nh"],
        n_layers=params["nl"],
        drop_mult=0.5,
        pretrained=False,
    )

    if args.pretrain:
        enc_path = P["generator"] / f"{args.model}_{args.level}"
        learn.load_encoder(str(enc_path))
        learn.freeze_to(-1)
        print("[critic] Fine-tuning de la última capa...")
        _train_and_save(learn, 3e-4, 1, P["critic_dir"] / f"{args.model}_fine_tune.pth")

    levels  = ["light", "med", "full", "extra"]
    lrs_map = {"light": 3e-3, "med": 3e-4, "full": 3e-6, "extra": 3e-8}
    for level in levels:
        _train_and_save(learn, lrs_map[level], args.epochs,
                        P["critic_dir"] / f"{args.model}_{level}.pth")

    print(f"[critic] Completado → {P['critic_dir']}")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — DATOS Y ENTRENAMIENTO DEL CLASIFICADOR DE COMPOSITORES
# (composer-data / composer)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_composer_data(args):
    """Prepara los datos para el clasificador de compositores."""
    P = _paths()
    import pickle

    params = pickle.loads((P["generator"] / f"{args.model}_params.pkl").read_bytes())

    chordwise  = args.chordwise
    chamber    = args.chamber
    small_nr   = args.small_note_range
    ensemble   = "chamber" if chamber else "piano_solo"
    encoding   = "chordwise" if chordwise else "notewise"
    note_range = "note_range38" if small_nr else "note_range62"
    sf = (f"sample_freq{args.sample_freq}" if args.sample_freq
          else ("sample_freq4" if chordwise else "sample_freq12"))

    source = P["data"] / "composers" / encoding / ensemble / note_range / sf
    if not source.exists():
        sys.exit(f"[composer-data] Fuente no encontrada: {source}")

    composers = (args.composers.split(",") if args.composers
                 else [d.name for d in source.iterdir() if d.is_dir()])

    target_train = P["composer_data"] / "train"
    target_test  = P["composer_data"] / "test"
    shutil.rmtree(target_train, ignore_errors=True)
    shutil.rmtree(target_test,  ignore_errors=True)

    bptt = params["bptt"]

    print("[composer-data] Creando dataset...")
    for composer in tqdm(composers):
        cdir = source / composer
        if not cdir.exists():
            continue
        (target_train / composer).mkdir(parents=True, exist_ok=True)
        (target_test  / composer).mkdir(parents=True, exist_ok=True)

        for fpath in cdir.glob("*.txt"):
            if random.random() > args.sample:
                continue
            tokens = fpath.read_text().split(" ")[1:-1]
            for k in range(len(tokens) // bptt):
                dst = (target_test / composer if random.random() < args.tt_split
                       else target_train / composer)
                (dst / f"{k}_{fpath.name}").write_text(
                    " ".join(tokens[k * bptt:(k + 1) * bptt])
                )

    print(f"[composer-data] Listo → {P['composer_data']}")


def cmd_composer(args):
    """Entrena el clasificador de compositores."""
    torch  = _require("torch")
    import pickle
    from fastai.text.all import (
        TextDataLoaders, text_classifier_learner, AWD_LSTM,
    )

    P = _paths()
    params = pickle.loads((P["generator"] / f"{args.model}_params.pkl").read_bytes())
    vocab  = pickle.loads((P["generator"] / f"{args.model}_text.pkl").read_bytes())

    print("[composer] Cargando datos...")
    dls = TextDataLoaders.from_folder(
        P["composer_data"],
        valid_pct=0.05,
        tok_func=_music_tokenizer,
        bs=args.bs,
        seq_len=params["bptt"],
        vocab=vocab,
    )

    from functools import partial
    import torch.optim as optim
    learn = text_classifier_learner(
        dls, AWD_LSTM,
        emb_sz=params["em_sz"],
        n_hid=params["nh"],
        n_layers=params["nl"],
        drop_mult=0.5,
        pretrained=False,
        opt_func=partial(optim.Adam, betas=(0.6, 0.95)),
    )

    levels  = ["light", "med", "full", "extra"]
    lrs_map = {"light": 3e-4, "med": 3e-4, "full": 3e-6, "extra": 3e-8}
    for level in levels:
        _train_and_save(learn, lrs_map[level], args.epochs,
                        P["composer_dir"] / f"{args.model}_{level}.pth")

    print(f"[composer] Completado → {P['composer_dir']}")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — SERVIDOR HTTP  (serve)
# ═════════════════════════════════════════════════════════════════════════════

def _make_playlist_html(music_dir):
    """Genera HTML + CSS + JS para la playlist de audio."""
    music_dir = Path(music_dir)
    entries = sorted(f for f in music_dir.rglob("*.wav")
                     if not f.name.startswith("prom"))
    if not entries:
        return "<p>No hay piezas generadas todavía.</p>", "", ""

    first = entries[0]
    audio = (f'<audio id="audio" controls preload="auto">'
             f'<source src="{first}" type="audio/wav">'
             f'Sorry, your browser does not support HTML5 audio.</audio>')
    items = "".join(
        f'<li class="{"active" if i == 0 else ""}"><a href="{e}">{e.name}</a></li>\n'
        for i, e in enumerate(entries)
    )
    html = audio + f'<ul id="playlist">\n{items}</ul>'
    css  = "#playlist .active a{color:#FF5964;}"
    js   = """
<script>
var audio=document.getElementById('audio');
document.querySelectorAll('#playlist a').forEach(function(a){
  a.addEventListener('click',function(e){
    e.preventDefault();
    document.querySelectorAll('#playlist li').forEach(l=>l.classList.remove('active'));
    a.parentElement.classList.add('active');
    audio.src=a.href; audio.load(); audio.play();
  });
});
audio.addEventListener('ended',function(){
  var active=document.querySelector('#playlist .active');
  var nxt=active?active.nextElementSibling:null;
  if(nxt){nxt.classList.add('active');active.classList.remove('active');
    audio.src=nxt.querySelector('a').href;audio.load();audio.play();}
});
</script>"""
    return html, css, js


def cmd_serve(args):
    """Lanza un servidor HTTP con playlist interactiva (genera piezas on demand)."""
    from http.server import SimpleHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

    P = _paths()
    music_dir = P["output"] / "temp"
    music_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model
    level      = args.level
    rand_freq  = args.random_freq
    trunc      = args.trunc

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith((".wav", ".mid", ".gif")):
                super().do_GET()
                return

            if self.path.endswith("/music"):
                # Generar nuevas piezas
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                try:
                    lm, params, vocab = _load_lm(P, model_name, level, 16)
                    prompts = _load_prompts(P["data"] / "train")
                    _, results = _generate_batch(
                        lm, vocab, params,
                        prompts_text=_pick_prompts(prompts, params["bptt"], 16),
                        num_words=600, rand_freq=rand_freq, trunc=trunc,
                        bs=16, bptt=params["bptt"],
                    )
                    for i, text in enumerate(results):
                        _write_midi_and_audio(text, f"{i:02d}.mid", 4, 33,
                                              music_dir, False)
                except Exception as e:
                    self.wfile.write(f"<p>Error generando: {e}</p>".encode())
                    return

            html, css, js = _make_playlist_html(music_dir)
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            page = (f"<!doctype html><html><head>"
                    f"<title>Clara</title><style>{css}</style></head>"
                    f"<body><h1>Clara — Music Neural Net</h1>"
                    f"{html}{js}</body></html>")
            self.wfile.write(page.encode())

        def log_message(self, fmt, *a):
            pass  # silenciar logs de acceso

    class ThreadedServer(ThreadingMixIn, HTTPServer):
        pass

    addr = ("", args.port)
    httpd = ThreadedServer(addr, Handler)
    print(f"[serve] Servidor en http://localhost:{args.port}")
    print(f"[serve] Ctrl+C para detener.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] Detenido.")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — CLI PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="clara",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMANDO")

    # ── encode ────────────────────────────────────────────────────────────────
    p = sub.add_parser("encode",
        help="MIDI corpus → archivos de texto codificados",
        description="Convierte todos los MIDIs de data/composers/midi/ a texto.")
    p.add_argument("--chamber", action="store_true",
        help="Música de cámara (piano+violín). Default: piano solo.")
    p.add_argument("--composers",
        help="Lista separada por comas. Default: todos.")
    p.add_argument("--replace", action="store_true",
        help="Recodificar aunque ya existan los .txt.")

    # ── split ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("split",
        help="Distribuye .txt codificados en train / test",
        description="Construye data/train y data/test listos para train.")
    p.add_argument("--example", action="store_true",
        help="Usar los datos de ejemplo incluidos (ignora el resto de opciones).")
    p.add_argument("--chordwise", action="store_true",
        help="Codificación chordwise (default: notewise).")
    p.add_argument("--chamber", action="store_true",
        help="Música de cámara.")
    p.add_argument("--small-note-range", dest="small_note_range",
        action="store_true", help="38 notas (default: 62).")
    p.add_argument("--sample-freq", dest="sample_freq", type=int,
        help="Frecuencia de muestreo (default: 4 chordwise / 12 notewise).")
    p.add_argument("--composers",
        help="Compositores a incluir (default: todos).")
    p.add_argument("--tt-split", dest="tt_split", type=float, default=0.1,
        help="Fracción para test (default: 0.1).")
    p.add_argument("--sample", type=float, default=1.0,
        help="Fracción del corpus a usar (default: 1.0).")
    p.add_argument("--no-wait", dest="no_wait", action="store_true",
        help="Recodificar marcas wait (notewise).")

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("train",
        help="Entrena el modelo AWD-LSTM generador",
        description="Entrena el modelo sobre los datos de data/train y data/test.")
    p.add_argument("--prefix", default="mod",
        help="Prefijo para los archivos del modelo (default: mod).")
    p.add_argument("--load",
        help="Nombre de modelo preentrenado desde el que continuar.")
    p.add_argument("--level", default="light",
        choices=["light", "med", "full", "extra"],
        help="Nivel del modelo cargado (default: light).")
    p.add_argument("--train-folder", dest="train_folder", default="train",
        help="Subcarpeta de data/ con los datos de entrenamiento.")
    p.add_argument("--test-folder", dest="test_folder", default="test",
        help="Subcarpeta de data/ con los datos de validación.")
    p.add_argument("--bs", type=int, default=32, help="Batch size (default: 32).")
    p.add_argument("--bptt", type=int, default=200,
        help="Back-prop through time (default: 200).")
    p.add_argument("--em-sz", dest="em_sz", type=int, default=400,
        help="Tamaño de embedding (default: 400).")
    p.add_argument("--nh", type=int, default=600,
        help="Neuronas ocultas por capa (default: 600).")
    p.add_argument("--nl", type=int, default=4,
        help="Número de capas LSTM (default: 4).")
    p.add_argument("--min-freq", dest="min_freq", type=int, default=1,
        help="Frecuencia mínima de token (default: 1).")
    p.add_argument("--dropout", type=float, default=1.0,
        help="Multiplicador de dropout (default: 1.0).")
    p.add_argument("--epochs", type=int, default=3,
        help="Épocas por nivel de entrenamiento (default: 3).")

    # ── generate ──────────────────────────────────────────────────────────────
    p = sub.add_parser("generate",
        help="Genera piezas nuevas con un modelo entrenado",
        description="Genera bs piezas y las guarda en data/output/<output>/.")
    p.add_argument("--model", required=True,
        help="Nombre del modelo (sin sufijo _light.pth, etc.).")
    p.add_argument("--output", required=True,
        help="Carpeta de salida dentro de data/output/.")
    p.add_argument("--level", default="light",
        choices=["light", "med", "full", "extra"],
        help="Nivel del modelo (default: light).")
    p.add_argument("--size", type=int, default=2000,
        help="Tokens a generar (default: 2000).")
    p.add_argument("--bs", type=int, default=16,
        help="Número de piezas a generar en paralelo (default: 16).")
    p.add_argument("--trunc", type=int, default=5,
        help="Top-k para el muestreo (default: 5).")
    p.add_argument("--random-freq", dest="random_freq", type=float, default=0.5,
        help="Fracción de pasos con muestreo aleatorio (default: 0.5).")
    p.add_argument("--sample-freq", dest="sample_freq", type=int,
        help="Override de frecuencia de muestreo MIDI.")
    p.add_argument("--chordwise", action="store_true",
        help="Usar codificación chordwise.")
    p.add_argument("--small-note-range", dest="small_note_range",
        action="store_true", help="Rango de 38 notas.")
    p.add_argument("--use-test-prompt", dest="use_test_prompt",
        action="store_true", help="Usar prompts del conjunto de test.")
    p.add_argument("--prompt-size", dest="prompt_size", type=int,
        help="Tamaño del prompt (default: bptt del modelo).")
    p.add_argument("--train-folder", dest="train_folder", default="train")
    p.add_argument("--test-folder",  dest="test_folder",  default="test")

    # ── critic-data ───────────────────────────────────────────────────────────
    p = sub.add_parser("critic-data",
        help="Genera datos de entrenamiento para el crítico",
        description="Crea critic_data/train/{real,fake} y critic_data/test/{real,fake}.")
    p.add_argument("--model", required=True)
    p.add_argument("--level", default="light",
        choices=["light", "med", "full", "extra"])
    p.add_argument("--num", type=int, default=1000,
        help="Número de fragmentos falsos a generar (default: 1000).")
    p.add_argument("--size", type=int, default=2000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--prefix", default="",
        help="Prefijo para los nombres de archivo.")
    p.add_argument("--replace", action="store_true",
        help="Borrar y reemplazar datos existentes.")
    p.add_argument("--use-test-prompt", dest="use_test_prompt",
        action="store_true")
    p.add_argument("--tt-split", dest="tt_split", type=float, default=0.1)

    # ── critic ────────────────────────────────────────────────────────────────
    p = sub.add_parser("critic",
        help="Entrena el clasificador real / generado",
        description="Entrena un clasificador sobre critic_data/.")
    p.add_argument("--model", required=True)
    p.add_argument("--level", default="light",
        choices=["light", "med", "full", "extra"])
    p.add_argument("--pretrain", action="store_true",
        help="Inicializar con los pesos del generador.")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=32)

    # ── composer-data ─────────────────────────────────────────────────────────
    p = sub.add_parser("composer-data",
        help="Prepara datos para el clasificador de compositores",
        description="Crea composer_data/train/<compositor>/ y composer_data/test/.")
    p.add_argument("--model", required=True,
        help="Modelo generador del que tomar bptt y vocabulario.")
    p.add_argument("--composers")
    p.add_argument("--chordwise", action="store_true")
    p.add_argument("--chamber", action="store_true")
    p.add_argument("--small-note-range", dest="small_note_range",
        action="store_true")
    p.add_argument("--sample-freq", dest="sample_freq", type=int)
    p.add_argument("--tt-split", dest="tt_split", type=float, default=0.05)
    p.add_argument("--sample", type=float, default=1.0)

    # ── composer ──────────────────────────────────────────────────────────────
    p = sub.add_parser("composer",
        help="Entrena el clasificador de compositores",
        description="Clasifica fragmentos por compositor.")
    p.add_argument("--model", required=True)
    p.add_argument("--level", default="light",
        choices=["light", "med", "full", "extra"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=32)

    # ── serve ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("serve",
        help="Lanza un servidor HTTP con playlist interactiva",
        description="Sirve data/output/temp/ y genera piezas on demand.")
    p.add_argument("--model", default="mod")
    p.add_argument("--level", default="light",
        choices=["light", "med", "full", "extra"])
    p.add_argument("--random-freq", dest="random_freq", type=float, default=0.8)
    p.add_argument("--trunc", type=int, default=2)
    p.add_argument("--port", type=int, default=8000)

    # ── dispatch ──────────────────────────────────────────────────────────────
    args = parser.parse_args()

    dispatch = {
        "encode":        cmd_encode,
        "split":         cmd_split,
        "train":         cmd_train,
        "generate":      cmd_generate,
        "critic-data":   cmd_critic_data,
        "critic":        cmd_critic,
        "composer-data": cmd_composer_data,
        "composer":      cmd_composer,
        "serve":         cmd_serve,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(0)

    random.seed(os.urandom(10))
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
