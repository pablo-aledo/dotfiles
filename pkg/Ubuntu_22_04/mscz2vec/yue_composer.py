#!/usr/bin/env python3
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         YUE COMPOSER  v1.0                                   ║
║         Generación de canciones completas con letra mediante YuE (乐)        ║
║                                                                              ║
║  YuE es un modelo open-source de música generativa (lyrics → canción).      ║
║  Dado un texto de letra y tags de género, genera una canción completa con   ║
║  voz y acompañamiento separados, de varios minutos de duración.              ║
║                                                                              ║
║  ARQUITECTURA INTERNA:                                                       ║
║    Stage 1 (7B)  — LM que genera tokens de audio comprimidos (1 codebook)   ║
║    Stage 2 (1B)  — Upsampler: expande a 8 codebooks (mayor fidelidad)       ║
║    XCodec Mini   — Codec neuronal que convierte tokens ↔ audio              ║
║    Decoders      — Dos decoders separados: voz e instrumental               ║
║                                                                              ║
║  COMANDOS:                                                                   ║
║    install      — Descarga modelos de HuggingFace y dependencias            ║
║    generate     — Letras + género → canción completa (modo CoT)             ║
║    icl          — Como generate pero con audio de referencia (ICL)          ║
║    tags         — Busca tags válidos en el vocabulario de YuE               ║
║    check        — Verifica entorno: CUDA, modelos, dependencias             ║
║                                                                              ║
║  USO RÁPIDO:                                                                 ║
║    # 1. Instalar modelos (una sola vez):                                     ║
║    python yue_composer.py install                                            ║
║    python yue_composer.py install --model-dir ~/mis_modelos/yue/            ║
║                                                                              ║
║    # 2. Generar una canción:                                                 ║
║    python yue_composer.py generate --genre genre.txt --lyrics lyrics.txt    ║
║    python yue_composer.py generate --genre genre.txt --lyrics lyrics.txt \  ║
║        --segments 4 --output cancion.wav --listen                           ║
║                                                                              ║
║    # 3. Generar con referencia de estilo (ICL):                             ║
║    python yue_composer.py icl --genre genre.txt --lyrics lyrics.txt \       ║
║        --vocal ref_voz.mp3 --instrumental ref_inst.mp3                      ║
║    python yue_composer.py icl --genre genre.txt --lyrics lyrics.txt \       ║
║        --audio ref.mp3 --start 0 --end 30                                   ║
║                                                                              ║
║    # 4. Buscar tags de género:                                               ║
║    python yue_composer.py tags tango                                         ║
║    python yue_composer.py tags --list-all                                    ║
║                                                                              ║
║    # 5. Verificar instalación:                                               ║
║    python yue_composer.py check                                              ║
║                                                                              ║
║  OPCIONES GENERATE / ICL:                                                   ║
║    --genre FILE        Fichero de texto con tags de género (requerido)      ║
║    --lyrics FILE       Fichero de texto con la letra (requerido)            ║
║    --segments N        Número de segmentos ~30s a generar (default: 2)      ║
║    --output FILE       Fichero de salida WAV (default: yue_output.wav)      ║
║    --keep-stems        Conservar pistas vocal e instrumental separadas      ║
║    --language LANG     Idioma: en | zh | jp-kr (default: en)                ║
║    --mode cot|icl      Checkpoint a usar (default: cot en generate)         ║
║    --max-tokens N      Tokens máximos por segmento (default: 3000)          ║
║    --rep-penalty F     Penalización de repetición 1.0–2.0 (default: 1.1)   ║
║    --seed N            Semilla aleatoria (default: 42)                      ║
║    --batch-size N      Batch size del stage 2 (default: 4)                  ║
║    --cuda N            Índice de GPU a usar (default: 0)                    ║
║    --listen            Reproducir el resultado al terminar                  ║
║    --verbose           Informe detallado de progreso                        ║
║                                                                              ║
║  OPCIONES ICL (adicionales):                                                ║
║    --vocal FILE        Pista vocal de referencia (~30s, modo dual-track)    ║
║    --instrumental FILE Pista instrumental de referencia (modo dual-track)   ║
║    --audio FILE        Audio de referencia unificado (modo single-track)    ║
║    --start F           Tiempo inicio del fragmento de referencia (def: 0)   ║
║    --end F             Tiempo fin del fragmento de referencia (def: 30)     ║
║                                                                              ║
║  OPCIONES INSTALL:                                                           ║
║    --model-dir DIR     Directorio donde instalar modelos (def: ~/.yue/)     ║
║    --language LANG     Solo descargar modelos de un idioma: en|zh|jp-kr     ║
║    --no-flash-attn     No instalar flash-attention (más lento, menos VRAM) ║
║                                                                              ║
║  CONFIGURACIÓN DE RUTA DE MODELOS:                                          ║
║    Por orden de prioridad:                                                   ║
║      1. Flag --model-dir en la línea de comandos                            ║
║      2. Variable de entorno YUE_MODEL_DIR                                   ║
║      3. Ruta por defecto: ~/.yue/                                            ║
║                                                                              ║
║  REQUISITOS DE HARDWARE:                                                     ║
║    GPU ≥ 24 GB VRAM  →  hasta 2 segmentos (~1 min de canción)               ║
║    GPU ≥ 80 GB VRAM  →  canciones largas (4+ segmentos)                     ║
║    H800: 30s audio ≈ 150s cómputo  |  RTX 4090: 30s ≈ 360s                 ║
║    Flash Attention 2 muy recomendado para evitar OOM en secuencias largas   ║
║                                                                              ║
║  DEPENDENCIAS:                                                               ║
║    pip install torch torchaudio transformers sentencepiece                   ║
║    pip install omegaconf einops soundfile numpy tqdm accelerate             ║
║    pip install huggingface_hub                                               ║
║    pip install flash-attn --no-build-isolation   (opcional, recomendado)    ║
║    pip install python-audio-separator             (opcional, para --split)  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import textwrap
import shutil
import subprocess
import re
import random
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0"

# Ruta por defecto de modelos (puede sobreescribirse con --model-dir o YUE_MODEL_DIR)
DEFAULT_MODEL_DIR = Path.home() / ".yue"

# Repositorios en HuggingFace
HF_REPOS = {
    "s1_en_cot":   "m-a-p/YuE-s1-7B-anneal-en-cot",
    "s1_en_icl":   "m-a-p/YuE-s1-7B-anneal-en-icl",
    "s1_zh_cot":   "m-a-p/YuE-s1-7B-anneal-zh-cot",
    "s1_zh_icl":   "m-a-p/YuE-s1-7B-anneal-zh-icl",
    "s1_jpkr_cot": "m-a-p/YuE-s1-7B-anneal-jp-kr-cot",
    "s1_jpkr_icl": "m-a-p/YuE-s1-7B-anneal-jp-kr-icl",
    "s2":          "m-a-p/YuE-s2-1B-general",
    "xcodec":      "m-a-p/xcodec_mini_infer",
}

# Tokenizador (incluido en el repo de xcodec)
TOKENIZER_SUBPATH = "mm_tokenizer_v0.2_hf/tokenizer.model"

# Mapeo idioma → sufijo de modelo
LANG_TO_SUFFIX = {
    "en":    "en",
    "zh":    "zh",
    "jp-kr": "jpkr",
}

# Top-200 tags integrados (subset representativo — install descarga el JSON completo)
BUILTIN_TAGS = [
    "pop", "rock", "jazz", "blues", "electronic", "classical", "folk",
    "country", "r&b", "soul", "hip-hop", "metal", "punk", "indie",
    "ambient", "tango", "flamenco", "bossa nova", "reggae", "latin",
    "orchestral", "cinematic", "dark", "bright", "melancholic", "uplifting",
    "energetic", "calm", "aggressive", "romantic", "mysterious", "epic",
    "nostalgic", "playful", "tense", "ethereal", "groovy", "hypnotic",
    "piano", "guitar", "strings", "brass", "woodwind", "drums", "bass",
    "synthesizer", "violin", "cello", "flute", "trumpet", "saxophone",
    "female", "male", "vocal", "instrumental", "choir", "airy vocal",
    "raspy vocal", "bright vocal", "deep vocal", "falsetto",
    "fast", "slow", "moderate", "120bpm", "90bpm", "140bpm",
    "major", "minor", "modal", "atonal", "pentatonic",
    "4/4", "3/4", "6/8", "waltz", "swing",
    "verse", "chorus", "bridge", "outro",
    "Mandarin", "Cantonese", "Japanese", "Korean",
]


def _resolve_model_dir(args_model_dir: str | None) -> Path:
    """Resuelve la ruta de modelos por orden de prioridad."""
    if args_model_dir:
        return Path(args_model_dir).expanduser()
    env = os.environ.get("YUE_MODEL_DIR")
    if env:
        return Path(env).expanduser()
    return DEFAULT_MODEL_DIR


def _check_model_dir(model_dir: Path) -> dict:
    """
    Verifica qué componentes están instalados en model_dir.
    Devuelve dict con claves: tokenizer, xcodec, s2, s1_* → bool.
    """
    status = {}
    import glob as _glob
    _tok_hits = _glob.glob(str(model_dir / "**" / "tokenizer.model"), recursive=True)
    status["tokenizer"] = len(_tok_hits) > 0
    status["xcodec"]    = (model_dir / "xcodec_mini_infer" / "final_ckpt" / "ckpt_00360000.pth").exists()
    status["s2"]        = (model_dir / "s2" / "config.json").exists()
    for lang in ("en", "zh", "jpkr"):
        for mode in ("cot", "icl"):
            key = f"s1_{lang}_{mode}"
            status[key] = (model_dir / key / "config.json").exists()
    return status


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE AUDIO Y FICHEROS
# ══════════════════════════════════════════════════════════════════════════════

def _find_tokenizer(model_dir: Path):
    """Busca tokenizer.model bajo model_dir, prefiriendo s1_en_cot y s2."""
    candidates = [
        model_dir / "mm_tokenizer_v0.2_hf" / "tokenizer.model",
        model_dir / "xcodec_mini_infer" / "mm_tokenizer_v0.2_hf" / "tokenizer.model",
        model_dir / "s1_en_cot" / "tokenizer.model",
        model_dir / "s2" / "tokenizer.model",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    import glob
    hits = glob.glob(str(model_dir / "**" / "tokenizer.model"), recursive=True)
    hits = [h for h in hits if ".cache" not in h and "__pycache__" not in h]
    preferred = [h for h in hits if any(s in h for s in ("s1_en", "s1_zh", "s2/", "s2\\"))]
    return (preferred or hits or [None])[0]


def _load_audio_mono(path: str, sr: int = 16000):
    """Carga audio a mono 16kHz. Devuelve (tensor, sr)."""
    try:
        import torchaudio, torch
        audio, orig_sr = torchaudio.load(path)
        audio = torch.mean(audio, dim=0, keepdim=True)
        if orig_sr != sr:
            from torchaudio.transforms import Resample
            audio = Resample(orig_freq=orig_sr, new_freq=sr)(audio)
        return audio, sr
    except ImportError:
        import soundfile as sf, numpy as np
        audio, orig_sr = sf.read(path, always_2d=True)
        audio = audio.mean(axis=1)
        if orig_sr != sr:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, orig_sr)
            audio = resample_poly(audio, sr // g, orig_sr // g).astype("float32")
        import torch
        return torch.tensor(audio).unsqueeze(0), sr


def _listen(path: str, verbose: bool = False):
    """Reproduce un fichero de audio usando el player disponible."""
    players = ["mpg123", "ffplay", "aplay", "mplayer", "cvlc"]
    for player in players:
        if shutil.which(player):
            if verbose:
                print(f"[listen] Reproduciendo con {player}: {path}")
            try:
                if player == "ffplay":
                    subprocess.run([player, "-nodisp", "-autoexit", path],
                                   check=True, capture_output=not verbose)
                elif player == "cvlc":
                    subprocess.run([player, "--play-and-exit", path],
                                   check=True, capture_output=not verbose)
                else:
                    subprocess.run([player, path],
                                   check=True, capture_output=not verbose)
                return
            except subprocess.CalledProcessError:
                continue
    print(f"[listen] ⚠  No se encontró player de audio.")
    print(f"[listen]    Fichero generado en: {path}")


def _split_lyrics(lyrics: str) -> list:
    """Divide la letra en segmentos estructurados por etiquetas [verse]/[chorus]/etc."""
    import re
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    if not segments:
        paragraphs = [p.strip() for p in lyrics.split("\n\n") if p.strip()]
        labels = ["verse", "chorus"] * (len(paragraphs) // 2 + 1)
        return [f"[{labels[i]}]\n{p}\n\n" for i, p in enumerate(paragraphs)]
    return [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]


def _read_genre(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def _read_lyrics(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return _split_lyrics(f.read())


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MÓDULOS INTERNOS DE XCODEC (importación dinámica)
# ══════════════════════════════════════════════════════════════════════════════

def _setup_xcodec_imports(xcodec_dir: Path, model_dir: Path):
    """
    Añade al sys.path todos los lugares donde YuE puede tener sus módulos Python.
    infer.py original asume que se ejecuta desde el directorio inference/ del repo,
    donde mmtokenizer.py y codecmanipulator.py están en el mismo nivel.
    Aquí los buscamos en varios lugares posibles.
    """
    # Directorio del propio script yue_composer.py (por si el usuario copió los .py allí)
    script_dir = Path(__file__).parent

    # Buscar mmtokenizer.py recursivamente bajo model_dir
    import glob
    mm_hits = glob.glob(str(model_dir / "**" / "mmtokenizer.py"), recursive=True)
    mm_hits += glob.glob(str(script_dir / "**" / "mmtokenizer.py"), recursive=True)
    mm_dirs = list(dict.fromkeys(str(Path(h).parent) for h in mm_hits))  # sin duplicados

    paths = mm_dirs + [
        str(xcodec_dir),
        str(xcodec_dir / "descriptaudiocodec"),
        str(xcodec_dir / "models"),
        str(script_dir),
    ]
    for p in paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _import_yue_modules(xcodec_dir: Path, model_dir: Path):
    """
    Importa los módulos internos de YuE/xcodec una vez configurado el path.
    mmtokenizer.py y codecmanipulator.py pueden estar en el repo descargado
    junto a infer.py, NO dentro de xcodec_mini_infer.
    """
    _setup_xcodec_imports(xcodec_dir, model_dir)

    # Diagnóstico de paths si falla
    import glob
    mm_hits = glob.glob(str(model_dir / "**" / "mmtokenizer.py"), recursive=True)
    script_dir = Path(__file__).parent
    mm_hits += glob.glob(str(script_dir / "**" / "mmtokenizer.py"), recursive=True)

    if not mm_hits:
        # Intentar descarga automática de los módulos desde GitHub
        print("  [import] mmtokenizer.py no encontrado localmente.")
        print("  [import] Descargando módulos de inferencia desde GitHub...")
        import urllib.request
        base_url = "https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/inference/"
        modules = ["mmtokenizer.py", "codecmanipulator.py",
                   "vocoder.py", "post_process_audio.py"]
        dest_dir = xcodec_dir
        for mod in modules:
            dst = dest_dir / mod
            if not dst.exists():
                try:
                    urllib.request.urlretrieve(base_url + mod, str(dst))
                    print(f"    ✓  {mod}")
                except Exception as e:
                    print(f"    ⚠  No se pudo descargar {mod}: {e}")
        sys.path.insert(0, str(dest_dir))

    # ══ PASO 1: instalar MetaPathFinder ANTES de cualquier import ══════════
    # descriptaudiocodec y audiotools se interceptan completamente en memoria.
    # El finder debe estar en sys.meta_path[0] antes de que Python intente
    # buscar ningún fichero de estos paquetes.
    import types as _types, importlib as _il
    import importlib.abc as _ilabc, importlib.machinery as _ilm
    import torch.nn as _nn

    class _BaseModel(_nn.Module):
        INTERN = []; EXTERN = []
        def __init__(self, *a, **kw):
            try: super().__init__()
            except Exception: pass
        def forward(self, *a, **kw): return None
        def __getattr__(self, k):
            try: return super().__getattr__(k)
            except AttributeError: return _BaseModel()

    class _MockMod(_types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__path__ = []
            self.__package__ = name
            self.BaseModel = _BaseModel
            _ml = _types.ModuleType(name + ".ml")
            _ml.__path__ = []
            _ml.BaseModel = _BaseModel
            object.__setattr__(self, "ml", _ml)
        def __getattr__(self, k):
            if k.startswith("__"): raise AttributeError(k)
            if k and k[0].isupper():
                cls = type(k, (_BaseModel,), {})
                object.__setattr__(self, k, cls)
                return cls
            child = _MockMod(self.__name__ + "." + k)
            object.__setattr__(self, k, child)
            return child

    _INTERCEPT = ("descriptaudiocodec", "audiotools")

    class _MockLoader(_ilabc.Loader):
        def create_module(self, spec):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            m = _MockMod(spec.name)
            sys.modules[spec.name] = m
            return m
        def exec_module(self, module): pass

    class _MockFinder(_ilabc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith(_INTERCEPT):
                spec = _ilm.ModuleSpec(fullname, _MockLoader(), is_package=True)
                spec.submodule_search_locations = []
                return spec
            return None

    # Limpiar sys.modules PRIMERO — eliminar cualquier versión previa
    # (incluyendo el módulo no-paquete que quedó de ejecuciones anteriores)
    for _k in list(sys.modules.keys()):
        if _k.startswith(_INTERCEPT) or any(x in _k for x in
                ["soundstream", "quantization", "utils.utils",
                 "vocoder", "models.sound", "dac"]):
            del sys.modules[_k]

    # Instalar el finder AL FRENTE de sys.meta_path
    sys.meta_path = [f for f in sys.meta_path
                     if not isinstance(f, _MockFinder)]
    sys.meta_path.insert(0, _MockFinder())

    # ══ PASO 2: parchear utils/utils.py (imports de entrenamiento) ══════════
    _utils_py = xcodec_dir / "utils" / "utils.py"
    if _utils_py.exists():
        _src = _utils_py.read_text(encoding="utf-8")
        _bad = ["import torch.utils.tensorboard", "import tensorboard",
                "import matplotlib", "matplotlib.use(",
                "from matplotlib", "import librosa", "from librosa",
                "import wandb", "from wandb"]
        if any(b in _src for b in _bad) and "# [yue_composer patched]" not in _src:
            _pl = []
            for _ln in _src.splitlines():
                _s = _ln.strip()
                if any(_s.startswith(b) for b in _bad):
                    _pl.append("# [yue_composer patched] " + _ln)
                else:
                    _pl.append(_ln)
            _utils_py.write_text("\n".join(_pl), encoding="utf-8")
            print("  [import] utils/utils.py parcheado")

    # ══ PASO 3: revertir parches de disco de descriptaudiocodec/soundstream ═
    # Ya no necesitamos esos parches — el finder lo gestiona todo en memoria.
    for _f in [
        xcodec_dir / "descriptaudiocodec" / "dac" / "__init__.py",
        xcodec_dir / "models" / "soundstream_hubert_new.py",
    ]:
        if _f.exists():
            _src = _f.read_text(encoding="utf-8")
            if "# [yue_composer patched]" in _src:
                _reverted = "\n".join(
                    _ln[len("# [yue_composer patched] "):]
                    if _ln.startswith("# [yue_composer patched] ")
                    else _ln
                    for _ln in _src.splitlines()
                )
                _f.write_text(_reverted, encoding="utf-8")
                print(f"  [import] {_f.name} revertido")

    try:
        from mmtokenizer import _MMSentencePieceTokenizer
        from codecmanipulator import CodecManipulator
        from vocoder import build_codec_model, process_audio
        from models.soundstream_hubert_new import SoundStream
        try:
            from post_process_audio import replace_low_freq_with_energy_matched
        except ImportError:
            replace_low_freq_with_energy_matched = None
        return (_MMSentencePieceTokenizer, CodecManipulator,
                build_codec_model, process_audio,
                replace_low_freq_with_energy_matched, SoundStream)
    except ImportError as e:
        searched = [p for p in sys.path if "yue" in p.lower() or "xcodec" in p.lower()]
        raise ImportError(
            f"No se pudieron importar módulos internos de YuE.\n"
            f"Error: {e}\n"
            f"Paths buscados: {searched}\n"
            f"\n"
            f"Solución: copia mmtokenizer.py y codecmanipulator.py al directorio\n"
            f"del script, o al directorio de modelos:\n"
            f"  cp /ruta/al/repo/YuE/inference/mmtokenizer.py {xcodec_dir}/\n"
            f"  cp /ruta/al/repo/YuE/inference/codecmanipulator.py {xcodec_dir}/\n"
            f"  cp /ruta/al/repo/YuE/inference/vocoder.py {xcodec_dir}/\n"
            f"  cp /ruta/al/repo/YuE/inference/post_process_audio.py {xcodec_dir}/"
        )




# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLETO DE GENERACIÓN  (fiel a infer.py de YuE)
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(
    genre: str,
    lyrics_segments: list,
    model_dir: Path,
    stage1_model_id: str,
    stage2_model_id: str,
    output_path: str,
    n_segments: int = 2,
    max_new_tokens: int = 3000,
    repetition_penalty: float = 1.1,
    stage2_batch_size: int = 4,
    seed: int = 42,
    cuda_idx: int = 0,
    keep_stems: bool = False,
    audio_prompt: dict = None,
    disable_offload: bool = False,
    verbose: bool = False,
):
    import numpy as np
    from einops import rearrange

    try:
        import torch
        import torchaudio
        from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError(f"Dependencia faltante: {e}")

    # ── Semilla ───────────────────────────────────────────────────────────────
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"[pipeline] Device: {device}")

    # ── Importar módulos internos de YuE ──────────────────────────────────────
    xcodec_dir = model_dir / "xcodec_mini_infer"
    (MMSentencePieceTokenizer, CodecManipulator,
     build_codec_model, process_audio,
     replace_low_freq, SoundStream) = _import_yue_modules(xcodec_dir, model_dir)

    # ── Tokenizador ───────────────────────────────────────────────────────────
    tokenizer_path = _find_tokenizer(model_dir)
    if not tokenizer_path:
        raise FileNotFoundError(
            f"tokenizer.model no encontrado bajo {model_dir}\n"
            f"Prueba: find {model_dir} -name tokenizer.model\n"
            f"O reinstala: python yue_composer.py install --model-dir {model_dir}"
        )
    if verbose:
        print(f"[pipeline] Tokenizador: {tokenizer_path}")
    mmtokenizer = MMSentencePieceTokenizer(tokenizer_path)

    # ── XCodec (encoder) ─────────────────────────────────────────────────────
    config_path  = xcodec_dir / "final_ckpt" / "config.yaml"
    ckpt_path    = xcodec_dir / "final_ckpt" / "ckpt_00360000.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"XCodec checkpoint no encontrado: {ckpt_path}\n"
            f"Reinstala: python yue_composer.py install --model-dir {model_dir}"
        )
    if verbose:
        print(f"[xcodec] Cargando codec desde {ckpt_path}")
    model_config = OmegaConf.load(str(config_path))
    # SoundStream usa rutas relativas (./xcodec_mini_infer/...) en su __init__
    # Hay que ejecutarlo desde el directorio de modelos para que funcionen
    _orig_cwd = os.getcwd()
    try:
        os.chdir(str(model_dir))
        codec_model = eval(model_config.generator.name)(**model_config.generator.config)
    finally:
        os.chdir(_orig_cwd)
    codec_model = codec_model.to(device)
    param_dict   = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    result = codec_model.load_state_dict(param_dict["codec_model"], strict=False)
    if result.missing_keys and verbose:
        print(f"  [xcodec] Claves faltantes en codec: {len(result.missing_keys)}")
    if result.unexpected_keys and verbose:
        print(f"  [xcodec] Claves inesperadas en codec: {len(result.unexpected_keys)}")
    codec_model.to(device)
    codec_model.eval()

    # ── Decoders Vocos ────────────────────────────────────────────────────────
    decoder_cfg_path    = xcodec_dir / "decoders" / "config.yaml"
    vocal_decoder_path  = xcodec_dir / "decoders" / "decoder_131000.pth"
    inst_decoder_path   = xcodec_dir / "decoders" / "decoder_151000.pth"
    if verbose:
        print("[xcodec] Cargando decoders vocal + instrumental")
    # build_codec_model de vocoder.py usa torch.load sin map_location.
    # Lo reimplementamos inline para forzar map_location="cpu".
    try:
        from vocos import Vocos
        def _load_vocos_decoder(cfg_path, ckpt_path):
            dec = Vocos.from_hparams(str(cfg_path))
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            dec.load_state_dict(state)
            dec.eval()
            return dec
        vocal_decoder = _load_vocos_decoder(decoder_cfg_path, vocal_decoder_path)
        inst_decoder  = _load_vocos_decoder(decoder_cfg_path, inst_decoder_path)
    except Exception as _e:
        if verbose:
            print(f"  [vocos] Fallo al cargar Vocos ({_e}), usando build_codec_model")
        # Parchear vocoder.py para que use map_location
        _vocoder_py = xcodec_dir / "vocoder.py"
        if _vocoder_py.exists():
            _vsrc = _vocoder_py.read_text(encoding="utf-8")
            if "torch.load(" in _vsrc and "map_location" not in _vsrc:
                _vsrc = _vsrc.replace(
                    "torch.load(vocal_decoder_path)",
                    'torch.load(vocal_decoder_path, map_location="cpu", weights_only=False)'
                ).replace(
                    "torch.load(inst_decoder_path)",
                    'torch.load(inst_decoder_path, map_location="cpu", weights_only=False)'
                )
                _vocoder_py.write_text(_vsrc, encoding="utf-8")
                print("  [import] vocoder.py parcheado (map_location=cpu)")
                # Recargar módulo con parche
                for _k in list(sys.modules.keys()):
                    if "vocoder" in _k:
                        del sys.modules[_k]
                from vocoder import build_codec_model as _bcm
                vocal_decoder, inst_decoder = _bcm(
                    str(decoder_cfg_path), str(vocal_decoder_path), str(inst_decoder_path)
                )
            else:
                vocal_decoder, inst_decoder = build_codec_model(
                    str(decoder_cfg_path), str(vocal_decoder_path), str(inst_decoder_path)
                )
        else:
            raise

    # ── CodecManipulator ──────────────────────────────────────────────────────
    codectool       = CodecManipulator("xcodec", 0, 1)
    codectool_stage2 = CodecManipulator("xcodec", 0, 8)

    # ── Stage 1: LM 7B ───────────────────────────────────────────────────────
    print(f"\n[stage1] Cargando {stage1_model_id} ...")
    stage1_kwargs = {"torch_dtype": torch.bfloat16}
    try:
        import flash_attn  # noqa
        stage1_kwargs["attn_implementation"] = "flash_attention_2"
        if verbose:
            print("[stage1] Flash Attention 2 habilitado")
    except ImportError:
        pass
    model = AutoModelForCausalLM.from_pretrained(stage1_model_id, **stage1_kwargs)
    model.to(device)
    model.eval()
    if torch.__version__ >= "2.0.0":
        model = torch.compile(model)

    # ── Prompt de audio ICL ───────────────────────────────────────────────────
    def _encode_audio(tensor, bw=0.5):
        if len(tensor.shape) < 3:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            codes = codec_model.encode(tensor.to(device), target_bw=bw)
        return codes.transpose(0, 1).cpu().numpy().astype(np.int16)

    # ── Tokens especiales de bloqueo ─────────────────────────────────────────
    class _BlockRange(LogitsProcessor):
        def __init__(self, s, e):
            self.blocked = list(range(s, e))
        def __call__(self, input_ids, scores):
            scores[:, self.blocked] = float("-inf")
            return scores

    start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
    end_of_segment   = mmtokenizer.tokenize("[end_of_segment]")

    # ── Construir prompt base ─────────────────────────────────────────────────
    full_lyrics   = "\n".join(lyrics_segments)
    prompt_texts  = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genre}\n{full_lyrics}"]
    prompt_texts += lyrics_segments

    top_p       = 0.93
    temperature = 1.0
    run_n_segs  = min(n_segments + 1, len(prompt_texts))
    raw_output  = None

    print(f"[stage1] Generando {run_n_segs - 1} segmento(s)...")
    for i, p in enumerate(prompt_texts[:run_n_segs]):
        if i == 0:
            continue

        section_text   = p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        guidance_scale = 1.5 if i <= 1 else 1.2

        if i == 1:
            # Prompt ICL (audio de referencia)
            if audio_prompt:
                if "vocal" in audio_prompt and "inst" in audio_prompt:
                    vocal_audio, _ = _load_audio_mono(audio_prompt["vocal"]) if isinstance(audio_prompt["vocal"], str) else (audio_prompt["vocal"], None)
                    inst_audio,  _ = _load_audio_mono(audio_prompt["inst"])  if isinstance(audio_prompt["inst"],  str) else (audio_prompt["inst"],  None)
                    vocals_ids      = codectool.npy2ids(_encode_audio(vocal_audio)[0])
                    inst_ids        = codectool.npy2ids(_encode_audio(inst_audio)[0])
                    interleaved     = rearrange([np.array(vocals_ids), np.array(inst_ids)], "b n -> (n b)")
                    t0 = int(audio_prompt.get("start", 0) * 50 * 2)
                    t1 = int(audio_prompt.get("end",   30) * 50 * 2)
                    audio_codec_ids = interleaved[t0:t1].tolist()
                else:
                    ref_audio, _ = _load_audio_mono(audio_prompt["audio"]) if isinstance(audio_prompt["audio"], str) else (audio_prompt["audio"], None)
                    raw_codes       = _encode_audio(ref_audio)
                    code_ids        = codectool.npy2ids(raw_codes[0])
                    t0 = int(audio_prompt.get("start", 0) * 50)
                    t1 = int(audio_prompt.get("end",   30) * 50)
                    audio_codec_ids = code_ids[t0:t1]
                audio_prompt_ids = ([mmtokenizer.soa] + codectool.sep_ids
                                    + audio_codec_ids + [mmtokenizer.eoa])
                ref_ids   = (mmtokenizer.tokenize("[start_of_reference]")
                             + audio_prompt_ids
                             + mmtokenizer.tokenize("[end_of_reference]"))
                head_id   = mmtokenizer.tokenize(prompt_texts[0]) + ref_ids
            else:
                head_id = mmtokenizer.tokenize(prompt_texts[0])

            prompt_ids = (head_id + start_of_segment
                          + mmtokenizer.tokenize(section_text)
                          + [mmtokenizer.soa] + codectool.sep_ids)
        else:
            prompt_ids = (end_of_segment + start_of_segment
                          + mmtokenizer.tokenize(section_text)
                          + [mmtokenizer.soa] + codectool.sep_ids)

        prompt_tensor = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        input_ids     = (torch.cat([raw_output, prompt_tensor], dim=1)
                         if raw_output is not None and i > 1 else prompt_tensor)

        # Recortar contexto si supera el límite
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            if verbose:
                print(f"  [stage1] seg {i}: contexto {input_ids.shape[-1]} > {max_context}, recortando")
            input_ids = input_ids[:, -max_context:]

        print(f"  [stage1] Segmento {i}/{run_n_segs - 1} ...")
        with torch.no_grad():
            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([
                    _BlockRange(0, 32002),
                    _BlockRange(32016, 32016),
                ]),
                guidance_scale=guidance_scale,
            )
        if output_seq[0][-1].item() != mmtokenizer.eoa:
            output_seq = torch.cat(
                [output_seq, torch.as_tensor([[mmtokenizer.eoa]]).to(device)], dim=1
            )

        if raw_output is not None and i > 1:
            raw_output = torch.cat([raw_output, prompt_tensor,
                                    output_seq[:, input_ids.shape[-1]:]], dim=1)
        else:
            raw_output = output_seq

    # ── Extraer vocal / instrumental de Stage 1 ───────────────────────────────
    ids     = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()

    if len(soa_idx) != len(eoa_idx):
        print(f"[pipeline] ⚠  soa/eoa desbalanceados ({len(soa_idx)}/{len(eoa_idx)}), ajustando...")
        min_len = min(len(soa_idx), len(eoa_idx))
        soa_idx = soa_idx[:min_len]
        eoa_idx = eoa_idx[:min_len]

    vocals_list = []
    insts_list  = []
    range_begin = 1 if audio_prompt else 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
        if len(codec_ids) > 0 and codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[:2 * (len(codec_ids) // 2)]
        if len(codec_ids) == 0:
            continue
        codec_2d    = rearrange(codec_ids, "(n b) -> b n", b=2)
        vocals_list.append(codectool.ids2npy(codec_2d[0]))
        insts_list.append(codectool.ids2npy(codec_2d[1]))

    if not vocals_list:
        raise RuntimeError("Stage 1 no generó tokens de audio válidos.")

    vocals = np.concatenate(vocals_list, axis=1)
    insts  = np.concatenate(insts_list,  axis=1)

    # ── Offload Stage 1 ───────────────────────────────────────────────────────
    if not disable_offload:
        model.cpu()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if verbose:
            print("[pipeline] Stage 1 liberado de GPU")

    # ── Stage 2: upsampling ───────────────────────────────────────────────────
    print(f"\n[stage2] Cargando {stage2_model_id} ...")
    s2_kwargs = {"torch_dtype": torch.bfloat16}
    try:
        import flash_attn  # noqa
        s2_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass
    model_stage2 = AutoModelForCausalLM.from_pretrained(stage2_model_id, **s2_kwargs)
    model_stage2.to(device)
    model_stage2.eval()
    if torch.__version__ >= "2.0.0":
        model_stage2 = torch.compile(model_stage2)

    def _stage2_generate(prompt_npy):
        """Upsampling de 1 codebook a 8. Fiel a stage2_generate() de infer.py."""
        codec_ids = codectool_stage2.unflatten(prompt_npy, n_quantizer=1)
        codec_ids = codectool_stage2.offset_tok_ids(
            codec_ids,
            global_offset=codectool_stage2.global_offset,
            codebook_size=codectool_stage2.codebook_size,
            num_codebooks=codectool_stage2.num_codebooks,
        ).astype(np.int32)

        bs = stage2_batch_size
        if bs > 1:
            codec_list = [codec_ids[:, i*300:(i+1)*300] for i in range(bs)]
            codec_ids_b = np.concatenate(codec_list, axis=0)
            prompt_ids  = np.concatenate([
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (bs, 1)),
                codec_ids_b,
                np.tile([mmtokenizer.stage_2], (bs, 1)),
            ], axis=1)
        else:
            prompt_ids = np.concatenate([
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),
                np.array([mmtokenizer.stage_2]),
            ]).astype(np.int32)[np.newaxis, ...]

        codec_ids_t = torch.as_tensor(codec_ids).to(device)
        prompt_ids_t = torch.as_tensor(prompt_ids).to(device)

        block_list = LogitsProcessorList([
            _BlockRange(0, 46358),
            _BlockRange(53526, mmtokenizer.vocab_size),
        ])

        generated = []
        for fi in range(codec_ids_t.shape[1]):
            cb0 = codec_ids_t[:, fi:fi+1]
            prompt_ids_t = torch.cat([prompt_ids_t, cb0], dim=1)
            with torch.no_grad():
                out = model_stage2.generate(
                    input_ids=prompt_ids_t,
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=block_list,
                )
            generated.append(out[:, prompt_ids_t.shape[-1]:].cpu().numpy())
            prompt_ids_t = out

        if not generated:
            return np.zeros((8, 0), dtype=np.int16)
        generated = np.concatenate(generated, axis=1)
        # Unoffset
        generated = codectool_stage2.unoffset_tok_ids(
            generated,
            global_offset=codectool_stage2.global_offset,
            codebook_size=codectool_stage2.codebook_size,
            num_codebooks=codectool_stage2.num_codebooks,
        )
        return generated

    print("[stage2] Generando pista vocal...")
    vocals_s2 = _stage2_generate(vocals)
    print("[stage2] Generando pista instrumental...")
    insts_s2  = _stage2_generate(insts)

    # Liberar Stage 2
    model_stage2.cpu()
    del model_stage2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Decodificar con XCodec → audio ────────────────────────────────────────
    print("\n[xcodec] Decodificando a audio...")
    output_p    = Path(output_path)
    sample_rate = 44100

    def _decode_track(codes_npy, decoder, label):
        t = torch.as_tensor(codes_npy.astype(np.int16), dtype=torch.long)
        t = t.unsqueeze(0).permute(1, 0, 2).to(device)
        with torch.no_grad():
            wav = codec_model.decode(t)
        wav = wav.cpu().squeeze(0)
        if verbose:
            print(f"  [{label}] {wav.shape[-1]/sample_rate:.1f}s")
        return wav

    # Usar vocoder Vocos para mayor calidad (post-processing del paper)
    def _process_track(npy_codes, decoder, label):
        return process_audio(npy_codes, decoder, device)

    try:
        vocal_wav = _process_track(vocals_s2, vocal_decoder, "vocal")
        inst_wav  = _process_track(insts_s2,  inst_decoder,  "inst")
    except Exception as e:
        if verbose:
            print(f"  [vocoder] Vocos falló ({e}), usando decoder básico")
        vocal_wav = _decode_track(vocals_s2, vocal_decoder, "vocal")
        inst_wav  = _decode_track(insts_s2,  inst_decoder,  "inst")

    # Mezcla
    min_len   = min(vocal_wav.shape[-1], inst_wav.shape[-1])
    mix_wav   = (vocal_wav[..., :min_len] + inst_wav[..., :min_len]) * 0.5
    limit     = 0.99
    max_val   = mix_wav.abs().max()
    mix_wav   = mix_wav * min(limit / max_val, 1) if max_val > limit else mix_wav.clamp(-limit, limit)

    # Guardar
    torchaudio.save(str(output_p), mix_wav, sample_rate=sample_rate,
                    encoding="PCM_S", bits_per_sample=16)
    print(f"\n✓  Canción guardada en: {output_p}")

    if keep_stems:
        stem_v = output_p.with_name(output_p.stem + "_vocal" + output_p.suffix)
        stem_i = output_p.with_name(output_p.stem + "_inst"  + output_p.suffix)
        torchaudio.save(str(stem_v), vocal_wav, sample_rate=sample_rate,
                        encoding="PCM_S", bits_per_sample=16)
        torchaudio.save(str(stem_i), inst_wav,  sample_rate=sample_rate,
                        encoding="PCM_S", bits_per_sample=16)
        print(f"   Voz:           {stem_v}")
        print(f"   Instrumental:  {stem_i}")

    duration = mix_wav.shape[-1] / sample_rate
    print(f"   Duración: {duration:.1f}s  ({duration/60:.1f} min)")
    return str(output_p)


def cmd_install(args):
    model_dir = _resolve_model_dir(getattr(args, "model_dir", None))
    language  = getattr(args, "language", None)
    no_flash  = getattr(args, "no_flash_attn", False)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║             YuE Composer — Instalación de modelos            ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"  Directorio de modelos: {model_dir}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Verificar dependencias de Python
    print("\n[1/4] Verificando dependencias Python...")
    deps_ok = True
    required = ["torch", "torchaudio", "transformers", "sentencepiece",
                "omegaconf", "einops", "soundfile", "numpy", "tqdm",
                "scipy", "audiotools", "accelerate", "huggingface_hub"]
    missing = []
    for dep in required:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing.append(dep)
            deps_ok = False

    if missing:
        print(f"  ⚠  Dependencias faltantes: {', '.join(missing)}")
        print(f"     Instalando con pip...")
        in_venv = sys.prefix != sys.base_prefix
        pip_cmd = [sys.executable, "-m", "pip", "install"] + missing
        if not in_venv:
            pip_cmd.append("--break-system-packages")
        result = subprocess.run(pip_cmd)
        if result.returncode != 0:
            print()
            print("  ✗  No se pudieron instalar las dependencias automáticamente.")
            print("     Opciones:")
            print("       1. Crear un venv (recomendado):")
            print("            python3 -m venv ~/venvs/yue")
            print("            source ~/venvs/yue/bin/activate")
            print("            python yue_composer.py install")
            print()
            print("       2. Instalar directamente (puede afectar al sistema):")
            print(f"            pip install {' '.join(missing)} --break-system-packages")
            sys.exit(1)
        print("  ✓  Dependencias instaladas")
    else:
        print("  ✓  Todas las dependencias presentes")

    # Flash Attention (opcional)
    if not no_flash:
        try:
            import flash_attn
            print("  ✓  Flash Attention 2 ya instalado")
        except ImportError:
            print("\n  Flash Attention 2 no detectada.")
            print("  Instalando (puede tardar varios minutos)...")
            in_venv = sys.prefix != sys.base_prefix
            fa_cmd = [sys.executable, "-m", "pip", "install",
                      "flash-attn", "--no-build-isolation"]
            if not in_venv:
                fa_cmd.append("--break-system-packages")
            result = subprocess.run(fa_cmd, capture_output=False)
            if result.returncode == 0:
                print("  ✓  Flash Attention 2 instalada")
            else:
                print("  ⚠  No se pudo instalar Flash Attention 2.")
                print("  ⚠  No se pudo instalar Flash Attention 2.")
                print("     Causa probable: conflicto entre la versión CUDA del driver")
                print("     y la versión CUDA con la que se compiló PyTorch.")
                print("     Soluciones: actualizar driver, reinstalar PyTorch con CUDA correcto,")
                print("     o añadir --no-flash-attn para omitirla.")
                print("     La generación funcionará igualmente, con algo más de VRAM.")
        print("  ⚠  Flash Attention 2 omitida (--no-flash-attn)")

    # Descargar modelos con huggingface_hub
    print("\n[2/4] Descargando tokenizador y XCodec...")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub no instalado: pip install huggingface_hub")

    # XCodec + tokenizador (siempre necesarios)
    xcodec_dir = model_dir / "xcodec_mini_infer"
    if not xcodec_dir.exists() or not (xcodec_dir / "final_ckpt" / "ckpt_00360000.pth").exists():
        print(f"  Descargando XCodec Mini ({HF_REPOS['xcodec']})...")
        snapshot_download(
            repo_id=HF_REPOS["xcodec"],
            local_dir=str(xcodec_dir),
            ignore_patterns=["*.md", "*.txt"],
        )
        print(f"  ✓  XCodec descargado en {xcodec_dir}")
    else:
        print(f"  ✓  XCodec ya presente en {xcodec_dir}")

    # Tokenizador — buscar dentro del xcodec descargado y copiar a model_dir
    import glob as _glob
    existing_tok = _glob.glob(str(model_dir / "**" / "tokenizer.model"), recursive=True)
    if not existing_tok:
        # Buscar en el directorio de xcodec recién descargado
        tok_hits = _glob.glob(str(xcodec_dir / "**" / "tokenizer.model"), recursive=True)
        if tok_hits:
            tok_src_dir = Path(tok_hits[0]).parent
            tok_dst_dir = model_dir / tok_src_dir.name
            if not tok_dst_dir.exists():
                shutil.copytree(str(tok_src_dir), str(tok_dst_dir))
            print(f"  ✓  Tokenizador copiado a {tok_dst_dir}")
        else:
            print("  ⚠  tokenizer.model no encontrado en xcodec_mini_infer")
            print("     Puede que HuggingFace haya cambiado la estructura del repo.")
            print(f"     Busca manualmente: find {xcodec_dir} -name tokenizer.model")
    else:
        print(f"  ✓  Tokenizador ya presente: {existing_tok[0]}")
    # Stage 2 (siempre necesario)
    print("\n[3/4] Descargando Stage 2 (upsampler 1B)...")
    s2_dir = model_dir / "s2"
    if not s2_dir.exists() or not (s2_dir / "config.json").exists():
        print(f"  Descargando {HF_REPOS['s2']}...")
        snapshot_download(
            repo_id=HF_REPOS["s2"],
            local_dir=str(s2_dir),
            ignore_patterns=["*.md"],
        )
        print(f"  ✓  Stage 2 descargado en {s2_dir}")
    else:
        print(f"  ✓  Stage 2 ya presente en {s2_dir}")

    # Stage 1 según idioma
    print("\n[4/4] Descargando Stage 1 (LM 7B)...")
    langs_to_install = [language] if language else ["en", "zh", "jp-kr"]
    modes_to_install = ["cot", "icl"]

    for lang in langs_to_install:
        suffix = LANG_TO_SUFFIX.get(lang, lang.replace("-", ""))
        for mode in modes_to_install:
            repo_key = f"s1_{suffix}_{mode}"
            if repo_key not in HF_REPOS:
                continue
            local_key = f"s1_{suffix}_{mode}"
            local_dir = model_dir / local_key
            repo_id   = HF_REPOS[repo_key]
            if not local_dir.exists() or not (local_dir / "config.json").exists():
                print(f"  Descargando {repo_id}  (puede tardar, ~14 GB)...")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    ignore_patterns=["*.md"],
                )
                print(f"  ✓  {local_key} descargado en {local_dir}")
            else:
                print(f"  ✓  {local_key} ya presente en {local_dir}")


    # Módulos de inferencia de YuE (mmtokenizer.py, codecmanipulator.py, etc.)
    # Estos ficheros están en inference/ del repo GitHub, no en HuggingFace.
    print("\n  Descargando módulos de inferencia de YuE...")
    import urllib.request
    base_url = "https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/inference/"
    inference_modules = [
        "mmtokenizer.py", "codecmanipulator.py",
        "vocoder.py", "post_process_audio.py",
    ]
    infer_dir = model_dir / "xcodec_mini_infer"
    for mod in inference_modules:
        dst = infer_dir / mod
        if not dst.exists():
            try:
                urllib.request.urlretrieve(base_url + mod, str(dst))
                print(f"  ✓  {mod}")
            except Exception as e:
                print(f"  ⚠  No se pudo descargar {mod}: {e}")
        else:
            print(f"  ✓  {mod} ya presente")

    # Tags JSON
    # Tags JSON — descargado del repo GitHub oficial de YuE
    tags_path = model_dir / "top_200_tags.json"
    if not tags_path.exists():
        print("\n  Descargando top_200_tags.json...")
        try:
            import urllib.request
            url = ("https://raw.githubusercontent.com/"
                   "multimodal-art-projection/YuE/main/inference/top_200_tags.json")
            urllib.request.urlretrieve(url, str(tags_path))
            print(f"  ✓  Tags descargados en {tags_path}")
        except Exception as e:
            # No es crítico, usamos los BUILTIN_TAGS
            print(f"  ⚠  No se pudo descargar top_200_tags.json ({e}), usando tags integrados")

    print(f"║  ✓  Instalación completa                                      ║")
    print(f"║     Directorio: {str(model_dir):<46}║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  Prueba con:")
    print(f"    python yue_composer.py check")
    print(f"    python yue_composer.py generate --genre genre.txt --lyrics lyrics.txt")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: generate
# ══════════════════════════════════════════════════════════════════════════════

def cmd_generate(args):
    model_dir  = _resolve_model_dir(getattr(args, "model_dir", None))
    language   = getattr(args, "language", "en")
    mode       = getattr(args, "mode", "cot")
    verbose    = getattr(args, "verbose", False)

    lang_suffix = LANG_TO_SUFFIX.get(language, language.replace("-", ""))
    s1_key      = f"s1_{lang_suffix}_{mode}"
    s1_repo     = HF_REPOS.get(s1_key, f"m-a-p/YuE-s1-7B-anneal-{language}-{mode}")

    # Resolver path Stage 1: primero local, luego HuggingFace
    s1_local = model_dir / s1_key
    stage1_id = str(s1_local) if (s1_local / "config.json").exists() else s1_repo

    s2_local = model_dir / "s2"
    stage2_id = str(s2_local) if (s2_local / "config.json").exists() else HF_REPOS["s2"]

    genre   = _read_genre(args.genre)
    lyrics  = _read_lyrics(args.lyrics)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  YuE Composer — Generación (CoT)                             ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"  Género:    {genre[:70]}")
    print(f"  Segmentos: {len(lyrics)} en letra  →  generando {args.segments}")
    print(f"  Stage 1:   {stage1_id}")
    print(f"  Stage 2:   {stage2_id}")
    print(f"  Salida:    {args.output}")

    output_path = _run_pipeline(
        genre=genre,
        lyrics_segments=lyrics,
        model_dir=model_dir,
        stage1_model_id=stage1_id,
        stage2_model_id=stage2_id,
        output_path=args.output,
        n_segments=args.segments,
        max_new_tokens=args.max_tokens,
        repetition_penalty=args.rep_penalty,
        stage2_batch_size=args.batch_size,
        seed=args.seed,
        cuda_idx=args.cuda,
        keep_stems=args.keep_stems,
        disable_offload=getattr(args, "disable_offload", False),
        verbose=verbose,
    )

    if args.listen:
        _listen(output_path, verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: icl (in-context learning con audio de referencia)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_icl(args):
    model_dir   = _resolve_model_dir(getattr(args, "model_dir", None))
    language    = getattr(args, "language", "en")
    verbose     = getattr(args, "verbose", False)

    lang_suffix = LANG_TO_SUFFIX.get(language, language.replace("-", ""))
    s1_key      = f"s1_{lang_suffix}_icl"
    s1_repo     = HF_REPOS.get(s1_key, f"m-a-p/YuE-s1-7B-anneal-{language}-icl")

    s1_local  = model_dir / s1_key
    stage1_id = str(s1_local) if (s1_local / "config.json").exists() else s1_repo

    s2_local  = model_dir / "s2"
    stage2_id = str(s2_local) if (s2_local / "config.json").exists() else HF_REPOS["s2"]

    genre  = _read_genre(args.genre)
    lyrics = _read_lyrics(args.lyrics)

    # Cargar audio de referencia
    audio_prompt = {}
    if args.vocal and args.instrumental:
        print(f"[icl] Modo dual-track: vocal={args.vocal}  inst={args.instrumental}")
        audio_vocal, _ = _load_audio_mono(args.vocal)
        audio_inst,  _ = _load_audio_mono(args.instrumental)
        audio_prompt = {
            "vocal": audio_vocal,
            "inst":  audio_inst,
            "start": args.start,
            "end":   args.end,
        }
    elif args.audio:
        print(f"[icl] Modo single-track: {args.audio}")
        audio, _ = _load_audio_mono(args.audio)
        audio_prompt = {
            "audio": audio,
            "start": args.start,
            "end":   args.end,
        }
    else:
        print("[icl] ⚠  No se proporcionó audio de referencia. "
              "Usa --vocal+--instrumental o --audio.")
        sys.exit(1)

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  YuE Composer — Generación con referencia (ICL)              ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"  Género:    {genre[:70]}")
    print(f"  Segmentos: {len(lyrics)} en letra  →  generando {args.segments}")
    print(f"  Referencia: {args.start:.0f}s – {args.end:.0f}s")
    print(f"  Stage 1:   {stage1_id}")
    print(f"  Salida:    {args.output}")

    output_path = _run_pipeline(
        genre=genre,
        lyrics_segments=lyrics,
        model_dir=model_dir,
        stage1_model_id=stage1_id,
        stage2_model_id=stage2_id,
        output_path=args.output,
        n_segments=args.segments,
        max_new_tokens=args.max_tokens,
        repetition_penalty=args.rep_penalty,
        stage2_batch_size=args.batch_size,
        seed=args.seed,
        cuda_idx=args.cuda,
        keep_stems=args.keep_stems,
        audio_prompt=audio_prompt,
        disable_offload=getattr(args, "disable_offload", False),
        verbose=verbose,
    )

    if args.listen:
        _listen(output_path, verbose=verbose)


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: tags
# ══════════════════════════════════════════════════════════════════════════════

def cmd_tags(args):
    model_dir = _resolve_model_dir(getattr(args, "model_dir", None))
    query     = getattr(args, "query", None)
    list_all  = getattr(args, "list_all", False)

    # Intentar cargar el JSON completo si está disponible
    tags_path = model_dir / "top_200_tags.json"
    if tags_path.exists():
        with open(tags_path, encoding="utf-8") as f:
            data = json.load(f)
        # El JSON puede ser dict o lista
        if isinstance(data, dict):
            all_tags = list(data.keys())
        else:
            all_tags = list(data)
    else:
        all_tags = BUILTIN_TAGS

    if list_all:
        # Mostrar en columnas
        print(f"Tags disponibles ({len(all_tags)} total):\n")
        col_w = 22
        cols  = 4
        rows  = (len(all_tags) + cols - 1) // cols
        for r in range(rows):
            row_tags = [all_tags[r + c * rows] for c in range(cols)
                        if r + c * rows < len(all_tags)]
            print("  " + "".join(t.ljust(col_w) for t in row_tags))
        return

    if query:
        q = query.lower()
        matches = [t for t in all_tags if q in t.lower()]
        if matches:
            print(f"Tags que contienen '{query}':\n")
            for t in matches:
                print(f"  {t}")
            print(f"\n  ({len(matches)} resultados de {len(all_tags)} tags)")
        else:
            print(f"No se encontraron tags que contengan '{query}'.")
            # Sugerir similares por distancia Levenshtein simple
            def _dist(a, b):
                if len(a) > len(b): a, b = b, a
                row = list(range(len(a) + 1))
                for c in b:
                    nr = [row[0] + 1]
                    for j, cc in enumerate(a):
                        nr.append(min(nr[-1]+1, row[j+1]+1, row[j]+(c!=cc)))
                    row = nr
                return row[-1]
            close = sorted(all_tags, key=lambda t: _dist(q, t.lower()))[:5]
            print(f"\n  Quizás quisiste decir: {', '.join(close)}")
        return

    # Sin argumentos: mostrar categorías
    categories = {
        "Géneros":    [t for t in all_tags if t in
                       {"pop","rock","jazz","blues","electronic","classical","folk",
                        "country","r&b","soul","hip-hop","metal","punk","indie",
                        "ambient","tango","flamenco","bossa nova","reggae","latin",
                        "orchestral","cinematic"}],
        "Estado de ánimo": [t for t in all_tags if t in
                       {"dark","bright","melancholic","uplifting","energetic","calm",
                        "aggressive","romantic","mysterious","epic","nostalgic",
                        "playful","tense","ethereal","groovy","hypnotic"}],
        "Instrumentos":    [t for t in all_tags if t in
                       {"piano","guitar","strings","brass","woodwind","drums","bass",
                        "synthesizer","violin","cello","flute","trumpet","saxophone"}],
        "Voz":        [t for t in all_tags if "vocal" in t.lower() or t in
                       {"female","male","instrumental","choir","falsetto"}],
        "Idioma":     [t for t in all_tags if t in
                       {"Mandarin","Cantonese","Japanese","Korean"}],
    }
    print("Tags de género de YuE  (usa 'tags BÚSQUEDA' o 'tags --list-all')\n")
    for cat, items in categories.items():
        if items:
            print(f"  {cat}:")
            print("    " + "  ".join(items))
            print()
    print("  Ejemplo de prompt de género:")
    print('    "inspiring female uplifting pop airy vocal electronic bright vocal vocal"')
    print()
    print("  Estructura recomendada:  género  instrumento  estado-de-ánimo  género-vocal  timbre")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDO: check
# ══════════════════════════════════════════════════════════════════════════════

def cmd_check(args):
    model_dir = _resolve_model_dir(getattr(args, "model_dir", None))

    # Auto-detectar si el directorio default está vacío pero hay otra ruta común
    if not model_dir.exists() or not any(model_dir.iterdir() if model_dir.exists() else []):
        candidates = [
            Path.home() / "Personal" / "yue",
            Path.home() / "yue",
            Path.home() / "models" / "yue",
            Path.home() / "modelos" / "yue",
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "xcodec_mini_infer").exists():
                print(f"  ℹ  Directorio default vacío, usando ruta detectada: {candidate}")
                print(f"     (Pasa --model-dir {candidate} o define YUE_MODEL_DIR para evitar este aviso)\n")
                model_dir = candidate
                break

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  YuE Composer — Verificación del entorno                     ║")
    print(f"╚══════════════════════════════════════════════════════════════╝\n")

    # Python
    print(f"  Python:      {sys.version.split()[0]}")

    # CUDA — diagnóstico detallado
    cuda_ok = False
    try:
        import torch
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()
            cuda_warns = [str(w.message) for w in caught if "CUDA" in str(w.message)]

        if cuda_available:
            cuda_ok = True
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}:       {name}  ({mem:.1f} GB VRAM)")
                if mem < 24:
                    print(f"             ⚠  < 24 GB: solo 1 segmento recomendado")
                elif mem < 80:
                    print(f"             ✓  ≥ 24 GB: hasta 2 segmentos (~1 min)")
                else:
                    print(f"             ✓  ≥ 80 GB: canciones largas (4+ segmentos)")
        else:
            print("  GPU:         ✗  CUDA no disponible")
            if cuda_warns:
                # Detectar mismatch de driver
                for w in cuda_warns:
                    if "too old" in w or "driver" in w.lower():
                        import re
                        ver_match = re.search(r"found version (\d+)", w)
                        driver_ver = ver_match.group(1) if ver_match else "?"
                        # Convertir formato NNNNN → X.Y
                        if len(driver_ver) == 5:
                            major = int(driver_ver[:3])
                            minor = int(driver_ver[3:])
                            driver_str = f"{major}.{minor:02d}"
                        else:
                            driver_str = driver_ver
                        torch_cuda = getattr(torch.version, "cuda", "?")
                        print(f"             Driver NVIDIA detectado: CUDA {driver_str}")
                        print(f"             PyTorch compilado para:  CUDA {torch_cuda}")
                        print(f"")
                        print(f"             Soluciones (elige una):")
                        print(f"               A) Actualizar driver NVIDIA:")
                        print(f"                    sudo apt install nvidia-driver-560")
                        print(f"                    sudo reboot")
                        print(f"               B) Reinstalar PyTorch para CUDA {driver_str}:")
                        print(f"                    pip install torch torchaudio --index-url \\")
                        print(f"                      https://download.pytorch.org/whl/cu128")
                        print(f"               C) Generar en CPU (muy lento, pero funcional):")
                        print(f"                    python yue_composer.py generate ... --cuda -1")
                    elif "mismatch" in w.lower():
                        print(f"             ⚠  {w[:120]}")
            else:
                print(f"             ⚠  Sin GPU disponible. La generación en CPU tomará horas.")

        print(f"  PyTorch:     {torch.__version__}")
        torch_cuda = getattr(torch.version, "cuda", None)
        if torch_cuda:
            print(f"  CUDA build:  {torch_cuda}")
    except ImportError:
        print("  PyTorch:     ✗  NO INSTALADO")

    # Dependencias
    print()
    deps = {
        "transformers":     "transformers",
        "sentencepiece":    "sentencepiece",
        "omegaconf":        "omegaconf",
        "einops":           "einops",
        "soundfile":        "soundfile",
        "scipy":             "scipy",
        "audiotools":        "audiotools",
        "torchaudio":       "torchaudio",
        "accelerate":       "accelerate",
        "huggingface_hub":  "huggingface_hub",
    }
    optional = {
        "flash_attn":       "flash-attn (recomendado: menor VRAM)",
        "audio_separator":  "python-audio-separator (para separar stems)",
    }

    print("  Dependencias requeridas:")
    for mod, label in deps.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"    ✓  {label:<22} {ver}")
        except ImportError:
            print(f"    ✗  {label:<22} NO INSTALADO  →  pip install {label}")

    print("\n  Dependencias opcionales:")
    for mod, label in optional.items():
        try:
            __import__(mod)
            print(f"    ✓  {label}")
        except ImportError:
            print(f"    —  {label}")

    # Modelos
    print(f"\n  Directorio de modelos: {model_dir}")
    status = _check_model_dir(model_dir)

    components = [
        ("tokenizer",    "Tokenizador"),
        ("xcodec",       "XCodec Mini (codec + decoders)"),
        ("s2",           "Stage 2 — upsampler 1B"),
        ("s1_en_cot",    "Stage 1 — inglés CoT (7B)"),
        ("s1_en_icl",    "Stage 1 — inglés ICL (7B)"),
        ("s1_zh_cot",    "Stage 1 — chino CoT (7B)"),
        ("s1_jpkr_cot",  "Stage 1 — japonés/coreano CoT (7B)"),
    ]
    print("\n  Modelos instalados:")
    for key, label in components:
        ok = status.get(key, False)
        mark = "✓" if ok else "—"
        print(f"    {mark}  {label}")

    # Mínimo necesario para generate
    min_ok = status.get("tokenizer") and status.get("xcodec") and status.get("s2")
    has_s1 = any(status.get(f"s1_{lang}_cot") for lang in ["en", "zh", "jpkr"])

    print()
    if min_ok and has_s1:
        if cuda_ok:
            print("  ✓  Entorno listo para generar canciones")
        else:
            print("  ⚠  Modelos OK, pero sin GPU — resuelve el problema CUDA antes de generar")
    elif not min_ok:
        print("  ✗  Instalación incompleta")
        print(f"     Ejecuta: python yue_composer.py install --model-dir {model_dir}")
    else:
        print("  ⚠  Faltan modelos Stage 1")
        print(f"     Ejecuta: python yue_composer.py install --model-dir {model_dir}")

    # Players de audio
    players = ["mpg123", "ffplay", "aplay", "mplayer", "cvlc"]
    found = [p for p in players if shutil.which(p)]
    if found:
        print(f"\n  Players de audio disponibles: {', '.join(found)}")
    else:
        print(f"\n  ⚠  No se encontró player de audio para --listen")
        print(f"     Instala uno: sudo apt install mpg123")

    # Recordatorio ruta
    if str(model_dir) != str(DEFAULT_MODEL_DIR):
        print(f"\n  💡 Para no tener que pasar --model-dir cada vez:")
        print(f"       export YUE_MODEL_DIR={model_dir}")
        print(f"     (añádelo a tu ~/.bashrc para que sea permanente)")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL PARSER CLI
# ══════════════════════════════════════════════════════════════════════════════

def _common_generate_args(p):
    """Añade los argumentos compartidos por generate e icl."""
    p.add_argument("--genre",    required=True,  metavar="FILE",
                   help="Fichero de texto con tags de género")
    p.add_argument("--lyrics",   required=True,  metavar="FILE",
                   help="Fichero de texto con la letra (con etiquetas [verse] etc.)")
    p.add_argument("--segments", type=int,   default=2,    metavar="N",
                   help="Número de segmentos ~30s a generar (default: 2)")
    p.add_argument("--output",   default="yue_output.wav", metavar="FILE",
                   help="Fichero WAV de salida (default: yue_output.wav)")
    p.add_argument("--keep-stems", action="store_true", dest="keep_stems",
                   help="Conservar pistas vocal e instrumental separadas")
    p.add_argument("--language", default="en",
                   choices=["en", "zh", "jp-kr"],
                   help="Idioma de los modelos Stage 1 (default: en)")
    p.add_argument("--max-tokens", type=int, default=3000, dest="max_tokens",
                   metavar="N",
                   help="Tokens máximos por segmento (default: 3000)")
    p.add_argument("--rep-penalty", type=float, default=1.1, dest="rep_penalty",
                   metavar="F",
                   help="Penalización de repetición 1.0–2.0 (default: 1.1)")
    p.add_argument("--seed",     type=int,   default=42,   metavar="N",
                   help="Semilla aleatoria (default: 42)")
    p.add_argument("--batch-size", type=int, default=4, dest="batch_size",
                   metavar="N",
                   help="Batch size del stage 2 (default: 4)")
    p.add_argument("--cuda",     type=int,   default=0,    metavar="N",
                   help="Índice de GPU (default: 0)")
    p.add_argument("--model-dir", default=None, metavar="DIR", dest="model_dir",
                   help="Directorio de modelos (default: ~/.yue/ o YUE_MODEL_DIR)")
    p.add_argument("--disable-offload", action="store_true", dest="disable_offload",
                   help="No mover Stage 1 a CPU tras la generación")
    p.add_argument("--listen",   action="store_true",
                   help="Reproducir el resultado al terminar")
    p.add_argument("--verbose",  action="store_true",
                   help="Informe detallado de progreso")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yue_composer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            YuE Composer v{ver} — Generación de canciones con letra (lyrics → song)
            Modelo open-source por HKUST/M-A-P  |  Apache 2.0
        """.format(ver=VERSION)),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMANDO")
    sub.required = True

    # ── install ────────────────────────────────────────────────────────────────
    p_inst = sub.add_parser(
        "install",
        help="Descarga modelos de HuggingFace y verifica dependencias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Descarga todos los modelos necesarios de HuggingFace y verifica
            las dependencias de Python.  Solo necesitas ejecutarlo una vez.

            Los modelos Stage 1 tienen ~14 GB por variante; Stage 2 ~2 GB;
            XCodec Mini ~500 MB.

            Ejemplos:
              install
              install --model-dir ~/mis_modelos/yue/
              install --language en          # solo inglés (cot + icl)
              install --no-flash-attn        # omitir Flash Attention 2
        """),
    )
    p_inst.add_argument("--model-dir", default=None, metavar="DIR", dest="model_dir",
                        help=f"Directorio de instalación (default: {DEFAULT_MODEL_DIR})")
    p_inst.add_argument("--language", default=None, choices=["en", "zh", "jp-kr"],
                        metavar="LANG",
                        help="Solo descargar modelos de un idioma (default: todos)")
    p_inst.add_argument("--no-flash-attn", action="store_true", dest="no_flash_attn",
                        help="No instalar Flash Attention 2")
    p_inst.set_defaults(func=cmd_install)

    # ── generate ───────────────────────────────────────────────────────────────
    p_gen = sub.add_parser(
        "generate",
        help="Genera una canción desde letras y tags de género (modo CoT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Genera una canción completa en modo CoT (Chain-of-Thought).
            Sin audio de referencia: máxima diversidad creativa.

            Ejemplos:
              generate --genre genre.txt --lyrics lyrics.txt
              generate --genre genre.txt --lyrics lyrics.txt --segments 4 --listen
              generate --genre genre.txt --lyrics lyrics.txt \\
                  --language en --output mi_cancion.wav --keep-stems
        """),
    )
    _common_generate_args(p_gen)
    p_gen.add_argument("--mode", default="cot", choices=["cot", "icl"],
                       help="Checkpoint a usar: cot (default) | icl")
    p_gen.set_defaults(func=cmd_generate)

    # ── icl ────────────────────────────────────────────────────────────────────
    p_icl = sub.add_parser(
        "icl",
        help="Genera condicionado a un audio de referencia (in-context learning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Genera una canción usando un fragmento de audio de referencia (~30s)
            para guiar el estilo.  Soporta dos modos:

              Dual-track (mejor calidad):
                icl --genre g.txt --lyrics l.txt \\
                    --vocal ref_voz.mp3 --instrumental ref_inst.mp3

              Single-track (mix/voz/instrumental):
                icl --genre g.txt --lyrics l.txt --audio ref.mp3

            Para separar vocal e instrumental de un audio usa:
              pip install python-audio-separator
              audio-separator ref.mp3 --model_filename 2stems.onnx

            Ejemplos:
              icl --genre genre.txt --lyrics lyrics.txt \\
                  --vocal pop_vocals.mp3 --instrumental pop_inst.mp3 --listen
              icl --genre genre.txt --lyrics lyrics.txt \\
                  --audio referencia.mp3 --start 15 --end 45
        """),
    )
    _common_generate_args(p_icl)
    p_icl.add_argument("--vocal",        default=None, metavar="FILE",
                       help="Pista vocal de referencia (modo dual-track)")
    p_icl.add_argument("--instrumental", default=None, metavar="FILE",
                       help="Pista instrumental de referencia (modo dual-track)")
    p_icl.add_argument("--audio",        default=None, metavar="FILE",
                       help="Audio de referencia unificado (modo single-track)")
    p_icl.add_argument("--start", type=float, default=0.0, metavar="F",
                       help="Tiempo inicio del fragmento de referencia en segundos (default: 0)")
    p_icl.add_argument("--end",   type=float, default=30.0, metavar="F",
                       help="Tiempo fin del fragmento de referencia en segundos (default: 30)")
    p_icl.set_defaults(func=cmd_icl)

    # ── tags ───────────────────────────────────────────────────────────────────
    p_tags = sub.add_parser(
        "tags",
        help="Busca tags válidos en el vocabulario de YuE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Muestra y filtra los tags disponibles para el prompt de género.

            Ejemplos:
              tags                    # muestra categorías principales
              tags tango              # busca tags que contienen 'tango'
              tags --list-all         # lista todos los tags disponibles
        """),
    )
    p_tags.add_argument("query", nargs="?", default=None, metavar="BÚSQUEDA",
                        help="Texto a buscar en los tags")
    p_tags.add_argument("--list-all", action="store_true", dest="list_all",
                        help="Listar todos los tags disponibles")
    p_tags.add_argument("--model-dir", default=None, metavar="DIR", dest="model_dir",
                        help="Directorio de modelos (para top_200_tags.json)")
    p_tags.set_defaults(func=cmd_tags)

    # ── check ──────────────────────────────────────────────────────────────────
    p_chk = sub.add_parser(
        "check",
        help="Verifica el entorno: CUDA, modelos instalados, dependencias",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Comprueba que el entorno está correctamente configurado:
            GPU disponible, dependencias instaladas, modelos descargados.

            Ejemplos:
              check
              check --model-dir ~/mis_modelos/yue/
        """),
    )
    p_chk.add_argument("--model-dir", default=None, metavar="DIR", dest="model_dir",
                       help="Directorio de modelos a verificar")
    p_chk.set_defaults(func=cmd_check)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
