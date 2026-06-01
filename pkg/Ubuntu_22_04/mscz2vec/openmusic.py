#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         OPENMUSIC  v1.0                                      ║
║              Text-to-Music con QA-MDT  (Quality-Aware MDT)                   ║
║                                                                              ║
║  Genera música de alta calidad a partir de texto. Un único fichero           ║
║  autocontenido: no depende de ningún otro script del proyecto.               ║
║                                                                              ║
║  ARQUITECTURA (QA-MDT):                                                      ║
║    VAE mel ─► Latent Diffusion (MDT/DiT 28L) ─► HiFi-GAN vocoder            ║
║    Condición: Flan-T5-large (cross-attention) + token de calidad MOS         ║
║                                                                              ║
║  SUBCOMANDOS:                                                                 ║
║    setup        — Verifica entorno y guía descarga de checkpoints            ║
║    prepare      — Audio corpus → dataset LMDB listo para entrenar            ║
║    train        — Entrena o fine-tunea el modelo                             ║
║    compose      — Genera audio desde un prompt de texto                      ║
║    describe     — LLM/heurísticas: intención libre → prompt técnico          ║
║    inspect      — Diagnóstico: config, checkpoint, dataset LMDB              ║
║                                                                              ║
║  DEPENDENCIAS (para compose):                                                 ║
║    pip install torch torchaudio transformers librosa soundfile               ║
║             einops omegaconf pytorch_lightning laion-clap pyyaml            ║
║                                                                              ║
║  DEPENDENCIAS ADICIONALES (para prepare/train):                              ║
║    pip install lmdb "protobuf==3.20"                                         ║
║                                                                              ║
║  LLM OPCIONAL (para describe):                                               ║
║    pip install anthropic   →  ANTHROPIC_API_KEY                             ║
║    pip install openai      →  OPENAI_API_KEY                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

# ── Configuración inicial ──────────────────────────────────────────────────────
python openmusic.py setup --ckpt-dir checkpoints/

# ── Generar música desde un prompt ────────────────────────────────────────────
python openmusic.py compose \\
    --prompt "calm jazz piano trio, late night, melancholic, 90 BPM" \\
    --ckpt  checkpoints/model.ckpt \\
    --config audioldm_train/config/mos_as_token/qa_mdt.yaml \\
    --quality 5 --duration 10 --output salida.wav

# Calidad 3 (más variedad, menos restricción):
python openmusic.py compose \\
    --prompt "dark ambient drone, slow evolving textures" \\
    --ckpt  checkpoints/model.ckpt \\
    --quality 3 --duration 20 --output ambient.wav

# ── Traducir intención libre a prompt técnico ──────────────────────────────────
python openmusic.py describe "algo melancólico que suene a lluvia de otoño"
python openmusic.py describe "jazz nocturno relajado" --llm-provider openai
python openmusic.py describe "batalla épica medieval" --no-llm

# Encadenar describe → compose:
python openmusic.py describe "lluvia melancólica" --pipe | \\
    python openmusic.py compose --ckpt checkpoints/model.ckpt --from-stdin

# ── Preparar dataset propio ────────────────────────────────────────────────────
python openmusic.py prepare \\
    --audio-dir  corpus/audios/ \\
    --lmdb-dir   data/mi_corpus/ \\
    --captions   corpus/captions.json  # {nombre_archivo: "descripción"}

# ── Entrenar / Fine-tune ───────────────────────────────────────────────────────
python openmusic.py train \\
    --config   audioldm_train/config/mos_as_token/qa_mdt.yaml \\
    --lmdb-dir data/mi_corpus/ \\
    --ckpt-dir checkpoints/ \\
    --batch-size 8 --epochs 200

# Fine-tune desde checkpoint existente:
python openmusic.py train \\
    --config   audioldm_train/config/mos_as_token/qa_mdt.yaml \\
    --lmdb-dir data/mi_corpus/ \\
    --resume   checkpoints/model.ckpt --epochs 50

# ── Diagnóstico ───────────────────────────────────────────────────────────────
python openmusic.py inspect --ckpt checkpoints/model.ckpt
python openmusic.py inspect --config audioldm_train/config/mos_as_token/qa_mdt.yaml
python openmusic.py inspect --lmdb-dir data/mi_corpus/
"""

import sys
import os
import json
import argparse
import textwrap
import shutil
import re
import time
import importlib
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "1.0"

# Prefijo de texto que alinea con el proceso de entrenamiento
QUALITY_PREFIXES = {5: "high quality, ", 4: "high quality, ", 3: "", 2: "", 1: ""}

# Config por defecto (la del proyecto original)
DEFAULT_CONFIG  = "audioldm_train/config/mos_as_token/qa_mdt.yaml"
DEFAULT_CKPT_JSON = "offset_pretrained_checkpoints.json"

# Parámetros de audio que el proyecto usa por defecto
AUDIO_DEFAULTS = {
    "sampling_rate": 16000,
    "duration":      10.24,
    "filter_length": 1024,
    "hop_length":    160,
    "win_length":    1024,
    "n_mel":         64,
    "mel_fmin":      0,
    "mel_fmax":      8000,
}

# ══════════════════════════════════════════════════════════════════════════════
#  LLM — BACKENDS  (anthropic / openai, mismo patrón en todos los scripts)
# ══════════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM = """\
Eres un compositor y productor musical experto. Traduce la descripción \
musical del usuario a un prompt técnico en inglés optimizado para un \
modelo text-to-music (QA-MDT, difusión latente sobre espectrogramas mel).

El prompt debe ser conciso (20-60 palabras) e incluir: género, tempo \
(ej. "90 BPM"), instrumentación, carácter emocional, textura y \
referencias estilísticas cuando corresponda.

Devuelve EXCLUSIVAMENTE un JSON válido sin texto adicional ni backticks:
{
  "prompt":   "<prompt técnico en inglés>",
  "quality":  <int 1-5, nivel MOS recomendado: 5=máxima calidad>,
  "duration": <int segundos sugeridos: 10, 20 o 30>,
  "reasoning":"<una frase breve sobre las decisiones musicales>"
}
"""


def _llm_debug(label: str, text: str):
    bar = "─" * max(0, 44 - len(label))
    print(f"\n  ┌─ [llm:debug] {label} {bar}")
    for line in text.splitlines():
        print(f"  │ {line}")
    print(f"  └{'─'*58}")


def _parse_json_response(raw: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _call_anthropic(user_prompt: str, api_key: str,
                    model: str | None, verbose: bool, debug: bool) -> dict:
    try:
        import anthropic
    except ImportError:
        if verbose:
            print("  [llm:anthropic] No instalado → pip install anthropic")
        return {}
    model = model or "claude-sonnet-4-6"
    if debug:
        _llm_debug("USER PROMPT → anthropic/" + model, user_prompt)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model, max_tokens=512,
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = msg.content[0].text.strip()
        if debug:
            _llm_debug("RESPUESTA RAW", raw[:600])
        return _parse_json_response(raw)
    except Exception as e:
        if verbose:
            print(f"  [llm:anthropic] Error: {e}")
        return {}


def _call_openai(user_prompt: str, api_key: str,
                 model: str | None, verbose: bool, debug: bool) -> dict:
    try:
        import openai
    except ImportError:
        if verbose:
            print("  [llm:openai] No instalado → pip install openai")
        return {}
    model = model or "gpt-4o"
    if debug:
        _llm_debug("USER PROMPT → openai/" + model, user_prompt)
    try:
        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model, max_tokens=512,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if debug:
            _llm_debug("RESPUESTA RAW", raw[:600])
        return _parse_json_response(raw)
    except Exception as e:
        if verbose:
            print(f"  [llm:openai] Error: {e}")
        return {}


LLM_BACKENDS = {
    "anthropic": (_call_anthropic, "ANTHROPIC_API_KEY",
                  ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"]),
    "openai":    (_call_openai,    "OPENAI_API_KEY",
                  ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]),
}


def call_llm(user_prompt: str, provider: str = "anthropic",
             api_key: str | None = None, model: str | None = None,
             verbose: bool = False, debug: bool = False) -> dict:
    provider = provider.lower()
    if provider not in LLM_BACKENDS:
        if verbose:
            print(f"  [llm] Proveedor desconocido: {provider}")
        return {}
    fn, env_var, _ = LLM_BACKENDS[provider]
    if api_key is None:
        api_key = os.environ.get(env_var)
    if not api_key:
        if verbose:
            print(f"  [llm:{provider}] Sin API key "
                  f"(usa --api-key o exporta {env_var})")
        return {}
    if verbose:
        print(f"  [llm:{provider}] modelo={model or '(default)'}")
    return fn(user_prompt, api_key, model, verbose, debug)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR LOCAL — describe sin LLM
# ══════════════════════════════════════════════════════════════════════════════

_DESCRIBE_MAP = {
    # Géneros
    "jazz":        {"genre": "jazz"},
    "clasic":      {"genre": "classical orchestral"},           # cubre clásica/classical
    "ambient":     {"genre": "ambient", "character": "atmospheric, evolving"},
    "rock":        {"genre": "rock", "instruments": "electric guitar, drums, bass"},
    "folk":        {"genre": "folk", "instruments": "acoustic guitar, fiddle"},
    "lo-fi":       {"genre": "lo-fi hip hop", "character": "relaxed, warm, nostalgic"},
    "lofi":        {"genre": "lo-fi hip hop", "character": "relaxed, warm, nostalgic"},
    "trap":        {"genre": "trap", "tempo": "140 BPM"},
    "reggaeton":   {"genre": "reggaeton", "instruments": "dembow, bass"},
    "flamenco":    {"genre": "flamenco", "instruments": "Spanish guitar, palmas"},
    "blues":       {"genre": "blues", "instruments": "guitar, harmonica"},
    "soul":        {"genre": "soul", "character": "warm, emotional"},
    "bossa":       {"genre": "bossa nova", "instruments": "acoustic guitar, light percussion"},
    "electronic":  {"genre": "electronic", "instruments": "synthesizers"},
    "electroni":   {"genre": "electronic", "instruments": "synthesizers"},
    "orquestal":   {"instruments": "full orchestra"},
    "orchestra":   {"instruments": "full orchestra"},
    "medieval":    {"genre": "medieval", "instruments": "lute, horn, drums"},
    "medieval":    {"genre": "medieval", "instruments": "lute, horn, drums"},
    # Carácter
    "melanc":      {"character": "melancholic, nostalgic"},     # melancólico/melancholic
    "triste":      {"character": "sad, melancholic"},
    "alegre":      {"character": "upbeat, joyful"},
    "tenso":       {"character": "tense, suspenseful"},
    "tranquil":    {"character": "calm, peaceful"},
    "oscuro":      {"character": "dark, ominous"},
    "dark":        {"character": "dark, ominous"},
    "luminoso":    {"character": "bright, uplifting"},
    "bright":      {"character": "bright, uplifting"},
    "misterios":   {"character": "mysterious, enigmatic"},
    "romantic":    {"character": "romantic, tender"},
    "urgente":     {"character": "urgent, intense", "tempo": "fast"},
    "energe":      {"character": "energetic, powerful"},
    "epico":       {"genre": "epic orchestral", "character": "powerful, dramatic"},
    "epic":        {"genre": "epic orchestral", "character": "powerful, dramatic"},
    # Entornos / metáforas
    "lluvia":      {"character": "rainy, contemplative", "texture": "sparse, flowing"},
    "rain":        {"character": "rainy, contemplative", "texture": "sparse, flowing"},
    "tormenta":    {"character": "stormy, turbulent, intense"},
    "storm":       {"character": "stormy, turbulent, intense"},
    "noche":       {"character": "nocturnal, intimate"},
    "night":       {"character": "nocturnal, intimate"},
    "amanecer":    {"character": "dawn, hopeful, gradually building"},
    "batalla":     {"character": "battle, epic, intense, dramatic"},
    "battle":      {"character": "battle, epic, intense, dramatic"},
    "naturaleza":  {"character": "organic, natural, peaceful"},
    # Tempo
    "lento":       {"tempo": "60 BPM"},
    "rapido":      {"tempo": "140 BPM"},
    "fast":        {"tempo": "140 BPM"},
    "moderado":    {"tempo": "100 BPM"},
    # Textura
    "minimalista": {"texture": "minimal, sparse"},
    "minimal":     {"texture": "minimal, sparse"},
    "piano":       {"instruments": "solo piano"},
    "guitarra":    {"instruments": "acoustic guitar"},
    "cuerdas":     {"instruments": "strings"},
    "strings":     {"instruments": "strings"},
}

_HIGH_QUALITY = {"orquestal", "orchestra", "clasic", "flamenco", "jazz", "soul", "epico", "epic"}


def _norm(text: str) -> str:
    reps = {"á":"a","é":"e","í":"i","ó":"o","ú":"u","à":"a","è":"e",
            "ì":"i","ò":"o","ù":"u","ñ":"n","ü":"u"}
    t = text.lower().strip()
    for k, v in reps.items():
        t = t.replace(k, v)
    return t


def local_describe(description: str, duration: int = 10) -> dict:
    """Heurísticas locales: descripción libre → prompt técnico (sin LLM)."""
    t = _norm(description)
    gathered: dict[str, str] = {}
    quality = 4

    for stem, attrs in _DESCRIBE_MAP.items():
        if stem in t:
            gathered.update(attrs)
            if stem in _HIGH_QUALITY:
                quality = 5

    parts = [gathered[k] for k in ("genre", "instruments", "character", "tempo", "texture")
             if k in gathered]

    if not parts:
        prompt = description
        reasoning = "Sin coincidencias en el diccionario local; usando descripción literal."
    else:
        prompt = ", ".join(parts)
        reasoning = f"Conceptos detectados: {', '.join(gathered.keys())}"

    return {"prompt": prompt, "quality": quality,
            "duration": duration, "reasoning": reasoning}


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIO — mel-spectrogram y dataset mínimo (sin imports del proyecto)
# ══════════════════════════════════════════════════════════════════════════════

def _load_audio(path: str, sr: int = 16000) -> "np.ndarray":
    """Carga y resamplea audio a mono float32 en [-1, 1]."""
    import numpy as np
    try:
        import librosa
        audio, _ = librosa.load(path, sr=sr, mono=True)
    except Exception:
        import torchaudio, torch
        wav, orig_sr = torchaudio.load(path)
        if orig_sr != sr:
            wav = torchaudio.functional.resample(wav, orig_sr, sr)
        audio = wav.mean(0).numpy()
    # Normalizar
    audio = audio - audio.mean()
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio / peak * 0.5).astype(np.float32)


def _audio_to_mel(audio: "np.ndarray", cfg: dict) -> "torch.Tensor":
    """
    audio (float32 numpy) → log mel-spectrogram tensor [T, mel_bins].
    Usa torchaudio, que es una dependencia directa del proyecto.
    """
    import torch
    import torchaudio

    sr     = cfg.get("sampling_rate", AUDIO_DEFAULTS["sampling_rate"])
    n_fft  = cfg.get("filter_length", AUDIO_DEFAULTS["filter_length"])
    hop    = cfg.get("hop_length",    AUDIO_DEFAULTS["hop_length"])
    win    = cfg.get("win_length",    AUDIO_DEFAULTS["win_length"])
    n_mel  = cfg.get("n_mel",         AUDIO_DEFAULTS["n_mel"])
    fmin   = cfg.get("mel_fmin",      AUDIO_DEFAULTS["mel_fmin"])
    fmax   = cfg.get("mel_fmax",      AUDIO_DEFAULTS["mel_fmax"])

    wav = torch.FloatTensor(audio).unsqueeze(0)  # [1, T]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop,
        win_length=win, n_mels=n_mel, f_min=fmin, f_max=fmax,
        power=1.0,
    )
    mel = mel_transform(wav).squeeze(0)           # [n_mel, T]
    mel = torch.log(torch.clamp(mel, min=1e-5))   # log-mel
    return mel.T                                   # [T, n_mel]


class _PromptDataset:
    """
    Dataset mínimo para inferencia: lista de prompts → batches
    con el formato que espera LatentDiffusion.generate_sample().
    No depende de ningún módulo del proyecto.
    """

    def __init__(self, prompts: list[str], quality: int, cfg: dict):
        self.prompts = prompts
        self.quality = quality
        self.cfg = cfg
        sr       = cfg.get("sampling_rate", AUDIO_DEFAULTS["sampling_rate"])
        duration = cfg.get("duration",      AUDIO_DEFAULTS["duration"])
        hop      = cfg.get("hop_length",    AUDIO_DEFAULTS["hop_length"])
        n_mel    = cfg.get("n_mel",         AUDIO_DEFAULTS["n_mel"])
        self.target_len = int(duration * sr / hop)
        self.n_mel      = n_mel
        self.sr         = sr

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        import torch
        text = self.prompts[idx]
        # Espectrograma mel silencioso (el modelo lo ignora para text-to-audio)
        log_mel = torch.zeros(self.target_len, self.n_mel)
        # Forma de onda silenciosa
        waveform = torch.zeros(1, int(self.cfg.get("duration", 10.24) * self.sr))
        return {
            "text":        text,
            "fname":       text[:40].replace(" ", "_"),
            "waveform":    waveform.float(),
            "log_mel_spec": log_mel.float(),
            "stft":        torch.zeros(self.target_len, self.n_mel).float(),
            "label_vector": torch.zeros(527).float(),
            "duration":    self.cfg.get("duration", AUDIO_DEFAULTS["duration"]),
            "sampling_rate": self.sr,
            "random_start_sample_in_original_audio_file": 0,
            "mos":         self.quality,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE CONFIG Y MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        print("  [error] pyyaml no instalado → pip install pyyaml")
        sys.exit(1)
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _audio_cfg_from_config(config: dict) -> dict:
    pre = config.get("preprocessing", {})
    return {
        "sampling_rate": pre.get("audio", {}).get("sampling_rate", AUDIO_DEFAULTS["sampling_rate"]),
        "duration":      pre.get("audio", {}).get("duration",      AUDIO_DEFAULTS["duration"]),
        "filter_length": pre.get("stft",  {}).get("filter_length", AUDIO_DEFAULTS["filter_length"]),
        "hop_length":    pre.get("stft",  {}).get("hop_length",    AUDIO_DEFAULTS["hop_length"]),
        "win_length":    pre.get("stft",  {}).get("win_length",    AUDIO_DEFAULTS["win_length"]),
        "n_mel":         pre.get("mel",   {}).get("n_mel_channels", AUDIO_DEFAULTS["n_mel"]),
        "mel_fmin":      pre.get("mel",   {}).get("mel_fmin",       AUDIO_DEFAULTS["mel_fmin"]),
        "mel_fmax":      pre.get("mel",   {}).get("mel_fmax",       AUDIO_DEFAULTS["mel_fmax"]),
    }


def _instantiate_from_config(config: dict):
    """
    Réplica mínima de model_util.instantiate_from_config().
    Importa dinámicamente la clase indicada en config['target'].
    """
    target = config.get("target")
    if not target:
        raise KeyError("'target' no encontrado en la config del modelo")
    module_str, cls_str = target.rsplit(".", 1)
    try:
        module = importlib.import_module(module_str)
    except ModuleNotFoundError as e:
        print(f"  [error] No se pudo importar '{module_str}': {e}")
        print("          ¿Está instalado el proyecto QA-MDT en el PYTHONPATH?")
        sys.exit(1)
    cls = getattr(module, cls_str)
    return cls(**config.get("params", {}))


def _load_model(ckpt_path: str, config: dict, verbose: bool = False):
    """Carga LatentDiffusion con el checkpoint dado."""
    try:
        import torch
    except ImportError:
        print("  [error] torch no instalado → pip install torch")
        sys.exit(1)

    if verbose:
        print("  [model] Instanciando modelo desde config...")

    model = _instantiate_from_config(config["model"])

    if verbose:
        print(f"  [model] Cargando checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    model = model.cuda()

    if verbose:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  [model] {n:.1f}M parámetros cargados en GPU")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: setup
# ══════════════════════════════════════════════════════════════════════════════

def cmd_setup(args):
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  OPENMUSIC v{VERSION}  —  Verificación del entorno")
    print(f"{'═'*62}\n")

    print(f"  Python: {sys.version.split()[0]}")

    # Dependencias
    deps = [
        ("torch",             "pip install torch==2.3.0"),
        ("torchaudio",        "pip install torchaudio==2.3.0"),
        ("transformers",      "pip install transformers==4.40.2"),
        ("librosa",           "pip install librosa==0.9.2"),
        ("soundfile",         "pip install soundfile"),
        ("einops",            "pip install einops"),
        ("omegaconf",         "pip install omegaconf==2.0.6"),
        ("pytorch_lightning", "pip install pytorch_lightning==2.1.3"),
        ("laion_clap",        "pip install laion-clap==1.1.4"),
        ("yaml",              "pip install pyyaml"),
        ("lmdb",              "pip install lmdb  # solo para prepare/train"),
    ]
    print("\n  Dependencias:")
    for pkg, hint in deps:
        try:
            importlib.import_module(pkg)
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg}   →  {hint}")

    # GPU
    print("\n  GPU/CUDA:")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"    ✓ GPU {i}: {name}  ({mem} GB)")
        else:
            print("    ✗ CUDA no disponible — se requiere GPU para inferencia/entrenamiento")
    except ImportError:
        print("    ? torch no instalado")

    # Checkpoints
    print(f"\n  Checkpoints esperados en: {ckpt_dir}/")
    items = [
        ("flan-t5-large/",
         "huggingface-cli download google/flan-t5-large --local-dir checkpoints/flan-t5-large"),
        ("roberta-base/",
         "huggingface-cli download FacebookAI/roberta-base --local-dir checkpoints/roberta-base"),
        ("clap_music_speech_audioset_epoch_15_esc_89.98.pt",
         "wget https://huggingface.co/lukewys/laion_clap/resolve/main/"
         "music_speech_audioset_epoch_15_esc_89.98.pt -P checkpoints/"),
        ("hifi-gan/",
         "Descarga manual desde https://drive.google.com/file/d/1T6EnuAHIc8ioeZ9kB1OZ_WGgwXAVGOZS"),
    ]
    for name, hint in items:
        mark = "✓" if (ckpt_dir / name).exists() else "✗"
        print(f"    {mark} {name}")
        if mark == "✗":
            print(f"         {hint}")

    # Generar offset_pretrained_checkpoints.json
    json_path = Path(DEFAULT_CKPT_JSON)
    if not json_path.exists() or args.force:
        cfg = {
            "clap_music":   str(ckpt_dir / "clap_music_speech_audioset_epoch_15_esc_89.98.pt"),
            "flan_t5":      str(ckpt_dir / "flan-t5-large"),
            "hifi-gan":     str(ckpt_dir / "hifi-gan"),
            "roberta-base": str(ckpt_dir / "roberta-base"),
        }
        with open(json_path, "w") as f:
            json.dump(cfg, f, indent=4)
        print(f"\n  ✓ {DEFAULT_CKPT_JSON} generado")
    else:
        print(f"\n  ✓ {DEFAULT_CKPT_JSON} ya existe")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: describe
# ══════════════════════════════════════════════════════════════════════════════

def cmd_describe(args):
    if args.verbose:
        print(f"\n  Descripción: \"{args.description}\"")

    result = {}
    if not args.no_llm:
        user_msg = (
            f"Descripción musical: \"{args.description}\"\n"
            f"Duración deseada aproximada: {args.duration} segundos."
        )
        result = call_llm(user_msg, provider=args.llm_provider,
                          api_key=args.api_key, model=args.llm_model,
                          verbose=args.verbose, debug=args.llm_debug)

    if not result:
        if not args.no_llm and args.verbose:
            print("  [describe] LLM no disponible → motor local")
        result = local_describe(args.description, duration=args.duration)

    prompt    = result.get("prompt", args.description)
    quality   = result.get("quality", 4)
    duration  = result.get("duration", args.duration)
    reasoning = result.get("reasoning", "")

    if reasoning and args.verbose:
        print(f"\n  Razonamiento: {reasoning}")

    print(f"\n  Prompt técnico:       {prompt}")
    print(f"  Calidad recomendada:  {quality}/5")
    print(f"  Duración sugerida:    {duration}s")

    if args.pipe:
        # Salida estructurada para encadenar con compose --from-stdin
        print(f"\n---PIPE---")
        print(json.dumps({"prompt": prompt, "quality": quality, "duration": duration}))

    if args.output_json:
        out = {"description": args.description, "prompt": prompt,
               "quality": quality, "duration": duration,
               "reasoning": reasoning, "timestamp": datetime.now().isoformat()}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Guardado en: {args.output_json}")


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: compose
# ══════════════════════════════════════════════════════════════════════════════

def cmd_compose(args):
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [error] CUDA no disponible. Se requiere GPU.")
            sys.exit(1)
    except ImportError:
        print("  [error] torch no instalado → pip install torch")
        sys.exit(1)

    # ── Resolver prompt ───────────────────────────────────────────────────────
    if args.from_stdin:
        raw = sys.stdin.read()
        pipe_json = raw.split("---PIPE---")[-1].strip() if "---PIPE---" in raw else raw
        pipe_data = json.loads(pipe_json)
        prompt   = pipe_data.get("prompt", "")
        quality  = pipe_data.get("quality", args.quality)
        duration = pipe_data.get("duration", args.duration)
    else:
        prompt, quality, duration = args.prompt, args.quality, args.duration

    if not prompt:
        print("  [error] Proporciona --prompt o usa --from-stdin")
        sys.exit(1)

    full_prompt = QUALITY_PREFIXES.get(quality, "") + prompt

    print(f"\n{'═'*62}")
    print(f"  OPENMUSIC v{VERSION}  —  Composición")
    print(f"{'═'*62}")
    print(f"  Prompt:   {full_prompt}")
    print(f"  Calidad:  {quality}/5")
    print(f"  Duración: {duration}s")
    print()

    # ── Config y checkpoint ───────────────────────────────────────────────────
    config_path = args.config or DEFAULT_CONFIG
    if not Path(config_path).exists():
        print(f"  [error] Config no encontrada: {config_path}")
        sys.exit(1)
    config = _load_config(config_path)

    if not args.ckpt or not Path(args.ckpt).exists():
        print(f"  [error] Checkpoint no encontrado: {args.ckpt}")
        print("          Descarga desde https://huggingface.co/lichang0928/QA-MDT")
        sys.exit(1)

    audio_cfg = _audio_cfg_from_config(config)
    seg_dur   = audio_cfg["duration"]
    n_segs    = max(1, round(duration / seg_dur))
    prompts   = [full_prompt] * n_segs

    # ── Cargar modelo ─────────────────────────────────────────────────────────
    print("  [1/3] Cargando modelo...")
    t0 = time.time()
    model = _load_model(args.ckpt, config, verbose=args.verbose)
    print(f"        Listo en {time.time()-t0:.1f}s")

    # ── Dataset y DataLoader ──────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    dataset = _PromptDataset(prompts, quality, audio_cfg)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False)

    # ── Parámetros de muestreo ────────────────────────────────────────────────
    ep      = config["model"]["params"].get("evaluation_params", {})
    g_scale = args.guidance_scale or ep.get("unconditional_guidance_scale", 3.5)
    d_steps = args.ddim_steps     or ep.get("ddim_sampling_steps", 200)
    n_cand  = args.n_candidates   or ep.get("n_candidates_per_samples", 3)

    if args.verbose:
        print(f"  [sampler] guidance={g_scale}, ddim_steps={d_steps}, candidates={n_cand}")

    # ── Generar ───────────────────────────────────────────────────────────────
    out_path = Path(args.output) if args.output \
        else Path(f"openmusic_{int(time.time())}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = str(out_path.parent)
    model.set_log_dir(log_dir, "openmusic", "compose")

    print(f"  [2/3] Generando ({d_steps} pasos DDIM)...")
    t1 = time.time()

    with torch.no_grad():
        model.generate_sample(
            loader,
            unconditional_guidance_scale=g_scale,
            ddim_steps=d_steps,
            n_gen=n_cand,
        )

    print(f"        Generado en {time.time()-t1:.1f}s")

    # ── Localizar el WAV generado ─────────────────────────────────────────────
    print("  [3/3] Guardando resultado...")
    wavs = sorted(Path(log_dir).rglob("*.wav"),
                  key=lambda p: p.stat().st_mtime, reverse=True)

    if not wavs:
        print(f"  [warn] No se encontró .wav en {log_dir}")
    else:
        if str(wavs[0]) != str(out_path):
            shutil.copy(str(wavs[0]), str(out_path))
        print(f"\n  ✓ Audio guardado en: {out_path}")
        if n_cand > 1:
            print(f"    Candidatos adicionales en: {log_dir}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: prepare
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    try:
        import lmdb
    except ImportError:
        print("  [error] lmdb no instalado → pip install lmdb")
        sys.exit(1)

    # Protobuf: datum_all_pb2.py debe existir en audioldm_train/utilities/data/
    # o generarse con: protoc --python_out=./ datum_all.proto
    pb2_candidates = [
        Path("audioldm_train/utilities/data/datum_all_pb2.py"),
        Path("datum_all_pb2.py"),
    ]
    pb2_path = next((p for p in pb2_candidates if p.exists()), None)
    if pb2_path is None:
        print("  [error] datum_all_pb2.py no encontrado.")
        print("          Genera con: protoc --python_out=./ datum_all.proto")
        print("          (protoc v3.4: https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0)")
        sys.exit(1)

    sys.path.insert(0, str(pb2_path.parent))
    from datum_all_pb2 import Datum_all as DatumProto

    audio_dir = Path(args.audio_dir)
    lmdb_dir  = Path(args.lmdb_dir)
    lmdb_dir.mkdir(parents=True, exist_ok=True)

    # Captions opcionales
    captions: dict[str, str] = {}
    if args.captions and Path(args.captions).exists():
        with open(args.captions, encoding="utf-8") as f:
            if args.captions.endswith(".json"):
                captions = json.load(f)
            else:                           # txt / lst: "nombre|caption"
                for line in f:
                    line = line.strip()
                    if "|" in line:
                        k, v = line.split("|", 1)
                        captions[k.strip()] = v.strip()

    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff"}
    files = [p for p in audio_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        print(f"  [error] No se encontraron archivos de audio en: {audio_dir}")
        sys.exit(1)

    print(f"\n{'═'*62}")
    print(f"  OPENMUSIC v{VERSION}  —  Preparación de dataset")
    print(f"{'═'*62}")
    print(f"  Archivos: {len(files)}")
    print(f"  SR:       {args.sample_rate} Hz")
    print(f"  Destino:  {lmdb_dir}\n")

    env = lmdb.open(str(lmdb_dir), map_size=int(1e12))
    txn = env.begin(write=True)
    keys, errors = [], 0

    for i, audio_path in enumerate(files):
        key = str(audio_path).replace("/", "_").replace("\\", "_")
        try:
            audio = _load_audio(str(audio_path), sr=args.sample_rate)
        except Exception as e:
            if args.verbose:
                print(f"  [skip] {audio_path.name}: {e}")
            errors += 1
            continue

        caption = (captions.get(audio_path.name)
                   or captions.get(audio_path.stem)
                   or args.default_caption or "")

        datum = DatumProto()
        datum.wav_file.extend(audio.tolist())
        datum.caption_original = caption
        datum.caption_generated.append(caption)
        datum.mos = float(args.mos)

        txn.put(key.encode(), datum.SerializeToString())
        keys.append(key)

        if (i + 1) % 100 == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"  {i+1}/{len(files)} procesados...")

    try:
        txn.commit()
    except Exception:
        pass
    env.close()

    key_file = lmdb_dir / "data_key.key"
    with open(key_file, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(k + "\n")

    print(f"\n  ✓ {len(keys)} muestras guardadas  ({errors} errores)")
    print(f"    LMDB:  {lmdb_dir}")
    print(f"    Keys:  {key_file}")
    print(f"\n  Siguiente paso:")
    print(f"    python openmusic.py train --lmdb-dir {lmdb_dir} --ckpt-dir checkpoints/\n")


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: train
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    try:
        import torch
        if not torch.cuda.is_available():
            print("  [error] CUDA no disponible.")
            sys.exit(1)
    except ImportError:
        print("  [error] torch no instalado")
        sys.exit(1)

    config_path = args.config or DEFAULT_CONFIG
    if not Path(config_path).exists():
        print(f"  [error] Config no encontrada: {config_path}")
        sys.exit(1)

    config = _load_config(config_path)

    # Inyectar paths de datos
    lmdb_dir = Path(args.lmdb_dir).resolve()
    key_file = lmdb_dir / "data_key.key"
    config["train_path"]["train_lmdb_path"] = [str(lmdb_dir)]
    config["val_path"]["val_lmdb_path"]     = [str(lmdb_dir)]
    config["val_path"]["val_key_path"]      = [str(key_file)]

    if args.ckpt_dir:
        config["log_directory"] = args.ckpt_dir
    if args.batch_size:
        config["model"]["params"]["batchsize"] = args.batch_size
    if args.lr:
        config["model"]["params"]["base_learning_rate"] = args.lr
    if args.epochs:
        config["step"]["max_steps"] = args.epochs * 1000

    print(f"\n{'═'*62}")
    print(f"  OPENMUSIC v{VERSION}  —  Entrenamiento")
    print(f"{'═'*62}")
    print(f"  Config:     {config_path}")
    print(f"  LMDB:       {lmdb_dir}")
    print(f"  Batch:      {config['model']['params']['batchsize']}")
    print(f"  Max steps:  {config['step']['max_steps']}")
    if args.resume:
        print(f"  Resumiendo: {args.resume}")
    print()

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    # El entrypoint de entrenamiento está en el proyecto; lo importamos
    # dinámicamente para no depender de él en los demás comandos.
    try:
        from audioldm_train.train.latent_diffusion import main as _train_main
    except ImportError:
        print("  [error] audioldm_train no encontrado en el PYTHONPATH.")
        print("          Ejecuta desde la raíz del proyecto QA-MDT o añade el path.")
        sys.exit(1)

    exp_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    try:
        _train_main(configs=config, config_yaml_path=config_path,
                    exp_group_name="openmusic", exp_name=exp_name,
                    perform_validation=False)
    except KeyboardInterrupt:
        print("\n  Entrenamiento interrumpido.")


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDO: inspect
# ══════════════════════════════════════════════════════════════════════════════

def cmd_inspect(args):
    print(f"\n{'═'*62}")
    print(f"  OPENMUSIC v{VERSION}  —  Inspección")
    print(f"{'═'*62}\n")

    # ── Config ────────────────────────────────────────────────────────────────
    if args.config:
        p = args.config
        if not Path(p).exists():
            print(f"  [config] No encontrada: {p}")
        else:
            cfg = _load_config(p)
            mp  = cfg.get("model", {}).get("params", {})
            uc  = mp.get("unet_config", {}).get("params", {})
            pre = cfg.get("preprocessing", {})
            print(f"  CONFIG: {p}")
            print(f"    Sample rate:    {pre.get('audio',{}).get('sampling_rate','?')} Hz")
            print(f"    Duración seg.:  {pre.get('audio',{}).get('duration','?')}s")
            print(f"    Diffusion steps:{mp.get('timesteps','?')}")
            print(f"    Batch size:     {mp.get('batchsize','?')}")
            print(f"    Transformer:    {mp.get('unet_config',{}).get('target','?').split('.')[-1]}")
            print(f"    Depth / heads:  {uc.get('depth','?')} / {uc.get('num_heads','?')}")
            print(f"    Hidden size:    {uc.get('hidden_size','?')}")
            print(f"    Guidance scale: {mp.get('evaluation_params',{}).get('unconditional_guidance_scale','?')}")
            print(f"    DDIM steps:     {mp.get('evaluation_params',{}).get('ddim_sampling_steps','?')}")
            print()

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if args.ckpt:
        if not Path(args.ckpt).exists():
            print(f"  [ckpt] No encontrado: {args.ckpt}")
        else:
            try:
                import torch
                ckpt    = torch.load(args.ckpt, map_location="cpu")
                state   = ckpt.get("state_dict", {})
                n_param = sum(v.numel() for v in state.values() if hasattr(v, "numel"))
                size_mb = Path(args.ckpt).stat().st_size / (1024**2)
                print(f"  CHECKPOINT: {args.ckpt}")
                print(f"    Tamaño:      {size_mb:.1f} MB")
                print(f"    Epoch:       {ckpt.get('epoch','?')}")
                print(f"    Global step: {ckpt.get('global_step','?')}")
                print(f"    Parámetros:  {n_param/1e6:.1f}M")
                # Bloques principales
                blocks: dict[str, int] = {}
                for k in state:
                    blocks[k.split(".")[0]] = blocks.get(k.split(".")[0], 0) + 1
                print(f"    Bloques (top 6):")
                for blk, cnt in sorted(blocks.items(), key=lambda x: -x[1])[:6]:
                    print(f"      {blk}: {cnt} tensores")
                print()
            except Exception as e:
                print(f"  [ckpt] Error al leer: {e}\n")

    # ── Dataset LMDB ──────────────────────────────────────────────────────────
    if args.lmdb_dir:
        if not Path(args.lmdb_dir).exists():
            print(f"  [lmdb] No encontrado: {args.lmdb_dir}")
        else:
            try:
                import lmdb
                env = lmdb.open(args.lmdb_dir, readonly=True, lock=False)
                with env.begin() as txn:
                    n_entries = txn.stat()["entries"]
                env.close()
                key_file = Path(args.lmdb_dir) / "data_key.key"
                n_keys = sum(1 for _ in open(key_file)) if key_file.exists() else "?"
                size_gb = sum(p.stat().st_size for p in Path(args.lmdb_dir).iterdir()) / (1024**3)
                print(f"  LMDB: {args.lmdb_dir}")
                print(f"    Entradas: {n_entries}")
                print(f"    Claves:   {n_keys}")
                print(f"    Tamaño:   {size_gb:.2f} GB")
                print()
            except ImportError:
                print("  [lmdb] lmdb no instalado → pip install lmdb")
            except Exception as e:
                print(f"  [lmdb] Error: {e}\n")

    # ── Rutas de checkpoints ──────────────────────────────────────────────────
    if Path(DEFAULT_CKPT_JSON).exists():
        print(f"  RUTAS ({DEFAULT_CKPT_JSON}):")
        with open(DEFAULT_CKPT_JSON) as f:
            data = json.load(f)
        for k, v in data.items():
            mark = "✓" if Path(v).exists() else "✗"
            print(f"    {mark} {k}: {v}")
        print()
    else:
        print(f"  [rutas] {DEFAULT_CKPT_JSON} no existe")
        print("          Ejecuta: python openmusic.py setup\n")


# ══════════════════════════════════════════════════════════════════════════════
#  PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openmusic.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            OPENMUSIC v1.0 — Generación text-to-music con QA-MDT
            ──────────────────────────────────────────────────────
            Genera música de alta calidad a partir de texto usando
            difusión latente condicionada en calidad (MOS token).

            Ejemplos rápidos:
              python openmusic.py setup --ckpt-dir checkpoints/
              python openmusic.py describe "jazz melancólico nocturno"
              python openmusic.py compose --prompt "epic strings, 90 BPM" \\
                  --ckpt checkpoints/model.ckpt --quality 5
        """),
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # ── setup ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("setup", help="Verifica entorno y guía descarga de checkpoints")
    p.add_argument("--ckpt-dir", default="checkpoints/", metavar="DIR",
                   help="Directorio de checkpoints (default: checkpoints/)")
    p.add_argument("--force", action="store_true",
                   help="Regenerar offset_pretrained_checkpoints.json aunque exista")
    p.set_defaults(func=cmd_setup)

    # ── describe ──────────────────────────────────────────────────────────────
    p = sub.add_parser("describe",
                        help="Intención musical libre → prompt técnico (LLM o local)")
    p.add_argument("description", help="Descripción musical en lenguaje libre")
    p.add_argument("--duration",     type=int, default=10, metavar="N",
                   help="Duración deseada en segundos (default: 10)")
    p.add_argument("--output-json",  metavar="FILE",
                   help="Guardar resultado en JSON")
    p.add_argument("--pipe",         action="store_true",
                   help="Emitir JSON para encadenar con compose --from-stdin")
    p.add_argument("--no-llm",       action="store_true",
                   help="Usar solo heurísticas locales (sin API)")
    p.add_argument("--llm-provider", default="anthropic",
                   choices=list(LLM_BACKENDS.keys()),
                   help="Proveedor LLM: anthropic (default) | openai")
    p.add_argument("--llm-model",    default=None,
                   help="Modelo concreto (default: auto). "
                        "Anthropic: claude-sonnet-4-6, claude-opus-4-6 | "
                        "OpenAI: gpt-4o, gpt-4o-mini")
    p.add_argument("--api-key",      default=None,
                   help="API key (o ANTHROPIC_API_KEY / OPENAI_API_KEY en entorno)")
    p.add_argument("--llm-debug",    action="store_true",
                   help="Mostrar prompt enviado y respuesta raw del LLM")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_describe)

    # ── compose ───────────────────────────────────────────────────────────────
    p = sub.add_parser("compose", help="Genera audio desde un prompt de texto")
    p.add_argument("--prompt",      metavar="TEXT",
                   help="Prompt descriptivo en inglés")
    p.add_argument("--from-stdin",  action="store_true",
                   help="Leer prompt desde stdin (salida de describe --pipe)")
    p.add_argument("--ckpt",        metavar="FILE",
                   help="Checkpoint del modelo (.ckpt)")
    p.add_argument("--config",      metavar="FILE", default=None,
                   help=f"Config YAML del modelo (default: {DEFAULT_CONFIG})")
    p.add_argument("--quality",     type=int, default=5,
                   choices=[1, 2, 3, 4, 5],
                   help="Nivel de calidad MOS 1-5 (default: 5)")
    p.add_argument("--duration",    type=int, default=10, metavar="N",
                   help="Duración aproximada en segundos (default: 10)")
    p.add_argument("--output",      metavar="FILE",
                   help="Ruta del .wav de salida (default: openmusic_<ts>.wav)")
    p.add_argument("--guidance-scale", type=float, default=None, metavar="F",
                   help="Guidance scale CFG (default: valor del config, típico 3.5)")
    p.add_argument("--ddim-steps",  type=int, default=None, metavar="N",
                   help="Pasos DDIM (default: valor del config, típico 200)")
    p.add_argument("--n-candidates", type=int, default=None, metavar="N",
                   help="Candidatos generados; se guarda el mejor (default: 3)")
    p.add_argument("--seed",        type=int, default=0,
                   help="Semilla aleatoria (default: 0)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_compose)

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser("prepare",
                        help="Audio corpus → dataset LMDB listo para entrenar")
    p.add_argument("--audio-dir",       metavar="DIR", required=True,
                   help="Directorio con archivos de audio (.wav .mp3 .flac ...)")
    p.add_argument("--lmdb-dir",        metavar="DIR", required=True,
                   help="Directorio de salida del LMDB")
    p.add_argument("--sample-rate",     type=int, default=16000, metavar="N",
                   help="Sample rate de resampleo (default: 16000)")
    p.add_argument("--captions",        metavar="FILE", default=None,
                   help="JSON {archivo: caption} o lista txt archivo|caption")
    p.add_argument("--default-caption", metavar="TEXT", default="",
                   help="Caption por defecto para archivos sin descripción")
    p.add_argument("--mos",             type=float, default=-1.0, metavar="F",
                   help="Puntuación MOS asignada al corpus (-1 = desconocida)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser("train", help="Entrena o fine-tunea el modelo QA-MDT")
    p.add_argument("--config",      metavar="FILE", default=None,
                   help=f"Config YAML (default: {DEFAULT_CONFIG})")
    p.add_argument("--lmdb-dir",    metavar="DIR", required=True,
                   help="Dataset LMDB (salida de prepare)")
    p.add_argument("--ckpt-dir",    metavar="DIR", default="checkpoints/",
                   help="Directorio para guardar checkpoints (default: checkpoints/)")
    p.add_argument("--resume",      metavar="FILE", default=None,
                   help="Continuar desde este checkpoint")
    p.add_argument("--epochs",      type=int, default=None, metavar="N",
                   help="Épocas aproximadas (se convierte a max_steps ×1000)")
    p.add_argument("--batch-size",  type=int, default=None, metavar="N",
                   help="Batch size (sobreescribe el config)")
    p.add_argument("--lr",          type=float, default=None, metavar="F",
                   help="Learning rate (sobreescribe el config)")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_train)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser("inspect",
                        help="Diagnóstico: config, checkpoint, dataset LMDB")
    p.add_argument("--config",   metavar="FILE", default=None,
                   help="Config YAML a inspeccionar")
    p.add_argument("--ckpt",     metavar="FILE", default=None,
                   help="Checkpoint a inspeccionar")
    p.add_argument("--lmdb-dir", metavar="DIR", default=None,
                   help="Directorio LMDB a inspeccionar")
    p.set_defaults(func=cmd_inspect)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
