# MIDI Music Generator — backend-agnostic prototype

Text → MIDI → Audio pipeline with swappable generation backends.

```
prompt (text)
    │
    ▼
┌─────────────────────────────────┐
│  Backend (text2midi / Aria / MMT)│  ← swap with one click in the UI
└─────────────┬───────────────────┘
              │ MIDI bytes
              ▼
       FluidSynth + SF2
              │ WAV
              ▼
         Audio preview
```

---

## Quick start (text2midi — recommended for first run)

```bash
# 1. Clone this repo
git clone <this-repo> midi-gen && cd midi-gen

# 2. Create a virtual environment (Python 3.11 required)
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install base dependencies
pip install -r requirements.txt

# 4. Launch
python app.py
# → open http://localhost:7860
```

text2midi downloads its weights (~500 MB) from HuggingFace on first use.
It includes a bundled soundfont, so audio preview works out of the box
if FluidSynth is installed (see below).

---

## Backends

| Backend | Model | Strengths | Python | Status |
|---------|-------|-----------|--------|--------|
| `text2midi` | amaai-lab/text2midi | Multi-track, text→MIDI end-to-end | ≥ 3.10 | ✓ works |
| `aria` | loubb/aria-medium-base | Expressive piano, high quality | ≥ 3.11 | ✓ works |
| `mmt` | salu133445/mmt | Multi-track, research-grade | ≥ 3.10 | manual setup |

### text2midi (default)

No extra setup needed beyond `requirements.txt`. The model uses a custom
architecture (`transformer_model.py`) and REMI tokenizer bundled in the
HuggingFace repo — everything is downloaded automatically on first run.

### Aria

Requires Python 3.11+ and the `ariautils` package:

```bash
# Must use Python 3.11 — ariautils does not support 3.10
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/EleutherAI/aria-utils.git
```

Aria is a continuation model — it generates piano music from a seed sequence
rather than from free text. Text prompts are used as a style hint only.

### MMT (Multitrack Music Transformer)

MMT is not on PyPI and requires manual setup:

```bash
# 1. Clone the repo
git clone https://github.com/salu133445/mmt ~/repos/mmt

# 2. Download the pretrained checkpoint from the GitHub releases page
#    and place it at:
#    ~/repos/mmt/checkpoints/best_model.pt

# 3. Install muspy
pip install muspy
```

The backend looks for the repo at `~/repos/mmt` by default. You can change
this by editing `MMT_REPO_PATH` in `backends/mmt_backend.py`.

---

## Audio rendering (optional)

Install FluidSynth to get WAV previews alongside the MIDI output.
text2midi ships with its own soundfont so no extra download is needed.

```bash
# Ubuntu / Debian
sudo apt install fluidsynth

# macOS
brew install fluidsynth

# Then install the Python bindings
pip install pyfluidsynth
```

If you want higher quality audio, download one of these free soundfonts
and point the UI to its path:

- **GeneralUser GS** — general-purpose, 30 MB
- **SGM-v2.01** — orchestral, 270 MB, excellent quality
- **Fluid R3** — ships with FluidSynth, decent baseline

---

## Generation parameters

| Parameter | Effect |
|-----------|--------|
| `temperature` | Randomness. 0.7 = conservative, 1.2 = creative, >1.5 = chaotic |
| `top_k` | Vocabulary filter. Lower = safer, higher = more variety |
| `max_tokens` | Sequence length. ~512 ≈ 30s, ~2048 ≈ 2 min (model-dependent) |
| `bpm` | Tempo hint — passed to the backend and used in audio rendering |

---

## Adding your own backend

1. Create `backends/my_backend.py`
2. Subclass `MusicBackend` and implement `load()` and `generate()`
3. Register it in `app.py`:

```python
from backends.my_backend import MyBackend

BACKENDS = {
    "text2midi": Text2MidiBackend,
    "aria":      AriaBackend,
    "mmt":       MMTBackend,
    "mine":      MyBackend,   # ← add here
}
```

The UI picks it up automatically. The only contract is:
- `load()` downloads weights and sets `self._loaded = True`
- `generate(config)` returns raw MIDI bytes

---

## Project structure

```
midi-gen/
├── app.py                      # Gradio UI + orchestration
├── requirements.txt
├── outputs/                    # generated MIDI + WAV files
└── backends/
    ├── base.py                 # MusicBackend abstract class + GenerationConfig
    ├── text2midi_backend.py    # amaai-lab/text2midi
    ├── aria_backend.py         # loubb/aria-medium-base
    └── mmt_backend.py          # salu133445/mmt
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'aria'`**
→ The package is called `ariautils`, not `aria`. Install with:
`pip install git+https://github.com/EleutherAI/aria-utils.git`

**`ariautils requires Python >=3.11`**
→ Recreate the virtualenv with Python 3.11:
`python3.11 -m venv .venv && source .venv/bin/activate`

**`ValueError: Unexpected vocab type`**
→ text2midi's `vocab_remi.pkl` contains a MidiTok REMI object, not a dict.
Make sure you are using the latest `text2midi_backend.py`.

**`size mismatch for input_emb.weight`**
→ The model hyperparameters are inferred automatically from the checkpoint.
Make sure you are using the latest `text2midi_backend.py`.

**`Transformer.forward() missing 2 required positional arguments`**
→ text2midi is an encoder-decoder model. The backend handles this internally.
Make sure you are using the latest `text2midi_backend.py`.
