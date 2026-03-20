"""
MIDI Music Generator — backend-agnostic prototype
Supports: text2midi, Aria, MMT (swap via MODEL_BACKEND env var)
"""

import os
import time
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr

from backends.base import MusicBackend, GenerationConfig
from backends.text2midi_backend import Text2MidiBackend
from backends.aria_backend import AriaBackend
from backends.mmt_backend import MMTBackend

# ── Backend registry ──────────────────────────────────────────────────────────
BACKENDS: dict[str, type[MusicBackend]] = {
    "text2midi": Text2MidiBackend,
    "aria":      AriaBackend,
    "mmt":       MMTBackend,
}

def load_backend(name: str) -> MusicBackend:
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {list(BACKENDS)}")
    print(f"[app] Loading backend: {name}")
    backend = BACKENDS[name]()
    backend.load()
    print(f"[app] Backend ready: {backend.name} v{backend.version}")
    return backend

# ── State: lazy-loaded backend ────────────────────────────────────────────────
_active_backend: Optional[MusicBackend] = None

def get_backend(name: str) -> MusicBackend:
    global _active_backend
    if _active_backend is None or _active_backend.backend_id != name:
        _active_backend = load_backend(name)
    return _active_backend

# ── Generation pipeline ───────────────────────────────────────────────────────
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def generate(
    prompt: str,
    backend_name: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    bpm: int,
    soundfont_path: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple[str, str, str]:
    """
    Returns (midi_path, audio_path, status_message)
    """
    if not prompt.strip():
        return None, None, "⚠ Please enter a prompt."

    try:
        backend = get_backend(backend_name)

        config = GenerationConfig(
            prompt=prompt.strip(),
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            bpm=bpm,
        )

        progress(0.1, desc="Generating MIDI tokens…")
        t0 = time.time()
        midi_bytes = backend.generate(config)
        elapsed = time.time() - t0

        # Save MIDI
        slug = prompt[:40].replace(" ", "_").replace("/", "-")
        midi_path = OUTPUT_DIR / f"{slug}_{backend_name}.mid"
        midi_path.write_bytes(midi_bytes)

        progress(0.7, desc="Synthesising audio…")
        sf = soundfont_path
        if not sf and hasattr(backend, "soundfont_path") and backend.soundfont_path:
            sf = backend.soundfont_path
        audio_path = synthesise_audio(midi_path, sf, bpm)

        progress(1.0, desc="Done")
        status = (
            f"✓ Generated in {elapsed:.1f}s  ·  backend: {backend.name}  "
            f"·  MIDI: {len(midi_bytes)/1024:.1f} KB"
        )
        return str(midi_path), str(audio_path) if audio_path else None, status

    except Exception as e:
        traceback.print_exc()
        return None, None, f"✗ Error: {e}"


def synthesise_audio(midi_path: Path, soundfont: str, bpm: int) -> Optional[Path]:
    """Convert MIDI → WAV with FluidSynth (optional, skipped if not installed)."""
    import shutil
    if not shutil.which("fluidsynth"):
        print("[audio] fluidsynth not found — skipping audio synthesis")
        return None
    if not soundfont or not Path(soundfont).exists():
        print(f"[audio] soundfont not found at '{soundfont}' — skipping")
        return None

    audio_path = midi_path.with_suffix(".wav")
    cmd = (
        f'fluidsynth -ni "{soundfont}" "{midi_path}" '
        f'-F "{audio_path}" -r 44100 -q'
    )
    ret = os.system(cmd)
    if ret != 0 or not audio_path.exists():
        print(f"[audio] fluidsynth failed (exit {ret})")
        return None
    return audio_path


# ── Gradio UI ─────────────────────────────────────────────────────────────────
DESCRIPTION = """
**Symbolic music generator** — text → MIDI → audio  
Swap the backend to compare text2midi, Aria, or MMT without changing the UI.
"""

with gr.Blocks(title="MIDI Generator") as demo:

    gr.Markdown("# 🎹 Symbolic Music Generator")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="e.g. melancholic jazz piano trio in D minor, slow tempo, 4/4",
                lines=3,
            )
            backend_choice = gr.Radio(
                choices=list(BACKENDS.keys()),
                value="text2midi",
                label="Backend model",
            )

        with gr.Column(scale=1):
            temperature = gr.Slider(0.1, 2.0, value=0.95, step=0.05, label="Temperature")
            top_k       = gr.Slider(1, 200,  value=50,   step=1,    label="Top-k")
            max_tokens  = gr.Slider(128, 4096, value=512, step=128,  label="Max tokens")
            bpm         = gr.Slider(40,  240,  value=120, step=1,    label="BPM hint")

    soundfont = gr.Textbox(
        label="Soundfont path (SF2) — optional, for audio rendering",
        placeholder="/path/to/soundfont.sf2",
        value="",
    )

    run_btn = gr.Button("Generate", variant="primary")

    with gr.Row():
        midi_out  = gr.File(label="MIDI file")
        audio_out = gr.Audio(label="Audio preview", type="filepath")

    status_out = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=generate,
        inputs=[prompt, backend_choice, temperature, top_k, max_tokens, bpm, soundfont],
        outputs=[midi_out, audio_out, status_out],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
