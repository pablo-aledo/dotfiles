#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     LATENTSCORE COMPOSER  v1.1                               ║
║         Síntesis procedimental de audio ambiente desde texto o config        ║
║                                                                              ║
║  Motor de síntesis 100% numpy/scipy, integrado directamente desde           ║
║  latentscore (github.com/beatsbyprabal/latentscore).                         ║
║  Sin dependencias externas salvo: numpy  scipy  soundfile                   ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║                                                                              ║
║    render   Texto o config → WAV                                             ║
║               python latentscore_composer.py render "lluvia en ciudad"       ║
║               python latentscore_composer.py render "jazz cafe" --dur 30     ║
║               python latentscore_composer.py render --config cfg.json        ║
║               python latentscore_composer.py render "calma"                  ║
║                   --patch tempo=slow brightness=dark                         ║
║                                                                              ║
║    morph    Transición suave entre dos textos/configs → WAV                  ║
║               python latentscore_composer.py morph "amanecer" "noche"        ║
║               python latentscore_composer.py morph a.json b.json --dur 60   ║
║               python latentscore_composer.py morph "calma" "tormenta"        ║
║                   --steps 8 --dur 40                                         ║
║                                                                              ║
║    chain    Secuencia de vibes → WAV concatenado con transiciones            ║
║               python latentscore_composer.py chain vibes.txt                 ║
║               python latentscore_composer.py chain "alba" "día" "ocaso"      ║
║                   --dur-each 30 --fade 4                                     ║
║               python latentscore_composer.py chain playlist.json             ║
║                                                                              ║
║    config   Mostrar / editar / exportar un MusicConfig                       ║
║               python latentscore_composer.py config show cfg.json            ║
║               python latentscore_composer.py config from-vibe "lluvia"       ║
║               python latentscore_composer.py config patch cfg.json           ║
║                   tempo=slow brightness=dark --output out.json               ║
║               python latentscore_composer.py config list-styles              ║
║                                                                              ║
║    inspect  Diagnóstico del entorno y del motor de síntesis                  ║
║               python latentscore_composer.py inspect synth                   ║
║               python latentscore_composer.py inspect wav salida.wav          ║
║                                                                              ║
║  PARÁMETROS (--patch / config patch):                                        ║
║    root            c c# d d# e f f# g g# a a# b                             ║
║    mode            major minor dorian mixolydian                             ║
║    tempo           very_slow slow medium fast very_fast                      ║
║    brightness      very_dark dark medium bright very_bright                  ║
║    space           dry small medium large vast                               ║
║    motion          static slow medium fast chaotic                           ║
║    stereo          mono narrow medium wide ultra_wide                        ║
║    echo            none subtle medium heavy infinite                         ║
║    human           robotic tight natural loose drunk                         ║
║    grain           clean warm gritty                                         ║
║    attack          soft medium sharp                                         ║
║    density         2 3 4 5 6                                                 ║
║    bass            drone sustained pulsing walking fifth_drone               ║
║                    sub_pulse octave arp_bass                                 ║
║    pad             warm_slow dark_sustained cinematic thin_high              ║
║                    ambient_drift stacked_fifths bright_open                  ║
║    melody          procedural contemplative rising falling minimal           ║
║                    ornamental arp_melody contemplative_minor                 ║
║                    call_response heroic                                      ║
║    rhythm          none minimal heartbeat soft_four hats_only                ║
║                    electronic kit_light kit_medium military brush            ║
║    texture         none shimmer shimmer_slow vinyl_crackle breath            ║
║                    stars glitch noise_wash crystal pad_whisper               ║
║    accent          none bells pluck chime bells_dense blip                   ║
║                    blip_random brass_hit wind arp_accent piano_note          ║
║    melody_density  very_sparse sparse medium busy very_busy                  ║
║    syncopation     straight light medium heavy                               ║
║    swing           none light medium heavy                                   ║
║    motif_repeat_prob rare sometimes often                                    ║
║    step_bias       step balanced leapy                                       ║
║    chromatic_prob  none light medium heavy                                   ║
║    cadence_strength weak medium strong                                       ║
║    tension_curve   arc ramp waves                                            ║
║    harmony_style   auto pop jazz cinematic ambient                           ║
║    chord_change_bars very_slow slow medium fast                              ║
║    chord_extensions triads sevenths lush                                     ║
║    phrase_len_bars 2 4 8                                                     ║
║    depth           true false                                                ║
║                                                                              ║
║  OPCIONES GLOBALES:                                                          ║
║    --output FILE   Ruta del WAV de salida (default: auto)                   ║
║    --dur N         Duración en segundos (default: 16)                        ║
║    --sr N          Sample rate (default: 44100)                              ║
║    --verbose       Mostrar parámetros resueltos                              ║
║                                                                              ║
║  DEPENDENCIAS:  pip install numpy scipy soundfile                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import math
import textwrap
from pathlib import Path
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Callable, TypeAlias, cast

import numpy as np
from scipy.signal import butter, decimate, lfilter  # type: ignore[import]

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

VERSION     = "1.1"
DEFAULT_SR  = 44_100
DEFAULT_DUR = 16.0

# Literales válidos
ROOT_NOTES    = ["c","c#","d","d#","e","f","f#","g","g#","a","a#","b"]
MODES         = ["major","minor","dorian","mixolydian"]
TEMPOS        = ["very_slow","slow","medium","fast","very_fast"]
BRIGHTNESS    = ["very_dark","dark","medium","bright","very_bright"]
SPACES        = ["dry","small","medium","large","vast"]
MOTIONS       = ["static","slow","medium","fast","chaotic"]
STEREOS       = ["mono","narrow","medium","wide","ultra_wide"]
ECHOES        = ["none","subtle","medium","heavy","infinite"]
HUMANS        = ["robotic","tight","natural","loose","drunk"]
GRAINS        = ["clean","warm","gritty"]
ATTACKS       = ["soft","medium","sharp"]
DENSITIES     = [2,3,4,5,6]
BASS_STYLES   = ["drone","sustained","pulsing","walking","fifth_drone","sub_pulse","octave","arp_bass"]
PAD_STYLES    = ["warm_slow","dark_sustained","cinematic","thin_high","ambient_drift","stacked_fifths","bright_open"]
MELODY_STYLES = ["procedural","contemplative","rising","falling","minimal","ornamental","arp_melody","contemplative_minor","call_response","heroic"]
RHYTHM_STYLES = ["none","minimal","heartbeat","soft_four","hats_only","electronic","kit_light","kit_medium","military","tabla_essence","brush"]
TEXTURE_STLS  = ["none","shimmer","shimmer_slow","vinyl_crackle","breath","stars","glitch","noise_wash","crystal","pad_whisper"]
ACCENT_STLS   = ["none","bells","pluck","chime","bells_dense","blip","blip_random","brass_hit","wind","arp_accent","piano_note"]
MEL_DENSITY   = ["very_sparse","sparse","medium","busy","very_busy"]
SYNCOPATION   = ["straight","light","medium","heavy"]
SWINGS        = ["none","light","medium","heavy"]
MOTIF_REPEAT  = ["rare","sometimes","often"]
STEP_BIAS     = ["step","balanced","leapy"]
CHROMATIC     = ["none","light","medium","heavy"]
CADENCE       = ["weak","medium","strong"]
CHORD_CHANGE  = ["very_slow","slow","medium","fast"]
CHORD_EXT     = ["triads","sevenths","lush"]
PHRASE_BARS   = [2,4,8]
TENSION_CURVES= ["arc","ramp","waves"]
HARMONY_STLS  = ["auto","pop","jazz","cinematic","ambient"]

# ── Mapas label → float (de latentscore/config.py) ──────────────────────────
_TEMPO_MAP       = {"very_slow":0.15,"slow":0.3,"medium":0.5,"fast":0.7,"very_fast":0.9}
_BRIGHTNESS_MAP  = {"very_dark":0.1,"dark":0.3,"medium":0.5,"bright":0.7,"very_bright":0.9}
_SPACE_MAP       = {"dry":0.1,"small":0.3,"medium":0.5,"large":0.7,"vast":0.95}
_MOTION_MAP      = {"static":0.1,"slow":0.3,"medium":0.5,"fast":0.7,"chaotic":0.9}
_STEREO_MAP      = {"mono":0.0,"narrow":0.25,"medium":0.5,"wide":0.75,"ultra_wide":1.0}
_ECHO_MAP        = {"none":0.0,"subtle":0.25,"medium":0.5,"heavy":0.75,"infinite":0.95}
_HUMAN_MAP       = {"robotic":0.0,"tight":0.15,"natural":0.3,"loose":0.5,"drunk":0.8}
_MEL_DENSITY_MAP = {"very_sparse":0.15,"sparse":0.30,"medium":0.50,"busy":0.70,"very_busy":0.85}
_SYNCOPATION_MAP = {"straight":0.0,"light":0.2,"medium":0.5,"heavy":0.8}
_SWING_MAP       = {"none":0.0,"light":0.2,"medium":0.5,"heavy":0.8}
_MOTIF_RPT_MAP   = {"rare":0.2,"sometimes":0.5,"often":0.8}
_STEP_BIAS_MAP   = {"step":0.9,"balanced":0.7,"leapy":0.4}
_CHROMATIC_MAP   = {"none":0.0,"light":0.05,"medium":0.12,"heavy":0.25}
_CADENCE_MAP     = {"weak":0.3,"medium":0.6,"strong":0.9}
_CHORD_CHG_MAP   = {"very_slow":4,"slow":2,"medium":1,"fast":1}

# ══════════════════════════════════════════════════════════════════════════════
#  MUSICCONFIG — interfaz pública (strings legibles)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MusicConfig:
    root:               str  = "c"
    mode:               str  = "major"
    tempo:              str  = "medium"
    brightness:         str  = "medium"
    space:              str  = "medium"
    motion:             str  = "medium"
    stereo:             str  = "medium"
    echo:               str  = "none"
    human:              str  = "robotic"
    grain:              str  = "warm"
    attack:             str  = "soft"
    depth:              bool = False
    density:            int  = 3
    bass:               str  = "drone"
    pad:                str  = "warm_slow"
    melody:             str  = "procedural"
    rhythm:             str  = "none"
    texture:            str  = "none"
    accent:             str  = "none"
    melody_engine:      str  = "procedural"
    melody_density:     str  = "medium"
    syncopation:        str  = "straight"
    swing:              str  = "none"
    motif_repeat_prob:  str  = "sometimes"
    step_bias:          str  = "balanced"
    chromatic_prob:     str  = "none"
    cadence_strength:   str  = "medium"
    tension_curve:      str  = "arc"
    harmony_style:      str  = "auto"
    chord_change_bars:  str  = "medium"
    chord_extensions:   str  = "triads"
    phrase_len_bars:    int  = 4
    register_min_oct:   int  = 3
    register_max_oct:   int  = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MusicConfig":
        campos = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**campos)

    @classmethod
    def from_json(cls, path: str) -> "MusicConfig":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def apply_patch(self, patch: dict) -> "MusicConfig":
        d = self.to_dict()
        for k, v in patch.items():
            if k not in d:
                _die(f"Campo desconocido en patch: {k!r}")
            if isinstance(d[k], bool):
                d[k] = str(v).lower() in ("1","true","yes","sí")
            elif isinstance(d[k], int):
                d[k] = int(v)
            else:
                d[k] = str(v)
        return MusicConfig.from_dict(d)


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHCONFIG — interfaz interna del motor (floats)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SynthConfig:
    """Versión interna con campos numéricos float/int para el motor de síntesis."""
    root:               str   = "c"
    mode:               str   = "major"
    tempo:              float = 0.5
    brightness:         float = 0.5
    space:              float = 0.5
    motion:             float = 0.5
    stereo:             float = 0.5
    echo:               float = 0.0
    human:              float = 0.0
    grain:              str   = "warm"
    attack:             str   = "soft"
    depth:              bool  = False
    density:            int   = 3
    bass:               str   = "drone"
    pad:                str   = "warm_slow"
    melody:             str   = "procedural"
    rhythm:             str   = "none"
    texture:            str   = "none"
    accent:             str   = "none"
    melody_engine:      str   = "procedural"
    melody_density:     float = 0.5
    syncopation:        float = 0.0
    swing:              float = 0.0
    motif_repeat_prob:  float = 0.5
    step_bias:          float = 0.7
    chromatic_prob:     float = 0.0
    cadence_strength:   float = 0.6
    tension_curve:      str   = "arc"
    harmony_style:      str   = "auto"
    chord_change_bars:  int   = 1
    chord_extensions:   str   = "triads"
    phrase_len_bars:    int   = 4
    register_min_oct:   int   = 3
    register_max_oct:   int   = 5

    @classmethod
    def from_music_config(cls, mc: MusicConfig) -> "SynthConfig":
        return cls(
            root             = mc.root,
            mode             = mc.mode,
            tempo            = _TEMPO_MAP.get(mc.tempo, 0.5),
            brightness       = _BRIGHTNESS_MAP.get(mc.brightness, 0.5),
            space            = _SPACE_MAP.get(mc.space, 0.5),
            motion           = _MOTION_MAP.get(mc.motion, 0.5),
            stereo           = _STEREO_MAP.get(mc.stereo, 0.5),
            echo             = _ECHO_MAP.get(mc.echo, 0.0),
            human            = _HUMAN_MAP.get(mc.human, 0.0),
            grain            = mc.grain,
            attack           = mc.attack,
            depth            = mc.depth,
            density          = mc.density,
            bass             = mc.bass,
            pad              = mc.pad,
            melody           = mc.melody,
            rhythm           = mc.rhythm,
            texture          = mc.texture,
            accent           = mc.accent,
            melody_engine    = mc.melody_engine,
            melody_density   = _MEL_DENSITY_MAP.get(mc.melody_density, 0.5),
            syncopation      = _SYNCOPATION_MAP.get(mc.syncopation, 0.0),
            swing            = _SWING_MAP.get(mc.swing, 0.0),
            motif_repeat_prob= _MOTIF_RPT_MAP.get(mc.motif_repeat_prob, 0.5),
            step_bias        = _STEP_BIAS_MAP.get(mc.step_bias, 0.7),
            chromatic_prob   = _CHROMATIC_MAP.get(mc.chromatic_prob, 0.0),
            cadence_strength = _CADENCE_MAP.get(mc.cadence_strength, 0.6),
            tension_curve    = mc.tension_curve,
            harmony_style    = mc.harmony_style,
            chord_change_bars= _CHORD_CHG_MAP.get(mc.chord_change_bars, 1),
            chord_extensions = mc.chord_extensions,
            phrase_len_bars  = mc.phrase_len_bars,
            register_min_oct = mc.register_min_oct,
            register_max_oct = mc.register_max_oct,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "SynthConfig":
        campos = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**campos)


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE SÍNTESIS  (adaptado de latentscore/synth.py — MIT License)
#  github.com/beatsbyprabal/latentscore
# ══════════════════════════════════════════════════════════════════════════════

NDArray = np.ndarray  # alias local

# Tipos usados en el motor
FloatArray: TypeAlias = NDArray
OscFn: TypeAlias = Callable[[float, float, int, float, float], FloatArray]
PatternFn: TypeAlias = Callable[
    ["SynthConfig", float, Any, Any, np.random.Generator, float],
    FloatArray,
]

# Tipos de campos categóricos (aliases para compatibilidad con synth.py)
RootNote = str
ModeName = str
BassStyle = str
PadStyle = str
RhythmStyle = str
TextureStyle = str
AccentStyle = str
AttackStyle = str
GrainStyle = str
ChordExtensions = str
DensityLevel = int


# =============================================================================
# CONSTANTS
# =============================================================================

SAMPLE_RATE = 44100

# Feature flag: when True, rhythm patterns use config.tempo for beat spacing.
# When False, rhythm patterns divide the chunk duration into fixed slots (legacy behavior).
# Set to False to revert to old behavior if needed.
TEMPO_AWARE_RHYTHM = True

# Root note frequencies (octave 4) - mapping proxy for immutability
NOTE_FREQS: Mapping[RootNote, float] = MappingProxyType(
    {
        "c": 261.63,
        "c#": 277.18,
        "d": 293.66,
        "d#": 311.13,
        "e": 329.63,
        "f": 349.23,
        "f#": 369.99,
        "g": 392.00,
        "g#": 415.30,
        "a": 440.00,
        "a#": 466.16,
        "b": 493.88,
    }
)

# Root to semitone offset
ROOT_SEMITONES: Mapping[RootNote, int] = MappingProxyType(
    {
        "c": 0,
        "c#": 1,
        "d": 2,
        "d#": 3,
        "e": 4,
        "f": 5,
        "f#": 6,
        "g": 7,
        "g#": 8,
        "a": 9,
        "a#": 10,
        "b": 11,
    }
)

# Mode intervals (semitones from root) - tuples for JIT
MODE_INTERVALS: Mapping[ModeName, tuple[int, ...]] = MappingProxyType(
    {
        "major": (0, 2, 4, 5, 7, 9, 11),
        "minor": (0, 2, 3, 5, 7, 8, 10),
        "dorian": (0, 2, 3, 5, 7, 9, 10),
        "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    }
)

# V2 parameter mappings
ATTACK_MULT: Mapping[AttackStyle, float] = MappingProxyType(
    {"soft": 2.5, "medium": 1.0, "sharp": 0.3}
)
GRAIN_OSC: Mapping[GrainStyle, str] = MappingProxyType(
    {"clean": "sine", "warm": "triangle", "gritty": "sawtooth"}
)

# Density → active layers (tuples)
DENSITY_LAYERS: Mapping[DensityLevel, tuple[str, ...]] = MappingProxyType(
    {
        2: ("bass", "pad"),
        3: ("bass", "pad", "melody"),
        4: ("bass", "pad", "melody", "rhythm"),
        5: ("bass", "pad", "melody", "rhythm", "texture"),
        6: ("bass", "pad", "melody", "rhythm", "texture", "accent"),
    }
)

FloatArray: TypeAlias = NDArray[np.float64]
# OscFn signature: (freq, duration, sr, amp, t_offset) -> audio
OscFn: TypeAlias = Callable[[float, float, int, float, float], FloatArray]

# =============================================================================
# PART 1: SYNTHESIS PRIMITIVES
# =============================================================================

def freq_from_note(root: RootNote, semitones: int = 0, octave: int = 4) -> float:
    """Get frequency for a note."""
    root_value: RootNote = root
    if root_value not in NOTE_FREQS:
        raise ValueError(f"Unknown root note: {root}. Valid: {list(NOTE_FREQS.keys())}")
    base_freq = NOTE_FREQS[root_value]
    octave_shift = octave - 4
    return base_freq * (2**octave_shift) * (2 ** (semitones / 12))

def generate_sine(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate sine wave with optional time offset for phase continuity."""
    t = np.linspace(t_offset, t_offset + duration, int(sr * duration), False)
    return amp * np.sin(2 * np.pi * freq * t)

def generate_triangle(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate triangle wave with optional time offset for phase continuity."""
    t = np.linspace(t_offset, t_offset + duration, int(sr * duration), False)
    return amp * 2 * np.abs(2 * (t * freq - np.floor(t * freq + 0.5))) - amp

def generate_sawtooth(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    oversample: int = 2,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate anti-aliased sawtooth using 4-point PolyBLEP + oversampling.

    Note on streaming/chunk boundaries: The zero-phase decimation filter causes
    small transients at chunk boundaries when chunks are rendered independently.
    These artifacts are typically inaudible for ambient music due to:
    - Layering with other instruments
    - Reverb and delay effects
    - The nature of slowly-evolving ambient textures

    For perfect phase continuity, consider using sine/triangle oscillators
    which don't require decimation.
    """

    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample  # <- exact multiple
    dt = freq / sr_high

    # Phase accumulator: start from t_offset, go for duration
    t = np.linspace(t_offset * freq, (t_offset + duration) * freq, num_samples_high, endpoint=False)
    t = t % 1.0
    naive = 2.0 * t - 1.0
    correction = np.zeros(num_samples_high)

    # 4-point PolyBLEP correction
    # Region 1: 0 <= t < dt
    m1 = t < dt
    t1 = t[m1] / dt
    correction[m1] = t1 * t1 * (2 * t1 - 3) + 1

    # Region 2: dt <= t < 2*dt
    m2 = (t >= dt) & (t < 2 * dt)
    t2 = t[m2] / dt - 1
    correction[m2] = t2 * t2 * (2 * t2 - 3)

    # Region 3: 1-2*dt < t <= 1-dt
    m3 = (t > 1 - 2 * dt) & (t <= 1 - dt)
    t3 = (t[m3] - 1) / dt + 1
    correction[m3] = t3 * t3 * (2 * t3 + 3)

    # Region 4: 1-dt < t < 1
    m4 = t > 1 - dt
    t4 = (t[m4] - 1) / dt
    correction[m4] = t4 * t4 * (2 * t4 + 3) + 1

    signal_high = naive - correction
    signal = decimate(signal_high, oversample, ftype="fir", zero_phase=True)

    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    output = amp * signal
    assert isinstance(output, np.ndarray)
    return cast(FloatArray, output)

def generate_square(
    freq: float,
    duration: float,
    sr: int = SAMPLE_RATE,
    amp: float = 0.3,
    oversample: int = 2,
    t_offset: float = 0.0,
) -> FloatArray:
    """Generate anti-aliased square wave using 4-point PolyBLEP + oversampling.

    Note: Same streaming caveat as generate_sawtooth - see docstring there.
    """

    num_samples = int(sr * duration)
    sr_high = sr * oversample
    num_samples_high = num_samples * oversample  # <- exact multiple
    dt = freq / sr_high

    # Phase accumulator: start from t_offset, go for duration
    t = np.linspace(t_offset * freq, (t_offset + duration) * freq, num_samples_high, endpoint=False)
    t = t % 1.0
    naive = np.where(t < 0.5, 1.0, -1.0)
    correction = np.zeros(num_samples_high)

    def apply_4pt_blep(phase: FloatArray, corr: FloatArray, sign: float) -> None:
        """Apply 4-point PolyBLEP correction at discontinuity."""
        # Region 1: 0 <= phase < dt
        m1 = phase < dt
        t1 = phase[m1] / dt
        corr[m1] += sign * (t1 * t1 * (2 * t1 - 3) + 1)

        # Region 2: dt <= phase < 2*dt
        m2 = (phase >= dt) & (phase < 2 * dt)
        t2 = phase[m2] / dt - 1
        corr[m2] += sign * (t2 * t2 * (2 * t2 - 3))

        # Region 3: 1-2*dt < phase <= 1-dt
        m3 = (phase > 1 - 2 * dt) & (phase <= 1 - dt)
        t3 = (phase[m3] - 1) / dt + 1
        corr[m3] += sign * (t3 * t3 * (2 * t3 + 3))

        # Region 4: 1-dt < phase < 1
        m4 = phase > 1 - dt
        t4 = (phase[m4] - 1) / dt
        corr[m4] += sign * (t4 * t4 * (2 * t4 + 3) + 1)

    # Rising edge at phase = 0
    assert isinstance(t, np.ndarray)
    t_array = cast(FloatArray, t)
    apply_4pt_blep(t_array, correction, 1.0)

    # Falling edge at phase = 0.5
    t_shifted = (t + 0.5) % 1.0
    assert isinstance(t_shifted, np.ndarray)
    t_shifted_array = cast(FloatArray, t_shifted)
    apply_4pt_blep(t_shifted_array, correction, -1.0)

    signal_high = naive + correction
    signal = decimate(signal_high, oversample, ftype="fir", zero_phase=True)

    # Ensure exact output length
    if len(signal) > num_samples:
        signal = signal[:num_samples]
    elif len(signal) < num_samples:
        signal = np.pad(signal, (0, num_samples - len(signal)))

    output = amp * signal
    assert isinstance(output, np.ndarray)
    return cast(FloatArray, output)

def generate_noise(duration: float, sr: int = SAMPLE_RATE, amp: float = 0.1) -> FloatArray:
    """Generate white noise."""
    return amp * np.random.randn(int(sr * duration))

# NOTE: sawtooth/square have an extra 'oversample' parameter, so types don't
# match OscFn exactly. Callers use keyword args for t_offset, which works at runtime.
OSC_FUNCTIONS: Mapping[str, OscFn] = MappingProxyType(  # type: ignore[assignment]
    {
        "sine": generate_sine,
        "triangle": generate_triangle,
        "sawtooth": generate_sawtooth,
        "square": generate_square,
    }
)

def apply_adsr(
    signal: FloatArray,
    attack: float,
    decay: float,
    sustain: float,
    release: float,
    sr: int = SAMPLE_RATE,
) -> FloatArray:
    """Apply ADSR envelope to signal."""
    # Minimum times to prevent clicks (5ms attack, 10ms release)
    attack = max(attack, 0.005)
    release = max(release, 0.01)

    total = len(signal)
    a_samples = int(attack * sr)
    d_samples = int(decay * sr)
    r_samples = int(release * sr)
    s_samples = max(0, total - a_samples - d_samples - r_samples)

    envelope = np.concatenate(
        (
            np.linspace(0, 1, max(1, a_samples)),
            np.linspace(1, sustain, max(1, d_samples)),
            np.ones(max(1, s_samples)) * sustain,
            np.linspace(sustain, 0, max(1, r_samples)),
        )
    )

    # Match length
    if len(envelope) < total:
        envelope = np.pad(envelope, (0, total - len(envelope)))
    else:
        envelope = envelope[:total]

    return signal * envelope

def _quantize(value: float, step: float = 0.001) -> float:
    return round(value / step) * step

@lru_cache(maxsize=512)
def _butter_cached(
    kind: str, normalized_cutoff: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    coeffs = butter(2, normalized_cutoff, btype=kind, output="ba")
    assert isinstance(coeffs, tuple)
    assert len(coeffs) == 2
    b_raw, a_raw = coeffs
    assert isinstance(b_raw, np.ndarray)
    assert isinstance(a_raw, np.ndarray)
    b = np.asarray(b_raw, dtype=np.float64)
    a = np.asarray(a_raw, dtype=np.float64)
    return b, a

def apply_lowpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply lowpass filter (causal, analog-style)."""
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    b, a = _butter_cached("low", _quantize(normalized))
    filtered = lfilter(b, a, signal)
    return np.asarray(filtered, dtype=np.float64)

def apply_highpass(signal: FloatArray, cutoff: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply highpass filter (causal, analog-style)."""
    nyquist = sr / 2
    normalized = min(max(cutoff / nyquist, 0.001), 0.99)
    b, a = _butter_cached("high", _quantize(normalized))
    filtered = lfilter(b, a, signal)
    return np.asarray(filtered, dtype=np.float64)

def apply_delay(
    signal: FloatArray, delay_time: float, feedback: float, wet: float, sr: int = SAMPLE_RATE
) -> FloatArray:
    """Apply delay effect."""
    delay_samples = int(delay_time * sr)
    output = signal.copy()

    for i in range(1, 5):  # 5 delay taps
        offset = delay_samples * i
        if 0 < offset < len(signal):
            output[offset:] += signal[:-offset] * (feedback**i) * wet

    return output

def apply_reverb(signal: FloatArray, room: float, size: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Simple reverb via multiple delays.

    Pre-suaviza discontinuidades de fase antes de difundirlas por las líneas
    de delay: un salto brusco en la fuente se copiaría en cada tap del reverb
    multiplicando los chasquidos. El pre-suavizado los elimina en el origen.
    """
    # Pre-suavizado: elimina saltos > 0.02 con ventana de 5ms
    # antes de que el reverb los replique en cada tap
    n = len(signal)
    src = signal.copy().astype(np.float64)
    diff = np.abs(np.diff(src))
    clicks = np.where(diff > 0.02)[0]
    fade_n = max(8, int(0.005 * sr))  # 5ms
    for c in clicks:
        i0 = max(0, c - fade_n)
        i1 = min(n, c + fade_n + 1)
        seg = src[i0:i1]
        k   = min(len(seg) // 2, max(4, fade_n // 4))
        smoothed = np.convolve(seg, np.ones(k) / k, mode="same")
        hann = np.hanning(len(seg))
        src[i0:i1] = seg * (1.0 - hann) + smoothed * hann
    src = src.astype(signal.dtype)

    output = src.copy()
    delays = (0.029, 0.037, 0.041, 0.053, 0.067)
    for i, delay in enumerate(delays):
        delay_samples = int(delay * size * sr)
        if 0 < delay_samples < len(src):
            output[delay_samples:] += src[:-delay_samples] * room * (0.7 ** i)

    return output

def apply_stereo_width(signal: FloatArray, width: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply a subtle micro-delay to simulate stereo width while staying mono."""
    width = float(np.clip(width, 0.0, 1.0))
    if width <= 0.0 or signal.size == 0:
        return signal

    min_delay_sec = 0.001
    max_delay_sec = 0.01
    mix_max = 0.12

    delay_time = min_delay_sec + (max_delay_sec - min_delay_sec) * width
    delay_samples = int(delay_time * sr)
    if delay_samples <= 0 or delay_samples >= len(signal):
        return signal

    mix = mix_max * width
    delayed = np.zeros_like(signal)
    delayed[delay_samples:] = signal[:-delay_samples]
    output = signal * (1.0 - mix) + delayed * mix
    return output

def generate_lfo(duration: float, rate: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Generate LFO signal (0 to 1 range)."""
    t = np.linspace(0, duration, int(sr * duration), False)
    return 0.5 + 0.5 * np.sin(2 * np.pi * rate * t)

def apply_humanize(signal: FloatArray, amount: float, sr: int = SAMPLE_RATE) -> FloatArray:
    """Apply subtle timing/amplitude humanization."""
    if amount <= 0:
        return signal

    # Subtle amplitude variation
    amp_lfo = 1.0 + (np.random.randn(len(signal)) * amount * 0.1)
    amp_lfo = np.clip(amp_lfo, 0.9, 1.1)

    return signal * amp_lfo

def add_note(signal: FloatArray, note: FloatArray, start_index: int) -> None:
    """Safely adds a note to the signal buffer, clipping if necessary."""
    if start_index >= len(signal):
        return

    end_index = start_index + len(note)

    if end_index <= len(signal):
        signal[start_index:end_index] += note
    else:
        # Clip the note to fit the remaining signal space
        available = len(signal) - start_index
        clipped = note[:available].copy()

        # Apply quick fade-out to prevent click from abrupt cutoff
        fade_samples = min(int(SAMPLE_RATE * 0.01), available // 4)  # 10ms max
        if fade_samples > 1:
            clipped[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        signal[start_index:] += clipped

# =============================================================================
# PART 2: PATTERN GENERATORS
# =============================================================================

# -----------------------------------------------------------------------------
# HARMONY + PROCEDURAL MELODY STRUCTURES (compute-light)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChordEvent:
    """A chord specified in scale degrees relative to the key, scheduled in beats."""

    start_beat: float
    duration_beats: float
    root_degree: int

@dataclass(frozen=True)
class HarmonyPlan:
    """A sequence of chord events spanning the render duration."""

    chords: tuple[ChordEvent, ...] = ()
    beats_per_bar: int = 4

    def chord_at_beat(self, beat: float) -> ChordEvent:
        """Get the chord event active at the given beat (clamped)."""
        if not self.chords:
            return ChordEvent(0.0, float(self.beats_per_bar), 0)

        for chord in self.chords:
            if chord.start_beat <= beat < (chord.start_beat + chord.duration_beats):
                return chord

        if beat < self.chords[0].start_beat:
            return self.chords[0]
        return self.chords[-1]

@dataclass(frozen=True)
class MelodyPolicy:
    """Knobs for compute-light, phrase-aware, chord-aware melody generation."""

    phrase_len_bars: int = 4
    density: float = 0.45
    syncopation: float = 0.20
    swing: float = 0.0
    motif_repeat_prob: float = 0.50
    step_bias: float = 0.75
    chromatic_prob: float = 0.05
    cadence_strength: float = 0.65
    register_min_oct: int = 4
    register_max_oct: int = 6
    tension_curve: str = "arc"

@dataclass(frozen=True)
class NoteEvent:
    """A rendered melody note event (frequency already resolved)."""

    start_sec: float
    dur_sec: float
    freq: float
    amp: float
    is_anchor: bool = False

def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

def _tension_value(curve: str, x: float) -> float:
    """x in [0..1] -> tension in [0..1]."""
    x = _clamp01(x)
    if curve == "ramp":
        return x
    if curve == "waves":
        return 0.5 - 0.5 * float(np.cos(2.0 * np.pi * 2.0 * x))
    return float(np.sin(np.pi * x))

def _iter_chord_segments(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
) -> Iterable[tuple[float, float, int]]:
    """Yield (start_sec, dur_sec, chord_root_degree) segments."""
    if harmony is None or not harmony.chords:
        yield (0.0, duration, 0)
        return

    seconds_per_beat = _seconds_per_beat(config)
    for chord in harmony.chords:
        start = chord.start_beat * seconds_per_beat
        if start >= duration:
            break
        dur = chord.duration_beats * seconds_per_beat
        end = min(duration, start + dur)
        dur = max(0.0, end - start)
        if dur > 0:
            yield (start, dur, int(chord.root_degree))

def _weighted_choice(
    rng: np.random.Generator, items: Sequence[int], weights: Sequence[float]
) -> int:
    """Small helper around rng.choice for int items."""
    if not items:
        raise ValueError("No items to choose from")
    w = np.asarray(weights, dtype=np.float64)
    if np.all(w <= 0):
        w = np.ones_like(w)
    w = w / np.sum(w)
    idx = int(rng.choice(len(items), p=w))
    return int(items[idx])

def _chord_tones_ascending(
    chord_extensions: ChordExtensions, chord_root_degree: int
) -> tuple[int, ...]:
    """Chord tones as ascending scale degrees above the chord root (best for voicings)."""
    degree = int(chord_root_degree)
    tones: list[int] = [degree, degree + 2, degree + 4]
    if chord_extensions in ("sevenths", "lush"):
        tones.append(degree + 6)
    if chord_extensions == "lush":
        tones.append(degree + 8)
    return tuple(tones)

def _attack_mult(config: SynthConfig) -> float:
    return ATTACK_MULT.get(config.attack, 1.0)

def _osc_type(config: SynthConfig) -> str:
    return GRAIN_OSC.get(config.grain, "sine")

def _echo_mult(config: SynthConfig) -> float:
    return config.echo / 0.5

def _bpm(config: SynthConfig) -> float:
    t = float(np.clip(config.tempo, 0.0, 1.0))
    return 55.0 + 110.0 * t

def _seconds_per_beat(config: SynthConfig) -> float:
    return 60.0 / _bpm(config)

def _beats_total(config: SynthConfig, duration: float) -> float:
    return duration / _seconds_per_beat(config)

def _rhythm_slot_times(
    config: SynthConfig,
    duration: float,
    slots_per_beat: int = 2,
    t_offset: float = 0.0,
) -> tuple[list[float], int]:
    """
    Generate slot times for rhythm patterns based on tempo.

    Args:
        config: Synth config containing tempo
        slots_per_beat: Number of subdivisions per beat (2 = 8th notes, 4 = 16th notes)
        t_offset: Time offset for pattern continuity across chunks

    Returns:
        Tuple of (slot_times, slot_offset) where:
        - slot_times: List of time positions (in seconds) within the chunk, aligned to global grid
        - slot_offset: Starting pattern index based on t_offset (for pattern continuity)
    """
    if not TEMPO_AWARE_RHYTHM:
        # Legacy: not used in this path, but kept for clarity
        return [], 0

    spb = _seconds_per_beat(config)
    slot_dur = spb / slots_per_beat
    if slot_dur <= 0:
        return [], 0

    # Calculate which global slot contains t_offset
    slot_at_offset = int(t_offset / slot_dur)
    global_slot_start = slot_at_offset * slot_dur

    # If t_offset is exactly at a slot boundary, start there
    # Otherwise, start at the next slot
    if abs(global_slot_start - t_offset) < 1e-9:
        first_slot = slot_at_offset
    else:
        first_slot = slot_at_offset + 1

    # Generate slot times aligned to the global grid
    times: list[float] = []
    current_slot = first_slot
    while True:
        local_time = current_slot * slot_dur - t_offset
        if local_time >= duration:
            break
        times.append(local_time)
        current_slot += 1

    return times, first_slot

def _semitone_from_degree(mode: ModeName, degree: int) -> int:
    intervals = MODE_INTERVALS.get(mode, MODE_INTERVALS["minor"])
    return int(intervals[degree % len(intervals)] + (12 * (degree // len(intervals))))

def _scale_freq(config: SynthConfig, degree: int, octave: int = 4) -> float:
    semitone = _semitone_from_degree(config.mode, degree)
    return freq_from_note(config.root, semitone, octave)

def _chord_tone_classes(
    chord_root_degree: int, chord_extensions: ChordExtensions
) -> tuple[int, ...]:
    root_degree = int(chord_root_degree) % 7
    tones: list[int] = [root_degree, (root_degree + 2) % 7, (root_degree + 4) % 7]
    if chord_extensions in ("sevenths", "lush"):
        tones.append((root_degree + 6) % 7)
    if chord_extensions == "lush":
        tones.append((root_degree + 1) % 7)
    out: list[int] = []
    for tone in tones:
        if tone not in out:
            out.append(tone)
    return tuple(out)

def _chord_root_degree_at_beat(beat: float, harmony: HarmonyPlan | None) -> int:
    if harmony is None:
        return 0
    return int(harmony.chord_at_beat(float(beat)).root_degree) % 7

PatternFn: TypeAlias = Callable[
    [SynthConfig, float, HarmonyPlan | None, MelodyPolicy | None, np.random.Generator, float],
    FloatArray,
]
# PatternFn signature: (config, duration, harmony, melody_policy, rng, t_offset) -> audio

# -----------------------------------------------------------------------------
# BASS PATTERNS
# -----------------------------------------------------------------------------

def bass_drone(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Sustained drone bass (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)

    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)
        freq = _scale_freq(config, chord_root, 2)

        note = generate_sine(freq, note_dur, sr, 0.35, t_offset=t_offset + start_sec)
        note = apply_lowpass(note, 80 * config.brightness + 20, sr)
        note = apply_adsr(
            note,
            1.8 * attack_mult,
            0.5,
            0.95,
            2.2 * attack_mult,
            sr,
        )
        add_note(signal, note, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.5, config.space, sr)
    return signal

def bass_sustained(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Long sustained bass notes that follow chord roots."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        pattern = (0, 0, 4, 0)
        note_dur = seg_dur / len(pattern)

        for i, rel_degree in enumerate(pattern):
            deg = chord_root + rel_degree
            freq = _scale_freq(config, deg, 2)
            start = int((start_sec + i * note_dur) * sr)

            note_start_sec = start_sec + i * note_dur
            note = generate_sine(
                freq, note_dur * 0.95, sr, 0.32, t_offset=t_offset + note_start_sec
            )
            note = apply_lowpass(note, 100 * config.brightness + 30, sr)
            note = apply_adsr(
                note,
                0.7 * attack_mult,
                0.3,
                0.85,
                1.2 * attack_mult,
                sr,
            )
            add_note(signal, note, start)

    signal = apply_reverb(signal, config.space * 0.4, config.space, sr)
    return signal

def bass_pulsing(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Rhythmic pulsing bass (chord-aware)."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)
    seconds_per_beat = _seconds_per_beat(config)
    beats_total = max(1.0, _beats_total(config, duration))
    pulses_per_beat = 2.0
    num_pulses = int(np.ceil(beats_total * pulses_per_beat))
    pulse_beats = 1.0 / pulses_per_beat
    pulse_sec = pulse_beats * seconds_per_beat

    signal = np.zeros(int(sr * dur))

    for i in range(num_pulses):
        start_sec = i * pulse_sec
        if start_sec >= dur:
            break

        beat = i * pulse_beats
        chord_root = _chord_root_degree_at_beat(beat, harmony)
        freq = _scale_freq(config, chord_root, 2)

        note = generate_sine(freq, pulse_sec * 0.82, sr, 0.35, t_offset=t_offset + start_sec)
        note = apply_lowpass(note, 90 * config.brightness + 20, sr)
        note = apply_adsr(
            note,
            0.02 * attack_mult,
            0.08,
            0.6,
            0.25 * attack_mult,
            sr,
        )
        add_note(signal, note, int(start_sec * sr))

    return signal

def bass_walking(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Walking bass line that follows the chord root (diatonic)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    beats_total = int(np.ceil(_beats_total(config, duration)))
    note_sec = _seconds_per_beat(config)

    for i in range(beats_total):
        start_sec = i * note_sec
        if start_sec >= dur_total:
            break

        chord_root = _chord_root_degree_at_beat(float(i), harmony)
        rel = (0, 2, 4, 2)[i % 4]
        degree = chord_root + rel

        freq = _scale_freq(config, degree, 2)
        note = generate_triangle(freq, note_sec * 0.92, sr, 0.30, t_offset=t_offset + start_sec)
        note = apply_lowpass(note, 120 * config.brightness + 40, sr)
        note = apply_adsr(
            note,
            0.04 * attack_mult,
            0.12,
            0.7,
            0.22 * attack_mult,
            sr,
        )
        note = apply_humanize(note, config.human, sr)
        add_note(signal, note, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.3, config.space * 0.8, sr)
    return signal

def bass_fifth_drone(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Root + fifth drone that follows chord changes."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        root_freq = _scale_freq(config, chord_root, 2)
        fifth_freq = _scale_freq(config, chord_root + 4, 2)

        root = generate_sine(root_freq, note_dur, sr, 0.26, t_offset=t_offset + start_sec)
        root = apply_lowpass(root, 70 * config.brightness + 20, sr)
        root = apply_adsr(root, 2.2 * attack_mult, 0.5, 0.95, 2.6 * attack_mult, sr)

        fifth = generate_sine(fifth_freq, note_dur, sr, 0.18, t_offset=t_offset + start_sec)
        fifth = apply_lowpass(fifth, 100 * config.brightness + 30, sr)
        fifth = apply_adsr(fifth, 2.4 * attack_mult, 0.5, 0.9, 2.6 * attack_mult, sr)

        add_note(signal, root + fifth, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.5, config.space, sr)
    return signal

def bass_sub_pulse(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Deep sub-bass pulse (chord-aware)."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur))

    beats_total = max(1.0, _beats_total(config, duration))
    pulses_per_bar = 2
    beats_per_pulse = 4.0 / pulses_per_bar
    pulse_sec = beats_per_pulse * _seconds_per_beat(config)

    num_pulses = int(np.ceil(beats_total / beats_per_pulse))
    for i in range(num_pulses):
        start_sec = i * pulse_sec
        if start_sec >= dur:
            break

        beat = i * beats_per_pulse
        chord_root = _chord_root_degree_at_beat(beat, harmony)

        freq = _scale_freq(config, chord_root, 1)
        note = generate_sine(freq, pulse_sec * 0.95, sr, 0.4, t_offset=t_offset + start_sec)
        note = apply_lowpass(note, 55, sr)
        note = apply_adsr(
            note,
            0.25 * attack_mult,
            0.2,
            0.9,
            0.7 * attack_mult,
            sr,
        )
        add_note(signal, note, int(start_sec * sr))

    return signal

BASS_PATTERNS: Mapping[BassStyle, PatternFn] = MappingProxyType(
    {
        "drone": bass_drone,
        "sustained": bass_sustained,
        "pulsing": bass_pulsing,
        "walking": bass_walking,
        "fifth_drone": bass_fifth_drone,
        "sub_pulse": bass_sub_pulse,
        "octave": bass_sustained,
        "arp_bass": bass_pulsing,
    }
)

# -----------------------------------------------------------------------------
# PAD PATTERNS
# -----------------------------------------------------------------------------

def pad_warm_slow(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Warm, slowly evolving pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_sine)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for degree in _chord_tones_ascending(config.chord_extensions, chord_root)[:3]:
            freq = _scale_freq(config, degree, 3)
            tone = osc(freq, note_dur, sr, 0.15, t_offset=t_offset + start_sec)  # type: ignore[call-arg]

            lfo_rate = 0.1 / (config.motion + 0.1)
            lfo = generate_lfo(note_dur, lfo_rate, sr)

            base_cutoff = 300 * config.brightness + 100
            tone_low = apply_lowpass(tone, base_cutoff * 0.5, sr)
            tone_high = apply_lowpass(tone, base_cutoff * 1.5, sr)
            tone = tone_low * (1 - lfo) + tone_high * lfo

            tone = apply_adsr(tone, 1.5 * attack_mult, 0.8, 0.85, 2.5 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.7, config.space * 0.9, sr)
    signal = apply_delay(signal, 0.35, 0.3 * echo_mult, 0.25 * echo_mult, sr)
    signal = apply_humanize(signal, config.human, sr)

    return signal

def pad_dark_sustained(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Dark, heavy sustained pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for degree in _chord_tones_ascending(config.chord_extensions, chord_root)[:3]:
            freq = _scale_freq(config, degree, 3)
            tone = generate_sawtooth(freq, note_dur, sr, 0.12, t_offset=t_offset + start_sec)
            tone = apply_lowpass(tone, 200 * config.brightness + 80, sr)
            tone = apply_adsr(tone, 2.0 * attack_mult, 1.0, 0.9, 3.0 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.8, config.space, sr)
    return signal

def pad_cinematic(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Big, cinematic pad with movement (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)
    signal = np.zeros(int(sr * dur_total))

    voicings = ((0, 3), (2, 3), (4, 3), (0, 4), (4, 4))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.10, dur_total - start_sec)

        for rel_degree, octave in voicings:
            degree = chord_root + rel_degree
            freq = _scale_freq(config, degree, octave)

            tone = generate_sawtooth(freq, note_dur, sr, 0.08, t_offset=t_offset + start_sec)
            tone += generate_triangle(
                freq * 1.002, note_dur, sr, 0.06, t_offset=t_offset + start_sec
            )

            tone = apply_lowpass(tone, 400 * config.brightness + 150, sr)
            tone = apply_adsr(tone, 1.8 * attack_mult, 0.8, 0.88, 2.8 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.85, config.space, sr)
    signal = apply_delay(signal, 0.4, 0.35 * echo_mult, 0.3 * echo_mult, sr)

    return signal

def pad_thin_high(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Thin, high pad (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.15, dur_total - start_sec)

        for rel_degree in (0, 4):
            degree = chord_root + rel_degree
            freq = _scale_freq(config, degree, 4)
            tone = generate_sine(freq, note_dur, sr, 0.12, t_offset=t_offset + start_sec)
            tone = apply_lowpass(tone, 800 * config.brightness + 200, sr)
            tone = apply_highpass(tone, 200, sr)
            tone = apply_adsr(tone, 1.2 * attack_mult, 0.6, 0.8, 2.0 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.75, config.space * 0.95, sr)
    return signal

def pad_ambient_drift(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Slowly drifting ambient pad (chord-aware, with gentle color notes)."""
    sr = SAMPLE_RATE
    dur_total = duration
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)
    signal = np.zeros(int(sr * dur_total))

    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.35, dur_total - start_sec)

        tones = list(_chord_tones_ascending(config.chord_extensions, chord_root)[:3])

        if config.motion > 0.4 and rng.random() < (0.15 + 0.25 * config.motion):
            tones.append(chord_root + 5)

        for degree in tones:
            freq = _scale_freq(config, degree, 3)
            tone = generate_sine(freq, note_dur, sr, 0.14, t_offset=t_offset + start_sec)
            tone = apply_lowpass(tone, 350 * config.brightness + 100, sr)
            tone = apply_adsr(tone, 1.6 * attack_mult, 0.5, 0.85, 2.2 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.8, config.space, sr)
    signal = apply_delay(signal, 0.5, 0.4 * echo_mult, 0.35 * echo_mult, sr)

    return signal

def pad_stacked_fifths(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Fifths stacked for a powerful sound (chord-aware)."""
    sr = SAMPLE_RATE
    dur_total = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)

    signal = np.zeros(int(sr * dur_total))

    voicings = ((0, 3), (4, 3), (0, 4), (4, 4))
    for start_sec, seg_dur, chord_root in _iter_chord_segments(config, duration, harmony):
        note_dur = min(seg_dur * 1.10, dur_total - start_sec)

        for rel_degree, octave in voicings:
            degree = chord_root + rel_degree
            freq = _scale_freq(config, degree, octave)
            tone = osc(freq, note_dur, sr, 0.10, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
            tone = apply_lowpass(tone, 500 * config.brightness + 150, sr)
            tone = apply_adsr(tone, 1.3 * attack_mult, 0.7, 0.88, 2.2 * attack_mult, sr)
            add_note(signal, tone, int(start_sec * sr))

    signal = apply_reverb(signal, config.space * 0.7, config.space * 0.85, sr)
    return signal

PAD_PATTERNS: Mapping[PadStyle, PatternFn] = MappingProxyType(
    {
        "warm_slow": pad_warm_slow,
        "dark_sustained": pad_dark_sustained,
        "cinematic": pad_cinematic,
        "thin_high": pad_thin_high,
        "ambient_drift": pad_ambient_drift,
        "stacked_fifths": pad_stacked_fifths,
        "bright_open": pad_thin_high,
    }
)

# -----------------------------------------------------------------------------
# MELODY PATTERNS
# -----------------------------------------------------------------------------

def _resolve_melody_policy(
    policy: MelodyPolicy | None,
    config: SynthConfig,
) -> MelodyPolicy:
    """Return the active MelodyPolicy with safety clamps."""
    base = policy or build_melody_policy(config)
    phrase_len = int(max(1, min(16, base.phrase_len_bars)))
    reg_min = int(min(base.register_min_oct, base.register_max_oct))
    reg_max = int(max(base.register_min_oct, base.register_max_oct))
    return MelodyPolicy(
        phrase_len_bars=phrase_len,
        density=_clamp01(base.density),
        syncopation=_clamp01(base.syncopation),
        swing=_clamp01(base.swing),
        motif_repeat_prob=_clamp01(base.motif_repeat_prob),
        step_bias=_clamp01(base.step_bias),
        chromatic_prob=_clamp01(base.chromatic_prob),
        cadence_strength=_clamp01(base.cadence_strength),
        register_min_oct=reg_min,
        register_max_oct=reg_max,
        tension_curve=base.tension_curve
        if base.tension_curve in ("arc", "ramp", "waves")
        else "arc",
    )

def _grid_step_beats(density: float) -> float:
    """Density->grid step in beats."""
    if density < 0.25:
        return 1.0
    if density < 0.60:
        return 0.5
    return 0.25

def _nearest_chord_tone_pitch(
    pitch: int,
    chord_root_degree: int,
    min_pitch: int,
    max_pitch: int,
    chord_extensions: ChordExtensions,
) -> int:
    """Find the nearest chord-tone pitch (diatonic) to the given pitch."""
    chord_tones = _chord_tone_classes(chord_root_degree, chord_extensions)
    candidates: list[int] = []
    for tone in chord_tones:
        for oct_i in range((max_pitch // 7) + 1):
            cand = tone + 7 * oct_i
            if min_pitch <= cand <= max_pitch:
                candidates.append(cand)
    if not candidates:
        return int(np.clip(pitch, min_pitch, max_pitch))
    return int(min(candidates, key=lambda c: abs(c - pitch)))

def _choose_anchor_pitch(
    rng: np.random.Generator,
    prev_pitch: int,
    chord_root_degree: int,
    min_pitch: int,
    max_pitch: int,
    tension: float,
    cadence_strength: float,
    chord_extensions: ChordExtensions,
) -> int:
    """Choose an anchor pitch (chord tone) with simple voice-leading."""
    chord_root = chord_root_degree % 7
    chord_tones = _chord_tone_classes(chord_root_degree, chord_extensions)

    midpoint = 0.5 * (min_pitch + max_pitch)
    desired = midpoint + (tension - 0.5) * 0.35 * (max_pitch - min_pitch)

    items: list[int] = []
    weights: list[float] = []

    for tone in chord_tones:
        weight = 1.0
        if tone == chord_root:
            weight *= 1.0 + 1.8 * cadence_strength
        elif tone == (chord_root + 2) % 7:
            weight *= 1.0
        elif tone == (chord_root + 4) % 7:
            weight *= 0.95
        elif tone == (chord_root + 6) % 7:
            weight *= 0.85
        else:
            weight *= 0.75

        for oct_i in range((max_pitch // 7) + 1):
            cand = tone + 7 * oct_i
            if cand < min_pitch or cand > max_pitch:
                continue

            d_prev = abs(cand - prev_pitch)
            d_reg = abs(cand - desired)
            w = weight * float(np.exp(-d_prev / 2.6)) * float(np.exp(-d_reg / 4.0))
            items.append(int(cand))
            weights.append(float(w))

    if not items:
        return int(np.clip(prev_pitch, min_pitch, max_pitch))

    return _weighted_choice(rng, items, weights)

def _generate_procedural_melody_events(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> list[NoteEvent]:
    """Generate phrase-aware, chord-aware melody events."""
    policy = _resolve_melody_policy(melody_policy, config)

    beats_per_bar = 4
    spb = _seconds_per_beat(config)
    total_beats = max(1.0, _beats_total(config, duration))
    total_bars = int(np.ceil(total_beats / beats_per_bar))

    base_oct = policy.register_min_oct
    oct_span = max(0, policy.register_max_oct - policy.register_min_oct)
    min_pitch = 0
    max_pitch = oct_span * 7 + 6

    anchors: list[int] = []
    prev = int(np.clip((max_pitch - min_pitch) // 2, min_pitch, max_pitch))

    for bar in range(total_bars):
        beat0 = float(bar * beats_per_bar)
        chord_root = _chord_root_degree_at_beat(beat0, harmony)

        phrase_len = policy.phrase_len_bars
        pos_in_phrase = bar % phrase_len
        denom = max(1, phrase_len - 1)
        x = pos_in_phrase / denom
        tension = _tension_value(policy.tension_curve, x)

        is_cadence_bar = (pos_in_phrase == phrase_len - 1) and (phrase_len > 1)
        if is_cadence_bar:
            tension = 0.0

        anchor = _choose_anchor_pitch(
            rng=rng,
            prev_pitch=prev,
            chord_root_degree=chord_root,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            tension=tension,
            cadence_strength=policy.cadence_strength if is_cadence_bar else 0.10,
            chord_extensions=config.chord_extensions,
        )
        anchors.append(anchor)
        prev = anchor

    events: list[NoteEvent] = []

    step_beats = _grid_step_beats(policy.density)
    steps_per_bar = int(round(beats_per_bar / step_beats))
    steps_per_bar = max(1, min(64, steps_per_bar))

    motif: list[tuple[int, int]] = []

    for bar in range(total_bars):
        beat_bar = float(bar * beats_per_bar)
        chord_root = _chord_root_degree_at_beat(beat_bar, harmony)

        phrase_len = policy.phrase_len_bars
        pos_in_phrase = bar % phrase_len
        denom = max(1, phrase_len - 1)
        x = pos_in_phrase / denom
        tension = _tension_value(policy.tension_curve, x)

        is_phrase_start = pos_in_phrase == 0
        is_cadence_bar = (pos_in_phrase == phrase_len - 1) and (phrase_len > 1)

        anchor_pitch = anchors[bar]
        next_anchor = anchors[bar + 1] if bar + 1 < len(anchors) else anchor_pitch

        use_motif = bool(motif) and is_phrase_start and (rng.random() < policy.motif_repeat_prob)

        need_resolve = False
        current_pitch = anchor_pitch

        bar_notes_for_motif: list[tuple[int, int]] = []

        for step_idx in range(steps_per_bar):
            pos_beats = step_idx * step_beats
            beat = beat_bar + pos_beats
            start_sec = beat * spb

            if start_sec >= duration:
                break

            if step_beats == 0.5 and (step_idx % 2 == 1):
                start_sec += policy.swing * 0.12 * spb

            if step_idx == 0:
                play = True
                is_anchor = True
            elif use_motif and any(step == step_idx for step, _ in motif):
                play = True
                is_anchor = False
            else:
                is_offbeat = abs(pos_beats - round(pos_beats)) > 1e-6

                prob = 0.10 + 0.80 * policy.density
                prob += 0.20 * tension
                prob += (0.25 * policy.syncopation) if is_offbeat else (-0.15 * policy.syncopation)

                if is_cadence_bar and pos_beats >= 2.0:
                    prob *= 0.35 + 0.40 * (1.0 - policy.cadence_strength)

                if events and (events[-1].start_sec > (start_sec - 0.001)):
                    prob *= 0.6

                play = rng.random() < _clamp01(prob)
                is_anchor = False

            if not play:
                continue

            if step_idx == 0:
                pitch = anchor_pitch
            elif use_motif:
                offset = next((off for step, off in motif if step == step_idx), 0)
                pitch = int(np.clip(anchor_pitch + offset, min_pitch, max_pitch))
            else:
                delta = next_anchor - current_pitch
                if delta == 0:
                    direction = int(rng.choice([-1, 1]))
                else:
                    direction = 1 if delta > 0 else -1

                if need_resolve:
                    pitch = _nearest_chord_tone_pitch(
                        current_pitch,
                        chord_root,
                        min_pitch,
                        max_pitch,
                        config.chord_extensions,
                    )
                    need_resolve = False
                else:
                    leap_prob = (1.0 - policy.step_bias) * (0.25 + 0.35 * tension)
                    step_size = 1

                    if rng.random() < (0.06 + 0.10 * tension):
                        step_size = 0
                    elif abs(delta) >= 4 and (rng.random() < leap_prob):
                        step_size = int(rng.choice([2, 3, 4]))

                    pitch = int(
                        np.clip(current_pitch + direction * step_size, min_pitch, max_pitch)
                    )

                chord_tones = _chord_tone_classes(chord_root, config.chord_extensions)
                if (pitch % 7) not in chord_tones:
                    if rng.random() < (0.55 - 0.45 * tension):
                        pitch = _nearest_chord_tone_pitch(
                            pitch,
                            chord_root,
                            min_pitch,
                            max_pitch,
                            config.chord_extensions,
                        )
                    else:
                        need_resolve = True

            dur_beats = step_beats * (0.70 + 0.25 * float(rng.random()))
            if is_anchor:
                dur_beats = max(dur_beats, 1.0)
                if is_cadence_bar:
                    dur_beats = max(dur_beats, 1.6)

            dur_sec = dur_beats * spb
            dur_sec = min(dur_sec, max(0.02, duration - start_sec))

            jitter = 0.0
            if config.human > 0 and not is_anchor:
                jitter = float(rng.normal(0.0, config.human * 0.006 * spb))
                jitter = float(np.clip(jitter, -0.02 * spb, 0.02 * spb))
            start_sec_j = float(np.clip(start_sec + jitter, 0.0, max(0.0, duration - 0.01)))

            amp = 0.12 + 0.05 * tension
            if is_cadence_bar and pos_beats >= 2.0:
                amp *= 0.85
            if is_anchor:
                amp *= 1.10

            use_chromatic = (
                (policy.chromatic_prob > 0)
                and (not is_anchor)
                and (abs((next_anchor - pitch)) <= 2)
                and (rng.random() < (policy.chromatic_prob * (0.35 + 0.65 * tension)))
            )

            if use_chromatic:
                target = next_anchor
                sign = 1 if (target - pitch) >= 0 else -1
                semitone = _semitone_from_degree(config.mode, target) - sign
                freq = freq_from_note(config.root, semitone, base_oct)
            else:
                freq = _scale_freq(config, pitch, base_oct)

            events.append(
                NoteEvent(
                    start_sec=start_sec_j,
                    dur_sec=dur_sec,
                    freq=freq,
                    amp=float(amp),
                    is_anchor=is_anchor,
                )
            )

            if is_phrase_start and not use_motif and not motif and not is_anchor:
                bar_notes_for_motif.append((step_idx, pitch - anchor_pitch))

            current_pitch = pitch

        if not motif and is_phrase_start and bar_notes_for_motif:
            motif = bar_notes_for_motif[: min(5, len(bar_notes_for_motif))]

    return events

def melody_procedural(
    config: SynthConfig,
    duration: float,
    harmony: HarmonyPlan | None,
    melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Procedural, chord-aware melody with phrases + tension/resolution."""
    sr = SAMPLE_RATE
    dur = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    events = _generate_procedural_melody_events(config, duration, harmony, melody_policy, rng)

    signal = np.zeros(int(sr * dur))

    cutoff = 700 * config.brightness + 180
    for event in events:
        start_idx = int(event.start_sec * sr)
        if start_idx >= len(signal):
            continue

        note = osc(event.freq, event.dur_sec, sr, event.amp, t_offset=t_offset + event.start_sec)  # type: ignore[call-arg]
        note = apply_lowpass(note, cutoff, sr)

        atk = (0.05 if event.is_anchor else 0.03) * attack_mult
        rel = (0.22 if event.is_anchor else 0.16) * attack_mult
        note = apply_adsr(note, atk, 0.12, 0.55, rel, sr)
        note = apply_humanize(note, config.human * 0.6, sr)

        add_note(signal, note, start_idx)

    signal = apply_delay(signal, 0.33, 0.35 * echo_mult, 0.28 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.55, config.space * 0.75, sr)

    return signal

def melody_contemplative(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Slow, contemplative melody."""
    sr = SAMPLE_RATE
    dur = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    # Sparse melody pattern (tuple), -1 = rest
    pattern = (0, -1, 2, -1, 4, -1, 2, -1, 0, -1, -1, -1, 2, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = _scale_freq(config, degree, 5)
        start_sec = i * note_dur
        start = int(start_sec * sr)

        note = osc(freq, note_dur * 1.5, sr, 0.18, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
        note = apply_lowpass(note, 800 * config.brightness + 200, sr)
        note = apply_adsr(note, 0.1 * attack_mult, 0.3, 0.6, 0.8 * attack_mult, sr)
        note = apply_humanize(note, config.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.35, 0.4 * echo_mult, 0.3 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.6, config.space * 0.8, sr)

    return signal

def melody_rising(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Ascending melodic line."""
    sr = SAMPLE_RATE
    dur = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    # Rising pattern (tuple)
    pattern = (0, -1, 2, -1, 4, -1, -1, 5, -1, -1, 6, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = _scale_freq(config, degree, 5)
        start_sec = i * note_dur
        start = int(start_sec * sr)

        note = osc(freq, note_dur * 1.8, sr, 0.16, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
        note = apply_lowpass(note, 900 * config.brightness + 250, sr)
        note = apply_adsr(note, 0.08 * attack_mult, 0.25, 0.55, 0.9 * attack_mult, sr)
        note = apply_humanize(note, config.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.33, 0.35 * echo_mult, 0.28 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.55, config.space * 0.75, sr)

    return signal

def melody_falling(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Descending melodic line."""
    sr = SAMPLE_RATE
    dur = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    # Falling pattern (tuple)
    pattern = (6, -1, -1, 5, -1, -1, 4, -1, 2, -1, -1, 0, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = _scale_freq(config, degree, 5)
        start_sec = i * note_dur
        start = int(start_sec * sr)

        note = osc(freq, note_dur * 1.6, sr, 0.17, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
        note = apply_lowpass(note, 850 * config.brightness + 220, sr)
        note = apply_adsr(note, 0.1 * attack_mult, 0.28, 0.52, 0.85 * attack_mult, sr)
        note = apply_humanize(note, config.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.38, 0.38 * echo_mult, 0.3 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.58, config.space * 0.78, sr)

    return signal

def melody_minimal(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Very sparse, minimal melody."""
    sr = SAMPLE_RATE
    dur = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_sine)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    # Extremely sparse (tuple)
    pattern = (4, -1, -1, -1, -1, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        freq = _scale_freq(config, degree, 5)
        start_sec = i * note_dur
        start = int(start_sec * sr)

        note = osc(freq, note_dur * 2.5, sr, 0.20, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
        note = apply_lowpass(note, 600 * config.brightness + 150, sr)
        note = apply_adsr(note, 0.15 * attack_mult, 0.4, 0.5, 1.2 * attack_mult, sr)
        note = apply_humanize(note, config.human, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.5, 0.45 * echo_mult, 0.4 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.7, config.space * 0.9, sr)

    return signal

def melody_ornamental(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Ornamental melody with grace notes."""
    sr = SAMPLE_RATE
    dur = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    # Pattern with ornaments (tuples)
    main_notes = (0, 4, 2, 0)
    grace_offsets = (2, 5, 3, 2)
    note_dur = dur / (len(main_notes) * 2)
    signal = np.zeros(int(sr * dur))

    for i, (main, grace) in enumerate(zip(main_notes, grace_offsets)):
        start_sec = i * 2 * note_dur
        start = int(start_sec * sr)

        # Grace note (quick)
        grace_freq = _scale_freq(config, grace, 5)
        grace_note = osc(grace_freq, note_dur * 0.15, sr, 0.10, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
        grace_note = apply_lowpass(grace_note, 1000 * config.brightness, sr)
        grace_note = apply_adsr(grace_note, 0.01, 0.05, 0.3, 0.1, sr)

        # Main note
        main_freq = _scale_freq(config, main, 5)
        main_start_sec = start_sec + note_dur * 0.15
        main_note = osc(main_freq, note_dur * 1.5, sr, 0.18, t_offset=t_offset + main_start_sec)  # type: ignore[call-arg]
        main_note = apply_lowpass(main_note, 900 * config.brightness + 200, sr)
        main_note = apply_adsr(main_note, 0.08 * attack_mult, 0.3, 0.55, 0.8 * attack_mult, sr)
        main_note = apply_humanize(main_note, config.human, sr)

        grace_start = start
        main_start = start + int(note_dur * 0.15 * sr)

        add_note(signal, grace_note, grace_start)
        add_note(signal, main_note, main_start)

    signal = apply_delay(signal, 0.3, 0.35 * echo_mult, 0.25 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.5, config.space * 0.7, sr)

    return signal

def melody_arp(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Arpeggiated melody."""
    sr = SAMPLE_RATE
    dur = duration
    osc = OSC_FUNCTIONS.get(_osc_type(config), generate_triangle)
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    # Fast arpeggio pattern (tuple * 8)
    base_pattern = (0, 2, 4, 2)
    pattern = base_pattern * 8
    note_dur = dur / len(pattern)
    signal = np.zeros(int(sr * dur))

    for i, degree in enumerate(pattern):
        freq = _scale_freq(config, degree, 4)
        start_sec = i * note_dur
        start = int(start_sec * sr)

        note = osc(freq, note_dur * 0.8, sr, 0.14, t_offset=t_offset + start_sec)  # type: ignore[call-arg]
        note = apply_lowpass(note, 1200 * config.brightness + 300, sr)
        note = apply_adsr(note, 0.02 * attack_mult, 0.1, 0.4, 0.15 * attack_mult, sr)

        add_note(signal, note, start)

    signal = apply_delay(signal, 0.25, 0.3 * echo_mult, 0.25 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.4, config.space * 0.6, sr)

    return signal

MELODY_PATTERNS: Mapping[str, PatternFn] = MappingProxyType(
    {
        "procedural": melody_procedural,
        "contemplative": melody_contemplative,
        "contemplative_minor": melody_contemplative,
        "rising": melody_rising,
        "falling": melody_falling,
        "minimal": melody_minimal,
        "ornamental": melody_ornamental,
        "arp_melody": melody_arp,
        "call_response": melody_contemplative,
        "heroic": melody_rising,
    }
)

# -----------------------------------------------------------------------------
# RHYTHM PATTERNS
# -----------------------------------------------------------------------------

def rhythm_none(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """No rhythm."""
    return np.zeros(int(SAMPLE_RATE * duration))

def rhythm_minimal(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Minimal rhythm - just occasional hits."""
    sr = SAMPLE_RATE
    dur = duration
    signal = np.zeros(int(sr * dur))

    # Sparse kicks: hit on beat 0 of each bar (8 slots per bar at 2 slots/beat in 4/4)
    pattern = (1, 0, 0, 0, 0, 0, 0, 0)

    if TEMPO_AWARE_RHYTHM:
        slot_times, slot_offset = _rhythm_slot_times(
            config, dur, slots_per_beat=2, t_offset=t_offset
        )
        for i, t in enumerate(slot_times):
            if not pattern[(slot_offset + i) % len(pattern)]:
                continue
            start = int(t * sr)
            kick_dur = 0.15
            kick = generate_sine(60, kick_dur, sr, 0.35, t_offset + t)
            pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 150, sr)
            kick = apply_humanize(kick, config.human, sr)
            add_note(signal, kick, start)
    else:
        # Legacy: divide chunk into fixed slots
        pattern_full = pattern * 2
        hit_dur = dur / len(pattern_full)
        for i, hit in enumerate(pattern_full):
            if not hit:
                continue
            start = int(i * hit_dur * sr)
            kick_dur = 0.15
            kick = generate_sine(60, kick_dur, sr, 0.35, t_offset)
            pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 150, sr)
            kick = apply_humanize(kick, config.human, sr)
            add_note(signal, kick, start)

    signal = apply_reverb(signal, config.space * 0.3, config.space * 0.5, sr)
    return signal

def rhythm_heartbeat(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Heartbeat-like rhythm."""
    sr = SAMPLE_RATE
    dur = duration
    signal = np.zeros(int(sr * dur))

    # Double-hit pattern: two quick hits then rest (heartbeat feel)
    pattern = (1, 1, 0, 0, 0, 0, 0, 0)

    if TEMPO_AWARE_RHYTHM:
        slot_times, slot_offset = _rhythm_slot_times(
            config, dur, slots_per_beat=2, t_offset=t_offset
        )
        for i, t in enumerate(slot_times):
            if not pattern[(slot_offset + i) % len(pattern)]:
                continue
            start = int(t * sr)
            kick_dur = 0.12
            kick = generate_sine(55, kick_dur, sr, 0.32, t_offset + t)
            pitch_env = np.exp(-np.linspace(0, 10, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 120, sr)
            add_note(signal, kick, start)
    else:
        # Legacy: divide chunk into fixed slots
        pattern_full = pattern * 2
        hit_dur = dur / len(pattern_full)
        for i, hit in enumerate(pattern_full):
            if not hit:
                continue
            start = int(i * hit_dur * sr)
            kick_dur = 0.12
            kick = generate_sine(55, kick_dur, sr, 0.32, t_offset)
            pitch_env = np.exp(-np.linspace(0, 10, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 120, sr)
            add_note(signal, kick, start)

    return signal

def rhythm_soft_four(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Soft four-on-the-floor."""
    sr = SAMPLE_RATE
    dur = duration
    signal = np.zeros(int(sr * dur))

    # Kick on every beat (4-on-the-floor)
    if TEMPO_AWARE_RHYTHM:
        # slots_per_beat=1 means one slot per beat = quarter notes
        slot_times, _slot_offset = _rhythm_slot_times(
            config, dur, slots_per_beat=1, t_offset=t_offset
        )
        for t in slot_times:
            start = int(t * sr)
            kick_dur = 0.18
            kick = generate_sine(50, kick_dur, sr, 0.28, t_offset + t)
            pitch_env = np.exp(-np.linspace(0, 6, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 100, sr)
            kick = apply_humanize(kick, config.human, sr)
            add_note(signal, kick, start)
    else:
        # Legacy: divide chunk into 8 fixed beats
        num_beats = 8
        beat_dur = dur / num_beats
        for i in range(num_beats):
            start = int(i * beat_dur * sr)
            kick_dur = 0.18
            kick = generate_sine(50, kick_dur, sr, 0.28, t_offset)
            pitch_env = np.exp(-np.linspace(0, 6, len(kick)))
            kick = kick * pitch_env
            kick = apply_lowpass(kick, 100, sr)
            kick = apply_humanize(kick, config.human, sr)
            add_note(signal, kick, start)

    signal = apply_reverb(signal, config.space * 0.25, config.space * 0.4, sr)
    return signal

def rhythm_hats_only(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Just hi-hats."""
    sr = SAMPLE_RATE
    dur = duration
    signal = np.zeros(int(sr * dur))

    # 16th note hats (4 subdivisions per beat)
    if TEMPO_AWARE_RHYTHM:
        slot_times, _slot_offset = _rhythm_slot_times(
            config, dur, slots_per_beat=4, t_offset=t_offset
        )
        for t in slot_times:
            start = int(t * sr)
            hat_dur = 0.05
            hat = generate_noise(hat_dur, sr, 0.08)
            hat = apply_highpass(hat, 6000, sr)
            hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.03, sr)
            hat = apply_humanize(hat, config.human, sr)
            add_note(signal, hat, start)
    else:
        # Legacy: divide chunk into 32 fixed slots
        num_hits = 32
        hit_dur = dur / num_hits
        for i in range(num_hits):
            start = int(i * hit_dur * sr)
            hat_dur = 0.05
            hat = generate_noise(hat_dur, sr, 0.08)
            hat = apply_highpass(hat, 6000, sr)
            hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.03, sr)
            hat = apply_humanize(hat, config.human, sr)
            add_note(signal, hat, start)

    return signal

def rhythm_electronic(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Electronic beat."""
    sr = SAMPLE_RATE
    dur = duration
    signal = np.zeros(int(sr * dur))

    # 16-slot pattern over 4 beats (16th note grid)
    # Kick on beats 1,2,3,4 (slots 0,4,8,12), hats on off-beats (slots 2,6,10,14)
    kick_pattern = (1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0)
    hat_pattern = (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0)

    if TEMPO_AWARE_RHYTHM:
        slot_times, slot_offset = _rhythm_slot_times(
            config, dur, slots_per_beat=4, t_offset=t_offset
        )
        for i, t in enumerate(slot_times):
            idx = (slot_offset + i) % len(kick_pattern)
            start = int(t * sr)

            if kick_pattern[idx]:
                kick = generate_sine(45, 0.2, sr, 0.35, t_offset + t)
                pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
                kick = kick * pitch_env
                kick = apply_lowpass(kick, 100, sr)
                add_note(signal, kick, start)

            if hat_pattern[idx]:
                hat = generate_noise(0.06, sr, 0.06)
                hat = apply_highpass(hat, 7000, sr)
                hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.04, sr)
                add_note(signal, hat, start)
    else:
        # Legacy: divide chunk into 16 fixed slots
        beat_dur = dur / 16
        for i in range(16):
            start = int(i * beat_dur * sr)

            if kick_pattern[i]:
                kick = generate_sine(45, 0.2, sr, 0.35, t_offset)
                pitch_env = np.exp(-np.linspace(0, 8, len(kick)))
                kick = kick * pitch_env
                kick = apply_lowpass(kick, 100, sr)
                add_note(signal, kick, start)

            if hat_pattern[i]:
                hat = generate_noise(0.06, sr, 0.06)
                hat = apply_highpass(hat, 7000, sr)
                hat = apply_adsr(hat, 0.001, 0.02, 0.1, 0.04, sr)
                add_note(signal, hat, start)

    return signal

RHYTHM_PATTERNS: Mapping[RhythmStyle, PatternFn] = MappingProxyType(
    {
        "none": rhythm_none,
        "minimal": rhythm_minimal,
        "heartbeat": rhythm_heartbeat,
        "soft_four": rhythm_soft_four,
        "hats_only": rhythm_hats_only,
        "electronic": rhythm_electronic,
        "kit_light": rhythm_minimal,
        "kit_medium": rhythm_soft_four,
        "military": rhythm_soft_four,
        "tabla_essence": rhythm_heartbeat,
        "brush": rhythm_minimal,
    }
)

# -----------------------------------------------------------------------------
# TEXTURE PATTERNS
# -----------------------------------------------------------------------------

def texture_none(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """No texture."""
    del t_offset  # unused
    return np.zeros(int(SAMPLE_RATE * duration))

def texture_shimmer(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """High, shimmering texture."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    signal = np.zeros(int(sr * dur))

    # High sine clusters with amplitude modulation (tuple)
    for degree in (0, 2, 4):
        freq = _scale_freq(config, degree, 6)
        tone = generate_sine(freq, dur, sr, 0.04, t_offset=t_offset)

        # Amplitude modulation for shimmer
        lfo_rate = 2.0 / (config.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = tone * (0.5 + 0.5 * lfo)

        tone = apply_adsr(tone, 0.5 * attack_mult, 0.3, 0.8, 1.5 * attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.4, 0.5 * echo_mult, 0.4 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.8, config.space, sr)

    return signal

def texture_shimmer_slow(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Slow, gentle shimmer."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)
    echo_mult = _echo_mult(config)

    signal = np.zeros(int(sr * dur))

    for degree in (0, 4):
        freq = _scale_freq(config, degree, 6)
        tone = generate_sine(freq, dur, sr, 0.035, t_offset=t_offset)

        # Very slow amplitude modulation
        lfo_rate = 0.5 / (config.motion + 0.2)
        lfo = generate_lfo(dur, lfo_rate, sr)
        tone = tone * (0.4 + 0.6 * lfo)

        tone = apply_adsr(tone, 1.0 * attack_mult, 0.5, 0.85, 2.0 * attack_mult, sr)
        signal += tone

    signal = apply_delay(signal, 0.5, 0.55 * echo_mult, 0.45 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.85, config.space, sr)

    return signal

def texture_vinyl_crackle(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Vinyl crackle texture."""
    del t_offset  # unused - noise-based, no oscillators
    sr = SAMPLE_RATE
    dur = duration

    # Sparse noise impulses
    signal = np.zeros(int(sr * dur))

    num_crackles = int(dur * 20)  # ~20 crackles per second

    for _ in range(num_crackles):
        pos = int(rng.integers(0, max(1, len(signal) - 100)))
        crackle = generate_noise(0.002, sr, float(rng.uniform(0.01, 0.04)))
        crackle = apply_highpass(crackle, 2000, sr)

        add_note(signal, crackle, pos)

    # Soft background hiss
    hiss = generate_noise(dur, sr, 0.008)
    hiss = apply_lowpass(hiss, 8000, sr)
    hiss = apply_highpass(hiss, 1000, sr)
    signal += hiss

    return signal

def texture_breath(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Breathing texture."""
    del t_offset  # unused - noise-based, no oscillators
    sr = SAMPLE_RATE
    dur = duration

    # Filtered noise with slow envelope
    signal = generate_noise(dur, sr, 0.06)

    # Bandpass around a note frequency
    freq = _scale_freq(config, 0, 3)
    signal = apply_lowpass(signal, freq * 2, sr)
    signal = apply_highpass(signal, freq * 0.5, sr)

    # Breathing envelope (slow LFO)
    breath_rate = 0.2 / (config.motion + 0.1)
    lfo = generate_lfo(dur, breath_rate, sr)
    signal = signal * lfo

    signal = apply_reverb(signal, config.space * 0.6, config.space * 0.8, sr)

    return signal

def texture_stars(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Twinkling stars texture."""
    sr = SAMPLE_RATE
    dur = duration
    echo_mult = _echo_mult(config)

    signal = np.zeros(int(sr * dur))

    # Random high plinks
    num_stars = int(dur * 3)  # ~3 per second

    # Scale degrees for stars (tuple)
    star_degrees = (0, 2, 4, 5)

    for _ in range(num_stars):
        pos = int(rng.integers(0, max(1, len(signal) - sr)))

        degree = int(rng.choice(star_degrees))
        freq = _scale_freq(config, degree, 6)

        start_sec = pos / sr
        star = generate_sine(
            freq, 0.3, sr, float(rng.uniform(0.02, 0.05)), t_offset=t_offset + start_sec
        )
        star = apply_adsr(star, 0.01, 0.1, 0.1, 0.2, sr)

        add_note(signal, star, pos)

    signal = apply_delay(signal, 0.4, 0.5 * echo_mult, 0.4 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.9, config.space, sr)

    return signal

TEXTURE_PATTERNS: Mapping[TextureStyle, PatternFn] = MappingProxyType(
    {
        "none": texture_none,
        "shimmer": texture_shimmer,
        "shimmer_slow": texture_shimmer_slow,
        "vinyl_crackle": texture_vinyl_crackle,
        "breath": texture_breath,
        "stars": texture_stars,
        "glitch": texture_shimmer,
        "noise_wash": texture_breath,
        "crystal": texture_stars,
        "pad_whisper": texture_breath,
    }
)

# -----------------------------------------------------------------------------
# ACCENT PATTERNS
# -----------------------------------------------------------------------------

def accent_none(
    _config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """No accent."""
    del t_offset  # unused
    return np.zeros(int(SAMPLE_RATE * duration))

def accent_bells(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Bell-like accents."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)

    signal = np.zeros(int(sr * dur))

    # Sparse bell hits (tuple)
    pattern = (0, -1, -1, -1, 4, -1, -1, -1, 2, -1, -1, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        start_sec = start / sr
        freq = _scale_freq(config, degree, 5)

        # Bell: mix of harmonics with fast decay
        bell_dur = 0.8
        bell = generate_sine(freq, bell_dur, sr, 0.12, t_offset=t_offset + start_sec)
        bell += generate_sine(freq * 2.0, bell_dur, sr, 0.06, t_offset=t_offset + start_sec)
        bell += generate_sine(freq * 3.0, bell_dur, sr, 0.03, t_offset=t_offset + start_sec)
        bell = apply_adsr(bell, 0.005 * attack_mult, 0.2, 0.1, 0.6 * attack_mult, sr)
        bell = apply_humanize(bell, config.human, sr)

        add_note(signal, bell, start)

    signal = apply_reverb(signal, config.space * 0.6, config.space * 0.8, sr)
    return signal

def accent_pluck(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    _rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Plucked string accents."""
    sr = SAMPLE_RATE
    dur = duration
    attack_mult = _attack_mult(config)

    signal = np.zeros(int(sr * dur))

    # Pattern (tuple)
    pattern = (0, -1, -1, 4, -1, -1, 2, -1, -1, -1, 0, -1, -1, -1, -1, -1)
    hit_dur = dur / len(pattern)

    for i, degree in enumerate(pattern):
        if degree < 0:
            continue

        start = int(i * hit_dur * sr)
        start_sec = start / sr
        freq = _scale_freq(config, degree, 4)

        # Pluck: sharp attack, quick decay
        pluck = generate_triangle(freq, 0.5, sr, 0.15, t_offset=t_offset + start_sec)
        pluck = apply_lowpass(pluck, 1500 * config.brightness + 400, sr)
        pluck = apply_adsr(pluck, 0.003 * attack_mult, 0.15, 0.05, 0.4 * attack_mult, sr)
        pluck = apply_humanize(pluck, config.human, sr)

        add_note(signal, pluck, start)

    signal = apply_reverb(signal, config.space * 0.5, config.space * 0.7, sr)
    return signal

def accent_chime(
    config: SynthConfig,
    duration: float,
    _harmony: HarmonyPlan | None,
    _melody_policy: MelodyPolicy | None,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> FloatArray:
    """Wind chime accents."""
    sr = SAMPLE_RATE
    dur = duration
    echo_mult = _echo_mult(config)

    signal = np.zeros(int(sr * dur))

    # Random chime hits
    num_chimes = int(dur * 1.5)

    # Chime degrees (tuple)
    chime_degrees = (0, 2, 4, 5, 6)

    for _ in range(num_chimes):
        max_start = len(signal) - sr
        if max_start <= 0:
            pos = 0
        else:
            pos = int(rng.integers(0, max_start))

        degree = int(rng.choice(chime_degrees))
        freq = _scale_freq(config, degree, 5)

        chime_dur = 1.2
        start_sec = pos / sr
        chime = generate_sine(
            freq, chime_dur, sr, float(rng.uniform(0.06, 0.12)), t_offset=t_offset + start_sec
        )
        chime += generate_sine(freq * 2.0, chime_dur, sr, 0.03, t_offset=t_offset + start_sec)
        chime = apply_adsr(chime, 0.002, 0.3, 0.05, 0.9, sr)

        add_note(signal, chime, pos)

    signal = apply_delay(signal, 0.3, 0.4 * echo_mult, 0.3 * echo_mult, sr)
    signal = apply_reverb(signal, config.space * 0.75, config.space * 0.9, sr)

    return signal

ACCENT_PATTERNS: Mapping[AccentStyle, PatternFn] = MappingProxyType(
    {
        "none": accent_none,
        "bells": accent_bells,
        "bells_dense": accent_bells,
        "pluck": accent_pluck,
        "chime": accent_chime,
        "blip": accent_bells,
        "blip_random": accent_chime,
        "brass_hit": accent_bells,
        "wind": accent_chime,
        "arp_accent": accent_pluck,
        "piano_note": accent_pluck,
    }
)

def build_melody_policy(config: SynthConfig) -> MelodyPolicy:
    """Create a MelodyPolicy from config fields."""
    return MelodyPolicy(
        phrase_len_bars=int(max(1, min(16, config.phrase_len_bars))),
        density=float(np.clip(config.melody_density, 0.0, 1.0)),
        syncopation=float(np.clip(config.syncopation, 0.0, 1.0)),
        swing=float(np.clip(config.swing, 0.0, 1.0)),
        motif_repeat_prob=float(np.clip(config.motif_repeat_prob, 0.0, 1.0)),
        step_bias=float(np.clip(config.step_bias, 0.0, 1.0)),
        chromatic_prob=float(np.clip(config.chromatic_prob, 0.0, 1.0)),
        cadence_strength=float(np.clip(config.cadence_strength, 0.0, 1.0)),
        register_min_oct=int(config.register_min_oct),
        register_max_oct=int(config.register_max_oct),
        tension_curve=config.tension_curve,
    )

def build_harmony_plan(
    config: SynthConfig,
    duration: float,
    rng: np.random.Generator,
    t_offset: float = 0.0,
) -> HarmonyPlan:
    """
    Generate a simple diatonic chord timeline in scale-degrees.

    This is intentionally compute-light: a handful of templates + repetition.
    """
    beats_per_bar = 4
    beats_total = _beats_total(config, duration)
    bars_total = int(np.ceil(beats_total / beats_per_bar))
    bars_total = max(1, bars_total)

    chord_change_bars = int(max(1, config.chord_change_bars))
    n_chords = int(np.ceil(bars_total / chord_change_bars))
    n_chords = max(1, n_chords)

    style = (config.harmony_style or "auto").lower()
    if style == "auto":
        if config.rhythm == "none" and config.space >= 0.7:
            style = "ambient"
        elif config.pad in ("cinematic",) or config.bass in ("drone", "sub_pulse"):
            style = "cinematic"
        elif config.texture in ("vinyl_crackle",) or config.melody in ("ornamental",):
            style = "jazz"
        else:
            style = "pop"

    mode = (config.mode or "minor").lower()
    majorish = mode in ("major", "mixolydian")

    pop_major = (
        (0, 5, 3, 4),
        (0, 3, 4, 0),
        (0, 4, 5, 3),
    )
    pop_minor = (
        (0, 5, 3, 6),
        (0, 3, 6, 4),
        (0, 6, 3, 5),
        (0, 4, 5, 3),
    )
    jazz_major = (
        (1, 4, 0, 0),
        (1, 4, 0, 5),
        (5, 1, 4, 0),
    )
    jazz_minor = (
        (1, 4, 0, 0),
        (5, 1, 4, 0),
    )
    cinematic_minor = (
        (0, 6, 5, 3),
        (0, 5, 6, 4),
        (0, 6, 3, 4),
    )
    ambient_any = (
        (0, 5, 0, 3),
        (0, 4, 0, 6),
        (0, 3, 0, 5),
    )

    if style == "jazz":
        template = rng.choice(jazz_major if majorish else jazz_minor)
    elif style == "cinematic":
        template = rng.choice(cinematic_minor if not majorish else pop_major)
    elif style == "ambient":
        template = rng.choice(ambient_any)
    else:
        template = rng.choice(pop_major if majorish else pop_minor)

    roots: list[int] = []
    while len(roots) < n_chords:
        roots.extend(int(value) for value in template)
    roots = roots[:n_chords]

    chord_beats = float(chord_change_bars * beats_per_bar)

    chords: list[ChordEvent] = []
    for i, degree in enumerate(roots):
        chords.append(
            ChordEvent(
                start_beat=i * chord_beats, duration_beats=chord_beats, root_degree=int(degree) % 7
            )
        )

    return HarmonyPlan(chords=tuple(chords), beats_per_bar=beats_per_bar)

# =============================================================================
# PART 3: ASSEMBLER - CONFIG → AUDIO
# =============================================================================

def _suppress_clicks(
    signal: FloatArray,
    sr: int = SAMPLE_RATE,
    threshold: float = 0.10,
    fade_ms: float = 8.0,
) -> FloatArray:
    """
    Detecta saltos bruscos de amplitud (|diff| > threshold) —
    típicamente discontinuidades de fase en los bordes de segmento de acorde —
    y los suaviza con una ventana de Hann de ±fade_ms ms centrada en el salto.
    En cada región usa una mezcla ponderada entre la señal original y una
    versión suavizada (media móvil local), con peso máximo en el click exacto.
    """
    out  = signal.copy().astype(np.float64)
    diff = np.abs(np.diff(out))
    clicks = np.where(diff > threshold)[0]

    if len(clicks) == 0:
        return out.astype(signal.dtype)

    fade_n = max(8, int(fade_ms * 0.001 * sr))

    for c in clicks:
        i0 = max(0, c - fade_n)
        i1 = min(len(out), c + fade_n + 1)
        n  = i1 - i0
        if n < 4:
            continue

        seg         = out[i0:i1]
        kernel_size = min(n // 2, max(4, fade_n // 4))
        kernel      = np.ones(kernel_size) / kernel_size
        smoothed    = np.convolve(seg, kernel, mode="same")
        hann        = np.hanning(n)

        out[i0:i1] = seg * (1.0 - hann) + smoothed * hann

    return out.astype(signal.dtype)


def assemble(
    config: SynthConfig,
    duration: float = 16.0,
    normalize: bool = True,
    rng: np.random.Generator | None = None,
    t_offset: float = 0.0,
) -> FloatArray:
    """
    Assemble all layers into final audio.

    This is the core function that converts a config into audio.

    Args:
        config: SynthConfig object
        duration: Duration in seconds
        normalize: If True, maximizes volume. Set False for automation chunks
                   to preserve relative dynamics.
        rng: Optional RNG for deterministic generation.
        t_offset: Time offset in seconds for phase continuity across chunks.
    """
    sr = SAMPLE_RATE
    local_rng = rng or np.random.default_rng()
    harmony = build_harmony_plan(config, duration, local_rng)
    melody_policy = build_melody_policy(config)

    # Determine active layers based on density
    active_layers = DENSITY_LAYERS[config.density]

    # Initialize output
    output = np.zeros(int(sr * duration))

    # Generate each layer with t_offset for phase continuity
    if "bass" in active_layers:
        bass_fn = BASS_PATTERNS.get(config.bass, bass_drone)
        output += bass_fn(config, duration, harmony, melody_policy, local_rng, t_offset)

    if config.depth:
        # Add sub-bass layer
        output += (
            bass_sub_pulse(config, duration, harmony, melody_policy, local_rng, t_offset) * 0.6
        )

    if "pad" in active_layers:
        pad_fn = PAD_PATTERNS.get(config.pad, pad_warm_slow)
        output += pad_fn(config, duration, harmony, melody_policy, local_rng, t_offset)

    if "melody" in active_layers:
        if config.melody_engine == "procedural":
            output += melody_procedural(
                config, duration, harmony, melody_policy, local_rng, t_offset
            )
        else:
            melody_fn = MELODY_PATTERNS.get(config.melody, melody_contemplative)
            output += melody_fn(config, duration, harmony, melody_policy, local_rng, t_offset)

    if "rhythm" in active_layers and config.rhythm != "none":
        rhythm_fn = RHYTHM_PATTERNS.get(config.rhythm, rhythm_none)
        output += rhythm_fn(config, duration, harmony, melody_policy, local_rng, t_offset)

    if "texture" in active_layers and config.texture != "none":
        texture_fn = TEXTURE_PATTERNS.get(config.texture, texture_none)
        output += texture_fn(config, duration, harmony, melody_policy, local_rng, t_offset)

    if "accent" in active_layers and config.accent != "none":
        accent_fn = ACCENT_PATTERNS.get(config.accent, accent_none)
        output += accent_fn(config, duration, harmony, melody_policy, local_rng, t_offset)

    output = apply_stereo_width(output, config.stereo, sr)

    # Normalize only if requested
    if normalize:
        max_val = np.max(np.abs(output))
        if max_val > 0:
            output = output / max_val * 0.85

    # ── Supresión de chasquidos ────────────────────────────────────────────
    # Se aplica DESPUÉS de normalizar: la normalización puede amplificar
    # saltos residuales hasta hacerlos audibles. threshold=0.10 elimina
    # los clicks periódicos de los bordes de segmento de acorde.
    output = _suppress_clicks(output, sr, threshold=0.04, fade_ms=8.0)

    return output

def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: t=0 returns a, t=1 returns b"""
    return a + (b - a) * t

def _clamp_density(value: float) -> DensityLevel:
    raw = int(round(value))
    match raw:
        case 2:
            return 2
        case 3:
            return 3
        case 4:
            return 4
        case 5:
            return 5
        case 6:
            return 6
        case _:
            return 2 if raw < 2 else 6

def interpolate_configs(config_a: SynthConfig, config_b: SynthConfig, t: float) -> SynthConfig:
    """
    Interpolate between two configs.
    t=0.0 → config_a
    t=1.0 → config_b
    """
    return SynthConfig(
        tempo=lerp(config_a.tempo, config_b.tempo, t),
        root=config_a.root if t < 0.5 else config_b.root,
        mode=config_a.mode if t < 0.5 else config_b.mode,
        brightness=lerp(config_a.brightness, config_b.brightness, t),
        space=lerp(config_a.space, config_b.space, t),
        density=_clamp_density(lerp(config_a.density, config_b.density, t)),
        # Layer selections: staggered switching
        bass=config_a.bass if t < 0.4 else config_b.bass,
        pad=config_a.pad if t < 0.5 else config_b.pad,
        melody=config_a.melody if t < 0.6 else config_b.melody,
        rhythm=config_a.rhythm if t < 0.5 else config_b.rhythm,
        texture=config_a.texture if t < 0.7 else config_b.texture,
        accent=config_a.accent if t < 0.8 else config_b.accent,
        # V2 parameters
        motion=lerp(config_a.motion, config_b.motion, t),
        attack=config_a.attack if t < 0.5 else config_b.attack,
        stereo=lerp(config_a.stereo, config_b.stereo, t),
        depth=config_a.depth if t < 0.5 else config_b.depth,
        echo=lerp(config_a.echo, config_b.echo, t),
        human=lerp(config_a.human, config_b.human, t),
        grain=config_a.grain if t < 0.5 else config_b.grain,
        melody_engine=config_a.melody_engine if t < 0.5 else config_b.melody_engine,
        phrase_len_bars=round(lerp(config_a.phrase_len_bars, config_b.phrase_len_bars, t)),
        melody_density=lerp(config_a.melody_density, config_b.melody_density, t),
        syncopation=lerp(config_a.syncopation, config_b.syncopation, t),
        swing=lerp(config_a.swing, config_b.swing, t),
        motif_repeat_prob=lerp(config_a.motif_repeat_prob, config_b.motif_repeat_prob, t),
        step_bias=lerp(config_a.step_bias, config_b.step_bias, t),
        chromatic_prob=lerp(config_a.chromatic_prob, config_b.chromatic_prob, t),
        cadence_strength=lerp(config_a.cadence_strength, config_b.cadence_strength, t),
        register_min_oct=round(lerp(config_a.register_min_oct, config_b.register_min_oct, t)),
        register_max_oct=round(lerp(config_a.register_max_oct, config_b.register_max_oct, t)),
        tension_curve=config_a.tension_curve if t < 0.5 else config_b.tension_curve,
        harmony_style=config_a.harmony_style if t < 0.5 else config_b.harmony_style,
        chord_change_bars=round(lerp(config_a.chord_change_bars, config_b.chord_change_bars, t)),
        chord_extensions=config_a.chord_extensions if t < 0.5 else config_b.chord_extensions,
    )

def morph_audio(
    config_a: SynthConfig, config_b: SynthConfig, duration: float = 60.0, segments: int = 8
) -> FloatArray:
    """
    Generate audio that morphs from config_a to config_b over duration.
    """
    segment_duration = duration / segments
    output = []

    for i in range(segments):
        t = i / (segments - 1) if segments > 1 else 0.0
        interpolated = interpolate_configs(config_a, config_b, t)
        # Don't normalize individual segments — preserve relative dynamics
        segment = assemble(interpolated, segment_duration, normalize=False)
        output.append(segment)

    result = np.concatenate(output)

    # Normalize the final result
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.85

    return result

def crossfade(audio_a: FloatArray, audio_b: FloatArray, crossfade_samples: int) -> FloatArray:
    """Crossfade between two audio arrays at the midpoint."""
    min_len = min(len(audio_a), len(audio_b))
    mid = min_len // 2
    half_cf = crossfade_samples // 2

    fade_out = np.linspace(1.0, 0.0, crossfade_samples)
    fade_in = np.linspace(0.0, 1.0, crossfade_samples)

    result = np.concatenate(
        (
            audio_a[: mid - half_cf],
            audio_a[mid - half_cf : mid + half_cf] * fade_out
            + audio_b[mid - half_cf : mid + half_cf] * fade_in,
            audio_b[mid + half_cf : min_len],
        )
    )

    return result

def transition(config_a: SynthConfig, config_b: SynthConfig, duration: float = 60.0) -> FloatArray:
    """Generate with crossfade transition."""
    # Don't normalize individual tracks — normalize after crossfade
    audio_a = assemble(config_a, duration, normalize=False)
    audio_b = assemble(config_b, duration, normalize=False)

    crossfade_duration = 4.0  # seconds
    crossfade_samples = int(crossfade_duration * SAMPLE_RATE)

    result = crossfade(audio_a, audio_b, crossfade_samples)

    # Normalize the final result
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.85

    return result

def generate_tween_with_automation(
    config_a: SynthConfig,
    config_b: SynthConfig,
    duration: float = 120.0,
    chunk_seconds: float = 2.0,
    overlap_seconds: float = 0.05,
) -> FloatArray:
    """
    Generate audio with automated parameters using Cached Block Processing.

    Performance: ~8x faster than per-chunk generation.
    Trade-off: Automation updates occur every 16s (Pattern Length) instead of every 2s.
    """
    sr = SAMPLE_RATE
    num_chunks = int(np.ceil(duration / chunk_seconds))  # ceil ensures we cover full duration
    overlap_samples = int(overlap_seconds * sr)

    PATTERN_LEN = 16.0
    chunks_per_pattern = int(PATTERN_LEN / chunk_seconds)

    chunk_len_sec = chunk_seconds + overlap_seconds
    chunk_samples = int(chunk_len_sec * sr)

    # Initialize output buffer
    output = np.zeros(int(duration * sr))

    cached_pattern: FloatArray | None = None
    cached_pattern_idx: int = -1

    for i in range(num_chunks):
        # 1. Determine which 16s Block we are in
        pattern_idx = i // chunks_per_pattern

        # 2. Check Cache
        if pattern_idx != cached_pattern_idx:
            # Interpolate parameters for this specific 16s block
            # Note: This "steps" the parameters every 16s.
            t = (pattern_idx * chunks_per_pattern) / max(1, num_chunks - 1)
            t = np.clip(t, 0.0, 1.0)
            t_eased = 0.5 - 0.5 * np.cos(t * np.pi)

            config = interpolate_configs(config_a, config_b, t_eased)

            # Generate the FULL 16s block
            # normalize=False preserves relative dynamics between blocks
            cached_pattern = assemble(config, PATTERN_LEN, normalize=False)
            cached_pattern_idx = pattern_idx

        # Ensure cache exists (typing safety)
        if cached_pattern is None:
            continue

        # 3. Slice the 2s window + overlap
        local_chunk_idx = i % chunks_per_pattern

        # Calculate indices relative to the cached pattern
        start_idx = int(local_chunk_idx * chunk_seconds * sr)
        end_idx = start_idx + chunk_samples

        # Handle wrapping (Circular Buffer logic)
        pattern_len_samples = len(cached_pattern)

        if end_idx <= pattern_len_samples:
            chunk = cached_pattern[start_idx:end_idx].copy()
        else:
            # Wrap around to start
            # part_a: from start_idx to end of buffer
            # part_b: from 0 to remainder
            part_a = cached_pattern[start_idx:]
            remainder = end_idx - pattern_len_samples
            part_b = cached_pattern[:remainder]
            chunk = np.concatenate((part_a, part_b))

        # 4. Apply Crossfade Envelopes (Overlap-Add)
        if i > 0:
            chunk[:overlap_samples] *= np.linspace(0.0, 1.0, overlap_samples)

        if i < num_chunks - 1:
            chunk[-overlap_samples:] *= np.linspace(1.0, 0.0, overlap_samples)

        # 5. Add to Main Output
        out_start = int(i * chunk_seconds * sr)
        out_end = min(out_start + len(chunk), len(output))
        available = out_end - out_start

        if available > 0:
            output[out_start:out_end] += chunk[:available]

    # Global Normalization
    max_val = np.max(np.abs(output))
    if max_val > 0:
        output = output / max_val * 0.85

    return output

# =============================================================================
# MAIN - Demo
# =============================================================================

# if __name__ == "__main__":
#     print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
#     print("=" * 50)

#     # Example 3: Bubblegum, sad, dead (from llm_to_synth.py, see @file_context_1)
#     import json

#     # These examples mirror the demo config blocks captured in @file_context_0:
#     demo_configs = [
#         {
#             # "thinking": "The 'Bubblegum'-style sound is characterized by a bright, slightly distorted, and playful melody with a prominent bass line and a smooth, evolving pad.",
#             "tempo": 0.5,
#             "root": "a",
#             "mode": "major",
#             "brightness": 0.75,
#             "space": 0.5,
#             "density": 2,
#             "bass": "drone",
#             "pad": "warm_slow",
#             "melody": "contemplative",
#             "rhythm": "none",
#             "texture": "shimmer",
#             "accent": "bells",
#             "motion": 0.5,
#             "attack": "medium",
#             "stereo": 0.5,
#             "depth": False,
#             "echo": 0.25,
#             "human": 0.5,
#             "grain": "clean",
#         },
#         {
#             # "thinking": "The sad vibe is characterized by a low-key sound and a sense of melancholy. The low intensity and muted tones create a feeling of quiet sorrow. The 'low' intensity of the bass and the 'dark' tone of the pad support this feeling. The 'soft' attack and 'warm' tone of the pad create a sense of longing and a feeling of sadness.",
#             "tempo": 0.25,
#             "root": "e",
#             "mode": "minor",
#             "brightness": 0.25,
#             "space": 0.25,
#             "density": 5,
#             "bass": "drone",
#             "pad": "warm_slow",
#             "melody": "contemplative",
#             "rhythm": "soft_four",
#             "texture": "shimmer_slow",
#             "accent": "bells",
#             "motion": 0.25,
#             "attack": "soft",
#             "stereo": 0.25,
#             "depth": True,
#             "echo": 0.5,
#             "human": 0.5,
#             "grain": "gritty",
#         },
#         {
#             # "thinking": "The 'dead', a low volume synth with a simple sine wave and a low pitch.",
#             "tempo": 0.25,
#             "root": "c",
#             "mode": "minor",
#             "brightness": 0.25,
#             "space": 0.25,
#             "density": 2,
#             "bass": "drone",
#             "pad": "warm_slow",
#             "melody": "contemplative",
#             "rhythm": "none",
#             "texture": "shimmer_slow",
#             "accent": "bells",
#             "motion": 0.25,
#             "attack": "soft",
#             "stereo": 0.25,
#             "depth": False,
#             "echo": 0.25,
#             "human": 0.25,
#             "grain": "clean",
#         },
#     ]
#     demo_names = ["bubblegum.wav", "sad.wav", "dead.wav"]
#     demo_vibes = ["Bubblegum", "sad", "dead"]

#     for vibe, fname, config_dict in zip(demo_vibes, demo_names, demo_configs):
#         print(f"\n{'=' * 60}")
#         print(f"Vibe: {vibe}")
#         print("=" * 60)
#         config = SynthConfig.from_dict(config_dict)
#         print(json.dumps(config_dict, indent=2))
#         config_to_audio(config, fname, duration=20.0)
#         print(f"   Saved: {fname}")

#     print("\nAll demo audio exported.")

# if __name__ == "__main__":
#     print("VibeSynth V1/V2 - Pure Python Synthesis Engine")
#     print("=" * 50)

#     # Example 1: Direct config
#     print("\n1. Generating from direct config (Indian Wedding)...")
#     config = SynthConfig(
#         tempo=0.36,
#         root="d",
#         mode="dorian",
#         brightness=0.5,
#         space=0.75,
#         density=5,
#         bass="drone",
#         pad="warm_slow",
#         melody="ornamental",
#         rhythm="minimal",
#         texture="shimmer_slow",
#         accent="pluck",
#         motion=0.5,
#         attack="soft",
#         stereo=0.65,
#         depth=True,
#         echo=0.55,
#         human=0.18,
#         grain="warm",
#     )
#     config_to_audio(config, "indian_wedding.wav", duration=20.0)
#     print("   Saved: indian_wedding.wav")

#     # Example 2: From dict (simulating JSON from LLM)
#     print("\n2. Generating from dict (Dark Electronic)...")
#     dark_config = {
#         "tempo": 0.42,
#         "root": "a",
#         "mode": "minor",
#         "brightness": 0.4,
#         "space": 0.6,
#         "density": 6,
#         "layers": {
#             "bass": "pulsing",
#             "pad": "dark_sustained",
#             "melody": "arp_melody",
#             "rhythm": "electronic",
#             "texture": "shimmer",
#             "accent": "bells",
#         },
#         "motion": 0.65,
#         "attack": "sharp",
#         "stereo": 0.7,
#         "depth": True,
#         "echo": 0.5,
#         "human": 0.0,
#         "grain": "gritty",
#     }
#     dict_to_audio(dark_config, "dark_electronic.wav", duration=20.0)
#     print("   Saved: dark_electronic.wav")

#     # Example 3: From vibe string
#     print("\n3. Generating from vibe string (Underwater Cave)...")
#     generate_from_vibe(
#         "slow, peaceful, underwater cave, bioluminescence", "underwater_cave.wav", duration=20.0
#     )
#     print("   Saved: underwater_cave.wav")

#     print("\n" + "=" * 50)

#     # Example 4: Morph between two configs
#     print("\n4. Generating morph (Morning → Evening)...")

#     morning = SynthConfig(
#         tempo=0.30,
#         root="c",
#         mode="major",
#         brightness=0.6,
#         space=0.8,
#         bass="drone",
#         pad="warm_slow",
#         melody="minimal",
#         motion=0.3,
#         attack="soft",
#         echo=0.7,
#     )

#     evening = SynthConfig(
#         tempo=0.42,
#         root="a",
#         mode="minor",
#         brightness=0.45,
#         space=0.6,
#         bass="pulsing",
#         pad="dark_sustained",
#         melody="arp_melody",
#         motion=0.65,
#         attack="medium",
#         echo=0.5,
#     )

#     # Generate 2-minute morph with overlap-add (no volume dips)
#     audio = generate_tween_with_automation(morning, evening, duration=120.0)
#     # audio = assemble(morning, duration=30.0)  # Just morning, no morphing

#     sf.write("morning_to_evening.wav", audio, SAMPLE_RATE)
#     print("   Saved: morning_to_evening.wav")

#     print("\nDone! Generated 4 audio files.")
# ══════════════════════════════════════════════════════════════════════════════
#  HEURÍSTICA OFFLINE (texto → MusicConfig sin modelo)
# ══════════════════════════════════════════════════════════════════════════════

_PRESETS: dict[str, dict] = {
    "lluvia":     dict(mode="minor", root="d", tempo="slow", space="large",
                       texture="vinyl_crackle", pad="ambient_drift", rhythm="none",
                       melody="contemplative", brightness="dark"),
    "jazz":       dict(mode="dorian", root="f", tempo="medium", swing="light",
                       bass="walking", pad="warm_slow", melody="ornamental",
                       rhythm="brush", accent="piano_note", brightness="medium"),
    "épico":      dict(mode="major", root="c", tempo="fast", space="vast",
                       pad="cinematic", melody="heroic", rhythm="kit_medium",
                       accent="brass_hit", brightness="very_bright"),
    "ambient":    dict(mode="major", root="g", tempo="very_slow", space="vast",
                       texture="stars", pad="stacked_fifths", melody="minimal",
                       rhythm="none", brightness="dark"),
    "meditación": dict(mode="dorian", root="a", tempo="very_slow", space="large",
                       texture="breath", pad="dark_sustained", melody="minimal",
                       rhythm="heartbeat", brightness="very_dark"),
    "tormenta":   dict(mode="minor", root="e", tempo="fast", space="vast",
                       texture="shimmer", pad="cinematic", melody="rising",
                       rhythm="electronic", brightness="dark", depth=True),
    "noche":      dict(mode="minor", root="a", tempo="slow", space="large",
                       texture="stars", pad="dark_sustained", melody="contemplative",
                       rhythm="none", brightness="very_dark"),
    "amanecer":   dict(mode="major", root="d", tempo="slow", space="large",
                       texture="shimmer_slow", pad="warm_slow", melody="rising",
                       rhythm="none", brightness="bright"),
}


def _config_desde_vibe(texto: str) -> MusicConfig:
    texto_l = texto.lower()
    mejor, mejor_score = None, -1
    for kw, preset in _PRESETS.items():
        score = sum(1 for w in kw.split() if w in texto_l)
        if score > mejor_score:
            mejor_score, mejor = score, preset
    cfg = MusicConfig()
    if mejor:
        cfg = cfg.apply_patch(mejor)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
#  SÍNTESIS PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def _sintetizar(mc: MusicConfig, duracion: float, sr: int = DEFAULT_SR,
                t_offset: float = 0.0) -> np.ndarray:
    sc    = SynthConfig.from_music_config(mc)
    audio = assemble(sc, duracion, normalize=True, t_offset=t_offset)
    audio = np.asarray(audio, dtype=np.float32)
    # Fade-out de 50ms al final para evitar el corte brusco del fichero
    fade_n = min(int(0.050 * sr), len(audio) // 8)
    audio[-fade_n:] *= np.linspace(1.0, 0.0, fade_n, dtype=np.float32)
    return audio


def _guardar_wav(audio: np.ndarray, path: str, sr: int = DEFAULT_SR) -> None:
    try:
        import soundfile as sf  # type: ignore
        sf.write(path, audio, sr)
        return
    except ImportError:
        pass
    import wave
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())


def _interpolar_configs(a: MusicConfig, b: MusicConfig, t: float) -> MusicConfig:
    sa = SynthConfig.from_music_config(a)
    sb = SynthConfig.from_music_config(b)
    si = interpolate_configs(sa, sb, t)
    # Reconstruir MusicConfig desde SynthConfig interpolado
    def _nearest(val, tabla):
        return min(tabla, key=lambda k: abs(tabla[k] - val))
    mc = MusicConfig()
    mc.root            = si.root
    mc.mode            = si.mode
    mc.tempo           = _nearest(si.tempo, _TEMPO_MAP)
    mc.brightness      = _nearest(si.brightness, _BRIGHTNESS_MAP)
    mc.space           = _nearest(si.space, _SPACE_MAP)
    mc.motion          = _nearest(si.motion, _MOTION_MAP)
    mc.stereo          = _nearest(si.stereo, _STEREO_MAP)
    mc.echo            = _nearest(si.echo, _ECHO_MAP)
    mc.human           = _nearest(si.human, _HUMAN_MAP)
    mc.grain           = si.grain
    mc.attack          = si.attack
    mc.depth           = si.depth
    mc.density         = int(np.clip(si.density, 2, 6))
    mc.bass            = si.bass
    mc.pad             = si.pad
    mc.melody          = si.melody
    mc.rhythm          = si.rhythm
    mc.texture         = si.texture
    mc.accent          = si.accent
    mc.melody_engine   = si.melody_engine
    mc.melody_density  = _nearest(si.melody_density, _MEL_DENSITY_MAP)
    mc.syncopation     = _nearest(si.syncopation, _SYNCOPATION_MAP)
    mc.swing           = _nearest(si.swing, _SWING_MAP)
    mc.motif_repeat_prob = _nearest(si.motif_repeat_prob, _MOTIF_RPT_MAP)
    mc.step_bias       = _nearest(si.step_bias, _STEP_BIAS_MAP)
    mc.chromatic_prob  = _nearest(si.chromatic_prob, _CHROMATIC_MAP)
    mc.cadence_strength= _nearest(si.cadence_strength, _CADENCE_MAP)
    mc.tension_curve   = si.tension_curve
    mc.harmony_style   = si.harmony_style
    _inv_chg = {v: k for k, v in _CHORD_CHG_MAP.items()}
    mc.chord_change_bars = _inv_chg.get(si.chord_change_bars, "medium")
    mc.chord_extensions= si.chord_extensions
    mc.phrase_len_bars = si.phrase_len_bars
    mc.register_min_oct= si.register_min_oct
    mc.register_max_oct= si.register_max_oct
    return mc


def _resolver_entrada(valor: str) -> MusicConfig:
    if valor.endswith(".json") and Path(valor).exists():
        return MusicConfig.from_json(valor)
    return _config_desde_vibe(valor)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES CLI
# ══════════════════════════════════════════════════════════════════════════════

def _die(msg: str) -> None:
    print(f"✗ Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _banner(titulo: str) -> None:
    print("═" * 65)
    print(f"  LATENTSCORE COMPOSER v{VERSION} — {titulo}")
    print("═" * 65)


def _parsear_patch(parches: list[str]) -> dict:
    resultado = {}
    for p in parches:
        if "=" not in p:
            _die(f"Formato de patch inválido: {p!r}  (usar clave=valor)")
        k, _, v = p.partition("=")
        resultado[k.strip()] = v.strip()
    return resultado


def _imprimir_config(cfg: MusicConfig) -> None:
    col = 22
    for k, v in cfg.to_dict().items():
        print(f"  {k:<{col}}: {v}")


def _nombre_salida(base: str, sufijo: str = "") -> str:
    return f"ls_{base.replace(' ','_')[:30]}{sufijo}.wav"


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_render(args: argparse.Namespace) -> None:
    _banner("RENDER")
    if args.config:
        print(f"  Config   : {args.config}")
        cfg = MusicConfig.from_json(args.config)
    elif args.vibe:
        print(f"  Vibe     : {args.vibe!r}")
        print(f"\n[1/3] Resolviendo config…")
        cfg = _config_desde_vibe(args.vibe)
    else:
        _die("Indica un texto (vibe) o --config FILE")

    if args.patch:
        patch = _parsear_patch(args.patch)
        cfg = cfg.apply_patch(patch)
        print(f"  Patch    : {patch}")

    duracion = args.dur
    sr       = args.sr
    salida   = args.output or _nombre_salida(args.vibe or Path(args.config).stem)

    if args.verbose:
        print("\n  Parámetros resueltos:")
        _imprimir_config(cfg)

    print(f"\n[2/3] Sintetizando {duracion:.0f} s a {sr} Hz…")
    audio = _sintetizar(cfg, duracion, sr)

    print(f"\n[3/3] Guardando → {salida}")
    _guardar_wav(audio, salida, sr)

    print("\n" + "═" * 65)
    print(f"  Salida   : {salida}")
    print(f"  Duración : {duracion:.1f} s  ({len(audio)/sr:.2f} s efectivos)")
    print(f"  Pico     : {np.max(np.abs(audio)):.4f}")
    print("═" * 65)


def cmd_morph(args: argparse.Namespace) -> None:
    _banner("MORPH")
    print(f"  A        : {args.a!r}")
    print(f"  B        : {args.b!r}")
    print(f"  Pasos    : {args.steps}")
    print(f"  Duración : {args.dur} s total")

    print(f"\n[1/4] Resolviendo config A…")
    cfg_a = _resolver_entrada(args.a)
    print(f"[2/4] Resolviendo config B…")
    cfg_b = _resolver_entrada(args.b)

    if args.verbose:
        print("\n  Config A:")
        _imprimir_config(cfg_a)
        print("\n  Config B:")
        _imprimir_config(cfg_b)

    steps    = max(2, args.steps)
    sr       = args.sr
    salida   = args.output or f"ls_morph_{steps}steps.wav"

    # Crossfade: 20% de dur_paso, mínimo 2s, máximo 4s
    # Un crossfade largo elimina los cortes de energía entre pasos:
    # el sintetizador tiene mucho sustain (pad, bass drone) y un fade
    # corto produce un escalón de volumen audible.
    dur_paso = args.dur / steps
    fade_s   = float(np.clip(dur_paso * 0.20, 2.0, 4.0))
    fade_n   = int(fade_s * sr)

    print(f"  Crossfade: {fade_s:.1f} s  ({dur_paso:.1f} s/paso)")

    # Sintetizar sin normalización individual: normalizar solo al final
    # sobre el resultado completo, para que pasos suaves y fuertes
    # tengan niveles coherentes entre sí.
    print(f"\n[3/4] Sintetizando {steps} pasos…")
    sc_a = SynthConfig.from_music_config(cfg_a)
    sc_b = SynthConfig.from_music_config(cfg_b)

    fragmentos: list[np.ndarray] = []
    t_acum = 0.0
    for i in range(steps):
        t = i / (steps - 1)
        cfg_i = _interpolar_configs(cfg_a, cfg_b, t)
        sc_i  = SynthConfig.from_music_config(cfg_i)
        # Sin normalizar: assemble con normalize=False para preservar
        # niveles relativos entre pasos
        audio_i = assemble(sc_i, dur_paso, normalize=False, t_offset=t_acum)
        audio_i = np.asarray(audio_i, dtype=np.float32)
        fragmentos.append(audio_i)
        t_acum += dur_paso
        print(f"  paso {i+1}/{steps}  t={t:.2f}  rms={np.sqrt(np.mean(audio_i**2)):.4f}  ✓")

    print(f"\n[4/4] Ensamblando con crossfade de {fade_s:.1f} s…")

    # Crossfade con curva de igual potencia (√t) en lugar de lineal:
    # mantiene el RMS constante durante la transición, eliminando
    # el "hueco" de volumen que produce el crossfade lineal.
    resultado = fragmentos[0].copy()
    for frag in fragmentos[1:]:
        n_fade = min(fade_n, len(resultado), len(frag))
        if n_fade < 2:
            resultado = np.concatenate([resultado, frag])
            continue
        ramp   = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        fade_in  = np.sqrt(ramp)        # curva de igual potencia
        fade_out = np.sqrt(1.0 - ramp)
        resultado[-n_fade:] = resultado[-n_fade:] * fade_out + frag[:n_fade] * fade_in
        resultado = np.concatenate([resultado, frag[n_fade:]])

    # Normalización global única al final
    peak = np.max(np.abs(resultado))
    if peak > 0:
        resultado = resultado / peak * 0.85

    # Fade-out de 100ms al final
    fade_end = min(int(0.10 * sr), len(resultado) // 8)
    resultado[-fade_end:] *= np.linspace(1.0, 0.0, fade_end, dtype=np.float32)

    _guardar_wav(resultado, salida, sr)
    print("\n" + "═" * 65)
    print(f"  Salida   : {salida}")
    print(f"  Duración : {len(resultado)/sr:.2f} s")
    print(f"  RMS      : {np.sqrt(np.mean(resultado**2)):.4f}")
    print("═" * 65)


def cmd_chain(args: argparse.Namespace) -> None:
    _banner("CHAIN")
    vibes: list[str] = []
    if len(args.items) == 1:
        item = args.items[0]
        p = Path(item)
        if p.suffix == ".txt" and p.exists():
            vibes = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()
                     if l.strip() and not l.startswith("#")]
        elif p.suffix == ".json" and p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                vibes = [str(x) for x in data]
            else:
                _die("El JSON de chain debe ser una lista")
        else:
            vibes = [item]
    else:
        vibes = list(args.items)

    if not vibes:
        _die("Lista de vibes vacía")

    print(f"  Vibes    : {len(vibes)}")
    print(f"  Dur/item : {args.dur_each} s")
    print(f"  Fade     : {args.fade} s")

    sr     = args.sr
    dur    = args.dur_each
    fade_n = int(args.fade * sr)
    salida = args.output or f"ls_chain_{len(vibes)}vibes.wav"
    resultado = np.array([], dtype=np.float32)

    for idx, vibe in enumerate(vibes, 1):
        print(f"\n[{idx}/{len(vibes)}] {vibe!r}")
        cfg   = _resolver_entrada(vibe)
        audio = _sintetizar(cfg, dur, sr)
        if args.verbose:
            _imprimir_config(cfg)
        if len(resultado) == 0:
            resultado = audio
        elif fade_n > 0 and len(resultado) >= fade_n and len(audio) >= fade_n:
            resultado[-fade_n:] = (resultado[-fade_n:] * np.linspace(1,0,fade_n)
                                 + audio[:fade_n] * np.linspace(0,1,fade_n))
            resultado = np.concatenate([resultado, audio[fade_n:]])
        else:
            resultado = np.concatenate([resultado, audio])
        print(f"  ✓  acumulado: {len(resultado)/sr:.1f} s")

    print(f"\nGuardando → {salida}")
    _guardar_wav(resultado, salida, sr)
    print("\n" + "═" * 65)
    print(f"  Salida   : {salida}")
    print(f"  Duración : {len(resultado)/sr:.2f} s")
    print("═" * 65)


def cmd_config(args: argparse.Namespace) -> None:
    sub = args.config_cmd
    if sub == "show":
        _banner("CONFIG — SHOW")
        cfg = MusicConfig.from_json(args.file)
        _imprimir_config(cfg)
    elif sub == "from-vibe":
        _banner("CONFIG — FROM-VIBE")
        print(f"  Texto    : {args.text!r}")
        cfg = _config_desde_vibe(args.text)
        salida = args.output or f"ls_{args.text.replace(' ','_')[:30]}.json"
        cfg.to_json(salida)
        print(f"\n  Config guardada → {salida}")
        _imprimir_config(cfg)
    elif sub == "patch":
        _banner("CONFIG — PATCH")
        cfg = MusicConfig.from_json(args.file)
        if args.patch:
            cfg = cfg.apply_patch(_parsear_patch(args.patch))
        salida = args.output or args.file.replace(".json", "_patched.json")
        cfg.to_json(salida)
        print(f"  Config guardada → {salida}")
        _imprimir_config(cfg)
    elif sub == "list-styles":
        _banner("CONFIG — ESTILOS DISPONIBLES")
        estilos = {
            "root": ROOT_NOTES, "mode": MODES, "tempo": TEMPOS,
            "brightness": BRIGHTNESS, "space": SPACES, "motion": MOTIONS,
            "stereo": STEREOS, "echo": ECHOES, "human": HUMANS,
            "grain": GRAINS, "attack": ATTACKS,
            "density": [str(d) for d in DENSITIES],
            "bass": BASS_STYLES, "pad": PAD_STYLES, "melody": MELODY_STYLES,
            "rhythm": RHYTHM_STYLES, "texture": TEXTURE_STLS, "accent": ACCENT_STLS,
            "melody_density": MEL_DENSITY, "syncopation": SYNCOPATION,
            "swing": SWINGS, "motif_repeat_prob": MOTIF_REPEAT,
            "step_bias": STEP_BIAS, "chromatic_prob": CHROMATIC,
            "cadence_strength": CADENCE, "chord_change_bars": CHORD_CHANGE,
            "chord_extensions": CHORD_EXT,
            "phrase_len_bars": [str(p) for p in PHRASE_BARS],
            "tension_curve": TENSION_CURVES, "harmony_style": HARMONY_STLS,
        }
        col = 22
        for campo, vals in estilos.items():
            vals_str = "  ".join(vals)
            lineas = textwrap.wrap(vals_str, width=50)
            print(f"  {campo:<{col}}: {lineas[0]}")
            for l in lineas[1:]:
                print(f"  {' '*col}  {l}")
    else:
        _die(f"Subcomando desconocido: {sub!r}")


def cmd_inspect(args: argparse.Namespace) -> None:
    _banner("INSPECT")
    sub = args.inspect_cmd
    if sub == "synth":
        def _check(nombre, fn):
            try:
                fn(); print(f"    ✓  {nombre}")
            except ImportError:
                print(f"    ✗  {nombre}  (pip install {nombre.split()[0]})")
            except Exception as e:
                print(f"    ⚠  {nombre}  ({e})")
        print("\n  Dependencias:")
        _check("numpy",     lambda: __import__("numpy"))
        _check("scipy",     lambda: __import__("scipy"))
        _check("soundfile", lambda: __import__("soundfile"))
        print("\n  Motor de síntesis: integrado (latentscore/synth.py)")
        print("\n  Presets heurísticos offline:")
        for k in _PRESETS:
            print(f"    • {k}")
    elif sub == "wav":
        path = getattr(args, "file", None)
        if not path or not Path(path).exists():
            _die(f"Fichero no encontrado: {path}")
        try:
            import soundfile as sf  # type: ignore
            data, sr = sf.read(path, dtype="float32")
            dur  = len(data) / sr
            pico = float(np.max(np.abs(data)))
            rms  = float(np.sqrt(np.mean(data**2)))
            print(f"\n  Fichero  : {path}")
            print(f"  SR       : {sr} Hz")
            print(f"  Muestras : {len(data)}")
            print(f"  Duración : {dur:.2f} s")
            print(f"  Canales  : {1 if data.ndim == 1 else data.shape[1]}")
            print(f"  Pico     : {pico:.4f}  ({20*math.log10(pico+1e-9):.1f} dBFS)")
            print(f"  RMS      : {rms:.4f}   ({20*math.log10(rms+1e-9):.1f} dBFS)")
        except ImportError:
            _die("soundfile no instalado: pip install soundfile")
    else:
        _die(f"Subcomando de inspect desconocido: {sub!r}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latentscore_composer",
        description="Síntesis procedimental de audio ambiente (motor latentscore integrado)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")
    sub.required = True

    def _common(p, vibe=False):
        if vibe:
            p.add_argument("vibe", nargs="?", default=None, metavar="TEXTO")
        p.add_argument("--output",  metavar="FILE", default=None)
        p.add_argument("--dur",     type=float, default=DEFAULT_DUR, metavar="SEG")
        p.add_argument("--sr",      type=int, default=DEFAULT_SR, metavar="HZ")
        p.add_argument("--verbose", action="store_true")

    # render
    p_r = sub.add_parser("render", help="Texto o config → WAV")
    _common(p_r, vibe=True)
    p_r.add_argument("--config", metavar="FILE", default=None)
    p_r.add_argument("--patch", nargs="+", metavar="CAMPO=VALOR")
    p_r.set_defaults(func=cmd_render)

    # morph
    p_m = sub.add_parser("morph", help="Transición suave entre dos vibes")
    p_m.add_argument("a", metavar="A")
    p_m.add_argument("b", metavar="B")
    p_m.add_argument("--steps", type=int, default=6)
    _common(p_m)
    p_m.set_defaults(func=cmd_morph)

    # chain
    p_c = sub.add_parser("chain", help="Secuencia de vibes → WAV concatenado")
    p_c.add_argument("items", nargs="+", metavar="VIBE|FILE")
    p_c.add_argument("--dur-each", type=float, default=DEFAULT_DUR, metavar="SEG")
    p_c.add_argument("--fade",     type=float, default=2.0, metavar="SEG")
    p_c.add_argument("--output",   metavar="FILE", default=None)
    p_c.add_argument("--sr",       type=int, default=DEFAULT_SR)
    p_c.add_argument("--verbose",  action="store_true")
    p_c.set_defaults(func=cmd_chain)

    # config
    p_cfg = sub.add_parser("config", help="Gestión de MusicConfig")
    cfg_sub = p_cfg.add_subparsers(dest="config_cmd")
    cfg_sub.required = True

    p_show = cfg_sub.add_parser("show");  p_show.add_argument("file", metavar="FILE")
    p_fv   = cfg_sub.add_parser("from-vibe")
    p_fv.add_argument("text", metavar="TEXTO")
    p_fv.add_argument("--output", metavar="FILE", default=None)
    p_pat  = cfg_sub.add_parser("patch")
    p_pat.add_argument("file",  metavar="FILE")
    p_pat.add_argument("patch", nargs="*", metavar="CAMPO=VALOR")
    p_pat.add_argument("--output", metavar="FILE", default=None)
    cfg_sub.add_parser("list-styles")
    p_cfg.set_defaults(func=cmd_config)

    # inspect
    p_ins = sub.add_parser("inspect", help="Diagnóstico del entorno")
    ins_sub = p_ins.add_subparsers(dest="inspect_cmd")
    ins_sub.required = True
    ins_sub.add_parser("synth")
    p_iw = ins_sub.add_parser("wav")
    p_iw.add_argument("file", metavar="FILE", nargs="?", default=None)
    p_ins.set_defaults(func=cmd_inspect)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
