#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          DRUMMER  v0.2                                       ║
║         Motor de percusión rítmica para obras MIDI                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FLUJO DE TRABAJO TÍPICO                                                     ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  1. learn    → construir biblioteca desde tu colección                       ║
║  2. extract  → añadir grooves favoritos con nombre propio                    ║
║  3. add      → poner percusión a una obra nueva                              ║
║  4. transform / morph → editar y evolucionar                                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  learn — Aprende grooves desde una carpeta de MIDIs mixtos                  ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Extrae el canal 10 de cada MIDI, vectoriza los patrones y los agrupa        ║
║  por similitud con KMeans. Genera drum_library.json con prototipos           ║
║  etiquetables manualmente.                                                   ║
║                                                                              ║
║    # Básico: aprende de todos los MIDIs de la carpeta                        ║
║    python drummer.py learn ./midis/                                          ║
║                                                                              ║
║    # Con más clusters y salida personalizada                                 ║
║    python drummer.py learn ./midis/ --clusters 32 --output mi_lib.json      ║
║                                                                              ║
║    # Ver qué se omite durante el proceso                                     ║
║    python drummer.py learn ./midis/ --verbose                                ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  analyze — Analiza la percusión de un MIDI                                  ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Muestra síncopa, densidad, feel, fills detectados y grid ASCII.             ║
║                                                                              ║
║    # Informe en pantalla                                                     ║
║    python drummer.py analyze cancion.mid                                     ║
║                                                                              ║
║    # Exportar análisis a JSON (útil para scripts externos)                   ║
║    python drummer.py analyze cancion.mid --export analisis.json              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  extract — Extrae un groove concreto y lo añade a la biblioteca             ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  A diferencia de learn (automático), extract es curatorial: tú eliges       ║
║  qué MIDI y qué nombre. Separa el patrón base de los fills detectados        ║
║  estadísticamente. Si el patrón no es repetitivo, extrae y avisa.           ║
║                                                                              ║
║    # Añadir groove a la biblioteca por defecto                               ║
║    python drummer.py extract bossa_nova.mid --name "bossa_suave"            ║
║                                                                              ║
║    # Añadir a una biblioteca concreta                                        ║
║    python drummer.py extract jazz_trio.mid --name "jazz_brush" --library mi_lib.json ║
║                                                                              ║
║    # Exportar también el patrón base como MIDI independiente                 ║
║    python drummer.py extract funk_jam.mid --name "funk_16th" --export-midi base.mid  ║
║                                                                              ║
║    # Sobreescribir si ya existe ese nombre                                   ║
║    python drummer.py extract nuevo.mid --name "funk_16th" --force           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  add — Añade percusión a una obra que no la tiene                           ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Detecta la duración real de la obra (ticks absolutos, no deltas),          ║
║  divide en secciones con tensión variable y elige grooves de la             ║
║  biblioteca para cada una. Genera fills en las transiciones.                ║
║                                                                              ║
║    # Básico: usa grooves sintéticos de semilla                               ║
║    python drummer.py add melodia.mid                                         ║
║                                                                              ║
║    # Con biblioteca aprendida (grooves reales)                               ║
║    python drummer.py add melodia.mid --library drum_library.json            ║
║                                                                              ║
║    # Con curvas de tension_designer (5 dimensiones dinámicas)                ║
║    python drummer.py add melodia.mid --tension-curve curves.json            ║
║                                                                              ║
║    # Con fingerprint de midi_dna_unified (secciones A/B/C automáticas)      ║
║    python drummer.py add melodia.mid --fingerprint melodia.fingerprint.json ║
║                                                                              ║
║    # Todo junto + salida personalizada                                       ║
║    python drummer.py add melodia.mid --library drum_library.json            ║
║                       --tension-curve curves.json --output con_bateria.mid   ║
║                                                                              ║
║    # Sin humanización (grid perfecto, útil para comparar)                   ║
║    python drummer.py add melodia.mid --no-humanize                          ║
║                                                                              ║
║    # Forzar aunque ya tenga percusión                                        ║
║    python drummer.py add obra.mid --force                                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  transform — Transforma la percusión existente                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Todas las opciones son combinables. Las curvas de tension_designer          ║
║  activan automáticamente síncopa, densidad, swing y polirritmo dinámicos    ║
║  salvo que se sobrescriban con flags estáticos.                              ║
║                                                                              ║
║  · SÍNCOPA                                                                   ║
║    # Estática: 0=cuadrado, 1=máxima síncopa                                 ║
║    python drummer.py transform obra.mid --syncopate 0.6                     ║
║                                                                              ║
║    # Dinámica desde curva tension (varía compás a compás)                   ║
║    python drummer.py transform obra.mid --tension-curve curves.json         ║
║                                                                              ║
║  · SWING                                                                     ║
║    # Estático: 0=straight, 0.34≈jazz (ratio 0.67), 1=hard swing            ║
║    python drummer.py transform obra.mid --swing 0.34                        ║
║                                                                              ║
║    # Dinámico: incluir clave "swing" en curves.json                         ║
║    python drummer.py transform obra.mid --tension-curve curves.json         ║
║    # (el hihat swingea al 100%, kick al 60%, snare siempre straight)        ║
║                                                                              ║
║  · DENSIDAD                                                                  ║
║    # Reducir golpes (0=sin cambio, 1=eliminar casi todo)                    ║
║    python drummer.py transform obra.mid --thin 0.4                          ║
║                                                                              ║
║    # Añadir ghost notes (0=sin cambio, 1=máximas ghost notes)               ║
║    python drummer.py transform obra.mid --thicken 0.5                       ║
║                                                                              ║
║    # Dinámica desde curva activity (thin en partes bajas, thicken en altas) ║
║    python drummer.py transform obra.mid --tension-curve curves.json         ║
║                                                                              ║
║  · POLIRRITMO                                                                ║
║    # Añadir capa de 3 golpes por compás (3:4) en clave/maracas              ║
║    python drummer.py transform obra.mid --polyrhythm 3                      ║
║                                                                              ║
║    # 5:4 (quintillo sobre cuatro)                                            ║
║    python drummer.py transform obra.mid --polyrhythm 5                      ║
║                                                                              ║
║    # Dinámico: se activa sólo donde harmony > 0.7 en curves.json            ║
║    python drummer.py transform obra.mid --tension-curve curves.json         ║
║                                                                              ║
║  · PHASE SHIFT (Steve Reich)                                                 ║
║    # Desplazar 1 paso de grid por compás                                     ║
║    python drummer.py transform obra.mid --phase-shift 1                     ║
║                                                                              ║
║    # Desplazar 2 pasos por compás (más rápido)                               ║
║    python drummer.py transform obra.mid --phase-shift 2                     ║
║                                                                              ║
║  · HUMANIZACIÓN                                                              ║
║    # 0=mecánico, 1=máxima variación de timing y velocity                    ║
║    python drummer.py transform obra.mid --humanize 0.7                      ║
║                                                                              ║
║  · TARGET EMOCIONAL (lenguaje natural)                                       ║
║    python drummer.py transform obra.mid --emotion "más urgente"             ║
║    python drummer.py transform obra.mid --emotion "relajado"                ║
║    python drummer.py transform obra.mid --emotion "groovy"                  ║
║    python drummer.py transform obra.mid --emotion "minimalista"             ║
║    python drummer.py transform obra.mid --emotion "denso"                   ║
║                                                                              ║
║  · COMBINACIONES TÍPICAS                                                     ║
║    # Jazz: swing de triplillo + humanización alta                            ║
║    python drummer.py transform obra.mid --swing 0.34 --humanize 0.8        ║
║                                                                              ║
║    # Funk: síncopa alta + ghost notes + humanización                         ║
║    python drummer.py transform obra.mid --syncopate 0.65 --thicken 0.4 --humanize 0.6 ║
║                                                                              ║
║    # Todo dinámico desde tension_designer                                    ║
║    python drummer.py transform obra.mid --tension-curve curves.json --humanize 0.6    ║
║                       --output obra_transformed.mid                                  ║
║                                                                              ║
║    # Reich: phase shift + sin humanizar (el efecto requiere precisión)      ║
║    python drummer.py transform obra.mid --phase-shift 1 --no-humanize       ║
║    # (nota: --no-humanize no existe en transform, simplemente omitir        ║
║    #  --humanize ya deja el timing perfecto)                                 ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  morph — Morphing gradual entre dos grooves en N pasos                      ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Genera N MIDIs intermedios interpolando estructura (kick/snare),            ║
║  subdivisión (hihat + swing ratio) y color (toms/clave) como capas          ║
║  independientes. También interpola tempo entre A y B.                        ║
║  Exporta un manifiesto JSON con la lista de archivos en orden.               ║
║                                                                              ║
║    # Básico: 8 pasos lineales entre dos grooves                              ║
║    python drummer.py morph rock.mid jazz.mid                                 ║
║                                                                              ║
║    # 6 pasos, interpolación sigmoide (suave en extremos, rápida en centro)  ║
║    python drummer.py morph rock.mid jazz.mid --steps 6 --mode sigmoid       ║
║                                                                              ║
║    # Normalizar ambos a 2 compases (si tienen distinta longitud)             ║
║    python drummer.py morph rock.mid jazz.mid --steps 8 --bars 2             ║
║                                                                              ║
║    # Guardar en carpeta específica                                            ║
║    python drummer.py morph grooveA.mid grooveB.mid --steps 10 --output-dir ./morphs/ ║
║                                                                              ║
║    # Sin humanización (útil para escuchar el morphing puro)                 ║
║    python drummer.py morph grooveA.mid grooveB.mid --no-humanize            ║
║                                                                              ║
║    # Modos de interpolación disponibles:                                     ║
║    #   linear      — cambio constante (default)                              ║
║    #   sigmoid     — suave en extremos, rápido en el centro                 ║
║    #   exponential — lento al principio, acelerado al final                 ║
║    #   sinusoidal  — suavizado coseno, muy fluido                           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  SWING DINÁMICO — cómo funciona                                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  El ratio (0.50=straight … 0.75=hard swing) se aplica por instrumento        ║
║  según su función, no de forma uniforme:                                     ║
║    hihat / ride  (subdivision) → 100% del ratio                             ║
║    kick          (pulse)       →  60% del ratio                             ║
║    snare         (backbeat)    →   0% (siempre straight)                    ║
║    toms / color                →  50–70%                                    ║
║  A tempos > 120bpm el swing se reduce automáticamente (feel físico real).   ║
║                                                                              ║
║  El flag --swing N normaliza así:  0 → ratio 0.50 (straight)               ║
║                                   0.34 → ratio 0.59 (funk suave)           ║
║                                   0.67 → ratio 0.67 (jazz, triplillo)      ║
║                                    1.0 → ratio 0.75 (hard swing)           ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INTEGRACIÓN CON TENSION_DESIGNER (curves.json)                              ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Pasar --tension-curve curves.json activa todos los parámetros              ║
║  dinámicos a la vez. Formato esperado:                                       ║
║                                                                              ║
║    {                                                                         ║
║      "tension":  [0.1, 0.3, 0.7, 0.9, 0.6, 0.2],  → síncopa kick/snare   ║
║      "activity": [0.2, 0.4, 0.8, 1.0, 0.7, 0.3],  → densidad (thin/thick)║
║      "swing":    [0.0, 0.0, 0.4, 0.8, 0.5, 0.0],  → ratio swing hihat    ║
║      "harmony":  [0.2, 0.3, 0.5, 0.8, 0.6, 0.2],  → polirritmo dinámico  ║
║      "register": [0.4, 0.5, 0.7, 1.0, 0.8, 0.4]   → velocity media       ║
║    }                                                                         ║
║                                                                              ║
║  Los arrays pueden tener cualquier longitud; se interpolan al número        ║
║  de compases real de la obra.                                                ║
║  Build-up automático: si activity sube > 0.3 en 2 compases → fill.         ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  INSTRUMENTOS GM (canal 10, MIDI note)                                       ║
║    36 Kick   38 Snare  40 Snare-e  42 HH-closed  44 HH-pedal               ║
║    46 HH-open  49 Crash  51 Ride  57 Crash2                                 ║
║    41 Tom-lo  45 Tom-mid  48 Tom-hi  70 Maracas  75 Clave                  ║
║                                                                              ║
║  DEPENDENCIAS                                                                ║
║    mido, numpy                    (requeridas)                               ║
║    scikit-learn                   (opcional — clustering en learn)           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mido
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES GM
# ══════════════════════════════════════════════════════════════════════════════

GM = {
    "kick":      36,
    "snare":     38,
    "snare_e":   40,
    "hihat_c":   42,
    "hihat_p":   44,
    "hihat_o":   46,
    "crash":     49,
    "ride":      51,
    "crash2":    57,
    "tom_lo":    41,
    "tom_mid":   45,
    "tom_hi":    48,
    "maracas":   70,
    "clave":     75,
}
GM_INV = {v: k for k, v in GM.items()}

# Grupo funcional de cada instrumento
FUNC = {
    "kick":    "pulse",      # lleva el pulso
    "snare":   "backbeat",   # acentúa 2 y 4
    "snare_e": "backbeat",
    "hihat_c": "subdivision",# subdivide el tiempo
    "hihat_p": "subdivision",
    "hihat_o": "accent",     # acento
    "crash":   "marker",     # marcador de sección
    "crash2":  "marker",
    "ride":    "subdivision",
    "tom_lo":  "fill",
    "tom_mid": "fill",
    "tom_hi":  "fill",
    "maracas": "color",
    "clave":   "color",
}

# Peso métrico de cada subdivisión en un compás de 4/4 a 16 pasos
# 1=tiempo fuerte, 0=parte más débil
METRIC_WEIGHT_16 = [
    1.0, 0.0, 0.25, 0.0,   # tiempo 1
    0.5, 0.0, 0.25, 0.0,   # tiempo 2
    0.75,0.0, 0.25, 0.0,   # tiempo 3
    0.5, 0.0, 0.25, 0.0,   # tiempo 4
]

METRIC_WEIGHT_32 = []
for w in METRIC_WEIGHT_16:
    METRIC_WEIGHT_32.extend([w, w * 0.5])

PERCUSSION_CHANNEL = 9   # canal MIDI 10 (0-indexed)

# Ratio de swing que cada función de instrumento aplica sobre el ratio global.
# subdivision (hihat, ride) = 100%; pulse (kick) = 60%; backbeat (snare) = 0%
SWING_FACTOR_BY_FUNC = {
    "subdivision": 1.00,
    "pulse":       0.60,
    "backbeat":    0.00,
    "accent":      0.80,
    "fill":        0.50,
    "color":       0.70,
    "marker":      0.00,
}

# Etiquetas de feel según swing ratio inferido del género
GENRE_SWING_PROFILE = {
    # label → (swing_ratio, swing_factor_override_por_func)
    "jazz":        (0.67, None),   # triplillo clásico
    "swing":       (0.67, None),
    "funk":        (0.55, None),   # suave, casi straight
    "bossa":       (0.50, None),   # straight
    "rock":        (0.50, None),
    "metal":       (0.50, None),
    "afrobeat":    (0.50, None),
    "reggae":      (0.50, None),
    "blues":       (0.62, None),
    "gospel":      (0.60, None),
    "generic":     (0.50, None),
}

# ══════════════════════════════════════════════════════════════════════════════
#  CURVE INTERPOLATOR — base compartida para add, transform, morph
# ══════════════════════════════════════════════════════════════════════════════

class CurveInterpolator:
    """
    Interpola las curvas de tension_designer compás a compás y las traduce
    a parámetros concretos de percusión.

    Curvas soportadas (todas opcionales):
      tension   → síncopa del kick y snare         (0–1)
      activity  → densidad general                  (0–1)
      swing     → ratio de swing del hihat          (0–1, mapea a 0.50–0.75)
      harmony   → complejidad/polirritmo            (0–1)
      register  → velocity media                    (0–1)

    Si una curva no está presente se usa el valor por defecto.
    """

    SWING_MIN = 0.50   # straight
    SWING_MAX = 0.75   # hard swing

    DEFAULTS = {
        "tension":  0.4,
        "activity": 0.5,
        "swing":    0.0,   # 0 = straight (sin swing)
        "harmony":  0.3,
        "register": 0.5,
    }

    def __init__(self, curves: dict, total_bars: int, tempo: float = 120.0):
        self.curves     = curves
        self.total_bars = total_bars
        self.tempo      = tempo

    def _interp(self, name: str, bar: int) -> float:
        """Interpola linealmente la curva 'name' para el compás 'bar'."""
        arr = self.curves.get(name)
        default = self.DEFAULTS.get(name, 0.5)
        if not arr or len(arr) == 0:
            return default
        if len(arr) == 1:
            return float(arr[0])
        idx = bar / max(1, self.total_bars - 1) * (len(arr) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(arr) - 1)
        alpha = idx - lo
        return float(arr[lo]) * (1 - alpha) + float(arr[hi]) * alpha

    def get(self, name: str, bar: int) -> float:
        return self._interp(name, bar)

    def swing_ratio(self, bar: int) -> float:
        """
        Devuelve el ratio de swing absoluto (0.50–0.75) para el compás 'bar'.
        A tempos altos el swing se aplana automáticamente hacia straight.
        """
        raw   = self._interp("swing", bar)   # 0–1
        ratio = self.SWING_MIN + raw * (self.SWING_MAX - self.SWING_MIN)

        # Reducción por tempo: a 200bpm no se puede mantener swing de 0.67
        # Fórmula lineal: full swing hasta 120bpm, llano a 200bpm
        tempo_factor = 1.0 - max(0, (self.tempo - 120) / 80) * 0.5
        ratio = self.SWING_MIN + (ratio - self.SWING_MIN) * tempo_factor
        return float(np.clip(ratio, self.SWING_MIN, self.SWING_MAX))

    def drum_params(self, bar: int) -> dict:
        """
        Devuelve todos los parámetros de percusión para el compás 'bar'.
        """
        return {
            "syncope":      self._interp("tension",  bar),
            "density":      self._interp("activity", bar),
            "swing_ratio":  self.swing_ratio(bar),
            "complexity":   self._interp("harmony",  bar),
            "vel_scale":    0.7 + self._interp("register", bar) * 0.6,
        }

    def build_up_bars(self, threshold: float = 0.30) -> list[int]:
        """
        Detecta compases donde 'activity' sube bruscamente → colocar fill.
        Umbral: incremento > threshold en 2 compases consecutivos.
        """
        arr = self.curves.get("activity", [])
        if len(arr) < 2:
            return []
        fills = []
        for i in range(1, self.total_bars):
            v_prev = self._interp("activity", i - 1)
            v_curr = self._interp("activity", i)
            if v_curr - v_prev > threshold:
                fills.append(i - 1)   # fill al final del compás anterior
        return fills


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrumEvent:
    """Un golpe de percusión listo para exportar a MIDI."""
    time_ticks: int          # posición absoluta en ticks
    pitch: int               # nota GM
    velocity: int            # 1-127
    duration_ticks: int = 20 # duración (percusión: corta)
    timing_offset: int = 0   # micro-timing humanización (±ticks)

    @property
    def instrument(self) -> str:
        return GM_INV.get(self.pitch, f"note_{self.pitch}")


@dataclass
class RhythmPattern:
    """
    Representación dual de un patrón rítmico.
    Grid float por instrumento + metadatos semánticos.
    """
    # CAPA GRID: instrumento → array[resolution] de velocidades (0=silencio)
    grid: dict = field(default_factory=dict)
    resolution: int = 16      # subdivisiones por compás
    bars: int = 1             # cuántos compases cubre el patrón
    tempo: float = 120.0

    # METADATOS SEMÁNTICOS
    groove_family: str = "generic"
    syncope_level: float = 0.0     # 0=cuadrado, 1=máxima síncopa
    density: float = 0.0           # golpes / subdivisión
    feel: str = "straight"         # straight | swing | behind | ahead
    has_fills: bool = False
    energy: float = 0.5            # 0-1

    # VECTOR DE CARACTERÍSTICAS (para clustering/búsqueda)
    feature_vector: list = field(default_factory=list)

    def copy(self) -> "RhythmPattern":
        import copy
        return copy.deepcopy(self)

    def to_events(self, ticks_per_beat: int = 480, start_bar: int = 0) -> list:
        """Convierte el grid a lista de DrumEvent."""
        events = []
        ticks_per_bar = ticks_per_beat * 4  # asume 4/4
        ticks_per_step = ticks_per_bar // self.resolution

        for instr, vel_array in self.grid.items():
            pitch = GM.get(instr)
            if pitch is None:
                continue
            for step, vel_float in enumerate(vel_array):
                if vel_float <= 0.0:
                    continue
                bar_offset = step // self.resolution
                step_in_bar = step % self.resolution
                abs_bar = start_bar + bar_offset
                t = abs_bar * ticks_per_bar + step_in_bar * ticks_per_step
                velocity = int(np.clip(vel_float * 127, 1, 127))
                events.append(DrumEvent(
                    time_ticks=t,
                    pitch=pitch,
                    velocity=velocity,
                    duration_ticks=max(10, ticks_per_step // 2),
                ))
        return sorted(events, key=lambda e: e.time_ticks)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE PERCUSIÓN DE MIDIS MIXTOS
# ══════════════════════════════════════════════════════════════════════════════

def extract_percussion_track(mid: mido.MidiFile) -> list[tuple]:
    """
    Extrae todos los eventos note_on del canal 10 (percusión).
    Devuelve lista de (time_ticks_abs, pitch, velocity).
    """
    events = []
    ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "note_on" and msg.channel == PERCUSSION_CHANNEL:
                if msg.velocity > 0:
                    events.append((abs_tick, msg.note, msg.velocity))

    return sorted(events, key=lambda x: x[0])


def get_tempo_and_tpb(mid: mido.MidiFile) -> tuple[float, int]:
    """Extrae tempo (BPM) y ticks_per_beat del MIDI."""
    tpb = mid.ticks_per_beat
    tempo_us = 500000  # 120 BPM por defecto
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
                break
    bpm = 60_000_000 / tempo_us
    return bpm, tpb


def events_to_grid(events: list[tuple], tpb: int,
                   resolution: int = 16, bars: int = 2) -> dict:
    """
    Convierte lista de (tick, pitch, vel) a grids por instrumento.
    El grid tiene resolución * bars pasos totales.
    """
    ticks_per_bar = tpb * 4
    ticks_per_step = ticks_per_bar // resolution
    total_steps = resolution * bars

    grids = {}
    for tick, pitch, vel in events:
        instr = GM_INV.get(pitch)
        if instr is None:
            continue
        # Posición en el grid global
        step = round(tick / ticks_per_step) % total_steps
        if instr not in grids:
            grids[instr] = [0.0] * total_steps
        # Tomar la velocity máxima si hay colisión
        v_norm = vel / 127.0
        if v_norm > grids[instr][step]:
            grids[instr][step] = v_norm

    return grids


def detect_repeat_unit(grids: dict, resolution: int) -> int:
    """
    Detecta la unidad de repetición del patrón (1, 2 o 4 compases).
    Compara el primer compás contra el segundo, etc.
    """
    total_steps = sum(len(v) for v in grids.values())
    if not grids:
        return 1

    # Usar el instrumento con más pasos
    ref_instr = max(grids, key=lambda k: len(grids[k]))
    arr = np.array(grids[ref_instr])
    n = len(arr)
    bars = n // resolution

    if bars <= 1:
        return 1

    # Comparar compás 0 con compás 1
    for unit in [1, 2]:
        if bars < unit * 2:
            break
        block_a = arr[:resolution * unit]
        block_b = arr[resolution * unit: resolution * unit * 2]
        if len(block_b) == len(block_a):
            diff = np.mean(np.abs(block_a - block_b))
            if diff < 0.15:
                return unit

    return bars  # sin repetición clara


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZACIÓN Y MÉTRICAS
# ══════════════════════════════════════════════════════════════════════════════

def compute_syncope_level(grid: list[float], resolution: int = 16) -> float:
    """
    Calcula el nivel de síncopa: proporción de golpes en posiciones débiles.
    0 = todos en tiempos fuertes, 1 = todos en posiciones débiles.
    """
    weights = METRIC_WEIGHT_16 if resolution == 16 else METRIC_WEIGHT_32
    # Repetir weights para cubrir múltiples compases
    n = len(grid)
    w = (weights * (n // len(weights) + 1))[:n]

    hits = [i for i, v in enumerate(grid) if v > 0]
    if not hits:
        return 0.0

    # Síncopa = golpes en posiciones de peso bajo
    synco_score = sum(1.0 - w[i] for i in hits) / len(hits)
    return float(np.clip(synco_score, 0, 1))


def compute_pattern_vector(pattern: RhythmPattern) -> list[float]:
    """
    Vector de 24 dimensiones para clustering/búsqueda de similitud.
    """
    resolution = pattern.resolution
    weights = METRIC_WEIGHT_16 if resolution == 16 else METRIC_WEIGHT_32

    # Grids por función
    pulse_grid = np.zeros(resolution)
    backbeat_grid = np.zeros(resolution)
    subdiv_grid = np.zeros(resolution)

    for instr, arr in pattern.grid.items():
        func = FUNC.get(instr, "color")
        # Usar solo el primer compás para el vector
        g = np.array(arr[:resolution], dtype=float)
        if func == "pulse":
            pulse_grid = np.maximum(pulse_grid, g)
        elif func == "backbeat":
            backbeat_grid = np.maximum(backbeat_grid, g)
        elif func == "subdivision":
            subdiv_grid = np.maximum(subdiv_grid, g)

    # Histograma de acentos métricos (8 bins)
    combined = pulse_grid + backbeat_grid * 0.7 + subdiv_grid * 0.4
    hist_bins = 8
    hist = np.array([
        float(np.mean(combined[i * resolution // hist_bins:
                                (i+1) * resolution // hist_bins]))
        for i in range(hist_bins)
    ])

    # Métricas escalares
    all_hits = np.zeros(resolution)
    for arr in pattern.grid.values():
        all_hits = np.maximum(all_hits, np.array(arr[:resolution]))

    density = float(np.mean(all_hits > 0))
    syncope = compute_syncope_level(list(all_hits), resolution)
    kick_on_1 = float(pulse_grid[0] > 0) if len(pulse_grid) > 0 else 0.0
    snare_backbeat = float(backbeat_grid[resolution//4] > 0 if resolution >= 4 else 0)
    energy = float(np.mean(all_hits))
    subdiv_density = float(np.mean(subdiv_grid > 0))

    # Ratio de swing (diferencia entre posiciones pares e impares)
    if len(subdiv_grid) >= 4:
        even = float(np.mean(subdiv_grid[::2]))
        odd  = float(np.mean(subdiv_grid[1::2]))
        swing_ratio = odd / (even + 1e-6)
    else:
        swing_ratio = 0.0

    vec = list(hist) + [
        density,
        syncope,
        kick_on_1,
        snare_backbeat,
        energy,
        subdiv_density,
        swing_ratio,
        float(len(pattern.grid)),   # número de instrumentos
    ]

    return vec


def cosine_similarity(a: list, b: list) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: LEARN
# ══════════════════════════════════════════════════════════════════════════════

def cmd_learn(args):
    """
    Aprende grooves desde una carpeta de MIDIs mixtos.
    Extrae la pista de percusión, vectoriza y agrupa por clustering.
    """
    folder = Path(args.folder)
    mid_files = list(folder.glob("**/*.mid")) + list(folder.glob("**/*.midi"))

    if not mid_files:
        print(f"[learn] No se encontraron MIDIs en {folder}")
        sys.exit(1)

    print(f"[learn] Procesando {len(mid_files)} MIDIs...")

    patterns = []
    skipped = 0

    for path in mid_files:
        try:
            mid = mido.MidiFile(str(path))
            bpm, tpb = get_tempo_and_tpb(mid)
            perc_events = extract_percussion_track(mid)

            if len(perc_events) < 8:
                skipped += 1
                continue

            # Extraer grid de 2 compases
            grids = events_to_grid(perc_events, tpb, resolution=16, bars=2)

            # Detectar unidad de repetición y recortar
            unit = detect_repeat_unit(grids, resolution=16)
            steps = 16 * unit
            grids_trim = {k: v[:steps] + [0.0] * max(0, steps - len(v))
                          for k, v in grids.items()}

            p = RhythmPattern(
                grid=grids_trim,
                resolution=16,
                bars=unit,
                tempo=bpm,
            )

            # Calcular síncopa y densidad antes del vector
            all_flat = [v for arr in grids_trim.values() for v in arr]
            total_steps = 16 * unit
            p.density = sum(1 for v in all_flat if v > 0) / (total_steps * len(grids_trim) + 1e-6)

            # Síncopa usando el grid combinado del primer compás
            combined_first = [0.0] * 16
            for arr in grids_trim.values():
                for i in range(min(16, len(arr))):
                    combined_first[i] = max(combined_first[i], arr[i])
            p.syncope_level = compute_syncope_level(combined_first, 16)

            p.feature_vector = compute_pattern_vector(p)
            p.tempo = bpm

            patterns.append({
                "source": str(path.name),
                "bars": unit,
                "tempo": round(bpm, 1),
                "density": round(p.density, 3),
                "syncope_level": round(p.syncope_level, 3),
                "grid": grids_trim,
                "feature_vector": p.feature_vector,
            })

        except Exception as e:
            skipped += 1
            if args.verbose:
                print(f"  [!] {path.name}: {e}")

    if not patterns:
        print("[learn] No se extrajeron patrones con percusión suficiente.")
        sys.exit(1)

    print(f"[learn] {len(patterns)} patrones extraídos, {skipped} omitidos.")

    # Clustering con KMeans si scikit-learn disponible
    n_clusters = min(args.clusters, len(patterns))
    cluster_labels = list(range(len(patterns)))  # default: cada uno es su cluster

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        X = np.array([p["feature_vector"] for p in patterns])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        cluster_labels = labels.tolist()

        # Calcular centroide de cada cluster (patrón más representativo)
        centroids = []
        for c in range(n_clusters):
            members = [i for i, l in enumerate(labels) if l == c]
            if not members:
                continue
            # Patrón con menor distancia media al resto del cluster
            vecs = np.array([patterns[i]["feature_vector"] for i in members])
            dists = np.mean(np.abs(vecs[:, None] - vecs[None, :]), axis=(1, 2))
            best = members[np.argmin(dists)]

            # Estadísticas del cluster
            densities = [patterns[i]["density"] for i in members]
            syncopes = [patterns[i]["syncope_level"] for i in members]
            centroids.append({
                "cluster_id": c,
                "label": f"cluster_{c:02d}",
                "size": len(members),
                "representative": patterns[best]["source"],
                "avg_density": round(float(np.mean(densities)), 3),
                "avg_syncope": round(float(np.mean(syncopes)), 3),
                "grid": patterns[best]["grid"],
                "feature_vector": patterns[best]["feature_vector"],
            })

        print(f"[learn] {n_clusters} clusters generados.")

    except ImportError:
        print("[learn] scikit-learn no disponible — guardando sin clustering.")
        centroids = [
            {
                "cluster_id": i,
                "label": f"groove_{i:03d}",
                "size": 1,
                "representative": p["source"],
                "avg_density": p["density"],
                "avg_syncope": p["syncope_level"],
                "grid": p["grid"],
                "feature_vector": p["feature_vector"],
            }
            for i, p in enumerate(patterns)
        ]

    library = {
        "version": "0.1",
        "total_patterns": len(patterns),
        "clusters": centroids,
        "all_patterns": patterns,
    }

    out = args.output or "drum_library.json"
    with open(out, "w") as f:
        json.dump(library, f, indent=2)

    print(f"[learn] Biblioteca guardada en: {out}")
    print(f"        {len(centroids)} grooves de referencia")
    for c in centroids[:8]:
        print(f"        · {c['label']:20s}  density={c['avg_density']:.2f}  "
              f"syncope={c['avg_syncope']:.2f}  ({c['size']} ejemplos)")
    if len(centroids) > 8:
        print(f"        … y {len(centroids)-8} más")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: ANALYZE
# ══════════════════════════════════════════════════════════════════════════════

def cmd_analyze(args):
    """Analiza la percusión de un MIDI y muestra un informe."""
    mid = mido.MidiFile(args.input)
    bpm, tpb = get_tempo_and_tpb(mid)
    events = extract_percussion_track(mid)

    if not events:
        print(f"[analyze] {args.input}: no contiene pista de percusión (canal 10).")
        return

    total_bars = int(math.ceil(events[-1][0] / (tpb * 4))) + 1
    grids = events_to_grid(events, tpb, resolution=16, bars=total_bars)

    # Combinar en un grid global (primer compás)
    combined = [0.0] * 16
    for arr in grids.values():
        for i in range(min(16, len(arr))):
            combined[i] = max(combined[i], arr[i])

    syncope = compute_syncope_level(combined)
    total_hits = sum(1 for e in events if e[2] > 0)
    density = total_hits / (total_bars * 16)

    # Instrumentos presentes
    instrs_present = set(GM_INV.get(e[1], f"note_{e[1]}") for e in events)
    funcs_present = set(FUNC.get(i, "color") for i in instrs_present)

    # Detección de fills: clusters de toms
    fill_pitches = {GM["tom_lo"], GM["tom_mid"], GM["tom_hi"]}
    fill_events = [e for e in events if e[1] in fill_pitches]
    n_fills = 0
    if fill_events:
        tpb4 = tpb * 4  # ticks por compás
        bars_with_toms = set(e[0] // tpb4 for e in fill_events)
        n_fills = len(bars_with_toms)

    # Polirritmos: buscar capas en 3 o 5
    polyrhythm_hints = detect_polyrhythm_hints(events, tpb)

    # Tipo de feel (swing ratio)
    hihat_events = [e for e in events if e[1] in (GM["hihat_c"], GM["hihat_p"])]
    swing_ratio = 1.0
    if len(hihat_events) >= 4:
        tpb16 = tpb // 4
        positions = [e[0] % (tpb * 4) for e in hihat_events]
        even_pos = [p for p in positions if (p // tpb16) % 2 == 0]
        odd_pos  = [p for p in positions if (p // tpb16) % 2 == 1]
        swing_ratio = len(odd_pos) / (len(even_pos) + 1e-6)

    feel = "straight" if swing_ratio < 0.6 else ("swing" if swing_ratio > 1.2 else "groove")

    # Energía media (velocity media normalizada)
    avg_vel = np.mean([e[2] for e in events]) / 127.0

    print(f"\n{'═'*60}")
    print(f"  DRUMMER · Análisis de percusión")
    print(f"{'═'*60}")
    print(f"  Archivo:       {args.input}")
    print(f"  Tempo:         {bpm:.1f} BPM")
    print(f"  Compases:      {total_bars}")
    print(f"  Golpes totales:{total_hits}")
    print(f"  Densidad:      {density:.3f}  golpes/subdivisión")
    print(f"  Síncopa:       {syncope:.2f}  (0=cuadrado, 1=máxima)")
    print(f"  Feel:          {feel}  (swing_ratio={swing_ratio:.2f})")
    print(f"  Energía media: {avg_vel:.2f}")
    print(f"  Fills:         {n_fills} compases con toms")
    print(f"\n  Instrumentos presentes:")
    for instr in sorted(instrs_present):
        pitch = GM.get(instr, "?")
        func = FUNC.get(instr, "color")
        print(f"    · {instr:12s} (MIDI {pitch})  [{func}]")

    if polyrhythm_hints:
        print(f"\n  Posibles polirritmos detectados:")
        for hint in polyrhythm_hints:
            print(f"    · {hint}")

    # Visualización ASCII del grid (primer compás)
    print(f"\n  Grid (primer compás, 16 subdivisiones):")
    print(f"  {'Paso':6s}  " + " ".join(f"{i+1:2d}" for i in range(16)))
    for instr in sorted(grids.keys()):
        row = grids[instr][:16]
        cells = ""
        for v in row:
            if v > 0.7:    cells += " ■ "
            elif v > 0.3:  cells += " ▪ "
            elif v > 0:    cells += " · "
            else:          cells += " . "
        print(f"  {instr:10s}  {cells}")

    print(f"{'═'*60}\n")

    if args.export:
        result = {
            "tempo": bpm,
            "bars": total_bars,
            "density": density,
            "syncope_level": syncope,
            "feel": feel,
            "energy": float(avg_vel),
            "instruments": list(instrs_present),
            "functions": list(funcs_present),
            "n_fills": n_fills,
            "polyrhythm_hints": polyrhythm_hints,
            "grid_first_bar": {k: v[:16] for k, v in grids.items()},
        }
        out = args.export
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[analyze] Informe exportado: {out}")


def detect_polyrhythm_hints(events: list, tpb: int) -> list[str]:
    """
    Intenta detectar capas que sugieran polirritmos (3, 5, 7 sobre 4).
    Muy heurístico: busca instrumentos de color/clave con periodicidad N.
    """
    hints = []
    color_pitches = {GM["maracas"], GM["clave"]}
    color_events = [e for e in events if e[1] in color_pitches]

    if len(color_events) < 3:
        return hints

    # Medir intervalos entre golpes consecutivos
    ticks = sorted(e[0] for e in color_events)
    intervals = [ticks[i+1] - ticks[i] for i in range(len(ticks)-1)]
    if not intervals:
        return hints

    avg_interval = np.mean(intervals)
    tpb4 = tpb * 4  # ticks por compás

    # ¿El intervalo medio divide el compás en 3, 5 o 7 partes?
    for n in [3, 5, 7]:
        expected = tpb4 / n
        if abs(avg_interval - expected) / (expected + 1) < 0.15:
            hints.append(f"{n} contra 4 (posible clave/{n})")

    return hints


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: ADD (añadir percusión a obra sin ella)
# ══════════════════════════════════════════════════════════════════════════════

def cmd_add(args):
    """Añade una pista de percusión a un MIDI que no la tiene."""
    mid = mido.MidiFile(args.input)
    bpm, tpb = get_tempo_and_tpb(mid)

    # Verificar si ya tiene percusión
    existing = extract_percussion_track(mid)
    if existing and not args.force:
        print(f"[add] El MIDI ya tiene percusión ({len(existing)} eventos).")
        print("      Usa --force para sobreescribirla o 'transform' para modificarla.")
        sys.exit(1)

    # Calcular número de compases — acumular ticks absolutos por pista
    max_tick = 0
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if abs_tick > max_tick:
                max_tick = abs_tick
    if max_tick == 0:
        max_tick = tpb * 4 * 8   # fallback: 8 compases
    total_bars = int(math.ceil(max_tick / (tpb * 4))) + 1

    # Cargar fingerprint si existe
    tension_curve = None
    sections = None
    fp_path = args.fingerprint or (args.input.replace(".mid", ".fingerprint.json"))
    if os.path.exists(fp_path):
        try:
            with open(fp_path) as f:
                fp = json.load(f)
            tension_curve = fp.get("tension_curve") or fp.get("emotional", {}).get("tension_curve")
            sections = fp.get("sections")
            print(f"[add] Fingerprint cargado: {fp_path}")
        except Exception:
            pass

    # Cargar curvas de tensión si se proporcionan
    if args.tension_curve and os.path.exists(args.tension_curve):
        try:
            with open(args.tension_curve) as f:
                curves = json.load(f)
            tension_curve = curves.get("tension", tension_curve)
            print(f"[add] Curva de tensión cargada: {args.tension_curve}")
        except Exception:
            pass

    # Cargar biblioteca de grooves
    library = None
    if args.library and os.path.exists(args.library):
        with open(args.library) as f:
            library = json.load(f)
        print(f"[add] Biblioteca cargada: {len(library.get('clusters', []))} grooves")

    # Generar la pista de percusión compás a compás
    print(f"[add] Generando percusión para {total_bars} compases a {bpm:.1f} BPM...")

    all_events = []

    # Si hay secciones, generar groove diferente por sección
    if sections:
        for sec in sections:
            sec_start = sec.get("bar_start", 0)
            sec_end   = sec.get("bar_end", total_bars)
            sec_bars  = sec_end - sec_start
            tension   = sec.get("tension_mean", 0.5)
            label     = sec.get("label", "A")

            pattern = choose_groove(tension, library, bpm)
            sec_events = generate_groove_events(
                pattern, tpb, sec_bars, sec_start,
                tension=tension,
                add_fill_at_end=(label != sections[-1].get("label")),
            )
            all_events.extend(sec_events)
    else:
        # Sin secciones: dividir en 3 partes (intro, desarrollo, clímax)
        intro_bars = max(2, total_bars // 4)
        dev_bars   = total_bars - intro_bars * 2
        climax_bars = intro_bars

        # Tensión por sección si hay curva
        def avg_tension(start, end):
            if tension_curve and len(tension_curve) > 0:
                idxs = np.linspace(0, len(tension_curve)-1, total_bars)
                vals = [tension_curve[int(i)] for i in idxs[start:end]]
                return float(np.mean(vals)) if vals else 0.5
            return 0.5

        parts = [
            (0,            intro_bars,           avg_tension(0, intro_bars),           True),
            (intro_bars,   intro_bars + dev_bars, avg_tension(intro_bars, intro_bars+dev_bars), True),
            (intro_bars + dev_bars, total_bars,  avg_tension(intro_bars+dev_bars, total_bars), False),
        ]

        for start, end, tension, add_fill in parts:
            n = end - start
            if n <= 0:
                continue
            pattern = choose_groove(tension, library, bpm)
            evts = generate_groove_events(
                pattern, tpb, n, start,
                tension=tension,
                add_fill_at_end=add_fill,
            )
            all_events.extend(evts)

    # Humanizar
    if not args.no_humanize:
        all_events = humanize_events(all_events, tpb, strength=0.5)

    # Escribir MIDI de salida
    out_path = args.output or args.input.replace(".mid", "_drums.mid")
    write_midi_with_drums(mid, all_events, tpb, out_path)
    print(f"[add] MIDI con percusión guardado: {out_path}")
    print(f"      {len(all_events)} eventos de percusión generados")


def choose_groove(tension: float, library: Optional[dict], bpm: float) -> RhythmPattern:
    """
    Elige el groove más apropiado de la biblioteca según la tensión.
    Si no hay biblioteca, usa grooves hardcodeados de semilla.
    """
    if library and library.get("clusters"):
        # Buscar por similitud: tensión alta → alta síncopa y densidad
        target_syncope = tension * 0.8
        target_density = 0.2 + tension * 0.4

        best = None
        best_score = float("inf")
        for c in library["clusters"]:
            d = abs(c.get("avg_density", 0.3) - target_density)
            s = abs(c.get("avg_syncope", 0.3) - target_syncope)
            score = d + s
            if score < best_score:
                best_score = score
                best = c

        if best:
            p = RhythmPattern(
                grid=best["grid"],
                resolution=16,
                bars=1,
                tempo=bpm,
                syncope_level=best.get("avg_syncope", tension * 0.5),
                density=best.get("avg_density", 0.3),
                groove_family=best.get("label", "learned"),
            )
            return p

    # Grooves de semilla hardcodeados
    return build_seed_groove(tension, bpm)


def build_seed_groove(tension: float, bpm: float) -> RhythmPattern:
    """
    Construye un groove desde reglas musicales básicas.
    Tensión controla la síncopa y densidad del bombo.
    """
    # Bombo: tiempo 1 y 3 siempre; extras según tensión
    kick = [0.0] * 16
    kick[0]  = 0.9   # tiempo 1
    kick[8]  = 0.85  # tiempo 3
    if tension > 0.4:
        kick[6]  = 0.7  # synco en el "y" del 2
    if tension > 0.65:
        kick[10] = 0.65  # synco en el "y" del 3
    if tension > 0.8:
        kick[14] = 0.6   # synco en el "y" del 4

    # Caja: backbeat en 2 y 4 siempre
    snare = [0.0] * 16
    snare[4]  = 0.85  # tiempo 2
    snare[12] = 0.85  # tiempo 4
    if tension > 0.5:
        snare[10] = 0.35  # ghost note

    # Hi-hat: subdivisión básica, más densa con más tensión
    hihat_c = [0.0] * 16
    step = 2 if tension > 0.5 else 4  # 8vos vs 4tos
    for i in range(0, 16, step):
        hihat_c[i] = 0.55 if i % 4 != 0 else 0.65

    # Hi-hat abierto en el "y" del 4 si hay tensión
    hihat_o = [0.0] * 16
    if tension > 0.3:
        hihat_o[14] = 0.6

    grid = {
        "kick":    kick,
        "snare":   snare,
        "hihat_c": hihat_c,
    }
    if tension > 0.3:
        grid["hihat_o"] = hihat_o

    syncope = compute_syncope_level(kick)
    density = sum(1 for v in kick + snare + hihat_c if v > 0) / 48.0

    return RhythmPattern(
        grid=grid,
        resolution=16,
        bars=1,
        tempo=bpm,
        syncope_level=syncope,
        density=density,
        groove_family="seed",
        energy=tension,
    )


def generate_groove_events(pattern: RhythmPattern, tpb: int,
                            n_bars: int, start_bar: int,
                            tension: float = 0.5,
                            add_fill_at_end: bool = False) -> list[DrumEvent]:
    """Genera eventos repitiendo el patrón n_bars veces."""
    events = []
    ticks_per_bar = tpb * 4
    ticks_per_step = ticks_per_bar // pattern.resolution

    pat_bars = pattern.bars
    for bar_offset in range(n_bars):
        # Si el patrón es de 1 compás, repetir; si es de 2, alternar
        loop_bar = bar_offset % pat_bars
        abs_bar = start_bar + bar_offset

        # ¿Variar ligeramente el groove? (cada 4 compases)
        variation_seed = bar_offset // 4
        rng = random.Random(abs_bar * 31 + int(tension * 100))

        for instr, vel_array in pattern.grid.items():
            pitch = GM.get(instr)
            if pitch is None:
                continue

            # Slice del compás correspondiente
            bar_steps = vel_array[loop_bar * 16:(loop_bar + 1) * 16]
            if not bar_steps:
                bar_steps = vel_array[:16]

            for step, vel_f in enumerate(bar_steps):
                if vel_f <= 0.0:
                    continue

                # Pequeña variación de velocity compás a compás
                var = rng.uniform(-0.05, 0.05)
                vel_final = float(np.clip(vel_f + var, 0.05, 1.0))

                t = abs_bar * ticks_per_bar + step * ticks_per_step
                events.append(DrumEvent(
                    time_ticks=t,
                    pitch=pitch,
                    velocity=int(vel_final * 127),
                    duration_ticks=max(10, ticks_per_step // 2),
                ))

        # Crash en el inicio de secciones
        if bar_offset == 0:
            t = abs_bar * ticks_per_bar
            events.append(DrumEvent(
                time_ticks=t,
                pitch=GM["crash"],
                velocity=90,
                duration_ticks=tpb,
            ))

    # Fill al final si se pide
    if add_fill_at_end and n_bars > 0:
        fill_bar = start_bar + n_bars - 1
        fill = generate_fill(tpb, fill_bar, tension)
        events.extend(fill)

    return events


def generate_fill(tpb: int, bar: int, tension: float) -> list[DrumEvent]:
    """Genera un fill de batería de 1 compás."""
    events = []
    ticks_per_bar = tpb * 4
    ticks_per_step = ticks_per_bar // 16
    bar_start = bar * ticks_per_bar

    # Fill: secuencia descendente de toms + redoble de caja
    if tension > 0.6:
        # Fill denso: 8 golpes en el último compás
        seq = [GM["tom_hi"]] * 2 + [GM["tom_mid"]] * 2 + [GM["tom_lo"]] * 2 + [GM["snare"]] * 2
        steps = [8, 10, 11, 12, 13, 14, 15, 15]  # en los últimos 8 pasos
    else:
        # Fill suave: 4 toms
        seq = [GM["tom_hi"], GM["tom_mid"], GM["tom_lo"], GM["snare"]]
        steps = [12, 13, 14, 15]

    for pitch, step in zip(seq, steps):
        t = bar_start + step * ticks_per_step
        vel = int(np.clip(75 + tension * 30, 60, 110))
        events.append(DrumEvent(
            time_ticks=t,
            pitch=pitch,
            velocity=vel,
            duration_ticks=max(10, ticks_per_step // 2),
        ))

    return events


# ══════════════════════════════════════════════════════════════════════════════
#  HUMANIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# Sesgos de micro-timing por instrumento (mean, std) en ticks (a 480 tpb)
TIMING_BIAS = {
    "kick":    (-4, 3),   # ligeramente antes (push)
    "snare":   (+6, 4),   # ligeramente después (lay back)
    "hihat_c": ( 0, 2),   # casi perfecto
    "hihat_o": (+2, 3),
    "crash":   (-2, 2),
    "ride":    ( 1, 2),
    "tom_lo":  ( 0, 3),
    "tom_mid": ( 0, 3),
    "tom_hi":  ( 0, 3),
}

def humanize_events(events: list[DrumEvent], tpb: int,
                     strength: float = 0.6) -> list[DrumEvent]:
    """
    Añade micro-timing y variación de velocity.
    strength: 0=mecánico, 1=máxima humanización.
    """
    scale = tpb / 480.0  # escalar tiempos según tpb real
    rng = random.Random(42)
    result = []

    for e in events:
        instr = GM_INV.get(e.pitch, "snare")
        mean_t, std_t = TIMING_BIAS.get(instr, (0, 3))

        # Micro-timing
        offset_ticks = rng.gauss(mean_t * scale, std_t * scale * strength)
        new_time = max(0, e.time_ticks + int(offset_ticks))

        # Variación de velocity
        vel_var = rng.gauss(0, 4 * strength)
        new_vel = int(np.clip(e.velocity + vel_var, 10, 127))

        result.append(DrumEvent(
            time_ticks=new_time,
            pitch=e.pitch,
            velocity=new_vel,
            duration_ticks=e.duration_ticks,
            timing_offset=int(offset_ticks),
        ))

    return sorted(result, key=lambda e: e.time_ticks)


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: TRANSFORM
# ══════════════════════════════════════════════════════════════════════════════

def cmd_transform(args):
    """Transforma la percusión existente de un MIDI."""
    mid = mido.MidiFile(args.input)
    bpm, tpb = get_tempo_and_tpb(mid)
    events = extract_percussion_track(mid)

    if not events:
        print(f"[transform] El MIDI no tiene percusión en canal 10.")
        sys.exit(1)

    total_bars = int(math.ceil(events[-1][0] / (tpb * 4))) + 1
    print(f"[transform] {len(events)} eventos, {total_bars} compases, {bpm:.1f} BPM")

    grids = events_to_grid(events, tpb, resolution=16, bars=total_bars)

    # ── Cargar curvas de tension_designer ──────────────────────────────────
    curves = {}
    if args.tension_curve and os.path.exists(args.tension_curve):
        with open(args.tension_curve) as f:
            raw = json.load(f)
        # Acepta tanto {"tension": [...]} como {"curves": {"tension": [...]}}
        curves = raw.get("curves", raw)
        print(f"[transform] Curvas cargadas: {list(curves.keys())}")

    ci = CurveInterpolator(curves, total_bars, tempo=bpm)

    # Detectar groove_family del args o usar generic
    groove_family = getattr(args, "groove_family", "generic")

    transformations_applied = []

    # ── SÍNCOPA ────────────────────────────────────────────────────────────
    if args.syncopate is not None:
        grids = transform_syncopate(grids, float(args.syncopate), total_bars)
        transformations_applied.append(f"syncopate={args.syncopate:.2f}")
        print(f"[transform] Síncopa estática: {args.syncopate:.2f}")
    elif curves.get("tension"):
        grids = transform_syncopate_dynamic(grids, curves["tension"], total_bars, tpb)
        transformations_applied.append("syncopate_dynamic")
        print("[transform] Síncopa dinámica desde curva tension")

    # ── DENSIDAD DINÁMICA (curva activity) ────────────────────────────────
    if curves.get("activity") and args.thin is None and args.thicken is None:
        grids = apply_density_dynamic(grids, ci, total_bars)
        transformations_applied.append("density_dynamic")
        print("[transform] Densidad dinámica desde curva activity")

    # ── POLIRRITMO ────────────────────────────────────────────────────────
    if args.polyrhythm:
        grids = transform_add_polyrhythm(grids, int(args.polyrhythm), total_bars)
        transformations_applied.append(f"polyrhythm={args.polyrhythm}")
        print(f"[transform] Capa polirrítmica {args.polyrhythm}:4")
    elif curves.get("harmony"):
        # Activar polirritmo 3:4 cuando complexity > 0.7
        complex_bars = [b for b in range(total_bars)
                        if ci.get("harmony", b) > 0.70]
        if complex_bars:
            # Sólo en esos compases
            poly_grids = {k: [0.0] * (16 * total_bars) for k in grids}
            for bar in complex_bars:
                bar_grids = {k: v[bar*16:(bar+1)*16] for k, v in grids.items()}
                poly_bar  = transform_add_polyrhythm(bar_grids, 3, 1)
                for k, arr in poly_bar.items():
                    grids.setdefault(k, [0.0] * (16 * total_bars))
                    grids[k][bar*16:(bar+1)*16] = arr[:16]
            transformations_applied.append(f"polyrhythm_dynamic({len(complex_bars)}bars)")
            print(f"[transform] Polirritmo dinámico en {len(complex_bars)} compases de alta armonía")

    # ── PHASE SHIFT ───────────────────────────────────────────────────────
    if args.phase_shift is not None:
        grids = transform_phase_shift(grids, int(args.phase_shift), total_bars)
        transformations_applied.append(f"phase_shift={args.phase_shift}")
        print(f"[transform] Phase shift: {args.phase_shift} paso(s)/compás")

    # ── THIN / THICKEN ────────────────────────────────────────────────────
    if args.thin is not None:
        grids = transform_thin(grids, float(args.thin))
        transformations_applied.append(f"thin={args.thin:.2f}")
        print(f"[transform] Thin: {args.thin*100:.0f}%")

    if args.thicken is not None:
        grids = transform_thicken(grids, float(args.thicken))
        transformations_applied.append(f"thicken={args.thicken:.2f}")
        print(f"[transform] Thicken: ghost notes {args.thicken*100:.0f}%")

    # ── TARGET EMOCIONAL ─────────────────────────────────────────────────
    if args.emotion:
        grids = transform_emotional_target(grids, args.emotion, total_bars)
        transformations_applied.append(f"emotion='{args.emotion}'")
        print(f"[transform] Target emocional: '{args.emotion}'")

    # ── Convertir a eventos ───────────────────────────────────────────────
    new_events = grids_to_events(grids, tpb)

    # ── VELOCITY dinámica (curva register) ────────────────────────────────
    if curves.get("register"):
        new_events = apply_velocity_dynamic(new_events, ci, tpb)
        transformations_applied.append("velocity_dynamic")
        print("[transform] Velocity dinámica desde curva register")

    # ── SWING ─────────────────────────────────────────────────────────────
    if args.swing is not None:
        # Swing estático
        ratio = 0.50 + float(args.swing) * 0.25   # 0→0.50, 1→0.75
        new_events = apply_swing_to_events(new_events, tpb, ratio, groove_family)
        transformations_applied.append(f"swing={args.swing:.2f}(ratio={ratio:.2f})")
        print(f"[transform] Swing estático: ratio={ratio:.2f}")
    elif curves.get("swing"):
        new_events = apply_swing_dynamic(new_events, tpb, ci, groove_family)
        transformations_applied.append("swing_dynamic")
        print("[transform] Swing dinámico desde curva swing")

    # ── HUMANIZAR ────────────────────────────────────────────────────────
    if args.humanize is not None:
        new_events = humanize_events(new_events, tpb, float(args.humanize))
        transformations_applied.append(f"humanize={args.humanize:.2f}")
        print(f"[transform] Humanización: {args.humanize:.2f}")

    # ── Build-up fills automáticos desde curva activity ───────────────────
    if curves.get("activity"):
        buildup_bars = ci.build_up_bars(threshold=0.30)
        for bar in buildup_bars:
            fill_evts = generate_fill(tpb, bar,
                                       ci.get("tension", bar))
            new_events.extend(fill_evts)
        if buildup_bars:
            transformations_applied.append(f"auto_fills({len(buildup_bars)})")
            print(f"[transform] Fills automáticos en build-ups: compases {buildup_bars}")
        new_events.sort(key=lambda e: e.time_ticks)

    # ── Escribir MIDI ─────────────────────────────────────────────────────
    out_path = args.output or args.input.replace(".mid", "_transformed.mid")
    write_midi_with_drums(mid, new_events, tpb, out_path, replace_percussion=True)

    print(f"\n[transform] ✓ Guardado: {out_path}")
    print(f"            Transformaciones: {', '.join(transformations_applied)}")
    print(f"            Eventos: {len(events)} → {len(new_events)}")




# ══════════════════════════════════════════════════════════════════════════════
#  MODO: EXTRACT
# ══════════════════════════════════════════════════════════════════════════════

def cmd_extract(args):
    """
    Extrae el groove de un MIDI concreto y lo añade a la biblioteca.
    Separa el patrón base de los fills detectando compases que se
    desvían significativamente del patrón más frecuente.
    """
    mid = mido.MidiFile(args.input)
    bpm, tpb = get_tempo_and_tpb(mid)
    events = extract_percussion_track(mid)

    if not events:
        print(f"[extract] {args.input}: no tiene canal de percusión.")
        sys.exit(1)

    total_bars = int(math.ceil(events[-1][0] / (tpb * 4))) + 1
    grids = events_to_grid(events, tpb, resolution=16, bars=total_bars)

    print(f"[extract] {total_bars} compases, {bpm:.1f} BPM, "
          f"{len(grids)} instrumentos: {list(grids.keys())}")

    # ── Detectar patrón base vs fills ────────────────────────────────────
    # Extraer el grid de cada compás como vector
    bar_vectors = []
    for bar in range(total_bars):
        vec = []
        for instr in sorted(grids.keys()):
            segment = grids[instr][bar*16:(bar+1)*16]
            vec.extend(segment if len(segment) == 16 else segment + [0.0]*(16-len(segment)))
        bar_vectors.append(np.array(vec))

    # Patrón base = compás más frecuente (centroide de todos excepto outliers)
    if len(bar_vectors) > 1:
        all_vecs = np.array(bar_vectors)
        centroid = np.mean(all_vecs, axis=0)
        distances = [float(np.mean(np.abs(v - centroid))) for v in bar_vectors]
        median_dist = float(np.median(distances))
        std_dist    = float(np.std(distances))

        # Umbral: compás es fill si se aleja > 1.5σ del centro
        fill_threshold = median_dist + 1.5 * std_dist
        fill_bars   = [i for i, d in enumerate(distances) if d > fill_threshold]
        base_bars   = [i for i, d in enumerate(distances) if d <= fill_threshold]

        # Repetitividad: qué proporción de compases son "base"
        repeatability = len(base_bars) / total_bars if total_bars > 0 else 0.0
    else:
        fill_bars   = []
        base_bars   = [0]
        repeatability = 1.0
        distances   = [0.0]

    # Avisar si el patrón no es claro
    if repeatability < 0.5:
        print(f"[extract] ⚠  Patrón poco repetitivo (repetitividad={repeatability:.0%}).")
        print(f"           El groove puede no ser representativo.")
        print(f"           Extrayendo igualmente...")

    # Construir el grid del patrón base (mediana de los compases base)
    base_grid = {}
    for instr in sorted(grids.keys()):
        cols = []
        for bar in (base_bars or [0]):
            seg = grids[instr][bar*16:(bar+1)*16]
            if len(seg) == 16:
                cols.append(seg)
        if cols:
            # Mediana por paso: más robusto que la media para datos binarios
            mat = np.array(cols)
            base_grid[instr] = list(np.median(mat, axis=0))

    # Grid de fills: todos los compases marcados como fill
    fill_grids = []
    for bar in fill_bars:
        fg = {}
        for instr in sorted(grids.keys()):
            seg = grids[instr][bar*16:(bar+1)*16]
            if len(seg) == 16:
                fg[instr] = seg
        if fg:
            fill_grids.append(fg)

    # ── Calcular métricas del groove ──────────────────────────────────────
    combined_base = [0.0] * 16
    for arr in base_grid.values():
        for i in range(min(16, len(arr))):
            combined_base[i] = max(combined_base[i], arr[i])

    syncope  = compute_syncope_level(combined_base, 16)
    density  = sum(1 for v in combined_base if v > 0) / 16.0
    avg_vel  = float(np.mean([v for v in combined_base if v > 0])) if any(v > 0 for v in combined_base) else 0.5

    # Estimar swing desde hi-hat
    hh = base_grid.get("hihat_c", base_grid.get("ride", []))
    swing_ratio_est = 0.50
    if hh:
        on_hits  = sum(1 for i, v in enumerate(hh) if v > 0 and i % 4 == 0)
        off_hits = sum(1 for i, v in enumerate(hh) if v > 0 and i % 4 == 2)
        if on_hits > 0:
            swing_ratio_est = 0.50 + min(off_hits / on_hits, 0.5) * 0.5

    # Inferir familia de groove desde el nombre si se indica
    label = args.name
    detected_family = "generic"
    for fam in GENRE_SWING_PROFILE:
        if fam in label.lower():
            detected_family = fam
            break

    # ── Crear entrada de biblioteca ────────────────────────────────────────
    p = RhythmPattern(
        grid=base_grid,
        resolution=16,
        bars=1,
        tempo=bpm,
        groove_family=detected_family,
        syncope_level=syncope,
        density=density,
        feel="swing" if swing_ratio_est > 0.55 else "straight",
        energy=avg_vel,
    )
    p.feature_vector = compute_pattern_vector(p)

    entry = {
        "cluster_id":   -1,           # -1 = extraído manualmente
        "label":        label,
        "source":       str(Path(args.input).name),
        "groove_family":detected_family,
        "size":         1,
        "representative": str(Path(args.input).name),
        "avg_density":  round(density, 3),
        "avg_syncope":  round(syncope, 3),
        "swing_ratio":  round(swing_ratio_est, 3),
        "energy":       round(avg_vel, 3),
        "repeatability":round(repeatability, 3),
        "tempo":        round(bpm, 1),
        "grid":         base_grid,
        "fills":        fill_grids[:4],   # hasta 4 fills guardados
        "feature_vector": p.feature_vector,
        "bars_analyzed": total_bars,
        "fill_bars_detected": len(fill_bars),
    }

    # ── Guardar en biblioteca ─────────────────────────────────────────────
    lib_path = args.library or "drum_library.json"

    if os.path.exists(lib_path):
        with open(lib_path) as f:
            library = json.load(f)
    else:
        library = {"version": "0.1", "total_patterns": 0, "clusters": [], "all_patterns": []}

    # Comprobar si ya existe un groove con el mismo nombre
    existing = [c for c in library["clusters"] if c.get("label") == label]
    if existing and not args.force:
        print(f"[extract] Ya existe un groove llamado '{label}' en la biblioteca.")
        print(f"          Usa --force para sobreescribirlo.")
        sys.exit(1)
    elif existing:
        library["clusters"] = [c for c in library["clusters"] if c.get("label") != label]

    library["clusters"].append(entry)
    library["total_patterns"] = len(library["clusters"])

    with open(lib_path, "w") as f:
        json.dump(library, f, indent=2)

    # ── Informe ───────────────────────────────────────────────────────────
    print(f"\n[extract] ✓ Groove '{label}' añadido a {lib_path}")
    print(f"  Familia:       {detected_family}")
    print(f"  Densidad:      {density:.3f}")
    print(f"  Síncopa:       {syncope:.3f}")
    print(f"  Swing ratio:   {swing_ratio_est:.2f}  ({'swing' if swing_ratio_est > 0.55 else 'straight'})")
    print(f"  Repetitividad: {repeatability:.0%}  ({len(base_bars)}/{total_bars} compases son patrón base)")
    print(f"  Fills:         {len(fill_bars)} compases detectados, {len(fill_grids)} guardados")
    print(f"  Instrumentos:  {list(base_grid.keys())}")

    if fill_bars:
        print(f"  Compases fill: {fill_bars}")

    # Mostrar grid ASCII del patrón base
    print(f"\n  Grid base (16 subdivisiones):")
    for instr in sorted(base_grid.keys()):
        row = base_grid[instr][:16]
        cells = "".join(" ■" if v > 0.6 else (" ▪" if v > 0.2 else " .") for v in row)
        print(f"  {instr:10s}  {cells}")
    print()

    # Exportar el patrón base como MIDI independiente si se pide
    if args.export_midi:
        out_mid = mido.MidiFile(ticks_per_beat=tpb, type=0)
        track = mido.MidiTrack()
        out_mid.tracks.append(track)
        evts = p.to_events(tpb, start_bar=0)
        evts = humanize_events(evts, tpb, strength=0.3)
        prev = 0
        msgs = []
        for e in sorted(evts, key=lambda x: x.time_ticks):
            msgs.append((e.time_ticks, "note_on", e.pitch, e.velocity))
            msgs.append((e.time_ticks + e.duration_ticks, "note_off", e.pitch, 0))
        for t, typ, pitch, vel in sorted(msgs, key=lambda x: x[0]):
            track.append(mido.Message(typ, channel=9, note=pitch,
                                       velocity=vel, time=t - prev))
            prev = t
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi_out = args.export_midi
        out_mid.save(midi_out)
        print(f"[extract] Patrón base exportado como MIDI: {midi_out}")


# ══════════════════════════════════════════════════════════════════════════════
#  MODO: MORPH
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_pattern_bars(p: RhythmPattern, target_bars: int) -> RhythmPattern:
    """
    Ajusta el número de compases de un RhythmPattern al objetivo
    repitiendo o recortando el grid.
    """
    if p.bars == target_bars:
        return p
    new_p  = p.copy()
    steps  = 16 * target_bars
    new_grid = {}
    for instr, arr in p.grid.items():
        src = list(arr)
        # Repetir hasta tener suficientes pasos, luego recortar
        repeated = (src * (steps // len(src) + 1))[:steps]
        new_grid[instr] = repeated
    new_p.grid = new_grid
    new_p.bars = target_bars
    return new_p


def _morph_layer_structure(grid_a: dict, grid_b: dict,
                            alpha: float, rng: random.Random) -> dict:
    """
    Interpola la capa estructural (kick + snare) entre A y B.
    alpha=0 → A puro, alpha=1 → B puro.

    Para cada instrumento de función pulse/backbeat:
    - En cada paso, si A tiene golpe y B no: mantener con probabilidad (1-alpha)
    - En cada paso, si B tiene golpe y A no: añadir con probabilidad alpha
    - Si ambos tienen golpe: mantener siempre
    """
    result = {}
    struct_funcs = {"pulse", "backbeat"}
    instrs = set(grid_a.keys()) | set(grid_b.keys())

    for instr in instrs:
        func = FUNC.get(instr, "color")
        if func not in struct_funcs:
            continue
        arr_a = grid_a.get(instr, [0.0] * 16)
        arr_b = grid_b.get(instr, [0.0] * 16)
        n = max(len(arr_a), len(arr_b))
        arr_a = (list(arr_a) + [0.0] * n)[:n]
        arr_b = (list(arr_b) + [0.0] * n)[:n]

        new_arr = []
        for va, vb in zip(arr_a, arr_b):
            if va > 0 and vb > 0:
                # Ambos: interpolar velocity
                new_arr.append(va * (1 - alpha) + vb * alpha)
            elif va > 0 and vb == 0:
                # Solo A: mantener con prob (1-alpha)
                new_arr.append(va if rng.random() > alpha else 0.0)
            elif va == 0 and vb > 0:
                # Solo B: añadir con prob alpha
                new_arr.append(vb if rng.random() < alpha else 0.0)
            else:
                new_arr.append(0.0)
        result[instr] = new_arr

    return result


def _morph_layer_subdivision(grid_a: dict, grid_b: dict,
                               alpha: float,
                               swing_a: float, swing_b: float) -> dict:
    """
    Interpola la capa de subdivisión (hihat, ride).
    Interpola tanto la presencia de golpes como el swing ratio.
    El swing interpolado se devuelve en los metadatos del resultado.
    """
    result = {}
    subdiv_funcs = {"subdivision", "accent"}
    instrs = set(grid_a.keys()) | set(grid_b.keys())

    for instr in instrs:
        func = FUNC.get(instr, "color")
        if func not in subdiv_funcs:
            continue
        arr_a = grid_a.get(instr, [0.0] * 16)
        arr_b = grid_b.get(instr, [0.0] * 16)
        n = max(len(arr_a), len(arr_b))
        arr_a = (list(arr_a) + [0.0] * n)[:n]
        arr_b = (list(arr_b) + [0.0] * n)[:n]

        # Interpolación directa de velocidades
        new_arr = [va * (1 - alpha) + vb * alpha for va, vb in zip(arr_a, arr_b)]
        result[instr] = new_arr

    # Swing interpolado (se almacena en metadatos, no en el grid)
    result["_swing_ratio"] = swing_a * (1 - alpha) + swing_b * alpha
    return result


def _morph_layer_color(grid_a: dict, grid_b: dict, alpha: float) -> dict:
    """
    Interpola la capa de color (toms, maracas, clave).
    Fade out de A y fade in de B.
    """
    result = {}
    color_funcs = {"color", "fill", "marker"}
    instrs = set(grid_a.keys()) | set(grid_b.keys())

    for instr in instrs:
        func = FUNC.get(instr, "color")
        if func not in color_funcs:
            continue
        arr_a = grid_a.get(instr, [])
        arr_b = grid_b.get(instr, [])
        n = max(len(arr_a) if arr_a else 16, len(arr_b) if arr_b else 16)
        arr_a = (list(arr_a) + [0.0] * n)[:n]
        arr_b = (list(arr_b) + [0.0] * n)[:n]

        # A se apaga, B se enciende — cruce suave
        new_arr = [va * (1 - alpha) + vb * alpha for va, vb in zip(arr_a, arr_b)]
        result[instr] = new_arr

    return result


def cmd_morph(args):
    """
    Genera N pasos de morphing entre dos grooves MIDI.
    Cada paso es un MIDI independiente con un nombre numerado.
    """
    # ── Cargar los dos MIDIs ──────────────────────────────────────────────
    mid_a = mido.MidiFile(args.input_a)
    mid_b = mido.MidiFile(args.input_b)
    bpm_a, tpb_a = get_tempo_and_tpb(mid_a)
    bpm_b, tpb_b = get_tempo_and_tpb(mid_b)

    tpb = max(tpb_a, tpb_b)   # usar la mayor resolución

    events_a = extract_percussion_track(mid_a)
    events_b = extract_percussion_track(mid_b)

    if not events_a:
        print(f"[morph] {args.input_a}: no tiene percusión.")
        sys.exit(1)
    if not events_b:
        print(f"[morph] {args.input_b}: no tiene percusión.")
        sys.exit(1)

    bars_a = int(math.ceil(events_a[-1][0] / (tpb_a * 4))) + 1
    bars_b = int(math.ceil(events_b[-1][0] / (tpb_b * 4))) + 1

    # ── Normalizar duración ───────────────────────────────────────────────
    target_bars = args.bars if args.bars else max(bars_a, bars_b)

    grids_a = events_to_grid(events_a, tpb_a, resolution=16, bars=bars_a)
    grids_b = events_to_grid(events_b, tpb_b, resolution=16, bars=bars_b)

    # Extender al número de compases objetivo repitiendo
    def extend_grids(grids: dict, src_bars: int, tgt_bars: int) -> dict:
        result = {}
        steps  = 16 * tgt_bars
        for instr, arr in grids.items():
            src = list(arr)
            result[instr] = (src * (steps // len(src) + 1))[:steps]
        return result

    grids_a = extend_grids(grids_a, bars_a, target_bars)
    grids_b = extend_grids(grids_b, bars_b, target_bars)

    # ── Estimar swing ratio de cada fuente ────────────────────────────────
    def estimate_swing(grids: dict, bpm: float) -> float:
        hh = grids.get("hihat_c", grids.get("ride", []))
        if not hh:
            return 0.50
        on_hits  = sum(1 for i, v in enumerate(hh) if v > 0 and i % 4 == 0)
        off_hits = sum(1 for i, v in enumerate(hh) if v > 0 and i % 4 == 2)
        ratio = 0.50 + min(off_hits / (on_hits + 1e-6), 0.5) * 0.5
        return float(np.clip(ratio, 0.50, 0.75))

    swing_a = estimate_swing(grids_a, bpm_a)
    swing_b = estimate_swing(grids_b, bpm_b)

    # ── Modo de interpolación ─────────────────────────────────────────────
    steps = args.steps
    mode  = args.mode

    def alpha_for_step(i: int, n: int, mode: str) -> float:
        """Calcula alpha (0→1) para el paso i de n pasos totales."""
        if n <= 1:
            return 0.0
        t = i / (n - 1)
        if mode == "linear":
            return t
        elif mode == "sigmoid":
            # Suave en extremos, rápido en el centro
            return 1 / (1 + math.exp(-10 * (t - 0.5)))
        elif mode == "exponential":
            return t ** 2
        elif mode == "sinusoidal":
            return (1 - math.cos(math.pi * t)) / 2
        return t

    # ── Calcular tempo interpolado ────────────────────────────────────────
    def interp_tempo(alpha: float) -> float:
        return bpm_a * (1 - alpha) + bpm_b * alpha

    # ── Generar los N pasos ───────────────────────────────────────────────
    out_dir = Path(args.output_dir) if args.output_dir else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_a = Path(args.input_a).stem
    stem_b = Path(args.input_b).stem

    print(f"[morph] {stem_a} → {stem_b}  |  {steps} pasos  |  modo={mode}")
    print(f"        {target_bars} compases, swing A={swing_a:.2f}, B={swing_b:.2f}")
    print(f"        tempo A={bpm_a:.0f}bpm → B={bpm_b:.0f}bpm")

    generated = []

    for i in range(steps):
        alpha = alpha_for_step(i, steps, mode)
        rng   = random.Random(i * 31 + 7)

        # Interpolar cada capa independientemente
        struct = _morph_layer_structure(grids_a, grids_b, alpha, rng)
        subdiv = _morph_layer_subdivision(grids_a, grids_b, alpha, swing_a, swing_b)
        color  = _morph_layer_color(grids_a, grids_b, alpha)

        # Extraer swing interpolado de subdiv y limpiar la clave especial
        swing_ratio = subdiv.pop("_swing_ratio", 0.50)

        # Combinar las tres capas
        merged = {}
        for d in (struct, subdiv, color):
            for k, v in d.items():
                if k not in merged:
                    merged[k] = v
                else:
                    # Si colisionan, tomar el mayor (max element-wise)
                    merged[k] = [max(a, b) for a, b in zip(merged[k], v)]

        # Tempo interpolado
        tempo_i = interp_tempo(alpha)
        tempo_us = int(60_000_000 / tempo_i)

        # Convertir a eventos
        events_i = grids_to_events(merged, tpb)

        # Aplicar swing interpolado
        if swing_ratio > 0.51:
            events_i = apply_swing_to_events(events_i, tpb, swing_ratio)

        # Humanización suave
        if not args.no_humanize:
            events_i = humanize_events(events_i, tpb, strength=0.4)

        # Escribir MIDI
        fname = f"{stem_a}_morph_{i+1:02d}of{steps:02d}_{stem_b}.mid"
        out_path = out_dir / fname

        out_mid = mido.MidiFile(ticks_per_beat=tpb, type=0)
        track   = mido.MidiTrack()
        out_mid.tracks.append(track)

        # Set tempo
        track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

        # Escribir eventos
        prev_t = 0
        note_ons  = [(max(0, e.time_ticks + e.timing_offset),
                      "note_on", e.pitch, e.velocity) for e in events_i]
        note_offs = [(max(0, e.time_ticks + e.timing_offset + e.duration_ticks),
                      "note_off", e.pitch, 0) for e in events_i]
        all_msgs  = sorted(note_ons + note_offs, key=lambda x: x[0])

        for t_abs, msg_type, pitch, vel in all_msgs:
            delta = max(0, t_abs - prev_t)
            track.append(mido.Message(msg_type, channel=PERCUSSION_CHANNEL,
                                       note=pitch, velocity=vel, time=delta))
            prev_t = t_abs

        track.append(mido.MetaMessage("end_of_track", time=0))
        out_mid.save(str(out_path))

        pct = f"{alpha*100:.0f}%"
        print(f"  paso {i+1:2d}/{steps}  alpha={alpha:.2f} ({pct:>4s})  "
              f"tempo={tempo_i:.0f}  swing={swing_ratio:.2f}  → {fname}")
        generated.append(str(out_path))

    print(f"\n[morph] ✓ {steps} archivos generados en {out_dir}/")

    # Exportar manifiesto JSON
    manifest = {
        "source_a":   args.input_a,
        "source_b":   args.input_b,
        "steps":      steps,
        "mode":       mode,
        "target_bars":target_bars,
        "swing_a":    swing_a,
        "swing_b":    swing_b,
        "tempo_a":    bpm_a,
        "tempo_b":    bpm_b,
        "files":      generated,
    }
    manifest_path = out_dir / f"{stem_a}_morph_{stem_b}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[morph] Manifiesto: {manifest_path}")


def transform_syncopate(grids: dict, level: float, total_bars: int) -> dict:
    """
    Desplaza golpes de pulse/backbeat hacia posiciones de menor peso métrico.
    level=0: sin cambio; level=1: máxima síncopa.
    La probabilidad de mover un golpe es proporcional a level y a lo débil
    que es la posición actual (1 - peso_métrico).
    """
    result = {}
    weights = METRIC_WEIGHT_16
    weak_positions = sorted(range(16), key=lambda i: weights[i])

    for instr, arr in grids.items():
        func = FUNC.get(instr, "color")
        if func not in ("pulse", "backbeat"):
            result[instr] = arr
            continue

        new_arr = list(arr)
        n = len(arr)
        steps_per_bar = 16

        for bar in range(total_bars):
            bar_start = bar * steps_per_bar
            bar_end   = min(bar_start + steps_per_bar, n)
            bar_slice = list(new_arr[bar_start:bar_end])

            rng = random.Random(bar * 17 + hash(instr) % 100)
            for i, v in enumerate(bar_slice):
                if v <= 0:
                    continue
                w = weights[i % 16]
                prob_move = level * (1.0 - w)
                if rng.random() < prob_move:
                    for wp in weak_positions:
                        if abs(wp - i) <= 3 and bar_slice[wp] == 0:
                            bar_slice[wp] = v
                            bar_slice[i]  = 0.0
                            break

            new_arr[bar_start:bar_end] = bar_slice

        result[instr] = new_arr

    return result


def transform_syncopate_dynamic(grids: dict, tension_curve: list,
                                 total_bars: int, tpb: int) -> dict:
    """Síncopa variable compás a compás según la curva de tensión."""
    result = {k: list(v) for k, v in grids.items()}

    for bar in range(total_bars):
        # Interpolar tensión para este compás
        t_idx = bar / max(1, total_bars - 1) * (len(tension_curve) - 1)
        t_lo = int(t_idx)
        t_hi = min(t_lo + 1, len(tension_curve) - 1)
        alpha = t_idx - t_lo
        tension = tension_curve[t_lo] * (1 - alpha) + tension_curve[t_hi] * alpha

        # Aplicar síncopa solo a este compás
        bar_grids = {k: v[bar*16:(bar+1)*16] for k, v in result.items()}
        bar_synco = transform_syncopate(bar_grids, float(tension), 1)

        for k in result:
            if k in bar_synco:
                result[k][bar*16:(bar+1)*16] = bar_synco[k][:16]

    return result


def transform_add_polyrhythm(grids: dict, n: int, total_bars: int) -> dict:
    """
    Añade una capa de N golpes equidistantes por compás (N sobre 4).
    Usa maracas o clave como instrumento de color.
    """
    result = dict(grids)
    steps_per_bar = 16
    poly_grid = [0.0] * (steps_per_bar * total_bars)

    for bar in range(total_bars):
        bar_start = bar * steps_per_bar
        for i in range(n):
            # Distribuir N golpes en 16 subdivisiones
            step = round(i * steps_per_bar / n)
            step = min(step, steps_per_bar - 1)
            poly_grid[bar_start + step] = 0.65

    # Asignar a clave si no existe, o maracas
    instr = "clave" if "clave" not in result else "maracas"
    result[instr] = poly_grid

    return result


def transform_phase_shift(grids: dict, rate_steps: int,
                           total_bars: int) -> dict:
    """
    Desplaza el patrón de percusión rate_steps pasos por compás.
    Crea el efecto de phase shifting de Steve Reich.
    """
    result = {}
    steps_per_bar = 16

    for instr, arr in grids.items():
        new_arr = list(arr)
        n = len(arr)

        for bar in range(total_bars):
            shift = (bar * rate_steps) % steps_per_bar
            bar_start = bar * steps_per_bar
            bar_end   = min(bar_start + steps_per_bar, n)
            bar_slice = arr[bar_start:bar_end]

            if len(bar_slice) == steps_per_bar and shift > 0:
                rotated = bar_slice[shift:] + bar_slice[:shift]
                new_arr[bar_start:bar_end] = rotated

        result[instr] = new_arr

    return result


def transform_thin(grids: dict, amount: float) -> dict:
    """Elimina aleatoriamente 'amount' proporción de golpes."""
    rng = random.Random(99)
    result = {}
    for instr, arr in grids.items():
        func = FUNC.get(instr, "color")
        # No eliminar los tiempos 1 y 3 del bombo ni el backbeat principal
        new_arr = []
        for i, v in enumerate(arr):
            if v <= 0:
                new_arr.append(0.0)
                continue
            is_structural = (func == "pulse" and i % 16 in (0, 8)) or \
                            (func == "backbeat" and i % 16 in (4, 12))
            if not is_structural and rng.random() < amount:
                new_arr.append(0.0)
            else:
                new_arr.append(v)
        result[instr] = new_arr
    return result


def transform_thicken(grids: dict, amount: float) -> dict:
    """Añade ghost notes en posiciones vacías."""
    rng = random.Random(77)
    result = {k: list(v) for k, v in grids.items()}

    for instr in ("snare", "hihat_c"):
        if instr not in result:
            continue
        arr = result[instr]
        for i in range(len(arr)):
            if arr[i] <= 0 and rng.random() < amount * 0.4:
                arr[i] = rng.uniform(0.15, 0.30)  # ghost note: baja velocity

    return result


def transform_emotional_target(grids: dict, target: str, total_bars: int) -> dict:
    """
    Mapea un target emocional en texto a transformaciones concretas.
    """
    target_l = target.lower()

    # Diccionario semántico simple
    if any(w in target_l for w in ["urgente", "urgency", "intenso", "tenso", "climax"]):
        grids = transform_syncopate(grids, 0.7, total_bars)
        grids = transform_thicken(grids, 0.5)
        print(f"          → síncopa alta + ghost notes")

    elif any(w in target_l for w in ["relajado", "calmo", "suave", "tranquilo"]):
        grids = transform_syncopate(grids, 0.1, total_bars)
        grids = transform_thin(grids, 0.35)
        print(f"          → síncopa baja + thin")

    elif any(w in target_l for w in ["groovy", "funk", "bailable"]):
        grids = transform_syncopate(grids, 0.65, total_bars)
        grids = transform_thicken(grids, 0.4)
        print(f"          → síncopa media-alta + ghost notes funk")

    elif any(w in target_l for w in ["minimalista", "minimal", "espacio", "vacio"]):
        grids = transform_thin(grids, 0.55)
        print(f"          → thin agresivo")

    elif any(w in target_l for w in ["rico", "complejo", "denso", "lleno"]):
        grids = transform_thicken(grids, 0.7)
        grids = transform_add_polyrhythm(grids, 3, total_bars)
        print(f"          → ghost notes + capa polirrítmica 3:4")

    else:
        print(f"          → target '{target}' no reconocido; sin cambios")

    return grids


def grids_to_events(grids: dict, tpb: int) -> list[DrumEvent]:
    """
    Convierte un dict de grids {instrumento: [vel_float...]} a lista de
    DrumEvent con posiciones absolutas en ticks.
    Cada posición del grid corresponde a una subdivisión de 1/16 de compás.
    """
    events = []
    ticks_per_bar  = tpb * 4
    ticks_per_step = ticks_per_bar // 16

    for instr, arr in grids.items():
        pitch = GM.get(instr)
        if pitch is None:
            continue
        for i, vel_f in enumerate(arr):
            if vel_f <= 0:
                continue
            t = i * ticks_per_step
            events.append(DrumEvent(
                time_ticks    = t,
                pitch         = pitch,
                velocity      = int(np.clip(vel_f * 127, 1, 127)),
                duration_ticks= max(10, ticks_per_step // 2),
            ))

    return sorted(events, key=lambda e: e.time_ticks)


def apply_swing_to_events(events: list[DrumEvent], tpb: int,
                           swing_ratio: float,
                           groove_family: str = "generic") -> list[DrumEvent]:
    """
    Aplica swing a los eventos de percusión desplazando las corcheas
    'off-beat' (posiciones impares del grid de 8vos).

    swing_ratio: 0.50 = straight, 0.67 = swing de jazz, 0.75 = hard swing.
    La cantidad de swing aplicada a cada instrumento depende de su función
    y del género detectado de la biblioteca.

    Algoritmo:
      En un compás de 4/4 a 480 tpb, cada tiempo = 480 ticks.
      Una corchea = 240 ticks.
      Con swing_ratio r: on_beat = r*480 ticks, off_beat = (1-r)*480 ticks.
      Los golpes en posiciones 'off-beat' se desplazan a off_beat_pos.
    """
    if abs(swing_ratio - 0.50) < 0.01:
        return events   # straight, sin cambio

    ticks_per_beat  = tpb
    ticks_per_8th   = tpb // 2          # corchea base
    ticks_on_beat   = int(tpb * swing_ratio)       # duración parte fuerte
    ticks_off_beat  = tpb - ticks_on_beat          # duración parte débil

    result = []
    for e in events:
        instr = GM_INV.get(e.pitch, "snare")
        func  = FUNC.get(instr, "color")

        # Factor de swing según función
        sf = SWING_FACTOR_BY_FUNC.get(func, 0.5)
        if sf < 0.01:
            result.append(e)
            continue

        # ¿Está este golpe en una posición 'off-beat'?
        # Off-beat = posición impar en la rejilla de 8vos dentro de un tiempo
        pos_in_beat = e.time_ticks % ticks_per_beat
        eighth_idx  = pos_in_beat // ticks_per_8th   # 0 = on, 1 = off

        if eighth_idx == 1:   # posición off-beat
            # Posición original en straight
            beat_start   = e.time_ticks - pos_in_beat
            # Nueva posición con swing
            swung_pos    = beat_start + ticks_on_beat
            # Interpolar entre straight y swung según sf
            straight_pos = beat_start + ticks_per_8th
            new_pos = int(straight_pos + (swung_pos - straight_pos) * sf)
            result.append(DrumEvent(
                time_ticks    = max(0, new_pos),
                pitch         = e.pitch,
                velocity      = e.velocity,
                duration_ticks= e.duration_ticks,
                timing_offset = e.timing_offset,
            ))
        else:
            result.append(e)

    return sorted(result, key=lambda e: e.time_ticks)


def apply_swing_dynamic(events: list[DrumEvent], tpb: int,
                         curve_interp: CurveInterpolator,
                         groove_family: str = "generic") -> list[DrumEvent]:
    """
    Versión dinámica: el swing ratio cambia compás a compás siguiendo
    la curva 'swing' de CurveInterpolator.
    """
    ticks_per_bar = tpb * 4
    result = []

    # Agrupar eventos por compás para aplicar swing con el ratio correcto
    bar_buckets: dict[int, list] = {}
    for e in events:
        bar = e.time_ticks // ticks_per_bar
        bar_buckets.setdefault(bar, []).append(e)

    for bar, bar_events in bar_buckets.items():
        ratio = curve_interp.swing_ratio(bar)
        swung = apply_swing_to_events(bar_events, tpb, ratio, groove_family)
        result.extend(swung)

    return sorted(result, key=lambda e: e.time_ticks)


def apply_density_dynamic(grids: dict, curve_interp: CurveInterpolator,
                           total_bars: int) -> dict:
    """
    Ajusta la densidad compás a compás según la curva 'activity'.
    activity > 0.65 → thicken (ghost notes)
    activity < 0.35 → thin (eliminar golpes)
    """
    result = {k: list(v) for k, v in grids.items()}

    for bar in range(total_bars):
        activity = curve_interp.get("activity", bar)
        bar_grids = {k: v[bar*16:(bar+1)*16] for k, v in result.items()}

        if activity > 0.65:
            amount = (activity - 0.65) / 0.35    # 0–1
            bar_grids = transform_thicken(bar_grids, amount * 0.5)
        elif activity < 0.35:
            amount = (0.35 - activity) / 0.35
            bar_grids = transform_thin(bar_grids, amount * 0.4)

        for k in result:
            if k in bar_grids:
                result[k][bar*16:(bar+1)*16] = bar_grids[k][:16]

    return result


def apply_velocity_dynamic(events: list[DrumEvent],
                            curve_interp: CurveInterpolator,
                            tpb: int) -> list[DrumEvent]:
    """Escala la velocity de cada golpe según la curva 'register'."""
    ticks_per_bar = tpb * 4
    result = []
    for e in events:
        bar    = e.time_ticks // ticks_per_bar
        scale  = curve_interp.get("register", bar) * 0.6 + 0.7   # 0.70–1.30
        new_vel = int(np.clip(e.velocity * scale, 10, 127))
        result.append(DrumEvent(
            time_ticks    = e.time_ticks,
            pitch         = e.pitch,
            velocity      = new_vel,
            duration_ticks= e.duration_ticks,
            timing_offset = e.timing_offset,
        ))
    return result
    """Convierte grids a lista de DrumEvent."""
    events = []
    ticks_per_bar = tpb * 4

    for instr, arr in grids.items():
        pitch = GM.get(instr)
        if pitch is None:
            continue
        ticks_per_step = ticks_per_bar // 16

        for i, vel_f in enumerate(arr):
            if vel_f <= 0:
                continue
            t = i * ticks_per_step
            events.append(DrumEvent(
                time_ticks=t,
                pitch=pitch,
                velocity=int(np.clip(vel_f * 127, 1, 127)),
                duration_ticks=max(10, ticks_per_step // 2),
            ))

    return sorted(events, key=lambda e: e.time_ticks)


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def write_midi_with_drums(original_mid: mido.MidiFile,
                           drum_events: list[DrumEvent],
                           tpb: int,
                           out_path: str,
                           replace_percussion: bool = False):
    """
    Escribe un nuevo MIDI combinando las pistas originales con la percusión.
    Si replace_percussion=True, elimina la pista de percusión original.
    """
    new_mid = mido.MidiFile(ticks_per_beat=tpb, type=1)

    # Copiar pistas existentes (excluyendo percusión si se reemplaza)
    for track in original_mid.tracks:
        if replace_percussion:
            new_track = mido.MidiTrack()
            abs_tick = 0
            prev_tick = 0
            for msg in track:
                abs_tick += msg.time
                if hasattr(msg, "channel") and msg.channel == PERCUSSION_CHANNEL:
                    continue
                delta = abs_tick - prev_tick
                new_msg = msg.copy(time=delta)
                new_track.append(new_msg)
                prev_tick = abs_tick
            new_mid.tracks.append(new_track)
        else:
            new_mid.tracks.append(track)

    # Crear pista de percusión
    drum_track = mido.MidiTrack()
    drum_track.name = "Drums (drummer.py)"

    # Convertir eventos absolutos a mensajes delta
    events_sorted = sorted(drum_events, key=lambda e: e.time_ticks)
    prev_tick = 0

    note_ons = []
    note_offs = []

    for e in events_sorted:
        t_on  = max(0, e.time_ticks + e.timing_offset)
        t_off = t_on + e.duration_ticks
        note_ons.append( (t_on,  "note_on",  e.pitch, e.velocity) )
        note_offs.append((t_off, "note_off", e.pitch, 0) )

    all_msgs = sorted(note_ons + note_offs, key=lambda x: x[0])

    for t_abs, msg_type, pitch, vel in all_msgs:
        delta = max(0, t_abs - prev_tick)
        drum_track.append(mido.Message(
            msg_type,
            channel=PERCUSSION_CHANNEL,
            note=pitch,
            velocity=vel,
            time=delta,
        ))
        prev_tick = t_abs

    drum_track.append(mido.MetaMessage("end_of_track", time=0))
    new_mid.tracks.append(drum_track)

    new_mid.save(out_path)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DRUMMER v0.2 — Motor de percusión rítmica para MIDIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── learn ──────────────────────────────────────────────────────────────
    p_learn = sub.add_parser("learn", help="Aprende grooves desde una carpeta de MIDIs")
    p_learn.add_argument("folder", help="Carpeta con MIDIs (mixtos)")
    p_learn.add_argument("--output", "-o", default="drum_library.json")
    p_learn.add_argument("--clusters", "-k", type=int, default=16,
                         help="Número de clusters (default: 16)")
    p_learn.add_argument("--verbose", "-v", action="store_true")

    # ── analyze ────────────────────────────────────────────────────────────
    p_ana = sub.add_parser("analyze", help="Analiza la percusión de un MIDI")
    p_ana.add_argument("input", help="Archivo MIDI")
    p_ana.add_argument("--export", help="Exportar análisis a JSON")

    # ── extract ────────────────────────────────────────────────────────────
    p_ext = sub.add_parser("extract",
        help="Extrae el groove de un MIDI y lo añade a la biblioteca")
    p_ext.add_argument("input", help="MIDI del que extraer el groove")
    p_ext.add_argument("--name", "-n", required=True,
                        help="Nombre del groove en la biblioteca (ej: 'mi_groove_funk')")
    p_ext.add_argument("--library", "-l", default="drum_library.json",
                        help="Biblioteca destino (default: drum_library.json)")
    p_ext.add_argument("--export-midi", metavar="FILE",
                        help="Exportar el patrón base como MIDI independiente")
    p_ext.add_argument("--force", action="store_true",
                        help="Sobreescribir si ya existe un groove con ese nombre")

    # ── add ────────────────────────────────────────────────────────────────
    p_add = sub.add_parser("add", help="Añade percusión a un MIDI sin ella")
    p_add.add_argument("input", help="MIDI sin percusión")
    p_add.add_argument("--output", "-o")
    p_add.add_argument("--library", "-l", help="drum_library.json")
    p_add.add_argument("--fingerprint", "-f", help=".fingerprint.json")
    p_add.add_argument("--tension-curve", help="curves.json de tension_designer")
    p_add.add_argument("--feel",
                        choices=["supportive", "contrasting", "independent"],
                        default="supportive")
    p_add.add_argument("--no-humanize", action="store_true")
    p_add.add_argument("--force", action="store_true",
                        help="Añadir aunque ya tenga percusión")

    # ── transform ──────────────────────────────────────────────────────────
    p_tr = sub.add_parser("transform", help="Transforma la percusión existente")
    p_tr.add_argument("input", help="MIDI con percusión")
    p_tr.add_argument("--output", "-o")
    p_tr.add_argument("--tension-curve",
                       help="curves.json de tension_designer (activa sinc+densidad+swing+polirritmo dinámicos)")
    p_tr.add_argument("--syncopate", type=float, metavar="0-1",
                       help="Síncopa estática (sobrescribe curva tension)")
    p_tr.add_argument("--swing", type=float, metavar="0-1",
                       help="Swing estático: 0=straight, 0.34=jazz, 1=hard (sobrescribe curva swing)")
    p_tr.add_argument("--polyrhythm", type=int, metavar="N",
                       help="Capa polirrítmica N:4 en todos los compases")
    p_tr.add_argument("--phase-shift", type=int, metavar="STEPS",
                       help="Phase shift: N pasos/compás (Steve Reich)")
    p_tr.add_argument("--humanize", type=float, metavar="0-1",
                       help="Humanización: micro-timing + velocity")
    p_tr.add_argument("--thin", type=float, metavar="0-1",
                       help="Eliminar fracción de golpes (sobrescribe curva activity)")
    p_tr.add_argument("--thicken", type=float, metavar="0-1",
                       help="Añadir ghost notes (sobrescribe curva activity)")
    p_tr.add_argument("--emotion", metavar="TARGET",
                       help="Target emocional: 'más urgente', 'relajado', etc.")

    # ── morph ──────────────────────────────────────────────────────────────
    p_mo = sub.add_parser("morph",
        help="Morphing gradual entre dos grooves en N pasos")
    p_mo.add_argument("input_a", help="MIDI fuente A")
    p_mo.add_argument("input_b", help="MIDI fuente B")
    p_mo.add_argument("--steps", "-n", type=int, default=8,
                       help="Número de pasos intermedios (default: 8)")
    p_mo.add_argument("--bars", type=int, default=None,
                       help="Normalizar ambos patrones a N compases")
    p_mo.add_argument("--mode",
                       choices=["linear", "sigmoid", "exponential", "sinusoidal"],
                       default="linear",
                       help="Curva de interpolación (default: linear)")
    p_mo.add_argument("--output-dir", "-o", default=".",
                       help="Carpeta de salida (default: directorio actual)")
    p_mo.add_argument("--no-humanize", action="store_true",
                       help="Desactivar humanización en los pasos")

    args = parser.parse_args()

    if args.command == "learn":
        cmd_learn(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "add":
        cmd_add(args)
    elif args.command == "transform":
        cmd_transform(args)
    elif args.command == "morph":
        cmd_morph(args)


if __name__ == "__main__":
    main()

