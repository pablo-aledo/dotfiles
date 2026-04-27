#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                     CORPUS EXPLORER  v1.1                           ║
║         De la intención musical a la música — sin fricción          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Un flujo de trabajo en cinco pasos que parte de un corpus MIDI     ║
║  sin etiquetar y termina en audio reproducible. Cada paso es        ║
║  independiente: puedes entrar en cualquier punto.                   ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PASO 1 — HEALTH: ¿qué hay en el corpus?                           ║
║                                                                      ║
║    python corpus_explorer.py health ./midis/                        ║
║    python corpus_explorer.py health ./midis/ --sample 1000         ║
║                                                                      ║
║  Analiza una muestra del corpus y reporta:                          ║
║    · Porcentaje de archivos utilizables                             ║
║    · Distribución de duraciones (histograma)                        ║
║    · Distribución de pistas, tempo, densidad de notas              ║
║    · Estimación del tiempo de indexado                              ║
║                                                                      ║
║  Opciones:                                                           ║
║    --sample N     Tamaño de muestra (default: 500)                  ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PASO 2 — INDEX: vectorizar el corpus (una sola vez)               ║
║                                                                      ║
║    python corpus_explorer.py index ./midis/                         ║
║    python corpus_explorer.py index ./midis/ --output mi_corpus.npz ║
║                                                                      ║
║  Procesa todos los MIDIs y guarda un índice vectorial de            ║
║  10 dimensiones musicales (sin dependencias pesadas):               ║
║    pitch_center · pitch_range · interval_mean · contour            ║
║    density · rhythm_variance · polyphony                           ║
║    velocity_mean · velocity_variance · silence_ratio               ║
║                                                                      ║
║  El índice se guarda como .npz (numpy comprimido).                  ║
║  100.000 MIDIs ≈ 1-3 horas. Solo hay que correrlo una vez.         ║
║                                                                      ║
║  Opciones:                                                           ║
║    --output FILE   Nombre del índice (default: corpus.npz)          ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PASO 3 — CLUSTER: entender la estructura del corpus               ║
║                                                                      ║
║    python corpus_explorer.py cluster corpus.npz                     ║
║    python corpus_explorer.py cluster corpus.npz --k 15             ║
║                                                                      ║
║  Agrupa los MIDIs por similitud musical y describe cada grupo:      ║
║    "grave, muy sparse, descendente, con silencios"                  ║
║    "agudo, denso, polifónico, dinámica muy variable"                ║
║                                                                      ║
║  Útil para entender qué estilos y caracteres hay disponibles        ║
║  antes de buscar por intención.                                     ║
║  Guarda los grupos en corpus.clusters.json.                         ║
║                                                                      ║
║  Opciones:                                                           ║
║    --k N     Número de clusters (default: automático, ≈n/500)      ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PASO 4 — SEARCH: buscar por intención                             ║
║                                                                      ║
║    python corpus_explorer.py search corpus.npz "algo melancólico"  ║
║    python corpus_explorer.py search corpus.npz "denso y agitado"   ║
║    python corpus_explorer.py search corpus.npz                      ║
║                                                                      ║
║  Traduce una descripción en lenguaje natural a un vector de         ║
║  búsqueda y devuelve los N MIDIs más cercanos del corpus.           ║
║                                                                      ║
║  Palabras clave reconocidas (ejemplos):                             ║
║    grave · agudo · denso · sparse · silencioso · continuo          ║
║    descendente · ascendente · plano · oscilante · irregular        ║
║    melancólico · tenso · oscuro · calmo · etéreo · fragmentado     ║
║    lento · rápido · polifónico · monofónico · expresivo            ║
║                                                                      ║
║  Con LLM (requiere anthropic o openai): comprensión semántica enriquecida   ║
║  Sin LLM: mapa local de ~50 palabras clave en español                       ║
║                                                                      ║
║  Opciones:                                                           ║
║    --top N            Número de resultados (default: 5)             ║
║    --no-llm           Usar solo el mapa semántico local             ║
║    --llm-provider P   anthropic | openai | auto (default: auto)     ║
║    --play             Reproducir resultados con timidity            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  PASO 5 — SEED: de la intención a la música                        ║
║                                                                      ║
║    python corpus_explorer.py seed corpus.npz                        ║
║    python corpus_explorer.py seed corpus.npz --no-llm              ║
║    python corpus_explorer.py seed corpus.npz --soundfont ~/sf2/... ║
║                                                                      ║
║  Flujo interactivo completo en tres fases:                          ║
║                                                                      ║
║  [1] EXPLORACIÓN — búsqueda iterativa por intención                 ║
║      · Describes lo que quieres en lenguaje libre                   ║
║      · El sistema encuentra fragmentos del corpus que resuenan      ║
║      · Eliges, rechazas, o refinas ("más oscuro", "algo entre      ║
║        el 2 y el 3")                                                ║
║      · El vector de búsqueda se desplaza gradualmente hacia         ║
║        tus elecciones (con inercia — no salta de golpe)            ║
║                                                                      ║
║  [2] CRISTALIZACIÓN — preguntas contextuales                        ║
║      · Tras cada elección, el sistema hace UNA sola pregunta        ║
║      · Las preguntas se seleccionan según lo que aún es ambiguo    ║
║        en tus elecciones: si elegiste consistentemente fragmentos   ║
║        con silencios, te pregunta sobre el silencio — no sobre     ║
║        el contorno, que ya es evidente                              ║
║      · Sin LLM: banco de 15 preguntas con selección contextual     ║
║      · Con LLM: el modelo lee el historial y elige o adapta        ║
║      · Dimensiones cristalizables:                                  ║
║          contorno · densidad · silencio · polifonía · registro     ║
║          ritmo · dinámica · evolución · resolución · duración      ║
║          función · emoción · instrumentación                        ║
║                                                                      ║
║  [3] GENERACIÓN Y REFINAMIENTO — audio real                         ║
║      · Extrae el perfil generativo de los fragmentos elegidos       ║
║        (escala implícita, tempo, densidad, contorno, dinámica)     ║
║      · Genera un MIDI original inspirado en ese perfil             ║
║      · Reproduce con timidity (audio con soundfont orquestal)      ║
║      · Refinas en lenguaje natural:                                 ║
║          "más oscuro"    → baja registro, escala más oscura        ║
║          "más sparse"    → menos notas, más silencios              ║
║          "más lento"     → reduce tempo                            ║
║          "más tenso"     → ritmo irregular, escala armónica        ║
║          "cambiar sonido"→ elige otra instrumentación              ║
║          "otra"          → variante con nueva semilla aleatoria    ║
║          "guardar"       → guarda y termina                        ║
║                                                                      ║
║  Grupos de instrumentación disponibles:                             ║
║    cuerdas_solas      — cuarteto (Vl I, Vl II, Va, Vc)            ║
║    cuerdas_orquesta   — sección de cuerdas completa               ║
║    viento_madera      — cuarteto (Fl, Ob, Cl, Fg)                 ║
║    piano_solo         — piano solo                                  ║
║    camara_mixta       — piano + cuerdas                            ║
║    voz_cuerdas        — voz + cuarteto                             ║
║    electroacustico    — pads y texturas sintéticas                 ║
║    solo_melodico      — una voz (Vl/Fl/Vc según registro)         ║
║                                                                      ║
║  Salida:                                                             ║
║    seed_<intención>_<tónica>_<escala>.mid   — MIDI generado        ║
║    seed_<intención>_<tónica>_<escala>.seed.json — perfil y metadata║
║                                                                      ║
║    --no-llm            Usar solo lógica local (sin API)            ║
║    --llm-provider P    anthropic | openai | clipboard | auto       ║
║                        auto      → usa API si disponible           ║
║                        clipboard → muestra el prompt para copiarlo ║
║                                    en cualquier LLM manualmente    ║
║    --soundfont FILE    Soundfont .sf2 para timidity                 ║
║    --preview-seconds N Duración del preview de corpus (def: 20)    ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS                                                        ║
║                                                                      ║
║    pip install mido numpy          # obligatorio                    ║
║    pip install anthropic           # opcional: LLM (Anthropic API) ║
║    pip install openai              # opcional: LLM (OpenAI API)    ║
║    pip install scipy               # opcional: clustering mejorado  ║
║                                                                      ║
║    timidity                        # reproducción de audio          ║
║      sudo apt install timidity     # Ubuntu/Debian                  ║
║      brew install timidity         # macOS                          ║
║                                                                      ║
║  Variables de entorno para API keys:                                ║
║    ANTHROPIC_API_KEY               # para --llm-provider anthropic  ║
║    OPENAI_API_KEY                  # para --llm-provider openai     ║
║                                                                      ║
║  Modo clipboard (sin API key, cualquier LLM):                       ║
║    python corpus_explorer.py seed corpus.npz --llm-provider clipboard║
║    → el programa muestra el prompt formateado, tú lo copias a      ║
║      ChatGPT / Claude.ai / Gemini / modelo local / lo que sea,     ║
║      pegas la respuesta de vuelta, y el flujo continúa             ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  FLUJO TÍPICO DE PRIMERA VEZ                                        ║
║                                                                      ║
║    # 1. Ver qué hay (minutos)                                       ║
║    python corpus_explorer.py health ./midis/                        ║
║                                                                      ║
║    # 2. Indexar (horas, solo una vez)                               ║
║    python corpus_explorer.py index ./midis/ --output corpus.npz    ║
║                                                                      ║
║    # 3. Explorar estructura (opcional pero útil)                    ║
║    python corpus_explorer.py cluster corpus.npz                     ║
║                                                                      ║
║    # 4. Buscar material de referencia                               ║
║    python corpus_explorer.py search corpus.npz "algo melancólico"  ║
║                                                                      ║
║    # 5. Componer desde una intención                                ║
║    python corpus_explorer.py seed corpus.npz                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import math
import time
import struct
import random
import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

# ─── Dependencias opcionales ────────────────────────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False

try:
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


# ─── Abstracción LLM ────────────────────────────────────────────────

def _has_llm(provider: str = "auto") -> bool:
    """Devuelve True si el proveedor LLM solicitado está disponible."""
    if provider == "clipboard":
        return True           # siempre disponible — no necesita API
    if provider == "anthropic":
        return HAS_ANTHROPIC
    if provider == "openai":
        return HAS_OPENAI
    return HAS_ANTHROPIC or HAS_OPENAI


def _llm_call(system: str, user: str, provider: str = "auto",
              max_tokens: int = 400) -> str:
    """
    Llama al LLM elegido y devuelve el texto de la respuesta.
      provider: "anthropic" | "openai" | "auto" | "clipboard"
      auto      → anthropic si disponible, si no openai, si no lanza excepción.
      clipboard → muestra el prompt, espera que el usuario pegue la respuesta.
                  Compatible con cualquier LLM (ChatGPT web, Claude.ai, Gemini,
                  modelos locales, etc.) sin necesidad de API key.
    """
    resolved = provider
    if provider == "auto":
        if HAS_ANTHROPIC:
            resolved = "anthropic"
        elif HAS_OPENAI:
            resolved = "openai"
        else:
            raise RuntimeError("Ningún LLM disponible. "
                               "pip install anthropic  o  pip install openai  "
                               "o usa --llm-provider clipboard")

    if resolved == "clipboard":
        return _llm_call_clipboard(system, user)

    if resolved == "anthropic":
        if not HAS_ANTHROPIC:
            raise RuntimeError("anthropic no instalado. pip install anthropic")
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text.strip()

    if resolved == "openai":
        if not HAS_OPENAI:
            raise RuntimeError("openai no instalado. pip install openai")
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content.strip()

    raise RuntimeError(f"Proveedor desconocido: {resolved}")


def _llm_call_clipboard(system: str, user: str) -> str:
    """
    Modo clipboard: muestra el prompt completo para que el usuario lo copie
    y pegue en cualquier LLM, luego recoge la respuesta por stdin.

    El prompt se formatea para que cualquier modelo entienda qué hacer
    sin contexto adicional (el system prompt va incluido).
    """
    prompt = f"{system}\n\n---\n\n{user}"

    # Intentar copiar al portapapeles automáticamente si hay herramienta disponible
    copied = False
    try:
        import subprocess
        # Linux (xclip / xsel), macOS (pbcopy)
        for cmd in (["xclip", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--input"],
                    ["pbcopy"]):
            try:
                subprocess.run(cmd, input=prompt.encode(), check=True,
                               capture_output=True)
                copied = True
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    except Exception:
        pass

    print(f"\n  {'─'*54}")
    print(f"  {bold('MODO CLIPBOARD — copia este prompt en tu LLM')}")
    print(f"  {'─'*54}\n")
    print(prompt)
    print(f"\n  {'─'*54}")
    if copied:
        print(f"  {green('✓ Copiado al portapapeles automáticamente')}")
    else:
        print(f"  {yellow('↑ Copia el texto de arriba y pégalo en tu LLM')}")
    print(f"  {'─'*54}\n")

    # Recoger respuesta — acepta entrada multilínea hasta línea vacía o EOF
    print(f"  {bold('Pega aquí la respuesta del LLM')} "
          f"{dim('(línea vacía para terminar):')}\n")

    lines = []
    try:
        while True:
            line = input()
            if line == "" and lines:
                # Línea vacía después de contenido → fin
                break
            lines.append(line)
    except EOFError:
        pass   # Ctrl+D

    response = "\n".join(lines).strip()

    if not response:
        raise RuntimeError("Respuesta vacía — operación cancelada")

    return response



USE_COLOR = sys.stdout.isatty()

def c(text, code):
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def bold(t):    return c(t, "1")
def dim(t):     return c(t, "2")
def green(t):   return c(t, "32")
def yellow(t):  return c(t, "33")
def cyan(t):    return c(t, "36")
def red(t):     return c(t, "31")
def magenta(t): return c(t, "35")


# ════════════════════════════════════════════════════════════════════
#  VECTORIZADOR — 10 dimensiones, solo mido + numpy
# ════════════════════════════════════════════════════════════════════

DIM_NAMES = [
    "pitch_center",       # 0  altura media normalizada (0-1)
    "pitch_range",        # 1  rango de alturas normalizado (0-1)
    "interval_mean",      # 2  intervalo medio entre notas (0-1)
    "contour",            # 3  dirección global: -1 baja, 0 plana, 1 sube
    "density",            # 4  notas por segundo (0-1, saturado a 10 nps)
    "rhythm_variance",    # 5  irregularidad rítmica (0-1)
    "polyphony",          # 6  media de voces simultáneas (0-1, sat. a 8)
    "velocity_mean",      # 7  dinámica media (0-1)
    "velocity_variance",  # 8  variación dinámica (0-1)
    "silence_ratio",      # 9  proporción de silencio (0-1)
]

N_DIMS = len(DIM_NAMES)


def vectorize_midi(path: str) -> dict:
    """
    Extrae vector de 10 dimensiones de un archivo MIDI.
    Devuelve dict con 'vector', 'duration_s', 'n_tracks', 'n_notes', 'error'.
    """
    result = {
        "path": str(path),
        "vector": None,
        "duration_s": 0.0,
        "n_tracks": 0,
        "n_notes": 0,
        "tempo_bpm": 120,
        "error": None,
    }

    try:
        mid = mido.MidiFile(str(path), clip=True)
    except Exception as e:
        result["error"] = str(e)
        return result

    # ── Recolectar eventos ──────────────────────────────────────────
    tempo = 500000  # microseg/beat → 120 BPM
    ticks_per_beat = mid.ticks_per_beat or 480

    notes = []        # (time_s, pitch, velocity, duration_s)
    active = {}       # pitch → (time_s, velocity)
    current_time_s = 0.0
    all_times_s = []

    for track in mid.tracks:
        current_time_s = 0.0
        active = {}
        for msg in track:
            delta_s = mido.tick2second(msg.time, ticks_per_beat, tempo)
            current_time_s += delta_s

            if msg.type == "set_tempo":
                tempo = msg.tempo

            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (current_time_s, msg.velocity)
                all_times_s.append(current_time_s)

            elif msg.type in ("note_off",) or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note in active:
                    on_time, vel = active.pop(msg.note)
                    dur = current_time_s - on_time
                    notes.append((on_time, msg.note, vel, max(dur, 0.05)))

    result["n_tracks"] = len(mid.tracks)
    result["n_notes"] = len(notes)
    result["tempo_bpm"] = round(60_000_000 / tempo)

    if len(notes) < 4:
        result["error"] = "too_few_notes"
        return result

    # ── Duración total ──────────────────────────────────────────────
    end_times = [n[0] + n[3] for n in notes]
    total_dur = max(end_times) if end_times else 0.0
    result["duration_s"] = total_dur

    if total_dur < 2.0:
        result["error"] = "too_short"
        return result

    # ── Calcular dimensiones ────────────────────────────────────────
    pitches    = [n[1] for n in notes]
    velocities = [n[2] for n in notes]
    onset_times = sorted([n[0] for n in notes])

    # 0. pitch_center: media de alturas, normalizada 0-1 sobre rango MIDI
    pitch_mean = sum(pitches) / len(pitches)
    pitch_center = (pitch_mean - 21) / (108 - 21)  # A0=21, C8=108

    # 1. pitch_range
    pitch_range = (max(pitches) - min(pitches)) / 87.0

    # 2. interval_mean: promedio de intervalos absolutos entre notas consecutivas
    intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
    interval_mean = min((sum(intervals) / len(intervals)) / 12.0, 1.0)

    # 3. contour: dirección global (-1 a 1)
    first_half  = pitches[:len(pitches)//2]
    second_half = pitches[len(pitches)//2:]
    mean_first  = sum(first_half) / len(first_half)
    mean_second = sum(second_half) / len(second_half)
    contour_raw = mean_second - mean_first
    contour = max(-1.0, min(1.0, contour_raw / 12.0))
    contour_norm = (contour + 1) / 2  # 0-1

    # 4. density: notas por segundo, saturado a 10
    density = min(len(notes) / max(total_dur, 1.0) / 10.0, 1.0)

    # 5. rhythm_variance: varianza de IOI (inter-onset intervals)
    iois = [onset_times[i+1] - onset_times[i] for i in range(len(onset_times)-1)]
    if iois:
        mean_ioi = sum(iois) / len(iois)
        var_ioi  = sum((x - mean_ioi)**2 for x in iois) / len(iois)
        rhythm_variance = min(math.sqrt(var_ioi) / 2.0, 1.0)
    else:
        rhythm_variance = 0.0

    # 6. polyphony: media de notas simultáneas, saturado a 8
    # Muestreamos en 100 puntos de tiempo
    sample_times = [total_dur * i / 100 for i in range(100)]
    poly_counts = []
    for t in sample_times:
        count = sum(1 for (on, p, v, dur) in notes if on <= t <= on + dur)
        poly_counts.append(count)
    polyphony = min(sum(poly_counts) / len(poly_counts) / 8.0, 1.0)

    # 7. velocity_mean
    velocity_mean = sum(velocities) / len(velocities) / 127.0

    # 8. velocity_variance
    vel_mean = sum(velocities) / len(velocities)
    vel_var  = sum((v - vel_mean)**2 for v in velocities) / len(velocities)
    velocity_variance = min(math.sqrt(vel_var) / 64.0, 1.0)

    # 9. silence_ratio
    total_note_time = sum(n[3] for n in notes)
    silence_ratio   = max(0.0, 1.0 - (total_note_time / max(total_dur, 1.0)))

    vector = [
        pitch_center,
        pitch_range,
        interval_mean,
        contour_norm,
        density,
        rhythm_variance,
        polyphony,
        velocity_mean,
        velocity_variance,
        silence_ratio,
    ]

    result["vector"] = vector
    return result


# ════════════════════════════════════════════════════════════════════
#  MODO: HEALTH
# ════════════════════════════════════════════════════════════════════

def cmd_health(args):
    """Diagnóstico rápido del corpus — sin vectorizar en profundidad."""

    if not HAS_MIDO:
        print(red("Error: mido no instalado. pip install mido"))
        return

    midi_dir = Path(args.path)
    if not midi_dir.exists():
        print(red(f"Error: no existe '{midi_dir}'"))
        return

    midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    total = len(midi_files)

    if total == 0:
        print(yellow("No se encontraron archivos MIDI en el directorio."))
        return

    print(f"\n{bold('CORPUS EXPLORER — HEALTH CHECK')}")
    print(f"Directorio: {cyan(str(midi_dir))}")
    print(f"Archivos encontrados: {bold(str(total))}\n")

    # Muestreo para no tardar demasiado
    sample_size = min(args.sample, total)
    sample = random.sample(midi_files, sample_size)

    print(f"Analizando muestra de {sample_size} archivos...\n")

    stats = {
        "ok": 0,
        "corrupt": 0,
        "too_short": 0,
        "too_few_notes": 0,
        "durations": [],
        "n_notes": [],
        "n_tracks": [],
        "tempos": [],
        "errors": [],
    }

    bar_width = 40
    for i, path in enumerate(sample):
        # Barra de progreso
        pct = (i + 1) / sample_size
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {i+1}/{sample_size}", end="", flush=True)

        r = vectorize_midi(path)

        if r["error"] == "too_short":
            stats["too_short"] += 1
        elif r["error"] == "too_few_notes":
            stats["too_few_notes"] += 1
        elif r["error"]:
            stats["corrupt"] += 1
            stats["errors"].append(str(r["error"])[:60])
        else:
            stats["ok"] += 1
            stats["durations"].append(r["duration_s"])
            stats["n_notes"].append(r["n_notes"])
            stats["n_tracks"].append(r["n_tracks"])
            stats["tempos"].append(r["tempo_bpm"])

    print("\n")

    usable = stats["ok"]
    usable_pct = usable / sample_size * 100

    print("─" * 60)
    print(f"  {bold('USABILIDAD')}")
    print(f"  {'Utilizables':<25} {green(str(usable))} ({usable_pct:.1f}%)")
    print(f"  {'Corruptos':<25} {red(str(stats['corrupt']))}")
    print(f"  {'Demasiado cortos (<2s)':<25} {yellow(str(stats['too_short']))}")
    print(f"  {'Muy pocas notas (<4)':<25} {yellow(str(stats['too_few_notes']))}")
    print()

    if stats["durations"]:
        durs = sorted(stats["durations"])
        n = len(durs)

        def pct_val(p):
            return durs[min(int(n * p / 100), n - 1)]

        print(f"  {bold('DURACIÓN (segundos)')}")
        print(f"  {'Mínima':<25} {pct_val(0):.1f}s")
        print(f"  {'P10':<25} {pct_val(10):.1f}s")
        print(f"  {'Mediana':<25} {pct_val(50):.1f}s")
        print(f"  {'P90':<25} {pct_val(90):.1f}s")
        print(f"  {'Máxima':<25} {pct_val(100):.1f}s")
        print(f"  {'Media':<25} {sum(durs)/n:.1f}s")
        print()

        # Histograma de duraciones
        buckets = [0, 10, 30, 60, 120, 300, float("inf")]
        labels  = ["<10s", "10-30s", "30-60s", "1-2min", "2-5min", ">5min"]
        counts  = [0] * len(labels)
        for d in durs:
            for j, b in enumerate(buckets[1:]):
                if d < b:
                    counts[j] += 1
                    break

        print(f"  {bold('DISTRIBUCIÓN DE DURACIÓN')}")
        max_count = max(counts) or 1
        for label, count in zip(labels, counts):
            bar_len = int(30 * count / max_count)
            bar = "▓" * bar_len
            print(f"  {label:<10} {bar:<30} {count}")
        print()

        notes_sorted = sorted(stats["n_notes"])
        print(f"  {bold('NOTAS POR ARCHIVO')}")
        print(f"  {'Mediana':<25} {notes_sorted[n//2]}")
        print(f"  {'P90':<25} {notes_sorted[int(n*0.9)]}")
        print()

        tracks_c = Counter(stats["n_tracks"])
        print(f"  {bold('DISTRIBUCIÓN DE PISTAS')}")
        for k in sorted(tracks_c.keys())[:8]:
            pct = tracks_c[k] / n * 100
            print(f"  {k} pista(s){'s' if k>1 else ' ':<12} {tracks_c[k]:>5}  ({pct:.1f}%)")
        print()

        tempos_sorted = sorted(stats["tempos"])
        print(f"  {bold('TEMPO (BPM)')}")
        print(f"  {'Mínimo':<25} {tempos_sorted[0]}")
        print(f"  {'Mediano':<25} {tempos_sorted[n//2]}")
        print(f"  {'Máximo':<25} {tempos_sorted[-1]}")
        print()

    # Estimación para corpus completo
    est_usable = int(total * usable_pct / 100)
    print("─" * 60)
    print(f"  {bold('ESTIMACIÓN CORPUS COMPLETO')}")
    print(f"  Archivos totales:    {total:>8,}")
    print(f"  Estimado utilizables:{est_usable:>8,} ({usable_pct:.1f}%)")

    # Tiempo estimado para indexar
    time_per_file = 0.05  # estimación conservadora en segundos
    est_index_time = est_usable * time_per_file
    if est_index_time < 60:
        time_str = f"{est_index_time:.0f} segundos"
    elif est_index_time < 3600:
        time_str = f"{est_index_time/60:.1f} minutos"
    else:
        time_str = f"{est_index_time/3600:.1f} horas"

    print(f"  Tiempo estimado de indexado: {cyan(time_str)}")
    print()

    if usable_pct > 70:
        print(green("  ✓ Corpus en buenas condiciones. Puedes proceder al indexado."))
        print(f"  Siguiente paso: {bold('python corpus_explorer.py index ./midis/ --output corpus.npz')}")
    elif usable_pct > 40:
        print(yellow("  ⚠ Corpus con bastante ruido. El indexado filtrará automáticamente."))
    else:
        print(red("  ✗ Corpus con muchos archivos inutilizables. Considera limpiar antes."))
    print()


# ════════════════════════════════════════════════════════════════════
#  MODO: INDEX
# ════════════════════════════════════════════════════════════════════

def cmd_index(args):
    """Vectoriza el corpus completo y guarda índice .npz"""

    if not HAS_MIDO:
        print(red("Error: mido no instalado. pip install mido"))
        return
    if not HAS_NUMPY:
        print(red("Error: numpy no instalado. pip install numpy"))
        return

    midi_dir = Path(args.path)
    output   = Path(args.output) if args.output else Path("corpus.npz")

    midi_files = list(midi_dir.rglob("*.mid")) + list(midi_dir.rglob("*.midi"))
    total = len(midi_files)

    if total == 0:
        print(yellow("No se encontraron archivos MIDI."))
        return

    print(f"\n{bold('CORPUS EXPLORER — INDEX')}")
    print(f"Archivos a procesar: {total:,}")
    print(f"Salida: {cyan(str(output))}\n")

    vectors  = []
    paths    = []
    metadata = []
    errors   = 0
    skipped  = 0

    bar_width = 40
    t_start   = time.time()

    for i, path in enumerate(midi_files):
        # Progreso
        pct    = (i + 1) / total
        filled = int(bar_width * pct)
        bar    = "█" * filled + "░" * (bar_width - filled)
        elapsed = time.time() - t_start
        eta     = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(f"\r  [{bar}] {i+1}/{total}  ETA:{eta:.0f}s  OK:{len(vectors)}  Err:{errors}",
              end="", flush=True)

        r = vectorize_midi(path)

        if r["error"]:
            errors += 1
            continue

        vectors.append(r["vector"])
        paths.append(str(path))
        metadata.append({
            "duration_s": r["duration_s"],
            "n_notes":    r["n_notes"],
            "n_tracks":   r["n_tracks"],
            "tempo_bpm":  r["tempo_bpm"],
        })

    print()

    if not vectors:
        print(red("\nNo se pudo vectorizar ningún archivo."))
        return

    # Filtrar vectores con dimensiones incorrectas (debería ser N_DIMS)
    clean_vectors, clean_paths, clean_meta = [], [], []
    bad = 0
    for v, p, m in zip(vectors, paths, metadata):
        if v is not None and len(v) == N_DIMS and all(x == x for x in v):  # len OK y sin NaN
            clean_vectors.append(v)
            clean_paths.append(p)
            clean_meta.append(m)
        else:
            bad += 1
    if bad:
        print(yellow(f"  {bad} vectores descartados por dimensiones incorrectas"))

    # Guardar índice
    arr_vectors  = np.array(clean_vectors, dtype=np.float32)
    arr_paths    = np.array(clean_paths)
    arr_meta     = np.array([json.dumps(m) for m in clean_meta])

    np.savez_compressed(
        str(output),
        vectors  = arr_vectors,
        paths    = arr_paths,
        metadata = arr_meta,
        dim_names= np.array(DIM_NAMES),
    )

    elapsed = time.time() - t_start
    print(f"\n{bold('ÍNDICE GENERADO')}")
    print(f"  Vectorizados: {green(str(len(clean_vectors)))}")
    print(f"  Errores:      {red(str(errors + bad))}")
    print(f"  Tiempo:       {elapsed:.1f}s")
    print(f"  Archivo:      {cyan(str(output))}")
    print(f"\n  Siguiente: {bold(f'python corpus_explorer.py cluster {output}')}\n")


# ════════════════════════════════════════════════════════════════════
#  MODO: CLUSTER
# ════════════════════════════════════════════════════════════════════

def _kmeans_simple(X, k, n_iter=50):
    """K-means mínimo sin sklearn."""
    n = X.shape[0]
    # Inicialización aleatoria
    centers = X[np.random.choice(n, k, replace=False)]

    for _ in range(n_iter):
        # Asignación
        diffs   = X[:, None, :] - centers[None, :, :]      # (n, k, d)
        dists   = np.sqrt((diffs ** 2).sum(axis=2))         # (n, k)
        labels  = dists.argmin(axis=1)                      # (n,)
        # Actualización
        new_centers = np.zeros_like(centers)
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)
            else:
                new_centers[j] = centers[j]
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return labels, centers


def _describe_cluster(vectors, indices, paths_arr, meta_arr):
    """Genera descripción musical de un cluster."""
    vecs = vectors[indices]
    means = vecs.mean(axis=0)

    desc = []

    # Registro
    pc = means[0]
    if pc < 0.35:
        desc.append("registro grave")
    elif pc > 0.65:
        desc.append("registro agudo")
    else:
        desc.append("registro medio")

    # Densidad
    dens = means[4]
    if dens < 0.2:
        desc.append("muy pocas notas")
    elif dens < 0.5:
        desc.append("densidad moderada")
    else:
        desc.append("muy denso")

    # Contorno
    cont = means[3]
    if cont < 0.35:
        desc.append("tendencia descendente")
    elif cont > 0.65:
        desc.append("tendencia ascendente")
    else:
        desc.append("contorno plano/oscilante")

    # Silencio
    sil = means[9]
    if sil > 0.5:
        desc.append("muchos silencios")
    elif sil < 0.15:
        desc.append("continuo")

    # Polifonía
    poly = means[6]
    if poly < 0.15:
        desc.append("monofónico")
    elif poly > 0.5:
        desc.append("muy polifónico")

    # Dinámica
    vel_var = means[8]
    if vel_var > 0.5:
        desc.append("dinámica muy variable")
    elif vel_var < 0.15:
        desc.append("dinámica uniforme")

    return ", ".join(desc[:4])


def cmd_cluster(args):
    """Agrupa el corpus vectorizado por similitud."""

    if not HAS_NUMPY:
        print(red("Error: numpy no instalado."))
        return

    index_path = Path(args.path)
    if not index_path.exists():
        print(red(f"Error: no existe '{index_path}'"))
        print(f"  Primero corre: {bold('python corpus_explorer.py index ./midis/')}")
        return

    print(f"\n{bold('CORPUS EXPLORER — CLUSTER')}")
    print(f"Cargando índice {cyan(str(index_path))}...", end=" ", flush=True)

    data      = np.load(str(index_path), allow_pickle=True)
    vectors   = data["vectors"]
    paths_arr = data["paths"]
    meta_arr  = data["metadata"]
    n         = len(vectors)
    print(f"{green(str(n))} vectores")

    k = args.k if args.k else max(5, min(20, n // 500))
    print(f"Clustering en {k} grupos...\n")

    t0 = time.time()
    labels, centers = _kmeans_simple(vectors, k)

    # Calcular distancias intra-cluster (cohesión)
    cluster_info = []
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        vecs_j = vectors[idx]
        dists  = np.sqrt(((vecs_j - centers[j]) ** 2).sum(axis=1))
        cohesion = float(dists.mean())
        desc = _describe_cluster(vectors, idx, paths_arr, meta_arr)
        # Ejemplo aleatorio
        example_path = str(paths_arr[idx[random.randint(0, len(idx)-1)]])
        cluster_info.append({
            "id":       j,
            "n":        len(idx),
            "cohesion": cohesion,
            "desc":     desc,
            "example":  example_path,
            "indices":  idx.tolist(),
        })

    cluster_info.sort(key=lambda x: -x["n"])

    print(f"  {'ID':<4} {'Tamaño':>8}  {'%':>5}  {'Descripción'}")
    print("  " + "─" * 65)
    for ci in cluster_info:
        pct = ci["n"] / n * 100
        coh_bar = "▓" * int((1 - ci["cohesion"]) * 10)  # más cohesión = barra más larga
        print(f"  {ci['id']:<4} {ci['n']:>8,}  {pct:>4.1f}%  {dim(ci['desc'])}")

    print()
    print(f"  Tiempo: {time.time()-t0:.1f}s")

    # Guardar resultado de clustering
    out_path = index_path.with_suffix(".clusters.json")
    cluster_data = []
    for ci in cluster_info:
        cluster_data.append({
            "id":       ci["id"],
            "n":        ci["n"],
            "desc":     ci["desc"],
            "example":  ci["example"],
            "indices":  ci["indices"],
            "center":   centers[ci["id"]].tolist(),
        })

    with open(out_path, "w") as f:
        json.dump({"k": k, "n_total": n, "clusters": cluster_data}, f, indent=2)

    print(f"\n  Clusters guardados: {cyan(str(out_path))}")
    _next = f"python corpus_explorer.py search {index_path} \"tu intención\""
    print(f"\n  Siguiente: {bold(_next)}\n")


# ════════════════════════════════════════════════════════════════════
#  BÚSQUEDA — motor compartido por search y seed
# ════════════════════════════════════════════════════════════════════

# Mapa semántico: palabras/conceptos → pesos por dimensión
# Cada entrada: lista de (dim_index, delta, weight)
# delta: dirección en esa dimensión (+1 sube, -1 baja)

SEMANTIC_MAP = {
    # Registros
    "grave":       [(0, -1, 0.9)],
    "agudo":       [(0, +1, 0.9)],
    "bajo":        [(0, -1, 0.8)],
    "alto":        [(0, +1, 0.8)],

    # Densidad
    "denso":       [(4, +1, 0.9), (9, -1, 0.7)],
    "sparse":      [(4, -1, 0.9), (9, +1, 0.7)],
    "pocas notas": [(4, -1, 0.9)],
    "silencio":    [(9, +1, 0.9)],
    "silencioso":  [(9, +1, 0.9), (4, -1, 0.6)],
    "lleno":       [(4, +1, 0.8), (9, -1, 0.6)],

    # Contorno
    "descendente": [(3, -1, 0.9)],
    "ascendente":  [(3, +1, 0.9)],
    "plano":       [(3,  0, 0.7), (5, -1, 0.5)],
    "oscilante":   [(5, +1, 0.8)],

    # Ritmo
    "regular":     [(5, -1, 0.9)],
    "irregular":   [(5, +1, 0.9)],
    "sincopado":   [(5, +1, 0.7)],
    "estático":    [(4, -1, 0.7), (5, -1, 0.7)],
    "fluido":      [(5, -1, 0.6), (4, +0.5, 0.5)],

    # Polifonía
    "monofónico":  [(6, -1, 0.9)],
    "polifónico":  [(6, +1, 0.9)],
    "unísono":     [(6, -1, 0.8)],
    "coral":       [(6, +1, 0.8), (5, -1, 0.5)],
    "cuarteto":    [(6, +0.5, 0.7)],

    # Dinámica
    "suave":       [(7, -1, 0.8)],
    "fuerte":      [(7, +1, 0.8)],
    "expresivo":   [(8, +1, 0.8)],
    "uniforme":    [(8, -1, 0.7)],

    # Emociones / texturas (mapeadas a combinaciones)
    "melancólico": [(0, -0.3, 0.6), (4, -0.5, 0.7), (9, +0.4, 0.6)],
    "melancólica": [(0, -0.3, 0.6), (4, -0.5, 0.7), (9, +0.4, 0.6)],
    "triste":      [(0, -0.3, 0.6), (4, -0.5, 0.7), (9, +0.4, 0.6)],
    "alegre":      [(0, +0.2, 0.5), (4, +0.5, 0.7), (7, +0.3, 0.5)],
    "tenso":       [(5, +0.6, 0.7), (8, +0.5, 0.6), (2, +0.4, 0.5)],
    "tensa":       [(5, +0.6, 0.7), (8, +0.5, 0.6), (2, +0.4, 0.5)],
    "oscuro":      [(0, -0.5, 0.8), (4, -0.3, 0.5), (9, +0.3, 0.5)],
    "oscura":      [(0, -0.5, 0.8), (4, -0.3, 0.5), (9, +0.3, 0.5)],
    "luminoso":    [(0, +0.5, 0.7), (7, +0.3, 0.5)],
    "luminosa":    [(0, +0.5, 0.7), (7, +0.3, 0.5)],
    "calmo":       [(4, -0.6, 0.8), (5, -0.5, 0.7), (9, +0.3, 0.5)],
    "calma":       [(4, -0.6, 0.8), (5, -0.5, 0.7), (9, +0.3, 0.5)],
    "tranquilo":   [(4, -0.6, 0.8), (5, -0.5, 0.7)],
    "agitado":     [(4, +0.6, 0.8), (5, +0.7, 0.8)],
    "lento":       [(4, -0.5, 0.7)],
    "rápido":      [(4, +0.7, 0.8)],
    "fragmentado": [(9, +0.5, 0.7), (5, +0.5, 0.6)],
    "continuo":    [(9, -0.6, 0.7), (4, +0.3, 0.5)],
    "etéreo":      [(0, +0.5, 0.6), (4, -0.4, 0.6), (9, +0.4, 0.5)],
    "profundo":    [(0, -0.6, 0.8), (6, +0.3, 0.5)],
    "íntimo":      [(6, -0.3, 0.6), (4, -0.3, 0.5), (7, -0.2, 0.4)],
    "épico":       [(6, +0.6, 0.7), (4, +0.4, 0.6), (7, +0.4, 0.6)],
    "roto":        [(5, +0.8, 0.8), (9, +0.6, 0.7), (4, -0.4, 0.5)],
    "suspendido":  [(9, +0.4, 0.6), (5, +0.3, 0.5)],
    "disonante":   [(2, +0.7, 0.7), (5, +0.4, 0.5)],
    "consonante":  [(2, -0.5, 0.7)],
}


def text_to_query_vector(text: str) -> tuple:
    """
    Convierte texto libre en (vector_consulta, pesos_por_dim).
    Usa el mapa semántico local; si hay LLM, lo enriquece.
    Devuelve (query_vec: np.array shape (N_DIMS,), weights: np.array shape (N_DIMS,))
    """
    text_lower = text.lower()

    delta   = np.zeros(N_DIMS)
    weights = np.zeros(N_DIMS)

    for keyword, effects in SEMANTIC_MAP.items():
        if keyword in text_lower:
            for dim_idx, direction, weight in effects:
                delta[dim_idx]   += direction * weight
                weights[dim_idx] += weight

    # Normalizar deltas a [-1, 1]
    max_abs = np.abs(delta).max()
    if max_abs > 0:
        delta = delta / max_abs

    # Vector consulta centrado en 0.5 + delta escalado
    query_vec = np.clip(0.5 + delta * 0.4, 0.0, 1.0)

    # Pesos: dimensiones sin señal tienen peso bajo
    weights = np.clip(weights, 0, 1)
    weights[weights == 0] = 0.1  # mínimo peso para no ignorar ninguna dim

    return query_vec, weights


def llm_enrich_query(text: str, base_vec: np.ndarray, base_weights: np.ndarray,
                     provider: str = "auto"):
    """
    Usa el LLM para enriquecer el vector de consulta.
    Devuelve (enriched_vec, enriched_weights) o los originales si falla.
    """
    if not _has_llm(provider):
        return base_vec, base_weights

    system = """Eres un asistente de composición musical.
Tu tarea es traducir una descripción musical en lenguaje natural a parámetros numéricos.
Responde SOLO con un objeto JSON válido, sin texto adicional.

Las dimensiones son (0-1):
0. pitch_center: altura media (0=muy grave, 1=muy agudo)
1. pitch_range: rango de alturas (0=estrecho, 1=amplio)
2. interval_mean: intervalo medio (0=semitonos, 1=saltos grandes)
3. contour: dirección global (0=descendente, 0.5=plano, 1=ascendente)
4. density: densidad de notas (0=pocas, 1=muchas)
5. rhythm_variance: irregularidad rítmica (0=regular, 1=muy irregular)
6. polyphony: voces simultáneas (0=monofónico, 1=muy polifónico)
7. velocity_mean: dinámica media (0=pp, 1=ff)
8. velocity_variance: variación dinámica (0=uniforme, 1=muy expresivo)
9. silence_ratio: proporción de silencio (0=continuo, 1=muchos silencios)

Devuelve:
{
  "values": [v0, v1, ..., v9],   // valores 0-1 para cada dimensión
  "weights": [w0, w1, ..., w9],  // confianza 0-1 (0.1=sin info, 1.0=muy seguro)
  "interpretation": "breve descripción musical en español"
}"""

    try:
        raw  = _llm_call(system, text, provider=provider, max_tokens=400)
        data = json.loads(raw)
        vec  = np.array(data["values"], dtype=np.float32)
        wts  = np.array(data["weights"], dtype=np.float32)
        vec  = np.clip(vec, 0, 1)
        wts  = np.clip(wts, 0.1, 1)
        print(f"  {dim('Interpretación LLM: ' + data.get('interpretation', ''))}")
        return vec, wts
    except Exception:
        return base_vec, base_weights


def search_index(vectors: np.ndarray, query_vec: np.ndarray,
                 weights: np.ndarray, top_n: int = 5,
                 exclude: list = None) -> list:
    """
    Búsqueda por distancia euclidiana ponderada.
    Devuelve lista de (índice, distancia).
    """
    exclude = set(exclude or [])
    # Distancia ponderada
    diff = vectors - query_vec[None, :]           # (n, d)
    dist = np.sqrt((diff ** 2 * weights[None, :]).sum(axis=1))  # (n,)

    # Ordenar
    order = np.argsort(dist)
    results = []
    for idx in order:
        if int(idx) not in exclude:
            results.append((int(idx), float(dist[idx])))
        if len(results) >= top_n:
            break
    return results


def format_duration(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    m = int(s // 60)
    sec = int(s % 60)
    return f"{m}:{sec:02d}"


def describe_vector(vec: np.ndarray) -> str:
    """Descripción musical corta de un vector."""
    parts = []

    # Registro
    pc = vec[0]
    if pc < 0.35:   parts.append("grave")
    elif pc > 0.65: parts.append("agudo")
    else:           parts.append("medio")

    # Densidad
    dens = vec[4]
    if dens < 0.25:   parts.append("muy sparse")
    elif dens < 0.5:  parts.append("moderado")
    else:             parts.append("denso")

    # Contorno
    cont = vec[3]
    if cont < 0.35:   parts.append("descendente")
    elif cont > 0.65: parts.append("ascendente")
    else:             parts.append("plano")

    # Silencio
    if vec[9] > 0.5:  parts.append("con silencios")

    # Polifonía
    if vec[6] < 0.15: parts.append("monofónico")
    elif vec[6] > 0.5: parts.append("polifónico")

    return ", ".join(parts)


# ════════════════════════════════════════════════════════════════════
#  MODO: SEARCH
# ════════════════════════════════════════════════════════════════════

def cmd_search(args):
    """Búsqueda por intención textual en el corpus indexado."""

    if not HAS_NUMPY:
        print(red("Error: numpy no instalado."))
        return

    index_path = Path(args.path)
    if not index_path.exists():
        print(red(f"Error: no existe '{index_path}'"))
        return

    query_text = args.query
    if not query_text:
        query_text = input("Describe lo que buscas: ").strip()

    print(f"\n{bold('CORPUS EXPLORER — SEARCH')}")
    print(f"Consulta: {cyan(query_text)}\n")

    data    = np.load(str(index_path), allow_pickle=True)
    vectors = data["vectors"]
    paths   = data["paths"]
    meta    = data["metadata"]

    # Construir vector de consulta
    base_vec, base_weights = text_to_query_vector(query_text)

    provider = getattr(args, "llm_provider", "auto")
    if _has_llm(provider) and not args.no_llm:
        print(f"  {dim('Consultando LLM...')}", end=" ", flush=True)
        query_vec, weights = llm_enrich_query(query_text, base_vec, base_weights,
                                              provider=provider)
        print()
    else:
        query_vec, weights = base_vec, base_weights
        print(f"  {dim('Modo local (sin LLM)')}")

    results = search_index(vectors, query_vec, weights, top_n=args.top)

    print(f"\n  {bold('RESULTADOS')} ({len(results)} más cercanos)\n")
    for rank, (idx, dist) in enumerate(results, 1):
        m         = json.loads(str(meta[idx]))
        path_str  = str(paths[idx])
        filename  = Path(path_str).name
        dur_str   = format_duration(m["duration_s"])
        desc      = describe_vector(vectors[idx])
        sim       = max(0, 1 - dist)

        sim_bar = "█" * int(sim * 10) + "░" * (10 - int(sim * 10))
        print(f"  [{rank}] {bold(filename)}")
        print(f"      {dur_str:<8} {dim(desc)}")
        print(f"      Similitud: {cyan(sim_bar)} {sim:.2f}")
        print(f"      {dim(path_str)}")
        print()

    if args.play and HAS_PYGAME:
        _play_results(results, paths, meta, vectors)


# ════════════════════════════════════════════════════════════════════
#  REPRODUCCIÓN
# ════════════════════════════════════════════════════════════════════

def _play_midi(path: str, seconds: int = 20):
    """Reproduce un MIDI brevemente con pygame."""
    if not HAS_PYGAME:
        print(yellow("  pygame no instalado — no se puede reproducir."))
        return
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        print(f"  {green('▶')} Reproduciendo {Path(path).name} ({seconds}s)... ", end="", flush=True)
        time.sleep(seconds)
        pygame.mixer.music.stop()
        print(green("■"))
    except Exception as e:
        print(red(f"  Error reproduciendo: {e}"))


# ════════════════════════════════════════════════════════════════════
#  MODO: SEED
# ════════════════════════════════════════════════════════════════════

def cmd_seed(args):
    """
    Flujo completo interactivo:
    intención → búsqueda → escucha → refinamiento → orientación
    """
    if not HAS_NUMPY:
        print(red("Error: numpy no instalado."))
        return

    index_path = Path(args.path)
    if not index_path.exists():
        print(red(f"Error: no existe '{index_path}'"))
        print(f"  Primero: {bold('python corpus_explorer.py index ./midis/')}")
        return

    print(f"\n{bold('╔══════════════════════════════════════════════╗')}")
    print(f"{bold('║         CORPUS EXPLORER — SEED MODE         ║')}")
    print(f"{bold('╚══════════════════════════════════════════════╝')}\n")

    print("  Este modo te ayuda a encontrar material musical que resuene")
    print("  con tu intención, antes de componer nada.\n")

    provider = getattr(args, "llm_provider", "auto")
    use_llm  = _has_llm(provider) and not args.no_llm

    if not use_llm:
        print(f"  {yellow('⚠ LLM no disponible o desactivado — usando modo local')}")
        if not HAS_ANTHROPIC and not HAS_OPENAI:
            print(f"  {dim('  pip install anthropic  o  pip install openai')}\n")
    else:
        provider_name = "Anthropic" if (provider == "anthropic" or
                        (provider == "auto" and HAS_ANTHROPIC)) else "OpenAI"
        print(f"  {dim(f'LLM: {provider_name}')}\n")

    # ── Cargar índice ───────────────────────────────────────────────
    print(f"  Cargando corpus...", end=" ", flush=True)
    data    = np.load(str(index_path), allow_pickle=True)
    vectors = data["vectors"]
    paths   = data["paths"]
    meta    = data["metadata"]
    n       = len(vectors)
    print(f"{green(str(n))} fragmentos\n")

    # ── Estado de la sesión seed ────────────────────────────────────
    seed_state = {
        "original_intent":  "",
        "current_query":    "",
        "query_vec":        None,
        "weights":          None,
        "seen_indices":     [],
        "chosen_indices":   [],
        "rejected_indices": [],
        "history":          [],
        "crystallized":     {},
    }

    # ── Paso 1: Intención inicial ───────────────────────────────────
    print("─" * 50)
    print(f"  {bold('¿Qué quieres hacer?')}")
    print(f"  {dim('Describe libremente: una imagen, emoción, referencia, idea...')}\n")

    intent = input("  > ").strip()
    if not intent:
        intent = "algo lento y melancólico"

    seed_state["original_intent"] = intent
    seed_state["current_query"]   = intent
    seed_state["history"].append(("intent", intent))

    # ── Paso 2: Construir vector inicial ────────────────────────────
    print()
    base_vec, base_weights = text_to_query_vector(intent)

    if use_llm:
        print(f"  {dim('Interpretando con LLM...')}")
        query_vec, weights = llm_enrich_query(intent, base_vec, base_weights,
                                              provider=provider)
    else:
        query_vec, weights = base_vec, base_weights

    seed_state["query_vec"] = query_vec
    seed_state["weights"]   = weights

    # ── Bucle principal ─────────────────────────────────────────────
    round_n  = 0
    max_rounds = 8

    while round_n < max_rounds:
        round_n += 1
        print(f"\n{'─'*50}")
        print(f"  {bold(f'RONDA {round_n}')}")

        # Buscar
        top_n   = 5
        results = search_index(
            vectors,
            seed_state["query_vec"],
            seed_state["weights"],
            top_n=top_n,
            exclude=seed_state["seen_indices"],
        )

        if not results:
            print(yellow("  No quedan más fragmentos por explorar en esta dirección."))
            break

        # Mostrar resultados
        print(f"\n  Encontré {len(results)} fragmentos:\n")
        shown = []
        for rank, (idx, dist) in enumerate(results, 1):
            m        = json.loads(str(meta[idx]))
            filename = Path(str(paths[idx])).name
            dur_str  = format_duration(m["duration_s"])
            desc     = describe_vector(vectors[idx])
            sim      = max(0, 1 - dist)
            sim_bar  = "█" * int(sim * 10) + "░" * (10 - int(sim * 10))

            print(f"  [{rank}] {bold(filename)}")
            print(f"      {dur_str:<8} {dim(desc)}")
            print(f"      {cyan(sim_bar)} {sim:.2f}")

            if HAS_PYGAME:
                print(f"      {dim('tecla p para escuchar')}")

            shown.append(idx)
            seed_state["seen_indices"].append(idx)

        print()

        # ── Opciones de respuesta ───────────────────────────────────
        print(f"  {bold('¿Qué hacemos?')}")
        print(f"  {dim('Opciones:')}")
        print(f"    {cyan('1-5')}        — este se acerca")
        print(f"    {cyan('varios')}     — p.ej. \"1 3\" (elegir varios)")
        if HAS_PYGAME:
            print(f"    {cyan('p1-p5')}      — escuchar ese fragmento")
        print(f"    {cyan('ninguno')}    — nada de esto")
        print(f"    {cyan('más')}        — buscar más similares a los elegidos")
        print(f"    {cyan('refinar')}    — ajustar la búsqueda con más palabras")
        print(f"    {cyan('listo')}      — tengo suficiente orientación")
        print(f"    {cyan('guardar')}    — guardar estado y salir")
        print()

        resp = input("  > ").strip().lower()
        seed_state["history"].append(("response", resp))

        # ── Interpretar respuesta ───────────────────────────────────

        if resp in ("listo", "fin", "done", "exit", "q"):
            break

        elif resp in ("guardar", "save"):
            _save_seed_state(seed_state, index_path)
            break

        elif resp in ("ninguno", "no", "nada", "none"):
            seed_state["rejected_indices"].extend(shown)
            # Ampliar radio de búsqueda
            seed_state["weights"] = np.clip(seed_state["weights"] * 0.7, 0.05, 1.0)
            print(f"  {dim('Ampliando búsqueda...')}")

        elif resp.startswith("p") and resp[1:].isdigit():
            # Reproducir
            rank_play = int(resp[1:]) - 1
            if 0 <= rank_play < len(shown):
                path_play = str(paths[shown[rank_play]])
                _play_midi(path_play, seconds=args.preview_seconds)
            else:
                print(yellow("  Número fuera de rango."))

        elif resp in ("más", "mas", "more"):
            if seed_state["chosen_indices"]:
                # Centrar en los elegidos
                chosen_vecs = vectors[seed_state["chosen_indices"]]
                seed_state["query_vec"] = chosen_vecs.mean(axis=0)
                print(f"  {dim('Centrando en fragmentos elegidos...')}")
            else:
                print(f"  {dim('Buscando más en la misma dirección...')}")

        elif resp in ("refinar", "refine", "ajustar"):
            print(f"\n  {bold('¿Cómo refinar?')}")
            _ejemplos = 'Añade palabras: "más oscuro", "menos denso", "algo como el 2 pero más grave"'
            print(f"  {dim(_ejemplos)}\n")
            refinement = input("  > ").strip()
            if refinement:
                seed_state["current_query"] = refinement
                seed_state["history"].append(("refine", refinement))
                new_base, new_weights = text_to_query_vector(refinement)
                if use_llm:
                    new_vec, new_wts = llm_enrich_query(refinement, new_base, new_weights,
                                                        provider=provider)
                else:
                    new_vec, new_wts = new_base, new_weights
                # Mezclar con el vector anterior (inercia)
                alpha = 0.4
                seed_state["query_vec"] = (
                    (1 - alpha) * seed_state["query_vec"] + alpha * new_vec
                )
                seed_state["weights"] = np.maximum(seed_state["weights"], new_wts)

        else:
            # Intentar parsear números
            nums = []
            for token in resp.replace(",", " ").split():
                if token.isdigit():
                    n_int = int(token)
                    if 1 <= n_int <= len(shown):
                        nums.append(n_int - 1)

            if nums:
                chosen = [shown[i] for i in nums]
                seed_state["chosen_indices"].extend(chosen)
                remaining = [shown[i] for i in range(len(shown)) if i not in nums]
                seed_state["rejected_indices"].extend(remaining)

                # Actualizar vector hacia los elegidos
                chosen_vecs = vectors[chosen]
                target_vec  = chosen_vecs.mean(axis=0)
                alpha       = 0.5
                seed_state["query_vec"] = (
                    (1 - alpha) * seed_state["query_vec"] + alpha * target_vec
                )
                # Aumentar peso de dimensiones donde los elegidos son distintos al resto
                if len(chosen) < len(shown) and remaining:
                    rejected_vecs = vectors[remaining]
                    diff          = np.abs(target_vec - rejected_vecs.mean(axis=0))
                    seed_state["weights"] = np.clip(
                        seed_state["weights"] + diff * 0.3, 0.1, 1.0
                    )

                print(f"  {green(f'✓ {len(chosen)} fragmento(s) seleccionado(s)')}")

                # Preguntar una sola pregunta de cristalización
                _ask_crystallization_question(seed_state, provider=provider)
            else:
                print(yellow("  No entendí. Prueba con un número (1-5) o una palabra clave."))

    # ── Resumen y generación ────────────────────────────────────────
    _show_seed_summary(seed_state, vectors, paths, meta)
    _generation_loop(seed_state, vectors, paths, args)


def _ask_crystallization_question(seed_state: dict, provider: str = "auto"):
    """
    Selecciona y hace UNA sola pregunta de cristalización basada en el
    estado actual de la sesión.
    """
    if _has_llm(provider) and len(seed_state["chosen_indices"]) >= 1:
        _ask_crystallization_llm(seed_state, provider=provider)
    else:
        _ask_crystallization_local(seed_state)


# ── Banco de preguntas ───────────────────────────────────────────────────────
#
# Estructura por dimensión:
#   dim_index  : índice en DIM_NAMES
#   condition  : función(vec_val) → bool  (cuándo aplica esta variante)
#   key        : clave de cristalización
#   min_chosen : mínimo de elegidos para activar esta pregunta
#   text       : pregunta al usuario
#   opts       : opciones de respuesta
#   vals       : valor a cristalizar (None = "no sé aún", no cristaliza)
#   dim_deltas : {dim_index: delta} para ajustar el vector si se elige esa opción
#   theorist_hint : sugerencia para el comando theorist final

QUESTION_BANK = [

    # ── CONTORNO ────────────────────────────────────────────────────────────
    {
        "dim_index": 3, "condition": lambda v: 0.3 < v < 0.7,
        "key": "contorno", "min_chosen": 1,
        "text": "El movimiento melódico que elegiste, ¿adónde tiende a ir?",
        "opts": ["Baja (como una caída, un descenso)",
                 "Sube (como tensión que se acumula)",
                 "Se queda en el mismo sitio / oscila",
                 "No lo sé todavía"],
        "vals": ["descendente", "ascendente", "plano/oscilante", None],
        "dim_deltas": [{3: -0.3}, {3: +0.3}, {3: 0.0, 5: +0.1}, {}],
        "theorist_hint": {"descendente": "--arc tragedy", "ascendente": "--arc hero"},
    },
    {
        "dim_index": 3, "condition": lambda v: v <= 0.3,
        "key": "contorno_confirmado", "min_chosen": 2,
        "text": "Los fragmentos que elegiste tienden a bajar. ¿Ese descenso es gradual o cae de golpe?",
        "opts": ["Gradual, por pasos",
                 "Cae de golpe, con saltos",
                 "Mezcla de los dos"],
        "vals": ["descenso_gradual", "descenso_abrupto", "descenso_mixto"],
        "dim_deltas": [{2: -0.2}, {2: +0.3}, {}],
        "theorist_hint": {},
    },

    # ── DENSIDAD ────────────────────────────────────────────────────────────
    {
        "dim_index": 4, "condition": lambda v: 0.3 < v < 0.7,
        "key": "densidad", "min_chosen": 1,
        "text": "¿Cuántas notas quieres que haya?",
        "opts": ["Pocas notas, mucho espacio",
                 "Flujo continuo, sin pausas grandes",
                 "Depende del momento de la pieza"],
        "vals": ["sparse", "continuo", "variable"],
        "dim_deltas": [{4: -0.3, 9: +0.2}, {4: +0.3, 9: -0.2}, {}],
        "theorist_hint": {},
    },
    {
        "dim_index": 9, "condition": lambda v: v > 0.5,
        "key": "silencio", "min_chosen": 1,
        "text": "Elegiste fragmentos con bastante silencio. ¿Ese silencio es intencionado o es algo a evitar?",
        "opts": ["Intencionado — el silencio es parte de la música",
                 "Preferiría más continuidad",
                 "Me es indiferente"],
        "vals": ["silencio_estructural", "continuo", None],
        "dim_deltas": [{9: +0.1}, {9: -0.3, 4: +0.2}, {}],
        "theorist_hint": {"silencio_estructural": "--arc meditation"},
    },

    # ── POLIFONÍA ───────────────────────────────────────────────────────────
    {
        "dim_index": 6, "condition": lambda v: 0.25 < v < 0.6,
        "key": "polifonia", "min_chosen": 1,
        "text": "¿Cuántas voces o líneas simultáneas imaginas en la pieza?",
        "opts": ["Una sola voz (melodía sola)",
                 "Dos voces (melodía + bajo, diálogo)",
                 "Bloque completo (coral, cuarteto)",
                 "Varía a lo largo de la pieza"],
        "vals": ["monofónico", "dos_voces", "polifónico_completo", "variable"],
        "dim_deltas": [{6: -0.3}, {6: +0.1}, {6: +0.4}, {}],
        "theorist_hint": {"polifónico_completo": "--persona fux"},
    },
    {
        "dim_index": 6, "condition": lambda v: v > 0.6,
        "key": "movimiento_voces", "min_chosen": 2,
        "text": "En los fragmentos polifónicos que elegiste, ¿las voces se mueven juntas o de forma independiente?",
        "opts": ["Juntas (homofónico — como un coral)",
                 "Independientes (contrapunto — cada voz va por su lado)",
                 "Mezcla: a veces juntas, a veces independientes"],
        "vals": ["homofónico", "contrapuntístico", "mixto"],
        "dim_deltas": [{5: -0.2}, {5: +0.2}, {}],
        "theorist_hint": {"contrapuntístico": "--persona fux"},
    },

    # ── REGISTRO ────────────────────────────────────────────────────────────
    {
        "dim_index": 0, "condition": lambda v: 0.35 < v < 0.65,
        "key": "registro", "min_chosen": 1,
        "text": "¿En qué zona del espectro sonoro quieres que viva la pieza principalmente?",
        "opts": ["Grave (chelo, contrabajo, piano bajo)",
                 "Medio (viola, clarinete, voz)",
                 "Agudo (violín, flauta, voz soprano)",
                 "Amplio — recorre todo el rango"],
        "vals": ["grave", "medio", "agudo", "amplio"],
        "dim_deltas": [{0: -0.3}, {0: 0.0}, {0: +0.3}, {1: +0.3}],
        "theorist_hint": {"grave": "--key Dm", "agudo": "--key G"},
    },
    {
        "dim_index": 1, "condition": lambda v: v < 0.3,
        "key": "rango_estrecho", "min_chosen": 2,
        "text": "Elegiste fragmentos con notas muy cercanas entre sí. ¿Esa restricción es parte de la idea?",
        "opts": ["Sí — quiero que todo esté concentrado en pocas notas",
                 "No — prefiero más variedad de alturas",
                 "No lo había notado"],
        "vals": ["micromelódico", "amplio", None],
        "dim_deltas": [{1: -0.1}, {1: +0.3}, {}],
        "theorist_hint": {},
    },

    # ── RITMO ───────────────────────────────────────────────────────────────
    {
        "dim_index": 5, "condition": lambda v: 0.3 < v < 0.7,
        "key": "ritmo", "min_chosen": 1,
        "text": "¿Cómo quieres que sea el ritmo?",
        "opts": ["Regular y predecible (pulso claro)",
                 "Libre e irregular (sin pulso fijo)",
                 "Empieza regular y se fragmenta",
                 "No tengo preferencia clara"],
        "vals": ["regular", "libre", "fragmentación_gradual", None],
        "dim_deltas": [{5: -0.3}, {5: +0.3}, {5: +0.1}, {}],
        "theorist_hint": {"libre": "--arc meditation", "fragmentación_gradual": "--arc tragedy"},
    },
    {
        "dim_index": 5, "condition": lambda v: v > 0.65,
        "key": "irregularidad", "min_chosen": 2,
        "text": "Los fragmentos que elegiste tienen ritmo muy irregular. ¿Esa irregularidad viene de...?",
        "opts": ["Silencios inesperados",
                 "Cambios de velocidad (accelerando/rallentando)",
                 "Métricas irregulares (5/4, 7/8...)",
                 "No lo sé, me gusta el resultado"],
        "vals": ["silencios_rítmicos", "tempo_variable", "métrica_irregular", "irregular_libre"],
        "dim_deltas": [{9: +0.1}, {8: +0.2}, {5: +0.1}, {}],
        "theorist_hint": {},
    },

    # ── DINÁMICA ────────────────────────────────────────────────────────────
    {
        "dim_index": 7, "condition": lambda v: 0.3 < v < 0.7,
        "key": "dinámica", "min_chosen": 1,
        "text": "¿En qué nivel dinámico vive la pieza?",
        "opts": ["Suave y contenido (pp-mp)",
                 "Fuerte y presente (mf-ff)",
                 "Con grandes contrastes dinámicos",
                 "No lo tengo claro"],
        "vals": ["suave", "fuerte", "contrastado", None],
        "dim_deltas": [{7: -0.3}, {7: +0.3}, {8: +0.4}, {}],
        "theorist_hint": {},
    },
    {
        "dim_index": 8, "condition": lambda v: v > 0.6,
        "key": "expresividad", "min_chosen": 2,
        "text": "Elegiste fragmentos muy expresivos dinámicamente. ¿Esa expresividad es...?",
        "opts": ["Continua — siempre hay movimiento de intensidad",
                 "Puntual — momentos de climax sobre fondo estable",
                 "Un arco — crece hacia un punto y luego baja"],
        "vals": ["expresivo_continuo", "climax_puntual", "arco_dinámico"],
        "dim_deltas": [{8: +0.1}, {8: +0.2}, {}],
        "theorist_hint": {"arco_dinámico": "--arc hero"},
    },

    # ── FORMA / EVOLUCIÓN ───────────────────────────────────────────────────
    {
        "dim_index": -1, "condition": lambda v: True,  # siempre aplica
        "key": "evolución", "min_chosen": 2,
        "text": "¿La pieza cambia a lo largo del tiempo o permanece en un estado?",
        "opts": ["Evoluciona — hay un arco, un viaje",
                 "Es estática — medita en el mismo lugar",
                 "Alterna — secciones de tensión y reposo",
                 "No lo sé aún"],
        "vals": ["evolutiva", "estática", "alternante", None],
        "dim_deltas": [{}, {}, {}, {}],
        "theorist_hint": {
            "evolutiva":   "--arc hero",
            "estática":    "--arc meditation",
            "alternante":  "--arc rondo",
        },
    },
    {
        "dim_index": -1, "condition": lambda v: True,
        "key": "resolución", "min_chosen": 2,
        "text": "Al final de la pieza, ¿quieres que el oyente se quede con...?",
        "opts": ["Sensación de cierre (llegamos a algún lugar)",
                 "Suspensión (la pregunta queda abierta)",
                 "Ambigüedad (no está claro si resolvió o no)",
                 "No lo he pensado"],
        "vals": ["cierre", "suspensión", "ambigüedad", None],
        "dim_deltas": [{}, {9: +0.1}, {5: +0.1, 9: +0.1}, {}],
        "theorist_hint": {
            "suspensión":  "--arc meditation",
            "ambigüedad":  "--dialectical",
        },
    },
    {
        "dim_index": -1, "condition": lambda v: True,
        "key": "duración", "min_chosen": 1,
        "text": "¿Tienes intuición sobre la duración de la pieza?",
        "opts": ["Breve — un momento (1-3 minutos)",
                 "Media — una pieza completa (4-8 minutos)",
                 "Larga — una obra extensa (10+ minutos)",
                 "Sin límite — lo que necesite"],
        "vals": ["breve", "media", "larga", "libre"],
        "dim_deltas": [{}, {}, {}, {}],
        "theorist_hint": {
            "breve": "--bars 24",
            "media": "--bars 64",
            "larga": "--bars 128",
        },
    },

    # ── CARÁCTER / FUNCIÓN ──────────────────────────────────────────────────
    {
        "dim_index": -1, "condition": lambda v: True,
        "key": "función", "min_chosen": 1,
        "text": "¿Para qué contexto imaginas esta pieza?",
        "opts": ["Sala de concierto — obra autónoma para escucha activa",
                 "Instalación — sonido ambiental, se puede entrar y salir",
                 "Cine / imagen — acompaña algo visual",
                 "No lo sé, la pieza manda"],
        "vals": ["concierto", "instalación", "cine", None],
        "dim_deltas": [{}, {4: -0.2, 9: +0.2}, {8: +0.1}, {}],
        "theorist_hint": {},
    },
    {
        "dim_index": -1, "condition": lambda v: True,
        "key": "referencia_emocional", "min_chosen": 3,
        "text": "Si tuvieras que describir el estado emocional que quieres provocar en una sola palabra, ¿cuál sería?",
        "opts": ["Melancolía",
                 "Tensión / expectativa",
                 "Calma / contemplación",
                 "Asombro",
                 "Otra (escribe)"],
        "vals": ["melancolía", "tensión", "calma", "asombro", "__input__"],
        "dim_deltas": [
            {0: -0.1, 4: -0.2, 9: +0.1},
            {5: +0.2, 8: +0.2},
            {4: -0.2, 5: -0.2},
            {1: +0.2, 8: +0.2},
            {},
        ],
        "theorist_hint": {
            "melancolía":  "--key Dm --persona romantic",
            "tensión":     "--arc tragedy",
            "calma":       "--arc meditation --persona modal",
        },
    },
]


def _analyze_session_state(seed_state: dict, vectors) -> dict:
    """
    Analiza el estado de la sesión para entender qué dimensiones
    discriminan entre elegidos y rechazados, y cuáles tienen alta
    varianza entre los elegidos (aún no decididas).

    Devuelve:
      discriminating: list de (dim_index, importancia) — qué usó el usuario para elegir
      uncertain:      list de (dim_index, varianza)    — qué no está claro aún
      current_vals:   np.array con los valores actuales del vector
    """
    chosen   = seed_state["chosen_indices"]
    rejected = seed_state["rejected_indices"]

    current_vals = seed_state["query_vec"]

    if not chosen:
        return {"discriminating": [], "uncertain": [], "current_vals": current_vals}

    chosen_vecs   = vectors[chosen]
    chosen_mean   = chosen_vecs.mean(axis=0)

    # Discriminación: diferencia entre elegidos y rechazados por dimensión
    discriminating = []
    if rejected:
        rejected_vecs = vectors[rejected]
        rejected_mean = rejected_vecs.mean(axis=0)
        diff = np.abs(chosen_mean - rejected_mean)
        for i, d in enumerate(diff):
            if d > 0.1:
                discriminating.append((i, float(d)))
        discriminating.sort(key=lambda x: -x[1])

    # Incertidumbre: varianza interna entre los elegidos
    uncertain = []
    if len(chosen) >= 2:
        chosen_var = chosen_vecs.var(axis=0)
        for i, v in enumerate(chosen_var):
            if v > 0.03:  # umbral mínimo de varianza
                uncertain.append((i, float(v)))
        uncertain.sort(key=lambda x: -x[1])

    return {
        "discriminating": discriminating,
        "uncertain":      uncertain,
        "current_vals":   current_vals,
    }


def _select_best_question(seed_state: dict, analysis: dict) -> dict | None:
    """
    Selecciona la mejor pregunta del banco dado el estado de la sesión.

    Criterios de selección (en orden de prioridad):
    1. La pregunta toca una dimensión con alta incertidumbre entre los elegidos
    2. La pregunta toca una dimensión discriminante (el usuario la está usando)
    3. La clave no está ya cristalizada
    4. El mínimo de elegidos se cumple
    5. La condición sobre el valor actual de la dimensión se cumple
    """
    crystallized  = seed_state["crystallized"]
    n_chosen      = len(seed_state["chosen_indices"])
    current_vals  = analysis["current_vals"]

    uncertain_dims      = {d: v for d, v in analysis["uncertain"]}
    discriminating_dims = {d: v for d, v in analysis["discriminating"]}

    scored = []
    for q in QUESTION_BANK:
        # Filtros duros
        if q["key"] in crystallized:
            continue
        if n_chosen < q["min_chosen"]:
            continue

        dim_idx = q["dim_index"]
        dim_val = float(current_vals[dim_idx]) if dim_idx >= 0 else 0.5
        if not q["condition"](dim_val):
            continue

        # Puntuación
        score = 0.0
        if dim_idx >= 0:
            score += uncertain_dims.get(dim_idx, 0.0) * 3.0       # incertidumbre pesa más
            score += discriminating_dims.get(dim_idx, 0.0) * 2.0  # discriminación pesa
        else:
            # Preguntas globales (dim_index == -1): puntúan por número de elegidos
            score += min(n_chosen / 5.0, 1.0)

        # Bonus si toca dimensión muy incierta
        if dim_idx >= 0 and uncertain_dims.get(dim_idx, 0) > 0.08:
            score += 1.0

        scored.append((score, q))

    if not scored:
        return None

    # Elegir la de mayor puntuación (con algo de aleatoriedad para no repetir siempre la misma)
    scored.sort(key=lambda x: -x[0])
    # Entre las 3 mejores, elegir al azar para variedad
    top = scored[:3]
    _, best = random.choice(top)
    return best


def _ask_crystallization_local(seed_state: dict):
    """
    Versión local (sin LLM) de la cristalización.
    Analiza el estado de la sesión y selecciona la pregunta más útil del banco.
    """
    if not HAS_NUMPY or seed_state["query_vec"] is None:
        return

    # Necesitamos los vectores — los tomamos del estado si están disponibles
    # En este punto seed_state no tiene referencia directa a vectors,
    # así que trabajamos con el query_vec como proxy del estado actual
    chosen_indices = seed_state.get("chosen_indices", [])
    if not chosen_indices:
        return

    # Análisis simplificado usando solo el query_vec y los metadatos del estado
    current_vals = seed_state["query_vec"]

    # Incertidumbre: dimensiones del query_vec cerca de 0.5 (sin señal clara)
    uncertain_dims = {}
    for i, v in enumerate(current_vals):
        distance_from_center = abs(v - 0.5)
        if distance_from_center < 0.15:  # zona ambigua
            uncertain_dims[i] = 1.0 - distance_from_center * 6

    # Discriminación: aproximamos con las dimensiones que más se alejaron del centro
    discriminating_dims = {}
    for i, v in enumerate(current_vals):
        distance_from_center = abs(v - 0.5)
        if distance_from_center > 0.2:
            discriminating_dims[i] = distance_from_center * 2

    analysis = {
        "discriminating": list(discriminating_dims.items()),
        "uncertain":      list(uncertain_dims.items()),
        "current_vals":   current_vals,
    }

    q = _select_best_question(seed_state, analysis)
    if q is None:
        return

    _present_and_record_question(q, seed_state)


def _ask_crystallization_llm(seed_state: dict, provider: str = "auto"):
    """
    Versión LLM de la cristalización.
    El modelo lee el estado completo y decide qué preguntar,
    eligiendo del banco o formulando algo más específico.
    """
    crystallized = seed_state["crystallized"]
    n_chosen     = len(seed_state["chosen_indices"])
    current_vals = seed_state["query_vec"]

    profile   = describe_vector(current_vals)
    cryst_str = json.dumps(crystallized, ensure_ascii=False) if crystallized else "ninguna"

    available = [
        q for q in QUESTION_BANK
        if q["key"] not in crystallized and n_chosen >= q["min_chosen"]
    ]
    available_keys = [q["key"] for q in available]

    system = """Eres un asistente de composición musical que ayuda a un compositor
a clarificar su intención creativa a través de preguntas precisas.

Tu tarea es elegir UNA sola pregunta del banco disponible (o formular una variante
más específica) basándote en el estado actual de la sesión.

Criterios: elige la pregunta que más aclare lo que aún es ambiguo dado lo que
el compositor ha elegido. Evita preguntar lo que ya está implícito en el perfil.

Responde SOLO con JSON válido:
{
  "key": "clave_de_la_pregunta",
  "text": "texto de la pregunta (puedes adaptar el tono al contexto)",
  "opts": ["opción 1", "opción 2", "opción 3", "No lo sé aún"],
  "vals": ["val1", "val2", "val3", null],
  "reasoning": "por qué esta pregunta ahora (una frase)"
}"""

    user_msg = f"""Estado de la sesión:
- Intención original: "{seed_state['original_intent']}"
- Perfil musical emergente: {profile}
- Fragmentos elegidos: {n_chosen}
- Ya cristalizado: {cryst_str}
- Preguntas disponibles: {available_keys}

Elige la pregunta más útil en este momento."""

    try:
        raw  = _llm_call(system, user_msg, provider=provider, max_tokens=400)
        data = json.loads(raw)

        q = {
            "key":           data["key"],
            "text":          data["text"],
            "opts":          data["opts"],
            "vals":          data["vals"],
            "dim_deltas":    [{} for _ in data["opts"]],
            "theorist_hint": {},
        }

        for bank_q in QUESTION_BANK:
            if bank_q["key"] == data["key"]:
                q["dim_deltas"]    = bank_q["dim_deltas"]
                q["theorist_hint"] = bank_q["theorist_hint"]
                break

        if "reasoning" in data:
            print(f"  {dim('→ ' + data['reasoning'])}")

        _present_and_record_question(q, seed_state)

    except Exception:
        _ask_crystallization_local(seed_state)


def _present_and_record_question(q: dict, seed_state: dict):
    """
    Presenta la pregunta al usuario, recoge la respuesta,
    cristaliza el valor y ajusta el vector de búsqueda.
    """
    print(f"\n  {bold('Una pregunta:')}")
    print(f"  {q['text']}\n")

    for i, opt in enumerate(q["opts"], 1):
        print(f"    [{i}] {opt}")

    resp = input("\n  > ").strip()

    # Aceptar número o texto parcial
    chosen_idx = None
    if resp.isdigit():
        n = int(resp) - 1
        if 0 <= n < len(q["opts"]):
            chosen_idx = n
    else:
        # Buscar coincidencia parcial en las opciones
        resp_lower = resp.lower()
        for i, opt in enumerate(q["opts"]):
            if resp_lower in opt.lower():
                chosen_idx = i
                break

    if chosen_idx is None:
        print(f"  {yellow('No entendí la respuesta — pregunta omitida.')}")
        return

    val = q["vals"][chosen_idx]

    # Caso especial: input libre
    if val == "__input__":
        val = input("  Escribe: ").strip() or None

    if val is not None:
        seed_state["crystallized"][q["key"]] = val
        key_label = q["key"]
        print(f"  {green(f'✓ {key_label}: {val}')}")

        # Añadir hint de theorist si aplica
        hint = q.get("theorist_hint", {}).get(val, "")
        if hint:
            if "theorist_hints" not in seed_state:
                seed_state["theorist_hints"] = []
            seed_state["theorist_hints"].append(hint)

    # Ajustar vector de búsqueda según la respuesta
    deltas = q.get("dim_deltas", [{}] * len(q["opts"]))
    if chosen_idx < len(deltas) and deltas[chosen_idx]:
        qv = seed_state["query_vec"]
        for dim_i, delta in deltas[chosen_idx].items():
            qv[dim_i] = float(np.clip(qv[dim_i] + delta, 0.0, 1.0))
        seed_state["query_vec"] = qv


# ════════════════════════════════════════════════════════════════════
#  INSTRUMENTACIÓN — grupos con carácter musical y programas GM
# ════════════════════════════════════════════════════════════════════

# Cada grupo: nombre, descripción, lista de (program_GM, nombre_instrumento, canal)
# canal 9 reservado para percusión — no se usa aquí
INSTRUMENT_GROUPS = {
    "cuerdas_solas": {
        "label": "Cuerdas solas (cuarteto)",
        "desc":  "Violín I, Violín II, Viola, Chelo — íntimo y expresivo",
        "voices": [
            (40, "Violín I",  0),
            (40, "Violín II", 1),
            (41, "Viola",     2),
            (42, "Chelo",     3),
        ],
        "register_bias": 0.55,   # tendencia de registro (0=grave, 1=agudo)
        "polyphony_bias": 0.5,
    },
    "cuerdas_orquesta": {
        "label": "Cuerdas de orquesta",
        "desc":  "Cuerdas en sección — más presencia y cuerpo",
        "voices": [
            (48, "Cuerdas agudas", 0),
            (48, "Cuerdas medias", 1),
            (43, "Chelo sección",  2),
            (43, "Contrabajo",     3),
        ],
        "register_bias": 0.5,
        "polyphony_bias": 0.6,
    },
    "viento_madera": {
        "label": "Viento madera",
        "desc":  "Flauta, Oboe, Clarinete, Fagot — lírico y transparente",
        "voices": [
            (73, "Flauta",    0),
            (68, "Oboe",      1),
            (71, "Clarinete", 2),
            (70, "Fagot",     3),
        ],
        "register_bias": 0.6,
        "polyphony_bias": 0.35,
    },
    "piano_solo": {
        "label": "Piano solo",
        "desc":  "Piano solo — amplio rango, polifónico por naturaleza",
        "voices": [
            (0, "Piano derecha", 0),
            (0, "Piano izquierda", 1),
        ],
        "register_bias": 0.5,
        "polyphony_bias": 0.5,
    },
    "camara_mixta": {
        "label": "Cámara mixta",
        "desc":  "Piano + cuerdas — combinación expresiva y versátil",
        "voices": [
            (0,  "Piano",   0),
            (40, "Violín",  1),
            (41, "Viola",   2),
            (42, "Chelo",   3),
        ],
        "register_bias": 0.5,
        "polyphony_bias": 0.55,
    },
    "voz_cuerdas": {
        "label": "Voz y cuerdas",
        "desc":  "Soprano/tenor + cuarteto — íntimo, cantábile",
        "voices": [
            (52, "Voz",    0),
            (40, "Violín", 1),
            (41, "Viola",  2),
            (42, "Chelo",  3),
        ],
        "register_bias": 0.6,
        "polyphony_bias": 0.4,
    },
    "electroacustico": {
        "label": "Electroacústico",
        "desc":  "Pad + texturas sintéticas — difuso, ambiental",
        "voices": [
            (88, "Pad cálido",   0),
            (89, "Pad frío",     1),
            (95, "Glide",        2),
            (92, "Pad cristal",  3),
        ],
        "register_bias": 0.5,
        "polyphony_bias": 0.7,
    },
    "solo_melodico": {
        "label": "Solo melódico",
        "desc":  "Una sola voz — violín, flauta o chelo según el registro",
        "voices": [
            (40, "Solo", 0),   # se ajusta según registro
        ],
        "register_bias": 0.55,
        "polyphony_bias": 0.1,
    },
}

# Pregunta de instrumentación para el banco
INSTRUMENT_QUESTION = {
    "dim_index": -1,
    "condition": lambda v: True,
    "key": "instrumentación",
    "min_chosen": 1,
    "text": "¿Qué tipo de sonido imaginas para esta pieza?",
    "opts": [
        "Cuerdas solas (cuarteto) — íntimo y expresivo",
        "Cuerdas de orquesta — más presencia y cuerpo",
        "Viento madera — lírico y transparente",
        "Piano solo — amplio rango",
        "Cámara mixta (piano + cuerdas)",
        "Voz y cuerdas",
        "Electroacústico — pads y texturas",
        "Una sola voz melódica",
        "No lo sé aún",
    ],
    "vals": [
        "cuerdas_solas", "cuerdas_orquesta", "viento_madera",
        "piano_solo", "camara_mixta", "voz_cuerdas",
        "electroacustico", "solo_melodico", None,
    ],
    "dim_deltas": [{} for _ in range(9)],
    "theorist_hint": {},
}


def _infer_instrumentation(seed_state: dict) -> str:
    """
    Infiere el grupo de instrumentación más probable a partir del perfil
    vectorial si no está cristalizado explícitamente.
    """
    cryst = seed_state["crystallized"]
    if "instrumentación" in cryst:
        return cryst["instrumentación"]

    qv = seed_state["query_vec"]

    # Reglas de inferencia basadas en el perfil
    polyphony = qv[6]
    register  = qv[0]
    density   = qv[4]
    silence   = qv[9]

    if polyphony < 0.15:
        return "solo_melodico"

    if density < 0.2 and silence > 0.5:
        return "electroacustico"

    if register < 0.35:
        return "cuerdas_orquesta"

    if polyphony < 0.35:
        return "viento_madera"

    if polyphony > 0.55:
        return "cuerdas_solas"

    return "camara_mixta"


# ════════════════════════════════════════════════════════════════════
#  GENERADOR MUSICAL — extrae perfil y genera MIDI original
# ════════════════════════════════════════════════════════════════════

# Escalas como intervalos desde la tónica (semitonos)
SCALES = {
    "mayor":         [0, 2, 4, 5, 7, 9, 11],
    "menor_natural": [0, 2, 3, 5, 7, 8, 10],
    "menor_armónica":[0, 2, 3, 5, 7, 8, 11],
    "dórico":        [0, 2, 3, 5, 7, 9, 10],
    "frigio":        [0, 1, 3, 5, 7, 8, 10],
    "frigio_mayor":  [0, 1, 4, 5, 7, 8, 10],
    "lidio":         [0, 2, 4, 6, 7, 9, 11],
    "mixolidio":     [0, 2, 4, 5, 7, 9, 10],
    "locrio":        [0, 1, 3, 5, 6, 8, 10],
    "pentatónica_m": [0, 3, 5, 7, 10],
    "pentatónica_M": [0, 2, 4, 7, 9],
    "blues":         [0, 3, 5, 6, 7, 10],
    "cromática":     list(range(12)),
}

# Tónicas MIDI (octava 4 como referencia)
TONICS = {
    "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
    "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
    "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71,
}


def _extract_generative_profile(seed_state: dict, vectors, paths_arr) -> dict:
    """
    Extrae el perfil generativo de los fragmentos elegidos.
    Combina análisis de vectores con las decisiones cristalizadas.
    """
    chosen = seed_state["chosen_indices"]
    cryst  = seed_state["crystallized"]
    qv     = seed_state["query_vec"]

    # Vector promedio de los elegidos
    if chosen:
        chosen_mean = vectors[chosen].mean(axis=0)
    else:
        chosen_mean = qv

    # ── Escala ──────────────────────────────────────────────────────
    # Inferir desde el carácter emocional cristalizado o el registro
    emotion   = cryst.get("referencia_emocional", "")
    resolución = cryst.get("resolución", cryst.get("resolution", ""))
    contorno  = cryst.get("contorno", "")

    scale_name = "menor_natural"   # default
    if emotion in ("melancolía", "tensión"):
        scale_name = "frigio" if chosen_mean[0] < 0.4 else "menor_armónica"
    elif emotion == "calma":
        scale_name = "dórico"
    elif emotion == "asombro":
        scale_name = "lidio"
    elif resolución == "ambigüedad":
        scale_name = "locrio"
    elif resolución == "suspensión":
        scale_name = "frigio_mayor"

    # Si hay fragmentos del corpus, intentar detectar la escala predominante
    if chosen and len(chosen) >= 2:
        scale_name = _detect_scale_from_corpus(paths_arr, chosen) or scale_name

    # ── Tónica ──────────────────────────────────────────────────────
    register = float(chosen_mean[0])
    if register < 0.35:
        tonic_name, tonic_midi = "D", 50   # Re grave
    elif register < 0.5:
        tonic_name, tonic_midi = "A", 57   # La medio-grave
    elif register < 0.65:
        tonic_name, tonic_midi = "E", 64   # Mi medio
    else:
        tonic_name, tonic_midi = "G", 67   # Sol agudo

    # Override si hay cristalización explícita de registro
    reg_cryst = cryst.get("registro", "")
    if reg_cryst == "grave":
        tonic_name, tonic_midi = "D", 50
    elif reg_cryst == "agudo":
        tonic_name, tonic_midi = "G", 67
    elif reg_cryst == "medio":
        tonic_name, tonic_midi = "A", 57

    # ── Duración ────────────────────────────────────────────────────
    dur_cryst = cryst.get("duración", "media")
    bars = {"breve": 20, "media": 32, "larga": 64, "libre": 40}.get(dur_cryst, 32)

    # ── Tempo ───────────────────────────────────────────────────────
    density = float(chosen_mean[4])
    if density < 0.15:
        tempo = random.randint(42, 58)
    elif density < 0.35:
        tempo = random.randint(58, 76)
    elif density < 0.6:
        tempo = random.randint(76, 100)
    else:
        tempo = random.randint(100, 132)

    # ── Densidad rítmica ─────────────────────────────────────────────
    notes_per_bar = max(1, int(density * 12))   # 1-12 notas por compás

    # ── Irregularidad rítmica ────────────────────────────────────────
    rhythm_var = float(chosen_mean[5])

    # ── Silencio ────────────────────────────────────────────────────
    silence_ratio = float(chosen_mean[9])

    # ── Rango de alturas ────────────────────────────────────────────
    pitch_range_norm = float(chosen_mean[1])
    pitch_range_semi = int(pitch_range_norm * 24 + 6)  # 6-30 semitonos

    # ── Contorno ────────────────────────────────────────────────────
    contour_val = float(chosen_mean[3])
    if contorno == "descendente":    contour_val = 0.2
    elif contorno == "ascendente":   contour_val = 0.8
    elif contorno == "plano/oscilante": contour_val = 0.5

    # ── Dinámica ────────────────────────────────────────────────────
    vel_mean = int(chosen_mean[7] * 80 + 30)   # 30-110
    vel_var  = int(chosen_mean[8] * 40)         # 0-40

    return {
        "scale_name":      scale_name,
        "scale":           SCALES[scale_name],
        "tonic_name":      tonic_name,
        "tonic_midi":      tonic_midi,
        "bars":            bars,
        "tempo":           tempo,
        "notes_per_bar":   notes_per_bar,
        "rhythm_variance": rhythm_var,
        "silence_ratio":   silence_ratio,
        "pitch_range":     pitch_range_semi,
        "contour":         contour_val,
        "vel_mean":        vel_mean,
        "vel_var":         vel_var,
        "polyphony":       float(chosen_mean[6]),
        "resolución":      resolución,
    }


def _detect_scale_from_corpus(paths_arr, chosen_indices: list) -> str | None:
    """
    Detecta la escala predominante analizando los fragmentos elegidos.
    Usa distribución de clases de altura (pitch class profile).
    """
    if not HAS_MIDO:
        return None

    pitch_counts = Counter()

    for idx in chosen_indices[:3]:   # máximo 3 para no tardar
        try:
            mid = mido.MidiFile(str(paths_arr[idx]), clip=True)
            for track in mid.tracks:
                for msg in track:
                    if msg.type == "note_on" and msg.velocity > 0:
                        pitch_counts[msg.note % 12] += 1
        except Exception:
            continue

    if not pitch_counts:
        return None

    # Perfil de clases de pitch
    total = sum(pitch_counts.values())
    profile = [pitch_counts.get(i, 0) / total for i in range(12)]

    # Comparar con plantillas de escalas
    best_scale, best_score = None, -1.0
    for scale_name, intervals in SCALES.items():
        if scale_name in ("cromática",):
            continue
        for root in range(12):
            template = [1.0 if (i - root) % 12 in intervals else 0.0 for i in range(12)]
            score    = sum(p * t for p, t in zip(profile, template))
            if score > best_score:
                best_score = score
                best_scale = scale_name

    return best_scale if best_score > 0.4 else None


def _generate_voice(profile: dict, voice_idx: int,
                    n_voices: int, seed: int = 0) -> list:
    """
    Genera una voz melódica como lista de (pitch_midi, duration_ticks, velocity).
    Cada voz tiene un rol distinto según su índice.
    """
    rng = random.Random(seed + voice_idx * 137)

    scale      = profile["scale"]
    tonic      = profile["tonic_midi"]
    bars       = profile["bars"]
    npb        = profile["notes_per_bar"]
    rvar       = profile["rhythm_variance"]
    silence    = profile["silence_ratio"]
    contour    = profile["contour"]
    pitch_range= profile["pitch_range"]
    vel_mean   = profile["vel_mean"]
    vel_var    = profile["vel_var"]
    resolución = profile["resolución"]

    ticks_per_bar = 1920   # resolución interna alta

    # Registro por voz
    if n_voices == 1:
        octave_offset = 0
    elif voice_idx == 0:   # melodía / soprano
        octave_offset = 12
    elif voice_idx == 1:   # segunda voz / alto
        octave_offset = 0
    elif voice_idx == 2:   # tenor / viola
        octave_offset = -12
    else:                  # bajo / chelo
        octave_offset = -24

    # Ajuste adicional de registro
    register_semitones = int((profile.get("polyphony", 0.5) - 0.5) * 12)
    base_pitch = tonic + octave_offset + register_semitones

    # Construir notas de escala disponibles en el rango
    def scale_notes_in_range(center, span):
        notes = []
        for oct_shift in range(-3, 4):
            for interval in scale:
                note = center + oct_shift * 12 + interval
                if center - span // 2 <= note <= center + span // 2:
                    if 21 <= note <= 108:
                        notes.append(note)
        return sorted(set(notes))

    available = scale_notes_in_range(base_pitch, pitch_range)
    if not available:
        available = [base_pitch]

    # Posición inicial: cerca del centro del rango disponible
    current_pitch_idx = len(available) // 2

    events = []   # (pitch, ticks, velocity) — pitch=0 es silencio

    total_ticks = bars * ticks_per_bar

    # Duraciones base en ticks (negra=480, blanca=960, redonda=1920, corchea=240)
    base_durations = [240, 480, 720, 960, 1440, 1920]

    tick = 0
    while tick < total_ticks:
        remaining = total_ticks - tick

        # ── Silencio estructural ─────────────────────────────────
        if rng.random() < silence * 0.4:
            dur = rng.choice([480, 960, 1440])
            dur = min(dur, remaining)
            events.append((0, dur, 0))
            tick += dur
            continue

        # ── Nota ─────────────────────────────────────────────────
        # Contorno: sesgar la dirección del movimiento
        move_up_prob = contour   # 0=siempre baja, 1=siempre sube

        # Movimiento por grados o salto
        if rvar > 0.6 and rng.random() < 0.3:
            # Salto
            jump = rng.choice([-7, -5, -4, 5, 7])
            new_idx = current_pitch_idx + jump
        else:
            # Grado
            direction = 1 if rng.random() < move_up_prob else -1
            step      = rng.choice([1, 1, 1, 2])  # más pasos que saltos
            new_idx   = current_pitch_idx + direction * step

        # Mantener dentro del rango
        new_idx = max(0, min(len(available) - 1, new_idx))
        current_pitch_idx = new_idx
        pitch = available[new_idx]

        # ── Resolución al final ──────────────────────────────────
        if tick > total_ticks * 0.85 and resolución == "cierre":
            # Acercar a la tónica
            tonic_candidates = [n for n in available if n % 12 == tonic % 12]
            if tonic_candidates:
                pitch = min(tonic_candidates, key=lambda n: abs(n - pitch))
                current_pitch_idx = available.index(pitch)

        # ── Duración ─────────────────────────────────────────────
        # Más irregular si rvar es alto
        if rvar > 0.5:
            weights = [3, 4, 2, 3, 1, 1]
        else:
            weights = [1, 5, 2, 4, 1, 0]

        dur_choices = base_durations[:len(weights)]
        # Selección ponderada manual
        total_w = sum(weights)
        r       = rng.random() * total_w
        cumul   = 0
        dur     = dur_choices[-1]
        for d, w in zip(dur_choices, weights):
            cumul += w
            if r <= cumul:
                dur = d
                break

        dur = min(dur, remaining)

        # ── Velocidad ─────────────────────────────────────────────
        vel = int(vel_mean + rng.gauss(0, vel_var))
        vel = max(20, min(120, vel))

        # Acento en tiempo fuerte
        if tick % ticks_per_bar < 120:
            vel = min(120, vel + 10)

        events.append((pitch, dur, vel))
        tick += dur

    return events


def _build_midi(profile: dict, instrument_group: dict, seed: int = 42) -> mido.MidiFile:
    """
    Construye el archivo MIDI completo a partir del perfil y la instrumentación.
    """
    voices_def  = instrument_group["voices"]
    n_voices    = len(voices_def)
    ticks_per_beat = 480
    ticks_per_bar  = ticks_per_beat * 4   # 4/4

    # Ajustar ticks del perfil
    profile_ticks = dict(profile)
    # notas_per_bar ya es int, durations en ticks se calculan en _generate_voice

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    tempo_us = int(60_000_000 / profile["tempo"])

    for v_idx, (program, v_name, channel) in enumerate(voices_def):
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Nombre de pista
        track.append(mido.MetaMessage("track_name", name=v_name, time=0))

        # Tempo solo en primera pista
        if v_idx == 0:
            track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))
            track.append(mido.MetaMessage("time_signature",
                                          numerator=4, denominator=4,
                                          clocks_per_click=24,
                                          notated_32nd_notes_per_beat=8, time=0))

        # Instrumento GM
        # Ajuste de instrumento para solo_melodico según registro
        prog = program
        if instrument_group == INSTRUMENT_GROUPS.get("solo_melodico"):
            reg = profile["tonic_midi"]
            if reg < 55:
                prog = 42   # chelo
            elif reg > 65:
                prog = 73   # flauta
            else:
                prog = 40   # violín

        track.append(mido.Message("program_change", channel=channel,
                                  program=prog, time=0))

        # Generar eventos de voz
        events = _generate_voice(profile, v_idx, n_voices,
                                  seed=seed + v_idx * 31)

        # Convertir a mensajes MIDI
        # Escalar ticks: _generate_voice usa ticks_per_bar=1920, nosotros usamos 4*480=1920 → OK
        for pitch, dur_ticks, vel in events:
            if pitch == 0 or vel == 0:
                # Silencio: no emitir nota, solo avanzar el tiempo
                # Usamos un note_on con vel=0 en nota fantasma para mantener el tiempo
                track.append(mido.Message("note_on",  channel=channel,
                                          note=0, velocity=0, time=0))
                track.append(mido.Message("note_off", channel=channel,
                                          note=0, velocity=0, time=dur_ticks))
            else:
                note_dur = max(int(dur_ticks * 0.92), 60)  # articulación: 8% de espacio
                gap      = dur_ticks - note_dur
                track.append(mido.Message("note_on",  channel=channel,
                                          note=pitch, velocity=vel, time=0))
                track.append(mido.Message("note_off", channel=channel,
                                          note=pitch, velocity=0,   time=note_dur))
                if gap > 0:
                    track.append(mido.Message("note_on",  channel=channel,
                                              note=0, velocity=0, time=0))
                    track.append(mido.Message("note_off", channel=channel,
                                              note=0, velocity=0, time=gap))

        track.append(mido.MetaMessage("end_of_track", time=0))

    return mid


def _play_with_timidity(midi_path: str, soundfont: str = None):
    """Reproduce un MIDI con timidity."""
    import subprocess

    cmd = ["timidity"]
    if soundfont:
        cmd += ["-x", f"soundfont {soundfont}"]
    cmd.append(midi_path)

    print(f"  {green('▶')} Reproduciendo... {dim('(Ctrl+C para parar)')}")
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(yellow("  timidity no encontrado. Instálalo con: sudo apt install timidity"))
    except KeyboardInterrupt:
        print()


# ════════════════════════════════════════════════════════════════════
#  BUCLE DE GENERACIÓN Y REFINAMIENTO
# ════════════════════════════════════════════════════════════════════

def _generation_loop(seed_state: dict, vectors, paths_arr, args):
    """
    Fase de generación: extrae perfil, genera MIDI, reproduce,
    y permite refinamiento iterativo hasta que el usuario esté satisfecho.
    """
    print(f"\n{'═'*50}")
    print(f"  {bold('GENERACIÓN')}\n")
    print(f"  Basándome en lo que elegiste, voy a generar una primera frase.\n")

    # ── Asegurar que tenemos instrumentación ────────────────────────
    if "instrumentación" not in seed_state["crystallized"]:
        _present_and_record_question(INSTRUMENT_QUESTION, seed_state)
        print()

    # ── Extraer perfil generativo ───────────────────────────────────
    profile        = _extract_generative_profile(seed_state, vectors, paths_arr)
    instr_key      = _infer_instrumentation(seed_state)
    instrument_grp = INSTRUMENT_GROUPS[instr_key]

    print(f"  {bold('Perfil generativo:')}")
    print(f"    Escala:       {cyan(profile['tonic_name'])} {profile['scale_name']}")
    print(f"    Tempo:        {profile['tempo']} BPM")
    print(f"    Duración:     {profile['bars']} compases")
    print(f"    Instrumentos: {instrument_grp['label']}")
    print(f"    Carácter:     {describe_vector(seed_state['query_vec'])}")
    print()

    gen_seed = random.randint(0, 9999)
    output_path = Path(f"seed_output_{gen_seed}.mid")

    iteration = 0
    max_iter  = 10

    while iteration < max_iter:
        iteration += 1

        # ── Generar ─────────────────────────────────────────────────
        print(f"  {dim(f'Generando ({iteration})...')}", end=" ", flush=True)
        try:
            mid = _build_midi(profile, instrument_grp, seed=gen_seed)
            mid.save(str(output_path))
            print(green("✓"))
        except Exception as e:
            print(red(f"Error: {e}"))
            break

        # ── Reproducir ──────────────────────────────────────────────
        _play_with_timidity(str(output_path), soundfont=getattr(args, "soundfont", None))

        # ── Opciones de refinamiento ─────────────────────────────────
        print(f"\n  {bold('¿Qué hacemos?')}\n")
        print(f"    {cyan('guardar')}          — me gusta, guardar y terminar")
        print(f"    {cyan('otra')}             — generar otra variante diferente")
        print(f"    {cyan('más lento/rápido')} — ajustar tempo")
        print(f"    {cyan('más oscuro')}       — bajar registro y densidad")
        print(f"    {cyan('más denso')}        — más notas, más actividad")
        print(f"    {cyan('más sparse')}       — menos notas, más silencios")
        print(f"    {cyan('más largo')}        — más compases")
        print(f"    {cyan('cambiar sonido')}   — cambiar instrumentación")
        print(f"    {cyan('repetir')}          — escuchar de nuevo")
        print(f"    {cyan('salir')}            — terminar sin guardar")
        print()

        resp = input("  > ").strip().lower()

        if resp in ("guardar", "save", "ok", "sí", "si", "perfecto", "me gusta"):
            final_path = _save_final(output_path, seed_state, profile, instrument_grp)
            print(f"\n  {green('✓')} Guardado: {bold(str(final_path))}")
            break

        elif resp in ("salir", "exit", "q", "no"):
            output_path.unlink(missing_ok=True)
            print(f"  {dim('Saliendo sin guardar.')}")
            break

        elif resp in ("repetir", "replay", "escuchar", "play"):
            continue   # vuelve a reproducir sin regenerar

        elif resp in ("otra", "variante", "diferente", "otra vez"):
            gen_seed = random.randint(0, 9999)
            print(f"  {dim('Nueva semilla aleatoria...')}")

        elif "lento" in resp:
            profile["tempo"] = max(30, int(profile["tempo"] * 0.8))
            profile["notes_per_bar"] = max(1, profile["notes_per_bar"] - 1)
            _t = profile["tempo"]; print(f"  {dim(f'Tempo → {_t} BPM')}")

        elif "rápido" in resp or "rapido" in resp:
            profile["tempo"] = min(200, int(profile["tempo"] * 1.25))
            _t = profile["tempo"]; print(f"  {dim(f'Tempo → {_t} BPM')}")

        elif "oscuro" in resp:
            profile["tonic_midi"]   = max(36, profile["tonic_midi"] - 5)
            profile["vel_mean"]     = max(20, profile["vel_mean"] - 10)
            profile["silence_ratio"]= min(0.9, profile["silence_ratio"] + 0.1)
            # Cambiar a escala más oscura
            if profile["scale_name"] in ("mayor", "lidio", "mixolidio"):
                profile["scale_name"] = "menor_natural"
                profile["scale"]      = SCALES["menor_natural"]
            elif profile["scale_name"] == "menor_natural":
                profile["scale_name"] = "frigio"
                profile["scale"]      = SCALES["frigio"]
            print(f"  {dim('Más oscuro: registro más grave, escala más oscura')}")

        elif "luminoso" in resp or "claro" in resp or "brillante" in resp:
            profile["tonic_midi"] = min(84, profile["tonic_midi"] + 5)
            profile["vel_mean"]   = min(110, profile["vel_mean"] + 10)
            if profile["scale_name"] in ("frigio", "locrio", "menor_natural"):
                profile["scale_name"] = "mayor"
                profile["scale"]      = SCALES["mayor"]
            print(f"  {dim('Más luminoso: registro más agudo, escala mayor')}")

        elif "denso" in resp:
            profile["notes_per_bar"]  = min(12, profile["notes_per_bar"] + 3)
            profile["silence_ratio"]  = max(0.0, profile["silence_ratio"] - 0.15)
            print(f"  {dim('Más denso: más notas por compás')}")

        elif "sparse" in resp or "vacío" in resp or "vacio" in resp or "espacio" in resp:
            profile["notes_per_bar"]  = max(1, profile["notes_per_bar"] - 2)
            profile["silence_ratio"]  = min(0.85, profile["silence_ratio"] + 0.2)
            print(f"  {dim('Más sparse: menos notas, más silencios')}")

        elif "largo" in resp or "más compases" in resp:
            profile["bars"] = min(128, profile["bars"] + 16)
            _b = profile["bars"]; print(f"  {dim(f'Duración → {_b} compases')}")

        elif "corto" in resp:
            profile["bars"] = max(8, profile["bars"] - 8)
            _b = profile["bars"]; print(f"  {dim(f'Duración → {_b} compases')}")

        elif "tenso" in resp or "tensión" in resp or "tension" in resp:
            profile["rhythm_variance"] = min(1.0, profile["rhythm_variance"] + 0.2)
            profile["vel_var"]         = min(40, profile["vel_var"] + 10)
            if profile["scale_name"] not in ("locrio", "frigio", "menor_armónica"):
                profile["scale_name"] = "menor_armónica"
                profile["scale"]      = SCALES["menor_armónica"]
            print(f"  {dim('Más tensión: ritmo más irregular, escala armónica')}")

        elif "calmo" in resp or "calma" in resp or "tranquilo" in resp:
            profile["rhythm_variance"] = max(0.0, profile["rhythm_variance"] - 0.2)
            profile["vel_var"]         = max(0, profile["vel_var"] - 10)
            profile["silence_ratio"]   = min(0.7, profile["silence_ratio"] + 0.1)
            print(f"  {dim('Más calmo: ritmo más regular, más silencios')}")

        elif "cambiar sonido" in resp or "instrumento" in resp or "sonido" in resp:
            _present_and_record_question(INSTRUMENT_QUESTION, seed_state)
            instr_key      = _infer_instrumentation(seed_state)
            instrument_grp = INSTRUMENT_GROUPS[instr_key]
            _lbl = instrument_grp["label"]; print(f"  {dim(f'Instrumentación → {_lbl}')}")

        elif "irregular" in resp or "libre" in resp:
            profile["rhythm_variance"] = min(1.0, profile["rhythm_variance"] + 0.25)
            print(f"  {dim('Ritmo más irregular')}")

        elif "regular" in resp or "pulso" in resp:
            profile["rhythm_variance"] = max(0.0, profile["rhythm_variance"] - 0.25)
            print(f"  {dim('Ritmo más regular')}")

        else:
            # Intentar interpretar con el mapa semántico
            base_vec, _ = text_to_query_vector(resp)
            # Aplicar cambios según dimensiones más afectadas
            diff = base_vec - np.full(N_DIMS, 0.5)
            if abs(diff[0]) > 0.15:   # registro
                shift = int(diff[0] * 12)
                profile["tonic_midi"] = max(36, min(84, profile["tonic_midi"] + shift))
            if abs(diff[4]) > 0.15:   # densidad
                profile["notes_per_bar"] = max(1, min(12,
                    profile["notes_per_bar"] + int(diff[4] * 6)))
            if abs(diff[9]) > 0.15:   # silencio
                profile["silence_ratio"] = float(np.clip(
                    profile["silence_ratio"] + diff[9] * 0.3, 0.0, 0.9))
            print(f"  {dim('Ajustando perfil según descripción...')}")

        print()

    if iteration >= max_iter:
        _save_final(output_path, seed_state, profile, instrument_grp)
        print(f"  {dim('Límite de iteraciones alcanzado. Guardado automáticamente.')}")


def _save_final(midi_path: Path, seed_state: dict,
                profile: dict, instrument_grp: dict) -> Path:
    """Guarda el MIDI final con nombre descriptivo y metadata JSON."""
    intent_slug = seed_state["original_intent"][:30].replace(" ", "_")
    intent_slug = "".join(c for c in intent_slug if c.isalnum() or c == "_")
    final_name  = f"seed_{intent_slug}_{profile['tonic_name']}_{profile['scale_name']}.mid"
    final_path  = Path(final_name)

    if midi_path.exists():
        midi_path.rename(final_path)

    # Metadata
    meta = {
        "intent":          seed_state["original_intent"],
        "crystallized":    seed_state["crystallized"],
        "profile":         {k: v for k, v in profile.items() if k != "scale"},
        "instrumentation": instrument_grp["label"],
        "chosen_files":    [str(seed_state.get("chosen_indices", []))],
    }
    meta_path = final_path.with_suffix(".seed.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return final_path


def _show_seed_summary(seed_state: dict, vectors, paths, meta):
    """Muestra resumen breve antes de pasar a la generación."""
    print(f"\n{'─'*50}")
    print(f"  {bold('EXPLORACIÓN COMPLETADA')}\n")

    if seed_state["chosen_indices"]:
        print(f"  Fragmentos de referencia elegidos: {len(seed_state['chosen_indices'])}")
        for idx in seed_state["chosen_indices"][:3]:
            m    = json.loads(str(meta[idx]))
            name = Path(str(paths[idx])).name
            dur  = format_duration(m["duration_s"])
            desc = describe_vector(vectors[idx])
            print(f"    · {name} — {dur} — {dim(desc)}")
    else:
        print(f"  {yellow('Sin fragmentos de referencia — usando solo el perfil vectorial.')}")

    if seed_state["crystallized"]:
        print(f"\n  Decisiones tomadas:")
        for k, v in seed_state["crystallized"].items():
            print(f"    · {k}: {green(v)}")
    print()


def _save_seed_state(seed_state: dict, index_path: Path):
    """Guarda el estado de la sesión seed en JSON."""
    out = index_path.with_suffix(".seed.json")
    state_to_save = {k: v for k, v in seed_state.items() if k != "query_vec"}
    if seed_state["query_vec"] is not None:
        state_to_save["query_vec"] = seed_state["query_vec"].tolist()
    if seed_state["weights"] is not None:
        state_to_save["weights"] = seed_state["weights"].tolist()

    with open(out, "w") as f:
        json.dump(state_to_save, f, indent=2, ensure_ascii=False)

    print(f"  {green(f'Estado guardado: {out}')}")


# ════════════════════════════════════════════════════════════════════
#  ENTRADA PRINCIPAL
# ════════════════════════════════════════════════════════════════════

def main():
    if not HAS_MIDO and not (len(sys.argv) > 1 and sys.argv[1] in ("cluster", "search", "seed")):
        print(red("Error: mido no instalado. pip install mido numpy"))
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Corpus Explorer — exploración semántica de corpus MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # health
    p_health = sub.add_parser("health", help="Diagnóstico del corpus")
    p_health.add_argument("path", help="Directorio con MIDIs")
    p_health.add_argument("--sample", type=int, default=500,
                          help="Tamaño de muestra para el diagnóstico (default: 500)")

    # index
    p_index = sub.add_parser("index", help="Vectorizar corpus")
    p_index.add_argument("path", help="Directorio con MIDIs")
    p_index.add_argument("--output", default="corpus.npz", help="Archivo de salida .npz")

    # cluster
    p_cluster = sub.add_parser("cluster", help="Agrupar por similitud")
    p_cluster.add_argument("path", help="Archivo .npz del índice")
    p_cluster.add_argument("--k", type=int, default=None, help="Número de clusters (auto si no se especifica)")

    # search
    p_search = sub.add_parser("search", help="Búsqueda por intención")
    p_search.add_argument("path", help="Archivo .npz del índice")
    p_search.add_argument("query", nargs="?", default="", help="Texto de búsqueda")
    p_search.add_argument("--top", type=int, default=5, help="Número de resultados (default: 5)")
    p_search.add_argument("--no-llm", action="store_true", help="No usar LLM")
    p_search.add_argument("--llm-provider", default="auto",
                          choices=["auto", "anthropic", "openai", "clipboard"],
                          help="Proveedor LLM (default: auto)")
    p_search.add_argument("--play", action="store_true", help="Reproducir resultados (requiere pygame)")

    # seed
    p_seed = sub.add_parser("seed", help="Flujo interactivo completo")
    p_seed.add_argument("path", help="Archivo .npz del índice")
    p_seed.add_argument("--no-llm", action="store_true", help="No usar LLM")
    p_seed.add_argument("--llm-provider", default="auto",
                        choices=["auto", "anthropic", "openai", "clipboard"],
                        help="Proveedor LLM: auto | anthropic | openai | clipboard (default: auto)")
    p_seed.add_argument("--soundfont", default=None,
                        help="Path al soundfont .sf2 para timidity (opcional)")
    p_seed.add_argument("--preview-seconds", type=int, default=20,
                        help="Segundos de preview al reproducir fragmentos del corpus (default: 20)")

    args = parser.parse_args()

    if args.mode == "health":
        cmd_health(args)
    elif args.mode == "index":
        cmd_index(args)
    elif args.mode == "cluster":
        cmd_cluster(args)
    elif args.mode == "search":
        cmd_search(args)
    elif args.mode == "seed":
        cmd_seed(args)


if __name__ == "__main__":
    main()
