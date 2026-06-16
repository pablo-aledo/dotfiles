#!/usr/bin/env python3
"""
================================================================================
                       LYRICS MELODY  v3.1
       Generacion de melodias condicionadas por el contenido emocional
              del texto (lyrics), con fluidez entre frases

  Aprende la relacion entre el contenido SEMANTICO/EMOCIONAL de cada verso
  de una cancion y la melodia completa que la acompana, a partir de un
  corpus de ficheros ABC con lineas de letra (campo "w:" alineado con la
  melodia).

  Modela dos cosas a la vez:
    [1] EMBEDDING EMOCIONAL de cada verso, mediante un modelo de
        sentence-embeddings preentrenado y multilingue (MiniLM), inyectado
        como "segment embedding": cada posicion de la melodia recibe el
        embedding del verso al que pertenece (acoplamiento texto-melodia).
    [2] MODELO DE LENGUAJE MELODICO (Transformer causal con atencion
        relativa, estilo Music Transformer) sobre la SECUENCIA COMPLETA de
        la cancion (todas las frases concatenadas, separadas por un token
        especial <phrase>), de forma que cada nota se condiciona en TODO el
        historial melodico anterior, dando fluidez en las uniones entre
        versos.

  NOVEDADES v3.1
  ---------------
  Se incorporan tres mejoras de consistencia melodica en el motor transformer,
  sin necesidad de reentrenar el modelo:

  [A] SUAVIZADO DE OCTAVA (_smooth_octave):
      Tras decodificar cada token, la octava se corrige eligiendo la mas
      proxima a la nota anterior (minimizando el salto en semitonos). Elimina
      los grandes saltos de registro que aparecian cuando el modelo saltaba
      de octava de forma abrupta entre notas consecutivas o entre versos.

  [B] MASCARA DE ESCALA (--mode, _build_scale_mask):
      Antes de muestrear, los tokens N_* cuyo pitch class no pertenece a la
      escala (--key, --mode) se ponen a -inf en los logits. La melodia
      resultante queda estrictamente dentro de la tonalidad pedida sin
      alterar el modelo entrenado.
      Modos disponibles: major, minor, harmonic_minor, dorian, phrygian,
      mixolydian, pentatonic_major, pentatonic_minor, blues, chromatic.

  [C] SESGO DE INTERVALO POR PERFIL (--interval-bias-strength,
      _build_interval_bias):
      En cada paso se calcula un sesgo aditivo en log-space sobre los logits,
      proporcional a los pesos de INTERVAL_WEIGHTS[profile]. Los intervalos
      tipicos del perfil emocional (p. ej. semitonos para melancholic, quintas
      para heroic) se refuerzan sin forzar; los ajenos se penalizan
      suavemente. La intensidad es regulable con --interval-bias-strength
      (0.0 = desactivado, 1.0 = sesgo fuerte; default: 0.4).

  COMANDOS:
    train     - Entrena el modelo conjunto a partir de un corpus .abc
    generate  - Genera una melodia MIDI/ABC a partir de un fichero de letra
    inspect   - Diagnostico: vocabularios, dataset y config de un checkpoint

  MOTORES DE GENERACION (--engine):
    transformer - Modelo neuronal entrenado (requiere --model). Por defecto.
    markov      - Bigrama de notas agrupado por clusters emocionales del
                  corpus, con historial continuo entre versos.
    (Ambos motores requieren el modelo de embeddings; ver DEPENDENCIAS)

  PERFILES EMOCIONALES (--profile):
    neutral     - sin sesgo; temperatura y sesgo de intervalo neutros
    heroic      - quintas y octavas ascendentes; temperatura alta
    melancholic - semitonos y segundas descendentes; temperatura baja
    tense       - cromatismo, intervalos de tritono; temperatura muy alta
    serene      - grados conjuntos, movimiento suave; temperatura baja
    playful     - saltos cortos variados; ritmo irregular
    mysterious  - tritonos y sextas; silencios frecuentes
    triumphant  - quintas y octavas; clímax en el ultimo tercio
    tanguero    - segundas y terceras con cromatismo; arrabal
    flamenco    - semitonos descendentes; segundo grado frigio

  MODOS DE ESCALA (--mode):
    major            - escala mayor diatonica (default)
    minor            - escala menor natural
    harmonic_minor   - menor armonica (VII#)
    dorian           - menor con VI mayor (modal folk/jazz)
    phrygian         - menor con II bemol (flamenco, metal)
    mixolydian       - mayor con VII menor (rock, folk)
    pentatonic_major - pentatonica mayor (5 notas)
    pentatonic_minor - pentatonica menor (5 notas)
    blues            - pentatonica menor + V# (blue note)
    chromatic        - sin restriccion de escala (12 notas)

  FORMATO ESPERADO DE LOS FICHEROS ABC DE ENTRENAMIENTO
  ------------------------------------------------------
  Cabecera ABC estandar (X:, T:, M:, L:, K:) + linea(s) "w:" justo despues
  de la linea de melodia. Cada linea w: se trata como UN VERSO; su texto
  completo (ignorando '-' y '*') se usa para calcular el embedding
  emocional de la frase melodica correspondiente. Todas las frases de una
  misma melodia (X:) se tratan como UNA secuencia continua:

    X:1
    T:Ejemplo
    M:4/4
    L:1/8
    K:C
    C D E F | G2 A2 |
    w:Ho-la_ mun-do mu-si-cal a-qui

  EJEMPLOS DE USO
  ----------------
  # Entrenar un modelo a partir de una carpeta de .abc con lyrics
  python3 lyrics_melody.py train --corpus ./canciones_abc/ \
      --epochs 60 --model-out lyrics_melody.pt

  # Retomar entrenamiento desde un checkpoint existente
  python3 lyrics_melody.py train --corpus ./canciones_abc/ \
      --model-out lyrics_melody.pt --resume --epochs 30

  # Generacion basica: tonalidad y perfil
  python3 lyrics_melody.py generate --model lyrics_melody.pt \
      --lyrics letra.txt --key C --mode major --profile melancholic \
      --tempo 96 --out melodia.mid --abc-out melodia.abc

  # Generacion con escala menor y sesgo de intervalo moderado
  python3 lyrics_melody.py generate --model lyrics_melody.pt \
      --lyrics letra.txt --key Am --mode minor --profile melancholic \
      --interval-bias-strength 0.4 --out melodia.mid --abc-out melodia.abc

  # Generacion con modo frigio (flamenco) y sesgo fuerte
  python3 lyrics_melody.py generate --model lyrics_melody.pt \
      --lyrics letra.txt --key E --mode phrygian --profile flamenco \
      --interval-bias-strength 0.7 --temperature 0.9 \
      --out melodia.mid --abc-out melodia.abc

  # Solo mascara de escala, sin sesgo de intervalo (0.0 = desactivado)
  python3 lyrics_melody.py generate --model lyrics_melody.pt \
      --lyrics letra.txt --key G --mode major --profile neutral \
      --interval-bias-strength 0.0 --out melodia.mid

  # Generacion sin modelo entrenado, con el motor markov de respaldo
  python3 lyrics_melody.py generate --engine markov --corpus ./canciones_abc/ \
      --lyrics letra.txt --out melodia.mid

  # Inspeccionar vocabulario, dataset y config de un modelo entrenado
  python3 lyrics_melody.py inspect --model lyrics_melody.pt
  python3 lyrics_melody.py inspect --corpus ./canciones_abc/

  OPCIONES DE GENERATE
  ---------------------
    --lyrics FILE          Fichero de texto (una linea = un verso = una frase)
    --out FILE             MIDI de salida
    --abc-out FILE         ABC de salida (opcional)
    --engine               transformer (default) | markov
    --model FILE           Checkpoint .pt (motor transformer)
    --corpus DIR           Carpeta .abc (motor markov)
    --key KEY              Tonalidad: C, Am, F#, Bb... (default: C)
    --mode MODE            Modo de escala para mascara tonal (default: major)
    --profile PROFILE      Perfil emocional (default: neutral)
    --interval-bias-strength S  Intensidad del sesgo de intervalo 0.0-1.0
                                (default: 0.4; 0.0 = desactivado)
    --tempo BPM            Tempo en BPM (default: 100)
    --meter M/N            Compas (default: 4/4)
    --temperature T        Temperatura de muestreo (default: 1.0)
    --top-k K              Top-k sampling; 0 = desactivado (default: 0)
    --notes-per-line N     Notas fijas por verso; 0 = estimar (default: 0)
    --unit-beats F         Duracion de una unidad L: en negras (default: 0.5)
    --seed N               Semilla aleatoria (motor markov; default: 42)
    --n-clusters N         Clusters emocionales (motor markov; default: 6)
    --cpu                  Forzar CPU

  DEPENDENCIAS
    Siempre:  mido                  -> pip install mido
              sentence-transformers -> pip install sentence-transformers
              (instala torch como dependencia transitiva si no esta)

    El modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2, ~470MB)
    se descarga automaticamente la primera vez y se cachea localmente en
    ~/.cache/torch/sentence_transformers/. Corre comodamente en CPU
    (decenas de ms por verso) o en una GPU pequena (p.ej. 6GB VRAM).
    Sin conexion y sin cache previa, train/generate (motores transformer y
    markov) no podran calcular embeddings; 'inspect --corpus' sigue
    funcionando sin el modelo de embeddings.

  NOTAS DE DISENO
  ----------------
  - Las notas se representan RELATIVAS a la tonica (K:) del fragmento ABC,
    como (grado_cromatico 0-11, octava_relativa, duracion cuantizada en
    fracciones de L:). Permite generar en cualquier tonalidad sin reentrenar.
  - FLUIDEZ ENTRE FRASES: cada melodia (X:) se entrena/genera como UNA SOLA
    secuencia continua. Las frases se separan con el token especial
    <phrase>, pero la atencion causal ve todo el historial anterior, por lo
    que la nota inicial de un verso se condiciona en la nota final del
    verso previo. El "segment embedding" emocional cambia por posicion
    segun a que verso pertenece, permitiendo que el contenido emocional
    module la melodia verso a verso sin romper la continuidad musical.
  - El acoplamiento texto-melodia se modela proyectando el embedding de
    384 dims del modelo de sentence-embeddings a d_model y sumandolo en
    cada posicion (segment embedding), junto a note_embed + pos_embed.
  - Atencion relativa de posicion (truco de skewing, Music Transformer,
    Huang et al. 2019) para capturar motivos recurrentes independientemente
    de su posicion dentro de la secuencia.
  - ORDEN DE APLICACION en el muestreo (v3.1):
      logits del modelo
        -> mascara de escala (-inf a notas fuera de tonalidad)
        -> sesgo de intervalo (log-space, proporcional a INTERVAL_WEIGHTS)
        -> division por temperatura
        -> top-k (opcional)
        -> softmax + multinomial
        -> _smooth_octave (correccion de octava por proximidad)
================================================================================
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

# ==============================================================================
#  CONSTANTES
# ==============================================================================

VERSION = "3.1"

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

KEY_TO_PC = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5,
    "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}

DURATIONS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

PAD, BOS, EOS, REST, PHRASE = "<pad>", "<bos>", "<eos>", "<rest>", "<phrase>"
SPECIAL_TOKENS = [PAD, BOS, EOS, REST, PHRASE]

# Perfiles emocionales manuales: se SUMAN al embedding emocional derivado
# del texto, como sesgo global.
EMOTIONAL_PROFILES = {
    "neutral":     {"temperature_scale": 1.00, "octave_bias": 0.00, "duration_bias": 0.00, "climax_pos": 0.618},
    "heroic":      {"temperature_scale": 1.05, "octave_bias": 0.35, "duration_bias": -0.10, "climax_pos": 0.75},
    "melancholic": {"temperature_scale": 0.85, "octave_bias": -0.25, "duration_bias": 0.25, "climax_pos": 0.40},
    "tense":       {"temperature_scale": 1.20, "octave_bias": 0.10, "duration_bias": -0.30, "climax_pos": 0.85},
    "serene":      {"temperature_scale": 0.75, "octave_bias": -0.10, "duration_bias": 0.30, "climax_pos": 0.618},
    "playful":     {"temperature_scale": 1.15, "octave_bias": 0.15, "duration_bias": -0.20, "climax_pos": 0.618},
    "mysterious":  {"temperature_scale": 0.95, "octave_bias": -0.15, "duration_bias": 0.10, "climax_pos": 0.618},
    "triumphant":  {"temperature_scale": 1.10, "octave_bias": 0.40, "duration_bias": -0.10, "climax_pos": 0.85},
    "tanguero":    {"temperature_scale": 1.00, "octave_bias": 0.00, "duration_bias": 0.05, "climax_pos": 0.618},
    "flamenco":    {"temperature_scale": 1.10, "octave_bias": 0.10, "duration_bias": -0.05, "climax_pos": 0.70},
}

# Pesos de intervalo (en semitonos) por perfil emocional.
# Adaptados de melody_generator.py para modular los logits del transformer.
# Clave = intervalo en semitonos (positivo = ascendente, negativo = descendente).
INTERVAL_WEIGHTS: dict = {
    "neutral":     {0: 1.0, 1: 1.5, 2: 2.0, -1: 1.5, -2: 2.0, 3: 1.0, -3: 1.0, 5: 0.8, -5: 0.8},
    "heroic":      {0: 0.5, 2: 2.0, 3: 1.5, 4: 2.0, 5: 2.5, 7: 3.0, 12: 2.5, -2: 2.0, -5: 1.5, -7: 1.5},
    "melancholic": {0: 0.3, 1: 2.5, 2: 3.0, 3: 2.0, -1: 3.0, -2: 3.5, -3: 2.0, -5: 1.5, 5: 1.0, 7: 0.5},
    "playful":     {0: 0.5, 1: 1.5, 2: 2.5, 3: 2.0, 4: 2.0, 5: 1.5, -1: 1.5, -2: 2.5, -3: 2.0, 7: 1.0},
    "tense":       {0: 0.5, 1: 3.0, 2: 2.0, 6: 2.0, -1: 3.0, -2: 2.0, -6: 2.0, 11: 1.5, -11: 1.5},
    "serene":      {0: 1.0, 2: 3.0, 3: 2.0, 4: 2.5, 5: 2.0, -2: 3.0, -3: 2.0, -4: 2.0, -5: 1.5, 7: 1.0},
    "mysterious":  {0: 1.0, 1: 2.0, 6: 2.5, -1: 2.0, -6: 2.5, 3: 1.5, -3: 1.5, 8: 1.5, -8: 1.5},
    "triumphant":  {0: 0.3, 2: 2.0, 4: 2.5, 5: 2.0, 7: 3.5, 12: 2.0, -2: 1.5, -5: 1.5, 3: 2.0},
    "tanguero":    {0: 0.5, 1: 2.0, 2: 2.5, -1: 2.5, -2: 3.0, 3: 1.5, -3: 2.0, 6: 1.5, -6: 1.5},
    "flamenco":    {0: 0.5, 1: 3.0, 2: 2.0, -1: 3.5, -2: 2.5, 3: 1.0, -3: 1.5, 6: 1.0, 4: 1.5},
}

# Intervalos de escala por modo (grados sobre la tonica).
SCALE_INTERVALS: dict = {
    "major":            [0, 2, 4, 5, 7, 9, 11],
    "minor":            [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor":   [0, 2, 3, 5, 7, 8, 11],
    "dorian":           [0, 2, 3, 5, 7, 9, 10],
    "phrygian":         [0, 1, 3, 5, 7, 8, 10],
    "mixolydian":       [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues":            [0, 3, 5, 6, 7, 10],
    "chromatic":        list(range(12)),
}

# ==============================================================================
#  PARSER ABC MINIMAL (notas + lineas w: como versos completos)
# ==============================================================================

NOTE_RE = re.compile(
    r"(?P<acc>\^\^|\^|=|__|_)?"
    r"(?P<note>[A-Ga-g])"
    r"(?P<oct>[,']*)"
    r"(?P<dur>\d+/\d+|\d+|/\d+|/+)?"
)

REST_RE = re.compile(r"(?P<rest>[zZxX])(?P<dur>\d+/\d+|\d+|/\d+|/+)?")


def parse_key(key_field):
    """Extrae la tonica (pitch class 0-11) de un campo K: de ABC."""
    m = re.match(r"\s*([A-G])([#b]?)", key_field)
    if not m:
        return 0
    name = m.group(1) + m.group(2)
    return KEY_TO_PC.get(name, KEY_TO_PC.get(m.group(1), 0))


def parse_default_len(l_field):
    """Extrae L:1/8 -> 0.125 (fraccion de redonda) -> se usa como unidad base 1."""
    m = re.match(r"\s*(\d+)/(\d+)", l_field)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return 1.0 / 8.0


def note_token_to_pc_octave(token, accidentals):
    """Convierte una letra de nota ABC en (pitch_class 0-11, octava_relativa)."""
    letter = token.upper()
    base_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    pc = (base_pc + accidentals) % 12
    octv = 0
    return pc, octv


def parse_duration_str(dur_str, default=1.0):
    if not dur_str:
        return default
    if "/" in dur_str:
        parts = dur_str.split("/")
        num = 1.0 if parts[0] == "" else float(parts[0])
        den = 2.0 if parts[1] == "" else float(parts[1])
        return default * num / den
    if dur_str.startswith("/"):
        n = len(dur_str)
        return default / (2 ** n)
    try:
        return default * float(dur_str)
    except ValueError:
        return default


def quantize_duration(dur):
    """Ajusta una duracion al valor mas cercano del vocabulario DURATIONS."""
    return min(DURATIONS, key=lambda d: abs(d - dur))


def _clean_verse_fragment(fragment):
    """Limpia un fragmento de linea w: (entre barras o la linea completa),
    eliminando marcadores de alineacion silabica ('-', '_', '*')."""
    fragment = fragment.replace("-", "")
    fragment = fragment.replace("_", " ")
    fragment = fragment.replace("*", "")
    fragment = re.sub(r"\s+", " ", fragment).strip()
    return fragment


def w_line_to_text(w_line):
    """Convierte una linea w: de ABC en el texto completo del verso,
    eliminando marcadores de alineacion silabica ('-', '_', '*')."""
    content = w_line.split(":", 1)[1] if ":" in w_line else w_line
    return _clean_verse_fragment(content.strip())


def split_w_line_by_bars(w_line, n_bars):
    """
    Divide el texto de una linea w: en n_bars fragmentos, uno por compas de
    la melodia asociada:
      - Si la linea w: contiene '|', se usa esa division directamente
        (descartando fragmentos sobrantes o repitiendo el ultimo si faltan).
      - Si no contiene '|', se reparte el texto en n_bars grupos de palabras
        lo mas equilibrados posible.
    Devuelve una lista de n_bars strings (puede contener "").
    """
    if n_bars <= 0:
        return []
    content = w_line.split(":", 1)[1] if ":" in w_line else w_line
    content = content.strip()

    if "|" in content:
        raw_bars = [b for b in content.split("|")]
        fragments = [_clean_verse_fragment(b) for b in raw_bars if b.strip() != ""]
        if not fragments:
            fragments = [""]
        while len(fragments) < n_bars:
            fragments.append(fragments[-1])
        return fragments[:n_bars]

    cleaned = _clean_verse_fragment(content)
    words = cleaned.split(" ") if cleaned else []
    if not words:
        return [""] * n_bars

    fragments = []
    n_words = len(words)
    base = n_words // n_bars
    extra = n_words % n_bars
    idx = 0
    for i in range(n_bars):
        take = base + (1 if i < extra else 0)
        fragments.append(" ".join(words[idx:idx + take]))
        idx += take
    return fragments


def parse_abc_file(path):
    """
    Parsea un fichero .abc devolviendo una lista de "canciones", cada una un
    dict:
      {
        "phrases": [
            {"events": [{"pc": int|None, "octave": int, "dur": float}, ...],
             "text": str},
            ...
        ]
      }
    Cada "frase" corresponde a un tramo de melodia delimitado por '|' con su
    verso asociado (linea w: completa, best-effort). Las frases de una misma
    cancion (bloque X:) se mantienen en orden, para poder concatenarlas en
    una secuencia continua.
    """
    tunes = []
    current_lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.strip() == "" and current_lines:
                tunes.append(current_lines)
                current_lines = []
            else:
                current_lines.append(line)
    if current_lines:
        tunes.append(current_lines)

    all_songs = []

    for tune_lines in tunes:
        key_pc = 0
        unit_len = 1.0 / 8.0
        melody_w_pairs = []
        pending_melody = None

        for line in tune_lines:
            if re.match(r"^K:", line):
                key_pc = parse_key(line[2:])
                continue
            if re.match(r"^L:", line):
                unit_len = parse_default_len(line[2:])
                continue
            if re.match(r"^[A-Zx]:", line):
                continue
            if line.strip().startswith("w:"):
                if pending_melody is not None:
                    melody_w_pairs.append((pending_melody, line))
                    pending_melody = None
                continue
            if line.strip() == "" or line.strip().startswith("%"):
                continue
            if pending_melody is not None:
                melody_w_pairs.append((pending_melody, None))
            pending_melody = line

        if pending_melody is not None:
            melody_w_pairs.append((pending_melody, None))

        song_phrases = []

        for melody_line, w_line in melody_w_pairs:
            cleaned = re.sub(r'!.*?!', '', melody_line)
            cleaned = re.sub(r'".*?"', '', cleaned)
            cleaned = re.sub(r'\[.*?\]', '', cleaned)

            # Numero de compases (separados por '|') con contenido en esta
            # linea de melodia, para repartir el texto de w: entre ellos.
            n_bars = max(1, len([b for b in cleaned.split("|") if b.strip() != ""]))
            if w_line:
                bar_texts = split_w_line_by_bars(w_line, n_bars)
            else:
                bar_texts = [""] * n_bars
            bar_idx = 0

            phrase_events = []
            pos = 0
            while pos < len(cleaned):
                ch = cleaned[pos]
                if ch in "|":
                    if phrase_events:
                        verse_text = bar_texts[bar_idx] if bar_idx < len(bar_texts) else ""
                        song_phrases.append({"events": phrase_events, "text": verse_text})
                        phrase_events = []
                        bar_idx += 1
                    pos += 1
                    continue
                if ch in " \t":
                    pos += 1
                    continue
                if ch in "(){}><.~":
                    pos += 1
                    continue

                rm = REST_RE.match(cleaned, pos)
                if rm and rm.group("rest"):
                    dur = parse_duration_str(rm.group("dur"), default=unit_len)
                    qdur = quantize_duration(dur / unit_len)
                    phrase_events.append({"pc": None, "octave": 0, "dur": qdur})
                    pos = rm.end()
                    continue

                nm = NOTE_RE.match(cleaned, pos)
                if nm and nm.group("note"):
                    acc_str = nm.group("acc") or ""
                    acc_map = {"^": 1, "^^": 2, "_": -1, "__": -2, "=": 0}
                    acc_val = acc_map.get(acc_str, 0)
                    letter = nm.group("note")
                    base_pc, _ = note_token_to_pc_octave(letter, acc_val)
                    octv = 1 if letter.islower() else 0
                    for c in nm.group("oct") or "":
                        if c == "'":
                            octv += 1
                        elif c == ",":
                            octv -= 1
                    rel_pc = (base_pc - key_pc) % 12
                    dur = parse_duration_str(nm.group("dur"), default=unit_len)
                    qdur = quantize_duration(dur / unit_len)
                    phrase_events.append({"pc": rel_pc, "octave": octv, "dur": qdur})
                    pos = nm.end()
                    continue

                pos += 1

            if phrase_events:
                verse_text = bar_texts[bar_idx] if bar_idx < len(bar_texts) else ""
                song_phrases.append({"events": phrase_events, "text": verse_text})

        if song_phrases:
            all_songs.append({"phrases": song_phrases})

    return all_songs


def collect_corpus_songs(corpus_path):
    """Recorre un fichero o carpeta de .abc y devuelve (songs, files).
    Cada elemento de songs es {"phrases": [{"events": [...], "text": str}, ...]}."""
    abc_files = []
    if os.path.isdir(corpus_path):
        for root, _, files in os.walk(corpus_path):
            for fn in files:
                if fn.lower().endswith(".abc"):
                    abc_files.append(os.path.join(root, fn))
    else:
        abc_files = [corpus_path]

    all_songs = []
    for f in abc_files:
        try:
            songs = parse_abc_file(f)
        except Exception as e:
            print(f"[aviso] error parseando {f}: {e}", file=sys.stderr)
            continue
        all_songs.extend(songs)

    return all_songs, abc_files


def flatten_phrases(songs):
    """Devuelve una lista plana de todas las frases de todas las canciones
    (util para inspect / estadisticas globales)."""
    phrases = []
    for song in songs:
        phrases.extend(song["phrases"])
    return phrases


# ==============================================================================
#  TOKENIZACION DE NOTAS
# ==============================================================================

def build_note_vocab():
    """Vocabulario de notas: tokens especiales + (pc, octava, duracion)."""
    vocab = list(SPECIAL_TOKENS)
    for octv in (-1, 0, 1, 2):
        for pc in range(12):
            for dur in DURATIONS:
                vocab.append(f"N_{pc}_{octv}_{dur}")
    for dur in DURATIONS:
        vocab.append(f"R_{dur}")
    tok2id = {t: i for i, t in enumerate(vocab)}
    id2tok = {i: t for i, t in enumerate(vocab)}
    return tok2id, id2tok


def event_to_token(ev):
    if ev["pc"] is None:
        return f"R_{ev['dur']}"
    octv = max(-1, min(2, ev["octave"]))
    return f"N_{ev['pc']}_{octv}_{ev['dur']}"


def token_to_event(tok):
    if tok.startswith("R_"):
        return {"pc": None, "octave": 0, "dur": float(tok.split("_")[1])}
    if tok.startswith("N_"):
        _, pc, octv, dur = tok.split("_")
        return {"pc": int(pc), "octave": int(octv), "dur": float(dur)}
    return {"pc": None, "octave": 0, "dur": 1.0}

# ==============================================================================
#  EMBEDDING EMOCIONAL DE TEXTO (sentence-transformers, import perezoso)
# ==============================================================================

_EMOTION_MODEL = None


def _load_emotion_model(device="cpu"):
    """Carga (una vez) el modelo de sentence-embeddings multilingue.
    Se descarga automaticamente la primera vez y se cachea localmente."""
    global _EMOTION_MODEL
    if _EMOTION_MODEL is not None:
        return _EMOTION_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers no disponible. Instala con:\n"
            "  pip install sentence-transformers\n"
            "(esto instalara tambien torch si no esta presente)"
        )
    print(f"[emotion] Cargando modelo de embeddings '{EMBEDDING_MODEL_NAME}' "
          f"(puede tardar la primera vez)...")
    _EMOTION_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _EMOTION_MODEL


def embed_texts(texts, device="cpu", batch_size=64):
    """Calcula embeddings (N, EMBEDDING_DIM) para una lista de textos.
    Textos vacios se mapean al vector cero."""
    model = _load_emotion_model(device=device)
    non_empty_idx = [i for i, t in enumerate(texts) if t and t.strip()]
    embeddings = [None] * len(texts)

    if non_empty_idx:
        to_embed = [texts[i] for i in non_empty_idx]
        vecs = model.encode(to_embed, batch_size=batch_size,
                             show_progress_bar=False, convert_to_numpy=True)
        for idx, vec in zip(non_empty_idx, vecs):
            embeddings[idx] = vec

    import numpy as np
    zero = np.zeros(EMBEDDING_DIM, dtype="float32")
    for i in range(len(embeddings)):
        if embeddings[i] is None:
            embeddings[i] = zero
    return np.stack(embeddings)


# ==============================================================================
#  MOTOR MARKOV (clusters emocionales, historial continuo) - fallback ligero
# ==============================================================================
#
# Agrupa los versos del corpus en K clusters segun su embedding emocional
# (k-means simple, sin sklearn). Construye un bigrama de notas GLOBAL (no
# reiniciado por frase) pero con conteos separados por cluster:
#   model[cluster][prev_token] -> Counter(next_token)
# En generacion, se mantiene prev_tok a traves de toda la cancion (el
# historial NO se reinicia entre versos); solo cambia el cluster usado para
# elegir la distribucion de muestreo en cada verso, dando fluidez en las
# transiciones y a la vez modulacion emocional por verso.

def _kmeans_simple(vectors, k, n_iter=20, seed=42):
    """K-means minimalista en numpy. Devuelve (labels, centroids)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    n = vectors.shape[0]
    k = min(k, n)
    idx = rng.choice(n, size=k, replace=False)
    centroids = vectors[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = ((vectors[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = dists.argmin(axis=1)
        if (new_labels == labels).all():
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            mask = labels == c
            if mask.any():
                centroids[c] = vectors[mask].mean(axis=0)
    return labels, centroids


def build_markov_model(songs, n_clusters=6):
    """Construye un modelo bigrama por cluster emocional, recorriendo cada
    cancion como secuencia continua (el bigrama no se reinicia entre frases
    dentro de una misma cancion).
    Devuelve (model, start_tokens, centroids) donde:
      model[cluster] -> dict[prev_token] -> Counter(next_token)
      start_tokens[cluster] -> Counter(first_token de cada cancion)
    """
    import numpy as np

    all_phrases = flatten_phrases(songs)
    texts = [p["text"] for p in all_phrases]
    embeddings = embed_texts(texts)
    labels, centroids = _kmeans_simple(embeddings, n_clusters)

    song_phrase_clusters = []
    flat_idx = 0
    for song in songs:
        clusters_for_song = []
        for _ in song["phrases"]:
            clusters_for_song.append(int(labels[flat_idx]))
            flat_idx += 1
        song_phrase_clusters.append(clusters_for_song)

    model = defaultdict(lambda: defaultdict(Counter))
    start_tokens = defaultdict(Counter)

    for song, clusters_for_song in zip(songs, song_phrase_clusters):
        prev_tok = None
        for phrase, cluster in zip(song["phrases"], clusters_for_song):
            for ev in phrase["events"]:
                tok = event_to_token(ev)
                if prev_tok is None:
                    start_tokens[cluster][tok] += 1
                else:
                    model[cluster][prev_tok][tok] += 1
                prev_tok = tok

    return model, start_tokens, centroids


def markov_sample(candidates, rng, temperature=1.0):
    if not candidates:
        return None
    items = list(candidates.items())
    weights = [max(1e-6, c) ** (1.0 / max(1e-6, temperature)) for _, c in items]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for (tok, _), w in zip(items, weights):
        acc += w
        if r <= acc:
            return tok
    return items[-1][0]


def nearest_cluster(vec, centroids):
    import numpy as np
    dists = ((centroids - vec[None, :]) ** 2).sum(axis=1)
    return int(dists.argmin())


def generate_markov(songs, line_texts, line_embeddings, profile, temperature,
                     notes_per_line, n_clusters=6, seed=42):
    """Genera una secuencia continua de eventos para toda la cancion usando
    el motor markov. Devuelve una lista de listas de eventos (una por verso),
    manteniendo el historial de transiciones entre versos."""
    model, start_tokens, centroids = build_markov_model(songs, n_clusters=n_clusters)

    rng = random.Random(seed)
    n_lines = len(line_texts)

    all_events = []
    prev_tok = None

    for li, (text, vec) in enumerate(zip(line_texts, line_embeddings)):
        cluster = nearest_cluster(vec, centroids)
        local_temp = temperature * _profile_temperature(profile, li, n_lines)

        n_notes = notes_per_line or _estimate_notes_for_line(text)
        phrase_events = []
        for i in range(n_notes):
            if prev_tok is not None and prev_tok in model.get(cluster, {}):
                tok = markov_sample(model[cluster][prev_tok], rng, local_temp)
            elif prev_tok is not None:
                merged = Counter()
                for c in model:
                    if prev_tok in model[c]:
                        merged.update(model[c][prev_tok])
                tok = markov_sample(merged, rng, local_temp) if merged else \
                    markov_sample(start_tokens.get(cluster, {}), rng, local_temp)
            else:
                tok = markov_sample(start_tokens.get(cluster, {}), rng, local_temp)

            if tok is None or tok in (BOS, EOS, PAD, PHRASE, REST):
                ev = {"pc": rng.randrange(12), "octave": 0, "dur": 1.0}
            else:
                ev = token_to_event(tok)
            ev = _apply_profile_bias(ev, profile, i, n_notes)
            phrase_events.append(ev)
            prev_tok = event_to_token(ev)

        all_events.append(phrase_events)

    return all_events


def _estimate_notes_for_line(text):
    """Estima un numero de notas razonable para un verso, a partir de su
    longitud (aprox. 1 nota cada 2-3 caracteres no-espacio)."""
    n_chars = len(re.sub(r"\s+", "", text))
    return max(4, min(16, round(n_chars / 2.5)))


# ==============================================================================
#  PERFILES EMOCIONALES MANUALES - utilidades comunes a ambos motores
# ==============================================================================

def _profile_temperature(profile, idx, total):
    """Multiplicador de temperatura, mayor cerca del climax del perfil."""
    cfg = EMOTIONAL_PROFILES.get(profile, EMOTIONAL_PROFILES["neutral"])
    if total <= 1:
        return cfg["temperature_scale"]
    pos = idx / max(1, total - 1)
    climax = cfg["climax_pos"]
    dist = abs(pos - climax)
    bump = max(0.0, 1.0 - dist * 2.0) * 0.4
    return cfg["temperature_scale"] * (1.0 + bump)


def _apply_profile_bias(ev, profile, idx, total):
    """Aplica sesgo de octava/duracion del perfil, mas intenso cerca del
    climax de la pieza."""
    cfg = EMOTIONAL_PROFILES.get(profile, EMOTIONAL_PROFILES["neutral"])
    if total <= 1:
        return ev
    pos = idx / max(1, total - 1)
    climax = cfg["climax_pos"]
    dist = abs(pos - climax)
    intensity = max(0.0, 1.0 - dist * 1.5)

    ev = dict(ev)
    if ev["pc"] is not None:
        octave_shift = cfg["octave_bias"] * intensity
        if octave_shift > 0.3 and random.random() < octave_shift:
            ev["octave"] = min(2, ev["octave"] + 1)
        elif octave_shift < -0.3 and random.random() < -octave_shift:
            ev["octave"] = max(-1, ev["octave"] - 1)

    dur_shift = cfg["duration_bias"] * intensity
    if dur_shift > 0.15 and random.random() < dur_shift:
        idx_dur = DURATIONS.index(ev["dur"]) if ev["dur"] in DURATIONS else 1
        ev["dur"] = DURATIONS[min(len(DURATIONS) - 1, idx_dur + 1)]
    elif dur_shift < -0.15 and random.random() < -dur_shift:
        idx_dur = DURATIONS.index(ev["dur"]) if ev["dur"] in DURATIONS else 1
        ev["dur"] = DURATIONS[max(0, idx_dur - 1)]

    return ev

# ==============================================================================
#  DATASET (secuencias continuas por cancion, con segment embedding)
# ==============================================================================

def build_training_sequences(songs, note_tok2id, max_seq_len=128):
    """
    Para cada cancion, construye una secuencia continua:
      <bos> [eventos frase 1] <phrase> [eventos frase 2] <phrase> ... <eos>
    junto con un array paralelo `emotion_idx` que indica, para cada posicion,
    el INDICE de frase (0-based) cuyo embedding emocional debe usarse en esa
    posicion (el token <phrase> usa el indice de la frase que ACABA, y el
    <bos> usa el indice de la primera frase).

    Devuelve:
      sequences: lista de (note_ids, emotion_idx, song_texts)
        - note_ids: lista de ints, longitud max_seq_len (con padding)
        - emotion_idx: lista de ints (indices dentro de song_texts), misma longitud
        - song_texts: lista de textos de los versos de esta cancion (para
          poder calcular sus embeddings despues, agrupando por cancion)
    """
    sequences = []
    for song in songs:
        phrases = song["phrases"]
        if not phrases:
            continue

        note_ids = [note_tok2id[BOS]]
        emotion_idx = [0]
        song_texts = [p["text"] for p in phrases]

        for pi, phrase in enumerate(phrases):
            for ev in phrase["events"]:
                if len(note_ids) >= max_seq_len - 1:
                    break
                tok = event_to_token(ev)
                if tok not in note_tok2id:
                    tok = REST
                note_ids.append(note_tok2id[tok])
                emotion_idx.append(pi)
            if len(note_ids) >= max_seq_len - 1:
                break
            if pi < len(phrases) - 1:
                note_ids.append(note_tok2id[PHRASE])
                emotion_idx.append(pi)

        if len(note_ids) < max_seq_len:
            note_ids.append(note_tok2id[EOS])
            emotion_idx.append(len(phrases) - 1)

        pad_id = note_tok2id[PAD]
        last_idx = emotion_idx[-1] if emotion_idx else 0
        while len(note_ids) < max_seq_len:
            note_ids.append(pad_id)
            emotion_idx.append(last_idx)

        sequences.append((note_ids[:max_seq_len], emotion_idx[:max_seq_len], song_texts))

    return sequences


# ==============================================================================
#  TORCH (import perezoso con stub) - modelos del motor transformer
# ==============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    class _TorchStub:
        Module = object

        def __getattr__(self, name):
            raise ImportError(
                "PyTorch no disponible. Instala con: pip install torch\n"
                "(el comando 'inspect' sobre corpus no lo requiere)"
            )
    torch = _TorchStub()
    nn = _TorchStub()
    F = _TorchStub()
    _TORCH_OK = False


def _require_torch():
    if not _TORCH_OK:
        raise ImportError(
            "Esta operacion requiere PyTorch. Instala con: pip install torch"
        )


class RelativeGlobalAttention(nn.Module):
    """
    Atencion multi-cabeza con embeddings relativos de posicion (truco de
    skewing, Music Transformer / Huang et al. 2019). Permite que el modelo
    reconozca patrones melodicos (intervalos, motivos) de forma relativa a
    su posicion dentro de la secuencia, no solo absoluta.
    """

    def __init__(self, dim, heads, max_seq=512):
        super().__init__()
        self.h = heads
        self.d = dim
        self.dh = dim // heads
        self.max_seq = max_seq

        self.Wq = nn.Linear(dim, dim)
        self.Wk = nn.Linear(dim, dim)
        self.Wv = nn.Linear(dim, dim)
        self.fc = nn.Linear(dim, dim)

        self.E = nn.Parameter(torch.randn(max_seq, self.dh) * 0.02)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        def project(t, W):
            t = W(t).view(B, T, self.h, self.dh)
            return t.permute(0, 2, 1, 3)

        q = project(x, self.Wq)
        k = project(x, self.Wk)
        v = project(x, self.Wv)

        E = self.E[max(0, self.max_seq - T):, :]
        QE = torch.einsum('bhld,md->bhlm', q, E)
        QE = self._qe_masking(QE)
        Srel = self._skewing(QE)

        logits = torch.matmul(q, k.transpose(-2, -1)) + Srel
        logits = logits / math.sqrt(self.dh)

        if mask is not None:
            logits = logits + mask

        w = torch.softmax(logits, dim=-1)
        out = torch.matmul(w, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.d)
        return self.fc(out)

    @staticmethod
    def _qe_masking(qe):
        L = qe.size(-1)
        S = qe.size(-2)
        lengths = torch.arange(L - 1, L - S - 1, -1, device=qe.device)
        idx = torch.arange(L, device=qe.device).unsqueeze(0)
        mask = idx >= lengths.unsqueeze(1)
        return qe * mask.to(qe.dtype)

    @staticmethod
    def _skewing(tensor):
        B, H, L, M = tensor.shape
        padded = F.pad(tensor, (1, 0))
        reshaped = padded.reshape(B, H, M + 1, L)
        return reshaped[:, :, 1:, :]


class RelativeAttentionBlock(nn.Module):
    """Bloque Transformer (atencion relativa + FFN) con norm y residuales."""

    def __init__(self, dim, heads, ff_dim, dropout, max_seq):
        super().__init__()
        self.attn = RelativeGlobalAttention(dim, heads, max_seq=max_seq)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask=mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


def build_model(note_vocab_size, d_model=128, n_heads=4, n_layers=3,
                 max_seq=128, dropout=0.1, emotion_dim=EMBEDDING_DIM):
    _require_torch()

    class MelodyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.note_embed = nn.Embedding(note_vocab_size, d_model)
            self.pos_embed = nn.Embedding(max_seq, d_model)
            self.emotion_proj = nn.Sequential(
                nn.Linear(emotion_dim, d_model), nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.blocks = nn.ModuleList([
                RelativeAttentionBlock(d_model, n_heads, d_model * 4, dropout, max_seq)
                for _ in range(n_layers)
            ])
            self.out_proj = nn.Linear(d_model, note_vocab_size)

        def forward(self, note_ids, segment_emotion):
            """
            note_ids: (B, T)
            segment_emotion: (B, T, emotion_dim) - embedding emocional del
              verso correspondiente a cada posicion (segment embedding).
            """
            B, T = note_ids.shape
            pos = torch.arange(T, device=note_ids.device).unsqueeze(0).expand(B, T)
            cond = self.emotion_proj(segment_emotion)  # (B, T, d_model)

            x = self.note_embed(note_ids) + self.pos_embed(pos) + cond

            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=note_ids.device), diagonal=1
            )
            for block in self.blocks:
                x = block(x, mask=causal_mask)
            return self.out_proj(x)

    return MelodyModel()

# ==============================================================================
#  MODO TRAIN
# ==============================================================================

def _build_segment_emotion_tensor(emotion_idx_batch, song_embeddings_batch):
    """
    emotion_idx_batch: (B, T) array de indices de frase
    song_embeddings_batch: lista de B arrays (n_phrases_i, emotion_dim)

    Devuelve un array (B, T, emotion_dim) indexando, para cada posicion,
    el embedding de la frase correspondiente.
    """
    import numpy as np
    B, T = emotion_idx_batch.shape
    out = np.zeros((B, T, EMBEDDING_DIM), dtype="float32")
    for b in range(B):
        emb = song_embeddings_batch[b]
        idx = emotion_idx_batch[b]
        idx_clamped = np.clip(idx, 0, emb.shape[0] - 1)
        out[b] = emb[idx_clamped]
    return out


def cmd_train(args):
    _require_torch()
    import numpy as np

    print(f"[train] Buscando ficheros .abc en {args.corpus} ...")
    all_songs, abc_files = collect_corpus_songs(args.corpus)

    if not abc_files:
        print(f"[train] No se encontraron ficheros .abc en {args.corpus}", file=sys.stderr)
        sys.exit(1)

    all_phrases = flatten_phrases(all_songs)
    print(f"[train] Ficheros ABC encontrados: {len(abc_files)}")
    print(f"[train] Canciones (bloques X:): {len(all_songs)}")
    print(f"[train] Frases totales: {len(all_phrases)}")
    n_with_text = sum(1 for p in all_phrases if p["text"])
    print(f"[train] Frases con verso asociado: {n_with_text} / {len(all_phrases)}")

    if not all_songs:
        print("[train] No se pudo extraer ninguna cancion valida. Abortando.", file=sys.stderr)
        sys.exit(1)

    note_tok2id, _ = build_note_vocab()

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    resumed = False
    checkpoint = None
    if args.resume and os.path.exists(args.model_out):
        print(f"[train] Cargando checkpoint existente desde {args.model_out} ...")
        checkpoint = torch.load(args.model_out, map_location="cpu")
        cfg = checkpoint["config"]
        resumed = True
    else:
        cfg = {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "max_seq_len": args.max_seq_len,
            "emotion_model": EMBEDDING_MODEL_NAME,
            "emotion_dim": EMBEDDING_DIM,
        }

    sequences = build_training_sequences(all_songs, note_tok2id, max_seq_len=cfg["max_seq_len"])
    print(f"[train] Secuencias de entrenamiento (canciones): {len(sequences)}")
    if len(sequences) < 4:
        print("[train] Muy pocas secuencias para entrenar de forma fiable.", file=sys.stderr)

    print("[train] Calculando embeddings emocionales de los versos...")
    song_embeddings = []
    for note_ids, emotion_idx, song_texts in sequences:
        emb = embed_texts(song_texts, device=device)
        song_embeddings.append(emb)
    print(f"[train] Embeddings calculados para {len(song_embeddings)} canciones.")

    model = build_model(
        len(note_tok2id), d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], max_seq=cfg["max_seq_len"], dropout=args.dropout,
        emotion_dim=cfg["emotion_dim"],
    )

    if resumed:
        model.load_state_dict(checkpoint["model_state"], strict=False)
        print("[train] Pesos del checkpoint cargados (resume).")

    model.to(device)

    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=note_tok2id[PAD],
                                     label_smoothing=args.label_smooth)

    note_ids_arr = np.array([s[0] for s in sequences], dtype=np.int64)
    emotion_idx_arr = np.array([s[1] for s in sequences], dtype=np.int64)

    n = note_ids_arr.shape[0]
    batch_size = min(args.batch_size, n)

    print(f"[train] Entrenando {args.epochs} epocas en {device} "
          f"(d_model={cfg['d_model']}, max_seq_len={cfg['max_seq_len']}, "
          f"batch={batch_size}, label_smooth={args.label_smooth})...")

    for epoch in range(1, args.epochs + 1):
        perm = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]

            note_ids = torch.tensor(note_ids_arr[idx], dtype=torch.long, device=device)
            emotion_idx_batch = emotion_idx_arr[idx]
            song_embeddings_batch = [song_embeddings[j] for j in idx]
            segment_emotion = _build_segment_emotion_tensor(emotion_idx_batch, song_embeddings_batch)
            segment_emotion = torch.tensor(segment_emotion, dtype=torch.float32, device=device)

            inp_notes = note_ids[:, :-1]
            tgt_notes = note_ids[:, 1:]
            inp_emotion = segment_emotion[:, :-1, :]

            logits = model(inp_notes, inp_emotion)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_notes.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        if epoch % max(1, args.epochs // 20) == 0 or epoch == 1:
            print(f"[train] Epoca {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}")

    checkpoint_out = {
        "model_state": model.state_dict(),
        "note_tok2id": note_tok2id,
        "config": cfg,
        "version": VERSION,
        "n_sequences": len(sequences),
        "n_source_files": len(abc_files),
    }
    torch.save(checkpoint_out, args.model_out)
    print(f"[train] Modelo guardado en {args.model_out}")

# ==============================================================================
#  MODO GENERATE
# ==============================================================================

def lyrics_to_lines(text):
    """Cada linea no vacia del fichero de lyrics es un verso/frase."""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            lines.append(line)
    return lines


def _build_scale_mask(note_tok2id: dict, key_pc: int, mode: str) -> "torch.Tensor":
    """
    Devuelve un tensor booleano de longitud vocab_size donde True indica que
    el token esta FUERA de la escala (key_pc, mode). Solo afecta a tokens N_*;
    los tokens especiales y los silencios R_* se dejan intactos.

    Se usa para poner a -inf los logits de notas cromaticas ajenas a la
    tonalidad antes de muestrear, mejorando la coherencia tonal.
    """
    scale_pcs = set(
        (key_pc + iv) % 12
        for iv in SCALE_INTERVALS.get(mode, SCALE_INTERVALS["major"])
    )
    mask = torch.zeros(len(note_tok2id), dtype=torch.bool)
    for tok, idx in note_tok2id.items():
        if tok.startswith("N_"):
            parts = tok.split("_")          # ['N', pc, octave, dur]
            pc = int(parts[1])
            if pc not in scale_pcs:
                mask[idx] = True
    return mask


def _build_interval_bias(
    note_tok2id: dict,
    prev_ev: dict,
    profile: str,
    bias_strength: float = 0.4,
) -> "torch.Tensor":
    """
    Devuelve un tensor de logit-bias de longitud vocab_size calculado a partir
    de INTERVAL_WEIGHTS[profile] y el evento anterior (prev_ev).

    Para cada token N_* calcula el intervalo en semitonos respecto a prev_ev,
    busca su peso en la tabla del perfil (con fallback a 0.5 si no esta) y
    convierte: bias = bias_strength * log(weight).

    Tokens especiales y silencios R_* reciben bias 0 (sin efecto).
    bias_strength controla cuanto se desvian los logits del modelo base;
    0.0 = desactivado, 1.0 = sesgo fuerte.
    """
    iw = INTERVAL_WEIGHTS.get(profile, INTERVAL_WEIGHTS["neutral"])
    bias = torch.zeros(len(note_tok2id))

    if prev_ev is None or prev_ev.get("pc") is None:
        return bias

    prev_midi = prev_ev["pc"] + 12 * prev_ev["octave"]

    for tok, idx in note_tok2id.items():
        if not tok.startswith("N_"):
            continue
        parts = tok.split("_")              # ['N', pc, octave, dur]
        pc   = int(parts[1])
        octv = int(parts[2])
        curr_midi = pc + 12 * octv
        interval  = curr_midi - prev_midi   # semitonos con signo

        # Buscar peso: primero exacto, luego por modulo de octava
        weight = iw.get(interval, iw.get(interval % 12, 0.5))
        bias[idx] = bias_strength * math.log(max(weight, 1e-6))

    return bias


def sample_from_logits(logits, temperature=1.0, top_k=0,
                       scale_mask=None, interval_bias=None):
    """
    Muestrea un token de los logits del modelo.

    Parametros adicionales:
      scale_mask    (BoolTensor, vocab_size) — tokens fuera de escala: se
                    ponen a -inf antes de escalar por temperatura.  [mejora 3]
      interval_bias (FloatTensor, vocab_size) — sesgo aditivo en log-space
                    calculado por INTERVAL_WEIGHTS del perfil.  [mejora 1]
    El orden de aplicacion es: mask → bias → temperatura → top-k → softmax.
    """
    # Mejora 3: suprimir notas fuera de escala
    if scale_mask is not None:
        logits = logits.clone()
        logits[scale_mask] = float("-inf")

    # Mejora 1: sesgo de intervalo por perfil emocional
    if interval_bias is not None:
        logits = logits + interval_bias.to(logits.device)

    logits = logits / max(1e-6, temperature)
    if top_k > 0:
        v, ix = torch.topk(logits, min(top_k, logits.size(-1)))
        probs = torch.zeros_like(logits).scatter_(0, ix, torch.softmax(v, dim=-1))
    else:
        probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def write_midi(phrases_events, out_path, key_pc=0, tempo_bpm=100, unit_len_beats=0.5,
               line_texts=None):
    import mido

    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.MetaMessage("track_name", name="lyrics_melody", time=0))

    base_note = 60  # C4
    ticks_per_unit = int(480 * unit_len_beats)

    for li, events in enumerate(phrases_events):
        if line_texts:
            track.append(mido.MetaMessage("lyrics", text=line_texts[li], time=0))
        for ev in events:
            dur_ticks = int(ticks_per_unit * ev["dur"])
            if ev["pc"] is None:
                track.append(mido.Message("note_off", note=0, velocity=0, time=dur_ticks))
                continue
            pitch = base_note + key_pc + ev["pc"] + 12 * ev["octave"]
            pitch = max(0, min(127, pitch))
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=dur_ticks))

    mid.save(out_path)


DUR_TO_ABC = {0.5: "/2", 1.0: "", 1.5: "3/2", 2.0: "2", 3.0: "3", 4.0: "4"}
PC_TO_ABC = {0: "C", 1: "^C", 2: "D", 3: "^D", 4: "E", 5: "F", 6: "^F",
              7: "G", 8: "^G", 9: "A", 10: "^A", 11: "B"}


def write_abc(phrases_events, out_path, line_texts=None, key_name="C", meter="4/4",
               unit_len="1/8", title="Generado por lyrics_melody.py"):
    lines = [f"X:1", f"T:{title}", f"M:{meter}", f"L:{unit_len}", f"K:{key_name}"]

    for li, events in enumerate(phrases_events):
        melody_chars = []
        for ev in events:
            if ev["pc"] is None:
                melody_chars.append("z" + DUR_TO_ABC.get(ev["dur"], ""))
            else:
                letter = PC_TO_ABC[ev["pc"]]
                octv = ev["octave"]
                if octv >= 1:
                    letter = letter + "'" * octv
                elif octv <= -1:
                    letter = letter + "," * (-octv)
                melody_chars.append(letter + DUR_TO_ABC.get(ev["dur"], ""))
        melody_chars.append("|")
        lines.append(" ".join(melody_chars))
        if line_texts:
            lines.append("w: " + line_texts[li])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _smooth_octave(ev, prev_ev, max_leap=7):
    """Ajusta la octava de ev para minimizar el salto de semitonos respecto a
    prev_ev. Si el intervalo minimo posible supera max_leap semitonos, elige
    igualmente la octava mas proxima. Solo actua sobre notas (pc != None)."""
    if ev["pc"] is None or prev_ev is None or prev_ev["pc"] is None:
        return ev
    prev_midi = prev_ev["pc"] + 12 * prev_ev["octave"]
    best_oct = ev["octave"]
    best_dist = abs((ev["pc"] + 12 * ev["octave"]) - prev_midi)
    for oct_cand in (-1, 0, 1, 2):
        d = abs((ev["pc"] + 12 * oct_cand) - prev_midi)
        if d < best_dist:
            best_dist = d
            best_oct = oct_cand
    ev = dict(ev)
    ev["octave"] = best_oct
    return ev


def generate_transformer(args, line_texts, line_embeddings):
    """
    Genera la cancion completa como UNA secuencia autoregresiva continua.
    En cada paso, el segment-embedding emocional usado es el de la frase
    "actual" (la que se esta generando en ese momento); al emitir <phrase>
    se avanza a la frase siguiente, manteniendo todo el historial de notas.
    """
    checkpoint = torch.load(args.model, map_location="cpu")
    note_tok2id = checkpoint["note_tok2id"]
    note_id2tok = {v: k for k, v in note_tok2id.items()}
    cfg = checkpoint["config"]

    model = build_model(
        len(note_tok2id), d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], max_seq=cfg["max_seq_len"],
        emotion_dim=cfg.get("emotion_dim", EMBEDDING_DIM),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    max_seq_len = cfg["max_seq_len"]
    n_lines = len(line_texts)

    note_ids = [note_tok2id[BOS]]
    phrases_events = [[] for _ in range(n_lines)]

    current_line = 0
    notes_in_current_phrase = 0
    target_notes = args.notes_per_line or _estimate_notes_for_line(line_texts[0])
    prev_ev = None  # ultima nota generada (para suavizado de octava)

    emotion_vecs = [torch.tensor(v, dtype=torch.float32) for v in line_embeddings]

    max_total_notes = sum(
        (args.notes_per_line or _estimate_notes_for_line(t)) for t in line_texts
    ) + n_lines * 2

    total_notes = 0

    # ── Mejora 3: mascara de escala (se construye una sola vez) ───────────────
    key_pc = parse_key(args.key)
    scale_mode = getattr(args, "mode", "major")
    scale_mask = _build_scale_mask(note_tok2id, key_pc, scale_mode)
    print(f"[generate] Mascara de escala: {args.key} {scale_mode} "
          f"({scale_mask.sum().item()} tokens suprimidos)")

    with torch.no_grad():
        while current_line < n_lines and total_notes < max_total_notes:
            window_ids = note_ids[-max_seq_len:]
            T = len(window_ids)
            seq = torch.tensor(window_ids, dtype=torch.long).unsqueeze(0)  # (1, T)

            # Aproximacion: usamos el embedding emocional de la frase actual
            # para toda la ventana de contexto. El historial de notas
            # (atencion causal) sigue dando continuidad melodica con la
            # frase anterior; el segment-embedding solo module el "color"
            # emocional de las proximas notas a generar.
            seg_emb = emotion_vecs[current_line].unsqueeze(0).unsqueeze(0).expand(1, T, -1)

            local_temp = args.temperature * _profile_temperature(args.profile, current_line, n_lines)

            logits = model(seq, seg_emb)
            next_logits = logits[0, -1, :].clone()

            next_logits[note_tok2id[PAD]] = float("-inf")
            next_logits[note_tok2id[BOS]] = float("-inf")

            if current_line < n_lines - 1:
                next_logits[note_tok2id[EOS]] = float("-inf")
            else:
                next_logits[note_tok2id[PHRASE]] = float("-inf")

            force_advance = notes_in_current_phrase >= target_notes
            if force_advance:
                if current_line < n_lines - 1:
                    next_id = note_tok2id[PHRASE]
                else:
                    next_id = note_tok2id[EOS]
            else:
                # Mejora 1: bias de intervalo recalculado por nota (depende de prev_ev)
                iv_bias = _build_interval_bias(
                    note_tok2id, prev_ev, args.profile,
                    bias_strength=getattr(args, "interval_bias_strength", 0.4),
                )
                next_id = sample_from_logits(
                    next_logits, temperature=local_temp, top_k=args.top_k,
                    scale_mask=scale_mask, interval_bias=iv_bias,
                )

            note_ids.append(next_id)
            tok = note_id2tok[next_id]

            if tok == EOS:
                break
            if tok == PHRASE:
                current_line += 1
                notes_in_current_phrase = 0
                if current_line < n_lines:
                    target_notes = args.notes_per_line or _estimate_notes_for_line(
                        line_texts[current_line]
                    )
                continue

            if tok in (PAD, BOS, REST):
                ev = {"pc": None, "octave": 0, "dur": 1.0}
            else:
                ev = token_to_event(tok)
            ev = _apply_profile_bias(ev, args.profile, notes_in_current_phrase, target_notes)
            ev = _smooth_octave(ev, prev_ev)
            if ev["pc"] is not None:
                prev_ev = ev
            phrases_events[current_line].append(ev)
            notes_in_current_phrase += 1
            total_notes += 1

    for i, events in enumerate(phrases_events):
        if not events:
            phrases_events[i] = [{"pc": 0, "octave": 0, "dur": 1.0}]

    return phrases_events


def cmd_generate(args):
    with open(args.lyrics, "r", encoding="utf-8") as f:
        text = f.read()
    line_texts = lyrics_to_lines(text)
    if not line_texts:
        print("[generate] El fichero de letras esta vacio.", file=sys.stderr)
        sys.exit(1)

    print(f"[generate] Versos a musicalizar: {len(line_texts)}")
    print(f"[generate] Motor: {args.engine}  Perfil: {args.profile}")

    key_pc = parse_key(args.key)

    device = "cuda" if (_TORCH_OK and torch.cuda.is_available() and not args.cpu) else "cpu"
    print("[generate] Calculando embeddings emocionales de los versos...")
    line_embeddings = embed_texts(line_texts, device=device)

    if args.engine == "transformer":
        if not args.model:
            print("[generate] --engine transformer requiere --model", file=sys.stderr)
            sys.exit(1)
        _require_torch()
        phrases_events = generate_transformer(args, line_texts, line_embeddings)
    else:
        if not args.corpus:
            print("[generate] --engine markov requiere --corpus", file=sys.stderr)
            sys.exit(1)
        all_songs, abc_files = collect_corpus_songs(args.corpus)
        if not all_songs:
            print(f"[generate] No se encontraron canciones en {args.corpus}", file=sys.stderr)
            sys.exit(1)
        print(f"[generate] Corpus markov: {len(abc_files)} ficheros, {len(all_songs)} canciones")
        phrases_events = generate_markov(
            all_songs, line_texts, line_embeddings, args.profile, args.temperature,
            notes_per_line=args.notes_per_line, n_clusters=args.n_clusters, seed=args.seed,
        )

    write_midi(phrases_events, args.out, key_pc=key_pc, tempo_bpm=args.tempo,
               unit_len_beats=args.unit_beats, line_texts=line_texts)
    print(f"[generate] MIDI escrito en {args.out}")

    if args.abc_out:
        write_abc(phrases_events, args.abc_out, line_texts=line_texts, key_name=args.key,
                  meter=args.meter, title=os.path.basename(args.lyrics))
        print(f"[generate] ABC escrito en {args.abc_out}")

# ==============================================================================
#  MODO INSPECT
# ==============================================================================

def cmd_inspect(args):
    if args.model:
        if not _TORCH_OK:
            print("[inspect] Aviso: PyTorch no disponible; no se puede cargar "
                  "el checkpoint.", file=sys.stderr)
        else:
            checkpoint = torch.load(args.model, map_location="cpu")
            cfg = checkpoint["config"]
            print(f"Modelo: {args.model}")
            print(f"  Version: {checkpoint.get('version', '?')}")
            print(f"  Configuracion: {json.dumps(cfg, indent=2)}")
            print(f"  Secuencias (canciones) de entrenamiento: {checkpoint.get('n_sequences', '?')}")
            print(f"  Ficheros fuente: {checkpoint.get('n_source_files', '?')}")
            print(f"  Tamano vocab notas: {len(checkpoint['note_tok2id'])}")

    if args.corpus:
        all_songs, abc_files = collect_corpus_songs(args.corpus)
        phrases = flatten_phrases(all_songs)
        print(f"\nCorpus: {args.corpus}")
        print(f"  Ficheros .abc: {len(abc_files)}")
        print(f"  Canciones (bloques X:): {len(all_songs)}")
        print(f"  Frases: {len(phrases)}")
        print(f"  Notas/eventos totales: {sum(len(p['events']) for p in phrases)}")
        n_with_text = sum(1 for p in phrases if p["text"])
        print(f"  Frases con verso asociado: {n_with_text} / {len(phrases)}")

        if all_songs:
            phrases_per_song = [len(s["phrases"]) for s in all_songs]
            print(f"  Frases por cancion: min={min(phrases_per_song)} "
                  f"max={max(phrases_per_song)} media={sum(phrases_per_song)/len(phrases_per_song):.1f}")

        if phrases:
            lengths = [len(p["events"]) for p in phrases]
            print(f"  Longitud de frase: min={min(lengths)} "
                  f"max={max(lengths)} media={sum(lengths)/len(lengths):.1f}")

            dur_counter = Counter()
            pc_counter = Counter()
            for p in phrases:
                for ev in p["events"]:
                    dur_counter[ev["dur"]] += 1
                    if ev["pc"] is not None:
                        pc_counter[ev["pc"]] += 1

            print("  Distribucion de duraciones:")
            for d in DURATIONS:
                print(f"    {d:>4}: {dur_counter.get(d, 0)}")

            total_pc = sum(pc_counter.values()) or 1
            print("  Distribucion de grados (relativos a la tonica):")
            for pc in range(12):
                bar = "#" * min(40, pc_counter.get(pc, 0) // max(1, total_pc // 200))
                print(f"    {pc:>2} ({PITCH_CLASSES[pc]:>2}): {pc_counter.get(pc, 0):>5} {bar}")

    if not args.model and not args.corpus:
        print("Indica --model y/o --corpus para inspeccionar.", file=sys.stderr)
        print("\nPerfiles emocionales disponibles:")
        for name, cfg in EMOTIONAL_PROFILES.items():
            print(f"  {name:<14} climax_pos={cfg['climax_pos']:.2f}  "
                  f"temp_scale={cfg['temperature_scale']:.2f}")
        print(f"\nModelo de embeddings: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM} dims)")


# ==============================================================================
#  CLI
# ==============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generacion de melodias condicionadas por el contenido "
                     "emocional de lyrics, con fluidez entre frases "
                     "(ABC -> modelo -> MIDI)",
    )
    parser.add_argument("--version", action="version", version=f"lyrics_melody {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Entrenar modelo a partir de un corpus ABC")
    p_train.add_argument("--corpus", required=True,
                          help="Fichero .abc o carpeta con ficheros .abc")
    p_train.add_argument("--model-out", default="lyrics_melody.pt")
    p_train.add_argument("--resume", action="store_true",
                          help="Retomar entrenamiento desde --model-out si existe")
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--d-model", type=int, default=128)
    p_train.add_argument("--n-heads", type=int, default=4)
    p_train.add_argument("--n-layers", type=int, default=3)
    p_train.add_argument("--max-seq-len", type=int, default=128,
                          help="Longitud maxima de secuencia (cancion completa). "
                               "Aumentar si tus canciones tienen muchas frases.")
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--label-smooth", type=float, default=0.1,
                          help="Label smoothing en la cross-entropy (default: 0.1)")
    p_train.add_argument("--cpu", action="store_true", help="Forzar uso de CPU")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Generar melodia a partir de lyrics")
    p_gen.add_argument("--lyrics", required=True,
                        help="Fichero de texto con la letra (una linea = un verso = una frase)")
    p_gen.add_argument("--out", required=True, help="Fichero MIDI de salida")
    p_gen.add_argument("--abc-out", default=None, help="Fichero ABC de salida (opcional)")
    p_gen.add_argument("--engine", choices=["transformer", "markov"], default="transformer",
                        help="Motor de generacion (default: transformer)")
    p_gen.add_argument("--model", default=None, help="Checkpoint .pt (motor transformer)")
    p_gen.add_argument("--corpus", default=None,
                        help="Fichero o carpeta .abc (motor markov)")
    p_gen.add_argument("--n-clusters", type=int, default=6,
                        help="Numero de clusters emocionales (motor markov)")
    p_gen.add_argument("--profile", choices=list(EMOTIONAL_PROFILES.keys()), default="neutral",
                        help="Perfil emocional manual (se suma al embedding del texto)")
    p_gen.add_argument("--notes-per-line", type=int, default=0,
                        help="Numero fijo de notas por verso (0 = estimar segun longitud)")
    p_gen.add_argument("--key", default="C", help="Tonalidad destino (ej: C, G, Eb)")
    p_gen.add_argument("--mode", default="major",
                        choices=list(SCALE_INTERVALS.keys()),
                        help="Modo de escala para la mascara tonal (default: major). "
                             "Los tokens fuera de esta escala se suprimen durante el muestreo.")
    p_gen.add_argument("--interval-bias-strength", type=float, default=0.4,
                        metavar="S",
                        help="Intensidad del sesgo de intervalo por perfil emocional "
                             "[0.0 = desactivado, 1.0 = fuerte] (default: 0.4)")
    p_gen.add_argument("--meter", default="4/4")
    p_gen.add_argument("--tempo", type=int, default=100)
    p_gen.add_argument("--unit-beats", type=float, default=0.5,
                        help="Duracion en pulsos (negras) de una unidad L: del modelo")
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=0, help="Solo motor transformer")
    p_gen.add_argument("--seed", type=int, default=42, help="Solo motor markov")
    p_gen.add_argument("--cpu", action="store_true",
                        help="Forzar CPU para el modelo de embeddings y el transformer")
    p_gen.set_defaults(func=cmd_generate)

    p_insp = sub.add_parser("inspect", help="Diagnostico de un modelo y/o corpus")
    p_insp.add_argument("--model", default=None, help="Checkpoint .pt a inspeccionar")
    p_insp.add_argument("--corpus", default=None,
                         help="Fichero o carpeta .abc a inspeccionar")
    p_insp.set_defaults(func=cmd_inspect)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
