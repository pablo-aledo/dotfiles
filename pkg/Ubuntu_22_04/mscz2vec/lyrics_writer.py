#!/usr/bin/env python3
"""
================================================================================
                       LYRICS WRITER  v1.0
       Generacion automatica de lyrics para melodias ABC mediante un
               modelo de lenguaje (Anthropic API u OpenAI API)

  Recibe un fichero ABC SIN lineas w: y una TEMATICA general de la cancion,
  y produce un nuevo fichero ABC con lineas w: anadidas, generadas por un
  LLM para que:
    - El numero de silabas de cada linea w: coincida con el numero de notas
      del compas correspondiente (una silaba por nota, igual que en el
      formato esperado por lyrics_melody.py).
    - El contenido tematico sea coherente con la TEMATICA indicada por el
      usuario.
    - Cada verso mantenga coherencia narrativa con los versos anteriores
      (se pasa el historial completo al LLM en cada llamada).

  COMANDOS:
    write   - Genera lyrics para todos los compases de un fichero ABC

  FLUJO
  -----
  1. Parsea el ABC y extrae, para cada cancion (bloque X:) y cada compas
     (separado por '|'), el numero de notas (= numero de silabas objetivo).
  2. Para cada compas, en orden, llama a la API de Anthropic pidiendo un
     fragmento de letra en espanol con EXACTAMENTE ese numero de silabas,
     coherente con la tematica y con los versos generados previamente.
  3. Cuenta las silabas del fragmento devuelto (heuristica para espanol).
     Si no coincide, reintenta (hasta --max-retries) pidiendo al modelo que
     ajuste la longitud, indicandole el numero de silabas obtenido.
  4. Convierte el fragmento aceptado a formato w: (silabas separadas por
     '-' dentro de cada palabra, palabras separadas por espacio, una
     silaba por nota).
  5. Inserta las lineas w: en el ABC, una por linea de melodia (los
     fragmentos de varios compases de una misma linea se concatenan con
     '|' en la linea w:, igual que puede escribirse en ABC estandar).

  EJEMPLOS DE USO
  ----------------
  # Generar lyrics para un ABC sin letra, tematica libre (Anthropic, default)
  python3 lyrics_writer.py write --abc-in melodia.abc --tema "amor de verano" \
      --abc-out melodia_con_letra.abc

  # Usando OpenAI en vez de Anthropic
  python3 lyrics_writer.py write --abc-in melodia.abc --tema "amor de verano" \
      --abc-out melodia_con_letra.abc --provider openai

  # Especificando modelo y maximo de reintentos por compas
  python3 lyrics_writer.py write --abc-in melodia.abc \
      --tema "soledad en la ciudad" --abc-out salida.abc \
      --provider anthropic --model claude-sonnet-4-6 --max-retries 4

  # Modo verboso: muestra cada compas, intento y resultado
  python3 lyrics_writer.py write --abc-in melodia.abc --tema "tango triste" \
      --abc-out salida.abc --verbose

  DEPENDENCIAS
    --provider anthropic (default): anthropic -> pip install anthropic
        Requiere variable de entorno ANTHROPIC_API_KEY (o --api-key).
    --provider openai: openai -> pip install openai
        Requiere variable de entorno OPENAI_API_KEY (o --api-key).
    Solo es necesario instalar el SDK del proveedor que se vaya a usar.

  NOTAS DE DISENO
  ----------------
  - El conteo de silabas es una heuristica para espanol (vocales fuertes,
    diptongos, triptongos, 'h' muda, agrupaciones consonanticas). No es
    perfecto para casos ambiguos (hiatos vs diptongos segun acentuacion),
    pero es suficientemente fiable para alinear letra y melodia de forma
    automatica con reintentos.
  - El LLM recibe en cada llamada: la tematica, el numero de silabas
    objetivo, el contexto musical del compas (tonalidad, posicion en la
    cancion, si es inicio/fin de frase), y el texto generado hasta el
    momento (para coherencia narrativa).
  - Si tras --max-retries intentos no se logra el numero exacto de
    silabas, se acepta el ultimo intento y se ajusta automaticamente
    (recortando o repitiendo la ultima silaba) para no romper la
    alineacion con la melodia, avisando por stderr.
================================================================================
"""

import argparse
import json
import os
import re
import sys
import time

# ==============================================================================
#  CONSTANTES Y UTILIDADES ABC (compartidas con lyrics_melody.py)
# ==============================================================================

NOTE_RE = re.compile(
    r"(?P<acc>\^\^|\^|=|__|_)?"
    r"(?P<note>[A-Ga-g])"
    r"(?P<oct>[,']*)"
    r"(?P<dur>\d+/\d+|\d+|/\d+|/+)?"
)
REST_RE = re.compile(r"(?P<rest>[zZxX])(?P<dur>\d+/\d+|\d+|/\d+|/+)?")

DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
DEFAULT_MODEL_OPENAI = "gpt-5.1"


# ==============================================================================
#  CONTEO DE SILABAS EN ESPANOL (heuristica)
# ==============================================================================

VOWELS_STRONG = set("aeoAEO")
VOWELS_WEAK_UNACC = set("iuIU")
VOWELS_WEAK_ACC = set("íúÍÚ")        # tilde en i/u -> rompe diptongo (hiato)
VOWELS_STRONG_ACC = set("áéóÁÉÓ")    # tilde en a/e/o -> solo marca acento tonico
VOWELS_ALL = (VOWELS_STRONG | VOWELS_WEAK_UNACC | VOWELS_WEAK_ACC
               | VOWELS_STRONG_ACC | set("üÜ"))


def _is_vowel(c):
    return c in VOWELS_ALL


def _is_strong(c):
    """a/e/o, con o sin tilde (la tilde en a/e/o no rompe diptongo)."""
    return c in VOWELS_STRONG or c in VOWELS_STRONG_ACC


def _is_tonic_weak(c):
    """i/u con tilde escrita -> rompe diptongo (hiato), p.ej. 'ri-o' en 'rio'."""
    return c in VOWELS_WEAK_ACC


def count_syllables_word(word):
    """
    Cuenta silabas de una palabra en espanol mediante una heuristica:
    agrupa vocales consecutivas en nucleos silabicos, tratando como
    diptongo/triptongo (UNA silaba) las combinaciones fuerte+debil,
    debil+fuerte o debil+debil sin tilde en la vocal debil, y como hiato
    (silabas separadas) dos vocales fuertes consecutivas o cuando la
    vocal debil lleva tilde (rompe diptongo).
    """
    word = word.strip()
    if not word:
        return 0

    # eliminar caracteres no alfabeticos (puntuacion) para el conteo
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return 0

    n = len(letters)
    i = 0
    syllables = 0
    while i < n:
        c = letters[i]
        if _is_vowel(c):
            # inicio de un nucleo vocalico; consumir vocales consecutivas
            group = [c]
            j = i + 1
            while j < n and _is_vowel(letters[j]):
                group.append(letters[j])
                j += 1

            # dividir el grupo de vocales en nucleos silabicos
            syllables += _split_vowel_group(group)
            i = j
        else:
            i += 1

    return max(1, syllables)


def _split_vowel_group(group):
    """
    Dada una lista de vocales consecutivas, devuelve cuantas silabas
    forman, aplicando reglas simplificadas de diptongo/triptongo/hiato:
      - Si cualquiera de las dos vocales es una i/u con tilde escrita
        (rompe diptongo), se cuenta como hiato (silabas separadas).
      - Si ambas vocales son "fuertes" (a/e/o, con o sin tilde), se
        cuenta como hiato.
      - En cualquier otro caso (fuerte+debil, debil+fuerte, debil+debil
        sin tilde que rompa), se cuenta como diptongo (misma silaba).
    """
    if len(group) == 1:
        return 1

    count = 1
    for k in range(len(group) - 1):
        a, b = group[k], group[k + 1]
        if _is_tonic_weak(a) or _is_tonic_weak(b):
            count += 1
        elif _is_strong(a) and _is_strong(b):
            count += 1
    return count


def count_syllables_text(text):
    """
    Cuenta el total de silabas de un texto (suma de las silabas de cada
    palabra), aplicando ademas SINALEFA: si una palabra termina en vocal
    y la siguiente empieza por vocal, se funden en una sola silaba (resta
    1 al total), tal y como ocurre habitualmente al cantar en espanol.
    """
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return 0

    total = 0
    prev_ends_vowel = False
    for w in words:
        letters = [c for c in w if c.isalpha()]
        if not letters:
            continue
        word_syll = count_syllables_word(w)
        starts_vowel = _is_vowel(letters[0])
        if prev_ends_vowel and starts_vowel and word_syll > 0:
            total -= 1  # sinalefa con la palabra anterior
        total += word_syll
        prev_ends_vowel = _is_vowel(letters[-1])

    return max(0, total)


def syllabify_text_to_w_format(text):
    """
    Convierte un fragmento de texto en formato w: de ABC: cada palabra se
    divide en sus silabas (separadas por '-'), y las palabras se separan
    por espacios. No aplica sinalefa en el formato w: (cada nota recibe su
    propia silaba/marca, siguiendo la convencion ABC habitual).
    """
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    out_words = []
    for w in words:
        syllables = _word_to_syllable_list(w)
        out_words.append("-".join(syllables) if len(syllables) > 1 else syllables[0])
    return " ".join(out_words)


def _word_to_syllable_list(word):
    """
    Divide una palabra en su lista de silabas (aproximacion ortografica
    simple): agrupa consonantes con la vocal/nucleo que les sigue, y separa
    nucleos vocalicos segun las mismas reglas de diptongo/hiato usadas en
    count_syllables_word.
    """
    n = len(word)
    if n == 0:
        return [""]

    # localizar nucleos vocalicos (posibles diptongos/triptongos) y dividirlos
    # en silabas segun _split_vowel_group, generando una lista de "bloques
    # vocalicos" donde cada bloque es una silaba (lista de indices)
    vowel_blocks = []  # lista de listas de indices de caracteres vocalicos
    i = 0
    while i < n:
        if _is_vowel(word[i]):
            group_start = i
            group = [word[i]]
            j = i + 1
            while j < n and _is_vowel(word[j]):
                group.append(word[j])
                j += 1
            n_syll = _split_vowel_group(group)
            if n_syll == 1:
                vowel_blocks.append(list(range(group_start, j)))
            else:
                # dividir el grupo en n_syll bloques lo mas equilibrados
                # posible (en la practica, para 2 vocales -> 1+1; para 3 con
                # algun hiato, se reparte de forma simple)
                size = len(group)
                base = size // n_syll
                extra = size % n_syll
                idx = group_start
                for s in range(n_syll):
                    take = base + (1 if s < extra else 0)
                    take = max(1, take)
                    vowel_blocks.append(list(range(idx, min(idx + take, group_start + size))))
                    idx += take
            i = j
        else:
            i += 1

    if not vowel_blocks:
        # palabra sin vocales (raro): devolver tal cual
        return [word]

    # grupos consonanticos que tipicamente NO se separan (forman el ataque
    # de la silaba siguiente): oclusiva/labio-dental + r/l
    UNSPLITTABLE_CLUSTERS = {
        "pr", "pl", "br", "bl", "tr", "dr", "cr", "cl", "gr", "gl", "fr", "fl",
    }

    # asignar cada grupo de consonantes entre dos nucleos: si son 0 o 1
    # consonantes, pasan enteras a la silaba siguiente; si son 2+, se
    # comprueban las DOS ULTIMAS: si forman un grupo "inseparable" (tr, bl,
    # etc.), ambas pasan a la silaba siguiente y el resto se queda en la
    # anterior; si no, solo la ultima consonante pasa a la silaba siguiente.
    syllable_ranges = []  # lista de [start_idx, end_idx] inclusive
    prev_end = -1
    for bi, block in enumerate(vowel_blocks):
        start_vowel = block[0]
        end_vowel = block[-1]

        cons_start = prev_end + 1
        cons_count = start_vowel - cons_start

        if bi == 0:
            syll_start = 0
        else:
            if cons_count <= 1:
                syll_start = cons_start
            else:
                last_two = word[start_vowel - 2:start_vowel].lower()
                if last_two in UNSPLITTABLE_CLUSTERS:
                    syll_start = start_vowel - 2
                else:
                    syll_start = start_vowel - 1

        syllable_ranges.append([syll_start, end_vowel])
        if bi > 0:
            syllable_ranges[bi - 1][1] = syll_start - 1

        prev_end = end_vowel

    # extender la ultima silaba hasta el final de la palabra
    syllable_ranges[-1][1] = n - 1
    # asegurar la primera silaba empieza en 0
    syllable_ranges[0][0] = 0

    syllables = []
    for start, end in syllable_ranges:
        if end < start:
            continue
        syllables.append(word[start:end + 1])

    return syllables if syllables else [word]


# ==============================================================================
#  PARSER ABC: extraccion de compases (notas por compas, sin lineas w:)
# ==============================================================================

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


def parse_default_len(l_field):
    m = re.match(r"\s*(\d+)/(\d+)", l_field)
    if m:
        return float(m.group(1)) / float(m.group(2))
    return 1.0 / 8.0


def count_notes_in_bar(bar_text):
    """Cuenta el numero de notas y silencios (eventos) en un fragmento de
    compas ABC (texto entre barras '|'), ignorando adornos/anotaciones."""
    cleaned = re.sub(r'!.*?!', '', bar_text)
    cleaned = re.sub(r'".*?"', '', cleaned)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)

    count = 0
    pos = 0
    n = len(cleaned)
    while pos < n:
        ch = cleaned[pos]
        if ch in " \t(){}><.~":
            pos += 1
            continue

        rm = REST_RE.match(cleaned, pos)
        if rm and rm.group("rest"):
            count += 1
            pos = rm.end()
            continue

        nm = NOTE_RE.match(cleaned, pos)
        if nm and nm.group("note"):
            count += 1
            pos = nm.end()
            continue

        pos += 1

    return count


class AbcLine:
    """Representa una linea de melodia del ABC junto con sus compases."""

    def __init__(self, raw_line, bar_texts, notes_per_bar):
        self.raw_line = raw_line
        self.bar_texts = bar_texts          # texto de cada compas (sin '|')
        self.notes_per_bar = notes_per_bar  # numero de notas/eventos por compas


class AbcSong:
    """Representa una cancion (bloque X:) ya parseada para escritura de
    lyrics: cabecera tal cual (lista de lineas) + lineas de melodia con sus
    compases."""

    def __init__(self, header_lines, melody_lines):
        self.header_lines = header_lines  # lineas X:, T:, M:, L:, K:, etc.
        self.melody_lines = melody_lines  # lista de AbcLine


def parse_abc_for_writing(path):
    """
    Parsea un fichero .abc identificando, para cada bloque X:, las lineas
    de cabecera y las lineas de melodia, dividiendo cada linea de melodia
    en sus compases y contando las notas de cada uno. Ignora cualquier
    linea w: existente (se sobrescribiran).

    Devuelve una lista de AbcSong.
    """
    songs = []
    header_lines = []
    melody_lines = []
    in_song = False

    def flush_song():
        if header_lines or melody_lines:
            songs.append(AbcSong(list(header_lines), list(melody_lines)))

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [l.rstrip("\n") for l in f]

    for line in raw_lines:
        if re.match(r"^X:", line):
            flush_song()
            header_lines = [line]
            melody_lines = []
            in_song = True
            continue

        if not in_song:
            # lineas antes del primer X: (poco habitual) -> ignorar
            continue

        if re.match(r"^[A-Zx]:", line) and not line.strip().startswith("w:"):
            header_lines.append(line)
            continue

        if line.strip().startswith("w:"):
            # ignorar lineas w: existentes, se regeneraran
            continue

        if line.strip() == "" or line.strip().startswith("%"):
            continue

        # linea de melodia
        unit_len = 1.0 / 8.0
        for h in header_lines:
            if re.match(r"^L:", h):
                unit_len = parse_default_len(h[2:])

        bar_texts = [b for b in line.split("|")]
        # eliminar fragmentos vacios al final (p.ej. linea termina en '|')
        while bar_texts and bar_texts[-1].strip() == "":
            bar_texts.pop()
        # eliminar fragmento vacio inicial si la linea empieza por '|'
        if bar_texts and bar_texts[0].strip() == "":
            bar_texts.pop(0)

        notes_per_bar = [count_notes_in_bar(b) for b in bar_texts]
        # descartar compases sin notas (p.ej. solo anotaciones)
        kept_bars = [(b, n) for b, n in zip(bar_texts, notes_per_bar) if n > 0]
        if not kept_bars:
            continue
        bar_texts2, notes_per_bar2 = zip(*kept_bars)

        melody_lines.append(AbcLine(line, list(bar_texts2), list(notes_per_bar2)))

    flush_song()
    return songs


# ==============================================================================
#  INTERACCION CON EL LLM (Anthropic API y OpenAI API)
# ==============================================================================
#
# Se soportan dos proveedores intercambiables, seleccionados con
# --provider {anthropic, openai}. Ambos exponen la misma interfaz interna:
#   _llm_call(client_info, model, system_prompt, user_prompt, max_tokens)
#       -> str (texto de respuesta, ya extraido)
#
# client_info es un dict {"provider": "anthropic"|"openai", "client": <SDK client>}

_CLIENT_CACHE = {}


def _get_client(provider, api_key=None):
    """Crea (una vez, cacheado) el cliente del SDK correspondiente al
    proveedor indicado. Devuelve {"provider": provider, "client": client}."""
    if provider in _CLIENT_CACHE:
        return _CLIENT_CACHE[provider]

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "El paquete 'anthropic' no esta disponible. Instala con:\n"
                "  pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "No se ha encontrado ANTHROPIC_API_KEY. Exporta la variable "
                "de entorno o usa --api-key."
            )
        client = anthropic.Anthropic(api_key=key)

    elif provider == "openai":
        try:
            import openai
        except ImportError:
            raise ImportError(
                "El paquete 'openai' no esta disponible. Instala con:\n"
                "  pip install openai"
            )
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "No se ha encontrado OPENAI_API_KEY. Exporta la variable de "
                "entorno o usa --api-key."
            )
        client = openai.OpenAI(api_key=key)

    else:
        raise ValueError(f"Proveedor desconocido: {provider!r} "
                          f"(usa 'anthropic' o 'openai')")

    info = {"provider": provider, "client": client}
    _CLIENT_CACHE[provider] = info
    return info


def _llm_call(client_info, model, system_prompt, user_prompt, max_tokens=200):
    """
    Realiza UNA llamada al LLM (Anthropic o OpenAI segun client_info) y
    devuelve el texto de respuesta como string, ya extraido del formato
    propio de cada SDK.
    """
    provider = client_info["provider"]
    client = client_info["client"]

    if provider == "anthropic":
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "".join(text_parts).strip().strip('"').strip()

    elif provider == "openai":
        response = client.chat.completions.create(
            model=model,
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content or ""
        return text.strip().strip('"').strip()

    else:
        raise ValueError(f"Proveedor desconocido: {provider!r}")


def _build_position_description(song_idx, n_songs, line_idx, n_lines,
                                  bar_idx, n_bars):
    """Describe la posicion del compas actual dentro de la cancion, para
    dar contexto narrativo al LLM (inicio, desarrollo, cierre)."""
    parts = []
    if n_songs > 1:
        parts.append(f"cancion {song_idx + 1} de {n_songs}")
    if line_idx == 0 and bar_idx == 0:
        parts.append("inicio de la cancion")
    elif line_idx == n_lines - 1 and bar_idx == n_bars - 1:
        parts.append("final de la cancion")
    elif bar_idx == 0:
        parts.append(f"inicio de la linea {line_idx + 1} de {n_lines}")
    elif bar_idx == n_bars - 1:
        parts.append(f"final de la linea {line_idx + 1} de {n_lines}")
    else:
        parts.append(f"linea {line_idx + 1} de {n_lines}, compas {bar_idx + 1} de {n_bars}")
    return ", ".join(parts)


def _llm_request_fragment(client_info, model, tema, target_syllables, position_desc,
                           key_signature, history_lines, previous_attempt=None,
                           previous_count=None, max_tokens=200):
    """
    Realiza UNA llamada al LLM (Anthropic u OpenAI, segun client_info)
    solicitando un fragmento de letra en espanol con target_syllables
    silabas, coherente con la tematica y el historial. Devuelve el texto
    crudo devuelto por el modelo (una linea).
    """
    history_text = "\n".join(f"  {i+1}. {h}" for i, h in enumerate(history_lines)) \
        if history_lines else "  (ninguno todavia; este es el primer verso)"

    system_prompt = (
        "Eres un letrista experto en canciones en español. Tu tarea es "
        "escribir UN UNICO fragmento de letra (parte de un verso) que "
        "encaje en una melodia, dado un numero EXACTO de silabas objetivo. "
        "Respondes UNICAMENTE con el fragmento de texto, sin numeracion, "
        "sin comillas, sin explicaciones, sin saltos de linea adicionales."
    )

    user_prompt = (
        f"Tematica general de la cancion: {tema}\n"
        f"Tonalidad musical: {key_signature}\n"
        f"Posicion de este fragmento en la cancion: {position_desc}\n\n"
        f"Versos/fragmentos ya escritos hasta ahora (en orden):\n{history_text}\n\n"
        f"Escribe el SIGUIENTE fragmento de letra, que debe tener EXACTAMENTE "
        f"{target_syllables} silabas (contando sinalefas como es habitual al "
        f"cantar en español), debe continuar el sentido del verso anterior si "
        f"corresponde (puede ser la continuacion de la misma frase o el inicio "
        f"de una nueva), y debe ser coherente con la tematica indicada.\n"
    )

    if previous_attempt is not None:
        user_prompt += (
            f"\nUn intento anterior fue: \"{previous_attempt}\" "
            f"({previous_count} silabas, objetivo {target_syllables}). "
            f"Ajusta la redaccion para conseguir EXACTAMENTE {target_syllables} "
            f"silabas, manteniendo el sentido."
        )

    return _llm_call(client_info, model, system_prompt, user_prompt, max_tokens=max_tokens)


def generate_fragment(client_info, model, tema, target_syllables, position_desc,
                       key_signature, history_lines, max_retries=3, verbose=False):
    """
    Genera un fragmento con el numero de silabas objetivo, reintentando
    hasta max_retries veces si el conteo no coincide. Si tras los
    reintentos sigue sin coincidir, ajusta el texto (recortando o
    repitiendo palabras) para forzar el conteo, avisando por stderr.

    Devuelve (texto_final, n_silabas_obtenidas, n_intentos).
    """
    if target_syllables <= 0:
        return "", 0, 0

    previous_attempt = None
    previous_count = None
    best_text = ""
    best_diff = None

    for attempt in range(1, max_retries + 1):
        text = _llm_request_fragment(
            client_info, model, tema, target_syllables, position_desc, key_signature,
            history_lines, previous_attempt=previous_attempt,
            previous_count=previous_count,
        )
        text = re.sub(r"\s+", " ", text).strip()
        count = count_syllables_text(text)

        if verbose:
            print(f"    intento {attempt}: \"{text}\" ({count} silabas, "
                  f"objetivo {target_syllables})", file=sys.stderr)

        diff = abs(count - target_syllables)
        if best_diff is None or diff < best_diff:
            best_text, best_diff = text, diff

        if count == target_syllables:
            return text, count, attempt

        previous_attempt = text
        previous_count = count

    # no se alcanzo el objetivo exacto: ajustar el mejor intento
    adjusted, final_count = _force_syllable_count(best_text, target_syllables)
    if verbose:
        print(f"    [aviso] tras {max_retries} intentos, ajustando "
              f"\"{best_text}\" ({best_diff and (target_syllables - best_diff)} "
              f"-> {final_count} silabas)", file=sys.stderr)

    return adjusted, final_count, max_retries


def _force_syllable_count(text, target):
    """
    Ajusta un texto para que tenga exactamente `target` silabas:
      - Si tiene mas silabas, elimina palabras finales hasta ajustarse
        (o se queda corto si no es posible exactamente).
      - Si tiene menos, repite la ultima palabra (o anade una silaba de
        relleno "ah") hasta alcanzar el objetivo.
    Es un ultimo recurso para no romper la alineacion con la melodia.
    """
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        words = ["la"] * max(1, target)

    count = count_syllables_text(" ".join(words))

    # eliminar palabras finales mientras sobren silabas y queden >=1 palabras
    while count > target and len(words) > 1:
        words = words[:-1]
        count = count_syllables_text(" ".join(words))

    # anadir relleno monosilabico mientras falten silabas
    filler_cycle = ["ah", "la", "oh", "ya"]
    fi = 0
    while count < target:
        words.append(filler_cycle[fi % len(filler_cycle)])
        fi += 1
        count = count_syllables_text(" ".join(words))

    return " ".join(words), count


# ==============================================================================
#  MODO WRITE
# ==============================================================================

def cmd_write(args):
    model = args.model or (
        DEFAULT_MODEL_OPENAI if args.provider == "openai" else DEFAULT_MODEL_ANTHROPIC
    )
    client_info = _get_client(args.provider, args.api_key)

    songs = parse_abc_for_writing(args.abc_in)
    if not songs:
        print(f"[write] No se encontraron canciones (bloques X:) en {args.abc_in}",
              file=sys.stderr)
        sys.exit(1)

    n_songs = len(songs)
    total_bars = sum(len(ml.bar_texts) for song in songs for ml in song.melody_lines)
    print(f"[write] Canciones encontradas: {n_songs}")
    print(f"[write] Compases totales a musicalizar: {total_bars}")
    print(f"[write] Tematica: {args.tema}")
    print(f"[write] Proveedor: {args.provider}  Modelo: {model}")

    history_lines = []
    bar_counter = 0

    for song_idx, song in enumerate(songs):
        key_signature = "C"
        for h in song.header_lines:
            if re.match(r"^K:", h):
                key_signature = h[2:].strip()

        n_lines = len(song.melody_lines)

        for line_idx, ml in enumerate(song.melody_lines):
            n_bars = len(ml.bar_texts)
            fragments = []

            for bar_idx, n_notes in enumerate(ml.notes_per_bar):
                bar_counter += 1
                position_desc = _build_position_description(
                    song_idx, n_songs, line_idx, n_lines, bar_idx, n_bars
                )

                print(f"[write] Compas {bar_counter}/{total_bars} "
                      f"({n_notes} silabas, {position_desc})...")

                fragment, count, attempts = generate_fragment(
                    client_info, model, args.tema, n_notes, position_desc,
                    key_signature, history_lines, max_retries=args.max_retries,
                    verbose=args.verbose,
                )

                if args.verbose:
                    print(f"    -> \"{fragment}\" ({count}/{n_notes} silabas, "
                          f"{attempts} intento(s))")

                fragments.append(fragment)
                history_lines.append(fragment)

                if args.delay > 0:
                    time.sleep(args.delay)

            ml.fragments = fragments  # guardar para escritura posterior

    write_abc_with_lyrics(songs, args.abc_out)
    print(f"[write] Fichero ABC con lyrics escrito en {args.abc_out}")


def write_abc_with_lyrics(songs, out_path):
    """Escribe las canciones parseadas a un fichero ABC, insertando una
    linea w: (con los compases separados por '|', igual que la melodia)
    despues de cada linea de melodia."""
    out_lines = []
    for song in songs:
        out_lines.extend(song.header_lines)
        for ml in song.melody_lines:
            out_lines.append(ml.raw_line)
            w_bars = [syllabify_text_to_w_format(f) for f in ml.fragments]
            out_lines.append("w: " + " | ".join(w_bars))
        out_lines.append("")  # separador entre canciones

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines).rstrip("\n") + "\n")


# ==============================================================================
#  CLI
# ==============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generacion automatica de lyrics para melodias ABC "
                     "mediante un LLM (Anthropic API u OpenAI API)",
    )
    parser.add_argument("--version", action="version", version="lyrics_writer 1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    p_write = sub.add_parser("write", help="Generar lyrics para un fichero ABC")
    p_write.add_argument("--abc-in", required=True,
                          help="Fichero ABC de entrada (sin lineas w:, o con "
                               "lineas w: que se sobrescribiran)")
    p_write.add_argument("--abc-out", required=True,
                          help="Fichero ABC de salida con lineas w: anadidas")
    p_write.add_argument("--tema", required=True,
                          help="Tematica general de la cancion (en lenguaje natural)")
    p_write.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                          help="Proveedor del LLM a usar (default: anthropic)")
    p_write.add_argument("--model", default=None,
                          help="Modelo a usar. Por defecto: "
                               f"'{DEFAULT_MODEL_ANTHROPIC}' (anthropic) o "
                               f"'{DEFAULT_MODEL_OPENAI}' (openai)")
    p_write.add_argument("--api-key", default=None,
                          help="API key del proveedor (por defecto, variable de "
                               "entorno ANTHROPIC_API_KEY u OPENAI_API_KEY segun "
                               "--provider)")
    p_write.add_argument("--max-retries", type=int, default=3,
                          help="Maximo de reintentos por compas si el numero de "
                               "silabas no coincide (default: 3)")
    p_write.add_argument("--delay", type=float, default=0.0,
                          help="Pausa en segundos entre llamadas a la API "
                               "(util para limites de tasa)")
    p_write.add_argument("--verbose", action="store_true",
                          help="Mostrar cada intento de generacion por compas")
    p_write.set_defaults(func=cmd_write)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
