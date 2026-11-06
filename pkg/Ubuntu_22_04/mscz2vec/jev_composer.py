#!/usr/bin/env python3
r"""
jev_composer.py — Jev Composer: composición guiada por Jev (TypeSafe) a partir
de una descripción en texto, con semillas de chord_table.py y motif_table.py
(versión 4.0: subcomando train; detección de acordes sin fusionar
los de 1 pulso o menos; cache con todos los datos de cada consulta).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Recibe la descripción en lenguaje natural de una obra y la construye en tres
fases, todas decididas por un clasificador zero-shot (Jev, vía OpenRouter, el
mismo que usa el prototipo jev_classifier.zsh):

  1. TONALIDAD   Jev elige el modo (mayor, menor, dórico, frigio, lidio,
                 mixolidio, menor armónico, frigio dominante) y luego la
                 tónica (12 notas).
  2. SEMILLAS    Se invocan chord_table.py y motif_table.py como comandos para
                 obtener su catálogo filtrado por el modo elegido; Jev elige
                 una progresión de acordes y un motivo melódico semilla, y los
                 mismos programas los resuelven en la tónica elegida.
  3. VENTANA     Ventana deslizante de profundidad configurable: en cada paso
                 el estado enviado a Jev contiene la descripción, las semillas
                 y los últimos N acordes con su melodía. Jev da una
                 distribución de probabilidad y se SAMPLEA (con temperatura):
                   · primero el acorde siguiente (acordes diatónicos del modo
                     más el dominante V7), y después
                   · nota a nota la melodía hasta llenar ese acorde.
                 Se repite hasta completar los compases pedidos.

  El ritmo armónico (duración de cada acorde) y el ritmo melódico (duración de
  cada nota o silencio) se toman de las semillas, en ciclo: Jev decide QUÉ
  suena, la semilla decide CUÁNDO. Cada consulta pide una sola pregunta
  (acorde o nota), igual que el prototipo.

  MODO ACORDES FIJOS (--chords-midi FILE): en vez de elegir y generar acordes,
  se leen de un MIDI (con el detector de acordes de chord_table.py). Jev solo
  genera la melodía sobre esos acordes: ve el acorde actual, los anteriores
  (ventana) y los dos siguientes. El modo lo sigue eligiendo Jev, ya viendo los
  acordes (o --mode); la tónica se infiere del MIDI (o --key). Solo se elige
  el motivo semilla.

  INDICACIONES POR COMPÁS (--notes / --notes-file): información extra en
  texto libre para compases concretos. Cuando la generación está en esos
  compases, el texto se añade al estado y a la instrucción de las consultas a
  Jev (acorde o nota), tanto en el modo normal como con --chords-midi. En los
  demás compases las consultas no cambian (y siguen sirviéndose de la cache).
  Formato: una indicación por línea (en fichero) o separadas por ';' (en la
  línea de comandos); los compases se numeran desde 1, como en la salida:
      compás 29: intensidad creciente, densidad creciente
      compases 30-32: mood calma
      c.31: explosión súbita
      28, 33: respirar
  El prefijo (compás, compases, c., m., bar, bars) es opcional. Varias
  indicaciones sobre el mismo compás se acumulan. En un fichero, las líneas
  vacías y las que empiezan por # se ignoran.

  MODO TRAIN (train DIR): analiza los MIDIs de un directorio y reconstruye en
  la cache las consultas que se le habrían hecho a Jev para componerlos, con
  la probabilidad 1.0 en el valor real leído del MIDI y 0.0 en el resto. No
  llama a la API. El formato de la cache es idéntico al de componer: las
  entradas se reutilizan con --cache / --cache-only siempre que la consulta
  (modelo, estado, pregunta y opciones) coincida exactamente.
    · Cada X.mid puede ir acompañado de X.txt (mismo nombre), con la
      descripción de la obra y las indicaciones por compás:
          descripción: Nana melancólica para una noche de lluvia
          tonalidad: Am            (opcional; si no, se infiere del MIDI)
          modo: dorian             (opcional; si no, se infiere del MIDI)
          compás 29: intensidad creciente, densidad creciente
          compases 30-32: mood calma
      Sin 'descripción:' se usa --desc, o el nombre del fichero.
    · Se separan melodía y armonía: la melodía es la pista (o canal) más
      aguda y monofónica; si no hay ninguna (p. ej. piano en una pista), se
      toma la nota más aguda de cada ataque. Lo demás es la armonía, de la
      que chord_table.py detecta los acordes.
    · Se recorren acordes y melodía con la misma ventana deslizante y las
      mismas preguntas que componer: modo, tónica, semilla de acordes y de
      motivo (la entrada de las tablas que mejor coincide con el arranque
      del MIDI), y después cada acorde y cada nota. El estilo 'fixed'
      genera además las entradas del modo --chords-midi.
    · No se puede registrar una respuesta que no estaba entre las opciones
      (acorde no diatónico, nota cromática o a más de --leap semitonos de la
      anterior): esa consulta se omite y se cuenta en el resumen.
    · La melodía se cuantiza a --quant pulsos (default 0.25) y las notas no
      cruzan cambios de acorde, igual que al componer.

  Todas las consultas pueden guardarse en una cache local (--cache): una
  consulta idéntica (mismo modelo, estado, pregunta y opciones) no vuelve a
  gastar créditos. Con --rng fijo, una composición repetida se resuelve
  íntegramente desde la cache.

INTEGRACIÓN CON EL ECOSISTEMA:
  chord_table.py   → se invoca como comando (--mode … --export-json; --id … )
  motif_table.py   → se invoca como comando (--mode … --export-json; --id … )
  melody_adapter.py→ <base>.melody.txt es una línea nota:beats compatible
  voice_leader.py  → <base>.chords.txt es un string --chords compatible

────────────────────────────────────────────────────────────────────────────
Formato de la cache (JSONL, una consulta por línea)
────────────────────────────────────────────────────────────────────────────
  Cada registro contiene todo lo necesario para reconstruir la pregunta y su
  distribución de salida (p. ej. para entrenar un modelo local):
    key           hash sha256 de (model, state, instructions, criteria)
    kind          modo | tonica | semilla_acordes | semilla_motivo | acorde | nota
    model         modelo de Jev que respondió
    source        api (respuesta de Jev) | train (valor real de un MIDI, 1.0/0.0)
    instructions  pregunta completa
    state         documento completo que vio Jev (descripción, semillas, ventana)
    criteria      {opción: descripción} en el orden enviado (las opciones)
    extra         (opcional) indicaciones por compás aplicadas, legible
    answer        {choice, probabilities}: probabilidad de cada opción; las que
                  faltan en 'probabilities' valen 0. Objetivo de entrenamiento:
                  la distribución normalizada sobre las claves de 'criteria'
  La clave se puede recalcular a partir de model, state, instructions y
  criteria. El fichero solo se amplía: si una clave aparece varias veces, vale
  la última línea. Las entradas antiguas (solo hash) se completan al
  reutilizarlas con --cache (no con --cache-only). 'cache' muestra cuántas
  están completas.

────────────────────────────────────────────────────────────────────────────
Subcomandos
────────────────────────────────────────────────────────────────────────────
  tonalidad DESC     Elegir modo y tónica de la obra
  semillas  DESC     Elegir progresión y motivo semilla (adaptados al modo)
  componer  DESC     Pipeline completo: tonalidad + semillas + ventana
  train     DIR      Reconstruir la cache a partir de MIDIs (ver MODO TRAIN)
  cache              Estadísticas de la cache (--clear para vaciarla)

  DESC es la descripción de la obra entre comillas, o bien --file FICHERO.

────────────────────────────────────────────────────────────────────────────
Entradas y salidas
────────────────────────────────────────────────────────────────────────────
  <base>.chords.txt   — acordes en texto ("Am:2 G:2 F:2 E7:2")
  <base>.melody.txt   — melodía en texto ("E4:1 G4:0.5 R:0.5 C5:2")
  <base>.jev.json     — obra completa, semillas, parámetros y registro de
                        consultas (probabilidad de cada elección, si vino de
                        la cache)
  <base>.mid          — MIDI de dos pistas: acordes en bloque y melodía

────────────────────────────────────────────────────────────────────────────
Opciones comunes (todos los subcomandos que consultan a Jev)
────────────────────────────────────────────────────────────────────────────
  --model MODEL       Modelo de Jev (default: typesafe/jev-1.14;
                      typesafe/jev-latest sigue la última versión)
  --url URL           Endpoint (default:
                      https://openrouter.ai/api/alpha/decisions)
  --cache             Guardar y reutilizar las consultas en la cache local
  --cache-only        Solo leer de la cache: nunca llama a la API; falla si
                      falta una consulta (garantiza gasto cero)
  --cache-file FILE   Fichero de cache JSONL (default:
                      ~/.cache/jev_composer/queries.jsonl)
  --file FILE         Leer la descripción de un fichero (en vez de DESC)
  --key KEY           Fijar la tónica (Jev no la elige). 'Am' implica menor
  --mode MODE         Fijar el modo (Jev no lo elige)
  --seed-temp T       Temperatura para elegir modo/tónica/semillas
                      (default: 0 = la más probable)
  --rng N             Semilla del generador aleatorio (reproducibilidad)
  --verbose           Mostrar las probabilidades de cada consulta

Opciones de semillas / componer:
  --octave N          Octava base de la tónica (default: 5, como motif_table)
  --chord-table FILE  Ruta de chord_table.py (default: junto a este script)
  --motif-table FILE  Ruta de motif_table.py (default: junto a este script)
  --export-json FILE  (semillas) Exportar las semillas elegidas a JSON

Opciones de train (además de --model, --cache-file, --key, --mode,
--bass-only, --min-dur, --chord-table, --motif-table, --leap):
  DIR                 Directorio con los .mid/.midi (y los .txt)
  --depth N[,N...]    Profundidades de ventana a reconstruir (default: 4);
                      cada una produce estados distintos
  --octave N          Octava base (default: automática por obra, centrando
                      la melodía; fijarla si luego se compone con --octave)
  --beats N           Pulsos por compás (default: el del MIDI, o 4)
  --quant BEATS       Rejilla de cuantización de la melodía (default: 0.25)
  --melody-track N    Forzar la pista (0-based) que contiene la melodía
  --style S           generate | fixed | both (default: both)
  --desc TEXTO        Descripción para los MIDIs sin 'descripción:' en su .txt
  --overwrite         Reescribir entradas ya existentes (por defecto se
                      respetan, p. ej. las obtenidas de Jev de verdad)

Opciones de componer:
  --bars N            Compases a generar (default: 16; con --chords-midi, los
                      que dure el MIDI; si pides más se recorta a su duración)
  --beats N           Pulsos por compás (default: 4)
  --tempo BPM         Tempo para el MIDI (default: 100)
  --depth N           Profundidad de la ventana: acordes previos (con su
                      melodía) que ve Jev en cada consulta (default: 4)
  --temperature T     Temperatura del sampling de acordes y notas
                      (default: 1.0; 0 = siempre la más probable; <1 más
                      conservador; >1 más aventurero)
  --leap N            Salto melódico máximo en semitonos entre notas
                      consecutivas (default: 9)
  --output OUTPUT     Nombre base para salidas (default: obra)
  --notes TEXTO       Indicaciones por compás (ver formato arriba); se puede
                      repetir. Ej: --notes "29: intensidad creciente; 30: calma"
  --notes-file FILE   Leer indicaciones por compás de un fichero
  --chords-midi FILE  Usar los acordes de este MIDI en vez de generarlos
  --bass-only         (con --chords-midi) detectar acordes solo con las notas
                      graves (≤ C4), como chord_table.py
  --min-dur BEATS     (con --chords-midi) los segmentos de acorde más cortos que
                      esto se funden con el anterior (default: 0.5; subirlo si
                      el MIDI es una interpretación humana con solapes)

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  export OPENROUTER_API_KEY=...

  # Solo la tonalidad
  python jev_composer.py tonalidad "Nana melancólica para una noche de lluvia"

  # Las semillas, con modo fijado y probabilidades visibles
  python jev_composer.py semillas "Marcha épica de fantasía" --mode dorian \
      --verbose --export-json semillas.json

  # Composición completa de 16 compases, con cache
  python jev_composer.py componer "Danza flamenca tensa y oscura" \
      --cache --bars 16 --depth 4 --temperature 0.8 --rng 7

  # Repetirla sin gastar créditos (misma descripción y mismo --rng)
  python jev_composer.py componer "Danza flamenca tensa y oscura" \
      --cache-only --bars 16 --depth 4 --temperature 0.8 --rng 7

  # Descripción larga desde fichero, tónica fijada, 32 compases en 3/4
  python jev_composer.py componer --file encargo.txt --key Am \
      --bars 32 --beats 3 --cache --output nocturno

  # Melodía sobre los acordes de un MIDI propio (no se generan acordes)
  python jev_composer.py componer "Balada íntima y esperanzada" \
      --chords-midi acordes.mid --cache --temperature 0.8 --rng 3

  # Indicaciones por compás (en línea de comandos y/o en fichero)
  python jev_composer.py componer "Obra para cuerdas con clímax" --bars 40 --cache \
      --notes "compás 29: intensidad creciente, densidad creciente" \
      --notes "compás 30: mood calma; compás 31: explosión súbita" \
      --notes-file indicaciones.txt

  # Reconstruir la cache a partir de MIDIs (X.mid + X.txt) en un directorio
  python jev_composer.py train ./corpus --depth 2,4 --octave 5 \
      --cache-file ~/.cache/jev_composer/queries.jsonl

  # Estado de la cache
  python jev_composer.py cache
  python jev_composer.py cache --clear

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  chord_table.py y motif_table.py (se invocan como comandos)
  OPENROUTER_API_KEY en el entorno (salvo con --cache-only)
  mido (opcional, solo para exportación MIDI)
"""

import sys
import os
import json
import math
import time
import random
import hashlib
import argparse
import re
import io
import types
import copy
import bisect
import statistics
import contextlib
import tempfile
import importlib.util
import subprocess
import urllib.request
import urllib.error
from collections import Counter

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES MUSICALES  (idénticas a chord_table.py / motif_table.py)
# ═══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES      = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

NOTE_PC = {n: i for i, n in enumerate(PITCH_NAMES)}
NOTE_PC.update({n: i for i, n in enumerate(PITCH_NAMES_FLAT)})
NOTE_PC.update({'Cb': 11, 'Fb': 4, 'B#': 0, 'E#': 5})

MODE_INTERVALS = {
    'major':              [0, 2, 4, 5, 7, 9, 11],
    'minor':              [0, 2, 3, 5, 7, 8, 10],   # eólico / menor natural
    'dorian':             [0, 2, 3, 5, 7, 9, 10],
    'phrygian':           [0, 1, 3, 5, 7, 8, 10],
    'lydian':             [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':         [0, 2, 4, 5, 7, 9, 10],
    'harmonic_minor':     [0, 2, 3, 5, 7, 8, 11],
    'phrygian_dominant':  [0, 1, 4, 5, 7, 8, 10],
}
MODES = list(MODE_INTERVALS.keys())

# Descripción de cada modo que se ofrece a Jev como criterio de la pregunta
MODE_INFO = {
    'major':             'Major (Ionian): bright, stable, optimistic, resolved',
    'minor':             'Natural minor (Aeolian): sad, serious, introspective',
    'dorian':            'Dorian: minor with a raised 6th; melancholic yet hopeful, '
                         'groovy, folk and jazz flavour',
    'phrygian':          'Phrygian: minor with a flat 2nd; dark, tense, Spanish or exotic',
    'lydian':            'Lydian: major with a raised 4th; dreamy, floating, wonder',
    'mixolydian':        'Mixolydian: major with a flat 7th; relaxed, bluesy, rock '
                         'and folk flavour',
    'harmonic_minor':    'Harmonic minor: minor with a raised 7th; dramatic, classical, '
                         'Eastern-flavoured',
    'phrygian_dominant': 'Phrygian dominant: major third over a flat 2nd; flamenco, '
                         'Middle-Eastern, intense',
}

# Si el catálogo de una herramienta no tiene entradas (o no admite el modo), se
# prueba con estos modos más cercanos y se avisa por stderr.
MODE_FALLBACK = {
    'harmonic_minor':    ['minor'],
    'phrygian_dominant': ['phrygian', 'minor'],
    'phrygian':          ['minor'],
    'dorian':            ['minor'],
    'lydian':            ['major'],
    'mixolydian':        ['major'],
}

# Acordes: calidades con las que chord_table.py sabe trabajar
TRIAD_QUALITY   = {(0, 4, 7): '', (0, 3, 7): 'm', (0, 3, 6): 'dim', (0, 4, 8): 'aug'}
SEVENTH_QUALITY = {(0, 4, 7, 11): 'M7', (0, 4, 7, 10): '7', (0, 3, 7, 10): 'm7',
                   (0, 3, 6, 10): 'hdim7', (0, 3, 6, 9): 'dim7'}
SEVENTH_SUFFIX  = {'M7': 'M7', '7': '7', 'm7': '7', 'hdim7': 'ø7', 'dim7': '°7'}

ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
DEGREE_FUNCTION = ['tonic', 'supertonic (pre-dominant)', 'mediant (tonic substitute)',
                   'subdominant', 'dominant', 'submediant (tonic substitute)',
                   'leading/subtonic (dominant substitute)']
CHORD_ROLE = {0: 'root', 2: '9th', 3: 'minor 3rd', 4: 'major 3rd', 6: 'diminished 5th',
              7: '5th', 8: 'augmented 5th', 9: '6th', 10: 'minor 7th', 11: 'major 7th'}

MAX_OPTIONS = 255      # límite por pregunta de los clientes oficiales de TypeSafe
EPS = 1e-6
DEFAULT_MODEL = 'typesafe/jev-1.14'
DEFAULT_URL   = 'https://openrouter.ai/api/alpha/decisions'
DEFAULT_CACHE = os.path.join('~', '.cache', 'jev_composer', 'queries.jsonl')
HERE = os.path.dirname(os.path.abspath(__file__))

COL = {
    'reset': '\033[0m', 'bold': '\033[1m',
    'cyan':  '\033[36m', 'yellow': '\033[33m',
    'green': '\033[32m', 'red': '\033[31m',
    'gray':  '\033[90m', 'blue': '\033[34m',
    'magenta': '\033[35m',
}


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg):
    print(f"[AVISO] {msg}", file=sys.stderr)


def fmt(x: float) -> str:
    return f"{x:g}"


def midi_to_name(midi: int) -> str:
    return f"{PITCH_NAMES[midi % 12]}{midi // 12 - 1}"


def parse_key(text: str) -> tuple:
    """'C', 'F#', 'Bb', 'Am' → (pitch_class, modo_implícito|None)."""
    s = text.strip()
    hint = None
    if len(s) >= 2 and s.endswith('m') and (s[:-1][:1].upper() + s[:-1][1:]) in NOTE_PC:
        s, hint = s[:-1], 'minor'
    s = s[:1].upper() + s[1:]
    if s not in NOTE_PC:
        fail(f"tónica no reconocida: '{text}'")
    return NOTE_PC[s] % 12, hint


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENTE JEV CON CACHE LOCAL
# ═══════════════════════════════════════════════════════════════════════════════

class Jev:
    """Cliente de Jev (TypeSafe) vía OpenRouter, con cache JSONL opcional.

    ask() devuelve (elección más probable, {opción: probabilidad normalizada}).
    La clave de cache es el hash de (modelo, estado, instrucciones, opciones):
    solo una consulta idéntica se sirve desde la cache.
    """

    def __init__(self, args):
        self.model      = args.model
        self.url        = args.url
        self.api_key    = os.environ.get('OPENROUTER_API_KEY')
        self.cache_only = args.cache_only
        self.use_cache  = args.cache or args.cache_only
        self.path       = os.path.expanduser(args.cache_file)
        self.verbose    = args.verbose
        self.cache, self.full = (load_cache(self.path) if self.use_cache else ({}, set()))
        self.stats      = Counter()
        self.log        = []

    # ── API ──────────────────────────────────────────────────────────────────
    def ask(self, kind: str, state: str, instructions: str, criteria: dict):
        if len(criteria) > MAX_OPTIONS:
            fail(f"la pregunta '{kind}' tiene {len(criteria)} opciones (máx. {MAX_OPTIONS})")
        if len(criteria) == 1:                       # nada que preguntar
            only = next(iter(criteria))
            return only, {only: 1.0}

        ckey = self.cache_key(state, instructions, criteria)

        ans = self.cache.get(ckey) if self.use_cache else None
        cached = ans is not None
        if cached:
            self.stats['cache'] += 1
            if not self.cache_only and ckey not in self.full:
                # entrada antigua (solo hash): se completa con todos los datos
                self._store(ckey, kind, instructions, ans, state, criteria,
                            source=infer_source(ans))
                self.stats['completadas'] += 1
        else:
            if self.cache_only:
                fail(f"--cache-only: consulta '{kind}' no encontrada en {self.path}. "
                     f"Repite con --cache (y los mismos parámetros y --rng) para "
                     f"rellenarla.")
            ans = self._request(state, instructions, criteria)
            self.stats['api'] += 1
            if self.use_cache:
                self._store(ckey, kind, instructions, ans, state, criteria)

        choice = ans['choice']
        if choice not in criteria:
            fail(f"Jev devolvió una opción que no estaba en la lista: {choice!r}")
        probs = _normalize_probs(ans, criteria)
        if not ans.get('probabilities'):
            self.stats['sin_probs'] += 1
        self.log.append({'kind': kind, 'choice': choice,
                         'p': round(probs.get(choice, 1.0), 4), 'cached': cached})
        return choice, probs

    def cache_key(self, state: str, instructions: str, criteria: dict) -> str:
        """Hash de (modelo, estado, instrucciones, opciones): la clave de cache."""
        return hashlib.sha256(json.dumps(
            {'model': self.model, 'state': state, 'instructions': instructions,
             'criteria': criteria}, sort_keys=True, ensure_ascii=False
        ).encode('utf-8')).hexdigest()

    def _request(self, state, instructions, criteria):
        if not self.api_key:
            fail("falta la variable OPENROUTER_API_KEY (o usa --cache-only)")
        body = json.dumps({
            'model': self.model,
            'state': state,
            'questions': {'pick': {'type': 'choice', 'instructions': instructions,
                                   'criteria': criteria}},
        }).encode('utf-8')
        req = urllib.request.Request(self.url, data=body, method='POST', headers={
            'Authorization': 'Bearer ' + self.api_key,
            'Content-Type': 'application/json',
            'User-Agent': 'jev-composer/4.0',
        })
        # Reintentos solo para errores transitorios (límite de peticiones, 5xx, red)
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    resp = json.load(r)
                break
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}"
                if e.code not in (408, 429, 500, 502, 503, 504):
                    fail(last)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = str(e)
            except ValueError:
                fail("la respuesta de Jev no es JSON válido")
            if attempt < 3:
                time.sleep(2 ** attempt)
        else:
            fail(f"Jev sin respuesta tras reintentos ({last})")
        try:
            ans = resp['answers']['pick']
            ans['choice']
        except (KeyError, TypeError):
            fail(f"respuesta inesperada de Jev: {json.dumps(resp)[:300]}")
        return {'choice': ans['choice'], 'probabilities': ans.get('probabilities') or {}}

    def _store(self, ckey, kind, instructions, ans, state='', criteria=None,
               source='api'):
        """Añade un registro a la cache. Lleva todo lo necesario para reconstruir
        la pregunta y su distribución de salida (p. ej. para entrenar un modelo):
        estado, instrucciones y opciones completos; ver FORMATO DE LA CACHE."""
        self.cache[ckey] = ans
        rec = {'key': ckey, 'kind': kind, 'model': self.model, 'source': source,
               'instructions': instructions, 'state': state, 'criteria': criteria}
        marker = 'EXTRA INFORMATION FOR THIS POSITION:\n'
        if marker in state:      # indicaciones por compás: texto visible (solo informativo)
            rec['extra'] = state.split(marker, 1)[1].strip().replace('\n', ' | ')
        rec['answer'] = ans
        if state and criteria:
            self.full.add(ckey)
        with open(_ensure_parent(self.path), 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    def summary(self) -> str:
        api, hit = self.stats['api'], self.stats['cache']
        s = f"{api + hit} consulta(s): {api} a la API, {hit} desde la cache"
        if self.stats['sin_probs']:
            s += (f"  [AVISO] {self.stats['sin_probs']} respuesta(s) sin "
                  f"probabilidades: en ellas el sampling es determinista")
        return s


def infer_source(ans: dict) -> str:
    """Origen de una respuesta antigua sin campo 'source': one-hot (1.0 en una
    opción y 0.0 en el resto) → reconstruida por train; si no, de la API."""
    vals = list((ans.get('probabilities') or {}).values())
    return 'train' if len(vals) > 1 and sorted(set(vals)) == [0.0, 1.0] \
        and vals.count(1.0) == 1 else 'api'


def load_cache(path: str) -> tuple:
    """(cache {clave: respuesta}, claves con registro completo)."""
    cache, full = {}, set()
    if not os.path.isfile(path):
        return cache, full
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[rec['key']] = rec['answer']
                if rec.get('state') and rec.get('criteria'):
                    full.add(rec['key'])
            except (ValueError, KeyError):
                warn(f"{path}:{n}: línea de cache ilegible, se ignora")
    return cache, full


def _normalize_probs(ans: dict, criteria: dict) -> dict:
    probs = {}
    for k, v in (ans.get('probabilities') or {}).items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if k in criteria and v > 0:
            probs[k] = v
    total = sum(probs.values())
    if total <= 0:
        return {ans['choice']: 1.0}
    return {k: v / total for k, v in probs.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def pick(probs: dict, temperature: float, rng: random.Random) -> str:
    """Elige una opción de la distribución de Jev reescalada con temperatura:
    p_i ∝ p_i^(1/T). T <= 0 → la más probable."""
    items = sorted(probs.items(), key=lambda kv: -kv[1])
    if temperature <= 0 or len(items) == 1:
        return items[0][0]
    weights = [math.exp(math.log(p) / temperature) for _, p in items]
    r = rng.random() * sum(weights)
    acc = 0.0
    for (k, _), w in zip(items, weights):
        acc += w
        if r <= acc:
            return k
    return items[-1][0]


def top_probs(probs: dict, n: int = 3) -> str:
    items = sorted(probs.items(), key=lambda kv: -kv[1])[:n]
    return ' · '.join(f"{k} {p:.2f}" for k, p in items)


# ═══════════════════════════════════════════════════════════════════════════════
# INVOCACIÓN DE chord_table.py Y motif_table.py COMO COMANDOS
# ═══════════════════════════════════════════════════════════════════════════════

def run_tool(script: str, argv: list):
    """Ejecuta la herramienta con el mismo intérprete. (stdout, '') o (None, stderr)."""
    if not os.path.isfile(script):
        fail(f"no se encuentra {script} (usa --chord-table / --motif-table)")
    r = subprocess.run([sys.executable, script] + argv, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout).strip()
    return r.stdout, ''


def tool_catalog(script: str, mode: str) -> tuple:
    """Catálogo de la herramienta filtrado por modo (--mode M --export-json F).
    Devuelve (entradas, modo_usado). Aplica MODE_FALLBACK si el modo no existe."""
    chain = [mode] + MODE_FALLBACK.get(mode, [])
    with tempfile.TemporaryDirectory() as td:
        for m in chain:
            out = os.path.join(td, f'cat_{m}.json')
            ok, _ = run_tool(script, ['--mode', m, '--export-json', out])
            if ok is None or not os.path.isfile(out):
                continue          # modo no admitido o sin entradas
            with open(out, encoding='utf-8') as f:
                entries = json.load(f)
            if entries:
                if m != mode:
                    warn(f"{os.path.basename(script)} no tiene entradas en modo "
                         f"'{mode}': se usa '{m}'")
                return entries, m
    fail(f"{os.path.basename(script)}: sin entradas para el modo '{mode}'")


def resolve_chord_seed(script: str, entry_id: int, tonic: int) -> list:
    """chord_table.py --id N --key K --reps 1 → lista de acordes concretos."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'c.json')
        ok, err = run_tool(script, ['--id', str(entry_id), '--key', PITCH_NAMES[tonic],
                                    '--reps', '1', '--export-json', out,
                                    '--output', os.path.join(td, 's')])
        if ok is None:
            fail(f"chord_table.py --id {entry_id}: {err}")
        with open(out, encoding='utf-8') as f:
            return json.load(f)['progression']


def resolve_motif_seed(script: str, entry_id: int, tonic: int, octave: int) -> list:
    """motif_table.py --id N --key K --octave O --reps 1 → lista de notas."""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, 'm.json')
        ok, err = run_tool(script, ['--id', str(entry_id), '--key', PITCH_NAMES[tonic],
                                    '--octave', str(octave), '--reps', '1',
                                    '--export-json', out,
                                    '--output', os.path.join(td, 's')])
        if ok is None:
            fail(f"motif_table.py --id {entry_id}: {err}")
        with open(out, encoding='utf-8') as f:
            return json.load(f)['notes']


# ═══════════════════════════════════════════════════════════════════════════════
# FASES 1 Y 2: TONALIDAD Y SEMILLAS
# ═══════════════════════════════════════════════════════════════════════════════

def description_state(desc: str, tonic=None, mode=None, chord_seed=None,
                      chord_label='CHORD SEED', tonic_hint=None) -> str:
    s = f"PIECE DESCRIPTION:\n{desc.strip()}\n"
    if mode is not None:
        s += f"\nMODE: {mode}\n"
    if tonic is not None:
        s += f"KEY: {PITCH_NAMES[tonic]} {mode}\n"
    elif tonic_hint is not None:
        s += f"\nTONIC: {PITCH_NAMES[tonic_hint]}\n"
    if chord_seed:
        text = progression_text(chord_seed[:16])
        if len(chord_seed) > 16:
            text += f" ... ({len(chord_seed)} chords in total)"
        s += f"{chord_label}: {text}\n"
    return s


def show_choice(title: str, chosen: str, probs: dict, verbose: bool) -> None:
    line = f"  {COL['gray']}{title:<10}{COL['reset']} {COL['bold']}{chosen}{COL['reset']}"
    if verbose:
        line += f"  {COL['gray']}({top_probs(probs)}){COL['reset']}"
    print(line)


def choose_key_mode(jev, desc, args, rng, fixed_chords=None, inferred_tonic=None):
    """Jev elige el modo y después la tónica (salvo --mode / --key).
    Con acordes fijos (--chords-midi) el modo se elige viendo esos acordes y la
    tónica se infiere del MIDI en lugar de preguntarla."""
    forced_tonic, hint = (parse_key(args.key) if args.key else (None, None))
    mode = args.mode or hint
    if fixed_chords and forced_tonic is None:
        forced_tonic = inferred_tonic
    fixed_label = {'chord_label': 'CHORDS (fixed)'}

    if mode is None:
        if fixed_chords:
            mode_state = description_state(desc, chord_seed=fixed_chords,
                                           tonic_hint=forced_tonic, **fixed_label)
        else:
            mode_state = description_state(desc)
        choice, probs = jev.ask(
            'modo', mode_state,
            'Which musical mode best suits the piece described in this document? '
            'Choose the single best fit.', dict(MODE_INFO))
        mode = pick(probs, args.seed_temp, args.rng_obj)
        show_choice('modo', mode, probs, args.verbose)
    else:
        show_choice('modo', f"{mode} (fijado)", {}, False)

    if forced_tonic is None:
        criteria = {(PITCH_NAMES[i] if PITCH_NAMES[i] == PITCH_NAMES_FLAT[i]
                     else f"{PITCH_NAMES[i]}/{PITCH_NAMES_FLAT[i]}"):
                    f"Tonic note {PITCH_NAMES[i]}" for i in range(12)}
        choice, probs = jev.ask(
            'tonica', description_state(desc, mode=mode),
            f'The piece will be in {mode} mode. Which tonic note (key centre) best '
            f'suits the piece described in this document? Choose the single best fit.',
            criteria)
        label = pick(probs, args.seed_temp, args.rng_obj)
        tonic = NOTE_PC[label.split('/')[0]]
        show_choice('tónica', label, probs, args.verbose)
    else:
        tonic = forced_tonic
        src = 'inferida del MIDI' if (fixed_chords and not args.key) else 'fijada'
        show_choice('tónica', f"{PITCH_NAMES[tonic]} ({src})", {}, False)
    return tonic, mode


def choose_seeds(jev, desc, tonic, mode, args, fixed_chords=None):
    """Jev elige la progresión y el motivo semilla entre los del catálogo
    (obtenido por invocación de comandos) para el modo elegido."""
    rng = args.rng_obj
    chord_script = args.chord_table
    motif_script = args.motif_table

    # ── Progresión de acordes ────────────────────────────────────────────────
    if fixed_chords:                      # --chords-midi: no se elige ni se genera
        chord_entry, chord_mode, progression = None, None, fixed_chords
        chord_label = 'CHORDS (fixed)'
        show_choice('acordes', f"{len(progression)} del MIDI", {}, False)
    else:
        chord_label = 'CHORD SEED'
        entries, chord_mode = tool_catalog(chord_script, mode)
        by_label = {}
        for e in entries:
            label = f"#{e['id']} {e['prog']}"
            by_label[label] = (e, (f"{e['nombre']}: {e['style']}/{e['mode']}, tension "
                                   f"{e['tension']}/10, emotion {e['emocion']}. {e['desc']}"))
        choice, probs = jev.ask(
            'semilla_acordes', description_state(desc, tonic, mode),
            'Which chord progression best suits the piece described in this document? '
            'Choose the single best fit.', {k: v[1] for k, v in by_label.items()})
        label = pick(probs, args.seed_temp, rng)
        chord_entry = by_label[label][0]
        show_choice('acordes', label, probs, args.verbose)
        progression = resolve_chord_seed(chord_script, chord_entry['id'], tonic)

    # ── Motivo melódico (ve ya la progresión elegida) ────────────────────────
    entries, motif_mode = tool_catalog(motif_script, mode)
    by_label = {}
    for e in entries:
        label = f"#{e['id']} {e['nombre']} [{e['degrees']}]"
        by_label[label] = (e, (f"{e['tipo']}, {e['contorno']} contour, {e['style']}/"
                               f"{e['mode']}, tension {e['tension']}/10, emotion "
                               f"{e['emocion']}. {e['desc']}"))
    choice, probs = jev.ask(
        'semilla_motivo',
        description_state(desc, tonic, mode, progression, chord_label=chord_label),
        'Which melodic motif best suits the piece described in this document, '
        'played over the chord seed? Choose the single best fit.'
        if not fixed_chords else
        'Which melodic motif best suits the piece described in this document, '
        'played over the fixed chords? Choose the single best fit.',
        {k: v[1] for k, v in by_label.items()})
    label = pick(probs, args.seed_temp, rng)
    motif_entry = by_label[label][0]
    show_choice('motivo', label, probs, args.verbose)
    notes = resolve_motif_seed(motif_script, motif_entry['id'], tonic, args.octave)

    if not fixed_chords:
        print(f"\n  {COL['cyan']}{progression_text(progression)}{COL['reset']}")
    else:
        print()
    print(f"  {COL['cyan']}{melody_text(notes)}{COL['reset']}\n")
    return {'chord_entry': chord_entry, 'chord_mode': chord_mode,
            'progression': progression,
            'motif_entry': motif_entry, 'motif_mode': motif_mode, 'notes': notes}


# ═══════════════════════════════════════════════════════════════════════════════
# INDICACIONES POR COMPÁS
# ═══════════════════════════════════════════════════════════════════════════════

EXTRA_HINT = " Take into account the extra information given for this position."

BAR_NOTE_RE = re.compile(
    r"^(?:(?:compases|compás|compas|bars|bar|c|m)\.?\s*)?"
    r"([\d\s,\-–]+?)\s*[:=]\s*(.+)$", re.IGNORECASE)


def parse_bar_spec(spec: str, line: str) -> list:
    """'29' | '30-32' | '28, 33' | '1-4, 9' → lista de compases (desde 1)."""
    bars = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*[-–]\s*(\d+))?", part)
        if not m:
            fail(f"indicación de compás no válida: '{line}'")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a < 1 or b < a:
            fail(f"rango de compases no válido ({part}) en: '{line}'")
        bars.extend(range(a, b + 1))
    if not bars:
        fail(f"indicación sin compás: '{line}'")
    return bars


def parse_bar_notes(args) -> dict:
    """Une --notes (repetible, entradas separadas por ';') y --notes-file (una
    por línea) en {compás: [textos]}. Los compases se numeran desde 1."""
    raw = []
    for text in (getattr(args, 'notes', None) or []):
        raw.extend(text.split(';'))
    nf = getattr(args, 'notes_file', None)
    if nf:
        if not os.path.isfile(nf):
            fail(f"No se encuentra el archivo: {nf}")
        with open(nf, encoding='utf-8') as f:
            raw.extend(l for l in f.read().splitlines() if not l.lstrip().startswith('#'))
    return parse_bar_note_lines(raw, args.bars)


def parse_bar_note_lines(raw: list, max_bars) -> dict:
    """Líneas 'compás N: texto' → {compás: [textos]} (los fuera de rango se ignoran)."""
    notes = {}
    for line in raw:
        line = line.strip()
        if not line:
            continue
        m = BAR_NOTE_RE.match(line)
        if not m:
            fail(f"indicación no válida (formato 'compás N: texto'): '{line}'")
        text = m.group(2).strip()
        for b in parse_bar_spec(m.group(1), line):
            notes.setdefault(b, []).append(text)
    out_of_range = sorted(b for b in notes if max_bars and b > max_bars)
    if out_of_range:
        warn(f"indicaciones fuera de la obra ({max_bars} compases), se ignoran: "
             f"{', '.join(map(str, out_of_range))}")
        for b in out_of_range:
            del notes[b]
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# MODO ACORDES FIJOS: LECTURA DE LOS ACORDES DE UN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def load_chord_table_module(args):
    """Importa chord_table.py como módulo (su detector de acordes)."""
    if not os.path.isfile(args.chord_table):
        fail(f"no se encuentra {args.chord_table} (usa --chord-table)")
    spec = importlib.util.spec_from_file_location('chord_table_mod', args.chord_table)
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)
    return ct


def midi_to_chord_segments(ct, path: str, bass_only: bool = False,
                           min_dur: float = 0.5) -> tuple:
    """Segmenta un MIDI en acordes: [(pcs, bajo_midi, inicio_tick, dur_ticks)].

    Sustituye a ct.midi_to_chord_sequence, que muestrea en ventanas de 1 pulso y
    fusiona con el anterior todo segmento de <= 1 pulso (los acordes que cambian
    cada pulso, o más rápido, desaparecen). Aquí el corte es exacto: cada cambio
    del conjunto de notas activas abre un segmento. Se descartan los de menos de
    2 clases de altura (1 con bass_only), los contiguos con las mismas clases se
    unen, y los de menos de `min_dur` pulsos se funden con el anterior (limpia
    los solapes de una interpretación humana)."""
    BASS_THRESHOLD = 61                                  # igual que chord_table.py
    raw, tpb = ct._parse_midi_raw(path)
    if not raw:
        return [], tpb
    raw = sorted(raw, key=lambda e: (e[0], e[1] == 'on'))   # offs antes que ons
    active, intervals = {}, []
    for tick, etype, note in raw:
        if etype == 'on':
            active[note] = tick
        elif note in active:
            intervals.append((active.pop(note), tick, note))
    end_all = max(t for t, _, _ in raw) + tpb
    intervals += [(st, end_all, n) for n, st in active.items()]
    if bass_only:
        intervals = [i for i in intervals if i[2] < BASS_THRESHOLD]
    points = sorted([(st, 1, n) for st, _, n in intervals] +
                    [(en, 0, n) for _, en, n in intervals], key=lambda x: (x[0], x[1]))
    ticks = sorted({pt[0] for pt in points})

    min_pcs = 1 if bass_only else 2
    segs, act, idx = [], Counter(), 0
    for k, t in enumerate(ticks):
        while idx < len(points) and points[idx][0] == t:
            _, on, n = points[idx]
            idx += 1
            if on:
                act[n] += 1
            else:
                act[n] -= 1
                if act[n] <= 0:
                    del act[n]
        if k + 1 < len(ticks) and act:
            pcs = frozenset(n % 12 for n in act)
            if len(pcs) >= min_pcs:
                segs.append([pcs, min(act), t, ticks[k + 1] - t])

    def compress(items):
        out = []
        for pcs, bass, st, du in items:
            if out and out[-1][0] == pcs:
                out[-1][1] = min(out[-1][1], bass)
                out[-1][3] = st + du - out[-1][2]
            else:
                out.append([pcs, bass, st, du])
        return out

    segs = compress(segs)
    merged = []
    for pcs, bass, st, du in segs:
        if merged and du < min_dur * tpb:               # corto: se funde con el anterior
            merged[-1][3] = st + du - merged[-1][2]
        else:
            merged.append([pcs, bass, st, du])
    return [tuple(x) for x in compress(merged)], tpb


def detect_chords(path: str, args, ct, key_text=None) -> tuple:
    """Acordes de un MIDI con el detector de chord_table.py. Devuelve
    (progresión, tónica_inferida); la progresión es contigua desde el pulso 0:
    cada acorde dura hasta el inicio del siguiente."""
    try:
        events, tpb = midi_to_chord_segments(ct, path, args.bass_only, args.min_dur)
        recognized = ct.chord_sequence_to_numerals(events, tpb)
    except Exception as e:                      # MIDI corrupto o no estándar
        fail(f"no se pudo leer {path}: {e}")
    recognized = [c for c in recognized if c['root_pc'] is not None]
    if not recognized:
        fail(f"no se detectó ningún acorde en {path}")
    recognized.sort(key=lambda c: c['start'])
    unified = []                                # mismo acorde seguido (p. ej. otra inversión)
    for c in recognized:
        if unified and (unified[-1]['root_pc'], unified[-1]['quality']) == \
                (c['root_pc'], c['quality']):
            unified[-1]['dur_beats'] = c['start'] + c['dur_beats'] - unified[-1]['start']
        else:
            unified.append(dict(c))
    recognized = unified

    inferred = ct.infer_tonic(recognized)
    tonic = parse_key(key_text)[0] if key_text else inferred
    progression = []
    for i, c in enumerate(recognized):
        start = 0.0 if i == 0 else c['start']
        end = (recognized[i + 1]['start'] if i + 1 < len(recognized)
               else c['start'] + c['dur_beats'])
        dur = end - start
        if dur <= EPS:
            continue
        quality = c['quality'] or ''
        root = c['root_pc']
        progression.append({
            'numeral': ct.chord_to_numeral(root, quality, tonic),
            'chord': f"{PITCH_NAMES[root]}{quality}",
            'pitches': [(root + iv) % 12 for iv in ct.CHORD_INTERVALS.get(quality, [0, 4, 7])],
            'duration_beats': round(dur, 6),
        })
    return progression, inferred


def load_chords_midi(args) -> tuple:
    """--chords-midi: (progresión, tónica_inferida) del MIDI de acordes."""
    path = args.chords_midi
    if not os.path.isfile(path):
        fail(f"No se encuentra el archivo: {path}")
    return detect_chords(path, args, load_chord_table_module(args), args.key)


def fit_progression(progression: list, total: float) -> list:
    """Recorta la progresión a `total` pulsos o alarga el último acorde hasta él."""
    out, t = [], 0.0
    for c in progression:
        d = min(c['duration_beats'], total - t)
        if d <= EPS:
            break
        out.append(dict(c, duration_beats=d))
        t += d
    if out and t < total - EPS:
        out[-1]['duration_beats'] += total - t
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# ARMONÍA Y MELODÍA: CANDIDATOS OFRECIDOS A JEV
# ═══════════════════════════════════════════════════════════════════════════════

def diatonic_chords(tonic: int, mode: str) -> dict:
    """Acordes diatónicos del modo (tríadas y séptimas) + dominante V7.
    Devuelve {etiqueta: {...}}; la etiqueta es la que ve Jev ('bVII (G)')."""
    sc = MODE_INTERVALS[mode]
    major = MODE_INTERVALS['major']
    out = {}

    def add(numeral, root_iv, quality, tones, function):
        root_pc = (tonic + root_iv) % 12
        name = f"{PITCH_NAMES[root_pc]}{quality}"
        label = f"{numeral} ({name})"
        if label in out:
            return
        pcs = [(root_pc + iv) % 12 for iv in tones]
        out[label] = {
            'label': label, 'numeral': numeral, 'chord': name, 'pitches': pcs,
            'desc': f"{function}; notes {' '.join(PITCH_NAMES[p] for p in pcs)}",
        }

    for i in range(7):
        root = sc[i]
        tones = [sc[(i + k) % 7] + 12 * ((i + k) // 7) - root for k in (0, 2, 4, 6)]
        acc = 'b' if root < major[i] else '#' if root > major[i] else ''
        tq = TRIAD_QUALITY.get(tuple(t % 12 for t in tones[:3]))
        if tq is not None:
            numeral = acc + (ROMAN[i].lower() if tq in ('m', 'dim') else ROMAN[i])
            numeral += {'dim': '°', 'aug': '+'}.get(tq, '')
            add(numeral, root, tq, [t % 12 for t in tones[:3]], DEGREE_FUNCTION[i])
        sq = SEVENTH_QUALITY.get(tuple(t % 12 for t in tones[:4]))
        if sq is not None:
            numeral = acc + (ROMAN[i].lower() if sq in ('m7', 'hdim7', 'dim7') else ROMAN[i])
            numeral += SEVENTH_SUFFIX[sq]
            add(numeral, root, sq, [t % 12 for t in tones[:4]], DEGREE_FUNCTION[i] + ', with 7th')
    if not any(c['chord'] == f"{PITCH_NAMES[(tonic + 7) % 12]}7" for c in out.values()):
        add('V7', 7, '7', [0, 4, 7, 10],
            'dominant (borrowed), strong pull towards the tonic')
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3: VENTANA DESLIZANTE
# ═══════════════════════════════════════════════════════════════════════════════

class Composer:
    """Genera acordes y melodía con una ventana deslizante sobre Jev.

    Línea de tiempo en pulsos. `chords` y `melody` son listas de dicts con
    start/dur. El ritmo armónico y el melódico salen de las semillas (en ciclo).
    """

    def __init__(self, jev, desc, tonic, mode, seeds, args):
        self.jev, self.desc, self.tonic, self.mode = jev, desc, tonic, mode
        self.seeds, self.args = seeds, args
        self.rng = args.rng_obj
        self.fixed = bool(getattr(args, 'chords_midi', None))
        self.bar_notes = getattr(args, 'bar_notes_map', None) or {}
        self.beats = args.beats
        self.total = float(args.bars * args.beats)
        self.depth = max(1, args.depth)
        self.base = args.octave * 12 + tonic
        self.scale_pcs = {(tonic + iv) % 12 for iv in MODE_INTERVALS[mode]}
        self.candidates = diatonic_chords(tonic, mode)

        self.chord_durs = [float(c['duration_beats']) for c in seeds['progression']] \
            or [self.beats / 2.0]
        self.rhythm = [(float(n['duration_beats']), n['midi'] is None)
                       for n in seeds['notes']] or [(1.0, False)]
        self.rhythm_i = len(seeds['notes'])
        self.chords, self.melody = [], []
        self.mel_t = 0.0
        self.last_midi = None

    # ── estado enviado a Jev ─────────────────────────────────────────────────
    def pos(self, t: float) -> str:
        return f"bar {int(t // self.beats) + 1} beat {fmt(round(t % self.beats + 1, 4))}"

    def notes_of(self, chord: dict) -> list:
        end = chord['start'] + chord['dur']
        return [m for m in self.melody if chord['start'] - EPS <= m['start'] < end - EPS]

    def extra(self, t0: float, t1: float) -> str:
        """Indicaciones de los compases que cubre [t0, t1) (vacío si no hay)."""
        b0 = int(t0 // self.beats) + 1
        b1 = int((max(t1, t0 + EPS) - EPS) // self.beats) + 1
        return '\n'.join(f"bar {b}: {'; '.join(self.bar_notes[b])}"
                         for b in range(b0, b1 + 1) if b in self.bar_notes)

    def state(self, extra: str = '') -> str:
        scale = ' '.join(PITCH_NAMES[(self.tonic + iv) % 12]
                         for iv in MODE_INTERVALS[self.mode])
        lines = []
        visible = self.chords
        upcoming = ''
        if self.fixed:        # acordes fijos: ventana hasta el actual + 2 siguientes
            cur = self.chords.index(self.chord_at(self.mel_t))
            visible = self.chords[:cur + 1]
            nxt = self.chords[cur + 1:cur + 3]
            if nxt:
                upcoming = ('UPCOMING CHORDS: ' + ' '.join(
                    f"{c['chord']}({fmt(c['dur'])})" for c in nxt) + '\n')
        for c in visible[-self.depth:]:
            ns = ' '.join(f"{m['note']}({fmt(m['dur'])})" for m in self.notes_of(c))
            lines.append(f"{self.pos(c['start'])} | {c['chord']} ({fmt(c['dur'])} beats): "
                         f"{ns or '(no melody yet)'}")
        if self.fixed:
            chord_line = (f"CHORDS: fixed, {len(self.chords)} chords in total "
                          f"(supplied, not generated)\n")
        else:
            chord_line = f"CHORD SEED: {progression_text(self.seeds['progression'])}\n"
        return (f"PIECE DESCRIPTION:\n{self.desc.strip()}\n\n"
                f"KEY: {PITCH_NAMES[self.tonic]} {self.mode} (scale: {scale})\n"
                f"METER: {self.beats} beats per bar, {self.args.bars} bars in total\n"
                f"{chord_line}"
                f"MOTIF SEED: {melody_text(self.seeds['notes'])}\n\n"
                f"MUSIC SO FAR (last {min(self.depth, len(visible))} chord(s), "
                f"melody as note(beats)):\n" + '\n'.join(lines) + '\n' + upcoming
                + (f"\nEXTRA INFORMATION FOR THIS POSITION:\n{extra}\n" if extra else ''))

    # ── semilla ──────────────────────────────────────────────────────────────
    def place_seed(self) -> float:
        t = 0.0
        for c in self.seeds['progression']:
            d = min(float(c['duration_beats']), self.total - t)
            if d <= EPS:
                break
            self.chords.append({'chord': c['chord'], 'numeral': c.get('numeral', ''),
                                'pitches': c['pitches'], 'start': t, 'dur': d,
                                'source': 'midi' if self.fixed else 'seed'})
            t = round(t + d, 6)
        seed_end = t
        t = 0.0
        for n in self.seeds['notes']:
            d = min(float(n['duration_beats']), self.total - t)
            if d <= EPS:
                break
            self.melody.append({'note': n['note'], 'midi': n['midi'], 'start': t,
                                'dur': d, 'source': 'seed'})
            if n['midi'] is not None:
                self.last_midi = n['midi']
            t = round(t + d, 6)
        self.mel_t = t
        return seed_end

    # ── acorde siguiente ─────────────────────────────────────────────────────
    def chord_query(self, t: float, dur: float) -> tuple:
        """(estado, instrucciones, opciones) de la pregunta del acorde siguiente.
        Compartida con el modo train: garantiza hashes de cache idénticos."""
        criteria = {k: v['desc'] for k, v in self.candidates.items()}
        extra = self.extra(t, t + dur)
        instr = (f"Which chord should come next? It starts at {self.pos(t)} and lasts "
                 f"{fmt(dur)} beat(s). Choose the single best fit for the description, "
                 f"the seeds and the harmonic flow." + (EXTRA_HINT if extra else ''))
        return self.state(extra), instr, criteria

    def next_chord(self, t: float) -> dict:
        dur = min(self.chord_durs[len(self.chords) % len(self.chord_durs)],
                  self.total - t)
        state, instr, criteria = self.chord_query(t, dur)
        choice, probs = self.jev.ask('acorde', state, instr, criteria)
        label = pick(probs, self.args.temperature, self.rng)
        if self.args.verbose:
            print(f"    {COL['gray']}acorde {self.pos(t)}: {label}  "
                  f"({top_probs(probs)}){COL['reset']}")
        c = self.candidates[label]
        chord = {'chord': c['chord'], 'numeral': c['numeral'], 'pitches': c['pitches'],
                 'start': t, 'dur': dur, 'source': 'jev'}
        self.chords.append(chord)
        return chord

    # ── melodía ──────────────────────────────────────────────────────────────
    def chord_at(self, t: float) -> dict:
        for c in reversed(self.chords):
            if c['start'] <= t + EPS:
                return c
        return self.chords[0]

    def pitch_candidates(self, chord: dict) -> dict:
        """{nombre: (midi, descripción)} de las notas ofrecidas a Jev: notas de la
        escala y del acorde a menos de --leap semitonos de la nota anterior."""
        lo, hi = self.base - 5, self.base + 19
        prev = self.last_midi if self.last_midi is not None else self.base
        allowed = self.scale_pcs | set(chord['pitches'])
        pool = [m for m in range(max(lo, prev - self.args.leap),
                                 min(hi, prev + self.args.leap) + 1) if m % 12 in allowed]
        if not pool:
            pool = [m for m in range(lo, hi + 1) if m % 12 in allowed]
        root = chord['pitches'][0]
        scale = MODE_INTERVALS[self.mode]
        out = {}
        for m in pool:
            pc = m % 12
            if pc in chord['pitches']:
                role = f"chord tone ({CHORD_ROLE.get((pc - root) % 12, 'tone')} of {chord['chord']})"
            else:
                rel = (pc - self.tonic) % 12
                role = (f"passing scale tone (degree {scale.index(rel) + 1})"
                        if rel in scale else "chromatic tone")
            if pc == self.tonic:
                role += ', the tonic'
            d = m - prev
            motion = ('repeats the previous note' if d == 0
                      else f"{'up' if d > 0 else 'down'} {abs(d)} semitone(s)")
            out[midi_to_name(m)] = (m, f"{role}; {motion}")
        return out

    def note_query(self, c: dict, t: float, dur: float) -> tuple:
        """(estado, instrucciones, opciones, candidatas) de la pregunta de la
        nota siguiente. Compartida con el modo train."""
        cands = self.pitch_candidates(c)
        extra = self.extra(t, t + dur)
        instr = (f"Which note should the melody play next? It starts at "
                 f"{self.pos(t)}, lasts {fmt(dur)} beat(s) and sounds over "
                 f"the chord {c['chord']}. Choose the single best fit for the "
                 f"description, the motif seed and the harmony."
                 + (EXTRA_HINT if extra else ''))
        return (self.state(extra), instr,
                {k: v[1] for k, v in cands.items()}, cands)

    def fill_melody(self, until: float) -> None:
        """Genera notas (pitch por Jev, duración por el ritmo de la semilla) hasta
        que la melodía alcanza `until`. Ninguna nota cruza un cambio de acorde."""
        until = min(until, self.total)
        while self.mel_t < until - EPS:
            c = self.chord_at(self.mel_t)
            c_end = c['start'] + c['dur']
            slot, is_rest = self.rhythm[self.rhythm_i % len(self.rhythm)]
            self.rhythm_i += 1
            dur = min(slot, until - self.mel_t, c_end - self.mel_t)
            if dur <= EPS:
                break
            if is_rest:
                self.melody.append({'note': 'R', 'midi': None, 'start': self.mel_t,
                                    'dur': dur, 'source': 'jev'})
            else:
                state, instr, criteria, cands = self.note_query(c, self.mel_t, dur)
                choice, probs = self.jev.ask('nota', state, instr, criteria)
                name = pick(probs, self.args.temperature, self.rng)
                if self.args.verbose:
                    print(f"    {COL['gray']}nota  {self.pos(self.mel_t)}: {name}  "
                          f"({top_probs(probs)}){COL['reset']}")
                midi = cands[name][0]
                self.melody.append({'note': name, 'midi': midi, 'start': self.mel_t,
                                    'dur': dur, 'source': 'jev'})
                self.last_midi = midi
            self.mel_t = round(self.mel_t + dur, 6)

    # ── bucle principal ──────────────────────────────────────────────────────
    def report(self, c: dict) -> None:
        ns = ' '.join(f"{m['note']}:{fmt(m['dur'])}" for m in self.notes_of(c))
        tag = COL['gray'] + {'seed': 'semilla', 'midi': 'midi   '}.get(
            c['source'], 'jev    ') + COL['reset']
        print(f"  {COL['gray']}c.{int(c['start'] // self.beats) + 1:<3}{COL['reset']} "
              f"{tag} {COL['bold']}{c['chord']:<7}{COL['reset']}"
              f"{COL['gray']}{fmt(c['dur']):>3}b{COL['reset']}  "
              f"{COL['cyan']}{ns}{COL['reset']}"
              + (f"  {COL['yellow']}◂ indicaciones{COL['reset']}"
                 if self.extra(c['start'], c['start'] + c['dur']) else ''))

    def compose(self) -> None:
        seed_end = self.place_seed()
        if not self.chords:
            fail("la progresión semilla está vacía")
        if self.fixed:        # acordes fijos: solo se genera la melodía
            for c in list(self.chords):
                self.fill_melody(c['start'] + c['dur'])
                self.report(c)
            return
        self.fill_melody(seed_end)
        for c in list(self.chords):
            self.report(c)
        t = seed_end
        while t < self.total - EPS:
            c = self.next_chord(t)
            t = round(c['start'] + c['dur'], 6)
            self.fill_melody(t)
            self.report(c)


# ═══════════════════════════════════════════════════════════════════════════════
# TEXTO Y EXPORTACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def progression_text(progression: list) -> str:
    """Formato compatible con --chords de los otros módulos."""
    return ' '.join(f"{c['chord']}:{fmt(c.get('duration_beats', c.get('dur')))}"
                    for c in progression)


def melody_text(notes: list) -> str:
    """Formato 'nota:beats' (R = silencio), como motif_table.py."""
    return ' '.join(f"{n['note']}:{fmt(n.get('duration_beats', n.get('dur')))}"
                    for n in notes)


def piece_to_midi(chords: list, melody: list, ticks_per_beat: int,
                  tempo_bpm: float, output_path: str) -> None:
    """MIDI de dos pistas: acordes en bloque (canal 1) y melodía (canal 0)."""
    mid = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)

    tc = mido.MidiTrack()
    mid.tracks.append(tc)
    tc.append(mido.MetaMessage('set_tempo', tempo=int(60_000_000 / tempo_bpm), time=0))
    tc.append(mido.MetaMessage('track_name', name='Jev Composer - chords', time=0))
    tc.append(mido.Message('program_change', channel=1, program=0, time=0))
    for c in chords:
        dur_ticks = max(1, int(c['dur'] * ticks_per_beat))
        notes = [48 + pc % 12 for pc in c['pitches']]
        for n in notes:
            tc.append(mido.Message('note_on', channel=1, note=n, velocity=60, time=0))
        for i, n in enumerate(notes):
            tc.append(mido.Message('note_off', channel=1, note=n, velocity=0,
                                   time=dur_ticks if i == 0 else 0))

    tm = mido.MidiTrack()
    mid.tracks.append(tm)
    tm.append(mido.MetaMessage('track_name', name='Jev Composer - melody', time=0))
    tm.append(mido.Message('program_change', channel=0, program=0, time=0))
    pending = 0
    for m in melody:
        dur_ticks = max(1, int(m['dur'] * ticks_per_beat))
        if m['midi'] is None:
            pending += dur_ticks
            continue
        tm.append(mido.Message('note_on', channel=0, note=m['midi'], velocity=80, time=pending))
        pending = 0
        tm.append(mido.Message('note_off', channel=0, note=m['midi'], velocity=0, time=dur_ticks))
    mid.save(_ensure_parent(output_path))


def seeds_json(seeds: dict) -> dict:
    return {
        'chords': (dict(seeds['chord_entry']) if seeds['chord_entry'] else None),
        'chords_mode_used': seeds['chord_mode'],
        'progression': seeds['progression'],
        'motif': {k: v for k, v in seeds['motif_entry'].items()},
        'motif_mode_used': seeds['motif_mode'],
        'notes': seeds['notes'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MODO TRAIN: RECONSTRUIR LA CACHE A PARTIR DE MIDIS
# ═══════════════════════════════════════════════════════════════════════════════

TXT_DIRECTIVE_RE = re.compile(
    r"^(descripci[oó]n|description|desc|tonalidad|key|modo|mode)\s*:\s*(.+)$",
    re.IGNORECASE)


class TeacherJev(Jev):
    """«Jev maestro»: en vez de consultar a la API, conoce la respuesta real
    (la que dice el MIDI) y la registra en la cache con probabilidad 1.0 en
    ella y 0.0 en el resto de opciones. Mismo formato y misma clave que Jev:
    las entradas son indistinguibles de las obtenidas de la API.

    Las preguntas de modo, tónica y semillas pasan por ask() (así se reutilizan
    choose_key_mode y choose_seeds tal cual); las de acorde y nota las registra
    directamente TrainComposer con record()."""

    def __init__(self, args):
        super().__init__(args)
        self.overwrite = args.overwrite
        self.data = None
        self.args_tools = args
        self.counts = Counter()
        self._seed_chords = {}      # id → progresión resuelta en Do (memo)
        self._seed_motifs = {}      # id → notas resueltas en Do, octava 5 (memo)

    # ── registro ─────────────────────────────────────────────────────────────
    def record(self, kind, state, instructions, criteria, truth) -> None:
        if len(criteria) < 2:                  # componer no pregunta con 1 opción
            return
        if truth not in criteria:
            self.counts['omitida:' + kind] += 1
            return
        key = self.cache_key(state, instructions, criteria)
        if key in self.cache and not self.overwrite:
            if key not in self.full:           # entrada antigua: se completa
                old = self.cache[key]
                self._store(key, kind, instructions, old, state, criteria,
                            source=infer_source(old))
                self.counts['completada'] += 1
            self.counts['existente'] += 1
            return
        ans = {'choice': truth,
               'probabilities': {k: (1.0 if k == truth else 0.0) for k in criteria}}
        self._store(key, kind, instructions, ans, state, criteria, source='train')
        self.counts['nueva'] += 1
        self.counts['nueva:' + kind] += 1

    def ask(self, kind, state, instructions, criteria):
        if len(criteria) == 1:
            only = next(iter(criteria))
            return only, {only: 1.0}
        truth = self.truth_for(kind, criteria)
        self.record(kind, state, instructions, criteria, truth)
        return truth, {truth: 1.0}

    # ── respuesta real de cada pregunta ──────────────────────────────────────
    def truth_for(self, kind, criteria):
        d = self.data
        if kind == 'modo':
            return d.mode
        if kind == 'tonica':
            for label in criteria:
                if NOTE_PC[label.split('/')[0]] % 12 == d.tonic:
                    return label
        if kind == 'semilla_acordes':
            return self.best_chord_seed(criteria)
        if kind == 'semilla_motivo':
            return self.best_motif_seed(criteria)
        fail(f"pregunta '{kind}' no soportada por el modo train")

    def best_chord_seed(self, criteria):
        """Entrada del catálogo cuya progresión coincide mejor con el arranque
        del MIDI (acorde a acorde, en rejilla de 0.25 pulsos). A igualdad gana
        la más corta: deja más música real para entrenar."""
        d, q = self.data, 0.25
        best, best_score = None, -1.0
        for label in criteria:
            eid = int(re.match(r'#(\d+)', label).group(1))
            if eid not in self._seed_chords:
                self._seed_chords[eid] = resolve_chord_seed(
                    self.args_tools.chord_table, eid, 0)
            seed = self._seed_chords[eid]
            length = min(sum(c['duration_beats'] for c in seed), d.total)
            hits = slots = 0
            for k in range(int(length / q)):
                t = (k + 0.5) * q
                slots += 1
                hits += (_seed_name_at(seed, t, d.tonic) == _name_at(d.chords, t))
            score = hits / max(1, slots) - 1e-6 * length
            if score > best_score + 1e-12:
                best, best_score = label, score
        return best

    def best_motif_seed(self, criteria):
        """Entrada del catálogo cuyo motivo coincide mejor con el arranque de la
        melodía (por clases de altura, en rejilla de 0.25 pulsos)."""
        d, q = self.data, 0.25
        best, best_score = None, -1.0
        for label in criteria:
            eid = int(re.match(r'#(\d+)', label).group(1))
            if eid not in self._seed_motifs:
                self._seed_motifs[eid] = resolve_motif_seed(
                    self.args_tools.motif_table, eid, 0, 5)
            notes = self._seed_motifs[eid]
            length = min(sum(n['duration_beats'] for n in notes), d.total)
            shift = d.tonic + (d.octave - 5) * 12
            hits = slots = 0
            for k in range(int(length / q)):
                t = (k + 0.5) * q
                slots += 1
                seed_m = _midi_at_seed(notes, t)
                real_m = _midi_at_melody(d.melody, t)
                if seed_m is None or real_m is None:
                    hits += (seed_m is None and real_m is None)
                else:
                    hits += ((seed_m + shift) % 12 == real_m % 12)
            score = hits / max(1, slots)
            if score > best_score + 1e-12:
                best, best_score = label, score
        return best


def _name_at(chords: list, t: float):
    for c in chords:
        if c['start'] - EPS <= t < c['start'] + c['dur'] - EPS:
            return c['chord']
    return None


def _seed_name_at(seed: list, t: float, tonic: int):
    """Nombre del acorde de la semilla (resuelta en Do) en t, transpuesto a la tónica."""
    acc = 0.0
    for c in seed:
        if acc <= t < acc + c['duration_beats']:
            name = c['chord']
            rn = 2 if len(name) > 1 and name[1] in '#b' else 1
            return PITCH_NAMES[(c['pitches'][0] + tonic) % 12] + name[rn:]
        acc += c['duration_beats']
    return None


def _midi_at_seed(notes: list, t: float):
    acc = 0.0
    for n in notes:
        if acc <= t < acc + n['duration_beats']:
            return n['midi']
        acc += n['duration_beats']
    return None


def _midi_at_melody(melody: list, t: float):
    for n in melody:
        if n['start'] - EPS <= t < n['start'] + n['dur'] - EPS:
            return n['midi']
    return None


# ── Lectura del MIDI: separar melodía y armonía ──────────────────────────────

def read_midi_notes(path: str) -> tuple:
    """(notas, ticks_por_pulso, pulsos_por_compás) de un MIDI. Cada nota:
    {start, end (ticks), pitch, track, channel}. Se ignora el canal 10 (batería)."""
    try:
        mid = mido.MidiFile(path)
    except Exception as e:
        fail(f"no se pudo leer {path}: {e}")
    notes, beats = [], None
    for ti, track in enumerate(mid.tracks):
        t, active = 0, {}
        for msg in track:
            t += msg.time
            if msg.type == 'time_signature' and beats is None:
                beats = max(1, round(msg.numerator * 4 / msg.denominator))
            elif msg.type == 'note_on' and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append(t)
            elif msg.type in ('note_off', 'note_on'):
                lst = active.get((msg.channel, msg.note))
                if lst:
                    st = lst.pop(0)
                    notes.append({'start': st, 'end': max(t, st + 1), 'pitch': msg.note,
                                  'track': ti, 'channel': msg.channel})
        for (ch, n), lst in active.items():
            for st in lst:
                notes.append({'start': st, 'end': max(t, st + 1), 'pitch': n,
                              'track': ti, 'channel': ch})
    notes = [n for n in notes if n['channel'] != 9]
    if not notes:
        fail(f"{path}: sin notas")
    return notes, mid.ticks_per_beat, beats or 4


def split_melody_harmony(notes: list, tpb: int, melody_track=None) -> tuple:
    """Separa (melodía, armonía). Melodía: la pista/canal monofónico de registro
    más agudo; si no hay ninguno, la nota más aguda de cada ataque (skyline)."""
    mel = None
    if melody_track is not None:
        mel = [n for n in notes if n['track'] == melody_track]
        if not mel:
            fail(f"--melody-track {melody_track}: la pista no tiene notas")
    else:
        groups = {}
        for n in notes:
            groups.setdefault((n['track'], n['channel']), []).append(n)
        best = None
        for g in groups.values():
            if len(g) < 4:
                continue
            g = sorted(g, key=lambda n: n['start'])
            overlaps = sum(1 for a, b in zip(g, g[1:]) if b['start'] < a['end'] - tpb / 8)
            if overlaps / len(g) > 0.15:
                continue                                   # polifónica: no es melodía
            mean = sum(n['pitch'] for n in g) / len(g)
            if best is None or mean > best[0]:
                best = (mean, g)
        if best:
            mel = best[1]
    if mel is None:                                        # skyline
        by_onset = {}
        for n in notes:
            by_onset.setdefault(round(n['start'] / (tpb / 8)), []).append(n)
        mel = [max(g, key=lambda n: n['pitch']) for g in by_onset.values()]
    ids = {id(n) for n in mel}
    return mel, [n for n in notes if id(n) not in ids]


def write_harmony_midi(notes: list, tpb: int, path: str) -> None:
    """MIDI temporal con solo la armonía, para el detector de chord_table.py."""
    ev = []
    for n in notes:
        ev.append((n['start'], 1, n['pitch']))
        ev.append((n['end'], 0, n['pitch']))
    ev.sort()
    mid = mido.MidiFile(type=0, ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    last = 0
    for tick, on, pitch in ev:
        tr.append(mido.Message('note_on' if on else 'note_off', channel=0, note=pitch,
                               velocity=70 if on else 0, time=tick - last))
        last = tick
    mid.save(path)


def quantize_melody(mel: list, tpb: int, q: float) -> list:
    """Melodía monofónica en pulsos y cuantizada: [{start, dur, midi}]."""
    mono = []
    for n in sorted(mel, key=lambda n: (n['start'], -n['pitch'])):
        if mono and n['start'] == mono[-1]['start']:
            continue                                       # mismo ataque: la más aguda
        mono.append(dict(n))
    for a, b in zip(mono, mono[1:]):
        a['end'] = min(a['end'], b['start'])               # legato → sin solapes
    out, prev_end = [], 0.0
    for n in mono:
        s_ = max(round(n['start'] / tpb / q) * q, prev_end)
        e_ = round(n['end'] / tpb / q) * q
        if e_ - s_ < q - EPS:
            continue                                       # más corta que la rejilla
        out.append({'start': round(s_, 6), 'dur': round(e_ - s_, 6), 'midi': n['pitch']})
        prev_end = e_
    return out


def infer_mode(chords: list, melody: list, tonic: int) -> str:
    """Modo cuya escala (en la tónica dada) contiene más duración de notas y acordes."""
    w = [0.0] * 12
    for n in melody:
        w[n['midi'] % 12] += n['dur']
    for c in chords:
        for pc in c['pitches']:
            w[pc % 12] += c['duration_beats'] * 0.5
    total = sum(w) or 1.0
    best, best_score = MODES[0], -1.0
    for m in MODES:
        pcs = {(tonic + iv) % 12 for iv in MODE_INTERVALS[m]}
        score = sum(w[p] for p in pcs) / total
        if score > best_score + 1e-9:
            best, best_score = m, score
    return best


def read_train_txt(path: str):
    """X.txt → (descripción|None, tonalidad|None, modo|None, líneas de indicaciones)."""
    desc, key, mode, notes = [], None, None, []
    if not os.path.isfile(path):
        return None, None, None, []
    with open(path, encoding='utf-8') as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = TXT_DIRECTIVE_RE.match(line)
            if not m:
                notes.append(line)
                continue
            kind, value = m.group(1).lower(), m.group(2).strip()
            if kind.startswith(('descr', 'desc')):
                desc.append(value)
            elif kind in ('tonalidad', 'key'):
                key = value
            else:
                mode = value
    return ('\n'.join(desc) or None), key, mode, notes


# ── Recorrido con ventana deslizante ─────────────────────────────────────────

class TrainComposer(Composer):
    """Recorre la armonía y la melodía reales con la misma ventana deslizante
    que Composer y, en vez de preguntar a Jev, registra la respuesta real."""

    def __init__(self, jev, data, seeds, args):
        super().__init__(jev, data.desc, data.tonic, data.mode, seeds, args)
        self.data = data
        self.starts = {round(n['start'], 6): n for n in data.melody}
        self.start_list = sorted(self.starts)

    def follow_melody(self, until: float) -> None:
        """Equivalente de fill_melody con la melodía real: una consulta por nota
        (si estaba entre las opciones) y silencios en los huecos. Ninguna nota
        cruza un cambio de acorde (lo que sobra pasa a silencio)."""
        until = min(until, self.total)
        while self.mel_t < until - EPS:
            c = self.chord_at(self.mel_t)
            c_end = c['start'] + c['dur']
            note = self.starts.get(round(self.mel_t, 6))
            if note:
                dur = min(note['dur'], until - self.mel_t, c_end - self.mel_t)
                if dur <= EPS:
                    break
                state, instr, criteria, cands = self.note_query(c, self.mel_t, dur)
                name = midi_to_name(note['midi'])
                self.jev.record('nota', state, instr, criteria, name)
                self.melody.append({'note': name, 'midi': note['midi'],
                                    'start': self.mel_t, 'dur': dur, 'source': 'train'})
                self.last_midi = note['midi']
            else:
                k = bisect.bisect_right(self.start_list, self.mel_t + EPS)
                nxt = self.start_list[k] if k < len(self.start_list) else float('inf')
                dur = min(nxt - self.mel_t, until - self.mel_t, c_end - self.mel_t)
                if dur <= EPS:
                    break
                self.melody.append({'note': 'R', 'midi': None, 'start': self.mel_t,
                                    'dur': dur, 'source': 'train'})
            self.mel_t = round(self.mel_t + dur, 6)

    def train_chord(self, rc: dict, t: float, dur: float) -> None:
        state, instr, criteria = self.chord_query(t, dur)
        label = next((k for k, v in self.candidates.items()
                      if v['chord'] == rc['chord']), None)
        if label is not None:
            self.jev.record('acorde', state, instr, criteria, label)
            c = self.candidates[label]
            chord = {'chord': c['chord'], 'numeral': c['numeral'], 'pitches': c['pitches']}
        else:                                  # acorde no diatónico: no era una opción
            self.jev.counts['omitida:acorde'] += 1
            chord = {'chord': rc['chord'], 'numeral': rc.get('numeral', ''),
                     'pitches': rc['pitches']}
        self.chords.append(dict(chord, start=t, dur=dur, source='train'))

    def run(self) -> None:
        if self.fixed:                         # estilo fixed: solo la melodía
            self.place_seed()
            for c in list(self.chords):
                self.follow_melody(c['start'] + c['dur'])
            return
        t = self.place_seed()
        self.follow_melody(t)
        for rc in self.data.chords:
            end = min(rc['start'] + rc['dur'], self.total)
            if end <= t + EPS:
                continue
            start = max(rc['start'], t)
            self.train_chord(rc, start, end - start)
            t = end
            self.follow_melody(t)


def build_train_data(path: str, args, ct, teacher) -> types.SimpleNamespace:
    """Analiza un MIDI (+ su .txt) y lo deja listo para recorrerlo."""
    notes, tpb, midi_beats = read_midi_notes(path)
    mel_notes, harm_notes = split_melody_harmony(notes, tpb, args.melody_track)
    if not harm_notes:
        fail(f"{path}: no queda armonía tras separar la melodía")
    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, 'harmony.mid')
        write_harmony_midi(harm_notes, tpb, hp)
        progression, inferred = detect_chords(hp, args, ct, None)
    melody = quantize_melody(mel_notes, tpb, args.quant)
    if not melody:
        fail(f"{path}: no se detectó melodía")

    stem = os.path.splitext(os.path.basename(path))[0]
    desc, txt_key, txt_mode, note_lines = read_train_txt(
        os.path.splitext(path)[0] + '.txt')
    if desc is None:
        desc = args.desc or stem.replace('_', ' ')
        warn(f"{stem}: sin 'descripción:' en su .txt, se usa «{desc}»")

    key_text = txt_key or args.key
    key_pc, key_hint = parse_key(key_text) if key_text else (None, None)
    tonic = key_pc if key_pc is not None else inferred
    mode = txt_mode or args.mode or key_hint
    if mode is not None and mode not in MODES:
        fail(f"{stem}: modo no válido '{mode}' (opciones: {', '.join(MODES)})")

    beats = args.beats or midi_beats
    end = max(sum(c['duration_beats'] for c in progression),
              max(n['start'] + n['dur'] for n in melody))
    bars = max(1, int(math.ceil(end / beats - EPS)))
    total = float(bars * beats)
    fixed = fit_progression(progression, total)
    chords, t = [], 0.0
    for c in fixed:
        chords.append(dict(c, start=round(t, 6), dur=c['duration_beats']))
        t += c['duration_beats']
    melody = [dict(n, dur=min(n['dur'], total - n['start']))
              for n in melody if n['start'] < total - EPS]
    if mode is None:
        mode = infer_mode(fixed, melody, tonic)
    octave = args.octave
    if octave is None:                         # centrar el rango [base-5, base+19]
        med = statistics.median(n['midi'] for n in melody)
        octave = max(2, min(7, round((med - 7 - tonic) / 12)))
    return types.SimpleNamespace(
        path=path, stem=stem, desc=desc, tonic=tonic, mode=mode, inferred=inferred,
        key_forced=key_pc is not None, beats=beats, bars=bars, total=total,
        octave=octave, fixed_progression=fixed, chords=chords, melody=melody,
        bar_notes=parse_bar_note_lines(note_lines, bars))


def run_train_pass(teacher, data, base_args, style: str, depth: int) -> None:
    a = copy.copy(base_args)
    a.bars, a.beats, a.depth, a.octave = data.bars, data.beats, depth, data.octave
    a.rng_obj, a.seed_temp, a.temperature, a.verbose = random.Random(0), 0.0, 0.0, False
    a.bar_notes_map = data.bar_notes
    a.mode = None
    if style == 'fixed':
        a.chords_midi = data.path
        a.key = PITCH_NAMES[data.tonic] if data.key_forced else None
        fixed = data.fixed_progression
    else:
        a.chords_midi, a.key, fixed = None, None, None
    teacher.data, teacher.args_tools = data, a
    with contextlib.redirect_stdout(io.StringIO()):       # silenciar los print de componer
        tonic, mode = choose_key_mode(teacher, data.desc, a, a.rng_obj, fixed, data.inferred)
        seeds = choose_seeds(teacher, data.desc, tonic, mode, a, fixed)
    TrainComposer(teacher, data, seeds, a).run()


def cmd_train(args) -> None:
    if not MIDO_OK:
        fail("train necesita mido (pip install mido)")
    if not os.path.isdir(args.dir):
        fail(f"No es un directorio: {args.dir}")
    files = sorted(os.path.join(args.dir, f) for f in os.listdir(args.dir)
                   if f.lower().endswith(('.mid', '.midi')))
    if not files:
        fail(f"no hay .mid/.midi en {args.dir}")
    try:
        depths = sorted({int(x) for x in args.depth.split(',') if x.strip()})
    except ValueError:
        fail(f"--depth no válido: '{args.depth}' (ej. 4 ó 2,4,6)")
    if not depths or depths[0] < 1:
        fail("--depth debe ser >= 1")
    styles = ['generate', 'fixed'] if args.style == 'both' else [args.style]
    args.cache, args.cache_only, args.verbose = True, False, False
    ct = load_chord_table_module(args)
    teacher = TeacherJev(args)

    banner('JEV COMPOSER — Train')
    print(f"  {COL['gray']}cache: {teacher.path}  ({len(teacher.cache)} entradas){COL['reset']}\n")
    ok = bad = 0
    for path in files:
        before = Counter(teacher.counts)
        try:
            data = build_train_data(path, args, ct, teacher)
            for depth in depths:
                for style in styles:
                    run_train_pass(teacher, data, args, style, depth)
        except SystemExit:
            bad += 1
            print(f"  {COL['red']}✗ {os.path.basename(path)}{COL['reset']} (omitido)\n")
            continue
        ok += 1
        d = teacher.counts - before
        omit = ', '.join(f"{k.split(':')[1]} {v}" for k, v in sorted(d.items())
                         if k.startswith('omitida:'))
        print(f"  {COL['green']}✓{COL['reset']} {COL['bold']}{os.path.basename(path)}{COL['reset']}  "
              f"{PITCH_NAMES[data.tonic]} {data.mode} · {data.bars} compases "
              f"({data.beats}/4) · octava {data.octave}")
        print(f"      {COL['gray']}+{d['nueva']} nuevas, {d['existente']} ya existentes"
              + (f", omitidas: {omit}" if omit else '') + COL['reset'] + '\n')
    tot = teacher.counts
    kinds = ', '.join(f"{k.split(':')[1]} {v}" for k, v in sorted(tot.items())
                      if k.startswith('nueva:'))
    print(f"  {len(files)} MIDI(s): {ok} ok, {bad} omitidos")
    print(f"  {COL['gray']}{tot['nueva']} entradas nuevas ({kinds}), "
          f"{tot['existente']} ya existentes; cache: {teacher.path}{COL['reset']}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# SUBCOMANDOS
# ═══════════════════════════════════════════════════════════════════════════════

def read_description(args) -> str:
    if args.file:
        if not os.path.isfile(args.file):
            fail(f"No se encuentra el archivo: {args.file}")
        with open(args.file, encoding='utf-8') as f:
            desc = f.read()
    else:
        desc = args.desc or ''
    if not desc.strip():
        fail("falta la descripción de la obra (texto entre comillas o --file)")
    return desc


def banner(title: str) -> None:
    print(f"\n{COL['cyan']}{'═' * 72}{COL['reset']}")
    print(f"{COL['bold']}  {title}{COL['reset']}")
    print(f"{COL['cyan']}{'═' * 72}{COL['reset']}\n")


def setup(args):
    args.rng_obj = random.Random(args.rng)
    if args.cache_only and not args.cache:
        args.cache = True
    return Jev(args)


def cmd_tonalidad(args) -> None:
    desc = read_description(args)
    jev = setup(args)
    banner('JEV COMPOSER — Tonalidad')
    tonic, mode = choose_key_mode(jev, desc, args, args.rng_obj)
    print(f"\n  → {COL['yellow']}{PITCH_NAMES[tonic]} {mode}{COL['reset']}\n")
    print(f"  {COL['gray']}{jev.summary()}{COL['reset']}\n")


def cmd_semillas(args) -> None:
    desc = read_description(args)
    jev = setup(args)
    banner('JEV COMPOSER — Semillas')
    tonic, mode = choose_key_mode(jev, desc, args, args.rng_obj)
    seeds = choose_seeds(jev, desc, tonic, mode, args)
    if args.export_json:
        out = {'tonic': PITCH_NAMES[tonic], 'mode': mode, 'octave': args.octave,
               'seeds': seeds_json(seeds), 'queries': jev.log,
               'generator': 'jev_composer.py v4.0'}
        with open(_ensure_parent(args.export_json), 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  → JSON:  {args.export_json}")
    print(f"  {COL['gray']}{jev.summary()}{COL['reset']}\n")


def cmd_componer(args) -> None:
    desc = read_description(args)
    if args.beats < 1 or (args.bars is not None and args.bars < 1):
        fail("--bars y --beats deben ser >= 1")
    fixed, inferred = None, None
    if args.chords_midi:
        fixed, inferred = load_chords_midi(args)
        midi_beats = sum(c['duration_beats'] for c in fixed)
        max_bars = max(1, int(math.ceil(midi_beats / args.beats - EPS)))
        if args.bars is None:
            args.bars = max_bars
        elif args.bars > max_bars:
            warn(f"--bars {args.bars} supera la duración del MIDI ({midi_beats:g} "
                 f"pulsos): se recorta a {max_bars} compases")
            args.bars = max_bars
        fixed = fit_progression(fixed, float(args.bars * args.beats))
    elif args.bars is None:
        args.bars = 16
    args.bar_notes_map = parse_bar_notes(args)
    jev = setup(args)
    banner('JEV COMPOSER — Composición')
    if args.bar_notes_map:
        print(f"  {COL['gray']}indicaciones en {len(args.bar_notes_map)} compás(es): "
              f"{', '.join(map(str, sorted(args.bar_notes_map)))}{COL['reset']}\n")
    tonic, mode = choose_key_mode(jev, desc, args, args.rng_obj, fixed, inferred)
    seeds = choose_seeds(jev, desc, tonic, mode, args, fixed)

    print(f"  {COL['gray']}{'c.':<5} {'origen':<8} {'acorde':<7}{'dur':>4}  melodía{COL['reset']}")
    print(f"  {'─' * 68}")
    comp = Composer(jev, desc, tonic, mode, seeds, args)
    comp.compose()

    chords, melody = comp.chords, comp.melody
    chord_text, mel_text = progression_text(chords), melody_text(melody)
    json_path = f"{args.output}.jev.json"
    txt_c     = f"{args.output}.chords.txt"
    txt_m     = f"{args.output}.melody.txt"
    mid_path  = f"{args.output}.mid"

    out = {
        'meta': {'description': desc.strip(), 'tonic': PITCH_NAMES[tonic], 'mode': mode,
                 'bars': args.bars, 'beats': args.beats, 'tempo': args.tempo,
                 'depth': args.depth, 'temperature': args.temperature,
                 'seed_temp': args.seed_temp, 'rng': args.rng, 'octave': args.octave,
                 'leap': args.leap, 'model': args.model,
                 'chords_midi': args.chords_midi,
                 'bar_notes': {str(b): v for b, v in sorted(args.bar_notes_map.items())}},
        'seeds':        seeds_json(seeds),
        'chord_string': chord_text,
        'melody_string': mel_text,
        'chords':       chords,
        'melody':       melody,
        'queries':      jev.log,
        'generator':    'jev_composer.py v4.0',
    }
    print()
    with open(_ensure_parent(json_path), 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  → JSON:    {json_path}")
    with open(_ensure_parent(txt_c), 'w', encoding='utf-8') as f:
        f.write(chord_text + '\n')
    print(f"  → Acordes: {txt_c}")
    with open(_ensure_parent(txt_m), 'w', encoding='utf-8') as f:
        f.write(mel_text + '\n')
    print(f"  → Melodía: {txt_m}")
    if MIDO_OK:
        piece_to_midi(chords, melody, 480, args.tempo, mid_path)
        print(f"  → MIDI:    {mid_path}")
    else:
        print("  [AVISO] mido no instalado — MIDI omitido. pip install mido")
    print(f"\n  {COL['gray']}{jev.summary()}{COL['reset']}\n")


def cmd_cache(args) -> None:
    path = os.path.expanduser(args.cache_file)
    if args.clear:
        if os.path.isfile(path):
            os.remove(path)
            print(f"\n  Cache vaciada: {path}\n")
        else:
            print(f"\n  No hay cache en {path}\n")
        return
    banner('JEV COMPOSER — Cache')
    if not os.path.isfile(path):
        print(f"  Sin cache en {path}\n")
        return
    recs = {}                                   # la última línea de cada clave gana
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            recs[rec.get('key')] = rec
    n = len(recs)
    kinds = Counter(r.get('kind', '?') for r in recs.values())
    models = Counter(r.get('model', '?') for r in recs.values())
    complete = sum(1 for r in recs.values() if r.get('state') and r.get('criteria'))
    sources = Counter(r.get('source') or infer_source(r.get('answer') or {})
                      for r in recs.values())
    print(f"  Fichero:    {path}")
    print(f"  Tamaño:     {os.path.getsize(path) / 1024:.1f} KiB")
    print(f"  Consultas:  {n}")
    print(f"  Completas:  {complete} de {n} (con estado y opciones; las demás se "
          f"completan al reutilizarlas con --cache)")
    print(f"  Origen:     " + ', '.join(f"{k} {v}" for k, v in sources.most_common()) + "\n")
    print(f"  {COL['bold']}Por tipo:{COL['reset']}")
    for k, v in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {k:<18} {v:>5}")
    print(f"\n  {COL['bold']}Por modelo:{COL['reset']}")
    for k, v in sorted(models.items(), key=lambda x: -x[1]):
        print(f"    {k:<28} {v:>5}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def seed_opts_train():
    """Opciones de train (parser padre)."""
    t = argparse.ArgumentParser(add_help=False)
    t.add_argument('dir', metavar='DIR', help='Directorio con los .mid/.midi (y sus .txt)')
    t.add_argument('--model', type=str, default=DEFAULT_MODEL,
                   help=f'Modelo con el que se firman las entradas (default: {DEFAULT_MODEL})')
    t.add_argument('--url', type=str, default=DEFAULT_URL, help=argparse.SUPPRESS)
    t.add_argument('--cache-file', type=str, default=DEFAULT_CACHE, dest='cache_file',
                   metavar='FILE', help=f'Fichero de cache JSONL (default: {DEFAULT_CACHE})')
    t.add_argument('--overwrite', action='store_true',
                   help='Reescribir entradas ya existentes en la cache')
    t.add_argument('--depth', type=str, default='4', metavar='N[,N...]',
                   help='Profundidades de ventana a reconstruir (default: 4)')
    t.add_argument('--octave', type=int, default=None,
                   help='Octava base (default: automática por obra)')
    t.add_argument('--leap', type=int, default=9,
                   help='Salto melódico máximo en semitonos (default: 9)')
    t.add_argument('--beats', type=int, default=None,
                   help='Pulsos por compás (default: el del MIDI, o 4)')
    t.add_argument('--quant', type=float, default=0.25, metavar='BEATS',
                   help='Rejilla de cuantización de la melodía (default: 0.25)')
    t.add_argument('--melody-track', type=int, default=None, dest='melody_track',
                   metavar='N', help='Forzar la pista (0-based) con la melodía')
    t.add_argument('--style', choices=['generate', 'fixed', 'both'], default='both',
                   help='Entradas del modo normal, del modo --chords-midi, o ambas')
    t.add_argument('--desc', type=str, default=None, metavar='TEXTO',
                   help="Descripción para los MIDIs sin 'descripción:' en su .txt")
    t.add_argument('--key', type=str, help='Fijar la tónica de todos los MIDIs')
    t.add_argument('--mode', type=str, choices=MODES, help='Fijar el modo de todos los MIDIs')
    t.add_argument('--bass-only', action='store_true', dest='bass_only',
                   help='Detectar acordes solo con las notas graves')
    t.add_argument('--min-dur', type=float, default=0.5, dest='min_dur', metavar='BEATS',
                   help='Fundir con el anterior los acordes más cortos que esto (default: 0.5)')
    t.add_argument('--chord-table', type=str, dest='chord_table',
                   default=os.path.join(HERE, 'chord_table.py'), metavar='FILE',
                   help='Ruta de chord_table.py (default: junto a este script)')
    t.add_argument('--motif-table', type=str, dest='motif_table',
                   default=os.path.join(HERE, 'motif_table.py'), metavar='FILE',
                   help='Ruta de motif_table.py (default: junto a este script)')
    return t


def build_parser():
    parser = argparse.ArgumentParser(
        description='Composición guiada por Jev (TypeSafe) con semillas de '
                    'chord_table.py y motif_table.py.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='cmd', metavar='SUBCOMANDO')

    # Opciones de Jev compartidas
    jev_opts = argparse.ArgumentParser(add_help=False)
    jev_opts.add_argument('desc', nargs='?', metavar='DESC',
                          help='Descripción de la obra (o --file)')
    jev_opts.add_argument('--file', type=str, metavar='FILE',
                          help='Leer la descripción de un fichero')
    jev_opts.add_argument('--model', type=str, default=DEFAULT_MODEL,
                          help=f'Modelo de Jev (default: {DEFAULT_MODEL})')
    jev_opts.add_argument('--url', type=str, default=DEFAULT_URL,
                          help='Endpoint de decisiones de OpenRouter')
    jev_opts.add_argument('--cache', action='store_true',
                          help='Guardar y reutilizar las consultas en la cache local')
    jev_opts.add_argument('--cache-only', action='store_true', dest='cache_only',
                          help='Solo leer de la cache: nunca llama a la API')
    jev_opts.add_argument('--cache-file', type=str, default=DEFAULT_CACHE,
                          dest='cache_file', metavar='FILE',
                          help=f'Fichero de cache JSONL (default: {DEFAULT_CACHE})')
    jev_opts.add_argument('--key', type=str, help="Fijar la tónica ('Am' implica menor)")
    jev_opts.add_argument('--mode', type=str, choices=MODES, help='Fijar el modo')
    jev_opts.add_argument('--seed-temp', type=float, default=0.0, dest='seed_temp',
                          metavar='T',
                          help='Temperatura para elegir modo/tónica/semillas '
                               '(default: 0 = la más probable)')
    jev_opts.add_argument('--rng', type=int, default=None, metavar='N',
                          help='Semilla del generador aleatorio')
    jev_opts.add_argument('--verbose', action='store_true',
                          help='Mostrar las probabilidades de cada consulta')

    # Opciones de semillas / componer
    seed_opts = argparse.ArgumentParser(add_help=False)
    seed_opts.add_argument('--octave', type=int, default=5,
                           help='Octava base de la tónica (default: 5)')
    seed_opts.add_argument('--chord-table', type=str, dest='chord_table',
                           default=os.path.join(HERE, 'chord_table.py'), metavar='FILE',
                           help='Ruta de chord_table.py (default: junto a este script)')
    seed_opts.add_argument('--motif-table', type=str, dest='motif_table',
                           default=os.path.join(HERE, 'motif_table.py'), metavar='FILE',
                           help='Ruta de motif_table.py (default: junto a este script)')

    p = sub.add_parser('tonalidad', parents=[jev_opts],
                       help='Elegir modo y tónica de la obra',
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(func=cmd_tonalidad)

    p = sub.add_parser('semillas', parents=[jev_opts, seed_opts],
                       help='Elegir progresión y motivo semilla',
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--export-json', type=str, dest='export_json', metavar='FILE',
                   help='Exportar las semillas elegidas a JSON')
    p.set_defaults(func=cmd_semillas)

    p = sub.add_parser('componer', parents=[jev_opts, seed_opts],
                       help='Pipeline completo: tonalidad + semillas + ventana',
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--bars', type=int, default=None,
                   help='Compases a generar (default: 16; con --chords-midi, los '
                        'del MIDI)')
    p.add_argument('--beats', type=int, default=4, help='Pulsos por compás (default: 4)')
    p.add_argument('--tempo', type=float, default=100.0,
                   help='Tempo BPM para exportación MIDI (default: 100)')
    p.add_argument('--depth', type=int, default=4,
                   help='Profundidad de la ventana en acordes (default: 4)')
    p.add_argument('--temperature', type=float, default=1.0, metavar='T',
                   help='Temperatura del sampling (default: 1.0; 0 = la más probable)')
    p.add_argument('--leap', type=int, default=9,
                   help='Salto melódico máximo en semitonos (default: 9)')
    p.add_argument('--output', type=str, default='obra',
                   help='Nombre base para salidas (default: obra)')
    p.add_argument('--notes', type=str, action='append', metavar='TEXTO',
                   help="Indicaciones por compás: 'compás 29: texto; 30-32: texto' "
                        "(repetible)")
    p.add_argument('--notes-file', type=str, dest='notes_file', metavar='FILE',
                   help='Leer indicaciones por compás de un fichero (una por línea)')
    p.add_argument('--chords-midi', type=str, dest='chords_midi', metavar='FILE',
                   help='Usar los acordes de este MIDI en vez de generarlos')
    p.add_argument('--bass-only', action='store_true', dest='bass_only',
                   help='(con --chords-midi) detectar acordes solo con notas graves')
    p.add_argument('--min-dur', type=float, default=0.5, dest='min_dur', metavar='BEATS',
                   help='(con --chords-midi) fundir con el anterior los acordes más '
                        'cortos que esto (default: 0.5)')
    p.set_defaults(func=cmd_componer)

    p = sub.add_parser('train', parents=[seed_opts_train()],
                       help='Reconstruir la cache a partir de MIDIs',
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser('cache', help='Estadísticas de la cache de consultas',
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache-file', type=str, default=DEFAULT_CACHE, dest='cache_file',
                   metavar='FILE', help=f'Fichero de cache JSONL (default: {DEFAULT_CACHE})')
    p.add_argument('--clear', action='store_true', help='Vaciar la cache')
    p.set_defaults(func=cmd_cache)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, 'func', None):
        parser.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()
