#!/usr/bin/env python3
r"""
jev_composer.py — Jev Composer: composición guiada por Jev (TypeSafe) a partir
de una descripción en texto, con semillas de chord_table.py y motif_table.py
(versión 1.0).

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
Subcomandos
────────────────────────────────────────────────────────────────────────────
  tonalidad DESC     Elegir modo y tónica de la obra
  semillas  DESC     Elegir progresión y motivo semilla (adaptados al modo)
  componer  DESC     Pipeline completo: tonalidad + semillas + ventana
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
  --chords-midi FILE  Usar los acordes de este MIDI en vez de generarlos
  --bass-only         (con --chords-midi) detectar acordes solo con las notas
                      graves (≤ C4), como chord_table.py
  --min-dur BEATS     (con --chords-midi) ignorar acordes más cortos que esto

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
        self.cache      = load_cache(self.path) if self.use_cache else {}
        self.stats      = Counter()
        self.log        = []

    # ── API ──────────────────────────────────────────────────────────────────
    def ask(self, kind: str, state: str, instructions: str, criteria: dict):
        if len(criteria) > MAX_OPTIONS:
            fail(f"la pregunta '{kind}' tiene {len(criteria)} opciones (máx. {MAX_OPTIONS})")
        if len(criteria) == 1:                       # nada que preguntar
            only = next(iter(criteria))
            return only, {only: 1.0}

        ckey = hashlib.sha256(json.dumps(
            {'model': self.model, 'state': state, 'instructions': instructions,
             'criteria': criteria}, sort_keys=True, ensure_ascii=False
        ).encode('utf-8')).hexdigest()

        ans = self.cache.get(ckey) if self.use_cache else None
        cached = ans is not None
        if cached:
            self.stats['cache'] += 1
        else:
            if self.cache_only:
                fail(f"--cache-only: consulta '{kind}' no encontrada en {self.path}. "
                     f"Repite con --cache (y los mismos parámetros y --rng) para "
                     f"rellenarla.")
            ans = self._request(state, instructions, criteria)
            self.stats['api'] += 1
            if self.use_cache:
                self._store(ckey, kind, instructions, ans)

        choice = ans['choice']
        if choice not in criteria:
            fail(f"Jev devolvió una opción que no estaba en la lista: {choice!r}")
        probs = _normalize_probs(ans, criteria)
        if len(probs) == 1:
            self.stats['sin_probs'] += 1
        self.log.append({'kind': kind, 'choice': choice,
                         'p': round(probs.get(choice, 1.0), 4), 'cached': cached})
        return choice, probs

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
            'User-Agent': 'jev-composer/1.0',
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

    def _store(self, ckey, kind, instructions, ans):
        self.cache[ckey] = ans
        with open(_ensure_parent(self.path), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'key': ckey, 'kind': kind, 'model': self.model,
                                'instructions': instructions[:160], 'answer': ans},
                               ensure_ascii=False) + '\n')

    def summary(self) -> str:
        api, hit = self.stats['api'], self.stats['cache']
        s = f"{api + hit} consulta(s): {api} a la API, {hit} desde la cache"
        if self.stats['sin_probs']:
            s += (f"  [AVISO] {self.stats['sin_probs']} respuesta(s) sin "
                  f"probabilidades: en ellas el sampling es determinista")
        return s


def load_cache(path: str) -> dict:
    cache = {}
    if not os.path.isfile(path):
        return cache
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[rec['key']] = rec['answer']
            except (ValueError, KeyError):
                warn(f"{path}:{n}: línea de cache ilegible, se ignora")
    return cache


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
# MODO ACORDES FIJOS: LECTURA DE LOS ACORDES DE UN MIDI
# ═══════════════════════════════════════════════════════════════════════════════

def load_chords_midi(args) -> tuple:
    """Lee los acordes de un MIDI con el detector de chord_table.py (importado
    como módulo: ningún comando suyo devuelve la secuencia detectada).
    Devuelve (progresión, tónica_inferida) con la progresión contigua desde el
    pulso 0: cada acorde dura hasta el inicio del siguiente."""
    path = args.chords_midi
    if not os.path.isfile(path):
        fail(f"No se encuentra el archivo: {path}")
    if not os.path.isfile(args.chord_table):
        fail(f"no se encuentra {args.chord_table} (usa --chord-table)")
    spec = importlib.util.spec_from_file_location('chord_table_mod', args.chord_table)
    ct = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct)

    try:
        events, tpb = ct.midi_to_chord_sequence(path, bass_only=args.bass_only)
        recognized = ct.chord_sequence_to_numerals(events, tpb)
    except Exception as e:                      # MIDI corrupto o no estándar
        fail(f"no se pudo leer {path}: {e}")
    recognized = [c for c in recognized
                  if c['root_pc'] is not None and c['dur_beats'] >= args.min_dur]
    if not recognized:
        fail(f"no se detectó ningún acorde en {path}")
    recognized.sort(key=lambda c: c['start'])

    inferred = ct.infer_tonic(recognized)
    tonic = parse_key(args.key)[0] if args.key else inferred
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

    def state(self) -> str:
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
                f"melody as note(beats)):\n" + '\n'.join(lines) + '\n' + upcoming)

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
    def next_chord(self, t: float) -> dict:
        dur = min(self.chord_durs[len(self.chords) % len(self.chord_durs)],
                  self.total - t)
        criteria = {k: v['desc'] for k, v in self.candidates.items()}
        choice, probs = self.jev.ask(
            'acorde', self.state(),
            f"Which chord should come next? It starts at {self.pos(t)} and lasts "
            f"{fmt(dur)} beat(s). Choose the single best fit for the description, "
            f"the seeds and the harmonic flow.", criteria)
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
                cands = self.pitch_candidates(c)
                choice, probs = self.jev.ask(
                    'nota', self.state(),
                    f"Which note should the melody play next? It starts at "
                    f"{self.pos(self.mel_t)}, lasts {fmt(dur)} beat(s) and sounds over "
                    f"the chord {c['chord']}. Choose the single best fit for the "
                    f"description, the motif seed and the harmony.",
                    {k: v[1] for k, v in cands.items()})
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
              f"{COL['cyan']}{ns}{COL['reset']}")

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
               'generator': 'jev_composer.py v1.0'}
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
    jev = setup(args)
    banner('JEV COMPOSER — Composición')
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
                 'chords_midi': args.chords_midi},
        'seeds':        seeds_json(seeds),
        'chord_string': chord_text,
        'melody_string': mel_text,
        'chords':       chords,
        'melody':       melody,
        'queries':      jev.log,
        'generator':    'jev_composer.py v1.0',
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
    kinds, models, n = Counter(), Counter(), 0
    seen = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get('key') in seen:
                continue
            seen.add(rec.get('key'))
            n += 1
            kinds[rec.get('kind', '?')] += 1
            models[rec.get('model', '?')] += 1
    print(f"  Fichero:    {path}")
    print(f"  Tamaño:     {os.path.getsize(path) / 1024:.1f} KiB")
    print(f"  Consultas:  {n}\n")
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
    p.add_argument('--chords-midi', type=str, dest='chords_midi', metavar='FILE',
                   help='Usar los acordes de este MIDI en vez de generarlos')
    p.add_argument('--bass-only', action='store_true', dest='bass_only',
                   help='(con --chords-midi) detectar acordes solo con notas graves')
    p.add_argument('--min-dur', type=float, default=0.0, dest='min_dur', metavar='BEATS',
                   help='(con --chords-midi) ignorar acordes más cortos que esto')
    p.set_defaults(func=cmd_componer)

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
