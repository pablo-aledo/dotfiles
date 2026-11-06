#!/usr/bin/env python3
r"""
jev_local.py — Jev Local: estimador local de probabilidades entrenado con la
cache de jev_composer (sustituto futuro de Jev/TypeSafe)
(versión 0.1: modelo de dos torres con rasgos hasheados, solo numpy).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
  jev_composer.py guarda en su cache cada consulta hecha a Jev con todos
  los datos para reconstruirla (estado, instrucciones, opciones y la
  distribución de probabilidad devuelta). Este script aprende de esas
  entradas a reproducir esa distribución sin llamar a la API.

  Modelo: para cada consulta (estado S, instrucciones I, opciones o_1..o_n con
  su descripción) estima p(o_i | S, I) = softmax_i( a_i + <e_Q, e_Oi> / τ ):
    · torre de la consulta  e_Q : suma de embeddings de rasgos del estado y de
      las instrucciones (palabras de la descripción de la obra, tonalidad,
      últimos acordes y notas relativos a la tónica, posición en el compás,
      indicaciones por compás, acorde en curso...).
    · torre de la opción    e_Oi: suma de embeddings de rasgos de la opción
      (etiqueta, palabras de su descripción, grado y calidad del acorde,
      intervalo respecto a la nota anterior, función de la nota en el acorde).
    · a_i: término lineal de la opción. τ: temperatura de calibración, ajustada
      al final del entrenamiento con el conjunto de validación.
  Los rasgos se identifican por hash (crc32) en tablas de 2^bits filas, así que
  no hay vocabulario que mantener. Entrenamiento por Adagrad sobre filas
  dispersas, con gradientes calculados a mano: no hace falta PyTorch.
  El objetivo es la distribución de Jev (entradas source=api) o el valor real
  de un MIDI (source=train, 1.0/0.0); se minimiza la entropía cruzada.

  Es un aproximador estadístico: aprende lo que está en los datos. Con pocas
  consultas o con descripciones de obra muy distintas de las vistas, no
  generaliza como un LLM. 'evaluate' mide cuánto se acerca a Jev.

  COMPATIBILIDAD CON JEV: la entrada de 'compute' y 'serve' es el cuerpo que
  jev_composer envía a Jev y la salida es la misma respuesta:
      {"answers": {"<pregunta>": {"choice": "...", "probabilities": {...}}}}
  'serve' abre un servidor HTTP con ese protocolo, de modo que el compositor
  lo usa sin cambios, solo apuntando --url a él.

────────────────────────────────────────────────────────────────────────────
Modos de uso (subcomandos)
────────────────────────────────────────────────────────────────────────────
  prepare     Cache de jev_composer → dataset de rasgos hasheados (.npz).
  train       Entrena el modelo y lo calibra (checkpoints, parada anticipada).
  compute     Calcula las probabilidades de una consulta (formato Jev).
  serve       Servidor HTTP con el protocolo de Jev (--url del compositor).
  evaluate    Compara el modelo con las respuestas guardadas, por tipo.
  inspect     Resumen del dataset, del modelo y de las curvas de entrenamiento.

────────────────────────────────────────────────────────────────────────────
Entradas y salidas
────────────────────────────────────────────────────────────────────────────
  prepare    Entrada: --cache-file (JSONL de jev_composer; las entradas sin
             estado/opciones, de versiones anteriores, se omiten). Salida en
             --data-dir: dataset.npz y dataset_meta.json. Separa validación por
             descripción de obra (nunca la misma obra en train y validación).
  train      Entrada: --data-dir. Salida en --model-dir: model_config.json,
             best_model.npz, checkpoint.npz y train_history.json.
  compute    Entrada: cuerpo JSON de una consulta a Jev (--input o stdin), o una
             entrada de cache (--cache-file y --key). Salida: la respuesta de
             Jev en JSON (stdout o --output).
  serve      Escucha en --host:--port. POST con el cuerpo de Jev → respuesta de
             Jev. GET → estado del servidor.

  Uso con jev_composer (el modelo local sustituye a la API):
      export OPENROUTER_API_KEY=local            # cualquier valor
      python jev_local.py serve --model-dir model/ &
      python jev_composer.py componer "..." --model local-v1 \
          --url http://127.0.0.1:8808/api/alpha/decisions
  Usa un --model distinto del de Jev: así la cache no mezcla respuestas del
  modelo local con las de Jev. Para reentrenar sin ellas, prepare las filtra
  con --model-filter.

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
Una línea por parámetro: <nombre> <tipo> [clave=valor …] [# descripción].
Tipos: int float str bool enum file dir list kv. Claves: default min max typical
unit values of io ext format required advanced when excludes alias. Sin min/max =
sin límite conocido. Se verifica contra el argparse con «suite_index.py lint
--check spec-vs-code».

@params v1
[prepare]
--cache-file   file  default=~/.cache/jev_composer/queries.jsonl io=in ext=jsonl  # Cache JSONL de jev_composer
--data-dir     dir   io=out required  # Directorio de salida del dataset
--model-filter str   # Usar solo entradas hechas con este modelo (default: todas)
--sources      list  of=enum values=api|train  # Orígenes a incluir (default: api train)
--val-frac     float default=0.1 min=0 max=0.5  # Fracción de obras para validación
--buckets-bits int   default=18 min=10 max=24  # Filas por tabla de rasgos: 2^bits
--seed         int   default=42  # Semilla del reparto train/validación
--report       bool  default=false  # Estadísticas detalladas del dataset
[train]
--data-dir       dir   io=in required  # Directorio con el dataset de prepare
--model-dir      dir   io=out required  # Directorio de salida del modelo
--epochs         int   default=50 min=0 unit=épocas  # Épocas máximas
--batch-size     int   default=32 min=1  # Consultas por paso de optimización
--dim            int   default=32 min=1  # Dimensión de los embeddings de las torres
--lr             float default=0.05  # Tasa de aprendizaje de Adagrad
--l2             float default=1e-6  # Regularización L2 de las filas actualizadas
--patience       int   default=8 min=0 unit=épocas  # Parada anticipada: épocas sin mejora
--one-hot-weight float default=1.0  # Peso de las entradas source=train frente a source=api
--no-calibrate   bool  default=false  # No ajustar la temperatura final con validación
--seed           int   default=42  # Semilla de inicialización y orden de los datos
--resume         bool  default=false  # Reanudar desde el último checkpoint
--report         bool  default=false  # Mostrar la curva de pérdida al finalizar
[compute]
--model-dir   dir   io=in required  # Directorio con el modelo entrenado
--input       file  default=- io=in ext=json  # Cuerpo JSON de la consulta ('-' = stdin)
--cache-file  file  io=in ext=jsonl  # Calcular una entrada de cache (con --key)
--key         str   # Clave (o prefijo) de la entrada de cache
--output      file  io=out ext=json  # Respuesta JSON (default: stdout)
--temperature float default=1.0 typical=0.5..2  # Multiplica la temperatura calibrada
--kind        enum  values=modo|tonica|semilla_acordes|semilla_motivo|acorde|nota|otro  # Forzar el tipo de pregunta
--pretty      bool  default=false  # Resumen legible por stderr
[serve]
--model-dir   dir   io=in required  # Directorio con el modelo entrenado
--host        str   default=127.0.0.1  # Dirección de escucha
--port        int   default=8808 min=1 max=65535  # Puerto
--temperature float default=1.0 typical=0.5..2  # Multiplica la temperatura calibrada
--token       str   # Exigir 'Authorization: Bearer <token>'
--verbose     bool  default=false  # Registrar cada petición por stderr
[evaluate]
--model-dir   dir   io=in required  # Directorio con el modelo entrenado
--data-dir    dir   io=in required  # Directorio con el dataset de prepare
--split       enum  default=val values=val|train|all  # Subconjunto a evaluar
--temperature float default=1.0 typical=0.5..2  # Multiplica la temperatura calibrada
[inspect]
--model-dir   dir   io=in  # Modelo entrenado
--data-dir    dir   io=in  # Dataset preparado
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  # 1. Preparar el dataset a partir de la cache (solo respuestas de Jev y de train)
  python jev_local.py prepare --data-dir data/ --model-filter typesafe/jev-1.14 --report

  # 2. Entrenar y calibrar
  python jev_local.py train --data-dir data/ --model-dir model/ --epochs 60 --report

  # 3. Medir cuánto se parece a Jev en las obras de validación
  python jev_local.py evaluate --model-dir model/ --data-dir data/

  # 4. Calcular una consulta (formato Jev)
  python jev_local.py compute --model-dir model/ --input consulta.json --pretty

  # 5. Recalcular una entrada de la cache y compararla con la respuesta guardada
  python jev_local.py compute --model-dir model/ --cache-file ~/.cache/jev_composer/queries.jsonl \
        --key 3fa9c2 --pretty

  # 6. Servir el modelo con el protocolo de Jev
  python jev_local.py serve --model-dir model/ --port 8808
"""

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
import zlib
from collections import Counter
from pathlib import Path

try:
    import numpy as np
except ImportError:                                  # pragma: no cover
    sys.exit("[ERROR] jev_local.py necesita numpy (pip install numpy)")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

KINDS   = ['modo', 'tonica', 'semilla_acordes', 'semilla_motivo', 'acorde', 'nota', 'otro']
SOURCES = ['api', 'train']

PITCH_NAMES      = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTE_PC = {n: i for i, n in enumerate(PITCH_NAMES)}
NOTE_PC.update({n: i for i, n in enumerate(PITCH_NAMES_FLAT)})

DEFAULT_CACHE = os.path.join('~', '.cache', 'jev_composer', 'queries.jsonl')
DATASET_NAME  = 'dataset.npz'
META_NAME     = 'dataset_meta.json'
CONFIG_NAME   = 'model_config.json'
BEST_NAME     = 'best_model.npz'
CKPT_NAME     = 'checkpoint.npz'
HISTORY_NAME  = 'train_history.json'
QUESTION_KEY  = 'pick'          # nombre de la pregunta que usa jev_composer

MAX_DESC_WORDS = 120            # palabras de la descripción de la obra que se usan
MAX_OPT_WORDS  = 60             # palabras de la descripción de cada opción


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _fmt_time(seconds: float) -> str:
    """Formatea segundos como string legible: 1h 23m 45s / 4m 12s / 38s."""
    s = int(seconds)
    if s >= 3600:
        return f"{s//3600}h {(s%3600)//60}m {s%60:02d}s"
    if s >= 60:
        return f"{s//60}m {s%60:02d}s"
    return f"{s}s"


def _fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE LA CONSULTA (estado e instrucciones de jev_composer)
# ══════════════════════════════════════════════════════════════════════════════

_KIND_PATTERNS = [
    ('modo',            r'Which musical mode'),
    ('tonica',          r'Which tonic note'),
    ('semilla_acordes', r'Which chord progression'),
    ('semilla_motivo',  r'Which melodic motif'),
    ('acorde',          r'Which chord should come next'),
    ('nota',            r'Which note should the melody play next'),
]


def infer_kind(instructions: str) -> str:
    """Tipo de pregunta a partir del texto de las instrucciones."""
    for kind, pat in _KIND_PATTERNS:
        if re.search(pat, instructions):
            return kind
    return 'otro'


def words(text: str) -> list:
    """Palabras en minúsculas, sin acentos (≥ 2 caracteres alfanuméricos)."""
    t = unicodedata.normalize('NFKD', text.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]{2,}', t)


def split_chord(name: str):
    """'C#M7' → (1, 'M7'); None si no se reconoce."""
    m = re.match(r'^([A-G](?:#|b)?)(.*)$', name)
    if not m or m.group(1) not in NOTE_PC:
        return None
    return NOTE_PC[m.group(1)], m.group(2)


def parse_note(name: str):
    """'E4' → midi 64; None si no es una nota (p. ej. 'R')."""
    m = re.match(r'^([A-G]#?)(-?\d+)$', name)
    if not m:
        return None
    return 12 * (int(m.group(2)) + 1) + NOTE_PC[m.group(1)]


def rel_chord(name: str, tonic) -> str:
    sc = split_chord(name)
    if sc is None or tonic is None:
        return name
    return f"{(sc[0] - tonic) % 12}_{sc[1]}"


_SECTION_RE = re.compile(
    r'^(MODE|KEY|TONIC|METER|CHORD SEED|CHORDS \(fixed\)|CHORDS|MOTIF SEED|'
    r'UPCOMING CHORDS):\s*(.*)$')
_WINDOW_RE = re.compile(r'^bar (\d+) beat ([\d.]+) \| (\S+) \(([\d.]+) beats\): (.*)$')


def parse_state(state: str) -> dict:
    """Descompone el estado que jev_composer envía a Jev."""
    st = {'desc': '', 'mode': None, 'tonic': None, 'beats': None, 'bars': None,
          'chord_seed': [], 'motif': [], 'window': [], 'upcoming': [],
          'extra': '', 'fixed': False}
    section, desc, extra = None, [], []
    for line in state.split('\n'):
        if line.startswith('PIECE DESCRIPTION:'):
            section = 'desc'
            continue
        if section != 'extra':
            if line.startswith('MUSIC SO FAR'):
                section = 'window'
                continue
            if line.startswith('EXTRA INFORMATION FOR THIS POSITION:'):
                section = 'extra'
                continue
            m = _SECTION_RE.match(line)
            if m:
                section, val = 'field', m.group(2)
                label = m.group(1)
                if label == 'MODE':
                    st['mode'] = val.strip() or None
                elif label == 'KEY':
                    mk = re.match(r'^([A-G](?:#|b)?) (\w+)', val)
                    if mk:
                        st['tonic'], st['mode'] = NOTE_PC[mk.group(1)], mk.group(2)
                elif label == 'TONIC':
                    if val.strip() in NOTE_PC:
                        st['tonic'] = NOTE_PC[val.strip()]
                elif label == 'METER':
                    mm = re.match(r'(\d+) beats per bar, (\d+) bars', val)
                    if mm:
                        st['beats'], st['bars'] = int(mm.group(1)), int(mm.group(2))
                elif label in ('CHORD SEED', 'CHORDS (fixed)'):
                    st['chord_seed'] = [(a, float(b)) for a, b in
                                        re.findall(r'(\S+?):([\d.]+)(?=\s|$)', val)]
                    st['fixed'] = label != 'CHORD SEED'
                elif label == 'CHORDS':
                    st['fixed'] = val.startswith('fixed')
                elif label == 'MOTIF SEED':
                    st['motif'] = [(a, float(b)) for a, b in
                                   re.findall(r'(\S+?):([\d.]+)(?=\s|$)', val)]
                elif label == 'UPCOMING CHORDS':
                    st['upcoming'] = [(a, float(b)) for a, b in
                                      re.findall(r'(\S+?)\(([\d.]+)\)', val)]
                continue
        if section == 'desc':
            desc.append(line)
        elif section == 'window':
            m = _WINDOW_RE.match(line)
            if m:
                notes = [(a, float(b)) for a, b in re.findall(r'(\S+?)\(([\d.]+)\)', m.group(5))]
                st['window'].append({'bar': int(m.group(1)), 'beat': float(m.group(2)),
                                     'chord': m.group(3), 'dur': float(m.group(4)),
                                     'notes': notes})
        elif section == 'extra':
            extra.append(line)
    st['desc'] = '\n'.join(desc).strip()
    st['extra'] = '\n'.join(extra).strip()
    return st


def parse_instr(instructions: str) -> dict:
    """Posición, duración y acorde que mencionan las instrucciones."""
    out = {'bar': None, 'beat': None, 'dur': None, 'chord': None}
    m = re.search(r'bar (\d+) beat ([\d.]+)', instructions)
    if m:
        out['bar'], out['beat'] = int(m.group(1)), float(m.group(2))
    m = re.search(r'lasts ([\d.]+) beat', instructions)
    if m:
        out['dur'] = float(m.group(1))
    m = re.search(r'over the chord (\S+?)\. Choose', instructions)
    if m:
        out['chord'] = m.group(1)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  RASGOS
# ══════════════════════════════════════════════════════════════════════════════

def _dedupe(items: list) -> list:
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _last_pitched(st: dict):
    """Última nota con altura (midi) de la ventana, o None."""
    for c in reversed(st['window']):
        for name, _ in reversed(c['notes']):
            m = parse_note(name)
            if m is not None:
                return m
    return None


def query_tokens(kind: str, st: dict, ins: dict) -> list:
    """Rasgos de la consulta (estado + instrucciones). Los relativos a la tónica
    permiten generalizar entre tonalidades."""
    T = []

    def add(tok, kp=True):
        T.append(tok)
        if kp:
            T.append(f"{kind}/{tok}")

    tonic = st['tonic']
    rel = (lambda n: rel_chord(n, tonic))
    T.append(f"k={kind}")

    dw = words(st['desc'])[:MAX_DESC_WORDS]
    for w in dw:
        add(f"w:{w}")
    T.extend(f"b:{a}_{b}" for a, b in zip(dw, dw[1:]))
    xw = words(st['extra'])
    if xw:
        add('hasx')
    for w in xw:
        add(f"x:{w}")

    if st['mode']:
        add(f"mode={st['mode']}")
    if tonic is not None:
        add(f"tonic={PITCH_NAMES[tonic]}")
    if st['beats']:
        add(f"meter={st['beats']}")
    if ins['bar'] and st['bars']:
        add(f"pos={min(7, int(8 * (ins['bar'] - 1) / st['bars']))}")
        add(f"toend={min(st['bars'] - ins['bar'], 4)}")
    if ins['beat'] is not None:
        add(f"beat={ins['beat']:g}")
    if ins['dur'] is not None:
        add(f"dur={ins['dur']:g}")

    for name, _ in st['chord_seed'][:8]:
        add(f"cs:{rel(name)}", kp=False)
    for name, _ in st['motif'][:12]:
        m = parse_note(name)
        if m is not None and tonic is not None:
            add(f"ms:{(m - tonic) % 12}", kp=False)

    chords = st['window']
    for i, c in enumerate(reversed(chords[-4:])):
        add(f"c-{i+1}:{rel(c['chord'])}")
    if len(chords) >= 2:
        add(f"c-12:{rel(chords[-2]['chord'])}>{rel(chords[-1]['chord'])}")

    pitched = []
    for c in chords:
        for name, dur in c['notes']:
            m = parse_note(name)
            if m is not None:
                pitched.append((m, dur))
    for i, (m, dur) in enumerate(reversed(pitched[-4:])):
        if tonic is not None:
            add(f"n-{i+1}:{(m - tonic) % 12}")
        add(f"d-{i+1}:{dur:g}")
    if len(pitched) >= 2:
        add(f"i-1:{max(-12, min(12, pitched[-1][0] - pitched[-2][0]))}")
    if pitched:
        add(f"reg:{pitched[-1][0] // 6}")

    if ins['chord']:
        add(f"cur:{rel(ins['chord'])}")
    for i, (name, _) in enumerate(st['upcoming'][:2]):
        add(f"up{i+1}:{rel(name)}")
    return _dedupe(T)


def option_tokens(kind: str, label: str, desc: str, st: dict, ins: dict, prev_midi) -> list:
    """Rasgos de una opción (etiqueta, descripción y, según el tipo, su relación
    con la tónica, el acorde en curso y la nota anterior)."""
    O = [f"o:{kind}|{label}"]

    def add(tok, kp=True):
        O.append(tok)
        if kp:
            O.append(f"{kind}/{tok}")

    tonic = st['tonic']
    for w in words(desc.split('; notes ')[0])[:MAX_OPT_WORDS]:
        add(f"ow:{w}")

    if kind == 'acorde':
        m = re.match(r'^(\S+) \((\S+)\)$', label)
        if m:
            add(f"num:{m.group(1)}")
            sc = split_chord(m.group(2))
            if sc:
                root, q = sc
                add(f"q:{q}")
                if tonic is not None:
                    add(f"rel:{(root - tonic) % 12}")
                    add(f"rq:{(root - tonic) % 12}_{q}")
                if st['window']:
                    last = st['window'][-1]['chord']
                    if last == m.group(2):
                        add('repeat')
                    lsc = split_chord(last)
                    if lsc:
                        add(f"rm:{(root - lsc[0]) % 12}")
                        add(f"rm_q:{(root - lsc[0]) % 12}_{lsc[1]}>{q}")
    elif kind == 'nota':
        mm = parse_note(label)
        if mm is not None:
            add(f"pitch:{mm}")
            add(f"reg:{mm // 6}")
            if tonic is not None:
                add(f"rp:{(mm - tonic) % 12}")
            if prev_midi is not None:
                d = mm - prev_midi
                add(f"mv:{max(-12, min(12, d))}")
                add(f"mva:{min(abs(d), 12)}")
                add('dir:' + ('up' if d > 0 else 'down' if d < 0 else 'same'))
            sc = split_chord(ins['chord']) if ins['chord'] else None
            if sc:
                add(f"cr:{(mm - sc[0]) % 12}")
                add(f"crq:{(mm - sc[0]) % 12}_{sc[1]}")
            if 'chord tone' in desc:
                add('ct')
            elif 'passing' in desc:
                add('pass')
            elif 'chromatic' in desc:
                add('chrom')
            if 'the tonic' in desc:
                add('istonic')
    elif kind in ('semilla_acordes', 'semilla_motivo'):
        m = re.search(r'tension (\d+)/10', desc)
        if m:
            add(f"ten:{m.group(1)}")
    return _dedupe(O)


def _hash(tok: str, mask: int) -> int:
    return zlib.crc32(tok.encode('utf-8')) & mask


def featurize(kind: str, state: str, instructions: str, criteria: dict, bits: int):
    """(índices de la consulta, índices de opciones concatenados, punteros por
    opción). Es la única función de rasgos: prepare y compute la comparten."""
    st, ins = parse_state(state), parse_instr(instructions)
    mask = (1 << bits) - 1
    qi = np.array([_hash(t, mask) for t in query_tokens(kind, st, ins)], dtype=np.int32)
    prev = _last_pitched(st)
    oi, optr = [], [0]
    for label, desc in criteria.items():
        oi.extend(_hash(t, mask) for t in option_tokens(kind, label, desc, st, ins, prev))
        optr.append(len(oi))
    return qi, np.array(oi, dtype=np.int32), np.array(optr, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO (dos torres con embeddings hasheados; gradientes a mano)
# ══════════════════════════════════════════════════════════════════════════════

def softmax(s):
    s = s - np.max(s)
    e = np.exp(s)
    return e / e.sum()


class TwoTower:
    """s_i = a_i + <e_Q, e_Oi>.  e_Q = Σ EQ[q]/√|Q|;  e_Oi = Σ EO[o]/√|O_i|;
    a_i = Σ w[o]/√|O_i|.  Entrenamiento: Adagrad sobre las filas tocadas."""

    PARAMS = ('EQ', 'EO', 'w')

    def __init__(self, bits: int, dim: int, seed: int = 0, init: bool = True):
        n = 1 << bits
        self.bits, self.dim = bits, dim
        if init:
            rng = np.random.default_rng(seed)
            self.EQ = (rng.standard_normal((n, dim)) * 0.1).astype(np.float32)
            self.EO = (rng.standard_normal((n, dim)) * 0.1).astype(np.float32)
            self.w = np.zeros(n, dtype=np.float32)
        self.G = None                               # acumuladores de Adagrad

    def init_optimizer(self):
        self.G = {k: np.full_like(getattr(self, k), 0.1) for k in self.PARAMS}

    # ── paso hacia delante ───────────────────────────────────────────────────
    def forward(self, qi, oi, optr):
        cq = 1.0 / math.sqrt(len(qi))
        eQ = self.EQ[qi].sum(axis=0) * cq
        starts, lens = optr[:-1], np.diff(optr)
        sq = np.sqrt(lens).astype(np.float32)
        eO = np.add.reduceat(self.EO[oi], starts, axis=0) / sq[:, None]
        a = np.add.reduceat(self.w[oi], starts) / sq
        return a + eO @ eQ, (eQ, eO, cq, lens, sq)

    # ── paso hacia atrás: acumula gradientes de las filas tocadas ────────────
    def backward(self, g, qi, oi, ctx, acc):
        eQ, eO, cq, lens, sq = ctx
        gA = g / sq
        seg = np.repeat(np.arange(len(lens)), lens)
        acc['w'].append((oi, gA[seg]))
        acc['EO'].append((oi, gA[seg][:, None] * eQ[None, :]))
        gq = (g[:, None] * eO).sum(axis=0) * cq
        acc['EQ'].append((qi, np.broadcast_to(gq, (len(qi), self.dim))))

    def apply(self, acc, lr: float, l2: float):
        for name in self.PARAMS:
            if not acc[name]:
                continue
            idx = np.concatenate([a for a, _ in acc[name]])
            gr = np.concatenate([b for _, b in acc[name]]).astype(np.float32)
            uniq, inv = np.unique(idx, return_inverse=True)
            tot = np.zeros((len(uniq),) + gr.shape[1:], dtype=np.float32)
            np.add.at(tot, inv, gr)
            par, G = getattr(self, name), self.G[name]
            if l2:
                tot += l2 * par[uniq]
            G[uniq] += tot * tot
            par[uniq] -= lr * tot / np.sqrt(G[uniq])

    # ── disco ────────────────────────────────────────────────────────────────
    def save(self, path: Path, with_optimizer: bool = False, extra: dict = None):
        data = {k: getattr(self, k) for k in self.PARAMS}
        if with_optimizer and self.G:
            data.update({f"G_{k}": v for k, v in self.G.items()})
        for k, v in (extra or {}).items():
            data[k] = np.asarray(v)
        np.savez(_ensure_parent(str(path)), **data)

    @classmethod
    def load(cls, path: Path, bits: int, dim: int):
        z = np.load(str(path), allow_pickle=False)
        m = cls(bits, dim, init=False)
        for k in cls.PARAMS:
            setattr(m, k, z[k])
        if all(f"G_{k}" in z for k in cls.PARAMS):
            m.G = {k: z[f"G_{k}"] for k in cls.PARAMS}
        return m, z


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════════════════════

def infer_source(ans: dict) -> str:
    """Origen de una entrada sin campo 'source' (igual criterio que jev_composer)."""
    vals = list((ans.get('probabilities') or {}).values())
    return 'train' if len(vals) > 1 and sorted(set(vals)) == [0.0, 1.0] \
        and vals.count(1.0) == 1 else 'api'


def read_cache(path: str) -> dict:
    """{clave: registro}; si una clave se repite vale la última línea."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        _fail(f"no se encuentra la cache: {path}")
    recs = {}
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                recs[r['key']] = r
            except (ValueError, KeyError):
                print(f"[prepare] {path}:{n}: línea ilegible, se ignora", file=sys.stderr)
    return recs


def target_vector(criteria: dict, answer: dict):
    """Distribución de Jev normalizada sobre las opciones (las ausentes valen 0)."""
    probs = answer.get('probabilities') or {}
    t = np.array([max(0.0, float(probs.get(k, 0.0))) for k in criteria], dtype=np.float64)
    if t.sum() <= 0:
        if answer.get('choice') not in criteria:
            return None
        t = np.array([1.0 if k == answer['choice'] else 0.0 for k in criteria])
    return (t / t.sum()).astype(np.float32)


class Dataset:
    """Consultas con rasgos hasheados en arrays planos (formato de dataset.npz)."""

    def __init__(self, path: Path):
        z = np.load(str(path), allow_pickle=False)
        for k in ('q_ptr', 'q_tok', 'o_ptr', 'ot_ptr', 'o_tok', 'target',
                  'kind', 'source', 'group', 'split', 'keys'):
            setattr(self, k, z[k])
        self.n = len(self.kind)

    def query(self, i: int):
        qi = self.q_tok[self.q_ptr[i]:self.q_ptr[i + 1]]
        a, b = self.o_ptr[i], self.o_ptr[i + 1]
        oi = self.o_tok[self.ot_ptr[a]:self.ot_ptr[b]]
        optr = self.ot_ptr[a:b + 1] - self.ot_ptr[a]
        return qi, oi, optr, self.target[a:b]


# ══════════════════════════════════════════════════════════════════════════════
#  PREPARE
# ══════════════════════════════════════════════════════════════════════════════

def cmd_prepare(args):
    """Cache de jev_composer → dataset de rasgos hasheados."""
    recs = read_cache(args.cache_file)
    sources = args.sources or SOURCES
    bits = args.buckets_bits
    print(f"[prepare] {len(recs)} entradas distintas en {args.cache_file}")

    q_tok, o_tok, targets = [], [], []
    q_ptr, o_ptr, ot_ptr = [0], [0], [0]
    kinds, srcs, groups, keys = [], [], [], []
    group_ids, skipped = {}, Counter()
    kind_mismatch = 0

    for key, r in recs.items():
        if not (r.get('state') and r.get('criteria')):
            skipped['incompleta (sin estado/opciones)'] += 1
            continue
        if args.model_filter and r.get('model') != args.model_filter:
            skipped['otro modelo'] += 1
            continue
        ans = r.get('answer') or {}
        source = r.get('source') or infer_source(ans)
        if source not in sources:
            skipped[f'origen {source} excluido'] += 1
            continue
        criteria = r['criteria']
        if len(criteria) < 2:
            skipped['una sola opción'] += 1
            continue
        t = target_vector(criteria, ans)
        if t is None:
            skipped['respuesta no válida'] += 1
            continue
        kind = infer_kind(r['instructions'])
        if r.get('kind') and r['kind'] != kind:
            kind_mismatch += 1
        qi, oi, optr = featurize(kind, r['state'], r['instructions'], criteria, bits)

        q_tok.append(qi)
        o_tok.append(oi)
        targets.append(t)
        q_ptr.append(q_ptr[-1] + len(qi))
        o_ptr.append(o_ptr[-1] + len(criteria))
        ot_ptr.extend((ot_ptr[-1] + optr[1:]).tolist())
        kinds.append(KINDS.index(kind))
        srcs.append(SOURCES.index(source))
        desc = parse_state(r['state'])['desc']
        groups.append(group_ids.setdefault(desc, len(group_ids)))
        keys.append(key)

    n = len(kinds)
    if n == 0:
        _fail("no hay entradas utilizables (¿cache de una versión anterior, sin "
              "estado/opciones? Vuelve a ejecutar las composiciones con --cache)")
    for why, c in skipped.most_common():
        print(f"[prepare] omitidas: {c:>6}  {why}")
    if kind_mismatch:
        print(f"[prepare] AVISO: {kind_mismatch} entradas cuyo 'kind' guardado no coincide "
              f"con el inferido de las instrucciones (se usa el inferido)")

    # ── reparto train/validación por descripción de obra ─────────────────────
    rng = np.random.default_rng(args.seed)
    split = np.zeros(n, dtype=np.int8)
    groups_arr = np.array(groups, dtype=np.int32)
    n_groups = len(group_ids)
    if args.val_frac > 0:
        if n_groups >= 3:
            n_val_g = max(1, int(round(args.val_frac * n_groups)))
            val_groups = set(rng.permutation(n_groups)[:n_val_g].tolist())
            split = np.array([1 if g in val_groups else 0 for g in groups], dtype=np.int8)
            how = f"por obra ({n_val_g} de {n_groups} descripciones)"
        else:
            split = (rng.random(n) < args.val_frac).astype(np.int8)
            how = f"por consulta (solo {n_groups} descripción(es) distinta(s): la validación no es independiente)"
        print(f"[prepare] validación {how}")
    if args.val_frac > 0 and split.sum() == 0:
        split[rng.integers(n)] = 1

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _ensure_parent(str(Path(args.data_dir) / DATASET_NAME)),
        q_ptr=np.array(q_ptr, dtype=np.int64), q_tok=np.concatenate(q_tok),
        o_ptr=np.array(o_ptr, dtype=np.int64), ot_ptr=np.array(ot_ptr, dtype=np.int64),
        o_tok=np.concatenate(o_tok), target=np.concatenate(targets),
        kind=np.array(kinds, dtype=np.int8), source=np.array(srcs, dtype=np.int8),
        group=groups_arr, split=split, keys=np.array(keys))
    meta = {
        'cache_file': os.path.expanduser(args.cache_file), 'bits': bits,
        'n_queries': n, 'n_options': int(o_ptr[-1]), 'n_groups': n_groups,
        'n_train': int((split == 0).sum()), 'n_val': int((split == 1).sum()),
        'by_kind': dict(Counter(KINDS[k] for k in kinds)),
        'by_source': dict(Counter(SOURCES[s] for s in srcs)),
        'model_filter': args.model_filter, 'sources': sources,
        'skipped': dict(skipped), 'seed': args.seed, 'val_frac': args.val_frac,
    }
    with open(_ensure_parent(str(Path(args.data_dir) / META_NAME)), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"[prepare] {n} consultas ({meta['n_train']} train / {meta['n_val']} val), "
          f"{meta['n_options']} opciones → {args.data_dir}/{DATASET_NAME}")
    if args.report:
        _print_dataset_meta(meta, np.diff(np.array(q_ptr)), np.diff(np.array(o_ptr)))


def _print_dataset_meta(meta: dict, q_lens=None, opt_counts=None):
    print(f"\n  Consultas: {meta['n_queries']}  (train {meta['n_train']} / val {meta['n_val']})"
          f"  Obras distintas: {meta['n_groups']}  Opciones: {meta['n_options']}")
    print("  Por tipo:   " + ', '.join(f"{k} {v}" for k, v in sorted(meta['by_kind'].items())))
    print("  Por origen: " + ', '.join(f"{k} {v}" for k, v in sorted(meta['by_source'].items())))
    if q_lens is not None and len(q_lens):
        print(f"  Rasgos por consulta: media {q_lens.mean():.0f}, máx {q_lens.max()}"
              f"   Opciones por consulta: media {opt_counts.mean():.1f}, máx {opt_counts.max()}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICAS Y CALIBRACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _entropy(t):
    nz = t[t > 0]
    return float(-(nz * np.log(nz)).sum())


def query_metrics(s, t, tau: float = 1.0) -> dict:
    """KL(t‖p), entropía cruzada, acierto del argmax y masa en el argmax de t."""
    p = softmax(s / tau)
    logp = np.log(np.maximum(p, 1e-12))
    ce = float(-(t * logp).sum())
    kl = ce - _entropy(t)
    top = int(np.argmax(t))
    uni_kl = float(-(t * np.log(1.0 / len(t))).sum()) - _entropy(t)
    return {'ce': ce, 'kl': kl, 'top1': float(np.argmax(p) == top),
            'ptop': float(p[top]), 'uni_kl': uni_kl}


def fit_temperature(logits: list, targets: list) -> float:
    """Temperatura global τ que minimiza la entropía cruzada de validación."""
    if not logits:
        return 1.0
    best_tau, best_ce = 1.0, float('inf')
    for tau in np.exp(np.linspace(math.log(0.2), math.log(5.0), 81)):
        ce = 0.0
        for s, t in zip(logits, targets):
            z = s / tau
            z = z - z.max()
            logp = z - math.log(np.exp(z).sum())
            ce += float(-(t * logp).sum())
        if ce < best_ce:
            best_tau, best_ce = float(tau), ce
    return best_tau


def evaluate_split(model: TwoTower, ds: Dataset, idx, tau: float = 1.0):
    """Métricas por consulta sobre el subconjunto idx (sin gradientes)."""
    rows = []
    for i in idx:
        qi, oi, optr, t = ds.query(i)
        s, _ = model.forward(qi, oi, optr)
        m = query_metrics(s, t, tau)
        m['kind'], m['source'] = KINDS[ds.kind[i]], SOURCES[ds.source[i]]
        rows.append(m)
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def _batch_weights(ds: Dataset, one_hot_weight: float):
    return np.where(ds.source == SOURCES.index('train'), one_hot_weight, 1.0).astype(np.float32)


def cmd_train(args):
    """Entrena el modelo de dos torres sobre el dataset de prepare."""
    data_dir, model_dir = Path(args.data_dir), Path(args.model_dir)
    if not (data_dir / DATASET_NAME).exists():
        _fail(f"no se encuentra {data_dir / DATASET_NAME}: ¿has ejecutado prepare?")
    with open(data_dir / META_NAME, encoding='utf-8') as f:
        meta = json.load(f)
    ds = Dataset(data_dir / DATASET_NAME)
    train_idx = np.flatnonzero(ds.split == 0)
    val_idx = np.flatnonzero(ds.split == 1)
    if len(train_idx) == 0:
        _fail("el dataset no tiene consultas de entrenamiento")
    print(f"[train] {ds.n} consultas: train {len(train_idx)}  val {len(val_idx)}  "
          f"(bits={meta['bits']}, dim={args.dim})")
    if len(val_idx) == 0:
        print("[train] AVISO: sin validación; la parada anticipada usa la pérdida de train")

    model_dir.mkdir(parents=True, exist_ok=True)
    weights = _batch_weights(ds, args.one_hot_weight)
    model = TwoTower(meta['bits'], args.dim, args.seed)
    model.init_optimizer()
    history, best_val, bad, start = [], float('inf'), 0, 0
    config = {'bits': meta['bits'], 'dim': args.dim, 'kinds': KINDS,
              'calibration_temperature': 1.0, 'trained_on': meta['by_kind'],
              'sources': meta['sources'], 'model_filter': meta['model_filter'],
              'one_hot_weight': args.one_hot_weight, 'lr': args.lr, 'l2': args.l2}
    with open(model_dir / CONFIG_NAME, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    if args.resume:
        ck = model_dir / CKPT_NAME
        if not ck.exists():
            _fail(f"--resume: no existe {ck}")
        model, z = TwoTower.load(ck, meta['bits'], args.dim)
        if model.G is None:
            model.init_optimizer()
        start, best_val, bad = int(z['epoch']), float(z['best_val']), int(z['bad'])
        hp = model_dir / HISTORY_NAME
        history = json.load(open(hp, encoding='utf-8')) if hp.exists() else []
        print(f"[train] Reanudando desde la época {start} (mejor val {best_val:.4f})")

    rng = np.random.default_rng(args.seed + start)
    t_start = time.time()
    for epoch in range(start, args.epochs):
        t0 = time.time()
        order = rng.permutation(train_idx)
        tot, wsum = 0.0, 0.0
        for b in range(0, len(order), args.batch_size):
            batch = order[b:b + args.batch_size]
            acc = {k: [] for k in TwoTower.PARAMS}
            for i in batch:
                qi, oi, optr, t = ds.query(i)
                s, ctx = model.forward(qi, oi, optr)
                p = softmax(s)
                w = float(weights[i])
                tot += -w * float((t * np.log(np.maximum(p, 1e-12))).sum())
                wsum += w
                model.backward((p - t) * (w / len(batch)), qi, oi, ctx, acc)
            model.apply(acc, args.lr, args.l2)

        train_ce = tot / max(wsum, 1e-9)
        if len(val_idx):
            rows = evaluate_split(model, ds, val_idx)
            val_kl = float(np.mean([r['kl'] for r in rows]))
            val_acc = float(np.mean([r['top1'] for r in rows]))
        else:
            val_kl, val_acc = train_ce, float('nan')
        improved = val_kl < best_val - 1e-5
        if improved:
            best_val, bad = val_kl, 0
            model.save(model_dir / BEST_NAME)
        else:
            bad += 1
        history.append({'epoch': epoch + 1, 'train_ce': train_ce, 'val_kl': val_kl,
                        'val_top1': val_acc, 'seconds': time.time() - t0})
        model.save(model_dir / CKPT_NAME, with_optimizer=True,
                   extra={'epoch': epoch + 1, 'best_val': best_val, 'bad': bad})
        with open(model_dir / HISTORY_NAME, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
        print(f"[train] época {epoch+1:3d}/{args.epochs}  train_ce {train_ce:.4f}  "
              f"val_KL {val_kl:.4f}  val_acc {val_acc:.3f}  "
              f"{'*' if improved else ' '} {time.time()-t0:.1f}s")
        if args.patience and bad >= args.patience:
            print(f"[train] Parada anticipada: {bad} épocas sin mejora")
            break

    if not (model_dir / BEST_NAME).exists():
        model.save(model_dir / BEST_NAME)
    best, _ = TwoTower.load(model_dir / BEST_NAME, meta['bits'], args.dim)

    if len(val_idx) and not args.no_calibrate:
        logits, targets = [], []
        for i in val_idx:
            qi, oi, optr, t = ds.query(i)
            logits.append(best.forward(qi, oi, optr)[0])
            targets.append(t)
        tau = fit_temperature(logits, targets)
        config['calibration_temperature'] = tau
        print(f"[train] Temperatura calibrada: τ = {tau:.3f}")
    config['best_val_kl'] = best_val
    config['epochs_run'] = len(history)
    with open(model_dir / CONFIG_NAME, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    print(f"[train] Hecho en {_fmt_time(time.time() - t_start)}. Mejor val_KL {best_val:.4f} "
          f"→ {model_dir}/{BEST_NAME}")
    if args.report:
        _train_report(history)


def _train_report(history: list):
    if not history:
        return
    vals = [h['val_kl'] for h in history]
    lo, hi = min(vals), max(vals)
    print("\n  época  val_KL")
    for h in history:
        bar = int(40 * (h['val_kl'] - lo) / (hi - lo)) if hi > lo else 0
        print(f"  {h['epoch']:5d}  {h['val_kl']:.4f}  {'█' * (bar + 1)}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DEL MODELO Y CÁLCULO (formato Jev)
# ══════════════════════════════════════════════════════════════════════════════

class LocalEstimator:
    """Estimador cargado desde --model-dir. estimate() devuelve lo mismo que una
    pregunta de Jev: {'choice': ..., 'probabilities': {opción: prob}}."""

    def __init__(self, model_dir: str, temperature: float = 1.0):
        d = Path(model_dir)
        if not (d / CONFIG_NAME).exists() or not (d / BEST_NAME).exists():
            _fail(f"no hay modelo en {d}: ¿has ejecutado train?")
        with open(d / CONFIG_NAME, encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.model, _ = TwoTower.load(d / BEST_NAME, self.cfg['bits'], self.cfg['dim'])
        self.tau = float(self.cfg.get('calibration_temperature', 1.0)) * max(temperature, 1e-6)

    def estimate(self, state: str, instructions: str, criteria: dict, kind: str = None) -> dict:
        if not isinstance(criteria, dict) or not criteria:
            raise ValueError("'criteria' debe ser un objeto {opción: descripción} no vacío")
        kind = kind or infer_kind(instructions)
        if len(criteria) == 1:
            only = next(iter(criteria))
            return {'choice': only, 'probabilities': {only: 1.0}}
        qi, oi, optr = featurize(kind, state, instructions, criteria, self.cfg['bits'])
        s, _ = self.model.forward(qi, oi, optr)
        p = softmax(s / self.tau)
        labels = list(criteria)
        return {'choice': labels[int(np.argmax(p))],
                'probabilities': {k: round(float(v), 6) for k, v in zip(labels, p)}}

    def answer_request(self, body: dict, kind: str = None) -> dict:
        """Cuerpo de una consulta a Jev → respuesta de Jev."""
        if not isinstance(body, dict) or not isinstance(body.get('questions'), dict):
            raise ValueError("el cuerpo debe tener 'questions' (objeto de preguntas)")
        state = body.get('state', '')
        answers = {}
        for name, q in body['questions'].items():
            if q.get('type', 'choice') != 'choice':
                raise ValueError(f"pregunta '{name}': solo se admite type='choice'")
            answers[name] = self.estimate(state, q.get('instructions', ''),
                                          q.get('criteria'), kind)
        return {'answers': answers}


def _pretty(resp: dict, stored: dict = None):
    for name, a in resp['answers'].items():
        print(f"\n  pregunta '{name}': {a['choice']}", file=sys.stderr)
        ref = (stored or {}).get('probabilities') or {}
        for k, v in sorted(a['probabilities'].items(), key=lambda kv: -kv[1])[:8]:
            extra = f"   Jev: {ref.get(k, 0.0):.3f}" if stored else ''
            print(f"    {v:6.3f}  {k[:60]}{extra}", file=sys.stderr)
        if stored:
            print(f"    (elección guardada: {stored.get('choice')})", file=sys.stderr)


def cmd_compute(args):
    """Probabilidades de una consulta, en el formato de respuesta de Jev."""
    est = LocalEstimator(args.model_dir, args.temperature)
    stored = None
    if args.key:
        if not args.cache_file:
            _fail("--key necesita --cache-file")
        recs = read_cache(args.cache_file)
        hits = [r for k, r in recs.items() if k.startswith(args.key)]
        if len(hits) != 1:
            _fail(f"la clave '{args.key}' coincide con {len(hits)} entradas")
        r = hits[0]
        if not (r.get('state') and r.get('criteria')):
            _fail("esa entrada es antigua (sin estado/opciones); complétala con "
                  "jev_composer --cache")
        body = {'model': r.get('model'), 'state': r['state'],
                'questions': {QUESTION_KEY: {'type': 'choice',
                                             'instructions': r['instructions'],
                                             'criteria': r['criteria']}}}
        stored = r.get('answer')
    else:
        try:
            raw = sys.stdin.read() if args.input == '-' else open(args.input, encoding='utf-8').read()
            body = json.loads(raw)
        except (OSError, ValueError) as e:
            _fail(f"no se pudo leer la consulta: {e}")
    try:
        resp = est.answer_request(body, args.kind)
    except ValueError as e:
        _fail(str(e))
    text = json.dumps(resp, ensure_ascii=False, indent=2)
    if args.output:
        with open(_ensure_parent(args.output), 'w', encoding='utf-8') as f:
            f.write(text + '\n')
    else:
        print(text)
    if args.pretty:
        _pretty(resp, stored)


# ══════════════════════════════════════════════════════════════════════════════
#  SERVE
# ══════════════════════════════════════════════════════════════════════════════

def cmd_serve(args):
    """Servidor HTTP con el protocolo de Jev: POST {model,state,questions} →
    {answers}. Cualquier ruta acepta la petición."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    est = LocalEstimator(args.model_dir, args.temperature)

    class Handler(BaseHTTPRequestHandler):
        server_version = 'JevLocal/0.1'

        def _send(self, code, payload):
            data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {'status': 'ok', 'model': 'jev-local',
                             'bits': est.cfg['bits'], 'dim': est.cfg['dim'],
                             'tau': est.tau})

        def do_POST(self):
            if args.token and self.headers.get('Authorization') != f"Bearer {args.token}":
                return self._send(401, {'error': 'token no válido'})
            try:
                n = int(self.headers.get('Content-Length', 0))
                body = json.loads(self.rfile.read(n))
                self._send(200, est.answer_request(body))
            except ValueError as e:
                self._send(400, {'error': str(e)})
            except Exception as e:                      # nunca tumbar el servidor
                self._send(500, {'error': f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *a):
            if args.verbose:
                sys.stderr.write(f"[serve] {self.address_string()} {fmt % a}\n")

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[serve] Jev Local escuchando en http://{args.host}:{args.port}/api/alpha/decisions"
          f"  (τ={est.tau:.3f}). Ctrl+C para parar.", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] Parado.", file=sys.stderr)
    finally:
        srv.server_close()


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATE / INSPECT
# ══════════════════════════════════════════════════════════════════════════════

def cmd_evaluate(args):
    """Distancia entre el modelo y las respuestas guardadas, por tipo y origen."""
    est = LocalEstimator(args.model_dir, args.temperature)
    ds_dir = Path(args.data_dir)
    if not (ds_dir / DATASET_NAME).exists():
        _fail(f"no se encuentra {ds_dir / DATASET_NAME}: ¿has ejecutado prepare?")
    meta = json.load(open(ds_dir / META_NAME, encoding='utf-8'))
    if meta['bits'] != est.cfg['bits']:
        _fail(f"el dataset usa bits={meta['bits']} y el modelo bits={est.cfg['bits']}: "
              f"vuelve a ejecutar prepare con --buckets-bits {est.cfg['bits']}")
    ds = Dataset(ds_dir / DATASET_NAME)
    sel = {'val': ds.split == 1, 'train': ds.split == 0, 'all': np.ones(ds.n, bool)}[args.split]
    idx = np.flatnonzero(sel)
    if len(idx) == 0:
        _fail(f"el subconjunto '{args.split}' está vacío")
    rows = evaluate_split(est.model, ds, idx, est.tau)
    print(f"\n[evaluate] {len(rows)} consultas ({args.split}), τ={est.tau:.3f}\n")
    hdr = f"  {'tipo':<16}{'origen':<7}{'n':>6}{'KL modelo':>11}{'KL uniforme':>13}{'acierto':>9}{'p(argmax)':>11}"
    print(hdr)
    print('  ' + '─' * (len(hdr) - 2))
    groups = sorted({(r['kind'], r['source']) for r in rows}, key=lambda x: (KINDS.index(x[0]), x[1]))
    for key in groups + [('TOTAL', '')]:
        sub = rows if key[0] == 'TOTAL' else [r for r in rows if (r['kind'], r['source']) == key]
        print(f"  {key[0]:<16}{key[1]:<7}{len(sub):>6}"
              f"{np.mean([r['kl'] for r in sub]):>11.4f}{np.mean([r['uni_kl'] for r in sub]):>13.4f}"
              f"{np.mean([r['top1'] for r in sub]):>9.3f}{np.mean([r['ptop'] for r in sub]):>11.3f}")
    print("\n  KL: divergencia respecto a la distribución guardada (0 = idéntica).")
    print("  acierto: el argmax del modelo coincide con el de la respuesta guardada.")
    print("  KL uniforme: lo que se obtendría repartiendo la probabilidad a partes iguales.\n")


def cmd_inspect(args):
    """Resumen del dataset, del modelo y de la curva de entrenamiento."""
    if not args.model_dir and not args.data_dir:
        _fail("indica --model-dir y/o --data-dir")
    if args.data_dir:
        p = Path(args.data_dir) / META_NAME
        if not p.exists():
            _fail(f"no se encuentra {p}")
        print(f"\n[inspect] Dataset: {args.data_dir}")
        meta = json.load(open(p, encoding='utf-8'))
        _print_dataset_meta(meta)
        if meta.get('skipped'):
            print("  Omitidas: " + ', '.join(f"{k} {v}" for k, v in meta['skipped'].items()) + "\n")
    if args.model_dir:
        d = Path(args.model_dir)
        if not (d / CONFIG_NAME).exists():
            _fail(f"no se encuentra {d / CONFIG_NAME}")
        cfg = json.load(open(d / CONFIG_NAME, encoding='utf-8'))
        print(f"[inspect] Modelo: {args.model_dir}")
        for k in ('bits', 'dim', 'calibration_temperature', 'best_val_kl', 'epochs_run',
                  'sources', 'model_filter', 'one_hot_weight', 'lr', 'l2'):
            if k in cfg:
                print(f"  {k:<24} {cfg[k]}")
        n_par = 2 * (1 << cfg['bits']) * cfg['dim'] + (1 << cfg['bits'])
        print(f"  {'parámetros':<24} {n_par:,}")
        hp = d / HISTORY_NAME
        if hp.exists():
            _train_report(json.load(open(hp, encoding='utf-8')))


# ══════════════════════════════════════════════════════════════════════════════
#  ARGPARSE
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='jev_local',
        description='Estimador local de probabilidades entrenado con la cache de jev_composer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', metavar='COMMAND')
    sub.required = True

    # ── prepare ───────────────────────────────────────────────────────────────
    p = sub.add_parser('prepare', help='Cache de jev_composer → dataset de rasgos (.npz)')
    p.add_argument('--cache-file', default=DEFAULT_CACHE, metavar='FILE',
                   help=f'Cache JSONL de jev_composer (default: {DEFAULT_CACHE})')
    p.add_argument('--data-dir', required=True, metavar='DIR',
                   help='Directorio de salida del dataset')
    p.add_argument('--model-filter', metavar='STR',
                   help='Usar solo entradas hechas con este modelo (default: todas)')
    p.add_argument('--sources', nargs='+', choices=SOURCES, metavar='SRC',
                   help='Orígenes a incluir: api train (default: ambos)')
    p.add_argument('--val-frac', type=float, default=0.1, metavar='FLOAT',
                   help='Fracción de obras para validación (default: 0.1)')
    p.add_argument('--buckets-bits', type=int, default=18, metavar='INT',
                   help='Filas por tabla de rasgos: 2^bits (default: 18)')
    p.add_argument('--seed', type=int, default=42, metavar='INT',
                   help='Semilla del reparto train/validación (default: 42)')
    p.add_argument('--report', action='store_true',
                   help='Estadísticas detalladas del dataset')
    p.set_defaults(func=cmd_prepare)

    # ── train ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('train', help='Entrena y calibra el modelo')
    p.add_argument('--data-dir', required=True, metavar='DIR',
                   help='Directorio con el dataset de prepare')
    p.add_argument('--model-dir', required=True, metavar='DIR',
                   help='Directorio de salida del modelo')
    p.add_argument('--epochs', type=int, default=50, metavar='INT',
                   help='Épocas máximas (default: 50)')
    p.add_argument('--batch-size', type=int, default=32, metavar='INT',
                   help='Consultas por paso de optimización (default: 32)')
    p.add_argument('--dim', type=int, default=32, metavar='INT',
                   help='Dimensión de los embeddings de las torres (default: 32)')
    p.add_argument('--lr', type=float, default=0.05, metavar='FLOAT',
                   help='Tasa de aprendizaje de Adagrad (default: 0.05)')
    p.add_argument('--l2', type=float, default=1e-6, metavar='FLOAT',
                   help='Regularización L2 de las filas actualizadas (default: 1e-6)')
    p.add_argument('--patience', type=int, default=8, metavar='INT',
                   help='Parada anticipada: épocas sin mejora (default: 8)')
    p.add_argument('--one-hot-weight', type=float, default=1.0, metavar='FLOAT',
                   dest='one_hot_weight',
                   help='Peso de las entradas source=train frente a source=api (default: 1.0)')
    p.add_argument('--no-calibrate', action='store_true', dest='no_calibrate',
                   help='No ajustar la temperatura final con validación')
    p.add_argument('--seed', type=int, default=42, metavar='INT',
                   help='Semilla de inicialización y orden de los datos (default: 42)')
    p.add_argument('--resume', action='store_true',
                   help='Reanudar desde el último checkpoint')
    p.add_argument('--report', action='store_true',
                   help='Mostrar la curva de pérdida al finalizar')
    p.set_defaults(func=cmd_train)

    # ── compute ───────────────────────────────────────────────────────────────
    p = sub.add_parser('compute', help='Probabilidades de una consulta (formato Jev)')
    p.add_argument('--model-dir', required=True, metavar='DIR',
                   help='Directorio con el modelo entrenado')
    p.add_argument('--input', default='-', metavar='FILE',
                   help="Cuerpo JSON de la consulta ('-' = stdin, default)")
    p.add_argument('--cache-file', metavar='FILE',
                   help='Calcular una entrada de cache (con --key)')
    p.add_argument('--key', metavar='STR', help='Clave (o prefijo) de la entrada de cache')
    p.add_argument('--output', metavar='FILE', help='Respuesta JSON (default: stdout)')
    p.add_argument('--temperature', type=float, default=1.0, metavar='FLOAT',
                   help='Multiplica la temperatura calibrada (default: 1.0)')
    p.add_argument('--kind', choices=KINDS, metavar='KIND',
                   help='Forzar el tipo de pregunta (default: inferido de las instrucciones)')
    p.add_argument('--pretty', action='store_true', help='Resumen legible por stderr')
    p.set_defaults(func=cmd_compute)

    # ── serve ─────────────────────────────────────────────────────────────────
    p = sub.add_parser('serve', help='Servidor HTTP con el protocolo de Jev')
    p.add_argument('--model-dir', required=True, metavar='DIR',
                   help='Directorio con el modelo entrenado')
    p.add_argument('--host', default='127.0.0.1', metavar='STR',
                   help='Dirección de escucha (default: 127.0.0.1)')
    p.add_argument('--port', type=int, default=8808, metavar='INT',
                   help='Puerto (default: 8808)')
    p.add_argument('--temperature', type=float, default=1.0, metavar='FLOAT',
                   help='Multiplica la temperatura calibrada (default: 1.0)')
    p.add_argument('--token', metavar='STR', help="Exigir 'Authorization: Bearer <token>'")
    p.add_argument('--verbose', action='store_true', help='Registrar cada petición por stderr')
    p.set_defaults(func=cmd_serve)

    # ── evaluate ──────────────────────────────────────────────────────────────
    p = sub.add_parser('evaluate', help='Compara el modelo con las respuestas guardadas')
    p.add_argument('--model-dir', required=True, metavar='DIR',
                   help='Directorio con el modelo entrenado')
    p.add_argument('--data-dir', required=True, metavar='DIR',
                   help='Directorio con el dataset de prepare')
    p.add_argument('--split', choices=['val', 'train', 'all'], default='val',
                   help='Subconjunto a evaluar (default: val)')
    p.add_argument('--temperature', type=float, default=1.0, metavar='FLOAT',
                   help='Multiplica la temperatura calibrada (default: 1.0)')
    p.set_defaults(func=cmd_evaluate)

    # ── inspect ───────────────────────────────────────────────────────────────
    p = sub.add_parser('inspect', help='Resumen del dataset, del modelo y de las curvas')
    p.add_argument('--model-dir', metavar='DIR', help='Modelo entrenado')
    p.add_argument('--data-dir', metavar='DIR', help='Dataset preparado')
    p.set_defaults(func=cmd_inspect)

    return parser


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
