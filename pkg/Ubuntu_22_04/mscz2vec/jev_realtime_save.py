#!/usr/bin/env python3
r"""
jev_realtime.py — Jev Realtime: compositor en tiempo real con modelo Jev local propio
(versión 0.1; un solo fichero: entrenamiento + inferencia en vivo).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Por CADA COMPÁS se construye una consulta («query») a Jev con:
    · la descripción de la obra (texto, como en jev_composer),
    · el valor de los 8 sliders en ese compás (cuantizado a --levels niveles),
    · el histórico de acordes (últimos --depth compases con sus notas).
Con ella Jev da una distribución sobre los acordes candidatos (diatónicos + V7)
y se SAMPLEA el acorde del compás. Después, nota a nota, se hacen consultas con
la misma información + el acorde elegido y se samplea cada nota sobre candidatos
legales (escala/acorde, ventana de registro, salto máximo). La capa rítmica
(densidad, síncopa, respiros) decide CUÁNDO suena cada nota; Jev decide QUÉ suena.

El «Jev local» es el modelo de dos torres con rasgos hasheados de jev_local.py
(solo numpy), copiado aquí y adaptado: los rasgos se calculan directamente de la
estructura de la consulta (no hay servidor HTTP ni parseo de texto; el texto del
estado solo se genera para depurar: --dump-queries).

Sliders (todos en 0..1, 0.5 = compás «mediano» del corpus de entrenamiento):
    valencia     brillo del modo + calidad del acorde (mayor/menor/dim)
    energia      velocidad media + densidad total de ataques
    densidad     ataques de melodía por pulso
    tension      función del acorde (dominante/sensible…) + 7ª/dim + notas ajenas al acorde
    registro     altura media de la melodía (lineal entre p5 y p95 del corpus)
    complejidad  acordes con 7ª/raros + clases de altura distintas + cobertura baja
    sorpresa     rareza de la transición de acorde (bigramas del corpus) + saltos
    groove       síncopa (ataques en semicorcheas + notas ligadas sobre el pulso)
Las mismas fórmulas se aplican a los MIDIs en entrenamiento («analyze» las muestra),
por lo que el slider tiene el mismo significado al entrenar y al tocar.
Además, parámetros directos sin modelo: tempo y velocidad (energía), textura del
acompañamiento (energía/complejidad), swing y articulación (groove), densidad
rítmica (densidad), registro (centro de la ventana de notas).

Fuente del acorde (--chord-source):
    model      el acorde de cada compás lo genera Jev (por defecto)
    keyboard   el acorde se lee de un teclado MIDI (notas pulsadas, con latch,
               pedal de sustain y reconocimiento de calidad); Jev solo genera las
               notas sobre ese acorde. Sin teclado: --kb-chords "C Am F G7"
               (un acorde por compás, útil con render) o «chord Am» por consola.

Sin --model-dir se usa un estimador de reglas (NO es el modelo) para probar el motor.

────────────────────────────────────────────────────────────────────────────
Referencia MIDI (--reference FILE.mid)
────────────────────────────────────────────────────────────────────────────
Con `--reference` el MIDI guía la generación: de él salen la tonalidad, el modo, el
compás, el tempo, la descripción y —según los modos— los acordes, el contorno melódico
y el ritmo por compás. La salida no es una copia: Jev sigue decidiendo y se mezcla con
el prior de la referencia según `--ref-strength` (0 = ignorarla, 1 = casi literal; con
`--temperature 0.05` y strength ≥ 0.95 se vuelve casi determinista).

  --ref-chords   follow | bias | off   acordes de la referencia o mezclados con Jev
  --ref-melody   bias | off            contorno de la referencia mezclado con Jev
  --ref-rhythm   follow | bias | off   duraciones y silencios de la referencia
  --ref-sliders                        los 8 sliders siguen los valores de la referencia
  --ref-slider-gain K                  fader en 0.5 = «como la referencia»; moverlo desvía
  --ref-slider-relax / --no-ref-slider-relax   mover un slider reduce la fuerza de la ref
  --ref-tempo / --no-ref-tempo         usar el tempo del MIDI como --bpm base
  --ref-loop / --no-ref-loop           al acabar la referencia, repetir o dejar libre a Jev
  --ref-accomp                         acompañamiento fiel (armonía del MIDI) en vez de plantillas
  --ref-melody-track N                 pista (0-based) con la melodía de la referencia

Dos referencias (--references A.mid B.mid --ref-fader NOMBRE --ref-top T):
  Con s = valor normalizado (0..1) del fader elegido, cada decisión (acorde, nota, duración)
  mezcla geométricamente  Jev^(1-T) · A^(T·s) · B^(T·(1-s)).  s=1 → solo A, s=0 → solo B.
  Tonalidad, modo, compás, tempo y descripción salen de A; B se transpone a la tónica de A
  (y ambas a --tonic si se da). T sustituye a --ref-strength y no se relaja con los sliders.
  Con --ref-chords follow el acorde se SAMPLEA de la mezcla (no se copia); con --ref-rhythm
  follow y --ref-accomp cada compás manda una referencia, sorteada con el peso T·s : T·(1-s).
  Con --ref-sliders las trayectorias de las dos referencias se mezclan con s. El fader de
  mezcla deja de entrar en la consulta a Jev. No se combina con --reference.

Sin `--reference`, `--desc/--tonic/--mode/--beats` conservan sus valores por defecto
(«obra instrumental», C, major, 4).

────────────────────────────────────────────────────────────────────────────
Modo curvas (--curve-bars N): sliders leídos de fichero + flujo iterativo por vueltas
────────────────────────────────────────────────────────────────────────────
Con `--curve-bars N` los 8 sliders que modulan la música NO se leen del controlador sino
de una «curva» por compás (fichero de texto). Jev sigue generando en tiempo real; lo único
que cambia es de dónde salen los valores de los sliders. Una VUELTA = N compases.

Formato del fichero (una línea por compás, compás desde 1, '#' comenta):
    compás: valencia energia densidad tension registro complejidad sorpresa groove
    1: 0.60 0.50 0.50 0.25 0.50 0.30 0.35 0.20
Cada valor se aplica en el INICIO de su compás. Los compases que falten repiten el valor
de la línea anterior; los que sobren (> N) se ignoran. Interpolación (--curve-interp):
    lineal      recta entre el valor de un compás y el del siguiente
    escalonada  el valor del compás se mantiene durante todo el compás
    suave       cúbica monótona (Hermite/PCHIP): C1, pasa por los puntos, sin rebasarlos

Mientras suena la vuelta, los faders físicos NO afectan al sonido: se graban en una curva
nueva. Regla de grabación: si un fader se mueve durante el compás k, su valor (el que tenga
al acabar ese compás) se guarda como valor del compás k+1; si no se mueve, se conserva el
valor del fichero que está sonando. Al completar N compases la curva se guarda como
NOMBRE_%04d.txt (el contador no pisa ficheros existentes) y ESA curva suena en la vuelta
siguiente. El bloqueo M1-8 de un fader equivale a «no moverlo». Los knobs, botones y la
consola siguen funcionando en vivo; los movimientos en el último compás de una vuelta no
tienen «compás siguiente» dentro del fichero y se descartan.

  --curve-bars N            activa el modo; compases por vuelta (≥ 2)
  --curve-in FILE           curva inicial (sin ella: todos los sliders en su valor por defecto)
  --curve-out NOMBRE        base de los ficheros de salida (default: curva → curva_0001.txt);
                            si contiene '%' se usa como plantilla (p. ej. «tmp/take_%04d.txt»)
  --curve-interp MODO       lineal (default) | escalonada | suave
  --no-curve-sliders        dejar los faders en vivo (útil si solo quieres la curva de acordes)

Acordes en modo curvas (--chord-source curve, requiere --curve-bars): el acorde de cada compás
se lee de un fichero «compás: acorde» (Am, G7, F#m7, Bdim, Bhdim7…; «-» = lo genera Jev) y el
teclado MIDI (--kb-port) solo GRABA. Un acorde pulsado se asigna al compás cuya frontera
(su inicio) esté más cerca: en la 1ª mitad de un compás vale para ESE compás (pulsación tardía);
en la 2ª, para el SIGUIENTE (pulsación adelantada); al final de la vuelta la 2ª mitad del último
compás cae en el compás 1. Si no pulsas nada en un compás se conserva el del fichero. Mismas
vueltas que los faders: al completar N compases se guarda NOMBRE_%04d.txt y esa curva suena en
la vuelta siguiente. Lo que pulsas no suena en la vuelta en curso (salvo --thru). Sin fichero
inicial (o con «-»), Jev genera el acorde de ese compás y queda como valor del fichero.
  --chord-curve-in FILE     curva de acordes inicial
  --chord-curve-out NOMBRE  base de los ficheros (default: acordes → acordes_0001.txt)

────────────────────────────────────────────────────────────────────────────
Subcomandos
────────────────────────────────────────────────────────────────────────────
  train DIR --model-dir model/     MIDIs (y X.txt opcionales) → datos → entrena
  train X.mid --reference-mode     Un MIDI: validación por bloques, sin calibrar
  analyze DIR                      Muestra lo que se extrae de cada MIDI
  run --model-dir model/ --desc "Nana melancólica"          Tocar en vivo
  run --model-dir model/ --reference X.mid                  Tocar guiado por un MIDI
  run --chord-source keyboard --kb-port "Keystation"        Acordes desde teclado
  render --bars 32 --demo --out-file demo.mid               Offline a .mid
  render --reference X.mid --out-file seg.mid               Seguir un MIDI offline
  compare REF.mid OUT.mid          Similitud de clases, intervalos, ritmo y acordes
  bench --model-dir model/                                  Latencia por consulta
  ports | monitor                                           Diagnóstico MIDI

X.txt junto a X.mid (opcional):  descripción: texto / tonalidad: Am / modo: dorian
Sin descripción se usa --desc o el nombre del fichero.

Controles (nanoKONTROL2): faders 1-8 = sliders; knobs 1-3 volúmenes, 5 tempo ±20 %,
6 humanización, 7 temperatura, 8 amplitud de registro; S1-3 mute melodía/acordes/bajo;
M1-8 bloquea el fader; R1 cadencia (V7→I); R2 modular +5ª; Play pausa; Stop parada;
Rew/FF octava; Rec grabar; Track ◄ ► tonalidad ∓1 semitono.
Consola: «valencia 0.8», «cadence», «key 1», «octave -1», «chord Am», «driftreset», «q».

DEPENDENCIAS: numpy; mido + python-rtmidi (MIDI en vivo y train); pyfluidsynth + .sf2
(sonido). render, bench y el modelo no necesitan hardware.
"""
import argparse
import bisect
import heapq
import itertools
import json
import math
import os
import random
import re
import struct
import sys
import threading
import time
import unicodedata
import zlib
from collections import Counter, deque
from pathlib import Path

try:
    import numpy as np
except ImportError:                                           # pragma: no cover
    sys.exit("[ERROR] jev_realtime.py necesita numpy (pip install numpy)")

EPS = 1e-6

# ═══════════════════════════════════════════════════════════════════════════════
# TEORÍA
# ═══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PITCH_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTE_PC = {n: i for i, n in enumerate(PITCH_NAMES)}
NOTE_PC.update({n: i for i, n in enumerate(PITCH_NAMES_FLAT)})

MODE_INTERVALS = {
    'major':             [0, 2, 4, 5, 7, 9, 11],
    'minor':             [0, 2, 3, 5, 7, 8, 10],
    'dorian':            [0, 2, 3, 5, 7, 9, 10],
    'phrygian':          [0, 1, 3, 5, 7, 8, 10],
    'lydian':            [0, 2, 4, 6, 7, 9, 11],
    'mixolydian':        [0, 2, 4, 5, 7, 9, 10],
    'harmonic_minor':    [0, 2, 3, 5, 7, 8, 11],
    'phrygian_dominant': [0, 1, 4, 5, 7, 8, 10],
}
MODES = list(MODE_INTERVALS)
BRIGHT = {'phrygian': 0.0, 'phrygian_dominant': 0.1, 'harmonic_minor': 0.15, 'minor': 0.25,
          'dorian': 0.45, 'mixolydian': 0.65, 'major': 0.85, 'lydian': 1.0}
LADDER = ['phrygian', 'minor', 'dorian', 'mixolydian', 'major', 'lydian']   # oscuro → brillante

TRIAD_QUALITY = {(0, 4, 7): '', (0, 3, 7): 'm', (0, 3, 6): 'dim', (0, 4, 8): 'aug'}
SEVENTH_QUALITY = {(0, 4, 7, 11): 'M7', (0, 4, 7, 10): '7', (0, 3, 7, 10): 'm7',
                   (0, 3, 6, 10): 'hdim7', (0, 3, 6, 9): 'dim7'}
SEVENTH_SUFFIX = {'M7': 'M7', '7': '7', 'm7': '7', 'hdim7': 'ø7', 'dim7': '°7'}
CHORD_TEMPLATES = [('', (0, 4, 7)), ('m', (0, 3, 7)), ('dim', (0, 3, 6)), ('aug', (0, 4, 8)),
                   ('7', (0, 4, 7, 10)), ('M7', (0, 4, 7, 11)), ('m7', (0, 3, 7, 10)),
                   ('hdim7', (0, 3, 6, 10)), ('dim7', (0, 3, 6, 9)), ('sus4', (0, 5, 7)),
                   ('sus2', (0, 2, 7)), ('5', (0, 7))]
TPL = dict(CHORD_TEMPLATES)
SEVENTH_QS = {'7', 'M7', 'm7', 'hdim7', 'dim7'}
ODD_Q = {'dim', 'aug', 'hdim7', 'dim7', 'sus2', 'sus4'}
MINOR_Q = {'m', 'dim', 'm7', 'hdim7', 'dim7'}
ROMAN_CHROM = ['I', 'bII', 'II', 'bIII', 'III', 'IV', '#IV', 'V', 'bVI', 'VI', 'bVII', 'VII']
FN_OF_REL = {0: 'tonic', 1: 'other', 2: 'supertonic', 3: 'mediant', 4: 'mediant', 5: 'subdominant',
             6: 'other', 7: 'dominant', 8: 'submediant', 9: 'submediant', 10: 'subtonic',
             11: 'leading'}
FN_TENSION = {'tonic': 0.0, 'mediant': .2, 'submediant': .2, 'subdominant': .4, 'supertonic': .55,
              'other': .7, 'subtonic': .8, 'dominant': 1.0, 'leading': 1.0}
FN_CLASS = {'tonic': 'T', 'mediant': 'T', 'submediant': 'T', 'subdominant': 'S',
            'supertonic': 'P', 'dominant': 'D', 'leading': 'D', 'subtonic': 'D', 'other': 'P'}
CHORD_ROLE = {0: 'root', 2: '9th', 3: 'minor 3rd', 4: 'major 3rd', 5: '4th', 6: 'diminished 5th',
              7: '5th', 8: 'augmented 5th', 9: '6th', 10: 'minor 7th', 11: 'major 7th'}

SLIDERS = ['valencia', 'energia', 'densidad', 'tension', 'registro', 'complejidad',
           'sorpresa', 'groove']
SLIDER_DEFAULTS = {'valencia': .6, 'energia': .5, 'densidad': .5, 'tension': .25,
                   'registro': .5, 'complejidad': .3, 'sorpresa': .35, 'groove': .2}
KNOBS = ['vol_melodia', 'vol_acordes', 'vol_bajo', 'reverb', 'tempo', 'humanizar',
         'temperatura', 'amplitud']
LAYERS = ['melodia', 'acordes', 'bajo']
DEMO_PERIOD = {'valencia': 41, 'energia': 53, 'densidad': 19, 'tension': 29,
               'registro': 37, 'complejidad': 61, 'sorpresa': 23, 'groove': 47}
DEFAULT_PARAMS = {'levels': 5, 'depth': 4, 'leap': 9, 'half_width': 14,
                  'reg_lo': 48.0, 'reg_hi': 84.0}
KINDS = ['acorde', 'nota']


def fail(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def warn(msg):
    print(f"[AVISO] {msg}", file=sys.stderr)


def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def fmt(x):
    return f"{x:g}"


def midi_to_name(m):
    return f"{PITCH_NAMES[m % 12]}{m // 12 - 1}"


def parse_key(text):
    s = text.strip()
    hint = None
    if len(s) >= 2 and s.endswith('m') and (s[:-1][:1].upper() + s[:-1][1:]) in NOTE_PC:
        s, hint = s[:-1], 'minor'
    s = s[:1].upper() + s[1:]
    if s not in NOTE_PC:
        fail(f"tónica no reconocida: '{text}'")
    return NOTE_PC[s] % 12, hint


def make_chord(root, q, tonic, ivs=None):
    ivs = ivs or TPL.get(q, (0, 4, 7))
    rel = (root - tonic) % 12
    num = ROMAN_CHROM[rel]
    if q in MINOR_Q:
        num = num.lower()
    num += {'dim': '°', 'aug': '+', 'hdim7': 'ø7', 'dim7': '°7', '7': '7', 'M7': 'M7',
            'm7': '7'}.get(q, '')
    return {'name': f"{PITCH_NAMES[root]}{q}", 'root': root, 'q': q,
            'pitches': [(root + i) % 12 for i in ivs], 'fn': FN_OF_REL[rel], 'numeral': num}


def chord_desc(c):
    return (f"{c['fn']}{', with 7th' if c['q'] in SEVENTH_QS else ''}; "
            f"notes {' '.join(PITCH_NAMES[p] for p in c['pitches'])}")


def diatonic_chords(tonic, mode):
    """{etiqueta: acorde} con los acordes diatónicos del modo (tríadas y séptimas) + V7."""
    sc = MODE_INTERVALS[mode]
    out = {}

    def add(c):
        c = dict(c)
        c['label'] = f"{c['numeral']} ({c['name']})"
        c['desc'] = chord_desc(c)
        out.setdefault(c['label'], c)

    for i in range(7):
        root = (tonic + sc[i]) % 12
        tones = [sc[(i + k) % 7] + 12 * ((i + k) // 7) - sc[i] for k in (0, 2, 4, 6)]
        tq = TRIAD_QUALITY.get(tuple(t % 12 for t in tones[:3]))
        if tq is not None:
            add(make_chord(root, tq, tonic, [t % 12 for t in tones[:3]]))
        sq = SEVENTH_QUALITY.get(tuple(t % 12 for t in tones[:4]))
        if sq is not None:
            add(make_chord(root, sq, tonic, [t % 12 for t in tones[:4]]))
    v = (tonic + 7) % 12
    if not any(c['name'] == f"{PITCH_NAMES[v]}7" for c in out.values()):
        add(make_chord(v, '7', tonic))
    return out


def chord_options(tonic, mode):
    return [{'label': k, 'desc': c['desc'], 'chord': c}
            for k, c in diatonic_chords(tonic, mode).items()]


def recognize_chord(pcs, bass=None):
    """Conjunto de clases de altura → (raíz, calidad). Plantillas con puntuación."""
    pcs = set(pcs)
    if not pcs:
        return None
    if len(pcs) == 1:
        return next(iter(pcs)), ''
    best, best_s = None, -1e9
    for root in range(12):
        for q, ivs in CHORD_TEMPLATES:
            tpl = {(root + i) % 12 for i in ivs}
            hit = len(pcs & tpl)
            if hit < 2:
                continue
            s = hit - 0.6 * len(pcs - tpl) - 0.35 * len(tpl - pcs) - 0.02 * len(ivs)
            if root == bass:
                s += 0.4
            if root in pcs:
                s += 0.05
            if s > best_s:
                best, best_s = (root, q), s
    return best


CHORD_NAME_RE = re.compile(r'^([A-Ga-g][#b]?)(hdim7|ø7|dim7|maj7|M7|m7|dim|aug|sus2|sus4|m|7|5|)$')


def parse_chord_name(text):
    m = CHORD_NAME_RE.match(text.strip())
    if not m:
        return None
    root = NOTE_PC.get(m.group(1)[:1].upper() + m.group(1)[1:])
    if root is None:
        return None
    q = {'maj7': 'M7', 'ø7': 'hdim7'}.get(m.group(2), m.group(2))
    return root, q


# ═══════════════════════════════════════════════════════════════════════════════
# CONSULTA A JEV (descripción + sliders + histórico de acordes)
# ═══════════════════════════════════════════════════════════════════════════════

class Memory:
    """Histórico de compases: [{'chord', 'bar', 'notes': [(midi, dur)]}]."""

    def __init__(self, depth):
        self.depth, self.entries, self.last_midi = depth, [], None

    def add(self, chord, bar):
        e = {'chord': chord, 'bar': bar, 'notes': []}
        self.entries.append(e)
        if len(self.entries) > 64:
            del self.entries[:-32]
        return e

    def window(self):
        return self.entries[-self.depth:]


class Query:
    def __init__(self, kind, desc, tonic, mode, beats, bar, levels, L, window, options,
                 beat=0.0, dur=0.0, cur=None, center=60):
        self.kind, self.desc, self.tonic, self.mode, self.beats = kind, desc, tonic, mode, beats
        self.bar, self.levels, self.L, self.window, self.options = bar, levels, L, window, options
        self.beat, self.dur, self.cur, self.center = beat, dur, cur, center

    def criteria(self):
        return {o['label']: o['desc'] for o in self.options}

    def instructions(self):
        if self.kind == 'acorde':
            return (f"Which chord should come next? It starts at bar {self.bar + 1} beat 1 and "
                    f"lasts {fmt(self.beats)} beats. Choose the single best fit for the "
                    f"description, the slider values and the chord history.")
        return (f"Which note should the melody play next? It starts at bar {self.bar + 1} beat "
                f"{fmt(round(self.beat + 1, 4))}, lasts {fmt(self.dur)} beat(s) and sounds over "
                f"the chord {self.cur['name']}. Choose the single best fit for the description, "
                f"the slider values and the harmony.")

    def state_text(self):
        sc = ' '.join(PITCH_NAMES[(self.tonic + iv) % 12] for iv in MODE_INTERVALS[self.mode])
        lines = []
        for e in self.window:
            ns = ' '.join(f"{midi_to_name(m)}({fmt(d)})" for m, d in e['notes'] if m is not None)
            lines.append(f"bar {e['bar'] + 1} | {e['chord']['name']} ({fmt(self.beats)} beats): "
                         f"{ns or '(no melody yet)'}")
        sl = ', '.join(f"{v} {self.levels[v]}/{self.L - 1}" for v in SLIDERS)
        return (f"PIECE DESCRIPTION:\n{self.desc.strip()}\n\n"
                f"KEY: {PITCH_NAMES[self.tonic]} {self.mode} (scale: {sc})\n"
                f"METER: {self.beats} beats per bar\n"
                f"SLIDERS FOR THIS BAR: {sl}\n\n"
                f"CHORD HISTORY (last {len(self.window)} bar(s), melody as note(beats)):\n"
                + '\n'.join(lines) + '\n')


def pitched(window):
    return [(m, d) for e in window for m, d in e['notes'] if m is not None]


def note_option(tonic, mode, chord, prev, m):
    """Opción de nota con su papel armónico y el movimiento respecto a `prev`."""
    scale = MODE_INTERVALS[mode]
    root = chord['pitches'][0]
    pc = m % 12
    pv = prev if prev is not None else m
    if pc in chord['pitches']:
        role, rk = f"chord tone ({CHORD_ROLE.get((pc - root) % 12, 'tone')} of {chord['name']})", 'ct'
    else:
        rel = (pc - tonic) % 12
        if rel in scale:
            role, rk = f"passing scale tone (degree {scale.index(rel) + 1})", 'pass'
        else:
            role, rk = "chromatic tone", 'chrom'
    if pc == tonic:
        role += ', the tonic'
    d = m - pv
    motion = ('repeats the previous note' if d == 0
              else f"{'up' if d > 0 else 'down'} {abs(d)} semitone(s)")
    return {'label': midi_to_name(m), 'desc': f"{role}; {motion}", 'midi': m,
            'role': rk, 'istonic': pc == tonic, 'tone': (pc - root) % 12, 'delta': d}


def note_options(P, tonic, mode, chord, prev, center):
    """Notas candidatas: escala ∪ acorde, ventana de registro, salto ≤ leap."""
    hw, leap = int(P['half_width']), int(P['leap'])
    lo, hi = center - hw, center + hw
    pv = prev if prev is not None else center
    scale = MODE_INTERVALS[mode]
    scale_pcs = {(tonic + iv) % 12 for iv in scale}
    allowed = scale_pcs | set(chord['pitches'])
    pool = [m for m in range(max(lo, pv - leap), min(hi, pv + leap) + 1) if m % 12 in allowed]
    if not pool:
        pool = [m for m in range(lo, hi + 1) if m % 12 in allowed] or [pv]
    return [note_option(tonic, mode, chord, pv, m) for m in pool]


# ═══════════════════════════════════════════════════════════════════════════════
# RASGOS HASHEADOS Y MODELO DE DOS TORRES (de jev_local.py)
# ═══════════════════════════════════════════════════════════════════════════════

def words(text):
    t = unicodedata.normalize('NFKD', text.lower())
    t = ''.join(c for c in t if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]{2,}', t)


def _hash(tok, mask):
    return zlib.crc32(tok.encode('utf-8')) & mask


def _dedupe(items):
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def rel_chord(c, tonic):
    return f"{(c['root'] - tonic) % 12}_{c['q']}"


def query_tokens(q):
    kind, tonic, T = q.kind, q.tonic, [f"k={q.kind}"]

    def add(tok, kp=True):
        T.append(tok)
        if kp:
            T.append(f"{kind}/{tok}")

    dw = words(q.desc)[:120]
    for w in dw:
        add(f"w:{w}")
    T.extend(f"b:{a}_{b}" for a, b in zip(dw, dw[1:]))
    add(f"mode={q.mode}")
    add(f"tonic={PITCH_NAMES[tonic]}")
    add(f"meter={q.beats}")
    for v in SLIDERS:                                   # nivel + codificación «termómetro»
        lv = q.levels[v]
        add(f"{v}={lv}")
        for k in range(1, lv + 1):
            add(f"{v}>={k}")
    lv = q.levels
    add(f"valmode={lv['valencia']}|{q.mode}")
    add(f"tenen={lv['tension']}|{lv['energia']}")
    add(f"ph4={q.bar % 4}")
    add(f"ph8={q.bar % 8}")
    win = q.window
    for i, e in enumerate(reversed(win[-4:])):
        add(f"c-{i + 1}:{rel_chord(e['chord'], tonic)}")
    if len(win) >= 2:
        add(f"c-12:{rel_chord(win[-2]['chord'], tonic)}>{rel_chord(win[-1]['chord'], tonic)}")
    pn = pitched(win)
    for i, (m, d) in enumerate(reversed(pn[-4:])):
        add(f"n-{i + 1}:{(m - tonic) % 12}")
        add(f"d-{i + 1}:{d:g}")
    if len(pn) >= 2:
        add(f"i-1:{max(-12, min(12, pn[-1][0] - pn[-2][0]))}")
    if pn:
        add(f"reg:{pn[-1][0] // 6}")
    if kind == 'nota':
        add(f"beat={q.beat:g}")
        add(f"dur={q.dur:g}")
        add(f"cur:{rel_chord(q.cur, tonic)}")
        add(f"curfn:{q.cur['fn']}")
        add(f"cen:{int(q.center) // 6}")
    return _dedupe(T)


def option_tokens(q, o, prev_chord, prev_midi):
    kind, tonic = q.kind, q.tonic
    O = [f"o:{kind}|{o['label']}"]

    def add(tok, kp=True):
        O.append(tok)
        if kp:
            O.append(f"{kind}/{tok}")

    for w in words(o['desc'].split('; notes ')[0])[:60]:
        add(f"ow:{w}")
    if kind == 'acorde':
        c = o['chord']
        rel = (c['root'] - tonic) % 12
        add(f"num:{c['numeral']}")
        add(f"q:{c['q']}")
        add(f"fn:{c['fn']}")
        add(f"rel:{rel}")
        add(f"rq:{rel}_{c['q']}")
        if c['q'] in SEVENTH_QS:
            add('seventh')
        if prev_chord:
            if prev_chord['name'] == c['name']:
                add('repeat')
            d = (c['root'] - prev_chord['root']) % 12
            add(f"rm:{d}")
            add(f"rmq:{d}_{prev_chord['q']}>{c['q']}")
    else:
        m, ch = o['midi'], q.cur
        add(f"pitch:{m}")
        add(f"reg:{m // 6}")
        add(f"rp:{(m - tonic) % 12}")
        d = o['delta']
        add(f"mv:{max(-12, min(12, d))}")
        add(f"mva:{min(abs(d), 12)}")
        add('dir:' + ('up' if d > 0 else 'down' if d < 0 else 'same'))
        add(f"cr:{(m - ch['root']) % 12}")
        add(f"crq:{(m - ch['root']) % 12}_{ch['q']}")
        add(f"role:{o['role']}")
        if o['istonic']:
            add('istonic')
        add(f"cen:{max(-4, min(4, int(round((m - q.center) / 3.0))))}")
    return _dedupe(O)


def featurize(q, bits):
    mask = (1 << bits) - 1
    qi = np.array([_hash(t, mask) for t in query_tokens(q)], dtype=np.int32)
    prev_chord = q.window[-1]['chord'] if (q.kind == 'acorde' and q.window) else None
    oi, optr = [], [0]
    for o in q.options:
        oi.extend(_hash(t, mask) for t in option_tokens(q, o, prev_chord, None))
        optr.append(len(oi))
    return qi, np.array(oi, dtype=np.int32), np.array(optr, dtype=np.int64)


def softmax(s):
    s = s - np.max(s)
    e = np.exp(s)
    return e / e.sum()


class TwoTower:
    """s_i = a_i + <e_Q, e_Oi>; embeddings hasheados; Adagrad sobre filas dispersas."""
    PARAMS = ('EQ', 'EO', 'w')

    def __init__(self, bits, dim, seed=0, init=True):
        n = 1 << bits
        self.bits, self.dim, self.G = bits, dim, None
        if init:
            rng = np.random.default_rng(seed)
            self.EQ = (rng.standard_normal((n, dim)) * 0.1).astype(np.float32)
            self.EO = (rng.standard_normal((n, dim)) * 0.1).astype(np.float32)
            self.w = np.zeros(n, dtype=np.float32)

    def init_optimizer(self):
        self.G = {k: np.full_like(getattr(self, k), 0.1) for k in self.PARAMS}

    def forward(self, qi, oi, optr):
        cq = 1.0 / math.sqrt(len(qi))
        eQ = self.EQ[qi].sum(axis=0) * cq
        starts, lens = optr[:-1], np.diff(optr)
        sq = np.sqrt(lens).astype(np.float32)
        eO = np.add.reduceat(self.EO[oi], starts, axis=0) / sq[:, None]
        a = np.add.reduceat(self.w[oi], starts) / sq
        return a + eO @ eQ, (eQ, eO, cq, lens, sq)

    def backward(self, g, qi, oi, ctx, acc):
        eQ, eO, cq, lens, sq = ctx
        gA = g / sq
        seg = np.repeat(np.arange(len(lens)), lens)
        acc['w'].append((oi, gA[seg]))
        acc['EO'].append((oi, gA[seg][:, None] * eQ[None, :]))
        gq = (g[:, None] * eO).sum(axis=0) * cq
        acc['EQ'].append((qi, np.broadcast_to(gq, (len(qi), self.dim))))

    def apply(self, acc, lr, l2):
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

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **{k: getattr(self, k) for k in self.PARAMS})

    @classmethod
    def load(cls, path, bits, dim):
        z = np.load(str(path), allow_pickle=False)
        m = cls(bits, dim, init=False)
        for k in cls.PARAMS:
            setattr(m, k, z[k])
        return m


CONFIG_NAME, BEST_NAME = 'model_config.json', 'best_model.npz'


class JevLocal:
    """Estimador entrenado: probs(q) → lista de probabilidades por opción."""

    def __init__(self, model_dir):
        d = Path(model_dir)
        if not (d / CONFIG_NAME).exists() or not (d / BEST_NAME).exists():
            fail(f"no hay modelo en {d}: ¿has ejecutado train?")
        self.cfg = json.load(open(d / CONFIG_NAME, encoding='utf-8'))
        self.model = TwoTower.load(d / BEST_NAME, self.cfg['bits'], self.cfg['dim'])
        self.tau = float(self.cfg.get('calibration_temperature', 1.0))
        self.params = dict(DEFAULT_PARAMS)
        self.params.update(self.cfg['params'])
        self.name = f"jev_local (bits {self.cfg['bits']}, dim {self.cfg['dim']})"

    def probs(self, q):
        qi, oi, optr = featurize(q, self.cfg['bits'])
        s, _ = self.model.forward(qi, oi, optr)
        return softmax(s / self.tau).tolist()


class PriorEstimator:
    """Estimador de reglas (NO es el modelo): para probar el motor sin entrenar."""
    TRANS = {'T': {'T': .3, 'S': 1.0, 'P': .7, 'D': .6}, 'S': {'T': .8, 'S': .2, 'P': .4, 'D': 1.1},
             'P': {'T': .3, 'S': .1, 'P': .1, 'D': 1.5}, 'D': {'T': 1.8, 'S': .2, 'P': .1, 'D': .2}}

    def __init__(self):
        self.params = dict(DEFAULT_PARAMS)
        self.name = 'estimador de reglas (sin modelo)'

    def probs(self, q):
        L = max(1, q.L - 1)
        v = {k: q.levels[k] / L for k in SLIDERS}
        t, c = v['tension'] - .5, v['complejidad'] - .5
        sc = []
        if q.kind == 'acorde':
            prev = FN_CLASS[q.window[-1]['chord']['fn']] if q.window else 'T'
            for o in q.options:
                ch = o['chord']
                cl = FN_CLASS[ch['fn']]
                w = math.log(self.TRANS[prev][cl] + .05)
                if q.window and q.window[-1]['chord']['name'] == ch['name']:
                    w -= 1.5
                w += 2.0 * t * (1 if cl == 'D' else -0.8 if cl == 'T' else 0)
                w += (1.5 * c if ch['q'] in SEVENTH_QS else -0.8 * c)
                if ch['q'] in ('dim', 'hdim7', 'dim7'):
                    w += 1.0 * t - 0.3
                sc.append(w)
        else:
            hw = 14
            for o in q.options:
                w = {'ct': 1.1, 'pass': .2, 'chrom': -1.5}[o['role']] - 1.0 * t * (o['role'] == 'ct') \
                    + 1.0 * t * (o['role'] != 'ct')
                d = abs(o['delta'])
                w -= (.5 if d == 0 else .28 * max(0, d - 2) + (0 if d <= 2 else .2))
                w += .3 * o['istonic'] - 0.5 * ((o['midi'] - q.center) / (0.75 * hw)) ** 2
                sc.append(w)
        return softmax(np.array(sc, dtype=np.float64)).tolist()


class LiveJev:
    """ask(q) → probabilidades, con estadísticas de latencia."""

    def __init__(self, est):
        self.est, self.t = est, {}

    def ask(self, q):
        if len(q.options) == 1:
            return [1.0]
        t0 = time.perf_counter()
        p = self.est.probs(q)
        self.t.setdefault(q.kind, []).append((time.perf_counter() - t0) * 1000)
        return p

    def report(self):
        out = []
        for k, v in sorted(self.t.items()):
            v = sorted(v)
            out.append(f"{k}: n={len(v)} media {sum(v) / len(v):.2f} ms  "
                       f"p95 {v[int(.95 * (len(v) - 1))]:.2f}  máx {v[-1]:.2f}")
        return out


def sample(probs, T, rng):
    T = max(T, 0.05)
    lg = [math.log(max(p, 1e-9)) / T for p in probs]
    m = max(lg)
    w = [math.exp(x - m) for x in lg]
    r, acc = rng.random() * sum(w), 0.0
    for i, x in enumerate(w):
        acc += x
        if r <= acc:
            return i
    return len(w) - 1


def combine(probs, prior, s):
    """Mezcla geométrica probs^(1-s) · prior^s, renormalizada con protección."""
    p = [max(a, 1e-12) ** (1.0 - s) * max(b, 1e-12) ** s for a, b in zip(probs, prior)]
    tot = sum(p)
    return [x / tot for x in p] if tot > 1e-30 else [1.0 / len(p)] * len(p)


def combine_multi(probs, w0, items):
    """Mezcla geométrica probs^w0 · Π prior_j^w_j (items = [(prior, w)]), renormalizada.
    Con un solo prior y w0 = 1-s, w = s equivale a combine()."""
    out = []
    for i, p in enumerate(probs):
        v = max(p, 1e-12) ** w0
        for pr, w in items:
            v *= max(pr[i], 1e-12) ** w
        out.append(v)
    tot = sum(out)
    return [x / tot for x in out] if tot > 1e-30 else [1.0 / len(out)] * len(out)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE MIDIS: tonalidad, acorde por compás y variables de los sliders
# ═══════════════════════════════════════════════════════════════════════════════

def read_midi(path):
    try:
        import mido
    except ImportError:
        fail("train/analyze necesitan mido (pip install mido)")
    try:
        mid = mido.MidiFile(path)
    except Exception as e:
        fail(f"no se pudo leer {path}: {e}")
    tpb, notes, beats, tempo_us = mid.ticks_per_beat, [], None, None
    for ti, track in enumerate(mid.tracks):
        t, active = 0, {}
        for msg in track:
            t += msg.time
            if msg.type == 'time_signature' and beats is None:
                beats = max(1, round(msg.numerator * 4 / msg.denominator))
            elif msg.type == 'set_tempo' and tempo_us is None:
                tempo_us = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                active.setdefault((msg.channel, msg.note), []).append((t, msg.velocity))
            elif msg.type in ('note_off', 'note_on'):
                lst = active.get((msg.channel, msg.note))
                if lst:
                    st, vel = lst.pop(0)
                    notes.append({'start': st / tpb, 'end': max(t, st + 1) / tpb, 'pitch': msg.note,
                                  'vel': vel, 'track': ti, 'channel': msg.channel})
        for (ch, n), lst in active.items():
            for st, vel in lst:
                notes.append({'start': st / tpb, 'end': max(t, st + 1) / tpb, 'pitch': n,
                              'vel': vel, 'track': ti, 'channel': ch})
    notes = [n for n in notes if n['channel'] != 9]
    if not notes:
        fail(f"{path}: sin notas")
    bpm = 60_000_000 / tempo_us if tempo_us else None
    return notes, beats or 4, bpm


def split_melody_harmony(notes, melody_track=None):
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
            ov = sum(1 for a, b in zip(g, g[1:]) if b['start'] < a['end'] - 1 / 8)
            if ov / len(g) > 0.15:
                continue
            mean = sum(n['pitch'] for n in g) / len(g)
            if best is None or mean > best[0]:
                best = (mean, g)
        if best:
            mel = best[1]
    if mel is None:                                          # skyline
        by = {}
        for n in notes:
            by.setdefault(round(n['start'] * 8), []).append(n)
        mel = [max(g, key=lambda n: n['pitch']) for g in by.values()]
    ids = {id(n) for n in mel}
    return mel, [n for n in notes if id(n) not in ids]


def quantize_melody(mel, q):
    mono = []
    for n in sorted(mel, key=lambda n: (n['start'], -n['pitch'])):
        if mono and abs(n['start'] - mono[-1]['start']) < EPS:
            continue
        mono.append(dict(n))
    for a, b in zip(mono, mono[1:]):
        a['end'] = min(a['end'], b['start'])
    out, prev_end = [], 0.0
    for n in mono:
        s_ = max(round(n['start'] / q) * q, prev_end)
        e_ = round(n['end'] / q) * q
        if e_ - s_ < q - EPS:
            continue
        out.append({'start': round(s_, 6), 'dur': round(e_ - s_, 6), 'midi': n['pitch'],
                    'vel': n['vel']})
        prev_end = e_
    return out


def infer_key(harm, mel, nbars, beats):
    w = [0.0] * 12
    for n in harm:
        w[n['pitch'] % 12] += (n['end'] - n['start']) * (1.5 if n['pitch'] < 60 else 1.0)
    for n in mel:
        w[n['midi'] % 12] += 0.6 * n['dur']
    tot = sum(w) or 1.0

    def bass_pc(b0, b1):
        c = [n for n in harm if n['start'] < b1 and n['end'] > b0]
        return min(c, key=lambda n: n['pitch'])['pitch'] % 12 if c else None
    end_pc = bass_pc(max(0, nbars - 2) * beats, nbars * beats)
    start_pc = bass_pc(0, beats)
    best, best_s = None, -1.0
    for t in range(12):
        for m in MODES:
            iv = MODE_INTERVALS[m]
            sc = {(t + i) % 12 for i in iv}
            s = (2.0 * sum(w[p] for p in sc) / tot + 0.6 * w[t] / tot + 0.25 * w[(t + 7) % 12] / tot
                 + 0.3 * w[(t + iv[2]) % 12] / tot + 0.4 * (end_pc == t) + 0.2 * (start_pc == t)
                 + (0.03 if m in ('major', 'minor') else 0.0))
            if s > best_s:
                best, best_s = (t, m), s
    return best


def read_txt(path):
    desc, key, mode = [], None, None
    if not os.path.isfile(path):
        return None, None, None
    for line in open(path, encoding='utf-8').read().splitlines():
        m = re.match(r'^(descripci[oó]n|description|desc|tonalidad|key|modo|mode)\s*:\s*(.+)$',
                     line.strip(), re.I)
        if not m:
            continue
        k, v = m.group(1).lower(), m.group(2).strip()
        if k.startswith('desc'):
            desc.append(v)
        elif k in ('tonalidad', 'key'):
            key = v
        else:
            mode = v
    return ('\n'.join(desc) or None), key, mode


def bucket(items, beats, nbars, key_s, key_e):
    out = [[] for _ in range(nbars)]
    for n in items:
        b0 = int(n[key_s] // beats)
        b1 = int((n[key_e] - EPS) // beats)
        for b in range(max(0, b0), min(nbars - 1, b1) + 1):
            out[b].append(n)
    return out


def detect_bar_chord(hn, mn, b0, b1, cands):
    w = [0.0] * 12
    low, low_ov = None, None
    for n in hn:
        ov = min(n['end'], b1) - max(n['start'], b0)
        if ov <= EPS:
            continue
        w[n['pitch'] % 12] += ov * (1.5 if n['pitch'] < 60 else 1.0)
        if ov >= 0.75 and (low is None or n['pitch'] < low):
            low = n['pitch']
        if low_ov is None or n['pitch'] < low_ov:
            low_ov = n['pitch']
    bass = (low if low is not None else low_ov)
    bass = bass % 12 if bass is not None else None
    for n in mn:
        w[n['midi'] % 12] += 0.35 * n['dur']
    tot = sum(w)
    if tot <= EPS:
        return None, 0.0
    w = [x / tot for x in w]
    best, best_s, best_cov = None, -1e9, 0.0
    for c in cands:
        ch = c['chord']
        inn = sum(w[p] for p in ch['pitches'])
        s = (inn - 0.7 * (1 - inn) + 0.25 * (bass == ch['root']) + 0.15 * (w[ch['root']] > 0)
             - 0.03 * len(ch['pitches']))
        if s > best_s:
            best, best_s, best_cov = c, s, inn
    return (best if best_cov >= 0.5 else None), best_cov


QUAL_VALENCE = {'': 1.0, 'M7': .9, '7': .6, 'sus4': .5, 'sus2': .5, '5': .5, 'aug': .4,
                'm': .1, 'm7': .2, 'hdim7': .05, 'dim': 0.0, 'dim7': 0.0}


def bar_raws(b0, b1, beats, chord, cov, hn, mn, mode):
    ons = ([(n['start'], n['end'], n['vel']) for n in hn if b0 - EPS <= n['start'] < b1 - EPS]
           + [(n['start'], n['start'] + n['dur'], n['vel']) for n in mn])
    nons = len(ons)
    vel = sum(o[2] for o in ons) / nons if nons else 64.0
    q = chord['q']
    mel_tot = sum(n['dur'] for n in mn)
    nonch = (sum(n['dur'] for n in mn if n['midi'] % 12 not in chord['pitches']) / mel_tot
             if mel_tot else 0.0)
    sev = 1.0 if q in ODD_Q else 0.5 if q in SEVENTH_QS else 0.0
    allpcs = {n['pitch'] % 12 for n in hn if n['start'] < b1 and n['end'] > b0} | \
             {n['midi'] % 12 for n in mn}
    mpitch = [n['midi'] for n in mn]
    steps = [abs(a - b) for a, b in zip(mpitch, mpitch[1:])]
    sixt = sum(1 for o in ons if abs(((o[0] - b0) * 2) % 1.0) > 1e-3 and abs((o[0] - b0) % 0.5) > 1e-3)
    cross = sum(1 for o in ons if abs((o[0] - b0) % 1.0) > 1e-3
                and o[1] > math.floor(o[0] - b0 + EPS) + 1 + b0 + EPS)
    return {
        'valencia': 0.5 * BRIGHT[mode] + 0.5 * QUAL_VALENCE.get(q, .5),
        'energia': 0.6 * vel / 127 + 0.4 * min(1.0, math.log2(1 + nons / beats) / 3),
        'densidad': len(mn) / beats,
        'tension': 0.45 * FN_TENSION[chord['fn']] + 0.25 * sev + 0.30 * nonch,
        'complejidad': (0.35 * (q in SEVENTH_QS) + 0.15 * (q in ODD_Q)
                        + 0.35 * min(1.0, len(allpcs) / 10) + 0.15 * (1 - cov)),
        'groove': 0.5 * sixt / max(1, nons) + 0.5 * cross / max(1, nons),
        'regmean': (sum(mpitch) / len(mpitch)) if mpitch else None,
        'melstep': (sum(steps) / len(steps)) if steps else 0.0,
    }


class Piece:
    pass


def analyze_midi(path, a):
    notes, beats0, bpm = read_midi(path)
    beats = a.beats or beats0
    mel, harm = split_melody_harmony(notes, a.melody_track)
    mel = quantize_melody(mel, a.quant)
    if not mel:
        fail(f"{path}: no se detectó melodía")
    end = max(max((n['end'] for n in harm), default=0.0), max(n['start'] + n['dur'] for n in mel))
    nbars = max(1, int(math.ceil(end / beats - EPS)))
    stem = os.path.splitext(os.path.basename(path))[0]
    desc, tkey, tmode = read_txt(os.path.splitext(path)[0] + '.txt')
    desc = desc or a.desc or stem.replace('_', ' ')
    key_text, mode = tkey or a.key, tmode or a.mode
    tonic = hint = None
    if key_text:
        tonic, hint = parse_key(key_text)
    mode = mode or hint
    if mode is not None and mode not in MODES:
        fail(f"{stem}: modo no válido '{mode}'")
    if tonic is None or mode is None:
        t_inf, m_inf = infer_key(harm, mel, nbars, beats)
        tonic = t_inf if tonic is None else tonic
        mode = m_inf if mode is None else mode
    cands = chord_options(tonic, mode)
    hb = bucket(harm, beats, nbars, 'start', 'end')
    mb = [[] for _ in range(nbars)]
    for n in mel:
        b = int(n['start'] // beats)
        if b < nbars:
            mb[b].append(n)
    p = Piece()
    (p.name, p.desc, p.tonic, p.mode, p.beats, p.nbars, p.path,
     p.bpm) = stem, desc, tonic, mode, beats, nbars, path, bpm
    p.bars = []
    for b in range(nbars):
        b0, b1 = b * beats, (b + 1) * beats
        notes_b = [{'start': n['start'] - b0, 'dur': min(n['dur'], b1 - n['start']), 'midi': n['midi']}
                   for n in mb[b]]
        harm_b = [{'start': n['start'] - b0, 'dur': n['end'] - n['start'], 'midi': n['pitch']}
                  for n in hb[b] if n['start'] < b1 and n['end'] > b0]
        chord, cov = detect_bar_chord(hb[b], mb[b], b0, b1, cands)
        bar = {'b': b, 'chord': None, 'notes': notes_b, 'harm': harm_b}
        if chord is not None:
            bar.update(chord=chord['chord'], label=chord['label'],
                       raw=bar_raws(b0, b1, beats, chord['chord'], cov, hb[b], mb[b], mode))
        p.bars.append(bar)
    return p


class Norm:
    """Normalización del corpus: rangos ordenados (rank), bigramas de acordes y rango de registro."""
    RANKED = ['valencia', 'energia', 'densidad', 'tension', 'complejidad', 'sorpresa', 'groove']

    def __init__(self, pieces):
        uni, bi, K = Counter(), Counter(), 16
        regs = []
        for p in pieces:
            prev = None
            for bar in p.bars:
                if bar['chord'] is None:
                    prev = None
                    continue
                k = rel_chord(bar['chord'], p.tonic)
                uni[k] += 1
                if prev is not None:
                    bi[(prev, k)] += 1
                prev = k
                if bar['raw']['regmean'] is not None:
                    regs.append(bar['raw']['regmean'])
        self.reg_lo, self.reg_hi = ((float(np.percentile(regs, 5)), float(np.percentile(regs, 95)))
                                    if regs else (48.0, 84.0))
        if self.reg_hi - self.reg_lo < 12:
            mid = (self.reg_hi + self.reg_lo) / 2
            self.reg_lo, self.reg_hi = mid - 6, mid + 6
        n_all = sum(uni.values())
        prev_count = Counter()
        for (a_, _), c in bi.items():
            prev_count[a_] += c
        vals = {k: [] for k in self.RANKED}
        for p in pieces:
            prev, last_reg = None, (self.reg_lo + self.reg_hi) / 2
            for bar in p.bars:
                if bar['chord'] is None:
                    prev = None
                    continue
                r = bar['raw']
                k = rel_chord(bar['chord'], p.tonic)
                if prev is None:
                    sur = -math.log((uni[k] + .5) / (n_all + .5 * K))
                else:
                    sur = -math.log((bi[(prev, k)] + .5) / (prev_count[prev] + .5 * K))
                r['sorpresa'] = 0.7 * min(1.0, sur / 5.0) + 0.3 * min(1.0, r['melstep'] / 12.0)
                prev = k
                if r['regmean'] is None:
                    r['regmean'] = last_reg
                last_reg = r['regmean']
                for v in self.RANKED:
                    vals[v].append(r[v])
        self.sorted = {v: sorted(x) for v, x in vals.items()}

    def rank(self, v, x):
        arr = self.sorted[v]
        n = len(arr)
        return (bisect.bisect_left(arr, x) + bisect.bisect_right(arr, x)) / 2.0 / n if n else .5

    def values(self, raw):
        out = {v: self.rank(v, raw[v]) for v in self.RANKED}
        out['registro'] = clamp((raw['regmean'] - self.reg_lo) / (self.reg_hi - self.reg_lo))
        return out


def to_levels(values, L):
    return {v: min(L - 1, int(values[v] * L)) for v in SLIDERS}


def transpose_chord(chord, shift, tonic):
    """Transpone un acorde y recalcula su numeral respecto a la nueva tónica."""
    root = (chord['root'] + shift) % 12
    ivs = [(pc - chord['root']) % 12 for pc in chord['pitches']]
    return make_chord(root, chord['q'], tonic, ivs)


class Reference:
    """MIDI de referencia: guía armónica, melódica y rítmica por compás.

    `transpose` desplaza las alturas cuando el usuario fija una --tonic distinta; los
    numerales se recalculan respecto a la tónica resultante. Solo lectura tras la carga."""

    def __init__(self, p, transpose, tonic, mode):
        self.tonic, self.mode = tonic, mode
        self.beats, self.bpm = p.beats, p.bpm
        self.desc, self.nbars, self.path = p.desc, p.nbars, p.path
        norm = Norm([p])
        bars, prev = [], None
        for bar in p.bars:
            chord = bar['chord']
            if chord is not None:
                chord = transpose_chord(chord, transpose, tonic) if transpose else chord
                prev = chord
            notes = [{'start': n['start'], 'dur': n['dur'], 'midi': n['midi'] + transpose}
                     for n in bar['notes']]
            harm = [{'start': n['start'], 'dur': n['dur'], 'midi': n['midi'] + transpose}
                    for n in bar.get('harm', [])]
            sliders = norm.values(bar['raw']) if bar.get('raw') else None
            bars.append({'chord': chord, 'notes': notes, 'harm': harm, 'sliders': sliders})
        for b in bars:                                        # heredar el acorde anterior
            if b['chord'] is None:
                b['chord'] = prev
            else:
                prev = b['chord']
        if bars and bars[0]['chord'] is None:                 # sin acorde: tónica del modo
            bars[0]['chord'] = make_chord(tonic, 'm' if MODE_INTERVALS[mode][2] == 3 else '', tonic)
            for b in bars:
                if b['chord'] is None:
                    b['chord'] = bars[0]['chord']
        self.bars = bars
        allm = [n['midi'] for b in bars for n in b['notes']]
        self.mel_mean = sum(allm) / len(allm) if allm else 60.0
        self.pc_hist, self.iv_hist, self.dur_by_pos = Counter(), Counter(), {}
        prev_m = None
        for b in bars:
            for n in b['notes']:
                self.pc_hist[(n['midi'] - tonic) % 12] += 1
                if prev_m is not None:
                    self.iv_hist[max(-12, min(12, n['midi'] - prev_m))] += 1
                prev_m = n['midi']
                self.dur_by_pos.setdefault(round((n['start'] % 1.0) * 4) / 4, Counter())[n['dur']] += 1

    def bar(self, idx, loop=True):
        if idx < self.nbars:
            return self.bars[idx]
        return self.bars[idx % self.nbars] if loop and self.nbars else None

    def note_at(self, b, local, tol=0.125):
        for n in self.bars[b]['notes']:
            if abs(n['start'] - local) <= tol:
                return n
        return None


def resolve_eng(args):
    """Resuelve --desc/--tonic/--mode/--beats/--bpm: explícito > referencia > default.
    Devuelve el `Reference` cargado (o None)."""
    refs = getattr(args, 'references', None)
    if refs:
        if getattr(args, 'reference', None):
            fail("--reference y --references son incompatibles")
        if not args.ref_fader:
            fail("--references necesita --ref-fader NOMBRE")
        args.reference = refs[0]
    if getattr(args, 'reference', None):
        ns = argparse.Namespace(beats=None, melody_track=args.ref_melody_track, quant=0.25,
                                desc=None, key=None, mode=None)
        p = analyze_midi(args.reference, ns)
        user_tonic = user_hint = None
        if args.tonic:
            user_tonic, user_hint = parse_key(args.tonic)
        user_mode = args.mode or user_hint
        tonic = p.tonic if user_tonic is None else user_tonic
        mode = p.mode if user_mode is None else user_mode
        shift = 0
        if user_tonic is not None:
            d = (user_tonic - p.tonic) % 12
            shift = d - 12 if d > 6 else d                         # sentido más corto
        ref = Reference(p, shift, tonic, mode)
        args.tonic, args.mode = PITCH_NAMES[tonic], mode
        if args.desc is None:
            args.desc = ref.desc
        if args.beats is None:
            args.beats = ref.beats
        if args.bpm is None and args.ref_tempo and ref.bpm:
            args.bpm = ref.bpm
        ref2 = None
        if refs:                                              # 2ª referencia: misma tónica y modo
            ns2 = argparse.Namespace(beats=args.beats, melody_track=args.ref_melody_track,
                                     quant=0.25, desc=None, key=None, mode=None)
            p2 = analyze_midi(refs[1], ns2)
            d2 = (tonic - p2.tonic) % 12
            ref2 = Reference(p2, d2 - 12 if d2 > 6 else d2, tonic, mode)
        return ref, ref2
    if args.tonic is None:
        args.tonic = 'C'
    tu, th = parse_key(args.tonic)
    args.tonic = PITCH_NAMES[tu]
    if args.mode is None:
        args.mode = th or 'major'
    if args.desc is None:
        args.desc = 'obra instrumental'
    if args.beats is None:
        args.beats = 4
    return None, None


def piece_queries(p, norm, P, stats):
    """(Query, índice de la respuesta real) en el mismo orden y con las mismas ventanas
    que el motor en vivo: primero el acorde del compás, luego cada nota."""
    mem = Memory(P['depth'])
    L = P['levels']
    copts = chord_options(p.tonic, p.mode)
    cidx = {o['label']: i for i, o in enumerate(copts)}
    for bar in p.bars:
        if bar['chord'] is None:
            stats['compases sin acorde'] += 1
            continue
        values = norm.values(bar['raw'])
        levels = to_levels(values, L)
        b = bar['b']
        q = Query('acorde', p.desc, p.tonic, p.mode, p.beats, b, levels, L, list(mem.window()), copts,
                  dur=p.beats)
        yield q, cidx[bar['label']]
        chord = bar['chord']
        entry = mem.add(chord, b)
        center = int(clamp(round(norm.reg_lo + values['registro'] * (norm.reg_hi - norm.reg_lo)), 30, 100))
        for n in bar['notes']:
            opts = note_options(P, p.tonic, p.mode, chord, mem.last_midi, center)
            idx = next((i for i, o in enumerate(opts) if o['midi'] == n['midi']), None)
            if idx is None:
                stats['notas fuera de candidatas'] += 1
            else:
                yield Query('nota', p.desc, p.tonic, p.mode, p.beats, b, levels, L,
                            list(mem.window()), opts, beat=n['start'], dur=n['dur'], cur=chord,
                            center=center), idx
            entry['notes'].append((n['midi'], n['dur']))
            mem.last_midi = n['midi']


def gather_midis(path):
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if not p.is_dir():
        fail(f"No existe: {path}")
    files = sorted(str(f) for f in p.rglob('*') if f.suffix.lower() in ('.mid', '.midi'))
    if not files:
        fail(f"no hay .mid/.midi en {path}")
    return files


def fmt_time(s):
    s = int(max(0, s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


class Progress:
    """Barra de progreso con porcentaje, velocidad y ETA. En una terminal se redibuja en la
    misma línea; si la salida va a un fichero/pipe imprime una línea cada ~10 %."""

    def __init__(self, label, total, unit='', width=26, min_dt=0.15, tty_only=False):
        self.tty_only = tty_only
        self.label, self.total, self.unit, self.width, self.min_dt = label, max(1, total), unit, width, min_dt
        self.t0, self.last, self.tty = time.time(), 0.0, sys.stdout.isatty()
        self.n, self.next_pct, self.extra, self.shown = 0, 10, '', False

    def update(self, n, extra=''):
        self.n, self.extra = n, extra
        now = time.time()
        if self.tty:
            if now - self.last < self.min_dt and n < self.total:
                return
            self.last = now
            sys.stdout.write('\r' + self._line() + '\033[K')
            sys.stdout.flush()
            self.shown = True
        elif not self.tty_only:
            pct = 100.0 * n / self.total
            if pct >= self.next_pct or n >= self.total:
                self.next_pct = (int(pct) // 10 + 1) * 10
                print(self._line(), flush=True)

    def _line(self):
        frac = min(1.0, self.n / self.total)
        el = time.time() - self.t0
        rate = self.n / el if el > 0 else 0.0
        eta = (self.total - self.n) / rate if rate > 0 else 0.0
        fill = int(round(frac * self.width))
        return (f"{self.label} [{'█' * fill}{'░' * (self.width - fill)}] {100 * frac:5.1f}% "
                f"{self.n}/{self.total}{self.unit}  {rate:.0f}/s  {fmt_time(el)}<{fmt_time(eta)}"
                + (f"  {self.extra}" if self.extra else ''))

    def done(self, summary=''):
        if self.tty and self.shown:
            sys.stdout.write('\r\033[K')
        if summary:
            print(summary, flush=True)


def analyze_corpus(a):
    pieces = []
    files = gather_midis(a.dir)
    prog = Progress('[analyze]', len(files), ' MIDIs')
    for k, f in enumerate(files, 1):
        prog.update(k - 1, os.path.basename(f)[:30])
        try:
            p = analyze_midi(f, a)
        except SystemExit:
            warn(f"{f}: omitido")
            continue
        if sum(1 for b in p.bars if b['chord'] is not None) == 0:
            warn(f"{p.name}: no se detectó ningún acorde, omitido")
            continue
        pieces.append(p)
    prog.done(f"[analyze] {len(pieces)}/{len(files)} MIDIs utilizables en {fmt_time(time.time() - prog.t0)}")
    if not pieces:
        fail("ningún MIDI utilizable")
    return pieces, Norm(pieces)


def _pcs_hist(p):
    c = Counter()
    for b in p.bars:
        for n in b['notes']:
            c[(n['midi'] - p.tonic) % 12] += 1
    return np.array([c.get(i, 0) for i in range(12)], dtype=float)


def _iv_hist(p):
    c, prev = Counter(), None
    for b in p.bars:
        for n in b['notes']:
            if prev is not None:
                c[max(-12, min(12, n['midi'] - prev))] += 1
            prev = n['midi']
    return c


def _grid_hist(p):
    c = Counter()
    for b in p.bars:
        for n in b['notes']:
            c[(round(n['start'] * 4), round(n['dur'] * 4))] += 1
    return c


def _cos(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0


def _hist_cos(ca, cb, keys=None):
    keys = keys if keys is not None else (set(ca) | set(cb))
    keys = sorted(keys)
    return _cos([ca.get(k, 0) for k in keys], [cb.get(k, 0) for k in keys])


def _edit_sim(sa, sb):
    if not sa and not sb:
        return 1.0
    d = list(range(len(sb) + 1))
    for i, x in enumerate(sa, 1):
        nd = [i]
        for j, y in enumerate(sb, 1):
            nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (x != y)))
        d = nd
    return 1.0 - d[-1] / max(len(sa), len(sb), 1)


def cmd_compare(a):
    ns = argparse.Namespace(beats=a.beats, melody_track=a.melody_track, quant=0.25,
                            desc=None, key=a.key, mode=a.mode)
    r, o = analyze_midi(a.ref, ns), analyze_midi(a.out, ns)
    pcs = _cos(_pcs_hist(r), _pcs_hist(o))
    ivs = _hist_cos(_iv_hist(r), _iv_hist(o))
    grid = _hist_cos(_grid_hist(r), _grid_hist(o))
    rc = [rel_chord(b['chord'], r.tonic) for b in r.bars if b['chord']]
    oc = [rel_chord(b['chord'], o.tonic) for b in o.bars if b['chord']]
    m = min(len(rc), len(oc))
    chords = _edit_sim(rc[:m], oc[:m])
    if a.verbose:
        print('  ref acordes:', ' '.join(rc))
        print('  out acordes:', ' '.join(oc))
    print(f"ref: {r.nbars} compases, {PITCH_NAMES[r.tonic]} {r.mode}, {r.beats}/4")
    print(f"out: {o.nbars} compases, {PITCH_NAMES[o.tonic]} {o.mode}, {o.beats}/4")
    print(f"  clases de altura (coseno):   {pcs:.3f}")
    print(f"  intervalos melódicos:        {ivs:.3f}")
    print(f"  onsets/duraciones (rejilla): {grid:.3f}")
    print(f"  secuencia de acordes:        {chords:.3f}")


def cmd_analyze(a):
    pieces, norm = analyze_corpus(a)
    L = a.levels
    print(f"\nRegistro (p5..p95 de la altura media de la melodía): {norm.reg_lo:.1f}..{norm.reg_hi:.1f}\n")
    for p in pieces:
        n_ok = sum(1 for b in p.bars if b['chord'] is not None)
        print(f"■ {p.name}: {PITCH_NAMES[p.tonic]} {p.mode}, {p.nbars} compases "
              f"({n_ok} con acorde), {p.beats}/4 — «{p.desc}»")
        print("   c.  acorde      " + ' '.join(s[:4].ljust(4) for s in SLIDERS))
        for bar in p.bars[:a.show_bars]:
            if bar['chord'] is None:
                print(f"  {bar['b'] + 1:>3}  —")
                continue
            lv = to_levels(norm.values(bar['raw']), L)
            print(f"  {bar['b'] + 1:>3}  {bar['chord']['numeral'] + ' ' + bar['chord']['name']:<11} "
                  + ' '.join(str(lv[s]).ljust(4) for s in SLIDERS))
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def fit_temperature(logits, targets):
    best_tau, best = 1.0, float('inf')
    for tau in np.exp(np.linspace(math.log(0.2), math.log(5.0), 61)):
        ce = 0.0
        for s, t in zip(logits, targets):
            z = s / tau
            z = z - z.max()
            ce += float(-(t * (z - math.log(np.exp(z).sum()))).sum())
        if ce < best:
            best_tau, best = float(tau), ce
    return best_tau


def cmd_train(a):
    pieces, norm = analyze_corpus(a)
    P = dict(DEFAULT_PARAMS, levels=a.levels, depth=a.depth, leap=a.leap, half_width=a.half_width,
             reg_lo=round(norm.reg_lo, 2), reg_hi=round(norm.reg_hi, 2))
    stats, samples, dump = Counter(), [], None
    if a.dump_queries:
        os.makedirs(os.path.dirname(os.path.abspath(a.dump_queries)), exist_ok=True)
        dump = open(a.dump_queries, 'w', encoding='utf-8')
    gprog = Progress('[consultas]', len(pieces), ' obras')
    for gi, p in enumerate(pieces):
        gprog.update(gi, f"{p.name[:24]}  {len(samples)} consultas")
        for q, idx in piece_queries(p, norm, P, stats):
            qi, oi, optr = featurize(q, a.bits)
            samples.append((qi, oi, optr, idx, len(q.options), KINDS.index(q.kind), gi, q.bar))
            stats['consultas ' + q.kind] += 1
            if dump:
                dump.write(json.dumps({'piece': p.name, 'kind': q.kind, 'state': q.state_text(),
                                       'instructions': q.instructions(), 'criteria': q.criteria(),
                                       'truth': q.options[idx]['label']}, ensure_ascii=False) + '\n')
    gprog.done()
    if dump:
        dump.close()
    print(f"[train] {len(pieces)} MIDI(s), {len(samples)} consultas: "
          + ', '.join(f"{k} {v}" for k, v in sorted(stats.items())))
    if len(samples) < 20:
        fail("muy pocas consultas para entrenar")

    rng = np.random.default_rng(a.seed)
    ng = len(pieces)
    if a.reference_mode and ng < 3 and a.val_frac > 0:
        val = []
        for gi, p in enumerate(pieces):
            th = (1.0 - a.val_frac) * p.nbars                # bloques de compases contiguos
            val += [i for i, s in enumerate(samples) if s[6] == gi and s[7] >= th]
        if val:
            print(f"[train] --reference-mode: validación por bloques de compases (últimos "
                  f"{100 * a.val_frac:.0f}%), sin calibrar la temperatura")
        else:
            warn("--reference-mode: no quedan compases para validar")
    elif ng >= 3 and a.val_frac > 0:
        nv = max(1, int(round(a.val_frac * ng)))
        vg = set(rng.permutation(ng)[:nv].tolist())
        val = [i for i, s in enumerate(samples) if s[6] in vg]
        print(f"[train] validación por obra: {nv} de {ng} MIDIs")
    elif a.val_frac > 0:
        val = [i for i in range(len(samples)) if rng.random() < a.val_frac]
        warn("pocos MIDIs: la validación se reparte por consulta (no es independiente)")
    else:
        val = []
    vs = set(val)
    train = [i for i in range(len(samples)) if i not in vs]
    if not val:
        warn("sin validación: la parada anticipada usa la pérdida de train")

    mdir = Path(a.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    model = TwoTower(a.bits, a.dim, a.seed)
    model.init_optimizer()

    def target(s):
        t = np.full(s[4], a.label_smoothing / s[4], dtype=np.float32)
        t[s[3]] += 1.0 - a.label_smoothing
        return t

    def evaluate(idx):
        ce, top, uni = {}, {}, {}
        for i in idx:
            s = samples[i]
            sc, _ = model.forward(s[0], s[1], s[2])
            p = softmax(sc)
            k = KINDS[s[5]]
            ce.setdefault(k, []).append(-math.log(max(p[s[3]], 1e-12)))
            top.setdefault(k, []).append(float(np.argmax(p) == s[3]))
            uni.setdefault(k, []).append(math.log(s[4]))
        return ce, top, uni

    best_val, bad, best_ep = float('inf'), 0, 0
    hist, t_start = [], time.time()
    for ep in range(a.epochs):
        t0 = time.time()
        order = rng.permutation(train)
        tot = 0.0
        eprog = Progress(f"[época {ep + 1:>{len(str(a.epochs))}}/{a.epochs}]", len(order), ' consultas',
                         min_dt=0.25, tty_only=True)
        for b in range(0, len(order), a.batch_size):
            batch = order[b:b + a.batch_size]
            eprog.update(b, f"ce {tot / max(1, b):.3f}" if b else '')
            acc = {k: [] for k in TwoTower.PARAMS}
            for i in batch:
                s = samples[i]
                t = target(s)
                sc, ctx = model.forward(s[0], s[1], s[2])
                p = softmax(sc)
                tot += -float((t * np.log(np.maximum(p, 1e-12))).sum())
                model.backward((p - t) / len(batch), s[0], s[1], ctx, acc)
            model.apply(acc, a.lr, a.l2)
        train_ce = tot / len(train)
        eprog.done()
        if val:
            ce, top, _ = evaluate(val)
            vloss = float(np.mean([x for v in ce.values() for x in v]))
            vacc = float(np.mean([x for v in top.values() for x in v]))
        else:
            vloss, vacc = train_ce, float('nan')
        improved = vloss < best_val - 1e-5
        if improved:
            best_val, bad, best_ep = vloss, 0, ep + 1
            model.save(str(mdir / BEST_NAME))
        else:
            bad += 1
        hist.append({'epoch': ep + 1, 'train_ce': train_ce, 'val_ce': vloss, 'val_top1': vacc})
        print(f"[train] época {ep + 1:3d}/{a.epochs}  train_ce {train_ce:.4f}  val_ce {vloss:.4f}  "
              f"acierto {vacc:.3f} {'*' if improved else ' '} {time.time() - t0:.1f}s  "
              f"(mejor época {best_ep}, paciencia {bad}/{a.patience}, "
              f"ETA total {fmt_time((time.time() - t_start) / (ep + 1) * (a.epochs - ep - 1))})")
        if a.patience and bad >= a.patience:
            print(f"[train] Parada anticipada: {bad} épocas sin mejora")
            break
    if not (mdir / BEST_NAME).exists():
        model.save(str(mdir / BEST_NAME))
    best = TwoTower.load(mdir / BEST_NAME, a.bits, a.dim)
    tau = 1.0
    if val and not a.no_calibrate and not (a.reference_mode and ng < 3):
        lg, tg = [], []
        for i in val:
            s = samples[i]
            lg.append(best.forward(s[0], s[1], s[2])[0])
            tg.append(np.eye(s[4], dtype=np.float32)[s[3]])
        tau = fit_temperature(lg, tg)
        print(f"[train] Temperatura calibrada τ = {tau:.3f}")
    cfg = {'bits': a.bits, 'dim': a.dim, 'calibration_temperature': tau, 'params': P,
           'best_epoch': best_ep, 'best_val_ce': best_val, 'n_pieces': len(pieces),
           'n_queries': dict(stats), 'lr': a.lr, 'l2': a.l2,
           'descriptions': sorted({p.desc for p in pieces})[:50]}
    json.dump(cfg, open(mdir / CONFIG_NAME, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(hist, open(mdir / 'train_history.json', 'w', encoding='utf-8'), indent=1)
    if val:
        model = best
        ce, top, uni = evaluate(val)
        print("\n  Validación por tipo (CE del modelo vs. uniforme, acierto del argmax):")
        for k in ce:
            print(f"    {k:<7} n={len(ce[k]):<6} CE {np.mean(ce[k]):.3f}  "
                  f"uniforme {np.mean(uni[k]):.3f}  acierto {np.mean(top[k]):.3f}")
    print(f"\n[train] Hecho en {time.time() - t_start:.0f}s → {mdir}/")


# ═══════════════════════════════════════════════════════════════════════════════
# SALIDAS: puerto MIDI virtual, FluidSynth, registro y despachador con tokens
# ═══════════════════════════════════════════════════════════════════════════════

class VirtualPortBackend:
    def __init__(self, name):
        import mido
        self.mido, self.out = mido, mido.open_output(name, virtual=True)

    def send(self, m):
        mm, k = self.mido.Message, m[0]
        msg = (mm("note_on", channel=m[1], note=m[2], velocity=m[3]) if k == "on" else
               mm("note_off", channel=m[1], note=m[2], velocity=0) if k == "off" else
               mm("control_change", channel=m[1], control=m[2], value=m[3]) if k == "cc" else
               mm("program_change", channel=m[1], program=m[2]))
        self.out.send(msg)

    def close(self):
        self.out.close()


SF2_PATHS = ["/usr/share/sounds/sf2/FluidR3_GM.sf2", "/usr/share/soundfonts/FluidR3_GM.sf2",
             "/usr/share/sounds/sf2/default-GM.sf2"]


class FluidBackend:
    def __init__(self, soundfont=None, driver="pulseaudio", gain=0.7):
        import fluidsynth
        sf = soundfont or next((p for p in SF2_PATHS if os.path.exists(p)), None)
        if not sf:
            raise RuntimeError("No encuentro un .sf2 (apt install fluid-soundfont-gm o --soundfont)")
        self.fs = fluidsynth.Synth(gain=gain, samplerate=44100.0)
        try:
            self.fs.setting("audio.period-size", 128)
        except Exception:
            pass
        self.fs.start(driver=driver)
        sfid = self.fs.sfload(sf)
        for ch in range(16):
            self.fs.program_select(ch, sfid, 128 if ch == 9 else 0, 0)

    def send(self, m):
        k = m[0]
        if k == "on":
            self.fs.noteon(m[1], m[2], m[3])
        elif k == "off":
            self.fs.noteoff(m[1], m[2])
        elif k == "cc":
            self.fs.cc(m[1], m[2], m[3])
        else:
            self.fs.program_change(m[1], m[2])

    def close(self):
        self.fs.delete()


class MultiBackend:
    def __init__(self, backends):
        self.backends = list(backends)

    def send(self, m):
        for b in self.backends:
            b.send(m)

    def close(self):
        for b in self.backends:
            try:
                b.close()
            except Exception:
                pass


class Out:
    """Salida con tokens (cada off solo apaga SU nota) y registro opcional para .mid."""

    def __init__(self, backend=None, record=False):
        self.backend, self.record = backend, record
        self.lock = threading.RLock()
        self.sounding, self.log, self.tempos, self.programs = {}, [], [], {}
        self.meter = (4, 4)

    def _be(self, m):
        if self.backend:
            self.backend.send(m)

    def _log(self, beat, kind, ch, n, v):
        if self.record:
            self.log.append((beat, kind, ch, n, v))

    def note_on(self, beat, ch, n, v, tok):
        with self.lock:
            if (ch, n) in self.sounding:
                self._be(("off", ch, n))
                self._log(beat, 'off', ch, n, 0)
            self.sounding[(ch, n)] = tok
            self._be(("on", ch, n, v))
            self._log(beat, 'on', ch, n, v)

    def note_off(self, beat, ch, n, tok):
        with self.lock:
            if self.sounding.get((ch, n)) == tok:
                del self.sounding[(ch, n)]
                self._be(("off", ch, n))
                self._log(beat, 'off', ch, n, 0)

    def silence(self, beat=0.0, ch=None):
        with self.lock:
            for k in [k for k in self.sounding if ch is None or k[0] == ch]:
                del self.sounding[k]
                self._be(("off", k[0], k[1]))
                self._log(beat, 'off', k[0], k[1], 0)

    def panic(self, beat=0.0):
        self.silence(beat)
        for ch in range(16):
            self._be(("cc", ch, 123, 0))

    def program(self, ch, p):
        self.programs[ch] = int(p)
        self._be(("pc", ch, int(p)))

    def cc(self, ch, c, v):
        self._be(("cc", ch, int(c), int(v)))

    def tempo(self, beat, bpm):
        if self.record:
            self.tempos.append((beat, bpm))


class Scheduler:
    def __init__(self, out):
        self.heap, self.n, self.out = [], 0, out
        self.lock = threading.Lock()

    def add(self, beat, kind, ch, note, vel=0, tok=None):
        with self.lock:
            self.n += 1
            heapq.heappush(self.heap, (beat, 0 if kind == 'off' else 1, self.n, kind, ch, note, vel, tok))

    def pump(self, beat):
        due = []
        with self.lock:
            while self.heap and self.heap[0][0] <= beat + 1e-9:
                due.append(heapq.heappop(self.heap))
        for b, _, _, kind, ch, note, vel, tok in due:
            if kind == 'on':
                self.out.note_on(b, ch, note, vel, tok)
            else:
                self.out.note_off(b, ch, note, tok)

    def silence(self, beat=0.0, ch=None):
        self.out.silence(beat, ch)

    def panic(self, beat=0.0):
        with self.lock:
            self.heap.clear()
        self.out.panic(beat)


def _vlq(n):
    n = max(0, int(n))
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def write_smf(path, out, tpb=480):
    """MIDI tipo 1 (pista de tempo + una por canal), sin depender de mido."""
    def chunk(tag, data):
        return tag + len(data).to_bytes(4, 'big') + data

    def track(evs):
        evs.sort(key=lambda e: (e[0], e[1]))
        data, last = b'', 0
        for tick, _, raw in evs:
            data += _vlq(tick - last) + raw
            last = tick
        return chunk(b'MTrk', data + _vlq(0) + b'\xff\x2f\x00')

    tempos = sorted(out.tempos) or [(0.0, 100.0)]
    tev = [(int(round(b * tpb)), 0, b'\xff\x51\x03' + int(60_000_000 / bpm).to_bytes(3, 'big'))
           for b, bpm in tempos]
    num, den = getattr(out, 'meter', (4, 4))
    tev.insert(0, (0, -1, b'\xff\x58\x04' + bytes([max(1, num), int(math.log2(max(1, den))),
                                                   24, 8])))
    tracks = [track(tev)]
    for ch in sorted({e[2] for e in out.log}):
        evs = []
        if ch in out.programs:
            evs.append((0, -1, bytes([0xC0 | ch, out.programs[ch]])))
        for beat, kind, c, note, vel in out.log:
            if c == ch:
                evs.append((int(round(beat * tpb)), 0 if kind == 'off' else 1,
                            bytes([(0x80 if kind == 'off' else 0x90) | ch, note, vel])))
        tracks.append(track(evs))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(chunk(b'MThd', (1).to_bytes(2, 'big') + len(tracks).to_bytes(2, 'big')
                      + tpb.to_bytes(2, 'big')) + b''.join(tracks))


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRADAS: Korg nanoKONTROL2 (sliders) y teclado MIDI (acordes)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_MAP = {
    "faders": list(range(0, 8)), "knobs": list(range(16, 24)), "solo": list(range(32, 40)),
    "mute": list(range(48, 56)), "rec": list(range(64, 72)),
    "transport": {"play": 41, "stop": 42, "rew": 43, "ff": 44, "rec": 45, "cycle": 46},
    "nav": {"track_prev": 58, "track_next": 59, "marker_set": 60, "marker_prev": 61,
            "marker_next": 62},
}


class Panel:
    """8 faders con suavizado y bloqueo por tira + 8 potenciómetros + botones (callbacks)."""

    def __init__(self, cc_map=None, tau=0.08):
        self.map, self.tau = cc_map or DEFAULT_MAP, tau
        self.lock = threading.RLock()
        self.target, self.value = dict(SLIDER_DEFAULTS), dict(SLIDER_DEFAULTS)
        self.moved = set()              # faders movidos desde la última take_moved() (modo curvas)
        self.override = None            # callable(nombre) -> float | None: sustituye al slider (modo curvas)
        self.knob = {k: 0.5 for k in KNOBS}
        self.locked = {k: False for k in SLIDERS}
        self.callbacks, self.toggles = {}, {}
        self.fcc = {cc: SLIDERS[i] for i, cc in enumerate(self.map["faders"][:8])}
        self.kcc = {cc: KNOBS[i] for i, cc in enumerate(self.map["knobs"][:8])}
        self.mcc = {cc: SLIDERS[i] for i, cc in enumerate(self.map["mute"][:8])}

    def handle_cc(self, cc, raw):
        x = clamp(raw / 127.0)
        with self.lock:
            if cc in self.fcc:
                nm = self.fcc[cc]
                if not self.locked[nm]:
                    if x != self.target[nm]:
                        self.moved.add(nm)
                    self.target[nm] = x
            elif cc in self.kcc:
                self.knob[self.kcc[cc]] = x
            elif cc in self.mcc:
                if raw >= 64:
                    k = self.mcc[cc]
                    self.locked[k] = not self.locked[k]
            elif raw >= 64:
                fn = self.callbacks.get(cc)
                if fn:
                    fn()

    def get(self, name):
        if self.override is not None and name in SLIDERS:
            v = self.override(name)
            if v is not None:
                return v
        with self.lock:
            return self.value[name] if name in self.value else self.knob[name]

    def phys(self, name):
        """Posición física (sin suavizar ni sustituir) de un fader."""
        with self.lock:
            return self.target[name]

    def take_moved(self):
        """Faders movidos desde la última llamada (y los olvida)."""
        with self.lock:
            m, self.moved = self.moved, set()
            return m

    def peek_moved(self):
        with self.lock:
            return set(self.moved)

    def set(self, name, v):
        v = clamp(float(v))
        with self.lock:
            if name in SLIDERS:
                if not self.locked[name]:
                    if v != self.target[name]:
                        self.moved.add(name)
                    self.target[name] = v
            elif name in KNOBS:
                self.knob[name] = v

    def tick(self, dt):
        a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        with self.lock:
            for k in SLIDERS:
                self.value[k] += (self.target[k] - self.value[k]) * a


class KorgInput:
    def __init__(self, panel, hint="nanoKONTROL2", rate=200.0):
        self.panel, self.hint, self.dt = panel, hint, 1.0 / rate
        self._run, self._port, self.on_cc = False, None, None

    def start(self):
        import mido
        names = [n for n in mido.get_input_names() if self.hint in n]
        if not names:
            raise RuntimeError(f"No hay puerto con '{self.hint}'. Disponibles: {mido.get_input_names()}")
        self._port = mido.open_input(names[0], callback=self._on_msg)
        self.start_headless()

    def start_headless(self):
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False
        if self._port:
            self._port.close()

    def _on_msg(self, msg):
        if msg.type == "control_change":
            self.panel.handle_cc(msg.control, msg.value)
            if self.on_cc:
                self.on_cc(msg.control, msg.value)

    def _loop(self):
        while self._run:
            self.panel.tick(self.dt)
            time.sleep(self.dt)


# ═══════════════════════════════════════════════════════════════════════════════
# MODO CURVAS: sliders y acordes desde fichero + grabación iterativa por vueltas
# ═══════════════════════════════════════════════════════════════════════════════

CURVE_INTERP = {'lineal': 'linear', 'linear': 'linear', 'escalonada': 'step', 'step': 'step',
                'suave': 'smooth', 'smooth': 'smooth'}


def _cp(curve):
    return [list(r) if isinstance(r, list) else r for r in curve]


def _strip_comment(line):
    """Quita un comentario «# …» (al inicio de línea o tras un espacio: «F#m7» no es un comentario)."""
    return re.sub(r'(?:^|\s)#.*$', '', line).strip()


def _open_curve(path):
    try:
        return open(path, encoding='utf-8')
    except OSError as e:
        fail(f"no puedo abrir la curva '{path}': {e}")


def _fill_holes(rows, nbars, path):
    if max(rows) > nbars:
        warn(f"{path}: tiene compases hasta el {max(rows)} y la vuelta es de {nbars}; se ignoran los sobrantes")
    out, last = [], rows[min(rows)]
    for b in range(1, nbars + 1):
        last = rows.get(b, last)
        out.append(list(last) if isinstance(last, list) else last)
    return out


def load_curve(path, nbars):
    """Lee «compás: v1 … v8» (compás desde 1; '#' comenta). Devuelve nbars filas de 8 valores.
    Los compases ausentes repiten la línea anterior (los previos a la primera, la primera)."""
    rows = {}
    with _open_curve(path) as f:
        for n, line in enumerate(f, 1):
            line = _strip_comment(line)
            if not line:
                continue
            m = re.match(r'^(\d+)\s*:\s*(.*)$', line)
            if not m:
                fail(f"{path}:{n}: línea no válida (se espera «compás: " + ' '.join(['v'] * len(SLIDERS)) + "»)")
            try:
                vals = [float(x) for x in m.group(2).replace(',', ' ').split()]
            except ValueError:
                fail(f"{path}:{n}: valor no numérico")
            if len(vals) != len(SLIDERS):
                fail(f"{path}:{n}: {len(vals)} valores; se esperan {len(SLIDERS)} ({' '.join(SLIDERS)})")
            if int(m.group(1)) < 1:
                fail(f"{path}:{n}: los compases empiezan en 1")
            rows[int(m.group(1))] = [clamp(v) for v in vals]
    if not rows:
        fail(f"la curva '{path}' no tiene ninguna línea")
    return _fill_holes(rows, nbars, path)


def save_curve(path, curve, interp):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# jev_realtime curva — {len(curve)} compases, interpolación {interp}\n")
        f.write("# compás: " + ' '.join(SLIDERS) + "\n")
        for i, row in enumerate(curve, 1):
            f.write(f"{i}: " + ' '.join(f"{v:.4f}" for v in row) + "\n")


def chord_text(rq):
    return '-' if rq is None else f"{PITCH_NAMES[rq[0]]}{rq[1]}"


def load_chord_curve(path, nbars):
    """Lee «compás: acorde» (Am, G7, F#m7, Bdim…; «-» = lo genera Jev). Los compases ausentes
    repiten el acorde de la línea anterior. Devuelve nbars entradas (raíz, calidad) | None."""
    rows = {}
    with _open_curve(path) as f:
        for n, line in enumerate(f, 1):
            line = _strip_comment(line)
            if not line:
                continue
            m = re.match(r'^(\d+)\s*:\s*(\S+)$', line)
            if not m:
                fail(f"{path}:{n}: línea no válida (se espera «compás: acorde»)")
            if int(m.group(1)) < 1:
                fail(f"{path}:{n}: los compases empiezan en 1")
            if m.group(2) == '-':
                rows[int(m.group(1))] = None
                continue
            rq = parse_chord_name(m.group(2))
            if rq is None:
                fail(f"{path}:{n}: acorde no reconocido: '{m.group(2)}'")
            rows[int(m.group(1))] = rq
    if not rows:
        fail(f"la curva de acordes '{path}' no tiene ninguna línea")
    return _fill_holes(rows, nbars, path)


def save_chord_curve(path, curve):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"# jev_realtime acordes — {len(curve)} compases\n")
        f.write("# compás: acorde   («-» = lo genera Jev)\n")
        for i, rq in enumerate(curve, 1):
            f.write(f"{i}: {chord_text(rq)}\n")


class LapStore:
    """Curvas por vuelta de una pista. `curves[lap]` es la que SUENA en la vuelta `lap`;
    `rec` es la que se GRABA durante la vuelta `rec_lap` (reloj) y sonará en la siguiente."""

    def __init__(self, nbars, base, out, saver, label):
        self.n, self.saver, self.label = nbars, saver, label
        if '%' in out:
            self.tpl = out
        else:
            p = Path(out)
            self.tpl = str(p.with_name(f"{p.stem}_%04d{p.suffix or '.txt'}"))
        self.counter, self.last_saved = 1, None
        self.curves = {0: base}
        self.rec = _cp(base)
        self.rec_lap = 0

    def curve_for(self, lap):
        c = self.curves.get(lap)
        if c is None:
            if lap == self.rec_lap + 1:            # el generador se adelanta al reloj: congelar ya
                c = self.curves[lap] = _cp(self.rec)
            else:
                c = self.curves[max(self.curves)]
        return c

    def finish(self, lap):
        """Fin de la vuelta `lap` (reloj): la grabación pasa a ser la curva de la siguiente."""
        cur = _cp(self.rec)
        self.curves[lap + 1] = cur
        while Path(self.tpl % self.counter).exists():
            self.counter += 1
        path = self.tpl % self.counter
        try:
            self.saver(path, cur)
            self.last_saved = path
            print(f"\n{self.label} guardada: {path}")
        except OSError as e:
            warn(f"no se pudo guardar '{path}': {e}")
        self.counter += 1
        self.rec = _cp(cur)
        self.rec_lap = lap + 1
        for old in [l for l in self.curves if l < lap]:
            del self.curves[old]


class SliderTrack:
    """Los 8 sliders leídos de la curva. Un fader movido durante el compás k del reloj fija el
    compás k+1 de la curva nueva (el valor que tenga al acabar k); sin mover, se conserva."""

    def __init__(self, panel, nbars, interp, out, init=None):
        self.panel, self.n, self.interp = panel, nbars, interp
        self.col = {s: i for i, s in enumerate(SLIDERS)}
        base = init or [[SLIDER_DEFAULTS[s] for s in SLIDERS] for _ in range(nbars)]
        self.store = LapStore(nbars, base, out, lambda p, c: save_curve(p, c, interp), 'Curva de sliders')

    def value(self, name, pos):
        """Valor del slider en la posición `pos` (compases desde el inicio, fraccionario)."""
        lap, x = divmod(max(pos, 0.0) + EPS, self.n)
        cur, j = self.store.curve_for(int(lap)), self.col[name]
        i = min(int(x), self.n - 1)
        a = cur[i][j]
        if self.interp == 'step' or i + 1 >= self.n:
            return a
        t, b = x - i, cur[i + 1][j]
        if self.interp == 'linear':
            return a + (b - a) * t
        def tang(k):                               # tangente monótona (Fritsch–Butland)
            if k <= 0:
                return cur[1][j] - cur[0][j]
            if k >= self.n - 1:
                return cur[k][j] - cur[k - 1][j]
            p, q = cur[k][j] - cur[k - 1][j], cur[k + 1][j] - cur[k][j]
            return 0.0 if p * q <= 0 else 2 * p * q / (p + q)
        m0, m1, t2, t3 = tang(i), tang(i + 1), t * t, t * t * t
        return clamp((2 * t3 - 3 * t2 + 1) * a + (t3 - 2 * t2 + t) * m0
                     + (-2 * t3 + 3 * t2) * b + (t3 - t2) * m1)

    def leave_bar(self, pb):
        moved = self.panel.take_moved()
        k = pb % self.n
        if k + 1 < self.n:                         # en el último compás no hay «siguiente» en el fichero
            for nm in moved:
                self.store.rec[k + 1][self.col[nm]] = round(self.panel.phys(nm), 4)

    def indicator(self):
        marks = ''.join(str(self.col[s] + 1) for s in sorted(self.panel.peek_moved(), key=self.col.get))
        return self.interp + (f" ✎{marks}" if marks else '')


class ChordTrack:
    """Un acorde por compás leído de la curva. Un acorde pulsado se asigna al compás cuya
    frontera (su inicio) esté más cerca: en la 1ª mitad del compás vale para ese compás, en la
    2ª para el siguiente. Si no se pulsa nada se conserva el de la curva; las entradas «-» (o sin
    curva) las genera Jev y se registran como valor del fichero."""

    def __init__(self, nbars, beats, out, init=None):
        self.n, self.beats, self.last = nbars, beats, None
        self.store = LapStore(nbars, init or [None] * nbars, out, save_chord_curve, 'Curva de acordes')

    def chord_at(self, bar):
        lap, k = divmod(bar, self.n)
        return self.store.curve_for(lap)[k]

    def press(self, t, rq):
        """Pulsación en el pulso `t` del reloj → entrada de la curva en grabación."""
        x = max(t, 0.0) / self.beats
        g = int(x)
        target = g if x - g < 0.5 else g + 1
        k = target % self.n
        self.store.rec[k] = (rq[0], rq[1])
        self.last = (k + 1, chord_text(rq))

    def fill(self, bar, rq):
        k = bar % self.n
        if self.store.rec[k] is None:
            self.store.rec[k] = rq

    def leave_bar(self, pb):
        pass

    def indicator(self):
        return f"♪{self.last[1]}→c.{self.last[0]}" if self.last else ''


class CurveSession:
    """Coordina las pistas (sliders y/o acordes) con el RELOJ: vueltas de N compases."""

    def __init__(self, nbars, beats, sliders=None, chords=None):
        self.n, self.beats, self.sliders, self.chords = nbars, beats, sliders, chords
        self.tracks = [t for t in (sliders, chords) if t is not None]
        self.cb = None

    def tick(self, now):
        cb = int(now // self.beats + EPS)
        if self.cb is None:
            self.cb = cb
            if self.sliders is not None:
                self.sliders.panel.take_moved()
            return
        while self.cb < cb:
            pb = self.cb
            for t in self.tracks:
                t.leave_bar(pb)
            if pb % self.n == self.n - 1:
                for t in self.tracks:
                    t.store.finish(pb // self.n)
                if self.chords is not None:
                    self.chords.last = None
            self.cb += 1

    def indicator(self):
        cb = self.cb or 0
        parts = [f"↻{cb // self.n + 1} c.{cb % self.n + 1}/{self.n}"] + [t.indicator() for t in self.tracks]
        return ' '.join(p for p in parts if p)


class ChordKeyboard:
    """Teclado MIDI → acorde actual. Las notas pulsadas casi a la vez forman un acorde;
    al soltar se conserva (latch) hasta el siguiente. Admite pedal de sustain (CC 64).
    get() → (versión, (raíz, calidad) | None)."""

    def __init__(self, hint=None, settle=0.05, thru=None):
        self.hint, self.settle, self.thru = hint, settle, thru
        self.held, self.sus_notes, self.burst = set(), set(), set()
        self.sus, self.last_event, self.last_on = False, 0.0, 0.0
        self.pending, self.chord, self.version = False, None, 0
        self.lock, self._run, self._port = threading.Lock(), False, None
        self.record_events, self.events = False, []      # pulsaciones (t monotónico, acorde), aunque repitan acorde
        self.new_on, self.burst_t = False, 0.0

    def start(self):
        import mido
        names = [n for n in mido.get_input_names() if 'nanoKONTROL' not in n and 'Through' not in n
                 and 'RT Composer' not in n and 'Jev' not in n]
        if self.hint:
            names = [n for n in mido.get_input_names() if self.hint in n]
        if not names:
            raise RuntimeError(f"No encuentro el teclado MIDI. Entradas: {mido.get_input_names()} "
                               f"(usa --kb-port)")
        self._port = mido.open_input(names[0], callback=self._on_msg)
        print(f"Teclado de acordes: {names[0]}")
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False
        if self._port:
            self._port.close()

    def set_chord(self, rq):
        with self.lock:
            self.chord, self.version = rq, self.version + 1
            if self.record_events:
                self.events.append((time.monotonic(), rq))

    def take_events(self):
        with self.lock:
            ev, self.events = self.events, []
            return ev

    def _on_msg(self, msg):
        now = time.monotonic()
        with self.lock:
            if msg.type == 'note_on' and msg.velocity > 0:
                if now - self.last_on > self.settle:
                    self.burst = set()
                    self.burst_t = now
                self.new_on = True
                self.burst.add(msg.note)
                self.held.add(msg.note)
                self.last_on = self.last_event = now
                self.pending = True
                if self.thru:
                    self.thru('on', msg.note, msg.velocity)
            elif msg.type in ('note_off', 'note_on'):
                self.held.discard(msg.note)
                if self.sus:
                    self.sus_notes.add(msg.note)
                self.last_event = now
                self.pending = True
                if self.thru and not self.sus:
                    self.thru('off', msg.note, 0)
            elif msg.type == 'control_change' and msg.control == 64:
                self.sus = msg.value >= 64
                if not self.sus:
                    if self.thru:
                        for n in self.sus_notes - self.held:
                            self.thru('off', n, 0)
                    self.sus_notes.clear()

    def _loop(self):
        while self._run:
            time.sleep(0.005)
            with self.lock:
                if not self.pending or time.monotonic() - self.last_event < self.settle:
                    continue
                self.pending = False
                fresh, self.new_on = self.new_on, False
                notes = set(self.held) | (self.sus_notes if self.sus else set())
                if not notes:
                    notes = set(self.burst)
                if not notes:
                    continue
                rq = recognize_chord({n % 12 for n in notes}, min(notes) % 12)
                if rq and fresh and self.record_events:
                    self.events.append((self.burst_t, rq))
                if rq and rq != self.chord:
                    self.chord, self.version = rq, self.version + 1

    def get(self, t=None):
        with self.lock:
            return self.version, self.chord


class ScriptChords:
    """Acordes sin teclado: uno por compás, en ciclo («C Am F G7»)."""

    def __init__(self, names, beats):
        self.seq, self.beats = [], beats
        for n in names.replace(',', ' ').split():
            rq = parse_chord_name(n)
            if rq is None:
                fail(f"acorde no válido en --kb-chords: '{n}'")
            self.seq.append(rq)
        if not self.seq:
            fail("--kb-chords vacío")
        self.manual = None

    def set_chord(self, rq):
        self.manual = rq

    def get(self, t):
        bar = int(t // self.beats + EPS)
        if self.manual is not None:
            return ('m', self.manual), self.manual
        return bar, self.seq[bar % len(self.seq)]


# ═══════════════════════════════════════════════════════════════════════════════
# RELOJES Y CAPA RÍTMICA
# ═══════════════════════════════════════════════════════════════════════════════

class BeatClock:
    def __init__(self, bpm):
        self.lock = threading.Lock()
        self.bpm, self.beat, self.t0, self.running = bpm, 0.0, time.perf_counter(), True

    def _now(self):
        return self.beat + ((time.perf_counter() - self.t0) * self.bpm / 60.0 if self.running else 0.0)

    def now(self):
        with self.lock:
            return self._now()

    def set_bpm(self, bpm):
        with self.lock:
            self.beat, self.t0, self.bpm = self._now(), time.perf_counter(), bpm

    def set_running(self, on):
        with self.lock:
            self.beat, self.t0 = self._now(), time.perf_counter()
            self.running = bool(on)


class VirtualClock:
    def __init__(self, bpm):
        self.bpm, self.beat, self.running = bpm, 0.0, True

    def now(self):
        return self.beat

    def set_bpm(self, bpm):
        self.bpm = bpm

    def set_running(self, on):
        self.running = bool(on)

    def advance(self, dt):
        if self.running:
            self.beat += dt * self.bpm / 60.0


class Rhythm:
    GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

    def sample(self, rng, t, max_dur, density, groove, beats, bar_idx, last_rest, prior=None):
        opb = 0.25 * (16.0 ** density)
        target = math.log2(1.0 / opb)
        ws = []
        for i, d in enumerate(self.GRID):
            if d > max_dur + EPS:
                ws.append(0.0)
                continue
            w = math.exp(-((math.log2(d) - target) ** 2) / (2 * 0.55 ** 2))
            end = (t + d) % 1.0
            if min(end, 1.0 - end) > 0.01 and abs(end - 0.5) > 0.01:
                w *= 0.35
            if abs((t + d) % 0.5) > 0.01:
                w *= 0.3
            if d in (0.75, 1.5, 3.0):
                w *= 0.5 + 1.2 * groove
            if prior is not None:
                w *= max(prior[i], 1e-6)
            ws.append(w)
        if sum(ws) <= 0:
            d = min(max_dur, 0.25)
        else:
            r, acc, d = rng.random() * sum(ws), 0.0, self.GRID[0]
            for dd, w in zip(self.GRID, ws):
                acc += w
                d = dd
                if r <= acc:
                    break
        d = min(d, max_dur)
        rest_p = 0.28 * (1 - density) ** 2
        if bar_idx % 4 == 3 and (t % beats) >= beats - 1.0 - EPS:
            rest_p += 0.35
        return d, (not last_rest and rng.random() < rest_p)


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR EN TIEMPO REAL
# ═══════════════════════════════════════════════════════════════════════════════

COMMANDS = ['cadence', 'modulate', 'play', 'stop', 'driftreset']


class Engine:
    def __init__(self, args, est, panel, clock, out, source=None, ref=None, ref2=None):
        self.args, self.est, self.panel, self.clock, self.out = args, est, panel, clock, out
        self.jev = LiveJev(est)
        self.P = est.params
        self.sched, self.rhythm = Scheduler(out), Rhythm()
        self.rng = random.Random(args.rng)
        self.tonic, self.mode = parse_key(args.tonic)[0], args.mode
        self.desc, self.beats = args.desc, args.beats
        self.source, self.ref = source, ref
        self.ref2, self.multi = ref2, ref2 is not None
        self.cur_ref2, self.lead_ref, self.lead_idx = None, None, None
        self.cur_ref, self.ref_extra_until = None, 0.0
        self.mem = Memory(self.P['depth'])
        self.mel_t, self.cur_bar, self.bar_start = 0.0, -1, 0.0
        self.cur_chord, self.cur_entry, self.kb_seen = None, None, None
        self.levels = None
        self.cur_center, self.octave = 60, 0
        self.last_rest, self.force = False, []
        self.pending_key = None
        self.layers = {l: True for l in LAYERS}
        self.playing, self.demo, self.t_sec = True, False, 0.0
        self.ticks_panel = True
        self.cmds, self.tok = deque(), itertools.count(1)
        self.vc_center = 60.0
        self.script, self.script_i = [], 0
        self.ladder_i = LADDER.index(self.mode) if self.mode in LADDER else 4
        self.bpm = clock.bpm
        self.progs = [int(x) for x in args.programs.split(',')]
        self.drift_name = args.drift_slider
        self.drift_hist = deque(maxlen=8)           # últimos intervalos melódicos reales (semitonos)
        self.drift_off = 0                          # desplazamiento acumulado de la ventana (semitonos, múltiplos de 12)
        self.curve, self.cpos = None, 0.0           # modo curvas (CurveSession) y posición de lectura en compases
        self.chord_kb, self.chord_kb_ver = None, None   # teclado que GRABA acordes (--chord-source curve)
        self.info = {'tempo': self.bpm, 'chord': '-', 'bar': 0}

    def post(self, *c):
        self.cmds.append(c)

    def curve_value(self, name):
        return self.curve.sliders.value(name, self.cpos)

    def poll_chord_kb(self, now):
        """Pulsaciones nuevas del teclado de grabación → [(pulso del reloj, (raíz, calidad))]."""
        kb = self.chord_kb
        if isinstance(kb, ChordKeyboard):
            ev = kb.take_events()
            mono, bpm = time.monotonic(), self.clock.bpm
            return [(now - (mono - ts) * bpm / 60.0, rq) for ts, rq in ev]
        ver, rq = kb.get(now)                        # ScriptChords (--kb-chords): un acorde por compás
        if ver != self.chord_kb_ver:
            self.chord_kb_ver = ver
            return [(now, rq)]
        return []

    def curve_chord(self, bar):
        """Acorde del compás desde la curva; sin entrada, lo genera Jev y se registra."""
        ct = self.curve.chords
        rq = ct.chord_at(bar)
        if rq is not None:
            self.force.clear()
            return make_chord(rq[0], rq[1], self.tonic)
        chord = self.decide_chord(bar)
        ct.fill(bar, (chord['root'], chord['q']))
        return chord

    # ── utilidades ───────────────────────────────────────────────────────────
    def sv(self, name):
        """Valor efectivo de un slider: con --ref-sliders es la trayectoria de la referencia
        más el desplazamiento del fader del usuario; sin él, el fader tal cual."""
        u = self.panel.get(name)
        if name not in SLIDERS or self.ref is None or not self.args.ref_sliders:
            return u
        if self.multi:                              # trayectorias de las dos refs mezcladas por el fader
            sf = clamp(self.panel.get(self.args.ref_fader))
            rvs = []
            for r_, bi_, w_ in ((self.ref, self.cur_ref, sf), (self.ref2, self.cur_ref2, 1.0 - sf)):
                if bi_ is not None and r_.bars[bi_]['sliders']:
                    rvs.append((r_.bars[bi_]['sliders'][name], w_))
            if not rvs:
                return u
            tw = sum(w for _, w in rvs)
            rv = (sum(v * w for v, w in rvs) / tw if tw > 1e-9
                  else sum(v for v, _ in rvs) / len(rvs))
            return clamp(rv + (u - 0.5) * 2.0 * self.args.ref_slider_gain)
        if self.cur_ref is None:
            return u
        rv = self.ref.bars[self.cur_ref]['sliders']
        if not rv:
            return u
        return clamp(rv[name] + (u - 0.5) * 2.0 * self.args.ref_slider_gain)

    def ref_bar(self, bar, ref=None):
        ref = ref or self.ref
        if ref is None:
            return None
        if bar < ref.nbars:
            return bar
        return bar % ref.nbars if self.args.ref_loop else None

    # ── modo --references: pesos (1-top) modelo, top·s ref A, top·(1-s) ref B ──
    def top(self):
        return clamp(self.args.ref_top)

    def ref_weights(self):
        """(w_modelo, w_A, w_B) con s = valor normalizado del fader --ref-fader."""
        top = self.top()
        s = clamp(self.panel.get(self.args.ref_fader))
        return 1.0 - top, top * s, top * (1.0 - s)

    def ref_infos(self):
        """[(ref, índice de compás, peso)] de las referencias activas en este compás."""
        _, wa, wb = self.ref_weights()
        return [(r, bi, w) for r, bi, w in ((self.ref, self.cur_ref, wa), (self.ref2, self.cur_ref2, wb))
                if bi is not None and w > 1e-9]

    def pick_lead(self):
        """Referencia que manda en ritmo 'follow' y acompañamiento fiel este compás (sorteo por peso)."""
        infos = self.ref_infos()
        if not infos:
            self.lead_ref, self.lead_idx = None, None
            return
        r, acc = self.rng.random() * sum(w for _, _, w in infos), 0.0
        for rr, bi, w in infos:
            acc += w
            if r <= acc:
                break
        self.lead_ref, self.lead_idx = rr, bi

    def mix_indicator(self):
        if not self.multi:
            return ''
        w0, wa, wb = self.ref_weights()
        return f" mix J{100 * w0:.0f} A{100 * wa:.0f} B{100 * wb:.0f}"

    def ref_strength(self, dims):
        """Fuerza de la referencia para una decisión, relajada por los sliders que la gobiernan
        (con los sliders en el centro valen la fuerza nominal)."""
        s = self.args.ref_strength
        if self.args.ref_slider_relax:
            dev = sum(abs(self.panel.get(v) - 0.5) * 2.0 for v in dims) / len(dims)
            s *= max(0.0, 1.0 - dev)
        return clamp(s)

    def sample_T(self, s):
        return min(self.temp(), 0.05) if s >= 0.95 else self.temp()

    def gain(self, name):
        return min((2.0 * self.sv(name)) ** 1.3, 1.8)

    def temp(self):
        return self.args.temperature * (0.5 + self.panel.get('temperatura'))

    def levels_now(self):
        L, out = self.P['levels'], {}
        for v in SLIDERS:
            f = self.sv(v) * L - 0.5
            if f <= 0:
                lv = 0
            elif f >= L - 1:
                lv = L - 1
            else:
                lo = int(f)
                lv = lo + 1 if self.rng.random() < f - lo else lo
            out[v] = lv
        if self.drift_name in out:                  # el slider de drift no entra en la query
            out[self.drift_name] = L // 2
        if self.multi and self.args.ref_fader in out:   # ni el fader de mezcla de referencias
            out[self.args.ref_fader] = L // 2
        return out

    def drift_state(self):
        """(d, s, shift) del drift: d = valor − 0.5; s = fuerza 0..1 (0 = sin efecto);
        shift = desplazamiento del centro de la gaussiana en semitonos respecto a la nota actual."""
        if not self.drift_name:
            return 0.0, 0.0, 0.0
        d = self.sv(self.drift_name) - 0.5
        return d, min(1.0, 2.0 * abs(d)), d * self.args.drift_gain

    def drift_indicator(self):
        """Texto compacto para la línea de estado: posición del control (┊ = centro, ● = valor),
        sentido (▲ agudo / ▼ grave / = inactivo), desplazamiento del centro, fuerza, y el
        intervalo medio REAL de las últimas notas (para ver si el drift está surtiendo efecto)."""
        if not self.drift_name:
            return ''
        d, s, shift = self.drift_state()
        n = 11
        pos = max(0, min(n - 1, int(round((n - 1) / 2 + d * (n - 1)))))
        cells = ['·'] * n
        cells[(n - 1) // 2] = '┊'
        cells[pos] = '●'
        arrow = '=' if s < 1e-3 else ('▲' if d > 0 else '▼')
        real = (sum(self.drift_hist) / len(self.drift_hist)) if self.drift_hist else 0.0
        return (f" drift:{self.drift_name[:4]} [{''.join(cells)}] {arrow}{shift:+5.1f}st "
                f"{100 * s:3.0f}% Δ{real:+4.1f} 8ª{self.drift_off // 12:+d}")

    def drift_weights(self, opts, prev):
        """Pesos gaussianos sobre las notas candidatas (None = sin efecto).
        Centro = nota actual + (valor − 0.5) · ganancia (semitonos); σ configurable.
        En el punto medio el efecto es nulo: se eleva la gaussiana a s = min(1, 2·|valor − 0.5|),
        lo que equivale a ensanchar σ a σ/√s (s=0 → uniforme, s=1 → σ configurada)."""
        if not self.drift_name:
            return None
        d, s, shift = self.drift_state()
        if s < 1e-3:
            return None
        mu = prev + shift
        sig = max(0.5, self.args.drift_sigma)
        return [math.exp(-s * (o['midi'] - mu) ** 2 / (2 * sig * sig)) for o in opts]

    def eff_center(self):
        """Centro de la ventana de notas = registro + octavas acumuladas por el drift."""
        return int(clamp(self.cur_center + self.drift_off, 30, 100))

    def drift_octave_step(self, prev):
        """Si el drift sigue empujando y la última nota ya está en el borde de la ventana en ese
        sentido, la ventana salta una octava (hasta --drift-octaves) y la melodía puede seguir."""
        if not self.drift_name or prev is None:
            return
        d, s, _ = self.drift_state()
        if s < 0.05:
            return
        hw = self.note_params()['half_width']
        edge = max(2, hw // 4)
        c = self.eff_center()
        lim = 12 * max(0, int(self.args.drift_octaves))
        if d > 0 and prev >= c + hw - edge and self.drift_off + 12 <= lim and c < 100:
            self.drift_off += 12
        elif d < 0 and prev <= c - hw + edge and self.drift_off - 12 >= -lim and c > 30:
            self.drift_off -= 12

    def drift_relax(self):
        """Con el drift en reposo, la ventana vuelve hacia la octava base a --drift-return st/compás."""
        r = self.args.drift_return
        if not self.drift_name or r <= 0 or self.drift_off == 0 or self.drift_state()[1] >= 0.05:
            return
        step = 12 if r >= 12 else r
        if abs(self.drift_off) <= step:
            self.drift_off = 0
        else:
            self.drift_off -= math.copysign(step, self.drift_off)

    def center(self):
        lo, hi = self.P['reg_lo'], self.P['reg_hi']
        return int(clamp(lo + self.sv('registro') * (hi - lo) + 12 * self.octave, 30, 100))

    def note_params(self):
        return dict(self.P, half_width=max(4, int(self.P['half_width'] * (0.5 + self.panel.get('amplitud')))))

    def query(self, kind, bar, opts, chord=None, beat=0.0, dur=0.0):
        return Query(kind, self.desc, self.tonic, self.mode, self.beats, bar, self.levels,
                     self.P['levels'], list(self.mem.window()), opts, beat=beat, dur=dur,
                     cur=chord, center=self.eff_center())

    # ── salida de eventos ────────────────────────────────────────────────────
    def emit(self, start, dur, ch, note, vel):
        if not self.layers[LAYERS[ch]]:
            return
        g, h = self.sv('groove'), 0.6 * self.panel.get('humanizar')
        sw = g * 0.8 / 6.0
        jit = (self.rng.random() - .5) * 2 * h * 0.02

        def swung(b):
            return b + (sw if abs(b - math.floor(b) - 0.5) < 1e-3 else 0.0)
        s = max(0.0, swung(start) + (jit if ch == 0 else 0.0))
        e = max(swung(start + dur * (1.0 - 0.45 * g if ch == 0 else 0.98)), s + 0.08)
        tok = next(self.tok)
        self.sched.add(s, 'on', ch, note, int(clamp(vel, 1, 127)), tok)
        self.sched.add(e, 'off', ch, note, 0, tok)

    def vel_mel(self, t):
        e, h = self.sv('energia'), 0.6 * self.panel.get('humanizar')
        v = 42 + 62 * e
        fr = t % self.beats
        v += 10 if fr < EPS else 5 if abs(fr - round(fr)) < EPS else 0
        v += (self.rng.random() - .5) * 2 * h * 9
        return clamp(v * self.gain('vol_melodia'), 1, 127)

    def voice(self, pcs):
        out = sorted({min([m for m in range(50, 74) if m % 12 == pc],
                          key=lambda m: abs(m - self.vc_center)) for pc in pcs})
        self.vc_center = 0.5 * self.vc_center + 0.5 * (sum(out) / len(out))
        return out

    def play_bass(self, chord, t, d):
        e = self.sv('energia')
        vb = clamp((55 + 45 * e) * self.gain('vol_bajo'), 1, 127)
        root = 36 + (chord['root'] - 36) % 12
        if e < 0.35 or d < 2:
            self.emit(t, d, 2, root, vb)
        elif e < 0.7:
            self.emit(t, d / 2, 2, root, vb)
            self.emit(t + d / 2, d / 2, 2, root + 7, vb * 0.85)
        else:
            x, i = 0.0, 0
            while x < d - EPS:
                self.emit(t + x, min(1.0, d - x), 2, root if i % 2 == 0 else root + 12, vb)
                x += 1.0
                i += 1

    def play_chord(self, chord, t, d):
        e, c = self.sv('energia'), self.sv('complejidad')
        pcs = list(chord['pitches'])
        if c > 0.7 and len(pcs) < 5:
            pcs.append((pcs[0] + 2) % 12)
        notes = self.voice(pcs)
        vc = clamp((30 + 50 * e) * self.gain('vol_acordes'), 1, 127)
        if e < 0.35:
            for n in notes:
                self.emit(t, d, 1, n, vc)
        else:
            step = 1.0 if e < 0.6 else 0.5
            seq = notes if e < 0.8 else notes + notes[-2:0:-1]
            x, i = 0.0, 0
            while x < d - EPS:
                self.emit(t + x, min(step, d - x), 1, seq[i % len(seq)], vc * (0.8 if x else 1.0))
                x += step
                i += 1
        self.play_bass(chord, t, d)

    # ── referencia: prior de acordes, notas, ritmo y acompañamiento ───────────
    def _chord_option(self, opts, chord):
        """Opción de acorde de la referencia; si no es diatónica, se añade a opts."""
        for o in opts:
            if o['chord']['name'] == chord['name']:
                return o
        c = dict(chord)
        c['label'] = f"{c['numeral']} ({c['name']})"
        c['desc'] = chord_desc(c)
        o = {'label': c['label'], 'desc': c['desc'], 'chord': c}
        opts.append(o)
        return o

    def ref_chord_dist(self, opts, b, dev_v, dev_t, s, ref=None):
        """Prior de acordes: (1-s)·uniforme + s·uno en el de la referencia, multiplicado por los
        términos de calidad/función que activan valencia/tension (sección 3.6.3)."""
        target = self._chord_option(opts, (ref or self.ref).bars[b]['chord'])
        n = len(opts)
        p = [1e-6] * n
        p[opts.index(target)] = 1.0
        p = [(1.0 - s) / n + s * x for x in p]
        if dev_v or dev_t:
            for i, o in enumerate(opts):
                c = o['chord']
                f = 1.0
                if dev_v:
                    f *= math.exp(1.2 * dev_v * (QUAL_VALENCE.get(c['q'], .5) - 0.5))
                if dev_t:
                    f *= math.exp(0.8 * dev_t * (FN_TENSION[c['fn']] - 0.5))
                p[i] *= f
        tot = sum(p)
        return [x / tot for x in p]

    def ref_note_target(self, b, local, center, ref=None):
        """Nota de la referencia en esa posición, desplazada una octava global hacia el centro."""
        ref = ref or self.ref
        m = ref.note_at(b, local, tol=0.25)
        if m is None:
            return None
        return m['midi'] + 12 * round((center - ref.mel_mean) / 12.0)

    def ref_note_dist(self, opts, b, local, target=None, ref=None):
        """Prior de notas: gaussiana sobre la nota de la referencia en esa posición; fuera de
        ella, contorno (intervalos) e histograma de clases de altura de la referencia."""
        if target is not None:
            return [math.exp(-((o['midi'] - target) ** 2) / (2 * 0.7 ** 2)) + 1e-6 for o in opts]
        ref = ref or self.ref
        hp, hi = ref.pc_hist, ref.iv_hist
        thp, thi = sum(hp.values()) or 1, sum(hi.values()) or 1
        prev, out = self.mem.last_midi, []
        for o in opts:
            w = 0.5 + 6.0 * hp.get((o['midi'] - self.tonic) % 12, 0) / thp
            if prev is not None:
                d = max(-12, min(12, o['midi'] - prev))
                w *= 0.3 + 8.0 * hi.get(d, 0) / thi
            out.append(w)
        return out

    def ref_dur_prior(self, b, local, ref=None):
        c = (ref or self.ref).dur_by_pos.get(round((local % 1.0) * 4) / 4)
        if not c:
            return None
        return [c.get(d, 0.0) + 0.02 for d in Rhythm.GRID]

    def ref_rhythm_step(self, t, ref=None, b=None):
        """Ritmo en modo follow: (dur, rest) de la referencia, ajustado por densidad."""
        ref = ref or self.ref
        b = self.cur_ref if b is None else b
        local, bar_end = t - self.bar_start, self.beats
        devd = clamp((self.panel.get('densidad') - 0.5) * 2.0, -1.0, 1.0)
        match = ref.note_at(b, local)
        if match is None:
            if t < self.ref_extra_until - EPS:                 # continuación de una subdivisión
                return None
            nxt = min([n['start'] for n in ref.bars[b]['notes'] if n['start'] > local + 0.125],
                      default=bar_end)
            return max(0.25, nxt - local), True
        dur = match['dur']
        if devd > 0 and dur > 0.5 and self.rng.random() < devd:
            half = max(0.25, round(dur / 2 / 0.25) * 0.25)
            self.ref_extra_until = self.bar_start + local + dur
            return half, False
        if devd < 0 and self.rng.random() < -devd:
            return dur, True
        return dur, False

    def play_ref_accomp(self, b, t, d, ref=None):
        """Toca la armonía de la referencia en los canales 1/2 según tesitura."""
        e = self.sv('energia')
        for n in (ref or self.ref).bars[b]['harm']:
            st = max(0.0, n['start'])
            if st >= d - EPS:
                continue
            dur = min(n['dur'] + min(0.0, n['start']), d - st)   # recorta notas que cruzan el compás
            if dur <= EPS:
                continue
            ch = 2 if n['midi'] < 60 else 1
            v = clamp((40 + 40 * e) * self.gain('vol_bajo' if ch == 2 else 'vol_acordes'), 1, 127)
            self.emit(t + st, dur, ch, n['midi'], v)

    # ── compás: acorde (modelo o teclado) ────────────────────────────────────
    def keyboard_chord(self, t):
        ver, rq = self.source.get(t)
        self.kb_seen = ver
        if rq is None:
            return None
        return make_chord(rq[0], rq[1], self.tonic)

    def decide_chord(self, bar):
        opts = chord_options(self.tonic, self.mode)
        while self.force:
            want = self.force.pop(0)
            for o in opts:
                c = o['chord']
                if (want == 'V7' and c['numeral'] == 'V7') or \
                   (want == 'tonic' and c['numeral'] in ('I', 'i') and len(c['pitches']) == 3):
                    return c
        if self.multi and self.args.ref_chords != 'off':
            return self.decide_chord_multi(bar, opts)
        b = self.ref_bar(bar)
        use_ref = (self.ref is not None and b is not None
                   and self.args.ref_chords != 'off'
                   and not (self.source is not None))
        s = self.ref_strength(['valencia', 'tension', 'complejidad', 'sorpresa']) if use_ref else 0.0
        if use_ref and s > 1e-9 and self.args.ref_chords == 'follow':
            return self.ref.bars[b]['chord']
        q = self.query('acorde', bar, opts, dur=self.beats)
        p = self.jev.ask(q)
        if use_ref and s > 1e-9:
            dev_v = clamp((self.panel.get('valencia') - 0.5) * 2.0)
            dev_t = clamp((self.panel.get('tension') - 0.5) * 2.0)
            pr = self.ref_chord_dist(opts, b, dev_v, dev_t, s)
            p = [a * bb for a, bb in zip(p, pr)]
            tot = sum(p)
            p = [x / tot for x in p] if tot > 1e-30 else [1.0 / len(p)] * len(p)
        return opts[sample(p, self.sample_T(s), self.rng)]['chord']

    def decide_chord_multi(self, bar, opts):
        """Acorde con dos referencias: p_J^(1-top) · prior_A^(top·s) · prior_B^(top·(1-s))."""
        infos = self.ref_infos()
        for r, bi, _ in infos:                       # añadir antes de consultar a Jev (si no son diatónicos)
            self._chord_option(opts, r.bars[bi]['chord'])
        q = self.query('acorde', bar, opts, dur=self.beats)
        p = self.jev.ask(q)
        if infos:
            fd = self.args.ref_fader
            dev_v = 0.0 if fd == 'valencia' else clamp((self.panel.get('valencia') - 0.5) * 2.0)
            dev_t = 0.0 if fd == 'tension' else clamp((self.panel.get('tension') - 0.5) * 2.0)
            items = [(self.ref_chord_dist(opts, bi, dev_v, dev_t, 1.0, r), w) for r, bi, w in infos]
            p = combine_multi(p, 1.0 - self.top(), items)
        return opts[sample(p, self.sample_T(sum(w for _, _, w in infos)), self.rng)]['chord']

    def begin_bar(self, bar):
        t0 = bar * self.beats
        b = self.ref_bar(bar)
        self.cur_ref, self.ref_extra_until = b, 0.0
        if self.multi:
            self.cur_ref2 = self.ref_bar(bar, self.ref2)
            self.pick_lead()
        else:
            self.lead_ref, self.lead_idx = self.ref, b
        if self.pending_key is not None:
            self.tonic, self.pending_key = self.pending_key, None
        if self.args.valence_mode:
            f = self.sv('valencia') * (len(LADDER) - 1)
            if abs(f - self.ladder_i) > 0.65:
                self.ladder_i = int(round(f))
            self.mode = LADDER[self.ladder_i]
        self.cur_bar, self.bar_start = bar, t0
        self.levels = self.levels_now()
        self.cur_center = self.center()
        self.drift_relax()
        if self.source is not None:
            chord = self.keyboard_chord(t0)
            if chord is None:                                   # aún no se ha tocado ningún acorde
                chord = make_chord(self.tonic, 'm' if MODE_INTERVALS[self.mode][2] == 3 else '',
                                   self.tonic)
        elif self.curve is not None and self.curve.chords is not None:
            chord = self.curve_chord(bar)
        else:
            chord = self.decide_chord(bar)
        self.cur_chord = chord
        self.cur_entry = self.mem.add(chord, bar)
        if (self.args.ref_accomp and self.lead_ref is not None and self.lead_idx is not None
                and self.lead_ref.bars[self.lead_idx]['harm']):
            self.play_ref_accomp(self.lead_idx, t0, self.beats, self.lead_ref)
        elif self.source is None:
            self.play_chord(chord, t0, self.beats)
        else:
            self.play_bass(chord, t0, self.beats)
        self.info.update(chord=f"{chord['numeral']} {chord['name']}", bar=bar + 1)

    def refresh_keyboard(self, t):
        ver, rq = self.source.get(t)
        if ver == self.kb_seen or rq is None:
            return
        self.kb_seen = ver
        chord = make_chord(rq[0], rq[1], self.tonic)
        self.cur_chord = chord
        self.cur_entry['chord'] = chord
        self.out.silence(self.clock.now(), 2)
        self.play_bass(chord, t, self.bar_start + self.beats - t)
        self.info.update(chord=f"{chord['numeral']} {chord['name']}")

    # ── nota ─────────────────────────────────────────────────────────────────
    def gen_note(self, t):
        if self.source is not None:
            self.refresh_keyboard(t)
        chord = self.cur_chord
        bar_end = self.bar_start + self.beats
        b = self.cur_ref
        rinfos = []
        if self.multi:
            rinfos = self.ref_infos() if self.args.ref_rhythm != 'off' else []
            use_ref = bool(rinfos)
            s_r = sum(w for _, _, w in rinfos)
            lref, lb = self.lead_ref, self.lead_idx
        else:
            use_ref = (self.ref is not None and b is not None and self.args.ref_rhythm != 'off')
            s_r = self.ref_strength(['densidad', 'groove']) if use_ref else 0.0
            lref, lb = self.ref, b
        step = (self.ref_rhythm_step(t, lref, lb) if use_ref and self.args.ref_rhythm == 'follow'
                and s_r > 1e-9 and lref is not None else None)
        if step is not None:
            dur, rest = step
            if dur <= EPS:
                dur = bar_end - t
        else:
            prior = None
            if use_ref and self.args.ref_rhythm == 'bias' and s_r > 0:
                if self.multi:
                    for r_, bi_, w_ in rinfos:
                        raw = self.ref_dur_prior(bi_, t - self.bar_start, r_)
                        if raw:
                            pw = [x ** w_ for x in raw]
                            prior = pw if prior is None else [a * c for a, c in zip(prior, pw)]
                else:
                    raw = self.ref_dur_prior(b, t - self.bar_start)
                    if raw:
                        prior = [x ** s_r for x in raw]
            dur, rest = self.rhythm.sample(self.rng, t, bar_end - t, self.sv('densidad'),
                                           self.sv('groove'), self.beats, self.cur_bar,
                                           self.last_rest, prior=prior)
            if dur <= EPS:
                dur = bar_end - t
        self.last_rest = rest
        if not rest:
            if self.args.live_sliders:
                self.levels = self.levels_now()
                self.cur_center = self.center()
            self.drift_octave_step(self.mem.last_midi)
            s_n, ref_target, mtargets = 0.0, None, []
            if self.multi:
                if self.args.ref_melody == 'bias' and t >= self.ref_extra_until - EPS:
                    ec = self.eff_center()
                    mtargets = [(r_, bi_, w_, self.ref_note_target(bi_, t - self.bar_start, ec, r_))
                                for r_, bi_, w_ in self.ref_infos()]
                    s_n = sum(m_[2] for m_ in mtargets)
            elif self.ref is not None and b is not None and self.args.ref_melody == 'bias' \
                    and t >= self.ref_extra_until - EPS:
                s_n = self.ref_strength(['valencia', 'tension', 'sorpresa', 'registro'])
                if s_n > 0:
                    ref_target = self.ref_note_target(b, t - self.bar_start, self.eff_center())
            opts = note_options(self.note_params(), self.tonic, self.mode, chord,
                                self.mem.last_midi, self.eff_center())
            for m_ in mtargets:
                if m_[3] is not None and not any(o['midi'] == m_[3] for o in opts):
                    opts.append(note_option(self.tonic, self.mode, chord, self.mem.last_midi, m_[3]))
            if ref_target is not None and not any(o['midi'] == ref_target for o in opts):
                opts.append(note_option(self.tonic, self.mode, chord, self.mem.last_midi, ref_target))
            q = self.query('nota', self.cur_bar, opts, chord, t - self.bar_start, dur)
            probs = self.jev.ask(q)
            if self.multi and mtargets:
                items = [(self.ref_note_dist(opts, bi_, t - self.bar_start, tg_, r_), w_)
                         for r_, bi_, w_, tg_ in mtargets]
                probs = combine_multi(probs, 1.0 - self.top(), items)
            elif s_n > 0:
                pr = self.ref_note_dist(opts, b, t - self.bar_start, ref_target)
                probs = combine(probs, pr, s_n)
            # drift al final: gaussiana de pitch sobre la distribución ya mezclada (Jev + refs)
            w = self.drift_weights(opts, self.mem.last_midi if self.mem.last_midi is not None
                                   else self.eff_center())
            if w is not None:
                probs = [p * x for p, x in zip(probs, w)]
                tot = sum(probs)
                probs = [p / tot for p in probs] if tot > 1e-30 else [1.0 / len(probs)] * len(probs)
            midi = opts[sample(probs, self.sample_T(s_n), self.rng)]['midi']
            if self.drift_name and self.mem.last_midi is not None:
                self.drift_hist.append(midi - self.mem.last_midi)
            self.cur_entry['notes'].append((midi, dur))
            self.mem.last_midi = midi
            ed = (dur / max(1e-6, 1.0 - 0.45 * self.sv('groove'))
                  if step is not None else dur)
            self.emit(t, ed, 0, midi, self.vel_mel(t))
        self.mel_t = round(t + dur, 6)

    def advance(self, upto):
        while self.mel_t < upto - EPS:
            self.cpos = self.mel_t / self.beats
            bar = int(self.mel_t // self.beats + EPS)
            if bar != self.cur_bar:
                self.begin_bar(bar)
            self.gen_note(self.mel_t)

    # ── comandos, guion, demo, tempo ─────────────────────────────────────────
    def handle_cmds(self):
        while self.cmds:
            c = self.cmds.popleft()
            k = c[0]
            if k == 'cadence':
                self.force = ['V7', 'tonic']
            elif k == 'modulate':
                base = self.tonic if self.pending_key is None else self.pending_key
                self.pending_key = (base + 7) % 12
            elif k == 'key':
                base = self.tonic if self.pending_key is None else self.pending_key
                self.pending_key = (base + c[1]) % 12
            elif k == 'driftreset':
                self.drift_off = 0
            elif k == 'octave':
                self.octave = max(-2, min(2, self.octave + c[1]))
            elif k == 'layer':
                self.layers[c[1]] = not self.layers[c[1]]
                if not self.layers[c[1]]:
                    self.sched.silence(self.clock.now(), LAYERS.index(c[1]))
            elif k == 'chord' and self.source is not None:
                self.source.set_chord(c[1])
            elif k == 'chord' and self.chord_kb is not None:
                self.chord_kb.set_chord(c[1])
            elif k == 'play':
                self.playing = not self.playing
                self.clock.set_running(self.playing)
                if not self.playing:
                    self.sched.silence(self.clock.now())
            elif k == 'stop':
                self.playing = False
                self.clock.set_running(False)
                self.sched.panic(self.clock.now())

    def apply_script(self, beat):
        bar = beat / self.beats
        while self.script_i < len(self.script) and self.script[self.script_i]['bar'] <= bar:
            e = self.script[self.script_i]
            for k, v in e.get('set', {}).items():
                if k in COMMANDS:
                    self.post(k)
                else:
                    self.panel.set(k, v)
            for c in e.get('cmd', []):
                self.post(*c)
            self.script_i += 1

    def apply_demo(self):
        for k, p in DEMO_PERIOD.items():
            self.panel.set(k, 0.5 + 0.45 * math.sin(2 * math.pi * self.t_sec / p + sum(map(ord, k)) % 7))

    def update_tempo(self, dt):
        if self.args.bpm:
            tgt = self.args.bpm * (0.8 + 0.4 * self.panel.get('tempo'))
        else:
            tgt = (62 + 78 * self.sv('energia')) * (0.8 + 0.4 * self.panel.get('tempo'))
        self.bpm += (tgt - self.bpm) * (1 - math.exp(-dt / 1.0))
        if abs(self.bpm - self.clock.bpm) > 0.5:
            self.clock.set_bpm(self.bpm)
            self.out.tempo(self.clock.now(), self.bpm)

    def start(self):
        self.out.meter = (self.beats, 4)
        for ch, p in enumerate(self.progs[:3]):
            self.out.program(ch, p)
        self.out.cc(0, 91, 55)
        self.out.cc(1, 91, 65)
        self.out.tempo(0.0, self.bpm)

    def tick(self, now, dt):
        if self.ticks_panel:
            self.panel.tick(dt)
        if self.demo:
            self.t_sec += dt
            self.apply_demo()
        if self.curve is not None:
            self.curve.tick(now)
        self.apply_script(now)
        self.handle_cmds()
        if self.curve is not None and self.curve.chords is not None and self.chord_kb is not None:
            for t, rq in self.poll_chord_kb(now):
                self.curve.chords.press(t, rq)
        self.cpos = now / self.beats
        self.update_tempo(dt)
        self.info['tempo'] = self.bpm
        self.advance(now + self.args.lookahead)
        self.cpos = now / self.beats

    def status_line(self):
        i = self.info
        bars = "▁▂▃▄▅▆▇█"
        meters = "".join(bars[min(7, int(self.sv(n) * 8))] for n in SLIDERS)
        locks = "".join(str(k + 1) for k, n in enumerate(SLIDERS) if self.panel.locked[n])
        lay = "".join(l[0].upper() if self.layers[l] else "·" for l in LAYERS)
        src = ('KB' if self.source is not None else
               'CV' if self.curve is not None and self.curve.chords is not None else 'JV')
        return (f"\r{PITCH_NAMES[self.tonic]:>2} {self.mode[:9]:<9} {i['chord']:<10} {i['tempo']:5.1f}bpm "
                f"c.{i['bar']:<4} [{meters}] {lay} {src}" + ('  ⏸' if not self.playing else '')
                + (f" 🔒{locks}" if locks else "") + self.drift_indicator() + self.mix_indicator()
                + (f"  {self.curve.indicator()}" if self.curve is not None else "") + "  ")


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN Y SUBCOMANDOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_estimator(args):
    if not args.model_dir:
        warn('sin --model-dir: estimador de reglas (no es el Jev local entrenado)')
        return PriorEstimator()
    est = JevLocal(args.model_dir)
    print(f'  modelo: {args.model_dir} — {est.name}')
    return est


def build(args, out, clock, live_kb=False):
    ref, ref2 = resolve_eng(args)
    if ref is not None and args.ref_sliders and args.demo:
        fail("--ref-sliders y --demo son incompatibles")
    if ref is not None and args.chord_source == 'keyboard' and args.ref_chords != 'off':
        warn("con --chord-source keyboard el acorde del teclado manda: --ref-chords se ignora")
    cbars = getattr(args, 'curve_bars', 0) or 0
    csrc = args.chord_source == 'curve'
    sl_on = bool(cbars) and getattr(args, 'curve_sliders', True)
    if cbars:
        if cbars < 2:
            fail("--curve-bars necesita al menos 2 compases")
        if ref is not None and args.ref_sliders and sl_on:
            fail("--curve-bars y --ref-sliders son incompatibles (ambos deciden los sliders)")
        if not sl_on and not csrc:
            fail("--no-curve-sliders sin --chord-source curve no deja nada que leer de fichero")
    elif csrc:
        fail("--chord-source curve necesita --curve-bars")
    if getattr(args, 'curve_in', None) and not sl_on:
        fail("--curve-in necesita --curve-bars (y sliders en modo curva)")
    if getattr(args, 'chord_curve_in', None) and not csrc:
        fail("--chord-curve-in necesita --chord-source curve")
    est = load_estimator(args)
    cc_map = json.load(open(args.map, encoding='utf-8')) if getattr(args, 'map', None) else None
    panel = Panel(cc_map)
    if ref is not None and args.ref_sliders:
        for v in SLIDERS:                                     # fader en 0.5 = «como la referencia»
            panel.value[v] = panel.target[v] = 0.5
    source = None
    if args.chord_source == 'keyboard':
        if args.kb_chords:
            source = ScriptChords(args.kb_chords, args.beats)
        elif live_kb:
            source = ChordKeyboard(args.kb_port)
        else:
            fail("render/bench con --chord-source keyboard necesitan --kb-chords \"C Am F G7\"")
    eng = Engine(args, est, panel, clock, out, source, ref, ref2)
    if cbars:
        if args.lookahead > args.beats:
            warn("--lookahead mayor que el compás: la curva de la vuelta siguiente puede congelarse "
                 "antes de registrar el penúltimo compás")
        sl = ch = None
        print(f"  modo curvas: vuelta de {cbars} compases")
        if sl_on:
            init = load_curve(args.curve_in, cbars) if args.curve_in else None
            sl = SliderTrack(panel, cbars, CURVE_INTERP[args.curve_interp], args.curve_out, init)
            panel.override = eng.curve_value
            print(f"    sliders: {sl.interp}, entrada {args.curve_in or '(valores por defecto)'}, "
                  f"salida {sl.store.tpl}")
        if csrc:
            if args.kb_chords:
                kb = ScriptChords(args.kb_chords, args.beats)
            else:                                   # en render no se arranca: solo «chord X» por script/consola
                kb = ChordKeyboard(args.kb_port if live_kb else None)
                kb.record_events = True
            eng.chord_kb = kb
            init = load_chord_curve(args.chord_curve_in, cbars) if args.chord_curve_in else None
            ch = ChordTrack(cbars, args.beats, args.chord_curve_out, init)
            print(f"    acordes: entrada {args.chord_curve_in or '(los genera Jev)'}, "
                  f"salida {ch.store.tpl}")
        eng.curve = CurveSession(cbars, args.beats, sl, ch)
    return eng


def bind_controls(korg, eng, out, args):
    m, cb = korg.panel.map, korg.panel.callbacks
    for i, layer in enumerate(LAYERS):
        cb[m['solo'][i]] = (lambda l: lambda: eng.post('layer', l))(layer)
    cb[m['rec'][0]] = lambda: eng.post('cadence')
    cb[m['rec'][1]] = lambda: eng.post('modulate')
    tr, nav = m['transport'], m['nav']
    cb[tr['play']] = lambda: eng.post('play')
    cb[tr['stop']] = lambda: eng.post('stop')
    cb[tr['rew']] = lambda: eng.post('octave', -1)
    cb[tr['ff']] = lambda: eng.post('octave', 1)
    cb[nav['track_prev']] = lambda: eng.post('key', -1)
    cb[nav['track_next']] = lambda: eng.post('key', 1)

    def rec():
        if out.record and out.log:
            path = time.strftime('jev_%y%m%d-%H%M%S.mid')
            write_smf(path, out)
            print(f'\nGrabado: {path}')
        else:
            out.record = True
    cb[tr['rec']] = rec


def stdin_thread(eng, stop):
    def run():
        for line in sys.stdin:
            p = line.split()
            if not p:
                continue
            if p[0] in ('q', 'quit', 'salir'):
                stop.set()
            elif p[0] in COMMANDS:
                eng.post(p[0])
            elif p[0] in ('key', 'octave') and len(p) == 2:
                try:
                    eng.post(p[0], int(p[1]))
                except ValueError:
                    pass
            elif p[0] == 'chord' and len(p) == 2:
                rq = parse_chord_name(p[1])
                if rq:
                    eng.post('chord', rq)
            elif len(p) == 2 and (p[0] in SLIDERS or p[0] in KNOBS):
                try:
                    eng.panel.set(p[0], float(p[1]))
                except ValueError:
                    pass
    threading.Thread(target=run, daemon=True).start()


def cmd_ports(args):
    import mido
    print('Entradas:', mido.get_input_names())
    print('Salidas: ', mido.get_output_names())


def cmd_monitor(args):
    panel = Panel()
    korg = KorgInput(panel, args.korg)
    korg.on_cc = lambda cc, v: print(f'CC {cc:3d} = {v:3d}')
    try:
        korg.start()
    except Exception as e:
        sys.exit(str(e))
    print('Mueve los controles (Ctrl-C para salir)')
    try:
        while True:
            time.sleep(0.25)
            print('  ' + '  '.join(f'{n[:4]}={panel.get(n):.2f}' for n in SLIDERS), end='\r')
    except KeyboardInterrupt:
        korg.stop()


def cmd_run(args):
    if args.no_synth and args.no_virtual:
        sys.exit('Desactivaste ambas salidas.')
    backends = []
    try:
        if not args.no_virtual:
            backends.append(VirtualPortBackend(args.port_name))
            print(f"Puerto MIDI virtual: '{args.port_name}'")
        if not args.no_synth:
            backends.append(FluidBackend(args.soundfont, args.driver))
            print('FluidSynth listo')
    except Exception as e:
        sys.exit(f'Error de salida: {e}')
    backend = MultiBackend(backends)
    out = Out(backend, record=bool(args.record))
    clock = BeatClock(100.0)
    eng = build(args, out, clock, live_kb=True)
    eng.demo, eng.ticks_panel = args.demo, False
    korg = KorgInput(eng.panel, args.korg)
    if args.demo:
        korg.start_headless()
    else:
        try:
            korg.start()
        except Exception as e:
            print(f'  [AVISO] {e}\n  Sin Korg: usa la consola («valencia 0.8», «cadence», «q») o --demo')
            korg.start_headless()
    bind_controls(korg, eng, out, args)
    kbd = (eng.source if isinstance(eng.source, ChordKeyboard)
           else eng.chord_kb if isinstance(eng.chord_kb, ChordKeyboard) else None)
    if kbd is not None:
        if args.thru:
            kbd.thru = lambda k, n, v: (out.note_on(clock.now(), 1, n, max(v, 1), 'kb')
                                        if k == 'on' else out.note_off(clock.now(), 1, n, 'kb'))
        try:
            kbd.start()
        except Exception as e:
            sys.exit(str(e))
    stop = threading.Event()
    stdin_thread(eng, stop)
    eng.start()

    def pump():
        while not stop.is_set():
            eng.sched.pump(clock.now())
            time.sleep(0.001)
    threading.Thread(target=pump, daemon=True).start()
    print('Tocando. Ctrl-C (o «q» + Enter) para salir.')
    last, last_status = time.perf_counter(), 0.0
    try:
        while not stop.is_set():
            now = time.perf_counter()
            dt, last = now - last, now
            if clock.running:
                eng.tick(clock.now(), dt)
            else:
                eng.handle_cmds()
            if now - last_status > 0.2:
                sys.stdout.write(eng.status_line())
                sys.stdout.flush()
                last_status = now
            time.sleep(0.004)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        eng.sched.panic(clock.now())
        if args.record and out.log:
            write_smf(args.record, out)
            print(f'\nGrabado: {args.record}')
        korg.stop()
        if kbd is not None:
            kbd.stop()
        backend.close()
        print('\n\nLatencia de consulta:')
        for line in eng.jev.report():
            print('   ', line)


def run_offline(args, bars=None, seconds=None, until_notes=None):
    out, clock = Out(None, record=True), VirtualClock(100.0)
    eng = build(args, out, clock)
    eng.demo = args.demo
    if getattr(args, 'script', None):
        eng.script = sorted(json.load(open(args.script, encoding='utf-8')), key=lambda e: e['bar'])
    eng.start()
    dt, t, n = 0.02, 0.0, 0
    limit = None if seconds else (bars or args.bars) * args.beats
    while True:
        eng.tick(clock.now(), dt)
        eng.sched.pump(clock.now())
        clock.advance(dt)
        t += dt
        n += 1
        if until_notes:
            if len(eng.jev.t.get('nota', [])) >= until_notes or n > 200000:
                break
        elif (seconds and t >= seconds) or (limit and clock.now() >= limit):
            break
    out.silence(clock.now())
    return eng, out, clock, t


def cmd_render(args):
    eng, out, clock, t = run_offline(args, args.bars, args.seconds)
    write_smf(args.out_file, out)
    n = sum(1 for e in out.log if e[1] == 'on')
    print(f'\n  {n} notas, {clock.now():.1f} pulsos, {t:.1f} s virtuales → {args.out_file}')
    prog = ' '.join(e['chord']['name'] for e in eng.mem.entries[-16:])
    print(f'  últimos acordes: {prog}')
    for line in eng.jev.report():
        print('   ', line)


def cmd_bench(args):
    eng, out, clock, t = run_offline(args, until_notes=args.n)
    for line in eng.jev.report():
        print(line)


# ═══════════════════════════════════════════════════════════════════════════════
# ARGPARSE
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser():
    ana = argparse.ArgumentParser(add_help=False)
    ana.add_argument('dir', metavar='DIR', help='Directorio (o fichero) con los .mid/.midi y sus .txt')
    ana.add_argument('--levels', type=int, default=5, help='Niveles de cada slider (default: 5)')
    ana.add_argument('--depth', type=int, default=4, help='Compases de histórico en la consulta (default: 4)')
    ana.add_argument('--leap', type=int, default=12, help='Salto melódico máximo en semitonos (default: 12)')
    ana.add_argument('--half-width', type=int, default=14, dest='half_width',
                     help='Semiancho de la ventana de registro (default: 14)')
    ana.add_argument('--quant', type=float, default=0.25, help='Rejilla de la melodía en pulsos (default: 0.25)')
    ana.add_argument('--beats', type=int, default=None, help='Pulsos por compás (default: el del MIDI o 4)')
    ana.add_argument('--desc', default=None, help='Descripción para los MIDIs sin .txt')
    ana.add_argument('--key', default=None, help='Fijar la tónica de todos los MIDIs')
    ana.add_argument('--mode', choices=MODES, default=None, help='Fijar el modo de todos los MIDIs')
    ana.add_argument('--melody-track', type=int, default=None, dest='melody_track',
                     help='Forzar la pista (0-based) con la melodía')

    eng = argparse.ArgumentParser(add_help=False)
    eng.add_argument('--model-dir', help='Modelo entrenado (sin él: estimador de reglas)')
    eng.add_argument('--desc', default=None, help='Descripción de la obra (default: el MIDI de referencia o «obra instrumental»)')
    eng.add_argument('--tonic', default=None, help='Tónica (default: la del MIDI de referencia o C)')
    eng.add_argument('--mode', default=None, choices=MODES, help='Modo (default: el del MIDI de referencia o major)')
    eng.add_argument('--beats', type=int, default=None, help='Pulsos por compás (default: el del MIDI de referencia o 4)')
    eng.add_argument('--reference', default=None, metavar='FILE.mid',
                     help='MIDI de referencia: guía la tonalidad, modo, compás, tempo, acordes, melodía y ritmo')
    eng.add_argument('--references', nargs=2, default=None, metavar=('A.mid', 'B.mid'),
                     help='Dos MIDIs de referencia mezclados con Jev: pesos (1-top) Jev, top·s para A y '
                          'top·(1-s) para B, con s = valor del fader --ref-fader. Tonalidad, compás y '
                          'tempo salen de A; B se transpone a la misma tónica. Requiere --ref-fader')
    eng.add_argument('--ref-fader', choices=SLIDERS + KNOBS, default=None, dest='ref_fader',
                     help='Slider/knob que mezcla las dos referencias (1 = solo A, 0 = solo B)')
    eng.add_argument('--ref-top', type=float, default=0.7, dest='ref_top',
                     help='Peso total de las referencias frente a Jev en --references (0 = solo Jev, '
                          '1 = solo MIDIs; default: 0.7). Sustituye a --ref-strength')
    eng.add_argument('--ref-strength', type=float, default=0.6, dest='ref_strength',
                     help='Intensidad con que se sigue la referencia (0 = ignorar, 1 = casi literal; default: 0.6)')
    eng.add_argument('--ref-chords', choices=['follow', 'bias', 'off'], default='follow', dest='ref_chords',
                     help='Acordes: follow = los de la referencia; bias = mezcla con Jev; off (default: follow)')
    eng.add_argument('--ref-melody', choices=['bias', 'off'], default='bias', dest='ref_melody',
                     help='Alturas: bias = mezcla el contorno de la referencia con Jev; off (default: bias)')
    eng.add_argument('--ref-rhythm', choices=['follow', 'bias', 'off'], default='follow', dest='ref_rhythm',
                     help='Ritmo: follow = duraciones y silencios de la referencia; bias; off (default: follow)')
    eng.add_argument('--ref-sliders', action='store_true', dest='ref_sliders',
                     help='Pilotar los 8 sliders con los valores por compás de la referencia')
    eng.add_argument('--ref-slider-gain', type=float, default=1.0, dest='ref_slider_gain',
                     help='Ganancia del desplazamiento del slider sobre la referencia (default: 1.0)')
    eng.add_argument('--ref-slider-relax', dest='ref_slider_relax', action='store_true', default=True,
                     help='Mover un slider reduce la fuerza de la referencia en su decisión (default: on)')
    eng.add_argument('--no-ref-slider-relax', dest='ref_slider_relax', action='store_false')
    eng.add_argument('--ref-tempo', dest='ref_tempo', action='store_true', default=True,
                     help='Usar el tempo del MIDI como --bpm base (default: on)')
    eng.add_argument('--no-ref-tempo', dest='ref_tempo', action='store_false')
    eng.add_argument('--ref-loop', dest='ref_loop', action='store_true', default=True,
                     help='Al acabar la referencia, volver al compás 0 (default: on)')
    eng.add_argument('--no-ref-loop', dest='ref_loop', action='store_false')
    eng.add_argument('--ref-melody-track', type=int, default=None, dest='ref_melody_track',
                     help='Pista (0-based) con la melodía de la referencia')
    eng.add_argument('--ref-accomp', action='store_true', dest='ref_accomp',
                     help='Acompañamiento fiel: toca la armonía de la referencia en vez de las plantillas')
    eng.add_argument('--bpm', type=float, default=None, help='Tempo base fijo (default: lo marca la energía)')
    eng.add_argument('--lookahead', type=float, default=1.5, help='Pulsos de antelación')
    eng.add_argument('--programs', default='0,48,32', help='Programas GM de melodía, acordes y bajo')
    eng.add_argument('--temperature', type=float, default=1.0, help='Temperatura base del sampling')
    eng.add_argument('--valence-mode', action='store_true', dest='valence_mode',
                     help='La valencia cambia el modo (frigio…lidio) en cada compás')
    eng.add_argument('--live-sliders', action='store_true', dest='live_sliders',
                     help='Las notas usan el valor actual de los sliders (no el del inicio del compás)')
    eng.add_argument('--drift-slider', choices=SLIDERS + KNOBS, default=None, dest='drift_slider',
                     help='Slider/knob que actúa como «pitch drift»: en 0.5 no hace nada; por encima '
                          'favorece notas más agudas que la actual, por debajo más graves. En la línea '
                          'de estado muestra posición, sentido, desplazamiento (st), fuerza (%%) y '
                          'Δ = intervalo medio real de las últimas 8 notas')
    eng.add_argument('--drift-gain', type=float, default=12.0, dest='drift_gain',
                     help='Semitonos de desplazamiento del centro por unidad de distancia al medio '
                          '(valor 1.0 = +gain/2 semitonos; default 12)')
    eng.add_argument('--drift-sigma', type=float, default=4.0, dest='drift_sigma',
                     help='Desviación típica (semitonos) de la gaussiana de drift (default 4)')
    eng.add_argument('--drift-octaves', type=int, default=3, dest='drift_octaves',
                     help='Octavas máximas (arriba/abajo) que el drift puede desplazar la ventana de '
                          'notas al llegar a su borde (0 = ventana fija; default 3)')
    eng.add_argument('--drift-return', type=float, default=0.0, dest='drift_return',
                     help='Con el drift en reposo (≈0.5), semitonos por compás con que la ventana vuelve '
                          'a la octava base (0 = se queda donde está; «driftreset» la devuelve; default 0)')
    eng.add_argument('--chord-source', choices=['model', 'keyboard', 'curve'], default='model',
                     dest='chord_source',
                     help='Acorde del compás: generado por Jev (model), leído de un teclado (keyboard) o '
                          'leído de un fichero «compás: acorde» y grabado con el teclado por vueltas (curve; '
                          'requiere --curve-bars)')
    eng.add_argument('--kb-port', default=None, dest='kb_port', help='Fragmento del nombre del teclado MIDI')
    eng.add_argument('--kb-chords', default=None, dest='kb_chords',
                     help='Acordes simulados, uno por compás: "C Am F G7"')
    eng.add_argument('--curve-bars', type=int, default=0, dest='curve_bars',
                     help='Modo curvas: los sliders se leen de un fichero por compás y se graba una curva '
                          'nueva cada N compases (una vuelta). 0 = desactivado (default)')
    eng.add_argument('--curve-in', default=None, dest='curve_in', metavar='FILE',
                     help='Curva inicial «compás: v1 … v8» (sin ella: sliders en su valor por defecto)')
    eng.add_argument('--curve-out', default='curva', dest='curve_out', metavar='NOMBRE',
                     help='Base de los ficheros grabados: NOMBRE_%%04d.txt (o plantilla con %%) (default: curva)')
    eng.add_argument('--curve-interp', choices=list(CURVE_INTERP), default='lineal', dest='curve_interp',
                     help='Interpolación entre compases: lineal | escalonada | suave (default: lineal)')
    eng.add_argument('--no-curve-sliders', dest='curve_sliders', action='store_false', default=True,
                     help='Con --curve-bars: dejar los faders en vivo (solo curva de acordes)')
    eng.add_argument('--chord-curve-in', default=None, dest='chord_curve_in', metavar='FILE',
                     help='Curva de acordes inicial «compás: acorde» (sin ella los genera Jev en la 1ª vuelta)')
    eng.add_argument('--chord-curve-out', default='acordes', dest='chord_curve_out', metavar='NOMBRE',
                     help='Base de los ficheros de acordes grabados: NOMBRE_%%04d.txt (default: acordes)')
    eng.add_argument('--rng', type=int, default=0)

    ap = argparse.ArgumentParser(description='Compositor en tiempo real con Jev local (sliders → acordes y notas).')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('train', parents=[ana], help='Entrenar el Jev local con MIDIs')
    p.add_argument('--model-dir', required=True, dest='model_dir')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=32, dest='batch_size')
    p.add_argument('--dim', type=int, default=32)
    p.add_argument('--bits', type=int, default=16, help='Filas por tabla de rasgos: 2^bits (default: 16)')
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--l2', type=float, default=1e-6)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--val-frac', type=float, default=0.15, dest='val_frac')
    p.add_argument('--label-smoothing', type=float, default=0.02, dest='label_smoothing')
    p.add_argument('--no-calibrate', action='store_true', dest='no_calibrate')
    p.add_argument('--reference-mode', action='store_true', dest='reference_mode',
                   help='Un solo MIDI: validación por bloques de compases contiguos y sin calibrar')
    p.add_argument('--dump-queries', default=None, dest='dump_queries',
                   help='Guardar las consultas generadas (texto legible) en un JSONL')
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser('analyze', parents=[ana], help='Mostrar qué se extrae de cada MIDI')
    p.add_argument('--show-bars', type=int, default=16, dest='show_bars')
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser('run', parents=[eng], help='Tocar en tiempo real')
    p.add_argument('--no-synth', action='store_true')
    p.add_argument('--no-virtual', action='store_true')
    p.add_argument('--soundfont')
    p.add_argument('--driver', default='pulseaudio')
    p.add_argument('--port-name', default='Jev Realtime')
    p.add_argument('--korg', default='nanoKONTROL2')
    p.add_argument('--map', help='JSON con el mapa de CC del Korg')
    p.add_argument('--record', help='Grabar la sesión a este .mid')
    p.add_argument('--thru', action='store_true', help='(keyboard) oír también los acordes que tocas')
    p.add_argument('--demo', action='store_true', help='Sliders en piloto automático')
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('render', parents=[eng], help='Generar un .mid offline')
    p.add_argument('--bars', type=int, default=32)
    p.add_argument('--seconds', type=float)
    p.add_argument('--out-file', default='jev_demo.mid', dest='out_file')
    p.add_argument('--script', help='[{"bar": 8, "set": {"valencia": 0.9, "cadence": 1}}, ...]')
    p.add_argument('--demo', action='store_true')
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser('compare', help='Comparar un .mid generado con el de referencia')
    p.add_argument('ref', metavar='REF.mid')
    p.add_argument('out', metavar='OUT.mid')
    p.add_argument('--beats', type=int, default=None)
    p.add_argument('--key', default=None)
    p.add_argument('--mode', default=None, choices=MODES)
    p.add_argument('--melody-track', type=int, default=None, dest='melody_track')
    p.add_argument('--verbose', action='store_true')
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser('bench', parents=[eng], help='Latencia por consulta')
    p.add_argument('--n', type=int, default=300)
    p.add_argument('--bars', type=int, default=32)
    p.set_defaults(fn=cmd_bench, demo=True)

    p = sub.add_parser('ports')
    p.set_defaults(fn=cmd_ports)
    p = sub.add_parser('monitor')
    p.add_argument('--korg', default='nanoKONTROL2')
    p.set_defaults(fn=cmd_monitor)
    return ap


def main():
    args = build_parser().parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
