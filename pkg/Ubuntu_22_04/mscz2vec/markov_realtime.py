#!/usr/bin/env python3
r"""
markov_realtime.py — Markov RealTime: generador musical interactivo en tiempo real
controlado con un Korg nanoKONTROL2
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Generador CAUSAL paso a paso (semicorchea a semicorchea, 4/4). En cada paso
se muestrea de distribuciones recalculadas con el estado actual de las
variables latentes (faders). Mover un fader no reinicia nada: cambia las
probabilidades de lo que viene después.

  Armonía   máquina de estados sobre grados (I ii iii IV V vi vii) con pesos de
            flujo funcional × cercanía a la tensión objetivo; temperatura =
            previsibilidad. Cambio de acorde en barra/media barra.
  Melodía   muestreo de altura por producto de expertos (escala continua
            según valencia × notas del acorde × salto gaussiano × relleno de
            saltos × registro); ritmo por probabilidad de ataque métrica;
            repetición de motivos del compás anterior.
  Bajo      patrones por nivel de actividad (raíz sostenida → pulso).
  Acordes   voicings con conducción de voces mínima, pad o stabs.
  Batería   probabilidades por paso (bombo, caja, fantasmas, hi-hat/ride).

  Valencia = continuo Lidio→Jónico→Mixolidio→Dórico→Eolio→Frigio (perfil de
  clases de altura interpolado, sin saltos).

────────────────────────────────────────────────────────────────────────────
Controles (nanoKONTROL2, scene por defecto)
────────────────────────────────────────────────────────────────────────────
  Faders 1-8  valence, arousal, density, tension, brightness, predictability,
              syncopation, articulation
  Knobs 1-8   (0.5 = neutro) 1 centro modal · 2 techo de tempo · 3 densidad
              máxima · 4 techo de disonancia · 5 amplitud de registro ·
              6 tendencia a repetir motivos · 7 swing (derecha del centro) ·
              8 humanización
  S 1-4       mute melodía / bajo / acordes / batería (LED = capa activa)
  S 5-8       timbre de melodía: vibráfono / flauta / lead / cuerdas
  M 1-8       bloquear la variable de esa tira
  R 1-5       fill · cadencia (V→I) · modular +5ª · silencio dramático · motivo nuevo
  Play/Stop   pausa-continúa / parada (panic)
  Rew/FF      octava − / +
  Rec         grabar sesión a .mid (con mapa de tempo)
  Cycle       congelar: repite en bucle los últimos 4 compases
  Track ◄ ►   tonalidad −1 / +1 semitono
  Marker      Set guarda escena (scenes.json) · ◄ ► recuperan (con pickup)

────────────────────────────────────────────────────────────────────────────
Uso
────────────────────────────────────────────────────────────────────────────
  python markov_realtime.py                      # Korg → puerto virtual + FluidSynth
  python markov_realtime.py --no-synth           # solo puerto MIDI virtual (a tu DAW)
  python markov_realtime.py --no-virtual         # solo sintetizador interno
  python markov_realtime.py --demo               # sin hardware: faders en piloto automático
  python markov_realtime.py --render demo.mid --bars 64 --demo --seed 1
  python markov_realtime.py --list-ports
  python markov_realtime.py --monitor            # ver CC del Korg y valores (diagnóstico)
  python markov_realtime.py --leds               # LEDs (Korg Kontrol Editor: LED mode = External)

FICHERO ÚNICO: no depende de ningún otro módulo propio.
DEPENDENCIAS externas: mido + python-rtmidi (puertos MIDI) y pyfluidsynth (+ fluidsynth
y un .sf2) para el sintetizador interno. El render offline y las pruebas son Python puro.
"""
import argparse
import bisect
import collections
import heapq
import itertools
import json
import math
import os
import random
import struct
import sys
import threading
import time
from types import SimpleNamespace


# ══════════════════════════════════════════════════════════════════════════
# Entrada del Korg nanoKONTROL2: CC → variables latentes (0..1)
#   suavizado (slew), pickup, bloqueo por tira, toggles, mapa por JSON
# SliderBank es Python puro (se prueba sin hardware); KorgInput abre el puerto.
# ══════════════════════════════════════════════════════════════════════════
FADER_NAMES = ["valence", "arousal", "density", "tension",
               "brightness", "predictability", "syncopation", "articulation"]

DEFAULT_MAP = {
    "faders": list(range(0, 8)),        # CC 0-7
    "knobs":  list(range(16, 24)),      # CC 16-23
    "solo":   list(range(32, 40)),      # S
    "mute":   list(range(48, 56)),      # M  -> bloqueo de la tira
    "rec":    list(range(64, 72)),      # R  -> eventos
    "transport": {"play": 41, "stop": 42, "rew": 43, "ff": 44,
                  "rec": 45, "cycle": 46},
    "nav": {"track_prev": 58, "track_next": 59, "marker_set": 60,
            "marker_prev": 61, "marker_next": 62},
}


class SliderBank:
    """Estado de las 8 variables latentes. Lógica pura, sin MIDI."""

    def __init__(self, names=FADER_NAMES, cc_map=None, tau=0.08,
                 pickup_eps=0.03):
        self.map = cc_map or DEFAULT_MAP
        self.names = list(names)
        self.tau = tau
        self.eps = pickup_eps
        self._lock = threading.RLock()
        n = len(self.names)
        self.target = {k: 0.5 for k in self.names}
        self.value = {k: 0.5 for k in self.names}
        self.knob = {k: 0.5 for k in self.names}
        self.locked = {k: False for k in self.names}
        self.armed = {k: False for k in self.names}   # pickup pendiente
        self._last_in = {k: None for k in self.names}
        self.toggles = {}                             # cc -> bool
        self.callbacks = {}                           # cc -> fn(cc, on)
        self._fader_cc = {cc: self.names[i]
                          for i, cc in enumerate(self.map["faders"][:n])}
        self._knob_cc = {cc: self.names[i]
                         for i, cc in enumerate(self.map["knobs"][:n])}
        self._mute_cc = {cc: self.names[i]
                         for i, cc in enumerate(self.map["mute"][:n])}

    # ── entrada ────────────────────────────────────────────────────────
    def handle_cc(self, cc, raw):
        x = max(0.0, min(1.0, raw / 127.0))
        with self._lock:
            if cc in self._fader_cc:
                self._fader(self._fader_cc[cc], x)
            elif cc in self._knob_cc:
                self.knob[self._knob_cc[cc]] = x
            elif cc in self._mute_cc:
                if raw >= 64:                         # solo al pulsar
                    k = self._mute_cc[cc]
                    self.locked[k] = not self.locked[k]
                    self._fire(cc, self.locked[k])
            else:
                if raw >= 64:                         # botones: toggle
                    self.toggles[cc] = not self.toggles.get(cc, False)
                    self._fire(cc, self.toggles[cc])

    def _fire(self, cc, on):
        fn = self.callbacks.get(cc)
        if fn:
            fn(cc, on)

    def _fader(self, k, x):
        prev = self._last_in[k]
        self._last_in[k] = x
        if self.locked[k]:
            return
        if self.armed[k]:
            cur = self.target[k]
            crossed = prev is not None and (prev - cur) * (x - cur) <= 0
            if abs(x - cur) <= self.eps or crossed:
                self.armed[k] = False                 # pickup logrado
            else:
                return
        self.target[k] = x

    # ── API de escenas / estado ────────────────────────────────────────
    def set_value(self, k, v, pickup=True):
        """Fija un valor (p. ej. al cargar una escena)."""
        with self._lock:
            self.target[k] = float(v)
            self.armed[k] = bool(pickup)

    def snapshot(self):
        with self._lock:
            return dict(self.target)

    def load_scene(self, scene):
        for k, v in scene.items():
            self.set_value(k, v, pickup=True)

    # ── salida ─────────────────────────────────────────────────────────
    def tick(self, dt):
        """Avanza el suavizado dt segundos."""
        a = 1.0 - math.exp(-dt / self.tau) if self.tau > 0 else 1.0
        with self._lock:
            for k in self.names:
                self.value[k] += (self.target[k] - self.value[k]) * a

    def get(self, k):
        with self._lock:
            return self.value[k]


class KorgInput:
    """Puerto real: hilo de suavizado + callback de mido."""

    def __init__(self, port_hint="nanoKONTROL2", map_file=None, rate=200.0):
        cc_map = DEFAULT_MAP
        if map_file:
            with open(map_file) as f:
                cc_map = json.load(f)
        self.bank = SliderBank(cc_map=cc_map)
        self.port_hint = port_hint
        self.dt = 1.0 / rate
        self._run = False
        self._port = None
        self.on_cc = None                             # callback opcional

    def get(self, k):
        return self.bank.get(k)

    def start(self):
        import mido
        names = [n for n in mido.get_input_names() if self.port_hint in n]
        if not names:
            raise RuntimeError(
                f"No hay puerto con '{self.port_hint}'. Disponibles: "
                f"{mido.get_input_names()}")
        self._port = mido.open_input(names[0], callback=self._on_msg)
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def start_headless(self):
        """Solo suavizado, sin puerto MIDI (demo / pruebas)."""
        self._run = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self._run = False
        if self._port:
            self._port.close()

    def _on_msg(self, msg):
        if msg.type == "control_change":
            self.bank.handle_cc(msg.control, msg.value)
            if self.on_cc:
                self.on_cc(msg.control, msg.value)

    def _loop(self):
        while self._run:
            self.bank.tick(self.dt)
            time.sleep(self.dt)


# ══════════════════════════════════════════════════════════════════════════
# Utilidades y teoría
# ══════════════════════════════════════════════════════════════════════════
def clip(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, t):
    return a + (b - a) * t


def wchoice(rng, items, weights, temp=1.0):
    if temp != 1.0:
        weights = [w ** (1.0 / temp) for w in weights]
    s = sum(weights)
    if s <= 0:
        return rng.choice(items)
    r = rng.random() * s
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROMAN = ["I", "ii", "iii", "IV", "V", "vi", "vii"]

# De más brillante a más oscuro (el locrio se excluye: tónica inestable)
MODES = [("lidio",     [0, 2, 4, 6, 7, 9, 11]),
         ("jónico",    [0, 2, 4, 5, 7, 9, 11]),
         ("mixolidio", [0, 2, 4, 5, 7, 9, 10]),
         ("dórico",    [0, 2, 3, 5, 7, 9, 10]),
         ("eolio",     [0, 2, 3, 5, 7, 8, 10]),
         ("frigio",    [0, 1, 3, 5, 7, 8, 10])]

TENSION_DEG = [0.0, 0.4, 0.3, 0.3, 0.7, 0.2, 0.9]
FLOW = {
    0: {3: 2.0, 5: 1.6, 1: 1.2, 4: 1.4, 2: 0.8, 6: 0.3, 0: 0.4},
    1: {4: 3.0, 6: 1.0, 3: 0.8, 0: 0.3, 5: 0.6, 2: 0.4},
    2: {5: 2.0, 3: 1.6, 1: 1.0, 4: 0.6, 0: 0.4},
    3: {4: 2.0, 0: 1.6, 1: 1.0, 5: 0.8, 6: 0.5, 2: 0.3},
    4: {0: 3.0, 5: 1.4, 3: 0.6, 2: 0.3, 1: 0.2},
    5: {3: 2.0, 1: 1.6, 4: 1.0, 2: 0.8, 0: 0.5},
    6: {0: 3.0, 5: 0.8, 2: 0.6, 4: 0.3},
}

METRIC = [1.0, .25, .5, .25, .8, .25, .5, .25, .9, .25, .5, .25, .8, .25, .5, .25]
OFFBEAT = [.35, .5, .9, .75] * 4
KICK = [1.0, 0, .15, 0, .55, 0, .1, .3, .9, 0, .2, 0, .5, 0, .1, .25]
DURS = [1, 2, 3, 4, 6, 8]


def onset_weight(step, sync):
    return (1 - sync) * METRIC[step] + sync * OFFBEAT[step]


def mode_profile(valence):
    """Posición modal continua y pertenencia (0..1) de cada clase de altura."""
    pos = (1.0 - valence) * 5.0
    i = min(int(pos), 4)
    f = pos - i
    a, b = set(MODES[i][1]), set(MODES[i + 1][1])
    return pos, [(1 - f) * (pc in a) + f * (pc in b) for pc in range(12)]


def build_chord(scale, deg, seventh, ninth, alter_dom):
    root = scale[deg]
    ivs = [0, (scale[(deg + 2) % 7] - root) % 12, (scale[(deg + 4) % 7] - root) % 12]
    if seventh:
        ivs.append((scale[(deg + 6) % 7] - root) % 12)
    if ninth:
        ivs.append((scale[(deg + 1) % 7] - root) % 12 + 12)
    if alter_dom and deg == 4:
        ivs[1] = 4                          # V mayor (sensible) en modos menores
    return root, ivs


def voice_chord(root_abs, ivs, prev, center=62):
    """Voicing por inversión/octava con movimiento mínimo respecto al anterior."""
    n = len(ivs)
    best, best_cost = None, 1e9
    for inv in range(n):
        order = ivs[inv:] + [iv + 12 for iv in ivs[:inv]]
        for base in (48, 52, 56, 60):
            low = base + ((root_abs + order[0] - base) % 12)
            v = [low + (o - order[0]) for o in order]
            ref = (sum(prev) / len(prev)) if prev else center
            cost = abs(sum(v) / n - ref)
            if prev:
                cost += 0.5 * sum(abs(a - b) for a, b in zip(v, prev)) / n
            if v[-1] > 77:
                cost += 6
            if v[0] < 48:
                cost += 6
            if cost < best_cost:
                best, best_cost = v, cost
    return best


# ══════════════════════════════════════════════════════════════════════════
# Motor generativo (sin tiempo real: devuelve eventos con retardos en s)
# Mensajes: ("on",ch,nota,vel,tok) ("off",ch,nota,tok) ("cc",ch,cc,val) ("pc",ch,prog)
# ══════════════════════════════════════════════════════════════════════════
class Engine:
    CH = {"melody": 0, "bass": 1, "chords": 2, "drums": 9}
    GM = {"melody": 11, "bass": 33, "chords": 4}
    LAYER_GAIN = {"melody": 1.0, "bass": 0.9, "chords": 0.62, "drums": 0.9}

    def __init__(self, bank, seed=None, key=0):
        self.bank = bank
        self.rng = random.Random(seed)
        self.key = key % 12
        self.step_abs = 0
        self.bar = 0
        self.layers = {k: True for k in self.CH}
        self.octave = 0
        self.playing = True
        self.cmds = collections.deque()
        self.tok = itertools.count(1)
        # armonía
        self.mode_idx = 1
        self.chord = None
        self.chord_new = False
        self.next_chord_step = 0
        self.forced = []
        self.voicing = None
        self.comp_active = []
        # melodía
        self.mel_last = None
        self.mel_leap = 0
        self.mel_dir = 0
        self.mel_busy = 0
        self.mel_live = None
        self.bass_live = None
        self.bar_notes = []
        self.motif = None
        self.replay_map = None
        self.replay_first = None
        # eventos
        self.pending_key = None
        self.fill_bar = None
        self.silence_req = False
        self.silence_left = 0
        self.crash_now = False
        self.freeze_req = False
        self.frozen = False
        self.freeze_i = 0
        self.loop_bars = collections.deque(maxlen=4)
        self.cur_rec = []
        self.last_cc74 = -10
        self.jitter = 0.0
        self.info = {"tempo": 100.0, "mode": "jónico", "chord": "-", "bar": 0,
                     "key": NOTE_NAMES[self.key], "frozen": False}

    # ── comandos desde hilos externos ──────────────────────────────────
    def post(self, *cmd):
        self.cmds.append(cmd)

    def _drain(self, out):
        while self.cmds:
            c = self.cmds.popleft()
            k = c[0]
            if k == "fill":
                self.fill_bar = self.bar if self.step_abs % 16 < 12 else self.bar + 1
            elif k == "cadence":
                self.forced = [4, 0]
            elif k == "modulate":
                self.pending_key = (self.key + 7) % 12
            elif k == "silence":
                self.silence_req = True
            elif k == "motif":
                self.motif = None
                self.replay_map = None
            elif k == "key":
                self.pending_key = (self.key + c[1]) % 12
            elif k == "octave":
                self.octave = max(-2, min(2, self.octave + c[1]))
            elif k == "freeze":
                if self.frozen or self.freeze_req:
                    self.frozen = self.freeze_req = False
                else:
                    self.freeze_req = True
            elif k == "program":
                out.append((0.0, ("pc", self.CH[c[1]], c[2])))
            elif k == "playing":
                self.playing = c[1]

    def setup_events(self):
        ev = [(0.0, ("pc", self.CH[l], p)) for l, p in self.GM.items()]
        ev += [(0.0, ("cc", self.CH["melody"], 91, 55)),
               (0.0, ("cc", self.CH["chords"], 91, 65))]
        return ev

    # ── mapeo perceptual: latentes → parámetros musicales ──────────────
    def derive(self):
        b = self.bank
        f = {k: b.get(k) for k in FADER_NAMES}
        kb = b.knob

        def eff(name, gain):
            return clip(f[name] + (kb[name] - 0.5) * gain)

        P = SimpleNamespace()
        P.val = eff("valence", 0.8)
        P.ar = f["arousal"]
        ceiling = 140.0 + (kb["arousal"] - 0.5) * 100.0
        P.tempo = lerp(52.0, max(70.0, ceiling), P.ar ** 0.9)
        P.vel = 48 + 62 * P.ar
        P.dens = eff("density", 0.8)
        P.ten = eff("tension", 0.6)
        P.bri = f["brightness"]
        P.span = 0.6 + kb["brightness"] * 0.8
        P.pred = f["predictability"]
        P.motif_p = P.pred * (0.3 + 0.7 * kb["predictability"])
        P.temp = 0.35 + 1.4 * (1 - P.pred)
        P.temp_h = 0.4 + 1.2 * (1 - P.pred)
        P.sync = f["syncopation"]
        P.swing = clip((kb["syncopation"] - 0.5) * 0.7, 0.0, 0.35)
        P.art = f["articulation"]
        P.hum = kb["articulation"] * 0.02
        P.pos, member = mode_profile(P.val)
        eps = 0.005 + 0.35 * P.ten ** 2
        pcw = [m + (1 - m) * eps for m in member]
        pcw[0] *= 1.25
        pcw[7] *= 1.1
        P.pcw = pcw
        return P

    # ── emisión de notas ───────────────────────────────────────────────
    def _note(self, out, layer, pitch, vel, steps, sd, gate=1.0, record=True):
        pitch = int(max(0, min(127, pitch)))
        if record:
            self.cur_rec.append((self.step_abs % 16, layer, pitch, vel, steps, gate))
        if not self.layers[layer]:
            return None
        ch = self.CH[layer]
        tok = next(self.tok)
        v = int(clip(vel * self.LAYER_GAIN[layer], 1, 127))
        d = self.rng.uniform(0, 2 * self.jitter) if layer != "drums" else 0.0
        out.append((d, ("on", ch, pitch, v, tok)))
        out.append((d + max(0.03, steps * sd * gate), ("off", ch, pitch, tok)))
        return (ch, pitch, tok)

    def _cut(self, out, h):
        if h:
            out.append((0.0, ("off", h[0], h[1], h[2])))

    def _cut_all(self, out):
        for h in self.comp_active:
            self._cut(out, h)
        self.comp_active = []
        self._cut(out, self.mel_live)
        self._cut(out, self.bass_live)
        self.mel_live = self.bass_live = None

    # ── paso principal ─────────────────────────────────────────────────
    def step(self):
        out = []
        self._drain(out)
        P = self.derive()
        self.jitter = P.hum
        sd = 60.0 / P.tempo / 4.0
        sib = self.step_abs % 16
        if sib == 0:
            self._bar_start(P, out)
        if self.frozen and self.loop_bars:
            for (s, layer, pitch, vel, steps, gate) in self.loop_bars[self.freeze_i]:
                if s == sib:
                    self._note(out, layer, pitch, vel, steps, sd, gate, record=False)
        elif self.silence_left == 0:
            self._harmony(P, out)
            self._comp(P, sd, sib, out)
            self._bass(P, sd, sib, out)
            self._melody(P, sd, sib, out)
            self._drums(P, sd, sib, out)
        cc = int(P.bri * 127)
        if abs(cc - self.last_cc74) >= 2:
            self.last_cc74 = cc
            for l in ("melody", "chords"):
                out.append((0.0, ("cc", self.CH[l], 74, cc)))
        c = self.chord
        self.info = {"tempo": P.tempo, "mode": MODES[self.mode_idx][0],
                     "chord": (f"{ROMAN[c.deg]} {NOTE_NAMES[c.root_abs]}" if c else "-"),
                     "bar": self.bar, "key": NOTE_NAMES[self.key],
                     "frozen": self.frozen or self.freeze_req}
        self.step_abs += 1
        if sib == 15:
            self.bar += 1
        dur = sd * (1 + P.swing) if sib % 2 == 0 else sd * (1 - P.swing)
        return dur, out

    def _bar_start(self, P, out):
        if self.pending_key is not None:
            self.key = self.pending_key
            self.pending_key = None
        if not self.frozen:
            if self.cur_rec:
                self.loop_bars.append(self.cur_rec)
            self.cur_rec = []
            if len(self.bar_notes) >= 3:
                f0 = self.bar_notes[0][1]
                self.motif = [(s, p - f0, d) for s, p, d in self.bar_notes]
            self.bar_notes = []
            if self.motif and self.rng.random() < P.motif_p:
                self.replay_map = {s: (o, d) for s, o, d in self.motif}
            else:
                self.replay_map = None
            self.replay_first = None
        if self.freeze_req and not self.frozen and self.loop_bars:
            self.frozen, self.freeze_i = True, 0
        elif self.frozen and self.loop_bars:
            self.freeze_i = (self.freeze_i + 1) % len(self.loop_bars)
        if self.silence_left > 0:
            self.silence_left -= 1
            if self.silence_left == 0:
                self.crash_now = True
        elif self.silence_req:
            self.silence_left, self.silence_req = 1, False
            self._cut_all(out)

    # ── armonía ────────────────────────────────────────────────────────
    def _harmony(self, P, out):
        forced_now = bool(self.forced) and self.step_abs % 8 == 0
        if self.chord is None or self.step_abs >= self.next_chord_step or forced_now:
            self._change_chord(P, out)

    def _next_degree(self, P):
        cur = self.chord.deg if self.chord else 0
        items = list(range(7))
        ws = []
        for n in items:
            w = FLOW[cur].get(n, 0.15)
            w *= math.exp(-((TENSION_DEG[n] - P.ten) / 0.4) ** 2)
            ws.append(w)
        return wchoice(self.rng, items, ws, P.temp_h)

    def _change_chord(self, P, out):
        if abs(P.pos - self.mode_idx) > 0.65:
            self.mode_idx = max(0, min(5, int(round(P.pos))))
        scale = MODES[self.mode_idx][1]
        deg = self.forced.pop(0) if self.forced else self._next_degree(P)
        seventh = self.rng.random() < 0.1 + 0.75 * P.ten
        ninth = seventh and self.rng.random() < max(0.0, P.ten - 0.4) * 1.2
        alter = P.ten > 0.45 and self.mode_idx >= 3
        root_rel, ivs = build_chord(scale, deg, seventh, ninth, alter)
        root_abs = (self.key + root_rel) % 12
        self.voicing = voice_chord(root_abs, ivs, self.voicing)
        for h in self.comp_active:
            self._cut(out, h)
        self.comp_active = []
        self.chord = SimpleNamespace(deg=deg, root_abs=root_abs, ivs=ivs,
                                     pcs=[(root_abs + iv) % 12 for iv in ivs],
                                     voicing=self.voicing, start=self.step_abs)
        self.next_chord_step = self.step_abs + (32 if P.ar < 0.25 else 16 if P.ar < 0.75 else 8)
        self.chord_new = True

    def _comp(self, P, sd, sib, out):
        c = self.chord
        vel = P.vel * 0.9 + (6 if sib == 0 else 0)
        if self.chord_new:
            self.chord_new = False
            left = self.next_chord_step - self.step_abs
            dur = left * 0.98 if P.dens < 0.4 else min(left, 4)
            for n in c.voicing:
                h = self._note(out, "chords", n, vel, dur, sd)
                if h:
                    self.comp_active.append(h)
        elif P.dens >= 0.4:
            p = clip((P.dens - 0.35) * 1.7) * onset_weight(sib, P.sync)
            p *= 1.0 if sib % 2 == 0 else 0.35
            if self.rng.random() < p:
                for n in c.voicing:
                    h = self._note(out, "chords", n, vel * 0.8, 2, sd, gate=0.9)
                    if h:
                        self.comp_active.append(h)

    # ── bajo ───────────────────────────────────────────────────────────
    def _bass(self, P, sd, sib, out):
        c = self.chord
        rel = (self.step_abs - c.start) % 16
        base = 36 + ((c.root_abs - 36) % 12)
        fifth = base + ((c.pcs[2] - c.root_abs) % 12)
        bd = 0.5 * P.dens + 0.5 * P.ar
        left = self.next_chord_step - self.step_abs
        hit = None
        if bd < 0.25:
            if rel == 0:
                hit = (base, min(left, 16))
        elif bd < 0.5:
            if rel == 0:
                hit = (base, 7)
            elif rel == 8:
                hit = (fifth if self.rng.random() < 0.4 else base, 7)
        elif bd < 0.75:
            hit = {0: (base, 5), 6: (base, 2), 8: (fifth, 4), 12: (base + 12 if self.rng.random() < 0.3 else base, 3)}.get(rel)
        elif rel % 2 == 0:
            alt = [base, base + 12, base, fifth][(rel // 2) % 4]
            hit = (alt, 1.6)
        if hit and rel < 16:
            vel = P.vel * 0.95 + (8 if rel == 0 else 0)
            self._cut(out, self.bass_live)
            self.bass_live = self._note(out, "bass", hit[0], vel, hit[1], sd, gate=0.92)

    # ── melodía ────────────────────────────────────────────────────────
    def _pick_pitch(self, P, sib):
        c = self.chord
        center = int(52 + 28 * P.bri) + 12 * self.octave
        span = int(4 + 8 * P.span)
        last = self.mel_last if self.mel_last is not None else center
        sig = 2.0 + 4.0 * P.ar
        cb = 3.0 if sib % 4 == 0 else 1.8 if sib % 2 == 0 else 1.1
        cb = 1 + (cb - 1) * (1 - 0.6 * P.ten)
        cands, ws = [], []
        for n in range(center - span, center + span + 1):
            w = P.pcw[(n - self.key) % 12]
            if n % 12 in c.pcs:
                w *= cb
            iv = n - last
            a = abs(iv)
            w *= math.exp(-0.5 * (a / sig) ** 2)
            if iv == 0:
                w *= 0.45
            if self.mel_leap >= 5:
                if iv * self.mel_dir > 0:
                    w *= 0.3
                elif 0 < a <= 2:
                    w *= 1.8
            w *= math.exp(-0.5 * ((n - center) / (span * 0.9)) ** 2)
            cands.append(n)
            ws.append(w)
        return wchoice(self.rng, cands, ws, P.temp)

    def _snap(self, target, P):
        c = self.chord
        best, bs = target, -1e9
        for d in (0, 1, -1, 2, -2):
            n = target + d
            s = P.pcw[(n - self.key) % 12] + (0.5 if n % 12 in c.pcs else 0) - 0.15 * abs(d)
            if s > bs:
                best, bs = n, s
        return best

    def _melody(self, P, sd, sib, out):
        if self.replay_map is not None:
            ent = self.replay_map.get(sib)
            if not ent:
                return
            off, dur = ent
            if self.replay_first is None:
                pitch = self._pick_pitch(P, sib)
                self.replay_first = pitch - off
            else:
                pitch = self._snap(self.replay_first + off, P)
        else:
            if self.step_abs < self.mel_busy:
                return
            p = clip(onset_weight(sib, P.sync) * (0.12 + 1.6 * P.dens ** 1.4))
            if self.rng.random() >= p:
                return
            pitch = self._pick_pitch(P, sib)
            target = 3.0 * (1 - P.dens)
            ws = [math.exp(-((math.log2(d) - target) ** 2) / (2 * 0.49)) for d in DURS]
            dur = wchoice(self.rng, DURS, ws)
        if self.mel_last is not None:
            iv = pitch - self.mel_last
            self.mel_leap = abs(iv)
            self.mel_dir = (iv > 0) - (iv < 0)
        self.mel_last = pitch
        self.mel_busy = self.step_abs + dur
        self.bar_notes.append((sib, pitch, dur))
        acc = 12 if sib == 0 else 7 if sib % 4 == 0 else 3 if sib % 2 == 0 else 0
        vel = clip(P.vel + acc + self.rng.randint(-5, 5), 20, 120)
        self._cut(out, self.mel_live)
        self.mel_live = self._note(out, "melody", pitch, vel, dur, sd,
                                   gate=lerp(0.35, 0.98, P.art))

    # ── batería ────────────────────────────────────────────────────────
    def _drums(self, P, sd, sib, out):
        r = self.rng
        ar, d, sy = P.ar, P.dens, P.sync

        def hit(n, v):
            self._note(out, "drums", n, int(clip(v, 15, 127)), 1, sd, gate=0.5)

        if self.crash_now and sib == 0:
            self.crash_now = False
            hit(49, 118)
            hit(36, 115)
            return
        if self.fill_bar == self.bar and sib >= 12:
            hit({12: 50, 13: 48, 14: 45, 15: 38}[sib], 85 + (sib - 12) * 8)
            if sib == 15 and ar > 0.5:
                self.crash_now = True
            return
        if ar < 0.18:
            return
        base = 60 + 40 * ar
        pk = 1.0 if sib == 0 else KICK[sib] * (0.35 + 0.9 * ar)
        if sib in (6, 7, 10, 11, 14, 15):
            pk += sy * 0.3 * ar
        if r.random() < pk:
            hit(36, base + (10 if sib == 0 else 0))
        if sib in (4, 12):
            hit(38 if ar > 0.4 else 37, base - 5 if ar > 0.4 else 55) if r.random() < (0.96 if ar > 0.3 else 0.5) else None
        elif sib in (3, 7, 10, 15) and r.random() < 0.35 * d * (0.3 + sy):
            hit(38, 32 + 10 * r.random())
        ph = clip(0.15 + d) if sib % 2 == 0 else clip(1.2 * d * d - 0.1) * (0.5 + 0.5 * sy)
        if r.random() < ph:
            if ar < 0.35:
                hit(51, base - 15 + (8 if sib % 4 == 0 else 0))
            elif sib == 14 and r.random() < 0.3 * ar:
                hit(46, base - 10)
            else:
                hit(42, base - 18 + (12 if sib % 4 == 0 else 0))
        if sib == 0 and ar > 0.8 and self.bar % 8 == 0 and self.bar > 0:
            hit(49, 105)


# ══════════════════════════════════════════════════════════════════════════
# Backends de salida
# ══════════════════════════════════════════════════════════════════════════
class VirtualPortBackend:
    def __init__(self, name="Mutopia Live"):
        import mido
        self.mido = mido
        self.out = mido.open_output(name, virtual=True)
        self.name = name

    def send(self, m):
        k = m[0]
        mm = self.mido.Message
        if k == "on":
            msg = mm("note_on", channel=m[1], note=m[2], velocity=m[3])
        elif k == "off":
            msg = mm("note_off", channel=m[1], note=m[2], velocity=0)
        elif k == "cc":
            msg = mm("control_change", channel=m[1], control=m[2], value=m[3])
        else:
            msg = mm("program_change", channel=m[1], program=m[2])
        self.out.send(msg)

    def close(self):
        self.out.close()


SF2_PATHS = ["/usr/share/sounds/sf2/FluidR3_GM.sf2",
             "/usr/share/soundfonts/FluidR3_GM.sf2",
             "/usr/share/sounds/sf2/default-GM.sf2"]


class FluidBackend:
    def __init__(self, soundfont=None, driver="pulseaudio", gain=0.7):
        import fluidsynth
        sf = soundfont or next((p for p in SF2_PATHS if os.path.exists(p)), None)
        if not sf:
            raise RuntimeError("No encuentro un .sf2 (apt install fluid-soundfont-gm o usa --soundfont)")
        self.fs = fluidsynth.Synth(gain=gain, samplerate=44100.0)
        try:
            self.fs.setting("audio.period-size", 128)
        except Exception:
            pass
        self.fs.start(driver=driver)
        self.sfid = self.fs.sfload(sf)
        for ch in range(16):
            self.fs.program_select(ch, self.sfid, 128 if ch == 9 else 0, 0)

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


def _varlen(n):
    b = [n & 0x7F]
    n >>= 7
    while n:
        b.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(b))


class Recorder:
    """Graba la sesión y la escribe como SMF tipo 0 con mapa de tempo."""

    def __init__(self):
        self.active = False
        self.t0 = 0.0
        self.now_fn = time.monotonic
        self.events, self.tempos = [], []

    def start(self):
        self.events, self.tempos = [], []
        self.t0 = self.now_fn()
        self.active = True

    def send(self, m):
        if self.active:
            self.events.append((self.now_fn() - self.t0, m))

    def note_tempo(self, t_abs, bpm):
        if self.active:
            self.tempos.append((max(0.0, t_abs - self.t0), bpm))

    def close(self):
        pass

    def write(self, path, tpq=480):
        tempos = sorted(self.tempos) or [(0.0, 100.0)]
        kept = [(0.0, tempos[0][1])]
        for t, b in tempos[1:]:
            if abs(b - kept[-1][1]) > 0.5:
                kept.append((t, b))
        segs, tick = [], 0.0
        for i, (t, b) in enumerate(kept):
            segs.append((t, b, tick))
            if i + 1 < len(kept):
                tick += (kept[i + 1][0] - t) * b / 60.0 * tpq
        times = [s[0] for s in segs]

        def to_tick(t):
            s = segs[max(0, bisect.bisect_right(times, t) - 1)]
            return int(round(s[2] + (t - s[0]) * s[1] / 60.0 * tpq))

        ev = [(to_tick(t), 0, b"\xFF\x51\x03" + int(60e6 / b).to_bytes(3, "big")) for t, b in kept]
        ev.append((0, 0, b"\xFF\x58\x04\x04\x02\x18\x08"))
        for t, m in self.events:
            k = m[0]
            if k == "on":
                ev.append((to_tick(t), 3, bytes([0x90 | m[1], m[2], m[3]])))
            elif k == "off":
                ev.append((to_tick(t), 1, bytes([0x80 | m[1], m[2], 0])))
            elif k == "cc":
                ev.append((to_tick(t), 2, bytes([0xB0 | m[1], m[2], m[3]])))
            else:
                ev.append((to_tick(t), 2, bytes([0xC0 | m[1], m[2]])))
        ev.sort(key=lambda e: (e[0], e[1]))
        body, last = b"", 0
        for tk, _, data in ev:
            body += _varlen(tk - last) + data
            last = tk
        body += _varlen(0) + b"\xFF\x2F\x00"
        with open(path, "wb") as f:
            f.write(b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq))
            f.write(b"MTrk" + struct.pack(">I", len(body)) + body)
        return path


class Dispatcher:
    """Cola temporizada. Los off llevan token: solo apagan su propia nota."""

    def __init__(self, backend):
        self.backend = backend
        self.heap = []
        self.seq = itertools.count()
        self.cv = threading.Condition()
        self.sounding = {}
        self.running = False

    def schedule(self, when, msg):
        with self.cv:
            heapq.heappush(self.heap, (when, next(self.seq), msg))
            self.cv.notify()

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify()

    def _run(self):
        while self.running:
            with self.cv:
                if not self.heap:
                    self.cv.wait(0.05)
                    continue
                when = self.heap[0][0]
                now = time.monotonic()
                if when > now:
                    self.cv.wait(min(when - now, 0.05))
                    continue
                _, _, msg = heapq.heappop(self.heap)
            self._emit(msg)

    def _emit(self, msg):
        k = msg[0]
        if k == "on":
            _, ch, n, v, tok = msg
            if (ch, n) in self.sounding:
                self.backend.send(("off", ch, n))
            self.sounding[(ch, n)] = tok
            self.backend.send(("on", ch, n, v))
        elif k == "off":
            _, ch, n, tok = msg
            if self.sounding.get((ch, n)) == tok:
                del self.sounding[(ch, n)]
                self.backend.send(("off", ch, n))
        else:
            self.backend.send(msg)

    def panic(self):
        with self.cv:
            self.heap.clear()
            items = list(self.sounding)
            self.sounding.clear()
        for ch, n in items:
            self.backend.send(("off", ch, n))
        for ch in range(16):
            self.backend.send(("cc", ch, 123, 0))


class Scheduler(threading.Thread):
    def __init__(self, engine, dispatcher, recorder=None, lookahead=0.12):
        super().__init__(daemon=True)
        self.engine, self.disp, self.rec = engine, dispatcher, recorder
        self.lookahead = lookahead
        self.running = True

    def run(self):
        e = self.engine
        for d, m in e.setup_events():
            self.disp.schedule(time.monotonic() + 0.05 + d, m)
        t = time.monotonic() + 0.3
        while self.running:
            if not e.playing:
                e.cmds and e._drain([])
                time.sleep(0.02)
                t = time.monotonic() + 0.15
                continue
            while self.running and time.monotonic() < t - self.lookahead:
                time.sleep(0.003)
            dur, evs = e.step()
            for d, m in evs:
                self.disp.schedule(t + d, m)
            if self.rec:
                self.rec.note_tempo(t, e.info["tempo"])
            t += dur


# ══════════════════════════════════════════════════════════════════════════
# Piloto automático y render offline
# ══════════════════════════════════════════════════════════════════════════
def make_demo(bank):
    tau = 2 * math.pi
    s = math.sin

    def drive(t):
        tg = bank.target
        tg["valence"] = 0.5 + 0.45 * s(tau * t / 70)
        tg["arousal"] = 0.5 + 0.40 * s(tau * t / 45 + 1.0)
        tg["density"] = 0.5 + 0.40 * s(tau * t / 37 + 2.0)
        tg["tension"] = 0.35 + 0.30 * s(tau * t / 53 + 0.5)
        tg["brightness"] = 0.5 + 0.30 * s(tau * t / 61 + 3.0)
        tg["predictability"] = 0.55 + 0.25 * s(tau * t / 83)
        tg["syncopation"] = 0.35 + 0.30 * s(tau * t / 47 + 4.0)
        tg["articulation"] = 0.5 + 0.35 * s(tau * t / 59 + 5.0)
    return drive


def render_offline(engine, bank, path, bars, driver=None):
    rec = Recorder()
    cur = [0.0]
    rec.now_fn = lambda: cur[0]
    rec.start()
    disp = Dispatcher(rec)
    heap, seq, t = [], itertools.count(), 0.0
    for d, m in engine.setup_events():
        heapq.heappush(heap, (d, next(seq), m))
    for _ in range(bars * 16):
        if driver:
            driver(t)
        dur, evs = engine.step()
        rec.note_tempo(t, engine.info["tempo"])
        for d, m in evs:
            heapq.heappush(heap, (t + d, next(seq), m))
        bank.tick(dur)
        t += dur
    while heap:
        when, _, m = heapq.heappop(heap)
        cur[0] = when
        disp._emit(m)
    cur[0] = t + 1.0
    for (ch, n) in list(disp.sounding):
        rec.send(("off", ch, n))
    rec.write(path)
    return rec, t


# ══════════════════════════════════════════════════════════════════════════
# Controles: enlace del nanoKONTROL2 con el motor
# ══════════════════════════════════════════════════════════════════════════
class Leds:
    def __init__(self, hint="nanoKONTROL2"):
        self.out = None
        try:
            import mido
            names = [n for n in mido.get_output_names() if hint in n]
            if names:
                self.mido = mido
                self.out = mido.open_output(names[0])
        except Exception:
            self.out = None

    def set(self, cc, on):
        if self.out:
            self.out.send(self.mido.Message("control_change", channel=0,
                                            control=cc, value=127 if on else 0))


SCENES_FILE = "scenes.json"
MELODY_PROGRAMS = [11, 73, 81, 48]       # vibráfono, flauta, lead sierra, cuerdas


def bind_controls(korg, engine, dispatcher, recorder, leds, scheduler):
    bank = korg.bank
    m = bank.map
    cb = bank.callbacks
    layer_names = ["melody", "bass", "chords", "drums"]
    scenes = []
    if os.path.exists(SCENES_FILE):
        try:
            scenes = json.load(open(SCENES_FILE))
        except Exception:
            scenes = []
    state = {"scene": -1}

    def toggle_layer(name, cc):
        def f(_cc, _on):
            engine.layers[name] = not engine.layers[name]
            leds.set(cc, engine.layers[name])
        return f

    for i, name in enumerate(layer_names):
        cb[m["solo"][i]] = toggle_layer(name, m["solo"][i])
        leds.set(m["solo"][i], True)
    for i, prog in enumerate(MELODY_PROGRAMS):
        cb[m["solo"][4 + i]] = (lambda p: lambda c, o: engine.post("program", "melody", p))(prog)
    for i in range(8):
        cb[m["mute"][i]] = (lambda cc: lambda c, locked: leds.set(cc, locked))(m["mute"][i])
    for i, cmd in enumerate(["fill", "cadence", "modulate", "silence", "motif"]):
        cb[m["rec"][i]] = (lambda cmd_: lambda c, o: engine.post(cmd_))(cmd)

    tr, nav = m["transport"], m["nav"]

    def play(c, o):
        scheduler_playing = not engine.playing
        engine.post("playing", scheduler_playing)
        leds.set(tr["play"], scheduler_playing)

    def stop(c, o):
        engine.post("playing", False)
        dispatcher.panic()
        leds.set(tr["play"], False)

    def rec(c, o):
        if recorder.active:
            recorder.active = False
            path = time.strftime("live_%y%m%d-%H%M%S.mid")
            recorder.write(path)
            print(f"\nGrabado: {path}")
            leds.set(tr["rec"], False)
        else:
            recorder.start()
            leds.set(tr["rec"], True)

    def freeze(c, o):
        engine.post("freeze")
        leds.set(tr["cycle"], not (engine.frozen or engine.freeze_req))

    def save_scene(c, o):
        scenes.append(bank.snapshot())
        json.dump(scenes, open(SCENES_FILE, "w"), indent=1)
        state["scene"] = len(scenes) - 1
        print(f"\nEscena {len(scenes)} guardada")

    def recall(delta):
        def f(c, o):
            if not scenes:
                return
            state["scene"] = (state["scene"] + delta) % len(scenes)
            bank.load_scene(scenes[state["scene"]])
        return f

    cb[tr["play"]] = play
    cb[tr["stop"]] = stop
    cb[tr["rew"]] = lambda c, o: engine.post("octave", -1)
    cb[tr["ff"]] = lambda c, o: engine.post("octave", 1)
    cb[tr["rec"]] = rec
    cb[tr["cycle"]] = freeze
    cb[nav["track_prev"]] = lambda c, o: engine.post("key", -1)
    cb[nav["track_next"]] = lambda c, o: engine.post("key", 1)
    cb[nav["marker_set"]] = save_scene
    cb[nav["marker_prev"]] = recall(-1)
    cb[nav["marker_next"]] = recall(1)
    leds.set(tr["play"], True)


def status_line(engine, bank):
    i = engine.info
    bars = "▁▂▃▄▅▆▇█"
    meters = "".join(bars[min(7, int(bank.get(n) * 8))] for n in FADER_NAMES)
    flags = ("❄" if i["frozen"] else " ") + ("●" if not engine.playing else " ")
    lay = "".join(l[0].upper() if engine.layers[l] else "·" for l in ("melody", "bass", "chords", "drums"))
    return (f"\r{i['key']:>2} {i['mode']:<9} {i['chord']:<8} {i['tempo']:5.1f}bpm "
            f"c.{i['bar']:<4} [{meters}] {lay} {flags}  ")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="faders en piloto automático (sin hardware)")
    ap.add_argument("--no-synth", action="store_true", help="sin sintetizador interno")
    ap.add_argument("--no-virtual", action="store_true", help="sin puerto MIDI virtual")
    ap.add_argument("--soundfont", help="archivo .sf2")
    ap.add_argument("--driver", default="pulseaudio", help="driver de audio de FluidSynth")
    ap.add_argument("--port-name", default="Mutopia Live", help="nombre del puerto virtual")
    ap.add_argument("--korg", default="nanoKONTROL2", help="fragmento del nombre del puerto del Korg")
    ap.add_argument("--map", help="JSON con mapa de CC")
    ap.add_argument("--leds", action="store_true")
    ap.add_argument("--record", help="grabar toda la sesión a este .mid")
    ap.add_argument("--render", help="render offline a .mid (sin audio, sin hardware)")
    ap.add_argument("--bars", type=int, default=32, help="compases del render")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--key", type=int, default=0, help="tónica inicial (0=Do … 11=Si)")
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--monitor", action="store_true", help="solo mostrar los CC del Korg y los valores suavizados")
    a = ap.parse_args()

    if a.list_ports:
        import mido
        print("Entradas:", mido.get_input_names())
        print("Salidas: ", mido.get_output_names())
        return

    korg = KorgInput(a.korg, a.map)
    if a.monitor:
        korg.on_cc = lambda cc, v: print(f"CC {cc:3d} = {v:3d}")
        try:
            korg.start()
        except Exception as e:
            sys.exit(str(e))
        print("Mueve los controles (Ctrl-C para salir)")
        try:
            while True:
                time.sleep(0.25)
                print("  " + "  ".join(f"{n[:4]}={korg.get(n):.2f}" for n in FADER_NAMES), end="\r")
        except KeyboardInterrupt:
            korg.stop()
        return
    engine = Engine(korg.bank, seed=a.seed, key=a.key)

    if a.render:
        driver = make_demo(korg.bank) if a.demo else None
        _, secs = render_offline(engine, korg.bank, a.render, a.bars, driver)
        print(f"Render: {a.render} ({a.bars} compases, {secs:.1f} s)")
        return

    if a.no_synth and a.no_virtual:
        sys.exit("Desactivaste ambas salidas.")
    backends = []
    try:
        if not a.no_virtual:
            backends.append(VirtualPortBackend(a.port_name))
            print(f"Puerto MIDI virtual: '{a.port_name}'")
        if not a.no_synth:
            backends.append(FluidBackend(a.soundfont, a.driver))
            print("FluidSynth listo")
    except Exception as e:
        sys.exit(f"Error de salida: {e}")
    recorder = Recorder()
    out = MultiBackend(backends + [recorder])

    if a.demo:
        korg.start_headless()
        drive = make_demo(korg.bank)
        t_demo = time.monotonic()

        def demo_loop():
            while True:
                drive(time.monotonic() - t_demo)
                time.sleep(0.05)
        threading.Thread(target=demo_loop, daemon=True).start()
    else:
        try:
            korg.start()
        except Exception as e:
            sys.exit(f"{e}\n(usa --demo para probar sin el controlador)")

    disp = Dispatcher(out)
    disp.start()
    sch = Scheduler(engine, disp, recorder)
    leds = Leds(a.korg) if a.leds else Leds("__none__")
    bind_controls(korg, engine, disp, recorder, leds, sch)
    if a.record:
        recorder.start()
    sch.start()
    print("Tocando. Ctrl-C para salir.")
    try:
        while True:
            sys.stdout.write(status_line(engine, korg.bank))
            sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        sch.running = False
        disp.panic()
        disp.stop()
        if a.record and recorder.active:
            recorder.active = False
            recorder.write(a.record)
            print(f"\nGrabado: {a.record}")
        korg.stop()
        out.close()


if __name__ == "__main__":
    main()
