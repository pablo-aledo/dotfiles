"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ARC SUPERVISOR  v1.0                                 ║
║         Supervisor de arco emocional para obras MIDI multi-sección           ║
║                                                                              ║
║  Dado un conjunto de secciones MIDI ya generadas y un plan emocional         ║
║  opcional, mide si el arco emocional real coincide con el planificado,       ║
║  diagnostica los problemas y sugiere valores concretos para corregirlos.     ║
║                                                                              ║
║  DIMENSIONES ANALIZADAS:                                                     ║
║    tension   — tensión armónica (intervalos disonantes)                      ║
║    activity  — densidad rítmica (notas/beat × velocity)                      ║
║    register  — registro melódico (pitch medio ponderado)                     ║
║    harmony   — complejidad armónica (pitch-classes distintos + cromatismo)   ║
║    contrast  — meta-métrica: distancia entre secciones consecutivas          ║
║                                                                              ║
║  MODOS DE ENTRADA DE MIDIS:                                                  ║
║    Modo A — secciones separadas:                                             ║
║      python arc_supervisor.py sec_A.mid sec_B.mid sec_C.mid                 ║
║    Modo B — MIDI único con cortes:                                           ║
║      python arc_supervisor.py obra.mid --sections 8 8 8 8                   ║
║                                                                              ║
║  PLAN EMOCIONAL (combinables):                                               ║
║    --curves FILE       JSON de tension_designer (.curves.json)               ║
║    --arc TEXT          palabras por sección: "calma tension climax"          ║
║    --tension TEXT      shorthand numérico: "0.2,0.6,0.95,0.3"               ║
║    --lexicon FILE      léxico emocional propio (.json)                       ║
║                                                                              ║
║  UMBRALES:                                                                   ║
║    --contrast-min F    contraste mínimo entre secciones (default: 0.25)     ║
║    --contrast-max F    contraste máximo tolerable       (default: 0.80)     ║
║    --jump-threshold F  salto abrupto en una dimensión   (default: 0.60)     ║
║                                                                              ║
║  SALIDA:                                                                     ║
║    --report FILE       guardar diagnóstico en JSON                           ║
║    --plot              visualizar arco real vs esperado (matplotlib)         ║
║    --verbose           detalle por dimensión y por compás                   ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import argparse
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    import mido
    MIDO_OK = True
except ImportError:
    MIDO_OK = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MPL_OK = True
except ImportError:
    MPL_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  LÉXICO EMOCIONAL
# ══════════════════════════════════════════════════════════════════════════════

# Cada entrada: (tension, activity, register, harmony)
DEFAULT_LEXICON: Dict[str, Tuple[float, float, float, float]] = {
    # Estado         tension  activity  register  harmony
    "calma":        (0.10,    0.15,     0.45,     0.15),
    "reposo":       (0.10,    0.10,     0.45,     0.10),
    "silencio":     (0.05,    0.05,     0.40,     0.05),
    "serenidad":    (0.15,    0.20,     0.50,     0.20),
    "nostalgia":    (0.40,    0.25,     0.45,     0.45),
    "melancolia":   (0.50,    0.30,     0.40,     0.50),
    "tristeza":     (0.45,    0.20,     0.35,     0.40),
    "flotacion":    (0.25,    0.15,     0.65,     0.30),
    "tension":      (0.65,    0.55,     0.55,     0.60),
    "angustia":     (0.75,    0.50,     0.50,     0.70),
    "inquietud":    (0.60,    0.60,     0.50,     0.55),
    "agitacion":    (0.70,    0.80,     0.60,     0.65),
    "fragmentado":  (0.60,    0.70,     0.55,     0.65),
    "irrupcion":    (0.80,    0.95,     0.70,     0.75),
    "climax":       (0.90,    0.90,     0.80,     0.85),
    "apogeo":       (0.95,    0.95,     0.85,     0.90),
    "drama":        (0.85,    0.75,     0.70,     0.80),
    "resolucion":   (0.20,    0.30,     0.50,     0.25),
    "cadencia":     (0.15,    0.25,     0.48,     0.20),
    "coda":         (0.20,    0.20,     0.45,     0.20),
    "desarrollo":   (0.65,    0.65,     0.60,     0.70),
    "exposicion":   (0.35,    0.45,     0.55,     0.40),
    "recapitulacion":(0.30,   0.40,     0.52,     0.35),
    "misterio":     (0.55,    0.30,     0.55,     0.65),
    "expectativa":  (0.60,    0.40,     0.55,     0.55),
    "alegria":      (0.25,    0.70,     0.65,     0.30),
    "triunfo":      (0.45,    0.85,     0.75,     0.50),
}

DIMS = ["tension", "activity", "register", "harmony"]


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SectionVector:
    """Vector emocional de una sección."""
    name: str
    tension:  float = 0.0
    activity: float = 0.0
    register: float = 0.0
    harmony:  float = 0.0
    n_bars:   int   = 0
    n_notes:  int   = 0

    def as_array(self) -> np.ndarray:
        return np.array([self.tension, self.activity, self.register, self.harmony])

    def as_dict(self) -> dict:
        return {
            "tension":  round(self.tension,  3),
            "activity": round(self.activity, 3),
            "register": round(self.register, 3),
            "harmony":  round(self.harmony,  3),
        }


@dataclass
class PlanVector:
    """Vector emocional planificado para una sección."""
    name: str
    tension:  Optional[float] = None
    activity: Optional[float] = None
    register: Optional[float] = None
    harmony:  Optional[float] = None
    source:   str = "inferred"   # "curves", "arc_text", "tension_shorthand", "inferred"

    def as_array(self) -> np.ndarray:
        return np.array([
            self.tension  if self.tension  is not None else float('nan'),
            self.activity if self.activity is not None else float('nan'),
            self.register if self.register is not None else float('nan'),
            self.harmony  if self.harmony  is not None else float('nan'),
        ])

    def has_dim(self, dim: str) -> bool:
        return getattr(self, dim) is not None


@dataclass
class SectionDiagnosis:
    """Diagnóstico de una sección."""
    name: str
    measured: SectionVector
    planned:  Optional[PlanVector]
    contrast: float = 0.0          # contraste con sección anterior
    problems: List[dict] = field(default_factory=list)
    suggestions: List[dict] = field(default_factory=list)
    ok: bool = True


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE FEATURES DESDE MIDI
# ══════════════════════════════════════════════════════════════════════════════

# Intervalos disonantes en semitonos (clase de intervalo, sin octava)
DISSONANT_ICS = {1, 2, 6, 10, 11}  # 2ª m/M, tritono, 7ª m/M

# Perfiles de Krumhansl-Schmuckler para mayor y menor
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
# Normalizar
KK_MAJOR = KK_MAJOR / KK_MAJOR.sum()
KK_MINOR = KK_MINOR / KK_MINOR.sum()


def _ticks_to_beats(midi_file: "mido.MidiFile") -> float:
    """Retorna el factor ticks-por-beat."""
    return float(midi_file.ticks_per_beat)


def extract_notes(midi_file: "mido.MidiFile") -> List[dict]:
    """
    Extrae todas las notas del MIDI como lista de dicts:
    {pitch, velocity, start_beat, duration_beat, channel}
    """
    tpb = _ticks_to_beats(midi_file)
    notes = []

    for track in midi_file.tracks:
        active = {}  # (channel, pitch) -> (start_tick, velocity)
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (current_tick, msg.velocity)
            elif msg.type in ('note_off',) or (msg.type == 'note_on' and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start_tick, vel = active.pop(key)
                    dur_tick = current_tick - start_tick
                    notes.append({
                        'pitch':          msg.note,
                        'velocity':       vel,
                        'start_beat':     start_tick / tpb,
                        'duration_beat':  max(dur_tick / tpb, 0.0625),
                        'channel':        msg.channel,
                    })

    return sorted(notes, key=lambda n: n['start_beat'])


def estimate_bars(midi_file: "mido.MidiFile") -> int:
    """Estima el número de compases del MIDI (asume 4/4 si no hay señal)."""
    tpb = _ticks_to_beats(midi_file)
    beats_per_bar = 4  # default

    total_ticks = 0
    for track in midi_file.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == 'time_signature':
                beats_per_bar = msg.numerator
        total_ticks = max(total_ticks, tick)

    total_beats = total_ticks / tpb
    return max(1, int(math.ceil(total_beats / beats_per_bar)))


def compute_tension(notes: List[dict]) -> float:
    """
    Tensión armónica: proporción de intervalos simultáneos disonantes.
    Se consideran pares de notas que se solapan en el tiempo.
    """
    if len(notes) < 2:
        return 0.0

    dissonant = 0
    total = 0

    # Ventana deslizante: para cada nota, buscar notas simultáneas
    for i, n1 in enumerate(notes):
        n1_end = n1['start_beat'] + n1['duration_beat']
        for n2 in notes[i+1:]:
            if n2['start_beat'] >= n1_end:
                break
            # Solapamiento
            ic = abs(n1['pitch'] - n2['pitch']) % 12
            if ic == 0:
                ic = 12  # unísono/octava — consonante
            total += 1
            if ic in DISSONANT_ICS:
                dissonant += 1

    if total == 0:
        # Melodía monofónica: usar intervalos melódicos consecutivos
        intervals = [
            abs(notes[i+1]['pitch'] - notes[i]['pitch']) % 12
            for i in range(len(notes) - 1)
        ]
        if not intervals:
            return 0.0
        dissonant = sum(1 for ic in intervals if ic in DISSONANT_ICS)
        return dissonant / len(intervals)

    return dissonant / total


def compute_activity(notes: List[dict], total_beats: float) -> float:
    """
    Densidad rítmica: notas por beat × velocity media normalizada.
    Normalizado a [0,1] usando referencia de 2 notas/beat a vel=90 → ~1.0
    """
    if not notes or total_beats <= 0:
        return 0.0

    notes_per_beat = len(notes) / total_beats
    vel_mean = np.mean([n['velocity'] for n in notes]) / 127.0

    raw = notes_per_beat * vel_mean
    # Normalizar: 2 notas/beat a vel media = 1.0 como referencia
    return float(min(1.0, raw / (2.0 * (90.0 / 127.0))))


def compute_register(notes: List[dict]) -> float:
    """
    Registro melódico: pitch medio ponderado por duración.
    Normalizado al rango MIDI [21, 108] (A0–C8).
    """
    if not notes:
        return 0.5

    total_dur = sum(n['duration_beat'] for n in notes)
    if total_dur == 0:
        return 0.5

    weighted_pitch = sum(n['pitch'] * n['duration_beat'] for n in notes) / total_dur
    return float((weighted_pitch - 21) / (108 - 21))


def compute_harmony(notes: List[dict], total_beats: float) -> float:
    """
    Complejidad armónica: combinación de:
      - Proporción de ventanas con ≥4 pitch-classes distintos simultáneos
      - Cromatismo: proporción de pitch-classes usados sobre 12
    """
    if not notes or total_beats <= 0:
        return 0.0

    # Cromatismo global
    pcs_used = len(set(n['pitch'] % 12 for n in notes))
    chromaticism = pcs_used / 12.0

    # Densidad armónica por ventanas de 1 beat
    n_windows = max(1, int(total_beats))
    dense_windows = 0
    for w in range(n_windows):
        w_start = float(w)
        w_end = w_start + 1.0
        pcs_in_window = set(
            n['pitch'] % 12
            for n in notes
            if n['start_beat'] < w_end and (n['start_beat'] + n['duration_beat']) > w_start
        )
        if len(pcs_in_window) >= 4:
            dense_windows += 1

    density_score = dense_windows / n_windows

    return float(0.5 * chromaticism + 0.5 * density_score)


def analyze_midi_file(path: str, name: str = None) -> SectionVector:
    """Extrae el vector emocional completo de un archivo MIDI."""
    if name is None:
        name = os.path.splitext(os.path.basename(path))[0]

    mid = mido.MidiFile(path)
    notes = extract_notes(mid)
    n_bars = estimate_bars(mid)

    tpb = _ticks_to_beats(mid)
    total_ticks = max(
        sum(msg.time for msg in track) for track in mid.tracks
    )
    total_beats = total_ticks / tpb if tpb > 0 else 4.0

    # Excluir canal 9 (percusión) del análisis melódico/armónico
    melodic_notes = [n for n in notes if n['channel'] != 9]
    if not melodic_notes:
        melodic_notes = notes  # fallback si solo hay percusión

    return SectionVector(
        name=name,
        tension=compute_tension(melodic_notes),
        activity=compute_activity(melodic_notes, total_beats),
        register=compute_register(melodic_notes),
        harmony=compute_harmony(melodic_notes, total_beats),
        n_bars=n_bars,
        n_notes=len(melodic_notes),
    )


def split_midi(path: str, sections_bars: List[int]) -> List["mido.MidiFile"]:
    """
    Divide un MIDI en secciones según el número de compases indicado.
    Retorna una lista de MidiFile en memoria.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat
    beats_per_bar = 4

    # Detectar compás
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                beats_per_bar = msg.numerator
                break

    ticks_per_bar = tpb * beats_per_bar
    boundaries = []
    acc = 0
    for bars in sections_bars:
        boundaries.append((acc, acc + bars * ticks_per_bar))
        acc += bars * ticks_per_bar

    result = []
    for (start_tick, end_tick) in boundaries:
        new_mid = mido.MidiFile(ticks_per_beat=tpb)
        for track in mid.tracks:
            new_track = mido.MidiTrack()
            current_tick = 0
            pending = []
            for msg in track:
                current_tick += msg.time
                if start_tick <= current_tick < end_tick:
                    # Ajustar tiempo relativo al inicio de la sección
                    adjusted_time = current_tick - start_tick
                    pending.append(msg.copy(time=0))
                    if len(pending) == 1:
                        pending[0] = msg.copy(time=int(adjusted_time))
                    else:
                        prev_abs = start_tick + sum(m.time for m in pending[:-1])
                        pending[-1] = msg.copy(time=int(current_tick - start_tick - sum(m.time for m in pending[:-1])))
            # Reconstruir track con tiempos delta correctos
            abs_tick = 0
            prev_abs = start_tick
            new_track2 = mido.MidiTrack()
            current_tick2 = 0
            for msg in track:
                current_tick2 += msg.time
                if start_tick <= current_tick2 < end_tick:
                    delta = current_tick2 - max(prev_abs, start_tick)
                    new_track2.append(msg.copy(time=int(delta)))
                    prev_abs = current_tick2
            new_mid.tracks.append(new_track2)
        result.append(new_mid)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN EMOCIONAL
# ══════════════════════════════════════════════════════════════════════════════

def load_lexicon(path: Optional[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """Carga léxico externo y lo fusiona con el default."""
    lexicon = dict(DEFAULT_LEXICON)
    if path and os.path.exists(path):
        with open(path) as f:
            ext = json.load(f)
        for word, vals in ext.items():
            if isinstance(vals, (list, tuple)) and len(vals) == 4:
                lexicon[word.lower()] = tuple(float(v) for v in vals)
            elif isinstance(vals, dict):
                lexicon[word.lower()] = (
                    float(vals.get('tension',  0.5)),
                    float(vals.get('activity', 0.5)),
                    float(vals.get('register', 0.5)),
                    float(vals.get('harmony',  0.5)),
                )
    return lexicon


def word_to_vector(word: str, lexicon: dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Busca una palabra en el léxico. Si no existe, interpola desde las
    más cercanas por distancia de coseno sobre los valores del léxico.
    """
    w = word.lower().strip()
    if w in lexicon:
        return lexicon[w]

    # Interpolación: encontrar las 2 entradas más cercanas por nombre
    # (distancia de Levenshtein simplificada: coincidencia de prefijo)
    candidates = []
    for key, vec in lexicon.items():
        # Similitud de caracteres comunes
        common = sum(1 for a, b in zip(w, key) if a == b)
        candidates.append((common, key, vec))
    candidates.sort(reverse=True)

    if candidates:
        # Media ponderada de los 2 mejores
        top = candidates[:2]
        total_w = sum(c[0] for c in top) or 1
        t = sum(c[0] * c[2][0] for c in top) / total_w
        a = sum(c[0] * c[2][1] for c in top) / total_w
        r = sum(c[0] * c[2][2] for c in top) / total_w
        h = sum(c[0] * c[2][3] for c in top) / total_w
        return (t, a, r, h)

    return None


def parse_arc_text(arc: str, n_sections: int,
                   lexicon: dict, verbose: bool = False) -> List[PlanVector]:
    """
    Parsea '--arc "calma tension climax resolucion"' en vectores de plan.
    Si hay menos palabras que secciones, la última se repite.
    """
    words = arc.strip().split()
    plans = []
    warnings = []

    for i in range(n_sections):
        word = words[min(i, len(words) - 1)]
        vec = word_to_vector(word, lexicon)
        if vec is None:
            vec = (0.5, 0.5, 0.5, 0.5)
            warnings.append(f"  AVISO: palabra '{word}' no reconocida — usando vector neutro")
        else:
            if word.lower() not in lexicon and verbose:
                warnings.append(f"  AVISO: '{word}' interpolado desde entradas del léxico")

        plans.append(PlanVector(
            name=f"sec_{i+1}",
            tension=vec[0], activity=vec[1],
            register=vec[2], harmony=vec[3],
            source="arc_text"
        ))

    for w in warnings:
        print(w)

    return plans


def parse_tension_shorthand(tension_str: str, n_sections: int) -> List[float]:
    """Parsea '--tension "0.2,0.6,0.95,0.3"' en lista de floats."""
    vals = [float(v.strip()) for v in tension_str.split(',')]
    # Extender o recortar al número de secciones
    while len(vals) < n_sections:
        vals.append(vals[-1])
    return vals[:n_sections]


def load_curves_json(path: str, n_sections: int,
                     section_bars: List[int]) -> List[PlanVector]:
    """
    Carga un .curves.json de tension_designer y agrega los valores
    por compás en vectores por sección.
    """
    with open(path) as f:
        data = json.load(f)

    plans = []
    # Calcular boundaries de compases por sección
    boundaries = []
    acc = 0
    for bars in section_bars:
        boundaries.append((acc, acc + bars))
        acc += bars

    for i, (start_bar, end_bar) in enumerate(boundaries):
        pv = PlanVector(name=f"sec_{i+1}", source="curves")

        for dim in DIMS:
            if dim in data and 'values' in data[dim]:
                values = data[dim]['values']
                # Agregar valores en el rango de compases de esta sección
                sec_vals = values[start_bar:end_bar]
                if sec_vals:
                    setattr(pv, dim, float(np.mean(sec_vals)))

        plans.append(pv)

    return plans


def merge_plans(base: List[PlanVector],
                override: List[PlanVector]) -> List[PlanVector]:
    """
    Fusiona dos listas de planes. Override sobreescribe dimensiones
    que base no tiene, o cuando override tiene una fuente más específica.
    """
    result = []
    for b, o in zip(base, override):
        merged = PlanVector(name=b.name, source=b.source)
        for dim in DIMS:
            b_val = getattr(b, dim)
            o_val = getattr(o, dim)
            # Usar override si base no tiene valor para esta dimensión
            setattr(merged, dim, b_val if b_val is not None else o_val)
        result.append(merged)
    return result


def infer_plan_from_sections(sections: List[SectionVector]) -> List[PlanVector]:
    """
    Modo inferido: sin plan externo, construye un plan de referencia
    desde los propios MIDIs. Usa la primera sección como ancla y
    detecta la forma del arco subyacente.
    """
    if not sections:
        return []

    # La referencia es el promedio de todas las secciones
    # El plan inferido es simplemente el vector de cada sección —
    # en modo inferido el supervisor detecta inconsistencias internas,
    # no desviaciones respecto a un objetivo externo.
    plans = []
    for s in sections:
        plans.append(PlanVector(
            name=s.name,
            tension=s.tension,
            activity=s.activity,
            register=s.register,
            harmony=s.harmony,
            source="inferred"
        ))
    return plans


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARACIÓN Y DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def normalize_curve(values: List[float]) -> List[float]:
    """
    Normaliza una curva a media 0 y std 1 para comparación de formas.
    Si la std es 0 (curva plana), retorna ceros.
    """
    arr = np.array(values, dtype=float)
    std = arr.std()
    if std < 1e-6:
        return [0.0] * len(values)
    return list((arr - arr.mean()) / std)


def compute_contrast(vec_a: SectionVector,
                     vec_b: SectionVector) -> float:
    """
    Distancia euclidiana normalizada entre dos vectores de sección.
    Retorna valor en [0, 1] (aprox.) — 0 = idénticos, 1 = opuestos.
    """
    a = vec_a.as_array()
    b = vec_b.as_array()
    dist = float(np.linalg.norm(a - b))
    # Máxima distancia posible en espacio [0,1]^4 = sqrt(4) = 2.0
    return min(1.0, dist / 2.0)


def deviation_score(measured: float,
                    planned: float,
                    tolerance: float = 0.15) -> float:
    """
    Desviación absoluta entre valor medido y planificado.
    Retorna 0 si dentro de la tolerancia, valor positivo si fuera.
    """
    diff = abs(measured - planned)
    return max(0.0, diff - tolerance)


def diagnose_section(
    idx: int,
    measured: SectionVector,
    planned: Optional[PlanVector],
    prev_measured: Optional[SectionVector],
    contrast_min: float,
    contrast_max: float,
    jump_threshold: float,
) -> SectionDiagnosis:
    """Genera el diagnóstico completo de una sección."""

    diag = SectionDiagnosis(
        name=measured.name,
        measured=measured,
        planned=planned,
    )

    # ── Contraste con sección anterior ───────────────────────────────────────
    if prev_measured is not None:
        contrast = compute_contrast(prev_measured, measured)
        diag.contrast = contrast

        if contrast < contrast_min:
            diag.ok = False
            # Calcular qué dimensiones son más parecidas
            diffs = {
                dim: abs(getattr(measured, dim) - getattr(prev_measured, dim))
                for dim in DIMS
            }
            flattest_dim = min(diffs, key=diffs.get)

            suggestions = {}
            for dim in DIMS:
                current = getattr(measured, dim)
                prev = getattr(prev_measured, dim)
                diff = abs(current - prev)
                if diff < 0.20:
                    # Sugerir incremento o decremento según posición en el arco
                    if idx < 2:  # primeras secciones: sugerir aumento
                        suggestions[dim] = round(min(1.0, current + 0.35), 2)
                    else:  # secciones posteriores: sugerir cambio basado en plan
                        target = getattr(planned, dim) if planned and getattr(planned, dim) is not None else None
                        if target is not None:
                            suggestions[dim] = round(target, 2)
                        else:
                            suggestions[dim] = round(min(1.0, current + 0.30), 2)

            diag.problems.append({
                "type": "contrast_insufficient",
                "contrast": round(contrast, 3),
                "threshold": contrast_min,
                "description": f"Sección demasiado similar a la anterior (contrast={contrast:.2f}, mínimo={contrast_min})",
                "flattest_dimension": flattest_dim,
            })
            if suggestions:
                diag.suggestions.append({
                    "type": "increase_contrast",
                    "description": "Aumentar diferenciación respecto a la sección anterior",
                    "values": suggestions,
                })

        elif contrast > contrast_max:
            diag.ok = False
            # Salto abrupto: sugerir valores intermedios
            suggestions = {}
            for dim in DIMS:
                current = getattr(measured, dim)
                prev = getattr(prev_measured, dim)
                if abs(current - prev) > jump_threshold:
                    suggestions[dim] = round((current + prev) / 2.0, 2)

            diag.problems.append({
                "type": "contrast_excessive",
                "contrast": round(contrast, 3),
                "threshold": contrast_max,
                "description": f"Salto demasiado brusco respecto a la sección anterior (contrast={contrast:.2f}, máximo={contrast_max})",
            })
            if suggestions:
                diag.suggestions.append({
                    "type": "smooth_transition",
                    "description": "Suavizar la transición — valores intermedios sugeridos",
                    "values": suggestions,
                    "note": "Considera insertar una sección de transición con estos valores",
                })

    # ── Desviación respecto al plan ───────────────────────────────────────────
    if planned is not None and planned.source != "inferred":
        deviations = {}
        for dim in DIMS:
            plan_val = getattr(planned, dim)
            if plan_val is not None:
                dev = deviation_score(getattr(measured, dim), plan_val)
                if dev > 0:
                    deviations[dim] = {
                        "measured": round(getattr(measured, dim), 3),
                        "planned":  round(plan_val, 3),
                        "deviation": round(dev, 3),
                    }

        if deviations:
            diag.ok = False
            worst_dim = max(deviations, key=lambda d: deviations[d]['deviation'])

            diag.problems.append({
                "type": "plan_deviation",
                "description": f"La sección se desvía del plan emocional (dimensión más afectada: {worst_dim})",
                "deviations": deviations,
            })

            correction = {}
            for dim, dev_info in deviations.items():
                correction[dim] = dev_info['planned']

            diag.suggestions.append({
                "type": "align_to_plan",
                "description": "Ajustar valores para alinearse al plan emocional",
                "values": correction,
            })

    return diag


# ══════════════════════════════════════════════════════════════════════════════
#  SALIDA
# ══════════════════════════════════════════════════════════════════════════════

W = 72  # ancho de la salida en consola

def _bar(label: str, value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    bar = '█' * filled + '░' * (width - filled)
    return f"{label:<10} {bar} {value:.2f}"


def print_report(diagnoses: List[SectionDiagnosis],
                 verbose: bool = False):
    """Imprime el informe en consola."""
    B = '\033[1m'
    R = '\033[0m'
    RED = '\033[91m'
    YEL = '\033[93m'
    GRN = '\033[92m'
    DIM = '\033[2m'

    print()
    print("═" * W)
    print(f"  ARC SUPERVISOR v1.0")
    print("═" * W)

    # Tabla resumen
    header = f"{'Sección':<14} {'tension':>8} {'activity':>9} {'register':>9} {'harmony':>8} {'contrast':>9}  estado"
    print(f"\n{DIM}{header}{R}")
    print(f"{DIM}{'─'*W}{R}")

    for d in diagnoses:
        m = d.measured
        contrast_str = f"{d.contrast:.2f}" if d.contrast > 0 else "  —  "

        # Color del contraste
        if d.contrast > 0:
            if d.contrast < 0.25:
                c_col = YEL
            elif d.contrast > 0.80:
                c_col = RED
            else:
                c_col = GRN
        else:
            c_col = DIM

        status = f"{GRN}OK{R}" if d.ok else f"{RED}PROBLEMA{R}"

        print(
            f"  {m.name:<12} "
            f"{m.tension:>8.2f} "
            f"{m.activity:>9.2f} "
            f"{m.register:>9.2f} "
            f"{m.harmony:>8.2f} "
            f"{c_col}{contrast_str:>9}{R}  "
            f"{status}"
        )

    print()

    # Detalle de problemas y sugerencias
    any_problem = any(not d.ok for d in diagnoses)
    if not any_problem:
        print(f"  {GRN}✓ Sin problemas detectados en el arco emocional.{R}\n")
        return

    for d in diagnoses:
        if d.ok:
            continue

        print(f"{'─'*W}")
        print(f"  {B}SECCIÓN {d.measured.name}{R}")

        for prob in d.problems:
            ptype = prob['type']
            print(f"\n  {YEL}▸ {prob['description']}{R}")

            if ptype == 'plan_deviation' and verbose:
                for dim, info in prob.get('deviations', {}).items():
                    print(f"    {dim:<10}  medido={info['measured']:.2f}  "
                          f"plan={info['planned']:.2f}  "
                          f"desv={info['deviation']:.2f}")

            if ptype == 'contrast_insufficient':
                print(f"    dimensión más plana: {prob.get('flattest_dimension','?')}")

        for sug in d.suggestions:
            print(f"\n  {GRN}→ {sug['description']}{R}")
            vals = sug.get('values', {})
            if vals:
                for dim, val in vals.items():
                    current = getattr(d.measured, dim)
                    arrow = "↑" if val > current else "↓"
                    print(f"    {dim:<10}  actual={current:.2f}  →  sugerido={val:.2f}  {arrow}")
            if 'note' in sug:
                print(f"    {DIM}nota: {sug['note']}{R}")

        if verbose and d.measured.n_notes > 0:
            print(f"\n  {DIM}info: {d.measured.n_notes} notas  ·  {d.measured.n_bars} compases{R}")

    print()


def build_json_report(diagnoses: List[SectionDiagnosis],
                      mode: str,
                      plan_source: str) -> dict:
    """Construye el JSON de reporte completo."""
    return {
        "arc_supervisor": "v1.0",
        "mode": mode,
        "plan_source": plan_source,
        "n_sections": len(diagnoses),
        "summary": {
            "ok": all(d.ok for d in diagnoses),
            "n_problems": sum(1 for d in diagnoses if not d.ok),
            "problem_sections": [d.name for d in diagnoses if not d.ok],
        },
        "sections": [
            {
                "name": d.name,
                "measured": d.measured.as_dict(),
                "planned": {
                    dim: getattr(d.planned, dim)
                    for dim in DIMS
                    if d.planned and getattr(d.planned, dim) is not None
                } if d.planned else None,
                "contrast": round(d.contrast, 3),
                "ok": d.ok,
                "problems": d.problems,
                "suggestions": d.suggestions,
            }
            for d in diagnoses
        ],
    }


def plot_arc(diagnoses: List[SectionDiagnosis], plan_source: str):
    """Visualiza el arco real vs planificado con matplotlib."""
    if not MPL_OK:
        print("  AVISO: matplotlib no disponible para --plot")
        return

    n = len(diagnoses)
    labels = [d.name for d in diagnoses]
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle('ARC SUPERVISOR — Arco emocional real vs planificado',
                 fontsize=13, fontweight='bold')
    fig.patch.set_facecolor('#1a1a2e')

    dim_colors = {
        'tension':  '#e74c3c',
        'activity': '#3498db',
        'register': '#2ecc71',
        'harmony':  '#9b59b6',
    }

    for ax, dim in zip(axes.flat, DIMS):
        ax.set_facecolor('#16213e')
        ax.spines[:].set_color('#4a4a6a')
        ax.tick_params(colors='white')
        ax.set_title(dim, color='white', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=8)
        ax.yaxis.label.set_color('white')
        ax.grid(True, color='#2a2a4a', linewidth=0.5, linestyle='--')

        measured = [getattr(d.measured, dim) for d in diagnoses]
        ax.plot(x, measured, 'o-', color=dim_colors[dim],
                linewidth=2, markersize=6, label='medido')

        # Plan si existe y no es inferido
        has_plan = any(
            d.planned and d.planned.source != 'inferred'
            and getattr(d.planned, dim) is not None
            for d in diagnoses
        )
        if has_plan:
            planned = [
                getattr(d.planned, dim) if d.planned and getattr(d.planned, dim) is not None
                else float('nan')
                for d in diagnoses
            ]
            ax.plot(x, planned, 's--', color='white',
                    linewidth=1, markersize=4, alpha=0.6, label='planificado')

        # Marcar secciones problemáticas
        for i, d in enumerate(diagnoses):
            if not d.ok:
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color='red')

        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                  framealpha=0.7)

    # Contraste en subgráfico adicional
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ARC SUPERVISOR — Supervisor de arco emocional para obras MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Entradas MIDI
    parser.add_argument('inputs', nargs='+',
                        help='MIDIs de sección (uno por sección) o un único MIDI con --sections')
    parser.add_argument('--sections', nargs='+', type=int, default=None,
                        metavar='N',
                        help='Compases por sección (solo con un único MIDI de entrada)')

    # Plan emocional
    parser.add_argument('--curves', default=None,
                        help='JSON de tension_designer (.curves.json)')
    parser.add_argument('--arc', default=None,
                        help='Palabras por sección: "calma tension climax resolucion"')
    parser.add_argument('--tension', default=None,
                        help='Shorthand numérico para tensión: "0.2,0.6,0.95,0.3"')
    parser.add_argument('--lexicon', default=None,
                        help='Léxico emocional propio (.json)')

    # Umbrales
    parser.add_argument('--contrast-min', type=float, default=0.25,
                        help='Contraste mínimo entre secciones (default: 0.25)')
    parser.add_argument('--contrast-max', type=float, default=0.80,
                        help='Contraste máximo tolerable (default: 0.80)')
    parser.add_argument('--jump-threshold', type=float, default=0.60,
                        help='Umbral de salto abrupto en una dimensión (default: 0.60)')

    # Salida
    parser.add_argument('--report', default=None,
                        help='Guardar diagnóstico en JSON')
    parser.add_argument('--plot', action='store_true',
                        help='Visualizar arco real vs esperado (requiere matplotlib)')
    parser.add_argument('--verbose', action='store_true',
                        help='Detalle por dimensión')

    args = parser.parse_args()

    if not MIDO_OK:
        print("ERROR: mido no disponible. Instala con: pip install mido")
        sys.exit(1)

    # ── Validar entradas ──────────────────────────────────────────────────────
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"ERROR: No se encontró el archivo: {path}")
            sys.exit(1)

    # ── Determinar modo de entrada ────────────────────────────────────────────
    if len(args.inputs) == 1 and args.sections:
        # Modo B: un MIDI, cortar por secciones
        mode = "split"
        print(f"  Modo: MIDI único → {len(args.sections)} secciones")
        midi_parts = split_midi(args.inputs[0], args.sections)
        section_names = [f"sec_{i+1}" for i in range(len(args.sections))]
        section_bars = args.sections

        # Analizar cada parte en memoria
        import tempfile
        sections = []
        for i, (part, name) in enumerate(zip(midi_parts, section_names)):
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                part.save(tmp.name)
                sv = analyze_midi_file(tmp.name, name)
                os.unlink(tmp.name)
            sections.append(sv)

    else:
        # Modo A: varios MIDIs separados
        if args.sections:
            print("  AVISO: --sections ignorado en modo multi-MIDI (un archivo por sección)")
        mode = "separate"
        sections = []
        section_bars = []
        for path in args.inputs:
            name = os.path.splitext(os.path.basename(path))[0]
            sv = analyze_midi_file(path, name)
            sections.append(sv)
            section_bars.append(sv.n_bars)
        section_names = [s.name for s in sections]

    n_sections = len(sections)

    if args.verbose:
        print(f"\n  Secciones analizadas: {n_sections}")
        for s in sections:
            print(f"    {s.name}: {s.n_notes} notas · {s.n_bars} compases")

    # ── Construir plan emocional ──────────────────────────────────────────────
    lexicon = load_lexicon(args.lexicon)
    plan_source = "inferred"
    plans: List[PlanVector] = []

    # Base: curves.json si existe
    if args.curves:
        if not os.path.exists(args.curves):
            print(f"ERROR: No se encontró --curves: {args.curves}")
            sys.exit(1)
        plans = load_curves_json(args.curves, n_sections, section_bars)
        plan_source = "curves"

    # Sobreescribir con --arc si se especifica
    if args.arc:
        arc_plans = parse_arc_text(args.arc, n_sections, lexicon, args.verbose)
        if plans:
            plans = merge_plans(plans, arc_plans)
        else:
            plans = arc_plans
        plan_source = "arc_text" if plan_source == "inferred" else plan_source + "+arc"

    # Sobreescribir tensión con --tension si se especifica
    if args.tension:
        tension_vals = parse_tension_shorthand(args.tension, n_sections)
        if not plans:
            plans = [PlanVector(name=section_names[i], source="tension_shorthand")
                     for i in range(n_sections)]
        for pv, tv in zip(plans, tension_vals):
            pv.tension = tv
            if pv.source == "inferred":
                pv.source = "tension_shorthand"
        plan_source = "tension_shorthand" if plan_source == "inferred" else plan_source + "+tension"

    # Si no hay ningún plan externo, usar modo inferido
    if not plans:
        plans = infer_plan_from_sections(sections)
        plan_source = "inferred"

    # Nombrar planes con los nombres de sección reales
    for pv, name in zip(plans, section_names):
        pv.name = name

    # ── Diagnóstico ───────────────────────────────────────────────────────────
    diagnoses: List[SectionDiagnosis] = []
    for i, (measured, planned) in enumerate(zip(sections, plans)):
        prev = sections[i-1] if i > 0 else None
        diag = diagnose_section(
            idx=i,
            measured=measured,
            planned=planned if plan_source != "inferred" else None,
            prev_measured=prev,
            contrast_min=args.contrast_min,
            contrast_max=args.contrast_max,
            jump_threshold=args.jump_threshold,
        )
        diagnoses.append(diag)

    # ── Salida ────────────────────────────────────────────────────────────────
    print_report(diagnoses, verbose=args.verbose)

    if args.report:
        report_data = build_json_report(diagnoses, mode, plan_source)
        with open(args.report, 'w') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Reporte guardado: {args.report}\n")

    if args.plot:
        plot_arc(diagnoses, plan_source)

    # Código de salida: 0 si todo OK, 1 si hay problemas
    sys.exit(0 if all(d.ok for d in diagnoses) else 1)


if __name__ == '__main__':
    main()
