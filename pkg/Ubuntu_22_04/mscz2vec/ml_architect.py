#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       ML ARCHITECT  v5.0                                     ║
║     Composición estructural dirigida por gramáticas aprendidas de corpus     ║
║                                                                              ║
║  MEJORAS v5.0 respecto a v4.0 — variedad musical:                            ║
║    [A] Múltiples frases fuente — selección de ventana distinta por sección   ║
║    [B] Humanización rítmica — micro-desplazamientos de onset y velocity      ║
║    [C] Variación de textura — dos mitades con estilos de acompañamiento      ║
║        distintos dentro de cada sección                                      ║
║    [D] Selección de fragmento por posición — la fuente se divide en N        ║
║        ventanas y se elige la que corresponde a la posición de la sección    ║
║    [E] Contrapunto independiente — segunda voz con movimiento contrario      ║
║    [F] Reharmonización intra-sección — cambio armónico cada 2 compases      ║
║    [G] Silencios estructurales — respiraciones entre frases melódicas        ║
║                                                                              ║
║  MEJORAS v4.0 respecto a v3.0:                                               ║
║    El modelo aprende la distribución de duración (compases) y el número      ║
║    típico de secciones de cada pieza directamente del corpus.                ║
║                                                                              ║
║  MEJORAS v3.0 respecto a v2.0:                                               ║
║                                                                              ║
║  [A] Representación polifónica — extracción de voz soprano, tenor y bajo    ║
║      de MIDIs multi-track. Las transformaciones preservan relaciones         ║
║      entre voces.                                                            ║
║  [B] Representación simbólica — pitches normalizados a grados de escala     ║
║      (1-7) y figuras rítmicas cuantizadas (corchea=1..redonda=8). Más       ║
║      generalizable que MIDI crudo.                                           ║
║  [C] Embedding de frases (autoencoder ligero) — aprende representación      ║
║      latente 16D de las frases para similitud y clustering más finos.       ║
║  [D] Alineación temporal del corpus — normaliza la posición relativa de     ║
║      cada sección antes de hacer clustering, capturando narrativa común.    ║
║  [E] GMM sobre espacio de transformaciones — modela correlaciones entre     ║
║      dimensiones de θ (ej. inversión + diminución van juntas).              ║
║  [F] Evaluación objetiva del modelo — métrica de reconstrucción: dado el   ║
║      source de una pieza del corpus, mide si θ predicha produce una        ║
║      sección más parecida a la real que una baseline aleatoria.             ║
║  [G] Coherencia motívica — extrae el motivo de apertura (4-6 notas) y lo  ║
║      siembra en cada sección con la transformación apropiada.               ║
║  [H] Voice leading continuo — puentes melódicos calculados entre el final   ║
║      de cada sección y el inicio de la siguiente.                           ║
║  [I] Variaciones internas — dentro de cada sección aplica variaciones      ║
║      progresivas: repetición 1 = ornamentada, repetición 2 = contrapunto.  ║
║  [J] Comando inspect — muestra matriz de similitud de roles, arcos típicos  ║
║      del corpus, y progresiones representativas por rol.                    ║
║  [K] Integración con song_architect — genera secciones con nombres          ║
║      canónicos (intro/verse/chorus…) compatibles con song_architect.py.    ║
║  [L] Entrenamiento incremental — modo --update añade MIDIs a un modelo     ║
║      existente sin reentrenar todo el corpus.                               ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    segment   — detecta secciones de un MIDI                                ║
║    train     — entrena modelo sobre corpus                                  ║
║    update    — añade MIDIs a un modelo existente (incremental)              ║
║    generate  — genera obra completa                                          ║
║    inspect   — inspecciona un modelo entrenado                              ║
║                                                                              ║
║  EJEMPLOS:                                                                   ║
║    python ml_architect_v3.py segment  cancion.mid --verbose                 ║
║    python ml_architect_v3.py train    corpus/ -o model.pkl --n-roles 5     ║
║    python ml_architect_v3.py update   model.pkl nuevos/ -o model2.pkl      ║
║    python ml_architect_v3.py generate frase.mid --model m.pkl --arc arch   ║
║    python ml_architect_v3.py generate frase.mid --model m.pkl --sa-compat  ║
║    python ml_architect_v3.py inspect  model.pkl                             ║
║    python ml_architect_v3.py inspect  model.pkl --roles --progressions      ║
║                                                                              ║
║  ARCOS (--arc): flat rise fall arch inverse_arch wave                       ║
║  ASIGNACIÓN (--assignment): similarity position hybrid                      ║
║  DEPENDENCIAS: mido, numpy, scikit-learn, scipy                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, os, math, copy, json, random, pickle, argparse
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

try:
    import mido
except ImportError:
    print("[ERROR] mido no encontrado. pip install mido"); sys.exit(1)

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import normalize as sk_normalize
except ImportError:
    print("[ERROR] scikit-learn no encontrado. pip install scikit-learn"); sys.exit(1)

try:
    from scipy.ndimage import gaussian_filter
    from scipy.signal import find_peaks
except ImportError:
    print("[ERROR] scipy no encontrado. pip install scipy"); sys.exit(1)

VERSION = "5.0"

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES     = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
MIN_NOTE_DUR   = 0.125
MELODY_MIN_DUR = 0.25

SCALE_INTERVALS = {
    "major":    [0,2,4,5,7,9,11],
    "minor":    [0,2,3,5,7,8,10],
    "dorian":   [0,2,3,5,7,9,10],
    "phrygian": [0,1,3,5,7,8,10],
}

CHORD_INTERVALS = {
    "M":[0,4,7],"m":[0,3,7],"7":[0,4,7,10],"M7":[0,4,7,11],"m7":[0,3,7,10],
}

NUMERAL_MAP = {
    "I":(0,"M"),"i":(0,"m"),"II":(2,"M"),"ii":(2,"m"),
    "III":(4,"M"),"iii":(4,"m"),"IV":(5,"M"),"iv":(5,"m"),
    "V":(7,"M"),"v":(7,"m"),"VI":(9,"M"),"vi":(9,"m"),
    "VII":(11,"M"),"vii":(11,"m"),"V7":(7,"7"),"IM7":(0,"M7"),
    "vi7":(9,"m7"),"ii7":(2,"m7"),"bVII":(10,"M"),
}

PITCH_OPS   = ["identity","transpose","invert","retrograde","retro_invert",
               "modal_shift","diatonic_sequence","liquidate"]
RHYTHM_OPS  = ["identity","augment","diminish","rhythmic_repattern","syncopate"]
DYNAMIC_OPS = ["identity","velocity_scale","tension_curve","ornament","add_passing_notes"]
HARMONY_OPS = ["identity","progression_select","acc_style","reharmonize"]
ACC_STYLES  = ["block","arpeggio","alberti","bass_only","waltz"]
TENSION_SHAPES = ["flat","rise","fall","arch","inverse_arch"]
ARC_SHAPES  = ["flat","rise","fall","arch","inverse_arch","wave"]

# [B] Cuantización rítmica — log2(beats)+3 → índice 0..7
RHYTHM_BINS = {0.125:0, 0.25:1, 0.5:2, 1.0:3, 2.0:4, 4.0:5, 8.0:6}
RHYTHM_VALS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

CONSONANCE_MAP = {
    0:1.0, 1:0.1, 2:0.2, 3:0.6, 4:0.7, 5:0.8,
    6:0.0, 7:0.9, 8:0.65, 9:0.75, 10:0.3, 11:0.15,
}

FALLBACK_PROGRESSIONS = [
    [("I",4),("V",4),("vi",4),("IV",4)],
    [("I",4),("vi",4),("IV",4),("V",4)],
    [("IV",4),("V",4),("I",4),("vi",4)],
    [("I",4),("IV",4),("I",4),("I",4)],
    [("ii",4),("V",4),("I",4),("vi",4)],
]

# [K] Mapa de roles a nombres canónicos de song_architect
SA_ROLE_MAP = {
    "intro": "intro", "verse": "verse1", "development": "prechorus",
    "climax": "chorus", "outro": "outro", "bridge": "bridge",
}


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RawNote:
    pitch:    int
    duration: float
    velocity: int
    offset:   float
    voice:    int = 0   # [A] 0=melody, 1=inner, 2=bass

    def transpose(self, s: int) -> "RawNote":
        return RawNote(max(0,min(127,self.pitch+s)), self.duration,
                       self.velocity, self.offset, self.voice)


@dataclass
class SectionSegment:
    label:     str
    bar_start: int
    bar_end:   int
    notes:     List[RawNote]
    symbol:    str


@dataclass
class Motif:
    """[G] Motivo principal extraído del material fuente."""
    scale_degrees: List[int]    # grados de escala (0-6), invariante a transposición
    rhythm_bins:   List[int]    # índices rítmicos cuantizados
    pitches:       List[int]    # pitches originales (para reconstrucción)
    length:        int          # número de notas


@dataclass
class SectionSignature:
    """v3: añade scale_degree_hist [7] (mejora B) y embedding latente [16] (mejora C)."""
    position_ratio:    float
    tension:           float
    harmonic_tension:  float
    velocity_mean:     float
    interval_hist:     np.ndarray  # [24]
    rhythm_hist:       np.ndarray  # [8]
    scale_degree_hist: np.ndarray  # [7]  v3-B
    contour_dir:       float
    density:           float
    latent:            Optional[np.ndarray] = None  # [16] v3-C; None si AE no entrenado

    def to_vector(self) -> np.ndarray:
        ht = getattr(self, "harmonic_tension", 0.0)
        sdh = getattr(self, "scale_degree_hist", np.zeros(7))
        lat = getattr(self, "latent", None)
        base = np.concatenate([
            [self.position_ratio, self.tension, ht,
             self.velocity_mean, self.contour_dir, min(self.density/4.0, 1.0)],
            self.interval_hist,   # 24
            self.rhythm_hist,     # 8
            sdh,                  # 7  v3-B
        ])  # dim = 6+24+8+7 = 45
        if lat is not None:
            return np.concatenate([base, lat])  # dim = 61
        return base


@dataclass
class TransformVector:
    op_pitch:       str   = "identity"
    param_pitch:    float = 0.0
    op_rhythm:      str   = "identity"
    param_rhythm:   float = 1.0
    op_dynamics:    str   = "identity"
    param_dynamics: float = 1.0
    op_harmony:     str   = "identity"
    param_harmony:  float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            PITCH_OPS.index(self.op_pitch)      / max(len(PITCH_OPS)-1,1),
            np.clip(self.param_pitch/12.0,-1,1),
            RHYTHM_OPS.index(self.op_rhythm)    / max(len(RHYTHM_OPS)-1,1),
            np.clip(self.param_rhythm/2.0,0,1),
            DYNAMIC_OPS.index(self.op_dynamics) / max(len(DYNAMIC_OPS)-1,1),
            np.clip(self.param_dynamics/1.5,0,1),
            HARMONY_OPS.index(self.op_harmony)  / max(len(HARMONY_OPS)-1,1),
            np.clip(self.param_harmony/max(len(ACC_STYLES)-1,1),0,1),
        ])

    @staticmethod
    def from_array(arr: np.ndarray) -> "TransformVector":
        def ni(v,n): return int(round(np.clip(v,0,1)*(n-1)))
        return TransformVector(
            op_pitch       = PITCH_OPS[ni(arr[0],len(PITCH_OPS))],
            param_pitch    = float(arr[1])*12.0,
            op_rhythm      = RHYTHM_OPS[ni(arr[2],len(RHYTHM_OPS))],
            param_rhythm   = float(arr[3])*2.0,
            op_dynamics    = DYNAMIC_OPS[ni(arr[4],len(DYNAMIC_OPS))],
            param_dynamics = float(arr[5])*1.5,
            op_harmony     = HARMONY_OPS[ni(arr[6],len(HARMONY_OPS))],
            param_harmony  = float(arr[7])*max(len(ACC_STYLES)-1,1),
        )


@dataclass
class RoleProfile:
    centroid:           SectionSignature
    theta_mean:         TransformVector
    theta_cov:          np.ndarray
    position_dist:      Tuple[float,float]
    tension_dist:       Tuple[float,float]
    harmonic_tens_dist: Tuple[float,float]
    progressions:       List[List[Tuple[str,int]]]
    label:              str = "?"
    # [E] GMM sobre el espacio de θ
    gmm:                Any = None   # GaussianMixture serializable
    # v4: distribuciones de duración aprendidas del corpus
    bars_dist:          Tuple[float,float] = (8.0, 2.0)  # (media, std) compases/sección
    n_sections_dist:    Tuple[float,float] = (4.0, 1.0)  # (media, std) secciones/pieza


@dataclass
class TransformationModel:
    n_roles:         int
    roles:           List[RoleProfile]
    assignment_mode: str
    corpus_size:     int
    knn_regressor:   Any = None
    knn_X:           Any = None
    knn_y:           Any = None
    # [C] Autoencoder ligero (pesos numpy)
    ae_weights:      Any = None   # dict con W1,b1,W2,b2,W3,b3
    # [F] Métrica de evaluación
    eval_score:      float = 0.0
    eval_baseline:   float = 0.0
    version:         str = VERSION


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES MUSICALES
# ══════════════════════════════════════════════════════════════════════════════

def _detect_key(notes: List[RawNote]) -> Tuple[int, str]:
    if not notes: return 0, "major"
    w = np.zeros(12)
    for n in notes: w[n.pitch % 12] += n.duration
    M = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    m = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    best_k, best_s, best_mode = 0,-999.0,"major"
    for r in range(12):
        rot = np.roll(w,-r)
        for name,p in [("major",M),("minor",m)]:
            s = float(np.corrcoef(rot,p)[0,1])
            if s > best_s: best_s=s; best_k=r; best_mode=name
    return best_k, best_mode

def _get_scale_pcs(root_pc: int, mode: str) -> List[int]:
    return [(root_pc+i)%12 for i in SCALE_INTERVALS.get(mode,SCALE_INTERVALS["major"])]

def _snap_to_scale(pitch: int, root_pc: int, mode: str) -> int:
    sc = set(_get_scale_pcs(root_pc, mode))
    if pitch%12 in sc: return pitch
    for d in range(1,7):
        if (pitch+d)%12 in sc: return pitch+d
        if (pitch-d)%12 in sc: return pitch-d
    return pitch

def _numeral_to_root(num: str, key_pc: int) -> Tuple[int,str]:
    e = NUMERAL_MAP.get(num,(0,"M"))
    return (key_pc+e[0])%12, e[1]

def _chord_pitches(root_pc: int, quality: str, octave: int=4,
                   prev: Optional[List[int]]=None) -> List[int]:
    ivs = CHORD_INTERVALS.get(quality,[0,4,7])
    base = root_pc+octave*12
    while base<48: base+=12
    while base>72: base-=12
    raw = [base+i for i in ivs if 24<=base+i<=96]
    if not raw or prev is None: return raw
    def mot(a,b): return sum(abs(b[i]-a[i]) for i in range(min(len(a),len(b))))
    best,bm = raw, mot(prev,raw)
    rot = list(raw)
    for _ in range(len(raw)-1):
        rot = rot[1:]+[rot[0]+12]
        mv = mot(prev,rot)
        if mv<bm: bm=mv; best=list(rot)
    return best

def _score_melodic_interest(notes: List[RawNote]) -> float:
    if len(notes)<2: return 0.0
    ps=[n.pitch for n in notes]; ds=[n.duration for n in notes]; vs=[n.velocity for n in notes]
    pcv=len(set(p%12 for p in ps))/12.0
    pr=min(1.0,(max(ps)-min(ps))/24.0)
    tot=max(n.offset+n.duration for n in notes)
    den=min(1.0,len(notes)/max(tot,1)/2.0)
    vv=min(1.0,float(np.std(vs))/40.0)
    rv=min(1.0,float(np.std(ds))/1.0)
    return pcv*0.3+pr*0.2+den*0.2+vv*0.1+rv*0.2

# [B] Conversión pitch → grado de escala (0-6)
def _pitch_to_scale_degree(pitch: int, root_pc: int, mode: str) -> int:
    sc = _get_scale_pcs(root_pc, mode)
    pc = pitch % 12
    if pc in sc:
        return sc.index(pc)
    # buscar el grado más cercano
    return min(range(len(sc)), key=lambda i: min(abs(sc[i]-pc), 12-abs(sc[i]-pc)))

def _quantize_duration(dur: float) -> int:
    """Cuantiza duración al bin rítmico más cercano (0-6)."""
    best_idx, best_dist = 0, float('inf')
    for i, v in enumerate(RHYTHM_VALS):
        d = abs(dur - v)
        if d < best_dist: best_dist=d; best_idx=i
    return best_idx


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA MIDI
# ══════════════════════════════════════════════════════════════════════════════

def _load_midi_raw(path: str) -> Tuple[mido.MidiFile,int,int,int]:
    mid = mido.MidiFile(path)
    tpb=mid.ticks_per_beat; tempo_us=500_000; bpb=4
    for msg in mid.tracks[0]:
        if msg.type=="set_tempo":       tempo_us=msg.tempo
        elif msg.type=="time_signature": bpb=msg.numerator
    return mid, tpb, round(60_000_000/tempo_us), bpb

def _parse_track_to_notes(track, tpb: int, voice: int=0) -> List[RawNote]:
    active: Dict[int,Tuple[float,int]] = {}
    notes: List[RawNote] = []
    abs_ticks=0
    for msg in track:
        abs_ticks+=msg.time; beat=abs_ticks/tpb
        if msg.type=="note_on" and msg.velocity>0:
            active[msg.note]=(beat,msg.velocity)
        elif msg.type in ("note_off",) or (msg.type=="note_on" and msg.velocity==0):
            if msg.note in active:
                start,vel=active.pop(msg.note); dur=beat-start
                if dur>=MIN_NOTE_DUR:
                    notes.append(RawNote(msg.note,max(MIN_NOTE_DUR,round(dur*4)/4),
                                          vel,start,voice))
    if not notes: return []
    mn=min(n.offset for n in notes)
    for n in notes: n.offset-=mn
    return sorted(notes,key=lambda n:n.offset)

def _choose_melody_track_idx(mid: mido.MidiFile) -> int:
    tpb=mid.ticks_per_beat
    best_idx,best_score=0,-1.0
    for i,track in enumerate(mid.tracks):
        ns=_parse_track_to_notes(track,tpb)
        if ns:
            s=_score_melodic_interest(ns)
            if s>best_score: best_score=s; best_idx=i
    return best_idx

# [A] Extracción polifónica — soprano (más agudo), tenor (medio), bajo (más grave)
def _load_polyphonic(mid: mido.MidiFile) -> Dict[str,List[RawNote]]:
    """Extrae soprano, inner y bass de un MIDI multi-track."""
    tpb = mid.ticks_per_beat
    all_notes: List[RawNote] = []
    for i, track in enumerate(mid.tracks):
        ns = _parse_track_to_notes(track, tpb, voice=i)
        all_notes.extend(ns)
    if not all_notes:
        return {"soprano":[], "inner":[], "bass":[]}
    # Separar por registro
    pitches = [n.pitch for n in all_notes]
    p33 = float(np.percentile(pitches, 33))
    p66 = float(np.percentile(pitches, 66))
    soprano = [n for n in all_notes if n.pitch >= p66]
    inner   = [n for n in all_notes if p33 <= n.pitch < p66]
    bass    = [n for n in all_notes if n.pitch < p33]
    # Asignar voice tag
    for n in soprano: n.voice = 0
    for n in inner:   n.voice = 1
    for n in bass:    n.voice = 2
    return {"soprano": soprano, "inner": inner, "bass": bass}

def _load_all_tracks(mid: mido.MidiFile) -> Dict[int,List[RawNote]]:
    tpb=mid.ticks_per_beat
    return {i: _parse_track_to_notes(t,tpb)
            for i,t in enumerate(mid.tracks) if _parse_track_to_notes(t,tpb)}

def _tick_to_bar(tick: int, tpb: int, bpb: int) -> int:
    return tick//(tpb*bpb)+1

def _segment_to_notes(notes: List[RawNote], bar_from: int, bar_to: int,
                       tpb: int, bpb: int) -> List[RawNote]:
    lo=(bar_from-1)*bpb; hi=bar_to*bpb
    res=[n for n in notes if lo<=n.offset<hi]
    if res:
        mn=min(n.offset for n in res)
        res=[RawNote(n.pitch,n.duration,n.velocity,n.offset-mn,n.voice) for n in res]
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  TENSIÓN ARMÓNICA (Helmholtz)
# ══════════════════════════════════════════════════════════════════════════════

def _harmonic_tension_profile(notes: List[RawNote], tpb: int, bpb: int) -> List[float]:
    if not notes: return []
    total=max(n.offset+n.duration for n in notes)
    n_bars=max(1,int(total/bpb)+1)
    profile=[]
    for bar in range(n_bars):
        lo,hi=bar*bpb,(bar+1)*bpb
        bn=[n for n in notes if lo<=n.offset<hi]
        if not bn: profile.append(0.0); continue
        ps=list(set(n.pitch%12 for n in bn))
        if len(ps)<2: profile.append(0.0); continue
        td=0.0; cnt=0
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                iv=abs(ps[i]-ps[j])%12
                td+=1.0-CONSONANCE_MAP.get(iv,0.5); cnt+=1
        profile.append(td/cnt if cnt>0 else 0.0)
    return profile

def _mean_harmonic_tension(notes: List[RawNote], tpb: int, bpb: int) -> float:
    p=_harmonic_tension_profile(notes,tpb,bpb)
    return float(np.mean(p)) if p else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  [C] AUTOENCODER LIGERO (numpy puro, sin torch/tensorflow)
# ══════════════════════════════════════════════════════════════════════════════

def _relu(x): return np.maximum(0, x)
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def _ae_encode(x: np.ndarray, weights: Dict) -> np.ndarray:
    """Pasa x por el encoder: input→32→16."""
    h = _relu(x @ weights["W1"] + weights["b1"])
    return _relu(h @ weights["W2"] + weights["b2"])

def _ae_decode(z: np.ndarray, weights: Dict) -> np.ndarray:
    """Pasa z por el decoder: 16→32→input."""
    h = _relu(z @ weights["W3"] + weights["b3"])
    return _sigmoid(h @ weights["W4"] + weights["b4"])

def _train_autoencoder(X: np.ndarray, latent_dim: int = 16,
                        epochs: int = 200, lr: float = 0.01,
                        batch_size: int = 32) -> Dict:
    """
    [C] Entrena un autoencoder simple por SGD para aprender embeddings de firmas.
    X: (N, input_dim) normalizado [0,1]
    """
    N, D = X.shape
    np.random.seed(42)
    hidden = 32
    # Inicialización He
    W1 = np.random.randn(D, hidden)      * np.sqrt(2.0/D)
    b1 = np.zeros(hidden)
    W2 = np.random.randn(hidden, latent_dim) * np.sqrt(2.0/hidden)
    b2 = np.zeros(latent_dim)
    W3 = np.random.randn(latent_dim, hidden) * np.sqrt(2.0/latent_dim)
    b3 = np.zeros(hidden)
    W4 = np.random.randn(hidden, D)      * np.sqrt(2.0/hidden)
    b4 = np.zeros(D)

    weights = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3,"W4":W4,"b4":b4}

    for epoch in range(epochs):
        idx = np.random.permutation(N)
        for start in range(0, N, batch_size):
            batch = X[idx[start:start+batch_size]]
            if len(batch) == 0: continue

            # Forward
            h1  = _relu(batch @ W1 + b1)
            z   = _relu(h1 @ W2 + b2)
            h3  = _relu(z @ W3 + b3)
            out = _sigmoid(h3 @ W4 + b4)

            # Loss BCE
            out_c = np.clip(out, 1e-7, 1-1e-7)
            dloss = (out_c - batch) / max(len(batch), 1)

            # Backward decoder
            dh4 = dloss * out_c * (1 - out_c)
            dW4 = h3.T @ dh4; db4 = dh4.sum(0)
            dh3 = dh4 @ W4.T * (h3 > 0)
            dW3 = z.T @ dh3;  db3 = dh3.sum(0)

            # Backward encoder
            dz  = dh3 @ W3.T * (z > 0)
            dW2 = h1.T @ dz;  db2 = dz.sum(0)
            dh1 = dz @ W2.T * (h1 > 0)
            dW1 = batch.T @ dh1; db1 = dh1.sum(0)

            # SGD step
            for name, grad in [("W1",dW1),("b1",db1),("W2",dW2),("b2",db2),
                                 ("W3",dW3),("b3",db3),("W4",dW4),("b4",db4)]:
                weights[name] -= lr * grad

        # Decay lr
        if (epoch+1) % 50 == 0:
            lr *= 0.7

    return weights

def _encode_signature(sig: SectionSignature, ae_weights: Optional[Dict]) -> np.ndarray:
    """Obtiene el embedding latente de una firma si hay AE disponible."""
    if ae_weights is None:
        return np.zeros(16)
    # Input: vector de firma sin el latent (45 dims)
    v = sig.to_vector()[:45]
    # Normalizar
    mn, mx = v.min(), v.max()
    if mx > mn: v = (v - mn) / (mx - mn)
    return _ae_encode(v, ae_weights)


# ══════════════════════════════════════════════════════════════════════════════
#  CÁLCULO DE FIRMA (SectionSignature) — v3
# ══════════════════════════════════════════════════════════════════════════════

def _compute_signature(notes: List[RawNote],
                        position_ratio: float = 0.5,
                        key_pc: int = 0,
                        tpb: int = 480,
                        bpb: int = 4,
                        ae_weights: Optional[Dict] = None) -> SectionSignature:
    if not notes:
        return SectionSignature(
            position_ratio=position_ratio, tension=0.0, harmonic_tension=0.0,
            velocity_mean=0.5, interval_hist=np.zeros(24), rhythm_hist=np.zeros(8),
            scale_degree_hist=np.zeros(7), contour_dir=0.0, density=0.0)

    pitches  = [n.pitch for n in notes]
    durs     = [n.duration for n in notes]
    vels     = [n.velocity for n in notes]
    total    = max(n.offset+n.duration for n in notes)

    # Intervalos relativos a tónica (invariante transposición)
    dk, _ = _detect_key(notes)
    rel   = [(p-dk)%12-6 for p in pitches]
    diffs = [rel[i+1]-rel[i] for i in range(len(rel)-1)]
    ih, _ = np.histogram(diffs, bins=24, range=(-12,12), density=False)
    s = ih.sum(); ih = ih/s if s > 0 else ih.astype(float)

    # Histograma rítmico
    qd  = [min(7,max(0,int(round(math.log2(max(d,0.125))+3)))) for d in durs]
    rh  = np.zeros(8)
    for q in qd: rh[q]+=1
    rs=rh.sum(); rh=rh/rs if rs>0 else rh

    # [B] Histograma de grados de escala
    mode = "major"
    sdh = np.zeros(7)
    for p in pitches:
        deg = _pitch_to_scale_degree(p, key_pc, mode)
        sdh[deg] += 1
    sds = sdh.sum(); sdh = sdh/sds if sds>0 else sdh

    # Tensiones
    scale_pcs = set(_get_scale_pcs(key_pc,"major"))
    oos = sum(1 for p in pitches if p%12 not in scale_pcs)
    tension = oos/len(pitches)
    ht = _mean_harmonic_tension(notes, tpb, bpb)

    intervals = [pitches[i+1]-pitches[i] for i in range(len(pitches)-1)]
    contour  = float(np.mean([1 if i>0 else (-1 if i<0 else 0) for i in intervals])) if intervals else 0.0
    density  = len(notes)/max(total,1.0)
    vel_mean = float(np.mean(vels))/127.0

    sig = SectionSignature(
        position_ratio=position_ratio, tension=float(tension),
        harmonic_tension=float(ht), velocity_mean=vel_mean,
        interval_hist=ih, rhythm_hist=rh, scale_degree_hist=sdh,
        contour_dir=contour, density=density)

    # [C] Añadir embedding si hay AE
    if ae_weights is not None:
        sig.latent = _encode_signature(sig, ae_weights)

    return sig

def _signature_distance(a: SectionSignature, b: SectionSignature) -> float:
    va=a.to_vector(); vb=b.to_vector()
    na=np.linalg.norm(va); nb=np.linalg.norm(vb)
    if na<1e-9 or nb<1e-9: return 1.0
    return float(1.0-np.dot(va/na,vb/nb))

def _signature_diff_vector(src: SectionSignature, tgt: SectionSignature) -> np.ndarray:
    sv=src.to_vector(); tv=tgt.to_vector()
    if len(sv)!=len(tv):
        mn=min(len(sv),len(tv)); sv=sv[:mn]; tv=tv[:mn]
    return tv-sv


# ══════════════════════════════════════════════════════════════════════════════
#  Re-Pair (invariante a transposición — v2+)
# ══════════════════════════════════════════════════════════════════════════════

class _RePair:
    def __init__(self):
        self.rules: Dict[str,List] = {}
        self._next_id = 1
        self.root: List = []

    def _new_rule(self) -> str:
        name=f"R{self._next_id}"; self._next_id+=1; return name

    def build(self, pitches: List[int], key_pc: int=0) -> None:
        def to_rel(p): iv=(p-key_pc)%12; return iv if iv<=6 else iv-12
        seq=[to_rel(p) for p in pitches]
        while True:
            pairs=Counter()
            for i in range(len(seq)-1): pairs[(seq[i],seq[i+1])]+=1
            if not pairs: break
            best,count=pairs.most_common(1)[0]
            if count<2: break
            name=self._new_rule(); self.rules[name]=list(best)
            new=[]; i=0
            while i<len(seq):
                if i<len(seq)-1 and seq[i]==best[0] and seq[i+1]==best[1]:
                    new.append(name); i+=2
                else: new.append(seq[i]); i+=1
            seq=new
        self.root=seq

    def expand(self,sym) -> List[int]:
        if sym not in self.rules: return [sym]
        r=[]
        for s in self.rules[sym]: r.extend(self.expand(s))
        return r

    def expand_with_indices(self,sym,start_idx:int=0) -> Tuple[List[int],List[int]]:
        if sym not in self.rules: return [sym],[start_idx]
        notes,indices=[],[]; pos=start_idx
        for s in self.rules[sym]:
            sn,si=self.expand_with_indices(s,pos)
            notes.extend(sn); indices.extend(si); pos+=len(sn)
        return notes,indices


# ══════════════════════════════════════════════════════════════════════════════
#  SEGMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _ssm(descs: List[np.ndarray]) -> np.ndarray:
    X=np.array(descs,dtype=float)
    norms=np.linalg.norm(X,axis=1,keepdims=True)
    Xn=np.divide(X,norms,out=np.zeros_like(X),where=norms>0)
    return np.dot(Xn,Xn.T)

def _checkerboard_kernel(size: int) -> np.ndarray:
    m=size//2
    k=np.kron(np.array([[1,-1],[-1,1]]),np.ones((m,m)))
    return gaussian_filter(k.astype(float),sigma=size/6)

def _infer_num_sections(notes: List[RawNote], tpb: int, bpb: int,
                         kernel_size:int=8, threshold:float=0.15) -> int:
    bar_ticks=tpb*bpb
    total=max(int((n.offset+n.duration)*tpb) for n in notes)
    n_bars=max(1,total//bar_ticks)
    if n_bars<kernel_size: return max(2,n_bars//2)

    def bar_desc(bar_n):
        lo=(bar_n-1)*bar_ticks/tpb; hi=bar_n*bar_ticks/tpb
        bn=[n for n in notes if lo<=n.offset<hi]
        if not bn: return np.zeros(9)
        ps=[n.pitch for n in bn]; ds=[n.duration for n in bn]; vs=[n.velocity for n in bn]
        ivs=np.diff(ps) if len(ps)>1 else [0]
        ht=_mean_harmonic_tension(bn,tpb,bpb)
        return np.array([
            min(len(bn)/(bar_ticks/tpb)/4,1.0),
            np.mean(ps)/127, np.std(ps)/64 if len(ps)>1 else 0,
            np.mean(vs)/127, np.std(vs)/64 if len(vs)>1 else 0,
            np.mean(ds),
            float(np.mean(np.array(ivs)>0)) if len(ivs)>0 else 0.5,
            float(np.mean(np.abs(np.array(ivs))>5)) if len(ivs)>0 else 0,
            ht,
        ])

    descs=[bar_desc(b) for b in range(1,n_bars+1)]
    ssm=_ssm(descs)
    ks=min(kernel_size,len(descs)//2*2)
    if ks<2: return 2
    kernel=_checkerboard_kernel(ks)
    N=ssm.shape[0]; nov=np.zeros(N); m=ks//2
    for i in range(m,N-m):
        sub=ssm[i-m:i+m,i-m:i+m]
        if sub.shape==kernel.shape: nov[i]=np.sum(sub*kernel)
    mx=np.max(np.abs(nov))
    if mx>0: nov/=mx
    peaks,_=find_peaks(nov,height=threshold,distance=2)
    return int(np.clip(len(peaks)+1,2,max(2,n_bars//4)))

def _compute_segment_vector(syms: List[int], n:int=3, bins:int=12) -> np.ndarray:
    if len(syms)<2: return np.zeros(bins*2)
    ivs=np.diff(syms)
    hist,_=np.histogram(ivs,bins=bins,range=(-12,12),density=True)
    q=[max(-12,min(12,int(i))) for i in ivs]
    ngs=[tuple(q[i:i+n]) for i in range(len(q)-n+1)]
    ngv=np.zeros(bins)
    for ng in ngs: ngv[abs(hash(str(ng))%bins)]+=1
    norm=np.linalg.norm(ngv)
    if norm>0: ngv/=norm
    return np.concatenate([hist,ngv])

def _merge_similar_clusters(labels: np.ndarray, vectors: np.ndarray,
                              sim_thresh: float) -> np.ndarray:
    unique=sorted(set(labels)); centroids={}
    for l in unique:
        idxs=np.where(labels==l)[0]; c=np.mean(vectors[idxs],axis=0)
        nm=np.linalg.norm(c); centroids[l]=c/nm if nm>0 else c
    mapping={l:l for l in unique}
    for l1 in unique:
        for l2 in unique:
            if l1>=l2 or mapping[l1]==mapping[l2]: continue
            sim=float(np.dot(centroids[l1],centroids[l2]))
            if sim>sim_thresh:
                old=mapping[l2]; new=mapping[l1]
                for k in mapping:
                    if mapping[k]==old: mapping[k]=new
    return np.array([mapping[l] for l in labels])

def extract_sections(midi_path: str,
                     num_sections: Optional[int]=None,
                     track_idx: Optional[int]=None,
                     all_tracks: bool=False,
                     similarity_threshold: float=0.85,
                     verbose: bool=False) -> List[SectionSegment]:
    mid,tpb,bpm,bpb=_load_midi_raw(midi_path)
    if track_idx is None: track_idx=_choose_melody_track_idx(mid)
    melody_notes=_parse_track_to_notes(mid.tracks[track_idx],tpb)
    if not melody_notes:
        if verbose: print(f"  [warn] Track {track_idx} sin notas")
        return []
    all_track_notes=_load_all_tracks(mid) if all_tracks else None
    if verbose:
        print(f"  Track melódico: #{track_idx}  |  {len(melody_notes)} notas  "
              f"|  {bpm} BPM  |  {bpb}/4")

    if num_sections is None:
        num_sections=_infer_num_sections(melody_notes,tpb,bpb)
        if verbose: print(f"  Secciones inferidas: {num_sections}")
    else:
        if verbose: print(f"  Secciones forzadas: {num_sections}")

    key_pc,_=_detect_key(melody_notes)
    pitches=[n.pitch for n in melody_notes]
    if len(pitches)<4:
        if verbose: print("  [warn] Secuencia demasiado corta")
        return []

    grammar=_RePair(); grammar.build(pitches,key_pc)
    root_seq=grammar.root
    if verbose:
        print(f"  Reglas Re-Pair: {len(grammar.rules)}  |  Raíz: {len(root_seq)} síms")
    if len(root_seq)==1 and isinstance(root_seq[0],str):
        root_seq=grammar.rules.get(root_seq[0],root_seq)

    segs_raw=[]; cur_idx=0
    for sym in root_seq:
        if isinstance(sym,str) and sym in grammar.rules:
            se,idx=grammar.expand_with_indices(sym,cur_idx)
        else:
            se=[sym] if isinstance(sym,int) else grammar.expand(sym)
            idx=list(range(cur_idx,cur_idx+len(se)))
        segs_raw.append({"symbol":str(sym),"notes_vals":se,"note_indices":idx,"len":len(se)})
        cur_idx+=len(se)

    def simplify(segs,target):
        cur=deepcopy(segs)
        while len(cur)>target:
            ml=min(s["len"] for s in cur); mi=next(i for i,s in enumerate(cur) if s["len"]==ml)
            if mi==0: merge=1
            elif mi==len(cur)-1: merge=mi-1; mi=len(cur)-1
            else:
                p,nx=cur[mi-1]["len"],cur[mi+1]["len"]
                merge=mi if p<=nx else mi+1
            if merge>mi: mi,merge=merge,mi
            a,b=cur[mi],cur[merge]
            cur[mi]={"symbol":f"({a['symbol']}+{b['symbol']})",
                     "notes_vals":a["notes_vals"]+b["notes_vals"],
                     "note_indices":a["note_indices"]+b["note_indices"],
                     "len":a["len"]+b["len"]}
            del cur[merge]
        return cur

    segs_raw=simplify(segs_raw,num_sections)

    vecs=[]
    for seg in segs_raw:
        v=_compute_segment_vector(seg["notes_vals"])
        vecs.append(v if not np.all(v==0) else v+1e-9)
    X=np.array(vecs); k=min(num_sections,len(segs_raw))

    if k<2:
        raw_labels=np.zeros(len(segs_raw),dtype=int)
    else:
        cl=AgglomerativeClustering(n_clusters=k,metric="cosine",linkage="average")
        raw_labels=cl.fit_predict(X)
        raw_labels=_merge_similar_clusters(raw_labels,X,similarity_threshold)

    lmap: Dict[int,str]={}; nc=ord("A"); form_labels=[]
    for l in raw_labels:
        if l not in lmap: lmap[l]=chr(nc); nc+=1
        form_labels.append(lmap[l])

    bar_map={i: _tick_to_bar(int(n.offset*tpb),tpb,bpb) for i,n in enumerate(melody_notes)}
    result: List[SectionSegment]=[]
    for i,(seg,lbl) in enumerate(zip(segs_raw,form_labels)):
        idxs=[idx for idx in seg["note_indices"] if 0<=idx<len(melody_notes)]
        if idxs:
            meas=[bar_map.get(idx,1) for idx in idxs]
            bar_start,bar_end=min(meas),max(meas)
        else: bar_start=bar_end=i+1
        if all_tracks and all_track_notes:
            sn=[]
            for tk in all_track_notes.values():
                sn.extend(_segment_to_notes(tk,bar_start,bar_end,tpb,bpb))
        else:
            sn=_segment_to_notes(melody_notes,bar_start,bar_end,tpb,bpb)
        result.append(SectionSegment(label=lbl,bar_start=bar_start,bar_end=bar_end,
                                      notes=sn,symbol=seg["symbol"]))

    if verbose:
        print(f"  Forma: {''.join(s.label for s in result)}")
        for s in result:
            print(f"    {s.label}  c{s.bar_start}–{s.bar_end}  ({len(s.notes)} notas)")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  [G] MOTIVO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _extract_motif(notes: List[RawNote], key_pc: int, mode: str,
                   length: int = 5) -> Optional[Motif]:
    """Extrae el motivo de apertura como secuencia de grados de escala + ritmo."""
    mel = sorted([n for n in notes if n.duration >= MELODY_MIN_DUR],
                  key=lambda n: n.offset)
    if len(mel) < length: return None
    motif_notes = mel[:length]
    sds = [_pitch_to_scale_degree(n.pitch, key_pc, mode) for n in motif_notes]
    rbs = [_quantize_duration(n.duration) for n in motif_notes]
    pits = [n.pitch for n in motif_notes]
    return Motif(scale_degrees=sds, rhythm_bins=rbs, pitches=pits, length=length)

def _apply_motif_to_notes(notes: List[RawNote], motif: Motif,
                           key_pc: int, mode: str,
                           theta: "TransformVector") -> List[RawNote]:
    """
    [G] 'Siembra' la primera nota de la sección con una versión transformada del motivo.
    Preserva el contorno relativo del motivo original, adaptándolo a la transformación θ.
    """
    if not notes or motif is None: return notes
    result = deepcopy(notes)

    # Transponer el motivo según op_pitch de θ
    scale_pcs = _get_scale_pcs(key_pc, mode)
    first_pitch = result[0].pitch

    motif_pitches: List[int] = []
    for sd in motif.scale_degrees:
        target_pc = scale_pcs[sd % 7]
        octave = first_pitch // 12
        p = octave * 12 + target_pc
        # Acercar a la primera nota
        while abs(p - first_pitch) > 6: p += 12 if p < first_pitch else -12
        motif_pitches.append(max(36, min(96, p)))

    # Aplicar la transformación de pitch al motivo
    if theta.op_pitch == "invert":
        pivot = float(np.mean(motif_pitches))
        motif_pitches = [max(36,min(96,int(round(2*pivot-p)))) for p in motif_pitches]
    elif theta.op_pitch == "retrograde":
        motif_pitches = list(reversed(motif_pitches))

    # Sustituir las primeras N notas de result con el motivo
    rb_to_beats = {0:0.125,1:0.25,2:0.5,3:1.0,4:2.0,5:4.0,6:8.0}
    offset = result[0].offset
    seeded = []
    for k, (p, rb) in enumerate(zip(motif_pitches, motif.rhythm_bins)):
        dur = rb_to_beats.get(rb, 1.0)
        vel = result[min(k, len(result)-1)].velocity
        seeded.append(RawNote(p, dur, vel, offset, voice=0))
        offset += dur

    # Ajustar offset del resto de notas
    if len(result) > len(motif_pitches):
        shift = offset - result[len(motif_pitches)].offset
        tail = [RawNote(n.pitch, n.duration, n.velocity, n.offset + shift, n.voice)
                for n in result[len(motif_pitches):]]
        return seeded + tail
    return seeded


# ══════════════════════════════════════════════════════════════════════════════
#  [H] VOICE LEADING CONTINUO (puentes entre secciones)
# ══════════════════════════════════════════════════════════════════════════════

def _build_bridge(prev_notes: List[RawNote], next_notes: List[RawNote],
                   key_pc: int, mode: str, n_bridge: int = 2) -> List[RawNote]:
    """
    [H] Genera n_bridge notas de conexión entre el final de prev y el inicio de next.
    Las notas del puente están en la escala y forman movimiento por grados.
    """
    if not prev_notes or not next_notes: return []
    mel_prev = sorted([n for n in prev_notes if n.pitch>=48], key=lambda n: n.offset+n.duration)
    mel_next = sorted([n for n in next_notes if n.pitch>=48], key=lambda n: n.offset)
    if not mel_prev or not mel_next: return []

    p_last  = mel_prev[-1].pitch
    p_first = mel_next[0].pitch
    if abs(p_first - p_last) <= 2: return []

    scale_pcs = _get_scale_pcs(key_pc, mode)
    # Construir escala completa en el registro apropiado
    scale_full = []
    for oct in range(3, 7):
        for pc in scale_pcs:
            scale_full.append(oct*12 + pc)
    scale_full = sorted(set(scale_full))

    # Ruta desde p_last hacia p_first por grados de escala
    def nearest_scale(p):
        return min(scale_full, key=lambda s: abs(s-p))

    start = nearest_scale(p_last)
    end   = nearest_scale(p_first)

    if start == end or n_bridge < 1: return []

    step = 1 if end > start else -1
    path = []
    cur  = start
    while cur != end and len(path) < n_bridge:
        candidates = [p for p in scale_full if p*step > cur*step and p*step <= end*step]
        if not candidates: break
        # siguiente nota en la dirección correcta, lo más cercano posible
        nxt = min(candidates, key=lambda p: abs(p - cur - step*2))
        path.append(nxt)
        cur = nxt

    if not path: return []

    # Duración de cada nota del puente: corchea (0.5 beats)
    bridge_dur = 0.5
    bridge_offset = mel_prev[-1].offset + mel_prev[-1].duration
    vel = int(np.mean([mel_prev[-1].velocity, mel_next[0].velocity]))
    return [RawNote(p, bridge_dur, vel, bridge_offset + k*bridge_dur, voice=0)
            for k, p in enumerate(path)]

def _apply_voice_leading(prev: List[RawNote], nxt: List[RawNote],
                          key_pc: int, mode: str) -> List[RawNote]:
    if not prev or not nxt: return nxt
    mel_prev=[n for n in prev if n.pitch>=48]
    mel_next=[n for n in nxt  if n.pitch>=48]
    if not mel_prev or not mel_next: return nxt
    last=max(mel_prev,key=lambda n:n.offset+n.duration)
    first=min(mel_next,key=lambda n:n.offset)
    if abs(first.pitch-last.pitch)<=2: return nxt
    sc=set(_get_scale_pcs(key_pc,mode))
    bp,bd=first.pitch,abs(first.pitch-last.pitch)
    for c in range(max(24,last.pitch-12),min(108,last.pitch+13)):
        if c%12 in sc:
            d=abs(c-last.pitch)
            if d<bd: bd=d; bp=c
    if bp==first.pitch: return nxt
    adj=False; nn=[]
    for n in nxt:
        if not adj and n is first:
            nn.append(RawNote(bp,n.duration,n.velocity,n.offset,n.voice)); adj=True
        else: nn.append(n)
    return nn


# ══════════════════════════════════════════════════════════════════════════════
#  [I] VARIACIONES INTERNAS PROGRESIVAS
# ══════════════════════════════════════════════════════════════════════════════

def _apply_internal_variations(notes: List[RawNote], repetition: int,
                                 key_pc: int, mode: str,
                                 rng: random.Random) -> List[RawNote]:
    """
    [I] Aplica variaciones crecientes según el número de repetición dentro de la sección.
    repetition=0: original, 1: ornamentado, 2: contrapunto, 3+: aumentado
    """
    if not notes or repetition == 0: return notes
    result = deepcopy(notes)

    if repetition == 1:
        # Ornamentación ligera: añadir notas de paso en saltos grandes
        ornamented = []
        for i, n in enumerate(result):
            ornamented.append(n)
            if i+1 < len(result) and n.duration >= 1.0:
                nxt = result[i+1]
                if abs(nxt.pitch - n.pitch) >= 4 and rng.random() < 0.6:
                    pp = _snap_to_scale((n.pitch+nxt.pitch)//2, key_pc, mode)
                    half = max(0.5, n.duration/2)
                    ornamented[-1] = RawNote(n.pitch, half, n.velocity, n.offset, n.voice)
                    ornamented.append(RawNote(pp, half, max(20,n.velocity-8),
                                               n.offset+half, 1))
        result = ornamented

    elif repetition == 2:
        # Contrapunto simple: duplicar en tercera superior snapped a escala
        scale_pcs = _get_scale_pcs(key_pc, mode)
        counterpoint = []
        for n in result:
            if n.duration >= 0.5 and rng.random() < 0.5:
                # Tercera diatónica superior
                pc = n.pitch % 12
                oct = n.pitch // 12
                if pc in scale_pcs:
                    idx = scale_pcs.index(pc)
                    third_pc = scale_pcs[(idx+2) % 7]
                    third_oct = oct if third_pc >= pc else oct+1
                    third_pitch = third_oct*12 + third_pc
                    if 36 <= third_pitch <= 96:
                        counterpoint.append(RawNote(third_pitch, n.duration,
                                                     max(20, n.velocity-15),
                                                     n.offset, 1))
        result = result + counterpoint

    else:
        # Aumentación leve: expandir duraciones un 10%
        factor = 1.1
        result = [RawNote(n.pitch, max(0.25, n.duration*factor),
                           n.velocity, n.offset*factor, n.voice)
                  for n in result]

    return sorted(result, key=lambda n: n.offset)


# ══════════════════════════════════════════════════════════════════════════════
#  [D] ALINEACIÓN TEMPORAL DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def _align_position_ratio(sections: List[SectionSegment]) -> List[float]:
    """
    [D] Normaliza las posiciones relativas de las secciones dentro de una pieza
    usando DTW simplificado: mapea posiciones a [0, 1] uniformemente.
    Esto permite que el clustering compare secciones con la misma función
    dramática aunque la pieza tenga distinto número de secciones.
    """
    n = len(sections)
    if n <= 1: return [0.0]
    # Posición normalizada lineal + pequeño jitter según peso de cada sección
    weights = [max(1, len(s.notes)) for s in sections]
    cumw = np.cumsum([0] + weights)
    total = cumw[-1]
    return [float(cumw[i]/total) for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
#  ESTIMACIÓN DE θ + KNN + [E] GMM
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_transform_heuristic(src: SectionSignature,
                                   tgt: SectionSignature) -> TransformVector:
    tv = TransformVector()
    cd = tgt.contour_dir - src.contour_dir
    src_s = float(np.sum(src.interval_hist[10:14]))
    tgt_s = float(np.sum(tgt.interval_hist[10:14]))
    src_l = float(np.sum(src.interval_hist[:4])+np.sum(src.interval_hist[20:]))
    tgt_l = float(np.sum(tgt.interval_hist[:4])+np.sum(tgt.interval_hist[20:]))
    ht_src = getattr(src,"harmonic_tension",0.0)
    ht_tgt = getattr(tgt,"harmonic_tension",0.0)

    if abs(cd)>0.5: tv.op_pitch="invert"; tv.param_pitch=cd
    elif tgt_s>src_s+0.15: tv.op_pitch="diatonic_sequence"; tv.param_pitch=1.0 if cd>=0 else -1.0
    elif tgt_l>src_l+0.15:
        sc=float(np.sum(np.arange(-12,12)*src.interval_hist))
        tc=float(np.sum(np.arange(-12,12)*tgt.interval_hist))
        tv.op_pitch="transpose"; tv.param_pitch=float(np.clip(tc-sc,-12,12))
    elif ht_tgt>ht_src+0.2: tv.op_pitch="modal_shift"; tv.param_pitch=1.0
    else: tv.op_pitch="identity"; tv.param_pitch=0.0

    dr=tgt.density/max(src.density,0.01)
    if dr>1.4: tv.op_rhythm="diminish"; tv.param_rhythm=1.0/min(dr,2.0)
    elif dr<0.7: tv.op_rhythm="augment"; tv.param_rhythm=1.0/max(dr,0.5)
    else: tv.op_rhythm="identity"; tv.param_rhythm=1.0

    vr=tgt.velocity_mean/max(src.velocity_mean,0.01)
    htd=ht_tgt-ht_src
    if abs(vr-1.0)>0.15: tv.op_dynamics="velocity_scale"; tv.param_dynamics=float(np.clip(vr,0.5,1.5))
    elif abs(htd)>0.2:
        tv.op_dynamics="tension_curve"
        if htd>0.3: tv.param_dynamics=1.0
        elif htd<-0.3: tv.param_dynamics=2.0
        elif ht_tgt>0.6: tv.param_dynamics=3.0
        else: tv.param_dynamics=4.0
    else: tv.op_dynamics="identity"; tv.param_dynamics=1.0

    if ht_tgt>0.6: tv.op_harmony="progression_select"; tv.param_harmony=2.0
    elif ht_tgt<0.2: tv.op_harmony="progression_select"; tv.param_harmony=3.0
    else: tv.op_harmony="acc_style"; tv.param_harmony=float(ACC_STYLES.index("arpeggio"))
    return tv

def _build_knn_regressor(X: np.ndarray, y: np.ndarray, k: int=5):
    if len(X)<k+1: return None
    knn=KNeighborsRegressor(n_neighbors=min(k,len(X)),weights="distance",metric="cosine")
    knn.fit(X,y); return knn

def _predict_transform_knn(knn, diff_vec: np.ndarray, fallback: TransformVector) -> TransformVector:
    if knn is None: return fallback
    try:
        pred=knn.predict(diff_vec.reshape(1,-1))[0]
        return TransformVector.from_array(np.clip(pred,0,1))
    except Exception: return fallback

def _build_gmm(theta_arrays: np.ndarray, n_components: int=3):
    """[E] Entrena un GMM sobre el espacio de transformaciones."""
    if len(theta_arrays) < n_components*2: return None
    try:
        gmm=GaussianMixture(n_components=min(n_components,len(theta_arrays)//2),
                             covariance_type="full",random_state=42,max_iter=100)
        gmm.fit(theta_arrays); return gmm
    except Exception: return None

def _sample_transform_gmm(gmm, fallback: TransformVector,
                            arc_val: float) -> TransformVector:
    """[E] Samplea θ del GMM. Usa el componente cuya media tiene tensión dinámica
    más cercana al arc_val."""
    if gmm is None: return fallback
    try:
        means = gmm.means_  # (n_comp, 8)
        # El dim 4 es op_dynamics (índice normalizado) — correlaciona con tensión
        dyn_vals = means[:, 4]
        best_comp = int(np.argmin(np.abs(dyn_vals - arc_val)))
        sample = gmm.means_[best_comp].copy()
        # Añadir algo de varianza del componente
        cov = gmm.covariances_[best_comp]
        noise = np.random.multivariate_normal(np.zeros(8), cov*0.05)
        return TransformVector.from_array(np.clip(sample+noise, 0, 1))
    except Exception: return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  [F] EVALUACIÓN OBJETIVA DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate_model(model: "TransformationModel",
                     eval_samples: List[Tuple[SectionSignature,
                                               SectionSignature,
                                               TransformVector]]) -> Tuple[float,float]:
    """
    [F] Mide la calidad del modelo comparando la distancia entre la sección predicha
    y la real, versus una baseline aleatoria.

    Para cada (source_sig, real_target_sig, real_theta):
      1. Predecir θ_pred con el KNN
      2. Simular la firma resultante de aplicar θ_pred (aproximación: ajustar
         tensión y densidad según los operadores)
      3. Comparar dist(simulated, real_target) vs dist(random_theta, real_target)

    Devuelve (model_score, baseline_score). Menor = mejor.
    """
    if not eval_samples or model.knn_regressor is None:
        return 0.0, 0.0

    model_dists = []
    base_dists  = []

    for src_sig, tgt_sig, real_theta in eval_samples:
        # Predicción del modelo
        diff = _signature_diff_vector(src_sig, tgt_sig)
        pred_tv = _predict_transform_knn(model.knn_regressor, diff,
                                          _estimate_transform_heuristic(src_sig, tgt_sig))
        # Aproximación de la firma resultante: aplicar delta de tensión y densidad
        sim_ht = src_sig.harmonic_tension
        if pred_tv.op_dynamics == "tension_curve":
            sim_ht = min(1.0, src_sig.harmonic_tension + 0.2)
        elif pred_tv.op_dynamics == "velocity_scale":
            sim_ht = src_sig.harmonic_tension * pred_tv.param_dynamics
        sim_density = src_sig.density
        if pred_tv.op_rhythm == "augment":
            sim_density *= 0.5
        elif pred_tv.op_rhythm == "diminish":
            sim_density *= 1.5

        # Firma simulada (usando valores del src + delta predicho)
        sim_v = src_sig.to_vector().copy()
        sim_v[2] = np.clip(sim_ht, 0, 1)
        sim_v[5] = np.clip(sim_density/4.0, 0, 1)

        tgt_v = tgt_sig.to_vector()
        mn = min(len(sim_v), len(tgt_v))
        d_model = float(np.linalg.norm(sim_v[:mn] - tgt_v[:mn]))
        model_dists.append(d_model)

        # Baseline: θ aleatorio
        rand_tv = TransformVector(
            op_pitch   = PITCH_OPS[np.random.randint(len(PITCH_OPS))],
            op_rhythm  = RHYTHM_OPS[np.random.randint(len(RHYTHM_OPS))],
            op_dynamics= DYNAMIC_OPS[np.random.randint(len(DYNAMIC_OPS))],
            op_harmony = HARMONY_OPS[np.random.randint(len(HARMONY_OPS))],
        )
        rand_v = src_sig.to_vector().copy()
        # Baseline no cambia nada → distancia es simplemente dist(src, tgt)
        d_base = float(np.linalg.norm(src_sig.to_vector()[:mn] - tgt_v[:mn]))
        base_dists.append(d_base)

    return float(np.mean(model_dists)), float(np.mean(base_dists))


# ══════════════════════════════════════════════════════════════════════════════
#  PROGRESIONES DEL CORPUS
# ══════════════════════════════════════════════════════════════════════════════

def _extract_progression_from_notes(notes: List[RawNote], key_pc: int,
                                     bpb: int=4) -> List[Tuple[str,int]]:
    if not notes: return []
    total=max(n.offset+n.duration for n in notes)
    n_bars=max(1,int(total/bpb)+1)
    Mp=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    prog=[]
    for bar in range(n_bars):
        lo,hi=bar*bpb,(bar+1)*bpb
        bn=[n for n in notes if lo<=n.offset<hi]
        if not bn: prog.append(("I",bpb)); continue
        w=np.zeros(12)
        for n in bn: w[n.pitch%12]+=n.duration
        best_r,best_s=0,-999.0
        for r in range(12):
            s=float(np.corrcoef(np.roll(w,-r),Mp)[0,1])
            if s>best_s: best_s=s; best_r=r
        sem=(best_r-key_pc)%12
        nmap={0:"I",2:"II",4:"III",5:"IV",7:"V",9:"VI",11:"VII"}
        prog.append((nmap.get(sem,"I"),bpb))
    compressed=[]
    for n,d in prog:
        if compressed and compressed[-1][0]==n: compressed[-1]=(n,compressed[-1][1]+d)
        else: compressed.append((n,d))
    return compressed or [("I",bpb*4)]


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def _compat_model(model: "TransformationModel") -> None:
    """Parchea atributos ausentes de modelos v1/v2/v3."""
    for role in model.roles:
        if not hasattr(role,"harmonic_tens_dist"): role.harmonic_tens_dist=(0.3,0.1)
        if not hasattr(role,"progressions") or not role.progressions:
            role.progressions=FALLBACK_PROGRESSIONS[:2]
        if not hasattr(role,"gmm"): role.gmm=None
        # v4: distribuciones de duración
        if not hasattr(role,"bars_dist"): role.bars_dist=(8.0,2.0)
        if not hasattr(role,"n_sections_dist"): role.n_sections_dist=(4.0,1.0)
        cent=role.centroid
        if not hasattr(cent,"harmonic_tension"): cent.__dict__["harmonic_tension"]=0.3
        if not hasattr(cent,"scale_degree_hist"): cent.__dict__["scale_degree_hist"]=np.zeros(7)
        if not hasattr(cent,"latent"): cent.__dict__["latent"]=None
    for attr in ["knn_regressor","knn_X","knn_y","ae_weights","eval_score","eval_baseline"]:
        if not hasattr(model,attr): setattr(model,attr,None if attr not in ("eval_score","eval_baseline") else 0.0)
    if not hasattr(model,"version"): model.version="1.0"


def train(midi_dir: str,
          output_model: str,
          n_roles: int = 4,
          assignment_mode: str = "hybrid",
          num_sections: Optional[int] = None,
          existing_model_path: Optional[str] = None,   # [L] incremental
          verbose: bool = False) -> "TransformationModel":

    midi_files = (sorted(Path(midi_dir).glob("**/*.mid")) +
                  sorted(Path(midi_dir).glob("**/*.midi")))
    if not midi_files:
        print(f"[ERROR] No se encontraron MIDIs en {midi_dir}"); sys.exit(1)

    # [L] Modo incremental: cargar modelo existente
    existing_sigs: List[SectionSignature] = []
    existing_transforms: List[TransformVector] = []
    existing_diffs: List[np.ndarray] = []
    existing_corpus_size = 0

    if existing_model_path and Path(existing_model_path).exists():
        print(f"  [L] Modo incremental: cargando modelo base de {existing_model_path}")
        with open(existing_model_path,"rb") as f:
            base_model: TransformationModel = pickle.load(f)
        _compat_model(base_model)
        if base_model.knn_X is not None and base_model.knn_y is not None:
            # Reconstruir firmas/transforms desde la info guardada
            existing_corpus_size = base_model.corpus_size
            print(f"  Modelo base: {existing_corpus_size} MIDIs, {len(base_model.knn_X)} secciones")

    print(f"\n  Corpus: {len(midi_files)} MIDIs en {midi_dir}")
    print(f"  Roles: {n_roles}  |  Asignación: {assignment_mode}\n")

    all_signatures:  List[SectionSignature] = []
    all_transforms:  List[TransformVector]  = []
    all_diff_vecs:   List[np.ndarray]       = []
    eval_samples:    List[Tuple]            = []
    # v4: recoger duración real (compases) por sección y nº de secciones por pieza
    # all_section_bars[k] = compases de la k-ésima sección en all_signatures
    all_section_bars:  List[float]          = []
    # all_piece_nsecs[k] = nº de secciones de la pieza que contiene la sección k
    all_piece_nsecs:   List[int]            = []
    corpus_size = 0

    # Primera pasada: extraer firmas y transformaciones
    for i, midi_path in enumerate(midi_files):
        try:
            mid, tpb, bpm, bpb = _load_midi_raw(str(midi_path))
            sections = extract_sections(str(midi_path),
                                         num_sections=num_sections, verbose=False)
            if len(sections) < 2: continue

            source_notes = [n for s in sections for n in s.notes
                            if n.duration >= MELODY_MIN_DUR]
            if not source_notes: continue

            key_pc,_ = _detect_key(source_notes)
            source_sig = _compute_signature(source_notes, 0.0, key_pc, tpb, bpb)

            # [D] Alineación temporal
            aligned_positions = _align_position_ratio(sections)
            total_sections = len(sections)

            for j, (section, aligned_pos) in enumerate(zip(sections, aligned_positions)):
                mel = [n for n in section.notes if n.duration >= MELODY_MIN_DUR]
                if not mel: continue
                sig   = _compute_signature(mel, aligned_pos, key_pc, tpb, bpb)
                theta = _estimate_transform_heuristic(source_sig, sig)
                diff  = _signature_diff_vector(source_sig, sig)
                all_signatures.append(sig)
                all_transforms.append(theta)
                all_diff_vecs.append(diff)
                # v4: medir duración real de esta sección en compases de 4/4 equivalente
                sec_bars = max(section.bar_end - section.bar_start + 1, 1)
                # Convertir compases del compás original a compases 4/4 equivalente
                sec_bars_4 = max(1.0, sec_bars * bpb / 4.0)
                all_section_bars.append(sec_bars_4)
                all_piece_nsecs.append(total_sections)
                # Guardar para evaluación (10% del corpus)
                if i % 10 == 0:
                    eval_samples.append((source_sig, sig, theta))

            corpus_size += 1
            if verbose:
                form = "".join(s.label for s in sections)
                print(f"  [{i+1:3d}/{len(midi_files)}] {midi_path.name:<40}"
                      f"  forma={form}  pos_alineadas={[f'{p:.2f}' for p in aligned_positions]}")
            elif (i+1) % 10 == 0 or i == 0:
                print(f"  Procesados: {i+1}/{len(midi_files)}  (secs: {len(all_signatures)})")

        except Exception as e:
            if verbose: print(f"  [warn] {Path(midi_path).name}: {e}")
            continue

    # [L] Añadir datos del modelo base si hay
    if existing_model_path and Path(existing_model_path).exists() and \
       base_model.knn_X is not None:
        # Reconstruir sigs/transforms desde vectores guardados
        prev_diffs   = base_model.knn_X
        prev_thetas  = base_model.knn_y
        for diff, theta_arr in zip(prev_diffs, prev_thetas):
            all_diff_vecs.append(diff)
            all_transforms.append(TransformVector.from_array(theta_arr))
        corpus_size += existing_corpus_size
        print(f"  [L] Añadidos {len(prev_diffs)} ejemplos del modelo base")

    if len(all_signatures) < n_roles:
        print(f"[ERROR] {len(all_signatures)} secciones insuficientes para {n_roles} roles.")
        sys.exit(1)

    print(f"\n  Secciones totales: {len(all_signatures)}  |  MIDIs: {corpus_size}")

    # [C] Entrenar autoencoder
    print(f"  Entrenando autoencoder (dim latente 16)...")
    sig_vecs_base = np.array([s.to_vector()[:45] for s in all_signatures])
    # Normalizar al rango [0,1]
    v_min = sig_vecs_base.min(axis=0)
    v_max = sig_vecs_base.max(axis=0)
    v_range = np.where(v_max > v_min, v_max - v_min, 1.0)
    sig_vecs_norm = (sig_vecs_base - v_min) / v_range
    ae_weights = _train_autoencoder(sig_vecs_norm, latent_dim=16, epochs=150)
    print(f"  Autoencoder entrenado.")

    # Añadir embeddings latentes a todas las firmas
    for sig, raw_v in zip(all_signatures, sig_vecs_norm):
        sig.latent = _ae_encode(raw_v, ae_weights)

    # Clustering con vectores completos (incluyendo latente)
    print(f"  Clustering en {n_roles} roles...")
    sig_vecs_full = np.array([s.to_vector() for s in all_signatures])
    # Normalizar para clustering
    norms = np.linalg.norm(sig_vecs_full, axis=1, keepdims=True)
    sig_vecs_norm_full = np.where(norms > 0, sig_vecs_full / norms, sig_vecs_full)

    clustering = AgglomerativeClustering(n_clusters=n_roles,
                                          metric="cosine", linkage="average")
    labels = clustering.fit_predict(sig_vecs_norm_full)

    roles: List[RoleProfile] = []
    for role_idx in range(n_roles):
        idxs = [i for i, l in enumerate(labels) if l == role_idx]
        if not idxs: continue
        role_sigs   = [all_signatures[i] for i in idxs]
        role_thetas = [all_transforms[i]  for i in idxs]

        mean_v    = np.mean([s.to_vector()[:45] for s in role_sigs], axis=0)
        mean_lat  = np.mean([s.latent for s in role_sigs if s.latent is not None], axis=0) \
                    if any(s.latent is not None for s in role_sigs) else np.zeros(16)
        centroid  = SectionSignature(
            position_ratio   = float(mean_v[0]),
            tension          = float(mean_v[1]),
            harmonic_tension = float(mean_v[2]),
            velocity_mean    = float(mean_v[3]),
            contour_dir      = float(mean_v[4]),
            density          = float(mean_v[5])*4.0,
            interval_hist    = mean_v[6:30],
            rhythm_hist      = mean_v[30:38],
            scale_degree_hist= mean_v[38:45],
            latent           = mean_lat,
        )

        theta_arrs     = np.array([t.to_array() for t in role_thetas])
        theta_mean     = TransformVector.from_array(np.mean(theta_arrs, axis=0))
        theta_cov      = np.cov(theta_arrs.T) if len(idxs)>1 else np.eye(8)*0.01

        positions = [all_signatures[i].position_ratio  for i in idxs]
        tensions  = [all_signatures[i].tension          for i in idxs]
        ht_vals   = [getattr(all_signatures[i],"harmonic_tension",0.0) for i in idxs]
        pos_dist  = (float(np.mean(positions)), float(np.std(positions))+1e-6)
        ten_dist  = (float(np.mean(tensions)),  float(np.std(tensions))+1e-6)
        ht_dist   = (float(np.mean(ht_vals)),   float(np.std(ht_vals))+1e-6)

        # [E] GMM sobre θ del rol
        gmm = _build_gmm(theta_arrs, n_components=min(3,len(idxs)//3+1))

        # v4: distribuciones de duración aprendidas
        # all_section_bars puede ser más corto que all_signatures si algunas secs
        # no aportaron señal — usar len mínimo para seguridad
        bars_list  = [all_section_bars[k]  for k in idxs if k < len(all_section_bars)]
        nsecs_list = [all_piece_nsecs[k]   for k in idxs if k < len(all_piece_nsecs)]
        if bars_list:
            bars_dist  = (float(np.mean(bars_list)),  max(0.5, float(np.std(bars_list))))
        else:
            bars_dist  = (8.0, 2.0)
        if nsecs_list:
            nsecs_dist = (float(np.mean(nsecs_list)), max(0.5, float(np.std(nsecs_list))))
        else:
            nsecs_dist = (4.0, 1.0)

        # Etiqueta
        pm = pos_dist[0]; htm = ht_dist[0]
        if   pm < 0.2:  lbl = "intro"
        elif pm > 0.8:  lbl = "outro"
        elif htm > 0.55: lbl = "climax"
        elif htm > 0.35: lbl = "development"
        else:            lbl = "verse"

        roles.append(RoleProfile(
            centroid=centroid, theta_mean=theta_mean, theta_cov=theta_cov,
            position_dist=pos_dist, tension_dist=ten_dist, harmonic_tens_dist=ht_dist,
            progressions=[], label=lbl, gmm=gmm,
            bars_dist=bars_dist, n_sections_dist=nsecs_dist))

    # Extraer progresiones reales por rol
    print(f"  Extrayendo progresiones del corpus...")
    prog_by_role: Dict[int,List] = {i: [] for i in range(len(roles))}
    for midi_path in midi_files[:min(len(midi_files),80)]:
        try:
            mid, tpb, bpm, bpb = _load_midi_raw(str(midi_path))
            sections = extract_sections(str(midi_path), num_sections=num_sections,
                                         verbose=False)
            if not sections: continue
            src = [n for s in sections for n in s.notes if n.duration>=MELODY_MIN_DUR]
            if not src: continue
            kpc,_ = _detect_key(src)
            src_sig = _compute_signature(src, 0.0, kpc, tpb, bpb)
            for j, section in enumerate(sections):
                mel = [n for n in section.notes if n.duration>=MELODY_MIN_DUR]
                if not mel: continue
                pos = j/max(len(sections)-1,1)
                sig = _compute_signature(mel, pos, kpc, tpb, bpb)
                br  = min(range(len(roles)), key=lambda ri: _signature_distance(sig,roles[ri].centroid))
                prog = _extract_progression_from_notes(mel, kpc, bpb)
                if prog: prog_by_role[br].append(prog)
        except Exception: continue

    for i,role in enumerate(roles):
        role.progressions = prog_by_role[i][:10] if prog_by_role[i] else FALLBACK_PROGRESSIONS[:2]

    # [F] Evaluar modelo
    print(f"  Evaluando modelo...")
    diff_matrix  = np.array(all_diff_vecs) if all_diff_vecs else np.zeros((1,45))
    theta_matrix = np.array([t.to_array() for t in all_transforms])
    knn = _build_knn_regressor(diff_matrix, theta_matrix, k=5)
    model_tmp = TransformationModel(
        n_roles=len(roles), roles=roles, assignment_mode=assignment_mode,
        corpus_size=corpus_size, knn_regressor=knn,
        knn_X=diff_matrix, knn_y=theta_matrix, ae_weights=ae_weights)
    model_score, baseline_score = _evaluate_model(model_tmp, eval_samples[:50])
    improvement = ((baseline_score - model_score) / max(baseline_score,1e-6)) * 100
    print(f"  Evaluación: score={model_score:.4f}  baseline={baseline_score:.4f}"
          f"  mejora={improvement:.1f}%")

    roles.sort(key=lambda r: r.position_dist[0])
    model = TransformationModel(
        n_roles=len(roles), roles=roles, assignment_mode=assignment_mode,
        corpus_size=corpus_size, knn_regressor=knn,
        knn_X=diff_matrix, knn_y=theta_matrix,
        ae_weights=ae_weights, eval_score=model_score, eval_baseline=baseline_score)

    out_path = Path(output_model); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path,"wb") as f: pickle.dump(model, f)
    print(f"\n  Modelo guardado: {out_path}")
    print(f"  Roles aprendidos:")
    for i,r in enumerate(model.roles):
        bd = getattr(r, "bars_dist", (8.0, 2.0))
        nd = getattr(r, "n_sections_dist", (4.0, 1.0))
        print(f"    [{i}] {r.label:<15}  pos={r.position_dist[0]:.2f}±{r.position_dist[1]:.2f}"
              f"  ht={r.harmonic_tens_dist[0]:.2f}  gmm={'sí' if r.gmm else 'no'}"
              f"  bars={bd[0]:.1f}±{bd[1]:.1f}  n_secs={nd[0]:.1f}±{nd[1]:.1f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  APLICACIÓN DE TRANSFORMACIONES
# ══════════════════════════════════════════════════════════════════════════════

def _apply_transform(notes: List[RawNote], theta: TransformVector,
                      key_pc: int, mode: str, target_bars: int,
                      rng: random.Random) -> List[RawNote]:
    if not notes: return []
    result = deepcopy(notes)
    op=theta.op_pitch; param=theta.param_pitch

    if op=="transpose":
        s=int(round(np.clip(param,-12,12)))
        result=[RawNote(max(0,min(127,n.pitch+s)),n.duration,n.velocity,n.offset,n.voice) for n in result]
    elif op=="invert":
        pv=float(np.mean([n.pitch for n in result]))
        result=[RawNote(max(0,min(127,int(round(2*pv-n.pitch)))),n.duration,n.velocity,n.offset,n.voice) for n in result]
        result=[RawNote(_snap_to_scale(n.pitch,key_pc,mode),n.duration,n.velocity,n.offset,n.voice) for n in result]
    elif op=="retrograde":
        ps=[n.pitch for n in reversed(result)]
        result=[RawNote(p,n.duration,n.velocity,n.offset,n.voice) for p,n in zip(ps,result)]
    elif op=="retro_invert":
        ps=[n.pitch for n in result]; pv=float(np.mean(ps))
        ir=list(reversed([int(round(2*pv-p)) for p in ps]))
        result=[RawNote(max(0,min(127,_snap_to_scale(p,key_pc,mode))),n.duration,n.velocity,n.offset,n.voice) for p,n in zip(ir,result)]
    elif op=="modal_shift":
        ps=[n.pitch for n in result]; np2=[ps[0]]
        for i in range(len(ps)-1):
            d=ps[i+1]-ps[i]
            d=3 if d==4 else(-3 if d==-4 else(4 if d==3 else(-4 if d==-3 else d)))
            np2.append(max(0,min(127,np2[-1]+d)))
        result=[RawNote(p,n.duration,n.velocity,n.offset,n.voice) for p,n in zip(np2,result)]
    elif op=="diatonic_sequence":
        steps=int(round(np.clip(param,-4,4))); sc=_get_scale_pcs(key_pc,mode)
        def dt(pitch,ns):
            pc=pitch%12; idx=sc.index(pc) if pc in sc else min(range(len(sc)),key=lambda i:abs(sc[i]-pc))
            ni=(idx+ns)%7; eo=(idx+ns)//7
            return 12*(pitch//12+eo)+sc[ni]
        result=[RawNote(max(0,min(127,dt(n.pitch,steps))),n.duration,n.velocity,n.offset,n.voice) for n in result]
    elif op=="liquidate":
        keep=max(2,int(round(abs(param))) if param!=0 else 3)
        strong=[n for n in result if n.offset%1.0<0.1]
        result=strong[:keep] if len(strong)>=keep else result[:keep]

    op=theta.op_rhythm; param=theta.param_rhythm
    if op=="augment":
        f=float(np.clip(param,1.0,2.0))
        result=[RawNote(n.pitch,n.duration*f,n.velocity,n.offset*f,n.voice) for n in result]
    elif op=="diminish":
        f=float(np.clip(param,0.5,1.0))
        result=[RawNote(n.pitch,max(0.25,n.duration*f),n.velocity,n.offset*f,n.voice) for n in result]
    elif op=="rhythmic_repattern":
        ps=[n.pitch for n in result]; vs=[n.velocity for n in result]
        pats=[[1.0,0.5,0.5,1.0,1.0],[0.5,0.5,0.5,0.5,1.0,1.0],[0.25,0.25,0.5,1.0],[2.0,0.5,0.5,1.0]]
        pat=pats[int(round(param))%len(pats)]; tb=target_bars*4
        cur=0.0; nn=[]; pi=0; pati=0
        while cur<tb-0.05 and pi<len(ps):
            d=min(pat[pati%len(pat)],tb-cur)
            nn.append(RawNote(ps[pi%len(ps)],d,vs[pi%len(vs)],cur,0))
            cur+=d; pi+=1; pati+=1
        result=nn
    elif op=="syncopate":
        sh=float(np.clip(param if param!=1.0 else 0.25,0.125,0.5))
        result=[RawNote(n.pitch,n.duration,n.velocity,n.offset+(sh if n.offset%1.0<0.1 else 0.0),n.voice) for n in result]

    op=theta.op_dynamics; param=theta.param_dynamics
    if op=="velocity_scale":
        r=float(np.clip(param,0.5,1.5))
        result=[RawNote(n.pitch,n.duration,int(np.clip(n.velocity*r,20,120)),n.offset,n.voice) for n in result]
    elif op=="tension_curve":
        shape=TENSION_SHAPES[int(round(param))%len(TENSION_SHAPES)]
        tot=max(n.offset+n.duration for n in result) if result else 1.0
        def vs(off):
            t=off/max(tot,1.0)
            if shape=="flat":    return 1.0
            elif shape=="rise":  return 0.7+0.3*t
            elif shape=="fall":  return 1.0-0.3*t
            elif shape=="arch":  return 0.7+0.6*math.sin(math.pi*t)
            else:                return 1.0-0.3*math.sin(math.pi*t)
        result=[RawNote(n.pitch,n.duration,int(np.clip(n.velocity*vs(n.offset),20,120)),n.offset,n.voice) for n in result]
    elif op=="ornament":
        d=float(np.clip(param,0.0,1.0)); orn=[]
        for i,n in enumerate(result):
            orn.append(n)
            if i+1<len(result) and n.duration>=1.0 and rng.random()<d:
                nx=result[i+1]
                if abs(nx.pitch-n.pitch)>=4:
                    pp=_snap_to_scale((n.pitch+nx.pitch)//2,key_pc,mode)
                    h=max(0.5,n.duration/2)
                    orn[-1]=RawNote(n.pitch,h,n.velocity,n.offset,n.voice)
                    orn.append(RawNote(pp,h,max(20,n.velocity-10),n.offset+h,1))
        result=orn
    elif op=="add_passing_notes":
        thr=int(np.clip(param if param>0 else 4,3,6)); exp=[]
        for i,n in enumerate(result):
            exp.append(n)
            if i+1<len(result) and n.duration>=1.0:
                nx=result[i+1]
                if abs(nx.pitch-n.pitch)>=thr:
                    pp=_snap_to_scale((n.pitch+nx.pitch)//2,key_pc,mode)
                    h=max(0.5,n.duration/2)
                    exp[-1]=RawNote(n.pitch,h,n.velocity,n.offset,n.voice)
                    exp.append(RawNote(pp,h,max(20,n.velocity-8),n.offset+h,1))
        result=exp

    # Fit to target (siempre 4/4 equivalente)
    if result:
        tb=target_bars*4
        ct=max(n.offset+n.duration for n in result)
        if ct>0.01:
            factor=tb/ct
            if factor<0.5:
                rs=sorted(result,key=lambda n:n.offset)
                dm=float(np.mean([n.duration for n in rs])) or 1.0
                nk=max(4,int(tb/dm)); si=max(0,(len(rs)-nk)//2)
                frag=rs[si:si+nk]
                if frag:
                    mn=min(n.offset for n in frag)
                    frag=[RawNote(n.pitch,n.duration,n.velocity,n.offset-mn,n.voice) for n in frag]
                    ft=max(n.offset+n.duration for n in frag)
                    fine=tb/max(ft,0.01)
                    if not (0.8<fine<1.2):
                        mf=0.25/max(min(n.duration for n in frag),0.01)
                        fine=max(fine,mf)
                        frag=[RawNote(n.pitch,max(0.25,n.duration*fine),n.velocity,n.offset*fine,n.voice) for n in frag]
                    result=frag
            elif not (0.8<factor<1.2):
                mf=0.25/max(min(n.duration for n in result),0.01)
                factor=max(factor,mf)
                result=[RawNote(n.pitch,max(0.25,n.duration*factor),n.velocity,n.offset*factor,n.voice) for n in result]

    # Micro-variaciones solo en notas largas
    result=[RawNote(n.pitch,
                     max(0.25,n.duration*rng.uniform(0.95,1.05)) if n.duration>=0.25 else n.duration,
                     int(np.clip(n.velocity*rng.uniform(0.97,1.03),20,120)),
                     n.offset, n.voice)
            for n in result]

    return sorted(result,key=lambda n:n.offset)


# ══════════════════════════════════════════════════════════════════════════════
#  ARCO DRAMÁTICO
# ══════════════════════════════════════════════════════════════════════════════

def _arc_value(arc: str, pos: float) -> float:
    t=float(np.clip(pos,0,1))
    if arc=="flat":           return 0.5
    elif arc=="rise":         return t
    elif arc=="fall":         return 1.0-t
    elif arc=="arch":         return math.sin(math.pi*t)
    elif arc=="inverse_arch": return 1.0-math.sin(math.pi*t)
    elif arc=="wave":         return 0.5+0.5*math.sin(2*math.pi*t)
    return 0.5

def _modulate_theta_by_arc(theta: TransformVector, av: float) -> TransformVector:
    tv=deepcopy(theta)
    if tv.op_dynamics=="identity":
        tv.op_dynamics="velocity_scale"; tv.param_dynamics=0.7+av*0.6
    elif tv.op_dynamics=="velocity_scale":
        tv.param_dynamics=float(np.clip(tv.param_dynamics*(0.7+av*0.6),0.5,1.5))
    elif tv.op_dynamics=="tension_curve":
        if av>0.7: tv.param_dynamics=1.0
        elif av<0.3: tv.param_dynamics=2.0
    return tv


# ══════════════════════════════════════════════════════════════════════════════
#  ACOMPAÑAMIENTO Y EXPORTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def _select_progression(role: RoleProfile, key_pc: int,
                         target_bars: int, arc_val: float) -> List[Tuple[str,int]]:
    tmap={"I":0.1,"i":0.15,"IV":0.2,"iv":0.25,"V":0.5,"v":0.55,
           "VI":0.3,"vi":0.3,"II":0.4,"ii":0.45,"VII":0.6,"vii":0.65,"V7":0.55}
    def pt(p): return float(np.mean([tmap.get(n,0.35) for n,_ in p])) if p else 0.35
    progs = role.progressions if role.progressions else FALLBACK_PROGRESSIONS[:2]
    best  = min(progs, key=lambda p: abs(pt(p)-arc_val))
    bn    = target_bars*4
    bi    = sum(d for _,d in best)
    reps  = max(1,math.ceil(bn/max(bi,1)))
    full  = best*reps; trunc=[]; cur=0.0
    for n,d in full:
        if cur>=bn: break
        a=min(d,bn-cur); trunc.append((n,a)); cur+=a
    return trunc or [("I",4.0)]


# ══════════════════════════════════════════════════════════════════════════════
#  v5 — MEJORAS DE VARIEDAD MUSICAL
# ══════════════════════════════════════════════════════════════════════════════

# [A] / [D] Selección de ventana de la fuente por posición de sección
def _select_source_window(notes: List[RawNote],
                           position_ratio: float,
                           window_beats: float,
                           rng: random.Random) -> List[RawNote]:
    """
    [A+D] Devuelve una ventana del material fuente centrada en la posición
    relativa indicada. Esto garantiza que cada sección usa un fragmento
    distinto del material en vez de siempre el mismo.
    Si hay un solo MIDI de entrada la ventana se desliza por él;
    si hay varios el sistema ya los mezcló en source_notes_all con offsets
    distintos, y la ventana selecciona la región correspondiente.
    """
    if not notes: return notes
    total = max(n.offset + n.duration for n in notes)
    if total <= 0: return notes

    # Centro de la ventana según posición
    center = position_ratio * total
    half   = window_beats / 2.0
    lo     = max(0.0, center - half)
    hi     = lo + window_beats

    # Ajustar si sobrepasa el final
    if hi > total:
        hi = total
        lo = max(0.0, total - window_beats)

    # Añadir pequeño jitter (±10% del tamaño de ventana) para variedad extra
    jitter = rng.uniform(-window_beats * 0.1, window_beats * 0.1)
    lo = max(0.0, min(total - window_beats * 0.5, lo + jitter))
    hi = lo + window_beats

    window = [n for n in notes if lo <= n.offset < hi]
    if len(window) < 2:
        # Ventana demasiado pequeña — usar todas las notas
        return notes

    mn = min(n.offset for n in window)
    return [RawNote(n.pitch, n.duration, n.velocity, n.offset - mn, n.voice)
            for n in sorted(window, key=lambda n: n.offset)]


# [B] Humanización rítmica
def _humanize(notes: List[RawNote], rng: random.Random,
               onset_jitter: float = 0.03,
               velocity_jitter: float = 8) -> List[RawNote]:
    """
    [B] Añade micro-variaciones de timing y dinámica para evitar el efecto
    mecánico de MIDI cuantizado perfectamente.
    onset_jitter : desplazamiento máximo de onset en beats
    velocity_jitter: variación máxima de velocidad MIDI
    Solo se humaniza si la nota es lo suficientemente larga para absorber el jitter.
    """
    result = []
    for n in notes:
        # Solo humanizar notas con duración suficiente
        if n.duration < 0.5:
            result.append(n)
            continue
        # Jitter de onset: máximo 10% de la duración para no crear solapamientos
        max_j = min(onset_jitter, n.duration * 0.10)
        j = rng.gauss(0, max_j * 0.4)
        new_offset = max(0.0, n.offset + j)
        # Jitter de velocity
        vj = int(rng.gauss(0, velocity_jitter * 0.5))
        new_vel = int(np.clip(n.velocity + vj, 20, 120))
        # Sin jitter de duración (evita notas demasiado cortas)
        result.append(RawNote(n.pitch, n.duration, new_vel, new_offset, n.voice))
    return result


# [C] Variación de textura intra-sección
def _build_accompaniment_split(prog: List[Tuple[str,int]], key_pc: int,
                                velocity: int, style_a: str, style_b: str,
                                rng: random.Random) -> List[RawNote]:
    """
    [C] Divide la progresión por la mitad y aplica estilos de acompañamiento
    distintos a cada mitad. Crea contraste textural dentro de la sección.
    """
    if not prog: return []
    total_beats = sum(d for _, d in prog)
    half = total_beats / 2.0
    cursor = 0.0
    prog_a, prog_b = [], []
    for num, dur in prog:
        if cursor < half:
            take = min(dur, half - cursor)
            if take > 0: prog_a.append((num, take))
            rest = dur - take
            if rest > 0: prog_b.append((num, rest))
        else:
            prog_b.append((num, dur))
        cursor += dur

    notes_a = _build_accompaniment(prog_a, key_pc, velocity, style_a)
    # Calcular offset base de la segunda mitad
    offset_b = sum(d for _, d in prog_a)
    notes_b_raw = _build_accompaniment(prog_b, key_pc, velocity, style_b)
    notes_b = [RawNote(n.pitch, n.duration, n.velocity,
                        n.offset + offset_b, n.voice) for n in notes_b_raw]
    return notes_a + notes_b


# [E] Contrapunto independiente con movimiento contrario
def _build_counterpoint(melody: List[RawNote], key_pc: int, mode: str,
                         velocity: int, rng: random.Random) -> List[RawNote]:
    """
    [E] Genera una segunda voz de contrapunto con movimiento mayoritariamente
    contrario al de la melodía, a una décima o sexta de distancia.
    Solo se aplica a notas melódicas largas (≥ 1 beat).
    """
    if not melody: return []
    scale_pcs = _get_scale_pcs(key_pc, mode)
    result = []

    # Analizar dirección global de la melodía
    pitches = [n.pitch for n in melody if n.duration >= 0.5]
    if len(pitches) < 2: return []
    global_dir = 1 if pitches[-1] > pitches[0] else -1

    for i, n in enumerate(melody):
        if n.duration < 1.0 or rng.random() > 0.55:
            continue  # saltar notas cortas y ~45% al azar

        # Movimiento contrario: si la melodía sube, el contrapunto baja
        local_dir = -global_dir
        # Intervalos de contrapunto: 6ª menor (8) o 10ª mayor (16)
        interval = rng.choice([8, 9, 10, 15, 16])
        cp_pitch = n.pitch + local_dir * interval
        # Snap a la escala
        cp_pitch = _snap_to_scale(cp_pitch, key_pc, mode)
        # Mantener en rango razonable
        cp_pitch = max(36, min(84, cp_pitch))
        # Velocidad más suave que la melodía
        cp_vel = max(20, velocity - rng.randint(10, 25))
        result.append(RawNote(cp_pitch, max(0.5, n.duration * 0.9),
                               cp_vel, n.offset, 1))

    return result


# [F] Reharmonización intra-sección (cambio armónico cada 2 compases)
def _reharmonize_progression(prog: List[Tuple[str,int]], key_pc: int,
                               tension: float, rng: random.Random) -> List[Tuple[str,int]]:
    """
    [F] Sustituye acordes individuales por substituciones armónicas plausibles.
    A mayor tensión, más substituciones cromáticas y acordes secundarios.
    """
    substitutions_low = {
        "I":  [("I",1.0), ("IM7",0.2)],
        "IV": [("IV",1.0), ("ii",0.3)],
        "V":  [("V",1.0),  ("V7",0.4)],
        "vi": [("vi",1.0), ("IV",0.2)],
    }
    substitutions_high = {
        "I":  [("I",1.0), ("bVII",0.3), ("IM7",0.25)],
        "IV": [("IV",1.0), ("ii7",0.4),  ("ii",0.3)],
        "V":  [("V7",0.8), ("V",0.5),    ("VII",0.2)],
        "vi": [("vi7",0.5),("vi",0.5),   ("IV",0.2)],
    }
    subs = substitutions_high if tension > 0.5 else substitutions_low
    result = []
    for num, dur in prog:
        candidates = subs.get(num, [(num, 1.0)])
        # Samplear según pesos
        choices, weights = zip(*candidates)
        total_w = sum(weights)
        r = rng.random() * total_w
        chosen = choices[0]
        acc = 0.0
        for ch, w in zip(choices, weights):
            acc += w
            if r <= acc:
                chosen = ch; break
        result.append((chosen, dur))
    return result


# [G] Silencios estructurales entre frases
def _add_phrase_rests(notes: List[RawNote], phrase_len_beats: float = 4.0,
                       rest_dur: float = 0.25,
                       rng: random.Random = None) -> List[RawNote]:
    """
    [G] Acorta ligeramente las notas que caen en el límite de frase para crear
    una respiración perceptible entre frases melódicas. Solo afecta a notas
    melódicas largas (voice=0, dur≥0.5 beats) en tiempos de downbeat de frase.
    """
    if not notes or rng is None: return notes
    result = []
    for n in notes:
        # ¿Esta nota está en un límite de frase?
        is_phrase_boundary = (n.offset % phrase_len_beats) < 0.1 and n.offset > 0.1
        # Solo acortar notas de melodía largas en límites de frase
        if (is_phrase_boundary and n.voice == 0 and n.duration >= 1.0
                and rng.random() < 0.6):
            shortened = max(0.25, n.duration - rest_dur - rng.uniform(0, 0.25))
            result.append(RawNote(n.pitch, shortened, n.velocity, n.offset, n.voice))
        else:
            result.append(n)
    return result


def _build_accompaniment(prog: List[Tuple[str,int]], key_pc: int,
                          velocity: int, style: str="arpeggio") -> List[RawNote]:
    notes=[]; cursor=0.0; prev=None
    for num,dur in prog:
        rpc,qual=_numeral_to_root(num,key_pc); ps=_chord_pitches(rpc,qual,4,prev); prev=ps
        if style=="block":
            for p in ps: notes.append(RawNote(p,max(0.25,dur*0.9),velocity,cursor,1))
        elif style in ("arpeggio","alberti"):
            step=max(0.5,dur/max(len(ps),1))
            if step*len(ps)>dur*1.5:
                for p in ps: notes.append(RawNote(p,max(0.25,dur*0.9),velocity,cursor,1))
            else:
                for i,p in enumerate(ps): notes.append(RawNote(p,max(0.25,step*0.85),velocity,cursor+i*step,1))
        elif style=="bass_only":
            if ps: notes.append(RawNote(min(ps),max(0.25,dur*0.9),velocity,cursor,2))
        cursor+=dur
    return notes

def _build_bass_line(prog: List[Tuple[str,int]], key_pc: int, velocity: int) -> List[RawNote]:
    notes=[]; cursor=0.0; prev=None
    for num,dur in prog:
        rpc,_=_numeral_to_root(num,key_pc); bass=rpc+36
        while bass<28: bass+=12
        while bass>52: bass-=12
        if prev is not None and abs(bass-prev)>7:
            alt=bass+(12 if bass<prev else -12)
            if 28<=alt<=52: bass=alt
        notes.append(RawNote(bass,max(0.25,dur*0.85),velocity,cursor,2)); prev=bass; cursor+=dur
    return notes

def _deduplicate(notes: List[RawNote]) -> List[RawNote]:
    sn=sorted(notes,key=lambda n:(n.offset,n.pitch)); active: Dict[int,int]={}; result=[]
    for n in sn:
        if n.duration < 0.25: continue   # descartar micronotas antes de procesar
        if n.pitch in active:
            pi=active[n.pitch]; pv=result[pi]
            if pv.offset+pv.duration>n.offset:
                nd=n.offset-pv.offset-0.01
                if nd < 0.25:
                    # La nota previa queda demasiado corta — eliminarla en vez de acortarla
                    result[pi]=RawNote(pv.pitch,0.0,pv.velocity,pv.offset,pv.voice)
                else:
                    result[pi]=RawNote(pv.pitch,nd,pv.velocity,pv.offset,pv.voice)
        result.append(n); active[n.pitch]=len(result)-1
    # Filtro final: eliminar notas con duración 0 o < 0.25
    return [n for n in result if n.duration >= 0.25]

def _notes_to_midi_track(notes: List[RawNote], tpb: int,
                          tempo_bpm: int, name: str) -> mido.MidiTrack:
    track=mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo",tempo=mido.bpm2tempo(tempo_bpm),time=0))
    track.append(mido.MetaMessage("track_name",name=name,time=0))
    track.append(mido.Message("program_change",channel=0,program=0,time=0))
    track.append(mido.Message("program_change",channel=1,program=32,time=0))
    evs=[]
    for n in notes:
        if n.duration < 0.125: continue  # descartar micronotas
        ton=int(n.offset*tpb); toff=int((n.offset+max(0.125,n.duration))*tpb)
        vel=max(1,min(127,n.velocity)); p=max(0,min(127,n.pitch)); ch=1 if p<48 else 0
        evs+=[(ton,"on",p,vel,ch),(toff,"off",p,0,ch)]
    evs.sort(key=lambda e:(e[0],0 if e[1]=="off" else 1))
    cur=0
    for at,et,pitch,vel,ch in evs:
        delta=max(0,at-cur)
        track.append(mido.Message("note_on" if et=="on" else "note_off",
                                   channel=ch,note=pitch,velocity=vel,time=delta))
        cur=at
    track.append(mido.MetaMessage("end_of_track",time=0))
    return track


# ══════════════════════════════════════════════════════════════════════════════
#  ASIGNACIÓN DE ROLES
# ══════════════════════════════════════════════════════════════════════════════

def _assign_roles_similarity(src: SectionSignature, model: "TransformationModel", n: int) -> List[int]:
    ds=sorted(range(len(model.roles)),key=lambda i: _signature_distance(src,model.roles[i].centroid))
    return [ds[i%len(ds)] for i in range(n)]

def _assign_roles_position(model: "TransformationModel", n: int) -> List[int]:
    return [min(range(len(model.roles)),key=lambda i:abs(model.roles[i].position_dist[0]-j/max(n-1,1)))
            for j in range(n)]

def _assign_roles_hybrid(src: SectionSignature, model: "TransformationModel", n: int) -> List[int]:
    ds=[_signature_distance(src,r.centroid) for r in model.roles]
    return [sorted(range(len(model.roles)),key=lambda i:ds[i]+abs(model.roles[i].position_dist[0]-j/max(n-1,1)))[0]
            for j in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def generate(input_midis: List[str],
             model_path: str,
             assignment_mode: Optional[str]=None,
             n_sections: Optional[int]=None,
             role_order_override: Optional[List[int]]=None,
             arc: str="arch",
             bars_per_section: Optional[int]=None,   # v4: None = aprender del corpus
             out_dir: str="output",
             output_name: str="song",
             tempo: Optional[int]=None,
             sa_compat: bool=False,       # [K]
             variety: float=0.7,           # v5: [0,1] intensidad de variedad musical
             dry_run: bool=False,
             seed: int=42,
             verbose: bool=False) -> None:

    rng=random.Random(seed); np.random.seed(seed)

    with open(model_path,"rb") as f:
        model: TransformationModel = pickle.load(f)
    _compat_model(model)
    mode=assignment_mode or model.assignment_mode

    print(f"\n  Modelo: {model_path}  ({model.n_roles} roles, corpus={model.corpus_size}, v={model.version})")
    knn_info = "KNN+GMM" if any(r.gmm for r in model.roles) else ("KNN" if model.knn_regressor else "heurístico")
    ae_info  = "AE activo" if model.ae_weights else "sin AE"
    print(f"  Asignación: {mode}  |  Arco: {arc}  |  variedad={variety:.1f}  |  {knn_info}  |  {ae_info}")

    # Cargar frases
    source_notes_all: List[RawNote] = []
    bpm_detected=120; bpb_detected=4

    for midi_path in input_midis:
        try:
            mid,tpb_in,bpm,bpb=_load_midi_raw(midi_path)
            track_idx=_choose_melody_track_idx(mid)
            notes=_parse_track_to_notes(mid.tracks[track_idx],tpb_in)
            notes=[n for n in notes if n.duration>=MELODY_MIN_DUR]
            if notes:
                source_notes_all.extend(notes); bpm_detected=bpm; bpb_detected=bpb
                if verbose: print(f"  Frase: {Path(midi_path).name}  {len(notes)} notas  {bpm} BPM")
        except Exception as e: print(f"  [warn] {midi_path}: {e}")

    if not source_notes_all:
        print("[ERROR] No se pudieron cargar frases."); sys.exit(1)

    final_tempo=tempo or bpm_detected
    key_pc,mode_str=_detect_key(source_notes_all)
    source_sig=_compute_signature(source_notes_all,0.0,key_pc,480,bpb_detected,model.ae_weights)

    print(f"  Tonalidad: {NOTE_NAMES[key_pc]} {mode_str}  Tempo: {final_tempo} BPM")

    # [G] Extraer motivo principal
    motif = _extract_motif(source_notes_all, key_pc, mode_str, length=5)
    if verbose and motif:
        print(f"  Motivo extraído: {motif.scale_degrees} (grados) {motif.rhythm_bins} (ritmo)")

    # v4: inferir n_sections y bars_per_section desde las distribuciones aprendidas
    # cuando el usuario no los especifica explícitamente.
    user_gave_nsecs = n_sections is not None
    user_gave_bars  = bars_per_section is not None

    if not user_gave_nsecs or not user_gave_bars:
        # Calcular medias ponderadas de las distribuciones de todos los roles
        # (ponderadas por la probabilidad de que cada rol sea asignado)
        all_bars_means  = [getattr(r, "bars_dist",  (8.0, 2.0))[0] for r in model.roles]
        all_nsecs_means = [getattr(r, "n_sections_dist", (4.0, 1.0))[0] for r in model.roles]
        all_bars_stds   = [getattr(r, "bars_dist",  (8.0, 2.0))[1] for r in model.roles]
        learned_bars  = float(np.mean(all_bars_means))
        learned_nsecs = float(np.mean(all_nsecs_means))
        learned_bars_std = float(np.mean(all_bars_stds))

        if not user_gave_bars:
            # Redondear a potencia de 2 más cercana (4, 8, 16, 32)
            raw = max(4.0, learned_bars)
            bars_per_section = int(2 ** round(math.log2(raw)))
            bars_per_section = max(4, min(64, bars_per_section))
            if verbose:
                print(f"  [v4] bars_per_section inferido del corpus: "
                      f"{learned_bars:.1f}±{learned_bars_std:.1f} → {bars_per_section}c")

        if not user_gave_nsecs:
            n_sections = max(2, int(round(learned_nsecs)))
            if verbose:
                print(f"  [v4] n_sections inferido del corpus: "
                      f"{learned_nsecs:.1f} → {n_sections}")

    if n_sections is None: n_sections = model.n_roles
    if bars_per_section is None: bars_per_section = 8

    print(f"  Secciones: {n_sections}  |  {bars_per_section} compases/sección"
          + ("  (inferidos del corpus)" if not user_gave_nsecs or not user_gave_bars else ""))

    # Asignar roles
    if role_order_override is not None:
        role_assignment=[r%model.n_roles for r in role_order_override]
        while len(role_assignment)<n_sections: role_assignment.append(role_assignment[-1])
        role_assignment=role_assignment[:n_sections]
    elif mode=="similarity":  role_assignment=_assign_roles_similarity(source_sig,model,n_sections)
    elif mode=="position":    role_assignment=_assign_roles_position(model,n_sections)
    else:                     role_assignment=_assign_roles_hybrid(source_sig,model,n_sections)

    if verbose: print(f"  Roles: {[model.roles[i].label for i in role_assignment]}")

    if dry_run:
        print(f"\n  [dry-run] Secciones:")
        for i,ri in enumerate(role_assignment):
            r=model.roles[ri]; av=_arc_value(arc,i/max(n_sections-1,1))
            sa_name=SA_ROLE_MAP.get(r.label,r.label) if sa_compat else r.label
            bd=getattr(r,"bars_dist",(8.0,2.0))
            dry_bars=bars_per_section if user_gave_bars else max(4,min(64,int(2**round(math.log2(max(4.0,bd[0]))))))
            print(f"    Sec {i+1}: {r.label:<14} {'→'+sa_name if sa_compat else ''}  arc={av:.2f}  bars={dry_bars}c  (corpus:{bd[0]:.1f}±{bd[1]:.1f})")
        return

    tpb=480; out_path=Path(out_dir); out_path.mkdir(parents=True,exist_ok=True)
    generated_sections: List[Tuple[str,List[RawNote],List[RawNote]]] = []
    prev_melody: Optional[List[RawNote]] = None
    bridge_notes_list: List[List[RawNote]] = []

    print(f"\n  Generando secciones...")
    print(f"  {'─'*62}")

    for i, role_idx in enumerate(role_assignment):
        role=model.roles[role_idx]
        pos=i/max(n_sections-1,1); av=_arc_value(arc,pos)

        # v4: duración de esta sección específica según el rol aprendido
        if not user_gave_bars:
            bd = getattr(role, "bars_dist", (bars_per_section, 2.0))
            # Samplear con algo de varianza controlada (±20% de la std)
            raw_bars = bd[0] + rng.gauss(0, bd[1] * 0.2)
            sec_bars = int(2 ** round(math.log2(max(4.0, raw_bars))))
            sec_bars = max(4, min(64, sec_bars))
        else:
            sec_bars = bars_per_section

        # [K] Nombre de sección compatible con song_architect
        if sa_compat:
            sec_label=SA_ROLE_MAP.get(role.label,role.label)
            if i==0 and sec_label in ("verse","development"): sec_label="intro"
            if i==n_sections-1 and sec_label not in ("outro",): sec_label="outro"
        else:
            sec_label=role.label
        sec_name=f"sec{i+1:02d}_{sec_label}"

        # Predecir θ: GMM si disponible, si no KNN, si no heurístico
        diff_vec=_signature_diff_vector(source_sig,role.centroid)
        fallback=_estimate_transform_heuristic(source_sig,role.centroid)
        if role.gmm is not None:
            theta_base=_sample_transform_gmm(role.gmm,fallback,av)
        else:
            theta_base=_predict_transform_knn(model.knn_regressor,diff_vec,fallback)
        theta=_modulate_theta_by_arc(theta_base,av)

        # [A+D] Seleccionar ventana de la fuente según posición de sección
        window_beats = sec_bars * 4
        if variety > 0 and len(source_notes_all) > 4:
            source_window = _select_source_window(
                source_notes_all, pos, window_beats, rng)
        else:
            source_window = source_notes_all

        # Transformar material fuente (sobre la ventana seleccionada)
        melody_notes=_apply_transform(source_window,theta,key_pc,mode_str,sec_bars,rng)

        # [G] Sembrar motivo
        if motif is not None:
            melody_notes=_apply_motif_to_notes(melody_notes,motif,key_pc,mode_str,theta)

        # [G] Silencios estructurales entre frases
        if variety > 0.3:
            melody_notes = _add_phrase_rests(melody_notes, phrase_len_beats=4.0,
                                              rest_dur=0.2, rng=rng)

        # [H] Voice leading + puente con sección anterior
        if prev_melody:
            bridge=_build_bridge(prev_melody,melody_notes,key_pc,mode_str,n_bridge=2)
            bridge_notes_list.append(bridge)
            melody_notes=_apply_voice_leading(prev_melody,melody_notes,key_pc,mode_str)
        else:
            bridge_notes_list.append([])

        # [I] Variaciones internas (detectar si la sección es más corta que el target)
        total_melody = max((n.offset+n.duration for n in melody_notes), default=0)
        target_beats = sec_bars * 4
        if total_melody > 0 and total_melody < target_beats * 0.6:
            reps_needed = int(math.ceil(target_beats / max(total_melody, 0.1)))
            full_melody = list(melody_notes)
            for rep in range(1, min(reps_needed, 4)):
                varied = _apply_internal_variations(melody_notes, rep, key_pc, mode_str, rng)
                offset_shift = rep * total_melody
                shifted = [RawNote(n.pitch, n.duration, n.velocity,
                                    n.offset+offset_shift, n.voice) for n in varied]
                full_melody.extend(shifted)
            melody_notes = full_melody

        # [B] Humanización rítmica (intensidad escalada por variety)
        if variety > 0:
            onset_j  = 0.02 * variety    # hasta ~36ms a 120BPM con variety=1
            vel_j    = int(6 * variety)  # hasta ±6 MIDI velocity
            melody_notes = _humanize(melody_notes, rng, onset_j, vel_j)

        # Acompañamiento con variedad
        # [F] Reharmonización: sustituir acordes según tensión del arco
        progression = _select_progression(role, key_pc, sec_bars, av)
        if variety > 0.4:
            progression = _reharmonize_progression(progression, key_pc, av, rng)

        acc_smap = {"intro":"bass_only","verse":"arpeggio","development":"block",
                    "climax":"block","outro":"bass_only","prechorus":"block",
                    "chorus":"block","bridge":"arpeggio"}
        acc_style = acc_smap.get(sec_label,"arpeggio")
        acc_vel   = max(20,int(np.mean([n.velocity for n in melody_notes]))-20)

        # [C] Variación de textura: segunda mitad con estilo diferente
        style_pairs = {
            "arpeggio": "block",  "block": "arpeggio",
            "bass_only": "arpeggio", "alberti": "block",
        }
        if variety > 0.5 and sec_bars >= 8:
            style_b = style_pairs.get(acc_style, acc_style)
            acc_notes = _build_accompaniment_split(
                progression, key_pc, acc_vel, acc_style, style_b, rng)
        else:
            acc_notes = _build_accompaniment(progression, key_pc, acc_vel, acc_style)

        bass_notes = _build_bass_line(progression, key_pc, max(20, acc_vel-10))

        # [E] Contrapunto independiente (solo en desarrollo y clímax, variety alto)
        cp_notes: List[RawNote] = []
        if variety > 0.6 and sec_label in ("development","climax","chorus","prechorus"):
            cp_notes = _build_counterpoint(melody_notes, key_pc, mode_str,
                                            max(20, acc_vel-5), rng)

        # [B] Humanizar también el acompañamiento (más suave que la melodía)
        if variety > 0:
            acc_notes  = _humanize(acc_notes,  rng, onset_j*0.5, vel_j//2)
            bass_notes = _humanize(bass_notes, rng, onset_j*0.3, vel_j//3)

        all_notes=_deduplicate(melody_notes+acc_notes+bass_notes+cp_notes)
        all_notes.sort(key=lambda n:n.offset)

        generated_sections.append((sec_name,melody_notes,all_notes,sec_bars))
        prev_melody=melody_notes

        sec_mid=mido.MidiFile(ticks_per_beat=tpb)
        sec_mid.tracks.append(_notes_to_midi_track(all_notes,tpb,final_tempo,sec_name))
        sec_path=out_path/f"{output_name}_{sec_name}.mid"
        sec_mid.save(str(sec_path))

        dur_s=sec_bars*4*60.0/final_tempo; ms,ss=divmod(int(dur_s),60)
        motif_mark="[M]" if motif else ""
        print(f"  ✓ {sec_path.name:<50} {sec_bars}c  {len(all_notes):>3}n"
              f"  arc={av:.2f}  {motif_mark}  ~{ms}:{ss:02d}")

    # Concatenar MIDI completo con puentes
    all_events=[]; marker_events=[]; cursor_ticks=0
    for k,(sec_name,_,sec_notes,s_bars) in enumerate(generated_sections):
        # [H] Insertar puente antes de la sección (excepto la primera)
        bridge = bridge_notes_list[k] if k < len(bridge_notes_list) else []
        for n in bridge:
            if n.duration < 0.25: continue  # descartar micronotas de puente
            ton=cursor_ticks+int(n.offset*tpb)
            toff=cursor_ticks+int((n.offset+max(0.25,n.duration))*tpb)
            vel=max(1,min(127,n.velocity)); p=max(0,min(127,n.pitch))
            all_events+=[(ton,"on",p,vel,0),(toff,"off",p,0,0)]

        marker_events.append((cursor_ticks,sec_name))
        for n in sec_notes:
            if n.duration < 0.125: continue  # descartar micronotas
            ton=cursor_ticks+int(n.offset*tpb)
            toff=cursor_ticks+int((n.offset+max(0.125,n.duration))*tpb)
            vel=max(1,min(127,n.velocity)); p=max(0,min(127,n.pitch)); ch=1 if p<48 else 0
            all_events+=[(ton,"on",p,vel,ch),(toff,"off",p,0,ch)]
        cursor_ticks+=s_bars*4*tpb

    all_events.sort(key=lambda e:(e[0],0 if e[1]=="off" else 1))

    # Filtrar notas con duración MIDI efectiva < 0.25 beats
    # (pueden surgir de jitter de onset en límites de sección)
    MIN_MIDI_DUR_TICKS = int(0.25 * tpb)
    active_ev: Dict[Tuple[int,int],int] = {}   # (pitch,ch) -> tick de note_on
    valid_events = []
    for ev in all_events:
        at, etype, pitch, vel, ch = ev
        key = (pitch, ch)
        if etype == "on":
            active_ev[key] = at
            valid_events.append(ev)
        else:  # off
            on_tick = active_ev.pop(key, None)
            if on_tick is not None and (at - on_tick) >= MIN_MIDI_DUR_TICKS:
                valid_events.append(ev)
            elif on_tick is not None:
                # Nota demasiado corta — eliminar también el note_on correspondiente
                valid_events = [e for e in valid_events
                                if not (e[1]=="on" and e[2]==pitch and e[4]==ch and e[0]==on_tick)]
    all_events = valid_events

    combined=[]
    for at,nm in marker_events:
        combined.append((at,0,mido.MetaMessage("marker",text=nm,time=0)))
    for at,et,pitch,vel,ch in all_events:
        combined.append((at,1 if et=="off" else 2,
                          mido.Message("note_on" if et=="on" else "note_off",
                                        channel=ch,note=pitch,velocity=vel,time=0)))
    combined.sort(key=lambda x:(x[0],x[1]))

    full_mid=mido.MidiFile(ticks_per_beat=tpb)
    full_track=mido.MidiTrack(); full_mid.tracks.append(full_track)
    full_track.append(mido.MetaMessage("set_tempo",tempo=mido.bpm2tempo(final_tempo),time=0))
    full_track.append(mido.MetaMessage("track_name",name=f"{output_name}_full",time=0))
    full_track.append(mido.Message("program_change",channel=0,program=0,time=0))
    full_track.append(mido.Message("program_change",channel=1,program=32,time=0))
    cur_tick=0
    for at,_,msg in combined:
        msg.time=max(0,at-cur_tick); full_track.append(msg); cur_tick=at
    full_track.append(mido.MetaMessage("end_of_track",time=0))

    full_path=out_path/f"{output_name}_full.mid"
    full_mid.save(str(full_path))

    ng=len(generated_sections)
    total_bars=sum(s[3] for s in generated_sections)
    ts=total_bars*4*60.0/final_tempo; mt,st=divmod(int(ts),60)
    print(f"\n{'═'*64}")
    bars_str="+".join(str(s[3]) for s in generated_sections)
    print(f"  Obra generada: {ng} secciones, {total_bars} compases ({bars_str}), ~{mt}:{st:02d}")
    print(f"  Arco: {arc}  |  Roles: {' → '.join(model.roles[i].label for i in role_assignment)}")
    if motif: print(f"  Motivo sembrado en {ng} secciones (grados: {motif.scale_degrees})")
    if sa_compat: print(f"  Nombres SA: {[SA_ROLE_MAP.get(model.roles[i].label,model.roles[i].label) for i in role_assignment]}")
    print(f"  Salida: {out_path}/")
    print(f"{'═'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  [J] COMANDO INSPECT
# ══════════════════════════════════════════════════════════════════════════════

def inspect_model(model_path: str, show_roles: bool=True,
                   show_progressions: bool=False,
                   show_eval: bool=True) -> None:
    with open(model_path,"rb") as f:
        model: TransformationModel = pickle.load(f)
    _compat_model(model)

    print(f"\n{'═'*70}")
    print(f"  ML ARCHITECT v{VERSION} — Inspección de modelo")
    print(f"{'═'*70}")
    print(f"  Fichero:     {model_path}")
    print(f"  Versión:     {model.version}")
    print(f"  Corpus:      {model.corpus_size} MIDIs")
    print(f"  Roles:       {model.n_roles}")
    print(f"  Asignación:  {model.assignment_mode}")
    print(f"  KNN:         {'sí' if model.knn_regressor else 'no'}")
    print(f"  Autoencoder: {'sí' if model.ae_weights else 'no'}")
    es = getattr(model,"eval_score",0.0); eb = getattr(model,"eval_baseline",0.0)
    if show_eval and eb > 0:
        imp = ((eb-es)/eb)*100
        print(f"  Evaluación:  score={es:.4f}  baseline={eb:.4f}  mejora={imp:.1f}%")

    if show_roles:
        print(f"\n  {'─'*68}")
        print(f"  ROLES:")
        for i,r in enumerate(model.roles):
            print(f"\n  [{i}] {r.label.upper()}")
            print(f"    Posición:    {r.position_dist[0]:.2f} ± {r.position_dist[1]:.2f}")
            print(f"    Tensión:     {r.tension_dist[0]:.3f} ± {r.tension_dist[1]:.3f}")
            print(f"    Helm. tens:  {r.harmonic_tens_dist[0]:.3f} ± {r.harmonic_tens_dist[1]:.3f}")
            tv = r.theta_mean
            print(f"    θ_pitch:     {tv.op_pitch} ({tv.param_pitch:.2f})")
            print(f"    θ_rhythm:    {tv.op_rhythm} ({tv.param_rhythm:.2f})")
            print(f"    θ_dynamics:  {tv.op_dynamics} ({tv.param_dynamics:.2f})")
            print(f"    θ_harmony:   {tv.op_harmony} ({tv.param_harmony:.2f})")
            gmm_str = ('sí ('+str(r.gmm.n_components)+' componentes)') if r.gmm else 'no'
            print(f"    GMM:         {gmm_str}")
            bd=getattr(r,"bars_dist",(8.0,2.0)); nd=getattr(r,"n_sections_dist",(4.0,1.0))
            print(f"    Bars/secc:   {bd[0]:.1f} ± {bd[1]:.1f} compases (aprendido del corpus)")
            print(f"    N secciones: {nd[0]:.1f} ± {nd[1]:.1f} por pieza")
            print(f"    Progresiones:{len(r.progressions)}")

        # Matriz de similitud entre roles
        print(f"\n  {'─'*68}")
        print(f"  MATRIZ DE SIMILITUD ENTRE ROLES (coseno):")
        n = len(model.roles)
        header = "         " + "".join(f"  [{i}]{model.roles[i].label[:5]:<6}" for i in range(n))
        print(f"  {header}")
        for i in range(n):
            row = f"  [{i}]{model.roles[i].label[:5]:<6}  "
            for j in range(n):
                if i==j: row+="  1.000  "
                else:
                    d=_signature_distance(model.roles[i].centroid,model.roles[j].centroid)
                    row+=f"  {1-d:.3f}  "
            print(row)

        # Arcos típicos del corpus
        print(f"\n  {'─'*68}")
        print(f"  ARCOS TÍPICOS POR ROL (distribución de posición):")
        for i,r in enumerate(model.roles):
            p,s=r.position_dist
            bar_w=int(p*30); bar_s=max(1,int(s*30))
            bar="─"*bar_w + "█" + "─"*(30-bar_w)
            spread=" "*(max(0,bar_w-bar_s))+"░"*min(bar_s*2,30-bar_w)
            print(f"  [{i}] {r.label:<14} pos={p:.2f}±{s:.2f}  [{bar}]")

    if show_progressions:
        print(f"\n  {'─'*68}")
        print(f"  PROGRESIONES POR ROL:")
        for i,r in enumerate(model.roles):
            print(f"\n  [{i}] {r.label}:")
            for k,prog in enumerate(r.progressions[:3]):
                prog_str=" → ".join(f"{n}({int(d)}b)" for n,d in prog)
                print(f"    {k+1}. {prog_str}")

    print(f"\n{'═'*70}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_segment(args) -> None:
    print(f"\n{'═'*64}\n  ML ARCHITECT v{VERSION} — Segmentación\n{'═'*64}")
    print(f"  MIDI: {args.midi}")
    sections=extract_sections(midi_path=args.midi,num_sections=args.num_sections,
                               track_idx=args.track,all_tracks=args.all_tracks,
                               similarity_threshold=args.similarity,verbose=args.verbose)
    if not sections: print("  [ERROR] No se detectaron secciones."); return
    print(f"\n  Forma: {''.join(s.label for s in sections)}")
    print(f"  {'─'*60}")
    for s in sections:
        print(f"  {s.label}  c{s.bar_start:3d}–{s.bar_end:3d}  {len(s.notes):>4}n"
              f"  [{s.symbol[:30]}{'…' if len(s.symbol)>30 else ''}]")
    if args.verbose:
        print(f"\n  Firmas v3 (con grados de escala y tensión armónica):")
        for i,s in enumerate(sections):
            sig=_compute_signature(s.notes,i/max(len(sections)-1,1))
            top_deg=int(np.argmax(sig.scale_degree_hist))
            print(f"  {s.label}  cromatismo={sig.tension:.2f}  helmholtz={sig.harmonic_tension:.2f}"
                  f"  grado_dom={top_deg+1}  contorno={'↑' if sig.contour_dir>0.1 else '↓' if sig.contour_dir<-0.1 else '→'}")
    if args.output:
        data={"source":args.midi,"form":"".join(s.label for s in sections),
              "sections":[{"label":s.label,"bar_start":s.bar_start,"bar_end":s.bar_end,
                           "n_notes":len(s.notes),"symbol":s.symbol} for s in sections]}
        Path(args.output).write_text(json.dumps(data,indent=2,ensure_ascii=False))
        print(f"\n  Informe: {args.output}")
    print()

def cmd_train(args) -> None:
    print(f"\n{'═'*64}\n  ML ARCHITECT v{VERSION} — Entrenamiento\n{'═'*64}")
    train(midi_dir=args.corpus,output_model=args.output,n_roles=args.n_roles,
          assignment_mode=args.assignment,num_sections=args.num_sections,verbose=args.verbose)

def cmd_update(args) -> None:
    """[L] Entrenamiento incremental."""
    print(f"\n{'═'*64}\n  ML ARCHITECT v{VERSION} — Actualización incremental\n{'═'*64}")
    train(midi_dir=args.new_corpus,output_model=args.output,
          n_roles=args.n_roles,assignment_mode=args.assignment,
          num_sections=args.num_sections,existing_model_path=args.model,verbose=args.verbose)

def cmd_generate(args) -> None:
    print(f"\n{'═'*64}\n  ML ARCHITECT v{VERSION} — Generación\n{'═'*64}")
    generate(input_midis=args.input,model_path=args.model,
             assignment_mode=args.assignment,n_sections=args.n_sections,
             role_order_override=args.role_order,arc=args.arc,
             bars_per_section=args.bars_per_section,out_dir=args.out_dir,
             output_name=args.output_name,tempo=args.tempo,
             sa_compat=args.sa_compat,variety=args.variety,
             dry_run=args.dry_run,seed=args.seed,verbose=args.verbose)

def cmd_inspect(args) -> None:
    inspect_model(model_path=args.model,show_roles=True,
                   show_progressions=args.progressions,show_eval=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="ml_architect_v3",
        description=f"ML ARCHITECT v{VERSION}",formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version",action="version",version=f"%(prog)s {VERSION}")
    sub=p.add_subparsers(dest="command",required=True)

    # SEGMENT
    seg=sub.add_parser("segment",help="Detecta secciones formales de un MIDI")
    seg.add_argument("midi"); seg.add_argument("--num-sections","-n",type=int,default=None)
    seg.add_argument("--track",type=int,default=None); seg.add_argument("--all-tracks",action="store_true")
    seg.add_argument("--similarity",type=float,default=0.85)
    seg.add_argument("--output","-o",type=str,default=None); seg.add_argument("--verbose",action="store_true")
    seg.set_defaults(func=cmd_segment)

    # TRAIN
    trn=sub.add_parser("train",help="Entrena el modelo sobre un corpus")
    trn.add_argument("corpus"); trn.add_argument("--output","-o",default="model_v3.pkl")
    trn.add_argument("--n-roles",type=int,default=4)
    trn.add_argument("--assignment",choices=["similarity","position","hybrid"],default="hybrid")
    trn.add_argument("--num-sections","-n",type=int,default=None)
    trn.add_argument("--verbose",action="store_true")
    trn.set_defaults(func=cmd_train)

    # UPDATE [L]
    upd=sub.add_parser("update",help="[L] Actualización incremental de un modelo existente")
    upd.add_argument("new_corpus",help="Directorio con los MIDIs nuevos")
    upd.add_argument("--model","-m",required=True,help="Modelo base a actualizar")
    upd.add_argument("--output","-o",default="model_v3_updated.pkl")
    upd.add_argument("--n-roles",type=int,default=4)
    upd.add_argument("--assignment",choices=["similarity","position","hybrid"],default="hybrid")
    upd.add_argument("--num-sections","-n",type=int,default=None)
    upd.add_argument("--verbose",action="store_true")
    upd.set_defaults(func=cmd_update)

    # GENERATE
    gen=sub.add_parser("generate",help="Genera una obra completa")
    gen.add_argument("input",nargs="+"); gen.add_argument("--model","-m",required=True)
    gen.add_argument("--assignment",choices=["similarity","position","hybrid"],default=None)
    gen.add_argument("--n-sections",type=int,default=None)
    gen.add_argument("--role-order",type=int,nargs="+",default=None)
    gen.add_argument("--arc",choices=ARC_SHAPES,default="arch",
                      help=f"Arco dramático (default: arch). Opciones: {', '.join(ARC_SHAPES)}")
    gen.add_argument("--bars-per-section",type=int,default=None,
                      help="Compases por sección (default: inferido del corpus)")
    gen.add_argument("--out-dir",default="output"); gen.add_argument("--output-name",default="song")
    gen.add_argument("--tempo",type=int,default=None)
    gen.add_argument("--sa-compat",action="store_true",
                      help="[K] Nombres de sección compatibles con song_architect.py")
    gen.add_argument("--dry-run",action="store_true")
    gen.add_argument("--variety",type=float,default=0.7,metavar="V",
                      help="Intensidad de variedad musical [0-1] (default: 0.7). "
                           "0=mínima (comportamiento v4), 1=máxima variedad")
    gen.add_argument("--seed",type=int,default=42); gen.add_argument("--verbose",action="store_true")
    gen.set_defaults(func=cmd_generate)

    # INSPECT [J]
    ins=sub.add_parser("inspect",help="[J] Inspecciona un modelo entrenado")
    ins.add_argument("model",help="Fichero .pkl del modelo")
    ins.add_argument("--progressions",action="store_true",help="Mostrar progresiones por rol")
    ins.set_defaults(func=cmd_inspect)

    return p


def main() -> None:
    parser=build_parser(); args=parser.parse_args(); args.func(args)


if __name__=="__main__":
    main()
