#!/usr/bin/env python3
"""
MIDI Musical & Emotional Analyzer v3.0
Uso:   python3 midi_analyzer_v3.py archivo.mid
       python3 midi_analyzer_v3.py archivo.mid --sections 8 --output informe.txt
Deps:  pip install mido
"""
import sys, os, math, argparse, collections
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

try:
    import mido
except ImportError:
    print("ERROR: pip install mido"); sys.exit(1)



# ═══════════════════════════════════════════════════════════════
#  TABLAS MUSICALES
# ═══════════════════════════════════════════════════════════════

NOTE_NAMES   = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
NOTE_NAMES_ES= ['Do','Do#','Re','Re#','Mi','Fa','Fa#','Sol','Sol#','La','La#','Si']

# Krumhansl-Schmuckler profiles
KS_MAJOR = [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
KS_MINOR = [6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]

# Mode profiles (for specific mode detection beyond major/minor)
KS_MODES = {
    'major':      KS_MAJOR,
    'minor':      KS_MINOR,
    'dorian':     [6.35,2.23,3.48,5.38,2.60,4.09,2.52,5.19,3.98,2.39,3.34,2.88],
    'phrygian':   [6.35,5.38,2.23,3.48,2.33,4.09,2.52,5.19,2.39,3.66,2.29,2.88],
    'lydian':     [6.35,2.23,3.48,2.33,4.38,2.52,4.09,5.19,2.39,3.66,2.29,2.88],
    'mixolydian': [6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,4.09,2.88],
    'locrian':    [6.35,5.38,2.23,3.48,2.60,4.09,4.38,2.52,3.98,2.39,3.34,3.17],
}

# Mode emotional signatures
MODE_EMOTIONS = {
    'major':      ('Mayor',      'luminoso, estable, abierto — alegría, confianza'),
    'minor':      ('Menor nat.', 'oscuro, introspectivo — tristeza, drama, añoranza'),
    'dorian':     ('Dórico',     'oscuro pero esperanzador — soul, jazz, melancolía activa'),
    'phrygian':   ('Frigio',     'amenazante, tenso, flamenco — misterio, fatalismo'),
    'lydian':     ('Lidio',      'onírico, flotante, irreal — fantasía, lo sublime'),
    'mixolydian': ('Mixolidio',  'épico, folk, ambiguo — poder sin resolución clara'),
    'locrian':    ('Locrio',     'inestable, disonante, angular — ansiedad extrema'),
}

# Dissonance table (0=consonant, 1=max dissonant)
DISSONANCE = {0:0.00,1:1.00,2:0.80,3:0.25,4:0.20,5:0.10,
              6:0.95,7:0.00,8:0.20,9:0.10,10:0.45,11:0.85}

# Chord templates (semitones from root)
CHORD_TEMPLATES = {
    'maj':[0,4,7],'min':[0,3,7],'dim':[0,3,6],'aug':[0,4,8],
    'maj7':[0,4,7,11],'min7':[0,3,7,10],'dom7':[0,4,7,10],
    'dim7':[0,3,6,9],'hdim7':[0,3,6,10],'sus2':[0,2,7],
    'sus4':[0,5,7],'add9':[0,4,7,14],'maj9':[0,4,7,11,14],
}

# Chord quality → tension contribution
CHORD_TENSION = {
    'maj':0.0,'min':0.1,'sus2':0.2,'sus4':0.3,'aug':0.5,
    'maj7':0.2,'min7':0.25,'dom7':0.6,'dim':0.7,'dim7':0.75,'hdim7':0.65,
}

# Scale intervals (semitones from root)
SCALE_INTERVALS = {
    'major':      [0,2,4,5,7,9,11],
    'minor':      [0,2,3,5,7,8,10],
    'dorian':     [0,2,3,5,7,9,10],
    'phrygian':   [0,1,3,5,7,8,10],
    'lydian':     [0,2,4,6,7,9,11],
    'mixolydian': [0,2,4,5,7,9,10],
    'locrian':    [0,1,3,5,6,8,10],
}

# Canonical chord progressions (degree intervals, chord types)
CANONICAL_PROGRESSIONS = {
    'I-V-vi-IV (Pop)':         [(0,'maj'),(7,'maj'),(9,'min'),(5,'maj')],
    'I-IV-V (Blues/Rock)':     [(0,'maj'),(5,'maj'),(7,'maj')],
    'ii-V-I (Jazz)':           [(2,'min'),(7,'dom7'),(0,'maj')],
    'I-vi-IV-V (Doo-Wop)':    [(0,'maj'),(9,'min'),(5,'maj'),(7,'maj')],
    'Andaluza (i-VII-VI-V)':   [(0,'min'),(10,'maj'),(8,'maj'),(7,'maj')],
    'i-VI-III-VII (Modal)':    [(0,'min'),(8,'maj'),(3,'maj'),(10,'maj')],
    'I-V-vi-iii-IV (Canon)':   [(0,'maj'),(7,'maj'),(9,'min'),(4,'min'),(5,'maj')],
    'i-iv-V (Menor clásico)':  [(0,'min'),(5,'min'),(7,'maj')],
    'I-bVII-IV (Mixolydian)':  [(0,'maj'),(10,'maj'),(5,'maj')],
    'i-bVI-bVII (Aeolian)':    [(0,'min'),(8,'maj'),(10,'maj')],
    'I-III-IV-iv (Cromático)': [(0,'maj'),(4,'maj'),(5,'maj'),(5,'min')],
    'I-bII-I (Napolitano)':    [(0,'maj'),(1,'maj'),(0,'maj')],
    'i-bII (Frigio)':          [(0,'min'),(1,'maj')],
    'I-IV-bVII-IV (Rock)':     [(0,'maj'),(5,'maj'),(10,'maj'),(5,'maj')],
}

# Modal borrowing tables
BORROWED_MAJOR = {
    'bII Napolitano':  (1, 'maj'),
    'bIII mayor':      (3, 'maj'),
    'iv menor (4ª m)': (5, 'min'),
    'bVI mayor':       (8, 'maj'),
    'bVII mayor':      (10,'maj'),
    'vii° disminuido': (11,'dim'),
}
BORROWED_MINOR = {
    'IV mayor (Dórico)':  (5, 'maj'),
    'I mayor (Picardía)': (0, 'maj'),
    'bII Napolitano':     (1, 'maj'),
    '#VI Lidio':          (9, 'maj'),
}

# Harmonic function classification (degree → function)
HARMONIC_FUNCTION = {
    # major
    ('major', 0):  ('T',  'Tónica',       'reposo total'),
    ('major', 2):  ('SP', 'Subdominante', 'movimiento suave'),
    ('major', 4):  ('DP', 'Dominante',    'tensión hacia tónica'),
    ('major', 5):  ('S',  'Subdominante', 'movimiento, apertura'),
    ('major', 7):  ('D',  'Dominante',    'máxima tensión → resolución'),
    ('major', 9):  ('Tp', 'T. paralela',  'color menor, ambigüedad'),
    ('major', 11): ('D7', 'Sensible',     'atracción extrema hacia tónica'),
    # minor
    ('minor', 0):  ('t',  'Tónica',       'reposo oscuro'),
    ('minor', 2):  ('sp', 'Subdominante', 'movimiento sombrío'),
    ('minor', 3):  ('tP', 'T. paralela',  'relativa mayor, luz momentánea'),
    ('minor', 5):  ('s',  'Subdominante', 'movimiento menor'),
    ('minor', 7):  ('D',  'Dominante',    'tensión → resolución'),
    ('minor', 8):  ('sP', 'S. paralela',  'color mayor prestado'),
    ('minor', 10): ('tP', 'Subtónica',    'resolución modal suave'),
}

# Cadence types
CADENCE_TYPES = {
    'perfecta':   ('V→I o V7→I',  'cierre definitivo, certeza, conclusión'),
    'imperfecta': ('I→V o ii→V',  'apertura, pregunta, continuación'),
    'rota':       ('V→vi',        'sorpresa emocional, evasión, giro dramático'),
    'plagal':     ('IV→I',        'conclusión espiritual, "amén", reposo suave'),
    'frigia':     ('iv6→V',       'cadencia modal, tensión oscura sin resolver'),
    'evitada':    ('V→cualquier', 'frustración de expectativa, ambigüedad'),
}

# General MIDI instrument emotional map
GM_EMOTIONS = {
    range(0,8):    ('Piano',           'intimidad, reflexión, lirismo expresivo'),
    range(8,16):   ('Celesta/Teclado', 'delicadeza, fantasía, luminosidad'),
    range(16,24):  ('Órgano',          'solemnidad, espiritualidad, poder'),
    range(24,32):  ('Guitarra',        'calidez, nostalgia folk, cercanía'),
    range(32,40):  ('Bajo',            'profundidad, groove, fundamento rítmico'),
    range(40,48):  ('Cuerda solista',  'drama, emoción intensa, lirismo'),
    range(48,56):  ('Cuerdas ens.',    'grandiosidad, plenitud orquestal'),
    range(56,64):  ('Brass',           'triunfo, poder, urgencia heroica'),
    range(64,72):  ('Reed/Saxo',       'nostalgia, soul, intimidad jazz'),
    range(72,80):  ('Flauta/Madera',   'ligereza, naturaleza, inocencia'),
    range(80,88):  ('Synth lead',      'modernidad, tensión electrónica'),
    range(88,96):  ('Synth pad',       'atmósfera, suspensión, ensoñación'),
    range(96,104): ('Synth FX',        'misterio, lo etéreo, lo desconocido'),
    range(104,112):('Étnico/World',    'identidad cultural, lo ancestral'),
    range(112,120):('Percusión mel.',  'ritmo, danza, celebración'),
    range(120,128):('SFX',             'ambiente, textura no musical'),
}

# Valence-Arousal base values by mode+tempo
VA_BASE = {
    ('major','slow'):  ( 0.4,-0.5), ('major','medium'):( 0.6, 0.1), ('major','fast'):( 0.7, 0.7),
    ('minor','slow'):  (-0.6,-0.5), ('minor','medium'):(-0.4, 0.1), ('minor','fast'):(-0.2, 0.6),
    ('dorian','slow'): (-0.2,-0.4), ('dorian','medium'):(-0.1, 0.2),('dorian','fast'):( 0.1, 0.6),
    ('phrygian','slow'):(-0.7,-0.3),('phrygian','medium'):(-0.6,0.3),('phrygian','fast'):(-0.4,0.7),
    ('lydian','slow'): ( 0.5,-0.3), ('lydian','medium'):( 0.6, 0.2),('lydian','fast'):( 0.8, 0.5),
    ('mixolydian','slow'):(0.2,-0.3),('mixolydian','medium'):(0.4,0.3),('mixolydian','fast'):(0.5,0.7),
}

# ═══════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class NoteEvent:
    time_sec: float
    pitch: int
    velocity: int
    duration: float
    channel: int
    track: int
    tick: int = 0

@dataclass
class ChordEvent:
    time: float
    root: int
    chord_type: str
    score: float
    name: str
    function: str = ''
    function_label: str = ''
    tension: float = 0.0

@dataclass
class TensionPoint:
    time: float
    tension: float
    event: str = ''   # what caused the tension

@dataclass
class Section:
    index: int
    start: float
    end: float
    notes: List[NoteEvent] = field(default_factory=list)

    @property
    def duration(self): return max(self.end - self.start, 0.001)
    @property
    def density(self): return len(self.notes) / self.duration
    @property
    def avg_velocity(self):
        return sum(n.velocity for n in self.notes) / len(self.notes) if self.notes else 0
    @property
    def avg_pitch(self):
        return sum(n.pitch for n in self.notes) / len(self.notes) if self.notes else 60
    @property
    def pitch_range(self):
        if not self.notes: return 0
        return max(n.pitch for n in self.notes) - min(n.pitch for n in self.notes)
    @property
    def chroma(self):
        vec = [0.0]*12
        for n in self.notes: vec[n.pitch%12] += (n.velocity/127.0) * n.duration
        t = sum(vec); return [v/t for v in vec] if t > 0 else vec

# ═══════════════════════════════════════════════════════════════
#  MIDI PARSING
# ═══════════════════════════════════════════════════════════════

def parse_midi(filepath: str):
    mid = mido.MidiFile(filepath)
    tpb = mid.ticks_per_beat
    meta = {
        'tpb': tpb, 'format': mid.type, 'num_tracks': len(mid.tracks),
        'time_signatures': [], 'tempo_changes': [], 'key_signature': None,
        'track_names': [], 'program_changes': {}, 'pitch_bend_events': [],
    }
    all_events = []
    for ti, track in enumerate(mid.tracks):
        at = 0; nm = None
        for msg in track:
            at += msg.time; all_events.append((at, ti, msg))
            if msg.type == 'track_name': nm = msg.name
        meta['track_names'].append(nm or f'Track {ti}')
    all_events.sort(key=lambda x: (x[0], x[1]))

    # Build tempo map
    tempo_map = [(0, 500000)]
    for at, _, msg in all_events:
        if msg.type == 'set_tempo': tempo_map.append((at, msg.tempo))
    tempo_map.sort()

    def t2s(ticks):
        sec = 0.0; pt, pT = 0, 500000
        for tk, tT in tempo_map:
            if tk >= ticks: break
            sec += (min(tk, ticks) - pt) * pT / 1e6 / tpb; pt, pT = tk, tT
        return sec + (ticks - pt) * pT / 1e6 / tpb

    for tk, tT in tempo_map:
        meta['tempo_changes'].append((t2s(tk), round(6e7/tT, 2)))
    for at, _, msg in all_events:
        if msg.type == 'time_signature':
            meta['time_signatures'].append({
                'time': t2s(at), 'numerator': msg.numerator, 'denominator': msg.denominator
            })
        elif msg.type == 'key_signature': meta['key_signature'] = msg.key
        elif msg.type == 'pitchwheel': meta['pitch_bend_events'].append((t2s(at), msg.pitch))
        elif msg.type == 'program_change': meta['program_changes'][msg.channel] = msg.program

    notes = []; open_n = {}
    for at, ti, msg in all_events:
        if msg.type in ('note_on', 'note_off'):
            key = (ti, msg.channel, msg.note); ts = t2s(at)
            if msg.type == 'note_on' and msg.velocity > 0:
                open_n[key] = (ts, msg.velocity, at)
            elif key in open_n:
                s, v, stk = open_n.pop(key)
                notes.append(NoteEvent(s, msg.note, v, max(ts-s, 0.01), msg.channel, ti, stk))
    last_s = t2s(max(e[0] for e in all_events) if all_events else 0)
    for (tr, ch, p), (s, v, stk) in open_n.items():
        notes.append(NoteEvent(s, p, v, max(last_s-s, 0.01), ch, tr, stk))
    notes.sort(key=lambda n: n.time_sec)
    return notes, meta, t2s

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def classify_tempo(b):
    for lim, name in [(50,'Grave'),(66,'Largo'),(76,'Larghetto'),(108,'Andante/Moderato'),
                      (120,'Allegretto'),(156,'Allegro'),(176,'Vivace')]:
        if b < lim: return name
    return 'Presto'

def tempo_cat(b): return 'slow' if b < 80 else ('fast' if b > 120 else 'medium')

def get_instrument(prog):
    for r, (name, emo) in GM_EMOTIONS.items():
        if prog in r: return name, emo
    return 'Desconocido', 'expresividad genérica'

def pearson(a, b):
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(len(a)))
    den = math.sqrt(sum((a[i]-ma)**2 for i in range(len(a))) *
                    sum((b[i]-mb)**2 for i in range(len(b))))
    return num/den if den > 0 else 0.0

GOLDEN_RATIO = 0.6180339887

def ts_str(sec):
    return f"{int(sec//60)}:{int(sec%60):02d}"

def wrap(text, width=68, indent='  '):
    words = text.split(); lines = []; buf = []
    for w in words:
        if sum(len(x)+1 for x in buf) + len(w) > width:
            lines.append(indent + ' '.join(buf)); buf = [w]
        else: buf.append(w)
    if buf: lines.append(indent + ' '.join(buf))
    return '\n'.join(lines)




# ═══════════════════════════════════════════════════════════════
#  A. TONALIDAD Y MODO ESPECÍFICO
# ═══════════════════════════════════════════════════════════════

def detect_key_and_mode(notes: List[NoteEvent]) -> Tuple[int, str, float]:
    """
    Detecta tonalidad usando Krumhansl-Schmuckler.
    Devuelve (root, mode_name, confidence).
    mode_name puede ser: major/minor/dorian/phrygian/lydian/mixolydian/locrian
    """
    pc = [0.0]*12
    for n in notes: pc[n.pitch%12] += (n.velocity/127.0) * n.duration
    total = sum(pc)
    if total == 0: return 0, 'major', 0.0
    pc = [v/total for v in pc]

    best_root, best_mode, best_r = 0, 'major', -999
    for mode_name, profile in KS_MODES.items():
        for rot in range(12):
            rotated = profile[rot:] + profile[:rot]
            r = pearson(pc, rotated)
            if r > best_r:
                best_r, best_root, best_mode = r, rot, mode_name
    return best_root, best_mode, best_r

def detect_local_key(notes: List[NoteEvent]) -> Tuple[int, str]:
    """Fast key detection for a small window (no confidence needed)."""
    if not notes: return 0, 'major'
    root, mode, _ = detect_key_and_mode(notes)
    return root, mode

# ═══════════════════════════════════════════════════════════════
#  B. DETECCIÓN DE ACORDES CON FUNCIÓN ARMÓNICA
# ═══════════════════════════════════════════════════════════════

def detect_chords(notes: List[NoteEvent], window: float = 0.75) -> List[ChordEvent]:
    if not notes: return []
    total = max(n.time_sec + n.duration for n in notes)
    events = []; prev = None; t = 0.0
    while t < total:
        active = [n for n in notes if n.time_sec < t+window and n.time_sec+n.duration > t]
        if len(active) >= 2:
            pcs = collections.Counter(n.pitch%12 for n in active)
            bname, br, bt, bs = '?', 0, 'maj', -1
            for root in range(12):
                for ct, ivs in CHORD_TEMPLATES.items():
                    cp = set((root+i)%12 for i in ivs)
                    sc = sum(pcs[p] for p in cp) / sum(pcs.values())
                    if sc > bs: bs, bname, br, bt = sc, f"{NOTE_NAMES[root]}{ct}", root, ct
            if bs > 0.45 and bname != prev:
                ct = CHORD_TENSION.get(bt, 0.3)
                events.append(ChordEvent(t, br, bt, bs, bname, tension=ct))
                prev = bname
        t += window
    return events

def assign_harmonic_functions(chords: List[ChordEvent], key_root: int, mode: str) -> List[ChordEvent]:
    """Asigna función armónica (T/S/D/etc.) a cada acorde."""
    mode_key = 'minor' if mode in ('minor','phrygian','locrian') else 'major'
    for c in chords:
        deg = (c.root - key_root) % 12
        func_data = HARMONIC_FUNCTION.get((mode_key, deg))
        if func_data:
            c.function, c.function_label, _ = func_data
        else:
            c.function = '?'
            c.function_label = 'función cromática'
    return chords

# ═══════════════════════════════════════════════════════════════
#  C. ANÁLISIS DE CADENCIAS
# ═══════════════════════════════════════════════════════════════

def analyze_cadences(chords: List[ChordEvent], key_root: int, mode: str) -> Dict:
    """
    Detecta y clasifica todas las cadencias de la pieza.
    Devuelve counts y descripción.
    """
    if len(chords) < 2:
        return {'counts': {}, 'total': 0, 'dominant_type': 'indeterminado', 'desc': ''}

    mode_key = 'minor' if mode in ('minor','phrygian','locrian') else 'major'
    dom = (key_root + 7) % 12
    subdiv = (key_root + 5) % 12
    subdom_minor = (key_root + 5) % 12  # for plagal
    relative = (key_root + 9) % 12 if mode_key == 'major' else (key_root + 3) % 12

    counts = collections.Counter()
    cadence_list = []

    for i in range(len(chords)-1):
        a, b = chords[i], chords[i+1]
        ra, rb = a.root, b.root

        # Perfecta: V→I or V7→I
        if ra == dom and rb == key_root:
            counts['perfecta'] += 1
            cadence_list.append(('perfecta', a.time))
        # Plagal: IV→I
        elif ra == subdiv and rb == key_root and a.chord_type in ('maj','min'):
            counts['plagal'] += 1
            cadence_list.append(('plagal', a.time))
        # Rota/evitada: V→vi (or bVI in minor)
        elif ra == dom and rb == relative:
            counts['rota'] += 1
            cadence_list.append(('rota', a.time))
        # Imperfecta: I→V or ii→V
        elif rb == dom and ra == key_root:
            counts['imperfecta'] += 1
            cadence_list.append(('imperfecta', a.time))
        # Frigia: iv6→V (characteristic of phrygian/minor)
        elif ra == (key_root+5)%12 and a.chord_type == 'min' and rb == dom:
            counts['frigia'] += 1
            cadence_list.append(('frigia', a.time))
        # Napolitana: bII→V or bII→I
        elif ra == (key_root+1)%12 and rb in (dom, key_root):
            counts['napolitana'] += 1
            cadence_list.append(('napolitana', a.time))

    total = sum(counts.values())

    # Determine dominant cadence type
    if total == 0:
        dominant_type = 'sin cadencias claras'
    else:
        dominant_type = counts.most_common(1)[0][0] if counts else 'mixta'

    # Build description
    desc_parts = []
    if counts.get('perfecta', 0) > 0:
        desc_parts.append(f"perfectas ×{counts['perfecta']} — cierres definitivos")
    if counts.get('rota', 0) > 0:
        desc_parts.append(f"rotas ×{counts['rota']} — sorpresas emocionales, giros dramáticos")
    if counts.get('plagal', 0) > 0:
        desc_parts.append(f"plagales ×{counts['plagal']} — conclusión espiritual")
    if counts.get('imperfecta', 0) > 0:
        desc_parts.append(f"imperfectas ×{counts['imperfecta']} — apertura, preguntas")
    if counts.get('frigia', 0) > 0:
        desc_parts.append(f"frigias ×{counts['frigia']} — tensión modal oscura")
    if counts.get('napolitana', 0) > 0:
        desc_parts.append(f"napolitanas ×{counts['napolitana']} — color cromático, drama")

    return {
        'counts': counts, 'total': total,
        'dominant_type': dominant_type,
        'list': cadence_list,
        'desc': '; '.join(desc_parts) if desc_parts else 'sin cadencias claras detectadas'
    }

# ═══════════════════════════════════════════════════════════════
#  D. CURVA DE TENSIÓN CONTINUA
# ═══════════════════════════════════════════════════════════════

def compute_tension_curve(
    notes: List[NoteEvent],
    chords: List[ChordEvent],
    key_root: int,
    mode: str,
    resolution: float = 0.5
) -> List[TensionPoint]:
    """
    Calcula tensión nota a nota en ventana deslizante.
    Combina:
      1. Disonancia armónica entre notas simultáneas
      2. Tensión del acorde activo
      3. Presencia de nota sensible (leading tone)
      4. Distancia de registro (graves dan peso, agudos dan tensión)
      5. Densidad local (más notas = más tensión)
      6. Dinámica (velocidad)
      7. Distancia de la tónica (centricidad)
    """
    if not notes: return []
    total = max(n.time_sec + n.duration for n in notes)
    curve = []
    t = 0.0
    # Sensible dependiente del modo:
    # - Mayor, lidio, mixolidio, menor (armónica): VII elevado (+11)
    # - Dórico: también usa VII elevado en contexto dominante
    # - Frigio, locrio: no tienen sensible funcional (dominante es bVII);
    #   usar None para no inflar artificialmente la tensión en piezas frigias
    if mode in ('phrygian', 'locrian'):
        leading_tone = None
    else:
        leading_tone = (key_root + 11) % 12
    dom = (key_root + 7) % 12

    # Build chord lookup
    chord_at = {}
    for c in chords:
        chord_at[c.time] = c
    chord_times = sorted(chord_at.keys())

    def get_chord_at(t):
        if not chord_times: return None
        idx = 0
        for i, ct in enumerate(chord_times):
            if ct <= t: idx = i
            else: break
        return chord_at[chord_times[idx]]

    while t < total:
        window = [n for n in notes if n.time_sec < t + resolution and n.time_sec + n.duration > t]
        if not window:
            curve.append(TensionPoint(t, 0.0, 'silencio'))
            t += resolution; continue

        # 1. Harmonic dissonance
        pitches = [n.pitch % 12 for n in window]
        dis = 0.0; pairs = 0
        for i in range(len(pitches)):
            for j in range(i+1, min(i+5, len(pitches))):
                dis += DISSONANCE[abs(pitches[i]-pitches[j]) % 12]
                pairs += 1
        harmonic_tension = dis / pairs if pairs > 0 else 0.0

        # 2. Chord tension
        chord = get_chord_at(t)
        chord_tension = chord.tension if chord else 0.3

        # 3. Leading tone presence (strong pull)
        lt_present = (leading_tone is not None and
                      any(n.pitch % 12 == leading_tone for n in window))
        lt_tension = 0.6 if lt_present else 0.0

        # 4. Register tension (very high notes = bright tension)
        avg_pitch = sum(n.pitch for n in window) / len(window)
        register_tension = max(0, (avg_pitch - 72) / 36) if avg_pitch > 72 else 0.0

        # 5. Density tension
        density = len(window) / resolution
        density_tension = min(density / 20.0, 1.0)

        # 6. Dynamic tension
        avg_vel = sum(n.velocity for n in window) / len(window)
        dynamic_tension = avg_vel / 127.0

        # 7. Distance from tonic (chromatic notes = tension)
        scale = set(SCALE_INTERVALS.get(mode, SCALE_INTERVALS['minor']))
        chromatic = sum(1 for p in pitches if (p - key_root) % 12 not in scale)
        chromatic_tension = chromatic / len(pitches) if pitches else 0

        # Dominant presence (anticipatory tension)
        dom_present = any(n.pitch % 12 == dom for n in window)
        dom_tension = 0.4 if dom_present and not any(n.pitch % 12 == key_root for n in window) else 0.0

        # Weighted combination
        tension = (
            harmonic_tension * 0.25 +
            chord_tension    * 0.15 +
            lt_tension       * 0.15 +
            register_tension * 0.10 +
            density_tension  * 0.10 +
            dynamic_tension  * 0.10 +
            chromatic_tension* 0.10 +
            dom_tension      * 0.05
        )
        tension = max(0.0, min(1.0, tension))

        # Annotate significant events
        event = ''
        if lt_tension > 0 and tension > 0.5: event = 'nota sensible'
        elif chord_tension > 0.5: event = 'acorde tenso'
        elif chromatic_tension > 0.5: event = 'cromatismo'

        curve.append(TensionPoint(t, tension, event))
        t += resolution

    return curve

def tension_curve_ascii(curve: List[TensionPoint], width: int = 60, height: int = 10) -> str:
    """Renders tension curve as ASCII art."""
    if not curve: return ''
    values = [p.tension for p in curve]
    total_time = curve[-1].time
    # Downsample to width
    step = max(1, len(values) // width)
    sampled = [max(values[i:i+step]) for i in range(0, len(values), step)][:width]
    if not sampled: return ''

    lines = []
    for row in range(height, 0, -1):
        threshold = row / height
        line = ''
        for v in sampled:
            if v >= threshold: line += '█'
            elif v >= threshold - 0.5/height: line += '▄'
            else: line += ' '
        label = f"{threshold:.1f}" if row in (height, height//2, 1) else '    '
        lines.append(f"  {label} │{line}│")

    lines.append(f"       └{'─'*len(sampled)}┘")
    lines.append(f"       0{' '*(len(sampled)//2-1)}{ts_str_local(total_time/2)}{' '*(len(sampled)//2-3)}{ts_str_local(total_time)}")
    return '\n'.join(lines)

def ts_str_local(sec):
    return f"{int(sec//60)}:{int(sec%60):02d}"

def find_tension_peaks(curve: List[TensionPoint], min_prominence: float = 0.15) -> List[TensionPoint]:
    """Finds local maxima in tension curve."""
    if len(curve) < 3: return []
    peaks = []
    for i in range(1, len(curve)-1):
        if curve[i].tension > curve[i-1].tension and curve[i].tension > curve[i+1].tension:
            # Check prominence
            local_min = min(curve[max(0,i-4):i+5], key=lambda p: p.tension).tension
            if curve[i].tension - local_min >= min_prominence:
                peaks.append(curve[i])
    # Deduplicate (keep highest within 2 seconds)
    deduped = []
    last_t = -999
    for p in sorted(peaks, key=lambda x: -x.tension):
        if p.time - last_t > 2.0:
            deduped.append(p)
            last_t = p.time
    return sorted(deduped, key=lambda x: x.time)

def find_tension_valleys(curve: List[TensionPoint]) -> List[TensionPoint]:
    """Finds release moments (local minima after a peak)."""
    if len(curve) < 3: return []
    valleys = []
    for i in range(1, len(curve)-1):
        if curve[i].tension < curve[i-1].tension and curve[i].tension < curve[i+1].tension:
            if curve[i].tension < 0.2:
                valleys.append(curve[i])
    return valleys

def describe_tension_arc(curve: List[TensionPoint]) -> str:
    """Describes the overall shape of the tension arc."""
    if not curve: return 'indeterminado'
    values = [p.tension for p in curve]
    n = len(values)
    thirds = n // 3
    t1 = sum(values[:thirds]) / thirds if thirds > 0 else 0
    t2 = sum(values[thirds:2*thirds]) / thirds if thirds > 0 else 0
    t3 = sum(values[2*thirds:]) / (n - 2*thirds) if n > 2*thirds else 0

    if t2 > t1 + 0.1 and t2 > t3 + 0.1:
        return 'climático central (la tensión crece, alcanza el pico y se libera)'
    elif t3 > t1 + 0.1 and t3 > t2 + 0.1:
        return 'climático final (la tensión se acumula hacia el desenlace)'
    elif t1 > t2 + 0.1 and t1 > t3 + 0.1:
        return 'apertura intensa (alta tensión inicial que se libera gradualmente)'
    elif abs(t3 - t1) < 0.05:
        return 'circular (la pieza termina con el mismo nivel de tensión que empieza)'
    elif t3 > t1:
        return 'ascendente (tensión que crece progresivamente sin liberarse del todo)'
    else:
        return 'descendente (liberación gradual de la tensión a lo largo de la obra)'

# ═══════════════════════════════════════════════════════════════
#  E. VOICE LEADING
# ═══════════════════════════════════════════════════════════════

def analyze_voice_leading(chords: List[ChordEvent], notes: List[NoteEvent]) -> Dict:
    """
    Analiza el movimiento de voces entre acordes consecutivos.
    Detecta:
      - Movimiento paralelo de 5as/8as (error de escritura o efecto)
      - Cromatismo (semitono) — máxima fluidez
      - Saltos (>4 semitonos) — dramatismo o ruptura
      - Resolución de la sensible
      - Bajo obstinado (pedal)
    """
    if len(chords) < 2:
        return {'smoothness': 0.5, 'chromatic_moves': 0, 'leaps': 0,
                'pedal_detected': False, 'desc': 'insuficientes datos'}

    transitions = []
    for i in range(len(chords)-1):
        a, b = chords[i], chords[i+1]
        # Root movement
        root_move = abs((b.root - a.root + 6) % 12 - 6)  # circular distance
        transitions.append(root_move)

    if not transitions:
        return {'smoothness': 0.5, 'chromatic_moves': 0, 'leaps': 0,
                'pedal_detected': False, 'desc': 'sin datos'}

    chromatic = sum(1 for t in transitions if t <= 1)
    stepwise  = sum(1 for t in transitions if t == 2)
    thirds    = sum(1 for t in transitions if t in (3,4))
    fifths    = sum(1 for t in transitions if t in (5,7))
    leaps     = sum(1 for t in transitions if t >= 6)
    total     = len(transitions)

    smoothness = (chromatic * 1.0 + stepwise * 0.8 + thirds * 0.5) / total

    # Pedal point detection: bass note repeating while harmony changes
    bass_notes = sorted([n for n in notes if n.pitch < 55], key=lambda n: n.time_sec)
    pedal_detected = False
    if len(bass_notes) > 8:
        bass_pcs = [n.pitch % 12 for n in bass_notes[:20]]
        dominant_bass = collections.Counter(bass_pcs).most_common(1)[0]
        if dominant_bass[1] / 20 > 0.5:
            pedal_detected = True

    # Build description
    parts = []
    if smoothness > 0.7:
        parts.append("voice leading muy fluido — transiciones suaves, fluidez emocional")
    elif smoothness > 0.4:
        parts.append("voice leading equilibrado")
    else:
        parts.append("voice leading con saltos — dramatismo, rupturas expresivas")

    if chromatic / total > 0.3:
        parts.append(f"movimiento cromático frecuente ({chromatic}/{total}) — máxima tensión expresiva")
    if fifths / total > 0.3:
        parts.append(f"movimientos de 5ª frecuentes ({fifths}/{total}) — cadencias fuertes")
    if leaps / total > 0.2:
        parts.append(f"saltos amplios ({leaps}/{total}) — gestos dramáticos")
    if pedal_detected:
        parts.append("pedal de bajo detectado — tensión sostenida sobre fundamento fijo")

    return {
        'smoothness': smoothness,
        'chromatic_moves': chromatic,
        'stepwise': stepwise,
        'fifths': fifths,
        'leaps': leaps,
        'pedal_detected': pedal_detected,
        'desc': '; '.join(parts) if parts else 'voice leading neutro'
    }

# ═══════════════════════════════════════════════════════════════
#  F. ANÁLISIS DE EXPECTATIVA (simplificado IDyOM)
# ═══════════════════════════════════════════════════════════════

def analyze_melodic_expectation(notes: List[NoteEvent], key_root: int, mode: str) -> Dict:
    """
    Modelo simplificado de expectativa melódica basado en:
      1. Principio de buena continuación (intervalos pequeños son esperados)
      2. Principio de implicación-realización (Narmour)
      3. Escala diatónica (notas de la escala = esperadas)
      4. Repetición de patrones (lo repetido es esperado)

    Calcula "sorpresa" por nota: alta sorpresa = emoción intensa.
    """
    if len(notes) < 4: return {'mean_surprise': 0.0, 'surprise_peaks': [], 'desc': ''}

    sorted_notes = sorted(notes, key=lambda n: n.time_sec)
    scale = set(SCALE_INTERVALS.get(mode, SCALE_INTERVALS['minor']))
    intervals = [sorted_notes[i+1].pitch - sorted_notes[i].pitch
                 for i in range(len(sorted_notes)-1)]

    surprise_values = []
    for i, interval in enumerate(intervals[1:], 1):
        prev_interval = intervals[i-1]
        surprise = 0.0

        # 1. Size surprise: large intervals are more surprising
        abs_iv = abs(interval)
        if abs_iv <= 2:   size_s = 0.0
        elif abs_iv <= 4: size_s = 0.2
        elif abs_iv <= 7: size_s = 0.5
        else:             size_s = 0.8

        # 2. Direction surprise (Narmour: large intervals imply continuation)
        if abs(prev_interval) >= 5:
            # Large interval implies continuation in same direction
            expected_same_dir = (prev_interval > 0) == (interval > 0)
            direction_s = 0.0 if expected_same_dir else 0.6
        else:
            # Small interval implies reversal
            expected_reversal = (prev_interval > 0) != (interval > 0)
            direction_s = 0.0 if expected_reversal else 0.4

        # 3. Scale membership
        next_pc = (sorted_notes[i+1].pitch - key_root) % 12
        scale_s = 0.0 if next_pc in scale else 0.5

        # 4. Tritone surprise
        tritone_s = 0.7 if abs_iv == 6 else 0.0

        surprise = (size_s * 0.35 + direction_s * 0.30 +
                    scale_s * 0.20 + tritone_s * 0.15)
        surprise_values.append((sorted_notes[i+1].time_sec, surprise))

    if not surprise_values:
        return {'mean_surprise': 0.0, 'surprise_peaks': [], 'desc': ''}

    mean_s = sum(v for _, v in surprise_values) / len(surprise_values)
    peaks = [(t, s) for t, s in surprise_values if s > 0.5]
    peaks.sort(key=lambda x: -x[1])
    top_peaks = peaks[:5]

    if mean_s < 0.15:
        desc = "melodía muy predecible — confort, familiaridad, seguridad emocional"
    elif mean_s < 0.25:
        desc = "melodía con sorpresas moderadas — equilibrio entre expectativa y novedad"
    elif mean_s < 0.4:
        desc = "melodía frecuentemente sorprendente — tensión, interés, inquietud"
    else:
        desc = "melodía muy impredecible — ansiedad, desorientación, intensidad extrema"

    return {
        'mean_surprise': mean_s,
        'surprise_peaks': top_peaks,
        'all_values': surprise_values,
        'desc': desc
    }

# ═══════════════════════════════════════════════════════════════
#  G. ANÁLISIS DE REGISTROS POR FUNCIÓN
# ═══════════════════════════════════════════════════════════════

def analyze_register_functions(notes: List[NoteEvent]) -> Dict:
    """
    Separa las notas en capas funcionales (bajo, tenor, contralto, soprano)
    y analiza cada una independientemente.
    """
    if not notes: return {}

    # Define register ranges (MIDI pitch)
    layers = {
        'bajo':     (0,  47,  'fundamento armónico, profundidad, raíz'),
        'tenor':    (48, 59,  'relleno armónico, calor, cuerpo'),
        'contralto':(60, 71,  'zona media, equilibrio, transición'),
        'soprano':  (72, 127, 'melodía principal, brillo, expresión'),
    }

    result = {}
    for name, (lo, hi, role) in layers.items():
        layer_notes = [n for n in notes if lo <= n.pitch <= hi]
        if not layer_notes:
            result[name] = {'count': 0, 'role': role, 'desc': 'ausente'}
            continue

        avg_vel = sum(n.velocity for n in layer_notes) / len(layer_notes)
        avg_pitch = sum(n.pitch for n in layer_notes) / len(layer_notes)
        total_notes = len(notes)
        proportion = len(layer_notes) / total_notes

        # Detect if this layer has melodic or harmonic role
        layer_sorted = sorted(layer_notes, key=lambda n: n.time_sec)
        if len(layer_sorted) > 2:
            ivs = [abs(layer_sorted[i+1].pitch - layer_sorted[i].pitch)
                   for i in range(len(layer_sorted)-1)]
            avg_iv = sum(ivs) / len(ivs)
            melodic = avg_iv > 2.5
        else:
            melodic = False

        desc = f"{proportion*100:.0f}% de las notas, vel.μ={avg_vel:.0f}"
        if proportion > 0.4:
            desc += " — capa dominante"
        if melodic:
            desc += ", movimiento melódico activo"
        else:
            desc += ", función armónica/tonal"

        result[name] = {
            'count': len(layer_notes),
            'proportion': proportion,
            'avg_velocity': avg_vel,
            'avg_pitch': avg_pitch,
            'melodic': melodic,
            'role': role,
            'desc': desc
        }

    # Detect unusual register usage
    anomalies = []
    if result.get('bajo', {}).get('proportion', 0) > 0.4:
        anomalies.append("el bajo domina la escritura — peso, gravedad, lo subterráneo")
    if result.get('soprano', {}).get('proportion', 0) > 0.5:
        anomalies.append("registro agudo dominante — fragilidad, urgencia, o luminosidad extrema")
    if not result.get('tenor', {}).get('count', 0) and not result.get('contralto', {}).get('count', 0):
        anomalies.append("ausencia del registro medio — sonido hueco, polaridad dramática")

    result['anomalies'] = anomalies
    return result

# ═══════════════════════════════════════════════════════════════
#  H. PEDAL POINT DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_pedal_points(notes: List[NoteEvent], key_root: int) -> List[Dict]:
    """
    Detecta puntos de pedal: nota sostenida mientras la armonía cambia encima.
    Clasifica: tónica (estabilidad), dominante (tensión urgente), otro (color).
    """
    if not notes: return []

    # Work with bass register
    bass = sorted([n for n in notes if n.pitch < 60], key=lambda n: n.time_sec)
    if len(bass) < 4: return []

    pedals = []
    window = 4.0  # seconds

    total = max(n.time_sec for n in bass)
    t = 0.0
    while t < total - window:
        window_bass = [n for n in bass if t <= n.time_sec < t + window]
        if len(window_bass) >= 3:
            pcs = [n.pitch % 12 for n in window_bass]
            dominant_pc = collections.Counter(pcs).most_common(1)[0]
            if dominant_pc[1] / len(pcs) > 0.6:
                # This pitch dominates — check if harmony above changes
                upper = [n for n in notes if n.pitch >= 60 and t <= n.time_sec < t + window]
                upper_pcs = set(n.pitch % 12 for n in upper)
                upper_variety = len(upper_pcs)

                if upper_variety >= 4:  # harmony moving above a stable bass
                    pc = dominant_pc[0]
                    if pc == key_root:
                        pedal_type = 'tónica'
                        pedal_emotion = 'estabilidad bajo tensión — la tónica sostiene el caos armónico'
                    elif pc == (key_root + 7) % 12:
                        pedal_type = 'dominante'
                        pedal_emotion = 'máxima tensión anticipatoria — reclama resolución urgente'
                    else:
                        pedal_type = f"{NOTE_NAMES[pc]} (interior)"
                        pedal_emotion = 'color inusual — ambigüedad tonal, suspenso'

                    # Avoid duplicate detections
                    if not pedals or pedals[-1]['start'] < t - window/2:
                        pedals.append({
                            'start': t,
                            'pitch_class': pc,
                            'note': NOTE_NAMES[pc],
                            'type': pedal_type,
                            'emotion': pedal_emotion
                        })
        t += window / 2

    return pedals

# ═══════════════════════════════════════════════════════════════
#  I. RED DE RELACIONES ARMÓNICAS (GRAFO TONAL)
# ═══════════════════════════════════════════════════════════════

def build_harmonic_graph(chords: List[ChordEvent]) -> Dict:
    """
    Construye un grafo de transiciones armónicas.
    Nodos = acordes, aristas = transiciones, peso = frecuencia.
    Detecta: centro gravitacional, movimientos más frecuentes,
    aventuras armónicas (transiciones poco frecuentes).
    """
    if len(chords) < 3:
        return {'center': None, 'transitions': {}, 'desc': 'insuficientes datos'}

    # Build transition matrix
    transitions = collections.Counter()
    for i in range(len(chords)-1):
        a, b = chords[i].name, chords[i+1].name
        transitions[(a, b)] += 1

    # Node weights (chord frequency)
    chord_freq = collections.Counter(c.name for c in chords)
    total = sum(chord_freq.values())

    # Tonal center = most connected + most frequent node
    center = chord_freq.most_common(1)[0][0] if chord_freq else None

    # Most frequent transitions
    top_transitions = transitions.most_common(5)

    # Entropy of the transition distribution (harmonic predictability)
    total_trans = sum(transitions.values())
    if total_trans > 0:
        probs = [v/total_trans for v in transitions.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        max_entropy = math.log2(len(transitions)) if len(transitions) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = 0

    # Describe
    if normalized_entropy < 0.4:
        harmony_desc = "armonía muy predecible — pocos movimientos que se repiten (ostinato armónico)"
    elif normalized_entropy < 0.7:
        harmony_desc = "armonía moderadamente variada — equilibrio entre repetición y exploración"
    else:
        harmony_desc = "armonía muy exploratoria — gran variedad de movimientos, riqueza cromática"

    top_str = ', '.join(f"{a}→{b} (×{n})" for (a,b),n in top_transitions[:3])

    return {
        'center': center,
        'transitions': dict(transitions),
        'entropy': normalized_entropy,
        'top_transitions': top_transitions,
        'chord_freq': chord_freq,
        'desc': harmony_desc,
        'top_str': top_str,
    }

# ═══════════════════════════════════════════════════════════════
#  J. ANÁLISIS DE FORMA MUSICAL
# ═══════════════════════════════════════════════════════════════

def analyze_musical_form(sections: List[Section]) -> Dict:
    """
    Intenta identificar la forma musical:
    Sonata, Rondó, Binaria, Ternaria, Tema con variaciones, Through-composed.
    Usa similitud chroma entre secciones.
    """
    active = [s for s in sections if s.notes]
    if len(active) < 2:
        return {'form': 'indeterminada', 'pattern': '', 'desc': ''}

    def chroma_sim(a, b):
        dot = sum(x*y for x,y in zip(a, b))
        ma = math.sqrt(sum(x**2 for x in a))
        mb = math.sqrt(sum(x**2 for x in b))
        return dot/(ma*mb) if ma > 0 and mb > 0 else 0

    # Build similarity matrix
    n = len(active)
    sim_matrix = [[chroma_sim(active[i].chroma, active[j].chroma)
                   for j in range(n)] for i in range(n)]

    # Label sections as A, B, C based on similarity threshold
    threshold = 0.90
    labels = ['A']
    label_map = {0: 'A'}
    next_label = ord('B')

    for i in range(1, n):
        matched = False
        for j in range(i):
            if sim_matrix[i][j] >= threshold:
                labels.append(label_map[j])
                label_map[i] = label_map[j]
                matched = True
                break
        if not matched:
            lbl = chr(next_label)
            labels.append(lbl)
            label_map[i] = lbl
            next_label += 1

    pattern = ''.join(labels)
    unique_sections = len(set(labels))

    # Identify form
    if n >= 3 and labels[0] == labels[-1] and labels[0] != labels[n//2]:
        if n == 3:
            form = 'Ternaria (A-B-A)'
            desc = 'Exposición → contraste → recapitulación. Arco narrativo completo.'
        else:
            form = 'Rondó o forma ampliada (A-...-A)'
            desc = 'El tema principal regresa tras secciones contrastantes.'
    elif unique_sections == 1:
        form = 'Through-composed (material único)'
        desc = 'Todo el material es similar — cohesión máxima, desarrollo lineal.'
    elif unique_sections >= n * 0.8:
        form = 'Through-composed (material siempre nuevo)'
        desc = 'Cada sección introduce material nuevo — narrativa abierta, sin recapitulación.'
    elif n >= 4 and labels[0] == labels[2] and labels[1] == labels[3]:
        form = 'Binaria con repetición (A-B-A-B)'
        desc = 'Alternancia de dos ideas — equilibrio, diálogo entre elementos.'
    elif n == 2:
        form = 'Binaria (A-B)'
        desc = 'Dos secciones contrastantes — pregunta y respuesta.'
    elif all(labels[i] != labels[i+1] for i in range(n-1)):
        form = 'Variaciones o desarrollo continuo'
        desc = 'Material en constante transformación — narrativa de exploración.'
    else:
        form = f'Libre / Mixta ({pattern})'
        desc = 'Estructura orgánica sin forma canónica clara.'

    return {
        'form': form,
        'pattern': pattern,
        'unique_sections': unique_sections,
        'sim_matrix': sim_matrix,
        'labels': labels,
        'desc': desc,
    }

# ═══════════════════════════════════════════════════════════════
#  K. VALENCE-AROUSAL AVANZADO
# ═══════════════════════════════════════════════════════════════

def compute_va(sec: Section, mode: str, bpm: float, tension: float, sync: float,
               expectation: float = 0.2) -> Tuple[float, float]:
    tc = tempo_cat(bpm)
    bv, ba = VA_BASE.get((mode, tc), VA_BASE.get(('minor' if 'min' in mode else 'major', tc), (0, 0)))
    vn = (sec.avg_velocity - 64) / 63
    dn = min(sec.density / 15, 1) * 2 - 1
    v = max(-1, min(1, bv - (tension - 0.5) * 0.6 + vn * 0.15 - expectation * 0.2))
    a = max(-1, min(1, ba + vn * 0.3 + dn * 0.3 + sync * 0.3 + expectation * 0.15))
    return v, a

def describe_va(v: float, a: float) -> str:
    q = {(True,True):'alegría activa/euforia', (True,False):'serenidad/paz',
         (False,True):'angustia/urgencia',     (False,False):'tristeza/melancolía profunda'}
    base = q[(v >= 0, a >= 0)]
    intensity = math.sqrt(v**2 + a**2)
    if intensity < 0.25: return 'emoción neutra/ambigua (centro del plano)'
    mag = 'leve' if intensity < 0.5 else ('moderada' if intensity < 0.75 else 'intensa')
    return f"{base} ({mag}) — V={v:+.2f}, A={a:+.2f}"

def va_chart(vas: List[Tuple[float, float, str]]) -> str:
    W, H = 34, 13
    grid = [['·']*W for _ in range(H)]
    mx, my = W//2, H//2
    for x in range(W): grid[my][x] = '─'
    for y in range(H): grid[y][mx] = '│'
    grid[my][mx] = '┼'
    lbs = '①②③④⑤⑥⑦⑧⑨⑩'
    for idx, (v, a, _) in enumerate(vas[:10]):
        px = max(0, min(W-1, int((v+1)/2*(W-1))))
        py = max(0, min(H-1, int((1-(a+1)/2)*(H-1))))
        grid[py][px] = lbs[idx]
    header = f"         +Activado(A)"
    lines = [header, f"   ┌{'─'*W}┐"]
    lines.append(f"   │{''.join(grid[0])}│")
    for row in grid[1:-1]: lines.append(f"   │{''.join(row)}│")
    lines.append(f"   │{''.join(grid[-1])}│  -Activado")
    lines.append(f"   └{'─'*W}┘")
    lines.append(f"  -Valence {'─'*12}+Valence")
    return '\n'.join(lines)

# ═══════════════════════════════════════════════════════════════
#  L. RESTO DE ANÁLISIS (from v2, improved)
# ═══════════════════════════════════════════════════════════════

def detect_key_v2(notes):  # alias
    return detect_key_and_mode(notes)

def find_canonical_progressions(chords, key_root):
    if len(chords) < 2: return []
    degs = [((c.root-key_root)%12, c.chord_type) for c in chords]
    found = []
    for pname, pat in CANONICAL_PROGRESSIONS.items():
        pl = len(pat)
        for i in range(len(degs)-pl+1):
            if all(degs[i+j] == pat[j] for j in range(pl)):
                if pname not in found: found.append(pname)
    return found

def detect_modal_borrowing(chords, key_root, mode):
    if not chords: return []
    scale = set(SCALE_INTERVALS.get(mode, SCALE_INTERVALS['minor']))
    table = BORROWED_MAJOR if mode in ('major','lydian','mixolydian') else BORROWED_MINOR
    found = []
    for c in chords:
        rel = (c.root-key_root)%12
        if rel not in scale:
            for bn, (br, bt) in table.items():
                if rel == br and c.chord_type.startswith(bt[:3]):
                    if bn not in found: found.append(bn)
    return found

def tonal_centricity(chords, key_root):
    if not chords: return 0.5
    return sum(1 for c in chords if c.root == key_root) / len(chords)

def resolution_tendency(chords, key_root):
    if len(chords) < 2: return 0.0
    dom = (key_root+7)%12
    return sum(1 for i in range(len(chords)-1) if chords[i].root==dom and chords[i+1].root==key_root)/(len(chords)-1)

def analyze_intervals(mel):
    if len(mel) < 2: return {}
    sn = sorted(mel, key=lambda n: n.time_sec)
    ivs = [sn[i+1].pitch-sn[i].pitch for i in range(len(sn)-1)]
    if not ivs: return {}
    ai = [abs(i) for i in ivs]
    conj  = sum(1 for i in ai if i<=2)/len(ai)
    leaps = sum(1 for i in ai if i>=7)/len(ai)
    trit  = sum(1 for i in ai if i==6)/len(ai)
    avg   = sum(ai)/len(ai)
    asc   = sum(1 for i in ivs if i>0)/len(ivs)
    dc    = sum(1 for i in range(1,len(ivs)) if ivs[i]!=0 and ivs[i-1]!=0 and (ivs[i]>0)!=(ivs[i-1]>0))
    parts = []
    if conj > 0.7:    parts.append("muy fluida y cantable")
    elif conj > 0.5:  parts.append("equilibrada entre grados conjuntos y saltos")
    else:             parts.append("angulosa, con saltos amplios")
    if trit > 0.05:   parts.append(f"tritonos ({trit*100:.0f}%) — tensión, inquietud")
    if leaps > 0.2:   parts.append(f"saltos ≥7st ({leaps*100:.0f}%) — dramatismo")
    if asc > 0.6:     parts.append("impulso ascendente — búsqueda")
    elif asc < 0.4:   parts.append("tendencia descendente — resignación")
    return {'conjunct':conj,'leaps':leaps,'tritones':trit,'avg_interval':avg,
            'ascending':asc,'dir_changes':dc,'desc':'; '.join(parts)}

def analyze_rhythm(notes, bpm, ts_num=4):
    if not notes or bpm<=0: return {'syncopation':0,'swing':1,'variety':0,'desc':'indeterminado'}
    beat=60/bpm; sub=beat/4
    sn=sorted(notes,key=lambda n:n.time_sec)
    sync=sum(1 for n in notes if 0.25<(n.time_sec%sub)/sub<0.75)/len(notes)
    pairs=[sn[i+1].time_sec-sn[i].time_sec for i in range(min(40,len(sn)-1)) if 0.05<sn[i+1].time_sec-sn[i].time_sec<beat*1.5]
    swing=1.0
    if len(pairs)>=4:
        lo,sh=pairs[0::2],pairs[1::2]
        if sh and sum(sh)/len(sh)>0: swing=(sum(lo)/len(lo))/(sum(sh)/len(sh))
    iois=[sn[i+1].time_sec-sn[i].time_sec for i in range(len(sn)-1) if sn[i+1].time_sec-sn[i].time_sec>0]
    variety=0.0
    if iois:
        mi=sum(iois)/len(iois); var=sum((x-mi)**2 for x in iois)/len(iois)
        variety=math.sqrt(var)/mi if mi>0 else 0
    parts=[]
    if sync>0.4: parts.append(f"alta síncopa ({sync*100:.0f}%) — groove, tensión rítmica")
    elif sync>0.2: parts.append(f"síncopa moderada ({sync*100:.0f}%)")
    else: parts.append("ritmo regular — estabilidad")
    if swing>1.4: parts.append(f"swing pronunciado ({swing:.2f}x)")
    elif swing>1.15: parts.append(f"ligero swing ({swing:.2f}x)")
    if variety>1.0: parts.append("gran variedad rítmica")
    elif variety<0.3: parts.append("ostinato rítmico")
    return {'syncopation':sync,'swing':swing,'variety':variety,'desc':'; '.join(parts)}

def extract_motifs(notes, min_len=3, max_len=5, min_occ=2):
    if len(notes)<min_len*2: return []
    sn=sorted(notes,key=lambda n:n.time_sec)
    ivs=[sn[i+1].pitch-sn[i].pitch for i in range(len(sn)-1)]
    found=[]; seen=set()
    for length in range(min_len,min(max_len+1,len(ivs))):
        ctr=collections.Counter(tuple(ivs[i:i+length]) for i in range(len(ivs)-length+1))
        for pat,cnt in ctr.most_common(3):
            if cnt>=min_occ and pat not in seen:
                seen.add(pat); span=sum(pat); ab=sum(abs(x) for x in pat)
                d='ascendente' if span>3 else ('descendente' if span<-3 else 'circular')
                d2='conjuntos' if ab<=len(pat)*2 else 'con saltos'
                t6='con tritono ' if 6 in pat or -6 in pat else ''
                istr=' '.join((f"+{x}" if x>0 else str(x)) for x in pat)
                found.append({'pat':pat,'len':length,'occ':cnt,
                              'desc':f"Motivo {d} {t6}por grados {d2} [{istr}] ×{cnt}"})
                if len(found)>=5: return found
    return found

def analyze_texture(notes, res=0.25):
    if not notes: return {'avg_voices':0,'max_voices':0,'type':'silencio','spacing':0,'desc':''}
    total=max(n.time_sec+n.duration for n in notes)
    vc=[]; sp=[]; t=0.0
    while t<total:
        ac=[n for n in notes if n.time_sec<=t<n.time_sec+n.duration]; vc.append(len(ac))
        if len(ac)>=2: sp.append(max(n.pitch for n in ac)-min(n.pitch for n in ac))
        t+=res
    av=sum(vc)/len(vc) if vc else 0; mx=max(vc) if vc else 0; asp=sum(sp)/len(sp) if sp else 0
    if av<1.3: tt,td='monofónica','Una sola línea — máxima fragilidad y exposición'
    elif av<2.5: tt,td='homofónica simple','Melodía+acompañamiento — claridad y dirección'
    elif av<4: tt,td='homofónica plena','Bloque de voces — plenitud, solidez'
    elif av<6: tt,td='polifónica','Múltiples voces independientes — riqueza contrapuntística'
    else: tt,td='orquestal densa','Masa sonora — poder o caos'
    sd=('muy cercanas — tensión, opresión' if asp<12 else
        'natural — equilibrio' if asp<24 else
        'amplio — grandeza' if asp<36 else 'extremo — lo sublime')
    return {'avg_voices':av,'max_voices':mx,'type':tt,'spacing':asp,
            'desc':f"{td}. Espaciado: voces {sd}."}

def detect_accompaniment(notes, bpm):
    if len(notes)<8 or bpm<=0: return 'indeterminado'
    beat=60/bpm; bass=[n for n in notes if n.pitch<60]; upper=[n for n in notes if n.pitch>=60]
    if not bass or not upper: return 'sin patrón claro'
    bs=sorted(bass,key=lambda n:n.time_sec)
    iois=[bs[i+1].time_sec-bs[i].time_sec for i in range(min(20,len(bs)-1))]
    if iois:
        mi=sum(iois)/len(iois)
        reg=1-(sum(abs(x-mi) for x in iois)/(mi*len(iois)+0.001))
        if reg>0.7:
            pv=max(n.pitch for n in bs[:12])-min(n.pitch for n in bs[:12])
            if mi<beat/3: return 'arpegio rápido — movimiento continuo, romanticismo'
            elif mi<beat*0.75: return ('bajo alberti — clasicismo' if pv>8 else 'ostinato en corcheas')
            else: return 'bajo en tiempos fuertes — estabilidad rítmica'
    ratio=len(upper)/max(len(bass),1)
    if 1.8<ratio<2.5: return 'patrón de vals (bajo-acorde-acorde) — danza, lirismo'
    return 'acompañamiento libre o mixto'

def detect_tempo_gestures(tempo_changes):
    if len(tempo_changes)<2: return []
    gest=[]
    for i in range(1,len(tempo_changes)):
        t0,b0=tempo_changes[i-1]; t1,b1=tempo_changes[i]
        diff=b1-b0; pct=abs(diff)/b0*100 if b0>0 else 0
        if pct<2: continue
        tstr=ts_str(t1)
        if diff>0: gest.append(f"  Accelerando en {tstr} ({b0:.0f}→{b1:.0f} BPM, +{pct:.0f}%)")
        else: gest.append(f"  Rallentando en {tstr} ({b0:.0f}→{b1:.0f} BPM, -{pct:.0f}%)")
    return gest

def melodic_contour(notes):
    if len(notes)<4: return 'indeterminado',[]
    mel=sorted(notes,key=lambda n:n.time_sec)
    bt=collections.defaultdict(list)
    for n in mel: bt[n.track].append(n)
    mt=max(bt,key=lambda t:sum(n.pitch for n in bt[t])/len(bt[t]))
    mn=sorted(bt[mt],key=lambda n:n.time_sec)
    if len(mn)<4: mn=mel
    n5=5; ck=max(len(mn)//n5,1)
    avp=[sum(n.pitch for n in mn[i*ck:(i+1)*ck])/ck for i in range(n5) if mn[i*ck:(i+1)*ck]]
    if not avp: return 'indeterminado',[]
    p=avp; pk=p.index(max(p)); tr=p.index(min(p))
    if pk==0 and p[-1]<p[0]: sh='descendente — caída, abandono'
    elif pk==len(p)-1: sh='ascendente — búsqueda, tensión creciente'
    elif 1<=pk<=len(p)-2 and p[0]<p[pk] and p[-1]<p[pk]: sh='arco ▲ — construcción y liberación'
    elif 1<=tr<=len(p)-2 and p[0]>p[tr] and p[-1]>p[tr]: sh='valle ▽ — descenso y recuperación'
    elif abs(p[-1]-p[0])<2: sh='circular — retorno al inicio, equilibrio'
    else: sh='ondulante — ambivalencia, narrativa compleja'
    return sh,avp

def find_similar_sections(sections):
    def sim(a,b):
        dot=sum(x*y for x,y in zip(a,b))
        ma=math.sqrt(sum(x**2 for x in a)); mb=math.sqrt(sum(x**2 for x in b))
        return dot/(ma*mb) if ma>0 and mb>0 else 0
    ac=[s for s in sections if s.notes]; result=[]
    for i in range(len(ac)):
        for j in range(i+1,len(ac)):
            s=sim(ac[i].chroma,ac[j].chroma)
            if s>0.90: result.append((ac[i].index,ac[j].index,s))
    return result

def detect_buildups(sections):
    evts=[]
    for i in range(1,len(sections)):
        a,b=sections[i-1],sections[i]
        if not a.notes or not b.notes: continue
        vd=b.avg_velocity-a.avg_velocity; dd=b.density-a.density
        tstr=ts_str(b.start)
        if vd>15 and dd>2: evts.append(f"  Build-up en {tstr} (+{vd:.0f} vel, +{dd:.1f} n/s)")
        elif vd<-15 and dd<-2: evts.append(f"  Release en {tstr} ({vd:.0f} vel, {dd:.1f} n/s)")
        elif vd>12: evts.append(f"  Crescendo en {tstr} (+{vd:.0f})")
        elif vd<-12: evts.append(f"  Decrescendo en {tstr} ({vd:.0f})")
    return evts

def section_tension_score(sec: Section) -> float:
    if len(sec.notes)<2: return 0.0
    ps=[n.pitch%12 for n in sec.notes]; tot=0.0; pairs=0
    for i in range(len(ps)):
        for j in range(i+1,min(i+6,len(ps))):
            tot+=DISSONANCE[abs(ps[i]-ps[j])%12]; pairs+=1
    return tot/pairs if pairs>0 else 0

def find_climax(sections):
    sc=[s.avg_velocity*0.4+s.density*4+section_tension_score(s)*25 for s in sections]
    return sc.index(max(sc)) if sc else 0

def get_mel_track(notes):
    bt=collections.defaultdict(list)
    for n in notes: bt[n.track].append(n)
    if not bt: return notes
    best=max(bt,key=lambda t:sum(n.pitch for n in bt[t])/len(bt[t]))
    return bt[best]




SECTION_NAMES = ['Introducción','Desarrollo A','Desarrollo B','Clímax',
                 'Recapitulación','Coda','Epílogo','Cierre Final']

def generate_report(
    filepath, notes, meta, sections, chords,
    key_root, mode, kconf, avg_bpm, main_bpm,
    ts_str_val, total_dur,
    # analysis results
    canon_progs, modal_borrow, centricity, resolution,
    interval_anal, contour, contour_p,
    motifs, rhythm, texture, accomp,
    tempo_gestures, instruments,
    similar_secs, builds, overall_tension,
    tension_curve, cadences, voice_leading,
    expectation, register_analysis,
    pedal_points, harmonic_graph, musical_form,
    n_sections,
    # v4
    silence_analysis=None, energy_profile=None,
    narrative_arc=None, emotional_transitions=None,
    suspense=None, thematic_transforms=None,
    polarity=None, event_density=None,
    emotional_genre=None, chromaticism=None,
    climax_comparison=None, groove=None, sva_list=None,
    # v5
    phrasing=None, counterpoint=None, microexpression=None,
    dissonance_res=None, golden=None, info_density=None,
    tonal_ambiguity=None, perceptual=None, inner_voice=None,
    cinematic=None, fingerprint=None, eternity=None,
    coherence=None, story=None,
    # v6
    dyn_segments=None, tonal_gravity=None, subtext=None,
    momentum=None, auditory_fatigue=None, polyrhythm=None,
    fractal=None, emotional_weight=None, pnr=None,
    narrative_density=None, cumulative_state=None,
    harmonic_color=None, register_harmony=None, pitch_bend=None,
    # v7
    cultural=None, narrative_voice=None, micro_timing=None,
    param_convergence=None, comfort_zones=None, metric_hierarchy=None,
    elliptical=None, thematic_recurrence=None, listener_model=None,
    harmonic_semantic=None, bass_narrative=None, layer_sync=None,
    # v8
    essay_text=None,
    # ssm
    ssm=None,
    # v9 — deep emotional & intentionality
    dynamic_valence=None, catharsis=None,
    emotional_ambivalence=None, emotional_trajectory=None,
    anti_conventional=None, multilevel_density=None,
    narrative_intention=None,
    # v10 — roughness, markov, semantic
    roughness=None, melodic_markov=None, semantic_enrichment=None,
    # v11 — genre detection
    genre_detection=None,
    # v12 — unified emotional map
    unified_emotional_map=None,
) -> str:

    L = []
    SEP = '═' * 66
    S2  = '─' * 52
    climax = find_climax(sections)

    # Compute VA for all sections
    sync = rhythm.get('syncopation', 0)
    mean_exp = expectation.get('mean_surprise', 0.2)
    sva = []
    for s in sections:
        if not s.notes: sva.append((0, 0, 'sin notas')); continue
        v, a = compute_va(s, mode, avg_bpm, section_tension_score(s), sync, mean_exp)
        sva.append((v, a, describe_va(v, a)))

    title = filepath.split('/')[-1].replace('.mid','').replace('.midi','').replace('_',' ')
    mode_label, mode_emotion = MODE_EMOTIONS.get(mode, ('?', '?'))
    key_name = f"{NOTE_NAMES[key_root]} {mode_label}"
    if meta.get('key_signature'):
        key_name += f"  [MIDI: {meta['key_signature']}]"

    # ── CABECERA ───────────────────────────────────────────────
    L += [SEP, f"  🎵  ANÁLISIS MUSICAL COMPLETO  v12.0",
          f"  {title.upper()}", SEP]

    # ── RESUMEN EJECUTIVO ──────────────────────────────────────
    L.append(write_executive_summary(
        key_name, mode, avg_bpm, total_dur,
        genre_detection=genre_detection,
        dynamic_valence=dynamic_valence,
        narrative_intention=narrative_intention,
        catharsis=catharsis,
        anti_conventional=anti_conventional,
        semantic_enrichment=semantic_enrichment,
        unified_emotional_map=unified_emotional_map,
        roughness=roughness,
        melodic_markov=melodic_markov,
    ))

    # ── ENSAYO NARRATIVO ───────────────────────────────────────
    if essay_text:
        L += ["", "╔══ 71. ANÁLISIS MUSICAL — ENSAYO ESTRUCTURADO " + "═"*17, ""]
        for ln in essay_text.split('\n'):
            L.append(ln)
        L.append("")
        L += ["", '─' * 66,
              "  ▼  INFORME ESTADÍSTICO COMPLETO  ▼",
              '─' * 66, ""]

    # ══ 1. FICHA TÉCNICA ═══════════════════════════════════════
    L += ["", f"╔══ 1. FICHA TÉCNICA {'═'*46}"]
    L.append(f"  Duración        : {int(total_dur//60)}m {int(total_dur%60):02d}s  ({total_dur:.1f}s)")
    L.append(f"  Tonalidad       : {key_name}  (conf. KS: {kconf:.3f})")
    L.append(f"  Modo específico : {mode_label} — {mode_emotion}")
    L.append(f"  Tempo principal : {main_bpm:.1f} BPM — {classify_tempo(main_bpm)}")
    if len(meta['tempo_changes']) > 1:
        L.append(f"  Cambios tempo   : {len(meta['tempo_changes'])-1}")
    L.append(f"  Compás          : {ts_str_val}")
    L.append(f"  Pistas/Canales  : {meta['num_tracks']} / {len(set(n.channel for n in notes))}")
    L.append(f"  Total notas     : {len(notes)}")
    vel = [n.velocity for n in notes]
    L.append(f"  Dinámica        : pp={min(vel)}  ff={max(vel)}  μ={sum(vel)/len(vel):.1f}")
    mn_p = min(n.pitch for n in notes); mx_p = max(n.pitch for n in notes)
    L.append(f"  Rango           : {NOTE_NAMES[mn_p%12]}{mn_p//12-1} → {NOTE_NAMES[mx_p%12]}{mx_p//12-1}")
    if instruments:
        L.append(f"  Instrumentos GM :")
        for ch, (prog, name, emo) in instruments.items():
            L.append(f"    Canal {ch:2d}: {name} (prog.{prog}) — {emo}")

    # ══ 2. MODO Y CARÁCTER EMOCIONAL GLOBAL ═══════════════════
    tc = tempo_cat(avg_bpm)
    em_map = {
        ('major','slow'):'contemplación, nostalgia tierna',
        ('major','medium'):'calidez, esperanza, serenidad activa',
        ('major','fast'):'alegría, triunfo, energía',
        ('minor','slow'):'tristeza profunda, introspección, duelo',
        ('minor','medium'):'melancolía, añoranza, drama',
        ('minor','fast'):'urgencia, angustia oscura',
        ('dorian','medium'):'melancolía activa, soul, esperanza oscura',
        ('phrygian','medium'):'amenaza, misterio, fatalismo',
        ('lydian','medium'):'ensueño, flotación, lo sublime',
        ('mixolydian','medium'):'épico, folk, ambigüedad heroica',
    }
    ge = em_map.get((mode, tc), em_map.get(('minor' if 'min' in mode else 'major', tc), 'emoción compleja'))

    L += ["", f"╔══ 2. MODO Y CARÁCTER EMOCIONAL {'═'*33}",
          f"  Modo            : {mode_label}",
          f"  Emoción de modo : {mode_emotion}",
          f"  Emoción global  : {ge}",
          f"  Tensión media   : {overall_tension:.3f}/1.0  "
          f"({'alta' if overall_tension>0.5 else 'moderada' if overall_tension>0.25 else 'baja'})",
          f"  Centricidad ton.: {centricity:.2f}  "
          f"({'anclada en tónica' if centricity>0.4 else 'exploratoria' if centricity<0.15 else 'equilibrada'})",
          f"  Cadencias V→I   : {resolution*100:.0f}% resueltas",
          f"  Contorno mel.   : {contour}",
          f"  Textura         : {texture['type']} — μ {texture['avg_voices']:.1f} voces"]

    # ══ 3. CURVA DE TENSIÓN CONTINUA ══════════════════════════
    L += ["", f"╔══ 3. CURVA DE TENSIÓN CONTINUA {'═'*33}"]
    arc_desc = describe_tension_arc(tension_curve)
    L.append(f"  Arco de tensión : {arc_desc}")
    peaks = find_tension_peaks(tension_curve)
    valleys = find_tension_valleys(tension_curve)
    if peaks:
        peak_strs = [f"{ts_str(p.time)} ({p.tension:.2f}{'  ←'+p.event if p.event else ''})"
                     for p in peaks[:5]]
        L.append(f"  Picos de tensión: {', '.join(peak_strs)}")
    if valleys:
        val_strs = [ts_str(v.time) for v in valleys[:4]]
        L.append(f"  Momentos de alivio: {', '.join(val_strs)}")
    L.append("")
    chart = tension_curve_ascii(tension_curve, width=58, height=8)
    if chart:
        L.append(f"  Tensión (eje Y: 0.0→1.0, eje X: tiempo)")
        for ln in chart.split('\n'):
            L.append('  ' + ln)

    # ══ 4. ANÁLISIS DE CADENCIAS ══════════════════════════════
    L += ["", f"╔══ 4. ANÁLISIS DE CADENCIAS {'═'*37}",
          f"  {cadences['desc']}",
          f"  Total detectadas: {cadences['total']}  |  Tipo dominante: {cadences['dominant_type']}"]
    if cadences.get('counts'):
        for ctype, count in cadences['counts'].most_common():
            cdata = CADENCE_TYPES.get(ctype, ('?','?'))
            L.append(f"    {ctype:14s} ×{count:2d} — {cdata[1]}")

    # ══ 5. FUNCIÓN ARMÓNICA Y PROGRESIONES ════════════════════
    L += ["", f"╔══ 5. FUNCIÓN ARMÓNICA Y PROGRESIONES {'═'*27}"]
    # Show chord sequence with function labels
    if chords:
        uq = list(dict.fromkeys(c.name for c in chords))
        func_seq = []
        for c in chords[:16]:
            fn = c.function if c.function else '?'
            func_seq.append(f"{c.name}[{fn}]")
        L.append(f"  Secuencia (inicio): {' → '.join(func_seq[:10])}")
        L.append(f"  Acordes únicos    : {len(uq)}")
        L.append(f"  Principales       : {' — '.join(uq[:8])}")

    if canon_progs:
        L.append(f"  Prog. canónicas :")
        for p in canon_progs: L.append(f"    ✓ {p}")
    else:
        L.append(f"  Prog. canónicas   : ninguna estándar — lenguaje harmónico propio")

    if modal_borrow:
        L.append(f"  Préstamos modales :")
        for b in modal_borrow: L.append(f"    ◆ {b}")
    else:
        L.append(f"  Préstamos modales : armonía diatónica pura")

    # Harmonic graph summary
    hg = harmonic_graph
    L += ["", f"  Grafo armónico:"]
    L.append(f"    Centro gravitacional : {hg.get('center','?')}")
    L.append(f"    Entropía armónica    : {hg.get('entropy',0):.2f}  — {hg.get('desc','')}")
    if hg.get('top_str'):
        L.append(f"    Transiciones + frec. : {hg['top_str']}")

    # ══ 6. VOICE LEADING Y PEDAL ══════════════════════════════
    L += ["", f"╔══ 6. VOICE LEADING Y PEDAL POINT {'═'*30}",
          f"  {voice_leading['desc']}",
          f"  Fluidez VL   : {voice_leading['smoothness']:.2f}/1.0",
          f"  Mov. cromático: {voice_leading.get('chromatic_moves',0)}  |  "
          f"Mov. 5ª: {voice_leading.get('fifths',0)}  |  "
          f"Saltos: {voice_leading.get('leaps',0)}"]
    if pedal_points:
        L.append(f"  Pedales detectados:")
        for p in pedal_points:
            L.append(f"    {ts_str(p['start'])}: pedal de {p['note']} ({p['type']}) — {p['emotion']}")
    else:
        L.append(f"  Sin puntos de pedal detectados")

    # ══ 7. EXPECTATIVA MELÓDICA (IDyOM simplificado) ══════════
    L += ["", f"╔══ 7. EXPECTATIVA MELÓDICA (modelo IDyOM simplif.) {'═'*14}",
          f"  Sorpresa media : {expectation.get('mean_surprise',0):.3f}",
          f"  Descripción    : {expectation.get('desc','')}"]
    if expectation.get('surprise_peaks'):
        peaks_str = ', '.join(f"{ts_str(t)} ({s:.2f})"
                              for t,s in expectation['surprise_peaks'][:5])
        L.append(f"  Momentos de máxima sorpresa: {peaks_str}")

    # ══ 8. ANÁLISIS DE REGISTROS ══════════════════════════════
    L += ["", f"╔══ 8. ANÁLISIS DE REGISTROS POR FUNCIÓN {'═'*25}"]
    for layer in ['soprano','contralto','tenor','bajo']:
        info = register_analysis.get(layer, {})
        if info.get('count', 0) == 0: continue
        L.append(f"  {layer.capitalize():12s}: {info.get('desc','')}  [{info.get('role','')}]")
    if register_analysis.get('anomalies'):
        L.append(f"  Anomalías de registro:")
        for a in register_analysis['anomalies']:
            L.append(f"    ⚠ {a}")

    # ══ 9. FORMA MUSICAL ═════════════════════════════════════
    L += ["", f"╔══ 9. FORMA MUSICAL {'═'*45}",
          f"  Forma detectada : {musical_form['form']}",
          f"  Patrón seccional: {musical_form['pattern']}",
          f"  {musical_form['desc']}"]
    similar = find_similar_sections(sections)  # re-use
    if similar:
        L.append(f"  Similitudes chroma:")
        for s1, s2, sim in similar:
            tag = 'recapitulación' if sim > 0.97 else 'material relacionado'
            L.append(f"    Sec.{s1+1} ({ts_str(sections[s1].start)}) ≈ "
                     f"Sec.{s2+1} ({ts_str(sections[s2].start)})  "
                     f"sim={sim:.3f} — {tag}")

    # ══ 9b. SELF-SIMILARITY MATRIX ═══════════════════════════
    if ssm and ssm.get('n', 0) > 0:
        L.append("")
        L.append(_format_ssm_report(ssm, total_dur))

    # ══ 10. ANÁLISIS MELÓDICO ════════════════════════════════
    L += ["", f"╔══ 10. ANÁLISIS MELÓDICO {'═'*40}"]
    if interval_anal:
        L += [f"  Movimiento  : {interval_anal.get('desc','')}",
              f"  Grados conj.: {interval_anal.get('conjunct',0)*100:.0f}%  |  "
              f"Saltos ≥7st: {interval_anal.get('leaps',0)*100:.0f}%",
              f"  Tritonos    : {interval_anal.get('tritones',0)*100:.1f}%  |  "
              f"Intervalo μ: {interval_anal.get('avg_interval',0):.1f} st",
              f"  Ascendente  : {interval_anal.get('ascending',0)*100:.0f}%  |  "
              f"Cambios dir: {interval_anal.get('dir_changes',0)}"]
    L.append(f"  Acompañ.    : {accomp}")
    if motifs:
        L.append(f"  Motivos recurrentes:")
        for m in motifs: L.append(f"    • {m['desc']}")

    # ══ 11. ANÁLISIS RÍTMICO ═════════════════════════════════
    L += ["", f"╔══ 11. ANÁLISIS RÍTMICO {'═'*41}",
          f"  {rhythm.get('desc','')}",
          f"  Síncopa  : {rhythm.get('syncopation',0)*100:.0f}%",
          f"  Swing    : {rhythm.get('swing',1):.2f}x  "
          f"{'(jazz/blues)' if rhythm.get('swing',1)>1.4 else '(straight)' if rhythm.get('swing',1)<1.1 else '(ligero)'}",
          f"  Variedad : {rhythm.get('variety',0):.2f}  "
          f"({'libre/expresivo' if rhythm.get('variety',0)>1 else 'mecánico/ostinato' if rhythm.get('variety',0)<0.3 else 'equilibrado'})"]

    # ══ 12. TEXTURA Y ORQUESTACIÓN ═══════════════════════════
    L += ["", f"╔══ 12. TEXTURA Y ORQUESTACIÓN {'═'*35}",
          f"  {texture['desc']}",
          f"  Voces: μ={texture['avg_voices']:.1f}  máx={texture['max_voices']}  "
          f"espaciado μ={texture['spacing']:.1f}st"]

    # ══ 13. TEMPO Y DINÁMICA ═════════════════════════════════
    L += ["", f"╔══ 13. TEMPO Y DINÁMICA {'═'*41}"]
    if tempo_gestures:
        for g in tempo_gestures: L.append(g)
    else:
        L.append(f"  Tempo estable ({main_bpm:.1f} BPM) a lo largo de toda la pieza")
    L.append(f"\n  Dinámica por sección:")
    BW = 38
    for i, s in enumerate(sections):
        if not s.notes: continue
        nb = int(s.avg_velocity/127*BW); bar = '█'*nb + '░'*(BW-nb)
        cm = ' ◀CLÍMAX' if i == climax else ''
        L.append(f"  {ts_str(s.start)}  [{bar}]  {s.avg_velocity:.0f}{cm}")
    if builds:
        L.append(f"\n  Gestos dinámicos:")
        for ev in builds: L.append(ev)

    # ══ 14. ARCO EMOCIONAL ═══════════════════════════════════
    L += ["", f"╔══ 14. ARCO EMOCIONAL POR SECCIONES {'═'*29}"]
    for i, s in enumerate(sections):
        if not s.notes: continue
        nm = SECTION_NAMES[i] if i < len(SECTION_NAMES) else f'Sección {i+1}'
        t_range = f"{ts_str(s.start)}–{ts_str(s.end)}"
        v, a, vad = sva[i]; ten = section_tension_score(s)
        cm = '  ◀ PUNTO ÁLGIDO' if i == climax else ''
        L.append(f"\n  [{i+1}] {nm.upper()}  |  {t_range}{cm}")
        ib = '█'*int(s.avg_velocity/127*20) + '░'*(20-int(s.avg_velocity/127*20))
        tb = '▓'*int(ten*10) + '░'*(10-int(ten*10))
        L += [f"      Intensidad [{ib}] {s.avg_velocity:.0f}/127   Tensión [{tb}] {ten:.2f}",
              f"      Densidad {s.density:.1f} n/s  |  Rango: {s.pitch_range}st",
              f"      Valence-Arousal: {vad}"]

    # ══ 15. MAPA VALENCE-AROUSAL ═════════════════════════════
    act_va = [(v, a, d) for (v,a,d), s in zip(sva, sections) if s.notes]
    L += ["", f"╔══ 15. MAPA VALENCE–AROUSAL (Russell/Thayer) {'═'*19}", ""]
    for ln in va_chart(act_va[:10]).split('\n'): L.append('  ' + ln)
    L.append("\n  Leyenda:")
    lbs = '①②③④⑤⑥⑦⑧⑨⑩'
    for i, (s, (v, a, d)) in enumerate(zip(sections, sva)):
        if not s.notes: continue
        nm = SECTION_NAMES[i] if i < len(SECTION_NAMES) else f'Sec {i+1}'
        L.append(f"  {lbs[i] if i<len(lbs) else f'[{i+1}]'} {nm}: {d}")

    # ══ 16. SILENCIO Y RESPIRACIÓN ═══════════════════════════
    sa = silence_analysis or {}
    L += ["", f"╔══ 16. SILENCIO Y RESPIRACIÓN {'═'*35}",
          f"  {sa.get('desc','')}",
          f"  Ratio silencio  : {sa.get('ratio',0)*100:.1f}% de la duración",
          f"  Silencios totales: {len(sa.get('silences',[]))}"]
    for stype, cnt in sorted((sa.get('by_type') or {}).items(), key=lambda x:-x[1]):
        L.append(f"    {stype:22s}: x{cnt}")
    md = sa.get('most_dramatic')
    if md: L.append(f"  Más dramático: {ts_str(md['start'])} — {md['duration']:.1f}s ({md['type']})")

    # ══ 17. PERFIL DE ENERGÍA ═════════════════════════════════
    ep = energy_profile or {}
    L += ["", f"╔══ 17. PERFIL DE ENERGÍA CONTINUA {'═'*30}",
          f"  {ep.get('desc','')}",
          f"  Pico       : {ts_str(ep.get('peak_time',0))}",
          f"  RMS        : {ep.get('rms',0):.3f}  |  Rango dinámico: {ep.get('dyn_range',0):.2f}",""]
    if ep.get('curve'):
        echart = energy_ascii(ep['curve'], width=56, height=6)
        if echart:
            L.append("  Energía (masa sonora × tiempo):")
            for ln in echart.split('\n'): L.append('  ' + ln)

    # ══ 18. ARCO NARRATIVO ════════════════════════════════════
    na = narrative_arc or {}
    L += ["", f"╔══ 18. ARCO NARRATIVO EN TRES ACTOS {'═'*28}",
          f"  {na.get('arc_type','')}",
          f"  {na.get('arc_desc','')}",
          f"  Narrativa  : {na.get('narrative','')}",
          f"  Clímax en  : {ts_str(na.get('climax_time',0))} ({na.get('climax_position',0)*100:.0f}% de la pieza)"]
    for act in na.get('acts',[]):
        L.append(f"    [{ts_str(act['start'])}–{ts_str(act['end'])}] {act['name']} — {act['role']}")

    # ══ 19. TRANSICIONES EMOCIONALES ═════════════════════════
    et = emotional_transitions or {}
    L += ["", f"╔══ 19. TRANSICIONES EMOCIONALES {'═'*32}",
          f"  {et.get('journey','')}",
          f"  Suavidad narrativa: {et.get('smoothness',0):.2f}/1.0  |  Mayor giro: {et.get('max_shift',0):.3f} VA"]
    for t in et.get('transitions',[]):
        L.append(f"    {t['emoji']} {ts_str(t['time'])}: {t['type']} — {t['direction']} (delta={t['distance']:.2f})")

    # ══ 20. SUSPENSE Y ANTICIPACIÓN ══════════════════════════
    sp = suspense or {}
    L += ["", f"╔══ 20. SUSPENSE Y ANTICIPACIÓN {'═'*33}",
          f"  {sp.get('desc','')}",
          f"  Índice suspense   : {sp.get('suspense_index',0):.2f}/1.0",
          f"  Tensión sostenida : {sp.get('total_sustained',0):.0f}s ({sp.get('suspense_ratio',0)*100:.0f}%)",
          f"  Falsas resoluc.   : {sp.get('false_resolutions',0)}",
          f"  Ventana más larga : {sp.get('max_sustained',0):.0f}s continua"]

    # ══ 21. TRANSFORMACIONES TEMÁTICAS ════════════════════════
    tt = thematic_transforms or {}
    L += ["", f"╔══ 21. TRANSFORMACIONES TEMÁTICAS {'═'*30}",
          f"  {tt.get('desc','')}"]
    for t in tt.get('transformations',[]):
        tlist = ', '.join(f"{k} x{v}" for k,v in t['transforms'].items())
        L.append(f"    Motivo: {t['motif'][:55]}")
        L.append(f"      Transforms: {tlist}")

    # ══ 22. POLARIDAD EMOCIONAL ═══════════════════════════════
    po = polarity or {}
    L += ["", f"╔══ 22. POLARIDAD EMOCIONAL {'═'*38}",
          f"  {po.get('desc','')}",
          f"  Tiempo positivo: {po.get('positive_ratio',0)*100:.0f}%  |  Polaridad media: {po.get('mean_polarity',0):+.2f}"]
    labs = po.get('polarity_labels',[])
    scrs = po.get('polarity_scores',[])
    if labs and scrs:
        bar = ''.join('█' if s>0.2 else ('▓' if s>-0.2 else '░') for s in scrs)
        L.append(f"  Arco: {bar}  (█=luz  ▓=neutro  ░=sombra)")

    # ══ 23. DENSIDAD DE EVENTOS ═══════════════════════════════
    ed = event_density or {}
    L += ["", f"╔══ 23. DENSIDAD DE EVENTOS SIGNIFICATIVOS {'═'*22}",
          f"  {ed.get('desc','')}",
          f"  Eventos/min   : {ed.get('events_per_minute',0):.0f}  ({ed.get('style','')}) ",
          f"  Cambios acorde: {ed.get('chord_changes',0)}  |  Picos tensión: {ed.get('tension_peaks',0)}",
          f"  Silencios sign: {ed.get('significant_silences',0)}  |  Saltos registro: {ed.get('register_leaps',0)}"]

    # ══ 24. GÉNERO EMOCIONAL ══════════════════════════════════
    eg = emotional_genre or {}
    L += ["", f"╔══ 24. CLASIFICACIÓN DE GÉNERO EMOCIONAL {'═'*23}",
          f"  {eg.get('desc','')}"]
    for genre, score in eg.get('ranking',[]):
        bar = '█'*int(score*20)+'░'*(20-int(score*20))
        adesc = ARCHETYPES.get(genre,{}).get('desc','')
        L.append(f"    {genre:22s} [{bar}] {score*100:.0f}%")
        L.append(f"      {adesc}")

    # ══ 25. CROMATISMO ════════════════════════════════════════
    ch2 = chromaticism or {}
    L += ["", f"╔══ 25. ANÁLISIS DE CROMATISMO {'═'*34}",
          f"  {ch2.get('desc','')}",
          f"  Dirigido  : {ch2.get('directed_ratio',0)*100:.1f}%  (semitonal con dirección)",
          f"  De color  : {ch2.get('color_ratio',0)*100:.1f}%  (cromatismo sin dirección)"]
    for line in ch2.get('lines',[])[:3]:
        L.append(f"    {line['direction']:12s} en {ts_str(line['start'])}: {line['length']} semitonos")

    # ══ 26. CLÍMAX EMOCIONAL vs SONORO ════════════════════════
    cc = climax_comparison or {}
    L += ["", f"╔══ 26. CLÍMAX EMOCIONAL vs CLÍMAX SONORO {'═'*23}",
          f"  {cc.get('desc','')}",
          f"  Clímax emocional: {ts_str(cc.get('emotional_climax',0))}",
          f"  Clímax sonoro   : {ts_str(cc.get('sonic_climax',0))}",
          f"  Desfase         : {cc.get('offset',0):.1f}s  ({cc.get('offset_pct',0):.1f}% de la pieza)"]

    # ══ 27. GROOVE E HIPNOSIS ═════════════════════════════════
    gr = groove or {}
    L += ["", f"╔══ 27. GROOVE E HIPNOSIS {'═'*39}",
          f"  {gr.get('desc','')}",
          f"  Fuerza groove  : {gr.get('groove_strength',0):.2f}/1.0",
          f"  Ostinato ratio : {gr.get('ostinato_ratio',0)*100:.1f}%",
          f"  Secc. repetit. : {gr.get('max_repetition_sec',0):.0f}s"]

    # ══ v5 NEW REPORT SECTIONS ══════════════════════════════

    # ══ 29. FRASEO ═══════════════════════════════════════════
    ph = phrasing or {}
    L += ["", f"╔══ 29. ANÁLISIS DE FRASEO {'═'*38}",
          f"  {ph.get('desc','')}",
          f"  Frases detectadas : {ph.get('count',0)}",
          f"  Longitud media    : {ph.get('avg_length',0):.1f}s  ({ph.get('avg_length_bars',0):.1f} compases)",
          f"  Simetría          : {ph.get('symmetry',0):.2f}/1.0",
          f"  Pregunta-respuesta: {'sí' if ph.get('antecedent_consequent') else 'no'} (ratio={ph.get('ac_ratio',0):.2f})"]
    for i, p in enumerate(ph.get('phrases',[])[:6]):
        L.append(f"    [{ts_str(p['start'])}] {p['length_sec']:.1f}s — {p['contour']} — {p['note_count']} notas")

    # ══ 30. CONTRAPUNTO ═══════════════════════════════════════
    cp = counterpoint or {}
    L += ["", f"╔══ 30. CONTRAPUNTO E INTERDEPENDENCIA DE VOCES {'═'*17}",
          f"  {cp.get('desc','')}",
          f"  Tipo de movimiento : {cp.get('motion_type','')}",
          f"  Paralelo: {cp.get('parallel_ratio',0)*100:.0f}%  Contrario: {cp.get('contrary_ratio',0)*100:.0f}%  Oblicuo: {cp.get('oblique_ratio',0)*100:.0f}%",
          f"  Independencia voces: {cp.get('independence',0):.2f}/1.0",
          f"  Imitación: {'detectada' if cp.get('imitation') else 'no detectada'}"]

    # ══ 31. MICRODINÁMICAS ════════════════════════════════════
    me = microexpression or {}
    L += ["", f"╔══ 31. MICRODINÁMICAS EXPRESIVAS {'═'*32}",
          f"  {me.get('desc','')}",
          f"  Expresividad   : {me.get('expressiveness',0):.2f}/1.0",
          f"  Varianza media : {me.get('mean_variance',0):.1f}  |  Rango vel.: {me.get('velocity_range',0)}",
          f"  Acentos        : {me.get('accent_ratio',0)*100:.0f}%  |  Hairpins: {me.get('hairpin_count',0)}"]

    # ══ 32. RESOLUCIÓN DE DISONANCIA ═════════════════════════
    dr = dissonance_res or {}
    L += ["", f"╔══ 32. RESOLUCIÓN DE DISONANCIA EN EL TIEMPO {'═'*18}",
          f"  {dr.get('desc','')}",
          f"  Latencia media  : {dr.get('mean_latency',0):.2f}s",
          f"  Sin resolver    : {dr.get('unresolved_ratio',0)*100:.0f}%  ({dr.get('unresolved',0)} eventos)",
          f"  Ornamental (<0.5s): {dr.get('ornamental',0)}  |  Expresiva: {dr.get('expressive',0)}  |  Estructural: {dr.get('structural',0)}"]

    # ══ 33. PROPORCIÓN ÁUREA ═════════════════════════════════
    gd = golden or {}
    L += ["", f"╔══ 33. SIMETRÍA Y PROPORCIÓN ÁUREA (φ=0.618) {'═'*18}",
          f"  {gd.get('desc','')}",
          f"  Clímax en φ: {'SÍ ✓' if gd.get('golden_climax') else 'no'}"]
    for prop, events in (gd.get('proximity') or {}).items():
        evts_str = ', '.join(f"{e[0]} ({e[1]*100:.1f}%)" for e in events)
        L.append(f"  {prop}: {evts_str}")

    # ══ 34. ENTROPÍA INFORMACIONAL ═══════════════════════════
    id2 = info_density or {}
    L += ["", f"╔══ 34. ENTROPÍA INFORMACIONAL POR SECCIÓN {'═'*22}",
          f"  {id2.get('desc','')}",
          f"  Entropía media : {id2.get('mean_entropy',0):.2f} bits",
          f"  Rango          : {id2.get('min_entropy',0):.2f} – {id2.get('max_entropy',0):.2f} bits"]
    for e in id2.get('section_entropies',[]):
        if e.get('entropy',0) > 0:
            bar = '█'*int(e['entropy']/4*20)+'░'*(20-int(e['entropy']/4*20))
            L.append(f"    Sec.{e['index']+1} ({ts_str(e.get('start',0))}): [{bar}] {e['entropy']:.2f} bits  novelty={e.get('novelty',0):.2f}")

    # ══ 35. AMBIGÜEDAD TONAL ═════════════════════════════════
    ta = tonal_ambiguity or {}
    L += ["", f"╔══ 35. AMBIGÜEDAD TONAL INTENCIONAL {'═'*28}",
          f"  {ta.get('desc','')}",
          f"  Índice ambigüedad: {ta.get('ambiguity_index',0):.2f}/1.0",
          f"  Tonos enteros    : {ta.get('whole_tone_coverage',0)*100:.0f}%  |  Octatónica: {ta.get('octatonic_coverage',0)*100:.0f}%",
          f"  Acordes pivote   : {ta.get('pivot_ratio',0)*100:.0f}%"]

    # ══ 36. INTENSIDAD PERCEPTUAL ═════════════════════════════
    pc2 = perceptual or {}
    L += ["", f"╔══ 36. INTENSIDAD PERCEPTUAL (PSICOFONÍA) {'═'*22}",
          f"  {pc2.get('desc','')}",
          f"  Pico perceptual  : {ts_str(pc2.get('peak_time',0))}  (val={pc2.get('peak_val',0):.2f})",
          f"  Intensidad media : {pc2.get('mean',0):.3f}"]
    if pc2.get('curve'):
        pchart = energy_ascii(pc2['curve'], width=54, height=5)
        if pchart:
            L.append("  Perfil perceptual ponderado:")
            for ln in pchart.split('\n'): L.append('  ' + ln)

    # ══ 37. VOZ INTERNA ══════════════════════════════════════
    iv2 = inner_voice or {}
    L += ["", f"╔══ 37. VOZ INTERNA MÁS EXPRESIVA {'═'*31}",
          f"  {iv2.get('desc','')}",
          f"  Expresividad   : {iv2.get('expressiveness',0):.2f}",
          f"  Notas          : {iv2.get('note_count',0)}"]

    # ══ 38. TROPOS CINEMATOGRÁFICOS ══════════════════════════
    ci = cinematic or {}
    n_tropes = len(ci.get('tropes') or {})
    L += ["", f"╔══ 38. TROPOS CINEMATOGRÁFICOS {'═'*33}",
          f"  Tropos detectados: {n_tropes}  |  {ci.get('desc','')}"]
    for trope_name, trope_data in (ci.get('tropes') or {}).items():
        L.append(f"  ▸ {trope_name}")
        if trope_data.get('times'):
            L.append(f"    t={', '.join(trope_data['times'][:3])}")

    # ══ 39. FINGERPRINT ARMÓNICO ═════════════════════════════
    fp2 = fingerprint or {}
    L += ["", f"╔══ 39. FINGERPRINT ARMÓNICO {'═'*36}",
          f"  {fp2.get('desc','')}"]
    fp_data = fp2.get('fingerprint', {})
    if fp_data:
        L += [f"  Maj/Min ratio    : {fp_data.get('maj_ratio',0)*100:.0f}% / {fp_data.get('min_ratio',0)*100:.0f}%",
              f"  Extensiones 7ª+  : {fp_data.get('extension_ratio',0)*100:.0f}%  |  Sus: {fp_data.get('suspended_ratio',0)*100:.0f}%",
              f"  Movim. 5ª        : {fp_data.get('fifths_preference',0)*100:.0f}%  |  3ª: {fp_data.get('thirds_preference',0)*100:.0f}%  |  Semitono: {fp_data.get('semitone_preference',0)*100:.0f}%"]

    # ══ 40. MOMENTOS DE ETERNIDAD ════════════════════════════
    et2 = eternity or {}
    L += ["", f"╔══ 40. MOMENTOS DE ETERNIDAD {'═'*35}",
          f"  {et2.get('desc','')}"]
    for m in et2.get('moments',[]):
        L.append(f"  ◉ {ts_str(m['time'])} [{m['type']}] — {m['desc']}")

    # ══ 41. COHERENCIA EMOCIONAL ═════════════════════════════
    co = coherence or {}
    L += ["", f"╔══ 41. COHERENCIA EMOCIONAL INTERNA {'═'*28}",
          f"  Índice coherencia: {co.get('coherence',1.0):.2f}/1.0  |  "
          f"Contradicciones: {co.get('contradiction_count',0)}"]
    for c2 in co.get('contradictions',[]):
        L.append(f"  ↔ [{c2['type']}]")

    # ══ 42. HISTORIA EMOCIONAL ═══════════════════════════════
    L += ["", f"╔══ 42. HISTORIA EMOCIONAL {'═'*39}",
          "  → Desarrollado en sección IX del ensayo narrativo."]

    # ══ v6 SECTIONS ════════════════════════════════════════

    # ══ 44. SEGMENTACIÓN DINÁMICA ════════════════════════════
    ds = dyn_segments or []
    L += ["", f"╔══ 44. SEGMENTACIÓN DINÁMICA (límites naturales) {chr(9552)*14}",
          f"  Secciones detectadas: {len(ds)} (vs {n_sections} fijas del análisis estático)"]
    for seg in ds:
        bv = int(seg['avg_velocity']/127*20)
        bar = chr(9608)*bv + chr(9617)*(20-bv)
        L.append(f"  [{ts_str(seg['start'])}–{ts_str(seg['end'])}] {seg['duration']:.1f}s  [{bar}] vel={seg['avg_velocity']:.0f}  {seg['note_count']} notas")

    # ══ 45. GRAVEDAD TONAL DINÁMICA ══════════════════════════
    tg = tonal_gravity or {}
    L += ["", f"╔══ 45. GRAVEDAD TONAL DINÁMICA {chr(9552)*34}",
          f"  {tg.get('desc','')}",
          f"  Centro dominante: {tg.get('dominant_key','')} ({tg.get('dominant_pct',0):.0f}% del tiempo)",
          f"  Modulaciones    : {tg.get('n_modulations',0)}  |  Distancia total: {tg.get('total_distance',0):.0f} quintas",
          f"  Itinerario tonal: {tg.get('tonal_journey','')}"]
    for m in (tg.get('modulations') or [])[:5]:
        L.append(f"    {ts_str(m['time'])}: {m['from']} → {m['to']}  (dist={m['distance']} quintas)")

    # ══ 46. SUBTEXT EMOCIONAL ════════════════════════════════
    sx = subtext or {}
    L += ["", f"╔══ 46. SUBTEXT EMOCIONAL {chr(9552)*39}",
          f"  {sx.get('desc','')}",
          f"  Índice subtext: {sx.get('subtext_index',0):.2f}"]
    for ev in (sx.get('contradictions') or []):
        L.append(f"  ↔ {ts_str(ev['time'])} [{ev['type']}]")
        L.append(f"    Superficie: {ev['surface']}  /  Profundidad: {ev['depth']}")
        L.append(f"    → {ev['emotion']}")

    # ══ 47. MOMENTUM MUSICAL ═════════════════════════════════
    mo = momentum or {}
    L += ["", f"╔══ 47. MOMENTUM MUSICAL {chr(9552)*41}",
          f"  {mo.get('desc','')}",
          f"  Pico      : {ts_str(mo.get('peak_time',0))}  ({mo.get('peak_momentum',0):+.3f})",
          f"  Mínimo    : {ts_str(mo.get('trough_time',0))}  ({mo.get('trough_momentum',0):+.3f})",
          f"  Reversiones: {mo.get('reversals',0)}  |  Avance: {mo.get('pos_duration',0):.0f}s  Retroceso: {mo.get('neg_duration',0):.0f}s", ""]
    mc = mo.get('curve', [])
    if mc:
        mchart = momentum_ascii(mc, width=56, height=4)
        if mchart:
            L.append("  Momentum (arriba=avanza, abajo=retrocede):")
            for ln in mchart.split('\n'): L.append('  ' + ln)

    # ══ 48. FATIGA AUDITIVA ══════════════════════════════════
    af = auditory_fatigue or {}
    L += ["", f"╔══ 48. FATIGA AUDITIVA {chr(9552)*42}",
          f"  {af.get('desc','')}",
          f"  Índice fatiga: {af.get('fatigue_index',0)*100:.0f}%"]
    for ev in (af.get('contrast_events') or []):
        L.append(f"  Restauración en {ts_str(ev['time'])}: +{ev['impact']:.2f}")

    # ══ 49. POLIRRITMIA Y HEMIOLA ════════════════════════════
    pr = polyrhythm or {}
    L += ["", f"╔══ 49. POLIRRITMIA Y HEMIOLA {chr(9552)*35}",
          f"  {pr.get('desc','')}",
          f"  Polirritmia: {'sí' if pr.get('polyrhythm') else 'no'}  |  Hemiola: {'sí' if pr.get('hemiola') else 'no'}"]
    if pr.get('treble_pulse') and pr.get('bass_pulse'):
        L.append(f"  Pulso agudo: {pr['treble_pulse']*1000:.0f}ms  |  Pulso grave: {pr['bass_pulse']*1000:.0f}ms")
    for t_ev in (pr.get('hemiola_events') or [])[:3]:
        L.append(f"  Hemiola en {t_ev}")

    # ══ 50. ESTRUCTURA FRACTAL ═══════════════════════════════
    fr = fractal or {}
    L += ["", f"╔══ 50. ESTRUCTURA FRACTAL {chr(9552)*38}",
          f"  {fr.get('desc','')}",
          f"  Micro↔Meso : {fr.get('micro_meso',0):.3f}",
          f"  Meso↔Macro : {fr.get('meso_macro',0):.3f}",
          f"  Micro↔Macro: {fr.get('micro_macro',0):.3f}",
          f"  Índice fractal: {fr.get('fractal_index',0):.3f}"]

    # ══ 51. PESO EMOCIONAL ACUMULADO ═════════════════════════
    ew = emotional_weight or {}
    L += ["", f"╔══ 51. PESO EMOCIONAL ACUMULADO {chr(9552)*32}",
          f"  {ew.get('desc','')}"]
    for ev in (ew.get('key_events') or []):
        L.append(f"  ◉ {ev[0]:10s} en {ts_str(ev[1])}: intensidad {ev[2]:.2f}")

    # ══ 52. PUNTO DE NO RETORNO ══════════════════════════════
    pn = pnr or {}
    L += ["", f"╔══ 52. PUNTO DE NO RETORNO {chr(9552)*37}",
          f"  {pn.get('desc','')}"]
    if pn.get('time') is not None:
        L.append(f"  Exactamente: {ts_str(pn['time'])}  ({pn.get('position',0)*100:.1f}% de la pieza)")

    # ══ 53. DENSIDAD NARRATIVA POR PULSO ═════════════════════
    nd = narrative_density or {}
    L += ["", f"╔══ 53. DENSIDAD NARRATIVA POR PULSO {chr(9552)*28}",
          f"  {nd.get('desc','')}",
          f"  Eventos/pulso  : {nd.get('events_per_beat',0):.3f}",
          f"  Densidad efect.: {nd.get('effective_density',0):.3f}  ({nd.get('style','')})"]

    # ══ 54. ESTADO EMOCIONAL ACUMULATIVO ═════════════════════
    cs = cumulative_state or {}
    fv, fa = cs.get('final_state', (0,0))
    L += ["", f"╔══ 54. ESTADO EMOCIONAL ACUMULATIVO {chr(9552)*27}",
          f"  {cs.get('desc','')}",
          f"  Tipo de viaje   : {cs.get('journey_type','')}",
          f"  Estado final    : V={fv:+.2f}  A={fa:+.2f}",
          f"  Estrés acum. máx: {cs.get('max_cumulative_stress',0):.2f}"]
    sts = cs.get('states', [])
    if sts:
        bar_p = ''.join(chr(9608) if s['cum_valence']>0.2 else (chr(9619) if s['cum_valence']>-0.2 else chr(9617)) for s in sts)
        L.append(f"  Arco acumulativo: {bar_p}")

    # ══ 55. COLOR ARMÓNICO ═══════════════════════════════════
    hc = harmonic_color or {}
    L += ["", f"╔══ 55. COLOR ARMÓNICO (Scriabin / Rimsky-Korsakov) {chr(9552)*13}",
          f"  {hc.get('desc','')}"]
    for p in (hc.get('palette') or [])[:6]:
        bw = int(p['weight']*36)
        bar = chr(9608)*bw + chr(9617)*(36-bw)
        L.append(f"  {p['note']:3s} {p['color']:20s} [{bar}] {p['time_pct']:.0f}%")
        L.append(f"     {p['emotion']}")

    # ══ 56. DENSIDAD ARMÓNICA POR REGISTRO ═══════════════════
    rh = register_harmony or {}
    L += ["", f"╔══ 56. DENSIDAD ARMÓNICA POR REGISTRO {chr(9552)*26}",
          f"  {rh.get('complexity_location','')}"]
    for rn, info in (rh.get('profiles') or {}).items():
        if info.get('note_count', 0) == 0: continue
        bd = min(int(info['density']*3), 20)
        L.append(f"  {rn:14s}: {info['note_count']:4d} notas  dis={info['dissonance']:.2f}  vel={info['avg_velocity']:.0f}  {chr(9608)*bd+chr(9617)*(20-bd)}")

    # ══ 57. PITCH BEND Y MICROTONALISMO ═════════════════════
    pb2 = pitch_bend or {}
    L += ["", f"╔══ 57. PITCH BEND Y MICROTONALISMO {chr(9552)*29}",
          f"  {pb2.get('desc','')}",
          f"  Vibrato   : {'sí' if pb2.get('vibrato') else 'no'}  |  Glissando: {'sí' if pb2.get('glissando') else 'no'}",
          f"  Expresividad: {pb2.get('expressiveness',0):.2f}"]

    # ══ v7 SECTIONS ═════════════════════════════════════════

    # ══ 59. MARCADORES CULTURALES ════════════════════════════
    cu = cultural or {}
    L += ["", "╔══ 59. MARCADORES CULTURALES Y SEMÁNTICA " + "═"*22,
          f"  {cu.get('desc','')}",
          f"  Estilo dominante: {cu.get('dominant_style','')}"]
    for m in (cu.get('markers') or [])[:5]:
        L.append(f"  {m['affinity']*100:.0f}%  {m['style']}")
        L.append(f"       → {m['emotion']}")

    # ══ 60. VOZ CONDUCTORA DEL DISCURSO ══════════════════════
    nv = narrative_voice or {}
    L += ["", "╔══ 60. VOZ CONDUCTORA DEL DISCURSO " + "═"*28,
          f"  {nv.get('desc','')}",
          f"  Cambios de protagonista: {nv.get('protagonist_shifts',0)}"]
    for entry in (nv.get('voice_map') or []):
        L.append(f"  Sec.{entry['section']+1} ({ts_str(entry['start'])}): voz {entry['register']} lidera (score={entry['score']:.2f})")

    # ══ 61. MICRO-TIMING Y HUMANIZACIÓN ══════════════════════
    mt = micro_timing or {}
    L += ["", "╔══ 61. MICRO-TIMING Y HUMANIZACIÓN " + "═"*28,
          f"  {mt.get('desc','')}",
          f"  Humanización     : {mt.get('humanization',0):.2f}/1.0",
          f"  Desviación media : {mt.get('mean_deviation_ms',0):.1f}ms  (std={mt.get('std_deviation_ms',0):.1f}ms)",
          f"  Drag/anticipación: {mt.get('drag_ratio',0)*100:.0f}% / {mt.get('anticipation_ratio',0)*100:.0f}%"]

    # ══ 62. CONVERGENCIA DE PARÁMETROS ═══════════════════════
    pc3 = param_convergence or {}
    L += ["", "╔══ 62. CONVERGENCIA DE PARÁMETROS " + "═"*29,
          f"  {pc3.get('desc','')}",
          f"  Convergencia media: {pc3.get('mean_convergence',0):.2f}"]
    for pk in (pc3.get('peaks') or [])[:5]:
        L.append(f"  ◉ {ts_str(pk['time'])} — {pk['direction']} ({pk['convergence']:.2f})")

    # ══ 63. ZONA DE CONFORT TONAL ════════════════════════════
    cz = comfort_zones or {}
    L += ["", "╔══ 63. ZONA DE CONFORT TONAL " + "═"*34,
          f"  {cz.get('desc','')}",
          f"  Confort medio: {cz.get('mean_comfort',0)*100:.0f}%  |  Zonas aventura: {len(cz.get('adventure_zones',[]))}"]
    scs = cz.get('section_comfort',[])
    if scs:
        bar = ''.join('█' if s['comfort']>0.8 else ('▓' if s['comfort']>0.6 else '░') for s in scs)
        L.append(f"  Arco diatónico: {bar}  (█=confort  ▓=neutro  ░=aventura)")

    # ══ 64. JERARQUÍA MÉTRICA ════════════════════════════════
    mh = metric_hierarchy or {}
    L += ["", "╔══ 64. JERARQUÍA MÉTRICA (Lerdahl-Jackendoff) " + "═"*17,
          f"  {mh.get('desc','')}",
          f"  Refuerzo métrico : {mh.get('metric_reinforcement',0):.2f}/1.0",
          f"  Síncopa estructural: {mh.get('syncopation_score',0)*100:.0f}%"]
    wd = mh.get('weight_distribution',{})
    wlabels = {4:'Tiempo 1 (downbeat)',3:'Tiempo fuerte',2:'Tiempo débil',1:'Contratiempo',0:'Nota de paso'}
    total_n = mh.get('total_notes',1)
    for w in sorted(wd.keys(),reverse=True):
        pct = wd[w]/total_n*100
        bar = '█'*int(pct/3)+'░'*(33-int(pct/3))
        L.append(f"  {wlabels.get(w,'?'):22s} [{bar}] {pct:.0f}%")

    # ══ 65. CADENCIAS ELÍPTICAS ══════════════════════════════
    el = elliptical or {}
    L += ["", "╔══ 65. CADENCIAS ELÍPTICAS " + "═"*36,
          f"  {el.get('desc','')}",
          f"  Total detectadas: {el.get('count',0)}"]
    for e in (el.get('ellipses') or [])[:5]:
        L.append(f"  {ts_str(e['time'])}: {e['type']} — {e['from']}→{e['to']}")
        L.append(f"    → {e['emotion']}")

    # ══ 66. RECURRENCIA TEMÁTICA CON DISTANCIA ═══════════════
    tr = thematic_recurrence or {}
    L += ["", "╔══ 66. RECURRENCIA TEMÁTICA CON DISTANCIA " + "═"*20,
          f"  {tr.get('desc','')}",
          f"  Distancia media: {tr.get('mean_distance',0):.1f}s  |  Reconocimiento: {tr.get('recognition_effect',0):.2f}"]
    for r in (tr.get('recurrences') or []):
        times_str = ', '.join(ts_str(t) for t in r['times'][:5])
        L.append(f"  {r['motif'][:45]}")
        L.append(f"    ×{r['appearances']}  gap_μ={r['mean_gap']:.1f}s  gap_max={r['max_gap']:.1f}s")
        L.append(f"    Apariciones: {times_str}")

    # ══ 67. LISTENER MODELING ════════════════════════════════
    lm = listener_model or {}
    L += ["", "╔══ 67. EXPERIENCIA DEL OYENTE PROYECTADA " + "═"*22,
          f"  {lm.get('desc','')}",
          f"  Pico experiencia: {ts_str(lm.get('peak_time',0))}  ({lm.get('peak_experience',0):.2f})",
          f"  Experiencia media: {lm.get('mean_experience',0):.2f}  |  Agotamiento: {lm.get('exhaustion_index',0):.2f}", ""]
    lc = lm.get('curve',[])
    if lc:
        lchart = listener_ascii(lc, width=56, height=5)
        if lchart:
            L.append("  Intensidad de experiencia subjetiva proyectada:")
            for ln in lchart.split('\n'): L.append('  ' + ln)

    # ══ 68. DENSIDAD SEMÁNTICA ARMÓNICA ══════════════════════
    hs = harmonic_semantic or {}
    L += ["", "╔══ 68. DENSIDAD SEMÁNTICA ARMÓNICA " + "═"*28,
          f"  {hs.get('desc','')}",
          f"  Novedad media: {hs.get('mean_novelty',0):.2f}"]
    for pk in (hs.get('high_novelty_peaks') or [])[:4]:
        L.append(f"  ◉ {ts_str(pk['time'])}: {pk['chord']} (novedad={pk['novelty']:.2f})")

    # ══ 69. FUNCIÓN NARRATIVA DEL BAJO ═══════════════════════
    bn = bass_narrative or {}
    L += ["", "╔══ 69. FUNCIÓN NARRATIVA DEL BAJO " + "═"*29,
          f"  {bn.get('desc','')}",
          f"  Función: {bn.get('function','')}  |  Nota dominante: {bn.get('dominant_pc','')} ({bn.get('dom_ratio',0)*100:.0f}%)",
          f"  Rango: {bn.get('pitch_range',0)}st  |  Intervalo μ: {bn.get('mean_interval',0):.1f}st  |  Regularidad: {bn.get('regularity',0):.2f}"]

    # ══ 70. SINCRONÍA ENTRE CAPAS ════════════════════════════
    ls = layer_sync or {}
    L += ["", "╔══ 70. SINCRONÍA ENTRE CAPAS " + "═"*34,
          f"  {ls.get('desc','')}",
          f"  Sincronía media: {ls.get('mean_synchrony',0):.2f}"]
    for pk in (ls.get('sync_peaks') or [])[:4]:
        L.append(f"  ◉ {ts_str(pk['time'])}: sincronía total ({', '.join(pk['layers'])})")

    # ══ v9. ANÁLISIS EMOCIONAL PROFUNDO ══════════════════════

    # ══ 72. VALENCIA DINÁMICA CONTINUA ═══════════════════════
    dv = dynamic_valence or {}
    L += ["", "╔══ 72. VALENCIA EMOCIONAL DINÁMICA (continua) " + "═"*17]
    L.append(f"  {dv.get('desc','')}")
    L.append(f"  Valencia media : {dv.get('mean_valence',0):+.3f}  |  "
             f"Volatilidad: {dv.get('volatility',0):.3f}  |  "
             f"Arco: {dv.get('arc_shape','?')}")
    if dv.get('peaks_light'):
        L.append(f"  Picos luminosos:")
        for p in dv['peaks_light'][:3]:
            L.append(f"    ☀  {ts_str(p['time'])}  val={p['valence']:+.2f}")
    if dv.get('peaks_dark'):
        L.append(f"  Picos oscuros:")
        for p in dv['peaks_dark'][:3]:
            L.append(f"    ☾  {ts_str(p['time'])}  val={p['valence']:+.2f}")
    if dv.get('transitions'):
        L.append(f"  Transiciones bruscas de valencia:")
        for tr in dv['transitions'][:4]:
            arrow = '↑' if tr['delta'] > 0 else '↓'
            L.append(f"    {arrow} {ts_str(tr['time'])}  {tr['direction']}  Δ={tr['delta']:+.2f}  ({tr['from']:+.2f}→{tr['to']:+.2f})")
    # ASCII curve of valence
    if dv.get('curve'):
        vc = dv['curve']
        width = 56
        step = max(1, len(vc) // width)
        sampled = [vc[i][1] for i in range(0, len(vc), step)][:width]
        height = 6
        L.append("  Curva de valencia (+ luminoso / - oscuro):")
        for row in range(height, -height - 1, -1):
            threshold = row / height
            line = ''
            for v2 in sampled:
                if threshold >= 0:
                    line += '█' if v2 >= threshold else ' '
                else:
                    line += '▄' if v2 <= threshold else ' '
            label = f"{threshold:+.1f}" if row in (height, 0, -height) else '     '
            L.append(f"  {label} │{line}│")
        L.append(f"        └{'─'*len(sampled)}┘")

    # ══ 73. ANÁLISIS DE CATARSIS ════════════════════════════
    cat = catharsis or {}
    L += ["", "╔══ 73. ANÁLISIS DE CATARSIS " + "═"*37]
    L.append(f"  Tipo: {cat.get('catharsis_type','ausente').upper()}  |  "
             f"Fuerza: {cat.get('release_strength',0):.2f}  |  "
             f"Resolución: {cat.get('resolution_quality',0):.2f}")
    if cat.get('present'):
        L.append(f"  Acumulación: {ts_str(cat.get('buildup_start') or 0)} "
                 f"({cat.get('buildup_duration',0):.0f}s)  →  "
                 f"Liberación: {ts_str(cat.get('moment') or 0)}")
        if len(cat.get('waves', [])) > 1:
            L.append(f"  Oleadas: {len(cat['waves'])}")
    L.append("  → Lectura narrativa en sección VIII del ensayo.")

    # ══ 74. MAPA DE AMBIVALENCIA EMOCIONAL ══════════════════
    amb = emotional_ambivalence or {}
    L += ["", "╔══ 74. MAPA DE AMBIVALENCIA EMOCIONAL " + "═"*26]
    L.append(f"  Índice medio: {amb.get('mean_ambivalence',0):.3f}/1.0  |  "
             f"Zonas activas: {len(amb.get('zones',[]))}")
    if amb.get('peak_moments'):
        for pm in amb['peak_moments'][:3]:
            L.append(f"  ⊗ {ts_str(pm['time'])}  [{pm['ambivalence']:.2f}]  {pm['desc']}")
    L.append("  → Lectura narrativa en sección VIII del ensayo.")

    # ══ 75. TRAYECTORIA EMOCIONAL VA ════════════════════════
    traj = emotional_trajectory or {}
    L += ["", "╔══ 75. TRAYECTORIA EN ESPACIO VALENCE-AROUSAL " + "═"*17]
    cx, cy = traj.get('centroid', (0.0, 0.0))
    L.append(f"  Forma: {traj.get('path_shape','?')}  |  "
             f"Distancia: {traj.get('total_distance',0):.3f}  |  "
             f"Eficiencia: {traj.get('efficiency',0):.2f}  |  "
             f"Centroide: ({cx:+.2f}, {cy:+.2f})")
    L.append(f"  Cuadrante dom.: {traj.get('dominant_quadrant','?')}  |  "
             f"Cruces: {traj.get('crossings',0)}")
    qt = traj.get('quadrant_time', {})
    if qt:
        for quad, dur in sorted(qt.items(), key=lambda x: -x[1]):
            pct = dur / total_dur * 100 if total_dur > 0 else 0
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            L.append(f"  [{bar}] {pct:4.0f}%  {quad}")

    # ══ 76. ELECCIONES ANTI-CONVENCIONALES ══════════════════
    ac = anti_conventional or {}
    L += ["", "╔══ 76. ELECCIONES ANTI-CONVENCIONALES " + "═"*26]
    L.append(f"  Índice originalidad: {ac.get('deviation_score',0):.2f}/1.0")
    if ac.get('signature_choices'):
        L.append(f"  Firma: {' | '.join(ac['signature_choices'][:3])}")
    devs = ac.get('deviations', [])
    if devs:
        L.append(f"  Desviaciones ({len(devs)}):")
        for d in devs[:4]:
            L.append(f"    [{d['type']}] peso={d['weight']:.2f}")
    L.append("  → Lectura narrativa en sección IX del ensayo.")

    # ══ 77. DENSIDAD INFORMACIONAL MULTINIVEL ═══════════════
    mld = multilevel_density or {}
    L += ["", "╔══ 77. DENSIDAD INFORMACIONAL MULTINIVEL " + "═"*23]
    L.append(f"  Micro: {mld.get('micro_entropy',0):.3f}b  |  "
             f"Meso: {mld.get('meso_entropy',0):.3f}b  |  "
             f"Macro: {mld.get('macro_entropy',0):.3f}b  |  "
             f"Economía: {mld.get('composer_economy',0):.2f}")
    L.append(f"  Perfil: {mld.get('density_profile','?')}  |  "
             f"Contraste niveles: {mld.get('level_contrast',0):.3f}")
    nc = mld.get('novelty_curve', [])
    if nc:
        L.append("  Novedad/sección:")
        for item in nc:
            bw = int(item.get('novelty', 0) * 20)
            bar = '█' * bw + '░' * (20 - bw)
            L.append(f"    Sec.{item['section']} [{bar}] nov={item.get('novelty',0):.2f} H={item.get('entropy',0):.2f}b")

    # ══ 78. INTENCIÓN NARRATIVA DEL COMPOSITOR ══════════════
    ni = narrative_intention or {}
    L += ["", "╔══ 78. INTENCIÓN NARRATIVA DEL COMPOSITOR " + "═"*21]
    L.append(f"  Arquetipo: {ni.get('archetype','?')}  "
             f"(confianza: {ni.get('confidence',0):.0%})")
    if ni.get('alternative'):
        L.append(f"  Alternativo: {ni['alternative']}")
    if ni.get('evidence'):
        for ev in ni['evidence'][:3]:
            L.append(f"    ✓ {ev}")
    L.append("  → Lectura narrativa en sección IX del ensayo.")

    # ══ 79. ROUGHNESS PSICOACÚSTICO (Plomp-Levelt) ══════════
    rgh = roughness or {}
    L += ["", "╔══ 79. ASPEREZA SENSORIAL — ROUGHNESS (Plomp-Levelt) " + "═"*11]
    L.append(f"  {rgh.get('desc','')}")
    L.append(f"  Roughness media : {rgh.get('mean_roughness',0):.3f}  |  "
             f"Pico: {rgh.get('max_roughness',0):.3f}  |  "
             f"Arco: {rgh.get('roughness_arc','?')}")
    pk_r = rgh.get('peak_roughness', {})
    if pk_r.get('time') is not None:
        L.append(f"  Máxima aspereza : {ts_str(pk_r['time'])}  "
                 f"(val={pk_r.get('value',0):.3f})")
    reg_impact = rgh.get('register_impact', 0)
    if reg_impact > 0.05:
        L.append(f"  Impacto reg. grave: {reg_impact*100:.0f}% — "
                 f"el registro bajo amplifica la aspereza acústica")
    if rgh.get('valleys'):
        L.append(f"  Momentos de consonancia máxima:")
        for v2 in rgh['valleys'][:3]:
            L.append(f"    ◎ {ts_str(v2['time'])}  roughness={v2['value']:.3f}")
    # ASCII curve roughness
    rc = rgh.get('curve_norm', [])
    if rc:
        width = 56
        step = max(1, len(rc) // width)
        sampled_r = [rc[i][1] for i in range(0, len(rc), step)][:width]
        height_r = 5
        L.append("  Curva de aspereza sensorial (0=liso · 1=máxima aspereza):")
        for row in range(height_r, 0, -1):
            threshold = row / height_r
            line = ''
            for v2 in sampled_r:
                line += '█' if v2 >= threshold else (' ' if v2 < threshold - 0.5/height_r else '▄')
            label = f"{threshold:.1f}" if row in (height_r, height_r//2 + 1, 1) else '   '
            L.append(f"  {label} │{line}│")
        L.append(f"      └{'─'*len(sampled_r)}┘")

    # ══ 80. MARKOV MELÓDICO DE 2º ORDEN ════════════════════
    mm = melodic_markov or {}
    L += ["", "╔══ 80. MARKOV MELÓDICO (orden 2) — PREDECIBILIDAD " + "═"*14]
    L.append(f"  {mm.get('desc','')}")
    L.append(f"  Perfil de estilo : {mm.get('style_profile','?').upper()}")
    L.append(f"  P(transición) μ  : {mm.get('mean_probability',0):.3f}  |  "
             f"Entropía Markov: {mm.get('entropy_markov',0):.2f} bits/estado")
    L.append(f"  Estados únicos   : {mm.get('n_unique_states',0)}  |  "
             f"Transiciones totales: {mm.get('n_transitions',0)}")
    top_t = mm.get('top_transitions', [])
    if top_t:
        L.append(f"  Firma melódica (transiciones más frecuentes):")
        for tt in top_t[:4]:
            L.append(f"    ▸ {tt['desc']}  ×{tt['count']}")
    sing = mm.get('singular_gestures', [])
    if sing:
        L.append(f"  Gestos únicos (hapax melódicos): {len(sing)}")
    sc = mm.get('surprise_curve', [])
    if sc:
        width_m = 56
        step_m = max(1, len(sc) // width_m)
        sampled_m = [sc[i][1] for i in range(0, len(sc), step_m)][:width_m]
        L.append("  Curva de sorpresa melódica (alta = transición inesperada):")
        for row in range(5, 0, -1):
            threshold = row / 5
            line = ''.join('█' if v2 >= threshold else ' ' for v2 in sampled_m)
            label = f"{threshold:.1f}" if row in (5, 3, 1) else '   '
            L.append(f"  {label} │{line}│")
        L.append(f"      └{'─'*len(sampled_m)}┘")

    # ══ 81. ENRIQUECIMIENTO SEMÁNTICO (Affektenlehre) ══════
    se = semantic_enrichment or {}
    L += ["", "╔══ 81. ENRIQUECIMIENTO SEMÁNTICO (Affektenlehre) " + "═"*15]
    L.append(f"  Concepto: '{se.get('concept','?')}'  ajuste={se.get('fit_score',0):.2f}  "
             f"arco={se.get('arc','')}  persona={se.get('persona_name','?')}")
    alts = se.get('alternatives', [])
    if alts:
        L.append(f"  Alternativos: " +
                 "  |  ".join(f"'{a['concept']}' ({a['score']:.2f})" for a in alts))
    L.append("  → Lectura narrativa en sección IX del ensayo.")

    # ══ 82. DETECCIÓN DE GÉNERO MUSICAL ═════════════════════
    gd = genre_detection or {}
    L += ["", "╔══ 82. DETECCIÓN DE GÉNERO MUSICAL " + "═"*30]
    if gd.get('genre','') != 'indeterminado':
        conf_bar_w = int(gd.get('confidence', 0) * 30)
        conf_bar = '█' * conf_bar_w + '░' * (30 - conf_bar_w)
        L.append(f"  Género principal : {gd.get('genre_label','?')}")
        L.append(f"  Confianza        : [{conf_bar}] {gd.get('confidence',0):.0%}  "
                 f"({gd.get('confidence_label','')})")
        if gd.get('subgenre'):
            L.append(f"  Subgénero        : {gd['subgenre']}")
        if gd.get('regions'):
            L.append(f"  Origen cultural  : {gd['regions']}")
        L.append(f"  Firma del género : {gd.get('key_signature','')}")
        L.append("")
        # Top 3
        top3 = gd.get('top3', [])
        if len(top3) > 1:
            L.append(f"  Géneros más probables:")
            for i, g3 in enumerate(top3[:3]):
                bar_w = int(g3['score'] * 30)
                bar = '█' * bar_w + '░' * (30 - bar_w)
                marker = '◉' if i == 0 else '○'
                L.append(f"  {marker} [{bar}] {g3['score']:.0%}  {g3['label']}")
        # Evidencia
        ev = gd.get('evidence', [])
        if ev:
            L.append(f"")
            L.append(f"  Evidencia que sustenta la clasificación:")
            for e in ev[:4]:
                feat_name = e['feature'].replace('_', ' ')
                L.append(f"    ✓ {feat_name:<28} score={e['score']:.2f}  peso={e['weight']:.2f}  match={e['match']:.2f}")
        # Rasgos atípicos
        neg = gd.get('negative_evidence', [])
        if neg:
            L.append(f"  Rasgos atípicos para este género:")
            for n in neg[:3]:
                L.append(f"    ✗ {n.replace('_',' ')}")
        # Ambigüedad
        if gd.get('ambiguous') and len(top3) > 1:
            L.append(f"")
            L.append(f"  Clasificación ambigua: '{top3[0]['label']}' y '{top3[1]['label']}'")
            L.append(f"  comparten features similares — el material es estilísticamente híbrido.")
        L.append(f"")
        # Descripción del género
        desc_g = gd.get('description','')
        if desc_g:
            L.append(f"  Descripción del género:")
            for line in wrap(desc_g, width=62, indent='    ').split('\n'):
                L.append(line)
        if gd.get('subgenres_list'):
            L.append(f"  Subgéneros posibles: {gd['subgenres_list']}")
    else:
        L.append("  No se pudo determinar el género — material atípico o insuficiente.")

    # ══ 83. MAPA EMOCIONAL UNIFICADO ════════════════════════
    if unified_emotional_map and unified_emotional_map.get('slots'):
        L.append("")
        L.append(_format_unified_map_report(unified_emotional_map, total_dur))

    # ══ 58. SÍNTESIS TÉCNICA FINAL ═══════════════════════════
    L += ["", f"╔══ 58. SÍNTESIS TÉCNICA FINAL {'═'*39}"]
    L.append(build_narrative(
        notes, sections, key_name, key_root, mode, avg_bpm,
        climax, contour, total_dur, overall_tension,
        centricity, interval_anal, texture, motifs,
        canon_progs, modal_borrow, rhythm, cadences,
        voice_leading, expectation, tension_curve,
        musical_form, pedal_points, harmonic_graph,
        register_analysis, mode_emotion
    ))

    L += ["", SEP, ""]
    return '\n'.join(L)

def build_narrative(notes, sections, key_name, key_root, mode, avg_bpm,
                    climax, contour, total_dur, overall_tension,
                    centricity, ival, texture, motifs,
                    canon_progs, modal_borrow, rhythm, cadences,
                    voice_leading, expectation, tension_curve,
                    musical_form, pedal_points, harmonic_graph,
                    register_analysis, mode_emotion) -> str:

    tc = tempo_cat(avg_bpm)
    em_map = {
        ('major','slow'):'contemplación nostálgica', ('major','medium'):'calidez esperanzadora',
        ('major','fast'):'alegría y vitalidad',       ('minor','slow'):'tristeza e introspección',
        ('minor','medium'):'melancolía y añoranza',   ('minor','fast'):'urgencia y drama oscuro',
        ('dorian','medium'):'melancolía activa',      ('phrygian','medium'):'fatalismo y misterio',
        ('lydian','medium'):'ensueño y lo sublime',   ('mixolydian','medium'):'épica y ambigüedad',
    }
    ge = em_map.get((mode, tc), em_map.get(('minor' if 'min' in mode else 'major', tc), 'emoción compleja'))

    active = [s for s in sections if s.notes]
    iv_vel = active[0].avg_velocity if active else 64
    ev_vel = active[-1].avg_velocity if active else 64
    arc = ('La pieza culmina con mayor energía, afirmando su discurso sin reservas.' if ev_vel > iv_vel+15
           else 'La pieza se desvanece hacia el silencio, como una pregunta que queda suspendida.' if ev_vel < iv_vel-15
           else 'La pieza regresa a su punto de partida dinámico, cerrando un arco circular y completo.')

    cs = sections[climax] if climax < len(sections) else None
    ct = ts_str(cs.start) if cs else '?'

    mq = ('de gran cantabilidad y naturalidad vocal' if ival and ival.get('conjunct',0)>0.65
          else 'angular y expresiva, con saltos amplios' if ival and ival.get('leaps',0)>0.2
          else 'de equilibrio entre grados conjuntos y saltos')

    # Tension arc description
    t_arc = describe_tension_arc(tension_curve) if tension_curve else 'indeterminado'

    # Cadence personality
    cad_dom = cadences.get('dominant_type','')
    if cad_dom == 'perfecta':
        cad_note = "Las cadencias perfectas frecuentes confieren certeza y resolución a la narrativa."
    elif cad_dom == 'rota':
        cad_note = "La abundancia de cadencias rotas genera constantes giros emocionales inesperados, frustrando la expectativa del oyente."
    elif cad_dom == 'imperfecta':
        cad_note = "El predominio de cadencias imperfectas mantiene la pieza en perpetuo movimiento, sin cerrarse del todo."
    elif cad_dom == 'plagal':
        cad_note = "Las cadencias plagales imprimen a la pieza una cualidad espiritual y de reposo sereno."
    else:
        cad_note = ""

    motif_note = (f"Se articulan {len(motifs)} motivo(s) recurrente(s) que actúan como firma emocional y garantizan cohesión interna." if motifs else "")
    prog_note = (f"Armónicamente, se reconocen progresiones de {', '.join(canon_progs[:2])}." if canon_progs else "")
    borrow_note = (f"Los préstamos modales ({', '.join(modal_borrow[:2])}) añaden color y ambigüedad al discurso." if modal_borrow else "")
    tension_note = (" Alta tensión armónica sostenida genera inestabilidad y búsqueda." if overall_tension>0.5
                    else " Armonía consonante aporta serenidad o resignación." if overall_tension<0.2 else "")
    centricity_note = (" La pieza evita sistemáticamente la resolución en la tónica — suspensión permanente." if centricity<0.15
                       else " El frecuente retorno a la tónica da carácter afirmativo." if centricity>0.5 else "")
    rhythm_note = (" El ritmo sincopado inyecta nervio que contrasta con el lirismo melódico." if rhythm.get('syncopation',0)>0.35
                   else " El ostinato rítmico actúa como pulso hipnótico." if rhythm.get('variety',0)<0.3 else "")
    vl_note = (" El voice leading fluido disuelve las transiciones armónicas con naturalidad." if voice_leading.get('smoothness',0)>0.7
               else " Los saltos en el voice leading marcan rupturas expresivas deliberadas." if voice_leading.get('smoothness',0)<0.3 else "")
    pedal_note = (" El pedal de dominante crea una tensión que reclama resolución urgente." if any(p['type']=='dominante' for p in pedal_points)
                  else " El pedal de tónica sostiene la estabilidad bajo la tensión armónica." if any(p['type']=='tónica' for p in pedal_points) else "")
    exp_note = (" La alta imprevisibilidad melódica intensifica la respuesta emocional del oyente." if expectation.get('mean_surprise',0)>0.35
                else " La melodía predecible invita a la familiaridad y el confort." if expectation.get('mean_surprise',0)<0.15 else "")
    form_note = f"La forma {musical_form.get('form','?')} estructura el discurso en {musical_form.get('pattern','?')}."

    lines = [
        f"  Composición en {key_name}, a {avg_bpm:.0f} BPM ({classify_tempo(avg_bpm).lower()}).",
        f"  Construye un discurso de {ge}, impregnado de {mode_emotion}.",
        f"",
        f"  {form_note}",
        f"  Contorno melódico {contour}. Melodía {mq}.",
        f"  El arco de tensión es {t_arc}.",
        f"  El punto de máxima intensidad se sitúa en {ct}.",
        f"",
        f"  {cad_note}",
        f"  {motif_note} {prog_note} {borrow_note}",
        f" {tension_note}{centricity_note}{rhythm_note}",
        f" {vl_note}{pedal_note}{exp_note}",
        f"",
        f"  {arc}",
        f"",
        f"  {'Belleza melancólica' if 'min' in mode else 'Luminosidad expresiva'} sostenida en",
        f"  {int(total_dur//60)}m{int(total_dur%60):02d}s de "
        f"{'introspección y escucha íntima.' if avg_bpm<90 else 'energía y narrativa emocional profunda.'}",
    ]
    return '\n'.join(lines)

SECTION_NAMES_MAIN = ['Introducción','Desarrollo A','Desarrollo B','Clímax',
                      'Recapitulación','Coda','Epílogo','Cierre Final']




# ══ NEW ANALYSES v4 ══


# ═══════════════════════════════════════════════════════════════
#  DATOS COMPARTIDOS (copiados para prueba standalone)
# ═══════════════════════════════════════════════════════════════

DISSONANCE = {0:0.00,1:1.00,2:0.80,3:0.25,4:0.20,5:0.10,
              6:0.95,7:0.00,8:0.20,9:0.10,10:0.45,11:0.85}
SCALE_INTERVALS = {
    'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],
    'dorian':[0,2,3,5,7,9,10],'phrygian':[0,1,3,5,7,8,10],
    'lydian':[0,2,4,6,7,9,11],'mixolydian':[0,2,4,5,7,9,10],
    'locrian':[0,1,3,5,6,8,10],
}

# ───────────────────────────────────────────────────────────────
#  ARQUETIPOS EMOCIONALES DE REFERENCIA
#  Cada arquetipo es un vector de 12 dimensiones de características
#  normalizadas: [modo_bin, bpm_norm, tension, centricity,
#   resolution, conjunct, leap, syncopation, voice_count,
#   spacing_norm, valence, arousal]
# ───────────────────────────────────────────────────────────────

ARCHETYPES = {
    'Lamento/Elegy':        {'mode':'minor','bpm':(40,70), 'tension':(0.2,0.5),'centricity':(0.0,0.3),'syncopation':(0.0,0.2),'valence':(-0.8,-0.3),'arousal':(-0.6,0.0),'desc':'tristeza profunda, duelo, despedida'},
    'Himno/Anthem':         {'mode':'major','bpm':(80,120),'tension':(0.1,0.3),'centricity':(0.4,1.0),'syncopation':(0.0,0.2),'valence':(0.4,1.0), 'arousal':(0.0,0.6), 'desc':'triunfo colectivo, pertenencia, gloria'},
    'Danza/Dance':          {'mode':'major','bpm':(120,180),'tension':(0.1,0.3),'centricity':(0.3,0.8),'syncopation':(0.2,0.6),'valence':(0.3,0.9), 'arousal':(0.5,1.0), 'desc':'alegría física, celebración, movimiento'},
    'Contemplación':        {'mode':'minor','bpm':(50,90), 'tension':(0.1,0.3),'centricity':(0.1,0.4),'syncopation':(0.0,0.15),'valence':(-0.4,0.1),'arousal':(-0.8,-0.2),'desc':'reflexión íntima, quietud, meditación'},
    'Drama/Tragedy':        {'mode':'minor','bpm':(60,110),'tension':(0.4,0.8),'centricity':(0.0,0.2),'syncopation':(0.1,0.4),'valence':(-0.9,-0.4),'arousal':(0.2,0.8), 'desc':'conflicto, sufrimiento, catarsis'},
    'Éxtasis/Rapture':      {'mode':'major','bpm':(100,160),'tension':(0.3,0.6),'centricity':(0.2,0.5),'syncopation':(0.1,0.4),'valence':(0.5,1.0), 'arousal':(0.6,1.0), 'desc':'arrebato, trascendencia, lo sublime'},
    'Lullaby/Nana':         {'mode':'major','bpm':(50,80), 'tension':(0.0,0.2),'centricity':(0.5,1.0),'syncopation':(0.0,0.1),'valence':(0.3,0.7), 'arousal':(-0.9,-0.4),'desc':'calma, ternura, protección, sueño'},
    'Marcha/March':         {'mode':'major','bpm':(90,130),'tension':(0.1,0.3),'centricity':(0.4,0.9),'syncopation':(0.0,0.2),'valence':(0.2,0.7), 'arousal':(0.3,0.7), 'desc':'determinación, avance, disciplina colectiva'},
    'Nostalgia/Yearning':   {'mode':'minor','bpm':(60,100),'tension':(0.2,0.4),'centricity':(0.1,0.4),'syncopation':(0.1,0.3),'valence':(-0.5,0.0),'arousal':(-0.2,0.4), 'desc':'añoranza, lo que fue, melancolía activa'},
    'Misterio/Mystery':     {'mode':'minor','bpm':(50,90), 'tension':(0.4,0.7),'centricity':(0.0,0.2),'syncopation':(0.0,0.2),'valence':(-0.6,-0.1),'arousal':(0.0,0.5), 'desc':'ambigüedad, lo desconocido, suspense'},
    'Rabia/Fury':           {'mode':'minor','bpm':(130,200),'tension':(0.5,0.9),'centricity':(0.0,0.3),'syncopation':(0.3,0.7),'valence':(-0.9,-0.5),'arousal':(0.7,1.0), 'desc':'ira, urgencia violenta, energía destructiva'},
    'Esperanza/Hope':       {'mode':'dorian','bpm':(70,110),'tension':(0.2,0.4),'centricity':(0.1,0.4),'syncopation':(0.1,0.3),'valence':(-0.1,0.4),'arousal':(0.0,0.5), 'desc':'ambivalencia luminosa, anhelo con futuro'},
    'Ensueño/Dream':        {'mode':'lydian','bpm':(50,90), 'tension':(0.2,0.5),'centricity':(0.0,0.3),'syncopation':(0.0,0.2),'valence':(0.2,0.7), 'arousal':(-0.5,0.2), 'desc':'flotación, irrealidad, lo onírico'},
    'Épica/Epic':           {'mode':'mixolydian','bpm':(80,130),'tension':(0.3,0.6),'centricity':(0.2,0.5),'syncopation':(0.0,0.3),'valence':(0.1,0.6),'arousal':(0.4,0.8),'desc':'grandiosidad, aventura, lo heroico ambiguo'},
}

# ts_str definida en el módulo superior (línea 326) — no redefinir aquí.


# ═══════════════════════════════════════════════════════════════
#  1. ANÁLISIS DE SILENCIO Y RESPIRACIÓN
# ═══════════════════════════════════════════════════════════════

def analyze_silences(notes, total_dur: float) -> Dict:
    """
    Detecta y clasifica todos los silencios de la pieza.
    Tipos:
      - Respiración frasal  (<0.3s): puntuación melódica
      - Pausa expresiva     (0.3–1.5s): énfasis, suspenso
      - Silencio dramático  (1.5–4s): ruptura, impacto
      - Silencio estructural (>4s): separación de secciones
    También calcula:
      - Ratio silencio/sonido (densidad de aire)
      - Distribución temporal (¿dónde respira la pieza?)
      - Silencio pre-clímax (el más expresivo)
    """
    if not notes:
        return {'silences': [], 'ratio': 0, 'desc': 'sin datos'}

    sorted_notes = sorted(notes, key=lambda n: n.time_sec)

    # Find gaps between note endings and next note start
    silences = []
    for i in range(len(sorted_notes) - 1):
        end_of_current = sorted_notes[i].time_sec + sorted_notes[i].duration
        start_of_next = sorted_notes[i + 1].time_sec
        gap = start_of_next - end_of_current

        if gap > 0.15:  # ignore micro-gaps (legato notation artifacts)
            # Classify
            if gap < 0.3:
                stype = 'respiración'
                weight = 0.1
            elif gap < 1.5:
                stype = 'pausa expresiva'
                weight = 0.4
            elif gap < 4.0:
                stype = 'silencio dramático'
                weight = 0.8
            else:
                stype = 'silencio estructural'
                weight = 1.0

            silences.append({
                'start': end_of_current,
                'duration': gap,
                'type': stype,
                'weight': weight,
                'position': end_of_current / total_dur  # 0-1 normalised
            })

    if not silences:
        return {'silences': [], 'ratio': 0.0,
                'total_silence': 0, 'breathing_rate': 0,
                'most_dramatic': None, 'desc': 'escritura ligada sin silencios notables'}

    total_silence = sum(s['duration'] for s in silences)
    ratio = total_silence / total_dur

    # Find most dramatic silence (longest weighted by position — late silences hit harder)
    dramatic = [s for s in silences if s['type'] in ('silencio dramático', 'silencio estructural')]
    most_dramatic = max(dramatic, key=lambda s: s['duration'] * (0.5 + s['position'])) if dramatic else None

    # Breathing rate (breaths per minute)
    breaths = [s for s in silences if s['type'] == 'respiración']
    breath_rate = len(breaths) / (total_dur / 60) if total_dur > 0 else 0

    # Silence distribution: early/middle/late
    early  = sum(s['duration'] for s in silences if s['position'] < 0.33)
    middle = sum(s['duration'] for s in silences if 0.33 <= s['position'] < 0.66)
    late   = sum(s['duration'] for s in silences if s['position'] >= 0.66)

    # Characterise
    parts = []
    if ratio < 0.05:
        parts.append("escritura muy densa, casi sin aire — sofocante, urgente, sin tregua")
    elif ratio < 0.15:
        parts.append("respiración ajustada — discurso continuo con pausas precisas")
    elif ratio < 0.30:
        parts.append("respiración natural — la música tiene espacio para resonar")
    else:
        parts.append("abundante espacio y silencio — contemplativa, minimalista, el silencio como voz")

    if most_dramatic:
        parts.append(f"silencio más dramático en {ts_str(most_dramatic['start'])} ({most_dramatic['duration']:.1f}s)")

    if late > early * 1.5:
        parts.append("los silencios se acumulan al final — la pieza necesita respirar antes del cierre")
    elif early > late * 1.5:
        parts.append("los silencios dominan el inicio — apertura contemplativa antes de lanzarse")

    by_type = collections.Counter(s['type'] for s in silences)

    return {
        'silences': silences,
        'total_silence': total_silence,
        'ratio': ratio,
        'breath_rate': breath_rate,
        'most_dramatic': most_dramatic,
        'by_type': by_type,
        'distribution': {'early': early, 'middle': middle, 'late': late},
        'desc': '; '.join(parts)
    }


# ═══════════════════════════════════════════════════════════════
#  2. PERFIL DE ENERGÍA CONTINUA
# ═══════════════════════════════════════════════════════════════

def compute_energy_profile(notes, total_dur: float, resolution: float = 0.5) -> Dict:
    """
    Calcula la energía física (masa sonora) en cada instante.
    Energía = sum(velocity × duration_overlap) por ventana.
    Más rico que la dinámica MIDI porque captura densidad × intensidad.
    """
    if not notes:
        return {'curve': [], 'peak_time': 0, 'mean': 0, 'variance': 0, 'desc': ''}

    curve = []
    t = 0.0
    while t < total_dur:
        active = [n for n in notes if n.time_sec < t + resolution and n.time_sec + n.duration > t]
        if active:
            # Overlap fraction × velocity
            energy = sum(
                (min(n.time_sec + n.duration, t + resolution) - max(n.time_sec, t)) / resolution
                * (n.velocity / 127.0)
                for n in active
            )
        else:
            energy = 0.0
        curve.append((t, energy))
        t += resolution

    values = [e for _, e in curve]
    if not values:
        return {'curve': curve, 'peak_time': 0, 'mean': 0, 'variance': 0, 'desc': ''}

    mean_e    = sum(values) / len(values)
    variance  = sum((v - mean_e)**2 for v in values) / len(values)
    peak_idx  = values.index(max(values))
    peak_time = curve[peak_idx][0]
    peak_val  = values[peak_idx]

    # RMS energy
    rms = math.sqrt(sum(v**2 for v in values) / len(values))

    # Dynamic range
    dyn_range = max(values) - min(v for v in values if v > 0.01) if any(v > 0.01 for v in values) else 0

    # Shape description
    thirds = len(values) // 3
    e1 = sum(values[:thirds]) / thirds if thirds else 0
    e2 = sum(values[thirds:2*thirds]) / thirds if thirds else 0
    e3 = sum(values[2*thirds:]) / max(1, len(values) - 2*thirds)

    if e2 > e1 + 0.3 and e2 > e3 + 0.3:
        shape = "energía en arco — construcción central y liberación"
    elif e3 > e1 + 0.3:
        shape = "energía acumulativa — la pieza explota hacia el final"
    elif e1 > e3 + 0.3:
        shape = "energía decreciente — impacto inicial, luego repliegue"
    elif variance < 0.05:
        shape = "energía uniforme — estabilidad hipnótica o mecánica"
    else:
        shape = "energía ondulante — alternancia dinámica expresiva"

    parts = [shape]
    if dyn_range > 1.5:
        parts.append(f"amplio rango dinámico ({dyn_range:.1f}) — contraste expresivo intenso")
    elif dyn_range < 0.5:
        parts.append("rango dinámico estrecho — escritura muy uniforme")
    parts.append(f"pico de energía en {ts_str(peak_time)}")

    return {
        'curve': curve,
        'peak_time': peak_time,
        'peak_val': peak_val,
        'mean': mean_e,
        'rms': rms,
        'variance': variance,
        'dyn_range': dyn_range,
        'shape': shape,
        'desc': '; '.join(parts)
    }


def energy_ascii(curve, width=58, height=7) -> str:
    """ASCII render of the energy profile."""
    if not curve: return ''
    values = [e for _, e in curve]
    step = max(1, len(values) // width)
    sampled = [max(values[i:i+step]) for i in range(0, len(values), step)][:width]
    mx = max(sampled) if sampled else 1
    if mx == 0: return ''
    sampled = [v/mx for v in sampled]

    lines = []
    for row in range(height, 0, -1):
        threshold = row / height
        line = ''.join('█' if v >= threshold else ('▄' if v >= threshold - 0.5/height else ' ')
                       for v in sampled)
        label = f"{threshold:.1f}" if row in (height, height//2, 1) else '    '
        lines.append(f"  {label} │{line}│")
    total_time = curve[-1][0]
    lines.append(f"       └{'─'*len(sampled)}┘")
    lines.append(f"       0{' '*(len(sampled)//2-2)}{ts_str(total_time/2)}{' '*(len(sampled)//2-3)}{ts_str(total_time)}")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  3. ARCO NARRATIVO EN TRES ACTOS (segmentación dinámica)
# ═══════════════════════════════════════════════════════════════

def analyze_narrative_arc(tension_curve, energy_curve, silences_data, total_dur) -> Dict:
    """
    Segmenta la pieza dinámicamente según su dramaturgia interna.
    No usa secciones fijas — respeta los puntos de inflexión naturales.
    Detecta: planteamiento, nudo, clímax, desenlace.
    También clasifica en arquetipos narrativos clásicos.
    """
    if not tension_curve or not energy_curve:
        return {'acts': [], 'archetype': 'indeterminado', 'desc': ''}

    t_values = [p.tension for p in tension_curve]
    e_values = [e for _, e in energy_curve]

    # Normalize to same length for comparison
    n = min(len(t_values), len(e_values), 100)
    t_norm = [t_values[int(i * len(t_values) / n)] for i in range(n)]
    e_norm = [e_values[int(i * len(e_values) / n)] for i in range(n)]

    # Combined drama score
    drama = [(t + e) / 2 for t, e in zip(t_norm, e_norm)]

    # Find natural breakpoints (local minima in drama after smoothing)
    smoothed = []
    w = 5
    for i in range(n):
        window = drama[max(0,i-w):i+w+1]
        smoothed.append(sum(window)/len(window))

    # Find the global climax
    climax_idx = smoothed.index(max(smoothed))
    climax_time = climax_idx / n * total_dur

    # Find resolution point (first significant drop after climax)
    resolution_idx = climax_idx
    for i in range(climax_idx, n):
        if smoothed[i] < smoothed[climax_idx] * 0.6:
            resolution_idx = i
            break

    # Define acts based on climax position
    climax_pos = climax_idx / n  # 0-1

    if climax_pos < 0.35:
        # Climax early: in medias res structure
        acts = [
            {'name': 'Apertura intensa',    'start': 0,          'end': climax_time,   'role': 'tensión inmediata'},
            {'name': 'Desarrollo',           'start': climax_time,'end': resolution_idx/n*total_dur, 'role': 'exploración y variación'},
            {'name': 'Desenlace / Repliegue','start': resolution_idx/n*total_dur,'end': total_dur,'role': 'resolución o fade'},
        ]
        arc_type = 'In medias res (climax prematuro)'
        arc_desc = 'La pieza arranca con máxima intensidad y luego decae — impacto inmediato, reflexión posterior'
    elif climax_pos > 0.65:
        # Climax late: traditional build-up structure
        acts = [
            {'name': 'Planteamiento',   'start': 0,           'end': total_dur*0.4, 'role': 'exposición, establecimiento'},
            {'name': 'Desarrollo/Nudo', 'start': total_dur*0.4,'end': climax_time,  'role': 'tensión creciente, conflicto'},
            {'name': 'Clímax y Coda',   'start': climax_time, 'end': total_dur,     'role': 'explosión emocional y cierre'},
        ]
        arc_type = 'Crescendo clásico (climax tardío)'
        arc_desc = 'La pieza construye pacientemente hacia su explosión final — expectativa sostenida, catarsis al final'
    else:
        # Climax central: arch form
        acts = [
            {'name': 'Planteamiento',  'start': 0,           'end': climax_time * 0.5, 'role': 'establecimiento del material'},
            {'name': 'Clímax central', 'start': climax_time * 0.5, 'end': resolution_idx/n*total_dur, 'role': 'máxima tensión y drama'},
            {'name': 'Desenlace',      'start': resolution_idx/n*total_dur,'end': total_dur,'role': 'resolución, retorno'},
        ]
        arc_type = 'Arco simétrico (climax central)'
        arc_desc = 'La pieza sube, alcanza su cima y desciende — estructura narrativa clásica y satisfactoria'

    # Narrative archetype classification
    drama_variance = sum((d - sum(drama)/len(drama))**2 for d in drama) / len(drama)
    final_drama = sum(drama[-10:]) / 10 if len(drama) >= 10 else drama[-1]
    initial_drama = sum(drama[:10]) / 10 if len(drama) >= 10 else drama[0]

    if final_drama > initial_drama * 1.3:
        narrative = 'Tragedia ascendente — la tensión no se libera, el final es el momento más tenso'
    elif final_drama < initial_drama * 0.7:
        narrative = 'Resolución catártica — la pieza libera toda su tensión al final'
    elif drama_variance < 0.02:
        narrative = 'Contemplación estática — la tensión es constante, sin arco dramático'
    elif climax_pos > 0.7 and final_drama < smoothed[climax_idx] * 0.5:
        narrative = 'Explosión y silencio — climax tardío seguido de colapso brusco'
    else:
        narrative = 'Desarrollo orgánico — tensión que fluye sin estructura rígida'

    return {
        'acts': acts,
        'climax_time': climax_time,
        'climax_position': climax_pos,
        'arc_type': arc_type,
        'arc_desc': arc_desc,
        'narrative': narrative,
        'drama_curve': drama,
        'desc': f"{arc_type}. {arc_desc}."
    }


# ═══════════════════════════════════════════════════════════════
#  4. DISTANCIA EMOCIONAL ENTRE SECCIONES
# ═══════════════════════════════════════════════════════════════

def analyze_emotional_transitions(sections_va: List[Tuple[float, float, str]],
                                   sections) -> Dict:
    """
    Calcula el vector de cambio emocional entre secciones consecutivas.
    Clasifica cada transición:
      - Giro dramático (distancia VA > 0.6)
      - Transformación gradual (distancia VA 0.2-0.6)
      - Continuidad (distancia VA < 0.2)
      - Ruptura súbita (gran cambio en un solo paso)
    """
    active_va = [(v, a, d, s) for (v,a,d), s in zip(sections_va, sections) if s.notes]
    if len(active_va) < 2:
        return {'transitions': [], 'max_shift': 0, 'narrative_smoothness': 1.0, 'desc': ''}

    transitions = []
    for i in range(len(active_va) - 1):
        v1, a1, _, s1 = active_va[i]
        v2, a2, _, s2 = active_va[i+1]

        dv = v2 - v1  # valence change
        da = a2 - a1  # arousal change
        dist = math.sqrt(dv**2 + da**2)  # euclidean distance in VA space

        # Classify
        if dist > 0.7:
            ttype = 'giro dramático'
            temoji = '⚡'
        elif dist > 0.4:
            ttype = 'cambio significativo'
            temoji = '↗'
        elif dist > 0.2:
            ttype = 'transformación gradual'
            temoji = '→'
        else:
            ttype = 'continuidad emocional'
            temoji = '≈'

        # Direction
        if abs(dv) > abs(da):
            direction = 'hacia +luminosidad' if dv > 0 else 'hacia +oscuridad'
        else:
            direction = 'hacia +activación' if da > 0 else 'hacia +calma'

        transitions.append({
            'from_sec': s1.index,
            'to_sec': s2.index,
            'time': s2.start,
            'dv': dv, 'da': da,
            'distance': dist,
            'type': ttype,
            'emoji': temoji,
            'direction': direction,
        })

    distances = [t['distance'] for t in transitions]
    max_shift = max(distances) if distances else 0
    mean_shift = sum(distances) / len(distances) if distances else 0

    # Narrative smoothness: 1 = very smooth, 0 = jarring
    smoothness = 1.0 - min(mean_shift / 0.8, 1.0)

    # Describe overall emotional journey
    if smoothness > 0.8:
        journey = 'viaje emocional muy fluido — transformación continua sin rupturas'
    elif smoothness > 0.5:
        journey = 'viaje emocional equilibrado — cambios claros pero no bruscos'
    else:
        journey = 'viaje emocional accidentado — giros dramáticos frecuentes, alta intensidad narrativa'

    return {
        'transitions': transitions,
        'max_shift': max_shift,
        'mean_shift': mean_shift,
        'smoothness': smoothness,
        'journey': journey,
        'desc': journey
    }


# ═══════════════════════════════════════════════════════════════
#  5. ANTICIPACIÓN Y SUSPENSE
# ═══════════════════════════════════════════════════════════════

def analyze_suspense(tension_curve, total_dur: float) -> Dict:
    """
    Mide la capacidad de la pieza para generar y sostener suspense.
    Métricas:
      - Tiempo total en zona de alta tensión (>0.5) sin resolución
      - Latencia media de resolución
      - Número de "falsas resoluciones" (bajada breve seguida de nueva subida)
      - Índice de suspense acumulativo
    """
    if not tension_curve:
        return {'suspense_index': 0, 'max_sustained': 0, 'false_resolutions': 0, 'desc': ''}

    values = [p.tension for p in tension_curve]
    times  = [p.time for p in tension_curve]
    HIGH   = 0.4
    LOW    = 0.25
    dt     = times[1] - times[0] if len(times) > 1 else 0.5

    # Measure sustained high-tension windows
    sustained_windows = []
    in_high = False
    window_start = 0
    for i, v in enumerate(values):
        if v >= HIGH and not in_high:
            in_high = True; window_start = times[i]
        elif v < LOW and in_high:
            in_high = False
            duration = times[i] - window_start
            if duration > 1.0:
                sustained_windows.append((window_start, duration))
    if in_high:
        sustained_windows.append((window_start, times[-1] - window_start))

    # False resolutions: dip below LOW then back above HIGH within 3 seconds
    false_resolutions = 0
    for i in range(1, len(values) - 1):
        if values[i-1] >= HIGH and values[i] < LOW:
            # Check if it goes back up within 6 steps (~3s)
            recovery = any(values[j] >= HIGH for j in range(i+1, min(i+7, len(values))))
            if recovery:
                false_resolutions += 1

    max_sustained = max((d for _, d in sustained_windows), default=0)
    total_sustained = sum(d for _, d in sustained_windows)
    suspense_ratio = total_sustained / total_dur if total_dur > 0 else 0

    # Suspense index: sustained tension × false resolutions × late position
    suspense_index = min(1.0, suspense_ratio * 0.5 + false_resolutions * 0.1 + max_sustained / 30 * 0.4)

    # Build description
    parts = []
    if suspense_index > 0.6:
        parts.append("tensión muy sostenida — máximo suspense, la resolución se niega constantemente")
    elif suspense_index > 0.35:
        parts.append("suspense moderado — la pieza construye y alivia de forma calculada")
    else:
        parts.append("tensión de baja duración — la música resuelve sus tensiones rápidamente")

    if false_resolutions > 3:
        parts.append(f"múltiples falsas resoluciones (×{false_resolutions}) — el oyente es engañado repetidamente")
    elif false_resolutions > 0:
        parts.append(f"algunas falsas resoluciones (×{false_resolutions}) — desvíos narrativos expresivos")

    if max_sustained > 10:
        parts.append(f"ventana de tensión máxima sostenida: {max_sustained:.0f}s seguidos")

    return {
        'suspense_index': suspense_index,
        'max_sustained': max_sustained,
        'total_sustained': total_sustained,
        'suspense_ratio': suspense_ratio,
        'sustained_windows': sustained_windows,
        'false_resolutions': false_resolutions,
        'desc': '; '.join(parts)
    }


# ═══════════════════════════════════════════════════════════════
#  6. TRANSFORMACIONES TEMÁTICAS (Variación y Desarrollo)
# ═══════════════════════════════════════════════════════════════

def analyze_thematic_transformations(motifs: List[Dict], notes, key_root: int) -> Dict:
    """
    Detecta transformaciones de los motivos principales:
      - Transposición (mismo motivo, diferente altura)
      - Inversión (intervalos invertidos en dirección)
      - Aumentación (mismos intervalos, notas más largas)
      - Disminución (mismos intervalos, notas más cortas)
      - Retrogradación (motivo al revés)
    """
    if not motifs or not notes:
        return {'transformations': [], 'richness': 0, 'desc': 'sin motivos detectables'}

    sorted_notes = sorted(notes, key=lambda n: n.time_sec)
    intervals = [sorted_notes[i+1].pitch - sorted_notes[i].pitch
                 for i in range(len(sorted_notes)-1)]
    durations = [n.duration for n in sorted_notes]

    transformations = []
    for motif in motifs[:3]:  # analyze top 3 motifs
        pat = list(motif['pat'])
        inv = [-i for i in pat]          # inversion
        ret = list(reversed(pat))         # retrograde
        ret_inv = list(reversed(inv))     # retrograde inversion

        counts = {
            'original':    0,
            'inversión':   0,
            'retrogrado':  0,
            'ret+inv':     0,
        }

        pl = len(pat)
        for i in range(len(intervals) - pl + 1):
            window = intervals[i:i+pl]
            if window == pat:          counts['original'] += 1
            elif window == inv:        counts['inversión'] += 1
            elif window == ret:        counts['retrogrado'] += 1
            elif window == ret_inv:    counts['ret+inv'] += 1

        # Augmentation/diminution: compare note durations
        # Find motif occurrences and check if durations are scaled
        aug_count = 0; dim_count = 0
        for i in range(len(intervals) - pl + 1):
            if intervals[i:i+pl] == pat:
                mot_durs = durations[i:i+pl+1]
                if all(d > 0 for d in mot_durs):
                    avg_dur = sum(mot_durs) / len(mot_durs)
                    # Augmentation: durations > 1.5x original average
                    orig_avg = sum(durations[:min(50,len(durations))]) / min(50,len(durations))
                    if avg_dur > orig_avg * 1.5: aug_count += 1
                    elif avg_dur < orig_avg * 0.6: dim_count += 1

        counts['aumentación'] = aug_count
        counts['diminución']  = dim_count

        active_transforms = {k: v for k, v in counts.items() if v > 0}
        if active_transforms:
            transformations.append({
                'motif': motif['desc'],
                'original_count': counts['original'],
                'transforms': active_transforms,
            })

    richness = len([t for t in transformations if len(t['transforms']) > 1])
    total_transforms = sum(len(t['transforms']) for t in transformations)

    if total_transforms == 0:
        desc = "los motivos aparecen sin transformación — repetición literal, coherencia estática"
    elif richness >= 2:
        desc = f"desarrollo temático rico — {total_transforms} tipos de transformación detectados; el material se reinventa"
    else:
        desc = f"transformaciones moderadas — el material evoluciona con variaciones puntuales"

    return {
        'transformations': transformations,
        'richness': richness,
        'total_transforms': total_transforms,
        'desc': desc
    }


# ═══════════════════════════════════════════════════════════════
#  7. POLARIDAD EMOCIONAL (positivo vs negativo, sección a sección)
# ═══════════════════════════════════════════════════════════════

def analyze_emotional_polarity(sections, mode: str, tension_curve,
                                key_root: int) -> Dict:
    """
    Mide cuánto tiempo pasa la pieza en estados 'positivos' vs 'negativos'
    usando múltiples indicadores:
      - Modo (mayor=positivo, menor/frigio=negativo, dórico=neutro)
      - Consonancia local
      - Registro (agudo=más luminoso, grave=más oscuro)
      - Dinámica relativa
    """
    if not sections:
        return {'positive_ratio': 0.5, 'polarity_curve': [], 'desc': ''}

    polarity_scores = []  # per-section: -1 (dark) to +1 (bright)
    polarity_labels = []

    mode_base = {
        'major': 0.4, 'lydian': 0.5, 'mixolydian': 0.2,
        'dorian': 0.0, 'minor': -0.3, 'phrygian': -0.5, 'locrian': -0.6
    }.get(mode, 0.0)

    for sec in sections:
        if not sec.notes:
            polarity_scores.append(0.0)
            polarity_labels.append('neutro')
            continue

        # Register component: above MIDI 66 = brighter
        avg_pitch = sec.avg_pitch
        register_score = (avg_pitch - 66) / 30  # normalised

        # Consonance component (inverse of tension)
        t_values = [p.tension for p in tension_curve
                    if sec.start <= p.time < sec.end]
        section_tension = sum(t_values)/len(t_values) if t_values else 0.3
        consonance_score = 1.0 - section_tension * 2  # -1 to +1

        # Dynamic component: louder = more active = more 'positive' valence in uptempo
        vel_score = (sec.avg_velocity - 64) / 63

        # Combine
        polarity = (mode_base * 0.40 +
                    consonance_score * 0.30 +
                    register_score * 0.15 +
                    vel_score * 0.15)
        polarity = max(-1.0, min(1.0, polarity))
        polarity_scores.append(polarity)

        if polarity > 0.3:   polarity_labels.append('luminoso')
        elif polarity > 0.0: polarity_labels.append('neutro-positivo')
        elif polarity > -0.3:polarity_labels.append('neutro-oscuro')
        else:                polarity_labels.append('oscuro')

    active_scores = [s for s, sec in zip(polarity_scores, sections) if sec.notes]
    if not active_scores:
        return {'positive_ratio': 0.5, 'polarity_curve': [], 'desc': ''}

    positive_time = sum(1 for s in active_scores if s > 0) / len(active_scores)
    mean_polarity = sum(active_scores) / len(active_scores)

    # Describe
    if positive_time > 0.7:
        desc = f"pieza mayoritariamente luminosa ({positive_time*100:.0f}% en estados positivos) — esperanza, apertura"
    elif positive_time > 0.5:
        desc = f"ligero predominio de luz ({positive_time*100:.0f}%) — ambivalencia con tendencia positiva"
    elif positive_time > 0.3:
        desc = f"ligero predominio de sombra ({(1-positive_time)*100:.0f}% en estados oscuros) — melancolía matizada"
    else:
        desc = f"pieza mayoritariamente oscura ({(1-positive_time)*100:.0f}% en estados negativos) — predominio de la sombra"

    return {
        'positive_ratio': positive_time,
        'mean_polarity': mean_polarity,
        'polarity_scores': polarity_scores,
        'polarity_labels': polarity_labels,
        'desc': desc
    }


# ═══════════════════════════════════════════════════════════════
#  8. DENSIDAD DE EVENTOS SIGNIFICATIVOS
# ═══════════════════════════════════════════════════════════════

def analyze_event_density(notes, chords, tension_curve, silences_data,
                           motifs, total_dur: float) -> Dict:
    """
    Cuenta cuántos 'eventos notables' ocurren por minuto:
      - Cambios de acorde
      - Picos de tensión
      - Entradas de motivos
      - Silencios expresivos (>0.5s)
      - Cambios de registro (saltos >12st)
    Alta densidad = narrativo. Baja densidad = contemplativo.
    """
    if total_dur <= 0:
        return {'events_per_minute': 0, 'style': 'indeterminado', 'desc': ''}

    # Count events
    chord_changes = len(chords)

    tension_peaks = sum(1 for i in range(1, len(tension_curve)-1)
                        if (tension_curve[i].tension > tension_curve[i-1].tension and
                            tension_curve[i].tension > tension_curve[i+1].tension and
                            tension_curve[i].tension > 0.35))

    significant_silences = len([s for s in silences_data.get('silences', [])
                                 if s['duration'] > 0.5])

    # Register leaps (jumps > octave between consecutive notes)
    sorted_notes = sorted(notes, key=lambda n: n.time_sec)
    register_leaps = sum(1 for i in range(len(sorted_notes)-1)
                         if abs(sorted_notes[i+1].pitch - sorted_notes[i].pitch) > 12)

    motif_occurrences = sum(m['occ'] for m in motifs)

    total_events = (chord_changes + tension_peaks +
                    significant_silences + register_leaps // 5 +
                    motif_occurrences // 3)

    events_per_minute = total_events / (total_dur / 60)

    if events_per_minute < 20:
        style = 'contemplativo'
        desc = f"muy pocos eventos por minuto ({events_per_minute:.0f}) — música de espacio y quietud"
    elif events_per_minute < 50:
        style = 'narrativo moderado'
        desc = f"densidad narrativa equilibrada ({events_per_minute:.0f} eventos/min)"
    elif events_per_minute < 100:
        style = 'narrativo denso'
        desc = f"alta densidad de eventos ({events_per_minute:.0f}/min) — mucho ocurre en poco tiempo"
    else:
        style = 'torrencial'
        desc = f"densidad extrema ({events_per_minute:.0f}/min) — música de alta información, exigente"

    return {
        'events_per_minute': events_per_minute,
        'chord_changes': chord_changes,
        'tension_peaks': tension_peaks,
        'significant_silences': significant_silences,
        'register_leaps': register_leaps,
        'motif_occurrences': motif_occurrences,
        'style': style,
        'desc': desc
    }


# ═══════════════════════════════════════════════════════════════
#  9. CLASIFICACIÓN DE GÉNERO EMOCIONAL
# ═══════════════════════════════════════════════════════════════

def classify_emotional_genre(mode: str, avg_bpm: float, overall_tension: float,
                              centricity: float, syncopation: float,
                              mean_valence: float, mean_arousal: float,
                              suspense_index: float) -> Dict:
    """
    Clasifica la pieza en géneros emocionales funcionales
    comparando sus métricas con los arquetipos definidos.
    Devuelve ranking de los 3 géneros más probables con puntuación.
    """
    scores = {}

    for genre, archetype in ARCHETYPES.items():
        score = 0.0
        n_criteria = 0

        # Mode match
        if archetype['mode'] == mode:
            score += 1.0
        elif ((archetype['mode'] in ('major','lydian','mixolydian')) ==
              (mode in ('major','lydian','mixolydian'))):
            score += 0.4
        n_criteria += 1

        # BPM range
        bpm_lo, bpm_hi = archetype['bpm']
        if bpm_lo <= avg_bpm <= bpm_hi:
            score += 1.0
        else:
            dist = min(abs(avg_bpm - bpm_lo), abs(avg_bpm - bpm_hi))
            score += max(0, 1.0 - dist / 40)
        n_criteria += 1

        # Tension range
        t_lo, t_hi = archetype['tension']
        if t_lo <= overall_tension <= t_hi:
            score += 1.0
        else:
            dist = min(abs(overall_tension - t_lo), abs(overall_tension - t_hi))
            score += max(0, 1.0 - dist / 0.3)
        n_criteria += 1

        # Centricity
        c_lo, c_hi = archetype['centricity']
        if c_lo <= centricity <= c_hi:
            score += 1.0
        else:
            dist = min(abs(centricity - c_lo), abs(centricity - c_hi))
            score += max(0, 1.0 - dist / 0.3)
        n_criteria += 1

        # Syncopation
        s_lo, s_hi = archetype['syncopation']
        if s_lo <= syncopation <= s_hi:
            score += 1.0
        else:
            dist = min(abs(syncopation - s_lo), abs(syncopation - s_hi))
            score += max(0, 1.0 - dist / 0.3)
        n_criteria += 1

        # Valence
        v_lo, v_hi = archetype['valence']
        if v_lo <= mean_valence <= v_hi:
            score += 1.0
        else:
            dist = min(abs(mean_valence - v_lo), abs(mean_valence - v_hi))
            score += max(0, 1.0 - dist / 0.4)
        n_criteria += 1

        # Arousal
        a_lo, a_hi = archetype['arousal']
        if a_lo <= mean_arousal <= a_hi:
            score += 1.0
        else:
            dist = min(abs(mean_arousal - a_lo), abs(mean_arousal - a_hi))
            score += max(0, 1.0 - dist / 0.4)
        n_criteria += 1

        scores[genre] = score / n_criteria

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top3 = ranked[:3]

    return {
        'top_genre': top3[0][0],
        'top_score': top3[0][1],
        'ranking': top3,
        'all_scores': scores,
        'desc': f"Género emocional predominante: {top3[0][0]} ({top3[0][1]*100:.0f}%) — {ARCHETYPES[top3[0][0]]['desc']}"
    }


# ═══════════════════════════════════════════════════════════════
#  10. CROMATISMO DIRIGIDO VS CROMATISMO DE COLOR
# ═══════════════════════════════════════════════════════════════

def analyze_chromaticism(notes, key_root: int, mode: str) -> Dict:
    """
    Distingue entre:
      - Cromatismo dirigido: movimiento semitonal con dirección clara
        (Do → Do# → Re) — sensación de movimiento inevitable, tensión acumulativa
      - Cromatismo de color: acordes/notas cromáticas sin dirección
        — ambigüedad, color, exotismo
    También detecta líneas cromáticas (3+ semitonos consecutivos en la misma dirección).
    """
    if not notes:
        return {'directed': 0, 'color': 0, 'lines': [], 'desc': ''}

    sorted_notes = sorted(notes, key=lambda n: n.time_sec)
    scale = set(SCALE_INTERVALS.get(mode, SCALE_INTERVALS['minor']))

    intervals = [sorted_notes[i+1].pitch - sorted_notes[i].pitch
                 for i in range(len(sorted_notes)-1)]

    chromatic_moves = [(i, iv) for i, iv in enumerate(intervals) if abs(iv) == 1]
    total_moves = len(intervals)

    # Directed chromaticism: consecutive semitones in same direction
    directed = 0
    chromatic_lines = []
    i = 0
    while i < len(intervals) - 1:
        if abs(intervals[i]) == 1:
            # Check if next move continues in same direction
            direction = 1 if intervals[i] > 0 else -1
            length = 1
            j = i + 1
            while j < len(intervals) and intervals[j] == direction:
                length += 1; j += 1
            if length >= 2:
                directed += length
                start_time = sorted_notes[i].time_sec
                chromatic_lines.append({
                    'start': start_time,
                    'length': length,
                    'direction': 'ascendente' if direction > 0 else 'descendente',
                    'semitones': length
                })
            i = j
        else:
            i += 1

    color_chromatic = len(chromatic_moves) - directed

    if total_moves == 0:
        return {'directed': 0, 'color': 0, 'lines': [], 'desc': 'sin datos'}

    directed_ratio = directed / total_moves
    color_ratio = color_chromatic / total_moves

    parts = []
    if directed_ratio > 0.15:
        parts.append(f"cromatismo dirigido prominente ({directed_ratio*100:.0f}%) — tensión acumulativa, movimiento inexorable")
    if color_ratio > 0.1:
        parts.append(f"cromatismo de color ({color_ratio*100:.0f}%) — ambigüedad, exotismo, tensión estática")
    if chromatic_lines:
        longest = max(chromatic_lines, key=lambda l: l['length'])
        parts.append(f"línea cromática más larga: {longest['length']} semitonos {longest['direction']} en {ts_str(longest['start'])}")
    if not parts:
        parts.append("cromatismo mínimo — lenguaje diatónico predominante")

    return {
        'directed': directed,
        'color': color_chromatic,
        'directed_ratio': directed_ratio,
        'color_ratio': color_ratio,
        'lines': chromatic_lines,
        'desc': '; '.join(parts)
    }


# ═══════════════════════════════════════════════════════════════
#  11. CLÍMAX EMOCIONAL VS CLÍMAX SONORO
# ═══════════════════════════════════════════════════════════════

def compare_emotional_vs_sonic_climax(tension_curve, energy_curve,
                                       total_dur: float) -> Dict:
    """
    Detecta si el clímax emocional (máxima tensión sin resolver)
    coincide con el clímax sonoro (máxima energía/dinámica).
    La disincronía es una decisión compositiva muy expresiva.
    """
    if not tension_curve or not energy_curve:
        return {'sync': True, 'offset': 0, 'desc': 'datos insuficientes'}

    t_values = [p.tension for p in tension_curve]
    t_times  = [p.time    for p in tension_curve]
    e_values = [e for _, e in energy_curve]
    e_times  = [t for t, _ in energy_curve]

    emotional_climax_t = t_times[t_values.index(max(t_values))] if t_values else 0
    sonic_climax_t     = e_times[e_values.index(max(e_values))] if e_values else 0

    offset = abs(emotional_climax_t - sonic_climax_t)
    offset_pct = offset / total_dur * 100 if total_dur > 0 else 0

    if offset < 3.0:
        sync = True
        desc = (f"clímax emocional y sonoro coinciden (Δ={offset:.1f}s) — "
                "máximo impacto integrado, la música y la emoción estallan juntas")
    elif emotional_climax_t < sonic_climax_t:
        sync = False
        desc = (f"el clímax emocional precede al sonoro en {offset:.1f}s — "
                "la tensión se construye en silencio antes de explotar; técnica de anticipación sofisticada")
    else:
        sync = False
        desc = (f"el clímax sonoro precede al emocional en {offset:.1f}s — "
                "la explosión sonora no es el punto más tenso; la verdadera emoción llega después del ruido")

    return {
        'sync': sync,
        'emotional_climax': emotional_climax_t,
        'sonic_climax': sonic_climax_t,
        'offset': offset,
        'offset_pct': offset_pct,
        'desc': desc
    }


# ═══════════════════════════════════════════════════════════════
#  12. ANÁLISIS DE GROOVES E HIPNOSIS
# ═══════════════════════════════════════════════════════════════

def analyze_groove_hypnosis(notes, avg_bpm: float, total_dur: float) -> Dict:
    """
    Detecta patrones repetitivos que crean efecto de trance/hipnosis.
    Mide: ostinato rítmico, repetición de patrones de altura,
    y longitud de las secuencias repetitivas.
    """
    if not notes or total_dur <= 0:
        return {'groove_strength': 0, 'ostinato_ratio': 0, 'desc': 'sin datos'}

    beat = 60.0 / avg_bpm if avg_bpm > 0 else 0.5
    sorted_notes = sorted(notes, key=lambda n: n.time_sec)

    # Rhythmic repetition: bucket notes into beat bins
    bin_size = beat / 2  # eighth note bins
    rhythm_pattern = collections.Counter()
    for n in sorted_notes:
        bin_idx = int(n.time_sec / bin_size) % 16  # 8-bar pattern
        rhythm_pattern[bin_idx] += 1

    # How concentrated is the rhythm in certain bins?
    total_hits = sum(rhythm_pattern.values())
    if total_hits == 0:
        return {'groove_strength': 0, 'ostinato_ratio': 0, 'desc': 'sin datos'}

    top_bins = rhythm_pattern.most_common(4)
    concentration = sum(v for _, v in top_bins) / total_hits

    # Pitch ostinato: repeated pitch-class sequences
    pitch_classes = [n.pitch % 12 for n in sorted_notes]
    pc_seq_len = 4
    pc_counter = collections.Counter(
        tuple(pitch_classes[i:i+pc_seq_len])
        for i in range(len(pitch_classes) - pc_seq_len)
    )
    top_pc = pc_counter.most_common(1)
    ostinato_ratio = top_pc[0][1] / len(pitch_classes) if top_pc else 0

    # Groove strength: combination of rhythmic concentration + pitch repetition
    groove_strength = (concentration * 0.6 + min(ostinato_ratio * 10, 1.0) * 0.4)

    # Find longest continuous repetitive section
    window_sec = 8.0  # 8-second windows
    max_repetition = 0
    t = 0.0
    while t < total_dur - window_sec:
        w1 = tuple(n.pitch % 12 for n in sorted_notes if t <= n.time_sec < t + window_sec)
        w2 = tuple(n.pitch % 12 for n in sorted_notes if t + window_sec <= n.time_sec < t + 2*window_sec)
        if w1 and w2:
            # Simple similarity: shared elements
            overlap = len(set(w1) & set(w2)) / max(len(set(w1) | set(w2)), 1)
            if overlap > 0.8:
                max_repetition += window_sec
        t += window_sec / 2

    parts = []
    if groove_strength > 0.7:
        parts.append(f"groove muy fuerte ({groove_strength:.2f}) — efecto hipnótico, ritmo obsesivo")
    elif groove_strength > 0.45:
        parts.append(f"groove moderado ({groove_strength:.2f}) — patrón reconocible, sensación de ciclo")
    else:
        parts.append(f"groove débil ({groove_strength:.2f}) — sin patrón rítmico dominante, escritura libre")

    if max_repetition > 20:
        parts.append(f"sección repetitiva sostenida de ~{max_repetition:.0f}s — zona de trance deliberada")

    return {
        'groove_strength': groove_strength,
        'ostinato_ratio': ostinato_ratio,
        'rhythmic_concentration': concentration,
        'max_repetition_sec': max_repetition,
        'desc': '; '.join(parts)
    }


# ═══════════════════════════════════════════════════════════════
#  TEST RÁPIDO
# ═══════════════════════════════════════════════════════════════



# ══ NEW ANALYSES v5 ══

# ── constantes musicales ──────────────────────────────────────
NOTE_NAMES   = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
DISSONANCE   = {0:0.00,1:1.00,2:0.80,3:0.25,4:0.20,5:0.10,
                6:0.95,7:0.00,8:0.20,9:0.10,10:0.45,11:0.85}
SCALE_IVLS   = {
    'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10],
    'dorian':[0,2,3,5,7,9,10],'phrygian':[0,1,3,5,7,8,10],
    'lydian':[0,2,4,6,7,9,11],'mixolydian':[0,2,4,5,7,9,10],
    'locrian':[0,1,3,5,6,8,10],
}


# ts_str definida en el módulo superior (línea 326) — no redefinir aquí.


# ═══════════════════════════════════════════════════════════════
#  1. ANÁLISIS DE FRASEO
# ═══════════════════════════════════════════════════════════════

def analyze_phrasing(notes, bpm: float, time_sig_num: int = 4) -> Dict:
    """
    Detecta frases musicales como unidades de sentido.
    Una frase se delimita por:
      - Silencios > umbral
      - Descenso en velocidad seguido de pausa
      - Patrones cadenciales
    Analiza: longitud, simetría, relación antecedente-consecuente.
    """
    if not notes or bpm <= 0:
        return {'phrases': [], 'avg_length': 0, 'symmetry': 0, 'desc': ''}

    beat = 60.0 / bpm
    bar  = beat * time_sig_num
    sn   = sorted(notes, key=lambda n: n.time_sec)
    total = sn[-1].time_sec + sn[-1].duration

    # Find phrase boundaries: gaps > half a bar
    gap_threshold = bar * 0.5
    boundaries = [0.0]
    for i in range(len(sn) - 1):
        gap = sn[i+1].time_sec - (sn[i].time_sec + sn[i].duration)
        if gap > gap_threshold:
            boundaries.append(sn[i].time_sec + sn[i].duration)
    boundaries.append(total)

    # Also add boundary at velocity drops (phrase endings)
    for i in range(4, len(sn) - 4):
        vel_before = sum(n.velocity for n in sn[i-4:i]) / 4
        vel_after  = sum(n.velocity for n in sn[i:i+4]) / 4
        if vel_before > vel_after + 20 and sn[i].time_sec not in boundaries:
            # Check there's at least a minimal gap here
            gap = sn[i].time_sec - (sn[i-1].time_sec + sn[i-1].duration)
            if gap > beat * 0.25:
                boundaries.append(sn[i].time_sec)

    boundaries = sorted(set(boundaries))

    # Build phrases
    phrases = []
    for i in range(len(boundaries) - 1):
        t0, t1 = boundaries[i], boundaries[i+1]
        ph_notes = [n for n in sn if t0 <= n.time_sec < t1]
        if len(ph_notes) < 2:
            continue
        length_sec = t1 - t0
        length_bars = length_sec / bar
        avg_vel = sum(n.velocity for n in ph_notes) / len(ph_notes)
        avg_pit = sum(n.pitch for n in ph_notes) / len(ph_notes)
        # Contour: rising, falling, arch, valley
        first_pit = ph_notes[0].pitch
        last_pit  = ph_notes[-1].pitch
        peak_pit  = max(n.pitch for n in ph_notes)
        trough_pit= min(n.pitch for n in ph_notes)
        if peak_pit == first_pit:     contour = 'descendente'
        elif peak_pit == last_pit:    contour = 'ascendente'
        elif (peak_pit - first_pit > 3) and (peak_pit - last_pit > 3):
            contour = 'arco'
        elif (first_pit - trough_pit > 3) and (last_pit - trough_pit > 3):
            contour = 'valle'
        else:                         contour = 'llana'
        phrases.append({
            'start': t0, 'end': t1, 'length_sec': length_sec,
            'length_bars': length_bars, 'note_count': len(ph_notes),
            'avg_velocity': avg_vel, 'avg_pitch': avg_pit,
            'contour': contour,
        })

    if not phrases:
        return {'phrases': [], 'avg_length': 0, 'symmetry': 0,
                'antecedent_consequent': False, 'desc': 'frases no detectables'}

    lengths = [p['length_sec'] for p in phrases]
    avg_len = sum(lengths) / len(lengths)
    # Symmetry: how similar are phrase lengths to each other
    variance = sum((l - avg_len)**2 for l in lengths) / len(lengths)
    symmetry = max(0, 1.0 - math.sqrt(variance) / (avg_len + 0.001))

    # Antecedent-consequent: pairs of similar-length phrases
    ac_pairs = 0
    for i in range(0, len(phrases) - 1, 2):
        ratio = phrases[i]['length_sec'] / max(phrases[i+1]['length_sec'], 0.1)
        if 0.75 < ratio < 1.33:
            ac_pairs += 1
    ac_ratio = ac_pairs / max(len(phrases) // 2, 1)

    # Description
    if len(phrases) == 0:
        desc = 'escritura continua sin frases claras — discurso musical ininterrumpido'
    elif avg_len < bar * 2:
        desc = f'frases muy cortas (μ={avg_len:.1f}s) — música aforística, puntuación densa'
    elif avg_len < bar * 4:
        desc = f'frases de 2-4 compases (μ={avg_len:.1f}s) — estructura clásica, equilibrio'
    elif avg_len < bar * 8:
        desc = f'frases largas (μ={avg_len:.1f}s) — discurso lírico amplio, romanticismo'
    else:
        desc = f'frases muy largas (μ={avg_len:.1f}s) — música de largas respiraciones, meditativa'

    if symmetry > 0.8:
        desc += '; alta simetría entre frases — arquitectura equilibrada, formalismo'
    elif symmetry < 0.4:
        desc += '; frases asimétricas — prosa musical libre, expresión orgánica'

    if ac_ratio > 0.5:
        desc += '; estructura pregunta-respuesta clara — diálogo interno, lógica narrativa'

    return {
        'phrases': phrases,
        'count': len(phrases),
        'avg_length': avg_len,
        'avg_length_bars': avg_len / bar if bar > 0 else 0,
        'symmetry': symmetry,
        'antecedent_consequent': ac_ratio > 0.5,
        'ac_ratio': ac_ratio,
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  2. NARRATIVA EMOCIONAL COMO ARCO DE PERSONAJE
# ═══════════════════════════════════════════════════════════════

def generate_emotional_story(
    sections, sva_list, tension_curve, energy_curve,
    mode: str, key_root: int, avg_bpm: float,
    cadences: Dict, motifs: List, musical_form: Dict,
    silence_data: Dict, phrasing: Dict,
    key_name: str, total_dur: float
) -> str:
    """
    Genera una narrativa en lenguaje natural que describe la "historia"
    que cuenta la música, como un arco de personaje literario.
    """
    active = [(s, va) for s, va in zip(sections, sva_list) if s.notes]
    if not active:
        return '  Insuficientes datos para generar narrativa.'

    # Initial state
    s0, va0 = active[0]
    v0, a0, _ = va0
    vel0 = s0.avg_velocity

    # Final state
    sf, vaf = active[-1]
    vf, af, _ = vaf
    velf = sf.avg_velocity

    # Find biggest emotional shift
    max_shift = 0; shift_idx = 0
    for i in range(1, len(active)):
        v1, a1, _ = active[i-1][1]
        v2, a2, _ = active[i][1]
        d = math.sqrt((v2-v1)**2 + (a2-a1)**2)
        if d > max_shift:
            max_shift = d; shift_idx = i

    # Tension peak
    if tension_curve:
        t_vals = [p.tension for p in tension_curve]
        t_peak_t = tension_curve[t_vals.index(max(t_vals))].time
    else:
        t_peak_t = total_dur * 0.7

    # Energy peak
    if energy_curve:
        e_vals = [e for _, e in energy_curve]
        e_peak_t = energy_curve[e_vals.index(max(e_vals))][0]
    else:
        e_peak_t = total_dur * 0.75

    # Mode character
    mode_chars = {
        'major':      ('un estado de apertura y luminosidad',       'afirmación'),
        'minor':      ('una oscuridad introspectiva',                'tristeza'),
        'dorian':     ('una melancolía con chispa de esperanza',     'anhelo activo'),
        'phrygian':   ('una tensión fatalista y amenazante',         'fatalismo'),
        'lydian':     ('un ensueño flotante e irreal',               'trascendencia'),
        'mixolydian': ('una ambigüedad épica sin resolución clara',  'épica'),
        'locrian':    ('una inestabilidad extrema y disonante',      'ansiedad'),
    }
    initial_char, emotional_core = mode_chars.get(mode, ('un estado complejo', 'ambigüedad'))

    # Opening paragraph
    if vel0 < 50:
        opening = (f"La pieza arranca en voz baja, casi en susurro. "
                   f"Desde los primeros compases en {key_name}, establece "
                   f"{initial_char} — un espacio íntimo donde algo todavía no ha tomado forma.")
    elif vel0 > 80:
        opening = (f"Sin preámbulos, la pieza irrumpe con determinación en {key_name}. "
                   f"La energía inicial es alta, como si el discurso ya llevara tiempo "
                   f"gestándose antes de que comenzara la música.")
    else:
        opening = (f"La pieza se presenta en {key_name} con una energía mesurada — "
                   f"ni tímida ni arrolladora. Establece desde el principio "
                   f"{initial_char}.")

    # Development paragraph
    form_desc = musical_form.get('form', '')
    if 'Arco' in form_desc or 'Ternaria' in form_desc:
        development = ("El material se desarrolla siguiendo una lógica de exposición "
                       "y retorno — la pieza recuerda lo que ha dicho y vuelve a ello "
                       "transformado por el viaje.")
    elif 'Through' in form_desc:
        development = ("No hay vuelta atrás: la música avanza de forma lineal, "
                       "cada sección introduce material nuevo sin recapitulación. "
                       "Es una música que huye hacia adelante.")
    else:
        development = ("La pieza despliega su material de forma orgánica, "
                       "sin adherirse a una forma canónica — la lógica interna "
                       "dicta la estructura.")

    if motifs:
        best_motif = motifs[0]
        development += (f" Un motivo de {best_motif['len']} notas "
                        f"({best_motif['desc'].split('[')[0].strip()}) "
                        f"actúa como hilo conductor, reapareciendo {best_motif['occ']} veces "
                        f"— la firma reconocible que unifica el discurso.")

    # Climax paragraph
    climax_time = ts_str(t_peak_t)
    if abs(t_peak_t - e_peak_t) < 5:
        climax_para = (f"El momento decisivo llega en {climax_time}: energía emocional "
                       f"y sonora convergen en un único punto de máximo impacto. "
                       f"La música no esconde su clímax — lo entrega con plena convicción.")
    elif t_peak_t > e_peak_t:
        climax_para = (f"El punto más ruidoso llega en {ts_str(e_peak_t)}, "
                       f"pero la verdadera tensión emocional alcanza su cima en {climax_time} — "
                       f"más tarde, más quieta, más perturbadora. "
                       f"El silencio cargado supera al fortissimo.")
    else:
        climax_para = (f"La tensión emocional alcanza su punto álgido en {climax_time} "
                       f"antes de que la música explote sonoramente — "
                       f"la emoción anticipa y precede al impacto físico.")

    # Resolution paragraph
    vel_delta = velf - vel0
    if vel_delta > 15:
        resolution = ("La pieza no se detiene ni se rinde: termina más fuerte que como empezó. "
                      "No hay catarsis ni liberación — hay afirmación. "
                      "El estado final es el de alguien que ha decidido habitar su tensión.")
    elif vel_delta < -15:
        resolution = ("La música se disuelve gradualmente, devolviendo al oyente "
                      "a un estado más quieto que el inicial. "
                      "La energía se libera, la tensión se depone — es un final de rendición o de paz.")
    else:
        resolution = ("La pieza regresa aproximadamente al nivel dinámico con que comenzó, "
                      "completando un arco circular. Lo que ha ocurrido en el medio "
                      "queda suspendido entre los dos silencios que enmarcan la obra.")

    # Emotional core summary
    cad_dom = cadences.get('dominant_type', '')
    if cad_dom == 'rota':
        cad_comment = ("Las cadencias rotas frecuentes revelan una música que se niega "
                       "a llegar donde el oyente espera — un discurso de evasiones deliberadas.")
    elif cad_dom == 'perfecta':
        cad_comment = ("Las cadencias perfectas dan a la pieza su sentido de certeza: "
                       "sabe adónde va y llega.")
    elif not cad_dom or 'sin' in cad_dom:
        cad_comment = ("La ausencia de cadencias convencionales convierte la pieza "
                       "en un espacio armónico suspendido, sin anclas de resolución — "
                       "música que vive en la pregunta, no en la respuesta.")
    else:
        cad_comment = ""

    silence_ratio = silence_data.get('ratio', 0)
    if silence_ratio < 0.05:
        silence_comment = ("La densísima escritura, casi sin aire, crea una presión "
                           "psicológica constante — no hay descanso, no hay respiro.")
    elif silence_ratio > 0.25:
        silence_comment = ("Los amplios silencios son tan compositivos como las notas. "
                           "La música habita el espacio vacío con tanta intención "
                           "como el lleno.")
    else:
        silence_comment = ""

    paragraphs = [
        f"  {opening}",
        f"",
        f"  {development}",
        f"",
        f"  {climax_para}",
        f"",
        f"  {resolution}",
    ]
    if cad_comment:
        paragraphs += ["", f"  {cad_comment}"]
    if silence_comment:
        paragraphs += ["", f"  {silence_comment}"]

    paragraphs += [
        "",
        f"  En su conjunto, es una pieza de {emotional_core} — "
        f"{'no busca consolación sino comprensión.' if 'min' in mode or mode == 'dorian' else 'afirma su mundo con la fuerza de quien ya no necesita justificarlo.'}",
    ]

    return '\n'.join(paragraphs)


# ═══════════════════════════════════════════════════════════════
#  3. CONTRAPUNTO E INTERDEPENDENCIA DE VOCES
# ═══════════════════════════════════════════════════════════════

def analyze_counterpoint(notes, meta: Dict) -> Dict:
    """
    Analiza cómo interactúan las voces/canales entre sí.
    Detecta: movimiento paralelo, contrario, oblicuo, directo.
    Imitación entre voces. Independencia melódica.
    """
    # Group by channel
    channels = collections.defaultdict(list)
    for n in notes:
        channels[n.channel].append(n)

    if len(channels) < 2:
        # Try by track
        tracks = collections.defaultdict(list)
        for n in notes:
            tracks[n.track].append(n)
        if len(tracks) < 2:
            return {'voices': 1, 'motion_type': 'monofónico',
                    'imitation': False, 'independence': 0,
                    'desc': 'una sola voz — análisis de contrapunto no aplicable'}
        voices = {k: sorted(v, key=lambda n: n.time_sec) for k,v in tracks.items()}
    else:
        voices = {k: sorted(v, key=lambda n: n.time_sec) for k,v in channels.items()}

    voice_keys = list(voices.keys())
    n_voices = len(voice_keys)

    if n_voices < 2:
        return {'voices': 1, 'motion_type': 'monofónico',
                'imitation': False, 'independence': 0,
                'desc': 'una sola voz'}

    # Compare top two voices (highest avg pitch = soprano, next = alto)
    sorted_voices = sorted(voice_keys,
                           key=lambda k: -sum(n.pitch for n in voices[k])/max(len(voices[k]),1))
    v1_notes = voices[sorted_voices[0]]
    v2_notes = voices[sorted_voices[1]]

    # Sample simultaneous pairs at regular intervals
    total = max(n.time_sec + n.duration for n in notes)
    resolution = 0.5
    parallel = contrary = oblique = similar = 0
    samples = 0

    t = resolution
    while t < total - resolution:
        # Get pitch at time t for each voice
        def pitch_at(voice_notes, time):
            active = [n for n in voice_notes if n.time_sec <= time < n.time_sec + n.duration]
            return active[0].pitch if active else None

        p1a = pitch_at(v1_notes, t - resolution)
        p1b = pitch_at(v1_notes, t)
        p2a = pitch_at(v2_notes, t - resolution)
        p2b = pitch_at(v2_notes, t)

        if all(p is not None for p in [p1a, p1b, p2a, p2b]):
            d1 = p1b - p1a  # voice 1 motion
            d2 = p2b - p2a  # voice 2 motion
            samples += 1
            if d1 == 0 and d2 == 0:
                oblique += 1  # both stationary
            elif d1 == 0 or d2 == 0:
                oblique += 1  # one stationary
            elif (d1 > 0) == (d2 > 0):
                if d1 == d2:
                    parallel += 1  # same direction, same interval
                else:
                    similar += 1   # same direction, different interval
            else:
                contrary += 1  # opposite directions
        t += resolution

    if samples == 0:
        return {'voices': n_voices, 'motion_type': 'indeterminado',
                'imitation': False, 'independence': 0.5, 'desc': 'sin datos suficientes'}

    par_r  = parallel  / samples
    con_r  = contrary  / samples
    obl_r  = oblique   / samples
    sim_r  = similar   / samples

    # Imitation detection: check if voice 2 echoes voice 1's intervals with a delay
    ivs1 = [v1_notes[i+1].pitch - v1_notes[i].pitch for i in range(min(30, len(v1_notes)-1))]
    ivs2 = [v2_notes[i+1].pitch - v2_notes[i].pitch for i in range(min(30, len(v2_notes)-1))]
    imitation = False
    if len(ivs1) >= 4 and len(ivs2) >= 4:
        for offset in range(1, min(8, len(ivs2) - 3)):
            matches = sum(1 for i in range(min(len(ivs1), len(ivs2)-offset))
                         if ivs1[i] == ivs2[i+offset])
            if matches / max(len(ivs1), 1) > 0.4:
                imitation = True; break

    # Independence: how different are the two voices?
    independence = (con_r * 1.0 + sim_r * 0.7 + obl_r * 0.3 + par_r * 0.0)

    # Motion type classification
    if par_r > 0.5:
        motion_type = 'paralelo predominante'
        motion_desc = 'las voces se mueven juntas — cohesión, homofónica, densidad emocional unificada'
    elif con_r > 0.4:
        motion_type = 'contrario predominante'
        motion_desc = 'las voces se oponen frecuentemente — independencia, tensión contrapuntística, diálogo'
    elif obl_r > 0.5:
        motion_type = 'oblicuo predominante'
        motion_desc = 'una voz se mueve mientras la otra se sostiene — pedal expresivo, anclaje'
    else:
        motion_type = 'movimiento mixto'
        motion_desc = 'combinación equilibrada de tipos de movimiento — riqueza polifónica'

    imit_desc = ' Con imitación entre voces detectada — técnica canónica, eco, respuesta.' if imitation else ''

    return {
        'voices': n_voices,
        'motion_type': motion_type,
        'parallel_ratio': par_r,
        'contrary_ratio': con_r,
        'oblique_ratio': obl_r,
        'similar_ratio': sim_r,
        'imitation': imitation,
        'independence': independence,
        'desc': f"{motion_desc}.{imit_desc}",
    }


# ═══════════════════════════════════════════════════════════════
#  4. MICRODINÁMICAS EXPRESIVAS DE NOTA INDIVIDUAL
# ═══════════════════════════════════════════════════════════════

def analyze_microexpression(notes, bpm: float) -> Dict:
    """
    Analiza la varianza de velocidades dentro de frases cortas.
    Alta varianza = interpretación expresiva.
    Detecta: acentos estratégicos, hairpins, agógica.
    """
    if not notes or bpm <= 0:
        return {'expressiveness': 0, 'accent_ratio': 0, 'desc': ''}

    beat = 60.0 / bpm
    sn   = sorted(notes, key=lambda n: n.time_sec)
    total = sn[-1].time_sec

    # Compute local velocity variance in 2-beat windows
    window_variances = []
    t = 0.0
    while t < total - beat * 2:
        window = [n for n in sn if t <= n.time_sec < t + beat * 2]
        if len(window) >= 3:
            vels = [n.velocity for n in window]
            mean_v = sum(vels) / len(vels)
            var = sum((v - mean_v)**2 for v in vels) / len(vels)
            window_variances.append(math.sqrt(var))
        t += beat

    if not window_variances:
        return {'expressiveness': 0, 'accent_ratio': 0, 'desc': 'datos insuficientes'}

    mean_var = sum(window_variances) / len(window_variances)
    expressiveness = min(mean_var / 20.0, 1.0)  # normalise: 20 std dev = max

    # Detect accents: notes significantly louder than their neighbors
    accents = 0
    for i in range(1, len(sn) - 1):
        local_mean = (sn[i-1].velocity + sn[i+1].velocity) / 2
        if sn[i].velocity > local_mean + 15:
            accents += 1
    accent_ratio = accents / len(sn) if sn else 0

    # Detect hairpins (gradual velocity ramps)
    hairpins = 0
    window_size = max(4, int(beat * 2 / 0.1))
    for i in range(0, len(sn) - window_size, window_size // 2):
        chunk_vels = [sn[j].velocity for j in range(i, min(i+window_size, len(sn)))]
        if len(chunk_vels) < 4:
            continue
        # Linear regression slope
        n = len(chunk_vels)
        xs = list(range(n))
        mean_x = sum(xs) / n; mean_y = sum(chunk_vels) / n
        num = sum((xs[j]-mean_x)*(chunk_vels[j]-mean_y) for j in range(n))
        den = sum((xs[j]-mean_x)**2 for j in range(n))
        slope = num/den if den > 0 else 0
        if abs(slope) > 1.5:  # significant ramp
            hairpins += 1

    # Overall velocity range
    all_vels = [n.velocity for n in sn]
    vel_range = max(all_vels) - min(all_vels)

    # Description
    if expressiveness > 0.7:
        desc = (f"expresividad dinámica muy alta ({mean_var:.1f} std dev) — "
                "interpretación viva, con grandes contrastes de ataque")
    elif expressiveness > 0.4:
        desc = (f"expresividad moderada ({mean_var:.1f} std dev) — "
                "escritura con matices claros pero no extremos")
    else:
        desc = (f"escritura dinámica uniforme ({mean_var:.1f} std dev) — "
                "mecánica, ostinato, o estilo minimalista deliberado")

    if accent_ratio > 0.15:
        desc += f"; acentos estratégicos frecuentes ({accent_ratio*100:.0f}%)"
    if hairpins > 5:
        desc += f"; {hairpins} gestos de crescendo/decrescendo detectados"

    return {
        'expressiveness': expressiveness,
        'mean_variance': mean_var,
        'accent_ratio': accent_ratio,
        'hairpin_count': hairpins,
        'velocity_range': vel_range,
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  5. RESOLUCIÓN DE DISONANCIA EN EL TIEMPO
# ═══════════════════════════════════════════════════════════════

def analyze_dissonance_resolution(notes, key_root: int, mode: str,
                                   resolution: float = 0.25) -> Dict:
    """
    Para cada momento disonante, mide cuánto tiempo tarda en resolverse.
    Clasifica: ornamental (<0.5s), expresivo (0.5-2s),
               estructural (2-6s), irresuelto (>6s o nunca).
    """
    if not notes:
        return {'events': [], 'mean_latency': 0, 'unresolved_ratio': 0, 'desc': ''}

    total = max(n.time_sec + n.duration for n in notes)
    scale = set(SCALE_IVLS.get(mode, SCALE_IVLS['minor']))
    tonic = key_root

    events = []
    t = 0.0
    in_dissonance = False
    dis_start = 0.0
    prev_dis = 0.0

    while t < total:
        active = [n for n in notes if n.time_sec <= t < n.time_sec + n.duration]
        if len(active) < 2:
            if in_dissonance:
                duration = t - dis_start
                events.append({'start': dis_start, 'duration': duration,
                                'resolved': True, 'level': prev_dis})
                in_dissonance = False
            t += resolution; continue

        pitches = [n.pitch % 12 for n in active]
        # Compute dissonance
        dis = 0.0; pairs = 0
        for i in range(len(pitches)):
            for j in range(i+1, min(i+4, len(pitches))):
                dis += DISSONANCE[abs(pitches[i]-pitches[j]) % 12]
                pairs += 1
        dis = dis/pairs if pairs > 0 else 0

        # Chromatic notes add to dissonance
        chromatic = sum(1 for p in pitches if (p-tonic)%12 not in scale)
        dis += chromatic / len(pitches) * 0.3

        threshold = 0.45
        if dis >= threshold and not in_dissonance:
            in_dissonance = True; dis_start = t; prev_dis = dis
        elif dis < threshold * 0.7 and in_dissonance:
            duration = t - dis_start
            events.append({'start': dis_start, 'duration': duration,
                           'resolved': True, 'level': prev_dis})
            in_dissonance = False
        elif in_dissonance:
            prev_dis = max(prev_dis, dis)
        t += resolution

    if in_dissonance:
        events.append({'start': dis_start, 'duration': total - dis_start,
                       'resolved': False, 'level': prev_dis})

    if not events:
        return {'events': [], 'mean_latency': 0, 'unresolved_ratio': 0,
                'ornamental': 0, 'expressive': 0, 'structural': 0, 'unresolved': 0,
                'desc': 'no se detectaron momentos de disonancia significativa'}

    ornamental   = [e for e in events if e['duration'] < 0.5]
    expressive   = [e for e in events if 0.5 <= e['duration'] < 2.0]
    structural   = [e for e in events if 2.0 <= e['duration'] < 6.0]
    unresolved   = [e for e in events if e['duration'] >= 6.0 or not e['resolved']]

    resolved_e   = [e for e in events if e['resolved']]
    mean_latency = sum(e['duration'] for e in resolved_e)/len(resolved_e) if resolved_e else 0
    unres_ratio  = len(unresolved) / len(events)

    parts = []
    if unres_ratio > 0.4:
        parts.append(f"disonancia frecuentemente sin resolver ({unres_ratio*100:.0f}%) — tensión estructural permanente, ansiedad sostenida")
    elif mean_latency < 0.5:
        parts.append("disonancia resuelta casi instantáneamente — ornamental, sin peso emocional")
    elif mean_latency < 2.0:
        parts.append(f"resolución expresiva (μ={mean_latency:.1f}s) — la tensión se sostiene el tiempo justo para sentirse")
    else:
        parts.append(f"resolución lenta (μ={mean_latency:.1f}s) — la disonancia es estructural, pesa, incomoda")

    if structural:
        parts.append(f"{len(structural)} momento(s) de disonancia estructural (2-6s) — puntos de máxima inestabilidad emocional")

    return {
        'events': events,
        'mean_latency': mean_latency,
        'unresolved_ratio': unres_ratio,
        'ornamental': len(ornamental),
        'expressive': len(expressive),
        'structural': len(structural),
        'unresolved': len(unresolved),
        'desc': '; '.join(parts) if parts else 'disonancia moderada y bien resuelta',
    }


# ═══════════════════════════════════════════════════════════════
#  6. SIMETRÍA Y PROPORCIÓN ÁUREA
# ═══════════════════════════════════════════════════════════════

def analyze_golden_ratio(tension_curve, energy_curve, sections,
                          silence_data: Dict, total_dur: float) -> Dict:
    """
    Detecta si los puntos estructurales clave caen cerca de proporciones
    significativas: mitad (0.5), tercio (0.333, 0.666), proporción áurea (0.618).
    """
    if total_dur <= 0:
        return {'golden_climax': False, 'proportions': {}, 'desc': ''}

    golden = GOLDEN_RATIO
    significant_positions = {}

    # Tension peak
    if tension_curve:
        t_vals = [p.tension for p in tension_curve]
        t_peak = tension_curve[t_vals.index(max(t_vals))].time / total_dur
        significant_positions['clímax tensión'] = t_peak

    # Energy peak
    if energy_curve:
        e_vals = [e for _, e in energy_curve]
        e_peak = energy_curve[e_vals.index(max(e_vals))][0] / total_dur
        significant_positions['clímax energía'] = e_peak

    # Most dramatic silence
    md = silence_data.get('most_dramatic')
    if md:
        significant_positions['silencio dramático'] = md['start'] / total_dur

    # Section boundaries
    for s in sections:
        if s.notes and s.start > 0:
            significant_positions[f'sec.{s.index+1}'] = s.start / total_dur

    # Check proximity to golden ratio and other proportions
    targets = {
        'proporción áurea (φ=0.618)': golden,
        'mitad exacta (0.500)':        0.500,
        'tercio (0.333)':              0.333,
        'doble tercio (0.667)':        0.667,
        'cuarto (0.250)':              0.250,
        'tres cuartos (0.750)':        0.750,
    }

    proximity = {}
    for event, pos in significant_positions.items():
        for prop_name, prop_val in targets.items():
            dist = abs(pos - prop_val)
            if dist < 0.05:  # within 5% of total duration
                if prop_name not in proximity:
                    proximity[prop_name] = []
                proximity[prop_name].append((event, pos, dist))

    # Check if climax is near golden ratio
    golden_climax = False
    golden_events = []
    for event, pos in significant_positions.items():
        if abs(pos - golden) < 0.06:
            golden_climax = True
            golden_events.append(f"{event} ({pos*100:.1f}%)")

    parts = []
    if golden_climax:
        parts.append(f"el clímax cae cerca de la proporción áurea ({golden*100:.1f}%) — "
                     f"arquitectura que el oído percibe como naturalmente satisfactoria: {', '.join(golden_events)}")
    else:
        # Find closest structural point to golden ratio
        if significant_positions:
            closest_event = min(significant_positions.items(),
                                key=lambda x: abs(x[1] - golden))
            dist_pct = abs(closest_event[1] - golden) * 100
            parts.append(f"el punto estructural más cercano a φ es '{closest_event[0]}' "
                         f"({closest_event[1]*100:.1f}%), a {dist_pct:.1f}% de la proporción áurea")

    if len(proximity) >= 2:
        parts.append(f"{len(proximity)} puntos estructurales coinciden con proporciones significativas — "
                     "arquitectura temporal cuidadosa")

    return {
        'golden_climax': golden_climax,
        'golden_events': golden_events,
        'positions': significant_positions,
        'proximity': proximity,
        'desc': '; '.join(parts) if parts else 'no se detectan proporciones estructurales especiales',
    }


# ═══════════════════════════════════════════════════════════════
#  7. ENTROPÍA INFORMACIONAL POR SECCIÓN
# ═══════════════════════════════════════════════════════════════

def analyze_information_density(sections, chords) -> Dict:
    """
    Calcula la entropía de Shannon por sección:
    cuánta información nueva introduce cada una.
    Alta entropía = mucho material nuevo, complejidad.
    Baja entropía = repetición, familiaridad.
    """
    if not sections:
        return {'section_entropies': [], 'mean_entropy': 0, 'desc': ''}

    section_entropies = []
    global_pc_seen = set()
    global_rhythm_seen = set()

    for s in sections:
        if not s.notes:
            section_entropies.append({'index': s.index, 'entropy': 0, 'novelty': 0})
            continue

        # Pitch class entropy
        pc_counts = collections.Counter(n.pitch % 12 for n in s.notes)
        total = sum(pc_counts.values())
        pc_probs = [v/total for v in pc_counts.values()]
        pc_entropy = -sum(p * math.log2(p) for p in pc_probs if p > 0)

        # Rhythmic entropy (using quantised IOIs)
        sn = sorted(s.notes, key=lambda n: n.time_sec)
        if len(sn) > 1:
            iois = [round((sn[i+1].time_sec - sn[i].time_sec) * 8) / 8
                    for i in range(len(sn)-1)]
            ioi_counts = collections.Counter(iois)
            ioi_total = sum(ioi_counts.values())
            ioi_probs = [v/ioi_total for v in ioi_counts.values()]
            rhythm_entropy = -sum(p * math.log2(p) for p in ioi_probs if p > 0)
        else:
            rhythm_entropy = 0

        # Novelty: how much new material vs seen before
        section_pcs = set(n.pitch % 12 for n in s.notes)
        new_pcs = len(section_pcs - global_pc_seen)
        novelty = new_pcs / max(len(section_pcs), 1)
        global_pc_seen |= section_pcs

        combined_entropy = (pc_entropy * 0.6 + rhythm_entropy * 0.4)

        section_entropies.append({
            'index': s.index,
            'start': s.start,
            'pc_entropy': pc_entropy,
            'rhythm_entropy': rhythm_entropy,
            'entropy': combined_entropy,
            'novelty': novelty,
        })

    active = [e for e in section_entropies if e.get('entropy', 0) > 0]
    if not active:
        return {'section_entropies': section_entropies, 'mean_entropy': 0,
                'desc': 'insuficientes datos'}

    mean_e = sum(e['entropy'] for e in active) / len(active)
    max_e  = max(e['entropy'] for e in active)
    min_e  = min(e['entropy'] for e in active)
    range_e= max_e - min_e

    if mean_e > 3.0:
        desc = f"alta densidad informacional (μ={mean_e:.2f} bits) — música rica en material, exigente para el oyente"
    elif mean_e > 2.0:
        desc = f"densidad informacional moderada (μ={mean_e:.2f} bits) — equilibrio entre novedad y repetición"
    else:
        desc = f"baja densidad informacional (μ={mean_e:.2f} bits) — escritura económica, minimalista o ostinato"

    if range_e > 1.5:
        desc += f"; gran contraste entre secciones (rango={range_e:.2f} bits) — partes de alta novedad alternan con zonas de consolidación"

    return {
        'section_entropies': section_entropies,
        'mean_entropy': mean_e,
        'max_entropy': max_e,
        'min_entropy': min_e,
        'range_entropy': range_e,
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  8. AMBIGÜEDAD TONAL INTENCIONAL
# ═══════════════════════════════════════════════════════════════

def analyze_tonal_ambiguity(notes, key_root: int, mode: str, chords) -> Dict:
    """
    Detecta herramientas de ambigüedad tonal:
    - Escala de tonos enteros (whole-tone)
    - Escala octatónica (alternates T and S)
    - Acordes pivote (pertenecen a dos tonalidades)
    - Polialtonalidad momentánea
    """
    if not notes:
        return {'whole_tone': 0, 'octatonic': 0, 'pivot_chords': 0,
                'ambiguity_index': 0, 'desc': ''}

    pcs = [n.pitch % 12 for n in notes]
    pc_set = set(pcs)

    # Whole-tone scale: 6 notes, all a tone apart
    whole_tone_sets = [
        {0,2,4,6,8,10},
        {1,3,5,7,9,11},
    ]
    whole_tone_coverage = max(len(pc_set & wt) / max(len(wt), 1) for wt in whole_tone_sets)

    # Octatonic scales: 8 notes, alternating T and S
    octatonic_sets = [
        {0,1,3,4,6,7,9,10},
        {0,2,3,5,6,8,9,11},
        {1,2,4,5,7,8,10,11},
    ]
    octatonic_coverage = max(len(pc_set & oc) / max(len(oc), 1) for oc in octatonic_sets)

    # Pivot chords: count chords whose root belongs to multiple diatonic scales
    scale_major = set(SCALE_IVLS['major'])
    scale_minor = set(SCALE_IVLS['minor'])

    def in_scale(root, key, scale_type):
        return (root - key) % 12 in set(SCALE_IVLS[scale_type])

    pivot_chords = 0
    nearby_keys = [(key_root + i) % 12 for i in [5, 7, 3, 9]]  # closely related keys
    for c in chords:
        keys_match = sum(1 for k in nearby_keys
                         if in_scale(c.root, k, 'major') or in_scale(c.root, k, 'minor'))
        if keys_match >= 2:
            pivot_chords += 1
    pivot_ratio = pivot_chords / max(len(chords), 1)

    # Ambiguity index
    ambiguity_index = (whole_tone_coverage * 0.4 +
                       octatonic_coverage * 0.3 +
                       pivot_ratio * 0.3)

    parts = []
    if whole_tone_coverage > 0.7:
        parts.append(f"fuerte presencia de escala de tonos enteros ({whole_tone_coverage*100:.0f}%) — flotación, irrealidad, Debussy")
    elif whole_tone_coverage > 0.5:
        parts.append(f"escala de tonos enteros parcial ({whole_tone_coverage*100:.0f}%) — color impresionista")

    if octatonic_coverage > 0.7:
        parts.append(f"escala octatónica predominante ({octatonic_coverage*100:.0f}%) — jazz modal, Stravinsky, música cinematográfica")
    elif octatonic_coverage > 0.5:
        parts.append(f"influencia octatónica ({octatonic_coverage*100:.0f}%)")

    if pivot_ratio > 0.3:
        parts.append(f"muchos acordes pivote ({pivot_ratio*100:.0f}%) — tonalidad fluida, modulación constante")

    if ambiguity_index < 0.3:
        parts.append("tonalidad clara y definida — sin ambigüedad intencional significativa")

    return {
        'whole_tone_coverage': whole_tone_coverage,
        'octatonic_coverage': octatonic_coverage,
        'pivot_ratio': pivot_ratio,
        'ambiguity_index': ambiguity_index,
        'desc': '; '.join(parts) if parts else 'tonalidad estable, sin recursos de ambigüedad especiales',
    }


# ═══════════════════════════════════════════════════════════════
#  9. INTENSIDAD PERCEPTUAL (PSICOFONÍA)
# ═══════════════════════════════════════════════════════════════

def compute_perceptual_intensity(notes, total_dur: float,
                                  resolution: float = 0.4) -> Dict:
    """
    Modela el impacto perceptual considerando no-linealidades del oído:
    - El oído es más sensible en el rango 2-4kHz (MIDI 69-81)
    - Notas agudas parecen más intensas que graves a igual dinámica
    - Adaptación auditiva: notas largas pierden intensidad percibida
    - Enmascaramiento: muchas notas simultáneas reducen claridad individual
    """
    if not notes:
        return {'curve': [], 'peak_time': 0, 'mean': 0, 'desc': ''}

    curve = []
    t = 0.0
    while t < total_dur:
        active = [n for n in notes if n.time_sec < t + resolution and n.time_sec + n.duration > t]
        if not active:
            curve.append((t, 0.0)); t += resolution; continue

        intensity = 0.0
        for n in active:
            # Base loudness (MIDI velocity → perceived loudness, roughly logarithmic)
            base = (n.velocity / 127.0) ** 0.6  # Stevens' power law

            # Pitch sensitivity curve (peak around MIDI 75)
            pitch_factor = math.exp(-((n.pitch - 75)**2) / (2 * 18**2)) * 0.4 + 0.6

            # Duration fatigue: notes >2s lose ~20% perceived intensity
            time_in_note = t - n.time_sec
            fatigue = 1.0 - min(time_in_note / 10.0, 0.3)

            # Overlap fraction
            overlap = (min(n.time_sec + n.duration, t + resolution) -
                       max(n.time_sec, t)) / resolution

            intensity += base * pitch_factor * fatigue * overlap

        # Masking: many simultaneous notes reduce individual clarity
        if len(active) > 4:
            masking_factor = 1.0 - min((len(active) - 4) * 0.05, 0.4)
        else:
            masking_factor = 1.0

        curve.append((t, intensity * masking_factor))
        t += resolution

    values = [v for _, v in curve]
    if not values:
        return {'curve': curve, 'peak_time': 0, 'mean': 0, 'desc': ''}

    peak_idx  = values.index(max(values))
    peak_time = curve[peak_idx][0]
    mean_i    = sum(values) / len(values)
    max_i     = max(values)

    # Shape
    thirds = len(values) // 3
    i1 = sum(values[:thirds]) / thirds if thirds else 0
    i2 = sum(values[thirds:2*thirds]) / thirds if thirds else 0
    i3 = sum(values[2*thirds:]) / max(1, len(values)-2*thirds)

    if i3 > i1 + i2 * 0.3:
        shape = 'acumulativa — la percepción del oyente crece hacia el final'
    elif i1 > i3 + i2 * 0.3:
        shape = 'decreciente — el mayor impacto perceptual es al inicio'
    elif i2 > i1 + i3:
        shape = 'en arco — el impacto perceptual máximo es central'
    else:
        shape = 'uniforme — la intensidad perceptual se distribuye equilibradamente'

    return {
        'curve': curve,
        'peak_time': peak_time,
        'peak_val': max_i,
        'mean': mean_i,
        'shape': shape,
        'desc': f"Perfil perceptual {shape}. Pico en {ts_str(peak_time)}.",
    }


# ═══════════════════════════════════════════════════════════════
#  10. VOZ INTERNA MÁS EXPRESIVA
# ═══════════════════════════════════════════════════════════════

def find_inner_voice(notes) -> Dict:
    """
    Identifica la voz interna más melódicamente rica
    (no la más aguda ni la más grave).
    """
    if not notes:
        return {'voice': None, 'expressiveness': 0, 'desc': ''}

    # Group by channel or track
    groups = collections.defaultdict(list)
    for n in notes:
        groups[(n.channel, n.track)].append(n)

    if len(groups) < 3:
        return {'voice': None, 'expressiveness': 0,
                'desc': 'menos de 3 voces — voz interna no aplica'}

    # Sort by average pitch
    sorted_groups = sorted(groups.items(),
                           key=lambda x: sum(n.pitch for n in x[1])/len(x[1]))
    # Inner voices: everything except highest and lowest
    inner_voices = sorted_groups[1:-1]

    if not inner_voices:
        return {'voice': None, 'expressiveness': 0, 'desc': 'sin voces internas'}

    best_key = None; best_score = -1
    for key, voice_notes in inner_voices:
        sn = sorted(voice_notes, key=lambda n: n.time_sec)
        if len(sn) < 4:
            continue
        ivs = [abs(sn[i+1].pitch - sn[i].pitch) for i in range(len(sn)-1)]
        # Score: variety of intervals + velocity range
        iv_variety = len(set(ivs)) / max(len(ivs), 1)
        vel_range  = max(n.velocity for n in sn) - min(n.velocity for n in sn)
        pitch_range= max(n.pitch for n in sn) - min(n.pitch for n in sn)
        score = iv_variety * 0.4 + vel_range/127 * 0.3 + pitch_range/48 * 0.3
        if score > best_score:
            best_score = score; best_key = key

    if best_key is None:
        return {'voice': None, 'expressiveness': 0,
                'desc': 'no se pudo identificar una voz interna destacada'}

    best_voice = groups[best_key]
    avg_pitch = sum(n.pitch for n in best_voice) / len(best_voice)

    desc = (f"voz interna en registro {NOTE_NAMES[int(avg_pitch)%12]}{int(avg_pitch)//12-1} "
            f"(expresividad={best_score:.2f}) — "
            f"{'portadora del contenido emocional más sutil' if best_score > 0.5 else 'voz de relleno armónico con matices melódicos'}")

    return {
        'voice': best_key,
        'avg_pitch': avg_pitch,
        'expressiveness': best_score,
        'note_count': len(best_voice),
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  11. TROPOS CINEMATOGRÁFICOS
# ═══════════════════════════════════════════════════════════════

def detect_cinematic_tropes(notes, tension_curve, energy_curve,
                             chords, silence_data: Dict,
                             bpm: float, total_dur: float) -> Dict:
    """
    Detecta tropos emocionales cinematográficos:
    - Stinger: acorde súbito de impacto
    - Swell: crescendo hacia climax
    - Leitmotif: tema recurrente
    - Underscoring: música descriptiva, dinámica uniforme baja
    - Pedal drone: nota sostenida de fondo
    - Heartbeat: pulso rítmico regular y sugestivo
    - Silence before strike: silencio → impacto súbito
    """
    tropes = {}
    beat = 60.0 / bpm if bpm > 0 else 0.5
    sn   = sorted(notes, key=lambda n: n.time_sec)

    # STINGER: sudden loud note/chord after soft passage
    stingers = []
    for i in range(2, len(sn)):
        prev_mean = sum(sn[j].velocity for j in range(max(0,i-5),i)) / min(5,i)
        if sn[i].velocity > prev_mean + 35 and sn[i].velocity > 90:
            stingers.append(ts_str(sn[i].time_sec))
    if stingers:
        tropes['Stinger (impacto súbito)'] = {
            'count': len(stingers),
            'times': stingers[:3],
            'desc': 'acorde/nota súbita de alto impacto tras pasaje suave — shock emocional'
        }

    # SWELL: sustained crescendo lasting >4 beats
    if energy_curve:
        e_vals = [e for _, e in energy_curve]
        e_times= [t for t,_ in energy_curve]
        window = max(1, int(beat * 4 / 0.4))
        swells = []
        for i in range(window, len(e_vals)):
            slope = (e_vals[i] - e_vals[i-window]) / window
            if slope > 0.05 and e_vals[i] > e_vals[i-window] * 1.5:
                swells.append(ts_str(e_times[i-window]))
        if swells:
            tropes['Swell (crescendo cinematográfico)'] = {
                'count': len(swells),
                'times': swells[:3],
                'desc': 'crescendo sostenido hacia un clímax — anticipación, construcción de tensión'
            }

    # UNDERSCORING: very uniform low dynamics throughout
    all_vels = [n.velocity for n in sn]
    if all_vels:
        vel_range = max(all_vels) - min(all_vels)
        mean_vel  = sum(all_vels) / len(all_vels)
        if vel_range < 25 and mean_vel < 65:
            tropes['Underscoring (música de fondo)'] = {
                'count': 1,
                'desc': f'dinámica uniforme y baja (rango={vel_range}, μ={mean_vel:.0f}) — música descriptiva de fondo, sin protagonismo propio'
            }

    # PEDAL DRONE: single pitch class sustained for >8s
    pedal_duration = {}
    for i, n in enumerate(sn):
        pc = n.pitch % 12
        if pc not in pedal_duration:
            pedal_duration[pc] = 0
        pedal_duration[pc] += n.duration
    longest_pedal = max(pedal_duration.items(), key=lambda x: x[1])
    if longest_pedal[1] > 8.0:
        tropes['Pedal Drone (bordón sostenido)'] = {
            'note': NOTE_NAMES[longest_pedal[0]],
            'duration': longest_pedal[1],
            'desc': f"nota {NOTE_NAMES[longest_pedal[0]]} sostenida ~{longest_pedal[1]:.0f}s — hipnosis, tensión de fondo, identidad tonal"
        }

    # HEARTBEAT: regular low-register pulse
    bass_sn = [n for n in sn if n.pitch < 55]
    if len(bass_sn) > 8:
        bass_iois = [bass_sn[i+1].time_sec - bass_sn[i].time_sec
                     for i in range(len(bass_sn)-1) if bass_sn[i+1].time_sec - bass_sn[i].time_sec < beat*2]
        if bass_iois:
            mean_ioi = sum(bass_iois)/len(bass_iois)
            regularity = 1 - sum(abs(x-mean_ioi) for x in bass_iois)/(mean_ioi*len(bass_iois)+0.001)
            if regularity > 0.7 and 0.4 < mean_ioi < 1.2:
                tropes['Heartbeat (pulso cardíaco)'] = {
                    'bpm_equiv': 60/mean_ioi,
                    'regularity': regularity,
                    'desc': f"pulso bajo regular (~{60/mean_ioi:.0f} BPM, regularidad={regularity:.2f}) — tensión visceral, urgencia física"
                }

    # SILENCE BEFORE STRIKE: silence >1s immediately followed by loud notes
    silences = silence_data.get('silences', [])
    sbs_events = []
    for sil in silences:
        if sil['duration'] > 1.0:
            after = [n for n in sn if sil['start'] + sil['duration'] <= n.time_sec < sil['start'] + sil['duration'] + 0.5]
            if after and max(n.velocity for n in after) > 85:
                sbs_events.append(ts_str(sil['start'] + sil['duration']))
    if sbs_events:
        tropes['Silence Before Strike (silencio→golpe)'] = {
            'count': len(sbs_events),
            'times': sbs_events[:3],
            'desc': 'silencio seguido de entrada brusca y fuerte — máxima sorpresa, impacto dramático'
        }

    if not tropes:
        return {'tropes': {}, 'count': 0,
                'desc': 'no se detectaron tropos cinematográficos específicos — escritura más abstracta o camerística'}

    return {
        'tropes': tropes,
        'count': len(tropes),
        'desc': f"{len(tropes)} tropo(s) cinematográfico(s) detectado(s): {', '.join(tropes.keys())}"
    }


# ═══════════════════════════════════════════════════════════════
#  12. FINGERPRINTING ARMÓNICO (firma del compositor)
# ═══════════════════════════════════════════════════════════════

def compute_harmonic_fingerprint(chords, key_root: int, mode: str) -> Dict:
    """
    Calcula la 'huella digital' del estilo armónico:
    - Ratio maj/min
    - Uso de acordes de 7ª y extensiones
    - Preferencia por movimientos de 3ª vs 5ª
    - Frecuencia de dominantes secundarias
    - Cromatismo armónico
    """
    if not chords:
        return {'fingerprint': {}, 'style_tags': [], 'desc': ''}

    total = len(chords)
    scale = set(SCALE_IVLS.get(mode, SCALE_IVLS['minor']))

    maj_count  = sum(1 for c in chords if c.chord_type in ('maj','maj7','add9','maj9'))
    min_count  = sum(1 for c in chords if c.chord_type in ('min','min7','min9'))
    ext_count  = sum(1 for c in chords if '7' in c.chord_type or '9' in c.chord_type)
    sus_count  = sum(1 for c in chords if 'sus' in c.chord_type)
    dim_count  = sum(1 for c in chords if 'dim' in c.chord_type)
    chrom_count= sum(1 for c in chords if (c.root-key_root)%12 not in scale)

    # Root movement preferences
    thirds_move = 0; fifths_move = 0; semi_move = 0; step_move = 0
    for i in range(len(chords)-1):
        mv = abs((chords[i+1].root - chords[i].root + 6) % 12 - 6)
        if mv in (3,4): thirds_move += 1
        elif mv in (5,7): fifths_move += 1
        elif mv == 1: semi_move += 1
        elif mv == 2: step_move += 1

    moves_total = thirds_move + fifths_move + semi_move + step_move + 0.001

    fp = {
        'maj_ratio':    maj_count / total,
        'min_ratio':    min_count / total,
        'extension_ratio': ext_count / total,
        'suspended_ratio': sus_count / total,
        'diminished_ratio': dim_count / total,
        'chromatic_ratio': chrom_count / total,
        'thirds_preference': thirds_move / moves_total,
        'fifths_preference': fifths_move / moves_total,
        'semitone_preference': semi_move / moves_total,
    }

    # Style tags
    tags = []
    if fp['extension_ratio'] > 0.4:
        tags.append('jazz/contemporáneo (uso intensivo de extensiones)')
    if fp['suspended_ratio'] > 0.2:
        tags.append('modal/ambient (sus chords frecuentes)')
    if fp['fifths_preference'] > 0.4:
        tags.append('tonal clásico (movimientos de 5ª dominantes)')
    if fp['thirds_preference'] > 0.35:
        tags.append('romántico/moderno (movimientos de 3ª frecuentes)')
    if fp['semitone_preference'] > 0.2:
        tags.append('cromatismo expresivo (movimientos semitonales)')
    if fp['chromatic_ratio'] > 0.3:
        tags.append('armonía extendida/modal (acordes fuera de escala)')
    if fp['diminished_ratio'] > 0.1:
        tags.append('tardorromántico/tenso (uso de acordes disminuidos)')

    if not tags:
        tags.append('diatónico estable')

    return {
        'fingerprint': fp,
        'style_tags': tags,
        'desc': 'Firma armónica: ' + '; '.join(tags),
    }


# ═══════════════════════════════════════════════════════════════
#  13. MOMENTOS DE ETERNIDAD
# ═══════════════════════════════════════════════════════════════

def detect_eternity_moments(notes, tension_curve, energy_curve,
                             silence_data: Dict, bpm: float,
                             total_dur: float) -> Dict:
    """
    Detecta instantes donde el tiempo subjetivo se dilata:
    - Notas largas sobre pedal
    - Repetición hipnótica (después del análisis de groove)
    - Silencio post-densidad
    - Convergencia de baja tensión + baja energía + nota larga
    """
    if not notes:
        return {'moments': [], 'desc': ''}

    beat = 60.0 / bpm if bpm > 0 else 0.5
    sn   = sorted(notes, key=lambda n: n.time_sec)
    moments = []

    # 1. Long sustained notes (>3 beats) with few simultaneous voices
    for n in sn:
        if n.duration > beat * 3:
            # Check how many simultaneous notes
            simultaneous = [m for m in sn if abs(m.time_sec - n.time_sec) < 0.1
                           and m is not n]
            if len(simultaneous) <= 2:
                moments.append({
                    'time': n.time_sec,
                    'type': 'nota suspendida',
                    'duration': n.duration,
                    'desc': f"nota {NOTE_NAMES[n.pitch%12]} sostenida {n.duration:.1f}s — tiempo dilatado"
                })

    # 2. Silence after dense passage
    silences = silence_data.get('silences', [])
    for sil in silences:
        if sil['duration'] > 1.5:
            # Check density before the silence
            before_notes = [n for n in sn
                           if sil['start'] - 3 <= n.time_sec < sil['start']]
            if len(before_notes) > 10:
                moments.append({
                    'time': sil['start'],
                    'type': 'silencio post-clímax',
                    'duration': sil['duration'],
                    'desc': f"silencio de {sil['duration']:.1f}s tras pasaje denso — suspensión total, el tiempo se detiene"
                })

    # 3. Low tension + low energy convergence
    if tension_curve and energy_curve:
        t_vals = [p.tension for p in tension_curve]
        e_vals = [e for _, e in energy_curve]
        # Normalise energy
        max_e = max(e_vals) if e_vals else 1
        e_norm = [e/max_e for e in e_vals]

        for i, tp in enumerate(tension_curve):
            e_idx = min(int(tp.time / 0.4), len(e_norm)-1)
            e = e_norm[e_idx]
            if tp.tension < 0.2 and e < 0.2 and tp.time > beat * 4:
                moments.append({
                    'time': tp.time,
                    'type': 'reposo total',
                    'duration': 0.4,
                    'desc': f"convergencia de baja tensión y baja energía — quietud completa, meditación"
                })

    # Deduplicate (keep one per 4-second window)
    moments.sort(key=lambda m: m['time'])
    deduped = []
    last_t = -999
    for m in moments:
        if m['time'] - last_t > 4.0:
            deduped.append(m)
            last_t = m['time']

    deduped = deduped[:8]  # max 8 moments

    if not deduped:
        desc = 'no se detectaron momentos de dilatación temporal — música de flujo continuo sin pausas contemplativas'
    else:
        desc = f"{len(deduped)} momento(s) de eternidad — instantes donde el tiempo subjetivo se dilata"

    return {'moments': deduped, 'count': len(deduped), 'desc': desc}


# ═══════════════════════════════════════════════════════════════
#  14. COHERENCIA EMOCIONAL INTERNA
# ═══════════════════════════════════════════════════════════════

def analyze_emotional_coherence(sections, sva_list, mode: str, avg_bpm: float,
                                 tension_curve, cadences: Dict) -> Dict:
    """
    Mide si la pieza mantiene su intención emocional de forma consistente
    o si contiene contradicciones expresivas deliberadas.
    """
    active = [(s, va) for s, va in zip(sections, sva_list) if s.notes]
    if len(active) < 2:
        return {'coherence': 1.0, 'contradictions': [], 'desc': ''}

    contradictions = []

    # Check mode vs rhythm contradiction
    if mode in ('minor','phrygian') and avg_bpm > 140:
        contradictions.append({
            'type': 'modo-tempo',
            'desc': f'modo oscuro ({mode}) con tempo muy rápido ({avg_bpm:.0f} BPM) — tensión expresiva entre melancolía y urgencia'
        })

    # Check VA consistency across sections
    va_vals = [(v, a) for v, a, _ in sva_list if _ != 'sin notas']
    if va_vals:
        v_vals = [v for v,_ in va_vals]
        a_vals = [a for _,a in va_vals]
        v_range = max(v_vals) - min(v_vals)
        a_range = max(a_vals) - min(a_vals)

        if v_range > 1.0:
            contradictions.append({
                'type': 'valencia-variable',
                'desc': f'la valencia emocional oscila fuertemente (rango={v_range:.2f}) — pieza de contrastes emocionales extremos'
            })
        if a_range > 1.2:
            contradictions.append({
                'type': 'activación-variable',
                'desc': f'la activación varía drásticamente (rango={a_range:.2f}) — alternancia entre calma y euforia'
            })

    # Check tension vs dynamics contradiction
    for s, (v, a, _) in active:
        t_vals_sec = [p.tension for p in tension_curve if s.start <= p.time < s.end]
        if t_vals_sec:
            avg_t = sum(t_vals_sec)/len(t_vals_sec)
            if avg_t > 0.5 and s.avg_velocity < 45:
                contradictions.append({
                    'type': 'tensión-dinámica',
                    'desc': f'sección {s.index+1}: alta tensión armónica con dinámica suave — máxima inquietud contenida, clásico de la música de cámara contemporánea'
                })
            elif avg_t < 0.15 and s.avg_velocity > 85:
                contradictions.append({
                    'type': 'consonancia-forte',
                    'desc': f'sección {s.index+1}: armonía consonante pero muy fuerte — poder sin angustia, afirmación luminosa'
                })

    # Coherence score: fewer contradictions = more coherent
    raw_coherence = 1.0 - min(len(contradictions) * 0.15, 0.6)

    if raw_coherence > 0.85:
        desc = 'alta coherencia emocional — la pieza mantiene su intención expresiva de forma consistente'
    elif raw_coherence > 0.6:
        desc = f'coherencia moderada con {len(contradictions)} contradicción(es) expresiva(s) — ambivalencia deliberada'
    else:
        desc = f'múltiples contradicciones expresivas ({len(contradictions)}) — pieza de alta complejidad emocional o carácter experimental'

    return {
        'coherence': raw_coherence,
        'contradictions': contradictions,
        'contradiction_count': len(contradictions),
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  SELF-SIMILARITY MATRIX (SSM)
# ═══════════════════════════════════════════════════════════════

def compute_ssm(notes: List[NoteEvent], total_dur: float,
                resolution: float = 4.0) -> Dict:
    """
    Calcula la Self-Similarity Matrix (SSM) de la pieza.

    Divide la pieza en segmentos de `resolution` segundos, representa cada
    segmento como un vector de clase de pitch (chroma) ponderado por velocidad
    y duración, y compara todos los pares por similitud coseno.

    La SSM revela la estructura formal real de la pieza:
      - La diagonal principal siempre es 1.0 (cada segmento es idéntico a sí mismo).
      - Diagonales paralelas = secciones repetidas (A-A', estrofas, etc.)
      - Bloques de alta similitud = secciones con material relacionado
      - Asimetría = contraste, desarrollo, no hay retorno al material original

    Devuelve:
      matrix        : lista de listas NxN de similitud coseno [0..1]
      n             : número de segmentos
      segment_times : lista de tiempos de inicio de cada segmento
      form_labels   : etiquetas de forma musical detectadas (A, B, C...)
      repetitions   : lista de pares de segmentos muy similares
      contrasts     : lista de pares de segmentos muy distintos
      symmetry      : índice de simetría espejo (0=asimétrica, 1=palindrómica)
      block_structure: lista de bloques homogéneos detectados
      desc          : descripción en lenguaje natural
    """
    if not notes or total_dur <= 0:
        return {
            'matrix': [], 'n': 0, 'segment_times': [],
            'form_labels': [], 'repetitions': [], 'contrasts': [],
            'symmetry': 0.0, 'block_structure': [],
            'ascii': '', 'desc': 'insuficientes datos para SSM'
        }

    # ── 1. Construir vectores chroma por segmento ─────────────────
    n_segs = max(2, int(math.ceil(total_dur / resolution)))
    seg_times = [i * resolution for i in range(n_segs)]
    chromas = []

    for i in range(n_segs):
        t0 = i * resolution
        t1 = t0 + resolution
        window = [n for n in notes if n.time_sec < t1 and n.time_sec + n.duration > t0]
        vec = [0.0] * 12
        for n in window:
            # Overlap between note and window
            overlap = min(n.time_sec + n.duration, t1) - max(n.time_sec, t0)
            weight = (n.velocity / 127.0) * max(overlap, 0.0)
            vec[n.pitch % 12] += weight
        # Normalise to unit vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        chromas.append(vec)

    # ── 2. Matriz de similitud coseno ────────────────────────────
    matrix = []
    for i in range(n_segs):
        row = []
        for j in range(n_segs):
            dot = sum(chromas[i][k] * chromas[j][k] for k in range(12))
            # Clamp to [0, 1] (cosine with non-negative weights is already ≥ 0)
            row.append(max(0.0, min(1.0, dot)))
        matrix.append(row)

    # ── 3. Detección de repeticiones y contrastes ─────────────────
    REPEAT_THRESH  = 0.80   # umbral: segmentos muy similares
    CONTRAST_THRESH = 0.25  # umbral: segmentos muy distintos

    repetitions = []
    contrasts   = []
    for i in range(n_segs):
        for j in range(i + 1, n_segs):
            sim = matrix[i][j]
            if sim >= REPEAT_THRESH:
                repetitions.append({
                    'seg_a': i, 'seg_b': j,
                    'time_a': seg_times[i], 'time_b': seg_times[j],
                    'similarity': sim,
                })
            elif sim <= CONTRAST_THRESH:
                contrasts.append({
                    'seg_a': i, 'seg_b': j,
                    'time_a': seg_times[i], 'time_b': seg_times[j],
                    'similarity': sim,
                })

    # ── 4. Etiquetado de forma: clustering por similitud ─────────
    # Asigna una letra (A, B, C...) a cada segmento usando greedy clustering.
    form_labels = ['?'] * n_segs
    label_chr   = ord('A')
    assigned    = {}  # segmento → label

    for i in range(n_segs):
        if i in assigned:
            form_labels[i] = assigned[i]
            continue
        # Buscar si se parece a algún segmento ya etiquetado
        matched = None
        for ref_idx, ref_label in assigned.items():
            if matrix[i][ref_idx] >= REPEAT_THRESH:
                matched = ref_label
                break
        if matched:
            form_labels[i] = matched
            assigned[i]    = matched
        else:
            new_label       = chr(label_chr)
            form_labels[i]  = new_label
            assigned[i]     = new_label
            label_chr = ord('A') + (label_chr - ord('A') + 1) % 26

    # ── 5. Estructura de bloques ──────────────────────────────────
    block_structure = []
    if n_segs > 0:
        cur_label = form_labels[0]
        cur_start = seg_times[0]
        for i in range(1, n_segs):
            if form_labels[i] != cur_label:
                block_structure.append({
                    'label': cur_label,
                    'start': cur_start,
                    'end':   seg_times[i],
                    'n_segments': i - seg_times.index(cur_start) if cur_start in seg_times else 1,
                })
                cur_label = form_labels[i]
                cur_start = seg_times[i]
        # Último bloque
        block_structure.append({
            'label': cur_label,
            'start': cur_start,
            'end':   total_dur,
            'n_segments': 1,
        })

    # ── 6. Simetría espejo ────────────────────────────────────────
    # Compara la primera mitad con la segunda mitad invertida
    half = n_segs // 2
    if half >= 2:
        sym_vals = []
        for k in range(half):
            mirror_idx = n_segs - 1 - k
            if mirror_idx < n_segs:
                sym_vals.append(matrix[k][mirror_idx])
        symmetry = sum(sym_vals) / len(sym_vals) if sym_vals else 0.0
    else:
        symmetry = 0.0

    # ── 7. Descripción en lenguaje natural ───────────────────────
    unique_labels  = list(dict.fromkeys(form_labels))  # orden de aparición
    n_unique       = len(set(unique_labels))
    form_str       = '-'.join(form_labels)

    # Detectar forma canónica
    form_canon = ''
    label_seq  = form_labels
    if n_segs >= 3:
        if label_seq[0] == label_seq[-1] and n_unique == 2:
            form_canon = 'ABA (ternaria) — forma clásica con retorno al material inicial'
        elif n_unique == 1:
            form_canon = 'forma estrófica — todo el material es homogéneo'
        elif n_unique == n_segs:
            form_canon = 'through-composed — cada sección es única, sin retornos'
        elif len(repetitions) > n_segs * 0.3:
            form_canon = 'forma con variaciones — material recurrente con transformaciones'
        else:
            form_canon = f'forma libre con {n_unique} célula(s) de material distinto'

    rep_desc = ''
    if repetitions:
        n_rep = len(repetitions)
        rep_desc = (f'{n_rep} par(es) de secciones muy similares (similitud ≥ {REPEAT_THRESH:.0%}) — '
                    f'material temático recurrente')
    else:
        rep_desc = 'sin repeticiones literales detectadas — cada sección aporta material nuevo'

    sym_desc = ''
    if symmetry > 0.75:
        sym_desc = 'alta simetría espejo — la pieza es casi palindrómica'
    elif symmetry > 0.5:
        sym_desc = 'simetría espejo moderada — la segunda mitad recuerda a la primera invertida'
    else:
        sym_desc = 'sin simetría espejo — la pieza es asimétrica, narrativa lineal'

    desc_parts = [form_canon, rep_desc, sym_desc]
    desc = ' | '.join(p for p in desc_parts if p)

    # ── 8. Visualización ASCII ────────────────────────────────────
    ascii_art = _ssm_ascii(matrix, form_labels, seg_times, n_segs)

    return {
        'matrix':         matrix,
        'n':              n_segs,
        'segment_times':  seg_times,
        'form_labels':    form_labels,
        'form_str':       form_str,
        'form_canon':     form_canon,
        'unique_labels':  unique_labels,
        'n_unique':       n_unique,
        'repetitions':    repetitions,
        'contrasts':      contrasts,
        'symmetry':       symmetry,
        'block_structure': block_structure,
        'ascii':          ascii_art,
        'desc':           desc,
    }


def _ssm_ascii(matrix: list, form_labels: list,
               seg_times: list, n: int,
               max_size: int = 32) -> str:
    """
    Renderiza la SSM como arte ASCII en escala de grises.
    Usa downsampling si la matriz es mayor que max_size × max_size.

    Leyenda de densidad (similitud coseno):
      █  ≥ 0.90   ▓  ≥ 0.70   ▒  ≥ 0.50   ░  ≥ 0.30   ·  < 0.30
    """
    if n == 0:
        return ''

    # Downsampling
    if n > max_size:
        step = n / max_size
        indices = [int(i * step) for i in range(max_size)]
    else:
        indices = list(range(n))

    m = len(indices)

    # Mapa de glifos
    def glyph(v: float) -> str:
        if v >= 0.90: return '█'
        if v >= 0.70: return '▓'
        if v >= 0.50: return '▒'
        if v >= 0.30: return '░'
        return '·'

    # Eje X: etiquetas de forma (una por columna, truncadas)
    col_labels = [form_labels[i] if i < len(form_labels) else '?' for i in indices]
    x_axis     = '  ' + ' '.join(col_labels)

    # Separador con tiempos (inicio + fin)
    t_start = ts_str_local(seg_times[indices[0]])
    t_end   = ts_str_local(seg_times[indices[-1]])
    x_time  = f"  {t_start}{' ' * max(0, m - len(t_start) - len(t_end))}{t_end}"

    lines = []
    lines.append("  SSM — Self-Similarity Matrix (similitud de chroma por segmento)")
    lines.append("")
    lines.append(x_axis)
    lines.append("  ┌" + "─" * (m * 2 - 1) + "┐")

    for row_idx, ri in enumerate(indices):
        row_label = form_labels[ri] if ri < len(form_labels) else '?'
        cells = [glyph(matrix[ri][ci]) for ci in indices]
        lines.append(f"{row_label} │{' '.join(cells)}│")

    lines.append("  └" + "─" * (m * 2 - 1) + "┘")
    lines.append(x_time)
    lines.append("")
    lines.append("  Leyenda: █ ≥0.90  ▓ ≥0.70  ▒ ≥0.50  ░ ≥0.30  · <0.30")

    return '\n'.join(lines)


def _format_ssm_report(ssm: Dict, total_dur: float) -> str:
    """Formatea la sección SSM para el informe de texto."""
    if not ssm or ssm.get('n', 0) == 0:
        return ''

    lines = []
    SEP = '─' * 66
    lines.append(SEP)
    lines.append("  SSM — SELF-SIMILARITY MATRIX / ESTRUCTURA FORMAL REAL")
    lines.append(SEP)
    lines.append("")

    # ASCII matrix
    if ssm.get('ascii'):
        for line in ssm['ascii'].split('\n'):
            lines.append('  ' + line)
        lines.append("")

    # Forma detectada
    form_str   = ssm.get('form_str', '')
    form_canon = ssm.get('form_canon', '')
    n_unique   = ssm.get('n_unique', 1)

    lines.append(f"  Secuencia de secciones : {form_str}")
    lines.append(f"  Forma detectada        : {form_canon}")
    lines.append(f"  Materiales distintos   : {n_unique} célula(s) temática(s)")
    lines.append("")

    # Bloques
    blocks = ssm.get('block_structure', [])
    if blocks:
        lines.append("  Estructura de bloques:")
        for b in blocks:
            t0 = ts_str_local(b['start'])
            t1 = ts_str_local(b['end'])
            dur = b['end'] - b['start']
            pct = dur / total_dur * 100 if total_dur > 0 else 0
            lines.append(f"    [{b['label']}]  {t0} → {t1}  ({dur:.1f}s, {pct:.0f}%)")
        lines.append("")

    # Repeticiones
    reps = ssm.get('repetitions', [])
    if reps:
        lines.append(f"  Secciones repetidas ({len(reps)} par(es)):")
        for r in reps[:6]:  # máximo 6
            ta = ts_str_local(r['time_a'])
            tb = ts_str_local(r['time_b'])
            lines.append(f"    {ta} ↔ {tb}  (sim={r['similarity']:.2f})")
        lines.append("")
    else:
        lines.append("  Sin repeticiones literales — material completamente renovado.")
        lines.append("")

    # Simetría
    sym = ssm.get('symmetry', 0.0)
    sym_desc = ssm.get('desc', '').split(' | ')[-1] if ssm.get('desc') else ''
    lines.append(f"  Simetría espejo        : {sym:.2f}  —  {sym_desc}")
    lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  v9-A. CURVA DE VALENCIA DINÁMICA CONTINUA
# ═══════════════════════════════════════════════════════════════

# Valence contributions by mode (local, per window)
_MODE_VALENCE = {
    'major': 0.7, 'lydian': 0.8, 'mixolydian': 0.5,
    'dorian': 0.2, 'minor': -0.5, 'phrygian': -0.7, 'locrian': -0.9,
}
# Pitch-class valence (distance from major triad of key)
_PC_VALENCE = {0: 0.5, 2: 0.3, 4: 0.6, 5: 0.2, 7: 0.5,
               9: 0.1, 11: -0.4, 1: -0.6, 3: -0.3, 6: -0.8, 8: -0.2, 10: -0.1}

def compute_dynamic_valence(
    notes: List[NoteEvent],
    tension_curve: List[TensionPoint],
    key_root: int,
    mode: str,
    total_dur: float,
    resolution: float = 1.0,
) -> Dict:
    """
    Calcula la curva de valencia emocional momento a momento.

    Combina 6 dimensiones en un score continuo [-1, +1]:
      1. Modo local (ventana deslizante KS)
      2. Distancia de las notas a la tríada mayor de la tónica
      3. Tensión armónica inversa (alta tensión → valencia más oscura)
      4. Registro: notas agudas tienden a mayor luminosidad
      5. Dinámica: forte activa, piano suave pero puede ser oscuro
      6. Densidad: silencio y notas muy escasas = valencia indefinida

    Devuelve:
      curve        : lista de (time, valence) con valence ∈ [-1, +1]
      mean_valence : promedio global
      peaks_light  : momentos más luminosos
      peaks_dark   : momentos más oscuros
      transitions  : cambios bruscos de valencia (±0.4 en <2s)
      volatility   : desviación estándar de la curva
      arc_shape    : forma del arco de valencia ('ascendente','descendente','cúpula','valle','estable','sinusoidal')
      desc         : descripción en lenguaje natural
    """
    if not notes or total_dur <= 0:
        return {'curve': [], 'mean_valence': 0.0, 'peaks_light': [],
                'peaks_dark': [], 'transitions': [], 'volatility': 0.0,
                'arc_shape': 'indeterminado', 'desc': 'sin datos'}

    sn = sorted(notes, key=lambda n: n.time_sec)
    t_map = {p.time: p.tension for p in tension_curve}
    t_times = sorted(t_map.keys())

    def get_tension(t):
        if not t_times: return 0.3
        closest = min(t_times, key=lambda x: abs(x - t))
        return t_map[closest]

    curve = []
    t = 0.0
    while t < total_dur:
        window = [n for n in sn if n.time_sec < t + resolution and n.time_sec + n.duration > t]

        if not window:
            curve.append((t, 0.0))
            t += resolution
            continue

        # 1. Local mode valence
        pc = [0.0] * 12
        for n in window:
            overlap = min(n.time_sec + n.duration, t + resolution) - max(n.time_sec, t)
            pc[n.pitch % 12] += (n.velocity / 127.0) * max(overlap, 0)
        tot = sum(pc)
        if tot > 0:
            pc_norm = [v / tot for v in pc]
            best_r, best_m, best_c = 0, mode, -999
            for mn, prof in KS_MODES.items():
                for rot in range(12):
                    r = pearson(pc_norm, prof[rot:] + prof[:rot])
                    if r > best_c:
                        best_c, best_r, best_m = r, rot, mn
            mode_val = _MODE_VALENCE.get(best_m, 0.0)
        else:
            mode_val = _MODE_VALENCE.get(mode, 0.0)

        # 2. Pitch-class valence relative to tonic
        avg_pc_val = sum(
            _PC_VALENCE.get((n.pitch % 12 - key_root) % 12, 0.0)
            for n in window
        ) / len(window)

        # 3. Tension → darker valence
        tension = get_tension(t)
        tension_val = -tension * 0.6  # high tension pulls valence negative

        # 4. Register: higher notes = brighter
        avg_pitch = sum(n.pitch for n in window) / len(window)
        register_val = (avg_pitch - 60) / 36  # C5=0, higher is positive
        register_val = max(-0.5, min(0.5, register_val))

        # 5. Dynamics
        avg_vel = sum(n.velocity for n in window) / len(window)
        # Moderate dynamics are most positive; extremes shift color
        if avg_vel < 35:
            dyn_val = -0.1  # pianissimo — shadowy
        elif avg_vel > 100:
            dyn_val = 0.1   # fortissimo — energized but not necessarily bright
        else:
            dyn_val = 0.15  # comfortable dynamic range

        # 6. Density weight (sparse textures = ambiguous valence)
        density = len(window) / resolution
        density_weight = min(density / 8.0, 1.0)

        raw = (
            mode_val      * 0.30 +
            avg_pc_val    * 0.20 +
            tension_val   * 0.20 +
            register_val  * 0.15 +
            dyn_val       * 0.15
        ) * density_weight

        valence = max(-1.0, min(1.0, raw))
        curve.append((t, valence))
        t += resolution

    if not curve:
        return {'curve': [], 'mean_valence': 0.0, 'peaks_light': [],
                'peaks_dark': [], 'transitions': [], 'volatility': 0.0,
                'arc_shape': 'indeterminado', 'desc': 'sin datos'}

    vals = [v for _, v in curve]
    mean_val = sum(vals) / len(vals)
    volatility = math.sqrt(sum((v - mean_val) ** 2 for v in vals) / len(vals))

    # Peaks of light (local maxima above +0.3)
    peaks_light = []
    for i in range(1, len(curve) - 1):
        if curve[i][1] > curve[i-1][1] and curve[i][1] > curve[i+1][1] and curve[i][1] > 0.3:
            peaks_light.append({'time': curve[i][0], 'valence': curve[i][1]})
    peaks_light.sort(key=lambda x: -x['valence'])

    # Peaks of darkness (local minima below -0.3)
    peaks_dark = []
    for i in range(1, len(curve) - 1):
        if curve[i][1] < curve[i-1][1] and curve[i][1] < curve[i+1][1] and curve[i][1] < -0.3:
            peaks_dark.append({'time': curve[i][0], 'valence': curve[i][1]})
    peaks_dark.sort(key=lambda x: x['valence'])

    # Abrupt transitions (Δvalence > 0.4 in one step)
    transitions = []
    for i in range(1, len(curve)):
        delta = curve[i][1] - curve[i-1][1]
        if abs(delta) > 0.40:
            direction = 'iluminación' if delta > 0 else 'oscurecimiento'
            transitions.append({
                'time': curve[i][0],
                'delta': delta,
                'direction': direction,
                'from': curve[i-1][1],
                'to': curve[i][1],
            })

    # Arc shape
    n = len(vals)
    thirds = max(1, n // 3)
    v1 = sum(vals[:thirds]) / thirds
    v2 = sum(vals[thirds:2*thirds]) / thirds
    v3 = sum(vals[2*thirds:]) / max(1, n - 2*thirds)

    if abs(v3 - v1) < 0.1 and abs(v2 - v1) < 0.1:
        arc_shape = 'estable'
    elif v2 > v1 + 0.15 and v2 > v3 + 0.15:
        arc_shape = 'cúpula'       # bright centre
    elif v2 < v1 - 0.15 and v2 < v3 - 0.15:
        arc_shape = 'valle'        # dark centre
    elif v3 > v1 + 0.12:
        arc_shape = 'ascendente'
    elif v3 < v1 - 0.12:
        arc_shape = 'descendente'
    else:
        arc_shape = 'sinusoidal'

    arc_descs = {
        'estable':     'valencia constante — una emoción sostenida sin variación expresiva significativa',
        'cúpula':      'luminosidad central — la pieza se abre hacia la luz y luego se cierra; el corazón es brillante',
        'valle':       'oscuridad central — el momento más oscuro es el centro; la pieza bordea el abismo y regresa',
        'ascendente':  'arco hacia la luz — la pieza emerge progresivamente de la sombra hacia la claridad',
        'descendente': 'arco hacia la sombra — la pieza parte luminosa y se oscurece sin retorno',
        'sinusoidal':  'oscilación luminosa — la pieza alterna entre luz y sombra de forma continua',
    }

    if mean_val > 0.35:
        overall = 'predominantemente luminosa (valencia positiva dominante)'
    elif mean_val > 0.1:
        overall = 'ligeramente positiva con momentos de ambigüedad'
    elif mean_val > -0.1:
        overall = 'emocionalmente ambivalente — equilibrio entre luz y sombra'
    elif mean_val > -0.35:
        overall = 'ligeramente oscura con tensión latente'
    else:
        overall = 'predominantemente oscura (valencia negativa dominante)'

    desc = (f"Valencia media {mean_val:+.2f} — {overall}. "
            f"Arco: {arc_descs.get(arc_shape, arc_shape)}. "
            f"Volatilidad {volatility:.2f} — "
            f"{'pieza de contrastes emocionales extremos' if volatility > 0.35 else 'perfil emocional estable'}. "
            + (f"{len(transitions)} transición(es) brusca(s) de valencia." if transitions else ""))

    return {
        'curve':        curve,
        'mean_valence': mean_val,
        'volatility':   volatility,
        'arc_shape':    arc_shape,
        'arc_desc':     arc_descs.get(arc_shape, arc_shape),
        'peaks_light':  peaks_light[:4],
        'peaks_dark':   peaks_dark[:4],
        'transitions':  transitions[:8],
        'desc':         desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v9-B. ANÁLISIS DE CATARSIS
# ═══════════════════════════════════════════════════════════════

def analyze_catharsis(
    tension_curve: List[TensionPoint],
    dynamic_valence: Dict,
    silence_analysis: Dict,
    energy_profile: Dict,
    total_dur: float,
) -> Dict:
    """
    Detecta si la pieza ofrece catarsis emocional y dónde ocurre.

    La catarsis clásica requiere tres fases:
      1. ACUMULACIÓN — tensión sostenida durante un período significativo
      2. LIBERACIÓN   — caída abrupta de tensión + apertura de valencia
      3. RESOLUCIÓN   — estabilización a un nivel bajo tras la liberación

    También detecta:
      - Catarsis negada: acumulación sin liberación (angustia sostenida)
      - Catarsis prematura: liberación demasiado temprana (<40% de la pieza)
      - Pseudo-catarsis: liberación que no va seguida de resolución
      - Catarsis múltiple: varias oleadas de acumulación/liberación

    Devuelve:
      present         : bool — ¿hay catarsis?
      catharsis_type  : 'clásica' | 'negada' | 'prematura' | 'pseudo' | 'múltiple' | 'ausente'
      moment          : tiempo del pico de liberación (o None)
      buildup_start   : tiempo de inicio de la acumulación principal
      buildup_duration: duración de la acumulación en segundos
      release_strength: intensidad de la liberación [0..1]
      resolution_quality: calidad de la resolución [0..1]
      waves           : lista de oleadas (catarsis múltiple)
      emotional_effect: descripción del efecto sobre el oyente
      desc            : descripción técnica
    """
    if not tension_curve or total_dur <= 0:
        return {'present': False, 'catharsis_type': 'ausente',
                'moment': None, 'desc': 'sin datos para análisis de catarsis'}

    t_vals = [(p.time, p.tension) for p in tension_curve]
    val_curve = dynamic_valence.get('curve', [])
    silences = (silence_analysis or {}).get('silences', [])
    e_curve = (energy_profile or {}).get('curve', [])
    max_e = max((e for _, e in e_curve), default=1) or 1

    # ── Detectar picos y valles de tensión ───────────────────
    times = [t for t, _ in t_vals]
    tensions = [v for _, v in t_vals]

    # Smooth tension with 3-point moving average
    smooth = []
    for i in range(len(tensions)):
        w = tensions[max(0, i-1):i+2]
        smooth.append(sum(w) / len(w))

    # Find peaks (local maxima)
    tension_peaks = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > 0.45:
            tension_peaks.append((times[i], smooth[i]))

    # Find valleys (local minima after a peak)
    tension_valleys = []
    for i in range(1, len(smooth) - 1):
        if smooth[i] < smooth[i-1] and smooth[i] < smooth[i+1] and smooth[i] < 0.25:
            tension_valleys.append((times[i], smooth[i]))

    if not tension_peaks:
        return {
            'present': False, 'catharsis_type': 'ausente',
            'moment': None, 'buildup_start': None,
            'buildup_duration': 0, 'release_strength': 0.0,
            'resolution_quality': 0.0, 'waves': [],
            'emotional_effect': 'pieza de tensión plana — sin acumulación ni liberación',
            'desc': 'no se detectaron picos de tensión suficientes para catarsis',
        }

    # ── Buscar oleadas de catarsis ────────────────────────────
    waves = []
    for peak_time, peak_tension in tension_peaks:
        # Find valley after this peak
        after_valleys = [(t, v) for t, v in tension_valleys if t > peak_time and t < peak_time + total_dur * 0.3]
        if not after_valleys:
            continue

        valley_time, valley_tension = after_valleys[0]
        release_strength = peak_tension - valley_tension  # drop magnitude

        if release_strength < 0.25:
            continue  # not a real release

        # Measure buildup: find when tension started rising toward this peak
        before = [(t, v) for t, v in t_vals if t < peak_time]
        buildup_start = peak_time - 5.0  # default: 5s buildup
        for i in range(len(before) - 1, max(0, len(before) - 30), -1):
            if before[i][1] < peak_tension * 0.5:
                buildup_start = before[i][0]
                break
        buildup_duration = peak_time - buildup_start

        # Measure resolution: how stable is tension after the valley?
        after_valley = [(t, v) for t, v in t_vals if t > valley_time and t < valley_time + 5.0]
        if after_valley:
            post_vals = [v for _, v in after_valley]
            resolution_quality = 1.0 - (max(post_vals) - min(post_vals))
            resolution_quality = max(0.0, min(1.0, resolution_quality))
        else:
            resolution_quality = 0.5

        # Check if there's a silence near the release (amplifies catharsis)
        nearby_silence = any(
            abs(s['start'] - valley_time) < 3.0
            for s in silences if s.get('duration', 0) > 0.5
        )

        waves.append({
            'peak_time':          peak_time,
            'peak_tension':       peak_tension,
            'release_time':       valley_time,
            'release_strength':   release_strength,
            'buildup_start':      buildup_start,
            'buildup_duration':   buildup_duration,
            'resolution_quality': resolution_quality,
            'position_ratio':     valley_time / total_dur,
            'silence_at_release': nearby_silence,
        })

    # ── Classify catharsis type ───────────────────────────────
    if not waves:
        # No release found — check if there was buildup
        max_tension = max(tensions)
        if max_tension > 0.5:
            cat_type = 'negada'
            effect = ('angustia sostenida — la pieza acumula tensión sin liberarla. '
                      'El oyente queda en estado de activación no resuelta. '
                      'Efecto: inquietud residual, sensación de incompletitud, '
                      'incomodidad productiva o frustración expresiva.')
        else:
            cat_type = 'ausente'
            effect = 'tensión insuficiente para catarsis — pieza contemplativa o neutra'
        return {
            'present': False, 'catharsis_type': cat_type,
            'moment': None, 'buildup_start': None,
            'buildup_duration': 0, 'release_strength': 0.0,
            'resolution_quality': 0.0, 'waves': [],
            'emotional_effect': effect,
            'desc': f'catarsis {cat_type}: {effect}',
        }

    # Select primary wave (strongest release)
    primary = max(waves, key=lambda w: w['release_strength'])
    pos = primary['position_ratio']

    if len(waves) >= 3:
        cat_type = 'múltiple'
    elif pos < 0.40:
        cat_type = 'prematura'
    elif primary['resolution_quality'] < 0.3:
        cat_type = 'pseudo'
    else:
        cat_type = 'clásica'

    # Emotional effect descriptions
    effects = {
        'clásica': ('purificación emocional completa — la pieza conduce al oyente '
                    'a través de la tensión hacia una liberación genuina. '
                    'Efecto residual: alivio, apertura, sensación de haber '
                    'atravesado algo y salido al otro lado.'),
        'prematura': ('liberación demasiado temprana — la catarsis ocurre antes de que '
                      'la tensión haya madurado. Efecto: el oyente se queda sin '
                      'el impacto emocional completo; la pieza no exige suficiente.'),
        'pseudo': ('liberación sin resolución — la tensión cae pero no se estabiliza. '
                   'El oyente experimenta alivio momentáneo seguido de nueva '
                   'incertidumbre. Efecto: ambivalencia, apertura inconclusa.'),
        'múltiple': ('catarsis en oleadas — la pieza construye y libera tensión varias '
                     'veces. Efecto: experiencia emocional intensa y variada; '
                     'el oyente es llevado a través de múltiples ciclos de '
                     'tensión-alivio como una montaña rusa emocional.'),
        'negada': ('catarsis negada — la pieza acumula tensión pero no la libera. '
                   'Efecto deliberado: el oyente queda en un estado de activación '
                   'irresuelto, lo que puede ser una elección expresiva poderosa '
                   'o una pieza que "no sabe cómo terminar".'),
    }

    effect = effects.get(cat_type, '')

    rs = primary['release_strength']
    rq = primary['resolution_quality']
    bd = primary['buildup_duration']

    desc = (f"Catarsis {cat_type} detectada. "
            f"Acumulación de {bd:.0f}s, "
            f"liberación en {ts_str_local(primary['release_time'])} "
            f"(fuerza={rs:.2f}, resolución={rq:.2f}). "
            + (f"Silencio amplificador en el momento de liberación. " if primary.get('silence_at_release') else "")
            + (f"{len(waves)} oleadas totales. " if len(waves) > 1 else ""))

    return {
        'present':            True,
        'catharsis_type':     cat_type,
        'moment':             primary['release_time'],
        'buildup_start':      primary['buildup_start'],
        'buildup_duration':   bd,
        'release_strength':   rs,
        'resolution_quality': rq,
        'waves':              waves,
        'emotional_effect':   effect,
        'desc':               desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v9-C. MAPA DE AMBIVALENCIA EMOCIONAL
# ═══════════════════════════════════════════════════════════════

# Per-dimension emotional polarity: (mode, tempo, dynamics, register)
_DIM_POLES = {
    # mode → (valence_sign, label)
    'mode': {
        'major': +1, 'lydian': +1, 'mixolydian': +1,
        'dorian': 0, 'minor': -1, 'phrygian': -1, 'locrian': -1,
    },
    # velocity → polarity
    'dynamics': lambda v: +1 if v > 80 else (-1 if v < 35 else 0),
    # avg pitch → polarity
    'register': lambda p: +1 if p > 72 else (-1 if p < 52 else 0),
    # tempo → polarity  (fast=positive energy, slow=negative)
    'tempo': lambda b: +1 if b > 120 else (-1 if b < 70 else 0),
}

def analyze_emotional_ambivalence(
    notes: List[NoteEvent],
    tension_curve: List[TensionPoint],
    dynamic_valence: Dict,
    mode: str,
    avg_bpm: float,
    total_dur: float,
    resolution: float = 2.0,
) -> Dict:
    """
    Detecta momentos de ambivalencia emocional genuina — cuando múltiples
    dimensiones expresivas apuntan en direcciones contradictorias.

    Ejemplo clásico: modo lidio (luminoso) + fortísimo (poderoso) + tempo lento
    (pesante) = euforia aplastante / grandiosidad fúnebre. No es tristeza ni
    alegría: es un estado mixto sin nombre directo.

    Cuantifica el conflicto entre:
      - Modo local (KS en ventana)
      - Dinámica (velocidad MIDI)
      - Registro (altura media)
      - Tensión armónica
      - Valencia estimada

    Para cada ventana calcula un índice de ambivalencia [0..1]:
      0 = todas las dimensiones apuntan en la misma dirección
      1 = máximo conflicto (cada dimensión dice algo distinto)

    Devuelve:
      map            : lista de (time, ambivalence_index, description)
      mean_ambivalence: índice medio global
      peak_moments   : momentos de máxima ambivalencia con descripción narrativa
      zones          : zonas continuas de alta ambivalencia (>0.5)
      profile        : tipo de ambivalencia dominante
      desc           : descripción global
    """
    if not notes or total_dur <= 0:
        return {'map': [], 'mean_ambivalence': 0.0, 'peak_moments': [],
                'zones': [], 'profile': 'indeterminado', 'desc': 'sin datos'}

    sn = sorted(notes, key=lambda n: n.time_sec)
    t_map = {p.time: p.tension for p in tension_curve}
    t_times_sorted = sorted(t_map.keys())
    val_curve_dict = {t: v for t, v in dynamic_valence.get('curve', [])}
    val_times = sorted(val_curve_dict.keys())

    def get_tension(t):
        if not t_times_sorted: return 0.3
        return t_map[min(t_times_sorted, key=lambda x: abs(x - t))]

    def get_valence(t):
        if not val_times: return 0.0
        return val_curve_dict[min(val_times, key=lambda x: abs(x - t))]

    ambivalence_map = []
    t = 0.0

    while t < total_dur:
        window = [n for n in sn if n.time_sec < t + resolution and n.time_sec + n.duration > t]

        if not window:
            ambivalence_map.append((t, 0.0, 'silencio'))
            t += resolution
            continue

        avg_vel = sum(n.velocity for n in window) / len(window)
        avg_pitch = sum(n.pitch for n in window) / len(window)
        tension = get_tension(t)
        valence = get_valence(t)

        # Detect local mode
        pc = [0.0] * 12
        for n in window:
            pc[n.pitch % 12] += n.velocity / 127.0
        tot = sum(pc)
        if tot > 0:
            pc_norm = [v / tot for v in pc]
            best_r, local_mode, best_c = 0, mode, -999
            for mn, prof in KS_MODES.items():
                for rot in range(12):
                    r = pearson(pc_norm, prof[rot:] + prof[:rot])
                    if r > best_c:
                        best_c, best_r, local_mode = r, rot, mn
        else:
            local_mode = mode

        # Polarity per dimension
        poles = {
            'modo':    _DIM_POLES['mode'].get(local_mode, 0),
            'dinámica': _DIM_POLES['dynamics'](avg_vel),
            'registro': _DIM_POLES['register'](avg_pitch),
            'tempo':    _DIM_POLES['tempo'](avg_bpm),
            'tensión':  -1 if tension > 0.55 else (1 if tension < 0.2 else 0),
            'valencia': 1 if valence > 0.2 else (-1 if valence < -0.2 else 0),
        }

        active = {k: v for k, v in poles.items() if v != 0}
        if not active:
            ambivalence_map.append((t, 0.0, 'zona neutral'))
            t += resolution
            continue

        pos_dims = [k for k, v in active.items() if v > 0]
        neg_dims = [k for k, v in active.items() if v < 0]

        # Ambivalence = balance between opposing forces
        n_pos = len(pos_dims)
        n_neg = len(neg_dims)
        n_total = n_pos + n_neg
        balance = 1.0 - abs(n_pos - n_neg) / n_total if n_total > 0 else 0.0
        # Scale: both sides must have at least 1 member for real ambivalence
        ambivalence = balance if (n_pos > 0 and n_neg > 0) else 0.0

        # Build narrative label for this moment
        if ambivalence > 0.5:
            light_tags = {'modo': 'modo brillante', 'dinámica': 'forte', 'registro': 'registro agudo',
                          'tensión': 'armonía relajada', 'valencia': 'valencia positiva', 'tempo': 'tempo rápido'}
            dark_tags  = {'modo': 'modo oscuro', 'dinámica': 'piano', 'registro': 'registro grave',
                          'tensión': 'alta tensión', 'valencia': 'valencia negativa', 'tempo': 'tempo lento'}
            lights = ', '.join(light_tags.get(d, d) for d in pos_dims[:2])
            darks  = ', '.join(dark_tags.get(d, d) for d in neg_dims[:2])
            label = f"{lights} vs {darks}"
        else:
            label = f"leve conflicto ({', '.join(list(active.keys())[:3])})"

        ambivalence_map.append((t, ambivalence, label))
        t += resolution

    if not ambivalence_map:
        return {'map': [], 'mean_ambivalence': 0.0, 'peak_moments': [],
                'zones': [], 'profile': 'coherente', 'desc': 'sin ambivalencia detectada'}

    amb_vals = [a for _, a, _ in ambivalence_map]
    mean_amb = sum(amb_vals) / len(amb_vals)

    # Peak moments of ambivalence
    peak_moments = []
    for i in range(1, len(ambivalence_map) - 1):
        t_i, a_i, lbl_i = ambivalence_map[i]
        if (a_i > ambivalence_map[i-1][1] and a_i > ambivalence_map[i+1][1]
                and a_i > 0.45):
            peak_moments.append({'time': t_i, 'ambivalence': a_i, 'desc': lbl_i})
    peak_moments.sort(key=lambda x: -x['ambivalence'])

    # Continuous zones of high ambivalence
    zones = []
    in_zone = False
    zone_start = 0.0
    for t_i, a_i, _ in ambivalence_map:
        if a_i > 0.5 and not in_zone:
            in_zone = True
            zone_start = t_i
        elif a_i <= 0.5 and in_zone:
            zones.append({'start': zone_start, 'end': t_i, 'duration': t_i - zone_start})
            in_zone = False
    if in_zone:
        zones.append({'start': zone_start, 'end': total_dur, 'duration': total_dur - zone_start})

    # Profile
    if mean_amb > 0.5:
        profile = 'alta ambivalencia estructural — la pieza vive en la contradicción emocional permanente'
    elif mean_amb > 0.3:
        profile = 'ambivalencia moderada — estados mixtos frecuentes y deliberados'
    elif mean_amb > 0.15:
        profile = 'ambivalencia puntual — momentos específicos de conflicto emocional'
    else:
        profile = 'emocionalmente coherente — las dimensiones expresivas apuntan en la misma dirección'

    pct_zone = sum(z['duration'] for z in zones) / total_dur * 100 if total_dur > 0 else 0

    desc = (f"Ambivalencia media {mean_amb:.2f} — {profile}. "
            + (f"{pct_zone:.0f}% del tiempo en zonas de conflicto emocional activo. " if pct_zone > 5 else "")
            + (f"{len(peak_moments)} momento(s) de máxima contradicción expresiva." if peak_moments else ""))

    return {
        'map':             ambivalence_map,
        'mean_ambivalence': mean_amb,
        'peak_moments':    peak_moments[:5],
        'zones':           zones,
        'profile':         profile,
        'desc':            desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v9-D. DISTANCIA EMOCIONAL TOTAL (trayectoria VA)
# ═══════════════════════════════════════════════════════════════

def compute_emotional_trajectory(
    sva_list: List[Tuple],
    sections: List,
    dynamic_valence: Dict,
    total_dur: float,
) -> Dict:
    """
    Calcula la trayectoria de la pieza en el espacio Valence-Arousal (VA)
    como una ruta geométrica, no solo como puntos estáticos.

    Métricas:
      total_distance  : longitud total del camino en el espacio VA
      net_displacement: distancia entre punto inicial y final (desplazamiento neto)
      efficiency      : desplazamiento/distancia — 1.0=línea recta, ~0=bucle
      centroid        : baricentro del camino (dónde "vive" la pieza en VA)
      path_shape      : forma del camino ('lineal','circular','espiral','bucle','errático')
      quadrant_time   : tiempo pasado en cada cuadrante VA
      crossings       : veces que cruza los ejes VA (cambios de cuadrante)
      desc            : descripción narrativa
    """
    active_sva = [(s, v, a) for s, (v, a, d) in zip(sections, sva_list)
                  if d != 'sin notas']
    if len(active_sva) < 2:
        return {'total_distance': 0.0, 'net_displacement': 0.0,
                'efficiency': 1.0, 'centroid': (0.0, 0.0),
                'path_shape': 'indeterminado', 'quadrant_time': {},
                'crossings': 0, 'desc': 'insuficientes secciones para trayectoria'}

    points = [(v, a) for _, v, a in active_sva]

    # Total path length
    total_dist = sum(
        math.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)
        for i in range(1, len(points))
    )

    # Net displacement (start → end)
    net_disp = math.sqrt(
        (points[-1][0] - points[0][0])**2 + (points[-1][1] - points[0][1])**2
    )

    # Efficiency
    efficiency = net_disp / total_dist if total_dist > 0 else 1.0

    # Centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    # Quadrant time (equal-time sections assumed)
    quad_time = {'I (alegre/activo)': 0, 'II (tenso/agitado)': 0,
                 'III (triste/pasivo)': 0, 'IV (sereno/calmado)': 0}
    sec_dur = total_dur / max(len(active_sva), 1)
    for _, v, a in active_sva:
        if v >= 0 and a >= 0:   quad_time['I (alegre/activo)']   += sec_dur
        elif v < 0 and a >= 0:  quad_time['II (tenso/agitado)']  += sec_dur
        elif v < 0 and a < 0:   quad_time['III (triste/pasivo)'] += sec_dur
        else:                   quad_time['IV (sereno/calmado)'] += sec_dur

    # Axis crossings (quadrant changes)
    crossings = 0
    for i in range(1, len(points)):
        pv, pa = points[i-1]
        cv2, ca = points[i]
        if (pv >= 0) != (cv2 >= 0) or (pa >= 0) != (ca >= 0):
            crossings += 1

    # Path shape heuristics
    if efficiency > 0.75:
        path_shape = 'lineal'
    elif efficiency < 0.15:
        path_shape = 'bucle'   # ends near where it started
    elif crossings >= len(points) * 0.5:
        path_shape = 'errático'
    elif total_dist > 2.5 and efficiency < 0.4:
        path_shape = 'espiral'
    else:
        path_shape = 'circular'

    shape_descs = {
        'lineal':   'trayectoria rectilínea — la pieza avanza en una dirección emocional constante sin retroceder',
        'bucle':    'trayectoria en bucle — la pieza regresa casi exactamente al estado emocional inicial',
        'errático': 'trayectoria errática — cambios de cuadrante frecuentes; viaje emocional imprevisible',
        'espiral':  'trayectoria espiral — cada ciclo emocional lleva más lejos del punto de partida',
        'circular': 'trayectoria circular — la pieza recorre el espacio emocional de forma orgánica',
    }

    dominant_quad = max(quad_time, key=quad_time.get)
    dom_pct = quad_time[dominant_quad] / total_dur * 100 if total_dur > 0 else 0

    centroid_desc = ''
    if cx > 0.2 and cy > 0.2:   centroid_desc = 'núcleo emocional alegre y activo'
    elif cx < -0.2 and cy > 0.2: centroid_desc = 'núcleo emocional sombrío y agitado'
    elif cx < -0.2 and cy < -0.2: centroid_desc = 'núcleo emocional oscuro y pasivo'
    elif cx > 0.2 and cy < -0.2: centroid_desc = 'núcleo emocional luminoso y sereno'
    else:                         centroid_desc = 'núcleo emocional ambivalente (cerca del centro VA)'

    desc = (f"Distancia total recorrida en espacio VA: {total_dist:.2f} unidades. "
            f"Desplazamiento neto: {net_disp:.2f} (eficiencia={efficiency:.2f}). "
            f"Forma del camino: {shape_descs.get(path_shape, path_shape)}. "
            f"Centroide VA ({cx:+.2f}, {cy:+.2f}) — {centroid_desc}. "
            f"Cuadrante dominante: {dominant_quad} ({dom_pct:.0f}% del tiempo). "
            f"{crossings} cruce(s) de eje emocional.")

    return {
        'total_distance':   total_dist,
        'net_displacement': net_disp,
        'efficiency':       efficiency,
        'centroid':         (cx, cy),
        'path_shape':       path_shape,
        'path_shape_desc':  shape_descs.get(path_shape, path_shape),
        'quadrant_time':    quad_time,
        'crossings':        crossings,
        'dominant_quadrant': dominant_quad,
        'points':           points,
        'desc':             desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v9-E. ANÁLISIS DE ELECCIONES ANTI-CONVENCIONALES
# ═══════════════════════════════════════════════════════════════

# Expected harmonic choices by mode (most common root movements)
_EXPECTED_ROOT_MOVES = {
    'major':      {7: 0.25, 5: 0.20, 2: 0.15, 9: 0.10, 4: 0.08, 0: 0.07},
    'minor':      {7: 0.22, 3: 0.18, 5: 0.15, 8: 0.12, 10: 0.10, 0: 0.08},
    'dorian':     {7: 0.20, 2: 0.18, 5: 0.15, 9: 0.12, 0: 0.10},
    'phrygian':   {1: 0.25, 7: 0.18, 5: 0.15, 3: 0.12},
    'lydian':     {7: 0.22, 6: 0.18, 2: 0.15, 4: 0.12, 9: 0.10},
    'mixolydian': {7: 0.22, 5: 0.18, 10: 0.15, 2: 0.12},
    'locrian':    {6: 0.25, 3: 0.18, 8: 0.15, 10: 0.12},
}

# Expected cadence distribution
_EXPECTED_CADENCES = {
    'major': {'perfecta': 0.45, 'imperfecta': 0.25, 'plagal': 0.15, 'rota': 0.10, 'frigia': 0.05},
    'minor': {'perfecta': 0.35, 'imperfecta': 0.25, 'frigia': 0.20, 'plagal': 0.12, 'rota': 0.08},
}

def analyze_anti_conventional_choices(
    chords: List[ChordEvent],
    cadences: Dict,
    key_root: int,
    mode: str,
    motifs: List,
    rhythm: Dict,
    musical_form: Dict,
    avg_bpm: float,
    total_dur: float,
) -> Dict:
    """
    Identifica decisiones compositivas que se desvían significativamente
    de lo estadísticamente esperado dado el contexto (tonalidad, modo, género).

    Analiza desviaciones en:
      1. Progresiones armónicas (movimientos de raíz inesperados)
      2. Cadencias (evitación sistemática de cadencias perfectas)
      3. Forma musical (evitar la simetría canónica)
      4. Rítmica (uso del metro inusual)
      5. Rango dinámico (uso extremo o ausencia de contraste)

    Devuelve:
      deviations      : lista de desviaciones detectadas con peso y descripción
      deviation_score : índice global de anti-convencionalidad [0..1]
      voice_profile   : perfil de la voz compositiva
      signature_choices: las elecciones más características del compositor
      desc            : análisis narrativo
    """
    if not chords:
        return {'deviations': [], 'deviation_score': 0.0,
                'voice_profile': 'indeterminado', 'signature_choices': [],
                'desc': 'sin acordes para análisis de convencionalidad'}

    deviations = []

    # ── 1. Movimientos armónicos anti-convencionales ──────────
    expected = _EXPECTED_ROOT_MOVES.get(mode, _EXPECTED_ROOT_MOVES['major'])
    root_moves = collections.Counter()
    for i in range(len(chords) - 1):
        move = (chords[i+1].root - chords[i].root) % 12
        root_moves[move] += 1
    total_moves = sum(root_moves.values()) or 1

    # Unexpected moves: those with high frequency but low expectation
    for move, count in root_moves.most_common(5):
        freq = count / total_moves
        expected_freq = expected.get(move, 0.03)
        deviation_ratio = freq / expected_freq if expected_freq > 0 else freq * 10
        if deviation_ratio > 2.5 and freq > 0.08:
            interval_name = {0:'unísono',1:'2ªm',2:'2ªM',3:'3ªm',4:'3ªM',
                             5:'4ªJ',6:'tritono',7:'5ªJ',8:'6ªm',9:'6ªM',
                             10:'7ªm',11:'7ªM'}.get(move, f'{move}st')
            deviations.append({
                'type': 'movimiento armónico',
                'desc': (f"Movimiento de {interval_name} ({move}st) frecuente: "
                         f"{freq*100:.0f}% de las progresiones "
                         f"(esperado: {expected_freq*100:.0f}%). "
                         f"Uso {deviation_ratio:.1f}× más frecuente que la norma."),
                'weight': min(1.0, deviation_ratio / 5.0),
                'signature': f"preferencia por movimientos de {interval_name}",
            })

    # ── 2. Desviación en cadencias ────────────────────────────
    cad_counts = cadences.get('counts', collections.Counter())
    cad_total = cadences.get('total', 0)
    mode_key = 'minor' if mode in ('minor', 'phrygian', 'locrian') else 'major'
    exp_cad = _EXPECTED_CADENCES.get(mode_key, _EXPECTED_CADENCES['major'])

    if cad_total > 0:
        # Perfect cadence avoidance
        perf_freq = cad_counts.get('perfecta', 0) / cad_total
        exp_perf = exp_cad.get('perfecta', 0.4)
        if perf_freq < exp_perf * 0.4:
            deviations.append({
                'type': 'evitación cadencial',
                'desc': (f"Cadencias perfectas: {perf_freq*100:.0f}% "
                         f"(esperado: {exp_perf*100:.0f}%). "
                         f"El compositor evita sistemáticamente el cierre tonal "
                         f"definitivo — preferencia por la apertura y la ambigüedad."),
                'weight': 0.7,
                'signature': 'evitación de la cadencia perfecta',
            })

        # Deceptive cadence overuse
        rota_freq = cad_counts.get('rota', 0) / cad_total
        exp_rota = exp_cad.get('rota', 0.08)
        if rota_freq > exp_rota * 2.5:
            deviations.append({
                'type': 'cadencia engañosa sistemática',
                'desc': (f"Cadencias rotas/engañosas: {rota_freq*100:.0f}% "
                         f"(esperado: {exp_rota*100:.0f}%). "
                         f"El compositor usa la sorpresa cadencial como recurso expresivo "
                         f"habitual — cada cierre esperado se convierte en una evasión."),
                'weight': 0.75,
                'signature': 'cadencias engañosas como firma',
            })

    # ── 3. Forma anti-convencional ────────────────────────────
    form = musical_form.get('form', '')
    if 'through-composed' in form.lower() or form == 'libre':
        deviations.append({
            'type': 'forma through-composed',
            'desc': ('Sin estructura de repetición formal — la pieza avanza '
                     'continuamente sin retornar al material inicial. '
                     'Decisión anti-convencional: rechaza la simetría y el '
                     'retorno esperados; cada momento es irrepetible.'),
            'weight': 0.6,
            'signature': 'forma through-composed (sin retorno)',
        })

    # ── 4. Tempo extremo ──────────────────────────────────────
    if avg_bpm < 45:
        deviations.append({
            'type': 'tempo extremadamente lento',
            'desc': (f"Tempo de {avg_bpm:.0f} BPM — muy por debajo del rango habitual. "
                     f"El tiempo se dilata hasta el límite de la percepción de pulso."),
            'weight': 0.5,
            'signature': 'tempo extremo como recurso temporal',
        })
    elif avg_bpm > 180:
        deviations.append({
            'type': 'tempo extremadamente rápido',
            'desc': (f"Tempo de {avg_bpm:.0f} BPM — umbral de fusión de pulso. "
                     f"La velocidad suprime la percepción melódica individual."),
            'weight': 0.5,
            'signature': 'saturación temporal como recurso',
        })

    # ── 5. Extensiones armónicas (uso de acordes complejos) ───
    if chords:
        complex_types = {'maj7', 'min7', 'dom7', 'dim7', 'hdim7', 'maj9', 'add9'}
        complex_count = sum(1 for c in chords if c.chord_type in complex_types)
        complex_ratio = complex_count / len(chords)
        if complex_ratio > 0.55:
            deviations.append({
                'type': 'saturación de extensiones armónicas',
                'desc': (f"{complex_ratio*100:.0f}% acordes con 7ª o extensiones "
                         f"— muy por encima de la media. El compositor evita la "
                         f"tríada simple; cada acorde tiene capas adicionales de color."),
                'weight': 0.6,
                'signature': f'preferencia por armonía extendida ({complex_ratio*100:.0f}%)',
            })
        elif complex_ratio < 0.05 and len(chords) > 10:
            deviations.append({
                'type': 'purismo armónico',
                'desc': (f"Solo {complex_ratio*100:.0f}% acordes con extensiones. "
                         f"El compositor usa casi exclusivamente tríadas puras — "
                         f"elección de claridad armónica extrema."),
                'weight': 0.4,
                'signature': 'purismo de tríadas (sin extensiones)',
            })

    # ── Score global ──────────────────────────────────────────
    if not deviations:
        deviation_score = 0.05
        voice_profile = 'convencional — la escritura sigue los patrones estadísticos esperados para su modo y contexto'
    else:
        # Weighted average
        deviation_score = min(1.0, sum(d['weight'] for d in deviations) / max(3, len(deviations)) * 1.5)

        if deviation_score > 0.65:
            voice_profile = 'voz altamente original — el compositor rechaza sistemáticamente las convenciones de su contexto tonal'
        elif deviation_score > 0.40:
            voice_profile = 'voz distintiva — desviaciones significativas que definen un estilo reconocible'
        elif deviation_score > 0.20:
            voice_profile = 'voz con rasgos propios — algunas elecciones características sobre base convencional'
        else:
            voice_profile = 'voz mayormente convencional con detalles personales'

    signature_choices = [d['signature'] for d in deviations if 'signature' in d]

    desc_parts = [voice_profile]
    for d in deviations[:3]:
        desc_parts.append(f"[{d['type']}] {d['desc'][:100]}...")
    desc = ' | '.join(desc_parts[:4])

    return {
        'deviations':       deviations,
        'deviation_score':  deviation_score,
        'voice_profile':    voice_profile,
        'signature_choices': signature_choices,
        'desc':             desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v9-F. DENSIDAD DE INFORMACIÓN POR NIVEL ESTRUCTURAL
# ═══════════════════════════════════════════════════════════════

def analyze_multilevel_information_density(
    notes: List[NoteEvent],
    chords: List[ChordEvent],
    motifs: List,
    sections: List,
    avg_bpm: float,
    total_dur: float,
) -> Dict:
    """
    Mide cuánta información nueva introduce la pieza en cada escala temporal:
      - Nivel micro (nota a nota, ~0.1-0.5s)
      - Nivel meso (frase, ~2-8s)
      - Nivel macro (sección, ~10s+)

    Un compositor "económico" concentra la novedad en pocos gestos que se
    desarrollan. Un compositor "barroco" distribuye información densa en todos
    los niveles simultáneamente.

    Usa entropía de Shannon sobre vectores de pitch-class como proxy de
    información musical, computada en tres resoluciones temporales.

    Devuelve:
      micro_entropy   : entropía media a nivel de nota (~0.25s ventanas)
      meso_entropy    : entropía media a nivel de frase (~4s ventanas)
      macro_entropy   : entropía media a nivel de sección
      density_profile : 'económico' | 'equilibrado' | 'barroco' | 'maximalista'
      level_contrast  : diferencia entre niveles (alta = información concentrada)
      novelty_curve   : curva de novedad por sección
      composer_economy: índice de economía compositiva [0..1]
      desc            : descripción narrativa
    """
    if not notes or total_dur <= 0:
        return {'micro_entropy': 0.0, 'meso_entropy': 0.0, 'macro_entropy': 0.0,
                'density_profile': 'indeterminado', 'level_contrast': 0.0,
                'novelty_curve': [], 'composer_economy': 0.5, 'desc': 'sin datos'}

    def chroma_entropy(note_window: List[NoteEvent]) -> float:
        """Shannon entropy of pitch-class distribution."""
        if not note_window:
            return 0.0
        pc = [0.0] * 12
        for n in note_window:
            pc[n.pitch % 12] += n.velocity / 127.0
        tot = sum(pc)
        if tot == 0:
            return 0.0
        probs = [v / tot for v in pc if v > 0]
        return -sum(p * math.log2(p) for p in probs)  # max = log2(12) ≈ 3.58 bits

    sn = sorted(notes, key=lambda n: n.time_sec)

    # ── Micro entropy (0.25s windows) ────────────────────────
    micro_res = 0.25
    micro_entropies = []
    t = 0.0
    while t < total_dur:
        win = [n for n in sn if n.time_sec < t + micro_res and n.time_sec + n.duration > t]
        if win:
            micro_entropies.append(chroma_entropy(win))
        t += micro_res
    micro_entropy = sum(micro_entropies) / len(micro_entropies) if micro_entropies else 0.0

    # ── Meso entropy (4s windows) ─────────────────────────────
    meso_res = 4.0
    meso_entropies = []
    t = 0.0
    while t < total_dur:
        win = [n for n in sn if n.time_sec < t + meso_res and n.time_sec + n.duration > t]
        if win:
            meso_entropies.append(chroma_entropy(win))
        t += meso_res
    meso_entropy = sum(meso_entropies) / len(meso_entropies) if meso_entropies else 0.0

    # ── Macro entropy (per section) ───────────────────────────
    macro_entropies = []
    for s in sections:
        if s.notes:
            macro_entropies.append(chroma_entropy(s.notes))
    macro_entropy = sum(macro_entropies) / len(macro_entropies) if macro_entropies else 0.0

    # ── Novelty curve: how much new material per section ─────
    novelty_curve = []
    prev_chroma = [0.0] * 12
    for s in sections:
        if not s.notes:
            novelty_curve.append({'section': s.index + 1, 'novelty': 0.0})
            continue
        pc = [0.0] * 12
        for n in s.notes:
            pc[n.pitch % 12] += n.velocity / 127.0
        tot = sum(pc)
        if tot > 0:
            pc = [v / tot for v in pc]
        prev_tot = sum(prev_chroma)
        if prev_tot > 0:
            prev_norm = [v / prev_tot for v in prev_chroma]
        else:
            prev_norm = prev_chroma
        # Cosine distance (1 - similarity) = novelty
        dot = sum(pc[i] * prev_norm[i] for i in range(12))
        novelty = 1.0 - dot
        novelty_curve.append({'section': s.index + 1, 'start': s.start,
                               'novelty': novelty, 'entropy': chroma_entropy(s.notes)})
        prev_chroma = [p * tot for p in pc]  # un-normalize for next iteration

    # ── Level contrast ────────────────────────────────────────
    # High contrast = information is concentrated at one level
    levels = [micro_entropy, meso_entropy, macro_entropy]
    level_mean = sum(levels) / 3
    level_contrast = math.sqrt(sum((e - level_mean)**2 for e in levels) / 3)

    # ── Composer economy ─────────────────────────────────────
    # Economy: macro captures most info (structured), micro is low (not random)
    max_entropy = math.log2(12)  # 3.58 bits
    norm_micro = micro_entropy / max_entropy
    norm_macro = macro_entropy / max_entropy
    # Economy = macro dominates over micro noise
    composer_economy = max(0.0, min(1.0, norm_macro - norm_micro * 0.5 + 0.5))

    # ── Profile ──────────────────────────────────────────────
    if composer_economy > 0.70 and norm_micro < 0.40:
        profile = 'económico'
        profile_desc = ('escritura económica — pocos materiales desplegados con máxima '
                        'eficiencia estructural; cada nota lleva el peso de la forma')
    elif norm_micro > 0.70 and norm_macro > 0.70:
        profile = 'maximalista'
        profile_desc = ('escritura maximalista — alta densidad de información en todos '
                        'los niveles; textura saturada de novedad constante')
    elif norm_micro > 0.55 and norm_macro < 0.45:
        profile = 'barroco'
        profile_desc = ('escritura barroca — rica en detalle micro pero menos variada '
                        'a gran escala; la ornamentación prima sobre la arquitectura')
    else:
        profile = 'equilibrado'
        profile_desc = ('escritura equilibrada — información distribuida de forma '
                        'coherente entre los niveles micro, meso y macro')

    desc = (f"Perfil compositivo: {profile_desc}. "
            f"Entropía micro={micro_entropy:.2f} | meso={meso_entropy:.2f} | macro={macro_entropy:.2f} bits. "
            f"Contraste entre niveles: {level_contrast:.2f} "
            f"(alto = información concentrada en un nivel). "
            f"Economía compositiva: {composer_economy:.2f}/1.0.")

    return {
        'micro_entropy':    micro_entropy,
        'meso_entropy':     meso_entropy,
        'macro_entropy':    macro_entropy,
        'density_profile':  profile,
        'profile_desc':     profile_desc,
        'level_contrast':   level_contrast,
        'novelty_curve':    novelty_curve,
        'composer_economy': composer_economy,
        'desc':             desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v9-G. MODELO DE INTENCIÓN NARRATIVA
# ═══════════════════════════════════════════════════════════════

def analyze_narrative_intention(
    tension_curve: List[TensionPoint],
    energy_profile: Dict,
    dynamic_valence: Dict,
    catharsis: Dict,
    musical_form: Dict,
    sections: List,
    sva_list: List,
    total_dur: float,
    avg_bpm: float,
) -> Dict:
    """
    Infiere la intención narrativa del compositor a partir de la estructura
    global de tensión, energía, valencia y forma.

    Clasifica la pieza en uno de 7 arquetipos narrativos, y dentro de cada
    uno describe la relación del compositor con el tiempo y la expectativa:

    1. ARCO DRAMÁTICO CLÁSICO  — exposición → complicación → clímax → resolución
    2. CONTEMPLACIÓN            — sin arco, estado sostenido, meditación
    3. CICLO                    — retorno al estado inicial; forma circular
    4. RUPTURA                  — quiebre deliberado de expectativas narrativas
    5. ESPIRAL ASCENDENTE       — acumulación sin retorno, hacia más luz/tensión
    6. ESPIRAL DESCENDENTE      — disolución progresiva
    7. DIÁLOGO                  — alternancia entre dos estados o voces opuestos

    Devuelve:
      archetype       : nombre del arquetipo
      archetype_desc  : descripción del arquetipo
      evidence        : lista de evidencias que soportan el arquetipo
      confidence      : confianza en la clasificación [0..1]
      time_relationship: relación del compositor con el tiempo
      expectation_strategy: cómo maneja la expectativa del oyente
      composer_intent : descripción de la intención compositiva inferida
      alternative     : arquetipo alternativo con segunda mayor puntuación
      desc            : resumen narrativo completo
    """
    if not tension_curve or total_dur <= 0:
        return {'archetype': 'indeterminado', 'confidence': 0.0,
                'desc': 'sin datos suficientes para inferir intención narrativa'}

    t_vals = [p.tension for p in tension_curve]
    e_curve = energy_profile.get('curve', [])
    e_vals = [e for _, e in e_curve] if e_curve else []
    max_e = max(e_vals) if e_vals else 1
    e_norm = [e / max_e for e in e_vals] if max_e > 0 else e_vals

    val_curve = dynamic_valence.get('curve', [])
    val_vals = [v for _, v in val_curve] if val_curve else []

    n = len(t_vals)
    thirds = max(1, n // 3)
    quarters = max(1, n // 4)

    t1 = sum(t_vals[:thirds]) / thirds
    t2 = sum(t_vals[thirds:2*thirds]) / thirds
    t3 = sum(t_vals[2*thirds:]) / max(1, n - 2*thirds)
    t_max_idx = t_vals.index(max(t_vals))
    t_max_pos = t_max_idx / max(n - 1, 1)  # 0..1 position of global max tension

    # Valence arc
    if val_vals:
        nv = len(val_vals)
        v1 = sum(val_vals[:nv//3]) / max(1, nv//3)
        v3 = sum(val_vals[2*nv//3:]) / max(1, nv - 2*nv//3)
        val_change = v3 - v1
    else:
        val_change = 0.0

    cat_type = catharsis.get('catharsis_type', 'ausente')
    form_pattern = musical_form.get('pattern', '')
    arc_shape = dynamic_valence.get('arc_shape', 'estable')

    # ── Score each archetype ──────────────────────────────────
    scores = {}
    evidence = {}

    # 1. Arco dramático clásico
    s1 = 0.0
    ev1 = []
    if 0.55 <= t_max_pos <= 0.85:
        s1 += 0.3; ev1.append(f'clímax en posición {t_max_pos:.0%} (centro-tardío, canónico)')
    if cat_type in ('clásica', 'múltiple'):
        s1 += 0.35; ev1.append(f'catarsis {cat_type} presente')
    if t3 < t1 - 0.1:
        s1 += 0.2; ev1.append('tensión final menor que inicial (resolución)')
    if t2 > t1 + 0.1:
        s1 += 0.15; ev1.append('acumulación en parte central')
    scores['Arco dramático clásico'] = s1; evidence['Arco dramático clásico'] = ev1

    # 2. Contemplación
    s2 = 0.0
    ev2 = []
    t_range = max(t_vals) - min(t_vals)
    if t_range < 0.25:
        s2 += 0.4; ev2.append(f'rango de tensión mínimo ({t_range:.2f}) — planitud emocional')
    if arc_shape == 'estable':
        s2 += 0.3; ev2.append('arco de valencia estable')
    if avg_bpm < 75:
        s2 += 0.2; ev2.append(f'tempo lento ({avg_bpm:.0f} BPM)')
    if cat_type == 'ausente':
        s2 += 0.1; ev2.append('sin catarsis — ausencia de arco dramático')
    scores['Contemplación'] = s2; evidence['Contemplación'] = ev2

    # 3. Ciclo
    s3 = 0.0
    ev3 = []
    if abs(t3 - t1) < 0.08:
        s3 += 0.35; ev3.append(f'tensión inicial ({t1:.2f}) ≈ tensión final ({t3:.2f})')
    if val_vals and abs(val_change) < 0.15:
        s3 += 0.25; ev3.append('valencia inicial ≈ valencia final (retorno emocional)')
    if 'A' in form_pattern and form_pattern.count('A') > 1:
        s3 += 0.3; ev3.append('forma con retorno al material A')
    scores['Ciclo'] = s3; evidence['Ciclo'] = ev3

    # 4. Ruptura
    s4 = 0.0
    ev4 = []
    transitions = dynamic_valence.get('transitions', [])
    if len(transitions) > 3:
        s4 += 0.3; ev4.append(f'{len(transitions)} transiciones bruscas de valencia')
    if cat_type == 'negada':
        s4 += 0.3; ev4.append('catarsis negada — resolución rechazada deliberadamente')
    if t_range > 0.55:
        s4 += 0.2; ev4.append(f'rango de tensión extremo ({t_range:.2f})')
    scores['Ruptura'] = s4; evidence['Ruptura'] = ev4

    # 5. Espiral ascendente
    s5 = 0.0
    ev5 = []
    if t3 > t1 + 0.15 and t2 > t1:
        s5 += 0.35; ev5.append('tensión creciente monotónica')
    if val_change > 0.2:
        s5 += 0.3; ev5.append(f'valencia final más luminosa ({val_change:+.2f})')
    if cat_type == 'prematura':
        s5 -= 0.2  # premature catharsis contradicts upward spiral
    scores['Espiral ascendente'] = max(0.0, s5); evidence['Espiral ascendente'] = ev5

    # 6. Espiral descendente
    s6 = 0.0
    ev6 = []
    if t3 > t1 + 0.10 and arc_shape == 'descendente':
        s6 += 0.3; ev6.append('arco de valencia descendente con tensión creciente')
    if val_change < -0.2:
        s6 += 0.35; ev6.append(f'valencia final más oscura ({val_change:+.2f})')
    if t_max_pos > 0.85:
        s6 += 0.2; ev6.append('clímax en los últimos compases (sin retorno)')
    scores['Espiral descendente'] = s6; evidence['Espiral descendente'] = ev6

    # 7. Diálogo
    s7 = 0.0
    ev7 = []
    sva_active = [(v, a) for v, a, d in sva_list if d != 'sin notas']
    if sva_active:
        v_vals_sva = [v for v, _ in sva_active]
        v_oscillation = sum(
            abs(v_vals_sva[i] - v_vals_sva[i-1])
            for i in range(1, len(v_vals_sva))
        ) / max(1, len(v_vals_sva) - 1)
        if v_oscillation > 0.35:
            s7 += 0.4; ev7.append(f'oscilación de valencia por sección: {v_oscillation:.2f}')
    if arc_shape == 'sinusoidal':
        s7 += 0.3; ev7.append('arco sinusoidal — alternancia continua')
    scores['Diálogo'] = s7; evidence['Diálogo'] = ev7

    # ── Select winner ─────────────────────────────────────────
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    archetype, confidence_raw = sorted_scores[0]
    alternative = sorted_scores[1][0] if len(sorted_scores) > 1 else None
    confidence = min(1.0, confidence_raw)

    ev = evidence.get(archetype, [])

    # ── Time relationship ─────────────────────────────────────
    time_rels = {
        'Arco dramático clásico': 'teleológico — el tiempo avanza hacia una meta; cada momento prepara el siguiente',
        'Contemplación':          'estático — el tiempo se suspende; la pieza no "va" a ningún sitio',
        'Ciclo':                  'circular — el tiempo regresa a su origen; la pieza es un mandala temporal',
        'Ruptura':                'anti-narrativo — el tiempo es interrumpido deliberadamente',
        'Espiral ascendente':     'acumulativo — el tiempo construye sin destruir; solo hay ganancia',
        'Espiral descendente':    'entropico — el tiempo consume; la energía se disipa',
        'Diálogo':                'dialéctico — el tiempo avanza por tensión de opuestos',
    }

    # ── Expectation strategy ──────────────────────────────────
    exp_strats = {
        'Arco dramático clásico': 'cumple y supera la expectativa clásica; el oyente es guiado con precisión',
        'Contemplación':          'suspende la expectativa; no hay donde "ir"',
        'Ciclo':                  'satisface la expectativa de retorno; el oyente es confirmado',
        'Ruptura':                'frustra sistemáticamente la expectativa; el oyente es desestabilizado',
        'Espiral ascendente':     'supera constantemente la expectativa; siempre hay más',
        'Espiral descendente':    'decepciona la expectativa de resolución; la pieza se niega a sí misma',
        'Diálogo':                'mantiene la expectativa en tensión constante; dos voces opuestas',
    }

    # ── Composer intent ───────────────────────────────────────
    intents = {
        'Arco dramático clásico': ('El compositor diseña una experiencia emocional completa con inicio, '
                                   'desarrollo y cierre. Busca que el oyente sea transformado por la travesía.'),
        'Contemplación':          ('El compositor ofrece un espacio de presencia, no de narrativa. '
                                   'La intención no es contar sino ser; la pieza es un estado, no un camino.'),
        'Ciclo':                  ('El compositor inscribe la pieza en el tiempo cíclico. '
                                   'La intención es que el oyente salga donde entró, pero habiendo circulado.'),
        'Ruptura':                ('El compositor usa la forma musical para cuestionar sus propias convenciones. '
                                   'La intención es la incomodidad productiva y la desestabilización consciente.'),
        'Espiral ascendente':     ('El compositor construye hacia lo sublime sin ofrecer retorno. '
                                   'La intención es la expansión permanente; no hay vuelta al reposo.'),
        'Espiral descendente':    ('El compositor trabaja con la disolución y la pérdida. '
                                   'La intención es descender al silencio, a la ausencia, al final inevitable.'),
        'Diálogo':                ('El compositor articula una tensión irreconciliable entre dos fuerzas. '
                                   'La intención no es resolver sino sostener la contradicción como forma.'),
    }

    desc = (f"Arquetipo narrativo: '{archetype}' (confianza: {confidence:.0%}). "
            f"{intents.get(archetype, '')} "
            f"Relación con el tiempo: {time_rels.get(archetype, '')}. "
            f"Estrategia de expectativa: {exp_strats.get(archetype, '')}. "
            + (f"Evidencias: {'; '.join(ev[:3])}." if ev else ""))

    return {
        'archetype':            archetype,
        'archetype_desc':       intents.get(archetype, ''),
        'evidence':             ev,
        'confidence':           confidence,
        'time_relationship':    time_rels.get(archetype, ''),
        'expectation_strategy': exp_strats.get(archetype, ''),
        'composer_intent':      intents.get(archetype, ''),
        'alternative':          alternative,
        'scores':               scores,
        'desc':                 desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v10-A. SENSORY ROUGHNESS (Plomp-Levelt, portado a mido)
# ═══════════════════════════════════════════════════════════════
#
#  A diferencia de la tabla DISSONANCE ya existente (que mide
#  disonancia teórica por clase de intervalo), esta implementación
#  mide la *aspereza sensorial física*:
#    - Penaliza pares en registro grave (C3 y por debajo) porque
#      el oído no resuelve bien las frecuencias graves cercanas.
#    - Normaliza por √(n_pares) para que textura densa y ligera
#      sean comparables en calidad, no en volumen.
#    - Produce una curva temporal continua, no solo un valor medio.
#
#  Tabla de pesos: basada en Plomp & Levelt (1965) / Sethares (1997)
#  0=consonancia pura · 1=máxima aspereza

_ROUGHNESS_WEIGHTS = {
    0: 0.00,   # P1  unísono
    1: 1.00,   # m2  máxima disonancia sensorial
    2: 0.80,   # M2  disonante
    3: 0.20,   # m3  consonancia suave
    4: 0.10,   # M3  consonante
    5: 0.05,   # P4  casi perfecta
    6: 0.90,   # TT  diabolus in musica
    7: 0.02,   # P5  perfecta
    8: 0.25,   # m6
    9: 0.15,   # M6
   10: 0.70,   # m7  disonante blues
   11: 0.95,   # M7  punzante
}

_LOW_REG_THRESHOLD = 48   # C3 en MIDI; debajo de aquí, penalización por registro grave
_LOW_REG_PENALTY   = 1.25  # factor multiplicador

def compute_roughness_curve(
    notes: List[NoteEvent],
    total_dur: float,
    resolution: float = 0.5,
) -> Dict:
    """
    Calcula la curva de aspereza sensorial (roughness) continua usando
    el modelo simplificado de Plomp-Levelt portado a datos MIDI.

    Para cada ventana temporal:
      1. Extrae todas las notas activas
      2. Calcula la aspereza de cada par único (intervalo mod 12)
      3. Aplica penalización si alguna nota está en registro grave (< C3)
      4. Normaliza por √(n_pares) para independencia de densidad

    La roughness difiere de la disonancia teórica en que:
      - Es perceptual, no armónica
      - Depende del registro real (no solo de la clase de intervalo)
      - Responde a intervalos compuestos (10as, 17as suenan distinto a 3as/10as)

    Devuelve:
      curve          : lista de (time, roughness) con roughness ∈ [0, ∞)
      curve_norm     : curva normalizada al máximo local [0..1]
      mean_roughness : media global
      peak_roughness : máximo global con timestamp
      valleys        : momentos de mínima aspereza (consonancia máxima)
      roughness_arc  : forma del arco ('crescendo','decrescendo','estable','accidentado')
      register_impact: porcentaje de la roughness atribuible al registro grave
      desc           : descripción en lenguaje natural
    """
    if not notes or total_dur <= 0:
        return {
            'curve': [], 'curve_norm': [], 'mean_roughness': 0.0,
            'peak_roughness': {'time': 0, 'value': 0},
            'valleys': [], 'roughness_arc': 'indeterminado',
            'register_impact': 0.0, 'desc': 'sin datos'
        }

    sn = sorted(notes, key=lambda n: n.time_sec)
    curve_raw = []
    t = 0.0

    while t < total_dur:
        # Notas activas en esta ventana
        window = [n for n in sn
                  if n.time_sec < t + resolution and n.time_sec + n.duration > t]

        if len(window) < 2:
            curve_raw.append((t, 0.0))
            t += resolution
            continue

        pitches = [n.pitch for n in window]
        total_r = 0.0
        total_r_no_reg = 0.0  # sin penalización de registro, para medir impacto
        pairs = 0

        for i in range(len(pitches)):
            for j in range(i + 1, len(pitches)):
                p1, p2 = pitches[i], pitches[j]
                interval = abs(p2 - p1) % 12
                base_w = _ROUGHNESS_WEIGHTS.get(interval, 0.5)

                # Penalización por registro grave
                penalty = _LOW_REG_PENALTY if (p1 < _LOW_REG_THRESHOLD or
                                               p2 < _LOW_REG_THRESHOLD) else 1.0
                total_r        += base_w * penalty
                total_r_no_reg += base_w
                pairs += 1

        # Normalizar por √(pares) — calidad, no cantidad
        denom = math.sqrt(pairs) if pairs > 0 else 1
        roughness     = total_r        / denom
        roughness_nor = total_r_no_reg / denom

        curve_raw.append((t, roughness))
        t += resolution

    if not curve_raw:
        return {
            'curve': [], 'curve_norm': [], 'mean_roughness': 0.0,
            'peak_roughness': {'time': 0, 'value': 0},
            'valleys': [], 'roughness_arc': 'indeterminado',
            'register_impact': 0.0, 'desc': 'sin datos'
        }

    times_r = [x[0] for x in curve_raw]
    vals_r  = [x[1] for x in curve_raw]
    max_r   = max(vals_r) if vals_r else 1.0
    mean_r  = sum(vals_r) / len(vals_r)

    # Normalizar a [0..1]
    curve_norm = [(t2, v / max_r if max_r > 0 else 0.0) for t2, v in curve_raw]

    # Pico global
    peak_idx  = vals_r.index(max_r)
    peak_info = {'time': times_r[peak_idx], 'value': max_r}

    # Valles (momentos de máxima consonancia sensorial)
    valleys = []
    for i in range(1, len(vals_r) - 1):
        if vals_r[i] < vals_r[i-1] and vals_r[i] < vals_r[i+1] and vals_r[i] < mean_r * 0.5:
            valleys.append({'time': times_r[i], 'value': vals_r[i]})
    valleys.sort(key=lambda x: x['value'])

    # Arco de roughness
    n = len(vals_r)
    thirds = max(1, n // 3)
    r1 = sum(vals_r[:thirds]) / thirds
    r2 = sum(vals_r[thirds:2*thirds]) / max(1, thirds)
    r3 = sum(vals_r[2*thirds:]) / max(1, n - 2*thirds)
    std_r = math.sqrt(sum((v - mean_r)**2 for v in vals_r) / len(vals_r))
    rel_std = std_r / mean_r if mean_r > 0 else 0

    if rel_std < 0.15:
        roughness_arc = 'estable'
    elif r3 > r1 + r1 * 0.2:
        roughness_arc = 'crescendo'
    elif r3 < r1 - r1 * 0.2:
        roughness_arc = 'decrescendo'
    else:
        roughness_arc = 'accidentado'

    # Impacto del registro grave (estimado)
    # Comparar roughness con y sin penalización re-calculando una muestra
    sample_windows = [i for i in range(0, len(curve_raw), max(1, len(curve_raw)//20))]
    reg_penalty_sum = 0.0
    total_sum = sum(vals_r[i] for i in sample_windows) or 1
    for i in sample_windows:
        t2 = curve_raw[i][0]
        window = [n for n in sn if n.time_sec < t2 + resolution and n.time_sec + n.duration > t2]
        has_low = any(n.pitch < _LOW_REG_THRESHOLD for n in window)
        if has_low:
            reg_penalty_sum += vals_r[i] * (_LOW_REG_PENALTY - 1.0) / _LOW_REG_PENALTY
    register_impact = min(1.0, reg_penalty_sum / total_sum) if total_sum > 0 else 0.0

    arc_descs = {
        'estable':     'aspereza sensorial homogénea — textura acústica constante',
        'crescendo':   'aspereza creciente — la textura se vuelve progresivamente más áspera',
        'decrescendo': 'aspereza decreciente — la textura se suaviza hacia el final',
        'accidentado': 'aspereza irregular — alternancia de zonas ásperas y consonantes',
    }

    if mean_r < 0.3:
        quality = 'textura muy consonante — predominan intervalos puros y terceras'
    elif mean_r < 0.6:
        quality = 'textura moderadamente disonante — equilibrio entre consonancia y tensión sensorial'
    elif mean_r < 1.0:
        quality = 'textura áspera — disonancia sensorial elevada, tensión física sostenida'
    else:
        quality = 'textura muy áspera — alta densidad de intervalos disonantes en registro tenso'

    desc = (f"Roughness media {mean_r:.3f} — {quality}. "
            f"Arco: {arc_descs.get(roughness_arc, roughness_arc)}. "
            f"Pico de aspereza máxima en {ts_str_local(peak_info['time'])} "
            f"(val={peak_info['value']:.3f}). "
            + (f"Impacto del registro grave: {register_impact*100:.0f}% de la roughness total. " if register_impact > 0.05 else "")
            + (f"{len(valleys)} momento(s) de consonancia sensorial máxima." if valleys else ""))

    return {
        'curve':           curve_raw,
        'curve_norm':      curve_norm,
        'mean_roughness':  mean_r,
        'max_roughness':   max_r,
        'peak_roughness':  peak_info,
        'valleys':         valleys[:4],
        'roughness_arc':   roughness_arc,
        'roughness_arc_desc': arc_descs.get(roughness_arc, roughness_arc),
        'register_impact': register_impact,
        'desc':            desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v10-B. MARKOV MELÓDICO DE 2º ORDEN
# ═══════════════════════════════════════════════════════════════
#
#  Cadena de Markov de orden 2 sobre pares (intervalo, duración
#  cuantizada). Adaptado de midi_dna_unified.py/MarkovMelody
#  para trabajar directamente con NoteEvent (sin music21).
#
#  Qué mide:
#    - predecibilidad_media: probabilidad media de cada transición
#      (alta → la melodía "suena natural/esperada", baja → sorpresiva)
#    - entropia_markov: entropía de la distribución de transiciones
#      (alta → vocabulario melódico rico, baja → repetitivo)
#    - singularidades: transiciones con probabilidad < 5% (gestos únicos)
#    - perfil_estilo: 'predecible' | 'equilibrado' | 'sorpresivo' | 'caótico'
#
#  La diferencia con analyze_melodic_expectation (ya existente) es que
#  éste usa reglas a priori de Narmour, mientras que Markov aprende la
#  distribución real de la pieza y mide cuánto se desvía de ella.

_DUR_BINS = [0.0, 0.15, 0.35, 0.6, 1.1, 2.1, 999]  # límites de cuantización
_DUR_LABELS = ['64th', '32nd', '8th', 'q', 'h', 'w+']

def _quantize_dur(dur_sec: float, bpm: float) -> str:
    """Convierte duración en segundos a valor rítmico cuantizado."""
    beat = 60.0 / bpm if bpm > 0 else 0.5
    ratio = dur_sec / beat  # en tiempos de negra
    for i, lim in enumerate(_DUR_BINS[1:]):
        if ratio < lim:
            return _DUR_LABELS[i]
    return _DUR_LABELS[-1]

def analyze_melodic_markov(
    notes: List[NoteEvent],
    key_root: int,
    avg_bpm: float,
) -> Dict:
    """
    Analiza la melodía principal mediante una cadena de Markov de 2º orden.

    Estado = (intervalo_anterior, duración_cuantizada_anterior)
    Transición = (intervalo_siguiente, duración_siguiente)

    Construye la tabla de transiciones y calcula:
      - Predecibilidad de cada nota (P(siguiente | estado_actual))
      - Entropía de Markov (bits) — diversidad del vocabulario melódico
      - Gestos únicos — transiciones que ocurren exactamente una vez
      - Motivos estadísticos — transiciones más frecuentes (el DNA melódico)
      - Perfil de estilo

    Devuelve:
      transition_table  : dict {estado: Counter(transiciones)}
      mean_probability  : probabilidad media de cada transición observada
      entropy_markov    : entropía media en bits (log2 del vocabulario efectivo)
      surprise_curve    : lista de (time, surprise) — inverso de probabilidad
      singular_gestures : transiciones que ocurren 1 sola vez (unicidad)
      top_transitions   : las 5 transiciones más frecuentes (firma melódica)
      style_profile     : 'predecible' | 'equilibrado' | 'sorpresivo' | 'caótico'
      desc              : descripción en lenguaje natural
    """
    mel = get_mel_track(notes)
    sn  = sorted(mel, key=lambda n: n.time_sec)

    if len(sn) < 6:
        return {
            'transition_table': {}, 'mean_probability': 0.5,
            'entropy_markov': 0.0, 'surprise_curve': [],
            'singular_gestures': [], 'top_transitions': [],
            'style_profile': 'indeterminado', 'desc': 'melodía insuficiente para Markov'
        }

    # Construir secuencia de (intervalo, duración_q)
    seq = []
    for i in range(len(sn) - 1):
        iv  = sn[i+1].pitch - sn[i].pitch          # intervalo en semitonos (con signo)
        iv  = max(-12, min(12, iv))                  # limitar a ±octava
        dur = _quantize_dur(sn[i].duration, avg_bpm)
        seq.append((iv, dur))

    if len(seq) < 4:
        return {
            'transition_table': {}, 'mean_probability': 0.5,
            'entropy_markov': 0.0, 'surprise_curve': [],
            'singular_gestures': [], 'top_transitions': [],
            'style_profile': 'indeterminado', 'desc': 'secuencia demasiado corta'
        }

    # Construir tabla de transiciones (orden 2)
    table: Dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for i in range(len(seq) - 2):
        state = (seq[i], seq[i+1])       # dos estados anteriores
        nxt   = seq[i+2]                 # siguiente
        table[state][nxt] += 1

    # Calcular probabilidades y entropía por estado
    entropies    = []
    probs_global = []

    for state, counts in table.items():
        total = sum(counts.values())
        for sym, cnt in counts.items():
            p = cnt / total
            probs_global.append(p)
        # Entropía del estado: -Σ p·log2(p)
        e = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        entropies.append(e)

    mean_prob   = sum(probs_global) / len(probs_global) if probs_global else 0.5
    entropy_avg = sum(entropies) / len(entropies)       if entropies    else 0.0

    # Curva de sorpresa (1 - P de cada transición observada)
    surprise_curve = []
    for i in range(len(seq) - 2):
        state = (seq[i], seq[i+1])
        nxt   = seq[i+2]
        if state in table:
            total = sum(table[state].values())
            p = table[state][nxt] / total
        else:
            p = 0.01  # transición no vista → máxima sorpresa
        surprise = 1.0 - p
        t_note = sn[i+2].time_sec
        surprise_curve.append((t_note, surprise))

    # Gestos únicos (transiciones que ocurren 1 vez)
    singular = []
    for state, counts in table.items():
        for sym, cnt in counts.items():
            if cnt == 1:
                iv_prev, dur_prev = state[1]
                iv_next, dur_next = sym
                singular.append({
                    'interval_in':  iv_prev,
                    'dur_in':       dur_prev,
                    'interval_out': iv_next,
                    'dur_out':      dur_next,
                })

    # Top transiciones (firma melódica)
    all_transitions: collections.Counter = collections.Counter()
    for state, counts in table.items():
        for sym, cnt in counts.items():
            all_transitions[(state[1], sym)] += cnt

    top_transitions = []
    for (from_sym, to_sym), cnt in all_transitions.most_common(5):
        iv_f, dur_f = from_sym
        iv_t, dur_t = to_sym
        direction_f = '↑' if iv_f > 0 else ('↓' if iv_f < 0 else '–')
        direction_t = '↑' if iv_t > 0 else ('↓' if iv_t < 0 else '–')
        top_transitions.append({
            'from': f"{direction_f}{abs(iv_f)}st/{dur_f}",
            'to':   f"{direction_t}{abs(iv_t)}st/{dur_t}",
            'count': cnt,
            'desc': f"{direction_f}{abs(iv_f)} semitonos [{dur_f}] → {direction_t}{abs(iv_t)} st [{dur_t}]",
        })

    # Perfil de estilo
    if mean_prob > 0.60:
        style_profile = 'predecible'
        style_desc = ('melodía altamente predecible — cada nota sigue de forma estadísticamente '
                      'natural a las anteriores; máxima coherencia interna')
    elif mean_prob > 0.35:
        style_profile = 'equilibrado'
        style_desc = ('melodía equilibrada — combina gestos esperados con sorpresas puntuales')
    elif mean_prob > 0.18:
        style_profile = 'sorpresivo'
        style_desc = ('melodía sorpresiva — las transiciones desafían frecuentemente las expectativas '
                      'estadísticas; escritura idiomáticamente rica')
    else:
        style_profile = 'caótico'
        style_desc = ('melodía muy imprevisible — pocas transiciones se repiten; '
                      'vocabulario extremadamente diverso o atonalismo de facto')

    n_unique_states = len(table)
    n_total_trans   = sum(sum(c.values()) for c in table.values())

    desc = (f"Perfil Markov de orden 2: {style_desc}. "
            f"Probabilidad media de transición: {mean_prob:.2f}. "
            f"Entropía melódica: {entropy_avg:.2f} bits/estado "
            f"({n_unique_states} estados únicos sobre {n_total_trans} transiciones). "
            + (f"{len(singular)} gestos únicos (hapax melódicos)." if singular else ""))

    return {
        'transition_table':  dict(table),
        'mean_probability':  mean_prob,
        'entropy_markov':    entropy_avg,
        'surprise_curve':    surprise_curve,
        'singular_gestures': singular[:8],
        'top_transitions':   top_transitions,
        'style_profile':     style_profile,
        'style_desc':        style_desc,
        'n_unique_states':   n_unique_states,
        'n_transitions':     n_total_trans,
        'desc':              desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v10-C. ENRIQUECIMIENTO SEMÁNTICO (theorist.py SEMANTIC_DICT)
# ═══════════════════════════════════════════════════════════════
#
#  Adaptación directa del SEMANTIC_DICTIONARY de theorist.py:
#  dado el análisis ya realizado (modo, BPM, tensión, densidad,
#  valencia), busca el concepto semántico más próximo y enriquece
#  el informe con:
#    - Nombre del estado emocional canónico
#    - Justificación teórica (Mattheson, Huron, Sloboda...)
#    - Arco narrativo asociado (hero, tragedy, romance, mystery...)
#    - Persona teórica más afín (Schenker, Fux, Jazz, Ravel...)
#    - Descripción de la paleta orquestal ideal
#    - Grado de ajuste con el material real [0..1]
#
#  No añade dependencias. Usa solo los datos del análisis existente.

_SEM_DICT = {
    'tristeza': {
        'aliases': ['triste','sad','melancólico','melancholy','nostalgia','dolor'],
        'key_modes': ['minor'], 'tempo_range': (52, 84),
        'tension': (0.2, 0.55), 'arc': 'tragedy',
        'justification': 'La modalidad menor es la asociación más consolidada en la teoría tonal occidental (Mattheson, 1739). Tempo lento y densidad ligera sostienen la contemplación.',
        'persona': 'romantic',
    },
    'alegría': {
        'aliases': ['alegre','happy','feliz','joyful','festivo','jubiloso'],
        'key_modes': ['major','lydian'], 'tempo_range': (108, 168),
        'tension': (0.1, 0.45), 'arc': 'hero',
        'justification': 'Modo mayor, tempo allegro y armonías diatónicas simples son los marcadores canónicos de júbilo (Kirnberger, Affektenlehre).',
        'persona': 'baroque',
    },
    'ansiedad': {
        'aliases': ['ansioso','anxious','angustia','inquietud','urgencia'],
        'key_modes': ['minor','phrygian','locrian'], 'tempo_range': (126, 200),
        'tension': (0.55, 1.0), 'arc': 'hero',
        'justification': 'Alta densidad rítmica con tensión armónica inestable y registro agudo: los correlatos físicos de la ansiedad se traducen directamente en parámetros musicales.',
        'persona': 'spectral',
    },
    'calma': {
        'aliases': ['tranquilo','sereno','peaceful','calm','paz','quietud'],
        'key_modes': ['major','lydian','mixolydian'], 'tempo_range': (44, 80),
        'tension': (0.0, 0.30), 'arc': 'meditation',
        'justification': 'Tensión mínima, densidad esparsa y tempo lento reducen la carga cognitiva y producen reposo perceptual (Huron, Sweet Anticipation, 2006).',
        'persona': 'modal',
    },
    'misterio': {
        'aliases': ['misterioso','mysterious','enigmático','ambiguo','inquietante'],
        'key_modes': ['phrygian','locrian','dorian'], 'tempo_range': (56, 100),
        'tension': (0.35, 0.70), 'arc': 'mystery',
        'justification': 'El modo frigio con su b2 crea ambigüedad tonal que impide la resolución predecible. La densidad baja aumenta la tensión por ausencia.',
        'persona': 'ravel',
    },
    'grandiosidad': {
        'aliases': ['grandioso','majestuoso','épico','heroico','poderoso','sublime'],
        'key_modes': ['major','mixolydian'], 'tempo_range': (80, 130),
        'tension': (0.40, 0.80), 'arc': 'hero',
        'justification': 'Modo mayor con armonías extendidas, densidad alta y registro amplio activan la respuesta de escalofrío musical asociada a lo sublime (Sloboda, 1991).',
        'persona': 'romantic',
    },
    'melancolía': {
        'aliases': ['melancolia','wistful','añoranza','soledad','introspección'],
        'key_modes': ['dorian','minor'], 'tempo_range': (52, 88),
        'tension': (0.20, 0.55), 'arc': 'romance',
        'justification': 'El dórico combina la oscuridad del menor con la VI mayor, creando la ambigüedad tonal característica de la melancolía (jazz modal, escalas de blues).',
        'persona': 'jazz',
    },
    'ternura': {
        'aliases': ['tierno','delicado','suave','gentle','sweet','íntimo','cariño'],
        'key_modes': ['major','lydian'], 'tempo_range': (56, 92),
        'tension': (0.10, 0.35), 'arc': 'romance',
        'justification': 'Registro medio-agudo, dinámica suave y cadencias evitadas o a la mediante: la ternura evita la conclusividad.',
        'persona': 'ravel',
    },
    'expectativa': {
        'aliases': ['espera','suspense','anticipación','inminente','tensión sin resolver'],
        'key_modes': ['phrygian','minor','dorian'], 'tempo_range': (58, 100),
        'tension': (0.50, 0.85), 'arc': 'mystery',
        'justification': 'Dominantes sin resolver y semicadencias repetidas como marca estructural. El modo frigio resiste la resolución conclusiva (Ravel, Ma mère l\'Oye).',
        'persona': 'spectral',
    },
    'tormenta': {
        'aliases': ['storm','tempestad','turbulento','caos','furioso','agitado'],
        'key_modes': ['minor','phrygian','locrian'], 'tempo_range': (132, 220),
        'tension': (0.65, 1.0), 'arc': 'tragedy',
        'justification': 'Densidad máxima, tempo agitato, armonías disminuidas y aumentadas, registro extremo. Modelo: Beethoven op.31 no.2.',
        'persona': 'romantic',
    },
    'serenidad nocturna': {
        'aliases': ['noche','nocturno','night','medianoche','oscuridad suave'],
        'key_modes': ['minor','dorian'], 'tempo_range': (44, 76),
        'tension': (0.15, 0.45), 'arc': 'meditation',
        'justification': 'El nocturno como género: modo menor, registro grave-medio, densidad esparsa, dinámicas pp-mp, cadencias evitadas (Chopin, Nocturnos op.9).',
        'persona': 'romantic',
    },
    'vacío': {
        'aliases': ['vacio','empty','desolado','abandono','silencio interior'],
        'key_modes': ['phrygian','locrian'], 'tempo_range': (40, 68),
        'tension': (0.0, 0.25), 'arc': 'meditation',
        'justification': 'El vacío como principio compositivo: Feldman (Rothko Chapel) y Satie usan la ausencia como argumento. Los silencios son parte del material.',
        'persona': 'spectral',
    },
}

_PERSONAS = {
    'schenker':  ('Analista schenkeriano',      'Identifica el Ursatz, la línea fundamental y las dominantes estructurales.'),
    'fux':       ('Contrapuntista (Fux)',         'Aplica reglas de contrapunto por especies: movimiento contrario, prohibición de paralelas.'),
    'ravel':     ('Impresionista (Ravel/Debussy)','Armonía como color. Escalas de tonos enteros, quintas paralelas, sin cadencia auténtica obligatoria.'),
    'jazz':      ('Teórico de jazz',              'ii-V-I, sustituciones de tritono, tensiones 9ª/11ª/13ª, reharmonización cromática.'),
    'spectral':  ('Compositor espectral',         'Alturas derivadas de la serie armónica. El timbre es el parámetro compositivo central.'),
    'baroque':   ('Contrapuntista barroco (Bach)','Fuga, coral, secuencias, inversión motívica y contrapunto imitativo.'),
    'romantic':  ('Compositor romántico',         'Arco emocional largo, cromatismo, modulaciones a terceras, rubato y expresividad.'),
    'modal':     ('Compositor modal',             'Los modos no son tonalidades: cada uno tiene color intrínseco. Sin dominante como agente cadencial.'),
}

_ARC_DESCS = {
    'hero':      'arco heroico — tensión que crece hacia una victoria o transformación',
    'tragedy':   'arco trágico — acumulación dramática sin resolución luminosa',
    'romance':   'arco de romance — arco emocional íntimo y circular',
    'mystery':   'arco de misterio — tensión sostenida sin revelación completa',
    'meditation':'arco contemplativo — sin clímax, estado de presencia sostenida',
    'rondo':     'arco de rondó — retorno periódico al material principal',
    'sonata':    'arco de sonata — dialéctica entre exposición, desarrollo y recapitulación',
}

def analyze_semantic_enrichment(
    mode: str,
    avg_bpm: float,
    overall_tension: float,
    dynamic_valence: Dict,
    catharsis: Dict,
    narrative_intention: Dict,
    silence_analysis: Dict,
) -> Dict:
    """
    Busca el concepto semántico más próximo al perfil musical real
    y enriquece el análisis con vocabulario de la teoría del afecto.

    Algoritmo de matching:
      1. Penaliza o bonifica según modo (match exacto = +0.35)
      2. Penaliza si BPM está fuera del rango esperado
      3. Penaliza si la tensión media está fuera del rango esperado
      4. Bonifica si la valencia dinámica es coherente con el concepto
      5. Bonifica si el arco de catarsis corresponde al arco del concepto

    Devuelve:
      concept         : nombre del concepto semántico más próximo
      fit_score       : grado de ajuste [0..1]
      justification   : justificación teórica del concepto
      arc             : arco narrativo asociado al concepto
      arc_desc        : descripción del arco
      persona         : persona teórica más afín
      persona_desc    : descripción de la persona
      persona_lens    : cómo analizaría esta pieza esa persona
      alternatives    : los 2 conceptos alternativos con su score
      desc            : descripción narrativa completa
    """
    mean_val = dynamic_valence.get('mean_valence', 0.0)
    cat_type = catharsis.get('catharsis_type', 'ausente')
    arc_inferred = narrative_intention.get('archetype', '')
    silence_ratio = (silence_analysis or {}).get('silence_ratio', 0.05)

    scores: Dict[str, float] = {}

    for concept, data in _SEM_DICT.items():
        s = 0.0

        # 1. Modo
        if mode in data['key_modes']:
            s += 0.35
        elif any(m in ['minor', 'phrygian', 'locrian'] for m in data['key_modes']) and \
             mode in ['minor', 'phrygian', 'locrian']:
            s += 0.12  # misma familia oscura

        # 2. Tempo
        lo, hi = data['tempo_range']
        if lo <= avg_bpm <= hi:
            s += 0.20
        else:
            penalty = min(abs(avg_bpm - lo), abs(avg_bpm - hi)) / 200.0
            s -= min(penalty, 0.15)

        # 3. Tensión
        t_lo, t_hi = data['tension']
        if t_lo <= overall_tension <= t_hi:
            s += 0.20
        else:
            t_dist = min(abs(overall_tension - t_lo), abs(overall_tension - t_hi))
            s -= min(t_dist * 0.5, 0.15)

        # 4. Valencia
        if mean_val > 0.15 and mode in ['major', 'lydian', 'mixolydian']:
            if concept in ['alegría', 'grandiosidad', 'ternura', 'calma']:
                s += 0.10
        elif mean_val < -0.15:
            if concept in ['tristeza', 'ansiedad', 'melancolía', 'misterio', 'tormenta', 'vacío']:
                s += 0.10

        # 5. Catarsis / arco
        arc = data['arc']
        if cat_type == 'clásica' and arc in ('hero', 'tragedy', 'romance'):
            s += 0.08
        elif cat_type == 'negada' and arc in ('mystery', 'meditation'):
            s += 0.08
        elif cat_type == 'ausente' and arc == 'meditation':
            s += 0.08

        # 6. Silencio como material
        if silence_ratio > 0.15 and concept == 'vacío':
            s += 0.10

        scores[concept] = max(0.0, s)

    # Ordenar por score
    sorted_concepts = sorted(scores.items(), key=lambda x: -x[1])
    best_concept, best_score = sorted_concepts[0]
    alternatives = [{'concept': c, 'score': round(s2, 2)}
                    for c, s2 in sorted_concepts[1:3]]

    data = _SEM_DICT[best_concept]
    fit_score = min(1.0, best_score / 0.93)   # normalizar al máximo teórico
    persona_key = data.get('persona', 'romantic')
    persona_name, persona_desc = _PERSONAS.get(persona_key, ('?', '?'))
    arc = data['arc']
    arc_desc = _ARC_DESCS.get(arc, arc)

    # Cómo analizaría esta pieza esa persona
    persona_lens_map = {
        'schenker':  (f"Buscaría la Urlinie (línea fundamental descendente en la voz superior) "
                      f"y el Bassbrechung (arpegio I–V–I en el bajo), usando el cromatismo y "
                      f"las modulaciones como prolongaciones de la estructura profunda."),
        'fux':       (f"Evaluaría el movimiento de voces: ¿hay paralelas de 5ª o 8ª? "
                      f"¿Las disonancias se preparan y resuelven? "
                      f"La corrección contrapuntística es el criterio supremo."),
        'ravel':     (f"Leería la armonía como color, no como función. "
                      f"Los modos, las quintas paralelas y las escalas modales "
                      f"no son 'errores' sino recursos colorísticos deliberados."),
        'jazz':      (f"Identificaría los ii-V-I explícitos e implícitos, las sustituciones "
                      f"de tritono y las tensiones extendidas. "
                      f"El voice leading cromático es la firma del lenguaje."),
        'spectral':  (f"Maperaría cada nota a su posición en la serie armónica del fundamental. "
                      f"Las transiciones no son progresiones de acordes sino morphings espectrales. "
                      f"El timbre es el argumento."),
        'baroque':   (f"Buscaría la coherencia motívica y las secuencias armónicas "
                      f"(ciclos de quintas, Rosalia). La pieza es buena si cada voz "
                      f"tiene independencia melódica y el motivo se desarrolla."),
        'romantic':  (f"El arco emocional es el argumento central. Las modulaciones a terceras, "
                      f"el cromatismo ascendente y las dinámicas extremas "
                      f"son los recursos de la narrativa emocional."),
        'modal':     (f"Cada modo tiene una sonoridad intrínseca. "
                      f"La pieza no 'va' a la dominante: el modo define el color permanente. "
                      f"Los pedales y ostinati son el lenguaje, no el relleno."),
    }
    persona_lens = persona_lens_map.get(persona_key, '')

    fit_desc = ('alta correspondencia' if fit_score > 0.70 else
                'correspondencia moderada' if fit_score > 0.45 else
                'correspondencia aproximada')

    desc = (f"Concepto semántico más próximo: '{best_concept}' ({fit_desc}, ajuste={fit_score:.2f}). "
            f"{data['justification']} "
            f"Arco narrativo asociado: {arc_desc}. "
            f"Persona teórica afín: {persona_name} — {persona_desc}")

    return {
        'concept':       best_concept,
        'fit_score':     fit_score,
        'justification': data['justification'],
        'arc':           arc,
        'arc_desc':      arc_desc,
        'persona':       persona_key,
        'persona_name':  persona_name,
        'persona_desc':  persona_desc,
        'persona_lens':  persona_lens,
        'alternatives':  alternatives,
        'desc':          desc,
    }


# ═══════════════════════════════════════════════════════════════
#  v11. DETECCIÓN DE GÉNERO MUSICAL
# ═══════════════════════════════════════════════════════════════
#
#  Clasificador basado en reglas ponderadas + similitud de coseno
#  contra perfiles de 14 géneros. No requiere ML ni dependencias.
#
#  Features usadas (en orden de poder discriminativo):
#    1. Patrón rítmico: síncopa, acentuación, swing ratio
#    2. Compás (ts_num): 2, 3, 4, 6, 12
#    3. Progresiones cadenciales características
#    4. Modo / escala predominante
#    5. Patrón de bajo (walking, oom-pah, ostinato, boogie)
#    6. Densidad armónica (extensiones 7ª+)
#    7. Instrumentos GM (parcial, como indicio)
#    8. Tempo y su relación con el género
#    9. Cromatismo y blue notes
#   10. Uso del registro y contorno melódico

# ── Perfiles de género ──────────────────────────────────────────
# Cada género se define por un dict de features con valor esperado [0..1]
# y peso [0..1]. La similitud se calcula como suma ponderada de matches.

_GENRE_PROFILES = {
    # ─── TANGO ───
    'tango': {
        'label': 'Tango',
        'subgenres': ['tango porteño', 'milonga', 'vals criollo'],
        'regions': ['Argentina', 'Uruguay'],
        'features': {
            'ts_2_or_4':       (1.0, 0.12),  # compás 2/4 o 4/4
            'mode_minor':      (1.0, 0.14),  # modo menor predominante
            'habanera_rhythm': (1.0, 0.22),  # célula habanera (♩. ♪ ♩ ♩)
            'syncopation_high':(1.0, 0.15),  # síncopa alta
            'chromatic_moves': (1.0, 0.12),  # cromatismo frecuente
            'cadence_napl':    (1.0, 0.10),  # cadencia napolitana
            'bass_on_beat':    (1.0, 0.08),  # bajo en tiempos fuertes
            'tempo_medium':    (1.0, 0.07),  # 60-110 BPM
        },
        'description': 'Forma de danza rioplatense de finales del s.XIX. Compás 2/4 con célula rítmica de habanera, modo menor, cromatismo dramático y tensión expresiva.',
        'key_signature': 'Célula habanera: ♩. ♪ ♩ ♩ en compás 2/4',
    },
    # ─── VALS / WALTZ ───
    'vals': {
        'label': 'Vals / Waltz',
        'subgenres': ['vals vienés', 'vals romántico', 'vals lento', 'vals criollo'],
        'regions': ['Austria', 'Alemania', 'Europa central'],
        'features': {
            'ts_3':            (1.0, 0.35),  # compás 3/4 — diagnóstico
            'bass_on_1':       (1.0, 0.20),  # bajo en tiempo 1
            'chord_on_2_3':    (1.0, 0.18),  # acordes en 2 y 3
            'mode_major':      (1.0, 0.10),  # mayor predominante
            'stepwise_melody': (1.0, 0.08),  # melodía conjunta
            'tempo_medium':    (1.0, 0.05),  # 80-180 BPM
            'low_chromaticism':(1.0, 0.04),  # pocas notas cromáticas
        },
        'description': 'Danza en 3/4 originada en Austria s.XVIII. Patrón oom-pah-pah con bajo en tiempo 1 y acordes en 2-3. Melodía cantabile y armonia diatónica.',
        'key_signature': 'Compás 3/4 con bajo en 1 y acordes en 2-3',
    },
    # ─── CLÁSICA EUROPEA ───
    'clasica': {
        'label': 'Música clásica europea',
        'subgenres': ['barroco', 'clasicismo', 'romanticismo', 'impresionismo'],
        'regions': ['Europa occidental'],
        'features': {
            'cadence_perfect':  (1.0, 0.18),  # muchas cadencias perfectas V→I
            'voice_leading_smooth': (1.0, 0.14), # voice leading fluido
            'alberti_bass':     (1.0, 0.12),  # bajo Alberti
            'motif_development':(1.0, 0.12),  # desarrollo motívico
            'ts_4_or_3':        (1.0, 0.08),  # 3/4 o 4/4
            'instruments_orch': (1.0, 0.12),  # instrumentos orquestales GM
            'formal_structure': (1.0, 0.10),  # estructura formal clara
            'no_swing':         (1.0, 0.07),  # sin swing
            'diatonic_harmony': (1.0, 0.07),  # armonía diatónica
        },
        'description': 'Tradición académica europea s.XVII-XX. Armonía funcional tonal, desarrollo motívico, forma sonata/rondó, instrumentación orquestal.',
        'key_signature': 'Cadencias perfectas V→I, desarrollo motívico, forma estructurada',
    },
    # ─── ROCK / POP ───
    'rock': {
        'label': 'Rock / Pop',
        'subgenres': ['rock clásico', 'pop', 'rock duro', 'alternativo'],
        'regions': ['EE.UU.', 'Reino Unido'],
        'features': {
            'ts_4':             (1.0, 0.15),  # 4/4 casi exclusivo
            'backbeat':         (1.0, 0.25),  # acento en 2 y 4
            'mixolydian_prog':  (1.0, 0.15),  # I-bVII-IV mixolidio
            'pentatonic':       (1.0, 0.12),  # escala pentatónica
            'bass_on_4beats':   (1.0, 0.12),  # bajo en cuatro tiempos
            'guitar_gm':        (1.0, 0.10),  # guitarra GM (prog 25-31)
            'medium_high_tempo':(1.0, 0.08),  # 100-180 BPM
            'simple_cadences':  (1.0, 0.03),  # cadencias simples
        },
        'description': 'Género surgido en EE.UU./UK en los 50. Compás 4/4 con backbeat en 2 y 4, progresiones de acordes de poder, escala pentatónica y bajo eléctrico.',
        'key_signature': 'Backbeat en 2 y 4, I-bVII-IV, pentatónica',
    },
    # ─── JAZZ ───
    'jazz': {
        'label': 'Jazz',
        'subgenres': ['bebop', 'swing', 'cool jazz', 'jazz modal', 'jazz fusión'],
        'regions': ['EE.UU. (Nueva Orleans, Nueva York)'],
        'features': {
            'swing_ratio':      (1.0, 0.28),  # ratio de swing > 0.55
            'ii_V_I':           (1.0, 0.22),  # progresión ii-V-I
            'ext_chords':       (1.0, 0.18),  # acordes con 7ª/9ª/13ª
            'walking_bass':     (1.0, 0.15),  # bajo walking
            'ts_4':             (1.0, 0.07),  # 4/4 predominante
            'brass_woodwind_gm':(1.0, 0.07),  # saxo, trompeta GM
            'chromatic_melody': (1.0, 0.03),  # melodía cromática
        },
        'description': 'Género afroamericano surgido en Nueva Orleans ~1900. Swing, improvisación, armonía extendida (7ª, 9ª, 13ª), blues notes y estructura ii-V-I.',
        'key_signature': 'Swing ratio > 0.55, ii-V-I, acordes extendidos',
    },
    # ─── FLAMENCO ───
    'flamenco': {
        'label': 'Flamenco',
        'subgenres': ['soleá', 'bulería', 'siguiriya', 'rumba flamenca', 'fandango'],
        'regions': ['España (Andalucía)'],
        'features': {
            'phrygian_mode':    (1.0, 0.22),  # modo frigio o frigio andaluz
            'andalusian_cadence':(1.0, 0.25), # i-bVII-bVI-V cadencia andaluza
            'ts_12_or_3':       (1.0, 0.18),  # 12 tiempos o compás ternario
            'chromatic_melody': (1.0, 0.12),  # melodía cromática / melismática
            'guitar_gm':        (1.0, 0.10),  # guitarra GM
            'asymmetric_accent':(1.0, 0.08),  # acentuación asimétrica
            'low_ext_chords':   (1.0, 0.05),  # pocas extensiones jazzísticas
        },
        'description': 'Arte musical andaluz con raíces gitanas, árabes y judías. Compás de 12 tiempos (bulería/soleá), modo frigio andaluz, cadencia i-bVII-bVI-V.',
        'key_signature': 'Cadencia andaluza i-bVII-bVI-V, modo frigio, 12 tiempos',
    },
    # ─── BOSSA NOVA ───
    'bossa_nova': {
        'label': 'Bossa nova',
        'subgenres': ['bossa nova clásica', 'samba-jazz', 'MPB'],
        'regions': ['Brasil (Río de Janeiro)'],
        'features': {
            'clave_3_2':        (1.0, 0.22),  # patrón de clave 3-2
            'ts_4':             (1.0, 0.10),  # 4/4 sincopado
            'ext_chords_jazz':  (1.0, 0.20),  # maj7, min7, dom7 extensos
            'bass_on_1_3':      (1.0, 0.18),  # bajo en 1 y 3 (no en 2 y 4)
            'mode_major_minor_mix':(1.0, 0.12),# mezcla mayor/menor
            'medium_tempo':     (1.0, 0.08),  # 80-130 BPM
            'stepwise_melody':  (1.0, 0.05),  # melodía lirica conjunta
            'guitar_gm':        (1.0, 0.05),  # guitarra GM
        },
        'description': 'Fusión brasileña de samba y jazz surgida ~1958. Ritmo de clave 3-2, acordes extendidos jazísticos, bajo en 1 y 3, melodía lírica.',
        'key_signature': 'Clave 3-2, maj7/min7 extendidos, bajo en tiempos 1 y 3',
    },
    # ─── BLUES ───
    'blues': {
        'label': 'Blues',
        'subgenres': ['delta blues', 'chicago blues', 'blues eléctrico', 'blues rápido'],
        'regions': ['EE.UU. (Mississippi, Chicago)'],
        'features': {
            'blues_form_12':    (1.0, 0.25),  # estructura de 12 compases
            'dom7_on_tonic':    (1.0, 0.22),  # I7 como tónica (no dominante)
            'blue_notes':       (1.0, 0.18),  # b3, b5, b7 melódicos
            'shuffle':          (1.0, 0.15),  # shuffle tripletado
            'I7_IV7_V7':        (1.0, 0.12),  # todos los acordes son dom7
            'ts_4_or_12_8':     (1.0, 0.05),  # 4/4 o 12/8
            'guitar_gm':        (1.0, 0.03),  # guitarra GM
        },
        'description': 'Género afroamericano del Delta (s.XIX-XX). Estructura de 12 compases, dominantes sobre la tónica (I7), blue notes (b3 b5 b7), shuffle tripletado.',
        'key_signature': 'I7-IV7-V7, blue notes b3/b5/b7, shuffle, 12 compases',
    },
    # ─── FUNK ───
    'funk': {
        'label': 'Funk',
        'subgenres': ['funk clásico', 'p-funk', 'soul funk', 'neo-soul'],
        'regions': ['EE.UU.'],
        'features': {
            'syncopation_16th': (1.0, 0.28),  # síncopa en subdivisión de 16
            'ts_4':             (1.0, 0.12),  # 4/4 estricto
            'dom7_vamps':       (1.0, 0.18),  # vamps en dom7 (estáticos)
            'bass_syncopated':  (1.0, 0.20),  # bajo muy sincopado
            'groove_tight':     (1.0, 0.12),  # groove repetitivo
            'bass_electric_gm': (1.0, 0.08),  # bajo eléctrico GM
            'medium_tempo':     (1.0, 0.02),  # 80-120 BPM
        },
        'description': 'Género afroamericano surgido ~1965. Groove de 16 corcheas con síncopa intensa, bajo eléctrico protagonista, vamps en dom7, mínima progresión armónica.',
        'key_signature': 'Síncopa en 16as, bajo sincopado, vamps dom7 estáticos',
    },
    # ─── PASODOBLE / MARCHA ───
    'pasodoble': {
        'label': 'Pasodoble / Marcha',
        'subgenres': ['pasodoble español', 'marcha militar', 'chotis'],
        'regions': ['España', 'contexto militar europeo'],
        'features': {
            'ts_2_or_6_8':      (1.0, 0.30),  # 2/4 o 6/8 marcial
            'mode_major':       (1.0, 0.15),  # mayor predominante
            'march_rhythm':     (1.0, 0.25),  # ritmo marcato regular
            'brass_gm':         (1.0, 0.15),  # metales GM (prog 56-63)
            'simple_harmony':   (1.0, 0.08),  # I-IV-V diatónico
            'fast_tempo':       (1.0, 0.07),  # 100-140 BPM
        },
        'description': 'Danza española en 2/4 con raíces militares. Ritmo marcato regular, modo mayor, armonía diatónica I-IV-V, uso de metales.',
        'key_signature': 'Compás 2/4, ritmo marcato, modo mayor, metales GM',
    },
    # ─── SAMBA ───
    'samba': {
        'label': 'Samba',
        'subgenres': ['samba de enredo', 'pagode', 'samba-canção'],
        'regions': ['Brasil (Bahía, Río de Janeiro)'],
        'features': {
            'clave_2_3':        (1.0, 0.22),  # clave 2-3 o anticipación
            'ts_2_4':           (1.0, 0.15),  # 2/4
            'syncopation_high': (1.0, 0.20),  # síncopa alta
            'bass_on_beat1':    (1.0, 0.15),  # bajo en tiempo fuerte
            'fast_tempo':       (1.0, 0.12),  # 90-140 BPM
            'major_mode':       (1.0, 0.10),  # mayor alegre
            'percussion_dense': (1.0, 0.06),  # percusión densa (ch.9 activo)
        },
        'description': 'Género afrobrasileño surgido en Bahía. Compás 2/4, síncopa intensa, clave 2-3, tempo vivo, melodía alegre en modo mayor.',
        'key_signature': 'Compás 2/4, clave 2-3, síncopa alta, tempo vivo',
    },
    # ─── REGGAE ───
    'reggae': {
        'label': 'Reggae',
        'subgenres': ['roots reggae', 'dancehall', 'ska', 'rocksteady'],
        'regions': ['Jamaica'],
        'features': {
            'skank_chord':      (1.0, 0.30),  # acordes en contratiempos 2+ y 4+
            'ts_4':             (1.0, 0.12),  # 4/4
            'bass_dominant':    (1.0, 0.18),  # bajo protagonista y melódico
            'slow_medium_tempo':(1.0, 0.12),  # 60-100 BPM
            'offbeat_accent':   (1.0, 0.18),  # énfasis en contratiempos
            'minor_mode':       (1.0, 0.05),  # menor frecuente
            'root_5_bass':      (1.0, 0.05),  # bajo en tónica y dominante
        },
        'description': 'Género jamaicano ~1960. Compás 4/4 con acordes en contratiempos (skank), bajo melódico protagonista, tempo moderado.',
        'key_signature': 'Skank (acordes en contratiempos 2+ y 4+), bajo melódico',
    },
    # ─── COUNTRY / BLUEGRASS ───
    'country': {
        'label': 'Country / Bluegrass',
        'subgenres': ['country clásico', 'bluegrass', 'country-rock'],
        'regions': ['EE.UU. (Sur, Appalachia)'],
        'features': {
            'ts_4':             (1.0, 0.12),  # 4/4
            'major_mode':       (1.0, 0.18),  # mayor casi siempre
            'I_IV_V_simple':    (1.0, 0.22),  # I-IV-V diatónico simple
            'banjo_guitar_gm':  (1.0, 0.15),  # guitarra/banjo GM
            'medium_tempo':     (1.0, 0.08),  # 80-160 BPM
            'no_ext_chords':    (1.0, 0.12),  # sin acordes extendidos
            'bass_alternating': (1.0, 0.08),  # bajo alternante tónica/5ª
            'backbeat_light':   (1.0, 0.05),  # backbeat suave
        },
        'description': 'Tradición musical del sur de EE.UU. Compás 4/4, modo mayor, I-IV-V diatónico simple, guitarra y banjo, sin extensiones jazzísticas.',
        'key_signature': 'I-IV-V diatónico, modo mayor, guitarra GM, sin extensiones',
    },
    # ─── LATIN (MAMBO / SALSA) ───
    'latin': {
        'label': 'Latin (Mambo / Salsa)',
        'subgenres': ['mambo', 'salsa', 'cha-cha-chá', 'cumbia', 'merengue'],
        'regions': ['Cuba', 'Puerto Rico', 'Colombia'],
        'features': {
            'clave_3_2_or_2_3': (1.0, 0.25),  # clave latinoamericana
            'ts_4':             (1.0, 0.10),  # 4/4
            'syncopation_high': (1.0, 0.18),  # síncopa alta
            'major_mode':       (1.0, 0.10),  # mayor / mixolidio
            'brass_section_gm': (1.0, 0.12),  # metales GM
            'fast_medium_tempo':(1.0, 0.10),  # 90-200 BPM
            'montuno_harmony':  (1.0, 0.10),  # I-IV o I-V vamp
            'percussion_ch9':   (1.0, 0.05),  # canal 9 activo (percusión)
        },
        'description': 'Familia de géneros afrocaribeños. Clave 3-2 o 2-3, síncopa intensa, sección de metales, montuno armónico (vamp I-IV o I-V).',
        'key_signature': 'Clave 3-2/2-3, síncopa, metales GM, montuno I-IV',
    },
}

# ── Subgénero refinado por scores internos ──────────────────────

_SUBGENRE_RULES = {
    'tango': [
        ('tango porteño', lambda f: f.get('tempo_slow', 0) > 0.5 and f.get('chromatic_high', 0) > 0.5),
        ('milonga',       lambda f: f.get('fast_tempo', 0) > 0.6 and f.get('habanera', 0) > 0.5),
        ('vals criollo',  lambda f: f.get('ts_3', 0) > 0.5),
    ],
    'jazz': [
        ('bebop',       lambda f: f.get('fast_tempo', 0) > 0.6 and f.get('chromatic_melody', 0) > 0.5),
        ('swing',       lambda f: f.get('medium_tempo', 0) > 0.5 and f.get('swing_high', 0) > 0.6),
        ('jazz modal',  lambda f: f.get('static_harmony', 0) > 0.5),
        ('cool jazz',   lambda f: f.get('slow_tempo', 0) > 0.4 and f.get('low_tension', 0) > 0.4),
    ],
    'blues': [
        ('delta blues',    lambda f: f.get('slow_tempo', 0) > 0.5),
        ('chicago blues',  lambda f: f.get('medium_tempo', 0) > 0.5 and f.get('fast_tempo', 0) < 0.3),
        ('blues rápido',   lambda f: f.get('fast_tempo', 0) > 0.6),
    ],
    'clasica': [
        ('barroco',        lambda f: f.get('counterpoint_rich', 0) > 0.5),
        ('romanticismo',   lambda f: f.get('chromatic_high', 0) > 0.5 and f.get('dynamic_range', 0) > 0.5),
        ('impresionismo',  lambda f: f.get('modal_harmony', 0) > 0.4),
        ('clasicismo',     lambda f: True),  # fallback
    ],
}


def _extract_genre_features(
    notes: List[NoteEvent],
    chords: List[ChordEvent],
    rhythm: Dict,
    ts_num: int,
    avg_bpm: float,
    mode: str,
    key_root: int,
    cadences: Dict,
    meta: Dict,
    total_dur: float,
) -> Dict:
    """
    Extrae el vector de features para la clasificación de género.
    Devuelve un dict {feature_name: score [0..1]}.
    """
    f: Dict[str, float] = {}

    sn = sorted(notes, key=lambda n: n.time_sec)

    # ── COMPÁS ────────────────────────────────────────────────
    f['ts_2_or_4']     = 1.0 if ts_num in (2, 4) else 0.0
    f['ts_3']          = 1.0 if ts_num == 3 else 0.0
    f['ts_4']          = 1.0 if ts_num == 4 else 0.0
    f['ts_2_4']        = 1.0 if ts_num == 2 else 0.0
    f['ts_12_or_3']    = 1.0 if ts_num in (3, 6, 12) else 0.0
    f['ts_4_or_3']     = 1.0 if ts_num in (3, 4) else 0.0
    f['ts_4_or_12_8']  = 1.0 if ts_num in (4, 6) else 0.0
    f['ts_2_or_6_8']   = 1.0 if ts_num in (2, 6) else 0.0

    # ── MODO ──────────────────────────────────────────────────
    f['mode_minor']    = 1.0 if mode in ('minor', 'phrygian', 'dorian', 'locrian') else 0.0
    f['mode_major']    = 1.0 if mode in ('major', 'lydian', 'mixolydian') else 0.0
    f['major_mode']    = f['mode_major']
    f['minor_mode']    = f['mode_minor']
    f['phrygian_mode'] = 1.0 if mode == 'phrygian' else (0.6 if mode in ('minor', 'locrian') else 0.0)
    f['modal_harmony'] = 1.0 if mode in ('dorian', 'phrygian', 'lydian', 'mixolydian', 'locrian') else 0.0
    f['mode_major_minor_mix'] = 0.7  # aproximación — siempre hay mezcla en bossa

    # ── TEMPO ─────────────────────────────────────────────────
    f['tempo_slow']       = 1.0 if avg_bpm < 70  else max(0.0, 1.0 - (avg_bpm - 70) / 30)
    f['slow_medium_tempo']= 1.0 if avg_bpm < 100 else max(0.0, 1.0 - (avg_bpm - 100) / 40)
    f['medium_tempo']     = 1.0 if 70 <= avg_bpm <= 130 else (max(0.0, 1.0 - abs(avg_bpm - 100) / 50))
    f['tempo_medium']     = f['medium_tempo']
    f['medium_high_tempo']= 1.0 if 100 <= avg_bpm <= 180 else max(0.0, 1.0 - abs(avg_bpm - 140) / 60)
    f['fast_tempo']       = 1.0 if avg_bpm > 130 else max(0.0, (avg_bpm - 80) / 50)
    f['fast_medium_tempo']= 1.0 if avg_bpm > 90  else max(0.0, (avg_bpm - 60) / 60)

    # ── SÍNCOPA ───────────────────────────────────────────────
    sync = rhythm.get('syncopation', 0.0)
    f['syncopation_high']  = min(1.0, sync * 2.0)
    f['syncopation_16th']  = min(1.0, sync * 2.5)  # funk: síncopa extrema
    f['no_swing']          = max(0.0, 1.0 - sync * 1.5)

    # ── SWING RATIO ───────────────────────────────────────────
    # Estimar swing ratio desde el análisis de ritmo existente
    swing_r = rhythm.get('swing_ratio', 0.0) if 'swing_ratio' in rhythm else 0.0
    if swing_r == 0.0:
        # Aproximar: contar pares de corcheas con ratio duración > 1.4
        beat = 60.0 / avg_bpm if avg_bpm > 0 else 0.5
        eighth = beat / 2
        triplet_ratios = []
        for i in range(len(sn) - 1):
            gap = sn[i+1].time_sec - sn[i].time_sec
            if 0.15 * beat < sn[i].duration < 0.85 * beat:
                if 0.15 * beat < gap < 0.85 * beat:
                    ratio = sn[i].duration / max(gap, 0.01)
                    if 1.2 < ratio < 3.5:
                        triplet_ratios.append(ratio)
        if triplet_ratios:
            swing_r = min(1.0, (sum(triplet_ratios) / len(triplet_ratios) - 1.0) / 1.5)
        else:
            swing_r = 0.0
    f['swing_ratio']  = swing_r
    f['swing_high']   = swing_r
    f['shuffle']      = min(1.0, swing_r * 1.2)

    # ── PATRONES DE BAJO ──────────────────────────────────────
    bass_notes = [n for n in sn if n.pitch < 55]

    # Walking bass: notas de bajo cambian en cada tiempo
    if bass_notes and total_dur > 0:
        beat_dur = 60.0 / avg_bpm if avg_bpm > 0 else 0.5
        bass_pcs = [n.pitch % 12 for n in bass_notes]
        # Contar cambios de PC entre notas de bajo consecutivas
        bass_changes = sum(1 for i in range(len(bass_pcs)-1) if bass_pcs[i] != bass_pcs[i+1])
        bass_change_ratio = bass_changes / max(len(bass_pcs)-1, 1)
        f['walking_bass']    = min(1.0, bass_change_ratio * 1.5)
        f['bass_dominant']   = min(1.0, bass_change_ratio * 1.2)
        f['bass_syncopated'] = f['syncopation_high'] * bass_change_ratio
        f['bass_electric_gm']= 0.8 if any(v == 33 for v in meta.get('program_changes', {}).values()) else 0.2
        f['root_5_bass']     = 1.0 - bass_change_ratio  # bajo estático = raíz/5ª

        # Bajo en tiempo 1 (vals): buscar notas de bajo cerca del inicio de compás
        if ts_num > 0:
            compas_dur = beat_dur * ts_num
            beat1_bass = sum(1 for n in bass_notes if (n.time_sec % compas_dur) < beat_dur * 0.3)
            f['bass_on_1']      = min(1.0, beat1_bass / max(len(bass_notes) * 0.3, 1))
            f['bass_on_beat']   = f['bass_on_1']
            f['bass_on_beat1']  = f['bass_on_1']
            # Bajo en tiempos 1 y 3 (bossa)
            beat3_bass = sum(1 for n in bass_notes
                            if compas_dur * 0 < (n.time_sec % compas_dur) < beat_dur * 0.3
                            or compas_dur * 0.4 < (n.time_sec % compas_dur) < compas_dur * 0.6 + beat_dur * 0.3)
            f['bass_on_1_3']    = min(1.0, beat3_bass / max(len(bass_notes) * 0.4, 1))
            f['bass_on_4beats'] = min(1.0, bass_change_ratio * 1.1)
            # Boogie bass: cambio en cada corchea
            f['bass_alternating']= 0.5 + bass_change_ratio * 0.5
        else:
            for k in ('bass_on_1','bass_on_beat','bass_on_beat1','bass_on_1_3','bass_on_4beats','bass_alternating'):
                f[k] = 0.5
    else:
        for k in ('walking_bass','bass_dominant','bass_syncopated','bass_electric_gm',
                  'root_5_bass','bass_on_1','bass_on_beat','bass_on_beat1',
                  'bass_on_1_3','bass_on_4beats','bass_alternating'):
            f[k] = 0.0

    # ── PATRÓN RÍTMICO ESPECÍFICO ─────────────────────────────

    # Habanera: ♩. ♪ ♩ ♩ — negra con puntillo, corchea, dos negras
    # Buscar duración ~1.5 beats seguida de ~0.5 beats en bajo o melodía
    beat_d = 60.0 / avg_bpm if avg_bpm > 0 else 0.5
    habanera_hits = 0
    for i in range(len(sn) - 2):
        d1 = sn[i].duration
        d2 = sn[i+1].duration
        if 1.2 * beat_d < d1 < 1.8 * beat_d and 0.3 * beat_d < d2 < 0.7 * beat_d:
            habanera_hits += 1
    f['habanera_rhythm'] = min(1.0, habanera_hits / max(total_dur / 2, 1))

    # Backbeat (acento en 2 y 4): notas con alta velocidad en esos tiempos
    if ts_num == 4 and avg_bpm > 0:
        compas_dur = beat_d * 4
        beat2_vel = [n.velocity for n in sn
                     if compas_dur * 0.22 < (n.time_sec % compas_dur) < compas_dur * 0.40]
        beat4_vel = [n.velocity for n in sn
                     if compas_dur * 0.72 < (n.time_sec % compas_dur) < compas_dur * 0.90]
        beat1_vel = [n.velocity for n in sn
                     if (n.time_sec % compas_dur) < compas_dur * 0.20]
        if beat1_vel and (beat2_vel or beat4_vel):
            avg_24 = (sum(beat2_vel + beat4_vel) / max(len(beat2_vel) + len(beat4_vel), 1))
            avg_1  = sum(beat1_vel) / len(beat1_vel)
            f['backbeat'] = min(1.0, max(0.0, (avg_24 - avg_1) / 30 + 0.5))
        else:
            f['backbeat'] = 0.3
        f['backbeat_light'] = f['backbeat'] * 0.6
    else:
        f['backbeat'] = 0.0
        f['backbeat_light'] = 0.0

    # Offbeat (reggae): acordes en contratiempos
    if ts_num == 4 and avg_bpm > 0:
        compas_dur = beat_d * 4
        offbeat_notes = [n for n in sn
                         if any(abs((n.time_sec % compas_dur) - b * beat_d) < beat_d * 0.15
                                for b in (0.5, 1.5, 2.5, 3.5))]
        f['skank_chord']    = min(1.0, len(offbeat_notes) / max(len(sn) * 0.3, 1))
        f['offbeat_accent'] = f['skank_chord']
    else:
        f['skank_chord'] = 0.0
        f['offbeat_accent'] = 0.0

    # March rhythm: notas cortas y regulares
    if sn:
        short_notes = sum(1 for n in sn if n.duration < beat_d * 0.6)
        f['march_rhythm'] = min(1.0, short_notes / max(len(sn) * 0.5, 1))
    else:
        f['march_rhythm'] = 0.0

    # Clave 3-2 o 2-3 (bossa/latin): patrón de 8 posiciones con 5 ataques
    # Simplificado: alta síncopa + bajo en 1 y 3 + ts=4
    f['clave_3_2']        = (f['syncopation_high'] * 0.5 + f['bass_on_1_3'] * 0.3 +
                             (0.2 if ts_num == 4 else 0.0))
    f['clave_2_3']        = f['clave_3_2']
    f['clave_3_2_or_2_3'] = f['clave_3_2']

    # Asymmetric accent (flamenco 12-beat)
    if ts_num in (6, 12):
        f['asymmetric_accent'] = 0.8
        f['ts_12_or_3'] = 1.0
    else:
        f['asymmetric_accent'] = f['syncopation_high'] * 0.4

    # ── ARMONÍA CARACTERÍSTICA ────────────────────────────────
    cad_counts = cadences.get('counts', collections.Counter())
    cad_total  = max(cadences.get('total', 1), 1)

    # Cadencia perfecta (clásica)
    f['cadence_perfect']   = min(1.0, cad_counts.get('perfecta', 0) / cad_total * 2.5)
    # Cadencia napolitana (tango)
    f['cadence_napl']      = min(1.0, cad_counts.get('napolitana', 0) / cad_total * 4.0)
    # Cadencia andaluza i-bVII-bVI-V
    # Detectar en progresiones: buscar descenso de raíces 0,10,8,7 (relativo)
    andalusian_hits = 0
    if len(chords) >= 4:
        for i in range(len(chords) - 3):
            roots = [(chords[i+j].root - key_root) % 12 for j in range(4)]
            if roots == [0, 10, 8, 7]:   # i-bVII-bVI-V desde la tónica
                andalusian_hits += 1
            elif roots[3] == 7 and roots[2] == 8:   # terminar en V desde bVI
                andalusian_hits += 0.5
    f['andalusian_cadence'] = min(1.0, andalusian_hits * 2.0)

    # ii-V-I (jazz)
    ii_V_I_hits = 0
    if len(chords) >= 3:
        for i in range(len(chords) - 2):
            r0 = (chords[i].root   - key_root) % 12
            r1 = (chords[i+1].root - key_root) % 12
            r2 = (chords[i+2].root - key_root) % 12
            if r0 == 2 and r1 == 7 and r2 == 0:
                ii_V_I_hits += 1
    f['ii_V_I'] = min(1.0, ii_V_I_hits / max(total_dur / 4, 1) * 3)

    # Acordes extendidos (jazz, bossa)
    if chords:
        ext = {'maj7','min7','dom7','dim7','hdim7','maj9','add9'}
        ext_ratio = sum(1 for c in chords if c.chord_type in ext) / len(chords)
        f['ext_chords']          = min(1.0, ext_ratio * 1.5)
        f['ext_chords_jazz']     = f['ext_chords']
        f['mode_major_minor_mix']= ext_ratio  # bossa usa muchas extensiones mixtas
        f['no_ext_chords']       = max(0.0, 1.0 - ext_ratio * 2)
        f['low_ext_chords']      = f['no_ext_chords']
        f['diatonic_harmony']    = f['no_ext_chords']
        f['simple_harmony']      = f['no_ext_chords']
        f['simple_cadences']     = f['no_ext_chords']
    else:
        for k in ('ext_chords','ext_chords_jazz','no_ext_chords','low_ext_chords',
                  'diatonic_harmony','simple_harmony','simple_cadences','mode_major_minor_mix'):
            f[k] = 0.5

    # Dom7 sobre tónica (blues)
    dom7_on_tonic = sum(1 for c in chords
                        if c.root == key_root and c.chord_type == 'dom7')
    f['dom7_on_tonic'] = min(1.0, dom7_on_tonic / max(len(chords) * 0.2, 1))
    f['I7_IV7_V7']     = f['dom7_on_tonic']

    # Dom7 como vamp estático (funk)
    if chords and total_dur > 0:
        dom7_notes = sum(1 for c in chords if c.chord_type == 'dom7')
        f['dom7_vamps'] = min(1.0, dom7_notes / max(len(chords) * 0.3, 1))
    else:
        f['dom7_vamps'] = 0.0

    # Mixolidio (rock)
    f['mixolydian_prog'] = (1.0 if mode == 'mixolydian' else
                            0.5 if mode == 'major' else 0.0)

    # I-IV-V simple (country/marcha)
    I_IV_V_hits = sum(1 for c in chords
                      if (c.root - key_root) % 12 in (0, 5, 7) and
                      c.chord_type in ('maj', 'min'))
    f['I_IV_V_simple'] = min(1.0, I_IV_V_hits / max(len(chords) * 0.5, 1))

    # Montuno (latin): vamp I-IV o I-V repetido
    f['montuno_harmony'] = f['I_IV_V_simple'] * 0.6 + f['syncopation_high'] * 0.4

    # Static harmony (jazz modal)
    if chords and total_dur > 0:
        unique_roots = len(set(c.root for c in chords))
        f['static_harmony'] = max(0.0, 1.0 - unique_roots / max(total_dur / 8, 3))
    else:
        f['static_harmony'] = 0.0

    # ── BLUE NOTES ────────────────────────────────────────────
    # b3, b5, b7 relativas a la tónica
    blue_pcs = set([(key_root + 3) % 12, (key_root + 6) % 12, (key_root + 10) % 12])
    blue_count = sum(1 for n in sn if n.pitch % 12 in blue_pcs)
    f['blue_notes']      = min(1.0, blue_count / max(len(sn) * 0.1, 1))
    f['chromatic_melody']= min(1.0, blue_count / max(len(sn) * 0.08, 1))
    # Cromatismo general
    scale = set(SCALE_INTERVALS.get(mode, SCALE_INTERVALS['major']))
    chromatic_notes = sum(1 for n in sn if (n.pitch % 12 - key_root) % 12 not in scale)
    f['chromatic_moves'] = min(1.0, chromatic_notes / max(len(sn) * 0.15, 1))
    f['chromatic_high']  = f['chromatic_moves']
    f['low_chromaticism']= max(0.0, 1.0 - f['chromatic_moves'])

    # ── MELODÍA ───────────────────────────────────────────────
    if len(sn) > 4:
        mel = get_mel_track(sn)
        ivs = [abs(mel[i+1].pitch - mel[i].pitch) for i in range(len(mel)-1)] if len(mel) > 1 else [0]
        stepwise = sum(1 for iv in ivs if iv <= 2) / max(len(ivs), 1)
        f['stepwise_melody'] = stepwise
        f['pentatonic'] = (1.0 if mode in ('major', 'minor') and stepwise > 0.4 and
                          f['chromatic_moves'] < 0.3 else 0.3)
    else:
        f['stepwise_melody'] = 0.5
        f['pentatonic'] = 0.3

    # ── ESTRUCTURA FORMAL ─────────────────────────────────────
    # Forma de 12 compases (blues): detectar si hay repetición de sección ~12 compases
    beat_dur2 = 60.0 / avg_bpm if avg_bpm > 0 else 0.5
    twelve_bar_dur = beat_dur2 * ts_num * 12
    if total_dur > twelve_bar_dur * 1.5:
        # Comprobar si hay estructura cada 12 compases
        f['blues_form_12'] = min(1.0, f['dom7_on_tonic'] * 0.6 + f['I_IV_V_simple'] * 0.4)
    else:
        f['blues_form_12'] = 0.0

    f['formal_structure'] = min(1.0, f['cadence_perfect'] * 0.6 + (1 - f['static_harmony']) * 0.4)
    f['motif_development'] = 0.5  # aproximación — depende del análisis de motivos

    # Bajo Alberti (clásica): oom-pah-pah
    if bass_notes and total_dur > 0:
        bass_pc = [n.pitch % 12 for n in bass_notes]
        repeated = sum(1 for i in range(len(bass_pc)-2)
                      if bass_pc[i] == bass_pc[i+2] and bass_pc[i] != bass_pc[i+1])
        f['alberti_bass'] = min(1.0, repeated / max(len(bass_pc) * 0.3, 1))
    else:
        f['alberti_bass'] = 0.0
    f['chord_on_2_3'] = f['alberti_bass'] * 0.5 + (0.5 if ts_num == 3 else 0.0)

    # Voice leading fluido
    f['voice_leading_smooth'] = 0.5  # se obtiene del análisis existente si disponible

    # ── INSTRUMENTOS GM ───────────────────────────────────────
    progs = set(meta.get('program_changes', {}).values())
    f['guitar_gm']         = 1.0 if any(p in range(24, 32) for p in progs) else 0.1
    f['banjo_guitar_gm']   = 1.0 if any(p in range(24, 32) for p in progs) else 0.1
    f['bass_electric_gm']  = 1.0 if any(p in range(32, 40) for p in progs) else 0.2
    f['brass_gm']          = 1.0 if any(p in range(56, 64) for p in progs) else 0.1
    f['brass_section_gm']  = f['brass_gm']
    f['brass_woodwind_gm'] = 1.0 if any(p in range(56, 72) for p in progs) else 0.1
    f['instruments_orch']  = 1.0 if any(p in range(40, 56) for p in progs) else 0.2
    f['percussion_dense']  = 1.0 if 9 in meta.get('program_changes', {}) else 0.1
    f['percussion_ch9']    = f['percussion_dense']

    # ── BAJA TENSIÓN / ALTO DINAMISMO ─────────────────────────
    if sn:
        vel = [n.velocity for n in sn]
        dyn_range = (max(vel) - min(vel)) / 127
        f['dynamic_range']   = dyn_range
        f['low_tension']     = max(0.0, 1.0 - sum(vel) / len(vel) / 127)
        f['groove_tight']    = 1.0 - dyn_range  # funk: dinamismo uniforme = groove
        f['counterpoint_rich'] = 0.5  # requiere análisis de contraunto existente
    else:
        f['dynamic_range'] = f['low_tension'] = f['groove_tight'] = f['counterpoint_rich'] = 0.5

    return f


def detect_genre(
    notes: List[NoteEvent],
    chords: List[ChordEvent],
    rhythm: Dict,
    ts_num: int,
    avg_bpm: float,
    mode: str,
    key_root: int,
    cadences: Dict,
    meta: Dict,
    total_dur: float,
) -> Dict:
    """
    Detecta el género musical de la pieza mediante scoring ponderado
    contra 14 perfiles de género.

    Algoritmo:
      1. Extraer vector de ~50 features del material analizado
      2. Para cada género, calcular la puntuación:
           score = Σ (feature_score × weight × match_quality)
      3. Normalizar scores al mejor posible de cada género
      4. El género con mayor score normalizado es el ganador
      5. Refinar con subgénero si la confianza es alta

    Devuelve:
      genre           : género principal detectado
      genre_label     : nombre legible del género
      confidence      : confianza en la clasificación [0..1]
      top3            : los 3 géneros más probables con scores
      subgenre        : subgénero refinado (si aplica)
      evidence        : features que más contribuyeron a la decisión
      negative_evidence: features que contradicen el género detectado
      description     : descripción musical del género
      key_signature   : firma rítmico-armónica del género
      ambiguous       : True si hay empate o confianza baja
      cultural_context: contexto cultural e histórico
      desc            : descripción narrativa del resultado
    """
    if not notes:
        return {'genre': 'indeterminado', 'genre_label': 'Género indeterminado',
                'confidence': 0.0, 'top3': [], 'subgenre': None,
                'evidence': [], 'negative_evidence': [], 'ambiguous': True,
                'desc': 'sin notas para clasificar género'}

    # Extraer features
    fv = _extract_genre_features(
        notes, chords, rhythm, ts_num, avg_bpm,
        mode, key_root, cadences, meta, total_dur
    )

    # Puntuar cada género
    raw_scores: Dict[str, float] = {}
    max_possible: Dict[str, float] = {}

    for gname, gdata in _GENRE_PROFILES.items():
        score = 0.0
        max_s = 0.0
        for feat, (expected, weight) in gdata['features'].items():
            actual = fv.get(feat, 0.0)
            # Match quality: similitud entre valor esperado y actual
            match = 1.0 - abs(expected - actual)
            match = max(0.0, match)
            score += match * weight
            max_s += weight
        raw_scores[gname]  = score
        max_possible[gname] = max_s

    # Normalizar al máximo teórico de cada género
    norm_scores = {g: raw_scores[g] / max(max_possible[g], 0.01)
                   for g in raw_scores}

    # Ordenar
    sorted_genres = sorted(norm_scores.items(), key=lambda x: -x[1])
    winner, winner_score = sorted_genres[0]
    second, second_score = sorted_genres[1] if len(sorted_genres) > 1 else ('?', 0)

    confidence = winner_score
    ambiguous  = (winner_score - second_score) < 0.06 or winner_score < 0.45

    # Top 3
    top3 = [{'genre': g, 'label': _GENRE_PROFILES[g]['label'], 'score': round(s, 3)}
            for g, s in sorted_genres[:3]]

    # Subgénero
    subgenre = None
    if winner in _SUBGENRE_RULES and confidence > 0.45:
        for sg_name, sg_rule in _SUBGENRE_RULES[winner]:
            try:
                if sg_rule(fv):
                    subgenre = sg_name
                    break
            except Exception:
                continue

    # Evidence: features que más contribuyeron al ganador
    evidence = []
    if winner in _GENRE_PROFILES:
        for feat, (expected, weight) in sorted(
                _GENRE_PROFILES[winner]['features'].items(),
                key=lambda x: -x[1][1]):
            actual = fv.get(feat, 0.0)
            match = 1.0 - abs(expected - actual)
            if match > 0.65 and weight > 0.06:
                evidence.append({
                    'feature': feat,
                    'score':   round(actual, 2),
                    'weight':  weight,
                    'match':   round(match, 2),
                })
        evidence = evidence[:5]

    # Negative evidence: features débiles del ganador
    neg_evidence = []
    if winner in _GENRE_PROFILES:
        for feat, (expected, weight) in _GENRE_PROFILES[winner]['features'].items():
            actual = fv.get(feat, 0.0)
            if abs(expected - actual) > 0.45 and weight > 0.08:
                neg_evidence.append(feat)

    # Contexto cultural
    gdata_w = _GENRE_PROFILES.get(winner, {})
    regions = ', '.join(gdata_w.get('regions', []))
    subgenre_list = ', '.join(gdata_w.get('subgenres', []))

    # Descripción
    conf_label = ('alta' if confidence > 0.72 else
                  'moderada' if confidence > 0.52 else
                  'baja — múltiples géneros compatibles')

    desc_parts = [
        f"Género detectado: {gdata_w.get('label','?')} "
        f"(confianza {conf_label}: {confidence:.0%}).",
    ]
    if subgenre:
        desc_parts.append(f"Subgénero más probable: {subgenre}.")
    if ambiguous and len(sorted_genres) > 1:
        desc_parts.append(
            f"Clasificación ambigua: también compatible con "
            f"{_GENRE_PROFILES.get(second,{}).get('label', second)} "
            f"({second_score:.0%})."
        )
    desc_parts.append(gdata_w.get('description',''))
    if evidence:
        ev_names = [e['feature'].replace('_', ' ') for e in evidence[:3]]
        desc_parts.append(f"Evidencia principal: {', '.join(ev_names)}.")
    if neg_evidence:
        neg_names = [n.replace('_', ' ') for n in neg_evidence[:2]]
        desc_parts.append(f"Rasgos atípicos para este género: {', '.join(neg_names)}.")

    return {
        'genre':            winner,
        'genre_label':      gdata_w.get('label', winner),
        'confidence':       confidence,
        'confidence_label': conf_label,
        'top3':             top3,
        'subgenre':         subgenre,
        'subgenres_list':   subgenre_list,
        'evidence':         evidence,
        'negative_evidence':neg_evidence,
        'description':      gdata_w.get('description',''),
        'key_signature':    gdata_w.get('key_signature',''),
        'regions':          regions,
        'ambiguous':        ambiguous,
        'feature_vector':   fv,
        'desc':             ' '.join(desc_parts),
    }


# ═══════════════════════════════════════════════════════════════
#  v12. MAPA EMOCIONAL UNIFICADO — EVOLUCIÓN TEMPORAL COMPLETA
# ═══════════════════════════════════════════════════════════════
#
#  Consolida en un único eje temporal las 5 dimensiones emocionales
#  continuas que el sistema calcula por separado:
#    1. Tensión armónica          (compute_tension_curve)
#    2. Valencia emocional        (compute_dynamic_valence)
#    3. Energía física            (compute_energy_profile)
#    4. Aspereza sensorial        (compute_roughness_curve)
#    5. Arousal (activación)      (derivado de velocidad + densidad)
#
#  Para cada slot temporal calcula además:
#    - Velocidad de cambio emocional (Δ por segundo)
#    - Estado emocional combinado (label de 2 palabras)
#    - Coherencia entre dimensiones (¿apuntan todas en la misma dirección?)
#
#  El resultado permite responder:
#    "¿Qué emoción exacta sentía el oyente en el minuto X:XX?"
#    "¿Dónde están los giros emocionales más bruscos?"
#    "¿Qué secciones son estables vs. volátiles?"

_EMOTIONAL_STATE_LABELS = {
    # (tension_high, valence_pos, energy_high) → label
    (True,  True,  True):  ('Euforia',        'tensión luminosa, alta energía'),
    (True,  True,  False): ('Éxtasis quieto',  'tensión brillante, recogida'),
    (True,  False, True):  ('Angustia activa', 'tensión oscura, agitada'),
    (True,  False, False): ('Terror helado',   'tensión oscura, quietud'),
    (False, True,  True):  ('Alegría plena',   'apertura luminosa, energía'),
    (False, True,  False): ('Serenidad',       'reposo luminoso, calma'),
    (False, False, True):  ('Melancolía viva', 'oscuridad activa, movimiento'),
    (False, False, False): ('Vacío sereno',    'reposo oscuro, quietud'),
}


def compute_unified_emotional_map(
    tension_curve:    List,
    dynamic_valence:  Dict,
    energy_profile:   Dict,
    roughness:        Dict,
    notes:            List,
    total_dur:        float,
    resolution:       float = 2.0,
) -> Dict:
    """
    Construye el mapa emocional unificado: una tabla temporal donde cada
    fila es un slot de `resolution` segundos y las columnas son las 5
    dimensiones emocionales sincronizadas.

    Algoritmo:
      1. Re-muestrea cada curva a la misma resolución temporal
      2. Normaliza todas las dimensiones a [0..1]
      3. Calcula Δ (velocidad de cambio) entre slots consecutivos
      4. Asigna label emocional combinado por umbral de cada dimensión
      5. Detecta puntos de inflexión (giros emocionales)
      6. Calcula coherencia: std de las 5 dimensiones (baja = coherente)
      7. Genera representación ASCII de la matriz completa

    Devuelve:
      slots           : lista de dicts con todas las métricas por slot
      inflection_points: momentos de cambio emocional brusco
      volatility_curve : velocidad de cambio emocional por slot
      mean_coherence  : coherencia media entre dimensiones [0..1]
      emotional_phases : fases continuas con el mismo estado emocional
      peak_complexity : momento de máxima complejidad emocional
      ascii_matrix    : visualización ASCII de la matriz completa
      desc            : descripción narrativa
    """
    if not tension_curve or total_dur <= 0:
        return {
            'slots': [], 'inflection_points': [], 'volatility_curve': [],
            'mean_coherence': 0.0, 'emotional_phases': [],
            'peak_complexity': 0.0, 'ascii_matrix': '', 'desc': 'sin datos'
        }

    # ── 1. Construir mapas de lookup por tiempo ───────────────
    # Tensión
    t_map = {p.time: p.tension for p in tension_curve}
    t_times = sorted(t_map.keys())

    def get_tension(t: float) -> float:
        if not t_times: return 0.3
        return t_map[min(t_times, key=lambda x: abs(x - t))]

    # Valencia
    val_curve = dynamic_valence.get('curve', [])
    val_dict  = {tv: vv for tv, vv in val_curve}
    val_times = sorted(val_dict.keys())

    def get_valence(t: float) -> float:
        if not val_times: return 0.0
        return val_dict[min(val_times, key=lambda x: abs(x - t))]

    # Energía
    e_curve = energy_profile.get('curve', [])
    e_dict  = {te: ve for te, ve in e_curve}
    e_times = sorted(e_dict.keys())
    e_max   = max(e_dict.values()) if e_dict else 1.0

    def get_energy(t: float) -> float:
        if not e_times: return 0.0
        raw = e_dict[min(e_times, key=lambda x: abs(x - t))]
        return raw / e_max if e_max > 0 else 0.0

    # Roughness
    r_curve = roughness.get('curve_norm', [])
    r_dict  = {tr: vr for tr, vr in r_curve}
    r_times = sorted(r_dict.keys())

    def get_roughness(t: float) -> float:
        if not r_times: return 0.0
        return r_dict[min(r_times, key=lambda x: abs(x - t))]

    # Arousal: derivado de velocidad MIDI + densidad
    sn = sorted(notes, key=lambda n: n.time_sec)

    def get_arousal(t: float) -> float:
        window = [n for n in sn
                  if n.time_sec < t + resolution and n.time_sec + n.duration > t]
        if not window: return 0.0
        avg_vel   = sum(n.velocity for n in window) / len(window) / 127.0
        density   = len(window) / resolution / 10.0   # notas/seg normalizado
        return min(1.0, avg_vel * 0.6 + density * 0.4)

    # ── 2. Muestrear todas las dimensiones en slots regulares ──
    slots = []
    t = 0.0
    while t < total_dur:
        tension  = get_tension(t)
        # Valencia: convertir de [-1,1] a [0,1]
        valence_raw = get_valence(t)
        valence  = (valence_raw + 1.0) / 2.0
        energy   = get_energy(t)
        rgh      = get_roughness(t)
        arousal  = get_arousal(t)

        # Coherencia del slot: std de las 5 dimensiones
        vals5   = [tension, valence, energy, rgh, arousal]
        mean5   = sum(vals5) / 5.0
        std5    = math.sqrt(sum((v - mean5) ** 2 for v in vals5) / 5.0)
        coherence = 1.0 - std5 * 2.0  # alta coherencia = std bajo

        # Label emocional combinado
        key_label = (tension > 0.50, valence > 0.55, energy > 0.45)
        label, label_desc = _EMOTIONAL_STATE_LABELS.get(
            key_label, ('Ambiguo', 'estado mixto sin etiqueta clara'))

        slots.append({
            'time':      t,
            'time_str':  ts_str_local(t),
            'tension':   round(tension, 3),
            'valence':   round(valence_raw, 3),   # mantener [-1,1] para legibilidad
            'valence_n': round(valence, 3),        # [0,1] para cálculos
            'energy':    round(energy, 3),
            'roughness': round(rgh, 3),
            'arousal':   round(arousal, 3),
            'coherence': round(max(0.0, coherence), 3),
            'label':     label,
            'label_desc':label_desc,
        })
        t += resolution

    if not slots:
        return {
            'slots': [], 'inflection_points': [], 'volatility_curve': [],
            'mean_coherence': 0.0, 'emotional_phases': [],
            'peak_complexity': 0.0, 'ascii_matrix': '', 'desc': 'sin slots'
        }

    # ── 3. Velocidad de cambio emocional (Δ total por slot) ───
    volatility_curve = [{'time': slots[0]['time'], 'delta': 0.0}]
    dims = ('tension', 'valence_n', 'energy', 'roughness', 'arousal')

    for i in range(1, len(slots)):
        prev, curr = slots[i-1], slots[i]
        delta = math.sqrt(
            sum((curr[d] - prev[d]) ** 2 for d in dims) / len(dims)
        )
        volatility_curve.append({'time': curr['time'], 'delta': round(delta, 4)})
        slots[i]['delta'] = round(delta, 4)
    slots[0]['delta'] = 0.0

    # ── 4. Puntos de inflexión (giros bruscos) ────────────────
    if len(volatility_curve) > 2:
        deltas     = [v['delta'] for v in volatility_curve]
        mean_delta = sum(deltas) / len(deltas)
        std_delta  = math.sqrt(sum((d - mean_delta)**2 for d in deltas) / len(deltas))
        threshold  = mean_delta + 1.5 * std_delta

        inflection_points = []
        for i in range(1, len(slots) - 1):
            if slots[i]['delta'] > threshold:
                # Describir la naturaleza del giro
                prev_s, curr_s = slots[i-1], slots[i]
                changes = []
                for dim, name in [('tension','tensión'), ('valence_n','valencia'),
                                   ('energy','energía'), ('roughness','aspereza')]:
                    diff = curr_s[dim] - prev_s[dim]
                    if abs(diff) > 0.20:
                        direction = 'sube' if diff > 0 else 'baja'
                        changes.append(f"{name} {direction} {abs(diff):.2f}")

                inflection_points.append({
                    'time':    slots[i]['time'],
                    'time_str':slots[i]['time_str'],
                    'delta':   slots[i]['delta'],
                    'from_label': prev_s['label'],
                    'to_label':   curr_s['label'],
                    'changes': changes,
                    'desc': f"{prev_s['label']} → {curr_s['label']}"
                             + (f": {', '.join(changes[:2])}" if changes else ''),
                })
    else:
        inflection_points = []

    # ── 5. Fases emocionales continuas ────────────────────────
    emotional_phases = []
    if slots:
        cur_label  = slots[0]['label']
        phase_start= slots[0]['time']
        phase_vals = {d: [slots[0][d]] for d in dims + ('coherence',)}

        for s in slots[1:]:
            if s['label'] != cur_label:
                dur = s['time'] - phase_start
                means = {d: sum(phase_vals[d])/len(phase_vals[d]) for d in dims + ('coherence',)}
                emotional_phases.append({
                    'label':    cur_label,
                    'start':    phase_start,
                    'end':      s['time'],
                    'duration': dur,
                    'start_str':ts_str_local(phase_start),
                    'end_str':  ts_str_local(s['time']),
                    'means':    means,
                })
                cur_label   = s['label']
                phase_start = s['time']
                phase_vals  = {d: [s[d]] for d in dims + ('coherence',)}
            else:
                for d in dims + ('coherence',):
                    phase_vals[d].append(s[d])

        # Última fase
        dur = total_dur - phase_start
        means = {d: sum(phase_vals[d])/len(phase_vals[d]) for d in dims + ('coherence',)}
        emotional_phases.append({
            'label':    cur_label,
            'start':    phase_start,
            'end':      total_dur,
            'duration': dur,
            'start_str':ts_str_local(phase_start),
            'end_str':  ts_str_local(total_dur),
            'means':    means,
        })

    # ── 6. Momento de máxima complejidad emocional ────────────
    # Máxima simultaniedad de dimensiones extremas
    complexity_scores = []
    for s in slots:
        # Cuántas dimensiones están lejos del centro (>0.3 de 0.5)
        extremes = sum(1 for d in dims if abs(s[d] - 0.5) > 0.3)
        complexity_scores.append((s['time'], extremes))
    if complexity_scores:
        peak_complexity_t = max(complexity_scores, key=lambda x: x[1])[0]
    else:
        peak_complexity_t = 0.0

    # ── 7. Estadísticas globales ──────────────────────────────
    mean_coherence = sum(s['coherence'] for s in slots) / len(slots)

    # ── 8. Visualización ASCII de la matriz ───────────────────
    ascii_matrix = _build_emotional_matrix_ascii(slots, total_dur)

    # ── 9. Descripción narrativa ──────────────────────────────
    n_phases  = len(emotional_phases)
    n_inflex  = len(inflection_points)
    dom_phase = max(emotional_phases, key=lambda p: p['duration']) if emotional_phases else None
    dom_label = dom_phase['label'] if dom_phase else '?'
    dom_pct   = dom_phase['duration'] / total_dur * 100 if dom_phase else 0

    coh_desc = ('alta coherencia emocional — las 5 dimensiones se mueven juntas'
                if mean_coherence > 0.65 else
                'coherencia moderada — algunas dimensiones divergen con frecuencia'
                if mean_coherence > 0.40 else
                'baja coherencia — las dimensiones emocionales operan de forma independiente')

    desc = (
        f"La pieza atraviesa {n_phases} fase(s) emocional(es) diferenciada(s). "
        f"Estado dominante: '{dom_label}' ({dom_pct:.0f}% del tiempo). "
        f"{n_inflex} giro(s) emocional(es) brusco(s) detectado(s). "
        f"Coherencia media entre dimensiones: {mean_coherence:.2f} — {coh_desc}."
    )

    return {
        'slots':             slots,
        'inflection_points': inflection_points,
        'volatility_curve':  volatility_curve,
        'mean_coherence':    round(mean_coherence, 3),
        'emotional_phases':  emotional_phases,
        'peak_complexity':   peak_complexity_t,
        'ascii_matrix':      ascii_matrix,
        'desc':              desc,
    }


def _build_emotional_matrix_ascii(slots: List[Dict], total_dur: float,
                                   max_cols: int = 60) -> str:
    """
    Genera la matriz ASCII del mapa emocional unificado.

    Formato:
      Filas  = 5 dimensiones emocionales + volatilidad
      Columnas = tiempo (muestreado a max_cols puntos)

    Escala de densidad para cada dimensión:
      · < 0.20   ░ 0.20-0.40   ▒ 0.40-0.60   ▓ 0.60-0.80   █ > 0.80
    """
    if not slots:
        return ''

    # Downsample a max_cols columnas
    step = max(1, len(slots) // max_cols)
    sampled = [slots[i] for i in range(0, len(slots), step)][:max_cols]
    n_cols  = len(sampled)

    def glyph(v: float, flip: bool = False) -> str:
        if flip: v = 1.0 - v
        if v >= 0.80: return '█'
        if v >= 0.60: return '▓'
        if v >= 0.40: return '▒'
        if v >= 0.20: return '░'
        return '·'

    # Normalizar valencia de [-1,1] a [0,1]
    rows = {
        'Tensión  ': [glyph(s['tension'])          for s in sampled],
        'Valencia ': [glyph(s['valence_n'])         for s in sampled],
        'Energía  ': [glyph(s['energy'])            for s in sampled],
        'Aspereza ': [glyph(s['roughness'])         for s in sampled],
        'Arousal  ': [glyph(s['arousal'])           for s in sampled],
        'Cambio   ': [glyph(min(1.0, s.get('delta', 0) * 8)) for s in sampled],
    }

    # Ejes de tiempo
    t_start = ts_str_local(0)
    t_mid   = ts_str_local(total_dur / 2)
    t_end   = ts_str_local(total_dur)
    pad_mid = max(0, n_cols // 2 - len(t_start) - len(t_mid) // 2)
    pad_end = max(0, n_cols - len(t_start) - len(t_mid) - len(t_end) - pad_mid)

    lines = []
    lines.append("  Mapa emocional unificado  (· bajo  ░ medio-bajo  ▒ medio  ▓ alto  █ máx)")
    lines.append("")
    lines.append(f"          {t_start}{' ' * pad_mid}{t_mid}{' ' * pad_end}{t_end}")
    lines.append(f"          ┌{'─' * n_cols}┐")

    for dim_name, row_glyphs in rows.items():
        line = ''.join(row_glyphs)
        lines.append(f"  {dim_name}│{line}│")

    lines.append(f"          └{'─' * n_cols}┘")
    lines.append("  Leyenda: · <0.2  ░ 0.2-0.4  ▒ 0.4-0.6  ▓ 0.6-0.8  █ >0.8")

    # Marcar puntos de inflexión en el eje
    return '\n'.join(lines)


def _format_unified_map_report(uem: Dict, total_dur: float) -> str:
    """Formatea la sección §83 del informe."""
    if not uem or not uem.get('slots'):
        return ''

    lines = []
    SEP = '─' * 66
    lines.append(SEP)
    lines.append("  MAPA EMOCIONAL UNIFICADO — EVOLUCIÓN TEMPORAL")
    lines.append(SEP)
    lines.append("")

    # Matriz ASCII
    if uem.get('ascii_matrix'):
        for line in uem['ascii_matrix'].split('\n'):
            lines.append('  ' + line)
        lines.append("")

    # Descripción global
    lines.append(f"  {uem.get('desc','')}")
    lines.append(f"  Coherencia media : {uem.get('mean_coherence',0):.2f}/1.0")
    lines.append(f"  Pico complejidad : {ts_str_local(uem.get('peak_complexity',0))}")
    lines.append("")

    # Fases emocionales
    phases = uem.get('emotional_phases', [])
    if phases:
        lines.append(f"  Fases emocionales ({len(phases)}):")
        for ph in phases:
            dur_pct = ph['duration'] / total_dur * 100 if total_dur > 0 else 0
            bar_w   = int(dur_pct / 2)
            bar     = '█' * bar_w + '░' * (50 // 2 - bar_w)
            means   = ph.get('means', {})
            t_str   = f"{ph['start_str']} → {ph['end_str']}"
            lines.append(
                f"    [{bar}] {dur_pct:4.0f}%  "
                f"{ph['label']:<18}  {t_str}"
            )
            if means:
                lines.append(
                    f"         T={means.get('tension',0):.2f} "
                    f"V={means.get('valence_n',0):.2f} "
                    f"E={means.get('energy',0):.2f} "
                    f"R={means.get('roughness',0):.2f} "
                    f"A={means.get('arousal',0):.2f}"
                )
        lines.append("")

    # Puntos de inflexión
    inflex = uem.get('inflection_points', [])
    if inflex:
        lines.append(f"  Giros emocionales bruscos ({len(inflex)}):")
        for ip in inflex[:8]:
            lines.append(
                f"    ⚡ {ip['time_str']}  Δ={ip['delta']:.3f}  "
                f"{ip['desc']}"
            )
        lines.append("")

    # Tabla temporal detallada (cada 10 slots ~ cada 20 segundos)
    slots = uem.get('slots', [])
    step  = max(1, len(slots) // 20)
    sampled_slots = [slots[i] for i in range(0, len(slots), step)]

    if sampled_slots:
        lines.append("  Tabla temporal (muestra cada ~"
                     f"{step * 2:.0f}s):")
        lines.append(
            f"  {'Tiempo':<8} {'Tensión':>8} {'Valencia':>9} "
            f"{'Energía':>8} {'Aspereza':>9} {'Arousal':>8} "
            f"{'Δ/s':>6}  Estado"
        )
        lines.append("  " + "─" * 84)
        for s in sampled_slots:
            val_sign = '+' if s['valence'] >= 0 else ''
            lines.append(
                f"  {s['time_str']:<8} "
                f"{s['tension']:>8.2f} "
                f"{val_sign}{s['valence']:>8.2f} "
                f"{s['energy']:>8.2f} "
                f"{s['roughness']:>9.2f} "
                f"{s['arousal']:>8.2f} "
                f"{s.get('delta',0):>6.3f}  "
                f"{s['label']}"
            )
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  TEST
# ═══════════════════════════════════════════════════════════════



# ══ ANALYSES v6 ══
def dynamic_segmentation(notes,tension_curve,energy_curve,bpm,total_dur):
    if not notes or total_dur<=0: return []
    beat=60.0/bpm if bpm>0 else 0.5
    sn=sorted(notes,key=lambda n:n.time_sec)
    res=0.5
    t_map={round(p.time/res):p.tension for p in tension_curve} if tension_curve else {}
    e_map={round(t_/res):e for t_,e in energy_curve} if energy_curve else {}
    max_e=max(e_map.values(),default=1)
    steps=int(total_dur/res)+1
    features=[]
    for step in range(steps):
        t=step*res
        window=[n for n in sn if t<=n.time_sec<t+res*2]
        density=len(window)/(res*2) if window else 0
        avg_p=sum(n.pitch for n in window)/len(window) if window else 60
        avg_v=sum(n.velocity for n in window)/len(window) if window else 0
        tension=t_map.get(step,0)
        energy=e_map.get(step,0)/max_e if max_e>0 else 0
        features.append((t,density,avg_p,avg_v,tension,energy))
    if len(features)<4: return [{'start':0,'end':total_dur,'index':0,'notes':sn,'note_count':len(sn),'density':len(sn)/total_dur,'avg_velocity':sum(n.velocity for n in sn)/len(sn) if sn else 0,'avg_pitch':sum(n.pitch for n in sn)/len(sn) if sn else 60,'duration':total_dur}]
    boundaries=[0.0]
    win=max(2,int(beat*2/res))
    def avg_f(fs,idx): return sum(f[idx] for f in fs)/len(fs) if fs else 0
    for i in range(win,len(features)-win):
        t,d,p,v,tn,en=features[i]
        before=features[max(0,i-win):i]; after=features[i:min(len(features),i+win)]
        d_chg=abs(avg_f(after,1)-avg_f(before,1))/(max(avg_f(before,1),avg_f(after,1),0.1))
        p_chg=abs(avg_f(after,2)-avg_f(before,2))/12
        v_chg=abs(avg_f(after,3)-avg_f(before,3))/127
        t_win=[features[j][4] for j in range(max(0,i-3),min(len(features),i+3))]
        is_valley=(tn<min(t_win[:3]+[1]) and tn<min(t_win[-3:]+[1]) and tn<0.25) if len(t_win)>=4 else False
        score=d_chg*0.35+p_chg*0.25+v_chg*0.20
        if is_valley: score+=0.15
        if avg_f(features[max(0,i-2):i+1],3)<5: score+=0.20
        if score>0.35 and t-boundaries[-1]>beat*4:
            boundaries.append(t)
    boundaries.append(total_dur)
    min_len=beat*4
    merged=[boundaries[0]]
    for b in boundaries[1:]:
        if b-merged[-1]>=min_len: merged.append(b)
    if merged[-1]<total_dur: merged.append(total_dur)
    segments=[]
    for i in range(len(merged)-1):
        t0,t1=merged[i],merged[i+1]
        sn2=[n for n in sn if t0<=n.time_sec<t1]
        segments.append({'index':i,'start':t0,'end':t1,'duration':t1-t0,'notes':sn2,'note_count':len(sn2),'density':len(sn2)/(t1-t0) if t1>t0 else 0,'avg_velocity':sum(n.velocity for n in sn2)/len(sn2) if sn2 else 0,'avg_pitch':sum(n.pitch for n in sn2)/len(sn2) if sn2 else 60})
    return segments

# ── 2. GRAVEDAD TONAL DINÁMICA ────────────────────────────────

HARMONIC_COLORS = {
    0:('C','blanco puro','luz, pureza, reposo absoluto'),
    1:('C#','violeta oscuro','misterio, lo oculto, profundidad'),
    2:('D','amarillo dorado','calidez, alegría, afirmación solar'),
    3:('D#','azul grisáceo','melancolía, lo indeterminado, niebla'),
    4:('E','azul brillante','serenidad, claridad, vastedad'),
    5:('F','rojo oscuro','pasión, tierra, fuerza primaria'),
    6:('F#','azul intenso','éxtasis, lo sublime, vértigo'),
    7:('G','naranja dorado','heroísmo, calor activo, movimiento'),
    8:('G#','violeta púrpura','majestad, lo sagrado, tensión sublime'),
    9:('A','verde esmeralda','naturaleza, esperanza, equilibrio vivo'),
    10:('A#','gris acerado','dureza, fatalismo, peso'),
    11:('B','azul pálido','distancia, lo inalcanzable, anhelo'),
}

def _cof_dist(r1, r2):
    """Distance in circle of fifths between two roots."""
    p1 = (r1 * 7) % 12; p2 = (r2 * 7) % 12
    return min(abs(p1 - p2), 12 - abs(p1 - p2))

def dynamic_tonal_gravity(notes,total_dur,window_sec=5.0,hop_sec=1.0):
    if not notes or total_dur<=0: return {'map':[],'modulations':[],'tonal_journey':'','n_modulations':0,'dominant_key':'','dominant_pct':0,'total_distance':0,'desc':''}
    sn=sorted(notes,key=lambda n:n.time_sec)
    tonal_map=[]; t=0.0
    while t<total_dur-hop_sec:
        window=[n for n in sn if t<=n.time_sec<t+window_sec]
        if len(window)<3: t+=hop_sec; continue
        pc=[0.0]*12
        for n in window: pc[n.pitch%12]+=(n.velocity/127.0)*n.duration
        tot=sum(pc)
        if tot==0: t+=hop_sec; continue
        pc=[v/tot for v in pc]
        best_r,best_m,best_c=0,'major',-999
        for mn,prof in KS_MODES.items():
            for rot in range(12):
                r=pearson(pc,prof[rot:]+prof[:rot])
                if r>best_c: best_c,best_r,best_m=r,rot,mn
        tonal_map.append({'time':t,'root':best_r,'mode':best_m,'confidence':best_c,'key_name':f"{NOTE_NAMES[best_r]} {best_m}"})
        t+=hop_sec
    if not tonal_map: return {'map':[],'modulations':[],'tonal_journey':'','n_modulations':0,'dominant_key':'','dominant_pct':0,'total_distance':0,'desc':'tonalidad no detectada'}
    modulations=[]
    for i in range(1,len(tonal_map)):
        prev,curr=tonal_map[i-1],tonal_map[i]
        if curr['root']!=prev['root'] or curr['mode']!=prev['mode']:
            if i+2<len(tonal_map) and tonal_map[i+1]['root']==curr['root'] and tonal_map[i+1]['mode']==curr['mode']:
                modulations.append({'time':curr['time'],'from':f"{NOTE_NAMES[prev['root']]} {prev['mode']}",'to':f"{NOTE_NAMES[curr['root']]} {curr['mode']}",'distance':_cof_dist(prev['root'],curr['root'])})
    seen=[]
    for e in tonal_map:
        k=(e['root'],e['mode'])
        if not seen or seen[-1]!=k: seen.append(k)
    journey=' → '.join(f"{NOTE_NAMES[r]} {m}" for r,m in seen[:8])
    key_times=collections.Counter()
    for e in tonal_map: key_times[e['key_name']]+=hop_sec
    tot_dist=sum(m['distance'] for m in modulations)
    n=len(modulations)
    if n==0: desc='tonalidad estable — pieza anclada en un único centro tonal'
    elif n<=2: desc=f'{n} modulación(es) — desvíos colorísticos puntuales'
    elif n<=5: desc=f'{n} modulaciones — itinerario tonal moderado'
    else: desc=f'{n} modulaciones — armonía tonalmente nómada'
    dom=key_times.most_common(1)[0] if key_times else ('',0)
    return {'map':tonal_map,'modulations':modulations,'tonal_journey':journey,'n_modulations':n,'dominant_key':dom[0],'dominant_pct':dom[1]/total_dur*100,'total_distance':tot_dist,'desc':desc}

# ── 3. SUBTEXT EMOCIONAL ──────────────────────────────────────
def analyze_emotional_subtext(sections,tension_curve,mode,avg_bpm):
    if not sections: return {'contradictions':[],'subtext_index':0,'desc':''}
    events=[]
    for sec in sections:
        if not sec.notes: continue
        surface_bright=sec.avg_pitch>69; surface_loud=sec.avg_velocity>72
        t_vals=[p.tension for p in tension_curve if sec.start<=p.time<sec.end]
        avg_t=sum(t_vals)/len(t_vals) if t_vals else 0.2
        mode_dark=mode in('minor','phrygian','locrian','dorian')
        deep_dark=mode_dark or avg_t>0.4
        if surface_bright and deep_dark:
            events.append({'time':sec.start,'type':'luminoso-sobre-oscuro','surface':'melodía aguda y brillante','depth':f"modo {mode}{'+ tensión' if avg_t>0.4 else ''}","emotion":'alegría que oculta tristeza — máscara emocional, dualidad dolorosa'})
        elif not surface_loud and avg_t>0.45:
            events.append({'time':sec.start,'type':'suave-con-tension','surface':'dinámica contenida','depth':'alta tensión armónica','emotion':'angustia susurrada — lo más perturbador es lo más callado'})
        elif surface_loud and avg_t<0.15:
            events.append({'time':sec.start,'type':'fuerte-sin-tension','surface':'alta dinámica','depth':'armonía consonante','emotion':'afirmación pura, celebración sin sombra'})
    active=sum(1 for s in sections if s.notes)
    idx=len(events)/active if active>0 else 0
    if idx>0.6: desc=f"alto subtext ({idx*100:.0f}%) — brecha sistemática entre superficie y profundidad"
    elif idx>0.3: desc=f"subtext moderado ({idx*100:.0f}%)"
    elif idx>0: desc=f"subtext puntual ({idx*100:.0f}%)"
    else: desc="coherencia expresiva total — superficie y profundidad alineadas"
    return {'contradictions':events,'subtext_index':idx,'desc':desc}

# ── 4. MOMENTUM MUSICAL ───────────────────────────────────────
def compute_momentum(tension_curve,energy_curve,resolution=0.5):
    if not tension_curve or not energy_curve: return {'curve':[],'peak_momentum':0,'peak_time':0,'trough_time':0,'reversals':0,'shape':'','desc':''}
    t_vals=[p.tension for p in tension_curve]
    e_vals=[e for _,e in energy_curve]
    max_e=max(e_vals) if e_vals else 1
    e_norm=[e/max_e for e in e_vals]
    total=max((tension_curve[-1].time if tension_curve else 0),(energy_curve[-1][0] if energy_curve else 0))
    steps=int(total/resolution)+1
    combined=[]
    for step in range(steps):
        t=step*resolution
        ti=min(int(t/resolution),len(t_vals)-1)
        ei=min(int(t*len(e_norm)/max(total,1)),len(e_norm)-1)
        combined.append((t,(t_vals[ti]+e_norm[ei])/2))
    mom_curve=[]
    sw=max(2,int(2.0/resolution))
    for i in range(len(combined)):
        t=combined[i][0]
        i2=min(i+sw,len(combined)-1)
        dt=combined[i2][0]-combined[i][0]
        mom=(combined[i2][1]-combined[i][1])/dt if dt>0 else 0
        mom_curve.append((t,mom))
    if not mom_curve: return {'curve':[],'peak_momentum':0,'peak_time':0,'trough_time':0,'reversals':0,'shape':'','desc':''}
    mv=[m for _,m in mom_curve]
    pi=max(range(len(mv)),key=lambda i:mv[i]); ti2=min(range(len(mv)),key=lambda i:mv[i])
    reversals=sum(1 for i in range(1,len(mv)-1) if (mv[i]>0)!=(mv[i-1]>0))
    pos_d=sum(resolution for _,m in mom_curve if m>0.02)
    neg_d=sum(resolution for _,m in mom_curve if m<-0.02)
    if pos_d>neg_d*2: shape='predominantemente positivo — avanza constantemente'
    elif neg_d>pos_d*2: shape='predominantemente negativo — se disuelve o retrocede'
    elif reversals>8: shape=f'oscilante ({reversals} reversiones) — vaivén emocional'
    else: shape='equilibrado'
    desc=f"Momentum {shape}. Pico en {ts_str(mom_curve[pi][0])}, mínimo en {ts_str(mom_curve[ti2][0])}."
    return {'curve':mom_curve,'peak_momentum':mv[pi],'peak_time':mom_curve[pi][0],'trough_momentum':mv[ti2],'trough_time':mom_curve[ti2][0],'reversals':reversals,'pos_duration':pos_d,'neg_duration':neg_d,'shape':shape,'desc':desc}

def momentum_ascii(curve,width=56,height=5):
    if not curve: return ''
    vals=[m for _,m in curve]
    max_v=max(abs(v) for v in vals) if vals else 1
    if max_v==0: return ''
    step=max(1,len(vals)//width)
    sampled=[sum(vals[i:i+step])/max(len(vals[i:i+step]),1) for i in range(0,len(vals),step)][:width]
    lines=[]
    for row in range(height,-height-1,-1):
        threshold=row/height*max_v
        line=''.join('▲' if row>0 and v>=threshold else '▼' if row<0 and v<=threshold else '─' if row==0 else ' ' for v in sampled)
        lbl=f"{threshold:+.2f}" if row in(height,0,-height) else '      '
        lines.append(f"  {lbl} │{line}│")
    lines.append(f"         └{'─'*len(sampled)}┘")
    total=curve[-1][0]
    lines.append(f"         0{' '*(len(sampled)//2-2)}{ts_str(total/2)}{' '*(len(sampled)//2-3)}{ts_str(total)}")
    return '\n'.join(lines)

# ── 5. FATIGA AUDITIVA ────────────────────────────────────────
def compute_auditory_fatigue(notes,tension_curve,total_dur,resolution=0.5):
    if not notes: return {'fatigue_index':0,'contrast_events':[],'physical_curve':[],'perceived_curve':[],'desc':''}
    sn=sorted(notes,key=lambda n:n.time_sec)
    phys_c=[]; perc_c=[]; adapt=0.5; tau=8.0
    t=0.0
    while t<total_dur:
        active=[n for n in sn if n.time_sec<=t<n.time_sec+n.duration]
        phys=sum(n.velocity/127.0 for n in active)/max(len(active),1) if active else 0
        alpha=1-math.exp(-resolution/tau)
        adapt=alpha*phys+(1-alpha)*adapt
        perceived=max(0,phys-adapt*0.5)
        phys_c.append((t,phys)); perc_c.append((t,perceived))
        t+=resolution
    if not phys_c: return {'fatigue_index':0,'contrast_events':[],'physical_curve':[],'perceived_curve':[],'desc':''}
    pv=[v for _,v in phys_c]; perv=[v for _,v in perc_c]
    mp=sum(pv)/len(pv); mpe=sum(perv)/len(perv)
    fi=max(0,1-mpe/(mp+0.001))
    ce=[{'time':perc_c[i][0],'impact':perc_c[i][1]-perc_c[i-1][1]} for i in range(1,len(perc_c)) if perc_c[i][1]-perc_c[i-1][1]>0.3]
    if fi>0.5: desc=f"alta fatiga auditiva ({fi*100:.0f}%) — escritura sin suficiente contraste; {len(ce)} momento(s) de restauración"
    elif fi>0.3: desc=f"fatiga moderada ({fi*100:.0f}%) — algunos pasajes pierden impacto"
    else: desc=f"baja fatiga ({fi*100:.0f}%) — la música mantiene frescura mediante variedad"
    return {'fatigue_index':fi,'contrast_events':ce[:5],'physical_curve':phys_c,'perceived_curve':perc_c,'desc':desc}

# ── 6. POLIRRITMIA Y HEMIOLA ──────────────────────────────────
def analyze_polyrhythm(notes,bpm,time_sig_num=4):
    if not notes or bpm<=0: return {'hemiola':False,'polyrhythm':False,'cross_rhythm':False,'desc':'ritmo regular'}
    beat=60.0/bpm; bar=beat*time_sig_num
    sn=sorted(notes,key=lambda n:n.time_sec)
    treble=[n for n in sn if n.pitch>60]; bass=[n for n in sn if n.pitch<=60]
    def dom_pulse(voice,mx=40):
        v=sorted(voice,key=lambda n:n.time_sec)[:mx]
        iois=[v[i+1].time_sec-v[i].time_sec for i in range(len(v)-1) if 0.05<v[i+1].time_sec-v[i].time_sec<bar]
        if not iois: return None
        mean=sum(iois)/len(iois)
        for s in [beat/4,beat/3,beat/2,beat*2/3,beat,beat*1.5,beat*2]:
            if abs(mean-s)<s*0.2: return s
        return mean
    tp=dom_pulse(treble); bp=dom_pulse(bass)
    poly=False; cross=False; poly_desc=''
    if tp and bp and tp>0 and bp>0:
        ratio=max(tp,bp)/min(tp,bp)
        for num,den in [(3,2),(4,3),(5,4),(3,4),(2,3)]:
            if abs(ratio-num/den)<0.15:
                poly=True; poly_desc=f"polirritmia {num}:{den} entre voces"; break
    hemiola=False; hem_events=[]
    if time_sig_num in(3,6,9):
        for i in range(0,len(sn)-3):
            w=sn[i:i+4]
            durs=[w[j+1].time_sec-w[j].time_sec for j in range(3)]
            if durs and max(durs)>0 and abs(durs[0]-durs[1])<0.05 and abs(durs[0]*2-bar*2/3)<0.1:
                hemiola=True; hem_events.append(ts_str(w[0].time_sec))
                if len(hem_events)>=3: break
    parts=[]
    if poly: parts.append(poly_desc)
    if hemiola: parts.append(f"hemiola en {', '.join(hem_events[:3])} — pulso de 3 se convierte en grupos de 2")
    if not parts: parts.append("sin polirritmia — metro regular")
    return {'hemiola':hemiola,'hemiola_events':hem_events[:5],'polyrhythm':poly,'cross_rhythm':cross,'treble_pulse':tp,'bass_pulse':bp,'desc':'; '.join(parts)}

# ── 7. ESTRUCTURA FRACTAL ─────────────────────────────────────
def analyze_fractal_structure(notes,sections,total_dur):
    if not notes or not sections: return {'fractal_index':0,'desc':''}
    sn=sorted(notes,key=lambda n:n.time_sec)
    def contour(nl,n=8):
        if len(nl)<2: return None
        step=max(1,len(nl)//n)
        pts=[nl[i].pitch for i in range(0,len(nl),step)][:n]
        if len(pts)<2: return None
        mn,mx=min(pts),max(pts); rng=mx-mn
        return [(p-mn)/rng for p in pts] if rng>0 else [0.5]*len(pts)
    def cos_sim(a,b):
        if not a or not b or len(a)!=len(b): return 0
        dot=sum(x*y for x,y in zip(a,b))
        ma=math.sqrt(sum(x**2 for x in a)); mb=math.sqrt(sum(x**2 for x in b))
        return dot/(ma*mb) if ma>0 and mb>0 else 0
    micro=contour(sn[:20])
    q=[n for n in sn if n.time_sec<total_dur/4]
    meso=contour(q,8); macro=contour(sn,8)
    s_mm=cos_sim(micro,meso) if micro and meso else 0
    s_mM=cos_sim(meso,macro) if meso and macro else 0
    s_mMa=cos_sim(micro,macro) if micro and macro else 0
    fi=(s_mm+s_mM+s_mMa)/3
    if fi>0.85: desc=f"estructura fractal pronunciada ({fi:.2f}) — material replicado a múltiples escalas"
    elif fi>0.70: desc=f"autosimilitud moderada ({fi:.2f})"
    elif fi>0.55: desc=f"similitud débil transescala ({fi:.2f})"
    else: desc=f"sin estructura fractal ({fi:.2f})"
    return {'micro_meso':s_mm,'meso_macro':s_mM,'micro_macro':s_mMa,'fractal_index':fi,'desc':desc}

# ── 8. PESO EMOCIONAL ACUMULADO ───────────────────────────────
def compute_emotional_weight(tension_curve,energy_curve,cadences,motifs,silence_data,total_dur):
    w=0.0; events=[]
    if tension_curve:
        for p in tension_curve:
            if p.tension>0.45:
                w+=p.tension*0.4
                if p.tension>0.55: events.append(('tension',p.time,p.tension))
    if energy_curve:
        ev=[e for _,e in energy_curve]; mx=max(ev) if ev else 1
        for t,e in energy_curve:
            if e/mx>0.8:
                w+=(e/mx)*0.3; events.append(('energy',t,e/mx))
    for sil in silence_data.get('silences',[]):
        if sil['duration']>1.0:
            w+=sil['duration']*sil['weight']*0.5
    cc=cadences.get('counts',{})
    w+=cc.get('rota',0)*2.0+cc.get('frigia',0)*1.5+cc.get('perfecta',0)*0.5
    w+=len(motifs)*1.0
    wpm=w/(total_dur/60) if total_dur>0 else 0
    if wpm<5: ic='contemplativa'
    elif wpm<15: ic='moderada'
    elif wpm<30: ic='intensa'
    else: ic='extrema'
    return {'total_weight':w,'weight_per_minute':wpm,'intensity_class':ic,'key_events':sorted(events,key=lambda e:-e[2])[:5],'desc':f"Peso emocional: {w:.1f} ({wpm:.1f}/min) — {ic}"}

# ── 9. PUNTO DE NO RETORNO ────────────────────────────────────
def find_point_of_no_return(tension_curve,energy_curve,total_dur):
    if not tension_curve or not energy_curve: return {'time':None,'position':None,'desc':'sin datos suficientes'}
    tv=[p.tension for p in tension_curve]
    ev=[e for _,e in energy_curve]; mx=max(ev) if ev else 1
    en=[e/mx for e in ev]
    bl=max(1,len(tv)//10)
    bt=sum(tv[:bl])/bl; be=sum(en[:min(bl,len(en))])/bl; base=(bt+be)/2
    thr=0.25; mins=int(8.0/0.5); cons=0; pnr=None
    for i,tp in enumerate(tension_curve):
        ei=min(int(tp.time*len(en)/max(total_dur,1)),len(en)-1)
        c=(tp.tension+en[ei])/2
        if c>base+thr:
            cons+=1
            if cons>=mins and pnr is None: pnr=tp.time-mins*0.5
        else: cons=0
    if pnr is None: return {'time':None,'position':None,'desc':'no se detectó punto de no retorno — reversibilidad emocional mantenida'}
    pos=pnr/total_dur
    if pos<0.33: desc=f"punto de no retorno muy temprano ({ts_str(pnr)}, {pos*100:.0f}%)"
    elif pos<0.5: desc=f"punto de no retorno en primer tercio ({ts_str(pnr)}, {pos*100:.0f}%)"
    elif pos<0.7: desc=f"punto de no retorno central ({ts_str(pnr)}, {pos*100:.0f}%) — estructura clásica"
    else: desc=f"punto de no retorno tardío ({ts_str(pnr)}, {pos*100:.0f}%) — tensión acumulada pacientemente"
    return {'time':pnr,'position':pos,'desc':desc}

# ── 10. DENSIDAD NARRATIVA POR PULSO ─────────────────────────
def analyze_narrative_density_per_pulse(notes,chords,motifs,bpm,total_dur):
    if not notes or bpm<=0 or total_dur<=0: return {'events_per_beat':0,'effective_density':0,'desc':''}
    beat=60.0/bpm; total_beats=total_dur/beat
    sn=sorted(notes,key=lambda n:n.time_sec)
    leaps=sum(1 for i in range(len(sn)-1) if abs(sn[i+1].pitch-sn[i].pitch)>=12)
    accents=sum(1 for i in range(1,len(sn)-1) if sn[i].velocity>(sn[i-1].velocity+sn[i+1].velocity)/2+15)
    total_e=len(chords)+sum(m['occ'] for m in motifs)//3+leaps//5+accents//4
    epb=total_e/max(total_beats,1)
    eff=epb*(bpm/80)
    if eff<0.3: style='contemplativo — pocas cosas por pulso'
    elif eff<0.8: style='narrativo moderado'
    elif eff<1.5: style='narrativo denso'
    else: style='torrencial'
    return {'events_per_beat':epb,'effective_density':eff,'total_beats':total_beats,'style':style,'desc':f"{style}. {epb:.2f} eventos/pulso."}

# ── 11. PROGRESIÓN EMOCIONAL ACUMULATIVA ─────────────────────
def compute_cumulative_emotional_state(sva_list,sections,tension_curve):
    active=[(s,va) for s,va in zip(sections,sva_list) if s.notes and va[2]!='sin notas']
    if not active: return {'states':[],'final_state':(0,0),'journey_type':'','desc':''}
    iner=0.4; states=[]; cv=0.0; ca=0.0
    for sec,(v,a,_) in active:
        cv=cv*iner+v*(1-iner); ca=ca*iner+a*(1-iner)
        tv=[p.tension for p in tension_curve if sec.start<=p.time<sec.end]
        at=sum(tv)/len(tv) if tv else 0
        ca=min(1.0,ca+at*0.15)
        states.append({'section':sec.index,'start':sec.start,'cum_valence':cv,'cum_arousal':ca,'instant_v':v,'instant_a':a,'cumulative_stress':min(1.0,abs(ca)*0.3+at*0.2)})
    fv,fa=states[-1]['cum_valence'],states[-1]['cum_arousal']
    vt=[s['cum_valence'] for s in states]; at2=[s['cum_arousal'] for s in states]
    vd=vt[-1]-vt[0]; ad=at2[-1]-at2[0]
    if vd>0.2 and ad>0.2: j='ascendente — oyente más activado y positivo'
    elif vd<-0.2 and ad<-0.2: j='descendente — oyente más exhausto y oscuro'
    elif vd>0.2 and ad<-0.2: j='resolutivo — serenidad positiva al final'
    elif vd<-0.2 and ad>0.2: j='perturbador — más activado pero más oscuro'
    elif abs(vd)<0.1 and abs(ad)<0.1: j='circular — retorno al estado inicial'
    else: j='complejo — trayectoria no lineal'
    mx=max(s['cumulative_stress'] for s in states)
    return {'states':states,'final_state':(fv,fa),'v_trend':vd,'a_trend':ad,'journey_type':j,'max_cumulative_stress':mx,'desc':f"Viaje acumulativo: {j}. Estrés máx: {mx:.2f}."}

# ── 12. COLOR ARMÓNICO ────────────────────────────────────────
def analyze_harmonic_color(chords,sections):
    if not chords: return {'dominant_color':None,'palette':[],'desc':'sin acordes para análisis de color'}
    rt=collections.Counter()
    for i,c in enumerate(chords):
        dur=chords[i+1].time-c.time if i+1<len(chords) else 2.0
        rt[c.root]+=max(0,dur)
    tot=sum(rt.values())
    if tot==0: return {'dominant_color':None,'palette':[],'desc':''}
    palette=[{'root':r,'note':HARMONIC_COLORS[r][0],'color':HARMONIC_COLORS[r][1],'emotion':HARMONIC_COLORS[r][2],'weight':d/tot,'time_pct':d/tot*100} for r,d in rt.most_common(6)]
    dom=palette[0] if palette else None
    top3=' / '.join(p['color'] for p in palette[:3])
    desc=f"Paleta cromática: {top3}. Centro {dom['note']} = {dom['color']} ({dom['emotion'][:40]})." if dom else ''
    return {'dominant_color':dom,'palette':palette,'desc':desc}

# ── 13. DENSIDAD ARMÓNICA POR REGISTRO ───────────────────────
def analyze_harmonic_register_density(notes,total_dur,resolution=1.0):
    if not notes: return {'profiles':{},'complexity_location':'','desc':''}
    registers={'sub-bajo':(0,47),'bajo':(48,59),'tenor':(60,71),'contralto':(72,83),'soprano':(84,127)}
    profiles={}
    for rn,(lo,hi) in registers.items():
        rn2=[n for n in notes if lo<=n.pitch<=hi]
        if not rn2: profiles[rn]={'density':0,'dissonance':0,'avg_velocity':0,'note_count':0}; continue
        density=len(rn2)/total_dur
        pcs=[n.pitch%12 for n in rn2]; dis=0; pairs=0
        for i in range(min(len(pcs),50)):
            for j in range(i+1,min(i+5,len(pcs))):
                dis+=DISSONANCE[abs(pcs[i]-pcs[j])%12]; pairs+=1
        profiles[rn]={'density':density,'dissonance':dis/pairs if pairs>0 else 0,'avg_velocity':sum(n.velocity for n in rn2)/len(rn2),'note_count':len(rn2)}
    md=max(profiles.items(),key=lambda x:x[1].get('density',0))
    mdis=max(profiles.items(),key=lambda x:x[1].get('dissonance',0))
    if md[0]==mdis[0]: cl=f"complejidad concentrada en {md[0]}"
    else: cl=f"mayor densidad en {md[0]}, mayor disonancia en {mdis[0]}"
    return {'profiles':profiles,'complexity_location':cl,'desc':cl}

# ── 14. PITCH BEND ────────────────────────────────────────────
def analyze_pitch_bend(meta,total_dur):
    pb=meta.get('pitch_bend_events',[])
    if not pb: return {'has_pitch_bend':False,'vibrato':False,'glissando':False,'expressiveness':0,'desc':'sin pitch bend — notas perfectamente afinadas'}
    vals=[v for _,v in pb]; times=[t for t,_ in pb]
    max_b=max(abs(v) for v in vals); rng=max(vals)-min(vals)
    oscs=sum(1 for i in range(1,len(vals)-1) if (vals[i]-vals[i-1])*(vals[i+1]-vals[i])<0)
    vib_rate=oscs/(total_dur+0.001); vib=vib_rate>2
    diffs=[vals[i+1]-vals[i] for i in range(len(vals)-1)]
    same_dir=sum(1 for d in diffs if d>0)/len(diffs) if diffs else 0
    glis=same_dir>0.8 and rng>2000
    expr=min(1.0,rng/8192)
    parts=[]
    if vib: parts.append(f"vibrato ({vib_rate:.1f} osc/s)")
    if glis: parts.append("glissando")
    if max_b>4000: parts.append(f"desviación microtonal (±{max_b/8192*200:.0f} centavos)")
    return {'has_pitch_bend':True,'vibrato':vib,'glissando':glis,'max_bend':max_b,'range_bend':rng,'expressiveness':expr,'desc':'; '.join(parts) if parts else 'pitch bend sutil'}


# ══ ANALYSES v7 ══

CULTURAL_MARKERS = [
    ((2,1,2,2,1,2,2),   'Escala mayor (diatónica occidental)',  'familiaridad, tonalidad clásica/pop'),
    ((2,1,2,2,1,3,1),   'Escala menor armónica',                'drama, exotismo, tensión orientalizante'),
    ((1,3,1,2,1,3,1),   'Escala disminuida (octatónica)',        'jazz, Stravinsky, tensión simétrica'),
    ((2,2,2,2,2,2),     'Escala de tonos enteros',               'Debussy, impresionismo, flotación'),
    ((2,1,4,1,4),       'Escala pentatónica menor',              'blues, folk, universalidad emocional'),
    ((2,2,1,2,2,2,1),   'Escala lidio',                          'épico, cinematográfico, John Williams'),
    ((1,2,2,1,3,2,1),   'Escala frigia (flamenco/árabe)',        'fatalismo, flamenco, exotismo mediterráneo'),
]

# Cultural style markers: (interval_pattern, style, emotion)


def analyze_cultural_markers(notes, key_root: int, mode: str, chords) -> Dict:
    """
    Detecta marcadores estilísticos culturales en la música:
    escalas, progresiones y gestos con memoria cultural específica.
    """
    if not notes:
        return {'markers': [], 'dominant_style': '', 'desc': ''}

    sn = sorted(notes, key=lambda n: n.time_sec)
    pc_set = set(n.pitch % 12 for n in sn)
    key_shifted = {(p - key_root) % 12 for p in pc_set}

    markers = []

    # Check scale affinity
    for ivls, style_name, emotion in CULTURAL_MARKERS:
        scale_pcs = set()
        p = 0
        for step in ivls:
            scale_pcs.add(p % 12)
            p += step
        overlap = len(key_shifted & scale_pcs) / max(len(scale_pcs), 1)
        if overlap > 0.75:
            markers.append({'style': style_name, 'affinity': overlap, 'emotion': emotion})

    # Chord-based markers
    chord_names = [c.name for c in chords]
    if any('dom7' in c for c in chord_names):
        markers.append({'style': 'Blues/Jazz (dominante 7ª)', 'affinity': 0.85,
                        'emotion': 'tensión blues, grito del alma, jazz'})
    if any('maj7' in c for c in chord_names) and any('min7' in c for c in chord_names):
        markers.append({'style': 'Jazz modal (maj7 + min7)', 'affinity': 0.80,
                        'emotion': 'sofisticación armónica, Bill Evans, Miles Davis'})
    sus_count = sum(1 for c in chord_names if 'sus' in c)
    if sus_count > len(chords) * 0.15:
        markers.append({'style': 'Rock/Ambient (sus chords)', 'affinity': sus_count/len(chords),
                        'emotion': 'apertura modal, ambigüedad tonal, años 90-2000'})
    if any('dim7' in c for c in chord_names):
        markers.append({'style': 'Romántico/Tardorromántico (dim7)', 'affinity': 0.75,
                        'emotion': 'drama, Chopin, Liszt, tensión cromática'})

    # Detect Andalusian cadence: i-VII-VI-V
    deg_seq = [((c.root - key_root) % 12) for c in chords]
    andal = [0, 10, 8, 7]
    for i in range(len(deg_seq) - 3):
        if deg_seq[i:i+4] == andal:
            markers.append({'style': 'Cadencia andaluza (flamenco)', 'affinity': 1.0,
                            'emotion': 'fatalismo ibérico, duende, lo inevitable'})
            break

    # Sort by affinity
    markers.sort(key=lambda m: -m['affinity'])
    markers = markers[:5]  # top 5

    dom = markers[0]['style'] if markers else 'indeterminado'
    desc_parts = [f"{m['style']} ({m['affinity']*100:.0f}%)" for m in markers[:3]]
    desc = 'Marcadores culturales: ' + ' / '.join(desc_parts) if desc_parts else 'sin marcadores culturales dominantes'

    return {'markers': markers, 'dominant_style': dom, 'desc': desc}


# ═══════════════════════════════════════════════════════════════
#  2. VOZ CONDUCTORA DEL DISCURSO
# ═══════════════════════════════════════════════════════════════

def analyze_narrative_voice(notes, sections, tension_curve) -> Dict:
    """
    Identifica qué voz lleva el hilo narrativo principal en cada sección.
    Criterios: mayor variedad interválica, mayor independencia, más cambios de dirección.
    """
    if not notes:
        return {'voice_map': [], 'protagonist_shifts': 0, 'desc': ''}

    # Group by channel
    by_ch = collections.defaultdict(list)
    for n in notes: by_ch[n.channel].append(n)

    if len(by_ch) < 2:
        by_ch = collections.defaultdict(list)
        for n in notes: by_ch[n.track].append(n)

    if len(by_ch) < 2:
        return {'voice_map': [], 'protagonist_shifts': 0,
                'desc': 'una sola voz — la narrativa recae íntegramente sobre ella'}

    voice_map = []
    prev_leader = None
    shifts = 0

    for sec in sections:
        if not sec.notes:
            continue

        scores = {}
        for ch, ch_notes in by_ch.items():
            sec_notes = sorted([n for n in ch_notes if sec.start <= n.time_sec < sec.end],
                               key=lambda n: n.time_sec)
            if len(sec_notes) < 3:
                scores[ch] = 0; continue

            # Intervallic variety
            ivs = [abs(sec_notes[i+1].pitch - sec_notes[i].pitch) for i in range(len(sec_notes)-1)]
            iv_variety = len(set(ivs)) / max(len(ivs), 1)

            # Direction changes (melodic activity)
            dir_changes = sum(1 for i in range(1, len(ivs))
                              if ivs[i] > 0 and ivs[i-1] > 0 and
                              (sec_notes[i+1].pitch - sec_notes[i].pitch) *
                              (sec_notes[i].pitch - sec_notes[i-1].pitch) < 0)
            dir_rate = dir_changes / max(len(ivs), 1)

            # Dynamic range
            vels = [n.velocity for n in sec_notes]
            vel_range = max(vels) - min(vels)

            # Tension alignment: does this voice's pitch changes correlate with tension changes?
            t_vals = [p.tension for p in tension_curve if sec.start <= p.time < sec.end]
            if t_vals and len(sec_notes) >= len(t_vals):
                pitch_changes = [abs(sec_notes[min(i, len(sec_notes)-1)].pitch -
                                     sec_notes[max(i-1, 0)].pitch)
                                 for i in range(len(t_vals))]
                t_align = pearson(
                    [v/max(t_vals) if max(t_vals) > 0 else 0 for v in t_vals],
                    [v/max(pitch_changes) if max(pitch_changes) > 0 else 0 for v in pitch_changes]
                ) if max(pitch_changes) > 0 else 0
            else:
                t_align = 0

            scores[ch] = iv_variety * 0.30 + dir_rate * 0.25 + vel_range/127 * 0.20 + max(0, t_align) * 0.25

        if scores:
            leader = max(scores, key=lambda k: scores[k])
            avg_pitch = sum(n.pitch for n in by_ch[leader]) / max(len(by_ch[leader]), 1)
            register = ('soprano' if avg_pitch > 72 else 'contralto' if avg_pitch > 60
                        else 'tenor' if avg_pitch > 48 else 'bajo')

            if prev_leader is not None and leader != prev_leader:
                shifts += 1

            voice_map.append({
                'section': sec.index,
                'start': sec.start,
                'leader': leader,
                'score': scores[leader],
                'register': register,
                'all_scores': scores
            })
            prev_leader = leader

    if not voice_map:
        return {'voice_map': [], 'protagonist_shifts': 0, 'desc': 'sin datos suficientes'}

    if shifts == 0:
        desc = 'una sola voz domina el discurso — narrativa monológica, coherencia máxima'
    elif shifts <= 2:
        desc = f'{shifts} cambio(s) de voz protagonista — narrativa con transiciones expresivas'
    else:
        desc = f'{shifts} cambios de voz protagonista — diálogo polifónico, múltiples perspectivas'

    return {'voice_map': voice_map, 'protagonist_shifts': shifts, 'desc': desc}


# ═══════════════════════════════════════════════════════════════
#  3. MICRO-TIMING Y HUMANIZACIÓN
# ═══════════════════════════════════════════════════════════════

def analyze_micro_timing(notes, bpm: float, meta: Dict) -> Dict:
    """
    Detecta desviaciones del grid rítmico perfecto.
    Mide: humanización, drag, anticipación, mecanicidad.
    """
    if not notes or bpm <= 0:
        return {'humanization': 0, 'mean_deviation': 0, 'drag': 0,
                'anticipation': 0, 'desc': 'datos insuficientes'}

    beat = 60.0 / bpm
    tpb = meta.get('tpb', 480)
    sn = sorted(notes, key=lambda n: n.time_sec)

    # Quantise each note to nearest 16th note grid
    grid = beat / 4
    deviations = []
    drags = 0; anticipations = 0

    for n in sn:
        nearest_grid = round(n.time_sec / grid) * grid
        dev = n.time_sec - nearest_grid
        deviations.append(dev)
        if dev > grid * 0.08:   drags += 1        # late = "dragged"
        elif dev < -grid * 0.08: anticipations += 1  # early = "pushed"

    if not deviations:
        return {'humanization': 0, 'mean_deviation': 0, 'desc': 'sin datos de timing'}

    abs_devs = [abs(d) for d in deviations]
    mean_dev = sum(abs_devs) / len(abs_devs)
    max_dev  = max(abs_devs)
    std_dev  = math.sqrt(sum((d - mean_dev)**2 for d in abs_devs) / len(abs_devs))

    drag_ratio = drags / len(sn)
    ant_ratio  = anticipations / len(sn)

    # Humanization: 0 = perfect mechanical, 1 = highly expressive/loose
    humanization = min(1.0, mean_dev / (grid * 0.15))

    # Style classification
    if humanization < 0.1:
        style = 'mecánico/cuantizado — precisión total, sin swing ni expresión rítmica'
    elif humanization < 0.3:
        style = 'ligeramente humanizado — timing limpio con matices expresivos sutiles'
    elif humanization < 0.6:
        style = 'humanizado — expressión rítmica clara, feel natural'
    else:
        style = 'muy libre rítmicamente — rubato, improvisación o interpretación muy expresiva'

    feel = ''
    if drag_ratio > ant_ratio + 0.1:
        feel = ' Tendencia al "drag" (detrás del tiempo) — feel relajado, groove laid-back.'
    elif ant_ratio > drag_ratio + 0.1:
        feel = ' Tendencia a anticipar (delante del tiempo) — urgencia, energía hacia adelante.'

    return {
        'humanization': humanization,
        'mean_deviation_ms': mean_dev * 1000,
        'std_deviation_ms': std_dev * 1000,
        'max_deviation_ms': max_dev * 1000,
        'drag_ratio': drag_ratio,
        'anticipation_ratio': ant_ratio,
        'style': style,
        'desc': f"{style}.{feel}",
    }


# ═══════════════════════════════════════════════════════════════
#  4. CONVERGENCIA DE PARÁMETROS (confluencia expresiva)
# ═══════════════════════════════════════════════════════════════

def analyze_parameter_convergence(notes, tension_curve, energy_curve,
                                   sections, total_dur: float) -> Dict:
    """
    Mide cuántos parámetros se mueven en la misma dirección simultáneamente.
    Alta convergencia = momentos de máximo impacto emocional.
    """
    if not notes or not tension_curve:
        return {'peaks': [], 'mean_convergence': 0, 'curve': [], 'desc': ''}

    resolution = 1.0
    sn = sorted(notes, key=lambda n: n.time_sec)
    t_map = {p.time: p.tension for p in tension_curve}
    e_vals = {t: e for t, e in energy_curve} if energy_curve else {}
    max_e = max(e_vals.values()) if e_vals else 1

    curve = []
    t = resolution
    while t < total_dur - resolution:
        # Get values at t and t-resolution
        def get_tension(time):
            times = list(t_map.keys())
            if not times: return 0
            closest = min(times, key=lambda x: abs(x - time))
            return t_map[closest]

        def get_energy(time):
            if not e_vals: return 0
            times = list(e_vals.keys())
            closest = min(times, key=lambda x: abs(x - time))
            return e_vals[closest] / max_e

        # Parameter changes over last `resolution` seconds
        notes_now  = [n for n in sn if t - resolution <= n.time_sec < t]
        notes_prev = [n for n in sn if t - 2*resolution <= n.time_sec < t - resolution]

        params = {}

        # 1. Tension direction
        dt = get_tension(t) - get_tension(t - resolution)
        params['tension'] = dt

        # 2. Energy direction
        de = get_energy(t) - get_energy(t - resolution)
        params['energy'] = de

        # 3. Density direction
        d_now  = len(notes_now) / resolution
        d_prev = len(notes_prev) / resolution if notes_prev is not None else d_now
        params['density'] = d_now - d_prev

        # 4. Pitch direction (average pitch change)
        p_now  = sum(n.pitch for n in notes_now) / max(len(notes_now), 1) if notes_now else 60
        p_prev = sum(n.pitch for n in notes_prev) / max(len(notes_prev), 1) if notes_prev else p_now
        params['pitch'] = p_now - p_prev

        # 5. Velocity direction
        v_now  = sum(n.velocity for n in notes_now) / max(len(notes_now), 1) if notes_now else 64
        v_prev = sum(n.velocity for n in notes_prev) / max(len(notes_prev), 1) if notes_prev else v_now
        params['velocity'] = v_now - v_prev

        # Normalize directions to -1, 0, +1
        dirs = []
        for k, v in params.items():
            threshold = {'tension': 0.02, 'energy': 0.02, 'density': 0.5,
                         'pitch': 0.5, 'velocity': 2.0}.get(k, 0.1)
            dirs.append(1 if v > threshold else (-1 if v < -threshold else 0))

        # Convergence: how many non-zero params agree on direction
        active = [d for d in dirs if d != 0]
        if not active:
            convergence = 0.0
        else:
            pos = sum(1 for d in active if d > 0)
            neg = sum(1 for d in active if d < 0)
            dominant = max(pos, neg)
            convergence = dominant / len(dirs)

        curve.append((t, convergence, 1 if (active and sum(active)/len(active) > 0) else -1))
        t += resolution

    if not curve:
        return {'peaks': [], 'mean_convergence': 0, 'curve': [], 'desc': ''}

    conv_vals = [c for _, c, _ in curve]
    mean_conv = sum(conv_vals) / len(conv_vals)

    # Find convergence peaks (score > 0.7)
    peaks = [(t, c, d) for t, c, d in curve if c > 0.7]
    # Deduplicate (keep one per 3-second window)
    deduped = []
    last_t = -999
    for t, c, d in sorted(peaks, key=lambda x: -x[1]):
        if t - last_t > 3.0:
            deduped.append({'time': t, 'convergence': c,
                            'direction': 'crescendo' if d > 0 else 'decrescendo'})
            last_t = t
    deduped.sort(key=lambda x: x['time'])

    desc_parts = []
    if mean_conv > 0.6:
        desc_parts.append(f"alta convergencia media ({mean_conv:.2f}) — parámetros frecuentemente alineados, impactos fuertes")
    elif mean_conv > 0.4:
        desc_parts.append(f"convergencia moderada ({mean_conv:.2f})")
    else:
        desc_parts.append(f"parámetros frecuentemente independientes ({mean_conv:.2f}) — riqueza de capas")
    if deduped:
        desc_parts.append(f"{len(deduped)} pico(s) de confluencia expresiva")

    return {
        'peaks': deduped[:8],
        'mean_convergence': mean_conv,
        'curve': [(t, c) for t, c, _ in curve],
        'desc': '; '.join(desc_parts),
    }


# ═══════════════════════════════════════════════════════════════
#  5. ZONA DE CONFORT TONAL
# ═══════════════════════════════════════════════════════════════

def analyze_tonal_comfort_zones(notes, sections, key_root: int, mode: str) -> Dict:
    """
    Calcula el % de notas diatónicas por sección.
    Alta pertenencia = zona de confort.
    Baja pertenencia = zona de aventura/riesgo emocional.
    """
    if not notes:
        return {'section_comfort': [], 'mean_comfort': 0, 'adventure_zones': [], 'desc': ''}

    scale = set(SCALE_IVLS.get(mode, SCALE_IVLS['minor']))
    section_comfort = []

    for sec in sections:
        if not sec.notes:
            section_comfort.append({'index': sec.index, 'start': sec.start, 'comfort': 0.5})
            continue
        total = len(sec.notes)
        diatonic = sum(1 for n in sec.notes if (n.pitch - key_root) % 12 in scale)
        comfort = diatonic / total
        section_comfort.append({
            'index': sec.index,
            'start': sec.start,
            'comfort': comfort,
            'diatonic': diatonic,
            'total': total
        })

    if not section_comfort:
        return {'section_comfort': [], 'mean_comfort': 0, 'adventure_zones': [], 'desc': ''}

    comforts = [s['comfort'] for s in section_comfort]
    mean_c = sum(comforts) / len(comforts)
    adventure_zones = [s for s in section_comfort if s['comfort'] < 0.6]
    comfort_zones   = [s for s in section_comfort if s['comfort'] > 0.85]

    if mean_c > 0.85:
        desc = f"pieza altamente diatónica ({mean_c*100:.0f}%) — zona de confort tonal permanente"
    elif mean_c > 0.7:
        desc = f"predominantemente diatónica ({mean_c*100:.0f}%) con aventuras cromáticas puntuales"
    elif mean_c > 0.55:
        desc = f"equilibrio entre diatonismo y cromatismo ({mean_c*100:.0f}%)"
    else:
        desc = f"escritura cromática predominante ({mean_c*100:.0f}%) — riesgo tonal elevado"

    if adventure_zones:
        az_strs = [ts_str(z['start']) for z in adventure_zones[:3]]
        desc += f"; zonas de aventura en {', '.join(az_strs)}"

    return {
        'section_comfort': section_comfort,
        'mean_comfort': mean_c,
        'adventure_zones': adventure_zones,
        'comfort_zones': comfort_zones,
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  6. JERARQUÍA MÉTRICA (Lerdahl-Jackendoff simplificado)
# ═══════════════════════════════════════════════════════════════

def analyze_metric_hierarchy(notes, bpm: float, time_sig_num: int = 4) -> Dict:
    """
    Asigna peso métrico a cada nota según su posición en el compás.
    Detecta si el compositor refuerza o contradice la métrica.
    """
    if not notes or bpm <= 0:
        return {'metric_reinforcement': 0.5, 'syncopation_score': 0, 'desc': ''}

    beat     = 60.0 / bpm
    bar      = beat * time_sig_num
    sixteenth= beat / 4

    # Metric weight by position within bar (simplified Lerdahl-Jackendoff)
    def metric_weight(t):
        pos = t % bar
        # Bar-level: beat 1 strongest
        beat_pos = pos / beat
        frac = beat_pos % 1.0
        if frac < 0.05:        # on the beat
            beat_num = round(beat_pos)
            if beat_num == 0:   return 4   # downbeat
            elif beat_num == time_sig_num // 2: return 3  # middle beat
            else:               return 2   # other beats
        elif frac < 0.25:      return 1   # 8th note off beat
        else:                  return 0   # 16th level

    sn = sorted(notes, key=lambda n: n.time_sec)
    weights_and_notes = [(metric_weight(n.time_sec), n) for n in sn]

    # Metric reinforcement: do high-weight positions get loud/high notes?
    correlations = []
    for w, n in weights_and_notes:
        if w > 0:
            # Normalize: velocity and weight both 0-1
            v_norm = n.velocity / 127.0
            w_norm = w / 4.0
            correlations.append(v_norm * w_norm)

    reinforcement = sum(correlations) / len(correlations) if correlations else 0.5

    # Syncopation: how often do strong notes land on weak beats?
    accents = [n for n in sn[1:-1]
               if n.velocity > (sn[sn.index(n)-1].velocity + sn[sn.index(n)+1].velocity) / 2 + 15]
    syncopated_accents = sum(1 for n in accents if metric_weight(n.time_sec) == 0)
    sync_score = syncopated_accents / max(len(accents), 1)

    # Distribution of notes by metric weight
    weight_dist = collections.Counter(metric_weight(n.time_sec) for n in sn)
    total = len(sn)

    # Build description
    parts = []
    if reinforcement > 0.6:
        parts.append("escritura métricamente reforzada — los tiempos fuertes son más intensos, claridad rítmica")
    elif reinforcement < 0.35:
        parts.append("escritura métricamente contradictoria — la intensidad no sigue el pulso, efecto flotante")
    else:
        parts.append("relación métrica equilibrada")

    if sync_score > 0.4:
        parts.append(f"acentuación sincopada frecuente ({sync_score*100:.0f}%) — desplazamiento del acento, groove")

    # Most populated metric level
    top_level = weight_dist.most_common(1)[0][0] if weight_dist else 0
    level_desc = {4: 'tiempos principales', 3: 'tiempos secundarios',
                  2: 'partes de tiempo', 1: 'contratiempos', 0: 'notas de paso'}
    parts.append(f"densidad principal en {level_desc.get(top_level, 'nivel indeterminado')} ({weight_dist.get(top_level,0)/total*100:.0f}%)")

    return {
        'metric_reinforcement': reinforcement,
        'syncopation_score': sync_score,
        'weight_distribution': dict(weight_dist),
        'total_notes': total,
        'desc': '; '.join(parts),
    }


# ═══════════════════════════════════════════════════════════════
#  7. CADENCIAS ELÍPTICAS
# ═══════════════════════════════════════════════════════════════

def detect_elliptical_cadences(chords, key_root: int, mode: str,
                                notes, bpm: float) -> Dict:
    """
    Detecta elipsis cadenciales: frases que "se saltan" la resolución esperada,
    continúan sin resolución o interrumpen el patrón esperado.
    """
    if len(chords) < 3 or bpm <= 0:
        return {'ellipses': [], 'count': 0, 'desc': ''}

    beat  = 60.0 / bpm
    dom   = (key_root + 7) % 12
    subd  = (key_root + 5) % 12
    tonic = key_root
    ellipses = []

    for i in range(1, len(chords) - 1):
        c_prev = chords[i-1]
        c_curr = chords[i]
        c_next = chords[i+1]

        # Type 1: Dominant that doesn't resolve (V → not-I)
        if c_curr.root == dom and c_next.root != tonic:
            gap = c_next.time - c_curr.time
            if gap < beat * 4:  # quick non-resolution
                ellipses.append({
                    'time': c_curr.time,
                    'type': 'dominante evadida',
                    'from': c_curr.name,
                    'to': c_next.name,
                    'emotion': 'tensión que escapa — la resolución se niega, suspense prolongado'
                })

        # Type 2: Expected continuation interrupted (phrase cut short)
        # Sub → Dom → (expected tonic, got something else)
        if (c_prev.root == subd and c_curr.root == dom and
                c_next.root != tonic and c_next.root != (tonic + 4) % 12):
            ellipses.append({
                'time': c_next.time,
                'type': 'frase truncada',
                'from': f"{c_prev.name}→{c_curr.name}",
                'to': c_next.name,
                'emotion': 'interrupción brusca — la frase se corta antes de completarse'
            })

        # Type 3: IV → I expected, got IV → something else (avoided plagal)
        if c_prev.root == subd and c_curr.root != tonic and c_prev.chord_type in ('maj','min'):
            pass  # too many false positives, skip

    # Deduplicate by time window
    seen_times = set()
    deduped = []
    for e in ellipses:
        t_bucket = round(e['time'] / (beat * 2))
        if t_bucket not in seen_times:
            seen_times.add(t_bucket)
            deduped.append(e)

    count = len(deduped)
    if count == 0:
        desc = 'sin elipsis cadenciales — frases bien completadas'
    elif count <= 3:
        desc = f'{count} elipsis cadencial(es) — desvíos narrativos expresivos'
    else:
        desc = f'{count} elipsis cadenciales — escritura sistemáticamente evasiva, ansiedad tonal'

    return {'ellipses': deduped[:8], 'count': count, 'desc': desc}


# ═══════════════════════════════════════════════════════════════
#  8. RECURRENCIA TEMÁTICA CON DISTANCIA TEMPORAL
# ═══════════════════════════════════════════════════════════════

def analyze_thematic_recurrence(motifs, notes, total_dur: float, bpm: float) -> Dict:
    """
    Para cada motivo, rastrea sus apariciones en el tiempo y calcula:
    - Distancia media entre apariciones
    - Variación entre apariciones (¿siempre igual o transformado?)
    - Efecto de "reconocimiento" por distancia
    """
    if not motifs or not notes or total_dur <= 0:
        return {'recurrences': [], 'mean_distance': 0, 'recognition_effect': 0, 'desc': ''}

    sn = sorted(notes, key=lambda n: n.time_sec)
    intervals = [sn[i+1].pitch - sn[i].pitch for i in range(len(sn)-1)]
    times = [sn[i].time_sec for i in range(len(sn)-1)]
    beat = 60.0 / bpm if bpm > 0 else 0.5

    recurrences = []
    for motif in motifs[:3]:
        pat = list(motif['pat'])
        pl = len(pat)
        appearances = []
        for i in range(len(intervals) - pl + 1):
            if intervals[i:i+pl] == pat:
                appearances.append(times[i])

        if len(appearances) < 2:
            continue

        # Time gaps between appearances
        gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0
        max_gap  = max(gaps) if gaps else 0

        # Recognition effect: long gaps → stronger recognition
        recognition = min(1.0, mean_gap / 20.0)  # 20s gap = max recognition

        # Temporal distribution: early/late concentration
        early = sum(1 for a in appearances if a < total_dur * 0.33)
        late  = sum(1 for a in appearances if a > total_dur * 0.66)

        recurrences.append({
            'motif': motif['desc'][:50],
            'appearances': len(appearances),
            'mean_gap': mean_gap,
            'max_gap': max_gap,
            'recognition_effect': recognition,
            'early_count': early,
            'late_count': late,
            'times': appearances[:8]
        })

    if not recurrences:
        return {'recurrences': [], 'mean_distance': 0, 'recognition_effect': 0,
                'desc': 'motivos sin suficientes apariciones para análisis de recurrencia'}

    overall_recognition = sum(r['recognition_effect'] for r in recurrences) / len(recurrences)
    mean_dist = sum(r['mean_gap'] for r in recurrences) / len(recurrences)

    if overall_recognition > 0.6:
        desc = f"alta distancia entre apariciones ({mean_dist:.1f}s μ) — fuerte efecto de reconocimiento y nostalgia"
    elif overall_recognition > 0.3:
        desc = f"distancia media ({mean_dist:.1f}s μ) — reconocimiento claro sin efecto de lejanía"
    else:
        desc = f"apariciones frecuentes ({mean_dist:.1f}s μ) — familiaridad por repetición, efecto hipnótico"

    return {
        'recurrences': recurrences,
        'mean_distance': mean_dist,
        'recognition_effect': overall_recognition,
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  9. LISTENER MODELING (arco de escucha proyectado)
# ═══════════════════════════════════════════════════════════════

def model_listener_experience(tension_curve, energy_curve, expectation,
                               auditory_fatigue, cumulative_state,
                               total_dur: float) -> Dict:
    """
    Modela la experiencia subjetiva proyectada del oyente combinando:
    - Tensión emocional acumulada
    - Fatiga y adaptación auditiva
    - Sorpresa melódica (expectativa)
    - Estado emocional acumulativo
    Produce una curva de "intensidad de experiencia subjetiva".
    """
    if not tension_curve:
        return {'curve': [], 'peak_experience': 0, 'peak_time': 0,
                'exhaustion_index': 0, 'desc': ''}

    t_vals = [p.tension for p in tension_curve]
    t_times = [p.time for p in tension_curve]

    # Get perceived energy (fatigue-adjusted)
    phys_c = auditory_fatigue.get('physical_curve', []) if auditory_fatigue else []
    perc_c = auditory_fatigue.get('perceived_curve', []) if auditory_fatigue else []
    mean_surprise = expectation.get('mean_surprise', 0.2) if expectation else 0.2

    # Cumulative stress from cumulative state
    cum_states = cumulative_state.get('states', []) if cumulative_state else []
    stress_map = {s['section']: s.get('cumulative_stress', 0) for s in cum_states}

    # Build experience curve
    resolution = t_times[1] - t_times[0] if len(t_times) > 1 else 0.5
    curve = []
    cumulative_fatigue = 0.0
    tau_fatigue = 30.0  # fatigue decay constant

    for i, (t, tension) in enumerate(zip(t_times, t_vals)):
        # Physical energy at this time
        if perc_c:
            pc_idx = min(int(t / resolution), len(perc_c)-1)
            perceived_energy = perc_c[pc_idx][1] if pc_idx < len(perc_c) else 0
        else:
            perceived_energy = tension * 0.5

        # Surprise contribution
        surprise_contrib = mean_surprise * 0.3

        # Cumulative fatigue (exponential buildup and decay)
        alpha = 1 - math.exp(-resolution / tau_fatigue)
        cumulative_fatigue = alpha * (tension * 0.5) + (1-alpha) * cumulative_fatigue
        fatigue_penalty = cumulative_fatigue * 0.2

        # Experience = tension + energy - fatigue + surprise
        experience = (tension * 0.40 +
                      perceived_energy * 0.30 +
                      surprise_contrib * 0.20 -
                      fatigue_penalty)
        experience = max(0.0, min(1.0, experience))

        curve.append((t, experience))

    if not curve:
        return {'curve': [], 'peak_experience': 0, 'peak_time': 0,
                'exhaustion_index': 0, 'desc': ''}

    exp_vals = [e for _, e in curve]
    peak_idx = exp_vals.index(max(exp_vals))
    peak_time = curve[peak_idx][0]
    peak_val  = exp_vals[peak_idx]
    mean_exp  = sum(exp_vals) / len(exp_vals)

    # Exhaustion: experience drops significantly in the last 20%
    last_fifth = exp_vals[int(len(exp_vals)*0.8):]
    if last_fifth:
        exhaustion = max(0, mean_exp - sum(last_fifth)/len(last_fifth))
    else:
        exhaustion = 0

    # Shape
    thirds = len(exp_vals) // 3
    e1 = sum(exp_vals[:thirds])/thirds if thirds else 0
    e2 = sum(exp_vals[thirds:2*thirds])/thirds if thirds else 0
    e3 = sum(exp_vals[2*thirds:])/max(1,len(exp_vals)-2*thirds)

    if e3 > e1 + 0.1:
        shape = 'experiencia acumulativa — el oyente está más involucrado al final'
    elif e1 > e3 + 0.1:
        shape = 'experiencia decreciente — máximo impacto inicial, luego adaptación'
    elif e2 > e1 + 0.05 and e2 > e3 + 0.05:
        shape = 'experiencia en arco — pico central, apertura y cierre más calmados'
    else:
        shape = 'experiencia relativamente uniforme'

    return {
        'curve': curve,
        'peak_experience': peak_val,
        'peak_time': peak_time,
        'mean_experience': mean_exp,
        'exhaustion_index': exhaustion,
        'shape': shape,
        'desc': f"Experiencia proyectada: {shape}. Pico en {ts_str(peak_time)} ({peak_val:.2f}). Agotamiento: {exhaustion:.2f}.",
    }


def listener_ascii(curve, width=56, height=6) -> str:
    """ASCII chart of listener experience curve."""
    if not curve: return ''
    vals = [e for _, e in curve]
    step = max(1, len(vals) // width)
    sampled = [max(vals[i:i+step]) for i in range(0, len(vals), step)][:width]
    mx = max(sampled) if sampled else 1
    if mx == 0: return ''
    sampled = [v/mx for v in sampled]
    lines = []
    for row in range(height, 0, -1):
        thr = row / height
        line = ''.join('█' if v >= thr else ('▄' if v >= thr-0.5/height else ' ')
                       for v in sampled)
        lbl = f"{thr:.1f}" if row in (height, height//2, 1) else '    '
        lines.append(f"  {lbl} │{line}│")
    total = curve[-1][0]
    lines.append(f"       └{'─'*len(sampled)}┘")
    lines.append(f"       0{' '*(len(sampled)//2-2)}{ts_str(total/2)}{' '*(len(sampled)//2-3)}{ts_str(total)}")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  10. DENSIDAD SEMÁNTICA ARMÓNICA
# ═══════════════════════════════════════════════════════════════

def analyze_harmonic_semantic_density(chords, total_dur: float) -> Dict:
    """
    Calcula la "novedad semántica" de cada acorde según su contexto previo.
    Un acorde que repite lo ya dicho tiene densidad baja.
    Un acorde nuevo en contexto impredecible tiene densidad alta.
    """
    if len(chords) < 2:
        return {'novelty_curve': [], 'mean_novelty': 0, 'desc': ''}

    novelty_curve = []
    seen_chords = collections.Counter()
    total_seen = 0

    for i, chord in enumerate(chords):
        seen_chords[chord.name] += 1
        total_seen += 1

        if total_seen <= 1:
            novelty = 1.0
        else:
            # Base novelty: inverse of how often this chord has appeared
            freq = seen_chords[chord.name] / total_seen
            repetition_novelty = 1.0 - freq

            # Context novelty: was this chord expected after the previous one?
            if i > 0:
                prev = chords[i-1].name
                # Count how often prev → this has occurred
                prev_occurrences = sum(1 for j in range(max(0,i-1), i)
                                       if j > 0 and chords[j-1].name == prev and chords[j].name == chord.name)
                total_after_prev = sum(1 for j in range(1, i) if chords[j-1].name == prev)
                if total_after_prev > 0:
                    context_freq = prev_occurrences / total_after_prev
                    context_novelty = 1.0 - context_freq
                else:
                    context_novelty = 1.0  # first time after this previous chord
            else:
                context_novelty = 1.0

            novelty = repetition_novelty * 0.4 + context_novelty * 0.6

        novelty_curve.append({'time': chord.time, 'chord': chord.name, 'novelty': novelty})

    novelties = [n['novelty'] for n in novelty_curve]
    mean_n = sum(novelties) / len(novelties) if novelties else 0

    # High novelty peaks
    peaks = sorted(novelty_curve, key=lambda x: -x['novelty'])[:5]

    if mean_n > 0.7:
        desc = f"alta densidad semántica ({mean_n:.2f}) — la armonía constantemente dice algo nuevo"
    elif mean_n > 0.5:
        desc = f"densidad semántica moderada ({mean_n:.2f}) — equilibrio entre novedad y confirmación"
    else:
        desc = f"baja densidad semántica ({mean_n:.2f}) — armonía repetitiva, familiar, reconfortante"

    return {
        'novelty_curve': novelty_curve,
        'mean_novelty': mean_n,
        'high_novelty_peaks': peaks,
        'desc': desc,
    }


# ═══════════════════════════════════════════════════════════════
#  11. ANÁLISIS DEL BAJO (función narrativa)
# ═══════════════════════════════════════════════════════════════

def analyze_bass_narrative(notes, bpm: float, key_root: int) -> Dict:
    """
    Analiza el bajo de forma aislada y determina su función narrativa:
    pedal, ostinato, walking bass, contrapunto, o mixta.
    """
    if not notes or bpm <= 0:
        return {'function': 'indeterminado', 'desc': ''}

    beat = 60.0 / bpm
    bass = sorted([n for n in notes if n.pitch < 55], key=lambda n: n.time_sec)

    if len(bass) < 8:
        return {'function': 'ausente o mínimo',
                'desc': 'bajo ausente o mínimo — registro grave sin definición propia'}

    total = max(n.time_sec for n in bass) - min(n.time_sec for n in bass)
    if total <= 0:
        return {'function': 'indeterminado', 'desc': ''}

    # Metrics
    pcs = [n.pitch % 12 for n in bass]
    pc_variety = len(set(pcs))
    dom_pc = collections.Counter(pcs).most_common(1)[0]
    dom_ratio = dom_pc[1] / len(pcs)

    iois = [bass[i+1].time_sec - bass[i].time_sec for i in range(len(bass)-1)
            if 0.05 < bass[i+1].time_sec - bass[i].time_sec < beat*4]
    if iois:
        mean_ioi = sum(iois) / len(iois)
        reg = 1 - sum(abs(x-mean_ioi) for x in iois) / (mean_ioi * len(iois) + 0.001)
    else:
        mean_ioi = beat; reg = 0.5

    ivs = [abs(bass[i+1].pitch - bass[i].pitch) for i in range(len(bass)-1)]
    mean_iv = sum(ivs) / len(ivs) if ivs else 0
    pitch_range = max(n.pitch for n in bass) - min(n.pitch for n in bass)

    # Classify
    if dom_ratio > 0.7 and pc_variety < 3:
        function = 'pedal'
        emotion = f"nota pedal de {NOTE_NAMES[dom_pc[0]]} — fundamento hipnótico, tensión estática"
    elif reg > 0.75 and mean_iv < 3:
        function = 'ostinato'
        emotion = 'ostinato rítmico — pulso mecánico, motor de tensión'
    elif mean_iv > 5 and pitch_range > 18 and reg > 0.5:
        function = 'walking bass'
        emotion = 'walking bass — impulso melódico hacia adelante, calidad jazzística'
    elif mean_iv > 4 and reg < 0.5:
        function = 'contrapunto'
        emotion = 'bajo contrapuntístico — voz independiente, diálogo con la melodía'
    elif reg > 0.6:
        function = 'acompañamiento rítmico'
        emotion = 'fundamento rítmico regular — soporte estable del discurso'
    else:
        function = 'mixto'
        emotion = 'bajo de función mixta — cambia de rol a lo largo de la pieza'

    return {
        'function': function,
        'dominant_pc': NOTE_NAMES[dom_pc[0]],
        'dom_ratio': dom_ratio,
        'pitch_range': pitch_range,
        'mean_interval': mean_iv,
        'regularity': reg,
        'note_count': len(bass),
        'emotion': emotion,
        'desc': f"Bajo {function}: {emotion}. Rango: {pitch_range}st, intervalo μ: {mean_iv:.1f}st.",
    }


# ═══════════════════════════════════════════════════════════════
#  12. SINCRONÍA ENTRE CAPAS
# ═══════════════════════════════════════════════════════════════

def analyze_layer_synchrony(notes, chords, bpm: float, total_dur: float) -> Dict:
    """
    Mide con qué frecuencia los cambios en melodía, bajo y armonía
    ocurren simultáneamente.
    """
    if not notes or bpm <= 0:
        return {'mean_synchrony': 0, 'sync_peaks': [], 'desc': ''}

    beat = 60.0 / bpm
    melody = [n for n in notes if n.pitch >= 65]
    bass   = [n for n in notes if n.pitch < 55]

    resolution = beat * 0.5
    t = resolution
    sync_scores = []

    while t < total_dur - resolution:
        events = []

        # Melodic change: new note in melody voice
        mel_now  = [n for n in melody if t - resolution <= n.time_sec < t]
        mel_prev = [n for n in melody if t - 2*resolution <= n.time_sec < t - resolution]
        if mel_now and mel_prev:
            p1 = sum(n.pitch for n in mel_now) / len(mel_now)
            p2 = sum(n.pitch for n in mel_prev) / len(mel_prev)
            if abs(p1 - p2) > 2: events.append('melody')

        # Bass change
        bas_now  = [n for n in bass if t - resolution <= n.time_sec < t]
        bas_prev = [n for n in bass if t - 2*resolution <= n.time_sec < t - resolution]
        if bas_now and bas_prev:
            pb1 = sum(n.pitch for n in bas_now) / len(bas_now)
            pb2 = sum(n.pitch for n in bas_prev) / len(bas_prev)
            if abs(pb1 - pb2) > 1: events.append('bass')

        # Harmonic change
        harm_now  = [c for c in chords if t - resolution <= c.time < t]
        harm_prev = [c for c in chords if t - 2*resolution <= c.time < t - resolution]
        if harm_now and harm_prev and harm_now[-1].name != harm_prev[-1].name:
            events.append('harmony')

        n_events = len(events)
        sync = n_events / 3.0
        sync_scores.append((t, sync, events))
        t += resolution

    if not sync_scores:
        return {'mean_synchrony': 0, 'sync_peaks': [], 'desc': ''}

    vals = [s for _, s, _ in sync_scores]
    mean_s = sum(vals) / len(vals)

    # Find sync peaks (all 3 layers change simultaneously)
    peaks = [(t, s, ev) for t, s, ev in sync_scores if s >= 0.99]
    deduped_peaks = []
    last_t = -999
    for t, s, ev in peaks:
        if t - last_t > beat * 2:
            deduped_peaks.append({'time': t, 'layers': ev})
            last_t = t

    if mean_s > 0.6:
        desc = f"alta sincronía entre capas ({mean_s:.2f}) — la música toma decisiones colectivas frecuentemente"
    elif mean_s > 0.35:
        desc = f"sincronía moderada ({mean_s:.2f}) — capas con cierta independencia"
    else:
        desc = f"baja sincronía ({mean_s:.2f}) — capas muy independientes, textura fluida y continua"

    if deduped_peaks:
        desc += f"; {len(deduped_peaks)} momento(s) de sincronía total"

    return {
        'mean_synchrony': mean_s,
        'sync_peaks': deduped_peaks[:6],
        'desc': desc,
    }


# ══ NARRATIVE ENGINE ══

MODE_CHARS = {
    'major':      ('Mayor',      'luminoso y abierto',         'afirmación, confianza, luz'),
    'minor':      ('Menor',      'oscuro e introspectivo',      'tristeza, drama, añoranza'),
    'dorian':     ('Dórico',     'oscuro con chispa de esperanza', 'melancolía activa, soul, anhelo'),
    'phrygian':   ('Frigio',     'tenso y fatalista',           'misterio, fatalismo, exotismo'),
    'lydian':     ('Lidio',      'flotante e irreal',           'ensueño, lo sublime, fantasía'),
    'mixolydian': ('Mixolidio',  'épico y ambiguo',             'heroísmo sin certeza, folk'),
    'locrian':    ('Locrio',     'inestable y disonante',       'ansiedad extrema, ruptura'),
}
TEMPO_WORDS = {
    (0,   60): ('muy lento', 'cada nota pesa, el tiempo se dilata'),
    (60,  80): ('lento',     'espacio para cada gesto, contemplación'),
    (80, 100): ('moderado',  'paso humano, naturalidad'),
    (100,120): ('animado',   'energía contenida, impulso hacia adelante'),
    (120,144): ('rápido',    'urgencia, presión, actividad'),
    (144,200): ('muy rápido','intensidad extrema, vértigo'),
}

def ts(sec):
    return f"{int(sec//60)}:{int(sec%60):02d}"

def tempo_desc(bpm):
    for (lo,hi),(word,qual) in TEMPO_WORDS.items():
        if lo <= bpm < hi: return word, qual
    return 'muy rápido', 'intensidad extrema'

def _pct(v, total, decimals=0):
    if total == 0: return '0%'
    fmt = f"{{:.{decimals}f}}%"
    return fmt.format(v/total*100)

def _or(val, default='—'):
    return val if val else default

def wrap_prose(text, width=72, indent=''):
    """Wraps text into readable paragraphs."""
    words = text.split()
    lines = []; buf = []
    for w in words:
        if sum(len(x)+1 for x in buf) + len(w) > width:
            lines.append(indent + ' '.join(buf)); buf = [w]
        else: buf.append(w)
    if buf: lines.append(indent + ' '.join(buf))
    return '\n'.join(lines)

# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 1: IDENTIDAD DE LA PIEZA
# ═══════════════════════════════════════════════════════════════

def write_identity(key_root, mode, kconf, avg_bpm, total_dur,
                   fingerprint, cultural, time_sig,
                   instruments, key_name,
                   genre_detection=None, semantic_enrichment=None) -> str:
    mode_label, mode_qual, mode_core = MODE_CHARS.get(mode, ('?','?','?'))
    t_word, t_qual = tempo_desc(avg_bpm)
    dur_min = int(total_dur//60); dur_sec = int(total_dur%60)
    ts_str = time_sig if time_sig else '4/4'

    inst_desc = ''
    if instruments:
        names = [name for _,(prog,name,emo) in instruments.items()]
        if names:
            inst_desc = f"escrita para {', '.join(names[:3])}"

    style_tags = (fingerprint or {}).get('style_tags', [])
    style_str = ''
    if style_tags:
        style_str = '; '.join(style_tags[:2])

    cult_dom = (cultural or {}).get('dominant_style', '')
    cult_markers = (cultural or {}).get('markers', [])

    gd = genre_detection or {}
    se = semantic_enrichment or {}

    lines = []
    lines.append("─" * 66)
    lines.append("  I.  IDENTIDAD")
    lines.append("─" * 66)
    lines.append("")

    p1 = (f"  {key_name} · {avg_bpm:.0f} BPM · {ts_str} · {dur_min}m{dur_sec:02d}s")
    lines.append(p1)
    lines.append("")

    p2 = (f"  La pieza se construye sobre el modo {mode_label} — {mode_qual} —, "
          f"un marco que predispone emocionalmente hacia {mode_core}. "
          f"La confianza en la detección de tonalidad es {kconf:.2f}/1.0"
          + (", alta" if kconf > 0.75 else ", moderada" if kconf > 0.55 else ", baja — la tonalidad es ambigua") + ".")
    lines.append(wrap_prose(p2, indent='  '))
    lines.append("")

    p3 = (f"  El tempo de {avg_bpm:.0f} BPM es {t_word}: {t_qual}. "
          f"En compás de {ts_str}, el pulso organiza el discurso "
          + ("con fluidez ternaria, propia de la danza y el lied." if '3' in ts_str
             else "con la cuadratura binaria habitual de la música occidental."))
    lines.append(wrap_prose(p3, indent='  '))
    lines.append("")

    # Género musical detectado
    if gd.get('genre_label') and gd.get('confidence', 0) > 0.3:
        subg_str = f" (subgénero: {gd['subgenre']})" if gd.get('subgenre') else ""
        conf_str = (f"con alta confianza ({gd['confidence']:.0%})"
                    if gd['confidence'] > 0.65 else
                    f"con confianza moderada ({gd['confidence']:.0%})"
                    if gd['confidence'] > 0.45 else
                    f"con confianza aproximada ({gd['confidence']:.0%})")
        p_genre = (f"  El análisis la clasifica como {gd['genre_label']}{subg_str} "
                   f"{conf_str}. {gd.get('key_signature','')}.")
        lines.append(wrap_prose(p_genre, indent='  '))
        lines.append("")

    # Concepto semántico (Affektenlehre)
    if se.get('concept') and se.get('fit_score', 0) > 0.35:
        p_sem = (f"  Desde la teoría del afecto, el perfil de la pieza converge con "
                 f"el concepto de '{se['concept']}': {se.get('justification','')}")
        lines.append(wrap_prose(p_sem, indent='  '))
        lines.append("")

    if inst_desc or style_str:
        p4 = "  " + '. '.join(filter(None, [
            inst_desc.capitalize() if inst_desc else '',
            f"El fingerprint armónico apunta hacia {style_str}" if style_str else '',
        ])) + '.'
        lines.append(wrap_prose(p4, indent='  '))

    if cult_markers:
        top = cult_markers[:2]
        refs = ' y '.join(m['emotion'].split(',')[0] for m in top)
        p5 = (f"  Los marcadores culturales predominantes evocan {refs}, "
              f"situando la pieza en un linaje estilístico reconocible "
              f"para el oído occidental contemporáneo.")
        lines.append(wrap_prose(p5, indent='  '))

    lines.append("")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 2: ARQUITECTURA TEMPORAL
# ═══════════════════════════════════════════════════════════════

def write_architecture(musical_form, golden, pnr, narrative_arc,
                       dyn_segments, sections, tension_curve,
                       total_dur, avg_bpm,
                       ssm=None, narrative_intention=None, catharsis=None) -> str:
    lines = []
    lines.append("─" * 66)
    lines.append("  II.  ARQUITECTURA TEMPORAL")
    lines.append("─" * 66)
    lines.append("")

    form_name = (musical_form or {}).get('form', 'indeterminada')
    form_desc = (musical_form or {}).get('desc', '')
    form_pattern = (musical_form or {}).get('pattern', '')
    n_segs = len(dyn_segments) if dyn_segments else 0

    # Forma musical — enriquecida con SSM
    ssm_form = (ssm or {}).get('form_canon', '')
    if ssm_form and ssm_form != form_name:
        p1 = (f"  La pieza adopta una forma {form_name}"
              + (f" (patrón {form_pattern})" if form_pattern else "") + ". "
              + form_desc + " "
              f"La Self-Similarity Matrix confirma y matiza: {ssm_form}.")
    else:
        p1 = (f"  La pieza adopta una forma {form_name}"
              + (f" (patrón {form_pattern})" if form_pattern else "") + ". "
              + form_desc)
    lines.append(wrap_prose(p1, indent='  '))
    lines.append("")

    # Repeticiones detectadas por SSM
    ssm_reps = (ssm or {}).get('repetitions', [])
    ssm_sym  = (ssm or {}).get('symmetry', 0.0)
    if ssm_reps:
        p_ssm = (f"  La matriz de auto-similitud detecta {len(ssm_reps)} par(es) de "
                 f"secciones prácticamente idénticas en contenido de alturas — "
                 f"la memoria temática de la pieza es {'fuerte' if len(ssm_reps) > 3 else 'selectiva'}. "
                 + (f"La simetría espejo es notable ({ssm_sym:.2f}/1.0): la segunda mitad "
                    f"recuerda a la primera como en un espejo." if ssm_sym > 0.65 else ""))
        lines.append(wrap_prose(p_ssm, indent='  '))
        lines.append("")
    elif ssm:
        n_unique = (ssm or {}).get('n_unique', 0)
        if n_unique > 1:
            p_ssm = (f"  Cada sección aporta material temático nuevo: "
                     f"{n_unique} células distintas, sin retornos literales al material previo. "
                     f"La pieza avanza linealmente, sin nostalgia de lo anterior.")
            lines.append(wrap_prose(p_ssm, indent='  '))
            lines.append("")

    if n_segs:
        p1b = (f"  La segmentación dinámica detecta {n_segs} secciones naturales, "
               f"definidas por cambios internos de densidad, registro y textura — "
               f"no por divisiones arbitrarias del cronómetro.")
        lines.append(wrap_prose(p1b, indent='  '))
        lines.append("")

    # Proporción áurea
    gold = golden or {}
    if gold.get('golden_climax'):
        gevts = ', '.join(gold.get('golden_events', []))
        p2 = (f"  Un hallazgo notable: {gevts} cae cerca de la proporción áurea "
              f"(φ = 0.618). La arquitectura temporal respeta —consciente o no— "
              f"una de las proporciones que el oído percibe como más naturalmente "
              f"satisfactorias.")
    else:
        prox = gold.get('proximity', {})
        if prox:
            first_prop = next(iter(prox))
            first_evts = prox[first_prop]
            p2 = (f"  Los puntos estructurales clave tienden hacia proporciones "
                  f"significativas: {first_evts[0][0]} ({first_evts[0][1]*100:.0f}%) "
                  f"se aproxima a {first_prop.lower()}.")
        else:
            p2 = ("  Las proporciones temporales no siguen patrones matemáticos "
                  "reconocibles — la arquitectura es orgánica, no calculada.")
    lines.append(wrap_prose(p2, indent='  '))
    lines.append("")

    # Arco narrativo
    na = narrative_arc or {}
    arc_type  = na.get('arc_type', '')
    arc_desc  = na.get('arc_desc', '')
    narrative = na.get('narrative', '')
    climax_t  = na.get('climax_time', 0)
    climax_pos= na.get('climax_position', 0)

    if arc_type:
        p3 = (f"  El arco narrativo es de tipo '{arc_type}': {arc_desc} "
              f"El clímax se sitúa en {ts(climax_t)} ({climax_pos*100:.0f}% "
              f"de la duración total). {narrative}.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Intención narrativa (nueva)
    ni = narrative_intention or {}
    ni_arch = ni.get('archetype', '')
    ni_conf = ni.get('confidence', 0)
    ni_time = ni.get('time_relationship', '')
    if ni_arch and ni_conf > 0.35:
        p_ni = (f"  La intención narrativa inferida corresponde al arquetipo '{ni_arch}' "
                f"(confianza {ni_conf:.0%}). {ni.get('composer_intent','')} "
                + (f"Relación con el tiempo: {ni_time}." if ni_time else ""))
        lines.append(wrap_prose(p_ni, indent='  '))
        lines.append("")

    # Catarsis (nueva)
    cat = catharsis or {}
    cat_type = cat.get('catharsis_type', 'ausente')
    if cat_type not in ('ausente', None):
        p_cat = (f"  En términos de catarsis emocional, la pieza ofrece una "
                 f"experiencia de tipo '{cat_type}'. {cat.get('emotional_effect','')}")
        lines.append(wrap_prose(p_cat, indent='  '))
        lines.append("")

    # Punto de no retorno
    pnr_data = pnr or {}
    if pnr_data.get('time') is not None:
        p4 = (f"  El punto de no retorno — el momento en que la tensión acumulada "
              f"hace imposible volver al estado inicial — se localiza en "
              f"{ts(pnr_data['time'])} ({pnr_data.get('position',0)*100:.1f}%). "
              f"A partir de ahí, la pieza solo puede avanzar hacia su desenlace.")
        lines.append(wrap_prose(p4, indent='  '))
        lines.append("")

    acts = na.get('acts', [])
    if acts:
        lines.append("  Estructura en actos:")
        for act in acts:
            lines.append(f"    [{ts(act['start'])}–{ts(act['end'])}]  "
                         f"{act['name']} — {act['role']}")
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 3: LENGUAJE ARMÓNICO
# ═══════════════════════════════════════════════════════════════

def write_harmony(chords, key_root, mode, canon_progs, modal_borrow,
                  cadences, harmonic_graph, tonal_gravity, tonal_ambiguity,
                  harmonic_color, harmonic_semantic, subtext,
                  voice_leading, pedal_points, elliptical,
                  chromaticism=None, dissonance_res=None) -> str:
    lines = []
    lines.append("─" * 66)
    lines.append("  III.  LENGUAJE ARMÓNICO")
    lines.append("─" * 66)
    lines.append("")

    # Overall harmonic character
    hg = harmonic_graph or {}
    entropy = hg.get('entropy', 0)
    center  = hg.get('center', '—')
    centricity_val = 0
    if chords:
        centricity_val = sum(1 for c in chords if c.root==key_root)/len(chords)

    if entropy > 0.85:
        harm_char = ("armonía tonalmente nómada, de alta entropía — "
                     "cada acorde abre nuevas posibilidades en lugar de confirmar las anteriores")
    elif entropy > 0.6:
        harm_char = "armonía variada pero con centro reconocible"
    else:
        harm_char = "armonía de movimiento acotado, repetición de gestos conocidos"

    p1 = (f"  El lenguaje armónico presenta {harm_char} "
          f"(entropía {entropy:.2f}/1.00). "
          f"El centro gravitacional detectado es {center}. ")
    if centricity_val < 0.1:
        p1 += ("La tónica aparece con extrema rareza, "
               "lo que genera una sensación de suspensión tonal permanente.")
    elif centricity_val > 0.4:
        p1 += ("El retorno frecuente a la tónica imprime "
               "al discurso un carácter afirmativo y anclado.")
    lines.append(wrap_prose(p1, indent='  '))
    lines.append("")

    # Cromatismo (nuevo)
    ch = chromaticism or {}
    ch_idx   = ch.get('chromaticism_index', 0)
    ch_desc  = ch.get('desc', '')
    ch_zones = ch.get('chromatic_zones', [])
    if ch_desc:
        p_chr = f"  El cromatismo: {ch_desc.lower()}"
        if ch_zones:
            zone_times = ', '.join(ts(z['start']) for z in ch_zones[:3])
            p_chr += (f" Las zonas de mayor densidad cromática se concentran en "
                      f"{zone_times} — momentos de máxima tensión armónica expresiva.")
        lines.append(wrap_prose(p_chr, indent='  '))
        lines.append("")

    # Canonical progressions and borrowings
    if canon_progs:
        progs_str = ', '.join(f"'{p}'" for p in canon_progs[:3])
        p2 = (f"  Se reconocen progresiones canónicas: {progs_str}. "
              f"Estas referencias a un vocabulario armónico compartido "
              f"activan la memoria cultural del oyente y anclan "
              f"la pieza en una tradición reconocible.")
        lines.append(wrap_prose(p2, indent='  '))
        lines.append("")

    if modal_borrow:
        borrow_str = ', '.join(modal_borrow[:3])
        p3 = (f"  La pieza recurre al préstamo modal ({borrow_str}): "
              f"acordes que pertenecen a escalas ajenas a la tonalidad principal. "
              f"Este recurso amplía el espectro emocional más allá "
              f"de lo estrictamente diatónico.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Cadences
    cad = cadences or {}
    cad_dom   = cad.get('dominant_type', '')
    cad_total = cad.get('total', 0)
    cad_desc  = cad.get('desc', '')
    if cad_total == 0:
        p4 = ("  Llamativamente, no se detectan cadencias convencionales. "
              "La pieza no busca resolución en ningún punto: "
              "vive en la pregunta, no en la respuesta. "
              "Este es uno de los mecanismos principales de su tensión sostenida.")
    else:
        p4 = (f"  El vocabulario cadencial ({cad_desc}) revela "
              f"la actitud de la pieza hacia la resolución. "
              + ({
                  'rota':     "Las cadencias rotas — que eluden el descanso esperado — dominan el discurso, generando giros emocionales constantes.",
                  'perfecta': "Las cadencias perfectas frecuentes otorgan a la pieza su sentido de certeza y dirección.",
                  'plagal':   "El predominio de cadencias plagales imprime una cualidad espiritual y de reposo sereno.",
                  'imperfecta': "Las cadencias imperfectas mantienen la pieza en movimiento perpetuo, sin cerrarse del todo.",
              }.get(cad_dom, "La diversidad cadencial refleja un discurso armónico matizado.")))
    lines.append(wrap_prose(p4, indent='  '))
    lines.append("")

    # Resolución de la disonancia (nuevo)
    dr = dissonance_res or {}
    dr_desc    = dr.get('desc', '')
    dr_ratio   = dr.get('resolution_ratio', 0)
    dr_pending = dr.get('unresolved_count', 0)
    if dr_desc:
        p_dr = f"  La gestión de la disonancia: {dr_desc.lower()}"
        if dr_pending > 3:
            p_dr += (f" Las {dr_pending} disonancias sin resolver acumulan "
                     f"una deuda armónica que la pieza nunca salda completamente — "
                     f"la tensión es el estado, no la excepción.")
        elif dr_ratio > 0.7:
            p_dr += (" La alta tasa de resolución describe una escritura que "
                     "genera tensión para liberarla: la disonancia es siempre transitoria.")
        lines.append(wrap_prose(p_dr, indent='  '))
        lines.append("")

    # Tonal gravity dynamics
    tg = tonal_gravity or {}
    n_mod = tg.get('n_modulations', 0)
    journey = tg.get('tonal_journey', '')
    if n_mod > 5:
        p5 = (f"  La gravedad tonal dinámica revela {n_mod} modulaciones "
              f"a lo largo de la pieza. La tonalidad no es un estado fijo "
              f"sino un territorio en movimiento: {journey}.")
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")
    elif n_mod > 0:
        p5 = (f"  La tonalidad es relativamente estable, con {n_mod} "
              f"desvío(s) tonal(es) que aportan color sin desestabilizar el centro.")
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")

    # Harmonic color
    hc = harmonic_color or {}
    dom_color = hc.get('dominant_color')
    if dom_color:
        palette = hc.get('palette', [])[:3]
        colors_str = ', '.join(f"{p['note']} ({p['color']})" for p in palette)
        p6 = (f"  Siguiendo la teoría del color armónico de Scriabin y "
              f"Rimsky-Korsakov, la paleta cromática de la pieza es "
              f"{colors_str}. El color dominante — "
              f"{dom_color['note']}, {dom_color['color']} — evoca "
              f"{dom_color['emotion']}.")
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    # Subtext
    sx = subtext or {}
    sx_idx = sx.get('subtext_index', 0)
    sx_desc = sx.get('desc', '')
    if sx_idx > 0.3:
        p7 = (f"  La pieza presenta subtext emocional significativo ({sx_desc}): "
              f"hay una brecha entre lo que la superficie musical parece decir "
              f"y lo que la armonía y el registro dicen en profundidad. "
              f"Es la música que 'suena de una manera pero se siente de otra'.")
        lines.append(wrap_prose(p7, indent='  '))
        lines.append("")
    elif sx_desc:
        lines.append(f"  {sx_desc.capitalize()}.")
        lines.append("")

    # Elliptical cadences
    el = elliptical or {}
    el_count = el.get('count', 0)
    if el_count > 0:
        p8 = (f"  Se detectan {el_count} elipsis cadenciales — momentos donde "
              f"la frase 'se va sin despedirse', continuando sin resolver "
              f"donde el oído esperaba un descanso. Cada una intensifica "
              f"la sensación de que algo permanece inconcluso.")
        lines.append(wrap_prose(p8, indent='  '))
        lines.append("")

    # Semantic density
    hs = harmonic_semantic or {}
    hs_desc = hs.get('desc', '')
    if hs_desc:
        lines.append(f"  {hs_desc.capitalize()}.")
        lines.append("")

    # Voice leading
    vl = voice_leading or {}
    vl_desc = vl.get('desc', '')
    if vl_desc:
        lines.append(f"  En cuanto al movimiento de voces: {vl_desc.lower()}")
        lines.append("")

    # Pedal points
    if pedal_points:
        for p in pedal_points[:2]:
            lines.append(f"  Se detecta un punto de pedal de {p['note']} "
                         f"en {ts(p['start'])}: {p['emotion']}.")
        lines.append("")

    return '\n'.join(lines)
    lines = []
    lines.append("─" * 66)
    lines.append("  III.  LENGUAJE ARMÓNICO")
    lines.append("─" * 66)
    lines.append("")

    # Overall harmonic character
    hg = harmonic_graph or {}
    entropy = hg.get('entropy', 0)
    center  = hg.get('center', '—')
    top_str = hg.get('top_str', '')
    centricity_val = 0
    if chords:
        centricity_val = sum(1 for c in chords if c.root==key_root)/len(chords)

    if entropy > 0.85:
        harm_char = ("armonía tonalmente nómada, de alta entropía — "
                     "cada acorde abre nuevas posibilidades en lugar de confirmar las anteriores")
    elif entropy > 0.6:
        harm_char = "armonía variada pero con centro reconocible"
    else:
        harm_char = "armonía de movimiento acotado, repetición de gestos conocidos"

    p1 = (f"  El lenguaje armónico presenta {harm_char} "
          f"(entropía {entropy:.2f}/1.00). "
          f"El centro gravitacional detectado es {center}. ")
    if centricity_val < 0.1:
        p1 += ("La tónica aparece con extrema rareza, "
               "lo que genera una sensación de suspensión tonal permanente.")
    elif centricity_val > 0.4:
        p1 += ("El retorno frecuente a la tónica imprime "
               "al discurso un carácter afirmativo y anclado.")
    lines.append(wrap_prose(p1, indent='  '))
    lines.append("")

    # Canonical progressions and borrowings
    if canon_progs:
        progs_str = ', '.join(f"'{p}'" for p in canon_progs[:3])
        p2 = (f"  Se reconocen progresiones canónicas: {progs_str}. "
              f"Estas referencias a un vocabulario armónico compartido "
              f"activan la memoria cultural del oyente y anclan "
              f"la pieza en una tradición reconocible.")
        lines.append(wrap_prose(p2, indent='  '))
        lines.append("")

    if modal_borrow:
        borrow_str = ', '.join(modal_borrow[:3])
        p3 = (f"  La pieza recurre al préstamo modal ({borrow_str}): "
              f"acordes que pertenecen a escalas ajenas a la tonalidad principal. "
              f"Este recurso amplía el espectro emocional más allá "
              f"de lo estrictamente diatónico.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Cadences
    cad = cadences or {}
    cad_dom = cad.get('dominant_type', '')
    cad_total = cad.get('total', 0)
    cad_desc = cad.get('desc', '')
    if cad_total == 0:
        p4 = ("  Llamativamente, no se detectan cadencias convencionales. "
              "La pieza no busca resolución en ningún punto: "
              "vive en la pregunta, no en la respuesta. "
              "Este es uno de los mecanismos principales de su tensión sostenida.")
    else:
        p4 = (f"  El vocabulario cadencial ({cad_desc}) revela "
              f"la actitud de la pieza hacia la resolución. "
              + ({
                  'rota':     "Las cadencias rotas — que eluden el descanso esperado — dominan el discurso, generando giros emocionales constantes.",
                  'perfecta': "Las cadencias perfectas frecuentes otorgan a la pieza su sentido de certeza y dirección.",
                  'plagal':   "El predominio de cadencias plagales imprime una cualidad espiritual y de reposo sereno.",
                  'imperfecta': "Las cadencias imperfectas mantienen la pieza en movimiento perpetuo, sin cerrarse del todo.",
              }.get(cad_dom, "La diversidad cadencial refleja un discurso armónico matizado.")))
    lines.append(wrap_prose(p4, indent='  '))
    lines.append("")

    # Tonal gravity dynamics
    tg = tonal_gravity or {}
    n_mod = tg.get('n_modulations', 0)
    journey = tg.get('tonal_journey', '')
    if n_mod > 5:
        p5 = (f"  La gravedad tonal dinámica revela {n_mod} modulaciones "
              f"a lo largo de la pieza. La tonalidad no es un estado fijo "
              f"sino un territorio en movimiento: {journey}.")
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")
    elif n_mod > 0:
        p5 = (f"  La tonalidad es relativamente estable, con {n_mod} "
              f"desvío(s) tonal(es) que aportan color sin desestabilizar el centro.")
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")

    # Harmonic color
    hc = harmonic_color or {}
    dom_color = hc.get('dominant_color')
    if dom_color:
        palette = hc.get('palette', [])[:3]
        colors_str = ', '.join(f"{p['note']} ({p['color']})" for p in palette)
        p6 = (f"  Siguiendo la teoría del color armónico de Scriabin y "
              f"Rimsky-Korsakov, la paleta cromática de la pieza es "
              f"{colors_str}. El color dominante — "
              f"{dom_color['note']}, {dom_color['color']} — evoca "
              f"{dom_color['emotion']}.")
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    # Subtext
    sx = subtext or {}
    sx_idx = sx.get('subtext_index', 0)
    sx_desc = sx.get('desc', '')
    if sx_idx > 0.3:
        p7 = (f"  La pieza presenta subtext emocional significativo ({sx_desc}): "
              f"hay una brecha entre lo que la superficie musical parece decir "
              f"y lo que la armonía y el registro dicen en profundidad. "
              f"Es la música que 'suena de una manera pero se siente de otra'.")
        lines.append(wrap_prose(p7, indent='  '))
        lines.append("")
    elif sx_desc:
        p7 = f"  {sx_desc.capitalize()}."
        lines.append(wrap_prose(p7, indent='  '))
        lines.append("")

    # Elliptical cadences
    el = elliptical or {}
    el_count = el.get('count', 0)
    if el_count > 0:
        p8 = (f"  Se detectan {el_count} elipsis cadenciales — momentos donde "
              f"la frase 'se va sin despedirse', continuando sin resolver "
              f"donde el oído esperaba un descanso. Cada una intensifica "
              f"la sensación de que algo permanece inconcluso.")
        lines.append(wrap_prose(p8, indent='  '))
        lines.append("")

    # Semantic density
    hs = harmonic_semantic or {}
    hs_desc = hs.get('desc', '')
    if hs_desc:
        lines.append(f"  {hs_desc.capitalize()}.")
        lines.append("")

    # Voice leading
    vl = voice_leading or {}
    vl_desc = vl.get('desc', '')
    if vl_desc:
        lines.append(f"  En cuanto al movimiento de voces: {vl_desc.lower()}")
        lines.append("")

    # Pedal points
    if pedal_points:
        for p in pedal_points[:2]:
            lines.append(f"  Se detecta un punto de pedal de {p['note']} "
                         f"en {ts(p['start'])}: {p['emotion']}.")
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 4: DISCURSO MELÓDICO
# ═══════════════════════════════════════════════════════════════

def write_melody(interval_anal, motifs, contour, phrasing,
                 thematic_transforms, thematic_recurrence,
                 expectation, narrative_voice, comfort_zones,
                 accomp, inner_voice,
                 counterpoint=None, story=None) -> str:
    lines = []
    lines.append("─" * 66)
    lines.append("  IV.  DISCURSO MELÓDICO")
    lines.append("─" * 66)
    lines.append("")

    # Interval character
    ia = interval_anal or {}
    ia_desc = ia.get('desc', '')
    conj = ia.get('conjunct', 0)
    leaps = ia.get('leaps', 0)
    asc = ia.get('ascending', 0)

    if ia_desc:
        p1 = f"  La melodía es {ia_desc}. "
        if asc > 0.55:
            p1 += ("La dirección predominantemente ascendente confiere a la pieza "
                   "un impulso de búsqueda, de tensión que no cede.")
        elif asc < 0.45:
            p1 += ("La tendencia descendente sugiere repliegue, "
                   "resignación o una resolución que se busca hacia abajo.")
        else:
            p1 += ("El equilibrio entre ascenso y descenso describe "
                   "una melodía que no tiene una dirección impuesta — "
                   "oscila, duda, busca.")
        lines.append(wrap_prose(p1, indent='  '))
        lines.append("")

    # Contour
    if contour and contour != 'indeterminado':
        p2 = (f"  El contorno melódico global describe una forma {contour}, "
              f"que anticipa la estructura emocional de la pieza antes "
              f"de que el análisis armónico o dinámico la confirme.")
        lines.append(wrap_prose(p2, indent='  '))
        lines.append("")

    # Phrasing
    ph = phrasing or {}
    ph_desc = ph.get('desc', '')
    ph_ac   = ph.get('antecedent_consequent', False)
    if ph_desc:
        p3 = f"  En términos de fraseo: {ph_desc.lower()}"
        if ph_ac:
            p3 += (" La estructura de pregunta-respuesta entre frases "
                   "establece un diálogo interno, como si la melodía "
                   "se interrogara a sí misma.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Motifs
    if motifs:
        best = motifs[0]
        p4 = (f"  El material temático se organiza en torno a {len(motifs)} "
              f"motivo(s) recurrente(s). El principal — {best['desc']} — "
              f"aparece {best['occ']} veces. ")
        tt = thematic_transforms or {}
        if tt.get('total_transforms', 0) > 0:
            p4 += (f"La pieza desarrolla ese material mediante "
                   f"{tt['total_transforms']} tipo(s) de transformación "
                   f"(inversión, retrogradación, aumentación), "
                   f"lo que convierte el motivo en un organismo vivo "
                   f"que evoluciona sin perder su identidad.")
        else:
            p4 += ("La repetición literal del motivo, sin transformación, "
                   "crea familiaridad e hipnosis en lugar de desarrollo.")
        lines.append(wrap_prose(p4, indent='  '))
        lines.append("")

    # Recurrence distances
    tr = thematic_recurrence or {}
    tr_desc = tr.get('desc', '')
    if tr_desc:
        lines.append(f"  {tr_desc.capitalize()}.")
        lines.append("")

    # Counterpoint (nuevo)
    cp = counterpoint or {}
    cp_desc     = cp.get('desc', '')
    cp_voices   = cp.get('independent_voices', 0)
    cp_parallel = cp.get('parallel_fifths', 0)
    cp_contrary = cp.get('contrary_motion_ratio', 0)
    if cp_desc:
        p_cp = f"  El contrapunto e independencia de voces: {cp_desc.lower()}"
        if cp_voices > 2:
            p_cp += (f" Las {cp_voices} voces independientes detectadas "
                     f"crean una textura de múltiples capas narrativas simultáneas.")
        if cp_parallel > 3:
            p_cp += (f" Las {cp_parallel} quintas paralelas no son errores "
                     f"sino un recurso expresivo deliberado — movimiento en bloque "
                     f"que sacrifica independencia por densidad de color.")
        lines.append(wrap_prose(p_cp, indent='  '))
        lines.append("")

    # Expectation
    exp = expectation or {}
    exp_mean = exp.get('mean_surprise', 0)
    exp_desc = exp.get('desc', '')
    if exp_desc:
        p5 = (f"  El modelo de expectativa melódica indica que "
              f"la melodía es {exp_desc.lower()} "
              f"(sorpresa media: {exp_mean:.3f}). "
              + ("Esta imprevisibilidad es un motor de tensión emocional constante." if exp_mean > 0.3
                 else "Esta predictibilidad crea confort y permite que el oyente anticipe y participe."))
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")

    # Narrative voice
    nv = narrative_voice or {}
    nv_desc   = nv.get('desc', '')
    nv_shifts = nv.get('protagonist_shifts', 0)
    if nv_desc:
        p6 = (f"  La voz conductora del discurso: {nv_desc.lower()} "
              + (f"Los {nv_shifts} cambio(s) de protagonista "
                 "crean la sensación de un diálogo entre voces distintas." if nv_shifts > 0 else ""))
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    # Comfort zones
    cz = comfort_zones or {}
    cz_desc = cz.get('desc', '')
    if cz_desc:
        lines.append(f"  Zona tonal: {cz_desc.lower()}.")
        lines.append("")

    # Accompaniment
    if accomp and accomp not in ('indeterminado', 'sin patrón claro'):
        lines.append(f"  El acompañamiento sigue un patrón de {accomp.lower()}.")
        lines.append("")

    # Inner voice
    iv = inner_voice or {}
    iv_desc = iv.get('desc', '')
    if iv_desc and iv.get('expressiveness', 0) > 0.3:
        lines.append(f"  {iv_desc.capitalize()}.")
        lines.append("")

    # Historia emocional (story) (nuevo)
    if story and len(story) > 20:
        # story es una narrativa ya formateada; incluimos solo el primer párrafo
        story_lines = [l.strip() for l in story.split('\n') if l.strip()]
        if story_lines:
            p_story = f"  Historia emocional: {story_lines[0]}"
            lines.append(wrap_prose(p_story, indent='  '))
            lines.append("")

    return '\n'.join(lines)
    lines = []
    lines.append("─" * 66)
    lines.append("  IV.  DISCURSO MELÓDICO")
    lines.append("─" * 66)
    lines.append("")

    # Interval character
    ia = interval_anal or {}
    ia_desc = ia.get('desc', '')
    conj = ia.get('conjunct', 0)
    leaps = ia.get('leaps', 0)
    asc = ia.get('ascending', 0)

    if ia_desc:
        p1 = f"  La melodía es {ia_desc}. "
        if asc > 0.55:
            p1 += ("La dirección predominantemente ascendente confiere a la pieza "
                   "un impulso de búsqueda, de tensión que no cede.")
        elif asc < 0.45:
            p1 += ("La tendencia descendente sugiere repliegue, "
                   "resignación o una resolución que se busca hacia abajo.")
        else:
            p1 += ("El equilibrio entre ascenso y descenso describe "
                   "una melodía que no tiene una dirección impuesta — "
                   "oscila, duda, busca.")
        lines.append(wrap_prose(p1, indent='  '))
        lines.append("")

    # Contour
    if contour and contour != 'indeterminado':
        p2 = (f"  El contorno melódico global describe una forma {contour}, "
              f"que anticipa la estructura emocional de la pieza antes "
              f"de que el análisis armónico o dinámico la confirme.")
        lines.append(wrap_prose(p2, indent='  '))
        lines.append("")

    # Phrasing
    ph = phrasing or {}
    ph_count = ph.get('count', 0)
    ph_len = ph.get('avg_length', 0)
    ph_sym = ph.get('symmetry', 0)
    ph_ac = ph.get('antecedent_consequent', False)
    ph_desc = ph.get('desc', '')

    if ph_desc:
        p3 = f"  En términos de fraseo: {ph_desc.lower()}"
        if ph_ac:
            p3 += (" La estructura de pregunta-respuesta entre frases "
                   "establece un diálogo interno, como si la melodía "
                   "se interrogara a sí misma.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Motifs
    if motifs:
        best = motifs[0]
        p4 = (f"  El material temático se organiza en torno a {len(motifs)} "
              f"motivo(s) recurrente(s). El principal — {best['desc']} — "
              f"aparece {best['occ']} veces. ")
        tt = thematic_transforms or {}
        if tt.get('total_transforms', 0) > 0:
            p4 += (f"La pieza desarrolla ese material mediante "
                   f"{tt['total_transforms']} tipo(s) de transformación "
                   f"(inversión, retrogradación, aumentación), "
                   f"lo que convierte el motivo en un organismo vivo "
                   f"que evoluciona sin perder su identidad.")
        else:
            p4 += ("La repetición literal del motivo, sin transformación, "
                   "crea familiaridad e hipnosis en lugar de desarrollo.")
        lines.append(wrap_prose(p4, indent='  '))
        lines.append("")

    # Recurrence distances
    tr = thematic_recurrence or {}
    tr_desc = tr.get('desc', '')
    if tr_desc:
        lines.append(f"  {tr_desc.capitalize()}.")
        lines.append("")

    # Expectation
    exp = expectation or {}
    exp_mean = exp.get('mean_surprise', 0)
    exp_desc = exp.get('desc', '')
    if exp_desc:
        p5 = (f"  El modelo de expectativa melódica indica que "
              f"la melodía es {exp_desc.lower()} "
              f"(sorpresa media: {exp_mean:.3f}). "
              + ("Esta imprevisibilidad es un motor de tensión emocional constante." if exp_mean > 0.3
                 else "Esta predictibilidad crea confort y permite que el oyente anticipe y participe."))
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")

    # Narrative voice
    nv = narrative_voice or {}
    nv_desc = nv.get('desc', '')
    nv_shifts = nv.get('protagonist_shifts', 0)
    if nv_desc:
        p6 = (f"  La voz conductora del discurso: {nv_desc.lower()} "
              + (f"Los {nv_shifts} cambio(s) de protagonista "
                 "crean la sensación de un diálogo entre voces distintas." if nv_shifts > 0 else ""))
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    # Comfort zones
    cz = comfort_zones or {}
    cz_desc = cz.get('desc', '')
    if cz_desc:
        lines.append(f"  Zona tonal: {cz_desc.lower()}.")
        lines.append("")

    # Accompaniment
    if accomp and accomp not in ('indeterminado', 'sin patrón claro'):
        lines.append(f"  El acompañamiento sigue un patrón de {accomp.lower()}.")
        lines.append("")

    # Inner voice
    iv = inner_voice or {}
    iv_desc = iv.get('desc', '')
    if iv_desc and iv.get('expressiveness', 0) > 0.3:
        lines.append(f"  {iv_desc.capitalize()}.")
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 5: DIMENSIÓN RÍTMICA
# ═══════════════════════════════════════════════════════════════

def write_rhythm(rhythm, micro_timing, metric_hierarchy, polyrhythm,
                 groove, bass_narrative, layer_sync, bpm, time_sig,
                 tempo_gestures=None, builds=None) -> str:
    lines = []
    lines.append("─" * 66)
    lines.append("  V.  DIMENSIÓN RÍTMICA")
    lines.append("─" * 66)
    lines.append("")

    # Rhythmic character
    rh = rhythm or {}
    rh_desc = rh.get('desc', '')
    sync    = rh.get('syncopation', 0)
    if rh_desc:
        lines.append(f"  {rh_desc.capitalize()}.")
        lines.append("")

    # Metric hierarchy
    mh = metric_hierarchy or {}
    mh_desc = mh.get('desc', '')
    if mh_desc:
        p2 = (f"  Desde la perspectiva de la jerarquía métrica (Lerdahl-Jackendoff): "
              f"{mh_desc.lower()}")
        lines.append(wrap_prose(p2, indent='  '))
        lines.append("")

    # Tempo gestures (nuevo)
    tg = tempo_gestures or []
    significant_tg = [g for g in tg if isinstance(g, dict) and
                      g.get('type') not in ('stable', 'neutro', None)]
    if significant_tg:
        tg_descs = []
        for g in significant_tg[:3]:
            gtype = g.get('type', '')
            gtime = g.get('time', 0)
            gmag  = g.get('magnitude', 0)
            if gtype and gtime is not None:
                tg_descs.append(f"{gtype} en {ts(gtime)}")
        if tg_descs:
            p_tg = (f"  Los gestos de tempo — {', '.join(tg_descs)} — "
                    f"no son meramente técnicos: cada cambio de velocidad "
                    f"es una declaración expresiva. El accelerando precipita; "
                    f"el ritardando suspende el tiempo.")
            lines.append(wrap_prose(p_tg, indent='  '))
            lines.append("")
    elif isinstance(tempo_gestures, list) and len(tempo_gestures) == 0:
        pass  # sin gestos detectados, silencio
    elif tempo_gestures and not significant_tg:
        lines.append("  El tempo permanece estable a lo largo de la pieza — "
                     "una decisión expresiva que niega el rubato y "
                     "prioriza el pulso como principio organizador.")
        lines.append("")

    # Micro-timing
    mt = micro_timing or {}
    mt_style = mt.get('style', '')
    mt_dev   = mt.get('mean_deviation_ms', 0)
    mt_hum   = mt.get('humanization', 0)
    mt_drag  = mt.get('drag_ratio', 0)
    mt_ant   = mt.get('anticipation_ratio', 0)
    if mt_style:
        p3 = (f"  El micro-timing revela que la pieza es {mt_style.lower()} "
              f"(desviación media del grid: {mt_dev:.1f}ms, "
              f"humanización: {mt_hum:.2f}/1.0). ")
        if mt_drag > mt_ant + 0.1:
            p3 += ("La tendencia al 'drag' — llegar ligeramente tarde al tiempo — "
                   "crea un feel relajado y laid-back.")
        elif mt_ant > mt_drag + 0.1:
            p3 += ("La tendencia a anticipar imprime urgencia "
                   "y energía hacia adelante.")
        else:
            p3 += ("El equilibrio entre anticipación y drag produce "
                   "esa sensación de flotación rítmica.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Polyrhythm
    pr = polyrhythm or {}
    pr_desc = pr.get('desc', '')
    if pr_desc and pr_desc != 'sin polirritmia — metro regular':
        lines.append(f"  {pr_desc.capitalize()}.")
        lines.append("")

    # Buildups (nuevo)
    blds = builds or []
    if blds:
        n_builds = len(blds)
        build_times = ', '.join(
            ts(b['start']) if isinstance(b, dict) and 'start' in b
            else ts(b) if isinstance(b, (int, float)) else '?'
            for b in blds[:3]
        )
        p_bld = (f"  Se detectan {n_builds} buildup(s) estructural(es) en {build_times}. "
                 f"Cada buildup es una promesa de clímax: la escritura "
                 f"acumula densidad, registro y dinámica hasta que la "
                 f"tensión se vuelve insostenible y debe resolverse o romperse.")
        lines.append(wrap_prose(p_bld, indent='  '))
        lines.append("")

    # Groove
    gr = groove or {}
    gr_desc = gr.get('desc', '')
    if gr_desc:
        lines.append(f"  {gr_desc.capitalize()}.")
        lines.append("")

    # Bass rhythm
    bn = bass_narrative or {}
    bn_func = bn.get('function', '')
    bn_desc = bn.get('desc', '')
    if bn_desc:
        p5 = (f"  El bajo, voz fundamental del ritmo armónico, "
              f"actúa como {bn_func}: {bn_desc.lower()}")
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")

    # Layer synchrony
    ls = layer_sync or {}
    ls_desc = ls.get('desc', '')
    if ls_desc:
        p6 = (f"  La sincronía entre capas es {ls_desc.lower()} "
              + ("Los momentos de sincronía total son los picos de impacto más precisos de la pieza." if ls.get('sync_peaks') else ""))
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    return '\n'.join(lines)
    lines = []
    lines.append("─" * 66)
    lines.append("  V.  DIMENSIÓN RÍTMICA")
    lines.append("─" * 66)
    lines.append("")

    # Rhythmic character
    rh = rhythm or {}
    rh_desc = rh.get('desc', '')
    sync = rh.get('syncopation', 0)
    swing = rh.get('swing', 1.0)
    variety = rh.get('variety', 0)

    if rh_desc:
        lines.append(f"  {rh_desc.capitalize()}.")
        lines.append("")

    # Metric hierarchy
    mh = metric_hierarchy or {}
    mh_desc = mh.get('desc', '')
    mh_reinf = mh.get('metric_reinforcement', 0.5)
    if mh_desc:
        p2 = (f"  Desde la perspectiva de la jerarquía métrica (Lerdahl-Jackendoff): "
              f"{mh_desc.lower()}")
        lines.append(wrap_prose(p2, indent='  '))
        lines.append("")

    # Micro-timing
    mt = micro_timing or {}
    mt_hum = mt.get('humanization', 0)
    mt_dev = mt.get('mean_deviation_ms', 0)
    mt_style = mt.get('style', '')
    mt_drag = mt.get('drag_ratio', 0)
    mt_ant  = mt.get('anticipation_ratio', 0)

    if mt_style:
        p3 = (f"  El micro-timing revela que la pieza es {mt_style.lower()} "
              f"(desviación media del grid: {mt_dev:.1f}ms, "
              f"humanización: {mt_hum:.2f}/1.0). ")
        if mt_drag > mt_ant + 0.1:
            p3 += ("La tendencia al 'drag' — llegar ligeramente tarde al tiempo — "
                   "crea un feel relajado y laid-back.")
        elif mt_ant > mt_drag + 0.1:
            p3 += ("La tendencia a anticipar imprime urgencia "
                   "y energía hacia adelante.")
        else:
            p3 += ("El equilibrio entre anticipación y drag produce "
                   "esa sensación de flotación rítmica.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Polyrhythm
    pr = polyrhythm or {}
    pr_desc = pr.get('desc', '')
    if pr_desc and pr_desc != 'sin polirritmia — metro regular':
        lines.append(f"  {pr_desc.capitalize()}.")
        lines.append("")

    # Groove
    gr = groove or {}
    gr_desc = gr.get('desc', '')
    gr_strength = gr.get('groove_strength', 0)
    if gr_desc:
        lines.append(f"  {gr_desc.capitalize()}.")
        lines.append("")

    # Bass rhythm
    bn = bass_narrative or {}
    bn_func = bn.get('function', '')
    bn_desc = bn.get('desc', '')
    if bn_desc:
        p5 = (f"  El bajo, voz fundamental del ritmo armónico, "
              f"actúa como {bn_func}: {bn_desc.lower()}")
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")

    # Layer synchrony
    ls = layer_sync or {}
    ls_desc = ls.get('desc', '')
    ls_mean = ls.get('mean_synchrony', 0)
    if ls_desc:
        p6 = (f"  La sincronía entre capas es {ls_desc.lower()} "
              + ("Los momentos de sincronía total son los picos de impacto más precisos de la pieza." if ls.get('sync_peaks') else ""))
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 6: EXPERIENCIA DEL OYENTE
# ═══════════════════════════════════════════════════════════════

def write_listener_experience(listener_model, cumulative_state, auditory_fatigue,
                               emotional_transitions, suspense, param_convergence,
                               tension_curve, energy_profile, silence_analysis,
                               eternity, total_dur,
                               dynamic_valence=None, emotional_ambivalence=None,
                               emotional_trajectory=None,
                               unified_emotional_map=None,
                               roughness=None, perceptual=None,
                               cinematic=None, climax_comparison=None,
                               momentum=None, polarity=None) -> str:
    lines = []
    lines.append("─" * 66)
    lines.append("  VI.  EXPERIENCIA DEL OYENTE")
    lines.append("─" * 66)
    lines.append("")

    # Valencia dinámica — visión continua (nueva)
    dv = dynamic_valence or {}
    dv_arc = dv.get('arc_shape', '')
    dv_arc_desc = dv.get('arc_desc', '')
    dv_mean = dv.get('mean_valence', 0.0)
    dv_vol  = dv.get('volatility', 0.0)
    dv_trans = dv.get('transitions', [])
    if dv_arc:
        light_dark = ('luminosa' if dv_mean > 0.15 else
                      'oscura' if dv_mean < -0.15 else 'ambivalente')
        p0 = (f"  La curva de valencia emocional describe una experiencia "
              f"predominantemente {light_dark} (media {dv_mean:+.2f}). "
              f"El arco sigue un patrón de '{dv_arc}': {dv_arc_desc}. "
              + (f"La volatilidad ({dv_vol:.2f}) es alta — la pieza oscila "
                 f"constantemente entre la luz y la sombra." if dv_vol > 0.35
                 else f"La estabilidad emocional ({1-dv_vol:.2f}) es notable — "
                      f"la pieza mantiene su color afectivo sin grandes sobresaltos."
                      if dv_vol < 0.15 else ""))
        if dv_trans:
            first = dv_trans[0]
            p0 += (f" El giro más brusco de valencia ocurre en "
                   f"{ts_str_local(first['time'])} "
                   f"({first['direction']}, Δ={first['delta']:+.2f}).")
        lines.append(wrap_prose(p0, indent='  '))
        lines.append("")

    # Fases del mapa unificado (nueva)
    uem = unified_emotional_map or {}
    uem_phases = uem.get('emotional_phases', [])
    uem_inflex  = uem.get('inflection_points', [])
    if uem_phases:
        dom_phase = max(uem_phases, key=lambda p: p['duration'])
        dom_pct   = dom_phase['duration'] / total_dur * 100 if total_dur > 0 else 0
        p_uem = (f"  El mapa emocional unificado — tensión, valencia, energía, "
                 f"aspereza y arousal sincronizados segundo a segundo — "
                 f"revela {len(uem_phases)} fase(s) emocional(es) diferenciada(s). "
                 f"La fase dominante es '{dom_phase['label']}' "
                 f"({dom_pct:.0f}% del tiempo). ")
        if uem_inflex:
            inflex_times = ', '.join(ip['time_str'] for ip in uem_inflex[:3])
            p_uem += (f"Los giros emocionales más bruscos se producen "
                      f"en {inflex_times}.")
        lines.append(wrap_prose(p_uem, indent='  '))
        lines.append("")

    # Ambivalencia emocional (nueva)
    amb = emotional_ambivalence or {}
    amb_profile = amb.get('profile', '')
    amb_peaks   = amb.get('peak_moments', [])
    if amb_profile and amb.get('mean_ambivalence', 0) > 0.2:
        p_amb = (f"  La ambivalencia emocional — momentos en que distintas "
                 f"dimensiones expresivas apuntan en direcciones opuestas — "
                 f"es {amb_profile.lower()}. "
                 + (f"El momento de máximo conflicto interno ocurre en "
                    f"{ts_str_local(amb_peaks[0]['time'])}: "
                    f"{amb_peaks[0]['desc']}." if amb_peaks else ""))
        lines.append(wrap_prose(p_amb, indent='  '))
        lines.append("")

    # Trayectoria VA (nueva)
    traj = emotional_trajectory or {}
    traj_shape = traj.get('path_shape', '')
    traj_desc  = traj.get('path_shape_desc', '')
    traj_dist  = traj.get('total_distance', 0.0)
    traj_eff   = traj.get('efficiency', 1.0)
    if traj_shape and traj_dist > 0:
        p_traj = (f"  En el espacio Valence-Arousal, la pieza describe una "
                  f"trayectoria {traj_shape}: {traj_desc}. "
                  f"Distancia total recorrida: {traj_dist:.2f} unidades "
                  f"con eficiencia {traj_eff:.2f} "
                  + ("— avanza casi en línea recta hacia su destino emocional."
                     if traj_eff > 0.75
                     else "— la pieza regresa cerca de donde empezó."
                     if traj_eff < 0.2
                     else "— un camino con rodeos y retornos parciales."))
        lines.append(wrap_prose(p_traj, indent='  '))
        lines.append("")

    # Listener model
    lm = listener_model or {}
    lm_shape = lm.get('shape', '')
    lm_peak  = lm.get('peak_time', 0)
    lm_peak_v= lm.get('peak_experience', 0)
    lm_exhaust = lm.get('exhaustion_index', 0)
    if lm_shape:
        p1 = (f"  La experiencia subjetiva proyectada del oyente es de tipo "
              f"'{lm_shape}'. El pico de máxima implicación emocional se "
              f"alcanza en {ts(lm_peak)} (intensidad {lm_peak_v:.2f}/1.0). ")
        if lm_exhaust > 0.15:
            p1 += (f"El índice de agotamiento auditivo ({lm_exhaust:.2f}) "
                   f"indica que la pieza exige atención sostenida — "
                   f"el oyente llega al final algo exhausto.")
        else:
            p1 += ("El bajo índice de agotamiento indica que la pieza "
                   "mantiene frescura sin agotar la atención.")
        lines.append(wrap_prose(p1, indent='  '))
        lines.append("")

    # Cumulative emotional state
    cs = cumulative_state or {}
    cs_journey = cs.get('journey_type', '')
    fv, fa = cs.get('final_state', (0, 0))
    if cs_journey:
        p2 = (f"  El estado emocional acumulativo sigue un viaje "
              f"'{cs_journey}'. "
              f"El estado final del oyente — valencia {fv:+.2f}, "
              f"activación {fa:+.2f} — describe "
              + ("un estado de energía positiva residual." if fv > 0.1 and fa > 0.1
                 else "una calma melancólica." if fv < -0.1 and fa < 0
                 else "una activación sin resolución clara." if fa > 0.2
                 else "un estado de serenidad ambigua."))
        lines.append(wrap_prose(p2, indent='  '))
        lines.append("")

    # Emotional transitions
    et = emotional_transitions or {}
    et_smooth = et.get('smoothness', 1.0)
    et_journey = et.get('journey', '')
    if et_journey:
        p3 = (f"  El viaje emocional entre secciones es "
              f"{'muy fluido' if et_smooth > 0.8 else 'accidentado con giros bruscos' if et_smooth < 0.4 else 'moderadamente fluido'} "
              f"(suavidad {et_smooth:.2f}/1.0). "
              f"{et_journey.capitalize()}.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Suspense
    sp = suspense or {}
    sp_desc = sp.get('desc', '')
    sp_false = sp.get('false_resolutions', 0)
    if sp_desc:
        p4 = f"  En cuanto al suspense: {sp_desc.lower()}"
        if sp_false > 2:
            p4 += (f" Las {sp_false} falsas resoluciones son momentos donde "
                   f"el oído cree que la tensión se liberará — y no lo hace. "
                   f"Cada una renueva la deuda emocional.")
        lines.append(wrap_prose(p4, indent='  '))
        lines.append("")

    # Convergence peaks
    pc3 = param_convergence or {}
    pc_peaks = pc3.get('peaks', [])
    if pc_peaks:
        crescendo_peaks = [p for p in pc_peaks if p['direction'] == 'crescendo']
        if crescendo_peaks:
            times_str = ', '.join(ts(p['time']) for p in crescendo_peaks[:3])
            p5 = (f"  Los momentos de mayor impacto emocional — donde melodía, "
                  f"dinámica, densidad y tensión convergen simultáneamente — "
                  f"ocurren en {times_str}. En esos instantes, "
                  f"todos los parámetros dicen lo mismo a la vez.")
            lines.append(wrap_prose(p5, indent='  '))
            lines.append("")

    # Silence
    sa = silence_analysis or {}
    sa_ratio = sa.get('ratio', 0)
    md_sil = sa.get('most_dramatic')
    if sa_ratio < 0.05:
        p6 = ("  La densísima escritura, casi sin respiros, ejerce una "
              "presión psicológica constante. El silencio, cuando "
              "aparece, resulta extraordinariamente expresivo precisamente "
              "por su rareza.")
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")
    elif sa_ratio > 0.25 and md_sil:
        p6 = (f"  Los silencios prolongados — el más dramático en "
              f"{ts(md_sil['start'])} ({md_sil['duration']:.1f}s) — "
              f"son tan compositivos como las notas. El vacío habla.")
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    # Eternity moments
    et2 = eternity or {}
    et2_moments = et2.get('moments', [])
    et2_count = et2.get('count', 0)
    if et2_count > 0:
        p7 = (f"  Se identifican {et2_count} 'momentos de eternidad' — "
              f"instantes donde el tiempo subjetivo se dilata: notas "
              f"sostenidas, silencios post-clímax, convergencias de "
              f"reposo total. El más notable: {et2_moments[0]['desc'].lower()} "
              f"en {ts(et2_moments[0]['time'])}.")
        lines.append(wrap_prose(p7, indent='  '))
        lines.append("")

    # Roughness sensorial (nuevo)
    rgh = roughness or {}
    rgh_mean = rgh.get('mean_roughness', 0)
    rgh_arc  = rgh.get('roughness_arc_desc', '')
    rgh_reg  = rgh.get('register_impact', 0)
    rgh_pk   = rgh.get('peak_roughness', {})
    if rgh_arc:
        quality = ('extremadamente áspera — la tensión física del sonido es "dura" al oído'
                   if rgh_mean > 1.0 else
                   'áspera — disonancia sensorial sostenida que el oído percibe como fricción'
                   if rgh_mean > 0.6 else
                   'moderadamente áspera — equilibrio entre aspereza y suavidad'
                   if rgh_mean > 0.3 else
                   'sensualmente suave — textura acústica consonante y limpia')
        p_rgh = (f"  La textura psicoacústica es {quality} "
                 f"(roughness media {rgh_mean:.3f}). {rgh_arc}.")
        if rgh_reg > 0.1:
            p_rgh += (f" El registro grave amplifica la aspereza en un "
                      f"{rgh_reg*100:.0f}% — en los bajos, los mismos intervalos "
                      f"suenan más tensos que en el agudo.")
        if rgh_pk.get('time') is not None and rgh_pk.get('value', 0) > rgh_mean * 1.5:
            p_rgh += (f" El momento de máxima aspereza sensorial ocurre en "
                      f"{ts(rgh_pk['time'])}.")
        lines.append(wrap_prose(p_rgh, indent='  '))
        lines.append("")

    # Intensidad perceptual (nuevo)
    pc = perceptual or {}
    pc_mean = pc.get('mean', 0)
    pc_desc = pc.get('desc', '')
    pc_peak = pc.get('peak_time', 0)
    if pc_desc:
        p_pc = (f"  La intensidad perceptual ponderada — que combina densidad, "
                f"velocidad y registro en una única curva de impacto — "
                f"es {pc_desc.lower()}. "
                f"El instante de máximo impacto perceptual se localiza en "
                f"{ts(pc_peak)}.")
        lines.append(wrap_prose(p_pc, indent='  '))
        lines.append("")

    # Polaridad emocional (nuevo)
    pol = polarity or {}
    pol_type = pol.get('polarity_type', '')
    pol_desc = pol.get('desc', '')
    if pol_desc and pol_type:
        p_pol = (f"  La polaridad emocional de la pieza es '{pol_type}': "
                 f"{pol_desc.lower()}")
        lines.append(wrap_prose(p_pol, indent='  '))
        lines.append("")

    # Momentum (nuevo)
    mom = momentum or {}
    mom_desc  = mom.get('desc', '')
    mom_trend = mom.get('trend', '')
    mom_peak  = mom.get('peak_time', 0)
    if mom_desc:
        p_mom = (f"  El impulso musical acumulado: {mom_desc.lower()} "
                 + (f"El pico de impulso máximo se alcanza en {ts(mom_peak)}." if mom_peak else ""))
        lines.append(wrap_prose(p_mom, indent='  '))
        lines.append("")

    # Clímax emocional vs. sonoro (nuevo)
    cc = climax_comparison or {}
    cc_desc = cc.get('desc', '')
    cc_emo  = cc.get('emotional_climax_time', None)
    cc_son  = cc.get('sonic_climax_time', None)
    if cc_desc:
        p_cc = f"  Comparación clímax emocional vs. sonoro: {cc_desc.lower()}"
        if cc_emo is not None and cc_son is not None:
            diff = abs(cc_emo - cc_son)
            if diff > 5:
                p_cc += (f" El clímax emocional ({ts(cc_emo)}) y el sonoro "
                         f"({ts(cc_son)}) están separados {diff:.0f}s — "
                         f"la pieza disocia deliberadamente el pico de tensión "
                         f"del pico de energía física.")
            else:
                p_cc += (f" Clímax emocional y sonoro coinciden en {ts(cc_emo)} — "
                         f"máximo impacto simultáneo en todos los niveles.")
        lines.append(wrap_prose(p_cc, indent='  '))
        lines.append("")

    # Tropos cinematográficos (nuevo)
    ci = cinematic or {}
    ci_tropes = ci.get('tropes', {})
    ci_desc   = ci.get('desc', '')
    if ci_tropes:
        trope_names = list(ci_tropes.keys())[:3]
        p_ci = (f"  La pieza activa {len(ci_tropes)} tropo(s) cinematográfico(s) "
                f"reconocibles: {', '.join(trope_names)}. "
                f"Estos patrones — heredados del lenguaje musical del cine — "
                f"disparan respuestas emocionales condicionadas en el oyente "
                f"contemporáneo con la fuerza de un reflejo cultural.")
        lines.append(wrap_prose(p_ci, indent='  '))
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN 7: INTENCIÓN COMPOSITIVA
# ═══════════════════════════════════════════════════════════════

def write_compositional_intention(fractal, emotional_weight, golden,
                                   emotional_coherence, tonal_ambiguity,
                                   harmonic_semantic, narrative_arc,
                                   microexpression, fingerprint,
                                   motifs, thematic_transforms,
                                   mode, avg_bpm,
                                   anti_conventional=None,
                                   multilevel_density=None,
                                   melodic_markov=None,
                                   semantic_enrichment=None) -> str:
    lines = []
    lines.append("─" * 66)
    lines.append("  VII.  INTENCIÓN COMPOSITIVA")
    lines.append("─" * 66)
    lines.append("")

    # Voz compositiva — elecciones anti-convencionales (nueva)
    ac = anti_conventional or {}
    ac_score = ac.get('deviation_score', 0.0)
    ac_profile = ac.get('voice_profile', '')
    ac_sigs = ac.get('signature_choices', [])
    ac_devs = ac.get('deviations', [])
    if ac_profile:
        p_ac = (f"  La voz compositiva es {ac_profile} "
                f"(índice de originalidad: {ac_score:.2f}/1.0). ")
        if ac_sigs:
            p_ac += (f"Las elecciones más características: "
                     f"{'; '.join(ac_sigs[:3])}.")
        elif ac_score < 0.2:
            p_ac += ("La escritura sigue con fidelidad los patrones estadísticos "
                     "esperados para su modo y contexto — convencionalidad deliberada "
                     "o dominio de la tradición.")
        lines.append(wrap_prose(p_ac, indent='  '))
        # Describir las desviaciones más significativas
        for dev in ac_devs[:2]:
            lines.append(wrap_prose(f"  · [{dev['type']}] {dev['desc']}", indent='    '))
        lines.append("")

    # Densidad informacional multinivel (nueva)
    mld = multilevel_density or {}
    mld_profile = mld.get('density_profile', '')
    mld_profile_desc = mld.get('profile_desc', '')
    mld_economy = mld.get('composer_economy', 0.5)
    if mld_profile:
        p_mld = (f"  La organización de la información musical revela un perfil "
                 f"'{mld_profile}': {mld_profile_desc}. "
                 f"La economía compositiva es {mld_economy:.2f}/1.0 — "
                 + ("el compositor hace mucho con poco: cada gesto carga "
                    "con el peso de la forma." if mld_economy > 0.65
                    else "la riqueza de detalle en todos los niveles crea "
                         "una textura densa y elaborada." if mld_economy < 0.35
                    else "un equilibrio entre economía y ornamentación."))
        lines.append(wrap_prose(p_mld, indent='  '))
        lines.append("")

    # Predecibilidad melódica — Markov (nueva)
    mm = melodic_markov or {}
    mm_style = mm.get('style_profile', '')
    mm_prob  = mm.get('mean_probability', 0.5)
    mm_ent   = mm.get('entropy_markov', 0.0)
    if mm_style and mm_style != 'indeterminado':
        p_mm = (f"  El análisis markoviano de la melodía revela un perfil "
                f"'{mm_style}' (probabilidad media de transición: {mm_prob:.2f}). "
                + ("Cada nota fluye de forma estadísticamente natural desde las anteriores — "
                   "la melodía tiene coherencia interna máxima." if mm_style == 'predecible'
                   else "La melodía desafía frecuentemente las expectativas estadísticas "
                        "de su propio vocabulario — gestos únicos e inesperados." if mm_style in ('sorpresivo','caótico')
                   else "La melodía combina con equilibrio gestos esperados y sorpresas."))
        lines.append(wrap_prose(p_mm, indent='  '))
        lines.append("")

    # Persona teórica (nueva — desde semantic enrichment)
    se = semantic_enrichment or {}
    se_persona = se.get('persona_name', '')
    se_lens    = se.get('persona_lens', '')
    if se_persona and se_lens:
        p_pers = (f"  Desde la perspectiva de un {se_persona.lower()}: "
                  f"{se_lens}")
        lines.append(wrap_prose(p_pers, indent='  '))
        lines.append("")

    # Fractal structure
    fr = fractal or {}
    fi = fr.get('fractal_index', 0)
    fr_desc = fr.get('desc', '')
    if fr_desc:
        p1 = (f"  La coherencia estructural transescala revela {fr_desc.lower()}. "
              + ("Esto implica que los gestos locales y el arco global "
                 "están construidos desde el mismo material generativo — "
                 "la pieza es fractal, se contiene a sí misma." if fi > 0.75
                 else "La estructura a gran escala no replica la pequeña escala — "
                      "el todo y las partes tienen lógicas propias."))
        lines.append(wrap_prose(p1, indent='  '))
        lines.append("")

    # Motivic coherence
    if motifs:
        tt = thematic_transforms or {}
        tt_rich = tt.get('richness', 0)
        tt_total = tt.get('total_transforms', 0)
        n_motifs = len(motifs)
        if tt_total > 0:
            p2 = (f"  El material motívico ({n_motifs} motivo(s)) se desarrolla "
                  f"mediante {tt_total} tipo(s) de transformación, "
                  f"con {'alta' if tt_rich >= 2 else 'moderada' if tt_rich == 1 else 'baja'} "
                  f"riqueza transformativa. "
                  + ("La escritura demuestra dominio técnico del desarrollo temático." if tt_rich >= 2
                     else "La transformación es puntual, no sistemática."))
            lines.append(wrap_prose(p2, indent='  '))
            lines.append("")

    # Emotional coherence
    ec = emotional_coherence or {}
    ec_coh = ec.get('coherence', 1.0)
    ec_desc = ec.get('desc', '')
    ec_contra = ec.get('contradictions', [])
    if ec_desc:
        p3 = f"  Coherencia emocional: {ec_desc.lower()}"
        if ec_contra:
            types = [c['type'] for c in ec_contra[:2]]
            p3 += (f" Las contradicciones detectadas ({', '.join(types)}) "
                   + ("son recursos expresivos deliberados que crean ambivalencia." if ec_coh > 0.5
                      else "sugieren una escritura que trabaja con la tensión entre opuestos."))
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Tonal ambiguity
    ta = tonal_ambiguity or {}
    ta_idx = ta.get('ambiguity_index', 0)
    ta_desc = ta.get('desc', '')
    if ta_idx > 0.3:
        p4 = (f"  La ambigüedad tonal intencional ({ta_desc.lower()}) "
              f"revela una escritura que no quiere ser 'leída' de una sola manera. "
              f"El compositor ofrece múltiples lecturas tonales simultáneas.")
        lines.append(wrap_prose(p4, indent='  '))
        lines.append("")
    elif ta_desc:
        lines.append(f"  {ta_desc.capitalize()}.")
        lines.append("")

    # Emotional weight
    ew = emotional_weight or {}
    ew_class = ew.get('intensity_class', '')
    ew_wpm = ew.get('weight_per_minute', 0)
    if ew_class:
        p5 = (f"  El peso emocional total de la pieza es {ew_class} "
              f"({ew_wpm:.1f} unidades/min). "
              + ("Una pieza densa emocionalmente es exigente — "
                 "pide atención total y no da descanso." if ew_class in ('intensa','extrema')
                 else "Una pieza de peso moderado permite al oyente respirar "
                      "entre los momentos de mayor intensidad."))
        lines.append(wrap_prose(p5, indent='  '))
        lines.append("")

    # Microexpression
    me = microexpression or {}
    me_expr = me.get('expressiveness', 0)
    me_hairpins = me.get('hairpin_count', 0)
    if me_expr > 0 or me_hairpins > 0:
        p6 = (f"  Las microdinámicas expresivas — variación de velocidad nota a nota — "
              f"muestran una escritura {'altamente expresiva' if me_expr > 0.6 else 'moderadamente dinámica' if me_expr > 0.3 else 'uniforme'}. "
              + (f"Los {me_hairpins} gestos de crescendo/decrescendo detectados "
                 f"son la huella de la expresividad interpretativa." if me_hairpins > 0 else ""))
        lines.append(wrap_prose(p6, indent='  '))
        lines.append("")

    # Fingerprint
    fp = fingerprint or {}
    fp_tags = fp.get('style_tags', [])
    if fp_tags:
        p7 = (f"  La huella armónica del compositor apunta hacia: "
              f"{'; '.join(fp_tags[:3])}. "
              f"Esta combinación define un estilo reconocible.")
        lines.append(wrap_prose(p7, indent='  '))
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN VIII: EVOLUCIÓN EMOCIONAL A LO LARGO DEL TIEMPO
# ═══════════════════════════════════════════════════════════════

def write_emotional_evolution(
    unified_emotional_map=None,
    dynamic_valence=None,
    emotional_ambivalence=None,
    emotional_trajectory=None,
    catharsis=None,
    polarity=None,
    momentum=None,
    total_dur=0,
) -> str:
    """
    Sección VIII del ensayo: evolución emocional continua.
    Usa los datos del mapa unificado y las curvas temporales
    para construir una narrativa de lo que ocurre segundo a segundo.
    """
    lines = []
    lines.append("─" * 66)
    lines.append("  VIII.  EVOLUCIÓN EMOCIONAL A LO LARGO DEL TIEMPO")
    lines.append("─" * 66)
    lines.append("")

    uem = unified_emotional_map or {}
    dv  = dynamic_valence or {}
    amb = emotional_ambivalence or {}
    traj = emotional_trajectory or {}
    cat  = catharsis or {}
    pol  = polarity or {}
    mom  = momentum or {}

    phases    = uem.get('emotional_phases', [])
    inflexions = uem.get('inflection_points', [])
    uem_coh   = uem.get('mean_coherence', 0.0)

    # ── Párrafo de apertura: panorama global ──────────────────
    dv_arc   = dv.get('arc_shape', '')
    dv_mean  = dv.get('mean_valence', 0.0)
    dv_vol   = dv.get('volatility', 0.0)
    light_dark = ('luminosa' if dv_mean > 0.15 else
                  'oscura'   if dv_mean < -0.15 else 'emocionalmente ambivalente')

    if phases:
        dom_phase = max(phases, key=lambda p: p['duration'])
        dom_pct   = dom_phase['duration'] / total_dur * 100 if total_dur > 0 else 0
        n_phases  = len(phases)
        p_open = (
            f"  La pieza atraviesa {n_phases} fase(s) emocional(es) diferenciada(s) "
            f"a lo largo de sus {int(total_dur//60)}m{int(total_dur%60):02d}s. "
            f"El estado dominante es '{dom_phase['label']}' ({dom_pct:.0f}% del tiempo), "
            f"lo que convierte la pieza en una experiencia {light_dark} en términos netos."
        )
        lines.append(wrap_prose(p_open, indent='  '))
        lines.append("")

    # ── Descripción de fases ──────────────────────────────────
    if phases:
        lines.append("  Recorrido emocional por fases:")
        for i, ph in enumerate(phases):
            dur_pct = ph['duration'] / total_dur * 100 if total_dur > 0 else 0
            means   = ph.get('means', {})
            t_str   = f"{ph['start_str']} → {ph['end_str']}"
            # Build descriptive sentence for this phase
            t_val = means.get('tension', 0)
            v_val = means.get('valence_n', 0.5)
            e_val = means.get('energy', 0)
            quality_parts = []
            if t_val > 0.6:   quality_parts.append('alta tensión')
            elif t_val < 0.3: quality_parts.append('tensión baja')
            if v_val > 0.65:  quality_parts.append('color luminoso')
            elif v_val < 0.4: quality_parts.append('color oscuro')
            if e_val > 0.6:   quality_parts.append('energía alta')
            elif e_val < 0.25: quality_parts.append('quietud')
            quality = ', '.join(quality_parts) if quality_parts else 'equilibrio emocional'
            lines.append(
                f"    [{i+1}] {t_str}  ({dur_pct:.0f}%)  "
                f"'{ph['label']}' — {quality}"
            )
        lines.append("")

    # ── Giros emocionales bruscos ─────────────────────────────
    if inflexions:
        n_inf = len(inflexions)
        p_inf = (
            f"  {'Un' if n_inf==1 else str(n_inf)} giro{'s' if n_inf>1 else ''} "
            f"emocional{'es' if n_inf>1 else ''} brusco{'s' if n_inf>1 else ''} "
            f"quiebra{'n' if n_inf>1 else ''} la continuidad de la pieza. "
        )
        top = inflexions[:3]
        descs = [f"{ip['time_str']} ({ip['desc']})" for ip in top]
        p_inf += f"{'El más' if n_inf==1 else 'Los más'} significativo{'s' if n_inf>1 else ''}: {'; '.join(descs)}."
        lines.append(wrap_prose(p_inf, indent='  '))
        lines.append("")

    # ── Ambivalencia emocional ────────────────────────────────
    amb_mean = amb.get('mean_ambivalence', 0.0)
    amb_profile = amb.get('profile', '')
    amb_peaks   = amb.get('peak_moments', [])
    if amb_mean > 0.2 and amb_profile:
        p_amb = (
            f"  A nivel de ambivalencia — momentos en que el modo, la dinámica, "
            f"el registro y la tensión apuntan en direcciones opuestas simultáneamente — "
            f"la pieza es {amb_profile.lower()}. "
        )
        if amb_peaks:
            pk = amb_peaks[0]
            p_amb += (
                f"El punto de mayor conflicto interno se produce en "
                f"{ts_str_local(pk['time'])}: {pk['desc']}. "
                f"En ese instante, la pieza no tiene un solo color emocional "
                f"sino varios a la vez, irreconciliables."
            )
        lines.append(wrap_prose(p_amb, indent='  '))
        lines.append("")

    # ── Trayectoria VA ────────────────────────────────────────
    traj_shape = traj.get('path_shape', '')
    traj_desc  = traj.get('path_shape_desc', '')
    traj_dist  = traj.get('total_distance', 0.0)
    traj_eff   = traj.get('efficiency', 1.0)
    dom_quad   = traj.get('dominant_quadrant', '')
    cx, cy     = traj.get('centroid', (0.0, 0.0))
    if traj_shape and traj_dist > 0:
        p_traj = (
            f"  Trazada como ruta en el espacio Valence-Arousal, la pieza "
            f"describe un camino {traj_shape} ({traj_desc}). "
            f"Recorre {traj_dist:.2f} unidades emocionales con una eficiencia de "
            f"{traj_eff:.2f} — "
        )
        if traj_eff > 0.75:
            p_traj += "avanza en línea casi recta hacia su destino emocional, sin rodeos."
        elif traj_eff < 0.2:
            p_traj += "regresa casi exactamente al estado de inicio, como si nada hubiese ocurrido."
        else:
            p_traj += "un camino con desvíos y retornos parciales."
        if dom_quad:
            p_traj += f" El centroide emocional ({cx:+.2f}, {cy:+.2f}) pertenece al cuadrante '{dom_quad}'."
        lines.append(wrap_prose(p_traj, indent='  '))
        lines.append("")

    # ── Catarsis ──────────────────────────────────────────────
    cat_type   = cat.get('catharsis_type', 'ausente')
    cat_effect = cat.get('emotional_effect', '')
    cat_moment = cat.get('moment')
    if cat_type not in ('ausente', None):
        p_cat = (
            f"  En términos de experiencia catártica, la pieza ofrece "
            f"una catarsis de tipo '{cat_type}'. "
        )
        if cat_moment is not None and cat_type != 'negada':
            p_cat += f"El momento de liberación se localiza en {ts_str_local(cat_moment)}. "
        if cat_effect:
            p_cat += cat_effect
        lines.append(wrap_prose(p_cat, indent='  '))
        lines.append("")
    elif cat_type == 'negada' or (not cat.get('present') and
                                   max([p.tension for p in []] or [0]) > 0.5):
        lines.append(wrap_prose(
            "  La pieza acumula tensión pero no la libera — catarsis negada. "
            "El oyente llega al final en un estado de activación irresuelto.",
            indent='  '))
        lines.append("")

    # ── Polaridad y momentum ──────────────────────────────────
    pol_type  = pol.get('polarity_type', '')
    pol_desc  = pol.get('desc', '')
    mom_desc  = mom.get('desc', '')
    mom_trend = mom.get('trend', '')
    if pol_type and pol_desc:
        lines.append(wrap_prose(
            f"  La polaridad emocional es '{pol_type}': {pol_desc.lower()}",
            indent='  '))
        lines.append("")
    if mom_desc:
        lines.append(wrap_prose(
            f"  El impulso musical: {mom_desc.lower()}",
            indent='  '))
        lines.append("")

    # ── Coherencia inter-dimensional ─────────────────────────
    if uem_coh > 0:
        coh_desc = (
            'alta — las 5 dimensiones emocionales se mueven juntas como un bloque'
            if uem_coh > 0.65 else
            'moderada — algunas dimensiones divergen con frecuencia, creando matices'
            if uem_coh > 0.40 else
            'baja — las dimensiones operan de forma independiente, '
            'creando una textura emocional compleja y polifónica'
        )
        lines.append(wrap_prose(
            f"  La coherencia entre las dimensiones emocionales es {coh_desc} "
            f"(índice {uem_coh:.2f}/1.0).",
            indent='  '))
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SECCIÓN IX: HISTORIA EMOCIONAL
# ═══════════════════════════════════════════════════════════════

def write_emotional_story_section(
    story=None,
    catharsis=None,
    narrative_intention=None,
    emotional_ambivalence=None,
    cinematic=None,
    anti_conventional=None,
    semantic_enrichment=None,
    coherence=None,
) -> str:
    """
    Sección IX del ensayo: lectura narrativa de la pieza como historia.
    Consolida los análisis más interpretativos en una sola sección.
    """
    lines = []
    lines.append("─" * 66)
    lines.append("  IX.  HISTORIA EMOCIONAL Y LECTURA NARRATIVA")
    lines.append("─" * 66)
    lines.append("")

    ni  = narrative_intention or {}
    se  = semantic_enrichment or {}
    cat = catharsis or {}
    ci  = cinematic or {}
    ac  = anti_conventional or {}
    co  = coherence or {}

    # ── Arquetipo narrativo e intención ──────────────────────
    ni_arch   = ni.get('archetype', '')
    ni_intent = ni.get('composer_intent', '')
    ni_conf   = ni.get('confidence', 0)
    ni_exp_s  = ni.get('expectation_strategy', '')
    ni_alt    = ni.get('alternative', '')

    if ni_arch and ni_conf > 0.3:
        p_ni = (
            f"  La intención narrativa de esta pieza corresponde al arquetipo "
            f"'{ni_arch}' (confianza {ni_conf:.0%}). {ni_intent} "
        )
        if ni_exp_s:
            p_ni += f"Su estrategia de expectativa: {ni_exp_s.lower()}"
        if ni_alt and ni_conf < 0.65:
            p_ni += f" — aunque también hay rasgos compatibles con el arquetipo '{ni_alt}'."
        lines.append(wrap_prose(p_ni, indent='  '))
        lines.append("")

    # ── Historia emocional generada ───────────────────────────
    if story and len(story.strip()) > 30:
        story_lines = [l for l in story.split('\n') if l.strip()]
        # Incluir los primeros párrafos sustantivos (no encabezados)
        para_count = 0
        for ln in story_lines:
            stripped = ln.strip()
            if stripped.startswith('╔') or stripped.startswith('─') or stripped.startswith('═'):
                continue
            if stripped:
                lines.append(f"  {stripped}")
                para_count += 1
                if para_count >= 8:   # máximo 8 párrafos del story
                    break
        lines.append("")

    # ── Concepto Affektenlehre ────────────────────────────────
    se_concept = se.get('concept', '')
    se_just    = se.get('justification', '')
    se_arc     = se.get('arc_desc', '')
    se_persona = se.get('persona_name', '')
    se_lens    = se.get('persona_lens', '')
    se_fit     = se.get('fit_score', 0)
    alts       = se.get('alternatives', [])

    if se_concept and se_fit > 0.3:
        p_se = (
            f"  Desde la teoría del afecto (Affektenlehre), el perfil de la pieza "
            f"converge con el concepto de '{se_concept}' (ajuste {se_fit:.0%}). "
            f"{se_just} "
            f"El arco narrativo asociado: {se_arc}."
        )
        lines.append(wrap_prose(p_se, indent='  '))
        lines.append("")

        if se_persona and se_lens:
            p_pers = (
                f"  Una lectura desde la perspectiva de un {se_persona.lower()}: "
                f"{se_lens}"
            )
            lines.append(wrap_prose(p_pers, indent='  '))
            lines.append("")

        if alts:
            alt_str = '  |  '.join(f"'{a['concept']}' ({a['score']:.0%})" for a in alts[:2])
            lines.append(f"  Conceptos alternativos compatibles: {alt_str}.")
            lines.append("")

    # ── Elecciones anti-convencionales ────────────────────────
    ac_score   = ac.get('deviation_score', 0)
    ac_profile = ac.get('voice_profile', '')
    ac_sigs    = ac.get('signature_choices', [])
    ac_devs    = ac.get('deviations', [])

    if ac_profile:
        p_ac = f"  La voz compositiva: {ac_profile}. "
        if ac_sigs:
            p_ac += f"Las elecciones más características: {'; '.join(ac_sigs[:3])}."
        lines.append(wrap_prose(p_ac, indent='  '))
        if ac_devs:
            for dev in ac_devs[:2]:
                lines.append(wrap_prose(
                    f"  · [{dev['type']}] {dev['desc']}",
                    indent='    '))
        lines.append("")

    # ── Coherencia emocional y contradicciones ────────────────
    co_desc   = co.get('desc', '')
    co_coh    = co.get('coherence', 1.0)
    co_contra = co.get('contradictions', [])

    if co_desc:
        p_co = f"  Coherencia emocional: {co_desc.lower()}"
        if co_contra:
            types = [c['type'] for c in co_contra[:2]]
            p_co += (
                f" Las tensiones detectadas ({', '.join(types)}) "
                + ("son recursos expresivos deliberados — ambivalencia como método."
                   if co_coh > 0.5
                   else "revelan una escritura que trabaja con la contradicción como forma.")
            )
        lines.append(wrap_prose(p_co, indent='  '))
        lines.append("")

    # ── Tropos cinematográficos ───────────────────────────────
    ci_tropes = ci.get('tropes', {})
    if ci_tropes:
        trope_items = list(ci_tropes.items())[:3]
        p_ci = (
            f"  La pieza activa {len(ci_tropes)} tropo(s) del lenguaje musical "
            f"cinematográfico contemporáneo: "
        )
        trope_descs = []
        for tname, tdata in trope_items:
            trope_descs.append(
                f"'{tname}'"
                + (f" ({tdata['desc'][:60]}...)" if tdata.get('desc') and len(tdata['desc']) > 20 else "")
            )
        p_ci += '; '.join(trope_descs) + '. '
        p_ci += (
            "Estos patrones, heredados del cine, disparan respuestas emocionales "
            "condicionadas en el oyente contemporáneo con la fuerza de un reflejo cultural."
        )
        lines.append(wrap_prose(p_ci, indent='  '))
        lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  RESUMEN EJECUTIVO
# ═══════════════════════════════════════════════════════════════

def write_executive_summary(
    key_name, mode, avg_bpm, total_dur,
    genre_detection=None,
    dynamic_valence=None,
    narrative_intention=None,
    catharsis=None,
    anti_conventional=None,
    semantic_enrichment=None,
    unified_emotional_map=None,
    roughness=None,
    melodic_markov=None,
) -> str:
    """
    Resumen ejecutivo de una página: los 5-7 hallazgos más
    significativos de toda la cadena de análisis.
    Va al inicio del output, antes del ensayo y del informe.
    """
    SEP = '═' * 66
    lines = []
    lines.append(SEP)
    lines.append("  RESUMEN EJECUTIVO")
    lines.append(SEP)
    lines.append("")

    gd  = genre_detection or {}
    dv  = dynamic_valence or {}
    ni  = narrative_intention or {}
    cat = catharsis or {}
    ac  = anti_conventional or {}
    se  = semantic_enrichment or {}
    uem = unified_emotional_map or {}
    rgh = roughness or {}
    mm  = melodic_markov or {}

    dur_min = int(total_dur//60); dur_sec = int(total_dur%60)
    mode_lbl = MODE_CHARS.get(mode, ('?','?','?'))[0]

    # ── Línea de identidad ────────────────────────────────────
    genre_str = gd.get('genre_label', '')
    subg_str  = f" / {gd['subgenre']}" if gd.get('subgenre') else ''
    conf_str  = f"{gd.get('confidence',0):.0%}" if genre_str else ''

    id_line = f"  {key_name}  ·  {avg_bpm:.0f} BPM  ·  {dur_min}m{dur_sec:02d}s"
    if genre_str:
        id_line += f"  ·  {genre_str}{subg_str} ({conf_str})"
    lines.append(id_line)
    lines.append("")

    # ── Hallazgos clave (máx 7) ───────────────────────────────
    findings = []

    # 1. Estado emocional dominante
    phases = uem.get('emotional_phases', [])
    if phases:
        dom = max(phases, key=lambda p: p['duration'])
        dom_pct = dom['duration'] / total_dur * 100 if total_dur > 0 else 0
        dv_mean = dv.get('mean_valence', 0.0)
        light_dark = ('luminosa' if dv_mean > 0.15 else
                      'oscura' if dv_mean < -0.15 else 'ambivalente')
        findings.append(
            f"Estado dominante '{dom['label']}' ({dom_pct:.0f}% del tiempo) — "
            f"experiencia globalmente {light_dark} (valencia media {dv_mean:+.2f})."
        )

    # 2. Arquetipo narrativo
    ni_arch = ni.get('archetype', '')
    ni_conf = ni.get('confidence', 0)
    if ni_arch and ni_conf > 0.35:
        findings.append(
            f"Arquetipo narrativo '{ni_arch}' ({ni_conf:.0%}): "
            f"{ni.get('time_relationship','')}"
        )

    # 3. Catarsis
    cat_type = cat.get('catharsis_type', 'ausente')
    if cat_type not in ('ausente', None):
        cat_moment = cat.get('moment')
        findings.append(
            f"Catarsis '{cat_type}'"
            + (f" en {ts_str_local(cat_moment)}" if cat_moment else "")
            + f": {cat.get('emotional_effect','')[:80]}..."
        )

    # 4. Concepto semántico
    se_concept = se.get('concept', '')
    se_fit     = se.get('fit_score', 0)
    if se_concept and se_fit > 0.4:
        findings.append(
            f"Concepto Affektenlehre: '{se_concept}' (ajuste {se_fit:.0%}) — "
            f"{se.get('justification','')[:80]}..."
        )

    # 5. Voz compositiva
    ac_score   = ac.get('deviation_score', 0)
    ac_profile = ac.get('voice_profile', '')
    ac_sigs    = ac.get('signature_choices', [])
    if ac_profile and ac_score > 0.15:
        sig_str = f" · {'; '.join(ac_sigs[:2])}" if ac_sigs else ''
        findings.append(f"Voz compositiva: {ac_profile}{sig_str}.")

    # 6. Textura psicoacústica
    rgh_mean = rgh.get('mean_roughness', 0)
    rgh_arc  = rgh.get('roughness_arc', '')
    if rgh_mean > 0:
        quality = ('muy suave' if rgh_mean < 0.2 else
                   'moderada' if rgh_mean < 0.5 else
                   'áspera' if rgh_mean < 0.9 else 'muy áspera')
        findings.append(
            f"Textura psicoacústica {quality} (roughness {rgh_mean:.3f}) — "
            f"arco {rgh_arc}."
        )

    # 7. Melodía markov
    mm_style = mm.get('style_profile', '')
    mm_prob  = mm.get('mean_probability', 0)
    if mm_style and mm_style != 'indeterminado':
        findings.append(
            f"Melodía {mm_style} (P media {mm_prob:.2f}) — "
            f"{mm.get('style_desc','')[:80]}"
        )

    # ── Formato de hallazgos ──────────────────────────────────
    for i, f_text in enumerate(findings[:7], 1):
        lines.append(f"  {i}. {f_text}")
    lines.append("")
    lines.append(SEP)
    lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  SÍNTESIS FINAL
# ═══════════════════════════════════════════════════════════════

def write_synthesis(key_name, mode, avg_bpm, total_dur, emotional_genre,
                    narrative_arc, cumulative_state, silence_analysis,
                    tension_curve, pnr, motifs, fingerprint,
                    cultural,
                    genre_detection=None, catharsis=None,
                    emotional_trajectory=None,
                    narrative_intention=None) -> str:
    lines = []
    lines.append("─" * 66)
    lines.append("  VIII.  SÍNTESIS")
    lines.append("─" * 66)
    lines.append("")

    eg = emotional_genre or {}
    top_genre = eg.get('top_genre', '')
    top_score = eg.get('top_score', 0)
    ranking   = eg.get('ranking', [])

    na = narrative_arc or {}
    arc_type  = na.get('arc_type', '')
    climax_pos= na.get('climax_position', 0)

    cs = cumulative_state or {}
    cs_journey = cs.get('journey_type', '')
    fv, fa = cs.get('final_state', (0, 0))

    pn = pnr or {}
    pn_pos  = pn.get('position')
    pn_time = pn.get('time')

    sa = silence_analysis or {}
    sa_ratio = sa.get('ratio', 0)

    gd = genre_detection or {}
    cat = catharsis or {}
    traj = emotional_trajectory or {}
    ni = narrative_intention or {}

    mode_label = MODE_CHARS.get(mode, ('?','?','?'))[0]
    dur_min = int(total_dur//60); dur_sec = int(total_dur%60)

    # Opening — combina género musical y emocional
    genre_str = ''
    if gd.get('genre_label') and gd.get('confidence', 0) > 0.35:
        genre_str = gd['genre_label']
        if gd.get('subgenre'):
            genre_str += f" ({gd['subgenre']})"
    elif top_genre:
        genre_str = f"género emocional '{top_genre}'"

    p1 = (f"  Esta pieza en {key_name} — {dur_min}m{dur_sec:02d}s a "
          f"{avg_bpm:.0f} BPM en modo {mode_label}")
    if genre_str:
        p1 += f" — se clasifica como {genre_str}"
        if top_genre and gd.get('genre_label'):
            p1 += f" con perfil emocional '{top_genre}'"
    p1 += "."
    lines.append(wrap_prose(p1, indent='  '))
    lines.append("")

    # Arquetipo narrativo (nuevo)
    ni_arch = ni.get('archetype', '')
    ni_intent = ni.get('composer_intent', '')
    ni_time_rel = ni.get('time_relationship', '')
    if ni_arch:
        p_ni = (f"  La intención narrativa es la de un '{ni_arch}': "
                f"{ni_intent} {ni_time_rel}.")
        lines.append(wrap_prose(p_ni, indent='  '))
        lines.append("")

    # Catarsis (nuevo)
    cat_type = cat.get('catharsis_type', 'ausente')
    if cat_type not in ('ausente', None):
        p_cat = (f"  En términos de experiencia catártica, la pieza ofrece "
                 f"una catarsis de tipo '{cat_type}'. "
                 f"{cat.get('emotional_effect','')}")
        lines.append(wrap_prose(p_cat, indent='  '))
        lines.append("")

    # Trayectoria VA (nuevo)
    if traj.get('path_shape') and traj.get('total_distance', 0) > 0:
        dom_quad = traj.get('dominant_quadrant', '')
        traj_shape = traj.get('path_shape', '')
        p_traj = (f"  La trayectoria en el espacio emocional es '{traj_shape}', "
                  f"con {traj.get('crossings',0)} cruce(s) de eje. "
                  f"El estado emocional dominante pertenece al cuadrante "
                  f"'{dom_quad}'.")
        lines.append(wrap_prose(p_traj, indent='  '))
        lines.append("")

    # Structural identity
    p2 = (f"  La arquitectura es de tipo '{arc_type}', "
          f"con el clímax al {climax_pos*100:.0f}% de la duración. "
          + (f"El punto de no retorno llega en {ts(pn_time)} ({pn_pos*100:.1f}%), "
             f"tras el cual la tensión acumulada hace imposible el regreso al estado inicial."
             if pn_time is not None else ""))
    lines.append(wrap_prose(p2, indent='  '))
    lines.append("")

    # Melodic-harmonic essence
    n_motifs = len(motifs) if motifs else 0
    fp_tags = (fingerprint or {}).get('style_tags', [])
    if n_motifs and fp_tags:
        p3 = (f"  El discurso melódico gira en torno a {n_motifs} motivo(s) "
              f"que actúan como firma emocional. "
              f"El fingerprint armónico ({'; '.join(fp_tags[:2])}) "
              f"sitúa la pieza en una tradición estilística definida.")
        lines.append(wrap_prose(p3, indent='  '))
        lines.append("")

    # Listener journey
    if cs_journey:
        p4 = (f"  El oyente completa un viaje emocional '{cs_journey}', "
              f"llegando al final en un estado de "
              + ("energía positiva residual" if fv > 0.1 and fa > 0.1
                 else "calma melancólica" if fv < -0.1 and fa < -0.1
                 else "activación sin resolución" if fa > 0.2 and fv < 0
                 else "ambigüedad emocional") + ". "
              + ("La alta densidad escritural no da respiros al oyente." if sa_ratio < 0.05 else ""))
        lines.append(wrap_prose(p4, indent='  '))
        lines.append("")

    # Closing
    mode_core = MODE_CHARS.get(mode, ('?','?','?'))[2]
    cat_closing = (f"una catarsis {cat_type}" if cat_type not in ('ausente',None)
                   else "una experiencia sin catarsis — la tensión es el estado")
    closing = (f"  En definitiva: una pieza de {mode_core}, "
               f"construida con rigor técnico. "
               f"Ofrece {cat_closing}. "
               f"El análisis confirma lo que el oído percibe: "
               f"{'no hay resolución porque no se busca' if sa_ratio < 0.05 and (not pn_time or pn_pos < 0.85) else 'la pieza sabe adónde va y llega'}.")
    lines.append(wrap_prose(closing, indent='  '))
    lines.append("")
    lines.append("─" * 66)
    lines.append("")

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════
#  ASSEMBLER PRINCIPAL
# ═══════════════════════════════════════════════════════════════

def generate_essay(filepath, **kw) -> str:
    """
    Genera el ensayo musical estructurado.
    kw contiene todos los resultados del análisis.
    """
    title = filepath.split('/')[-1].replace('.mid','').replace('.midi','').replace('_',' ')
    SEP = '═' * 66

    parts = []
    parts.append(SEP)
    parts.append(f"  ANÁLISIS MUSICAL — {title.upper()}")
    parts.append(f"  Ensayo estructurado · midi_analyzer v12.0")
    parts.append(SEP)
    parts.append("")

    parts.append(write_identity(
        kw['key_root'], kw['mode'], kw['kconf'], kw['avg_bpm'],
        kw['total_dur'], kw.get('fingerprint'), kw.get('cultural'),
        kw.get('ts_str_val'), kw.get('instruments'), kw.get('key_name'),
        genre_detection=kw.get('genre_detection'),
        semantic_enrichment=kw.get('semantic_enrichment'),
    ))

    parts.append(write_architecture(
        kw.get('musical_form'), kw.get('golden'), kw.get('pnr'),
        kw.get('narrative_arc'), kw.get('dyn_segments'),
        kw.get('sections', []), kw.get('tension_curve', []),
        kw['total_dur'], kw['avg_bpm'],
        ssm=kw.get('ssm'),
        narrative_intention=kw.get('narrative_intention'),
        catharsis=kw.get('catharsis'),
    ))

    parts.append(write_harmony(
        kw.get('chords', []), kw['key_root'], kw['mode'],
        kw.get('canon_progs', []), kw.get('modal_borrow', []),
        kw.get('cadences'), kw.get('harmonic_graph'),
        kw.get('tonal_gravity'), kw.get('tonal_ambiguity'),
        kw.get('harmonic_color'), kw.get('harmonic_semantic'),
        kw.get('subtext'), kw.get('voice_leading'),
        kw.get('pedal_points', []), kw.get('elliptical'),
        chromaticism=kw.get('chromaticism'),
        dissonance_res=kw.get('dissonance_res'),
    ))

    parts.append(write_melody(
        kw.get('interval_anal'), kw.get('motifs', []),
        kw.get('contour', ''), kw.get('phrasing'),
        kw.get('thematic_transforms'), kw.get('thematic_recurrence'),
        kw.get('expectation'), kw.get('narrative_voice'),
        kw.get('comfort_zones'), kw.get('accomp', ''),
        kw.get('inner_voice'),
        counterpoint=kw.get('counterpoint'),
        story=kw.get('story'),
    ))

    parts.append(write_rhythm(
        kw.get('rhythm'), kw.get('micro_timing'),
        kw.get('metric_hierarchy'), kw.get('polyrhythm'),
        kw.get('groove'), kw.get('bass_narrative'),
        kw.get('layer_sync'), kw['avg_bpm'], kw.get('ts_str_val'),
        tempo_gestures=kw.get('tempo_gestures'),
        builds=kw.get('builds'),
    ))

    parts.append(write_listener_experience(
        kw.get('listener_model'), kw.get('cumulative_state'),
        kw.get('auditory_fatigue'), kw.get('emotional_transitions'),
        kw.get('suspense'), kw.get('param_convergence'),
        kw.get('tension_curve', []), kw.get('energy_profile'),
        kw.get('silence_analysis'), kw.get('eternity'),
        kw['total_dur'],
        dynamic_valence=kw.get('dynamic_valence'),
        emotional_ambivalence=kw.get('emotional_ambivalence'),
        emotional_trajectory=kw.get('emotional_trajectory'),
        unified_emotional_map=kw.get('unified_emotional_map'),
        roughness=kw.get('roughness'),
        perceptual=kw.get('perceptual'),
        cinematic=kw.get('cinematic'),
        climax_comparison=kw.get('climax_comparison'),
        momentum=kw.get('momentum'),
        polarity=kw.get('polarity'),
    ))

    parts.append(write_compositional_intention(
        kw.get('fractal'), kw.get('emotional_weight'),
        kw.get('golden'), kw.get('coherence'),
        kw.get('tonal_ambiguity'), kw.get('harmonic_semantic'),
        kw.get('narrative_arc'), kw.get('microexpression'),
        kw.get('fingerprint'), kw.get('motifs', []),
        kw.get('thematic_transforms'), kw['mode'], kw['avg_bpm'],
        anti_conventional=kw.get('anti_conventional'),
        multilevel_density=kw.get('multilevel_density'),
        melodic_markov=kw.get('melodic_markov'),
        semantic_enrichment=kw.get('semantic_enrichment'),
    ))

    parts.append(write_emotional_evolution(
        unified_emotional_map=kw.get('unified_emotional_map'),
        dynamic_valence=kw.get('dynamic_valence'),
        emotional_ambivalence=kw.get('emotional_ambivalence'),
        emotional_trajectory=kw.get('emotional_trajectory'),
        catharsis=kw.get('catharsis'),
        polarity=kw.get('polarity'),
        momentum=kw.get('momentum'),
        total_dur=kw['total_dur'],
    ))

    parts.append(write_emotional_story_section(
        story=kw.get('story'),
        catharsis=kw.get('catharsis'),
        narrative_intention=kw.get('narrative_intention'),
        emotional_ambivalence=kw.get('emotional_ambivalence'),
        cinematic=kw.get('cinematic'),
        anti_conventional=kw.get('anti_conventional'),
        semantic_enrichment=kw.get('semantic_enrichment'),
        coherence=kw.get('coherence'),
    ))

    parts.append(write_synthesis(
        kw.get('key_name', ''), kw['mode'], kw['avg_bpm'],
        kw['total_dur'], kw.get('emotional_genre'),
        kw.get('narrative_arc'), kw.get('cumulative_state'),
        kw.get('silence_analysis'), kw.get('tension_curve', []),
        kw.get('pnr'), kw.get('motifs', []),
        kw.get('fingerprint'), kw.get('cultural'),
        genre_detection=kw.get('genre_detection'),
        catharsis=kw.get('catharsis'),
        emotional_trajectory=kw.get('emotional_trajectory'),
        narrative_intention=kw.get('narrative_intention'),
    ))

    return '\n'.join(parts)

def run_analysis(filepath: str, n_sections: int = 6) -> dict:
    # ── PARSE ────────────────────────────────────────────────
    print("  [1/12] Parseando MIDI...", flush=True)
    notes, meta, t2s = parse_midi(filepath)
    if not notes:
        return "⚠️  No se encontraron notas en el archivo MIDI."

    total_dur = max(n.time_sec + n.duration for n in notes)
    bpms      = [b for _, b in meta['tempo_changes']]
    avg_bpm   = sum(bpms)/len(bpms) if bpms else 120.0
    main_bpm  = meta['tempo_changes'][0][1] if meta['tempo_changes'] else 120.0

    ts        = meta['time_signatures'][0] if meta['time_signatures'] else None
    ts_num    = ts['numerator'] if ts else 4
    ts_s      = f"{ts['numerator']}/{ts['denominator']}" if ts else '4/4'

    # ── KEY & MODE ───────────────────────────────────────────
    print("  [2/12] Detectando tonalidad y modo...", flush=True)
    key_root, mode, kconf = detect_key_and_mode(notes)

    # ── CHORDS + HARMONIC FUNCTION ───────────────────────────
    print("  [3/12] Detectando acordes y función armónica...", flush=True)
    chords = detect_chords(notes)
    chords = assign_harmonic_functions(chords, key_root, mode)

    # ── CADENCES ─────────────────────────────────────────────
    print("  [4/12] Analizando cadencias...", flush=True)
    cadences = analyze_cadences(chords, key_root, mode)

    # ── TENSION CURVE ────────────────────────────────────────
    print("  [5/12] Calculando curva de tensión continua...", flush=True)
    tension_curve = compute_tension_curve(notes, chords, key_root, mode, resolution=0.4)

    # ── MELODY ───────────────────────────────────────────────
    print("  [6/12] Analizando melodía y expectativa...", flush=True)
    mel_notes     = get_mel_track(notes)
    interval_anal = analyze_intervals(mel_notes)
    contour, cpitch = melodic_contour(notes)
    motifs        = extract_motifs(mel_notes)
    expectation   = analyze_melodic_expectation(mel_notes, key_root, mode)

    # ── HARMONY ANALYSIS ─────────────────────────────────────
    print("  [7/12] Analizando armonía y progresiones...", flush=True)
    canon_progs   = find_canonical_progressions(chords, key_root)
    modal_borrow  = detect_modal_borrowing(chords, key_root, mode)
    centricity    = tonal_centricity(chords, key_root)
    resolution    = resolution_tendency(chords, key_root)
    harmonic_graph = build_harmonic_graph(chords)
    voice_leading  = analyze_voice_leading(chords, notes)
    pedal_points   = detect_pedal_points(notes, key_root)

    # ── RHYTHM ───────────────────────────────────────────────
    print("  [8/12] Analizando ritmo...", flush=True)
    rhythm = analyze_rhythm(notes, main_bpm, ts_num)
    accomp = detect_accompaniment(notes, main_bpm)

    # ── TEXTURE & REGISTERS ──────────────────────────────────
    print("  [9/12] Analizando textura y registros...", flush=True)
    texture           = analyze_texture(notes)
    register_analysis = analyze_register_functions(notes)

    # ── TEMPO & DYNAMICS ─────────────────────────────────────
    print("  [10/12] Analizando tempo y dinámica...", flush=True)
    tempo_gestures = detect_tempo_gestures(meta['tempo_changes'])
    instruments = {ch: (prog, *get_instrument(prog))
                   for ch, prog in meta['program_changes'].items()}

    # ── SECTIONS ─────────────────────────────────────────────
    print("  [11/12] Segmentando y analizando secciones...", flush=True)
    sl = total_dur / n_sections
    sections = [
        Section(i, i*sl, (i+1)*sl,
                [n for n in notes if i*sl <= n.time_sec < (i+1)*sl])
        for i in range(n_sections)
    ]
    overall_tension = (sum(section_tension_score(s) for s in sections if s.notes) /
                       max(1, sum(1 for s in sections if s.notes)))
    similar_secs = find_similar_sections(sections)
    builds       = detect_buildups(sections)
    musical_form = analyze_musical_form(sections)

    # ── v4 NEW ANALYSES ──────────────────────────────────────
    print("  [v4] Silencios...", flush=True)
    silence_analysis = analyze_silences(notes, total_dur)
    print("  [v4] Energía...", flush=True)
    energy_profile = compute_energy_profile(notes, total_dur, resolution=0.4)
    print("  [v4] Arco narrativo...", flush=True)
    narrative_arc = analyze_narrative_arc(tension_curve, energy_profile["curve"], silence_analysis, total_dur)
    sync_val = rhythm.get("syncopation", 0)
    mean_exp_v2 = expectation.get("mean_surprise", 0.2)
    sva_list = []
    for s in sections:
        if not s.notes: sva_list.append((0,0,"sin notas")); continue
        vv,aa = compute_va(s, mode, avg_bpm, section_tension_score(s), sync_val, mean_exp_v2)
        sva_list.append((vv, aa, describe_va(vv, aa)))
    print("  [v4] Transiciones emocionales...", flush=True)
    emotional_transitions = analyze_emotional_transitions(sva_list, sections)
    print("  [v4] Suspense...", flush=True)
    suspense = analyze_suspense(tension_curve, total_dur)
    print("  [v4] Transformaciones temáticas...", flush=True)
    thematic_transforms = analyze_thematic_transformations(motifs, notes, key_root)
    print("  [v4] Polaridad emocional...", flush=True)
    polarity = analyze_emotional_polarity(sections, mode, tension_curve, key_root)
    print("  [v4] Densidad de eventos...", flush=True)
    event_density = analyze_event_density(notes, chords, tension_curve, silence_analysis, motifs, total_dur)
    mean_va_v = sum(v for v,a,d in sva_list if d != "sin notas") / max(1, sum(1 for _,_,d in sva_list if d != "sin notas"))
    mean_va_a = sum(a for v,a,d in sva_list if d != "sin notas") / max(1, sum(1 for _,_,d in sva_list if d != "sin notas"))
    print("  [v4] Género emocional...", flush=True)
    emotional_genre = classify_emotional_genre(mode, avg_bpm, overall_tension, centricity, sync_val, mean_va_v, mean_va_a, 0)
    print("  [v4] Cromatismo...", flush=True)
    chromaticism = analyze_chromaticism(notes, key_root, mode)
    print("  [v4] Clímax emocional vs sonoro...", flush=True)
    climax_comparison = compare_emotional_vs_sonic_climax(tension_curve, energy_profile["curve"], total_dur)
    print("  [v4] Groove...", flush=True)
    groove = analyze_groove_hypnosis(notes, avg_bpm, total_dur)

    # ── v6 NEW ANALYSES ──────────────────────────────────────
    print("  [v6] Segmentación dinámica...", flush=True)
    dyn_segments = dynamic_segmentation(notes, tension_curve, energy_profile["curve"], avg_bpm, total_dur)

    print("  [v6] Gravedad tonal dinámica...", flush=True)
    tonal_gravity = dynamic_tonal_gravity(notes, total_dur)

    print("  [v6] Subtext emocional...", flush=True)
    subtext = analyze_emotional_subtext(sections, tension_curve, mode, avg_bpm)

    print("  [v6] Momentum musical...", flush=True)
    momentum = compute_momentum(tension_curve, energy_profile["curve"])

    print("  [v6] Fatiga auditiva...", flush=True)
    auditory_fatigue = compute_auditory_fatigue(notes, tension_curve, total_dur)

    print("  [v6] Polirritmia...", flush=True)
    polyrhythm = analyze_polyrhythm(notes, avg_bpm, ts_num)

    print("  [v6] Estructura fractal...", flush=True)
    fractal = analyze_fractal_structure(notes, sections, total_dur)

    print("  [v6] Peso emocional...", flush=True)
    emotional_weight = compute_emotional_weight(tension_curve, energy_profile["curve"], cadences, motifs, silence_analysis, total_dur)

    print("  [v6] Punto de no retorno...", flush=True)
    pnr = find_point_of_no_return(tension_curve, energy_profile["curve"], total_dur)

    print("  [v6] Densidad narrativa por pulso...", flush=True)
    narrative_density = analyze_narrative_density_per_pulse(notes, chords, motifs, avg_bpm, total_dur)

    print("  [v6] Estado emocional acumulativo...", flush=True)
    cumulative_state = compute_cumulative_emotional_state(sva_list, sections, tension_curve)

    print("  [v6] Color armónico...", flush=True)
    harmonic_color = analyze_harmonic_color(chords, sections)

    print("  [v6] Densidad armónica por registro...", flush=True)
    register_harmony = analyze_harmonic_register_density(notes, total_dur)

    print("  [v6] Pitch bend / microtonalismo...", flush=True)
    pitch_bend = analyze_pitch_bend(meta, total_dur)

    # ── v7 NEW ANALYSES ──────────────────────────────────────
    print("  [v7] Marcadores culturales...", flush=True)
    cultural = analyze_cultural_markers(notes, key_root, mode, chords)
    print("  [v7] Voz conductora...", flush=True)
    narrative_voice = analyze_narrative_voice(notes, sections, tension_curve)
    print("  [v7] Micro-timing...", flush=True)
    micro_timing = analyze_micro_timing(notes, avg_bpm, meta)
    print("  [v7] Convergencia parámetros...", flush=True)
    param_convergence = analyze_parameter_convergence(notes, tension_curve, energy_profile["curve"], sections, total_dur)
    print("  [v7] Zonas de confort tonal...", flush=True)
    comfort_zones = analyze_tonal_comfort_zones(notes, sections, key_root, mode)
    print("  [v7] Jerarquía métrica...", flush=True)
    metric_hierarchy = analyze_metric_hierarchy(notes, avg_bpm, ts_num)
    print("  [v7] Cadencias elípticas...", flush=True)
    elliptical = detect_elliptical_cadences(chords, key_root, mode, notes, avg_bpm)
    print("  [v7] Recurrencia temática...", flush=True)
    thematic_recurrence = analyze_thematic_recurrence(motifs, notes, total_dur, avg_bpm)
    print("  [v7] Listener modeling...", flush=True)
    listener_model = model_listener_experience(tension_curve, energy_profile["curve"], expectation, auditory_fatigue, cumulative_state, total_dur)
    print("  [v7] Densidad semántica armónica...", flush=True)
    harmonic_semantic = analyze_harmonic_semantic_density(chords, total_dur)
    print("  [v7] Función narrativa del bajo...", flush=True)
    bass_narrative = analyze_bass_narrative(notes, avg_bpm, key_root)
    print("  [v7] Sincronía entre capas...", flush=True)
    layer_sync = analyze_layer_synchrony(notes, chords, avg_bpm, total_dur)

    # ── v5 NEW ANALYSES ──────────────────────────────────────
    print('  [v5] Fraseo...', flush=True)
    phrasing = analyze_phrasing(notes, main_bpm, ts_num)
    print('  [v5] Contrapunto...', flush=True)
    counterpoint = analyze_counterpoint(notes, meta)
    print('  [v5] Microdinámicas...', flush=True)
    microexpression = analyze_microexpression(notes, main_bpm)
    print('  [v5] Resolución disonancia...', flush=True)
    dissonance_res = analyze_dissonance_resolution(notes, key_root, mode)
    print('  [v5] Proporción áurea...', flush=True)
    golden = analyze_golden_ratio(tension_curve, energy_profile["curve"], sections, silence_analysis, total_dur)
    print('  [v5] Entropía informacional...', flush=True)
    info_density = analyze_information_density(sections, chords)
    print('  [v5] Ambigüedad tonal...', flush=True)
    tonal_ambiguity = analyze_tonal_ambiguity(notes, key_root, mode, chords)
    print('  [v5] Intensidad perceptual...', flush=True)
    perceptual = compute_perceptual_intensity(notes, total_dur)
    print('  [v5] Voz interna...', flush=True)
    inner_voice = find_inner_voice(notes)
    print('  [v5] Tropos cinematográficos...', flush=True)
    cinematic = detect_cinematic_tropes(notes, tension_curve, energy_profile["curve"], chords, silence_analysis, avg_bpm, total_dur)
    print('  [v5] Fingerprint armónico...', flush=True)
    fingerprint = compute_harmonic_fingerprint(chords, key_root, mode)
    print('  [v5] Momentos de eternidad...', flush=True)
    eternity = detect_eternity_moments(notes, tension_curve, energy_profile["curve"], silence_analysis, avg_bpm, total_dur)
    print('  [v5] Coherencia emocional...', flush=True)
    coherence = analyze_emotional_coherence(sections, sva_list, mode, avg_bpm, tension_curve, cadences)
    print('  [v5] Historia emocional...', flush=True)
    story = generate_emotional_story(sections, sva_list, tension_curve, energy_profile["curve"], mode, key_root, avg_bpm, cadences, motifs, musical_form, silence_analysis, phrasing, NOTE_NAMES[key_root] + " " + mode, total_dur)

    # ── SSM ──────────────────────────────────────────────────
    print('  [ssm] Self-Similarity Matrix...', flush=True)
    ssm = compute_ssm(notes, total_dur, resolution=max(2.0, total_dur / 40))

    # ── v9 DEEP EMOTIONAL ANALYSES ───────────────────────────
    print('  [v9] Valencia dinámica continua...', flush=True)
    dynamic_valence = compute_dynamic_valence(notes, tension_curve, key_root, mode, total_dur)

    print('  [v9] Análisis de catarsis...', flush=True)
    catharsis = analyze_catharsis(tension_curve, dynamic_valence, silence_analysis, energy_profile, total_dur)

    print('  [v9] Mapa de ambivalencia emocional...', flush=True)
    emotional_ambivalence = analyze_emotional_ambivalence(notes, tension_curve, dynamic_valence, mode, avg_bpm, total_dur)

    print('  [v9] Trayectoria emocional VA...', flush=True)
    emotional_trajectory = compute_emotional_trajectory(sva_list, sections, dynamic_valence, total_dur)

    print('  [v9] Elecciones anti-convencionales...', flush=True)
    anti_conventional = analyze_anti_conventional_choices(chords, cadences, key_root, mode, motifs, rhythm, musical_form, avg_bpm, total_dur)

    print('  [v9] Densidad informacional multinivel...', flush=True)
    multilevel_density = analyze_multilevel_information_density(notes, chords, motifs, sections, avg_bpm, total_dur)

    print('  [v9] Intención narrativa...', flush=True)
    narrative_intention = analyze_narrative_intention(tension_curve, energy_profile, dynamic_valence, catharsis, musical_form, sections, sva_list, total_dur, avg_bpm)

    # ── v10 ANALYSES ─────────────────────────────────────────
    print('  [v10] Roughness psicoacústico (Plomp-Levelt)...', flush=True)
    roughness = compute_roughness_curve(notes, total_dur, resolution=0.5)

    print('  [v10] Markov melódico de 2º orden...', flush=True)
    melodic_markov = analyze_melodic_markov(notes, key_root, avg_bpm)

    print('  [v10] Enriquecimiento semántico...', flush=True)
    semantic_enrichment = analyze_semantic_enrichment(
        mode, avg_bpm, overall_tension,
        dynamic_valence, catharsis,
        narrative_intention, silence_analysis
    )

    # ── v11 GENRE DETECTION ──────────────────────────────────
    print('  [v11] Detección de género musical...', flush=True)
    genre_detection = detect_genre(
        notes, chords, rhythm, ts_num, avg_bpm,
        mode, key_root, cadences, meta, total_dur
    )

    # ── v12 UNIFIED EMOTIONAL MAP ────────────────────────────
    print('  [v12] Mapa emocional unificado...', flush=True)
    unified_emotional_map = compute_unified_emotional_map(
        tension_curve, dynamic_valence, energy_profile,
        roughness, notes, total_dur, resolution=2.0
    )

    # ── REPORT ───────────────────────────────────────────────

    print("  [12/12] Generando informe...", flush=True)
    # ── GENERATE ESSAY ───────────────────────────────────────
    print("  [essay] Generando ensayo narrativo...", flush=True)
    # Collect all analysis results into a dict for the essay engine
    key_name_str = NOTE_NAMES[key_root] + " " + mode
    if meta.get('key_signature'):
        key_name_str += f"  [MIDI: {meta['key_signature']}]"
    essay_kwargs = dict(
        key_root=key_root, mode=mode, kconf=kconf,
        avg_bpm=avg_bpm, total_dur=total_dur,
        key_name=key_name_str, ts_str_val=ts_s,
        instruments={ch:(prog,*get_instrument(prog)) for ch,prog in meta['program_changes'].items()},
        chords=chords, sections=sections, tension_curve=tension_curve,
        canon_progs=canon_progs, modal_borrow=modal_borrow,
        cadences=cadences, harmonic_graph=harmonic_graph,
        tonal_gravity=tonal_gravity, tonal_ambiguity=tonal_ambiguity,
        harmonic_color=harmonic_color, harmonic_semantic=harmonic_semantic,
        subtext=subtext, voice_leading=voice_leading,
        pedal_points=pedal_points, elliptical=elliptical,
        interval_anal=interval_anal, motifs=motifs,
        contour=contour, phrasing=phrasing,
        thematic_transforms=thematic_transforms, thematic_recurrence=thematic_recurrence,
        expectation=expectation, narrative_voice=narrative_voice,
        comfort_zones=comfort_zones, accomp=accomp,
        inner_voice=inner_voice, rhythm=rhythm,
        micro_timing=micro_timing, metric_hierarchy=metric_hierarchy,
        polyrhythm=polyrhythm, groove=groove,
        bass_narrative=bass_narrative, layer_sync=layer_sync,
        listener_model=listener_model, cumulative_state=cumulative_state,
        auditory_fatigue=auditory_fatigue, emotional_transitions=emotional_transitions,
        suspense=suspense, param_convergence=param_convergence,
        energy_profile=energy_profile, silence_analysis=silence_analysis,
        eternity=eternity, fractal=fractal,
        emotional_weight=emotional_weight, golden=golden,
        microexpression=microexpression, fingerprint=fingerprint,
        cultural=cultural, emotional_genre=emotional_genre,
        narrative_arc=narrative_arc, pnr=pnr,
        musical_form=musical_form, dyn_segments=dyn_segments,
        ssm=ssm,
        dynamic_valence=dynamic_valence, catharsis=catharsis,
        emotional_ambivalence=emotional_ambivalence,
        emotional_trajectory=emotional_trajectory,
        anti_conventional=anti_conventional,
        multilevel_density=multilevel_density,
        narrative_intention=narrative_intention,
        roughness=roughness,
        melodic_markov=melodic_markov,
        semantic_enrichment=semantic_enrichment,
        genre_detection=genre_detection,
        unified_emotional_map=unified_emotional_map,
        chromaticism=chromaticism,
        dissonance_res=dissonance_res,
        counterpoint=counterpoint,
        story=story,
        tempo_gestures=tempo_gestures,
        builds=builds,
        perceptual=perceptual,
        cinematic=cinematic,
        climax_comparison=climax_comparison,
        momentum=momentum,
        polarity=polarity,
    )
    essay_text = generate_essay(filepath, **essay_kwargs)


    # ═══════════════════════════════════════════════════════
    #  run_analysis() devuelve un dict con TODOS los resultados.
    #  Los tres renderers (essay, html, yaml) consumen este dict.
    # ═══════════════════════════════════════════════════════
    results = dict(
        filepath=filepath,
        title=filepath.split('/')[-1].replace('.mid','').replace('.midi','').replace('_',' '),
        key_root=key_root, mode=mode, kconf=kconf,
        avg_bpm=avg_bpm, main_bpm=main_bpm,
        ts_str_val=ts_s, ts_num=ts_num, total_dur=total_dur,
        n_sections=n_sections,
        notes_count=len(notes),
        instruments={ch:(prog,*get_instrument(prog)) for ch,prog in meta['program_changes'].items()},
        meta=meta, sections=sections, chords=chords,
        contour_p=cpitch,
        canon_progs=canon_progs, modal_borrow=modal_borrow,
        centricity=centricity, resolution=resolution,
        cadences=cadences, harmonic_graph=harmonic_graph,
        tonal_gravity=tonal_gravity, tonal_ambiguity=tonal_ambiguity,
        harmonic_color=harmonic_color, harmonic_semantic=harmonic_semantic,
        subtext=subtext, voice_leading=voice_leading,
        pedal_points=pedal_points, elliptical=elliptical,
        chromaticism=chromaticism, dissonance_res=dissonance_res,
        interval_anal=interval_anal, contour=contour,
        motifs=motifs, phrasing=phrasing,
        thematic_transforms=thematic_transforms, thematic_recurrence=thematic_recurrence,
        expectation=expectation, narrative_voice=narrative_voice,
        comfort_zones=comfort_zones, accomp=accomp,
        inner_voice=inner_voice, counterpoint=counterpoint,
        rhythm=rhythm, texture=texture,
        tempo_gestures=tempo_gestures, builds=builds,
        groove=groove, polyrhythm=polyrhythm,
        micro_timing=micro_timing, metric_hierarchy=metric_hierarchy,
        bass_narrative=bass_narrative, layer_sync=layer_sync,
        musical_form=musical_form, similar_secs=similar_secs,
        register_analysis=register_analysis, register_harmony=register_harmony,
        ssm=ssm,
        tension_curve=tension_curve,
        energy_profile=energy_profile,
        silence_analysis=silence_analysis,
        dynamic_valence=dynamic_valence,
        roughness=roughness,
        perceptual=perceptual,
        unified_emotional_map=unified_emotional_map,
        sva_list=sva_list,
        emotional_genre=emotional_genre,
        narrative_arc=narrative_arc,
        emotional_transitions=emotional_transitions,
        suspense=suspense, polarity=polarity,
        event_density=event_density,
        climax_comparison=climax_comparison,
        emotional_ambivalence=emotional_ambivalence,
        emotional_trajectory=emotional_trajectory,
        catharsis=catharsis,
        cumulative_state=cumulative_state,
        auditory_fatigue=auditory_fatigue,
        param_convergence=param_convergence,
        momentum=momentum, eternity=eternity,
        narrative_intention=narrative_intention,
        anti_conventional=anti_conventional,
        semantic_enrichment=semantic_enrichment,
        multilevel_density=multilevel_density,
        fractal=fractal, emotional_weight=emotional_weight,
        coherence=coherence, story=story,
        genre_detection=genre_detection,
        cinematic=cinematic, fingerprint=fingerprint,
        cultural=cultural, golden=golden,
        pnr=pnr, narrative_density=narrative_density,
        info_density=info_density,
        pitch_bend=pitch_bend,
        microexpression=microexpression,
        melodic_markov=melodic_markov,
        overall_tension=overall_tension,
        essay_text=essay_text,
        # pass-through for legacy generate_report
        notes=notes, meta_full=meta,
        main_bpm_val=main_bpm,
        n_sections_val=n_sections,
    )
    return results
