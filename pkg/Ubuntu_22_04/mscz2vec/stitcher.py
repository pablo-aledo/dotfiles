#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COHERENCE STITCHER  v1.0                                  ║
║         Orquestador de fragmentos MIDI con coherencia armónica               ║
║                                                                              ║
║  Usa los fingerprints (.fingerprint.json) generados por midi_dna_unified    ║
║  v1.4 para:                                                                  ║
║                                                                              ║
║  [1] ANALIZAR los fragmentos disponibles y sus bordes                        ║
║  [2] ORDENAR los fragmentos en la secuencia más coherente                    ║
║      (o respetar el orden manual que elijas)                                 ║
║  [3] PUNTUAR cada transición posible con un score de compatibilidad          ║
║      armónica, de registro, de tempo y de tensión                            ║
║  [4] GENERAR puentes automáticos entre fragmentos incompatibles              ║
║      llamando a midi_dna_unified.py con los parámetros correctos             ║
║  [5] ENSAMBLAR el MIDI final concatenando fragmentos + puentes               ║
║                                                                              ║
║  FLUJO TÍPICO:                                                               ║
║    1. Genera fragmentos con midi_dna_unified.py --export-fingerprint         ║
║    2. Corre este script con los fingerprints (o los MIDIs directamente)      ║
║    3. El script analiza, ordena, genera puentes y produce obra_final.mid     ║
╚══════════════════════════════════════════════════════════════════════════════╝

USO:
    # Modo automático: pasa los fingerprints (o los MIDIs, detecta los JSON)
    python stitcher.py seccion_A.fingerprint.json seccion_B.fingerprint.json seccion_C.fingerprint.json

    # También acepta los MIDIs directamente (busca el .fingerprint.json junto a ellos)
    python stitcher.py seccion_A.mid seccion_B.mid seccion_C.mid

    # Con orden manual (no reordena, solo genera puentes donde hagan falta)
    python stitcher.py *.fingerprint.json --fixed-order

    # Sólo analizar y mostrar la matriz de compatibilidad (sin generar nada)
    python stitcher.py *.fingerprint.json --analyze-only

    # Especificar el script mixer y las fuentes MIDI para los puentes
    python stitcher.py *.fingerprint.json \\
        --mixer midi_dna_unified.py \\
        --bridge-sources style1.mid style2.mid \\
        --output obra_epica.mid

    # Con fuentes separadas para puentes de "mundo style1" y "mundo style2"
    python stitcher.py *.fingerprint.json \\
        --bridge-sources style1.mid style2.mid \\
        --bridge-bars 8 \\
        --tempo-map "seccion_A.mid:112, seccion_B.mid:96" \\
        --output obra_final.mid --verbose

OPCIONES:
    --fixed-order       No reordenar fragmentos; solo generar puentes
    --analyze-only      Mostrar análisis y matriz de compatibilidad, salir
    --mixer PATH        Ruta a midi_dna_unified.py (default: ./midi_dna_unified.py)
    --bridge-sources    MIDIs fuente para generar los puentes (1 o más)
    --bridge-bars N     Compases de cada puente (default: 4)
    --bridge-mode MODE  Modo del mixer para los puentes (default: emotion)
    --min-score F       Score mínimo para NO generar puente (default: 0.55)
    --tempo-map         Overrides de tempo por fragmento: "file.mid:BPM, ..."
    --output FILE       MIDI final ensamblado (default: obra_final.mid)
    --no-bridge         Ensamblar sin generar puentes (solo concatenar)
    --verbose           Informe detallado de scores y decisiones
    --dry-run           Mostrar comandos de puentes sin ejecutarlos

DEPENDENCIAS:
    pip install mido
    (midi_dna_unified.py v1.4 para generación de puentes)
"""

import sys
import os
import json
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from itertools import permutations

try:
    import mido
except ImportError:
    print("ERROR: pip install mido")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE COMPATIBILIDAD ARMÓNICA
# ══════════════════════════════════════════════════════════════════════════════

# Qué acordes de salida conectan bien con qué acordes de entrada
# Puntuación 0.0–1.0 (1.0 = conexión perfecta)
HARMONIC_COMPATIBILITY = {
    # Cadencia auténtica clásica: V → I / i
    ('V',  'I' ): 1.0,  ('V',  'i' ): 1.0,
    # Cadencia plagal: IV → I
    ('IV', 'I' ): 0.85, ('iv', 'i' ): 0.85,
    # Cadencia rota: V → vi / VI
    ('V',  'vi'): 0.80, ('V',  'VI'): 0.80,
    # Repetición tónica (misma sección o variación)
    ('I',  'I' ): 0.75, ('i',  'i' ): 0.75,
    # ii → V (semiprogresión, espera resolución)
    ('ii', 'V' ): 0.75, ('ii°','V' ): 0.70,
    # IV → V (preparación dominante)
    ('IV', 'V' ): 0.70, ('iv', 'V' ): 0.70,
    # I → IV (movimiento subdominante)
    ('I',  'IV'): 0.65, ('i',  'iv'): 0.65,
    # I → V (apertura hacia dominante)
    ('I',  'V' ): 0.60, ('i',  'V' ): 0.60,
    # vi → IV (movimiento pop/romántico)
    ('vi', 'IV'): 0.60,
    # Semicadencia: I → V (inicio de frase)
    ('i',  'III'): 0.55, ('i', 'VII'): 0.55,
    # Movimientos más ambiguos
    ('V',  'IV'): 0.40,  # anticlímax
    ('I',  'vi'): 0.55,
    ('vi', 'V' ): 0.55,
    ('vi', 'ii'): 0.55,
}

# Ciclo de quintas: distancia entre notas de bajo (semitones)
# Distancia 0 = misma nota, 5 o 7 = quinta, 12 = octava
def _fifth_distance(pc_a, pc_b):
    """Distancia en el círculo de quintas entre dos pitch classes (0-11)."""
    diff = (pc_b - pc_a) % 12
    # En el ciclo de quintas: cada paso es 7 semitonos
    fifth_steps = (diff * 7) % 12
    return min(fifth_steps, 12 - fifth_steps)


# Tonalidades relativas y vecinas (para calcular distancia tonal)
RELATIVE_KEYS = {
    # mayor → relativo menor
    'C major':   'A minor',  'G major':  'E minor',  'D major':  'B minor',
    'A major':   'F# minor', 'E major':  'C# minor', 'B major':  'G# minor',
    'F major':   'D minor',  'Bb major': 'G minor',  'Eb major': 'C minor',
    'Ab major':  'F minor',  'Db major': 'Bb minor', 'Gb major': 'Eb minor',
}
# Invertir para tener menor → mayor también
RELATIVE_KEYS.update({v: k for k, v in RELATIVE_KEYS.items()})

def _key_distance(key_a_str, key_b_str):
    """
    Distancia tonal entre dos tonalidades (0=misma, 1=relativa/paralela,
    2=vecina en ciclo de quintas, ...).
    Devuelve float 0.0–1.0 donde 0 es más cercana.
    """
    if key_a_str == key_b_str:
        return 0.0
    if RELATIVE_KEYS.get(key_a_str) == key_b_str:
        return 0.1   # tonalidad relativa: muy compatible
    # Distancia en ciclo de quintas
    NOTE_ORDER = ['C','G','D','A','E','B','F#','Db','Ab','Eb','Bb','F']
    def tonic(ks):
        parts = ks.split()
        t = parts[0].replace('b', 'b')
        return NOTE_ORDER.index(t) if t in NOTE_ORDER else 6
    try:
        dist = abs(tonic(key_a_str) - tonic(key_b_str))
        dist = min(dist, 12 - dist)
        return min(dist / 6.0, 1.0)
    except Exception:
        return 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE FINGERPRINTS
# ══════════════════════════════════════════════════════════════════════════════

def load_fingerprint(path):
    """
    Carga un fingerprint JSON. Acepta:
    - Ruta directa a .fingerprint.json
    - Ruta a .mid (busca el .fingerprint.json junto a él)
    """
    p = Path(path)
    if p.suffix in ('.mid', '.midi'):
        fp_path = p.with_suffix('').with_suffix('.fingerprint.json')
        if not fp_path.exists():
            # Intento alternativo: mismo nombre con .fingerprint.json al final
            fp_path = Path(str(p) + '.fingerprint.json')
        if not fp_path.exists():
            raise FileNotFoundError(
                f"No se encontró fingerprint para {path}.\n"
                f"  Esperado: {p.with_suffix('').with_suffix('.fingerprint.json')}\n"
                f"  Genera el fragmento con --export-fingerprint primero."
            )
        path = str(fp_path)

    with open(path, encoding='utf-8') as f:
        fp = json.load(f)

    # Normalizar: asegurar que meta.midi_file apunta a un fichero existente
    midi_file = fp['meta'].get('midi_file', '')
    if midi_file and not os.path.exists(midi_file):
        # Intentar buscar el MIDI en el mismo directorio que el JSON
        candidate = str(Path(path).parent / Path(midi_file).name)
        if os.path.exists(candidate):
            fp['meta']['midi_file'] = candidate

    fp['_fingerprint_path'] = str(Path(path).resolve())
    return fp


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING DE COMPATIBILIDAD ENTRE DOS FRAGMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def score_transition(fp_a, fp_b, verbose=False):
    """
    Calcula un score de compatibilidad 0.0–1.0 entre el final del fragmento A
    y el inicio del fragmento B. Cuanto mayor, mejor conectan sin puente.

    Componentes del score:
      - Armonía    (40%): compatibilidad acorde salida→entrada
      - Bajo       (20%): movimiento del bajo (círculo de quintas)
      - Tonalidad  (15%): distancia entre tonalidades
      - Registro   (10%): salto de registro de la melodía
      - Tempo      (10%): diferencia de BPM
      - Tensión    ( 5%): continuidad de la curva de tensión
    """
    exit_  = fp_a['exit']
    entry_ = fp_b['entry']
    meta_a = fp_a['meta']
    meta_b = fp_b['meta']
    hints_a = fp_a['stitching_hints']

    details = {}

    # ── 1. Armonía (40%) ─────────────────────────────────────────────────────
    chord_exit  = exit_['chord_roman']
    chord_entry = entry_['chord_roman']
    harm_score = HARMONIC_COMPATIBILITY.get(
        (chord_exit, chord_entry),
        HARMONIC_COMPATIBILITY.get((chord_entry, chord_exit), 0.30)
    )
    # Si alguno es 'unknown', penalizar pero no anular
    if chord_exit == 'unknown' or chord_entry == 'unknown':
        harm_score = max(harm_score, 0.35)
    details['harmonia'] = harm_score

    # ── 2. Bajo (20%) ────────────────────────────────────────────────────────
    bass_dist = _fifth_distance(
        exit_['bass_pitch_class'],
        entry_['bass_pitch_class']
    )
    # dist 0→1.0, dist 1(quinta)→0.85, dist 6(tritono)→0.0
    bass_score = max(0.0, 1.0 - bass_dist / 6.0)
    details['bajo'] = bass_score

    # ── 3. Tonalidad (15%) ───────────────────────────────────────────────────
    key_a = f"{meta_a['key_tonic']} {meta_a['key_mode']}"
    key_b = f"{meta_b['key_tonic']} {meta_b['key_mode']}"
    key_dist = _key_distance(key_a, key_b)
    key_score = 1.0 - key_dist
    details['tonalidad'] = key_score

    # ── 4. Registro de melodía (10%) ─────────────────────────────────────────
    REGISTER_VALS = {'low': 0, 'mid-low': 1, 'mid': 2, 'mid-high': 3, 'high': 4}
    reg_exit  = REGISTER_VALS.get(exit_.get('melody_register', 'mid'), 2)
    reg_entry = REGISTER_VALS.get(entry_.get('melody_register', 'mid'), 2)
    reg_diff  = abs(reg_exit - reg_entry)
    reg_score = max(0.0, 1.0 - reg_diff / 4.0)
    details['registro'] = reg_score

    # ── 5. Tempo (10%) ───────────────────────────────────────────────────────
    bpm_a = meta_a['tempo_bpm']
    bpm_b = meta_b['tempo_bpm']
    bpm_diff = abs(bpm_a - bpm_b)
    tol = hints_a.get('tempo_tolerance_bpm', 15.0)
    tempo_score = max(0.0, 1.0 - bpm_diff / (tol * 4))
    details['tempo'] = tempo_score

    # ── 6. Tensión (5%) ──────────────────────────────────────────────────────
    tension_exit  = fp_a['tension_curve']['exit']
    tension_entry = fp_b['tension_curve']['entry']
    tension_diff  = abs(tension_exit - tension_entry)
    tension_score = max(0.0, 1.0 - tension_diff * 2)
    details['tension'] = tension_score

    # ── Score ponderado ───────────────────────────────────────────────────────
    weights = {
        'harmonia':  0.40,
        'bajo':      0.20,
        'tonalidad': 0.15,
        'registro':  0.10,
        'tempo':     0.10,
        'tension':   0.05,
    }
    total = sum(details[k] * weights[k] for k in weights)

    if verbose:
        a_name = Path(meta_a['midi_file']).stem
        b_name = Path(meta_b['midi_file']).stem
        print(f"\n  Score {a_name} → {b_name}: {total:.3f}")
        print(f"    Acorde : {chord_exit} → {chord_entry}  ({details['harmonia']:.2f})")
        print(f"    Bajo   : {fp_a['exit']['bass_note_name']} → "
              f"{fp_b['entry']['bass_note_name']}  ({details['bajo']:.2f})")
        print(f"    Tonal. : {key_a} → {key_b}  ({details['tonalidad']:.2f})")
        print(f"    Regist.: {exit_.get('melody_register')} → "
              f"{entry_.get('melody_register')}  ({details['registro']:.2f})")
        print(f"    Tempo  : {bpm_a:.0f} → {bpm_b:.0f} BPM  ({details['tempo']:.2f})")
        print(f"    Tensión: {tension_exit:.2f} → {tension_entry:.2f}  ({details['tension']:.2f})")

    return total, details


# ══════════════════════════════════════════════════════════════════════════════
#  MATRIZ DE COMPATIBILIDAD
# ══════════════════════════════════════════════════════════════════════════════

def build_compatibility_matrix(fingerprints, verbose=False):
    """
    Construye la matriz NxN de scores de transición entre todos los fragmentos.
    matrix[i][j] = score de A[i] → A[j]
    """
    n = len(fingerprints)
    matrix = [[0.0] * n for _ in range(n)]
    details_matrix = [[None] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0.0
                continue
            score, details = score_transition(fingerprints[i], fingerprints[j],
                                              verbose=verbose)
            matrix[i][j] = score
            details_matrix[i][j] = details

    return matrix, details_matrix


def print_compatibility_matrix(fingerprints, matrix):
    """Imprime la matriz de compatibilidad en ASCII."""
    names = [Path(fp['meta']['midi_file']).stem[:12] for fp in fingerprints]
    n = len(names)
    col_w = 8

    print("\n" + "═" * (14 + col_w * n))
    print("  MATRIZ DE COMPATIBILIDAD  (salida → entrada)")
    print("  0.0=incompatible  0.55=límite puente  1.0=perfecto")
    print("═" * (14 + col_w * n))

    # Cabecera
    header = f"{'':14s}" + "".join(f"{n[:col_w-1]:>{col_w}}" for n in names)
    print(header)
    print("─" * (14 + col_w * n))

    for i, row_name in enumerate(names):
        row = f"  {row_name:<12s}"
        for j in range(n):
            if i == j:
                cell = "  ──"
            else:
                s = matrix[i][j]
                bar = "██" if s >= 0.80 else "▓▓" if s >= 0.65 else "░░" if s >= 0.55 else "  "
                cell = f" {bar}{s:.2f}"
            row += f"{cell:>{col_w}}"
        print(row)

    print("═" * (14 + col_w * n))


# ══════════════════════════════════════════════════════════════════════════════
#  ORDENACIÓN ÓPTIMA DE FRAGMENTOS
# ══════════════════════════════════════════════════════════════════════════════

def _greedy_order(matrix, n):
    """
    Heurística greedy: desde cada fragmento elige el siguiente con mayor score.
    Evalúa todas las posiciones de inicio y devuelve la mejor secuencia total.
    """
    best_seq, best_score = None, -1.0

    for start in range(n):
        seq = [start]
        remaining = set(range(n)) - {start}
        total = 0.0

        while remaining:
            current = seq[-1]
            # Mejor siguiente disponible
            best_next = max(remaining, key=lambda j: matrix[current][j])
            total += matrix[current][best_next]
            seq.append(best_next)
            remaining.remove(best_next)

        if total > best_score:
            best_score = total
            best_seq = seq

    return best_seq, best_score


def _exhaustive_order(matrix, n):
    """Búsqueda exhaustiva (solo para n <= 8, evitar explosión combinatoria)."""
    best_seq, best_score = None, -1.0
    for perm in permutations(range(n)):
        score = sum(matrix[perm[i]][perm[i+1]] for i in range(n-1))
        if score > best_score:
            best_score = score
            best_seq = list(perm)
    return best_seq, best_score


def find_optimal_order(matrix, n, force_first=None, force_last=None):
    """
    Encuentra el orden de fragmentos que maximiza la suma de scores de
    transición. Usa exhaustivo para n≤7, greedy para n>7.
    """
    if force_first is not None or force_last is not None:
        # Con restricciones: greedy con penalización de posiciones forzadas
        ff = force_first if force_first is not None else -1
        fl = force_last  if force_last  is not None else -1
        free = [i for i in range(n) if i != ff and i != fl]
        if n <= 7:
            best_middle, best_score = None, -1.0
            for perm in permutations(free):
                seq = (([ff] if ff >= 0 else []) +
                       list(perm) +
                       ([fl] if fl >= 0 else []))
                score = sum(matrix[seq[i]][seq[i+1]] for i in range(len(seq)-1))
                if score > best_score:
                    best_score, best_middle = score, seq
            return best_middle, best_score
        else:
            seq_g, sc_g = _greedy_order(matrix, n)
            # Reordenar para respetar restricciones (reubicación simple)
            if ff >= 0 and seq_g[0] != ff:
                seq_g.remove(ff); seq_g.insert(0, ff)
            if fl >= 0 and seq_g[-1] != fl:
                seq_g.remove(fl); seq_g.append(fl)
            sc_g = sum(matrix[seq_g[i]][seq_g[i+1]] for i in range(n-1))
            return seq_g, sc_g

    if n <= 7:
        return _exhaustive_order(matrix, n)
    else:
        return _greedy_order(matrix, n)


# ══════════════════════════════════════════════════════════════════════════════
#  DECISIÓN DE PUENTE
# ══════════════════════════════════════════════════════════════════════════════

def decide_bridge(fp_a, fp_b, score, min_score):
    """
    Decide si hay que generar un puente y con qué parámetros.
    Devuelve un dict con la decisión y los parámetros sugeridos para el mixer.
    """
    needs_bridge = score < min_score

    meta_a = fp_a['meta']
    meta_b = fp_b['meta']
    exit_  = fp_a['exit']
    entry_ = fp_b['entry']

    # Diferencia de tempo: si es grande, sugerir ritardando en el puente
    bpm_diff = abs(meta_a['tempo_bpm'] - meta_b['tempo_bpm'])
    tempo_bridge = (meta_a['tempo_bpm'] + meta_b['tempo_bpm']) / 2.0

    # Diferencia de tensión: determinar el arco emocional del puente
    t_exit  = fp_a['tension_curve']['exit']
    t_entry = fp_b['tension_curve']['entry']
    if t_entry > t_exit + 0.2:
        bridge_arc = 'rising'
        emotion_morph = "0:0.0, {B}:1.0"
    elif t_entry < t_exit - 0.2:
        bridge_arc = 'falling'
        emotion_morph = "0:1.0, {B}:0.0"
    else:
        bridge_arc = 'stable'
        emotion_morph = None

    # Registro: si cambia, el puente debe hacer la transición
    REGISTER_VALS = {'low': 0, 'mid-low': 1, 'mid': 2, 'mid-high': 3, 'high': 4}
    reg_exit  = REGISTER_VALS.get(exit_.get('melody_register', 'mid'), 2)
    reg_entry = REGISTER_VALS.get(entry_.get('melody_register', 'mid'), 2)
    REG_NAMES = {0:'low', 1:'mid-low', 2:'mid', 3:'mid-high', 4:'high'}
    if abs(reg_exit - reg_entry) >= 2:
        mt_register = f"0:{REG_NAMES[reg_exit]}, {{B}}:{REG_NAMES[reg_entry]}"
    else:
        mt_register = None

    # Acorde pivot: la tonalidad del puente
    # Si las tonalidades son distintas, usar la del fragmento B (transición hacia allá)
    key_a = f"{meta_a['key_tonic']} {meta_a['key_mode']}"
    key_b = f"{meta_b['key_tonic']} {meta_b['key_mode']}"
    bridge_key = key_b  # el puente "aterriza" en la tonalidad de B

    return {
        'needs_bridge':   needs_bridge,
        'score':          score,
        'tempo':          tempo_bridge,
        'key':            bridge_key,
        'arc':            bridge_arc,
        'bpm_diff':       bpm_diff,
        'emotion_morph':  emotion_morph,
        'mt_register':    mt_register,
        'chord_from':     exit_['chord_roman'],
        'chord_to':       entry_['chord_roman'],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE PUENTES
# ══════════════════════════════════════════════════════════════════════════════

def build_bridge_command(fp_a, fp_b, bridge_info, bridge_sources,
                         bridge_bars, bridge_mode, output_path, mixer_path):
    """
    Construye el comando para llamar a midi_dna_unified.py y generar
    el puente entre fp_a y fp_b.
    """
    cmd = [sys.executable, mixer_path]

    # Fuentes MIDI para el puente
    cmd += bridge_sources

    # Parámetros básicos
    cmd += ['--bars', str(bridge_bars)]
    cmd += ['--mode', bridge_mode]
    cmd += ['--key', bridge_info['key']]
    cmd += ['--tempo', f"{bridge_info['tempo']:.1f}"]
    cmd += ['--output', output_path]
    cmd += ['--seed', '99']

    # Morphing emocional si hay cambio de tensión
    if bridge_info['emotion_morph']:
        morph_spec = bridge_info['emotion_morph'].replace('{B}', str(bridge_bars))
        cmd += ['--mt-emotion-morph', morph_spec]

    # Morphing de registro si hay salto grande
    if bridge_info['mt_register']:
        reg_spec = bridge_info['mt_register'].replace('{B}', str(bridge_bars))
        cmd += ['--mt-register', reg_spec]

    # Si el tempo cambia mucho, hacer morphing de densidad
    if bridge_info['bpm_diff'] > 20:
        cmd += ['--mt-density', f"0:medium, {bridge_bars}:sparse"]

    # Siempre morphing rítmico si hay 2+ fuentes
    if len(bridge_sources) >= 2:
        cmd += ['--mt-rhythm-morph', f"0:0.0, {bridge_bars}:1.0"]

    return cmd


def generate_bridge(fp_a, fp_b, bridge_info, bridge_sources,
                    bridge_bars, bridge_mode, mixer_path,
                    work_dir, verbose=False, dry_run=False):
    """
    Llama al mixer para generar el puente y devuelve la ruta al MIDI generado.
    Si dry_run=True, solo muestra el comando sin ejecutar.
    """
    name_a = Path(fp_a['meta']['midi_file']).stem
    name_b = Path(fp_b['meta']['midi_file']).stem
    bridge_name = f"bridge_{name_a}_to_{name_b}.mid"
    bridge_path = str(Path(work_dir) / bridge_name)

    cmd = build_bridge_command(
        fp_a, fp_b, bridge_info,
        bridge_sources, bridge_bars, bridge_mode,
        bridge_path, mixer_path
    )

    if verbose or dry_run:
        print(f"\n  🔧 Puente: {name_a} → {name_b}")
        print(f"     Acorde: {bridge_info['chord_from']} → {bridge_info['chord_to']}")
        print(f"     Arco  : {bridge_info['arc']}  |  Tonalidad: {bridge_info['key']}")
        print(f"     Tempo : {bridge_info['tempo']:.1f} BPM  ({bridge_bars} compases)")
        print(f"     CMD   : {' '.join(cmd)}")

    if dry_run:
        return None

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            timeout=120
        )
        if result.returncode != 0:
            print(f"  ⚠ Error generando puente {bridge_name}:")
            if result.stderr:
                print(f"    {result.stderr[:300]}")
            return None
        if not os.path.exists(bridge_path):
            print(f"  ⚠ El mixer no generó: {bridge_path}")
            return None
        print(f"  ✓ Puente generado: {bridge_name}")
        return bridge_path
    except subprocess.TimeoutExpired:
        print(f"  ⚠ Timeout generando puente {bridge_name}")
        return None
    except Exception as e:
        print(f"  ⚠ Excepción generando puente: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  ENSAMBLADO MIDI FINAL
# ══════════════════════════════════════════════════════════════════════════════

def _get_tempo(mid):
    """Primer tempo encontrado en el MIDI (microsegundos por beat)."""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    return 500000  # 120 BPM por defecto


def _track_duration_ticks(track):
    """Duración de UNA pista en ticks (suma de deltas)."""
    t = 0
    for msg in track:
        t += msg.time
    return t


def _seg_duration_ticks(mid):
    """
    Duración real del segmento en ticks.
    En MIDI tipo 1 las pistas corren en paralelo: la duración real
    es el máximo entre todas las pistas que tienen notas.
    Fallback: máximo de todas las pistas si ninguna tiene notas.
    """
    note_track_durs = [
        _track_duration_ticks(t) for t in mid.tracks
        if any(not m.is_meta for m in t)
    ]
    if note_track_durs:
        return max(note_track_durs)
    # Fallback: cualquier pista
    all_durs = [_track_duration_ticks(t) for t in mid.tracks]
    return max(all_durs) if all_durs else 0


def _track_label(track):
    """
    Devuelve el nombre de la pista.
    mido expone track.name directamente (lo serializa como MetaMessage
    track_name al guardar). También busca en el stream como fallback.
    """
    # Método directo: atributo del objeto MidiTrack
    name = getattr(track, 'name', None)
    if name and name.strip():
        return name.strip()
    # Fallback: buscar MetaMessage track_name en el stream de mensajes
    for msg in track:
        if msg.is_meta and msg.type == 'track_name':
            if msg.name and msg.name.strip():
                return msg.name.strip()
    return None


def assemble_midi(sequence, output_path, verbose=False):
    """
    Concatena una secuencia de ficheros MIDI en un único MIDI de salida,
    preservando TODAS las pistas de cada segmento.

    Estrategia:
    - MIDI tipo 1 de salida: una pista de tempo global + N pistas de notas
    - Las pistas con el mismo nombre entre segmentos se fusionan en una sola
      pista continua (ej: todas las pistas "Melody" se unen en una sola pista)
    - Las pistas sin nombre se identifican por índice de posición en el MIDI
    - Los tiempos se re-escalan si ticks_per_beat difiere entre segmentos
    - El cursor se avanza correctamente usando max(duración pistas) por segmento
    """
    if not sequence:
        print("ERROR: secuencia vacía")
        return False

    # ── Cargar MIDIs ──────────────────────────────────────────────────────────
    midis = []
    for path in sequence:
        if path and os.path.exists(path):
            try:
                midis.append((path, mido.MidiFile(path)))
            except Exception as e:
                print(f"  ⚠ No se pudo cargar {path}: {e}")
        else:
            if path:
                print(f"  ⚠ No encontrado: {path}")

    if not midis:
        print("ERROR: ningún MIDI cargable")
        return False

    # ── Referencia de resolución ──────────────────────────────────────────────
    ref_tpb = midis[0][1].ticks_per_beat

    # ── Primera pasada: inventariar todas las pistas que aparecen ────────────
    # Clave de pista: nombre si existe, si no "track_N" donde N es su posición
    # en el fichero (excluyendo la pista 0 de tempo).
    # El conjunto final de claves determina cuántas pistas tendrá el MIDI final.
    all_track_keys = []   # lista ordenada (para mantener orden de aparición)
    seen_keys      = set()

    for path, mid in midis:
        # Process ALL tracks: the mixer does NOT have a separate tempo-only
        # track 0 — each data track carries its own set_tempo internally.
        for t_idx, track in enumerate(mid.tracks):
            label = _track_label(track)
            # Skip tracks that are purely meta (no note_on events)
            has_notes = any(
                not m.is_meta for m in track
            )
            if not has_notes:
                continue
            key = label if label else f"track_{t_idx}"
            if key not in seen_keys:
                all_track_keys.append(key)
                seen_keys.add(key)

    if verbose:
        print(f"  Pistas detectadas ({len(all_track_keys)}): "
              f"{', '.join(all_track_keys)}")

    # ── Inicializar acumuladores por pista ────────────────────────────────────
    # track_events[key] = lista de (abs_tick_global, msg)
    track_events = {key: [] for key in all_track_keys}

    # ── MIDI de salida ────────────────────────────────────────────────────────
    out = mido.MidiFile(type=1, ticks_per_beat=ref_tpb)

    # Pista 0: solo tempos y metadatos globales
    tempo_track = mido.MidiTrack()
    out.tracks.append(tempo_track)

    # ── Segunda pasada: poblar eventos con offset correcto ───────────────────
    cursor_ticks = 0   # posición global en ticks (ref_tpb)

    for seg_idx, (path, mid) in enumerate(midis):
        seg_name  = Path(path).stem
        seg_tempo = _get_tempo(mid)
        seg_tpb   = mid.ticks_per_beat
        scale     = ref_tpb / seg_tpb if seg_tpb != ref_tpb else 1.0

        # Duración real del segmento: máximo entre pistas (paralelo, no serie)
        seg_duration = int(_seg_duration_ticks(mid) * scale)

        if verbose:
            bpm = round(60_000_000 / seg_tempo)
            n_data = len(mid.tracks) - 1 if len(mid.tracks) > 1 else len(mid.tracks)
            print(f"  [{seg_idx+1}] {seg_name}: {bpm} BPM | "
                  f"{n_data} pistas | {seg_duration/ref_tpb:.1f} beats")

        # Evento de tempo: delta=0 salvo en el primero (que empieza en 0)
        tempo_track.append(
            mido.MetaMessage('set_tempo', tempo=seg_tempo, time=0)
        )

        # Procesar TODAS las pistas (el mixer no tiene pista 0 pura de tempo)
        for t_idx, track in enumerate(mid.tracks):
            has_notes = any(not m.is_meta for m in track)
            if not has_notes:
                continue
            label = _track_label(track)
            key   = label if label else f"track_{t_idx}"

            # Convertir delta → absoluto → re-escalar → desplazar con cursor
            abs_t = 0
            for msg in track:
                abs_t += msg.time
                abs_t_scaled = int(abs_t * scale)
                abs_t_global = abs_t_scaled + cursor_ticks

                if msg.is_meta:
                    if msg.type not in ('set_tempo', 'end_of_track', 'track_name'):
                        track_events[key].append((abs_t_global, msg.copy(time=0)))
                else:
                    track_events[key].append((abs_t_global, msg.copy(time=0)))

        cursor_ticks += seg_duration

    # Cerrar pista de tempo
    tempo_track.append(mido.MetaMessage('end_of_track', time=0))

    # ── Construir pistas de salida ────────────────────────────────────────────
    for key in all_track_keys:
        events = track_events[key]
        if not events:
            continue   # pista vacía: omitir

        track = mido.MidiTrack()
        # Añadir nombre de pista
        track.append(mido.MetaMessage('track_name', name=key, time=0))

        # Ordenar por tiempo absoluto (debería ya estarlo, pero por seguridad)
        events.sort(key=lambda x: x[0])

        # Convertir absolutos → delta
        prev_t = 0
        for abs_t, msg in events:
            delta = max(0, abs_t - prev_t)
            track.append(msg.copy(time=delta))
            prev_t = abs_t

        track.append(mido.MetaMessage('end_of_track', time=0))
        out.tracks.append(track)

    # ── Guardar ───────────────────────────────────────────────────────────────
    out.save(output_path)

    total_beats = cursor_ticks / ref_tpb
    total_bars  = total_beats / 4
    print(f"\n  ✓ MIDI final: {output_path}")
    print(f"    Segmentos : {len(midis)}")
    print(f"    Pistas    : {len(out.tracks) - 1}  ({', '.join(all_track_keys)})")
    print(f"    Duración  : {total_beats:.1f} beats (~{total_bars:.0f} compases aprox.)")

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_tempo_map(s):
    """Parsea "file.mid:BPM, file2.mid:BPM" → dict"""
    result = {}
    if not s:
        return result
    for item in s.split(','):
        item = item.strip()
        if ':' in item:
            parts = item.rsplit(':', 1)
            try:
                result[parts[0].strip()] = float(parts[1].strip())
            except ValueError:
                pass
    return result


def _inject_leitmotif_at_bar(output_mid, bar: int, fragment_midi: str,
                               instrument: str = None, dynamic: str = 'mp',
                               verbose: bool = False):
    """
    Inserta un fragmento MIDI de leitmotif en el compás indicado del MIDI de salida,
    añadiéndolo como una nueva pista con offset temporal correcto.
    Llamado desde main() si se pasa --leitmotif-schedule.
    """
    vel_map = {'pp': 25, 'p': 45, 'mp': 60, 'mf': 75, 'f': 90, 'ff': 110}
    target_vel = vel_map.get(dynamic, 64)
    tpb = output_mid.ticks_per_beat
    insert_tick = bar * tpb * 4  # asume 4/4

    try:
        frag = mido.MidiFile(fragment_midi)
        new_track = mido.MidiTrack()
        track_name = f'leitmotif_{instrument or "motif"}_bar{bar}'
        new_track.append(mido.MetaMessage('track_name', name=track_name, time=0))

        first_note = True
        for track in frag.tracks:
            for msg in track:
                if msg.is_meta:
                    continue
                if first_note:
                    # El primer mensaje lleva el offset absoluto como delta
                    # respecto al inicio del MIDI (tick 0)
                    if hasattr(msg, 'velocity') and msg.velocity > 0:
                        new_track.append(msg.copy(
                            time=insert_tick,
                            velocity=min(msg.velocity, target_vel)
                        ))
                    else:
                        new_track.append(msg.copy(time=insert_tick))
                    first_note = False
                else:
                    if hasattr(msg, 'velocity') and msg.velocity > 0:
                        new_track.append(msg.copy(
                            velocity=min(msg.velocity, target_vel)
                        ))
                    else:
                        new_track.append(msg)

        new_track.append(mido.MetaMessage('end_of_track', time=0))
        output_mid.tracks.append(new_track)
        if verbose:
            print(f"    ✓ Leitmotif inyectado: compás {bar} "
                  f"({instrument or '?'}, {dynamic}) ← {Path(fragment_midi).name}")
    except Exception as e:
        print(f"  ⚠ No se pudo inyectar leitmotif en compás {bar}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='COHERENCE STITCHER — Orquestador de fragmentos MIDI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('inputs', nargs='+',
        help='Fingerprints (.fingerprint.json) o MIDIs (.mid) de los fragmentos')
    parser.add_argument('--fixed-order', action='store_true',
        help='No reordenar; usar el orden dado en la línea de comandos')
    parser.add_argument('--analyze-only', action='store_true',
        help='Solo mostrar análisis y matriz, sin generar nada')
    parser.add_argument('--mixer', default='./midi_dna_unified.py',
        help='Ruta al script midi_dna_unified.py (default: ./midi_dna_unified.py)')
    parser.add_argument('--bridge-sources', nargs='+', default=[],
        metavar='MIDI',
        help='MIDIs fuente para generar los puentes')
    parser.add_argument('--bridge-bars', type=int, default=4,
        help='Compases de cada puente generado (default: 4)')
    parser.add_argument('--bridge-mode', default='emotion',
        choices=['auto','rhythm_melody','harmony_melody',
                 'full_blend','mosaic','energy','emotion'],
        help='Modo del mixer para puentes (default: emotion)')
    parser.add_argument('--min-score', type=float, default=0.55,
        help='Score mínimo para NO generar puente (default: 0.55)')
    parser.add_argument('--tempo-map', default=None,
        help='Overrides de tempo: "file.mid:112, file2.mid:96"')
    parser.add_argument('--no-bridge', action='store_true',
        help='Ensamblar sin puentes (solo concatenar)')
    parser.add_argument('--force-first', type=int, default=None,
        help='Índice del fragmento que debe ir primero (0-based)')
    parser.add_argument('--force-last', type=int, default=None,
        help='Índice del fragmento que debe ir último (0-based)')
    parser.add_argument('--output', default='obra_final.mid',
        help='MIDI de salida (default: obra_final.mid)')
    parser.add_argument('--leitmotif-schedule', default=None,
        metavar='JSON',
        help='Schedule de leitmotifs generado por leitmotif_tracker.py inject '
             '(leitmotif_stitcher.json). Inserta fragmentos de leitmotif en los '
             'compases indicados del MIDI final.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
        help='Mostrar comandos de puentes sin ejecutarlos')

    args = parser.parse_args()

    print("═" * 65)
    print("  COHERENCE STITCHER  v1.0")
    print("  Armado coherente de fragmentos MIDI con fingerprints")
    print("═" * 65)

    # ── Cargar fingerprints ───────────────────────────────────────────────────
    print(f"\n[1/5] Cargando {len(args.inputs)} fingerprint(s)…")
    fingerprints = []
    for inp in args.inputs:
        try:
            fp = load_fingerprint(inp)
            name = Path(fp['meta']['midi_file']).stem
            key  = f"{fp['meta']['key_tonic']} {fp['meta']['key_mode']}"
            bpm  = fp['meta']['tempo_bpm']
            bars = fp['meta']['n_bars']
            print(f"  ✓ {name}  [{key}  {bpm:.0f}BPM  {bars}c]  "
                  f"exit={fp['exit']['chord_roman']}({fp['stitching_hints']['cadence_type']})")
            fingerprints.append(fp)
        except FileNotFoundError as e:
            print(f"  ✗ {inp}: {e}")
        except Exception as e:
            print(f"  ✗ {inp}: Error inesperado: {e}")

    if len(fingerprints) < 2:
        print("\nERROR: se necesitan al menos 2 fragmentos con fingerprint.")
        print("  Genera los fragmentos con: midi_dna_unified.py ... --export-fingerprint")
        sys.exit(1)

    n = len(fingerprints)

    # ── Matriz de compatibilidad ──────────────────────────────────────────────
    print(f"\n[2/5] Calculando matriz de compatibilidad ({n}×{n})…")
    matrix, details_matrix = build_compatibility_matrix(
        fingerprints, verbose=args.verbose
    )
    print_compatibility_matrix(fingerprints, matrix)

    if args.analyze_only:
        print("\n  Modo --analyze-only: terminando aquí.")
        sys.exit(0)

    # ── Ordenación ────────────────────────────────────────────────────────────
    print(f"\n[3/5] Determinando orden óptimo…")
    if args.fixed_order:
        order = list(range(n))
        order_score = sum(matrix[order[i]][order[i+1]] for i in range(n-1))
        print(f"  Orden fijo: {' → '.join(str(i) for i in order)}")
        print(f"  Score total: {order_score:.3f}")
    else:
        order, order_score = find_optimal_order(
            matrix, n,
            force_first=args.force_first,
            force_last=args.force_last
        )
        names = [Path(fp['meta']['midi_file']).stem for fp in fingerprints]
        print(f"  Orden óptimo: {' → '.join(names[i] for i in order)}")
        print(f"  Score total: {order_score:.3f}")

    # Mostrar score de cada transición en el orden elegido
    print("\n  Transiciones:")
    for k in range(len(order) - 1):
        i, j = order[k], order[k+1]
        s = matrix[i][j]
        bar = "✓ OK" if s >= args.min_score else "⚡ PUENTE"
        name_i = Path(fingerprints[i]['meta']['midi_file']).stem
        name_j = Path(fingerprints[j]['meta']['midi_file']).stem
        print(f"    {name_i} → {name_j}:  {s:.3f}  {bar}")

    # ── Generación de puentes ─────────────────────────────────────────────────
    if args.no_bridge:
        print(f"\n[4/5] --no-bridge: omitiendo generación de puentes")
        bridges = {(order[k], order[k+1]): None for k in range(len(order)-1)}
    else:
        print(f"\n[4/5] Generando puentes (umbral: {args.min_score:.2f})…")

        # Verificar mixer
        mixer_path = args.mixer
        if not os.path.exists(mixer_path):
            # Buscar en el mismo directorio que este script
            candidate = Path(__file__).parent / 'midi_dna_unified.py'
            if candidate.exists():
                mixer_path = str(candidate)
            else:
                print(f"  ⚠ Mixer no encontrado: {mixer_path}")
                print(f"    Usa --mixer para especificar la ruta.")
                if not args.dry_run:
                    print(f"    Continuando sin generar puentes.")
                    args.no_bridge = True

        # Fuentes para puentes
        bridge_sources = args.bridge_sources
        if not bridge_sources and not args.no_bridge and not args.dry_run:
            # Intentar usar los MIDIs de los propios fragmentos como fuentes
            bridge_sources = [
                fp['meta']['midi_file']
                for fp in fingerprints
                if os.path.exists(fp['meta']['midi_file'])
            ][:2]
            if bridge_sources:
                print(f"  ℹ Sin --bridge-sources: usando los propios fragmentos como fuente")
                print(f"    {', '.join(Path(s).name for s in bridge_sources)}")

        # Directorio temporal para los puentes
        work_dir = tempfile.mkdtemp(prefix='stitcher_bridges_')

        bridges = {}
        for k in range(len(order) - 1):
            i, j = order[k], order[k+1]
            s = matrix[i][j]
            bridge_info = decide_bridge(fingerprints[i], fingerprints[j],
                                        s, args.min_score)
            if bridge_info['needs_bridge']:
                if bridge_sources:
                    bridge_mid = generate_bridge(
                        fingerprints[i], fingerprints[j],
                        bridge_info, bridge_sources,
                        args.bridge_bars, args.bridge_mode,
                        mixer_path, work_dir,
                        verbose=args.verbose,
                        dry_run=args.dry_run
                    )
                    bridges[(i, j)] = bridge_mid
                else:
                    print(f"  ⚠ Se necesita puente {Path(fingerprints[i]['meta']['midi_file']).stem}"
                          f" → {Path(fingerprints[j]['meta']['midi_file']).stem}"
                          f" (score={s:.2f}) pero no hay --bridge-sources")
                    bridges[(i, j)] = None
            else:
                print(f"  ─ Sin puente: {Path(fingerprints[i]['meta']['midi_file']).stem}"
                      f" → {Path(fingerprints[j]['meta']['midi_file']).stem}"
                      f" (score={s:.2f} ≥ {args.min_score:.2f})")
                bridges[(i, j)] = None

    # ── Ensamblado final ──────────────────────────────────────────────────────
    print(f"\n[5/5] Ensamblando MIDI final…")

    if args.dry_run:
        print("  Modo --dry-run: mostrando secuencia sin ensamblar")
        for k, idx in enumerate(order):
            fp = fingerprints[idx]
            print(f"  [{k+1}] {Path(fp['meta']['midi_file']).stem}")
            if k < len(order) - 1:
                pair = (order[k], order[k+1])
                bridge = bridges.get(pair)
                if bridge:
                    print(f"      ↓  puente: {Path(bridge).name}")
        sys.exit(0)

    # Construir secuencia de MIDIs
    sequence = []
    for k, idx in enumerate(order):
        fp = fingerprints[idx]
        midi_path = fp['meta']['midi_file']
        if os.path.exists(midi_path):
            sequence.append(midi_path)
        else:
            print(f"  ⚠ MIDI no encontrado: {midi_path}")

        if k < len(order) - 1:
            pair = (order[k], order[k+1])
            bridge = bridges.get(pair)
            if bridge and os.path.exists(bridge):
                sequence.append(bridge)

    if not sequence:
        print("ERROR: no hay MIDIs para ensamblar")
        sys.exit(1)

    print(f"  Secuencia ({len(sequence)} ficheros):")
    for s in sequence:
        print(f"    → {Path(s).name}")

    ok = assemble_midi(sequence, args.output, verbose=args.verbose)

    # ── Inyección de leitmotifs ───────────────────────────────────────────────
    if ok and args.leitmotif_schedule and Path(args.leitmotif_schedule).exists():
        print(f"\n  [+] Aplicando leitmotif schedule: {args.leitmotif_schedule}")
        try:
            with open(args.leitmotif_schedule, 'r', encoding='utf-8') as _f:
                lm_sched = json.load(_f)
            injections = lm_sched.get('leitmotif_injections', [])
            if injections:
                # Reabrir el MIDI ensamblado para añadir las pistas de leitmotif
                assembled = mido.MidiFile(args.output)
                for inj in injections:
                    midi_frag = inj.get('midi')
                    if midi_frag and Path(midi_frag).exists():
                        _inject_leitmotif_at_bar(
                            assembled,
                            bar=inj.get('bar', 0),
                            fragment_midi=midi_frag,
                            instrument=inj.get('instrument'),
                            dynamic=inj.get('dynamic', 'mp'),
                            verbose=args.verbose,
                        )
                    else:
                        print(f"  ⚠ Fragmento de leitmotif no encontrado: {midi_frag}")
                assembled.save(args.output)
                print(f"  ✓ {len(injections)} leitmotifs inyectados en {args.output}")
            else:
                print("  ℹ Schedule vacío, sin inyecciones.")
        except Exception as _e:
            print(f"  ⚠ Error leyendo leitmotif schedule: {_e}")

    # Limpiar directorio temporal
    if not args.no_bridge and 'work_dir' in dir():
        try:
            shutil.rmtree(work_dir)
        except Exception:
            pass

    # ── Resumen ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESUMEN FINAL")
    print("═" * 65)
    names_order = [Path(fingerprints[i]['meta']['midi_file']).stem for i in order]
    print(f"  Fragmentos : {' → '.join(names_order)}")
    n_bridges = sum(1 for v in bridges.values() if v is not None)
    print(f"  Puentes    : {n_bridges} generados")
    print(f"  Score medio: {order_score / max(len(order)-1, 1):.3f}")
    print(f"  Salida     : {args.output}")
    print("═" * 65)

    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
