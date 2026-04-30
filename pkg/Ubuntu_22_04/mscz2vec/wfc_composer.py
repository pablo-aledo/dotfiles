#!/usr/bin/env python3
"""
WFC_COMPOSER v3.0 — Wave Function Collapse + Z3
Grafo de ventana temporal con aristas tipadas: MELODIC / HARMONIC / CADENTIAL

FIXES v3 vs v2:
  1. c_no_voice_overlap GLOBAL (no solo aristas HARMONIC)
  2. Distribucion temporal ponderada por hueco disponible
  3. Cobertura minima por frase (AtLeast constraint)
  4. Arco de tension en likelyhood (parabolico por posicion en frase)
  5. Duraciones con no-solapamiento real entre nodos de misma voz

USO:
  python wfc_composer.py
  python wfc_composer.py --key 2 --mode minor --beats 32 --nodes 32 --window 2
  python wfc_composer.py --tempo 80 --seed 42 --verbose

DEPENDENCIAS: z3-solver  pretty_midi  networkx
"""

import z3, random, sys, argparse
from collections import Counter
from datetime import datetime
from enum import IntEnum

try:    import pretty_midi
except ImportError: print("pip install pretty_midi"); sys.exit(1)

try:    import networkx as nx
except ImportError: print("pip install networkx"); sys.exit(1)


# ─── DEBUG ────────────────────────────────────────────────────────────────────
VERBOSE = False
def log(*a):
    if VERBOSE: print("[WFC]", *a)
def info(*a): print("[INFO]", *a)


# ─── ENUMERACIONES ────────────────────────────────────────────────────────────
class EventType(IntEnum):
    REST=0; MELODY=1; HARMONY=2; PASSING=3; CADENCE=4

class Voice(IntEnum):
    BASS=0; TENOR=1; ALTO=2; SOPRANO=3

VOICE_RANGE = {
    Voice.BASS:    (40, 64),
    Voice.TENOR:   (48, 72),
    Voice.ALTO:    (55, 79),
    Voice.SOPRANO: (60, 84),
}
PITCH_REST     = -1
N_VOICES       = 4
DEFAULT_WINDOW = 2

class EdgeType:
    MELODIC   = "mel"
    HARMONIC  = "har"
    CADENTIAL = "cad"


# ─── TEORIA MUSICAL ───────────────────────────────────────────────────────────
def scale_pcs(tonic, mode):
    patterns = {
        "major":          [0,2,4,5,7,9,11],
        "minor":          [0,2,3,5,7,8,10],
        "harmonic_minor": [0,2,3,5,7,8,11],
        "dorian":         [0,2,3,5,7,9,10],
        "phrygian":       [0,1,3,5,7,8,10],
    }
    return set((tonic+s) % 12 for s in patterns.get(mode, patterns["minor"]))

def scale_midi(tonic, mode, lo=21, hi=108):
    pcs = scale_pcs(tonic, mode)
    return [p for p in range(lo, hi+1) if p % 12 in pcs]

def pc_midi(pc, lo=21, hi=108):
    return [p for p in range(lo, hi+1) if p % 12 == pc]

def tension_val(pitch, tonic, mode):
    """
    Tension armónica del pitch en contexto tonal (0=reposo, 10=máxima).
    Mapa por modo basado en la jerarquía de Lerdahl:
      I/i   = 0  (estabilidad máxima)
      V     = 2  (dominante: tensión funcional baja, quiere resolver)
      IV/iv = 4  (subdominante)
      II/ii = 6  (superiónica: tensión media-alta)
      VI/vi = 4  (relativa: consonante)
      III   = 5
      VII/vii = 8 (sensible/subtónica: alta tensión)
      cromatismos = 10
    """
    pc = pitch % 12
    ordered = sorted(scale_pcs(tonic, mode), key=lambda x: (x - tonic) % 12)
    if pc not in ordered: return 10
    deg = ordered.index(pc)
    tmaps = {
        "major":          [0, 6, 5, 4, 2, 4, 8],  # I ii iii IV V vi vii
        "minor":          [0, 6, 5, 4, 2, 4, 7],  # i ii III iv V VI VII(nat)
        "harmonic_minor": [0, 6, 5, 4, 2, 4, 8],  # i ii III iv V VI vii(arm)
        "dorian":         [0, 6, 5, 4, 2, 4, 7],
        "phrygian":       [0, 3, 4, 5, 2, 5, 7],  # i II III iv v VI VII
    }
    tmap = tmaps.get(mode, tmaps["minor"])
    return tmap[deg % len(tmap)]

def note_name(midi):
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return f"{names[midi%12]}{(midi//12)-1}"


# ─── NODOS ────────────────────────────────────────────────────────────────────
def generate_nodes(n):
    return [{
        "name":  f"ev{i:04d}",
        "beat":  z3.Int(f"b{i:04d}"),
        "voice": z3.Int(f"v{i:04d}"),
        "pitch": z3.Int(f"p{i:04d}"),
        "dur":   z3.Int(f"d{i:04d}"),
        "etype": z3.Int(f"e{i:04d}"),
        "phr":   z3.Int(f"ph{i:04d}"),
        "ten":   z3.Int(f"t{i:04d}"),
    } for i in range(n)]


# ─── GRAFO DE VENTANA TEMPORAL ───────────────────────────────────────────────
def build_graph(nodes, cfg, window=DEFAULT_WINDOW):
    """
    Tres tipos de arista con ventanas distintas (W = window):
      MELODIC   |i-j| <= W*N_VOICES   voice leading local
      HARMONIC  |i-j| <= N_VOICES     consonancia simultanea
      CADENTIAL |i-j| <= N_VOICES*2   pre-cadencia de frase
    """
    G     = nx.MultiGraph()
    n     = len(nodes)
    mel_w = window * N_VOICES
    har_w = N_VOICES
    cad_w = N_VOICES * 2

    for i, nd in enumerate(nodes):
        G.add_node(i+1, **nd)
    for i in range(n):
        for j in range(i+1, n):
            d = j - i
            if d <= mel_w: G.add_edge(i+1, j+1, et=EdgeType.MELODIC)
            if d <= har_w: G.add_edge(i+1, j+1, et=EdgeType.HARMONIC)
            if d <= cad_w: G.add_edge(i+1, j+1, et=EdgeType.CADENTIAL)

    counts = {t: sum(1 for _,_,d in G.edges(data=True) if d["et"]==t)
              for t in (EdgeType.MELODIC, EdgeType.HARMONIC, EdgeType.CADENTIAL)}
    log(f"Grafo: {n} nodos, {G.number_of_edges()} aristas W={window}")
    log(f"  MEL={counts[EdgeType.MELODIC]} HAR={counts[EdgeType.HARMONIC]}"
        f" CAD={counts[EdgeType.CADENTIAL]}")
    return G

def etype_edges(G, t):
    for u,v,d in G.edges(data=True):
        if d.get("et") == t: yield u, v

def nbrs_of_type(G, node, t):
    s = set()
    for u,v,d in G.edges(node, data=True):
        if d.get("et") == t:
            s.add(v if u == node else u)
    return list(s)


# ─── CAPA 1: ESTRUCTURAL ──────────────────────────────────────────────────────
def c_domain(solver, G, cfg):
    """
    Dominio + escala integrada.
    pitch pertenece a {REST} union {notas de escala} salvo PASSING.
    """
    NB, NP = cfg["n_beats"], cfg["n_phrases"]
    valid = [PITCH_REST] + scale_midi(cfg["tonic"], cfg["mode"])
    for nd in G.nodes():
        n = G.nodes()[nd]
        solver.add(n["beat"]  >= 0,         n["beat"]  < NB)
        solver.add(n["voice"] >= 0,         n["voice"] < N_VOICES)
        solver.add(n["pitch"] >= PITCH_REST, n["pitch"] <= 108)
        solver.add(n["dur"]   >= 1,         n["dur"]   <= 8)
        solver.add(n["etype"] >= 0,         n["etype"] <= int(EventType.CADENCE))
        solver.add(n["phr"]   >= 0,         n["phr"]   < NP)
        solver.add(n["ten"]   >= 0,         n["ten"]   <= 10)
        solver.add(z3.Or(
            n["etype"] == int(EventType.PASSING),
            z3.Or(*[n["pitch"] == v for v in valid])
        ))

def c_voice_range(solver, G):
    for nd in G.nodes():
        n = G.nodes()[nd]
        for v, (lo, hi) in VOICE_RANGE.items():
            vp = [PITCH_REST] + list(range(lo, hi+1))
            solver.add(z3.Implies(
                n["voice"] == int(v),
                z3.Or(*[n["pitch"] == p for p in vp])
            ))

def c_rest(solver, G):
    for nd in G.nodes():
        n = G.nodes()[nd]
        solver.add(z3.Implies(n["pitch"] == PITCH_REST,
                               z3.And(n["etype"] == int(EventType.REST),
                                      n["ten"] == 0)))
        solver.add(z3.Implies(n["etype"] == int(EventType.REST),
                               n["pitch"] == PITCH_REST))

def c_no_voice_overlap(solver, G):
    """
    FIX: constraint global. Ningún par de nodos puede compartir
    voz+beat, independientemente de su distancia de índice.
    Cada constraint es O(1) (no aritmética), el coste total O(N²)
    es asumible para N<=64.
    """
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            na = G.nodes()[nodes[i]]
            nb = G.nodes()[nodes[j]]
            solver.add(z3.Not(z3.And(
                na["voice"] == nb["voice"],
                na["beat"]  == nb["beat"]
            )))

def c_beat_ordering(solver, G, cfg):
    """
    Ancla beat 0 y beat final, y fuerza diversidad minima entre
    los primeros nodos. El Or de todos los pares O(N2) es demasiado
    costoso para Z3 con N>16; la diversidad se garantiza mejor por
    el no_voice_overlap global y el muestreador ponderado.
    """
    NB = cfg["n_beats"]
    ns = list(G.nodes())
    solver.add(z3.Or(*[G.nodes()[n]["beat"] == 0    for n in ns]))
    solver.add(z3.Or(*[G.nodes()[n]["beat"] == NB-1 for n in ns]))
    # Diversidad minima: primeros min(N_VOICES*2, N) nodos con beats distintos
    k = min(N_VOICES * 2, len(ns))
    for i in range(k):
        for j in range(i+1, k):
            solver.add(G.nodes()[ns[i]]["beat"] != G.nodes()[ns[j]]["beat"])

def c_cadence_end(solver, G, cfg):
    pl = cfg["n_beats"] // cfg["n_phrases"]
    for nd in G.nodes():
        n = G.nodes()[nd]
        solver.add(z3.Implies(
            n["etype"] == int(EventType.CADENCE),
            n["beat"] % pl == pl - 1
        ))

def c_phrase_align(solver, G, cfg):
    """
    phrase_id determinado por beat: Or de asignaciones directas por rango.
    Mucho mas rapido que Implies anidados o division entera en Z3.
    O(N * NP) constraints, cada una simple.
    """
    NB, NP = cfg["n_beats"], cfg["n_phrases"]
    pl = NB // NP
    for nd in G.nodes():
        n = G.nodes()[nd]
        cases = [
            z3.And(n["beat"] >= p * pl,
                   n["beat"] <= (p+1) * pl - 1,
                   n["phr"] == p)
            for p in range(NP)
        ]
        solver.add(z3.Or(*cases))

def c_phrase_coverage(solver, G, cfg):
    """
    FIX: garantiza cobertura mínima por frase.
    Exige que en cada frase haya al menos 1 nodo distinto de REST,
    usando un Or existencial (no AtLeast, que es caro).
    La distribución proporcional se asegura por beat_counts en collapse_phase.
    """
    NP = cfg["n_phrases"]
    nodes = list(G.nodes())
    for p in range(NP):
        # Al menos un nodo activo (no REST) en cada frase
        solver.add(z3.Or(*[
            z3.And(G.nodes()[nd]["phr"] == p,
                   G.nodes()[nd]["etype"] != int(EventType.REST))
            for nd in nodes
        ]))
        # Al menos un nodo en cada mitad de la frase (evita clustering)
        pl = cfg["n_beats"] // NP
        mid = p * pl + pl // 2
        solver.add(z3.Or(*[
            z3.And(G.nodes()[nd]["beat"] >= p * pl,
                   G.nodes()[nd]["beat"] < mid)
            for nd in nodes
        ]))
        solver.add(z3.Or(*[
            z3.And(G.nodes()[nd]["beat"] >= mid,
                   G.nodes()[nd]["beat"] < (p+1) * pl)
            for nd in nodes
        ]))

def c_duration_no_overlap(solver, G, cfg):
    """
    FIX v3: no-solapamiento real por duracion.
    Si nodo A tiene (voice=v, beat=b, dur=d), ningun otro nodo
    en la misma voz puede tener beat en [b, b+d).

    Implementado sobre aristas MELODIC para mantener coste manejable:
    solo pares de nodos dentro de la ventana temporal.
    """
    for u, v in etype_edges(G, EdgeType.MELODIC):
        na, nb = G.nodes()[u], G.nodes()[v]
        # Si na y nb son de la misma voz, beat_nb no puede caer
        # dentro del intervalo de na, ni viceversa.
        for x, y in [(na, nb), (nb, na)]:
            solver.add(z3.Implies(
                z3.And(
                    x["voice"] == y["voice"],
                    y["beat"]  >= x["beat"],
                    y["beat"]  <  x["beat"] + x["dur"],
                    x["pitch"] != PITCH_REST,
                    y["pitch"] != PITCH_REST,
                ),
                z3.Or(
                    y["beat"] == x["beat"],   # mismo beat: handled by no_overlap
                    y["beat"] >= x["beat"] + x["dur"],
                )
            ))


# ─── CAPA 2: ARMONICA ─────────────────────────────────────────────────────────
def c_consonance(solver, G, cfg):
    """Soft: gestionada en likelyhood()."""
    pass

def c_tension_consistency(solver, G, cfg):
    """Soft: gestionada en likelyhood()."""
    pass

def c_tension_arc(solver, G):
    """Dura: CADENCE -> tension baja."""
    for nd in G.nodes():
        n = G.nodes()[nd]
        solver.add(z3.Implies(
            n["etype"] == int(EventType.CADENCE),
            n["ten"] <= 3
        ))

def c_leading_tone(solver, G, cfg):
    """Soft: gestionada en likelyhood()."""
    pass


# ─── CAPA 3: VOICE LEADING ───────────────────────────────────────────────────
def c_voice_crossing(solver, G):
    """Aristas HARMONIC: voz grave -> pitch grave en simultaneidad."""
    for u, v in etype_edges(G, EdgeType.HARMONIC):
        na, nb = G.nodes()[u], G.nodes()[v]
        sim = z3.And(na["beat"] == nb["beat"],
                     na["pitch"] != PITCH_REST,
                     nb["pitch"] != PITCH_REST)
        solver.add(z3.Implies(z3.And(sim, na["voice"] < nb["voice"]),
                               na["pitch"] < nb["pitch"]))
        solver.add(z3.Implies(z3.And(sim, nb["voice"] < na["voice"]),
                               nb["pitch"] < na["pitch"]))

def c_parallel_fifths(solver, G, cfg):
    """Aristas MELODIC: sin 5as/8as paralelas. O(E_mel2)."""
    mel = list(etype_edges(G, EdgeType.MELODIC))
    for i1, (u1,v1) in enumerate(mel):
        for i2, (u2,v2) in enumerate(mel):
            if i2 <= i1: continue
            na, na2 = G.nodes()[u1], G.nodes()[v1]
            nb, nb2 = G.nodes()[u2], G.nodes()[v2]
            cond = z3.And(
                na["voice"]  == na2["voice"],
                nb["voice"]  == nb2["voice"],
                na["voice"]  != nb["voice"],
                na["beat"]   == nb["beat"],
                na2["beat"]  == nb2["beat"],
                na2["beat"]  == na["beat"] + 1,
                na["pitch"]  != PITCH_REST,  nb["pitch"]  != PITCH_REST,
                na2["pitch"] != PITCH_REST,  nb2["pitch"] != PITCH_REST,
            )
            d1 = z3.If(na["pitch"]  >= nb["pitch"],
                       na["pitch"]  -  nb["pitch"],
                       nb["pitch"]  -  na["pitch"])
            d2 = z3.If(na2["pitch"] >= nb2["pitch"],
                       na2["pitch"] -  nb2["pitch"],
                       nb2["pitch"] -  na2["pitch"])
            for p in (0, 7):
                solver.add(z3.Not(z3.And(
                    cond, d1%12 == p, d2%12 == p,
                    z3.Or(
                        z3.And(na2["pitch"]>na["pitch"], nb2["pitch"]>nb["pitch"]),
                        z3.And(na2["pitch"]<na["pitch"], nb2["pitch"]<nb["pitch"]),
                    )
                )))

def c_leap_resolution(solver, G, cfg):
    """Aristas MELODIC: salto >9 semitonos -> movimiento contrario."""
    for u1, v1 in etype_edges(G, EdgeType.MELODIC):
        na, na2 = G.nodes()[u1], G.nodes()[v1]
        con = z3.And(
            na["voice"] == na2["voice"], na2["beat"] == na["beat"] + 1,
            na["pitch"] != PITCH_REST,   na2["pitch"] != PITCH_REST,
        )
        succs = [G.nodes()[nb] for nb in nbrs_of_type(G, v1, EdgeType.MELODIC) if nb != u1]
        if not succs: continue
        cont = z3.Or(*[
            z3.And(
                na3["voice"] == na["voice"],
                na3["beat"]  == na2["beat"] + 1,
                na3["pitch"] != PITCH_REST,
                z3.Or(
                    z3.And(na2["pitch"] > na["pitch"], na3["pitch"] < na2["pitch"]),
                    z3.And(na2["pitch"] < na["pitch"], na3["pitch"] > na2["pitch"]),
                )
            )
            for na3 in succs
        ])
        solver.add(z3.Implies(z3.And(con, z3.Abs(na2["pitch"]-na["pitch"]) > 9), cont))

def c_post_phase1(solver, G, cfg):
    """Limita saltos melodicos entre voces ya conocidas."""
    solver.check(); model = solver.model()
    for u, v in etype_edges(G, EdgeType.MELODIC):
        na, nb = G.nodes()[u], G.nodes()[v]
        try:
            va = model[na["voice"]].as_long(); vb = model[nb["voice"]].as_long()
            ba = model[na["beat"]].as_long();  bb = model[nb["beat"]].as_long()
        except Exception: continue
        if va == vb and abs(ba-bb) == 1:
            solver.add(z3.Implies(
                z3.And(na["pitch"] != PITCH_REST, nb["pitch"] != PITCH_REST),
                z3.Abs(nb["pitch"] - na["pitch"]) <= 16
            ))


# ─── CAPA 4: CADENCIAS ───────────────────────────────────────────────────────
def c_cadence_harmony(solver, G, cfg):
    """CADENCE -> tonica o quinta. Bajo en cadencia -> tonica."""
    tonic  = cfg["tonic"]
    tps    = pc_midi(tonic%12)
    fps    = pc_midi((tonic+7)%12)
    valid  = tps + fps
    bass_t = [p for p in tps
              if VOICE_RANGE[Voice.BASS][0] <= p <= VOICE_RANGE[Voice.BASS][1]]
    for nd in G.nodes():
        n  = G.nodes()[nd]
        ic = n["etype"] == int(EventType.CADENCE)
        solver.add(z3.Implies(
            z3.And(ic, n["pitch"] != PITCH_REST),
            z3.Or(*[n["pitch"] == p for p in valid])
        ))
        if bass_t:
            solver.add(z3.Implies(
                z3.And(ic, n["voice"] == int(Voice.BASS)),
                z3.And(n["pitch"] != PITCH_REST,
                       z3.Or(*[n["pitch"] == p for p in bass_t]))
            ))

def c_pre_cadence(solver, G, cfg):
    """Aristas CADENTIAL: CADENCE -> vecino en beat-1 tiene dominante/subdominante."""
    tonic  = cfg["tonic"]
    pre_ps = pc_midi((tonic+7)%12) + pc_midi((tonic+5)%12)
    for nd in G.nodes():
        na   = G.nodes()[nd]
        ic   = na["etype"] == int(EventType.CADENCE)
        cns  = nbrs_of_type(G, nd, EdgeType.CADENTIAL)
        solver.add(z3.Implies(ic, na["ten"] <= 3))
        if len(cns) >= 2:
            pre = z3.Or(*[
                z3.And(
                    G.nodes()[nb]["beat"]  == na["beat"] - 1,
                    G.nodes()[nb]["pitch"] != PITCH_REST,
                    z3.Or(*[G.nodes()[nb]["pitch"] == p for p in pre_ps])
                )
                for nb in cns
            ])
            solver.add(z3.Implies(ic, pre))


# ─── VEROSIMILITUD ───────────────────────────────────────────────────────────
def likelyhood(model, G, nd, cfg, beat_counts=None):
    """
    Función de aceptación WFC (0-100).

    Criterios:
      1. Tensión tonal del pitch
      2. Consonancia con vecinos HARMONIC (soft, reemplaza c_consonance)
      3. Movimiento melódico y resolución de sensible
      4. Cadencia en posición correcta
      5. Arco de tensión parabólico dentro de la frase
      6. Arco global de tensión entre frases (crece → clímax → resuelve)
      7. Penalización de beats saturados (distribución uniforme)
    """
    n = G.nodes()[nd]
    try:
        pitch = model[n["pitch"]].as_long(); voice = model[n["voice"]].as_long()
        beat  = model[n["beat"]].as_long();  ten   = model[n["ten"]].as_long()
        et    = model[n["etype"]].as_long(); phr   = model[n["phr"]].as_long()
    except Exception: return 50

    score = 60

    # 1. Tensión tonal del pitch
    if pitch != PITCH_REST:
        t = tension_val(pitch, cfg["tonic"], cfg["mode"])
        score += (5 - t) * 2   # tónica/dom: +10, sensible: -6, cromático: -10

    tonic      = cfg["tonic"]
    leading_pc = (tonic - 1) % 12
    tonic_pc   = tonic % 12
    NP         = cfg["n_phrases"]
    pl         = cfg["n_beats"] // NP

    # 2. Consonancia con vecinos HARMONIC (soft)
    for nb in nbrs_of_type(G, nd, EdgeType.HARMONIC):
        on = G.nodes()[nb]
        try:
            ob = model[on["beat"]].as_long(); op = model[on["pitch"]].as_long()
        except Exception: continue
        if ob == beat and op != PITCH_REST and pitch != PITCH_REST:
            ic = abs(pitch - op) % 12
            if ic in (1, 6, 11) and et != int(EventType.PASSING):
                score -= 25
            elif ic in (0, 7):
                score += 8    # 8ª / 5ª justa
            elif ic in (3, 4):
                score += 5    # 3ª
            elif ic in (8, 9):
                score += 3    # 6ª

    # 3. Movimiento melódico y resolución de sensible
    for nb in nbrs_of_type(G, nd, EdgeType.MELODIC):
        on = G.nodes()[nb]
        try:
            ov = model[on["voice"]].as_long(); ob = model[on["beat"]].as_long()
            op = model[on["pitch"]].as_long()
        except Exception: continue
        if ov == voice and ob == beat - 1 and op != PITCH_REST and pitch != PITCH_REST:
            iv = abs(pitch - op)
            if iv == 0:       score -= 5    # unísono: aburrido
            elif iv <= 2:     score += 15   # grado conjunto
            elif iv <= 4:     score += 8    # tercera
            elif iv <= 7:     score += 2    # quinta
            elif iv > 9:      score -= 20   # salto grande
            # Resolución de sensible (soft, reemplaza c_leading_tone)
            if op % 12 == leading_pc and pitch % 12 == tonic_pc:
                score += 22
            elif op % 12 == leading_pc and pitch % 12 != tonic_pc:
                score -= 18

    # 4. Cadencia en posición correcta
    if et == int(EventType.CADENCE):
        score += 20 if (beat + 1) % pl == 0 else -40
    # Premiar MELODY/HARMONY en tiempos fuertes
    if beat % 4 == 0 and et in (int(EventType.MELODY), int(EventType.HARMONY)):
        score += 8

    # 5. Arco de tensión parabólico dentro de la frase
    # Tensión debe crecer hacia el centro y resolver al final
    phrase_pos = (beat % pl) / max(pl - 1, 1)          # 0.0 → 1.0
    ideal_ten  = int(6 * 4 * phrase_pos * (1 - phrase_pos))  # parábola, max=6 en pos=0.5
    ten_error  = abs(ten - ideal_ten)
    score += max(0, 6 - ten_error) * 2                   # hasta +12

    # 6. Arco global de tensión entre frases
    # Frases 0..NP-2: tensión creciente; frase NP-1: resolver
    if NP > 1:
        global_pos = phr / (NP - 1)   # 0.0 → 1.0
        if phr < NP - 1:
            # Frases internas: premiar más tensión a medida que avanzamos
            ideal_global = int(global_pos * 6)
            score += max(0, 4 - abs(ten - ideal_global)) * 2
        else:
            # Última frase: premiar tensión baja (resolución)
            score += max(0, 4 - ten) * 3   # +12 si ten=0

    # 7. Penalizar beats saturados
    if beat_counts is not None:
        ocupacion = beat_counts.get(beat, 0)
        if ocupacion >= N_VOICES:
            score -= 60   # beat lleno: no añadir
        elif ocupacion >= 2:
            score -= ocupacion * 8
        # Bonus por rellenar un beat vacío
        if ocupacion == 0:
            score += 10

    return max(0, min(100, score))

# ─── COLAPSO WFC ─────────────────────────────────────────────────────────────
def collapse_phase(G, solver, cfg, phase, N):
    """
    WFC por lotes con distribución temporal garantizada.

    Fase 1: asigna (beat, voice) rastreando las combinaciones ya usadas,
            garantizando que no haya colisiones voice+beat.
    Fase 2: muestrea pitch/dur/ten con aceptación probabilística.

    Velocidad: solo llama solver.check() una vez por iteración de lote.
    """
    attrs_p2 = ["pitch", "dur", "ten"]

    nodes    = list(G.nodes())
    anchored = set()
    NB = cfg["n_beats"]
    NP = cfg["n_phrases"]
    pl = NB // NP
    BATCH = max(2, N // 5)

    # Rastrear (voice, beat) ya usados para evitar colisiones
    used_vb  = set()    # conjunto de (voice, beat) ya anclados
    beat_counts = Counter()

    for attempt in range(30):
        if solver.check() != z3.sat:
            info(f"  fase {phase} UNSAT iter={attempt}")
            break
        model = solver.model()

        # Actualizar rastreo desde el modelo
        if phase == 1:
            used_vb     = set()
            beat_counts = Counter()
            for nd in anchored:
                n = G.nodes()[nd]
                try:
                    v = model[n["voice"]].as_long()
                    b = model[n["beat"]].as_long()
                    used_vb.add((v, b))
                    beat_counts[b] += 1
                except Exception: pass

        pending = [nd for nd in nodes if nd not in anchored]
        if not pending: break

        # Puntuar nodos pendientes con el modelo actual
        scored = sorted(
            [(likelyhood(model, G, nd, cfg, beat_counts if phase == 1 else None), nd)
             for nd in pending],
            reverse=True
        )

        batch_done = 0
        threshold  = max(15, 60 - attempt * 2)

        for prob, nd in scored:
            if batch_done >= BATCH: break
            if random.randint(0, 100) > min(prob, threshold + 20):
                continue

            n = G.nodes()[nd]

            if phase == 1:
                # Pesos de beat: inversamente proporcionales a ocupación,
                # con sesgo hacia frases sub-pobladas
                phr_counts = Counter(beat_counts.get(b, 0)
                                     for b in range(NB))
                phr_node_counts = Counter()
                for an in anchored:
                    an_n = G.nodes()[an]
                    try: phr_node_counts[model[an_n["phr"]].as_long()] += 1
                    except Exception: pass

                # Elegir voz primero
                cv = model[n["voice"]].as_long()

                # Beats libres para esta voz (sin colisión)
                free_beats = [b for b in range(NB) if (cv, b) not in used_vb]
                if not free_beats:
                    # Voz llena — intentar otra voz
                    for alt_v in range(N_VOICES):
                        free_beats = [b for b in range(NB) if (alt_v, b) not in used_vb]
                        if free_beats:
                            cv = alt_v; break
                if not free_beats:
                    continue   # sin slots disponibles

                # Pesos sobre beats libres
                weights = []
                for b in free_beats:
                    p_idx = b // pl
                    w = 1.0 / (1 + beat_counts.get(b, 0) * 0.4)
                    w *= 1.0 / (1 + phr_node_counts.get(p_idx, 0) * 0.15)
                    weights.append(max(0.01, w))
                total_w = sum(weights)
                weights = [w / total_w for w in weights]
                cb = random.choices(free_beats, weights=weights)[0]
                cp = cb // pl
                ce = model[n["etype"]].as_long()

                # Validar con push/pop antes de anclar definitivamente
                solver.push()
                solver.add(n["beat"]  == cb)
                solver.add(n["voice"] == cv)
                solver.add(n["etype"] == ce)
                solver.add(n["phr"]   == cp)
                if solver.check() != z3.sat:
                    solver.pop()
                    log(f"  [p1] n{nd} UNSAT con b={cb} v={cv}, skip")
                    continue
                solver.pop()
                # Ancla válida: añadir permanentemente
                solver.add(n["beat"]  == cb)
                solver.add(n["voice"] == cv)
                solver.add(n["etype"] == ce)
                solver.add(n["phr"]   == cp)
                used_vb.add((cv, cb))
                beat_counts[cb] += 1
            else:
                # Fase 2: muestreo activo de pitch / dur / tension
                try:
                    voice_val = model[n["voice"]].as_long()
                    etype_val = model[n["etype"]].as_long()
                    beat_val  = model[n["beat"]].as_long()
                    phr_val   = model[n["phr"]].as_long()
                except Exception:
                    voice_val, etype_val, beat_val, phr_val = 3, 1, 0, 0

                if etype_val == int(EventType.REST):
                    pp, pd, pt = PITCH_REST, 1, 0
                else:
                    lo, hi = VOICE_RANGE.get(voice_val, (48, 84))
                    if etype_val == int(EventType.PASSING):
                        cands = list(range(lo, hi+1))
                    else:
                        cands = [p for p in scale_midi(cfg["tonic"], cfg["mode"])
                                 if lo <= p <= hi]
                        if not cands:
                            cands = list(range(lo, hi+1))

                    # Sesgar hacia pitches de baja tensión en cadencias,
                    # y variedad de tensión en frases internas
                    phrase_pos = (beat_val % pl) / max(pl-1, 1)
                    ideal_ten  = int(6 * 4 * phrase_pos * (1-phrase_pos))
                    global_pos = phr_val / max(NP-1, 1)
                    if phr_val == NP - 1:
                        ideal_ten = max(0, ideal_ten - 3)  # resolver última frase

                    # Pesos de pitch por tensión
                    p_weights = []
                    for pc in cands:
                        t = tension_val(pc, cfg["tonic"], cfg["mode"])
                        dist = abs(t - ideal_ten)
                        p_weights.append(max(0.1, 1.0 / (1 + dist)))
                    total_pw = sum(p_weights)
                    p_weights = [w/total_pw for w in p_weights]
                    pp = random.choices(cands, weights=p_weights)[0]

                    # Duración: preferir 1-4 según posición
                    if etype_val == int(EventType.CADENCE):
                        pd = random.choice([2, 2, 4, 4, 8])
                    elif beat_val % 4 == 0:
                        pd = random.choice([1, 2, 2, 4])
                    else:
                        pd = random.choice([1, 1, 1, 2, 2])
                    pt = tension_val(pp, cfg["tonic"], cfg["mode"])

                # Validar con push/pop
                solver.push()
                solver.add(n["pitch"] == pp)
                solver.add(n["dur"]   == pd)
                solver.add(n["ten"]   == pt)
                if solver.check() != z3.sat:
                    solver.pop()
                    # Fallback: tónica de la voz
                    lo, hi = VOICE_RANGE.get(voice_val, (48, 84))
                    tonic_ps = [p for p in pc_midi(cfg["tonic"]%12) if lo<=p<=hi]
                    pp2 = tonic_ps[0] if tonic_ps else (lo + hi)//2
                    solver.push()
                    solver.add(n["pitch"] == pp2)
                    solver.add(n["dur"]   == 1)
                    solver.add(n["ten"]   == 0)
                    if solver.check() != z3.sat:
                        solver.pop()
                        log(f"  [p2] n{nd} UNSAT incluso con fallback, skip")
                        continue
                    solver.pop()
                    solver.add(n["pitch"] == pp2)
                    solver.add(n["dur"]   == 1)
                    solver.add(n["ten"]   == 0)
                else:
                    solver.pop()
                    solver.add(n["pitch"] == pp)
                    solver.add(n["dur"]   == pd)
                    solver.add(n["ten"]   == pt)

            anchored.add(nd)
            batch_done += 1
            log(f"  [p{phase}] n{nd} p={prob}%")

        log(f"  fase {phase} iter={attempt}: +{batch_done}, total={len(anchored)}/{N}")
        if len(anchored) >= N: break
        if batch_done == 0 and attempt >= 12: break

    # Anclar restantes desde el modelo actual
    remaining = [nd for nd in nodes if nd not in anchored]
    if remaining and solver.check() == z3.sat:
        model = solver.model()
        for nd in remaining:
            n = G.nodes()[nd]
            if phase == 1:
                cv = model[n["voice"]].as_long()
                free = [b for b in range(NB) if (cv, b) not in used_vb]
                if not free:
                    for alt_v in range(N_VOICES):
                        free = [b for b in range(NB) if (alt_v, b) not in used_vb]
                        if free: cv = alt_v; break
                if not free: continue
                # Frase con menos nodos
                phr_counts = Counter()
                for an in anchored:
                    an_n = G.nodes()[an]
                    try: phr_counts[model[an_n["phr"]].as_long()] += 1
                    except Exception: pass
                min_phr = min(range(NP), key=lambda p: phr_counts.get(p, 0))
                phr_beats = [b for b in free
                             if min_phr*pl <= b < (min_phr+1)*pl]
                pool = phr_beats if phr_beats else free
                cb = random.choice(pool)
                cp = cb // pl
                ce = model[n["etype"]].as_long()
                solver.add(n["beat"]  == cb)
                solver.add(n["voice"] == cv)
                solver.add(n["etype"] == ce)
                solver.add(n["phr"]   == cp)
                used_vb.add((cv, cb))
                beat_counts[cb] += 1
            else:
                # Muestreo para restantes en fase 2
                try:
                    voice_val = model[n["voice"]].as_long()
                    etype_val = model[n["etype"]].as_long()
                    beat_val  = model[n["beat"]].as_long()
                    phr_val   = model[n["phr"]].as_long()
                except Exception:
                    voice_val, etype_val, beat_val, phr_val = 3, 1, 0, 0
                if etype_val == int(EventType.REST):
                    pp, pd, pt = PITCH_REST, 1, 0
                else:
                    lo, hi = VOICE_RANGE.get(voice_val, (48, 84))
                    cands = ([p for p in scale_midi(cfg["tonic"], cfg["mode"]) if lo<=p<=hi]
                             or list(range(lo, hi+1)))
                    phrase_pos = (beat_val % pl) / max(pl-1, 1)
                    ideal_ten  = int(6 * 4 * phrase_pos * (1-phrase_pos))
                    if phr_val == NP-1: ideal_ten = max(0, ideal_ten-3)
                    p_weights = [max(0.1, 1.0/(1+abs(tension_val(pc, cfg["tonic"], cfg["mode"])-ideal_ten)))
                                 for pc in cands]
                    pp = random.choices(cands, weights=[w/sum(p_weights) for w in p_weights])[0]
                    pd = random.choice([1,1,2,2,4]) if beat_val%4==0 else random.choice([1,1,1,2])
                    pt = tension_val(pp, cfg["tonic"], cfg["mode"])
                solver.add(n["pitch"] == pp)
                solver.add(n["dur"]   == pd)
                solver.add(n["ten"]   == pt)
            anchored.add(nd)

    info(f"Fase {phase}: {len(anchored)}/{N} anclados.")


def collapse_remaining(G, solver):
    if solver.check() != z3.sat:
        info("  collapse_remaining: UNSAT, saltando")
        return
    model = solver.model()
    for nd in G.nodes():
        n = G.nodes()[nd]
        for a in ["beat","voice","pitch","dur","etype","phr","ten"]:
            try: solver.add(n[a] == model[n[a]].as_long())
            except Exception: pass


# ─── EXPORTACION MIDI ─────────────────────────────────────────────────────────
def to_midi(model, G, cfg, path):
    """
    Exporta el modelo a MIDI respetando duraciones reales.
    Las notas con dur>1 se extienden en el tiempo correctamente.
    """
    pm      = pretty_midi.PrettyMIDI(initial_tempo=cfg["tempo"])
    beat_s  = 60.0 / cfg["tempo"]
    progs   = {0:(43,"Bass"), 1:(42,"Cello"), 2:(41,"Viola"), 3:(40,"Violin")}
    insts   = {v: pretty_midi.Instrument(program=p, name=nm)
               for v, (p, nm) in progs.items()}

    # Recoger eventos
    events = []
    for nd in G.nodes():
        n = G.nodes()[nd]
        try:
            pt = model[n["pitch"]].as_long()
            bt = model[n["beat"]].as_long()
            dr = model[n["dur"]].as_long()
            vo = model[n["voice"]].as_long()
            et = model[n["etype"]].as_long()
        except Exception: continue
        if pt == PITCH_REST: continue
        events.append((bt, vo, pt, dr, et))

    # Resolver solapamientos: si dos notas de la misma voz solapan,
    # truncar la anterior para que termine donde empieza la siguiente
    events.sort(key=lambda e: (e[1], e[0]))   # por voz, luego beat
    voice_events = {}
    for bt, vo, pt, dr, et in events:
        voice_events.setdefault(vo, []).append([bt, vo, pt, dr, et])

    for vo, evs in voice_events.items():
        evs.sort(key=lambda e: e[0])
        for i in range(len(evs) - 1):
            bt_cur, _, _, dr_cur, _ = evs[i]
            bt_nxt = evs[i+1][0]
            # Si la nota actual se solapa con la siguiente, truncar
            if bt_cur + dr_cur > bt_nxt:
                evs[i][3] = max(1, bt_nxt - bt_cur)

    # Añadir notas al MIDI
    for vo, evs in voice_events.items():
        for bt, _, pt, dr, et in evs:
            vel = {0:75, 1:68, 2:70, 3:80}.get(vo, 72)
            # Notas PASSING ligeramente más suaves
            if et == int(EventType.PASSING): vel = max(45, vel - 20)
            s = bt * beat_s
            e = max((bt + dr) * beat_s - 0.03, s + 0.08)
            insts[vo].notes.append(
                pretty_midi.Note(velocity=vel, pitch=pt, start=s, end=e))

    for inst in insts.values():
        if inst.notes:
            inst.notes.sort(key=lambda n: n.start)
            pm.instruments.append(inst)
    pm.write(path)
    info(f"MIDI guardado: {path}")


def show_solution(model, G, cfg):
    vn = {0:"BAJO", 1:"TENOR", 2:"ALTO", 3:"SOPRANO"}
    en = {0:"REST", 1:"MEL", 2:"HAR", 3:"PAS", 4:"CAD"}
    print("\n" + "="*72)
    print(f"  SOLUCIÓN — Tónica: {note_name(cfg['tonic']+60)}  Modo: {cfg['mode']}"
          f"  Tempo: {cfg['tempo']} BPM")
    print("="*72)

    evs = []
    for nd in G.nodes():
        n = G.nodes()[nd]
        try:
            evs.append((
                model[n["beat"]].as_long(),  model[n["voice"]].as_long(),
                model[n["pitch"]].as_long(), model[n["dur"]].as_long(),
                model[n["etype"]].as_long(), model[n["ten"]].as_long(),
                model[n["phr"]].as_long(),
            ))
        except Exception: pass
    evs.sort(key=lambda e: (e[6], e[0], e[1]))

    # Estadísticas globales
    non_rest = [e for e in evs if e[2] != PITCH_REST]
    if non_rest:
        avg_ten = sum(e[5] for e in non_rest) / len(non_rest)
        voices_used = set(e[1] for e in non_rest)
        beats_used  = set(e[0] for e in non_rest)
        etypes = Counter(e[4] for e in non_rest)
        print(f"  Nodos activos: {len(non_rest)}/{len(evs)} | "
              f"Beats: {len(beats_used)}/{cfg['n_beats']} | "
              f"Voces: {len(voices_used)} | "
              f"Tensión media: {avg_ten:.1f}")
        print(f"  Tipos: " + "  ".join(f"{en[k]}={v}" for k,v in sorted(etypes.items())))

    # Por frase
    cp = -1
    for beat, voice, pitch, dur, et, ten, phr in evs:
        if phr != cp:
            cp = phr
            phr_evs = [e for e in non_rest if e[6] == phr]
            phr_beats = len(set(e[0] for e in phr_evs))
            phr_ten = (sum(e[5] for e in phr_evs)/len(phr_evs) if phr_evs else 0)
            print(f"\n  ── Frase {phr+1} ({phr_beats} beats activos, "
                  f"tensión media: {phr_ten:.1f}) ──")
        if pitch == PITCH_REST: continue
        ps  = note_name(pitch)
        bar = "▓"*ten + "░"*(10-ten)
        print(f"  b{beat:3d}  {vn[voice]:7s}  {ps:5s}  d={dur}"
              f"  [{en[et]:3s}]  [{bar}] {ten}")
    print("="*72 + "\n")

# ─── ORQUESTACION ─────────────────────────────────────────────────────────────
def build_and_solve(cfg):
    W = cfg.get("window", DEFAULT_WINDOW)
    info(f"Grafo: {cfg['n_nodes']}n {cfg['n_beats']}b {cfg['n_phrases']}fr W={W}")

    G      = build_graph(generate_nodes(cfg["n_nodes"]), cfg, window=W)
    solver = z3.Solver()

    info("Capa 1: estructural...")
    c_domain(solver, G, cfg)
    c_voice_range(solver, G)
    c_rest(solver, G)
    c_no_voice_overlap(solver, G)
    c_beat_ordering(solver, G, cfg)
    c_cadence_end(solver, G, cfg)
    c_phrase_align(solver, G, cfg)
    c_phrase_coverage(solver, G, cfg)

    info("Capa 2: armonica...")
    c_consonance(solver, G, cfg)
    c_tension_consistency(solver, G, cfg)
    c_leading_tone(solver, G, cfg)
    c_tension_arc(solver, G)

    info("Capa 4: cadencias...")
    c_cadence_harmony(solver, G, cfg)
    c_pre_cadence(solver, G, cfg)

    info("Verificando SAT inicial...")
    if solver.check() != z3.sat:
        info("ERROR: UNSAT."); return False
    info("OK.")

    info("WFC Fase 1...")
    collapse_phase(G, solver, cfg, phase=1, N=cfg["n_nodes"])
    if solver.check() != z3.sat:
        info("ERROR: UNSAT tras fase 1."); return False

    info("Duración no-solapamiento (post-fase 1, dur aún libre)...")
    # c_duration_no_overlap se aplica DESPUÉS de anclar beats, antes de pitches
    # En este punto voice+beat están fijados; dur es variable pero acotada a [1,8]
    # La constraint es ahora manejable porque los beats están concretos.
    c_duration_no_overlap(solver, G, cfg)
    if solver.check() != z3.sat:
        info("Advertencia: UNSAT en dur_no_overlap, continuando sin ella...")

    info("Capa 3: voice leading (post-fase 1)...")
    c_post_phase1(solver, G, cfg)
    c_voice_crossing(solver, G)
    c_parallel_fifths(solver, G, cfg)
    c_leap_resolution(solver, G, cfg)
    if solver.check() != z3.sat:
        info("Advertencia: UNSAT en voice leading, continuando...")
    else:
        info("Voice leading OK.")

    info("WFC Fase 2...")
    collapse_phase(G, solver, cfg, phase=2, N=cfg["n_nodes"])
    collapse_remaining(G, solver)

    if solver.check() != z3.sat:
        info("ERROR: UNSAT final."); return False

    model = solver.model()
    info("Solucion encontrada!")
    show_solution(model, G, cfg)
    to_midi(model, G, cfg, cfg["output"])
    return True


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    global VERBOSE
    p = argparse.ArgumentParser(
        description="WFC_COMPOSER v3.0 -- Wave Function Collapse + Z3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument("--key",     type=int, default=2)
    p.add_argument("--mode",    type=str, default="minor",
                   choices=["major","minor","harmonic_minor","dorian","phrygian"])
    p.add_argument("--beats",   type=int, default=16)
    p.add_argument("--phrases", type=int, default=2)
    p.add_argument("--nodes",   type=int, default=20)
    p.add_argument("--tempo",   type=int, default=96)
    p.add_argument("--window",  type=int, default=DEFAULT_WINDOW)
    p.add_argument("--output",  type=str, default="output.mid")
    p.add_argument("--seed",    type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(); VERBOSE = args.verbose

    seed = args.seed if args.seed else int(datetime.now().timestamp())
    random.seed(seed); info(f"Semilla: {seed}")

    cfg = {
        "tonic": args.key%12, "mode": args.mode,
        "n_beats": args.beats, "n_phrases": args.phrases,
        "n_nodes": args.nodes, "tempo": args.tempo,
        "output": args.output, "window": args.window,
    }
    info(f"Tonica={note_name(cfg['tonic']+60)} modo={cfg['mode']} "
         f"beats={cfg['n_beats']} frases={cfg['n_phrases']} "
         f"nodos={cfg['n_nodes']} W={cfg['window']} tempo={cfg['tempo']}")
    sys.exit(0 if build_and_solve(cfg) else 1)

if __name__ == "__main__":
    main()
