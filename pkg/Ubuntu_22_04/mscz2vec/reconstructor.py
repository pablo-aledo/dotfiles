#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        RECONSTRUCTOR  v1.0                                   ║
║     Reconstrucción pedagógica de una pieza completa con las herramientas     ║
║                                                                              ║
║  Dado un MIDI, lo ANALIZA, captura su material (acordes, motivo, clímax) y   ║
║  lo REGENERA con la suite en vez de copiarlo, dejando como rastro la receta  ║
║  (.sh), un plan para runner.py (.yaml) y un documento didáctico (.md).       ║
║  Es el "generador de planes" que le faltaba a runner.py: el flujo            ║
║  análisis → captura → regeneración → ensamblado → expresión → crítica        ║
║  deja de orquestarse a mano.                                                 ║
║                                                                              ║
║  FLUJO (≈27 pasos, todos invocaciones reales de herramientas):               ║
║   análisis    harmonic_analyzer · metric_analyzer · melody_critic · dna      ║
║   captura     midi_split_harmony (acordes reales, melodía separada)          ║
║   armonía     chord_progression_generator · tonicization_engine              ║
║   bajo/textura bass_line_composer · chord_pattern_voicer                     ║
║   melodía     melody_conditioned_v5 (motivo real como semilla, N candidatos) ║
║               + melody_selector (elige con melody_critic)                    ║
║   ensamblado  midi_merge · forma: climax_engineer                            ║
║   expresión   performer · tempo_designer · orchestral_colorist               ║
║   crítica     harmonic/metric/melody · playability · quality · arc · compare ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    plan    MIDI   Analiza y escribe plan.yaml + receta.sh (no ejecuta)       ║
║    run     MIDI   plan + ejecución + documento (todo en uno)                 ║
║    doc     DIR    (Re)genera RECONSTRUCCION.md desde un directorio ya hecho  ║
║    batch   ENTRADAS…  run sobre muchos MIDIs (carpeta, ficheros o globs)     ║
║    profile        Imprime el perfil por defecto (JSON editable, ver abajo)   ║
║    meta    FILE   Reinyecta compás/tonalidad de otro MIDI (paso de seguridad)║
║    ensure  A B…   Copia el primer fichero existente a --out (paso interno)   ║
║                                                                              ║
║  USO:                                                                        ║
║    python reconstructor.py plan  cancion.mid --out-dir rec/cancion           ║
║    python reconstructor.py run   cancion.mid --out-dir rec/cancion           ║
║    python reconstructor.py run   cancion.mid --profile mi_perfil.json        ║
║    python reconstructor.py run   cancion.mid --resume          # reanuda     ║
║    python reconstructor.py batch midis/ --out-root rec --limit 20 --jobs 3   ║
║    python reconstructor.py profile > mi_perfil.json                          ║
║    python runner.py rec/cancion/plan.yaml          # el plan es de runner    ║
║    bash rec/cancion/receta.sh                      # y la receta, autónoma   ║
║                                                                              ║
║  DECISIONES (todas editables con --profile; se documentan en el .md):        ║
║    · tonalidad: key_signature del MIDI; si falta, perfiles de Krumhansl      ║
║    · compás/tempo del original; nº de compases = duración / compás real      ║
║    · voz melódica = (pista,canal) más aguda con peso suficiente              ║
║    · estilo de bajo y de performer según la familia tímbrica (GM)            ║
║    · emoción del colorista según modo y tempo                                ║
║    · patrón de acompañamiento según nº de pulsos del compás                  ║
║    · clímax en la posición temporal del punto más agudo del original         ║
║                                                                              ║
║  OPCIONES (plan/run):                                                        ║
║    --out-dir DIR    Carpeta de la pieza (default: ./<nombre>_rec)            ║
║    --tools-dir DIR  Donde viven las herramientas (default: junto a este .py) ║
║    --profile FILE   JSON parcial que se fusiona sobre el perfil por defecto  ║
║    --candidates N   Candidatos de melodía a generar (default: perfil)        ║
║    --no-critique    Omitir el bloque de crítica final                        ║
║    --strict         Abortar al primer paso fallido (default: continuar)      ║
║    --resume         No repetir pasos ya terminados bien (pipeline_log.json)  ║
║    --from STEP      Empezar en el paso cuyo nombre empieza por STEP          ║
║    --timeout S      Timeout por paso (default: 900)                          ║
║    --dry-run        (run) Mostrar los comandos sin ejecutar                  ║
║                                                                              ║
║  SALIDA (DIR):                                                               ║
║    original.mid · obra_base.mid · obra_climax.mid · obra_perf.mid ·          ║
║    obra_final.mid (resultado) · plan.yaml · receta.sh ·                      ║
║    reconstruction.json (decisiones) · pipeline_log.json (resultado de cada   ║
║    paso) · RECONSTRUCCION.md · analisis/*.log · material/ (stems, JSON)      ║
║                                                                              ║
║  REQUIERE: herramientas con los arreglos de compás/metadatos (mutopia_fixes) ║
║  DEPENDENCIAS: mido · stdlib (numpy no hace falta)                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import copy
import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: requiere mido (pip install mido)")
    sys.exit(1)

VERSION = "1.0"
SELF = os.path.abspath(__file__)
PY = sys.executable or "python3"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MIDO_MAJOR = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_MIDO_MINOR = ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm']


def _c(k: str) -> str:
    return _COLORS.get(k, "") if _USE_COLOR else ""


# ══════════════════════════════════════════════════════════════════════════════
#  PERFIL (todas las reglas de decisión, editables)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_PROFILE: Dict = {
    "_ayuda": "Fusiona sobre este perfil cualquier subconjunto: solo hay que "
              "indicar las claves que se quieran cambiar.",
    # familia tímbrica GM del original → [estilo de performer, estilo de bajo]
    "family_styles": {
        "organo": ["mecanico", "pedal"],
        "ensemble": ["mecanico", "pedal"],
        "cuerdas": ["romantico", "classical"],
        "viento_madera": ["romantico", "classical"],
        "viento": ["romantico", "classical"],
        "metal": ["romantico", "classical"],
        "guitarra": ["clasico", "classical"],
        "piano": ["clasico", "classical"],
        "default": ["clasico", "classical"],
    },
    # emoción para orchestral_colorist
    "emotion": {"slow_bpm": 90, "minor_slow": "melancolia", "minor_fast": "tension",
                "major_slow": "serenidad", "major_fast": "triunfo"},
    # patrón de chord_pattern_voicer según pulsos (negras) por compás
    "pattern": {"bpb_ge_4": "alberti", "bpb_ge_3": "arpeggio-up", "default": "block"},
    "melody": {"engine": "motif", "fallback_engine": "chord_guided", "candidates": 4,
               "seed": 5, "profile_major": "pastoral", "profile_minor": "elegiac",
               "rhythm": "flowing", "contour": "arch", "range": [60, 84],
               "motif_notes": 8},
    "tonicization": {"density": 0.15, "min_gap": 4, "seventh": True, "seed": 3},
    "bass": {"seed": 11},
    "climax": {"width": 1, "hollow_bars": 1, "intensity": 0.5},
    "performer": {"intensity": 0.5},
    "tempo": {"section_factors": [0.96, 1.04, 0.96], "final_rit": 2,
              "final_rit_factor": 1.2, "shape": "smooth"},
    "cpg": {"seed": 7, "tension_profile": "arch"},
    "max_bars": 200,
    "min_tempo": 30,
    "max_tempo": 300,
    "critique": True,
}


def deep_merge(base: Dict, over: Dict) -> Dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_profile(path: Optional[str]) -> Dict:
    if not path:
        return copy.deepcopy(DEFAULT_PROFILE)
    with open(path, encoding="utf-8") as fh:
        return deep_merge(DEFAULT_PROFILE, json.load(fh))


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DEL MIDI DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _corr(a, b) -> float:
    ma, mb = sum(a) / 12, sum(b) / 12
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def detect_key(pcs: List[int]) -> Tuple[int, str]:
    """Krumhansl–Schmuckler sobre el histograma de clases de altura."""
    h = [0.0] * 12
    for p in pcs:
        h[p % 12] += 1
    best = (-2.0, 0, "major")
    for root in range(12):
        for mode, prof in (("major", _MAJ), ("minor", _MIN)):
            rot = [prof[(i - root) % 12] for i in range(12)]
            s = _corr(h, rot)
            if s > best[0]:
                best = (s, root, mode)
    return best[1], best[2]


def key_name(root_pc: int, mode: str) -> str:
    """Nombre corto compatible con las herramientas (C, Bb, F#m…)."""
    return (_MIDO_MINOR if mode == "minor" else _MIDO_MAJOR)[root_pc % 12]


def parse_mido_key(k: str) -> Tuple[str, str]:
    minor = k.endswith("m")
    return (k[:-1] if minor else k), ("minor" if minor else "major")


def safe_time_sig(n: int, d: int) -> str:
    return f"{n}/{d}" if 0 < n <= 64 and d in (1, 2, 4, 8, 16, 32) else "4/4"


def program_family(programs: List[int]) -> str:
    """Familia tímbrica a partir del programa GM predominante."""
    if not programs:
        return "piano"
    p = Counter(programs).most_common(1)[0][0]
    if p <= 7:
        return "piano"
    if p <= 15:
        return "piano"            # percusión cromática ≈ teclado/mallets
    if p <= 23:
        return "organo"
    if p <= 31:
        return "guitarra"
    if p <= 47:
        return "cuerdas"
    if p <= 55:
        return "ensemble"
    if p <= 63:
        return "metal"
    if p <= 71:
        return "viento_madera"
    if p <= 79:
        return "viento"
    return "piano"


def analyze_midi(path: str, profile: Dict) -> Dict:
    mf = mido.MidiFile(path)
    tpb = mf.ticks_per_beat or 480
    tempo, ts, key_sigs = None, None, []
    ts_all: List[str] = []
    voices: Dict[Tuple[int, int], Dict] = {}
    progs: Dict[Tuple[int, int], List[int]] = {}
    end_ticks = 0
    for ti, tr in enumerate(mf.tracks):
        t = 0
        cur_prog: Dict[int, int] = {}
        for m in tr:
            t += m.time
            if m.type == "set_tempo" and tempo is None:
                tempo = 60_000_000 / m.tempo
            elif m.type == "time_signature":
                if ts is None:
                    ts = (m.numerator, m.denominator)
                s_ = f"{m.numerator}/{m.denominator}"
                if s_ not in ts_all:
                    ts_all.append(s_)
            elif m.type == "key_signature":
                key_sigs.append(m.key)
            elif m.type == "program_change":
                cur_prog[m.channel] = m.program
            elif m.type == "note_on" and m.velocity > 0 and m.channel != 9:
                v = voices.setdefault((ti, m.channel), {"notes": []})
                v["notes"].append((t / tpb, m.note))
                progs.setdefault((ti, m.channel), []).append(cur_prog.get(m.channel, 0))
        end_ticks = max(end_ticks, t)
    tempo = tempo or 120.0
    ts_s = safe_time_sig(*(ts or (4, 4)))
    n, d = (int(x) for x in ts_s.split("/"))
    bpb = n * 4.0 / d                          # compás en NEGRAS (N·4/D)
    total_q = end_ticks / tpb
    n_bars = max(1, int(round(total_q / bpb)))

    # voces
    vinfo = []
    for (ti, ch), v in voices.items():
        ps = [p for _, p in v["notes"]]
        vinfo.append({"track": ti, "channel": ch, "n": len(ps),
                      "median": float(median(ps)), "lo": min(ps), "hi": max(ps)})
    melody = None
    if vinfo:
        nmax = max(v["n"] for v in vinfo)
        eligible = [v for v in vinfo if v["n"] >= max(8, 0.15 * nmax)] or vinfo
        melody = max(eligible, key=lambda v: (v["median"], v["n"]))

    all_pcs = [p for v in voices.values() for _, p in v["notes"]]
    if key_sigs:
        tonic, mode = parse_mido_key(key_sigs[0])
        key_src = "key_signature del MIDI"
    elif all_pcs:
        r, mode = detect_key(all_pcs)
        tonic = NOTE_NAMES[r] if "#" not in NOTE_NAMES[r] else _MIDO_MAJOR[r]
        key_src = "detección Krumhansl (el MIDI no trae armadura)"
    else:
        tonic, mode, key_src = "C", "major", "por defecto"
    key_short = tonic + ("m" if mode == "minor" else "")
    key_long = f"{tonic} {mode}"

    fam_progs = progs.get((melody["track"], melody["channel"]), []) if melody else []
    family = program_family([p for ps in progs.values() for p in ps] or fam_progs)

    # motivo + clímax de la voz melódica
    motif, climax = None, 0.75
    if melody:
        mnotes = sorted(voices[(melody["track"], melody["channel"])]["notes"])
        motif = extract_motif(mnotes, profile["melody"]["motif_notes"],
                              profile["melody"]["range"])
        if len(mnotes) > 1:
            hi = max(p for _, p in mnotes)
            t_hi = next(t for t, p in mnotes if p == hi)
            span = max(1e-6, mnotes[-1][0] - mnotes[0][0])
            climax = round((t_hi - mnotes[0][0]) / span, 2)

    return {
        "file": os.path.abspath(path), "tpb": tpb, "tempo": round(tempo, 2),
        "time_sig": ts_s, "time_sigs_all": ts_all, "bpb": bpb, "total_quarters": round(total_q, 2),
        "n_bars": n_bars, "key_short": key_short, "tonic": tonic, "mode": mode,
        "key_long": key_long, "key_source": key_src, "family": family,
        "voices": vinfo, "melody_voice": melody, "motif": motif, "climax": climax,
        "n_notes": sum(v["n"] for v in vinfo),
    }


def extract_motif(notes: List[Tuple[float, int]], max_notes: int,
                  rng: List[int]) -> Optional[str]:
    """Skyline de las primeras notas → 'C4:0.5 E4:1.0 …' (transportado a `rng`)."""
    if len(notes) < 4:
        return None
    groups: "OrderedDict[float, int]" = OrderedDict()
    for t, p in notes:
        slot = round(t * 2) / 2.0
        if slot not in groups or p > groups[slot]:
            groups[slot] = p
    items = list(groups.items())[:max_notes]
    if len(items) < 3:
        return None
    # transporta por octavas para que la mediana caiga dentro del ámbito pedido
    med = median(p for _, p in items)
    centre = (rng[0] + rng[1]) / 2
    shift = 12 * round((centre - med) / 12)
    parts = []
    for j, (t, p) in enumerate(items):
        nxt = items[j + 1][0] if j + 1 < len(items) else t + 1.0
        dur = max(0.25, min(nxt - t, 4.0))
        q = max(0, min(127, p + shift))
        parts.append(f"{NOTE_NAMES[q % 12]}{q // 12 - 1}:{round(dur, 2)}")
    return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  DECISIONES
# ══════════════════════════════════════════════════════════════════════════════

def decide(info: Dict, prof: Dict, candidates: Optional[int]) -> Dict:
    d: List[Dict] = []

    def add(what, value, why):
        d.append({"que": what, "valor": value, "porque": why})

    tempo = max(prof["min_tempo"], min(prof["max_tempo"], info["tempo"]))
    n_bars = min(info["n_bars"], prof["max_bars"])
    add("tonalidad", info["key_long"], info["key_source"])
    add("compás", info["time_sig"],
        f"del original; un compás dura {info['bpb']:g} negras")
    add("tempo", f"{round(tempo)} BPM", "del original (acotado a "
        f"{prof['min_tempo']}–{prof['max_tempo']})")
    add("compases", n_bars,
        f"{info['total_quarters']:g} negras de música / {info['bpb']:g} negras por compás")
    if len(info.get("time_sigs_all", [])) > 1:
        add("cambios de compás", " → ".join(info["time_sigs_all"]),
            "el original cambia de compás; se regenera en el primero y se conserva la "
            "DURACIÓN total (los compases contados por metric_analyzer pueden diferir)")

    fam = info["family"]
    perf_style, bass_style = prof["family_styles"].get(fam, prof["family_styles"]["default"])
    add("estilo performer", perf_style, f"familia tímbrica detectada: {fam}")
    add("estilo bajo", bass_style, f"familia tímbrica detectada: {fam}")

    e = prof["emotion"]
    slow = tempo < e["slow_bpm"]
    emotion = e[("minor_" if info["mode"] == "minor" else "major_") +
                ("slow" if slow else "fast")]
    add("emoción (colorista)", emotion,
        f"modo {info['mode']} y tempo {'lento' if slow else 'rápido'} "
        f"(umbral {e['slow_bpm']} BPM)")

    pt = prof["pattern"]
    pattern = pt["bpb_ge_4"] if info["bpb"] >= 4 else (
        pt["bpb_ge_3"] if info["bpb"] >= 3 else pt["default"])
    add("patrón de acompañamiento", pattern, f"compás de {info['bpb']:g} negras")

    mel = prof["melody"]
    engine = mel["engine"] if info["motif"] else mel["fallback_engine"]
    add("motor de melodía", engine,
        "semilla = motivo real de la voz melódica" if info["motif"]
        else "no se pudo extraer motivo: melodía guiada por acordes")
    if info["motif"]:
        add("motivo semilla", info["motif"], "primeras notas de la voz melódica (skyline)")
    add("clímax", f"{int(info['climax'] * 100)}%",
        "posición temporal del punto más agudo del original")
    n_cand = candidates or mel["candidates"]
    add("candidatos de melodía", n_cand, "se elige el mejor con melody_selector (melody_critic)")

    return {"tempo": tempo, "tempo_i": int(round(tempo)), "n_bars": n_bars,
            "perf_style": perf_style, "bass_style": bass_style, "emotion": emotion,
            "pattern": pattern, "engine": engine, "candidates": n_cand,
            "decisions": d}


# ══════════════════════════════════════════════════════════════════════════════
#  PLAN DE PASOS
# ══════════════════════════════════════════════════════════════════════════════
# Cada paso: dict(name, stage, tool, args, note, informative).
# En `args`, {D} = carpeta de la pieza y {T} = carpeta de herramientas.

def build_steps(info: Dict, dec: Dict, prof: Dict, critique: bool) -> List[Dict]:
    S: List[Dict] = []
    tonic, mode, ks, kl = info["tonic"], info["mode"], info["key_short"], info["key_long"]
    ts, T = info["time_sig"], dec["tempo_i"]
    n_bars, bpb = dec["n_bars"], info["bpb"]
    mv = info["melody_voice"]
    mtrack = mv["track"] if mv else 0
    orig = "{D}/original.mid"
    mat = "{D}/material"

    def st(name, stage, tool, args, note, informative=False, internal=False):
        S.append({"name": name, "stage": stage, "tool": tool,
                  "args": [str(a) for a in args], "note": note,
                  "informative": informative, "internal": internal})

    # ── análisis ──
    st("01_armonia_orig", "análisis", "harmonic_analyzer.py",
       [orig, "--key", ks, "--json", f"{mat}/chords_orig.json", "--max-bars", 24],
       "Armonía funcional del original en su tonalidad real")
    st("02_metrica_orig", "análisis", "metric_analyzer.py",
       [orig, "--json", f"{mat}/metric_orig.json"], "Compás, síncopa, nPVI, hipermetro")
    st("03_melodia_orig", "análisis", "melody_critic.py",
       [orig, "--track", mtrack, "--json", f"{mat}/critica_orig.json"],
       f"Crítica de la voz melódica (pista {mtrack})", informative=True)
    st("04_dna_orig", "análisis", "midi_dna_unified.py",
       [orig, "--fingerprint-only", "--output", f"{mat}/original.fingerprint.json"],
       "Huella vectorial (ADN) del original")
    # ── captura ──
    st("05_split", "captura", "midi_split_harmony.py",
       [orig, "--out-dir", mat, "--split-files", "--key", kl, "--tempo", T, "--report"],
       "Separación melodía/armonía y extracción de acordes reales (HMM)")
    chords_split = f"{mat}/original_chords.mid"
    melody_split = f"{mat}/original_melody.mid"
    # ── armonía ──
    intent = (f"pieza en {kl} de carácter {dec['emotion']}, compás {ts}, "
              f"con arco expresivo hacia un clímax tardío")
    st("06_cpg", "armonía", "chord_progression_generator.py",
       [intent, "--key", tonic, "--mode", mode, "--style", "auto", "--bars", n_bars,
        "--beats", max(1, int(round(bpb))), "--tempo", T,
        "--tension-profile", prof["cpg"]["tension_profile"], "--seed", prof["cpg"]["seed"],
        "--export-midi", f"{mat}/cpg.mid", "--export-text", f"{mat}/cpg.chords.txt",
        "--export-json", f"{mat}/cpg.json"],
       "Alternativa generativa: progresión desde la intención (exploración)",
       informative=True)
    st("07_tonicizacion_report", "armonía", "tonicization_engine.py",
       [chords_split, "--key", ks, "--report"],
       "¿Dónde caben dominantes secundarias?", informative=True)
    tn = prof["tonicization"]
    targs = [chords_split, "--key", ks, "--density", tn["density"], "--min-gap",
             tn["min_gap"]] + (["--seventh"] if tn["seventh"] else []) + \
            ["--seed", tn["seed"], "--output", f"{mat}/chords_ton.mid"]
    st("08_tonicizacion", "armonía", "tonicization_engine.py", targs,
       "Inserta V7/x sobre la armonía real extraída", informative=True)
    st("08b_base_armonica", "armonía", None,
       ["ensure", f"{mat}/chords_ton.mid", chords_split, f"{mat}/cpg.mid",
        "--out", f"{mat}/chords_base.mid"],
       "Base armónica: tonicizada; si falla, la extraída; si no, la generada",
       internal=True)
    base = f"{mat}/chords_base.mid"
    # ── bajo / textura ──
    st("09_bajo", "bajo", "bass_line_composer.py",
       ["", "--key", tonic, "--mode", mode, "--bars", n_bars, "--tempo", T,
        "--time-sig", ts, "--style", dec["bass_style"], "--chords-midi", base,
        "--seed", prof["bass"]["seed"], "--output", f"{mat}/bajo.mid"],
       f"Línea de bajo autónoma ({dec['bass_style']}) sobre la armonía")
    st("10_patron", "textura", "chord_pattern_voicer.py",
       [base, "--pattern", dec["pattern"], "--out", f"{mat}/acomp.mid", "--report"],
       f"Refiguración del patrón armónico ({dec['pattern']})", informative=True)
    st("10b_acomp", "textura", None,
       ["ensure", f"{mat}/acomp.mid", base, "--out", f"{mat}/acomp_ok.mid"],
       "Si el voicer falla, se usa la armonía en bloque", internal=True)
    # ── melodía ──
    m = prof["melody"]
    margs = ["--engine", dec["engine"]]
    if dec["engine"] == "motif" and info["motif"]:
        margs += ["--motif", info["motif"]]
    margs += ["--key", tonic, "--mode", mode, "--bars", n_bars, "--time-sig", ts,
              "--tempo", T,
              "--profile", m["profile_major"] if mode == "major" else m["profile_minor"],
              "--rhythm", m["rhythm"], "--contour", m["contour"],
              "--range", m["range"][0], m["range"][1],
              "--chords-midi", base, "--chords-midi-window", max(1, int(bpb)),
              "--candidates", dec["candidates"], "--seed", m["seed"],
              "--output", f"{mat}/cand/mel.mid"]
    st("11_melodia", "melodía", "melody_conditioned_v5.py", margs,
       f"{dec['candidates']} candidatos de melodía "
       f"({'sembrados con el motivo real' if dec['engine'] == 'motif' else 'guiados por acordes'})")
    st("12_seleccion", "melodía", "melody_selector.py",
       ["pick", f"{mat}/cand/*.mid", "--track", 1, "--output", f"{mat}/melodia_sel.mid",
        "--json", f"{mat}/seleccion_melodia.json"],
       "Elige el candidato mejor puntuado por melody_critic", informative=True)
    st("12b_melodia", "melodía", None,
       ["ensure", f"{mat}/melodia_sel.mid", f"{mat}/cand/mel.mid", melody_split,
        "--out", f"{mat}/melodia.mid"],
       "Melodía final: la elegida; si no, el primer candidato; si no, la separada",
       internal=True)
    # ── ensamblado / forma / expresión ──
    stems = [f"{mat}/bajo.mid", f"{mat}/acomp_ok.mid", f"{mat}/melodia.mid"]
    st("13_merge", "ensamblado", "midi_merge.py",
       stems + ["--output", "{D}/obra_base.mid", "--channels"] +
       [f"{p}:{i}" for i, p in enumerate(stems)],
       "Stems (bajo · acompañamiento · melodía) → MIDI multicanal")
    cl = prof["climax"]
    st("14_climax", "forma", "climax_engineer.py",
       ["{D}/obra_base.mid", "--output", "{D}/obra_climax.mid", "--position",
        info["climax"], "--width", cl["width"], "--hollow-bars", cl["hollow_bars"],
        "--intensity", cl["intensity"], "--melody-track", 3],
       f"Refuerzo del clímax en el {int(info['climax'] * 100)}% (posición real)",
       informative=True)
    st("14b_etapa1", "forma", None,
       ["ensure", "{D}/obra_climax.mid", "{D}/obra_base.mid", "--out", "{D}/obra_etapa1.mid"],
       "Continúa con el clímax o, si falló, con el ensamblado", internal=True)
    st("15_performer", "expresión", "performer.py",
       ["{D}/obra_etapa1.mid", "--output", "{D}/obra_perf.mid", "--style", dec["perf_style"],
        "--intensity", prof["performer"]["intensity"], "--melody-track", 3],
       f"Dinámicas, pedal y fraseo ({dec['perf_style']})", informative=True)
    st("15b_etapa2", "expresión", None,
       ["ensure", "{D}/obra_perf.mid", "{D}/obra_etapa1.mid", "--out", "{D}/obra_etapa2.mid"],
       "Continúa con la interpretación o, si falló, con la etapa anterior", internal=True)
    tp = prof["tempo"]
    f1, f2, f3 = tp["section_factors"]
    mid_bar, late_bar = max(1, n_bars // 2), max(1, int(n_bars * 0.8))
    st("16_tempo", "expresión", "tempo_designer.py",
       ["{D}/obra_etapa2.mid", "--output", "{D}/obra_final.mid", "--base", dec["tempo"],
        "--section", f"1:{round(dec['tempo'] * f1)}",
        "--section", f"{mid_bar}:{round(dec['tempo'] * f2)}",
        "--section", f"{late_bar}:{round(dec['tempo'] * f3)}",
        "--final-rit", tp["final_rit"], "--final-rit-factor", tp["final_rit_factor"],
        "--shape", tp["shape"]],
       "Arco agógico con ritardando final", informative=True)
    st("16b_final", "expresión", None,
       ["ensure", "{D}/obra_final.mid", "{D}/obra_etapa2.mid", "--out", "{D}/obra_final.mid"],
       "Garantiza obra_final.mid", internal=True)
    st("17_meta", "expresión", None,
       ["meta", "{D}/obra_final.mid", "--from", orig, "--out", "{D}/obra_final.mid"],
       "Red de seguridad: compás y armadura del original en el MIDI final",
       internal=True)
    if not critique:
        return S
    # ── color + crítica ──
    st("18_color", "color", "orchestral_colorist.py",
       ["{D}/obra_final.mid", "--emotion", dec["emotion"], "--report"],
       f"Reasignación tímbrica propuesta para «{dec['emotion']}»", informative=True)
    st("19_armonia_final", "crítica", "harmonic_analyzer.py",
       ["{D}/obra_final.mid", "--key", ks, "--json", f"{mat}/chords_final.json",
        "--max-bars", 24], "Re-análisis armónico del resultado", informative=True)
    st("20_metrica_final", "crítica", "metric_analyzer.py",
       ["{D}/obra_final.mid", "--json", f"{mat}/metric_final.json"],
       "Métrica del resultado", informative=True)
    st("21_melodia_final", "crítica", "melody_critic.py",
       ["{D}/obra_final.mid", "--track", 3, "--json", f"{mat}/critica_final.json"],
       "Crítica de la melodía final", informative=True)
    st("22_tocabilidad", "crítica", "playability_auditor.py",
       ["{D}/obra_base.mid", "--quiet"], "Auditoría de tocabilidad", informative=True)
    st("23_calidad", "crítica", "quality_scorer.py",
       ["{D}/obra_final.mid", "--fast"], "Calidad en 5 dimensiones", informative=True)
    q = max(1, n_bars // 4)
    st("24_arco", "crítica", "arc_supervisor.py",
       ["{D}/obra_final.mid", "--sections", q, q, q, q, "--tension", "0.3,0.5,0.7,0.9"],
       "¿Coincide el arco real con el planificado?", informative=True)
    st("25_export_yaml", "exportación", "midi2yaml.py",
       [f"{mat}/melodia.mid", "--compas", ts, "--bpm", dec["tempo_i"], "--track", 1],
       "Serialización didáctica de la melodía a partitura.yaml", informative=True)
    st("26_compare", "crítica", "analyzer.py",
       ["compare", orig, "{D}/obra_final.mid"],
       "Comparación vectorial original ↔ reconstrucción", informative=True)
    return S


# ── render de comandos ───────────────────────────────────────────────────────

def step_argv(step: Dict, D: str, T: str) -> List[str]:
    """argv concreto (para ejecutar / plan.yaml)."""
    rep = lambda s: s.replace("{D}", D).replace("{T}", T)  # noqa: E731
    args = [rep(a) for a in step["args"]]
    if step["tool"] is None:                       # subcomando interno
        return [PY, SELF] + args
    return [PY, os.path.join(T, step["tool"])] + args


def step_shell(step: Dict) -> str:
    """Línea para receta.sh, con $D y $TOOLS (portable)."""
    def q(tok: str) -> str:
        if "{D}" in tok:
            return '"' + tok.replace("{D}", "$D").replace('"', '\\"') + '"'
        return shlex.quote(tok)
    toks = [q(a) for a in step["args"]]
    if step["tool"] is None:
        return f'python3 "$SELF" ' + " ".join(toks)
    return f'python3 "$TOOLS/{step["tool"]}" ' + " ".join(toks)


def write_plan_files(D: str, T: str, steps: List[Dict], info: Dict, dec: Dict) -> None:
    # plan.yaml (formato "manual" de runner.py: name + cmd)
    lines = ["# Plan generado por reconstructor.py — ejecutable con runner.py",
             f"# Pieza: {os.path.basename(info['file'])}", "steps:"]
    for s in steps:
        cmd = " ".join(shlex.quote(a) for a in step_argv(s, D, T))
        lines += [f"  - name: {s['name']}", f"    label: {json.dumps(s['note'], ensure_ascii=False)}",
                  f"    cmd: {json.dumps(cmd, ensure_ascii=False)}"]
    Path(D, "plan.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # receta.sh (autónoma)
    sh = ["#!/usr/bin/env bash",
          "# " + "═" * 74,
          f"#  RECETA de reconstrucción — {os.path.basename(info['file'])}",
          "#  Generada por reconstructor.py. Variables sobreescribibles:",
          "#    TOOLS = carpeta con las herramientas · SELF = ruta de reconstructor.py",
          "# " + "═" * 74,
          "set -u",
          'D="$(cd "$(dirname "$0")" && pwd)"',
          f'TOOLS="${{TOOLS:-{T}}}"', f'SELF="${{SELF:-{SELF}}}"',
          'mkdir -p "$D/analisis" "$D/material/cand"', ""]
    stage = None
    for s in steps:
        if s["stage"] != stage:
            stage = s["stage"]
            sh += [f"# ── {stage.upper()} " + "─" * (60 - len(stage))]
        sh += [f"# {s['name']} — {s['note']}", step_shell(s) +
               ("   || echo \"(aviso: {0} terminó con error)\"".format(s["name"])
                if s["informative"] or s["internal"] else ""), ""]
    Path(D, "receta.sh").write_text("\n".join(sh), encoding="utf-8")
    os.chmod(Path(D, "receta.sh"), 0o755)


# ══════════════════════════════════════════════════════════════════════════════
#  EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def run_steps(D: str, T: str, steps: List[Dict], timeout: int, strict: bool,
              resume: bool, start: Optional[str], dry: bool) -> List[Dict]:
    log_path = os.path.join(D, "pipeline_log.json")
    prev: Dict[str, Dict] = {}
    if resume and os.path.exists(log_path):
        try:
            prev = {s["name"]: s for s in json.load(open(log_path))["steps"]}
        except Exception:  # noqa: BLE001
            prev = {}
    os.makedirs(os.path.join(D, "analisis"), exist_ok=True)
    os.makedirs(os.path.join(D, "material", "cand"), exist_ok=True)
    results: List[Dict] = []
    started = start is None
    for s in steps:
        if not started:
            if s["name"].startswith(start):
                started = True
            else:
                if s["name"] in prev:
                    results.append(prev[s["name"]])
                continue
        if resume and prev.get(s["name"], {}).get("status") == "ok":
            results.append(prev[s["name"]])
            continue
        argv = step_argv(s, D, T)
        # globs (p. ej. cand/*.mid) los expande la herramienta; aquí solo si hace falta
        argv = [a for a in _expand_globs(argv)]
        if dry:
            print(f"  {_c('G')}{s['name']}{_c('R')}  " + " ".join(shlex.quote(a) for a in argv))
            continue
        tool_missing = s["tool"] and not os.path.exists(os.path.join(T, s["tool"]))
        t0 = time.time()
        if tool_missing:
            status, rc = ("aviso" if s["informative"] else "fallo"), 127
            out = f"herramienta no encontrada: {s['tool']}"
        else:
            try:
                p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                   cwd=T)
                rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
                status = "ok" if rc == 0 else ("aviso" if s["informative"] else "fallo")
            except subprocess.TimeoutExpired:
                status, rc, out = "timeout", 124, f"(timeout {timeout}s)"
            except Exception as e:  # noqa: BLE001
                status, rc, out = "error", 1, str(e)
        dt = round(time.time() - t0, 2)
        logf = os.path.join(D, "analisis", re.sub(r"[^A-Za-z0-9_.-]", "_", s["name"]) + ".log")
        Path(logf).write_text("$ " + " ".join(shlex.quote(a) for a in argv) + "\n\n" + out,
                              encoding="utf-8")
        rec = {"name": s["name"], "stage": s["stage"], "tool": s["tool"] or "reconstructor.py",
               "note": s["note"], "status": status, "rc": rc, "seconds": dt,
               "informative": s["informative"], "internal": s["internal"],
               "log": os.path.relpath(logf, D),
               "tail": "\n".join(out.strip().splitlines()[-4:])}
        results.append(rec)
        col = {"ok": "GRN", "aviso": "YEL"}.get(status, "RED")
        print(f"  {_c(col)}{status:>7}{_c('R')}  {s['name']:<24} {_c('G')}{dt:>6.1f}s  "
              f"{s['note'][:46]}{_c('R')}")
        Path(log_path).write_text(json.dumps({"version": VERSION, "steps": results},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
        if strict and status in ("fallo", "timeout", "error"):
            print(f"  {_c('RED')}--strict: se aborta en {s['name']}{_c('R')}")
            break
    return results


def _expand_globs(argv: List[str]) -> List[str]:
    out: List[str] = []
    for a in argv:
        if any(ch in a for ch in "*?[") and os.sep in a:
            hits = sorted(glob.glob(a))
            out += hits if hits else [a]
        else:
            out.append(a)
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENTO
# ══════════════════════════════════════════════════════════════════════════════

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _jload(p: str):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def quality_from_log(D: str) -> Optional[float]:
    p = os.path.join(D, "analisis", "23_calidad.log")
    if not os.path.exists(p):
        return None
    txt = _ANSI.sub("", Path(p).read_text(encoding="utf-8", errors="replace"))
    m = re.search(r"SCORE GLOBAL.*?(\d\.\d{2,3})\s+\[", txt, re.S)
    return float(m.group(1)) if m else None


def collect_metrics(D: str) -> Dict:
    mo, mf = _jload(f"{D}/material/metric_orig.json"), _jload(f"{D}/material/metric_final.json")
    co, cf = _jload(f"{D}/material/critica_orig.json"), _jload(f"{D}/material/critica_final.json")
    sel = _jload(f"{D}/material/seleccion_melodia.json")
    return {
        "bars_orig": (mo or {}).get("n_bars"), "bars_final": (mf or {}).get("n_bars"),
        "melody_orig": (co or {}).get("overall"), "melody_final": (cf or {}).get("overall"),
        "melody_selected": (sel or {}).get("winner_score"),
        "quality": quality_from_log(D),
    }


def write_doc(D: str) -> str:
    rec = _jload(os.path.join(D, "reconstruction.json"))
    log = _jload(os.path.join(D, "pipeline_log.json")) or {"steps": []}
    if not rec:
        raise FileNotFoundError(f"{D}/reconstruction.json no existe (ejecuta plan/run antes)")
    info, dec = rec["info"], rec["dec"]
    steps = {s["name"]: s for s in rec["steps"]}
    res = {s["name"]: s for s in log["steps"]}
    met = collect_metrics(D)
    L: List[str] = []
    A = L.append
    name = os.path.basename(info["file"])
    A(f"# Reconstrucción pedagógica de `{name}`\n")
    A("> Documento generado por `reconstructor.py`. La pieza no se copia: se **analiza**, "
      "se **captura su material** (acordes, motivo, clímax) y se **regenera** con las "
      "herramientas del ecosistema mutopia.\n")
    A("| | |\n|---|---|")
    A(f"| **Original** | `{name}` |")
    A(f"| **Tonalidad** | {info['key_long']} ({info['key_source']}) |")
    A(f"| **Compás** | {info['time_sig']} ({info['bpb']:g} negras por compás) |")
    A(f"| **Tempo** | {dec['tempo_i']} BPM |")
    A(f"| **Compases** | {dec['n_bars']} |")
    A(f"| **Familia tímbrica** | {info['family']} |")
    mv = info.get("melody_voice")
    if mv:
        A(f"| **Voz melódica** | pista {mv['track']}, canal {mv['channel']} "
          f"({mv['n']} notas, mediana MIDI {mv['median']:.0f}) |")
    A("| **MIDI final** | `obra_final.mid` |")
    A("| **Receta / plan** | `receta.sh` · `plan.yaml` (runner.py) |\n")
    cnt = Counter(s["status"] for s in log["steps"])
    if log["steps"]:
        A(f"**Resumen:** {cnt.get('ok', 0)} pasos correctos, {cnt.get('aviso', 0)} avisos, "
          f"{sum(cnt.get(k, 0) for k in ('fallo', 'timeout', 'error'))} fallos "
          f"(ver §6).\n")

    A("## 1. Decisiones tomadas y por qué\n")
    A("| Decisión | Valor | Motivo |\n|---|---|---|")
    for x in dec["decisions"]:
        A(f"| {x['que']} | `{x['valor']}` | {x['porque']} |")
    A("\nTodas estas reglas viven en el perfil (`reconstructor.py profile`) y pueden "
      "cambiarse con `--profile`.\n")

    def block(title: str, names: List[str], intro: str = ""):
        present = [n for n in names if n in steps]
        if not present:
            return
        A(f"## {title}\n")
        if intro:
            A(intro + "\n")
        A("| Paso | Herramienta | Qué hace | Estado |\n|---|---|---|---|")
        for n in present:
            s = steps[n]
            st = res.get(n, {}).get("status", "—")
            A(f"| `{n}` | `{s['tool'] or 'reconstructor.py'}` | {s['note']} | {st} |")
        A("\n```bash")
        for n in present:
            if not steps[n]["internal"]:
                A(step_shell_doc(steps[n]))
        A("```\n")

    block("2. Diagnóstico del original",
          ["01_armonia_orig", "02_metrica_orig", "03_melodia_orig", "04_dna_orig"],
          "Antes de tocar nada, la suite **entiende** la pieza.")
    mo = _jload(f"{D}/material/metric_orig.json")
    if mo:
        A(f"- **Métrica:** {mo.get('time_signature', info['time_sig'])}, "
          f"{mo.get('n_bars', '?')} compases.")
    ch = _jload(f"{D}/material/chords_orig.json")
    if ch and ch.get("chords"):
        seq = ", ".join((c.get("chord") or "N.C.") for c in ch["chords"][:8])
        A(f"- **Armonía:** primeros acordes: {seq}…")
    if met["melody_orig"] is not None:
        A(f"- **Melodía original (melody_critic):** {met['melody_orig']:.2f}")
    A("")
    block("3. Captura del material", ["05_split"],
          "`midi_split_harmony` separa melodía y armonía y extrae los **acordes reales** "
          "con un HMM de funciones armónicas. Lo que se conserva es el ADN armónico y temático.")
    if info.get("motif"):
        A(f"- **Motivo semilla:** `{info['motif']}`\n")
    block("4. Regeneración",
          ["06_cpg", "07_tonicizacion_report", "08_tonicizacion", "08b_base_armonica",
           "09_bajo", "10_patron", "10b_acomp", "11_melodia", "12_seleccion",
           "12b_melodia", "13_merge"])
    sel = _jload(f"{D}/material/seleccion_melodia.json")
    if sel and sel.get("candidates"):
        A("**Selección de melodía** (`melody_selector`, criterio: `melody_critic`):\n")
        A("| Candidato | Score |\n|---|---:|")
        for c in sel["candidates"]:
            A(f"| `{os.path.basename(c['file'])}` | "
              f"{'—' if c.get('overall') is None else format(c['overall'], '.2f')} |")
        A("")
    block("5. Forma, expresión y color",
          ["14_climax", "14b_etapa1", "15_performer", "15b_etapa2", "16_tempo",
           "16b_final", "17_meta", "18_color"])
    block("6. Crítica del resultado (el bucle)",
          ["19_armonia_final", "20_metrica_final", "21_melodia_final", "22_tocabilidad",
           "23_calidad", "24_arco", "25_export_yaml", "26_compare"])
    if any(v is not None for v in met.values()):
        A("**Métricas:**\n")
        A("| Métrica | Valor |\n|---|---|")
        if met["bars_orig"] is not None or met["bars_final"] is not None:
            ok = ""
            try:
                if len(info.get("time_sigs_all", [])) > 1:
                    ok = " (el original cambia de compás: comparar duraciones)"
                else:
                    ok = " ✔" if abs(met["bars_final"] - met["bars_orig"]) <= 1 else \
                        " ⚠ (longitud distinta; el rubato de performer estira el tiempo)"
            except Exception:  # noqa: BLE001
                pass
            A(f"| Compases original → final | {met['bars_orig']} → {met['bars_final']}{ok} |")
        for k, lab in (("melody_orig", "Melodía original (melody_critic)"),
                       ("melody_selected", "Melodía elegida por el selector"),
                       ("melody_final", "Melodía en el MIDI final"),
                       ("quality", "Calidad global (quality_scorer)")):
            if met[k] is not None:
                A(f"| {lab} | {met[k]:.2f} |")
        A("")
    A("## 7. Incidencias\n")
    bad = [s for s in log["steps"] if s["status"] in ("fallo", "timeout", "error", "aviso")]
    if not bad:
        A("Ninguna: todos los pasos terminaron correctamente.\n")
    else:
        for s in bad:
            A(f"- `{s['name']}` ({s['tool']}): **{s['status']}** (rc={s['rc']}) — "
              f"`{s['log']}`")
        A("")
    A("## 8. Ficheros\n")
    A("```\n" + os.path.basename(D) + "/\n  original.mid  obra_base.mid  obra_climax.mid  "
      "obra_perf.mid  obra_final.mid\n  plan.yaml  receta.sh  reconstruction.json  "
      "pipeline_log.json  RECONSTRUCCION.md\n  analisis/   (log de cada paso)\n  material/   "
      "(acordes, bajo, acompañamiento, melodías candidatas, JSON)\n```\n")
    A("## 9. Cómo reproducirlo\n")
    A("```bash\nbash receta.sh                      # autónoma\n"
      "python runner.py plan.yaml           # con el ejecutor de pipelines\n"
      "python reconstructor.py run ORIGINAL.mid --out-dir DIR --resume\n```\n")
    out = os.path.join(D, "RECONSTRUCCION.md")
    Path(out).write_text("\n".join(L), encoding="utf-8")
    return out


def step_shell_doc(step: Dict) -> str:
    s = step_shell(step)
    return s.replace('"$D/', '"').replace('"$TOOLS/', '"')


# ══════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def prepare(a, midi: str, out_dir: Optional[str] = None) -> Tuple[str, str, Dict, Dict, List[Dict], Dict]:
    if not os.path.isfile(midi):
        raise FileNotFoundError(f"no existe el MIDI: {midi}")
    prof = load_profile(a.profile)
    T = os.path.abspath(a.tools_dir or os.path.dirname(SELF))
    stem = os.path.splitext(os.path.basename(midi))[0]
    D = os.path.abspath(out_dir or a.out_dir or f"{stem}_rec")
    os.makedirs(D, exist_ok=True)
    shutil.copyfile(midi, os.path.join(D, "original.mid"))
    info = analyze_midi(midi, prof)
    dec = decide(info, prof, getattr(a, "candidates", None))
    crit = prof["critique"] and not getattr(a, "no_critique", False)
    steps = build_steps(info, dec, prof, crit)
    write_plan_files(D, T, steps, info, dec)
    Path(D, "reconstruction.json").write_text(json.dumps(
        {"version": VERSION, "created": datetime.now().isoformat(timespec="seconds"),
         "info": info, "dec": dec, "profile": prof, "steps": steps},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return D, T, info, dec, steps, prof


def _print_plan(info: Dict, dec: Dict, D: str, n_steps: int) -> None:
    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}\n  {B}RECONSTRUCTOR v{VERSION}{R}  ·  {os.path.basename(info['file'])}\n{'═' * 78}")
    for x in dec["decisions"]:
        v = str(x["valor"])
        print(f"  {x['que']:<24} {B}{v[:40]}{R}  {G}{x['porque'][:40]}{R}")
    print(f"\n  {n_steps} pasos → {D}\n")


def cmd_plan(a) -> int:
    D, T, info, dec, steps, _ = prepare(a, a.midi)
    _print_plan(info, dec, D, len(steps))
    print(f"  {_c('CYA')}plan.yaml · receta.sh · reconstruction.json escritos en {D}{_c('R')}\n")
    return 0


def cmd_run(a) -> int:
    D, T, info, dec, steps, _ = prepare(a, a.midi)
    _print_plan(info, dec, D, len(steps))
    res = run_steps(D, T, steps, a.timeout, a.strict, a.resume, a.from_step, a.dry_run)
    if a.dry_run:
        return 0
    final = os.path.join(D, "obra_final.mid")
    doc = write_doc(D)
    cnt = Counter(s["status"] for s in res)
    bad = sum(cnt.get(k, 0) for k in ("fallo", "timeout", "error"))
    ok = os.path.exists(final)
    col = "GRN" if ok and not bad else "YEL" if ok else "RED"
    print(f"\n  {_c(col)}{'✔' if ok else '✘'} {final if ok else 'no se generó obra_final.mid'}{_c('R')}"
          f"   ok={cnt.get('ok', 0)} avisos={cnt.get('aviso', 0)} fallos={bad}")
    print(f"  {_c('CYA')}{doc}{_c('R')}\n")
    return 0 if ok else 2


def cmd_doc(a) -> int:
    print(write_doc(a.dir))
    return 0


def cmd_profile(a) -> int:
    print(json.dumps(DEFAULT_PROFILE, ensure_ascii=False, indent=2))
    return 0


def _read_meta(path: str) -> List[mido.MetaMessage]:
    seen, out = set(), []
    for tr in mido.MidiFile(path).tracks:
        for m in tr:
            if m.type in ("time_signature", "key_signature") and m.type not in seen:
                seen.add(m.type)
                out.append(m.copy(time=0))
    return out


def cmd_meta(a) -> int:
    mf = mido.MidiFile(a.file)
    metas: List = []
    if a.src:
        metas += _read_meta(a.src)
    if a.time_sig:
        n, d = (int(x) for x in a.time_sig.split("/"))
        metas = [m for m in metas if m.type != "time_signature"] + [
            mido.MetaMessage("time_signature", numerator=n, denominator=d, time=0)]
    if a.key:
        metas = [m for m in metas if m.type != "key_signature"] + [
            mido.MetaMessage("key_signature", key=a.key, time=0)]
    if not metas:
        print("[ERROR] nada que inyectar (usa --from, --time-sig o --key)", file=sys.stderr)
        return 1
    kinds = {m.type for m in metas}
    if not mf.tracks:
        mf.tracks.append(mido.MidiTrack())
    for tr in mf.tracks:       # quita los existentes del mismo tipo (todas las pistas)
        for m in [m for m in tr if m.type in kinds]:
            tr.remove(m)
    t0 = mf.tracks[0]
    for m in reversed(metas):
        t0.insert(0, m)
    mf.save(a.out or a.file)
    desc = [m.type + "=" + (m.key if m.type == "key_signature"
                            else f"{m.numerator}/{m.denominator}") for m in metas]
    print("meta: " + ", ".join(desc) + f" → {a.out or a.file}")
    return 0


def cmd_ensure(a) -> int:
    for c in a.candidates:
        if os.path.isfile(c) and os.path.getsize(c) > 0:
            if os.path.abspath(c) != os.path.abspath(a.out):
                shutil.copyfile(c, a.out)
            print(f"ensure: {c} → {a.out}")
            return 0
    print(f"[ERROR] ninguno de los ficheros existe: {a.candidates}", file=sys.stderr)
    return 1


def cmd_batch(a) -> int:
    files: List[str] = []
    for p in a.inputs:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.mid")) + glob.glob(os.path.join(p, "*.midi")))
        else:
            files += sorted(glob.glob(p)) or ([p] if os.path.isfile(p) else [])
    files = list(dict.fromkeys(files))
    if a.limit:
        files = files[: a.limit]
    if not files:
        print("[ERROR] sin MIDIs de entrada", file=sys.stderr)
        return 1
    root = os.path.abspath(a.out_root)
    os.makedirs(root, exist_ok=True)
    print(f"\n  {_c('B')}BATCH{_c('R')}: {len(files)} piezas → {root}  (jobs={a.jobs})\n")

    def one(i_f):
        i, f = i_f
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(f))[0])
        D = os.path.join(root, f"{i:03d}_{stem}")
        if a.resume and os.path.exists(os.path.join(D, "obra_final.mid")) and \
                os.path.exists(os.path.join(D, "RECONSTRUCCION.md")):
            return D, f, 0, 0.0, "reanudada"
        cmd = [PY, SELF, "run", f, "--out-dir", D, "--timeout", str(a.timeout)]
        for flag, val in (("--profile", a.profile), ("--tools-dir", a.tools_dir)):
            if val:
                cmd += [flag, val]
        if a.no_critique:
            cmd.append("--no-critique")
        if a.resume:
            cmd.append("--resume")
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        os.makedirs(D, exist_ok=True)
        Path(D, "run.log").write_text((p.stdout or "") + (p.stderr or ""), encoding="utf-8")
        return D, f, p.returncode, time.time() - t0, "ok" if p.returncode == 0 else "fallo"

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        for D, f, rc, dt, st in ex.map(one, list(enumerate(files, 1))):
            m = collect_metrics(D)
            lg = _jload(os.path.join(D, "pipeline_log.json")) or {"steps": []}
            cnt = Counter(s["status"] for s in lg["steps"])
            rows.append({"dir": os.path.basename(D), "file": os.path.basename(f), "status": st,
                         "seconds": round(dt, 1), "pasos_ok": cnt.get("ok", 0),
                         "avisos": cnt.get("aviso", 0),
                         "fallos": sum(cnt.get(k, 0) for k in ("fallo", "timeout", "error")),
                         **m})
            r = rows[-1]
            print(f"  {r['dir']:<40} {st:<9} ok={r['pasos_ok']:<3} av={r['avisos']:<2} "
                  f"fa={r['fallos']:<2} bars {r['bars_orig']}→{r['bars_final']}  "
                  f"{_c('G')}{dt:.0f}s{_c('R')}")
    Path(root, "RESULTADOS.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    md = ["# Resultados del lote\n",
          "| pieza | estado | pasos ok | avisos | fallos | compases (orig→final) | melodía final | calidad |",
          "|---|---|---:|---:|---:|---|---:|---:|"]
    f2 = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    for r in rows:
        md.append(f"| [{r['dir']}]({r['dir']}/RECONSTRUCCION.md) | {r['status']} | {r['pasos_ok']} | "
                  f"{r['avisos']} | {r['fallos']} | {r['bars_orig']}→{r['bars_final']} | "
                  f"{f2(r['melody_final'])} | {f2(r['quality'])} |")
    Path(root, "RESULTADOS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n  {_c('CYA')}{root}/RESULTADOS.md · RESULTADOS.json{_c('R')}\n")
    return 0 if all(r["status"] != "fallo" for r in rows) else 2


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="reconstructor.py",
                                 description="Reconstrucción pedagógica de una pieza con mutopia.")
    sub = ap.add_subparsers(dest="name", required=True)

    def common(p):
        p.add_argument("--tools-dir", default=None)
        p.add_argument("--profile", default=None)
        p.add_argument("--no-critique", action="store_true")
        p.add_argument("--timeout", type=int, default=900)
        p.add_argument("--no-color", action="store_true")

    p = sub.add_parser("plan", help="escribe plan.yaml + receta.sh")
    p.add_argument("midi")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--candidates", type=int, default=None)
    common(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="plan + ejecución + documento")
    p.add_argument("midi")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--from", dest="from_step", default=None)
    p.add_argument("--dry-run", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("doc", help="regenera RECONSTRUCCION.md")
    p.add_argument("dir")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(fn=cmd_doc)

    p = sub.add_parser("batch", help="run sobre muchos MIDIs")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out-root", default="reconstrucciones")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    common(p)
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("profile", help="imprime el perfil por defecto (JSON)")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("meta", help="reinyecta compás/tonalidad")
    p.add_argument("file")
    p.add_argument("--from", dest="src", default=None, help="MIDI del que copiar compás y armadura")
    p.add_argument("--time-sig", default=None)
    p.add_argument("--key", default=None, help="armadura estilo mido: C, Bb, F#m…")
    p.add_argument("--out", default=None)
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(fn=cmd_meta)

    p = sub.add_parser("ensure", help="copia el primer fichero existente a --out")
    p.add_argument("candidates", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(fn=cmd_ensure)
    return ap


def main(argv=None) -> int:
    global _USE_COLOR
    a = build_parser().parse_args(argv)
    if getattr(a, "no_color", False):
        _USE_COLOR = False
    try:
        return a.fn(a)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[interrumpido]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
