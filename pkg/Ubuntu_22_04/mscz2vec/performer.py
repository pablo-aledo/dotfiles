#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          PERFORMER  v1.0                                     ║
║         Renderizado expresivo de interpretación para MIDIs cuantizados      ║
║                                                                              ║
║  Los compositores del ecosistema producen MIDI cuantizado y de velocidad    ║
║  uniforme. Gran parte de la emoción percibida no está en QUÉ notas suenan   ║
║  sino en CÓMO: esta herramienta aplica reglas interpretativas locales       ║
║  (inspiradas en las reglas KTH / Director Musices) conscientes de la        ║
║  estructura de frase, como último paso del pipeline antes del render.       ║
║                                                                              ║
║  REGLAS APLICADAS:                                                           ║
║    rubato       — arco agógico por frase: arranque contenido, aceleración   ║
║                   hacia el pico, ritardando en el cierre; ritardando final  ║
║    agogica      — alargamiento del pico melódico de cada frase              ║
║    dinamica     — crescendo hacia el pico de frase + jerarquía métrica      ║
║                   (parte fuerte +, contratiempos −) + jitter humano         ║
║    lead         — la melodía se adelanta ~15 ms al acompañamiento           ║
║    humanize     — micro-desincronización aleatoria de ataques               ║
║    pedal        — CC64 con retoma en cada cambio de armonía (teclados)      ║
║    swell        — CC11 (expresión) en arco dentro de notas largas y CC1     ║
║                   (vibrato) creciente en cuerdas/vientos/voz                ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] LECTURA    — MIDI + mapa de tempo/compás (vía playability_auditor)     ║
║  [2] FRASEO     — segmentación de frases por silencios y notas largas       ║
║  [3] WARP       — mapa de deformación temporal por pulso (rubato+agógica)   ║
║  [4] DINÁMICA   — remodelado de velocidades nota a nota                     ║
║  [5] CONTROL    — inserción de CC64 / CC11 / CC1                            ║
║  [6] ESCRITURA  — MIDI expresivo + informe de desviaciones                  ║
║                                                                              ║
║  USO:                                                                        ║
║    python performer.py obra.mid                                              ║
║    python performer.py obra.mid --style romantico --intensity 0.8           ║
║    python performer.py obra.mid --no-pedal --no-vibrato                     ║
║    python performer.py obra.mid --melody-track 1 --seed 7                   ║
║    python performer.py obra.mid --dry-run --verbose                         ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi               MIDI de entrada                                        ║
║    --output FILE      MIDI de salida (default: <obra>_perf.mid)             ║
║    --style S          romantico | clasico | mecanico (preset de intensidad) ║
║    --intensity F      Intensidad expresiva 0–1 (anula el preset)            ║
║    --melody-track N   Pista de melodía (default: autodetección)             ║
║    --no-rubato / --no-dynamics / --no-pedal / --no-vibrato / --no-humanize  ║
║    --lead MS          Adelanto de la melodía en ms (default: 15)            ║
║    --seed N           Semilla aleatoria (default: 42)                       ║
║    --dry-run          Solo informe, sin escribir MIDI                        ║
║    --verbose          Detalle de frases y curvas                             ║
║    --no-color         Desactivar colores ANSI                                ║
║                                                                              ║
║  COMO MÓDULO:                                                                ║
║    from performer import perform_midi                                        ║
║    stats = perform_midi("obra.mid", "obra_perf.mid", intensity=0.7)         ║
║                                                                              ║
║  DEPENDENCIAS: numpy · playability_auditor.py (mismo directorio, E/S MIDI)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from playability_auditor import (read_midi, write_midi, MidiEvent, TimeContext,
                                     extract_notes, Note, pitch_name,
                                     INSTRUMENT_DB, detect_instrument)
except ImportError:
    print("ERROR: requiere playability_auditor.py en el mismo directorio (E/S MIDI)")
    sys.exit(1)

VERSION = "1.0"

_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


STYLE_INTENSITY = {"romantico": 0.85, "clasico": 0.5, "mecanico": 0.15}

# ══════════════════════════════════════════════════════════════════════════════
#  FRASEO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Phrase:
    notes: List[Note]
    start: int
    end: int
    peak_idx: int          # índice (en notes) del pico melódico


def detect_phrases(notes: List[Note], tpb: int) -> List[Phrase]:
    """Segmenta por silencios >= medio pulso; frases mínimas de 3 notas."""
    if not notes:
        return []
    groups, cur = [], [notes[0]]
    for prev, n in zip(notes, notes[1:]):
        if n.start - max(p.end for p in cur) >= tpb * 0.45:
            groups.append(cur)
            cur = [n]
        else:
            cur.append(n)
    groups.append(cur)
    merged = []
    for g in groups:                                 # frases muy cortas se fusionan
        if merged and len(g) < 3:
            merged[-1].extend(g)
        else:
            merged.append(g)
    phrases = []
    for g in merged:
        peak = max(range(len(g)), key=lambda i: (g[i].pitch, g[i].end - g[i].start))
        phrases.append(Phrase(g, g[0].start, max(n.end for n in g), peak))
    return phrases


def pick_melody_track(tracks_notes: dict) -> int:
    """Pista melódica: mayor altura media ponderada por nº de notas."""
    best, best_v = None, -1e9
    for idx, notes in tracks_notes.items():
        if not notes:
            continue
        mean_p = sum(n.pitch for n in notes) / len(notes)
        v = mean_p + min(len(notes), 200) * 0.02
        if v > best_v:
            best, best_v = idx, v
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  WARP TEMPORAL (rubato + agógica)
# ══════════════════════════════════════════════════════════════════════════════

class TimeWarp:
    """Mapa monótono tick→tick construido desde factores de duración por pulso."""

    def __init__(self, factors: np.ndarray, tpb: int):
        self.tpb = tpb
        self.factors = factors
        self.cum = np.concatenate([[0.0], np.cumsum(factors) * tpb])

    def __call__(self, tick: int) -> int:
        b = tick / self.tpb
        i = min(int(b), len(self.factors) - 1)
        frac = b - i
        return int(round(self.cum[i] + frac * self.factors[i] * self.tpb))


def build_warp(phrases: List[Phrase], total_ticks: int, tpb: int,
               intensity: float) -> TimeWarp:
    n_beats = max(1, total_ticks // tpb + 4)
    f = np.ones(n_beats)
    for ph in phrases:
        b0, b1 = ph.start / tpb, ph.end / tpb
        if b1 - b0 < 1:
            continue
        length = b1 - b0
        # arranque ligeramente contenido
        f[int(b0)] *= 1 + 0.030 * intensity
        # aceleración en el cuerpo central
        c0, c1 = int(b0 + length * 0.2), int(b0 + length * 0.7)
        for b in range(c0, min(c1, n_beats)):
            f[b] *= 1 - 0.020 * intensity
        # ritardando en el cierre (último 15 %)
        r0 = int(b0 + length * 0.85)
        for k, b in enumerate(range(r0, min(int(b1) + 1, n_beats))):
            f[b] *= 1 + (0.030 + 0.040 * k) * intensity
        # agógica: el pico melódico se alarga
        pk = ph.notes[ph.peak_idx]
        f[min(int(pk.start / tpb), n_beats - 1)] *= 1 + 0.045 * intensity
    # ritardando final de la obra (últimos 3 pulsos)
    for k, b in enumerate(range(max(0, n_beats - 7), n_beats - 4)):
        f[b] *= 1 + (0.06 + 0.09 * k) * intensity
    return TimeWarp(f, tpb)


# ══════════════════════════════════════════════════════════════════════════════
#  DINÁMICA
# ══════════════════════════════════════════════════════════════════════════════

def shape_velocities(phrases: List[Phrase], tpb: int, bar_ticks: int,
                     intensity: float, rng: random.Random):
    for ph in phrases:
        n = len(ph.notes)
        for i, note in enumerate(ph.notes):
            # crescendo hacia el pico y caída posterior
            if ph.peak_idx > 0 and i <= ph.peak_idx:
                shape = i / ph.peak_idx
            elif n - 1 > ph.peak_idx:
                shape = 1 - (i - ph.peak_idx) / (n - 1 - ph.peak_idx)
            else:
                shape = 1.0
            dv = (shape - 0.5) * 20 * intensity
            # jerarquía métrica
            pos = (note.start % bar_ticks) / tpb
            if abs(pos - round(pos)) < 1e-6:          # sobre el pulso
                beat = int(round(pos))
                dv += (6 if beat == 0 else 2 if beat == 2 else 0) * intensity
            else:
                dv -= 4 * intensity                    # contratiempo
            dv += rng.uniform(-4, 4) * intensity       # jitter humano
            note.vel = max(1, min(127, int(round(note.vel + dv))))


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def _rebuild(trk, notes, extra_events=None):
    others = [e for e in trk.events if e.kind not in ("note_on", "note_off")]
    evs = list(others) + (extra_events or [])
    for n in notes:
        evs.append(MidiEvent(abs=n.start, kind="note_on", channel=n.channel,
                             pitch=n.pitch, vel=n.vel))
        evs.append(MidiEvent(abs=n.end, kind="note_off", channel=n.channel,
                             pitch=n.pitch, vel=0))
    trk.events = sorted(evs, key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))


def _cc(t, ch, num, val):
    return MidiEvent(abs=t, kind="channel", channel=ch,
                     data=bytes([0xB0 | ch, num, max(0, min(127, val))]))


def _bar_chroma(notes, bar_ticks, n_bars):
    ch = np.zeros((n_bars, 12))
    for n in notes:
        b = min(n.start // bar_ticks, n_bars - 1)
        ch[b, n.pitch % 12] += n.end - n.start
    return ch


def perform_midi(path: str, out_path: Optional[str] = None,
                 intensity: float = 0.7, melody_track: Optional[int] = None,
                 rubato: bool = True, dynamics: bool = True, pedal: bool = True,
                 vibrato: bool = True, humanize: bool = True,
                 lead_ms: float = 15.0, seed: int = 42,
                 dry_run: bool = False) -> dict:
    """API pública. Devuelve estadísticas del renderizado."""
    rng = random.Random(seed)
    mid = read_midi(path)
    tc = TimeContext(mid)
    tpb = mid.tpb
    num, den = mid.timesig_map[0][1], mid.timesig_map[0][2]
    bar_ticks = tpb * 4 * num // den

    tracks_notes = {i: extract_notes(t) for i, t in enumerate(mid.tracks)}
    tracks_notes = {i: ns for i, ns in tracks_notes.items() if ns}
    if not tracks_notes:
        raise ValueError("el MIDI no contiene notas")
    mel_idx = melody_track if melody_track is not None else pick_melody_track(tracks_notes)
    total_ticks = max(max(n.end for n in ns) for ns in tracks_notes.values())
    n_bars = total_ticks // bar_ticks + 1

    # [2] fraseo sobre la melodía (guía global del rubato)
    phrases = detect_phrases(tracks_notes[mel_idx], tpb)

    # [3] warp temporal global
    warp = build_warp(phrases, total_ticks, tpb, intensity) if rubato \
        else TimeWarp(np.ones(total_ticks // tpb + 4), tpb)

    us_per_beat = mid.tempo_map[0][1]
    lead_ticks = int(lead_ms / 1000 * 1e6 / us_per_beat * tpb * intensity)

    onset_dev, vel_before, vel_after = [], [], []
    pedal_events_n = cc_events_n = 0

    # armonía por compás (para el pedal): cambios de croma
    all_notes = [n for ns in tracks_notes.values() for n in ns]
    chroma = _bar_chroma(all_notes, bar_ticks, n_bars)
    harmony_changes = [0]
    for b in range(1, n_bars):
        a, c = chroma[b - 1], chroma[b]
        if a.sum() and c.sum():
            cos = float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-9))
            if cos < 0.995:
                harmony_changes.append(b * bar_ticks)

    for idx, trk in enumerate(mid.tracks):
        notes = tracks_notes.get(idx)
        if not notes:
            continue
        inst = detect_instrument(trk)
        fam = INSTRUMENT_DB[inst]["family"]
        vel_before.extend(n.vel for n in notes)

        # [4] dinámica por frase (cada pista con su propio fraseo)
        if dynamics:
            shape_velocities(detect_phrases(notes, tpb), tpb, bar_ticks,
                             intensity, rng)

        for n in notes:
            s0 = n.start
            n.start, n.end = warp(n.start), warp(n.end)
            if humanize:
                on_beat = (s0 % tpb) == 0
                jit = int(rng.uniform(-1, 1) * tpb * (0.008 if on_beat else 0.016)
                          * intensity)
                n.start = max(0, n.start + jit)
            if idx == mel_idx and lead_ticks:
                n.start = max(0, n.start - lead_ticks)
            if n.end <= n.start:
                n.end = n.start + 1
            onset_dev.append(abs(n.start - warp(s0)))
        vel_after.extend(n.vel for n in notes)

        # [5] controladores
        extra = []
        ch = notes[0].channel
        if pedal and fam == "teclado":
            for t in harmony_changes:
                wt = warp(t)
                extra.append(_cc(wt, ch, 64, 0))
                extra.append(_cc(wt + max(5, tpb // 32), ch, 64, 127))
            extra.append(_cc(warp(total_ticks) + tpb, ch, 64, 0))
            pedal_events_n += len(extra)
        if vibrato and fam in ("cuerda_frotada", "viento_madera",
                               "viento_metal", "voz") and len(notes) < 1500:
            for n in notes:
                dur = n.end - n.start
                if dur >= int(tpb * 1.5):
                    steps = min(8, dur // (tpb // 4))
                    for k in range(steps + 1):
                        t = n.start + dur * k // max(1, steps)
                        x = k / max(1, steps)
                        swell = int(70 + 35 * np.sin(np.pi * x))          # CC11 arco
                        vib = int(45 * intensity * min(1.0, x * 1.6))      # CC1 creciente
                        extra.append(_cc(t, ch, 11, swell))
                        extra.append(_cc(t, ch, 1, vib))
                    extra.append(_cc(n.end, ch, 1, 0))
            cc_events_n += len(extra) - pedal_events_n

        # warp del resto de eventos de la pista (CCs previos, programas...)
        for e in trk.events:
            if e.kind not in ("note_on", "note_off") and e.abs > 0:
                e.abs = warp(e.abs)
        _rebuild(trk, notes, extra)

    stats = {
        "intensity": intensity,
        "melody_track": mel_idx,
        "phrases": len(phrases),
        "vel_std_before": round(float(np.std(vel_before)), 2),
        "vel_std_after": round(float(np.std(vel_after)), 2),
        "harmony_changes": len(harmony_changes),
        "pedal_events": pedal_events_n,
        "cc_events": cc_events_n,
        "final_stretch_pct": round((warp(total_ticks) / total_ticks - 1) * 100, 2),
    }
    if not dry_run:
        out = out_path or str(Path(path).with_name(Path(path).stem + "_perf.mid"))
        write_midi(mid, out)
        stats["output"] = out
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="performer.py",
                                 description="Renderizado expresivo de interpretación.")
    ap.add_argument("midi")
    ap.add_argument("--output")
    ap.add_argument("--style", choices=sorted(STYLE_INTENSITY), default="romantico")
    ap.add_argument("--intensity", type=float)
    ap.add_argument("--melody-track", type=int)
    ap.add_argument("--no-rubato", action="store_true")
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--no-pedal", action="store_true")
    ap.add_argument("--no-vibrato", action="store_true")
    ap.add_argument("--no-humanize", action="store_true")
    ap.add_argument("--lead", type=float, default=15.0, metavar="MS")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        _USE_COLOR = False

    intensity = args.intensity if args.intensity is not None \
        else STYLE_INTENSITY[args.style]
    intensity = max(0.0, min(1.0, intensity))

    B, R, G = _c("B"), _c("R"), _c("G")
    print(f"\n{'═' * 78}")
    print(f"  {B}PERFORMER v{VERSION}  ·  {args.midi}{R}")
    print(f"  Estilo: {args.style}   intensidad: {intensity:.2f}")
    print(f"{'═' * 78}")

    try:
        stats = perform_midi(
            args.midi, args.output, intensity=intensity,
            melody_track=args.melody_track,
            rubato=not args.no_rubato, dynamics=not args.no_dynamics,
            pedal=not args.no_pedal, vibrato=not args.no_vibrato,
            humanize=not args.no_humanize, lead_ms=args.lead,
            seed=args.seed, dry_run=args.dry_run)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    print(f"  Melodía: pista {stats['melody_track']}   "
          f"frases: {stats['phrases']}   cambios armónicos: {stats['harmony_changes']}")
    print(f"  Velocidades  σ antes: {stats['vel_std_before']}  →  "
          f"σ después: {stats['vel_std_after']}")
    print(f"  Eventos CC64 (pedal): {stats['pedal_events']}   "
          f"CC11/CC1 (swell/vibrato): {stats['cc_events']}")
    print(f"  Estiramiento temporal acumulado: {stats['final_stretch_pct']:+.2f}%")
    if args.dry_run:
        print(f"  {G}(dry-run: no se escribió MIDI){R}")
    else:
        print(f"  {_c('GRN')}MIDI expresivo: {stats['output']}{R}")
    print(f"{'═' * 78}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
