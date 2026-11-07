#!/usr/bin/env python3
"""
key_consensus.py — Key Consensus: tonalidad ÚNICA con confianza, a partir de
varios «votantes» independientes, y avisos cuando el dato no es fiable
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Cada método de detección de tonalidad falla en sitios distintos: la armadura
MIDI suele ser un «Do mayor» por defecto, el perfil global se confunde en
música modulante, y un vamp/ostinato estático no tiene tonalidad funcional.
Esta herramienta hace votar a 5 métodos (perfil Krumhansl-Kessler global,
del bajo y de la melodía, armadura MIDI y nota final del bajo), mezcla sus
distribuciones de probabilidad sobre las 24 tonalidades y devuelve UNA
tonalidad con su confianza, el desacuerdo entre votantes y banderas de
fiabilidad (estático, relativo ambiguo, modal/cromático). La ortografía usa
bemoles o sostenidos según la armadura convencional (Sib, no La#).
Autocontenida: solo mido; numpy es opcional (no se usa).

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  detect    Tonalidad consensuada (una línea por archivo)
  compare   Tabla de votantes y discrepancias de UN archivo
  sections  Tonalidad por secciones de N compases (modulaciones)

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  detect
    midis MIDIS […]
    --json JSON
    --drums                     incluye el canal 10 (por defecto se excluye)
    --no-color

  compare
    midi MIDI
    --top TOP                   (defecto 3)
    --no-color

  sections
    midi MIDI
    --bars BARS                 compases por ventana (defecto 8)
    --json JSON
    --no-color

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[detect]
midis      list  of=file io=in required
--json     file  io=out ext=json
--drums    bool  default=false
--no-color bool  default=false
[compare]
midi       file  io=in ext=mid required
--top      int   default=3 min=1 max=24
--no-color bool  default=false
[sections]
midi       file  io=in ext=mid required
--bars     int   default=8 min=1
--json     file  io=out ext=json
--no-color bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python key_consensus.py detect obra.mid
    python key_consensus.py detect midis/*.mid --json tonalidades.json
    python key_consensus.py compare obra.mid
    python key_consensus.py sections obra.mid --bars 4

  FIABILIDAD (campo «fiab»): alta · media · baja. Banderas:
    estático    ≤ 3 clases de altura efectivas: vamp/ostinato, sin tonalidad funcional
    rel-ambig   la 2.ª candidata es la relativa mayor/menor y el margen es pequeño
    modal       el mejor perfil correlaciona < 0.60 (modal, cromático o atonal)
    desacuerdo  los votantes proponen tonalidades no emparentadas
    sig-defecto la armadura MIDI es Do mayor/La menor pero la música dice otra cosa

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 2 fichero ilegible

  API:  from key_consensus import consensus_file

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  mido
"""

import argparse
import json
import math
import os
import sys

try:
    import mido
except ImportError:  # pragma: no cover
    print("ERROR: pip install mido")
    sys.exit(1)

VERSION = "1.0"
_COLORS = {"R": "\033[0m", "B": "\033[1m", "G": "\033[90m", "GRN": "\033[92m",
           "YEL": "\033[93m", "RED": "\033[91m", "CYA": "\033[96m"}
_USE_COLOR = sys.stdout.isatty()


def _c(k):
    return _COLORS.get(k, "") if _USE_COLOR else ""


def _ensure_parent(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return path


SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Perfiles Krumhansl-Kessler
KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

VOTER_WEIGHT = {"global": 1.0, "bass": 1.0, "melody": 0.7, "sig": 0.8, "ending": 0.8}
TAU = 8.0  # temperatura del softmax sobre correlaciones


def key_name(tonic, mode):
    flats = (tonic in (0, 2, 3, 5, 7, 10)) if mode == "min" else (tonic in (1, 3, 5, 8, 10))
    return (FLAT if flats else SHARP)[tonic] + ("m" if mode == "min" else "")


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA MIDI (autocontenida)
# ══════════════════════════════════════════════════════════════════════════════

def load_midi(path, drums=False):
    """Devuelve dict(tpb, notes=[(start,end,pitch,ch)], tsigs, keysigs)."""
    mid = mido.MidiFile(path, clip=True)
    tpb = mid.ticks_per_beat
    notes, tsigs, keysigs = [], [], []
    for tr in mid.tracks:
        t = 0
        active = {}
        for m in tr:
            t += m.time
            if m.type == "note_on" and m.velocity > 0:
                active.setdefault((m.channel, m.note), []).append(t)
            elif m.type in ("note_off", "note_on"):
                st = active.get((m.channel, m.note))
                if st:
                    s = st.pop(0)
                    if drums or m.channel != 9:
                        notes.append((s, max(t, s + 1), m.note, m.channel))
            elif m.type == "time_signature":
                tsigs.append((t, m.numerator, m.denominator))
            elif m.type == "key_signature":
                keysigs.append((t, m.key))
        for (ch, p), st in active.items():       # notas sin cerrar
            for s in st:
                if drums or ch != 9:
                    notes.append((s, s + tpb, p, ch))
    notes.sort()
    tsigs.sort()
    return {"tpb": tpb, "notes": notes, "tsigs": tsigs, "keysigs": sorted(keysigs)}


# ══════════════════════════════════════════════════════════════════════════════
#  PERFILES Y VOTANTES
# ══════════════════════════════════════════════════════════════════════════════

def _corr(a, b):
    ma, mb = sum(a) / 12.0, sum(b) / 12.0
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def ks_scores(hist):
    """Correlación con las 24 tonalidades -> dict {(tonic, mode): r}."""
    out = {}
    for t in range(12):
        for mode, prof in (("maj", KK_MAJ), ("min", KK_MIN)):
            rot = [prof[(i - t) % 12] for i in range(12)]
            out[(t, mode)] = _corr(hist, rot)
    return out


def _softmax(scores):
    mx = max(scores.values())
    ex = {k: math.exp(TAU * (v - mx)) for k, v in scores.items()}
    z = sum(ex.values()) or 1.0
    return {k: v / z for k, v in ex.items()}


def _hist(notes, weight_cap=None):
    h = [0.0] * 12
    for s, e, p, _ in notes:
        h[p % 12] += (e - s) if weight_cap is None else min(e - s, weight_cap)
    return h


def _line_hist(notes, tpb, lowest=True):
    """Línea de bajo (o soprano): nota más grave/aguda que empieza en cada pulso."""
    best = {}
    for s, e, p, _ in notes:
        k = s // max(1, tpb // 2)       # ventana de corchea
        cur = best.get(k)
        if cur is None or (p < cur[2] if lowest else p > cur[2]):
            best[k] = (s, e, p)
    h = [0.0] * 12
    for s, e, p in best.values():
        h[p % 12] += min(e - s, 2 * tpb)
    return h


def _sig_key(ks):
    """Parsea 'Bb', 'F#m'... -> (tonic, mode)."""
    if not ks:
        return None
    letter = ks[0].upper()
    if letter not in _LETTER:
        return None
    pc = _LETTER[letter]
    i = 1
    while i < len(ks) and ks[i] in "#b":
        pc += 1 if ks[i] == "#" else -1
        i += 1
    return pc % 12, ("min" if ks[i:].lower().startswith("m") and not ks[i:].lower().startswith("maj") else "maj")


def _effective_pcs(hist):
    tot = sum(hist)
    if tot <= 0:
        return 0.0
    ent = -sum((h / tot) * math.log(h / tot) for h in hist if h > 0)
    return math.exp(ent)


def votes_for(notes, tpb, keysigs, want=("global", "bass", "melody", "sig", "ending")):
    """Devuelve {voter: (dist24, raw_scores|None)} y metadatos."""
    out = {}
    g = ks_scores(_hist(notes))
    if "global" in want:
        out["global"] = (_softmax(g), g)
    if "bass" in want and notes:
        b = ks_scores(_line_hist(notes, tpb, True))
        out["bass"] = (_softmax(b), b)
    if "melody" in want and notes:
        m = ks_scores(_line_hist(notes, tpb, False))
        out["melody"] = (_softmax(m), m)
    flags = []
    if "sig" in want and keysigs:
        sk = _sig_key(keysigs[0][1])
        if sk:
            t, mode = sk
            rel = ((t + 9) % 12, "min") if mode == "maj" else ((t + 3) % 12, "maj")
            gbest = max(g, key=g.get)
            default_c = sk in ((0, "maj"), (9, "min"))
            if default_c and gbest not in ((0, "maj"), (9, "min")):
                flags.append("sig-defecto")        # Do mayor «por defecto»: se abstiene
            else:
                pair = [sk, rel]
                a, bq = (pair[0], pair[1]) if g[pair[0]] >= g[pair[1]] else (pair[1], pair[0])
                d = {k: 0.01 / 22 for k in g}
                d[a], d[bq] = 0.62, 0.37
                out["sig"] = (d, None)
    if "ending" in want and notes:
        last = max(e for _, e, _, _ in notes)
        tail = [n for n in notes if n[1] >= last - 2 * tpb]
        if tail:
            bass_pc = min(tail, key=lambda n: n[2])[2] % 12
            mode = "min" if g[(bass_pc, "min")] > g[(bass_pc, "maj")] else "maj"
            d = {k: 0.3 / 23 for k in g}
            d[(bass_pc, mode)] = 0.7
            out["ending"] = (d, None)
    return out, flags


def combine(votes):
    keys = [(t, m) for t in range(12) for m in ("maj", "min")]
    tot = {k: 0.0 for k in keys}
    wsum = 0.0
    for name, (dist, _raw) in votes.items():
        w = VOTER_WEIGHT.get(name, 1.0)
        wsum += w
        for k in keys:
            tot[k] += w * dist[k]
    if wsum <= 0:
        return None
    return {k: v / wsum for k, v in tot.items()}


def _related(a, b):
    """0 idéntica, 1 relativa, 2 no emparentada (paralela cuenta como 2)."""
    if a == b:
        return 0
    (ta, ma), (tb, mb) = a, b
    if ma != mb:
        mj, mn = (a, b) if ma == "maj" else (b, a)
        if (mn[0] - mj[0]) % 12 == 9:
            return 1
    return 2


def consensus_notes(notes, tpb, keysigs):
    """Núcleo: devuelve dict con tonalidad, confianza, flags, votantes."""
    if not notes:
        return None
    votes, flags = votes_for(notes, tpb, keysigs)
    comb = combine(votes)
    ranked = sorted(comb.items(), key=lambda kv: -kv[1])
    (kb, pb), (k2, p2) = ranked[0], ranked[1]
    g_raw = votes["global"][1]
    r_best = max(g_raw.values())
    # Votantes
    voters = {}
    agree = 0.0
    for name, (dist, raw) in votes.items():
        top = max(dist, key=dist.get)
        rel = _related(top, kb)
        agree += 1.0 if rel == 0 else (0.5 if rel == 1 else 0.0)
        voters[name] = {"key": key_name(*top), "p": round(dist[top], 3),
                        "r": round(raw[top], 3) if raw else None, "relation": rel}
    agreement = agree / len(votes)
    eff = _effective_pcs(_hist(notes))
    margin = pb - p2
    if eff < 3.5:
        flags.append("estático")
    if _related(kb, k2) == 1 and margin < 0.10:
        flags.append("rel-ambig")
    if r_best < 0.60:
        flags.append("modal")
    if agreement < 0.55:
        flags.append("desacuerdo")
    soft = [f for f in flags if f != "sig-defecto"]
    if not soft and pb >= 0.55 and agreement >= 0.75:
        rel_lbl = "alta"
    elif len(soft) >= 2 or (agreement < 0.5) or "estático" in soft:
        rel_lbl = "baja"
    else:
        rel_lbl = "media"
    return {"key": key_name(*kb), "tonic": kb[0], "mode": kb[1], "confidence": round(pb, 3),
            "margin": round(margin, 3), "runner_up": key_name(*k2),
            "agreement": round(agreement, 2), "reliability": rel_lbl, "flags": flags,
            "eff_pcs": round(eff, 1), "voters": voters,
            "ranking": [(key_name(*k), round(p, 3)) for k, p in ranked[:5]]}


def consensus_file(path, drums=False):
    d = load_midi(path, drums=drums)
    r = consensus_notes(d["notes"], d["tpb"], d["keysigs"])
    if r is not None:
        r["file"] = path
    return r


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def _rel_color(r):
    return {"alta": _c("GRN"), "media": _c("YEL"), "baja": _c("RED")}.get(r, "")


def cmd_detect(args):
    results, rc = [], 0
    print(f"\n{_c('B')}{'archivo':<44}{'tonalidad':<10}{'conf':>6}  {'acuerdo':>7}  fiab   banderas{_c('R')}")
    for p in args.midis:
        try:
            r = consensus_file(p, drums=args.drums)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(p)[:43]:<44}{_c('RED')}ilegible: {type(e).__name__}: {e}{_c('R')}")
            rc = 2
            continue
        if r is None:
            print(f"{os.path.basename(p)[:43]:<44}(sin notas)")
            continue
        results.append(r)
        print(f"{os.path.basename(p)[:43]:<44}{r['key']:<10}{r['confidence']:>6.2f}  "
              f"{r['agreement']:>7.2f}  {_rel_color(r['reliability'])}{r['reliability']:<5}{_c('R')}  "
              f"{','.join(r['flags'])}")
    print()
    if args.json:
        _ensure_parent(args.json)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    return rc


def cmd_compare(args):
    try:
        d = load_midi(args.midi)
    except Exception as e:  # noqa: BLE001
        print(f"Ilegible: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    r = consensus_notes(d["notes"], d["tpb"], d["keysigs"])
    if r is None:
        print("Sin notas.")
        return 0
    print(f"\n{_c('B')}{os.path.basename(args.midi)}{_c('R')}")
    print(f"  Consenso: {_c('B')}{r['key']}{_c('R')}  conf={r['confidence']:.2f}  margen={r['margin']:.2f}  "
          f"fiabilidad={_rel_color(r['reliability'])}{r['reliability']}{_c('R')}  "
          f"banderas={','.join(r['flags']) or '—'}")
    print(f"  Armadura MIDI: {d['keysigs'][0][1] if d['keysigs'] else '—'}   "
          f"clases de altura efectivas: {r['eff_pcs']}\n")
    print(f"  {'votante':<9}{'propone':<9}{'prob':>6}{'r(KK)':>8}  relación con el consenso")
    names = {0: "coincide", 1: "relativa", 2: "DISCREPA"}
    for name, v in r["voters"].items():
        rr = f"{v['r']:.2f}" if v["r"] is not None else "—"
        col = _c("GRN") if v["relation"] == 0 else (_c("YEL") if v["relation"] == 1 else _c("RED"))
        print(f"  {name:<9}{v['key']:<9}{v['p']:>6.2f}{rr:>8}  {col}{names[v['relation']]}{_c('R')}")
    print(f"\n  Top {args.top}: " + "  ".join(f"{k} {p:.2f}" for k, p in r["ranking"][:args.top]))
    print()
    return 0


def cmd_sections(args):
    try:
        d = load_midi(args.midi)
    except Exception as e:  # noqa: BLE001
        print(f"Ilegible: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    tpb = d["tpb"]
    num, den = (d["tsigs"][0][1], d["tsigs"][0][2]) if d["tsigs"] else (4, 4)
    bar = int(tpb * 4 * num / den)
    if not d["notes"]:
        print("Sin notas.")
        return 0
    last = max(e for _, e, _, _ in d["notes"])
    nbars = int(math.ceil(last / bar))
    win = args.bars
    secs = []
    for b0 in range(0, nbars, win):
        t0, t1 = b0 * bar, min(nbars, b0 + win) * bar
        nn = [n for n in d["notes"] if n[0] < t1 and n[1] > t0]
        if len(nn) < 3:
            continue
        # ventana: solo votantes de contenido (la armadura/final no son locales)
        v, _ = votes_for(nn, tpb, [], want=("global", "bass"))
        comb = combine(v)
        ranked = sorted(comb.items(), key=lambda kv: -kv[1])
        k, p = ranked[0]
        g = max(v["global"][1].values())
        secs.append({"bar_start": b0 + 1, "bar_end": min(nbars, b0 + win), "key": key_name(*k),
                     "confidence": round(p, 3), "r": round(g, 2),
                     "static": _effective_pcs(_hist(nn)) < 3.5})
    merged = []
    for s in secs:
        if merged and merged[-1]["key"] == s["key"]:
            merged[-1]["bar_end"] = s["bar_end"]
            merged[-1]["confidence"] = round((merged[-1]["confidence"] + s["confidence"]) / 2, 3)
            merged[-1]["static"] = merged[-1]["static"] and s["static"]
        else:
            merged.append(dict(s))
    print(f"\n{_c('B')}{os.path.basename(args.midi)}{_c('R')}  ({nbars} compases, ventana {win})\n")
    for s in merged:
        note = "  (estático)" if s["static"] else ""
        print(f"  compases {s['bar_start']:>4}–{s['bar_end']:<4} {s['key']:<7} conf={s['confidence']:.2f}{note}")
    keys = [s["key"] for s in merged]
    print(f"\n  Modulaciones: {max(0, len(keys) - 1)}   ·   tonalidades: {', '.join(dict.fromkeys(keys))}\n")
    if args.json:
        _ensure_parent(args.json)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=1)
    return 0


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(prog="key_consensus.py",
                                 description="Tonalidad consensuada con confianza y banderas de fiabilidad.")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    d = sub.add_parser("detect", help="Tonalidad consensuada de uno o varios MIDIs")
    d.add_argument("midis", nargs="+")
    d.add_argument("--json")
    d.add_argument("--drums", action="store_true")
    d.add_argument("--no-color", action="store_true")
    d.set_defaults(fn=cmd_detect)

    c = sub.add_parser("compare", help="Votantes y discrepancias de un MIDI")
    c.add_argument("midi")
    c.add_argument("--top", type=int, default=3)
    c.add_argument("--no-color", action="store_true")
    c.set_defaults(fn=cmd_compare)

    s = sub.add_parser("sections", help="Tonalidad por secciones")
    s.add_argument("midi")
    s.add_argument("--bars", type=int, default=8)
    s.add_argument("--json")
    s.add_argument("--no-color", action="store_true")
    s.set_defaults(fn=cmd_sections)

    args = ap.parse_args(argv)
    if getattr(args, "no_color", False):
        _USE_COLOR = False
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
