#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                         CHORD EXPORT  v1.0                                   ║
║        De un MIDI de acordes (o de una obra) a la sintaxis de --chords       ║
║                                                                              ║
║  Cierra el bucle análisis → generación: lee la armonía REAL de un MIDI       ║
║  (la salida de midi_split_harmony.py, un *.chords.mid de                     ║
║  chord_progression_generator.py, o una obra completa) y la escribe como      ║
║  texto que aceptan --chords de bass_line_composer, voice_leader,             ║
║  melody_adapter, leadsheet_realizer, reharmonizer, etc.                      ║
║  Hace la operación inversa a `--export-text` / `--export-midi`.              ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    text  obra.mid            Imprime "Am:4 F:4 C:4 G7:4" (--chords)          ║
║    bars  obra.mid            Tabla compás a compás con confianza             ║
║    json  obra.mid            Todo el detalle en JSON (acordes, notas, key)   ║
║    selftest                  Comprobaciones internas                         ║
║                                                                              ║
║  EJEMPLOS:                                                                   ║
║    python chord_export.py text acordes.mid                                   ║
║    python chord_export.py text acordes.mid --roman --key C                   ║
║    python chord_export.py text obra.mid --track 2 --window half -o a.txt     ║
║    python chord_export.py bars obra.mid --simplify triads                    ║
║    python chord_export.py json obra.mid -o armonia.json                      ║
║                                                                              ║
║    # Reutilizar la armonía capturada del original:                           ║
║    python midi_split_harmony.py orig.mid                                     ║
║    python chord_export.py text orig_chords.mid -o acordes.txt                ║
║    python bass_line_composer.py --chords "$(cat acordes.txt)" --key Am       ║
║                                                                              ║
║  OPCIONES (comunes):                                                         ║
║    --track N / --channel N   Fuente de acordes (default: todas, sin batería) ║
║    --drop-top                Quita la nota más aguda de cada ataque          ║
║                              (descarta la melodía en partituras de piano)    ║
║    --window auto|bar|half|beat|N   Resolución. auto (default): un acorde por ║
║                              compás, o dos si dividirlo ajusta claramente    ║
║                              mejor; bar/half/beat/N: ventana fija (negras)   ║
║    --time-sig N/D            Fuerza el compás (default: el del MIDI, 4/4)    ║
║    --offset B                Desplaza el inicio de los compases B negras     ║
║                              (anacrusis)                                     ║
║    --simplify none|sevenths|triads   Reduce la calidad (default: none)       ║
║    --key K | auto           Tonalidad para ortografía/romanos (default auto)║
║    --roman                   Grados (I vi IV V7) en vez de nombres           ║
║    --no-merge                No fusiona ventanas consecutivas iguales        ║
║    --explicit                Escribe siempre ":beats" (default: omite :4)    ║
║    -o FILE                   Escribe la salida en un fichero                 ║
║                                                                              ║
║  NOTAS:                                                                      ║
║    · Las duraciones (":n") están en negras, como en bass_line_composer.      ║
║    · Un compás sin notas hereda el acorde anterior.                          ║
║    · Si el compás declarado en el MIDI es erróneo (p.ej. 1/4) usa            ║
║      --time-sig, o repáralo antes con midi_meter.py fix.                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import argparse
import json
import math
import sys
import tempfile
import os
from collections import defaultdict
from pathlib import Path

try:
    import mido
except ImportError:                                   # pragma: no cover
    sys.exit("ERROR: requiere mido (pip install mido)")

VERSION = "1.0"

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_NAME_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
FLAT_KEYS_MAJOR = {5, 10, 3, 8, 1, 6}          # F Bb Eb Ab Db Gb
FLAT_KEYS_MINOR = {2, 7, 0, 5, 10, 3}          # Dm Gm Cm Fm Bbm Ebm

# (sufijo, intervalos, penalización por complejidad)
TEMPLATES = [
    ("",      (0, 4, 7),     0.00),
    ("m",     (0, 3, 7),     0.00),
    ("7",     (0, 4, 7, 10), 0.05),
    ("maj7",  (0, 4, 7, 11), 0.06),
    ("m7",    (0, 3, 7, 10), 0.05),
    ("dim",   (0, 3, 6),     0.04),
    ("dim7",  (0, 3, 6, 9),  0.08),
    ("m7b5",  (0, 3, 6, 10), 0.07),
    ("aug",   (0, 4, 8),     0.08),
    ("sus4",  (0, 5, 7),     0.07),
    ("sus2",  (0, 2, 7),     0.08),
    ("6",     (0, 4, 7, 9),  0.15),
    ("m6",    (0, 3, 7, 9),  0.15),
    ("5",     (0, 7),        0.10),
]
SEVENTH_TO_TRIAD = {"7": "", "maj7": "", "m7": "m", "m7b5": "dim", "dim7": "dim",
                    "6": "", "m6": "m"}
SIMPLE_TRIAD_SET = {"", "m", "dim", "aug", "sus4", "sus2", "5"}

ROMAN = ["I", "bII", "II", "bIII", "III", "IV", "#IV", "V", "bVI", "VI", "bVII", "VII"]


# ─────────────────────────────────────────────────────────────────────────────
#  Lectura MIDI
# ─────────────────────────────────────────────────────────────────────────────

class Note:
    __slots__ = ("start", "end", "pitch", "vel", "ch", "trk")

    def __init__(self, start, end, pitch, vel, ch, trk):
        self.start, self.end, self.pitch, self.vel, self.ch, self.trk = \
            start, end, pitch, vel, ch, trk


def read_notes(path):
    """Devuelve (notas en negras, (num, den) del primer compás, tpb)."""
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat or 480
    notes, ts = [], None
    for ti, trk in enumerate(mid.tracks):
        t = 0
        pend = {}
        for m in trk:
            t += m.time
            if m.type == "time_signature" and ts is None:
                ts = (m.numerator, m.denominator)
            elif m.type == "note_on" and m.velocity > 0:
                pend.setdefault((m.channel, m.note), []).append((t, m.velocity))
            elif m.type in ("note_off", "note_on"):
                st = pend.get((m.channel, m.note))
                if st:
                    t0, v = st.pop(0)
                    if t > t0:
                        notes.append(Note(t0 / tpb, t / tpb, m.note, v, m.channel, ti))
        for (ch, p), lst in pend.items():          # notas sin cerrar
            for t0, v in lst:
                notes.append(Note(t0 / tpb, t / tpb + 1.0, p, v, ch, ti))
    return notes, ts, tpb


def select_source(notes, track=None, channel=None, drop_top=False):
    out = [n for n in notes if n.ch != 9]
    if track is not None:
        out = [n for n in out if n.trk == track]
    if channel is not None:
        out = [n for n in out if n.ch == channel]
    if drop_top and out:
        by = defaultdict(list)
        for n in out:
            by[round(n.start, 3)].append(n)
        keep = []
        for lst in by.values():
            if len(lst) > 1:
                top = max(lst, key=lambda n: n.pitch)
                keep += [n for n in lst if n is not top]
            else:
                keep += lst
        out = keep
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Tonalidad (Krumhansl-Schmuckler) y ortografía
# ─────────────────────────────────────────────────────────────────────────────

_KK_MAJ = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MIN = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _corr(a, b):
    ma, mb = sum(a) / 12, sum(b) / 12
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else 0.0


def detect_key(notes):
    w = [0.0] * 12
    for n in notes:
        w[n.pitch % 12] += n.end - n.start
    if not any(w):
        return 0, "major"
    best = (-2, 0, "major")
    for r in range(12):
        rot = [w[(r + i) % 12] for i in range(12)]
        for mode, prof in (("major", _KK_MAJ), ("minor", _KK_MIN)):
            c = _corr(rot, prof)
            if c > best[0]:
                best = (c, r, mode)
    return best[1], best[2]


def parse_key(text):
    """'C', 'Am', 'F#m', 'Bb' → (pc, mode)."""
    s = text.strip()
    minor = s.endswith("m") and not s.lower().endswith("maj")
    if minor:
        s = s[:-1]
    if not s or s[0].upper() not in _NAME_TO_PC:
        raise ValueError(f"tonalidad no válida: {text!r}")
    pc = _NAME_TO_PC[s[0].upper()] + s[1:].count("#") - s[1:].count("b")
    return pc % 12, ("minor" if minor else "major")


def use_flats(key_pc, mode):
    return key_pc in (FLAT_KEYS_MINOR if mode == "minor" else FLAT_KEYS_MAJOR)


# ─────────────────────────────────────────────────────────────────────────────
#  Detección de acordes por plantillas
# ─────────────────────────────────────────────────────────────────────────────

def window_weights(notes, t0, t1):
    """Peso por clase de altura en [t0, t1) y nota más grave presente."""
    w = [0.0] * 12
    lowest = None
    for n in notes:
        a, b = max(n.start, t0), min(n.end, t1)
        if b <= a:
            continue
        w[n.pitch % 12] += (b - a) * (0.5 + n.vel / 254.0)
        if lowest is None or n.pitch < lowest:
            lowest = n.pitch
    return w, lowest


def classify(w, lowest, allow=None):
    """Mejor (raíz, sufijo, confianza) para un vector de pesos."""
    total = sum(w)
    if total <= 0:
        return None
    bass_pc = lowest % 12 if lowest is not None else None
    scored = []
    for root in range(12):
        if w[root] < 0.08 * total:
            continue
        for suffix, ivs, pen in TEMPLATES:
            if allow is not None and suffix not in allow:
                continue
            pcs = {(root + i) % 12 for i in ivs}
            inw = sum(w[p] for p in pcs)
            outw = total - inw
            missing = sum(1 for p in pcs if w[p] < 0.04 * total)
            s = (inw - 1.2 * outw) / total - pen - 0.18 * missing
            if bass_pc == root:
                s += 0.06
            scored.append((s, root, suffix))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    second = scored[1][0] if len(scored) > 1 else best[0] - 0.5
    conf = max(0.0, min(1.0, 0.5 + (best[0] - second) + 0.3 * best[0]))
    return best[1], best[2], round(conf, 3), best[0]


def spell(root, suffix, flats):
    return (FLAT_NAMES if flats else SHARP_NAMES)[root] + suffix


def to_roman(root, suffix, key_pc):
    deg = ROMAN[(root - key_pc) % 12]
    minorish = suffix.startswith("m") and not suffix.startswith("maj") or suffix in ("dim", "dim7", "m7b5")
    base = deg.lower() if minorish else deg
    extra = {"": "", "m": "", "7": "7", "maj7": "maj7", "m7": "7", "dim": "°",
             "dim7": "°7", "m7b5": "ø7", "aug": "+", "sus4": "sus4", "sus2": "sus2",
             "6": "6", "m6": "6", "5": "5"}[suffix]
    return base + extra


def simplify_suffix(suffix, mode):
    if mode == "triads":
        return SEVENTH_TO_TRIAD.get(suffix, suffix)
    if mode == "sevenths":
        return suffix
    return suffix


# ─────────────────────────────────────────────────────────────────────────────
#  Análisis principal
# ─────────────────────────────────────────────────────────────────────────────

def parse_ts(text):
    a, b = text.split("/")
    return int(a), int(b)


def window_len(spec, bar_beats):
    s = str(spec).lower()
    if s == "auto":
        return bar_beats
    if s == "bar":
        return bar_beats
    if s == "half":
        return bar_beats / 2.0
    if s == "beat":
        return 1.0
    try:
        v = float(s)
        if v <= 0:
            raise ValueError
        return v
    except ValueError:
        raise ValueError(f"--window no válido: {spec!r} (auto|bar|half|beat|N)")


def analyze(path, track=None, channel=None, drop_top=False, window="auto",
            time_sig=None, offset=0.0, simplify="none", key=None, merge=True):
    notes, ts_decl, tpb = read_notes(path)
    src = select_source(notes, track, channel, drop_top)
    if not src:
        raise ValueError("no hay notas en la fuente seleccionada")
    ts = parse_ts(time_sig) if time_sig else (ts_decl or (4, 4))
    bar_beats = ts[0] * 4.0 / ts[1]
    if bar_beats <= 0:
        raise ValueError("compás no válido")
    wlen = window_len(window, bar_beats)

    if key and key != "auto":
        kpc, kmode = parse_key(key)
    else:
        kpc, kmode = detect_key(src)
    flats = use_flats(kpc, kmode)

    t_end = max(n.end for n in src)
    t_start = offset
    n_win = max(1, int(math.ceil((t_end - t_start) / wlen - 1e-9)))
    allow = SIMPLE_TRIAD_SET if simplify == "triads" else None
    auto = str(window).lower() == "auto"

    def detect(a, b):
        w, low = window_weights(src, a, b)
        res = classify(w, low, allow)
        return w, res

    wins = []
    prev = None

    def push(a, b, w, res):
        nonlocal prev
        inherited = res is None
        if inherited:
            if prev is None:
                wins.append(None)
                return
            root, suffix, conf = prev[0], prev[1], 0.0
        else:
            root, suffix, conf = res[0], res[1], res[2]
            suffix = simplify_suffix(suffix, simplify)
        prev = (root, suffix)
        tot = sum(w) or 1.0
        wins.append({"index": len(wins), "start_beat": a, "beats": b - a, "root": root,
                     "suffix": suffix, "name": spell(root, suffix, flats),
                     "roman": to_roman(root, suffix, kpc), "confidence": conf,
                     "inherited": inherited,
                     "pcs": sorted(p for p in range(12) if w[p] > 0.05 * tot)})

    def best_split(a, b, depth):
        """Segmentación recursiva (compás → mitades → pulsos). Devuelve lista de
        (a, b, pesos, resultado). Divide solo si las dos mitades son acordes
        DISTINTOS y ambos ajustan bien (mejor que un único acorde con extensiones)."""
        w, res = detect(a, b)
        whole = [(a, b, w, res)]
        if res is None or (b - a) < 2.0 - 1e-9 or depth >= 2:
            return whole
        m = (a + b) / 2.0
        w1, r1 = detect(a, m)
        w2, r2 = detect(m, b)
        if r1 is None or r2 is None:
            return whole
        wt1, wt2 = sum(w1), sum(w2)
        avg = (r1[3] * wt1 + r2[3] * wt2) / (wt1 + wt2)
        if res[3] < 0.55:
            # el tramo entero ajusta mal: se mira más abajo (hasta el pulso)
            segs = best_split(a, m, depth + 1) + best_split(m, b, depth + 1)
            tw = sum(sum(sw) for _, _, sw, _ in segs) or 1.0
            leaf = sum((sr[3] if sr else 0.0) * sum(sw) for _, _, sw, sr in segs) / tw
            return segs if leaf > res[3] + 0.15 else whole
        if (r1[0], r1[1]) == (r2[0], r2[1]):
            return whole
        fit = 0.6 if depth == 0 else 0.7
        if min(r1[3], r2[3]) < fit or avg < res[3] - 0.10:
            return whole
        return best_split(a, m, depth + 1) + best_split(m, b, depth + 1)

    for i in range(n_win):
        a, b = t_start + i * wlen, t_start + (i + 1) * wlen
        if auto and wlen >= 2.0:
            for (sa, sb, sw, sr) in best_split(a, b, 0):
                push(sa, sb, sw, sr)
        else:
            w, res = detect(a, b)
            push(a, b, w, res)
    # ventanas iniciales vacías: se rellenan con el primer acorde real
    first = next((x for x in wins if x), None)
    if first is None:
        raise ValueError("no se pudo detectar ningún acorde")
    wins = [x if x else dict(first, index=i, inherited=True, confidence=0.0,
                             start_beat=t_start + i * wlen, beats=wlen)
            for i, x in enumerate(wins)]
    for i, x in enumerate(wins):
        x["index"] = i

    chords = []
    for x in wins:
        if merge and chords and chords[-1]["name"] == x["name"]:
            chords[-1]["beats"] += x["beats"]
            chords[-1]["confidence"] = round(
                (chords[-1]["confidence"] * chords[-1]["n"] + x["confidence"]) / (chords[-1]["n"] + 1), 3)
            chords[-1]["n"] += 1
        else:
            chords.append({"name": x["name"], "roman": x["roman"], "beats": x["beats"],
                           "start_beat": x["start_beat"], "confidence": x["confidence"], "n": 1})
    for c in chords:
        c.pop("n", None)
    return {
        "file": str(path), "tool": "chord_export", "version": VERSION,
        "time_signature": f"{ts[0]}/{ts[1]}", "bar_beats": bar_beats,
        "time_signature_declared": f"{ts_decl[0]}/{ts_decl[1]}" if ts_decl else None,
        "window_beats": wlen, "window": str(window), "offset": offset,
        "key": f"{(FLAT_NAMES if flats else SHARP_NAMES)[kpc]}{'m' if kmode == 'minor' else ''}",
        "key_pc": kpc, "key_mode": kmode, "simplify": simplify,
        "n_source_notes": len(src), "windows": wins, "chords": chords,
    }


def fmt_beats(b):
    return str(int(b)) if abs(b - round(b)) < 1e-9 else f"{b:.3f}".rstrip("0").rstrip(".")


def render_text(res, roman=False, explicit=False):
    toks = []
    for c in res["chords"]:
        name = c["roman"] if roman else c["name"]
        if explicit or abs(c["beats"] - 4.0) > 1e-9:
            toks.append(f"{name}:{fmt_beats(c['beats'])}")
        else:
            toks.append(name)
    return " ".join(toks)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def _run(a):
    return analyze(a.midi, track=a.track, channel=a.channel, drop_top=a.drop_top,
                   window=a.window, time_sig=a.time_sig, offset=a.offset,
                   simplify=a.simplify, key=a.key, merge=not a.no_merge)


def _emit(text, out):
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
        print(f"  → {out}", file=sys.stderr)
    else:
        print(text)


def cmd_text(a):
    res = _run(a)
    if a.roman and a.key == "auto":
        print(f"[AVISO] --roman con tonalidad detectada: {res['key']}", file=sys.stderr)
    _emit(render_text(res, a.roman, a.explicit), a.output)
    return 0


def cmd_bars(a):
    res = _run(a)
    lines = [f"  {res['file']}   compás {res['time_signature']}"
             + (f" (declarado {res['time_signature_declared']})"
                if res["time_signature_declared"] not in (None, res["time_signature"]) else "")
             + f"   tonalidad {res['key']}   fuente: {res['n_source_notes']} notas",
             f"  {'#':>4} {'inicio':>7} {'acorde':<9} {'grado':<8} {'conf':>5}  notas"]
    for w in res["windows"]:
        pcs = " ".join(SHARP_NAMES[p] for p in w["pcs"])
        lines.append(f"  {w['index'] + 1:>4} {w['start_beat']:>7.2f} {w['name']:<9} "
                     f"{w['roman']:<8} {w['confidence']:>5.2f}  {pcs}"
                     + ("  (hereda)" if w["inherited"] else ""))
    avg = sum(w["confidence"] for w in res["windows"]) / len(res["windows"])
    lines.append(f"  confianza media: {avg:.2f}   acordes distintos: "
                 f"{len({w['name'] for w in res['windows']})}")
    _emit("\n".join(lines), a.output)
    return 0


def cmd_json(a):
    res = _run(a)
    res["text"] = render_text(res, False, True)
    res["text_roman"] = render_text(res, True, True)
    _emit(json.dumps(res, ensure_ascii=False, indent=1), a.output)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Selftest
# ─────────────────────────────────────────────────────────────────────────────

def _build_test_midi(path, chords, tpb=480, bar_beats=4, ts=(4, 4), melody=False):
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("time_signature", numerator=ts[0], denominator=ts[1]))
    mid.tracks.append(tr)
    events = []
    t = 0
    for name, beats in chords:
        root, suffix = _parse_name(name)
        ivs = next(i for s, i, _ in TEMPLATES if s == suffix)
        pitches = [36 + root] + [60 + root + i for i in ivs]
        for p in pitches:
            events.append((t, "on", p))
            events.append((t + int(beats * tpb), "off", p))
        if melody:
            events.append((t, "on", 84)); events.append((t + int(beats * tpb), "off", 84))
        t += int(beats * tpb)
    events.sort(key=lambda e: (e[0], e[1] == "on"))
    last = 0
    for tt, kind, p in events:
        tr.append(mido.Message("note_on" if kind == "on" else "note_off", note=p,
                               velocity=80 if kind == "on" else 0, time=tt - last))
        last = tt
    mid.save(path)


def _parse_name(name):
    if len(name) > 1 and name[1] in "#b":
        r, s = name[:2], name[2:]
    else:
        r, s = name[:1], name[1:]
    pc = _NAME_TO_PC[r[0]] + (1 if "#" in r else 0) - (1 if "b" in r else 0)
    return pc % 12, s


def cmd_selftest(a):
    ok = True

    def check(cond, msg):
        nonlocal ok
        print(("  ✔ " if cond else "  ✘ ") + msg)
        ok &= bool(cond)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mid")
        prog = [("C", 4), ("Am", 4), ("F", 4), ("G7", 4), ("Dm7", 4), ("G", 4), ("Cmaj7", 4)]
        _build_test_midi(p, prog)
        res = analyze(p, key="C")
        got = [c["name"] for c in res["chords"]]
        check(got == ["C", "Am", "F", "G7", "Dm7", "G", "Cmaj7"], f"acordes en 4/4 → {got}")
        check(render_text(res) == "C Am F G7 Dm7 G Cmaj7", "texto sin ':4' redundante")
        res2 = analyze(p, key="C", simplify="triads")
        check([c["name"] for c in res2["chords"]] == ["C", "Am", "F", "G", "Dm", "G", "C"],
              "simplify triads")
        check(render_text(analyze(p, key="C"), roman=True) == "I vi IV V7 ii7 V Imaj7",
              "grados romanos")
        prog2 = [("Bb", 3), ("Gm", 3), ("Eb", 6), ("F7", 3)]
        _build_test_midi(p, prog2, ts=(3, 4))
        res3 = analyze(p, key="Bb")
        check(res3["time_signature"] == "3/4" and
              [c["name"] for c in res3["chords"]] == ["Bb", "Gm", "Eb", "F7"],
              "3/4, ortografía en bemoles y fusión de ventanas")
        check(render_text(res3).split()[2] == "Eb:6", "duración fusionada Eb:6")
        _build_test_midi(p, [("C", 4), ("G", 4)], melody=True)
        r4 = analyze(p, key="C", drop_top=True)
        check([c["name"] for c in r4["chords"]] == ["C", "G"], "drop-top con melodía")
        # medio compás
        _build_test_midi(p, [("C", 2), ("G", 2), ("Am", 2), ("F", 2)])
        r5 = analyze(p, key="C", window="half")
        check(render_text(r5) == "C:2 G:2 Am:2 F:2", "ventana half → C:2 G:2 Am:2 F:2")
        # roundtrip con bass_line_composer.parse_chords_text si está disponible
    print("  SELFTEST OK" if ok else "  SELFTEST FALLÓ")
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(prog="chord_export.py",
                                 description="MIDI de acordes → sintaxis de --chords")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("midi")
    common.add_argument("--track", type=int, default=None)
    common.add_argument("--channel", type=int, default=None)
    common.add_argument("--drop-top", action="store_true")
    common.add_argument("--window", default="auto", help="auto|bar|half|beat|N (negras)")
    common.add_argument("--time-sig", default=None, metavar="N/D")
    common.add_argument("--offset", type=float, default=0.0, metavar="BEATS")
    common.add_argument("--simplify", choices=["none", "sevenths", "triads"], default="none")
    common.add_argument("--key", default="auto")
    common.add_argument("--no-merge", action="store_true")
    common.add_argument("-o", "--output", default=None)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("text", help="Imprime la progresión como texto de --chords", parents=[common])
    p.add_argument("--roman", action="store_true")
    p.add_argument("--explicit", action="store_true")
    p.set_defaults(fn=cmd_text)
    p = sp.add_parser("bars", help="Tabla compás a compás", parents=[common])
    p.set_defaults(fn=cmd_bars)
    p = sp.add_parser("json", help="Detalle completo en JSON", parents=[common])
    p.set_defaults(fn=cmd_json)
    p = sp.add_parser("selftest", help="Comprobaciones internas")
    p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except (ValueError, OSError, EOFError, KeyError, IndexError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
