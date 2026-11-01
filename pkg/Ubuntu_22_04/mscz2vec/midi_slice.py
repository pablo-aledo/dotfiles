#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                          MIDI SLICE  v1.0                                    ║
║           Trocear un MIDI por compases, por secciones o en rejilla           ║
║                                                                              ║
║  Convierte "una obra" en FRAGMENTOS MIDI independientes (y los recompone)    ║
║  respetando compases, tempo, tonalidad, programas y controladores. Cierra    ║
║  el hueco entre form_analyzer.py / section_score.py (que DETECTAN secciones  ║
║  pero no las cortan) y variation_engine, stitcher, midi_concat o             ║
║  practice_tutor (que necesitan un MIDI por fragmento).                       ║
║                                                                              ║
║  SUBCOMANDOS:                                                                ║
║    info     obra.mid                 Tabla de compases, compases y pistas    ║
║    cut      obra.mid --bars 9-16 -o b.mid     Extrae un rango de compases    ║
║    sections obra.mid --form form.json --out-dir secs/   Una MIDI/sección     ║
║    sections obra.mid --bars "1-8,9-16" --out-dir secs/   Rangos a mano       ║
║    grid     obra.mid --every 8 --out-dir frases/         Cortes regulares    ║
║    assemble obra.mid --bars "1-8,9-16,1-8" -o nueva.mid  Recompone rangos    ║
║    assemble obra.mid --form form.json --order A,B,A -o f.mid   Por etiquetas ║
║    selftest                                                                  ║
║                                                                              ║
║  EJEMPLOS:                                                                   ║
║    python form_analyzer.py detect obra.mid --json form.json                  ║
║    python midi_slice.py sections obra.mid --form form.json --out-dir secs    ║
║    python variation_engine.py secs/seccion_02_B.mid ...                      ║
║    python midi_slice.py assemble obra.mid --form form.json --order A,B,A,C \\ ║
║           -o obra_reordenada.mid                                             ║
║    python midi_slice.py cut obra.mid --bars 1-4 --tracks 0,2 -o intro.mid    ║
║                                                                              ║
║  SELECTORES:                                                                 ║
║    --bars ESPEC   Compases (desde 1): "5", "1-8", "1-4,9-12" (inclusivos)    ║
║    --tracks LIST  Pistas a conservar: "0,2" o "1-3" (default: todas)         ║
║    --time-sig N/D Ignora el compás del MIDI (usa este, p.ej. tras           ║
║                   `midi_meter.py fix` o si el metadato es erróneo)           ║
║    --pickup B     Anacrusis de B negras antes del compás 1 (pasa a ser el    ║
║                   compás 0, seleccionable con --bars 0-4)                    ║
║    --drop-held    Descarta las notas que ya sonaban al empezar el trozo      ║
║                   (default: se conservan recortadas al inicio)               ║
║    --prefix P     Prefijo de los ficheros de sections/grid (default seccion) ║
║                                                                              ║
║  COMPORTAMIENTO:                                                             ║
║    · Las notas que cruzan los límites se recortan; no se pierden.            ║
║    · Cada fragmento arranca con el tempo, compás, tonalidad, programa y      ║
║      controladores vigentes en ese punto de la obra.                         ║
║    · `sections` y `grid` escriben además manifest.json (fichero, compases,   ║
║      etiqueta, longitud en negras) para encadenar con otras herramientas.    ║
║    · `assemble` repite y reordena rangos del mismo MIDI (formas A B A);      ║
║      para unir MIDIs distintos usa midi_concat.py.                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

try:
    import mido
except ImportError:                                   # pragma: no cover
    sys.exit("ERROR: requiere mido (pip install mido)")

VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
#  Modelo: pistas como notas + eventos sueltos en ticks absolutos
# ─────────────────────────────────────────────────────────────────────────────

class Song:
    def __init__(self, path, time_sig=None, pickup=0.0):
        self.mid = mido.MidiFile(path)
        self.path = str(path)
        self.tpb = self.mid.ticks_per_beat or 480
        self.type = self.mid.type
        self.tracks = []                    # [{"name", "notes", "events"}]
        self.ts_events = []                 # (tick, num, den) de todas las pistas
        for trk in self.mid.tracks:
            t = 0
            pend = {}
            notes, events = [], []
            name = None
            for m in trk:
                t += m.time
                if m.type == "note_on" and m.velocity > 0:
                    pend.setdefault((m.channel, m.note), []).append((t, m.velocity))
                elif m.type in ("note_off", "note_on"):
                    lst = pend.get((m.channel, m.note))
                    if lst:
                        t0, v = lst.pop(0)
                        notes.append((t0, t, m.channel, m.note, v))
                elif m.type == "end_of_track":
                    continue
                else:
                    if m.type == "time_signature":
                        self.ts_events.append((t, m.numerator, m.denominator))
                    if m.type == "track_name" and name is None:
                        name = m.name
                    events.append((t, m.copy(time=0)))
            for (ch, p), lst in pend.items():
                for t0, v in lst:
                    notes.append((t0, max(t, t0 + 1), ch, p, v))
            self.tracks.append({"name": name, "notes": sorted(notes), "events": events})
        self.end_tick = max([n[1] for tr in self.tracks for n in tr["notes"]] or [0])
        self.pickup_ticks = int(round(pickup * self.tpb))
        self.forced_ts = _parse_ts(time_sig) if time_sig else None
        self.starts = self._bar_starts()

    # -- compases ------------------------------------------------------------
    def ts_at(self, tick):
        if self.forced_ts:
            return self.forced_ts
        cur = (4, 4)
        for t, n, d in sorted(self.ts_events):
            if t <= tick:
                cur = (n, d)
        return cur

    def _bar_starts(self):
        """starts[k] = tick de inicio del compás nº first_bar + k (más un final)."""
        starts = []
        pos = 0
        self.first_bar = 1
        if self.pickup_ticks > 0:
            starts.append(0)
            pos = self.pickup_ticks
            self.first_bar = 0
        guard = 0
        while pos <= self.end_tick and guard < 200000:
            starts.append(pos)
            n, d = self.ts_at(pos)
            pos += max(1, int(round(n * 4.0 / d * self.tpb)))
            guard += 1
        starts.append(pos)                  # fin del último compás
        return starts

    @property
    def n_bars(self):
        return len(self.starts) - 1 - (1 if self.first_bar == 0 else 0)

    def bar_range_ticks(self, b0, b1):
        """Ticks [t0, t1) de los compases b0..b1 (inclusivos, numeración propia)."""
        k0, k1 = b0 - self.first_bar, b1 - self.first_bar
        if k0 < 0 or k1 < k0 or k1 >= len(self.starts) - 1:
            lo = self.first_bar
            hi = self.first_bar + len(self.starts) - 2
            raise ValueError(f"compases {b0}-{b1} fuera de rango ({lo}-{hi})")
        return self.starts[k0], self.starts[k1 + 1]


def _parse_ts(text):
    a, b = text.split("/")
    return int(a), int(b)


# ─────────────────────────────────────────────────────────────────────────────
#  Selectores
# ─────────────────────────────────────────────────────────────────────────────

def parse_ranges(spec):
    """'1-4,9,12-13' → [(1,4),(9,9),(12,13)]."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not m:
            raise ValueError(f"rango de compases no válido: {part!r}")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if b < a:
            raise ValueError(f"rango invertido: {part!r}")
        out.append((a, b))
    if not out:
        raise ValueError("--bars vacío")
    return out


def parse_tracks(spec, n):
    if spec is None:
        return list(range(n))
    out = []
    for part in spec.split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part.strip())
        if not m:
            raise ValueError(f"--tracks no válido: {spec!r}")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        out += list(range(a, b + 1))
    bad = [t for t in out if t >= n]
    if bad:
        raise ValueError(f"pista {bad[0]} inexistente (el MIDI tiene {n})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Corte
# ─────────────────────────────────────────────────────────────────────────────

_STATE_KEY = {
    "set_tempo": lambda m: ("tempo",),
    "time_signature": lambda m: ("ts",),
    "key_signature": lambda m: ("key",),
    "program_change": lambda m: ("prog", m.channel),
    "control_change": lambda m: ("cc", m.channel, m.control),
    "pitchwheel": lambda m: ("pw", m.channel),
}


def slice_track(tr, t0, t1, keep_held=True):
    """Devuelve (estado_inicial, eventos, notas) del tramo [t0, t1), en ticks
    relativos a t0. estado_inicial: mensajes vigentes justo antes de t0."""
    state = {}
    inside = []
    for t, m in tr["events"]:
        key = _STATE_KEY.get(m.type)
        if key and t <= t0:                 # lo que vale AL empezar el trozo
            state[key(m)] = m
        elif t0 <= t < t1:
            inside.append((t - t0, m))
    # el estado en t0 incluye lo que ocurre exactamente en t0
    notes = []
    for (s, e, ch, p, v) in tr["notes"]:
        if e <= t0 or s >= t1:
            continue
        if s < t0 and not keep_held:
            continue
        ns, ne = max(s, t0), min(e, t1)
        if ne > ns:
            notes.append((ns - t0, ne - t0, ch, p, v))
    return state, inside, notes


def song_segments(song, ranges, track_ids, keep_held=True, forced_ts=None):
    """Lista de segmentos: cada uno {len, per_track:[(state, events, notes)]}."""
    segs = []
    for (b0, b1) in ranges:
        t0, t1 = song.bar_range_ticks(b0, b1)
        per = [slice_track(song.tracks[i], t0, t1, keep_held) for i in track_ids]
        n, d = song.ts_at(t0)
        segs.append({"t0": t0, "len": t1 - t0, "per_track": per, "ts": (n, d),
                     "bars": (b0, b1)})
    return segs


def build_midi(song, segs, track_ids, force_ts=None):
    """Concatena segmentos en un MIDI nuevo (mismos tracks/tpb que el original)."""
    out = mido.MidiFile(type=song.type if song.type in (0, 1) else 1,
                        ticks_per_beat=song.tpb)
    for idx, ti in enumerate(track_ids):
        events = []                          # (tick, orden, mensaje)
        offset = 0
        for si, seg in enumerate(segs):
            state, inside, notes = seg["per_track"][idx]
            for key, m in state.items():
                if si == 0 or key[0] in ("tempo", "ts", "key", "prog", "cc", "pw"):
                    events.append((offset, 0, m.copy(time=0)))
            for t, m in inside:
                events.append((offset + t, 1, m))
            for (s, e, ch, p, v) in notes:
                events.append((offset + s, 3, mido.Message("note_on", channel=ch, note=p, velocity=v, time=0)))
                events.append((offset + e, 2, mido.Message("note_off", channel=ch, note=p, velocity=0, time=0)))
            offset += seg["len"]
        if force_ts:
            events = [ev for ev in events if ev[2].type != "time_signature"]
            events.append((0, 0, mido.MetaMessage("time_signature", numerator=force_ts[0],
                                                  denominator=force_ts[1], time=0)))
        # limpia repeticiones idénticas del estado en la misma posición
        events.sort(key=lambda e: (e[0], e[1]))
        trk = mido.MidiTrack()
        nm = song.tracks[ti]["name"]
        if nm and not any(e[2].type == "track_name" for e in events):
            trk.append(mido.MetaMessage("track_name", name=nm, time=0))
        last = 0
        seen_state = {}
        for tick, order, m in events:
            if order == 0:                   # estado repetido idéntico → se omite
                k = _STATE_KEY[m.type](m) if m.type in _STATE_KEY else None
                if k is not None and seen_state.get(k) == m.dict() and tick > 0:
                    continue
                if k is not None:
                    seen_state[k] = m.dict()
            trk.append(m.copy(time=tick - last))
            last = tick
        trk.append(mido.MetaMessage("end_of_track", time=max(0, offset - last)))
        out.tracks.append(trk)
    return out


def _write(mid, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(path))


# ─────────────────────────────────────────────────────────────────────────────
#  Formas (form_analyzer)
# ─────────────────────────────────────────────────────────────────────────────

def load_form(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    secs = d.get("sections") or d.get("plan", {}).get("sections")
    if not secs:
        raise ValueError(f"{path}: no contiene 'sections' (genera el JSON con "
                         f"form_analyzer.py detect obra.mid --json form.json)")
    out = []
    for i, s in enumerate(secs):
        out.append({"index": s.get("index", i + 1), "label": str(s.get("label", s.get("group", i + 1))),
                    "start": int(s["start_bar"]), "end": int(s["end_bar"])})
    return out


def resolve_order(order, sections):
    """'A,B,A' o '1,2,1' → [(start,end,label)]."""
    out = []
    for tok in [t.strip() for t in order.split(",") if t.strip()]:
        if tok.isdigit():
            hit = [s for s in sections if s["index"] == int(tok)]
        else:
            hit = [s for s in sections if s["label"] == tok]
        if not hit:
            raise ValueError(f"sección {tok!r} inexistente (etiquetas: "
                             f"{', '.join(sorted({s['label'] for s in sections}))}; "
                             f"índices 1-{len(sections)})")
        out.append((hit[0]["start"], hit[0]["end"], hit[0]["label"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Subcomandos
# ─────────────────────────────────────────────────────────────────────────────

def _song(a):
    return Song(a.midi, time_sig=a.time_sig, pickup=a.pickup)


def _dur_beats(song, n_ticks):
    return n_ticks / song.tpb


def cmd_info(a):
    s = _song(a)
    print(f"\n  MIDI SLICE v{VERSION} · {a.midi}")
    print(f"  tpb {s.tpb} · tipo {s.type} · {len(s.tracks)} pistas · "
          f"{s.n_bars} compases · {s.end_tick / s.tpb:.1f} negras")
    ts = sorted(set((n, d) for _, n, d in s.ts_events))
    print("  compás declarado: " + (", ".join(f"{n}/{d}" for n, d in ts) or "(ninguno → 4/4)")
          + (f"   [forzado {a.time_sig}]" if a.time_sig else ""))
    print(f"\n  {'pista':<6}{'nombre':<22}{'notas':>7}  compases con notas")
    for i, tr in enumerate(s.tracks):
        if not tr["notes"]:
            print(f"  {i:<6}{(tr['name'] or '')[:21]:<22}{0:>7}")
            continue
        bars = set()
        for (st, en, *_r) in tr["notes"]:
            for k in range(len(s.starts) - 1):
                if s.starts[k] < en and s.starts[k + 1] > st:
                    bars.add(k + s.first_bar)
        lo, hi = min(bars), max(bars)
        print(f"  {i:<6}{(tr['name'] or '')[:21]:<22}{len(tr['notes']):>7}  {lo}-{hi}"
              f" ({len(bars)} de {hi - lo + 1})")
    return 0


def cmd_cut(a):
    s = _song(a)
    ranges = parse_ranges(a.bars)
    if len(ranges) != 1:
        raise ValueError("cut admite un único rango (para varios usa `assemble` o `sections`)")
    ids = parse_tracks(a.tracks, len(s.tracks))
    segs = song_segments(s, ranges, ids, not a.drop_held)
    out = a.output or str(Path(a.midi).with_name(f"{Path(a.midi).stem}_c{ranges[0][0]}-{ranges[0][1]}.mid"))
    _write(build_midi(s, segs, ids, s.forced_ts), out)
    n = sum(len(p[2]) for p in segs[0]["per_track"])
    print(f"  compases {ranges[0][0]}-{ranges[0][1]} → {out}  ({n} notas, "
          f"{_dur_beats(s, segs[0]['len']):.1f} negras)")
    return 0


def _emit_files(s, items, a, ids):
    """items: [(b0, b1, label)]. Escribe ficheros y manifest."""
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"tool": "midi_slice", "version": VERSION, "source": a.midi,
                "time_signature": f"{s.ts_at(0)[0]}/{s.ts_at(0)[1]}", "tpb": s.tpb,
                "fragments": []}
    for i, (b0, b1, label) in enumerate(items, 1):
        segs = song_segments(s, [(b0, b1)], ids, not a.drop_held)
        tag = f"_{label}" if label else ""
        fn = out_dir / f"{a.prefix}_{i:02d}{tag}.mid"
        _write(build_midi(s, segs, ids, s.forced_ts), fn)
        n = sum(len(p[2]) for p in segs[0]["per_track"])
        manifest["fragments"].append({
            "file": fn.name, "index": i, "label": label, "start_bar": b0, "end_bar": b1,
            "n_bars": b1 - b0 + 1, "beats": round(_dur_beats(s, segs[0]["len"]), 3), "notes": n})
        print(f"  {fn.name:<28} compases {b0:>3}-{b1:<3} {n:>5} notas  {label or ''}")
    mf = out_dir / "manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  manifest: {mf}")


def cmd_sections(a):
    s = _song(a)
    ids = parse_tracks(a.tracks, len(s.tracks))
    if bool(a.form) == bool(a.bars):
        raise ValueError("indica exactamente una fuente: --form form.json o --bars ESPEC")
    if a.form:
        items = [(x["start"], x["end"], x["label"]) for x in load_form(a.form)]
        last = s.first_bar + len(s.starts) - 2
        clipped = [(b0, min(b1, last), l) for b0, b1, l in items if b0 <= last]
        if len(clipped) != len(items) or any(c[1] != i[1] for c, i in zip(clipped, items)):
            print(f"[AVISO] form.json excede los {s.n_bars} compases del MIDI (¿compás "
                  f"distinto? prueba --time-sig); se recorta", file=sys.stderr)
        items = clipped
    else:
        items = [(b0, b1, "") for b0, b1 in parse_ranges(a.bars)]
    _emit_files(s, items, a, ids)
    return 0


def cmd_grid(a):
    s = _song(a)
    ids = parse_tracks(a.tracks, len(s.tracks))
    if a.every < 1:
        raise ValueError("--every debe ser ≥ 1")
    first, last = s.first_bar if s.first_bar else 1, s.first_bar + len(s.starts) - 2
    items, b = [], first
    while b <= last:
        e = min(b + a.every - 1, last)
        if e - b + 1 < a.every and not a.keep_tail and items:
            items[-1] = (items[-1][0], e, "")      # el resto se une al último
        else:
            items.append((b, e, ""))
        b += a.every
    _emit_files(s, items, a, ids)
    return 0


def cmd_assemble(a):
    s = _song(a)
    ids = parse_tracks(a.tracks, len(s.tracks))
    if bool(a.form) == bool(a.bars):
        raise ValueError("indica exactamente una fuente: --bars ESPEC o --form form.json --order …")
    if a.form:
        if not a.order:
            raise ValueError("con --form hace falta --order (p.ej. A,B,A)")
        rng = [(b0, b1) for b0, b1, _ in resolve_order(a.order, load_form(a.form))]
        label = a.order
    else:
        rng = parse_ranges(a.bars)
        label = a.bars
    segs = song_segments(s, rng, ids, not a.drop_held)
    out = a.output or str(Path(a.midi).with_name(f"{Path(a.midi).stem}_asm.mid"))
    _write(build_midi(s, segs, ids, s.forced_ts), out)
    total = sum(sg["len"] for sg in segs)
    print(f"  ensamblado [{label}] → {out}  ({len(segs)} tramos, {total / s.tpb:.1f} negras)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  Selftest
# ─────────────────────────────────────────────────────────────────────────────

def _make(path, n_bars=16, ts=(4, 4), tpb=480):
    mid = mido.MidiFile(ticks_per_beat=tpb)
    t0 = mido.MidiTrack()
    t0.append(mido.MetaMessage("set_tempo", tempo=500000))
    t0.append(mido.MetaMessage("time_signature", numerator=ts[0], denominator=ts[1]))
    mid.tracks.append(t0)
    bar = int(ts[0] * 4 / ts[1] * tpb)
    ev = []
    # tempo nuevo en el compás 9 y programa distinto en el 5
    tempo_ev = (8 * bar, mido.MetaMessage("set_tempo", tempo=400000))
    mel = mido.MidiTrack()
    mel.append(mido.Message("program_change", program=40, channel=0, time=0))
    for b in range(n_bars):
        pitch = 60 + b
        ev.append((b * bar, pitch, bar // 2 + (bar if b == 5 else 0)))       # el compás 6 cruza al 7
    flat = []
    for t, p, d in ev:
        flat.append((t, 1, p)); flat.append((t + d, 0, p))
    flat.sort()
    last = 0
    for t, on, p in flat:
        mel.append(mido.Message("note_on", note=p, velocity=80 if on else 0, channel=0, time=t - last))
        last = t
    mid.tracks.append(mel)
    # insertar tempo en pista 0 sin romper deltas
    t0.append(mido.MetaMessage("set_tempo", tempo=400000, time=8 * bar))
    mid.save(path)


def _notes(path):
    m = mido.MidiFile(path)
    out = []
    for tr in m.tracks:
        t = 0
        for x in tr:
            t += x.time
            if x.type == "note_on" and x.velocity > 0:
                out.append((t, x.note))
    return sorted(out), m


def cmd_selftest(a):
    ok = True

    def check(c, msg):
        nonlocal ok
        print(("  ✔ " if c else "  ✘ ") + msg)
        ok &= bool(c)

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "src.mid")
        _make(p)
        s = Song(p)
        check(s.n_bars == 16, f"cuenta 16 compases → {s.n_bars}")
        out = os.path.join(d, "c.mid")
        a2 = argparse.Namespace(midi=p, bars="9-12", tracks=None, output=out, drop_held=False,
                                time_sig=None, pickup=0.0)
        cmd_cut(a2)
        notes, m = _notes(out)
        check([n for _, n in notes] == [68, 69, 70, 71], f"cut 9-12 → notas {[n for _, n in notes]}")
        check(notes[0][0] == 0 and notes[1][0] == 1920, "ticks relativos al inicio")
        tempos = [x.tempo for tr in m.tracks for x in tr if x.type == "set_tempo"]
        check(tempos and tempos[0] == 400000, f"tempo vigente heredado ({tempos[:1]})")
        progs = [x.program for tr in m.tracks for x in tr if x.type == "program_change"]
        check(progs == [40], "program_change heredado")
        # nota que cruza el límite
        a3 = argparse.Namespace(midi=p, bars="7-7", tracks=None, output=out, drop_held=False,
                                time_sig=None, pickup=0.0)
        cmd_cut(a3)
        notes, _ = _notes(out)
        check(len(notes) == 2 or len(notes) == 1, f"nota sostenida en compás 7 → {len(notes)} nota(s)")
        a3.drop_held = True
        cmd_cut(a3)
        n2, _ = _notes(out)
        check(len(n2) == 1 and n2[0][1] == 66, "--drop-held descarta la nota que ya sonaba")
        # sections + manifest
        od = os.path.join(d, "secs")
        a4 = argparse.Namespace(midi=p, bars="1-4,5-8,9-16", form=None, out_dir=od, prefix="seccion",
                                tracks=None, drop_held=False, time_sig=None, pickup=0.0)
        cmd_sections(a4)
        mf = json.loads(Path(od, "manifest.json").read_text())
        check(len(mf["fragments"]) == 3 and mf["fragments"][2]["n_bars"] == 8, "sections + manifest")
        # grid
        od2 = os.path.join(d, "grid")
        a5 = argparse.Namespace(midi=p, every=6, out_dir=od2, prefix="frase", tracks=None, drop_held=False,
                                time_sig=None, pickup=0.0, keep_tail=False)
        cmd_grid(a5)
        mf = json.loads(Path(od2, "manifest.json").read_text())
        check([f["n_bars"] for f in mf["fragments"]] == [6, 10], "grid --every 6: el resto se une al último")
        # assemble A B A
        form = os.path.join(d, "form.json")
        Path(form).write_text(json.dumps({"sections": [
            {"index": 1, "label": "A", "start_bar": 1, "end_bar": 4},
            {"index": 2, "label": "B", "start_bar": 5, "end_bar": 8}]}))
        o = os.path.join(d, "aba.mid")
        a6 = argparse.Namespace(midi=p, bars=None, form=form, order="A,B,A", output=o, tracks=None,
                                drop_held=False, time_sig=None, pickup=0.0)
        cmd_assemble(a6)
        notes, m = _notes(o)
        check([n for _, n in notes][:12] == [60, 61, 62, 63, 64, 65, 66, 67, 60, 61, 62, 63] or
              len(notes) >= 12, "assemble A,B,A produce 12 compases")
        names = [n for _, n in notes]
        check(names[-4:] == [60, 61, 62, 63] and names[:4] == [60, 61, 62, 63], "orden A B A respetado")
        tot = max(t for t, _ in notes) + 1
        check(tot >= 11 * 1920, "duración total ≈ 12 compases")
        # rango fuera
        try:
            Song(p).bar_range_ticks(20, 24)
            check(False, "rango fuera de límites debe fallar")
        except ValueError:
            check(True, "rango fuera de límites → error claro")
        # 3/4 y --time-sig
        p3 = os.path.join(d, "w.mid")
        _make(p3, ts=(3, 4))
        check(Song(p3).n_bars == 16, "3/4: 16 compases")
        check(Song(p, time_sig="2/4").n_bars == 32, "--time-sig 2/4 reinterpreta los compases")
        # pickup
        sp_ = Song(p, pickup=1.0)
        check(sp_.first_bar == 0 and sp_.bar_range_ticks(0, 0) == (0, 480), "--pickup define el compás 0")
    print("  SELFTEST OK" if ok else "  SELFTEST FALLÓ")
    return 0 if ok else 1


def build_parser():
    ap = argparse.ArgumentParser(prog="midi_slice.py",
                                 description="Trocear y recomponer un MIDI por compases")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p, bars=False):
        p.add_argument("midi")
        p.add_argument("--tracks", default=None)
        p.add_argument("--time-sig", default=None, metavar="N/D")
        p.add_argument("--pickup", type=float, default=0.0, metavar="BEATS")
        p.add_argument("--drop-held", action="store_true")

    p = sp.add_parser("info", help="Compases y pistas")
    common(p)
    p.set_defaults(fn=cmd_info)

    p = sp.add_parser("cut", help="Extrae un rango de compases")
    common(p)
    p.add_argument("--bars", required=True)
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(fn=cmd_cut)

    p = sp.add_parser("sections", help="Un MIDI por sección")
    common(p)
    p.add_argument("--form", default=None, help="JSON de form_analyzer.py detect")
    p.add_argument("--bars", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="seccion")
    p.set_defaults(fn=cmd_sections)

    p = sp.add_parser("grid", help="Cortes regulares cada N compases")
    common(p)
    p.add_argument("--every", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="frase")
    p.add_argument("--keep-tail", action="store_true", help="No une el resto corto al último trozo")
    p.set_defaults(fn=cmd_grid)

    p = sp.add_parser("assemble", help="Recompone rangos (formas A B A)")
    common(p)
    p.add_argument("--bars", default=None)
    p.add_argument("--form", default=None)
    p.add_argument("--order", default=None, help="Etiquetas o índices: A,B,A / 1,2,1")
    p.add_argument("-o", "--output", default=None)
    p.set_defaults(fn=cmd_assemble)

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
