#!/usr/bin/env python3
"""
chord_parse.py — Chord Parse: gramática única de cifrado de acordes (cifrado → notas)
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Cada herramienta de armonía de la suite interpreta el cifrado a su manera:
generar_midi.py con una tabla cerrada (CHORD_VOICINGS), voice_leader.py,
chord_pattern_voicer.py, leadsheet_realizer.py y bass_line_composer.py con sus propios
parsers. El resultado: un acorde válido en una herramienta es «no reconocido» en otra, y
algunas caen en silencio al acorde por defecto (Ebmaj7, Dmaj7, Gdim7 o C#dim7 no estaban
en la tabla de generar_midi). Esta herramienta es la referencia: raíz + calidad
[+ /bajo] → intervalos, notas y voicing, y un VALIDADOR que dice qué acordes de un texto o
de una partitura YAML no se entienden y en qué línea, ANTES de renderizar.

Se integra con el resto como todas las herramientas del ecosistema, por subprocess:

  chord_parse.py parse  "Ebmaj7 C#dim7 Bb/D" --json     → estructura de cada acorde
  chord_parse.py voice  "Dm7 G7 Cmaj7" --midi acordes.mid → notas MIDI / fichero
  chord_parse.py validate --yaml partitura.yaml           → rc=3 si hay acordes no válidos

  parse     Raíz, calidad canónica, intervalos, notas, bajo (slash) y nombre normalizado
  voice     Notas MIDI de cada acorde. Layout «table» (por defecto): bajo F1..E2, raíz
            duplicada una octava arriba y el resto sobre ella; es el MISMO patrón de
            registro de generar_midi.py. Layout «close»: posición fundamental compacta
            alrededor de C4. Con --midi escribe un acorde por compás
  validate  Comprueba acordes sueltos, un fichero de texto (--file) o los campos
            «acorde:» de una partitura YAML (--yaml), con número de línea y sugerencias
  table     Lista las calidades soportadas con sus alias e intervalos
  selftest  Comprueba el parser con casos conocidos

Notas
  · Alias de alteraciones: # ♯ (sostenido), b ♭ (bemol). Calidades con alias: maj7/M7/Δ,
    m7/-7, dim7/o7/°7, m7b5/ø, aug/+, sus/sus4… (ver «table»).
  · «Bb» se interpreta como Si♭; «Bbm», «Bb7»… igual. Si la calidad no existe tras una raíz
    con bemol, se reintenta como raíz natural + calidad que empieza por «b» (p. ej. «Bb5»).
  · N.C. / NC = sin acorde (válido, sin notas).
  · La enarmonía de los nombres de nota es orientativa: bemoles si la raíz lleva bemol,
    sostenidos en el resto.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  parse     ACORDES... [--json]
  voice     ACORDES... [--layout table|close] [--json]
                       [--midi SALIDA.mid] [--bpm 60] [--beats 4] [--channel 0]
  validate  [ACORDES...] [--yaml F] [--file F] [--json]
  table     [--json]
  selftest

ACORDES: uno o varios argumentos; dentro de cada uno se separa por espacios o comas.
Códigos de salida: 0 correcto · 2 argumento no válido / no se puede generar ·
3 (parse, validate) si hay algún acorde no válido.

EJEMPLOS
  python chord_parse.py parse "Ebmaj7 C#dim7 Bb/D Cm7b5"
  python chord_parse.py voice Dm7 G7 Cmaj7 --layout close
  python chord_parse.py voice "Dm Gm A7 Dm" --midi acordes.mid --bpm 72
  python chord_parse.py validate --yaml partitura_obra.yaml
  python chord_parse.py table

@params v1
[parse]
chords      list  of=str required  # Acordes (separados por espacios o comas)
--json      bool  default=false  # Salida en JSON
[voice]
chords      list  of=str required  # Acordes (separados por espacios o comas)
--layout    enum  default=table values=table|close  # table = patrón de generar_midi
--midi      file  io=out ext=mid  # Escribe un acorde por compás
--bpm       float default=60 min=20 max=300 unit=BPM
--beats     int   default=4 min=1 max=16 unit=tiempos  # Duración de cada acorde
--channel   int   default=0 min=0 max=15
--json      bool  default=false  # Salida en JSON
[validate]
chords      list  of=str  # Acordes sueltos (opcional)
--yaml      file  io=in ext=yaml,yml  # Partitura YAML: se leen los campos «acorde:»
--file      file  io=in  # Texto con acordes (uno o varios por línea)
--json      bool  default=false  # Salida en JSON
[table]
--json      bool  default=false  # Salida en JSON
@end
"""
import argparse
import difflib
import json
import os
import re
import sys

VERSION = "1.0"

# (canónica, intervalos, alias, descripción)
QUALITIES = [
    ("",        [0, 4, 7],              ["maj", "M"],                      "mayor"),
    ("m",       [0, 3, 7],              ["min", "-"],                      "menor"),
    ("5",       [0, 7],                 [],                                "quinta"),
    ("7",       [0, 4, 7, 10],          [],                                "dominante"),
    ("maj7",    [0, 4, 7, 11],          ["M7", "Δ", "Δ7", "Maj7"],         "séptima mayor"),
    ("m7",      [0, 3, 7, 10],          ["min7", "-7"],                    "menor con séptima"),
    ("mmaj7",   [0, 3, 7, 11],          ["mM7", "minmaj7", "m(maj7)"],     "menor con séptima mayor"),
    ("dim",     [0, 3, 6],              ["o", "°"],                        "disminuido"),
    ("dim7",    [0, 3, 6, 9],           ["o7", "°7"],                      "séptima disminuida"),
    ("m7b5",    [0, 3, 6, 10],          ["m7-5", "ø", "ø7"],               "semidisminuido"),
    ("aug",     [0, 4, 8],              ["+"],                             "aumentado"),
    ("aug7",    [0, 4, 8, 10],          ["+7", "7#5"],                     "aumentado con séptima"),
    ("sus4",    [0, 5, 7],              ["sus"],                           "suspendido 4ª"),
    ("sus2",    [0, 2, 7],              [],                                "suspendido 2ª"),
    ("7sus4",   [0, 5, 7, 10],          ["7sus"],                          "dominante suspendido"),
    ("6",       [0, 4, 7, 9],           [],                                "sexta"),
    ("m6",      [0, 3, 7, 9],           [],                                "menor con sexta"),
    ("6/9",     [0, 4, 7, 9, 14],       ["69"],                            "sexta con novena"),
    ("9",       [0, 4, 7, 10, 14],      [],                                "novena de dominante"),
    ("maj9",    [0, 4, 7, 11, 14],      ["M9"],                            "séptima mayor con novena"),
    ("m9",      [0, 3, 7, 10, 14],      [],                                "menor con novena"),
    ("add9",    [0, 4, 7, 14],          [],                                "mayor con novena añadida"),
    ("madd9",   [0, 3, 7, 14],          [],                                "menor con novena añadida"),
    ("11",      [0, 4, 7, 10, 14, 17],  [],                                "undécima de dominante"),
    ("m11",     [0, 3, 7, 10, 14, 17],  [],                                "menor con undécima"),
    ("13",      [0, 4, 7, 10, 14, 21],  [],                                "trecena de dominante"),
    ("7b5",     [0, 4, 6, 10],          [],                                "dominante con 5ª disminuida"),
    ("7b9",     [0, 4, 7, 10, 13],      [],                                "dominante con 9ª menor"),
    ("7#9",     [0, 4, 7, 10, 15],      [],                                "dominante con 9ª aumentada"),
    ("maj7#11", [0, 4, 7, 11, 18],      [],                                "séptima mayor con 11ª aumentada"),
]
_BY_SUFFIX = {}
for _canon, _iv, _aliases, _desc in QUALITIES:
    for _s in [_canon] + _aliases:
        _BY_SUFFIX[_s] = (_canon, _iv)

_ROOT_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARPS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLATS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_CHORD_RE = re.compile(r"^([A-G](?:#|♯|b|♭)?)(.*?)(?:/([A-G](?:#|♯|b|♭)?))?$")
_RESTS = {"N.C.", "NC", "N.C"}


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre."""
    if isinstance(path, (str, bytes, os.PathLike)):
        d = os.path.dirname(os.path.abspath(os.fsdecode(path)))
        if d:
            os.makedirs(d, exist_ok=True)
    return path


def _pc(txt):
    pc = _ROOT_PC[txt[0]]
    for ch in txt[1:]:
        pc += 1 if ch in ("#", "♯") else -1
    return pc % 12


def _flat(txt):
    return any(ch in ("b", "♭") for ch in txt[1:])


def _names(pcs, flats):
    tab = FLATS if flats else SHARPS
    return [tab[p % 12] for p in pcs]


def split_chords(items):
    out = []
    for it in items:
        out += [c for c in re.split(r"[\s,]+", str(it).strip()) if c]
    return out


def parse_chord(sym):
    """Cifrado → dict o None si no se entiende."""
    sym = str(sym).strip()
    if sym.upper() in _RESTS:
        return {"input": sym, "rest": True, "canonical": "N.C.", "notes": [], "intervals": []}
    m = _CHORD_RE.match(sym)
    if not m:
        return None
    root_txt, qual, bass_txt = m.groups()
    cands = [(root_txt, qual)]
    if len(root_txt) == 2 and root_txt[1] in ("b", "♭"):
        cands.append((root_txt[0], root_txt[1] + qual))
    for rt, q in cands:
        if q not in _BY_SUFFIX:
            continue
        canon, iv = _BY_SUFFIX[q]
        root_pc = _pc(rt)
        # Ortografía: bemoles si la raíz los lleva, si es una raíz «bemolada» (Do, Mi♭, Fa,
        # La♭, Si♭) o si es Re/Sol con tercera menor; sostenidos en el resto.
        sharp_root = rt.endswith(("#", "♯"))
        flats = (not sharp_root) and (_flat(rt) or root_pc in (0, 3, 5, 8, 10)
                                      or (root_pc in (2, 7) and 3 in iv)
                                      or (bass_txt is not None and _flat(bass_txt)))
        bass_pc = _pc(bass_txt) if bass_txt else None
        pcs = []
        for i in iv:
            p = (root_pc + i) % 12
            if p not in pcs:
                pcs.append(p)
        rname = _names([root_pc], flats)[0]
        bname = _names([bass_pc], flats)[0] if bass_pc is not None else None
        return {"input": sym, "rest": False, "root": rname, "root_pc": root_pc,
                "quality": canon if canon else "major", "suffix": canon,
                "intervals": list(iv), "notes": _names(pcs, flats),
                "bass": bname, "bass_pc": bass_pc,
                "canonical": rname + canon + (f"/{bname}" if bname else "")}
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  VOICING
# ══════════════════════════════════════════════════════════════════════════════

def _bass_midi(pc):
    """Nota grave en el rango F1..E2 (29..40), como CHORD_VOICINGS de generar_midi."""
    m = 24 + pc
    return m + 12 if m < 29 else m


def voice_chord(ch, layout="table"):
    """dict de parse_chord → lista de notas MIDI."""
    if ch.get("rest"):
        return []
    pc, iv = ch["root_pc"], ch["intervals"]
    bass_pc = ch["bass_pc"] if ch["bass_pc"] is not None else pc
    if layout == "close":
        root = 60 + pc
        if root > 66:
            root -= 12
        notes = [root + i for i in iv]
        if ch["bass_pc"] is not None:
            b = root - 1
            while b % 12 != bass_pc:
                b -= 1
            notes = [b] + notes
        return notes
    root3 = _bass_midi(pc) + 12
    return [_bass_midi(bass_pc), root3] + [root3 + i for i in iv if i != 0]


def midi_name(p):
    return f"{SHARPS[p % 12]}{p // 12 - 1}"


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def suggest(sym):
    """Sugerencias para un cifrado no válido."""
    m = _CHORD_RE.match(sym)
    if not m:
        if sym and sym[0].lower() in "abcdefg" and sym[0] not in _ROOT_PC:
            return [sym[0].upper() + sym[1:]]
        return []
    root, qual, bass = m.groups()
    sufs = [s for s in _BY_SUFFIX if s]
    close = difflib.get_close_matches(qual, sufs, n=3, cutoff=0.5)
    return [root + c + (f"/{bass}" if bass else "") for c in close]


_ACORDE_RE = re.compile(r"""acorde\s*:\s*(?:"([^"]+)"|'([^']+)'|([^,}\s#]+))""")


def _scan_text(path, yaml_mode):
    found = []
    with open(path, encoding="utf8") as fh:
        for n, line in enumerate(fh, 1):
            if yaml_mode:
                for mt in _ACORDE_RE.finditer(line):
                    found.append((n, mt.group(1) or mt.group(2) or mt.group(3)))
            else:
                body = line.split("#", 1)[0]
                for c in split_chords([body]):
                    found.append((n, c))
    return found


def _emit_midi(chords, voiced, path, bpm, beats, channel):
    import mido
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("set_tempo", tempo=int(60_000_000 / bpm)))
    tr.append(mido.MetaMessage("track_name", name="chords"))
    first = True
    dur = tpb * beats
    for notes in voiced:
        if not notes:
            tr.append(mido.Message("note_on", channel=channel, note=0, velocity=0,
                                   time=dur if not first else 0))
            first = False
            continue
        for i, p in enumerate(notes):
            tr.append(mido.Message("note_on", channel=channel, note=p, velocity=80, time=0))
        for i, p in enumerate(notes):
            tr.append(mido.Message("note_off", channel=channel, note=p, velocity=0,
                                   time=dur if i == 0 else 0))
        first = False
    mid.tracks.append(tr)
    mid.save(_ensure_parent(path))


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_parse(a):
    chords = split_chords(a.chords)
    res, bad = [], 0
    for c in chords:
        p = parse_chord(c)
        if p is None:
            bad += 1
            res.append({"input": c, "valid": False, "suggestions": suggest(c)})
        else:
            res.append({"valid": True, **p})
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 3 if bad else 0
    print(f"\n  CHORD PARSE v{VERSION} · parse")
    for r in res:
        if not r["valid"]:
            sg = f"   ¿{', '.join(r['suggestions'])}?" if r["suggestions"] else ""
            print(f"  ✘ {r['input']:12} no reconocido{sg}")
        elif r["rest"]:
            print(f"  ✔ {r['input']:12} sin acorde")
        else:
            bass = f" /{r['bass']}" if r["bass"] else ""
            print(f"  ✔ {r['input']:12} {r['canonical']:12} {r['quality']:9} notas: "
                  f"{' '.join(r['notes'])}{bass}")
    return 3 if bad else 0


def cmd_voice(a):
    chords = split_chords(a.chords)
    parsed = []
    for c in chords:
        p = parse_chord(c)
        if p is None:
            sg = suggest(c)
            raise ValueError(f"acorde no reconocido: {c!r}" + (f" (¿{', '.join(sg)}?)" if sg else ""))
        parsed.append(p)
    voiced = [voice_chord(p, a.layout) for p in parsed]
    if a.midi:
        _emit_midi(chords, voiced, a.midi, a.bpm, a.beats, a.channel)
    if a.json:
        print(json.dumps([{"chord": p["input"], "canonical": p["canonical"], "midi": v,
                           "names": [midi_name(x) for x in v]} for p, v in zip(parsed, voiced)],
                         ensure_ascii=False, indent=2))
        return 0
    print(f"\n  CHORD PARSE v{VERSION} · voice ({a.layout})")
    for p, v in zip(parsed, voiced):
        print(f"  {p['input']:12} {' '.join(midi_name(x) for x in v) or '—':36} {v}")
    if a.midi:
        print(f"\n  Guardado en: {a.midi}  ({len(voiced)} acordes · {a.beats} tiempos · {a.bpm:g} BPM)")
    return 0


def cmd_validate(a):
    items = [(None, c) for c in split_chords(a.chords or [])]
    if a.yaml:
        items += _scan_text(a.yaml, True)
    if a.file:
        items += _scan_text(a.file, False)
    if not items:
        raise ValueError("nada que validar: indique acordes, --yaml o --file")
    seen, bad = {}, []
    for line, c in items:
        ok = c in seen and seen[c] or (c not in seen and parse_chord(c) is not None)
        seen[c] = ok
        if not ok:
            bad.append({"line": line, "chord": c, "suggestions": suggest(c)})
    distinct = len(seen)
    if a.json:
        print(json.dumps({"checked": len(items), "distinct": distinct, "invalid": bad},
                         ensure_ascii=False, indent=2))
        return 3 if bad else 0
    print(f"\n  CHORD PARSE v{VERSION} · validate: {len(items)} acordes ({distinct} distintos)")
    for b in bad:
        where = f"línea {b['line']}: " if b["line"] else ""
        sg = f"   ¿{', '.join(b['suggestions'])}?" if b["suggestions"] else ""
        print(f"  ✘ {where}{b['chord']} no reconocido{sg}")
    print(f"\n  {'✔ todos los acordes son válidos' if not bad else f'✘ {len(bad)} no válidos'}")
    return 3 if bad else 0


def cmd_table(a):
    rows = [{"suffix": c, "aliases": al, "intervals": iv, "description": d}
            for c, iv, al, d in QUALITIES]
    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    print(f"\n  CHORD PARSE v{VERSION} · {len(rows)} calidades (ejemplo sobre C)")
    for r in rows:
        ex = "C" + r["suffix"]
        al = ", ".join(r["aliases"]) or "—"
        print(f"  {ex:10} alias: {al:24} intervalos: {r['intervals']}   {r['description']}")
    return 0


def cmd_selftest(a):
    res = []

    def chk(name, cond):
        res.append((name, bool(cond)))

    p = parse_chord("Ebmaj7")
    chk("Ebmaj7 → Eb G Bb D", p and p["notes"] == ["Eb", "G", "Bb", "D"])
    p = parse_chord("C#dim7")
    chk("C#dim7 tiene 4 notas distintas y 3 semitonos entre ellas",
        p and p["intervals"] == [0, 3, 6, 9] and p["root_pc"] == 1)
    p = parse_chord("Bb/D")
    chk("Bb/D: raíz Bb, bajo D", p and p["root"] == "Bb" and p["bass"] == "D")
    chk("ortografía: Cm7b5 = C Eb Gb Bb, D7 = D F# A C",
        parse_chord("Cm7b5")["notes"] == ["C", "Eb", "Gb", "Bb"]
        and parse_chord("D7")["notes"] == ["D", "F#", "A", "C"])
    chk("Bbm se lee como Si♭ menor", parse_chord("Bbm")["root_pc"] == 10
        and parse_chord("Bbm")["suffix"] == "m")
    chk("Bb5 se lee como Si♭ + quinta", parse_chord("Bb5")["root_pc"] == 10
        and parse_chord("Bb5")["suffix"] == "5")
    chk("alias: M7=maj7, ø=m7b5, °7=dim7",
        parse_chord("CM7")["suffix"] == "maj7" and parse_chord("Cø")["suffix"] == "m7b5"
        and parse_chord("C°7")["suffix"] == "dim7")
    chk("C6/9 no se confunde con slash", parse_chord("C6/9")["suffix"] == "6/9")
    chk("N.C. es válido sin notas", parse_chord("N.C.")["rest"])
    chk("acorde inexistente → None", parse_chord("Xyz") is None and parse_chord("Cfoo") is None)
    chk("voicing table de Dm: D1 D2 F2 A2",
        voice_chord(parse_chord("Dm")) == [38, 50, 53, 57])
    chk("voicing table de F: bajo F1 (29)", voice_chord(parse_chord("F"))[0] == 29)
    chk("voicing table de E: bajo E2 (40)", voice_chord(parse_chord("E"))[0] == 40)
    chk("voicing close de C: C4 E4 G4", voice_chord(parse_chord("C"), "close") == [60, 64, 67])
    chk("slash en table: bajo = nota del bajo", voice_chord(parse_chord("C/E"))[0] == 40)
    chk("sugerencia para Cmaj77", suggest("Cmaj77") != [])
    ok = all(c for _n, c in res)
    for n, c in res:
        print(f"  {'✔' if c else '✘'} {n}")
    print("\n  SELFTEST OK" if ok else "\n  SELFTEST FALLA")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Gramática única de cifrado de acordes (cifrado → notas)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse"); p.add_argument("chords", nargs="+")
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_parse)
    p = sub.add_parser("voice"); p.add_argument("chords", nargs="+")
    p.add_argument("--layout", default="table", choices=("table", "close"))
    p.add_argument("--midi"); p.add_argument("--bpm", type=float, default=60)
    p.add_argument("--beats", type=int, default=4); p.add_argument("--channel", type=int, default=0)
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_voice)
    p = sub.add_parser("validate"); p.add_argument("chords", nargs="*")
    p.add_argument("--yaml"); p.add_argument("--file")
    p.add_argument("--json", action="store_true"); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("table"); p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_table)
    p = sub.add_parser("selftest"); p.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    try:
        sys.exit(a.fn(a))
    except FileNotFoundError as e:
        print(f"[ERROR] fichero no encontrado: {e.filename}", file=sys.stderr)
        sys.exit(2)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
