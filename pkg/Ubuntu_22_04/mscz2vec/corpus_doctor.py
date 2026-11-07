#!/usr/bin/env python3
"""
corpus_doctor.py — Corpus Doctor: escaneo EXHAUSTIVO de la salud de un corpus
de MIDIs, clasificación de fallos y cuarentena/reparación segura
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
Un solo MIDI roto aborta un pipeline largo, y los diagnósticos por muestreo
(p. ej. «health --sample 60») no los ven. Esta herramienta recorre TODO el
corpus, clasifica cada archivo con una categoría estable y dice de una vez
qué porcentaje está sano, qué falla y qué es reparable. Es autocontenida:
solo usa mido y la biblioteca estándar.

────────────────────────────────────────────────────────────────────────────
Modos de uso
────────────────────────────────────────────────────────────────────────────
  scan        Recorre ficheros/carpetas y clasifica cada MIDI
  report      Re-muestra un informe JSON guardado por «scan --json»
  quarantine  Aparta (y opcionalmente repara) los archivos con problemas.
              Por defecto SOLO MUESTRA el plan; hace falta --apply para
              tocar el disco. Nunca modifica ni borra el original salvo
              que se pida mover (sin --copy) con --apply.

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  scan
    paths PATHS […]             ficheros, globs o carpetas (recursivo)
    --json JSON                 guarda el informe completo
    --min-notes MIN_NOTES       menos notas que esto = NO_NOTES (defecto 1)
    --strict                    rc=3 si hay algún archivo con problemas
    --quiet, -q                 solo el resumen
    --no-color

  report
    json JSON
    --category CATEGORY         filtra por categoría
    --no-color

  quarantine
    paths PATHS […]
    --dest DEST                 carpeta de cuarentena (defecto quarantine/)
    --repair                    intenta reparar y deja la copia en DEST/repaired/
    --apply                     ejecuta de verdad (sin él: simulación)
    --copy                      copia en vez de mover los originales
    --no-color

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
@params v1
[scan]
paths       list  of=file io=in required
--json      file  io=out ext=json
--min-notes int   default=1 min=0
--strict    bool  default=false
--quiet     bool  default=false alias=-q
--no-color  bool  default=false
[report]
json        file  io=in ext=json required
--category  str
--no-color  bool  default=false
[quarantine]
paths       list  of=file io=in required
--dest      dir   io=out default=quarantine
--repair    bool  default=false
--apply     bool  default=false
--copy      bool  default=false
--no-color  bool  default=false
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
    python corpus_doctor.py scan midis/
    python corpus_doctor.py scan midis/ --json salud.json --strict
    python corpus_doctor.py report salud.json --category TRUNCATED
    python corpus_doctor.py quarantine midis/ --dest cuarentena --repair
    python corpus_doctor.py quarantine midis/ --dest cuarentena --repair --apply --copy

  CATEGORÍAS:
    OK            parsea en modo estricto y tiene notas
    NOT_MIDI      no es MIDI (p. ej. una página HTML de error 404)      ✗
    EMPTY_FILE    0 bytes                                               ✗
    JUNK_PREFIX   bytes previos a «MThd» o contenedor RIFF/RMID         ⚙ reparable
    TRUNCATED     pistas incompletas (EOFError / longitud excede)       ⚙ salvable
    BAD_BYTES     bytes de dato >127 (mido estricto los rechaza)        ⚙ reparable
    BAD_KEYSIG    armadura inválida (p. ej. modo 255)                   ⚙ reparable
    PARSE_ERROR   otro fallo de lectura (se indica el mensaje)
    NO_NOTES      válido pero sin notas (< --min-notes)

  CÓDIGOS DE SALIDA: 0 ok · 1 error de uso · 3 problemas con --strict

  API:  from corpus_doctor import diagnose, repair_bytes

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  mido
"""

import argparse
import glob
import io
import json
import os
import shutil
import struct
import sys
from collections import Counter, OrderedDict

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


REPAIRABLE = {"JUNK_PREFIX", "TRUNCATED", "BAD_BYTES", "BAD_KEYSIG"}
SEVERITY = {"NOT_MIDI": 0, "EMPTY_FILE": 0, "PARSE_ERROR": 0, "TRUNCATED": 1,
            "BAD_BYTES": 1, "BAD_KEYSIG": 1, "JUNK_PREFIX": 1, "NO_NOTES": 2, "OK": 9}


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def _sniff_not_midi(data):
    head = data[:512].lstrip().lower()
    if head.startswith((b"<!doctype", b"<html", b"<head", b"<?xml", b"<body")) or b"<html" in head:
        return "página HTML (¿error 404 descargado como .mid?)"
    if head.startswith(b"{") or head.startswith(b"[{"):
        return "JSON/texto"
    if data[:4] == b"RIFF":
        return None  # RMID: lo trata JUNK_PREFIX
    if data[:3] == b"ID3" or data[:4] in (b"OggS", b"fLaC") or data[:4] == b"RIFF":
        return "audio, no MIDI"
    try:
        data[:200].decode("ascii")
        return "texto plano"
    except UnicodeDecodeError:
        return "cabecera desconocida (no empieza por MThd)"


def _walk_chunks(data, start=0):
    """Devuelve (ntracks_declarados, [(offset, declared_len, available_len)])."""
    if data[start:start + 4] != b"MThd" or len(data) < start + 14:
        return 0, []
    hlen = struct.unpack(">I", data[start + 4:start + 8])[0]
    ntr = struct.unpack(">H", data[start + 10:start + 12])[0]
    pos = start + 8 + hlen
    chunks = []
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        ln = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        avail = max(0, min(ln, len(data) - (pos + 8)))
        chunks.append((pos, cid, ln, avail))
        pos += 8 + ln
    return ntr, chunks


def _parse(data, clip):
    return mido.MidiFile(file=io.BytesIO(data), clip=clip)


def _count_notes(mid):
    n = 0
    for tr in mid.tracks:
        for m in tr:
            if m.type == "note_on" and m.velocity > 0:
                n += 1
    return n


def diagnose(path, min_notes=1):
    """Clasifica un archivo. Devuelve dict con category/detail/repairable/..."""
    res = OrderedDict(path=path, size=0, category="OK", detail="", repairable=False,
                      notes=0, tracks=0, duration_s=None)
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        res.update(category="PARSE_ERROR", detail=f"no se puede leer: {e}")
        return res
    res["size"] = len(data)
    if not data:
        res.update(category="EMPTY_FILE", detail="0 bytes")
        return res

    # Cabecera
    if data[:4] != b"MThd":
        why = _sniff_not_midi(data)
        if why is not None:
            res.update(category="NOT_MIDI", detail=why)
            return res
        off = data.find(b"MThd", 0, 4096)
        if off > 0:
            res.update(category="JUNK_PREFIX", repairable=True,
                       detail=f"MThd aparece en el byte {off} (contenedor RIFF/prefijo basura)")
            return res
        res.update(category="NOT_MIDI", detail="sin cabecera MThd")
        return res

    # Estructura de chunks (detecta truncado antes de que mido lo confunda)
    ntr, chunks = _walk_chunks(data)
    tracks = [c for c in chunks if c[1] == b"MTrk"]
    short = [c for c in tracks if c[3] < c[2]]
    if short or len(tracks) < ntr:
        if short:
            d = f"pista truncada: declara {short[0][2]} bytes y hay {short[0][3]}"
        else:
            d = f"cabecera declara {ntr} pistas y solo hay {len(tracks)}"
        res.update(category="TRUNCATED", detail=d, repairable=len(tracks) > 0)
        return res

    # Lectura estricta
    try:
        mid = _parse(data, clip=False)
    except EOFError:
        res.update(category="TRUNCATED", detail="EOFError al leer", repairable=True)
        return res
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        low = msg.lower()
        cat = "PARSE_ERROR"
        if "0..127" in low or "data byte" in low:
            cat = "BAD_BYTES"
        elif "key" in low and ("signature" in low or "keyerror" in low or "255" in low or "mode" in low):
            cat = "BAD_KEYSIG"
        elif isinstance(e, KeyError):
            cat = "BAD_KEYSIG"   # mido lanza KeyError al decodificar armaduras inválidas
        rep = False
        if cat in ("BAD_BYTES", "BAD_KEYSIG", "PARSE_ERROR"):
            fixed, _ = repair_bytes(data)
            rep = fixed is not None
        res.update(category=cat, detail=msg[:140], repairable=rep)
        return res

    res["tracks"] = len(mid.tracks)
    res["notes"] = _count_notes(mid)
    try:
        res["duration_s"] = round(mid.length, 1)
    except Exception:  # noqa: BLE001
        pass
    if res["notes"] < min_notes:
        res.update(category="NO_NOTES", detail=f"{res['notes']} notas")
    return res


# ══════════════════════════════════════════════════════════════════════════════
#  REPARACIÓN (siempre devuelve bytes nuevos; no toca el original)
# ══════════════════════════════════════════════════════════════════════════════

def _patch_keysigs(data):
    """Corrige meta key_signature con sf fuera de -7..7 o modo fuera de 0/1."""
    b = bytearray(data)
    i = 0
    n = 0
    while True:
        i = b.find(b"\xff\x59\x02", i)
        if i < 0 or i + 5 > len(b):
            break
        sf = b[i + 3] - 256 if b[i + 3] > 127 else b[i + 3]
        mi = b[i + 4]
        if sf < -7 or sf > 7 or mi not in (0, 1):
            b[i + 3] = 0
            b[i + 4] = 0
            n += 1
        i += 3
    return bytes(b), n


def _salvage_truncated(data):
    """Recorta la última pista incompleta hasta que mido la lee."""
    ntr, chunks = _walk_chunks(data)
    hdr_end = 8 + struct.unpack(">I", data[4:8])[0]
    out_tracks = []
    for pos, cid, ln, avail in chunks:
        if cid != b"MTrk":
            continue
        body = data[pos + 8:pos + 8 + avail]
        if avail < ln:
            out_tracks.append(("trunc", body))
            break
        out_tracks.append(("ok", body))
    if not out_tracks:
        return None
    header = bytearray(data[:hdr_end])
    header[10:12] = struct.pack(">H", len(out_tracks))
    base = b"".join(b"MTrk" + struct.pack(">I", len(bd)) + bd
                    for kind, bd in out_tracks if kind == "ok")
    ntk = [t for t in out_tracks if t[0] == "trunc"]
    if not ntk:
        return bytes(header) + base
    body = ntk[0][1]
    for cut in range(0, 40):
        piece = body[:len(body) - cut] if cut else body
        cand_body = piece + b"\x00\xff\x2f\x00"
        cand = bytes(header) + base + b"MTrk" + struct.pack(">I", len(cand_body)) + cand_body
        try:
            _parse(cand, clip=True)
            return cand
        except Exception:  # noqa: BLE001
            continue
    # La pista truncada no se salva: se descarta y se deja lo completo
    if base:
        header[10:12] = struct.pack(">H", len(out_tracks) - 1)
        return bytes(header) + base
    return None


def repair_bytes(data):
    """Intenta reparar. Devuelve (bytes|None, nota)."""
    notes = []
    work = data
    if work[:4] != b"MThd":
        off = work.find(b"MThd", 0, 4096)
        if off <= 0:
            return None, "sin cabecera MThd"
        work = work[off:]
        notes.append(f"recortado prefijo de {off} bytes")
    work, nks = _patch_keysigs(work)
    if nks:
        notes.append(f"{nks} armadura(s) inválida(s) normalizada(s) a Do mayor")
    # ¿truncado?
    ntr, chunks = _walk_chunks(work)
    tracks = [c for c in chunks if c[1] == b"MTrk"]
    if any(c[3] < c[2] for c in tracks) or len(tracks) < ntr:
        sal = _salvage_truncated(work)
        if sal is None:
            return None, "no salvable (ninguna pista legible)"
        work = sal
        notes.append("pista truncada recortada y cerrada")
    try:
        mid = _parse(work, clip=True)
    except Exception as e:  # noqa: BLE001
        return None, f"sigue ilegible: {type(e).__name__}: {e}"
    if _count_notes(mid) == 0:
        return None, "tras reparar no quedan notas"
    # Reescritura limpia con mido (recorta bytes >127 al guardar)
    buf = io.BytesIO()
    try:
        mid.save(file=buf)
    except Exception as e:  # noqa: BLE001
        return None, f"no se pudo reescribir: {e}"
    if any(True for tr in mid.tracks for m in tr if getattr(m, "type", "") == "note_on"):
        notes.append("reescrito con mido (bytes de dato recortados a 0..127)")
    return buf.getvalue(), "; ".join(notes) or "reescrito"


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES CLI
# ══════════════════════════════════════════════════════════════════════════════

def _expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for root, _d, files in os.walk(p):
                for fn in sorted(files):
                    if fn.lower().endswith((".mid", ".midi", ".kar", ".rmi")):
                        out.append(os.path.join(root, fn))
        else:
            g = sorted(glob.glob(p))
            out.extend(g if g else [p])
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return sorted(uniq)


def _summary(results):
    cnt = Counter(r["category"] for r in results)
    total = len(results)
    ok = cnt.get("OK", 0)
    return cnt, total, ok


def _print_report(results, only=None, quiet=False):
    cnt, total, ok = _summary(results)
    pct = 100.0 * ok / total if total else 0.0
    rep = sum(1 for r in results if r["category"] != "OK" and r["repairable"])
    print(f"\n{_c('B')}CORPUS DOCTOR — {total} archivos{_c('R')}")
    col = _c("GRN") if pct >= 95 else (_c("YEL") if pct >= 80 else _c("RED"))
    print(f"  Sanos: {col}{ok}/{total} ({pct:.1f}%){_c('R')}   ·   "
          f"con problemas: {total - ok}   ·   reparables/salvables: {rep}\n")
    for cat in sorted(cnt, key=lambda c: (SEVERITY.get(c, 5), c)):
        if cat == "OK":
            continue
        tag = " ⚙" if cat in REPAIRABLE else ""
        print(f"  {cat:<12} {cnt[cat]:>5}{tag}")
    if quiet:
        return
    bad = [r for r in results if r["category"] != "OK" and (not only or r["category"] == only)]
    if bad:
        print(f"\n  {'categoría':<12} {'rep':<4} archivo / detalle")
        for r in sorted(bad, key=lambda r: (SEVERITY.get(r["category"], 5), r["path"])):
            mark = "sí" if r["repairable"] else "no"
            print(f"  {r['category']:<12} {mark:<4} {r['path']}")
            if r["detail"]:
                print(f"  {'':<17}{_c('G')}{r['detail']}{_c('R')}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  COMANDOS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_scan(args):
    files = _expand(args.paths)
    if not files:
        print("No se encontraron archivos MIDI.", file=sys.stderr)
        return 1
    results = []
    for i, p in enumerate(files, 1):
        if not args.quiet and sys.stderr.isatty():
            print(f"\r  escaneando {i}/{len(files)}", end="", file=sys.stderr, flush=True)
        results.append(diagnose(p, min_notes=args.min_notes))
    if sys.stderr.isatty() and not args.quiet:
        print("\r" + " " * 30 + "\r", end="", file=sys.stderr)
    _print_report(results, quiet=args.quiet)
    if args.json:
        _ensure_parent(args.json)
        cnt, total, ok = _summary(results)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"version": VERSION, "total": total, "ok": ok,
                       "healthy_pct": round(100.0 * ok / total, 2),
                       "counts": dict(cnt), "files": results}, f, ensure_ascii=False, indent=1)
        print(f"  Informe guardado en {args.json}")
    bad = sum(1 for r in results if r["category"] != "OK")
    return 3 if (args.strict and bad) else 0


def cmd_report(args):
    try:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"No se puede leer el informe: {e}", file=sys.stderr)
        return 1
    _print_report(data["files"], only=args.category)
    return 0


def cmd_quarantine(args):
    files = _expand(args.paths)
    results = [diagnose(p) for p in files]
    bad = [r for r in results if r["category"] != "OK"]
    mode = "EJECUCIÓN" if args.apply else "SIMULACIÓN (usa --apply para ejecutar)"
    print(f"\n{_c('B')}QUARANTINE — {mode}{_c('R')}  ({len(bad)} de {len(results)} con problemas)\n")
    manifest = []
    for r in bad:
        src = r["path"]
        cat = r["category"]
        dst = os.path.join(args.dest, cat, os.path.basename(src))
        line = f"  {cat:<12} {src}"
        entry = {"path": src, "category": cat, "quarantined_to": dst, "repaired_to": None, "note": ""}
        if args.repair and r["repairable"]:
            with open(src, "rb") as f:
                fixed, note = repair_bytes(f.read())
            entry["note"] = note
            if fixed is not None:
                rdst = os.path.join(args.dest, "repaired", os.path.basename(src))
                entry["repaired_to"] = rdst
                line += f"\n  {'':<12} ⚙ reparable → {rdst}  ({note})"
                if args.apply:
                    _ensure_parent(rdst)
                    with open(rdst, "wb") as f:
                        f.write(fixed)
            else:
                line += f"\n  {'':<12} ✗ no se pudo reparar: {note}"
        print(line)
        if args.apply:
            _ensure_parent(dst)
            (shutil.copy2 if args.copy else shutil.move)(src, dst)
        manifest.append(entry)
    if args.apply:
        mp = os.path.join(args.dest, "manifest.json")
        _ensure_parent(mp)
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        print(f"\n  Manifiesto: {mp}")
    print()
    return 0


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="corpus_doctor.py",
        description="Escaneo exhaustivo de salud de un corpus MIDI, con cuarentena y reparación.")
    sub = ap.add_subparsers(dest="cmd_name", required=True)

    s = sub.add_parser("scan", help="Clasifica todos los MIDIs de un corpus")
    s.add_argument("paths", nargs="+")
    s.add_argument("--json")
    s.add_argument("--min-notes", type=int, default=1)
    s.add_argument("--strict", action="store_true")
    s.add_argument("--quiet", "-q", action="store_true")
    s.add_argument("--no-color", action="store_true")
    s.set_defaults(fn=cmd_scan)

    r = sub.add_parser("report", help="Muestra un informe JSON guardado")
    r.add_argument("json")
    r.add_argument("--category")
    r.add_argument("--no-color", action="store_true")
    r.set_defaults(fn=cmd_report)

    q = sub.add_parser("quarantine", help="Aparta y/o repara los archivos con problemas")
    q.add_argument("paths", nargs="+")
    q.add_argument("--dest", default="quarantine")
    q.add_argument("--repair", action="store_true")
    q.add_argument("--apply", action="store_true")
    q.add_argument("--copy", action="store_true")
    q.add_argument("--no-color", action="store_true")
    q.set_defaults(fn=cmd_quarantine)

    args = ap.parse_args(argv)
    if getattr(args, "no_color", False):
        _USE_COLOR = False
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
