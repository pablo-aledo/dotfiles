#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    PLAYABILITY AUDITOR  v1.0                                 ║
║      Auditoría de tocabilidad e idiomática instrumental para MIDIs          ║
║                                                                              ║
║  El ecosistema genera MIDIs orquestales (orchestrator, piano_to_orchestra,  ║
║  piano_expander, ml_expander, stitcher...) y evalúa su calidad musical      ║
║  (quality_scorer, section_score, voice_aligner), pero ninguna herramienta   ║
║  verifica si un intérprete HUMANO podría tocar cada parte. Esta herramienta ║
║  cubre ese hueco: audita cada pista contra el perfil físico y técnico del   ║
║  instrumento que la toca, y opcionalmente corrige lo corregible.            ║
║                                                                              ║
║  COMPROBACIONES:                                                             ║
║    rango        — notas fuera del rango físico (ERROR) o de la zona         ║
║                   cómoda / tesitura idiomática (AVISO)                       ║
║    polifonia    — acordes en instrumentos monofónicos; dobles/triples       ║
║                   cuerdas no idiomáticas en cuerda frotada                   ║
║    respiracion  — frases continuas sin pausa para respirar (vientos/voz)    ║
║    velocidad    — densidad de notas/segundo por encima del límite técnico   ║
║    repeticion   — repetición de la misma nota más rápida que la             ║
║                   articulación posible (picado/staccato)                     ║
║    saltos       — saltos interválicos grandes a gran velocidad              ║
║    dinamica     — combinaciones dinámica×registro poco realistas            ║
║                   (ff en flauta grave, pp en metal sobreagudo...)            ║
║    percusion    — más golpes simultáneos de los que permiten dos manos      ║
║                                                                              ║
║  FLUJO:                                                                      ║
║  [1] LECTURA    — parser MIDI propio (formato 0/1, mapa de tempo y compás)  ║
║  [2] DETECCIÓN  — instrumento por pista: program change + nombre + canal 10 ║
║  [3] AUDITORÍA  — cada comprobación emite incidencias ERROR/AVISO/INFO      ║
║  [4] INFORME    — por pista y global, con score de tocabilidad 0.0–1.0     ║
║  [5] FIX (opc.) — transpone octavas fuera de rango, recorta solapes en      ║
║                   monofónicos e inserta respiraciones acortando notas       ║
║                                                                              ║
║  USO:                                                                        ║
║    python playability_auditor.py obra.mid                                    ║
║    python playability_auditor.py obra.mid --verbose                          ║
║    python playability_auditor.py obra.mid --fix --output obra_fixed.mid     ║
║    python playability_auditor.py obra.mid --json informe.json               ║
║    python playability_auditor.py obra.mid --tracks 1 3                      ║
║    python playability_auditor.py obra.mid --scope 8:24                      ║
║    python playability_auditor.py obra.mid --skip respiracion saltos         ║
║    python playability_auditor.py obra.mid --only rango polifonia            ║
║    python playability_auditor.py obra.mid --profile estricto               ║
║    python playability_auditor.py obra.mid --quiet                           ║
║    python playability_auditor.py --list-instruments                         ║
║                                                                              ║
║  OPCIONES:                                                                   ║
║    midi               Archivo(s) MIDI a auditar                              ║
║    --tracks N...      Auditar solo estas pistas (índices)                    ║
║    --scope S:E        Limitar a compases S:E (ej: --scope 8:24)             ║
║    --skip CHECK...    Omitir comprobaciones                                  ║
║    --only CHECK...    Ejecutar solo estas comprobaciones                     ║
║    --profile P        estricto | estandar | tolerante (default: estandar)   ║
║    --instrument T=I   Forzar instrumento de la pista T (ej: 2=violin)       ║
║    --fix              Generar MIDI corregido                                 ║
║    --output FILE      Ruta del MIDI corregido (default: <obra>_fixed.mid)   ║
║    --json FILE        Exportar informe estructurado a JSON                   ║
║    --max-issues N     Máx. incidencias detalladas por pista (default: 12)   ║
║    --quiet            Solo resumen de una línea por pista + exit code        ║
║    --verbose          Mostrar todas las incidencias con compás y detalle     ║
║    --no-color         Desactivar colores ANSI                                ║
║    --list-instruments Mostrar el catálogo de perfiles y salir                ║
║                                                                              ║
║  EXIT CODE: 0 = sin errores · 1 = hay ERRORes (integrable en runner.py)     ║
║                                                                              ║
║  COMO MÓDULO (para runner, quality_scorer u otros scripts):                  ║
║    from playability_auditor import audit_midi, fix_midi                      ║
║    report = audit_midi("obra.mid", profile="estandar")                       ║
║    print(report.global_score, len(report.errors()))                          ║
║                                                                              ║
║  DEPENDENCIAS: ninguna (solo stdlib) — incluye lector/escritor MIDI propio   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

VERSION = "1.0"

# ══════════════════════════════════════════════════════════════════════════════
#  COLORES
# ══════════════════════════════════════════════════════════════════════════════

_COLORS = {
    "R": "\033[0m",   # reset
    "B": "\033[1m",   # bold
    "G": "\033[90m",  # gris
    "RED": "\033[91m",
    "YEL": "\033[93m",
    "GRN": "\033[92m",
    "CYA": "\033[96m",
}
_USE_COLOR = sys.stdout.isatty()


def _c(key: str) -> str:
    return _COLORS.get(key, "") if _USE_COLOR else ""


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_name(p: int) -> str:
    return f"{NOTE_NAMES[p % 12]}{p // 12 - 1}"


# ══════════════════════════════════════════════════════════════════════════════
#  LECTOR / ESCRITOR MIDI (SMF 0/1, solo stdlib)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MidiEvent:
    """Evento MIDI con tiempo absoluto en ticks."""
    abs: int
    kind: str            # note_on | note_off | meta | channel | sysex
    channel: int = 0
    data: bytes = b""    # bytes crudos del mensaje (sin delta), para re-escritura
    pitch: int = 0
    vel: int = 0
    meta_type: int = -1


@dataclass
class MidiTrackData:
    name: str = ""
    events: List[MidiEvent] = field(default_factory=list)
    programs: List[int] = field(default_factory=list)   # program changes vistos
    channels: List[int] = field(default_factory=list)   # canales usados por notas


@dataclass
class MidiData:
    fmt: int = 1
    tpb: int = 480
    tracks: List[MidiTrackData] = field(default_factory=list)
    tempo_map: List[Tuple[int, int]] = field(default_factory=list)      # (tick, us/negra)
    timesig_map: List[Tuple[int, int, int]] = field(default_factory=list)  # (tick, num, den)


def _read_vlq(buf: bytes, i: int) -> Tuple[int, int]:
    val = 0
    while True:
        b = buf[i]; i += 1
        val = (val << 7) | (b & 0x7F)
        if not b & 0x80:
            return val, i


def _write_vlq(val: int) -> bytes:
    out = [val & 0x7F]
    val >>= 7
    while val:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    return bytes(reversed(out))


def read_midi(path: str) -> MidiData:
    raw = Path(path).read_bytes()
    if raw[:4] != b"MThd":
        raise ValueError(f"{path}: no es un archivo MIDI (falta MThd)")
    fmt = int.from_bytes(raw[8:10], "big")
    n_tracks = int.from_bytes(raw[10:12], "big")
    division = int.from_bytes(raw[12:14], "big")
    if division & 0x8000:
        raise ValueError(f"{path}: división SMPTE no soportada")
    mid = MidiData(fmt=fmt, tpb=division)

    i = 14
    for _ in range(n_tracks):
        if raw[i:i+4] != b"MTrk":
            raise ValueError(f"{path}: chunk de pista corrupto en offset {i}")
        length = int.from_bytes(raw[i+4:i+8], "big")
        chunk = raw[i+8:i+8+length]
        i += 8 + length
        trk = MidiTrackData()
        t = 0
        j = 0
        status = 0
        while j < len(chunk):
            delta, j = _read_vlq(chunk, j)
            t += delta
            b0 = chunk[j]
            if b0 & 0x80:
                status = b0
                j += 1
            if status == 0xFF:                                  # meta
                mtype = chunk[j]; j += 1
                mlen, j = _read_vlq(chunk, j)
                mdata = chunk[j:j+mlen]; j += mlen
                ev = MidiEvent(abs=t, kind="meta", meta_type=mtype,
                               data=bytes([0xFF, mtype]) + _write_vlq(mlen) + mdata)
                if mtype == 0x03 and not trk.name:
                    trk.name = mdata.decode("latin-1", errors="replace").strip()
                elif mtype == 0x51 and mlen == 3:
                    mid.tempo_map.append((t, int.from_bytes(mdata, "big")))
                elif mtype == 0x58 and mlen >= 2:
                    mid.timesig_map.append((t, mdata[0], 2 ** mdata[1]))
                if mtype != 0x2F:                               # end-of-track se regenera
                    trk.events.append(ev)
            elif status in (0xF0, 0xF7):                        # sysex
                slen, j = _read_vlq(chunk, j)
                sdata = chunk[j:j+slen]; j += slen
                trk.events.append(MidiEvent(abs=t, kind="sysex",
                                            data=bytes([status]) + _write_vlq(slen) + sdata))
            else:                                               # mensaje de canal
                hi, ch = status & 0xF0, status & 0x0F
                if hi in (0xC0, 0xD0):
                    d1 = chunk[j]; j += 1
                    ev = MidiEvent(abs=t, kind="channel", channel=ch,
                                   data=bytes([status, d1]))
                    if hi == 0xC0:
                        trk.programs.append(d1)
                    trk.events.append(ev)
                else:
                    d1, d2 = chunk[j], chunk[j+1]; j += 2
                    if hi == 0x90 and d2 > 0:
                        trk.events.append(MidiEvent(abs=t, kind="note_on", channel=ch,
                                                    pitch=d1, vel=d2))
                        if ch not in trk.channels:
                            trk.channels.append(ch)
                    elif hi == 0x80 or (hi == 0x90 and d2 == 0):
                        trk.events.append(MidiEvent(abs=t, kind="note_off", channel=ch,
                                                    pitch=d1, vel=d2))
                    else:
                        trk.events.append(MidiEvent(abs=t, kind="channel", channel=ch,
                                                    data=bytes([status, d1, d2])))
        mid.tracks.append(trk)

    if not mid.tempo_map:
        mid.tempo_map = [(0, 500000)]                            # 120 bpm por defecto
    mid.tempo_map.sort()
    if not mid.timesig_map:
        mid.timesig_map = [(0, 4, 4)]
    mid.timesig_map.sort()
    return mid


def write_midi(mid: MidiData, path: str):
    chunks = []
    for trk in mid.tracks:
        evs = sorted(trk.events, key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))
        body = bytearray()
        last = 0
        for ev in evs:
            body += _write_vlq(max(0, ev.abs - last))
            last = ev.abs
            if ev.kind == "note_on":
                body += bytes([0x90 | (ev.channel & 0x0F), ev.pitch & 0x7F, ev.vel & 0x7F])
            elif ev.kind == "note_off":
                body += bytes([0x80 | (ev.channel & 0x0F), ev.pitch & 0x7F, ev.vel & 0x7F])
            else:
                body += ev.data
        body += _write_vlq(0) + bytes([0xFF, 0x2F, 0x00])
        chunks.append(b"MTrk" + len(body).to_bytes(4, "big") + bytes(body))
    header = (b"MThd" + (6).to_bytes(4, "big") + mid.fmt.to_bytes(2, "big")
              + len(mid.tracks).to_bytes(2, "big") + mid.tpb.to_bytes(2, "big"))
    Path(path).write_bytes(header + b"".join(chunks))


# ── Conversión ticks → segundos / compases ───────────────────────────────────

class TimeContext:
    """Mapa tempo+compás para convertir ticks a segundos y números de compás."""

    def __init__(self, mid: MidiData):
        self.tpb = mid.tpb
        self.tempo_map = mid.tempo_map
        self.timesig_map = mid.timesig_map
        # segmentos acumulados de tempo: (tick, sec_acumulados, us_por_tick)
        self._segs = []
        sec, prev_tick, prev_us = 0.0, 0, self.tempo_map[0][1]
        for tick, us in self.tempo_map:
            sec += (tick - prev_tick) * prev_us / 1e6 / self.tpb
            self._segs.append((tick, sec, us))
            prev_tick, prev_us = tick, us
        if not self._segs or self._segs[0][0] > 0:
            self._segs.insert(0, (0, 0.0, self.tempo_map[0][1]))
        # segmentos de compás: (tick_inicio, compas_inicio (1-based), ticks_por_compas)
        self._bars = []
        bar, prev_tick = 1.0, 0
        prev_tpc = self.tpb * 4 * self.timesig_map[0][1] // self.timesig_map[0][2]
        for tick, num, den in self.timesig_map:
            bar += (tick - prev_tick) / prev_tpc
            tpc = max(1, self.tpb * 4 * num // den)
            self._bars.append((tick, bar, tpc))
            prev_tick, prev_tpc = tick, tpc
        if not self._bars or self._bars[0][0] > 0:
            self._bars.insert(0, (0, 1.0, prev_tpc))

    def sec(self, tick: int) -> float:
        seg = self._segs[0]
        for s in self._segs:
            if s[0] <= tick:
                seg = s
            else:
                break
        t0, sec0, us = seg
        return sec0 + (tick - t0) * us / 1e6 / self.tpb

    def bar(self, tick: int) -> int:
        seg = self._bars[0]
        for s in self._bars:
            if s[0] <= tick:
                seg = s
            else:
                break
        t0, bar0, tpc = seg
        return int(bar0 + (tick - t0) / tpc)

    def bar_range_ticks(self, bar_start: int, bar_end: int) -> Tuple[int, int]:
        """Ticks [ini, fin) del rango de compases [bar_start, bar_end] (1-based)."""
        # búsqueda lineal inversa sencilla (suficiente para auditoría)
        t0, t1 = None, None
        tick, step = 0, max(1, self.tpb // 8)
        limit = 10_000_000
        while tick < limit and (t0 is None or t1 is None):
            b = self.bar(tick)
            if t0 is None and b >= bar_start:
                t0 = tick
            if t1 is None and b > bar_end:
                t1 = tick
            tick += step
        return (t0 or 0), (t1 if t1 is not None else limit)


# ══════════════════════════════════════════════════════════════════════════════
#  BASE DE DATOS DE INSTRUMENTOS
# ══════════════════════════════════════════════════════════════════════════════
#  range:  rango físico [min, max] en pitch MIDI
#  sweet:  zona cómoda / tesitura idiomática
#  mono:   instrumento monofónico (una nota a la vez)
#  breath: necesita respirar; max_breath = segundos de frase continua (AVISO)
#  max_nps: notas/segundo sostenibles (ventana de 2 s)
#  max_rep: repeticiones de la misma nota por segundo (picado/staccato)
#  max_poly: nº máximo de notas simultáneas (cuerdas: dobles cuerdas, etc.)

INSTRUMENT_DB = {
    "piano":       dict(es="Piano", family="teclado", range=(21, 108), sweet=(28, 103),
                        mono=False, breath=False, max_breath=0, max_nps=24, max_rep=13, max_poly=10),
    "celesta":     dict(es="Celesta", family="teclado", range=(60, 108), sweet=(60, 103),
                        mono=False, breath=False, max_breath=0, max_nps=18, max_rep=10, max_poly=8),
    "organo":      dict(es="Órgano", family="teclado", range=(24, 108), sweet=(29, 96),
                        mono=False, breath=False, max_breath=0, max_nps=18, max_rep=9, max_poly=10),
    "clave":       dict(es="Clave", family="teclado", range=(29, 89), sweet=(36, 84),
                        mono=False, breath=False, max_breath=0, max_nps=20, max_rep=11, max_poly=8),
    "guitarra":    dict(es="Guitarra", family="cuerda_pulsada", range=(40, 88), sweet=(40, 81),
                        mono=False, breath=False, max_breath=0, max_nps=16, max_rep=10, max_poly=6),
    "arpa":        dict(es="Arpa", family="cuerda_pulsada", range=(24, 103), sweet=(31, 96),
                        mono=False, breath=False, max_breath=0, max_nps=18, max_rep=6, max_poly=8),
    "bajo":        dict(es="Bajo", family="cuerda_pulsada", range=(28, 67), sweet=(28, 60),
                        mono=False, breath=False, max_breath=0, max_nps=12, max_rep=8, max_poly=2),
    "violin":      dict(es="Violín", family="cuerda_frotada", range=(55, 105), sweet=(55, 96),
                        mono=False, breath=False, max_breath=0, max_nps=16, max_rep=12, max_poly=2),
    "viola":       dict(es="Viola", family="cuerda_frotada", range=(48, 88), sweet=(48, 81),
                        mono=False, breath=False, max_breath=0, max_nps=14, max_rep=11, max_poly=2),
    "violonchelo": dict(es="Violonchelo", family="cuerda_frotada", range=(36, 84), sweet=(36, 76),
                        mono=False, breath=False, max_breath=0, max_nps=13, max_rep=10, max_poly=2),
    "contrabajo":  dict(es="Contrabajo", family="cuerda_frotada", range=(28, 67), sweet=(28, 55),
                        mono=False, breath=False, max_breath=0, max_nps=10, max_rep=8, max_poly=2),
    "cuerdas":     dict(es="Sección de cuerdas", family="cuerda_frotada", range=(28, 105), sweet=(36, 96),
                        mono=False, breath=False, max_breath=0, max_nps=14, max_rep=11, max_poly=8),
    "flautin":     dict(es="Flautín", family="viento_madera", range=(74, 108), sweet=(76, 103),
                        mono=True, breath=True, max_breath=18, max_nps=13, max_rep=9, max_poly=1),
    "flauta":      dict(es="Flauta", family="viento_madera", range=(60, 98), sweet=(62, 93),
                        mono=True, breath=True, max_breath=18, max_nps=13, max_rep=9, max_poly=1),
    "oboe":        dict(es="Oboe", family="viento_madera", range=(58, 93), sweet=(60, 88),
                        mono=True, breath=True, max_breath=24, max_nps=11, max_rep=8, max_poly=1),
    "corno_ingles": dict(es="Corno inglés", family="viento_madera", range=(52, 86), sweet=(54, 81),
                        mono=True, breath=True, max_breath=22, max_nps=10, max_rep=8, max_poly=1),
    "clarinete":   dict(es="Clarinete", family="viento_madera", range=(50, 94), sweet=(50, 89),
                        mono=True, breath=True, max_breath=22, max_nps=13, max_rep=9, max_poly=1),
    "fagot":       dict(es="Fagot", family="viento_madera", range=(34, 75), sweet=(34, 69),
                        mono=True, breath=True, max_breath=20, max_nps=11, max_rep=8, max_poly=1),
    "saxofon":     dict(es="Saxofón", family="viento_madera", range=(49, 88), sweet=(51, 83),
                        mono=True, breath=True, max_breath=18, max_nps=13, max_rep=9, max_poly=1),
    "trompeta":    dict(es="Trompeta", family="viento_metal", range=(54, 86), sweet=(56, 82),
                        mono=True, breath=True, max_breath=14, max_nps=11, max_rep=9, max_poly=1),
    "trompa":      dict(es="Trompa", family="viento_metal", range=(35, 77), sweet=(41, 72),
                        mono=True, breath=True, max_breath=14, max_nps=9, max_rep=7, max_poly=1),
    "trombon":     dict(es="Trombón", family="viento_metal", range=(40, 77), sweet=(40, 72),
                        mono=True, breath=True, max_breath=14, max_nps=8, max_rep=7, max_poly=1),
    "tuba":        dict(es="Tuba", family="viento_metal", range=(26, 65), sweet=(29, 58),
                        mono=True, breath=True, max_breath=12, max_nps=7, max_rep=6, max_poly=1),
    "voz":         dict(es="Voz", family="voz", range=(40, 84), sweet=(48, 79),
                        mono=True, breath=True, max_breath=12, max_nps=8, max_rep=6, max_poly=1),
    "timbal":      dict(es="Timbales", family="percusion_afinada", range=(40, 60), sweet=(41, 57),
                        mono=False, breath=False, max_breath=0, max_nps=14, max_rep=14, max_poly=2),
    "laminas":     dict(es="Láminas (xilófono/marimba/vibráfono)", family="percusion_afinada",
                        range=(45, 108), sweet=(48, 103),
                        mono=False, breath=False, max_breath=0, max_nps=16, max_rep=12, max_poly=4),
    "percusion":   dict(es="Percusión (batería/set)", family="percusion", range=(27, 87), sweet=(35, 81),
                        mono=False, breath=False, max_breath=0, max_nps=18, max_rep=14, max_poly=4),
    "sintetizador": dict(es="Sintetizador / otros", family="electronico", range=(0, 127), sweet=(24, 108),
                        mono=False, breath=False, max_breath=0, max_nps=30, max_rep=20, max_poly=16),
}

# Mapeo General MIDI (program 0-127) → clave de INSTRUMENT_DB
_GM_MAP = [
    (range(0, 8), "piano"), (range(8, 9), "celesta"), (range(9, 16), "laminas"),
    (range(16, 24), "organo"), (range(24, 32), "guitarra"), (range(32, 40), "bajo"),
    (range(40, 41), "violin"), (range(41, 42), "viola"), (range(42, 43), "violonchelo"),
    (range(43, 44), "contrabajo"), (range(44, 46), "cuerdas"), (range(46, 47), "arpa"),
    (range(47, 48), "timbal"), (range(48, 52), "cuerdas"), (range(52, 56), "voz"),
    (range(56, 57), "trompeta"), (range(57, 58), "trombon"), (range(58, 59), "tuba"),
    (range(59, 60), "trompeta"), (range(60, 61), "trompa"), (range(61, 64), "viento_metal_gen"),
    (range(64, 68), "saxofon"), (range(68, 69), "oboe"), (range(69, 70), "corno_ingles"),
    (range(70, 71), "fagot"), (range(71, 72), "clarinete"), (range(72, 73), "flautin"),
    (range(73, 74), "flauta"), (range(74, 80), "flauta"), (range(80, 128), "sintetizador"),
]
INSTRUMENT_DB["viento_metal_gen"] = dict(
    es="Sección de metales", family="viento_metal", range=(35, 86), sweet=(41, 79),
    mono=False, breath=True, max_breath=14, max_nps=10, max_rep=8, max_poly=4)

# Palabras clave en nombres de pista (ES / EN / abreviaturas) → clave de perfil
_NAME_KEYWORDS = [
    (r"flaut[ií]n|piccolo|picc\b", "flautin"),
    (r"flaut|flute|\bfl\.?\b", "flauta"),
    (r"oboe|\bob\.?\b", "oboe"),
    (r"corno ingl|english horn", "corno_ingles"),
    (r"clarinet|\bcl\.?\b", "clarinete"),
    (r"fagot|bassoon|\bfg\.?\b|\bbsn\b", "fagot"),
    (r"sax", "saxofon"),
    (r"trompeta|trumpet|\btpt\b|\btrp\b", "trompeta"),
    (r"tromb[oó]n|trombone|\btbn\b", "trombon"),
    (r"trompa|french horn|\bhorn\b|\bhn\.?\b", "trompa"),
    (r"tuba", "tuba"),
    (r"timbal|timpani|timp", "timbal"),
    (r"xilof|xylo|marimba|vibraf|vibrap|glocken", "laminas"),
    (r"viol[ií]n|violin|\bvln\b|\bvl\.?\b", "violin"),
    (r"violonchelo|chelo|cello|\bvc\.?\b|\bvcl\b", "violonchelo"),
    (r"viola|\bvla\b", "viola"),
    (r"contrabajo|double ?bass|\bcb\.?\b|\bdb\b", "contrabajo"),
    (r"cuerdas|strings", "cuerdas"),
    (r"arpa|harp", "arpa"),
    (r"guitarra|guitar|\bgtr\b", "guitarra"),
    (r"\bbajo\b|\bbass\b", "bajo"),
    (r"piano|\bpno\b", "piano"),
    (r"organo|órgano|organ", "organo"),
    (r"celesta", "celesta"),
    (r"voz|voice|soprano|alto\b|tenor|bar[ií]tono|coro|choir|vocal", "voz"),
    (r"percusi|drum|bater[ií]a|perc", "percusion"),
    (r"sint|synth|pad\b|lead\b", "sintetizador"),
]


def gm_to_profile(program: int) -> str:
    for rng, key in _GM_MAP:
        if program in rng:
            return key
    return "sintetizador"


def name_to_profile(name: str) -> Optional[str]:
    low = name.lower()
    for pattern, key in _NAME_KEYWORDS:
        if re.search(pattern, low):
            return key
    return None


# Escalado de umbrales por perfil de exigencia
_PROFILE_SCALE = {"estricto": 0.85, "estandar": 1.0, "tolerante": 1.2}

ALL_CHECKS = ["rango", "polifonia", "respiracion", "velocidad",
              "repeticion", "saltos", "dinamica", "percusion"]


# ══════════════════════════════════════════════════════════════════════════════
#  ESTRUCTURAS DE AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Note:
    pitch: int
    vel: int
    start: int     # ticks
    end: int       # ticks
    channel: int


@dataclass
class Issue:
    check: str
    severity: str         # ERROR | AVISO | INFO
    track: int
    bar: int
    msg: str
    pitch: Optional[int] = None
    count: int = 1        # incidencias agrupadas (mismo check+compás)


@dataclass
class TrackAudit:
    index: int
    name: str
    instrument: str        # clave de INSTRUMENT_DB
    instrument_es: str
    n_notes: int
    issues: List[Issue] = field(default_factory=list)
    score: float = 1.0

    def errors(self):
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self):
        return [i for i in self.issues if i.severity == "AVISO"]


@dataclass
class AuditReport:
    file: str
    profile: str
    tracks: List[TrackAudit] = field(default_factory=list)
    global_score: float = 1.0
    fixed_path: Optional[str] = None

    def errors(self):
        return [i for t in self.tracks for i in t.errors()]

    def warnings(self):
        return [i for t in self.tracks for i in t.warnings()]


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACCIÓN DE NOTAS Y DETECCIÓN DE INSTRUMENTO
# ══════════════════════════════════════════════════════════════════════════════

def extract_notes(trk: MidiTrackData) -> List[Note]:
    notes, open_n = [], {}
    for ev in trk.events:
        key = (ev.channel, ev.pitch)
        if ev.kind == "note_on":
            if key in open_n:                      # nota re-disparada sin off
                n = open_n.pop(key)
                n.end = ev.abs
                if n.end > n.start:
                    notes.append(n)
            open_n[key] = Note(ev.pitch, ev.vel, ev.abs, ev.abs, ev.channel)
        elif ev.kind == "note_off" and key in open_n:
            n = open_n.pop(key)
            n.end = ev.abs
            if n.end > n.start:
                notes.append(n)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def detect_instrument(trk: MidiTrackData, override: Optional[str] = None) -> str:
    if override:
        if override not in INSTRUMENT_DB:
            raise ValueError(f"Instrumento desconocido: {override} "
                             f"(usa --list-instruments para ver el catálogo)")
        return override
    if 9 in trk.channels:                          # canal 10 = percusión GM
        return "percusion"
    by_name = name_to_profile(trk.name) if trk.name else None
    if by_name:
        return by_name
    if trk.programs:
        return gm_to_profile(trk.programs[0])
    return "sintetizador"


# ══════════════════════════════════════════════════════════════════════════════
#  COMPROBACIONES
# ══════════════════════════════════════════════════════════════════════════════

def _group(issues: List[Issue]) -> List[Issue]:
    """Agrupa incidencias idénticas (check+severidad+compás+msg base)."""
    seen: Dict[tuple, Issue] = {}
    for it in issues:
        key = (it.check, it.severity, it.bar, it.msg)
        if key in seen:
            seen[key].count += 1
        else:
            seen[key] = it
    return list(seen.values())


def check_rango(notes, prof, tc, scale, tidx):
    lo, hi = prof["range"]
    slo, shi = prof["sweet"]
    out = []
    for n in notes:
        if n.pitch < lo or n.pitch > hi:
            d = "grave" if n.pitch < lo else "agudo"
            out.append(Issue("rango", "ERROR", tidx, tc.bar(n.start),
                             f"{pitch_name(n.pitch)} fuera de rango físico "
                             f"({pitch_name(lo)}–{pitch_name(hi)}), demasiado {d}",
                             pitch=n.pitch))
        elif n.pitch < slo or n.pitch > shi:
            out.append(Issue("rango", "AVISO", tidx, tc.bar(n.start),
                             f"{pitch_name(n.pitch)} fuera de la zona cómoda "
                             f"({pitch_name(slo)}–{pitch_name(shi)})",
                             pitch=n.pitch))
    return out


def check_polifonia(notes, prof, tc, scale, tidx):
    out = []
    if prof["family"] == "percusion":
        return out
    max_poly = prof["max_poly"]
    events = []
    for n in notes:
        events.append((n.start, 1, n))
        events.append((n.end, -1, n))
    events.sort(key=lambda e: (e[0], e[1]))
    active = []
    for t, kind, n in events:
        if kind == 1:
            active.append(n)
            k = len(active)
            if prof["mono"] and k > 1:
                out.append(Issue("polifonia", "ERROR", tidx, tc.bar(t),
                                 f"{k} notas simultáneas en instrumento monofónico"))
            elif not prof["mono"] and k > max_poly:
                out.append(Issue("polifonia", "ERROR", tidx, tc.bar(t),
                                 f"{k} notas simultáneas (máx. idiomático: {max_poly})"))
            elif prof["family"] == "cuerda_frotada" and k == 2:
                iv = abs(active[-1].pitch - active[-2].pitch)
                dur = min(a.end for a in active) - t
                if iv > 12 and tc.sec(t + max(0, dur)) - tc.sec(t) > 0.5:
                    out.append(Issue("polifonia", "AVISO", tidx, tc.bar(t),
                                     f"doble cuerda de {iv} semitonos sostenida "
                                     f"(difícil más allá de una octava)"))
        else:
            if n in active:
                active.remove(n)
    return out


def check_respiracion(notes, prof, tc, scale, tidx):
    out = []
    if not prof["breath"] or not notes:
        return out
    limit = prof["max_breath"] * scale
    gap_min = 0.25                                  # silencio mínimo que cuenta como respiración
    phrase_start = tc.sec(notes[0].start)
    prev_end = tc.sec(notes[0].end)
    start_bar = tc.bar(notes[0].start)
    for n in notes[1:]:
        s, e = tc.sec(n.start), tc.sec(n.end)
        if s - prev_end >= gap_min:
            dur = prev_end - phrase_start
            if dur > limit:
                sev = "ERROR" if dur > limit * 1.5 else "AVISO"
                out.append(Issue("respiracion", sev, tidx, start_bar,
                                 f"frase continua de {dur:.1f}s sin respiración "
                                 f"(límite ≈ {limit:.0f}s)"))
            phrase_start = s
            start_bar = tc.bar(n.start)
        prev_end = max(prev_end, e)
    dur = prev_end - phrase_start
    if dur > limit:
        sev = "ERROR" if dur > limit * 1.5 else "AVISO"
        out.append(Issue("respiracion", sev, tidx, start_bar,
                         f"frase continua de {dur:.1f}s sin respiración "
                         f"(límite ≈ {limit:.0f}s)"))
    return out


def check_velocidad(notes, prof, tc, scale, tidx):
    out = []
    limit = prof["max_nps"] * scale
    win = 2.0
    # onsets únicos: un acorde / golpe múltiple cuenta como un solo gesto
    uniq = sorted({n.start for n in notes})
    if len(uniq) < 6:
        return out
    onsets = [tc.sec(t) for t in uniq]
    i = 0
    reported_bars = set()
    for j in range(len(onsets)):
        while onsets[j] - onsets[i] > win:
            i += 1
        k = j - i + 1
        if k >= 6:
            nps = k / max(0.25, onsets[j] - onsets[i])
            if nps > limit:
                bar = tc.bar(uniq[i])
                if bar not in reported_bars:
                    sev = "ERROR" if nps > limit * 1.3 else "AVISO"
                    out.append(Issue("velocidad", sev, tidx, bar,
                                     f"{nps:.1f} notas/s sostenidas "
                                     f"(límite técnico ≈ {limit:.0f}/s)"))
                    reported_bars.add(bar)
    return out


def check_repeticion(notes, prof, tc, scale, tidx):
    out = []
    limit = prof["max_rep"] * scale
    reported_bars = set()
    by_pitch: Dict[int, list] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(tc.sec(n.start))
    for pitch, ts in by_pitch.items():
        run = 1
        for a, b in zip(ts, ts[1:]):
            ioi = b - a
            if 0 < ioi < 1.0 / limit:
                run += 1
                if run >= 4:
                    bar = tc.bar(notes[0].start)  # aproximado; se refina abajo
                    # localizar compás real de la repetición
                    for n in notes:
                        if n.pitch == pitch and abs(tc.sec(n.start) - b) < 1e-6:
                            bar = tc.bar(n.start)
                            break
                    if (pitch, bar) not in reported_bars:
                        out.append(Issue("repeticion", "AVISO", tidx, bar,
                                         f"repetición de {pitch_name(pitch)} a "
                                         f"{1/ioi:.1f} golpes/s (límite ≈ {limit:.0f}/s)",
                                         pitch=pitch))
                        reported_bars.add((pitch, bar))
            else:
                run = 1
    return out


def check_saltos(notes, prof, tc, scale, tidx):
    out = []
    if prof["family"] in ("percusion", "percusion_afinada", "electronico", "teclado"):
        return out
    leap_lim = 19 if prof["family"] == "cuerda_frotada" else 16   # semitonos
    for a, b in zip(notes, notes[1:]):
        if b.start < a.end:                          # acorde, no salto melódico
            continue
        iv = abs(b.pitch - a.pitch)
        ioi = tc.sec(b.start) - tc.sec(a.start)
        if iv > leap_lim and 0 < ioi < 0.18 / scale:
            out.append(Issue("saltos", "AVISO", tidx, tc.bar(b.start),
                             f"salto de {iv} semitonos en {ioi*1000:.0f} ms "
                             f"({pitch_name(a.pitch)}→{pitch_name(b.pitch)})"))
    return out


def check_dinamica(notes, prof, tc, scale, tidx):
    out = []
    fam = prof["family"]
    slo, shi = prof["sweet"]
    for n in notes:
        if fam == "viento_madera" and n.pitch <= slo + 4 and n.vel >= 100:
            out.append(Issue("dinamica", "AVISO", tidx, tc.bar(n.start),
                             f"ff en registro grave ({pitch_name(n.pitch)}): "
                             f"proyección muy limitada en madera grave", pitch=n.pitch))
        elif fam == "viento_metal" and n.pitch >= shi - 2 and n.vel <= 40:
            out.append(Issue("dinamica", "AVISO", tidx, tc.bar(n.start),
                             f"pp en registro sobreagudo ({pitch_name(n.pitch)}): "
                             f"casi imposible de controlar en metal", pitch=n.pitch))
        elif fam == "voz" and n.pitch >= shi and n.vel <= 45:
            out.append(Issue("dinamica", "AVISO", tidx, tc.bar(n.start),
                             f"pp en el extremo agudo vocal ({pitch_name(n.pitch)})",
                             pitch=n.pitch))
    return out


def check_percusion(notes, prof, tc, scale, tidx):
    out = []
    if prof["family"] not in ("percusion", "percusion_afinada"):
        return out
    max_hands = 2 if prof["family"] == "percusion_afinada" else 4   # set: 2 manos + 2 pies
    by_tick: Dict[int, set] = {}
    for n in notes:
        by_tick.setdefault(n.start, set()).add(n.pitch)
    for t, pitches in sorted(by_tick.items()):
        if len(pitches) > max_hands:
            out.append(Issue("percusion", "AVISO", tidx, tc.bar(t),
                             f"{len(pitches)} golpes simultáneos "
                             f"(máx. ejecutable: {max_hands})"))
    return out


_CHECK_FUNCS = {
    "rango": check_rango, "polifonia": check_polifonia, "respiracion": check_respiracion,
    "velocidad": check_velocidad, "repeticion": check_repeticion, "saltos": check_saltos,
    "dinamica": check_dinamica, "percusion": check_percusion,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DE AUDITORÍA
# ══════════════════════════════════════════════════════════════════════════════

def _track_score(issues: List[Issue], n_notes: int) -> float:
    if n_notes == 0:
        return 1.0
    pen = 0.0
    for it in issues:
        w = {"ERROR": 0.06, "AVISO": 0.015, "INFO": 0.0}[it.severity]
        pen += w * min(it.count, 5)
    return max(0.0, round(1.0 - min(pen, 1.0), 3))


def audit_midi(path: str,
               profile: str = "estandar",
               tracks: Optional[List[int]] = None,
               scope: Optional[Tuple[int, int]] = None,
               checks: Optional[List[str]] = None,
               instrument_overrides: Optional[Dict[int, str]] = None) -> AuditReport:
    """API pública: audita un MIDI y devuelve un AuditReport."""
    if profile not in _PROFILE_SCALE:
        raise ValueError(f"Perfil desconocido: {profile}")
    scale = _PROFILE_SCALE[profile]
    checks = checks or ALL_CHECKS
    overrides = instrument_overrides or {}

    mid = read_midi(path)
    tc = TimeContext(mid)
    scope_ticks = tc.bar_range_ticks(*scope) if scope else None

    report = AuditReport(file=str(path), profile=profile)
    scores, weights = [], []
    for idx, trk in enumerate(mid.tracks):
        notes = extract_notes(trk)
        if not notes:
            continue                                  # pista de metadatos / vacía
        if tracks is not None and idx not in tracks:
            continue
        if scope_ticks:
            t0, t1 = scope_ticks
            notes = [n for n in notes if t0 <= n.start < t1]
            if not notes:
                continue
        inst = detect_instrument(trk, overrides.get(idx))
        prof = INSTRUMENT_DB[inst]
        issues: List[Issue] = []
        for ch in checks:
            issues.extend(_CHECK_FUNCS[ch](notes, prof, tc, scale, idx))
        issues = _group(issues)
        issues.sort(key=lambda i: (0 if i.severity == "ERROR" else 1, i.bar))
        ta = TrackAudit(index=idx, name=trk.name or f"Pista {idx}",
                        instrument=inst, instrument_es=prof["es"],
                        n_notes=len(notes), issues=issues,
                        score=_track_score(issues, len(notes)))
        report.tracks.append(ta)
        scores.append(ta.score)
        weights.append(len(notes))

    if scores:
        report.global_score = round(
            sum(s * w for s, w in zip(scores, weights)) / sum(weights), 3)
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  FIX — CORRECCIONES AUTOMÁTICAS
# ══════════════════════════════════════════════════════════════════════════════

def fix_midi(path: str, out_path: str,
             profile: str = "estandar",
             instrument_overrides: Optional[Dict[int, str]] = None) -> Dict[str, int]:
    """Corrige lo corregible y escribe un nuevo MIDI. Devuelve contadores.

    - rango:       transpone por octavas hacia el interior del rango físico
    - polifonia:   en monofónicos, recorta la nota anterior al entrar la nueva
                   (conserva legato) y elimina duplicados simultáneos
    - respiracion: acorta la nota que cruza el límite de frase para abrir
                   un hueco de ~0.3 s
    """
    scale = _PROFILE_SCALE.get(profile, 1.0)
    overrides = instrument_overrides or {}
    mid = read_midi(path)
    tc = TimeContext(mid)
    stats = {"rango": 0, "polifonia": 0, "respiracion": 0}

    for idx, trk in enumerate(mid.tracks):
        notes = extract_notes(trk)
        if not notes:
            continue
        inst = detect_instrument(trk, overrides.get(idx))
        prof = INSTRUMENT_DB[inst]
        lo, hi = prof["range"]

        # [1] rango → transposición de octavas
        for n in notes:
            orig = n.pitch
            while n.pitch < lo:
                n.pitch += 12
            while n.pitch > hi:
                n.pitch -= 12
            if lo <= n.pitch <= hi and n.pitch != orig:
                stats["rango"] += 1
        # colapsar duplicados exactos creados por la transposición
        seen = set()
        dedup = []
        for n in notes:
            key = (n.start, n.pitch)
            if key in seen:
                stats["polifonia"] += 1
                continue
            seen.add(key)
            dedup.append(n)
        notes = dedup

        # [2] monofonía → recortar solapes (gana la nota que entra)
        if prof["mono"]:
            notes.sort(key=lambda n: (n.start, -n.pitch))
            kept = []
            for n in notes:
                clipped = False
                for p in kept:
                    if p.end > n.start:
                        if p.start >= n.start:        # arranque simultáneo: descarta la grave
                            clipped = True
                            stats["polifonia"] += 1
                            break
                        p.end = n.start               # legato: recorta la anterior
                        stats["polifonia"] += 1
                if not clipped:
                    kept.append(n)
            notes = kept
        elif prof["max_poly"] < 10:
            # polifonía limitada → al exceder max_poly se descarta la nota
            # entrante más débil (menor velocidad; a igualdad, la más grave)
            notes.sort(key=lambda n: (n.start, -n.vel, -n.pitch))
            kept = []
            for n in notes:
                active = [p for p in kept if p.end > n.start]
                if len(active) >= prof["max_poly"]:
                    stats["polifonia"] += 1
                    continue
                kept.append(n)
            notes = kept

        # [3] respiración → abrir huecos
        if prof["breath"] and notes:
            limit = prof["max_breath"] * scale
            us_per_beat = mid.tempo_map[0][1]                  # tempo inicial
            gap_ticks = max(1, int(0.3 * 1e6 / us_per_beat * mid.tpb))
            phrase_start = tc.sec(notes[0].start)
            prev = notes[0]
            for n in notes[1:]:
                s = tc.sec(n.start)
                if s - tc.sec(prev.end) >= 0.25:
                    phrase_start = s
                elif s - phrase_start > limit:
                    new_end = max(prev.start + 1, n.start - gap_ticks)
                    if new_end < prev.end:
                        prev.end = new_end
                        stats["respiracion"] += 1
                    phrase_start = s
                if n.end > prev.end or n.start >= prev.end:
                    prev = n

        # reconstruir eventos de nota de la pista
        others = [e for e in trk.events if e.kind not in ("note_on", "note_off")]
        nevents = []
        for n in notes:
            nevents.append(MidiEvent(abs=n.start, kind="note_on",
                                     channel=n.channel, pitch=n.pitch, vel=n.vel))
            nevents.append(MidiEvent(abs=n.end, kind="note_off",
                                     channel=n.channel, pitch=n.pitch, vel=0))
        trk.events = sorted(others + nevents,
                            key=lambda e: (e.abs, 0 if e.kind == "note_off" else 1))

    write_midi(mid, out_path)
    return stats


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME POR CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def _score_label(s: float) -> str:
    if s >= 0.95: return "excelente"
    if s >= 0.85: return "tocable"
    if s >= 0.65: return "exigente"
    if s >= 0.40: return "problemática"
    return "intocable"


def _score_color(s: float) -> str:
    if s >= 0.85: return _c("GRN")
    if s >= 0.65: return _c("YEL")
    return _c("RED")


def print_report(report: AuditReport, max_issues: int = 12,
                 verbose: bool = False, quiet: bool = False):
    B, R, G = _c("B"), _c("R"), _c("G")
    if quiet:
        for t in report.tracks:
            ne, nw = len(t.errors()), len(t.warnings())
            print(f"[{t.index}] {t.name:<22} {t.instrument_es:<22} "
                  f"score={t.score:.2f}  errores={ne}  avisos={nw}")
        print(f"GLOBAL score={report.global_score:.2f} "
              f"errores={len(report.errors())} avisos={len(report.warnings())}")
        return

    print(f"\n{'═' * 78}")
    print(f"  {B}PLAYABILITY AUDITOR v{VERSION}  ·  {report.file}{R}")
    print(f"  Perfil de exigencia: {report.profile}")
    print(f"{'═' * 78}")

    for t in report.tracks:
        ne, nw = len(t.errors()), len(t.warnings())
        col = _score_color(t.score)
        print(f"\n  {B}[{t.index}] {t.name}{R}  ·  {t.instrument_es}  ·  "
              f"{t.n_notes} notas")
        print(f"      tocabilidad: {col}{t.score:.2f} ({_score_label(t.score)}){R}"
              f"   errores: {ne}   avisos: {nw}")
        shown = t.issues if verbose else t.issues[:max_issues]
        for it in shown:
            mark = f"{_c('RED')}✖" if it.severity == "ERROR" else f"{_c('YEL')}⚑"
            times = f" ×{it.count}" if it.count > 1 else ""
            print(f"      {mark} [{it.check}] c.{it.bar}: {it.msg}{times}{R}")
        if not verbose and len(t.issues) > max_issues:
            print(f"      {G}... y {len(t.issues) - max_issues} incidencias más "
                  f"(usa --verbose){R}")
        if not t.issues:
            print(f"      {_c('GRN')}✔ sin incidencias{R}")

    col = _score_color(report.global_score)
    print(f"\n{'─' * 78}")
    print(f"  {B}GLOBAL{R}  tocabilidad: {col}{report.global_score:.2f} "
          f"({_score_label(report.global_score)}){R}   "
          f"errores: {len(report.errors())}   avisos: {len(report.warnings())}")
    if report.fixed_path:
        print(f"  MIDI corregido: {report.fixed_path}")
    print(f"{'═' * 78}\n")


def report_to_dict(report: AuditReport) -> dict:
    return {
        "tool": "playability_auditor",
        "version": VERSION,
        "file": report.file,
        "profile": report.profile,
        "global_score": report.global_score,
        "fixed_path": report.fixed_path,
        "tracks": [
            {
                "index": t.index, "name": t.name,
                "instrument": t.instrument, "instrument_es": t.instrument_es,
                "n_notes": t.n_notes, "score": t.score,
                "issues": [asdict(i) for i in t.issues],
            } for t in report.tracks
        ],
    }


def list_instruments():
    B, R = _c("B"), _c("R")
    print(f"\n{B}Catálogo de perfiles instrumentales{R}")
    print(f"{'─' * 78}")
    print(f"  {'clave':<14} {'instrumento':<26} {'rango':<12} {'cómodo':<12} "
          f"{'mono':<5} {'resp.':<6} {'nps'}")
    for key, p in INSTRUMENT_DB.items():
        rng = f"{pitch_name(p['range'][0])}–{pitch_name(p['range'][1])}"
        sw = f"{pitch_name(p['sweet'][0])}–{pitch_name(p['sweet'][1])}"
        br = f"{p['max_breath']}s" if p["breath"] else "—"
        print(f"  {key:<14} {p['es']:<26} {rng:<12} {sw:<12} "
              f"{'sí' if p['mono'] else 'no':<5} {br:<6} {p['max_nps']}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_scope(s: str) -> Tuple[int, int]:
    m = re.fullmatch(r"(\d+):(\d+)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("scope debe ser S:E (ej: 8:24)")
    a, b = int(m.group(1)), int(m.group(2))
    if a < 1 or b < a:
        raise argparse.ArgumentTypeError("scope inválido: requiere 1 <= S <= E")
    return a, b


def _parse_override(s: str) -> Tuple[int, str]:
    m = re.fullmatch(r"(\d+)=([a-z_]+)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("formato: PISTA=instrumento (ej: 2=violin)")
    return int(m.group(1)), m.group(2)


def main():
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="playability_auditor.py",
        description="Auditoría de tocabilidad e idiomática instrumental para MIDIs.")
    ap.add_argument("midi", nargs="*", help="Archivo(s) MIDI a auditar")
    ap.add_argument("--tracks", type=int, nargs="+", help="Auditar solo estas pistas")
    ap.add_argument("--scope", type=_parse_scope, help="Limitar a compases S:E")
    ap.add_argument("--skip", nargs="+", choices=ALL_CHECKS, default=[],
                    help="Omitir comprobaciones")
    ap.add_argument("--only", nargs="+", choices=ALL_CHECKS,
                    help="Ejecutar solo estas comprobaciones")
    ap.add_argument("--profile", choices=sorted(_PROFILE_SCALE), default="estandar",
                    help="Perfil de exigencia (default: estandar)")
    ap.add_argument("--instrument", type=_parse_override, nargs="+", default=[],
                    metavar="T=I", help="Forzar instrumento de una pista (ej: 2=violin)")
    ap.add_argument("--fix", action="store_true", help="Generar MIDI corregido")
    ap.add_argument("--output", help="Ruta del MIDI corregido")
    ap.add_argument("--json", dest="json_path", help="Exportar informe a JSON")
    ap.add_argument("--max-issues", type=int, default=12,
                    help="Máx. incidencias detalladas por pista (default: 12)")
    ap.add_argument("--quiet", action="store_true", help="Solo resumen por pista")
    ap.add_argument("--verbose", action="store_true", help="Mostrar todas las incidencias")
    ap.add_argument("--no-color", action="store_true", help="Desactivar colores ANSI")
    ap.add_argument("--list-instruments", action="store_true",
                    help="Mostrar el catálogo de perfiles y salir")
    args = ap.parse_args()

    if args.no_color:
        _USE_COLOR = False
    if args.list_instruments:
        list_instruments()
        return 0
    if not args.midi:
        ap.error("indica al menos un archivo MIDI (o usa --list-instruments)")

    checks = args.only if args.only else [c for c in ALL_CHECKS if c not in args.skip]
    overrides = dict(args.instrument)

    any_error = False
    json_reports = []
    for path in args.midi:
        if not Path(path).exists():
            print(f"[ERROR] no existe: {path}", file=sys.stderr)
            any_error = True
            continue
        try:
            report = audit_midi(path, profile=args.profile, tracks=args.tracks,
                                scope=args.scope, checks=checks,
                                instrument_overrides=overrides)
        except Exception as e:
            print(f"[ERROR] {path}: {e}", file=sys.stderr)
            any_error = True
            continue

        if args.fix:
            out = args.output or str(Path(path).with_name(Path(path).stem + "_fixed.mid"))
            stats = fix_midi(path, out, profile=args.profile,
                             instrument_overrides=overrides)
            report.fixed_path = out
            if not args.quiet:
                print(f"\n  Correcciones aplicadas → {out}")
                print(f"    rango: {stats['rango']} notas transpuestas · "
                      f"polifonía: {stats['polifonia']} solapes resueltos · "
                      f"respiración: {stats['respiracion']} huecos abiertos")

        print_report(report, max_issues=args.max_issues,
                     verbose=args.verbose, quiet=args.quiet)
        json_reports.append(report_to_dict(report))
        if report.errors():
            any_error = True

    if args.json_path and json_reports:
        payload = json_reports[0] if len(json_reports) == 1 else json_reports
        Path(args.json_path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"  Informe JSON: {args.json_path}\n")

    return 1 if any_error else 0


if __name__ == "__main__":
    sys.exit(main())
