#!/usr/bin/env python3
"""
transcription_refiner.py — Transcription Refiner: refina un MIDI transcrito
contrastándolo con el audio (dinámicas, voces duplicadas, notas sin sustento)
— fichero único
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
  QUÉ HACE
    Los transcriptores audio → MIDI (muscriptor_transcribe, pitch_transcriber)
    dejan tres defectos típicos que esta herramienta corrige mirando de nuevo
    el audio, sin reentrenar nada:

      1. Dinámicas planas. MuScriptor solo emite inicio y fin de nota, así que
         todas salen con velocity 100. `dynamics` estima una velocidad por
         nota a partir de la energía de su fundamental en el audio (o en el
         stem de su instrumento).
      2. Voces duplicadas. Con varios instrumentos de la misma familia
         (violín + viola) el modelo tiende a copiar una línea en la otra a
         la octava o al unísono. `analyze` mide cuántas notas de cada pista
         son copia de otra pista; `prune --duplicates-only` elimina las copias
         que además no tienen energía real en el audio (las copias legítimas,
         p. ej. un doblaje en octavas realmente tocado, se conservan).
      3. Notas fantasma. `prune` elimina las notas cuyo fundamental no tiene
         sustento acústico.

    Opera sobre el MIDI original conservando tempo, compás, programas y todo
    lo que no toca; el resultado es un MIDI nuevo.

  SOPORTE ACÚSTICO (ratio 0-1)
    Energía CQT del fundamental de la nota (±1/3 de semitono) durante su
    duración, dividida por el máximo del fotograma. Solo se mide el
    fundamental, a propósito: un doblaje a la octava inferior alucinado tiene
    ese fundamental casi vacío (en las pruebas, mediana 1,7 % de la del violín)
    aunque el armónico de la nota superior caiga justo encima. Limitación: un
    instrumento grave con fundamental débil (piano en el registro más bajo)
    puede puntuar bajo; usa --stem o limita con --parts.

  ESTIMACIÓN DE VELOCIDAD
    · Piano y percusivos (programas GM < 16): pico de energía en los primeros
      150 ms (el ataque manda).
    · Resto: energía media durante la nota.
    · En dB, normalizada POR PISTA: el percentil bajo (--percentiles, default
      5) → --vel-min y el alto (95) → --vel-max, recortando. El equilibrio
      entre pistas se ajusta con --offset PISTA=N.
    · --smooth SEC aplica mediana móvil entre notas vecinas de la pista.
    La estimación sobre una mezcla es más ruidosa que sobre un stem; con
    stems separados (stem_separator.py) mejora claramente.

  COMANDOS:
    analyze     informe: soporte, velocidades y duplicados por pista
    prune       elimina notas sin sustento (y/o duplicadas) → MIDI
    dynamics    estima velocidades desde el audio → MIDI
    refine      prune + dynamics en una pasada → MIDI

  PISTAS
    Una «pista» es cada pista del MIDI con notas (tipo 1) o cada canal si todo
    está en una sola pista. Se designan por índice o por parte del nombre
    (sin distinguir mayúsculas): --parts viola, --stem Violin=stem.wav.

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  analyze
    audio AUDIO                 audio de referencia (mezcla o stem)
    midi MIDI                   MIDI transcrito
    --stem PISTA=WAV […]        stem propio para una pista
    --threshold RATIO           soporte mínimo para dar una nota por real
                                (default: 0.10)
    --tol SEC                   tolerancia de inicio para detectar copias
                                (default: 0.08)
    --sr SR                     frecuencia de análisis (default: 22050)
    --json                      salida JSON

  prune
    audio AUDIO                 audio de referencia (mezcla o stem)
    midi MIDI                   MIDI transcrito
    --output FILE               MIDI de salida (default: <midi>_pruned.mid)
    --stem PISTA=WAV […]        stem propio para una pista
    --parts PISTA […]           pistas a limpiar (default: todas)
    --threshold RATIO           soporte mínimo (default: 0.10)
    --duplicates-only           solo elimina notas que son copia (unísono u
                                octava) de otra pista Y no tienen soporte
    --tol SEC                   tolerancia de inicio para copias (default: 0.08)
    --sr SR                     frecuencia de análisis (default: 22050)
    --dry-run                   informa sin escribir el MIDI
    --report-json FILE          guarda el informe en JSON

  dynamics
    audio AUDIO                 audio de referencia (mezcla o stem)
    midi MIDI                   MIDI transcrito
    --output FILE               MIDI de salida (default: <midi>_dyn.mid)
    --stem PISTA=WAV […]        stem propio para una pista
    --parts PISTA […]           pistas a las que se aplica (default: todas)
    --vel-min N                 velocidad del percentil bajo (default: 45)
    --vel-max N                 velocidad del percentil alto (default: 110)
    --percentiles LO,HI         percentiles de normalización (default: 5,95)
    --smooth SEC                mediana móvil entre notas vecinas (default: 0)
    --offset PISTA=N […]        suma N a la velocidad de esa pista
    --sr SR                     frecuencia de análisis (default: 22050)

  refine
    (acepta las opciones de prune y de dynamics; primero limpia y después
    estima las velocidades de las notas que quedan)

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
Una línea por parámetro: <nombre> <tipo> [clave=valor …] [# descripción].
Tipos: int float str bool enum file dir list kv. Claves: default min max typical
unit values of io ext format required advanced when excludes alias. Sin min/max =
sin límite conocido. Se verifica contra el argparse con «suite_index.py lint
--check spec-vs-code».

@params v1
[analyze]
audio         file  io=in required  # audio de referencia
midi          file  io=in ext=mid,midi required  # MIDI transcrito
--stem        list  of=str  # PISTA=WAV: stem propio de una pista
--threshold   float default=0.10 min=0 max=1  # soporte mínimo
--tol         float default=0.08 min=0.005 unit=s  # tolerancia de inicio para copias
--sr          int   default=22050 min=8000 unit=Hz  # frecuencia de análisis
--json        bool  default=false  # salida JSON
[prune]
audio         file  io=in required  # audio de referencia
midi          file  io=in ext=mid,midi required  # MIDI transcrito
--output      file  io=out ext=mid  # MIDI de salida
--stem        list  of=str  # PISTA=WAV: stem propio de una pista
--parts       list  of=str  # pistas a limpiar
--threshold   float default=0.10 min=0 max=1  # soporte mínimo
--duplicates-only bool default=false  # solo copias sin soporte
--tol         float default=0.08 min=0.005 unit=s  # tolerancia de inicio para copias
--sr          int   default=22050 min=8000 unit=Hz  # frecuencia de análisis
--dry-run     bool  default=false  # informa sin escribir
--report-json file  io=out ext=json  # informe JSON
[dynamics]
audio         file  io=in required  # audio de referencia
midi          file  io=in ext=mid,midi required  # MIDI transcrito
--output      file  io=out ext=mid  # MIDI de salida
--stem        list  of=str  # PISTA=WAV: stem propio de una pista
--parts       list  of=str  # pistas a las que se aplica
--vel-min     int   default=45 min=1 max=127  # velocidad del percentil bajo
--vel-max     int   default=110 min=1 max=127  # velocidad del percentil alto
--percentiles str   default=5,95  # percentiles LO,HI de normalización
--smooth      float default=0 min=0 unit=s  # mediana móvil entre notas vecinas
--offset      list  of=str  # PISTA=N: suma N a la velocidad
--sr          int   default=22050 min=8000 unit=Hz  # frecuencia de análisis
[refine]
audio         file  io=in required  # audio de referencia
midi          file  io=in ext=mid,midi required  # MIDI transcrito
--output      file  io=out ext=mid  # MIDI de salida
--stem        list  of=str  # PISTA=WAV: stem propio de una pista
--parts       list  of=str  # pistas a las que se aplica
--threshold   float default=0.10 min=0 max=1  # soporte mínimo
--duplicates-only bool default=false  # solo copias sin soporte
--tol         float default=0.08 min=0.005 unit=s  # tolerancia de inicio para copias
--vel-min     int   default=45 min=1 max=127  # velocidad del percentil bajo
--vel-max     int   default=110 min=1 max=127  # velocidad del percentil alto
--percentiles str   default=5,95  # percentiles LO,HI de normalización
--smooth      float default=0 min=0 unit=s  # mediana móvil entre notas vecinas
--offset      list  of=str  # PISTA=N: suma N a la velocidad
--sr          int   default=22050 min=8000 unit=Hz  # frecuencia de análisis
--dry-run     bool  default=false  # informa sin escribir
--report-json file  io=out ext=json  # informe JSON
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python transcription_refiner.py analyze original.mp3 transcripcion.mid
  python transcription_refiner.py prune original.mp3 transcripcion.mid --duplicates-only
  python transcription_refiner.py prune original.mp3 t.mid --parts viola --threshold 0.15
  python transcription_refiner.py dynamics original.mp3 t.mid --output t_dyn.mid
  python transcription_refiner.py dynamics mezcla.wav t.mid --stem piano=stem_piano.wav --stem violin=stem_other.wav
  python transcription_refiner.py refine original.mp3 t.mid --duplicates-only --smooth 1.0 --output final.mid

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  numpy, librosa, soundfile, mido
"""

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
import tempfile

try:
    import numpy as np
    import librosa
    import soundfile as sf
    import mido
except ImportError as _e:
    print(f"ERROR: falta una dependencia ({_e}). pip install numpy librosa soundfile mido")
    sys.exit(1)

DRUM_CHANNEL = 9
CQT_FMIN_MIDI = 24  # C1
CQT_HOP = 512
CQT_BPO = 36
CQT_OCT = 7
ATTACK_S = 0.15
COPY_INTERVALS = {0: "unisono", -12: "octava_abajo", 12: "octava_arriba",
                  -24: "octava_abajo", 24: "octava_arriba"}


def _ensure_parent(path):
    """Contrato de la suite: toda salida crea su directorio padre.
    Devuelve `path` sin modificar (solo actúa sobre rutas, no sobre ficheros)."""
    import os as _os
    if isinstance(path, (str, bytes, _os.PathLike)):
        d = _os.path.dirname(_os.path.abspath(_os.fsdecode(path)))
        if d:
            _os.makedirs(d, exist_ok=True)
    return path


def _info(msg):
    print(msg, file=sys.stderr)


# ════════════════════════════════════════════════════════════════════════════
#  MIDI EDITABLE: notas con referencia a sus mensajes
# ════════════════════════════════════════════════════════════════════════════

class EditableMidi:
    """Carga un MIDI conservando los mensajes en ticks absolutos para poder
    cambiar velocidades o eliminar notas sin tocar nada más."""

    def __init__(self, path):
        self.path = path
        self.mid = mido.MidiFile(str(path))
        tpb = self.mid.ticks_per_beat
        tempos = []
        self.tracks = []  # por pista: lista de [abs_tick, msg, alive]
        for tr in self.mid.tracks:
            t, ev = 0, []
            for m in tr:
                t += m.time
                ev.append([t, m, True])
                if m.type == "set_tempo":
                    tempos.append((t, m.tempo))
            self.tracks.append(ev)
        tempos.sort()
        if not tempos or tempos[0][0] != 0:
            tempos.insert(0, (0, 500000))
        cum = [0.0]
        for i in range(1, len(tempos)):
            cum.append(cum[-1] + mido.tick2second(tempos[i][0] - tempos[i - 1][0], tpb, tempos[i - 1][1]))
        self._ticks = [t for t, _ in tempos]
        self._cum, self._tempos, self._tpb = cum, tempos, tpb

        raw, had_notes = [], 0
        names = {}
        for ti, ev in enumerate(self.tracks):
            prog, open_, name, had = {}, {}, None, False
            for ei, (t, m, _) in enumerate(ev):
                if m.type == "track_name":
                    name = m.name
                elif m.type == "program_change":
                    prog[m.channel] = m.program
                elif m.type == "note_on" and m.velocity > 0:
                    open_.setdefault((m.channel, m.note), []).append((ei, t, prog.get(m.channel, 0)))
                elif m.type in ("note_off", "note_on"):
                    k = (m.channel, m.note)
                    if open_.get(k):
                        on_i, t0, pg = open_[k].pop(0)
                        raw.append((ti, m.channel, m.note, on_i, ei, t0, t, pg, name))
                        had = True
            had_notes += had
            names[ti] = name
        self.notes, self.parts = [], {}
        for ti, ch, pitch, on_i, off_i, t0, t1, pg, nm in raw:
            pid = ti if had_notes > 1 else ch
            self.parts.setdefault(pid, nm or (f"ch{ch}" if had_notes <= 1 else f"track{ti}"))
            s0, s1 = self.sec(t0), self.sec(t1)
            self.notes.append(dict(part=pid, track=ti, on=on_i, off=off_i, pitch=pitch,
                                   t0=s0, t1=max(s1, s0 + 0.01), vel=self.tracks[ti][on_i][1].velocity,
                                   program=pg, drum=(ch == DRUM_CHANNEL), keep=True, new_vel=None))
        self.notes.sort(key=lambda n: (n["t0"], n["pitch"]))

    def sec(self, tick):
        i = bisect.bisect_right(self._ticks, tick) - 1
        return self._cum[i] + mido.tick2second(tick - self._ticks[i], self._tpb, self._tempos[i][1])

    def resolve_parts(self, keys):
        """Lista de ids de pista a partir de índices o fragmentos de nombre."""
        if not keys:
            return list(self.parts)
        out = []
        for k in keys:
            hit = [pid for pid, nm in self.parts.items()
                   if k == str(pid) or k.lower() in (nm or "").lower()]
            if not hit:
                raise SystemExit(f"Pista «{k}» no encontrada. Disponibles: "
                                 + ", ".join(f"{p}={n}" for p, n in self.parts.items()))
            out += [h for h in hit if h not in out]
        return out

    def save(self, path):
        mid = mido.MidiFile(type=self.mid.type, ticks_per_beat=self.mid.ticks_per_beat)
        dead = {(n["track"], n["on"]) for n in self.notes if not n["keep"]}
        dead |= {(n["track"], n["off"]) for n in self.notes if not n["keep"]}
        newvel = {(n["track"], n["on"]): n["new_vel"] for n in self.notes
                  if n["keep"] and n["new_vel"] is not None}
        for ti, ev in enumerate(self.tracks):
            tr = mido.MidiTrack()
            last = 0
            for ei, (t, m, _) in enumerate(ev):
                if (ti, ei) in dead:
                    continue
                m2 = m.copy(velocity=newvel[(ti, ei)]) if (ti, ei) in newvel else m.copy()
                m2.time = t - last
                last = t
                tr.append(m2)
            mid.tracks.append(tr)
        mid.save(_ensure_parent(str(path)))


# ════════════════════════════════════════════════════════════════════════════
#  AUDIO / SOPORTE ACÚSTICO
# ════════════════════════════════════════════════════════════════════════════

def load_audio(path, sr):
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
        return y
    except Exception:
        ff = shutil.which("ffmpeg")
        if not ff:
            raise
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "a.wav")
            subprocess.run([ff, "-y", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), wav],
                           check=True)
            y, _ = sf.read(wav, dtype="float32")
        return y


def cqt_mag(y, sr):
    return np.abs(librosa.cqt(y, sr=sr, hop_length=CQT_HOP, fmin=librosa.midi_to_hz(CQT_FMIN_MIDI),
                              n_bins=CQT_BPO * CQT_OCT, bins_per_octave=CQT_BPO)).astype(np.float32)


class Reference:
    """Audio de referencia por pista (mezcla o stems), con CQT en caché."""

    def __init__(self, audio, stems, sr):
        self.audio, self.stems, self.sr, self._c = audio, stems, sr, {}

    def path_for(self, pid, name):
        for key, wav in self.stems:
            if key == str(pid) or key.lower() in (name or "").lower():
                return wav
        return self.audio

    def cqt(self, pid, name):
        p = self.path_for(pid, name)
        if p not in self._c:
            _info(f"  CQT de {os.path.basename(p)} …")
            self._c[p] = cqt_mag(load_audio(p, self.sr), self.sr)
        return self._c[p]


def note_energy(C, sr, pitch, t0, t1, attack=False):
    """(energía del fundamental, referencia) sobre la nota; None si fuera de rango.
    attack=True usa solo los primeros ATTACK_S y el pico en lugar de la media."""
    center = int(round((pitch - CQT_FMIN_MIDI) * CQT_BPO / 12))
    if center < 1 or center >= C.shape[0] - 1:
        return None
    dt = CQT_HOP / sr
    f0 = max(0, int(t0 / dt))
    f1 = max(f0 + 1, int(t1 / dt))
    if attack:
        f1 = max(f0 + 1, min(f1, f0 + int(ATTACK_S / dt) + 1))
    seg = C[:, f0:f1]
    if seg.shape[1] == 0:
        return None
    band = seg[center - 1:center + 2, :].max(axis=0)
    ref = seg.max(axis=0)
    if attack:
        return float(band.max()), float(ref.max())
    return float(band.mean()), float(ref.mean())


def support_ratio(C, sr, n):
    r = note_energy(C, sr, n["pitch"], n["t0"], n["t1"])
    if r is None:
        return None
    e, ref = r
    return e / ref if ref > 0 else 0.0


def parse_pairs(pairs, what):
    out = []
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"{what} espera PISTA=VALOR, recibido «{p}»")
        k, v = p.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


# ════════════════════════════════════════════════════════════════════════════
#  DUPLICADOS ENTRE PISTAS
# ════════════════════════════════════════════════════════════════════════════

def classify_copies(notes, tol):
    """Para cada nota b marca, si existe, la nota a de OTRA pista que empieza a
    ±tol, se solapa con ella y cuyo intervalo es unísono u octava.
    Devuelve {id(b): (id_pista_original, tipo)}."""
    live = [n for n in notes if not n["drum"]]
    starts = [n["t0"] for n in live]
    out = {}
    for b in live:
        lo = bisect.bisect_left(starts, b["t0"] - tol)
        hi = bisect.bisect_right(starts, b["t0"] + tol)
        for a in live[lo:hi]:
            if a["part"] == b["part"]:
                continue
            iv = b["pitch"] - a["pitch"]
            if iv in COPY_INTERVALS and min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]) > 0.5 * min(
                    a["t1"] - a["t0"], b["t1"] - b["t0"]):
                out[id(b)] = (a["part"], COPY_INTERVALS[iv])
                break
    return out


# ════════════════════════════════════════════════════════════════════════════
#  OPERACIONES
# ════════════════════════════════════════════════════════════════════════════

def op_analyze(em, ref, threshold, tol):
    copies = classify_copies(em.notes, tol)
    res = []
    for pid, name in em.parts.items():
        ns = [n for n in em.notes if n["part"] == pid and not n["drum"]]
        C = ref.cqt(pid, name)
        ratios = [support_ratio(C, ref.sr, n) for n in ns]
        valid = [r for r in ratios if r is not None]
        vels = [n["vel"] for n in ns]
        dup = {}
        for n, r in zip(ns, ratios):
            c = copies.get(id(n))
            if c:
                d = dup.setdefault(c[0], dict(unisono=0, octava_abajo=0, octava_arriba=0, sin_soporte=0))
                d[c[1]] += 1
                if r is not None and r < threshold:
                    d["sin_soporte"] += 1
        res.append(dict(
            part=pid, name=name, n_notes=len(ns), n_measured=len(valid),
            support=round(float(np.mean([r >= threshold for r in valid])), 4) if valid else None,
            ratio_median=round(float(np.median(valid)), 4) if valid else None,
            velocity=dict(min=min(vels, default=None), max=max(vels, default=None),
                          distinct=len(set(vels))),
            copies_of={str(k): v for k, v in dup.items()},
            copies_total=sum(sum(v[t] for t in ("unisono", "octava_abajo", "octava_arriba"))
                             for v in dup.values()),
        ))
    return res


def op_prune(em, ref, parts, threshold, duplicates_only, tol):
    copies = classify_copies(em.notes, tol) if duplicates_only else {}
    stats = {pid: dict(name=nm, before=0, removed=0) for pid, nm in em.parts.items() if pid in parts}
    for pid in stats:
        C = ref.cqt(pid, em.parts[pid])
        for n in em.notes:
            if n["part"] != pid or n["drum"] or not n["keep"]:
                continue
            stats[pid]["before"] += 1
            r = support_ratio(C, ref.sr, n)
            if r is None or r >= threshold:
                continue
            if duplicates_only and id(n) not in copies:
                continue
            n["keep"] = False
            stats[pid]["removed"] += 1
    return stats


def op_dynamics(em, ref, parts, vmin, vmax, pcts, smooth, offsets):
    lo_p, hi_p = pcts
    stats = {}
    for pid in parts:
        name = em.parts[pid]
        C = ref.cqt(pid, name)
        ns = [n for n in em.notes if n["part"] == pid and not n["drum"] and n["keep"]]
        db, idx = [], []
        for i, n in enumerate(ns):
            r = note_energy(C, ref.sr, n["pitch"], n["t0"], n["t1"], attack=n["program"] < 16)
            if r is None:
                continue
            db.append(20 * np.log10(max(r[0], 1e-6)))
            idx.append(i)
        if len(db) < 2:
            stats[pid] = dict(name=name, notes=len(ns), measured=len(db))
            continue
        db = np.asarray(db)
        lo, hi = np.percentile(db, [lo_p, hi_p])
        span = max(hi - lo, 1e-6)
        vel = np.clip(vmin + (db - lo) / span * (vmax - vmin), vmin, vmax)  # fuera de percentiles: se satura
        if smooth > 0:
            t = np.asarray([ns[i]["t0"] for i in idx])
            sm = vel.copy()
            for k in range(len(vel)):
                m = np.abs(t - t[k]) <= smooth / 2
                sm[k] = np.median(vel[m])
            vel = sm
        off = sum(int(v) for key, v in offsets if key == str(pid) or key.lower() in (name or "").lower())
        for i, v in zip(idx, vel):
            ns[i]["new_vel"] = int(np.clip(round(v) + off, 1, 127))
        stats[pid] = dict(name=name, notes=len(ns), measured=len(idx),
                          db_low=round(float(lo), 1), db_high=round(float(hi), 1),
                          vel_min=int(vel.min()), vel_max=int(vel.max()), offset=off)
    return stats


# ════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ════════════════════════════════════════════════════════════════════════════

def _setup(args):
    em = EditableMidi(args.midi)
    ref = Reference(args.audio, parse_pairs(args.stem, "--stem"), args.sr)
    return em, ref


def _default_out(args, suffix):
    return args.output or os.path.splitext(args.midi)[0] + suffix


def _pcts(s):
    try:
        lo, hi = (float(x) for x in s.split(","))
        assert 0 <= lo < hi <= 100
        return lo, hi
    except Exception:
        raise SystemExit("--percentiles espera LO,HI con 0 ≤ LO < HI ≤ 100 (ej. 5,95)")


def _print_prune(stats):
    for pid, s in stats.items():
        print(f"  [{pid}] {str(s['name']):<14} {s['before']:>5} notas → quita {s['removed']:>4}")


def _print_dyn(stats):
    for pid, s in stats.items():
        if "vel_min" in s:
            print(f"  [{pid}] {str(s['name']):<14} {s['measured']:>5}/{s['notes']} notas medidas  "
                  f"vel {s['vel_min']}–{s['vel_max']}  (dB {s['db_low']}…{s['db_high']})")
        else:
            print(f"  [{pid}] {str(s['name']):<14} datos insuficientes: no se tocan sus velocidades")


def _write_report(path, data):
    if path:
        with open(_ensure_parent(path), "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)


def cmd_analyze(args):
    em, ref = _setup(args)
    res = op_analyze(em, ref, args.threshold, args.tol)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    print(f"Análisis de {args.midi} (soporte: ratio ≥ {args.threshold})")
    for r in res:
        v = r["velocity"]
        flat = "  ← velocidad constante" if v["distinct"] == 1 else ""
        print(f"  [{r['part']}] {str(r['name']):<14} notas {r['n_notes']:>5}  soporte {r['support']}  "
              f"ratio med {r['ratio_median']}  vel {v['min']}–{v['max']}{flat}")
        for src, d in r["copies_of"].items():
            tot = d["unisono"] + d["octava_abajo"] + d["octava_arriba"]
            pct = 100 * tot / max(1, r["n_notes"])
            print(f"        copia de pista {src}: {tot} notas ({pct:.0f} %)  "
                  f"unísono {d['unisono']}, 8ª abajo {d['octava_abajo']}, 8ª arriba {d['octava_arriba']}"
                  f"  · sin soporte {d['sin_soporte']}")
    return 0


def cmd_prune(args):
    em, ref = _setup(args)
    parts = em.resolve_parts(args.parts)
    stats = op_prune(em, ref, parts, args.threshold, args.duplicates_only, args.tol)
    print("Poda" + (" (solo copias sin soporte)" if args.duplicates_only else "")
          + f", umbral {args.threshold}:")
    _print_prune(stats)
    _write_report(args.report_json, dict(prune=stats))
    if args.dry_run:
        print("(--dry-run: no se escribe MIDI)")
        return 0
    out = _default_out(args, "_pruned.mid")
    em.save(out)
    print(f"→ {out}")
    return 0


def cmd_dynamics(args):
    em, ref = _setup(args)
    parts = em.resolve_parts(args.parts)
    stats = op_dynamics(em, ref, parts, args.vel_min, args.vel_max, _pcts(args.percentiles),
                        args.smooth, parse_pairs(args.offset, "--offset"))
    print("Dinámicas estimadas:")
    _print_dyn(stats)
    out = _default_out(args, "_dyn.mid")
    em.save(out)
    print(f"→ {out}")
    return 0


def cmd_refine(args):
    em, ref = _setup(args)
    parts = em.resolve_parts(args.parts)
    ps = op_prune(em, ref, parts, args.threshold, args.duplicates_only, args.tol)
    print("Poda" + (" (solo copias sin soporte)" if args.duplicates_only else "")
          + f", umbral {args.threshold}:")
    _print_prune(ps)
    ds = op_dynamics(em, ref, parts, args.vel_min, args.vel_max, _pcts(args.percentiles),
                     args.smooth, parse_pairs(args.offset, "--offset"))
    print("Dinámicas estimadas:")
    _print_dyn(ds)
    _write_report(args.report_json, dict(prune=ps, dynamics=ds))
    if args.dry_run:
        print("(--dry-run: no se escribe MIDI)")
        return 0
    out = _default_out(args, "_refined.mid")
    em.save(out)
    print(f"→ {out}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="transcription_refiner.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_an = sub.add_parser("analyze", help="soporte, velocidades y duplicados por pista")
    p_an.add_argument("audio", help="audio de referencia (mezcla o stem)")
    p_an.add_argument("midi", help="MIDI transcrito")
    p_an.add_argument("--stem", nargs="+", default=None, help="PISTA=WAV: stem propio de una pista")
    p_an.add_argument("--threshold", type=float, default=0.10, help="soporte mínimo")
    p_an.add_argument("--tol", type=float, default=0.08, help="tolerancia de inicio para copias (s)")
    p_an.add_argument("--sr", type=int, default=22050, help="frecuencia de análisis")
    p_an.add_argument("--json", action="store_true", help="salida JSON")
    p_an.set_defaults(func=cmd_analyze)

    p_pr = sub.add_parser("prune", help="elimina notas sin sustento (y/o duplicadas)")
    p_pr.add_argument("audio", help="audio de referencia (mezcla o stem)")
    p_pr.add_argument("midi", help="MIDI transcrito")
    p_pr.add_argument("--output", default=None, help="MIDI de salida")
    p_pr.add_argument("--stem", nargs="+", default=None, help="PISTA=WAV: stem propio de una pista")
    p_pr.add_argument("--parts", nargs="+", default=None, help="pistas a limpiar")
    p_pr.add_argument("--threshold", type=float, default=0.10, help="soporte mínimo")
    p_pr.add_argument("--duplicates-only", action="store_true", help="solo copias sin soporte")
    p_pr.add_argument("--tol", type=float, default=0.08, help="tolerancia de inicio para copias (s)")
    p_pr.add_argument("--sr", type=int, default=22050, help="frecuencia de análisis")
    p_pr.add_argument("--dry-run", action="store_true", help="informa sin escribir")
    p_pr.add_argument("--report-json", default=None, help="informe JSON")
    p_pr.set_defaults(func=cmd_prune)

    p_dy = sub.add_parser("dynamics", help="estima velocidades desde el audio")
    p_dy.add_argument("audio", help="audio de referencia (mezcla o stem)")
    p_dy.add_argument("midi", help="MIDI transcrito")
    p_dy.add_argument("--output", default=None, help="MIDI de salida")
    p_dy.add_argument("--stem", nargs="+", default=None, help="PISTA=WAV: stem propio de una pista")
    p_dy.add_argument("--parts", nargs="+", default=None, help="pistas a las que se aplica")
    p_dy.add_argument("--vel-min", type=int, default=45, help="velocidad del percentil bajo")
    p_dy.add_argument("--vel-max", type=int, default=110, help="velocidad del percentil alto")
    p_dy.add_argument("--percentiles", default="5,95", help="percentiles LO,HI de normalización")
    p_dy.add_argument("--smooth", type=float, default=0.0, help="mediana móvil entre notas vecinas (s)")
    p_dy.add_argument("--offset", nargs="+", default=None, help="PISTA=N: suma N a la velocidad")
    p_dy.add_argument("--sr", type=int, default=22050, help="frecuencia de análisis")
    p_dy.set_defaults(func=cmd_dynamics)

    p_rf = sub.add_parser("refine", help="prune + dynamics en una pasada")
    p_rf.add_argument("audio", help="audio de referencia (mezcla o stem)")
    p_rf.add_argument("midi", help="MIDI transcrito")
    p_rf.add_argument("--output", default=None, help="MIDI de salida")
    p_rf.add_argument("--stem", nargs="+", default=None, help="PISTA=WAV: stem propio de una pista")
    p_rf.add_argument("--parts", nargs="+", default=None, help="pistas a las que se aplica")
    p_rf.add_argument("--threshold", type=float, default=0.10, help="soporte mínimo")
    p_rf.add_argument("--duplicates-only", action="store_true", help="solo copias sin soporte")
    p_rf.add_argument("--tol", type=float, default=0.08, help="tolerancia de inicio para copias (s)")
    p_rf.add_argument("--vel-min", type=int, default=45, help="velocidad del percentil bajo")
    p_rf.add_argument("--vel-max", type=int, default=110, help="velocidad del percentil alto")
    p_rf.add_argument("--percentiles", default="5,95", help="percentiles LO,HI de normalización")
    p_rf.add_argument("--smooth", type=float, default=0.0, help="mediana móvil entre notas vecinas (s)")
    p_rf.add_argument("--offset", nargs="+", default=None, help="PISTA=N: suma N a la velocidad")
    p_rf.add_argument("--sr", type=int, default=22050, help="frecuencia de análisis")
    p_rf.add_argument("--dry-run", action="store_true", help="informa sin escribir")
    p_rf.add_argument("--report-json", default=None, help="informe JSON")
    p_rf.set_defaults(func=cmd_refine)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
