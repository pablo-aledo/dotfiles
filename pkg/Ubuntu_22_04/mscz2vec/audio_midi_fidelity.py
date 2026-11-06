#!/usr/bin/env python3
"""
audio_midi_fidelity.py — Audio MIDI Fidelity: ¿cuánto se parece un MIDI al
audio del que se transcribió? — fichero único
(versión 1.0).

────────────────────────────────────────────────────────────────────────────
Fundamento
────────────────────────────────────────────────────────────────────────────
  QUÉ HACE
    Cierra el bucle de la transcripción audio → MIDI. El resto de la suite
    compara MIDI con MIDI (fidelity_meter) o puntúa audio contra un corpus
    (audio_reference_scorer); ninguna herramienta responde a «este MIDI,
    renderizado, ¿suena como el original?». Esta sí, con métricas que NO
    dependen del timbre del sintetizador:

      chroma      similitud coseno del croma (CQT) por fotograma, media y
                  mediana: ¿suenan las mismas notas (módulo octava)?
      dtw         coste DTW normalizado entre cromas: tolera pequeñas
                  derivas de tempo que el coseno fotograma a fotograma castiga
      rms         correlación de la envolvente de volumen (dB) y diferencia
                  media absoluta tras igualar la ganancia global
      onsets      F1 entre los ataques detectados en el audio y los inicios
                  de nota del MIDI (tolerancia --tol, default 70 ms)
      score       combinación 0-1 de lo anterior (pesos fijos, ver abajo)

    Se descartó el error log-mel entre espectrogramas: depende sobre todo del
    timbre del soundfont y no distingue notas correctas de incorrectas
    (en pruebas daba mejor resultado quitando notas reales).

  CÓMO RENDERIZA
    Con FluidSynth + SoundFont si están (--soundfont, o los de rutas
    habituales). Si no, con un sintetizador aditivo interno (suficiente para
    croma, DTW y ataques, no para juzgar timbre). Con --rendered se aporta un
    WAV ya renderizado y no se sintetiza nada.

  COMANDOS:
    compare     audio + uno o varios MIDI → métricas y ranking; marca las
                ventanas peor transcritas
    per-track   por pista del MIDI: qué fracción de sus notas tiene energía
                real en el audio (o en el stem de ese instrumento)
    render      MIDI → WAV (útil por sí solo y para reutilizar con --rendered)

  SCORE (aproximado):
      0.40·chroma_media + 0.20·rms_corr⁺ + 0.20·onset_F1 + 0.20·(1 − dtw)
    Sirve para COMPARAR VARIANTES de una misma transcripción (p. ej. con y sin
    viola, distintos umbrales de limpieza); no es una nota absoluta.

  PER-TRACK — SOPORTE ACÚSTICO
    Para cada nota se mide la energía CQT en su fundamental (±1/3 de
    semitono) durante su duración, relativa al máximo del fotograma
    (ratio 0-1). «soporte» = fracción de notas con ratio ≥ --support.
    Es la medida con la que se detecta que una pista (p. ej. una viola
    alucinada como copia de un violín) no tiene sustento en el audio. Solo mide
    el fundamental: en registros graves con fundamental débil (piano grave)
    puede subestimar; con --stem se acota a la fuente del instrumento.

────────────────────────────────────────────────────────────────────────────
Opciones
────────────────────────────────────────────────────────────────────────────
  compare
    audio AUDIO                 audio original (wav, mp3, flac…)
    midis MIDI […]              uno o varios MIDI candidatos
    --rendered WAV              WAV ya renderizado (solo con un MIDI)
    --soundfont SF2             SoundFont para FluidSynth
    --engine ENGINE             (defecto auto; valores: auto|fluidsynth|builtin)
    --window SEC                tamaño de ventana para el desglose
                                temporal (default: 30)
    --tol SEC                   tolerancia de ataques en segundos
                                (default: 0.07)
    --sr SR                     frecuencia de análisis (default: 22050)
    --json                      salida JSON
    --output-json FILE          guarda además el JSON en fichero

  per-track
    audio AUDIO                 audio de referencia (mezcla o stem)
    midi MIDI                   MIDI a auditar
    --stem PISTA=WAV […]        stem propio para una pista (nombre parcial
                                o índice de pista), ej. Violin=stem_other.wav
    --support RATIO             ratio mínimo para dar la nota por
                                respaldada (default: 0.05)
    --sr SR                     frecuencia de análisis (default: 22050)
    --json                      salida JSON
    --output-json FILE          guarda además el JSON en fichero

  render
    midi MIDI                   MIDI de entrada
    --output WAV                WAV de salida (default: <midi>.wav)
    --soundfont SF2             SoundFont para FluidSynth
    --engine ENGINE             (defecto auto; valores: auto|fluidsynth|builtin)
    --sr SR                     frecuencia del WAV (default: 44100)

────────────────────────────────────────────────────────────────────────────
Parámetros (spec)
────────────────────────────────────────────────────────────────────────────
Una línea por parámetro: <nombre> <tipo> [clave=valor …] [# descripción].
Tipos: int float str bool enum file dir list kv. Claves: default min max typical
unit values of io ext format required advanced when excludes alias. Sin min/max =
sin límite conocido. Se verifica contra el argparse con «suite_index.py lint
--check spec-vs-code».

@params v1
[compare]
audio         file  io=in required  # audio original
midis         list  of=file io=in required  # MIDI candidatos
--rendered    file  io=in ext=wav  # WAV ya renderizado
--soundfont   file  io=in ext=sf2,sf3  # SoundFont para FluidSynth
--engine      enum  default=auto values=auto|fluidsynth|builtin  # motor de síntesis
--window      float default=30 min=1 unit=s  # ventana del desglose temporal
--tol         float default=0.07 min=0.005 unit=s  # tolerancia de ataques
--sr          int   default=22050 min=8000 unit=Hz  # frecuencia de análisis
--json        bool  default=false  # salida JSON
--output-json file  io=out ext=json  # guarda el JSON
[per-track]
audio         file  io=in required  # audio de referencia
midi          file  io=in ext=mid,midi required  # MIDI a auditar
--stem        list  of=str  # PISTA=WAV: stem propio de una pista
--support     float default=0.05 min=0 max=1  # ratio mínimo de soporte
--sr          int   default=22050 min=8000 unit=Hz  # frecuencia de análisis
--json        bool  default=false  # salida JSON
--output-json file  io=out ext=json  # guarda el JSON
[render]
midi          file  io=in ext=mid,midi required  # MIDI de entrada
--output      file  io=out ext=wav  # WAV de salida
--soundfont   file  io=in ext=sf2,sf3  # SoundFont para FluidSynth
--engine      enum  default=auto values=auto|fluidsynth|builtin  # motor de síntesis
--sr          int   default=44100 min=8000 unit=Hz  # frecuencia del WAV
@end

────────────────────────────────────────────────────────────────────────────
Ejemplos de uso
────────────────────────────────────────────────────────────────────────────
  python audio_midi_fidelity.py compare original.mp3 reconstruido.mid
  python audio_midi_fidelity.py compare original.mp3 v1.mid v2.mid v3.mid
  python audio_midi_fidelity.py compare original.mp3 v1.mid --rendered v1.wav
  python audio_midi_fidelity.py compare original.mp3 v1.mid --json --output-json r.json
  python audio_midi_fidelity.py per-track original.mp3 reconstruido.mid
  python audio_midi_fidelity.py per-track mezcla.wav obra.mid --stem Viola=stem_other.wav Violin=stem_other.wav
  python audio_midi_fidelity.py render reconstruido.mid --output reconstruido.wav

────────────────────────────────────────────────────────────────────────────
Dependencias
────────────────────────────────────────────────────────────────────────────
  numpy, librosa, soundfile, mido. Opcional: fluidsynth (binario) + SoundFont.
"""

import argparse
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
SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/default.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
]
W_CHROMA, W_RMS, W_ONSET, W_DTW = 0.40, 0.20, 0.20, 0.20


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
#  LECTURA DE MIDI
# ════════════════════════════════════════════════════════════════════════════

def read_notes(path):
    """Devuelve (notas, partes). Cada nota: dict(part, pitch, t0, t1, vel,
    program, drum). `partes`: dict id → nombre. La parte es la pista cuando
    hay varias con notas (MIDI tipo 1) o el canal si todo está en una pista."""
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat
    tempos = []  # (tick, tempo)
    for tr in mid.tracks:
        t = 0
        for m in tr:
            t += m.time
            if m.type == "set_tempo":
                tempos.append((t, m.tempo))
    tempos.sort()
    if not tempos or tempos[0][0] != 0:
        tempos.insert(0, (0, 500000))
    cum = [0.0]
    for i in range(1, len(tempos)):
        dt = tempos[i][0] - tempos[i - 1][0]
        cum.append(cum[-1] + mido.tick2second(dt, tpb, tempos[i - 1][1]))
    ticks = [t for t, _ in tempos]

    def sec(tick):
        import bisect
        i = bisect.bisect_right(ticks, tick) - 1
        return cum[i] + mido.tick2second(tick - ticks[i], tpb, tempos[i][1])

    raw, names = [], {}
    note_tracks = 0
    for ti, tr in enumerate(mid.tracks):
        t = 0
        prog = {}
        open_ = {}
        name = None
        had = False
        for m in tr:
            t += m.time
            if m.type == "track_name":
                name = m.name
            elif m.type == "program_change":
                prog[m.channel] = m.program
            elif m.type == "note_on" and m.velocity > 0:
                open_.setdefault((m.channel, m.note), []).append((t, m.velocity, prog.get(m.channel, 0)))
            elif m.type in ("note_off", "note_on"):
                k = (m.channel, m.note)
                if open_.get(k):
                    t0, v, pg = open_[k].pop(0)
                    raw.append((ti, m.channel, m.note, t0, t, v, pg, name))
                    had = True
        note_tracks += had
        names[ti] = name
    notes, parts = [], {}
    for ti, ch, pitch, t0, t1, v, pg, nm in raw:
        pid = ti if note_tracks > 1 else ch
        parts.setdefault(pid, nm or (f"ch{ch}" if note_tracks <= 1 else f"track{ti}"))
        notes.append(dict(part=pid, pitch=pitch, t0=sec(t0), t1=max(sec(t1), sec(t0) + 0.01),
                          vel=v, program=pg, drum=(ch == DRUM_CHANNEL)))
    notes.sort(key=lambda n: (n["t0"], n["pitch"]))
    return notes, parts


# ════════════════════════════════════════════════════════════════════════════
#  RENDER
# ════════════════════════════════════════════════════════════════════════════

def find_soundfont(explicit=None):
    if explicit:
        return explicit if os.path.exists(explicit) else None
    env = os.environ.get("SOUNDFONT")
    if env and os.path.exists(env):
        return env
    for c in SOUNDFONT_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def synth_builtin(notes, sr, duration=None):
    """Sintetizador aditivo mínimo: 6 armónicos, ataque corto, caída
    exponencial para programas percusivos (piano, cromáticos) y sostenido con
    ligero decaimiento para el resto. Sin batería."""
    notes = [n for n in notes if not n["drum"]]
    end = max((n["t1"] for n in notes), default=0.0) + 1.0
    total = int(sr * (duration if duration else end))
    out = np.zeros(total + sr, dtype=np.float32)
    for n in notes:
        f0 = 440.0 * 2 ** ((n["pitch"] - 69) / 12)
        if f0 > sr / 2.2:
            continue
        i0 = int(n["t0"] * sr)
        dur = n["t1"] - n["t0"]
        percussive = n["program"] < 16
        tail = 0.15 if not percussive else 0.0
        L = min(int((dur + tail) * sr), len(out) - i0)
        if L < 2 or i0 >= total:
            continue
        t = np.arange(L) / sr
        env = np.minimum(1.0, t / 0.01)
        if percussive:
            env *= np.exp(-t / max(0.4, min(2.5, 3.0 * dur)))
        else:
            env *= np.exp(-t / 3.0)
            rel = np.clip((dur + tail - t) / max(tail, 1e-3), 0, 1)
            env *= np.where(t > dur, rel, 1.0)
        sig = np.zeros(L, dtype=np.float32)
        for h in range(1, 7):
            if f0 * h > sr / 2.2:
                break
            sig += (1.0 / h) * np.sin(2 * np.pi * f0 * h * t).astype(np.float32)
        out[i0:i0 + L] += (n["vel"] / 127.0) * 0.12 * env.astype(np.float32) * sig
    peak = np.abs(out).max()
    if peak > 0:
        out *= 0.9 / peak
    return out[:total]


def render_midi(midi_path, sr=44100, soundfont=None, engine="auto"):
    """Devuelve (señal mono float32 a `sr`, motor usado)."""
    fs = shutil.which("fluidsynth")
    sfp = find_soundfont(soundfont)
    use_fs = engine == "fluidsynth" or (engine == "auto" and fs and sfp)
    if engine == "fluidsynth" and not (fs and sfp):
        raise RuntimeError("fluidsynth o SoundFont no disponibles (usa --soundfont o --engine builtin)")
    if use_fs:
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "r.wav")
            subprocess.run([fs, "-ni", "-F", wav, "-r", str(sr), sfp, str(midi_path)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            y, sr0 = sf.read(wav, dtype="float32", always_2d=True)
        return y.mean(axis=1), "fluidsynth"
    notes, _ = read_notes(midi_path)
    return synth_builtin(notes, sr), "builtin"


# ════════════════════════════════════════════════════════════════════════════
#  AUDIO Y CARACTERÍSTICAS
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


CQT_FMIN_MIDI = 24  # C1


def coarse_hop(sr):
    return int(round(sr / 10))


def features(y, sr):
    hop = coarse_hop(sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-5))
    ons = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units="time", backtrack=False)
    return dict(chroma=chroma, db=db, onsets=np.asarray(ons), hop=hop, sr=sr, dur=len(y) / sr)


def _cos(a, b):
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    return (a * b).sum(axis=0) / np.maximum(na * nb, 1e-9)


def onset_f1(ref_times, est_times, tol):
    """F1 con emparejamiento voraz 1-a-1 (listas ordenadas)."""
    ref, est = np.sort(ref_times), np.sort(est_times)
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0, 0.0
    i = j = hit = 0
    while i < len(ref) and j < len(est):
        d = est[j] - ref[i]
        if abs(d) <= tol:
            hit += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    p, r = hit / len(est), hit / len(ref)
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def midi_onsets(notes):
    ts = sorted({round(n["t0"], 3) for n in notes if not n["drum"]})
    merged = []
    for t in ts:
        if not merged or t - merged[-1] > 0.03:
            merged.append(t)
    return np.asarray(merged)


def compare_one(fa, y_r, sr, notes, tol, window):
    fr = features(y_r, sr)
    n = min(fa["chroma"].shape[1], fr["chroma"].shape[1])
    # Se compara solo hasta donde llega el MIDI (la cola de reverb del original
    # no es culpa de la transcripción) y se informa de la diferencia de duración.
    ca, cr = fa["chroma"][:, :n], fr["chroma"][:, :n]
    cos = _cos(ca, cr)
    # DTW sobre cromas: coste coseno medio a lo largo del camino óptimo
    # epsilon: un fotograma en silencio (croma nulo) daría NaN en la distancia coseno
    D, wp = librosa.sequence.dtw(X=ca + 1e-6, Y=cr + 1e-6, metric="cosine")
    dtw_cost = float(D[-1, -1] / len(wp))
    da, dr = fa["db"][:n], fr["db"][:n]
    # Los silencios del render (db ≈ -100) no cuentan para la correlación
    gain = np.median(da - dr)
    rms_corr = float(np.corrcoef(da, dr)[0, 1]) if n > 2 else 0.0
    rms_corr = 0.0 if np.isnan(rms_corr) else rms_corr
    rms_diff = float(np.mean(np.abs(da - (dr + gain))))
    f1, p, r = onset_f1(fa["onsets"], midi_onsets(notes), tol)
    chroma_mean = float(cos.mean())
    score = (W_CHROMA * chroma_mean + W_RMS * max(0.0, rms_corr) + W_ONSET * f1
             + W_DTW * (1.0 - min(1.0, dtw_cost)))
    fps = fa["sr"] / fa["hop"]
    w = max(1, int(window * fps))
    windows = []
    for s in range(0, n, w):
        seg = cos[s:s + w]
        if len(seg) >= max(3, w // 4):
            windows.append(dict(t0=round(s / fps, 1), t1=round(min(n, s + w) / fps, 1),
                                chroma=round(float(seg.mean()), 4)))
    return dict(
        score=round(score, 4), chroma_mean=round(chroma_mean, 4),
        chroma_median=round(float(np.median(cos)), 4), dtw_cost=round(dtw_cost, 4),
        rms_corr=round(rms_corr, 4), rms_db_diff=round(rms_diff, 2),
        onset_f1=round(f1, 4), onset_precision=round(p, 4), onset_recall=round(r, 4),
        duration_audio=round(fa["dur"], 2), duration_render=round(fr["dur"], 2),
        n_notes=len(notes), windows=windows,
    )


# ════════════════════════════════════════════════════════════════════════════
#  SOPORTE ACÚSTICO POR NOTA
# ════════════════════════════════════════════════════════════════════════════

CQT_HOP = 512
CQT_BPO = 36
CQT_OCT = 7


def cqt_mag(y, sr):
    return np.abs(librosa.cqt(y, sr=sr, hop_length=CQT_HOP, fmin=librosa.midi_to_hz(CQT_FMIN_MIDI),
                              n_bins=CQT_BPO * CQT_OCT, bins_per_octave=CQT_BPO)).astype(np.float32)


def note_ratio(C, sr, pitch, t0, t1):
    """Energía del fundamental / máximo del fotograma durante la nota (0-1);
    None si la nota queda fuera del rango del CQT."""
    center = int(round((pitch - CQT_FMIN_MIDI) * CQT_BPO / 12))
    if center < 1 or center >= C.shape[0] - 1:
        return None
    dt = CQT_HOP / sr
    f0 = max(0, int(t0 / dt))
    f1 = max(f0 + 1, int(t1 / dt))
    seg = C[:, f0:f1]
    if seg.shape[1] == 0:
        return None
    e = seg[center - 1:center + 2, :].max(axis=0).mean()
    ref = seg.max(axis=0).mean()
    return float(e / ref) if ref > 0 else 0.0


def parse_stems(pairs):
    out = []
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--stem espera PISTA=WAV, recibido «{p}»")
        k, v = p.split("=", 1)
        out.append((k.strip(), v.strip()))
    return out


def part_matches(key, pid, name):
    return key == str(pid) or key.lower() in (name or "").lower()


def per_track(audio_path, midi_path, stems, support, sr):
    notes, parts = read_notes(midi_path)
    cache = {}

    def cqt_of(path):
        if path not in cache:
            _info(f"  CQT de {os.path.basename(path)} …")
            cache[path] = cqt_mag(load_audio(path, sr), sr)
        return cache[path]

    res = []
    for pid, name in parts.items():
        ref = audio_path
        for key, wav in stems:
            if part_matches(key, pid, name):
                ref = wav
                break
        C = cqt_of(ref)
        ns = [n for n in notes if n["part"] == pid and not n["drum"]]
        ratios = [note_ratio(C, sr, n["pitch"], n["t0"], n["t1"]) for n in ns]
        valid = [r for r in ratios if r is not None]
        sup = float(np.mean([r >= support for r in valid])) if valid else 0.0
        res.append(dict(part=pid, name=name, n_notes=len(ns), n_measured=len(valid),
                        support=round(sup, 4),
                        ratio_median=round(float(np.median(valid)), 4) if valid else None,
                        ratio_mean=round(float(np.mean(valid)), 4) if valid else None,
                        reference=os.path.basename(ref),
                        pitch_range=[min((n["pitch"] for n in ns), default=None),
                                     max((n["pitch"] for n in ns), default=None)]))
    return res


# ════════════════════════════════════════════════════════════════════════════
#  SUBCOMANDOS
# ════════════════════════════════════════════════════════════════════════════

def cmd_render(args):
    out = args.output or os.path.splitext(args.midi)[0] + ".wav"
    y, eng = render_midi(args.midi, sr=args.sr, soundfont=args.soundfont, engine=args.engine)
    sf.write(_ensure_parent(out), y, args.sr)
    print(f"{out}  ({len(y) / args.sr:.1f} s, motor: {eng})")
    return 0


def _band(score):
    return ("excelente" if score >= 0.85 else "alta" if score >= 0.75 else
            "media" if score >= 0.6 else "baja")


def cmd_compare(args):
    if args.rendered and len(args.midis) != 1:
        print("[ERROR] --rendered solo es válido con un único MIDI", file=sys.stderr)
        return 1
    sr = args.sr
    _info(f"Analizando audio original: {args.audio}")
    fa = features(load_audio(args.audio, sr), sr)
    results = []
    for mp in args.midis:
        notes, _ = read_notes(mp)
        if args.rendered:
            y_r, eng = load_audio(args.rendered, sr), "externo"
        else:
            _info(f"Renderizando {os.path.basename(mp)} …")
            y44, eng = render_midi(mp, sr=44100, soundfont=args.soundfont, engine=args.engine)
            y_r = librosa.resample(y44, orig_sr=44100, target_sr=sr)
        r = compare_one(fa, y_r, sr, notes, args.tol, args.window)
        r["midi"] = mp
        r["engine"] = eng
        results.append(r)
    results.sort(key=lambda r: -r["score"])
    if args.output_json:
        with open(_ensure_parent(args.output_json), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=1))
        return 0
    for i, r in enumerate(results, 1):
        print(f"\n{'#' + str(i) if len(results) > 1 else ''} {r['midi']}  [{r['engine']}, {r['n_notes']} notas]")
        print(f"  score {r['score']:.3f} ({_band(r['score'])})")
        print(f"  chroma  media {r['chroma_mean']:.3f}  mediana {r['chroma_median']:.3f}   dtw {r['dtw_cost']:.3f}")
        print(f"  rms     corr {r['rms_corr']:.3f}  |Δ| {r['rms_db_diff']:.1f} dB")
        print(f"  ataques F1 {r['onset_f1']:.3f}  (precisión {r['onset_precision']:.2f}, "
              f"recall {r['onset_recall']:.2f})")
        if abs(r["duration_audio"] - r["duration_render"]) > 1.0:
            print(f"  duración audio {r['duration_audio']:.1f}s vs render {r['duration_render']:.1f}s "
                  "(se compara hasta la más corta)")
        worst = sorted(r["windows"], key=lambda w: w["chroma"])[:3]
        if worst:
            print("  ventanas más débiles: " + ", ".join(
                f"{w['t0']:.0f}-{w['t1']:.0f}s ({w['chroma']:.2f})" for w in worst))
    return 0


def cmd_per_track(args):
    stems = parse_stems(args.stem)
    res = per_track(args.audio, args.midi, stems, args.support, args.sr)
    if args.output_json:
        with open(_ensure_parent(args.output_json), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    print(f"Soporte acústico por pista (ratio ≥ {args.support}):")
    for r in res:
        flag = "  ← poco respaldada" if r["support"] < 0.5 else ""
        print(f"  [{r['part']}] {str(r['name']):<14} notas {r['n_notes']:>5}  soporte {r['support']:.2f}  "
              f"ratio med {r['ratio_median']}  ref {r['reference']}{flag}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="audio_midi_fidelity.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compare", help="métricas de fidelidad audio ↔ MIDI renderizado")
    p.add_argument("audio", help="audio original")
    p.add_argument("midis", nargs="+", help="MIDI candidatos")
    p.add_argument("--rendered", default=None, help="WAV ya renderizado (solo con un MIDI)")
    p.add_argument("--soundfont", default=None, help="SoundFont para FluidSynth")
    p.add_argument("--engine", default="auto", choices=["auto", "fluidsynth", "builtin"])
    p.add_argument("--window", type=float, default=30.0, help="ventana del desglose temporal (s)")
    p.add_argument("--tol", type=float, default=0.07, help="tolerancia de ataques (s)")
    p.add_argument("--sr", type=int, default=22050, help="frecuencia de análisis")
    p.add_argument("--json", action="store_true", help="salida JSON")
    p.add_argument("--output-json", default=None, help="guarda el JSON en fichero")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("per-track", help="soporte acústico de cada pista del MIDI")
    p.add_argument("audio", help="audio de referencia")
    p.add_argument("midi", help="MIDI a auditar")
    p.add_argument("--stem", nargs="+", default=None, help="PISTA=WAV: stem propio de una pista")
    p.add_argument("--support", type=float, default=0.05, help="ratio mínimo de soporte")
    p.add_argument("--sr", type=int, default=22050, help="frecuencia de análisis")
    p.add_argument("--json", action="store_true", help="salida JSON")
    p.add_argument("--output-json", default=None, help="guarda el JSON en fichero")
    p.set_defaults(func=cmd_per_track)

    p = sub.add_parser("render", help="MIDI → WAV")
    p.add_argument("midi", help="MIDI de entrada")
    p.add_argument("--output", default=None, help="WAV de salida")
    p.add_argument("--soundfont", default=None, help="SoundFont para FluidSynth")
    p.add_argument("--engine", default="auto", choices=["auto", "fluidsynth", "builtin"])
    p.add_argument("--sr", type=int, default=44100, help="frecuencia del WAV")
    p.set_defaults(func=cmd_render)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
