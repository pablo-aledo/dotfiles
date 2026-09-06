#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  synth_forge.py — Motor de síntesis multi-paradigma                   ║
╚══════════════════════════════════════════════════════════════════════╝

Implementa en Python puro (numpy + stdlib, sin Csound/SuperCollider/Faust
externos) seis paradigmas clásicos de síntesis sonora:

    sample      Síntesis por muestras (keymap + capas de velocity, SFZ-lite)
    waveguide   Modelado físico por guía de onda (Karplus-Strong extendido;
                modos plucked/bowed con fricción stick-slip)
    additive    Síntesis aditiva (suma de armónicos ponderados)
    fm          Síntesis FM de 2 operadores (relación armónica/inarmónica)
    brass       Modelado físico de tubo (labios no lineales) + banco modal
                resonante (excitador no lineal + resonadores tipo Klank)
    granular    Síntesis granular (nube de granos con ventana de Hann)

USO
    synth_forge.py sample     --keymap kb/keymap.json --midi tema.mid --out out.wav
    synth_forge.py sample     --make-demo-bank kb/          # crea un banco de prueba
    synth_forge.py waveguide  --mode bowed --midi tema.mid --out out.wav
    synth_forge.py waveguide  --mode plucked --notes "60:0:1.0:100" --out out.wav
    synth_forge.py additive   --harmonics 1,0,0.6,0,0.3,0,0.15 --midi tema.mid --out out.wav
    synth_forge.py fm         --ratio 2:1 --index-attack 3.2 --midi tema.mid --out out.wav
    synth_forge.py brass      --brightness 0.5 --fm-attack --midi tema.mid --out out.wav
    synth_forge.py granular   --source ruido --center-freq 900 --duration 6 --out out.wav
    synth_forge.py demo       --outdir demo/
    synth_forge.py info

ENTRADA DE NOTAS (compartida por sample/waveguide/additive/fm/brass)
    --midi FICHERO.mid     lee notas de un MIDI (requiere 'mido')
    --notes "n:ini:dur:vel,n:ini:dur:vel,..."
                            lista manual, ej: "62:0.0:0.75:90,65:0.75:0.375:95"
                            (nota MIDI, inicio en s, duración en s, velocity 0-127)

DEPENDENCIAS
    numpy (obligatorio)
    mido  (opcional; solo si se usa --midi — pip install mido)
"""

import argparse
import json
import math
import os
import random
import struct
import sys
import wave

import numpy as np

SR_DEFAULT = 48000


# ─────────────────────────────────────────────────────────────────────────
# Utilidades comunes: notas, envolventes, WAV
# ─────────────────────────────────────────────────────────────────────────

def midi_to_freq(note):
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def parse_notes_arg(s):
    """'60:0.0:1.0:100,64:1.0:0.5:90' -> [{note,start,dur,vel}, ...]"""
    notes = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(':')
        note = int(parts[0])
        start = float(parts[1]) if len(parts) > 1 else 0.0
        dur = float(parts[2]) if len(parts) > 2 else 1.0
        vel = int(parts[3]) if len(parts) > 3 else 100
        notes.append(dict(note=note, start=start, dur=dur, vel=vel))
    return notes


def load_midi_notes(path, channel=None):
    """Lee un fichero MIDI y empareja note_on/note_off en (note,start,dur,vel).
    Simplificación: cada pista se recorre con su propio reloj absoluto y los
    cambios de tempo se aplican globalmente conforme aparecen; para partituras
    de una sola pista de control (el caso habitual de este ecosistema) es
    suficiente."""
    try:
        import mido
    except ImportError:
        sys.exit("Se requiere 'mido' para leer --midi (pip install mido), "
                  "o usa --notes para introducir las notas a mano.")
    mid = mido.MidiFile(path)
    notes = []
    for track in mid.tracks:
        t = 0.0
        tempo = 500000  # 120 BPM por defecto
        active = {}
        for msg in track:
            t += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'note_on' and msg.velocity > 0:
                if channel is None or getattr(msg, 'channel', channel) == channel:
                    active[msg.note] = (t, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active:
                    start, vel = active.pop(msg.note)
                    notes.append(dict(note=msg.note, start=start,
                                       dur=max(t - start, 0.03), vel=vel))
    notes.sort(key=lambda n: n['start'])
    return notes


def get_notes(args):
    if getattr(args, 'midi', None):
        return load_midi_notes(args.midi, channel=getattr(args, 'channel', None))
    if getattr(args, 'notes', None):
        return parse_notes_arg(args.notes)
    sys.exit("Se necesita --midi o --notes para saber qué tocar.")


def adsr(n_total, note_dur, a, d, s, r, sr):
    """Envolvente ADSR clásica (Attack/Decay/Sustain/Release) sobre n_total
    muestras, con la nota sostenida hasta note_dur segundos antes de iniciar
    el release."""
    env = np.zeros(n_total)
    a_n = min(int(a * sr), n_total)
    if a_n > 0:
        env[:a_n] = np.linspace(0, 1, a_n)
    idx = a_n
    d_n = int(d * sr)
    d_end = min(idx + d_n, n_total)
    if d_end > idx:
        env[idx:d_end] = np.linspace(1, s, d_end - idx)
    idx = d_end
    sustain_end = min(max(int(note_dur * sr), idx), n_total)
    if sustain_end > idx:
        env[idx:sustain_end] = s
    idx = sustain_end
    r_n = int(r * sr)
    r_end = min(idx + r_n, n_total)
    if r_end > idx:
        env[idx:r_end] = np.linspace(s, 0, r_end - idx)
    return env


def write_wav(path, samples, sr=SR_DEFAULT):
    """Escribe WAV mono de 16 bits a partir de un array float32 en [-1, 1]."""
    samples = np.clip(samples, -1.0, 1.0)
    ints = (samples * 32767.0).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(ints.tobytes())


def load_wav_mono(path):
    with wave.open(path, 'rb') as w:
        sr = w.getframerate()
        n = w.getnframes()
        sw = w.getsampwidth()
        ch = w.getnchannels()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128.0
    else:
        raise ValueError(f"{path}: solo se soportan WAV de 8 o 16 bits")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def render_notes(notes, voice_fn, sr=SR_DEFAULT, tail=0.6):
    """Mezcla el render de cada nota (voice_fn(note,dur,vel)->array) en un
    único buffer, colocada en su instante de inicio."""
    if not notes:
        return np.zeros(1, dtype=np.float32)
    end = max(n['start'] + n['dur'] for n in notes) + tail
    buf = np.zeros(int(end * sr) + sr, dtype=np.float64)
    for n in notes:
        audio = voice_fn(n['note'], n['dur'], n['vel'])
        start_i = int(n['start'] * sr)
        end_i = start_i + len(audio)
        if end_i > len(buf):
            buf = np.concatenate([buf, np.zeros(end_i - len(buf))])
        buf[start_i:end_i] += audio
    peak = np.max(np.abs(buf))
    if peak > 0:
        buf = buf / peak * 0.9
    return buf.astype(np.float32)


def biquad_resonator(x, freq, bw, sr=SR_DEFAULT):
    """Resonador de 2 polos (equivalente simplificado a un modo de Klank):
    controla el ancho de banda 'bw' (Hz) y la frecuencia central 'freq'."""
    r = math.exp(-math.pi * bw / sr)
    a1 = 2 * r * math.cos(2 * math.pi * freq / sr)
    a2 = -r * r
    y = np.zeros_like(x)
    y1 = y2 = 0.0
    for i in range(len(x)):
        yi = x[i] + a1 * y1 + a2 * y2
        y[i] = yi
        y2, y1 = y1, yi
    # normaliza la ganancia de resonancia para que no dispare con bw pequeño
    gain = (1 - r) * math.sqrt(max(1 - a1 * r / (1 - r * r if r < 1 else 1), 0.01)) \
        if r < 1 else 1.0
    return y * gain


# ─────────────────────────────────────────────────────────────────────────
# 1. Síntesis por muestras (sampler SFZ-lite)
# ─────────────────────────────────────────────────────────────────────────

def resolve_region(regions, note, vel):
    candidates = [r for r in regions
                  if r.get('lokey', 0) <= note <= r.get('hikey', 127)
                  and r.get('lovel', 0) <= vel <= r.get('hivel', 127)]
    if not candidates:
        if not regions:
            return None
        candidates = sorted(
            regions,
            key=lambda r: abs(note - r.get('root', (r.get('lokey', 0) + r.get('hikey', 127)) // 2))
        )
        return candidates[0]
    if len(candidates) > 1:
        candidates.sort(key=lambda r: abs(vel - (r.get('lovel', 0) + r.get('hivel', 127)) / 2))
    return candidates[0]


def resample_linear(data, ratio, target_len=None):
    """Reproduce 'data' a una velocidad 'ratio' veces la original (>1 = agudo)."""
    n_src = len(data)
    n_dst = target_len if target_len is not None else max(1, int(n_src / ratio))
    if n_src < 2:
        return np.zeros(n_dst)
    x_old = np.arange(n_src)
    x_new = np.linspace(0, n_src - 1, n_dst) * ratio
    x_new = np.clip(x_new, 0, n_src - 1)
    return np.interp(x_new, x_old, data)


def make_demo_bank(outdir, sr=SR_DEFAULT):
    """Genera un pequeño banco de 'muestras' sintetizadas (vía waveguide
    pulsado) más su keymap.json, para poder probar 'sample' sin depender de
    una librería orquestal real tipo SSO/VPO."""
    os.makedirs(outdir, exist_ok=True)
    roots = [48, 55, 60, 67, 72]  # C3, G3, C4, G4, C5
    regions = []
    for i, root in enumerate(roots):
        for layer, (lovel, hivel, brightness) in enumerate([(0, 63, 0.25), (64, 127, 0.6)]):
            audio = waveguide_pluck_voice(root, 1.6, (lovel + hivel) // 2, brightness=brightness)
            fname = f"demo_{root}_{layer}.wav"
            write_wav(os.path.join(outdir, fname), audio.astype(np.float32), sr)
            lokey = 24 if i == 0 else (roots[i - 1] + root) // 2 + 1
            hikey = 108 if i == len(roots) - 1 else (root + roots[i + 1]) // 2
            regions.append(dict(file=fname, root=root, lokey=lokey, hikey=hikey,
                                 lovel=lovel, hivel=hivel, attack=0.005, release=0.4))
    with open(os.path.join(outdir, 'keymap.json'), 'w') as f:
        json.dump(regions, f, indent=2)


def cmd_sample(args):
    if args.make_demo_bank:
        make_demo_bank(args.make_demo_bank, sr=args.sr)
        print(f"Banco de demo creado en {args.make_demo_bank}/ "
              f"(WAVs sintetizados + keymap.json)")
        return
    if not args.out:
        sys.exit("--out es obligatorio.")
    if not args.keymap:
        sys.exit("--keymap es obligatorio (o usa --make-demo-bank DIR para "
                  "generar uno de prueba sin librería orquestal real).")
    with open(args.keymap) as f:
        regions = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(args.keymap))
    cache = {}

    def voice(note, dur, vel):
        region = resolve_region(regions, note, vel)
        if region is None:
            return np.zeros(int(dur * args.sr))
        path = os.path.join(base_dir, region['file'])
        if path not in cache:
            cache[path] = load_wav_mono(path)
        data, _sr = cache[path]
        root = region.get('root', 60)
        ratio = midi_to_freq(note) / midi_to_freq(root)
        release = region.get('release', 0.4)
        n_needed = int((dur + release) * args.sr)
        shifted = resample_linear(data, ratio, target_len=n_needed)
        env = adsr(len(shifted), dur, region.get('attack', 0.01), 0.05, 0.95, release, args.sr)
        return shifted * env * (vel / 127.0)

    notes = get_notes(args)
    audio = render_notes(notes, voice, sr=args.sr)
    write_wav(args.out, audio, args.sr)
    print(f"[sample] {len(notes)} notas -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────
# 2. Modelado físico por guía de onda (waveguide)
# ─────────────────────────────────────────────────────────────────────────

def waveguide_pluck_voice(note, dur, vel, brightness=0.5, damping=0.996, sr=SR_DEFAULT):
    """Karplus-Strong clásico: ráfaga de ruido circulando por una línea de
    retardo con filtro de pérdidas de un polo (controla el brillo/damping)."""
    freq = midi_to_freq(note)
    n_total = int((dur + 0.5) * sr)
    delay_len = max(2, int(round(sr / freq)))
    buf = np.random.rand(delay_len) * 2 - 1
    out = np.zeros(n_total)
    prev = 0.0
    idx = 0
    for i in range(n_total):
        cur = buf[idx]
        filtered = brightness * cur + (1 - brightness) * prev
        buf[idx] = filtered * damping
        prev = filtered
        out[i] = cur
        idx = (idx + 1) % delay_len
    return out * (vel / 127.0) * 0.5


def waveguide_bow_voice(note, dur, vel, bow_pressure=0.5, bow_velocity=0.5,
                         vibrato_freq=5.5, vibrato_gain=0.007, damping=0.999,
                         sr=SR_DEFAULT):
    """Cuerda frotada: excitador de fricción no lineal (stick-slip, vía tanh)
    realimentado continuamente en la línea de retardo mientras dura la nota,
    con vibrato y cola de release."""
    freq = midi_to_freq(note)
    release = 0.3
    n_total = int((dur + release) * sr)
    n_gate = int(dur * sr)
    delay_len = max(2, int(round(sr / freq)))
    buf = np.zeros(delay_len)
    out = np.zeros(n_total)
    prev = 0.0
    idx = 0
    vib_phase = 0.0
    for i in range(n_total):
        gate = 1.0 if i < n_gate else max(0.0, 1.0 - (i - n_gate) / (release * sr))
        vib_phase += 2 * math.pi * vibrato_freq / sr
        vib = vibrato_gain * math.sin(vib_phase)
        string_vel = buf[idx]
        rel_vel = bow_velocity * (1 + vib) - string_vel
        friction = math.tanh(rel_vel * 8.0) * bow_pressure
        excite = friction * gate
        new = buf[idx] + excite * 0.5
        filtered = 0.5 * new + 0.5 * prev
        buf[idx] = filtered * damping
        prev = filtered
        out[i] = filtered
        idx = (idx + 1) % delay_len
    return out * (vel / 127.0) * 0.9


def cmd_waveguide(args):
    notes = get_notes(args)
    if args.mode == 'bowed':
        def voice(note, dur, vel):
            return waveguide_bow_voice(note, dur, vel, bow_pressure=args.bow_pressure,
                                        bow_velocity=args.bow_velocity,
                                        vibrato_gain=args.vibrato, sr=args.sr)
    else:
        def voice(note, dur, vel):
            return waveguide_pluck_voice(note, dur, vel, brightness=args.brightness, sr=args.sr)
    audio = render_notes(notes, voice, sr=args.sr)
    write_wav(args.out, audio, args.sr)
    print(f"[waveguide:{args.mode}] {len(notes)} notas -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────
# 3. Síntesis aditiva
# ─────────────────────────────────────────────────────────────────────────

def additive_voice(note, dur, vel, harmonics, attack=0.01, decay=0.1, sustain=0.8,
                    release=0.15, sr=SR_DEFAULT):
    freq = midi_to_freq(note)
    n = int((dur + release) * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for k, w in enumerate(harmonics, start=1):
        if w == 0:
            continue
        sig += w * np.sin(2 * np.pi * freq * k * t)
    env = adsr(n, dur, attack, decay, sustain, release, sr)
    return sig * env * (vel / 127.0) * 0.3


def parse_harmonics(s):
    return [float(x) for x in s.split(',')]


def cmd_additive(args):
    notes = get_notes(args)
    harmonics = parse_harmonics(args.harmonics)

    def voice(note, dur, vel):
        return additive_voice(note, dur, vel, harmonics, sr=args.sr)

    audio = render_notes(notes, voice, sr=args.sr)
    write_wav(args.out, audio, args.sr)
    print(f"[additive] {len(harmonics)} armónicos, {len(notes)} notas -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────
# 4. Síntesis FM de 2 operadores
# ─────────────────────────────────────────────────────────────────────────

def parse_ratio(s):
    c, m = s.split(':')
    return float(c), float(m)


def fm_voice(note, dur, vel, ratio_c, ratio_m, index_attack=3.2, index_sustain=1.8,
             attack=0.02, decay=0.15, sustain=0.75, release=0.12, sr=SR_DEFAULT):
    base = midi_to_freq(note)
    fc = base * ratio_c
    fm = base * ratio_m
    n = int((dur + release) * sr)
    t = np.arange(n) / sr
    ramp_n = min(int(0.05 * sr), n)
    idx_env = np.empty(n)
    idx_env[:ramp_n] = np.linspace(index_attack, index_sustain, ramp_n)
    idx_env[ramp_n:] = index_sustain
    sig = np.sin(2 * np.pi * fc * t + idx_env * np.sin(2 * np.pi * fm * t))
    env = adsr(n, dur, attack, decay, sustain, release, sr)
    return sig * env * (vel / 127.0) * 0.35


def cmd_fm(args):
    notes = get_notes(args)
    ratio_c, ratio_m = parse_ratio(args.ratio)

    def voice(note, dur, vel):
        return fm_voice(note, dur, vel, ratio_c, ratio_m,
                         index_attack=args.index_attack,
                         index_sustain=args.index_sustain, sr=args.sr)

    audio = render_notes(notes, voice, sr=args.sr)
    write_wav(args.out, audio, args.sr)
    if ratio_c == 0 or ratio_m == 0:
        tipo = "degenerada (ratio con componente 0)"
    else:
        tipo = "armónica" if (ratio_c / ratio_m).is_integer() or (ratio_m / ratio_c).is_integer() else "inarmónica"
    print(f"[fm] ratio {args.ratio} ({tipo}), {len(notes)} notas -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────
# 5. Metales: tubo (labios no lineales) + banco modal
# ─────────────────────────────────────────────────────────────────────────

def brass_voice(note, dur, vel, brightness=0.4, fm_attack=False, sr=SR_DEFAULT):
    freq = midi_to_freq(note)
    release = 0.25
    n = int((dur + release) * sr)
    t = np.arange(n) / sr
    env = adsr(n, dur, 0.04, 0.15, 0.75, release, sr)

    lip_drive = np.sin(2 * np.pi * freq * t) * (1 + brightness * env)
    waveshape = np.tanh(lip_drive * 3)

    modes = (1.0, 2.0, 3.01, 4.02, 5.1)      # ligeramente desafinados respecto a la serie armónica
    mode_amps = (1.0, 0.6, 0.35, 0.2, 0.1)
    mode_bw = (60, 90, 140, 200, 260)
    sig = np.zeros(n)
    for m, amp, bw in zip(modes, mode_amps, mode_bw):
        sig += amp * biquad_resonator(waveshape, freq * m, bw, sr=sr)
    sig = sig / sum(mode_amps)

    if fm_attack:
        # ataque FM inarmónico (relación ~sqrt(2)): "chasquido metálico"
        atk_n = min(int(0.015 * sr), n)
        atk_t = np.arange(atk_n) / sr
        idx_env = np.linspace(8.0, 1.0, atk_n)
        click = np.sin(2 * np.pi * freq * atk_t
                        + idx_env * np.sin(2 * np.pi * freq * 1.41421356 * atk_t))
        sig[:atk_n] += click * 0.5

    return sig * env * (vel / 127.0) * 0.5


def cmd_brass(args):
    notes = get_notes(args)

    def voice(note, dur, vel):
        return brass_voice(note, dur, vel, brightness=args.brightness,
                            fm_attack=args.fm_attack, sr=args.sr)

    audio = render_notes(notes, voice, sr=args.sr)
    write_wav(args.out, audio, args.sr)
    print(f"[brass] brightness={args.brightness} fm_attack={args.fm_attack} "
          f"{len(notes)} notas -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────
# 6. Síntesis granular
# ─────────────────────────────────────────────────────────────────────────

def make_metal_source(duration, center_freq=900, bw=25, sr=SR_DEFAULT):
    """Fuente de ruido filtrado en banda estrecha con 'ring' resonante,
    imitando metal percutido antes de granular."""
    n = int(duration * sr)
    noise = np.random.randn(n)
    return biquad_resonator(noise, center_freq, bw, sr=sr)


def granulate(source, duration, density, grain_dur, pos_spread, pitch_spread, sr=SR_DEFAULT):
    n_out = int(duration * sr)
    out = np.zeros(n_out)
    grain_n = max(4, int(grain_dur * sr))
    hann = np.hanning(grain_n)
    src_len = len(source)
    if src_len <= grain_n:
        source = np.tile(source, int(np.ceil(grain_n * 2 / max(src_len, 1))) + 1)
        src_len = len(source)
    n_grains = max(0, int(duration * density))
    if n_grains == 0 or density <= 0:
        return out  # sin densidad no hay granos: textura en silencio, no es un error
    for g in range(n_grains):
        onset = int((g / density + random.uniform(-0.5, 0.5) / density) * sr)
        if onset < 0 or onset >= n_out:
            continue
        max_pos = src_len - grain_n
        center = random.uniform(0, 1) * max_pos
        half_width = pos_spread * max_pos / 2
        pos = int(np.clip(center, 0, max_pos))
        pitch_ratio = 1 + random.uniform(-pitch_spread, pitch_spread)
        read_idx = pos + np.arange(grain_n) * pitch_ratio
        read_idx = np.clip(read_idx, 0, src_len - 1).astype(int)
        grain = source[read_idx] * hann
        end = min(onset + grain_n, n_out)
        out[onset:end] += grain[:end - onset]
    return out


def cmd_granular(args):
    if args.source == 'ruido':
        source = make_metal_source(max(args.duration, 2.0), center_freq=args.center_freq, sr=args.sr)
    else:
        source, _sr = load_wav_mono(args.source)
    out = granulate(source, args.duration, args.density, args.grain_dur,
                     args.pos_spread, args.pitch_spread, sr=args.sr)
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.9
    write_wav(args.out, out.astype(np.float32), args.sr)
    n_grains = int(args.duration * args.density)
    print(f"[granular] fuente={args.source} {n_grains} granos, "
          f"{args.duration}s -> {args.out}")


# ─────────────────────────────────────────────────────────────────────────
# demo / info
# ─────────────────────────────────────────────────────────────────────────

DEMO_PHRASE_NOTES = [  # frase corta de ejemplo (motivo ascendente en re menor)
    dict(note=62, start=0.0, dur=0.9, vel=80),   # D4
    dict(note=65, start=0.9, dur=0.45, vel=85),  # F4
    dict(note=69, start=1.35, dur=0.45, vel=85), # A4
    dict(note=74, start=1.8, dur=0.9, vel=90),   # D5
    dict(note=73, start=2.7, dur=0.9, vel=75),   # Cs5
    dict(note=76, start=3.6, dur=0.45, vel=70),  # E5
]

INFO_TABLE = [
    ("sample", "Síntesis por muestras",
     "Keymap + capas de velocity, resample lineal por pitch. Requiere --keymap "
     "(usa --make-demo-bank para generar uno de prueba)."),
    ("waveguide", "Modelado físico (guía de onda)",
     "Karplus-Strong extendido. --mode plucked (pulsada) o bowed (frotada, "
     "con fricción stick-slip continua y vibrato)."),
    ("additive", "Síntesis aditiva",
     "Suma de armónicos ponderados (--harmonics), ideal para timbres 'puros' "
     "de referencia tipo clarinete."),
    ("fm", "Síntesis FM (2 operadores)",
     "--ratio c:m entero = espectro armónico (maderas); irracional (p.ej. "
     "1:1.41) = espectro inarmónico (campanas/metal)."),
    ("brass", "Metales: tubo + banco modal",
     "Labios no lineales (waveshaper tanh) excitando un banco de resonadores "
     "ligeramente desafinados; --fm-attack añade el chasquido inarmónico de ataque."),
    ("granular", "Síntesis granular",
     "Nube de granos con ventana de Hann sobre una fuente WAV o ruido "
     "resonante sintético (--source ruido)."),
]


def cmd_demo(args):
    os.makedirs(args.outdir, exist_ok=True)
    sr = args.sr

    renders = {
        'sample': None,  # se omite salvo que se pida banco (necesita WAVs reales)
        'waveguide_bowed': lambda n, d, v: waveguide_bow_voice(n, d, v, sr=sr),
        'waveguide_plucked': lambda n, d, v: waveguide_pluck_voice(n, d, v, sr=sr),
        'additive_clarinet': lambda n, d, v: additive_voice(
            n, d, v, [1, 0, 0.6, 0, 0.3, 0, 0.15], sr=sr),
        'fm_oboe': lambda n, d, v: fm_voice(n, d, v, 1.0, 2.0, sr=sr),
        'fm_inharmonic': lambda n, d, v: fm_voice(n, d, v, 1.0, 1.41421356, sr=sr),
        'brass': lambda n, d, v: brass_voice(n, d, v, fm_attack=True, sr=sr),
    }
    for name, voice in renders.items():
        if voice is None:
            continue
        audio = render_notes(DEMO_PHRASE_NOTES, voice, sr=sr)
        path = os.path.join(args.outdir, f"demo_{name}.wav")
        write_wav(path, audio, sr)
        print(f"  -> {path}")

    tex_path = os.path.join(args.outdir, "demo_granular.wav")
    source = make_metal_source(3.0, center_freq=900, sr=sr)
    tex = granulate(source, 3.0, 40, 0.04, 0.3, 0.15, sr=sr)
    tex = tex / (np.max(np.abs(tex)) or 1) * 0.9
    write_wav(tex_path, tex.astype(np.float32), sr)
    print(f"  -> {tex_path}")
    print(f"\nDemo completa: {len(renders) - 1 + 1} renders en {args.outdir}/")


def cmd_info(args):
    print("Técnicas de síntesis implementadas (ver cabecera del fichero):\n")
    for cmd, title, desc in INFO_TABLE:
        print(f"  {cmd:<10} {title}")
        print(f"             {desc}\n")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def add_note_input_args(sp):
    sp.add_argument('--midi', help="Fichero MIDI de entrada")
    sp.add_argument('--notes', help="Notas manuales 'n:ini:dur:vel,...'")
    sp.add_argument('--channel', type=int, default=None, help="Filtrar por canal MIDI")


def add_common_args(sp):
    sp.add_argument('--out', required=True, help="Fichero WAV de salida")
    sp.add_argument('--sr', type=int, default=SR_DEFAULT, help="Frecuencia de muestreo")


def main():
    p = argparse.ArgumentParser(
        prog='synth_forge.py',
        description="Motor de síntesis multi-paradigma (ver cabecera para detalles)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest='command', required=True)

    sp = sub.add_parser('sample', help="Síntesis por muestras (sampler SFZ-lite)")
    add_note_input_args(sp)
    sp.add_argument('--out', help="Fichero WAV de salida (no aplica con --make-demo-bank)")
    sp.add_argument('--sr', type=int, default=SR_DEFAULT)
    sp.add_argument('--keymap', help="keymap.json con las regiones")
    sp.add_argument('--make-demo-bank', metavar='DIR',
                     help="Genera un banco de muestras de prueba en DIR y termina")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser('waveguide', help="Modelado físico por guía de onda")
    add_note_input_args(sp)
    add_common_args(sp)
    sp.add_argument('--mode', choices=['plucked', 'bowed'], default='plucked')
    sp.add_argument('--brightness', type=float, default=0.5, help="[plucked] filtro de pérdidas")
    sp.add_argument('--bow-pressure', type=float, default=0.5)
    sp.add_argument('--bow-velocity', type=float, default=0.5)
    sp.add_argument('--vibrato', type=float, default=0.007, help="Profundidad de vibrato")
    sp.set_defaults(func=cmd_waveguide)

    sp = sub.add_parser('additive', help="Síntesis aditiva")
    add_note_input_args(sp)
    add_common_args(sp)
    sp.add_argument('--harmonics', default='1,0,0.6,0,0.3,0,0.15',
                     help="Pesos separados por comas (posición=armónico)")
    sp.set_defaults(func=cmd_additive)

    sp = sub.add_parser('fm', help="Síntesis FM de 2 operadores")
    add_note_input_args(sp)
    add_common_args(sp)
    sp.add_argument('--ratio', default='2:1', help="portadora:moduladora, ej. '2:1' o '1:1.41'")
    sp.add_argument('--index-attack', type=float, default=3.2)
    sp.add_argument('--index-sustain', type=float, default=1.8)
    sp.set_defaults(func=cmd_fm)

    sp = sub.add_parser('brass', help="Metales: tubo no lineal + banco modal")
    add_note_input_args(sp)
    add_common_args(sp)
    sp.add_argument('--brightness', type=float, default=0.4)
    sp.add_argument('--fm-attack', action='store_true', help="Añade chasquido FM inarmónico")
    sp.set_defaults(func=cmd_brass)

    sp = sub.add_parser('granular', help="Síntesis granular")
    add_common_args(sp)
    sp.add_argument('--source', default='ruido', help="'ruido' o ruta a un WAV")
    sp.add_argument('--center-freq', type=float, default=900.0, help="[fuente ruido]")
    sp.add_argument('--duration', type=float, default=4.0)
    sp.add_argument('--density', type=float, default=40.0, help="Granos por segundo")
    sp.add_argument('--grain-dur', type=float, default=0.04)
    sp.add_argument('--pos-spread', type=float, default=0.3)
    sp.add_argument('--pitch-spread', type=float, default=0.15)
    sp.set_defaults(func=cmd_granular)

    sp = sub.add_parser('demo', help="Renderiza una frase de ejemplo con todas las técnicas melódicas")
    sp.add_argument('--outdir', default='demo', help="Carpeta de salida")
    sp.add_argument('--sr', type=int, default=SR_DEFAULT)
    sp.set_defaults(func=cmd_demo)

    sp = sub.add_parser('info', help="Resumen de las técnicas de síntesis implementadas")
    sp.set_defaults(func=cmd_info)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
