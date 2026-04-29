#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        VOICE ALIGNER  v2.0                                   ║
║      Alineación de consonancia entre múltiples voces MIDI independientes     ║
║                                                                              ║
║  MEJORAS v2.0 respecto a v1.0:                                               ║
║    · Índice de importancia por nota (notas estructurales protegidas)         ║
║    · Métrica ponderada por duración del choque (no solo por presencia)       ║
║    · Score combinado vertical + coste melódico (preservación de contorno)   ║
║    · Detección de tonalidad por ventana deslizante + consenso entre voces   ║
║    · Nudge conjunto: optimización cartesiana de todos los pitches a la vez  ║
║    · Desplazamiento de zona para segmentos desalineados                      ║
║    · Ornamentación como estrategia de resolución (apoyatura, nota de paso)  ║
║    · Refinamiento iterativo hasta convergencia                               ║
║                                                                              ║
║  ESTRATEGIAS (cascada):                                                      ║
║    1. Shift global de voz                                                    ║
║    2. Shift local de nota individual                                         ║
║    3. Shift de zona (--allow-zone-shift)                                     ║
║    4. Nudge conjunto con optimización cartesiana                             ║
║    5. Sustitución por chord-tone                                             ║
║    6. Ornamentación (--allow-ornament)                                       ║
║    → Refinamiento iterativo (--max-iter)                                     ║
║                                                                              ║
║  USO:                                                                        ║
║    python voice_aligner.py voz1.mid voz2.mid [voz3.mid ...]                 ║
║    python voice_aligner.py *.mid --freeze-voice 0 --verbose --report        ║
║    python voice_aligner.py *.mid --melody-weight 0.4 --max-iter 5           ║
║    python voice_aligner.py *.mid --allow-zone-shift --allow-ornament        ║
║    python voice_aligner.py *.mid --key "D minor" --threshold 0.80           ║
║                                                                              ║
║  DEPENDENCIAS: mido, numpy                                                   ║
║  OPCIONALES:   music21 (mejora detección de tonalidad)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, json, math, copy, argparse, traceback
from itertools import product as iproduct
from pathlib import Path
from collections import defaultdict

import numpy as np

try:
    import mido
except ImportError:
    print("[ERROR] 'mido' no encontrado. Instala con: pip install mido")
    sys.exit(1)

try:
    from music21 import converter, key as m21key, pitch as m21pitch
    MUSIC21_OK = True
except ImportError:
    MUSIC21_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

PITCH_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

ALL_MODES = {
    'major':      [0,2,4,5,7,9,11],
    'minor':      [0,2,3,5,7,8,10],
    'harmonic':   [0,2,3,5,7,8,11],
    'dorian':     [0,2,3,5,7,9,10],
    'mixolydian': [0,2,4,5,7,9,10],
}

INTERVAL_CONSONANCE = {
    0:1.00, 7:0.95, 5:0.85, 4:0.90, 3:0.90,
    9:0.85, 8:0.82, 2:0.40, 10:0.35,
    11:0.15, 1:0.10, 6:0.20,
}

CHORD_INTERVALS = {
    'M':   [0,4,7],   'm':   [0,3,7],
    'M7':  [0,4,7,11],'m7':  [0,3,7,10],
    'dom7':[0,4,7,10],'dim': [0,3,6],
    'aug': [0,4,8],   'hd7': [0,3,6,10],
    'sus4':[0,5,7],   'sus2':[0,2,7],
}

VOICE_PROGRAMS = [0, 40, 41, 42, 73, 68, 56, 48]

# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MIDI
# ══════════════════════════════════════════════════════════════════════════════

def load_midi_voice(midi_path, verbose=False):
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir {midi_path}: {e}")

    tpb = mid.ticks_per_beat or 480
    tempo_us = 500_000
    ts_num, ts_den = 4, 4
    notes_by_ch = defaultdict(list)
    pending = {}

    for track in mid.tracks:
        abs_t = 0
        for msg in track:
            abs_t += msg.time
            if msg.type == 'set_tempo':
                tempo_us = msg.tempo
            elif msg.type == 'time_signature':
                ts_num, ts_den = msg.numerator, msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                pending[(msg.channel, msg.note)] = (abs_t, msg.velocity)
            elif msg.type in ('note_off', 'note_on'):
                k = (msg.channel, msg.note)
                if k in pending:
                    on_t, vel = pending.pop(k)
                    notes_by_ch[msg.channel].append(
                        [on_t/tpb, msg.note, max(0.05,(abs_t-on_t)/tpb), vel])

    for (ch, note),(on_t, vel) in pending.items():
        notes_by_ch[ch].append([on_t/tpb, note, 0.5, vel])

    if not notes_by_ch:
        raise RuntimeError(f"Sin notas en {midi_path}")

    main_ch = max(notes_by_ch, key=lambda c:(len(notes_by_ch[c]),
                  np.mean([n[1] for n in notes_by_ch[c]])))
    notes = sorted(notes_by_ch[main_ch], key=lambda n: n[0])
    tempo_bpm = round(60_000_000/max(tempo_us,1), 2)

    key_obj = None
    if MUSIC21_OK:
        try:
            key_obj = converter.parse(midi_path).analyze('key')
        except Exception:
            pass

    if verbose:
        pcs = [PITCH_NAMES[p] for p in sorted({n[1]%12 for n in notes})]
        print(f"      Canal {main_ch}: {len(notes)} notas, {tempo_bpm} BPM, "
              f"{ts_num}/{ts_den}" + (f", {key_obj}" if key_obj else ""))
        print(f"      PCs: {pcs}")

    return notes, tempo_bpm, tpb, (ts_num, ts_den), key_obj

# ══════════════════════════════════════════════════════════════════════════════
#  TONALIDAD: DETECCIÓN POR VENTANA + CONSENSO
# ══════════════════════════════════════════════════════════════════════════════

class KeyWindow:
    __slots__ = ('start','end','root','mode','scale_pcs','score')
    def __init__(self, s, e, r, m, sp, sc):
        self.start=s; self.end=e; self.root=r
        self.mode=m;  self.scale_pcs=sp; self.score=sc

def _best_key(pc_weight):
    total = sum(pc_weight.values()) or 1.0
    best_root, best_mode, best_sc = 0, 'major', -1.0
    for root in range(12):
        for mode, ivs in ALL_MODES.items():
            scale = {(root+i)%12 for i in ivs}
            sc = sum(pc_weight[pc] for pc in scale)/total
            if sc > best_sc:
                best_sc = sc; best_root = root; best_mode = mode
    return best_root, best_mode, best_sc

def detect_key_windows(all_notes, window_beats=4.0, hop_beats=2.0, verbose=False):
    if not all_notes:
        return []
    max_beat = max(n[0]+n[2] for notes in all_notes for n in notes)
    windows = []
    t = 0.0
    while t < max_beat:
        end = min(t+window_beats, max_beat)
        pc_w = defaultdict(float)
        for notes in all_notes:
            for onset,pitch,dur,vel in notes:
                ov = min(onset+dur,end)-max(onset,t)
                if ov > 0:
                    pc_w[pitch%12] += ov
        root, mode, sc = _best_key(pc_w)
        sp = [(root+i)%12 for i in ALL_MODES[mode]]
        windows.append(KeyWindow(t, end, root, mode, sp, sc))
        t += hop_beats

    if verbose:
        print(f"    {len(windows)} ventanas de tonalidad:")
        for w in windows[:5]:
            print(f"      [{w.start:.1f}-{w.end:.1f}] "
                  f"{PITCH_NAMES[w.root]} {w.mode}  ({w.score:.1%})")
        if len(windows) > 5:
            print(f"      … ({len(windows)-5} más)")
    return windows

def key_window_at(windows, beat):
    best = None
    for w in windows:
        if w.start <= beat < w.end:
            return w
        if w.start <= beat:
            best = w
    return best or (windows[0] if windows else None)

def consensus_scale_pcs(windows):
    if not windows:
        return list(range(12))
    pc_cnt = defaultdict(int)
    for w in windows:
        for pc in w.scale_pcs:
            pc_cnt[pc] += 1
    thr = len(windows)*0.5
    res = [pc for pc,cnt in pc_cnt.items() if cnt >= thr]
    return res if res else list(range(12))

def parse_key_string(key_str):
    parts = key_str.strip().split()
    tonic = parts[0] if parts else 'C'
    mode  = parts[1].lower() if len(parts)>1 else 'major'
    if MUSIC21_OK:
        try:
            return m21key.Key(tonic, mode)
        except Exception:
            pass
    class SimpleKey:
        def __init__(self,t,m):
            class T:
                name=t; pitchClass=PITCH_NAMES.index(t) if t in PITCH_NAMES else 0
            self.tonic=T(); self.mode=m
        def __str__(self): return f"{self.tonic.name} {self.mode}"
    return SimpleKey(tonic, mode)

def key_obj_to_windows(key_obj, max_beat):
    try:
        root = m21pitch.Pitch(key_obj.tonic.name).pitchClass
    except Exception:
        nm = key_obj.tonic.name
        root = PITCH_NAMES.index(nm) if nm in PITCH_NAMES else 0
    mode = key_obj.mode if hasattr(key_obj,'mode') else 'major'
    sp = [(root+i)%12 for i in ALL_MODES.get(mode, ALL_MODES['major'])]
    return [KeyWindow(0, max(max_beat,1), root, mode, sp, 1.0)]

# ══════════════════════════════════════════════════════════════════════════════
#  IMPORTANCIA POR NOTA
# ══════════════════════════════════════════════════════════════════════════════

def compute_note_importance(notes, beats_per_bar):
    n = len(notes)
    if n == 0:
        return []
    pitches   = [note[1] for note in notes]
    durations = [note[2] for note in notes]
    max_dur   = max(durations) or 1.0
    imp = []
    for i,(onset,pitch,dur,vel) in enumerate(notes):
        # Peso métrico
        pos = onset % beats_per_bar
        if pos < 0.05:                              metric = 1.0
        elif abs(pos - beats_per_bar/2) < 0.05:    metric = 0.75
        elif pos % 1.0 < 0.05:                     metric = 0.55
        else:                                        metric = 0.25
        # Duración relativa
        dur_f = (dur/max_dur)**0.5
        # Contorno
        neighbors = pitches[max(0,i-2):i] + pitches[i+1:min(n,i+3)]
        if neighbors:
            contour = 0.9 if (pitch>=max(neighbors) or pitch<=min(neighbors)) else 0.3
        else:
            contour = 0.5
        # Borde
        edge = 1.0 if i==0 or i==n-1 else 0.0
        v = max(edge, 0.35*metric + 0.35*dur_f + 0.30*contour)
        imp.append(round(min(1.0, v), 3))
    return imp

# ══════════════════════════════════════════════════════════════════════════════
#  MÉTRICA DE CONSONANCIA v2
# ══════════════════════════════════════════════════════════════════════════════

def interval_consonance(a, b):
    return INTERVAL_CONSONANCE.get(abs(a-b)%12, 0.40)

def metric_weight(beat_in_bar, beats_per_bar):
    pos = beat_in_bar % beats_per_bar
    if pos < 0.05:                          return 1.00
    if abs(pos - beats_per_bar/2) < 0.05:  return 0.75
    if pos % 1.0 < 0.05:                   return 0.55
    return 0.30

def build_event_grid(all_notes, resolution=0.125):
    times = set()
    for notes in all_notes:
        for onset,pitch,dur,vel in notes:
            times.add(round(onset,6))
            times.add(round(onset+dur,6))
    if not times:
        return []
    t_min, t_max = min(times), max(times)
    t = t_min
    while t <= t_max + resolution:
        times.add(round(t,6))
        t += resolution
    st = sorted(times)
    events = []
    for idx, t in enumerate(st):
        active = {}
        for vi,notes in enumerate(all_notes):
            for onset,pitch,dur,vel in notes:
                if onset <= t < onset+dur-1e-6:
                    active[vi] = pitch
                    break
        if len(active) >= 2:
            nxt = st[idx+1] if idx+1<len(st) else t+resolution
            events.append({'t':t,'active':active,'duration':max(0.0,nxt-t)})
    return events

def score_event_vertical(active):
    vs = list(active.values())
    if len(vs)<2: return 1.0
    scores = [interval_consonance(vs[i],vs[j])
              for i in range(len(vs)) for j in range(i+1,len(vs))]
    return float(np.mean(scores))

def melodic_smoothness(notes_orig, notes_mod):
    n = min(len(notes_orig), len(notes_mod))
    if n < 2: return 1.0
    penalties, total = 0, 0
    for i in range(1, n):
        d_orig = notes_orig[i][1] - notes_orig[i-1][1]
        d_mod  = notes_mod[i][1]  - notes_mod[i-1][1]
        if d_orig==0 and d_mod==0: continue
        total += 1
        if d_orig * d_mod < 0:
            penalties += 1.0
        elif abs(d_mod) - abs(d_orig) > 4:
            penalties += 0.5
    return 1.0 - (penalties/total) if total>0 else 1.0

def compute_consonance_metrics(all_notes, time_sig,
                               orig_notes_list=None, melody_weight=0.25,
                               verbose=False, label=""):
    beats_per_bar = time_sig[0]
    events = build_event_grid(all_notes)
    if not events:
        return {'consonance_score':1.0,'dissonance_rate':0.0,
                'melodic_cost':0.0,'combined_score':1.0,
                'n_events':0,'event_scores':[]}

    ws, tw, dc, ev_sc = [], 0.0, 0, []
    for ev in events:
        bib = ev['t'] % beats_per_bar
        mw  = metric_weight(bib, beats_per_bar)
        dur = max(ev['duration'], 0.01)
        w   = mw * dur                     # ponderación por duración
        s   = score_event_vertical(ev['active'])
        ws.append(s*w); tw += w; ev_sc.append(s)
        if s < 0.60: dc += 1

    cons  = sum(ws)/tw if tw>0 else 1.0
    dr    = dc/len(events)

    mel_smooth = 1.0
    if orig_notes_list is not None:
        mel_smooth = float(np.mean([
            melodic_smoothness(orig_notes_list[vi], all_notes[vi])
            for vi in range(len(all_notes))
        ]))
    mel_cost = 1.0 - mel_smooth
    combined = (1-melody_weight)*cons + melody_weight*mel_smooth

    if verbose and label:
        bar = lambda v: '▓'*int(v*20)+'░'*(20-int(v*20))
        print(f"\n  ┌─ Métricas {'['+label+']':─<50}─┐")
        print(f"  │  Consonance Score   : {cons:.4f}  ({bar(cons)})")
        print(f"  │  Melodic Smoothness : {mel_smooth:.4f}  ({bar(mel_smooth)})")
        print(f"  │  Combined Score     : {combined:.4f}  ({bar(combined)})")
        print(f"  │  Dissonance Rate    : {dr:.1%}  ({dc}/{len(events)} eventos)")
        bins   = [0,0.3,0.5,0.7,0.85,1.01]
        labels = ['Muy disonante','Disonante','Neutro','Consonante','Muy consonante']
        counts = [0]*5
        for s in ev_sc:
            for bi,b in enumerate(bins[1:]):
                if s<b: counts[bi]+=1; break
        print(f"  │  Distribución:")
        for lbl,cnt in zip(labels,counts):
            b2='█'*int(20*cnt/max(len(ev_sc),1))
            print(f"  │    {lbl:<15} {cnt:4d}  {b2}")
        print(f"  └{'─'*60}─┘")

    return {'consonance_score':round(cons,4),'dissonance_rate':round(dr,4),
            'melodic_cost':round(mel_cost,4),'combined_score':round(combined,4),
            'n_events':len(events),'event_scores':ev_sc}

def find_conflict_events(all_notes, time_sig, threshold=0.60):
    beats_per_bar = time_sig[0]
    out = []
    for ev in build_event_grid(all_notes):
        s = score_event_vertical(ev['active'])
        if s < threshold:
            voices = list(ev['active'].items())
            wp, wc = None, 1.0
            for i in range(len(voices)):
                for j in range(i+1,len(voices)):
                    c = interval_consonance(voices[i][1],voices[j][1])
                    if c<wc: wc=c; wp=(voices[i][0],voices[j][0])
            bib = ev['t'] % beats_per_bar
            out.append({'t':ev['t'],'duration':ev['duration'],'score':s,
                        'active':dict(ev['active']),'worst_pair':wp,
                        'metric_weight':metric_weight(bib,beats_per_bar)})
    return out

# ══════════════════════════════════════════════════════════════════════════════
#  E1: SHIFT GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

def try_global_time_shift(all_notes, vi, time_sig, orig, mw,
                          max_shift=0.5, steps=8, freeze=None, verbose=False):
    if freeze and vi in freeze: return 0.0
    base = compute_consonance_metrics(all_notes,time_sig,orig,mw)['combined_score']
    best_sh, best_sc = 0.0, base
    for sh in np.linspace(-max_shift, max_shift, steps*2+1):
        if abs(sh)<1e-4: continue
        shifted = [[n[0]+sh,n[1],n[2],n[3]] for n in all_notes[vi]]
        if any(n[0]<0 for n in shifted): continue
        orig_v = all_notes[vi]
        all_notes[vi] = shifted
        s = compute_consonance_metrics(all_notes,time_sig,orig,mw)['combined_score']
        all_notes[vi] = orig_v
        if s > best_sc+0.005: best_sc=s; best_sh=sh
    if abs(best_sh)>1e-4 and verbose:
        print(f"      Shift global voz {vi}: {best_sh:+.3f}b  "
              f"({base:.4f}→{best_sc:.4f})")
    return best_sh

# ══════════════════════════════════════════════════════════════════════════════
#  E2: SHIFT LOCAL
# ══════════════════════════════════════════════════════════════════════════════

def apply_local_time_shifts(all_notes, vi, conflicts, time_sig, imp,
                            max_shift=0.25, protect=0.7,
                            freeze=None, verbose=False):
    if freeze and vi in freeze: return all_notes[vi]
    notes = [list(n) for n in all_notes[vi]]
    bpb   = time_sig[0]
    cf_times = {cf['t'] for cf in conflicts if vi in cf['active']}
    modified = 0
    for i, note in enumerate(notes):
        if imp[i] >= protect: continue
        if not any(note[0]<=ct<note[0]+note[2] for ct in cf_times): continue
        best_sh, best_mw = 0.0, metric_weight(note[0]%bpb, bpb)
        for frac in [0.125, 0.25, min(0.5,max_shift)]:
            nmw = metric_weight((note[0]+frac)%bpb, bpb)
            if nmw < best_mw: best_mw=nmw; best_sh=frac
        if best_sh>1e-4:
            if i>0 and notes[i-1][0]+notes[i-1][2]>=note[0]-1e-4:
                notes[i-1][2] += best_sh
            note[0] += best_sh
            modified += 1
    if verbose and modified:
        print(f"      Shift local voz {vi}: {modified} notas")
    return notes

# ══════════════════════════════════════════════════════════════════════════════
#  E3: SHIFT DE ZONA
# ══════════════════════════════════════════════════════════════════════════════

def apply_zone_shift(all_notes, vi, conflicts, time_sig, imp,
                     max_shift=0.5, zone_beats=4.0,
                     freeze=None, verbose=False):
    if freeze and vi in freeze: return all_notes[vi]
    notes = [list(n) for n in all_notes[vi]]
    vc = [cf for cf in conflicts if vi in cf['active']]
    if not vc: return notes
    max_beat = max(n[0]+n[2] for n in notes)
    modified = 0
    for zs in np.arange(0, max_beat, zone_beats):
        ze = zs + zone_beats
        zc = [cf for cf in vc if zs<=cf['t']<ze]
        if len(zc)<3: continue
        zni = [i for i,n in enumerate(notes) if zs<=n[0]<ze]
        if not zni: continue
        if max(imp[i] for i in zni) >= 0.85: continue
        best_sh, best_sc = 0.0, -1.0
        for sh in np.linspace(-max_shift, max_shift, 9):
            if abs(sh)<1e-4: continue
            trial = [list(n) for n in notes]
            for i in zni: trial[i][0] = max(0, trial[i][0]+sh)
            fi = zni[0]
            if fi>0:
                gap = trial[fi][0]-(trial[fi-1][0]+trial[fi-1][2])
                if gap>0: trial[fi-1][2] += gap
            # score simplificado: consonancia con otras voces en la zona
            zone_pitches = [trial[i][1] for i in zni[:4]]
            other_pitches = []
            for vj,vn in enumerate(all_notes):
                if vj==vi: continue
                for n in vn:
                    if zs<=n[0]<ze: other_pitches.append(n[1])
            if not other_pitches: continue
            sc = float(np.mean([interval_consonance(p,op)
                                 for p in zone_pitches for op in other_pitches[:4]]))
            if sc > best_sc: best_sc=sc; best_sh=sh
        if abs(best_sh)>1e-4 and best_sc>0.6:
            for i in zni: notes[i][0]=max(0,notes[i][0]+best_sh)
            fi=zni[0]
            if fi>0:
                gap=notes[fi][0]-(notes[fi-1][0]+notes[fi-1][2])
                if gap>0: notes[fi-1][2]+=gap
            modified+=1
            if verbose:
                print(f"      Zona shift voz {vi}: [{zs:.1f}-{ze:.1f}] "
                      f"sh={best_sh:+.3f}")
    return notes

# ══════════════════════════════════════════════════════════════════════════════
#  E4: NUDGE CONJUNTO
# ══════════════════════════════════════════════════════════════════════════════

def joint_pitch_nudge(all_notes, time_sig, conflicts, windows,
                      importances, orig_notes_list, max_shift=2,
                      melody_weight=0.25, protect=0.7,
                      freeze=None, verbose=False):
    mod = [[list(n) for n in voice] for voice in all_notes]
    total_mod = defaultdict(int)
    processed = set()

    # Ordenar conflictos: primero los más graves y en tiempos fuertes
    sorted_cf = sorted(conflicts,
                       key=lambda c: -(c['metric_weight']*c['duration']*(1-c['score'])))

    for cf in sorted_cf:
        t = round(cf['t'], 4)
        if t in processed: continue
        processed.add(t)

        active   = cf['active']
        free_vis = [vi for vi in active if not (freeze and vi in freeze)]
        if not free_vis: continue

        # Construir candidatos por voz
        cand_by_vi = {}
        ni_by_vi   = {}
        for vi in free_vis:
            ni_t = None
            for ni,note in enumerate(mod[vi]):
                if note[0]<=t<note[0]+note[2]-1e-6:
                    ni_t = ni; break
            if ni_t is None: continue
            imp_v = importances[vi][ni_t] if ni_t<len(importances[vi]) else 0.5
            if imp_v >= protect:
                cand_by_vi[vi] = [mod[vi][ni_t][1]]
            else:
                w = key_window_at(windows, t)
                allowed = set(w.scale_pcs) if w else set(range(12))
                orig_p = mod[vi][ni_t][1]
                cands = [orig_p]
                for d in range(1, max_shift+1):
                    for c in [orig_p+d, orig_p-d]:
                        if 21<=c<=108 and (c%12 in allowed):
                            cands.append(c)
                cand_by_vi[vi] = cands
            ni_by_vi[vi] = ni_t

        if not cand_by_vi: continue

        free_list  = list(cand_by_vi.keys())
        cand_lists = [cand_by_vi[vi] for vi in free_list]
        ni_list    = [ni_by_vi[vi] for vi in free_list]
        frozen_act = {vi:active[vi] for vi in active if vi not in free_list}

        # Producto cartesiano (máx 500 combinaciones)
        total = 1
        for cl in cand_lists: total *= len(cl)
        if total <= 500:
            combos = list(iproduct(*cand_lists))
        else:
            rng = np.random.default_rng(int(t*1000)%2**32)
            combos = [tuple(int(rng.choice(cl)) for cl in cand_lists)
                      for _ in range(500)]

        best_combo = tuple(mod[vi][ni_list[idx]][1]
                           for idx,vi in enumerate(free_list))
        best_sc = -1.0

        for combo in combos:
            test = dict(frozen_act)
            for idx,vi in enumerate(free_list):
                test[vi] = combo[idx]
            ps = list(test.values())
            vert = float(np.mean([interval_consonance(ps[i],ps[j])
                                  for i in range(len(ps))
                                  for j in range(i+1,len(ps))])) if len(ps)>1 else 1.0
            # Penalización melódica
            mel_pen = 0.0
            for idx,vi in enumerate(free_list):
                ni  = ni_list[idx]
                op  = all_notes[vi][ni][1]   # pitch original
                np_ = combo[idx]
                if ni>0:
                    prev_o = all_notes[vi][ni-1][1]
                    prev_m = mod[vi][ni-1][1]
                    do = op - prev_o; dm = np_ - prev_m
                    if do*dm<0:           mel_pen+=0.4
                    elif abs(dm)-abs(do)>4: mel_pen+=0.2
            sc = (1-melody_weight)*vert - melody_weight*mel_pen
            if sc > best_sc: best_sc=sc; best_combo=combo

        for idx,vi in enumerate(free_list):
            ni  = ni_list[idx]
            old = mod[vi][ni][1]
            new = best_combo[idx]
            if new!=old:
                if verbose:
                    print(f"        voz {vi} t={t:.3f} "
                          f"{PITCH_NAMES[old%12]}{old//12-1}"
                          f"→{PITCH_NAMES[new%12]}{new//12-1}"
                          f"  sc={best_sc:.2f}")
                mod[vi][ni][1] = new
                total_mod[vi] += 1

    if verbose:
        for vi,cnt in total_mod.items():
            print(f"      Nudge conjunto voz {vi}: {cnt} notas")
    return mod

# ══════════════════════════════════════════════════════════════════════════════
#  E5: CHORD-TONE SUBSTITUTION
# ══════════════════════════════════════════════════════════════════════════════

def infer_local_chord(pitches):
    if not pitches: return None
    pcs = list({p%12 for p in pitches})
    if len(pcs)<2: return (pcs[0],'M')
    best_r,best_q,best_sc = pcs[0],'M',-1.0
    for r in pcs:
        ivs = {(p-r)%12 for p in pcs}
        for q,qi in CHORD_INTERVALS.items():
            sc = len(ivs&set(qi))/max(len(qi),1) - 0.15*len(ivs-set(qi))
            if sc>best_sc: best_sc=sc; best_r=r; best_q=q
    return (best_r, best_q)

def chord_tone_substitution(all_notes, time_sig, conflicts, windows,
                            importances, protect=0.7,
                            freeze=None, verbose=False):
    mod = [[list(n) for n in voice] for voice in all_notes]
    total_mod = defaultdict(int)

    others_at = defaultdict(lambda: defaultdict(list))
    for cf in conflicts:
        t = round(cf['t'],4)
        for vi,p in cf['active'].items():
            for vj,pj in cf['active'].items():
                if vi!=vj: others_at[vi][t].append(pj)

    for vi,notes in enumerate(mod):
        if freeze and vi in freeze: continue
        for ni,note in enumerate(notes):
            onset = note[0]
            if importances[vi][ni] >= protect if ni<len(importances[vi]) else False:
                continue
            rel = []
            for t_k,ops in others_at[vi].items():
                if onset<=t_k<onset+note[2]: rel.extend(ops)
            if not rel: continue
            ci = infer_local_chord(rel)
            if ci is None: continue
            root_pc,quality = ci
            cpcs = {(root_pc+i)%12 for i in CHORD_INTERVALS.get(quality,[0,4,7])}
            w   = key_window_at(windows, onset)
            sp  = set(w.scale_pcs) if w else set(range(12))
            p   = note[1]
            cur = float(np.mean([interval_consonance(p,op) for op in rel]))
            best_p, best_c = p, cur
            for d in range(-4,5):
                c = p+d
                if not (21<=c<=108): continue
                if (c%12) not in cpcs: continue
                if (c%12) not in sp:   continue
                s = float(np.mean([interval_consonance(c,op) for op in rel]))
                if s > best_c+0.05: best_c=s; best_p=c
            if best_p!=p:
                if verbose:
                    print(f"        voz {vi} t={onset:.3f} "
                          f"{PITCH_NAMES[p%12]}→{PITCH_NAMES[best_p%12]}"
                          f"  [{PITCH_NAMES[root_pc]}{quality}]"
                          f"  {cur:.2f}→{best_c:.2f}")
                note[1]=best_p; total_mod[vi]+=1

    if verbose:
        for vi,cnt in total_mod.items():
            print(f"      Chord-tone sub voz {vi}: {cnt} notas")
    return mod

# ══════════════════════════════════════════════════════════════════════════════
#  E6: ORNAMENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def apply_ornaments(all_notes, time_sig, conflicts, windows,
                    importances, protect=0.7,
                    freeze=None, verbose=False):
    bpb = time_sig[0]
    mod = [[list(n) for n in voice] for voice in all_notes]
    total_ins = defaultdict(int)

    # Apoyaturas en tiempos fuertes con disonancia fuerte
    strong = [cf for cf in conflicts
              if cf['metric_weight']>=0.75 and cf['score']<0.50]

    for cf in strong:
        t = cf['t']
        for vi in cf['active']:
            if freeze and vi in freeze: continue
            notes = mod[vi]
            ni = next((i for i,n in enumerate(notes)
                       if n[0]<=t<n[0]+n[2]-1e-6), None)
            if ni is None: continue
            imp = importances[vi][ni] if ni<len(importances[vi]) else 0.5
            if imp >= protect: continue
            note = notes[ni]
            if note[2]<0.25: continue
            pitch = note[1]
            if ni+1<len(notes):
                target = notes[ni+1][1]
                if abs(target-pitch)==1: continue
                others = [cf['active'][vj] for vj in cf['active'] if vj!=vi]
                if abs(target-pitch)<=2:
                    apog = pitch+(1 if target>pitch else -1)
                else:
                    uc = np.mean([interval_consonance(pitch+1,op) for op in others]) if others else 0
                    dc = np.mean([interval_consonance(pitch-1,op) for op in others]) if others else 0
                    apog = pitch+1 if uc>dc else pitch-1
                if not (21<=apog<=108): continue
                half = note[2]*0.5
                note[2] = half
                notes.insert(ni+1, [note[0]+half, apog, half*0.9, max(40,note[3]-10)])
                total_ins[vi]+=1
                if verbose:
                    print(f"        voz {vi} t={t:.3f} "
                          f"apoyatura {PITCH_NAMES[apog%12]}→{PITCH_NAMES[pitch%12]}")

    # Notas de paso en saltos grandes conflictivos
    for vi,notes in enumerate(mod):
        if freeze and vi in freeze: continue
        i=0
        while i<len(notes)-1:
            note, nxt = notes[i], notes[i+1]
            if abs(nxt[1]-note[1])>=3 and note[2]>=0.5:
                cf_here = any(
                    note[0]<=cf['t']<note[0]+note[2]
                    and vi in cf['active'] and cf['score']<0.50
                    for cf in strong)
                if cf_here:
                    pp = (note[1]+nxt[1])//2
                    if 21<=pp<=108:
                        half = note[2]*0.5
                        note[2]=half
                        notes.insert(i+1,[note[0]+half,pp,half*0.9,max(40,note[3]-15)])
                        total_ins[vi]+=1
                        i+=2; continue
            i+=1

    if verbose:
        for vi,cnt in total_ins.items():
            print(f"      Ornamentos voz {vi}: {cnt} insertados")
    return mod

# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE DE ALINEACIÓN CON REFINAMIENTO ITERATIVO
# ══════════════════════════════════════════════════════════════════════════════

def align_voices(all_notes, time_sig, windows, orig_notes_list,
                 threshold=0.72, melody_weight=0.25, strategy='auto',
                 freeze=None, max_pitch_shift=2, max_time_shift=0.5,
                 allow_zone_shift=False, allow_ornament=False,
                 protect=0.7, max_iter=3, verbose=False):

    logs = {vi:[] for vi in range(len(all_notes))}

    def cur_score():
        return compute_consonance_metrics(
            all_notes, time_sig, orig_notes_list, melody_weight
        )['combined_score']

    def log_op(vi, name, **kw):
        logs[vi].append({'strategy':name, **kw})

    score = cur_score()
    if score >= threshold:
        if verbose:
            print(f"\n  ✓ Score inicial ({score:.4f}) ≥ umbral. Sin ajustes.")
        return logs

    importances = [compute_note_importance(notes, time_sig[0])
                   for notes in all_notes]

    for it in range(1, max_iter+1):
        if verbose:
            print(f"\n  {'─'*28} Iteración {it}/{max_iter} "
                  f"(score={score:.4f}) {'─'*28}")
        prev = score

        # E1 — Shift global
        if strategy in ('auto','time'):
            if verbose: print(f"\n  [E1] Shift global…")
            for vi in range(len(all_notes)):
                sh = try_global_time_shift(
                    all_notes, vi, time_sig, orig_notes_list, melody_weight,
                    max_shift=min(max_time_shift,1.0), freeze=freeze, verbose=verbose)
                if abs(sh)>1e-4:
                    all_notes[vi] = [[n[0]+sh,n[1],n[2],n[3]] for n in all_notes[vi]]
                    log_op(vi,'global_time_shift', shift_beats=round(sh,4))
            score = cur_score()
            if verbose: print(f"      → score={score:.4f}")
            if score>=threshold: break

        # E2 — Shift local
        if strategy in ('auto','time'):
            if verbose: print(f"\n  [E2] Shift local…")
            cfs = find_conflict_events(all_notes, time_sig)
            if verbose: print(f"      Conflictos: {len(cfs)}")
            for vi in range(len(all_notes)):
                nn = apply_local_time_shifts(
                    all_notes, vi, cfs, time_sig, importances[vi],
                    max_shift=min(max_time_shift*0.5, 0.25),
                    protect=protect, freeze=freeze, verbose=verbose)
                mv = sum(1 for a,b in zip(all_notes[vi],nn) if a[0]!=b[0])
                if mv: all_notes[vi]=nn; log_op(vi,'local_time_shift',notes_moved=mv)
            score = cur_score()
            if verbose: print(f"      → score={score:.4f}")
            if score>=threshold: break

        # E3 — Shift de zona
        if allow_zone_shift and strategy in ('auto','time'):
            if verbose: print(f"\n  [E3] Shift de zona…")
            cfs = find_conflict_events(all_notes, time_sig)
            for vi in range(len(all_notes)):
                nn = apply_zone_shift(
                    all_notes, vi, cfs, time_sig, importances[vi],
                    max_shift=max_time_shift, freeze=freeze, verbose=verbose)
                ch = sum(1 for a,b in zip(all_notes[vi],nn) if a[0]!=b[0])
                if ch: all_notes[vi]=nn; log_op(vi,'zone_shift',zones=ch)
            score = cur_score()
            if verbose: print(f"      → score={score:.4f}")
            if score>=threshold: break

        # E4 — Nudge conjunto
        if strategy in ('auto','pitch','time'):
            if verbose: print(f"\n  [E4] Nudge conjunto (±{max_pitch_shift} st)…")
            cfs = find_conflict_events(all_notes, time_sig)
            new_all = joint_pitch_nudge(
                all_notes, time_sig, cfs, windows, importances, orig_notes_list,
                max_shift=max_pitch_shift, melody_weight=melody_weight,
                protect=protect, freeze=freeze, verbose=verbose)
            for vi in range(len(all_notes)):
                ch = sum(1 for a,b in zip(all_notes[vi],new_all[vi]) if a[1]!=b[1])
                if ch: all_notes[vi]=new_all[vi]; log_op(vi,'joint_pitch_nudge',notes=ch)
            score = cur_score()
            if verbose: print(f"      → score={score:.4f}")
            if score>=threshold: break

        # E5 — Chord-tone substitution
        if strategy in ('auto','chord','pitch'):
            if verbose: print(f"\n  [E5] Chord-tone substitution…")
            cfs = find_conflict_events(all_notes, time_sig, threshold=0.65)
            new_all = chord_tone_substitution(
                all_notes, time_sig, cfs, windows, importances,
                protect=protect, freeze=freeze, verbose=verbose)
            for vi in range(len(all_notes)):
                ch = sum(1 for a,b in zip(all_notes[vi],new_all[vi]) if a[1]!=b[1])
                if ch: all_notes[vi]=new_all[vi]; log_op(vi,'chord_tone_sub',notes=ch)
            score = cur_score()
            if verbose: print(f"      → score={score:.4f}")
            if score>=threshold: break

        # E6 — Ornamentación
        if allow_ornament and strategy in ('auto','ornament'):
            if verbose: print(f"\n  [E6] Ornamentación…")
            cfs = find_conflict_events(all_notes, time_sig, threshold=0.50)
            new_all = apply_ornaments(
                all_notes, time_sig, cfs, windows, importances,
                protect=protect, freeze=freeze, verbose=verbose)
            for vi in range(len(all_notes)):
                ins = len(new_all[vi])-len(all_notes[vi])
                if ins:
                    all_notes[vi]=new_all[vi]
                    importances[vi]=compute_note_importance(all_notes[vi],time_sig[0])
                    log_op(vi,'ornament',inserted=ins)
            score = cur_score()
            if verbose: print(f"      → score={score:.4f}")
            if score>=threshold: break

        # Convergencia
        if score-prev < 0.004:
            if verbose: print(f"\n  Convergencia (Δ={score-prev:.4f})")
            break

    return logs

# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA MIDI
# ══════════════════════════════════════════════════════════════════════════════

def notes_to_track(notes, ch, program, name, tpb=480):
    def tt(b): return max(1, int(round(b*tpb)))
    evs=[]
    for onset,pitch,dur,vel in notes:
        p=max(0,min(127,int(pitch))); v=max(1,min(127,int(vel)))
        evs.append((tt(onset),'on',p,v))
        evs.append((tt(onset+max(0.05,dur)),'off',p,0))
    evs.sort(key=lambda x:(x[0],0 if x[1]=='off' else 1))
    trk=mido.MidiTrack()
    trk.append(mido.MetaMessage('track_name',name=name,time=0))
    trk.append(mido.Message('program_change',channel=ch%16,program=program,time=0))
    prev=0
    for at,kind,p,v in evs:
        dt=max(0,at-prev)
        trk.append(mido.Message('note_on' if kind=='on' else 'note_off',
                                channel=ch%16,note=p,velocity=v,time=dt))
        prev=at
    trk.append(mido.MetaMessage('end_of_track',time=0))
    return trk

def _hdr_track(tempo_bpm, time_sig):
    tu=int(60_000_000/max(tempo_bpm,1))
    hdr=mido.MidiTrack()
    hdr.append(mido.MetaMessage('set_tempo',tempo=tu,time=0))
    hdr.append(mido.MetaMessage('time_signature',
               numerator=time_sig[0],denominator=4,
               clocks_per_click=24,notated_32nd_notes_per_beat=8,time=0))
    hdr.append(mido.MetaMessage('end_of_track',time=0))
    return hdr

def write_voice_midi(notes, tempo_bpm, time_sig, path, tpb=480, vi=0):
    mid=mido.MidiFile(type=1,ticks_per_beat=tpb)
    mid.tracks.append(_hdr_track(tempo_bpm, time_sig))
    mid.tracks.append(notes_to_track(notes, vi%16,
                      VOICE_PROGRAMS[vi%len(VOICE_PROGRAMS)],
                      f'Voice_{vi}', tpb))
    mid.save(path)

def write_combined_midi(all_notes, tempo_bpm, time_sig, path, tpb=480, names=None):
    mid=mido.MidiFile(type=1,ticks_per_beat=tpb)
    mid.tracks.append(_hdr_track(tempo_bpm, time_sig))
    for vi,notes in enumerate(all_notes):
        nm = names[vi] if names and vi<len(names) else f'Voice_{vi}'
        mid.tracks.append(notes_to_track(notes, vi%16,
                          VOICE_PROGRAMS[vi%len(VOICE_PROGRAMS)], nm, tpb))
    mid.save(path)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_voice_aligner(midi_paths,
                      threshold=0.72, strategy='auto',
                      freeze_voices=None, max_pitch_shift=2,
                      max_time_shift=0.5, melody_weight=0.25,
                      max_iter=3, window_beats=4.0,
                      allow_zone_shift=False, allow_ornament=False,
                      protect_threshold=0.7,
                      key_override=None, tempo_override=None,
                      out_dir=None, report=False, verbose=False):

    n = len(midi_paths)
    freeze_set = set(freeze_voices) if freeze_voices else set()

    print(f"\n{'═'*70}")
    print(f"  VOICE ALIGNER  v2.0  ·  {n} voces")
    print(f"{'═'*70}")
    for k,v in [('Estrategia',strategy),('Umbral',threshold),
                ('Peso melódico',melody_weight),
                ('Max pitch shift',f'{max_pitch_shift} st'),
                ('Max time shift', f'{max_time_shift} beats'),
                ('Max iteraciones',max_iter),
                ('Protección notas',f'imp ≥ {protect_threshold}')]:
        print(f"  {k:<20}: {v}")
    if allow_zone_shift: print(f"  Shift de zona       : activado")
    if allow_ornament:   print(f"  Ornamentación       : activada")
    if freeze_set:       print(f"  Voces fijadas       : {sorted(freeze_set)}")

    # ── Carga ────────────────────────────────────────────────────────────────
    print(f"\n  [1/5] Cargando voces…")
    all_notes,tpbs,tempos,time_sigs,stems = [],[],[],[],[]
    for vi,path in enumerate(midi_paths):
        print(f"    Voz {vi}: {Path(path).name}")
        notes,tempo,tpb,ts,_ = load_midi_voice(path, verbose=verbose)
        all_notes.append(notes); tpbs.append(tpb)
        tempos.append(tempo);    time_sigs.append(ts)
        stems.append(Path(path).stem)

    tempo_bpm = tempo_override or tempos[0]
    time_sig  = time_sigs[0]
    tpb_out   = tpbs[0]
    if out_dir is None:
        out_dir = str(Path(midi_paths[0]).parent)
    os.makedirs(out_dir, exist_ok=True)

    orig_notes_list = [[list(n) for n in voice] for voice in all_notes]
    max_beat = max(n[0]+n[2] for voice in all_notes for n in voice)

    # ── Tonalidad ─────────────────────────────────────────────────────────────
    print(f"\n  [2/5] Detección de tonalidad…")
    if key_override:
        kobj = parse_key_string(key_override)
        windows = key_obj_to_windows(kobj, max_beat)
        print(f"    Tonalidad forzada : {kobj}")
    else:
        windows = detect_key_windows(all_notes, window_beats, verbose=verbose)
        if not windows:
            windows = [KeyWindow(0,max_beat,0,'major',[0,2,4,5,7,9,11],1.0)]
        cons_pcs = consensus_scale_pcs(windows)
        mw = windows[len(windows)//2]
        print(f"    Tonalidad central : {PITCH_NAMES[mw.root]} {mw.mode}")
        print(f"    PCs de consenso   : {[PITCH_NAMES[p] for p in sorted(cons_pcs)]}")
        kobj = parse_key_string(f"{PITCH_NAMES[mw.root]} {mw.mode}")
    print(f"    Tempo: {tempo_bpm} BPM  |  Compás: {time_sig[0]}/{time_sig[1]}")
    print(f"    Notas: {sum(len(n) for n in all_notes)} total")

    # ── Importancia ───────────────────────────────────────────────────────────
    print(f"\n  [3/5] Calculando importancia…")
    importances = [compute_note_importance(notes, time_sig[0])
                   for notes in all_notes]
    if verbose:
        for vi,imp in enumerate(importances):
            prot = sum(1 for x in imp if x>=protect_threshold)
            print(f"    Voz {vi}: {prot}/{len(imp)} notas protegidas")

    # ── Análisis inicial ──────────────────────────────────────────────────────
    print(f"\n  [4/5] Análisis y alineación…")
    mb = compute_consonance_metrics(all_notes, time_sig, orig_notes_list,
                                    melody_weight, verbose=verbose, label="ANTES")
    print(f"\n  ┌─ Score inicial {'─'*52}─┐")
    print(f"  │  Combined Score   : {mb['combined_score']:.4f}")
    print(f"  │  Consonance Score : {mb['consonance_score']:.4f}")
    print(f"  │  Dissonance Rate  : {mb['dissonance_rate']:.1%}")
    print(f"  └{'─'*57}─┘")

    # ── Alineación ───────────────────────────────────────────────────────────
    align_logs = align_voices(
        all_notes, time_sig, windows, orig_notes_list,
        threshold=threshold, melody_weight=melody_weight,
        strategy=strategy, freeze=freeze_set,
        max_pitch_shift=max_pitch_shift, max_time_shift=max_time_shift,
        allow_zone_shift=allow_zone_shift, allow_ornament=allow_ornament,
        protect=protect_threshold, max_iter=max_iter, verbose=verbose)

    # ── Análisis final ────────────────────────────────────────────────────────
    ma = compute_consonance_metrics(all_notes, time_sig, orig_notes_list,
                                    melody_weight, verbose=verbose, label="DESPUÉS")
    dc   = ma['consonance_score']-mb['consonance_score']
    dcb  = ma['combined_score']-mb['combined_score']
    ddr  = mb['dissonance_rate']-ma['dissonance_rate']
    ok   = ma['combined_score']>=threshold

    print(f"\n  ┌─ Resumen de mejora {'─'*48}─┐")
    print(f"  │  Combined Score   : {mb['combined_score']:.4f} → {ma['combined_score']:.4f}  ({dcb:+.4f})")
    print(f"  │  Consonance Score : {mb['consonance_score']:.4f} → {ma['consonance_score']:.4f}  ({dc:+.4f})")
    print(f"  │  Dissonance Rate  : {mb['dissonance_rate']:.1%} → {ma['dissonance_rate']:.1%}  ({-ddr:+.1%})")
    print(f"  │  Melodic Cost     : {ma['melodic_cost']:.4f}")
    print(f"  │  Umbral {threshold}      : {'✓ ALCANZADO' if ok else '✗ no alcanzado'}")
    print(f"  │")
    for vi,entries in align_logs.items():
        ops  = [e['strategy'] for e in entries]
        mark = " [fija]" if vi in freeze_set else ""
        print(f"  │  Voz {vi} ({stems[vi]}){mark}: "
              f"{', '.join(ops) if ops else 'sin cambios'}")
    print(f"  └{'─'*57}─┘")

    # ── Escritura ─────────────────────────────────────────────────────────────
    print(f"\n  [5/5] Guardando archivos…")
    out_files = []
    for vi,(notes,stem) in enumerate(zip(all_notes,stems)):
        p = os.path.join(out_dir, f"{stem}.aligned.mid")
        write_voice_midi(notes, tempo_bpm, time_sig, p, tpb_out, vi)
        out_files.append(p); print(f"    → {stem}.aligned.mid")

    cp = os.path.join(out_dir,"aligned_combined.mid")
    write_combined_midi(all_notes, tempo_bpm, time_sig, cp, tpb_out, stems)
    print(f"    → aligned_combined.mid")

    rdata = {
        'version':'2.0','voices':[str(p) for p in midi_paths],
        'key':str(kobj),'tempo_bpm':tempo_bpm,'time_sig':list(time_sig),
        'threshold':threshold,'strategy':strategy,
        'melody_weight':melody_weight,'protect_threshold':protect_threshold,
        'freeze_voices':sorted(freeze_set),
        'metrics_before':{k:v for k,v in mb.items() if k!='event_scores'},
        'metrics_after': {k:v for k,v in ma.items() if k!='event_scores'},
        'improvement_combined':round(dcb,4),
        'improvement_consonance':round(dc,4),
        'threshold_reached':bool(ok),
        'align_log':{str(vi):entries for vi,entries in align_logs.items()},
        'output_files':out_files+[cp],
    }
    if report:
        rp = os.path.join(out_dir,"aligned_report.json")
        with open(rp,'w',encoding='utf-8') as f:
            json.dump(rdata,f,indent=2,ensure_ascii=False)
        print(f"    → aligned_report.json")

    print(f"\n{'═'*70}")
    print(f"  Completado.  Combined: "
          f"{mb['combined_score']:.4f} → {ma['combined_score']:.4f}  ({dcb:+.4f})")
    print(f"{'═'*70}\n")
    return rdata

# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(
        description='VOICE ALIGNER v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Estrategias (--strategy):
  auto      Cascada completa con refinamiento iterativo [default]
  time      Solo desplazamiento temporal
  pitch     Nudge conjunto + chord-tone
  chord     Solo chord-tone substitution
  ornament  Solo ornamentación

Ejemplos:
  python voice_aligner.py v1.mid v2.mid v3.mid --verbose
  python voice_aligner.py *.mid --freeze-voice 0 --report
  python voice_aligner.py *.mid --melody-weight 0.4 --max-iter 5
  python voice_aligner.py *.mid --allow-zone-shift --allow-ornament
  python voice_aligner.py *.mid --key "D minor" --threshold 0.80
        """)
    p.add_argument('voices', nargs='+')
    p.add_argument('--threshold',        type=float, default=0.72)
    p.add_argument('--strategy',         default='auto',
                   choices=['auto','time','pitch','chord','ornament'])
    p.add_argument('--freeze-voice',     type=int, nargs='+', default=None, metavar='N')
    p.add_argument('--max-pitch-shift',  type=int,   default=2,    metavar='N')
    p.add_argument('--max-time-shift',   type=float, default=0.5,  metavar='F')
    p.add_argument('--melody-weight',    type=float, default=0.25, metavar='F',
                   help='Peso del coste melódico 0-1 (default: 0.25)')
    p.add_argument('--max-iter',         type=int,   default=3,    metavar='N',
                   help='Máx. iteraciones de refinamiento (default: 3)')
    p.add_argument('--window-beats',     type=float, default=4.0,  metavar='F',
                   help='Ventana de detección de tonalidad en beats (default: 4.0)')
    p.add_argument('--allow-zone-shift', action='store_true')
    p.add_argument('--allow-ornament',   action='store_true')
    p.add_argument('--protect-threshold',type=float, default=0.7,  metavar='F',
                   help='Umbral de importancia para proteger notas (default: 0.7)')
    p.add_argument('--key',              default=None, metavar='KEY')
    p.add_argument('--tempo',            type=float, default=None)
    p.add_argument('--out-dir',          default=None)
    p.add_argument('--report',           action='store_true')
    p.add_argument('--verbose',          action='store_true')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(args.voices)<2:
        print("[ERROR] Se necesitan ≥2 voces MIDI.")
        sys.exit(1)
    for path in args.voices:
        if not os.path.isfile(path):
            print(f"[ERROR] No encontrado: {path}")
            sys.exit(1)
    try:
        run_voice_aligner(
            midi_paths=args.voices,
            threshold=args.threshold, strategy=args.strategy,
            freeze_voices=args.freeze_voice,
            max_pitch_shift=args.max_pitch_shift,
            max_time_shift=args.max_time_shift,
            melody_weight=args.melody_weight,
            max_iter=args.max_iter, window_beats=args.window_beats,
            allow_zone_shift=args.allow_zone_shift,
            allow_ornament=args.allow_ornament,
            protect_threshold=args.protect_threshold,
            key_override=args.key, tempo_override=args.tempo,
            out_dir=args.out_dir, report=args.report, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n[Interrumpido]"); sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR FATAL] {e}")
        if '--verbose' in sys.argv: traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
