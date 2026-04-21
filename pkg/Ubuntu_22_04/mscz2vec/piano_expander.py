#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      PIANO EXPANDER  v2.0                                    ║
║     Expansión orquestal de bocetos pianísticos / reducciones MIDI           ║
║                                                                              ║
║  Lee un MIDI de piano y genera un MIDI multitracks con orquestación         ║
║  idiomática. Cada instrumento genera su propia línea independiente —        ║
║  no una copia transpuesta del piano.                                         ║
║                                                                              ║
║  MEJORAS v2.0:                                                               ║
║  [A] Arquitectura línea-por-línea: cada instrumento genera su propio        ║
║      material de principio a fin, con coherencia interna                    ║
║  [B] Generadores rítmicos por familia y arquetipo: cuerdas hacen arpegios, ║
║      maderas hacen colchones, metales entran en clímax, etc.               ║
║  [C] Contrapunto real con mirada hacia adelante: la contramelodía tiene     ║
║      su propio arco y clímax, evita paralelas de 5ª/8ª                    ║
║  [D] Curva de tensión global: controla qué instrumentos entran y cuándo,   ║
║      reservando metales para el clímax formal                              ║
║                                                                              ║
║  PIPELINE:                                                                   ║
║  [1] Análisis       — melodía, bajo, acordes, forma, tonalidad              ║
║  [2] Curva tensión  — arco global sobre toda la obra                        ║
║  [3] Textura        — 5 arquetipos por sección                             ║
║  [4] Líneas instr.  — generador dedicado por rol de instrumento             ║
║  [5] Articulación   — duración/dinámica por arquetipo y familia             ║
║  [6] CC envelopes   — CC1/CC11 desde curva de tensión global               ║
║  [7] Salida         — MIDI multitracks + informe + fingerprint              ║
║                                                                              ║
║  PLANTILLAS: chamber | strings | full | <archivo.json>                      ║
║  ESTILOS:    auto | romantic | baroque | impressionist | cinematic | chamber ║
║                                                                              ║
║  USO:                                                                        ║
║    python piano_expander.py boceto.mid                                       ║
║    python piano_expander.py boceto.mid --template full --style romantic     ║
║    python piano_expander.py boceto.mid --split G4 --counter-density 0.5   ║
║    python piano_expander.py boceto.mid --export-fingerprint --verbose      ║
║    python piano_expander.py boceto.mid --report-only                       ║
║                                                                              ║
║  DEPENDENCIAS: pip install mido numpy                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import sys, os, json, math, argparse, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

try:
    import mido
    from mido import MidiFile, MidiTrack, Message, MetaMessage
except ImportError:
    print("ERROR: pip install mido"); sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy"); sys.exit(1)

try:
    import yaml; YAML_OK = True
except ImportError:
    YAML_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
TICKS = 480
PERC_CH = 9

def midi_to_name(n):
    return f"{NOTE_NAMES[n%12]}{n//12-1}"

def name_to_midi(s):
    flat = {'Db':1,'Eb':3,'Fb':4,'Gb':6,'Ab':8,'Bb':10,'Cb':11}
    s = s.strip(); u = s[0].upper()+s[1:]
    if len(u)>=2 and u[1]=='b' and u[:2] in flat:
        return (int(u[2:])+1)*12+flat[u[:2]]
    if len(u)>=2 and u[1]=='#':
        nm = {n:i for i,n in enumerate(NOTE_NAMES)}
        return (int(u[2:])+1)*12+nm.get(u[:2],0)
    nm = {n:i for i,n in enumerate(NOTE_NAMES)}
    return (int(u[1:])+1)*12+nm.get(u[0],0)

def pc(n): return n % 12
def clamp(v,lo,hi): return max(lo,min(hi,v))
def is_consonant(a,b): return abs(a-b)%12 in {0,3,4,7,8,9}


# ══════════════════════════════════════════════════════════════════════════════
#  RANGOS IDIOMÁTICOS
# ══════════════════════════════════════════════════════════════════════════════

RANGES = {
    'violin1':(55,96),'violin2':(55,91),'viola':(48,84),'cello':(36,76),
    'contrabass':(28,60),'flute':(60,96),'oboe':(58,91),'clarinet':(50,94),
    'bassoon':(34,75),'horn':(34,77),'trumpet':(52,82),'trombone':(34,72),'tuba':(28,58),
}
SWEET = {
    'violin1':(64,88),'violin2':(60,84),'viola':(55,79),'cello':(43,72),
    'contrabass':(33,52),'flute':(65,90),'oboe':(62,86),'clarinet':(55,86),
    'bassoon':(38,67),'horn':(40,70),'trumpet':(56,77),'trombone':(40,67),
}
INSTR_PROGRAM = {
    'violin1':40,'violin2':40,'viola':41,'cello':42,'contrabass':43,
    'flute':73,'oboe':68,'clarinet':71,'bassoon':70,
    'horn':60,'trumpet':56,'trombone':57,'tuba':58,
}
INSTR_DISPLAY = {
    'violin1':'Violin I','violin2':'Violin II','viola':'Viola',
    'cello':'Cello','contrabass':'Contrabass',
    'flute':'Flute','oboe':'Oboe','clarinet':'Clarinet','bassoon':'Bassoon',
    'horn':'Horn','trumpet':'Trumpet','trombone':'Trombone','tuba':'Tuba',
}
FAMILY = {
    'violin1':'strings','violin2':'strings','viola':'strings',
    'cello':'strings','contrabass':'strings',
    'flute':'woodwind','oboe':'woodwind','clarinet':'woodwind','bassoon':'woodwind',
    'horn':'brass','trumpet':'brass','trombone':'brass','tuba':'brass',
}

# Rol orquestal de cada instrumento
INSTR_ROLE = {
    'violin1':'melody','violin2':'harmony','viola':'inner',
    'cello':'bass_melodic','contrabass':'bass_root',
    'flute':'melody_high','oboe':'counter','clarinet':'harmony',
    'bassoon':'bass_melodic','horn':'pad','trumpet':'melody',
    'trombone':'pad_low','tuba':'bass_root',
}

# Umbral de tensión para que cada familia entre
FAMILY_ENTRY_THRESHOLD = {'strings':0.0,'woodwind':0.15,'brass':0.45}
FAMILY_VEL_BASE = {'strings':0.90,'woodwind':0.85,'brass':1.05}


# ══════════════════════════════════════════════════════════════════════════════
#  PLANTILLAS
# ══════════════════════════════════════════════════════════════════════════════

TEMPLATES = {
    'chamber':{'name':'Chamber',
               'instruments':['violin1','violin2','viola','cello','flute','oboe']},
    'strings':{'name':'Strings',
               'instruments':['violin1','violin2','viola','cello','contrabass']},
    'full':   {'name':'Full Orchestra',
               'instruments':['violin1','violin2','viola','cello','contrabass',
                              'flute','oboe','clarinet','bassoon',
                              'horn','trumpet','trombone','tuba']},
}

TEXTURE_ARCHETYPES = {
    'A':'Tutti homofónico','B':'Melodía + colchón',
    'C':'Melodía + arpegios','D':'Contrapuntístico','E':'Sparse / cámara',
}

STYLE_ARCHETYPE_BIAS = {
    'romantic':     {'A':0.20,'B':0.35,'C':0.25,'D':0.10,'E':0.10},
    'baroque':      {'A':0.10,'B':0.15,'C':0.20,'D':0.45,'E':0.10},
    'impressionist':{'A':0.05,'B':0.25,'C':0.20,'D':0.15,'E':0.35},
    'cinematic':    {'A':0.40,'B':0.25,'C':0.20,'D':0.05,'E':0.10},
    'chamber':      {'A':0.10,'B':0.20,'C':0.20,'D':0.40,'E':0.10},
    'auto':         {'A':0.20,'B':0.20,'C':0.20,'D':0.20,'E':0.20},
}


# ══════════════════════════════════════════════════════════════════════════════
#  TEORÍA MUSICAL
# ══════════════════════════════════════════════════════════════════════════════

def detect_key(notes):
    pm=[6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]
    pn=[6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17]
    counts=[0.0]*12
    for n in notes: counts[pc(n['pitch'])]+=n['duration']
    tot=sum(counts) or 1; counts=[c/tot for c in counts]
    best=-999; bk=0; bm='major'
    for t in range(12):
        for mode,prof in [('major',pm),('minor',pn)]:
            rot=[prof[(i-t)%12] for i in range(12)]
            s=float(np.corrcoef(counts,rot)[0,1])
            if s>best: best=s; bk=t; bm=mode
    return bk,bm

def scale_pcs(tonic,mode):
    itvs={'major':[0,2,4,5,7,9,11],'minor':[0,2,3,5,7,8,10]}
    return [(tonic+i)%12 for i in itvs.get(mode,[0,2,4,5,7,9,11])]

def snap_to_scale(pitch,scale):
    p=pc(pitch)
    if p in scale: return pitch
    for d in range(1,7):
        if (p+d)%12 in scale: return pitch+d
        if (p-d)%12 in scale: return pitch-d
    return pitch

def nearest_chord_tone(pitch,chord_pcs):
    best=pitch; best_d=999
    for oct in range(-2,3):
        for cp in chord_pcs:
            p=cp+(pitch//12+oct)*12
            if 21<=p<=108:
                d=abs(p-pitch)
                if d<best_d: best_d=d; best=p
    return best

def infer_chord(notes):
    if not notes: return [0,4,7]
    pcs=list(set(pc(n['pitch']) for n in notes))
    return pcs or [0,4,7]

def fit_range(pitch,lo,hi):
    p=pitch
    while p<lo:
        if p+12>hi: return None
        p+=12
    while p>hi:
        if p-12<lo: return None
        p-=12
    return p if lo<=p<=hi else None

def make_note(pitch,tick,duration,velocity,instr_name):
    lo,hi=RANGES.get(instr_name,(21,108))
    p=fit_range(pitch,lo,hi)
    if p is None: return None
    return {'tick':tick,'pitch':p,'velocity':clamp(velocity,20,127),
            'duration':max(duration,30),'channel':0}


# ══════════════════════════════════════════════════════════════════════════════
#  LECTURA Y ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════

def load_midi(path):
    mid=MidiFile(path); tpb=mid.ticks_per_beat; tempo=500000
    for track in mid.tracks:
        for msg in track:
            if msg.type=='set_tempo': tempo=msg.tempo; break
    return mid,tpb,tempo

def extract_notes(mid):
    result={}
    for i,track in enumerate(mid.tracks):
        name=track.name.strip().rstrip('\x00') or f"Track_{i}"
        on={}; notes=[]; t=0
        for msg in track:
            t+=msg.time
            if msg.type=='note_on' and msg.velocity>0 and msg.channel!=PERC_CH:
                on[(msg.channel,msg.note)]=(t,msg.velocity)
            elif (msg.type=='note_off' or
                  (msg.type=='note_on' and msg.velocity==0)) and msg.channel!=PERC_CH:
                key=(msg.channel,msg.note)
                if key in on:
                    t0,vel=on.pop(key); dur=t-t0
                    if dur>0:
                        notes.append({'tick':t0,'pitch':msg.note,
                                      'velocity':vel,'duration':dur,'channel':msg.channel})
        if notes: result[name]=notes
    return result

def split_hands(notes,split,tpb):
    SLACK=tpb//8; sn=sorted(notes,key=lambda n:(n['tick'],-n['pitch']))
    clusters=[]; i=0
    while i<len(sn):
        g=[sn[i]]; j=i+1
        while j<len(sn) and sn[j]['tick']-sn[i]['tick']<=SLACK:
            g.append(sn[j]); j+=1
        clusters.append(g); i=j
    rh=[]; lh=[]
    for c in clusters:
        cs=sorted(c,key=lambda n:-n['pitch'])
        above=[n for n in cs if n['pitch']>=split]
        below=[n for n in cs if n['pitch']<split]
        if above and below: rh.extend(above); lh.extend(below)
        elif len(cs)==1: (rh if cs[0]['pitch']>=split else lh).extend(cs)
        else: rh.append(cs[0]); lh.extend(cs[1:])
    return rh,lh

def extract_melody(notes,tpb):
    if not notes: return []
    SLACK=tpb//8; sn=sorted(notes,key=lambda n:n['tick'])
    mel=[]; i=0
    while i<len(sn):
        g=[sn[i]]; j=i+1
        while j<len(sn) and sn[j]['tick']-sn[i]['tick']<=SLACK:
            g.append(sn[j]); j+=1
        mel.append(max(g,key=lambda n:n['pitch'])); i=j
    return mel

def extract_bass_line(notes,tpb):
    if not notes: return []
    SLACK=tpb//8; sn=sorted(notes,key=lambda n:n['tick'])
    bass=[]; i=0
    while i<len(sn):
        g=[sn[i]]; j=i+1
        while j<len(sn) and sn[j]['tick']-sn[i]['tick']<=SLACK:
            g.append(sn[j]); j+=1
        bass.append(min(g,key=lambda n:n['pitch'])); i=j
    return bass

def extract_chords(notes,tpb):
    SLACK=tpb//4; sn=sorted(notes,key=lambda n:n['tick'])
    chords=[]; i=0
    while i<len(sn):
        g=[sn[i]]; j=i+1
        while j<len(sn) and sn[j]['tick']-sn[i]['tick']<=SLACK:
            g.append(sn[j]); j+=1
        dur=max(n['duration'] for n in g)
        pitches=sorted(set(n['pitch'] for n in g))
        chords.append({'tick':sn[i]['tick'],'duration':dur,'pitches':pitches,
                       'chord_pcs':infer_chord(g),
                       'mean_vel':float(np.mean([n['velocity'] for n in g]))})
        i=j
    return chords

def chord_at(tick,chords):
    if not chords: return {'tick':0,'duration':480,'pitches':[60],'chord_pcs':[0,4,7],'mean_vel':64}
    active=chords[0]
    for c in chords:
        if c['tick']<=tick: active=c
        else: break
    return active

def segment_sections(notes,tpb,sensitivity=0.5):
    if not notes: return []
    bar=tpb*4; max_tick=max(n['tick']+n['duration'] for n in notes)
    n_bars=max(int(math.ceil(max_tick/bar)),1)
    bd=np.zeros(n_bars); bv=np.zeros(n_bars)
    for n in notes:
        b=min(n['tick']//bar,n_bars-1)
        bd[b]+=1; bv[b]+=n['velocity']
    bd/=max(bd.max(),1); bv/=max(bv.max(),1)
    signal=(bd+bv)/2.0
    threshold=0.20+(1.0-sensitivity)*0.25
    window=max(2,int(4*(1-sensitivity)))
    boundaries=[0]
    for i in range(window,n_bars-window):
        left=np.mean(signal[max(0,i-window):i])
        right=np.mean(signal[i:min(n_bars,i+window)])
        if abs(right-left)>threshold and i-boundaries[-1]>=4:
            boundaries.append(i)
    boundaries.append(n_bars)
    sections=[]
    for i in range(len(boundaries)-1):
        bs,be=boundaries[i],boundaries[i+1]
        ts,te=bs*bar,be*bar
        sn=[n for n in notes if ts<=n['tick']<te]
        if not sn: continue
        sections.append({
            'index':i,'bar_start':bs+1,'bar_end':be,
            'tick_start':ts,'tick_end':te,'notes':sn,
            'density':float(np.mean(bd[bs:be])),
            'mean_vel':float(np.mean([n['velocity'] for n in sn])),
            'mean_dur':float(np.mean([n['duration'] for n in sn])),
            'n_notes':len(sn),
        })
    return sections

def classify_texture(section,tpb,style,sensitivity):
    density=section['density']; mean_vel=section['mean_vel']
    mean_dur=section['mean_dur']
    ns=sorted(section['notes'],key=lambda x:x['tick'])
    steps=sum(1 for i in range(1,len(ns)) if abs(ns[i]['pitch']-ns[i-1]['pitch'])<=2)
    step_ratio=steps/max(len(ns)-1,1)
    scores={'A':0.0,'B':0.0,'C':0.0,'D':0.0,'E':0.0}
    scores['A']+=density*0.4+(mean_vel/127)*0.4+(1-step_ratio)*0.2
    scores['B']+=(1-density)*0.3+step_ratio*0.4+(mean_dur/(tpb*2))*0.3
    scores['C']+=density*0.3+(1-mean_dur/(tpb*2))*0.4+step_ratio*0.3
    scores['D']+=density*0.3+(1-step_ratio)*0.4+(1-mean_vel/127)*0.3
    scores['E']+=(1-density)*0.5+(mean_dur/(tpb*4))*0.3+(1-mean_vel/127)*0.2
    bias=STYLE_ARCHETYPE_BIAS.get(style,STYLE_ARCHETYPE_BIAS['auto'])
    for k in scores:
        scores[k]=scores[k]*(1-sensitivity*0.3)+bias[k]*sensitivity*0.3
    return max(scores,key=lambda k:scores[k])


# ══════════════════════════════════════════════════════════════════════════════
#  CURVA DE TENSIÓN GLOBAL  [mejora D]
# ══════════════════════════════════════════════════════════════════════════════

def build_tension_curve(sections,all_notes,tpb):
    """
    Curva de tensión [0,1] por sección.
    Combina densidad/dinámica con un arco narrativo:
    el clímax tiende a los 2/3 de la obra.
    """
    if not sections: return np.array([0.5])
    n=len(sections)
    curve=np.zeros(n)
    for i,sec in enumerate(sections):
        musical=sec['density']*0.5+(sec['mean_vel']/127)*0.5
        pos=i/max(n-1,1)
        # Arco narrativo: sube hasta 2/3, baja al final
        narrative=math.sin(math.pi*pos*1.5)*0.6
        curve[i]=clamp(musical*0.7+narrative*0.3,0.0,1.0)
    if n>2:
        kernel=np.array([0.25,0.5,0.25])
        curve=np.convolve(curve,kernel,mode='same')
    return np.clip(curve,0.0,1.0)

def tension_at(tick,sections,curve):
    if not sections: return 0.5
    for i,sec in enumerate(sections):
        if sec['tick_start']<=tick<sec['tick_end']:
            return float(curve[i])
    return float(curve[-1])

def instr_active(instr_name,tension,archetype):
    """Decide si el instrumento toca según tensión y arquetipo."""
    family=FAMILY.get(instr_name,'strings')
    if archetype=='E' and family=='brass': return False
    if instr_name in ('tuba','trombone') and tension<0.55: return False
    if instr_name=='trumpet' and tension<0.40: return False
    return tension>=FAMILY_ENTRY_THRESHOLD.get(family,0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  GENERADORES DE FIGURA RÍTMICA POR ROL  [mejora B]
# ══════════════════════════════════════════════════════════════════════════════

def gen_melody(melody,instr_name,octave_shift,tpb):
    """Melodía principal transpuesta al registro del instrumento."""
    lo,hi=SWEET.get(instr_name,RANGES.get(instr_name,(36,96)))
    notes=[]
    for mn in melody:
        p=mn['pitch']+octave_shift*12
        p=fit_range(p,lo,hi)
        if p is None: continue
        n=make_note(p,mn['tick'],mn['duration'],mn['velocity'],instr_name)
        if n: notes.append(n)
    return notes

def gen_bass_melodic(bass_line,chords,instr_name,scale,tpb):
    """
    Bajo melódico: reproduce la línea de bajo del piano con notas
    de paso escalares entre saltos de 3ª o más.
    """
    lo,hi=RANGES.get(instr_name,(28,76))
    notes=[]; bl=sorted(bass_line,key=lambda n:n['tick'])
    for i,bn in enumerate(bl):
        p=fit_range(bn['pitch'],lo,hi)
        if p is None: continue
        notes.append(make_note(p,bn['tick'],bn['duration'],bn['velocity'],instr_name))
        if i<len(bl)-1:
            nxt=bl[i+1]
            gap=nxt['tick']-(bn['tick']+bn['duration'])
            interval=nxt['pitch']-bn['pitch']
            if gap>=tpb//2 and 2<abs(interval)<=7:
                step=1 if interval>0 else -1
                pp=snap_to_scale(p+step,scale)
                pp=fit_range(pp,lo,hi)
                if pp:
                    pt=bn['tick']+bn['duration']+gap//3
                    n=make_note(pp,pt,gap//3,max(bn['velocity']-15,30),instr_name)
                    if n: notes.append(n)
    return [n for n in notes if n]

def gen_bass_root(bass_line,chords,instr_name,tpb):
    """
    Contrabajo/tuba: solo raíces de acorde en tiempos 1 y 3.
    Da peso rítmico sin duplicar la línea melódica del cello.
    """
    lo,hi=RANGES.get(instr_name,(21,62))
    bar=tpb*4; notes=[]
    for chord in chords:
        beat_in_bar=chord['tick']%bar
        # Solo tiempo 1 (beat_in_bar~0) y tiempo 3 (beat_in_bar~tpb*2)
        on_beat1=beat_in_bar<=tpb//4
        on_beat3=abs(beat_in_bar-tpb*2)<=tpb//4
        if not (on_beat1 or on_beat3): continue
        root=min(chord['pitches'])
        p=fit_range(root,lo,hi)
        if p is None: continue
        dur=min(chord['duration'],tpb*2)
        n=make_note(p,chord['tick'],dur,int(chord['mean_vel']*0.85),instr_name)
        if n: notes.append(n)
    return notes

def gen_harmony(chords,instr_name,scale,archetype,tension,tpb):
    """
    Voces interiores (violin2, viola, clarinet, horn).
    El ritmo varía según arquetipo:
      A/D → negras
      B   → blancas (colchón)
      C   → tresillos de corchea
      E   → redondas
    """
    lo,hi=SWEET.get(instr_name,RANGES.get(instr_name,(36,96)))
    fam=FAMILY.get(instr_name,'strings')
    notes=[]
    for chord in chords:
        pcs=chord['chord_pcs']; pitches=chord['pitches']
        if not pitches: continue
        mid_idx=len(pitches)//2
        base=pitches[mid_idx] if len(pitches)>1 else pitches[0]
        p=fit_range(base,lo,hi)
        if p is None: continue
        p=nearest_chord_tone(p,pcs)
        p=fit_range(p,lo,hi)
        if p is None: continue
        vel=clamp(int(chord['mean_vel']*FAMILY_VEL_BASE.get(fam,0.9)),30,110)

        if archetype in ('A','D'):
            beat=tpb; n_beats=max(1,chord['duration']//beat)
            for b in range(n_beats):
                t=chord['tick']+b*beat
                n=make_note(p,t,int(beat*0.92),vel,instr_name)
                if n: notes.append(n)
        elif archetype=='B':
            dur=min(chord['duration'],tpb*2)
            n=make_note(p,chord['tick'],dur,max(vel-10,30),instr_name)
            if n: notes.append(n)
        elif archetype=='C':
            tresillo=tpb//3
            # raíz → 5ª → raíz (figura de arpegio)
            p5=nearest_chord_tone(p+7,pcs)
            p5=fit_range(p5,lo,hi) or p
            for i,pp in enumerate([p,p5,p]):
                t=chord['tick']+i*tresillo
                if t>=chord['tick']+chord['duration']: break
                n=make_note(pp,t,int(tresillo*0.85),vel,instr_name)
                if n: notes.append(n)
        else:  # E
            n=make_note(p,chord['tick'],chord['duration'],max(vel-20,20),instr_name)
            if n: notes.append(n)
    return notes

def gen_pad(chords,instr_name,tension_fn,scale,tpb):
    """
    Metales (horn, trombone): redondas en zonas de tensión alta.
    Usan voces internas del acorde, no la melodía ni el bajo.
    Entran gradualmente según la curva de tensión.
    """
    lo,hi=RANGES.get(instr_name,(34,77))
    bar=tpb*4; notes=[]
    by_bar={}
    for chord in chords:
        b=chord['tick']//bar
        if b not in by_bar: by_bar[b]=chord
    for bar_idx,chord in sorted(by_bar.items()):
        tension=tension_fn(chord['tick'])
        if not instr_active(instr_name,tension,'B'): continue
        pcs=chord['chord_pcs']; pitches=chord['pitches']
        if not pitches: continue
        mid_p=pitches[len(pitches)//2] if len(pitches)>1 else pitches[0]
        p=fit_range(mid_p,lo,hi)
        if p is None: continue
        p=nearest_chord_tone(p,pcs)
        p=fit_range(p,lo,hi)
        if p is None: continue
        # Asegurar consonancia con extremos
        if not is_consonant(p,max(pitches)):
            p=nearest_chord_tone(p+3,pcs)
            p=fit_range(p,lo,hi)
            if p is None: continue
        vel=clamp(int(chord['mean_vel']*FAMILY_VEL_BASE.get('brass',1.0)*tension),25,100)
        n=make_note(p,chord['tick'],min(chord['duration'],bar),vel,instr_name)
        if n: notes.append(n)
    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  CONTRAPUNTO REAL CON MIRADA HACIA ADELANTE  [mejora C]
# ══════════════════════════════════════════════════════════════════════════════

def gen_counterpoint(melody,chords,tonic,scale,instr_name,density,tpb):
    """
    Contramelodía con arco independiente:

    1. Mirada hacia adelante: analiza la dirección de la melodía
       en bloques de 4 compases para decidir la dirección contraria.
    2. Clímax propio: la contramelodía tiene su punto álgido cuando
       la melodía está en su registro más grave.
    3. Reglas estrictas:
       - Solo tonos del acorde activo
       - Preferencia por 3ªs y 6ªs bajo la melodía
       - Movimiento contrario a la melodía (cuando posible)
       - Sin 5ªs ni 8ªs paralelas consecutivas
       - Sin saltos de más de una 6ª
       - Sin disonancias sin preparación
    """
    if not melody or not chords: return []
    lo,hi=SWEET.get(instr_name,RANGES.get(instr_name,(48,84)))
    bar=tpb*4

    # Agrupar melodía por compás para lookahead
    mel_by_bar: Dict[int,List] = defaultdict(list)
    for mn in melody: mel_by_bar[mn['tick']//bar].append(mn)

    def future_direction(bar_idx,lookahead=4):
        """Dirección futura de la melodía: 1=sube, -1=baja, 0=estable."""
        future=[]
        for b in range(bar_idx,bar_idx+lookahead):
            future.extend(mel_by_bar.get(b,[]))
        if len(future)<2: return 0
        s=np.mean([n['pitch'] for n in future[:len(future)//2]])
        e=np.mean([n['pitch'] for n in future[len(future)//2:]])
        if e-s>2: return 1
        if s-e>2: return -1
        return 0

    # Clímax de la contramelodía: cuando la melodía está en su mínimo
    mel_pitches=[n['pitch'] for n in melody]
    climax_idx=int(np.argmin(mel_pitches))

    notes=[]; prev_p=None; prev_mel=None

    for i,mn in enumerate(melody):
        if random.random()>density: prev_mel=mn['pitch']; continue
        ch=chord_at(mn['tick'],chords)
        pcs=ch['chord_pcs']; mel_p=mn['pitch']
        bar_idx=mn['tick']//bar
        fut_dir=future_direction(bar_idx)
        desired_dir=-fut_dir if fut_dir!=0 else 0
        at_climax=(abs(i-climax_idx)<len(melody)//8)

        # Intervalos preferidos bajo la melodía
        preferred=[4,3,9,8,7,12]
        if at_climax: preferred=[9,8,4,3,12,7]

        candidate=None; best_score=-999

        for interval in preferred:
            p=mel_p-interval
            p=fit_range(p,lo,hi)
            if p is None: continue
            if pc(p) not in pcs:
                p=nearest_chord_tone(p,pcs)
                p=fit_range(p,lo,hi)
                if p is None: continue
            if not is_consonant(p,mel_p): continue
            if p>=mel_p: continue
            if prev_p is not None and abs(p-prev_p)>9: continue
            # Evitar 5ªs y 8ªs paralelas
            if prev_p is not None and prev_mel is not None:
                if (prev_mel-prev_p) in (7,12) and (mel_p-p)==(prev_mel-prev_p):
                    continue
            score=0
            if prev_p is not None:
                direction=p-prev_p
                if desired_dir!=0 and direction*desired_dir>0: score+=3
                if direction==0: score+=1
                if desired_dir!=0 and direction*desired_dir<0: score-=1
            if at_climax and prev_p is not None and abs(p-prev_p)<=2: score+=2
            if score>best_score: best_score=score; candidate=p

        if candidate is not None:
            vel=clamp(int(mn['velocity']*0.72),25,95)
            if at_climax: vel=clamp(vel+10,30,100)
            n=make_note(candidate,mn['tick'],mn['duration'],vel,instr_name)
            if n: notes.append(n); prev_p=candidate
        prev_mel=mel_p

    return notes


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE LÍNEA COMPLETA POR INSTRUMENTO  [mejora A]
# ══════════════════════════════════════════════════════════════════════════════

def generate_line(instr_name,melody,bass_line,chords,sections,
                  tension_curve,tonic,scale,style,
                  counter_density,tpb,no_counter):
    """
    Genera la línea completa de un instrumento de principio a fin.
    Cada rol usa su propio generador — no hay copia genérica.
    """
    role=INSTR_ROLE.get(instr_name,'harmony')

    def tfn(tick): return tension_at(tick,sections,tension_curve)

    # Arquetipo dominante (para generadores que no iteran por sección)
    arch_count: Dict[str,int]=defaultdict(int)
    for sec in sections: arch_count[sec.get('archetype','B')]+=1
    dom_arch=max(arch_count,key=lambda k:arch_count[k]) if arch_count else 'B'

    if role=='melody':
        return gen_melody(melody,instr_name,0,tpb)

    elif role=='melody_high':
        return gen_melody(melody,instr_name,1,tpb)

    elif role=='bass_melodic':
        return gen_bass_melodic(bass_line,chords,instr_name,scale,tpb)

    elif role=='bass_root':
        return gen_bass_root(bass_line,chords,instr_name,tpb)

    elif role in ('harmony','inner'):
        # Genera sección por sección con el arquetipo y tensión correctos
        all_notes=[]
        for sec in sections:
            arch=sec.get('archetype',dom_arch)
            t=tfn(sec['tick_start'])
            if not instr_active(instr_name,t,arch): continue
            sec_chords=[c for c in chords
                        if sec['tick_start']<=c['tick']<sec['tick_end']]
            if not sec_chords: continue
            all_notes.extend(gen_harmony(sec_chords,instr_name,scale,arch,t,tpb))
        return all_notes

    elif role=='counter':
        if no_counter: return []
        return gen_counterpoint(melody,chords,tonic,scale,
                                instr_name,counter_density,tpb)

    elif role in ('pad','pad_low'):
        return gen_pad(chords,instr_name,tfn,scale,tpb)

    return []


# ══════════════════════════════════════════════════════════════════════════════
#  ARTICULACIÓN Y CC
# ══════════════════════════════════════════════════════════════════════════════

def apply_articulation(notes,instr_name,archetype_map,tpb):
    family=FAMILY.get(instr_name,'strings'); result=[]
    for n in notes:
        arch=archetype_map.get(n['tick']//tpb,'B')
        dur=n['duration']; vel=n['velocity']
        if arch=='A':
            dur=int(dur*0.90)
            if family=='brass': vel=clamp(vel+8,30,127)
        elif arch=='C':
            dur=int(dur*(0.72 if family=='strings' else 0.95))
        elif arch=='D':
            dur=int(dur*0.98)
        elif arch=='E':
            dur=int(dur*1.02); vel=clamp(int(vel*0.80),20,90)
        result.append({**n,'duration':max(dur,30),'velocity':clamp(vel,20,127)})
    return result

def build_cc(instr_name,sections,tension_curve,tpb):
    family=FAMILY.get(instr_name,'strings'); cc=[]
    for i,sec in enumerate(sections):
        t=float(tension_curve[i]) if i<len(tension_curve) else 0.5
        vel_base=int(sec['mean_vel']*FAMILY_VEL_BASE.get(family,0.9))
        cc1=clamp(int(vel_base*0.85+t*30),1,127)
        cc11=clamp(int(cc1*0.92),1,127)
        n_pts=max(2,(sec['tick_end']-sec['tick_start'])//(tpb*2))
        for j in range(int(n_pts)+1):
            prog=j/max(n_pts,1)
            arc=math.sin(math.pi*prog)*8
            tt=sec['tick_start']+j*(sec['tick_end']-sec['tick_start'])//max(int(n_pts),1)
            cc.append((tt,'cc',1,clamp(cc1+int(arc),1,127)))
            cc.append((tt,'cc',11,clamp(cc11+int(arc*0.8),1,127)))
    return cc


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DEL MIDI
# ══════════════════════════════════════════════════════════════════════════════

def build_track(notes,cc_events,channel,name,program):
    track=MidiTrack()
    track.append(MetaMessage('track_name',name=name,time=0))
    track.append(Message('program_change',channel=channel,program=program,time=0))
    events=[]
    for n in notes:
        t=n['tick']; dur=max(n['duration'],1)
        events.append((t,'on',n['pitch'],n['velocity']))
        events.append((t+dur,'off',n['pitch'],0))
    for ev in cc_events:
        t,_,cc_num,cc_val=ev
        events.append((t,'cc',cc_num,cc_val))
    events.sort(key=lambda e:(e[0],{'off':0,'cc':1,'on':2}.get(e[1],1)))
    prev=0
    for abs_t,etype,a,b in events:
        delta=abs_t-prev; prev=abs_t
        if etype=='on':
            track.append(Message('note_on',channel=channel,note=a,velocity=b,time=delta))
        elif etype=='off':
            track.append(Message('note_off',channel=channel,note=a,velocity=0,time=delta))
        elif etype=='cc':
            track.append(Message('control_change',channel=channel,control=a,value=b,time=delta))
    return track

def build_midi(instr_notes,instr_cc,instruments,tempo,tpb,output_path):
    mid=MidiFile(type=1,ticks_per_beat=tpb)
    t0=MidiTrack()
    t0.append(MetaMessage('set_tempo',tempo=tempo,time=0))
    t0.append(MetaMessage('time_signature',numerator=4,denominator=4,
                           clocks_per_click=24,notated_32nd_notes_per_beat=8,time=0))
    mid.tracks.append(t0)
    created=[]
    for ch,name in enumerate(instruments):
        notes=instr_notes.get(name,[])
        cc=instr_cc.get(name,[])
        if not notes: continue
        prog=INSTR_PROGRAM.get(name,40)
        disp=INSTR_DISPLAY.get(name,name)
        track=build_track(notes,cc,ch%16,disp,prog)
        mid.tracks.append(track)
        created.append(disp)
    mid.save(output_path)
    return created


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def expand(rh,lh,tpb,tempo,instruments,style,args):
    all_piano=sorted(rh+lh,key=lambda n:n['tick'])
    report={'sections':[],'steps':[],'instruments':{}}

    # ── Análisis ──────────────────────────────────────────────────────────────
    tonic,mode=detect_key(all_piano)
    scale=scale_pcs(tonic,mode)
    key_name=f"{NOTE_NAMES[tonic]} {mode}"
    print(f"  Tonalidad           : {key_name}")
    report['key']=key_name

    if style=='auto':
        mv=np.mean([n['velocity'] for n in all_piano]) if all_piano else 64
        md=np.mean([n['duration'] for n in all_piano]) if all_piano else tpb
        density=len(all_piano)/max(max(n['tick']+n['duration'] for n in all_piano)/(tpb*4),1)
        if mv>80 and density>8: style='cinematic'
        elif md>tpb*2 and mv<65: style='impressionist'
        else: style='romantic'
        print(f"  Estilo auto         : {style}")
    report['style']=style

    melody   =extract_melody(rh if rh else all_piano,tpb)
    bass_line=extract_bass_line(lh if lh else all_piano,tpb)
    chords   =extract_chords(all_piano,tpb)
    print(f"  Melodía             : {len(melody)} notas")
    print(f"  Línea de bajo       : {len(bass_line)} notas")
    print(f"  Acordes             : {len(chords)}")
    report['steps'].append(f"melodía={len(melody)}, bajo={len(bass_line)}, acordes={len(chords)}")

    # ── Secciones y texturas ──────────────────────────────────────────────────
    sections=segment_sections(all_piano,tpb,args.texture_sensitivity)
    print(f"  Secciones           : {len(sections)}")
    for sec in sections:
        arch=classify_texture(sec,tpb,style,args.texture_sensitivity)
        sec['archetype']=arch
        if args.verbose:
            print(f"    Sec {sec['index']+1:>2} cc {sec['bar_start']:>3}-{sec['bar_end']:<4}"
                  f" [{arch}] {TEXTURE_ARCHETYPES[arch]:<22}"
                  f" vel={sec['mean_vel']:.0f} dens={sec['density']:.2f}")
        report['sections'].append({
            'index':sec['index'],'bars':f"{sec['bar_start']}-{sec['bar_end']}",
            'archetype':arch,'label':TEXTURE_ARCHETYPES[arch],
            'density':round(sec['density'],3),'mean_vel':round(sec['mean_vel'],1),
        })

    # ── Curva de tensión global ───────────────────────────────────────────────
    tension_curve=build_tension_curve(sections,all_piano,tpb)
    report['tension_curve']=[round(float(t),3) for t in tension_curve]
    if args.verbose:
        bar=''.join('▁▂▃▄▅▆▇█'[clamp(int(t*7.99),0,7)] for t in tension_curve)
        print(f"  Curva tensión       : {bar}")

    # Mapa arquetipo por compás (para articulación)
    archetype_map={}
    for sec in sections:
        arch=sec.get('archetype','B')
        for t_tick in range(sec['tick_start'],sec['tick_end'],tpb):
            archetype_map[t_tick//tpb]=arch

    # ── Generación de líneas (arquitectura invertida) ─────────────────────────
    instr_notes={}; instr_cc={}
    print(f"  Generando líneas...")
    for instr_name in instruments:
        line=generate_line(
            instr_name=instr_name,
            melody=melody,
            bass_line=bass_line,
            chords=chords,
            sections=sections,
            tension_curve=tension_curve,
            tonic=tonic,
            scale=scale,
            style=style,
            counter_density=args.counter_density,
            tpb=tpb,
            no_counter=args.no_counter,
        )
        if not args.no_ornaments and line:
            line=apply_articulation(line,instr_name,archetype_map,tpb)
        line.sort(key=lambda n:(n['tick'],n['pitch']))
        instr_notes[instr_name]=line
        if not args.no_cc and line:
            instr_cc[instr_name]=build_cc(instr_name,sections,tension_curve,tpb)
        role=INSTR_ROLE.get(instr_name,'?')
        disp=INSTR_DISPLAY.get(instr_name,instr_name)
        print(f"  [{disp:<16}] {len(line):>5} notas  rol={role}")
        report['instruments'][disp]={'notes':len(line),'role':role}

    total_out=sum(len(v) for v in instr_notes.values())
    report['total_notes_in']=len(all_piano)
    report['total_notes_out']=total_out
    report['expansion_ratio']=round(total_out/max(len(all_piano),1),2)
    return instr_notes,instr_cc,report


# ══════════════════════════════════════════════════════════════════════════════
#  INFORME
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(report,midi_in,midi_out,created):
    W=70; L=['═'*W,'  PIANO EXPANDER v2.0 — INFORME','═'*W]
    L+=[f"  Fuente    : {midi_in}",f"  Salida    : {midi_out}",
        f"  Tonalidad : {report.get('key','?')}",
        f"  Estilo    : {report.get('style','?')}",
        f"  Entrada   : {report.get('total_notes_in',0)} notas",
        f"  Salida    : {report.get('total_notes_out',0)} notas",
        f"  Expansión : {report.get('expansion_ratio',0):.1f}x",'']
    L+=['  CURVA DE TENSIÓN GLOBAL','  '+'─'*(W-2)]
    tc=report.get('tension_curve',[])
    if tc:
        bar=''.join('▁▂▃▄▅▆▇█'[clamp(int(t*7.99),0,7)] for t in tc)
        L+=[f"  {bar}",f"  min={min(tc):.2f}  max={max(tc):.2f}  media={float(np.mean(tc)):.2f}"]
    L+=['','  SECCIONES Y ARQUETIPOS','  '+'─'*(W-2)]
    for s in report.get('sections',[]):
        L.append(f"    Sec {s['index']+1:>2}  cc {s['bars']:<12}"
                 f" [{s['archetype']}] {s['label']:<22}"
                 f" vel={s['mean_vel']:.0f}  dens={s['density']:.2f}")
    L+=['','  LÍNEAS GENERADAS (nota por instrumento)','  '+'─'*(W-2)]
    for name,data in report.get('instruments',{}).items():
        L.append(f"    {name:<18} {data['notes']:>5} notas  rol={data['role']}")
    L+=['','  ROLES','  '+'─'*(W-2),
        '    melody       Melodía principal, registro original',
        '    melody_high  Melodía 8va alta (flauta)',
        '    bass_melodic Bajo con notas de paso propias',
        '    bass_root    Raíces en tiempos fuertes (cb/tuba)',
        '    harmony      Voces internas, figura por arquetipo',
        '    inner        Voz interior variable (viola)',
        '    counter      Contrapunto independiente con arco',
        '    pad          Metales sostenidos, entran con la tensión',
        '    pad_low      Pad en registro grave','',
        '  ARQUETIPOS','  '+'─'*(W-2),
        '    A Tutti       — ataca junto, dinámica alta',
        '    B Colchón     — melodía + voces largas de fondo',
        '    C Arpegios    — figuración activa en acomp.',
        '    D Contrapunto — líneas independientes simultáneas',
        '    E Sparse      — pocas notas, mucho espacio','',
        '  CURVA DE TENSIÓN','  '+'─'*(W-2),
        '    Metales entran cuando tensión supera su umbral.',
        '    Tuba/trombón: >0.55 | Trompeta: >0.40 | Maderas: >0.15',
        '    CC1 y CC11 siguen esta curva — no la dinámica local.','═'*W]
    return '\n'.join(L)

def export_fingerprint(report,path):
    fp={'source':'piano_expander_v2','key':report.get('key','?'),
        'style':report.get('style','?'),
        'expansion_ratio':report.get('expansion_ratio',1.0),
        'tension_curve':report.get('tension_curve',[]),
        'sections':[{'bars':s['bars'],'archetype':s['archetype'],
                     'label':s['label'],'density':s['density']}
                    for s in report.get('sections',[])]}
    with open(path,'w',encoding='utf-8') as f: json.dump(fp,f,indent=2,ensure_ascii=False)
    print(f"  Fingerprint: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE PLANTILLA
# ══════════════════════════════════════════════════════════════════════════════

def load_template(arg):
    if arg in TEMPLATES:
        t=TEMPLATES[arg]; return t['instruments'],t['name']
    path=Path(arg)
    if path.exists():
        try:
            with open(path,'r',encoding='utf-8') as f: data=json.load(f)
            if isinstance(data,list):
                instrs=data if isinstance(data[0],str) else [d['name'] for d in data if 'name' in d]
            elif 'instruments' in data:
                raw=data['instruments']
                instrs=raw if isinstance(raw[0],str) else [d['name'] for d in raw if 'name' in d]
            else: raise ValueError("Formato no reconocido")
            instrs=[i for i in instrs if i in INSTR_PROGRAM]
            if not instrs: raise ValueError("Sin instrumentos reconocidos")
            return instrs,path.stem
        except Exception as e:
            print(f"  ⚠ '{arg}': {e}. Usando 'chamber'.")
    else:
        print(f"  ⚠ Plantilla '{arg}' no reconocida. Usando 'chamber'.")
    t=TEMPLATES['chamber']; return t['instruments'],t['name']


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p=argparse.ArgumentParser(
        description='Piano Expander v2.0 — Expansión orquestal de bocetos pianísticos',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('midi')
    p.add_argument('--template',default='chamber',
                   help='chamber|strings|full|<archivo.json> (def: chamber)')
    p.add_argument('--style',
                   choices=['auto','romantic','baroque','impressionist','cinematic','chamber'],
                   default='auto')
    p.add_argument('--split',default='G4',help='Split si pista única (def: G4)')
    p.add_argument('--no-counter',action='store_true')
    p.add_argument('--counter-density',type=float,default=0.40)
    p.add_argument('--texture-sensitivity',type=float,default=0.5)
    p.add_argument('--no-ornaments',action='store_true')
    p.add_argument('--no-cc',action='store_true')
    p.add_argument('--export-fingerprint',action='store_true')
    p.add_argument('--export-yaml',action='store_true')
    p.add_argument('--output',default=None)
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--verbose',action='store_true')
    return p


def main():
    parser=build_parser(); args=parser.parse_args()
    args.counter_density=clamp(args.counter_density,0.0,1.0)
    args.texture_sensitivity=clamp(args.texture_sensitivity,0.0,1.0)
    try: split_note=name_to_midi(args.split)
    except: split_note=67

    print('═'*65); print('  PIANO EXPANDER  v2.0'); print('═'*65)
    print(f"  Entrada   : {args.midi}")
    print(f"  Plantilla : {args.template}")
    print(f"  Estilo    : {args.style}")

    if not Path(args.midi).exists():
        print(f"  ERROR: {args.midi} no encontrado"); sys.exit(1)

    mid,tpb,tempo=load_midi(args.midi)
    global TICKS; TICKS=tpb
    print(f"  TPB={tpb}  Tempo={round(60_000_000/tempo)} BPM")

    tracks=extract_notes(mid)
    if not tracks: print("  ERROR: Sin notas."); sys.exit(1)
    non_empty=[k for k,v in tracks.items() if v]
    print(f"  Pistas detectadas: {len(non_empty)}")

    if len(non_empty)>=2:
        flat=[n for k in non_empty for n in tracks[k]]
        flat.sort(key=lambda n:(n['tick'],-n['pitch']))
        rh=[n for n in flat if n['pitch']>=split_note]
        lh=[n for n in flat if n['pitch']< split_note]
        rh=rh if rh else flat
        print(f"  Split {midi_to_name(split_note)}: MD={len(rh)} MI={len(lh)}")
    else:
        all_n=list(tracks.values())[0] if tracks else []
        rh,lh=split_hands(all_n,split_note,tpb)
        print(f"  Split interno {midi_to_name(split_note)}: MD={len(rh)} MI={len(lh)}")

    if not rh and not lh: print("  ERROR: Sin notas."); sys.exit(1)

    instruments,tmpl_name=load_template(args.template)
    print(f"  Instrumentos  : {len(instruments)}  ({tmpl_name})")

    if args.report_only:
        all_p=sorted(rh+lh,key=lambda n:n['tick'])
        tonic,mode=detect_key(all_p); secs=segment_sections(all_p,tpb,args.texture_sensitivity)
        tc=build_tension_curve(secs,all_p,tpb)
        print(f"\n  Tonalidad: {NOTE_NAMES[tonic]} {mode}  |  Secciones: {len(secs)}")
        for i,sec in enumerate(secs):
            arch=classify_texture(sec,tpb,args.style,args.texture_sensitivity)
            t=float(tc[i]) if i<len(tc) else 0.5
            print(f"    cc {sec['bar_start']:>3}-{sec['bar_end']:<4}"
                  f" [{arch}] {TEXTURE_ARCHETYPES[arch]:<22} tensión={t:.2f}")
        sys.exit(0)

    stem=Path(args.midi).stem
    base=(args.output or stem+'_orquestado').replace('.mid','')
    out_midi=base+'.mid'; out_rep=base+'_report.txt'
    out_fp=base+'.fingerprint.json'

    print('\n  Expandiendo...')
    instr_notes,instr_cc,report_data=expand(rh,lh,tpb,tempo,instruments,args.style,args)

    print(f'\n  Escribiendo: {out_midi}')
    created=build_midi(instr_notes,instr_cc,instruments,tempo,tpb,out_midi)
    print(f"  Pistas: {len(created)}")

    if args.export_fingerprint: export_fingerprint(report_data,out_fp)

    if args.export_yaml and YAML_OK:
        ya={'obra':{'titulo':stem,'generado_por':'piano_expander_v2'},
            'tonalidad':report_data.get('key','?'),
            'estilo':report_data.get('style','?'),
            'instrumentos':[INSTR_DISPLAY.get(i,i) for i in instruments]}
        with open(base+'.yaml','w',encoding='utf-8') as f:
            yaml.dump(ya,f,allow_unicode=True,default_flow_style=False)
        print(f"  YAML: {base+'.yaml'}")

    report_text=generate_report(report_data,args.midi,out_midi,created)
    with open(out_rep,'w',encoding='utf-8') as f: f.write(report_text)
    print(f"  Informe: {out_rep}")

    print('\n'+'═'*65); print('  RESUMEN'); print('═'*65)
    print(f"  MIDI     : {out_midi}")
    print(f"  Informe  : {out_rep}")
    print(f"  Pistas   : {len(created)}")
    print(f"  Tonalidad: {report_data.get('key','?')}")
    print(f"  Estilo   : {report_data.get('style','?')}")
    print(f"  Expansión: {report_data.get('expansion_ratio',0):.1f}x")
    tc=report_data.get('tension_curve',[])
    if tc:
        bar=''.join('▁▂▃▄▅▆▇█'[clamp(int(t*7.99),0,7)] for t in tc)
        print(f"  Tensión  : {bar}")
    archcount: Dict[str,int]=defaultdict(int)
    for s in report_data.get('sections',[]): archcount[s['archetype']]+=1
    for k,v in sorted(archcount.items()):
        print(f"    [{k}] {TEXTURE_ARCHETYPES[k]:<22} {v} sec.")
    print('═'*65)
    print(f'\n  Importa {out_midi} en FL Studio.')
    for i,name in enumerate(created): print(f"  Canal {i+1:>2} → {name}")


if __name__=='__main__':
    main()
