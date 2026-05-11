#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MIDI TEST GENERATORS  v1.0                                ║
║       Generadores de MIDIs sintéticos multi-pista para pruebas              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  DESCRIPCIÓN:                                                                ║
║    Genera 10 MIDIs sintéticos de 5 pistas cada uno, cubriendo distintos     ║
║    estilos y formas musicales, para probar y evaluar midi_pianoroll_analyzer.║
║    Cada MIDI tiene roles asignados: melodía, contra-melodía, armonía,       ║
║    bajo y acompañamiento.                                                    ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MIDIs INCLUIDOS                                                             ║
║                                                                              ║
║    01_rondo_Dm      Rondo A-B-A-C-A, Re menor, allegro                      ║
║    02_blues_Am      Blues 12 compases × 3 choruses, La menor                ║
║    03_waltz_Ab      Vals 3/4, La bemol mayor, A-B-A'-Coda                   ║
║    04_fugue_Dm      Fuga a 4 voces, Re menor, sujeto + respuesta            ║
║    05_tango_Am      Tango con habanera, La menor, A-B-A                     ║
║    06_modal_Dorian  Modal jazz D Dorian, forma AABA × 3 choruses            ║
║    07_minimalist_C  Minimalismo en fase (estilo Reich), Do mayor            ║
║    08_flamenco_E    Flamenco frigio Mi, falseta-B-falseta                   ║
║    09_march_G       Marcha, Sol mayor, Intro-A-B-Trio                       ║
║    10_sonata_C      Forma sonata, Do mayor, Expo-Desarrollo-Recapitulación  ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  USO                                                                         ║
║                                                                              ║
║    python3 midi_test_generators.py                  # genera MIDIs + HTML   ║
║    python3 midi_test_generators.py -o ./salida      # directorio de salida  ║
║    python3 midi_test_generators.py --no-report      # solo MIDIs, sin HTML  ║
║    python3 midi_test_generators.py --list           # lista los MIDIs       ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  OPCIONES                                                                    ║
║                                                                              ║
║    -o / --output   Directorio de salida (default: ./batch)                  ║
║    --no-report     Solo genera los MIDIs, sin invocar el analyzer           ║
║    --list          Muestra la lista de MIDIs disponibles y sale             ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DEPENDENCIAS:  mido  (pip install mido)                                     ║
║                 midi_pianoroll_analyzer.py  (mismo directorio, para HTML)   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import sys, os, argparse

try:
    import mido
except ImportError:
    sys.exit("ERROR: instala mido  →  pip install mido")


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH TEST MIDI GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def _make_track(name, channel, notes_seq, add_meta=False, bpm=120, tpb=480):
    import mido
    t = mido.MidiTrack()
    t.name = name
    t.append(mido.MetaMessage('track_name', name=name, time=0))
    if add_meta:
        t.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))
        t.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                  clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    events = []
    for pitch, vel, start_beat, dur_beat in notes_seq:
        s = int(start_beat * tpb)
        e = int((start_beat + dur_beat) * tpb)
        events.append((s, 'on',  pitch, vel, channel))
        events.append((e, 'off', pitch, 0,   channel))
    events.sort(key=lambda ev: (ev[0], 0 if ev[1] == 'off' else 1))
    prev = 0
    for tick, kind, note, vel, ch in events:
        delta = tick - prev; prev = tick
        if kind == 'on':
            t.append(mido.Message('note_on',  note=note, velocity=vel, channel=ch, time=delta))
        else:
            t.append(mido.Message('note_off', note=note, velocity=0,   channel=ch, time=delta))
    t.append(mido.MetaMessage('end_of_track', time=0))
    return t

def _chord_block(root, ivs, start, dur, vel):
    return [(root + i, vel, start, dur) for i in ivs]

def gen_midi_01_rondo(path):
    """Forma Rondo A-B-A-C-A (D minor, allegro)"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    mel, ctr, har, bas, acc = [], [], [], [], []
    # A theme (bars 0-7): D minor motif
    A = [(62,90,0,.5),(65,85,.5,.5),(69,88,1,1),(67,82,2,.5),(65,80,2.5,.5),(62,85,3,1),
         (60,80,4,.5),(62,82,.5+4,.5),(65,85,5,1),(67,88,6,.5),(69,90,6.5,.5),(70,85,7,1)]
    A2 = [(p,v,t+8,d) for p,v,t,d in A]
    # B theme (bars 8-15): F major
    B = [(65,88,16,.5),(67,85,16.5,.5),(69,90,17,1),(72,92,18,1),(70,85,19,1),
         (69,88,20,.5),(67,82,20.5,.5),(65,85,21,1),(64,80,22,1),(65,82,23,1)]
    B2 = [(p,v,t+8,d) for p,v,t,d in B]
    # A again (bars 16-23)
    A3 = [(p,v,t+32,d) for p,v,t,d in A+A2]
    # C theme (bars 24-31): Bb major, lyrical
    C = [(70,75,48,2),(69,72,50,2),(67,75,52,2),(65,70,54,2),
         (67,75,56,2),(69,78,58,2),(70,80,60,2),(72,82,62,2)]
    # Final A (bars 32-39)
    A4 = [(p,v,t+64,d) for p,v,t,d in A+A2]
    mel = A+A2+B+B2+A3+C+A4
    # Harmony: Am-F-C-G in A sections, F-C-Dm-Bb in B, Bb-F-Gm-Eb in C
    def harm_block(prog, start, reps):
        h = []
        for r in range(reps):
            for ci,(root,ivs,vel) in enumerate(prog):
                h += _chord_block(root,ivs,start+r*16+ci*4,3.8,vel)
        return h
    har += harm_block([(57,[0,3,7],65),(53,[0,4,7],62),(60,[0,4,7],65),(55,[0,4,7],68)], 0, 2)
    har += harm_block([(53,[0,4,7],65),(60,[0,4,7],62),(62,[0,3,7],65),(58,[0,4,7],68)], 32, 2)
    har += harm_block([(57,[0,3,7],65),(53,[0,4,7],62),(60,[0,4,7],65),(55,[0,4,7],68)], 64, 2)
    har += harm_block([(58,[0,4,7],62),(53,[0,4,7],60),(55,[0,3,7],62),(51,[0,4,7],58)], 96, 2)
    har += harm_block([(57,[0,3,7],65),(53,[0,4,7],62),(60,[0,4,7],65),(55,[0,4,7],68)], 128, 2)
    # Bass, counter, acc (generic)
    total_beats = 160
    for b in range(40):
        bas += [(45,85,b*4,.9),(45,70,b*4+2,.9)]
        ctr += [(57,68,b*4,2),(60,65,b*4+2,2)]
        for i,p in enumerate([48,52,55,52]):
            acc.append((p,50,b*4+i,0.9))
    mid.tracks += [_make_track("Melodía",1,mel,True,132),_make_track("Contramelodía",2,ctr),
                   _make_track("Armonía",3,har),_make_track("Bajo",4,bas),_make_track("Acomp.",5,acc)]
    mid.save(path)

def gen_midi_02_blues(path):
    """Blues 12-compases en A, forma AAB, 3 choruses"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # 12-bar blues progression: A7-D7-A7-A7-D7-D7-A7-A7-E7-D7-A7-E7
    blues_prog = [
        (57,[0,4,7,10],70),(57,[0,4,7,10],70),(57,[0,4,7,10],70),(57,[0,4,7,10],70),
        (62,[0,4,7,10],70),(62,[0,4,7,10],70),(57,[0,4,7,10],70),(57,[0,4,7,10],70),
        (64,[0,4,7,10],72),(62,[0,4,7,10],70),(57,[0,4,7,10],70),(64,[0,4,7,10],72),
    ]
    har = []
    for chorus in range(3):
        for bi,(root,ivs,vel) in enumerate(blues_prog):
            har += _chord_block(root,ivs,chorus*48+bi*4,3.8,vel)
    # Melody: pentatonic licks
    pentatonic = [57,60,62,64,67,69]
    mel = []
    for chorus in range(3):
        for bar in range(12):
            base = chorus*48+bar*4
            lick = [pentatonic[(bar*3+i)%6] for i in range(4)]
            for i,p in enumerate(lick):
                mel.append((p,85+chorus*3,base+i,0.9))
    bas, ctr, acc = [], [], []
    for b in range(36):
        bas += [(45,88,b*4,.5),(45,75,b*4+2,.5)]
        ctr += [(57,65,b*4,4)]
        for i,p in enumerate([45,52,57,52]):
            acc.append((p,52,b*4+i*0.5,0.45))
    mid.tracks += [_make_track("Guitar",1,mel,True,96),_make_track("Keys",2,ctr),
                   _make_track("Piano",3,har),_make_track("Bass",4,bas),_make_track("Rhythm",5,acc)]
    mid.save(path)

def gen_midi_03_waltz(path):
    """Vals en 3/4, La bemol mayor, forma A-B-A'-Coda"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    mid2 = mido.MidiFile(ticks_per_beat=480, type=1)
    # Waltz: 3 beats/bar
    TPB = 480
    def waltz_track(name,ch,notes,meta=False,bpm=160):
        t = mido.MidiTrack()
        t.name=name
        t.append(mido.MetaMessage('track_name',name=name,time=0))
        if meta:
            t.append(mido.MetaMessage('set_tempo',tempo=mido.bpm2tempo(bpm),time=0))
            t.append(mido.MetaMessage('time_signature',numerator=3,denominator=4,
                                      clocks_per_click=24,notated_32nd_notes_per_beat=8,time=0))
        events=[]
        for pitch,vel,start,dur in notes:
            s=int(start*TPB); e=int((start+dur)*TPB)
            events+=[(s,'on',pitch,vel,ch),(e,'off',pitch,0,ch)]
        events.sort(key=lambda x:(x[0],0 if x[1]=='off' else 1))
        prev=0
        for tick,kind,note,vel,c in events:
            d=tick-prev;prev=tick
            if kind=='on': t.append(mido.Message('note_on',note=note,velocity=vel,channel=c,time=d))
            else:          t.append(mido.Message('note_off',note=note,velocity=0,channel=c,time=d))
        t.append(mido.MetaMessage('end_of_track',time=0))
        return t
    # A: Ab major theme (bar=3 beats)
    mel=[]
    A_mel=[(68,88,0,1.5),(70,82,1.5,1.5),(72,85,3,3),(70,80,6,1.5),(68,82,7.5,1.5),(65,85,9,3),
           (63,82,12,1.5),(65,80,13.5,1.5),(68,85,15,3),(70,88,18,3),(68,90,21,3),(65,85,24,3)]
    mel+=A_mel
    # B: Eb major, higher
    B_mel=[(75,90,27,1.5),(77,85,28.5,1.5),(79,88,30,3),(77,82,33,1.5),(75,80,34.5,1.5),(72,85,36,3),
           (70,82,39,1.5),(72,80,40.5,1.5),(75,85,42,3),(77,88,45,3),(75,90,48,3),(72,85,51,3)]
    mel+=B_mel
    # A' reprise
    mel+=[(p,v,t+54,d) for p,v,t,d in A_mel]
    # Coda
    mel+=[(68,95,81,3),(65,90,84,3),(63,88,87,3),(60,85,90,6)]
    har,bas,ctr,acc=[],[],[],[]
    for b in range(32):
        base=b*3
        har+=_chord_block(56,[0,4,7],base,2.8,65)
        bas+=[(44,85,base,1),(44,65,base+1,1),(44,65,base+2,1)]
        ctr+=[(60,68,base,3)]
        acc+=[(56,50,base,1),(60,50,base+1,1),(63,50,base+2,1)]
    mid.tracks+=[waltz_track("Violín",1,mel,True,160),waltz_track("Viola",2,ctr),
                 waltz_track("Piano",3,har),waltz_track("Cello",4,bas),waltz_track("Arpa",5,acc)]
    mid.save(path)

def gen_midi_04_fugue(path):
    """Fuga a 4 voces en Re menor, sujeto + respuesta + episodios"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    subject = [(62,85,0,1),(65,80,.5,.5),(69,82,1,.5),(67,78,1.5,.5),(65,82,2,1),(62,78,3,1)]
    answer  = [(69,85,0,1),(72,80,.5,.5),(76,82,1,.5),(74,78,1.5,.5),(72,82,2,1),(69,78,3,1)]
    def voice(subj,offset,transp,start_bar):
        return [(p+transp,v,t+start_bar*4,d) for p,v,t,d in subj[offset:]+subj[:offset]]
    mel  = subject + [(p,v,t+16,d) for p,v,t,d in subject]  # S enters bar 0,4
    mel += [(p,v,t+32,d) for p,v,t,d in subject]             # episode bar 8
    mel += [(p,v,t+48,d) for p,v,t,d in answer]              # stretto bar 12
    ctr  = [(p,v,t+8,d) for p,v,t,d in answer]               # A enters bar 2
    ctr += [(p,v,t+24,d) for p,v,t,d in subject]
    ctr += [(p,v,t+40,d) for p,v,t,d in answer]
    har  = [(p-12,v,t+12,d) for p,v,t,d in subject]          # T enters bar 3
    har += [(p-12,v,t+28,d) for p,v,t,d in answer]
    har += [(p-12,v,t+44,d) for p,v,t,d in subject]
    bas  = [(p-24,v,t+16,d) for p,v,t,d in answer]           # B enters bar 4
    bas += [(p-24,v,t+32,d) for p,v,t,d in subject]
    bas += [(p-24,v,t+48,d) for p,v,t,d in answer]
    # Fill with sustained harmony
    har_chords=[]
    for b in range(16):
        har_chords+=_chord_block(57,[0,3,7],b*4,3.8,60)
    acc=[]
    for b in range(16):
        for i,p in enumerate([45,50,53,57]):
            acc.append((p,45,b*4+i,0.9))
    mid.tracks+=[_make_track("Soprano",1,mel,True,80),_make_track("Alto",2,ctr),
                 _make_track("Tenor",3,har+har_chords),_make_track("Bajo",4,bas),
                 _make_track("Continuo",5,acc)]
    mid.save(path)

def gen_midi_05_tango(path):
    """Tango en La menor, forma A-B-A con habanera y bandoneón"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Habanera rhythm: 1 . . 1 1 . 1 1 (dotted)
    def habanera(root, start, bars=4):
        n=[]
        for b in range(bars):
            base=start+b*4
            n+=[(root,85,base,1.5),(root,80,base+1.5,.5),(root,78,base+2,1),(root,75,base+3,1)]
        return n
    # Melody A: Am pentatonic with chromatic passing tones
    A_mel=[(69,90,0,1),(68,85,1,1),(67,88,2,.5),(65,82,2.5,.5),(64,85,3,1),
           (62,88,4,1),(64,85,5,1),(65,82,6,1),(67,88,7,1),
           (69,90,8,1),(71,92,9,1),(72,88,10,2),(69,85,12,4),
           (67,88,16,1),(68,85,17,1),(69,88,18,.5),(71,85,18.5,.5),(72,90,19,1),
           (74,92,20,1),(72,88,21,1),(71,85,22,1),(69,82,23,1),
           (67,85,24,1),(65,80,25,1),(64,82,26,1),(62,85,27,1),(57,90,28,4)]
    # B: D minor, more lyrical
    B_mel=[(65,85,32,2),(67,82,34,2),(69,85,36,1),(67,80,37,1),(65,82,38,2),
           (64,80,40,2),(62,82,42,2),(60,85,44,4),
           (62,88,48,1),(64,85,49,1),(65,88,50,1),(67,85,51,1),(69,90,52,4),
           (67,85,56,1),(65,80,57,1),(64,82,58,1),(62,80,59,1),(57,88,60,4)]
    # A reprise
    A2 = [(p,v,t+64,d) for p,v,t,d in A_mel]
    mel = A_mel+B_mel+A2
    har,bas,ctr,acc=[],[],[],[]
    for b in range(24):
        root=[57,57,57,57,62,62,57,57,64,62,57,64][b%12]
        ivs=[0,3,7] if b%4<2 else [0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,68)
        bas+=habanera(root-12,b*4,1)
        ctr+=[(root+7,65,b*4,2),(root+5,62,b*4+2,2)]
        acc+=[(root,48,b*4+i,0.9) for i in range(4)]
    mid.tracks+=[_make_track("Bandoneón",1,mel,True,92),_make_track("Violín",2,ctr),
                 _make_track("Piano",3,har),_make_track("Contrabajo",4,bas),
                 _make_track("Guitarra",5,acc)]
    mid.save(path)

def gen_midi_06_modal(path):
    """Modal jazz en Dorian (D Dorian), forma AABA 32 bars × 3 choruses"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    dorian = [62,64,65,67,69,71,72,74]
    def dorian_line(start, offset=0, vel_base=82):
        notes=[]
        for i in range(8):
            p=dorian[(i+offset)%8]
            notes.append((p,vel_base+i%3*3,start+i*0.5,0.45))
        return notes
    mel=[]
    for chorus in range(3):
        base=chorus*128
        # AA: D dorian vamp (8 bars each)
        for rep in range(2):
            for bar in range(8):
                mel+=dorian_line(base+rep*32+bar*4, bar%4, 80+chorus*4)
        # B: G dorian (8 bars)
        g_dorian=[55,57,58,60,62,64,65,67]
        for bar in range(8):
            p=g_dorian[bar%8]
            mel+=[(p,85,base+64+bar*4+i*0.5,0.45) for i in range(8)]
        # A: D dorian (8 bars)
        for bar in range(8):
            mel+=dorian_line(base+96+bar*4, bar%6, 82)
    # Harmony: Dm7 - Gm7 alternating
    har=[]
    for b in range(48):
        root=62 if b%8<4 else 55
        har+=_chord_block(root,[0,3,7,10],b*4,3.8,62)
    bas,ctr,acc=[],[],[]
    for b in range(48):
        bas+=[(50,88,b*4,.9),(50,72,b*4+2,.9)]
        ctr+=[(57,65,b*4,4)]
        for i,p in enumerate([50,57,62,57]):
            acc.append((p,50,b*4+i,0.9))
    mid.tracks+=[_make_track("Sax",1,mel,True,120),_make_track("Trompeta",2,ctr),
                 _make_track("Piano",3,har),_make_track("Contrabajo",4,bas),
                 _make_track("Batería",5,acc)]
    mid.save(path)

def gen_midi_07_minimalist(path):
    """Minimalismo (estilo Reich), Do mayor, adición de fases"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Short motif repeated with phase shift
    motif=[60,64,67,72,71,67,64,60]
    mel,ctr,har,bas,acc=[],[],[],[],[]
    total_bars=48
    for rep in range(total_bars*4):   # 1 rep = 1 beat
        for i,p in enumerate(motif):
            beat=rep+i*0.125
            mel.append((p,72+(rep%8),beat,0.12))
    # Phase voice: same motif shifted by 1/8 beat, gradually drifts
    for rep in range(total_bars*4):
        shift=rep*(1/64)
        for i,p in enumerate(motif):
            beat=rep+i*0.125+shift
            if beat < total_bars*4+4:
                ctr.append((p-12,65,beat,0.12))
    # Sustained harmony pads
    prog=[(60,[0,4,7],58),(67,[0,4,7],55),(65,[0,4,7],55),(62,[0,3,7],52)]
    for b in range(total_bars):
        root,ivs,vel=prog[b%4]
        har+=_chord_block(root,ivs,b*4,3.8,vel)
        bas+=[(root-12,80,b*4,2),(root-12,72,b*4+2,2)]
        for i,p in enumerate([48,52,55,60]):
            acc.append((p,45,b*4+i,0.9))
    mid.tracks+=[_make_track("Voz1",1,mel,True,132),_make_track("Voz2",2,ctr),
                 _make_track("Pad",3,har),_make_track("Bajo",4,bas),
                 _make_track("Pulse",5,acc)]
    mid.save(path)

def gen_midi_08_flamenco(path):
    """Flamenco (Phrygian, Mi), forma A-B-A con falseta y compás por 12"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Phrygian E: E F G A B C D E
    phryg=[64,65,67,69,71,72,74,76]
    # Falseta A (12-beat compás × 4)
    def falseta(start):
        n=[]
        pat=[64,65,64,62,64,65,67,65,64,62,60,62]
        for i,p in enumerate(pat):
            n.append((p,88-i%3*4,start+i,0.9))
        return n
    mel=[]
    for rep in range(4):
        mel+=falseta(rep*12)
    # B section: more lyrical, higher
    B_mel=[(76,90,48,1.5),(74,85,49.5,1.5),(72,88,51,1.5),(71,82,52.5,1.5),
           (69,85,54,3),(71,88,57,3),(72,90,60,6),
           (74,88,66,1.5),(72,85,67.5,1.5),(71,88,69,1.5),(69,82,70.5,1.5),
           (67,85,72,3),(69,88,75,3),(71,90,78,6)]
    mel+=B_mel
    # Reprise A
    mel+=[(p,v,t+84,d) for p,v,t,d in [(p,v,t,d) for p,v,t,d in [item for item in [falseta(0)+falseta(12)][0]]]]
    # Harmony: Am - G - F - E (Andalusian cadence)
    har=[]
    for b in range(28):
        root=[57,55,53,52][b%4]
        ivs=[0,3,7] if b%4<3 else [0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,68)
    bas,ctr,acc=[],[],[]
    for b in range(28):
        rt2 = [57,55,53,52][b%4]
        bas+=[(rt2-12,85,b*4,1)]
        ctr+=[(rt2+7,65,b*4,2),(rt2+5,62,b*4+2,2)]
        for i in range(4):
            acc.append((rt2,50,b*4+i,0.9))
    mid.tracks+=[_make_track("Guitarra",1,mel,True,88),_make_track("Cante",2,ctr),
                 _make_track("Armonía",3,har),_make_track("Bajo",4,bas),
                 _make_track("Palmas",5,acc)]
    mid.save(path)

def gen_midi_09_march(path):
    """Marcha en Sol mayor, forma intro-A-B-Trio-A-B-Coda"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Intro: fanfare (4 bars)
    intro=[(67,95,0,1),(71,90,1,1),(74,92,2,1),(79,95,3,1),
           (79,90,4,.5),(77,85,4.5,.5),(79,90,5,1),(77,85,6,1),(74,88,7,1)]
    # A theme (8 bars): martial melody
    A_mel=[(67,88,8,1),(69,82,9,1),(71,85,10,1),(72,88,11,1),(74,90,12,2),(72,85,14,2),
           (71,88,16,1),(69,82,17,1),(67,85,18,1),(65,82,19,1),(64,85,20,2),(67,88,22,2),
           (69,85,24,1),(71,82,25,1),(72,85,26,1),(74,88,27,1),(76,90,28,2),(74,85,30,2),
           (72,88,32,.5),(71,85,32.5,.5),(69,82,33,1),(67,85,34,1),(65,82,35,1),(67,90,36,4)]
    # B theme (8 bars): contrasting
    B_mel=[(74,85,40,2),(72,80,42,2),(71,82,44,2),(69,80,46,2),
           (67,85,48,2),(69,82,50,2),(71,85,52,2),(74,88,54,2),
           (76,90,56,1),(74,85,57,1),(72,82,58,1),(71,80,59,1),(67,85,60,4),
           (69,88,64,.5),(71,85,64.5,.5),(72,82,65,1),(74,85,66,1),(76,88,67,1),(79,92,68,4)]
    # Trio (8 bars): in C major, softer
    trio_mel=[(72,78,72,2),(74,75,74,2),(76,78,76,2),(77,75,78,2),
              (79,80,80,2),(77,75,82,2),(76,78,84,2),(74,72,86,2),
              (72,75,88,2),(71,72,90,2),(69,75,92,2),(67,72,94,2),
              (69,75,96,2),(71,78,98,2),(72,80,100,2),(74,82,102,2),
              (76,85,104,4),(74,80,108,4),(72,78,112,8)]
    mel=intro+A_mel+B_mel+trio_mel
    # Harmony, bass, counter, acc
    har,bas,ctr,acc=[],[],[],[]
    for b in range(30):
        root=[55,55,60,55, 55,62,55,62][b%8]
        ivs=[0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,65)
        bas+=[(root-12,90,b*4,1),(root-12,72,b*4+2,1)]
        ctr+=[(root+4,68,b*4,2),(root+7,65,b*4+2,2)]
        for i in range(4):
            acc.append((root,50,b*4+i,0.9))
    mid.tracks+=[_make_track("Trompeta",1,mel,True,120),_make_track("Trombón",2,ctr),
                 _make_track("Armonía",3,har),_make_track("Tuba",4,bas),
                 _make_track("Tambor",5,acc)]
    mid.save(path)

def gen_midi_10_sonata(path):
    """Forma sonata: Expo(P-T-S-K) - Desarrollo - Recapitulación, Do mayor"""
    import mido
    mid = mido.MidiFile(ticks_per_beat=480, type=1)
    # Primary theme P (C major, bars 0-7)
    P=[(60,90,0,.5),(64,85,.5,.5),(67,88,1,.5),(72,90,1.5,.5),(74,88,2,1),(72,85,3,1),
       (71,88,4,.5),(69,82,4.5,.5),(67,85,5,1),(65,80,6,1),(64,85,7,1),
       (60,88,8,.5),(62,82,.5+8,.5),(64,85,9,.5),(65,82,9.5,.5),(67,85,10,2),(60,80,12,4)]
    # Transition T (bars 8-11): modulating to G
    T=[(62,85,16,1),(64,82,17,1),(65,85,18,1),(67,88,19,1),(69,90,20,1),(71,85,21,1),(72,82,22,2)]
    # Secondary theme S (G major, bars 12-19): lyrical
    S=[(67,80,24,2),(69,78,26,2),(71,80,28,2),(72,78,30,2),
       (74,82,32,2),(72,78,34,2),(71,80,36,2),(69,78,38,2),
       (67,82,40,4),(71,78,44,4),(74,80,48,4),(72,75,52,4)]
    # Closing K (bars 20-23): cadential
    K=[(74,85,56,1),(72,80,57,1),(71,82,58,1),(69,80,59,1),(67,88,60,4),(55,92,64,8)]
    # Development (bars 24-35): fragmentation
    dev=[]
    frag=P[:4]
    for step in range(12):
        transp=[0,2,4,5,7,9,10,12,7,5,2,0][step]
        dev+=[(p+transp,85,t+72+step*4,d) for p,v,t,d in frag]
    # Recapitulation (bars 36-51): P+T+S in C
    recap_P=[(p,v,t+120,d) for p,v,t,d in P]
    recap_S=[(p-7,v,t+144,d) for p,v,t,d in S]  # S transposed back to C
    recap_K=[(p-7,v,t+168,d) for p,v,t,d in K]
    mel=P+T+S+K+dev+recap_P+recap_S+recap_K
    har,bas,ctr,acc=[],[],[],[]
    for b in range(46):
        root=[60,60,67,60, 60,62,60,67][b%8]
        ivs=[0,4,7]
        har+=_chord_block(root,ivs,b*4,3.8,65)
        bas+=[(root-12,85,b*4,1.5),(root-12,70,b*4+2,1.5)]
        ctr+=[(root+4,65,b*4,2),(root+7,62,b*4+2,2)]
        for i in range(4):
            acc.append((root,48,b*4+i,0.9))
    mid.tracks+=[_make_track("Piano",1,mel,True,116),_make_track("Violín",2,ctr),
                 _make_track("Armonía",3,har),_make_track("Cello",4,bas),
                 _make_track("Continuo",5,acc)]
    mid.save(path)


BATCH_GENERATORS = [
    ("01_rondo_Dm",      gen_midi_01_rondo,      "Rondo A-B-A-C-A, Re menor"),
    ("02_blues_Am",      gen_midi_02_blues,       "Blues 12 compases, 3 choruses"),
    ("03_waltz_Ab",      gen_midi_03_waltz,       "Vals 3/4, La bemol mayor"),
    ("04_fugue_Dm",      gen_midi_04_fugue,       "Fuga a 4 voces, Re menor"),
    ("05_tango_Am",      gen_midi_05_tango,       "Tango, La menor"),
    ("06_modal_Dorian",  gen_midi_06_modal,       "Modal jazz, D Dorian, AABA × 3"),
    ("07_minimalist_C",  gen_midi_07_minimalist,  "Minimalismo en fase, Do mayor"),
    ("08_flamenco_E",    gen_midi_08_flamenco,    "Flamenco Frigio, Mi"),
    ("09_march_G",       gen_midi_09_march,       "Marcha, Sol mayor"),
    ("10_sonata_C",      gen_midi_10_sonata,      "Forma sonata, Do mayor"),
]


def generate_batch(out_dir: str, cfg: dict = None):
    import os
    os.makedirs(out_dir, exist_ok=True)
    cfg = cfg or {}
    for slug, gen_fn, desc in BATCH_GENERATORS:
        mid_path  = os.path.join(out_dir, f"{slug}.mid")
        html_path = os.path.join(out_dir, f"{slug}_report.html")
        print(f"\n  ── {slug}: {desc}")
        try:
            gen_fn(mid_path)
            html = analyze(mid_path, cfg)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"     ✓ {html_path}")
        except Exception as e:
            import traceback
            print(f"     ✗ ERROR: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Generador de MIDIs de prueba")
    parser.add_argument("-o", "--output", default="batch", help="Directorio de salida")
    parser.add_argument("--list", action="store_true", help="Lista los MIDIs disponibles")
    parser.add_argument("--no-report", action="store_true", help="Solo genera MIDIs, sin HTML")
    args = parser.parse_args()

    if args.list:
        print("MIDIs disponibles:")
        for slug, _, desc in BATCH_GENERATORS:
            print(f"  {slug}: {desc}")
        return

    os.makedirs(args.output, exist_ok=True)

    if args.no_report:
        for slug, gen_fn, desc in BATCH_GENERATORS:
            path = os.path.join(args.output, f"{slug}.mid")
            print(f"  Generando {slug}...")
            gen_fn(path)
    else:
        try:
            from midi_pianoroll_analyzer import analyze
        except ImportError:
            sys.exit("ERROR: midi_pianoroll_analyzer.py no encontrado en el mismo directorio")
        generate_batch(args.output)


if __name__ == "__main__":
    main()
