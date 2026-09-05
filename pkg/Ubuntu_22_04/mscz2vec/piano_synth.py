#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════╗
║  piano_synth.py — physics-based piano & keyboard synthesis toolkit      ║
╠════════════════════════════════════════════════════════════════════════╣
║  Grand Piano (modal synthesis) · Rhodes FM · Prism · DDSP comparison    ║
║  Reference-IR extraction · gradient/DE optimizers · spectral compare    ║
╚════════════════════════════════════════════════════════════════════════╝

Single self-contained tool consolidating the former generators/*.py scripts.
All synthesis and optimization logic is unchanged from the original scripts;
only the invocation surface, phase-table persistence, and dry-synthesis
toggle (formerly a monkeypatch) were adapted for a single-file CLI.

USAGE
    python3 piano_synth.py <command> [options]

GENERATE SAMPLES  (write MP3s to ../audio/<instrument>/)
    generate-grand    [--velocity-layers] [--no-ir]
    generate-rhodes   [--velocity-layers]
    generate-prism    [--velocity-layers]
    generate-ddsp     [--epochs N] [--generate-only]

RENDER A MIDI FILE
    render-midi <file.mid>  [--instrument {grand,rhodes,prism}] [--output PATH]
                            [--speed F] [--velocity-scale F]

FETCH REFERENCE RECORDINGS  (populates ../audio/piano|rhodes|salamander/)
    fetch-references  [--piano-only] [--rhodes-only] [--force]
                       [--piano-velocity N] [--rhodes-velocity N]

EXTRACT REFERENCE TRANSFER FUNCTIONS  (from ../audio/piano|rhodes/*.mp3)
    extract-soundboard-ir             grand piano soundboard IR (Salamander refs)
    extract-rhodes-tf                 Rhodes pickup/amp IR (jRhodes3d refs)

OPTIMIZE PARAMETERS
    optimize-phases   [--iters N]     grand piano phase table (gradient descent)
                                       writes straight to piano_synth_config.json —
                                       generate-grand picks it up on next run
    optimize-grand    [--all-notes]   grand piano physical params (diff. evolution)
                                       prints a paste-ready block (not auto-applied)
    optimize-rhodes                   FM Rhodes params (diff. evolution)
                                       prints a paste-ready block (not auto-applied)
    tune-warmth       {rolloff,bridge} [--rolloff-base F] [--rolloff-linear F]
                       [--rolloff-cubic F]   grand piano warmth/brightness sweep

COMPARE / ANALYZE
    compare-grand                     recorded vs generated grand piano (envelope+spectrum)
    compare-rhodes                    sampled vs FM Rhodes (spectral peaks)
    analyze-comparison                grand piano vs Salamander, full metric suite
    deep-compare                      sampled vs FM Rhodes (MFCC + envelope, needs librosa)

Examples:
    python3 piano_synth.py generate-grand --velocity-layers
    python3 piano_synth.py optimize-phases --iters 400
    python3 piano_synth.py compare-grand
"""

import argparse
import json
import os
import sys
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'piano_synth_config.json')
_CONFIG_CACHE = None


def _load_config():
    """Read piano_synth_config.json (cached). Returns {} if it doesn't exist yet."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH) as f:
                _CONFIG_CACHE = json.load(f)
        else:
            _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def _save_config(updates):
    """Merge `updates` into piano_synth_config.json and write it back."""
    cfg = _load_config()
    cfg.update(updates)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    return cfg


# ─── shared helpers for the extract-* commands (small enough not to warrant
#     their own namespace factory) ───────────────────────────────────────

def _load_mp3(path, sample_rate=44100):
    import numpy as np
    import subprocess, tempfile, wave
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run([
        'ffmpeg', '-y', '-i', path, '-ar', str(sample_rate),
        '-ac', '1', '-f', 'wav', tmp_path
    ], capture_output=True)
    with wave.open(tmp_path, 'r') as wf:
        raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    os.unlink(tmp_path)
    return audio


def _spectral_envelope(signal, n_fft, hop=2048):
    import numpy as np
    n = len(signal)
    window = np.hanning(n_fft)
    mag_sum = np.zeros(n_fft // 2 + 1)
    count = 0
    for start in range(0, n - n_fft, hop):
        frame = signal[start:start + n_fft] * window
        spec = np.fft.rfft(frame)
        mag_sum += np.abs(spec)
        count += 1
    if count == 0:
        return np.ones(n_fft // 2 + 1)
    return mag_sum / count


def _smooth_spectrum(spec, window_size=32):
    import numpy as np
    kernel = np.ones(window_size) / window_size
    padded = np.pad(spec, window_size // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(spec)]


# ─── namespace accessors (lazy-build + cache each instrument/task module) ──

_NS_CACHE = {}


def _ns(key, builder):
    if key not in _NS_CACHE:
        _NS_CACHE[key] = builder()
    return _NS_CACHE[key]


def grand():             return _ns('grand', _build_grand)
def rhodes():             return _ns('rhodes', _build_rhodes)
def prism():               return _ns('prism', _build_prism)
def ddsp():                 return _ns('ddsp', _build_ddsp)
def optphases():             return _ns('optphases', _build_optphases)
def optgrand():                return _ns('optgrand', _build_optgrand)
def optrhodes():                 return _ns('optrhodes', _build_optrhodes)
def tunewarmth():                  return _ns('tunewarmth', _build_tunewarmth)
def comparepiano():                  return _ns('comparepiano', _build_comparepiano)
def comparerhodes():                   return _ns('comparerhodes', _build_comparerhodes)
def deepcompare():                       return _ns('deepcompare', _build_deepcompare)
def analyzecomparison():                   return _ns('analyzecomparison', _build_analyzecomparison)

def _build_grand():
    """
    Generate Grand Piano samples using physics-based modal synthesis.

    Calibrated from measured parameters in the literature:
      - Bensa, Bilbao, Kronland-Martinet & Smith (JASA, 2003):
        String stiffness (epsilon), damping coefficients (b1, b2), string lengths
      - Chaigne & Askenfelt (JASA, 1994):
        Hammer force law F=K·x^p, hammer masses, strike positions
      - Weinreich, "Coupled Piano Strings" (JASA, 1977):
        Two-stage decay from coupled string modes (prompt/aftersound)
      - Bank & Sujbert (JASA, 2005):
        Phantom partials from longitudinal string vibrations
      - Steinway B inharmonicity measurements (U. Alabama Huntsville):
        A0: B≈0.00031, A3: B≈0.00021, A4: B≈0.00075

    Physical model per note:
      1. INHARMONIC PARTIALS: f_n = n·f₀·√(1 + B·n²)
      2. CALIBRATED DAMPING: α_n = b₁ + b₂·(nπ/L)²  (Bensa et al.)
      3. TWO-STAGE DECAY: prompt (soundboard-coupled) + aftersound (decoupled)
      4. NONLINEAR HAMMER: velocity→contact duration→spectral tilt (F=Kx^p)
      5. MULTIPLE STRINGS: 1-3 per note, slight detuning → beating + chorus
      6. HAMMER ATTACK TRANSIENT: soundboard impulse approximation
      7. PHANTOM PARTIALS: sum-frequency longitudinal modes in bass register

    Output: MP3 files ready for the learn-piano.html sampler.
    """

    import numpy as np
    import os
    import subprocess

    SAMPLE_RATE = 44100
    DURATION = 6.0  # enough for full two-stage decay

    # Per-note soundboard IRs. Two formats supported:
    #   'irs'                — time-domain IRs (synthesize_soundboard_ir.py)
    #   'transfer_functions' — magnitude responses (extract_soundboard_ir.py, legacy)
    _TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
    _SOUNDBOARD_IR = None  # kept for compatibility check in optimizer
    _TF_MODE = None
    if os.path.exists(_TF_PATH):
        _tf_data = np.load(_TF_PATH)
        if 'irs' in _tf_data:
            _TF_MIDI   = _tf_data['midi_points']
            _TF_IRS    = _tf_data['irs']           # [n_notes, ir_length]
            _TF_IR_LEN = int(_tf_data['ir_length'])
            _IR_CACHE  = {}
            _TF_MODE   = 'irs'
        elif 'transfer_functions' in _tf_data:
            _TF_MIDI   = _tf_data['midi_points']
            _TF_FREQS  = _tf_data['freqs']
            _TF_MAG    = _tf_data['transfer_functions']  # [n_notes, n_bins]
            _TF_IR_LEN = int(_tf_data['ir_length'])
            _IR_CACHE  = {}
            _TF_MODE   = 'mag'
        else:
            _tf_data = None
    else:
        _tf_data = None

    NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']


    def midi_to_name(midi):
        return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


    # Every chromatic note from B1 (35) to D6 (86)
    NOTES = [(m, midi_to_name(m)) for m in range(35, 87)]

    # ═══ MEASURED PARAMETERS FROM BENSA ET AL. (2003) ═══
    # Reference notes with calibrated values
    # Format: midi → (b1, b2, string_length_m)
    #   b1: frequency-independent damping (s⁻¹)
    #   b2: frequency-dependent damping (s)
    #   L:  vibrating string length (m)
    CALIB_NOTES = {
        36: {'b1': 0.25,  'b2': 7.5e-5,  'L': 1.92},  # C2
        60: {'b1': 1.1,   'b2': 2.7e-4,  'L': 0.62},  # C4
        96: {'b1': 9.17,  'b2': 2.1e-3,  'L': 0.09},  # C7
    }
    CALIB_MIDI = sorted(CALIB_NOTES.keys())

    # Per-partial per-string phase table, optimized via gradient descent
    # (optimize_phases.py) against recorded reference samples using mel-scale
    # STFT loss. Re-run optimize_phases.py after changing any synthesis parameters
    # (rolloff, bridge hill, decay, etc.) since phases are coupled to them.
    _PHASE_TABLE_DEFAULT = np.array([
        [4.0132, 2.4928, 1.6716, 0.8096, 0.9535, 5.3968, 3.8574, 2.4742, 4.9981, 1.8611, 3.7908, 5.9255, 0.6455, 0.3276, 5.8913, 5.4893, 4.6771, 1.2357, 3.2581, 2.7561, 0.8428, 4.0558, 5.2799, 0.9260, 0.6861, 5.1939, 1.2753, 1.4505, 0.1271, 3.7157, 0.7629, 1.4660, 0.2760, 2.8631, 4.7170, 4.0562, 1.4049, 2.9267, 2.4257, 0.3066, 1.6580, 1.7747, 5.1121, 0.7141, 0.1317, 3.3997, 4.9164, 4.2310, 0.9365, 1.5663, 6.2455, 1.8572, 2.0187, 5.6055, 1.2101, 0.2159, 2.9470, 3.6037, 1.2411, 3.0013, 2.1811, 4.8246, 5.0224, 3.2362],
        [1.3617, 4.1407, 3.1679, 2.0926, 2.8498, 5.8210, 4.4870, 2.3379, 5.2825, 2.0538, 4.1686, 6.1295, 0.6556, 0.3439, 5.9858, 5.2892, 4.1228, 0.8159, 2.6249, 2.3138, 0.0429, 3.2546, 4.8054, 1.0011, 5.9662, 4.5308, 0.1382, 0.4243, 5.1007, 2.6385, 5.3386, 6.0777, 4.8098, 2.4797, 3.5437, 2.2902, 5.5527, 0.8090, 0.7646, 4.7900, 5.6842, 5.9511, 2.8493, 4.6506, 4.0068, 1.0178, 2.3836, 1.5110, 4.2781, 5.6876, 3.9259, 5.7782, 5.7022, 2.6450, 4.3279, 3.0211, 5.3764, 6.0428, 3.6381, 5.4519, 4.6440, 0.9308, 1.0450, 5.1668],
        [5.3995, 5.6364, 4.5929, 3.5345, 4.3810, 0.1389, 5.3429, 2.5298, 5.5007, 2.4105, 4.6763, 0.2695, 0.9782, 0.8100, 0.1505, 5.2394, 3.1700, 0.0055, 1.8871, 2.4310, 2.5955, 1.1359, 4.7438, 1.2807, 0.7266, 3.9978, 5.2003, 5.8520, 3.4716, 1.8749, 3.4160, 4.3617, 2.2013, 2.4004, 2.6537, 0.4599, 3.2560, 4.7558, 5.4229, 3.1209, 2.9916, 3.3205, 0.2833, 2.1776, 1.5710, 4.9177, 6.1946, 5.1710, 0.3391, 0.8279, 0.8513, 4.7880, 0.2794, 6.2582, 3.6578, 5.5488, 4.6602, 1.7406, 2.8100, 1.5382, 3.6269, 5.0149, 3.2789, 5.5434],
    ])
    _PHASE_TABLE = np.array(_load_config().get('grand_phase_table', _PHASE_TABLE_DEFAULT.tolist()), dtype=np.float64)


    def _minimum_phase_ir(magnitude_response, ir_length):
        """Create minimum-phase FIR from magnitude response (cepstral method)."""
        mag = np.maximum(magnitude_response, 1e-10)
        n_fft = (len(mag) - 1) * 2
        full_mag = np.concatenate([mag, mag[-2:0:-1]])
        log_mag = np.log(full_mag)
        cepstrum = np.fft.ifft(log_mag).real
        min_cep = np.zeros_like(cepstrum)
        min_cep[0] = cepstrum[0]
        min_cep[1:n_fft // 2] = 2 * cepstrum[1:n_fft // 2]
        min_cep[n_fft // 2] = cepstrum[n_fft // 2]
        ir = np.fft.ifft(np.exp(np.fft.fft(min_cep))).real
        ir = ir[:ir_length]
        ir *= np.hanning(ir_length * 2)[ir_length:]
        return ir


    def _get_soundboard_ir(midi):
        """Get per-note soundboard IR, interpolated from reference notes."""
        if _tf_data is None:
            return None
        if midi in _IR_CACHE:
            return _IR_CACHE[midi]

        if _TF_MODE == 'irs':
            # Time-domain IRs (from synthesize_soundboard_ir.py): linear interp
            idx = np.searchsorted(_TF_MIDI, midi)
            if idx == 0:
                ir = _TF_IRS[0].copy()
            elif idx >= len(_TF_MIDI):
                ir = _TF_IRS[-1].copy()
            else:
                lo, hi = idx - 1, idx
                t = (midi - _TF_MIDI[lo]) / (_TF_MIDI[hi] - _TF_MIDI[lo])
                ir = (1.0 - t) * _TF_IRS[lo] + t * _TF_IRS[hi]
        else:
            # Magnitude responses (legacy extract_soundboard_ir.py): log interp → min-phase FIR
            log_tfs = np.log(_TF_MAG + 1e-10)
            interp_log = np.zeros(log_tfs.shape[1])
            for i_bin in range(log_tfs.shape[1]):
                interp_log[i_bin] = np.interp(midi, _TF_MIDI, log_tfs[:, i_bin])
            mag_response = np.exp(interp_log)
            ir = _minimum_phase_ir(mag_response, _TF_IR_LEN)
            ir_energy = np.sqrt(np.sum(ir ** 2))
            if ir_energy > 0:
                n_fft = (len(_TF_FREQS) - 1) * 2
                ir = ir / ir_energy * np.sqrt(_TF_IR_LEN) / np.sqrt(n_fft)

        _IR_CACHE[midi] = ir
        return ir


    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)


    def interp_param(midi, param):
        """Log-linear interpolation of a calibrated parameter across the keyboard."""
        vals = [CALIB_NOTES[m][param] for m in CALIB_MIDI]
        return np.exp(np.interp(midi, CALIB_MIDI, np.log(vals)))


    def generate_grand_piano_note(midi, duration=DURATION, velocity=0.8, use_ir=True):
        """Generate a grand piano note using physics-based modal synthesis."""
        freq = midi_to_freq(midi)
        n_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Reproducible randomness per note
        rng = np.random.RandomState(midi * 1000 + 42)

        # Key position 0-1 across keyboard (A0=21 to C8=108)
        key_pos = np.clip((midi - 21) / 87, 0, 1)

        # ═══ CALIBRATED STRING PARAMETERS ═══
        b1 = interp_param(midi, 'b1')     # frequency-independent damping
        b2 = interp_param(midi, 'b2')     # frequency-dependent damping
        L = interp_param(midi, 'L')       # string length
        piL2 = (np.pi / L) ** 2           # spatial frequency factor

        # ═══ INHARMONICITY COEFFICIENT B ═══
        # Calibrated from Steinway B grand measurements (U. Alabama Huntsville):
        #   A0: 0.00031, A3: 0.00021, A4: 0.00075
        # Wound strings (A0-A3): B dips in mid-bass (optimal string design)
        # Plain strings (A4+): B rises steeply as strings get shorter/stiffer
        # Log-interpolation between measured/estimated calibration points.
        _B_midi = [21,   33,    45,    57,    69,    84,    96]
        _B_vals = [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2]
        B = np.exp(np.interp(midi, _B_midi, np.log(_B_vals)))

        # ═══ MAX PARTIALS (up to Nyquist) ═══
        max_partial = 1
        while max_partial * freq * np.sqrt(1 + B * max_partial**2) < SAMPLE_RATE / 2 - 500:
            max_partial += 1
        max_partial = min(max_partial - 1, 64)

        # ═══ HAMMER MODEL (Chaigne & Askenfelt, Russell & Rossing 1998) ═══
        # Nonlinear felt: F = K·x^p with three-layer hardness gradient.
        # Real hammer felt has soft outer surface → medium → hard inner core.
        # pp only compresses soft surface (dark tone); ff reaches hard core (bright).
        # p exponent: bass ~2.0, treble ~3.5 (Russell & Rossing measurements).
        vel_clamp = max(velocity, 0.05)

        # [1] VELOCITY-DEPENDENT HARDNESS (nonlinear felt F=Kx^p)
        # Effective felt stiffness increases nonlinearly with compression depth.
        # Concave curve models the three-layer gradient: slow change pp→mp,
        # accelerating change mf→ff as harder inner layers engage.
        # Floor of 0.15 ensures pp retains some harmonic character (not pure sine).
        # pp(0.15)→0.20, mp(0.6)→0.54, f(0.8)→0.76, ff(0.9)→0.88, fff(1.0)→1.0
        effective_hardness = 0.15 + 0.85 * vel_clamp ** 1.5

        # [2] THREE-LAYER CONTACT TIME (Chaigne & Askenfelt 1994, Fig. 5)
        # T_c varies ~3x from pp to fff. Soft surface gives longer contact (darker),
        # hard core gives shorter contact (brighter).
        # Shortened vs previous (was 4ms/2.5ms/0.8ms) — analysis showed partials
        # h5+ were 10-30 dB too weak at A4, meaning the hammer filter cutoff was
        # too low. Shorter contact = higher cutoff = more surviving upper partials.
        T_c_base = np.interp(midi, [36, 60, 96], [0.003, 0.0018, 0.0006])
        T_c = T_c_base * (2.5 - 1.9 * effective_hardness)
        hammer_cutoff = 2.5 / T_c

        # [3] VELOCITY-DEPENDENT ROLLOFF STEEPNESS
        # Soft felt (pp) acts as a low-pass with steeper rolloff — fewer harmonics.
        # Hard felt (ff) has gentle rolloff — rich harmonic content.
        # This is the primary timbral mechanism: pp sounds "round", ff sounds "brilliant".
        # Reduced steepness (was 2.4-1.2) so upper partials survive better.
        hammer_rolloff_exp = 2.0 - 0.8 * effective_hardness
        # pp(0.15): 1.84 (steep), mp(0.6): 1.57, f(0.8): 1.39, fff(1.0): 1.20 (gentle)

        # ═══ STRIKE POSITION (Chaigne & Askenfelt) ═══
        # Bass: 0.12 of string length, treble: 0.0625
        strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

        # ═══ TWO-STAGE DECAY (Weinreich) ═══
        # [4] VELOCITY-DEPENDENT PROMPT/AFTERSOUND RATIO
        # Harder strikes couple more energy into the soundboard (stronger prompt),
        # but the prompt also decays faster because energy radiates away quickly.
        # Soft strikes: less initial energy transfer, more stays as aftersound.
        # Reduced prompt_factor vs previous: analysis showed prompt/aftersound ratio
        # was 3-8x too extreme in treble (Ds4: 36x vs Salamander's 6x).
        prompt_factor = (1.2 + 0.3 * key_pos) * (0.7 + 0.5 * effective_hardness)
        # Increased after_factor: aftersound was decaying too slowly (t_40dB 1.65x
        # longer than Salamander). Higher rate = faster tail decay.
        after_factor = 0.45
        # Aftersound fraction: slightly higher at pp (more energy stays in strings)
        A_after = (0.18 + 0.07 * key_pos) * (1.3 - 0.4 * effective_hardness)
        # pp: A_after ~0.24 (more sustain), fff: A_after ~0.17 (more prompt)

        # ═══ NUMBER OF STRINGS & DETUNING ═══
        if midi < 36:      # monochord (1 string)
            string_detunes = [0.0]
        elif midi < 48:    # bichord (2 strings)
            dc = 0.3 + 0.3 * key_pos
            string_detunes = [-dc, dc]
        else:              # trichord (3 strings)
            dc = 0.15 + 0.25 * key_pos
            string_detunes = [-dc, 0.0, dc]

        n_strings = len(string_detunes)

        # ═══ MODAL SYNTHESIS: partials per string ═══
        signal = np.zeros(n_samples)

        for s_idx, d_cents in enumerate(string_detunes):
            detune_ratio = 2 ** (d_cents / 1200)
            string_amp = 1.0 / n_strings

            for n in range(1, max_partial + 1):
                # Inharmonic partial frequency
                partial_freq = n * freq * detune_ratio * np.sqrt(1 + B * n**2)
                if partial_freq >= SAMPLE_RATE / 2:
                    break

                # --- Initial amplitude ---
                # Bass: moderate rolloff preserves harmonics without overwhelming fundamental.
                # Mid/treble: gentle rolloff — upper partials need to survive the hammer
                # filter and still be audible. Salamander A4 h8 is -22dB (not -54dB).
                # The hammer filter (below) provides most of the HF shaping, so
                # the base rolloff should be gentle, letting the hammer physics
                # determine brightness rather than an aggressive power law.
                rolloff = 1.2 + 0.3 * key_pos + 1.2 * key_pos ** 2
                # B1(0.16): 1.28, C4(0.45): 1.58, A4(0.55): 1.73, A5(0.69): 1.98, D6(0.75): 2.10
                # +0.2 offset vs v2: leaves fundamental unchanged, -1 to -5 dB on upper partials
                amp = 1.0 / (n ** rolloff)

                # Hammer spectral shaping (Chaigne & Askenfelt)
                # Velocity-dependent rolloff: steep at pp (soft felt), gentle at ff (hard felt).
                # Modulated by shallow half-cosine dips at 1.5/Tc, 2.5/Tc.
                # Real felt nonlinearity (F=Kx^p) fills theoretical nulls — dips ~3 dB.
                amp *= 1.0 / (1.0 + (partial_freq / hammer_cutoff) ** hammer_rolloff_exp)
                fTc = partial_freq * T_c
                denom = 1.0 - 4.0 * fTc * fTc
                if abs(denom) < 1e-6:
                    cosine_mod = 1.0
                else:
                    cosine_mod = min(abs(np.cos(np.pi * fTc) / denom), 1.0)
                amp *= 0.7 + 0.3 * cosine_mod

                # Strike position node suppression
                strike_factor = abs(np.sin(np.pi * n * strike_pos))
                amp *= max(strike_factor, 0.03)

                # Soundboard spectral envelope (Smith & Van Duyne 1995,
                # Boutillon & Ege 2013). The soundboard's resonance structure
                # colors each partial — the piano's "voice."
                # Below ~300 Hz: distinct global modes add warmth.
                # 1-4 kHz: "bridge hill" — broad resonance that gives brightness
                # (Giordano 1998, Conklin 1996).
                f_hz = partial_freq
                sb_response = 1.0
                for cf, bw, gain in [(90, 30, 0.15), (170, 35, 0.12),
                                      (260, 45, 0.10)]:
                    sb_response += gain * np.exp(-0.5 * ((f_hz - cf) / bw) ** 2)
                sb_response += 0.40 * np.exp(-0.5 * ((f_hz - 1800) / 800) ** 2)
                amp *= sb_response

                amp *= string_amp

                # --- Per-partial decay rate ---
                # Three-term model (Desvages & Bilbao 2016, Issanchou et al. 2017)
                # Splits Bensa's b₁ into support coupling + air viscosity.
                # Air drag ∝ 1/√f creates a mid-frequency dip where harmonics
                # h2-h5 sustain longer than the fundamental — key piano trait.
                # At the fundamental, total = b₁ (unchanged from Bensa).
                K_n = (n ** 2) * piL2
                air_frac = 0.2
                alpha_n = (b1 * (1.0 - air_frac)
                           + b1 * air_frac * np.sqrt(freq / max(partial_freq, 20.0))
                           + b2 * K_n)

                # Two-stage envelope: prompt + aftersound
                # Partial-dependent coupling (Bank et al. 2010, Miranda Valiente 2024):
                # partials near soundboard resonances couple more strongly →
                # faster prompt decay, less energy remains as aftersound.
                # Creates natural timbre evolution: brightness fades before fundamental.
                # Use sqrt(sb_response) to moderate coupling — full sb_response made
                # treble prompt/aftersound ratio 3-8x too extreme vs Salamander.
                sb_coupling = np.sqrt(sb_response)
                prompt_rate = alpha_n * prompt_factor * sb_coupling
                after_rate = alpha_n * after_factor  # decoupled mode — unaffected
                A_after_n = A_after / sb_coupling
                A_prompt_n = 1.0 - A_after_n
                env = A_prompt_n * np.exp(-t * prompt_rate) + A_after_n * np.exp(-t * after_rate)

                # Fixed phase from lookup table — consistent timbre across keyboard
                phase = _PHASE_TABLE[s_idx][n - 1]

                signal += amp * env * np.sin(2 * np.pi * partial_freq * t + phase)

        # ═══ PHANTOM PARTIALS (Bank & Sujbert 2005, Conklin 1999) ═══
        # Longitudinal string vibrations from geometric nonlinearity produce
        # "phantom partials" at sum frequencies of transverse partial pairs,
        # adding metallic shimmer to the attack. Extended to C5 (midi 72)
        # since they're audible higher than commonly assumed (Bank 2010).
        if midi < 72 and max_partial >= 4:
            # Precompute parent partial frequencies and decay rates
            parent_freqs = {}
            parent_alphas = {}
            for n in range(1, min(12, max_partial + 1)):
                f_n = n * freq * np.sqrt(1 + B * n**2)
                K_n = (n ** 2) * piL2
                air_frac = 0.2
                a_n = (b1 * (1.0 - air_frac)
                       + b1 * air_frac * np.sqrt(freq / max(f_n, 20.0))
                       + b2 * K_n)
                parent_freqs[n] = f_n
                parent_alphas[n] = a_n

            for n in range(2, min(8, max_partial)):
                for m in range(1, n):
                    # Correct sum frequency from actual inharmonic partials
                    phantom_freq = parent_freqs[n] + parent_freqs[m]
                    if phantom_freq >= SAMPLE_RATE / 2:
                        continue
                    # Real phantoms are 30-40 dB below fundamental (Bank 2005)
                    # Fade out above C3 — analysis showed 5-10dB too much inter-harmonic
                    # energy in bass vs Salamander. Tighter fade and lower amplitude.
                    phantom_scale = np.clip((52 - midi) / 16.0, 0.0, 1.0)  # full at B1, zero at E3+
                    phantom_amp = 0.002 * phantom_scale / (n * m) ** 0.5
                    # Decay = sum of parent rates (Bank 2010)
                    phantom_decay = parent_alphas[n] + parent_alphas[m]
                    # Concentrated in first ~15ms of attack (was 30ms, too persistent)
                    phantom_env = np.exp(-t * phantom_decay) * np.exp(-t * 50.0)
                    ph = rng.uniform(0, 2 * np.pi)
                    signal += phantom_amp * phantom_env * np.sin(2 * np.pi * phantom_freq * t + ph)

            # Free longitudinal modes: steel string v_L ≈ 5100 m/s
            # These ring at multiples of v_L/(2L), independent of transverse modes.
            # Produce a brief metallic "ping" at note onset (sound precursor).
            v_L = 5100.0  # m/s, longitudinal wave speed in steel
            f_long_1 = v_L / (2.0 * L)  # fundamental longitudinal frequency
            for k in range(1, 4):  # first 3 longitudinal modes
                f_long = k * f_long_1
                if f_long >= SAMPLE_RATE / 2 or f_long < 20:
                    continue
                # Very weak, very fast decay — barely audible ping in first ~20ms
                long_amp = 0.002 / k
                long_decay = b1 * 3 + k * 2.0 + 30.0  # ~20ms time constant
                long_env = np.exp(-t * long_decay)
                ph = rng.uniform(0, 2 * np.pi)
                signal += long_amp * long_env * np.sin(2 * np.pi * f_long * t + ph)

        # ═══ HAMMER ATTACK TRANSIENT ═══
        # [5] VELOCITY-DEPENDENT HAMMER NOISE
        # The soundboard is briefly excited by the hammer impact,
        # producing a short broadband "thump" that gives piano its
        # percussive attack character. Without this, piano → organ.
        # Harder strikes: louder thump, broader bandwidth, shorter duration.
        noise = rng.randn(n_samples)
        hammer_env = np.exp(-t / max(T_c * 1.5, 0.001))
        hammer_env *= (t < 0.03).astype(float)

        # Band-limit: harder strikes allow more HF through (smaller kernel)
        # Analysis showed Salamander has 5-80x more energy above 4kHz in attack,
        # especially C5+. Wider bandwidth and added HF emphasis to match.
        # pp: freq*6 bandwidth, fff: freq*16 bandwidth (was 4/10)
        noise_bw = freq * (6.0 + 10.0 * effective_hardness)
        # Ensure minimum 8kHz bandwidth for treble notes
        noise_bw = max(noise_bw, 8000.0)
        lp_size = max(int(SAMPLE_RATE / max(noise_bw, 500)), 3)
        if lp_size % 2 == 0:
            lp_size += 1
        kernel = np.ones(lp_size) / lp_size
        hammer_noise = np.convolve(noise * hammer_env, kernel, mode='same')

        # Add HF emphasis: real hammer impacts have a sharp crack component
        # that the low-pass alone can't produce. Mix in unfiltered noise
        # for the first 3ms, scaled by key position (more prominent in treble).
        # Salamander shows 5-80x more HF energy in attack, especially C5+.
        hf_dur = int(0.003 * SAMPLE_RATE)
        hf_env = np.zeros(n_samples)
        hf_env[:hf_dur] = np.exp(-np.linspace(0, 6, hf_dur))
        hf_crack = noise * hf_env * 0.6 * (0.2 + 0.8 * key_pos)
        hammer_noise += hf_crack

        # Quadratic velocity scaling: ff hammer thump is much more prominent than pp.
        # vel^2: pp(0.15)→0.023, mp(0.6)→0.36, ff(0.9)→0.81, fff(1.0)→1.0
        # Increased base level from 0.06 to 0.10 to better match Salamander's
        # attack centroid (was 176 Hz too low on average).
        hammer_level = 0.10 * vel_clamp ** 2 * (0.5 + 0.5 * key_pos) * (1.0 - 0.3 * key_pos)
        # Bass hammer thump leads the tone too much with the shorter attack ramp,
        # standing out as a click. Taper level down below G4 (MIDI 67).
        hammer_level *= np.interp(midi, [36, 67], [0.45, 1.0])

        signal += hammer_noise * hammer_level

        # ═══ ATTACK SHAPE ═══
        # Tightened targets vs v1 (was [0.050, 0.030, 0.012]):
        # v1 avg attack: 43ms; ref avg: 27ms. New values → 30ms avg (+3ms off ref).
        attack_peak = np.interp(midi, [36, 60, 96], [0.030, 0.018, 0.008])
        attack_env = np.where(t < attack_peak,
                              0.5 - 0.5 * np.cos(np.pi * t / attack_peak),
                              1.0)
        signal *= attack_env

        # ═══ VELOCITY AMPLITUDE ═══
        # Real piano: sound intensity roughly proportional to velocity^2 (kinetic energy).
        # Use a compressed power law so pp is audible but fff is significantly louder.
        vel_amp = velocity ** 1.5  # pp(0.15)→0.058, mp(0.60)→0.465, ff(0.90)→0.854, fff(1.0)→1.0
        signal *= vel_amp

        # ═══ SOUNDBOARD IR CONVOLUTION ═══
        # Per-note filter interpolated from reference recordings' transfer functions.
        # Models how soundboard resonance varies across the bridge — each note gets
        # its own spectral coloring (detrended: no room/mic tilt, just resonance detail).
        sb_ir = _get_soundboard_ir(midi) if use_ir else None
        if sb_ir is not None:
            wet = np.convolve(signal, sb_ir, mode='full')[:n_samples]
            dry_rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
            wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-10
            wet *= dry_rms / wet_rms
            sb_mix = 0.85
            signal = (1.0 - sb_mix) * signal + sb_mix * wet

        # ═══ FADE OUT ═══
        fade_samples = int(0.1 * SAMPLE_RATE)
        signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # ═══ NORMALIZE ═══
        # When generating velocity layers, normalization is done externally
        # (relative to loudest layer) to preserve dynamics. For single-velocity
        # generation, normalize to 0.85 peak.
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.85

        return signal, peak


    def write_wav(filename, signal):
        """Write a numpy array as a 16-bit WAV file."""
        pcm = (signal * 32767).astype(np.int16)
        import wave
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())


    def wav_to_mp3(wav_path, mp3_path):
        """Convert WAV to MP3 using ffmpeg."""
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-codec:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '44100', mp3_path
        ], capture_output=True)


    VELOCITY_LAYERS = [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 1.00]
    # Map MIDI velocity (1-127) to layer index:
    # Layer boundaries at midpoints: 0-24, 25-40, 41-56, 57-72, 73-88, 89-104, 105-116, 117-127

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_rhodes():
    """
    Generate Rhodes-style electric piano samples using DX7-style FM synthesis.

    Based on:
    - DX7 E.PIANO 1 patch (Algorithm 5): carrier-modulator FM pairs
      BODY: 1:1 ratio, moderate mod index, slow decay (warm fundamental)
      TINE: 9:1 ratio, fast decay (metallic bell attack)
    - Rhodes Mark I physics:
      * Tine vibrates as near-perfect sine wave
      * Tonebar couples as resonator (slight detuning → shimmer)
      * Magnetic pickup is nonlinear → asymmetric distortion → even harmonics
    - Deep spectral comparison against real 1977 Mark I samples:
      * Attack: near-instant, spectral flatness ~0 (pure tone, NO noise)
      * Centroid: only 1.5-3x fundamental even during attack
      * Harmonics persist through sustain (slow mod envelope decay)

    Physical model per note:
      1. BODY FM: 1:1 carrier:modulator with decaying mod index (warm tone)
      2. TINE FM: 9:1 ratio metallic attack with fast decay (bell character)
      3. SUB-HARMONIC: subtle 0.5:1 undertone (key-dependent)
      4. ADDITIVE HARMONICS: h2, h3 for spectral fill
      5. MAGNETIC PICKUP: asymmetric tanh distortion (even harmonics, warmth)
      6. SMOOTH ATTACK: half-cosine rise (no derivative discontinuity)

    Output: MP3 files ready for the learn-piano.html sampler.
    """

    import numpy as np
    import os
    import subprocess

    SAMPLE_RATE = 44100
    DURATION = 5.0  # seconds per sample

    # Per-note transfer functions extracted from real Rhodes recordings
    # via spectral division (extract_rhodes_tf.py). Captures pickup/amp/cabinet
    # character that varies across the keyboard.
    _TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rhodes_tf.npz')
    _TF_DATA = None
    _IR_CACHE = {}

    if os.path.exists(_TF_PATH):
        _TF_DATA = np.load(_TF_PATH)
        _TF_MIDI = _TF_DATA['midi_points']
        _TF_MAG = _TF_DATA['transfer_functions']
        _TF_IR_LEN = int(_TF_DATA['ir_length'])


    def _minimum_phase_ir(magnitude_response, ir_length):
        """Create minimum-phase FIR from magnitude response (cepstral method)."""
        mag = np.maximum(magnitude_response, 1e-10)
        n_fft = (len(mag) - 1) * 2
        full_mag = np.concatenate([mag, mag[-2:0:-1]])
        log_mag = np.log(full_mag)
        cepstrum = np.fft.ifft(log_mag).real
        min_cep = np.zeros_like(cepstrum)
        min_cep[0] = cepstrum[0]
        min_cep[1:n_fft // 2] = 2 * cepstrum[1:n_fft // 2]
        min_cep[n_fft // 2] = cepstrum[n_fft // 2]
        ir = np.fft.ifft(np.exp(np.fft.fft(min_cep))).real
        ir = ir[:ir_length]
        ir *= np.hanning(ir_length * 2)[ir_length:]
        return ir


    def _get_rhodes_ir(midi):
        """Get per-note Rhodes IR by interpolating reference transfer functions."""
        if _TF_DATA is None:
            return None
        if midi in _IR_CACHE:
            return _IR_CACHE[midi]

        log_tfs = np.log(_TF_MAG + 1e-10)
        interp_log = np.zeros(log_tfs.shape[1])
        for i_bin in range(log_tfs.shape[1]):
            interp_log[i_bin] = np.interp(midi, _TF_MIDI, log_tfs[:, i_bin])
        mag_response = np.exp(interp_log)

        ir = _minimum_phase_ir(mag_response, _TF_IR_LEN)
        ir_energy = np.sqrt(np.sum(ir ** 2))
        if ir_energy > 0:
            n_fft = (len(_TF_MAG[0]) - 1) * 2
            ir = ir / ir_energy * np.sqrt(_TF_IR_LEN) / np.sqrt(n_fft)

        _IR_CACHE[midi] = ir
        return ir


    NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']


    def midi_to_name(midi):
        return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


    # Every chromatic note from B1 (35) to D6 (86)
    NOTES = [(m, midi_to_name(m)) for m in range(35, 87)]

    # Per-component phase table from fixed seed (consistent timbre, varied onsets).
    # Components: 0=body, 1=tine, 2=sub, 3=h2, 4=h3
    _PHASE_TABLE = np.random.RandomState(7823).uniform(0, 2 * np.pi, (128, 5))


    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)


    VELOCITY_LAYERS = [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 1.00]


    def generate_rhodes_note(midi, duration=DURATION, velocity=0.75):
        """Generate a Rhodes note using DX7-style 3-component FM synthesis.

        Velocity model (Rhodes-specific):
          [1] FM mod index scales with velocity — brighter body at ff
          [2] Tine prominence increases with velocity — more bell/bark at ff
          [3] Pickup distortion drive scales with velocity — warm at pp, barky at ff
          [4] Tine decay slows at higher velocity — bell sustains longer at ff
          [5] Velocity amplitude — energy scaling
        """
        freq = midi_to_freq(midi)
        n_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        # Key-dependent scaling (1.0 at low register, 0.25 at high)
        key_scale = np.clip(1.0 - (midi - 40) / 60, 0.25, 1.0)

        # Per-note initial phases (varied waveform onsets across keyboard)
        ph = _PHASE_TABLE[midi]

        # Velocity-dependent effective intensity (same curve as grand piano)
        vel_clamp = max(velocity, 0.05)
        effective_vel = 0.15 + 0.85 * vel_clamp ** 1.5

        # --- Amplitude envelope (near-instant attack, smooth decay) ---
        # Half-cosine attack (C¹-smooth, no derivative discontinuity at peak)
        attack_time = 0.002  # 2ms — Rhodes attack is near-instant
        attack = np.where(t < attack_time,
                          0.5 - 0.5 * np.cos(np.pi * t / attack_time),
                          1.0)
        # Calibrated from sampled Rhodes D3: 50% at 1.18s, 10% at 3.56s
        decay_rate = np.interp(midi, [35, 60, 86], [0.415, 0.54, 0.67])
        env = attack * np.exp(-t * decay_rate)

        # ═══ COMPONENT 1: BODY (1:1 ratio) ═══
        # [1] FM mod index scales with velocity — more harmonics at ff.
        # pp: mod_idx ≈ 0.6 (nearly pure sine), fff: mod_idx ≈ 2.0 (rich harmonics)
        body_mod_idx = 1.979 * effective_vel * key_scale
        mod_decay_rate = 0.305 + (1.0 - key_scale) * 1.579
        # Mod envelope decays slower at high velocity — brightness persists longer
        mod_decay_rate *= (1.3 - 0.4 * effective_vel)
        body_mod_env = body_mod_idx * np.exp(-t * mod_decay_rate)
        body_mod = body_mod_env * np.sin(2 * np.pi * freq * t + ph[0])
        body = np.sin(2 * np.pi * freq * t + body_mod + ph[0])

        # ═══ TINE (9:1 ratio — metallic bell attack) ═══
        # [2] Tine prominence scales with velocity — the "bell" that defines Rhodes
        # character becomes much more prominent at ff (the "bark").
        # [4] Tine carrier decays slower at high velocity — bell rings longer.
        tine_ratio = 9.0
        tine_mod_freq = freq * tine_ratio
        if tine_mod_freq < SAMPLE_RATE / 2 - 1000:
            tine_mod_idx = (0.5 + 0.3 * key_scale) * effective_vel
            tine_mod_decay = (14.0 - 8.0 * key_scale) * (1.2 - 0.3 * effective_vel)
            tine_mod_env = tine_mod_idx * np.exp(-t * tine_mod_decay)
            tine_mod = tine_mod_env * np.sin(2 * np.pi * tine_mod_freq * t + ph[1])
            tine_carrier_decay = (5.0 - 3.0 * key_scale) * (1.3 - 0.4 * effective_vel)
            tine_carrier_env = np.exp(-t * tine_carrier_decay)
            tine = tine_carrier_env * np.sin(2 * np.pi * freq * t + tine_mod + ph[1])
        else:
            tine = np.zeros(n_samples)

        # ═══ SUB-HARMONIC ═══
        sub_freq = freq * 0.5
        if sub_freq >= 20:
            sub = 0.023 * key_scale * np.sin(2 * np.pi * sub_freq * t + ph[2])
        else:
            sub = np.zeros(n_samples)

        # ═══ ADDITIVE HARMONICS ═══
        h2 = np.zeros(n_samples)
        h3 = np.zeros(n_samples)
        if freq * 2 < SAMPLE_RATE / 2:
            h2 = 0.008 * np.sin(2 * np.pi * freq * 2 * t + ph[3])
        if freq * 3 < SAMPLE_RATE / 2:
            h3 = 0.069 * np.sin(2 * np.pi * freq * 3 * t + ph[4])

        # ═══ MIX ═══
        # [2] Tine mix increases with velocity — pp is warm body, ff has prominent bell
        tine_mix = (0.20 + 0.15 * key_scale) * (0.5 + 0.8 * effective_vel)
        # pp: tine_mix ~0.13, fff: tine_mix ~0.38
        signal = body * 0.80 + tine * tine_mix + sub + h2 + h3

        # Apply amplitude envelope
        signal *= env

        # ═══ MAGNETIC PICKUP SIMULATION ═══
        # [3] Pickup distortion drive scales with velocity.
        # pp: gentle, clean sound. ff: pickup saturates → "bark" (even harmonics).
        # Real Rhodes: the tine moves closer to the pickup at higher velocity,
        # increasing the magnetic flux change → more distortion.
        drive = 0.10 + 0.40 * effective_vel  # pp: 0.12, fff: 0.50
        asym = 0.510
        pos = np.maximum(signal, 0)
        neg = np.minimum(signal, 0)
        signal = (np.tanh(pos * drive) / np.tanh(drive) +
                  np.tanh(neg * drive * asym) / np.tanh(drive * asym))

        # ═══ PICKUP/AMP IR CONVOLUTION ═══
        rhodes_ir = _get_rhodes_ir(midi)
        if rhodes_ir is not None:
            wet = np.convolve(signal, rhodes_ir, mode='full')[:n_samples]
            dry_rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
            wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-10
            wet *= dry_rms / wet_rms
            signal = 0.3 * signal + 0.7 * wet

        # [5] Velocity amplitude — Rhodes has less dynamic range than grand piano
        vel_amp = vel_clamp ** 1.2  # pp(0.15)→0.11, mp(0.6)→0.54, fff(1.0)→1.0
        signal *= vel_amp

        # --- Fade out ---
        fade_samples = int(0.1 * SAMPLE_RATE)
        signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Normalize — returns (signal, raw_peak) for relative normalization
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.85

        return signal, peak


    def write_wav(filename, signal):
        """Write a numpy array as a 16-bit WAV file."""
        pcm = (signal * 32767).astype(np.int16)
        import wave
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())


    def wav_to_mp3(wav_path, mp3_path):
        """Convert WAV to MP3 using ffmpeg."""
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-codec:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '44100', mp3_path
        ], capture_output=True)

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_prism():
    """
    Generate "Prism Keys" — a retro-futuristic electronic piano.

    Layers:
      1. UNISON BASE: 3 slightly detuned copies of fundamental (chorus shimmer)
      2. FIFTH: Open fifth (3:2 ratio) for hollow, airy quality
      3. BELL: FM bell at 7:1 ratio — sparkly, crystalline attack
      4. SUB: Gentle sub-octave for warmth
      5. TREMOLO: Slow amplitude modulation for organic movement
      6. ATTACK CLICK: Tiny broadband FM burst for percussive definition

    Output: MP3 files for the learn-piano.html sampler.
    """

    import numpy as np
    import os
    import subprocess

    SAMPLE_RATE = 44100
    DURATION = 5.0

    # Borrow piano soundboard transfer functions for acoustic body/resonance
    _TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
    _TF_DATA = None
    _IR_CACHE = {}

    if os.path.exists(_TF_PATH):
        _TF_DATA = np.load(_TF_PATH)
        _TF_MIDI = _TF_DATA['midi_points']
        _TF_MAG = _TF_DATA['transfer_functions']
        _TF_IR_LEN = int(_TF_DATA['ir_length'])


    def _minimum_phase_ir(magnitude_response, ir_length):
        mag = np.maximum(magnitude_response, 1e-10)
        n_fft = (len(mag) - 1) * 2
        full_mag = np.concatenate([mag, mag[-2:0:-1]])
        log_mag = np.log(full_mag)
        cepstrum = np.fft.ifft(log_mag).real
        min_cep = np.zeros_like(cepstrum)
        min_cep[0] = cepstrum[0]
        min_cep[1:n_fft // 2] = 2 * cepstrum[1:n_fft // 2]
        min_cep[n_fft // 2] = cepstrum[n_fft // 2]
        ir = np.fft.ifft(np.exp(np.fft.fft(min_cep))).real
        ir = ir[:ir_length]
        ir *= np.hanning(ir_length * 2)[ir_length:]
        return ir


    def _get_body_ir(midi):
        """Get per-note IR from piano soundboard TFs — adds acoustic resonance."""
        if _TF_DATA is None:
            return None
        if midi in _IR_CACHE:
            return _IR_CACHE[midi]

        log_tfs = np.log(_TF_MAG + 1e-10)
        interp_log = np.zeros(log_tfs.shape[1])
        for i_bin in range(log_tfs.shape[1]):
            interp_log[i_bin] = np.interp(midi, _TF_MIDI, log_tfs[:, i_bin])
        mag_response = np.exp(interp_log)

        # Use shorter IR for prism — just coloring, not full piano resonance
        ir_len = min(_TF_IR_LEN, 1024)
        ir = _minimum_phase_ir(mag_response, ir_len)
        ir_energy = np.sqrt(np.sum(ir ** 2))
        if ir_energy > 0:
            n_fft = (len(_TF_MAG[0]) - 1) * 2
            ir = ir / ir_energy * np.sqrt(ir_len) / np.sqrt(n_fft)

        _IR_CACHE[midi] = ir
        return ir

    NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']

    NOTES = [(m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}") for m in range(35, 87)]

    VELOCITY_LAYERS = [0.15, 0.30, 0.45, 0.60, 0.70, 0.80, 0.90, 1.00]


    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)


    def generate_prism_note(midi, duration=DURATION, velocity=0.75):
        """Generate a Prism Keys note with velocity-sensitive synthesis.

        Velocity model (Prism-specific):
          [1] FM bell mod index scales with velocity — crystalline sparkle at ff
          [2] Unison detune widens with velocity — thicker chorus at ff
          [3] Attack click intensity scales with velocity — percussive snap at ff
          [4] Fifth prominence grows with velocity — more open voicing at ff
          [5] Saturation drive increases with velocity — warmer edge at ff
          [6] Velocity amplitude — energy scaling
        """
        freq = midi_to_freq(midi)
        n_samples = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)

        key_scale = np.clip(1.0 - (midi - 40) / 60, 0.25, 1.0)

        vel_clamp = max(velocity, 0.05)
        effective_vel = 0.15 + 0.85 * vel_clamp ** 1.5

        # --- Envelope: soft attack, smooth decay ---
        attack_time = 0.008  # 8ms — slightly softer than Rhodes
        attack = np.minimum(t / attack_time, 1.0)
        decay_rate = 0.4 + (midi - 35) * 0.006
        env = attack * np.exp(-t * decay_rate)

        # ═══ LAYER 1: UNISON (3 detuned voices) ═══
        # [2] Detune widens with velocity: pp ±2 cents (tight), ff ±5 cents (wide chorus)
        detune_cents = 2.0 + 3.0 * effective_vel
        detune_ratio = 2 ** (detune_cents / 1200)
        phase_c = 2 * np.pi * freq * t
        phase_up = 2 * np.pi * freq * detune_ratio * t
        phase_dn = 2 * np.pi * freq / detune_ratio * t
        unison = (np.sin(phase_c) * 0.5 +
                  np.sin(phase_up) * 0.25 +
                  np.sin(phase_dn) * 0.25)

        # ═══ LAYER 2: OPEN FIFTH ═══
        # [4] Fifth prominence grows with velocity — spacious shimmer at ff
        fifth_freq = freq * 1.5
        fifth_env = np.exp(-t * (decay_rate + 0.5))
        fifth_amp = 0.10 + 0.14 * effective_vel  # pp: 0.12, fff: 0.24
        if fifth_freq < SAMPLE_RATE / 2 - 500:
            fifth = fifth_amp * fifth_env * np.sin(2 * np.pi * fifth_freq * t)
        else:
            fifth = np.zeros(n_samples)

        # ═══ LAYER 3: FM BELL (7:1 ratio) ═══
        # [1] Bell mod index scales with velocity — crystalline sparkle at ff
        bell_ratio = 7.0
        bell_mod_freq = freq * bell_ratio
        if bell_mod_freq < SAMPLE_RATE / 2 - 1000:
            bell_mod_idx = (0.8 + 0.4 * key_scale) * (0.3 + 0.9 * effective_vel)
            bell_mod_decay = (8.0 - 4.0 * key_scale) * (1.3 - 0.4 * effective_vel)
            bell_mod_env = bell_mod_idx * np.exp(-t * bell_mod_decay)
            bell_mod = bell_mod_env * np.sin(2 * np.pi * bell_mod_freq * t)
            bell_carrier_decay = (4.0 - 2.0 * key_scale) * (1.2 - 0.3 * effective_vel)
            bell = np.exp(-t * bell_carrier_decay) * np.sin(phase_c + bell_mod)
        else:
            bell = np.zeros(n_samples)

        # ═══ LAYER 4: SUB OCTAVE ═══
        sub = 0.10 * key_scale * np.sin(np.pi * freq * t)

        # ═══ LAYER 5: ATTACK CLICK ═══
        # [3] Click intensity scales with velocity — percussive snap at ff
        click_mod_idx = 1.0 + 3.5 * effective_vel  # pp: 1.15, fff: 4.5
        click_env = np.exp(-t * 80.0)  # gone in ~25ms
        click_mod = click_mod_idx * click_env * np.sin(2 * np.pi * freq * 5.5 * t)
        click_amp = 0.05 + 0.15 * effective_vel  # pp: 0.07, fff: 0.20
        click = click_amp * click_env * np.sin(phase_c + click_mod)

        # ═══ MIX ═══
        bell_mix = (0.30 + 0.15 * key_scale) * (0.5 + 0.7 * effective_vel)
        signal = unison * 0.55 + bell * bell_mix + fifth + sub + click

        # Apply envelope
        signal *= env

        # ═══ TREMOLO ═══
        # Slow amplitude modulation — organic, breathing quality
        trem_rate = 4.5  # Hz
        trem_depth = 0.08  # subtle
        tremolo = 1.0 - trem_depth * (0.5 + 0.5 * np.sin(2 * np.pi * trem_rate * t))
        signal *= tremolo

        # ═══ SOFT SATURATION ═══
        # [5] Saturation drive increases with velocity — pp: clean, ff: warm edge
        drive = 0.8 + 0.8 * effective_vel  # pp: 0.92, fff: 1.6
        signal = np.tanh(signal * drive) / np.tanh(drive)

        # ═══ ACOUSTIC BODY IR ═══
        # Borrow piano soundboard resonance for richness — subtle mix
        body_ir = _get_body_ir(midi)
        if body_ir is not None:
            wet = np.convolve(signal, body_ir, mode='full')[:n_samples]
            dry_rms = np.sqrt(np.mean(signal ** 2)) + 1e-10
            wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-10
            wet *= dry_rms / wet_rms
            signal = 0.6 * signal + 0.4 * wet

        # [6] Velocity amplitude
        vel_amp = vel_clamp ** 1.0  # linear — electronic instrument, wide dynamics
        signal *= vel_amp

        # --- Fade out ---
        fade_samples = int(0.05 * SAMPLE_RATE)
        signal[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        # Normalize — returns (signal, raw_peak) for relative normalization
        peak = np.max(np.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.85

        return signal, peak


    def write_wav(filename, signal):
        pcm = (signal * 32767).astype(np.int16)
        import wave
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())


    def wav_to_mp3(wav_path, mp3_path):
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-codec:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '44100', mp3_path
        ], capture_output=True)

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_ddsp():
    """
    Differentiable Digital Signal Processing (DDSP) Piano Synthesizer.

    Physics-informed differentiable synthesis following Simionato et al. (2023)
    and DDSP-Piano (Renault et al. 2022):
      - Harmonic component: inharmonic additive synthesis with learned spectral envelope
      - Noise component: learned filtered noise for hammer/room/string noise
      - Neural network predicts per-note parameters for both components
      - Trained end-to-end with Adam + multi-scale STFT loss

    Usage:
        python3 ddsp_piano.py                  # Train + generate samples
        python3 ddsp_piano.py --epochs 2000    # More training iterations
        python3 ddsp_piano.py --generate-only  # Generate from saved model
    """

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import os
    import sys
    import time
    import subprocess
    import wave

    SAMPLE_RATE = 44100
    DURATION = 6.0
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ddsp_piano_model.pt')

    # Pre-allocate
    t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)
    PHASE_TABLE_GPU = torch.tensor(
        np.random.RandomState(6454).uniform(0, 2 * np.pi, (3, 64)),
        dtype=torch.float32, device=DEVICE,
    )
    # Deterministic noise seed for reproducible noise component
    NOISE_TABLE = torch.tensor(
        np.random.RandomState(7777).randn(N_SAMPLES),
        dtype=torch.float32, device=DEVICE,
    )

    # Training/generation note lists
    TRAIN_NOTES = [
        (36, 'C2'), (39, 'Ds2'), (42, 'Fs2'), (45, 'A2'),
        (48, 'C3'), (51, 'Ds3'), (54, 'Fs3'), (57, 'A3'),
        (60, 'C4'), (63, 'Ds4'), (66, 'Fs4'), (69, 'A4'),
        (72, 'C5'), (75, 'Ds5'), (78, 'Fs5'), (81, 'A5'),
        (84, 'C6'),
    ]

    NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
    ALL_NOTES = [(m, f"{NOTE_NAMES[m % 12]}{m // 12 - 1}") for m in range(35, 87)]

    # ═══ PHYSICS CONSTANTS ═══
    B_MIDI = torch.tensor([21, 33, 45, 57, 69, 84, 96], dtype=torch.float32, device=DEVICE)
    B_VALS_LOG = torch.log(torch.tensor(
        [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2],
        dtype=torch.float32, device=DEVICE))

    L_MIDI = torch.tensor([36, 60, 96], dtype=torch.float32, device=DEVICE)
    L_VALS_LOG = torch.log(torch.tensor([1.92, 0.62, 0.09], dtype=torch.float32, device=DEVICE))

    SB_MODE_CFS = torch.tensor([90.0, 170.0, 260.0], device=DEVICE)
    SB_MODE_BWS = torch.tensor([30.0, 35.0, 45.0], device=DEVICE)

    # ═══ SOUNDBOARD IR CONVOLUTION ═══
    # Per-note transfer functions extracted from reference recordings (extract_soundboard_ir.py).
    # Detrended to remove room/mic coloring — keeps only resonance detail across the bridge.
    _TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
    _SB_IR_CACHE = {}  # midi → torch tensor IR

    if os.path.exists(_TF_PATH):
        _tf_data = np.load(_TF_PATH)
        _TF_MIDI = _tf_data['midi_points']
        _TF_MAG = _tf_data['transfer_functions']  # [n_notes, n_bins]
        _TF_IR_LEN = int(_tf_data['ir_length'])

        def _min_phase_ir_np(mag_resp, ir_len):
            mag = np.maximum(mag_resp, 1e-10)
            n_fft = (len(mag) - 1) * 2
            full = np.concatenate([mag, mag[-2:0:-1]])
            cep = np.fft.ifft(np.log(full)).real
            mc = np.zeros_like(cep)
            mc[0] = cep[0]
            mc[1:n_fft//2] = 2 * cep[1:n_fft//2]
            mc[n_fft//2] = cep[n_fft//2]
            ir = np.fft.ifft(np.exp(np.fft.fft(mc))).real[:ir_len]
            ir *= np.hanning(ir_len * 2)[ir_len:]
            e = np.sqrt(np.sum(ir**2))
            if e > 0:
                ir = ir / e * np.sqrt(ir_len) / np.sqrt(n_fft)
            return ir

        def _get_sb_ir_torch(midi):
            if midi not in _SB_IR_CACHE:
                log_tfs = np.log(_TF_MAG + 1e-10)
                interp_log = np.array([np.interp(midi, _TF_MIDI, log_tfs[:, i])
                                       for i in range(log_tfs.shape[1])])
                mag = np.exp(interp_log)
                ir = _min_phase_ir_np(mag, _TF_IR_LEN)
                _SB_IR_CACHE[midi] = torch.tensor(ir.astype(np.float32), device=DEVICE)
            return _SB_IR_CACHE[midi]
    else:
        _get_sb_ir_torch = lambda midi: None


    def apply_soundboard_ir(signal, midi):
        """Apply per-note soundboard IR via differentiable convolution.

        Uses shorter effective IR (1024 samples / ~23ms) than the physics model
        to avoid smearing the attack, and lower wet mix since DDSP already has
        learned spectral corrections.
        """
        sb_ir = _get_sb_ir_torch(midi)
        if sb_ir is None:
            return signal
        # Truncate IR to reduce attack smearing (full IR = 2048 = ~46ms)
        max_ir_len = 1024
        short_ir = sb_ir[:max_ir_len] * torch.hann_window(max_ir_len * 2, device=DEVICE)[max_ir_len:]
        n_samples = len(signal)
        ir = short_ir.unsqueeze(0).unsqueeze(0)
        sig = signal.unsqueeze(0).unsqueeze(0)
        pad_len = len(short_ir) - 1
        wet = F.conv1d(F.pad(sig, (pad_len, 0)), ir).squeeze()[:n_samples]
        dry_rms = (signal ** 2).mean().sqrt() + 1e-10
        wet_rms = (wet ** 2).mean().sqrt() + 1e-10
        wet = wet * dry_rms / wet_rms
        return 0.55 * signal + 0.45 * wet


    # Spectral control: 8 points in log-partial space
    N_SPECTRAL_CTRL = 8
    SPECTRAL_CTRL_PARTIALS = torch.tensor([1, 2, 4, 8, 16, 24, 32, 48],
                                           dtype=torch.float32, device=DEVICE)

    # Noise filter: 16 frequency bands (bark-ish spacing)
    N_NOISE_BANDS = 16


    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)


    def log_interp(midi, calib_midi, log_vals):
        midi_f = float(midi)
        if midi_f <= calib_midi[0].item():
            return torch.exp(log_vals[0])
        if midi_f >= calib_midi[-1].item():
            return torch.exp(log_vals[-1])
        for i in range(len(calib_midi) - 1):
            if midi_f <= calib_midi[i + 1].item():
                frac = (midi_f - calib_midi[i].item()) / (calib_midi[i + 1].item() - calib_midi[i].item())
                return torch.exp(log_vals[i] + frac * (log_vals[i + 1] - log_vals[i]))
        return torch.exp(log_vals[-1])


    def interp_spectral_curve(ctrl_values, n_partials):
        """Interpolate control points to n_partials using log-partial spacing."""
        ctrl_x = torch.log(SPECTRAL_CTRL_PARTIALS[:len(ctrl_values)])
        target_x = torch.log(torch.arange(1, n_partials + 1, dtype=torch.float32, device=DEVICE))
        target_x_clamped = torch.clamp(target_x, ctrl_x[0], ctrl_x[-1])

        result = torch.zeros(n_partials, device=DEVICE)
        for i in range(len(ctrl_values) - 1):
            mask = (target_x_clamped >= ctrl_x[i]) & (target_x_clamped <= ctrl_x[i + 1])
            if mask.any():
                frac = (target_x_clamped[mask] - ctrl_x[i]) / (ctrl_x[i + 1] - ctrl_x[i] + 1e-8)
                result[mask] = ctrl_values[i] + frac * (ctrl_values[i + 1] - ctrl_values[i])
        beyond = target_x > ctrl_x[-1]
        if beyond.any():
            result[beyond] = ctrl_values[-1]
        return result


    # ═══ NEURAL NETWORK ═══

    class PianoParamNet(nn.Module):
        """Predicts per-note synthesis parameters from MIDI number.

        Harmonic params: damping, soundboard, two-stage decay, spectral envelope, decay curve
        Noise params: level, decay rate, spectral tilt, bandwidth
        """

        # 2 damping + 6 soundboard + 3 two-stage + 1 air + 2 hammer
        # + 8 spectral + 8 decay + 4 noise = 34
        N_PARAMS = 34

        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, self.N_PARAMS),
            )
            with torch.no_grad():
                self.net[-1].bias.zero_()
                self.net[-1].weight.normal_(std=0.01)

        def forward(self, midi_normalized):
            raw = self.net(midi_normalized)
            params = {}
            i = 0

            # All params are SMALL offsets from hand-tuned defaults.
            # Network output starts near zero → synthesis starts identical to grand piano.

            # Damping: ±0.3 (0.74x - 1.35x of hand-tuned value)
            params['log_b1_offset'] = torch.tanh(raw[i]) * 0.3; i += 1
            params['log_b2_offset'] = torch.tanh(raw[i]) * 0.3; i += 1

            # Soundboard: small offsets from hand-tuned (cf=2500, bw=1200, gain=0.3)
            params['sb_bridge_cf'] = 2500.0 + torch.tanh(raw[i]) * 500.0; i += 1   # 2000-3000
            params['sb_bridge_bw'] = 1200.0 + torch.tanh(raw[i]) * 400.0; i += 1   # 800-1600
            params['sb_bridge_gain'] = 0.3 + torch.tanh(raw[i]) * 0.15; i += 1     # 0.15-0.45
            # Low modes: hand-tuned (0.15, 0.12, 0.10) ± small offset
            params['sb_mode1_gain'] = 0.15 + torch.tanh(raw[i]) * 0.10; i += 1     # 0.05-0.25
            params['sb_mode2_gain'] = 0.12 + torch.tanh(raw[i]) * 0.08; i += 1     # 0.04-0.20
            params['sb_mode3_gain'] = 0.10 + torch.tanh(raw[i]) * 0.06; i += 1     # 0.04-0.16

            # Two-stage decay: hand-tuned defaults with key_pos factored into synth
            # prompt=1.5+0.5*kp, after=0.25, A_after=0.15+0.05*kp → allow small offsets
            params['prompt_factor_offset'] = torch.tanh(raw[i]) * 0.5; i += 1       # ±0.5 from default
            params['after_factor'] = 0.25 + torch.tanh(raw[i]) * 0.15; i += 1       # 0.10-0.40
            params['A_after_offset'] = torch.tanh(raw[i]) * 0.08; i += 1            # ±0.08

            # Air fraction: hand-tuned = 0.2
            params['air_frac'] = 0.2 + torch.tanh(raw[i]) * 0.15; i += 1           # 0.05-0.35

            # Hammer/attack: scale factors centered on 1.0
            params['hammer_cutoff_scale'] = 1.0 + torch.tanh(raw[i]) * 0.4; i += 1  # 0.6-1.4
            params['attack_peak_scale'] = 1.0 + torch.tanh(raw[i]) * 0.3; i += 1    # 0.7-1.3

            # Spectral envelope: ±1.5 range (±13 dB) — subtle corrections only
            params['spectral_ctrl'] = torch.tanh(raw[i:i+N_SPECTRAL_CTRL]) * 1.5; i += N_SPECTRAL_CTRL

            # Decay correction: ±0.5 (0.6x-1.6x of physics rate)
            params['decay_ctrl'] = torch.tanh(raw[i:i+N_SPECTRAL_CTRL]) * 0.5; i += N_SPECTRAL_CTRL

            # Noise: subtle attack transient only
            params['noise_level'] = torch.sigmoid(raw[i]) * 0.04; i += 1             # 0-0.04
            params['noise_decay'] = 15.0 + torch.sigmoid(raw[i]) * 60.0; i += 1      # fast: 15-75
            params['noise_tilt'] = torch.tanh(raw[i]) * 2.0; i += 1
            params['noise_bandwidth'] = 1.0 + torch.sigmoid(raw[i]) * 3.0; i += 1

            return params


    # ═══ DIFFERENTIABLE SYNTHESIS ═══

    def synthesize_harmonic(midi, params, t):
        """Harmonic (additive) component."""
        freq = midi_to_freq(midi)
        key_pos = max(0.0, min(1.0, (midi - 21) / 87.0))

        B = log_interp(midi, B_MIDI, B_VALS_LOG)
        L = log_interp(midi, L_MIDI, L_VALS_LOG)
        piL2 = (np.pi / L) ** 2

        b1_phys = log_interp(midi, torch.tensor([36.0, 60.0, 96.0], device=DEVICE),
                             torch.log(torch.tensor([0.25, 1.1, 9.17], device=DEVICE)))
        b2_phys = log_interp(midi, torch.tensor([36.0, 60.0, 96.0], device=DEVICE),
                             torch.log(torch.tensor([7.5e-5, 2.7e-4, 2.1e-3], device=DEVICE)))
        b1 = b1_phys * torch.exp(params['log_b1_offset'])
        b2 = b2_phys * torch.exp(params['log_b2_offset'])

        max_partial = 1
        while max_partial * freq * np.sqrt(1 + B.item() * max_partial**2) < SAMPLE_RATE / 2 - 500:
            max_partial += 1
        max_partial = min(max_partial - 1, 64)
        if max_partial < 1:
            return torch.zeros(len(t), device=DEVICE)

        p_exp = 2.3 + 0.7 * key_pos
        T_c_base = np.interp(midi, [36, 60, 96], [0.004, 0.0025, 0.0008])
        T_c = T_c_base * (0.75 / 0.8) ** (1.0 / (p_exp + 1))
        hammer_cutoff = 2.5 / T_c * params['hammer_cutoff_scale']
        strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

        spectral_correction = interp_spectral_curve(params['spectral_ctrl'], max_partial)
        decay_correction = interp_spectral_curve(params['decay_ctrl'], max_partial)

        sb_mode_gains = torch.stack([params['sb_mode1_gain'], params['sb_mode2_gain'],
                                      params['sb_mode3_gain']])

        if midi < 36:
            string_detunes = [0.0]
        elif midi < 48:
            dc = 0.3 + 0.3 * key_pos
            string_detunes = [-dc, dc]
        else:
            dc = 0.5 + 1.0 * key_pos
            string_detunes = [-dc, 0.0, dc]
        n_strings = len(string_detunes)

        ns = torch.arange(1, max_partial + 1, dtype=torch.float32, device=DEVICE)
        signal = torch.zeros(len(t), device=DEVICE)

        for s_idx, d_cents in enumerate(string_detunes):
            detune_ratio = 2 ** (d_cents / 1200)
            string_amp = 1.0 / n_strings

            partial_freqs = ns * freq * detune_ratio * torch.sqrt(1 + B * ns ** 2)
            valid = partial_freqs < SAMPLE_RATE / 2

            # Base rolloff from hand-tuned physics
            rolloff_base = 0.7 + 0.6 * key_pos + 4.5 * key_pos ** 3
            amps = 1.0 / (ns ** rolloff_base)

            # Learned spectral correction
            amps = amps * torch.exp(spectral_correction)

            # Hammer (physics)
            amps = amps / (1.0 + (partial_freqs / hammer_cutoff) ** 1.5)
            fTc = partial_freqs * T_c
            denom = 1.0 - 4.0 * fTc * fTc
            safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
            cosine_mod = torch.clamp(torch.abs(torch.cos(np.pi * fTc) / safe_denom), max=1.0)
            cosine_mod = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(cosine_mod), cosine_mod)
            amps = amps * (0.7 + 0.3 * cosine_mod)

            # Strike position (physics)
            strike_factor = torch.abs(torch.sin(np.pi * ns * strike_pos))
            amps = amps * torch.clamp(strike_factor, min=0.03)

            # Soundboard (learned)
            sb_response = torch.ones_like(partial_freqs)
            for idx in range(3):
                sb_response = sb_response + sb_mode_gains[idx] * torch.exp(
                    -0.5 * ((partial_freqs - SB_MODE_CFS[idx]) / SB_MODE_BWS[idx]) ** 2)
            sb_response = sb_response + params['sb_bridge_gain'] * torch.exp(
                -0.5 * ((partial_freqs - params['sb_bridge_cf']) / params['sb_bridge_bw']) ** 2)
            amps = amps * sb_response * string_amp

            # Decay (physics + learned correction)
            K_n = (ns ** 2) * piL2
            air_frac = params['air_frac']
            alpha_n = (b1 * (1.0 - air_frac)
                       + b1 * air_frac * torch.sqrt(torch.tensor(freq, device=DEVICE)
                                                      / torch.clamp(partial_freqs, min=20.0))
                       + b2 * K_n)
            alpha_n = alpha_n * torch.exp(decay_correction)

            # Two-stage envelope (hand-tuned defaults + learned offsets)
            prompt_factor = (1.5 + 0.5 * key_pos) + params['prompt_factor_offset']
            A_after_val = (0.15 + 0.05 * key_pos) + params['A_after_offset']
            prompt_rate = alpha_n * prompt_factor * sb_response
            after_rate = alpha_n * params['after_factor']
            A_after_n = torch.clamp(A_after_val / sb_response, max=0.95)
            A_prompt_n = 1.0 - A_after_n

            env = (A_prompt_n.unsqueeze(1) * torch.exp(-prompt_rate.unsqueeze(1) * t.unsqueeze(0))
                   + A_after_n.unsqueeze(1) * torch.exp(-after_rate.unsqueeze(1) * t.unsqueeze(0)))

            phases = PHASE_TABLE_GPU[s_idx, :max_partial]
            sines = torch.sin(2 * np.pi * partial_freqs.unsqueeze(1) * t.unsqueeze(0)
                              + phases.unsqueeze(1))

            partials = amps.unsqueeze(1) * env * sines
            partials = partials * valid.unsqueeze(1).float()
            signal = signal + partials.sum(dim=0)

        return signal


    def synthesize_noise(midi, params, t):
        """Noise component: filtered noise for hammer/room character.

        Uses frequency-domain filtering: multiply noise spectrum by learned
        spectral shape, then IFFT back.
        """
        freq = midi_to_freq(midi)
        n = len(t)

        noise_level = params['noise_level']
        noise_decay = params['noise_decay']
        noise_tilt = params['noise_tilt']
        noise_bw = params['noise_bandwidth']

        # Temporal envelope: exponential decay
        env = noise_level * torch.exp(-t * noise_decay)

        # Use deterministic noise
        raw_noise = NOISE_TABLE[:n]

        # Apply envelope
        shaped_noise = raw_noise * env

        # Frequency-domain filtering via STFT
        n_fft = 2048
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=DEVICE)

        # STFT of shaped noise
        S = torch.stft(shaped_noise, n_fft=n_fft, hop_length=hop,
                        window=window, return_complex=True)

        # Build spectral filter: peaked around note frequency with learned tilt
        freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
        center = freq * noise_bw
        # Broad filter centered on note's spectral region with tilt
        filter_shape = torch.exp(-0.5 * ((freqs - center) / (center * 0.8 + 100)) ** 2)
        # Add spectral tilt: negative tilt = darker (less HF)
        tilt_factor = torch.exp(noise_tilt * torch.log(freqs / (freq + 1e-8) + 1e-8))
        tilt_factor = tilt_factor / (tilt_factor.max() + 1e-8)
        filter_shape = filter_shape * tilt_factor
        # Ensure low frequencies always pass (body of noise)
        filter_shape = torch.clamp(filter_shape, min=0.01)

        # Apply filter
        S_filtered = S * filter_shape.unsqueeze(1)

        # ISTFT back to time domain
        noise_signal = torch.istft(S_filtered, n_fft=n_fft, hop_length=hop,
                                    window=window, length=n)

        return noise_signal


    def synthesize_note(midi, params, t=None):
        """Full synthesis: harmonic + noise, with attack shaping."""
        if t is None:
            t = t_gpu

        # Harmonic component (additive synthesis)
        harmonic = synthesize_harmonic(midi, params, t)

        # Noise component (filtered noise)
        noise = synthesize_noise(midi, params, t)

        # Mix
        signal = harmonic + noise

        # Attack shape
        attack_peak_base = np.interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
        attack_peak = attack_peak_base * params['attack_peak_scale']
        attack_env = torch.where(t < attack_peak,
                                  0.5 - 0.5 * torch.cos(np.pi * t / torch.clamp(attack_peak, min=1e-4)),
                                  torch.ones_like(t))
        signal = signal * attack_env

        # Soundboard IR convolution (per-note spectral coloring)
        signal = apply_soundboard_ir(signal, midi)

        # Fade out
        fade = int(0.1 * SAMPLE_RATE)
        signal = signal.clone()
        signal[-fade:] = signal[-fade:] * torch.linspace(1, 0, fade, device=DEVICE)

        # Normalize
        peak = torch.max(torch.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.85

        return signal


    # ═══ LOSS FUNCTIONS ═══

    def multi_scale_stft_loss(target, generated):
        loss = torch.tensor(0.0, device=DEVICE)
        for n_fft in [512, 1024, 2048, 4096]:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=DEVICE)
            S_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                        window=window, return_complex=True))
            S_g = torch.abs(torch.stft(generated, n_fft=n_fft, hop_length=hop,
                                        window=window, return_complex=True))
            nf = min(S_t.shape[1], S_g.shape[1])
            S_t, S_g = S_t[:, :nf], S_g[:, :nf]
            sc = torch.norm(S_t - S_g) / (torch.norm(S_t) + 1e-8)
            lm = torch.mean(torch.abs(torch.log(S_t + 1e-7) - torch.log(S_g + 1e-7)))
            loss = loss + sc * 2.0 + lm * 0.5
        return loss


    def envelope_loss(target, generated, frame_len=1024, hop=512):
        def rms_frames(x):
            n = (len(x) - frame_len) // hop + 1
            frames = x.unfold(0, frame_len, hop)[:n]
            return torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)
        rms_t = rms_frames(target)
        rms_g = rms_frames(generated)
        n = min(len(rms_t), len(rms_g))
        return torch.mean((rms_t[:n] - rms_g[:n]) ** 2) * 50.0


    def centroid_loss(target, generated, n_fft=2048, hop=512):
        window = torch.hann_window(n_fft, device=DEVICE)
        def compute(x):
            S = torch.abs(torch.stft(x, n_fft=n_fft, hop_length=hop,
                                      window=window, return_complex=True))
            freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
            energy = torch.sum(S, dim=0)
            centroid = torch.sum(freqs.unsqueeze(1) * S, dim=0) / (energy + 1e-8)
            return centroid, energy
        c_t, e_t = compute(target)
        c_g, _ = compute(generated)
        n = min(len(c_t), len(c_g))
        weights = e_t[:n] / (e_t[:n].sum() + 1e-8)
        diff = (c_t[:n] - c_g[:n]) / (c_t[:n] + 100.0)
        return torch.sum(weights * diff ** 2) * 10.0


    def anchor_regularizer(params):
        """L2 penalty pulling all params toward zero (= hand-tuned defaults).

        Every param is defined as an offset from the hand-tuned grand piano model.
        This regularizer ensures the network only deviates when the audio loss
        strongly justifies it, preserving the hand-tuned brightness and character.
        """
        reg = torch.tensor(0.0, device=DEVICE)
        # Damping offsets
        reg = reg + params['log_b1_offset'] ** 2 + params['log_b2_offset'] ** 2
        # Soundboard offsets from defaults
        reg = reg + ((params['sb_bridge_cf'] - 2500.0) / 500.0) ** 2
        reg = reg + ((params['sb_bridge_bw'] - 1200.0) / 400.0) ** 2
        reg = reg + ((params['sb_bridge_gain'] - 0.3) / 0.15) ** 2
        reg = reg + ((params['sb_mode1_gain'] - 0.15) / 0.10) ** 2
        reg = reg + ((params['sb_mode2_gain'] - 0.12) / 0.08) ** 2
        reg = reg + ((params['sb_mode3_gain'] - 0.10) / 0.06) ** 2
        # Decay offsets
        reg = reg + params['prompt_factor_offset'] ** 2
        reg = reg + ((params['after_factor'] - 0.25) / 0.15) ** 2
        reg = reg + params['A_after_offset'] ** 2
        reg = reg + ((params['air_frac'] - 0.2) / 0.15) ** 2
        # Hammer/attack scale offsets from 1.0
        reg = reg + (params['hammer_cutoff_scale'] - 1.0) ** 2
        reg = reg + (params['attack_peak_scale'] - 1.0) ** 2
        # Spectral and decay corrections (should stay small)
        reg = reg + torch.sum(params['spectral_ctrl'] ** 2)
        reg = reg + torch.sum(params['decay_ctrl'] ** 2)
        return reg * 0.03  # moderate anchor — allows meaningful per-note learning


    def compute_loss(target, generated, params=None):
        # No anchor regularizer — tight param bounds keep us near hand-tuned defaults
        return (multi_scale_stft_loss(target, generated)
                + envelope_loss(target, generated)
                + centroid_loss(target, generated))


    # ═══ TRAINING ═══

    def load_audio(path):
        """Load audio file as mono float32 numpy array at SAMPLE_RATE."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            'ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE),
            '-ac', '1', '-f', 'wav', tmp_path
        ], capture_output=True)
        with wave.open(tmp_path, 'r') as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        os.unlink(tmp_path)
        return audio


    def load_targets():
        targets = {}
        for midi, name in TRAIN_NOTES:
            path = os.path.join(BASE, 'audio', 'piano', f'{name}.mp3')
            if not os.path.exists(path):
                print(f"  WARNING: {path} not found")
                continue
            y = load_audio(path)
            if len(y) < N_SAMPLES:
                y = np.pad(y, (0, N_SAMPLES - len(y)))
            else:
                y = y[:N_SAMPLES]
            targets[(midi, name)] = torch.tensor(y, dtype=torch.float32, device=DEVICE)
            print(f"  {name} (MIDI {midi}): loaded")
        return targets


    def train(epochs=2000, lr=2e-3):
        print(f"\nDevice: {DEVICE}")
        if DEVICE.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print("\nLoading target samples...")
        targets = load_targets()
        if not targets:
            print("ERROR: No target samples found!")
            return None

        model = PianoParamNet().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nTraining on {len(targets)} notes for {epochs} epochs")
        print(f"Network: 1→128→128→128→{PianoParamNet.N_PARAMS} ({n_params:,} weights)")
        print(f"Components: harmonic (additive) + noise (filtered)")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        def norm_midi(midi):
            return torch.tensor([(midi - 60) / 30.0], dtype=torch.float32, device=DEVICE)

        best_loss = float('inf')
        best_state = None
        start_time = time.time()
        train_keys = list(targets.keys())

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            np.random.shuffle(train_keys)

            for midi, name in train_keys:
                target = targets[(midi, name)]
                params = model(norm_midi(midi))
                generated = synthesize_note(midi, params)
                loss = compute_loss(target, generated, params)
                epoch_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            avg_loss = epoch_loss / len(targets)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            elapsed = time.time() - start_time
            if epoch == 0 or (epoch + 1) % 50 == 0 or epoch == epochs - 1:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1:4d}/{epochs}: loss={avg_loss:.4f} "
                      f"(best={best_loss:.4f}) lr={lr_now:.5f} [{elapsed:.0f}s]")

            if (epoch + 1) % 250 == 0:
                model.eval()
                with torch.no_grad():
                    note_losses = []
                    for midi, name in train_keys:
                        params = model(norm_midi(midi))
                        gen = synthesize_note(midi, params)
                        nl = compute_loss(targets[(midi, name)], gen, params).item()
                        note_losses.append((name, nl))
                    note_losses.sort(key=lambda x: -x[1])
                    worst3 = ', '.join(f"{n}={l:.3f}" for n, l in note_losses[:3])
                    best3 = ', '.join(f"{n}={l:.3f}" for n, l in note_losses[-3:])
                    print(f"         Worst: {worst3}  |  Best: {best3}")
                model.train()

        if best_state:
            model.load_state_dict(best_state)

        torch.save({
            'model_state': model.state_dict(),
            'best_loss': best_loss,
            'epochs': epochs,
        }, MODEL_PATH)
        print(f"\nModel saved to {MODEL_PATH}")
        print(f"Best loss: {best_loss:.4f}")
        return model


    # ═══ GENERATION ═══

    def write_wav(filename, signal_np):
        pcm = (signal_np * 32767).astype(np.int16)
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())


    def wav_to_mp3(wav_path, mp3_path):
        subprocess.run([
            'ffmpeg', '-y', '-i', wav_path,
            '-codec:a', 'libmp3lame', '-b:a', '128k',
            '-ar', '44100', mp3_path
        ], capture_output=True)


    def generate_samples(model):
        out_dir = os.path.join(BASE, 'audio', 'ddsp-piano')
        os.makedirs(out_dir, exist_ok=True)

        def norm_midi(midi):
            return torch.tensor([(midi - 60) / 30.0], dtype=torch.float32, device=DEVICE)

        model.eval()
        print(f"\nGenerating {len(ALL_NOTES)} DDSP Piano samples...")

        with torch.no_grad():
            for midi, name in ALL_NOTES:
                freq = midi_to_freq(midi)
                params = model(norm_midi(midi))
                signal = synthesize_note(midi, params)
                signal_np = signal.cpu().numpy()

                wav_path = os.path.join(out_dir, f'{name}.wav')
                mp3_path = os.path.join(out_dir, f'{name}.mp3')
                write_wav(wav_path, signal_np)
                wav_to_mp3(wav_path, mp3_path)

                if os.path.exists(mp3_path):
                    os.remove(wav_path)
                    print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz")
                else:
                    print(f"  {name} — WARNING: ffmpeg failed, keeping WAV")

        print(f"\nDone! Samples in {out_dir}/")


    def compare_with_targets(model):
        print("\n" + "=" * 60)
        print("COMPARISON: DDSP Piano vs Recorded Samples")
        print("=" * 60)

        def norm_midi(midi):
            return torch.tensor([(midi - 60) / 30.0], dtype=torch.float32, device=DEVICE)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for midi, name in TRAIN_NOTES:
                path = os.path.join(BASE, 'audio', 'piano', f'{name}.mp3')
                if not os.path.exists(path):
                    continue
                y = load_audio(path)
                if len(y) < N_SAMPLES:
                    y = np.pad(y, (0, N_SAMPLES - len(y)))
                else:
                    y = y[:N_SAMPLES]
                target = torch.tensor(y, dtype=torch.float32, device=DEVICE)

                params = model(norm_midi(midi))
                gen = synthesize_note(midi, params)
                loss = compute_loss(target, gen).item()
                total_loss += loss

                window = torch.hann_window(2048, device=DEVICE)
                def centroid_of(x):
                    S = torch.abs(torch.stft(x[:SAMPLE_RATE * 2], n_fft=2048, hop_length=512,
                                              window=window, return_complex=True))
                    freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
                    return (torch.sum(freqs.unsqueeze(1) * S) / (torch.sum(S) + 1e-8)).item()

                c_t = centroid_of(target)
                c_g = centroid_of(gen)
                ratio = c_g / c_t if c_t > 0 else 0
                print(f"  {name:>4s}: loss={loss:.3f}  centroid={c_t:.0f}→{c_g:.0f} Hz (ratio={ratio:.2f})")

        print(f"\n  Average loss: {total_loss / len(TRAIN_NOTES):.3f}")

        # Show noise params for sample notes
        print(f"\n{'─' * 60}")
        print("LEARNED PARAMETERS (sample notes)")
        print(f"{'─' * 60}")
        for midi, name in [(36, 'C2'), (60, 'C4'), (81, 'A5')]:
            params = model(norm_midi(midi))
            sc = params['spectral_ctrl'].detach().cpu().numpy()
            dc = params['decay_ctrl'].detach().cpu().numpy()
            print(f"\n  {name} (MIDI {midi}):")
            print(f"    b1 offset: {params['log_b1_offset'].item():+.3f} "
                  f"(x{np.exp(params['log_b1_offset'].item()):.2f})")
            print(f"    b2 offset: {params['log_b2_offset'].item():+.3f}")
            print(f"    prompt_offset: {params['prompt_factor_offset'].item():+.3f}  "
                  f"after: {params['after_factor'].item():.3f}  "
                  f"A_after_offset: {params['A_after_offset'].item():+.3f}")
            print(f"    noise: level={params['noise_level'].item():.4f}  "
                  f"decay={params['noise_decay'].item():.1f}  "
                  f"tilt={params['noise_tilt'].item():+.2f}  "
                  f"bw={params['noise_bandwidth'].item():.2f}")
            print(f"    spectral: {' '.join(f'{v:+.1f}' for v in sc)}")
            print(f"    decay:    {' '.join(f'{v:+.1f}' for v in dc)}")

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_optphases():
    """
    Optimize the 3x64 phase table for grand piano synthesis using gradient descent.

    Only optimizes phases (192 values). All other params are fixed to match
    generate_grand_piano.py. Uses mel-scale STFT loss for perceptually weighted
    optimization.

    Re-run after changing any synthesis parameters (rolloff, bridge hill, etc.).
    """

    import numpy as np
    import torch
    import torch.nn.functional as F
    import subprocess
    import tempfile
    import os
    import time

    SAMPLE_RATE = 44100
    DURATION = 4.0

    CALIB_MIDI = [36, 60, 96]
    CALIB_B1 = [0.25, 1.1, 9.17]
    CALIB_B2 = [7.5e-5, 2.7e-4, 2.1e-3]
    CALIB_L = [1.92, 0.62, 0.09]
    B_MIDI = [21, 33, 45, 57, 69, 84, 96]
    B_VALS = [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Per-note soundboard IRs (must match generate_grand_piano.py)
    # Per-note soundboard IRs — mirrors generate_grand_piano.py format detection
    _TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
    _SB_IRS  = {}  # midi → torch tensor IR
    _tf_data = None
    _TF_MODE_OPT = None
    if os.path.exists(_TF_PATH):
        _tf_data = np.load(_TF_PATH)
        if 'irs' in _tf_data:
            _TF_MIDI_OPT   = _tf_data['midi_points']
            _TF_IRS_OPT    = _tf_data['irs']
            _TF_MODE_OPT   = 'irs'
        elif 'transfer_functions' in _tf_data:
            _TF_MIDI_OPT   = _tf_data['midi_points']
            _TF_MAG_OPT    = _tf_data['transfer_functions']
            _TF_IR_LEN_OPT = int(_tf_data['ir_length'])
            _TF_MODE_OPT   = 'mag'
        else:
            _tf_data = None

    if _tf_data is not None:
        def _min_phase_ir_np(mag_resp, ir_len):
            mag = np.maximum(mag_resp, 1e-10)
            n_fft = (len(mag) - 1) * 2
            full = np.concatenate([mag, mag[-2:0:-1]])
            cep = np.fft.ifft(np.log(full)).real
            mc = np.zeros_like(cep)
            mc[0] = cep[0]
            mc[1:n_fft//2] = 2 * cep[1:n_fft//2]
            mc[n_fft//2] = cep[n_fft//2]
            ir = np.fft.ifft(np.exp(np.fft.fft(mc))).real[:ir_len]
            ir *= np.hanning(ir_len * 2)[ir_len:]
            e = np.sqrt(np.sum(ir**2))
            if e > 0:
                ir = ir / e * np.sqrt(ir_len) / np.sqrt(n_fft)
            return ir

        def _get_sb_ir_torch(midi):
            if midi not in _SB_IRS:
                if _TF_MODE_OPT == 'irs':
                    idx = np.searchsorted(_TF_MIDI_OPT, midi)
                    if idx == 0:
                        ir = _TF_IRS_OPT[0].copy()
                    elif idx >= len(_TF_MIDI_OPT):
                        ir = _TF_IRS_OPT[-1].copy()
                    else:
                        lo, hi = idx - 1, idx
                        t = (midi - _TF_MIDI_OPT[lo]) / (_TF_MIDI_OPT[hi] - _TF_MIDI_OPT[lo])
                        ir = (1.0 - t) * _TF_IRS_OPT[lo] + t * _TF_IRS_OPT[hi]
                else:
                    log_tfs = np.log(_TF_MAG_OPT + 1e-10)
                    interp_log = np.array([np.interp(midi, _TF_MIDI_OPT, log_tfs[:, i])
                                           for i in range(log_tfs.shape[1])])
                    ir = _min_phase_ir_np(np.exp(interp_log), _TF_IR_LEN_OPT)
                _SB_IRS[midi] = torch.tensor(ir.astype(np.float32), device=DEVICE)
            return _SB_IRS[midi]
    else:
        _get_sb_ir_torch = lambda midi: None


    def log_interp(midi, midi_pts, vals):
        return float(np.exp(np.interp(midi, midi_pts, np.log(vals))))


    def lin_interp(midi, midi_pts, vals):
        return float(np.interp(midi, midi_pts, vals))


    def load_reference(path, duration=DURATION):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            'ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE),
            '-ac', '1', '-f', 'wav', tmp_path
        ], capture_output=True)
        import wave
        with wave.open(tmp_path, 'r') as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        os.unlink(tmp_path)
        n_samples = int(SAMPLE_RATE * duration)
        if len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))
        else:
            audio = audio[:n_samples]
        return torch.tensor(audio, device=DEVICE)


    def create_mel_filterbank(n_fft, n_mels=80, f_min=20, f_max=8000):
        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700)
        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)
        n_freqs = n_fft // 2 + 1
        mel_points = np.linspace(hz_to_mel(f_min), hz_to_mel(f_max), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.clip(np.round(hz_points * n_fft / SAMPLE_RATE).astype(int), 0, n_freqs - 1)
        filterbank = np.zeros((n_mels, n_freqs))
        for i in range(n_mels):
            left, center, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            if center > left:
                filterbank[i, left:center] = np.linspace(0, 1, center - left, endpoint=False)
            if right > center:
                filterbank[i, center:right] = np.linspace(1, 0, right - center, endpoint=False)
        return torch.tensor(filterbank, dtype=torch.float32, device=DEVICE)


    MEL_BANKS = {nfft: create_mel_filterbank(nfft) for nfft in [512, 1024, 2048, 4096]}


    def mel_stft_loss(predicted, target, n_ffts=[512, 1024, 2048, 4096]):
        loss = torch.tensor(0.0, device=predicted.device)
        for n_fft in n_ffts:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=predicted.device)
            pred_stft = torch.stft(predicted, n_fft, hop, window=window, return_complex=True)
            targ_stft = torch.stft(target, n_fft, hop, window=window, return_complex=True)
            pred_mel = torch.matmul(MEL_BANKS[n_fft], pred_stft.abs() + 1e-8)
            targ_mel = torch.matmul(MEL_BANKS[n_fft], targ_stft.abs() + 1e-8)
            loss += (targ_mel - pred_mel).norm() / targ_mel.norm()
            loss += F.l1_loss(torch.log(pred_mel + 1e-8), torch.log(targ_mel + 1e-8))
        return loss / len(n_ffts)


    def synthesize_note_gpu(midi, phase_table, duration=DURATION):
        freq = 440.0 * 2 ** ((midi - 69) / 12.0)
        n_samples = int(SAMPLE_RATE * duration)
        t = torch.linspace(0, duration, n_samples, device=DEVICE)
        key_pos = max(0, min(1, (midi - 21) / 87))
        velocity = 0.8

        b1 = log_interp(midi, CALIB_MIDI, CALIB_B1)
        b2 = log_interp(midi, CALIB_MIDI, CALIB_B2)
        L = log_interp(midi, CALIB_MIDI, CALIB_L)
        piL2 = (np.pi / L) ** 2
        B = log_interp(midi, B_MIDI, B_VALS)

        max_partial = 1
        while max_partial * freq * np.sqrt(1 + B * max_partial ** 2) < SAMPLE_RATE / 2 - 500:
            max_partial += 1
        max_partial = min(max_partial - 1, 64)

        # Must match generate_grand_piano.py exactly
        vel_clamp = max(velocity, 0.05)
        effective_hardness = 0.15 + 0.85 * vel_clamp ** 1.5
        T_c_base = lin_interp(midi, [36, 60, 96], [0.003, 0.0018, 0.0006])
        T_c = T_c_base * (2.5 - 1.9 * effective_hardness)
        hammer_cutoff = 2.5 / T_c
        hammer_rolloff_exp = 2.0 - 0.8 * effective_hardness
        strike_pos = lin_interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

        prompt_factor = (1.2 + 0.3 * key_pos) * (0.7 + 0.5 * effective_hardness)
        after_factor = 0.45
        A_after = (0.18 + 0.07 * key_pos) * (1.3 - 0.4 * effective_hardness)

        rolloff = 1.0 + 0.3 * key_pos + 1.2 * key_pos ** 2

        if midi < 36:
            string_detunes = [0.0]
        elif midi < 48:
            dc = 0.3 + 0.3 * key_pos
            string_detunes = [-dc, dc]
        else:
            dc = 0.15 + 0.25 * key_pos
            string_detunes = [-dc, 0.0, dc]

        n_strings = len(string_detunes)
        low_modes = [(90, 30, 0.15), (170, 35, 0.12), (260, 45, 0.10)]
        signal = torch.zeros(n_samples, device=DEVICE)

        for s_idx, d_cents in enumerate(string_detunes):
            detune_ratio = 2 ** (d_cents / 1200)
            string_amp = 1.0 / n_strings

            for n in range(1, max_partial + 1):
                partial_freq = n * freq * detune_ratio * np.sqrt(1 + B * n ** 2)
                if partial_freq >= SAMPLE_RATE / 2:
                    break

                amp = 1.0 / (n ** rolloff)
                amp *= 1.0 / (1.0 + (partial_freq / hammer_cutoff) ** hammer_rolloff_exp)
                fTc = partial_freq * T_c
                denom = 1.0 - 4.0 * fTc * fTc
                if abs(denom) < 1e-6:
                    cosine_mod = 1.0
                else:
                    cosine_mod = min(abs(np.cos(np.pi * fTc) / denom), 1.0)
                amp *= 0.7 + 0.3 * cosine_mod
                strike_factor = abs(np.sin(np.pi * n * strike_pos))
                amp *= max(strike_factor, 0.03)

                sb_response = 1.0
                for cf, bw, gain in low_modes:
                    sb_response += gain * np.exp(-0.5 * ((partial_freq - cf) / bw) ** 2)
                sb_response += 0.40 * np.exp(-0.5 * ((partial_freq - 1800) / 800) ** 2)
                amp *= sb_response * string_amp

                K_n = (n ** 2) * piL2
                air_frac = 0.2
                alpha_n = (b1 * (1.0 - air_frac)
                           + b1 * air_frac * np.sqrt(freq / max(partial_freq, 20.0))
                           + b2 * K_n)
                sb_coupling = np.sqrt(sb_response)
                prompt_rate = alpha_n * prompt_factor * sb_coupling
                after_rate = alpha_n * after_factor
                A_after_n = A_after / sb_coupling
                A_prompt_n = 1.0 - A_after_n
                env = A_prompt_n * torch.exp(-t * prompt_rate) + A_after_n * torch.exp(-t * after_rate)

                phase = phase_table[s_idx % 3, (n - 1) % 64]
                signal = signal + amp * env * torch.sin(2 * np.pi * partial_freq * t + phase)

        attack_peak = lin_interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
        attack_env = torch.where(t < attack_peak,
                                 0.5 - 0.5 * torch.cos(np.pi * t / attack_peak),
                                 torch.ones_like(t))
        signal = signal * attack_env

        # Per-note soundboard IR convolution (matching generate_grand_piano.py)
        sb_ir = _get_sb_ir_torch(midi)
        if sb_ir is not None:
            ir = sb_ir.unsqueeze(0).unsqueeze(0)
            sig = signal.unsqueeze(0).unsqueeze(0)
            pad_len = len(sb_ir) - 1
            wet = torch.nn.functional.conv1d(
                torch.nn.functional.pad(sig, (pad_len, 0)),
                ir
            ).squeeze()[:n_samples]
            dry_rms = (signal ** 2).mean().sqrt() + 1e-10
            wet_rms = (wet ** 2).mean().sqrt() + 1e-10
            wet = wet * dry_rms / wet_rms
            signal = 0.15 * signal + 0.85 * wet

        fade_samples = int(0.1 * SAMPLE_RATE)
        fade = torch.linspace(1, 0, fade_samples, device=DEVICE)
        signal = signal.clone()
        signal[-fade_samples:] = signal[-fade_samples:] * fade

        peak = signal.abs().max()
        if peak > 0:
            signal = signal / peak * 0.85
        return signal

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_optgrand():
    """
    GPU-accelerated optimization of grand piano synthesis parameters.

    Optimizes ~21 heuristic parameters against recorded piano samples using
    differential evolution with multi-scale STFT + envelope + centroid loss.
    Synthesis is vectorized on GPU (PyTorch) for ~75x speedup over NumPy.

    Now includes per-note soundboard IR convolution in the synthesis loop,
    so the optimizer focuses on actual physics parameters rather than trying
    to compensate for missing spectral coloring.

    Usage:
        python3 optimize_grand_piano.py              # 5 target notes
        python3 optimize_grand_piano.py --all-notes  # all 17 targets
    """

    import torch
    import numpy as np
    import os
    import sys
    import time
    import subprocess
    import tempfile
    import wave
    from scipy.optimize import differential_evolution

    SAMPLE_RATE = 44100
    DURATION = 4.0  # shorter for faster optimization
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)

    # Load current phase table from generator
    PHASE_TABLE_GPU = torch.tensor(grand()._PHASE_TABLE, dtype=torch.float32, device=DEVICE)

    # Load per-note soundboard transfer functions
    _TF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'soundboard_tf.npz')
    _SB_IRS = {}
    if os.path.exists(_TF_PATH):
        _tf_data = np.load(_TF_PATH)
        _TF_MIDI = _tf_data['midi_points']
        _TF_MAG = _tf_data['transfer_functions']
        _TF_IR_LEN = int(_tf_data['ir_length'])

        def _min_phase_ir_np(mag_resp, ir_len):
            mag = np.maximum(mag_resp, 1e-10)
            n_fft = (len(mag) - 1) * 2
            full = np.concatenate([mag, mag[-2:0:-1]])
            cep = np.fft.ifft(np.log(full)).real
            mc = np.zeros_like(cep)
            mc[0] = cep[0]
            mc[1:n_fft // 2] = 2 * cep[1:n_fft // 2]
            mc[n_fft // 2] = cep[n_fft // 2]
            ir = np.fft.ifft(np.exp(np.fft.fft(mc))).real[:ir_len]
            ir *= np.hanning(ir_len * 2)[ir_len:]
            e = np.sqrt(np.sum(ir ** 2))
            if e > 0:
                ir = ir / e * np.sqrt(ir_len) / np.sqrt(n_fft)
            return ir

        def get_sb_ir(midi):
            if midi not in _SB_IRS:
                log_tfs = np.log(_TF_MAG + 1e-10)
                interp_log = np.array([np.interp(midi, _TF_MIDI, log_tfs[:, i])
                                       for i in range(log_tfs.shape[1])])
                ir = _min_phase_ir_np(np.exp(interp_log), _TF_IR_LEN)
                _SB_IRS[midi] = torch.tensor(ir.astype(np.float32), device=DEVICE)
            return _SB_IRS[midi]
    else:
        get_sb_ir = lambda midi: None

    # ═══ TARGET NOTES ═══
    TARGET_5 = [
        (36, 'C2'), (45, 'A2'), (60, 'C4'), (69, 'A4'), (81, 'A5'),
    ]
    TARGET_ALL = [
        (36, 'C2'), (39, 'Ds2'), (42, 'Fs2'), (45, 'A2'),
        (48, 'C3'), (51, 'Ds3'), (54, 'Fs3'), (57, 'A3'),
        (60, 'C4'), (63, 'Ds4'), (66, 'Fs4'), (69, 'A4'),
        (72, 'C5'), (75, 'Ds5'), (78, 'Fs5'), (81, 'A5'),
        (84, 'C6'),
    ]

    # ═══ PARAMETER DEFINITIONS ═══
    # Current values match generate_grand_piano.py exactly
    PARAM_DEFS = [
        # Damping b1 calibration (log-interpolated at MIDI 36/60/96)
        ('b1_low',              0.08,    0.8),    # C2, currently 0.25
        ('b1_mid',              0.35,    3.5),    # C4, currently 1.1
        ('b1_high',             3.0,    25.0),    # C7, currently 9.17

        # Damping b2 calibration
        ('b2_low',              2e-5,    3e-4),   # C2, currently 7.5e-5
        ('b2_mid',              8e-5,    8e-4),   # C4, currently 2.7e-4
        ('b2_high',             7e-4,    7e-3),   # C7, currently 2.1e-3

        # Soundboard bridge hill
        ('sb_bridge_cf',        800,     3500),   # center freq, currently 1800
        ('sb_bridge_bw',        300,     2000),   # bandwidth, currently 800
        ('sb_bridge_gain',      0.1,     0.8),    # gain, currently 0.40

        # Soundboard low modes (gains only)
        ('sb_mode1_gain',       0.03,    0.35),   # 90 Hz mode, currently 0.15
        ('sb_mode2_gain',       0.03,    0.25),   # 170 Hz mode, currently 0.12
        ('sb_mode3_gain',       0.03,    0.25),   # 260 Hz mode, currently 0.10

        # Two-stage decay (Weinreich)
        ('prompt_base',         0.8,     3.0),    # currently 1.5
        ('prompt_slope',        0.0,     2.0),    # currently 0.5
        ('after_factor',        0.1,     0.6),    # currently 0.25
        ('A_after_base',        0.05,    0.35),   # currently 0.15
        ('A_after_slope',       0.0,     0.2),    # currently 0.05

        # Three-term damping
        ('air_frac',            0.05,    0.5),    # currently 0.2

        # Spectral rolloff
        ('rolloff_base',        0.4,     1.5),    # currently 0.9
        ('rolloff_linear',      0.2,     1.5),    # currently 0.8
        ('rolloff_cubic',       1.0,    10.0),    # currently 4.0
    ]

    BOUNDS = [(d[1], d[2]) for d in PARAM_DEFS]
    PARAM_NAMES = [d[0] for d in PARAM_DEFS]

    CURRENT_PARAMS = [
        0.25,    # b1_low
        1.1,     # b1_mid
        9.17,    # b1_high
        7.5e-5,  # b2_low
        2.7e-4,  # b2_mid
        2.1e-3,  # b2_high
        1800,    # sb_bridge_cf
        800,     # sb_bridge_bw
        0.40,    # sb_bridge_gain
        0.15,    # sb_mode1_gain
        0.12,    # sb_mode2_gain
        0.10,    # sb_mode3_gain
        1.5,     # prompt_base
        0.5,     # prompt_slope
        0.25,    # after_factor
        0.15,    # A_after_base
        0.05,    # A_after_slope
        0.2,     # air_frac
        0.9,     # rolloff_base
        0.8,     # rolloff_linear
        4.0,     # rolloff_cubic
    ]

    # ═══ PHYSICS CONSTANTS (not optimized) ═══
    B_MIDI = torch.tensor([21, 33, 45, 57, 69, 84, 96], dtype=torch.float32, device=DEVICE)
    B_VALS_LOG = torch.log(torch.tensor([3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2],
                                         dtype=torch.float32, device=DEVICE))
    L_MIDI = torch.tensor([36, 60, 96], dtype=torch.float32, device=DEVICE)
    L_VALS_LOG = torch.log(torch.tensor([1.92, 0.62, 0.09], dtype=torch.float32, device=DEVICE))
    SB_MODE_CFS = torch.tensor([90.0, 170.0, 260.0], device=DEVICE)
    SB_MODE_BWS = torch.tensor([30.0, 35.0, 45.0], device=DEVICE)


    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)


    def log_interp_gpu(midi, calib_midi, log_vals):
        midi_t = torch.tensor(float(midi), device=DEVICE)
        if midi <= calib_midi[0].item():
            return torch.exp(log_vals[0])
        if midi >= calib_midi[-1].item():
            return torch.exp(log_vals[-1])
        for i in range(len(calib_midi) - 1):
            if midi <= calib_midi[i + 1].item():
                frac = (midi_t - calib_midi[i]) / (calib_midi[i + 1] - calib_midi[i])
                return torch.exp(log_vals[i] + frac * (log_vals[i + 1] - log_vals[i]))
        return torch.exp(log_vals[-1])


    def generate_note_gpu(midi, params_dict):
        """Vectorized additive synthesis on GPU, matching generate_grand_piano.py."""
        p = params_dict
        freq = midi_to_freq(midi)
        key_pos = max(0.0, min(1.0, (midi - 21) / 87.0))
        t = t_gpu

        # Damping
        b1_log = torch.log(torch.tensor([p['b1_low'], p['b1_mid'], p['b1_high']],
                                         device=DEVICE))
        b2_log = torch.log(torch.tensor([p['b2_low'], p['b2_mid'], p['b2_high']],
                                         device=DEVICE))
        calib = torch.tensor([36.0, 60.0, 96.0], device=DEVICE)
        b1 = log_interp_gpu(midi, calib, b1_log)
        b2 = log_interp_gpu(midi, calib, b2_log)
        L = log_interp_gpu(midi, L_MIDI, L_VALS_LOG)
        piL2 = (np.pi / L) ** 2
        B = torch.exp(log_interp_gpu(midi, B_MIDI, B_VALS_LOG).log())

        # Max partials
        max_partial = 1
        while max_partial * freq * np.sqrt(1 + B.item() * max_partial ** 2) < SAMPLE_RATE / 2 - 500:
            max_partial += 1
        max_partial = min(max_partial - 1, 64)
        if max_partial < 1:
            return torch.zeros(N_SAMPLES, device=DEVICE)

        # Hammer (not optimized)
        p_exp = 2.3 + 0.7 * key_pos
        T_c_base = np.interp(midi, [36, 60, 96], [0.004, 0.0025, 0.0008])
        velocity = 0.8
        T_c = T_c_base * (0.75 / velocity) ** (1.0 / (p_exp + 1))
        hammer_cutoff = 2.5 / T_c
        strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

        # Two-stage decay
        prompt_factor = p['prompt_base'] + p['prompt_slope'] * key_pos
        after_factor = p['after_factor']
        A_after = p['A_after_base'] + p['A_after_slope'] * key_pos

        # Rolloff
        rolloff = p['rolloff_base'] + p['rolloff_linear'] * key_pos + p['rolloff_cubic'] * key_pos ** 3
        air_frac = p['air_frac']

        # Soundboard
        sb_mode_gains = torch.tensor([p['sb_mode1_gain'], p['sb_mode2_gain'], p['sb_mode3_gain']],
                                      device=DEVICE)
        sb_bridge_cf = p['sb_bridge_cf']
        sb_bridge_bw = p['sb_bridge_bw']
        sb_bridge_gain = p['sb_bridge_gain']

        # Detuning (matching generator exactly)
        if midi < 36:
            string_detunes = [0.0]
        elif midi < 48:
            dc = 0.3 + 0.3 * key_pos
            string_detunes = [-dc, dc]
        else:
            dc = 0.15 + 0.25 * key_pos
            string_detunes = [-dc, 0.0, dc]
        n_strings = len(string_detunes)

        ns = torch.arange(1, max_partial + 1, dtype=torch.float32, device=DEVICE)
        signal = torch.zeros(N_SAMPLES, device=DEVICE)

        for s_idx, d_cents in enumerate(string_detunes):
            detune_ratio = 2 ** (d_cents / 1200)
            string_amp = 1.0 / n_strings
            partial_freqs = ns * freq * detune_ratio * torch.sqrt(1 + B * ns ** 2)
            valid = partial_freqs < SAMPLE_RATE / 2
            if not valid.any():
                continue

            amps = 1.0 / (ns ** rolloff)
            amps = amps * 1.0 / (1.0 + (partial_freqs / hammer_cutoff) ** 1.5)
            fTc = partial_freqs * T_c
            denom = 1.0 - 4.0 * fTc * fTc
            safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
            cosine_mod = torch.clamp(torch.abs(torch.cos(np.pi * fTc) / safe_denom), max=1.0)
            cosine_mod = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(cosine_mod), cosine_mod)
            amps = amps * (0.7 + 0.3 * cosine_mod)

            strike_factor = torch.abs(torch.sin(np.pi * ns * strike_pos))
            amps = amps * torch.clamp(strike_factor, min=0.03)

            sb_response = torch.ones_like(partial_freqs)
            for i in range(3):
                sb_response = sb_response + sb_mode_gains[i] * torch.exp(
                    -0.5 * ((partial_freqs - SB_MODE_CFS[i]) / SB_MODE_BWS[i]) ** 2)
            sb_response = sb_response + sb_bridge_gain * torch.exp(
                -0.5 * ((partial_freqs - sb_bridge_cf) / sb_bridge_bw) ** 2)
            amps = amps * sb_response * string_amp

            K_n = (ns ** 2) * piL2
            alpha_n = (b1 * (1.0 - air_frac)
                       + b1 * air_frac * torch.sqrt(torch.tensor(freq, device=DEVICE)
                                                     / torch.clamp(partial_freqs, min=20.0))
                       + b2 * K_n)

            prompt_rate = alpha_n * prompt_factor * sb_response
            after_rate = alpha_n * after_factor
            A_after_n = torch.clamp(A_after / sb_response, max=0.95)
            A_prompt_n = 1.0 - A_after_n

            env = (A_prompt_n.unsqueeze(1) * torch.exp(-prompt_rate.unsqueeze(1) * t.unsqueeze(0))
                   + A_after_n.unsqueeze(1) * torch.exp(-after_rate.unsqueeze(1) * t.unsqueeze(0)))

            phases = PHASE_TABLE_GPU[s_idx % 3, :max_partial]
            sines = torch.sin(2 * np.pi * partial_freqs.unsqueeze(1) * t.unsqueeze(0)
                              + phases.unsqueeze(1))

            partials = amps.unsqueeze(1) * env * sines
            partials = partials * valid.unsqueeze(1).float()
            signal = signal + partials.sum(dim=0)

        # Attack shape
        attack_peak = np.interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
        attack_env = torch.where(t < attack_peak,
                                  0.5 - 0.5 * torch.cos(np.pi * t / attack_peak),
                                  torch.ones_like(t))
        signal = signal * attack_env

        # Per-note soundboard IR convolution
        sb_ir = get_sb_ir(midi)
        if sb_ir is not None:
            ir = sb_ir.unsqueeze(0).unsqueeze(0)
            sig = signal.unsqueeze(0).unsqueeze(0)
            pad_len = len(sb_ir) - 1
            wet = torch.nn.functional.conv1d(
                torch.nn.functional.pad(sig, (pad_len, 0)), ir
            ).squeeze()[:N_SAMPLES]
            dry_rms = (signal ** 2).mean().sqrt() + 1e-10
            wet_rms = (wet ** 2).mean().sqrt() + 1e-10
            wet = wet * dry_rms / wet_rms
            signal = 0.3 * signal + 0.7 * wet

        # Fade out
        fade = int(0.1 * SAMPLE_RATE)
        signal[-fade:] = signal[-fade:] * torch.linspace(1, 0, fade, device=DEVICE)

        # Normalize
        peak = torch.max(torch.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.85

        return signal


    # ═══ LOSS FUNCTIONS ═══

    def multi_scale_stft_loss(target, generated):
        loss = torch.tensor(0.0, device=DEVICE)
        for n_fft in [512, 1024, 2048, 4096]:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=DEVICE)
            S_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                        window=window, return_complex=True))
            S_g = torch.abs(torch.stft(generated, n_fft=n_fft, hop_length=hop,
                                        window=window, return_complex=True))
            nf = min(S_t.shape[1], S_g.shape[1])
            S_t, S_g = S_t[:, :nf], S_g[:, :nf]
            sc = torch.norm(S_t - S_g) / (torch.norm(S_t) + 1e-8)
            lm = torch.mean(torch.abs(torch.log(S_t + 1e-7) - torch.log(S_g + 1e-7)))
            loss = loss + sc * 2.0 + lm * 0.5
        return loss


    def envelope_loss(target, generated, frame_len=1024, hop=512):
        def rms_frames(x):
            n = (len(x) - frame_len) // hop + 1
            frames = x.unfold(0, frame_len, hop)[:n]
            return torch.sqrt(torch.mean(frames ** 2, dim=1))
        rms_t = rms_frames(target)
        rms_g = rms_frames(generated)
        n = min(len(rms_t), len(rms_g))
        return torch.mean((rms_t[:n] - rms_g[:n]) ** 2) * 50.0


    def centroid_loss(target, generated, n_fft=2048, hop=512):
        window = torch.hann_window(n_fft, device=DEVICE)
        def compute_centroid_and_energy(x):
            S = torch.abs(torch.stft(x, n_fft=n_fft, hop_length=hop,
                                      window=window, return_complex=True))
            freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
            energy = torch.sum(S, dim=0)
            centroid = torch.sum(freqs.unsqueeze(1) * S, dim=0) / (energy + 1e-8)
            return centroid, energy
        c_t, e_t = compute_centroid_and_energy(target)
        c_g, _ = compute_centroid_and_energy(generated)
        n = min(len(c_t), len(c_g))
        weights = e_t[:n] / (e_t[:n].sum() + 1e-8)
        diff = (c_t[:n] - c_g[:n]) / (c_t[:n] + 100.0)
        return torch.sum(weights * diff ** 2) * 10.0


    def compute_loss_gpu(target, generated):
        return (multi_scale_stft_loss(target, generated)
                + envelope_loss(target, generated)
                + centroid_loss(target, generated)).item()


    # ═══ REFERENCE LOADING ═══

    def load_mp3(path):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            'ffmpeg', '-y', '-i', path, '-ar', str(SAMPLE_RATE),
            '-ac', '1', '-f', 'wav', tmp_path
        ], capture_output=True)
        with wave.open(tmp_path, 'r') as wf:
            raw = wf.readframes(wf.getnframes())
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        os.unlink(tmp_path)
        return audio


    # ═══ OBJECTIVE ═══

    targets_gpu = {}
    TARGET_NOTES = []  # mutated in place by cmd_optimize_grand before DE runs


    def objective(params):
        params_dict = dict(zip(PARAM_NAMES, params))
        total_loss = 0.0
        for midi, name in TARGET_NOTES:
            try:
                gen = generate_note_gpu(midi, params_dict)
                total_loss += compute_loss_gpu(targets_gpu[name], gen)
            except Exception:
                return 1e6
        return total_loss / len(TARGET_NOTES)

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_optrhodes():
    """
    GPU-accelerated optimization of FM Rhodes parameters to match sampled Rhodes.

    Runs all synthesis and loss computation on CUDA (RTX 4080).
    Uses differential evolution with batched GPU evaluation.
    """

    import torch
    import torch.nn.functional as F
    import numpy as np
    import librosa
    import os
    import time
    from scipy.optimize import differential_evolution

    SAMPLE_RATE = 44100
    DURATION = 4.0
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    TARGET_NOTES = [
        (50, 'D3'),
        (62, 'D4'),
        (71, 'B4'),
    ]

    PARAM_DEFS = [
        ('body_mod_idx',       0.5,  4.0),
        ('mod_decay_base',     0.3,  5.0),
        ('mod_decay_kscale',   0.0,  3.0),
        ('tine_ratio',         3.0, 20.0),
        ('tine_mod_idx',       0.1,  3.0),
        ('tine_mod_decay',     2.0, 25.0),
        ('tine_carrier_decay', 1.0, 15.0),
        ('tine_level',         0.02, 0.40),
        ('sub_level',          0.0,  0.30),
        ('h2_level',           0.0,  0.25),
        ('h3_level',           0.0,  0.15),
        ('body_mix',           0.30, 0.85),
        ('drive_base',         0.1,  3.0),
        ('drive_kscale',       0.0,  2.0),
        ('asym',               0.2,  1.0),
        ('decay_rate_base',    0.2,  1.5),
        ('decay_rate_kscale',  0.0,  0.03),
    ]

    BOUNDS = [(d[1], d[2]) for d in PARAM_DEFS]
    PARAM_NAMES = [d[0] for d in PARAM_DEFS]

    # Pre-compute time vector on GPU
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)


    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)


    def generate_note_gpu(midi, params_dict):
        """Generate FM Rhodes note entirely on GPU."""
        p = params_dict
        freq = midi_to_freq(midi)
        t = t_gpu
        n = N_SAMPLES

        key_scale = max(0.25, min(1.0, 1.0 - (midi - 40) / 60))

        # Amplitude envelope
        attack = torch.clamp(t / 0.001, max=1.0)
        decay_rate = p['decay_rate_base'] + (midi - 35) * p['decay_rate_kscale']
        env = attack * torch.exp(-t * decay_rate)

        # Body (1:1 FM)
        body_mod_idx = p['body_mod_idx'] * 0.75 * key_scale
        mod_decay_rate = p['mod_decay_base'] + (1.0 - key_scale) * p['mod_decay_kscale']
        body_mod_env = body_mod_idx * torch.exp(-t * mod_decay_rate)
        phase = 2 * np.pi * freq * t
        body_mod = body_mod_env * torch.sin(phase)
        body = torch.sin(phase + body_mod)

        # Tine
        tine_mod_freq = freq * p['tine_ratio']
        if tine_mod_freq < SAMPLE_RATE / 2 - 1000:
            tine_mod_idx = p['tine_mod_idx'] * 0.75
            tine_mod_env = tine_mod_idx * torch.exp(-t * p['tine_mod_decay'])
            tine_phase = 2 * np.pi * tine_mod_freq * t
            tine_mod = tine_mod_env * torch.sin(tine_phase)
            tine_carrier_env = torch.exp(-t * p['tine_carrier_decay'])
            tine = tine_carrier_env * torch.sin(phase + tine_mod)
        else:
            tine = torch.zeros(n, device=DEVICE)

        # Sub-harmonic
        sub = p['sub_level'] * key_scale * torch.sin(np.pi * freq * t)

        # Additive harmonics
        h2 = p['h2_level'] * torch.sin(4 * np.pi * freq * t)
        h3 = p['h3_level'] * torch.sin(6 * np.pi * freq * t)

        # Mix
        signal = body * p['body_mix'] + tine * p['tine_level'] + sub + h2 + h3
        signal = signal * env

        # Pickup distortion
        drive = p['drive_base'] + p['drive_kscale'] * key_scale ** 2
        asym = p['asym']
        drive_c = max(drive, 0.01)
        asym_c = max(drive * asym, 0.01)
        pos = torch.clamp(signal, min=0)
        neg = torch.clamp(signal, max=0)
        signal = torch.tanh(pos * drive_c) / np.tanh(drive_c) + torch.tanh(neg * asym_c) / np.tanh(asym_c)

        # Fade out
        fade = int(0.05 * SAMPLE_RATE)
        fade_env = torch.linspace(1, 0, fade, device=DEVICE)
        signal[-fade:] *= fade_env

        # Normalize
        peak = torch.max(torch.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.85

        return signal


    def multi_scale_stft_loss(target, generated):
        """Multi-scale STFT loss computed on GPU."""
        loss = torch.tensor(0.0, device=DEVICE)

        for n_fft in [512, 1024, 2048]:
            hop = n_fft // 4
            # Compute STFT using torch
            S_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                        window=torch.hann_window(n_fft, device=DEVICE),
                                        return_complex=True))
            S_g = torch.abs(torch.stft(generated, n_fft=n_fft, hop_length=hop,
                                        window=torch.hann_window(n_fft, device=DEVICE),
                                        return_complex=True))

            # Trim to same size
            nf = min(S_t.shape[1], S_g.shape[1])
            S_t = S_t[:, :nf]
            S_g = S_g[:, :nf]

            # Spectral convergence
            sc = torch.norm(S_t - S_g) / (torch.norm(S_t) + 1e-8)

            # Log-magnitude loss
            log_t = torch.log(S_t + 1e-7)
            log_g = torch.log(S_g + 1e-7)
            lm = torch.mean(torch.abs(log_t - log_g))

            loss += sc * 2.0 + lm * 0.5

        return loss


    def envelope_loss(target, generated, frame_len=1024, hop=512):
        """RMS envelope trajectory loss on GPU."""
        def rms_frames(x):
            # Unfold into frames
            n = (len(x) - frame_len) // hop + 1
            frames = x.unfold(0, frame_len, hop)[:n]
            return torch.sqrt(torch.mean(frames ** 2, dim=1))

        rms_t = rms_frames(target)
        rms_g = rms_frames(generated)
        n = min(len(rms_t), len(rms_g))
        return torch.mean((rms_t[:n] - rms_g[:n]) ** 2) * 50.0


    def centroid_loss(target, generated, n_fft=2048, hop=512):
        """Spectral centroid trajectory loss on GPU."""
        window = torch.hann_window(n_fft, device=DEVICE)

        def compute_centroid(x):
            S = torch.abs(torch.stft(x, n_fft=n_fft, hop_length=hop,
                                      window=window, return_complex=True))
            freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
            centroid = torch.sum(freqs.unsqueeze(1) * S, dim=0) / (torch.sum(S, dim=0) + 1e-8)
            return centroid

        c_t = compute_centroid(target)
        c_g = compute_centroid(generated)
        n = min(len(c_t), len(c_g))
        return torch.mean(((c_t[:n] - c_g[:n]) / (c_t[:n] + 1e-8)) ** 2) * 10.0


    def compute_loss_gpu(target, generated):
        """Total perceptual loss on GPU."""
        loss = multi_scale_stft_loss(target, generated)
        loss += envelope_loss(target, generated)
        loss += centroid_loss(target, generated)
        return loss.item()

    targets_gpu = {}  # mutated in place by cmd_optimize_rhodes before DE runs

    def objective(params):
        """Total loss across all target notes."""
        params_dict = dict(zip(PARAM_NAMES, params))
        total_loss = 0.0
        for midi, name in TARGET_NOTES:
            try:
                gen = generate_note_gpu(midi, params_dict)
                total_loss += compute_loss_gpu(targets_gpu[name], gen)
            except Exception:
                return 1e6
        return total_loss / len(TARGET_NOTES)


    CURRENT_PARAMS = [
        2.0,   # body_mod_idx
        1.8,   # mod_decay_base
        0.5,   # mod_decay_kscale
        14.0,  # tine_ratio
        1.2,   # tine_mod_idx
        6.0,   # tine_mod_decay
        3.0,   # tine_carrier_decay
        0.18,  # tine_level
        0.18,  # sub_level
        0.10,  # h2_level
        0.05,  # h3_level
        0.58,  # body_mix
        0.3,   # drive_base
        0.2,   # drive_kscale
        0.5,   # asym
        0.5,   # decay_rate_base
        0.01,  # decay_rate_kscale
    ]

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_tunewarmth():
    """
    Iterative per-parameter fine-tuning of grand piano synthesis.

    Step 1: Rolloff (warmth) — sweeps rolloff_base, rolloff_linear, rolloff_cubic
    to find values that better match recorded samples while keeping the character.

    Uses GPU synthesis from optimize_grand_piano.py for fast evaluation.
    """

    import torch
    import numpy as np
    import librosa
    import os
    import time
    import itertools

    SAMPLE_RATE = 44100
    DURATION = 6.0
    N_SAMPLES = int(SAMPLE_RATE * DURATION)
    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    t_gpu = torch.linspace(0, DURATION, N_SAMPLES, device=DEVICE)
    PHASE_TABLE_GPU = torch.tensor(
        np.random.RandomState(6454).uniform(0, 2 * np.pi, (3, 64)),
        dtype=torch.float32, device=DEVICE,
    )

    # All 17 recorded references
    TARGET_NOTES = [
        (36, 'C2'), (39, 'Ds2'), (42, 'Fs2'), (45, 'A2'),
        (48, 'C3'), (51, 'Ds3'), (54, 'Fs3'), (57, 'A3'),
        (60, 'C4'), (63, 'Ds4'), (66, 'Fs4'), (69, 'A4'),
        (72, 'C5'), (75, 'Ds5'), (78, 'Fs5'), (81, 'A5'),
        (84, 'C6'),
    ]

    # Physics constants
    B_MIDI = torch.tensor([21, 33, 45, 57, 69, 84, 96], dtype=torch.float32, device=DEVICE)
    B_VALS_LOG = torch.log(torch.tensor(
        [3.1e-4, 2.5e-4, 2.0e-4, 2.2e-4, 7.5e-4, 5.0e-3, 4.0e-2],
        dtype=torch.float32, device=DEVICE))
    L_MIDI = torch.tensor([36, 60, 96], dtype=torch.float32, device=DEVICE)
    L_VALS_LOG = torch.log(torch.tensor([1.92, 0.62, 0.09], dtype=torch.float32, device=DEVICE))
    SB_MODE_CFS = torch.tensor([90.0, 170.0, 260.0], device=DEVICE)
    SB_MODE_BWS = torch.tensor([30.0, 35.0, 45.0], device=DEVICE)


    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)


    def log_interp(midi, calib_midi, log_vals):
        midi_f = float(midi)
        if midi_f <= calib_midi[0].item():
            return torch.exp(log_vals[0])
        if midi_f >= calib_midi[-1].item():
            return torch.exp(log_vals[-1])
        for i in range(len(calib_midi) - 1):
            if midi_f <= calib_midi[i + 1].item():
                frac = (midi_f - calib_midi[i].item()) / (calib_midi[i + 1].item() - calib_midi[i].item())
                return torch.exp(log_vals[i] + frac * (log_vals[i + 1] - log_vals[i]))
        return torch.exp(log_vals[-1])


    def generate_note_gpu(midi, rolloff_base, rolloff_linear, rolloff_cubic,
                          sb_bridge_cf=2500, sb_bridge_bw=1200, sb_bridge_gain=0.3,
                          prompt_base=1.5, prompt_slope=0.5, after_factor_val=0.25,
                          a_after_base=0.15, a_after_slope=0.05):
        """GPU synthesis matching generate_grand_piano.py but with tunable params."""
        freq = midi_to_freq(midi)
        key_pos = max(0.0, min(1.0, (midi - 21) / 87.0))
        t = t_gpu

        # Physics params
        b1_log = torch.log(torch.tensor([0.25, 1.1, 9.17], device=DEVICE))
        b2_log = torch.log(torch.tensor([7.5e-5, 2.7e-4, 2.1e-3], device=DEVICE))
        calib = torch.tensor([36.0, 60.0, 96.0], device=DEVICE)
        b1 = log_interp(midi, calib, b1_log)
        b2 = log_interp(midi, calib, b2_log)
        L = log_interp(midi, L_MIDI, L_VALS_LOG)
        piL2 = (np.pi / L) ** 2
        B = log_interp(midi, B_MIDI, B_VALS_LOG)

        max_partial = 1
        while max_partial * freq * np.sqrt(1 + B.item() * max_partial**2) < SAMPLE_RATE / 2 - 500:
            max_partial += 1
        max_partial = min(max_partial - 1, 64)
        if max_partial < 1:
            return torch.zeros(N_SAMPLES, device=DEVICE)

        # Hammer
        p_exp = 2.3 + 0.7 * key_pos
        T_c_base = np.interp(midi, [36, 60, 96], [0.004, 0.0025, 0.0008])
        T_c = T_c_base * (0.75 / 0.8) ** (1.0 / (p_exp + 1))
        hammer_cutoff = 2.5 / T_c
        strike_pos = np.interp(midi, [36, 60, 96], [0.12, 0.12, 0.0625])

        # Two-stage decay
        prompt_factor = prompt_base + prompt_slope * key_pos
        after_factor = after_factor_val
        A_after = a_after_base + a_after_slope * key_pos

        # Rolloff — THE PARAMETER BEING TUNED
        rolloff = rolloff_base + rolloff_linear * key_pos + rolloff_cubic * key_pos ** 3

        # Strings
        if midi < 36:
            string_detunes = [0.0]
        elif midi < 48:
            string_detunes = [-(0.3 + 0.3 * key_pos), (0.3 + 0.3 * key_pos)]
        else:
            dc = 0.5 + 1.0 * key_pos
            string_detunes = [-dc, 0.0, dc]
        n_strings = len(string_detunes)

        ns = torch.arange(1, max_partial + 1, dtype=torch.float32, device=DEVICE)
        signal = torch.zeros(N_SAMPLES, device=DEVICE)

        sb_mode_gains = torch.tensor([0.15, 0.12, 0.10], device=DEVICE)

        for s_idx, d_cents in enumerate(string_detunes):
            detune_ratio = 2 ** (d_cents / 1200)
            string_amp = 1.0 / n_strings

            partial_freqs = ns * freq * detune_ratio * torch.sqrt(1 + B * ns ** 2)
            valid = partial_freqs < SAMPLE_RATE / 2

            amps = 1.0 / (ns ** rolloff)

            # Hammer
            amps = amps / (1.0 + (partial_freqs / hammer_cutoff) ** 1.5)
            fTc = partial_freqs * T_c
            denom = 1.0 - 4.0 * fTc * fTc
            safe_denom = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(denom), denom)
            cosine_mod = torch.clamp(torch.abs(torch.cos(np.pi * fTc) / safe_denom), max=1.0)
            cosine_mod = torch.where(torch.abs(denom) < 1e-6, torch.ones_like(cosine_mod), cosine_mod)
            amps = amps * (0.7 + 0.3 * cosine_mod)

            # Strike
            strike_factor = torch.abs(torch.sin(np.pi * ns * strike_pos))
            amps = amps * torch.clamp(strike_factor, min=0.03)

            # Soundboard
            sb_response = torch.ones_like(partial_freqs)
            for i in range(3):
                sb_response = sb_response + sb_mode_gains[i] * torch.exp(
                    -0.5 * ((partial_freqs - SB_MODE_CFS[i]) / SB_MODE_BWS[i]) ** 2)
            sb_response = sb_response + sb_bridge_gain * torch.exp(
                -0.5 * ((partial_freqs - sb_bridge_cf) / sb_bridge_bw) ** 2)
            amps = amps * sb_response * string_amp

            # Decay
            K_n = (ns ** 2) * piL2
            air_frac = 0.2
            alpha_n = (b1 * (1.0 - air_frac)
                       + b1 * air_frac * torch.sqrt(torch.tensor(freq, device=DEVICE)
                                                      / torch.clamp(partial_freqs, min=20.0))
                       + b2 * K_n)

            prompt_rate = alpha_n * prompt_factor * sb_response
            after_rate = alpha_n * after_factor
            A_after_n = torch.clamp(torch.tensor(A_after, device=DEVICE) / sb_response, max=0.95)
            A_prompt_n = 1.0 - A_after_n

            env = (A_prompt_n.unsqueeze(1) * torch.exp(-prompt_rate.unsqueeze(1) * t.unsqueeze(0))
                   + A_after_n.unsqueeze(1) * torch.exp(-after_rate.unsqueeze(1) * t.unsqueeze(0)))

            phases = PHASE_TABLE_GPU[s_idx, :max_partial]
            sines = torch.sin(2 * np.pi * partial_freqs.unsqueeze(1) * t.unsqueeze(0)
                              + phases.unsqueeze(1))

            partials = amps.unsqueeze(1) * env * sines
            partials = partials * valid.unsqueeze(1).float()
            signal = signal + partials.sum(dim=0)

        # Attack
        attack_peak = np.interp(midi, [36, 60, 96], [0.050, 0.030, 0.012])
        attack_env = torch.where(t < attack_peak,
                                  0.5 - 0.5 * torch.cos(np.pi * t / attack_peak),
                                  torch.ones_like(t))
        signal = signal * attack_env

        # Fade
        fade = int(0.1 * SAMPLE_RATE)
        signal[-fade:] = signal[-fade:] * torch.linspace(1, 0, fade, device=DEVICE)

        # Normalize
        peak = torch.max(torch.abs(signal))
        if peak > 0:
            signal = signal / peak * 0.85

        return signal


    # Loss functions
    def multi_scale_stft_loss(target, generated):
        loss = torch.tensor(0.0, device=DEVICE)
        for n_fft in [512, 1024, 2048, 4096]:
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=DEVICE)
            S_t = torch.abs(torch.stft(target, n_fft=n_fft, hop_length=hop,
                                        window=window, return_complex=True))
            S_g = torch.abs(torch.stft(generated, n_fft=n_fft, hop_length=hop,
                                        window=window, return_complex=True))
            nf = min(S_t.shape[1], S_g.shape[1])
            S_t, S_g = S_t[:, :nf], S_g[:, :nf]
            sc = torch.norm(S_t - S_g) / (torch.norm(S_t) + 1e-8)
            lm = torch.mean(torch.abs(torch.log(S_t + 1e-7) - torch.log(S_g + 1e-7)))
            loss = loss + sc * 2.0 + lm * 0.5
        return loss


    def envelope_loss(target, generated, frame_len=1024, hop=512):
        def rms_frames(x):
            n = (len(x) - frame_len) // hop + 1
            frames = x.unfold(0, frame_len, hop)[:n]
            return torch.sqrt(torch.mean(frames ** 2, dim=1) + 1e-8)
        rms_t = rms_frames(target)
        rms_g = rms_frames(generated)
        n = min(len(rms_t), len(rms_g))
        return torch.mean((rms_t[:n] - rms_g[:n]) ** 2) * 50.0


    def centroid_loss(target, generated, n_fft=2048, hop=512):
        window = torch.hann_window(n_fft, device=DEVICE)
        def compute(x):
            S = torch.abs(torch.stft(x, n_fft=n_fft, hop_length=hop,
                                      window=window, return_complex=True))
            freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
            energy = torch.sum(S, dim=0)
            centroid = torch.sum(freqs.unsqueeze(1) * S, dim=0) / (energy + 1e-8)
            return centroid, energy
        c_t, e_t = compute(target)
        c_g, _ = compute(generated)
        n = min(len(c_t), len(c_g))
        weights = e_t[:n] / (e_t[:n].sum() + 1e-8)
        diff = (c_t[:n] - c_g[:n]) / (c_t[:n] + 100.0)
        return torch.sum(weights * diff ** 2) * 10.0


    def compute_loss(target, generated):
        return (multi_scale_stft_loss(target, generated)
                + envelope_loss(target, generated)
                + centroid_loss(target, generated)).item()


    def centroid_of(x, window):
        S = torch.abs(torch.stft(x[:SAMPLE_RATE * 2], n_fft=2048, hop_length=512,
                                  window=window, return_complex=True))
        freqs = torch.linspace(0, SAMPLE_RATE / 2, S.shape[0], device=DEVICE)
        return (torch.sum(freqs.unsqueeze(1) * S) / (torch.sum(S) + 1e-8)).item()

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_comparepiano():
    """Compare generated grand piano samples against the original recorded piano samples."""

    import numpy as np
    import subprocess
    import os
    import wave
    import tempfile


    def mp3_to_wav_array(mp3_path):
        """Decode MP3 to numpy array via ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_path, '-ar', '44100', '-ac', '1', tmp_path
        ], capture_output=True)
        with wave.open(tmp_path, 'r') as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(float)
            data /= 32768.0
        os.remove(tmp_path)
        return data


    def analyze_note(signal, sr=44100, label=""):
        """Analyze a piano note's characteristics."""
        n = len(signal)
        duration = n / sr

        # Find peak and measure attack time
        abs_sig = np.abs(signal)
        peak_idx = np.argmax(abs_sig)
        peak_time = peak_idx / sr
        peak_val = abs_sig[peak_idx]

        # Measure amplitude at key time points
        times = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
        amp_at = {}
        window = int(sr * 0.02)  # 20ms RMS window
        for t_sec in times:
            idx = int(t_sec * sr)
            if idx + window < n:
                rms = np.sqrt(np.mean(signal[idx:idx+window]**2))
                amp_at[t_sec] = rms

        # Normalize amplitudes to peak RMS
        peak_rms = max(amp_at.values()) if amp_at else 1
        amp_db = {}
        for t_sec, rms in amp_at.items():
            if rms > 0 and peak_rms > 0:
                amp_db[t_sec] = 20 * np.log10(rms / peak_rms)
            else:
                amp_db[t_sec] = -100

        # Decay analysis: time to -10dB, -20dB, -40dB
        rms_env = []
        hop = int(sr * 0.01)  # 10ms hop
        for i in range(0, n - window, hop):
            rms_env.append(np.sqrt(np.mean(signal[i:i+window]**2)))
        rms_env = np.array(rms_env)
        if len(rms_env) == 0:
            return {}
        peak_rms_env = np.max(rms_env)
        if peak_rms_env > 0:
            rms_db_env = 20 * np.log10(np.maximum(rms_env / peak_rms_env, 1e-10))
        else:
            rms_db_env = np.full_like(rms_env, -100)

        decay_times = {}
        for threshold in [-10, -20, -40]:
            below = np.where(rms_db_env < threshold)[0]
            if len(below) > 0:
                decay_times[threshold] = below[0] * 0.01
            else:
                decay_times[threshold] = duration

        # Spectral analysis: first 200ms (attack) and 500ms-1500ms (sustain)
        def spectral_profile(start_s, end_s):
            s = int(start_s * sr)
            e = min(int(end_s * sr), n)
            chunk = signal[s:e]
            if len(chunk) < 1024:
                return None, None, None
            # Apply window
            w = np.hanning(len(chunk))
            spectrum = np.abs(np.fft.rfft(chunk * w))
            freqs = np.fft.rfftfreq(len(chunk), 1/sr)
            # Spectral centroid
            if np.sum(spectrum) > 0:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
            else:
                centroid = 0
            # Energy in bands
            bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 20000)]
            band_energy = {}
            total_energy = np.sum(spectrum**2)
            for lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                be = np.sum(spectrum[mask]**2)
                band_energy[f"{lo}-{hi}"] = be / total_energy * 100 if total_energy > 0 else 0
            return centroid, band_energy, spectrum

        attack_centroid, attack_bands, attack_spec = spectral_profile(0.0, 0.2)
        sustain_centroid, sustain_bands, sustain_spec = spectral_profile(0.5, 1.5)

        # Spectral slope (dB/octave) in sustain region
        if sustain_spec is not None:
            freqs = np.fft.rfftfreq(int(1.0 * sr), 1/sr)  # approx
            # Measure energy at fundamental vs 5x fundamental
            # (we don't know the fundamental, but centroid gives a rough sense)

        return {
            'duration': duration,
            'peak_time': peak_time,
            'decay_10dB': decay_times.get(-10, None),
            'decay_20dB': decay_times.get(-20, None),
            'decay_40dB': decay_times.get(-40, None),
            'amp_envelope_dB': amp_db,
            'attack_centroid': attack_centroid,
            'sustain_centroid': sustain_centroid,
            'attack_bands': attack_bands,
            'sustain_bands': sustain_bands,
        }


    def format_bands(bands):
        if bands is None:
            return "N/A"
        parts = []
        for band, pct in sorted(bands.items(), key=lambda x: int(x[0].split('-')[0])):
            if pct > 0.5:
                parts.append(f"{band}:{pct:.1f}%")
        return "  ".join(parts)

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_comparerhodes():
    """Compare the sampled Rhodes vs FM Rhodes to identify spectral differences."""

    import numpy as np
    import subprocess
    import tempfile
    import wave
    import os

    SAMPLE_RATE = 44100

    def load_mp3_as_numpy(mp3_path):
        """Convert MP3 to WAV in memory, return numpy array."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        subprocess.run([
            'ffmpeg', '-y', '-i', mp3_path, '-ar', str(SAMPLE_RATE),
            '-ac', '1', '-f', 'wav', tmp_path
        ], capture_output=True)
        with wave.open(tmp_path, 'r') as wf:
            frames = wf.readframes(wf.getnframes())
            signal = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
        os.remove(tmp_path)
        return signal


    def analyze_note(signal, label, note_name):
        """Analyze a note's spectral and temporal characteristics."""
        print(f"\n{'='*60}")
        print(f"  {label} — {note_name}")
        print(f"{'='*60}")

        # Trim to 4 seconds max
        n = min(len(signal), SAMPLE_RATE * 4)
        signal = signal[:n]

        # --- Amplitude envelope ---
        # Compute RMS in 10ms windows
        win = int(0.01 * SAMPLE_RATE)
        rms = []
        for i in range(0, n - win, win):
            rms.append(np.sqrt(np.mean(signal[i:i+win]**2)))
        rms = np.array(rms)

        peak_idx = np.argmax(rms)
        peak_time = peak_idx * 0.01
        peak_val = rms[peak_idx]

        # Find decay to 50% and 10%
        half_idx = next((i for i in range(peak_idx, len(rms)) if rms[i] < peak_val * 0.5), len(rms))
        tenth_idx = next((i for i in range(peak_idx, len(rms)) if rms[i] < peak_val * 0.1), len(rms))

        print(f"  Peak time:     {peak_time:.3f}s")
        print(f"  Decay to 50%:  {half_idx * 0.01:.3f}s  ({(half_idx - peak_idx) * 10}ms after peak)")
        print(f"  Decay to 10%:  {tenth_idx * 0.01:.3f}s  ({(tenth_idx - peak_idx) * 10}ms after peak)")

        # --- Spectral analysis at different time points ---
        for t_start, t_label in [(0.01, "Attack (10ms)"), (0.05, "Early (50ms)"),
                                  (0.2, "Body (200ms)"), (1.0, "Sustain (1s)")]:
            start = int(t_start * SAMPLE_RATE)
            end = min(start + int(0.05 * SAMPLE_RATE), n)  # 50ms window
            if start >= n:
                continue

            chunk = signal[start:end]
            if len(chunk) < 256:
                continue

            # Apply window
            chunk = chunk * np.hanning(len(chunk))

            # FFT
            fft = np.abs(np.fft.rfft(chunk))
            freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)

            # Find top 8 peaks
            # Smooth to avoid noise peaks
            from scipy.ndimage import uniform_filter1d
            smooth = uniform_filter1d(fft, 5)

            # Find local maxima
            peaks = []
            for i in range(2, len(smooth) - 2):
                if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > smooth[i-2] and smooth[i] > smooth[i+2]:
                    if freqs[i] > 30:  # skip DC
                        peaks.append((smooth[i], freqs[i]))

            peaks.sort(reverse=True)
            top = peaks[:8]

            print(f"\n  {t_label}:")
            if not top:
                print(f"    (no significant peaks)")
                continue

            max_amp = top[0][0]
            for amp, freq in sorted(top, key=lambda x: x[1]):
                rel_db = 20 * np.log10(amp / max_amp + 1e-10)
                bar = '#' * max(1, int((amp / max_amp) * 30))
                print(f"    {freq:7.1f} Hz  {rel_db:+5.1f} dB  {bar}")

        # --- Overall harmonic content ---
        # Use first 0.5s for harmonic analysis
        chunk = signal[:min(int(0.5 * SAMPLE_RATE), n)]
        chunk = chunk * np.hanning(len(chunk))
        fft = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / SAMPLE_RATE)

        # Energy in bands
        total = np.sum(fft**2)
        bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 20000)]
        print(f"\n  Energy distribution (first 0.5s):")
        for lo, hi in bands:
            mask = (freqs >= lo) & (freqs < hi)
            energy = np.sum(fft[mask]**2) / total * 100
            bar = '#' * max(1, int(energy / 2))
            print(f"    {lo:5d}-{hi:5d} Hz:  {energy:5.1f}%  {bar}")

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_deepcompare():
    """
    Deep multi-dimensional comparison of sampled Rhodes vs FM Rhodes.
    Analyzes: spectral centroid, spectral rolloff, MFCCs, attack transient shape,
    harmonic-to-noise ratio, spectral flux, and temporal envelope.
    """

    import numpy as np
    import librosa
    import os

    BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


    def load_audio(path):
        y, sr = librosa.load(path, sr=44100, mono=True)
        return y, sr


    def analyze(y, sr, label):
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        duration = len(y) / sr

        # --- 1. Temporal envelope shape ---
        print(f"\n  [ENVELOPE]")
        rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)

        peak_idx = np.argmax(rms)
        peak_time = times[peak_idx]
        peak_val = rms[peak_idx]

        # Attack shape: measure RMS at specific early time points
        for ms in [1, 5, 10, 20, 50]:
            t_idx = np.argmin(np.abs(times - ms/1000))
            if t_idx < len(rms):
                pct = rms[t_idx] / peak_val * 100
                print(f"    At {ms:3d}ms: {pct:5.1f}% of peak")

        # Decay shape
        for pct_target in [75, 50, 25, 10]:
            idx = next((i for i in range(peak_idx, len(rms)) if rms[i] < peak_val * pct_target/100), len(rms)-1)
            print(f"    Decay to {pct_target:2d}%: {times[idx]:.3f}s")

        # --- 2. Spectral centroid over time (brightness tracker) ---
        print(f"\n  [SPECTRAL CENTROID] (higher = brighter)")
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        c_times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=512)
        for t_sec, t_label in [(0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (0.5, "500ms"), (1.0, "1s"), (2.0, "2s")]:
            idx = np.argmin(np.abs(c_times - t_sec))
            if idx < len(centroid):
                print(f"    {t_label:>5s}: {centroid[idx]:7.0f} Hz")

        # --- 3. Spectral rolloff (where 85% of energy is below) ---
        print(f"\n  [SPECTRAL ROLLOFF 85%] (high-frequency extent)")
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, hop_length=512)[0]
        for t_sec, t_label in [(0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (1.0, "1s")]:
            idx = np.argmin(np.abs(c_times - t_sec))
            if idx < len(rolloff):
                print(f"    {t_label:>5s}: {rolloff[idx]:7.0f} Hz")

        # --- 4. Spectral flatness (noise-like vs tonal) ---
        # Higher = more noise-like, lower = more tonal/harmonic
        print(f"\n  [SPECTRAL FLATNESS] (0=pure tone, 1=white noise)")
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=512)[0]
        for t_sec, t_label in [(0.005, "5ms"), (0.01, "10ms"), (0.03, "30ms"), (0.1, "100ms"), (0.5, "500ms"), (1.0, "1s")]:
            idx = np.argmin(np.abs(c_times - t_sec))
            if idx < len(flatness):
                print(f"    {t_label:>5s}: {flatness[idx]:.6f}")

        # --- 5. MFCCs (timbral fingerprint) ---
        print(f"\n  [MFCC MEANS] (timbral shape, first 0.5s)")
        y_short = y[:int(0.5 * sr)]
        mfccs = librosa.feature.mfcc(y=y_short, sr=sr, n_mfcc=13)
        mfcc_means = mfccs.mean(axis=1)
        for i, val in enumerate(mfcc_means):
            bar = '#' * max(1, int(abs(val) / 5))
            sign = '+' if val >= 0 else '-'
            print(f"    MFCC {i:2d}: {val:+8.1f}  {bar}")

        # --- 6. Spectral bandwidth (spread of energy) ---
        print(f"\n  [SPECTRAL BANDWIDTH] (spread around centroid)")
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
        for t_sec, t_label in [(0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (1.0, "1s")]:
            idx = np.argmin(np.abs(c_times - t_sec))
            if idx < len(bandwidth):
                print(f"    {t_label:>5s}: {bandwidth[idx]:7.0f} Hz")

        # --- 7. Zero crossing rate (rough texture measure) ---
        print(f"\n  [ZERO CROSSING RATE] (higher = more high-freq content/noise)")
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=512, hop_length=256)[0]
        z_times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=256)
        for t_sec, t_label in [(0.005, "5ms"), (0.01, "10ms"), (0.05, "50ms"), (0.2, "200ms"), (1.0, "1s")]:
            idx = np.argmin(np.abs(z_times - t_sec))
            if idx < len(zcr):
                print(f"    {t_label:>5s}: {zcr[idx]:.4f}")

        # --- 8. Harmonic vs percussive energy ---
        print(f"\n  [HARMONIC/PERCUSSIVE SPLIT]")
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harm**2)
        perc_energy = np.sum(y_perc**2)
        total = harm_energy + perc_energy
        print(f"    Harmonic:   {harm_energy/total*100:.1f}%")
        print(f"    Percussive: {perc_energy/total*100:.1f}%")

        # Percussive energy in first 50ms vs total percussive
        cutoff = int(0.05 * sr)
        perc_attack = np.sum(y_perc[:cutoff]**2)
        print(f"    Percussive in first 50ms: {perc_attack/perc_energy*100:.1f}% of all percussive")

        # --- 9. Attack transient spectral snapshot (first 20ms) ---
        print(f"\n  [ATTACK SPECTRUM] (first 20ms, top peaks)")
        attack_y = y[:int(0.02 * sr)]
        attack_y = attack_y * np.hanning(len(attack_y))
        fft = np.abs(np.fft.rfft(attack_y))
        freqs = np.fft.rfftfreq(len(attack_y), 1.0/sr)
        # Find peaks
        peaks = []
        for i in range(3, len(fft)-3):
            if fft[i] > fft[i-1] and fft[i] > fft[i+1] and fft[i] > fft[i-2] and freqs[i] > 50:
                peaks.append((fft[i], freqs[i]))
        peaks.sort(reverse=True)
        if peaks:
            max_a = peaks[0][0]
            for amp, freq in sorted(peaks[:10], key=lambda x: x[1]):
                db = 20 * np.log10(amp/max_a + 1e-10)
                print(f"    {freq:7.0f} Hz  {db:+5.1f} dB")

        return {
            'mfcc_means': mfcc_means,
            'centroid_early': centroid[np.argmin(np.abs(c_times - 0.05))] if len(centroid) > 0 else 0,
            'centroid_late': centroid[np.argmin(np.abs(c_times - 1.0))] if len(centroid) > 0 else 0,
            'harm_pct': harm_energy/total*100,
            'perc_pct': perc_energy/total*100,
        }


    def compare_mfccs(stats_a, stats_b, label_a, label_b):
        print(f"\n{'='*70}")
        print(f"  MFCC DISTANCE: {label_a} vs {label_b}")
        print(f"{'='*70}")
        diff = stats_a['mfcc_means'] - stats_b['mfcc_means']
        total_dist = np.sqrt(np.sum(diff**2))
        print(f"  Euclidean distance: {total_dist:.1f}")
        print(f"  Per-coefficient difference:")
        for i, d in enumerate(diff):
            bar = '#' * max(1, int(abs(d) / 3))
            direction = '→ FM brighter' if d < 0 else '→ FM duller'
            if i == 0:
                direction = '→ FM louder' if d < 0 else '→ FM quieter'
            print(f"    MFCC {i:2d}: {d:+7.1f}  {bar}  {direction if abs(d) > 5 else ''}")

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})
def _build_analyzecomparison():
    """
    Comprehensive comparison of generated grand piano vs Salamander (Yamaha C5) samples.

    Analyzes across multiple dimensions:
    1. Spectral envelope & rolloff
    2. Inharmonicity (partial stretching)
    3. Decay rates (per-partial and overall)
    4. Attack transient character
    5. Spectral centroid over time (brightness evolution)
    6. Two-stage decay (prompt vs aftersound)
    7. Dynamic range / loudness profile
    8. Phantom partials / metallic content
    """

    import numpy as np
    import subprocess
    import wave
    import os
    import json
    import tempfile

    SAMPLE_RATE = 44100

    NOTE_NAMES = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']

    def name_to_midi(name):
        for i, n in enumerate(NOTE_NAMES):
            if name.startswith(n) and name[len(n):].lstrip('-').isdigit():
                octave = int(name[len(n):])
                return (octave + 1) * 12 + i
        return None

    def midi_to_freq(midi):
        return 440.0 * 2 ** ((midi - 69) / 12.0)

    def load_mp3(path):
        """Load MP3 as mono float array via ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', path, '-ac', '1', '-ar', str(SAMPLE_RATE),
                '-sample_fmt', 's16', tmp_path
            ], capture_output=True, check=True)
            with wave.open(tmp_path, 'r') as wf:
                frames = wf.readframes(wf.getnframes())
                signal = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
                signal /= 32768.0
            return signal
        finally:
            os.unlink(tmp_path)


    def spectral_analysis(signal, freq, label):
        """Compute spectral envelope, partial amplitudes, and inharmonicity."""
        # Use first 2 seconds for spectral content
        n = min(len(signal), int(2.0 * SAMPLE_RATE))
        seg = signal[:n]

        # Window and FFT
        win = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(seg * win))
        freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)

        # Find partials (peaks near expected harmonic frequencies)
        partials = []
        for h in range(1, 33):  # up to 32 partials
            expected = h * freq
            if expected > SAMPLE_RATE / 2 - 200:
                break
            # Search window: ±3% of expected frequency (accounts for inharmonicity)
            search_width = max(expected * 0.03, 5.0)
            mask = (freqs > expected - search_width) & (freqs < expected + search_width)
            if not np.any(mask):
                continue
            idx = np.where(mask)[0]
            peak_idx = idx[np.argmax(spectrum[idx])]
            peak_freq = freqs[peak_idx]
            peak_amp = spectrum[peak_idx]

            # Inharmonicity: deviation from ideal harmonic
            ideal = h * freq
            cents_deviation = 1200 * np.log2(peak_freq / ideal) if peak_freq > 0 and ideal > 0 else 0

            partials.append({
                'harmonic': h,
                'freq': float(peak_freq),
                'amp': float(peak_amp),
                'cents_sharp': float(cents_deviation),
            })

        # Normalize amplitudes to fundamental
        if partials and partials[0]['amp'] > 0:
            fund_amp = partials[0]['amp']
            for p in partials:
                p['amp_db'] = float(20 * np.log10(max(p['amp'] / fund_amp, 1e-10)))

        return partials


    def decay_analysis(signal, freq):
        """Analyze decay envelope: overall and per-partial."""
        duration = len(signal) / SAMPLE_RATE

        # Overall RMS envelope in 50ms windows
        win_size = int(0.05 * SAMPLE_RATE)
        hop = win_size // 2
        rms_env = []
        times = []
        for i in range(0, len(signal) - win_size, hop):
            rms = np.sqrt(np.mean(signal[i:i+win_size] ** 2))
            rms_env.append(float(rms))
            times.append(float((i + win_size // 2) / SAMPLE_RATE))

        rms_env = np.array(rms_env)
        times = np.array(times)

        # Find peak and measure decay from there
        peak_idx = np.argmax(rms_env)
        peak_time = times[peak_idx]
        peak_rms = rms_env[peak_idx]

        # Time to -6dB, -20dB, -40dB from peak
        decay_times = {}
        for db_drop in [6, 20, 40]:
            threshold = peak_rms * 10 ** (-db_drop / 20)
            below = np.where((rms_env[peak_idx:] < threshold))[0]
            if len(below) > 0:
                decay_times[f't_{db_drop}dB'] = float(times[peak_idx + below[0]] - peak_time)
            else:
                decay_times[f't_{db_drop}dB'] = float(duration - peak_time)

        # Two-stage decay detection: fit biexponential
        # Look at log envelope after peak
        post_peak = rms_env[peak_idx:]
        post_times = times[peak_idx:] - times[peak_idx]
        valid = post_peak > peak_rms * 0.001  # above -60dB
        if np.sum(valid) > 10:
            log_env = np.log(post_peak[valid] + 1e-10)
            t_valid = post_times[valid]

            # Early decay (first 0.5s after peak) vs late decay (1s-3s)
            early_mask = t_valid < 0.5
            late_mask = (t_valid > 1.0) & (t_valid < 3.0)

            early_rate = None
            late_rate = None
            if np.sum(early_mask) > 3:
                p = np.polyfit(t_valid[early_mask], log_env[early_mask], 1)
                early_rate = float(-p[0])  # decay rate in nepers/s
            if np.sum(late_mask) > 3:
                p = np.polyfit(t_valid[late_mask], log_env[late_mask], 1)
                late_rate = float(-p[0])

            decay_times['early_rate'] = early_rate
            decay_times['late_rate'] = late_rate
            if early_rate and late_rate and late_rate > 0:
                decay_times['prompt_aftersound_ratio'] = float(early_rate / late_rate)

        decay_times['peak_time'] = float(peak_time)
        return decay_times


    def attack_analysis(signal):
        """Analyze attack transient: rise time, peak time, noise content."""
        # First 100ms
        n = min(len(signal), int(0.1 * SAMPLE_RATE))
        seg = signal[:n]

        # Envelope via Hilbert-like (rectify + smooth)
        rectified = np.abs(seg)
        smooth_size = int(0.002 * SAMPLE_RATE)  # 2ms smoothing
        kernel = np.ones(smooth_size) / smooth_size
        env = np.convolve(rectified, kernel, mode='same')

        peak_idx = np.argmax(env)
        peak_time = peak_idx / SAMPLE_RATE

        # Rise time: 10% to 90% of peak
        peak_val = env[peak_idx]
        t10 = np.where(env[:peak_idx+1] > 0.1 * peak_val)[0]
        t90 = np.where(env[:peak_idx+1] > 0.9 * peak_val)[0]
        rise_time = None
        if len(t10) > 0 and len(t90) > 0:
            rise_time = float((t90[0] - t10[0]) / SAMPLE_RATE)

        # Spectral centroid of attack (first 30ms) — indicates noise/brightness
        attack_n = min(len(signal), int(0.03 * SAMPLE_RATE))
        attack_seg = signal[:attack_n] * np.hanning(attack_n)
        spec = np.abs(np.fft.rfft(attack_seg))
        freqs = np.fft.rfftfreq(attack_n, 1.0 / SAMPLE_RATE)
        spec_sum = np.sum(spec)
        if spec_sum > 0:
            attack_centroid = float(np.sum(freqs * spec) / spec_sum)
        else:
            attack_centroid = 0.0

        # High-frequency energy ratio in attack (noise indicator)
        hf_mask = freqs > 4000
        hf_ratio = float(np.sum(spec[hf_mask] ** 2) / (np.sum(spec ** 2) + 1e-10))

        return {
            'peak_time_ms': float(peak_time * 1000),
            'rise_time_ms': float(rise_time * 1000) if rise_time else None,
            'attack_centroid_hz': attack_centroid,
            'hf_energy_ratio': hf_ratio,
        }


    def brightness_evolution(signal, freq):
        """Track spectral centroid over time — how brightness evolves during the note."""
        win_size = int(0.1 * SAMPLE_RATE)  # 100ms windows
        hop = win_size // 2
        centroids = []
        times = []

        for i in range(0, len(signal) - win_size, hop):
            seg = signal[i:i+win_size] * np.hanning(win_size)
            spec = np.abs(np.fft.rfft(seg))
            freqs = np.fft.rfftfreq(win_size, 1.0 / SAMPLE_RATE)
            spec_sum = np.sum(spec)
            if spec_sum > 1e-10:
                centroid = float(np.sum(freqs * spec) / spec_sum)
            else:
                centroid = 0.0
            centroids.append(centroid)
            times.append(float((i + win_size // 2) / SAMPLE_RATE))

        centroids = np.array(centroids)
        times = np.array(times)

        # Summarize: centroid at 0.1s, 0.5s, 1s, 2s, 4s
        summary = {}
        for t_target in [0.1, 0.5, 1.0, 2.0, 4.0]:
            idx = np.argmin(np.abs(times - t_target))
            if idx < len(centroids):
                summary[f'centroid_{t_target}s'] = float(centroids[idx])

        # Brightness decay: how fast does centroid drop?
        if len(centroids) > 5:
            peak_c_idx = np.argmax(centroids[:min(10, len(centroids))])
            half_centroid = centroids[peak_c_idx] * 0.5
            below = np.where(centroids[peak_c_idx:] < half_centroid)[0]
            if len(below) > 0:
                summary['brightness_halflife_s'] = float(times[peak_c_idx + below[0]] - times[peak_c_idx])

        return summary


    def phantom_partial_analysis(signal, freq):
        """Check for phantom partials (sum-frequency components) in bass notes."""
        if freq > 350:  # only relevant for bass/mid
            return {'relevant': False}

        n = min(len(signal), int(1.0 * SAMPLE_RATE))
        seg = signal[:n]
        win = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(seg * win))
        freqs_arr = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)

        # Look for energy at h2+h1 = 3f, h3+h1 = 4f (but these overlap with harmonics)
        # Better: look at h2+h3 = 5f region for non-harmonic bumps
        # Actually, phantom partials are at sum frequencies which may not be exactly harmonic
        # due to inharmonicity. Check energy between harmonics.

        inter_harmonic_energy = 0.0
        harmonic_energy = 0.0

        for h in range(2, 16):
            expected = h * freq
            if expected > SAMPLE_RATE / 2 - 500:
                break
            # Harmonic region: ±1%
            h_mask = (freqs_arr > expected * 0.99) & (freqs_arr < expected * 1.01)
            harmonic_energy += np.sum(spectrum[h_mask] ** 2)

            # Inter-harmonic region: between h and h+1
            mid = (h + 0.5) * freq
            ih_mask = (freqs_arr > mid * 0.97) & (freqs_arr < mid * 1.03)
            inter_harmonic_energy += np.sum(spectrum[ih_mask] ** 2)

        ratio = float(inter_harmonic_energy / (harmonic_energy + 1e-10))

        return {
            'relevant': True,
            'inter_harmonic_ratio_db': float(10 * np.log10(ratio + 1e-10)),
        }


    def analyze_note(grand_path, sala_path, note_name):
        """Full comparison for one note."""
        midi = name_to_midi(note_name)
        freq = midi_to_freq(midi)

        grand = load_mp3(grand_path)
        sala = load_mp3(sala_path)

        result = {
            'note': note_name,
            'midi': midi,
            'freq': round(freq, 2),
        }

        # 1. Spectral analysis
        g_partials = spectral_analysis(grand, freq, 'grand')
        s_partials = spectral_analysis(sala, freq, 'sala')

        # Compare partial amplitudes
        partial_diffs = []
        for gp in g_partials:
            sp = next((p for p in s_partials if p['harmonic'] == gp['harmonic']), None)
            if sp and 'amp_db' in gp and 'amp_db' in sp:
                partial_diffs.append({
                    'h': gp['harmonic'],
                    'grand_db': round(gp['amp_db'], 1),
                    'sala_db': round(sp['amp_db'], 1),
                    'diff_db': round(gp['amp_db'] - sp['amp_db'], 1),
                    'grand_cents': round(gp['cents_sharp'], 2),
                    'sala_cents': round(sp['cents_sharp'], 2),
                })
        result['partial_comparison'] = partial_diffs

        # Spectral rolloff summary
        if len(g_partials) > 4 and len(s_partials) > 4:
            g_amps = [p.get('amp_db', -60) for p in g_partials[:16]]
            s_amps = [p.get('amp_db', -60) for p in s_partials[:16]]
            # Linear fit to get rolloff slope
            g_slope = np.polyfit(range(len(g_amps)), g_amps, 1)[0] if len(g_amps) > 2 else 0
            s_slope = np.polyfit(range(len(s_amps)), s_amps, 1)[0] if len(s_amps) > 2 else 0
            result['spectral_rolloff'] = {
                'grand_slope_db_per_partial': round(float(g_slope), 2),
                'sala_slope_db_per_partial': round(float(s_slope), 2),
            }

        # 2. Decay analysis
        result['decay_grand'] = decay_analysis(grand, freq)
        result['decay_sala'] = decay_analysis(sala, freq)

        # 3. Attack analysis
        result['attack_grand'] = attack_analysis(grand)
        result['attack_sala'] = attack_analysis(sala)

        # 4. Brightness evolution
        result['brightness_grand'] = brightness_evolution(grand, freq)
        result['brightness_sala'] = brightness_evolution(sala, freq)

        # 5. Phantom partials
        result['phantoms_grand'] = phantom_partial_analysis(grand, freq)
        result['phantoms_sala'] = phantom_partial_analysis(sala, freq)

        return result


    def print_summary(results):
        """Print a readable summary of the comparison."""
        print("=" * 80)
        print("GRAND PIANO vs SALAMANDER (Yamaha C5) — COMPREHENSIVE COMPARISON")
        print("=" * 80)

        # Group findings
        print("\n1. SPECTRAL ROLLOFF (dB/partial, first 16 harmonics)")
        print(f"   {'Note':<6} {'Grand':>8} {'Sala':>8} {'Diff':>8}")
        print(f"   {'----':<6} {'-----':>8} {'----':>8} {'----':>8}")
        for r in results:
            if 'spectral_rolloff' in r:
                sr = r['spectral_rolloff']
                diff = sr['grand_slope_db_per_partial'] - sr['sala_slope_db_per_partial']
                print(f"   {r['note']:<6} {sr['grand_slope_db_per_partial']:>7.2f} {sr['sala_slope_db_per_partial']:>7.2f} {diff:>+7.2f}")

        print("\n2. INHARMONICITY (cents sharp from ideal, selected partials)")
        for r in results:
            if not r['partial_comparison']:
                continue
            print(f"\n   {r['note']} ({r['freq']} Hz):")
            print(f"   {'H':>4} {'Grand ¢':>9} {'Sala ¢':>9}")
            for p in r['partial_comparison']:
                if p['h'] in [1, 2, 4, 8, 16]:
                    print(f"   {p['h']:>4} {p['grand_cents']:>+8.2f} {p['sala_cents']:>+8.2f}")

        print("\n3. DECAY TIMES")
        print(f"   {'Note':<6} {'Metric':<22} {'Grand':>8} {'Sala':>8}")
        print(f"   {'----':<6} {'------':<22} {'-----':>8} {'----':>8}")
        for r in results:
            dg, ds = r['decay_grand'], r['decay_sala']
            for key in ['t_6dB', 't_20dB', 't_40dB']:
                gv = dg.get(key, None)
                sv = ds.get(key, None)
                gstr = f"{gv:.2f}s" if gv else "N/A"
                sstr = f"{sv:.2f}s" if sv else "N/A"
                print(f"   {r['note']:<6} {key:<22} {gstr:>8} {sstr:>8}")
            # Prompt/aftersound ratio
            gr = dg.get('prompt_aftersound_ratio')
            sr_val = ds.get('prompt_aftersound_ratio')
            if gr and sr_val:
                print(f"   {r['note']:<6} {'prompt/after ratio':<22} {gr:>7.1f}x {sr_val:>7.1f}x")
            print()

        print("\n4. ATTACK CHARACTER")
        print(f"   {'Note':<6} {'Metric':<22} {'Grand':>10} {'Sala':>10}")
        print(f"   {'----':<6} {'------':<22} {'-----':>10} {'----':>10}")
        for r in results:
            ag, a_s = r['attack_grand'], r['attack_sala']
            print(f"   {r['note']:<6} {'peak time':<22} {ag['peak_time_ms']:>8.1f}ms {a_s['peak_time_ms']:>8.1f}ms")
            rt_g = f"{ag['rise_time_ms']:.1f}ms" if ag['rise_time_ms'] else "N/A"
            rt_s = f"{a_s['rise_time_ms']:.1f}ms" if a_s['rise_time_ms'] else "N/A"
            print(f"   {r['note']:<6} {'rise time':<22} {rt_g:>10} {rt_s:>10}")
            print(f"   {r['note']:<6} {'attack centroid':<22} {ag['attack_centroid_hz']:>8.0f}Hz {a_s['attack_centroid_hz']:>8.0f}Hz")
            print(f"   {r['note']:<6} {'HF energy ratio':<22} {ag['hf_energy_ratio']:>9.4f} {a_s['hf_energy_ratio']:>9.4f}")
            print()

        print("\n5. BRIGHTNESS EVOLUTION (spectral centroid over time)")
        print(f"   {'Note':<6} {'Time':<8} {'Grand':>8} {'Sala':>8} {'Diff':>8}")
        print(f"   {'----':<6} {'----':<8} {'-----':>8} {'----':>8} {'----':>8}")
        for r in results:
            bg, bs = r['brightness_grand'], r['brightness_sala']
            for t in ['0.1', '0.5', '1.0', '2.0']:
                gk = f'centroid_{t}s'
                if gk in bg and gk in bs:
                    diff = bg[gk] - bs[gk]
                    print(f"   {r['note']:<6} {t+'s':<8} {bg[gk]:>7.0f} {bs[gk]:>7.0f} {diff:>+7.0f}")
            bhl_g = bg.get('brightness_halflife_s')
            bhl_s = bs.get('brightness_halflife_s')
            if bhl_g and bhl_s:
                print(f"   {r['note']:<6} {'halflife':<8} {bhl_g:>6.2f}s {bhl_s:>6.2f}s")
            print()

        print("\n6. PHANTOM PARTIALS (inter-harmonic energy, bass notes only)")
        print(f"   {'Note':<6} {'Grand dB':>10} {'Sala dB':>10}")
        for r in results:
            pg, ps = r['phantoms_grand'], r['phantoms_sala']
            if pg.get('relevant'):
                gdb = f"{pg['inter_harmonic_ratio_db']:.1f}" if 'inter_harmonic_ratio_db' in pg else "N/A"
                sdb = f"{ps['inter_harmonic_ratio_db']:.1f}" if 'inter_harmonic_ratio_db' in ps else "N/A"
                print(f"   {r['note']:<6} {gdb:>10} {sdb:>10}")

        # Per-partial amplitude comparison for a few representative notes
        print("\n7. DETAILED PARTIAL AMPLITUDES (dB relative to fundamental)")
        for r in results:
            if r['note'] in ['C2', 'C4', 'A4', 'C6']:
                print(f"\n   {r['note']} ({r['freq']} Hz):")
                print(f"   {'H':>4} {'Grand dB':>10} {'Sala dB':>10} {'Diff':>8}")
                for p in r['partial_comparison'][:16]:
                    print(f"   {p['h']:>4} {p['grand_db']:>9.1f} {p['sala_db']:>9.1f} {p['diff_db']:>+7.1f}")

        print("\n" + "=" * 80)
        print("OVERALL OBSERVATIONS")
        print("=" * 80)

        # Aggregate stats
        all_rolloff_diffs = []
        all_decay_ratios = {'6dB': [], '20dB': [], '40dB': []}
        all_attack_centroid_diffs = []
        all_brightness_diffs = []

        for r in results:
            if 'spectral_rolloff' in r:
                sr = r['spectral_rolloff']
                all_rolloff_diffs.append(sr['grand_slope_db_per_partial'] - sr['sala_slope_db_per_partial'])

            dg, ds = r['decay_grand'], r['decay_sala']
            for db in ['6', '20', '40']:
                k = f't_{db}dB'
                if dg.get(k) and ds.get(k) and ds[k] > 0:
                    all_decay_ratios[f'{db}dB'].append(dg[k] / ds[k])

            ag, a_s = r['attack_grand'], r['attack_sala']
            all_attack_centroid_diffs.append(ag['attack_centroid_hz'] - a_s['attack_centroid_hz'])

            bg, bs = r['brightness_grand'], r['brightness_sala']
            if 'centroid_0.5s' in bg and 'centroid_0.5s' in bs:
                all_brightness_diffs.append(bg['centroid_0.5s'] - bs['centroid_0.5s'])

        if all_rolloff_diffs:
            mean_rd = np.mean(all_rolloff_diffs)
            print(f"\n  Spectral rolloff: Grand is {'steeper' if mean_rd < 0 else 'gentler'} by {abs(mean_rd):.2f} dB/partial on average")

        for db, ratios in all_decay_ratios.items():
            if ratios:
                mean_r = np.mean(ratios)
                print(f"  Decay to -{db}: Grand is {mean_r:.2f}x Salamander ({('shorter' if mean_r < 1 else 'longer')})")

        if all_attack_centroid_diffs:
            mean_ac = np.mean(all_attack_centroid_diffs)
            print(f"  Attack brightness: Grand is {abs(mean_ac):.0f} Hz {'higher' if mean_ac > 0 else 'lower'} centroid on average")

        if all_brightness_diffs:
            mean_bd = np.mean(all_brightness_diffs)
            print(f"  Sustain brightness (0.5s): Grand is {abs(mean_bd):.0f} Hz {'higher' if mean_bd > 0 else 'lower'} centroid on average")

        print()

    return types.SimpleNamespace(**{k: v for k, v in locals().items() if not k.startswith('__')})

# ═══════════════════════════════════════════════════════════════════════
# cmd_* — one function per subcommand
# ═══════════════════════════════════════════════════════════════════════

def _write_layered(ns, midi, name, velocity_layers, out_dir, generate_fn):
    """Shared velocity-layer generation loop used by generate-grand/-rhodes/-prism."""
    np = ns.np
    layer_signals, layer_peaks = [], []
    for vel in velocity_layers:
        signal, raw_peak = generate_fn(midi, velocity=vel)
        signal = signal / 0.85 * raw_peak
        layer_signals.append(signal)
        layer_peaks.append(np.max(np.abs(signal)))
    max_peak = max(layer_peaks) if max(layer_peaks) > 0 else 1.0
    for v_idx, signal in enumerate(layer_signals):
        signal = signal / max_peak * 0.85
        v_dir = os.path.join(out_dir, f'v{v_idx + 1}')
        os.makedirs(v_dir, exist_ok=True)
        wav_path = os.path.join(v_dir, f'{name}.wav')
        mp3_path = os.path.join(v_dir, f'{name}.mp3')
        ns.write_wav(wav_path, signal)
        ns.wav_to_mp3(wav_path, mp3_path)
        if os.path.exists(mp3_path):
            os.remove(wav_path)
        print(f" v{v_idx + 1}", end='', flush=True)
    print()


def cmd_generate_grand(args):
    g = grand()
    out_dir = os.path.join(SCRIPT_DIR, '..', 'audio',
                            'grand-piano-dry' if args.no_ir else 'grand-piano')
    os.makedirs(out_dir, exist_ok=True)

    if args.velocity_layers:
        print(f"Generating {len(g.NOTES)} × {len(g.VELOCITY_LAYERS)} velocity layers "
              f"= {len(g.NOTES) * len(g.VELOCITY_LAYERS)} Grand Piano samples...")
        print(f"  Velocities: {g.VELOCITY_LAYERS}")
    else:
        print(f"Generating {len(g.NOTES)} Grand Piano samples...")
    print("  Model: Physics-based modal synthesis (Bensa/Chaigne/Weinreich)")
    print()

    use_ir = not args.no_ir
    for midi, name in g.NOTES:
        freq = g.midi_to_freq(midi)
        if args.velocity_layers:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz", end='', flush=True)
            _write_layered(g, midi, name, g.VELOCITY_LAYERS, out_dir,
                            lambda m, velocity: g.generate_grand_piano_note(m, velocity=velocity, use_ir=use_ir))
        else:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz")
            signal, _ = g.generate_grand_piano_note(midi, use_ir=use_ir)
            wav_path = os.path.join(out_dir, f'{name}.wav')
            mp3_path = os.path.join(out_dir, f'{name}.mp3')
            g.write_wav(wav_path, signal)
            g.wav_to_mp3(wav_path, mp3_path)
            if os.path.exists(mp3_path):
                os.remove(wav_path)
            else:
                print("    Warning: ffmpeg conversion failed, keeping WAV")

    print(f"\nDone! Samples written to {out_dir}/")


def cmd_generate_rhodes(args):
    r = rhodes()
    out_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'rhodes-fm')
    os.makedirs(out_dir, exist_ok=True)

    if args.velocity_layers:
        print(f"Generating {len(r.NOTES)} × {len(r.VELOCITY_LAYERS)} velocity layers "
              f"= {len(r.NOTES) * len(r.VELOCITY_LAYERS)} Rhodes FM samples...")
        print(f"  Velocities: {r.VELOCITY_LAYERS}")
    else:
        print(f"Generating {len(r.NOTES)} Rhodes FM samples...")
    print()

    for midi, name in r.NOTES:
        freq = r.midi_to_freq(midi)
        if args.velocity_layers:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz", end='', flush=True)
            _write_layered(r, midi, name, r.VELOCITY_LAYERS, out_dir, r.generate_rhodes_note)
        else:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz")
            signal, _ = r.generate_rhodes_note(midi)
            wav_path = os.path.join(out_dir, f'{name}.wav')
            mp3_path = os.path.join(out_dir, f'{name}.mp3')
            r.write_wav(wav_path, signal)
            r.wav_to_mp3(wav_path, mp3_path)
            if os.path.exists(mp3_path):
                os.remove(wav_path)
            else:
                print("    Warning: ffmpeg conversion failed, keeping WAV")

    print(f"\nDone! Samples written to {out_dir}/")


def cmd_generate_prism(args):
    p = prism()
    out_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'prism')
    os.makedirs(out_dir, exist_ok=True)

    if args.velocity_layers:
        print(f"Generating {len(p.NOTES)} × {len(p.VELOCITY_LAYERS)} velocity layers "
              f"= {len(p.NOTES) * len(p.VELOCITY_LAYERS)} Prism Keys samples...")
        print(f"  Velocities: {p.VELOCITY_LAYERS}")
    else:
        print(f"Generating {len(p.NOTES)} Prism Keys samples...")
    print()

    for midi, name in p.NOTES:
        freq = p.midi_to_freq(midi)
        if args.velocity_layers:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz", end='', flush=True)
            _write_layered(p, midi, name, p.VELOCITY_LAYERS, out_dir, p.generate_prism_note)
        else:
            print(f"  {name} (MIDI {midi}) — {freq:.1f} Hz")
            signal, _ = p.generate_prism_note(midi)
            wav_path = os.path.join(out_dir, f'{name}.wav')
            mp3_path = os.path.join(out_dir, f'{name}.mp3')
            p.write_wav(wav_path, signal)
            p.wav_to_mp3(wav_path, mp3_path)
            if os.path.exists(mp3_path):
                os.remove(wav_path)
            else:
                print("    Warning: ffmpeg conversion failed, keeping WAV")

    print(f"\nDone! Samples written to {out_dir}/")


def cmd_generate_ddsp(args):
    d = ddsp()
    if args.generate_only:
        if not os.path.exists(d.MODEL_PATH):
            print(f"ERROR: No saved model at {d.MODEL_PATH}")
            sys.exit(1)
        checkpoint = d.torch.load(d.MODEL_PATH, map_location=d.DEVICE, weights_only=True)
        model = d.PianoParamNet().to(d.DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        print(f"Loaded model (loss={checkpoint['best_loss']:.4f}, {checkpoint['epochs']} epochs)")
    else:
        model = d.train(epochs=args.epochs)
        if model is None:
            sys.exit(1)
    d.compare_with_targets(model)
    d.generate_samples(model)


def cmd_render_midi(args):
    """Render a .mid file with one of the synthesis engines: every note_on
    triggers a full natural-decay note (same as the engines already produce),
    mixed into a single buffer at the note's onset time — there's no
    note-off/damper modeling, matching how these engines work everywhere
    else in the tool."""
    import mido

    instruments = {
        'grand': (grand, 'generate_grand_piano_note'),
        'rhodes': (rhodes, 'generate_rhodes_note'),
        'prism': (prism, 'generate_prism_note'),
    }
    ns_fn, gen_attr = instruments[args.instrument]

    if not os.path.exists(args.midi_path):
        print(f"ERROR: {args.midi_path} not found")
        sys.exit(1)

    mf = mido.MidiFile(args.midi_path)
    print(f"Loaded {args.midi_path}: {len(mf.tracks)} track(s), ~{mf.length:.1f}s at file tempo")

    events = []  # (onset_seconds, midi_note, velocity_fraction)
    abs_time = 0.0
    for msg in mf:
        abs_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0 and msg.channel != 9:
            events.append((abs_time, msg.note, msg.velocity / 127.0))

    if not events:
        print("No pitched notes found (channel 10 / drums is skipped).")
        return
    events.sort(key=lambda e: e[0])

    ns = ns_fn()
    np = ns.np
    generate_note = getattr(ns, gen_attr)
    SAMPLE_RATE = 44100

    print(f"Rendering {len(events)} notes with '{args.instrument}'...")
    rendered = []
    max_end_sample = 0
    skipped = 0
    for i, (onset, note, vel) in enumerate(events):
        vel = max(0.05, min(1.0, vel * args.velocity_scale))
        try:
            signal, _ = generate_note(note, velocity=vel)
        except Exception as e:
            skipped += 1
            continue
        offset_sample = int(onset / args.speed * SAMPLE_RATE)
        rendered.append((offset_sample, signal))
        max_end_sample = max(max_end_sample, offset_sample + len(signal))
        if (i + 1) % 25 == 0 or i == len(events) - 1:
            print(f"\r  {i + 1}/{len(events)} notes synthesized", end='', flush=True)
    print()
    if skipped:
        print(f"  ({skipped} notes failed to synthesize and were skipped)")
    if not rendered:
        print("Nothing rendered.")
        return

    master = np.zeros(max_end_sample, dtype=np.float64)
    for offset_sample, signal in rendered:
        master[offset_sample:offset_sample + len(signal)] += signal
    peak = np.max(np.abs(master))
    if peak > 0:
        master = master / peak * 0.9

    stem = args.output or os.path.splitext(os.path.basename(args.midi_path))[0] + f'_{args.instrument}'
    if not stem.endswith('.mp3'):
        stem += '.mp3'
    out_path = stem if os.path.isabs(stem) else os.path.join(SCRIPT_DIR, stem)
    wav_path = out_path[:-4] + '.wav'
    ns.write_wav(wav_path, master)
    ns.wav_to_mp3(wav_path, out_path)
    if os.path.exists(out_path):
        os.remove(wav_path)
        print(f"\nWrote {out_path} ({max_end_sample / SAMPLE_RATE:.1f}s)")
    else:
        print(f"\nffmpeg conversion failed, kept WAV at {wav_path}")


def cmd_extract_soundboard_ir(args):
    import numpy as np
    g = grand()
    SAMPLE_RATE = 44100
    DURATION = 4.0
    N_FFT = 8192
    IR_LENGTH = 2048
    REFERENCE_NOTES = {'C2': 36, 'A2': 45, 'C3': 48, 'A3': 57,
                        'C4': 60, 'A4': 69, 'C5': 72, 'A5': 81}
    ref_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'piano')

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    idx_1k = np.argmin(np.abs(freqs - 1000))
    n_samples = int(SAMPLE_RATE * DURATION)

    midi_points, raw_tfs = [], []
    for name, midi in sorted(REFERENCE_NOTES.items(), key=lambda x: x[1]):
        path = os.path.join(ref_dir, f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  {name}: not found, skipping")
            continue
        print(f"  {name} (MIDI {midi})...")

        recorded = _load_mp3(path, SAMPLE_RATE)
        if len(recorded) > n_samples:
            recorded = recorded[:n_samples]
        elif len(recorded) < n_samples:
            recorded = np.pad(recorded, (0, n_samples - len(recorded)))
        rec_peak = np.max(np.abs(recorded))
        if rec_peak > 0:
            recorded = recorded / rec_peak * 0.85

        # Dry synthesis (use_ir=False) — was a monkeypatch on a module global
        # in the original scripts; now an explicit parameter.
        synth, _ = g.generate_grand_piano_note(midi, duration=DURATION, use_ir=False)
        if len(synth) > n_samples:
            synth = synth[:n_samples]
        elif len(synth) < n_samples:
            synth = np.pad(synth, (0, n_samples - len(synth)))

        rec_spec = _spectral_envelope(recorded, N_FFT)
        syn_spec = _spectral_envelope(synth, N_FFT)
        syn_floor = np.max(syn_spec) * 0.001
        H = rec_spec / np.maximum(syn_spec, syn_floor)
        H = _smooth_spectrum(H, window_size=24)
        H /= H[idx_1k]

        midi_points.append(midi)
        raw_tfs.append(H)
        print(f"    H range: {H.min():.3f} - {H.max():.3f}")

    if not raw_tfs:
        print("No reference samples found!")
        return

    midi_points = np.array(midi_points)
    raw_tfs = np.array(raw_tfs)

    avg_log_H = np.mean(np.log(raw_tfs + 1e-10), axis=0)
    log_f = np.log(freqs + 1.0)
    mask = (freqs >= 50) & (freqs <= 8000)
    coeffs = np.polyfit(log_f[mask], avg_log_H[mask], 1)
    trend = np.exp(np.polyval(coeffs, log_f))
    trend /= trend[idx_1k]
    print(f"\nShared spectral tilt: slope = {coeffs[0]:.2f}")

    detrended_tfs = np.zeros_like(raw_tfs)
    for i in range(len(midi_points)):
        detrended_tfs[i] = raw_tfs[i] / trend

    for i_bin, f in enumerate(freqs):
        if f < 2000:
            max_db = 12.0
        elif f < 3000:
            max_db = 12.0 - 6.0 * (f - 2000) / 1000
        else:
            max_db = 6.0
        max_boost = 10 ** (max_db / 20)
        detrended_tfs[:, i_bin] = np.clip(detrended_tfs[:, i_bin], 1.0 / max_boost, max_boost)

    for i_bin, f in enumerate(freqs):
        if f > 10000:
            t = min((f - 10000) / 5000, 1.0)
            detrended_tfs[:, i_bin] = 1.0 + (detrended_tfs[:, i_bin] - 1.0) * (1.0 - t)

    for i, midi in enumerate(midi_points):
        name = [n for n, m in REFERENCE_NOTES.items() if m == midi][0]
        print(f"\n  {name} (MIDI {midi}) detrended:")
        for f_check in [200, 500, 1000, 2000, 4000, 8000]:
            idx = np.argmin(np.abs(freqs - f_check))
            db = 20 * np.log10(detrended_tfs[i, idx])
            print(f"    {f_check:5d} Hz: {db:+.1f} dB")

    out_path = os.path.join(SCRIPT_DIR, 'soundboard_tf.npz')
    np.savez(out_path, midi_points=midi_points, freqs=freqs,
              transfer_functions=detrended_tfs, ir_length=IR_LENGTH)
    print(f"\nSaved {len(midi_points)} per-note transfer functions to {out_path}")


def cmd_extract_rhodes_tf(args):
    import numpy as np
    r = rhodes()
    SAMPLE_RATE = 44100
    DURATION = 4.0
    N_FFT = 8192
    IR_LENGTH = 1024
    REFERENCE_NOTES = {'A2': 45, 'B1': 35, 'B3': 59, 'B4': 71, 'D3': 50, 'D4': 62,
                        'D6': 86, 'E2': 40, 'E5': 76, 'F4': 65, 'G3': 55, 'A5': 81}
    ref_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'rhodes')

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / SAMPLE_RATE)
    idx_1k = np.argmin(np.abs(freqs - 1000))
    n_samples = int(SAMPLE_RATE * DURATION)

    midi_points, raw_tfs = [], []
    for name, midi in sorted(REFERENCE_NOTES.items(), key=lambda x: x[1]):
        path = os.path.join(ref_dir, f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  {name}: not found, skipping")
            continue
        print(f"  {name} (MIDI {midi})...")

        recorded = _load_mp3(path, SAMPLE_RATE)
        if len(recorded) > n_samples:
            recorded = recorded[:n_samples]
        elif len(recorded) < n_samples:
            recorded = np.pad(recorded, (0, n_samples - len(recorded)))
        rec_peak = np.max(np.abs(recorded))
        if rec_peak > 0:
            recorded = recorded / rec_peak * 0.85

        synth = r.generate_rhodes_note(midi, duration=DURATION)[0]
        if len(synth) > n_samples:
            synth = synth[:n_samples]
        elif len(synth) < n_samples:
            synth = np.pad(synth, (0, n_samples - len(synth)))

        rec_spec = _spectral_envelope(recorded, N_FFT)
        syn_spec = _spectral_envelope(synth, N_FFT)
        syn_floor = np.max(syn_spec) * 0.001
        H = rec_spec / np.maximum(syn_spec, syn_floor)
        H = _smooth_spectrum(H, window_size=24)
        H /= H[idx_1k]

        midi_points.append(midi)
        raw_tfs.append(H)
        print(f"    H range: {H.min():.3f} - {H.max():.3f}")

    if not raw_tfs:
        print("No reference samples found!")
        return

    midi_points = np.array(midi_points)
    raw_tfs = np.array(raw_tfs)

    avg_log_H = np.mean(np.log(raw_tfs + 1e-10), axis=0)
    log_f = np.log(freqs + 1.0)
    mask = (freqs >= 50) & (freqs <= 8000)
    coeffs = np.polyfit(log_f[mask], avg_log_H[mask], 1)
    trend = np.exp(np.polyval(coeffs, log_f))
    trend /= trend[idx_1k]
    print(f"\nShared spectral tilt: slope = {coeffs[0]:.2f}")

    detrended_tfs = np.zeros_like(raw_tfs)
    for i in range(len(midi_points)):
        detrended_tfs[i] = raw_tfs[i] / trend

    for i_bin, f in enumerate(freqs):
        if f < 1000:
            max_db = 8.0
        elif f < 3000:
            max_db = 8.0 - 5.0 * (f - 1000) / 2000
        else:
            max_db = 3.0
        max_boost = 10 ** (max_db / 20)
        detrended_tfs[:, i_bin] = np.clip(detrended_tfs[:, i_bin], 1.0 / max_boost, max_boost)

    for i_bin, f in enumerate(freqs):
        if f > 6000:
            t = min((f - 6000) / 3000, 1.0)
            detrended_tfs[:, i_bin] = 1.0 + (detrended_tfs[:, i_bin] - 1.0) * (1.0 - t)

    for i, midi in enumerate(midi_points):
        name = [n for n, m in REFERENCE_NOTES.items() if m == midi][0]
        print(f"\n  {name} (MIDI {midi}) detrended:")
        for f_check in [200, 500, 1000, 2000, 4000, 8000]:
            idx = np.argmin(np.abs(freqs - f_check))
            db = 20 * np.log10(detrended_tfs[i, idx])
            print(f"    {f_check:5d} Hz: {db:+.1f} dB")

    out_path = os.path.join(SCRIPT_DIR, 'rhodes_tf.npz')
    np.savez(out_path, midi_points=midi_points, freqs=freqs,
              transfer_functions=detrended_tfs, ir_length=IR_LENGTH)
    print(f"\nSaved {len(midi_points)} per-note transfer functions to {out_path}")


def cmd_optimize_phases(args):
    """Gradient-descent phase optimization. Writes the result straight into
    piano_synth_config.json (grand_phase_table) instead of printing a
    paste-ready block — generate-grand reads it back on its next run."""
    o = optphases()
    np, torch, time = o.np, o.torch, o.time

    ref_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'piano')
    target_notes = {'C2': 36, 'A2': 45, 'C3': 48, 'C4': 60, 'A4': 69, 'C5': 72, 'A5': 81}

    print(f"Device: {o.DEVICE}")
    print(f"Loading {len(target_notes)} reference samples...")
    refs = {}
    for name, midi in target_notes.items():
        path = os.path.join(ref_dir, f'{name}.mp3')
        if os.path.exists(path):
            refs[midi] = o.load_reference(path)
            print(f"  {name}: loaded")

    if not refs:
        print("No reference samples found in ../audio/piano — nothing to optimize against.")
        return

    current_phases = np.array(
        np.random.RandomState(6454).uniform(0, 2 * np.pi, (3, 64)), dtype=np.float32)
    phase_table = torch.tensor(current_phases, device=o.DEVICE, requires_grad=True)

    optimizer = torch.optim.Adam([phase_table], lr=0.05)
    n_iters = args.iters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iters)

    print("\nOptimizing 192 phase values (mel-scale STFT loss)...")
    best_loss = float('inf')
    best_phases = current_phases.copy()
    t_start = time.time()

    for iteration in range(n_iters):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=o.DEVICE)
        for midi, ref in refs.items():
            synth = o.synthesize_note_gpu(midi, phase_table)
            min_len = min(len(synth), len(ref))
            total_loss = total_loss + o.mel_stft_loss(synth[:min_len], ref[:min_len])
        total_loss = total_loss / len(refs)
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = total_loss.item()
        pct = (iteration + 1) / n_iters
        elapsed = time.time() - t_start
        eta = elapsed / max(pct, 0.001) * (1 - pct)
        bar_len = 30
        filled = int(bar_len * pct)
        bar = '=' * filled + '>' * (1 if filled < bar_len else 0) + '.' * (bar_len - filled - 1)
        print(f"\r  [{bar}] {pct*100:5.1f}%  loss={loss_val:.3f}  best={best_loss:.3f}  ETA={eta:.0f}s",
              end='', flush=True)

        if loss_val < best_loss:
            best_loss = loss_val
            best_phases = phase_table.detach().cpu().numpy().copy()

    best_phases = best_phases % (2 * np.pi)
    print(f"\n\nBest loss: {best_loss:.4f} (baseline: ~6.95)")
    print(f"Time: {time.time() - t_start:.0f}s")

    _save_config({'grand_phase_table': best_phases.tolist()})
    print(f"\nWrote optimized phase table to {CONFIG_PATH}")
    print("Run 'generate-grand' again to pick up the new phases.")


def cmd_optimize_grand(args):
    o = optgrand()
    torch, np, time = o.torch, o.np, o.time
    TARGET_NOTES = o.TARGET_ALL if args.all_notes else o.TARGET_5

    print(f"Device: {o.DEVICE}")
    if o.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading {len(TARGET_NOTES)} target piano samples...")
    targets_gpu = {}
    for midi, name in TARGET_NOTES:
        path = os.path.join(o.BASE, 'audio', 'piano', f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        y = o.load_mp3(path)
        if len(y) < o.N_SAMPLES:
            y = np.pad(y, (0, o.N_SAMPLES - len(y)))
        else:
            y = y[:o.N_SAMPLES]
        targets_gpu[name] = torch.tensor(y, dtype=torch.float32, device=o.DEVICE)
        print(f"  {name} (MIDI {midi}): loaded")

    TARGET_NOTES = [(m, n) for m, n in TARGET_NOTES if n in targets_gpu]
    print(f"\nOptimizing against {len(TARGET_NOTES)} notes: {', '.join(n for _, n in TARGET_NOTES)}")
    if not TARGET_NOTES:
        print("No reference samples found in ../audio/piano — nothing to optimize against.")
        return

    # objective() closes over these two containers — mutate in place so it sees them.
    o.targets_gpu.clear()
    o.targets_gpu.update(targets_gpu)
    o.TARGET_NOTES[:] = TARGET_NOTES

    print("\nBenchmarking...")
    current_loss = o.objective(o.CURRENT_PARAMS)
    print(f"Current hand-tuned loss: {current_loss:.4f}")

    t0 = time.time()
    for _ in range(3):
        o.objective(o.CURRENT_PARAMS)
    eval_time = (time.time() - t0) / 3
    print(f"Eval time: {eval_time * 1000:.0f}ms per evaluation")
    total_evals = 300 * 20
    print(f"Estimated optimization time: {total_evals * eval_time / 60:.1f} minutes ({total_evals} evals)")

    print(f"\nOptimizing {len(o.PARAM_DEFS)} parameters with differential evolution...")
    print("─" * 70)

    best_loss_so_far = [current_loss]
    iter_count = [0]
    start_time = time.time()

    def callback(xk, convergence):
        iter_count[0] += 1
        loss = o.objective(xk)
        elapsed = time.time() - start_time
        if loss < best_loss_so_far[0]:
            improvement = (1 - loss / current_loss) * 100
            print(f"\n  ★ Gen {iter_count[0]} ({elapsed:.0f}s): loss={loss:.4f} ({improvement:+.1f}% vs hand-tuned)")
            best_loss_so_far[0] = loss
        elif iter_count[0] % 20 == 0:
            print(f"  Gen {iter_count[0]} ({elapsed:.0f}s): best={best_loss_so_far[0]:.4f}, convergence={convergence:.4f}")

    result = o.differential_evolution(
        o.objective, bounds=o.BOUNDS, maxiter=300, popsize=20, tol=1e-4, seed=42,
        workers=1, callback=callback, disp=True, x0=o.CURRENT_PARAMS,
        init='sobol', mutation=(0.5, 1.5), recombination=0.8,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}\nOPTIMIZATION COMPLETE ({elapsed:.0f}s)\n{'=' * 70}")
    print(f"Best loss: {result.fun:.4f} (was {current_loss:.4f})")
    print(f"Improvement: {(1 - result.fun / current_loss) * 100:.1f}%")

    print(f"\n{'─' * 70}\nPASTE-READY PARAMETERS (not auto-applied)\n{'─' * 70}")
    b1_low, b1_mid, b1_high = result.x[0:3]
    b2_low, b2_mid, b2_high = result.x[3:6]
    print("CALIB_NOTES = {")
    print(f"    36: {{'b1': {b1_low:.4f},  'b2': {b2_low:.4e},  'L': 1.92}},  # C2")
    print(f"    60: {{'b1': {b1_mid:.4f},  'b2': {b2_mid:.4e},  'L': 0.62}},  # C4")
    print(f"    96: {{'b1': {b1_high:.4f},  'b2': {b2_high:.4e},  'L': 0.09}},  # C7")
    print("}")
    sb_cf, sb_bw, sb_gain = result.x[6:9]
    m1g, m2g, m3g = result.x[9:12]
    print(f"\n# Soundboard: low modes (90,30,{m1g:.3f}) (170,35,{m2g:.3f}) (260,45,{m3g:.3f})")
    print(f"#   bridge hill: cf={sb_cf:.0f} Hz bw={sb_bw:.0f} Hz gain={sb_gain:.3f}")
    pb, ps = result.x[12:14]
    af = result.x[14]
    aab, aas = result.x[15:17]
    print(f"\n# prompt_factor = {pb:.4f} + {ps:.4f} * key_pos; after_factor = {af:.4f}")
    print(f"# A_after = {aab:.4f} + {aas:.4f} * key_pos")
    air = result.x[17]
    rb, rl, rc = result.x[18:21]
    print(f"# air_frac = {air:.4f}")
    print(f"# rolloff = {rb:.4f} + {rl:.4f} * key_pos + {rc:.4f} * key_pos ** 3")


def cmd_optimize_rhodes(args):
    o = optrhodes()
    torch, np, librosa, time = o.torch, o.np, o.librosa, o.time

    print(f"Device: {o.DEVICE}")
    if o.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading target Rhodes samples...")
    targets_gpu = {}
    for midi, name in o.TARGET_NOTES:
        path = os.path.join(o.BASE, 'audio', 'rhodes', f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        y, sr = librosa.load(path, sr=o.SAMPLE_RATE, mono=True)
        y = y[:o.N_SAMPLES]
        if len(y) < o.N_SAMPLES:
            y = np.pad(y, (0, o.N_SAMPLES - len(y)))
        targets_gpu[name] = torch.tensor(y, dtype=torch.float32, device=o.DEVICE)
        print(f"  {name}: {len(y)} samples → GPU")

    if not targets_gpu:
        print("No reference samples found in ../audio/rhodes — nothing to optimize against.")
        return

    o.targets_gpu.clear()
    o.targets_gpu.update(targets_gpu)

    current_loss = o.objective(o.CURRENT_PARAMS)
    print(f"\nCurrent hand-tuned loss: {current_loss:.4f}")

    t0 = time.time()
    for _ in range(10):
        o.objective(o.CURRENT_PARAMS)
    eval_time = (time.time() - t0) / 10
    print(f"Eval time: {eval_time * 1000:.0f}ms per evaluation")
    total_evals = 200 * 20
    print(f"Estimated total time: {total_evals * eval_time / 60:.0f} minutes ({total_evals} evals)")

    print(f"\nOptimizing {len(o.PARAM_DEFS)} parameters with differential evolution...")
    best_loss_so_far = [current_loss]
    iter_count = [0]
    start_time = time.time()

    def callback(xk, convergence):
        iter_count[0] += 1
        loss = o.objective(xk)
        elapsed = time.time() - start_time
        if loss < best_loss_so_far[0]:
            improvement = (1 - loss / current_loss) * 100
            print(f"\n  ★ Gen {iter_count[0]} ({elapsed:.0f}s): loss={loss:.4f} ({improvement:+.1f}% vs hand-tuned)")
            best_loss_so_far[0] = loss
        elif iter_count[0] % 10 == 0:
            print(f"  Gen {iter_count[0]} ({elapsed:.0f}s): best={best_loss_so_far[0]:.4f}, convergence={convergence:.4f}")

    result = o.differential_evolution(
        o.objective, bounds=o.BOUNDS, maxiter=200, popsize=15, tol=1e-4, seed=42,
        workers=1, callback=callback, disp=True, x0=o.CURRENT_PARAMS,
        init='sobol', mutation=(0.5, 1.5), recombination=0.8,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}\nOPTIMIZATION COMPLETE ({elapsed:.0f}s)\n{'=' * 60}")
    print(f"Best loss: {result.fun:.4f} (was {current_loss:.4f})")
    print(f"Improvement: {(1 - result.fun / current_loss) * 100:.1f}%")
    print("\nPASTE-READY PARAMETERS (not auto-applied):")
    for name, val in zip(o.PARAM_NAMES, result.x):
        print(f"# {name} = {val:.6f}")


def cmd_tune_warmth(args):
    o = tunewarmth()
    torch, np, librosa, time, itertools = o.torch, o.np, o.librosa, o.time, o.itertools

    print(f"Device: {o.DEVICE}")
    if o.DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading target samples...")
    targets = {}
    for midi, name in o.TARGET_NOTES:
        path = os.path.join(o.BASE, 'audio', 'piano', f'{name}.mp3')
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        y, _ = librosa.load(path, sr=o.SAMPLE_RATE, mono=True)
        if len(y) < o.N_SAMPLES:
            y = np.pad(y, (0, o.N_SAMPLES - len(y)))
        else:
            y = y[:o.N_SAMPLES]
        targets[(midi, name)] = torch.tensor(y, dtype=torch.float32, device=o.DEVICE)
    print(f"  Loaded {len(targets)} samples")
    if not targets:
        print("No reference samples found in ../audio/piano — nothing to sweep against.")
        return

    window = torch.hann_window(2048, device=o.DEVICE)

    if args.param_group == 'rolloff':
        print("\n" + "=" * 70 + "\nSTEP 1: ROLLOFF (warmth)\n" + "=" * 70)
        candidates = []
        bases, linears, cubics = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [0.4, 0.6, 0.8], [3.0, 4.0, 4.5, 5.0]
        print(f"\nSweeping {len(bases)*len(linears)*len(cubics)} combinations...")
        t0 = time.time()
        for rb, rl, rc in itertools.product(bases, linears, cubics):
            total_loss = 0
            for midi, name in o.TARGET_NOTES:
                if (midi, name) not in targets:
                    continue
                gen = o.generate_note_gpu(midi, rb, rl, rc)
                total_loss += o.compute_loss(targets[(midi, name)], gen)
            candidates.append((total_loss / len(targets), rb, rl, rc))
        print(f"Done in {time.time() - t0:.0f}s")
        candidates.sort()
        best_loss, best_rb, best_rl, best_rc = candidates[0]
        print(f"\nBest: rolloff = {best_rb} + {best_rl} * key_pos + {best_rc} * key_pos ** 3  (loss={best_loss:.3f})")
        print("To apply, update the rolloff line in generate-grand's synthesis (not auto-applied).")

    elif args.param_group == 'bridge':
        print("\n" + "=" * 70 + "\nSTEP 2: BRIDGE HILL (brightness character)\n" + "=" * 70)
        rb, rl, rc = args.rolloff_base, args.rolloff_linear, args.rolloff_cubic
        candidates = []
        cfs, bws, gains = [1800, 2000, 2200, 2500, 2800], [800, 1000, 1200, 1500], [0.15, 0.20, 0.25, 0.30, 0.35]
        print(f"Using rolloff=({rb}, {rl}, {rc})")
        print(f"Sweeping {len(cfs)*len(bws)*len(gains)} bridge combinations...")
        t0 = time.time()
        for cf, bw, gain in itertools.product(cfs, bws, gains):
            total_loss = 0
            for midi, name in o.TARGET_NOTES:
                if (midi, name) not in targets:
                    continue
                gen = o.generate_note_gpu(midi, rb, rl, rc, sb_bridge_cf=cf, sb_bridge_bw=bw, sb_bridge_gain=gain)
                total_loss += o.compute_loss(targets[(midi, name)], gen)
            candidates.append((total_loss / len(targets), cf, bw, gain))
        print(f"Done in {time.time() - t0:.0f}s")
        candidates.sort()
        best_loss, best_cf, best_bw, best_g = candidates[0]
        print(f"\nBest bridge: center={best_cf} Hz, BW={best_bw} Hz, gain={best_g}  (loss={best_loss:.3f})")


# ─── reference-recording download (fetch-references) ──────────────────────
#
# Grand piano refs: Salamander Grand Piano V3 (Alexander Holm, CC BY 3.0),
#   re-published as per-note MP3s on the npm registry by darosh/samples-piano-mp3.
#   The raw instrument is only sampled every minor third (C, D#, F#, A per
#   octave) — which is exactly the note grid every command in this tool
#   already asks for.
# Rhodes refs: jRhodes3d (Jeff Learman, sfzinstruments/jlearman.jRhodes3d on
#   GitHub) — full-length FLAC, sampled roughly every 4th white key, which
#   again matches the tool's REFERENCE_NOTES/TARGET_NOTES for the Rhodes.
#
# audio/piano/ and audio/salamander/ end up as the same fetched files: the
# former is the calibration set every grand-piano command reads, the latter
# is what analyze-comparison diffs the generated audio against — there's no
# separate "ground truth" beyond the same Salamander recordings.

PIANO_NOTES_NEEDED = [
    'C2', 'Ds2', 'Fs2', 'A2', 'C3', 'Ds3', 'Fs3', 'A3',
    'C4', 'Ds4', 'Fs4', 'A4', 'C5', 'Ds5', 'Fs5', 'A5', 'C6',
]
RHODES_NOTES_NEEDED = {
    'B1': 35, 'E2': 40, 'A2': 45, 'D3': 50, 'G3': 55, 'B3': 59,
    'D4': 62, 'F4': 65, 'B4': 71, 'E5': 76, 'A5': 81, 'D6': 86,
}


def _to_sharp_notation(name):
    """'Ds2' -> 'D#2', 'C2' -> 'C2' (tarball uses '#', this tool uses 's')."""
    import re
    letter, sharp, octave = re.match(r'^([A-G])(s)?(-?\d+)$', name).groups()
    return f"{letter}{'#' if sharp else ''}{octave}"


def _fetch_piano_references(velocity, force):
    import urllib.request
    import urllib.parse
    import json
    import tarfile
    import io

    pkg = f'@audio-samples/piano-mp3-velocity{velocity}'
    meta_url = f"https://registry.npmjs.org/{urllib.parse.quote(pkg, safe='')}"
    print(f"Piano: querying npm registry for {pkg}...")
    try:
        with urllib.request.urlopen(meta_url, timeout=30) as r:
            meta = json.load(r)
    except Exception as e:
        print(f"  FAILED to reach npm registry: {e}")
        return 0

    latest = meta['dist-tags']['latest']
    tarball_url = meta['versions'][latest]['dist']['tarball']
    print(f"  {pkg}@{latest}")
    print(f"  downloading {tarball_url} ...")
    try:
        with urllib.request.urlopen(tarball_url, timeout=90) as r:
            data = r.read()
    except Exception as e:
        print(f"  FAILED to download tarball: {e}")
        return 0
    print(f"  got {len(data) / 1e6:.1f} MB")

    piano_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'piano')
    sala_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'salamander')
    os.makedirs(piano_dir, exist_ok=True)
    os.makedirs(sala_dir, exist_ok=True)

    got = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
        for name in PIANO_NOTES_NEEDED:
            piano_dest = os.path.join(piano_dir, f'{name}.mp3')
            sala_dest = os.path.join(sala_dir, f'{name}.mp3')
            if os.path.exists(piano_dest) and os.path.exists(sala_dest) and not force:
                print(f"  {name}: already present, skipping (--force to re-fetch)")
                got += 1
                continue
            member_name = f'package/audio/{_to_sharp_notation(name)}v{velocity}.mp3'
            try:
                member = tf.getmember(member_name)
                content = tf.extractfile(member).read()
            except KeyError:
                print(f"  {name}: not found in tarball as {member_name}")
                continue
            with open(piano_dest, 'wb') as f:
                f.write(content)
            with open(sala_dest, 'wb') as f:
                f.write(content)
            print(f"  {name}: OK")
            got += 1

    return got


def _fetch_rhodes_references(velocity_layer, force):
    import urllib.request

    rhodes_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'rhodes')
    os.makedirs(rhodes_dir, exist_ok=True)
    base_url = "https://raw.githubusercontent.com/sfzinstruments/jlearman.jRhodes3d/master/jRhodes3d-mono"

    print(f"Rhodes: fetching from jRhodes3d (velocity layer {velocity_layer})...")
    got = 0
    for name, midi in RHODES_NOTES_NEEDED.items():
        dest = os.path.join(rhodes_dir, f'{name}.mp3')
        if os.path.exists(dest) and not force:
            print(f"  {name}: already present, skipping (--force to re-fetch)")
            got += 1
            continue
        flac_name = f"A_{midi:03d}__{name}_{velocity_layer}.flac"
        url = f"{base_url}/{flac_name}"
        print(f"  {name} (MIDI {midi})...", end=' ', flush=True)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                flac_data = r.read()
        except Exception as e:
            print(f"FAILED ({e})")
            continue
        tmp_flac = dest + '.tmp.flac'
        with open(tmp_flac, 'wb') as f:
            f.write(flac_data)
        import subprocess
        subprocess.run(['ffmpeg', '-y', '-i', tmp_flac, '-codec:a', 'libmp3lame', '-b:a', '128k', dest],
                        capture_output=True)
        os.remove(tmp_flac)
        if os.path.exists(dest):
            print("OK")
            got += 1
        else:
            print("ffmpeg conversion FAILED")

    return got


def cmd_fetch_references(args):
    total_needed = 0
    total_got = 0

    if not args.rhodes_only:
        total_needed += len(PIANO_NOTES_NEEDED)
        total_got += _fetch_piano_references(args.piano_velocity, args.force)
        print()

    if not args.piano_only:
        total_needed += len(RHODES_NOTES_NEEDED)
        total_got += _fetch_rhodes_references(args.rhodes_velocity, args.force)
        print()

    print(f"Fetched {total_got}/{total_needed} reference files.")
    print()
    print("Attribution / licenses (keep these if you redistribute the audio):")
    if not args.rhodes_only:
        print("  Grand piano: Salamander Grand Piano V3 by Alexander Holm — CC BY 3.0")
        print("               (re-served as MP3 by darosh/samples-piano-mp3 on npm)")
    if not args.piano_only:
        print("  Rhodes:      jRhodes3d by Jeff Learman — CC BY-NC")
        print("               (github.com/sfzinstruments/jlearman.jRhodes3d)")
    if total_got < total_needed:
        sys.exit(1)


def cmd_compare_grand(args):
    c = comparepiano()
    orig_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'piano')
    gen_dir = os.path.join(SCRIPT_DIR, '..', 'audio', 'grand-piano')

    comparisons = [(45, 'A2', 'A2', 'A2 (midi 45)'), (81, 'A5', 'A5', 'A5 (midi 81)')]
    nearby = [
        (36, 'C2', 40, 'E2', 'Low register: C2 orig vs E2 gen'),
        (60, 'C4', 62, 'D4', 'Middle register: C4 orig vs D4 gen'),
        (72, 'C5', 76, 'E5', 'High register: C5 orig vs E5 gen'),
    ]

    print("=" * 90 + "\nGRAND PIANO COMPARISON: Recorded vs Generated\n" + "=" * 90)

    def _report(orig_file, gen_file, label):
        orig_path = os.path.join(orig_dir, f'{orig_file}.mp3')
        gen_path = os.path.join(gen_dir, f'{gen_file}.mp3')
        print(f"\n{'─' * 90}\n  {label}\n{'─' * 90}")
        if not os.path.exists(orig_path) or not os.path.exists(gen_path):
            print("  Missing files, skipping")
            return
        orig_a = c.analyze_note(c.mp3_to_wav_array(orig_path))
        gen_a = c.analyze_note(c.mp3_to_wav_array(gen_path))
        print(f"\n  {'Metric':<30} {'RECORDED':>20} {'GENERATED':>20}")
        print(f"  {'Duration':<30} {orig_a['duration']:>19.2f}s {gen_a['duration']:>19.2f}s")
        print(f"  {'Peak time':<30} {orig_a['peak_time']*1000:>18.1f}ms {gen_a['peak_time']*1000:>18.1f}ms")
        print(f"  {'Decay to -10dB':<30} {orig_a['decay_10dB']:>19.2f}s {gen_a['decay_10dB']:>19.2f}s")
        print(f"  {'Decay to -20dB':<30} {orig_a['decay_20dB']:>19.2f}s {gen_a['decay_20dB']:>19.2f}s")
        print(f"  {'Decay to -40dB':<30} {orig_a['decay_40dB']:>19.2f}s {gen_a['decay_40dB']:>19.2f}s")
        print("\n  Spectral energy distribution (attack 0-200ms):")
        print(f"    RECORDED:  {c.format_bands(orig_a['attack_bands'])}")
        print(f"    GENERATED: {c.format_bands(gen_a['attack_bands'])}")

    for _, orig_file, gen_file, name in comparisons:
        _report(orig_file, gen_file, name)
    for _, orig_file, _, gen_file, desc in nearby:
        _report(orig_file, gen_file, desc)


def cmd_compare_rhodes(args):
    c = comparerhodes()
    base = os.path.join(SCRIPT_DIR, '..')
    for note_name, midi in [('D3', 50), ('D4', 62), ('B4', 71)]:
        sampled_path = os.path.join(base, 'audio', 'rhodes', f'{note_name}.mp3')
        fm_path = os.path.join(base, 'audio', 'rhodes-fm', f'{note_name}.mp3')
        if not os.path.exists(sampled_path):
            print(f"Skipping {note_name} — sampled file not found")
            continue
        if not os.path.exists(fm_path):
            print(f"Skipping {note_name} — FM file not found")
            continue
        sampled = c.load_mp3_as_numpy(sampled_path)
        fm = c.load_mp3_as_numpy(fm_path)
        c.analyze_note(sampled, "SAMPLED Rhodes", note_name)
        c.analyze_note(fm, "FM Rhodes", note_name)
        print(f"\n{'~' * 60}")


def cmd_deep_compare(args):
    dc = deepcompare()
    for note_name, midi in [('D3', 50), ('D4', 62), ('B4', 71)]:
        sampled_path = os.path.join(SCRIPT_DIR, '..', 'audio', 'rhodes', f'{note_name}.mp3')
        fm_path = os.path.join(SCRIPT_DIR, '..', 'audio', 'rhodes-fm', f'{note_name}.mp3')
        if not os.path.exists(sampled_path) or not os.path.exists(fm_path):
            print(f"Skipping {note_name}")
            continue
        print(f"\n{'#' * 70}\n  NOTE: {note_name} (MIDI {midi})\n{'#' * 70}")
        y_s, sr = dc.load_audio(sampled_path)
        y_f, sr = dc.load_audio(fm_path)
        stats_s = dc.analyze(y_s, sr, f"SAMPLED Rhodes — {note_name}")
        stats_f = dc.analyze(y_f, sr, f"FM Rhodes — {note_name}")
        dc.compare_mfccs(stats_s, stats_f, "Sampled", "FM")
        print(f"\n{'~' * 70}")


def cmd_analyze_comparison(args):
    a = analyzecomparison()
    base = os.path.join(SCRIPT_DIR, '..', 'audio')
    grand_dir = os.path.join(base, 'grand-piano')
    sala_dir = os.path.join(base, 'salamander')
    if not os.path.isdir(grand_dir) or not os.path.isdir(sala_dir):
        print(f"Expected {grand_dir} and {sala_dir} — one or both are missing.")
        return

    grand_notes = {f.replace('.mp3', '') for f in os.listdir(grand_dir) if f.endswith('.mp3')}
    sala_notes = {f.replace('.mp3', '') for f in os.listdir(sala_dir) if f.endswith('.mp3')}
    common = sorted(grand_notes & sala_notes, key=lambda n: a.name_to_midi(n))

    print(f"Found {len(common)} overlapping notes: {', '.join(common)}\n")
    results = []
    for note in common:
        print(f"Analyzing {note}...", flush=True)
        results.append(a.analyze_note(os.path.join(grand_dir, f'{note}.mp3'),
                                       os.path.join(sala_dir, f'{note}.mp3'), note))
    print()
    a.print_summary(results)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='piano_synth.py',
        description='Physics-based piano & keyboard synthesis toolkit.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('generate-grand', help='Generate grand piano samples (modal synthesis)')
    p.add_argument('--velocity-layers', action='store_true', help='Generate all 8 velocity layers per note')
    p.add_argument('--no-ir', action='store_true', help='Skip soundboard IR convolution (dry synthesis)')
    p.set_defaults(func=cmd_generate_grand)

    p = sub.add_parser('generate-rhodes', help='Generate Rhodes FM samples')
    p.add_argument('--velocity-layers', action='store_true')
    p.set_defaults(func=cmd_generate_rhodes)

    p = sub.add_parser('generate-prism', help='Generate Prism Keys samples')
    p.add_argument('--velocity-layers', action='store_true')
    p.set_defaults(func=cmd_generate_prism)

    p = sub.add_parser('generate-ddsp', help='Train (or reload) and generate DDSP piano samples')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--generate-only', action='store_true', help='Skip training, load the saved model')
    p.set_defaults(func=cmd_generate_ddsp)

    p = sub.add_parser('render-midi', help='Render a .mid file to audio with grand/rhodes/prism synthesis')
    p.add_argument('midi_path', help='Path to the .mid file')
    p.add_argument('--instrument', choices=['grand', 'rhodes', 'prism'], default='grand')
    p.add_argument('--output', help='Output filename (.mp3); default derived from the MIDI filename')
    p.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (2.0 = twice as fast)')
    p.add_argument('--velocity-scale', type=float, default=1.0, help='Multiply all MIDI velocities before synthesis')
    p.set_defaults(func=cmd_render_midi)

    p = sub.add_parser('fetch-references', help='Download the reference recordings the other commands expect in ../audio/')
    p.add_argument('--piano-only', action='store_true')
    p.add_argument('--rhodes-only', action='store_true')
    p.add_argument('--piano-velocity', type=int, default=8, help='Salamander velocity layer 1-16 (default 8)')
    p.add_argument('--rhodes-velocity', type=int, default=4, help='jRhodes3d velocity layer 1-5 (default 4)')
    p.add_argument('--force', action='store_true', help='Re-download even if a file already exists')
    p.set_defaults(func=cmd_fetch_references)

    p = sub.add_parser('extract-soundboard-ir', help='Extract grand piano soundboard IR from reference recordings')
    p.set_defaults(func=cmd_extract_soundboard_ir)

    p = sub.add_parser('extract-rhodes-tf', help='Extract Rhodes pickup/amp transfer function from reference recordings')
    p.set_defaults(func=cmd_extract_rhodes_tf)

    p = sub.add_parser('optimize-phases', help='Gradient-descent grand piano phase table; writes to piano_synth_config.json')
    p.add_argument('--iters', type=int, default=400)
    p.set_defaults(func=cmd_optimize_phases)

    p = sub.add_parser('optimize-grand', help='Differential-evolution search over grand piano physical params (prints paste-ready block)')
    p.add_argument('--all-notes', action='store_true', help='Optimize against all 17 references instead of 5')
    p.set_defaults(func=cmd_optimize_grand)

    p = sub.add_parser('optimize-rhodes', help='Differential-evolution search over FM Rhodes params (prints paste-ready block)')
    p.set_defaults(func=cmd_optimize_rhodes)

    p = sub.add_parser('tune-warmth', help='Grid-sweep grand piano rolloff/bridge params against references')
    p.add_argument('param_group', choices=['rolloff', 'bridge'])
    p.add_argument('--rolloff-base', type=float, default=0.7, help='(bridge sweep only) fixed rolloff base to sweep against')
    p.add_argument('--rolloff-linear', type=float, default=0.6)
    p.add_argument('--rolloff-cubic', type=float, default=4.5)
    p.set_defaults(func=cmd_tune_warmth)

    p = sub.add_parser('compare-grand', help='Compare recorded vs generated grand piano')
    p.set_defaults(func=cmd_compare_grand)

    p = sub.add_parser('compare-rhodes', help='Compare sampled vs FM Rhodes (spectral peaks)')
    p.set_defaults(func=cmd_compare_rhodes)

    p = sub.add_parser('analyze-comparison', help='Full metric comparison: grand piano vs Salamander')
    p.set_defaults(func=cmd_analyze_comparison)

    p = sub.add_parser('deep-compare', help='MFCC + envelope comparison: sampled vs FM Rhodes (needs librosa)')
    p.set_defaults(func=cmd_deep_compare)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
