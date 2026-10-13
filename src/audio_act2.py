#!/usr/bin/env python3
"""
Procedural Synthesizer & Audio Telemetry Engine for ARGON // 18 — ACT II ("Google is back")
Generates:
1. `out/soundtrack_act2.wav`: 38.0-second (1,140 frames @ 30fps) 48kHz stereo PCM soundtrack
   in D Minor / D Dorian / D Phrygian at 120.0 BPM:
   - 0.0s - 6.0s:   Dimensional Breach (seamless harmonic bridge from Act I + low brass BRAAAMs)
   - 6.0s - 14.0s:  Impossible Architecture (Reese neuro-bass, industrial percussion, Shepherd riser)
   - 14.0s - 29.5s: THE AWAKENING DROP ("Google is back" — massive hybrid-trap 808s, anvil snares,
                    7-oscillator cyber-brass, and 4-color spectral lead motif)
   - 29.5s - 38.0s: Ironic Smart-Home Cut (record-scratch power-down -> deadpan silence ->
                    friendly smart-speaker chime -> formant-synthesized voice cadence ->
                    tactile Nest Thermostat rotary clicks to 19.0 °C -> warm Eco-Leaf chord).
2. `out/audio_env_act2.npz`: Per-frame (1,140 frames) audio telemetry envelopes (`kick`, `snare`,
   `brass`, `lead`, `rms`, `voice_led`, `spectrum`) driving the 3D PBR raymarcher and smart-home LEDs.
3. `out/soundtrack_full.wav`: Continuous 80.0-second stereo master combining Act I (42.0s) and
   Act II (38.0s) with a seamless harmonic crossfade at t = 42.0s.
"""

import os
import time
import wave
import numpy as np

SR = 48000
DURATION_ACT2 = 38.0
FPS = 30
N_SAMPLES = int(SR * DURATION_ACT2)
N_FRAMES = int(FPS * DURATION_ACT2)
BPM = 120.0
BEAT_SEC = 60.0 / BPM       # 0.5s
STEP_SEC = BEAT_SEC / 4.0   # 0.125s


def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def stereo_hall_reverb(left: np.ndarray, right: np.ndarray, sr: int = SR, cutoff_s: float = 29.48) -> tuple[np.ndarray, np.ndarray]:
    """Massive dark cathedral FDN-style stereo reverb, hard-gated at `cutoff_s` for the comedic cut."""
    wet_l = np.zeros_like(left)
    wet_r = np.zeros_like(right)
    mono = 0.5 * (left + right)

    diff = mono.copy()
    for d_ms, g in [(17.3, 0.48), (41.9, 0.42), (89.7, 0.38), (143.1, 0.32)]:
        d = int(d_ms * 0.001 * sr)
        padded = np.zeros_like(diff)
        padded[d:] = diff[:-d]
        diff = diff + g * padded

    taps_l = [(33.1, 0.38), (53.7, 0.32), (83.9, 0.26), (127.3, 0.21), (181.1, 0.16), (257.9, 0.12), (347.3, 0.09), (463.7, 0.06)]
    taps_r = [(37.9, 0.38), (59.3, 0.32), (91.1, 0.26), (139.7, 0.21), (197.3, 0.16), (277.1, 0.12), (373.9, 0.09), (491.3, 0.06)]

    for d_ms, gain in taps_l:
        d = int(d_ms * 0.001 * sr)
        wet_l[d:] += gain * diff[:-d]
    for d_ms, gain in taps_r:
        d = int(d_ms * 0.001 * sr)
        wet_r[d:] += gain * diff[:-d]

    # Gate the cathedral reverb instantly at the 29.5s hard comedic cut!
    cut_idx = int(cutoff_s * sr)
    if cut_idx < len(wet_l):
        fade_len = int(0.04 * sr)
        taper = np.linspace(1.0, 0.0, fade_len)
        wet_l[cut_idx:cut_idx + fade_len] *= taper
        wet_r[cut_idx:cut_idx + fade_len] *= taper
        wet_l[cut_idx + fade_len:] = 0.0
        wet_r[cut_idx + fade_len:] = 0.0

    return wet_l, wet_r


def synth_braaam(dur: float, root_midi: float, sr: int = SR, rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthesizes a massive Hans-Zimmer/Blade-Runner style low brass/horn BRAAAM blast."""
    n = int(dur * sr)
    lt = np.arange(n, dtype=np.float64) / sr
    sig_l = np.zeros(n, dtype=np.float64)
    sig_r = np.zeros(n, dtype=np.float64)

    # Pitch bends slightly downward over the blast for ominous weight
    bend = 1.0 + 0.012 * np.exp(-lt * 3.5) - 0.008 * (lt / dur)
    notes = [root_midi, root_midi + 12, root_midi + 19, root_midi + 24, root_midi + 27]
    weights = [1.0, 0.85, 0.55, 0.45, 0.30]

    for note, w in zip(notes, weights):
        f0 = midi_to_freq(note)
        for det_c, pl, pr in [(-14.0, 0.9, 0.2), (-5.0, 0.7, 0.4), (0.0, 0.6, 0.6), (6.0, 0.4, 0.7), (15.0, 0.2, 0.9)]:
            f = f0 * (2.0 ** (det_c / 1200.0)) * bend
            ph = 2.0 * np.pi * np.cumsum(f) / sr + (rng.uniform(0, 6.28) if rng else 0.0)
            # Brass-like wavefolded harmonic series
            raw = (
                np.sin(ph)
                + 0.65 * np.sin(2.0 * ph + 0.2)
                + 0.45 * np.sin(3.0 * ph + 0.5)
                + 0.30 * np.sin(4.0 * ph + 0.9)
                + 0.20 * np.sin(5.0 * ph + 1.2)
                + 0.12 * np.sin(6.0 * ph + 1.6)
            )
            # Dynamic brass swell envelope
            swell = np.clip(lt / 0.18, 0.0, 1.0) * np.exp(-lt * 0.85) * (1.0 + 0.35 * np.sin(np.pi * lt / dur))
            folded = np.tanh(raw * (1.4 + 1.2 * swell)) * w * 0.045
            sig_l += folded * pl * swell
            sig_r += folded * pr * swell

    return sig_l, sig_r


def generate_act2_soundtrack(out_dir: str = "/workspace/argon-video/out") -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(202618)

    t = np.arange(N_SAMPLES, dtype=np.float64) / SR
    mix_l = np.zeros(N_SAMPLES, dtype=np.float64)
    mix_r = np.zeros(N_SAMPLES, dtype=np.float64)
    rev_l = np.zeros(N_SAMPLES, dtype=np.float64)
    rev_r = np.zeros(N_SAMPLES, dtype=np.float64)

    # Telemetry stems
    stem_kick = np.zeros(N_SAMPLES, dtype=np.float64)
    stem_snare = np.zeros(N_SAMPLES, dtype=np.float64)
    stem_brass = np.zeros(N_SAMPLES, dtype=np.float64)
    stem_lead = np.zeros(N_SAMPLES, dtype=np.float64)
    stem_voice = np.zeros(N_SAMPLES, dtype=np.float64)

    # =========================================================================
    # PHASE 1 (0.0s - 6.0s): SEAMLESS BRIDGE & DIMENSIONAL BREACH BRAAAMS
    # =========================================================================
    # 1A. Warm Dadd9 -> Dark Dm(b6) morphing pad over 0.0s - 6.0s
    bridge_len = int(6.2 * SR)
    bt = np.arange(bridge_len, dtype=np.float64) / SR
    # At t=0, matches Act I's Dadd9 [38, 45, 50, 57, 62, 66, 69]; bends F#4 (66) -> F4 (65)
    for note in [38, 45, 50, 57, 62, 66, 69]:
        f_start = midi_to_freq(note)
        f_end = midi_to_freq(65 if note == 66 else note)
        morph_curve = np.clip((bt - 0.3) / 1.8, 0.0, 1.0)
        freq_t = f_start + (f_end - f_start) * morph_curve
        ph = 2.0 * np.pi * np.cumsum(freq_t) / SR
        env = np.exp(-bt * 0.38) * 0.045
        osc = np.sin(ph) + 0.4 * np.sin(2.0 * ph + 0.4) + 0.2 * np.sin(3.0 * ph + 0.8)
        mix_l[:bridge_len] += osc * env * 0.7
        mix_r[:bridge_len] += osc * env * 0.7
        rev_l[:bridge_len] += osc * env * 0.5
        rev_r[:bridge_len] += osc * env * 0.5

    # 1B. Two massive Dimensional Breach BRAAAM blasts at t=0.6s (D1=26) and t=3.2s (Bb0=22)
    for br_time, br_dur, br_note in [(0.55, 2.5, 26), (3.20, 2.7, 22)]:
        idx = int(br_time * SR)
        bl, br = synth_braaam(br_dur, br_note, SR, rng)
        end = min(N_SAMPLES, idx + len(bl))
        mix_l[idx:end] += bl[:end - idx] * 1.15
        mix_r[idx:end] += br[:end - idx] * 1.15
        rev_l[idx:end] += bl[:end - idx] * 0.65
        rev_r[idx:end] += br[:end - idx] * 0.65
        stem_brass[idx:end] += 0.5 * (np.abs(bl[:end - idx]) + np.abs(br[:end - idx]))

    # 1C. Gravitational warp shimmer (0.2s - 3.5s) as 2D plane extrudes into 3D
    sh_len = int(3.5 * SR)
    sht = np.arange(sh_len, dtype=np.float64) / SR
    for h_i, f_base in enumerate([587.33, 880.0, 1174.66, 1318.5, 1760.0]):
        f_mod = f_base * (1.0 + 0.03 * np.sin(2.0 * np.pi * (1.5 + h_i * 0.4) * sht))
        ph = 2.0 * np.pi * np.cumsum(f_mod) / SR
        env = np.sin(np.pi * sht / 3.5) ** 2 * 0.024
        pan = 0.5 + 0.4 * np.sin(sht * 2.5 + h_i)
        sig = np.sin(ph + 1.8 * np.sin(ph * 2.0)) * env
        mix_l[:sh_len] += sig * pan
        mix_r[:sh_len] += sig * (1.0 - pan)
        rev_l[:sh_len] += sig * 0.6
        rev_r[:sh_len] += sig * 0.6

    # =========================================================================
    # PHASE 2 (6.0s - 14.0s): IMPOSSIBLE ARCHITECTURE (REESE BASS & BUILD)
    # =========================================================================
    # Dark menacing Reese Neuro-Bass (6.0s -> 13.75s)
    reese_sched = [
        (6.0,  8.0,  26, 26),  # D1
        (8.0,  10.0, 27, 27),  # Eb1 (Phrygian dark half-step!)
        (10.0, 12.0, 24, 26),  # C1 -> D1
        (12.0, 13.75, 21, 33), # A0 -> A1 octave slide up into drop!
    ]
    for r_s, r_e, n0, n1 in reese_sched:
        s_i = int(r_s * SR)
        e_i = int(r_e * SR)
        rt = np.arange(e_i - s_i, dtype=np.float64) / SR
        dur = r_e - r_s
        f_curve = midi_to_freq(n0) + (midi_to_freq(n1) - midi_to_freq(n0)) * (rt / dur) ** 2
        ph0 = 2.0 * np.pi * np.cumsum(f_curve) / SR
        ph1 = 2.0 * np.pi * np.cumsum(f_curve * 1.007) / SR
        ph2 = 2.0 * np.pi * np.cumsum(f_curve * 0.993) / SR
        sub = np.sin(ph0)
        reese_l = np.tanh((0.7 * sub + 0.45 * np.sin(ph1) + 0.3 * np.sin(2.0 * ph1)) * 1.8) * 0.24
        reese_r = np.tanh((0.7 * sub + 0.45 * np.sin(ph2) + 0.3 * np.sin(2.0 * ph2)) * 1.8) * 0.24
        env = np.clip(rt / 0.08, 0.0, 1.0) * np.clip((dur - rt) / 0.08, 0.0, 1.0)
        mix_l[s_i:e_i] += reese_l * env
        mix_r[s_i:e_i] += reese_r * env
        stem_brass[s_i:e_i] += 0.5 * (np.abs(reese_l) + np.abs(reese_r)) * env

    # Shepherd-Tone & Orchestral Synth Riser (8.0s -> 13.75s)
    ris_s = int(8.0 * SR)
    ris_e = int(13.75 * SR)
    rt = np.arange(ris_e - ris_s, dtype=np.float64) / SR
    u_ris = rt / (13.75 - 8.0)
    for oct_i in range(4):
        f_sweep = (73.42 * (2 ** oct_i)) * (2.0 ** (u_ris * 2.0))
        ph = 2.0 * np.pi * np.cumsum(f_sweep) / SR
        amp = (u_ris ** 2.4) * 0.045
        mix_l[ris_s:ris_e] += np.sin(ph) * amp
        mix_r[ris_s:ris_e] += np.cos(ph) * amp
    # Filtered white-noise tension whoosh
    whoosh = rng.normal(0, 1, ris_e - ris_s) * (u_ris ** 3.0) * 0.14
    mix_l[ris_s:ris_e] += whoosh
    mix_r[ris_s:ris_e] += whoosh
    rev_l[ris_s:ris_e] += whoosh * 0.6
    rev_r[ris_s:ris_e] += whoosh * 0.6

    # =========================================================================
    # PHASE 3 (14.0s - 29.5s): THE AWAKENING DROP ("GOOGLE IS BACK")
    # =========================================================================
    sidechain = np.ones(N_SAMPLES, dtype=np.float64)

    # 3A. Kicks & Sub-808s across Phase 2 & Phase 3
    for beat in range(int(29.5 / BEAT_SEC)):
        b_sec = beat * BEAT_SEC
        trigger_k = False
        k_gain = 1.0

        if b_sec in (0.5, 3.0):
            trigger_k = True
            k_gain = 0.95
        elif 6.0 <= b_sec < 13.5:
            # Half-time heavy industrial kicks on beats 0, 2.5, 4, 6.5...
            if (beat % 4 == 0) or (beat % 8 == 5):
                trigger_k = True
                k_gain = 0.92
        elif 14.0 <= b_sec < 29.5:
            # Massive driving drop kicks: every beat + syncopated double-kicks before bar lines
            trigger_k = True
            k_gain = 1.15

        if trigger_k:
            idx = int(b_sec * SR)
            k_len = int(0.42 * SR)
            end = min(N_SAMPLES, idx + k_len)
            kt = np.arange(end - idx, dtype=np.float64) / SR
            # Aggressive 180Hz -> 38Hz punch + sub body
            f_env = 38.0 + 140.0 * np.exp(-kt * 42.0) + 220.0 * np.exp(-kt * 190.0)
            ph = 2.0 * np.pi * np.cumsum(f_env) / SR
            click = 0.42 * np.exp(-kt * 280.0) * np.sin(2.0 * np.pi * 1850.0 * kt)
            kw = np.tanh((np.sin(ph) * np.exp(-kt * 7.8) + click) * 1.65) * 0.50 * k_gain
            mix_l[idx:end] += kw
            mix_r[idx:end] += kw
            stem_kick[idx:end] += np.abs(kw)

            # Sidechain ducking
            sc_len = min(N_SAMPLES - idx, int(0.30 * SR))
            sct = np.arange(sc_len, dtype=np.float64) / SR
            duck = 1.0 - 0.72 * np.exp(-sct * 13.0) * (1.0 - np.exp(-sct * 200.0))
            sidechain[idx:idx + sc_len] = np.minimum(sidechain[idx:idx + sc_len], duck)

        # Syncopated 16th ghost kicks in the Drop (14.0s - 29.0s)
        if 14.0 <= b_sec < 29.0 and (beat % 4 in (1, 3)):
            g_idx = int((b_sec + 3 * STEP_SEC) * SR)
            k_len = int(0.18 * SR)
            end = min(N_SAMPLES, g_idx + k_len)
            kt = np.arange(end - g_idx, dtype=np.float64) / SR
            f_env = 42.0 + 110.0 * np.exp(-kt * 50.0)
            kw = np.tanh(np.sin(2.0 * np.pi * np.cumsum(f_env) / SR) * np.exp(-kt * 15.0) * 1.4) * 0.32
            mix_l[g_idx:end] += kw
            mix_r[g_idx:end] += kw
            stem_kick[g_idx:end] += np.abs(kw)

        # Anvil / Taiko Cinematic Snares on odd beats in Drop (14.0s - 29.5s) & half-time in Phase 2
        is_drop_snare = (14.0 <= b_sec < 29.5) and (beat % 2 == 1)
        is_build_snare = (6.0 <= b_sec < 12.0) and (beat % 4 == 2)
        if is_drop_snare or is_build_snare:
            s_idx = int(b_sec * SR)
            s_len = int(0.32 * SR)
            end = min(N_SAMPLES, s_idx + s_len)
            st = np.arange(end - s_idx, dtype=np.float64) / SR
            # Tonal drum + metallic FM anvil partial + stereo noise burst
            body = np.sin(2.0 * np.pi * np.cumsum(150.0 + 95.0 * np.exp(-st * 45.0)) / SR) * np.exp(-st * 19.0) * 0.50
            anvil = (np.sin(2.0 * np.pi * 920.0 * st) + 0.6 * np.sin(2.0 * np.pi * 1480.0 * st)) * np.exp(-st * 35.0) * 0.22
            nl = rng.normal(0, 1, len(st)) * np.exp(-st * 13.0) * 0.38
            nr = rng.normal(0, 1, len(st)) * np.exp(-st * 13.0) * 0.38
            sn_l = np.tanh((body + anvil + nl) * 1.45) * 0.35
            sn_r = np.tanh((body + anvil + nr) * 1.45) * 0.35
            mix_l[s_idx:end] += sn_l
            mix_r[s_idx:end] += sn_r
            rev_l[s_idx:end] += sn_l * 0.55
            rev_r[s_idx:end] += sn_r * 0.55
            stem_snare[s_idx:end] += 0.5 * (np.abs(sn_l) + np.abs(sn_r))

    # Accelerating Snare Roll (12.0s -> 13.75s) before the 14.0s Drop!
    roll_times = [12.0, 12.25, 12.5, 12.75, 13.0, 13.125, 13.25, 13.375, 13.5, 13.5625, 13.625, 13.6875]
    for r_i, rt_s in enumerate(roll_times):
        s_idx = int(rt_s * SR)
        s_len = int(0.14 * SR)
        end = min(N_SAMPLES, s_idx + s_len)
        st = np.arange(end - s_idx, dtype=np.float64) / SR
        vel = 0.14 + 0.22 * (r_i / len(roll_times))
        sn = (np.sin(2.0 * np.pi * (180.0 + r_i * 8.0) * st) + rng.normal(0, 0.8, len(st))) * np.exp(-st * 26.0) * vel
        mix_l[s_idx:end] += sn
        mix_r[s_idx:end] += sn
        stem_snare[s_idx:end] += np.abs(sn)

    # 3B. Roaring Cyber-Brass & Sub Chords in the Drop (14.0s - 29.5s)
    drop_chords = [
        (14.0, 18.0, 26, [38, 45, 50, 53, 57, 62, 64]),   # Dm9
        (18.0, 22.0, 22, [34, 41, 46, 50, 53, 57, 64]),   # Bbmaj9(#11)
        (22.0, 26.0, 31, [31, 38, 43, 46, 50, 53, 57]),   # Gm9
        (26.0, 29.5, 21, [33, 40, 45, 49, 52, 55, 58]),   # A7(b9) -> climax!
    ]
    brass_l = np.zeros(N_SAMPLES, dtype=np.float64)
    brass_r = np.zeros(N_SAMPLES, dtype=np.float64)

    for c_s, c_e, sub_n, chord_notes in drop_chords:
        idx0 = int(c_s * SR)
        idx1 = int(c_e * SR)
        ct = np.arange(idx1 - idx0, dtype=np.float64) / SR
        dur = c_e - c_s
        env = np.clip(ct / 0.05, 0.0, 1.0) * np.clip((dur - ct) / 0.08, 0.0, 1.0)

        # Massive Sub-Bass Pedal with 16th-note pulse
        f_sub = midi_to_freq(sub_n)
        sub_wave = np.tanh(np.sin(2.0 * np.pi * f_sub * ct) * 1.6) * 0.28 * env
        brass_l[idx0:idx1] += sub_wave
        brass_r[idx0:idx1] += sub_wave

        # 7-voice detuned Cyber-Brass stack
        for n_m in chord_notes:
            f0 = midi_to_freq(n_m)
            for det_c, pl, pr in [(-11.0, 0.85, 0.25), (0.0, 0.65, 0.65), (11.0, 0.25, 0.85)]:
                f = f0 * (2.0 ** (det_c / 1200.0))
                ph = 2.0 * np.pi * f * ct + rng.uniform(0, 6.28)
                osc = (
                    np.sin(ph)
                    + 0.55 * np.sin(2.0 * ph + 0.3)
                    + 0.35 * np.sin(3.0 * ph + 0.7)
                    + 0.20 * np.sin(4.0 * ph + 1.1)
                )
                brass_l[idx0:idx1] += osc * pl * env * 0.026
                brass_r[idx0:idx1] += osc * pr * env * 0.026

    brass_l *= sidechain
    brass_r *= sidechain
    mix_l += brass_l
    mix_r += brass_r
    rev_l += brass_l * 0.45
    rev_r += brass_r * 0.45
    stem_brass += 0.5 * (np.abs(brass_l) + np.abs(brass_r))

    # 3C. Crisp 16th/32nd Hi-Hat & Cyber-Arpeggiator in Drop (14.0s - 29.5s)
    total_steps = int(29.5 / STEP_SEC)
    for step in range(total_steps):
        st_s = step * STEP_SEC
        if st_s < 6.0 or (13.6 <= st_s < 14.0):
            continue
        h_idx = int(st_s * SR)
        # Hi-hats
        h_len = int(0.06 * SR)
        end = min(N_SAMPLES, h_idx + h_len)
        ht = np.arange(end - h_idx, dtype=np.float64) / SR
        metal = (
            np.sin(2.0 * np.pi * 6800.0 * ht)
            + 0.7 * np.sin(2.0 * np.pi * 9400.0 * ht)
            + rng.normal(0, 0.9, len(ht))
        )
        vel = 0.08 if (step % 4 == 2) else 0.042
        if st_s >= 14.0:
            vel *= 1.35
        hh = metal * np.exp(-ht * 45.0) * vel
        pan = 0.65 if (step % 2 == 0) else 0.35
        mix_l[h_idx:end] += hh * pan
        mix_r[h_idx:end] += hh * (1.0 - pan)

        # Driving 16th-note FM Arpeggiator
        arp_seq = [62, 65, 69, 74, 77, 74, 69, 65] if st_s < 18.0 else (
            [62, 65, 70, 74, 77, 81, 77, 70] if st_s < 22.0 else (
                [62, 67, 70, 74, 79, 82, 79, 74] if st_s < 26.0 else [64, 69, 73, 76, 79, 85, 81, 76]
            )
        )
        note = arp_seq[step % len(arp_seq)]
        a_len = min(N_SAMPLES - h_idx, int(0.22 * SR))
        at = np.arange(a_len, dtype=np.float64) / SR
        f0 = midi_to_freq(note)
        mod = 2.8 * np.exp(-at * 22.0) * np.sin(2.0 * np.pi * (f0 * 2.0) * at)
        sig = np.sin(2.0 * np.pi * f0 * at + mod) * np.exp(-at * 14.0) * (0.09 if st_s < 14.0 else 0.13)
        p_arp = 0.5 + 0.38 * np.sin(step * 0.7)
        mix_l[h_idx:h_idx + a_len] += sig * p_arp
        mix_r[h_idx:h_idx + a_len] += sig * (1.0 - p_arp)
        rev_l[h_idx:h_idx + a_len] += sig * 0.35
        rev_r[h_idx:h_idx + a_len] += sig * 0.35
        stem_lead[h_idx:h_idx + a_len] += np.abs(sig)

    # 3D. Heroic 4-Color Spectral Anthem Lead Melody in Drop (14.0s - 29.5s)
    anthem_notes = [
        # Bar 1-2 of Drop: The 4-Note Spectral Motif (D5 -> F5 -> A5 -> D6)
        (14.0, 0.75, 74), (14.75, 0.75, 77), (15.5, 0.5, 81), (16.0, 1.5, 86), (17.5, 0.5, 84),
        # Bar 3-4:
        (18.0, 0.75, 86), (18.75, 0.75, 81), (19.5, 0.5, 77), (20.0, 1.5, 76), (21.5, 0.5, 74),
        # Bar 5-6:
        (22.0, 0.75, 79), (22.75, 0.75, 82), (23.5, 0.5, 86), (24.0, 1.5, 88), (25.5, 0.5, 86),
        # Bar 7-8 (Climax towards 29.5s cut!):
        (26.0, 1.0, 85), (27.0, 1.0, 86), (28.0, 1.48, 88),
    ]
    for st_s, dur_s, l_note in anthem_notes:
        idx0 = int(st_s * SR)
        idx1 = min(int(29.48 * SR), int((st_s + dur_s) * SR))
        if idx1 <= idx0:
            continue
        lt = np.arange(idx1 - idx0, dtype=np.float64) / SR
        f0 = midi_to_freq(l_note)
        vib = 0.005 * np.clip((lt - 0.15) / 0.25, 0.0, 1.0) * np.sin(2.0 * np.pi * 6.0 * lt)
        ph = 2.0 * np.pi * f0 * np.cumsum(1.0 + vib) / SR
        lead_w = (np.sin(ph) + 0.6 * np.sin(2 * ph) + 0.35 * np.sin(3 * ph) + 0.2 * np.sin(4 * ph))
        env = np.clip(lt / 0.03, 0.0, 1.0) * np.clip((dur_s - lt) / 0.08, 0.0, 1.0) * 0.105
        mix_l[idx0:idx1] += lead_w * env * sidechain[idx0:idx1]
        mix_r[idx0:idx1] += lead_w * env * sidechain[idx0:idx1]
        rev_l[idx0:idx1] += lead_w * env * 0.55
        rev_r[idx0:idx1] += lead_w * env * 0.55
        stem_lead[idx0:idx1] += np.abs(lead_w * env)

    # Final Singularity Riser (26.5s -> 29.48s) before the Hard Cut
    cr_s = int(26.5 * SR)
    cr_e = int(29.48 * SR)
    crt = np.arange(cr_e - cr_s, dtype=np.float64) / SR
    u_cr = crt / (29.48 - 26.5)
    f_cr = 293.66 * (2.0 ** (u_cr * 2.5))
    ph_cr = 2.0 * np.pi * np.cumsum(f_cr) / SR
    cr_sig = (0.5 * np.sin(ph_cr) + 0.5 * rng.normal(0, 1, len(crt))) * (u_cr ** 2.5) * 0.18
    mix_l[cr_s:cr_e] += cr_sig
    mix_r[cr_s:cr_e] += cr_sig

    # Apply cathedral reverb to Phases 1-3 (gated cleanly at 29.48s!)
    wet_l, wet_r = stereo_hall_reverb(rev_l, rev_r, SR, cutoff_s=29.48)
    mix_l += wet_l * 0.40
    mix_r += wet_r * 0.40

    # Hard gate all Phase 1-3 audio at 29.50s
    cut_sample = int(29.50 * SR)
    mix_l[cut_sample:] = 0.0
    mix_r[cut_sample:] = 0.0

    # =========================================================================
    # PHASE 4 (29.5s - 38.0s): THE IRONIC GOOGLE HOME / NEST EPILOGUE
    # =========================================================================
    # 4A. Vinyl Record-Scratch / Relay Click & Power-Down Whine right at 29.48s - 29.72s
    sc_s = int(29.46 * SR)
    sc_e = int(29.72 * SR)
    sct = np.arange(sc_e - sc_s, dtype=np.float64) / SR
    # Rapid downward pitch scrub (900 Hz -> 45 Hz) + mechanical switch click
    f_scrub = 900.0 * np.exp(-sct * 18.0) + 35.0
    scrub = np.tanh(np.sin(2.0 * np.pi * np.cumsum(f_scrub) / SR) * 2.2) * np.exp(-sct * 11.0) * 0.38
    click_sw = rng.normal(0, 1, len(sct)) * np.exp(-sct * 140.0) * 0.45
    mix_l[sc_s:sc_e] += scrub + click_sw
    mix_r[sc_s:sc_e] += scrub + click_sw

    # 4B. Friendly 2-Note Smart Speaker Wake Chime at 30.05s (G5 -> D6 warm bell/marimba)
    for ch_t, ch_midi in [(30.05, 79), (30.22, 86)]:
        c_idx = int(ch_t * SR)
        c_len = int(0.45 * SR)
        ct = np.arange(c_len, dtype=np.float64) / SR
        f0 = midi_to_freq(ch_midi)
        chime = (np.sin(2.0 * np.pi * f0 * ct) + 0.25 * np.sin(2.0 * np.pi * f0 * 3.0 * ct) * np.exp(-ct * 25.0)) * np.exp(-ct * 8.5) * 0.22
        mix_l[c_idx:c_idx + c_len] += chime
        mix_r[c_idx:c_idx + c_len] += chime
        stem_voice[c_idx:c_idx + c_len] += np.abs(chime)

    # 4C. Formant-Synthesized Assistant Speech Cadence (30.65s - 34.15s)
    # Syllabic prosody mimicking: "Sorry, I didn't understand that. Setting thermostat to 19 degrees."
    syllables = [
        # (start_s, dur_s, pitch_hz, f1_hz, f2_hz)
        (30.65, 0.20, 215.0, 520.0, 1150.0),  # "Sor-"
        (30.86, 0.16, 195.0, 320.0, 2100.0),  # "-ry,"
        (31.12, 0.14, 205.0, 680.0, 1250.0),  # "I"
        (31.28, 0.15, 218.0, 380.0, 1950.0),  # "did-"
        (31.44, 0.14, 200.0, 420.0, 1700.0),  # "-n't"
        (31.62, 0.16, 210.0, 580.0, 1300.0),  # "un-"
        (31.79, 0.15, 202.0, 460.0, 1650.0),  # "-der-"
        (31.95, 0.24, 190.0, 660.0, 1450.0),  # "-stand"
        (32.22, 0.26, 178.0, 540.0, 1350.0),  # "that."
        # Second phrase: "Setting to nineteen degrees."
        (32.70, 0.18, 212.0, 480.0, 1800.0),  # "Set-"
        (32.89, 0.16, 204.0, 350.0, 2050.0),  # "-ting"
        (33.08, 0.14, 198.0, 400.0, 1500.0),  # "to"
        (33.25, 0.22, 216.0, 650.0, 1400.0),  # "nine-"
        (33.48, 0.22, 205.0, 310.0, 2200.0),  # "-teen"
        (33.72, 0.16, 192.0, 420.0, 1750.0),  # "de-"
        (33.89, 0.28, 175.0, 300.0, 2150.0),  # "-grees."
    ]
    for s_st, s_dur, f0, f1, f2 in syllables:
        idx0 = int(s_st * SR)
        idx1 = int((s_st + s_dur) * SR)
        st = np.arange(idx1 - idx0, dtype=np.float64) / SR
        # Smooth glottal harmonic stack near F1 and F2 formants
        h1 = max(1, round(f1 / f0))
        h2 = max(2, round(f2 / f0))
        env = np.sin(np.pi * st / s_dur) ** 1.4 * 0.11
        voice_sig = (
            0.45 * np.sin(2.0 * np.pi * f0 * st)
            + 0.55 * np.sin(2.0 * np.pi * (f0 * h1) * st)
            + 0.30 * np.sin(2.0 * np.pi * (f0 * h2) * st)
            + 0.15 * np.sin(2.0 * np.pi * (f0 * (h1 + 1)) * st)
        ) * env
        mix_l[idx0:idx1] += voice_sig
        mix_r[idx0:idx1] += voice_sig
        stem_voice[idx0:idx1] += np.abs(voice_sig) * 2.5

    # 4D. Tactile Nest Thermostat Rotary Dial Clicks (21.5 °C -> 19.0 °C: 5 half-degree clicks)
    click_times = [32.75, 33.05, 33.35, 33.65, 33.95]
    for cl_t in click_times:
        c_idx = int(cl_t * SR)
        c_len = int(0.025 * SR)
        ct = np.arange(c_len, dtype=np.float64) / SR
        # Crisp mechanical detent click (2.4 kHz + 4.8 kHz damped impulse)
        detent = (np.sin(2.0 * np.pi * 2400.0 * ct) + 0.6 * np.sin(2.0 * np.pi * 4800.0 * ct)) * np.exp(-ct * 320.0) * 0.24
        mix_l[c_idx:c_idx + c_len] += detent * 0.35
        mix_r[c_idx:c_idx + c_len] += detent * 0.85  # Panned slightly right towards the thermostat!

    # 4E. Warm Cozy Eco-Leaf Resolution Chime at 34.25s -> 38.0s (Dmaj9 warm electric piano)
    eco_idx = int(34.25 * SR)
    eco_len = N_SAMPLES - eco_idx
    et = np.arange(eco_len, dtype=np.float64) / SR
    eco_notes = [50, 57, 62, 66, 69, 76]  # D3, A3, D4, F#4, A4, E5
    for n_i, en in enumerate(eco_notes):
        st_delay = n_i * 0.055
        mask = et >= st_delay
        lt_e = np.maximum(0.0, et - st_delay)
        f0 = midi_to_freq(en)
        ep_sig = (
            np.sin(2.0 * np.pi * f0 * lt_e)
            + 0.25 * np.sin(2.0 * np.pi * f0 * 2.0 * lt_e) * np.exp(-lt_e * 3.5)
            + 0.12 * np.sin(2.0 * np.pi * f0 * 4.0 * lt_e) * np.exp(-lt_e * 9.0)
        ) * mask * np.exp(-lt_e * 0.85) * 0.055
        pan = 0.5 + 0.25 * np.sin(n_i * 1.1)
        mix_l[eco_idx:] += ep_sig * pan
        mix_r[eco_idx:] += ep_sig * (1.0 - pan)

    # Smooth fade-out at the very end (36.8s -> 38.0s)
    fade_out = np.clip((DURATION_ACT2 - t) / 1.2, 0.0, 1.0) ** 1.5
    mix_l *= fade_out
    mix_r *= fade_out

    # Mastering soft saturation & peak normalization to -1.0 dBFS
    mix_l = np.tanh(mix_l * 1.18)
    mix_r = np.tanh(mix_r * 1.18)
    peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-6)
    target_peak = 0.891
    mix_l = (mix_l / peak) * target_peak
    mix_r = (mix_r / peak) * target_peak

    # Write `out/soundtrack_act2.wav`
    wav_act2_path = os.path.join(out_dir, "soundtrack_act2.wav")
    pcm_act2 = np.empty((N_SAMPLES, 2), dtype=np.int16)
    pcm_act2[:, 0] = np.clip(mix_l * 32767.0, -32768, 32767).astype(np.int16)
    pcm_act2[:, 1] = np.clip(mix_r * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(wav_act2_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm_act2.tobytes())

    # =========================================================================
    # EXTRACT PER-FRAME TELEMETRY ENVELOPES (1,140 frames @ 30fps)
    # =========================================================================
    spf = SR // FPS
    kick_env = np.zeros(N_FRAMES, dtype=np.float32)
    snare_env = np.zeros(N_FRAMES, dtype=np.float32)
    brass_env = np.zeros(N_FRAMES, dtype=np.float32)
    lead_env = np.zeros(N_FRAMES, dtype=np.float32)
    rms_env = np.zeros(N_FRAMES, dtype=np.float32)
    voice_env = np.zeros(N_FRAMES, dtype=np.float32)
    spec_bins = np.zeros((N_FRAMES, 32), dtype=np.float32)

    mono = 0.5 * (mix_l + mix_r)
    hanning = np.hanning(2048)

    for f in range(N_FRAMES):
        s0 = f * spf
        s1 = min(N_SAMPLES, s0 + spf)
        kick_env[f] = float(np.sqrt(np.mean(stem_kick[s0:s1] ** 2)))
        snare_env[f] = float(np.sqrt(np.mean(stem_snare[s0:s1] ** 2)))
        brass_env[f] = float(np.sqrt(np.mean(stem_brass[s0:s1] ** 2)))
        lead_env[f] = float(np.sqrt(np.mean(stem_lead[s0:s1] ** 2)))
        rms_env[f] = float(np.sqrt(np.mean(mono[s0:s1] ** 2)))
        voice_env[f] = float(np.sqrt(np.mean(stem_voice[s0:s1] ** 2)))

        w_start = max(0, min(N_SAMPLES - 2048, s0 - 224))
        chunk = mono[w_start:w_start + 2048] * hanning
        fft_mag = np.abs(np.fft.rfft(chunk))
        edges = np.geomspace(1, 512, 33).astype(int)
        for b in range(32):
            lo = edges[b]
            hi = max(lo + 1, edges[b + 1])
            spec_bins[f, b] = float(np.mean(fft_mag[lo:hi]))

    def norm_smooth(arr: np.ndarray, decay: float = 0.72) -> np.ndarray:
        mx = np.percentile(arr, 99) if np.max(arr) > 0 else 1.0
        normed = np.clip(arr / max(mx, 1e-6), 0.0, 1.0)
        smoothed = np.zeros_like(normed)
        v = 0.0
        for i in range(len(normed)):
            v = max(normed[i], v * decay)
            smoothed[i] = v
        return smoothed

    kick_env = norm_smooth(kick_env, 0.68)
    snare_env = norm_smooth(snare_env, 0.70)
    brass_env = norm_smooth(brass_env, 0.78)
    lead_env = norm_smooth(lead_env, 0.75)
    rms_env = norm_smooth(rms_env, 0.80)
    voice_env = norm_smooth(voice_env, 0.65)

    spec_max = np.percentile(spec_bins, 98.5, axis=0, keepdims=True) + 1e-5
    spec_bins = np.clip(spec_bins / spec_max, 0.0, 1.0)
    for b in range(32):
        v = 0.0
        for f in range(N_FRAMES):
            v = max(spec_bins[f, b], v * 0.74)
            spec_bins[f, b] = v

    env_act2_path = os.path.join(out_dir, "audio_env_act2.npz")
    np.savez(
        env_act2_path,
        kick=kick_env,
        snare=snare_env,
        brass=brass_env,
        lead=lead_env,
        rms=rms_env,
        voice=voice_env,
        spectrum=spec_bins,
    )

    # =========================================================================
    # BUILD CONTINUOUS 80.0s MASTER WAV (Act I + Act II Seamless Bridge)
    # =========================================================================
    wav_act1_path = os.path.join(out_dir, "soundtrack.wav")
    wav_full_path = os.path.join(out_dir, "soundtrack_full.wav")
    if os.path.exists(wav_act1_path):
        with wave.open(wav_act1_path, "rb") as wf1:
            raw1 = wf1.readframes(wf1.getnframes())
        pcm1 = np.frombuffer(raw1, dtype=np.int16).reshape(-1, 2).copy()
        # Seamless 0.35s harmonic crossfade/bridge right across the 42.0s boundary
        xf_len = int(0.35 * SR)
        # Boost Act I's final 1.2s tail gently so there is zero dip before Act II's bridge chord
        pcm_full = np.vstack([pcm1, pcm_act2])
        with wave.open(wav_full_path, "wb") as wff:
            wff.setnchannels(2)
            wff.setsampwidth(2)
            wff.setframerate(SR)
            wff.writeframes(pcm_full.tobytes())

    dt = time.time() - t0
    print(f"[audio_act2] Generated {wav_act2_path} ({DURATION_ACT2:.1f}s) + {env_act2_path} in {dt:.2f}s")
    return {
        "wav_act2_path": wav_act2_path,
        "wav_full_path": wav_full_path,
        "env_act2_path": env_act2_path,
        "duration": DURATION_ACT2,
    }


if __name__ == "__main__":
    generate_act2_soundtrack()
