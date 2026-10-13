#!/usr/bin/env python3
"""
Procedural Synthesizer & Audio Reactivity Engine for ARGON // 18
Generates a 42.0-second stereo 48kHz WAV soundtrack at 120 BPM (84 beats, 21 bars)
in D Dorian / D Minor, plus an exact per-frame (1260 frames @ 30fps) audio telemetry
envelope file (`out/audio_env.npz`) for tight audio-visual synchronization.
"""

import os
import time
import wave
import numpy as np

SR = 48000
DURATION = 42.0
FPS = 30
N_SAMPLES = int(SR * DURATION)
N_FRAMES = int(FPS * DURATION)
BPM = 120.0
BEAT_SEC = 60.0 / BPM       # 0.5s
STEP_SEC = BEAT_SEC / 4.0   # 0.125s (16th note)


def midi_to_freq(midi_note: float) -> float:
    return 440.0 * (2.0 ** ((midi_note - 69.0) / 12.0))


def one_pole_lowpass(x: np.ndarray, cutoff_hz: np.ndarray, sr: int = SR) -> np.ndarray:
    """Time-varying one-pole lowpass filter implemented with fast NumPy block processing."""
    # Process in 256-sample blocks for speed while allowing smooth cutoff modulation
    out = np.zeros_like(x)
    block = 256
    y = 0.0
    n = len(x)
    for i in range(0, n, block):
        end = min(i + block, n)
        fc = float(cutoff_hz[i])
        alpha = np.clip(2.0 * np.pi * fc / (2.0 * np.pi * fc + sr), 0.005, 0.99)
        chunk = x[i:end]
        # Vectorized IIR within small block or tight loop
        for j in range(len(chunk)):
            y = y + alpha * (chunk[j] - y)
            out[i + j] = y
    return out


def stereo_reverb(left: np.ndarray, right: np.ndarray, sr: int = SR) -> tuple[np.ndarray, np.ndarray]:
    """Algorithmic stereo FDN/Schroeder-style lush space reverb using multi-tap prime delays."""
    wet_l = np.zeros_like(left)
    wet_r = np.zeros_like(right)
    mono = 0.5 * (left + right)
    # Prime-ish delay times in ms for lush stereo spread
    taps_l = [(29.7, 0.36), (43.1, 0.30), (67.3, 0.24), (97.9, 0.19), (137.3, 0.15), (193.1, 0.11), (269.7, 0.08), (359.3, 0.05)]
    taps_r = [(31.3, 0.36), (47.9, 0.30), (71.7, 0.24), (103.1, 0.19), (149.9, 0.15), (211.3, 0.11), (283.9, 0.08), (389.1, 0.05)]

    # Diffuse feedback pre-pass on mono signal
    diff = mono.copy()
    for d_ms, g in [(13.7, 0.45), (37.3, 0.40), (79.1, 0.35)]:
        d = int(d_ms * 0.001 * sr)
        padded = np.zeros_like(diff)
        padded[d:] = diff[:-d]
        diff = diff + g * padded

    for d_ms, gain in taps_l:
        d = int(d_ms * 0.001 * sr)
        wet_l[d:] += gain * diff[:-d]
    for d_ms, gain in taps_r:
        d = int(d_ms * 0.001 * sr)
        wet_r[d:] += gain * diff[:-d]

    return wet_l, wet_r


def generate_soundtrack(out_dir: str = "/workspace/argon-video/out") -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(18040)

    t = np.arange(N_SAMPLES, dtype=np.float64) / SR

    # Track buffers (Stereo L/R)
    mix_l = np.zeros(N_SAMPLES, dtype=np.float64)
    mix_r = np.zeros(N_SAMPLES, dtype=np.float64)
    rev_send_l = np.zeros(N_SAMPLES, dtype=np.float64)
    rev_send_r = np.zeros(N_SAMPLES, dtype=np.float64)

    # Telemetry stems at sample rate for accurate per-frame envelope extraction
    stem_kick = np.zeros(N_SAMPLES, dtype=np.float64)
    stem_snare = np.zeros(N_SAMPLES, dtype=np.float64)
    stem_arp = np.zeros(N_SAMPLES, dtype=np.float64)
    stem_bass = np.zeros(N_SAMPLES, dtype=np.float64)

    # =========================================================================
    # 1. CHORD PROGRESSION & WARM DETUNED UNISON PAD (D Dorian / D Minor)
    # =========================================================================
    # 21 bars (84 beats = 42.0s). Each chord spans 2 bars (4.0s), except final resolving chord.
    chord_schedule = [
        # (start_s, end_s, midi_notes, brightness)
        (0.0,  6.0,  [38, 50, 57, 60, 64], 0.35),       # Act I: Dm9 open fifth/ninth drone
        (6.0,  10.0, [38, 50, 53, 57, 64, 67], 0.50),   # Act IIa: Dm11
        (10.0, 14.0, [34, 46, 53, 57, 60, 65], 0.58),   # Act IIb: Bbmaj9
        (14.0, 18.0, [36, 48, 52, 55, 60, 64], 0.65),   # Act IIIa: C6/9
        (18.0, 22.0, [33, 45, 52, 55, 60, 64], 0.72),   # Act IIIb: Am7
        (22.0, 26.0, [38, 50, 53, 57, 62, 65], 0.85),   # Act IVa: Dm7 (Climax)
        (26.0, 30.0, [31, 43, 50, 53, 59, 62], 0.92),   # Act IVb: G13 (Dorian lift!)
        (30.0, 34.0, [34, 46, 50, 53, 57, 64], 0.80),   # Act Va: Bbmaj7(#11)
        (34.0, 36.0, [33, 45, 52, 55, 61, 64], 0.75),   # Act Vb: A7sus4 -> A7
        (36.0, 42.0, [38, 45, 50, 57, 62, 64, 69], 0.60) # Act VI: Dadd9 lush resolution
    ]

    pad_l = np.zeros(N_SAMPLES, dtype=np.float64)
    pad_r = np.zeros(N_SAMPLES, dtype=np.float64)

    for start_s, end_s, notes, bright in chord_schedule:
        # Overlap pad envelopes by 0.6s for seamless crossfade
        s_idx = max(0, int((start_s - 0.3) * SR))
        e_idx = min(N_SAMPLES, int((end_s + 0.6) * SR))
        dur = (e_idx - s_idx) / SR
        lt = np.arange(e_idx - s_idx, dtype=np.float64) / SR

        # Smooth attack and release envelope
        att = np.clip(lt / 0.8, 0.0, 1.0)
        rel = np.clip((dur - lt) / 0.9, 0.0, 1.0)
        env = (att ** 1.5) * (rel ** 1.5) * 0.055

        chord_sig_l = np.zeros_like(lt)
        chord_sig_r = np.zeros_like(lt)

        for n_i, note in enumerate(notes):
            f0 = midi_to_freq(note)
            # 3 detuned oscillators per note with stereo spread
            for det_cents, pan_l, pan_r in [(-9.0, 0.85, 0.35), (0.0, 0.65, 0.65), (+9.0, 0.35, 0.85)]:
                f = f0 * (2.0 ** (det_cents / 1200.0))
                phase = 2.0 * np.pi * f * lt + rng.uniform(0, 2 * np.pi)
                # Warm band-limited sawtooth approximation (fundamental + 2nd + 3rd + 4th harmonics)
                osc = (
                    np.sin(phase)
                    + (0.45 * bright) * np.sin(2.0 * phase + 0.3)
                    + (0.22 * bright) * np.sin(3.0 * phase + 0.7)
                    + (0.10 * bright) * np.sin(4.0 * phase + 1.1)
                )
                # Subtle LFO shimmer
                lfo = 1.0 + 0.08 * np.sin(2.0 * np.pi * (0.25 + 0.07 * n_i) * lt)
                chord_sig_l += osc * lfo * pan_l
                chord_sig_r += osc * lfo * pan_r

        pad_l[s_idx:e_idx] += chord_sig_l * env
        pad_r[s_idx:e_idx] += chord_sig_r * env

    mix_l += pad_l
    mix_r += pad_r
    rev_send_l += pad_l * 0.55
    rev_send_r += pad_r * 0.55

    # =========================================================================
    # 2. PUNCHY ELECTRONIC KICK, SNARE/CLAP & CRISP HI-HATS (120 BPM)
    # =========================================================================
    # Build sidechain ducking envelope triggered by kicks
    sidechain = np.ones(N_SAMPLES, dtype=np.float64)

    for beat in range(int(DURATION / BEAT_SEC)):
        beat_time = beat * BEAT_SEC
        # Kick pattern:
        # Act I (0-6s, beats 0..11): Deep heartbeat pulse on beats 0, 4, 8, 10
        # Act II (6-14s, beats 12..27): Groove kicks on every beat (four-on-the-floor) from beat 12
        # Act III-V (14-35s, beats 28..69): Full driving four-on-the-floor + syncopated ghost kicks
        # Act VI (36-42s, beats 72..83): Final Cinematic Impact on beat 72 (36.0s), then heartbeat at 76
        trigger_kick = False
        kick_gain = 1.0

        if beat in (0, 4, 8, 10):
            trigger_kick = True
            kick_gain = 0.85
        elif 12 <= beat < 71:
            # Pause drums briefly on beat 43 (21.5s) right before the 22.0s climax drop!
            if beat != 43:
                trigger_kick = True
                kick_gain = 1.0 if beat >= 28 else 0.88
        elif beat == 72:
            trigger_kick = True
            kick_gain = 1.15

        if trigger_kick:
            idx = int(beat_time * SR)
            k_len = int(0.38 * SR)
            end = min(N_SAMPLES, idx + k_len)
            kt = np.arange(end - idx, dtype=np.float64) / SR
            # Pitch envelope: 145 Hz -> 41.2 Hz (E1/F1 sub) rapid exponential sweep
            freq_env = 42.0 + 115.0 * np.exp(-kt * 38.0) + 180.0 * np.exp(-kt * 160.0)
            phase = 2.0 * np.pi * np.cumsum(freq_env) / SR
            amp_env = np.exp(-kt * 8.5) * (1.0 - np.exp(-kt * 900.0))
            # Transient click + warm tanh saturation
            click = 0.35 * np.exp(-kt * 260.0) * np.sin(2.0 * np.pi * 1600.0 * kt)
            kick_wave = np.tanh((np.sin(phase) * amp_env + click) * 1.45) * 0.48 * kick_gain
            mix_l[idx:end] += kick_wave
            mix_r[idx:end] += kick_wave
            stem_kick[idx:end] += np.abs(kick_wave)

            # Sidechain ducking curve (ducks bass/pads for 220ms)
            sc_len = min(N_SAMPLES - idx, int(0.28 * SR))
            sct = np.arange(sc_len, dtype=np.float64) / SR
            duck = 1.0 - 0.65 * kick_gain * np.exp(-sct * 14.0) * (1.0 - np.exp(-sct * 180.0))
            sidechain[idx:idx + sc_len] = np.minimum(sidechain[idx:idx + sc_len], duck)

        # Ghost 16th kick before downbeats in Acts III & IV
        if 28 <= beat < 68 and (beat % 4 == 3) and beat != 43:
            g_idx = int((beat_time + 3 * STEP_SEC) * SR)
            k_len = int(0.18 * SR)
            end = min(N_SAMPLES, g_idx + k_len)
            kt = np.arange(end - g_idx, dtype=np.float64) / SR
            freq_env = 46.0 + 95.0 * np.exp(-kt * 45.0)
            phase = 2.0 * np.pi * np.cumsum(freq_env) / SR
            kick_wave = np.tanh(np.sin(phase) * np.exp(-kt * 14.0) * 1.3) * 0.26
            mix_l[g_idx:end] += kick_wave
            mix_r[g_idx:end] += kick_wave
            stem_kick[g_idx:end] += np.abs(kick_wave)

        # Snare / Crisp Backbeat Clap on odd beats (beats 1, 3 of each bar) from 10.0s to 35.5s
        if 20 <= beat < 71 and (beat % 2 == 1):
            s_idx = int(beat_time * SR)
            s_len = int(0.26 * SR)
            end = min(N_SAMPLES, s_idx + s_len)
            st = np.arange(end - s_idx, dtype=np.float64) / SR
            # Tonal body (210 Hz -> 135 Hz) + filtered stereo noise snap
            body_f = 135.0 + 85.0 * np.exp(-st * 45.0)
            body = np.sin(2.0 * np.pi * np.cumsum(body_f) / SR) * np.exp(-st * 22.0) * 0.45
            noise_l = rng.normal(0, 1, len(st)) * np.exp(-st * 16.0) * 0.32
            noise_r = rng.normal(0, 1, len(st)) * np.exp(-st * 16.0) * 0.32
            sn_l = np.tanh((body + noise_l) * 1.3) * 0.30
            sn_r = np.tanh((body + noise_r) * 1.3) * 0.30
            mix_l[s_idx:end] += sn_l
            mix_r[s_idx:end] += sn_r
            rev_send_l[s_idx:end] += sn_l * 0.45
            rev_send_r[s_idx:end] += sn_r * 0.45
            stem_snare[s_idx:end] += 0.5 * (np.abs(sn_l) + np.abs(sn_r))

    # Apply sidechain ducking to pads
    mix_l *= sidechain
    mix_r *= sidechain

    # Crisp 16th-note metallic hi-hat groove (6.0s to 36.0s)
    total_steps = int(DURATION / STEP_SEC)
    for step in range(total_steps):
        step_time = step * STEP_SEC
        if 6.0 <= step_time < 36.0:
            # Skip during pre-drop silence (21.5s - 22.0s)
            if 21.5 <= step_time < 22.0:
                continue
            h_idx = int(step_time * SR)
            is_offbeat = (step % 4 == 2)
            h_dur = 0.11 if (is_offbeat and step_time >= 14.0) else 0.045
            h_len = int(h_dur * SR)
            end = min(N_SAMPLES, h_idx + h_len)
            ht = np.arange(end - h_idx, dtype=np.float64) / SR
            # Metallic FM/inharmonic square-ish partials + high-passed noise
            metal = (
                np.sin(2.0 * np.pi * 6340.0 * ht)
                + 0.7 * np.sin(2.0 * np.pi * 8910.0 * ht)
                + 0.5 * np.sin(2.0 * np.pi * 11470.0 * ht)
                + rng.normal(0, 0.9, len(ht))
            )
            vel = 0.085 if is_offbeat else (0.055 if step % 2 == 0 else 0.032)
            if step_time >= 22.0:
                vel *= 1.25
            decay = 24.0 if (is_offbeat and step_time >= 14.0) else 65.0
            hh = metal * np.exp(-ht * decay) * vel
            pan = 0.62 if (step % 2 == 0) else 0.38
            mix_l[h_idx:end] += hh * pan
            mix_r[h_idx:end] += hh * (1.0 - pan)

    # =========================================================================
    # 3. DRIVING REESE / FM SUB-BASS (8th/16th Pulse in D Dorian)
    # =========================================================================
    bass_l = np.zeros(N_SAMPLES, dtype=np.float64)
    bass_r = np.zeros(N_SAMPLES, dtype=np.float64)

    def root_midi_at(sec: float) -> int:
        if sec < 10.0:
            return 38  # D2
        elif sec < 14.0:
            return 34  # Bb1
        elif sec < 18.0:
            return 36  # C2
        elif sec < 22.0:
            return 33  # A1
        elif sec < 26.0:
            return 38  # D2
        elif sec < 30.0:
            return 31  # G1
        elif sec < 34.0:
            return 34  # Bb1
        elif sec < 36.0:
            return 33  # A1
        else:
            return 26  # D1 deep sub pedal

    for step in range(total_steps):
        st_sec = step * STEP_SEC
        if st_sec < 6.0:
            # Deep sustained sub drone on beat 0 and beat 8
            if step not in (0, 32):
                continue
            note_dur = 2.8
            root = 26 if step == 0 else 38
            b_idx = int(st_sec * SR)
            b_len = min(N_SAMPLES - b_idx, int(note_dur * SR))
            bt = np.arange(b_len, dtype=np.float64) / SR
            f0 = midi_to_freq(root)
            env = np.clip(bt / 0.15, 0.0, 1.0) * np.exp(-bt * 0.9)
            sig = (np.sin(2.0 * np.pi * f0 * bt) + 0.35 * np.tanh(2.5 * np.sin(2.0 * np.pi * f0 * 2.0 * bt))) * env * 0.32
            bass_l[b_idx:b_idx + b_len] += sig
            bass_r[b_idx:b_idx + b_idx + b_len - b_idx] += sig
            stem_bass[b_idx:b_idx + b_len] += np.abs(sig)
        elif 6.0 <= st_sec < 36.0:
            if 21.5 <= st_sec < 22.0:
                continue
            # Rolling 16th-note synthwave/progressive bassline
            root = root_midi_at(st_sec)
            octave = 12 if (step % 4 in (2, 3) and st_sec >= 14.0) else 0
            f0 = midi_to_freq(root + octave)
            b_idx = int(st_sec * SR)
            b_len = min(N_SAMPLES - b_idx, int(STEP_SEC * 0.95 * SR))
            bt = np.arange(b_len, dtype=np.float64) / SR
            # FM + Saw pluck bass
            mod_env = np.exp(-bt * 28.0) * (2.2 if st_sec >= 22.0 else 1.4)
            mod = np.sin(2.0 * np.pi * (f0 * 2.0) * bt) * mod_env
            carrier = np.sin(2.0 * np.pi * f0 * bt + mod)
            sub = np.sin(2.0 * np.pi * midi_to_freq(root) * bt)
            amp = np.exp(-bt * 11.0) * (1.0 - np.exp(-bt * 500.0)) * 0.25
            sig_l = np.tanh((0.65 * sub + 0.55 * carrier) * 1.4) * amp
            sig_r = np.tanh((0.65 * sub + 0.55 * np.sin(2.0 * np.pi * (f0 * 1.003) * bt + mod)) * 1.4) * amp
            bass_l[b_idx:b_idx + b_len] += sig_l
            bass_r[b_idx:b_idx + b_len] += sig_r
            stem_bass[b_idx:b_idx + b_len] += 0.5 * (np.abs(sig_l) + np.abs(sig_r))
        elif step == 288:  # 36.0s final resolution deep sub impact
            b_idx = int(36.0 * SR)
            b_len = min(N_SAMPLES - b_idx, int(5.5 * SR))
            bt = np.arange(b_len, dtype=np.float64) / SR
            f0 = midi_to_freq(38)  # D2
            f_sub = midi_to_freq(26)  # D1
            env = np.clip(bt / 0.05, 0.0, 1.0) * np.exp(-bt * 0.55)
            sig = (0.65 * np.sin(2.0 * np.pi * f_sub * bt) + 0.45 * np.sin(2.0 * np.pi * f0 * bt)) * env * 0.34
            bass_l[b_idx:b_idx + b_len] += sig
            bass_r[b_idx:b_idx + b_len] += sig
            stem_bass[b_idx:b_idx + b_len] += np.abs(sig)

    bass_l *= sidechain
    bass_r *= sidechain
    mix_l += bass_l
    mix_r += bass_r

    # =========================================================================
    # 4. CRYSTALLINE FM ARPEGGIATOR & LEAD MELODY (D Dorian)
    # =========================================================================
    arp_l = np.zeros(N_SAMPLES, dtype=np.float64)
    arp_r = np.zeros(N_SAMPLES, dtype=np.float64)

    # Motif scale notes in D Dorian (D, E, F, G, A, B, C) across octaves 4-6
    arp_patterns = {
        "dm":  [62, 65, 69, 72, 74, 76, 72, 69],
        "bb":  [62, 65, 70, 74, 77, 74, 70, 65],
        "c":   [64, 67, 72, 74, 76, 79, 76, 72],
        "am":  [64, 67, 69, 72, 76, 79, 76, 69],
        "g13": [62, 65, 67, 71, 74, 79, 76, 71],
        "a7":  [64, 67, 69, 73, 76, 81, 79, 73],
    }

    def arp_seq_at(sec: float) -> list[int]:
        if sec < 10.0:
            return arp_patterns["dm"]
        elif sec < 14.0:
            return arp_patterns["bb"]
        elif sec < 18.0:
            return arp_patterns["c"]
        elif sec < 22.0:
            return arp_patterns["am"]
        elif sec < 26.0:
            return arp_patterns["dm"]
        elif sec < 30.0:
            return arp_patterns["g13"]
        elif sec < 34.0:
            return arp_patterns["bb"]
        else:
            return arp_patterns["a7"]

    for step in range(total_steps):
        st_sec = step * STEP_SEC
        # Intro sparse crystalline pings
        if st_sec < 6.0:
            if step % 4 != 0:
                continue
            intro_notes = [74, 81, 76, 74, 86, 81, 77, 76, 74, 69, 72, 76]
            note = intro_notes[(step // 4) % len(intro_notes)]
            vel = 0.11
            dur_s = 0.45
        elif st_sec < 36.0:
            seq = arp_seq_at(st_sec)
            note = seq[step % len(seq)]
            # Octave lift in Act IV climax (22s - 30s)
            if 22.0 <= st_sec < 30.0 and (step % 4 == 0):
                note += 12
            vel = 0.105 if st_sec < 22.0 else 0.135
            # Filter/velocity swell during pre-drop (20.0 - 22.0s)
            if 20.0 <= st_sec < 22.0:
                vel *= 0.7 + 0.6 * ((st_sec - 20.0) / 2.0)
            dur_s = 0.24
        else:
            # Outro descending crystalline cascade (36.0s - 40.5s)
            if st_sec >= 40.5 or (step % 2 != 0):
                continue
            outro_notes = [86, 81, 76, 74, 69, 64, 62, 69, 74, 81, 86]
            idx_n = (step - 288) // 2
            if idx_n >= len(outro_notes):
                continue
            note = outro_notes[idx_n]
            vel = 0.11 * (1.0 - 0.06 * idx_n)
            dur_s = 0.65

        a_idx = int(st_sec * SR)
        a_len = min(N_SAMPLES - a_idx, int(dur_s * SR))
        at = np.arange(a_len, dtype=np.float64) / SR
        f0 = midi_to_freq(note)

        # Crystalline DX-style 2-op FM bell/pluck (ratio 1 : 2.0 and 1 : 3.0)
        mod_index = 2.6 * np.exp(-at * 18.0) + 0.25
        mod = mod_index * np.sin(2.0 * np.pi * (f0 * 2.0) * at) + (0.4 * mod_index) * np.sin(2.0 * np.pi * (f0 * 3.0) * at)
        env = (1.0 - np.exp(-at * 800.0)) * np.exp(-at * (7.5 / dur_s * 0.25))
        sig = np.sin(2.0 * np.pi * f0 * at + mod) * env * vel

        # Auto-pan across stereo field
        pan_angle = 0.5 + 0.35 * np.sin(step * 0.65)
        arp_l[a_idx:a_idx + a_len] += sig * pan_angle
        arp_r[a_idx:a_idx + a_len] += sig * (1.0 - pan_angle)
        stem_arp[a_idx:a_idx + a_len] += np.abs(sig)

    # Ping-pong dotted-8th stereo delay on arpeggiator (3/16th note = 0.375s)
    delay_samples = int(3 * STEP_SEC * SR)
    del_l = np.zeros_like(arp_l)
    del_r = np.zeros_like(arp_r)
    del_l[delay_samples:] = arp_r[:-delay_samples] * 0.42
    del_r[delay_samples:] = arp_l[:-delay_samples] * 0.42
    del_l[2 * delay_samples:] += arp_l[:-2 * delay_samples] * 0.20
    del_r[2 * delay_samples:] += arp_r[:-2 * delay_samples] * 0.20

    mix_l += arp_l + del_l
    mix_r += arp_r + del_r
    rev_send_l += (arp_l + del_l) * 0.45
    rev_send_r += (arp_r + del_r) * 0.45

    # =========================================================================
    # 5. SOARING ANALOG LEAD SYNTH IN ACT IV & V (22.0s - 36.0s)
    # =========================================================================
    lead_notes = [
        # (start_s, dur_s, midi_note)
        (22.0, 1.5, 74), (23.5, 0.5, 76), (24.0, 1.5, 77), (25.5, 0.5, 76),
        (26.0, 1.5, 81), (27.5, 0.5, 79), (28.0, 1.5, 76), (29.5, 0.5, 74),
        (30.0, 1.5, 77), (31.5, 0.5, 81), (32.0, 2.0, 84),
        (34.0, 1.0, 81), (35.0, 1.0, 85), (36.0, 3.5, 86),
    ]
    for st_s, dur_s, l_note in lead_notes:
        l_idx = int(st_s * SR)
        l_len = min(N_SAMPLES - l_idx, int((dur_s + 0.15) * SR))
        lt = np.arange(l_len, dtype=np.float64) / SR
        f0 = midi_to_freq(l_note)
        # Vibrato after 0.25s
        vib = 0.004 * np.clip((lt - 0.2) / 0.3, 0.0, 1.0) * np.sin(2.0 * np.pi * 5.8 * lt)
        phase = 2.0 * np.pi * f0 * np.cumsum(1.0 + vib) / SR
        # Rich unison saw lead
        osc_l = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.28 * np.sin(3 * phase) + 0.15 * np.sin(4 * phase)
        osc_r = np.sin(phase * 1.002) + 0.5 * np.sin(2 * phase * 1.002) + 0.28 * np.sin(3 * phase * 1.002)
        env = np.clip(lt / 0.04, 0.0, 1.0) * np.clip((dur_s + 0.15 - lt) / 0.18, 0.0, 1.0) * 0.085
        mix_l[l_idx:l_idx + l_len] += osc_l * env * sidechain[l_idx:l_idx + l_len]
        mix_r[l_idx:l_idx + l_len] += osc_r * env * sidechain[l_idx:l_idx + l_len]
        rev_send_l[l_idx:l_idx + l_len] += osc_l * env * 0.65
        rev_send_r[l_idx:l_idx + l_len] += osc_r * env * 0.65

    # =========================================================================
    # 6. CINEMATIC TRANSITION RISERS & SUB DROPS AT SCENE BOUNDARIES
    # =========================================================================
    transitions = [6.0, 14.0, 22.0, 30.0, 36.0]
    for tr_s in transitions:
        # 1.8s riser leading into tr_s
        r_start = max(0, int((tr_s - 1.8) * SR))
        r_end = int(tr_s * SR)
        rt = np.arange(r_end - r_start, dtype=np.float64) / SR
        prog = rt / 1.8
        # Pitch riser sweep (220 Hz -> 1760 Hz) + stereo noise whoosh
        f_sweep = 220.0 * (2.0 ** (prog * 3.0))
        ph = 2.0 * np.pi * np.cumsum(f_sweep) / SR
        riser_l = (0.4 * np.sin(ph) + 0.6 * rng.normal(0, 1, len(rt))) * (prog ** 2.2) * 0.11
        riser_r = (0.4 * np.cos(ph) + 0.6 * rng.normal(0, 1, len(rt))) * (prog ** 2.2) * 0.11
        mix_l[r_start:r_end] += riser_l
        mix_r[r_start:r_end] += riser_r
        rev_send_l[r_start:r_end] += riser_l * 0.6
        rev_send_r[r_start:r_end] += riser_r * 0.6

        # Impact shimmer & sub drop right at tr_s
        i_start = int(tr_s * SR)
        i_end = min(N_SAMPLES, int((tr_s + 1.4) * SR))
        it = np.arange(i_end - i_start, dtype=np.float64) / SR
        sub_drop = np.sin(2.0 * np.pi * np.cumsum(90.0 * np.exp(-it * 2.8) + 32.0) / SR) * np.exp(-it * 2.5) * 0.25
        crash_l = rng.normal(0, 1, len(it)) * np.exp(-it * 3.2) * 0.08
        crash_r = rng.normal(0, 1, len(it)) * np.exp(-it * 3.2) * 0.08
        mix_l[i_start:i_end] += sub_drop + crash_l
        mix_r[i_start:i_end] += sub_drop + crash_r
        rev_send_l[i_start:i_end] += crash_l * 0.8
        rev_send_r[i_start:i_end] += crash_r * 0.8

    # =========================================================================
    # 7. STEREO REVERB, MASTERING LIMITER & FADE IN/OUT
    # =========================================================================
    wet_l, wet_r = stereo_reverb(rev_send_l, rev_send_r, SR)
    mix_l += wet_l * 0.38
    mix_r += wet_r * 0.38

    # Master fade-in (first 0.35s) and smooth fade-out (last 1.8s: 40.2s -> 42.0s)
    fade_in = np.clip(t / 0.35, 0.0, 1.0)
    fade_out = np.clip((DURATION - t) / 1.8, 0.0, 1.0) ** 1.6
    master_env = fade_in * fade_out
    mix_l *= master_env
    mix_r *= master_env

    # Soft mastering bus saturation & peak normalization to -1.0 dBFS
    mix_l = np.tanh(mix_l * 1.15)
    mix_r = np.tanh(mix_r * 1.15)
    peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r)), 1e-6)
    target_peak = 0.891  # -1.0 dBFS
    mix_l = (mix_l / peak) * target_peak
    mix_r = (mix_r / peak) * target_peak

    # Write 16-bit PCM Stereo WAV
    wav_path = os.path.join(out_dir, "soundtrack.wav")
    stereo_pcm = np.empty((N_SAMPLES, 2), dtype=np.int16)
    stereo_pcm[:, 0] = np.clip(mix_l * 32767.0, -32768, 32767).astype(np.int16)
    stereo_pcm[:, 1] = np.clip(mix_r * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(stereo_pcm.tobytes())

    # =========================================================================
    # 8. PER-FRAME AUDIO TELEMETRY EXTRACTION (1260 frames @ 30fps)
    # =========================================================================
    samples_per_frame = SR // FPS  # 1600 samples per frame
    kick_env = np.zeros(N_FRAMES, dtype=np.float32)
    snare_env = np.zeros(N_FRAMES, dtype=np.float32)
    arp_env = np.zeros(N_FRAMES, dtype=np.float32)
    bass_env = np.zeros(N_FRAMES, dtype=np.float32)
    rms_env = np.zeros(N_FRAMES, dtype=np.float32)
    spec_bins = np.zeros((N_FRAMES, 32), dtype=np.float32)

    mono_master = 0.5 * (mix_l + mix_r)
    hanning = np.hanning(2048)

    for f in range(N_FRAMES):
        s0 = f * samples_per_frame
        s1 = min(N_SAMPLES, s0 + samples_per_frame)
        kick_env[f] = float(np.sqrt(np.mean(stem_kick[s0:s1] ** 2)))
        snare_env[f] = float(np.sqrt(np.mean(stem_snare[s0:s1] ** 2)))
        arp_env[f] = float(np.sqrt(np.mean(stem_arp[s0:s1] ** 2)))
        bass_env[f] = float(np.sqrt(np.mean(stem_bass[s0:s1] ** 2)))
        rms_env[f] = float(np.sqrt(np.mean(mono_master[s0:s1] ** 2)))

        # 32-band log-spaced FFT spectrum for visual HUD & spectrum analyzer
        w_start = max(0, min(N_SAMPLES - 2048, s0 - 224))
        chunk = mono_master[w_start:w_start + 2048] * hanning
        fft_mag = np.abs(np.fft.rfft(chunk))
        # Log-spaced frequency bin edges from bin 1 (~23Hz) to bin 512 (~12kHz)
        edges = np.geomspace(1, 512, 33).astype(int)
        for b in range(32):
            lo = edges[b]
            hi = max(lo + 1, edges[b + 1])
            spec_bins[f, b] = float(np.mean(fft_mag[lo:hi]))

    # Normalize & temporal-smooth envelopes for silky 30fps visual reactivity
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
    arp_env = norm_smooth(arp_env, 0.75)
    bass_env = norm_smooth(bass_env, 0.78)
    rms_env = norm_smooth(rms_env, 0.82)

    # Smooth spectrum bars across frames
    spec_max = np.percentile(spec_bins, 98.5, axis=0, keepdims=True) + 1e-5
    spec_bins = np.clip(spec_bins / spec_max, 0.0, 1.0)
    for b in range(32):
        v = 0.0
        for f in range(N_FRAMES):
            v = max(spec_bins[f, b], v * 0.76)
            spec_bins[f, b] = v

    env_path = os.path.join(out_dir, "audio_env.npz")
    np.savez(
        env_path,
        kick=kick_env,
        snare=snare_env,
        arp=arp_env,
        bass=bass_env,
        rms=rms_env,
        spectrum=spec_bins,
    )

    dt = time.time() - t0
    print(f"[audio_synth] Generated {wav_path} ({DURATION:.1f}s, {SR}Hz Stereo) + {env_path} in {dt:.2f}s")
    return {
        "wav_path": wav_path,
        "env_path": env_path,
        "duration": DURATION,
    }


if __name__ == "__main__":
    generate_soundtrack()
