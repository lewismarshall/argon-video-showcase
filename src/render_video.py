#!/usr/bin/env python3
"""
Master Parallel Video Builder & Poster Generator for ARGON // 18.
1. Generates the 42.0s 48kHz stereo procedural soundtrack + per-frame FFT/envelope telemetry.
2. Renders all 1,260 frames (1920x1080 @ 30fps) in parallel across all 8 CPU cores.
3. Streams raw RGB24 frames directly into `ffmpeg` (H.264 High Profile, CRF 16, yuv420p + AAC 256kbps).
4. Saves the high-impact poster frame (`out/poster.png`).
"""

import os
import sys
import time
import subprocess
import multiprocessing as mp
import numpy as np

# Ensure src directory is on path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from audio_synth import generate_soundtrack
from scenes import render_single_frame, WIDTH, HEIGHT, FPS, TOTAL_FRAMES

# Global worker state for zero-overhead multiprocessing
_WORKER_ENV = None


def _init_worker(env_path: str):
    global _WORKER_ENV
    _WORKER_ENV = dict(np.load(env_path))


def _render_worker(frame_idx: int) -> bytes:
    return render_single_frame(frame_idx, _WORKER_ENV, return_pil=False)


def build_showcase(out_dir: str = "/workspace/argon-video/out"):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # Step 1: Procedural Audio & Telemetry
    print("=== [1/4] Synthesizing Procedural Soundtrack & Audio Telemetry ===")
    audio_info = generate_soundtrack(out_dir)
    wav_path = audio_info["wav_path"]
    env_path = audio_info["env_path"]

    # Step 2: Export Poster Frame (out/poster.png)
    # Frame 120 (t = 4.00s) is the hero Act I Atomic Ignition + Title Lockup at full bloom
    print("=== [2/4] Rendering Hero Poster Frame (out/poster.png) ===")
    env_dict = dict(np.load(env_path))
    poster_img = render_single_frame(120, env_dict, return_pil=True)
    poster_path = os.path.join(out_dir, "poster.png")
    poster_img.save(poster_path, format="PNG", optimize=True)
    print(f"[poster] Saved {poster_path} ({WIDTH}x{HEIGHT})")

    # Step 3: Multi-core 1080p30 Frame Synthesis -> FFmpeg H.264 + AAC Muxing
    mp4_path = os.path.join(out_dir, "showcase.mp4")
    num_workers = min(8, os.cpu_count() or 4)
    print(f"=== [3/4] Rendering {TOTAL_FRAMES} Frames @ {WIDTH}x{HEIGHT} { int(FPS) }fps across {num_workers} CPU cores ===")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        # Video input: raw RGB24 pipe
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(int(FPS)),
        "-i", "-",
        # Audio input: procedural stereo WAV
        "-i", wav_path,
        # Encoding options: High-profile H.264 + high-bitrate AAC + web faststart
        "-c:v", "libx264",
        "-preset", "medium",
        "-profile:v", "high",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "256k",
        "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        mp4_path,
    ]

    t_render = time.time()
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    try:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(env_path,)) as pool:
            for idx, frame_bytes in enumerate(pool.imap(_render_worker, range(TOTAL_FRAMES), chunksize=12)):
                proc.stdin.write(frame_bytes)
                if (idx + 1) % 180 == 0 or (idx + 1) == TOTAL_FRAMES:
                    elapsed = time.time() - t_render
                    fps_rate = (idx + 1) / max(elapsed, 1e-3)
                    print(f"  -> Rendered & encoded {idx + 1:4d}/{TOTAL_FRAMES} frames ({(idx + 1)*100.0/TOTAL_FRAMES:5.1f}%) | {fps_rate:5.1f} fps")
    finally:
        if proc.stdin:
            proc.stdin.close()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with non-zero status {ret}")

    # Clean up temporary intermediate preview/wav/npz files if desired? Keep audio_env.npz & soundtrack.wav or clean preview
    total_dt = time.time() - t_start
    size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
    print(f"=== [4/4] Build Complete in {total_dt:.2f}s! Output: {mp4_path} ({size_mb:.2f} MB) ===")


if __name__ == "__main__":
    build_showcase()
