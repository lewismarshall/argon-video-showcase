#!/usr/bin/env python3
"""
Master Parallel Video Builder & Poster Generator for ARGON // 18 (Act I + Act II).
1. Synthesizes Act I (42.0s) and Act II (38.0s) procedural 48kHz stereo soundtracks
   plus the continuous 80.0s full master (`soundtrack_full.wav`) and per-frame telemetry.
2. Compiles the native C/OpenMP/AVX-512 3D PBR Raymarcher (`src/raymarch_act2.c` -> `src/libact2_ray.so`).
3. Renders Act I (`out/showcase.mp4`, 1,260 frames @ 1080p30) and `out/poster.png`.
4. Morphs seamlessly from the exact last frame of Act I into 3D PBR raymarching to render
   Act II (`out/act2.mp4`, 1,140 frames @ 1080p30) and `out/act2-poster.png`.
5. Combines Act I + Act II with the continuous 80.0s master audio into `out/showcase-full.mp4`.
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
from audio_act2 import generate_act2_soundtrack
from scenes import render_single_frame, WIDTH, HEIGHT, FPS, TOTAL_FRAMES
from scenes_act2 import render_act2_single_frame, TOTAL_FRAMES_ACT2

# Global worker state for zero-overhead multiprocessing
_WORKER_ENV = None
_WORKER_ENV_ACT2 = None
_WORKER_ACT1_LAST_RGB = None


def _init_worker_act1(env_path: str):
    global _WORKER_ENV
    _WORKER_ENV = dict(np.load(env_path))


def _render_worker_act1(frame_idx: int) -> bytes:
    return render_single_frame(frame_idx, _WORKER_ENV, return_pil=False)


def _init_worker_act2(env_path: str, act1_last_path: str, omp_threads: int):
    global _WORKER_ENV_ACT2, _WORKER_ACT1_LAST_RGB
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    _WORKER_ENV_ACT2 = dict(np.load(env_path))
    _WORKER_ACT1_LAST_RGB = np.ascontiguousarray(np.load(act1_last_path))


def _render_worker_act2(frame_idx: int) -> bytes:
    return render_act2_single_frame(
        frame_idx, _WORKER_ENV_ACT2, _WORKER_ACT1_LAST_RGB, return_pil=False
    )


def compile_native_raymarcher():
    c_src = os.path.join(SRC_DIR, "raymarch_act2.c")
    so_out = os.path.join(SRC_DIR, "libact2_ray.so")
    cmd = [
        "gcc",
        "-O3",
        "-ffast-math",
        "-march=native",
        "-fopenmp",
        "-shared",
        "-fPIC",
        c_src,
        "-o",
        so_out,
        "-lm",
    ]
    print(f"[gcc] Compiling C/OpenMP/AVX-512 3D PBR Raymarcher: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _open_ffmpeg_writer(mp4_path: str, wav_path: str) -> subprocess.Popen:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(int(FPS)),
        "-i", "-",
        "-i", wav_path,
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
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


def build_all(out_dir: str = "/workspace/argon-video/out"):
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Compile Native C/OpenMP/AVX-512 3D Raymarcher
    # -------------------------------------------------------------------------
    print("=== [1/6] Compiling Native 3D PBR Raymarcher (libact2_ray.so) ===")
    compile_native_raymarcher()

    # -------------------------------------------------------------------------
    # Step 2: Synthesize Act I, Act II, and Continuous Full Soundtracks
    # -------------------------------------------------------------------------
    print("=== [2/6] Synthesizing Act I (42s), Act II (38s) & Full (80s) Soundtracks ===")
    audio1 = generate_soundtrack(out_dir)
    wav1_path = audio1["wav_path"]
    env1_path = audio1["env_path"]

    audio2 = generate_act2_soundtrack(out_dir)
    wav2_path = audio2["wav_act2_path"]
    env2_path = audio2["env_act2_path"]
    wav_full_path = audio2["wav_full_path"]

    # -------------------------------------------------------------------------
    # Step 3: Render Act I Poster (out/poster.png) & Act I Video (out/showcase.mp4)
    # -------------------------------------------------------------------------
    print("=== [3/6] Rendering Act I Poster (out/poster.png) & Video (out/showcase.mp4) ===")
    env1_dict = dict(np.load(env1_path))
    poster1_img = render_single_frame(120, env1_dict, return_pil=True)
    poster1_path = os.path.join(out_dir, "poster.png")
    poster1_img.save(poster1_path, format="PNG", optimize=True)
    print(f"[poster-1] Saved {poster1_path} ({WIDTH}x{HEIGHT})")

    # Also capture exact final frame (Frame 1259) of Act I for Act II's 3D morph
    act1_last_pil = render_single_frame(TOTAL_FRAMES - 1, env1_dict, return_pil=True)
    act1_last_rgb = np.ascontiguousarray(np.asarray(act1_last_pil, dtype=np.uint8))
    act1_last_npy = os.path.join(out_dir, "act1_last_rgb.npy")
    np.save(act1_last_npy, act1_last_rgb)

    mp4_act1_path = os.path.join(out_dir, "showcase.mp4")
    num_workers = min(8, os.cpu_count() or 4)
    t_act1 = time.time()
    proc1 = _open_ffmpeg_writer(mp4_act1_path, wav1_path)
    try:
        with mp.Pool(processes=num_workers, initializer=_init_worker_act1, initargs=(env1_path,)) as pool:
            for idx, frame_bytes in enumerate(pool.imap(_render_worker_act1, range(TOTAL_FRAMES), chunksize=12)):
                proc1.stdin.write(frame_bytes)
                if (idx + 1) % 210 == 0 or (idx + 1) == TOTAL_FRAMES:
                    elapsed = time.time() - t_act1
                    fps_rate = (idx + 1) / max(elapsed, 1e-3)
                    print(f"  [Act I]  {idx + 1:4d}/{TOTAL_FRAMES} frames ({(idx + 1)*100.0/TOTAL_FRAMES:5.1f}%) | {fps_rate:5.1f} fps")
    finally:
        if proc1.stdin:
            proc1.stdin.close()
        ret1 = proc1.wait()
        if ret1 != 0:
            raise RuntimeError(f"ffmpeg (Act I) exited with status {ret1}")

    # -------------------------------------------------------------------------
    # Step 4: Render Act II Poster (out/act2-poster.png)
    # -------------------------------------------------------------------------
    print("=== [4/6] Rendering Act II Hero Poster (out/act2-poster.png) ===")
    os.environ["OMP_NUM_THREADS"] = str(num_workers)
    env2_dict = dict(np.load(env2_path))
    # Frame 540 (t = 18.00s): 3D Liquid-Chrome & Noble-Gas Plasma Singularity ("GOOGLE IS BACK.")
    poster2_img = render_act2_single_frame(540, env2_dict, act1_last_rgb, return_pil=True)
    poster2_path = os.path.join(out_dir, "act2-poster.png")
    poster2_img.save(poster2_path, format="PNG", optimize=True)
    print(f"[poster-2] Saved {poster2_path} ({WIDTH}x{HEIGHT})")

    # -------------------------------------------------------------------------
    # Step 5: Render Act II Video (out/act2.mp4, 1,140 frames @ 1080p30)
    # -------------------------------------------------------------------------
    mp4_act2_path = os.path.join(out_dir, "act2.mp4")
    print(f"=== [5/6] Rendering Act II ({TOTAL_FRAMES_ACT2} frames @ 1080p30) [{num_workers} AVX-512 OpenMP threads] ===", flush=True)
    t_act2 = time.time()
    proc2 = _open_ffmpeg_writer(mp4_act2_path, wav2_path)
    try:
        for idx in range(TOTAL_FRAMES_ACT2):
            frame_bytes = render_act2_single_frame(idx, env2_dict, act1_last_rgb, return_pil=False)
            proc2.stdin.write(frame_bytes)
            if (idx + 1) % 95 == 0 or (idx + 1) == TOTAL_FRAMES_ACT2:
                elapsed = time.time() - t_act2
                fps_rate = (idx + 1) / max(elapsed, 1e-3)
                print(
                    f"  [Act II] {idx + 1:4d}/{TOTAL_FRAMES_ACT2} frames ({(idx + 1)*100.0/TOTAL_FRAMES_ACT2:5.1f}%) | {fps_rate:5.1f} fps",
                    flush=True,
                )
    finally:
        if proc2.stdin:
            proc2.stdin.close()
        ret2 = proc2.wait()
        if ret2 != 0:
            raise RuntimeError(f"ffmpeg (Act II) exited with status {ret2}")

    # -------------------------------------------------------------------------
    # Step 6: Assemble Full Combined Showcase (out/showcase-full.mp4, 80.0s)
    # -------------------------------------------------------------------------
    print("=== [6/6] Assembling Combined 80.0s Master (out/showcase-full.mp4) ===")
    mp4_full_path = os.path.join(out_dir, "showcase-full.mp4")
    concat_list_path = os.path.join(out_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        f.write(f"file '{mp4_act1_path}'\n")
        f.write(f"file '{mp4_act2_path}'\n")

    concat_cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_path,
        "-i", wav_full_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "256k",
        "-ar", "48000",
        "-shortest",
        "-movflags", "+faststart",
        mp4_full_path,
    ]
    subprocess.run(concat_cmd, check=True)
    os.remove(concat_list_path)
    os.remove(act1_last_npy)

    total_dt = time.time() - t_start
    s1 = os.path.getsize(mp4_act1_path) / (1024 * 1024)
    s2 = os.path.getsize(mp4_act2_path) / (1024 * 1024)
    sf = os.path.getsize(mp4_full_path) / (1024 * 1024)
    print(f"=== Build Complete in {total_dt:.2f}s! ===")
    print(f"  -> Act I  : {mp4_act1_path} ({s1:.2f} MB)")
    print(f"  -> Act II : {mp4_act2_path} ({s2:.2f} MB)")
    print(f"  -> Full   : {mp4_full_path} ({sf:.2f} MB)")


if __name__ == "__main__":
    build_all()
