#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

echo "=================================================================="
echo "  ARGON // 18 — Procedural Audio-Visual Synthesis Pipeline"
echo "=================================================================="

mkdir -p out

# Run the full multi-core procedural audio + 1080p30 video synthesis pipeline
python3 src/render_video.py

# Clean up any temporary audio/telemetry intermediates so out/ stays clean
rm -f out/soundtrack.wav out/audio_env.npz
rm -rf out/preview

echo "=================================================================="
echo "  Verifying output streams with ffprobe:"
echo "=================================================================="
ffprobe -v error -show_entries format=filename,duration,size,bit_rate:stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels -of default=noprint_wrappers=1 out/showcase.mp4
