#!/usr/bin/env bash
set -euo pipefail

# Navigate to repository root
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

echo "=================================================================="
echo "  ARGON // 18 — Full Procedural Audio-Visual Pipeline (Act I + II)"
echo "=================================================================="

mkdir -p out

# Run the full multi-core procedural audio + 1080p30 video synthesis pipeline
python3 src/render_video.py

# Clean up temporary audio/telemetry intermediates so out/ contains only the final deliverables
rm -f out/soundtrack.wav out/soundtrack_act2.wav out/soundtrack_full.wav
rm -f out/audio_env.npz out/audio_env_act2.npz
rm -rf out/preview out/preview2

echo "=================================================================="
echo "  Verifying output streams with ffprobe:"
echo "=================================================================="
for f in out/showcase.mp4 out/act2.mp4 out/showcase-full.mp4; do
    echo "--- $f ---"
    ffprobe -v error -show_entries format=filename,duration,size,bit_rate:stream=index,codec_name,codec_type,width,height,r_frame_rate,sample_rate,channels -of default=noprint_wrappers=1 "$f"
done
