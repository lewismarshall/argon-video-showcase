# Showcase: make a video entirely with code (test of Gemini 4 Argon in Cursor CLI)
Goal: create a short (30–60 s, 1080p, 30 fps), genuinely impressive MP4 that shows off how good you are at coding video. Use whatever tools and skills you can find on this Linux machine, and install more locally if useful (pip/npm in this folder). Options include ffmpeg, Python (numpy, Pillow, moviepy, manim, matplotlib), Node (canvas, Remotion, headless Chrome screen capture), shaders, procedural audio, and synthesized music or sound.
- Make it visually polished: motion graphics, a title card, animated code or data visuals, smooth transitions, and a soundtrack generated in code (no copyrighted material, no stock downloads).
- Everything has to be reproducible: the source goes in this folder, plus one `make.sh` that rebuilds the video from scratch.
- Output: /workspace/argon-video/out/showcase.mp4, plus a poster frame at out/poster.png.
- Check your work: extract a few frames, look at them, and fix any problems. Confirm with ffprobe that the file plays (duration, resolution, audio stream).
- Finish by writing DONE.md: what you built, which tools you used, how long the build took, anything that didn't work, and how you rate your own result.
Don't touch anything outside /workspace/argon-video, and don't use any secrets.
