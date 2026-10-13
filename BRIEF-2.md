# Brief 2: ARGON // 18 Act II ("Google is back")
Extend the existing showcase. Make a NEW continuation (~30-45 s, 1080p30) that starts seamlessly by MORPHING from the exact last frame of out/showcase.mp4, then:
- Moves into 3D, as photorealistic as you can manage in code: path-traced or PBR raymarching (SDFs, volumetric fog, caustics, depth of field, motion blur, film grain). Find and use any tools you can (Blender headless/bpy, Taichi, moderngl/EGL, numpy raymarchers, etc.) and install locally inside this folder if needed (venv). CPU-only box, so keep render time sensible (budget ~30 min of rendering); lower samples and denoise rather than skip.
- Gets more surreal, dark and "wow": impossible geometry, liquid chrome, noble-gas plasma, things morphing. The message: Google is back in the game.
- Soundtrack: take it up a notch from the act 1 music (same key/tempo family so it flows), darker and bigger, with a cinematic build and drop. Still 100% generated in code.
- Ending: an ironic, humorous end screen involving Google Home products (e.g. a Nest speaker answering the big cinematic finale with "Sorry, I didn't understand that", or a Nest thermostat set to a sensible 19 °C). Celsius only. Draw the products yourself in code (no logos, images or downloaded assets).
- Output: out/act2.mp4 and a combined out/showcase-full.mp4 (act 1 + act 2, continuous audio), plus out/act2-poster.png. Extend make.sh so it rebuilds everything.
- Verify: check frames at the morph point and ending, and run ffprobe.
- Write DONE.md (overwrite) covering what you did, tools, render timings, problems and a self-rating. Commit source and push to main (large mp4s stay gitignored; never force-push). Nothing outside this folder, no secrets.
