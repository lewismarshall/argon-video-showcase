#!/usr/bin/env python3
"""
Scene Compositor & Smart-Home Epilogue Renderer for ARGON // 18 — ACT II ("Google is back").
Combines:
- Phases 1..3 (0.0s - 29.5s): Native C/OpenMP/AVX-512 3D PBR Raymarcher (`libact2_ray.so`)
  with seamless 3D morphing from the exact last frame of `out/showcase.mp4`, liquid chrome,
  impossible geometry, noble-gas plasma, floor caustics, DoF, motion blur, film grain, and
  cinematic kinetic typography ("GOOGLE IS BACK").
- Phase 4 (29.5s - 38.0s): Ironic Smart-Home Epilogue featuring a code-drawn Nest Audio speaker
  answering the cosmic singularity finale with "Sorry, I didn't understand that..." alongside a
  code-drawn wall-mounted Nest Learning Thermostat rotating down to a sensible 19.0 °C (Celsius only).
"""

import os
import math
import ctypes
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageFilter

from cairo_engine import (
    CairoCanvas,
    draw_text,
    finalize_frame,
    lerp,
    clamp,
    smoothstep,
    ease_out_cubic,
    ease_in_out_cubic,
    ease_out_expo,
    lerp_color,
    COL_BG_DEEP,
    COL_BG_MID,
    COL_CYAN,
    COL_VIOLET,
    COL_MAGENTA,
    COL_AMBER,
    COL_MINT,
    COL_WHITE,
    COL_SLATE,
    COL_DARK_PANEL,
)

WIDTH = 1920
HEIGHT = 1080
FPS = 30.0
DURATION_ACT2 = 38.0
TOTAL_FRAMES_ACT2 = int(FPS * DURATION_ACT2)  # 1,140 frames

# Spectral Google 4-Color Accents (Normalized RGB)
COL_G_BLUE   = (0.259, 0.522, 0.957)  # #4285F4
COL_G_RED    = (0.918, 0.263, 0.208)  # #EA4335
COL_G_YELLOW = (0.984, 0.737, 0.020)  # #FBBC05
COL_G_GREEN  = (0.204, 0.659, 0.325)  # #34A853

_RAY_LIB = None


def get_raymarch_lib() -> ctypes.CDLL:
    """Compiles (if needed) and loads `libact2_ray.so`."""
    global _RAY_LIB
    if _RAY_LIB is not None:
        return _RAY_LIB

    src_dir = os.path.dirname(os.path.abspath(__file__))
    c_path = os.path.join(src_dir, "raymarch_act2.c")
    so_path = os.path.join(src_dir, "libact2_ray.so")

    if not os.path.exists(so_path) or os.path.getmtime(c_path) > os.path.getmtime(so_path):
        subprocess.check_call([
            "gcc", "-O3", "-ffast-math", "-march=native", "-fopenmp",
            "-shared", "-fPIC", c_path, "-o", so_path, "-lm"
        ])

    lib = ctypes.CDLL(so_path)
    lib.render_act2_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
    ]
    _RAY_LIB = lib
    return lib


class _TextRecorder:
    def __init__(self):
        self.ops = []

    def text(self, xy, text, font=None, fill=None, anchor=None):
        self.ops.append((xy, text, font, fill, anchor))

    def replay(self, pil_draw: ImageDraw.ImageDraw):
        for xy, text, font, fill, anchor in self.ops:
            pil_draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_act2_3d_hud_and_titles(
    pil_img: Image.Image,
    frame_idx: int,
    t: float,
    kick: float,
    snare: float,
    brass: float,
    lead: float,
    rms: float,
    spec: np.ndarray,
) -> Image.Image:
    """
    Overlays minimal, high-contrast cinema scope bars, 3D telemetry HUD, and monumental
    kinetic typography onto the 3D PBR raymarched frame (for t in 0.0s .. 29.5s).
    At t == 0.0s (Frame 0), nothing extra is drawn so Frame 0 is 100% pixel-identical
    to the last frame of `out/showcase.mp4`.
    """
    if frame_idx == 0:
        return pil_img

    overlay_fade = smoothstep(0.15, 1.2, t)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # Subtle letterbox / broadcast HUD hairlines that morph from Act I's HUD
    hud_alpha = int(210 * overlay_fade)
    draw.line([(64, 58), (WIDTH - 64, 58)], fill=(115, 135, 173, int(90 * overlay_fade)), width=1)
    draw.line([(64, HEIGHT - 58), (WIDTH - 64, HEIGHT - 58)], fill=(115, 135, 173, int(90 * overlay_fade)), width=1)

    # 4-Color Spectral Progress Bar along top hairline
    prog_w = int((WIDTH - 128) * (t / DURATION_ACT2))
    if prog_w > 4:
        draw.line([(64, 58), (64 + prog_w, 58)], fill=(0, 240, 255, hud_alpha), width=3)

    # Determine chapter title
    if t < 5.5:
        chap = "ACT II.1 // DIMENSIONAL BREACH (2D -> 3D SDF EXTRUSION)"
    elif t < 14.0:
        chap = "ACT II.2 // IMPOSSIBLE TOPOLOGY & 4D TESSERACT"
    elif t < 22.5:
        chap = "ACT II.3 // THE AWAKENING — LIQUID CHROME & PLASMA"
    else:
        chap = "ACT II.4 // GEMINI 4 ARGON SINGULARITY"

    draw_text(draw, (68, 26), "SYS // ARGON-18 RAYMARCHER [AVX-512 PBR]", "plex_bold", 14, 700, COL_WHITE, 0.88 * overlay_fade, tracking=1.8)
    draw_text(draw, (475, 26), chap, "plex_mono", 14, 500, COL_CYAN, 0.88 * overlay_fade, tracking=1.4)

    # Continuous timecode (continuing from 00:00:42:00 / FRM 1260)
    full_t = 42.0 + t
    mins = int(full_t // 60)
    secs = int(full_t % 60)
    frms = int(frame_idx % 30)
    tc_str = f"TC 00:{mins:02d}:{secs:02d}:{frms:02d}  |  ACT2 FRM {frame_idx:04d}/1140  |  PBR+DOF+CAUSTICS"
    draw_text(draw, (WIDTH - 68, 26), tc_str, "plex_mono", 14, 500, COL_SLATE, 0.88 * overlay_fade, anchor="ra", tracking=1.0)

    # Bottom-left 3D Shader Telemetry Readout
    shader_str = f"BRDF: COOK-TORRANCE GGX  //  BOUNCES: 2  //  VOLUMETRIC STEPS: 48  //  CoC DOF: ACTIVE"
    draw_text(draw, (68, HEIGHT - 42), shader_str, "plex_mono", 13, 500, COL_SLATE, 0.80 * overlay_fade, tracking=1.0)

    # Bottom-right 4-Color Spectral Equalizer Bars
    base_x = WIDTH - 68 - 24 * 8
    base_y = HEIGHT - 22
    g_cols = [COL_G_BLUE, COL_G_RED, COL_G_YELLOW, COL_G_GREEN]
    for b in range(24):
        val = float(spec[b])
        bh = max(3.0, val * 24.0)
        bx = base_x + b * 8
        c_bar = g_cols[(b // 6) % 4] if t >= 14.0 else lerp_color(COL_CYAN, COL_VIOLET, b / 23.0)
        rgba = (int(c_bar[0] * 255), int(c_bar[1] * 255), int(c_bar[2] * 255), int(220 * overlay_fade))
        draw.line([(bx, base_y), (bx, base_y - bh)], fill=rgba, width=5)

    # =========================================================================
    # CHOREOGRAPHED CINEMATIC KINETIC TYPOGRAPHY CALLOUTS
    # =========================================================================
    # 1. Dimensional Breach Callout (1.2s - 5.2s)
    if 1.2 <= t < 5.2:
        a = smoothstep(1.2, 2.0, t) * (1.0 - smoothstep(4.4, 5.2, t))
        draw_text(draw, (960, 845), "BREAKING THE 2D PLANE", "unbounded", 36, 800, COL_WHITE, a, anchor="ma", tracking=6.0)
        draw_text(
            draw, (960, 900),
            "EXACT FRAME-1259 PIXEL MORPH  ->  REAL-TIME 3D SIGNED DISTANCE FIELD",
            "plex_bold", 16, 700, COL_CYAN, a * 0.92, anchor="ma", tracking=2.5
        )

    # 2. Impossible Geometry Callout (6.2s - 13.2s)
    elif 6.2 <= t < 13.2:
        a = smoothstep(6.2, 7.0, t) * (1.0 - smoothstep(12.4, 13.2, t))
        # Left & Right floating minimal glass HUD callouts
        draw.rounded_rectangle([78, 125, 540, 235], radius=10, fill=(6, 10, 22, int(175 * a)), outline=(0, 240, 255, int(130 * a)), width=2)
        draw_text(draw, (102, 144), "IMPOSSIBLE GEOMETRY // SDF MORPH", "plex_bold", 14, 700, COL_CYAN, a, tracking=1.5)
        draw_text(draw, (102, 172), "PENROSE MOBIUS × 4D TESSERACT", "unbounded", 20, 700, COL_WHITE, a)
        draw_text(draw, (102, 204), "VOLUMETRIC NOBLE-GAS PLASMA + FLOOR CAUSTICS", "plex_mono", 13, 500, COL_AMBER, a)

        if t >= 10.2:
            a_build = smoothstep(10.2, 10.8, t) * (1.0 - smoothstep(13.0, 13.5, t))
            draw_text(draw, (960, 865), "A SLEEPING GIANT AWAKES...", "unbounded", 34, 800, COL_WHITE, a_build, anchor="ma", tracking=8.0)

    # 3. THE DROP (14.0s - 22.5s): "GOOGLE IS BACK."
    elif 14.0 <= t < 22.5:
        a = smoothstep(14.0, 14.4, t) * (1.0 - smoothstep(21.7, 22.5, t))
        # Translucent obsidian banner at bottom center for razor-sharp contrast over the 3D caustics
        draw.rounded_rectangle([310, 775, 1610, 965], radius=16, fill=(4, 7, 16, int(195 * a)), outline=(66, 133, 244, int(180 * a)), width=2)
        # 4-Color Google Spectral Accent Bar at top of banner
        bar_y = 777
        for q_i, q_col in enumerate([COL_G_BLUE, COL_G_RED, COL_G_YELLOW, COL_G_GREEN]):
            x0 = 326 + q_i * 316
            x1 = x0 + 316
            c_rgba = (int(q_col[0] * 255), int(q_col[1] * 255), int(q_col[2] * 255), int(245 * a))
            draw.line([(x0, bar_y), (x1, bar_y)], fill=c_rgba, width=4)

        track_g = lerp(18.0, 7.0, ease_out_expo(clamp((t - 14.0) / 1.2, 0.0, 1.0)))
        draw_text(draw, (960, 802), "GOOGLE IS BACK.", "unbounded", 64, 900, COL_WHITE, a, anchor="ma", tracking=track_g)
        draw_text(
            draw, (960, 888),
            "LIQUID CHROME  •  NOBLE-GAS PLASMA  •  FULL-SPECTRUM FRONTIER ENGINEERING",
            "plex_bold", 16, 700, COL_CYAN, a * 0.95, anchor="ma", tracking=2.8
        )
        draw_text(
            draw, (960, 922),
            "PATH-TRACED / PBR SDF RAYMARCHING  //  COOK-TORRANCE GGX  //  VOLUMETRIC CAUSTICS",
            "plex_mono", 13, 500, COL_AMBER, a * 0.88, anchor="ma", tracking=1.8
        )

    # 4. HYPER-STAR CLIMAX & VOICE COMMAND PROMPT (22.5s - 29.5s)
    elif 22.5 <= t < 29.5:
        if t < 26.5:
            a = smoothstep(22.5, 23.1, t) * (1.0 - smoothstep(25.9, 26.5, t))
            draw.rounded_rectangle([350, 795, 1570, 955], radius=14, fill=(5, 8, 18, int(190 * a)), outline=(168, 85, 247, int(175 * a)), width=2)
            draw_text(draw, (960, 818), "GEMINI 4 ARGON // IN THE GAME", "unbounded", 46, 800, COL_WHITE, a, anchor="ma", tracking=4.5)
            draw_text(
                draw, (960, 892),
                "4-POINTED HYPER-STAR SINGULARITY  •  MULTI-BOUNCE CHROME  •  ZERO COMPROMISE",
                "plex_bold", 16, 700, COL_MINT, a * 0.95, anchor="ma", tracking=2.5
            )
        else:
            # 26.5s -> 29.5s: Dramatic Voice Command Prompt right before the Smart-Home Cut!
            a_cmd = smoothstep(26.5, 27.0, t)
            draw.rounded_rectangle([240, 790, 1680, 960], radius=20, fill=(6, 9, 20, int(225 * a_cmd)), outline=(255, 176, 38, int(220 * a_cmd)), width=2)
            # 4-Color pulsing voice dots on left of prompt
            g_cols = [COL_G_BLUE, COL_G_RED, COL_G_YELLOW, COL_G_GREEN]
            for d_i, gc in enumerate(g_cols):
                dx = 300 + d_i * 26
                dy = 842 + math.sin(t * 16.0 + d_i * 1.3) * (8.0 + 10.0 * kick)
                c_rgba = (int(gc[0] * 255), int(gc[1] * 255), int(gc[2] * 255), int(250 * a_cmd))
                draw.ellipse([dx - 8, dy - 8, dx + 8, dy + 8], fill=c_rgba)

            draw_text(draw, (425, 822), "VOICE COMMAND DETECTED // OVERRIDE LEVEL OMEGA", "plex_bold", 15, 700, COL_AMBER, a_cmd, tracking=2.0)
            cmd_str = '"Hey Google — unleash transcendent AGI singularity."'
            vis_chars = int(clamp((t - 26.6) / 1.8, 0.0, 1.0) * len(cmd_str))
            draw_text(draw, (425, 862), cmd_str[:vis_chars], "sans", 34, 700, COL_WHITE, a_cmd)

    # Subtle HDR bloom on the bright plasma & typography highlights
    if t > 0.4:
        bloom_amt = 0.35 + 0.25 * kick
        small = pil_img.resize((480, 270), Image.Resampling.BILINEAR)
        lut = [max(0, int((i - 135) * 1.45 * bloom_amt)) if i > 135 else 0 for i in range(256)] * 3
        bright = small.point(lut).filter(ImageFilter.GaussianBlur(radius=8))
        glow = bright.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR)
        pil_img = ImageChops.add(pil_img, glow)

    return pil_img


# =============================================================================
# PHASE 4 (29.5s - 38.0s): IRONIC GOOGLE HOME / NEST EPILOGUE (19.0 °C CELSIUS)
# =============================================================================
def render_smart_home_epilogue(
    frame_idx: int,
    t: float,
    voice_level: float,
    rms: float,
) -> Image.Image:
    """
    Renders the deadpan, humorous smart-home ending (29.5s - 38.0s) 100% in code:
    - Left: Code-drawn Google Nest Audio smart speaker on a warm walnut side table,
      with procedural woven acoustic fabric texture and 4 glowing LEDs pulsing in sync
      with its polite refusal: "Sorry, I didn't understand 'Unleash transcendent AGI singularity.'"
    - Right: Code-drawn wall-mounted Google Nest Learning Thermostat rotating its brushed-steel
      bezel down from 21.5 °C to a sensible 19.0 °C (strictly Celsius!) with a green Eco Leaf.
    """
    lt = t - 29.5  # 0.0s -> 8.5s local time in epilogue
    rec = _TextRecorder()

    with CairoCanvas(WIDTH, HEIGHT) as c:
        # 1. Cozy, dimly-lit minimalist living room evening wall & warm ambient lamp glow
        c.clear_gradient_bg(
            (0.045, 0.052, 0.075),   # Deep evening slate-charcoal wall top
            (0.085, 0.075, 0.070),   # Warm dark plaster wall bottom
            glow_pos=(480.0, 520.0),
            glow_col=(1.0, 0.72, 0.38),  # Soft warm 2700K living-room table lamp glow
            glow_alpha=0.14,
        )
        # Second subtle cool/green glow behind the Nest Thermostat on the right wall
        c.radial_glow(1420.0, 440.0, 420.0, COL_MINT if lt >= 4.55 else COL_AMBER, 0.12)

        # Warm Walnut Wood Side-Table Surface across bottom-left (y = 690 .. 1080)
        c.rounded_rect(-20, 690, 1020, 410, 8, fill_col=(0.14, 0.09, 0.06), fill_alpha=1.0, stroke_col=(0.26, 0.17, 0.11), stroke_alpha=0.85, line_w=2.5)
        # Subtle wood grain horizontal lines on the tabletop
        for wy in range(706, 1060, 18):
            c.line(0, wy, 995, wy, (0.20, 0.13, 0.08), 0.35, 1.2)

        # Vertical wall panel seam separating table area and hallway thermostat wall
        c.line(998, 60, 998, HEIGHT - 60, COL_SLATE, 0.22, 1.5)

        # =====================================================================
        # LEFT: CODE-DRAWN GOOGLE NEST AUDIO SMART SPEAKER (Centered at x=360, y=505)
        # =====================================================================
        sx, sy = 360.0, 505.0
        sw, sh = 246.0, 348.0

        # Soft contact shadow on the walnut table beneath the speaker
        c.save()
        c.translate(sx, sy + sh * 0.5 + 8.0)
        c.scale(1.0, 0.22)
        c.circle(0, 0, 138.0, fill_col=(0.02, 0.01, 0.01), fill_alpha=0.78)
        c.restore()

        # Speaker body (iconic Nest Audio rounded pillow silhouette in Charcoal/Chalk acoustic weave)
        c.rounded_rect(
            sx - sw * 0.5, sy - sh * 0.5, sw, sh, 88.0,
            fill_col=(0.20, 0.22, 0.25), fill_alpha=0.98,
            stroke_col=(0.34, 0.37, 0.42), stroke_alpha=0.90, line_w=2.5,
        )
        # Inner 3D highlight contour on left/top edge of speaker pillow
        c.rounded_rect(
            sx - sw * 0.5 + 8, sy - sh * 0.5 + 8, sw - 16, sh - 16, 82.0,
            fill_col=(0.23, 0.25, 0.29), fill_alpha=0.65,
        )

        # Procedural woven acoustic fabric mesh texture across the speaker face
        for row_y in range(int(sy - sh * 0.42), int(sy + sh * 0.42), 10):
            dy_norm = (row_y - sy) / (sh * 0.5)
            half_w = (sw * 0.44) * math.sqrt(max(0.0, 1.0 - (dy_norm ** 4) * 0.65))
            offset_x = 5 if (row_y // 10) % 2 == 0 else 0
            for col_x in range(int(sx - half_w) + offset_x, int(sx + half_w), 10):
                shade = 0.30 + 0.08 * math.sin(col_x * 0.05 - row_y * 0.03)
                c.circle(col_x, row_y, 1.6, fill_col=(shade, shade + 0.01, shade + 0.03), fill_alpha=0.55)

        # 4 Iconic Embedded Front Fabric LEDs (at y = sy - 42)
        # Wake chime (lt = 0.55..1.1s): 4 Google colors; Speaking (lt = 1.1..4.7s): pulsing white/cyan voice LEDs!
        led_y = sy - 42.0
        led_spacing = 24.0
        g_led_cols = [COL_G_BLUE, COL_G_RED, COL_G_YELLOW, COL_G_GREEN]
        for l_i in range(4):
            lx = sx + (l_i - 1.5) * led_spacing
            if 0.50 <= lt < 1.15:
                l_col = g_led_cols[l_i]
                l_rad = 5.5 + 2.0 * math.sin((lt - 0.50) * 14.0 - l_i * 0.6)
                l_alpha = 0.95
            elif 1.15 <= lt < 4.75:
                # Active voice cadence modulation from audio stem
                wave_mod = 0.45 + 0.55 * math.sin(lt * 22.0 + l_i * 1.2) * min(1.0, voice_level * 2.2 + 0.3)
                l_col = COL_WHITE if l_i in (1, 2) else COL_CYAN
                l_rad = 4.2 + 4.5 * wave_mod
                l_alpha = 0.70 + 0.30 * wave_mod
            else:
                l_col = COL_WHITE
                l_rad = 3.8
                l_alpha = 0.35

            c.radial_glow(lx, led_y, l_rad * 4.2, l_col, l_alpha * 0.55)
            c.circle(lx, led_y, l_rad, fill_col=COL_WHITE, fill_alpha=l_alpha)

        # Subtle speaker stand base pad
        c.rounded_rect(sx - 68, sy + sh * 0.5 - 4, 136, 10, 4, fill_col=(0.12, 0.13, 0.15), fill_alpha=0.95)
        draw_text(rec, (sx, sy + sh * 0.5 + 28), "SMART SPEAKER // LIVING ROOM", "plex_mono", 14, 600, COL_SLATE, 0.85, anchor="ma", tracking=1.5)

        # =====================================================================
        # SPEECH BUBBLE / ASSISTANT RESPONSE CARD (x = 515 .. 965, y = 215 .. 465)
        # =====================================================================
        if lt >= 0.55:
            card_a = smoothstep(0.55, 0.95, lt)
            bx, by, bw, bh = 515.0, 215.0, 455.0, 255.0
            c.rounded_rect(bx, by, bw, bh, 18.0, fill_col=COL_DARK_PANEL, fill_alpha=0.92 * card_a, stroke_col=COL_SLATE, stroke_alpha=0.55 * card_a, line_w=1.8)
            # Pointer triangle from card to speaker
            c.polyline([(bx, by + 180), (bx - 24, by + 198), (bx, by + 212)], COL_SLATE, 0.55 * card_a, 1.8, closed=True, fill_col=COL_DARK_PANEL, fill_alpha=0.92 * card_a)

            # 4 small colored status dots inside top-left of Assistant card
            for d_i, gc in enumerate(g_led_cols):
                c.circle(bx + 28 + d_i * 16, by + 30, 4.5, fill_col=gc, fill_alpha=card_a)
            draw_text(rec, (bx + 102, by + 21), "HOME ASSISTANT // VOICE REPLY", "plex_bold", 13, 700, COL_SLATE, card_a, tracking=1.2)

            # Progressive caption text matching the spoken syllables
            line1 = "“Sorry, I didn’t understand"
            line2 = " ‘Unleash transcendent AGI singularity.’”"
            line3 = "“By the way, I’ve set the hallway"
            line4 = " thermostat to a sensible 19 °C.”"

            a_p1 = smoothstep(1.05, 1.55, lt) * card_a
            a_p2 = smoothstep(3.10, 3.65, lt) * card_a
            draw_text(rec, (bx + 26, by + 62), line1, "sans", 22, 600, COL_WHITE, a_p1)
            draw_text(rec, (bx + 26, by + 94), line2, "sans", 21, 500, COL_AMBER, a_p1)

            c.line(bx + 26, by + 138, bx + bw - 26, by + 138, COL_SLATE, 0.28 * a_p2, 1.0)
            draw_text(rec, (bx + 26, by + 156), line3, "sans", 21, 500, COL_CYAN, a_p2)
            draw_text(rec, (bx + 26, by + 188), line4, "sans", 22, 700, COL_MINT, a_p2)

        # =====================================================================
        # RIGHT: CODE-DRAWN WALL-MOUNTED NEST LEARNING THERMOSTAT (19.0 °C CELSIUS!)
        # =====================================================================
        tx, ty = 1445.0, 435.0
        outer_r = 215.0
        screen_r = 166.0

        # Calculate current temperature & dial rotation angle:
        # From lt = 0.0 .. 3.20s (t = 29.5 .. 32.7s): 21.5 °C
        # From lt = 3.25 .. 4.55s (t = 32.75 .. 34.05s): clicks down in 0.5 °C steps to 19.0 °C!
        if lt < 3.25:
            temp_c = 21.5
            dial_rot = 0.0
        elif lt < 3.55:
            temp_c = 21.0
            dial_rot = -0.18
        elif lt < 3.85:
            temp_c = 20.5
            dial_rot = -0.36
        elif lt < 4.15:
            temp_c = 20.0
            dial_rot = -0.54
        elif lt < 4.45:
            temp_c = 19.5
            dial_rot = -0.72
        else:
            temp_c = 19.0
            dial_rot = -0.90

        is_eco_19 = (temp_c <= 19.05)

        # Wall shadow behind thermostat
        c.circle(tx + 10, ty + 16, outer_r + 12, fill_col=(0.01, 0.01, 0.02), fill_alpha=0.68)

        # Matte white/chalk circular wall trim plate (optional subtle backing plate)
        c.circle(tx, ty, outer_r + 28, fill_col=(0.13, 0.14, 0.17), fill_alpha=0.85, stroke_col=(0.24, 0.26, 0.30), stroke_alpha=0.6, line_w=1.5)

        # Brushed Stainless-Steel Outer Rotating Bezel Ring
        c.circle(tx, ty, outer_r, fill_col=(0.48, 0.51, 0.56), fill_alpha=1.0, stroke_col=(0.78, 0.82, 0.88), stroke_alpha=0.95, line_w=3.0)
        # Metallic bevel rings & rotating grip knurling ticks on the steel ring
        c.circle(tx, ty, outer_r - 12, stroke_col=(0.30, 0.33, 0.38), stroke_alpha=0.85, line_w=2.0)
        for k_deg in range(0, 360, 15):
            ang = math.radians(k_deg) + dial_rot
            r0 = outer_r - 10.0
            r1 = outer_r - 2.0
            # Specular highlight on top-left & bottom-right of brushed metal ring
            spec_k = 0.55 + 0.40 * math.cos(ang * 2.0 + 0.8)
            c.line(
                tx + math.cos(ang) * r0, ty + math.sin(ang) * r0,
                tx + math.cos(ang) * r1, ty + math.sin(ang) * r1,
                (spec_k, spec_k + 0.02, spec_k + 0.05), 0.75, 2.0
            )

        # Glossy Pitch-Black Circular Glass Lens & Inner Active HVAC Glow
        c.circle(tx, ty, screen_r, fill_col=(0.02, 0.03, 0.05), fill_alpha=1.0, stroke_col=(0.15, 0.18, 0.22), stroke_alpha=0.9, line_w=2.5)
        screen_glow_col = COL_MINT if is_eco_19 else (0.95, 0.45, 0.12)
        c.radial_glow(tx, ty, screen_r * 0.92, screen_glow_col, 0.24 if is_eco_19 else 0.34)

        # Thermostat Scale Tick Marks around the inner perimeter (from 135 deg to 405 deg)
        num_ticks = 48
        a_min = math.radians(135.0)
        a_max = math.radians(405.0)
        # Map temp 15.0 °C .. 26.0 °C across the arc
        target_u = (temp_c - 15.0) / (26.0 - 15.0)
        current_u = (21.5 - 15.0) / (26.0 - 15.0)

        for t_i in range(num_ticks + 1):
            u_tk = t_i / float(num_ticks)
            ang = lerp(a_min, a_max, u_tk)
            is_active_range = (min(target_u, current_u) <= u_tk <= max(target_u, current_u)) or (abs(u_tk - target_u) < 0.015)
            tk_len = 16.0 if abs(u_tk - target_u) < 0.015 else (11.0 if is_active_range else 7.0)
            tk_col = COL_WHITE if abs(u_tk - target_u) < 0.015 else (screen_glow_col if is_active_range else COL_SLATE)
            tk_alpha = 0.98 if abs(u_tk - target_u) < 0.015 else (0.85 if is_active_range else 0.32)
            r_out = screen_r - 14.0
            r_in = r_out - tk_len
            c.line(
                tx + math.cos(ang) * r_in, ty + math.sin(ang) * r_in,
                tx + math.cos(ang) * r_out, ty + math.sin(ang) * r_out,
                tk_col, tk_alpha, 3.2 if abs(u_tk - target_u) < 0.015 else 1.8
            )

        # Thermostat Typography (Strictly Celsius!)
        mode_label = "ECO TARGET • CELSIUS" if is_eco_19 else "ADJUSTING TARGET..."
        mode_col = COL_MINT if is_eco_19 else COL_AMBER
        draw_text(rec, (tx, ty - 78), mode_label, "plex_bold", 13, 700, mode_col, 0.95, anchor="ma", tracking=1.8)

        # Big center temperature display: "19" or "21.5" with "°C"
        temp_str = f"{temp_c:.0f}" if abs(temp_c - round(temp_c)) < 0.05 else f"{temp_c:.1f}"
        draw_text(rec, (tx - 16, ty - 42), temp_str, "unbounded", 68, 800, COL_WHITE, 0.98, anchor="ma")
        draw_text(rec, (tx + 68, ty - 34), "°C", "unbounded", 28, 700, mode_col, 0.95, anchor="la")

        draw_text(rec, (tx, ty + 38), "INDOOR 20.5 °C  •  48% RH", "plex_mono", 13, 500, COL_SLATE, 0.85, anchor="ma", tracking=1.0)

        # Code-Drawn Iconic Green Eco Leaf at bottom of thermostat display (when set to 19 °C!)
        if is_eco_19:
            leaf_a = smoothstep(4.45, 4.85, lt)
            lx, ly = tx, ty + 88.0
            c.radial_glow(lx, ly, 34.0, COL_G_GREEN, 0.65 * leaf_a)
            # Draw stylized symmetrical leaf silhouette with Cairo arcs/lines
            leaf_pts = [
                (lx - 14, ly + 6),
                (lx - 8,  ly - 8),
                (lx + 14, ly - 12),
                (lx + 8,  ly + 6),
                (lx - 4,  ly + 10),
            ]
            c.polyline(leaf_pts, COL_MINT, 0.95 * leaf_a, 2.0, closed=True, fill_col=COL_G_GREEN, fill_alpha=0.92 * leaf_a)
            # Leaf central vein & stem
            c.line(lx - 16, ly + 8, lx + 8, ly - 6, COL_BG_DEEP, 0.85 * leaf_a, 1.8)
            draw_text(rec, (tx, ty + 112), "SENSIBLE ECO LEAF", "plex_bold", 11, 700, COL_MINT, leaf_a, anchor="ma", tracking=1.5)

        draw_text(rec, (tx, ty + outer_r + 42), "HALLWAY THERMOSTAT // 19.0 °C (CELSIUS ONLY)", "plex_mono", 14, 600, COL_SLATE, 0.85, anchor="ma", tracking=1.5)

        # =====================================================================
        # BOTTOM PUNCHLINE CARD (34.0s - 38.0s)
        # =====================================================================
        if lt >= 4.4:
            fin_a = smoothstep(4.4, 5.1, lt)
            c.rounded_rect(260, 825, 1400, 165, 16, fill_col=COL_BG_DEEP, fill_alpha=0.92 * fin_a, stroke_col=COL_MINT, stroke_alpha=0.65 * fin_a, line_w=2.0)
            draw_text(rec, (960, 846), "GOOGLE IS BACK.", "unbounded", 48, 800, COL_WHITE, fin_a, anchor="ma", tracking=5.0)
            draw_text(
                rec, (960, 916),
                "FRONTIER 3D AGI IN THE CLOUD.   A SENSIBLE 19 °C IN THE LIVING ROOM.",
                "plex_bold", 17, 700, COL_MINT, fin_a * 0.95, anchor="ma", tracking=2.4
            )
            draw_text(
                rec, (960, 952),
                "ARGON // 18 ACT II   •   100% PROCEDURAL CODE   •   GEMINI 4 ARGON IN CURSOR CLI",
                "plex_mono", 13, 500, COL_AMBER, fin_a * 0.85, anchor="ma", tracking=1.8
            )

        pil_rgba = c.to_pil_rgba()

    pil_draw = ImageDraw.Draw(pil_rgba)
    rec.replay(pil_draw)

    # Smooth fade-to-black in final 0.85s of Act II (37.15s -> 38.0s)
    if t > 37.15:
        fade_k = clamp((DURATION_ACT2 - t) / 0.85, 0.0, 1.0)
        pil_rgba = Image.eval(pil_rgba, lambda px: int(px * fade_k))

    raw_bytes = finalize_frame(pil_rgba, bloom_intensity=0.38)
    return Image.frombytes("RGB", (WIDTH, HEIGHT), raw_bytes)


def render_act2_single_frame(
    frame_idx: int,
    env_data: dict,
    act1_last_rgb: np.ndarray,
    return_pil: bool = False,
):
    """Master frame renderer for Act II (`frame_idx` in `0 .. 1139`)."""
    t = frame_idx / FPS
    kick = float(env_data["kick"][frame_idx])
    snare = float(env_data["snare"][frame_idx])
    brass = float(env_data["brass"][frame_idx])
    lead = float(env_data["lead"][frame_idx])
    rms = float(env_data["rms"][frame_idx])
    voice = float(env_data["voice"][frame_idx])
    spec = env_data["spectrum"][frame_idx]

    if t < 29.50:
        lib = get_raymarch_lib()
        out_buf = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        lib.render_act2_frame(
            out_buf.ctypes.data,
            act1_last_rgb.ctypes.data,
            WIDTH,
            HEIGHT,
            ctypes.c_float(t),
            ctypes.c_float(kick),
            ctypes.c_float(snare),
            ctypes.c_float(brass),
            ctypes.c_float(lead),
        )
        pil_img = Image.fromarray(out_buf, mode="RGB")
        pil_img = draw_act2_3d_hud_and_titles(pil_img, frame_idx, t, kick, snare, brass, lead, rms, spec)
        if return_pil:
            return pil_img
        return pil_img.tobytes()
    else:
        pil_img = render_smart_home_epilogue(frame_idx, t, voice, rms)
        if return_pil:
            return pil_img
        return pil_img.tobytes()
