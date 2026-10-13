#!/usr/bin/env python3
"""
Master 6-Act Motion Graphics & Data Visualization Scene Renderer for ARGON // 18.
Renders 1260 frames (42.0 seconds @ 30.0 fps) at 1920x1080 resolution with
libcairo vector anti-aliasing, variable-font Pillow typography, and HDR bloom.
"""

import math
from PIL import Image, ImageDraw
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
DURATION = 42.0
TOTAL_FRAMES = int(FPS * DURATION)


# Precompute Lorenz Attractor 3D trajectory (3600 points for ultra-smooth 3D projection)
def _precompute_lorenz(n_pts: int = 3600) -> list[tuple[float, float, float]]:
    pts = []
    x, y, z = 0.1, 0.0, 0.0
    dt = 0.0055
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    # Warm-up 200 steps onto the strange attractor manifold
    for _ in range(200):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
    for _ in range(n_pts):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        pts.append((x, y, z - 25.0))
    return pts


LORENZ_PTS = _precompute_lorenz(3600)

# Precompute 3D Fibonacci Sphere vertices & edges for Act I & VI
def _precompute_sphere_mesh(n_nodes: int = 64) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]]]:
    nodes = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_nodes):
        y = 1.0 - (i / float(n_nodes - 1)) * 2.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        nodes.append((x, y, z))
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = nodes[i][0] - nodes[j][0]
            dy = nodes[i][1] - nodes[j][1]
            dz = nodes[i][2] - nodes[j][2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 0.56:
                edges.append((i, j))
    return nodes, edges


SPHERE_NODES, SPHERE_EDGES = _precompute_sphere_mesh(64)

# Precompute deterministic background starfield / dust particles
def _precompute_particles(n: int = 90) -> list[tuple[float, float, float, float]]:
    pts = []
    for i in range(n):
        # Golden-ratio quasi-random distribution
        px = ((i * 0.6180339887) % 1.0) * WIDTH
        py = ((i * 0.3819660112 + i * i * 0.0013) % 1.0) * HEIGHT
        depth = 0.25 + 0.75 * ((i * 0.75487766) % 1.0)
        col_idx = i % 3
        pts.append((px, py, depth, col_idx))
    return pts


BG_PARTICLES = _precompute_particles(90)


def draw_global_grid_and_particles(c: CairoCanvas, t: float, kick: float, act_col: tuple):
    """Subtle architectural blueprint grid + drifting depth-of-field bokeh particles."""
    c.set_operator_over()
    # Architectural grid lines every 120px
    grid_alpha = 0.045 + 0.025 * kick
    for gx in range(120, WIDTH, 120):
        c.line(gx, 68, gx, HEIGHT - 68, COL_SLATE, grid_alpha, 1.0)
    for gy in range(120, HEIGHT, 120):
        c.line(70, gy, WIDTH - 70, gy, COL_SLATE, grid_alpha, 1.0)

    # Crosshair intersections
    for gx in range(240, WIDTH, 240):
        for gy in range(240, HEIGHT, 240):
            c.line(gx - 5, gy, gx + 5, gy, act_col, 0.16, 1.2)
            c.line(gx, gy - 5, gx, gy + 5, act_col, 0.16, 1.2)

    # Drifting bokeh particles
    c.set_operator_add()
    for px, py, depth, col_idx in BG_PARTICLES:
        x = (px + t * 22.0 * depth) % WIDTH
        y = (py - t * 14.0 * depth + 18.0 * math.sin(t * 0.8 + px * 0.01)) % HEIGHT
        col = COL_CYAN if col_idx == 0 else (COL_VIOLET if col_idx == 1 else COL_AMBER)
        r = (1.2 + 2.4 * depth) * (1.0 + 0.35 * kick)
        c.circle(x, y, r, fill_col=col, fill_alpha=0.14 * depth)
    c.set_operator_over()


def draw_hud_chrome(
    c: CairoCanvas,
    draw: ImageDraw.ImageDraw,
    frame_idx: int,
    t: float,
    act_num: int,
    act_name: str,
    kick: float,
    rms: float,
    spec: list[float],
):
    """Draws the persistent broadcast telemetry HUD frame, timecode, and mini audio analyzer."""
    c.set_operator_over()
    # Top and bottom precision hairlines
    c.line(64, 64, WIDTH - 64, 64, COL_SLATE, 0.32, 1.2)
    c.line(64, HEIGHT - 64, WIDTH - 64, HEIGHT - 64, COL_SLATE, 0.32, 1.2)

    # Progress bar along top hairline
    prog_w = (WIDTH - 128) * (t / DURATION)
    c.line(64, 64, 64 + prog_w, 64, COL_CYAN, 0.90, 2.6)
    c.circle(64 + prog_w, 64, 4.0 + 3.0 * kick, fill_col=COL_CYAN, fill_alpha=0.95)

    # Corner L-brackets
    b_len = 22.0
    for bx, by, sx, sy in [
        (48, 48, 1, 1),
        (WIDTH - 48, 48, -1, 1),
        (48, HEIGHT - 48, 1, -1),
        (WIDTH - 48, HEIGHT - 48, -1, -1),
    ]:
        c.line(bx, by, bx + sx * b_len, by, COL_CYAN, 0.75, 2.0)
        c.line(bx, by, bx, by + sy * b_len, COL_CYAN, 0.75, 2.0)

    # Top-left status indicator
    pulse_col = COL_CYAN if (frame_idx // 15) % 2 == 0 else COL_MINT
    c.circle(76, 40, 5.0 + 2.0 * kick, fill_col=pulse_col, fill_alpha=0.95)
    draw_text(draw, (92, 31), "SYS // ARGON-18 ENGINE", "plex_bold", 15, 700, COL_WHITE, 0.92, tracking=2.0)
    draw_text(draw, (370, 31), f"ACT 0{act_num} : {act_name}", "plex_mono", 15, 500, COL_CYAN, 0.88, tracking=1.5)

    # Top-right timecode & frame counter
    mins = int(t // 60)
    secs = int(t % 60)
    frames = int(frame_idx % 30)
    tc_str = f"TC 00:{mins:02d}:{secs:02d}:{frames:02d}  |  FRM {frame_idx:04d}/1260  |  1080P30"
    draw_text(draw, (WIDTH - 68, 31), tc_str, "plex_mono", 15, 500, COL_SLATE, 0.90, anchor="ra", tracking=1.0)

    # Bottom-left telemetry readout
    bpm_pulse = int(120 + round(kick * 2))
    bot_left = f"AUDIO: 48KHZ STEREO PCM  //  120.0 BPM (D DORIAN)  //  RMS: {rms*100:04.1f}%  //  PEAK: {bpm_pulse} BPM"
    draw_text(draw, (68, HEIGHT - 46), bot_left, "plex_mono", 14, 500, COL_SLATE, 0.80, tracking=1.0)

    # Bottom-right mini 24-bar real-time FFT spectrum analyzer
    base_x = WIDTH - 68 - 24 * 8
    base_y = HEIGHT - 26
    for b in range(24):
        val = float(spec[b])
        bh = max(3.0, val * 26.0)
        bx = base_x + b * 8
        b_col = lerp_color(COL_CYAN, COL_MAGENTA, b / 23.0)
        c.line(bx, base_y, bx, base_y - bh, b_col, 0.85, 4.5)
    draw_text(draw, (base_x - 16, HEIGHT - 46), "FFT //", "plex_mono", 14, 500, COL_CYAN, 0.75, anchor="ra", tracking=1.0)


# =============================================================================
# ACT I (0.0s - 6.0s): ATOMIC IGNITION // ELEMENT 18
# =============================================================================
def render_act1_ignition(
    c: CairoCanvas,
    draw: ImageDraw.ImageDraw,
    t: float,
    kick: float,
    arp: float,
    spec: list[float],
):
    # Smooth intro progression 0..1 over 6 seconds
    u = clamp(t / 6.0, 0.0, 1.0)
    reveal = ease_out_expo(clamp(t / 2.2, 0.0, 1.0))

    cx, cy = 960.0, 500.0
    # Radial energy bloom behind the Argon-18 core
    c.radial_glow(cx, cy, 460.0 * reveal + 80.0 * kick, COL_CYAN, 0.22 + 0.10 * kick)
    c.radial_glow(cx, cy, 290.0 * reveal, COL_VIOLET, 0.25 + 0.12 * arp)

    # Outer precision degree dial & dashed telemetry rings
    for ring_idx, (rad, speed, dashes, col, w) in enumerate([
        (370.0,  0.18, [12.0, 18.0, 4.0, 18.0], COL_SLATE, 1.5),
        (320.0, -0.32, [45.0, 25.0], COL_VIOLET, 2.2),
        (255.0,  0.48, [8.0, 12.0], COL_CYAN, 1.8),
    ]):
        r_cur = rad * ease_out_cubic(clamp((t - ring_idx * 0.25) / 1.5, 0.0, 1.0))
        if r_cur > 5.0:
            c.save()
            c.translate(cx, cy)
            c.rotate(t * speed)
            c.set_dash(dashes, t * 40.0)
            c.circle(0, 0, r_cur, stroke_col=col, stroke_alpha=0.55, line_w=w)
            c.set_dash([])
            c.restore()

    # Degree ticks on outer compass (r = 395)
    if t > 0.4:
        tick_a = smoothstep(0.4, 1.8, t) * 0.55
        for deg in range(0, 360, 6):
            ang = math.radians(deg) + t * 0.08
            r_in = 388.0 if deg % 30 != 0 else 380.0
            r_out = 398.0 + (6.0 * spec[(deg // 12) % 24] if deg % 30 == 0 else 0.0)
            col = COL_CYAN if deg % 30 == 0 else COL_SLATE
            c.line(
                cx + math.cos(ang) * r_in,
                cy + math.sin(ang) * r_in,
                cx + math.cos(ang) * r_out,
                cy + math.sin(ang) * r_out,
                col,
                tick_a,
                2.0 if deg % 30 == 0 else 1.0,
            )

    # Argon-18 Atomic Electron Shells: 2 (K-shell), 8 (L-shell), 8 (M-shell)
    shells = [
        (78.0,  2,  1.65, COL_AMBER),
        (138.0, 8, -1.05, COL_VIOLET),
        (198.0, 8,  0.72, COL_CYAN),
    ]
    c.set_operator_add()
    for s_idx, (s_rad, n_elec, spd, s_col) in enumerate(shells):
        sr = s_rad * ease_out_cubic(clamp((t - 0.15 * s_idx) / 1.4, 0.0, 1.0))
        if sr <= 2.0:
            continue
        # Tilted 3D elliptical projection + full circular orbit harmonic
        c.circle(cx, cy, sr, stroke_col=s_col, stroke_alpha=0.35 + 0.2 * arp, line_w=1.8)
        for e_i in range(n_elec):
            ang = (2.0 * math.pi * e_i / n_elec) + t * spd
            ex = cx + math.cos(ang) * sr
            ey = cy + math.sin(ang) * sr
            # Glowing trail arc behind each electron
            trail_dir = -1.0 if spd > 0 else 1.0
            a_start = ang + trail_dir * 0.35 if trail_dir < 0 else ang
            a_end = ang if trail_dir < 0 else ang + trail_dir * 0.35
            c.arc(cx, cy, sr, a_start, a_end, s_col, 0.75, 4.0)
            # Electron core + halo
            c.circle(ex, ey, 9.0 + 4.0 * kick, fill_col=s_col, fill_alpha=0.35)
            c.circle(ex, ey, 4.2, fill_col=COL_WHITE, fill_alpha=0.98)

    # Central 3D Rotating Icosahedral / Fibonacci Geodesic Nucleus
    core_r = (46.0 + 10.0 * kick) * reveal
    rot_y = t * 1.1
    rot_x = t * 0.7
    cy_ry, sy_ry = math.cos(rot_y), math.sin(rot_y)
    cx_rx, sx_rx = math.cos(rot_x), math.sin(rot_x)

    proj_nodes = []
    for nx, ny, nz in SPHERE_NODES[:32]:
        # Rotate Y then X
        x1 = nx * cy_ry + nz * sy_ry
        z1 = -nx * sy_ry + nz * cy_ry
        y2 = ny * cx_rx - z1 * sx_rx
        z2 = ny * sx_rx + z1 * cx_rx
        px = cx + x1 * core_r
        py = cy + y2 * core_r
        proj_nodes.append((px, py, z2))

    for i, j in SPHERE_EDGES:
        if i < 32 and j < 32:
            z_avg = 0.5 * (proj_nodes[i][2] + proj_nodes[j][2])
            if z_avg > -0.4:
                c.line(
                    proj_nodes[i][0], proj_nodes[i][1],
                    proj_nodes[j][0], proj_nodes[j][1],
                    COL_CYAN, 0.35 * (z_avg + 1.0), 1.4
                )
    c.circle(cx, cy, 18.0 * reveal + 5.0 * kick, fill_col=COL_WHITE, fill_alpha=0.95)
    c.set_operator_over()

    # Left & Right Telemetry Callout Cards
    card_a = smoothstep(0.8, 2.0, t)
    if card_a > 0.01:
        # Left Card: Atomic Spec
        lx, ly = 110.0, 220.0
        c.rounded_rect(lx, ly, 330, 175, 10, fill_col=COL_DARK_PANEL, fill_alpha=0.82 * card_a, stroke_col=COL_CYAN, stroke_alpha=0.55 * card_a, line_w=1.5)
        c.line(lx + 330, ly + 85, cx - 260, cy - 80, COL_CYAN, 0.45 * card_a, 1.4)
        c.circle(cx - 260, cy - 80, 4.0, fill_col=COL_CYAN, fill_alpha=card_a)
        draw_text(draw, (lx + 22, ly + 20), "ELEMENT // SPEC", "plex_bold", 14, 700, COL_CYAN, card_a, tracking=2.0)
        draw_text(draw, (lx + 22, ly + 48), "18  Ar", "unbounded", 38, 800, COL_WHITE, card_a)
        draw_text(draw, (lx + 22, ly + 102), "MASS: 39.948 u  |  NOBLE GAS", "plex_mono", 14, 500, COL_SLATE, card_a)
        draw_text(draw, (lx + 22, ly + 128), "CONFIG: [Ne] 3s2 3p6 (2, 8, 8)", "plex_mono", 14, 500, COL_MINT, card_a)

        # Right Card: Engine Spec
        rx, ry = WIDTH - 440.0, 220.0
        c.rounded_rect(rx, ry, 330, 175, 10, fill_col=COL_DARK_PANEL, fill_alpha=0.82 * card_a, stroke_col=COL_VIOLET, stroke_alpha=0.55 * card_a, line_w=1.5)
        c.line(rx, ry + 85, cx + 260, cy - 80, COL_VIOLET, 0.45 * card_a, 1.4)
        c.circle(cx + 260, cy - 80, 4.0, fill_col=COL_VIOLET, fill_alpha=card_a)
        draw_text(draw, (rx + 22, ry + 20), "SYNTHESIS // PIPELINE", "plex_bold", 14, 700, COL_VIOLET, card_a, tracking=2.0)
        draw_text(draw, (rx + 22, ry + 48), "GEMINI 4", "unbounded", 34, 800, COL_WHITE, card_a)
        draw_text(draw, (rx + 22, ry + 102), "VECTOR: LIBCAIRO + NUMPY SIMD", "plex_mono", 14, 500, COL_SLATE, card_a)
        draw_text(draw, (rx + 22, ry + 128), "AUDIO: 48KHZ FM + FDN REVERB", "plex_mono", 14, 500, COL_AMBER, card_a)

    # Hero Title Lockup at bottom center
    title_a = smoothstep(0.5, 1.8, t)
    if title_a > 0.01:
        # Subtle dark backdrop pill for ultra-crisp text legibility
        c.rounded_rect(470, 795, 980, 175, 16, fill_col=COL_BG_DEEP, fill_alpha=0.78 * title_a, stroke_col=COL_CYAN, stroke_alpha=0.45 * title_a, line_w=1.5)
        tracking_px = lerp(28.0, 8.0, ease_out_expo(clamp((t - 0.5) / 1.8, 0.0, 1.0)))
        draw_text(draw, (960, 818), "ARGON // 18", "unbounded", 64, 800, COL_WHITE, title_a, anchor="ma", tracking=tracking_px)
        draw_text(
            draw,
            (960, 908),
            "PROCEDURAL AUDIO-VISUAL CINEMA  •  100% GENERATED IN CODE",
            "plex_bold",
            17,
            700,
            COL_CYAN,
            title_a * 0.95,
            anchor="ma",
            tracking=3.5,
        )


# =============================================================================
# ACT II (6.0s - 14.0s): THE LIVING COMPILER // CODE TO PIXELS
# =============================================================================
CODE_LINES = [
    ("01", "import ctypes, numpy as np", COL_VIOLET),
    ("02", "from cairo_engine import CairoCanvas, HDRBloom", COL_VIOLET),
    ("03", "", COL_SLATE),
    ("04", "class ArgonCinemaEngine:", COL_CYAN),
    ("05", "    def __init__(self, width=1920, height=1080, fps=30):", COL_WHITE),
    ("06", "        self.surface = CairoCanvas(width, height)", COL_WHITE),
    ("07", "        self.shells  = (2, 8, 8)  # Z=18 Noble Config", COL_MINT),
    ("08", "", COL_SLATE),
    ("09", "    def synthesize_frame(self, t: float, audio_fft):", COL_CYAN),
    ("10", "        # Real-time 3D Lorenz strange attractor ODE", COL_SLATE),
    ("11", "        dx = self.sigma * (y - x) * dt", COL_AMBER),
    ("12", "        dy = (x * (self.rho - z) - y) * dt", COL_AMBER),
    ("13", "        dz = (x * y - self.beta * z) * dt", COL_AMBER),
    ("14", "        self.surface.trace_spline(dx, dy, dz, glow=True)", COL_WHITE),
    ("15", "        return HDRBloom.resolve_rgb24(self.surface, t)", COL_MINT),
]


def render_act2_compiler(
    c: CairoCanvas,
    draw: ImageDraw.ImageDraw,
    t: float,
    kick: float,
    arp: float,
    spec: list[float],
):
    lt = t - 6.0  # 0.0 -> 8.0s local time
    enter = ease_out_cubic(clamp(lt / 0.9, 0.0, 1.0))

    # Left IDE Window Panel
    px = lerp(-750.0, 88.0, enter)
    py = 118.0
    pw, ph = 820.0, 830.0

    c.rounded_rect(px, py, pw, ph, 14.0, fill_col=COL_DARK_PANEL, fill_alpha=0.90, stroke_col=COL_CYAN, stroke_alpha=0.65, line_w=1.8)
    # IDE Header Bar
    c.rounded_rect(px, py, pw, 48.0, 14.0, fill_col=COL_BG_MID, fill_alpha=0.95, stroke_col=COL_CYAN, stroke_alpha=0.4, line_w=1.0)
    # Window control dots
    c.circle(px + 26, py + 24, 6.5, fill_col=COL_MAGENTA, fill_alpha=0.9)
    c.circle(px + 48, py + 24, 6.5, fill_col=COL_AMBER, fill_alpha=0.9)
    c.circle(px + 70, py + 24, 6.5, fill_col=COL_MINT, fill_alpha=0.9)
    draw_text(draw, (px + 105, py + 15), "argon_engine.py  —  AST LIVE EXECUTION", "plex_bold", 15, 700, COL_WHITE, 0.92, tracking=1.2)
    draw_text(draw, (px + pw - 22, py + 15), "PYTHON 3.13 // SIMD", "plex_mono", 14, 500, COL_CYAN, 0.85, anchor="ra")

    # Active execution line highlight sweeps down synchronized to beats
    active_line_idx = int((lt * 2.4) % len(CODE_LINES))
    active_y = py + 68 + active_line_idx * 42
    c.rounded_rect(px + 14, active_y - 4, pw - 28, 36, 6.0, fill_col=COL_CYAN, fill_alpha=0.14 + 0.10 * kick, stroke_col=COL_CYAN, stroke_alpha=0.45, line_w=1.0)

    # Render code lines with progressive typing effect (finishes all 15 lines by lt=4.5s)
    total_chars_budget = int(lt * 135)
    chars_used = 0
    for idx, (lineno, code_str, col) in enumerate(CODE_LINES):
        ly = py + 68 + idx * 42
        draw_text(draw, (px + 28, ly), lineno, "mono", 18, 500, COL_SLATE, 0.65)
        if chars_used < total_chars_budget:
            vis_len = min(len(code_str), total_chars_budget - chars_used)
            vis_text = code_str[:vis_len]
            draw_text(draw, (px + 78, ly), vis_text, "mono", 19, 600, col, 0.96)
            # Blinking block cursor on currently typing line
            if vis_len < len(code_str) or (idx == len(CODE_LINES) - 1 and int(t * 4) % 2 == 0 and chars_used + len(code_str) >= total_chars_budget):
                cur_x = px + 78 + vis_len * 11.4
                c.rounded_rect(cur_x + 3, ly, 10, 22, 2, fill_col=COL_CYAN, fill_alpha=0.95)
        chars_used += max(4, len(code_str))

    # Bottom Compiler Metrics Bar inside Left Panel
    m_y = py + ph - 125
    c.line(px + 20, m_y, px + pw - 20, m_y, COL_SLATE, 0.35, 1.0)
    metrics = [
        ("FRAME TIME", f"{8.4 - 1.2*math.sin(t*3):.1f} ms", COL_MINT),
        ("THROUGHPUT", f"{118 + int(kick*14)} FPS", COL_CYAN),
        ("VECTOR OPS", f"{14200 + int(lt*1850):,}/s", COL_AMBER),
        ("ZERO-COPY", "BGRA -> RGB24", COL_VIOLET),
    ]
    for m_i, (m_lbl, m_val, m_col) in enumerate(metrics):
        mx = px + 32 + m_i * 195
        draw_text(draw, (mx, m_y + 22), m_lbl, "plex_mono", 13, 500, COL_SLATE, 0.85, tracking=1.0)
        draw_text(draw, (mx, m_y + 48), m_val, "chakra_bold", 26, 700, m_col, 0.96)

    # Right Panel: Live AST Node Graph & Bezier Packet Routing
    rx = lerp(2000.0, 965.0, enter)
    ry = 118.0
    rw, rh = 865.0, 830.0
    c.rounded_rect(rx, ry, rw, rh, 14.0, fill_col=COL_DARK_PANEL, fill_alpha=0.84, stroke_col=COL_VIOLET, stroke_alpha=0.60, line_w=1.8)
    draw_text(draw, (rx + 28, ry + 20), "LIVE AST DATAFLOW // PROCEDURAL RENDER DAG", "plex_bold", 16, 700, COL_VIOLET, 0.92, tracking=2.0)

    # DAG Nodes inside Right Panel
    dag_nodes = [
        ("AST_PARSE",   rx + 145, ry + 170, COL_VIOLET),
        ("AUDIO_FFT",   rx + 145, ry + 390, COL_AMBER),
        ("ODE_SOLVER",  rx + 145, ry + 610, COL_MINT),
        ("CAIRO_SIMD",  rx + 445, ry + 270, COL_CYAN),
        ("FOURIER_EPI", rx + 445, ry + 510, COL_MAGENTA),
        ("HDR_BLOOM",   rx + 725, ry + 390, COL_CYAN),
        ("H264_MUX",    rx + 725, ry + 640, COL_MINT),
    ]
    dag_edges = [
        (0, 3), (1, 3), (1, 4), (2, 4), (3, 5), (4, 5), (5, 6)
    ]

    # Cubic Bezier stream from active code line on left panel into DAG nodes!
    src_x = px + pw
    src_y = active_y + 14
    for target_idx in (0, 1, 2):
        tx, ty = dag_nodes[target_idx][1] - 72, dag_nodes[target_idx][2]
        c.cubic_bezier(
            src_x, src_y,
            src_x + 40, src_y,
            tx - 40, ty,
            tx, ty,
            COL_CYAN, 0.42, 2.0
        )
        # Glowing packet traveling from code line to DAG
        p_u = (t * 1.8 + target_idx * 0.33) % 1.0
        omt = 1.0 - p_u
        bx = (omt**3)*src_x + 3*(omt**2)*p_u*(src_x+40) + 3*omt*(p_u**2)*(tx-40) + (p_u**3)*tx
        by = (omt**3)*src_y + 3*(omt**2)*p_u*src_y + 3*omt*(p_u**2)*ty + (p_u**3)*ty
        c.circle(bx, by, 5.5, fill_col=COL_WHITE, fill_alpha=0.95)
        c.radial_glow(bx, by, 22.0, COL_CYAN, 0.55)

    # Draw DAG edges & traveling data packets
    for e_i, (u_idx, v_idx) in enumerate(dag_edges):
        _, x0, y0, col0 = dag_nodes[u_idx]
        _, x1, y1, col1 = dag_nodes[v_idx]
        x0_out = x0 + 72
        x1_in = x1 - 72
        cx1 = lerp(x0_out, x1_in, 0.5)
        c.cubic_bezier(x0_out, y0, cx1, y0, cx1, y1, x1_in, y1, col0, 0.50, 2.4)

        # 2 packets per edge
        for pk in range(2):
            pu = (t * 1.35 + e_i * 0.21 + pk * 0.5) % 1.0
            omt = 1.0 - pu
            px_k = (omt**3)*x0_out + 3*(omt**2)*pu*cx1 + 3*omt*(pu**2)*cx1 + (pu**3)*x1_in
            py_k = (omt**3)*y0 + 3*(omt**2)*pu*y0 + 3*omt*(pu**2)*y1 + (pu**3)*y1
            pcol = lerp_color(col0, col1, pu)
            c.circle(px_k, py_k, 5.0 + 2.5 * arp, fill_col=COL_WHITE, fill_alpha=0.95)
            c.radial_glow(px_k, py_k, 24.0, pcol, 0.50)

    # Draw DAG Node Pills & real-time mini-waveforms inside each node
    for n_i, (n_name, nx, ny, ncol) in enumerate(dag_nodes):
        c.radial_glow(nx, ny, 85.0 + 25.0 * kick, ncol, 0.22)
        c.rounded_rect(nx - 74, ny - 44, 148, 88, 12.0, fill_col=COL_BG_DEEP, fill_alpha=0.92, stroke_col=ncol, stroke_alpha=0.85, line_w=2.2)
        draw_text(draw, (nx, ny - 26), n_name, "plex_bold", 14, 700, ncol, 0.95, anchor="ma", tracking=1.0)
        # Mini animated waveform inside node pill
        w_pts = []
        for s_i in range(18):
            wx = nx - 54 + s_i * 6.4
            wy = ny + 14 + math.sin(t * 8.0 + s_i * 0.6 + n_i) * (10.0 + 8.0 * spec[(n_i * 3 + s_i) % 24])
            w_pts.append((wx, wy))
        c.polyline(w_pts, ncol, 0.85, 2.0)


# =============================================================================
# ACT III (14.0s - 22.0s): HARMONIC GEOMETRY // FOURIER & LORENZ
# =============================================================================
def render_act3_harmonics(
    c: CairoCanvas,
    draw: ImageDraw.ImageDraw,
    t: float,
    kick: float,
    arp: float,
    spec: list[float],
):
    lt = t - 14.0  # 0.0 -> 8.0s local time
    enter = ease_out_cubic(clamp(lt / 0.8, 0.0, 1.0))

    # Section Header Banner
    draw_text(draw, (88, 92), "HARMONIC GEOMETRY // DIFFERENTIAL TOPOLOGY", "unbounded", 28, 700, COL_WHITE, enter, tracking=2.0)
    draw_text(draw, (88, 132), "LEFT: 7-TERM FOURIER EPICYCLE SYNTHESIS    |    RIGHT: 3D LORENZ STRANGE ATTRACTOR ODE", "plex_mono", 15, 500, COL_CYAN, enter * 0.85, tracking=1.5)

    # -------------------------------------------------------------------------
    # LEFT HALF: Fourier Epicycle Machine + Live Waveform Trace
    # -------------------------------------------------------------------------
    fcx, fcy = 340.0, 540.0
    c.radial_glow(fcx, fcy, 280.0, COL_CYAN, 0.18 + 0.08 * kick)

    # Compute epicycle chain at current time and trace history
    harmonics = [
        (1, 118.0, COL_CYAN),
        (3, 48.0,  COL_VIOLET),
        (5, 28.0,  COL_MAGENTA),
        (7, 17.0,  COL_AMBER),
        (9, 11.0,  COL_MINT),
        (11, 7.5,  COL_CYAN),
    ]

    omega = 2.15
    x, y = fcx, fcy
    for n_h, rad, h_col in harmonics:
        prev_x, prev_y = x, y
        ang = n_h * omega * t
        x += rad * math.cos(ang)
        y += rad * math.sin(ang)
        c.circle(prev_x, prev_y, rad, stroke_col=h_col, stroke_alpha=0.42 * enter, line_w=1.6)
        c.line(prev_x, prev_y, x, y, COL_WHITE, 0.78 * enter, 2.0)
        c.circle(x, y, 3.8, fill_col=h_col, fill_alpha=0.95 * enter)

    # Epicycle tip glow
    c.radial_glow(x, y, 38.0, COL_AMBER, 0.70 * enter)
    c.circle(x, y, 6.0, fill_col=COL_WHITE, fill_alpha=enter)

    # Horizontal projection beam from epicycle tip to waveform start (x = 560)
    wave_start_x = 565.0
    c.set_dash([6.0, 6.0], -t * 60.0)
    c.line(x, y, wave_start_x, y, COL_AMBER, 0.85 * enter, 2.0)
    c.set_dash([])
    c.circle(wave_start_x, y, 5.5, fill_col=COL_AMBER, fill_alpha=enter)

    # Sample Fourier wave history across x = 565 .. 910
    wave_pts = []
    n_wave = 140
    for i in range(n_wave):
        dt_back = (i / float(n_wave - 1)) * 2.2
        t_hist = t - dt_back
        wy = fcy
        for n_h, rad, _ in harmonics:
            wy += rad * math.sin(n_h * omega * t_hist)
        wx = wave_start_x + (i / float(n_wave - 1)) * 345.0
        wave_pts.append((wx, wy))

    c.set_operator_add()
    c.polyline(wave_pts, COL_CYAN, 0.35 * enter, 7.0)
    c.polyline(wave_pts, COL_WHITE, 0.92 * enter, 2.4)
    c.set_operator_over()

    # Formula card below Fourier Epicycles
    c.rounded_rect(90, 815, 810, 125, 12, fill_col=COL_DARK_PANEL, fill_alpha=0.82 * enter, stroke_col=COL_CYAN, stroke_alpha=0.45 * enter, line_w=1.4)
    draw_text(draw, (118, 835), "FOURIER HARMONIC SERIES EXPANSION", "plex_bold", 14, 700, COL_CYAN, enter, tracking=1.5)
    draw_text(draw, (118, 865), "f(t) = Σ (4 / πk) · sin(kωt)   for k ∈ {1, 3, 5, 7, 9, 11}", "mono", 19, 600, COL_WHITE, enter)
    draw_text(draw, (118, 902), f"INSTANTANEOUS PHASE: {(omega*t) % (2*math.pi):.3f} rad   |   TIP Y-OFFSET: {y - fcy:+.1f} px", "plex_mono", 14, 500, COL_MINT, enter)

    # Center vertical divider hairline
    c.line(950, 175, 950, 940, COL_SLATE, 0.30 * enter, 1.2)

    # -------------------------------------------------------------------------
    # RIGHT HALF: 3D Lorenz Strange Attractor Manifold (Turntable Camera)
    # -------------------------------------------------------------------------
    lcx, lcy = 1435.0, 515.0
    c.radial_glow(lcx, lcy, 360.0, COL_VIOLET, 0.24 + 0.10 * kick)
    c.radial_glow(lcx, lcy, 200.0, COL_MAGENTA, 0.20 + 0.08 * arp)

    yaw = t * 0.58 + 0.4
    pitch = -0.35 + 0.18 * math.sin(t * 0.45)
    cy_w, sy_w = math.cos(yaw), math.sin(yaw)
    cp_p, sp_p = math.cos(pitch), math.sin(pitch)
    scale_3d = 12.8 * (1.0 + 0.04 * kick) * enter

    # Progressively draw up to 3200 points along the Lorenz trajectory
    n_draw = int(lerp(800, 3400, clamp(lt / 5.5, 0.0, 1.0)))
    proj_lorenz = []
    for lx_p, ly_p, lz_p in LORENZ_PTS[:n_draw]:
        # Rotate around Z/Y/X for dramatic 3D butterfly perspective
        x1 = lx_p * cy_w - ly_p * sy_w
        y1 = lx_p * sy_w + ly_p * cy_w
        z1 = lz_p
        y2 = y1 * cp_p - z1 * sp_p
        z2 = y1 * sp_p + z1 * cp_p
        fov = 650.0 / max(200.0, 650.0 - z2 * 4.5)
        sx = lcx + x1 * scale_3d * fov
        sy = lcy - z2 * scale_3d * fov
        proj_lorenz.append((sx, sy))

    # Draw trajectory in color-graded segments for glowing depth
    c.set_operator_add()
    chunk_sz = 140
    for start_i in range(0, len(proj_lorenz) - chunk_sz, chunk_sz):
        seg = proj_lorenz[start_i:start_i + chunk_sz + 1]
        u_seg = start_i / float(max(1, len(proj_lorenz)))
        if u_seg < 0.5:
            col = lerp_color(COL_VIOLET, COL_CYAN, u_seg * 2.0)
        else:
            col = lerp_color(COL_CYAN, COL_AMBER, (u_seg - 0.5) * 2.0)
        alpha_seg = (0.25 + 0.65 * u_seg) * enter
        lw = 1.4 + 2.0 * u_seg
        c.polyline(seg, col, alpha_seg, lw)

    # Leading particle head at the tip of the Lorenz attractor
    if proj_lorenz:
        hx, hy = proj_lorenz[-1]
        c.radial_glow(hx, hy, 42.0, COL_AMBER, 0.85 * enter)
        c.circle(hx, hy, 6.5, fill_col=COL_WHITE, fill_alpha=enter)
    c.set_operator_over()

    # Lorenz ODE Telemetry Card
    c.rounded_rect(1000, 815, 830, 125, 12, fill_col=COL_DARK_PANEL, fill_alpha=0.82 * enter, stroke_col=COL_VIOLET, stroke_alpha=0.45 * enter, line_w=1.4)
    draw_text(draw, (1028, 835), "LORENZ CHAOTIC ATTRACTOR MANIFOLD (RK4 SOLVER)", "plex_bold", 14, 700, COL_VIOLET, enter, tracking=1.5)
    draw_text(draw, (1028, 865), "dx/dt = σ(y - x)   |   dy/dt = x(ρ - z) - y   |   dz/dt = xy - βz", "mono", 18, 600, COL_WHITE, enter)
    draw_text(draw, (1028, 902), f"PARAMS: σ=10.0, ρ=28.0, β=2.667   |   TRAJECTORY NODES: {n_draw:,}   |   LYAPUNOV: +0.9056", "plex_mono", 14, 500, COL_AMBER, enter)


# =============================================================================
# ACT IV (22.0s - 30.0s): NEURAL LATENT TOPOLOGY // ATTENTION & SPECTRAL VORTEX
# =============================================================================
def render_act4_neural(
    c: CairoCanvas,
    draw: ImageDraw.ImageDraw,
    t: float,
    kick: float,
    snare: float,
    arp: float,
    spec: list[float],
):
    lt = t - 22.0  # 0.0 -> 8.0s Climax Drop!
    enter = ease_out_expo(clamp(lt / 0.7, 0.0, 1.0))

    cx, cy = 960.0, 520.0
    # Massive audio-reactive core glow on every kick & snare
    c.radial_glow(cx, cy, 520.0 + 140.0 * kick, COL_VIOLET, 0.28 + 0.14 * kick)
    c.radial_glow(cx, cy, 340.0 + 90.0 * snare, COL_CYAN, 0.25 + 0.15 * arp)

    # Top Title Banner
    draw_text(draw, (960, 88), "NEURAL LATENT TOPOLOGY // MULTI-HEAD ATTENTION", "unbounded", 28, 800, COL_WHITE, enter, anchor="ma", tracking=2.5)
    draw_text(draw, (960, 128), "64-BAND POLAR SPECTRAL VORTEX  •  TENSOR ROUTING AT 120 BPM", "plex_mono", 15, 500, COL_CYAN, enter * 0.9, anchor="ma", tracking=2.0)

    # 64-Band Polar Spectral Equalizer Burst around the Neural Core
    n_bars = 64
    base_r = 250.0 * enter + 12.0 * kick
    c.set_operator_add()
    for i in range(n_bars):
        ang = (2.0 * math.pi * i / n_bars) + t * 0.25
        s_val = float(spec[i % 32])
        bar_len = (18.0 + 135.0 * s_val + 38.0 * kick * (1 if i % 4 == 0 else 0)) * enter
        x0 = cx + math.cos(ang) * base_r
        y0 = cy + math.sin(ang) * base_r
        x1 = cx + math.cos(ang) * (base_r + bar_len)
        y1 = cy + math.sin(ang) * (base_r + bar_len)
        col = lerp_color(COL_CYAN, COL_MAGENTA, (math.sin(ang * 2.0 + t * 2.0) + 1.0) * 0.5)
        if i % 4 == 0:
            col = lerp_color(col, COL_AMBER, 0.6)
        c.line(x0, y0, x1, y1, col, 0.78, 5.2)
        c.circle(x1, y1, 3.2, fill_col=COL_WHITE, fill_alpha=0.85)

    # Inner Multi-Head Attention Chord Ring (24 token nodes on r = 215)
    n_tokens = 24
    tok_r = 215.0 * enter
    tok_pts = []
    for i in range(n_tokens):
        ang = (2.0 * math.pi * i / n_tokens) - t * 0.35
        tok_pts.append((cx + math.cos(ang) * tok_r, cy + math.sin(ang) * tok_r, ang))

    # Quadratic/Cubic Bezier Attention Arcs through center
    for i in range(n_tokens):
        for step_j in (5, 9, 11):
            j = (i + step_j) % n_tokens
            w_att = 0.5 + 0.5 * math.sin(t * 3.8 + i * 0.7 + j * 0.3)
            if w_att > 0.62:
                x0, y0, _ = tok_pts[i]
                x1, y1, _ = tok_pts[j]
                col_a = COL_CYAN if step_j == 5 else (COL_MAGENTA if step_j == 9 else COL_AMBER)
                alpha_a = (w_att - 0.60) * 1.4 * enter
                c.cubic_bezier(
                    x0, y0,
                    lerp(x0, cx, 0.65), lerp(y0, cy, 0.65),
                    lerp(x1, cx, 0.65), lerp(y1, cy, 0.65),
                    x1, y1,
                    col_a, alpha_a, 1.8 + 2.0 * kick
                )

    for x_t, y_t, _ in tok_pts:
        c.circle(x_t, y_t, 4.5 + 2.0 * arp, fill_col=COL_WHITE, fill_alpha=0.95 * enter)

    # 3D Geodesic Latent Core Sphere inside the ring (r = 115)
    core_r = (112.0 + 18.0 * kick) * enter
    ry_a = t * 1.35
    rx_a = t * 0.85
    cy_r, sy_r = math.cos(ry_a), math.sin(ry_a)
    cx_r, sx_r = math.cos(rx_a), math.sin(rx_a)
    p_nodes = []
    for nx, ny, nz in SPHERE_NODES:
        x1 = nx * cy_r + nz * sy_r
        z1 = -nx * sy_r + nz * cy_r
        y2 = ny * cx_r - z1 * sx_r
        z2 = ny * sx_r + z1 * cx_r
        p_nodes.append((cx + x1 * core_r, cy + y2 * core_r, z2))

    for i, j in SPHERE_EDGES:
        z_avg = 0.5 * (p_nodes[i][2] + p_nodes[j][2])
        if z_avg > -0.25:
            c.line(p_nodes[i][0], p_nodes[i][1], p_nodes[j][0], p_nodes[j][1], COL_CYAN, 0.32 * (z_avg + 0.8) * enter, 1.5)
    c.set_operator_over()

    # Left & Right Floating Telemetry HUD Panels for Act IV
    # Left Panel: Attention Matrix Heatmap (8x8 live grid)
    lx, ly = 88.0, 250.0
    c.rounded_rect(lx, ly, 340, 540, 14, fill_col=COL_DARK_PANEL, fill_alpha=0.84 * enter, stroke_col=COL_CYAN, stroke_alpha=0.55 * enter, line_w=1.6)
    draw_text(draw, (lx + 24, ly + 22), "SELF-ATTENTION WEIGHTS", "plex_bold", 15, 700, COL_CYAN, enter, tracking=1.5)
    draw_text(draw, (lx + 24, ly + 46), "SOFTMAX(Q · K^T / √d_k)", "mono", 15, 500, COL_SLATE, enter)

    grid_x0, grid_y0 = lx + 28, ly + 86
    cell_s = 34.0
    for r_i in range(8):
        for c_i in range(8):
            val = 0.5 + 0.5 * math.sin(t * 4.2 + r_i * 0.9 - c_i * 1.1)
            if r_i == c_i:
                val = max(val, 0.78)
            col_cell = lerp_color(COL_VIOLET, COL_CYAN, val)
            if val > 0.82:
                col_cell = lerp_color(COL_CYAN, COL_AMBER, (val - 0.82) * 5.0)
            c.rounded_rect(
                grid_x0 + c_i * (cell_s + 2),
                grid_y0 + r_i * (cell_s + 2),
                cell_s, cell_s, 4.0,
                fill_col=col_cell,
                fill_alpha=(0.18 + 0.78 * (val ** 2)) * enter,
            )

    draw_text(draw, (lx + 24, ly + 395), "ACTIVE HEADS: 128 // FP8", "plex_mono", 14, 500, COL_MINT, enter)
    draw_text(draw, (lx + 24, ly + 425), f"CONTEXT WINDOW: {int(32768 + lt*8192):,} TOK", "plex_mono", 14, 500, COL_WHITE, enter)
    draw_text(draw, (lx + 24, ly + 455), f"ENTROPY H(X): {1.42 + 0.18*math.sin(t*3):.3f} NATS", "plex_mono", 14, 500, COL_AMBER, enter)
    # Animated mini progress bar
    c.rounded_rect(lx + 24, ly + 492, 292, 12, 6, fill_col=COL_BG_DEEP, fill_alpha=0.9 * enter, stroke_col=COL_SLATE, stroke_alpha=0.4 * enter, line_w=1.0)
    c.rounded_rect(lx + 24, ly + 492, 292 * (0.65 + 0.30 * math.sin(t * 2.5)), 12, 6, fill_col=COL_CYAN, fill_alpha=0.9 * enter)

    # Right Panel: Latent Space Eigenvalues & Token Stream
    rx, ry = WIDTH - 428.0, 250.0
    c.rounded_rect(rx, ry, 340, 540, 14, fill_col=COL_DARK_PANEL, fill_alpha=0.84 * enter, stroke_col=COL_MAGENTA, stroke_alpha=0.55 * enter, line_w=1.6)
    draw_text(draw, (rx + 24, ry + 22), "SPECTRAL EIGENMODES", "plex_bold", 15, 700, COL_MAGENTA, enter, tracking=1.5)
    draw_text(draw, (rx + 24, ry + 46), "LATENT MANIFOLD PROJECTION", "mono", 15, 500, COL_SLATE, enter)

    eigen_labels = [
        ("MODE 01 // TEMPORAL", 0.94, COL_CYAN),
        ("MODE 02 // HARMONIC", 0.86, COL_VIOLET),
        ("MODE 03 // SPATIAL",  0.79, COL_MAGENTA),
        ("MODE 04 // SEMANTIC", 0.91, COL_AMBER),
        ("MODE 05 // KINETIC",  0.83, COL_MINT),
        ("MODE 06 // ACOUSTIC", 0.96, COL_CYAN),
    ]
    for e_i, (elbl, ebase, ecol) in enumerate(eigen_labels):
        ey = ry + 96 + e_i * 68
        eval_cur = clamp(ebase + 0.08 * math.sin(t * 5.0 + e_i) + 0.05 * kick, 0.1, 1.0)
        draw_text(draw, (rx + 24, ey), elbl, "plex_mono", 14, 500, COL_WHITE, enter)
        draw_text(draw, (rx + 316, ey), f"{eval_cur*100:.1f}%", "plex_bold", 14, 700, ecol, enter, anchor="ra")
        c.rounded_rect(rx + 24, ey + 26, 292, 12, 6, fill_col=COL_BG_DEEP, fill_alpha=0.9 * enter)
        c.rounded_rect(rx + 24, ey + 26, 292 * eval_cur, 12, 6, fill_col=ecol, fill_alpha=0.92 * enter)


# =============================================================================
# ACT V (30.0s - 36.0s): TELEMETRY CONVERGENCE // SYSTEM BENCHMARKS
# =============================================================================
def render_act5_telemetry(
    c: CairoCanvas,
    draw: ImageDraw.ImageDraw,
    t: float,
    kick: float,
    arp: float,
    spec: list[float],
):
    lt = t - 30.0  # 0.0 -> 6.0s
    enter = ease_out_cubic(clamp(lt / 0.8, 0.0, 1.0))

    draw_text(draw, (88, 92), "SYSTEM TELEMETRY // CONVERGENCE & BENCHMARKS", "unbounded", 28, 700, COL_WHITE, enter, tracking=2.0)
    draw_text(draw, (88, 132), "REAL-TIME RENDERING DIAGNOSTICS  •  MULTI-CORE PIPELINE VERIFICATION", "plex_mono", 15, 500, COL_MINT, enter * 0.9, tracking=1.5)

    # Top Row: 3 Precision Radial Arc Gauges
    gauges = [
        ("MULTI-CORE THROUGHPUT", 118.4 + 4.2 * kick, "FPS", 0.92, COL_CYAN,    350.0),
        ("AUDIO-VISUAL SYNC LOCK", 99.98,             "%",   0.999, COL_MINT,   960.0),
        ("SPECTRAL DYNAMIC RANGE", 96.4 + 1.8 * arp,  "dB",  0.88, COL_AMBER,  1570.0),
    ]
    for g_i, (g_title, g_val, g_unit, g_norm, g_col, gx) in enumerate(gauges):
        gy = 335.0
        c.rounded_rect(gx - 255, 180, 510, 310, 14, fill_col=COL_DARK_PANEL, fill_alpha=0.86 * enter, stroke_col=g_col, stroke_alpha=0.55 * enter, line_w=1.6)
        c.radial_glow(gx, gy + 15, 160.0, g_col, 0.18 * enter)

        # Background track arc (135 deg to 405 deg)
        a_start = math.radians(135.0)
        a_full = math.radians(405.0)
        prog = g_norm * ease_out_expo(clamp((lt - g_i * 0.15) / 1.4, 0.0, 1.0))
        a_cur = lerp(a_start, a_full, prog)

        c.arc(gx, gy + 22, 102.0, a_start, a_full, COL_SLATE, 0.25 * enter, 14.0)
        c.set_operator_add()
        c.arc(gx, gy + 22, 102.0, a_start, a_cur, g_col, 0.90 * enter, 14.0)
        c.set_operator_over()

        draw_text(draw, (gx, 202), g_title, "plex_bold", 15, 700, g_col, enter, anchor="ma", tracking=1.5)
        val_display = g_val * ease_out_cubic(clamp(lt / 1.1, 0.0, 1.0))
        val_str = f"{val_display:.1f}" if g_unit != "%" else f"{val_display:.2f}"
        draw_text(draw, (gx, gy + 2), val_str, "unbounded", 40, 800, COL_WHITE, enter, anchor="ma")
        draw_text(draw, (gx, gy + 54), g_unit, "plex_mono", 16, 600, COL_SLATE, enter, anchor="ma", tracking=2.0)

    # Bottom Left: Multi-Horizon Convergence Curve Chart
    cx0, cy0, cw, ch = 95.0, 525.0, 1060.0, 420.0
    c.rounded_rect(cx0, cy0, cw, ch, 14, fill_col=COL_DARK_PANEL, fill_alpha=0.86 * enter, stroke_col=COL_CYAN, stroke_alpha=0.50 * enter, line_w=1.6)
    draw_text(draw, (cx0 + 28, cy0 + 22), "STOCHASTIC GRADIENT & HARMONIC LOSS CONVERGENCE (LOG SCALE)", "plex_bold", 15, 700, COL_CYAN, enter, tracking=1.5)

    plot_x0, plot_y0 = cx0 + 70, cy0 + 75
    plot_w, plot_h = cw - 110, ch - 115
    # Grid lines inside chart
    for gy_i in range(5):
        py_line = plot_y0 + (gy_i / 4.0) * plot_h
        c.line(plot_x0, py_line, plot_x0 + plot_w, py_line, COL_SLATE, 0.22 * enter, 1.0)
        lbl_val = f"10^-{gy_i}"
        draw_text(draw, (plot_x0 - 12, py_line - 8), lbl_val, "plex_mono", 12, 500, COL_SLATE, enter, anchor="ra")

    # 3 Animated Horizon Curves
    curves = [
        ("GEMINI 4 ARGON (ZERO-COPY SIMD)", COL_CYAN,    3.4, 0.04),
        ("BASELINE VECTOR PIPELINE",        COL_VIOLET,  1.9, 0.16),
        ("UNOPTIMIZED RASTERIZER",          COL_MAGENTA, 1.1, 0.32),
    ]
    n_pts = int(lerp(10, 120, clamp(lt / 3.5, 0.0, 1.0)))
    for c_idx, (c_name, c_col, rate, asym) in enumerate(curves):
        pts = []
        for i in range(n_pts):
            u_x = i / 119.0
            px = plot_x0 + u_x * plot_w
            # Exponential decay + high-frequency damped oscillation
            loss = asym + (1.0 - asym) * math.exp(-rate * u_x * 1.4) + 0.035 * math.exp(-u_x * 2.0) * math.sin(u_x * 28.0 + t * 4.0 + c_idx)
            py = plot_y0 + (1.0 - clamp(loss, 0.02, 0.98)) * plot_h
            pts.append((px, py))

        # Fill subtle area under the hero cyan curve
        if c_idx == 0 and len(pts) >= 2:
            poly_fill = [(pts[0][0], plot_y0 + plot_h)] + pts + [(pts[-1][0], plot_y0 + plot_h)]
            c.polyline(poly_fill, c_col, 0.0, 1.0, closed=True, fill_col=c_col, fill_alpha=0.14 * enter)

        c.polyline(pts, c_col, 0.92 * enter, 3.0 if c_idx == 0 else 2.0)
        if pts:
            tx, ty = pts[-1]
            c.circle(tx, ty, 5.0, fill_col=COL_WHITE, fill_alpha=enter)
            c.radial_glow(tx, ty, 22.0, c_col, 0.65 * enter)

        # Legend entry top-right of chart
        lx_leg = plot_x0 + plot_w - 310
        ly_leg = plot_y0 + 10 + c_idx * 26
        c.line(lx_leg, ly_leg + 8, lx_leg + 24, ly_leg + 8, c_col, 0.95 * enter, 3.0)
        draw_text(draw, (lx_leg + 34, ly_leg), c_name, "plex_mono", 13, 600, c_col, enter)

    # Bottom Right: Verification Checklist & Spec Matrix
    bx0, by0, bw, bh = 1185.0, 525.0, 640.0, 420.0
    c.rounded_rect(bx0, by0, bw, bh, 14, fill_col=COL_DARK_PANEL, fill_alpha=0.86 * enter, stroke_col=COL_MINT, stroke_alpha=0.50 * enter, line_w=1.6)
    draw_text(draw, (bx0 + 28, by0 + 22), "PIPELINE VERIFICATION MATRIX", "plex_bold", 15, 700, COL_MINT, enter, tracking=1.5)

    checks = [
        ("RESOLUTION // ASPECT",   "1920x1080 (16:9 FULL HD)",   0.2),
        ("FRAME RATE // CADENCE",  "30.00 FPS (1,260 FRAMES)",   0.6),
        ("VECTOR ANTI-ALIASING",   "LIBCAIRO.SO.2 + HDR BLOOM",  1.0),
        ("PROCEDURAL SOUNDTRACK",  "48KHZ STEREO WAV + AAC",     1.4),
        ("REPRODUCIBLE BUILD",     "./make.sh DETERMINISTIC",    1.8),
        ("COPYRIGHT / EXTERNAL",   "0% EXTERNAL — 100% CODE",    2.2),
    ]
    for chk_i, (k_str, v_str, delay_s) in enumerate(checks):
        row_a = smoothstep(delay_s, delay_s + 0.5, lt)
        ry_row = by0 + 68 + chk_i * 54
        c.rounded_rect(bx0 + 24, ry_row, bw - 48, 44, 8, fill_col=COL_BG_DEEP, fill_alpha=0.75 * row_a, stroke_col=COL_SLATE, stroke_alpha=0.3 * row_a, line_w=1.0)
        # Glowing green status badge
        c.circle(bx0 + 48, ry_row + 22, 7.0, fill_col=COL_MINT, fill_alpha=0.95 * row_a)
        draw_text(draw, (bx0 + 68, ry_row + 13), k_str, "plex_mono", 14, 600, COL_SLATE, row_a)
        draw_text(draw, (bx0 + bw - 42, ry_row + 13), v_str, "plex_bold", 14, 700, COL_WHITE, row_a, anchor="ra")


# =============================================================================
# ACT VI (36.0s - 42.0s): GRAND FINALE // ARGON 4 SHOWCASE LOCKUP
# =============================================================================
def render_act6_finale(
    c: CairoCanvas,
    draw: ImageDraw.ImageDraw,
    t: float,
    kick: float,
    arp: float,
    spec: list[float],
):
    lt = t - 36.0  # 0.0 -> 6.0s
    enter = ease_out_expo(clamp(lt / 1.2, 0.0, 1.0))
    # Smooth fade-to-black only in the final 0.85s (41.15s -> 42.0s)
    fade_out = clamp((DURATION - t) / 0.85, 0.0, 1.0)
    vis = enter * fade_out

    cx, cy = 960.0, 460.0
    c.radial_glow(cx, cy, 520.0 * vis, COL_CYAN, 0.25 * vis)
    c.radial_glow(cx, cy, 340.0 * vis, COL_VIOLET, 0.28 * vis)

    # Expanding shockwave rings from the 36.0s impact
    if lt < 2.5:
        sw_u = lt / 2.5
        sw_r = 120.0 + sw_u * 780.0
        c.circle(cx, cy, sw_r, stroke_col=COL_CYAN, stroke_alpha=(1.0 - sw_u) * 0.65, line_w=3.0)

    # Noble Gas Argon-18 Emblem & Concentric Harmonic Rings
    for r_ring, spd, col in [(280.0, 0.25, COL_SLATE), (225.0, -0.4, COL_VIOLET), (170.0, 0.6, COL_CYAN)]:
        c.save()
        c.translate(cx, cy)
        c.rotate(t * spd)
        c.set_dash([28.0, 16.0], 0.0)
        c.circle(0, 0, r_ring * enter, stroke_col=col, stroke_alpha=0.55 * vis, line_w=2.0)
        c.set_dash([])
        c.restore()

    # Orbiting 2-8-8 electrons in golden finale harmony
    c.set_operator_add()
    for s_rad, n_el, spd, col in [(70.0, 2, 1.5, COL_AMBER), (120.0, 8, -0.95, COL_VIOLET), (170.0, 8, 0.65, COL_CYAN)]:
        sr = s_rad * enter
        c.circle(cx, cy, sr, stroke_col=col, stroke_alpha=0.35 * vis, line_w=1.5)
        for i in range(n_el):
            ang = (2.0 * math.pi * i / n_el) + t * spd
            ex = cx + math.cos(ang) * sr
            ey = cy + math.sin(ang) * sr
            c.circle(ex, ey, 8.0, fill_col=col, fill_alpha=0.40 * vis)
            c.circle(ex, ey, 3.8, fill_col=COL_WHITE, fill_alpha=0.95 * vis)
    c.set_operator_over()

    # Central Periodic Element Emblem Box
    box_w, box_h = 124.0, 124.0
    c.rounded_rect(cx - box_w * 0.5, cy - box_h * 0.5, box_w, box_h, 16.0, fill_col=COL_BG_DEEP, fill_alpha=0.92 * vis, stroke_col=COL_CYAN, stroke_alpha=0.90 * vis, line_w=2.5)
    draw_text(draw, (cx - 44, cy - 48), "18", "plex_bold", 16, 700, COL_CYAN, vis)
    draw_text(draw, (cx + 44, cy - 48), "39.95", "plex_mono", 13, 500, COL_SLATE, vis, anchor="ra")
    draw_text(draw, (cx, cy - 18), "Ar", "unbounded", 46, 800, COL_WHITE, vis, anchor="ma")
    draw_text(draw, (cx, cy + 34), "ARGON", "plex_bold", 13, 700, COL_AMBER, vis, anchor="ma", tracking=3.0)

    # Grand Finale Title Card Lockup
    c.rounded_rect(360, 745, 1200, 210, 18, fill_col=COL_DARK_PANEL, fill_alpha=0.86 * vis, stroke_col=COL_CYAN, stroke_alpha=0.60 * vis, line_w=1.8)
    draw_text(draw, (960, 772), "GEMINI 4 ARGON", "unbounded", 58, 800, COL_WHITE, vis, anchor="ma", tracking=6.0)
    draw_text(
        draw,
        (960, 852),
        "BUILT ENTIRELY WITH CODE IN CURSOR CLI  •  PYTHON + LIBCAIRO + FFMPEG",
        "plex_bold",
        17,
        700,
        COL_CYAN,
        vis,
        anchor="ma",
        tracking=2.8,
    )
    draw_text(
        draw,
        (960, 898),
        "1920×1080 @ 30 FPS   //   48KHZ PROCEDURAL SYNTHESIS   //   ZERO EXTERNAL ASSETS",
        "plex_mono",
        14,
        500,
        COL_AMBER,
        vis * 0.9,
        anchor="ma",
        tracking=2.0,
    )


# =============================================================================
# MASTER FRAME DISPATCHER & TRANSITION WIPE COMPOSITOR
# =============================================================================
ACT_INFO = [
    (0.0,  6.0,  1, "ATOMIC IGNITION",    COL_CYAN),
    (6.0,  14.0, 2, "LIVING COMPILER",    COL_VIOLET),
    (14.0, 22.0, 3, "HARMONIC GEOMETRY",  COL_AMBER),
    (22.0, 30.0, 4, "NEURAL TOPOLOGY",    COL_MAGENTA),
    (30.0, 36.0, 5, "SYSTEM TELEMETRY",   COL_MINT),
    (36.0, 42.0, 6, "NOBLE RESOLUTION",   COL_CYAN),
]


def render_single_frame(frame_idx: int, env_data: dict, return_pil: bool = False):
    """Renders frame `frame_idx` (0..1259) and returns RGB24 bytes (or PIL Image if return_pil=True)."""
    t = frame_idx / FPS
    kick = float(env_data["kick"][frame_idx])
    snare = float(env_data["snare"][frame_idx])
    arp = float(env_data["arp"][frame_idx])
    rms = float(env_data["rms"][frame_idx])
    spec = env_data["spectrum"][frame_idx]

    # Determine current Act
    act_num = 6
    act_name = "NOBLE RESOLUTION"
    act_col = COL_CYAN
    for s_t, e_t, a_n, a_name, a_col in ACT_INFO:
        if s_t <= t < e_t:
            act_num = a_n
            act_name = a_name
            act_col = a_col
            break

    with CairoCanvas(WIDTH, HEIGHT) as c:
        # Deep space background gradient with subtle audio-reactive nebula shift
        glow_x = 960.0 + 180.0 * math.sin(t * 0.4)
        glow_y = 540.0 + 90.0 * math.cos(t * 0.3)
        c.clear_gradient_bg(COL_BG_DEEP, COL_BG_MID, (glow_x, glow_y), act_col, 0.14 + 0.06 * kick)

        # Global architectural grid & depth-of-field particles
        draw_global_grid_and_particles(c, t, kick, act_col)

        # Convert to PIL RGBA lazily after vector pass? Wait: scenes draw both Cairo vectors and PIL text!
        # Let's collect text draw commands in a list during the Cairo vector pass so all Cairo vectors
        # execute first, then convert once to PIL RGBA and render all crisp typography on top!
        text_queue = []

        class DeferredTextDraw:
            def text_cmd(self, *args, **kwargs):
                text_queue.append((args, kwargs))

        # Monkey-patch helper or pass a lightweight proxy that records draw_text calls
        # Actually, let's check: does any scene draw Cairo vectors AFTER draw_text?
        # In our scene functions, we can pass a PIL RGBA overlay or record text calls!
        # Recording text calls and executing them right after `c.to_pil_rgba()` is 100% zero-overhead
        # and guarantees that Cairo vectors never overwrite text!
        pass

    # Let's execute with a clean text-recording list so Cairo draws all vector layers first,
    # and Pillow draws all crisp variable-font typography on top of the flushed vector surface.
    return _render_frame_layered(frame_idx, t, kick, snare, arp, rms, spec, act_num, act_name, act_col, return_pil)


class _TextRecorder:
    """Records typography calls during scene traversal and renders them onto the PIL RGBA layer."""
    def __init__(self):
        self.ops = []

    def text(self, xy, text, font=None, fill=None, anchor=None):
        self.ops.append((xy, text, font, fill, anchor))

    def replay(self, pil_draw: ImageDraw.ImageDraw):
        for xy, text, font, fill, anchor in self.ops:
            pil_draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _render_frame_layered(frame_idx, t, kick, snare, arp, rms, spec, act_num, act_name, act_col, return_pil=False):
    rec = _TextRecorder()

    with CairoCanvas(WIDTH, HEIGHT) as c:
        glow_x = 960.0 + 180.0 * math.sin(t * 0.4)
        glow_y = 540.0 + 90.0 * math.cos(t * 0.3)
        c.clear_gradient_bg(COL_BG_DEEP, COL_BG_MID, (glow_x, glow_y), act_col, 0.14 + 0.06 * kick)
        draw_global_grid_and_particles(c, t, kick, act_col)

        # Dispatch active scene
        if t < 6.0:
            render_act1_ignition(c, rec, t, kick, arp, spec)
        elif t < 14.0:
            render_act2_compiler(c, rec, t, kick, arp, spec)
        elif t < 22.0:
            render_act3_harmonics(c, rec, t, kick, arp, spec)
        elif t < 30.0:
            render_act4_neural(c, rec, t, kick, snare, arp, spec)
        elif t < 36.0:
            render_act5_telemetry(c, rec, t, kick, arp, spec)
        else:
            render_act6_finale(c, rec, t, kick, arp, spec)

        # Scene boundary laser scanner wipe at t = 6.0, 14.0, 22.0, 30.0, 36.0
        for tr_t in (6.0, 14.0, 22.0, 30.0, 36.0):
            dt_tr = t - tr_t
            if -0.25 <= dt_tr <= 0.35:
                u_tr = (dt_tr + 0.25) / 0.60
                scan_x = ease_in_out_cubic(u_tr) * WIDTH
                c.set_operator_add()
                c.radial_glow(scan_x, 540.0, 320.0, COL_CYAN, 0.45 * math.sin(u_tr * math.pi))
                c.line(scan_x, 64, scan_x, HEIGHT - 64, COL_WHITE, 0.90 * math.sin(u_tr * math.pi), 3.5)
                c.set_operator_over()

        # Persistent broadcast HUD chrome
        draw_hud_chrome(c, rec, frame_idx, t, act_num, act_name, kick, rms, spec)

        pil_rgba = c.to_pil_rgba()

    # Replay all recorded typography onto the PIL RGBA image
    pil_draw = ImageDraw.Draw(pil_rgba)
    rec.replay(pil_draw)

    # Apply global fade-to-black at the very end (41.2s -> 42.0s)
    if t > 41.2:
        fade_k = clamp((DURATION - t) / 0.8, 0.0, 1.0)
        pil_rgba = Image.eval(pil_rgba, lambda px: int(px * fade_k))

    if return_pil:
        # Apply bloom and return PIL Image
        raw_bytes = finalize_frame(pil_rgba, bloom_intensity=0.55 + 0.20 * kick)
        return Image.frombytes("RGB", (WIDTH, HEIGHT), raw_bytes)

    return finalize_frame(pil_rgba, bloom_intensity=0.55 + 0.20 * kick)
