"""
Zero-copy ctypes wrapper for libcairo.so.2 + Pillow typography & post-processing engine.
Provides ultra-crisp 1080p anti-aliased vector drawing, additive energy glows,
linear/radial gradients, dashed splines, and multi-weight typography.
"""

import ctypes
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

# Load libcairo
_cairo = ctypes.CDLL("libcairo.so.2")

# Constants
CAIRO_FORMAT_ARGB32 = 0
CAIRO_OPERATOR_OVER = 2
CAIRO_OPERATOR_ADD = 12
CAIRO_LINE_CAP_BUTT = 0
CAIRO_LINE_CAP_ROUND = 1
CAIRO_LINE_CAP_SQUARE = 2
CAIRO_LINE_JOIN_MITER = 0
CAIRO_LINE_JOIN_ROUND = 1
CAIRO_LINE_JOIN_BEVEL = 2

# Function signatures
_cairo.cairo_image_surface_create_for_data.argtypes = [
    ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]
_cairo.cairo_image_surface_create_for_data.restype = ctypes.c_void_p

_cairo.cairo_create.argtypes = [ctypes.c_void_p]
_cairo.cairo_create.restype = ctypes.c_void_p

_cairo.cairo_destroy.argtypes = [ctypes.c_void_p]
_cairo.cairo_surface_destroy.argtypes = [ctypes.c_void_p]
_cairo.cairo_surface_flush.argtypes = [ctypes.c_void_p]

_cairo.cairo_save.argtypes = [ctypes.c_void_p]
_cairo.cairo_restore.argtypes = [ctypes.c_void_p]

_cairo.cairo_set_operator.argtypes = [ctypes.c_void_p, ctypes.c_int]
_cairo.cairo_set_source_rgba.argtypes = [
    ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]
_cairo.cairo_set_source.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

_cairo.cairo_set_line_width.argtypes = [ctypes.c_void_p, ctypes.c_double]
_cairo.cairo_set_line_cap.argtypes = [ctypes.c_void_p, ctypes.c_int]
_cairo.cairo_set_line_join.argtypes = [ctypes.c_void_p, ctypes.c_int]
_cairo.cairo_set_dash.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_double
]

_cairo.cairo_new_path.argtypes = [ctypes.c_void_p]
_cairo.cairo_close_path.argtypes = [ctypes.c_void_p]
_cairo.cairo_move_to.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
_cairo.cairo_line_to.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
_cairo.cairo_curve_to.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double,
]
_cairo.cairo_arc.argtypes = [
    ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]
_cairo.cairo_rectangle.argtypes = [
    ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]

_cairo.cairo_stroke.argtypes = [ctypes.c_void_p]
_cairo.cairo_stroke_preserve.argtypes = [ctypes.c_void_p]
_cairo.cairo_fill.argtypes = [ctypes.c_void_p]
_cairo.cairo_fill_preserve.argtypes = [ctypes.c_void_p]
_cairo.cairo_clip.argtypes = [ctypes.c_void_p]

_cairo.cairo_translate.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
_cairo.cairo_scale.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double]
_cairo.cairo_rotate.argtypes = [ctypes.c_void_p, ctypes.c_double]

_cairo.cairo_pattern_create_linear.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double
]
_cairo.cairo_pattern_create_linear.restype = ctypes.c_void_p

_cairo.cairo_pattern_create_radial.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_cairo.cairo_pattern_create_radial.restype = ctypes.c_void_p

_cairo.cairo_pattern_add_color_stop_rgba.argtypes = [
    ctypes.c_void_p, ctypes.c_double,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
]
_cairo.cairo_pattern_destroy.argtypes = [ctypes.c_void_p]


# Color Palette Constants (Normalized 0..1 RGB tuples)
COL_BG_DEEP    = (0.016, 0.024, 0.055)   # #04060E
COL_BG_MID     = (0.035, 0.055, 0.118)   # #090E1E
COL_CYAN       = (0.000, 0.941, 1.000)   # #00F0FF
COL_VIOLET     = (0.659, 0.333, 0.969)   # #A855F7
COL_MAGENTA    = (0.957, 0.247, 0.549)   # #F43F8C
COL_AMBER      = (1.000, 0.690, 0.149)   # #FFB026
COL_MINT       = (0.063, 0.929, 0.635)   # #10EDA2
COL_WHITE      = (0.961, 0.973, 1.000)   # #F5F8FF
COL_SLATE      = (0.450, 0.530, 0.680)   # #7387AD
COL_DARK_PANEL = (0.031, 0.047, 0.098)   # #080C19


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = clamp((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 4.0 * t * t * t if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_out_expo(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1.0 if t >= 1.0 else 1.0 - (2.0 ** (-10.0 * t))


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    t = clamp(t, 0.0, 1.0)
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )


def rgb255(c: tuple, alpha: float = 1.0) -> tuple:
    return (
        int(clamp(c[0], 0.0, 1.0) * 255),
        int(clamp(c[1], 0.0, 1.0) * 255),
        int(clamp(c[2], 0.0, 1.0) * 255),
        int(clamp(alpha, 0.0, 1.0) * 255),
    )


# Font cache per process
_FONT_CACHE = {}

FONT_PATHS = {
    "display": "/usr/share/fonts/truetype/sand-box/google/Syne/Syne-VariableFont_wght.ttf",
    "unbounded": "/usr/share/fonts/truetype/sand-box/google/Unbounded/Unbounded-VariableFont_wght.ttf",
    "sans": "/usr/share/fonts/truetype/sand-box/google/Space Grotesk/SpaceGrotesk-VariableFont_wght.ttf",
    "mono": "/usr/share/fonts/truetype/sand-box/google/JetBrains Mono/JetBrainsMono-VariableFont_wght.ttf",
    "chakra_bold": "/usr/share/fonts/truetype/sand-box/google/Chakra Petch/ChakraPetch-Bold.ttf",
    "chakra_med": "/usr/share/fonts/truetype/sand-box/google/Chakra Petch/ChakraPetch-Medium.ttf",
    "plex_mono": "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Medium.ttf",
    "plex_bold": "/usr/share/fonts/truetype/sand-box/google/IBM Plex Mono/IBMPlexMono-Bold.ttf",
}


def get_font(family: str, size: int, weight: int = 600) -> ImageFont.FreeTypeFont:
    key = (family, size, weight)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = FONT_PATHS.get(family, FONT_PATHS["sans"])
    font = ImageFont.truetype(path, size)
    try:
        axes = font.get_variation_axes()
        if axes:
            lo = axes[0]["minimum"]
            hi = axes[0]["maximum"]
            w = max(lo, min(hi, weight))
            font.set_variation_by_axes([w])
    except Exception:
        pass
    _FONT_CACHE[key] = font
    return font


class CairoCanvas:
    """High-precision 1080p Cairo vector canvas backed by a zero-copy NumPy BGRA array."""

    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
        self.buf = np.zeros((height, width, 4), dtype=np.uint8)
        # Initialize opaque alpha channel
        self.buf[:, :, 3] = 255
        self.surf = _cairo.cairo_image_surface_create_for_data(
            self.buf.ctypes.data, CAIRO_FORMAT_ARGB32, width, height, width * 4
        )
        self.ctx = _cairo.cairo_create(self.surf)
        _cairo.cairo_set_line_cap(self.ctx, CAIRO_LINE_CAP_ROUND)
        _cairo.cairo_set_line_join(self.ctx, CAIRO_LINE_JOIN_ROUND)

    def close(self):
        if self.ctx:
            _cairo.cairo_destroy(self.ctx)
            self.ctx = None
        if self.surf:
            _cairo.cairo_surface_destroy(self.surf)
            self.surf = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def save(self):
        _cairo.cairo_save(self.ctx)

    def restore(self):
        _cairo.cairo_restore(self.ctx)

    def set_operator_over(self):
        _cairo.cairo_set_operator(self.ctx, CAIRO_OPERATOR_OVER)

    def set_operator_add(self):
        _cairo.cairo_set_operator(self.ctx, CAIRO_OPERATOR_ADD)

    def set_color(self, c: tuple, alpha: float = 1.0):
        _cairo.cairo_set_source_rgba(self.ctx, float(c[0]), float(c[1]), float(c[2]), float(clamp(alpha, 0.0, 1.0)))

    def set_line_width(self, w: float):
        _cairo.cairo_set_line_width(self.ctx, float(w))

    def set_dash(self, dashes: list[float], offset: float = 0.0):
        if not dashes:
            _cairo.cairo_set_dash(self.ctx, None, 0, 0.0)
        else:
            arr = (ctypes.c_double * len(dashes))(*dashes)
            _cairo.cairo_set_dash(self.ctx, arr, len(dashes), float(offset))

    def translate(self, tx: float, ty: float):
        _cairo.cairo_translate(self.ctx, float(tx), float(ty))

    def scale(self, sx: float, sy: float):
        _cairo.cairo_scale(self.ctx, float(sx), float(sy))

    def rotate(self, angle_rad: float):
        _cairo.cairo_rotate(self.ctx, float(angle_rad))

    def clear_gradient_bg(self, top_col: tuple, bot_col: tuple, glow_pos: tuple = (960.0, 540.0), glow_col: tuple = COL_VIOLET, glow_alpha: float = 0.16):
        """Fill background with deep vertical gradient + radial atmospheric nebula."""
        self.set_operator_over()
        pat = _cairo.cairo_pattern_create_linear(0.0, 0.0, 0.0, float(self.height))
        _cairo.cairo_pattern_add_color_stop_rgba(pat, 0.0, top_col[0], top_col[1], top_col[2], 1.0)
        _cairo.cairo_pattern_add_color_stop_rgba(pat, 1.0, bot_col[0], bot_col[1], bot_col[2], 1.0)
        _cairo.cairo_set_source(self.ctx, pat)
        _cairo.cairo_rectangle(self.ctx, 0.0, 0.0, float(self.width), float(self.height))
        _cairo.cairo_fill(self.ctx)
        _cairo.cairo_pattern_destroy(pat)

        if glow_alpha > 0.001:
            rpat = _cairo.cairo_pattern_create_radial(
                glow_pos[0], glow_pos[1], 20.0,
                glow_pos[0], glow_pos[1], 780.0
            )
            _cairo.cairo_pattern_add_color_stop_rgba(rpat, 0.0, glow_col[0], glow_col[1], glow_col[2], glow_alpha)
            _cairo.cairo_pattern_add_color_stop_rgba(rpat, 0.5, glow_col[0], glow_col[1], glow_col[2], glow_alpha * 0.35)
            _cairo.cairo_pattern_add_color_stop_rgba(rpat, 1.0, glow_col[0], glow_col[1], glow_col[2], 0.0)
            _cairo.cairo_set_source(self.ctx, rpat)
            _cairo.cairo_rectangle(self.ctx, 0.0, 0.0, float(self.width), float(self.height))
            _cairo.cairo_fill(self.ctx)
            _cairo.cairo_pattern_destroy(rpat)

    def radial_glow(self, x: float, y: float, r: float, col: tuple, alpha: float = 0.25):
        if alpha <= 0.002 or r <= 1.0:
            return
        self.save()
        self.set_operator_add()
        rpat = _cairo.cairo_pattern_create_radial(x, y, 0.0, x, y, r)
        _cairo.cairo_pattern_add_color_stop_rgba(rpat, 0.0, col[0], col[1], col[2], alpha)
        _cairo.cairo_pattern_add_color_stop_rgba(rpat, 0.4, col[0], col[1], col[2], alpha * 0.4)
        _cairo.cairo_pattern_add_color_stop_rgba(rpat, 1.0, col[0], col[1], col[2], 0.0)
        _cairo.cairo_set_source(self.ctx, rpat)
        _cairo.cairo_arc(self.ctx, x, y, r, 0.0, 2.0 * math.pi)
        _cairo.cairo_fill(self.ctx)
        _cairo.cairo_pattern_destroy(rpat)
        self.restore()

    def circle(self, x: float, y: float, r: float, fill_col: tuple = None, fill_alpha: float = 1.0, stroke_col: tuple = None, stroke_alpha: float = 1.0, line_w: float = 2.0):
        if r <= 0.0:
            return
        _cairo.cairo_new_path(self.ctx)
        _cairo.cairo_arc(self.ctx, x, y, r, 0.0, 2.0 * math.pi)
        if fill_col is not None and stroke_col is not None:
            self.set_color(fill_col, fill_alpha)
            _cairo.cairo_fill_preserve(self.ctx)
            self.set_color(stroke_col, stroke_alpha)
            self.set_line_width(line_w)
            _cairo.cairo_stroke(self.ctx)
        elif fill_col is not None:
            self.set_color(fill_col, fill_alpha)
            _cairo.cairo_fill(self.ctx)
        elif stroke_col is not None:
            self.set_color(stroke_col, stroke_alpha)
            self.set_line_width(line_w)
            _cairo.cairo_stroke(self.ctx)

    def arc(self, x: float, y: float, r: float, a1: float, a2: float, col: tuple, alpha: float = 1.0, line_w: float = 2.5):
        if r <= 0.0 or abs(a2 - a1) < 1e-4:
            return
        _cairo.cairo_new_path(self.ctx)
        _cairo.cairo_arc(self.ctx, x, y, r, a1, a2)
        self.set_color(col, alpha)
        self.set_line_width(line_w)
        _cairo.cairo_stroke(self.ctx)

    def line(self, x1: float, y1: float, x2: float, y2: float, col: tuple, alpha: float = 1.0, line_w: float = 2.0):
        _cairo.cairo_new_path(self.ctx)
        _cairo.cairo_move_to(self.ctx, x1, y1)
        _cairo.cairo_line_to(self.ctx, x2, y2)
        self.set_color(col, alpha)
        self.set_line_width(line_w)
        _cairo.cairo_stroke(self.ctx)

    def polyline(self, pts: list[tuple[float, float]], col: tuple, alpha: float = 1.0, line_w: float = 2.0, closed: bool = False, fill_col: tuple = None, fill_alpha: float = 0.0):
        if len(pts) < 2:
            return
        _cairo.cairo_new_path(self.ctx)
        _cairo.cairo_move_to(self.ctx, pts[0][0], pts[0][1])
        for px, py in pts[1:]:
            _cairo.cairo_line_to(self.ctx, px, py)
        if closed:
            _cairo.cairo_close_path(self.ctx)
        if fill_col is not None and fill_alpha > 0.001:
            self.set_color(fill_col, fill_alpha)
            if alpha > 0.001:
                _cairo.cairo_fill_preserve(self.ctx)
            else:
                _cairo.cairo_fill(self.ctx)
        if alpha > 0.001:
            self.set_color(col, alpha)
            self.set_line_width(line_w)
            _cairo.cairo_stroke(self.ctx)

    def rounded_rect_path(self, x: float, y: float, w: float, h: float, r: float):
        r = min(r, w * 0.5, h * 0.5)
        _cairo.cairo_new_path(self.ctx)
        _cairo.cairo_arc(self.ctx, x + w - r, y + r, r, -math.pi * 0.5, 0.0)
        _cairo.cairo_arc(self.ctx, x + w - r, y + h - r, r, 0.0, math.pi * 0.5)
        _cairo.cairo_arc(self.ctx, x + r, y + h - r, r, math.pi * 0.5, math.pi)
        _cairo.cairo_arc(self.ctx, x + r, y + r, r, math.pi, math.pi * 1.5)
        _cairo.cairo_close_path(self.ctx)

    def rounded_rect(self, x: float, y: float, w: float, h: float, r: float = 12.0, fill_col: tuple = None, fill_alpha: float = 0.85, stroke_col: tuple = None, stroke_alpha: float = 0.6, line_w: float = 1.5):
        if w <= 0.5 or h <= 0.5:
            return
        self.rounded_rect_path(x, y, w, h, r)
        if fill_col is not None and stroke_col is not None:
            self.set_color(fill_col, fill_alpha)
            _cairo.cairo_fill_preserve(self.ctx)
            self.set_color(stroke_col, stroke_alpha)
            self.set_line_width(line_w)
            _cairo.cairo_stroke(self.ctx)
        elif fill_col is not None:
            self.set_color(fill_col, fill_alpha)
            _cairo.cairo_fill(self.ctx)
        elif stroke_col is not None:
            self.set_color(stroke_col, stroke_alpha)
            self.set_line_width(line_w)
            _cairo.cairo_stroke(self.ctx)

    def cubic_bezier(self, x0: float, y0: float, x1: float, y1: float, x2: float, y2: float, x3: float, y3: float, col: tuple, alpha: float = 1.0, line_w: float = 2.0):
        _cairo.cairo_new_path(self.ctx)
        _cairo.cairo_move_to(self.ctx, x0, y0)
        _cairo.cairo_curve_to(self.ctx, x1, y1, x2, y2, x3, y3)
        self.set_color(col, alpha)
        self.set_line_width(line_w)
        _cairo.cairo_stroke(self.ctx)

    def to_pil_rgba(self) -> Image.Image:
        """Convert Cairo BGRA buffer to a Pillow RGBA image for typography and compositing."""
        _cairo.cairo_surface_flush(self.surf)
        # Guarantee alpha=255 across the canvas before handing to PIL
        self.buf[:, :, 3] = 255
        return Image.frombuffer("RGBA", (self.width, self.height), self.buf, "raw", "BGRA", 0, 1).copy()


def draw_text(
    draw: ImageDraw.ImageDraw,
    pos: tuple[float, float],
    text: str,
    family: str = "sans",
    size: int = 24,
    weight: int = 600,
    col: tuple = COL_WHITE,
    alpha: float = 1.0,
    anchor: str = "la",
    tracking: float = 0.0,
):
    """Draw anti-aliased text with optional wide letter-spacing (tracking in px)."""
    if alpha <= 0.01 or not text:
        return
    font = get_font(family, size, weight)
    rgba = rgb255(col, alpha)

    if abs(tracking) < 0.1:
        draw.text((int(round(pos[0])), int(round(pos[1]))), text, font=font, fill=rgba, anchor=anchor)
        return

    # Custom tracked character placement
    advances = [font.getlength(ch) for ch in text]
    total_w = sum(advances) + tracking * max(0, len(text) - 1)
    x, y = float(pos[0]), float(pos[1])
    if anchor.startswith("m"):
        x -= total_w * 0.5
    elif anchor.startswith("r"):
        x -= total_w

    vert_anchor = "l" + (anchor[1] if len(anchor) > 1 else "a")
    cx = x
    for ch, adv in zip(text, advances):
        draw.text((int(round(cx)), int(round(y))), ch, font=font, fill=rgba, anchor=vert_anchor)
        cx += adv + tracking


def finalize_frame(pil_rgba: Image.Image, bloom_intensity: float = 0.55) -> bytes:
    """
    Applies a fast multi-scale C-level HDR bloom pass and returns raw RGB24 bytes
    ready for piping to ffmpeg.
    """
    rgb = pil_rgba.convert("RGB")
    if bloom_intensity > 0.02:
        # Downsample 4x -> Gaussian blur -> Upsample -> Scale intensity -> Additive blend
        small = rgb.resize((480, 270), Image.Resampling.BILINEAR)
        # Isolate bright highlights for clean bloom (avoid muddying dark backgrounds)
        lut = [max(0, int((i - 110) * 1.65 * bloom_intensity)) if i > 110 else 0 for i in range(256)] * 3
        bright = small.point(lut)
        blurred = bright.filter(ImageFilter.GaussianBlur(radius=9))
        glow = blurred.resize((1920, 1080), Image.Resampling.BILINEAR)
        rgb = ImageChops.add(rgb, glow)
    return rgb.tobytes()
