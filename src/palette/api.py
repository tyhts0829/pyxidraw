from __future__ import annotations

"""High-level public API for generating color palettes.

This module provides a single function, :func:`generate_palette`, which
coordinates color-space conversion, geometric hue patterns, style
application, and sRGB gamut mapping to produce a :class:`palette.Palette`.
"""

from typing import Optional

from .color_types import Color, ColorInput
from .engine import ColorEngine, DefaultColorEngine
from .gamut import to_srgb_gamut_safe
from .harmony import PaletteType, generate_raw_colors
from .palette import Palette
from .style import PaletteStyle, apply_style


def generate_palette(
    base_color: ColorInput,
    palette_type: PaletteType,
    palette_style: PaletteStyle,
    n_colors: int = 4,
    engine: Optional[ColorEngine] = None,
) -> Palette:
    """Generate a color palette from a base color.

    Parameters
    ----------
    base_color:
        ColorInput specifying the base color (OKLCH / sRGB / HEX).
    palette_type:
        PaletteType specifying the geometric color harmony.
    palette_style:
        PaletteStyle specifying the lightness/chroma distribution.
    n_colors:
        Number of colors to generate. Some PaletteType values may restrict
        or recommend specific values; see their docstrings and error
        behavior for details.
    engine:
        Optional ColorEngine for color space conversions. If None,
        DefaultColorEngine is used.

    Returns
    -------
    Palette
        Generated palette including the chosen base color and all
        palette colors in a visually natural order.
    """
    if n_colors <= 0:
        raise ValueError("n_colors must be positive.")

    if engine is None:
        engine = DefaultColorEngine()

    base_oklch = base_color.to_oklch(engine)
    raw_colors = generate_raw_colors(engine, palette_type, base_oklch, n_colors)
    styled_oklch = apply_style(base_oklch, raw_colors, palette_style)

    colors: list[Color] = []
    for L, C, h in styled_oklch:
        r, g, b, (L_adj, C_adj, h_adj) = to_srgb_gamut_safe(engine, L, C, h)
        c = Color(oklch=(L_adj, C_adj, h_adj), srgb=(r, g, b), hex=_srgb_to_hex(r, g, b))
        colors.append(c)

    # Choose the palette color closest to base_oklch as base_color.
    base_color_obj = _choose_base_color(base_oklch, colors)
    return Palette(
        base_color=base_color_obj,
        palette_type=palette_type,
        palette_style=palette_style,
        colors=colors,
    )


def _srgb_to_hex(r: float, g: float, b: float) -> str:
    r_i = int(round(max(0.0, min(1.0, r)) * 255))
    g_i = int(round(max(0.0, min(1.0, g)) * 255))
    b_i = int(round(max(0.0, min(1.0, b)) * 255))
    return f"#{r_i:02x}{g_i:02x}{b_i:02x}"


def _choose_base_color(base_oklch, colors: list[Color]) -> Color:
    L0, C0, h0 = base_oklch
    best = None
    best_dist = float("inf")
    for c in colors:
        L, C, h = c.oklch
        dh = _hue_delta(h, h0)
        dL = L - L0
        dC = C - C0
        dist = dL * dL + dC * dC + (dh / 180.0) ** 2
        if dist < best_dist:
            best_dist = dist
            best = c
    # Fallback: first color
    return best or colors[0]


def _hue_delta(h1: float, h2: float) -> float:
    d = (h1 - h2 + 180.0) % 360.0 - 180.0
    return abs(d)
