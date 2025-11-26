from __future__ import annotations

"""Palette styles controlling lightness and chroma distribution.

This module defines :class:`PaletteStyle` and helper functions that
reshape raw OKLCH colors into visually structured palettes.
"""

import math
from enum import Enum, auto
from typing import List, Tuple


OKLCH = Tuple[float, float, float]


class PaletteStyle(Enum):
    """Lightness/chroma distribution patterns for generated palettes."""

    SQUARE = auto()
    TRIANGLE = auto()
    CIRCLE = auto()
    DIAMOND = auto()


def _clamp_L(L: float) -> float:
    return max(0.0, min(100.0, L))


def apply_style(
    base_oklch: OKLCH,
    raw_colors: List[OKLCH],
    style: PaletteStyle,
) -> List[OKLCH]:
    """Apply palette style to raw colors.

    Parameters
    ----------
    base_oklch:
        Base color (L0, C0, h0) in OKLCH.
    raw_colors:
        Colors after applying PaletteType (before style), as (L, C, h).
    style:
        PaletteStyle specifying how to redistribute L/C (and optionally h).
    """
    if style == PaletteStyle.SQUARE:
        return _apply_square_style(raw_colors)
    if style == PaletteStyle.TRIANGLE:
        return _apply_triangle_style(base_oklch, raw_colors)
    if style == PaletteStyle.CIRCLE:
        return _apply_circle_style(base_oklch, raw_colors)
    if style == PaletteStyle.DIAMOND:
        return _apply_diamond_style(base_oklch, raw_colors)

    raise ValueError(f"Unsupported PaletteStyle: {style}")


def _apply_square_style(raw_colors: List[OKLCH]) -> List[OKLCH]:
    if not raw_colors:
        return []
    n = len(raw_colors)
    step_L = 4.0
    mid = (n - 1) / 2.0
    styled: List[OKLCH] = []
    for i, (L, C, h) in enumerate(raw_colors):
        delta_L = (i - mid) * step_L
        L_new = _clamp_L(L + delta_L)
        styled.append((L_new, C, h))
    return styled


def _apply_triangle_style(base_oklch: OKLCH, raw_colors: List[OKLCH]) -> List[OKLCH]:
    if not raw_colors:
        return []
    n = len(raw_colors)
    mid = (n - 1) / 2.0
    L0, C0, _ = base_oklch
    k_L = 8.0
    c_factor = 0.5
    styled: List[OKLCH] = []
    for i, (L, C, h) in enumerate(raw_colors):
        if mid > 0:
            d = abs(i - mid) / mid
        else:
            d = 0.0
        # Central color brightest, edges darker.
        L_new = _clamp_L(L0 + k_L * (1.0 - d))
        # Central color most chromatic, edges less.
        C_new = C0 * (1.0 - c_factor * d)
        styled.append((L_new, C_new, h))
    return styled


def _apply_circle_style(base_oklch: OKLCH, raw_colors: List[OKLCH]) -> List[OKLCH]:
    if not raw_colors:
        return []
    n = len(raw_colors)
    L0, C0, _ = base_oklch
    radius_L = 6.0
    radius_C = 0.06
    phase_L = 0.0
    phase_C = math.pi / 2.0
    styled: List[OKLCH] = []
    for i, (_, _, h) in enumerate(raw_colors):
        t = (i / n) * 2.0 * math.pi
        L_new = _clamp_L(L0 + radius_L * math.sin(t + phase_L))
        C_new = max(0.0, C0 + radius_C * math.cos(t + phase_C))
        styled.append((L_new, C_new, h))
    return styled


def _apply_diamond_style(base_oklch: OKLCH, raw_colors: List[OKLCH]) -> List[OKLCH]:
    if not raw_colors:
        return []
    n = len(raw_colors)
    L0, C0, _ = base_oklch
    mid = (n - 1) / 2.0
    L_amp = 10.0
    C_amp = 0.08
    styled: List[OKLCH] = []
    for i, (_, _, h) in enumerate(raw_colors):
        if mid > 0:
            d = abs(i - mid) / mid
        else:
            d = 0.0
        L_new = _clamp_L(L0 + (1.0 - d) * L_amp)
        C_new = max(0.0, C0 + (1.0 - d) * C_amp)
        styled.append((L_new, C_new, h))
    return styled
