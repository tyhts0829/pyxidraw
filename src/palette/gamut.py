from __future__ import annotations

"""sRGB gamut handling utilities for OKLCH colors.

This module provides a small helper to convert OKLCH colors into the sRGB
gamut by shrinking chroma until the color falls inside [0, 1]^3.
"""

from typing import Tuple

from .engine import ColorEngine


OKLCH = Tuple[float, float, float]
SRGB = Tuple[float, float, float]


def to_srgb_gamut_safe(
    engine: ColorEngine,
    L: float,
    C: float,
    h: float,
    max_iter: int = 16,
    reduction_factor: float = 0.9,
) -> Tuple[float, float, float, OKLCH]:
    """Convert OKLCH to in-gamut sRGB, reducing C until within gamut.

    Returns (r, g, b, (L_adj, C_adj, h_adj)).
    """
    L = max(0.0, min(100.0, L))
    C_curr = max(0.0, C)
    h_norm = engine.normalize_hue(h)

    r = g = b = 0.0
    for _ in range(max_iter):
        r, g, b = engine.oklch_to_srgb(L, C_curr, h_norm)
        if _in_gamut(r, g, b):
            return r, g, b, (L, C_curr, h_norm)
        C_curr *= reduction_factor

    # Fallback: clip to [0, 1]
    r = _clip01(r)
    g = _clip01(g)
    b = _clip01(b)
    return r, g, b, (L, C_curr, h_norm)


def _in_gamut(r: float, g: float, b: float) -> bool:
    return 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
