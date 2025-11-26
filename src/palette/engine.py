from __future__ import annotations

"""Color conversion engine for OKLCH and sRGB.

This module defines the :class:`ColorEngine` protocol and a default
implementation that converts between sRGB (D65) and OKLCH via OKLab.
"""

import math
from typing import Protocol, Tuple


OKLCH = Tuple[float, float, float]
SRGB = Tuple[float, float, float]


class ColorEngine(Protocol):
    """Protocol abstracting color space conversions."""

    def srgb_to_oklch(self, r: float, g: float, b: float) -> OKLCH: ...

    def oklch_to_srgb(self, L: float, C: float, h: float) -> SRGB: ...

    def normalize_hue(self, h: float) -> float: ...


class DefaultColorEngine:
    """Default implementation based on OKLab/OKLCH and sRGB (D65)."""

    def normalize_hue(self, h: float) -> float:
        """Normalize hue angle into [0, 360)."""
        return (h % 360.0 + 360.0) % 360.0

    def srgb_to_oklch(self, r: float, g: float, b: float) -> OKLCH:
        """Convert sRGB in [0, 1] to OKLCH with L in [0, 100]."""
        rl, gl, bl = _srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)

        # Linear RGB to LMS (OKLab)
        l = 0.4122214708 * rl + 0.5363325363 * gl + 0.0514459929 * bl
        m = 0.2119034982 * rl + 0.6806995451 * gl + 0.1073969566 * bl
        s = 0.0883024619 * rl + 0.2817188376 * gl + 0.6299787005 * bl

        l_ = math.copysign(abs(l) ** (1 / 3), l)
        m_ = math.copysign(abs(m) ** (1 / 3), m)
        s_ = math.copysign(abs(s) ** (1 / 3), s)

        L_ok = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a_ok = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b_ok = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        C = math.sqrt(a_ok * a_ok + b_ok * b_ok)
        if C < 1e-12:
            h_deg = 0.0
        else:
            h_rad = math.atan2(b_ok, a_ok)
            h_deg = math.degrees(h_rad)
        h_norm = self.normalize_hue(h_deg)

        # L in spec is 0–100
        return (max(0.0, min(100.0, L_ok * 100.0)), C, h_norm)

    def oklch_to_srgb(self, L: float, C: float, h: float) -> SRGB:
        """Convert OKLCH (L in [0, 100]) to sRGB in [0, 1]."""
        L_ok = max(0.0, min(100.0, L)) / 100.0
        C = max(0.0, C)
        h_rad = math.radians(self.normalize_hue(h))

        a = C * math.cos(h_rad)
        b = C * math.sin(h_rad)

        # OKLab to LMS
        l_ = L_ok + 0.3963377774 * a + 0.2158037573 * b
        m_ = L_ok - 0.1055613458 * a - 0.0638541728 * b
        s_ = L_ok - 0.0894841775 * a - 1.2914855480 * b

        l = l_**3
        m = m_**3
        s = s_**3

        rl = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        gl = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        bl = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

        r = _linear_to_srgb(rl)
        g = _linear_to_srgb(gl)
        b_ = _linear_to_srgb(bl)
        return (r, g, b_)


def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c: float) -> float:
    if c <= 0.0:
        return 0.0
    if c >= 1.0:
        return 1.0
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055
