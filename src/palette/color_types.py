from __future__ import annotations

"""Core color types used by the pyoklch library.

This module defines simple, explicit data structures for representing
colors in OKLCH and sRGB, and a small wrapper for user-supplied color
inputs in various formats.
"""

from dataclasses import dataclass
from typing import Tuple

from .engine import ColorEngine, DefaultColorEngine
from .gamut import to_srgb_gamut_safe


OKLCH = Tuple[float, float, float]
SRGB = Tuple[float, float, float]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class Color:
    """Concrete color representation in OKLCH and sRGB.

    Attributes
    ----------
    oklch:
        Tuple of (L, C, h). L is in [0, 100], C is non-negative,
        and h is in [0, 360).
    srgb:
        Tuple of (r, g, b) in [0, 1] sRGB space.
    hex:
        Hex representation "#rrggbb". Stored primarily as a cache.
    """

    oklch: OKLCH
    srgb: SRGB
    hex: str

    def to_hex(self) -> str:
        """Return hex representation of the color."""
        return self.hex

    def to_srgb(self) -> SRGB:
        """Return sRGB representation as (r, g, b) in [0, 1]."""
        return self.srgb

    def to_oklch(self) -> OKLCH:
        """Return OKLCH representation as (L, C, h)."""
        return self.oklch

    @classmethod
    def from_oklch(
        cls,
        L: float,
        C: float,
        h: float,
        engine: ColorEngine | None = None,
    ) -> "Color":
        """Create a gamut-mapped Color from OKLCH.

        Parameters
        ----------
        L, C, h:
            OKLCH coordinates. L is expected in [0, 100], h in degrees.
        engine:
            ColorEngine used for conversion. If None, DefaultColorEngine is used.
        """
        if engine is None:
            engine = DefaultColorEngine()
        r, g, b, (L_adj, C_adj, h_adj) = to_srgb_gamut_safe(engine, L, C, h)
        hex_value = _srgb_to_hex((r, g, b))
        return cls(oklch=(L_adj, C_adj, h_adj), srgb=(r, g, b), hex=hex_value)


class ColorInput:
    """User-facing color input wrapper supporting multiple formats.

    Use one of the constructor-like class methods to create instances.
    At runtime, a ColorEngine is used to convert into OKLCH as
    the internal representation.
    """

    def __init__(self, *, _mode: str, _value) -> None:
        self._mode = _mode
        self._value = _value

    @classmethod
    def from_oklch(cls, L: float, C: float, h: float) -> "ColorInput":
        """Create ColorInput from OKLCH values.

        Parameters
        ----------
        L, C, h:
            OKLCH coordinates. L must be in [0, 100], h will be normalized
            to [0, 360).
        """
        if not (0.0 <= L <= 100.0):
            raise ValueError("L must be in [0, 100].")
        if C < 0.0:
            raise ValueError("C must be non-negative.")
        # h is normalized later together with engine utilities
        return cls(_mode="oklch", _value=(L, C, h))

    @classmethod
    def from_srgb(cls, r: float, g: float, b: float) -> "ColorInput":
        """Create ColorInput from sRGB values in [0, 1]."""
        for name, v in (("r", r), ("g", g), ("b", b)):
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1].")
        return cls(_mode="srgb", _value=(r, g, b))

    @classmethod
    def from_hex(cls, hex_str: str) -> "ColorInput":
        """Create ColorInput from a hex string (#rrggbb or rrggbb)."""
        s = hex_str.strip()
        if s.startswith("#"):
            s = s[1:]
        if len(s) != 6:
            raise ValueError("HEX string must be 6 hex digits.")
        try:
            int(s, 16)
        except ValueError as exc:
            raise ValueError("HEX string must contain only hex digits.") from exc
        return cls(_mode="hex", _value=s.lower())

    def to_oklch(self, engine: ColorEngine | None = None) -> OKLCH:
        """Convert the input into OKLCH using the given ColorEngine."""
        if engine is None:
            engine = DefaultColorEngine()

        if self._mode == "oklch":
            L, C, h = self._value
            h_norm = engine.normalize_hue(h)
            return (L, C, h_norm)

        if self._mode == "srgb":
            r, g, b = self._value
            L, C, h = engine.srgb_to_oklch(r, g, b)
            return (L, C, h)

        if self._mode == "hex":
            s = self._value
            r = int(s[0:2], 16) / 255.0
            g = int(s[2:4], 16) / 255.0
            b = int(s[4:6], 16) / 255.0
            L, C, h = engine.srgb_to_oklch(r, g, b)
            return (L, C, h)

        raise RuntimeError(f"Unknown ColorInput mode: {self._mode}")


def _srgb_to_hex(rgb: SRGB) -> str:
    r, g, b = rgb
    r_i = int(round(_clamp01(r) * 255))
    g_i = int(round(_clamp01(g) * 255))
    b_i = int(round(_clamp01(b) * 255))
    return f"#{r_i:02x}{g_i:02x}{b_i:02x}"
