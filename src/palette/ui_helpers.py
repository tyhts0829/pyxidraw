from __future__ import annotations

"""Helper utilities for integrating pyoklch into external UIs.

This module exposes label/enum pairs for palette types/styles and export formats,
and provides a public `export_palette` helper to convert Palette objects into
simple color lists (sRGB/HEX/OKLCH) that UI code can consume easily.
"""

from enum import Enum
from typing import Dict, List

from .harmony import PaletteType
from .palette import Palette
from .style import PaletteStyle


class ExportFormat(Enum):
    """Supported output formats for exported color lists."""

    SRGB_01 = "srgb_01"
    SRGB_255 = "srgb_255"
    HEX = "hex"
    OKLCH = "oklch"

    @classmethod
    def from_value(cls, value: str) -> "ExportFormat":
        for fmt in cls:
            if fmt.value == value:
                return fmt
        raise ValueError(f"Unknown export format: {value}")


# Label/Enum pairs for UI choices
PALETTE_TYPE_OPTIONS: List[tuple[str, PaletteType]] = [
    ("Analogous", PaletteType.ANALOGOUS),
    ("Complementary", PaletteType.COMPLEMENTARY),
    ("Split Complementary", PaletteType.SPLIT_COMPLEMENTARY),
    ("Triadic", PaletteType.TRIADIC),
    ("Tetradic", PaletteType.TETRADIC),
    ("Tints & Shades", PaletteType.TINTS_SHADES),
]
PALETTE_STYLE_OPTIONS: List[tuple[str, PaletteStyle]] = [
    ("Square", PaletteStyle.SQUARE),
    ("Triangle", PaletteStyle.TRIANGLE),
    ("Circle", PaletteStyle.CIRCLE),
    ("Diamond", PaletteStyle.DIAMOND),
]
EXPORT_FORMAT_OPTIONS: List[tuple[str, ExportFormat]] = [
    ("sRGB (0-1)", ExportFormat.SRGB_01),
    ("sRGB (0-255)", ExportFormat.SRGB_255),
    ("HEX", ExportFormat.HEX),
    ("OKLCH", ExportFormat.OKLCH),
]

PALETTE_TYPE_LABEL_MAP: Dict[str, PaletteType] = {
    label: value for label, value in PALETTE_TYPE_OPTIONS
}
PALETTE_STYLE_LABEL_MAP: Dict[str, PaletteStyle] = {
    label: value for label, value in PALETTE_STYLE_OPTIONS
}


def export_palette(palette: Palette, fmt: ExportFormat | str) -> List[object]:
    """Convert a Palette to a list of colors in the desired format."""
    export_fmt = fmt if isinstance(fmt, ExportFormat) else ExportFormat.from_value(fmt)
    if export_fmt == ExportFormat.SRGB_01:
        return [c.srgb for c in palette.colors]
    if export_fmt == ExportFormat.SRGB_255:
        colors_255 = []
        for r, g, b in (c.srgb for c in palette.colors):
            colors_255.append((int(round(r * 255)), int(round(g * 255)), int(round(b * 255))))
        return colors_255
    if export_fmt == ExportFormat.HEX:
        return [c.hex for c in palette.colors]
    if export_fmt == ExportFormat.OKLCH:
        return [c.oklch for c in palette.colors]
    raise ValueError(f"Unsupported export format: {fmt}")


__all__ = [
    "ExportFormat",
    "PALETTE_TYPE_OPTIONS",
    "PALETTE_STYLE_OPTIONS",
    "EXPORT_FORMAT_OPTIONS",
    "PALETTE_TYPE_LABEL_MAP",
    "PALETTE_STYLE_LABEL_MAP",
    "export_palette",
]
