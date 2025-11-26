from __future__ import annotations

"""Container type for generated color palettes.

This module defines the :class:`Palette` dataclass, which groups the
base color, palette type/style, and the list of generated colors.
"""

from dataclasses import dataclass
from typing import List

from .color_types import Color
from .harmony import PaletteType
from .style import PaletteStyle


@dataclass
class Palette:
    """Generated color palette.

    Attributes
    ----------
    base_color:
        Base color from which the palette was generated.
    palette_type:
        Geometric palette type (e.g. Complementary, Triadic).
    palette_style:
        Style controlling lightness/chroma distribution.
    colors:
        List of colors in a visually natural order. The base color is
        included in this list, typically in the center or first position
        depending on PaletteType/Style.
    """

    base_color: Color
    palette_type: PaletteType
    palette_style: PaletteStyle
    colors: List[Color]
