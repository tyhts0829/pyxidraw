from __future__ import annotations

"""Geometric hue patterns and palette skeleton generation.

This module defines :class:`PaletteType` and the logic to compute
relative hue offsets and raw OKLCH colors before style application.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from .engine import ColorEngine


OKLCH = Tuple[float, float, float]


class PaletteType(Enum):
    """Geometric hue relationships used to build palette skeletons."""

    ANALOGOUS = auto()
    COMPLEMENTARY = auto()
    SPLIT_COMPLEMENTARY = auto()
    TRIADIC = auto()
    TETRADIC = auto()
    TINTS_SHADES = auto()


def compute_hue_offsets(palette_type: PaletteType, n_colors: int) -> List[float]:
    """Compute base hue offsets (in degrees) for the given palette type.

    The offsets are relative to the base color's hue.
    """
    if n_colors <= 0:
        raise ValueError("n_colors must be positive.")

    if palette_type == PaletteType.ANALOGOUS:
        if n_colors == 1:
            return [0.0]
        max_delta = 30.0
        step = 2 * max_delta / (n_colors - 1)
        return [-max_delta + step * i for i in range(n_colors)]

    if palette_type == PaletteType.COMPLEMENTARY:
        if n_colors == 1:
            return [0.0]
        offsets = [0.0, 180.0]
        # For n > 2, add small analogous offsets around both base and complement.
        if n_colors > 2:
            extras: List[float] = []
            delta_step = 10.0
            k = 1
            while len(offsets) + len(extras) < n_colors:
                for sign in (+1, -1):
                    if len(offsets) + len(extras) >= n_colors:
                        break
                    extras.append(sign * delta_step * k)
                    if len(offsets) + len(extras) >= n_colors:
                        break
                    extras.append(180.0 + sign * delta_step * k)
                k += 1
            offsets.extend(extras[: max(0, n_colors - len(offsets))])
        return offsets[:n_colors]

    if palette_type == PaletteType.SPLIT_COMPLEMENTARY:
        if n_colors == 1:
            return [0.0]
        delta = 30.0
        base_offsets = [0.0, 180.0 - delta, 180.0 + delta]
        if n_colors <= 3:
            return base_offsets[:n_colors]
        # For additional colors, add small variations around the split complements.
        offsets = base_offsets[:]
        extras: List[float] = []
        delta_step = 10.0
        k = 1
        while len(offsets) + len(extras) < n_colors:
            for base in (180.0 - delta, 180.0 + delta):
                for sign in (+1, -1):
                    if len(offsets) + len(extras) >= n_colors:
                        break
                    extras.append(base + sign * delta_step * k)
            k += 1
        offsets.extend(extras[: max(0, n_colors - len(offsets))])
        return offsets[:n_colors]

    if palette_type == PaletteType.TRIADIC:
        if n_colors != 3:
            raise ValueError("Triadic palette is defined for n_colors=3.")
        return [0.0, 120.0, 240.0]

    if palette_type == PaletteType.TETRADIC:
        if n_colors != 4:
            raise ValueError("Tetradic palette is defined for n_colors=4.")
        return [0.0, 90.0, 180.0, 270.0]

    if palette_type == PaletteType.TINTS_SHADES:
        # Hue offsets are not used; h remains fixed.
        return [0.0 for _ in range(n_colors)]

    raise ValueError(f"Unsupported PaletteType: {palette_type}")


@dataclass
class TintsShadesParams:
    """Parameters controlling Tints & Shades generation."""

    L_min: float
    L_max: float
    C_min: float
    C_max: float


def compute_tints_shades_params(base_L: float, base_C: float) -> TintsShadesParams:
    """Compute default L/C bounds for Tints & Shades."""
    L_min = max(0.0, base_L - 40.0)
    L_max = min(100.0, base_L + 40.0)
    C_max = base_C
    C_min = 0.2 * base_C
    return TintsShadesParams(L_min=L_min, L_max=L_max, C_min=C_min, C_max=C_max)


def generate_raw_colors(
    engine: ColorEngine,
    palette_type: PaletteType,
    base_oklch: OKLCH,
    n_colors: int,
) -> List[OKLCH]:
    """Generate raw (L, C, h) colors before style application."""
    L0, C0, h0 = base_oklch

    if palette_type == PaletteType.TINTS_SHADES:
        params = compute_tints_shades_params(L0, C0)
        if n_colors == 1:
            t_values = [0.0]
        else:
            t_values = [i / (n_colors - 1) for i in range(n_colors)]
        raw: List[OKLCH] = []
        for t in t_values:
            L = params.L_min + (params.L_max - params.L_min) * t
            # Triangular distribution for chroma: max at center, min at ends.
            if n_colors == 1:
                w = 1.0
            else:
                center = 0.5
                w = 1.0 - abs(t - center) / center
            C = params.C_min + (params.C_max - params.C_min) * w
            raw.append((L, C, engine.normalize_hue(h0)))
        return raw

    offsets = compute_hue_offsets(palette_type, n_colors)
    raw_colors: List[OKLCH] = []
    for offset in offsets:
        h_i = engine.normalize_hue(h0 + offset)
        raw_colors.append((L0, C0, h_i))
    return raw_colors
