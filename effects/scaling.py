from __future__ import annotations

from typing import Tuple

from .registry import effect
from engine.core.geometry import Geometry


@effect()
def scaling(
    g: Geometry,
    *,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Geometry:
    sx, sy, sz = scale
    return g.scale(sx, sy, sz, center=center)


# 後方互換クラスは廃止（関数APIのみ）
