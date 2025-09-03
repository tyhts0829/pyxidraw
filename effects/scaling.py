from __future__ import annotations

from typing import Tuple

from .registry import effect
from engine.core.geometry import Geometry
from common.types import Vec3


@effect()
def scaling(
    g: Geometry,
    *,
    center: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> Geometry:
    sx, sy, sz = scale
    return g.scale(sx, sy, sz, center=center)


# 後方互換クラスは廃止（関数APIのみ）
