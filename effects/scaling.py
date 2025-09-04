from __future__ import annotations

from typing import Tuple

from .registry import effect
from engine.core.geometry import Geometry
from common.types import Vec3


@effect()
def scale(
    g: Geometry,
    *,
    center: Vec3 = (0.0, 0.0, 0.0),  # 旧
    pivot: Vec3 | None = None,       # 新（推奨）
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> Geometry:
    sx, sy, sz = scale
    c = pivot if pivot is not None else center
    return g.scale(sx, sy, sz, center=c)


# 後方互換クラスは廃止（関数APIのみ）
