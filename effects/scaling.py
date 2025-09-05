from __future__ import annotations

from typing import Tuple

from .registry import effect
from engine.core.geometry import Geometry
from common.types import Vec3


@effect()
def scale(
    g: Geometry,
    *,
    pivot: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> Geometry:
    """スケール変換を適用（純関数）。

    Args:
        pivot: スケーリングの中心（`(x, y, z)`）
        scale: 各軸の倍率（`(sx, sy, sz)`）

    Returns:
        Geometry: スケーリング後のジオメトリ
    """
    sx, sy, sz = scale
    return g.scale(sx, sy, sz, center=pivot)


# 後方互換クラスは廃止（関数APIのみ）
scale.__param_meta__ = {
    "pivot": {"type": "vec3"},
    "scale": {"type": "vec3"},
}
