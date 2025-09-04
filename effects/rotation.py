from __future__ import annotations

from .registry import effect
from engine.core.geometry import Geometry
from common.types import Vec3


@effect()
def rotate(
    g: Geometry,
    *,
    pivot: Vec3 = (0.0, 0.0, 0.0),
    angles_rad: Vec3 = (0.0, 0.0, 0.0),
) -> Geometry:
    """回転（新形式のみ）。

    Args:
        pivot: 回転の中心
        angles_rad: (rx, ry, rz) ラジアン角
    """
    cx, cy, cz = pivot
    rx, ry, rz = float(angles_rad[0]), float(angles_rad[1]), float(angles_rad[2])
    return g.rotate(x=rx, y=ry, z=rz, center=(cx, cy, cz))


# 後方互換クラスは廃止（関数APIのみ）
rotate.__param_meta__ = {
    "pivot": {"type": "vec3"},
    "angles_rad": {"type": "vec3"},
}
