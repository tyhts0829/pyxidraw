"""
rotate エフェクト（回転）

- ピボット点を中心に、XYZ 各軸の回転角（ラジアン）を指定して回転します。
- 実装は `Geometry.rotate` に委譲し、新規インスタンスを返す純関数です。

パラメータ:
- pivot: 回転中心（Vec3）。
- angles_rad: (rx, ry, rz) [rad]（右手系）。
"""

from __future__ import annotations

from common.types import Vec3
from engine.core.geometry import Geometry

from .registry import effect


@effect()
def rotate(
    g: Geometry,
    *,
    pivot: Vec3 = (0.0, 0.0, 0.0),
    angles_rad: Vec3 = (0.0, 0.0, 0.0),
) -> Geometry:
    """回転（新形式のみ）。

    引数:
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
