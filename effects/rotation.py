from __future__ import annotations

import math
from typing import Tuple

from .registry import effect
from engine.core.geometry import Geometry
from common.param_utils import norm_to_rad
from common.types import Vec3


@effect()
def rotate(
    g: Geometry,
    *,
    # 旧API
    center: Vec3 = (0.0, 0.0, 0.0),
    rotate: Vec3 = (0.0, 0.0, 0.0),
    # 新API（推奨）
    pivot: Vec3 | None = None,
    angles_rad: Vec3 | None = None,
    angles_deg: Vec3 | None = None,
) -> Geometry:
    """回転。

    受理パラメータ（優先順）:
    - angles_rad（ラジアン直接指定）
    - angles_deg（度 → ラジアン変換）
    - rotate（0..1 正規化 → 2π 掛け）
    中心は `pivot`（推奨）/`center` のどちらでも可。
    """
    cx, cy, cz = (pivot if pivot is not None else center)

    if angles_rad is not None:
        rx, ry, rz = float(angles_rad[0]), float(angles_rad[1]), float(angles_rad[2])
    elif angles_deg is not None:
        rx, ry, rz = [math.radians(float(v)) for v in angles_deg]
    else:
        rx, ry, rz = rotate
        rx = norm_to_rad(rx)
        ry = norm_to_rad(ry)
        rz = norm_to_rad(rz)

    return g.rotate(x=rx, y=ry, z=rz, center=(cx, cy, cz))


# 後方互換クラスは廃止（関数APIのみ）
