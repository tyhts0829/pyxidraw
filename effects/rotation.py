from __future__ import annotations

import math
from typing import Tuple

from .registry import effect
from engine.core.geometry import Geometry
from common.param_utils import norm_to_rad
from common.types import Vec3


@effect()
def rotation(
    g: Geometry,
    *,
    center: Vec3 = (0.0, 0.0, 0.0),
    rotate: Vec3 = (0.0, 0.0, 0.0),
) -> Geometry:
    """回転（0..1 正規化入力を想定して 2π を掛ける）。"""
    rx, ry, rz = rotate
    rx = norm_to_rad(rx)
    ry = norm_to_rad(ry)
    rz = norm_to_rad(rz)
    return g.rotate(x=rx, y=ry, z=rz, center=center)


# 後方互換クラスは廃止（関数APIのみ）
