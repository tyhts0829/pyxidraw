from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import shape


@shape
def line(length: float = 1.0, angle_deg: float = 0.0, **params: Any) -> Geometry:
    """長さ 1 の正規化線分を生成します。

    Parameters
    ----------
    length : float, default 1.0
        正規化長さ。許容 [0, 1]。0 で空形状。
    angle_deg : float, default 0.0
        回転角度（度）。0 で X 軸正方向。
    """
    length_f = float(length)
    if length_f <= 0.0:
        return Geometry.from_lines([])

    max_length = 1.0
    if length_f > max_length:
        length_f = max_length

    half = 0.5 * length_f
    angle = float(angle_deg) % 360.0
    theta = np.deg2rad(angle)
    dx = half * np.cos(theta)
    dy = half * np.sin(theta)

    vertices = np.array(
        [
            [-dx, -dy, 0.0],
            [dx, dy, 0.0],
        ],
        dtype=np.float32,
    )
    return Geometry.from_lines([vertices])


line.__param_meta__ = {
    "length": {"type": "number", "min": 0.0, "max": 1.0, "step": 1e-4},
    "angle_deg": {"type": "number", "min": 0.0, "max": 360.0, "step": 1.0},
}
