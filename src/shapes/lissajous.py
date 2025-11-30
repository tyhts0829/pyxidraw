from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import shape


@shape
def lissajous(
    *,
    freq_x: float = 3.0,
    freq_y: float = 2.0,
    freq_z: float = 4.0,
    phase: float = 0.0,
    phase_y: float = 0.0,
    phase_z: float = 0.0,
    points: int = 1000,
    **params: Any,
) -> Geometry:
    """2D/3D リサージュ曲線を生成します。

    Parameters
    ----------
    freq_x : float, default 3.0
        X 軸方向の周波数。許容 [0.5, 10.0]。
    freq_y : float, default 2.0
        Y 軸方向の周波数。許容 [0.5, 10.0]。
    freq_z : float, default 4.0
        Z 軸方向の周波数。許容 [0.0, 10.0]。0 で 2D 曲線。
    phase : float, default 0.0
        X 軸の初期位相（度）。許容 [0, 360]。
    phase_y : float, default 0.0
        Y 軸の初期位相（度）。許容 [0, 360]。
    phase_z : float, default 0.0
        Z 軸の初期位相（度）。許容 [0, 360]。
    points : int, default 1000
        曲線をサンプリングする点数。許容 [100, 5000]。
    """
    t = np.linspace(0, 2 * np.pi, points)
    amp = 0.5
    phase_x_rad = np.deg2rad(phase)
    phase_y_rad = np.deg2rad(phase_y)
    phase_z_rad = np.deg2rad(phase_z)
    x = np.sin(freq_x * t + phase_x_rad) * amp
    y = np.sin(freq_y * t + phase_y_rad) * amp
    z = np.zeros_like(t) if freq_z == 0.0 else np.sin(freq_z * t + phase_z_rad) * amp
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)
    return Geometry.from_lines([vertices])


lissajous.__param_meta__ = {
    "freq_x": {"type": "number", "min": 0.5, "max": 10.0},
    "freq_y": {"type": "number", "min": 0.5, "max": 10.0},
    "freq_z": {"type": "number", "min": 0.0, "max": 10.0},
    "phase": {"type": "number", "min": 0.0, "max": 360.0, "step": 1.0},
    "phase_y": {"type": "number", "min": 0.0, "max": 360.0, "step": 1.0},
    "phase_z": {"type": "number", "min": 0.0, "max": 360.0, "step": 1.0},
    "points": {"type": "integer", "min": 100, "max": 5000},
}
