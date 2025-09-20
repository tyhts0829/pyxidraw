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
    """2D/3D リサージュ曲線を生成します。"""
    t = np.linspace(0, 2 * np.pi, points)
    amp = 0.5
    x = np.sin(freq_x * t + phase) * amp
    y = np.sin(freq_y * t + phase_y) * amp
    z = np.zeros_like(t) if freq_z == 0.0 else np.sin(freq_z * t + phase_z) * amp
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)
    return Geometry.from_lines([vertices])


lissajous.__param_meta__ = {
    "freq_x": {"type": "number", "min": 0.5, "max": 10.0},
    "freq_y": {"type": "number", "min": 0.5, "max": 10.0},
    "freq_z": {"type": "number", "min": 0.0, "max": 10.0},
    "phase": {"type": "number", "min": 0.0, "max": 6.28318},
    "phase_y": {"type": "number", "min": 0.0, "max": 6.28318},
    "phase_z": {"type": "number", "min": 0.0, "max": 6.28318},
    "points": {"type": "integer", "min": 100, "max": 5000},
}
