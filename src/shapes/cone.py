from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import shape


@shape
def cone(radius: float = 0.3, height: float = 0.6, segments: int = 32, **params: Any) -> Geometry:
    """円錐を生成します。"""
    vertices_list: list[np.ndarray] = []
    angles = np.linspace(0, 2 * np.pi, segments + 1)
    base_circle = [[radius * np.cos(a), radius * np.sin(a), -height / 2] for a in angles]
    vertices_list.append(np.array(base_circle, dtype=np.float32))
    apex = np.array([0, 0, height / 2], dtype=np.float32)
    for i in range(segments):
        a = 2 * np.pi * i / segments
        x = radius * np.cos(a)
        y = radius * np.sin(a)
        line = np.array([apex, [x, y, -height / 2]], dtype=np.float32)
        vertices_list.append(line)
    return Geometry.from_lines(vertices_list)


cone.__param_meta__ = {
    "radius": {"type": "number", "min": 0.05, "max": 1.0},
    "height": {"type": "number", "min": 0.1, "max": 2.0},
    "segments": {"type": "integer", "min": 6, "max": 128},
}
