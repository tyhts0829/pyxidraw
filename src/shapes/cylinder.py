from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import shape


@shape
def cylinder(
    radius: float = 0.3, height: float = 0.6, segments: int = 32, **params: Any
) -> Geometry:
    """円柱を生成します。"""
    vertices_list = []
    angles = np.linspace(0, 2 * np.pi, segments + 1)
    top_circle = [[radius * np.cos(a), radius * np.sin(a), height / 2] for a in angles]
    bottom_circle = [[radius * np.cos(a), radius * np.sin(a), -height / 2] for a in angles]
    vertices_list.append(np.array(top_circle, dtype=np.float32))
    vertices_list.append(np.array(bottom_circle, dtype=np.float32))
    for i in range(segments):
        a = 2 * np.pi * i / segments
        x = radius * np.cos(a)
        y = radius * np.sin(a)
        vertices_list.append(np.array([[x, y, -height / 2], [x, y, height / 2]], dtype=np.float32))
    return Geometry.from_lines(vertices_list)
