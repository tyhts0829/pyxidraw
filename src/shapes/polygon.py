from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import shape


def _polygon_cached(n_sides: int) -> np.ndarray:
    """多角形の頂点配列を生成します。

    引数:
        n_sides: 辺の数。

    返り値:
        頂点配列（最初の頂点を末尾に複製して閉ループ化）。
    """
    # 頂点座標を計算
    t = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x = np.cos(t) * 0.5
    y = np.sin(t) * 0.5
    z = np.zeros_like(x)
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)

    # 最初の頂点を末尾に追加して閉ループにする
    vertices = np.append(vertices, vertices[0:1], axis=0)

    return vertices


@shape
def polygon(n_sides: int | float = 6, **params: Any) -> Geometry:
    """直径 1 の円に内接する正多角形を生成します。"""
    MIN_SIDES = 3
    MAX_SIDES = 120

    sides = int(round(float(n_sides)))
    sides = max(MIN_SIDES, min(MAX_SIDES, sides))

    vertices = _polygon_cached(sides)
    return Geometry.from_lines([vertices])


polygon.__param_meta__ = {"n_sides": {"type": "integer", "min": 3, "max": 120, "step": 1}}
