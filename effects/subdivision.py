from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from engine.core.geometry import Geometry
from common.param_utils import norm_to_int
from .registry import effect


@effect()
def subdivision(g: Geometry, *, subdivisions: float = 0.5, **_params: Any) -> Geometry:
    """中間点を追加して線を細分化（純関数）。"""
    coords, offsets = g.as_arrays(copy=False)
    if subdivisions <= 0.0:
        return Geometry(coords.copy(), offsets.copy())

    MAX_DIVISIONS = 10
    divisions = norm_to_int(float(subdivisions), 0, MAX_DIVISIONS)
    if divisions <= 0:
        return Geometry(coords.copy(), offsets.copy())

    result = []
    for i in range(len(offsets) - 1):
        vertices = coords[offsets[i] : offsets[i + 1]]
        subdivided = _subdivide_core(vertices, divisions)
        result.append(subdivided)

    return Geometry.from_lines(result)


@njit(fastmath=True, cache=True)
def _subdivide_core(vertices: np.ndarray, subdivisions: int) -> np.ndarray:
    """単一頂点配列の細分化処理（Numba最適化）"""
    if len(vertices) < 2 or subdivisions <= 0:
        return vertices

    # 最小長チェック - 短すぎる線分は分割しない
    MIN_LENGTH = 0.01
    if np.linalg.norm(vertices[0] - vertices[1]) < MIN_LENGTH:
        return vertices

    # 分割回数制限 - フリーズ防止
    MAX_DIVISIONS = 10
    subdivisions = min(subdivisions, MAX_DIVISIONS)

    result = vertices.copy()
    for _ in range(subdivisions):
        n = len(result)
        new_vertices = np.zeros((2 * n - 1, result.shape[1]), dtype=result.dtype)

        # 偶数インデックスに元の頂点を配置
        new_vertices[::2] = result

        # 奇数インデックスに中点を配置
        new_vertices[1::2] = (result[:-1] + result[1:]) / 2

        result = new_vertices

        # 再度最小長チェック
        if len(result) >= 2 and np.linalg.norm(result[0] - result[1]) < MIN_LENGTH:
            break

    return result
