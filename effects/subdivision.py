from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from engine.core.geometry import Geometry

from .base import BaseEffect
from .registry import effect


@effect
class Subdivision(BaseEffect):
    """中間点を追加して線を細分化します。"""

    MAX_DIVISIONS = 10  # 最大分割回数

    def apply(self, coords: np.ndarray, offsets: np.ndarray, subdivisions: float = 0.5, **_params: Any) -> tuple[np.ndarray, np.ndarray]:
        """細分化エフェクトを適用します。

        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            subdivisions: 細分化レベル (0.0 = 変化なし, 1.0 = 最大分割) - デフォルト 0.5
            **_params: 追加パラメータ（無視される）

        Returns:
            (new_coords, new_offsets): 細分化された座標配列とオフセット配列
        """
        if subdivisions <= 0.0:
            return coords.copy(), offsets.copy()

        # Convert 0.0-1.0 to 0-MAX_DIVISIONS
        divisions = int(subdivisions * self.MAX_DIVISIONS)
        if divisions <= 0:
            return coords.copy(), offsets.copy()

        # 既存の線を取得し、細分化を適用
        result = []
        
        for i in range(len(offsets) - 1):
            vertices = coords[offsets[i]:offsets[i+1]]
            subdivided = _subdivide_core(vertices, divisions)
            result.append(subdivided)

        # 結果をGeometryに変換してから配列を取得
        result_geometry = Geometry.from_lines(result)
        return result_geometry.coords, result_geometry.offsets


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
