from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect
from .registry import effect


@njit(fastmath=True, cache=True)
def _apply_combined_transform(
    vertices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rotate: np.ndarray,
) -> np.ndarray:
    """頂点に組み合わせ変換を適用します。"""
    # 回転行列を一度だけ計算
    sx, sy, sz = np.sin(rotate)
    cx, cy, cz = np.cos(rotate)
    
    # Z * Y * X の結合行列を直接計算
    R = np.empty((3, 3), dtype=np.float32)
    R[0, 0] = cy * cz
    R[0, 1] = sx * sy * cz - cx * sz
    R[0, 2] = cx * sy * cz + sx * sz
    R[1, 0] = cy * sz
    R[1, 1] = sx * sy * sz + cx * cz
    R[1, 2] = cx * sy * sz - sx * cz
    R[2, 0] = -sy
    R[2, 1] = sx * cy
    R[2, 2] = cx * cy
    
    # 全頂点に変換を一度に適用（スケール -> 回転 -> 移動）
    scaled = vertices * scale
    rotated = scaled @ R.T
    transformed = rotated + center
    
    return transformed


@effect
class Transform(BaseEffect):
    """任意の変換行列を適用します。"""

    TAU = math.tau  # 全回転角度（2 * pi）

    def apply(
        self,
        coords: np.ndarray,
        offsets: np.ndarray,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """変換エフェクトを適用します。

        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            center: 変換の中心点 (x, y, z)
            scale: スケール係数 (x, y, z)
            rotate: 回転角度（ラジアン） (x, y, z) 入力は0.0-1.0の範囲を想定。内部でmath.tauを掛けてラジアンに変換される。
            **params: 追加パラメータ（無視される）

        Returns:
            (transformed_coords, offsets): 変換された座標配列とオフセット配列
        """

        # エッジケース: 空の座標配列
        if len(coords) == 0:
            return coords.copy(), offsets.copy()

        # エッジケース: 変換がない場合
        if (center == (0, 0, 0) and scale == (1, 1, 1) and 
            abs(rotate[0]) < 1e-10 and abs(rotate[1]) < 1e-10 and abs(rotate[2]) < 1e-10):
            return coords.copy(), offsets.copy()

        # NumPy配列に変換
        center_np = np.array(center, dtype=np.float32)
        scale_np = np.array(scale, dtype=np.float32)
        rotate_radians = np.array([
            rotate[0] * self.TAU,
            rotate[1] * self.TAU,
            rotate[2] * self.TAU,
        ], dtype=np.float32)

        # 全頂点に組み合わせ変換を一度に適用
        transformed_coords = _apply_combined_transform(coords, center_np, scale_np, rotate_radians)
        
        return transformed_coords, offsets.copy()
