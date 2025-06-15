from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_scaling(vertices: np.ndarray, scale_matrix: np.ndarray) -> np.ndarray:
    """頂点にスケール行列を適用します。"""
    # Apply scaling
    scaled = vertices @ scale_matrix.T
    return scaled.astype(np.float32)


class Scaling(BaseEffect):
    """指定された軸に沿って頂点をスケールします。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             scale_x: float = 1.0,
             scale_y: float = 1.0,
             scale_z: float = 1.0,
             uniform_scale: float | None = None,
             **params: Any) -> list[np.ndarray]:
        """スケールエフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            scale_x: X軸のスケール率
            scale_y: Y軸のスケール率
            scale_z: Z軸のスケール率
            uniform_scale: 指定された場合、個別のスケール率をオーバーライド
            **params: 追加パラメータ（無視される）
            
        Returns:
            スケールされた頂点配列
        """
        # Use uniform scale if provided
        if uniform_scale is not None:
            scale_x = scale_y = scale_z = uniform_scale
        
        # Create scaling matrix
        scale_matrix = np.diag([scale_x, scale_y, scale_z]).astype(np.float32)
        
        # Apply scaling to each vertex array using numba-optimized function
        new_vertices_list = []
        for vertices in vertices_list:
            scaled = _apply_scaling(vertices, scale_matrix)
            new_vertices_list.append(scaled)
        
        return new_vertices_list