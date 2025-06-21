from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_scaling(vertices: np.ndarray, scale_array: np.ndarray, center: np.ndarray) -> np.ndarray:
    """頂点にスケーリングを適用します。"""
    # 中心点に対してスケーリング
    centered = vertices - center
    scaled = centered * scale_array
    result = scaled + center
    return result


class Scaling(BaseEffect):
    """指定された軸に沿って頂点をスケールします。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             center: tuple[float, float, float] = (0, 0, 0),
             scale: tuple[float, float, float] = (1, 1, 1),
             **params: Any) -> list[np.ndarray]:
        """スケールエフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            center: スケーリングの中心点 (x, y, z)
            scale: 各軸のスケール率 (x, y, z)
            **params: 追加パラメータ（無視される）
            
        Returns:
            スケールされた頂点配列
        """
        # エッジケース: 空のリスト
        if not vertices_list:
            return []
        
        # スケール値がすべて1の場合は元のデータをコピーして返す
        if scale == (1, 1, 1):
            return [vertices.copy() for vertices in vertices_list]
        
        # NumPy配列に変換
        scale_np = np.array(scale, dtype=np.float32)
        center_np = np.array(center, dtype=np.float32)
        
        # 各頂点配列にスケーリングを適用
        new_vertices_list = []
        for vertices in vertices_list:
            if len(vertices) == 0:
                new_vertices_list.append(vertices)
            else:
                scaled = _apply_scaling(vertices, scale_np, center_np)
                new_vertices_list.append(scaled)
        
        return new_vertices_list