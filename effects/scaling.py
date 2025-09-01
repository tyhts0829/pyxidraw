from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect
from .registry import effect


@njit(fastmath=True, cache=True)
def _apply_scaling(vertices: np.ndarray, scale_array: np.ndarray, center: np.ndarray) -> np.ndarray:
    """頂点にスケーリングを適用します。"""
    # 中心点に対してスケーリング
    centered = vertices - center
    scaled = centered * scale_array
    result = scaled + center
    return result


@effect
class Scaling(BaseEffect):
    """指定された軸に沿って頂点をスケールします。"""
    
    def apply(self, coords: np.ndarray, offsets: np.ndarray,
             center: tuple[float, float, float] = (0, 0, 0),
             scale: tuple[float, float, float] = (1, 1, 1),
             **params: Any) -> tuple[np.ndarray, np.ndarray]:
        """スケールエフェクトを適用します。
        
        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            center: スケーリングの中心点 (x, y, z)
            scale: 各軸のスケール率 (x, y, z)
            **params: 追加パラメータ（無視される）
            
        Returns:
            (scaled_coords, offsets): スケールされた座標配列とオフセット配列
        """
        # スケール値がすべて1の場合は元のデータをそのまま返す
        if scale == (1, 1, 1):
            return coords.copy(), offsets.copy()
        
        # エッジケース: 空の座標配列
        if len(coords) == 0:
            return coords.copy(), offsets.copy()
        
        # NumPy配列に変換
        scale_np = np.array(scale, dtype=np.float32)
        center_np = np.array(center, dtype=np.float32)
        
        # 全頂点にスケーリングを一度に適用
        scaled_coords = _apply_scaling(coords, scale_np, center_np)
        
        return scaled_coords, offsets.copy()
