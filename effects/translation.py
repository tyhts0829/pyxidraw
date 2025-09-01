from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect
from .registry import effect


@njit(fastmath=True, cache=True)
def _apply_translation(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """頂点に移動を適用します。"""
    # Apply translation
    translated = vertices + offset
    return translated.astype(np.float32)


@effect
class Translation(BaseEffect):
    """指定されたオフセットで頂点を移動します。"""
    
    def apply(self, coords: np.ndarray, offsets: np.ndarray,
             offset_x: float = 0.0,
             offset_y: float = 0.0,
             offset_z: float = 0.0,
             **params: Any) -> tuple[np.ndarray, np.ndarray]:
        """移動エフェクトを適用します。
        
        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            offset_x: X軸の移動オフセット
            offset_y: Y軸の移動オフセット
            offset_z: Z軸の移動オフセット
            **params: 追加パラメータ（無視される）
            
        Returns:
            (translated_coords, offsets): 移動された座標配列とオフセット配列
        """
        # オフセットがゼロの場合は元のデータをそのまま返す
        if offset_x == 0.0 and offset_y == 0.0 and offset_z == 0.0:
            return coords.copy(), offsets.copy()
        
        # エッジケース: 空の座標配列
        if len(coords) == 0:
            return coords.copy(), offsets.copy()
        
        # Create offset vector
        offset = np.array([offset_x, offset_y, offset_z], dtype=np.float32)
        
        # 全頂点に移動を一度に適用
        translated_coords = _apply_translation(coords, offset)
        
        return translated_coords, offsets.copy()
