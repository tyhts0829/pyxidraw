from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_translation(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """頂点に移動を適用します。"""
    # Apply translation
    translated = vertices + offset
    return translated.astype(np.float32)


class Translation(BaseEffect):
    """指定されたオフセットで頂点を移動します。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             offset_x: float = 0.0,
             offset_y: float = 0.0,
             offset_z: float = 0.0,
             **params: Any) -> list[np.ndarray]:
        """移動エフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            offset_x: X軸の移動オフセット
            offset_y: Y軸の移動オフセット
            offset_z: Z軸の移動オフセット
            **params: 追加パラメータ（無視される）
            
        Returns:
            移動された頂点配列
        """
        # Create offset vector
        offset = np.array([offset_x, offset_y, offset_z], dtype=np.float32)
        
        # Apply translation to each vertex array using numba-optimized function
        new_vertices_list = []
        for vertices in vertices_list:
            translated = _apply_translation(vertices, offset)
            new_vertices_list.append(translated)
        
        return new_vertices_list