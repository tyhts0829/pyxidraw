from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_noise(vertices: np.ndarray, amplitude: float, seed: int) -> np.ndarray:
    """頂点にノイズを適用します。"""
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate random displacements
    noise = np.random.normal(0, amplitude, vertices.shape).astype(np.float32)
    
    # Apply noise
    noisy_vertices = vertices + noise
    return noisy_vertices.astype(np.float32)


class Noise(BaseEffect):
    """頂点にランダムノイズを追加します。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             amplitude: float = 0.01,
             seed: int | None = None,
             **params: Any) -> list[np.ndarray]:
        """ノイズエフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            amplitude: 最大変位振幅
            seed: 再現可能性のためのランダムシード
            **params: 追加パラメータ（無視される）
            
        Returns:
            ノイズが適用された頂点配列
        """
        # Use a default seed if none provided
        effective_seed = seed if seed is not None else 42
        
        # Apply noise to each vertex array using numba-optimized function
        new_vertices_list = []
        for i, vertices in enumerate(vertices_list):
            # Use different seed for each array to avoid identical noise
            array_seed = effective_seed + i
            noisy_vertices = _apply_noise(vertices, amplitude, array_seed)
            new_vertices_list.append(noisy_vertices)
        
        return new_vertices_list