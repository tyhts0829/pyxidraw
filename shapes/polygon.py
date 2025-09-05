from __future__ import annotations

import math
from typing import Any

import numpy as np

from engine.core.geometry import Geometry
from .registry import shape
from .base import BaseShape


def _polygon_cached(n_sides: int) -> np.ndarray:
    """Generate polygon vertices.
    
    Args:
        n_sides: Number of sides
        
        返り値:
            頂点配列
    """
    # Calculate vertex coords
    t = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x = np.cos(t) * 0.5
    y = np.sin(t) * 0.5
    z = np.zeros_like(x)
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)
    
    # Close the polygon by adding the first vertex at the end
    vertices = np.append(vertices, vertices[0:1], axis=0)
    
    return vertices



@shape
class Polygon(BaseShape):
    """Regular polygon shape generator."""
    
    def generate(self, n_sides: int | float = 3, **params: Any) -> Geometry:
        """直径1の円に内接する正多角形を生成します。

        引数:
            n_sides: 辺の数。float の場合は 0–1 を指数写像で 0–100 に変換。
            **params: 追加パラメータ（未使用）

        返り値:
            多角形の頂点を含む Geometry
        """
        MIN_SIDES = 3
        MAX_SIDES = 100 - MIN_SIDES
        
        if isinstance(n_sides, float):
            n_sides = self._nonlinear_map_exp(n_sides, MAX_SIDES)
            n_sides += MIN_SIDES
        elif isinstance(n_sides, int):
            if n_sides < MIN_SIDES:
                n_sides = MIN_SIDES
        
        # Use cached generation
        vertices = _polygon_cached(n_sides)
        
        return Geometry.from_lines([vertices])
    
    @staticmethod
    def _nonlinear_map_exp(value: float, N: int, a: float = 100.0) -> int:
        """Nonlinearly map a value from 0.0-1.0 to 0-N using exponential function.
        
        Args:
            value: Input value (0.0-1.0 range)
            N: Maximum value of output range (integer)
            a: Nonlinearity (>2.0 for sharp growth, close to 1.0 for nearly linear)
            
        返り値:
            マッピングされた整数値
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError("value は 0.0–1.0 の範囲で指定してください")
        if N <= 0:
            raise ValueError("N は正の整数である必要があります")
        if a <= 1.0:
            raise ValueError("a は 1.0 より大きい値を指定してください")
        
        # Exponential nonlinear mapping
        normalized_value = (math.pow(a, value) - 1) / (a - 1)
        mapped_value = normalized_value * N
        
        return int(round(mapped_value))
