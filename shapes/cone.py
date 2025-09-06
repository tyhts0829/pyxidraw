from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape
from engine.core.geometry import Geometry
from .registry import shape




@shape
class Cone(BaseShape):
    """Cone shape generator."""
    
    def generate(self, radius: float = 0.3, height: float = 0.6,
                segments: int = 32, **params: Any) -> Geometry:
        """円錐を生成します。

        引数:
            radius: 底面の半径
            height: 高さ
            segments: 周方向の分割数
            **params: 追加パラメータ（未使用）

        返り値:
            円錐の線群を含む Geometry
        """
        vertices_list = []
        
        # 底面の円
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        base_circle = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -height / 2
            base_circle.append([x, y, z])
        vertices_list.append(np.array(base_circle, dtype=np.float32))
        
        # 頂点から底面へのライン
        apex = np.array([0, 0, height / 2], dtype=np.float32)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -height / 2
            
            line = np.array([
                apex,
                [x, y, z]
            ], dtype=np.float32)
            vertices_list.append(line)
        
        return Geometry.from_lines(vertices_list)
