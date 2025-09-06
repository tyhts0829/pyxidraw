from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape
from engine.core.geometry import Geometry
from .registry import shape




@shape
class Cylinder(BaseShape):
    """Cylinder shape generator."""
    
    def generate(self, radius: float = 0.3, height: float = 0.6, 
                segments: int = 32, **params: Any) -> Geometry:
        """円柱を生成します。

        引数:
            radius: 半径
            height: 高さ
            segments: 周方向の分割数
            **params: 追加パラメータ（未使用）

        返り値:
            円柱の線群を含む Geometry
        """
        vertices_list = []
        
        # 上面と下面の円を生成
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        
        # 上面の円
        top_circle = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = height / 2
            top_circle.append([x, y, z])
        vertices_list.append(np.array(top_circle, dtype=np.float32))
        
        # 下面の円
        bottom_circle = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -height / 2
            bottom_circle.append([x, y, z])
        vertices_list.append(np.array(bottom_circle, dtype=np.float32))
        
        # 側面の垂直ライン
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            vertical_line = np.array([
                [x, y, -height / 2],
                [x, y, height / 2]
            ], dtype=np.float32)
            vertices_list.append(vertical_line)
        
        return Geometry.from_lines(vertices_list)
