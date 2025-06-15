from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Array(BaseEffect):
    """入力のコピーを配列状に生成します。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             count_x: int = 1, count_y: int = 1, count_z: int = 1,
             spacing_x: float = 1.0, spacing_y: float = 1.0, spacing_z: float = 1.0,
             **params: Any) -> list[np.ndarray]:
        """配列エフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            count_x, count_y, count_z: 各方向のコピー数
            spacing_x, spacing_y, spacing_z: コピー間の間隔
            **params: 追加パラメータ（無視される）
            
        Returns:
            配列化された頂点配列
        """
        new_vertices_list = []
        
        for iz in range(count_z):
            for iy in range(count_y):
                for ix in range(count_x):
                    offset = np.array([
                        ix * spacing_x,
                        iy * spacing_y,
                        iz * spacing_z
                    ], dtype=np.float32)
                    
                    for vertices in vertices_list:
                        translated = vertices + offset
                        new_vertices_list.append(translated.astype(np.float32))
        
        return new_vertices_list