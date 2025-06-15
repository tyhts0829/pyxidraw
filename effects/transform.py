from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Transform(BaseEffect):
    """任意の変換行列を適用します。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             matrix: np.ndarray | None = None,
             **params: Any) -> list[np.ndarray]:
        """変換エフェクトを適用します。
        
        Args:
            vertices_list: 入力适点配列
            matrix: 4x4変換行列（または3x3）
            **params: 追加パラメータ（無視される）
            
        Returns:
            変換された頂点配列
        """
        if matrix is None:
            return vertices_list
        
        new_vertices_list = []
        
        # Handle 3x3 or 4x4 matrices
        if matrix.shape == (3, 3):
            for vertices in vertices_list:
                transformed = vertices @ matrix.T
                new_vertices_list.append(transformed.astype(np.float32))
        elif matrix.shape == (4, 4):
            for vertices in vertices_list:
                # Convert to homogeneous coordinates
                ones = np.ones((len(vertices), 1), dtype=np.float32)
                homogeneous = np.hstack([vertices, ones])
                
                # Apply transformation
                transformed = homogeneous @ matrix.T
                
                # Convert back to 3D
                transformed_3d = transformed[:, :3] / transformed[:, 3:4]
                new_vertices_list.append(transformed_3d.astype(np.float32))
        else:
            # Invalid matrix shape
            return vertices_list
        
        return new_vertices_list