from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Culling(BaseEffect):
    """指定された範囲外の頂点を除去します。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             min_x: float = -1.0, max_x: float = 1.0,
             min_y: float = -1.0, max_y: float = 1.0,
             min_z: float = -1.0, max_z: float = 1.0,
             mode: str = "clip",
             **params: Any) -> list[np.ndarray]:
        """カリングエフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            min_x, max_x: X軸の範囲
            min_y, max_y: Y軸の範囲
            min_z, max_z: Z軸の範囲
            mode: 線を範囲でクリップする"clip"、線全体を除去する"remove"
            **params: 追加パラメータ（無視される）
            
        Returns:
            カリングされた頂点配列
        """
        new_vertices_list = []
        
        for vertices in vertices_list:
            if len(vertices) == 0:
                continue
            
            if mode == "remove":
                # Check if any vertex is outside bounds
                in_bounds = (
                    (vertices[:, 0] >= min_x) & (vertices[:, 0] <= max_x) &
                    (vertices[:, 1] >= min_y) & (vertices[:, 1] <= max_y) &
                    (vertices[:, 2] >= min_z) & (vertices[:, 2] <= max_z)
                )
                if np.all(in_bounds):
                    new_vertices_list.append(vertices)
            else:  # clip mode
                # Clip vertices to bounds
                clipped = self._clip_line_to_bounds(
                    vertices, min_x, max_x, min_y, max_y, min_z, max_z
                )
                if len(clipped) > 0:
                    new_vertices_list.append(clipped)
        
        return new_vertices_list
    
    def _clip_line_to_bounds(self, vertices: np.ndarray,
                            min_x: float, max_x: float,
                            min_y: float, max_y: float,
                            min_z: float, max_z: float) -> np.ndarray:
        """指定された範囲に線をクリップします。"""
        # For simplicity, just clamp vertices to bounds
        # A more sophisticated implementation would compute intersections
        clipped = vertices.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], min_x, max_x)
        clipped[:, 1] = np.clip(clipped[:, 1], min_y, max_y)
        clipped[:, 2] = np.clip(clipped[:, 2], min_z, max_z)
        
        # Remove duplicate consecutive vertices
        if len(clipped) > 1:
            diff = np.sum(np.abs(clipped[1:] - clipped[:-1]), axis=1)
            mask = np.concatenate([[True], diff > 1e-6])
            clipped = clipped[mask]
        
        return clipped.astype(np.float32)