from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Extrude(BaseEffect):
    """2D形状を3Dに押し出します。"""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """押し出しエフェクトを適用します。
        
        2D形状を指定された方向に押し出して3D構造を作成します。
        
        Args:
            vertices_list: 入力頂点配列
            direction: 押し出し方向ベクトル (x, y, z) - デフォルト (0, 0, 1)
            distance: 押し出し距離 - デフォルト 1.0
            scale: 押し出したジオメトリのスケール率 - デフォルト 1.0
            subdivisions: 細分化ステップ数 - デフォルト 0
            **params: 追加パラメータ
            
        Returns:
            元の形状、押し出し形状、接続エッジを含む押し出し頂点配列
        """
        direction = params.get('direction', (0.0, 0.0, 1.0))
        distance = params.get('distance', 1.0)
        scale = params.get('scale', 1.0)
        subdivisions = params.get('subdivisions', 0)
        
        # Apply subdivisions if requested
        working_vertices_list = vertices_list.copy()
        if subdivisions > 0:
            working_vertices_list = self._subdivide_vertices(working_vertices_list, subdivisions)
        
        # Normalize direction vector
        direction_array = np.array(direction, dtype=np.float64)
        direction_norm = np.linalg.norm(direction_array)
        if direction_norm == 0:
            return vertices_list  # Can't extrude with zero direction
        
        direction_normalized = direction_array / direction_norm
        extrude_vector = direction_normalized * distance
        
        extruded_vertices_list = []
        
        # Create extruded copies
        for vertices in working_vertices_list:
            # Ensure vertices are 3D
            if vertices.shape[1] == 2:
                vertices_3d = np.hstack([vertices, np.zeros((len(vertices), 1))])
            else:
                vertices_3d = vertices.copy()
            
            # Create extruded version
            extruded_vertices = (vertices_3d + extrude_vector) * scale
            extruded_vertices_list.append(extruded_vertices)
            
            # Create connecting edges between original and extruded vertices
            for i in range(len(vertices_3d)):
                segment = np.array([vertices_3d[i], extruded_vertices[i]])
                extruded_vertices_list.append(segment)
        
        # Add original geometry
        for vertices in working_vertices_list:
            if vertices.shape[1] == 2:
                vertices_3d = np.hstack([vertices, np.zeros((len(vertices), 1))])
            else:
                vertices_3d = vertices.copy()
            extruded_vertices_list.append(vertices_3d)
        
        return extruded_vertices_list
    
    def _subdivide_vertices(self, vertices_list: list[np.ndarray], subdivisions: int) -> list[np.ndarray]:
        """頂点密度を増やすための簡単な細分化を適用します。
        
        Args:
            vertices_list: 入力頂点配列
            subdivisions: 細分化反復回数
            
        Returns:
            細分化された頂点配列
        """
        result = []
        
        for vertices in vertices_list:
            current = vertices.copy()
            
            for _ in range(subdivisions):
                if len(current) < 2:
                    break
                
                # Linear subdivision - insert midpoints
                new_vertices = [current[0]]
                for i in range(len(current) - 1):
                    midpoint = (current[i] + current[i + 1]) / 2
                    new_vertices.append(midpoint)
                    new_vertices.append(current[i + 1])
                
                current = np.array(new_vertices)
            
            result.append(current)
        
        return result