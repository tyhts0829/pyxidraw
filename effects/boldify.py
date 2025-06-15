from __future__ import annotations

from typing import Any, List

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _calculate_line_normals_3d(vertices: np.ndarray) -> np.ndarray:
    """3D線分のXY平面での法線ベクトルを計算します。"""
    if vertices.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    
    # Direction vectors in XY plane
    directions = vertices[1:] - vertices[:-1]
    
    # Normals in XY plane (Z component is 0)
    normals = np.zeros_like(directions, dtype=np.float32)
    normals[:, 0] = -directions[:, 1]  # x = -dy
    normals[:, 1] = directions[:, 0]   # y = dx
    normals[:, 2] = 0                  # z = 0
    
    # Normalize
    lengths = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)
    lengths = np.where(lengths == 0, 1, lengths)
    normals[:, 0] /= lengths
    normals[:, 1] /= lengths
    
    return normals


@njit(fastmath=True, cache=True)
def _boldify_normal_based(
    vertices_list: List[np.ndarray],
    thickness: float = 0.01,
) -> List[np.ndarray]:
    """法線ベースの効率的な太線実装です。"""
    if thickness <= 0:
        return vertices_list
    
    new_vertices_list = []
    half_thickness = thickness / 2
    
    for vertices in vertices_list:
        if vertices.shape[0] < 2:
            new_vertices_list.append(vertices)
            continue
        
        # Add original line
        new_vertices_list.append(vertices)
        
        # Calculate normals for 3D vertices
        normals = _calculate_line_normals_3d(vertices)
        
        if normals.shape[0] == 0:
            continue
        
        # Calculate per-vertex normals
        vertex_normals = np.zeros_like(vertices, dtype=np.float32)
        
        # First vertex
        vertex_normals[0] = normals[0]
        
        # Middle vertices (average of adjacent segment normals)
        for i in range(1, vertices.shape[0] - 1):
            vertex_normals[i] = (normals[i - 1] + normals[i]) / 2
            # Re-normalize
            length = np.sqrt(vertex_normals[i, 0] ** 2 + vertex_normals[i, 1] ** 2)
            if length > 0:
                vertex_normals[i] /= length
        
        # Last vertex
        vertex_normals[-1] = normals[-1]
        
        # Generate left and right parallel lines
        left_line = vertices + vertex_normals * half_thickness
        right_line = vertices - vertex_normals * half_thickness
        
        new_vertices_list.append(left_line.astype(vertices.dtype))
        new_vertices_list.append(right_line.astype(vertices.dtype))
    
    return new_vertices_list


@njit(fastmath=True, cache=True)
def _boldify_adaptive(
    vertices_list: List[np.ndarray],
    density: float = 0.5,
    base_thickness: float = 0.01,
) -> List[np.ndarray]:
    """アダプティブ密度制御の太線実装です。"""
    if density <= 0:
        return vertices_list
    
    new_vertices_list = []
    
    for vertices in vertices_list:
        if vertices.shape[0] < 2:
            new_vertices_list.append(vertices)
            continue
        
        # Calculate line length
        line_length = np.sum(np.sqrt(np.sum((vertices[1:] - vertices[:-1]) ** 2, axis=1)))
        
        # Adaptive count (min 1, max 10)
        adaptive_count = max(1, min(10, int(density * np.sqrt(line_length) * 10)))
        
        # Add original line
        new_vertices_list.append(vertices)
        
        # Calculate normals
        normals = _calculate_line_normals_3d(vertices)
        
        if normals.shape[0] == 0:
            continue
        
        # Calculate vertex normals
        vertex_normals = np.zeros_like(vertices, dtype=np.float32)
        vertex_normals[0] = normals[0]
        for i in range(1, vertices.shape[0] - 1):
            vertex_normals[i] = (normals[i - 1] + normals[i]) / 2
            length = np.sqrt(vertex_normals[i, 0] ** 2 + vertex_normals[i, 1] ** 2)
            if length > 0:
                vertex_normals[i] /= length
        vertex_normals[-1] = normals[-1]
        
        # Generate adaptive number of parallel lines
        for i in range(1, adaptive_count + 1):
            thickness = base_thickness * i / adaptive_count
            left_line = vertices + vertex_normals * thickness
            right_line = vertices - vertex_normals * thickness
            new_vertices_list.append(left_line.astype(vertices.dtype))
            new_vertices_list.append(right_line.astype(vertices.dtype))
    
    return new_vertices_list


class Boldify(BaseEffect):
    """平行線を追加して線を太く見せます。"""
    
    def apply(self, vertices_list: list[np.ndarray], 
             offset: float = 1.0,
             num_offset: tuple[float, float, float] = (0.5, 0.5, 0.5),
             method: str = "normal",
             **params: Any) -> list[np.ndarray]:
        """太線化エフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            offset: 太さ係数（0.0-1.0、内部では0.1を乗算）
            num_offset: アダプティブ方式の密度制御
            method: 実装方式（"normal"または"adaptive"）
            **params: 追加パラメータ（無視される）
            
        Returns:
            太線化された頂点配列
        """
        if offset <= 0:
            return vertices_list
        
        thickness = offset * 0.1
        
        if method == "adaptive":
            density = sum(num_offset) / 3
            return _boldify_adaptive(vertices_list, density, thickness)
        else:
            return _boldify_normal_based(vertices_list, thickness)