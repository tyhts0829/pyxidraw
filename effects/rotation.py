from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _rotation_matrix_x(angle: float) -> np.ndarray:
    """X軸周りの回転行列を作成します。"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _rotation_matrix_y(angle: float) -> np.ndarray:
    """Y軸周りの回転行列を作成します。"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _rotation_matrix_z(angle: float) -> np.ndarray:
    """Z軸周りの回転行列を作成します。"""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _apply_rotation(vertices: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """頂点に回転行列を適用します。"""
    # Apply rotation
    rotated = vertices @ rotation_matrix.T  # Transpose for row vectors
    return rotated.astype(np.float32)


class Rotation(BaseEffect):
    """指定された軸周りに頂点を回転します。"""
    
    def apply(self, vertices_list: list[np.ndarray],
             angle_x: float = 0.0,
             angle_y: float = 0.0,
             angle_z: float = 0.0,
             **params: Any) -> list[np.ndarray]:
        """回転エフェクトを適用します。
        
        Args:
            vertices_list: 入力頂点配列
            angle_x: X軸周りの回転角（ラジアン）
            angle_y: Y軸周りの回転角（ラジアン）
            angle_z: Z軸周りの回転角（ラジアン）
            **params: 追加パラメータ（無視される）
            
        Returns:
            回転された頂点配列
        """
        # Create rotation matrices using numba-optimized functions
        Rx = _rotation_matrix_x(angle_x)
        Ry = _rotation_matrix_y(angle_y)
        Rz = _rotation_matrix_z(angle_z)
        
        # Combined rotation matrix (Z * Y * X order)
        R = Rz @ Ry @ Rx
        
        # Apply rotation to each vertex array using numba-optimized function
        new_vertices_list = []
        for vertices in vertices_list:
            rotated = _apply_rotation(vertices, R)
            new_vertices_list.append(rotated)
        
        return new_vertices_list