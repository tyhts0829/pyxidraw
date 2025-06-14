from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _rotation_matrix_x(angle: float) -> np.ndarray:
    """Create rotation matrix around X axis."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _rotation_matrix_y(angle: float) -> np.ndarray:
    """Create rotation matrix around Y axis."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _rotation_matrix_z(angle: float) -> np.ndarray:
    """Create rotation matrix around Z axis."""
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


@njit(fastmath=True, cache=True)
def _apply_rotation(vertices: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply rotation matrix to vertices."""
    # Apply rotation
    rotated = vertices @ rotation_matrix.T  # Transpose for row vectors
    return rotated.astype(np.float32)


class Rotation(BaseEffect):
    """Rotate vertices around specified axes."""
    
    def apply(self, vertices_list: list[np.ndarray],
             angle_x: float = 0.0,
             angle_y: float = 0.0,
             angle_z: float = 0.0,
             **params: Any) -> list[np.ndarray]:
        """Apply rotation effect.
        
        Args:
            vertices_list: Input vertex arrays
            angle_x: Rotation angle around X axis in radians
            angle_y: Rotation angle around Y axis in radians
            angle_z: Rotation angle around Z axis in radians
            **params: Additional parameters (ignored)
            
        Returns:
            Rotated vertex arrays
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