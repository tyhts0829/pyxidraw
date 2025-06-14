from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_scaling(vertices: np.ndarray, scale_matrix: np.ndarray) -> np.ndarray:
    """Apply scaling matrix to vertices."""
    # Apply scaling
    scaled = vertices @ scale_matrix.T
    return scaled.astype(np.float32)


class Scaling(BaseEffect):
    """Scale vertices along specified axes."""
    
    def apply(self, vertices_list: list[np.ndarray],
             scale_x: float = 1.0,
             scale_y: float = 1.0,
             scale_z: float = 1.0,
             uniform_scale: float | None = None,
             **params: Any) -> list[np.ndarray]:
        """Apply scaling effect.
        
        Args:
            vertices_list: Input vertex arrays
            scale_x: Scale factor for X axis
            scale_y: Scale factor for Y axis
            scale_z: Scale factor for Z axis
            uniform_scale: If provided, overrides individual scale factors
            **params: Additional parameters (ignored)
            
        Returns:
            Scaled vertex arrays
        """
        # Use uniform scale if provided
        if uniform_scale is not None:
            scale_x = scale_y = scale_z = uniform_scale
        
        # Create scaling matrix
        scale_matrix = np.diag([scale_x, scale_y, scale_z]).astype(np.float32)
        
        # Apply scaling to each vertex array using numba-optimized function
        new_vertices_list = []
        for vertices in vertices_list:
            scaled = _apply_scaling(vertices, scale_matrix)
            new_vertices_list.append(scaled)
        
        return new_vertices_list