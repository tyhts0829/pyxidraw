from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_translation(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply translation to vertices."""
    # Apply translation
    translated = vertices + offset
    return translated.astype(np.float32)


class Translation(BaseEffect):
    """Translate vertices by specified offset."""
    
    def apply(self, vertices_list: list[np.ndarray],
             offset_x: float = 0.0,
             offset_y: float = 0.0,
             offset_z: float = 0.0,
             **params: Any) -> list[np.ndarray]:
        """Apply translation effect.
        
        Args:
            vertices_list: Input vertex arrays
            offset_x: Translation offset for X axis
            offset_y: Translation offset for Y axis
            offset_z: Translation offset for Z axis
            **params: Additional parameters (ignored)
            
        Returns:
            Translated vertex arrays
        """
        # Create offset vector
        offset = np.array([offset_x, offset_y, offset_z], dtype=np.float32)
        
        # Apply translation to each vertex array using numba-optimized function
        new_vertices_list = []
        for vertices in vertices_list:
            translated = _apply_translation(vertices, offset)
            new_vertices_list.append(translated)
        
        return new_vertices_list