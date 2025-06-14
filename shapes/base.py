from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Sequence

import numpy as np


def apply_transformations(vertices: np.ndarray, 
                         center: tuple[float, float, float] = (0, 0, 0),
                         scale: tuple[float, float, float] = (1, 1, 1),
                         rotate: tuple[float, float, float] = (0, 0, 0)) -> np.ndarray:
    """Apply center, scale, and rotation transformations to vertices.
    
    Args:
        vertices: Vertex array with shape (N, 3)
        center: Translation (cx, cy, cz)
        scale: Scale factors (sx, sy, sz)
        rotate: Rotation angles in radians (rx, ry, rz)
        
    Returns:
        Transformed vertices
    """
    transformed = vertices.copy()
    
    # Apply scaling
    transformed[:, 0] *= scale[0]
    transformed[:, 1] *= scale[1]
    transformed[:, 2] *= scale[2]
    
    # Apply rotations (in order: X, Y, Z)
    if rotate[0] != 0:  # Rotation around X-axis
        cos_x, sin_x = np.cos(rotate[0]), np.sin(rotate[0])
        y, z = transformed[:, 1], transformed[:, 2]
        transformed[:, 1] = y * cos_x - z * sin_x
        transformed[:, 2] = y * sin_x + z * cos_x
    
    if rotate[1] != 0:  # Rotation around Y-axis
        cos_y, sin_y = np.cos(rotate[1]), np.sin(rotate[1])
        x, z = transformed[:, 0], transformed[:, 2]
        transformed[:, 0] = x * cos_y + z * sin_y
        transformed[:, 2] = -x * sin_y + z * cos_y
    
    if rotate[2] != 0:  # Rotation around Z-axis
        cos_z, sin_z = np.cos(rotate[2]), np.sin(rotate[2])
        x, y = transformed[:, 0], transformed[:, 1]
        transformed[:, 0] = x * cos_z - y * sin_z
        transformed[:, 1] = x * sin_z + y * cos_z
    
    # Apply translation
    transformed[:, 0] += center[0]
    transformed[:, 1] += center[1]
    transformed[:, 2] += center[2]
    
    return transformed


class BaseShape(ABC):
    """Base class for all shape generators with built-in caching support."""
    
    def __init__(self):
        self._cache_enabled = True
    
    @abstractmethod
    def generate(self, **params: Any) -> list[np.ndarray]:
        """Generate shape vertices.
        
        Returns:
            List of vertex arrays, where each array has shape (N, 3)
        """
        pass
    
    def __call__(self, 
                 center: tuple[float, float, float] = (0, 0, 0),
                 scale: tuple[float, float, float] = (1, 1, 1),
                 rotate: tuple[float, float, float] = (0, 0, 0),
                 **params: Any) -> list[np.ndarray]:
        """Generate shape with automatic caching and transformations."""
        # Generate base shape
        if self._cache_enabled:
            # Convert params to hashable format (excluding transformations)
            hashable_params = self._make_hashable(params)
            vertices_list = self._cached_generate(hashable_params)
        else:
            vertices_list = self.generate(**params)
        
        # Apply transformations if any are non-default
        if center != (0, 0, 0) or scale != (1, 1, 1) or rotate != (0, 0, 0):
            transformed_list = []
            for vertices in vertices_list:
                transformed = apply_transformations(vertices, center, scale, rotate)
                transformed_list.append(transformed)
            return transformed_list
        
        return vertices_list
    
    @lru_cache(maxsize=None)
    def _cached_generate(self, hashable_params: tuple) -> list[np.ndarray]:
        """Cached version of generate method."""
        params = self._unmake_hashable(hashable_params)
        return self.generate(**params)
    
    def _make_hashable(self, params: dict[str, Any]) -> tuple:
        """Convert parameters to hashable format for caching."""
        items = []
        for key, value in sorted(params.items()):
            if isinstance(value, (list, tuple)):
                # Convert sequences to tuples
                items.append((key, tuple(value)))
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to tuples
                items.append((key, tuple(value.flatten().tolist())))
            elif callable(value):
                # Skip callables (they can't be hashed)
                continue
            else:
                items.append((key, value))
        return tuple(items)
    
    def _unmake_hashable(self, hashable_params: tuple) -> dict[str, Any]:
        """Convert hashable parameters back to original format."""
        return dict(hashable_params)
    
    def clear_cache(self):
        """Clear the LRU cache."""
        if hasattr(self._cached_generate, 'cache_clear'):
            self._cached_generate.cache_clear()
    
    def disable_cache(self):
        """Disable caching for this shape."""
        self._cache_enabled = False
    
    def enable_cache(self):
        """Enable caching for this shape."""
        self._cache_enabled = True