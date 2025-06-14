from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np


class BaseEffect(ABC):
    """Base class for all effects with built-in caching support."""
    
    def __init__(self):
        self._cache_enabled = True
    
    @abstractmethod
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply effect to a list of vertex arrays.
        
        Args:
            vertices_list: List of vertex arrays to transform
            **params: Effect-specific parameters
            
        Returns:
            Transformed list of vertex arrays
        """
        pass
    
    def __call__(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply effect with automatic caching."""
        if self._cache_enabled:
            # Convert to hashable format
            hashable_vertices = self._vertices_to_hashable(vertices_list)
            hashable_params = self._params_to_hashable(params)
            return self._cached_apply(hashable_vertices, hashable_params)
        return self.apply(vertices_list, **params)
    
    @lru_cache(maxsize=128)
    def _cached_apply(self, hashable_vertices: tuple, hashable_params: tuple) -> list[np.ndarray]:
        """Cached version of apply method."""
        vertices_list = self._hashable_to_vertices(hashable_vertices)
        params = self._hashable_to_params(hashable_params)
        return self.apply(vertices_list, **params)
    
    def _vertices_to_hashable(self, vertices_list: list[np.ndarray]) -> tuple:
        """Convert vertices list to hashable format."""
        return tuple(tuple(map(tuple, v.tolist())) for v in vertices_list)
    
    def _hashable_to_vertices(self, hashable: tuple) -> list[np.ndarray]:
        """Convert hashable format back to vertices list."""
        return [np.array(v, dtype=np.float32) for v in hashable]
    
    def _params_to_hashable(self, params: dict[str, Any]) -> tuple:
        """Convert parameters to hashable format."""
        items = []
        for key, value in sorted(params.items()):
            if isinstance(value, (list, tuple)):
                items.append((key, ('list_tuple', tuple(value))))
            elif isinstance(value, np.ndarray):
                items.append((key, ('numpy_array', tuple(value.flatten().tolist()), value.shape)))
            elif callable(value):
                # Skip callables
                continue
            else:
                items.append((key, ('primitive', value)))
        return tuple(items)
    
    def _hashable_to_params(self, hashable: tuple) -> dict[str, Any]:
        """Convert hashable parameters back to dict."""
        params = {}
        for key, value_info in hashable:
            if isinstance(value_info, tuple) and len(value_info) >= 2:
                value_type = value_info[0]
                if value_type == 'numpy_array' and len(value_info) == 3:
                    flat_data, shape = value_info[1], value_info[2]
                    params[key] = np.array(flat_data).reshape(shape)
                elif value_type == 'list_tuple':
                    params[key] = value_info[1]
                elif value_type == 'primitive':
                    params[key] = value_info[1]
                else:
                    params[key] = value_info
            else:
                params[key] = value_info
        return params
    
    def clear_cache(self):
        """Clear the LRU cache."""
        if hasattr(self._cached_apply, 'cache_clear'):
            self._cached_apply.cache_clear()
    
    def disable_cache(self):
        """Disable caching for this effect."""
        self._cache_enabled = False
    
    def enable_cache(self):
        """Enable caching for this effect."""
        self._cache_enabled = True