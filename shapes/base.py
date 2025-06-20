from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np


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

    def __call__(
        self,
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params: Any,
    ) -> list[np.ndarray]:
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
            # Lazy import to avoid circular dependency
            from api import effects
            return effects.transform(vertices_list, center, scale, rotate)
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
        if hasattr(self._cached_generate, "cache_clear"):
            self._cached_generate.cache_clear()

    def disable_cache(self):
        """Disable caching for this shape."""
        self._cache_enabled = False

    def enable_cache(self):
        """Enable caching for this shape."""
        self._cache_enabled = True
