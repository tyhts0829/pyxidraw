from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_noise(vertices: np.ndarray, amplitude: float, seed: int) -> np.ndarray:
    """Apply noise to vertices."""
    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate random displacements
    noise = np.random.normal(0, amplitude, vertices.shape).astype(np.float32)
    
    # Apply noise
    noisy_vertices = vertices + noise
    return noisy_vertices.astype(np.float32)


class Noise(BaseEffect):
    """Add random noise to vertices."""
    
    def apply(self, vertices_list: list[np.ndarray],
             amplitude: float = 0.01,
             seed: int | None = None,
             **params: Any) -> list[np.ndarray]:
        """Apply noise effect.
        
        Args:
            vertices_list: Input vertex arrays
            amplitude: Maximum displacement amplitude
            seed: Random seed for reproducibility
            **params: Additional parameters (ignored)
            
        Returns:
            Noisy vertex arrays
        """
        # Use a default seed if none provided
        effective_seed = seed if seed is not None else 42
        
        # Apply noise to each vertex array using numba-optimized function
        new_vertices_list = []
        for i, vertices in enumerate(vertices_list):
            # Use different seed for each array to avoid identical noise
            array_seed = effective_seed + i
            noisy_vertices = _apply_noise(vertices, amplitude, array_seed)
            new_vertices_list.append(noisy_vertices)
        
        return new_vertices_list