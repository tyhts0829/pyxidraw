from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Subdivision(BaseEffect):
    """Subdivide lines by adding intermediate points."""
    
    def apply(self, vertices_list: list[np.ndarray],
             subdivisions: int = 1,
             smoothing: float = 0.0,
             **params: Any) -> list[np.ndarray]:
        """Apply subdivision effect.
        
        Args:
            vertices_list: Input vertex arrays
            subdivisions: Number of subdivision iterations
            smoothing: Smoothing factor (0.0 = linear, 1.0 = maximum smoothing)
            **params: Additional parameters (ignored)
            
        Returns:
            Subdivided vertex arrays
        """
        new_vertices_list = []
        
        for vertices in vertices_list:
            if len(vertices) < 2:
                new_vertices_list.append(vertices)
                continue
            
            # Apply subdivision iterations
            result = vertices
            for _ in range(subdivisions):
                result = self._subdivide_once(result, smoothing)
            
            new_vertices_list.append(result)
        
        return new_vertices_list
    
    def _subdivide_once(self, vertices: np.ndarray, smoothing: float) -> np.ndarray:
        """Perform one subdivision iteration."""
        n = len(vertices)
        if n < 2:
            return vertices
        
        # Calculate new vertex count
        new_n = 2 * n - 1
        new_vertices = np.zeros((new_n, 3), dtype=np.float32)
        
        # Copy original vertices
        new_vertices[::2] = vertices
        
        # Add midpoints
        for i in range(n - 1):
            if smoothing > 0 and i > 0 and i < n - 2:
                # Catmull-Rom interpolation for smoothing
                p0 = vertices[i - 1]
                p1 = vertices[i]
                p2 = vertices[i + 1]
                p3 = vertices[i + 2]
                
                # Interpolate at t=0.5
                t = 0.5
                t2 = t * t
                t3 = t2 * t
                
                midpoint = 0.5 * (
                    (2 * p1) +
                    (-p0 + p2) * t +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
                    (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
                
                # Blend between linear and smooth interpolation
                linear_midpoint = 0.5 * (vertices[i] + vertices[i + 1])
                new_vertices[2 * i + 1] = (1 - smoothing) * linear_midpoint + smoothing * midpoint
            else:
                # Simple linear interpolation at boundaries
                new_vertices[2 * i + 1] = 0.5 * (vertices[i] + vertices[i + 1])
        
        return new_vertices