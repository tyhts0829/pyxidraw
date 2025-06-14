from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Desolve(BaseEffect):
    """Dissolve/fragment lines into individual segments."""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply desolve effect.
        
        Breaks continuous polylines into individual line segments.
        
        Args:
            vertices_list: Input vertex arrays
            factor: Dissolution factor (0.0-1.0) - controls randomness (not used in basic implementation)
            seed: Random seed for reproducibility (not used in basic implementation)
            **params: Additional parameters
            
        Returns:
            List of individual line segments (each with 2 vertices)
        """
        factor = params.get('factor', 0.5)
        seed = params.get('seed', None)
        
        new_vertices_list = []
        
        for vertices in vertices_list:
            dissolved_segments = self._desolve_core(vertices, factor, seed)
            new_vertices_list.extend(dissolved_segments)
        
        return new_vertices_list
    
    def _desolve_core(self, vertices: np.ndarray, factor: float, seed: int | None) -> list[np.ndarray]:
        """Break a polyline into individual line segments.
        
        Args:
            vertices: (N, 3) array of vertices forming a polyline
            factor: Dissolution factor (currently unused, for future enhancement)
            seed: Random seed (currently unused, for future enhancement)
            
        Returns:
            List of (2, 3) arrays, each representing one line segment
        """
        # If vertices has 2 or fewer points, return as is
        if len(vertices) <= 2:
            return [vertices]
        
        # Break polyline into individual segments
        segments = []
        for i in range(len(vertices) - 1):
            segment = vertices[i:i+2].copy()
            segments.append(segment)
        
        return segments