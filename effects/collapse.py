from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Collapse(BaseEffect):
    """Collapse vertices toward a center point."""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply collapse effect.
        
        Moves vertices toward a center point by specified factor.
        
        Args:
            vertices_list: Input vertex arrays
            center: Center point to collapse toward (x, y, z) - default (0, 0, 0)
            factor: Collapse factor (0.0 = no change, 1.0 = complete collapse) - default 0.0
            **params: Additional parameters
            
        Returns:
            Collapsed vertex arrays
        """
        center = params.get('center', (0.0, 0.0, 0.0))
        factor = params.get('factor', 0.0)
        
        # Clamp factor to valid range
        factor = max(0.0, min(1.0, factor))
        
        if factor == 0.0:
            return vertices_list.copy()
        
        center_point = np.array(center)
        collapsed_results = []
        
        for vertices in vertices_list:
            if len(vertices) == 0:
                collapsed_results.append(vertices)
                continue
            
            # Ensure center point has same dimensions as vertices
            if len(center_point) != vertices.shape[1]:
                if vertices.shape[1] == 3 and len(center_point) == 2:
                    center_point = np.append(center_point, 0.0)
                elif vertices.shape[1] == 2 and len(center_point) == 3:
                    center_point = center_point[:2]
                else:
                    center_point = np.zeros(vertices.shape[1])
            
            # Calculate collapsed positions
            collapsed_vertices = vertices.copy()
            for i in range(len(vertices)):
                # Linear interpolation toward center
                collapsed_vertices[i] = vertices[i] + factor * (center_point - vertices[i])
            
            collapsed_results.append(collapsed_vertices)
        
        return collapsed_results