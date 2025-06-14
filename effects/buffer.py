from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Buffer(BaseEffect):
    """Buffer/offset paths by creating parallel lines."""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply buffer effect.
        
        Creates parallel lines at specified distance from input paths.
        
        Args:
            vertices_list: Input vertex arrays
            distance: Buffer distance (positive = outward, negative = inward) - default 0.1
            join_style: Join style for corners ("round", "miter", "bevel") - default "round"
            **params: Additional parameters
            
        Returns:
            Buffered vertex arrays including original and offset paths
        """
        distance = params.get('distance', 0.1)
        join_style = params.get('join_style', 'round')
        
        if distance == 0:
            return vertices_list.copy()
        
        buffered_results = []
        
        for vertices in vertices_list:
            if len(vertices) < 2:
                buffered_results.append(vertices)
                continue
            
            # Add original path
            buffered_results.append(vertices)
            
            # Create offset paths
            offset_paths = self._create_offset_paths(vertices, distance, join_style)
            buffered_results.extend(offset_paths)
        
        return buffered_results
    
    def _create_offset_paths(self, vertices: np.ndarray, distance: float, join_style: str) -> list[np.ndarray]:
        """Create offset paths on both sides of the input path."""
        if len(vertices) < 2:
            return []
        
        # Create left and right offset paths
        left_path = self._offset_path(vertices, distance)
        right_path = self._offset_path(vertices, -distance)
        
        paths = []
        if left_path is not None:
            paths.append(left_path)
        if right_path is not None:
            paths.append(right_path)
        
        return paths
    
    def _offset_path(self, vertices: np.ndarray, distance: float) -> np.ndarray | None:
        """Create a single offset path at specified distance."""
        if len(vertices) < 2:
            return None
        
        offset_points = []
        
        for i in range(len(vertices) - 1):
            p1 = vertices[i]
            p2 = vertices[i + 1]
            
            # Calculate perpendicular vector
            direction = p2 - p1
            if np.linalg.norm(direction) == 0:
                continue
            
            # Get perpendicular vector (2D projection for now)
            if len(direction) >= 2:
                perp = np.array([-direction[1], direction[0], 0.0])
                if len(direction) == 3:
                    perp[2] = 0.0
            else:
                continue
            
            # Normalize and scale by distance
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 0:
                perp = perp / perp_norm * distance
                
                # Create offset points
                offset_p1 = p1 + perp
                offset_p2 = p2 + perp
                
                if i == 0:
                    offset_points.append(offset_p1)
                offset_points.append(offset_p2)
        
        return np.array(offset_points) if len(offset_points) >= 2 else None