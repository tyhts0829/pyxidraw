from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Transform(BaseEffect):
    """Apply arbitrary transformation matrix."""
    
    def apply(self, vertices_list: list[np.ndarray],
             matrix: np.ndarray | None = None,
             **params: Any) -> list[np.ndarray]:
        """Apply transform effect.
        
        Args:
            vertices_list: Input vertex arrays
            matrix: 4x4 transformation matrix (or 3x3)
            **params: Additional parameters (ignored)
            
        Returns:
            Transformed vertex arrays
        """
        if matrix is None:
            return vertices_list
        
        new_vertices_list = []
        
        # Handle 3x3 or 4x4 matrices
        if matrix.shape == (3, 3):
            for vertices in vertices_list:
                transformed = vertices @ matrix.T
                new_vertices_list.append(transformed.astype(np.float32))
        elif matrix.shape == (4, 4):
            for vertices in vertices_list:
                # Convert to homogeneous coordinates
                ones = np.ones((len(vertices), 1), dtype=np.float32)
                homogeneous = np.hstack([vertices, ones])
                
                # Apply transformation
                transformed = homogeneous @ matrix.T
                
                # Convert back to 3D
                transformed_3d = transformed[:, :3] / transformed[:, 3:4]
                new_vertices_list.append(transformed_3d.astype(np.float32))
        else:
            # Invalid matrix shape
            return vertices_list
        
        return new_vertices_list