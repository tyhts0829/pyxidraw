from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Extrude(BaseEffect):
    """Extrude 2D shapes into 3D."""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply extrude effect.
        
        Extrudes 2D shapes in a specified direction to create 3D structures.
        
        Args:
            vertices_list: Input vertex arrays
            direction: Extrusion direction vector (x, y, z) - default (0, 0, 1)
            distance: Extrusion distance - default 1.0
            scale: Scale factor for extruded geometry - default 1.0
            subdivisions: Number of subdivision steps - default 0
            **params: Additional parameters
            
        Returns:
            Extruded vertex arrays including original, extruded, and connecting edges
        """
        direction = params.get('direction', (0.0, 0.0, 1.0))
        distance = params.get('distance', 1.0)
        scale = params.get('scale', 1.0)
        subdivisions = params.get('subdivisions', 0)
        
        # Apply subdivisions if requested
        working_vertices_list = vertices_list.copy()
        if subdivisions > 0:
            working_vertices_list = self._subdivide_vertices(working_vertices_list, subdivisions)
        
        # Normalize direction vector
        direction_array = np.array(direction, dtype=np.float64)
        direction_norm = np.linalg.norm(direction_array)
        if direction_norm == 0:
            return vertices_list  # Can't extrude with zero direction
        
        direction_normalized = direction_array / direction_norm
        extrude_vector = direction_normalized * distance
        
        extruded_vertices_list = []
        
        # Create extruded copies
        for vertices in working_vertices_list:
            # Ensure vertices are 3D
            if vertices.shape[1] == 2:
                vertices_3d = np.hstack([vertices, np.zeros((len(vertices), 1))])
            else:
                vertices_3d = vertices.copy()
            
            # Create extruded version
            extruded_vertices = (vertices_3d + extrude_vector) * scale
            extruded_vertices_list.append(extruded_vertices)
            
            # Create connecting edges between original and extruded vertices
            for i in range(len(vertices_3d)):
                segment = np.array([vertices_3d[i], extruded_vertices[i]])
                extruded_vertices_list.append(segment)
        
        # Add original geometry
        for vertices in working_vertices_list:
            if vertices.shape[1] == 2:
                vertices_3d = np.hstack([vertices, np.zeros((len(vertices), 1))])
            else:
                vertices_3d = vertices.copy()
            extruded_vertices_list.append(vertices_3d)
        
        return extruded_vertices_list
    
    def _subdivide_vertices(self, vertices_list: list[np.ndarray], subdivisions: int) -> list[np.ndarray]:
        """Apply simple subdivision to increase vertex density.
        
        Args:
            vertices_list: Input vertex arrays
            subdivisions: Number of subdivision iterations
            
        Returns:
            Subdivided vertex arrays
        """
        result = []
        
        for vertices in vertices_list:
            current = vertices.copy()
            
            for _ in range(subdivisions):
                if len(current) < 2:
                    break
                
                # Linear subdivision - insert midpoints
                new_vertices = [current[0]]
                for i in range(len(current) - 1):
                    midpoint = (current[i] + current[i + 1]) / 2
                    new_vertices.append(midpoint)
                    new_vertices.append(current[i + 1])
                
                current = np.array(new_vertices)
            
            result.append(current)
        
        return result