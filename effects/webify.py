from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Webify(BaseEffect):
    """Create web-like connections between vertices."""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply webify effect.
        
        Creates web-like connections between nearby vertices.
        
        Args:
            vertices_list: Input vertex arrays
            connection_probability: Probability of connecting nearby vertices - default 0.5
            max_distance: Maximum distance for connections - default 1.0
            **params: Additional parameters
            
        Returns:
            Original vertex arrays plus web connections
        """
        connection_probability = params.get('connection_probability', 0.5)
        max_distance = params.get('max_distance', 1.0)
        
        # Collect all vertices from all arrays
        all_vertices = []
        for vertices in vertices_list:
            all_vertices.extend(vertices)
        
        if len(all_vertices) < 2:
            return vertices_list.copy()
        
        all_vertices = np.array(all_vertices)
        
        # Create web connections
        web_connections = self._create_web_connections(
            all_vertices, connection_probability, max_distance
        )
        
        # Combine original geometry with web connections
        result = vertices_list.copy()
        result.extend(web_connections)
        
        return result
    
    def _create_web_connections(self, vertices: np.ndarray, probability: float, max_distance: float) -> list[np.ndarray]:
        """Create web connections between vertices."""
        connections = []
        n_vertices = len(vertices)
        
        # Use a simple random seed for reproducibility
        np.random.seed(42)
        
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                # Calculate distance between vertices
                distance = np.linalg.norm(vertices[j] - vertices[i])
                
                # Check if vertices are close enough
                if distance <= max_distance:
                    # Random chance to create connection
                    if np.random.random() < probability:
                        # Adjust probability based on distance (closer = more likely)
                        distance_factor = 1.0 - (distance / max_distance)
                        adjusted_probability = probability * distance_factor
                        
                        if np.random.random() < adjusted_probability:
                            connection = np.array([vertices[i], vertices[j]])
                            connections.append(connection)
        
        return connections