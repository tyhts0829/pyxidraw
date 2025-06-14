from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Sweep(BaseEffect):
    """Remove duplicate line segments from vertex lists."""
    
    def apply(self, vertices_list: list[np.ndarray], **params: Any) -> list[np.ndarray]:
        """Apply sweep effect.
        
        Removes duplicate line segments across all vertex arrays.
        This is useful for cleaning up overlapping geometry.
        
        Args:
            vertices_list: Input vertex arrays
            path: Path to sweep along (unused in current implementation)
            profile: Profile to sweep (unused in current implementation)
            **params: Additional parameters
            
        Returns:
            Vertex arrays with duplicate line segments removed
        """
        path = params.get('path', None)
        profile = params.get('profile', None)
        
        # For now, implement the duplicate removal functionality
        # Future versions could implement actual profile sweeping
        return self._remove_duplicate_segments(vertices_list)
    
    def _remove_duplicate_segments(self, vertices_list: list[np.ndarray]) -> list[np.ndarray]:
        """Remove duplicate line segments from vertex lists.
        
        Args:
            vertices_list: List of vertex arrays
            
        Returns:
            List of vertex arrays with duplicate segments removed
        """
        # Track seen segments to avoid duplicates
        # Use normalized tuple representation (smaller vertex first)
        seen_segments = set()
        new_vertices_list = []
        
        for vertices in vertices_list:
            # Skip if too few vertices to form line segments
            if len(vertices) < 2:
                new_vertices_list.append(vertices.copy())
                continue
            
            # Start with first vertex
            filtered_points = [vertices[0]]
            
            # Check each line segment
            for i in range(len(vertices) - 1):
                p = vertices[i]
                q = vertices[i + 1]
                
                # Create direction-independent segment key
                p_tuple = tuple(p.round(decimals=10))  # Round to avoid floating point issues
                q_tuple = tuple(q.round(decimals=10))
                segment_key = (p_tuple, q_tuple) if p_tuple <= q_tuple else (q_tuple, p_tuple)
                
                # If segment hasn't been seen before, add it
                if segment_key not in seen_segments:
                    seen_segments.add(segment_key)
                    filtered_points.append(q)
                # If segment is duplicate, skip adding the endpoint
                # This effectively removes the duplicate segment
            
            # Only add if we have at least 2 points (1 segment)
            if len(filtered_points) >= 2:
                new_vertices_list.append(np.array(filtered_points))
        
        return new_vertices_list