from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


class Wobble(BaseEffect):
    """Add wobble/wave distortion to lines."""
    
    def apply(self, vertices_list: list[np.ndarray],
             amplitude: float = 0.05,
             frequency: float = 5.0,
             phase: float = 0.0,
             axis: str = "y",
             **params: Any) -> list[np.ndarray]:
        """Apply wobble effect.
        
        Args:
            vertices_list: Input vertex arrays
            amplitude: Wave amplitude
            frequency: Wave frequency
            phase: Phase offset
            axis: Axis to apply wobble ("x", "y", or "z")
            **params: Additional parameters (ignored)
            
        Returns:
            Wobbled vertex arrays
        """
        axis_map = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_map:
            axis = "y"
        axis_idx = axis_map[axis]
        
        new_vertices_list = []
        
        for vertices in vertices_list:
            if len(vertices) == 0:
                new_vertices_list.append(vertices)
                continue
            
            # Calculate cumulative distances along the line
            if len(vertices) > 1:
                segments = vertices[1:] - vertices[:-1]
                distances = np.sqrt(np.sum(segments ** 2, axis=1))
                cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
            else:
                cumulative_distances = np.array([0])
            
            # Apply sinusoidal displacement
            displacement = amplitude * np.sin(frequency * cumulative_distances + phase)
            
            # Create wobbled vertices
            wobbled = vertices.copy()
            wobbled[:, axis_idx] += displacement
            
            new_vertices_list.append(wobbled.astype(np.float32))
        
        return new_vertices_list