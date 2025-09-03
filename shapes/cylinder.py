from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape
from engine.core.geometry import Geometry
from .registry import shape




@shape
class Cylinder(BaseShape):
    """Cylinder shape generator."""
    
    def generate(self, radius: float = 0.3, height: float = 0.6, 
                segments: int = 32, **params: Any) -> Geometry:
        """Generate a cylinder.
        
        Args:
            radius: Cylinder radius
            height: Cylinder height
            segments: Number of segments around circumference
            **params: Additional parameters (ignored)
            
        Returns:
            Geometry object containing cylinder lines
        """
        vertices_list = []
        
        # Generate top and bottom circles
        angles = np.linspace(0, 2 * np.pi, segments + 1)
        
        # Top circle
        top_circle = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = height / 2
            top_circle.append([x, y, z])
        vertices_list.append(np.array(top_circle, dtype=np.float32))
        
        # Bottom circle
        bottom_circle = []
        for angle in angles:
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -height / 2
            bottom_circle.append([x, y, z])
        vertices_list.append(np.array(bottom_circle, dtype=np.float32))
        
        # Vertical lines
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            vertical_line = np.array([
                [x, y, -height / 2],
                [x, y, height / 2]
            ], dtype=np.float32)
            vertices_list.append(vertical_line)
        
        return Geometry.from_lines(vertices_list)
