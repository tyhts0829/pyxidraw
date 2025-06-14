from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape


class Capsule(BaseShape):
    """Capsule (stadium of revolution) shape generator."""
    
    def generate(self, radius: float = 0.2, height: float = 0.4,
                segments: int = 32, **params: Any) -> list[np.ndarray]:
        """Generate a capsule shape.
        
        Args:
            radius: Radius of the hemispheres
            height: Height of the cylindrical section
            segments: Number of segments for curves
            **params: Additional parameters (ignored)
            
        Returns:
            List of vertex arrays for capsule lines
        """
        vertices_list = []
        
        # Generate profile lines (vertical)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            profile = []
            
            # Top hemisphere
            for j in range(segments // 4 + 1):
                phi = np.pi / 2 * j / (segments // 4)
                z = height / 2 + radius * np.sin(phi)
                r = radius * np.cos(phi)
                profile.append([r * np.cos(angle), r * np.sin(angle), z])
            
            # Cylindrical section
            profile.append([x, y, height / 2])
            profile.append([x, y, -height / 2])
            
            # Bottom hemisphere
            for j in range(segments // 4 + 1):
                phi = np.pi / 2 * j / (segments // 4)
                z = -height / 2 - radius * np.sin(phi)
                r = radius * np.cos(phi)
                profile.append([r * np.cos(angle), r * np.sin(angle), z])
            
            vertices_list.append(np.array(profile, dtype=np.float32))
        
        # Generate horizontal rings
        # Top hemisphere rings
        for j in range(1, segments // 4):
            phi = np.pi / 2 * j / (segments // 4)
            z = height / 2 + radius * np.sin(phi)
            r = radius * np.cos(phi)
            
            ring = []
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                ring.append([r * np.cos(angle), r * np.sin(angle), z])
            vertices_list.append(np.array(ring, dtype=np.float32))
        
        # Middle rings
        for z in [height / 2, -height / 2]:
            ring = []
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                ring.append([radius * np.cos(angle), radius * np.sin(angle), z])
            vertices_list.append(np.array(ring, dtype=np.float32))
        
        # Bottom hemisphere rings
        for j in range(1, segments // 4):
            phi = np.pi / 2 * j / (segments // 4)
            z = -height / 2 - radius * np.sin(phi)
            r = radius * np.cos(phi)
            
            ring = []
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                ring.append([r * np.cos(angle), r * np.sin(angle), z])
            vertices_list.append(np.array(ring, dtype=np.float32))
        
        return vertices_list