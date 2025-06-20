from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape


class Torus(BaseShape):
    """Torus shape generator."""

    def generate(
        self,
        major_radius: float = 0.25,
        minor_radius: float = 0.125,
        major_segments: int = 32,
        minor_segments: int = 16,
        **params: Any,
    ) -> list[np.ndarray]:
        """Generate a torus.

        Args:
            major_radius: Major radius (from center to tube center)
            minor_radius: Minor radius (tube radius)
            major_segments: Number of segments around major circle
            minor_segments: Number of segments around minor circle
            **params: Additional parameters (ignored)

        Returns:
            List of vertex arrays for torus lines
        """
        vertices_list = []

        # Generate lines along major circle
        for i in range(major_segments):
            theta = 2 * np.pi * i / major_segments
            ring = []

            for j in range(minor_segments + 1):
                phi = 2 * np.pi * j / minor_segments

                x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
                y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
                z = minor_radius * np.sin(phi)

                ring.append([x, y, z])

            vertices_list.append(np.array(ring, dtype=np.float32))

        # Generate lines along minor circles
        for j in range(minor_segments):
            phi = 2 * np.pi * j / minor_segments
            ring = []

            for i in range(major_segments + 1):
                theta = 2 * np.pi * i / major_segments

                x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
                y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
                z = minor_radius * np.sin(phi)

                ring.append([x, y, z])

            vertices_list.append(np.array(ring, dtype=np.float32))

        return vertices_list
