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
        # Pre-calculate trigonometric values
        theta_values = 2 * np.pi * np.arange(major_segments) / major_segments
        phi_values = 2 * np.pi * np.arange(minor_segments) / minor_segments
        
        cos_theta = np.cos(theta_values)
        sin_theta = np.sin(theta_values)
        cos_phi = np.cos(phi_values)
        sin_phi = np.sin(phi_values)

        vertices_list = []

        # Generate lines along major circle (meridians)
        for i in range(major_segments):
            phi_extended = 2 * np.pi * np.arange(minor_segments + 1) / minor_segments
            cos_phi_ext = np.cos(phi_extended)
            sin_phi_ext = np.sin(phi_extended)
            
            r = major_radius + minor_radius * cos_phi_ext
            x = r * cos_theta[i]
            y = r * sin_theta[i]
            z = minor_radius * sin_phi_ext
            
            vertices = np.column_stack([x, y, z]).astype(np.float32)
            vertices_list.append(vertices)

        # Generate lines along minor circles (parallels)
        for j in range(minor_segments):
            theta_extended = 2 * np.pi * np.arange(major_segments + 1) / major_segments
            cos_theta_ext = np.cos(theta_extended)
            sin_theta_ext = np.sin(theta_extended)
            
            r = major_radius + minor_radius * cos_phi[j]
            x = r * cos_theta_ext
            y = r * sin_theta_ext
            z = np.full_like(x, minor_radius * sin_phi[j])
            
            vertices = np.column_stack([x, y, z]).astype(np.float32)
            vertices_list.append(vertices)

        return vertices_list
