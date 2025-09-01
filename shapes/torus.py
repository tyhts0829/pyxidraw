from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseShape
from engine.core.geometry_data import GeometryData
from .registry import shape


@njit(fastmath=True, cache=True)
def _generate_meridian_line(
    major_radius: float, minor_radius: float, minor_segments: int, cos_theta: float, sin_theta: float
) -> np.ndarray:
    """Generate a single meridian line."""
    phi_values = 2 * np.pi * np.arange(minor_segments + 1) / minor_segments
    cos_phi = np.cos(phi_values)
    sin_phi = np.sin(phi_values)

    r = major_radius + minor_radius * cos_phi
    x = r * cos_theta
    y = r * sin_theta
    z = minor_radius * sin_phi

    vertices = np.empty((len(phi_values), 3), dtype=np.float32)
    vertices[:, 0] = x
    vertices[:, 1] = y
    vertices[:, 2] = z

    return vertices


@njit(fastmath=True, cache=True)
def _generate_parallel_line(
    major_radius: float, minor_radius: float, major_segments: int, cos_phi: float, sin_phi: float
) -> np.ndarray:
    """Generate a single parallel line."""
    theta_values = 2 * np.pi * np.arange(major_segments + 1) / major_segments
    cos_theta = np.cos(theta_values)
    sin_theta = np.sin(theta_values)

    r = major_radius + minor_radius * cos_phi
    x = r * cos_theta
    y = r * sin_theta
    z_value = minor_radius * sin_phi

    vertices = np.empty((len(theta_values), 3), dtype=np.float32)
    vertices[:, 0] = x
    vertices[:, 1] = y
    vertices[:, 2] = z_value

    return vertices




@shape
class Torus(BaseShape):
    """Torus shape generator."""

    def generate(
        self,
        major_radius: float = 0.25,
        minor_radius: float = 0.125,
        major_segments: int = 32,
        minor_segments: int = 16,
        **params: Any,
    ) -> GeometryData:
        """Generate a torus.

        Args:
            major_radius: Major radius (from center to tube center)
            minor_radius: Minor radius (tube radius)
            major_segments: Number of segments around major circle
            minor_segments: Number of segments around minor circle
            **params: Additional parameters (ignored)

        Returns:
            GeometryData object containing torus lines
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
            vertices = _generate_meridian_line(major_radius, minor_radius, minor_segments, cos_theta[i], sin_theta[i])
            vertices_list.append(vertices)

        # Generate lines along minor circles (parallels)
        for j in range(minor_segments):
            vertices = _generate_parallel_line(major_radius, minor_radius, major_segments, cos_phi[j], sin_phi[j])
            vertices_list.append(vertices)

        return GeometryData.from_lines(vertices_list)
