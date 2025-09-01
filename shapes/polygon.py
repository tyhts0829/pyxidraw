from __future__ import annotations

import math
from typing import Any

import numpy as np

from engine.core.geometry_data import GeometryData
from .registry import shape
from .base import BaseShape


def _polygon_cached(n_sides: int) -> np.ndarray:
    """Generate polygon vertices.
    
    Args:
        n_sides: Number of sides
        
    Returns:
        Array of vertices
    """
    # Calculate vertex coords
    t = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x = np.cos(t) * 0.5
    y = np.sin(t) * 0.5
    z = np.zeros_like(x)
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)
    
    # Close the polygon by adding the first vertex at the end
    vertices = np.append(vertices, vertices[0:1], axis=0)
    
    return vertices



@shape
class Polygon(BaseShape):
    """Regular polygon shape generator."""
    
    def generate(self, n_sides: int | float = 3, **params: Any) -> "GeometryData":
        """Generate a regular polygon inscribed in a circle of diameter 1.
        
        Args:
            n_sides: Number of sides. If float, exponentially mapped from 0-100.
            **params: Additional parameters (ignored)
            
        Returns:
            GeometryData object containing the polygon vertices
        """
        MIN_SIDES = 3
        MAX_SIDES = 100 - MIN_SIDES
        
        if isinstance(n_sides, float):
            n_sides = self._nonlinear_map_exp(n_sides, MAX_SIDES)
            n_sides += MIN_SIDES
        elif isinstance(n_sides, int):
            if n_sides < MIN_SIDES:
                n_sides = MIN_SIDES
        
        # Use cached generation
        vertices = _polygon_cached(n_sides)
        
        return GeometryData.from_lines([vertices])
    
    @staticmethod
    def _nonlinear_map_exp(value: float, N: int, a: float = 100.0) -> int:
        """Nonlinearly map a value from 0.0-1.0 to 0-N using exponential function.
        
        Args:
            value: Input value (0.0-1.0 range)
            N: Maximum value of output range (integer)
            a: Nonlinearity (>2.0 for sharp growth, close to 1.0 for nearly linear)
            
        Returns:
            Mapped integer value
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError("value must be in range 0.0-1.0")
        if N <= 0:
            raise ValueError("N must be a positive integer")
        if a <= 1.0:
            raise ValueError("a must be greater than 1.0")
        
        # Exponential nonlinear mapping
        normalized_value = (math.pow(a, value) - 1) / (a - 1)
        mapped_value = normalized_value * N
        
        return int(round(mapped_value))
