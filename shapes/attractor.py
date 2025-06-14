from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape


class Attractor(BaseShape):
    """Strange attractor shape generator."""
    
    def generate(self, attractor_type: str = "lorenz", points: int = 10000, 
                dt: float = 0.01, **params: Any) -> list[np.ndarray]:
        """Generate a strange attractor.
        
        Args:
            attractor_type: Type of attractor ("lorenz", "rossler", "chua")
            points: Number of points to generate
            dt: Time step for integration
            **params: Additional parameters (ignored)
            
        Returns:
            List containing a single array of vertices
        """
        if attractor_type == "lorenz":
            vertices = self._generate_lorenz(points, dt)
        elif attractor_type == "rossler":
            vertices = self._generate_rossler(points, dt)
        elif attractor_type == "chua":
            vertices = self._generate_chua(points, dt)
        else:
            # Default to Lorenz
            vertices = self._generate_lorenz(points, dt)
        
        # Normalize to fit in unit cube
        vertices = self._normalize_vertices(vertices)
        
        return [vertices]
    
    def _generate_lorenz(self, points: int, dt: float) -> np.ndarray:
        """Generate Lorenz attractor."""
        # Lorenz parameters
        sigma = 10.0
        rho = 28.0
        beta = 8.0 / 3.0
        
        # Initial conditions
        x, y, z = 1.0, 1.0, 1.0
        vertices = []
        
        for _ in range(points):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            vertices.append([x, y, z])
        
        return np.array(vertices, dtype=np.float32)
    
    def _generate_rossler(self, points: int, dt: float) -> np.ndarray:
        """Generate Rössler attractor."""
        # Rössler parameters
        a = 0.2
        b = 0.2
        c = 5.7
        
        # Initial conditions
        x, y, z = 1.0, 1.0, 1.0
        vertices = []
        
        for _ in range(points):
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            vertices.append([x, y, z])
        
        return np.array(vertices, dtype=np.float32)
    
    def _generate_chua(self, points: int, dt: float) -> np.ndarray:
        """Generate Chua's circuit attractor."""
        # Chua parameters
        alpha = 15.6
        beta = 28.0
        m0 = -1.143
        m1 = -0.714
        
        # Initial conditions
        x, y, z = 0.7, 0.0, 0.0
        vertices = []
        
        for _ in range(points):
            # Chua's diode function
            h = m1 * x + 0.5 * (m0 - m1) * (abs(x + 1) - abs(x - 1))
            
            dx = alpha * (y - x - h)
            dy = x - y + z
            dz = -beta * y
            
            x += dx * dt
            y += dy * dt
            z += dz * dt
            
            vertices.append([x, y, z])
        
        return np.array(vertices, dtype=np.float32)
    
    def _normalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Normalize vertices to fit in unit cube centered at origin."""
        # Find bounds
        min_vals = vertices.min(axis=0)
        max_vals = vertices.max(axis=0)
        
        # Center and scale
        center = (min_vals + max_vals) / 2
        scale = (max_vals - min_vals).max()
        
        if scale > 0:
            normalized = (vertices - center) / scale
        else:
            normalized = vertices - center
        
        return normalized