from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape


class AsemicGlyph(BaseShape):
    """Asemic glyph (abstract writing) shape generator."""
    
    def generate(self, complexity: int = 5, seed: int | None = None, **params: Any) -> list[np.ndarray]:
        """Generate abstract glyph-like shapes.
        
        Args:
            complexity: Number of strokes (1-10)
            seed: Random seed for reproducibility
            **params: Additional parameters (ignored)
            
        Returns:
            List of vertex arrays for glyph strokes
        """
        if seed is not None:
            np.random.seed(seed)
        
        vertices_list = []
        complexity = max(1, min(10, complexity))
        
        # Generate random strokes
        for _ in range(complexity):
            # Number of points in this stroke
            n_points = np.random.randint(3, 8)
            
            # Generate control points
            t = np.linspace(0, 1, n_points)
            
            # Random but smooth curve using sine waves
            freq_x = np.random.uniform(1, 3)
            freq_y = np.random.uniform(1, 3)
            phase_x = np.random.uniform(0, 2 * np.pi)
            phase_y = np.random.uniform(0, 2 * np.pi)
            
            x = np.sin(freq_x * t * 2 * np.pi + phase_x) * 0.3
            y = np.sin(freq_y * t * 2 * np.pi + phase_y) * 0.3
            
            # Add some noise
            x += np.random.normal(0, 0.02, size=n_points)
            y += np.random.normal(0, 0.02, size=n_points)
            
            # Center and scale
            x = x * np.random.uniform(0.5, 1.0)
            y = y * np.random.uniform(0.5, 1.0)
            
            # Random offset
            x += np.random.uniform(-0.2, 0.2)
            y += np.random.uniform(-0.2, 0.2)
            
            z = np.zeros_like(x)
            
            vertices = np.stack([x, y, z], axis=1).astype(np.float32)
            vertices_list.append(vertices)
        
        return vertices_list