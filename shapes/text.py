from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseShape


class Text(BaseShape):
    """Text shape generator (placeholder implementation)."""
    
    def generate(self, text: str = "HELLO", size: float = 0.1, **params: Any) -> list[np.ndarray]:
        """Generate text as line segments.
        
        Note: This is a placeholder implementation. In a real implementation,
        this would use a font rendering library to convert text to vectors.
        
        Args:
            text: Text string to render
            size: Text size
            **params: Additional parameters (ignored)
            
        Returns:
            List of vertex arrays for text outlines
        """
        # For now, just return a simple rectangle as placeholder
        # In production, this would use a proper font rendering system
        vertices_list = []
        
        # Simple box around where text would be
        width = len(text) * size * 0.6
        height = size
        
        box = np.array([
            [-width/2, -height/2, 0],
            [width/2, -height/2, 0],
            [width/2, height/2, 0],
            [-width/2, height/2, 0],
            [-width/2, -height/2, 0]
        ], dtype=np.float32)
        
        vertices_list.append(box)
        
        return vertices_list