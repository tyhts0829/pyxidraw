from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseShape


def _sphere_cached(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere vertices.
    
    Args:
        subdivisions: Subdivision level (0-5)
        
    Returns:
        List of vertex arrays for sphere triangles
    """
    # Number of segments based on subdivision level
    n_segments = 8 * (2 ** subdivisions)
    n_rings = n_segments // 2
    
    vertices_list = []
    
    # Generate sphere using latitude/longitude lines
    for i in range(n_rings):
        lat1 = np.pi * i / n_rings
        lat2 = np.pi * (i + 1) / n_rings
        
        ring = []
        for j in range(n_segments + 1):
            lon = 2 * np.pi * j / n_segments
            
            # Calculate vertices for this segment
            x1 = np.sin(lat1) * np.cos(lon) * 0.5
            y1 = np.sin(lat1) * np.sin(lon) * 0.5
            z1 = np.cos(lat1) * 0.5
            
            x2 = np.sin(lat2) * np.cos(lon) * 0.5
            y2 = np.sin(lat2) * np.sin(lon) * 0.5
            z2 = np.cos(lat2) * 0.5
            
            ring.extend([[x1, y1, z1], [x2, y2, z2]])
        
        vertices_list.append(np.array(ring, dtype=np.float32))
    
    return vertices_list


def _load_precomputed_sphere(subdivision: int) -> list[np.ndarray] | None:
    """Load pre-computed sphere vertex data.
    
    Args:
        subdivision: Subdivision level
        
    Returns:
        Vertex arrays or None if not found
    """
    data_dir = Path(__file__).parents[1] / "data" / "sphere"
    
    # Check if data directory exists
    if not data_dir.exists():
        return None
    
    pkl_file = data_dir / f"sphere_tri_{subdivision}_vertices_list.pkl"
    if pkl_file.exists():
        with open(pkl_file, "rb") as f:
            vertices_list = pickle.load(f)
            # Convert to list of numpy arrays if needed
            if isinstance(vertices_list, list):
                return [np.array(v, dtype=np.float32) for v in vertices_list]
            return vertices_list
    
    return None


class Sphere(BaseShape):
    """Sphere shape generator using pre-computed vertex data."""
    
    def generate(self, subdivisions: float = 0.5, **params: Any) -> list[np.ndarray]:
        """Generate a sphere with radius 1.
        
        Args:
            subdivisions: Subdivision level (0.0-1.0, mapped to 0-5)
            **params: Additional parameters (ignored)
            
        Returns:
            List of vertex arrays for sphere triangles
        """
        MIN_SUBDIVISIONS = 0
        MAX_SUBDIVISIONS = 5
        subdivisions_int = int(subdivisions * MAX_SUBDIVISIONS)
        if subdivisions_int < MIN_SUBDIVISIONS:
            subdivisions_int = MIN_SUBDIVISIONS
        
        # Try to load pre-computed data with caching
        precomputed = _load_precomputed_sphere(subdivisions_int)
        if precomputed is not None:
            return precomputed
        
        # Fallback: generate simple sphere with caching
        return _sphere_cached(subdivisions_int)