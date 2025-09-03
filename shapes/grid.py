from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry
from .registry import shape
from .base import BaseShape


def _generate_grid(nx: int, ny: int) -> list[np.ndarray]:
    """Generate grid vertices.

    Args:
        nx: Number of vertical lines
        ny: Number of horizontal lines

    Returns:
        List of vertex arrays for grid lines
    """
    if max(nx, ny) < 1:
        # Return empty grid if divisions are too small
        return []

    x_coords = np.linspace(-0.5, 0.5, nx)
    y_coords = np.linspace(-0.5, 0.5, ny)

    # Pre-allocate memory
    total_lines = nx + ny
    vertices_list: list[np.ndarray] = []

    # Generate vertical lines vectorized
    vertical_lines = np.empty((nx, 2, 3), dtype=np.float32)
    vertical_lines[:, :, 0] = x_coords[:, np.newaxis]  # x coords
    vertical_lines[:, 0, 1] = -0.5  # start y coordinate
    vertical_lines[:, 1, 1] = 0.5  # end y coordinate
    vertical_lines[:, :, 2] = 0.0  # z coordinate

    # Generate horizontal lines vectorized
    horizontal_lines = np.empty((ny, 2, 3), dtype=np.float32)
    horizontal_lines[:, 0, 0] = -0.5  # start x coordinate
    horizontal_lines[:, 1, 0] = 0.5  # end x coordinate
    horizontal_lines[:, :, 1] = y_coords[:, np.newaxis]  # y coords
    horizontal_lines[:, :, 2] = 0.0  # z coordinate

    # Store in vertices_list
    vertices_list.extend(vertical_lines)
    vertices_list.extend(horizontal_lines)

    return vertices_list




@shape
class Grid(BaseShape):
    """Grid shape generator."""

    MAX_DIVISIONS = 50

    def generate(self, subdivisions: tuple[float, float] = (0.1, 0.1), **params: Any) -> Geometry:
        """Generate a 1x1 square grid with specified divisions.

        Args:
            subdivisions: (x_divisions, y_divisions) as floats 0.0-1.0
            **params: Additional parameters (ignored)

        Returns:
            Geometry object containing grid lines
        """
        nx, ny = subdivisions
        nx = int(nx * Grid.MAX_DIVISIONS)
        ny = int(ny * Grid.MAX_DIVISIONS)

        # Generate grid (caching handled by BaseShape)
        vertices_list = _generate_grid(nx, ny)
        return Geometry.from_lines(vertices_list)
