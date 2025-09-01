from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry_data import GeometryData

from .base import BaseShape
from .registry import shape


@shape
class Lissajous(BaseShape):
    """Lissajous curve shape generator (2D/3D).

    - Default behavior matches previous 2D implementation (Z=0).
    - Provide `freq_z` and `phase_z` to generate a 3D Lissajous figure.
    """

    def generate(
        self,
        freq_x: float = 3.0,
        freq_y: float = 2.0,
        freq_z: float = 4.0,
        phase: float = 0.0,
        phase_y: float = 0.0,
        phase_z: float = 0.0,
        points: int = 1000,
        **params: Any,
    ) -> GeometryData:
        """Generate a 2D/3D Lissajous curve.

        Args:
            freq_x: X-axis frequency (a)
            freq_y: Y-axis frequency (b)
            freq_z: Z-axis frequency (c). 0.0 keeps Z flat (2D)
            phase: Phase offset for X in radians (δx)
            phase_y: Phase offset for Y in radians (δy)
            phase_z: Phase offset for Z in radians (δz)
            points: Number of sample points
            **params: Additional parameters (ignored)

        Returns:
            GeometryData object containing the curve vertices as a single polyline
        """
        # Sample parameter over one full cycle
        t = np.linspace(0, 2 * np.pi, points)

        # Amplitude 0.5 to match conventions of other shapes (roughly unit box)
        amp = 0.5

        x = np.sin(freq_x * t + phase) * amp
        y = np.sin(freq_y * t + phase_y) * amp
        # If freq_z == 0 (default), produce a flat Z (2D). Otherwise, full 3D.
        if freq_z == 0.0:
            z = np.zeros_like(t)
        else:
            z = np.sin(freq_z * t + phase_z) * amp

        vertices = np.stack([x, y, z], axis=1).astype(np.float32)

        return GeometryData.from_lines([vertices])
