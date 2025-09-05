from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

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
    ) -> Geometry:
        """2D/3D リサージュ曲線を生成します。

        引数:
            freq_x: X 軸の周波数（a）
            freq_y: Y 軸の周波数（b）
            freq_z: Z 軸の周波数（c）。0.0 なら Z はフラット（2D）
            phase: X の位相（ラジアン, δx）
            phase_y: Y の位相（ラジアン, δy）
            phase_z: Z の位相（ラジアン, δz）
            points: サンプル点数
            **params: 追加パラメータ（未使用）

        返り値:
            ポリライン1本としての曲線の Geometry
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

        return Geometry.from_lines([vertices])
