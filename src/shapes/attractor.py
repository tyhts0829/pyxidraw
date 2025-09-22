from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import shape


@shape
def attractor(
    *,
    attractor_type: str = "aizawa",
    points: int = 10000,
    dt: float = 0.01,
    scale: float = 1.0,
    **params: Any,
) -> Geometry:
    """各種アトラクタの軌跡を生成する。"""
    if attractor_type == "lorenz":
        att = LorenzAttractor(dt=dt, steps=points, scale=scale, **params)
    elif attractor_type == "rossler":
        att = RosslerAttractor(dt=dt, steps=points, scale=scale, **params)
    elif attractor_type == "aizawa":
        att = AizawaAttractor(dt=dt, steps=points, scale=scale, **params)
    elif attractor_type == "three_scroll":
        att = ThreeScrollAttractor(dt=dt, steps=points, scale=scale, **params)
    elif attractor_type == "dejong":
        att = DeJongAttractor(steps=points, scale=scale, **params)
    else:
        att = LorenzAttractor(dt=dt, steps=points, scale=scale, **params)
    vertices = att.integrate()
    if scale == 1.0:
        vertices = _normalize_vertices(vertices)
    return Geometry.from_lines([vertices])


def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    """頂点群を原点中心の単位立方体へ正規化する。"""
    min_vals = np.min(vertices, axis=0)
    max_vals = np.max(vertices, axis=0)
    center = (min_vals + max_vals) * 0.5
    rng = np.max(max_vals - min_vals)
    if rng > 0:
        return (vertices - center) / rng
    return vertices - center


class BaseAttractor(ABC):
    """全アトラクタの共通基底クラス（重複を削減）。"""

    def __init__(self, dt: float = 0.01, steps: int = 10000, scale: float = 1.0):
        self.dt = dt
        self.steps = steps
        self.scale = scale

    @abstractmethod
    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        """アトラクタ方程式の微分（時間発展）を計算。"""
        pass

    @abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        """既定の初期状態ベクトルを取得。"""
        pass

    def integrate(self, initial_state: np.ndarray | None = None) -> np.ndarray:
        """RK4 法でアトラクタを数値積分します。"""
        if initial_state is None:
            state = self._get_initial_state()
        else:
            state = np.array(initial_state, dtype=np.float32)

        # Pre-allocate trajectory array
        trajectory = np.empty((self.steps, len(state)), dtype=np.float32)

        # Vectorized RK4 integration
        for i in range(self.steps):
            trajectory[i] = state
            k1 = self._derivatives(state)
            k2 = self._derivatives(state + 0.5 * self.dt * k1)
            k3 = self._derivatives(state + 0.5 * self.dt * k2)
            k4 = self._derivatives(state + self.dt * k3)
            state = state + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        return trajectory * self.scale


class LorenzAttractor(BaseAttractor):
    def __init__(self, sigma=10.0, rho=28.0, beta=8 / 3, dt=0.01, steps=10000, scale=1.0):
        super().__init__(dt, steps, scale)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        return np.array([dx, dy, dz], dtype=np.float32)

    def _get_initial_state(self) -> np.ndarray:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)


class RosslerAttractor(BaseAttractor):
    def __init__(self, a=0.2, b=0.2, c=5.7, dt=0.01, steps=10000, scale=1.0):
        super().__init__(dt, steps, scale)
        self.a = a
        self.b = b
        self.c = c

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = -y - z
        dy = x + self.a * y
        dz = self.b + z * (x - self.c)
        return np.array([dx, dy, dz], dtype=np.float32)

    def _get_initial_state(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)


class AizawaAttractor(BaseAttractor):
    def __init__(self, a=0.95, b=0.7, c=0.6, d=3.5, dt=0.01, steps=10000, scale=1.0):
        super().__init__(dt, steps, scale)
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = (z - self.b) * x - self.d * y
        dy = self.d * x + (z - self.b) * y
        dz = self.c - self.a * z - z * (x**2 + y**2)
        return np.array([dx, dy, dz], dtype=np.float32)

    def _get_initial_state(self) -> np.ndarray:
        return np.array([0.1, 0.0, 0.0], dtype=np.float32)


class ThreeScrollAttractor(BaseAttractor):
    def __init__(self, a=40, b=0.833, c=0.5, d=0.5, e=0.65, dt=0.01, steps=10000, scale=1.0):
        super().__init__(dt, steps, scale)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e

    def _derivatives(self, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        dx = self.a * (y - x) + self.d * x * z
        dy = self.b * x - x * z + self.c * y
        dz = self.e * z + x * y
        return np.array([dx, dy, dz], dtype=np.float32)

    def _get_initial_state(self) -> np.ndarray:
        return np.array([0.1, 0.0, 0.0], dtype=np.float32)


class DeJongAttractor:
    def __init__(self, a=1.4, b=-2.3, c=2.4, d=-2.1, steps=10000, scale=1.0, initial_state=(0, 0)):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.steps = steps
        self.scale = scale
        self.initial_state = initial_state

    def _map(self, state: tuple[float, float]) -> tuple[float, float]:
        x, y = state
        x_new = np.sin(self.a * y) - np.cos(self.b * x)
        y_new = np.sin(self.c * x) - np.cos(self.d * y)
        return (x_new, y_new)

    def integrate(self, initial_state: tuple[float, float] | None = None) -> np.ndarray:
        if initial_state is None:
            state = self.initial_state
        else:
            state = tuple(initial_state)

        # Pre-allocate with dtype for better performance
        trajectory = np.empty((self.steps, 3), dtype=np.float32)

        # Vectorized time calculation
        time_values = np.arange(self.steps, dtype=np.float32) * self.scale * 0.001

        for i in range(self.steps):
            trajectory[i, 0], trajectory[i, 1] = state
            trajectory[i, 2] = time_values[i]
            state = self._map(state)

        trajectory[:, 0:2] *= self.scale
        return trajectory


attractor.__param_meta__ = {
    "attractor_type": {
        "choices": ["aizawa", "lorenz", "rossler", "three_scroll", "dejong"],
    },
    "points": {"type": "integer", "min": 100, "max": 20000},
    "dt": {"type": "number", "min": 0.001, "max": 0.05},
    "scale": {"type": "number", "min": 0.1, "max": 5.0},
}
