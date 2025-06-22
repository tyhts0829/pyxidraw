from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _apply_transformations(
    vertices_list: Sequence[np.ndarray],
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
) -> list[np.ndarray]:
    center_np = np.array(center, dtype=np.float32)
    scale_np = np.array(scale, dtype=np.float32)
    rotate_np = np.array(rotate, dtype=np.float32)
    transformed_list = []
    for vertices in vertices_list:
        sx = np.sin(rotate_np[0])
        cx = np.cos(rotate_np[0])
        sy = np.sin(rotate_np[1])
        cy = np.cos(rotate_np[1])
        sz = np.sin(rotate_np[2])
        cz = np.cos(rotate_np[2])
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        R = Rz @ Ry @ Rx
        rotated = vertices @ R.T
        transformed = center_np + rotated * scale_np
        transformed_list.append(transformed)
    return transformed_list


class Transform(BaseEffect):
    """任意の変換行列を適用します。"""

    TAU = math.tau  # 全回転角度（2 * pi）

    def apply(
        self,
        vertices_list: list[np.ndarray],
        center: tuple[float, float, float] = (0, 0, 0),
        scale: tuple[float, float, float] = (1, 1, 1),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params: Any,
    ) -> list[np.ndarray]:
        """

        Args:
            vertices_list: 入力頂点配列
            center: 変換の中心点 (x, y, z)
            scale: スケール係数 (x, y, z)
            rotate: 回転角度（ラジアン） (x, y, z) 入力は0.0-1.0の範囲を想定。内部でmath.tauを掛けてラジアンに変換される。
            **params: 追加パラメータ（無視される）

        """

        # エッジケース: 空のリスト
        if not vertices_list:
            return []

        # tau
        rotate_radians = (
            rotate[0] * self.TAU,
            rotate[1] * self.TAU,
            rotate[2] * self.TAU,
        )

        return _apply_transformations(vertices_list, center=center, scale=scale, rotate=rotate_radians)
