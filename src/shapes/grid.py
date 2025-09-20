from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import shape


def _generate_grid(nx: int, ny: int) -> list[np.ndarray]:
    """グリッドの頂点列を生成します。

    引数:
        nx: 垂直線の本数
        ny: 水平線の本数

    返り値:
        グリッド各線の頂点配列のリスト
    """
    if max(nx, ny) < 1:
        # Return empty grid if divisions are too small
        return []

    x_coords = np.linspace(-0.5, 0.5, nx)
    y_coords = np.linspace(-0.5, 0.5, ny)

    # Pre-allocate memory（行数は暗黙に nx + ny）
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


MAX_DIVISIONS = 100


@shape
def grid(subdivisions: tuple[float, float] = (20.0, 20.0), **params: Any) -> Geometry:
    """1x1 の正方形グリッドを生成します。"""
    nx_raw, ny_raw = subdivisions
    nx = min(MAX_DIVISIONS, max(1, int(round(nx_raw))))
    ny = min(MAX_DIVISIONS, max(1, int(round(ny_raw))))
    vertices_list = _generate_grid(nx, ny)
    return Geometry.from_lines(vertices_list)


grid.__param_meta__ = {
    "subdivisions": {
        "type": "number",
        "min": (1, 1),
        "max": (100, 100),
        "step": (1, 1),
    }
}
