from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect
from common.param_utils import ensure_vec3
from common.types import Vec3


def _wobble_vertices(
    vertices_list: list[np.ndarray], amplitude: float, frequency: Vec3, phase: float
) -> list[np.ndarray]:
    """各頂点に対してサイン波によるゆらぎ（wobble）を加える内部関数。"""
    new_vertices_list = []
    for vertices in vertices_list:
        if len(vertices) == 0:
            new_vertices_list.append(vertices)
            continue

        new_vertices = vertices.astype(np.float32).copy()
        # ベクトル化された計算
        # x軸方向のゆらぎ
        new_vertices[:, 0] += amplitude * np.sin(2 * np.pi * frequency[0] * new_vertices[:, 0] + phase)
        # y軸方向のゆらぎ
        new_vertices[:, 1] += amplitude * np.sin(2 * np.pi * frequency[1] * new_vertices[:, 1] + phase)
        # z軸方向のゆらぎ（2D の場合は 0 のまま）
        if new_vertices.shape[1] > 2:
            new_vertices[:, 2] += amplitude * np.sin(2 * np.pi * frequency[2] * new_vertices[:, 2] + phase)
        new_vertices_list.append(new_vertices)
    return new_vertices_list


@effect()
def wobble(
    g: Geometry,
    *,
    amplitude: float = 1.0,
    frequency: float | Vec3 = (0.1, 0.1, 0.1),
    phase: float = 0.0,
    **_params: Any,
) -> Geometry:
    """線にウォブル/波の歪みを追加（純関数）。

    Notes:
        - amplitude は座標単位（mm 相当）。0..1 正規化ではありません。
        - frequency は空間周波数 [cycles per unit]。float なら全軸同一、タプルは (fx, fy, fz)。
        - phase はラジアン。
    """
    coords, offsets = g.as_arrays(copy=False)

    # frequency をタプルに正規化（係数スケーリングは廃止）
    if isinstance(frequency, (int, float)):
        f = float(frequency)
        freq_tuple = (f, f, f)
    elif isinstance(frequency, (list, tuple)):
        fx, fy, fz = ensure_vec3(tuple(float(x) for x in frequency))
        freq_tuple = (fx, fy, fz)
    else:
        freq_tuple = (0.1, 0.1, 0.1)

    if len(coords) == 0:
        return Geometry(coords.copy(), offsets.copy())

    vertices_list = [coords[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    wobbled_vertices = _wobble_vertices(vertices_list, float(amplitude), freq_tuple, float(phase))
    if not wobbled_vertices:
        return Geometry(coords.copy(), offsets.copy())
    return Geometry.from_lines(wobbled_vertices)
