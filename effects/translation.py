from __future__ import annotations

import numpy as np
from numba import njit

from .registry import effect
from engine.core.geometry import Geometry
from common.types import Vec3


@njit(fastmath=True, cache=True)
def _apply_translation(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """頂点に移動を適用します。"""
    translated = vertices + offset
    return translated.astype(np.float32)


@effect()
def translation(
    g: Geometry,
    *,
    offset: Vec3 | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
) -> Geometry:
    """指定されたオフセットで頂点を移動（純関数）。"""
    coords, offsets = g.as_arrays(copy=False)
    if offset is not None:
        ox, oy, oz = float(offset[0]), float(offset[1]), float(offset[2])
    else:
        ox, oy, oz = float(offset_x), float(offset_y), float(offset_z)

    if (ox == 0.0 and oy == 0.0 and oz == 0.0) or coords.size == 0:
        return Geometry(coords.copy(), offsets.copy())

    vec = np.array([ox, oy, oz], dtype=np.float32)
    translated_coords = _apply_translation(coords, vec)
    return Geometry(translated_coords, offsets.copy())


# 後方互換クラスは廃止（関数APIのみ）
