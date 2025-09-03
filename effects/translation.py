from __future__ import annotations

import numpy as np
from numba import njit

from .registry import effect
from engine.core.geometry import Geometry


@njit(fastmath=True, cache=True)
def _apply_translation(vertices: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """頂点に移動を適用します。"""
    translated = vertices + offset
    return translated.astype(np.float32)


@effect()
def translation(
    g: Geometry,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
) -> Geometry:
    """指定されたオフセットで頂点を移動（純関数）。"""
    coords, offsets = g.as_arrays(copy=False)
    if offset_x == 0.0 and offset_y == 0.0 and offset_z == 0.0 or coords.size == 0:
        return Geometry(coords.copy(), offsets.copy())

    offset = np.array([offset_x, offset_y, offset_z], dtype=np.float32)
    translated_coords = _apply_translation(coords, offset)
    return Geometry(translated_coords, offsets.copy())


# 後方互換クラスは廃止（関数APIのみ）
