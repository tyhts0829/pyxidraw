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
def translate(
    g: Geometry,
    *,
    # 旧API
    offset: Vec3 | None = None,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    offset_z: float = 0.0,
    # 新API（推奨）
    delta: Vec3 | None = None,
) -> Geometry:
    """指定ベクトルで平行移動（純関数）。新APIは `delta` を推奨。"""
    coords, offsets = g.as_arrays(copy=False)
    if delta is not None:
        ox, oy, oz = float(delta[0]), float(delta[1]), float(delta[2])
    elif offset is not None:
        ox, oy, oz = float(offset[0]), float(offset[1]), float(offset[2])
    else:
        ox, oy, oz = float(offset_x), float(offset_y), float(offset_z)

    if (ox == 0.0 and oy == 0.0 and oz == 0.0) or coords.size == 0:
        return Geometry(coords.copy(), offsets.copy())

    vec = np.array([ox, oy, oz], dtype=np.float32)
    translated_coords = _apply_translation(coords, vec)
    return Geometry(translated_coords, offsets.copy())

translate.__param_meta__ = {
    "offset_x": {"type": "number"},
    "offset_y": {"type": "number"},
    "offset_z": {"type": "number"},
}


# 後方互換クラスは廃止（関数APIのみ）
