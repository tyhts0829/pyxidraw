from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

from .registry import effect
from engine.core.geometry import Geometry
from common.param_utils import ensure_vec3, norm_to_rad
from common.types import Vec3


@njit(fastmath=True, cache=True)
def _apply_combined_transform(
    vertices: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rotate: np.ndarray,
) -> np.ndarray:
    """頂点に組み合わせ変換を適用します。"""
    # 回転行列を一度だけ計算
    sx, sy, sz = np.sin(rotate)
    cx, cy, cz = np.cos(rotate)
    
    # Z * Y * X の結合行列を直接計算
    R = np.empty((3, 3), dtype=np.float32)
    R[0, 0] = cy * cz
    R[0, 1] = sx * sy * cz - cx * sz
    R[0, 2] = cx * sy * cz + sx * sz
    R[1, 0] = cy * sz
    R[1, 1] = sx * sy * sz + cx * cz
    R[1, 2] = cx * sy * sz - sx * cz
    R[2, 0] = -sy
    R[2, 1] = sx * cy
    R[2, 2] = cx * cy
    
    # 全頂点に変換を一度に適用（スケール -> 回転 -> 移動）
    scaled = vertices * scale
    rotated = scaled @ R.T
    transformed = rotated + center
    
    return transformed


@effect()
def affine(
    g: Geometry,
    *,
    # 旧API
    center: Vec3 = (0.0, 0.0, 0.0),
    rotate: Vec3 = (0.0, 0.0, 0.0),
    # 新API（推奨）
    pivot: Vec3 | None = None,
    angles_rad: Vec3 | None = None,
    angles_deg: Vec3 | None = None,
    # 共通
    scale: Vec3 = (1.0, 1.0, 1.0),
) -> Geometry:
    """任意の変換（スケール→回転→移動）を適用する純関数エフェクト。"""
    coords, offsets = g.as_arrays(copy=False)

    if len(coords) == 0:
        return Geometry(coords.copy(), offsets.copy())

    if (center == (0, 0, 0) and scale == (1, 1, 1) and abs(rotate[0]) < 1e-10 and abs(rotate[1]) < 1e-10 and abs(rotate[2]) < 1e-10):
        return Geometry(coords.copy(), offsets.copy())

    c = pivot if pivot is not None else center
    center_np = np.array(c, dtype=np.float32)
    scale_np = np.array(scale, dtype=np.float32)
    # 角度の優先解決
    if angles_rad is not None:
        rotate_radians = np.array(angles_rad, dtype=np.float32)
    elif angles_deg is not None:
        rx, ry, rz = [math.radians(float(v)) for v in angles_deg]
        rotate_radians = np.array([rx, ry, rz], dtype=np.float32)
    else:
        # 0..1 正規化角度 → ラジアンへ統一
        rx, ry, rz = ensure_vec3(rotate)
        rotate_radians = np.array([norm_to_rad(rx), norm_to_rad(ry), norm_to_rad(rz)], dtype=np.float32)

    transformed_coords = _apply_combined_transform(coords, center_np, scale_np, rotate_radians)
    return Geometry(transformed_coords, offsets.copy())
