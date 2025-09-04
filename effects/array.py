from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .registry import effect
from engine.core.geometry import Geometry
from common.param_utils import ensure_vec3, norm_to_int, norm_to_rad
from common.types import Vec3


@njit(fastmath=True, cache=True)
def _apply_transform_to_coords(
    coords: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    rotate: np.ndarray,
    offset: np.ndarray,
) -> np.ndarray:
    """座標に変換を適用します（中心移動 -> スケール -> 回転 -> オフセット -> 中心に戻す）。"""
    # 回転行列を計算
    sx, sy, sz = np.sin(rotate)
    cx, cy, cz = np.cos(rotate)
    
    # Z * Y * X の結合行列
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
    
    # 中心を原点に移動
    centered = coords - center
    # スケール適用
    scaled = centered * scale
    # 回転適用
    rotated = scaled @ R.T
    # オフセット適用
    offset_applied = rotated + offset
    # 中心に戻す
    transformed = offset_applied + center
    
    return transformed


@njit(fastmath=True, cache=True)
def _update_scale(current_scale: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """スケール値を更新します。"""
    return current_scale * scale


MAX_DUPLICATES = 10


@effect()
def repeat(
    g: Geometry,
    *,
    count: int = 0,
    offset: Vec3 = (0.0, 0.0, 0.0),
    angles_rad_step: Vec3 = (0.0, 0.0, 0.0),
    scale: Vec3 = (0.5, 0.5, 0.5),
    pivot: Vec3 = (0.0, 0.0, 0.0),
) -> Geometry:
    """入力のコピーを配列状に生成（純関数）。"""
    coords, offsets = g.as_arrays(copy=False)
    # count があれば優先。なければ 0..1 → 0..MAX_DUPLICATES（整数）
    n_int = int(count)
    if n_int <= 0 or g.is_empty or offsets.size <= 1:
        return Geometry(coords.copy(), offsets.copy())

    center_np = np.array(pivot, dtype=np.float32)
    offset_np = np.array(offset, dtype=np.float32)
    scale_np = np.array(scale, dtype=np.float32)

    rotate_radians = np.array(angles_rad_step, dtype=np.float32)

    # 生成する線のリスト（Geometry.from_lines で正しい offsets を構築）
    lines: list[np.ndarray] = []

    # 元の線を追加
    for i in range(len(offsets) - 1):
        lines.append(coords[offsets[i] : offsets[i + 1]].copy())

    current_coords = coords.copy()
    current_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    for n in range(n_int):
        current_scale = _update_scale(current_scale, scale_np)
        current_coords = _apply_transform_to_coords(
            current_coords,
            center_np,
            current_scale,
            rotate_radians * (n + 1),
            offset_np * (n + 1),
        )
        # 複製後の各線を追加
        for i in range(len(offsets) - 1):
            lines.append(current_coords[offsets[i] : offsets[i + 1]].copy())

    return Geometry.from_lines(lines)

# validate_spec 用のメタデータ（最小限）
repeat.__param_meta__ = {
    "count": {"type": "integer", "min": 0, "max": MAX_DUPLICATES},
}


# 後方互換クラスは廃止（関数APIのみ）
