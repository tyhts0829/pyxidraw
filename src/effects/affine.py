"""
affine エフェクト（合成アフィン：スケール→回転）

- ピボットを中心にスケール後、XYZ（Rz·Ry·Rx の合成）回転を一括適用します。
- 既定でわずかな非等方スケールと Z 回り回転を与え、効果を視認しやすくしています。

パラメータ:
- pivot: None の場合はジオメトリの平均座標を自動採用。
- angles_rad: (rx, ry, rz) [rad]。
- scale: (sx, sy, sz) 倍率。

注意:
- 平行移動は別エフェクト（translate）で行います。
"""

from __future__ import annotations

import numpy as np
from numba import njit

from common.types import Vec3
from engine.core.geometry import Geometry

from .registry import effect


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

    # 全頂点に変換を一度に適用（中心へ移動 -> スケール -> 回転 -> 中心へ戻す）
    centered = vertices - center
    scaled = centered * scale
    rotated = scaled @ R.T
    transformed = rotated + center

    return transformed


@effect()
def affine(
    g: Geometry,
    *,
    pivot: Vec3 | None = None,
    angles_rad: Vec3 = (0.0, 0.0, 0.35),  # ≈ 20° around Z to make intent visible
    scale: Vec3 = (1.1, 0.85, 1.0),  # slight anisotropic scaling for clarity
) -> Geometry:
    """任意の変換（スケール→回転→移動）を適用する純関数エフェクト。

    既定値ポリシー:
    - pivot: None の場合は入力ジオメトリの平均座標（概ね中心）を使用。
    - angles_rad: Z 軸に 0.35rad（約20°）。
    - scale: X=1.1, Y=0.85（わずかに非等方）で変形の意図を視覚化。
    """
    coords, offsets = g.as_arrays(copy=False)

    if len(coords) == 0:
        return Geometry(coords.copy(), offsets.copy())

    if (
        pivot in ((0, 0, 0), None)
        and scale == (1, 1, 1)
        and abs(angles_rad[0]) < 1e-10
        and abs(angles_rad[1]) < 1e-10
        and abs(angles_rad[2]) < 1e-10
    ):
        return Geometry(coords.copy(), offsets.copy())

    if pivot is None:
        # 幾何の平均座標をピボットにする（矩形中心に近い）
        center_np = coords.mean(axis=0).astype(np.float32)
    else:
        center_np = np.array(pivot, dtype=np.float32)
    scale_np = np.array(scale, dtype=np.float32)
    rotate_radians = np.array(angles_rad, dtype=np.float32)

    transformed_coords = _apply_combined_transform(coords, center_np, scale_np, rotate_radians)
    return Geometry(transformed_coords, offsets.copy())


# validate_spec 用（緩やかなメタ）
affine.__param_meta__ = {
    "pivot": {"type": "vec3"},
    "angles_rad": {"type": "vec3"},
    "scale": {"type": "vec3"},
}
