"""
affine エフェクト（合成アフィン：スケール→回転）

- ピボットを中心にスケール後、XYZ（Rz·Ry·Rx の合成）回転を一括適用します。

パラメータ（新API）:
- auto_center: True ならジオメトリの平均座標を中心に使用。False なら `pivot` を使用。
- pivot: `auto_center=False` のときの中心座標。
- angles_rad: (rx, ry, rz) [rad]。
- scale: (sx, sy, sz) 倍率。

注意:
- 平行移動は別エフェクト（translate）で行います。
"""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[attr-defined]

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
    auto_center: bool = True,
    pivot: Vec3 = (0.0, 0.0, 0.0),
    angles_rad: Vec3 = (np.pi / 4, np.pi / 4, np.pi / 4),
    scale: Vec3 = (0.5, 0.5, 0.5),
) -> Geometry:
    """スケール後に回転を適用（合成アフィン）。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。各行が 1 本のポリラインを表す（`offsets` で区切る）。
    auto_center : bool, default True
        True のとき形状の平均座標を中心に使用。False のとき `pivot` を使用。
    pivot : tuple[float, float, float], default (0.0, 0.0, 0.0)
        `auto_center=False` のときの変換中心 [mm]。
    angles_rad : tuple[float, float, float], default (π/4, π/4, π/4)
        回転角 [rad]（X, Y, Z）。
    scale : tuple[float, float, float], default (0.5, 0.5, 0.5)
        スケール倍率（X, Y, Z）。
    """
    coords, offsets = g.as_arrays(copy=False)

    if len(coords) == 0:
        return Geometry(coords.copy(), offsets.copy())

    # 恒等変換なら早期リターン（中心の選択に依存しない）
    if (
        scale == (1, 1, 1)
        and abs(angles_rad[0]) < 1e-10
        and abs(angles_rad[1]) < 1e-10
        and abs(angles_rad[2]) < 1e-10
    ):
        return Geometry(coords.copy(), offsets.copy())

    # 中心座標を決定
    if auto_center:
        center_np = coords.mean(axis=0).astype(np.float32)
    else:
        center_np = np.array(pivot, dtype=np.float32)
    scale_np = np.array(scale, dtype=np.float32)
    rotate_radians = np.array(angles_rad, dtype=np.float32)

    transformed_coords = _apply_combined_transform(coords, center_np, scale_np, rotate_radians)
    return Geometry(transformed_coords, offsets.copy())


# UI 表示のためのメタ情報（RangeHint 構築に使用）
affine.__param_meta__ = {
    "auto_center": {"type": "bool"},
    "pivot": {
        "type": "vec3",
        "min": (-300.0, -300.0, -300.0),
        "max": (300.0, 300.0, 300.0),
    },
    "angles_rad": {
        "type": "vec3",
        "min": (0, 0, 0),
        "max": (2 * np.pi, 2 * np.pi, 2 * np.pi),
    },
    "scale": {"type": "vec3", "min": (0.25, 0.25, 0.25), "max": (4.0, 4.0, 4.0)},
}
