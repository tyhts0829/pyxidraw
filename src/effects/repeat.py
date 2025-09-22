"""
repeat エフェクト（配列複製）

- 入力のポリライン群を複製して規則的な配列を作ります。
- 各複製に対して、ピボット基準のスケール、XYZ 回転（ラジアン）、平行移動を
  「ステップごとに累積」適用します（n 回目には (n+1) 倍の回転/オフセット）。
- 実装は numba 最適化した行列計算で一括変換し、元の線も結果に残します。

主なパラメータ:
- count: 複製回数。0 で変化なし。上限は MAX_DUPLICATES(=10)。
- offset: 各ステップの並進量 [mm]。n 回目には (n+1) 倍が適用。
- angles_rad_step: 各ステップの回転量 [rad]。Z→Y→X の順（合成行列）。
- scale: 各ステップのスケール。累積乗算。
- pivot: 変換の中心。

注意:
- `Geometry.from_lines` で offsets を再構築するため、複数線でも安全。
- 複製数やスケール・回転を大きくすると頂点数が増え描画コストが上がります。

使用例: `(E.pipeline.repeat(count=4, offset=(12,0,0)).build())(g)`
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from common.types import Vec3
from engine.core.geometry import Geometry

from .registry import effect


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
    count: int = 3,
    offset: Vec3 = (0.0, 0.0, 0.0),
    angles_rad_step: Vec3 = (0.1, 0.1, 0.1),
    scale: Vec3 = (0.8, 0.8, 0.8),
    pivot: Vec3 = (0.0, 0.0, 0.0),
) -> Geometry:
    """入力のコピーを配列状に生成（純関数）。

    既定値の方針（2025-09-06）:
        - count=3, offset=(12,0,0)mm, scale=(1,1,1), angles_rad_step=(0,0,0)
        - 300mm 正方キャンバス中央の立方体で、横に4つ並びが収まり、効果が一目で分かる。
          累積オフセットは 1,3,6…倍になる実装のため、1ステップ12mmに抑えて画面内に収める。
    """
    coords, offsets = g.as_arrays(copy=False)
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


# UI 表示のためのメタ情報（RangeHint 構築に使用）
repeat.__param_meta__ = {
    "count": {"type": "integer", "min": 0, "max": MAX_DUPLICATES, "step": 1},
    "offset": {
        "type": "vec3",
        "min": (-300.0, -300.0, -300.0),
        "max": (300.0, 300.0, 300.0),
    },
    "angles_rad_step": {
        "type": "vec3",
        "min": (-math.pi, -math.pi, -math.pi),
        "max": (math.pi, math.pi, math.pi),
    },
    "scale": {"type": "vec3", "min": (0.1, 0.1, 0.1), "max": (3.0, 3.0, 3.0)},
    "pivot": {
        "type": "vec3",
        "min": (-300.0, -300.0, -300.0),
        "max": (300.0, 300.0, 300.0),
    },
}


# 後方互換クラスは廃止（関数APIのみ）
