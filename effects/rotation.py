from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect
from .registry import effect


@njit(fastmath=True, cache=True)
def _create_rotation_matrix(angle_x: float, angle_y: float, angle_z: float) -> np.ndarray:
    """統合された回転行列を作成します。

    回転の順序: Z * Y * X
    """
    # 各軸の sin/cos を一度だけ計算
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)

    # 回転行列を直接計算（メモリ割り当てを最小化）
    R = np.empty((3, 3), dtype=np.float32)

    # Z * Y * X の結合行列を直接計算
    R[0, 0] = cy * cz
    R[0, 1] = sx * sy * cz - cx * sz
    R[0, 2] = cx * sy * cz + sx * sz

    R[1, 0] = cy * sz
    R[1, 1] = sx * sy * sz + cx * cz
    R[1, 2] = cx * sy * sz - sx * cz

    R[2, 0] = -sy
    R[2, 1] = sx * cy
    R[2, 2] = cx * cy

    return R


@njit(fastmath=True, cache=True)
def _apply_rotation_inplace(vertices: np.ndarray, rotation_matrix: np.ndarray, center: np.ndarray) -> np.ndarray:
    """頂点に回転行列を適用します（メモリ効率重視）。"""
    # 全頂点を一度に処理（ブロードキャスト利用）
    centered = vertices - center
    # 回転を適用（行ベクトルのため転置）
    rotated = centered @ rotation_matrix.T
    # 中心に戻す
    result = rotated + center
    return result


@effect
class Rotation(BaseEffect):
    """指定された軸周りに頂点を回転します。"""

    TAU = math.tau  # 2 * pi, 全回転角度

    def apply(
        self,
        coords: np.ndarray,
        offsets: np.ndarray,
        center: tuple[float, float, float] = (0, 0, 0),
        rotate: tuple[float, float, float] = (0, 0, 0),
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """回転エフェクトを適用します。

        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            center: 回転の中心点 (x, y, z)
            rotate: 各軸周りの回転角（ラジアン）(x, y, z) 入力は0.0-1.0の範囲を想定。内部でmath.tauを掛けてラジアンに変換される。
            **params: 追加パラメータ（無視される）

        Returns:
            (rotated_coords, offsets): 回転された座標配列とオフセット配列
        """
        # 回転角度を抽出
        rotate_x, rotate_y, rotate_z = rotate
        rotate_x *= self.TAU  # 0.0-1.0 -> 0.0-2π
        rotate_y *= self.TAU  # 0.0-1.0 -> 0.0-2π
        rotate_z *= self.TAU

        # エッジケース: 回転がない場合は元のデータをそのまま返す
        if abs(rotate_x) < 1e-10 and abs(rotate_y) < 1e-10 and abs(rotate_z) < 1e-10:
            return coords.copy(), offsets.copy()

        # エッジケース: 空の座標配列
        if len(coords) == 0:
            return coords.copy(), offsets.copy()

        # 統合された回転行列を作成
        R = _create_rotation_matrix(rotate_x, rotate_y, rotate_z)

        # 中心点をnumpy配列に変換
        center_np = np.array(center, dtype=np.float32)

        # 全頂点に回転を一度に適用
        rotated_coords = _apply_rotation_inplace(coords, R, center_np)

        return rotated_coords, offsets.copy()
