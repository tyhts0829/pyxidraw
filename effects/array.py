from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect
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


@effect
class Array(BaseEffect):
    """入力のコピーを配列状に生成します。"""

    MAX_DUPLICATES = 10
    TAU = 2 * np.pi  # 全回転角度

    def apply(
        self,
        coords: np.ndarray,
        offsets: np.ndarray,
        n_duplicates: float = 0.5,
        offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotate: tuple[float, float, float] = (0.5, 0.5, 0.5),
        scale: tuple[float, float, float] = (0.5, 0.5, 0.5),
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """配列エフェクトを適用します。

        Args:
            coords: 入力座標配列
            offsets: 入力オフセット配列
            n_duplicates: 複製数の係数（0.0-1.0、最大10個まで）
            offset: 各複製間のオフセット（x, y, z）
            rotate: 各複製における回転増分（0.0-1.0、0.5が中立）
            scale: 各複製におけるスケール係数（0.0-1.0、0.5が中立）
            center: 配列の中心点（x, y, z）
            **params: 追加パラメータ

        Returns:
            (array_coords, array_offsets): 配列化された座標配列とオフセット配列

        Note:
            n_duplicatesが0の場合、元のデータをそのまま返します。
            各複製では前の複製に対して累積的にtransformが適用されます。
        """
        n_duplicates_int = int(n_duplicates * self.MAX_DUPLICATES)
        if not n_duplicates_int:
            return coords.copy(), offsets.copy()

        # エッジケース: 空の座標配列
        if len(coords) == 0:
            return coords.copy(), offsets.copy()

        # NumPy配列に変換
        center_np = np.array(center, dtype=np.float32)
        offset_np = np.array(offset, dtype=np.float32)
        scale_np = np.array(scale, dtype=np.float32)
        
        # 回転を0.5が中立となるように調整し、ラジアンに変換
        rotate_adjusted = tuple((r - 0.5) for r in rotate)
        rotate_radians = np.array([
            rotate_adjusted[0] * self.TAU,
            rotate_adjusted[1] * self.TAU,
            rotate_adjusted[2] * self.TAU,
        ], dtype=np.float32)

        # 結果を格納するリスト
        all_coords = []
        all_offsets = []

        # 元のデータも含める
        all_coords.append(coords)
        all_offsets.append(offsets)

        # 累積的な変換のための変数
        current_coords = coords.copy()
        current_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        for n in range(n_duplicates_int):
            # 累積的にスケールを更新
            current_scale = _update_scale(current_scale, scale_np)
            
            # 変換を適用
            current_coords = _apply_transform_to_coords(
                current_coords,
                center_np,
                current_scale,
                rotate_radians * (n + 1),  # 回転は線形に増加
                offset_np * (n + 1),  # オフセットも線形に増加
            )
            
            all_coords.append(current_coords.copy())
            all_offsets.append(offsets.copy())

        # すべての座標とオフセットを結合
        combined_coords = np.vstack(all_coords)
        combined_offsets = np.concatenate(all_offsets)

        return combined_coords, combined_offsets
