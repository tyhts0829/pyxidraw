from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect
from .registry import effect


@njit(fastmath=True, cache=True)
def _calculate_line_normals_3d(vertices: np.ndarray) -> np.ndarray:
    """3D線分のXY平面での法線ベクトルを計算。"""
    if vertices.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)

    # Direction vectors in XY plane
    directions = vertices[1:] - vertices[:-1]

    # Normals in XY plane (Z component is 0)
    normals = np.zeros_like(directions, dtype=np.float32)
    normals[:, 0] = -directions[:, 1]  # x = -offset_y
    normals[:, 1] = directions[:, 0]  # y = offset_x
    normals[:, 2] = 0  # z = 0

    # Normalize
    lengths = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)
    lengths = np.where(lengths == 0, 1, lengths)
    normals[:, 0] /= lengths
    normals[:, 1] /= lengths

    return normals


@njit(fastmath=True, cache=True)
def _boldify_coords_with_offsets(
    coords: np.ndarray,
    offsets: np.ndarray,
    boldness: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """法線ベースの効率的な太線実装。
    
    各線分の法線ベクトルを計算し、元の線の両側に平行線を生成して太線効果を実現。
    
    処理の流れ:
    1. offsetsを使って各ポリラインを識別
    2. 各ポリラインについて法線ベクトル(垂直方向)を計算
    3. 各頂点で法線方向に太さの半分だけオフセットした左右の平行線を生成
    4. 元の線と左右の平行線を結果として返す
    
    Args:
        coords: 3D座標配列 (N, 3)
        offsets: オフセット配列 (M,)
        boldness: 太さ（単位：ミリメートル相当）
    
    Returns:
        (new_coords, new_offsets): 元の線 + 左の平行線 + 右の平行線を含む座標とオフセット
    """
    if boldness <= 0:
        return coords.copy(), offsets.copy()

    if len(coords) == 0:
        return coords.copy(), offsets.copy()

    half_boldness = boldness / 2
    
    # 結果を格納するリスト
    all_coords = []
    all_offsets = []
    
    # offsetsからポリラインを抽出
    start_idx = 0
    for end_idx in offsets:
        if start_idx >= end_idx:
            start_idx = end_idx
            continue
            
        vertices = coords[start_idx:end_idx]
        
        if vertices.shape[0] < 2:
            # 元のラインを追加
            all_coords.append(vertices)
            all_offsets.append(np.array([end_idx - start_idx], dtype=offsets.dtype))
            start_idx = end_idx
            continue

        # 元のラインを追加
        all_coords.append(vertices)
        all_offsets.append(np.array([vertices.shape[0]], dtype=offsets.dtype))

        # Calculate normals for 3D vertices
        normals = _calculate_line_normals_3d(vertices)

        if normals.shape[0] == 0:
            start_idx = end_idx
            continue

        # Calculate per-vertex normals
        vertex_normals = np.zeros_like(vertices, dtype=np.float32)

        # First vertex
        vertex_normals[0] = normals[0]

        # Middle vertices (average of adjacent segment normals)
        for i in range(1, vertices.shape[0] - 1):
            vertex_normals[i] = (normals[i - 1] + normals[i]) / 2
            # Re-normalize
            length = np.sqrt(vertex_normals[i, 0] ** 2 + vertex_normals[i, 1] ** 2)
            if length > 0:
                vertex_normals[i] /= length

        # Last vertex
        vertex_normals[-1] = normals[-1]

        # Generate left and right parallel lines
        left_line = vertices + vertex_normals * half_boldness
        right_line = vertices - vertex_normals * half_boldness

        all_coords.append(left_line.astype(vertices.dtype))
        all_coords.append(right_line.astype(vertices.dtype))
        all_offsets.append(np.array([left_line.shape[0]], dtype=offsets.dtype))
        all_offsets.append(np.array([right_line.shape[0]], dtype=offsets.dtype))
        
        start_idx = end_idx

    # すべての座標とオフセットを結合
    if len(all_coords) == 0:
        return coords.copy(), offsets.copy()
        
    combined_coords = np.vstack(all_coords)
    combined_offsets = np.cumsum(np.concatenate(all_offsets))

    return combined_coords, combined_offsets


@effect
class Boldify(BaseEffect):
    """平行線を追加して線を太く見せるエフェクト。
    
    ベクトルグラフィックスにおいて、単一の線分を太く見せるために
    元の線の両側に平行線を追加。ペンプロッターでの描画において
    実際の線の太さを制御できない場合に特に有効。
    
    固定太さの平行線を両側に1本ずつ追加（計3本の線）。
    """

    BOLDNESS_COEF = 0.6  # 太さ係数のデフォルト値

    def apply(
        self,
        coords: np.ndarray,
        offsets: np.ndarray,
        boldness: float = 0.5,
        **_: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """太線化エフェクトを適用。

        線分の両側に平行線を追加することで太線効果を実現。
        固定太さの平行線を両側に1本ずつ追加し、計3本の線で太線効果を実現。

        Args:
            coords: 3D座標配列 (N, 3)
            offsets: オフセット配列 (M,)
            boldness: 太さ係数（0.0-1.0）。BOLDNESS_COEFで内部的にスケーリング

        Returns:
            (boldified_coords, boldified_offsets): 太線化された座標とオフセット。
            元の線 + 平行線が含まれる
        """
        if boldness <= 0:
            return coords.copy(), offsets.copy()

        boldness = boldness * self.BOLDNESS_COEF
        return _boldify_coords_with_offsets(coords, offsets, boldness)
