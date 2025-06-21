from __future__ import annotations

from typing import Any, List

import numpy as np
from numba import njit

from .base import BaseEffect


@njit(fastmath=True, cache=True)
def _calculate_line_normals_3d(vertices: np.ndarray) -> np.ndarray:
    """3D線分のXY平面での法線ベクトルを計算。"""
    if vertices.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)

    # Direction vectors in XY plane
    directions = vertices[1:] - vertices[:-1]

    # Normals in XY plane (Z component is 0)
    normals = np.zeros_like(directions, dtype=np.float32)
    normals[:, 0] = -directions[:, 1]  # x = -dy
    normals[:, 1] = directions[:, 0]  # y = dx
    normals[:, 2] = 0  # z = 0

    # Normalize
    lengths = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2)
    lengths = np.where(lengths == 0, 1, lengths)
    normals[:, 0] /= lengths
    normals[:, 1] /= lengths

    return normals


@njit(fastmath=True, cache=True)
def _boldify_normal_based(
    vertices_list: List[np.ndarray],
    boldness: float = 0.01,
) -> List[np.ndarray]:
    """法線ベースの効率的な太線実装。
    
    各線分の法線ベクトルを計算し、元の線の両側に平行線を生成して太線効果を実現。
    
    処理の流れ:
    1. 各線分の方向ベクトルから法線ベクトル(垂直方向)を計算
    2. 隣接する線分の法線を平均化して頂点での法線を算出
    3. 各頂点で法線方向に太さの半分だけオフセットした左右の平行線を生成
    4. 元の線と左右の平行線を結果として返す
    
    Args:
        vertices_list: 3D頂点配列のリスト
        boldness: 太さ（単位：ミリメートル相当）
    
    Returns:
        元の線 + 左の平行線 + 右の平行線を含む頂点配列リスト
    """
    if boldness <= 0:
        return vertices_list

    new_vertices_list = []
    half_boldness = boldness / 2

    for vertices in vertices_list:
        if vertices.shape[0] < 2:
            new_vertices_list.append(vertices)
            continue

        # Add original line
        new_vertices_list.append(vertices)

        # Calculate normals for 3D vertices
        normals = _calculate_line_normals_3d(vertices)

        if normals.shape[0] == 0:
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

        new_vertices_list.append(left_line.astype(vertices.dtype))
        new_vertices_list.append(right_line.astype(vertices.dtype))

    return new_vertices_list




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
        vertices_list: list[np.ndarray],
        boldness: float = 0.5,
        **_: Any,
    ) -> list[np.ndarray]:
        """太線化エフェクトを適用。

        線分の両側に平行線を追加することで太線効果を実現。
        固定太さの平行線を両側に1本ずつ追加し、計3本の線で太線効果を実現。

        Args:
            vertices_list: 3D頂点配列のリスト（各配列は(N, 3)の形状）
            boldness: 太さ係数（0.0-1.0）。BOLDNESS_COEFで内部的にスケーリング

        Returns:
            太線化された頂点配列のリスト。元の線 + 平行線が含まれる
        """
        if boldness <= 0:
            return vertices_list

        boldness = boldness * self.BOLDNESS_COEF
        return _boldify_normal_based(vertices_list, boldness)
