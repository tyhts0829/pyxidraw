from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

from .base import BaseEffect
from .subdivision import _subdivide_core


class Collapse(BaseEffect):
    """線分を細分化してノイズで変形するエフェクト。"""

    def apply(
        self, vertices_list: list[np.ndarray], intensity: float = 0.5, n_divisions: float = 0.5, **params: Any
    ) -> list[np.ndarray]:
        """崩壊エフェクトを適用します。

        線分を細分化し、始点-終点方向と直交する方向にノイズを加えて変形します。

        Args:
            vertices_list: 入力頂点配列
            intensity: ノイズの強さ (デフォルト 0.5)
            n_divisions: 細分化の度合い (デフォルト 0.5)
            **params: 追加パラメータ

        Returns:
            変形された頂点配列
        """
        # 空リストの場合は早期リターン
        if not vertices_list:
            return []

        # Numba最適化された関数を呼び出し
        return _apply_collapse_numba(vertices_list, intensity, n_divisions)


@njit
def _apply_collapse_numba(vertices_list, intensity, n_divisions):
    """Numba最適化された崩壊エフェクト処理"""
    new_vertices_list = []

    if intensity == 0.0 or n_divisions == 0.0:
        # 同じ型の新しいリストを返す
        for i in range(len(vertices_list)):
            new_vertices_list.append(vertices_list[i])
        return new_vertices_list

    np.random.seed(0)
    divisions = max(1, int(n_divisions * 10))

    for i in range(len(vertices_list)):
        vertices = vertices_list[i]
        if len(vertices) < 2:
            new_vertices_list.append(vertices)
            continue

        # ラインを細分化
        subdivided = _subdivide_core(vertices, divisions)

        # 細分化したラインのペアを処理
        for j in range(len(subdivided) - 1):
            subdivided_vertices = subdivided[j : j + 2]

            # メイン方向を求める
            main_dir = subdivided_vertices[1] - subdivided_vertices[0]
            main_norm = np.linalg.norm(main_dir)
            if main_norm < 1e-12:
                new_vertices_list.append(subdivided_vertices)
                continue
            norm_main_dir = main_dir / main_norm

            # ノイズベクトルを求める
            noise_vector = np.random.randn(3) / 5.0

            # ノイズをメイン方向と直交する方向に変換
            ortho_dir = np.cross(norm_main_dir, noise_vector)
            ortho_norm = np.linalg.norm(ortho_dir)
            if ortho_norm < 1e-12:
                new_vertices_list.append(subdivided_vertices)
                continue
            ortho_dir = ortho_dir / ortho_norm

            # ノイズを加える（元の型を維持）
            noise = (ortho_dir * intensity).astype(vertices.dtype)
            offseted_vertices = subdivided_vertices + noise
            new_vertices_list.append(offseted_vertices)

    return new_vertices_list
