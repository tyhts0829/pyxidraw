from __future__ import annotations

from typing import Any

import numpy as np

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
            intensity: ノイズの強さ (デフォルト 0.1)
            n_divisions: 細分化の度合い (デフォルト 0.2)
            **params: 追加パラメータ

        Returns:
            変形された頂点配列
        """
        # 空リストの場合は早期リターン
        if not vertices_list:
            return []

        # Numbaの型推論を助けるため、通常のPython関数で実装
        return self._apply_collapse(vertices_list, intensity, n_divisions)

    def _apply_collapse(
        self, vertices_list: list[np.ndarray], intensity: float, n_divisions: float
    ) -> list[np.ndarray]:
        """内部実装：Numbaを使わずに処理"""
        if intensity == 0:
            return vertices_list.copy()
        if not n_divisions:
            return vertices_list.copy()

        np.random.seed(0)
        new_vertices_list = []

        # もし頂点列が１つしかない場合はそのまま返す
        for vertices in vertices_list:
            if len(vertices) < 2:
                new_vertices_list.append(vertices)
                continue

            # ラインを細分化
            # n_divisions is 0.0-1.0, convert to int divisions
            divisions = int(n_divisions * 10)  # 最大10回分割
            subdivided = _subdivide_core(vertices.astype(np.float64), divisions)

            # 細分化したラインのリストにする
            subdivided_vertices_list = [subdivided[i : i + 2] for i in range(len(subdivided) - 1)]
            for subdivided_vertices in subdivided_vertices_list:
                # メイン方向を求める（始点と終点から）
                main_dir = subdivided_vertices[-1] - subdivided_vertices[0]
                norm_main_dir = main_dir / (np.linalg.norm(main_dir) + 1e-12)
                # ノイズベクトルを求める
                noise_vector = np.random.randn(3) / 5
                # ノイズをメイン方向と直交する方向に変換
                ortho_dir = np.cross(norm_main_dir, noise_vector)
                ortho_dir = ortho_dir / (np.linalg.norm(ortho_dir) + 1e-12)
                # ノイズの強さを調整
                noise = ortho_dir * intensity
                # ノイズを加える
                offseted_vertices = subdivided_vertices + noise
                new_vertices_list.append(offseted_vertices)
        return new_vertices_list
