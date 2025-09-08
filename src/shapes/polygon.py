from __future__ import annotations

import math
from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .base import BaseShape
from .registry import shape


def _polygon_cached(n_sides: int) -> np.ndarray:
    """多角形の頂点配列を生成します。

    引数:
        n_sides: 辺の数。

    返り値:
        頂点配列（最初の頂点を末尾に複製して閉ループ化）。
    """
    # 頂点座標を計算
    t = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    x = np.cos(t) * 0.5
    y = np.sin(t) * 0.5
    z = np.zeros_like(x)
    vertices = np.stack([x, y, z], axis=1).astype(np.float32)

    # 最初の頂点を末尾に追加して閉ループにする
    vertices = np.append(vertices, vertices[0:1], axis=0)

    return vertices


@shape
class Polygon(BaseShape):
    """正多角形のシェイプを生成します。"""

    def generate(self, n_sides: int | float = 3, **params: Any) -> Geometry:
        """直径 1 の円に内接する正多角形を生成します。

        引数:
            n_sides: 辺の数。float の場合は 0–1 を指数写像で 3–100 に変換。
            **params: 追加パラメータ（未使用）

        返り値:
            多角形の頂点を含む Geometry
        """
        MIN_SIDES = 3
        MAX_SIDES = 100 - MIN_SIDES

        if isinstance(n_sides, float):
            n_sides = self._nonlinear_map_exp(n_sides, MAX_SIDES)
            n_sides += MIN_SIDES
        elif isinstance(n_sides, int):
            if n_sides < MIN_SIDES:
                n_sides = MIN_SIDES

        # 補助関数で頂点配列を生成
        vertices = _polygon_cached(n_sides)

        return Geometry.from_lines([vertices])

    @staticmethod
    def _nonlinear_map_exp(value: float, N: int, a: float = 100.0) -> int:
        """0.0–1.0 を指数関数で 0–N に非線形マッピングします。

        引数:
            value: 入力値（0.0–1.0 の範囲）。
            N: 出力範囲の最大値（整数）。
            a: 非線形度（2.0 以上で急峻、1.0 に近いほど線形に近い）。

        返り値:
            マッピングされた整数値。
        """
        if not (0.0 <= value <= 1.0):
            raise ValueError("value は 0.0–1.0 の範囲で指定してください")
        if N <= 0:
            raise ValueError("N は正の整数である必要があります")
        if a <= 1.0:
            raise ValueError("a は 1.0 より大きい値を指定してください")

        # 指数関数による非線形マッピング
        normalized_value = (math.pow(a, value) - 1) / (a - 1)
        mapped_value = normalized_value * N

        return int(round(mapped_value))
