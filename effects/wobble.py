from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect


def _wobble_vertices(
    vertices_list: list[np.ndarray], amplitude: float, frequency: tuple[float, float, float], phase: float
) -> list[np.ndarray]:
    """各頂点に対してサイン波によるゆらぎ（wobble）を加える内部関数。"""
    new_vertices_list = []
    for vertices in vertices_list:
        if len(vertices) == 0:
            new_vertices_list.append(vertices)
            continue

        new_vertices = vertices.astype(np.float32).copy()
        # ベクトル化された計算
        # x軸方向のゆらぎ
        new_vertices[:, 0] += amplitude * np.sin(2 * np.pi * frequency[0] * new_vertices[:, 0] + phase)
        # y軸方向のゆらぎ
        new_vertices[:, 1] += amplitude * np.sin(2 * np.pi * frequency[1] * new_vertices[:, 1] + phase)
        # z軸方向のゆらぎ（2D の場合は 0 のまま）
        if new_vertices.shape[1] > 2:
            new_vertices[:, 2] += amplitude * np.sin(2 * np.pi * frequency[2] * new_vertices[:, 2] + phase)
        new_vertices_list.append(new_vertices)
    return new_vertices_list


class Wobble(BaseEffect):
    """線にウォブル/波の歪みを追加します。

    各頂点の座標値に基づいてサイン波による変位を加えます。
    """

    AMPLITUDE_COEF = 5
    FREQUENCY_COEF = 0.05
    PHASE_COEF = 5

    def apply(
        self,
        vertices_list: list[np.ndarray],
        amplitude: float = 1.0,
        frequency: float | tuple[float, float, float] = (0.1, 0.1, 0.1),
        phase: float = 0.0,
        **params: Any,
    ) -> list[np.ndarray]:
        """ウォブルエフェクトを適用します。

        Args:
            vertices_list: 入力頂点配列
            amplitude: ゆらぎの大きさ
            frequency: 各軸 (x, y, z) に対する周波数。
                      float の場合は全軸に同じ値を使用。
                      tuple の場合は各軸に個別の周波数を指定。
            phase: 位相オフセット（全体に同じオフセットを付与）
            **params: 追加パラメータ（無視される）

        Returns:
            ウォブルが適用された頂点配列
        """
        # frequency を tuple に変換
        freq_tuple: tuple[float, float, float]
        if isinstance(frequency, (int, float)):
            frequency = self.FREQUENCY_COEF * frequency
            freq_tuple = (float(frequency), float(frequency), float(frequency))
        elif isinstance(frequency, (list, tuple)):
            if len(frequency) == 3:
                freq_tuple = (
                    float(frequency[0]) * self.FREQUENCY_COEF,
                    float(frequency[1]) * self.FREQUENCY_COEF,
                    float(frequency[2]) * self.FREQUENCY_COEF,
                )
            else:
                # 不正な長さの場合はデフォルト値を使用
                df = 0.5 * self.FREQUENCY_COEF
                freq_tuple = (df, df, df)
        else:
            df = 0.5 * self.FREQUENCY_COEF
            freq_tuple = (df, df, df)

        # 空のリストの場合は早期リターン
        if not vertices_list:
            return []

        # wobble 関数を呼び出し
        return _wobble_vertices(
            vertices_list, float(amplitude * self.AMPLITUDE_COEF), freq_tuple, float(phase * self.PHASE_COEF)
        )
