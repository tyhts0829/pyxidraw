from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseEffect
from .registry import effect


@effect
class Extrude(BaseEffect):
    """2D/3Dポリラインを指定方向に押し出し、側面エッジを生成します。

    BaseEffect 標準の coords/offsets 方式で動作します。
    入力の各ポリラインに対し、同形状を押し出して接続エッジを追加します。
    """

    # クラス定数（0.0-1.0のレンジから実際の値へのスケーリング）
    MAX_DISTANCE = 200.0  # 最大押し出し距離
    MAX_SCALE = 3.0  # 押し出し側のスケール倍率（0.0-1.0 -> 0..3）
    MAX_SUBDIVISIONS = 5  # 細分化ステップ数

    def apply(
        self,
        coords: np.ndarray,
        offsets: np.ndarray,
        direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
        distance: float = 0.5,
        scale: float = 0.5,
        subdivisions: float = 0.5,
        **params: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """押し出しエフェクトを適用します。

        Args:
            coords: 入力座標配列 (N, 3)
            offsets: 入力オフセット配列 (L), 各ポリラインの開始インデックス
            direction: 押し出し方向ベクトル (x, y, z)
            distance: 押し出し距離係数 (0.0-1.0)
            scale: 押し出し側のスケール係数 (0.0-1.0)
            subdivisions: 細分化係数 (0.0-1.0)

        Returns:
            (new_coords, new_offsets): 元ライン+押し出しライン+接続エッジを含む配列
        """
        # エッジケース
        if coords.size == 0 or offsets.size == 0 or len(offsets) < 2:
            return coords.copy(), offsets.copy()

        # パラメータスケーリング
        distance_scaled = float(distance) * self.MAX_DISTANCE
        scale_scaled = float(scale) * self.MAX_SCALE
        subdivisions_int = int(float(subdivisions) * self.MAX_SUBDIVISIONS)

        # 方向ベクトル正規化
        direction_vec = np.asarray(direction, dtype=np.float32)
        norm = np.linalg.norm(direction_vec)
        if norm == 0.0:
            return coords.copy(), offsets.copy()
        extrude_vec = (direction_vec / norm) * np.float32(distance_scaled)

        # 入力ラインを抽出
        lines: list[np.ndarray] = []
        for i in range(len(offsets) - 1):
            start, end = int(offsets[i]), int(offsets[i + 1])
            line = coords[start:end]
            if len(line) >= 2:
                lines.append(line.astype(np.float32, copy=False))

        # 細分化
        if subdivisions_int > 0:
            subdivided: list[np.ndarray] = []
            for line in lines:
                current = line
                for _ in range(subdivisions_int):
                    if len(current) < 2:
                        break
                    new_vertices = [current[0]]
                    for j in range(len(current) - 1):
                        mid = (current[j] + current[j + 1]) / 2
                        new_vertices.append(mid)
                        new_vertices.append(current[j + 1])
                    current = np.asarray(new_vertices, dtype=np.float32)
                subdivided.append(current)
            lines = subdivided

        # 元のライン + 押し出しライン + 接続エッジを構築
        out_lines: list[np.ndarray] = []
        out_lines.extend(lines)

        for line in lines:
            extruded_line = (line + extrude_vec) * np.float32(scale_scaled)
            out_lines.append(extruded_line.astype(np.float32, copy=False))
            # 各頂点を結ぶエッジ（2点ポリラインとして追加）
            for j in range(len(line)):
                seg = np.asarray([line[j], extruded_line[j]], dtype=np.float32)
                out_lines.append(seg)

        # 結果を coords/offsets に結合
        if not out_lines:
            return coords.copy(), offsets.copy()

        new_coords = np.vstack(out_lines).astype(np.float32, copy=False)
        new_offsets = [0]
        acc = 0
        for ln in out_lines:
            acc += len(ln)
            new_offsets.append(acc)
        new_offsets = np.asarray(new_offsets, dtype=np.int32)

        return new_coords, new_offsets
