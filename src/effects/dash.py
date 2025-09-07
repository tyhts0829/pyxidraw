"""
dash エフェクト（破線化）

- 各ポリラインを一定のダッシュ長とギャップ長で切り出し、破線の集合へ変換します。

主なパラメータ:
- dash_length: ダッシュ区間の長さ [mm]。
- gap_length: ギャップ区間の長さ [mm]。

仕様/注意:
- 端部は補間により部分ダッシュになり得ます。全長が短い場合は原線を保持します。
- 長さ単位は正規化値ではなく座標系の実寸（mm 相当）です。
"""

from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect


def _interpolate_segment(
    vertices: np.ndarray, cumulative_distances: np.ndarray, start_dist: float, end_dist: float
) -> np.ndarray:
    start_idx = np.searchsorted(cumulative_distances, start_dist)
    end_idx = np.searchsorted(cumulative_distances, end_dist)
    if start_idx >= len(vertices):
        return np.array([], dtype=np.float32).reshape(0, 3)
    if start_idx > 0 and start_dist > cumulative_distances[start_idx - 1]:
        t = (start_dist - cumulative_distances[start_idx - 1]) / (
            cumulative_distances[start_idx] - cumulative_distances[start_idx - 1]
        )
        start_point = vertices[start_idx - 1] + t * (vertices[start_idx] - vertices[start_idx - 1])
    else:
        start_point = vertices[start_idx]
    if end_idx > 0 and end_idx < len(vertices) and end_dist < cumulative_distances[end_idx]:
        t = (end_dist - cumulative_distances[end_idx - 1]) / (
            cumulative_distances[end_idx] - cumulative_distances[end_idx - 1]
        )
        end_point = vertices[end_idx - 1] + t * (vertices[end_idx] - vertices[end_idx - 1])
    else:
        end_point = vertices[min(end_idx, len(vertices) - 1)]
    if start_idx == end_idx:
        return np.array([start_point, end_point], dtype=np.float32)
    else:
        intermediate = vertices[start_idx:end_idx]
        if len(intermediate) == 0:
            return np.array([start_point, end_point], dtype=np.float32)
        return np.vstack(
            [start_point[np.newaxis, :], intermediate, end_point[np.newaxis, :]]
        ).astype(np.float32)


@effect()
def dash(
    g: Geometry,
    *,
    dash_length: float = 6.0,
    gap_length: float = 3.0,
) -> Geometry:
    """連続線を破線に変換（純関数）。

    備考:
        - dash_length/gap_length は座標単位（mm 相当）。0..1 正規化ではありません。
        - 線長に応じて端部のダッシュは補間されます（端は部分ダッシュになり得ます）。
        - 既定値（6mm/3mm）は 300mm キャンバス中央の立方体（辺=150mm）で視認性と密度のバランスが良好です。
    """
    coords, offsets = g.as_arrays(copy=False)
    if len(coords) == 0:
        return Geometry(coords.copy(), offsets.copy())
    lines: list[np.ndarray] = []
    pattern_length = dash_length + gap_length
    for i in range(len(offsets) - 1):
        vertices = coords[offsets[i] : offsets[i + 1]]
        if len(vertices) < 2:
            lines.append(vertices)
            continue
        segments = vertices[1:] - vertices[:-1]
        distances = np.sqrt(np.sum(segments**2, axis=1))
        cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
        total_length = cumulative_distances[-1]
        if total_length <= 0:
            lines.append(vertices)
            continue
        current_distance = 0.0
        while current_distance < total_length:
            start_distance = current_distance
            end_distance = min(current_distance + dash_length, total_length)
            dash_vertices = _interpolate_segment(
                vertices, cumulative_distances, start_distance, end_distance
            )
            if len(dash_vertices) > 1:
                lines.append(dash_vertices)
            current_distance += pattern_length
    if not lines:
        return Geometry(coords.copy(), offsets.copy())
    return Geometry.from_lines(lines)


dash.__param_meta__ = {
    "dash_length": {"type": "number", "min": 0.0},
    "gap_length": {"type": "number", "min": 0.0},
}
