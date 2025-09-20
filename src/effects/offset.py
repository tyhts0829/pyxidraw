"""
offset エフェクト（バッファ/輪郭オフセット）

- Shapely の `buffer` を用いて各ポリラインをオフセットし、膨張（外側）または収縮（内側）形状を生成します。
- 3D 入力は XY 平面に射影して処理後、元の姿勢に戻します。

- 主なパラメータ:
- distance / distance_mm: オフセット距離。`distance` は直指定 [mm]、`distance_mm` は別名として同値。
- join: 角の処理（`round` | `mitre` | `bevel`）。
- segments_per_circle: 円弧近似分割数（大きいほど滑らかだが重い）。

実装メモ/注意:
- 入力曲線は必要に応じて自動クローズしてから処理し、結果を 3D 姿勢へ復元します。
- 全体が膨張し過ぎないよう軽い縮尺補正を適用します。
- 自己交差や極端な距離ではトポロジが変化する可能性があります。
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from engine.core.geometry import Geometry
from util.geom3d_ops import transform_back, transform_to_xy_plane

from .registry import effect


@effect()
def offset(
    g: Geometry,
    *,
    join: str = "round",  # 'mitre'|'round'|'bevel'
    segments_per_circle: int = 12,  # shapelyのresolutionに相当（既定値を上げて円滑さを確保）
    distance: float = 5.0,
    distance_mm: float | None = None,
) -> Geometry:
    """Shapely を使用したバッファ/オフセット（純関数）。

    既定値の方針（2025-09-06）:
        - distance=5mm、join='round', segments_per_circle=12。
          300mm 正方キャンバス中央の立方体に適用した静止画で、明瞭かつ過度でない見た目。
    """
    coords, offsets = g.as_arrays(copy=False)
    # 0..1 → 実距離（mm）へ写像（提案5: 一貫写像）
    MAX_DISTANCE = 25.0
    if distance_mm is not None:
        actual_distance = float(distance_mm)
    else:
        actual_distance = float(distance)
    if actual_distance < 0.0:
        actual_distance = 0.0
    if actual_distance > MAX_DISTANCE:
        actual_distance = MAX_DISTANCE
    if actual_distance == 0:
        return Geometry(coords.copy(), offsets.copy())

    join_style_str = join
    resolution_int = int(segments_per_circle)

    vertices_list = []
    for i in range(len(offsets) - 1):
        vertices = coords[offsets[i] : offsets[i + 1]]
        if len(vertices) >= 2:
            vertices_list.append(vertices)

    new_vertices_list = _buffer(vertices_list, actual_distance, join_style_str, resolution_int)

    filtered_vertices_list = []
    for v in new_vertices_list:
        if v is not None and len(v) > 0 and isinstance(v, np.ndarray):
            filtered_vertices_list.append(v.astype(np.float32))

    if not filtered_vertices_list:
        return Geometry(coords.copy(), offsets.copy())

    all_coords = np.vstack(filtered_vertices_list)
    new_offsets = [0]
    acc = 0
    for line in filtered_vertices_list:
        acc += len(line)
        new_offsets.append(acc)

    return Geometry(all_coords, np.array(new_offsets, dtype=np.int32))


# validate_spec 用のメタデータ
offset.__param_meta__ = {
    "distance": {"type": "number", "min": 0.0, "max": 25.0},
    "distance_mm": {"type": "number", "min": 0.0},
    "join": {"type": "string", "choices": ["mitre", "round", "bevel"]},
    "segments_per_circle": {"type": "integer", "min": 1, "max": 1000},
}


def _buffer(
    vertices_list: list[np.ndarray], distance: float, join_style: str, resolution: int
) -> list[np.ndarray]:
    if distance == 0:
        return vertices_list

    new_vertices_list: list[np.ndarray] = []

    for vertices in vertices_list:
        vertices = _close_curve(vertices, 1e-3)
        vertices_on_xy, rotation_matrix, z_offset = transform_to_xy_plane(vertices)
        line = LineString(vertices_on_xy[:, :2])
        buffered_line = line.buffer(distance, join_style=join_style, resolution=resolution)  # type: ignore

        if buffered_line.is_empty:
            continue

        if isinstance(buffered_line, (LineString, MultiLineString)):
            new_vertices_list = _extract_vertices_from_line(
                new_vertices_list, buffered_line, rotation_matrix, z_offset
            )
        elif isinstance(buffered_line, (Polygon, MultiPolygon)):
            new_vertices_list = _extract_vertices_from_polygon(
                new_vertices_list, buffered_line, rotation_matrix, z_offset
            )

    scale_factor = 1 / (1 + distance * 2 / 25.0)
    new_vertices_list = _scaling(new_vertices_list, scale_factor)
    return new_vertices_list


def _extract_vertices_from_polygon(
    new_vertices_list: list,
    buffered_line: BaseGeometry,
    rotation_matrix: np.ndarray,
    z_offset: float,
) -> list:
    if isinstance(buffered_line, Polygon):
        polygons = [buffered_line]
    else:
        from shapely.geometry import MultiPolygon

        polygons = buffered_line.geoms if isinstance(buffered_line, MultiPolygon) else []

    for poly in polygons:
        coords = np.array(poly.exterior.coords)
        new_vertices = np.hstack([coords, np.zeros((len(coords), 1))])
        restored = transform_back(new_vertices, rotation_matrix, z_offset)
        new_vertices_list.append(restored)

    return new_vertices_list


def _extract_vertices_from_line(
    new_vertices_list: list,
    buffered_line: BaseGeometry,
    rotation_matrix: np.ndarray,
    z_offset: float,
) -> list:
    if isinstance(buffered_line, LineString):
        lines = [buffered_line]
    else:
        from shapely.geometry import MultiLineString

        lines = buffered_line.geoms if isinstance(buffered_line, MultiLineString) else []

    for line in lines:
        coords = np.array(line.coords)
        new_vertices = np.hstack([coords, np.zeros((len(coords), 1))])
        restored = transform_back(new_vertices, rotation_matrix, z_offset)
        new_vertices_list.append(restored)

    return new_vertices_list


def _close_curve(points: np.ndarray, threshold: float) -> np.ndarray:
    if len(points) < 2:
        return points
    start = points[0]
    end = points[-1]
    dist = np.linalg.norm(start - end)
    if dist <= threshold:
        points_copy = points[:-1]
        points = np.vstack([points_copy, start])
    return points


def _scaling(vertices_list: list[np.ndarray], scale_factor: float) -> list[np.ndarray]:
    scaled_vertices_list = []
    for vertices in vertices_list:
        if len(vertices) == 0:
            scaled_vertices_list.append(vertices)
            continue
        centroid = np.mean(vertices, axis=0)
        scaled_vertices = (vertices - centroid) * scale_factor + centroid
        scaled_vertices_list.append(scaled_vertices)
    return scaled_vertices_list
