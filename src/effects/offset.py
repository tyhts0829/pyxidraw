"""
offset エフェクト（バッファ/輪郭オフセット）

- Shapely の `buffer` を用いて各ポリラインをオフセットし、膨張（外側）または収縮（内側）形状を生成する。
- 3D 入力は XY 平面に射影して処理後、元の姿勢に戻す。

主なパラメータ:
- distance: オフセット距離 [mm]。
- join: 角の処理（`round` | `mitre` | `bevel`）。
- segments_per_circle: 円弧近似分割数（大きいほど滑らかだが重い）。

仕様/注意:
- 入力曲線が未クローズで、始終点が閾値内にある場合は自動クローズしてから処理。
- 結果は 3D 姿勢に復元する。自己交差や極端な距離ではトポロジが変化し得る。
- 安全のため距離は [0, 25.0] に丸める。距離 0 は no-op（入力コピー）。

実装メモ（詳細設計）:
- XY 射影 → Shapely `buffer(distance, join_style, resolution)` → 3D 復元の順で処理。
- `LineString/MultiLineString` と `Polygon/MultiPolygon` の両方の出力形に対応し、外輪郭頂点列を抽出。
- 全体が膨張し過ぎないよう軽いスケーリング補正を適用。
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
) -> Geometry:
    """Shapely を用いて輪郭をオフセット。

    Parameters
    ----------
    g : Geometry
        入力ジオメトリ。各行が 1 本のポリラインを表す（`offsets` で区切る）。
    join : str, default 'round'
        角の処理。`'mitre'|'round'|'bevel'` を指定。
    segments_per_circle : int, default 12
        円弧近似の分割数（Shapely の resolution 相当）。
    distance : float, default 15.0
        オフセット距離（mm）。
    """
    coords, offsets = g.as_arrays(copy=False)
    MAX_DISTANCE = 25.0
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


# UI 表示のためのメタ情報（RangeHint 構築に使用）
offset.__param_meta__ = {
    "distance": {"type": "number", "min": 0.0, "max": 25.0},
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
