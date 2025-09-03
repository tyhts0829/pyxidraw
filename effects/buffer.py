from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

from util.geometry import transform_back, transform_to_xy_plane
from common.param_utils import norm_to_int, norm_to_range

from .registry import effect
from engine.core.geometry import Geometry


@effect()
def buffer(
    g: Geometry,
    *,
    distance: float = 0.5,
    join_style: float = 0.5,
    resolution: float = 0.5,
    **_params: Any,
) -> Geometry:
    """Shapely を使用したバッファ/オフセット（純関数）。"""
    coords, offsets = g.as_arrays(copy=False)
    # 0..1 → 実距離（mm）へ写像（提案5: 一貫写像）
    MAX_DISTANCE = 25.0
    actual_distance = norm_to_range(float(distance), 0.0, MAX_DISTANCE)
    if actual_distance == 0:
        return Geometry(coords.copy(), offsets.copy())

    join_style_str = _determine_join_style(float(join_style))
    resolution_int = max(1, norm_to_int(float(resolution), 1, 10))

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

def _buffer(vertices_list: list[np.ndarray], distance: float, join_style: str, resolution: int) -> list[np.ndarray]:
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
            new_vertices_list = _extract_vertices_from_line(new_vertices_list, buffered_line, rotation_matrix, z_offset)
        elif isinstance(buffered_line, (Polygon, MultiPolygon)):
            new_vertices_list = _extract_vertices_from_polygon(new_vertices_list, buffered_line, rotation_matrix, z_offset)

    scale_factor = 1 / (1 + distance * 2 / 25.0)
    new_vertices_list = _scaling(new_vertices_list, scale_factor)
    return new_vertices_list

def _extract_vertices_from_polygon(new_vertices_list: list, buffered_line: BaseGeometry, rotation_matrix: np.ndarray, z_offset: float) -> list:
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

def _extract_vertices_from_line(new_vertices_list: list, buffered_line: BaseGeometry, rotation_matrix: np.ndarray, z_offset: float) -> list:
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

def _determine_join_style(join_style: float) -> str:
    if 0.0 <= join_style < 0.33:
        return "mitre"
    elif 0.33 <= join_style < 0.67:
        return "round"
    elif 0.67 <= join_style <= 1.0:
        return "bevel"
    else:
        return "round"

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
