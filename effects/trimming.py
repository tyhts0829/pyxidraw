from __future__ import annotations

from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .registry import effect


def _interpolate_at_distance(vertices: np.ndarray, distances: list[float], target_dist: float) -> np.ndarray | None:
    if target_dist <= 0:
        return vertices[0]
    if target_dist >= distances[-1]:
        return vertices[-1]
    for i in range(len(distances) - 1):
        if distances[i] <= target_dist <= distances[i + 1]:
            segment_start = distances[i]
            segment_end = distances[i + 1]
            segment_length = segment_end - segment_start
            if segment_length == 0:
                return vertices[i]
            t = (target_dist - segment_start) / segment_length
            return vertices[i] + t * (vertices[i + 1] - vertices[i])
    return None


def _trim_path(vertices: np.ndarray, start_param: float, end_param: float) -> np.ndarray | None:
    if len(vertices) < 2:
        return vertices
    distances = [0.0]
    for i in range(len(vertices) - 1):
        dist = np.linalg.norm(vertices[i + 1] - vertices[i])
        distances.append(distances[-1] + dist)
    total_length = distances[-1]
    if total_length == 0:
        return vertices
    start_dist = start_param * total_length
    end_dist = end_param * total_length
    trimmed_vertices = []
    start_point = _interpolate_at_distance(vertices, distances, start_dist)
    if start_point is not None:
        trimmed_vertices.append(start_point)
    for i, dist in enumerate(distances):
        if start_dist < dist < end_dist:
            trimmed_vertices.append(vertices[i])
    end_point = _interpolate_at_distance(vertices, distances, end_dist)
    if end_point is not None and (not trimmed_vertices or not np.allclose(trimmed_vertices[-1], end_point)):
        trimmed_vertices.append(end_point)
    return np.array(trimmed_vertices) if len(trimmed_vertices) >= 2 else None


@effect()
def trimming(g: Geometry, *, start_param: float = 0.0, end_param: float = 1.0, **_params: Any) -> Geometry:
    start_param = max(0.0, min(1.0, float(start_param)))
    end_param = max(0.0, min(1.0, float(end_param)))
    coords, offsets = g.as_arrays(copy=False)
    if start_param >= end_param or len(coords) == 0:
        return Geometry(coords.copy(), offsets.copy())
    results: list[np.ndarray] = []
    for i in range(len(offsets) - 1):
        line = coords[offsets[i] : offsets[i + 1]]
        if len(line) < 2:
            results.append(line)
            continue
        trimmed = _trim_path(line, start_param, end_param)
        if trimmed is not None and len(trimmed) >= 2:
            results.append(trimmed)
    if not results:
        return Geometry(coords.copy(), offsets.copy())
    return Geometry.from_lines(results)
