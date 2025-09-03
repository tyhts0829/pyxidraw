from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from engine.core.geometry import Geometry

from .base import BaseShape
from .registry import shape


@lru_cache(maxsize=128)
def _sphere_latlon(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere wireframe using longitude/latitude lines only.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere wireframe
    """
    segment_count = 16 + 32 * subdivisions  # Number of segments per ring
    ring_count = segment_count // 2  # Number of rings (latitude lines)

    vertices_list = []

    # Longitude lines
    for j in range(segment_count):
        lon = 2 * np.pi * j / segment_count
        line = []
        for i in range(ring_count + 1):
            lat = np.pi * i / ring_count
            x = np.sin(lat) * np.cos(lon) * 0.5
            y = np.sin(lat) * np.sin(lon) * 0.5
            z = np.cos(lat) * 0.5
            line.append([x, y, z])
        vertices_list.append(np.array(line, dtype=np.float32))

    # Latitude lines
    for i in range(1, ring_count):  # Skip poles
        lat = np.pi * i / ring_count
        line = []
        for j in range(segment_count + 1):
            lon = 2 * np.pi * j / segment_count
            x = np.sin(lat) * np.cos(lon) * 0.5
            y = np.sin(lat) * np.sin(lon) * 0.5
            z = np.cos(lat) * 0.5
            line.append([x, y, z])
        vertices_list.append(np.array(line, dtype=np.float32))

    return vertices_list


@lru_cache(maxsize=128)
def _sphere_zigzag(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere using zigzag pattern.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere spiral
    """
    # 螺旋の総回転数
    total_rotations = 8 + 4 * subdivisions
    # 1周あたりの点数
    # points_per_rotation = 16 + 8 * subdivisions
    points_per_rotation = 100
    # 総点数
    points = int(total_rotations * points_per_rotation)

    vertices = []

    for i in range(points):
        # Parametric sphere using controlled spiral
        y = 1 - (i / float(points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)

        # 制御された螺旋角度（1周あたりの点数で制御）
        theta = (2 * np.pi * total_rotations * i) / points

        x = np.cos(theta) * radius * 0.5
        z = np.sin(theta) * radius * 0.5
        y = y * 0.5

        vertices.append([x, y, z])

    # Convert to line segments for smooth drawing
    vertices_list = []
    vertices_array = np.array(vertices, dtype=np.float32)

    # Create short line segments between consecutive points
    for i in range(len(vertices_array) - 1):
        line_segment = np.array([vertices_array[i], vertices_array[i + 1]], dtype=np.float32)
        vertices_list.append(line_segment)

    return vertices_list


@lru_cache(maxsize=128)
def _sphere_icosphere(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere using icosphere pattern with hierarchical subdivision.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere icosphere
    """
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio

    # 12 vertices of an icosahedron
    base_vertices = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float32,
    )

    # Normalize vertices to unit sphere
    norms = np.linalg.norm(base_vertices, axis=1, keepdims=True)
    base_vertices = base_vertices / norms * 0.5

    # Base triangular faces for icosahedron
    base_faces = [
        # Top cap triangles
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        # Bottom cap triangles
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        # Middle band triangles
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (9, 5, 4),
        (4, 11, 2),
        (2, 10, 6),
        (6, 7, 8),
        (8, 1, 9),
    ]

    def subdivide_triangle(v1, v2, v3, level):
        """Recursively subdivide a triangle into smaller triangles."""
        if level <= 0:
            # Base case: return the triangle edges
            return [(v1, v2), (v2, v3), (v3, v1)]

        # Calculate midpoints and project to sphere surface
        def midpoint_on_sphere(p1, p2):
            mid = (p1 + p2) / 2
            norm = np.linalg.norm(mid)
            return mid / norm * 0.5  # Project to sphere radius 0.5

        m1 = midpoint_on_sphere(v1, v2)
        m2 = midpoint_on_sphere(v2, v3)
        m3 = midpoint_on_sphere(v3, v1)

        # Recursively subdivide the 4 new triangles
        edges = []
        edges.extend(subdivide_triangle(v1, m1, m3, level - 1))
        edges.extend(subdivide_triangle(m1, v2, m2, level - 1))
        edges.extend(subdivide_triangle(m3, m2, v3, level - 1))
        edges.extend(subdivide_triangle(m1, m2, m3, level - 1))

        return edges

    # Generate all edges with subdivision
    all_edges = []
    for face in base_faces:
        v1, v2, v3 = base_vertices[face[0]], base_vertices[face[1]], base_vertices[face[2]]
        edges = subdivide_triangle(v1, v2, v3, subdivisions)
        all_edges.extend(edges)

    # Convert edges to vertex arrays, removing duplicates
    vertices_list = []
    seen_edges = set()

    for edge in all_edges:
        # Create a hashable representation of the edge
        edge_key = tuple(sorted([tuple(edge[0]), tuple(edge[1])]))

        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            line = np.array([edge[0], edge[1]], dtype=np.float32)
            vertices_list.append(line)

    return vertices_list


@lru_cache(maxsize=128)
def _sphere_rings(subdivisions: int) -> list[np.ndarray]:
    """Generate sphere using horizontal ring slices.

    Args:
        subdivisions: Subdivision level (0-5)

    Returns:
        List of vertex arrays for sphere rings
    """
    ring_count = 5 + 12 * subdivisions  # Number of slices per axis
    segment_count = 64  # Points per ring

    vertices_list = []

    # Create horizontal rings at different heights
    for i in range(ring_count):
        # Height from -0.5 to 0.5
        y = -0.5 + (i / (ring_count - 1))

        # Radius at this height (sphere equation: x² + y² + z² = r²)
        if abs(y) <= 0.5:
            radius = np.sqrt(0.25 - y * y)  # radius of circle at height y

            # Generate circle points
            ring_points = []
            for j in range(segment_count + 1):  # +1 to close the circle
                angle = 2 * np.pi * j / segment_count
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                ring_points.append([x, y, z])

            vertices_list.append(np.array(ring_points, dtype=np.float32))

    # Create rings perpendicular to X-axis (slicing along YZ plane)
    for i in range(ring_count):
        # X position from -0.5 to 0.5
        x = -0.5 + (i / (ring_count - 1))

        # Radius at this X position
        if abs(x) <= 0.5:
            radius = np.sqrt(0.25 - x * x)

            # Generate circle points in YZ plane
            ring_points = []
            for j in range(segment_count + 1):
                angle = 2 * np.pi * j / segment_count
                y = radius * np.cos(angle)
                z = radius * np.sin(angle)
                ring_points.append([x, y, z])

            vertices_list.append(np.array(ring_points, dtype=np.float32))

    # Create rings perpendicular to Z-axis (slicing along XY plane)
    for i in range(ring_count):
        # Z position from -0.5 to 0.5
        z = -0.5 + (i / (ring_count - 1))

        # Radius at this Z position
        if abs(z) <= 0.5:
            radius = np.sqrt(0.25 - z * z)

            # Generate circle points in XY plane
            ring_points = []
            for j in range(segment_count + 1):
                angle = 2 * np.pi * j / segment_count
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                ring_points.append([x, y, z])

            vertices_list.append(np.array(ring_points, dtype=np.float32))

    return vertices_list


@shape
class Sphere(BaseShape):
    """Sphere shape generator with multiple drawing styles."""

    def generate(self, subdivisions: float = 0.5, sphere_type: float = 0.5, **_params: Any) -> Geometry:
        """Generate a sphere with radius 1.

        Args:
            subdivisions: Subdivision level (0.0-1.0, mapped to 0-5)
            sphere_type: Drawing style (0.0-1.0):
                        0.0-0.2: Lat-Lon (default style)
                        0.2-0.4: Wireframe
                        0.4-0.6: Zigzag
                        0.6-0.8: Icosphere
                        0.8-1.0: Rings
            **params: Additional parameters (ignored)

        Returns:
            Geometry object containing sphere geometry
        """
        MIN_SUBDIVISIONS = 0
        MAX_SUBDIVISIONS = 5
        subdivisions_int = int(subdivisions * MAX_SUBDIVISIONS)
        if subdivisions_int < MIN_SUBDIVISIONS:
            subdivisions_int = MIN_SUBDIVISIONS
        if subdivisions_int > MAX_SUBDIVISIONS:
            subdivisions_int = MAX_SUBDIVISIONS

        # Select sphere generation method based on sphere_type
        if sphere_type < 0.2:
            vertices_list = _sphere_latlon(subdivisions_int)
        elif sphere_type < 0.4:
            vertices_list = _sphere_zigzag(subdivisions_int)
        elif sphere_type < 0.6:
            vertices_list = _sphere_icosphere(subdivisions_int)
        elif sphere_type < 0.8:
            vertices_list = _sphere_rings(subdivisions_int)
        else:
            vertices_list = _sphere_latlon(subdivisions_int)

        return Geometry.from_lines(vertices_list)
