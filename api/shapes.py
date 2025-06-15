from __future__ import annotations

from typing import Any

import numpy as np

from shapes import ShapeFactory

# Global shape factory instance
_factory = ShapeFactory()


def polygon(
    n_sides: int | float = 3,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a regular polygon.

    Args:
        n_sides: Number of sides. If float, exponentially mapped from 0-100.
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List containing a single array of vertices
    """
    shape = _factory.create("polygon")
    return shape(n_sides=n_sides, center=center, scale=scale, rotate=rotate, **params)


def sphere(
    subdivisions: float = 0.5,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a sphere.

    Args:
        subdivisions: Subdivision level (0.0-1.0, mapped to 0-5)
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for sphere triangles
    """
    shape = _factory.create("sphere")
    return shape(subdivisions=subdivisions, center=center, scale=scale, rotate=rotate, **params)


def grid(
    n_divisions: tuple[float, float] = (0.1, 0.1),
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a grid.

    Args:
        n_divisions: (x_divisions, y_divisions) as floats 0.0-1.0
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for grid lines
    """
    shape = _factory.create("grid")
    return shape(n_divisions=n_divisions, center=center, scale=scale, rotate=rotate, **params)


def polyhedron(
    polygon_type: str | int = "tetrahedron",
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a regular polyhedron.

    Args:
        polygon_type: Type of polyhedron (name or number of faces)
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for polyhedron edges
    """
    shape = _factory.create("polyhedron")
    return shape(polygon_type=polygon_type, center=center, scale=scale, rotate=rotate, **params)


def lissajous(
    freq_x: float = 3.0,
    freq_y: float = 2.0,
    phase: float = 0.0,
    points: int = 1000,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a Lissajous curve.

    Args:
        freq_x: X-axis frequency
        freq_y: Y-axis frequency
        phase: Phase offset in radians
        points: Number of points to generate
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List containing a single array of vertices
    """
    shape = _factory.create("lissajous")
    return shape(
        freq_x=freq_x, freq_y=freq_y, phase=phase, points=points, center=center, scale=scale, rotate=rotate, **params
    )


def torus(
    major_radius: float = 0.3,
    minor_radius: float = 0.1,
    major_segments: int = 32,
    minor_segments: int = 16,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a torus.

    Args:
        major_radius: Major radius (from center to tube center)
        minor_radius: Minor radius (tube radius)
        major_segments: Number of segments around major circle
        minor_segments: Number of segments around minor circle
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for torus lines
    """
    shape = _factory.create("torus")
    return shape(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_segments=major_segments,
        minor_segments=minor_segments,
        center=center,
        scale=scale,
        rotate=rotate,
        **params,
    )


def cylinder(
    radius: float = 0.3,
    height: float = 0.6,
    segments: int = 32,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a cylinder.

    Args:
        radius: Cylinder radius
        height: Cylinder height
        segments: Number of segments around circumference
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for cylinder lines
    """
    shape = _factory.create("cylinder")
    return shape(radius=radius, height=height, segments=segments, center=center, scale=scale, rotate=rotate, **params)


def cone(
    radius: float = 0.3,
    height: float = 0.6,
    segments: int = 32,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a cone.

    Args:
        radius: Base radius
        height: Cone height
        segments: Number of segments around circumference
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for cone lines
    """
    shape = _factory.create("cone")
    return shape(radius=radius, height=height, segments=segments, center=center, scale=scale, rotate=rotate, **params)


def capsule(
    radius: float = 0.2,
    height: float = 0.4,
    segments: int = 32,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a capsule shape.

    Args:
        radius: Radius of the hemispheres
        height: Height of the cylindrical section
        segments: Number of segments for curves
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for capsule lines
    """
    shape = _factory.create("capsule")
    return shape(radius=radius, height=height, segments=segments, center=center, scale=scale, rotate=rotate, **params)


def attractor(
    attractor_type: str = "lorenz",
    points: int = 10000,
    dt: float = 0.01,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate a strange attractor.

    Args:
        attractor_type: Type of attractor ("lorenz", "rossler", "chua")
        points: Number of points to generate
        dt: Time step for integration
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List containing a single array of vertices
    """
    shape = _factory.create("attractor")
    return shape(
        attractor_type=attractor_type, points=points, dt=dt, center=center, scale=scale, rotate=rotate, **params
    )


def text(
    text: str = "HELLO",
    size: float = 0.1,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate text as line segments.

    Args:
        text: Text string to render
        size: Text size
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for text outlines
    """
    shape = _factory.create("text")
    return shape(text=text, size=size, center=center, scale=scale, rotate=rotate, **params)


def asemic_glyph(
    complexity: int = 5,
    seed: int | None = None,
    center: tuple[float, float, float] = (0, 0, 0),
    scale: tuple[float, float, float] = (1, 1, 1),
    rotate: tuple[float, float, float] = (0, 0, 0),
    **params: Any,
) -> list[np.ndarray]:
    """Generate abstract glyph-like shapes.

    Args:
        complexity: Number of strokes (1-10)
        seed: Random seed for reproducibility
        center: Position offset (x, y, z)
        scale: Scale factors (x, y, z)
        rotate: Rotation angles in radians (x, y, z)
        **params: Additional parameters

    Returns:
        List of vertex arrays for glyph strokes
    """
    shape = _factory.create("asemic_glyph")
    return shape(complexity=complexity, seed=seed, center=center, scale=scale, rotate=rotate, **params)
