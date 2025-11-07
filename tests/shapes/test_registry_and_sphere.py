from __future__ import annotations

from engine.core.geometry import Geometry
from shapes.registry import get_shape, is_shape_registered


def test_sphere_registered_and_generates_geometry() -> None:
    assert is_shape_registered("sphere")
    fn = get_shape("sphere")
    g = fn(subdivisions=0.0, sphere_type=0).realize()
    assert isinstance(g, Geometry)
    assert g.offsets[0] == 0
