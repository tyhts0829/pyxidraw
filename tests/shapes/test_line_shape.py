from __future__ import annotations

import numpy as np

from api import G
from engine.core.geometry import Geometry
from shapes.registry import is_shape_registered


def test_line_registered_and_generates_geometry() -> None:
    assert is_shape_registered("line")
    lazy = G.line()
    g = lazy.realize()
    assert isinstance(g, Geometry)


def test_line_default_is_unit_segment_centered() -> None:
    g = G.line()
    coords, offsets = g.as_arrays(copy=False)

    # 単一ポリラインで 2 頂点
    assert offsets.tolist() == [0, 2]

    p0, p1 = coords
    # 原点対称
    np.testing.assert_allclose(p0 + p1, np.zeros(3, dtype=np.float32), atol=1e-6)
    # 長さ 1
    length = np.linalg.norm(p1 - p0)
    assert np.isclose(length, 1.0, atol=1e-6)


def test_line_zero_length_produces_empty_geometry() -> None:
    g = G.line(length=0.0)
    coords, offsets = g.as_arrays(copy=False)
    assert coords.shape == (0, 3)
    assert offsets.tolist() == [0]


def test_line_angle_rotates_segment() -> None:
    g = G.line(length=1.0, angle=90.0)
    coords, offsets = g.as_arrays(copy=False)

    assert offsets.tolist() == [0, 2]
    x_coords = coords[:, 0]
    # 90 度回転で x=0 近傍の垂直線になる
    np.testing.assert_allclose(x_coords, np.zeros_like(x_coords), atol=1e-6)
