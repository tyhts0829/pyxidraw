from __future__ import annotations

import numpy as np
import pytest

from effects.collapse import collapse
from engine.core.geometry import Geometry


@pytest.mark.smoke
def test_collapse_intensity_zero_is_identity() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)])
    out = collapse(g, intensity=0.0, subdivisions=6.0)
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)


def test_collapse_subdivides_into_independent_segments() -> None:
    # 1 セグメント、subdivisions=3 → 3 本の独立ポリライン、各 2 頂点
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]], dtype=np.float32)])
    out = collapse(g, intensity=1.0, subdivisions=3.0)
    assert out.n_lines == 3
    assert np.array_equal(out.offsets, np.array([0, 2, 4, 6], dtype=np.int32))
    assert out.coords.shape == (6, 3)


def test_collapse_zero_length_segment_preserved() -> None:
    # ゼロ長セグメントは原状維持（2 頂点、1 本）
    g = Geometry.from_lines([np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)])
    out = collapse(g, intensity=2.0, subdivisions=4.0)
    assert out.n_lines == 1
    assert np.array_equal(out.offsets, np.array([0, 2], dtype=np.int32))
    assert np.allclose(out.coords[0], [1.0, 1.0, 0.0])
    assert np.allclose(out.coords[1], [1.0, 1.0, 0.0])


def test_collapse_single_point_preserved_as_one_line() -> None:
    g = Geometry.from_lines([np.array([[2.0, 2.0, 0.0]], dtype=np.float32)])
    out = collapse(g, intensity=1.0, subdivisions=3.0)
    assert out.n_lines == 1
    assert np.array_equal(out.offsets, np.array([0, 1], dtype=np.int32))
    assert np.allclose(out.coords, g.coords)
