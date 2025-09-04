import numpy as np
import pytest

from engine.core.geometry import Geometry
from effects.filling import fill
from effects.subdivision import subdivide
from effects.buffer import offset as buffer_effect


def _square(size=1.0):
    s = float(size) / 2.0
    # closed polygon
    pts = np.array([[ -s, -s, 0], [ s, -s, 0], [ s, s, 0], [ -s, s, 0], [ -s, -s, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


class TestBoundaryValues:
    def test_filling_density_zero_is_identity(self):
        g = _square()
        out = fill(g, density=0.0)
        c0, o0 = g.as_arrays()
        c1, o1 = out.as_arrays()
        np.testing.assert_allclose(c0, c1)
        np.testing.assert_array_equal(o0, o1)

    def test_filling_density_one_adds_elements(self):
        g = _square()
        out = fill(g, density=1.0, mode="lines")
        c0, _ = g.as_arrays()
        c1, _ = out.as_arrays()
        assert c1.shape[0] >= c0.shape[0]

    def test_subdivision_zero_is_identity(self):
        g = _square()
        out = subdivide(g, subdivisions=0.0)
        c0, o0 = g.as_arrays()
        c1, o1 = out.as_arrays()
        np.testing.assert_allclose(c0, c1)
        np.testing.assert_array_equal(o0, o1)

    def test_subdivision_increases_points(self):
        g = _square()
        out = subdivide(g, subdivisions=0.5)
        c0, _ = g.as_arrays()
        c1, _ = out.as_arrays()
        assert c1.shape[0] > c0.shape[0]

    def test_buffer_distance_zero_is_identity(self):
        g = _square()
        out = buffer_effect(g, distance=0.0)
        c0, o0 = g.as_arrays()
        c1, o1 = out.as_arrays()
        np.testing.assert_allclose(c0, c1)
        np.testing.assert_array_equal(o0, o1)

    @pytest.mark.parametrize("join_style", ["mitre", "round", "bevel"])
    def test_buffer_join_styles_no_exception(self, join_style):
        g = _square()
        _ = buffer_effect(g, distance=0.3, join=join_style)
