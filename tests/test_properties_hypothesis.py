import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis", reason="hypothesis is a dev optional dependency")
from hypothesis import given, strategies as st  # type: ignore

from engine.core.geometry import Geometry


def _geom():
    pts = np.array([[0, 0, 0], [1, 2, -1], [3, -4, 5]], dtype=np.float32)
    return Geometry.from_lines([pts])


@given(
    dx1=st.floats(-10, 10), dy1=st.floats(-10, 10), dz1=st.floats(-10, 10),
    dx2=st.floats(-10, 10), dy2=st.floats(-10, 10), dz2=st.floats(-10, 10),
)
def test_translate_composition(dx1, dy1, dz1, dx2, dy2, dz2):
    g = _geom()
    left = g.translate(dx1, dy1, dz1).translate(dx2, dy2, dz2)
    right = g.translate(dx1 + dx2, dy1 + dy2, dz1 + dz2)
    cL, oL = left.as_arrays()
    cR, oR = right.as_arrays()
    np.testing.assert_allclose(cL, cR, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(oL, oR)


def _geom_line(n=3):
    pts = np.stack([
        np.linspace(0, 1, n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
        np.zeros(n, dtype=np.float32),
    ], axis=1)
    return Geometry.from_lines([pts])


@given(n1=st.integers(2, 5), n2=st.integers(2, 5), n3=st.integers(2, 5))
def test_concat_associativity(n1, n2, n3):
    a = _geom_line(n1)
    b = _geom_line(n2)
    c = _geom_line(n3)
    left = a.concat(b).concat(c)
    right = a.concat(b.concat(c))
    cL, oL = left.as_arrays()
    cR, oR = right.as_arrays()
    np.testing.assert_allclose(cL, cR, rtol=1e-6)
    np.testing.assert_array_equal(oL, oR)
