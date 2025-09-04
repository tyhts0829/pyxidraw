import numpy as np

from engine.core.geometry import Geometry


def _geom():
    pts = np.array([[0, 0, 0], [1, 2, 3], [-1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_translate_inverse_is_identity():
    g = _geom()
    g2 = g.translate(1, -2, 3).translate(-1, 2, -3)
    c0, o0 = g.as_arrays()
    c2, o2 = g2.as_arrays()
    np.testing.assert_allclose(c0, c2)
    np.testing.assert_array_equal(o0, o2)


def test_rotate_zero_is_identity():
    g = _geom()
    g2 = g.rotate(0.0, 0.0, 0.0)
    c0, o0 = g.as_arrays()
    c2, o2 = g2.as_arrays()
    np.testing.assert_allclose(c0, c2)
    np.testing.assert_array_equal(o0, o2)


def test_scale_one_is_identity():
    g = _geom()
    g2 = g.scale(1.0)
    c0, o0 = g.as_arrays()
    c2, o2 = g2.as_arrays()
    np.testing.assert_allclose(c0, c2)
    np.testing.assert_array_equal(o0, o2)
