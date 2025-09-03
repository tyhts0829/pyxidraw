import math
import numpy as np

from engine.core.geometry import Geometry
from engine.core import transform_utils as tf


def make_g(points):
    return Geometry.from_lines([np.asarray(points, dtype=np.float32)])


def test_tf_translate_geometry():
    g = make_g([[0, 0, 0], [1, 2, 3]])
    g2 = tf.translate(g, 5, -1, 2)
    assert isinstance(g2, Geometry)
    c2, o2 = g2.as_arrays()
    assert o2.tolist() == [0, 2]
    assert np.allclose(c2[0], [5, -1, 2])


def test_tf_scale_rotate_geometry():
    g = make_g([[1, 0, 0]])
    g2 = tf.scale(g, 2.0, 2.0, 1.0)
    c2, _ = g2.as_arrays()
    assert np.allclose(c2[0], [2, 0, 0])

    g3 = tf.rotate_z(g2, math.pi / 2)
    c3, _ = g3.as_arrays()
    assert np.allclose(c3[0], [0, 2, 0], atol=1e-6)

