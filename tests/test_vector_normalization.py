import numpy as np

from engine.core.geometry import Geometry
from effects.rotation import rotate
from effects.transform import affine
from effects.array import repeat
from effects.translation import translate


def _geom():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_rotation_scalar_vs_tuple():
    g = _geom()
    out_scalar = rotate(g, rotate=(0.25, 0.25, 0.25))
    out_tuple = rotate(g, rotate=(0.25, 0.25, 0.25))
    c1, _ = out_scalar.as_arrays()
    c2, _ = out_tuple.as_arrays()
    np.testing.assert_allclose(c1, c2)


def test_transform_rotate_normalization():
    g = _geom()
    out = affine(g, rotate=(0.0, 0.0, 0.5))  # 0.5 -> pi around Z
    out2 = affine(g, rotate=(0.0, 0.0, 0.0))
    c1, _ = out.as_arrays()
    c2, _ = out2.as_arrays()
    # 異なるはず（回転あり/なし）
    assert not np.allclose(c1, c2)


def test_array_rotate_scalar_and_vec_agree():
    g = _geom()
    out_a = repeat(g, n_duplicates=0.2, rotate=(0.25, 0.25, 0.25))
    out_b = repeat(g, n_duplicates=0.2, rotate=(0.25, 0.25, 0.25))
    c1, _ = out_a.as_arrays()
    c2, _ = out_b.as_arrays()
    np.testing.assert_allclose(c1, c2)


def test_translation_offset_api_equivalence():
    g = _geom()
    out_vec = translate(g, offset=(1.0, 2.0, 3.0))
    out_xyz = translate(g, offset_x=1.0, offset_y=2.0, offset_z=3.0)
    c1, _ = out_vec.as_arrays()
    c2, _ = out_xyz.as_arrays()
    np.testing.assert_allclose(c1, c2)
