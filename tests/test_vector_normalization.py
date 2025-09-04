import numpy as np

from engine.core.geometry import Geometry
from effects.rotation import rotate
from effects.transform import affine
from effects.array import repeat
from effects.translation import translate


def _geom():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_rotation_angles_rad_consistency():
    g = _geom()
    ang = (np.pi/2, np.pi/2, np.pi/2)
    out_a = rotate(g, angles_rad=ang)
    out_b = rotate(g, angles_rad=(np.pi/2, np.pi/2, np.pi/2))
    c1, _ = out_a.as_arrays()
    c2, _ = out_b.as_arrays()
    np.testing.assert_allclose(c1, c2)


def test_transform_angles_rad_changes_geometry():
    g = _geom()
    out = affine(g, angles_rad=(0.0, 0.0, np.pi))
    out2 = affine(g, angles_rad=(0.0, 0.0, 0.0))
    c1, _ = out.as_arrays()
    c2, _ = out2.as_arrays()
    assert not np.allclose(c1, c2)


def test_array_angles_rad_step_consistency():
    g = _geom()
    out_a = repeat(g, count=2, angles_rad_step=(np.pi/2, np.pi/2, np.pi/2))
    out_b = repeat(g, count=2, angles_rad_step=(np.pi/2, np.pi/2, np.pi/2))
    c1, _ = out_a.as_arrays()
    c2, _ = out_b.as_arrays()
    np.testing.assert_allclose(c1, c2)


def test_translation_delta_is_applied():
    g = _geom()
    out = translate(g, delta=(1.0, 2.0, 3.0))
    c, _ = out.as_arrays()
    # 一点だけ確認
    assert np.allclose(c[1], np.array([1.0+1.0, 0.0+2.0, 0.0+3.0], dtype=np.float32))
