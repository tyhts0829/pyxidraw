import math
import numpy as np

from engine.core.geometry import Geometry


def make_line(points):
    return np.asarray(points, dtype=np.float32)


def test_from_lines_and_empty():
    g0 = Geometry.from_lines([])
    coords, offsets = g0.as_arrays()
    assert coords.shape == (0, 3)
    assert offsets.tolist() == [0]

    g1 = Geometry.from_lines([make_line([[0, 0, 0], [1, 0, 0]])])
    c1, o1 = g1.as_arrays()
    assert c1.shape == (2, 3)
    assert o1.tolist() == [0, 2]


def test_translate_is_pure():
    base = Geometry.from_lines([make_line([[0, 0, 0], [1, 0, 0]])])
    moved = base.translate(10, -2, 3)
    b_c, _ = base.as_arrays()
    m_c, _ = moved.as_arrays()
    assert np.allclose(b_c[0], [0, 0, 0])
    assert np.allclose(m_c[0], [10, -2, 3])


def test_scale_uniform_and_center():
    base = Geometry.from_lines([make_line([[1, 0, 0]])])
    g1 = base.scale(2.0)
    c1, _ = g1.as_arrays()
    assert np.allclose(c1[0], [2, 0, 0])

    # center=(1,0,0): 点(2,0,0)を中心に0.5倍 → (1.5,0,0)
    g2 = Geometry.from_lines([make_line([[2, 0, 0]])]).scale(0.5, center=(1.0, 0.0, 0.0))
    c2, _ = g2.as_arrays()
    assert np.allclose(c2[0], [1.5, 0, 0])


def test_rotate_z_90deg():
    base = Geometry.from_lines([make_line([[1, 0, 0]])])
    g = base.rotate(z=math.pi / 2)
    c, _ = g.as_arrays()
    assert np.allclose(c[0], [0, 1, 0], atol=1e-6)


def test_concat_lines():
    a = Geometry.from_lines([make_line([[0, 0, 0], [1, 0, 0]])])
    b = Geometry.from_lines([make_line([[0, 1, 0]])])
    c = a.concat(b)
    coords, offsets = c.as_arrays()
    assert coords.shape == (3, 3)
    assert offsets.tolist() == [0, 2, 3]


def test_as_arrays_copy_behavior():
    g = Geometry.from_lines([make_line([[0, 0, 0]])])
    coords_view, _ = g.as_arrays(copy=False)
    coords_copy, _ = g.as_arrays(copy=True)
    # view shares memory; copy does not
    coords_view[0, 0] += 1
    v_after, _ = g.as_arrays()
    assert np.allclose(v_after[0, 0], 1)
    assert not np.shares_memory(coords_copy, v_after)


def test_effect_noise_integration_path():
    # Verify function-based effect accepts Geometry and returns Geometry
    from effects.noise import displace

    base = Geometry.from_lines([make_line([[0, 0, 0], [1, 0, 0], [0, 1, 0]])])
    out = displace(base, amplitude_mm=0.01, spatial_freq=(0.1, 0.1, 0.1), t_sec=0.0)

    assert isinstance(out, Geometry)
    c_in, o_in = base.as_arrays()
    c_out, o_out = out.as_arrays()
    assert o_in.tolist() == o_out.tolist()  # topology preserved
    assert c_out.shape == c_in.shape
