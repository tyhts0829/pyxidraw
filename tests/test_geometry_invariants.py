import numpy as np

from engine.core.geometry import Geometry

# What this tests (TEST_PLAN.md §Geometry)
# - translate: 重心シフトがdelta、行数保持、ゼロ変位でも新インスタンス。
# - rotate: ピボットからの距離保存、行数保持。
# - scale: ピボット半径がスケール倍率で拡大。
# - concat: coords結合とoffsetsシフトの正しさ。


def _square_points(size: float = 1.0):
    s = float(size)
    return np.array(
        [
            [-s, -s, 0.0],
            [s, -s, 0.0],
            [s, s, 0.0],
            [-s, s, 0.0],
        ],
        dtype=np.float32,
    )


def test_translate_invariants_and_zero_delta_new_instance():
    g = Geometry.from_lines([_square_points(1.0)])
    coords0 = g.coords.copy()
    # centroid shift equals delta
    c0 = coords0.mean(axis=0)
    delta = np.array([2.0, -3.0, 0.5], dtype=np.float32)
    g_t = g.translate(*delta)
    c1 = g_t.coords.mean(axis=0)
    assert np.allclose(c1 - c0, delta, rtol=1e-6, atol=1e-6)
    # line count preserved
    assert len(g_t) == len(g) == 1
    # zero delta -> equal content but new instance
    g_zero = g.translate(0.0, 0.0, 0.0)
    assert g_zero is not g
    assert np.array_equal(g_zero.coords, g.coords)


def test_rotate_preserves_radii_from_pivot():
    g = Geometry.from_lines([_square_points(2.0)])
    pivot = (0.5, -0.25, 0.0)
    r0 = np.linalg.norm(g.coords - np.array(pivot, dtype=np.float32), axis=1)
    g_r = g.rotate(z=np.pi / 3, center=pivot)
    r1 = np.linalg.norm(g_r.coords - np.array(pivot, dtype=np.float32), axis=1)
    assert np.allclose(r0, r1, rtol=1e-6, atol=1e-6)
    # count preserved
    assert len(g_r) == len(g)


def test_scale_isotropic_scales_radii_about_pivot():
    g = Geometry.from_lines([_square_points(1.5)])
    pivot = (1.0, 2.0, 0.0)
    r0 = np.linalg.norm(g.coords - np.array(pivot, dtype=np.float32), axis=1)
    s = 2.0
    g_s = g.scale(sx=s, center=pivot)
    r1 = np.linalg.norm(g_s.coords - np.array(pivot, dtype=np.float32), axis=1)
    assert np.allclose(r1, r0 * s, rtol=1e-6, atol=1e-6)


def test_concat_joins_geometry_and_adjusts_offsets():
    a = Geometry.from_lines([_square_points(1.0)])
    b = a.translate(1.0, 0.0, 0.0)
    c = a.concat(b)
    # two polylines
    assert len(c) == 2
    # offsets: [0, len(a), len(a)+len(b)]
    assert c.offsets[0] == 0
    assert c.offsets[1] == a.coords.shape[0]
    assert c.offsets[2] == a.coords.shape[0] + b.coords.shape[0]
    # first segment equals a.coords, second equals b.coords
    assert np.array_equal(c.coords[: a.coords.shape[0]], a.coords)
    assert np.array_equal(c.coords[a.coords.shape[0] :], b.coords)
