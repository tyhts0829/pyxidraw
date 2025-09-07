import numpy as np

from api import G
from effects.rotate import rotate
from effects.translate import translate

# What this tests (TEST_PLAN.md §Effects)
# - translate: 行本数・offsets不変、重心移動がdeltaに一致、入力非破壊。
# - rotate: ピボット距離保存、入力非破壊。


def test_translate_invariants_and_non_destructive():
    g = G.polygon(n_sides=7)
    coords0, offsets0 = g.as_arrays(copy=True)

    delta = (12.3, -4.5, 1.0)
    g2 = translate(g, delta=delta)
    coords2, offsets2 = g2.as_arrays(copy=False)

    # New instance
    assert g2 is not g

    # Line count preserved
    assert len(g2) == len(g)
    assert np.array_equal(offsets0, offsets2)

    # Barycenter shift equals delta
    c0 = coords0.mean(axis=0)
    c2 = coords2.mean(axis=0)
    assert np.allclose(c2 - c0, np.array(delta, dtype=np.float32), rtol=1e-6, atol=1e-6)

    # Original input not mutated
    coords_after, _ = g.as_arrays(copy=False)
    assert np.array_equal(coords_after, coords0)


def test_rotate_distance_preserved_and_non_destructive():
    g = G.polygon(n_sides=11)
    coords0, offsets0 = g.as_arrays(copy=True)

    pivot = (0.0, 0.0, 0.0)
    angles = (0.0, 0.0, 0.7)
    g2 = rotate(g, pivot=pivot, angles_rad=angles)
    coords2, offsets2 = g2.as_arrays(copy=False)

    # New instance and same line count
    assert g2 is not g
    assert len(g2) == len(g)
    assert np.array_equal(offsets0, offsets2)

    # Distances to pivot preserved
    r0 = np.linalg.norm(coords0 - np.array(pivot, dtype=np.float32), axis=1)
    r2 = np.linalg.norm(coords2 - np.array(pivot, dtype=np.float32), axis=1)
    assert np.allclose(r0, r2, rtol=1e-6, atol=1e-6)

    # Original input not mutated
    coords_after, _ = g.as_arrays(copy=False)
    assert np.array_equal(coords_after, coords0)
