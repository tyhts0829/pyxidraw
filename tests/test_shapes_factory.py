import numpy as np

from api import G, Geometry

# What this tests (TEST_PLAN.md §Shapes/Factory)
# - G.list_shapes includes core shapes, and dynamic resolution `G.<name>(...)` works.
# - LRU identity for repeated params (ints), NumPy scalar acceptance with content equality.


def test_list_shapes_and_resolution():
    names = set(G.list_shapes())
    # At least these core shapes must be registered
    assert {"grid", "polygon", "sphere"}.issubset(names)

    g = G.grid(subdivisions=(0.2, 0.1))
    assert isinstance(g, Geometry)


def test_lru_identity_and_numpy_scalars():
    # Same params -> same cached instance (identity)
    g1 = G.polygon(n_sides=6)
    g2 = G.polygon(n_sides=6)
    assert g1 is g2

    # NumPy scalar params are accepted and normalize to the same cache key
    g3 = G.polygon(n_sides=np.int32(6))
    assert g3 is g1

    # Tuple containing numpy float scalars is accepted (content equal),
    # but float rounding may prevent cache identity — compare arrays instead.
    g4 = G.grid(subdivisions=(np.float32(0.2), np.float64(0.1)))
    g5 = G.grid(subdivisions=(0.2, 0.1))
    c4, o4 = g4.as_arrays(copy=False)
    c5, o5 = g5.as_arrays(copy=False)
    assert np.array_equal(o4, o5) and np.allclose(c4, c5, rtol=1e-6, atol=1e-6)
