import numpy as np
import pytest

from api import G
from shapes.grid import Grid

# What this tests (TEST_PLAN.md §Shapes/Output)
# - grid: 線本数は Grid.MAX_DIVISIONS に依存（定数参照で検証）、全Z=0。
# - polygon: 閉路（first==last）と半径≤0.5。
# - sphere: 代表スタイルで非空・半径上限≤0.5、複数スタイルを parametrize。
# - エッジケース: subdivisions(0,0)/ (1,1) の本数、未登録アクセスは AttributeError。


def test_grid_line_count_and_z_zero():
    # Choose fractional subdivisions and compute expected lines via constant
    sub = (0.5, 0.2)
    expected_nx = int(sub[0] * Grid.MAX_DIVISIONS)
    expected_ny = int(sub[1] * Grid.MAX_DIVISIONS)
    expected_lines = expected_nx + expected_ny

    g = G.grid(subdivisions=sub)
    coords, offsets = g.as_arrays(copy=False)

    # Line count equals generated vertical+horizontal lines
    assert len(g) == expected_lines

    # All Z must be 0 for grid
    assert coords.shape[1] == 3
    assert np.allclose(coords[:, 2], 0.0, rtol=1e-6, atol=1e-6)


def test_polygon_closed_and_radius():
    g = G.polygon(n_sides=12)
    coords, offsets = g.as_arrays(copy=False)

    # Single loop; first and last vertex of the loop coincide
    i0, i1 = int(offsets[0]), int(offsets[1])
    assert i1 - i0 >= 4  # 3+1 (closed)
    v_first = coords[i0]
    v_last = coords[i1 - 1]
    assert np.allclose(v_first, v_last, rtol=1e-6, atol=1e-6)

    # Radius should be <= 0.5 (unit diameter polygon)
    r = np.linalg.norm(coords[i0:i1, :2], axis=1)
    assert np.max(r) <= 0.5 + 1e-6


def test_sphere_non_empty_and_radius_bound():
    g = G.sphere(subdivisions=0.6, sphere_type=0.3)
    coords, offsets = g.as_arrays(copy=False)
    assert coords.size > 0 and offsets.size > 0

    # All points remain on or within radius 0.5
    norms = np.linalg.norm(coords, axis=1)
    assert float(np.max(norms)) <= 0.5 + 1e-6


def test_grid_boundary_cases():
    # Zero subdivisions -> no lines
    g0 = G.grid(subdivisions=(0.0, 0.0))
    assert len(g0) == 0
    c0, o0 = g0.as_arrays(copy=False)
    assert c0.shape == (0, 3)
    assert np.array_equal(o0, np.array([0], dtype=np.int32))

    # Full subdivisions -> 2 * MAX_DIVISIONS lines
    from shapes.grid import Grid

    g1 = G.grid(subdivisions=(1.0, 1.0))
    assert len(g1) == 2 * Grid.MAX_DIVISIONS


def test_polygon_min_and_max_sides():
    # Minimum sides (triangle)
    g_min = G.polygon(n_sides=3)
    c, o = g_min.as_arrays(copy=False)
    i0, i1 = int(o[0]), int(o[1])
    assert i1 - i0 >= 4  # 3 + 1 (closed)
    # Radius within bound
    r = np.linalg.norm(c[i0:i1, :2], axis=1)
    assert np.max(r) <= 0.5 + 1e-6

    # Large sides (close to circle)
    g_max = G.polygon(n_sides=100)
    c2, o2 = g_max.as_arrays(copy=False)
    j0, j1 = int(o2[0]), int(o2[1])
    assert j1 - j0 >= 4
    r2 = np.linalg.norm(c2[j0:j1, :2], axis=1)
    assert np.max(r2) <= 0.5 + 1e-6


@pytest.mark.parametrize("sphere_type", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_sphere_styles_nonempty_and_radius_bound(sphere_type: float):
    g = G.sphere(subdivisions=0.4, sphere_type=sphere_type)
    coords, offsets = g.as_arrays(copy=False)
    assert coords.size > 0 and offsets.size > 0
    norms = np.linalg.norm(coords, axis=1)
    assert float(np.max(norms)) <= 0.5 + 1e-6


def test_unregistered_shape_attribute_error():
    # Accessing unknown shape must raise AttributeError
    from api import G as _G

    with pytest.raises(AttributeError):
        getattr(_G, "nope")
