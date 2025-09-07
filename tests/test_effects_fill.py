import numpy as np

from api import G
from effects.fill import fill

# What this tests (TEST_PLAN.md §Effects)
# - fill: density=0 copies（新インスタンスだが内容等価）、density>0で線本数が増加。
# - モードの単調関係: cross ≥ lines, dots ≥ lines。
# - 角度 angle_rad により出力が変化する（digest/座標差で確認）。



def test_fill_density_zero_is_copy():
    g = G.polygon(n_sides=10)
    base_coords, base_offsets = g.as_arrays(copy=True)

    g2 = fill(g, density=0.0, mode="lines")
    coords2, offsets2 = g2.as_arrays(copy=False)

    # New instance but identical content
    assert g2 is not g
    assert np.array_equal(base_offsets, offsets2)
    assert np.array_equal(base_coords, coords2)


def test_fill_density_increases_line_count():
    g = G.polygon(n_sides=8)
    base_lines = len(g)

    g_lines = fill(g, density=0.35, mode="lines")
    assert len(g_lines) > base_lines


def test_fill_mode_monotonicity_lines_cross_dots():
    g = G.polygon(n_sides=12)
    d = 0.4
    g_lines = fill(g, density=d, mode="lines")
    g_cross = fill(g, density=d, mode="cross")
    g_dots = fill(g, density=d, mode="dots")

    n_lines = len(g_lines)
    n_cross = len(g_cross)
    n_dots = len(g_dots)

    # Cross hatching should produce at least as many lines as single lines
    assert n_cross >= n_lines
    # Dot pattern is expressed as tiny crosses; expect at least single-line fill count
    assert n_dots >= n_lines


def test_fill_angle_changes_geometry():
    g = G.polygon(n_sides=9)
    a0 = 0.0
    a1 = 0.9
    g0 = fill(g, density=0.45, mode="lines", angle_rad=a0)
    g1 = fill(g, density=0.45, mode="lines", angle_rad=a1)

    # Different angle should change produced geometry
    try:
        assert g0.digest != g1.digest
    except RuntimeError:
        # Digest may be disabled; fall back to coords mismatch
        c0, _ = g0.as_arrays(copy=False)
        c1, _ = g1.as_arrays(copy=False)
        assert not np.allclose(c0, c1, rtol=1e-6, atol=1e-6)
