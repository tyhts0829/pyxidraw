import numpy as np

from api import E, G


def _first_line_coords(g):
    coords, offsets = g.as_arrays(copy=False)
    if len(offsets) < 2:
        return np.empty((0, 3), dtype=np.float32)
    return coords[offsets[0] : offsets[1]]


def _all_lines(g):
    coords, offsets = g.as_arrays(copy=False)
    lines = []
    for i in range(len(offsets) - 1):
        lines.append(coords[offsets[i] : offsets[i + 1]])
    return lines


def test_fill_keep_boundary_default():
    g = G.polygon(n_sides=4)
    in_first = _first_line_coords(g)
    pipe = E.pipeline.fill(mode="lines", density=10).build()
    out = pipe(g)
    out_first = _first_line_coords(out)
    assert np.array_equal(in_first, out_first)


def test_fill_remove_boundary_true():
    g = G.polygon(n_sides=4)
    in_first = _first_line_coords(g)
    pipe = E.pipeline.fill(mode="lines", angle_rad=0.0, density=10, remove_boundary=True).build()
    out = pipe(g)
    lines = _all_lines(out)
    # 元ポリゴン線と完全一致するラインが含まれないこと
    assert not any(np.array_equal(in_first, ln) for ln in lines)


def test_fill_noop_when_density_zero():
    g = G.polygon(n_sides=4)
    pipe = E.pipeline.fill(density=0).build()
    out = pipe(g)
    in_coords, in_offsets = g.as_arrays(copy=False)
    out_coords, out_offsets = out.as_arrays(copy=False)
    assert np.array_equal(in_coords, out_coords)
    assert np.array_equal(in_offsets, out_offsets)
