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
    pipe = E.pipeline.fill(angle_sets=1, density=10).build()
    out = pipe(g)
    out_first = _first_line_coords(out)
    assert np.array_equal(in_first, out_first)


def test_fill_remove_boundary_true():
    g = G.polygon(n_sides=4)
    in_first = _first_line_coords(g)
    pipe = E.pipeline.fill(angle_sets=1, angle_rad=0.0, density=10, remove_boundary=True).build()
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


def test_fill_evenodd_excludes_inner_hole_square():
    # 外環と内環からなるドーナツ形状に対して、穴内部には塗り線が入らない
    outer_s = 2.0
    inner_s = 0.5
    outer = np.array(
        [
            [-outer_s, -outer_s, 0.0],
            [outer_s, -outer_s, 0.0],
            [outer_s, outer_s, 0.0],
            [-outer_s, outer_s, 0.0],
        ],
        dtype=np.float32,
    )
    inner = np.array(
        [
            [-inner_s, -inner_s, 0.0],
            [inner_s, -inner_s, 0.0],
            [inner_s, inner_s, 0.0],
            [-inner_s, inner_s, 0.0],
        ],
        dtype=np.float32,
    )
    from engine.core.geometry import Geometry

    g = Geometry.from_lines([outer, inner])
    pipe = E.pipeline.fill(angle_sets=1, angle_rad=0.0, density=20.0, remove_boundary=True).build()
    out = pipe(g)

    coords, offsets = out.as_arrays(copy=False)
    inside_hole = 0
    in_ring = 0
    for i in range(len(offsets) - 1):
        seg = coords[offsets[i] : offsets[i + 1]]
        mid = np.mean(seg[:, :2], axis=0)
        mx, my = float(mid[0]), float(mid[1])
        r_inf = max(abs(mx), abs(my))
        if r_inf < inner_s - 1e-3:
            inside_hole += 1
        if inner_s - 1e-3 <= r_inf <= outer_s + 1e-3:
            in_ring += 1

    assert in_ring > 0
    assert inside_hole == 0
