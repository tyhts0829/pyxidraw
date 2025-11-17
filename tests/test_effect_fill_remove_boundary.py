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


def _circle_loop(radius: float, n: int = 128) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False, dtype=np.float32)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x, dtype=np.float32)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def test_fill_evenodd_concentric_circles_four_loops():
    # 4 つの同心円ループに対して、偶奇規則により「交互のリングだけ」が塗られることを確認
    from engine.core.geometry import Geometry

    r0, r1, r2, r3 = 4.0, 3.0, 2.0, 1.0
    loops = [
        _circle_loop(r0),
        _circle_loop(r1),
        _circle_loop(r2),
        _circle_loop(r3),
    ]
    g = Geometry.from_lines(loops)

    pipe = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=30.0,
        remove_boundary=True,
    ).build()
    out = pipe(g)

    coords, offsets = out.as_arrays(copy=False)
    eps = 0.15
    outer_ring = inner_hole = inner_ring = center = 0

    for i in range(len(offsets) - 1):
        seg = coords[offsets[i] : offsets[i + 1]]
        mid = np.mean(seg[:, :2], axis=0)
        r = float(np.hypot(mid[0], mid[1]))
        if r1 + eps <= r <= r0 - eps:
            outer_ring += 1
        elif r2 + eps <= r <= r1 - eps:
            inner_hole += 1
        elif r3 + eps <= r <= r2 - eps:
            inner_ring += 1
        elif r <= r3 - eps:
            center += 1

    # 偶奇規則: [r0,r1] と [r2,r3] のリングのみ塗られ、それ以外は抜ける
    assert outer_ring > 0
    assert inner_ring > 0
    assert inner_hole == 0
    assert center == 0
