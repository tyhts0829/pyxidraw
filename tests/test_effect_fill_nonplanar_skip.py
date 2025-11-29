import numpy as np

from api import E, G
from engine.core.geometry import Geometry


def _coords_offsets(g: Geometry):
    return g.as_arrays(copy=False)


def test_fill_skips_nonplanar_quad_keeps_boundary():
    # 四角形だが1点だけzを持ち、明確に非平面
    quad = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.1],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([quad])
    pipe = E.pipeline.fill(angle_sets=1, angle=0.0, density=20).build()
    out = pipe(g)

    in_coords, in_offsets = _coords_offsets(g)
    out_coords, out_offsets = _coords_offsets(out)
    assert np.array_equal(in_coords, out_coords)
    assert np.array_equal(in_offsets, out_offsets)


def test_fill_skips_nonplanar_even_when_remove_boundary_true():
    quad = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.2],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([quad])
    pipe = E.pipeline.fill(angle_sets=3, density=30, remove_boundary=True).build()
    out = pipe(g)

    in_coords, in_offsets = _coords_offsets(g)
    out_coords, out_offsets = _coords_offsets(out)
    # 非平面時は境界を保持する（空出力は避ける）
    assert np.array_equal(in_coords, out_coords)
    assert np.array_equal(in_offsets, out_offsets)


def test_fill_planar_polygon_still_generates_fill():
    g = G.polygon(n_sides=4)
    pipe = E.pipeline.fill(angle_sets=1, angle=0.0, density=10).build()
    out = pipe(g)

    in_coords, in_offsets = _coords_offsets(g)
    out_coords, out_offsets = _coords_offsets(out)

    # 輪郭は先頭に残り、出力は線本数が増える想定
    assert np.array_equal(
        in_coords[in_offsets[0] : in_offsets[1]], out_coords[out_offsets[0] : out_offsets[1]]
    )
    assert out_offsets.size > in_offsets.size
