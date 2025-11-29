import math
from typing import List, Tuple

import numpy as np

from api import E
from engine.core.geometry import Geometry


def _square(cx: float, cy: float, s: float) -> np.ndarray:
    return np.array(
        [
            [cx - s, cy - s, 0.0],
            [cx + s, cy - s, 0.0],
            [cx + s, cy + s, 0.0],
            [cx - s, cy + s, 0.0],
        ],
        dtype=np.float32,
    )


def _collect_lines_by_region(g: Geometry, regions: List[Tuple[float, float, float, float]]):
    coords, offsets = g.as_arrays(copy=False)
    buckets: List[list[np.ndarray]] = [[] for _ in regions]
    for i in range(len(offsets) - 1):
        seg = coords[offsets[i] : offsets[i + 1]]
        mid = np.mean(seg[:, :2], axis=0)
        mx, my = float(mid[0]), float(mid[1])
        for ri, (xmin, ymin, xmax, ymax) in enumerate(regions):
            if xmin <= mx <= xmax and ymin <= my <= ymax:
                buckets[ri].append(seg)
                break
    return buckets


def _segment_angle(seg: np.ndarray) -> float:
    v = seg[1, :2] - seg[0, :2]
    return math.atan2(float(v[1]), float(v[0]))


def test_fill_angle_gradient_zero_is_noop():
    # angle_gradient=0 と angle_gradient 未指定で同一結果になる
    g = Geometry.from_lines([_square(0.0, 0.0, 1.0)])

    pipe_base = E.pipeline.fill(
        angle_sets=1, angle_rad=0.0, density=12.0, remove_boundary=True
    ).build()
    pipe_grad0 = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=12.0,
        angle_gradient=0.0,
        remove_boundary=True,
    ).build()

    out_base = pipe_base(g)
    out_grad0 = pipe_grad0(g)

    c0, o0 = out_base.as_arrays(copy=False)
    c1, o1 = out_grad0.as_arrays(copy=False)

    assert o0.shape == o1.shape
    assert c0.shape == c1.shape
    assert np.allclose(c0, c1)


def test_fill_angle_gradient_per_shape_offset():
    # 1 つのスクエアの中で、上下で線の角度に傾斜がつく
    g = Geometry.from_lines([_square(0.0, 0.0, 1.0)])

    pipe = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=20.0,
        angle_gradient=math.pi / 4,
        remove_boundary=True,
    ).build()
    out = pipe(g)

    # 上半分・下半分で平均角度を比較する
    coords, offsets = out.as_arrays(copy=False)
    top_angles: list[float] = []
    bottom_angles: list[float] = []
    for i in range(len(offsets) - 1):
        seg = coords[offsets[i] : offsets[i + 1]]
        mid = np.mean(seg[:, :2], axis=0)
        angle = _segment_angle(seg)
        if mid[1] > 0.0:
            top_angles.append(angle)
        else:
            bottom_angles.append(angle)

    assert top_angles, "top region empty"
    assert bottom_angles, "bottom region empty"

    mean_top = float(np.mean(top_angles))
    mean_bottom = float(np.mean(bottom_angles))

    # 正の angle_gradient なら、上側の方が下側よりも角度が大きく（より反時計回りに）傾く
    assert mean_top > mean_bottom
