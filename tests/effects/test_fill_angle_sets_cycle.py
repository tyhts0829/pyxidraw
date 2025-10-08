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


def _buckets_by_region(g: Geometry):
    coords, offsets = g.as_arrays(copy=False)
    regions = [(-3.0, -2.0, -1.0, 2.0), (-1.0, -2.0, 1.0, 2.0), (1.0, -2.0, 3.0, 2.0)]
    buckets = [[] for _ in regions]
    for i in range(len(offsets) - 1):
        seg = coords[offsets[i] : offsets[i + 1]]
        mid = np.mean(seg[:, :2], axis=0)
        x, y = float(mid[0]), float(mid[1])
        for ri, (xmin, ymin, xmax, ymax) in enumerate(regions):
            if xmin <= x <= xmax and ymin <= y <= ymax:
                buckets[ri].append(seg)
                break
    return buckets


def _count_hv(segments):
    h = 0
    v = 0
    for seg in segments:
        vec = seg[1, :2] - seg[0, :2]
        if abs(float(vec[1])) < abs(float(vec[0])):
            h += 1
        if abs(float(vec[0])) < abs(float(vec[1])):
            v += 1
    return h, v


def test_fill_angle_sets_cycles_per_shape():
    # 左・中・右の 3 つのスクエア
    g = Geometry.from_lines(
        [_square(-2.0, 0.0, 0.9), _square(0.0, 0.0, 0.9), _square(2.0, 0.0, 0.9)]
    )

    pipe = E.pipeline.fill(
        angle_sets=[1, 2],  # 左=1, 中=2, 右=1（サイクル）
        angle_rad=0.0,
        density=12.0,
        remove_boundary=True,
    ).build()
    out = pipe(g)

    left, center, right = _buckets_by_region(out)
    h_l, v_l = _count_hv(left)
    h_c, v_c = _count_hv(center)
    h_r, v_r = _count_hv(right)

    # 左右は横方向のみ（垂直ほぼ無し）。中央は横・縦の両方が含まれる。
    assert h_l > 0 and v_l == 0
    assert h_r > 0 and v_r == 0
    assert h_c > 0 and v_c > 0
