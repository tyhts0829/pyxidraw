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


def test_fill_per_shape_params_cycle_and_angle():
    # 3 つの離散スクエア（XY 平面）
    sq0 = _square(-2.0, 0.0, 0.9)
    sq1 = _square(0.0, 0.0, 0.9)
    sq2 = _square(2.0, 0.0, 0.9)
    g = Geometry.from_lines([sq0, sq1, sq2])

    pipe = E.pipeline.fill(
        angle_sets=1,
        angle_rad=[0.0, math.pi / 2],  # 0 番目: 横, 1 番目: 縦, 2 番目: 横（サイクル）
        density=[5.0, 15.0],  # 0 番目/2 番目: 疎, 1 番目: 密
        remove_boundary=True,
    ).build()
    out = pipe(g)

    # 各領域のバウンディングボックス
    regions = [(-3.0, -2.0, -1.0, 2.0), (-1.0, -2.0, 1.0, 2.0), (1.0, -2.0, 3.0, 2.0)]
    buckets = _collect_lines_by_region(out, regions)

    # 密度: 真ん中が最も多い、左右（サイクルで 5）が少ない
    counts = [len(b) for b in buckets]
    assert counts[1] > counts[0] >= 1
    assert abs(counts[0] - counts[2]) <= 2  # サイクルで概ね一致（端数ぶれ許容）

    # 角度: 左=横（|dy| < |dx|）、中=縦（|dx| < |dy|）
    def _is_horiz(seg: np.ndarray) -> bool:
        v = seg[1, :2] - seg[0, :2]
        return abs(float(v[1])) < abs(float(v[0]))

    def _is_vert(seg: np.ndarray) -> bool:
        v = seg[1, :2] - seg[0, :2]
        return abs(float(v[0])) < abs(float(v[1]))

    # サンプル 1 本ずつで十分（生成線は同方向）
    assert buckets[0], "left bucket empty"
    assert buckets[1], "center bucket empty"
    assert _is_horiz(buckets[0][0])
    assert _is_vert(buckets[1][0])
