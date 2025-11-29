import math

import numpy as np

from api import E
from engine.core.geometry import Geometry


def _circle(n: int, r: float) -> np.ndarray:
    ts = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False, dtype=np.float32)
    xs = r * np.cos(ts)
    ys = r * np.sin(ts)
    return np.stack([xs, ys, np.zeros_like(xs)], axis=1)


def test_fill_angle_gradient_coverage_scale_respects_angle_gradient():
    # 円に対して angle_gradient と coverage_scale を同時に与えたとき、
    # coverage_scale で増えた線にも angle_gradient による傾きが反映されているかを確認する。
    base = Geometry.from_lines([_circle(128, 1.0)])

    angle_gradient = math.pi / 2  # 上側で 90 度近く傾くような大きめの値

    pipe_cov1 = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=30.0,
        angle_gradient=angle_gradient,
        coverage_scale=1.0,
        remove_boundary=True,
    ).build()
    pipe_cov2 = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=30.0,
        angle_gradient=angle_gradient,
        coverage_boost=0.0,
        coverage_scale=2.0,
        remove_boundary=True,
    ).build()

    g1 = pipe_cov1(base)
    g2 = pipe_cov2(base)

    coords1, offs1 = g1.as_arrays(copy=False)
    coords2, offs2 = g2.as_arrays(copy=False)

    # coverage_scale を増やすことで線本数が増えていることを確認
    assert g2.n_lines >= g1.n_lines

    def _segment_angle(seg: np.ndarray) -> float:
        v = seg[1, :2] - seg[0, :2]
        return math.atan2(float(v[1]), float(v[0]))

    # 上半分（y > 0）の線の角度分布を比較し、coverage_scale を変えても
    # angle_gradient の向き（より大きく反時計回り）が維持されていることを確認する。
    top_angles1 = []
    for i in range(len(offs1) - 1):
        seg = coords1[offs1[i] : offs1[i + 1]]
        mid = np.mean(seg[:, :2], axis=0)
        if mid[1] > 0.0:
            top_angles1.append(_segment_angle(seg))

    top_angles2 = []
    for i in range(len(offs2) - 1):
        seg = coords2[offs2[i] : offs2[i + 1]]
        mid = np.mean(seg[:, :2], axis=0)
        if mid[1] > 0.0:
            top_angles2.append(_segment_angle(seg))

    assert top_angles1, "no segments in upper half for coverage_scale=1.0"
    assert top_angles2, "no segments in upper half for coverage_scale=2.0"

    mean_top1 = float(np.mean(top_angles1))
    mean_top2 = float(np.mean(top_angles2))

    # angle_gradient の方向（正方向）に対して、coverage_scale を変えても
    # 上側の平均角度が同じ向き（正の角度）で保たれていることを期待する。
    assert mean_top1 > 0.0
    assert mean_top2 > 0.0
