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


def test_fill_coverage_boost_line_count_monotonic():
    g = Geometry.from_lines([_square(0.0, 0.0, 1.0)])

    density = 20.0

    pipe_low = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=density,
        coverage_boost=-0.5,
        remove_boundary=True,
    ).build()
    pipe_zero = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=density,
        coverage_boost=0.0,
        remove_boundary=True,
    ).build()
    pipe_high = E.pipeline.fill(
        angle_sets=1,
        angle_rad=0.0,
        density=density,
        coverage_boost=1.0,
        remove_boundary=True,
    ).build()

    g_low = pipe_low(g)
    g_zero = pipe_zero(g)
    g_high = pipe_high(g)

    # coverage_boost が小さいほど本数が減り、大きいほど本数が増えることを確認
    assert g_low.n_lines <= g_zero.n_lines <= g_high.n_lines
