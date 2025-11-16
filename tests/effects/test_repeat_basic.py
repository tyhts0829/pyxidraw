from __future__ import annotations

import numpy as np

from effects.repeat import repeat
from engine.core.geometry import Geometry


def test_repeat_cumulative_mode_matches_previous_behavior() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([coords])

    out = repeat(
        g,
        count=2,
        cumulative=True,
        offset=(1.0, 0.0, 0.0),
        angles_rad_step=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
    )

    # 累積モードでは、1 回目 +1、2 回目さらに +2 で合計 +3 の平行移動になる
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(out.coords, expected)


def test_repeat_non_cumulative_mode_even_spacing() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([coords])

    out = repeat(
        g,
        count=3,
        cumulative=False,
        offset=(1.0, 0.0, 0.0),
        angles_rad_step=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
    )

    # 非累積モードでは、コピーは 0,1,2,3 の等間隔配置
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(out.coords, expected)


def test_repeat_count_zero_is_identity() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([coords])

    out = repeat(g, count=0)

    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)
