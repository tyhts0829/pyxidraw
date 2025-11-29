from __future__ import annotations


import numpy as np

from effects.repeat import repeat
from engine.core.geometry import Geometry


def test_repeat_linear_offset_spacing() -> None:
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
        cumulative_scale=False,
        cumulative_offset=False,
        cumulative_rotate=False,
        offset=(3.0, 0.0, 0.0),
        rotation_step=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
    )

    # 非累積（線形）オフセットでは、0→3 を count で等分する
    # count=3 のとき、コピーは 1,2,3 の平行移動となる
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


def test_repeat_linear_scale_same_endpoints() -> None:
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
        cumulative_scale=False,
        cumulative_offset=False,
        cumulative_rotate=False,
        offset=(0.0, 0.0, 0.0),
        rotation_step=(0.0, 0.0, 0.0),
        scale=(0.4, 0.4, 1.0),
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
    )

    # スケールは 1→0.4 を線形補間する
    # count=3 のとき、スケールは 1, 0.8, 0.6, 0.4 となる
    expected = np.array(
        [
            # 元（スケール 1.0）
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            # コピー 1（スケール 0.8）
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
            # コピー 2（スケール 0.6）
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            # コピー 3（スケール 0.4）
            [0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0],
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


def test_repeat_cumulative_offset_with_curve() -> None:
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
        cumulative_scale=False,
        cumulative_offset=True,
        cumulative_rotate=False,
        offset=(3.0, 0.0, 0.0),
        rotation_step=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        curve=2.0,
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
    )

    # curve=2.0 のとき t' = (i/count)^2
    # count=3 のとき、オフセットは約 0.333, 1.333, 3.0
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [1.0 + 1.0 / 3.0, 0.0, 0.0],
            [4.0 / 3.0, 0.0, 0.0],
            [7.0 / 3.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(out.coords, expected)


def test_repeat_linear_rotate_z() -> None:
    coords = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([coords])

    out = repeat(
        g,
        count=2,
        cumulative_scale=False,
        cumulative_offset=False,
        cumulative_rotate=False,
        offset=(0.0, 0.0, 0.0),
        rotation_step=(0.0, 0.0, 180.0),
        scale=(1.0, 1.0, 1.0),
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
    )

    # rotation_step = 180deg, count=2 のとき、角度は 0, 90deg, 180deg
    expected = np.array(
        [
            # 元（0 度）
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            # コピー 1（90 度回転）
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            # コピー 2（180 度回転）
            [-1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(out.coords, expected, atol=1e-5)
