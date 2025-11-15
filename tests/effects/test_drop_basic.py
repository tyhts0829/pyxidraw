from __future__ import annotations

import numpy as np

from effects.drop import drop
from engine.core.geometry import Geometry


def _make_lines(n: int, length: float = 10.0) -> Geometry:
    lines = []
    for i in range(n):
        y = float(i)
        lines.append(
            np.array(
                [[0.0, y, 0.0], [length, y, 0.0]],
                dtype=np.float32,
            )
        )
    return Geometry.from_lines(lines)


def test_drop_interval_keeps_every_second_line() -> None:
    g = _make_lines(4, length=10.0)

    out = drop(g, interval=2, offset=0, keep_mode="drop")

    assert out.n_lines == 2
    ys = []
    for i in range(out.n_lines):
        start = out.offsets[i]
        ys.append(float(out.coords[start, 1]))
    assert ys == [1.0, 3.0]


def test_drop_min_length_removes_short_lines() -> None:
    g = Geometry.from_lines(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 2.0, 0.0], [3.0, 2.0, 0.0]], dtype=np.float32),
        ]
    )

    out = drop(g, min_length=2.0, keep_mode="drop")

    assert out.n_lines == 1
    start = out.offsets[0]
    end = out.offsets[1]
    kept = out.coords[start:end]
    assert np.allclose(kept[[0, -1], 0], [0.0, 3.0])


def test_drop_keep_mode_keeps_only_short_lines() -> None:
    g = Geometry.from_lines(
        [
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0.0, 1.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32),
        ]
    )

    out = drop(g, min_length=2.0, keep_mode="keep")

    assert out.n_lines == 1
    start = out.offsets[0]
    end = out.offsets[1]
    kept = out.coords[start:end]
    # 長さ 1 の線だけ残る
    assert np.allclose(kept[[0, -1], 0], [0.0, 1.0])


def test_drop_probability_is_deterministic_with_seed() -> None:
    g = _make_lines(10, length=5.0)

    out1 = drop(g, probability=0.5, seed=123)
    out2 = drop(g, probability=0.5, seed=123)

    assert np.array_equal(out1.coords, out2.coords)
    assert np.array_equal(out1.offsets, out2.offsets)
