from __future__ import annotations

import numpy as np
import pytest

shapely = pytest.importorskip("shapely")  # noqa: F401 - 重依存の存在確認

from effects.offset import offset
from engine.core.geometry import Geometry


@pytest.mark.smoke
def test_offset_distance_zero_is_identity() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)])
    out = offset(g, distance=0.0)
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)


def test_offset_keep_original_adds_original_lines() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)])

    out = offset(g, distance=1.0, keep_original=True)

    # 少なくとも元の線を含む 2 本以上のラインがあることを期待
    assert isinstance(out, Geometry)
    assert len(out.offsets) >= 3

    # 末尾のラインが元のポリラインと一致することを確認
    last_start = out.offsets[-2]
    last_end = out.offsets[-1]
    last_line = out.coords[last_start:last_end]
    assert np.allclose(last_line, g.coords)


def test_offset_distance_zero_with_keep_original_is_identity() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)])
    out = offset(g, distance=0.0, keep_original=True)
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)
