from __future__ import annotations

import numpy as np
import pytest

from engine.core.geometry import Geometry


def test_as_arrays_copy_false_returns_readonly_and_view() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)])
    coords_view, offsets_view = g.as_arrays(copy=False)

    # 読み取り専用フラグ
    assert coords_view.flags.writeable is False
    assert offsets_view.flags.writeable is False

    # ビューであり元配列とメモリを共有
    assert np.shares_memory(coords_view, g.coords)
    assert np.shares_memory(offsets_view, g.offsets)

    # 書き込みは例外（読み取り専用ビュー）
    with pytest.raises(ValueError):
        coords_view[0, 0] = 123.0
    with pytest.raises(ValueError):
        offsets_view[0] = 999


def test_as_arrays_copy_true_returns_writable_and_copy() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)])
    coords_copy, offsets_copy = g.as_arrays(copy=True)

    # 書き込み可能で、元配列とメモリを共有しない
    assert coords_copy.flags.writeable is True
    assert offsets_copy.flags.writeable is True
    assert not np.shares_memory(coords_copy, g.coords)
    assert not np.shares_memory(offsets_copy, g.offsets)

    # 変更しても元に影響しない
    coords_copy[0, 0] = 123.0
    offsets_copy[0] = 999
    assert g.coords[0, 0] == 0.0
    assert g.offsets[0] == 0
