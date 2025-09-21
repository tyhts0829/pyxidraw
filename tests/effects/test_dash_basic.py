from __future__ import annotations

import numpy as np
import pytest

from effects.dash import dash
from engine.core.geometry import Geometry


@pytest.mark.smoke
def test_dash_simple_line_two_dashes() -> None:
    # 単一線分（長さ=10）に dash=3, gap=2 → パターン=5 → 2 本のダッシュ
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=3.0, gap_length=2.0)

    # ライン数と offsets の妥当性
    assert out.n_lines == 2
    # 頂点数の合計は >= 4（各ダッシュに少なくとも2点）
    assert out.offsets[0] == 0
    assert out.offsets[-1] >= 4

    # 各ダッシュの開始/終端 X 座標を検証（y,z は 0）
    first = out.coords[out.offsets[0] : out.offsets[1]]
    second = out.coords[out.offsets[1] : out.offsets[2]]
    assert np.allclose(first[[0, -1], 0], [0.0, 3.0])
    assert np.allclose(second[[0, -1], 0], [5.0, 8.0])
    assert np.allclose(out.coords[:, 1:], 0.0)


def test_dash_short_line_does_not_disappear() -> None:
    # 極短線（長さ=1）でも 1 本のダッシュが生成される（0→min(dash, L)）
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=3.0, gap_length=2.0)
    assert out.n_lines == 1
    line = out.coords[out.offsets[0] : out.offsets[1]]
    # 始点 0、終点 1（部分ダッシュ）
    assert np.allclose(line[[0, -1], 0], [0.0, 1.0])
    assert line.shape[0] >= 2


def test_dash_invalid_pattern_is_noop() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=0.0, gap_length=0.0)
    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)
