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


def test_dash_all_zero_dash_lengths_is_noop_and_consistent() -> None:
    # 全てのダッシュ長が 0 の場合でも Geometry の不変条件が壊れず、元線が保持される
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=[0.0, 0.0, 0.0], gap_length=[2.0, 2.0, 2.0])

    assert np.allclose(out.coords, g.coords)
    assert np.array_equal(out.offsets, g.offsets)


def test_dash_list_pattern_on_dash_length() -> None:
    # dash_length=[1,3,2], gap_length=2 → (1,2),(3,2),(2,2)... でサイクル
    # 単一線分（長さ=20） → 5 本のダッシュを想定
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=[1.0, 3.0, 2.0], gap_length=2.0)

    assert out.n_lines == 5

    starts = np.array([0.0, 3.0, 8.0, 12.0, 15.0])
    ends = np.array([1.0, 6.0, 10.0, 13.0, 18.0])

    for i in range(out.n_lines):
        line = out.coords[out.offsets[i] : out.offsets[i + 1]]
        assert np.allclose(line[0, 0], starts[i])
        assert np.allclose(line[-1, 0], ends[i])


def test_dash_mixed_scalar_and_list_gap() -> None:
    # dash_length=2, gap_length=[1,3] → (2,1),(2,3)... でサイクル
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [14.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=2.0, gap_length=[1.0, 3.0])

    # 期待されるダッシュ区間: [0,2], [3,5], [8,10], [11,13]
    assert out.n_lines == 4

    starts = np.array([0.0, 3.0, 8.0, 11.0])
    ends = np.array([2.0, 5.0, 10.0, 13.0])

    for i in range(out.n_lines):
        line = out.coords[out.offsets[i] : out.offsets[i + 1]]
        assert np.allclose(line[0, 0], starts[i])
        assert np.allclose(line[-1, 0], ends[i])


def test_dash_offset_shifts_pattern_simple() -> None:
    # offset によりパターン開始位置をシフト
    # ベース: dash=3, gap=2, L=10 → [0,3], [5,8]
    # offset=2 → u軸 [2,3],[5,8],[10,12] → t軸 [0,1],[3,6],[8,10]
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=3.0, gap_length=2.0, offset=2.0)

    assert out.n_lines == 3

    starts = np.array([0.0, 3.0, 8.0])
    ends = np.array([1.0, 6.0, 10.0])

    for i in range(out.n_lines):
        line = out.coords[out.offsets[i] : out.offsets[i + 1]]
        assert np.allclose(line[0, 0], starts[i])
        assert np.allclose(line[-1, 0], ends[i])


def test_dash_offset_with_list_pattern() -> None:
    # dash_length=[1,3,2], gap_length=2, L=20 のとき
    # offset=1 → u軸 [3,6],[8,10],[12,13],[15,18],[20,21] → t軸 [2,5],[7,9],[11,12],[14,17],[19,20]
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32)])
    out = dash(g, dash_length=[1.0, 3.0, 2.0], gap_length=2.0, offset=1.0)

    assert out.n_lines == 5

    starts = np.array([2.0, 7.0, 11.0, 14.0, 19.0])
    ends = np.array([5.0, 9.0, 12.0, 17.0, 20.0])

    for i in range(out.n_lines):
        line = out.coords[out.offsets[i] : out.offsets[i + 1]]
        assert np.allclose(line[0, 0], starts[i])
        assert np.allclose(line[-1, 0], ends[i])
