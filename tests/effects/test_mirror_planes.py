from __future__ import annotations

import numpy as np

from effects.mirror import mirror
from engine.core.geometry import Geometry


def _sorted_coords(g: Geometry) -> np.ndarray:
    arr = g.coords.copy()
    idx = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[idx]


def test_mirror_planes_n1_basic_remove_opposite_and_reflect() -> None:
    # 右側の2点と左側の2点
    pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [-3.0, 0.5, 0.0],
            [-4.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([pts[0:1], pts[1:2], pts[2:3], pts[3:4]])

    out = mirror(
        g,
        n_mirror=1,
        cx=0.0,
        cy=0.0,
        source_side=True,  # x >= 0 をソース
    )

    # 期待: 右側2点は残存、左側2点は消え、右側2点の鏡映が左側に生成
    expect = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    out_sorted = _sorted_coords(out)
    exp_sorted = _sorted_coords(Geometry.from_lines([expect]))
    assert np.allclose(out_sorted, exp_sorted)


def test_mirror_planes_n2_quadrant_fill_and_remove_others() -> None:
    # 各象限に1点ずつ
    pts = np.array(
        [
            [1.0, 1.0, 0.0],  # Q1
            [-1.0, 1.0, 0.0],  # Q2
            [1.0, -1.0, 0.0],  # Q4
            [-1.0, -1.0, 0.0],  # Q3
        ],
        dtype=np.float32,
    )
    g = Geometry.from_lines([pts[0:1], pts[1:2], pts[2:3], pts[3:4]])

    out = mirror(
        g,
        n_mirror=2,
        cx=0.0,
        cy=0.0,
        source_side=(True, True),  # 第1象限をソース
    )

    # 期待: Q1 のみをソースに、3象限へ鏡映 → 4点全てが Q1 から生成される
    expect = np.array(
        [
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    out_sorted = _sorted_coords(out)
    exp_sorted = _sorted_coords(Geometry.from_lines([expect]))
    assert np.allclose(out_sorted, exp_sorted)


def test_mirror_planes_n1_clip_crossing_segment_and_reflect() -> None:
    # x 軸に沿って -1→2 を結ぶ線（x=0で交差）。ソースは x>=0
    line = np.array([[-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([line])

    out = mirror(
        g,
        n_mirror=1,
        cx=0.0,
        cy=0.0,
        source_side=True,
    )

    # 期待: クリップされた [0,2] と、その鏡映 [0,-2]
    # 出力は2本のポリライン
    coords, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 2
    l0 = coords[offsets[0] : offsets[1]]
    l1 = coords[offsets[1] : offsets[2]]
    lines = [l0, l1]
    has_pos = any(
        np.allclose(ln, np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32))
        for ln in lines
    )
    has_neg = any(
        np.allclose(ln, np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]], dtype=np.float32))
        for ln in lines
    )
    assert has_pos and has_neg


def test_mirror_planes_preserve_z_and_boundary_dedup() -> None:
    # 境界 x=0 上の線は 1 度だけ保持、z は不変
    line = np.array([[0.0, -1.0, 5.0], [0.0, 2.0, 5.0]], dtype=np.float32)
    g = Geometry.from_lines([line])

    out = mirror(
        g,
        n_mirror=1,
        cx=0.0,
        cy=0.0,
        source_side=True,
    )

    coords, offsets = out.as_arrays(copy=False)
    # 1 本のみ、内容は変わらない（重複生成されない）
    assert len(offsets) - 1 == 1
    assert np.allclose(coords[offsets[0] : offsets[1]], line)
    assert np.allclose(coords[:, 2], 5.0)


def test_mirror_n3_sector_replication_and_remove_others() -> None:
    # n=3、ソースは楔 [0, π/3)。角度 15° の点をソースに置く。
    ang = np.deg2rad(15.0)
    p_src = np.array([[np.cos(ang), np.sin(ang), 0.0]], dtype=np.float32)

    # 他セクタにある点（例えば 100°）は消えるべき
    ang_other = np.deg2rad(100.0)
    p_other = np.array([[np.cos(ang_other), np.sin(ang_other), 0.0]], dtype=np.float32)

    g = Geometry.from_lines([p_src, p_other])

    out = mirror(g, n_mirror=3, cx=0.0, cy=0.0)

    # 期待: 2n=6 個の点が等角度に出現（±15° を 120° ずつ回転した配置）
    coords, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 6  # 6 本の 1 点ライン
    # 角度を正規化して比較
    phis = np.arctan2(coords[:, 1], coords[:, 0])
    phis = np.mod(phis, 2 * np.pi)
    phis_sorted = np.sort(phis)

    base = float(ang)
    step = 2 * np.pi / 3.0
    expected = []
    for m in range(3):
        expected.append((base + m * step) % (2 * np.pi))  # 非反転
        expected.append((-base + m * step) % (2 * np.pi))  # 反転
    expected = np.array(sorted(expected), dtype=np.float64)

    # 角度に対する全体一致（トレランス）
    assert np.allclose(phis_sorted, expected, atol=1e-5)
