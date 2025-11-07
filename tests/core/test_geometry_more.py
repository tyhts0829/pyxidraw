from __future__ import annotations

import math

import numpy as np

from engine.core.geometry import Geometry


def test_from_lines_normalizes_3d_and_1d_and_mixed() -> None:
    # 3D 入力
    l3d = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    g3 = Geometry.from_lines([l3d])
    assert g3.coords.shape == (2, 3)
    assert g3.offsets.tolist() == [0, 2]

    # 1D（(x,y,z) 並び）
    l1d = np.array([0.0, 0.0, 0.0, 4.0, 5.0, 6.0], dtype=np.float32)  # 2 点
    g1 = Geometry.from_lines([l1d])
    assert g1.coords.shape == (2, 3)
    assert np.allclose(g1.coords[1], [4.0, 5.0, 6.0])
    assert g1.offsets.tolist() == [0, 2]

    # 混在（2D + 3D + 1D）
    l2d = np.array([[10.0, 10.0], [11.0, 12.0]], dtype=np.float32)  # 2 点 → Z=0 補完
    l3d_b = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)  # 1 点
    l1d_b = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 1 点
    gmix = Geometry.from_lines([l2d, l3d_b, l1d_b])
    assert gmix.coords.shape == (4, 3)
    assert gmix.offsets.tolist() == [0, 2, 3, 4]
    # 2D は Z=0
    assert np.allclose(gmix.coords[:2, 2], 0.0)


def test_transforms_empty_noop_and_chain() -> None:
    g_empty = Geometry.from_lines([])
    # 各変換は空集合に対して no-op（ただし新インスタンス）
    for fn in (
        lambda g: g.translate(1.0, 2.0, 3.0),
        lambda g: g.scale(2.0),
        lambda g: g.rotate(z=math.pi / 2),
    ):
        g2 = fn(g_empty)
        assert g2 is not g_empty
        assert g2.coords.shape == (0, 3) and g2.offsets.tolist() == [0]

    # 変換の組み合わせ: translate → scale → rotate(Z=90°)
    p = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([p])
    g2 = g.translate(1.0, 1.0, 0.0).scale(2.0).rotate(z=math.pi / 2)
    # 手計算: (1,0,0) → + (1,1,0) = (2,1,0) → *2 = (4,2,0) → rotZ90 = (-2,4,0)
    assert np.allclose(g2.coords, np.array([[-2.0, 4.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_rotate_known_quarter_turns() -> None:
    # Z 軸 +90°: (1,0,0) -> (0,1,0)
    g = Geometry.from_lines([np.array([[1.0, 0.0, 0.0]], dtype=np.float32)])
    gz = g.rotate(z=math.pi / 2)
    assert np.allclose(gz.coords, np.array([[0.0, 1.0, 0.0]], dtype=np.float32), atol=1e-6)

    # X 軸 +90°: (0,1,0) -> (0,0,1)
    g2 = Geometry.from_lines([np.array([[0.0, 1.0, 0.0]], dtype=np.float32)])
    gx = g2.rotate(x=math.pi / 2)
    assert np.allclose(gx.coords, np.array([[0.0, 0.0, 1.0]], dtype=np.float32), atol=1e-6)


def test_translate_changes_coords_and_preserves_offsets() -> None:
    g = Geometry.from_lines([np.array([[0.0, 0.0, 0.0]], dtype=np.float32)])
    g2 = g.translate(1.0, 0.0, 0.0)
    # 内容が変わる（座標は +x 方向へ移動）
    assert np.allclose(g2.coords, g.coords + np.array([1.0, 0.0, 0.0], dtype=np.float32))
    # offsets は不変
    assert np.array_equal(g2.offsets, g.offsets)


def test_equality_is_identity_based() -> None:
    # 内容が同一でも別インスタンスは等価とならない（eq=False）
    pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    g1 = Geometry.from_lines([pts])
    g2 = Geometry.from_lines([pts.copy()])
    assert (g1 == g1) is True
    assert (g1 == g2) is False
