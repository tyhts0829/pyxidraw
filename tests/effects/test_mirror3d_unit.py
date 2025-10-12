from __future__ import annotations

import math

import numpy as np

from effects.mirror3d import (
    _clip_polyline_halfspace_3d,
    _clip_polyline_wedge,
    _compute_azimuth_plane_normals,
    _equator_normal,
    _reflect_across_plane,
    _reflect_matrix,
    _rotate_around_axis,
)


def test_reflect_across_plane_basic_and_non_unit_normal() -> None:
    p = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    c = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    n = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # y=0 平面
    out = _reflect_across_plane(p, n, c)
    assert np.allclose(out, np.array([[1.0, -2.0, 3.0]], dtype=np.float32))
    # 非正規化でも同じ
    out2 = _reflect_across_plane(p, 10.0 * n, c)
    assert np.allclose(out2, out)


def test_clip_polyline_halfspace_3d_cases() -> None:
    c = np.zeros(3, dtype=np.float32)
    n = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # x>=0
    # 内→内
    v = np.array([[0.1, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    pieces = _clip_polyline_halfspace_3d(v, normal=n, center=c)
    assert len(pieces) == 1 and np.allclose(pieces[0], v)
    # 内→外（交点で切る）
    v2 = np.array([[0.1, 0.0, 0.0], [-0.3, 0.0, 0.0]], dtype=np.float32)
    pieces = _clip_polyline_halfspace_3d(v2, normal=n, center=c)
    assert len(pieces) == 1 and np.allclose(pieces[0][-1, 0], 0.0)
    # 外→内（交点から始める）
    v3 = np.array([[-0.3, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    pieces = _clip_polyline_halfspace_3d(v3, normal=n, center=c)
    assert len(pieces) == 1 and np.allclose(pieces[0][0, 0], 0.0)
    # 境界上の点（保持）
    v4 = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
    pieces = _clip_polyline_halfspace_3d(v4, normal=n, center=c)
    assert len(pieces) == 1 and np.allclose(pieces[0], v4)


def test_rotate_around_axis_z_90_180() -> None:
    p = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    c = np.zeros(3, dtype=np.float32)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    out90 = _rotate_around_axis(p, z, math.pi / 2, c)
    assert np.allclose(out90, np.array([[0.0, 1.0, 0.0]], dtype=np.float32), atol=1e-6)
    out180 = _rotate_around_axis(p, z, math.pi, c)
    assert np.allclose(out180, np.array([[-1.0, 0.0, 0.0]], dtype=np.float32), atol=1e-6)


def test_compute_azimuth_plane_normals_axis_z_phi0_align_x() -> None:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n0, n1 = _compute_azimuth_plane_normals(1, axis, -math.pi / 2)  # phi0=-90° → u0=+X
    # n0 ≈ Z×X = +Y, n1 ≈ Z×(-X) = -Y（符号は規約依存のため絶対値比較）
    assert np.allclose(np.abs(n0), np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(np.abs(n1), np.array([0.0, 1.0, 0.0], dtype=np.float32), atol=1e-6)


def test_clip_polyline_wedge_z_axis() -> None:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    c = np.zeros(3, dtype=np.float32)
    n0, n1 = _compute_azimuth_plane_normals(3, axis, -math.pi / 2)
    # x 軸上の -1→+1 をくさび（phi0=-90°, Δ=60°）でクリップすると、正側のみが残る
    v = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    pieces = _clip_polyline_wedge(v, n0=n0, n1=n1, center=c)
    # 交点が 0 付近で生成され、正側の区間のみを保持
    assert len(pieces) == 1
    seg = pieces[0]
    assert seg[0, 0] >= -1e-6 and seg[-1, 0] >= 0.0


def test_equator_normal_is_axis_unit() -> None:
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    n = _equator_normal(a)
    assert np.isclose(np.linalg.norm(n), 1.0)
    # 向きは a と同一直線上
    cosang = float(np.dot(n, a) / (np.linalg.norm(a) * np.linalg.norm(n)))
    assert np.isclose(abs(cosang), 1.0)


def test_reflect_matrix_properties() -> None:
    n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    R = _reflect_matrix(n)
    I = np.eye(3, dtype=np.float32)
    # R^2 = I, det(R) = -1
    assert np.allclose(R @ R, I, atol=1e-6)
    assert np.isclose(np.linalg.det(R), -1.0, atol=1e-6)


def test_clip_polyhedron_triangle_positive_octant() -> None:
    # 3 半空間 AND: x>=0, y>=0, z>=0
    n1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n3 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    c = np.zeros(3, dtype=np.float32)
    v = np.array([[-1.0, -1.0, -1.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    from effects.mirror3d import _clip_polyhedron_triangle

    pieces = _clip_polyhedron_triangle(v, (n1, n2, n3), c)
    assert len(pieces) == 1
    seg = pieces[0]
    # 始点は境界上へスナップ、終点は正の八分体内
    assert seg[0, 0] >= -1e-6 and seg[0, 1] >= -1e-6 and seg[0, 2] >= -1e-6
    assert np.allclose(seg[-1], np.array([0.5, 0.5, 0.5], dtype=np.float32))
