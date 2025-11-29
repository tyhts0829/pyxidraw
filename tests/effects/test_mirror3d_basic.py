from __future__ import annotations

import math

import numpy as np

from effects.mirror3d import mirror3d
from engine.core.geometry import Geometry


def _angles_sorted_xy(coords: np.ndarray) -> np.ndarray:
    phis = np.mod(np.arctan2(coords[:, 1], coords[:, 0]), 2 * math.pi)
    return np.sort(phis)


def test_mirror3d_n3_sector_replication_and_remove_others() -> None:
    ang = math.radians(15.0)
    p_src = np.array([[math.cos(ang), math.sin(ang), 0.0]], dtype=np.float32)
    ang_other = math.radians(100.0)
    p_other = np.array([[math.cos(ang_other), math.sin(ang_other), 0.0]], dtype=np.float32)

    g = Geometry.from_lines([p_src, p_other])
    out = mirror3d(
        g,
        n_azimuth=3,
        cx=0.0,
        cy=0.0,
        cz=0.0,
        axis=(0.0, 0.0, 1.0),
        phi0=-90.0,  # 基準境界を +X に合わせる
        mirror_equator=False,
    )

    coords, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 6

    phis_sorted = _angles_sorted_xy(coords)
    base = ang
    step = 2 * math.pi / 3.0
    expected = []
    for m in range(3):
        expected.append((base + m * step) % (2 * math.pi))
        expected.append((-base + m * step) % (2 * math.pi))
    expected = np.array(sorted(expected), dtype=np.float64)
    assert np.allclose(phis_sorted, expected, atol=1e-5)


def test_mirror3d_equator_mirror_doubles_count_and_flips_z() -> None:
    ang = math.radians(15.0)
    p_src = np.array([[math.cos(ang), math.sin(ang), 0.5]], dtype=np.float32)
    p_other = np.array(
        [[math.cos(math.radians(100.0)), math.sin(math.radians(100.0)), -0.25]], dtype=np.float32
    )
    g = Geometry.from_lines([p_src, p_other])

    out = mirror3d(
        g,
        n_azimuth=2,
        cx=0.0,
        cy=0.0,
        cz=0.0,
        axis=(0.0, 0.0, 1.0),
        phi0=0.0,
        mirror_equator=True,
    )

    coords, offsets = out.as_arrays(copy=False)
    # 2n=4 → equator反転で 8
    assert len(offsets) - 1 == 8
    # z が ±0.5 の両方を含む
    zs = coords[:, 2]
    assert any(np.isclose(zs, 0.5)) and any(np.isclose(zs, -0.5))


def test_mirror3d_n1_halfspace_reflection_two_lines() -> None:
    # 右側の点と左側の点を入力。n=1 なら半空間（x>=0）をソースに 2 本生成。
    p_right = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
    p_left = np.array([[-0.5, 0.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([p_right, p_left])

    out = mirror3d(
        g,
        n_azimuth=1,
        cx=0.0,
        cy=0.0,
        cz=0.0,
        axis=(0.0, 0.0, 1.0),
        phi0=180.0,
        mirror_equator=False,
    )

    coords, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 2
    # 期待: [0.5,0,0] と [-0.5,0,0]
    lines = [coords[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    assert any(np.allclose(ln, p_right) for ln in lines)
    assert any(np.allclose(ln, p_left) for ln in lines)


def test_mirror3d_center_axis_reflection_around_cx() -> None:
    # 中心 (cx,0,0) で x 方向に対称。n=1 で x=cx について反射。
    cx = 1.0
    p_right = np.array([[cx + 0.4, 0.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([p_right])

    out = mirror3d(
        g,
        n_azimuth=1,
        cx=cx,
        cy=0.0,
        cz=0.0,
        axis=(0.0, 0.0, 1.0),
        phi0=180.0,
        mirror_equator=False,
    )

    coords, offsets = out.as_arrays(copy=False)
    assert len(offsets) - 1 == 2
    lines = [coords[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    expect_left = np.array([[cx - 0.4, 0.0, 0.0]], dtype=np.float32)
    assert any(np.allclose(ln, p_right) for ln in lines)
    assert any(np.allclose(ln, expect_left) for ln in lines)
