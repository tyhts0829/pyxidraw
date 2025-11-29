from __future__ import annotations

import numpy as np

from effects.affine import affine
from engine.core.geometry import Geometry


def _make_rect() -> Geometry:
    # 正方形（中心は (15, 15, 0)）
    pts = np.array(
        [
            [10.0, 10.0, 0.0],
            [20.0, 10.0, 0.0],
            [20.0, 20.0, 0.0],
            [10.0, 20.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )
    return Geometry(pts, np.array([0, len(pts)], dtype=np.int32))


def test_delta_only_translation_applied() -> None:
    g = _make_rect()
    delta = (5.0, -2.0, 3.0)
    out = affine(
        g,
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        delta=delta,
    )
    expected = g.coords + np.array(delta, dtype=np.float32)
    assert np.allclose(out.coords, expected)


def test_scale_about_origin_then_delta() -> None:
    # 原点を中心にスケール後、delta を後置加算
    g = _make_rect()
    scale = (2.0, 3.0, 1.0)
    delta = (1.0, 1.0, 0.0)
    out = affine(
        g,
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=scale,
        delta=delta,
    )
    expected = g.coords * np.array(scale, dtype=np.float32) + np.array(delta, dtype=np.float32)
    assert np.allclose(out.coords, expected)


def test_rotate_about_auto_center_then_delta() -> None:
    # 中心 (15,15,0) を基準に Z 回転後、delta を加算
    g = _make_rect()
    angle = np.pi / 2  # 90度（検証用は radian のまま）
    delta = (3.0, -1.0, 0.0)
    out = affine(
        g,
        auto_center=True,
        rotation=(0.0, 0.0, 90.0),
        scale=(1.0, 1.0, 1.0),
        delta=delta,
    )

    center = g.coords.mean(axis=0)
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    expected = ((g.coords - center) @ Rz.T) + center + np.array(delta, dtype=np.float32)
    assert np.allclose(out.coords, expected)
