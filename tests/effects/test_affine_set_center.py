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


def test_identity_when_no_transform() -> None:
    g = _make_rect()
    out1 = affine(
        g,
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
        angles_rad=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
    )
    out2 = affine(g, auto_center=True, angles_rad=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
    assert np.allclose(out1.coords, g.coords)
    assert np.allclose(out2.coords, g.coords)


def test_center_toggle_affects_result() -> None:
    g = _make_rect()
    # 180度回転のみ適用（スケールは恒等）。中心選択の違いで結果が変わるはず。
    out_origin = affine(
        g,
        auto_center=False,
        pivot=(0.0, 0.0, 0.0),
        angles_rad=(0.0, 0.0, np.pi),
        scale=(1.0, 1.0, 1.0),
    )
    out_center = affine(g, auto_center=True, angles_rad=(0.0, 0.0, np.pi), scale=(1.0, 1.0, 1.0))
    assert not np.allclose(out_origin.coords, out_center.coords)
