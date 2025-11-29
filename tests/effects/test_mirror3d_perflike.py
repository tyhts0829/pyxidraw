from __future__ import annotations

import math

import numpy as np

from effects.mirror3d import mirror3d
from engine.core.geometry import Geometry


def test_mirror3d_equator_dedup_many_points() -> None:
    # 入力: 楔内の z=0 上の点を多数（単点ライン）
    # 条件: mirror_equator=True で 4n 倍になるが、z=0 は赤道反転で重複 → 2n 倍に正規化されるはず。
    n_az = 5  # 2n = 10
    ang = math.radians(10.0)
    pts = []
    for i in range(50):
        r = 1.0 + 0.001 * i  # 半径を微妙に変えて重複を避ける
        pts.append(np.array([[r * math.cos(ang), r * math.sin(ang), 0.0]], dtype=np.float32))
    g = Geometry.from_lines(pts)

    out = mirror3d(
        g,
        n_azimuth=n_az,
        cx=0.0,
        cy=0.0,
        cz=0.0,
        axis=(0.0, 0.0, 1.0),
        phi0=-90.0,
        mirror_equator=True,
    )

    _, offsets = out.as_arrays(copy=False)
    n_lines = len(offsets) - 1
    assert n_lines == 50 * (2 * n_az)
