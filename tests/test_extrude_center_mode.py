import numpy as np

from engine.core.geometry import Geometry
from effects.extrude import extrude


def _polygon(n=6, r=1.0):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([r * np.cos(th), r * np.sin(th), np.zeros_like(th)], axis=1).astype(np.float32)
    # open polyline is fine for extrude
    return Geometry.from_lines([pts])


def test_extrude_center_mode_origin_vs_auto():
    g = _polygon()
    out_a = extrude(g, direction=(0, 0, 1), distance=0.5, scale=0.7, subdivisions=0.2, center_mode="origin")
    out_b = extrude(g, direction=(0, 0, 1), distance=0.5, scale=0.7, subdivisions=0.2, center_mode="auto")
    c1, _ = out_a.as_arrays()
    c2, _ = out_b.as_arrays()
    # 形状は異なる（中心の取り方が違う）
    assert not np.allclose(c1, c2)
    # 頂点数は同一（手続きは同じ）
    assert c1.shape == c2.shape
