"""
どこで: tests（shapes/polygon）。
何を: polygon の phase パラメータで頂点開始角を制御できることを確認。
なぜ: 菱形/水平配置を簡単に切り替えられることを保証するため。
"""

from __future__ import annotations

import math

from api import G  # type: ignore[attr-defined]


def _first_edge_vector(geom):
    coords, offsets = geom.as_arrays(copy=False)
    start = coords[offsets[0]]
    end = coords[offsets[0] + 1]
    return end - start


def test_polygon_default_phase_keeps_diamond_orientation():
    edge = _first_edge_vector(G.polygon(n_sides=4).realize())
    assert not math.isclose(float(edge[0]), 0.0, abs_tol=1e-6)
    assert not math.isclose(float(edge[1]), 0.0, abs_tol=1e-6)


def test_polygon_phase_aligns_first_edge_horizontally():
    edge = _first_edge_vector(G.polygon(n_sides=4, phase=45.0).realize())
    assert math.isclose(float(edge[1]), 0.0, abs_tol=1e-6)
    assert edge[0] < 0
