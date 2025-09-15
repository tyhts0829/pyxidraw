from __future__ import annotations

import sys

import pytest


@pytest.mark.smoke
def test_asemic_glyph_is_registered_and_generates_geometry():
    from api import G
    from engine.core.geometry import Geometry
    from shapes.registry import is_shape_registered

    assert is_shape_registered("asemic_glyph")
    # 既定領域でスモーク
    g = G.asemic_glyph(random_seed=1)
    assert isinstance(g, Geometry)
    # 何らかの線分が得られる（空でも許容したい場合は is_empty を緩める）
    assert not g.is_empty


def test_asemic_glyph_fallback_without_scipy(monkeypatch):
    import shapes.asemic_glyph as ag
    from api import G

    # SciPy を見えないようにして ImportError を誘発
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)

    # 生成ノードを小さく安定化するため、generate_nodes をモック
    def _mock_generate_nodes(region, cell_margin, placement_mode, config):
        return [(-0.05, -0.05, 0.0), (0.05, -0.05, 0.0), (0.05, 0.05, 0.0), (-0.05, 0.05, 0.0)]

    monkeypatch.setattr(ag, "generate_nodes", _mock_generate_nodes)

    g = G.asemic_glyph(region=(-0.1, -0.1, 0.1, 0.1), random_seed=2)
    assert not g.is_empty
