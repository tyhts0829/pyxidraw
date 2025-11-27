from __future__ import annotations

import numpy as np

from engine.core.geometry import Geometry
import engine.render.renderer as renderer_mod
from engine.render.renderer import _geometry_to_vertices_indices
from util.constants import PRIMITIVE_RESTART_INDEX


def test_geometry_to_vertices_indices_inserts_primitive_restart() -> None:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([a, b])
    verts, inds = _geometry_to_vertices_indices(g, PRIMITIVE_RESTART_INDEX)
    # N = 5, lines=2 → indices=5+2
    assert len(verts) == 5
    assert len(inds) == 7
    # 1本目の後に PR が入る（位置=2）
    assert inds[2] == PRIMITIVE_RESTART_INDEX
    # 2本目の後に PR が入る（位置=2 + 1 + 3 = 6）
    assert inds[6] == PRIMITIVE_RESTART_INDEX


def test_geometry_to_vertices_indices_uses_lru_cache() -> None:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    g = Geometry.from_lines([a, b])

    # グローバル状態を退避・初期化
    prev_enabled = renderer_mod._INDICES_CACHE_ENABLED
    prev_maxsize = renderer_mod._INDICES_CACHE_MAXSIZE
    prev_debug = renderer_mod._INDICES_DEBUG
    renderer_mod._INDICES_CACHE_ENABLED = True
    renderer_mod._INDICES_CACHE_MAXSIZE = 16
    renderer_mod._INDICES_DEBUG = True
    renderer_mod._INDICES_CACHE.clear()
    renderer_mod._IND_HITS = 0
    renderer_mod._IND_MISSES = 0
    renderer_mod._IND_STORES = 0
    renderer_mod._IND_EVICTS = 0

    try:
        # 1 回目はミスして indices を構築
        _, inds1 = _geometry_to_vertices_indices(g, PRIMITIVE_RESTART_INDEX)
        assert renderer_mod._IND_MISSES == 1
        assert renderer_mod._IND_STORES == 1

        # 2 回目は同じ Geometry/キーでヒットし、同一配列が返る
        _, inds2 = _geometry_to_vertices_indices(g, PRIMITIVE_RESTART_INDEX)
        assert renderer_mod._IND_HITS == 1
        assert inds1 is inds2
    finally:
        # 設定を元に戻す
        renderer_mod._INDICES_CACHE_ENABLED = prev_enabled
        renderer_mod._INDICES_CACHE_MAXSIZE = prev_maxsize
        renderer_mod._INDICES_DEBUG = prev_debug
