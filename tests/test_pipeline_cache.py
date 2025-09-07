from __future__ import annotations

import numpy as np

from tests._utils.dummies import install as install_dummies

# What this tests (TEST_PLAN.md §Pipeline/Cache)
# - Per-pipeline single-layer LRU cache: same (pipeline,input) returns identical object.
# - With maxsize=1, a different input geometry evicts the previous key.
# - Fallback hashing works when Geometry.digest is disabled.


def _geom_n(n: int = 10):
    from engine.core.geometry import Geometry

    pts = np.stack([np.linspace(0.0, 1.0, n), np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    return Geometry.from_lines([pts])


def test_cache_hit_and_lru_evict():
    install_dummies()
    from api import E

    g = _geom_n(8)
    pipe = E.pipeline.cache(maxsize=1).rotate(angles_rad=(0.0, 0.0, 0.1)).build()

    a = pipe(g)
    b = pipe(g)
    assert a is b  # cache hit returns same object identity

    # Same pipeline, different input geometry should evict previous when maxsize=1
    g2 = _geom_n(9)
    _ = pipe(g2)  # fills the only slot, evicting the previous (g, pipe) key
    c = pipe(g)
    assert c is not a  # previous entry evicted in LRU with maxsize=1


def test_cache_fallback_when_digest_disabled(monkeypatch):
    install_dummies()
    from api import E

    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "1")
    g = _geom_n(6)
    pipe = E.pipeline.cache(maxsize=2).rotate(angles_rad=(0.0, 0.0, 0.3)).build()
    a = pipe(g)
    b = pipe(g)
    assert a is b  # still hits cache via fallback geometry hashing
