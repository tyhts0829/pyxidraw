from __future__ import annotations

import numpy as np

from api import E
from engine.core.geometry import Geometry


def _geom_small() -> Geometry:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_pipeline_cache_hits_for_same_input_and_pipeline() -> None:
    g = _geom_small()
    p = E.pipeline.rotate(angles_rad=(0.0, 0.0, 0.0)).build()
    a = p(g)
    b = p(g)
    assert a is b  # 単層キャッシュヒット


def test_pipeline_cache_hits_for_same_input_and_pipeline_from_E_direct() -> None:
    g = _geom_small()
    p = E.rotate(angles_rad=(0.0, 0.0, 0.0)).build()
    a = p(g)
    b = p(g)
    assert a is b  # 単層キャッシュヒット


def test_pipeline_cache_disabled_with_maxsize_zero() -> None:
    g = _geom_small()
    p = E.pipeline.cache(maxsize=0).rotate(angles_rad=(0.0, 0.0, 0.0)).build()
    a = p(g)
    b = p(g)
    assert a is not b


def test_pipeline_cache_disabled_with_maxsize_zero_from_E_direct() -> None:
    g = _geom_small()
    p = E.rotate(angles_rad=(0.0, 0.0, 0.0)).cache(maxsize=0).build()
    a = p(g)
    b = p(g)
    assert a is not b


def test_pipeline_cache_lru_eviction_with_maxsize_one() -> None:
    g1 = _geom_small()
    g2 = g1.translate(1.0, 0.0, 0.0)
    p = E.pipeline.cache(maxsize=1).rotate(angles_rad=(0.0, 0.0, 0.0)).build()
    a1 = p(g1)
    _ = p(g2)  # cache エントリを g2 で更新（未使用）
    a1_again = p(g1)
    assert a1_again is not a1  # g2 が入って追い出される


def test_pipeline_cache_lru_eviction_with_maxsize_one_from_E_direct() -> None:
    g1 = _geom_small()
    g2 = g1.translate(1.0, 0.0, 0.0)
    p = E.rotate(angles_rad=(0.0, 0.0, 0.0)).cache(maxsize=1).build()
    a1 = p(g1)
    _ = p(g2)
    a1_again = p(g1)
    assert a1_again is not a1
