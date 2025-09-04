import os

import numpy as np

from api import E
from engine.core.geometry import Geometry


def _geom():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def test_cache_disabled_via_env(monkeypatch):
    monkeypatch.setenv("PXD_PIPELINE_CACHE_MAXSIZE", "0")
    g = _geom()
    p = (E.pipeline.rotate(angles_rad=(0.0, 0.0, 1.5707963267948966)).build())
    out1 = p(g)
    out2 = p(g)
    assert out1 is not out2  # cache disabled


def test_cache_works_without_digest(monkeypatch):
    # digest disabled should still permit caching (fallback hash path)
    monkeypatch.delenv("PXD_PIPELINE_CACHE_MAXSIZE", raising=False)
    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "1")
    g = _geom()
    p = (E.pipeline.rotate(angles_rad=(0.0, 0.0, 1.5707963267948966)).build())
    out1 = p(g)
    out2 = p(g)
    assert out1 is out2

