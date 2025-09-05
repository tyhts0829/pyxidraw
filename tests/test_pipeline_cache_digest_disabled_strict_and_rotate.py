import os

import numpy as np

from api import E
from engine.core.geometry import Geometry


def _geom_nonempty():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    return Geometry.from_lines([pts])


def _geom_empty():
    return Geometry.from_lines([])


def test_cache_works_when_digest_disabled_and_rotate_early_return(monkeypatch):
    # digest disabled should not break pipeline caching
    monkeypatch.delenv("PXD_PIPELINE_CACHE_MAXSIZE", raising=False)
    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "1")
    g = _geom_nonempty()

    # angles_rad=(0,0,0) triggers Geometry.rotate early-return branch
    p = (E.pipeline.rotate(angles_rad=(0.0, 0.0, 0.0)).build())
    out1 = p(g)
    out2 = p(g)
    assert out1 is out2


def test_cache_works_when_digest_disabled_and_scale_on_empty_geometry(monkeypatch):
    # scale() early-return path for empty geometry should also be safe
    monkeypatch.delenv("PXD_PIPELINE_CACHE_MAXSIZE", raising=False)
    monkeypatch.setenv("PXD_DISABLE_GEOMETRY_DIGEST", "1")
    g = _geom_empty()
    p = (E.pipeline.scale(scale=(1.0, 1.0, 1.0)).build())
    out1 = p(g)
    out2 = p(g)
    assert out1 is out2


def test_pipeline_builder_strict_param_validation_raises():
    # strict mode should detect unknown parameter names before execution
    builder = E.pipeline.strict(True)
    try:
        builder.translate(detz=(1.0, 2.0, 3.0)).build()
        assert False, "strict mode should have raised TypeError"
    except TypeError as e:
        assert "unknown params" in str(e)

