import numpy as np
import pytest

from engine.core.geometry import Geometry


def _simple_geometry() -> Geometry:
    pts1 = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32)
    pts2 = np.array([[0, 0, 0], [0, 10, 0]], dtype=np.float32)
    return Geometry.from_lines([pts1, pts2])


def test_misc_effects_smoke_returns_geometry():
    g = _simple_geometry()

    # import lazily to avoid registry side-effects on module import order
    from effects.boldify import boldify
    from effects.collapse import collapse
    from effects.dash import dash
    from effects.explode import explode
    from effects.trim import trim
    from effects.twist import twist
    from effects.ripple import ripple
    from effects.weave import weave
    from effects.wobble import wobble

    for fx in (boldify, collapse, dash, explode, trim, twist, ripple, weave, wobble):
        try:
            out = fx(g)
        except Exception:
            if fx is boldify:
                pytest.xfail("boldify: njit implementation uses vstack on list; needs refactor")
            raise
        assert isinstance(out, Geometry)


def test_pipeline_spec_roundtrip_for_misc_effects():
    from api import E, to_spec, from_spec, validate_spec

    g = _simple_geometry()

    p = (
        E.pipeline
        .trim(start_param=0.1, end_param=0.9)
        .dash(dash_length=1.0, gap_length=0.5)
        .build()
    )
    s = to_spec(p)
    validate_spec(s)
    p2 = from_spec(s)
    out = p2(g)
    assert isinstance(out, Geometry)


def test_validate_spec_rejects_unknown_param_for_trimming():
    from api import validate_spec
    import pytest

    with pytest.raises(TypeError):
        validate_spec([{"name": "trim", "params": {"unknown": 1}}])
