from __future__ import annotations

import numpy as np
import pytest

from tests._utils.dummies import install as install_dummies

# What this tests (TEST_PLAN.md §Pipeline/Strict & Spec)
# - strict(True): 既存エフェクトへの未知パラメータで TypeError。
# - validate_spec: JSON性/type/range/choices の検証、未登録エフェクト名で KeyError。
# - to_spec/from_spec の往復等価性（適用結果の数値一致）。


def _simple_geom():
    from engine.core.geometry import Geometry

    return Geometry.from_lines([np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)])


def test_strict_unknown_param_raises_typeerror():
    install_dummies()
    from api import E

    with pytest.raises(TypeError, match="unknown params"):
        _ = E.pipeline.strict(True).rotate(bogus=1).build()


def test_validate_spec_types_ranges_choices_and_unknown_name():
    install_dummies()
    from api import validate_spec

    # valid spec
    spec_ok = [
        {"name": "rotate", "params": {"pivot": (0.0, 0.0, 0.0), "angles_rad": (0.0, 0.0, 0.1)}}
    ]
    validate_spec(spec_ok)

    # JSON-like: numpy array is rejected
    spec_bad_json = [{"name": "rotate", "params": {"angles_rad": np.array([0.0, 0.0, 0.0])}}]
    with pytest.raises(TypeError):
        validate_spec(spec_bad_json)

    # range/type via param_meta: negative amplitude_mm -> TypeError
    spec_bad_range = [{"name": "displace", "params": {"amplitude_mm": -0.1, "t_sec": 0.0}}]
    with pytest.raises(TypeError):
        validate_spec(spec_bad_range)

    # choices via param_meta: invalid fill mode -> TypeError
    spec_bad_choice = [{"name": "fill", "params": {"mode": "invalid", "density": 0.1}}]
    with pytest.raises(TypeError):
        validate_spec(spec_bad_choice)

    # unknown effect name -> KeyError
    with pytest.raises(KeyError):
        validate_spec([{"name": "no_such_effect", "params": {}}])


def test_to_spec_from_spec_roundtrip_equivalence():
    install_dummies()
    from api import E, from_spec, to_spec

    g = _simple_geom()
    p = (
        E.pipeline.translate(delta=(1.0, 2.0, 0.0))
        .rotate(pivot=(0.0, 0.0, 0.0), angles_rad=(0.0, 0.0, 0.5))
        .build()
    )
    s = to_spec(p)
    p2 = from_spec(s)
    # Apply both and compare numerically
    out1 = p(g)
    out2 = p2(g)
    c1, o1 = out1.as_arrays(copy=False)
    c2, o2 = out2.as_arrays(copy=False)
    assert np.allclose(c1, c2, rtol=1e-6, atol=1e-6)
    assert np.array_equal(o1, o2)
