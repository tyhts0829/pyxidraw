from __future__ import annotations

import pytest

from api import E
from api.pipeline import Pipeline, _is_json_like, from_spec, to_spec, validate_spec
from engine.core.geometry import Geometry


def test_pipeline_to_from_spec_roundtrip() -> None:
    p = (
        E.pipeline.rotate(pivot=(1.0, 2.0, 3.0), angles_rad=(0.1, 0.2, 0.3))
        .displace(amplitude_mm=0.0, spatial_freq=0.5, t_sec=0.0)
        .build()
    )
    spec = to_spec(p)
    p2 = from_spec(spec)
    assert isinstance(p2, Pipeline)
    assert to_spec(p2) == spec


def test_validate_spec_param_meta_min_max_choices_and_type() -> None:
    # rotate has Vec3 types enforced by validate_spec; offset has choices on join; displace has min for amplitude
    ok = [
        {"name": "rotate", "params": {"pivot": (0.0, 0.0, 0.0), "angles_rad": (0.0, 0.0, 0.0)}},
        {"name": "offset", "params": {"join": "round", "segments_per_circle": 12, "distance": 0.1}},
        {"name": "displace", "params": {"amplitude_mm": 0.1, "spatial_freq": 0.5, "t_sec": 0.0}},
    ]
    validate_spec(ok)

    # choices
    bad_choice = [
        {"name": "offset", "params": {"join": "foo", "segments_per_circle": 12, "distance": 0.1}}
    ]
    with pytest.raises(TypeError):
        validate_spec(bad_choice)

    # min
    bad_min = [{"name": "displace", "params": {"amplitude_mm": -1.0}}]
    with pytest.raises(TypeError):
        validate_spec(bad_min)


def test_pipeline_cache_none_and_clear_cache_behavior() -> None:
    # maxsize=None（無制限） + clear_cache の挙動
    g = Geometry.from_lines([[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]])
    p = E.pipeline.cache(maxsize=None).rotate(angles_rad=(0.0, 0.0, 0.0)).build()
    a = p(g)
    b = p(g)
    assert a is b  # ヒット
    p.clear_cache()
    c = p(g)
    assert c is not b


def test__is_json_like_boundaries() -> None:
    assert _is_json_like({"a": [1, 2.0, "x", True, None]})
    assert _is_json_like(((1, 2), [3, 4]))
    assert not _is_json_like(set([1, 2]))

    class X:  # noqa: D401 - テスト用
        pass

    assert not _is_json_like(X())
