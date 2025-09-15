from __future__ import annotations

import pytest

from api import E
from api.effects import validate_spec


def test_strict_unknown_param_raises() -> None:
    builder = E.pipeline  # strict=True が既定
    builder = builder.rotate(bad_param=123)  # type: ignore[call-arg]
    with pytest.raises(TypeError) as ei:
        builder.build()
    assert "unknown params" in str(ei.value)
    assert "Allowed:" in str(ei.value)


def test_validate_spec_allows_known_params_and_rejects_unknown() -> None:
    ok = [{"name": "rotate", "params": {"pivot": (0.0, 0.0, 0.0), "angles_rad": (0.0, 0.0, 0.0)}}]
    validate_spec(ok)

    bad = [{"name": "rotate", "params": {"bogus": 1}}]
    with pytest.raises(TypeError):
        validate_spec(bad)


def test_validate_spec_rejects_non_json_like_values() -> None:
    bad = [{"name": "rotate", "params": {"pivot": {1, 2, 3}}}]  # set は不可
    with pytest.raises(TypeError):
        validate_spec(bad)
