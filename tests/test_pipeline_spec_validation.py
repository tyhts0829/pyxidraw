import pytest

from api.pipeline import validate_spec, from_spec


def test_validate_spec_accepts_valid_spec():
    spec = [
        {"name": "translate", "params": {"offset_x": 1.0}},
        {"name": "rotate", "params": {"center": (0.0, 0.0, 0.0), "rotate": (0.25, 0.0, 0.0)}},
    ]
    # Should not raise
    validate_spec(spec)


def test_validate_spec_rejects_unknown_effect():
    with pytest.raises(KeyError):
        validate_spec([{"name": "__not_registered__", "params": {}}])


def test_validate_spec_rejects_non_dict_params():
    with pytest.raises(TypeError):
        validate_spec([{"name": "translate", "params": 42}])  # type: ignore[list-item]


def test_validate_spec_rejects_non_json_like_param():
    class _X:
        pass

    with pytest.raises(TypeError):
        validate_spec([{"name": "translate", "params": {"bad": _X()}}])


def test_validate_spec_unknown_param_when_no_kwargs():
    # effects.rotate doesn't accept arbitrary kwargs
    with pytest.raises(TypeError):
        validate_spec([{"name": "rotate", "params": {"unknown": 1}}])


def test_from_spec_uses_validate_spec():
    # from_spec should propagate the same validation error
    with pytest.raises(TypeError):
        from_spec([{"name": "rotate", "params": {"unknown": 1}}])
