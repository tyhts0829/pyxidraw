import pytest

from api.pipeline import validate_spec


def test_vec3_accepts_scalar_and_tuple():
    # translate.delta as scalar
    validate_spec([
        {"name": "translate", "params": {"delta": 1.0}},
    ])

    # rotate.angles_rad as 1-tuple
    validate_spec([
        {"name": "rotate", "params": {"angles_rad": (0.1,)}},
    ])

    # scale.scale as 3-tuple
    validate_spec([
        {"name": "scale", "params": {"scale": (1.0, 2.0, 3.0)}},
    ])


def test_vec3_rejects_invalid():
    with pytest.raises(TypeError):
        validate_spec([{ "name": "translate", "params": {"delta": (1.0, 2.0)} }])

