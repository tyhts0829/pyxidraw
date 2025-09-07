import pytest

from api.pipeline import validate_spec

# What this tests (TEST_PLAN.md §Pipeline/Strict & Spec)
# - __param_meta__ に基づく型/範囲エラー（translate.delta に文字列、fill.density 範囲外）。
# - 既存エフェクトへの未知パラメータで TypeError、未登録エフェクト名で KeyError。



def test_validate_spec_rejects_type_mismatch_using_param_meta():
    # translate.delta expects a number/1-tuple/3-tuple (vec3). A string is invalid.
    spec = [
        {"name": "translate", "params": {"delta": "abc"}},
    ]
    with pytest.raises(TypeError):
        validate_spec(spec)


def test_validate_spec_unknown_param_on_existing_effect_type_error():
    # rotate does not accept 'bogus' parameter – should trigger TypeError via validate_spec
    spec = [
        {"name": "rotate", "params": {"bogus": 1}},
    ]
    import pytest

    with pytest.raises(TypeError):
        validate_spec(spec)


def test_validate_spec_unknown_effect_key_error():
    # effect name not registered -> KeyError
    spec = [
        {"name": "not_registered_effect", "params": {}},
    ]
    import pytest

    with pytest.raises(KeyError):
        validate_spec(spec)


def test_validate_spec_fill_density_out_of_range_type_error():
    # density must be in [0,1]
    import pytest

    for bad in (-0.01, 1.01):
        spec = [{"name": "fill", "params": {"density": bad}}]
        with pytest.raises(TypeError):
            validate_spec(spec)
