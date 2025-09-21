from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
    RangeHint,
)


def test_parameter_store_resolve_and_override():
    store = ParameterStore()
    descriptor = ParameterDescriptor(
        id="shape.sphere#0.subdivisions",
        label="sphere · subdivisions",
        source="shape",
        category="shape",
        value_type="float",
        default_value=0.5,
        range_hint=RangeHint(0.0, 1.0),
    )
    store.register(descriptor, 0.5)
    assert store.current_value(descriptor.id) == 0.5

    store.set_override(descriptor.id, 0.8)
    assert store.current_value(descriptor.id) == 0.8

    # 仕様: ストアはクランプせず実値をそのまま保持する
    result = store.set_override(descriptor.id, 5.0)
    assert result.clamped is False
    assert store.current_value(descriptor.id) == 5.0

    store.clear_override(descriptor.id)
    assert store.current_value(descriptor.id) == 0.5


def test_layout_config_derives_positive_range():
    layout = ParameterLayoutConfig(default_range_multiplier=1.0)
    hint = layout.derive_range(name="scale", value_type="float", default_value=0.0)
    assert hint.min_value == 0.0
    assert hint.max_value == 1.0

    hint_int = layout.derive_range(name="count", value_type="int", default_value=4)
    assert hint_int.min_value <= 4 <= hint_int.max_value
