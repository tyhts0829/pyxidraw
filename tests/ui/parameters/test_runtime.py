from typing import Mapping

from engine.ui.parameters.runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from engine.ui.parameters.state import ParameterLayoutConfig, ParameterStore


def dummy_shape(radius: float = 1.0) -> Mapping[str, float]:
    return {"radius": radius}


def dummy_effect(g, *, amplitude_mm: float = 0.5):  # noqa: ANN001
    return g


def dummy_shape_with_defaults(radius: float = 1.0, segments: int = 12) -> Mapping[str, float]:
    return {"radius": radius, "segments": float(segments)}


def dummy_effect_with_defaults(
    g, *, amplitude_mm: float = 0.5, enable: bool = True
):  # noqa: ANN001
    return g


dummy_effect_with_defaults.__param_meta__ = {
    "amplitude_mm": {"min": 0.0, "max": 5.0, "step": 0.5},
    "enable": {"min": 0, "max": 1, "step": 1},
}


def test_parameter_runtime_tracks_shape_overrides():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()
    params = {"radius": 2.0}
    updated = runtime.before_shape_call("circle", dummy_shape, params)
    assert updated["radius"] == 2.0
    descriptor_id = "shape.circle#0.radius"
    store.set_override(descriptor_id, 0.5)

    runtime.begin_frame()
    updated = runtime.before_shape_call("circle", dummy_shape, params)
    assert updated["radius"] == 0.5
    deactivate_runtime()


def test_parameter_runtime_handles_effect_vectors():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()
    params = {"angles_rad": (0.1, 0.2, 0.3)}
    updated = runtime.before_effect_call(
        step_index=0,
        effect_name="rotate",
        fn=dummy_effect,
        params=params,
    )
    assert updated["angles_rad"] == (0.1, 0.2, 0.3)
    descriptor_ids = {desc.id for desc in store.descriptors()}
    assert "effect.rotate#0.angles_rad.x" in descriptor_ids
    deactivate_runtime()


def test_parameter_runtime_registers_default_shape_parameters():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()

    updated = runtime.before_shape_call("capsule", dummy_shape_with_defaults, {})

    assert updated["radius"] == 1.0
    assert updated["segments"] == 12
    descriptor_ids = {desc.id for desc in store.descriptors()}
    assert "shape.capsule#0.radius" in descriptor_ids
    assert "shape.capsule#0.segments" in descriptor_ids
    deactivate_runtime()


def test_parameter_runtime_registers_default_effect_parameters():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()

    updated = runtime.before_effect_call(
        step_index=0,
        effect_name="displace",
        fn=dummy_effect_with_defaults,
        params={},
    )

    assert updated["amplitude_mm"] == 0.5
    assert updated["enable"] is True
    descriptor_ids = {desc.id for desc in store.descriptors()}
    assert "effect.displace#0.amplitude_mm" in descriptor_ids
    assert "effect.displace#0.enable" in descriptor_ids
    deactivate_runtime()


def test_parameter_runtime_applies_meta_range_hints():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()

    runtime.before_effect_call(
        step_index=0,
        effect_name="displace",
        fn=dummy_effect_with_defaults,
        params={},
    )

    descriptor = store.get_descriptor("effect.displace#0.amplitude_mm")
    assert descriptor.range_hint is not None
    hint = descriptor.range_hint
    assert hint.min_value == 0.0
    assert hint.max_value == 1.0
    assert hint.mapped_min == 0.0
    assert hint.mapped_max == 5.0
    assert hint.mapped_step == 0.5
    deactivate_runtime()


def test_parameter_runtime_uses_fallback_range_for_missing_meta():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()

    runtime.before_shape_call("capsule", dummy_shape_with_defaults, {})

    descriptor = store.get_descriptor("shape.capsule#0.radius")
    assert descriptor.range_hint is not None
    assert descriptor.range_hint.min_value == 0.0
    assert descriptor.range_hint.max_value == 1.0
    deactivate_runtime()
