from typing import Mapping

from engine.ui.parameters.runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from engine.ui.parameters.state import ParameterLayoutConfig, ParameterStore


def dummy_shape(radius: float = 1.0) -> Mapping[str, float]:
    return {"radius": radius}


def dummy_effect(g, *, amplitude_mm: float = 0.5):  # noqa: ANN001
    return g


def test_parameter_runtime_tracks_shape_overrides():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()
    params = {"radius": 2.0}
    updated = runtime.before_shape_call("circle", dummy_shape, params)
    assert updated["radius"] == 2.0
    descriptor_id = "shape.circle#0.radius"
    store.set_override(descriptor_id, 1.5)

    runtime.begin_frame()
    updated = runtime.before_shape_call("circle", dummy_shape, params)
    assert updated["radius"] == 1.5
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
