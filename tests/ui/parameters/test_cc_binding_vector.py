from engine.ui.parameters.cc_binding import CC
from engine.ui.parameters.runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from engine.ui.parameters.state import ParameterLayoutConfig, ParameterStore


def vector_effect(g, *, scale=(1.0, 1.0, 1.0)):
    return g


def test_cc_binding_vector_component_overrides_gui():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()

    # 初回: CCBinding を含むベクトルを解決（scale.z が CC(2)）
    params = {"scale": (1.0, 1.0, CC(2))}
    # cc=0.3 を想定
    runtime.set_inputs(0.0, {2: 0.3})
    updated = runtime.before_effect_call(
        step_index=0,
        effect_name="scale",
        fn=vector_effect,
        params=params,
    )
    assert updated["scale"][2] == 0.3

    # GUI が z=0.8 を設定
    pid_z = "effect.scale#0.scale.z"
    store.set_override(pid_z, 0.8, source="gui")

    # 次フレーム: CC を 0.1 に変更（midi が優先されること）
    runtime.begin_frame()
    runtime.set_inputs(0.0, {2: 0.1})
    updated2 = runtime.before_effect_call(
        step_index=0,
        effect_name="scale",
        fn=vector_effect,
        params=params,
    )
    assert updated2["scale"][2] == 0.1

    deactivate_runtime()
