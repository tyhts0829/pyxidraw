from __future__ import annotations

from api import E
from engine.ui.parameters.runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from engine.ui.parameters.state import ParameterLayoutConfig, ParameterStore


def test_E_label_postfix_updates_pipeline_category() -> None:
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    runtime.begin_frame()

    builder = E.rotate(rotation=(0.0, 0.0, 0.0))
    # Runtime が有効な状態で effect を 1 つ追加したので UID が振られている
    descriptors_before = {d.id: d for d in store.descriptors()}
    assert any(
        d.source == "effect" and d.category for d in descriptors_before.values()
    ), "effect descriptor should be registered before relabel"

    builder = builder.label("poly_effect")

    descriptors_after = {d.id: d for d in store.descriptors()}
    effect_descs = [d for d in descriptors_after.values() if d.source == "effect"]
    assert effect_descs, "effect descriptors should exist after relabel"
    categories = {d.category for d in effect_descs}
    assert any(c.startswith("poly_effect") for c in categories)

    deactivate_runtime()
