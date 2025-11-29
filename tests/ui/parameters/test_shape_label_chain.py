from __future__ import annotations

from api import G
from engine.ui.parameters.runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from engine.ui.parameters.state import ParameterLayoutConfig, ParameterStore


def _first_text_descriptor(store: ParameterStore) -> tuple[str, str]:
    """`shape.text#0.*` に対応する最初の Descriptor の (id, category) を返す。"""
    for desc in store.descriptors():
        if desc.source != "shape":
            continue
        if str(desc.id).startswith("shape.text#0."):
            return str(desc.id), str(desc.category)
    raise AssertionError("shape.text#0.* descriptor not found")


def test_shape_label_via_shapesapi_label():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    try:
        runtime.begin_frame()
        # `G.label("title").text(...)` パス
        G.label("title").text()

        desc_id, category = _first_text_descriptor(store)
        assert desc_id.startswith("shape.text#0.")
        assert category == "title"
    finally:
        deactivate_runtime()


def test_shape_label_via_lazygeometry_label():
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    try:
        runtime.begin_frame()
        # `G.text(...).label("title")` パス
        G.text().label("title")

        desc_id, category = _first_text_descriptor(store)
        assert desc_id.startswith("shape.text#0.")
        assert category == "title"
    finally:
        deactivate_runtime()
