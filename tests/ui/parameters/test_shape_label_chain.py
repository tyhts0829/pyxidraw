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


def _categories_for_text_indices(store: ParameterStore) -> dict[int, str]:
    """text shape の index ごとのカテゴリ名を返す。"""
    result: dict[int, str] = {}
    for desc in store.descriptors():
        if desc.source != "shape":
            continue
        id_str = str(desc.id)
        if not id_str.startswith("shape.text#"):
            continue
        try:
            _, rest = id_str.split("shape.", 1)
            name_part, _ = rest.split(".", 1)
            _, idx_str = name_part.split("#", 1)
            idx_val = int(idx_str)
        except Exception:
            continue
        result[idx_val] = str(desc.category)
    return result


def test_shape_label_multiple_labels_are_separated():
    """同一 shape 名に対する複数ラベルが別カテゴリになる。"""
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    try:
        runtime.begin_frame()
        # `G.label` パス
        G.label("v_text").text()
        G.label("h_text").text()

        cats = _categories_for_text_indices(store)
        assert cats[0] == "v_text"
        assert cats[1] == "h_text"
    finally:
        deactivate_runtime()


def test_shape_label_multiple_labels_via_lazygeometry_label():
    """`G.text().label(...)` パスでも複数ラベルが別カテゴリになる。"""
    store = ParameterStore()
    runtime = ParameterRuntime(store, layout=ParameterLayoutConfig())
    activate_runtime(runtime)
    try:
        runtime.begin_frame()
        G.text().label("v_text")
        G.text().label("h_text")

        cats = _categories_for_text_indices(store)
        assert cats[0] == "v_text"
        assert cats[1] == "h_text"
    finally:
        deactivate_runtime()
