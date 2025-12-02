import inspect

import pytest

from engine.ui.parameters.state import ParameterStore
from engine.ui.parameters.snapshot import extract_overrides
from engine.ui.parameters.value_resolver import ParameterContext, ParameterValueResolver


def effect_fn(g, *, amplitude_mm: float = 0.5, enable: bool = True):  # noqa: ANN001
    return g


def vector_effect(g, *, rotation=(0.0, 0.0, 0.0)):  # noqa: ANN001,B006
    return g


def test_parameter_value_resolver_registers_scalars_with_meta():
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="displace", index=0)
    signature = inspect.signature(effect_fn)
    param_meta = {
        "amplitude_mm": {"min": 0.0, "max": 5.0, "step": 0.5},
        "enable": {"min": 0, "max": 1, "step": 1},
    }

    resolved = resolver.resolve(
        context=context,
        params={},
        signature=signature,
        doc="Effect doc",
        param_meta=param_meta,
        skip={"g"},
    )

    assert resolved["amplitude_mm"] == 0.5
    descriptor = store.get_descriptor("effect.displace#0.amplitude_mm")
    assert descriptor.range_hint is not None
    hint = descriptor.range_hint
    assert hint.min_value == 0.0
    assert hint.max_value == 5.0
    assert hint.step == 0.5
    assert descriptor.help_text == "Effect doc"


def test_parameter_value_resolver_passes_through_provided_actual_value():
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="displace", index=0)
    signature = inspect.signature(effect_fn)
    param_meta = {
        "amplitude_mm": {"min": 0.0, "max": 50.0, "step": 0.5},
    }

    resolved = resolver.resolve(
        context=context,
        params={"amplitude_mm": 0.5},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )
    assert resolved["amplitude_mm"] == pytest.approx(0.5)
    # provided 値は GUI 対象外のため Store には登録されない
    stored = store.original_value("effect.displace#0.amplitude_mm")
    assert stored is None


def test_parameter_value_resolver_passes_through_large_actual_value():
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="displace", index=0)
    signature = inspect.signature(effect_fn)
    param_meta = {
        "amplitude_mm": {"min": 0.0, "max": 50.0, "step": 0.5},
    }

    resolved = resolver.resolve(
        context=context,
        params={"amplitude_mm": 2.5},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )
    assert resolved["amplitude_mm"] == pytest.approx(2.5)


def test_parameter_value_resolver_handles_vector_params_defaults_register_gui():
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="rotate", index=1)
    signature = inspect.signature(vector_effect)

    resolved = resolver.resolve(
        context=context,
        params={},
        signature=signature,
        doc=None,
        param_meta={"rotation": {"min": (-1.0, -1.0, -1.0), "max": (1.0, 1.0, 1.0)}},
        skip={"g"},
    )

    # 既定値採用のため GUI 登録され、親ベクトル Descriptor が 1 件作成される
    assert resolved["rotation"] == pytest.approx((0.0, 0.0, 0.0))
    descriptor_ids = {desc.id for desc in store.descriptors()}
    assert "effect.rotate#1.rotation" in descriptor_ids
    # value_type は vector、vector_hint(min/max) が設定される（メタがあれば）
    d = store.get_descriptor("effect.rotate#1.rotation")
    assert d.value_type == "vector"
    assert d.vector_hint is not None


def test_parameter_context_category_for_shapes_uses_index_suffix():
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    signature = inspect.signature(effect_fn)
    param_meta = {"amplitude_mm": {"min": 0.0, "max": 1.0, "step": 0.1}}

    # 1 回目の shape 呼び出し（index=0）は無印カテゴリ名
    ctx0 = ParameterContext(scope="shape", name="text", index=0)
    resolver.resolve(
        context=ctx0,
        params={},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )
    desc0 = store.get_descriptor("shape.text#0.amplitude_mm")
    assert desc0.category == "text"

    # 2 回目の shape 呼び出し（index=1）は `text_1`
    ctx1 = ParameterContext(scope="shape", name="text", index=1)
    resolver.resolve(
        context=ctx1,
        params={},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )
    desc1 = store.get_descriptor("shape.text#1.amplitude_mm")
    assert desc1.category == "text_1"


def test_cc_scaling_uses_range_override_scalar():
    store = ParameterStore()
    store.set_cc_provider(lambda: {5: 0.5})
    store.bind_cc("effect.displace#0.amplitude_mm", 5)
    store.set_range_override("effect.displace#0.amplitude_mm", 10.0, 20.0)

    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="displace", index=0)
    signature = inspect.signature(effect_fn)
    param_meta = {"amplitude_mm": {"min": 0.0, "max": 1.0, "step": 0.1}}

    resolved = resolver.resolve(
        context=context,
        params={},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )

    assert resolved["amplitude_mm"] == pytest.approx(15.0)


def test_cc_scaling_uses_range_override_vector_component():
    store = ParameterStore()
    store.set_cc_provider(lambda: {3: 0.25})
    store.bind_cc("effect.rotate#1.rotation::x", 3)
    store.set_range_override("effect.rotate#1.rotation", -2.0, 2.0)

    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="rotate", index=1)
    signature = inspect.signature(vector_effect)
    param_meta = {"rotation": {"min": (-1.0, -1.0, -1.0), "max": (1.0, 1.0, 1.0)}}

    resolved = resolver.resolve(
        context=context,
        params={},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )

    assert resolved["rotation"] == pytest.approx((-1.0, 0.0, 0.0))


def test_extract_overrides_applies_range_override_for_cc():
    store = ParameterStore()
    store.bind_cc("effect.displace#0.amplitude_mm", 7)
    store.set_range_override("effect.displace#0.amplitude_mm", 10.0, 20.0)

    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="displace", index=0)
    signature = inspect.signature(effect_fn)
    param_meta = {"amplitude_mm": {"min": 0.0, "max": 1.0, "step": 0.1}}

    resolver.resolve(
        context=context,
        params={},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )

    overrides = extract_overrides(store, {7: 0.5})

    assert overrides["effect.displace#0.amplitude_mm"] == pytest.approx(15.0)
