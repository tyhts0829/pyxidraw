import inspect

import pytest

from engine.ui.parameters.state import ParameterStore
from engine.ui.parameters.value_resolver import ParameterContext, ParameterValueResolver


def effect_fn(g, *, amplitude_mm: float = 0.5, enable: bool = True):  # noqa: ANN001
    return g


def vector_effect(g, *, angles_rad=(0.0, 0.0, 0.0)):  # noqa: ANN001,B006
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
    assert hint.max_value == 1.0
    assert hint.mapped_max == 5.0
    assert hint.mapped_step == 0.5
    assert descriptor.help_text == "Effect doc"


def test_parameter_value_resolver_denormalizes_provided_normalized_value():
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

    assert resolved["amplitude_mm"] == pytest.approx(25.0)
    stored = store.original_value("effect.displace#0.amplitude_mm")
    assert stored == pytest.approx(0.5)


def test_parameter_value_resolver_uses_normalized_only_and_allows_overscale():
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="displace", index=0)
    signature = inspect.signature(effect_fn)
    param_meta = {
        "amplitude_mm": {"min": 0.0, "max": 50.0, "step": 0.5},
    }

    resolved = resolver.resolve(
        context=context,
        # 仕様: 入力は常に正規化値。>1.0 もオーバースケールとして許容
        params={"amplitude_mm": 2.5},
        signature=signature,
        doc=None,
        param_meta=param_meta,
        skip={"g"},
    )
    # mapped_min=0, mapped_max=50 なので 2.5 → 実値 125.0
    assert resolved["amplitude_mm"] == pytest.approx(125.0)


def test_parameter_value_resolver_handles_vector_params():
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope="effect", name="rotate", index=1)
    signature = inspect.signature(vector_effect)

    resolved = resolver.resolve(
        context=context,
        params={"angles_rad": (0.1, 0.2, 0.3)},
        signature=signature,
        doc=None,
        param_meta={"angles_rad": {"min": (-1.0, -1.0, -1.0), "max": (1.0, 1.0, 1.0)}},
        skip={"g"},
    )

    assert resolved["angles_rad"] == pytest.approx((0.1, 0.2, 0.3))
    descriptor_ids = {desc.id for desc in store.descriptors()}
    assert "effect.rotate#1.angles_rad.x" in descriptor_ids
    assert "effect.rotate#1.angles_rad.z" in descriptor_ids
