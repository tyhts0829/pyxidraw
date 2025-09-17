"""ランタイム側のパラメータ適用ロジック。"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterRegistry,
    ParameterStore,
    ValueType,
)

_ACTIVE_RUNTIME: "ParameterRuntime | None" = None


def activate_runtime(runtime: "ParameterRuntime") -> None:
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = runtime


def deactivate_runtime() -> None:
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = None


def get_active_runtime() -> "ParameterRuntime | None":
    return _ACTIVE_RUNTIME


_VECTOR_SUFFIX = ("x", "y", "z", "w")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


class ParameterRuntime:
    """Shapes/Efffect 呼び出しをフックして ParameterStore を更新する。"""

    def __init__(
        self,
        store: ParameterStore,
        *,
        layout: ParameterLayoutConfig | None = None,
    ) -> None:
        self._store = store
        self._layout = layout or ParameterLayoutConfig()
        self._shape_registry = ParameterRegistry()
        self._effect_registry = ParameterRegistry()
        self._lazy = True
        self._doc_cache: dict[str, str | None] = {}

    def set_lazy(self, lazy: bool) -> None:
        self._lazy = lazy

    # --- フレーム制御 ---
    def begin_frame(self) -> None:
        self._shape_registry.reset()
        self._effect_registry.reset()

    # --- 形状 ---
    def before_shape_call(
        self,
        shape_name: str,
        fn: Any,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        index = self._shape_registry.next_index(shape_name)
        doc = self._doc_cache.get(shape_name)
        if doc is None:
            doc = self._extract_doc(fn)
            self._doc_cache[shape_name] = doc
        sig = self._safe_signature(fn)
        normalized = dict(params)
        updated: dict[str, Any] = {}
        for key, value in normalized.items():
            descriptor_id = f"shape.{shape_name}#{index}.{key}"
            default_value = self._default_from_signature(sig, key, value)
            group: str | None = None
            descriptors: list[tuple[ParameterDescriptor, Any, str | None]] = []
            if self._is_vector(value):
                group = descriptor_id
                seq = list(value)
                desc = self._build_vector_descriptors(
                    base_id=descriptor_id,
                    label_prefix=f"{shape_name}#{index}",
                    param_name=key,
                    values=seq,
                    default=default_value,
                    doc=doc,
                    group=group,
                )
                descriptors.extend(desc)
            elif self._supported_scalar(value):
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{shape_name}#{index} · {key}",
                    source="shape",
                    category="shape",
                    value_type=self._value_type(value),
                    default_value=default_value,
                    range_hint=self._layout.derive_range(
                        name=key,
                        value_type=self._value_type(value),
                        default_value=default_value,
                    ),
                    help_text=doc,
                    vector_group=group,
                )
                descriptors.append((descriptor, value, None))
            else:
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{shape_name}#{index} · {key}",
                    source="shape",
                    category="shape",
                    value_type="float",
                    default_value=default_value,
                    help_text=doc,
                    supported=False,
                )
                self._store.register(descriptor, value)
                updated[key] = value
                continue

            combined_value = self._register_and_resolve(descriptors)
            updated[key] = combined_value
        return updated

    # --- エフェクト ---
    def before_effect_call(
        self,
        *,
        step_index: int,
        effect_name: str,
        fn: Any,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        doc_key = f"effect::{effect_name}"
        doc = self._doc_cache.get(doc_key)
        if doc is None:
            doc = self._extract_doc(fn)
            self._doc_cache[doc_key] = doc
        sig = self._safe_signature(fn)
        normalized = dict(params)
        updated: dict[str, Any] = {}
        for key, value in normalized.items():
            descriptor_id = f"effect.{effect_name}#{step_index}.{key}"
            default_value = self._default_from_signature(sig, key, value)
            group: str | None = None
            descriptors: list[tuple[ParameterDescriptor, Any, str | None]] = []
            if self._is_vector(value):
                group = descriptor_id
                desc = self._build_vector_descriptors(
                    base_id=descriptor_id,
                    label_prefix=f"{effect_name}#{step_index}",
                    param_name=key,
                    values=list(value),
                    default=default_value,
                    doc=doc,
                    group=group,
                )
                descriptors.extend(desc)
            elif self._supported_scalar(value):
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{effect_name}#{step_index} · {key}",
                    source="effect",
                    category="effect",
                    value_type=self._value_type(value),
                    default_value=default_value,
                    range_hint=self._layout.derive_range(
                        name=key,
                        value_type=self._value_type(value),
                        default_value=default_value,
                    ),
                    help_text=doc,
                    vector_group=group,
                )
                descriptors.append((descriptor, value, None))
            else:
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{effect_name}#{step_index} · {key}",
                    source="effect",
                    category="effect",
                    value_type="float",
                    default_value=default_value,
                    help_text=doc,
                    supported=False,
                )
                self._store.register(descriptor, value)
                updated[key] = value
                continue
            combined_value = self._register_and_resolve(descriptors)
            updated[key] = combined_value
        return updated

    def _register_and_resolve(
        self, descriptors: list[tuple[ParameterDescriptor, Any, str | None]]
    ) -> Any:
        if not descriptors:
            return None
        if len(descriptors) == 1 and descriptors[0][2] is None:
            descriptor, value, _ = descriptors[0]
            self._store.register(descriptor, value)
            return self._store.resolve(descriptor.id, value)
        # vector: descriptors contains component descriptors with suffix marker
        resolved_components: list[Any] = []
        for descriptor, value, component in descriptors:
            self._store.register(descriptor, value)
            resolved = self._store.resolve(descriptor.id, value)
            resolved_components.append(resolved)
        # 返却時は元のコンポーネント数に合わせて tuple を構築
        return tuple(resolved_components)

    @staticmethod
    def _extract_doc(fn: Any) -> str | None:
        doc = inspect.getdoc(fn)
        if not doc:
            return None
        return doc.splitlines()[0]

    @staticmethod
    def _safe_signature(fn: Any) -> inspect.Signature | None:
        try:
            return inspect.signature(fn)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _default_from_signature(sig: inspect.Signature | None, key: str, fallback: Any) -> Any:
        if sig is None:
            return fallback
        param = sig.parameters.get(key)
        if param is None:
            return fallback
        if param.default is inspect._empty:  # type: ignore[attr-defined]
            return fallback
        return param.default

    @staticmethod
    def _supported_scalar(value: Any) -> bool:
        return isinstance(value, (int, float, bool))

    @staticmethod
    def _value_type(value: Any) -> ValueType:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int) and not isinstance(value, bool):
            return "int"
        return "float"

    @staticmethod
    def _is_vector(value: Any) -> bool:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if 1 < len(value) <= 4 and all(_is_number(v) for v in value):
                return True
        return False

    def _build_vector_descriptors(
        self,
        *,
        base_id: str,
        label_prefix: str,
        param_name: str,
        values: list[Any],
        default: Any,
        doc: str | None,
        group: str | None,
    ) -> list[tuple[ParameterDescriptor, Any, str | None]]:
        descriptors: list[tuple[ParameterDescriptor, Any, str | None]] = []
        for idx, value in enumerate(values):
            suffix = _VECTOR_SUFFIX[idx] if idx < len(_VECTOR_SUFFIX) else str(idx)
            descriptor_id = f"{base_id}.{suffix}"
            default_value = None
            if isinstance(default, Sequence) and idx < len(default):
                default_value = default[idx]
            elif isinstance(default, (int, float)):
                default_value = default
            else:
                default_value = value
            component_type: ValueType = "float" if isinstance(value, float) else "int"
            descriptor = ParameterDescriptor(
                id=descriptor_id,
                label=f"{label_prefix} · {param_name}.{suffix}",
                source="shape" if base_id.startswith("shape") else "effect",
                category="shape" if base_id.startswith("shape") else "effect",
                value_type=component_type,
                default_value=default_value,
                range_hint=self._layout.derive_range(
                    name=f"{param_name}.{suffix}",
                    value_type=component_type,
                    default_value=default_value,
                ),
                help_text=doc,
                vector_group=group,
            )
            descriptors.append((descriptor, value, suffix))
        return descriptors
