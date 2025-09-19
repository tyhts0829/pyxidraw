"""ParameterRuntime から切り出した値解決ユーティリティ。"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterStore,
    RangeHint,
    ValueType,
)

_VECTOR_SUFFIX = ("x", "y", "z", "w")


@dataclass(frozen=True)
class ParameterContext:
    """shape/effect 呼び出し時の識別情報。"""

    scope: str  # "shape" or "effect"
    name: str
    index: int

    @property
    def descriptor_prefix(self) -> str:
        return f"{self.scope}.{self.name}#{self.index}"

    @property
    def label_prefix(self) -> str:
        return f"{self.name}#{self.index}"


class ParameterValueResolver:
    """パラメータ値とメタデータを ParameterStore と同期させる責務を担う。"""

    def __init__(self, store: ParameterStore) -> None:
        self._store = store

    def resolve(
        self,
        *,
        context: ParameterContext,
        params: Mapping[str, Any],
        signature,
        doc: str | None,
        param_meta: Mapping[str, Mapping[str, Any]],
        skip: set[str] | None = None,
    ) -> dict[str, Any]:
        normalized = self._merge_with_defaults(params, signature, skip=skip)
        updated: dict[str, Any] = {}

        for key, value in normalized.items():
            descriptor_id = f"{context.descriptor_prefix}.{key}"
            default_value = self._default_from_signature(signature, key, value)
            meta_entry = param_meta.get(key, {})
            descriptors: list[tuple[ParameterDescriptor, Any, str | None]] = []

            if self._is_vector(value):
                group = descriptor_id
                descriptors.extend(
                    self._build_vector_descriptors(
                        context=context,
                        base_id=descriptor_id,
                        param_name=key,
                        values=list(value),
                        default=default_value,
                        doc=doc,
                        group=group,
                        param_meta=meta_entry,
                    )
                )
            elif self._supported_scalar(value):
                value_type = self._value_type(value)
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{context.label_prefix} · {key}",
                    source=context.scope,
                    category=context.scope,
                    value_type=value_type,
                    default_value=default_value,
                    range_hint=self._range_hint_from_meta(
                        value_type=value_type,
                        meta=meta_entry,
                        component_index=None,
                    ),
                    help_text=doc,
                    vector_group=None,
                )
                descriptors.append((descriptor, value, None))
            else:
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{context.label_prefix} · {key}",
                    source=context.scope,
                    category=context.scope,
                    value_type="float",
                    default_value=default_value,
                    help_text=doc,
                    vector_group=None,
                    supported=False,
                )
                self._store.register(descriptor, value)
                updated[key] = value
                continue

            resolved = self._register_and_resolve(descriptors)
            updated[key] = resolved

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

        resolved_components: list[Any] = []
        for descriptor, value, _ in descriptors:
            self._store.register(descriptor, value)
            resolved_components.append(self._store.resolve(descriptor.id, value))
        return tuple(resolved_components)

    @staticmethod
    def _merge_with_defaults(
        params: Mapping[str, Any],
        signature,
        *,
        skip: set[str] | None = None,
    ) -> dict[str, Any]:
        if signature is None:
            return dict(params)
        skip = skip or set()
        merged: dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if name in skip:
                continue
            if parameter.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            ):
                continue
            if name in params:
                merged[name] = params[name]
                continue
            if parameter.default is inspect._empty:  # type: ignore[attr-defined]
                continue
            merged[name] = parameter.default
        for name, value in params.items():
            if name in skip or name in merged:
                continue
            merged[name] = value
        return merged

    @staticmethod
    def _default_from_signature(signature, key: str, fallback: Any) -> Any:
        if signature is None:
            return fallback
        parameter = signature.parameters.get(key)
        if parameter is None:
            return fallback
        if parameter.default is inspect._empty:  # type: ignore[attr-defined]
            return fallback
        return parameter.default

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
            return 1 < len(value) <= 4 and all(isinstance(v, (int, float)) for v in value)
        return False

    def _build_vector_descriptors(
        self,
        *,
        context: ParameterContext,
        base_id: str,
        param_name: str,
        values: list[Any],
        default: Any,
        doc: str | None,
        group: str,
        param_meta: Mapping[str, Any],
    ) -> list[tuple[ParameterDescriptor, Any, str | None]]:
        descriptors: list[tuple[ParameterDescriptor, Any, str | None]] = []
        for idx, value in enumerate(values):
            suffix = _VECTOR_SUFFIX[idx] if idx < len(_VECTOR_SUFFIX) else str(idx)
            descriptor_id = f"{base_id}.{suffix}"
            default_component = self._component_default(default, idx, value)
            value_type: ValueType = "float" if isinstance(value, float) else "int"
            descriptor = ParameterDescriptor(
                id=descriptor_id,
                label=f"{context.label_prefix} · {param_name}.{suffix}",
                source=context.scope,
                category=context.scope,
                value_type=value_type,
                default_value=default_component,
                range_hint=self._range_hint_from_meta(
                    value_type=value_type,
                    meta=param_meta,
                    component_index=idx,
                ),
                help_text=doc,
                vector_group=group,
            )
            descriptors.append((descriptor, value, suffix))
        return descriptors

    @staticmethod
    def _component_default(default: Any, idx: int, fallback: Any) -> Any:
        if (
            isinstance(default, Sequence)
            and not isinstance(default, (str, bytes))
            and idx < len(default)
        ):
            component = default[idx]
            if isinstance(component, (int, float)):
                return component
        if isinstance(default, (int, float)):
            return default
        return fallback

    def _range_hint_from_meta(
        self,
        *,
        value_type: ValueType,
        meta: Mapping[str, Any],
        component_index: int | None,
    ) -> RangeHint:
        min_value = self._extract_meta_component(meta.get("min"), component_index)
        max_value = self._extract_meta_component(meta.get("max"), component_index)
        step = self._extract_meta_component(meta.get("step"), component_index)

        default_min, default_max, default_step = self._default_range(value_type)
        if min_value is None:
            min_value = default_min
        if max_value is None:
            max_value = default_max
        if step is None:
            step = default_step

        if value_type == "bool":
            return RangeHint(int(min_value), int(max_value), step=1)
        if value_type == "int":
            return RangeHint(
                int(min_value), int(max_value), step=int(step) if step is not None else 1
            )
        return RangeHint(
            float(min_value), float(max_value), step=float(step) if step is not None else None
        )

    @staticmethod
    def _extract_meta_component(value: Any, component_index: int | None) -> float | int | None:
        if component_index is None:
            if isinstance(value, (int, float)):
                return value
            return None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if component_index < len(value):
                component_value = value[component_index]
                if isinstance(component_value, (int, float)):
                    return component_value
        if isinstance(value, (int, float)):
            return value
        return None

    @staticmethod
    def _default_range(
        value_type: ValueType,
    ) -> tuple[float | int, float | int, float | int | None]:
        if value_type == "bool":
            return 0, 1, 1
        if value_type == "int":
            return 0, 1, 1
        return 0.0, 1.0, None


__all__ = ["ParameterContext", "ParameterValueResolver"]
