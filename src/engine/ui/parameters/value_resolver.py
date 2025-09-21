"""
値解決ユーティリティ（ParameterRuntime からの独立モジュール）。

どこで・何を・なぜ:
- どこで: `engine.ui.parameters` 層。`ParameterRuntime` から呼ばれる。
- 何を: 実値パラメータを `ParameterStore` に登録し、override を適用した実値を返す。
- なぜ: UI/自動化からの実値入力を一元管理し、関数呼び出し時に最新の実値を適用するため。

流れ（概要）:
1) merge: シグネチャ既定値・ユーザー入力・`skip` を考慮してパラメータをマージ。
2) resolve: 値種別を判定し、scalar/vector/passthrough に分岐。
   - scalar/vector: RangeHint を構築しつつ、実値を ParameterStore に register/resolve。
   - passthrough: 数値以外（列挙/真偽など）は RangeHint なしで register/resolve。
3) return: override 適用後の実値辞書を返す。

関連:
- 呼び出し元は `ParameterRuntime.before_shape_call/before_effect_call`。
- メタ情報は `FunctionIntrospector`（doc/signature/param_meta）から供給される。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterStore,
    RangeHint,
    SourceType,
    ValueType,
)

# 正規化レイヤは廃止

_VECTOR_SUFFIX = ("x", "y", "z", "w")


@dataclass(frozen=True)
class ParameterContext:
    """shape/effect 呼び出し時の識別情報。"""

    scope: SourceType
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
        merged, sources = self._merge_with_defaults(params, signature, skip=skip)
        updated: dict[str, Any] = {}

        for key, raw_value in merged.items():
            descriptor_id = f"{context.descriptor_prefix}.{key}"
            meta_entry = param_meta.get(key, {})
            default_actual = self._default_from_signature(signature, key, raw_value)
            source = sources.get(key, "provided")
            value_type = self._determine_value_type(meta_entry, default_actual, raw_value)

            if self._is_vector_value(raw_value, default_actual):
                updated[key] = self._resolve_vector(
                    context=context,
                    descriptor_id=descriptor_id,
                    param_name=key,
                    raw_value=raw_value,
                    source=source,
                    default_actual=default_actual,
                    doc=doc,
                    meta_entry=meta_entry,
                    has_default=self._has_default(signature, key),
                )
                continue

            if self._is_numeric_type(value_type):
                updated[key] = self._resolve_scalar(
                    context=context,
                    descriptor_id=descriptor_id,
                    param_name=key,
                    raw_value=raw_value,
                    source=source,
                    default_actual=default_actual,
                    doc=doc,
                    meta_entry=meta_entry,
                    value_type=value_type,
                    has_default=self._has_default(signature, key),
                )
                continue

            updated[key] = self._resolve_passthrough(
                context=context,
                descriptor_id=descriptor_id,
                param_name=key,
                value=raw_value,
                doc=doc,
                default_value=default_actual,
            )

        return updated

    def _resolve_scalar(
        self,
        *,
        context: ParameterContext,
        descriptor_id: str,
        param_name: str,
        raw_value: Any,
        source: str,
        default_actual: Any,
        doc: str | None,
        meta_entry: Mapping[str, Any],
        value_type: ValueType,
        has_default: bool,
    ) -> Any:
        hint = self._range_hint_from_meta(
            value_type=value_type,
            meta=meta_entry,
            component_index=None,
            default_value=default_actual,
        )
        descriptor = ParameterDescriptor(
            id=descriptor_id,
            label=f"{context.label_prefix} · {param_name}",
            source=context.scope,
            category=context.scope,
            value_type=value_type,
            default_value=default_actual,
            range_hint=hint,
            help_text=doc,
            vector_group=None,
        )
        actual_value = default_actual if source == "default" else raw_value
        return self._register_scalar(descriptor, actual_value)

    def _resolve_vector(
        self,
        *,
        context: ParameterContext,
        descriptor_id: str,
        param_name: str,
        raw_value: Any,
        source: str,
        default_actual: Any,
        doc: str | None,
        meta_entry: Mapping[str, Any],
        has_default: bool,
    ) -> tuple[Any, ...]:
        provided_values = self._ensure_sequence(raw_value)
        default_values = self._ensure_sequence(default_actual)
        length = max(len(provided_values), len(default_values))
        if length == 0:
            return tuple()

        if not meta_entry:
            components: list[float] = []
            for idx in range(length):
                value = (
                    default_values[idx]
                    if (source == "default" or not provided_values) and idx < len(default_values)
                    else (
                        provided_values[idx] if idx < len(provided_values) else provided_values[-1]
                    )
                )
                components.append(float(value))

            results: list[float] = []
            for idx, value in enumerate(components):
                suffix = _VECTOR_SUFFIX[idx] if idx < len(_VECTOR_SUFFIX) else str(idx)
                component_id = f"{descriptor_id}.{suffix}"
                default_component = default_values[idx] if idx < len(default_values) else value
                component_hint = self._range_hint_from_meta(
                    value_type="float",
                    meta={},
                    component_index=None,
                    default_value=default_component,
                )
                descriptor = ParameterDescriptor(
                    id=component_id,
                    label=f"{context.label_prefix} · {param_name}.{suffix}",
                    source=context.scope,
                    category=context.scope,
                    value_type="float",
                    default_value=float(default_component),
                    range_hint=component_hint,
                    help_text=doc,
                    vector_group=descriptor_id,
                )
                self._store.register(descriptor, value)
                results.append(float(self._store.resolve(component_id, value)))
            return tuple(results)

        descriptors: list[ParameterDescriptor] = []
        actual_values: list[float] = []

        for idx in range(length):
            suffix = _VECTOR_SUFFIX[idx] if idx < len(_VECTOR_SUFFIX) else str(idx)
            component_id = f"{descriptor_id}.{suffix}"
            component_hint = self._range_hint_from_meta(
                value_type="float",
                meta=meta_entry,
                component_index=idx,
                default_value=None,
            )

            default_component_actual = self._component_default_actual(
                default_values=default_values,
                idx=idx,
                has_default=has_default,
                hint=component_hint,
            )
            if source == "default" or not provided_values:
                actual_component = default_component_actual
            else:
                actual_component = (
                    provided_values[idx] if idx < len(provided_values) else provided_values[-1]
                )

            descriptor = ParameterDescriptor(
                id=component_id,
                label=f"{context.label_prefix} · {param_name}.{suffix}",
                source=context.scope,
                category=context.scope,
                value_type="float",
                default_value=default_component_actual,
                range_hint=component_hint,
                help_text=doc,
                vector_group=descriptor_id,
            )

            descriptors.append(descriptor)
            actual_values.append(float(actual_component))

        resolved_components = self._register_vector(descriptors, actual_values)
        return tuple(resolved_components)

    def _resolve_passthrough(
        self,
        *,
        context: ParameterContext,
        descriptor_id: str,
        param_name: str,
        value: Any,
        doc: str | None,
        default_value: Any,
    ) -> Any:
        value_type = self._value_type(default_value if default_value is not None else value)
        descriptor = ParameterDescriptor(
            id=descriptor_id,
            label=f"{context.label_prefix} · {param_name}",
            source=context.scope,
            category=context.scope,
            value_type=value_type,
            default_value=default_value,
            range_hint=None,
            help_text=doc,
            vector_group=None,
            supported=value_type in {"float", "int", "bool"},
        )
        self._store.register(descriptor, value)
        return self._store.resolve(descriptor.id, value)

    def _register_scalar(self, descriptor: ParameterDescriptor, value: Any) -> Any:
        self._store.register(descriptor, value)
        return self._store.resolve(descriptor.id, value)

    def _register_vector(
        self,
        descriptors: list[ParameterDescriptor],
        values: list[float],
    ) -> list[float]:
        resolved: list[float] = []
        for descriptor, value in zip(descriptors, values):
            self._store.register(descriptor, value)
            resolved_value = self._store.resolve(descriptor.id, value)
            resolved.append(float(resolved_value))
        return resolved

    @staticmethod
    def _merge_with_defaults(
        params: Mapping[str, Any],
        signature,
        *,
        skip: set[str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if signature is None:
            return dict(params), {k: "provided" for k in params}
        skip = skip or set()
        merged: dict[str, Any] = {}
        sources: dict[str, str] = {}
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
                sources[name] = "provided"
                continue
            if parameter.default is inspect._empty:  # type: ignore[attr-defined]
                continue
            merged[name] = parameter.default
            sources[name] = "default"
        for name, value in params.items():
            if name in skip or name in merged:
                continue
            merged[name] = value
            sources[name] = "provided"
        return merged, sources

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

    def _determine_value_type(
        self,
        meta: Mapping[str, Any],
        default_value: Any,
        raw_value: Any,
    ) -> ValueType:
        meta_type = meta.get("type")
        if isinstance(meta_type, str):
            lowered = meta_type.lower()
            if lowered in {"number", "float"}:
                return "float"
            if lowered in {"integer", "int"}:
                return "int"
            if lowered in {"bool", "boolean"}:
                return "bool"
            if lowered == "string":
                return "enum"
        if "choices" in meta:
            return "enum"
        if default_value is not None:
            return self._value_type(default_value)
        return self._value_type(raw_value)

    @staticmethod
    def _is_numeric_type(value_type: ValueType) -> bool:
        return value_type in {"float", "int", "bool"}

    def _is_vector_value(self, value: Any, default_value: Any) -> bool:
        if self._is_vector(value):
            return True
        if self._is_vector(default_value):
            return True
        return False

    @staticmethod
    def _value_type(value: Any) -> ValueType:
        if isinstance(value, str):
            return "enum"
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

    def _range_hint_from_meta(
        self,
        *,
        value_type: ValueType,
        meta: Mapping[str, Any],
        component_index: int | None,
        default_value: Any | None,
    ) -> RangeHint:
        min_value = self._extract_meta_component(meta.get("min"), component_index)
        max_value = self._extract_meta_component(meta.get("max"), component_index)
        step = self._extract_meta_component(meta.get("step"), component_index)

        default_min, default_max, default_step = self._default_actual_range(value_type)
        actual_min = min_value if min_value is not None else None
        actual_max = max_value if max_value is not None else None
        actual_step = step if step is not None else default_step

        if actual_min is None or actual_max is None:
            inferred_min, inferred_max = self._infer_range_from_default(
                default_value=default_value,
                value_type=value_type,
                fallback_min=default_min,
                fallback_max=default_max,
            )
            if actual_min is None:
                actual_min = inferred_min
            if actual_max is None:
                actual_max = inferred_max

        if actual_min == actual_max:
            if value_type == "int":
                actual_max = int(actual_min) + 1
            else:
                actual_max = float(actual_min) + 1.0

        return RangeHint(
            min_value=float(actual_min),
            max_value=float(actual_max),
            step=actual_step,
            scale="linear",
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
    def _default_actual_range(
        value_type: ValueType,
    ) -> tuple[float | int, float | int, float | int | None]:
        if value_type == "bool":
            return 0, 1, 1
        if value_type == "int":
            return 0, 1, 1
        return 0.0, 1.0, None

    @staticmethod
    def _infer_range_from_default(
        *,
        default_value: Any | None,
        value_type: ValueType,
        fallback_min: float | int,
        fallback_max: float | int,
    ) -> tuple[float | int, float | int]:
        if isinstance(default_value, (int, float)) and not isinstance(default_value, bool):
            if fallback_min == 0 and fallback_max == 1:
                return fallback_min, fallback_max
            center = float(default_value)
            span = max(abs(center), 1.0)
            if value_type == "int":
                span_int = max(int(round(span)), 1)
                lower = int(center - span_int)
                if lower < 0:
                    lower = 0
                upper = int(center + span_int)
                if upper <= lower:
                    upper = lower + 1
                return lower, upper
            lower_f = center - span
            upper_f = center + span
            if upper_f <= lower_f:
                upper_f = lower_f + 1.0
            return lower_f, upper_f
        return fallback_min, fallback_max

    @staticmethod
    def _component_default_actual(
        *,
        default_values: list[float],
        idx: int,
        has_default: bool,
        hint: RangeHint,
    ) -> float:
        if has_default and idx < len(default_values):
            return float(default_values[idx])
        lo = hint.min_value
        hi = hint.max_value
        if lo is not None and hi is not None:
            return float((float(lo) + float(hi)) / 2.0)
        if lo is not None:
            return float(lo)
        if hi is not None:
            return float(hi)
        return 0.0

    @staticmethod
    def _has_default(signature, key: str) -> bool:
        if signature is None:
            return False
        parameter = signature.parameters.get(key)
        if parameter is None:
            return False
        return parameter.default is not inspect._empty  # type: ignore[attr-defined]

    @staticmethod
    def _ensure_sequence(value: Any) -> list[float]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [float(v) for v in value]
        return []


__all__ = ["ParameterContext", "ParameterValueResolver"]
