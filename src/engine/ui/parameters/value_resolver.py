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
                source=source,
                doc=doc,
                default_value=default_actual,
                meta_entry=meta_entry,
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
        if source == "default":
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
            return self._register_scalar(descriptor, default_actual)
        # provided は登録せず、そのまま返す
        return raw_value

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
        # 提供値は数値列の可能性があるため、そのまま展開
        provided_values_mixed: list[Any]
        if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
            provided_values_mixed = list(raw_value)
        else:
            provided_values_mixed = []

        # 既定値は数値列として扱う（登録時 original に使用）
        default_values = self._ensure_sequence(default_actual)
        length = max(len(provided_values_mixed), len(default_values))
        if length == 0:
            return tuple()

        resolved_vals: list[float] = []
        for idx in range(length):
            suffix = _VECTOR_SUFFIX[idx] if idx < len(_VECTOR_SUFFIX) else str(idx)
            component_id = f"{descriptor_id}.{suffix}"

            # 範囲ヒントは meta があれば採用、無ければ推定
            component_hint = self._range_hint_from_meta(
                value_type="float",
                meta=meta_entry,
                component_index=idx,
                default_value=(
                    default_values[idx]
                    if idx < len(default_values)
                    else (default_values[-1] if default_values else None)
                ),
            )

            # default を基準に original を導出（後で provided 数値なら置き換える）
            base_original = self._component_default_actual(
                default_values=default_values,
                idx=idx,
                has_default=has_default,
                hint=component_hint,
            )

            # 実際に与えられた値（数値/未指定）
            if provided_values_mixed:
                if idx < len(provided_values_mixed):
                    v_prov = provided_values_mixed[idx]
                else:
                    # 足りない成分は最後の指定値を踏襲（従来互換）
                    v_prov = provided_values_mixed[-1]
            else:
                v_prov = None

            if source == "default":
                descriptor = ParameterDescriptor(
                    id=component_id,
                    label=f"{context.label_prefix} · {param_name}.{suffix}",
                    source=context.scope,
                    category=context.scope,
                    value_type="float",
                    default_value=base_original,
                    range_hint=component_hint,
                    help_text=doc,
                    vector_group=descriptor_id,
                )
                original_value = float(base_original)
                self._store.register(descriptor, original_value)
                resolved = float(self._store.resolve(component_id, original_value))
                resolved_vals.append(resolved)
            else:
                # provided 値（数値ならそれを採用、未指定なら base）
                if v_prov is not None:
                    try:
                        resolved = float(v_prov)
                    except Exception:
                        resolved = float(base_original)
                else:
                    resolved = float(base_original)
                resolved_vals.append(resolved)

        return tuple(resolved_vals)

    def _resolve_passthrough(
        self,
        *,
        context: ParameterContext,
        descriptor_id: str,
        param_name: str,
        value: Any,
        source: str,
        doc: str | None,
        default_value: Any,
        meta_entry: Mapping[str, Any] | None,
    ) -> Any:
        # enum 判定は choices の有無を優先し、無い文字列は自由入力として非対応のままにする
        value_type = self._value_type(default_value if default_value is not None else value)
        # choices 抽出
        choices_list: list[str] | None = None
        if isinstance(meta_entry, Mapping):
            raw_choices = meta_entry.get("choices")
            try:
                if isinstance(raw_choices, Sequence) and not isinstance(raw_choices, (str, bytes)):
                    cands = [str(x) for x in list(raw_choices)]
                    choices_list = cands if cands else None
            except Exception:
                choices_list = None
        # choices は上位 `resolve()` から渡されないため、ここで meta を使わずに判断する
        # ただし value が str であっても、choices が不明な場合は GUI 非対応（supported=False）にする
        supported = value_type in {"float", "int", "bool"} or (
            value_type == "enum" and bool(choices_list)
        )
        # choices は後で param_meta から渡すよう `_determine_value_type` と resolve 経路を拡張してもよい
        if source == "default":
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
                supported=supported,
                choices=choices_list,
            )
            self._store.register(descriptor, value)
            return self._store.resolve(descriptor.id, value)
        return value

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
    ) -> RangeHint | None:
        min_value = self._extract_meta_component(meta.get("min"), component_index)
        max_value = self._extract_meta_component(meta.get("max"), component_index)
        step = self._extract_meta_component(meta.get("step"), component_index)
        if min_value is None or max_value is None:
            return None
        # step は任意。scale は固定。
        return RangeHint(
            min_value=float(min_value),
            max_value=float(max_value),
            step=step,
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
    @staticmethod
    def _component_default_actual(
        *,
        default_values: list[float],
        idx: int,
        has_default: bool,
        hint: RangeHint | None,
    ) -> float:
        if has_default and idx < len(default_values):
            return float(default_values[idx])
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
