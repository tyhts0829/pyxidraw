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
                label=f"{context.label_prefix}: {param_name}",
                source=context.scope,
                category=context.scope,
                value_type=value_type,
                default_value=default_actual,
                range_hint=hint,
                help_text=doc,
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
        # 提供値（provided）は登録せず、実値をタプルとしてそのまま返す
        if source == "provided":
            if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
                try:
                    return tuple(float(v) for v in raw_value)  # type: ignore[return-value]
                except Exception:
                    pass
            return tuple(self._ensure_sequence(default_actual))

        # default 採用時のみ GUI に登録（親 Descriptor 1 件）。
        default_values = self._ensure_sequence(default_actual)
        if not default_values:
            # 次善: 長さ 3 のゼロベクトル
            default_values = [0.0, 0.0, 0.0]
        dim = max(2, min(len(default_values), 4))
        default_tuple = tuple(default_values[:dim])  # type: ignore[assignment]

        vector_hint = self._vector_range_hint_from_meta(meta_entry, dim)
        descriptor = ParameterDescriptor(
            id=descriptor_id,
            label=f"{context.label_prefix}: {param_name}",
            source=context.scope,
            category=context.scope,
            value_type="vector",
            default_value=default_tuple,
            range_hint=None,
            help_text=doc,
            vector_hint=vector_hint,
        )
        self._store.register(descriptor, default_tuple)
        resolved = self._store.resolve(descriptor_id, default_tuple)
        # store は Any を返すため、tuple を保証
        try:
            return tuple(resolved)  # type: ignore[return-value]
        except Exception:
            return default_tuple

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
        # 判定は meta 優先（choices→enum / type:"string"→string）、無指定時は値から推定
        meta_map: Mapping[str, Any] = meta_entry if isinstance(meta_entry, Mapping) else {}
        value_type = self._determine_value_type(meta_map, default_value, value)
        # choices 抽出
        choices_list: list[str] | None = None
        if isinstance(meta_map, Mapping):
            raw_choices = meta_map.get("choices")
            try:
                if isinstance(raw_choices, Sequence) and not isinstance(raw_choices, (str, bytes)):
                    cands = [str(x) for x in list(raw_choices)]
                    choices_list = cands if cands else None
            except Exception:
                choices_list = None
        # choices は上位 `resolve()` から渡されないため、ここで meta を使わずに判断する
        # string は自由入力として GUI 対応、enum は choices が無ければ非対応
        supported = value_type in {"float", "int", "bool", "string"} or (
            value_type == "enum" and bool(choices_list)
        )
        # choices は後で param_meta から渡すよう `_determine_value_type` と resolve 経路を拡張してもよい
        if source == "default":
            # string の複数行/高さヒント（meta）を解釈
            multiline = False
            height: int | None = None
            try:
                if value_type == "string" and isinstance(meta_map, Mapping):
                    ml_raw = meta_map.get("multiline")
                    if isinstance(ml_raw, bool):
                        multiline = ml_raw
                    h_raw = meta_map.get("height")
                    if isinstance(h_raw, (int, float)):
                        height = int(h_raw)
            except Exception:
                multiline = False
                height = None

            descriptor = ParameterDescriptor(
                id=descriptor_id,
                label=f"{context.label_prefix}: {param_name}",
                source=context.scope,
                category=context.scope,
                value_type=value_type,
                default_value=default_value,
                range_hint=None,
                help_text=doc,
                supported=supported,
                choices=choices_list,
                string_multiline=multiline,
                string_height=height,
            )
            self._store.register(descriptor, value)
            return self._store.resolve(descriptor.id, value)
        return value

    def _register_scalar(self, descriptor: ParameterDescriptor, value: Any) -> Any:
        self._store.register(descriptor, value)
        return self._store.resolve(descriptor.id, value)

    # _register_vector は親 Descriptor 化に伴い廃止

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
        # 列挙（choices）が与えられている場合は enum を優先
        if "choices" in meta:
            return "enum"
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
                return "string"
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
            return "string"
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

    def _vector_range_hint_from_meta(self, meta: Mapping[str, Any], dim: int):
        from engine.ui.parameters.state import VectorRangeHint  # 局所 import（循環回避）

        try:
            mins = []
            maxs = []
            steps = []
            for i in range(dim):
                mv = self._extract_meta_component(meta.get("min"), i)
                xv = self._extract_meta_component(meta.get("max"), i)
                sv = self._extract_meta_component(meta.get("step"), i)
                mins.append(float(mv) if mv is not None else None)
                maxs.append(float(xv) if xv is not None else None)
                steps.append(sv if isinstance(sv, (int, float)) else None)
            if any(v is None for v in mins) or any(v is None for v in maxs):
                return None
            if dim == 4:
                return VectorRangeHint(
                    min_values=(mins[0], mins[1], mins[2], mins[3]),  # type: ignore[arg-type]
                    max_values=(maxs[0], maxs[1], maxs[2], maxs[3]),  # type: ignore[arg-type]
                    steps=(steps[0], steps[1], steps[2], steps[3]),  # type: ignore[arg-type]
                    scale="linear",
                )
            return VectorRangeHint(
                min_values=(mins[0], mins[1], mins[2]),  # type: ignore[arg-type]
                max_values=(maxs[0], maxs[1], maxs[2]),  # type: ignore[arg-type]
                steps=(steps[0], steps[1], steps[2]),  # type: ignore[arg-type]
                scale="linear",
            )
        except Exception:
            return None

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
