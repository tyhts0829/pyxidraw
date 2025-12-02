"""
値解決ユーティリティ（ParameterRuntime からの独立モジュール）。

どこで・何を・なぜ:
- どこで: `engine.ui.parameters` 層。`ParameterRuntime` から呼ばれる。
- 何を: 実値パラメータを `ParameterStore` に登録し、override を適用した実値を返す。
- なぜ: UI/自動化からの実値入力を一元管理し、関数呼び出し時に最新の実値を適用するため。

流れ（概要）:
1) merge: シグネチャ既定値・ユーザー入力・`skip` を考慮してパラメータをマージ。
2) resolve: 値種別を判定し、scalar/vector/passthrough に分岐。
   - scalar/vector/passthrough: すべて Descriptor を登録し、GUI からの override を反映した実値を返す（明示値も GUI で上書き可能）。
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
    CategoryKind,
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
    RangeHint,
    SourceType,
    ValueType,
    effective_range_for_descriptor,
    vector_component_ranges_with_override,
)

# 正規化レイヤは廃止

_VECTOR_SUFFIX = ("x", "y", "z", "w")


@dataclass(frozen=True)
class ParameterContext:
    """shape/effect 呼び出し時の識別情報。"""

    scope: SourceType
    name: str
    index: int
    pipeline: str = ""
    # GUI 表示用のカテゴリ名（任意）。空の場合は pipeline → scope を使用。
    pipeline_label: str = ""

    @property
    def descriptor_prefix(self) -> str:
        # effect のみ pipeline が空でなければ一意なパイプライン識別を付与
        if self.scope == "effect":
            if self.pipeline:
                return f"effect@{self.pipeline}.{self.name}#{self.index}"
            return f"effect.{self.name}#{self.index}"
        return f"{self.scope}.{self.name}#{self.index}"

    @property
    def label_prefix(self) -> str:
        return f"{self.name}#{self.index}"

    @property
    def category(self) -> str:
        """GUI 上でのカテゴリ名を返す。"""

        if self.scope == "shape":
            # shape は呼び出しごとにヘッダを分割し、1 回目は無印、
            # 2 回目以降は name_1, name_2 ... のように連番を付与する。
            try:
                idx = int(self.index)
            except Exception:
                idx = 0
            if idx <= 0:
                return self.name
            return f"{self.name}_{idx}"
        return self.pipeline_label or self.pipeline or self.scope

    @property
    def category_kind(self) -> CategoryKind:
        """GUI 上でのカテゴリ種別を返す。"""

        return "shape" if self.scope == "shape" else "pipeline"

    @property
    def pipeline_uid(self) -> str | None:
        """effect 用のパイプライン識別子を返す。"""

        if self.scope != "effect":
            return None
        return self.pipeline or None

    @property
    def step_index(self) -> int | None:
        """effect のステップインデックスを返す。"""

        if self.scope != "effect":
            return None
        return int(self.index)


class ParameterValueResolver:
    """パラメータ値とメタデータを ParameterStore と同期させる責務を担う。"""

    def __init__(
        self,
        store: ParameterStore,
        *,
        layout: ParameterLayoutConfig | None = None,
    ) -> None:
        self._store = store
        self._layout = layout or ParameterLayoutConfig()

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
        # シグネチャ順（0..N）を param_order として付与（存在しないものは None）
        param_order_map = self._build_param_order_map(signature, skip)
        updated: dict[str, Any] = {}

        for key, raw_value in merged.items():
            descriptor_id = f"{context.descriptor_prefix}.{key}"
            meta_entry = param_meta.get(key, {})
            default_actual = self._default_from_signature(signature, key, raw_value)
            source = sources.get(key, "provided")
            value_type = self._determine_value_type(meta_entry, default_actual, raw_value)

            # vector は meta.type に従って優先判定（default が None でも表示したい）
            if value_type == "vector" or self._is_vector_value(raw_value, default_actual):
                updated[key] = self._resolve_vector(
                    context=context,
                    descriptor_id=descriptor_id,
                    param_name=key,
                    raw_value=raw_value,
                    source=source,
                    default_actual=default_actual,
                    doc=doc,
                    meta_entry=meta_entry,
                    param_order=param_order_map.get(key),
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
                    param_order=param_order_map.get(key),
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
                param_order=param_order_map.get(key),
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
        param_order: int | None,
    ) -> Any:
        hint = self._range_hint_from_meta(
            meta=meta_entry,
            component_index=None,
        )
        descriptor = self._build_descriptor(
            context=context,
            descriptor_id=descriptor_id,
            param_name=param_name,
            value_type=value_type,
            default_value=default_actual,
            range_hint=hint,
            help_text=doc,
            param_order=param_order,
        )
        initial_value = raw_value if source == "provided" else default_actual
        return self._register_scalar(descriptor, initial_value)

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
        param_order: int | None,
    ) -> tuple[Any, ...]:
        default_tuple = self._normalize_vector_default(default_actual)
        if not self._is_vector(default_actual) and self._is_vector(raw_value):
            default_tuple = self._normalize_vector_default(raw_value)
        vector_hint = self._vector_range_hint_from_meta(meta_entry, len(default_tuple))
        descriptor = self._build_descriptor(
            context=context,
            descriptor_id=descriptor_id,
            param_name=param_name,
            value_type="vector",
            default_value=default_tuple,
            range_hint=None,
            help_text=doc,
            param_order=param_order,
            vector_hint=vector_hint,
        )
        initial_value = self._initial_vector_value(raw_value, default_tuple, source)
        self._store.register(descriptor, initial_value)
        resolved = self._store.resolve(descriptor_id, initial_value)
        base_tuple = self._ensure_vector_tuple(resolved, default_tuple)
        return self._apply_cc_to_vector(descriptor_id, base_tuple)

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
        param_order: int | None,
    ) -> Any:
        meta_map: Mapping[str, Any] = meta_entry if isinstance(meta_entry, Mapping) else {}
        # 判定は meta 優先（choices→enum / type:"string"→string）、無指定時は値から推定
        value_type = self._determine_value_type(meta_map, default_value, value)
        choices_list = self._extract_choices(meta_map)
        supported = self._is_supported_passthrough_type(value_type, choices_list)
        multiline, height = self._string_meta(meta_map, value_type)
        searchable = bool(meta_map.get("searchable")) if value_type == "enum" else False
        descriptor = self._build_descriptor(
            context=context,
            descriptor_id=descriptor_id,
            param_name=param_name,
            value_type=value_type,
            default_value=default_value,
            range_hint=None,
            help_text=doc,
            param_order=param_order,
            supported=supported,
            choices=choices_list,
            string_multiline=multiline,
            string_height=height,
            searchable=searchable,
        )
        self._store.register(descriptor, value)
        return self._store.resolve(descriptor.id, value)

    def _register_scalar(self, descriptor: ParameterDescriptor, value: Any) -> Any:
        # Descriptor を登録（GUI 表示用）
        self._store.register(descriptor, value)
        # CC バインドがあれば、GUI 表示は変えず、実行値のみ CC による値へ置換
        try:
            cc_idx = self._store.cc_binding(descriptor.id)
        except Exception:
            cc_idx = None
        if cc_idx is not None and descriptor.value_type in {"float", "int"}:
            try:
                cc_val = float(self._store.cc_value(cc_idx))  # 0..1
                lo, hi = effective_range_for_descriptor(
                    descriptor, self._store, layout=self._layout
                )
                scaled = lo + (hi - lo) * cc_val
                if descriptor.value_type == "int":
                    return int(round(scaled))
                return float(scaled)
            except Exception:
                # フェイルソフト: CC 適用に失敗したら通常経路
                pass
        # 通常経路（override があればそれを適用）
        return self._store.resolve(descriptor.id, value)

    # _register_vector は親 Descriptor 化に伴い廃止

    @staticmethod
    def _build_param_order_map(
        signature,
        skip: set[str] | None,
    ) -> dict[str, int]:
        if signature is None:
            return {}
        order: dict[str, int] = {}
        try:
            for index, (name, parameter) in enumerate(signature.parameters.items()):
                if skip and name in skip:
                    continue
                if parameter.kind in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                ):
                    continue
                order[name] = int(index)
        except Exception:
            return {}
        return order

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
            if parameter.default is inspect.Parameter.empty:
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
        if parameter.default is inspect.Parameter.empty:
            return fallback
        return parameter.default

    def _determine_value_type(
        self,
        meta: Mapping[str, Any],
        default_value: Any,
        raw_value: Any,
    ) -> ValueType:
        # 列挙（choices）が与えられている場合は enum を優先し、次に meta.type、その後 default/raw の値から推定する。
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
            # ベクトル指定（vec/vec2/vec3/vec4/vector/rgb/rgba）は vector として扱う
            if any(k in lowered for k in ("vec", "vector", "rgb")):
                return "vector"
        if default_value is not None:
            return self._value_type(default_value)
        return self._value_type(raw_value)

    @staticmethod
    def _is_numeric_type(value_type: ValueType) -> bool:
        """数値系（スライダ表示対象）かどうかを判定する。"""

        return value_type in {"float", "int", "bool"}

    def _is_vector_value(self, value: Any, default_value: Any) -> bool:
        """値または既定値がベクトル（2–4 次元の数値列）かどうかを判定する。"""

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
        meta: Mapping[str, Any],
        component_index: int | None,
    ) -> RangeHint | None:
        """meta の min/max/step から RangeHint を構築する（表示レンジのみ扱い、値のクランプは行わない）。"""

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
    def _has_default(signature, key: str) -> bool:
        if signature is None:
            return False
        parameter = signature.parameters.get(key)
        if parameter is None:
            return False
        return parameter.default is not inspect.Parameter.empty

    @staticmethod
    def _ensure_sequence(value: Any) -> list[float]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [float(v) for v in value]
        return []

    @staticmethod
    def _normalize_vector_default(default_actual: Any) -> tuple[float, ...]:
        """ベクトル既定値を 2..4 次元のタプルに正規化する。"""

        default_values = ParameterValueResolver._ensure_sequence(default_actual)
        if not default_values:
            # 次善: 長さ 3 のゼロベクトル
            default_values = [0.0, 0.0, 0.0]
        dim = max(2, min(len(default_values), 4))
        return tuple(default_values[:dim])

    @staticmethod
    def _initial_vector_value(
        raw_value: Any, default_tuple: tuple[float, ...], source: str
    ) -> tuple[float, ...]:
        """登録時に使用するベクトルの初期値を決定する。"""

        if source == "provided":
            if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
                try:
                    return tuple(float(v) for v in raw_value)  # type: ignore[return-value]
                except Exception:
                    return default_tuple
            return default_tuple
        return default_tuple

    @staticmethod
    def _ensure_vector_tuple(resolved: Any, fallback: tuple[float, ...]) -> tuple[float, ...]:
        """store からの解決結果をベクトルタプルとして扱う。"""

        if isinstance(resolved, (list, tuple)):
            try:
                return tuple(float(v) for v in resolved)
            except Exception:
                return fallback
        return fallback

    def _apply_cc_to_vector(
        self,
        descriptor_id: str,
        base_values: tuple[float, ...],
    ) -> tuple[float, ...]:
        """ベクトル各成分に CC バインドを適用する。"""

        try:
            vec = list(base_values)
            dim_out = min(len(vec), 4)
            try:
                desc_obj = self._store.get_descriptor(descriptor_id)
            except Exception:
                desc_obj = None
            if desc_obj is not None:
                mins, maxs = vector_component_ranges_with_override(
                    desc_obj,
                    self._store,
                    layout=self._layout,
                    dim=dim_out or None,
                )
            else:
                mins = [0.0] * dim_out
                maxs = [1.0] * dim_out
            changed = False
            for index in range(dim_out):
                comp_id = f"{descriptor_id}::{_VECTOR_SUFFIX[index]}"
                try:
                    cc_idx = self._store.cc_binding(comp_id)
                except Exception:
                    cc_idx = None
                if cc_idx is None:
                    continue
                try:
                    cc_val = float(self._store.cc_value(cc_idx))
                    lo = float(mins[index]) if index < len(mins) else 0.0
                    hi = float(maxs[index]) if index < len(maxs) else 1.0
                    vec[index] = lo + (hi - lo) * cc_val
                    changed = True
                except Exception:
                    continue
            if changed:
                return tuple(vec)
        except Exception:
            return base_values
        return base_values

    @staticmethod
    def _extract_choices(meta_map: Mapping[str, Any]) -> list[str] | None:
        """enum 用の choices を meta から抽出する。"""

        raw_choices = meta_map.get("choices")
        if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, (str, bytes)):
            return None
        try:
            candidates = [str(item) for item in list(raw_choices)]
        except Exception:
            return None
        return candidates or None

    @staticmethod
    def _is_supported_passthrough_type(
        value_type: ValueType,
        choices: list[str] | None,
    ) -> bool:
        """GUI で扱えるかどうかを判定する。"""

        if value_type in {"float", "int", "bool", "string"}:
            return True
        if value_type == "enum":
            return bool(choices)
        return False

    @staticmethod
    def _string_meta(
        meta_map: Mapping[str, Any],
        value_type: ValueType,
    ) -> tuple[bool, int | None]:
        """string 入力用の UI ヒントを meta から解釈する。"""

        multiline = False
        height: int | None = None
        if value_type != "string":
            return multiline, height
        try:
            ml_raw = meta_map.get("multiline")
            if isinstance(ml_raw, bool):
                multiline = ml_raw
            h_raw = meta_map.get("height")
            if isinstance(h_raw, (int, float)):
                height = int(h_raw)
        except Exception:
            return False, None
        return multiline, height

    def _build_descriptor(
        self,
        *,
        context: ParameterContext,
        descriptor_id: str,
        param_name: str,
        value_type: ValueType,
        default_value: Any,
        range_hint: RangeHint | None,
        help_text: str | None,
        param_order: int | None,
        vector_hint=None,
        supported: bool = True,
        choices: list[str] | None = None,
        string_multiline: bool | None = None,
        string_height: int | None = None,
        searchable: bool = False,
    ) -> ParameterDescriptor:
        """ParameterDescriptor を組み立てる共通ヘルパ。"""

        return ParameterDescriptor(
            id=descriptor_id,
            label=f"{context.label_prefix}: {param_name}",
            source=context.scope,
            category=context.category,
            category_kind=context.category_kind,
            value_type=value_type,
            default_value=default_value,
            range_hint=range_hint,
            help_text=help_text,
            vector_hint=vector_hint,
            pipeline_uid=context.pipeline_uid,
            step_index=context.step_index,
            param_order=param_order,
            supported=supported,
            choices=choices,
            string_multiline=bool(string_multiline) if string_multiline is not None else False,
            string_height=string_height or 0,
            searchable=bool(searchable),
        )


__all__ = ["ParameterContext", "ParameterValueResolver"]
