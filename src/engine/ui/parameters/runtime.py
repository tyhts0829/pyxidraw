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
    RangeHint,
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
        self._meta_cache: dict[str, dict[str, Any]] = {}

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
        meta_key = f"shape::{shape_name}"
        param_meta = self._get_param_meta(cache_key=meta_key, fn=fn)
        normalized = self._merge_with_defaults(params, sig)
        updated: dict[str, Any] = {}
        for key, value in normalized.items():
            descriptor_id = f"shape.{shape_name}#{index}.{key}"
            default_value = self._default_from_signature(sig, key, value)
            group: str | None = None
            descriptors: list[tuple[ParameterDescriptor, Any, str | None]] = []
            meta_entry = param_meta.get(key)
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
                    param_meta=meta_entry,
                )
                descriptors.extend(desc)
            elif self._supported_scalar(value):
                value_type = self._value_type(value)
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{shape_name}#{index} · {key}",
                    source="shape",
                    category="shape",
                    value_type=value_type,
                    default_value=default_value,
                    range_hint=self._range_hint_from_meta(
                        value_type=value_type,
                        meta=meta_entry,
                        component_index=None,
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
        param_meta = self._get_param_meta(cache_key=doc_key, fn=fn)
        normalized = self._merge_with_defaults(
            params,
            sig,
            skip={"g"},
        )
        updated: dict[str, Any] = {}
        for key, value in normalized.items():
            descriptor_id = f"effect.{effect_name}#{step_index}.{key}"
            default_value = self._default_from_signature(sig, key, value)
            group: str | None = None
            descriptors: list[tuple[ParameterDescriptor, Any, str | None]] = []
            meta_entry = param_meta.get(key)
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
                    param_meta=meta_entry,
                )
                descriptors.extend(desc)
            elif self._supported_scalar(value):
                value_type = self._value_type(value)
                descriptor = ParameterDescriptor(
                    id=descriptor_id,
                    label=f"{effect_name}#{step_index} · {key}",
                    source="effect",
                    category="effect",
                    value_type=value_type,
                    default_value=default_value,
                    range_hint=self._range_hint_from_meta(
                        value_type=value_type,
                        meta=meta_entry,
                        component_index=None,
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
    def _merge_with_defaults(
        params: Mapping[str, Any],
        sig: inspect.Signature | None,
        *,
        skip: set[str] | None = None,
    ) -> dict[str, Any]:
        if sig is None:
            return dict(params)
        skip = skip or set()
        merged: dict[str, Any] = {}
        for name, parameter in sig.parameters.items():
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
        param_meta: Mapping[str, Any] | None,
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
                range_hint=self._range_hint_from_meta(
                    value_type=component_type,
                    meta=param_meta,
                    component_index=idx,
                ),
                help_text=doc,
                vector_group=group,
            )
            descriptors.append((descriptor, value, suffix))
        return descriptors

    def _get_param_meta(self, *, cache_key: str, fn: Any) -> dict[str, Any]:
        if cache_key in self._meta_cache:
            return self._meta_cache[cache_key]
        meta_raw = getattr(fn, "__param_meta__", None)
        if not isinstance(meta_raw, Mapping):
            meta: dict[str, Any] = {}
        else:
            # フィールドはシンプルな dict のみに絞る
            meta = {
                str(param_name): value
                for param_name, value in meta_raw.items()
                if isinstance(value, Mapping)
            }
        self._meta_cache[cache_key] = meta
        return meta

    def _range_hint_from_meta(
        self,
        *,
        value_type: ValueType,
        meta: Mapping[str, Any] | None,
        component_index: int | None,
    ) -> RangeHint:
        min_value: float | int | None = None
        max_value: float | int | None = None
        step: float | int | None = None
        if meta:
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
            # bool は 0/1 トグルとして扱う
            return RangeHint(int(min_value), int(max_value), step=1)
        if value_type == "int":
            return RangeHint(
                int(min_value), int(max_value), step=int(step) if step is not None else 1
            )
        return RangeHint(
            float(min_value), float(max_value), step=float(step) if step is not None else None
        )

    @staticmethod
    def _default_range(
        value_type: ValueType,
    ) -> tuple[float | int, float | int, float | int | None]:
        if value_type == "bool":
            return 0, 1, 1
        if value_type == "int":
            return 0, 1, 1
        return 0.0, 1.0, None

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
