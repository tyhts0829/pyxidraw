"""ランタイム側のパラメータ適用ロジック。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from engine.ui.parameters.state import (
    ParameterLayoutConfig,
    ParameterRegistry,
    ParameterStore,
)

from .introspection import FunctionIntrospector
from .value_resolver import ParameterContext, ParameterValueResolver

_ACTIVE_RUNTIME: "ParameterRuntime | None" = None
_OFFLINE_INTROSPECTOR = FunctionIntrospector()


def activate_runtime(runtime: "ParameterRuntime") -> None:
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = runtime


def deactivate_runtime() -> None:
    global _ACTIVE_RUNTIME
    _ACTIVE_RUNTIME = None


def get_active_runtime() -> "ParameterRuntime | None":
    return _ACTIVE_RUNTIME


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
        self._introspector = FunctionIntrospector()
        self._resolver = ParameterValueResolver(store)

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
        info = self._introspector.resolve(kind="shape", name=shape_name, fn=fn)
        context = ParameterContext(scope="shape", name=shape_name, index=index)
        return self._resolver.resolve(
            context=context,
            params=params,
            signature=info.signature,
            doc=info.doc,
            param_meta=info.param_meta,
        )

    # --- エフェクト ---
    def before_effect_call(
        self,
        *,
        step_index: int,
        effect_name: str,
        fn: Any,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        info = self._introspector.resolve(kind="effect", name=effect_name, fn=fn)
        context = ParameterContext(scope="effect", name=effect_name, index=step_index)
        return self._resolver.resolve(
            context=context,
            params=params,
            signature=info.signature,
            doc=info.doc,
            param_meta=info.param_meta,
            skip={"g"},
        )


def resolve_without_runtime(
    *,
    scope: str,
    name: str,
    fn: Any,
    params: Mapping[str, Any],
    index: int = 0,
) -> Mapping[str, Any]:
    """ParameterRuntime 非介在時に 0..1 入力を実レンジへ変換する補助。"""

    info = _OFFLINE_INTROSPECTOR.resolve(kind=scope, name=name, fn=fn)
    store = ParameterStore()
    resolver = ParameterValueResolver(store)
    context = ParameterContext(scope=scope, name=name, index=index)
    skip = {"g"} if scope == "effect" else set()
    return resolver.resolve(
        context=context,
        params=params,
        signature=info.signature,
        doc=info.doc,
        param_meta=info.param_meta,
        skip=skip,
    )
