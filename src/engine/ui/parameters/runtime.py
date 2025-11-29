"""
どこで: `engine.ui.parameters` のオーケストレーション層。
何を: Shapes/Effects 呼び出しをフックしてメタ（doc/signature/param_meta）を解決し、
      `ParameterValueResolver` で実値を登録・override 適用して関数へ渡す。オフライン解決も提供。
なぜ: 呼び出し側を汎用の関数 API のまま保ちつつ、UI/自動化からの実値パラメータ適用を透過化するため。
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Mapping

from engine.ui.parameters.state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterRegistry,
    ParameterStore,
    SourceType,
)

from .introspection import FunctionIntrospector
from .value_resolver import ParameterContext, ParameterValueResolver

_OFFLINE_INTROSPECTOR = FunctionIntrospector()
_RUNTIME_STACK: "ContextVar[list[ParameterRuntime]]" = ContextVar("_RUNTIME_STACK", default=[])


def activate_runtime(runtime: "ParameterRuntime") -> None:
    """ランタイムを現在のコンテキストに積む（ネスト可）。"""
    stack = list(_RUNTIME_STACK.get())
    stack.append(runtime)
    _RUNTIME_STACK.set(stack)


def deactivate_runtime() -> None:
    """現在のコンテキストからランタイムを外す（無ければ何もしない）。"""
    stack = list(_RUNTIME_STACK.get())
    if stack:
        stack.pop()
    _RUNTIME_STACK.set(stack)


def get_active_runtime() -> "ParameterRuntime | None":
    stack = _RUNTIME_STACK.get()
    return stack[-1] if stack else None


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
        self._t: float = 0.0
        self._pipeline_counter: int = 0
        # ラベル付きパイプラインの表示名管理（カテゴリ用）
        self._pipeline_label_by_uid: dict[str, str] = {}
        self._pipeline_label_counter: dict[str, int] = {}
        # shape 用のカスタムラベル管理（カテゴリ用）
        self._shape_label_by_name: dict[str, str] = {}
        self._shape_label_counter: dict[str, int] = {}
        # cc はランタイムでは扱わない（api.cc 内に閉じる）

    def set_lazy(self, lazy: bool) -> None:
        self._lazy = lazy

    def set_inputs(self, t: float) -> None:
        """時間`t`のみを登録する。"""
        self._t = float(t)

    # --- フレーム制御 ---
    def begin_frame(self) -> None:
        self._shape_registry.reset()
        self._effect_registry.reset()
        self._pipeline_counter = 0
        self._pipeline_label_by_uid.clear()
        self._pipeline_label_counter.clear()
        self._shape_label_by_name.clear()
        self._shape_label_counter.clear()

    # フレーム内のパイプライン順序に基づく UID を供給
    def next_pipeline_uid(self) -> str:
        n = int(self._pipeline_counter)
        self._pipeline_counter = n + 1
        return f"p{n}"

    # スナップショット（署名用の軽量アクセス）
    def current_time(self) -> float:
        return float(self._t)

    # --- パイプラインラベル管理 ---
    def _assign_pipeline_label(
        self, pipeline_uid: str, base_label: str, *, force: bool = False
    ) -> str:
        """パイプライン UID に対して表示ラベルを割り当てる。

        force=True の場合は既存ラベルを上書きして再割り当てする。
        """
        try:
            key = str(pipeline_uid or "")
        except Exception:
            key = ""
        if not key:
            return ""
        try:
            base = str(base_label or "").strip()
        except Exception:
            base = ""
        if not base:
            return ""
        label_map = self._pipeline_label_by_uid
        if not force and key in label_map:
            return label_map[key]
        count = int(self._pipeline_label_counter.get(base, 0)) + 1
        self._pipeline_label_counter[base] = count
        display_label = f"{base}_{count}"
        label_map[key] = display_label
        return display_label

    def relabel_pipeline(self, pipeline_uid: str, base_label: str) -> None:
        """既存パイプライン UID の表示ラベルを更新する。"""
        display_label = self._assign_pipeline_label(pipeline_uid, base_label, force=True)
        if not display_label:
            return
        try:
            from .state import ParameterDescriptor  # 局所 import
        except Exception:
            return

        pipeline_key = str(pipeline_uid or "")
        if not pipeline_key:
            return

        def _upd(desc: ParameterDescriptor) -> ParameterDescriptor:
            if desc.source != "effect":
                return desc
            if desc.pipeline_uid != pipeline_key:
                return desc
            if desc.category == display_label:
                return desc
            return ParameterDescriptor(
                id=desc.id,
                label=desc.label,
                source=desc.source,
                category=display_label,
                value_type=desc.value_type,
                default_value=desc.default_value,
                category_kind=desc.category_kind,
                range_hint=desc.range_hint,
                help_text=desc.help_text,
                vector_hint=desc.vector_hint,
                supported=desc.supported,
                choices=desc.choices,
                string_multiline=desc.string_multiline,
                string_height=desc.string_height,
                pipeline_uid=desc.pipeline_uid,
                step_index=desc.step_index,
                param_order=desc.param_order,
            )

        try:
            self._store.update_descriptors(_upd)
        except Exception:
            return

    # --- shape ラベル管理 ---
    def _assign_shape_label(self, shape_name: str, base_label: str) -> str:
        """shape 名に対して表示ラベルを割り当てる（フレーム内で連番付与）。"""
        try:
            key = str(shape_name or "")
        except Exception:
            key = ""
        if not key:
            return ""
        try:
            base = str(base_label or "").strip()
        except Exception:
            base = ""
        if not base:
            return ""
        label_map = self._shape_label_by_name
        if key in label_map:
            return label_map[key]
        count = int(self._shape_label_counter.get(base, 0)) + 1
        self._shape_label_counter[base] = count
        display_label = f"{base}_{count}" if count > 1 else base
        label_map[key] = display_label
        return display_label

    def relabel_shape(self, shape_name: str, base_label: str) -> None:
        """shape 名に対するカテゴリラベルを設定する。"""
        display_label = self._assign_shape_label(shape_name, base_label)
        if not display_label:
            return

        def _upd(desc: ParameterDescriptor) -> ParameterDescriptor:
            if desc.source != "shape":
                return desc
            # shape.id は "shape.<name>#<index>.<param>" 形式を前提とする
            try:
                parts = str(desc.id).split(".", 2)
                if len(parts) < 2:
                    return desc
                _, shape_part = parts[0], parts[1]
            except Exception:
                return desc
            if not shape_part.startswith(f"{shape_name}#"):
                return desc
            if desc.category == display_label:
                return desc
            return ParameterDescriptor(
                id=desc.id,
                label=desc.label,
                source=desc.source,
                category=display_label,
                value_type=desc.value_type,
                default_value=desc.default_value,
                category_kind=desc.category_kind,
                range_hint=desc.range_hint,
                help_text=desc.help_text,
                vector_hint=desc.vector_hint,
                supported=desc.supported,
                choices=desc.choices,
                string_multiline=desc.string_multiline,
                string_height=desc.string_height,
                pipeline_uid=desc.pipeline_uid,
                step_index=desc.step_index,
                param_order=desc.param_order,
            )

        try:
            self._store.update_descriptors(_upd)
        except Exception:
            return

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
        # `t` 引数があれば追加（存在時のみ注入）。その他の入力(CC/GUI)は ParameterStore 側の override で適用される。
        if info.signature is not None and "t" in info.signature.parameters and "t" not in params:
            params = {**params, "t": self._t}
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
        pipeline_uid: str = "",
        pipeline_label: str | None = None,
    ) -> Mapping[str, Any]:
        info = self._introspector.resolve(kind="effect", name=effect_name, fn=fn)

        # LazyGeometry をパラメータに含む場合はここで実体化して Geometry に揃える
        # （パイプライン署名生成でハッシュ可能にするため、また実行時に実体値を利用できるようにするため）。
        from common.types import ObjectRef as _ObjectRef  # 局所 import（循環回避）

        def _materialize(v: Any) -> Any:
            try:
                # 遅延ジオメトリ本体
                from engine.core.lazy_geometry import LazyGeometry  # local import

                if isinstance(v, LazyGeometry):
                    return _ObjectRef(v.realize())
            except Exception:
                pass
            # Geometry を直接包む（キャッシュ鍵のために hashable にする）
            try:
                from engine.core.geometry import Geometry as _Geometry

                if isinstance(v, _Geometry):
                    return _ObjectRef(v)
            except Exception:
                pass
            # コンテナ（浅い）
            try:
                if isinstance(v, (list, tuple)):
                    return [_materialize(x) for x in list(v)]
            except Exception:
                pass
            return v

        params = {k: _materialize(v) for k, v in dict(params).items()}
        # パイプラインラベル（カテゴリ名）の決定:
        # - E.label("uid") で指定されたラベルをベースに、同一ラベルに対してフレーム内で 1,2,... と連番を付与する。
        # - 例: "poly_effect" → "poly_effect_1", "poly_effect_2", ...
        pipeline_key = str(pipeline_uid or "")
        try:
            base_label = str(pipeline_label or "").strip()
        except Exception:
            base_label = ""
        display_label = ""
        if base_label and pipeline_key:
            display_label = self._assign_pipeline_label(pipeline_key, base_label)
        elif base_label:
            display_label = base_label

        context = ParameterContext(
            scope="effect",
            name=effect_name,
            index=step_index,
            pipeline=str(pipeline_uid or ""),
            pipeline_label=str(display_label or base_label or ""),
        )
        if info.signature is not None and "t" in info.signature.parameters and "t" not in params:
            params = {**params, "t": self._t}
        resolved_params = self._resolver.resolve(
            context=context,
            params=params,
            signature=info.signature,
            doc=info.doc,
            param_meta=info.param_meta,
            skip={"g"},
        )
        # ---- 共通バイパス（GUI/永続化）: 各ステップに bool パラメータを登録して解決 ----
        try:
            desc_id = f"{context.descriptor_prefix}.bypass"
            label = f"{context.label_prefix}: Bypass"
            category = (
                context.name
                if context.scope == "shape"
                else (context.pipeline_label or context.pipeline or context.scope)
            )
            bypass_desc = ParameterDescriptor(
                id=desc_id,
                label=label,
                source="effect",
                category=category,
                category_kind="pipeline",
                value_type="bool",
                default_value=False,
                pipeline_uid=(context.pipeline or None),
                step_index=int(context.index),
                param_order=-1,
            )
            self._store.register(bypass_desc, False)
            bypass_val = bool(self._store.resolve(desc_id, False))
        except Exception:
            bypass_val = False

        return {**resolved_params, "bypass": bypass_val}


def resolve_without_runtime(
    *,
    scope: SourceType,
    name: str,
    fn: Any,
    params: Mapping[str, Any],
    index: int = 0,
) -> Mapping[str, Any]:
    """ParameterRuntime 非介在時の補助（実値パラメータをそのまま返す）。"""

    # ランタイムが無い場合は変換せず実値をそのまま渡す。
    return dict(params)
