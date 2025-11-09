"""
どこで: `engine.ui.parameters` のスナップショット適用ユーティリティ。
何を: Parameter GUI の override 値スナップショットをワーカ側に適用するための最小ランタイム
      （SnapshotRuntime）と、適用関数 `apply_param_snapshot` を提供する。
なぜ: GUI をメインスレッドで維持しつつ、ワーカプロセスで `user_draw(t)` に GUI 値を反映するため。

設計要点:
- 依存境界を保つため、Engine 側からはトップレベル関数 `apply_param_snapshot` を関数注入で呼ぶだけにする。
- SnapshotRuntime は `ParameterRuntime` と互換のメソッドシグネチャ（before_*_call）を提供し、
  「明示引数 > GUI > 既定値」の優先順位を保つ（未指定引数にのみ GUI 値を適用）。
"""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Any, Mapping

from .runtime import activate_runtime, deactivate_runtime
from .state import ParameterStore


def extract_overrides(
    store: ParameterStore, cc_mapping: Mapping[int, float] | None = None
) -> dict[str, Any]:
    """ParameterStore から override のみを取り出して返す。

    - 量子化は行わない（実値）。
    - キーは `"{scope}.{name}#{index}.{param}"`。
    """
    overrides: dict[str, Any] = {}
    # 1) GUI override（original と異なる current のみ）
    for desc in store.descriptors():
        pid = desc.id
        try:
            cur = store.current_value(pid)
            org = store.original_value(pid)
        except Exception:
            continue
        if cur is None:
            continue
        if cur == org:
            continue
        # JSON保存互換で tuple はそのまま運ぶ（呼び出し側で解釈）
        overrides[pid] = cur

    # 2) CC バインドの適用（数値スカラ）。GUI 値があっても CC を優先。
    for desc in store.descriptors():
        if desc.value_type not in {"float", "int"}:
            continue
        pid = desc.id
        cc_idx = None
        try:
            cc_idx = store.cc_binding(pid)
        except Exception:
            cc_idx = None
        if cc_idx is None:
            continue
        try:
            # 優先: 呼び出し元から渡された cc_mapping（フレーム固有）
            if cc_mapping is not None and isinstance(cc_mapping, Mapping):
                cc_val = float(cc_mapping.get(int(cc_idx), 0.0))
            else:
                cc_val = float(store.cc_value(cc_idx))  # 0..1（プロバイダ）
            if desc.range_hint is not None:
                lo = float(desc.range_hint.min_value)
                hi = float(desc.range_hint.max_value)
            else:
                lo, hi = 0.0, 1.0
            scaled = lo + (hi - lo) * cc_val
            # デバッグ出力は抑制
            if desc.value_type == "int":
                overrides[pid] = int(round(scaled))
            else:
                overrides[pid] = float(scaled)
        except Exception:
            # CC の適用に失敗しても無視（GUI override のみ適用）
            continue

    # 3) CC バインドの適用（ベクトル）。成分ごとに CC があれば置換し、タプルで渡す。
    for desc in store.descriptors():
        if desc.value_type != "vector":
            continue
        pid = desc.id
        # 成分 ID と CC バインディングの有無
        suffixes = ("x", "y", "z", "w")
        try:
            comp_cc: list[int | None] = [store.cc_binding(f"{pid}::{s}") for s in suffixes]
        except Exception:
            comp_cc = [None, None, None, None]
        if not any(x is not None for x in comp_cc):
            continue
        # ベースベクトル（current→original→default の順で取得）
        try:
            base = store.current_value(pid)
            if base is None:
                base = store.original_value(pid)
        except Exception:
            base = None
        if not isinstance(base, (list, tuple)):
            base = (
                desc.default_value
                if isinstance(desc.default_value, (list, tuple))
                else (0.0, 0.0, 0.0)
            )
        vec = list(base)
        dim = min(max(3, len(vec)), 4)
        vec = (vec + [0.0] * dim)[:dim]
        # レンジ（ヒント優先、なければ 0..1）
        try:
            vh = desc.vector_hint
            mins = list(getattr(vh, "min_values", (0.0, 0.0, 0.0, 0.0)))
            maxs = list(getattr(vh, "max_values", (1.0, 1.0, 1.0, 1.0)))
        except Exception:
            mins = [0.0, 0.0, 0.0, 0.0]
            maxs = [1.0, 1.0, 1.0, 1.0]
        changed = False
        for i in range(dim):
            idx = comp_cc[i]
            if idx is None:
                continue
            try:
                if cc_mapping is not None and isinstance(cc_mapping, Mapping):
                    cc_val = float(cc_mapping.get(int(idx), 0.0))
                else:
                    cc_val = float(store.cc_value(int(idx)))
                lo = float(mins[i]) if i < len(mins) else 0.0
                hi = float(maxs[i]) if i < len(maxs) else 1.0
                vec[i] = lo + (hi - lo) * cc_val
                changed = True
            except Exception:
                continue
        if changed:
            try:
                overrides[pid] = tuple(vec)
            except Exception:
                overrides[pid] = vec

    return overrides


class SnapshotRuntime:
    """スナップショットから実行時の引数を解決する軽量 Runtime。

    - ParameterRuntime と同等の before_*_call を実装するが、メタ/登録は行わない。
    - t 引数の注入は `inspect.signature` で検出して実施する。
    - shape 呼び出し index は名称ごとにフレーム内カウントアップ。
    """

    def __init__(self, overrides: Mapping[str, Any] | None) -> None:
        self._overrides: dict[str, Any] = dict(overrides or {})
        self._shape_counters: dict[str, int] = defaultdict(int)
        self._t: float = 0.0
        self._pipeline_counter: int = 0

    def begin_frame(self) -> None:
        self._shape_counters.clear()
        self._pipeline_counter = 0

    def set_inputs(self, t: float) -> None:
        self._t = float(t)

    # ParameterRuntime 互換: 現フレームの時刻を返す
    def current_time(self) -> float:
        return float(self._t)

    # --- 内部ユーティリティ ---
    def _inject_t(self, fn: Any, params: dict[str, Any]) -> dict[str, Any]:
        try:
            sig = inspect.signature(fn)
            if "t" in sig.parameters and "t" not in params:
                params = {**params, "t": self._t}
        except Exception:
            pass
        return params

    def _apply_overrides(self, prefix: str, params: dict[str, Any]) -> dict[str, Any]:
        # 未指定の引数に限り、該当 prefix の override を適用
        updated = dict(params)
        plen = len(prefix)
        for key, value in self._overrides.items():
            if not key.startswith(prefix):
                continue
            # 期待形式: scope.name#index.param
            try:
                # param_name は prefix の次の要素（'.' 区切りの末尾成分）
                # 例: "shape.circle#0.radius" → "radius"
                param_name = key[plen:]
                if param_name.startswith("."):
                    param_name = param_name[1:]
                # ネストはサポートしない前提（現仕様）
                # 未指定（キーなし）または None の場合は GUI override を適用
                if param_name and (param_name not in updated or updated.get(param_name) is None):
                    updated[param_name] = value
            except Exception:
                continue
        return updated

    # --- 形状 ---
    def before_shape_call(
        self,
        shape_name: str,
        fn: Any,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        index = self._shape_counters[shape_name]
        self._shape_counters[shape_name] = index + 1
        prefix = f"shape.{shape_name}#{index}"
        updated = dict(params)
        updated = self._inject_t(fn, updated)
        updated = self._apply_overrides(prefix, updated)
        return updated

    # --- エフェクト ---
    def before_effect_call(
        self,
        *,
        step_index: int,
        effect_name: str,
        fn: Any,
        params: Mapping[str, Any],
        pipeline_uid: str = "",
    ) -> Mapping[str, Any]:
        # ParameterRuntime 互換: pipeline_uid があれば ID に含める
        if pipeline_uid:
            prefix = f"effect@{pipeline_uid}.{effect_name}#{step_index}"
        else:
            prefix = f"effect.{effect_name}#{step_index}"
        updated = dict(params)
        updated = self._inject_t(fn, updated)
        updated = self._apply_overrides(prefix, updated)
        return updated

    # ParameterRuntime 互換: パイプライン順序 UID を提供
    def next_pipeline_uid(self) -> str:
        n = int(self._pipeline_counter)
        self._pipeline_counter = n + 1
        return f"p{n}"


def apply_param_snapshot(overrides: Mapping[str, Any] | None, t: float) -> None:
    """パラメータスナップショットを適用/解除するトップレベル関数（spawn 互換）。

    - overrides が Truthy の場合: SnapshotRuntime を生成して有効化（t を注入）。
    - None/空 の場合: `deactivate_runtime()` を呼び、ランタイムを無効化。
    """
    if overrides:
        rt = SnapshotRuntime(overrides)
        rt.begin_frame()
        rt.set_inputs(t)
        activate_runtime(rt)
    else:
        deactivate_runtime()


__all__ = ["extract_overrides", "apply_param_snapshot", "SnapshotRuntime"]
