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


def extract_overrides(store: ParameterStore) -> dict[str, Any]:
    """ParameterStore から override のみを取り出して返す。

    - 量子化は行わない（実値）。
    - キーは `"{scope}.{name}#{index}.{param}"`。
    """
    overrides: dict[str, Any] = {}
    for desc in store.descriptors():
        pid = desc.id
        cur = store.current_value(pid)
        org = store.original_value(pid)
        if cur is None:
            continue
        if cur == org:
            continue
        # JSON保存互換で tuple はそのまま運ぶ（呼び出し側で解釈）
        overrides[pid] = cur
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

    def begin_frame(self) -> None:
        self._shape_counters.clear()

    def set_inputs(self, t: float) -> None:
        self._t = float(t)

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
                if param_name and param_name not in updated:
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
    ) -> Mapping[str, Any]:
        prefix = f"effect.{effect_name}#{step_index}"
        updated = dict(params)
        updated = self._inject_t(fn, updated)
        updated = self._apply_overrides(prefix, updated)
        return updated


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
