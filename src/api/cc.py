"""
どこで: `api.cc`（公開 API）。
何を: 現在フレームの MIDI CC スナップショットを保持する軽量プロキシを提供。
なぜ: `draw(t)` 内で `from api import cc; cc[i]` と書けるようにするため。

設計:
- `cc[i]` は 0.0–1.0 の float を返す（未定義は 0.0）。
- スナップショットはフレーム毎に `set_snapshot()` で差し替え（WorkerPool が設定）。
- `raw()` は現在スナップショットの浅いコピーを返す。
"""

from __future__ import annotations

from collections.abc import Mapping
from util.cc_provider import set_cc_snapshot_provider  # 依存反転フック（api→util のみ）


class CCAPI:
    __slots__ = ("_snapshot",)

    def __init__(self) -> None:
        self._snapshot: dict[int, float] = {}

    # --- 公開 API ---
    def __getitem__(self, index: int) -> float:
        """`cc[i]` で 0.0–1.0 の float を返す。未定義は 0.0。"""
        try:
            i = int(index)
        except Exception:
            i = 0
        v = self._snapshot.get(i, 0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    # ユーティリティ
    def raw(self) -> dict[int, float]:
        return dict(self._snapshot)

    # --- 内部/ランタイム向け API ---
    def set_snapshot(self, mapping: Mapping[int, float] | None) -> None:
        self._snapshot = dict(mapping or {})

    # 表示（print(cc) 用）
    def __repr__(self) -> str:  # noqa: D401 - 簡潔
        try:
            items = ", ".join(
                f"{k}: {v:.3f}" for k, v in sorted(self._snapshot.items(), key=lambda x: int(x[0]))
            )
            return f"cc{{{items}}}"
        except Exception:
            return "cc{}"

    __str__ = __repr__


# グローバルインスタンス
cc = CCAPI()


# ランタイムからの利用を簡易化するトップレベル関数（内部用）
def set_snapshot(mapping: Mapping[int, float] | None) -> None:
    cc.set_snapshot(mapping)


def raw() -> dict[int, float]:  # noqa: D401 - 簡潔
    return cc.raw()


__all__ = ["cc"]

# util 層へスナップショットプロバイダを公開し、engine 側が間接参照できるようにする。
try:  # フェイルソフト（テスト/実行環境での循環を避ける）
    set_cc_snapshot_provider(raw)
except Exception:
    pass
