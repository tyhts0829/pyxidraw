"""
どこで: `engine.core` の更新インターフェース。
何を: 1フレーム更新 `tick(dt)` を持つ `Tickable` Protocol を定義。
なぜ: フレーム駆動のオブジェクト（レンダラ/ワーカ/オーバーレイ等）を一様に扱うため。
"""

from typing import Protocol


class Tickable(Protocol):
    """1 フレーム分の更新を行うインターフェース。"""

    def tick(self, dt: float) -> None:
        """内部状態を `dt` 秒ぶん進める。"""
