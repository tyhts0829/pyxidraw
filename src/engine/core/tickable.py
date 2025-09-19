from typing import Protocol


class Tickable(Protocol):
    """1 フレーム分の更新を行うインターフェース。"""

    def tick(self, dt: float) -> None:
        """内部状態を `dt` 秒ぶん進める。"""
