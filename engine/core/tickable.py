from typing import Protocol


class Tickable(Protocol):
    """1 フレーム分の更新を行うインターフェース。"""

    def tick(self, dt: float) -> None:
        """Advance internal state by `dt` seconds."""
