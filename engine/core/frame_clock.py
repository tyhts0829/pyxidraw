from __future__ import annotations

import time
from typing import Sequence

from .tickable import Tickable


class FrameClock:
    """登録された Tickable を固定順序で実行するだけの極小クラス。"""

    def __init__(self, tickables: Sequence[Tickable]):
        self._tickables = tuple(tickables)
        self._last_time = time.perf_counter()

    # GUI フレームワークから schedule_interval で呼ばせる
    def tick(self, dt: float | None = None) -> None:
        if dt is None:  # pyglet は dt を渡してくれる
            now = time.perf_counter()  # 他フレームワーク用
            dt = now - self._last_time
            self._last_time = now

        for t in self._tickables:
            t.tick(dt)
