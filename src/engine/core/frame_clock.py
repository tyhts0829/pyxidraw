"""
どこで: `engine.core` の簡易フレームドライバ。
何を: `Tickable` の列を固定順序で呼び出す FrameClock（dt 測定とループ管理）。
なぜ: GUI/ループから呼び出すだけで複数コンポーネントの更新順を統一するため。
"""

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
