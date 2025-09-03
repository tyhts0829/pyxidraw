from __future__ import annotations

import os
import time
from typing import Mapping

import numpy as np
import psutil

from engine.core.geometry import Geometry

from ..core.tickable import Tickable
from ..pipeline.buffer import SwapBuffer


class MetricSampler(Tickable):
    """頂点数・CPU・MEM を一定間隔でサンプリングし dict に保持する。"""

    def __init__(self, swap: SwapBuffer, interval: float = 0.2):
        self._swap = swap
        self._proc = psutil.Process(os.getpid())
        self._interval = interval
        self._last = 0.0
        self.data: dict[str, str] = {}

    # -------- Tickable --------
    def tick(self, dt: float) -> None:
        now = time.time()
        if now - self._last < self._interval:
            return
        self._last = now

        verts = self._vertex_count(self._swap.get_front())
        self.data.update(
            VERTEX=f"{verts}",
            CPU=f"{self._proc.cpu_percent(0.0):4.1f}%",
            MEM=self._human(self._proc.memory_info().rss),
        )

    # -------- helpers --------
    @staticmethod
    def _vertex_count(geometry: Geometry | None) -> int:
        return 0 if geometry is None else len(geometry.coords)

    @staticmethod
    def _human(n: float) -> str:
        for u in "B KB MB GB TB".split():
            if n < 1024:
                return f"{n:4.1f}{u}"
            n /= 1024
        return f"{n:4.1f}PB"
