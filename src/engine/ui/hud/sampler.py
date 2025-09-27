"""
どこで: `engine.ui.hud` の計測サブモジュール。
何を: SwapBuffer のフロント Geometry から頂点数を読み、プロセスの CPU/MEM とあわせて
      一定間隔でサンプリング（実効FPSを含む）。HUD 描画向けに文字列辞書として保持。
なぜ: 実行時の簡易メトリクスを低コストに観測し、性能/負荷の傾向を把握するため。
"""

from __future__ import annotations

import os
import time
from typing import Optional

from engine.core.geometry import Geometry

from ...core.tickable import Tickable
from ...runtime.buffer import SwapBuffer
from .config import HUDConfig
from .fields import CPU, FPS, MEM, VERTEX


class MetricSampler(Tickable):
    """頂点数・CPU・MEM を一定間隔でサンプリングし dict に保持する。"""

    def __init__(self, swap: SwapBuffer, config: HUDConfig | None = None):
        self._swap = swap
        self._config = config or HUDConfig()
        self._interval = float(self._config.sample_interval)
        # psutil は必要時のみ遅延 import
        self._proc: Optional["psutil.Process"] = None
        if self._config.show_cpu_mem:
            try:
                import psutil  # type: ignore

                self._proc = psutil.Process(os.getpid())
            except Exception:
                self._proc = None
        # 前回サンプリング時刻とバージョン（実効FPS算出に使用）
        self._last = 0.0
        self._last_ver = self._swap.version()
        self.data: dict[str, str] = {}

    # -------- Tickable --------
    def tick(self, dt: float) -> None:
        now = time.time()
        if now - self._last < self._interval:
            return
        dt = now - self._last if self._last > 0.0 else 0.0
        self._last = now

        # 実効FPS: SwapBuffer.version() の増分 / 経過秒
        if self._config.show_fps:
            cur_ver = self._swap.version()
            dv = cur_ver - self._last_ver
            self._last_ver = cur_ver
            fps = (dv / dt) if dt > 0.0 else 0.0
            self.data[FPS] = f"{fps:4.1f}"
        else:
            self.data.pop(FPS, None)

        # 頂点数
        if self._config.show_vertex_count:
            verts = self._vertex_count(self._swap.get_front())
            self.data[VERTEX] = f"{verts}"
        else:
            self.data.pop(VERTEX, None)

        # CPU/MEM
        if self._config.show_cpu_mem and self._proc is not None:
            try:
                self.data[CPU] = f"{self._proc.cpu_percent(0.0):4.1f}%"
                self.data[MEM] = self._human(self._proc.memory_info().rss)
            except Exception:
                # 計測失敗時は無視（前回値があれば残す）
                pass
        else:
            self.data.pop(CPU, None)
            self.data.pop(MEM, None)

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
