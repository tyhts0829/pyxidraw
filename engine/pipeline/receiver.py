from __future__ import annotations

from queue import Empty
from typing import Optional

from ..core.tickable import Tickable
from .buffer import SwapBuffer


class StreamReceiver(Tickable):
    """結果キューを監視して DoubleBuffer に流し込むだけの責務。"""

    def __init__(self, double_buffer: SwapBuffer, result_q, max_packets_per_tick: int = 2):
        """
        _buffer:データを流し込む先のDoubleBuffer
        _q (Queue): ワーカープロセスが作成したデータ（RenderPacket）が入るキュー
        _max: 1回の更新(tick)で処理するパケットの最大数
        _latest_frame: 最新のフレーム番号（古いデータを無視するため）
        """
        self._buffer = double_buffer
        self._q = result_q
        self._max = max_packets_per_tick
        self._latest_frame: Optional[int] = None

    # -------- Tickable interface --------
    def tick(self, dt: float) -> None:
        processed = 0
        while processed < self._max:  # メインスレッドの負荷を抑えるため1回のtickで処理するパケットを制限
            try:
                packet = self._q.get_nowait()
            except Empty:  # キューが空なら何もしない
                break

            # 例外は親に投げ直す
            if isinstance(packet, Exception):
                raise packet

            # 最新のフレームならバッファに追加
            if (self._latest_frame is None) or (packet.frame_id > self._latest_frame):
                self._buffer.push(packet.geometry)
                self._latest_frame = packet.frame_id
            processed += 1
