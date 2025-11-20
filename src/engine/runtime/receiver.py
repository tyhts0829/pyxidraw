"""
どこで: `engine.runtime` の受信層。
何を: ワーカ結果キューから `RenderPacket`/例外を取り出し、最新フレームのみ `SwapBuffer` へ反映。
なぜ: メインスレッドの負荷を一定に保ちつつ、古いフレームの無駄な描画を避けるため。
"""

from __future__ import annotations

from queue import Empty
from typing import Callable, Mapping
from engine.runtime.frame import RenderFrame

from ..core.tickable import Tickable
from .buffer import SwapBuffer


class StreamReceiver(Tickable):
    """結果キューを監視して SwapBuffer に流し込むだけの責務。"""

    def __init__(
        self,
        swap_buffer: SwapBuffer,
        result_q,
        max_packets_per_tick: int = 2,
        on_metrics: Callable[[Mapping[str, str]], None] | None = None,
    ):
        """
        _swap_buffer: データを流し込む先の SwapBuffer
        _q (Queue): ワーカープロセスが作成したデータ（RenderPacket）が入るキュー
        _max: 1回の更新(tick)で処理するパケットの最大数
        _latest_frame: 最新のフレーム番号（古いデータを無視するため）
        """
        self._swap_buffer = swap_buffer
        self._q = result_q
        self._max = max_packets_per_tick
        self._latest_frame: int | None = None
        self._on_metrics = on_metrics

    # -------- Tickable interface --------
    def tick(self, dt: float) -> None:
        processed = 0
        while (
            processed < self._max
        ):  # メインスレッドの負荷を抑えるため1回のtickで処理するパケットを制限
            try:
                packet = self._q.get_nowait()
            except Empty:  # キューが空なら何もしない
                break

            # 例外は親に投げ直す
            if isinstance(packet, Exception):
                raise packet

            # 最新のフレームならバッファに追加
            if (self._latest_frame is None) or (packet.frame_id > self._latest_frame):
                layers = getattr(packet, "layers", None)
                geom = getattr(packet, "geometry", None)
                frame: RenderFrame | None = None
                if layers is not None:
                    frame = RenderFrame.from_layers(layers)
                elif geom is not None:
                    frame = RenderFrame.from_geometry(geom)
                if frame is not None:
                    self._swap_buffer.push(frame)
                self._latest_frame = packet.frame_id
                # HUD 用メトリクス（任意）
                try:
                    flags = getattr(packet, "cache_flags", None)
                    if flags and self._on_metrics is not None:
                        self._on_metrics(flags)
                except Exception:
                    pass
            processed += 1
