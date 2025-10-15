"""
どこで: `api.sketch_runner.recording`
何を: 録画用の品質モード（enter/leave）切替とスケジューリングを補助。
なぜ: `api.sketch` の関数を薄くし、切替処理を再利用可能にするため。
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Tuple

from engine.core.tickable import Tickable


def enter_quality_mode(
    *,
    fps: int,
    draw_callable: Callable[[float], Any],
    cc_snapshot_fn: Callable[[], Mapping[int, float]] | None,
    apply_cc_snapshot,
    apply_param_snapshot,
    param_snapshot_fn,
    metrics_snapshot_fn,
    swap_buffer,
    on_metrics_cb,
    midi_service: Tickable,
    sampler,
    overlay,
    line_renderer: Tickable,
    worker_pool,
    stream_receiver,
    frame_clock,
    pyglet_mod,
) -> Tuple[Any, Any, Any, Callable[[float], None]]:
    """品質モードへ移行し、インライン Worker と固定刻み tick を開始する。

    Returns
    -------
    (worker_pool, stream_receiver, frame_clock, quality_tick_cb)
    """

    # 既存スケジューラを停止
    try:
        pyglet_mod.clock.unschedule(frame_clock.tick)
    except Exception:
        pass
    # 既存ワーカを停止し、インラインへ切り替え
    try:
        worker_pool.close()
    except Exception:
        pass

    # 遅延 import（重い依存）
    from engine.runtime.receiver import StreamReceiver
    from engine.runtime.worker import WorkerPool

    worker_pool = WorkerPool(
        fps=fps,
        draw_callback=draw_callable,
        cc_snapshot=cc_snapshot_fn,
        apply_cc_snapshot=apply_cc_snapshot,
        num_workers=0,  # inline
        apply_param_snapshot=apply_param_snapshot,
        param_snapshot=param_snapshot_fn,
        metrics_snapshot=metrics_snapshot_fn,
    )
    stream_receiver = StreamReceiver(swap_buffer, worker_pool.result_q, on_metrics=on_metrics_cb)

    # 品質最優先: 固定刻みで駆動（overlayは描画はしないがtickは行う）
    tickables: list[Tickable] = [midi_service, worker_pool, stream_receiver, line_renderer]
    if sampler is not None:
        tickables.append(sampler)
    if overlay is not None:
        tickables.append(overlay)

    def _quality_tick(_dt: float) -> None:  # noqa: D401 - pyglet schedule 互換
        fixed = 1.0 / float(max(1, fps))
        for t in tickables:
            t.tick(fixed)

    quality_tick_cb = _quality_tick
    pyglet_mod.clock.schedule_interval(quality_tick_cb, 1 / fps)
    return worker_pool, stream_receiver, frame_clock, quality_tick_cb


def leave_quality_mode(
    *,
    fps: int,
    draw_callable: Callable[[float], Any],
    cc_snapshot_fn: Callable[[], Mapping[int, float]] | None,
    apply_cc_snapshot,
    apply_param_snapshot,
    param_snapshot_fn,
    metrics_snapshot_fn,
    swap_buffer,
    on_metrics_cb,
    midi_service: Tickable,
    sampler,
    overlay,
    line_renderer: Tickable,
    worker_count: int,
    quality_tick_cb,
    pyglet_mod,
) -> Tuple[Any, Any, Any]:
    """品質モードを解除し、通常の WorkerPool/FrameClock 駆動へ復帰する。"""

    # 品質モードのドライバ停止
    try:
        if quality_tick_cb is not None:
            pyglet_mod.clock.unschedule(quality_tick_cb)
    except Exception:
        pass

    # 遅延 import（重い依存）
    from engine.core.frame_clock import FrameClock
    from engine.runtime.receiver import StreamReceiver
    from engine.runtime.worker import WorkerPool

    worker_pool = WorkerPool(
        fps=fps,
        draw_callback=draw_callable,
        cc_snapshot=cc_snapshot_fn,
        apply_cc_snapshot=apply_cc_snapshot,
        num_workers=worker_count,
        apply_param_snapshot=apply_param_snapshot,
        param_snapshot=param_snapshot_fn,
        metrics_snapshot=metrics_snapshot_fn,
    )
    stream_receiver = StreamReceiver(swap_buffer, worker_pool.result_q, on_metrics=on_metrics_cb)
    tickables: list[Tickable] = [midi_service, worker_pool, stream_receiver, line_renderer]
    if sampler is not None:
        tickables.append(sampler)
    if overlay is not None:
        tickables.append(overlay)
    frame_clock = FrameClock(tickables)
    pyglet_mod.clock.schedule_interval(frame_clock.tick, 1 / fps)
    return worker_pool, stream_receiver, frame_clock


__all__ = ["enter_quality_mode", "leave_quality_mode"]
