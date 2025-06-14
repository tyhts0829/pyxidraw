from __future__ import annotations

from typing import Callable, Mapping, Sequence

import moderngl
import numpy as np
import pyglet
from pyglet.window import key

from engine.core.frame_clock import FrameClock
from engine.core.render_window import RenderWindow
from engine.io.manager import connect_midi_controllers
from engine.io.service import MidiService
from engine.monitor.sampler import MetricSampler
from engine.pipeline.buffer import SwapBuffer
from engine.pipeline.receiver import StreamReceiver
from engine.pipeline.worker import WorkerPool
from engine.render.renderer import LineRenderer
from engine.ui.overlay import OverlayHUD
from util.constants import CANVAS_SIZES


def run_sketch(
    user_draw: Callable[[float, Mapping[int, int]], Sequence[np.ndarray]],
    *,
    canvas_size: str | tuple[int, int] = "A5",
    render_scale: int = 4,
    fps: int = 60,
    background: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    workers: int = 6,
) -> None:
    """
    user_draw :
        ``t [sec], cc_dict → [np.ndarray(N, 3), ...]`` を返す関数。
    canvas_size :
        既定キー("A4","A5"...）または ``(width, height)`` mm。
    render_scale :
        mm単位の頂点座標群をレンダリングするときの拡大率。
    fps :
        描画更新レート。
    background :
        RGBA (0‑1)。Processing の ``background()`` と同義。
    workers :
        バックグラウンド計算プロセス数。
    """
    # ---- ① キャンバスサイズ決定 ------------------------------------
    if isinstance(canvas_size, str):
        canvas_width, canvas_height = CANVAS_SIZES[canvas_size.upper()]
    else:
        canvas_width, canvas_height = canvas_size
    window_width, window_height = int(canvas_width * render_scale), int(canvas_height * render_scale)

    # ---- ② MIDI ---------------------------------------------------
    midi_manager = connect_midi_controllers()
    midi_service = MidiService(midi_manager)

    # ---- ③ SwapBuffer + Worker/Receiver ---------------------------
    swap_buffer = SwapBuffer()
    worker_pool = WorkerPool(
        fps=fps,
        draw_callback=user_draw,
        cc_snapshot=midi_service.snapshot,
    )
    stream_receiver = StreamReceiver(swap_buffer, worker_pool.result_q)

    # ---- ④ Window & ModernGL --------------------------------------
    rendering_window = RenderWindow(window_width, window_height, on_draw_cb=lambda: None)
    mgl_ctx: moderngl.Context = moderngl.create_context()
    mgl_ctx.enable(moderngl.BLEND)
    mgl_ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # ----  モニタリング ----------------------------------------
    sampler = MetricSampler(swap_buffer)
    overlay = OverlayHUD(rendering_window, sampler)

    # ---- ⑤ 投影行列（正射影） --------------------------------------
    proj = np.array(
        [
            [2 / canvas_width, 0, 0, -1],
            [0, -2 / canvas_height, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    ).T  # 転置を適用

    renderer_subsys = LineRenderer(mgl_context=mgl_ctx, projection_matrix=proj, double_buffer=swap_buffer)  # type: ignore

    # set composite draw (renderer + HUD)
    def _composite_draw():
        renderer_subsys.draw()
        overlay.draw()

    rendering_window._on_draw_cb = _composite_draw

    # ---- ⑥ FrameCoordinator ---------------------------------------
    frame_clock = FrameClock([midi_service, worker_pool, stream_receiver, renderer_subsys, sampler, overlay])
    pyglet.clock.schedule_interval(frame_clock.tick, 1 / fps)

    # ---- ⑦ pyglet イベント -----------------------------------------
    @rendering_window.event
    def on_draw():  # noqa: ANN001
        renderer_subsys.clear(background)
        _composite_draw()

    @rendering_window.event
    def on_key_press(sym, _mods):  # noqa: ANN001
        if sym == key.ESCAPE:
            rendering_window.close()

    @rendering_window.event
    def on_close():  # noqa: ANN001
        worker_pool.close()
        midi_manager.save_cc()
        renderer_subsys.release()
        pyglet.app.exit()

    pyglet.app.run()
