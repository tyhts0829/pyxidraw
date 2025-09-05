from __future__ import annotations

from typing import Callable, Mapping
import logging
import sys
import os

import numpy as np

from engine.core.geometry import Geometry
from engine.core.tickable import Tickable
from util.constants import CANVAS_SIZES


def run_sketch(
    user_draw: Callable[[float, Mapping[int, int]], Geometry],
    *,
    canvas_size: str | tuple[int, int] = "A5",
    render_scale: int = 4,
    fps: int = 60,
    background: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    workers: int = 4,
    use_midi: bool = True,
    midi_strict: bool | None = None,
) -> None:
    """
    user_draw :
        ``t [sec], cc_dict → Geometry`` を返す関数。
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
    use_midi :
        True の場合は可能なら実機 MIDI を使用。未接続/未導入時は既定でフォールバック。
    midi_strict :
        True で厳格モード（初期化失敗時に SystemExit(2)）。None の場合は
        環境変数 ``PYXIDRAW_MIDI_STRICT`` を参照（未設定は False）。
    """
    # 遅延インポート（テスト環境のヘッドレス収集時にウィンドウ作成を避ける）
    import moderngl
    import pyglet
    from pyglet.window import key
    from engine.core.frame_clock import FrameClock
    from engine.core.render_window import RenderWindow
    from engine.monitor.sampler import MetricSampler
    from engine.pipeline.buffer import SwapBuffer
    from engine.pipeline.receiver import StreamReceiver
    from engine.pipeline.worker import WorkerPool
    from engine.render.renderer import LineRenderer
    from engine.ui.overlay import OverlayHUD

    # ---- ① キャンバスサイズ決定 ------------------------------------
    if isinstance(canvas_size, str):
        canvas_width, canvas_height = CANVAS_SIZES[canvas_size.upper()]
    else:
        canvas_width, canvas_height = canvas_size
    window_width, window_height = int(canvas_width * render_scale), int(canvas_height * render_scale)

    # ---- ② MIDI ---------------------------------------------------
    # 環境変数から厳格モードを補完（未指定時）。
    if midi_strict is None:
        env = os.environ.get("PYXIDRAW_MIDI_STRICT")
        if env is not None:
            midi_strict = env == "1" or env.lower() in ("true", "on", "yes")
        else:
            # 設定から既定を参照（なければ False）
            try:
                from util.utils import load_config  # noqa: WPS433

                cfg = load_config() or {}
                midi_cfg = cfg.get("midi", {}) if isinstance(cfg, dict) else {}
                midi_strict = bool(midi_cfg.get("strict_default", False))
            except Exception:
                midi_strict = False

    # ローカルな Null 実装（型安定のため 1 箇所に定義）
    class _NullMidi:
        def snapshot(self) -> Mapping[int, int]:  # CC は int→int の想定（外部で正規化）
            return {}

        def tick(self, dt: float) -> None:
            return None

    midi_service: Tickable
    cc_snapshot_fn: Callable[[], Mapping[int, int]]

    if use_midi:
        try:
            # 遅延インポート（依存未導入環境でもフォールバック可能に）
            from engine.io.manager import connect_midi_controllers  # noqa: WPS433
            from engine.io.service import MidiService  # noqa: WPS433

            midi_manager = connect_midi_controllers()
            # 0台接続もエラー扱いにするかは strict で切替
            if not getattr(midi_manager, "controllers", {}):
                raise RuntimeError("No MIDI devices connected")
            midi_service = MidiService(midi_manager)
            # MidiService は Tickable を実装し、snapshot() を提供する。
            # 型: Callable[[], Mapping[int, int]] に合わせて渡す。
            cc_snapshot_fn = midi_service.snapshot  # type: ignore[assignment]
        except Exception as e:  # ImportError / InvalidPortError / RuntimeError など
            logger = logging.getLogger(__name__)
            if midi_strict:
                logger.exception("MIDI initialization failed (strict): %s", e)
                raise SystemExit(2)
            else:
                logger.warning("MIDI unavailable; falling back to NullMidi: %s", e)
                midi_manager = None
                midi_service = _NullMidi()
                cc_snapshot_fn = midi_service.snapshot
    else:
        midi_manager = None
        # ダミーのスナップショット（常に空のCC）
        midi_service = _NullMidi()
        cc_snapshot_fn = midi_service.snapshot

    # ---- ③ SwapBuffer + Worker/Receiver ---------------------------
    swap_buffer = SwapBuffer()
    worker_pool = WorkerPool(fps=fps, draw_callback=user_draw, cc_snapshot=cc_snapshot_fn, num_workers=workers)
    stream_receiver = StreamReceiver(swap_buffer, worker_pool.result_q)

    # ---- ④ Window & ModernGL --------------------------------------
    rendering_window = RenderWindow(window_width, window_height, bg_color=background)  # type: ignore[abstract]
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

    line_renderer = LineRenderer(mgl_context=mgl_ctx, projection_matrix=proj, double_buffer=swap_buffer)  # type: ignore

    # ---- Draw callbacks ----------------------------------
    rendering_window.add_draw_callback(line_renderer.draw)
    rendering_window.add_draw_callback(overlay.draw)

    # ---- ⑥ FrameCoordinator ---------------------------------------
    frame_clock = FrameClock([midi_service, worker_pool, stream_receiver, line_renderer, sampler, overlay])
    pyglet.clock.schedule_interval(frame_clock.tick, 1 / fps)

    # ---- ⑦ pyglet イベント -----------------------------------------
    @rendering_window.event
    def on_key_press(sym, _mods):  # noqa: ANN001
        if sym == key.ESCAPE:
            rendering_window.close()

    @rendering_window.event
    def on_close():  # noqa: ANN001
        worker_pool.close()
        if use_midi and midi_manager is not None:
            midi_manager.save_cc()
        line_renderer.release()
        pyglet.app.exit()

    pyglet.app.run()
