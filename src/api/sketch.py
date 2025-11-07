"""
どこで: `api.sketch`（実行ランナー）。
何を: ユーザの `user_draw(t)->Geometry` をワーカで駆動し、GL でレンダ・HUD 表示・MIDI 入力を統合。
なぜ: 少ない記述で対話的なスケッチ実行と計測を可能にするため（UI/MIDI は任意で自動フォールバック）。

api.sketch — スケッチ実行・描画ランナー（リアルタイム UI + MIDI 入力）

本モジュールは、ユーザ定義の描画コールバック（`user_draw`）を中核に、
ウィンドウ生成・レンダリング・バックグラウンド計算（ワーカー）・MIDI 入力・
メトリクス表示（HUD）を統合して実行する高水準ランナーを提供する。

主エントリポイント:
- `run_sketch(user_draw, *, canvas_size="A5", render_scale=4, fps=None, ... )`:
  - `user_draw(t: float) -> Geometry` を一定レートで呼び出し、
    返された `Geometry` を GPU でレンダリングする。
  - ウィンドウは `pyglet`、描画は `ModernGL` を用いたラインレンダラにより行う。
  - バックグラウンド側で `WorkerPool` が `user_draw` を実行し、`SwapBuffer` 経由で
    フレームを主スレッドに受け渡す（`StreamReceiver`）。
  - MIDI 入力（任意）は `engine.io` サブシステムに委譲。未接続/未導入時は自動フォールバック。

実行フロー（概要）:
1) FPS/設定解決: `fps is None` の場合は `util.utils.load_config()` から既定値を取得（なければ 60）。
2) キャンバス設定: `util.constants.CANVAS_SIZES` のキーまたは `(width,height)` [mm] から
   論理サイズを確定し、`render_scale` 倍でピクセルサイズのウィンドウを作成。
3) MIDI 初期化: `use_midi` 有効時、デバイス検出・サービス生成。失敗時は Null 実装へフォールバックして継続。
4) パイプライン基盤: `SwapBuffer`・`WorkerPool`・`StreamReceiver` を結線し、
   ワーカーが生成した `Geometry` を非同期に受け取る。
5) ウィンドウ/GL: `RenderWindow` を生成し、`ModernGL` のブレンドを有効化。
6) 投影行列: キャンバス [mm] を直接座標系とする正射影行列を構築（Y 上向きを画面座標へ反映）。
7) 監視/HUD: `MetricSampler` と `OverlayHUD` をセットアップし、描画コールバックに登録。
8) フレーム駆動: `FrameClock` により各コンポーネントの `tick(dt)` を `pyglet.clock` で駆動。
   `ESC` でウィンドウを閉じ、ワーカー停止・MIDI 保存・GL リソース解放を行う。

引数の意味（要点）:
- `user_draw`: 時刻 `t` [sec] と CC 値辞書（0–127 → 0.0–1.0）を受け取り `Geometry` を返す純関数。
- `canvas_size`: `"A4"/"A5"/...` などのプリセット名、または `(width_mm, height_mm)` タプル。
- `render_scale`: mm→画素のスケーリング（px/mm, float 可）。見た目の解像度とアンチエイリアス品質に影響。
- `fps`: 描画更新レート。`None` で設定ファイルから解決、未設定時は 60。
- `background`: RGBA (0–1)。ウィンドウの背景色。
- `workers`: バックグラウンド計算の並列度（CPU コア/負荷に応じて調整）。
- `use_midi`: True で実機 MIDI を試行。未接続/未導入時は警告とともに Null 実装へ。
- `init_only`: True で重い依存の初期化をスキップし、作成フェーズの検証だけを行って終了。

環境変数・設定:
- 設定ファイル（`util.utils.load_config()` が読み取る YAML）から `fps` を補完可能（読み込み失敗時は安全側の既定にフォールバック）。

スレッド/プロセス・安全性:
- 本モジュールは UI イベントループ（`pyglet`）を主スレッドで回し、
  幾何生成（`user_draw`）は `WorkerPool` に委譲してメインループから切り離す。
- 受け渡しには `SwapBuffer` を利用し、フレーム境界での整合性を保つ。
- MIDI は `Tickable` として `FrameClock` に統合され、`snapshot()` により CC 値を取得。

例（最小スケッチ）:
    from api.sketch import run_sketch
    import numpy as np
    from engine.core.geometry import Geometry

    def user_draw(t):
        # 半径を時間と CC#1 で変調した円
        from api import cc
        r = 50 + 30 * (cc[1] * 1.0)
        theta = np.linspace(0, 2*np.pi, 200, endpoint=False)
        xy = np.c_[r*np.cos(theta), r*np.sin(theta)]
        return Geometry.from_lines([xy])

    run_sketch(user_draw, canvas_size="A5", render_scale=4, fps=60)

注意/制限:
- 3D ではなく 2D 線の正射影描画を前提としている。Z は重なり順の補助程度。
- ヘッドレス/仮想環境では `pyglet`/`ModernGL` の初期化に失敗する場合がある。


ロギング:
- 初期化エラーやフォールバックは `logging` で通知。必要に応じてハンドラを設定すること。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

from engine.core.geometry import Geometry
from engine.core.tickable import Tickable
from engine.ui.hud.config import HUDConfig
from engine.ui.parameters.manager import ParameterManager

from .sketch_runner.utils import build_projection
from .sketch_runner.utils import hud_metrics_snapshot as _hud_metrics_snapshot
from .sketch_runner.utils import resolve_canvas_size, resolve_fps

# 型エイリアス（RGBA 0..1）
RGBA = tuple[float, float, float, float]
logger = logging.getLogger(__name__)


"""
メモ: resolve_fps/build_projection/hud_metrics_snapshot は
`api.sketch_runner.utils` へ移設（トップレベルの純粋関数）。
"""


def run_sketch(
    user_draw: Callable[[float], Geometry],
    *,
    canvas_size: str | tuple[int, int] = "A5",
    render_scale: float = 4.0,
    line_thickness: float = 0.0006,
    line_color: str | tuple[float, float, float] | tuple[float, float, float, float] | None = None,
    fps: int | None = None,
    background: str | tuple[float, float, float, float] | None = None,
    workers: int = 4,
    use_midi: bool = True,
    init_only: bool = False,
    use_parameter_gui: bool = False,
    show_hud: bool | None = None,
    hud_config: HUDConfig | None = None,
) -> None:
    """スケッチを実行し、`user_draw` の結果を GPU で描画する。

    Parameters
    ----------
    user_draw : Callable[[float], Geometry]
        時刻 `t` [sec] を受け取り `Geometry` を返す純関数。
        CC は `from api import cc; cc[i]` で参照する。
    canvas_size : str | tuple[int, int], default "A5"
        プリセット（例: "A4", "A5"）または `(width_mm, height_mm)`。
    render_scale : float, default 4.0
        mm→px のスケーリング（px/mm）。ウィンドウ解像度に丸め、1px以上にクランプ。
    line_thickness : float, default 0.0006
        クリップ空間（-1..1）の半幅相当。目安: thickness_clip ≈ 2*mm/canvas_height。
    line_color : tuple[float,float,float,(float)] | str | None
        線色（RGBA 0–1 または #RRGGBB/#RRGGBBAA）。None で自動/設定を適用。
    fps : int | None
        描画更新レート。None で設定ファイルから解決。最終的に 1 以上にクランプ。
    background : tuple[float,float,float,(float)] | str | None
        背景色（RGBA 0–1 または #RRGGBB/#RRGGBBAA）。None で設定/白を適用。
    workers : int, default 4
        バックグラウンド計算プロセス数（0 でインライン）。負値は 0 にクランプ。
    use_midi : bool, default True
        True で実機 MIDI を試行。未接続/未導入時は自動フォールバック。
    use_parameter_gui : bool, default False
        True で描画パラメータ GUI を有効化。
    show_hud : bool | None, default None
        HUD の有効/無効。None で上書きしない（従来の既定/`hud_config` を尊重）。
        優先順位は「show_hud 明示 > hud_config.enabled > 既定(True)」。
    hud_config : HUDConfig | None
        HUD 表示設定。None で既定（FPS/VERTEX/CPU/MEM 表示、CACHE OFF）。

    Notes
    -----
    - 2D 線の正射影描画を対象。ヘッドレス環境では初期化に失敗する場合がある。
    - `ESC` で終了し、ワーカ停止・MIDI 保存・GL リソース解放を行う。
    """
    # ---- ① 設定からFPSを解決 --------------------------------------
    fps = resolve_fps(fps)

    # ---- ② キャンバスサイズ決定 ------------------------------------
    canvas_width, canvas_height = resolve_canvas_size(canvas_size)
    if float(render_scale) <= 0.0:
        raise ValueError(f"render_scale must be > 0, got {render_scale}")
    window_width = max(1, int(round(canvas_width * render_scale)))
    window_height = max(1, int(round(canvas_height * render_scale)))

    # ---- ③ MIDI ---------------------------------------------------
    from .sketch_runner.midi import setup_midi as _setup_midi

    midi_manager, midi_service, cc_snapshot_fn = _setup_midi(use_midi)

    # init_only の場合は重い依存を読み込まずに早期リターン
    parameter_manager: ParameterManager | None = None
    # ---- ③.5 Parameter GUI 準備 -----------------------------------
    draw_callable = user_draw
    worker_count = max(0, int(workers))
    if use_parameter_gui and not init_only:
        parameter_manager = ParameterManager(user_draw)
        parameter_manager.initialize()
        # ワーカへは生の user_draw を渡し、GUI 値はスナップショットで適用する

    if init_only:
        return None

    # 遅延インポート（ヘッドレス環境でのウィンドウ生成を避ける）
    import pyglet
    from pyglet.window import key

    from engine.core.frame_clock import FrameClock

    # 画像/G-code エクスポートは内部ヘルパへ委譲
    from engine.export.service import ExportService
    from engine.export.video import VideoRecorder
    from engine.runtime.buffer import SwapBuffer
    from engine.runtime.receiver import StreamReceiver
    from engine.runtime.worker import WorkerPool
    from engine.ui.hud.overlay import OverlayHUD
    from engine.ui.hud.sampler import MetricSampler

    # Parameter スナップショット適用（spawn 互換のトップレベル関数）
    from engine.ui.parameters.snapshot import apply_param_snapshot

    from .sketch_runner.export import make_gcode_export_handlers, save_png_screen_or_offscreen
    from .sketch_runner.params import apply_initial_colors as _apply_initial_colors
    from .sketch_runner.params import make_param_snapshot_fn as _make_param_snapshot_fn
    from .sketch_runner.params import subscribe_color_changes as _subscribe_color_changes

    # ---- ④ SwapBuffer + Worker/Receiver ---------------------------
    # ---- HUD 設定の解決（優先: show_hud 明示 > hud_config.enabled > 既定 True）----
    if hud_config is None:
        if show_hud is None:
            hud_conf: HUDConfig = HUDConfig()
        else:
            hud_conf = HUDConfig(enabled=bool(show_hud))
    else:
        hud_conf = hud_config if show_hud is None else replace(hud_config, enabled=bool(show_hud))
    swap_buffer = SwapBuffer()
    # API 層で CC スナップショット適用関数を注入（engine は api を知らない）
    try:
        from api.cc import set_snapshot as _apply_cc_snapshot
    except Exception:  # pragma: no cover - フォールバック
        _apply_cc_snapshot = None  # type: ignore[assignment]

    # メトリクス収集（HUD 用）。HUD/CACHE 無効時は None を渡す。
    # 注意: macOS 等の spawn 環境では、ワーカープロセスへ渡す関数は
    # ピクル可能（トップレベル定義）である必要がある。
    metrics_snapshot_fn = (
        _hud_metrics_snapshot if (hud_conf.enabled and hud_conf.show_cache_status) else None
    )

    # GUI の override のみを抽出するスナップショット関数
    _param_snapshot_fn = _make_param_snapshot_fn(parameter_manager, cc_snapshot_fn)

    worker_pool = WorkerPool(
        fps=fps,
        draw_callback=draw_callable,
        cc_snapshot=cc_snapshot_fn,
        apply_cc_snapshot=_apply_cc_snapshot,
        num_workers=worker_count,
        apply_param_snapshot=apply_param_snapshot,
        param_snapshot=_param_snapshot_fn,
        metrics_snapshot=metrics_snapshot_fn,
    )

    # HUD: キャッシュ HIT/MISS を受け取って更新（有効時のみ）
    on_metrics_cb: Callable[[Mapping[str, str]], None] | None = None
    if hud_conf.enabled and hud_conf.show_cache_status:

        def _on_metrics(flags):  # type: ignore[no-untyped-def]
            try:
                if sampler is None:
                    return
                prev_e = str(sampler.data.get("E_CACHE", "")).upper()
                prev_s = str(sampler.data.get("S_CACHE", "")).upper()
                effect_status = str(flags.get("effect", prev_e or "MISS")).upper()
                shape_status = str(flags.get("shape", prev_s or "MISS")).upper()
                # 効果 → シェイプの順で更新
                sampler.data["E_CACHE"] = effect_status
                sampler.data["S_CACHE"] = shape_status
            except Exception as e:
                logger.debug("hud cache status update failed: %s", e, exc_info=True)

        on_metrics_cb = _on_metrics

    stream_receiver = StreamReceiver(swap_buffer, worker_pool.result_q, on_metrics=on_metrics_cb)

    # ---- ⑤ Window & ModernGL --------------------------------------
    # ⑥ 投影行列（正射影）を先に構築
    proj = build_projection(float(canvas_width), float(canvas_height))

    from .sketch_runner.render import create_window_and_renderer

    rendering_window, mgl_ctx, line_renderer, _bg_rgba, _line_rgba = create_window_and_renderer(
        window_width,
        window_height,
        background=background,
        line_color=line_color,
        projection_matrix=proj,
        swap_buffer=swap_buffer,
        line_thickness=line_thickness,
    )

    # ----  モニタリング ----------------------------------------
    sampler: MetricSampler | None = None
    overlay: OverlayHUD | None = None
    if hud_conf.enabled:
        sampler = MetricSampler(swap_buffer, config=hud_conf)
        overlay = OverlayHUD(rendering_window, sampler, config=hud_conf)
        # HUD が LazyGeometry を実体化しないよう、Renderer の実測値を参照させる
        try:
            sampler.set_counts_provider(line_renderer.get_last_counts)
        except Exception:
            pass
        # 追加メトリクス（IBO/Indices LRU）の提供
        try:
            from engine.render.renderer import get_indices_cache_counters as _idx_counters

            def _extra_metrics():  # type: ignore[no-redef]
                d = {}
                try:
                    s = line_renderer.get_ibo_stats()
                    d["ibo_reused"] = int(s.get("reused", 0))
                    d["ibo_uploaded"] = int(s.get("uploaded", 0))
                    d["indices_built"] = int(s.get("indices_built", 0))
                except Exception:
                    pass
                try:
                    c = _idx_counters()
                    d["idx_hits"] = int(c.get("hits", 0))
                    d["idx_misses"] = int(c.get("misses", 0))
                    d["idx_stores"] = int(c.get("stores", 0))
                    d["idx_evicts"] = int(c.get("evicts", 0))
                    d["idx_size"] = int(c.get("size", 0))
                except Exception:
                    pass
                return d

            sampler.set_extra_metrics_provider(_extra_metrics)
        except Exception:
            pass
    # G-code エクスポート: 実 writer を接続（遅延 import）
    from engine.export.gcode import GCodeWriter  # 遅延 import（重依存の統一方針）

    export_service = ExportService(writer=GCodeWriter())

    # （line_renderer は create_window_and_renderer で初期化済み）

    # ---- 初期色の復帰（Parameter GUI の保存値があれば優先） ---------

    _apply_initial_colors(parameter_manager, rendering_window, line_renderer, overlay)

    # ---- Draw callbacks ----------------------------------
    # 品質最優先モード（Shift+V録画中）はウィンドウへは描画しない（FBO のみ）。
    quality_recording: bool = False

    def _draw_main() -> None:
        if not quality_recording:
            line_renderer.draw()
            if overlay is not None:
                overlay.draw()

    rendering_window.add_draw_callback(_draw_main)

    # ---- ⑦ FrameCoordinator ---------------------------------------
    tickables: list[Tickable] = [midi_service, worker_pool, stream_receiver, line_renderer]
    if sampler is not None:
        tickables.append(sampler)
    if overlay is not None:
        tickables.append(overlay)
    frame_clock = FrameClock(tickables)
    pyglet.clock.schedule_interval(frame_clock.tick, 1 / fps)
    quality_tick_cb: Callable[[float], None] | None = None

    # ---- ⑨ Parameter GUI からの色変更を監視 --------------------------
    _subscribe_color_changes(parameter_manager, overlay, line_renderer, rendering_window, pyglet)

    # ---- ⑧ pyglet イベント -----------------------------------------

    def _handle_save_png(mods: int) -> None:
        try:
            # ファイル名のプレフィックス（エントリスクリプト名）とキャンバス寸法 [mm]
            _name_prefix = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else None
            if mods & key.MOD_SHIFT:
                # 高解像度（overlayなし）: オフスクリーン描画でラインのみ保存
                p = save_png_screen_or_offscreen(
                    rendering_window,
                    mode="quality",
                    mgl_context=mgl_ctx,
                    draw=line_renderer.draw,
                    name_prefix=_name_prefix,
                    width_mm=float(canvas_width),
                    height_mm=float(canvas_height),
                )
            else:
                # 低コスト（overlayあり）: 画面バッファをそのまま保存
                p = save_png_screen_or_offscreen(
                    rendering_window,
                    mode="screen",
                    name_prefix=_name_prefix,
                    width_mm=float(canvas_width),
                    height_mm=float(canvas_height),
                )
            if overlay is not None:
                overlay.show_message(f"Saved PNG: {p}")
        except Exception as e:  # 失敗時のHUD表示
            if overlay is not None:
                overlay.show_message(f"PNG 保存失敗: {e}", level="error")

    # G-code エクスポートの開始/キャンセルハンドラ（ヘルパへ委譲）
    _start_gcode_export, _cancel_gcode_export = make_gcode_export_handlers(
        export_service=export_service,
        swap_buffer=swap_buffer,
        canvas_width=float(canvas_width),
        canvas_height=float(canvas_height),
        overlay=overlay,
        pyglet_mod=pyglet,
    )

    # ---- ⑤.5 Video Recorder -------------------------------------
    video_recorder = VideoRecorder()

    def _enter_quality_mode() -> None:
        nonlocal worker_pool, stream_receiver, frame_clock, quality_recording, quality_tick_cb
        from .sketch_runner.recording import enter_quality_mode as _enter_q

        worker_pool, stream_receiver, frame_clock, quality_tick_cb = _enter_q(
            fps=fps,
            draw_callable=draw_callable,
            cc_snapshot_fn=cc_snapshot_fn,
            apply_cc_snapshot=_apply_cc_snapshot,
            apply_param_snapshot=apply_param_snapshot,
            param_snapshot_fn=_param_snapshot_fn,
            metrics_snapshot_fn=metrics_snapshot_fn,
            swap_buffer=swap_buffer,
            on_metrics_cb=on_metrics_cb,
            midi_service=midi_service,
            sampler=sampler,
            overlay=overlay,
            line_renderer=line_renderer,
            worker_pool=worker_pool,
            stream_receiver=stream_receiver,
            frame_clock=frame_clock,
            pyglet_mod=pyglet,
        )
        quality_recording = True

    def _leave_quality_mode() -> None:
        nonlocal worker_pool, stream_receiver, frame_clock, quality_recording, quality_tick_cb
        from .sketch_runner.recording import leave_quality_mode as _leave_q

        worker_pool, stream_receiver, frame_clock = _leave_q(
            fps=fps,
            draw_callable=draw_callable,
            cc_snapshot_fn=cc_snapshot_fn,
            apply_cc_snapshot=_apply_cc_snapshot,
            apply_param_snapshot=apply_param_snapshot,
            param_snapshot_fn=_param_snapshot_fn,
            metrics_snapshot_fn=metrics_snapshot_fn,
            swap_buffer=swap_buffer,
            on_metrics_cb=on_metrics_cb,
            midi_service=midi_service,
            sampler=sampler,
            overlay=overlay,
            line_renderer=line_renderer,
            worker_count=worker_count,
            quality_tick_cb=quality_tick_cb,
            pyglet_mod=pyglet,
        )
        quality_tick_cb = None
        quality_recording = False

    @rendering_window.event
    def on_key_press(sym, mods):  # noqa: ANN001
        if sym == key.ESCAPE:
            rendering_window.close()
        # PNG 保存（P / Shift+P）
        if sym == key.P:
            _handle_save_png(mods)
        # G-code 保存（G / Shift+G）
        if sym == key.G and not (mods & key.MOD_SHIFT):
            _start_gcode_export()
        # Shift+G → キャンセル
        if sym == key.G and (mods & key.MOD_SHIFT):
            _cancel_gcode_export()
        # Video 録画トグル（V）
        if sym == key.V:
            try:
                if not video_recorder.is_recording:
                    _name_prefix = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else None
                    if mods & key.MOD_SHIFT:
                        # HUD を含まない録画（FBO 経由、ラインのみ）
                        video_recorder.start(
                            rendering_window,
                            fps=fps,
                            name_prefix=_name_prefix,
                            include_overlay=False,
                            mgl_context=mgl_ctx,
                            draw=line_renderer.draw,
                        )
                        # 品質最優先モードへ移行
                        _enter_quality_mode()
                        if overlay is not None:
                            overlay.show_message("品質最優先モード")
                    else:
                        # 画面そのまま（HUD 含む）
                        video_recorder.start(
                            rendering_window,
                            fps=fps,
                            name_prefix=_name_prefix,
                            include_overlay=True,
                        )
                    if overlay is not None:
                        overlay.show_message("REC 開始")
                        try:
                            overlay.set_recording(True)
                        except Exception:
                            pass
                else:
                    p = video_recorder.stop()
                    # 品質最優先モードを解除
                    _leave_quality_mode()
                    if overlay is not None:
                        overlay.show_message(f"Saved MP4: {p}")
                        try:
                            overlay.set_recording(False)
                        except Exception:
                            pass
            except Exception as e:
                if overlay is not None:
                    overlay.show_message(f"録画エラー: {e}", level="error")

    @rendering_window.event
    def on_close():  # noqa: ANN001
        # 冪等なクリーンアップ
        _closed = getattr(on_close, "_closed", False)
        if _closed:  # type: ignore[truthy-bool]
            return
        try:
            worker_pool.close()
        except Exception:
            pass
        try:
            _leave_quality_mode()
        except Exception:
            pass
        try:
            if video_recorder.is_recording:
                p = video_recorder.stop()
                if overlay is not None:
                    overlay.show_message(f"Saved MP4: {p}")
        except Exception:
            pass
        try:
            if use_midi and midi_manager is not None:
                midi_manager.save_cc()
        except Exception:
            pass
        try:
            line_renderer.release()
        except Exception:
            pass
        try:
            if parameter_manager is not None:
                parameter_manager.shutdown()
        except Exception:
            pass
        setattr(on_close, "_closed", True)
        pyglet.app.exit()

    # 録画フック（最後に呼ぶ）。録画中のみフレームを取り出す。
    def _capture_frame():
        try:
            video_recorder.capture_current_frame(rendering_window)
            # 品質最優先モード中は、FBO→screen ブリット後に HUD を重ねる
            if quality_recording and overlay is not None:
                try:
                    overlay.draw()
                except Exception:
                    pass
        except Exception as e:
            # 一度でも失敗したら停止を試み、HUD に通知
            try:
                if video_recorder.is_recording:
                    p = video_recorder.stop()
                    if overlay is not None:
                        overlay.show_message(f"録画を停止しました: {p}")
            except Exception:
                pass
            if overlay is not None:
                overlay.show_message(f"録画フレーム取得失敗: {e}", level="error")

    rendering_window.add_draw_callback(_capture_frame)

    pyglet.app.run()
