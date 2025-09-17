"""
api.sketch — スケッチ実行・描画ランナー（リアルタイム UI + MIDI 入力）

本モジュールは、ユーザ定義の描画コールバック（`user_draw`）を中核に、
ウィンドウ生成・レンダリング・バックグラウンド計算（ワーカー）・MIDI 入力・
メトリクス表示（HUD）を統合して実行する高水準ランナーを提供する。

主エントリポイント:
- `run_sketch(user_draw, *, canvas_size="A5", render_scale=4, fps=None, ... )`:
  - `user_draw(t: float, cc: Mapping[int, float]) -> Geometry` を一定レートで呼び出し、
    返された `Geometry` を GPU でレンダリングする。
  - ウィンドウは `pyglet`、描画は `ModernGL` を用いたラインレンダラにより行う。
  - バックグラウンド側で `WorkerPool` が `user_draw` を実行し、`SwapBuffer` 経由で
    フレームを主スレッドに受け渡す（`StreamReceiver`）。
  - MIDI 入力（任意）は `engine.io` サブシステムに委譲。未接続/未導入時は自動フォールバック。

実行フロー（概要）:
1) FPS/設定解決: `fps is None` の場合は `util.utils.load_config()` から既定値を取得（なければ 60）。
2) キャンバス設定: `util.constants.CANVAS_SIZES` のキーまたは `(width,height)` [mm] から
   論理サイズを確定し、`render_scale` 倍でピクセルサイズのウィンドウを作成。
3) MIDI 初期化: `use_midi` 有効時、デバイス検出・サービス生成。厳格モード（後述）では
   初期化失敗で `SystemExit(2)`。通常は Null 実装へフォールバックして継続。
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
- `render_scale`: mm→画素のスケーリング。見た目の解像度とアンチエイリアス品質に影響。
- `fps`: 描画更新レート。`None` で設定ファイルから解決、未設定時は 60。
- `background`: RGBA (0–1)。ウィンドウの背景色。
- `workers`: バックグラウンド計算の並列度（CPU コア/負荷に応じて調整）。
- `use_midi`: True で実機 MIDI を試行。未接続/未導入時は警告とともに Null 実装へ。
- `midi_strict`: True で初期化失敗時に即終了。None なら環境変数を参照（下記）。
- `init_only`: True で重い依存の初期化をスキップし、作成フェーズの検証だけを行って終了。

環境変数・設定:
- `PYXIDRAW_MIDI_STRICT`: `"1"/"true"/"on"/"yes"` で厳格モードを有効化。
- 設定ファイル（`util.utils.load_config()` が読み取る YAML）から `fps` と
  `midi.strict_default` を補完可能（読み込み失敗時は安全側の既定にフォールバック）。

スレッド/プロセス・安全性:
- 本モジュールは UI イベントループ（`pyglet`）を主スレッドで回し、
  幾何生成（`user_draw`）は `WorkerPool` に委譲してメインループから切り離す。
- 受け渡しには `SwapBuffer` を利用し、フレーム境界での整合性を保つ。
- MIDI は `Tickable` として `FrameClock` に統合され、`snapshot()` により CC 値を取得。

例（最小スケッチ）:
    from api.sketch import run_sketch
    import numpy as np
    from engine.core.geometry import Geometry

    def user_draw(t, cc):
        # 半径を時間と CC#1 で変調した円
        r = 50 + 30 * float(cc.get(1, 0.0))
        theta = np.linspace(0, 2*np.pi, 200, endpoint=False)
        xy = np.c_[r*np.cos(theta), r*np.sin(theta)]
        return Geometry.from_lines([xy])

    run_sketch(user_draw, canvas_size="A5", render_scale=4, fps=60)

注意/制限:
- 3D ではなく 2D 線の正射影描画を前提としている。Z は重なり順の補助程度。
- ヘッドレス/仮想環境では `pyglet`/`ModernGL` の初期化に失敗する場合がある。
- 厳格 MIDI モードではデバイス未検出等で即時終了する（非厳格時は警告ログのみ）。

ロギング:
- 初期化エラーやフォールバックは `logging` で通知。必要に応じてハンドラを設定すること。
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Mapping

import numpy as np

from engine.core.geometry import Geometry
from engine.core.tickable import Tickable
from util.constants import CANVAS_SIZES

from engine.ui.parameters.manager import ParameterManager


def run_sketch(
    user_draw: Callable[[float, Mapping[int, float]], Geometry],
    *,
    canvas_size: str | tuple[int, int] = "A5",
    render_scale: int = 4,
    fps: int | None = None,
    background: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    workers: int = 4,
    use_midi: bool = True,
    midi_strict: bool | None = None,
    init_only: bool = False,
    use_parameter_gui: bool = False,
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
        バックグラウンド計算プロセス数（パラメータ GUI 有効時は 0 固定でシングルスレッド実行）。
    use_midi :
        True の場合は可能なら実機 MIDI を使用。未接続/未導入時は既定でフォールバック。
    midi_strict :
        True で厳格モード（初期化失敗時に SystemExit(2)）。None の場合は
        環境変数 ``PYXIDRAW_MIDI_STRICT`` を参照（未設定は False）。
    use_parameter_gui :
        True で描画パラメータ編集ウィンドウを有効化し、`user_draw` の呼び出しをラップする。
    """
    # ---- ① 設定からFPSを解決 --------------------------------------
    # 引数fpsがNoneのときだけ、設定 (configs/default.yaml 等) を参照
    if fps is None:
        try:
            from util.utils import load_config  # noqa: WPS433

            cfg = load_config() or {}
            ccfg = cfg.get("canvas_controller", {}) if isinstance(cfg, dict) else {}
            # 不正値/未設定に備えつつ int へ
            fps = int(ccfg.get("fps", 60))
        except Exception:
            fps = 60

    # ---- ② キャンバスサイズ決定 ------------------------------------
    if isinstance(canvas_size, str):
        canvas_width, canvas_height = CANVAS_SIZES[canvas_size.upper()]
    else:
        canvas_width, canvas_height = canvas_size
    window_width, window_height = int(canvas_width * render_scale), int(
        canvas_height * render_scale
    )

    # ---- ③ MIDI ---------------------------------------------------
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
        def snapshot(self) -> Mapping[int, float]:  # CC は int→float の正規化値
            return {}

        def tick(self, dt: float) -> None:
            return None

    midi_service: Tickable
    cc_snapshot_fn: Callable[[], Mapping[int, float]]

    if use_midi:
        try:
            # 遅延インポート（依存未導入環境でもフォールバック可能に）
            from engine.io.manager import connect_midi_controllers  # noqa: WPS433
            from engine.io.service import MidiService  # noqa: WPS433

            midi_manager = connect_midi_controllers()
            # 0台接続もエラー扱いにするかは strict で切替
            if not getattr(midi_manager, "controllers", {}):
                raise RuntimeError("MIDI デバイスが接続されていません")
            midi_service = MidiService(midi_manager)
            # MidiService は Tickable を実装し、snapshot() を提供する。
            # 型に合わせて渡す。
            cc_snapshot_fn = midi_service.snapshot
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

    # init_only の場合は重い依存を読み込まずに早期リターン
    parameter_manager: ParameterManager | None = None
    if use_parameter_gui and not init_only:
        try:
            initial_cc = dict(cc_snapshot_fn())
        except Exception:
            initial_cc = {}
        parameter_manager = ParameterManager(user_draw)
        parameter_manager.initialize(initial_cc)
        draw_callable = parameter_manager.draw
        worker_count = 0
    else:
        draw_callable = user_draw
        worker_count = workers

    if init_only:
        return None

    # 遅延インポート（ヘッドレス環境でのウィンドウ生成を避ける）
    import moderngl
    import pyglet
    from pyglet.window import key

    from engine.core.frame_clock import FrameClock
    from engine.core.render_window import RenderWindow
    from engine.runtime.buffer import SwapBuffer
    from engine.runtime.receiver import StreamReceiver
    from engine.runtime.worker import WorkerPool
    from engine.render.renderer import LineRenderer
    from engine.ui.monitor import MetricSampler
    from engine.ui.overlay import OverlayHUD

    # ---- ④ SwapBuffer + Worker/Receiver ---------------------------
    swap_buffer = SwapBuffer()
    worker_pool = WorkerPool(
        fps=fps,
        draw_callback=draw_callable,
        cc_snapshot=cc_snapshot_fn,
        num_workers=worker_count,
    )
    stream_receiver = StreamReceiver(swap_buffer, worker_pool.result_q)

    # ---- ⑤ Window & ModernGL --------------------------------------
    rendering_window = RenderWindow(window_width, window_height, bg_color=background)  # type: ignore[abstract]
    mgl_ctx: moderngl.Context = moderngl.create_context()
    mgl_ctx.enable(moderngl.BLEND)
    mgl_ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # ----  モニタリング ----------------------------------------
    sampler = MetricSampler(swap_buffer)
    overlay = OverlayHUD(rendering_window, sampler)

    # ---- ⑥ 投影行列（正射影） --------------------------------------
    proj = np.array(
        [
            [2 / canvas_width, 0, 0, -1],
            [0, -2 / canvas_height, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    ).T  # 転置を適用

    line_renderer = LineRenderer(
        mgl_context=mgl_ctx,
        projection_matrix=proj,
        double_buffer=swap_buffer,
    )  # type: ignore

    # ---- Draw callbacks ----------------------------------
    rendering_window.add_draw_callback(line_renderer.draw)
    rendering_window.add_draw_callback(overlay.draw)

    # ---- ⑦ FrameCoordinator ---------------------------------------
    frame_clock = FrameClock(
        [midi_service, worker_pool, stream_receiver, line_renderer, sampler, overlay]
    )
    pyglet.clock.schedule_interval(frame_clock.tick, 1 / fps)

    # ---- ⑧ pyglet イベント -----------------------------------------
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
        if parameter_manager is not None:
            parameter_manager.shutdown()
        pyglet.app.exit()

    pyglet.app.run()
