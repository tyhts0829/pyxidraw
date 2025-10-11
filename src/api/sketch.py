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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import numpy as np

from engine.core.geometry import Geometry
from engine.core.tickable import Tickable
from engine.export.gcode import GCodeParams, GCodeWriter
from engine.ui.hud.config import HUDConfig
from engine.ui.parameters.manager import ParameterManager
from util.constants import CANVAS_SIZES

from .effects import global_cache_counters as _effects_counters
from .shapes import ShapesAPI as _ShapesAPI

# 型エイリアス（RGBA 0..1）
RGBA = tuple[float, float, float, float]


def _hud_metrics_snapshot() -> dict[str, dict[str, int]]:
    """shape/effect のキャッシュ累計を取得する（HUD 用差分の材）。

    multiprocessing の spawn 方式でもシリアライズ可能なトップレベル関数として定義する。
    """
    try:
        s_info = _ShapesAPI.cache_info()
    except Exception:
        s_info = {"hits": 0, "misses": 0}
    try:
        e_info = _effects_counters()
    except Exception:
        e_info = {"compiled": 0, "enabled": 0, "hits": 0, "misses": 0}
    return {
        "shape": {
            "hits": int(s_info.get("hits", 0)),
            "misses": int(s_info.get("misses", 0)),
        },
        "effect": {
            "compiled": int(e_info.get("compiled", 0)),
            "enabled": int(e_info.get("enabled", 0)),
            "hits": int(e_info.get("hits", 0)),
            "misses": int(e_info.get("misses", 0)),
        },
    }


# -----------------------------
# 小規模ヘルパ（Phase 2）
# -----------------------------
def resolve_fps(requested_fps: int | None) -> int:
    """指定がなければ設定から FPS を解決し、失敗時は 60 を返す。"""
    if requested_fps is not None:
        try:
            return int(requested_fps)
        except Exception:
            return 60
    try:
        from util.utils import load_config

        cfg = load_config() or {}
        ccfg = cfg.get("canvas_controller", {}) if isinstance(cfg, dict) else {}
        return int(ccfg.get("fps", 60))
    except Exception:
        return 60


def build_projection(canvas_width: float, canvas_height: float) -> "np.ndarray":
    """キャンバス mm を基準とする正射影行列（ModernGL 用の転置済み）を返す。"""
    proj = np.array(
        [
            [2 / canvas_width, 0, 0, -1],
            [0, -2 / canvas_height, 0, 1],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype="f4",
    ).T
    return proj


def make_gcode_export_handlers(
    *,
    export_service: "ExportService",
    swap_buffer: "SwapBuffer",
    canvas_width: float,
    canvas_height: float,
    overlay: "OverlayHUD | None",
    pyglet_mod: Any,
) -> tuple[Callable[[], None], Callable[[], None]]:
    """G-code エクスポートの開始/キャンセル関数を生成して返す。"""

    _current_g_job: str | None = None

    def _start() -> None:
        nonlocal _current_g_job
        if _current_g_job is not None:
            if overlay is not None:
                overlay.show_message("G-code エクスポート実行中", level="warn")
            return
        front = swap_buffer.get_front()
        if front is None or front.is_empty:
            if overlay is not None:
                overlay.show_message(
                    "G-code エクスポート対象なし（ジオメトリ未生成）", level="warn"
                )
            return
        coords, offsets = front.as_arrays(copy=True)
        try:
            gparams = GCodeParams(
                y_down=True,
                canvas_height_mm=float(canvas_height),
                canvas_width_mm=float(canvas_width),
            )
            _name_prefix = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else None
            job_id = export_service.submit_gcode_job(
                (coords, offsets), params=gparams, simulate=False, name_prefix=_name_prefix
            )
        except RuntimeError:
            if overlay is not None:
                overlay.show_message("G-code エクスポート実行中", level="warn")
            return
        _current_g_job = job_id

        def _poll_progress(_dt: float) -> None:  # noqa: ARG001
            nonlocal _current_g_job
            assert _current_g_job is not None
            prog = export_service.progress(_current_g_job)
            if overlay is not None:
                overlay.set_progress("gcode", prog.done_vertices, prog.total_vertices)
            if prog.state in ("completed", "failed", "cancelled"):
                if overlay is not None:
                    overlay.clear_progress("gcode")
                if prog.state == "completed" and prog.path is not None:
                    if overlay is not None:
                        overlay.show_message(f"Saved G-code: {prog.path}")
                elif prog.state == "failed":
                    if overlay is not None:
                        overlay.show_message(f"G-code 失敗: {prog.error}", level="error")
                elif prog.state == "cancelled":
                    if overlay is not None:
                        overlay.show_message(
                            "G-code エクスポートをキャンセルしました", level="warn"
                        )
                pyglet_mod.clock.unschedule(_poll_progress)
                _current_g_job = None

        pyglet_mod.clock.schedule_interval(_poll_progress, 0.1)

    def _cancel() -> None:
        nonlocal _current_g_job
        if _current_g_job is not None:
            export_service.cancel(_current_g_job)
            if overlay is not None:
                overlay.show_message("G-code エクスポートをキャンセルします", level="warn")

    return _start, _cancel


if TYPE_CHECKING:  # 型チェック専用の軽量 import（実行時は読み込まない）
    from engine.export.service import ExportService
    from engine.runtime.buffer import SwapBuffer
    from engine.ui.hud.overlay import OverlayHUD


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
    hud_config: HUDConfig | None = None,
) -> None:
    """
    user_draw :
        ``t [sec], cc_dict → Geometry`` を返す関数。
    canvas_size :
        既定キー("A4","A5"...）または ``(width, height)`` mm。
    render_scale :
        mm→px のスケーリング係数（px/mm）。float 可。小数点はウィンドウ解像度に丸めて適用。
    line_thickness :
        線の太さ（クリップ空間 -1..1 基準の半幅相当）。既定は 0.0006。
        将来的に mm 指定のサポートを検討（`thickness_clip ≈ 2*mm/canvas_height`）。
    line_color :
        線の色（未指定時は設定の `canvas.line_color` → 既定黒）。
        - RGBA (0–1) タプル、またはヘックス文字列 `#RRGGBB` / `#RRGGBBAA`（`0x`/接頭辞なし可）。
        - RGB（長さ3）の場合は α=1.0 を補完。
    fps :
        描画更新レート。
    background :
        背景色（未指定時は設定の `canvas.background_color` → 既定白）。
        - RGBA (0‑1) タプル、またはヘックス文字列 `#RRGGBB` / `#RRGGBBAA`（`0x`/接頭辞なし可）。
    workers :
        バックグラウンド計算プロセス数。Parameter GUI 併用時もワーカ併用可能（GUI 値はスナップショットで注入）。
    use_midi :
        True の場合は可能なら実機 MIDI を使用。未接続/未導入時は既定でフォールバック。
    use_parameter_gui :
        True で描画パラメータ編集ウィンドウを有効化し、`user_draw` の呼び出しをラップする。
    hud_config :
        HUD の表示設定。None の場合は既定（FPS/VERTEX/CPU/MEM を表示、CACHE は OFF）。
    """
    # ---- ① 設定からFPSを解決 --------------------------------------
    fps = resolve_fps(fps)

    # ---- ② キャンバスサイズ決定 ------------------------------------
    if isinstance(canvas_size, str):
        canvas_width, canvas_height = CANVAS_SIZES[canvas_size.upper()]
    else:
        canvas_width, canvas_height = canvas_size
    window_width, window_height = int(round(canvas_width * render_scale)), int(
        round(canvas_height * render_scale)
    )

    # ---- ③ MIDI ---------------------------------------------------
    def _setup_midi(_use_midi: bool):

        class _NullMidi:
            def snapshot(self) -> Mapping[int, float]:
                return {}

            def tick(self, dt: float) -> None:  # noqa: ARG002
                return None

        if not _use_midi:
            return None, _NullMidi(), _NullMidi().snapshot

        try:
            # 遅延インポート（依存未導入環境でもフォールバック可能に）
            from engine.io.manager import connect_midi_controllers
            from engine.io.service import MidiService

            midi_manager_local = connect_midi_controllers()
            # 0台接続もエラー扱いにするかは strict で切替
            if not getattr(midi_manager_local, "controllers", {}):
                raise RuntimeError("MIDI デバイスが接続されていません")
            midi_service_local = MidiService(midi_manager_local)
            return midi_manager_local, midi_service_local, midi_service_local.snapshot
        except Exception as e:  # ImportError / InvalidPortError / RuntimeError など
            logger = logging.getLogger(__name__)
            logger.warning("MIDI unavailable; falling back to NullMidi: %s", e)
            nm = None
            ns = _NullMidi()
            return nm, ns, ns.snapshot

    midi_manager, midi_service, cc_snapshot_fn = _setup_midi(use_midi)

    # init_only の場合は重い依存を読み込まずに早期リターン
    parameter_manager: ParameterManager | None = None
    # ---- ③.5 Parameter GUI 準備 -----------------------------------
    draw_callable = user_draw
    worker_count = workers
    if use_parameter_gui and not init_only:
        parameter_manager = ParameterManager(user_draw)
        parameter_manager.initialize()
        # ワーカへは生の user_draw を渡し、GUI 値はスナップショットで適用する

    if init_only:
        return None

    # 遅延インポート（ヘッドレス環境でのウィンドウ生成を避ける）
    import moderngl
    import pyglet
    from pyglet.window import key

    from engine.core.frame_clock import FrameClock
    from engine.core.render_window import RenderWindow
    from engine.export.image import save_png
    from engine.export.service import ExportService
    from engine.render.renderer import LineRenderer
    from engine.runtime.buffer import SwapBuffer
    from engine.runtime.receiver import StreamReceiver
    from engine.runtime.worker import WorkerPool
    from engine.ui.hud.overlay import OverlayHUD
    from engine.ui.hud.sampler import MetricSampler

    # Parameter スナップショット適用（spawn 互換のトップレベル関数）
    from engine.ui.parameters.snapshot import apply_param_snapshot, extract_overrides

    # ---- ④ SwapBuffer + Worker/Receiver ---------------------------
    hud_conf: HUDConfig = hud_config or HUDConfig()
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
    if parameter_manager is not None:

        def _param_snapshot_fn():  # type: ignore[no-redef]
            try:
                return extract_overrides(parameter_manager.store)
            except Exception:
                return None

    else:

        def _param_snapshot_fn():  # type: ignore[no-redef]
            return None

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
    on_metrics_cb: Optional[Callable[[Mapping[str, str]], None]] = None
    if hud_conf.enabled and hud_conf.show_cache_status:

        def _on_metrics(flags):  # type: ignore[no-untyped-def]
            try:
                shape_status = str(flags.get("shape", "MISS"))
                effect_status = str(flags.get("effect", "MISS"))
                # 効果 → シェイプの順で更新
                sampler.data["CACHE/EFFECT"] = effect_status
                sampler.data["CACHE/SHAPE"] = shape_status
            except Exception:
                pass

        on_metrics_cb = _on_metrics

    stream_receiver = StreamReceiver(swap_buffer, worker_pool.result_q, on_metrics=on_metrics_cb)

    # ---- ⑤ Window & ModernGL --------------------------------------
    # 色パラメータ正規化（背景/線色）と設定フォールバック
    from util.color import normalize_color as _normalize_color

    try:
        from util.utils import load_config as _load_cfg_colors
    except Exception:  # pragma: no cover - フォールバック
        _load_cfg_colors = lambda: {}  # type: ignore[assignment]

    cfg_all = _load_cfg_colors() or {}
    canvas_cfg = cfg_all.get("canvas", {}) if isinstance(cfg_all, dict) else {}
    cfg_bg = canvas_cfg.get("background_color") if isinstance(canvas_cfg, dict) else None
    cfg_line = canvas_cfg.get("line_color") if isinstance(canvas_cfg, dict) else None

    if background is None:
        bg_src = cfg_bg if cfg_bg is not None else (1.0, 1.0, 1.0, 1.0)
    else:
        bg_src = background
    bg_rgba = _normalize_color(bg_src)
    rendering_window = RenderWindow(window_width, window_height, bg_color=bg_rgba)  # type: ignore[abstract]
    mgl_ctx: moderngl.Context = moderngl.create_context()
    mgl_ctx.enable(moderngl.BLEND)
    mgl_ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # ----  モニタリング ----------------------------------------
    sampler: MetricSampler | None = None
    overlay: OverlayHUD | None = None
    if hud_conf.enabled:
        sampler = MetricSampler(swap_buffer, config=hud_conf)
        overlay = OverlayHUD(rendering_window, sampler, config=hud_conf)
    # G-code エクスポート: 実 writer を接続
    export_service = ExportService(writer=GCodeWriter())

    # ---- ⑥ 投影行列（正射影） --------------------------------------
    proj = build_projection(float(canvas_width), float(canvas_height))

    # 線色を正規化（文字列/タプル → RGBA 0–1）
    if line_color is None:
        if cfg_line is not None:
            lc_src = cfg_line
        else:
            # 背景の輝度に基づいて黒/白を自動選択
            try:
                br, bg_, bb, _ = bg_rgba
                luminance = 0.2126 * float(br) + 0.7152 * float(bg_) + 0.0722 * float(bb)
                lc_src = (0.0, 0.0, 0.0, 1.0) if luminance >= 0.5 else (1.0, 1.0, 1.0, 1.0)
            except Exception:
                lc_src = (0.0, 0.0, 0.0, 1.0)
    else:
        lc_src = line_color
    rgba = _normalize_color(lc_src)

    line_renderer = LineRenderer(
        mgl_context=mgl_ctx,
        projection_matrix=proj,
        swap_buffer=swap_buffer,
        line_thickness=line_thickness,
        line_color=rgba,
    )  # type: ignore

    # ---- 初期色の復帰（Parameter GUI の保存値があれば優先） ---------
    def _apply_initial_colors() -> None:
        if parameter_manager is None:
            return
        from util.color import normalize_color as _norm

        # 背景（store → window）
        bg_val = parameter_manager.store.current_value("runner.background")
        if bg_val is None:
            bg_val = parameter_manager.store.original_value("runner.background")
        if bg_val is not None:
            try:
                rendering_window.set_background_color(_norm(bg_val))
            except Exception:
                pass

        # 線色（store → renderer）。無ければ背景輝度で自動決定
        ln_val = parameter_manager.store.current_value("runner.line_color")
        if ln_val is None:
            ln_val = parameter_manager.store.original_value("runner.line_color")
        if ln_val is not None:
            try:
                line_renderer.set_line_color(_norm(ln_val))
            except Exception:
                pass
        else:
            try:
                br, bg_, bb, _ = rendering_window._bg_color  # type: ignore[attr-defined]
                luminance = 0.2126 * float(br) + 0.7152 * float(bg_) + 0.0722 * float(bb)
                auto = (0.0, 0.0, 0.0, 1.0) if luminance >= 0.5 else (1.0, 1.0, 1.0, 1.0)
                line_renderer.set_line_color(auto)
            except Exception:
                pass

    _apply_initial_colors()
    # HUD 初期適用（存在時）
    if parameter_manager is not None and overlay is not None:
        from util.color import normalize_color as _norm

        tx = parameter_manager.store.current_value(
            "runner.hud_text_color"
        ) or parameter_manager.store.original_value("runner.hud_text_color")
        if tx is not None:
            try:
                overlay.set_text_color(_norm(tx))
            except Exception:
                pass
        mt = parameter_manager.store.current_value(
            "runner.hud_meter_color"
        ) or parameter_manager.store.original_value("runner.hud_meter_color")
        if mt is not None:
            try:
                overlay.set_meter_color(_norm(mt))
            except Exception:
                pass
        mb = parameter_manager.store.current_value(
            "runner.hud_meter_bg_color"
        ) or parameter_manager.store.original_value("runner.hud_meter_bg_color")
        if mb is not None:
            try:
                overlay.set_meter_bg_color(_norm(mb))
            except Exception:
                pass

    # ---- Draw callbacks ----------------------------------
    rendering_window.add_draw_callback(line_renderer.draw)
    if overlay is not None:
        rendering_window.add_draw_callback(overlay.draw)

    # ---- ⑦ FrameCoordinator ---------------------------------------
    tickables: list[Tickable] = [midi_service, worker_pool, stream_receiver, line_renderer]
    if sampler is not None:
        tickables.append(sampler)
    if overlay is not None:
        tickables.append(overlay)
    frame_clock = FrameClock(tickables)
    pyglet.clock.schedule_interval(frame_clock.tick, 1 / fps)

    # ---- ⑨ Parameter GUI からの色変更を監視 --------------------------
    if parameter_manager is not None:

        def _apply_bg_color(_dt: float, raw_val) -> None:  # noqa: ANN001
            try:
                from util.color import normalize_color as _norm

                rendering_window.set_background_color(_norm(raw_val))
            except Exception:
                pass

        def _apply_line_color(_dt: float, raw_val) -> None:  # noqa: ANN001
            try:
                from util.color import normalize_color as _norm

                line_renderer.set_line_color(_norm(raw_val))
            except Exception:
                pass

        from typing import Iterable as _Iterable  # local alias to avoid top clutter

        def _on_param_store_change(ids: Mapping[str, object] | _Iterable[str]) -> None:
            # 受け取る ID 集合へ正規化（想定型に限定して例外を避ける）
            if isinstance(ids, Mapping):
                id_list = list(ids.keys())
            else:
                try:
                    id_list = list(ids)  # type: ignore[list-item]
                except TypeError:
                    id_list = []
            # 背景
            if "runner.background" in id_list:
                try:
                    val = parameter_manager.store.current_value("runner.background")
                    if val is None:
                        val = parameter_manager.store.original_value("runner.background")
                    if val is not None:
                        # GL 操作は pyglet のスケジューラでメインループ側に委譲
                        pyglet.clock.schedule_once(lambda dt, v=val: _apply_bg_color(dt, v), 0.0)
                        # line_color が未指定（override 無し）の場合は自動選択も委譲
                        lc_cur = parameter_manager.store.current_value("runner.line_color")
                        if lc_cur is None:
                            from util.color import normalize_color as _norm

                            br, bg_, bb, _ = _norm(val)
                            luminance = (
                                0.2126 * float(br) + 0.7152 * float(bg_) + 0.0722 * float(bb)
                            )
                            auto = (
                                (0.0, 0.0, 0.0, 1.0) if luminance >= 0.5 else (1.0, 1.0, 1.0, 1.0)
                            )
                            pyglet.clock.schedule_once(
                                lambda dt, v=auto: _apply_line_color(dt, v), 0.0
                            )
                except Exception:
                    pass
            # 線色
            if "runner.line_color" in id_list:
                try:
                    val = parameter_manager.store.current_value("runner.line_color")
                    if val is None:
                        val = parameter_manager.store.original_value("runner.line_color")
                    if val is not None:
                        pyglet.clock.schedule_once(lambda dt, v=val: _apply_line_color(dt, v), 0.0)
                except Exception:
                    pass

            # HUD テキスト色
            if "runner.hud_text_color" in id_list and overlay is not None:
                try:
                    val = parameter_manager.store.current_value("runner.hud_text_color")
                    if val is None:
                        val = parameter_manager.store.original_value("runner.hud_text_color")
                    if val is not None:
                        from util.color import normalize_color as _norm

                        rgba = _norm(val)
                        pyglet.clock.schedule_once(
                            lambda dt, v=rgba: overlay.set_text_color(v), 0.0
                        )
                except Exception:
                    pass

            # HUD メータ色
            if "runner.hud_meter_color" in id_list and overlay is not None:
                try:
                    val = parameter_manager.store.current_value("runner.hud_meter_color")
                    if val is None:
                        val = parameter_manager.store.original_value("runner.hud_meter_color")
                    if val is not None:
                        from util.color import normalize_color as _norm

                        rgba = _norm(val)
                        pyglet.clock.schedule_once(
                            lambda dt, v=rgba: overlay.set_meter_color(v), 0.0
                        )
                except Exception:
                    pass

            # HUD メータ背景色
            if "runner.hud_meter_bg_color" in id_list and overlay is not None:
                try:
                    val = parameter_manager.store.current_value("runner.hud_meter_bg_color")
                    if val is None:
                        val = parameter_manager.store.original_value("runner.hud_meter_bg_color")
                    if val is not None:
                        from util.color import normalize_color as _norm

                        rgba = _norm(val)
                        pyglet.clock.schedule_once(
                            lambda dt, v=rgba: overlay.set_meter_bg_color(v), 0.0
                        )
                except Exception:
                    pass

        try:
            parameter_manager.store.subscribe(_on_param_store_change)
        except Exception:
            pass

    # ---- ⑧ pyglet イベント -----------------------------------------

    def _handle_save_png(mods: int) -> None:
        try:
            # ファイル名のプレフィックス（エントリスクリプト名）とキャンバス寸法 [mm]
            _name_prefix = Path(sys.argv[0]).stem if sys.argv and sys.argv[0] else None
            if mods & key.MOD_SHIFT:
                # 高解像度（overlayなし）: オフスクリーン描画でラインのみ保存
                p = save_png(
                    rendering_window,
                    scale=2.0,
                    include_overlay=False,
                    transparent=False,
                    mgl_context=mgl_ctx,
                    draw=line_renderer.draw,
                    name_prefix=_name_prefix,
                    width_mm=float(canvas_width),
                    height_mm=float(canvas_height),
                )
            else:
                # 低コスト（overlayあり）: 画面バッファをそのまま保存
                p = save_png(
                    rendering_window,
                    scale=1.0,
                    include_overlay=True,
                    name_prefix=_name_prefix,
                    width_mm=float(canvas_width),
                    height_mm=float(canvas_height),
                )
            if overlay is not None:
                overlay.show_message(f"Saved PNG: {p}")
        except Exception as e:  # 失敗時のHUD表示
            if overlay is not None:
                overlay.show_message(f"PNG 保存失敗: {e}", level="error")

    # G-code エクスポートの開始/キャンセルハンドラ（Phase 2: ヘルパへ切り出し）
    _start_gcode_export, _cancel_gcode_export = make_gcode_export_handlers(
        export_service=export_service,
        swap_buffer=swap_buffer,
        canvas_width=float(canvas_width),
        canvas_height=float(canvas_height),
        overlay=overlay,
        pyglet_mod=pyglet,
    )

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

    pyglet.app.run()
