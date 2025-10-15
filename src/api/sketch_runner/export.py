"""
どこで: `api.sketch_runner.export`
何を: PNG/G-code エクスポートのヘルパ（開始/キャンセル/保存ラッパ）。
なぜ: `api.sketch` からエクスポート関連の責務を分離し、API 層を薄く保つため。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from engine.export.gcode import GCodeParams

if TYPE_CHECKING:  # 実行時依存を避けるための型ヒントのみ
    from engine.export.service import ExportService
    from engine.runtime.buffer import SwapBuffer
    from engine.ui.hud.overlay import OverlayHUD


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

    import sys

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


def save_png_screen_or_offscreen(
    rendering_window,
    *,
    mode: str,
    mgl_context=None,
    draw: Callable[[], None] | None = None,
    name_prefix: str | None = None,
    width_mm: float | None = None,
    height_mm: float | None = None,
):
    """PNG 保存を行う（画面コピー or オフスクリーン高解像度）。

    Parameters
    ----------
    rendering_window : Any
        RenderWindow インスタンス。
    mode : {"screen","quality"}
        "screen" は HUD 含む画面コピー、"quality" は HUD なしの FBO 描画。
    mgl_context : moderngl.Context | None
        quality モード時に必要。
    draw : Callable[[], None] | None
        quality モード時に必要（ラインのみを描画する関数）。
    name_prefix : str | None
        保存ファイル名のプレフィックス。
    width_mm, height_mm : float | None
        キャンバス寸法（mm）。
    """
    from engine.export.image import save_png

    if mode == "quality":
        if mgl_context is None or draw is None:
            raise ValueError("quality mode requires mgl_context and draw callable")
        return save_png(
            rendering_window,
            scale=2.0,
            include_overlay=False,
            transparent=False,
            mgl_context=mgl_context,
            draw=draw,
            name_prefix=name_prefix,
            width_mm=float(width_mm) if width_mm is not None else None,
            height_mm=float(height_mm) if height_mm is not None else None,
        )
    elif mode == "screen":
        return save_png(
            rendering_window,
            scale=1.0,
            include_overlay=True,
            name_prefix=name_prefix,
            width_mm=float(width_mm) if width_mm is not None else None,
            height_mm=float(height_mm) if height_mm is not None else None,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")


__all__ = ["make_gcode_export_handlers", "save_png_screen_or_offscreen"]
