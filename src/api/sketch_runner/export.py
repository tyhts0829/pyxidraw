"""
どこで: `api.sketch_runner.export`
何を: PNG/G-code エクスポートのヘルパ（開始/キャンセル/保存ラッパ）。
なぜ: `api.sketch` からエクスポート関連の責務を分離し、API 層を薄く保つため。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import StyledLayer
from engine.export.gcode import GCodeParams
from engine.runtime.frame import RenderFrame

if TYPE_CHECKING:  # 実行時依存を避けるための型ヒントのみ
    from engine.export.service import ExportService
    from engine.runtime.buffer import SwapBuffer
    from engine.ui.hud.overlay import OverlayHUD


def _normalize_front_to_geometry(
    front: object,
) -> Geometry | LazyGeometry | None:
    """SwapBuffer.get_front() の戻り値を Geometry ベースに正規化する。

    - Geometry / LazyGeometry: そのまま返す。
    - RenderFrame.layers: 各レイヤーの geometry を実体 Geometry に揃え、concat して 1 つにまとめる。
    - それ以外/空: None を返す。
    """
    if front is None:
        return None

    if isinstance(front, RenderFrame):
        if front.layers:
            return _normalize_front_to_geometry(front.layers)
        return (
            front.geometry
            if isinstance(front.geometry, (Geometry, LazyGeometry))
            else _normalize_front_to_geometry(front.geometry)
        )

    if isinstance(front, (Geometry, LazyGeometry)):
        return front

    if isinstance(front, Sequence) and front:
        fst = front[0]
        if isinstance(fst, StyledLayer) or (
            hasattr(fst, "geometry") and hasattr(fst, "color") and hasattr(fst, "thickness")
        ):
            geoms: list[Geometry] = []
            for layer in front:
                try:
                    g_obj = getattr(layer, "geometry", None)
                except Exception:
                    g_obj = None
                if g_obj is None:
                    continue
                if isinstance(g_obj, LazyGeometry):
                    try:
                        g_obj = g_obj.realize()
                    except Exception:
                        continue
                if isinstance(g_obj, Geometry):
                    if g_obj.is_empty:
                        continue
                    geoms.append(g_obj)
            if not geoms:
                return None
            base = geoms[0]
            for g in geoms[1:]:
                base = base.concat(g)
            return base

    return None


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
        geom = _normalize_front_to_geometry(front)
        if geom is None or geom.is_empty:
            if overlay is not None:
                overlay.show_message(
                    "G-code エクスポート対象なし（ジオメトリ未生成）", level="warn"
                )
            return
        coords, offsets = geom.as_arrays(copy=True)
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
