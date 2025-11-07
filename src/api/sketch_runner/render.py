"""
どこで: `api.sketch_runner.render`
何を: RenderWindow/ModernGL/LineRenderer の初期化と背景/線色の決定。
なぜ: `api.sketch` を薄くし、描画初期化の責務を分離するため。
"""

from __future__ import annotations

from typing import Any

import moderngl


def create_window_and_renderer(
    window_width: int,
    window_height: int,
    *,
    background: Any,
    line_color: Any,
    projection_matrix,
    swap_buffer,
    line_thickness: float,
):
    """ウィンドウ/ModernGL/LineRenderer を生成し、色を決定して返す。

    Returns
    -------
    (rendering_window, mgl_ctx, line_renderer, bg_rgba, line_rgba)
    """

    from engine.core.render_window import RenderWindow
    from engine.render.renderer import LineRenderer
    from util.color import normalize_color as _normalize_color

    try:
        from util.utils import load_config as _load_cfg_colors
    except Exception:  # pragma: no cover - フォールバック
        _load_cfg_colors = lambda: {}  # type: ignore[assignment]

    cfg_all = _load_cfg_colors() or {}
    canvas_cfg = cfg_all.get("canvas", {}) if isinstance(cfg_all, dict) else {}
    cfg_bg = canvas_cfg.get("background_color") if isinstance(canvas_cfg, dict) else None
    cfg_line = canvas_cfg.get("line_color") if isinstance(canvas_cfg, dict) else None

    # 背景色の決定
    if background is None:
        bg_src = cfg_bg if cfg_bg is not None else (1.0, 1.0, 1.0, 1.0)
    else:
        bg_src = background
    bg_rgba = _normalize_color(bg_src)
    rendering_window = RenderWindow(window_width, window_height, bg_color=bg_rgba)  # type: ignore[abstract]

    # ModernGL コンテキスト
    mgl_ctx: moderngl.Context = moderngl.create_context()
    mgl_ctx.enable(moderngl.BLEND)
    mgl_ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    # 線色の決定（設定→自動→指定）
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
    line_rgba = _normalize_color(lc_src)

    line_renderer = LineRenderer(
        mgl_context=mgl_ctx,
        projection_matrix=projection_matrix,
        swap_buffer=swap_buffer,
        line_thickness=line_thickness,
        line_color=line_rgba,
    )  # type: ignore

    return rendering_window, mgl_ctx, line_renderer, bg_rgba, line_rgba


__all__ = ["create_window_and_renderer"]
