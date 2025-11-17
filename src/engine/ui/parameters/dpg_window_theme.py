"""
どこで: `engine.ui.parameters` の Dear PyGui テーマ/フォント管理。
何を: パラメータウィンドウ全体とカテゴリ別のテーマ、フォント設定を一箇所で扱う。
なぜ: `dpg_window` 本体から見た目に関する責務を分離し、構造を単純化するため。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import dearpygui.dearpygui as dpg  # type: ignore

from .state import ParameterLayoutConfig, ParameterThemeConfig


class ParameterWindowThemeManager:
    """ParameterWindow のフォントとテーマを管理する。"""

    def __init__(
        self,
        *,
        layout: ParameterLayoutConfig,
        theme_cfg: ParameterThemeConfig | None,
    ) -> None:
        self._layout = layout
        self._theme_cfg = theme_cfg
        self._cat_header_theme: dict[str, Any] = {}
        self._cat_table_theme: dict[str, Any] = {}
        self._font_registry: Any | None = None

    def setup_fonts(self) -> None:
        """config.yaml に基づきフォントを登録し、既定フォントを Dear PyGui にバインドする。"""
        try:
            from util.utils import load_config as _load_cfg
        except Exception:
            return
        cfg = _load_cfg() or {}

        font_name: str | None = None
        try:
            pg = cfg.get("parameter_gui", {}) if isinstance(cfg, dict) else {}
            lcfg = pg.get("layout", {}) if isinstance(pg, dict) else {}
            fn = lcfg.get("font_name") if isinstance(lcfg, dict) else None
            if isinstance(fn, str) and fn.strip():
                font_name = fn.strip()
        except Exception:
            pass
        if not font_name:
            try:
                hud = cfg.get("hud", {}) if isinstance(cfg, dict) else {}
                fn = hud.get("font_name") if isinstance(hud, dict) else None
                if isinstance(fn, str) and fn.strip():
                    font_name = fn.strip()
            except Exception:
                pass

        fonts = cfg.get("fonts", {}) if isinstance(cfg, dict) else {}
        sdirs = fonts.get("search_dirs", []) if isinstance(fonts, dict) else []
        if isinstance(sdirs, (str, int, Path)):
            sdirs = [str(sdirs)]

        try:
            from util.fonts import glob_font_files, resolve_search_dirs  # type: ignore
        except Exception:
            glob_font_files = None  # type: ignore
            resolve_search_dirs = None  # type: ignore

        files: list[Path] = []
        if resolve_search_dirs is not None and glob_font_files is not None:
            try:
                files = glob_font_files(resolve_search_dirs(sdirs))
            except Exception:
                files = []
        if not files:
            return

        def _norm(s: str) -> str:
            return s.lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")

        req_norm = _norm(font_name) if isinstance(font_name, str) and font_name else None

        def _score(fp: Path) -> tuple[int, int, int, int, int]:
            name = fp.name.lower()
            ext = fp.suffix.lower()
            ext_rank = 0 if ext == ".ttf" else (1 if ext == ".otf" else 2)
            style_pen = 0
            for bad in (
                "italic",
                "oblique",
                "narrow",
                "condensed",
                "light",
                "thin",
                "black",
                "semibold",
            ):
                if bad in name:
                    style_pen += 2
            if "regular" in name:
                style_pen -= 1
            var_pen = 3 if ("variable" in name or "gx" in name) else 0
            req_pen = 0
            if req_norm:
                req_pen = 0 if req_norm in _norm(fp.stem) else 5
            len_pen = len(name)
            return (ext_rank, req_pen, style_pen, var_pen, len_pen)

        def _pref(candidates: list[Path]) -> list[Path]:
            ttf = [f for f in candidates if f.suffix.lower() == ".ttf"]
            otf = [f for f in candidates if f.suffix.lower() == ".otf"]
            return ttf + otf

        if req_norm:
            matched = [fp for fp in files if req_norm in _norm(fp.stem)]
            candidates = _pref(sorted(matched, key=_score))
        else:
            candidates = _pref(sorted(files, key=_score))

        try:
            bound = None
            with dpg.font_registry() as reg:
                self._font_registry = reg
                for fp in candidates[:20]:
                    if fp.suffix.lower() == ".ttc":
                        continue
                    try:
                        font = dpg.add_font(str(fp), int(self._layout.font_size), parent=reg)
                        bound = (fp, font)
                        break
                    except Exception:
                        continue
            if bound is not None:
                _file, default_font = bound
                dpg.bind_font(default_font)
        except Exception:
            return

    def setup_theme(self) -> None:
        """ウィンドウ全体のテーマ（スタイル/カラー）を構築して適用する。"""
        try:
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll):
                    if not self._apply_styles_from_config():
                        self._apply_default_styles()
                    self._apply_colors_from_config()
            dpg.bind_theme(theme)
        except Exception:
            return

    def _apply_default_styles(self) -> bool:
        pad = int(self._layout.padding)
        var_window_padding = getattr(dpg, "mvStyleVar_WindowPadding", None)
        var_frame_padding = getattr(dpg, "mvStyleVar_FramePadding", None)
        var_item_spacing = getattr(dpg, "mvStyleVar_ItemSpacing", None)
        if var_window_padding:
            dpg.add_theme_style(var_window_padding, pad, pad)
        if var_frame_padding:
            dpg.add_theme_style(var_frame_padding, pad, max(1, pad // 2))
        if var_item_spacing:
            dpg.add_theme_style(var_item_spacing, pad, max(1, pad // 2))
        return True

    def _apply_styles_from_config(self) -> bool:
        if self._theme_cfg is None or not isinstance(self._theme_cfg.style, dict):
            return False
        smap = {
            "window_padding": getattr(dpg, "mvStyleVar_WindowPadding", None),
            "frame_padding": getattr(dpg, "mvStyleVar_FramePadding", None),
            "item_spacing": getattr(dpg, "mvStyleVar_ItemSpacing", None),
            "frame_rounding": getattr(dpg, "mvStyleVar_FrameRounding", None),
            "grab_rounding": getattr(dpg, "mvStyleVar_GrabRounding", None),
            "grab_min_size": getattr(dpg, "mvStyleVar_GrabMinSize", None),
        }

        def _add_style_value(var: Any, value: float | int | Sequence[float | int]) -> None:
            if not var:
                return
            try:
                if isinstance(value, Sequence):
                    vals = list(value)
                    if len(vals) == 1:
                        dpg.add_theme_style(var, int(vals[0]))
                    elif len(vals) >= 2:
                        dpg.add_theme_style(var, int(vals[0]), int(vals[1]))
                    return
                dpg.add_theme_style(var, int(value))
            except Exception:
                return

        if smap["window_padding"]:
            dpg.add_theme_style(
                smap["window_padding"],
                int(getattr(self._layout, "window_padding_x", self._layout.padding)),
                int(getattr(self._layout, "window_padding_y", self._layout.padding)),
            )
        if smap["frame_padding"]:
            dpg.add_theme_style(
                smap["frame_padding"],
                int(getattr(self._layout, "frame_padding_x", self._layout.padding)),
                int(
                    getattr(
                        self._layout,
                        "frame_padding_y",
                        max(1, self._layout.padding // 2),
                    )
                ),
            )
        if smap["item_spacing"]:
            dpg.add_theme_style(
                smap["item_spacing"],
                int(getattr(self._layout, "item_spacing_x", self._layout.padding)),
                int(
                    getattr(
                        self._layout,
                        "item_spacing_y",
                        max(1, self._layout.padding // 2),
                    )
                ),
            )
        for key in ("frame_rounding", "grab_rounding", "grab_min_size"):
            var = smap.get(key)
            if var and key in self._theme_cfg.style:
                _add_style_value(var, self._theme_cfg.style[key])
        return True

    def _apply_colors_from_config(self) -> None:
        if self._theme_cfg is None or not isinstance(self._theme_cfg.colors, dict):
            return
        cmap = {
            "text": getattr(dpg, "mvThemeCol_Text", None),
            "window_bg": getattr(dpg, "mvThemeCol_WindowBg", None),
            "frame_bg": getattr(dpg, "mvThemeCol_FrameBg", None),
            "frame_bg_hovered": getattr(dpg, "mvThemeCol_FrameBgHovered", None),
            "frame_bg_active": getattr(dpg, "mvThemeCol_FrameBgActive", None),
            "header": getattr(dpg, "mvThemeCol_Header", None),
            "header_hovered": getattr(dpg, "mvThemeCol_HeaderHovered", None),
            "header_active": getattr(dpg, "mvThemeCol_HeaderActive", None),
            "accent": getattr(dpg, "mvThemeCol_SliderGrab", None),
            "accent_active": getattr(dpg, "mvThemeCol_SliderGrabActive", None),
        }
        for key, var in cmap.items():
            if not var or key not in self._theme_cfg.colors:
                continue
            col = self.to_dpg_color(self._theme_cfg.colors[key])
            if col is None:
                continue
            try:
                dpg.add_theme_color(var, col)
            except Exception:
                continue

    def get_category_header_theme(self, kind: str) -> Any | None:
        if kind in self._cat_header_theme:
            return self._cat_header_theme.get(kind)
        try:
            cfg = (
                self._theme_cfg.categories
                if isinstance(
                    self._theme_cfg,
                    ParameterThemeConfig,
                )
                else {}
            )
        except Exception:
            cfg = {}
        cat = cfg.get(kind) if isinstance(cfg, dict) else None
        if not isinstance(cat, dict):
            self._cat_header_theme[kind] = None
            return None
        keys = ["header", "header_hovered", "header_active"]
        if not any(k in cat for k in keys):
            self._cat_header_theme[kind] = None
            return None
        try:
            with dpg.theme() as th:
                comp_target = getattr(dpg, "mvCollapsingHeader", None) or dpg.mvAll
                with dpg.theme_component(comp_target):
                    cmap = {
                        "header": getattr(dpg, "mvThemeCol_Header", None),
                        "header_hovered": getattr(dpg, "mvThemeCol_HeaderHovered", None),
                        "header_active": getattr(dpg, "mvThemeCol_HeaderActive", None),
                    }
                    for key, var in cmap.items():
                        if not var or key not in cat:
                            continue
                        col = self.to_dpg_color(cat[key])
                        if col is None:
                            continue
                        try:
                            dpg.add_theme_color(var, col)
                        except Exception:
                            continue
            self._cat_header_theme[kind] = th
            return th
        except Exception:
            self._cat_header_theme[kind] = None
            return None

    def get_category_table_theme(self, kind: str) -> Any | None:
        if kind in self._cat_table_theme:
            return self._cat_table_theme.get(kind)
        try:
            cfg = (
                self._theme_cfg.categories
                if isinstance(
                    self._theme_cfg,
                    ParameterThemeConfig,
                )
                else {}
            )
        except Exception:
            cfg = {}
        cat = cfg.get(kind) if isinstance(cfg, dict) else None
        if not isinstance(cat, dict):
            self._cat_table_theme[kind] = None
            return None
        row_keys = ["table_row_bg", "table_row_bg_alt"]
        if not any(k in cat for k in row_keys):
            self._cat_table_theme[kind] = None
            return None
        try:
            with dpg.theme() as th:
                comp_target = getattr(dpg, "mvTable", None) or dpg.mvAll
                with dpg.theme_component(comp_target):
                    tmap = {
                        "table_row_bg": getattr(dpg, "mvThemeCol_TableRowBg", None),
                        "table_row_bg_alt": getattr(dpg, "mvThemeCol_TableRowBgAlt", None),
                    }
                    for key, var in tmap.items():
                        if not var or key not in cat:
                            continue
                        col = self.to_dpg_color(cat[key])
                        if col is None:
                            continue
                        try:
                            dpg.add_theme_color(var, col)
                        except Exception:
                            continue
            self._cat_table_theme[kind] = th
            return th
        except Exception:
            self._cat_table_theme[kind] = None
            return None

    def to_dpg_color(self, value: Any) -> Any:
        try:
            from util.color import to_u8_rgba as _to_u8_rgba

            r, g, b, a = _to_u8_rgba(value)
            return [int(r), int(g), int(b), int(a)]
        except Exception:
            try:
                if isinstance(value, (list, tuple)) and len(value) >= 4:
                    vals = [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
                    if all(0.0 <= v <= 1.0 for v in vals):
                        return [int(round(v * 255)) for v in vals]
                    return [int(round(v)) for v in vals]
            except Exception:
                return None
            return None
