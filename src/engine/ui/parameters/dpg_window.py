"""
どこで: `engine.ui.parameters` の Dear PyGui 実装（簡潔版）。
何を: ParameterStore の Descriptor から最小限のウィンドウを生成し、表示/非表示/終了を扱う。
なぜ: 実装を単純化しつつ、テストと基本動作（生成/表示切替/終了）を満たすため。

備考:
- Dear PyGui は実利用時に必須。未導入環境ではこのモジュール import 自体が失敗する。
- pyglet との連携（フレーム駆動）は行わず、必要なときだけ `show/hide` を切り替える。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from threading import Thread
from typing import Any, Iterable, Sequence, cast

import dearpygui.dearpygui as dpg  # type: ignore

from .state import ParameterDescriptor, ParameterLayoutConfig, ParameterStore, ParameterThemeConfig

# タグ定数
ROOT_TAG = "__pxd_param_root__"
SCROLL_TAG = "__pxd_param_scroll__"
STAGE_TAG = "__pxd_param_stage__"

logger = logging.getLogger("engine.ui.parameters.dpg")


class ParameterWindowBase:
    """パラメータウィンドウの最小インタフェース。"""

    def set_visible(self, _visible: bool) -> None:  # pragma: no cover
        return None

    def close(self) -> None:  # pragma: no cover
        return None

    def mount(self, _descriptors: list[ParameterDescriptor]) -> None:  # pragma: no cover
        return None


class ParameterWindow(ParameterWindowBase):  # type: ignore[override]
    """Dear PyGui による最小ウィンドウ実装。"""

    def __init__(
        self,
        *,
        store: ParameterStore,
        layout: ParameterLayoutConfig,
        width: int = 420,
        height: int = 640,
        title: str = "Parameters",
        theme: ParameterThemeConfig | None = None,
        auto_show: bool = True,
    ) -> None:
        self._store = store
        self._layout = layout
        self._width = width
        self._height = height
        self._title = title
        self._theme = theme
        self._visible = False
        self._driver: Any | None = None
        self._syncing: bool = False

        dpg.create_context()
        self._viewport = dpg.create_viewport(
            title=self._title, width=self._width, height=self._height
        )
        dpg.setup_dearpygui()

        # フォント登録（設定に基づき、fonts.search_dirs から優先的に探す）
        try:
            self._setup_fonts()
        except Exception:
            # フェイルソフト: フォントは既定のまま
            logger.debug("DPG font setup skipped due to error", exc_info=True)

        # ルート構築とテーマ適用
        self._build_root_window()
        self._setup_theme()

        # 初期マウントと購読
        self.mount(sorted(self._store.descriptors(), key=lambda d: d.id))

        def _on_store_change_wrapper(ids: Iterable[str]) -> None:
            try:
                self._on_store_change(ids)
                self.sync_display_from_store(self._store)
            except Exception:
                logger.exception("store change handling failed")

        self._store.subscribe(_on_store_change_wrapper)

        # 表示 + ドライバ
        if auto_show:
            dpg.show_viewport()
            self._visible = True
            self._start_driver()

    def set_visible(self, visible: bool) -> None:
        if visible and not self._visible:
            dpg.show_viewport()
            self._visible = True
            self._start_driver()
        elif not visible and self._visible:
            dpg.hide_viewport()
            self._visible = False
            self._stop_driver()

    def close(self) -> None:
        self._stop_driver()
        dpg.destroy_context()

    def mount(self, descriptors: list[ParameterDescriptor]) -> None:
        # ルート直下にセクション（Shapes/Effects）を配置し、Display/HUD と同レベルにする
        with dpg.stage(tag=STAGE_TAG):
            self._build_grouped_table(ROOT_TAG, descriptors)
        dpg.unstage(STAGE_TAG)

    # ---- フォント ----
    def _setup_fonts(self) -> None:
        try:
            from util.utils import _find_project_root as _root
            from util.utils import load_config as _load_cfg
        except Exception:
            return
        cfg = _load_cfg() or {}
        # フォント名の決定: parameter_gui.layout.font_name > hud.font_name > status_manager.font
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
        if not font_name:
            try:
                sm = cfg.get("status_manager", {}) if isinstance(cfg, dict) else {}
                fn = sm.get("font") if isinstance(sm, dict) else None
                if isinstance(fn, str) and fn.strip():
                    font_name = fn.strip()
            except Exception:
                pass

        # 検索ディレクトリ
        fonts = cfg.get("fonts", {}) if isinstance(cfg, dict) else {}
        sdirs = fonts.get("search_dirs", []) if isinstance(fonts, dict) else []
        if isinstance(sdirs, (str, int, Path)):
            sdirs = [str(sdirs)]
        root = _root(Path(__file__).parent)
        exts = (".ttf", ".otf", ".ttc")

        files: list[Path] = []
        for s in sdirs:
            try:
                p = Path(os.path.expandvars(os.path.expanduser(str(s))))
                if not p.is_absolute():
                    p = (root / p).resolve()
                if not p.exists() or not p.is_dir():
                    continue
                for ext in exts:
                    files.extend(p.glob(f"**/*{ext}"))
            except Exception:
                continue
        if not files:
            return

        chosen: Path | None = None
        if font_name:
            for fp in files:
                if font_name.lower() in fp.name.lower():
                    chosen = fp
                    break
        if chosen is None:
            chosen = files[0]

        # フォント登録（`.ttc` など失敗時は握りつぶす）
        try:
            self._font_registry = dpg.add_font_registry()
            try:
                default_font = dpg.add_font(str(chosen), int(self._layout.font_size))
            except Exception:
                # `.ttc` で失敗する可能性 → `.ttf/.otf` のみ再トライ
                default_font = None
                if chosen.suffix.lower() == ".ttc":
                    alt = next((f for f in files if f.suffix.lower() in (".ttf", ".otf")), None)
                    if alt is not None:
                        try:
                            default_font = dpg.add_font(str(alt), int(self._layout.font_size))
                        except Exception:
                            default_font = None
            if default_font is not None:
                dpg.bind_font(default_font)
        except Exception:
            # 失敗しても GUI 自体は続行
            pass

    # ---- ルート/Display/HUD ----
    def _build_root_window(self) -> None:
        with dpg.window(tag=ROOT_TAG, label=self._title, no_resize=False, no_collapse=True) as root:
            try:
                self.build_display_controls(parent=root, store=self._store)
            except Exception:
                logger.warning("failed to build runner controls; continue without Display/HUD")
            # ここでは子ウィンドウを使わず、Shapes/Effects も ROOT 直下に並べる
        dpg.set_primary_window(root, True)

    def force_set_rgb_u8(self, tag: int | str, rgb_u8: Sequence[int]) -> None:
        if not isinstance(rgb_u8, Sequence) or len(rgb_u8) < 3:
            raise ValueError("rgb_u8 must be a sequence of length >= 3")
        r = int(rgb_u8[0])
        g = int(rgb_u8[1])
        b = int(rgb_u8[2])
        dpg.set_value(tag, [r, g, b])

    def store_rgb01(self, pid: str, app_data: Any) -> None:
        try:
            from util.color import normalize_color as _norm

            rgba = _norm(app_data)
        except Exception:
            logger.exception("store_rgb01 failed to normalize: pid=%s val=%s", pid, app_data)
            return
        self._store.set_override(pid, rgba)

    # ---- ヘルパ ----
    def _safe_norm(
        self, value: Any, default: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        try:
            from util.color import normalize_color as _norm

            r, g, b, a = _norm(value)
            return float(r), float(g), float(b), float(a)
        except Exception:
            return default

    def _dpg_policy(self, names: Sequence[str]) -> Any | None:
        for n in names:
            var = getattr(dpg, n, None)
            if var is not None:
                return var
        return None

    def build_display_controls(self, parent: int | str, store: ParameterStore) -> None:
        try:
            from util.utils import load_config as _load_cfg
        except Exception:
            _load_cfg = lambda: {}

        cfg = _load_cfg() or {}
        canvas = cfg.get("canvas", {}) if isinstance(cfg, dict) else {}
        bg_raw = canvas.get("background_color", (1.0, 1.0, 1.0, 1.0))
        ln_raw = canvas.get("line_color", (0.0, 0.0, 0.0, 1.0))
        bgf = self._safe_norm(bg_raw, (1.0, 1.0, 1.0, 1.0))
        lnf = self._safe_norm(ln_raw, (0.0, 0.0, 0.0, 1.0))

        val = store.current_value("runner.background") or store.original_value("runner.background")
        if val is not None:
            bgf = self._safe_norm(val, bgf)
        val = store.current_value("runner.line_color") or store.original_value("runner.line_color")
        if val is not None:
            lnf = self._safe_norm(val, lnf)

        with dpg.collapsing_header(label="Display", default_open=True, parent=parent):
            table_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=table_policy):
                left, right = self._label_value_ratio()
                self._add_two_columns(left, right)
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Background")
                    with dpg.table_cell():
                        bg_picker = dpg.add_color_edit(
                            tag="runner.background",
                            default_value=[
                                int(round(bgf[0] * 255)),
                                int(round(bgf[1] * 255)),
                                int(round(bgf[2] * 255)),
                            ],
                            no_label=True,
                            no_alpha=True,
                            alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
                            display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                            display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                            input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                            alpha_bar=False,
                        )
                    # HUD と揃えるため、幅は既定値のままにする（set_item_width は適用しない）
                    r, g, b = float(bgf[0]), float(bgf[1]), float(bgf[2])
                    self.force_set_rgb_u8(
                        bg_picker,
                        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
                    )
                    dpg.configure_item(
                        bg_picker, callback=lambda s, a, u: self.store_rgb01("runner.background", a)
                    )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Line Color")
                    with dpg.table_cell():
                        ln_picker = dpg.add_color_edit(
                            tag="runner.line_color",
                            default_value=[
                                int(round(lnf[0] * 255)),
                                int(round(lnf[1] * 255)),
                                int(round(lnf[2] * 255)),
                            ],
                            no_label=True,
                            no_alpha=True,
                            alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
                            display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                            display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                            input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                            alpha_bar=False,
                        )
                    # HUD と揃えるため、幅は既定値のままにする（set_item_width は適用しない）
                    r, g, b = float(lnf[0]), float(lnf[1]), float(lnf[2])
                    self.force_set_rgb_u8(
                        ln_picker,
                        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
                    )
                    dpg.configure_item(
                        ln_picker, callback=lambda s, a, u: self.store_rgb01("runner.line_color", a)
                    )

        with dpg.collapsing_header(label="HUD", default_open=True, parent=parent):
            hud_tbl_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=hud_tbl_policy):
                left, right = self._label_value_ratio()
                self._add_two_columns(left, right)
                # Text Color
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Text Color")
                    with dpg.table_cell():
                        tx_val = store.current_value(
                            "runner.hud_text_color"
                        ) or store.original_value("runner.hud_text_color")
                        tr, tg, tb, _ = (
                            self._safe_norm(tx_val, (0.0, 0.0, 0.0, 1.0))
                            if tx_val is not None
                            else (
                                0.0,
                                0.0,
                                0.0,
                                1.0,
                            )
                        )
                        tx_picker = dpg.add_color_edit(
                            tag="runner.hud_text_color",
                            default_value=[
                                int(round(tr * 255)),
                                int(round(tg * 255)),
                                int(round(tb * 255)),
                            ],
                            no_label=True,
                            no_alpha=True,
                            alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
                            display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                            display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                            input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                            alpha_bar=False,
                        )
                        dpg.configure_item(
                            tx_picker,
                            callback=lambda s, a, u: self.store_rgb01("runner.hud_text_color", a),
                        )
                # Meter Color
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Meter Color")
                    with dpg.table_cell():
                        mt_val = store.current_value(
                            "runner.hud_meter_color"
                        ) or store.original_value("runner.hud_meter_color")
                        mr, mg, mb, _ = (
                            self._safe_norm(mt_val, (0.2, 0.2, 0.2, 1.0))
                            if mt_val is not None
                            else (
                                0.2,
                                0.2,
                                0.2,
                                1.0,
                            )
                        )
                        mt_picker = dpg.add_color_edit(
                            tag="runner.hud_meter_color",
                            default_value=[
                                int(round(mr * 255)),
                                int(round(mg * 255)),
                                int(round(mb * 255)),
                            ],
                            no_label=True,
                            no_alpha=True,
                            alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
                            display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                            display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                            input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                            alpha_bar=False,
                        )
                        dpg.configure_item(
                            mt_picker,
                            callback=lambda s, a, u: self.store_rgb01("runner.hud_meter_color", a),
                        )
                # Meter BG
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Meter BG")
                    with dpg.table_cell():
                        mb_val = store.current_value(
                            "runner.hud_meter_bg_color"
                        ) or store.original_value("runner.hud_meter_bg_color")
                        br, bg, bb, _ = (
                            self._safe_norm(mb_val, (0.196, 0.196, 0.196, 1.0))
                            if mb_val is not None
                            else (
                                0.196,
                                0.196,
                                0.196,
                                1.0,
                            )
                        )
                        mb_picker = dpg.add_color_edit(
                            tag="runner.hud_meter_bg_color",
                            default_value=[
                                int(round(br * 255)),
                                int(round(bg * 255)),
                                int(round(bb * 255)),
                            ],
                            no_label=True,
                            no_alpha=True,
                            alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
                            display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                            display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                            input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                            alpha_bar=False,
                        )
                        dpg.configure_item(
                            mb_picker,
                            callback=lambda s, a, u: self.store_rgb01(
                                "runner.hud_meter_bg_color", a
                            ),
                        )

        self.sync_display_from_store(store)

    def sync_display_from_store(self, store: ParameterStore) -> None:
        ids = [
            "runner.background",
            "runner.line_color",
            "runner.hud_text_color",
            "runner.hud_meter_color",
            "runner.hud_meter_bg_color",
        ]
        self._on_store_change(ids)

    # ---- テーブル/行 ----
    def _build_grouped_table(
        self, parent: int | str, descriptors: list[ParameterDescriptor]
    ) -> None:
        excluded = {
            "runner.background",
            "runner.line_color",
            "runner.hud_text_color",
            "runner.hud_meter_color",
            "runner.hud_meter_bg_color",
        }
        filtered = [d for d in descriptors if d.id not in excluded]
        sorted_desc = sorted(filtered, key=lambda d: (d.category, d.id))
        current_cat: str | None = None
        group_items: list[ParameterDescriptor] = []

        def flush(cat: str | None, items: list[ParameterDescriptor]) -> None:
            if not items or not any(it.supported for it in items):
                return
            label = cat if cat else "General"
            with dpg.collapsing_header(label=label, parent=parent, default_open=True):
                table_policy = self._dpg_policy(
                    ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
                )
                with dpg.table(header_row=False, policy=table_policy) as table:
                    var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                    if var_cell_padding is not None:
                        cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                        cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                        with dpg.theme() as _outer_tbl_theme:
                            with dpg.theme_component(dpg.mvAll):
                                dpg.add_theme_style(var_cell_padding, cx, cy)
                        dpg.bind_item_theme(table, _outer_tbl_theme)
                    # 3 列（Label | Bars | CC）
                    label_ratio = float(self._layout.label_column_ratio)
                    label_ratio = (
                        0.1 if label_ratio < 0.1 else (0.9 if label_ratio > 0.9 else label_ratio)
                    )
                    # 残りを Bars:CC = bars_cc_ratio : (1 - bars_cc_ratio) に配分
                    rest = max(0.1, 1.0 - label_ratio)
                    bcc = float(getattr(self._layout, "bars_cc_ratio", 0.7))
                    bcc = 0.05 if bcc < 0.05 else (0.95 if bcc > 0.95 else bcc)
                    bars_ratio = rest * bcc
                    cc_ratio = rest - bars_ratio
                    try:
                        dpg.add_table_column(
                            label="Parameter", width_stretch=True, init_width_or_weight=label_ratio
                        )
                        dpg.add_table_column(
                            label="Bars", width_stretch=True, init_width_or_weight=bars_ratio
                        )
                        dpg.add_table_column(
                            label="CC", width_stretch=True, init_width_or_weight=cc_ratio
                        )
                    except TypeError:
                        dpg.add_table_column(label="Parameter")
                        dpg.add_table_column(label="Bars")
                        dpg.add_table_column(label="CC")
                    for it in items:
                        if not it.supported:
                            continue
                        self._create_row_3cols(table, it)

        for desc in sorted_desc:
            if current_cat is None:
                current_cat = desc.category
                group_items = [desc]
                continue
            if desc.category != current_cat:
                flush(current_cat, group_items)
                current_cat = desc.category
                group_items = [desc]
            else:
                group_items.append(desc)
        flush(current_cat, group_items)

    def _label_value_ratio(self) -> tuple[float, float]:
        left = 0.5
        left = float(self._layout.label_column_ratio)
        left = 0.1 if left < 0.1 else (0.9 if left > 0.9 else left)
        right = max(0.1, 1.0 - left)
        return left, right

    def _add_two_columns(self, left: float, right: float) -> None:
        try:
            dpg.add_table_column(label="Parameter", width_stretch=True, init_width_or_weight=left)
            dpg.add_table_column(label="Value", width_stretch=True, init_width_or_weight=right)
        except TypeError:
            dpg.add_table_column(label="Parameter")
            dpg.add_table_column(label="Value")

    def _create_row_3cols(self, table: int | str, desc: ParameterDescriptor) -> None:
        with dpg.table_row(parent=table):
            # 1) Label（セル上端揃え）
            with dpg.table_cell():
                dpg.add_text(default_value=desc.label or desc.id)
            # 2) Bars
            with dpg.table_cell():
                self._create_bars(parent=dpg.last_item() or table, desc=desc)
            # 3) CC (numeric only)
            with dpg.table_cell():
                self._create_cc_inputs(parent=dpg.last_item() or table, desc=desc)

    def _create_bars(self, parent: int | str, desc: ParameterDescriptor) -> None:
        vt = desc.value_type
        value = self._current_or_default(desc)
        if vt == "vector":
            vec = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0, 0.0]
            dim = 4 if len(vec) >= 4 else 3
            vh = desc.vector_hint
            if vh is None:
                default_vh = self._layout.derive_vector_range(dim=dim)
                vmin = list(default_vh.min_values)
                vmax = list(default_vh.max_values)
            else:
                vmin = list(vh.min_values)
                vmax = list(vh.max_values)
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
            ) as bars_tbl:
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as _bars_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                    dpg.bind_item_theme(bars_tbl, _bars_theme)
                for _ in range(dim):
                    try:
                        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                    except TypeError:
                        dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    for i, suffix in enumerate(("x", "y", "z", "w")[:dim]):
                        with dpg.table_cell():
                            tag = f"{desc.id}::{suffix}"
                            default_component = float(vec[i]) if i < len(vec) else 0.0
                            slider_id = dpg.add_slider_float(
                                tag=tag,
                                label="",
                                default_value=default_component,
                                min_value=float(vmin[i]) if i < len(vmin) else 0.0,
                                max_value=float(vmax[i]) if i < len(vmax) else 1.0,
                                format=f"%.{self._layout.value_precision}f",
                                callback=self._on_widget_change,
                                user_data=(desc.id, i),
                            )
                            dpg.set_item_width(slider_id, -1)
            return
        # scalar/bool/enum/string → 内側1列テーブルで幅扱いを vector と揃える
        hint = desc.range_hint or self._layout.derive_range(
            name=desc.id, value_type=desc.value_type, default_value=desc.default_value
        )
        if vt in {"int", "float"}:
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
            ) as bars_tbl:
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as _bars_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                    dpg.bind_item_theme(bars_tbl, _bars_theme)
                try:
                    dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                except TypeError:
                    dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    with dpg.table_cell():
                        if vt == "int":
                            slider_id = dpg.add_slider_int(
                                tag=desc.id,
                                label="",
                                default_value=int(value) if value is not None else 0,
                                min_value=int(hint.min_value),
                                max_value=int(hint.max_value),
                                callback=self._on_widget_change,
                                user_data=desc.id,
                            )
                        else:
                            slider_id = dpg.add_slider_float(
                                tag=desc.id,
                                label="",
                                default_value=float(value) if value is not None else 0.0,
                                min_value=float(hint.min_value),
                                max_value=float(hint.max_value),
                                format=f"%.{self._layout.value_precision}f",
                                callback=self._on_widget_change,
                                user_data=desc.id,
                            )
                        dpg.set_item_width(slider_id, -1)
            return
        if vt == "bool":
            dpg.add_checkbox(
                tag=desc.id,
                label="",
                default_value=bool(value),
                callback=self._on_widget_change,
                user_data=desc.id,
            )
            return
        if vt == "enum":
            items = list(desc.choices or [])
            default = str(value) if value is not None else (items[0] if items else "")
            if len(items) <= 5:
                dpg.add_radio_button(
                    tag=desc.id,
                    items=items,
                    default_value=default,
                    horizontal=True,
                    callback=self._on_widget_change,
                    user_data=desc.id,
                )
            else:
                dpg.add_combo(
                    tag=desc.id,
                    items=items,
                    default_value=default,
                    callback=self._on_widget_change,
                    user_data=desc.id,
                )
            return
        if vt == "string":
            self._create_string(parent=parent, desc=desc, value=value)
            return

    def _create_cc_inputs(self, parent: int | str, desc: ParameterDescriptor) -> None:
        vt = desc.value_type
        if vt == "vector":
            vec = (
                desc.default_value
                if isinstance(desc.default_value, (list, tuple))
                else (0.0, 0.0, 0.0)
            )
            dim = 4 if len(vec) >= 4 else 3
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
            ) as cc_tbl:
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as _cc_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                dpg.bind_item_theme(cc_tbl, _cc_theme)
                for _ in range(dim):
                    try:
                        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                    except TypeError:
                        dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    for i in range(dim):
                        with dpg.table_cell():
                            self._add_cc_binding_input_component(None, desc, i)
            return
        if vt in {"float", "int"}:
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
            ) as cc_tbl:
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as _cc_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                dpg.bind_item_theme(cc_tbl, _cc_theme)
                try:
                    dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                except TypeError:
                    dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    with dpg.table_cell():
                        self._add_cc_binding_input(None, desc)
            return
        # 非数値は CC 入力なし

    # ---- 型別ウィジェット ----
    def _create_widget(self, parent: int | str, desc: ParameterDescriptor) -> int:
        value = self._current_or_default(desc)
        vt = desc.value_type
        if vt == "bool":
            return self._create_bool(parent, desc, bool(value))
        if vt == "enum":
            return self._create_enum(parent, desc, value)
        if vt == "string":
            return self._create_string(parent, desc, value)
        if vt == "vector":
            return self._create_vector(parent, desc, value)
        hint = desc.range_hint or self._layout.derive_range(
            name=desc.id, value_type=desc.value_type, default_value=desc.default_value
        )
        if vt == "int":
            return self._create_int(parent, desc, value, hint)
        return self._create_float(parent, desc, value, hint)

    def _create_bool(self, parent: int | str, desc: ParameterDescriptor, value: Any) -> int:
        return dpg.add_checkbox(
            tag=desc.id,
            parent=parent,
            label="",
            default_value=bool(value),
            callback=self._on_widget_change,
            user_data=desc.id,
        )

    def _create_enum(self, parent: int | str, desc: ParameterDescriptor, value: Any) -> int:
        items = list(desc.choices or [])
        default = str(value) if value is not None else (items[0] if items else "")
        if len(items) <= 5:
            return dpg.add_radio_button(
                tag=desc.id,
                items=items,
                parent=parent,
                default_value=default,
                horizontal=True,
                callback=self._on_widget_change,
                user_data=desc.id,
            )
        return dpg.add_combo(
            tag=desc.id,
            items=items,
            parent=parent,
            default_value=default,
            callback=self._on_widget_change,
            user_data=desc.id,
        )

    def _create_string(self, parent: int | str, desc: ParameterDescriptor, value: Any) -> int:
        ml = bool(getattr(desc, "string_multiline", False))
        h = getattr(desc, "string_height", None)
        _kwargs: dict[str, Any] = {
            "tag": desc.id,
            "parent": parent,
            "label": "",
            "default_value": str(value) if value is not None else "",
            "callback": self._on_widget_change,
            "user_data": desc.id,
        }
        if ml:
            _kwargs["multiline"] = True
            if isinstance(h, int) and h > 0:
                _kwargs["height"] = int(h)
        txt_id = dpg.add_input_text(**_kwargs)
        dpg.set_item_width(txt_id, -1)
        return txt_id

    def _create_vector(self, parent: int | str, desc: ParameterDescriptor, value: Any) -> int:
        vec = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0, 0.0]
        dim = 4 if len(vec) >= 4 else 3
        vh = desc.vector_hint
        if vh is None:
            default_vh = self._layout.derive_vector_range(dim=dim)
            vmin = list(default_vh.min_values)
            vmax = list(default_vh.max_values)
        else:
            vmin = list(vh.min_values)
            vmax = list(vh.max_values)
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
        ) as vec_table:
            var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
            if var_cell_padding is not None:
                pad = int(self._layout.padding)
                with dpg.theme() as _vec_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(var_cell_padding, pad, pad)
                dpg.bind_item_theme(vec_table, _vec_theme)
            # レイアウト: [バー群] | [CC群]
            try:
                # 左（バー群）を広く、右（CC群）をコンパクトに
                dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                dpg.add_table_column(width_stretch=True, init_width_or_weight=0.25)
            except TypeError:
                dpg.add_table_column(width_stretch=True)
                dpg.add_table_column(width_stretch=True)
            with dpg.table_row():
                # 1) バー群（テーブルで3/4列に分割し、均等に広げる）
                with dpg.table_cell():
                    with dpg.table(
                        header_row=False,
                        policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
                    ) as bars_tbl:
                        var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                        if var_cell_padding is not None:
                            pad = int(self._layout.padding)
                            with dpg.theme() as _bars_theme:
                                with dpg.theme_component(dpg.mvAll):
                                    dpg.add_theme_style(var_cell_padding, pad, pad)
                            dpg.bind_item_theme(bars_tbl, _bars_theme)
                        for _ in range(dim):
                            try:
                                dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                            except TypeError:
                                dpg.add_table_column(width_stretch=True)
                        with dpg.table_row():
                            for i, suffix in enumerate(("x", "y", "z", "w")[:dim]):
                                with dpg.table_cell():
                                    tag = f"{desc.id}::{suffix}"
                                    default_component = float(vec[i]) if i < len(vec) else 0.0
                                    slider_id = dpg.add_slider_float(
                                        tag=tag,
                                        label="",
                                        default_value=default_component,
                                        min_value=float(vmin[i]) if i < len(vmin) else 0.0,
                                        max_value=float(vmax[i]) if i < len(vmax) else 1.0,
                                        format=f"%.{self._layout.value_precision}f",
                                        callback=self._on_widget_change,
                                        user_data=(desc.id, i),
                                    )
                                    dpg.set_item_width(slider_id, -1)
                # 2) CC 群（水平グループ + item_spacing を個別設定）
                with dpg.table_cell():
                    # CC群にのみ item spacing を適用（vector_cc_spacing）
                    cc_group = dpg.add_group(horizontal=True)
                    try:
                        var_item_spacing = getattr(dpg, "mvStyleVar_ItemSpacing", None)
                        if var_item_spacing is not None:
                            with dpg.theme() as _cc_theme:
                                with dpg.theme_component(dpg.mvAll):
                                    s = int(getattr(self._layout, "vector_cc_spacing", 2))
                                    dpg.add_theme_style(var_item_spacing, s, s)
                            dpg.bind_item_theme(cc_group, _cc_theme)
                    except Exception:
                        pass
                    with dpg.group(horizontal=True, parent=cc_group):
                        for i in range(dim):
                            self._add_cc_binding_input_component(None, desc, i)
        return vec_table

    def _create_int(
        self, parent: int | str, desc: ParameterDescriptor, value: Any, hint: Any
    ) -> int:
        # スライダー（左）と CC 入力（右）をテーブルで配置し、右側の確保を保証する
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
        ) as tbl:
            var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
            if var_cell_padding is not None:
                pad = int(self._layout.padding)
                with dpg.theme() as _int_tbl_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(var_cell_padding, pad, pad)
                dpg.bind_item_theme(tbl, _int_tbl_theme)
            dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
            dpg.add_table_column(width_stretch=True, init_width_or_weight=0.35)
            with dpg.table_row():
                with dpg.table_cell():
                    slider_id = dpg.add_slider_int(
                        tag=desc.id,
                        label="",
                        default_value=int(value) if value is not None else 0,
                        min_value=int(hint.min_value),
                        max_value=int(hint.max_value),
                        callback=self._on_widget_change,
                        user_data=desc.id,
                    )
                    dpg.set_item_width(slider_id, -1)
                with dpg.table_cell():
                    self._add_cc_binding_input(None, desc)
        return slider_id

    def _create_float(
        self, parent: int | str, desc: ParameterDescriptor, value: Any, hint: Any
    ) -> int:
        # スライダー（左）と CC 入力（右）をテーブルで配置し、右側の確保を保証する
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
        ) as tbl:
            var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
            if var_cell_padding is not None:
                pad = int(self._layout.padding)
                with dpg.theme() as _flt_tbl_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(var_cell_padding, pad, pad)
                dpg.bind_item_theme(tbl, _flt_tbl_theme)
            dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
            dpg.add_table_column(width_stretch=True, init_width_or_weight=0.35)
            with dpg.table_row():
                with dpg.table_cell():
                    slider_id = dpg.add_slider_float(
                        tag=desc.id,
                        label="",
                        default_value=float(value) if value is not None else 0.0,
                        min_value=float(hint.min_value),
                        max_value=float(hint.max_value),
                        format=f"%.{self._layout.value_precision}f",
                        callback=self._on_widget_change,
                        user_data=desc.id,
                    )
                    dpg.set_item_width(slider_id, -1)
                with dpg.table_cell():
                    self._add_cc_binding_input(None, desc)
        return slider_id

    # ---- 値連携 ----
    def _current_or_default(self, desc: ParameterDescriptor) -> Any:
        v = self._store.current_value(desc.id)
        return v if v is not None else desc.default_value

    def _on_widget_change(self, sender: int, app_data: Any, user_data: Any) -> None:  # noqa: D401
        if self._syncing:
            return
        if isinstance(user_data, (list, tuple)) and len(user_data) == 2:
            parent_id = str(user_data[0])
            idx = int(user_data[1])
            try:
                current = self._store.current_value(parent_id)
                if not isinstance(current, (list, tuple)):
                    current = self._store.original_value(parent_id)
                vec = list(current) if isinstance(current, (list, tuple)) else [0.0, 0.0, 0.0]
                dim = 4 if len(vec) >= 4 else 3
                base = (vec + [0.0] * dim)[:dim]
                base[idx] = float(app_data)
                self._store.set_override(parent_id, tuple(base))
                return
            except (TypeError, ValueError):
                logger.exception("vector override failed: id=%s idx=%s", parent_id, idx)
                return
        pid = str(user_data)
        value = app_data
        self._store.set_override(pid, value)

    def _on_cc_binding_change(self, sender: int, app_data: Any, user_data: Any) -> None:
        """CC 番号入力の変更を Store へ反映（空で解除）。"""
        try:
            pid = str(user_data)
            text = str(app_data).strip()
        except Exception:
            return
        if not text:
            self._store.bind_cc(pid, None)
            return
        try:
            i = int(float(text))  # 入力が浮動小数でも整数化
        except Exception:
            # 無効入力は解除扱い
            self._store.bind_cc(pid, None)
            dpg.set_value(f"{pid}::cc", "")
            return
        # 軽いクランプ 0..127
        if i < 0:
            i = 0
        if i > 127:
            i = 127
        self._store.bind_cc(pid, i)
        # 正規化表示
        try:
            dpg.set_value(f"{pid}::cc", str(i))
        except Exception:
            pass

    def _add_cc_binding_input(self, parent: int | str | None, desc: ParameterDescriptor):
        """スライダー右に CC 番号入力を追加する（最小 UI）。"""
        try:
            current = self._store.cc_binding(desc.id)
        except Exception:
            current = None
        default_text = "" if current is None else str(int(current))
        kwargs: dict[str, Any] = {
            "tag": f"{desc.id}::cc",
            "default_value": default_text,
            "hint": "cc",
            "no_spaces": True,
            "callback": self._on_cc_binding_change,
            "user_data": desc.id,
        }
        if parent is not None:
            kwargs["parent"] = parent
        kwargs["height"] = int(getattr(self._layout, "row_height", 28))
        box = dpg.add_input_text(**kwargs)
        try:
            # 案A: CC 入力はセル幅いっぱいに伸ばす（固定幅は使わない）
            dpg.set_item_width(box, -1)
        except Exception:
            pass
        return box

    def _add_cc_binding_input_component(
        self, parent: int | str | None, desc: ParameterDescriptor, idx: int
    ):
        suffix = ("x", "y", "z", "w")[idx]
        pid_comp = f"{desc.id}::{suffix}"
        try:
            current = self._store.cc_binding(pid_comp)
        except Exception:
            current = None
        default_text = "" if current is None else str(int(current))
        kwargs: dict[str, Any] = {
            "tag": f"{pid_comp}::cc",
            "default_value": default_text,
            "hint": "cc",
            "no_spaces": True,
            "callback": self._on_cc_binding_change,
            "user_data": pid_comp,
        }
        if parent is not None:
            kwargs["parent"] = parent
        kwargs["height"] = int(getattr(self._layout, "row_height", 28))
        box = dpg.add_input_text(**kwargs)
        try:
            # 案A: CC 入力はセル幅いっぱいに伸ばす（固定幅は使わない）
            dpg.set_item_width(box, -1)
        except Exception:
            pass
        return box

    def _on_store_change(self, ids: Iterable[str]) -> None:
        self._syncing = True
        try:
            for pid in ids:
                if dpg.does_item_exist(pid):
                    value = self._store.current_value(pid)
                    if value is None:
                        value = self._store.original_value(pid)
                    if pid in (
                        "runner.background",
                        "runner.line_color",
                        "runner.hud_text_color",
                        "runner.hud_meter_color",
                        "runner.hud_meter_bg_color",
                    ):
                        r, g, b, _ = self._safe_norm(value, (1.0, 1.0, 1.0, 1.0))
                        dpg.set_value(
                            pid, [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
                        )
                    else:
                        dpg.set_value(pid, value)
                    continue
                try:
                    desc = self._store.get_descriptor(pid)
                except KeyError:
                    desc = None
                if desc is None or desc.value_type != "vector":
                    continue
                value = self._store.current_value(pid)
                if value is None:
                    value = self._store.original_value(pid)
                if not isinstance(value, (list, tuple)):
                    continue
                vec = list(value)
                for i, suffix in enumerate(("x", "y", "z", "w")[: len(vec)]):
                    tag = f"{pid}::{suffix}"
                    if not dpg.does_item_exist(tag):
                        continue
                    dpg.set_value(tag, float(vec[i]))
        finally:
            self._syncing = False

    # ---- テーマ（最小） ----
    def _setup_theme(self) -> None:
        try:
            with dpg.theme() as theme:
                with dpg.theme_component(dpg.mvAll):
                    if not self._apply_styles_from_config():
                        self._apply_default_styles()
                    self._apply_colors_from_config()
            dpg.bind_theme(theme)
        except Exception:
            logger.exception("setup_theme failed; continue with defaults")

    def _apply_default_styles(self) -> bool:
        # デフォルト適用（config が無い場合のフェイルセーフ）。詳細余白が無いときは padding
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
        if self._theme is None or not isinstance(self._theme.style, dict):
            return False
        smap = {
            "window_padding": getattr(dpg, "mvStyleVar_WindowPadding", None),
            "frame_padding": getattr(dpg, "mvStyleVar_FramePadding", None),
            "item_spacing": getattr(dpg, "mvStyleVar_ItemSpacing", None),
            "frame_rounding": getattr(dpg, "mvStyleVar_FrameRounding", None),
            "grab_rounding": getattr(dpg, "mvStyleVar_GrabRounding", None),
            "grab_min_size": getattr(dpg, "mvStyleVar_GrabMinSize", None),
        }

        def _add_style_value(
            var: Any, key: str, value: float | int | Sequence[float | int]
        ) -> None:
            if not var:
                return
            try:
                if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                    seq = cast(Sequence[float | int], value)
                    if len(seq) >= 2:
                        dpg.add_theme_style(var, float(seq[0]), float(seq[1]))
                        return
                    if len(seq) == 1:
                        dpg.add_theme_style(var, float(seq[0]))
                        return
                    return
                dpg.add_theme_style(var, float(cast(float | int, value)))
            except (TypeError, ValueError):
                logger.warning("add_theme_style failed: key=%s val=%s", key, value)

        # padding/spacing 系は layout.* から適用（テーマ指定は無視して統一）
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
                int(getattr(self._layout, "frame_padding_y", max(1, self._layout.padding // 2))),
            )
        if smap["item_spacing"]:
            dpg.add_theme_style(
                smap["item_spacing"],
                int(getattr(self._layout, "item_spacing_x", self._layout.padding)),
                int(getattr(self._layout, "item_spacing_y", max(1, self._layout.padding // 2))),
            )

        # その他（丸み/サイズ等）は config の値を適用
        for key in ("frame_rounding", "grab_rounding", "grab_min_size"):
            var = smap.get(key)
            if var and key in self._theme.style:
                _add_style_value(var, key, self._theme.style[key])
        return True

    def _apply_colors_from_config(self) -> None:
        if self._theme is None or not isinstance(self._theme.colors, dict):
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
            if not var or key not in self._theme.colors:
                continue
            col = self._to_dpg_color(self._theme.colors[key])
            if col is None:
                continue
            try:
                dpg.add_theme_color(var, col)
            except (TypeError, ValueError):
                logger.warning("add_theme_color failed: key=%s val=%s", key, col)

    def _to_dpg_color(self, value: Any) -> Any:
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

    # ---- internal: drivers ----
    def _tick(self, _dt: float) -> None:  # noqa: ANN001
        try:
            dpg.render_dearpygui_frame()
        except Exception:
            logger.exception("render_dearpygui_frame failed")

    def _start_driver(self) -> None:
        if self._driver is not None:
            return
        # Prefer pyglet clock on main thread; fallback to background thread loop
        try:
            import pyglet  # type: ignore

            interval = 1.0 / 60.0
            pyglet.clock.schedule_interval(self._tick, interval)
            self._driver = ("pyglet", self._tick)
            logger.debug("ParameterWindow: pyglet driver started")
        except ImportError:
            try:
                t = Thread(target=dpg.start_dearpygui, name="DPGLoop", daemon=True)
                t.start()
                self._driver = ("thread", t)
                logger.debug("ParameterWindow: thread driver started")
            except Exception:
                logger.exception("failed to start any driver")

    def _stop_driver(self) -> None:
        drv = self._driver
        self._driver = None
        if drv is None:
            return
        kind, handle = drv
        try:
            if kind == "pyglet":
                import pyglet  # type: ignore

                pyglet.clock.unschedule(handle)  # type: ignore[arg-type]
            elif kind == "thread":
                try:
                    dpg.stop_dearpygui()
                except Exception:
                    pass
        except Exception:
            logger.exception("failed to stop driver: %s", kind)
