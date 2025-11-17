"""
どこで: `engine.ui.parameters` の Dear PyGui コンテンツ構築。
何を: ParameterStore の Descriptor から Display/HUD とパラメータテーブルを構築し、Store と UI の同期を行う。
なぜ: レイアウトや値連携の責務を `dpg_window` 本体から分離し、見通しを良くするため。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import dearpygui.dearpygui as dpg  # type: ignore

from .dpg_window_theme import ParameterWindowThemeManager
from .state import ParameterDescriptor, ParameterLayoutConfig, ParameterStore

logger = logging.getLogger("engine.ui.parameters.dpg.content")


class ParameterWindowContentBuilder:
    """ParameterWindow の Display/HUD とパラメータテーブルを構築する。"""

    def __init__(
        self,
        *,
        store: ParameterStore,
        layout: ParameterLayoutConfig,
        theme_mgr: ParameterWindowThemeManager,
    ) -> None:
        self._store = store
        self._layout = layout
        self._theme_mgr = theme_mgr
        self._syncing: bool = False

    def build_root_window(self, root_tag: str, title: str) -> None:
        with dpg.window(tag=root_tag, label=title, no_resize=False, no_collapse=True) as root:
            try:
                self.build_display_controls(parent=root)
            except Exception:
                logger.warning("failed to build runner controls; continue without Display/HUD")
        dpg.set_primary_window(root, True)

    def build_display_controls(self, parent: int | str) -> None:
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

        val = self._store.current_value("runner.background") or self._store.original_value(
            "runner.background"
        )
        if val is not None:
            bgf = self._safe_norm(val, bgf)
        val = self._store.current_value("runner.line_color") or self._store.original_value(
            "runner.line_color"
        )
        if val is not None:
            lnf = self._safe_norm(val, lnf)

        with dpg.collapsing_header(label="Display", default_open=True, parent=parent) as disp_hdr:
            try:
                th = self._theme_mgr.get_category_header_theme("Display")
                if th is not None:
                    dpg.bind_item_theme(disp_hdr, th)
            except Exception:
                pass
            table_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=table_policy) as disp_tbl:
                try:
                    tth = self._theme_mgr.get_category_table_theme("Display")
                    if tth is not None:
                        dpg.bind_item_theme(disp_tbl, tth)
                except Exception:
                    pass
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
                    r, g, b = float(bgf[0]), float(bgf[1]), float(bgf[2])
                    self.force_set_rgb_u8(
                        bg_picker,
                        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
                    )
                    dpg.configure_item(
                        bg_picker,
                        callback=lambda s, a, u: self.store_rgb01("runner.background", a),
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
                    r, g, b = float(lnf[0]), float(lnf[1]), float(lnf[2])
                    self.force_set_rgb_u8(
                        ln_picker,
                        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
                    )
                    dpg.configure_item(
                        ln_picker,
                        callback=lambda s, a, u: self.store_rgb01("runner.line_color", a),
                    )

        with dpg.collapsing_header(label="HUD", default_open=False, parent=parent) as hud_hdr:
            try:
                th = self._theme_mgr.get_category_header_theme("HUD")
                if th is not None:
                    dpg.bind_item_theme(hud_hdr, th)
            except Exception:
                pass
            table_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=table_policy) as hud_tbl:
                try:
                    tth = self._theme_mgr.get_category_table_theme("HUD")
                    if tth is not None:
                        dpg.bind_item_theme(hud_tbl, tth)
                except Exception:
                    pass
                left, right = self._label_value_ratio()
                self._add_two_columns(left, right)

                # Show HUD トグル
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Show HUD")
                    with dpg.table_cell():
                        show_val = self._store.current_value(
                            "runner.show_hud"
                        ) or self._store.original_value("runner.show_hud")
                        show = bool(show_val) if show_val is not None else True
                        dpg.add_checkbox(
                            tag="runner.show_hud",
                            default_value=show,
                            callback=self._on_widget_change,
                            user_data="runner.show_hud",
                        )

                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Text Color")
                    with dpg.table_cell():
                        tv = self._store.current_value(
                            "runner.hud_text_color"
                        ) or self._store.original_value("runner.hud_text_color")
                        tr, tg, tb, _ = (
                            self._safe_norm(tv, (0.0, 0.0, 0.0, 1.0))
                            if tv is not None
                            else (0.0, 0.0, 0.0, 1.0)
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

                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Meter")
                    with dpg.table_cell():
                        mv = self._store.current_value(
                            "runner.hud_meter_color"
                        ) or self._store.original_value("runner.hud_meter_color")
                        mr, mg, mb, _ = (
                            self._safe_norm(mv, (0.0, 1.0, 0.0, 1.0))
                            if mv is not None
                            else (0.0, 1.0, 0.0, 1.0)
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

                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Meter BG")
                    with dpg.table_cell():
                        mb_val = self._store.current_value(
                            "runner.hud_meter_bg_color"
                        ) or self._store.original_value("runner.hud_meter_bg_color")
                        br, bg, bb, _ = (
                            self._safe_norm(mb_val, (0.196, 0.196, 0.196, 1.0))
                            if mb_val is not None
                            else (0.196, 0.196, 0.196, 1.0)
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
                                "runner.hud_meter_bg_color",
                                a,
                            ),
                        )

        self.sync_display_from_store()

    def sync_display_from_store(self) -> None:
        ids = [
            "runner.background",
            "runner.line_color",
            "runner.hud_text_color",
            "runner.hud_meter_color",
            "runner.hud_meter_bg_color",
        ]
        self.on_store_change(ids)

    def mount_descriptors(self, root_tag: str, descriptors: list[ParameterDescriptor]) -> None:
        self._build_grouped_table(root_tag, descriptors)

    def _build_grouped_table(
        self,
        parent: int | str,
        descriptors: list[ParameterDescriptor],
    ) -> None:
        excluded = {
            "runner.background",
            "runner.line_color",
            "runner.hud_text_color",
            "runner.hud_meter_color",
            "runner.hud_meter_bg_color",
            "runner.show_hud",
        }
        filtered = [d for d in descriptors if d.id not in excluded]
        cat_items: dict[Any, list[ParameterDescriptor]] = {}
        cat_order: list[tuple[Any, str | None]] = []
        for d in filtered:
            cat = d.category if d.category else None
            try:
                kind = d.category_kind
            except Exception:
                kind = "pipeline" if d.source == "effect" else "shape"
            key = (kind, cat)
            if key not in cat_items:
                cat_items[key] = [d]
                cat_order.append(key)
            else:
                cat_items[key].append(d)

        def _sort_items(items: list[ParameterDescriptor]) -> list[ParameterDescriptor]:
            if not items:
                return items
            is_effect_group = any(it.source == "effect" for it in items)
            if is_effect_group:

                def _key(it: ParameterDescriptor) -> tuple[int, int, str]:
                    si = it.step_index if isinstance(it.step_index, int) else 10**9
                    po = it.param_order if isinstance(it.param_order, int) else 10**9
                    return (si, po, it.id)

                return sorted(items, key=_key)
            return sorted(items, key=lambda x: x.id)

        for key in cat_order:
            items = cat_items.get(key, [])
            self._flush_group(parent, key, _sort_items(items))

    def _flush_group(
        self,
        parent: int | str,
        key: tuple[Any, str | None],
        items: list[ParameterDescriptor],
    ) -> None:
        if not items or not any(it.supported for it in items):
            return
        _kind, cat = key
        label = cat if cat else "General"
        kind = self._category_kind(items)
        with dpg.collapsing_header(label=label, parent=parent, default_open=True) as header:
            try:
                th = self._theme_mgr.get_category_header_theme(kind)
                if th is not None:
                    dpg.bind_item_theme(header, th)
            except Exception:
                pass
            table_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=table_policy) as table:
                try:
                    tth = self._theme_mgr.get_category_table_theme(kind)
                    if tth is not None:
                        dpg.bind_item_theme(table, tth)
                except Exception:
                    pass
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as outer_tbl_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                    dpg.bind_item_theme(table, outer_tbl_theme)
                label_ratio = float(self._layout.label_column_ratio)
                label_ratio = (
                    0.1 if label_ratio < 0.1 else (0.9 if label_ratio > 0.9 else label_ratio)
                )
                rest = max(0.1, 1.0 - label_ratio)
                bcc = float(getattr(self._layout, "bars_cc_ratio", 0.7))
                bcc = 0.05 if bcc < 0.05 else (0.95 if bcc > 0.95 else bcc)
                bars_ratio = rest * bcc
                cc_ratio = rest - bars_ratio
                try:
                    dpg.add_table_column(
                        label="Parameter",
                        width_stretch=True,
                        init_width_or_weight=label_ratio,
                    )
                    dpg.add_table_column(
                        label="Bars",
                        width_stretch=True,
                        init_width_or_weight=bars_ratio,
                    )
                    dpg.add_table_column(
                        label="CC",
                        width_stretch=True,
                        init_width_or_weight=cc_ratio,
                    )
                except TypeError:
                    dpg.add_table_column(label="Parameter")
                    dpg.add_table_column(label="Bars")
                    dpg.add_table_column(label="CC")
                for it in items:
                    if not it.supported:
                        continue
                    self._create_row_3cols(table, it)

    def _category_kind(self, items: list[ParameterDescriptor]) -> str:
        if not items:
            return "shape"
        try:
            first = items[0].category_kind
        except Exception:
            try:
                if any(it.source == "effect" for it in items):
                    return "pipeline"
                return "shape"
            except Exception:
                return "shape"
        try:
            kinds = {getattr(it, "category_kind", first) for it in items}
            if len(kinds) > 1:
                logger.debug("mixed category_kind in group: %s", kinds)
        except Exception:
            pass
        return first

    def _label_value_ratio(self) -> tuple[float, float]:
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
            with dpg.table_cell():
                dpg.add_text(default_value=desc.label or desc.id)
            with dpg.table_cell():
                self._create_bars(parent=dpg.last_item() or table, desc=desc)
            with dpg.table_cell():
                self._create_cc_inputs(parent=dpg.last_item() or table, desc=desc)

    def _create_bars(self, parent: int | str, desc: ParameterDescriptor) -> None:
        vt = desc.value_type
        value = self._current_or_default(desc)
        if self._is_style_color_desc(desc):
            self._create_style_color_picker(parent, desc, value)
            return
        if vt == "vector":
            vec = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0, 0.0]
            dim = max(2, min(len(vec), 4))
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
                    with dpg.theme() as bars_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                    dpg.bind_item_theme(bars_tbl, bars_theme)
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
        hint = desc.range_hint or self._layout.derive_range(
            name=desc.id,
            value_type=desc.value_type,
            default_value=desc.default_value,
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
                    with dpg.theme() as bars_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                    dpg.bind_item_theme(bars_tbl, bars_theme)
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
        self._create_widget(parent, desc)

    def _create_cc_inputs(self, parent: int | str, desc: ParameterDescriptor) -> None:
        vt = desc.value_type
        if vt == "vector":
            vec = (
                desc.default_value
                if isinstance(desc.default_value, (list, tuple))
                else (0.0, 0.0, 0.0)
            )
            dim = max(2, min(len(vec), 4))
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
            ) as cc_tbl:
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as cc_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                dpg.bind_item_theme(cc_tbl, cc_theme)
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
                    with dpg.theme() as cc_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                dpg.bind_item_theme(cc_tbl, cc_theme)
                try:
                    dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                except TypeError:
                    dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    with dpg.table_cell():
                        self._add_cc_binding_input(None, desc)

    def _create_widget(self, parent: int | str, desc: ParameterDescriptor) -> int:
        value = self._current_or_default(desc)
        vt = desc.value_type
        if vt == "bool":
            return dpg.add_checkbox(
                tag=desc.id,
                parent=parent,
                label="",
                default_value=bool(value),
                callback=self._on_widget_change,
                user_data=desc.id,
            )
        if vt == "enum":
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
        if vt == "string":
            ml = bool(getattr(desc, "string_multiline", False))
            kwargs: dict[str, Any] = {
                "tag": desc.id,
                "parent": parent,
                "default_value": "" if value is None else str(value),
                "callback": self._on_widget_change,
                "user_data": desc.id,
            }
            if ml:
                kwargs["multiline"] = True
                kwargs["height"] = int(getattr(self._layout, "row_height", 64))
            return dpg.add_input_text(**kwargs)
        if vt == "vector":
            value_vec = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0, 0.0]
            dim = max(2, min(len(value_vec), 4))
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
            ) as vec_tbl:
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as vec_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                    dpg.bind_item_theme(vec_tbl, vec_theme)
                for _ in range(dim):
                    try:
                        dpg.add_table_column(width_stretch=True, init_width_or_weight=1.0)
                    except TypeError:
                        dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    last_slider = 0
                    for i, suffix in enumerate(("x", "y", "z", "w")[:dim]):
                        with dpg.table_cell():
                            tag = f"{desc.id}::{suffix}"
                            default_component = float(value_vec[i]) if i < len(value_vec) else 0.0
                            last_slider = dpg.add_slider_float(
                                tag=tag,
                                label="",
                                default_value=default_component,
                                min_value=0.0,
                                max_value=1.0,
                                format=f"%.{self._layout.value_precision}f",
                                callback=self._on_widget_change,
                                user_data=(desc.id, i),
                            )
                            dpg.set_item_width(last_slider, -1)
            return last_slider
        hint = desc.range_hint or self._layout.derive_range(
            name=desc.id,
            value_type=desc.value_type,
            default_value=desc.default_value,
        )
        if vt == "int":
            return self._create_int(parent, desc, value, hint)
        return self._create_float(parent, desc, value, hint)

    def _create_int(
        self,
        parent: int | str,
        desc: ParameterDescriptor,
        value: Any,
        hint: Any,
    ) -> int:
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
        ) as tbl:
            var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
            if var_cell_padding is not None:
                pad = int(self._layout.padding)
                with dpg.theme() as int_tbl_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(var_cell_padding, pad, pad)
                dpg.bind_item_theme(tbl, int_tbl_theme)
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
        self,
        parent: int | str,
        desc: ParameterDescriptor,
        value: Any,
        hint: Any,
    ) -> int:
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]) or 0,
        ) as tbl:
            var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
            if var_cell_padding is not None:
                pad = int(self._layout.padding)
                with dpg.theme() as flt_tbl_theme:
                    with dpg.theme_component(dpg.mvAll):
                        dpg.add_theme_style(var_cell_padding, pad, pad)
                dpg.bind_item_theme(tbl, flt_tbl_theme)
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

    def _create_style_color_picker(
        self,
        parent: int | str,
        desc: ParameterDescriptor,
        value: Any,
    ) -> int:
        base = self._safe_norm(value, (0.0, 0.0, 0.0, 1.0))
        r, g, b, _ = base
        picker = dpg.add_color_edit(
            tag=desc.id,
            parent=parent,
            default_value=[int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
            no_label=True,
            no_alpha=True,
            alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
            display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
            display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
            input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
            alpha_bar=False,
        )
        self.force_set_rgb_u8(
            picker,
            [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
        )
        dpg.configure_item(
            picker,
            callback=lambda s, a, u: self.store_rgb01(u, a),
            user_data=desc.id,
        )
        return picker

    def _is_style_color_desc(self, desc: ParameterDescriptor) -> bool:
        return isinstance(desc.id, str) and desc.id.endswith(".color") and ".style#" in desc.id

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
                dim = max(2, min(len(vec), 4))
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

    def on_store_change(self, ids: Iterable[str]) -> None:
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
                            pid,
                            [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
                        )
                    elif isinstance(pid, str) and pid.endswith(".color") and ".style#" in pid:
                        r, g, b, _ = self._safe_norm(value, (0.0, 0.0, 0.0, 1.0))
                        self.force_set_rgb_u8(
                            pid,
                            [
                                int(round(r * 255)),
                                int(round(g * 255)),
                                int(round(b * 255)),
                            ],
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

    def _on_cc_binding_change(self, sender: int, app_data: Any, user_data: Any) -> None:
        try:
            pid = str(user_data)
            text = str(app_data).strip()
        except Exception:
            return
        if not text:
            self._store.bind_cc(pid, None)
            return
        try:
            i = int(float(text))
        except Exception:
            self._store.bind_cc(pid, None)
            dpg.set_value(f"{pid}::cc", "")
            return
        if i < 0:
            i = 0
        if i > 127:
            i = 127
        self._store.bind_cc(pid, i)
        try:
            dpg.set_value(f"{pid}::cc", str(i))
        except Exception:
            pass

    def _add_cc_binding_input(self, parent: int | str | None, desc: ParameterDescriptor) -> int:
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
            dpg.set_item_width(box, -1)
        except Exception:
            pass
        return box

    def _add_cc_binding_input_component(
        self,
        parent: int | str | None,
        desc: ParameterDescriptor,
        idx: int,
    ) -> int:
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
            dpg.set_item_width(box, -1)
        except Exception:
            pass
        return box

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
            if isinstance(pid, str) and pid.endswith(".color") and ".style#" in pid:
                value_tuple: tuple[float, ...] = (
                    float(rgba[0]),
                    float(rgba[1]),
                    float(rgba[2]),
                )
            else:
                try:
                    value_tuple = tuple(float(v) for v in rgba)  # type: ignore[misc]
                except Exception:
                    r, g, b, a = float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3])
                    value_tuple = (r, g, b, a)
        except Exception:
            logger.exception("store_rgb01 failed to normalize: pid=%s val=%s", pid, app_data)
            return
        self._store.set_override(pid, value_tuple)

    def _safe_norm(
        self,
        value: Any,
        default: tuple[float, float, float, float],
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
