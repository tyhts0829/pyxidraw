"""
どこで: `engine.ui.parameters` の Dear PyGui コンテンツ構築。
何を: ParameterStore の Descriptor から Style セクションとパラメータテーブルを構築し、Store と UI の同期を行う。
なぜ: レイアウトや値連携の責務を `dpg_window` 本体から分離し、見通しを良くするため。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import dearpygui.dearpygui as dpg  # type: ignore

from .dpg_window_theme import ParameterWindowThemeManager
from .state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
    base_range_for_descriptor,
    effective_range_for_descriptor,
)

logger = logging.getLogger("engine.ui.parameters.dpg.content")


# Style/HUD 用パラメータ ID
STYLE_BG_ID = "runner.background"
STYLE_LINE_COLOR_ID = "runner.line_color"
STYLE_LINE_THICKNESS_ID = "runner.line_thickness"

HUD_TEXT_COLOR_ID = "runner.hud_text_color"
HUD_METER_COLOR_ID = "runner.hud_meter_color"
HUD_METER_BG_COLOR_ID = "runner.hud_meter_bg_color"

STYLE_COLOR_IDS: set[str] = {STYLE_BG_ID, STYLE_LINE_COLOR_ID}
HUD_COLOR_IDS: set[str] = {HUD_TEXT_COLOR_ID, HUD_METER_COLOR_ID, HUD_METER_BG_COLOR_ID}
ALL_STYLE_PARAM_IDS: set[str] = STYLE_COLOR_IDS | HUD_COLOR_IDS | {STYLE_LINE_THICKNESS_ID}

# Palette 用パラメータ ID
PALETTE_L_ID = "palette.L"
PALETTE_C_ID = "palette.C"
PALETTE_H_ID = "palette.h"
PALETTE_TYPE_ID = "palette.type"
PALETTE_STYLE_ID = "palette.style"
PALETTE_N_COLORS_ID = "palette.n_colors"
ALL_PALETTE_PARAM_IDS: set[str] = {
    PALETTE_L_ID,
    PALETTE_C_ID,
    PALETTE_H_ID,
    PALETTE_TYPE_ID,
    PALETTE_STYLE_ID,
    PALETTE_N_COLORS_ID,
}

PALETTE_PREVIEW_TEXT_ID = "__pxd_palette_preview__"

# Style/HUD 用既定値
DEFAULT_BG_COLOR = (1.0, 1.0, 1.0, 1.0)
DEFAULT_LINE_COLOR = (0.0, 0.0, 0.0, 1.0)
DEFAULT_HUD_TEXT_COLOR = (0.0, 0.0, 0.0, 1.0)
DEFAULT_HUD_METER_COLOR = (0.0, 1.0, 0.0, 1.0)
DEFAULT_HUD_METER_BG_COLOR = (0.196, 0.196, 0.196, 1.0)

DEFAULT_LINE_THICKNESS = 0.0006
LINE_THICKNESS_MIN = 0.0001
LINE_THICKNESS_MAX = 0.001

# テーブル比率用のクランプ閾値
MIN_LABEL_RATIO = 0.1
MAX_LABEL_RATIO = 0.9
MIN_REST_RATIO = 0.1
MIN_BARS_CC_RATIO = 0.05
MAX_BARS_CC_RATIO = 0.95


@dataclass(frozen=True)
class _StyleLayerEntry:
    """Style セクションに表示するレイヤーごとのスタイル設定。"""

    key: str
    label: str
    color_desc: ParameterDescriptor | None = None
    thickness_desc: ParameterDescriptor | None = None


class ParameterWindowContentBuilder:
    """ParameterWindow の Style セクションとパラメータテーブルを構築する。"""

    def __init__(
        self,
        *,
        store: ParameterStore,
        layout: ParameterLayoutConfig,
        theme_mgr: ParameterWindowThemeManager,
    ) -> None:
        """Store/レイアウト/テーマ管理クラスを受け取り、状態を初期化する。"""
        self._store = store
        self._layout = layout
        self._theme_mgr = theme_mgr
        self._syncing: bool = False
        self._style_param_ids: set[str] = set()
        self._palette_param_ids: set[str] = set()
        self._palette_preview_id: int | str | None = None
        self._palette_swatches_container: int | str | None = None
        self._searchable_enum_state: dict[str, dict[str, Any]] = {}

    def build_root_window(self, root_tag: str, title: str) -> None:
        """ルートウィンドウを構築する。Style/パラメータ群は mount 時に追加する。"""
        with dpg.window(tag=root_tag, label=title, no_resize=False, no_collapse=True) as root:
            pass
        dpg.set_primary_window(root, True)

    def build_style_controls(
        self,
        parent: int | str,
        descriptors: list[ParameterDescriptor],
    ) -> None:
        """Style 用のコントロール群を構築し、Store と初期同期する。"""
        self._style_param_ids.clear()
        bgf, lnf = self._resolve_canvas_colors()
        bgf = self._resolve_style_color_from_store(STYLE_BG_ID, bgf)
        lnf = self._resolve_style_color_from_store(STYLE_LINE_COLOR_ID, lnf)

        tr, tg, tb, _ = self._resolve_style_color_from_store(
            HUD_TEXT_COLOR_ID,
            DEFAULT_HUD_TEXT_COLOR,
        )
        mr, mg, mb, _ = self._resolve_style_color_from_store(
            HUD_METER_COLOR_ID,
            DEFAULT_HUD_METER_COLOR,
        )
        br, bg, bb, _ = self._resolve_style_color_from_store(
            HUD_METER_BG_COLOR_ID,
            DEFAULT_HUD_METER_BG_COLOR,
        )
        layer_entries = self._collect_style_entries(descriptors)

        with dpg.collapsing_header(label="Style", default_open=True, parent=parent) as style_hdr:
            th = self._theme_mgr.get_category_header_theme("Style")
            if th is not None:
                dpg.bind_item_theme(style_hdr, th)
            table_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=table_policy) as style_tbl:
                tth = self._theme_mgr.get_category_table_theme("Style")
                if tth is not None:
                    dpg.bind_item_theme(style_tbl, tth)
                left, right = self._label_value_ratio()
                self._add_two_columns(left, right)
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Background Color")
                    with dpg.table_cell():
                        bg_picker = dpg.add_color_edit(
                            tag=STYLE_BG_ID,
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
                        self._style_param_ids.add(STYLE_BG_ID)
                        self._set_full_width(bg_picker)
                    r, g, b = float(bgf[0]), float(bgf[1]), float(bgf[2])
                    self.force_set_rgb_u8(
                        bg_picker,
                        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
                    )
                    dpg.configure_item(
                        bg_picker,
                        callback=lambda s, a, u: self.store_rgb01(STYLE_BG_ID, a),
                    )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Global Line Color")
                    with dpg.table_cell():
                        ln_picker = dpg.add_color_edit(
                            tag=STYLE_LINE_COLOR_ID,
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
                        self._style_param_ids.add(STYLE_LINE_COLOR_ID)
                        self._set_full_width(ln_picker)
                    r, g, b = float(lnf[0]), float(lnf[1]), float(lnf[2])
                    self.force_set_rgb_u8(
                        ln_picker,
                        [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))],
                    )
                    dpg.configure_item(
                        ln_picker,
                        callback=lambda s, a, u: self.store_rgb01(STYLE_LINE_COLOR_ID, a),
                    )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Global Thickness")
                    with dpg.table_cell():
                        try:
                            th_val = float(
                                self._store.current_value(STYLE_LINE_THICKNESS_ID)
                                or self._store.original_value(STYLE_LINE_THICKNESS_ID)
                                or DEFAULT_LINE_THICKNESS
                            )
                        except Exception:
                            th_val = DEFAULT_LINE_THICKNESS
                        thickness_desc = None
                        for d in descriptors:
                            if d.id == STYLE_LINE_THICKNESS_ID:
                                thickness_desc = d
                                break
                        if thickness_desc is not None:
                            mn, mx = self._effective_range(thickness_desc)
                        else:
                            mn, mx = LINE_THICKNESS_MIN, LINE_THICKNESS_MAX
                        precision = getattr(self._layout, "value_precision", 6)
                        if precision < 6:
                            precision = 6
                        with dpg.group(horizontal=False) as th_group:
                            th_picker = dpg.add_slider_float(
                                tag=STYLE_LINE_THICKNESS_ID,
                                label="",
                                default_value=th_val,
                                min_value=float(mn),
                                max_value=float(mx),
                                format=f"%.{precision}f",
                                callback=self._on_widget_change,
                                user_data=STYLE_LINE_THICKNESS_ID,
                            )
                            self._set_full_width(th_picker)
                            self._style_param_ids.add(STYLE_LINE_THICKNESS_ID)
                            if thickness_desc is not None:
                                self._create_minmax_inputs(th_group, thickness_desc)
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("HUD: Text Color")
                    with dpg.table_cell():
                        tx_picker = dpg.add_color_edit(
                            tag=HUD_TEXT_COLOR_ID,
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
                        self._style_param_ids.add(HUD_TEXT_COLOR_ID)
                        self._set_full_width(tx_picker)
                    dpg.configure_item(
                        tx_picker,
                        callback=lambda s, a, u: self.store_rgb01(HUD_TEXT_COLOR_ID, a),
                    )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("HUD: Meter Color")
                    with dpg.table_cell():
                        mt_picker = dpg.add_color_edit(
                            tag=HUD_METER_COLOR_ID,
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
                        self._style_param_ids.add(HUD_METER_COLOR_ID)
                        self._set_full_width(mt_picker)
                    dpg.configure_item(
                        mt_picker,
                        callback=lambda s, a, u: self.store_rgb01(HUD_METER_COLOR_ID, a),
                    )
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("HUD: Meter BG Color")
                    with dpg.table_cell():
                        mb_picker = dpg.add_color_edit(
                            tag=HUD_METER_BG_COLOR_ID,
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
                        self._style_param_ids.add(HUD_METER_BG_COLOR_ID)
                        self._set_full_width(mb_picker)
                    dpg.configure_item(
                        mb_picker,
                        callback=lambda s, a, u: self.store_rgb01(
                            HUD_METER_BG_COLOR_ID,
                            a,
                        ),
                    )
                # レイヤー行を同一テーブルに追加
                if layer_entries:
                    for entry in layer_entries:
                        with dpg.table_row():
                            with dpg.table_cell():
                                dpg.add_text(f"{entry.label or 'Layer'}: Color")
                            with dpg.table_cell():
                                if entry.color_desc is None:
                                    dpg.add_text("-")
                                else:
                                    self._style_param_ids.add(entry.color_desc.id)
                                    value = self._current_or_default(entry.color_desc)
                                    r, g, b, _ = self._safe_norm(value, (0.0, 0.0, 0.0, 1.0))
                                    picker = dpg.add_color_edit(
                                        tag=entry.color_desc.id,
                                        default_value=[
                                            int(round(r * 255)),
                                            int(round(g * 255)),
                                            int(round(b * 255)),
                                        ],
                                        no_label=True,
                                        no_alpha=True,
                                        alpha_preview=getattr(
                                            dpg, "mvColorEdit_AlphaPreviewHalf", 1
                                        ),
                                        display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                                        display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                                        input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                                        alpha_bar=False,
                                    )
                                    self._set_full_width(picker)
                                    self.force_set_rgb_u8(
                                        picker,
                                        [
                                            int(round(r * 255)),
                                            int(round(g * 255)),
                                            int(round(b * 255)),
                                        ],
                                    )
                                    dpg.configure_item(
                                        picker,
                                        callback=lambda s, a, u: self.store_rgb01(u, a),
                                        user_data=entry.color_desc.id,
                                    )
                        with dpg.table_row():
                            with dpg.table_cell():
                                dpg.add_text(f"{entry.label or 'Layer'}: Thickness")
                            with dpg.table_cell():
                                if entry.thickness_desc is None:
                                    dpg.add_text("-")
                                else:
                                    desc = entry.thickness_desc
                                    self._style_param_ids.add(desc.id)
                                    value = self._current_or_default(desc)
                                    hint = desc.range_hint or self._layout.derive_range(
                                        name=desc.id,
                                        value_type=desc.value_type,
                                        default_value=desc.default_value,
                                    )
                                    precision = getattr(self._layout, "value_precision", 6)
                                    if precision < 6:
                                        precision = 6
                                    slider_id = dpg.add_slider_float(
                                        tag=desc.id,
                                        label="",
                                        default_value=(
                                            float(value)
                                            if value is not None
                                            else float(hint.min_value)
                                        ),
                                        min_value=float(hint.min_value),
                                        max_value=float(hint.max_value),
                                        format=f"%.{precision}f",
                                        callback=self._on_widget_change,
                                        user_data=desc.id,
                                    )
                                    self._set_full_width(slider_id)

    def sync_style_from_store(self) -> None:
        if not self._style_param_ids:
            return
        self.on_store_change(self._style_param_ids)

    def build_palette_controls(
        self,
        parent: int | str,
        descriptors: list[ParameterDescriptor],
    ) -> None:
        """Palette 用のコントロール群を構築し、Store と初期同期する。"""
        self._palette_param_ids.clear()
        desc_index: dict[str, ParameterDescriptor] = {d.id: d for d in descriptors}
        l_desc = desc_index.get(PALETTE_L_ID)
        c_desc = desc_index.get(PALETTE_C_ID)
        h_desc = desc_index.get(PALETTE_H_ID)
        type_desc = desc_index.get(PALETTE_TYPE_ID)
        style_desc = desc_index.get(PALETTE_STYLE_ID)
        n_desc = desc_index.get(PALETTE_N_COLORS_ID)
        auto_mode_desc = desc_index.get("palette.auto_apply_mode")
        if (
            l_desc is None
            and c_desc is None
            and h_desc is None
            and type_desc is None
            and style_desc is None
            and n_desc is None
        ):
            return

        with dpg.collapsing_header(label="Palette", default_open=True, parent=parent) as pal_hdr:
            th = self._theme_mgr.get_category_header_theme("palette")
            if th is not None:
                dpg.bind_item_theme(pal_hdr, th)
            table_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=table_policy) as pal_tbl:
                # Style セクションと同等のセル間隔を適用
                self._apply_cell_padding_theme(pal_tbl)
                tth = self._theme_mgr.get_category_table_theme("palette")
                if tth is not None:
                    dpg.bind_item_theme(pal_tbl, tth)
                # shape/pipeline と同じ列比率（Parameter/Bars/MinMax/CC）を使う
                label_ratio = float(self._layout.label_column_ratio)
                label_ratio = max(MIN_LABEL_RATIO, min(MAX_LABEL_RATIO, label_ratio))
                rest = max(MIN_REST_RATIO, 1.0 - label_ratio)
                bcc = float(getattr(self._layout, "bars_cc_ratio", 0.7))
                bcc = max(MIN_BARS_CC_RATIO, min(MAX_BARS_CC_RATIO, bcc))
                bars_ratio = rest * bcc
                remaining = max(MIN_REST_RATIO, rest - bars_ratio)
                minmax_ratio = remaining * 0.5
                cc_ratio = remaining - minmax_ratio
                self._add_stretch_column("Parameter", label_ratio)
                self._add_stretch_column("Bars", bars_ratio)
                self._add_stretch_column("Min/Max", minmax_ratio)
                self._add_stretch_column("CC", cc_ratio)

                # Base color (L/C/h) + type/style/colors + auto_apply_mode を既存ロジックで 3 列行として構築
                for desc in (
                    l_desc,
                    c_desc,
                    h_desc,
                    type_desc,
                    style_desc,
                    n_desc,
                    auto_mode_desc,
                ):
                    if desc is None:
                        continue
                    self._palette_param_ids.add(desc.id)
                    self._create_row_3cols(pal_tbl, desc)
                # プレビュー行（ラベル列 + Bars 列にスウォッチ群, MinMax/CC 列は空）
                with dpg.table_row():
                    with dpg.table_cell():
                        dpg.add_text("Preview")
                    with dpg.table_cell():
                        self._palette_swatches_container = dpg.add_group(horizontal=True)
                    with dpg.table_cell():
                        pass
                    with dpg.table_cell():
                        pass

    def sync_palette_from_store(self) -> None:
        if not self._palette_param_ids:
            return
        self.on_store_change(self._palette_param_ids)
        self._refresh_palette_preview()

    def mount_descriptors(self, root_tag: str, descriptors: list[ParameterDescriptor]) -> None:
        self.build_style_controls(root_tag, descriptors)
        self.build_palette_controls(root_tag, descriptors)
        self._build_grouped_table(root_tag, descriptors)
        self.sync_style_from_store()
        self.sync_palette_from_store()

    def _build_grouped_table(
        self,
        parent: int | str,
        descriptors: list[ParameterDescriptor],
    ) -> None:
        excluded = (
            set(self._style_param_ids)
            | set(self._palette_param_ids)
            | {"runner.show_hud", STYLE_LINE_THICKNESS_ID}
        )
        filtered = [
            d for d in descriptors if d.id not in excluded and not self._is_style_descriptor(d)
        ]
        cat_items: dict[Any, list[ParameterDescriptor]] = {}
        cat_order: list[tuple[Any, str | None]] = []
        for d in filtered:
            cat = d.category if d.category else None
            kind = d.category_kind
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
            th = self._theme_mgr.get_category_header_theme(kind)
            if th is not None:
                dpg.bind_item_theme(header, th)
            table_policy = self._dpg_policy(
                ["mvTable_SizingStretchProp", "mvTable_SizingStretchSame"]
            )
            with dpg.table(header_row=False, policy=table_policy) as table:
                tth = self._theme_mgr.get_category_table_theme(kind)
                if tth is not None:
                    dpg.bind_item_theme(table, tth)
                var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
                if var_cell_padding is not None:
                    cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
                    cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
                    with dpg.theme() as outer_tbl_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(var_cell_padding, cx, cy)
                    dpg.bind_item_theme(table, outer_tbl_theme)
                label_ratio = float(self._layout.label_column_ratio)
                label_ratio = max(MIN_LABEL_RATIO, min(MAX_LABEL_RATIO, label_ratio))
                rest = max(MIN_REST_RATIO, 1.0 - label_ratio)
                bcc = float(getattr(self._layout, "bars_cc_ratio", 0.7))
                bcc = max(MIN_BARS_CC_RATIO, min(MAX_BARS_CC_RATIO, bcc))
                bars_ratio = rest * bcc
                remaining = max(MIN_REST_RATIO, rest - bars_ratio)
                minmax_ratio = remaining * 0.5
                cc_ratio = remaining - minmax_ratio
                self._add_stretch_column("Parameter", label_ratio)
                self._add_stretch_column("Bars", bars_ratio)
                self._add_stretch_column("Min/Max", minmax_ratio)
                self._add_stretch_column("CC", cc_ratio)
                for it in items:
                    if not it.supported:
                        continue
                    self._create_row_3cols(table, it)

    def _category_kind(self, items: list[ParameterDescriptor]) -> str:
        """グループ内の Descriptor からカテゴリ種別を決定する。"""
        if not items:
            return "shape"
        first = items[0].category_kind
        kinds = {it.category_kind for it in items}
        if len(kinds) > 1:
            logger.debug("mixed category_kind in group: %s", kinds)
        return first

    def _label_value_ratio(self) -> tuple[float, float]:
        """Style テーブル用のラベル列と値列の比率を計算する。"""
        left = float(self._layout.label_column_ratio)
        left = max(MIN_LABEL_RATIO, min(MAX_LABEL_RATIO, left))
        right = max(MIN_REST_RATIO, 1.0 - left)
        return left, right

    # ---- レンジ（min/max）ヘルパ ----
    def _base_range(self, desc: ParameterDescriptor) -> tuple[float, float]:
        """Descriptor から UI 用の基本レンジを取得する。"""
        return base_range_for_descriptor(desc, layout=self._layout)

    def _effective_range(self, desc: ParameterDescriptor) -> tuple[float, float]:
        """基本レンジと Store のオーバーライドをマージした UI レンジを返す。"""
        return effective_range_for_descriptor(
            desc,
            self._store,
            layout=self._layout,
        )

    def _resolve_canvas_colors(
        self,
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
        """config.yaml から背景色とライン色の既定値を解決する。"""
        try:
            from util.utils import load_config as _load_cfg

            cfg = _load_cfg() or {}
        except Exception:
            cfg = {}
        canvas = cfg.get("canvas", {}) if isinstance(cfg, dict) else {}
        bg_raw = canvas.get("background_color", DEFAULT_BG_COLOR)
        ln_raw = canvas.get("line_color", DEFAULT_LINE_COLOR)
        bgf = self._safe_norm(bg_raw, DEFAULT_BG_COLOR)
        lnf = self._safe_norm(ln_raw, DEFAULT_LINE_COLOR)
        return bgf, lnf

    def _add_two_columns(self, left: float, right: float) -> None:
        """ラベル列と値列の 2 列テーブルヘッダを追加する。"""
        self._add_stretch_column("Parameter", left)
        self._add_stretch_column("Value", right)

    def _add_stretch_column(self, label: str, weight: float | None = None) -> None:
        """幅ストレッチ付きのテーブル列を追加する。古い DPG では weight を省略して追加する。"""
        kwargs: dict[str, Any] = {"label": label}
        if weight is not None:
            kwargs["width_stretch"] = True
            kwargs["init_width_or_weight"] = float(weight)
        try:
            dpg.add_table_column(**kwargs)
        except TypeError:
            dpg.add_table_column(label=label)

    def _collect_style_entries(
        self, descriptors: list[ParameterDescriptor]
    ) -> list[_StyleLayerEntry]:
        """style 系 Descriptor をレイヤー単位に束ねる。"""
        grouped: dict[str, _StyleLayerEntry] = {}
        for desc in descriptors:
            if not self._is_style_descriptor(desc):
                continue
            key = self._style_owner_key(desc)
            entry = grouped.get(key) or _StyleLayerEntry(
                key=key,
                label=self._style_label(desc, key),
            )
            if desc.id.endswith(".color"):
                entry = _StyleLayerEntry(
                    key=entry.key,
                    label=entry.label,
                    color_desc=desc,
                    thickness_desc=entry.thickness_desc,
                )
            elif desc.id.endswith(".thickness"):
                entry = _StyleLayerEntry(
                    key=entry.key,
                    label=entry.label,
                    color_desc=entry.color_desc,
                    thickness_desc=desc,
                )
            grouped[key] = entry
        return sorted(grouped.values(), key=lambda e: (e.label, e.key))

    def _style_owner_key(self, desc: ParameterDescriptor) -> str:
        """スタイル行を束ねるキーを決める。"""
        if isinstance(desc.id, str) and desc.id.startswith("layer."):
            try:
                return desc.id.split(".")[1]
            except Exception:
                return desc.id
        if desc.pipeline_uid:
            return str(desc.pipeline_uid)
        if desc.category:
            return str(desc.category)
        return str(desc.id)

    def _style_label(self, desc: ParameterDescriptor, fallback_key: str) -> str:
        """スタイル行の表示ラベル。"""
        if isinstance(desc.id, str) and desc.id.startswith("layer."):
            if desc.label:
                return str(desc.label.replace(" Color", "").replace(" Thickness", ""))
            return fallback_key
        if desc.category:
            return str(desc.category)
        if desc.pipeline_uid:
            return str(desc.pipeline_uid)
        if desc.label:
            return str(desc.label)
        return str(desc.id)

    def _create_row_3cols(self, table: int | str, desc: ParameterDescriptor) -> None:
        """カテゴリテーブルに 4 列（Label/Bars/MinMax/CC）の行を追加する。"""
        with dpg.table_row(parent=table):
            with dpg.table_cell():
                dpg.add_text(default_value=desc.label or desc.id)
            with dpg.table_cell():
                self._create_bars(parent=dpg.last_item() or table, desc=desc)
            with dpg.table_cell():
                self._create_minmax_inputs(parent=dpg.last_item() or table, desc=desc)
            with dpg.table_cell():
                self._create_cc_inputs(parent=dpg.last_item() or table, desc=desc)

    def _create_bars(self, parent: int | str, desc: ParameterDescriptor) -> None:
        """値種別に応じた Bars 列のスライダ群を生成する。"""
        vt = desc.value_type
        value = self._current_or_default(desc)
        if self._is_style_color_desc(desc):
            self._create_style_color_picker(parent, desc, value)
            return
        if vt == "vector":
            vec = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0, 0.0]
            dim = max(2, min(len(vec), 4))
            # vector 用のレンジは全成分共通の min/max を用いる
            mn, mx = self._effective_range(desc)
            vmin = [mn] * dim
            vmax = [mx] * dim
            self._create_vector_sliders(parent, desc, vec, vmin, vmax)
            return
        mn, mx = self._effective_range(desc)
        if vt in {"int", "float"}:
            precision = getattr(self._layout, "value_precision", 6)
            desc_id = str(desc.id)
            if desc_id == STYLE_LINE_THICKNESS_ID or desc_id.endswith(".thickness"):
                if precision < 6:
                    precision = 6
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]),
            ) as bars_tbl:
                self._apply_cell_padding_theme(bars_tbl)
                self._add_stretch_column("", 1.0)
                with dpg.table_row():
                    with dpg.table_cell():
                        if vt == "int":
                            slider_id = dpg.add_slider_int(
                                tag=desc.id,
                                label="",
                                default_value=int(value) if value is not None else 0,
                                min_value=int(mn),
                                max_value=int(mx),
                                callback=self._on_widget_change,
                                user_data=desc.id,
                            )
                        else:
                            slider_id = dpg.add_slider_float(
                                tag=desc.id,
                                label="",
                                default_value=float(value) if value is not None else 0.0,
                                min_value=float(mn),
                                max_value=float(mx),
                                format=f"%.{precision}f",
                                callback=self._on_widget_change,
                                user_data=desc.id,
                            )
                        dpg.set_item_width(slider_id, -1)
            return
        self._create_widget(parent, desc)

    def _create_cc_inputs(self, parent: int | str, desc: ParameterDescriptor) -> None:
        """パラメータに対応する CC 番号入力群をテーブル内に生成する。"""
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
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]),
            ) as cc_tbl:
                self._apply_cell_padding_theme(cc_tbl)
                for _ in range(dim):
                    self._add_stretch_column("", 1.0)
                with dpg.table_row():
                    for i in range(dim):
                        with dpg.table_cell():
                            self._add_cc_binding_input_component(None, desc, i)
            return
        if vt in {"float", "int"}:
            with dpg.table(
                parent=parent,
                header_row=False,
                policy=self._dpg_policy(["mvTable_SizingStretchSame"]),
            ) as cc_tbl:
                self._apply_cell_padding_theme(cc_tbl)
                self._add_stretch_column("", 1.0)
                with dpg.table_row():
                    with dpg.table_cell():
                        self._add_cc_binding_input(None, desc)

    def _create_minmax_inputs(self, parent: int | str, desc: ParameterDescriptor) -> None:
        """Bars 列に対応する min/max 入力ボックスを生成する。"""
        vt = desc.value_type
        if vt not in {"float", "int", "vector"}:
            return
        mn, mx = self._effective_range(desc)
        # vector も共通の min/max を用いる
        tag_min = f"{desc.id}::min"
        tag_max = f"{desc.id}::max"
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]),
        ) as mm_tbl:
            self._apply_cell_padding_theme(mm_tbl)
            # Min / Max を 1:1 でセル幅にストレッチさせる
            self._add_stretch_column("", 1.0)
            self._add_stretch_column("", 1.0)
            with dpg.table_row():
                with dpg.table_cell():
                    if vt == "int":
                        min_id = dpg.add_input_int(
                            tag=tag_min,
                            default_value=int(mn),
                            step=0,
                            step_fast=0,
                            callback=self._on_minmax_change,
                            user_data=(desc.id, "min"),
                        )
                    else:
                        min_id = dpg.add_input_float(
                            tag=tag_min,
                            default_value=float(mn),
                            # min/max は必要最低限の桁数で表示する
                            format="%g",
                            step=0.0,
                            step_fast=0.0,
                            callback=self._on_minmax_change,
                            user_data=(desc.id, "min"),
                        )
                    self._set_full_width(min_id)
                with dpg.table_cell():
                    if vt == "int":
                        max_id = dpg.add_input_int(
                            tag=tag_max,
                            default_value=int(mx),
                            step=0,
                            step_fast=0,
                            callback=self._on_minmax_change,
                            user_data=(desc.id, "max"),
                        )
                    else:
                        max_id = dpg.add_input_float(
                            tag=tag_max,
                            default_value=float(mx),
                            # min/max は必要最低限の桁数で表示する
                            format="%g",
                            step=0.0,
                            step_fast=0.0,
                            callback=self._on_minmax_change,
                            user_data=(desc.id, "max"),
                        )
                    self._set_full_width(max_id)
        # それぞれに簡単な tooltip を付与
        try:
            with dpg.tooltip(tag_min):
                dpg.add_text("min")
            with dpg.tooltip(tag_max):
                dpg.add_text("max")
        except Exception:
            pass

    def _create_widget(self, parent: int | str, desc: ParameterDescriptor) -> int | str:
        """値種別に応じて単一ウィジェットを生成し、そのタグを返す。"""
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
            search_enabled = bool(getattr(desc, "searchable", False)) or str(desc.id).endswith(
                ".font"
            )
            use_radio = False
            if desc.id == "palette.type":
                use_radio = True
            elif len(items) <= 5:
                use_radio = True
            if search_enabled:
                use_radio = False
            if search_enabled:
                return self._create_searchable_combo(
                    parent=parent, desc=desc, items=items, default=default
                )
            if use_radio:
                return dpg.add_radio_button(
                    tag=desc.id,
                    items=items,
                    parent=parent,
                    default_value=default,
                    horizontal=True,
                    callback=self._on_widget_change,
                    user_data=desc.id,
                )
            combo = dpg.add_combo(
                tag=desc.id,
                items=items,
                parent=parent,
                default_value=default,
                callback=self._on_widget_change,
                user_data=desc.id,
            )
            try:
                dpg.set_item_width(combo, -1)
            except Exception:
                pass
            return combo
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
                height = int(getattr(desc, "string_height", 0) or 0)
                if height <= 0:
                    height = int(getattr(self._layout, "row_height", 64))
                kwargs["height"] = height
            box = dpg.add_input_text(**kwargs)
            try:
                dpg.set_item_width(box, -1)
            except Exception:
                pass
            return box
        if vt == "vector":
            value_vec = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0, 0.0]
            dim = max(2, min(len(value_vec), 4))
            mn, mx = self._effective_range(desc)
            vmin = [mn] * dim
            vmax = [mx] * dim
            return self._create_vector_sliders(parent, desc, value_vec, vmin, vmax)
        if vt == "int":
            return self._create_int(parent, desc, value)
        return self._create_float(parent, desc, value)

    def _create_searchable_combo(
        self, *, parent: int | str, desc: ParameterDescriptor, items: list[str], default: str
    ) -> int | str:
        """検索入力付きのコンボボックスを生成する。"""
        base_items = list(items)
        container = dpg.add_group(parent=parent, horizontal=False)
        search_tag = f"{desc.id}::search"
        search_box = dpg.add_input_text(
            tag=search_tag,
            parent=container,
            hint="filter",
            callback=self._on_enum_filter_change,
            user_data=desc.id,
        )
        self._set_full_width(search_box)
        combo = dpg.add_combo(
            tag=desc.id,
            items=base_items,
            parent=container,
            default_value=default,
            callback=self._on_widget_change,
            user_data=desc.id,
        )
        self._set_full_width(combo)
        self._searchable_enum_state[desc.id] = {"items": base_items, "combo_id": desc.id}
        return combo

    def _create_int(
        self,
        parent: int | str,
        desc: ParameterDescriptor,
        value: Any,
    ) -> int | str:
        """int パラメータ用のスライダと CC 入力を生成する。"""
        mn, mx = self._effective_range(desc)
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]),
        ) as tbl:
            self._apply_cell_padding_theme(tbl)
            self._add_stretch_column("", 1.0)
            self._add_stretch_column("", 0.35)
            with dpg.table_row():
                with dpg.table_cell():
                    slider_id = dpg.add_slider_int(
                        tag=desc.id,
                        label="",
                        default_value=int(value) if value is not None else 0,
                        min_value=int(mn),
                        max_value=int(mx),
                        callback=self._on_widget_change,
                        user_data=desc.id,
                    )
                    self._set_full_width(slider_id)
                with dpg.table_cell():
                    self._add_cc_binding_input(None, desc)
        return slider_id

    def _create_float(
        self,
        parent: int | str,
        desc: ParameterDescriptor,
        value: Any,
    ) -> int | str:
        """float パラメータ用のスライダと CC 入力を生成する。"""
        mn, mx = self._effective_range(desc)
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]),
        ) as tbl:
            self._apply_cell_padding_theme(tbl)
            self._add_stretch_column("", 1.0)
            self._add_stretch_column("", 0.35)
            with dpg.table_row():
                with dpg.table_cell():
                    slider_id = dpg.add_slider_float(
                        tag=desc.id,
                        label="",
                        default_value=float(value) if value is not None else 0.0,
                        min_value=float(mn),
                        max_value=float(mx),
                        format=f"%.{self._layout.value_precision}f",
                        callback=self._on_widget_change,
                        user_data=desc.id,
                    )
                    self._set_full_width(slider_id)
                with dpg.table_cell():
                    self._add_cc_binding_input(None, desc)
        return slider_id

    def _create_style_color_picker(
        self,
        parent: int | str,
        desc: ParameterDescriptor,
        value: Any,
    ) -> int | str:
        """style.color 用のカラー編集ウィジェットを生成し、そのタグを返す。"""
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

    def _is_style_descriptor(self, desc: ParameterDescriptor) -> bool:
        """Descriptor がレイヤー用かどうかを判定する。"""
        if not isinstance(desc.id, str):
            return False
        if desc.id.startswith("layer."):
            return desc.id.endswith(".color") or desc.id.endswith(".thickness")
        return False

    def _is_style_color_desc(self, desc: ParameterDescriptor) -> bool:
        """Descriptor が style.color パラメータかどうかを判定する。"""
        return self._is_style_descriptor(desc) and str(desc.id).endswith(".color")

    def _current_or_default(self, desc: ParameterDescriptor) -> Any:
        """Store の現在値があればそれを、なければ Descriptor の既定値を返す。"""
        v = self._store.current_value(desc.id)
        return v if v is not None else desc.default_value

    def _on_enum_filter_change(self, sender: int | str, app_data: Any, user_data: Any) -> None:
        """検索入力の変更に応じてコンボの候補をフィルターする。"""
        if self._syncing:
            return
        param_id = str(user_data)
        state = self._searchable_enum_state.get(param_id)
        if not state:
            return
        base_items = list(state.get("items", []))
        combo_id = state.get("combo_id", param_id)
        text = str(app_data or "")
        norm = text.lower()
        filtered = (
            [item for item in base_items if norm in str(item).lower()] if norm else list(base_items)
        )
        try:
            current = dpg.get_value(combo_id)
        except Exception:
            current = None
        if current:
            filtered.insert(0, str(current))
        filtered = self._dedup_items(filtered)
        try:
            dpg.configure_item(combo_id, items=filtered)
        except Exception:
            return

    def _on_widget_change(self, sender: int, app_data: Any, user_data: Any) -> None:  # noqa: D401
        """ウィジェット変更イベントから Store の override を更新する。"""
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

    def _on_minmax_change(self, sender: int, app_data: Any, user_data: Any) -> None:
        """min/max 入力変更からスライダーの表示レンジを更新する。"""
        if self._syncing:
            return
        try:
            pid, kind = user_data
            param_id = str(pid)
            kind_str = str(kind)
        except Exception:
            return
        try:
            desc = self._store.get_descriptor(param_id)
        except Exception:
            return
        base_min, base_max = self._base_range(desc)
        try:
            current_override = self._store.range_override(param_id)
        except Exception:
            current_override = None
        if current_override is None:
            cur_min, cur_max = base_min, base_max
        else:
            try:
                cur_min = float(current_override[0])
                cur_max = float(current_override[1])
            except Exception:
                cur_min, cur_max = base_min, base_max
        try:
            new_val = float(app_data)
        except Exception:
            # 入力が数値でない場合は元の値に戻す
            self._syncing = True
            try:
                if kind_str == "min":
                    dpg.set_value(sender, cur_min)
                else:
                    dpg.set_value(sender, cur_max)
            finally:
                self._syncing = False
            return

        if kind_str == "min":
            new_min, new_max = new_val, cur_max
        else:
            new_min, new_max = cur_min, new_val

        # min >= max は採用しない（UI を元に戻す）
        if not new_min < new_max:
            self._syncing = True
            try:
                dpg.set_value(sender, cur_min if kind_str == "min" else cur_max)
            finally:
                self._syncing = False
            return

        # Store にオーバーライドを保存し、関連スライダーのレンジを更新
        self._store.set_range_override(param_id, new_min, new_max)
        eff_min, eff_max = new_min, new_max

        self._syncing = True
        try:
            # 入力ボックスの値を正規化
            try:
                tag_min = f"{param_id}::min"
                tag_max = f"{param_id}::max"
                dpg.set_value(tag_min, eff_min)
                dpg.set_value(tag_max, eff_max)
            except Exception:
                pass

            # 対応するスライダー（scalar/vector）の min/max を更新
            if desc.value_type == "vector":
                for suffix in ("x", "y", "z", "w"):
                    tag = f"{param_id}::{suffix}"
                    if not dpg.does_item_exist(tag):
                        continue
                    try:
                        dpg.configure_item(tag, min_value=float(eff_min), max_value=float(eff_max))
                    except Exception:
                        continue
            elif desc.value_type in {"float", "int"}:
                if dpg.does_item_exist(param_id):
                    try:
                        dpg.configure_item(
                            param_id,
                            min_value=float(eff_min),
                            max_value=float(eff_max),
                        )
                    except Exception:
                        pass
        finally:
            self._syncing = False

    def on_store_change(self, ids: Iterable[str]) -> None:
        """Store から通知された ID 群に対応する DPG ウィジェット値を更新する。"""
        palette_dirty = False
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
                    elif isinstance(pid, str) and (
                        (pid.endswith(".color") and ".style#" in pid)
                        or (pid.startswith("layer.") and pid.endswith(".color"))
                    ):
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
                        if value is None:
                            continue
                        dpg.set_value(pid, value)
                    if isinstance(pid, str) and pid.startswith("palette."):
                        palette_dirty = True
                    continue
                try:
                    desc = self._store.get_descriptor(pid)
                except KeyError:
                    logger.debug("store notified unknown id (no descriptor): %s", pid)
                    continue
                if desc.value_type != "vector":
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
        if palette_dirty:
            try:
                self._refresh_palette_preview()
            except Exception:
                logger.debug("palette preview refresh failed", exc_info=True)

    def _on_cc_binding_change(self, sender: int, app_data: Any, user_data: Any) -> None:
        """CC 番号入力の変更を CC バインディングへ反映する。

        - 空文字列: バインディング解除（Store から削除し、UI は現状維持）。
        - 数値以外/パース失敗: バインディング解除し、対応する入力テキストを空文字に戻す。
        - 数値: int(float(...)) で解釈し、0..127 にクランプしてバインドする。
        """
        try:
            pid = str(user_data)
            raw_text = str(app_data)
        except Exception:
            return
        text = raw_text.strip()
        # 空文字列はバインディング解除のみ（UI の表示は維持）
        if not text:
            self._store.bind_cc(pid, None)
            return
        parsed = self._parse_cc_index(text)
        if parsed is None:
            # パース失敗時はバインド解除し、入力を空に戻す
            self._store.bind_cc(pid, None)
            dpg.set_value(f"{pid}::cc", "")
            return
        self._store.bind_cc(pid, parsed)
        try:
            dpg.set_value(f"{pid}::cc", str(parsed))
        except Exception:
            pass

    def _add_cc_binding_input(
        self, parent: int | str | None, desc: ParameterDescriptor
    ) -> int | str:
        """スカラーパラメータ用の CC 番号入力テキストを追加する。"""
        default_text = self._cc_binding_text(desc.id)
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
        # セル幅に素直にフィットさせる（固定ピクセル幅は使わない）
        self._set_full_width(box)
        return box

    def _add_cc_binding_input_component(
        self,
        parent: int | str | None,
        desc: ParameterDescriptor,
        idx: int,
    ) -> int | str:
        """ベクトル成分ごとの CC 番号入力テキストを追加する。"""
        suffix = ("x", "y", "z", "w")[idx]
        pid_comp = f"{desc.id}::{suffix}"
        default_text = self._cc_binding_text(pid_comp)
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
        # セル幅に素直にフィットさせる（固定ピクセル幅は使わない）
        self._set_full_width(box)
        return box

    def _cc_binding_text(self, param_id: str) -> str:
        """現在の CC バインディングから入力テキスト用の文字列を返す。"""
        try:
            current = self._store.cc_binding(param_id)
        except Exception:
            return ""
        return "" if current is None else str(int(current))

    def force_set_rgb_u8(self, tag: int | str, rgb_u8: Sequence[int]) -> None:
        """0–255 RGB 値を DPG カラーピッカーに設定する。"""
        if not isinstance(rgb_u8, Sequence) or len(rgb_u8) < 3:
            raise ValueError("rgb_u8 must be a sequence of length >= 3")
        r = int(rgb_u8[0])
        g = int(rgb_u8[1])
        b = int(rgb_u8[2])
        dpg.set_value(tag, [r, g, b])

    def store_rgb01(self, pid: str, app_data: Any) -> None:
        """DPG カラー入力を 0..1 RGBA（または vec3）に正規化して Store に保存する。

        - 一般パラメータ: RGBA (0..1) のタプルとして保存する。
        - style.color: HUD と同様の 0–255 相当入力を前提とし、vec3 (RGB 0..1) として保存する。
        - 正規化に失敗した場合は Store を更新せず、例外ログを出力する。
        """
        try:
            from util.color import normalize_color as _norm

            rgba = _norm(app_data)
            if isinstance(pid, str) and self._is_layer_style_color_id(pid):
                # style/layer の color は vec3 保存（HUD など他の GUI と色の扱いを揃える）
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
        """カラー値を RGBA (0..1) に正規化し、失敗時は既定値を返す。"""
        try:
            from util.color import normalize_color as _norm

            r, g, b, a = _norm(value)
            return float(r), float(g), float(b), float(a)
        except Exception:
            return default

    def _dpg_policy(self, names: Sequence[str]) -> int:
        """Dear PyGui テーブル policy 定数を候補名から解決し、安全な int に変換して返す。"""
        for n in names:
            var = getattr(dpg, n, None)
            if var is None:
                continue
            try:
                return int(var)
            except Exception:
                continue
        return 0

    @staticmethod
    def _dedup_items(items: Sequence[str]) -> list[str]:
        """順序を保ったまま重複を取り除く。"""
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _set_full_width(self, item_id: int | str) -> None:
        """ウィジェット幅をセル幅いっぱいに広げる（失敗時は既定幅のまま）。"""
        try:
            dpg.set_item_width(item_id, -1)
        except Exception:
            # 古い DPG バージョンや一部ウィジェットでは width 指定が無視されるため、そのまま続行する。
            pass

    def _apply_cell_padding_theme(self, table_id: int | str) -> None:
        """セルパディング付きテーブルテーマを適用する（Dear PyGui の有無に応じて安全に処理）。"""
        var_cell_padding = getattr(dpg, "mvStyleVar_CellPadding", None)
        if var_cell_padding is None:
            return
        cx = int(getattr(self._layout, "cell_padding_x", self._layout.padding))
        cy = int(getattr(self._layout, "cell_padding_y", self._layout.padding))
        with dpg.theme() as tbl_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(var_cell_padding, cx, cy)
        dpg.bind_item_theme(table_id, tbl_theme)

    def _resolve_style_color_from_store(
        self,
        pid: str,
        fallback: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Store の現在値/元値からスタイル用カラーを解決し、RGBA (0..1) で返す。"""
        value = self._store.current_value(pid) or self._store.original_value(pid)
        if value is None:
            return fallback
        return self._safe_norm(value, fallback)

    def _refresh_palette_preview(self) -> None:
        """現在の Store 値から Palette を再計算し、プレビュー/共有状態を更新する。"""
        if self._palette_swatches_container is None:
            return
        try:
            from engine.ui.palette.helpers import build_palette_from_values  # type: ignore[import]
            from palette.ui_helpers import (  # type: ignore[import]
                ExportFormat,
                export_palette,
            )
            from util.palette_state import set_palette as _set_palette  # type: ignore[import]
        except Exception:
            return

        try:
            L_val = self._store.current_value(PALETTE_L_ID) or self._store.original_value(
                PALETTE_L_ID
            )
        except Exception:
            L_val = None
        try:
            C_val = self._store.current_value(PALETTE_C_ID) or self._store.original_value(
                PALETTE_C_ID
            )
        except Exception:
            C_val = None
        try:
            h_val = self._store.current_value(PALETTE_H_ID) or self._store.original_value(
                PALETTE_H_ID
            )
        except Exception:
            h_val = None
        try:
            type_val = self._store.current_value(PALETTE_TYPE_ID) or self._store.original_value(
                PALETTE_TYPE_ID
            )
        except Exception:
            type_val = None
        try:
            style_val = self._store.current_value(PALETTE_STYLE_ID) or self._store.original_value(
                PALETTE_STYLE_ID
            )
        except Exception:
            style_val = None
        try:
            n_val = self._store.current_value(PALETTE_N_COLORS_ID) or self._store.original_value(
                PALETTE_N_COLORS_ID
            )
        except Exception:
            n_val = None

        palette_obj = None
        try:
            palette_obj = build_palette_from_values(
                base_color_value=None,
                palette_type_value=type_val,
                palette_style_value=style_val,
                n_colors_value=n_val,
                L_value=L_val,
                C_value=C_val,
                h_value=h_val,
            )
        except Exception:
            palette_obj = None

        try:
            _set_palette(palette_obj)
        except Exception:
            pass

        # 既存スウォッチをクリア
        try:
            if dpg.does_item_exist(self._palette_swatches_container):
                children = dpg.get_item_children(self._palette_swatches_container, 1)
                if isinstance(children, (list, tuple)):
                    for cid in list(children):
                        try:
                            dpg.delete_item(cid)
                        except Exception:
                            continue
        except Exception:
            logger.debug("failed to clear palette swatches", exc_info=True)

        if palette_obj is None:
            return

        # HEX と sRGB(0..1) の両方を取得
        try:
            hex_list = export_palette(palette_obj, ExportFormat.HEX)
            srgb_list = export_palette(palette_obj, ExportFormat.SRGB_01)
        except Exception:
            return

        # スウォッチを横に並べる
        for idx, (hex_str, rgb) in enumerate(zip(hex_list, srgb_list)):
            try:
                r, g, b = rgb  # type: ignore[misc]
                rv = int(round(float(r) * 255.0))
                gv = int(round(float(g) * 255.0))
                bv = int(round(float(b) * 255.0))
            except Exception:
                logger.debug("refresh_palette_preview: failed to decode rgb for swatch %d", idx)
                continue
            try:
                size = int(getattr(self._layout, "row_height", 28))
                width = size * 2
                height = max(1, size // 2)
                dpg.add_color_button(
                    parent=self._palette_swatches_container,
                    default_value=[rv, gv, bv, 255],
                    no_border=True,
                    width=width,
                    height=height,
                    callback=lambda *_, h=hex_str: self._on_palette_swatch_click(h),
                )
            except Exception:
                logger.debug(
                    "refresh_palette_preview: failed to add color button for %s",
                    hex_str,
                    exc_info=True,
                )
                continue

    def _on_palette_swatch_click(self, hex_str: str) -> None:
        """スウォッチクリックで HEX をクリップボードへコピーする。"""
        try:
            set_clip = getattr(dpg, "set_clipboard_text", None)
            if callable(set_clip):
                set_clip(str(hex_str))
        except Exception:
            logger.debug("failed to copy palette swatch HEX to clipboard", exc_info=True)

    def _create_vector_sliders(
        self,
        parent: int | str,
        desc: ParameterDescriptor,
        value_vec: Sequence[float],
        vmin: Sequence[float],
        vmax: Sequence[float],
    ) -> int | str:
        """ベクトル値用の水平スライダ群を生成する。"""
        vec = list(value_vec)
        dim = max(2, min(len(vec), 4))
        last_slider: int | str = 0
        with dpg.table(
            parent=parent,
            header_row=False,
            policy=self._dpg_policy(["mvTable_SizingStretchSame"]),
        ) as vec_tbl:
            self._apply_cell_padding_theme(vec_tbl)
            for _ in range(dim):
                self._add_stretch_column("", 1.0)
            with dpg.table_row():
                for i, suffix in enumerate(("x", "y", "z", "w")[:dim]):
                    with dpg.table_cell():
                        tag = f"{desc.id}::{suffix}"
                        default_component = float(vec[i]) if i < len(vec) else 0.0
                        min_val = float(vmin[i]) if i < len(vmin) else 0.0
                        max_val = float(vmax[i]) if i < len(vmax) else 1.0
                        last_slider = dpg.add_slider_float(
                            tag=tag,
                            label="",
                            default_value=default_component,
                            min_value=min_val,
                            max_value=max_val,
                            format=f"%.{self._layout.value_precision}f",
                            callback=self._on_widget_change,
                            user_data=(desc.id, i),
                        )
                        self._set_full_width(last_slider)
        return last_slider

    def _is_layer_style_color_id(self, pid: str) -> bool:
        """style/layer 系 color パラメータの ID かどうかを判定する。"""
        return (pid.endswith(".color") and ".style#" in pid) or (
            pid.startswith("layer.") and pid.endswith(".color")
        )

    def _parse_cc_index(self, text: str) -> int | None:
        """CC 入力文字列を 0..127 の整数にパースし、失敗時は None を返す。"""
        try:
            value = int(float(text))
        except Exception:
            return None
        if value < 0:
            return 0
        if value > 127:
            return 127
        return value
