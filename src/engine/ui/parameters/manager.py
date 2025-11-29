"""
どこで: `engine.ui.parameters` の統合ヘルパ層。
何を: `user_draw` をラップし、ParameterRuntime の有効化/初回トレース/GUI ウィンドウ起動・寿命管理を担う。
なぜ: 既存の描画関数に最小介入でパラメータランタイム/GUI を組み込むため。
"""

from __future__ import annotations

from typing import Callable, Sequence

from engine.core.geometry import Geometry
from engine.core.lazy_geometry import LazyGeometry
from engine.render.types import Layer
from util.utils import load_config

from .controller import ParameterWindowController
from .persistence import load_overrides, save_overrides
from .runtime import ParameterRuntime, activate_runtime, deactivate_runtime
from .state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
    ParameterThemeConfig,
    ParameterWindowConfig,
    RangeHint,
)


class ParameterManager:
    """`user_draw` をラップして ParameterRuntime を介在させる内部ヘルパー。"""

    def __init__(
        self,
        user_draw: Callable[
            [float],
            Geometry | LazyGeometry | Layer | Sequence[Geometry | LazyGeometry | Layer],
        ],
        *,
        layout: ParameterLayoutConfig | None = None,
        lazy_trace: bool = True,
        enable_palette_gui: bool = True,
    ) -> None:
        self._user_draw = user_draw
        self._enable_palette_gui = bool(enable_palette_gui)
        self.store = ParameterStore()
        # CC プロバイダを util 層のフック経由で注入（engine→api の依存を避ける）。
        try:
            from util.cc_provider import get_cc_snapshot as _get_cc_snapshot

            self.store.set_cc_provider(_get_cc_snapshot)
        except Exception:
            pass

        # 設定ファイルから Parameter GUI 見た目設定を読み込み（未指定時は既定）
        cfg = load_config() or {}
        pg_cfg = cfg.get("parameter_gui", {}) if isinstance(cfg.get("parameter_gui"), dict) else {}

        # layout は引数優先。未指定なら設定から生成し、それも無ければ既定値。
        if layout is None:
            lcfg = pg_cfg.get("layout", {}) if isinstance(pg_cfg.get("layout"), dict) else {}
            # 比率は 0.1..0.9 にクランプ
            try:
                ratio_val = float(lcfg.get("label_column_ratio", 0.5))
            except Exception:
                ratio_val = 0.5
            ratio_val = max(0.1, min(0.9, ratio_val))
            try:
                bars_cc = float(lcfg.get("bars_cc_ratio", 0.7))
            except Exception:
                bars_cc = 0.7
            bars_cc = 0.05 if bars_cc < 0.05 else (0.95 if bars_cc > 0.95 else bars_cc)
            try:
                ccw = int(lcfg.get("cc_box_width", 24))
            except Exception:
                ccw = 24
            ccw = 8 if ccw < 8 else ccw

            # 詳細余白（配列 [x, y]）は未指定時 padding にフォールバック
            def _pair(key: str, default_x: int, default_y: int) -> tuple[int, int]:
                raw = lcfg.get(key)
                if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                    try:
                        x = int(raw[0])
                        y = int(raw[1])
                        return x, y
                    except Exception:
                        return default_x, default_y
                return default_x, default_y

            pad = int(lcfg.get("padding", 8))
            win_x, win_y = _pair("window_padding", pad, pad)
            frm_x, frm_y = _pair("frame_padding", pad, max(1, pad // 2))
            itm_x, itm_y = _pair("item_spacing", pad, max(1, pad // 2))
            cel_x, cel_y = _pair("cell_padding", pad, pad)
            layout = ParameterLayoutConfig(
                row_height=int(lcfg.get("row_height", 28)),
                padding=pad,
                font_size=int(lcfg.get("font_size", 12)),
                value_precision=int(lcfg.get("value_precision", 6)),
                label_column_ratio=ratio_val,
                bars_cc_ratio=bars_cc,
                cc_box_width=ccw,
                window_padding_x=win_x,
                window_padding_y=win_y,
                frame_padding_x=frm_x,
                frame_padding_y=frm_y,
                item_spacing_x=itm_x,
                item_spacing_y=itm_y,
                cell_padding_x=cel_x,
                cell_padding_y=cel_y,
            )

        # window config
        wcfg_raw = pg_cfg.get("window", {}) if isinstance(pg_cfg.get("window"), dict) else {}
        window_cfg = ParameterWindowConfig(
            width=int(wcfg_raw.get("width", 420)),
            height=int(wcfg_raw.get("height", 640)),
            title=str(wcfg_raw.get("title", "Parameters")),
        )

        # theme config（自由形式のディクショナリ）
        tcfg_raw = pg_cfg.get("theme", {}) if isinstance(pg_cfg.get("theme"), dict) else {}
        theme_cfg = ParameterThemeConfig(
            style=dict(
                tcfg_raw.get("style", {}) if isinstance(tcfg_raw.get("style"), dict) else {}
            ),
            colors=dict(
                tcfg_raw.get("colors", {}) if isinstance(tcfg_raw.get("colors"), dict) else {}
            ),
            categories=dict(
                tcfg_raw.get("categories", {})
                if isinstance(tcfg_raw.get("categories"), dict)
                else {}
            ),
        )

        self.runtime = ParameterRuntime(self.store, layout=layout)
        self.runtime.set_lazy(lazy_trace)
        self.controller = ParameterWindowController(
            self.store, layout=layout, window_cfg=window_cfg, theme_cfg=theme_cfg
        )
        self._initialized = False
        self._layer_keys: list[str] = []

    def initialize(self) -> None:
        if self._initialized:
            return
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        initial_output = None
        try:
            # 初回は t=0 のみ登録（cc は api.cc 側に閉じる）
            self.runtime.set_inputs(0.0)
            initial_output = self._user_draw(0.0)
        finally:
            deactivate_runtime()
        # 追加のランナー系パラメータ（色）を登録してから override を復元
        try:
            from util.color import normalize_color as _norm
            from util.utils import load_config as _load_cfg

            cfg = _load_cfg() or {}
            canvas = cfg.get("canvas", {}) if isinstance(cfg, dict) else {}
            bg = _norm(canvas.get("background_color", (1.0, 1.0, 1.0, 1.0)))
            ln = _norm(canvas.get("line_color", (0.0, 0.0, 0.0, 1.0)))
            bg_desc = ParameterDescriptor(
                id="runner.background",
                label="Background",
                source="effect",
                category="Style",
                category_kind="style",
                value_type="vector",
                default_value=(float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3])),
            )
            ln_desc = ParameterDescriptor(
                id="runner.line_color",
                label="Line Color",
                source="effect",
                category="Style",
                category_kind="style",
                value_type="vector",
                default_value=(float(ln[0]), float(ln[1]), float(ln[2]), float(ln[3])),
            )
            thickness_desc = ParameterDescriptor(
                id="runner.line_thickness",
                label="Global Thickness",
                source="effect",
                category="Style",
                category_kind="style",
                value_type="float",
                default_value=0.0006,
                range_hint=RangeHint(0.0001, 0.005, step=1e-6),
            )
            self.store.register(bg_desc, bg_desc.default_value)
            self.store.register(ln_desc, ln_desc.default_value)
            self.store.register(thickness_desc, thickness_desc.default_value)

            # HUD colors (text/meter) defaults from config
            hud_cfg = cfg.get("hud", {}) if isinstance(cfg, dict) else {}
            # text color may be RGBA or hex
            hud_text = _norm(hud_cfg.get("text_color", (0, 0, 0, 155)))
            meters = hud_cfg.get("meters", {}) if isinstance(hud_cfg, dict) else {}
            mfg = meters.get("meter_color_fg", (50, 50, 50))
            try:
                mr = float(mfg[0]) / 255.0
                mg = float(mfg[1]) / 255.0
                mb = float(mfg[2]) / 255.0
            except Exception:
                mr, mg, mb = 0.2, 0.2, 0.2
            hud_text_desc = ParameterDescriptor(
                id="runner.hud_text_color",
                label="HUD Text",
                source="effect",
                category="Style",
                category_kind="style",
                value_type="vector",
                default_value=(
                    float(hud_text[0]),
                    float(hud_text[1]),
                    float(hud_text[2]),
                    float(hud_text[3]),
                ),
            )
            hud_meter_desc = ParameterDescriptor(
                id="runner.hud_meter_color",
                label="HUD Meter",
                source="effect",
                category="Style",
                category_kind="style",
                value_type="vector",
                default_value=(float(mr), float(mg), float(mb), 1.0),
            )
            hud_meter_bg_desc = ParameterDescriptor(
                id="runner.hud_meter_bg_color",
                label="HUD Meter BG",
                source="effect",
                category="Style",
                category_kind="style",
                value_type="vector",
                default_value=(0.196, 0.196, 0.196, 1.0),
            )
            self.store.register(hud_text_desc, hud_text_desc.default_value)
            self.store.register(hud_meter_desc, hud_meter_desc.default_value)
            self.store.register(hud_meter_bg_desc, hud_meter_bg_desc.default_value)
            # HUD 表示トグル（Parameter GUI から操作）。既定は True（後で API 側の解決結果で上書き）。
            show_hud_desc = ParameterDescriptor(
                id="runner.show_hud",
                label="Show HUD",
                source="effect",
                category="Style",
                category_kind="style",
                value_type="bool",
                default_value=True,
            )
            self.store.register(show_hud_desc, show_hud_desc.default_value)
        except Exception:
            pass
        # パレット関連パラメータ（Parameter GUI 用）
        try:
            if self._enable_palette_gui:
                self._register_palette_descriptors()
        except Exception:
            pass
        # レイヤー構成の登録（変動しない前提）
        try:
            if initial_output is not None:
                self._register_layer_descriptors(initial_output)
        except Exception:
            pass
        # ここで Descriptor が確定しているため、GUI マウント前に override を復元
        try:
            load_overrides(self.store)
        except Exception:
            pass
        descriptors = self.store.descriptors()
        if descriptors:
            self.controller.start()
        else:
            self.controller.set_visibility(False)
        self._initialized = True

    def draw(
        self, t: float
    ) -> Geometry | LazyGeometry | Layer | Sequence[Geometry | LazyGeometry | Layer]:
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        try:
            # 現在の CC は api.cc に閉じる（GUI は Store 経由）
            self.runtime.set_inputs(t)
            out = self._user_draw(t)
            return self._apply_layer_overrides(out)
        finally:
            deactivate_runtime()

    def shutdown(self) -> None:
        # 終了時に override を保存（フェイルソフト）
        try:
            save_overrides(self.store)
        except Exception:
            pass
        self.controller.shutdown()

    # ---- layer helpers -------------------------------------------------
    def _iter_layers(self, result) -> list[tuple[str, str, Layer]]:
        """draw の戻り値からレイヤー一覧を抽出する。"""
        layers: list[Layer] = []
        if isinstance(result, Layer):
            layers.append(
                Layer(
                    geometry=result.geometry,
                    color=result.color,
                    thickness=result.thickness,
                    name=getattr(result, "name", None),
                    meta=getattr(result, "meta", None),
                )
            )
        elif isinstance(result, (list, tuple)):
            for it in result:
                if isinstance(it, Layer):
                    layers.append(
                        Layer(
                            geometry=it.geometry,
                            color=it.color,
                            thickness=it.thickness,
                            name=getattr(it, "name", None),
                            meta=getattr(it, "meta", None),
                        )
                    )
        # key/label（worker._apply_layer_overrides と揃え、重複名に suffix を付与）
        out: list[tuple[str, str, Layer]] = []
        used: set[str] = set()
        for idx, layer in enumerate(layers):
            name = getattr(layer, "name", None)
            base = name if isinstance(name, str) and name else f"layer{idx}"
            key = base if base not in used else f"{base}_{idx}"
            used.add(key)
            label = name if isinstance(name, str) and name else f"Layer {idx + 1}"
            out.append((key, label, layer))
        return out

    def _register_layer_descriptors(self, result) -> None:
        """レイヤー色/太さを GUI 用に登録（初期フレームのみ想定）。"""
        entries = self._iter_layers(result)
        if not entries:
            return
        # 既定色は runner.line_color の original を使用（無ければ黒）
        try:
            base_col = self.store.original_value("runner.line_color")
        except Exception:
            base_col = None
        if base_col is None:
            base_col = (0.0, 0.0, 0.0, 1.0)
        try:
            from util.color import normalize_color as _norm

            base_col = _norm(base_col)
        except Exception:
            base_col = (0.0, 0.0, 0.0, 1.0)

        for key, label, layer in entries:
            if key not in self._layer_keys:
                self._layer_keys.append(key)
            # color
            try:
                col = layer.color if layer.color is not None else base_col
                c_desc = ParameterDescriptor(
                    id=f"layer.{key}.color",
                    label=f"{label} Color",
                    source="effect",
                    category="Style",
                    category_kind="style",
                    value_type="vector",
                    default_value=tuple(col) if isinstance(col, tuple) else tuple(base_col),
                )
                self.store.register(c_desc, c_desc.default_value)
            except Exception:
                pass
            # thickness
            try:
                # レイヤー指定が無ければグローバル既定（runner.line_thickness）をフォールバック
                th_default = layer.thickness
                if th_default is None:
                    th_base = self.store.original_value("runner.line_thickness")
                    th_default = float(th_base) if th_base is not None else 0.0006
                t_desc = ParameterDescriptor(
                    id=f"layer.{key}.thickness",
                    label=f"{label} Thickness",
                    source="effect",
                    category="Style",
                    category_kind="style",
                    value_type="float",
                    default_value=float(th_default),
                    range_hint=RangeHint(0.0001, 0.005, step=1e-6),
                )
                self.store.register(t_desc, t_desc.default_value)
            except Exception:
                pass

    def _register_palette_descriptors(self) -> None:
        """パレット GUI 用の Descriptor を登録する。"""
        try:
            from util.color import normalize_color as _norm  # type: ignore[import]
        except Exception:
            return

        # ベースカラーは runner.line_color の original を既定とする。
        try:
            base_raw = self.store.original_value("runner.line_color")
        except Exception:
            base_raw = None
        if base_raw is None:
            base_raw = (0.0, 0.0, 0.0, 1.0)
        try:
            r, g, b, a = _norm(base_raw)
            base_rgba = (float(r), float(g), float(b), float(a))
        except Exception:
            base_rgba = (0.0, 0.0, 0.0, 1.0)

        # RGBA から OKLCH を推定して L/C/h の既定値にする。
        try:
            from palette.engine import DefaultColorEngine  # type: ignore[import]

            eng = DefaultColorEngine()
            L0, C0, h0 = eng.srgb_to_oklch(base_rgba[0], base_rgba[1], base_rgba[2])
        except Exception:
            L0, C0, h0 = 60.0, 0.1, 0.0

        # 軽いクランプ
        if L0 < 0.0:
            L0 = 0.0
        if L0 > 100.0:
            L0 = 100.0
        if C0 < 0.0:
            C0 = 0.0
        if C0 > 0.4:
            C0 = 0.4
        h0 = float(h0 % 360.0)

        l_desc = ParameterDescriptor(
            id="palette.L",
            label="Lightness",
            source="effect",
            category="Palette",
            category_kind="palette",
            value_type="float",
            default_value=float(L0),
            range_hint=RangeHint(0.0, 100.0, step=0.1),
        )
        c_desc = ParameterDescriptor(
            id="palette.C",
            label="Chroma",
            source="effect",
            category="Palette",
            category_kind="palette",
            value_type="float",
            default_value=float(C0),
            range_hint=RangeHint(0.0, 0.4, step=0.01),
        )
        h_desc = ParameterDescriptor(
            id="palette.h",
            label="Hue",
            source="effect",
            category="Palette",
            category_kind="palette",
            value_type="float",
            default_value=float(h0),
            range_hint=RangeHint(0.0, 360.0, step=1.0),
        )
        self.store.register(l_desc, l_desc.default_value)
        self.store.register(c_desc, c_desc.default_value)
        self.store.register(h_desc, h_desc.default_value)

        # GUI 表示用の省略ラベル（ANA/COM/...）
        type_labels = ["ANA", "COM", "SPL", "TRI", "TET", "TAS"]
        # GUI 表示用のスタイルラベル（記号付き）
        style_labels = ["Square", "Triangle", "Circle", "Diamond"]

        if type_labels:
            type_desc = ParameterDescriptor(
                id="palette.type",
                label="Palette Type",
                source="effect",
                category="Palette",
                category_kind="palette",
                value_type="enum",
                default_value=type_labels[0],
                choices=type_labels,
            )
            self.store.register(type_desc, type_desc.default_value)

        if style_labels:
            style_desc = ParameterDescriptor(
                id="palette.style",
                label="Palette Style",
                source="effect",
                category="Palette",
                category_kind="palette",
                value_type="enum",
                default_value=style_labels[0],
                choices=style_labels,
            )
            self.store.register(style_desc, style_desc.default_value)

        n_desc = ParameterDescriptor(
            id="palette.n_colors",
            label="Colors",
            source="effect",
            category="Palette",
            category_kind="palette",
            value_type="int",
            default_value=4,
            range_hint=RangeHint(2, 6, step=1),
        )
        self.store.register(n_desc, n_desc.default_value)

        # palette 自動適用モード（背景固定で線/レイヤー色のみ反映）
        auto_choices = ["off", "bg_global_and_layers"]
        auto_desc = ParameterDescriptor(
            id="palette.auto_apply_mode",
            label="Apply palette to colors",
            source="effect",
            category="Palette",
            category_kind="palette",
            value_type="enum",
            default_value="bg_global_and_layers",
            choices=auto_choices,
        )
        self.store.register(auto_desc, auto_desc.default_value)

    def _apply_layer_overrides(self, result):
        """Store の override をレイヤーに適用する。"""
        entries = self._iter_layers(result)
        if not entries:
            return result
        patched: list[Layer] = []
        for key, _label, layer in entries:
            color_pid = f"layer.{key}.color"
            thick_pid = f"layer.{key}.thickness"
            color_val = self.store.current_value(color_pid)
            if color_val is None:
                color_val = self.store.original_value(color_pid)
            thickness_val = self.store.current_value(thick_pid)
            if thickness_val is None:
                thickness_val = self.store.original_value(thick_pid)
            patched.append(
                Layer(
                    geometry=layer.geometry,
                    color=color_val if color_val is not None else layer.color,
                    thickness=(
                        float(thickness_val) if thickness_val is not None else layer.thickness
                    ),
                    name=layer.name,
                    meta=layer.meta,
                )
            )
        return tuple(patched)
