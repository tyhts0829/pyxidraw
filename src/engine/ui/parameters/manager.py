"""
どこで: `engine.ui.parameters` の統合ヘルパ層。
何を: `user_draw` をラップし、ParameterRuntime の有効化/初回トレース/GUI ウィンドウ起動・寿命管理を担う。
なぜ: 既存の描画関数に最小介入でパラメータランタイム/GUI を組み込むため。
"""

from __future__ import annotations

from typing import Callable

from engine.core.geometry import Geometry
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
)


class ParameterManager:
    """`user_draw` をラップして ParameterRuntime を介在させる内部ヘルパー。"""

    def __init__(
        self,
        user_draw: Callable[[float], Geometry],
        *,
        layout: ParameterLayoutConfig | None = None,
        lazy_trace: bool = True,
    ) -> None:
        self._user_draw = user_draw
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
        )

        self.runtime = ParameterRuntime(self.store, layout=layout)
        self.runtime.set_lazy(lazy_trace)
        self.controller = ParameterWindowController(
            self.store, layout=layout, window_cfg=window_cfg, theme_cfg=theme_cfg
        )
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        try:
            # 初回は t=0 のみ登録（cc は api.cc 側に閉じる）
            self.runtime.set_inputs(0.0)
            self._user_draw(0.0)
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
                category="Display",
                value_type="vector",
                default_value=(float(bg[0]), float(bg[1]), float(bg[2]), float(bg[3])),
            )
            ln_desc = ParameterDescriptor(
                id="runner.line_color",
                label="Line Color",
                source="effect",
                category="Display",
                value_type="vector",
                default_value=(float(ln[0]), float(ln[1]), float(ln[2]), float(ln[3])),
            )
            self.store.register(bg_desc, bg_desc.default_value)
            self.store.register(ln_desc, ln_desc.default_value)

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
                category="HUD",
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
                category="HUD",
                value_type="vector",
                default_value=(float(mr), float(mg), float(mb), 1.0),
            )
            hud_meter_bg_desc = ParameterDescriptor(
                id="runner.hud_meter_bg_color",
                label="HUD Meter BG",
                source="effect",
                category="HUD",
                value_type="vector",
                default_value=(0.196, 0.196, 0.196, 1.0),
            )
            self.store.register(hud_text_desc, hud_text_desc.default_value)
            self.store.register(hud_meter_desc, hud_meter_desc.default_value)
            self.store.register(hud_meter_bg_desc, hud_meter_bg_desc.default_value)
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

    def draw(self, t: float) -> Geometry:
        activate_runtime(self.runtime)
        self.runtime.begin_frame()
        try:
            # 現在の CC は api.cc に閉じる（GUI は Store 経由）
            self.runtime.set_inputs(t)
            return self._user_draw(t)
        finally:
            deactivate_runtime()

    def shutdown(self) -> None:
        # 終了時に override を保存（フェイルソフト）
        try:
            save_overrides(self.store)
        except Exception:
            pass
        self.controller.shutdown()
