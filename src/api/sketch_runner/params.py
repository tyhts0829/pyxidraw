"""
どこで: `api.sketch_runner.params`
何を: Parameter GUI 連携（初期色適用、色変更の監視、スナップショット抽出）。
なぜ: `api.sketch` を薄く保ち、UIスレッド適用と Worker スナップショット生成を分離するため。
"""

from __future__ import annotations

from typing import Callable, Mapping


def make_param_snapshot_fn(
    parameter_manager, cc_snapshot_fn: Callable[[], Mapping[int, float]] | None
):
    """ParameterStore の override を抽出するスナップショット関数を生成する。"""

    if parameter_manager is None:
        return lambda: None

    # 遅延 import（トップに依存を増やさない）
    from engine.ui.parameters.snapshot import extract_overrides

    def _snapshot():  # noqa: ANN202 - 実行時に None/Mapping を返す
        try:
            cc_map = cc_snapshot_fn() if callable(cc_snapshot_fn) else None
            return extract_overrides(parameter_manager.store, cc_map)
        except Exception:
            return None

    return _snapshot


def apply_initial_colors(parameter_manager, rendering_window, line_renderer, overlay=None) -> None:
    """ParameterStore に保存された初期値を背景/線/HUDへ適用する。"""

    if parameter_manager is None:
        return
    from util.color import normalize_color as _norm

    # 背景（store → window）
    try:
        bg_val = parameter_manager.store.current_value("runner.background")
        if bg_val is None:
            bg_val = parameter_manager.store.original_value("runner.background")
        if bg_val is not None:
            rendering_window.set_background_color(_norm(bg_val))
    except Exception:
        pass

    # 線色（store → renderer）。無ければ背景輝度で自動決定
    try:
        ln_val = parameter_manager.store.current_value("runner.line_color")
        if ln_val is None:
            ln_val = parameter_manager.store.original_value("runner.line_color")
        if ln_val is not None:
            # 初期線色はベース色として適用し、レイヤー未指定時の既定色とする
            try:
                line_renderer.set_base_line_color(_norm(ln_val))
            except Exception:
                line_renderer.set_line_color(_norm(ln_val))
        else:
            br, bg_, bb, _ = getattr(rendering_window, "_bg_color", (1.0, 1.0, 1.0, 1.0))
            luminance = 0.2126 * float(br) + 0.7152 * float(bg_) + 0.0722 * float(bb)
            auto = (0.0, 0.0, 0.0, 1.0) if luminance >= 0.5 else (1.0, 1.0, 1.0, 1.0)
            try:
                line_renderer.set_base_line_color(auto)
            except Exception:
                line_renderer.set_line_color(auto)
    except Exception:
        pass

    # HUD 初期色（存在時）
    if overlay is not None:
        try:
            tx = parameter_manager.store.current_value(
                "runner.hud_text_color"
            ) or parameter_manager.store.original_value("runner.hud_text_color")
            if tx is not None:
                overlay.set_text_color(_norm(tx))
        except Exception:
            pass
    # 線太さ（store → renderer）
    try:
        th_val = parameter_manager.store.current_value("runner.line_thickness")
        if th_val is None:
            th_val = parameter_manager.store.original_value("runner.line_thickness")
        if th_val is not None:
            try:
                line_renderer.set_base_line_thickness(float(th_val))
            except Exception:
                line_renderer.set_line_thickness(float(th_val))
    except Exception:
        pass
        try:
            mt = parameter_manager.store.current_value(
                "runner.hud_meter_color"
            ) or parameter_manager.store.original_value("runner.hud_meter_color")
            if mt is not None:
                overlay.set_meter_color(_norm(mt))
        except Exception:
            pass
        try:
            mb = parameter_manager.store.current_value(
                "runner.hud_meter_bg_color"
            ) or parameter_manager.store.original_value("runner.hud_meter_bg_color")
            if mb is not None:
                overlay.set_meter_bg_color(_norm(mb))
        except Exception:
            pass


def subscribe_color_changes(
    parameter_manager, overlay, line_renderer, rendering_window, pyglet_mod
) -> None:
    """ParameterStore の変更に応じて UI スレッドで色を反映するよう購読する。"""

    if parameter_manager is None:
        return

    from typing import Iterable as _Iterable

    from util.color import normalize_color as _norm

    def _apply_bg_color(_dt: float, raw_val) -> None:  # noqa: ANN001, D401 - pyglet schedule 互換
        try:
            rendering_window.set_background_color(_norm(raw_val))
        except Exception:
            pass

    def _apply_line_color(_dt: float, raw_val) -> None:  # noqa: ANN001, D401 - pyglet schedule 互換
        """ランタイム線色を即時反映（単純化）。

        - レイヤー描画時は各レイヤーの色が優先されるため、ここでの更新は
          次の描画サイクルで上書きされ得るが、安全（副作用なし）。
        - 以前の layers-active ガードは冗長となったため削除（更新取りこぼし回避）。
        """
        try:
            # グローバル線色としてベース色も更新する
            try:
                line_renderer.set_base_line_color(_norm(raw_val))
            except Exception:
                line_renderer.set_line_color(_norm(raw_val))
        except Exception:
            pass

    def _apply_line_thickness(_dt: float, raw_val) -> None:  # noqa: ANN001, D401
        try:
            line_renderer.set_base_line_thickness(float(raw_val))
        except Exception:
            try:
                line_renderer.set_line_thickness(float(raw_val))
            except Exception:
                pass

    def _apply_palette_auto(_dt: float) -> None:  # noqa: ANN001, D401
        """Palette パラメータから背景/線色/レイヤー色を自動適用する。"""
        # 遅延 import で依存を局所化
        try:
            from engine.ui.palette.helpers import build_palette_from_values  # type: ignore[import]
            from palette import Palette  # type: ignore[import]
            from util.palette_state import set_palette as _set_palette  # type: ignore[import]
        except Exception:
            return

        store = parameter_manager.store

        # Palette 関連パラメータを Store から取得
        def _val(pid: str) -> object | None:
            try:
                v = store.current_value(pid)
                if v is None:
                    v = store.original_value(pid)
                return v
            except Exception:
                return None

        L_val = _val("palette.L")
        C_val = _val("palette.C")
        h_val = _val("palette.h")
        type_val = _val("palette.type")
        style_val = _val("palette.style")
        n_val = _val("palette.n_colors")

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

        if not isinstance(palette_obj, Palette):
            return

        # 自動適用モード（off / bg_global_and_layers）。古い値 "bg_and_global" は layers モード扱い。
        auto_mode = _val("palette.auto_apply_mode")
        raw_mode = str(auto_mode) if auto_mode is not None else "bg_global_and_layers"
        mode = "off" if raw_mode == "off" else "bg_global_and_layers"
        if mode == "off":
            return

        # 現在の背景色（固定キャンバス色）を取得
        try:
            bg_val = store.current_value("runner.background")
            if bg_val is None:
                bg_val = store.original_value("runner.background")
            if bg_val is None:
                bg_r, bg_g, bg_b = 1.0, 1.0, 1.0
            else:
                bg_r, bg_g, bg_b, _ = _norm(bg_val)
        except Exception:
            bg_r, bg_g, bg_b = 1.0, 1.0, 1.0
        bg_lum = 0.2126 * float(bg_r) + 0.7152 * float(bg_g) + 0.0722 * float(bg_b)

        # Palette.colors から OKLCH/L と sRGB を取得
        colors: list[tuple[float, float, float, float, float]] = []
        try:
            for c in palette_obj.colors:
                try:
                    L, _C, _h = c.oklch
                    r, g, b = c.srgb
                    colors.append((float(L), float(r), float(g), float(b), 1.0))
                except Exception:
                    continue
        except Exception:
            colors = []
        if not colors:
            return

        # グローバル線色: 背景との輝度差が最大の色（なければ黒/白フォールバック）
        line_rgba = None
        try:
            best = None
            best_diff = -1.0
            for _L, r, g, b, a in colors:
                lum = 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)
                diff = abs(lum - bg_lum)
                if diff > best_diff:
                    best_diff = diff
                    best = (r, g, b, a)
            if best is not None:
                line_rgba = best
        except Exception:
            line_rgba = None
        # しきい値が小さすぎる場合は黒/白へフォールバック
        if line_rgba is not None:
            try:
                r, g, b, _a = line_rgba
                lum_line = 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)
                if abs(lum_line - bg_lum) < 0.15:
                    line_rgba = None
            except Exception:
                line_rgba = None
        if line_rgba is None:
            if bg_lum >= 0.5:
                line_rgba = (0.0, 0.0, 0.0, 1.0)
            else:
                line_rgba = (1.0, 1.0, 1.0, 1.0)

        # Store とレンダラーへ反映（グローバル線色のみ、自動で背景は変えない）
        try:
            store.set_override("runner.line_color", line_rgba)
        except Exception:
            pass
        try:
            pyglet_mod.clock.schedule_once(
                lambda dt, v=line_rgba: _apply_line_color(dt, v),
                0.0,
            )
        except Exception:
            pass

        # レイヤー線色: layer.*.color に残りの色を循環適用
        if mode != "bg_global_and_layers":
            return
        try:
            layer_color_ids: list[str] = []
            for desc in store.descriptors():
                pid = desc.id
                if isinstance(pid, str) and pid.startswith("layer.") and pid.endswith(".color"):
                    layer_color_ids.append(pid)
            if not layer_color_ids:
                return
            layer_color_ids.sort()
            # 背景との輝度差が大きい順に色を並べる
            sorted_colors = sorted(
                colors,
                key=lambda it: abs(
                    (0.2126 * float(it[1]) + 0.7152 * float(it[2]) + 0.0722 * float(it[3])) - bg_lum
                ),
                reverse=True,
            )
            palette_cycle = [c[1:] for c in sorted_colors]  # (r,g,b,a) のリスト
            for idx, pid in enumerate(layer_color_ids):
                color_rgba = palette_cycle[idx % len(palette_cycle)]
                try:
                    store.set_override(pid, color_rgba)
                except Exception:
                    continue
        except Exception:
            return

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
                    pyglet_mod.clock.schedule_once(lambda dt, v=val: _apply_bg_color(dt, v), 0.0)
                    # line_color が未指定（override 無し）の場合は自動選択
                    lc_cur = parameter_manager.store.current_value("runner.line_color")
                    if lc_cur is None:
                        br, bg_, bb, _ = _norm(val)
                        luminance = 0.2126 * float(br) + 0.7152 * float(bg_) + 0.0722 * float(bb)
                        auto = (0.0, 0.0, 0.0, 1.0) if luminance >= 0.5 else (1.0, 1.0, 1.0, 1.0)
                        pyglet_mod.clock.schedule_once(
                            lambda dt, v=auto: _apply_line_color(dt, v), 0.0
                        )
                    # palette 自動適用モード有効時は、新しい背景に合わせて線色/レイヤー色を再計算
                    try:
                        pyglet_mod.clock.schedule_once(_apply_palette_auto, 0.0)
                    except Exception:
                        pass
            except Exception:
                pass
        # 線色
        if "runner.line_color" in id_list:
            try:
                val = parameter_manager.store.current_value("runner.line_color")
                if val is None:
                    val = parameter_manager.store.original_value("runner.line_color")
                if val is not None:
                    pyglet_mod.clock.schedule_once(lambda dt, v=val: _apply_line_color(dt, v), 0.0)
            except Exception:
                pass
        # 線太さ
        if "runner.line_thickness" in id_list:
            try:
                val = parameter_manager.store.current_value("runner.line_thickness")
                if val is None:
                    val = parameter_manager.store.original_value("runner.line_thickness")
                if val is not None:
                    pyglet_mod.clock.schedule_once(
                        lambda dt, v=val: _apply_line_thickness(dt, v), 0.0
                    )
            except Exception:
                pass
        # HUD テキスト色
        if "runner.hud_text_color" in id_list and overlay is not None:
            try:
                val = parameter_manager.store.current_value("runner.hud_text_color")
                if val is None:
                    val = parameter_manager.store.original_value("runner.hud_text_color")
                if val is not None:
                    rgba = _norm(val)
                    pyglet_mod.clock.schedule_once(
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
                    rgba = _norm(val)
                    pyglet_mod.clock.schedule_once(
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
                    rgba = _norm(val)
                    pyglet_mod.clock.schedule_once(
                        lambda dt, v=rgba: overlay.set_meter_bg_color(v), 0.0
                    )
            except Exception:
                pass
        # Palette 変更に応じた自動適用
        if any(isinstance(pid, str) and pid.startswith("palette.") for pid in id_list):
            try:
                pyglet_mod.clock.schedule_once(_apply_palette_auto, 0.0)
            except Exception:
                pass

    try:
        parameter_manager.store.subscribe(_on_param_store_change)
    except Exception:
        pass


__all__ = [
    "make_param_snapshot_fn",
    "apply_initial_colors",
    "subscribe_color_changes",
    "subscribe_hud_visibility_changes",
]


def subscribe_hud_visibility_changes(
    parameter_manager, overlay, pyglet_mod, *, lock: bool = False
) -> None:
    """HUD 表示可否を Parameter GUI の変更で反映する。

    - lock=True（= 明示引数で固定）の場合は購読しない。
    - overlay/parameter_manager が None の場合は何もしない。
    """

    if lock or parameter_manager is None or overlay is None:
        return

    def _on_param_store_change(ids):  # type: ignore[no-untyped-def]
        try:
            # ids は Iterable[str] または Mapping[str,Any]
            if isinstance(ids, Mapping):  # type: ignore[name-defined]
                id_list = list(ids.keys())
            else:
                try:
                    id_list = list(ids)  # type: ignore[list-item]
                except TypeError:
                    id_list = []
            if "runner.show_hud" in id_list:
                try:
                    v = parameter_manager.store.current_value("runner.show_hud")
                    on = bool(v) if v is not None else True
                except Exception:
                    on = True
                pyglet_mod.clock.schedule_once(lambda dt, val=on: overlay.set_enabled(val), 0.0)
        except Exception:
            return

    try:
        parameter_manager.store.subscribe(_on_param_store_change)
    except Exception:
        pass
