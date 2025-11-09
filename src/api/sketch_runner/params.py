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
            line_renderer.set_line_color(_norm(ln_val))
        else:
            br, bg_, bb, _ = getattr(rendering_window, "_bg_color", (1.0, 1.0, 1.0, 1.0))
            luminance = 0.2126 * float(br) + 0.7152 * float(bg_) + 0.0722 * float(bb)
            auto = (0.0, 0.0, 0.0, 1.0) if luminance >= 0.5 else (1.0, 1.0, 1.0, 1.0)
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
            line_renderer.set_line_color(_norm(raw_val))
        except Exception:
            pass

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

    try:
        parameter_manager.store.subscribe(_on_param_store_change)
    except Exception:
        pass


__all__ = [
    "make_param_snapshot_fn",
    "apply_initial_colors",
    "subscribe_color_changes",
]
