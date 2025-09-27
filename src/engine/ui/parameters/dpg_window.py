"""
どこで: `engine.ui.parameters` の Dear PyGui 実装。
何を: ParameterStore の Descriptor から DPG ウィジェットを生成し、双方向に値同期する。
なぜ: パラメータ GUI を DPG に一本化し、見た目と操作性/保守性を向上するため。

注意:
- 本実装はヘッドレス/未導入環境でも import エラーにならないようガードする。
- フレーム駆動は可能なら `pyglet.clock` 経由で `render_dearpygui_frame()` を定期呼出す。
  （pyglet 非利用時はバックグラウンドスレッドで最小駆動。）
"""

from __future__ import annotations

from threading import Thread
from typing import Any, Iterable

try:  # dearpygui の存在は環境依存なのでガード
    import dearpygui.dearpygui as _dpg  # type: ignore
except Exception:  # pragma: no cover - headless/未導入用フォールバック
    _dpg = None  # type: ignore[assignment]
# 解析器に Optional を意識させないため Any にアサイン
dpg: Any = _dpg

try:  # pyglet があれば描画フレームを統合（メインスレッド駆動）
    import pyglet as _pyglet  # type: ignore
except Exception:  # pragma: no cover - headless/未導入
    _pyglet = None  # type: ignore[assignment]
pyglet: Any = _pyglet

from .state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
)

# ------------------------------
# ヘッドレス/未導入環境のスタブ
# ------------------------------

if dpg is None:  # pragma: no cover - import 不可時のダミー実装

    class ParameterWindow:  # type: ignore[override]
        """DPG 未導入/ヘッドレス環境向けのダミーウィンドウ。"""

        def __init__(
            self,
            *,
            store: ParameterStore,
            layout: ParameterLayoutConfig,
            **_kwargs: Any,
        ) -> None:
            self._store = store
            self._layout = layout

        def set_visible(self, _visible: bool) -> None:  # noqa: D401
            return None

        def close(self) -> None:  # noqa: D401
            return None

        # 互換 API（テスト利便用）
        def mount(self, _descriptors: list[ParameterDescriptor]) -> None:
            return None

else:

    class ParameterWindow:  # type: ignore[override]
        """Dear PyGui によるパラメータウィンドウ。"""

        def __init__(
            self,
            *,
            store: ParameterStore,
            layout: ParameterLayoutConfig,
            width: int = 420,
            height: int = 640,
            title: str = "Parameters",
        ) -> None:
            self._store = store
            self._layout = layout
            self._width = width
            self._height = height
            self._title = title

            self._syncing = False
            self._thread: Thread | None = None
            self._using_pyglet = False
            self._highlight_theme: int | None = None

            # DPG ライフサイクル
            dpg.create_context()
            # ビューポート参照は明示破棄（close）まで保持（GC/順序の明確化）
            self._viewport = dpg.create_viewport(
                title=self._title, width=self._width, height=self._height
            )
            dpg.setup_dearpygui()

            # ルートウィンドウ
            with dpg.window(
                tag="__pxd_param_root__", label=self._title, no_resize=False, no_collapse=True
            ) as root:
                dpg.add_child_window(
                    tag="__pxd_param_scroll__", autosize_x=True, autosize_y=True, border=False
                )
            dpg.set_primary_window(root, True)

            # 最小テーマ（パディングのみ）
            self._setup_theme()

            # 初期マウント
            self.mount(sorted(self._store.descriptors(), key=lambda d: d.id))

            # 購読
            self._store.subscribe(self._on_store_change)

            # 表示 + 駆動
            dpg.show_viewport()
            if pyglet is not None:
                # macOS では UI イベントはメインスレッド限定のため、pyglet に統合してフレームを駆動
                self._using_pyglet = True
                pyglet.clock.schedule_interval(self._tick, 1 / 60)
            else:
                # pyglet が無ければバックグラウンドスレッドで Dear PyGui を駆動
                self._thread = Thread(target=self._run_loop, name="DPGLoop", daemon=True)
                self._thread.start()

        # ---- ライフサイクル ----
        def set_visible(self, visible: bool) -> None:
            try:
                if visible:
                    dpg.show_viewport()
                else:
                    dpg.hide_viewport()
            except Exception:
                pass

        def close(self) -> None:
            try:
                self._store.unsubscribe(self._on_store_change)
            except Exception:
                pass
            if self._using_pyglet and pyglet is not None:
                try:
                    pyglet.clock.unschedule(self._tick)
                except Exception:
                    pass
            if self._thread is not None:
                try:
                    dpg.stop_dearpygui()
                    self._thread.join(timeout=1.0)
                except Exception:
                    pass
            try:
                dpg.destroy_context()
            except Exception:
                pass

        # ---- マウント/構築 ----
        def mount(self, descriptors: list[ParameterDescriptor]) -> None:
            scroll = "__pxd_param_scroll__"
            with dpg.stage(tag="__pxd_param_stage__"):
                # カテゴリ別の折りたたみヘッダを作成し、その配下に 2 カラムテーブルを配置
                # カテゴリ順は名称の昇順、同一カテゴリ内は id の昇順
                sorted_desc = sorted(descriptors, key=lambda d: (d.category, d.id))
                # 走査しながらカテゴリ境界でグループを切替
                current_cat: str | None = None
                group_items: list[ParameterDescriptor] = []

                def flush_group(cat: str | None, items: list[ParameterDescriptor]) -> None:
                    if not items:
                        return
                    # 表示対象がない場合はスキップ
                    if not any(it.supported for it in items):
                        return
                    label = cat if cat else "General"
                    with dpg.collapsing_header(label=label, parent=scroll, default_open=True):
                        with dpg.table(header_row=False) as table:
                            dpg.add_table_column(label="Parameter")
                            dpg.add_table_column(label="Value")
                            for it in items:
                                if not it.supported:
                                    continue
                                self._create_row(table, it)
                                # 差分ハイライトは行わない

                for desc in sorted_desc:
                    if current_cat is None:
                        current_cat = desc.category
                        group_items = [desc]
                        continue
                    if desc.category != current_cat:
                        flush_group(current_cat, group_items)
                        current_cat = desc.category
                        group_items = [desc]
                    else:
                        group_items.append(desc)
                # 最後のグループをフラッシュ
                flush_group(current_cat, group_items)
            dpg.unstage("__pxd_param_stage__")

        def _create_row(self, table: int | str, desc: ParameterDescriptor) -> None:
            # テーブル行（左: ラベル / 右: 入力）
            with dpg.table_row(parent=table) as row:
                label = desc.label or desc.id
                dpg.add_text(default_value=label, parent=row)
                self._create_widget(row, desc)
            # ツールチップは非採用（ポップアップは表示しない）

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
            if vt == "vector":
                vec = list(value) if isinstance(value, (list, tuple)) else [0.0, 0.0, 0.0]
                if len(vec) == 4:
                    return dpg.add_input_float4(
                        tag=desc.id,
                        parent=parent,
                        label="",
                        default_value=vec,  # type: ignore[arg-type]
                        format=f"%.{self._layout.value_precision}f",
                        callback=self._on_widget_change,
                        user_data=desc.id,
                    )
                return dpg.add_input_float3(
                    tag=desc.id,
                    parent=parent,
                    label="",
                    default_value=(vec + [0.0, 0.0, 0.0])[:3],  # type: ignore[arg-type]
                    format=f"%.{self._layout.value_precision}f",
                    callback=self._on_widget_change,
                    user_data=desc.id,
                )
            # 数値（float/int）
            hint = desc.range_hint or self._layout.derive_range(
                name=desc.id, value_type=desc.value_type, default_value=desc.default_value
            )
            if vt == "int":
                return dpg.add_slider_int(
                    tag=desc.id,
                    parent=parent,
                    label="",
                    default_value=int(value) if value is not None else 0,
                    min_value=int(hint.min_value),
                    max_value=int(hint.max_value),
                    callback=self._on_widget_change,
                    user_data=desc.id,
                )
            # float 既定
            return dpg.add_slider_float(
                tag=desc.id,
                parent=parent,
                label="",
                default_value=float(value) if value is not None else 0.0,
                min_value=float(hint.min_value),
                max_value=float(hint.max_value),
                format=f"%.{self._layout.value_precision}f",
                callback=self._on_widget_change,
                user_data=desc.id,
            )

        # Reset 操作は未実装（要望により撤廃）

        # ---- 値連携 ----
        def _current_or_default(self, desc: ParameterDescriptor) -> Any:
            v = self._store.current_value(desc.id)
            return v if v is not None else desc.default_value

        def _on_widget_change(
            self, sender: int, app_data: Any, user_data: Any
        ) -> None:  # noqa: D401
            if self._syncing:
                return
            pid = str(user_data)
            value = app_data
            # vector は list → tuple へ
            if isinstance(value, list):
                value = tuple(value)
            self._store.set_override(pid, value)

        def _on_store_change(self, ids: Iterable[str]) -> None:
            # 差分のみ反映（同値更新は DPG 側で冪等）
            self._syncing = True
            try:
                for pid in ids:
                    if not dpg.does_item_exist(pid):
                        continue
                    value = self._store.current_value(pid)
                    if value is None:
                        value = self._store.original_value(pid)
                    # vector は list を期待するため変換
                    try:
                        desc = self._store.get_descriptor(pid)
                        if desc.value_type == "vector" and isinstance(value, tuple):
                            value = list(value)
                    except Exception:
                        pass
                    try:
                        dpg.set_value(pid, value)
                    except Exception:
                        pass
                    # 差分ハイライトは行わない
            finally:
                self._syncing = False

        # Reset 操作は未実装（要望により撤廃）

        # ---- 駆動 ----
        def _tick(self, _dt: float) -> None:
            # pyglet のスケジューラからメインスレッドで呼ばれる
            try:
                if dpg.is_dearpygui_running():
                    dpg.render_dearpygui_frame()
            except Exception:
                pass

        def _run_loop(self) -> None:
            try:
                dpg.start_dearpygui()
            except Exception:
                pass

        # ---- テーマ（最小: パディングのみ） ----
        def _setup_theme(self) -> None:
            try:
                with dpg.theme() as theme:
                    # 全体のパディング/スペーシングのみ調整
                    with dpg.theme_component(dpg.mvAll):
                        pad = int(self._layout.padding)
                        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, pad, pad)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, pad, max(1, pad // 2))
                        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, pad, max(1, pad // 2))
                dpg.bind_theme(theme)
            except Exception:
                # 失敗しても既定スタイルで継続
                pass


__all__ = ["ParameterWindow"]
