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

from dataclasses import dataclass
from threading import Event, Thread
from typing import Any, Iterable

try:  # dearpygui の存在は環境依存なのでガード
    import dearpygui.dearpygui as dpg  # type: ignore
except Exception:  # pragma: no cover - headless/未導入用フォールバック
    dpg = None  # type: ignore[assignment]

try:  # pyglet があれば描画フレームを統合（任意）
    import pyglet  # type: ignore
except Exception:  # pragma: no cover - headless/未導入
    pyglet = None  # type: ignore[assignment]

from .state import (
    ParameterDescriptor,
    ParameterLayoutConfig,
    ParameterStore,
)

# ------------------------------
# ヘッドレス/未導入環境のスタブ
# ------------------------------

if dpg is None:  # pragma: no cover - import 不可時のダミー実装

    class _ParameterWindowStub:  # type: ignore[override]
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

    __all__ = ["ParameterWindow"]

else:

    @dataclass
    class _ItemRefs:
        widget_id: int
        label_id: int | None
        reset_id: int | None

    class _ParameterWindowImpl:  # type: ignore[override]
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

            self._items: dict[str, _ItemRefs] = {}
            self._syncing = False
            self._closed = Event()
            self._thread: Thread | None = None
            self._using_pyglet = False

            # DPG ライフサイクル
            dpg.create_context()
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

            # 最小テーマ（パディングのみ + enum コントラスト）
            self._setup_theme()

            # 初期マウント
            self.mount(sorted(self._store.descriptors(), key=lambda d: d.id))

            # 購読
            self._store.subscribe(self._on_store_change)

            # 表示
            dpg.show_viewport()

            # 駆動（pyglet があれば統合、無ければスレッド）
            if pyglet is not None:
                self._using_pyglet = True
                pyglet.clock.schedule_interval(self._tick, 1 / 60)
            else:
                self._thread = Thread(target=self._run_loop, name="DPGLoop", daemon=True)
                self._thread.start()

        # ---- ライフサイクル ----
        def set_visible(self, visible: bool) -> None:
            try:
                if visible:
                    dpg.show_viewport()
                else:
                    dpg.minimize_viewport()
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
                self._closed.set()
                try:
                    self._thread.join(timeout=1.0)
                except Exception:
                    pass
            try:
                dpg.destroy_context()
            finally:
                self._items.clear()

        # ---- マウント/構築 ----
        def mount(self, descriptors: list[ParameterDescriptor]) -> None:
            scroll = dpg.get_item_alias("__pxd_param_scroll__")
            if scroll is None:
                scroll = "__pxd_param_scroll__"
            with dpg.stage(tag="__pxd_param_stage__"):
                for desc in descriptors:
                    if not desc.supported:
                        continue
                    self._create_row(scroll, desc)
            dpg.unstage("__pxd_param_stage__")

        def _create_row(self, parent: int | str, desc: ParameterDescriptor) -> None:
            # ラベル
            label_id: int | None = None
            if desc.label:
                label_id = dpg.add_text(default_value=desc.label, parent=parent)
            # ウィジェット
            widget_id = self._create_widget(parent, desc)
            # ツールチップは非採用（ポップアップは表示しない）
            self._items[desc.id] = _ItemRefs(widget_id=widget_id, label_id=label_id, reset_id=None)

        def _create_widget(self, parent: int | str, desc: ParameterDescriptor) -> int:
            value = self._current_or_default(desc)
            vt = desc.value_type
            if vt == "bool":
                return dpg.add_checkbox(
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
                        items=items,
                        parent=parent,
                        default_value=default,
                        horizontal=True,
                        callback=self._on_widget_change,
                        user_data=desc.id,
                    )
                return dpg.add_combo(
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
                        parent=parent,
                        label="",
                        default_value=vec,  # type: ignore[arg-type]
                        callback=self._on_widget_change,
                        user_data=desc.id,
                    )
                return dpg.add_input_float3(
                    parent=parent,
                    label="",
                    default_value=(vec + [0.0, 0.0, 0.0])[:3],  # type: ignore[arg-type]
                    callback=self._on_widget_change,
                    user_data=desc.id,
                )
            # 数値（float/int）
            hint = desc.range_hint or self._layout.derive_range(
                name=desc.id, value_type=desc.value_type, default_value=desc.default_value
            )
            if vt == "int":
                return dpg.add_slider_int(
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
            self._store.set_override(pid, value, source="gui")

        def _on_store_change(self, ids: Iterable[str]) -> None:
            # 差分のみ反映（同値更新は DPG 側で冪等）
            self._syncing = True
            try:
                for pid in ids:
                    refs = self._items.get(pid)
                    if not refs:
                        continue
                    value = self._store.current_value(pid)
                    if value is None:
                        # original に戻す
                        value = self._store.original_value(pid)
                    try:
                        dpg.set_value(refs.widget_id, value)
                    except Exception:
                        # 型不一致時などは黙ってスキップ
                        pass
            finally:
                self._syncing = False

        # Reset 操作は未実装（要望により撤廃）

        # ---- 駆動 ----
        def _tick(self, _dt: float) -> None:
            try:
                if dpg.is_dearpygui_running():
                    dpg.render_dearpygui_frame()
            except Exception:
                pass

        def _run_loop(self) -> None:
            while not self._closed.is_set():
                try:
                    if dpg.is_dearpygui_running():
                        dpg.render_dearpygui_frame()
                except Exception:
                    pass
                self._closed.wait(1.0 / 60.0)

        # ---- テーマ（最小: パディング + enum コントラスト） ----
        def _setup_theme(self) -> None:
            try:
                with dpg.theme() as theme:
                    # 全体のパディング/スペーシングのみ調整
                    with dpg.theme_component(dpg.mvAll):
                        pad = int(self._layout.padding)
                        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, pad, pad)
                        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, pad, max(1, pad // 2))
                        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, pad, max(1, pad // 2))
                    # enum（radio）の選択視認性: チェックマーク色を強調
                    with dpg.theme_component(dpg.mvRadioButton):
                        dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (30, 140, 255, 255))
                    # enum（combo 内の選択肢）: 選択ハイライトのコントラストを上げる
                    with dpg.theme_component(dpg.mvSelectable):
                        dpg.add_theme_color(dpg.mvThemeCol_Header, (30, 140, 255, 64))
                        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (30, 140, 255, 96))
                        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (30, 140, 255, 128))
                dpg.bind_theme(theme)
            except Exception:
                # 失敗しても既定スタイルで継続
                pass

    __all__ = ["ParameterWindow"]

# 公開エイリアス
ParameterWindow = _ParameterWindowStub if dpg is None else _ParameterWindowImpl
