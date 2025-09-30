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

from contextlib import ExitStack
from threading import Thread
from typing import Any, ContextManager, Iterable, Sequence, cast

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

from .state import ParameterDescriptor, ParameterLayoutConfig, ParameterStore, ParameterThemeConfig

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

    class ParameterWindow:  # type: ignore[override, no-redef]
        """Dear PyGui によるパラメータウィンドウ。"""

        def __init__(
            self,
            *,
            store: ParameterStore,
            layout: ParameterLayoutConfig,
            width: int = 420,
            height: int = 640,
            title: str = "Parameters",
            theme: ParameterThemeConfig | None = None,
        ) -> None:
            self._store = store
            self._layout = layout
            self._width = width
            self._height = height
            self._title = title
            self._theme = theme

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

            # テーマ適用（指定があれば優先、無くても最小テーマを適用）
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
                        # 列幅の比率を設定に基づいてストレッチ適用
                        table_policy = getattr(dpg, "mvTable_SizingStretchProp", None)
                        if table_policy is None:
                            table_policy = getattr(dpg, "mvTable_SizingStretchSame", None)
                        with dpg.table(header_row=False, policy=table_policy) as table:
                            # 0.1..0.9 にクランプし、左右の weight とする
                            try:
                                left = float(self._layout.label_column_ratio)
                            except Exception:
                                left = 0.5
                            left = 0.1 if left < 0.1 else (0.9 if left > 0.9 else left)
                            right = max(0.1, 1.0 - left)
                            try:
                                dpg.add_table_column(
                                    label="Parameter",
                                    width_stretch=True,
                                    init_width_or_weight=left,
                                )
                                dpg.add_table_column(
                                    label="Value",
                                    width_stretch=True,
                                    init_width_or_weight=right,
                                )
                            except Exception:
                                # 旧環境向けフォールバック（等分）
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
                dim = 4 if len(vec) >= 4 else 3
                # レンジは vector_hint を優先、無ければ既定 0..1
                vh = desc.vector_hint
                if vh is None:
                    default_vh = self._layout.derive_vector_range(dim=dim)
                    vmin = list(default_vh.min_values)
                    vmax = list(default_vh.max_values)
                else:
                    vmin = list(vh.min_values)
                    vmax = list(vh.max_values)
                # 右列に水平配置で 3/4 本生成（各スライダの幅は 1/3 or 1/4 に抑制）
                with dpg.table(
                    parent=parent, header_row=False, policy=dpg.mvTable_SizingStretchSame
                ) as vec_table:
                    # ネストしたテーブルのセル余白を 0 にして、単一スライダー行と縦サイズを揃える
                    try:
                        with dpg.theme() as _vec_theme:
                            with dpg.theme_component(dpg.mvAll):
                                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 1, 0)
                        dpg.bind_item_theme(vec_table, _vec_theme)
                    except Exception:
                        pass
                    for _ in range(dim):
                        dpg.add_table_column(width_stretch=True)
                    with dpg.table_row():
                        for i, suffix in enumerate(("x", "y", "z", "w")[:dim]):
                            # 列送りAPIに依存せず、各セルを明示して配置（クリーンな実装）
                            # `ExitStack` を用い、`table_cell()` が無い環境でも型安全に no-op とする
                            with ExitStack() as _cell_stack:
                                _cell_ctx = getattr(dpg, "table_cell", None)
                                if callable(_cell_ctx):
                                    _cm = cast(ContextManager[Any], _cell_ctx())
                                    _cell_stack.enter_context(_cm)
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
                                # セル幅にフィット
                                dpg.set_item_width(slider_id, -1)
                return vec_table
            # 数値（float/int）
            hint = desc.range_hint or self._layout.derive_range(
                name=desc.id, value_type=desc.value_type, default_value=desc.default_value
            )
            if vt == "int":
                slider_id = dpg.add_slider_int(
                    tag=desc.id,
                    parent=parent,
                    label="",
                    default_value=int(value) if value is not None else 0,
                    min_value=int(hint.min_value),
                    max_value=int(hint.max_value),
                    callback=self._on_widget_change,
                    user_data=desc.id,
                )
                # scalar もベクトルと同等に“3列分”の横幅（セル全幅）を占有
                dpg.set_item_width(slider_id, -1)
                return slider_id
            # float 既定
            slider_id_f = dpg.add_slider_float(
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
            # scalar もベクトルと同等に“3列分”の横幅（セル全幅）を占有
            dpg.set_item_width(slider_id_f, -1)
            return slider_id_f

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
            # user_data が (parent_id, axis_index) ならベクトルの一部更新
            if isinstance(user_data, (list, tuple)) and len(user_data) == 2:
                parent_id = str(user_data[0])
                idx = int(user_data[1])
                try:
                    current = self._store.current_value(parent_id)
                    if not isinstance(current, (list, tuple)):
                        current = self._store.original_value(parent_id)
                    vec = list(current) if isinstance(current, (list, tuple)) else [0.0, 0.0, 0.0]
                    dim = 4 if len(vec) >= 4 else 3
                    # 現在ベクトルと同次元で更新
                    base = (vec + [0.0] * dim)[:dim]
                    base[idx] = float(app_data)
                    self._store.set_override(parent_id, tuple(base))
                    return
                except Exception:
                    return
            # それ以外は単一値
            pid = str(user_data)
            value = app_data
            self._store.set_override(pid, value)

        def _on_store_change(self, ids: Iterable[str]) -> None:
            # 差分のみ反映（同値更新は DPG 側で冪等）
            self._syncing = True
            try:
                for pid in ids:
                    # まず直接タグが存在する場合（scalar 等）
                    if dpg.does_item_exist(pid):
                        value = self._store.current_value(pid)
                        if value is None:
                            value = self._store.original_value(pid)
                        try:
                            dpg.set_value(pid, value)
                        except Exception:
                            pass
                        continue
                    # 親ベクトル ID の場合は子スライダへ反映
                    try:
                        desc = self._store.get_descriptor(pid)
                    except Exception:
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
                        try:
                            dpg.set_value(tag, float(vec[i]))
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
                    with dpg.theme_component(dpg.mvAll):
                        # スタイル適用
                        style_applied = False
                        if self._theme is not None and isinstance(self._theme.style, dict):
                            style_applied = True
                            smap = {
                                "window_padding": getattr(dpg, "mvStyleVar_WindowPadding", None),
                                "frame_padding": getattr(dpg, "mvStyleVar_FramePadding", None),
                                "item_spacing": getattr(dpg, "mvStyleVar_ItemSpacing", None),
                                "frame_rounding": getattr(dpg, "mvStyleVar_FrameRounding", None),
                                "grab_rounding": getattr(dpg, "mvStyleVar_GrabRounding", None),
                                "grab_min_size": getattr(dpg, "mvStyleVar_GrabMinSize", None),
                            }

                            def _add_style(
                                var: Any, value: float | int | Sequence[float | int]
                            ) -> None:
                                if not var:
                                    return
                                try:
                                    # 数値のシーケンス（str/bytes は除外）を明確に扱う
                                    if isinstance(value, Sequence) and not isinstance(
                                        value, (str, bytes)
                                    ):
                                        seq = cast(Sequence[float | int], value)
                                        if len(seq) >= 2:
                                            dpg.add_theme_style(var, float(seq[0]), float(seq[1]))
                                            return
                                        if len(seq) == 1:
                                            dpg.add_theme_style(var, float(seq[0]))
                                            return
                                        # 長さ 0 の場合は適用しない
                                        return
                                    # 単一値
                                    dpg.add_theme_style(var, float(cast(float | int, value)))
                                except Exception:
                                    pass

                            for key, var in smap.items():
                                if key in self._theme.style:
                                    _add_style(var, self._theme.style[key])

                        if not style_applied:
                            # 既定の最小スタイル（padding ベース）
                            pad = int(self._layout.padding)
                            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, pad, pad)
                            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, pad, max(1, pad // 2))
                            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, pad, max(1, pad // 2))

                        # 色適用（config では RGBA 0..255 を推奨。0..1 float も許容し 0..255 に拡大）
                        if self._theme is not None and isinstance(self._theme.colors, dict):
                            cmap = {
                                "text": getattr(dpg, "mvThemeCol_Text", None),
                                "window_bg": getattr(dpg, "mvThemeCol_WindowBg", None),
                                "frame_bg": getattr(dpg, "mvThemeCol_FrameBg", None),
                                "frame_bg_hovered": getattr(dpg, "mvThemeCol_FrameBgHovered", None),
                                "frame_bg_active": getattr(dpg, "mvThemeCol_FrameBgActive", None),
                                "header": getattr(dpg, "mvThemeCol_Header", None),
                                "header_hovered": getattr(dpg, "mvThemeCol_HeaderHovered", None),
                                "header_active": getattr(dpg, "mvThemeCol_HeaderActive", None),
                                # アクセントはスライダーの Grab に反映
                                "accent": getattr(dpg, "mvThemeCol_SliderGrab", None),
                                "accent_active": getattr(dpg, "mvThemeCol_SliderGrabActive", None),
                            }

                            def _to_dpg_color(value: Any) -> Any:
                                try:
                                    if isinstance(value, (list, tuple)) and len(value) >= 4:
                                        vals = [
                                            float(value[0]),
                                            float(value[1]),
                                            float(value[2]),
                                            float(value[3]),
                                        ]
                                        # 0..1 の場合は 0..255 へ拡大
                                        if all(0.0 <= v <= 1.0 for v in vals):
                                            return [int(round(v * 255)) for v in vals]
                                        return [int(round(v)) for v in vals]
                                except Exception:
                                    return None
                                return None

                            for key, var in cmap.items():
                                if not var:
                                    continue
                                if key in self._theme.colors:
                                    col = _to_dpg_color(self._theme.colors[key])
                                    if col is not None:
                                        try:
                                            dpg.add_theme_color(var, col)
                                        except Exception:
                                            pass

                dpg.bind_theme(theme)
            except Exception:
                # 失敗しても既定スタイルで継続
                pass


__all__ = ["ParameterWindow"]
