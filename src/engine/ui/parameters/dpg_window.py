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

import logging
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

# ---------------------------------
# ロガーとタグ定数（集約）
# ---------------------------------
logger = logging.getLogger("engine.ui.parameters.dpg")

ROOT_TAG = "__pxd_param_root__"
SCROLL_TAG = "__pxd_param_scroll__"
STAGE_TAG = "__pxd_param_stage__"

# ---------------------------------
# 薄いベース/スタブ（duck-typing）
# ---------------------------------


class ParameterWindowBase:
    """パラメータウィンドウの最小インタフェース。

    役割: GUI 実装に依存せず、ライフサイクル操作の契約を最小化する。
    """

    def set_visible(self, _visible: bool) -> None:  # pragma: no cover - インタフェース
        return None

    def close(self) -> None:  # pragma: no cover - インタフェース
        return None

    def mount(self, _descriptors: list[ParameterDescriptor]) -> None:  # pragma: no cover
        return None


# ------------------------------
# ヘッドレス/未導入環境のスタブ
# ------------------------------

if dpg is None:  # pragma: no cover - import 不可時のダミー実装

    class ParameterWindow(ParameterWindowBase):  # type: ignore[override]
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
            logger.debug("NullParameterWindow initialized (headless/stub mode)")

else:

    # ------------------------------
    # 内部ドライバ（pyglet / thread）
    # ------------------------------

    class _PygletDriver:
        """pyglet.clock に Dear PyGui のフレームを統合して駆動する。"""

        def __init__(self, tick_cb: Any, *, fps: float = 60.0) -> None:
            self._tick_cb = tick_cb
            self._fps = float(fps)
            self._active = False

        def start(self) -> None:
            if pyglet is None:
                return
            try:
                interval = 1.0 / self._fps if self._fps > 0 else 1.0 / 60.0
                pyglet.clock.schedule_interval(self._tick_cb, interval)
                self._active = True
                logger.debug("PygletDriver started at %.2f FPS", self._fps)
            except Exception:
                logger.exception("failed to start PygletDriver")

        def stop(self) -> None:
            if pyglet is None or not self._active:
                return
            try:
                pyglet.clock.unschedule(self._tick_cb)
                logger.debug("PygletDriver stopped")
            except Exception:
                logger.exception("failed to stop PygletDriver")
            finally:
                self._active = False

    class _ThreadDriver:
        """バックグラウンドスレッドで Dear PyGui を起動/停止する。"""

        def __init__(self) -> None:
            self._thread: Thread | None = None

        def start(self) -> None:
            try:
                self._thread = Thread(target=dpg.start_dearpygui, name="DPGLoop", daemon=True)
                self._thread.start()
                logger.debug("ThreadDriver started")
            except Exception:
                logger.exception("failed to start ThreadDriver")

        def stop(self) -> None:
            if self._thread is None:
                return
            try:
                dpg.stop_dearpygui()
                self._thread.join(timeout=1.0)
                logger.debug("ThreadDriver stopped")
            except Exception:
                logger.exception("failed to stop ThreadDriver")
            finally:
                self._thread = None

    class ParameterWindow(ParameterWindowBase):  # type: ignore[override, no-redef]
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
            self._using_pyglet = False
            self._highlight_theme: int | None = None
            self._driver: Any | None = None

            # DPG ライフサイクル
            dpg.create_context()
            # ビューポート参照は明示破棄（close）まで保持（GC/順序の明確化）
            self._viewport = dpg.create_viewport(
                title=self._title, width=self._width, height=self._height
            )
            dpg.setup_dearpygui()

            # ルートウィンドウ
            self._build_root_window()

            # テーマ適用（指定があれば優先、無くても最小テーマを適用）
            self._setup_theme()

            # 初期マウント
            self.mount(sorted(self._store.descriptors(), key=lambda d: d.id))

            # 購読（変更時にランナー色を再同期）
            def _on_store_change_wrapper(ids: Iterable[str]) -> None:
                try:
                    self._on_store_change(ids)
                    # ランナー色を強制再同期（冪等）
                    self.sync_display_from_store(self._store)
                except Exception:
                    logger.exception("store change handling failed")

            self._store.subscribe(_on_store_change_wrapper)

            # 表示 + 駆動
            dpg.show_viewport()
            # ドライバ選択
            if pyglet is not None:
                self._using_pyglet = True
                self._driver = _PygletDriver(self._tick, fps=60.0)
            else:
                self._driver = _ThreadDriver()
            # 起動
            try:
                self._driver.start()
            except Exception:
                logger.exception("failed to start driver")

        # ---- ライフサイクル ----
        def set_visible(self, visible: bool) -> None:
            try:
                if visible:
                    dpg.show_viewport()
                else:
                    dpg.hide_viewport()
            except Exception:
                logger.exception("set_visible failed (visible=%s)", visible)

        def close(self) -> None:
            try:
                self._store.unsubscribe(self._on_store_change)
            except Exception:
                logger.exception("unsubscribe failed in close()")
            try:
                if self._driver is not None:
                    self._driver.stop()
            except Exception:
                logger.exception("driver.stop failed")
            try:
                dpg.destroy_context()
            except Exception:
                logger.exception("destroy_context failed")

        # ---- マウント/構築 ----
        def mount(self, descriptors: list[ParameterDescriptor]) -> None:
            with dpg.stage(tag=STAGE_TAG):
                self._build_grouped_table(SCROLL_TAG, descriptors)
            dpg.unstage(STAGE_TAG)

        def _build_root_window(self) -> None:
            """ルートウィンドウとスクロール領域を構築し、プライマリに設定する。

            Returns
            -------
            None
                副作用として Dear PyGui のウィンドウと子スクロール領域を作成する。
            """
            with dpg.window(
                tag=ROOT_TAG, label=self._title, no_resize=False, no_collapse=True
            ) as root:
                # 上部に Display セクション、下部にスクロール領域（通常のパラメータ）
                try:
                    self.build_display_controls(parent=root, store=self._store)
                except Exception:
                    logger.exception("failed to build runner controls")
                dpg.add_child_window(tag=SCROLL_TAG, autosize_x=True, autosize_y=True, border=False)
            dpg.set_primary_window(root, True)

        def _build_grouped_table(
            self, parent: int | str, descriptors: list[ParameterDescriptor]
        ) -> None:
            """カテゴリごとの折りたたみと2カラムテーブルを構築して行を追加する。

            Parameters
            ----------
            parent
                親アイテムのタグまたはID。
            descriptors
                表示するパラメータ記述子の一覧。

            Returns
            -------
            None
            """
            # Runner 専用の Display コントロール（runner.*）は上部に別枠を作っているため
            # 通常テーブルからは除外して二重表示を防ぐ。
            excluded_ids = {"runner.background", "runner.line_color"}
            filtered = [d for d in descriptors if d.id not in excluded_ids]
            sorted_desc = sorted(filtered, key=lambda d: (d.category, d.id))
            current_cat: str | None = None
            group_items: list[ParameterDescriptor] = []

            def flush_group(cat: str | None, items: list[ParameterDescriptor]) -> None:
                if not items or not any(it.supported for it in items):
                    return
                label = cat if cat else "General"
                with dpg.collapsing_header(label=label, parent=parent, default_open=True):
                    table_policy = getattr(dpg, "mvTable_SizingStretchProp", None) or getattr(
                        dpg, "mvTable_SizingStretchSame", None
                    )
                    with dpg.table(header_row=False, policy=table_policy) as table:
                        left, right = self._label_value_ratio()
                        self._add_two_columns(left, right)
                        for it in items:
                            if not it.supported:
                                continue
                            self._create_row(table, it)

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
            flush_group(current_cat, group_items)

        # Runner controls は個別に上部で生成済み

        def _label_value_ratio(self) -> tuple[float, float]:
            """ラベル列と値列の比率（0.1..0.9）を返す。

            Returns
            -------
            tuple[float, float]
                左右の比率（left, right）。
            """
            left = 0.5
            try:
                left = float(self._layout.label_column_ratio)
            except Exception:
                logger.debug("invalid label_column_ratio; fallback to 0.5")
            left = 0.1 if left < 0.1 else (0.9 if left > 0.9 else left)
            right = max(0.1, 1.0 - left)
            return left, right

        def _add_two_columns(self, left: float, right: float) -> None:
            """2カラムのストレッチテーブル列を追加（古い環境では等分フォールバック）。

            Parameters
            ----------
            left, right
                各列のストレッチ比率。
            """
            try:
                dpg.add_table_column(
                    label="Parameter", width_stretch=True, init_width_or_weight=left
                )
                dpg.add_table_column(label="Value", width_stretch=True, init_width_or_weight=right)
            except Exception:
                logger.debug("table column stretch not supported; using equal columns")
                dpg.add_table_column(label="Parameter")
                dpg.add_table_column(label="Value")

        def _create_row(self, table: int | str, desc: ParameterDescriptor) -> None:
            # テーブル行（左: ラベル / 右: 入力）
            with dpg.table_row(parent=table) as row:
                label = desc.label or desc.id
                dpg.add_text(default_value=label, parent=row)
                self._create_widget(row, desc)
            # ツールチップは非採用（ポップアップは表示しない）

        # ---- Runner controls (Display) ----
        def force_set_rgb_u8(self, tag: int | str, rgb_u8: Sequence[int]) -> None:
            """ColorEdit に 0..255 の RGB 整数を強制反映する。"""
            try:
                r = int(rgb_u8[0])
                g = int(rgb_u8[1])
                b = int(rgb_u8[2])
                dpg.set_value(tag, [r, g, b])
            except Exception:
                logger.exception("force_set_rgb_u8 failed: tag=%s val=%s", tag, rgb_u8)

        def store_rgb01(self, pid: str, app_data: Any) -> None:
            """ColorEdit の値を 0..1 RGBA に正規化して Store に保存する。"""
            try:
                from util.color import normalize_color as _norm

                rgba = _norm(app_data)
                self._store.set_override(pid, rgba)
            except Exception:
                logger.exception("store_rgb01 failed: pid=%s val=%s", pid, app_data)

        def build_display_controls(self, parent: int | str, store: ParameterStore) -> None:
            from util.color import normalize_color as _norm

            try:
                from util.utils import load_config as _load_cfg
            except Exception:
                _load_cfg = lambda: {}

            cfg = _load_cfg() or {}
            canvas = cfg.get("canvas", {}) if isinstance(cfg, dict) else {}
            bg_raw = canvas.get("background_color", (1.0, 1.0, 1.0, 1.0))
            ln_raw = canvas.get("line_color", (0.0, 0.0, 0.0, 1.0))
            try:
                bgf = _norm(bg_raw)
            except Exception:
                bgf = (1.0, 1.0, 1.0, 1.0)
            try:
                lnf = _norm(ln_raw)
            except Exception:
                lnf = (0.0, 0.0, 0.0, 1.0)

            # Store に override があればそれを初期値に反映
            try:
                val = store.current_value("runner.background")
                if val is None:
                    val = store.original_value("runner.background")
                if val is not None:
                    bgf = _norm(val)
            except Exception:
                pass
            try:
                val = store.current_value("runner.line_color")
                if val is None:
                    val = store.original_value("runner.line_color")
                if val is not None:
                    lnf = _norm(val)
            except Exception:
                pass

            with dpg.collapsing_header(label="Display", default_open=True, parent=parent):
                # 背景
                dpg.add_text("Background")
                # ColorEdit: 小プレビューを保持し、クリック時のみピッカーを表示
                bg_picker = dpg.add_color_edit(
                    tag="runner.background",
                    default_value=[
                        int(round(float(bgf[0]) * 255)),
                        int(round(float(bgf[1]) * 255)),
                        int(round(float(bgf[2]) * 255)),
                    ],
                    no_label=True,
                    no_picker=False,
                    no_small_preview=False,
                    no_options=False,
                    no_alpha=True,
                    alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
                    display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                    display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                    input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                    alpha_bar=False,
                )
                # 既存の Store 値を 0..255 の RGB 整数で明示反映
                try:
                    r, g, b = float(bgf[0]), float(bgf[1]), float(bgf[2])
                    self.force_set_rgb_u8(
                        bg_picker, [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
                    )
                except Exception:
                    pass

                def _on_bg_picker(_s, app_data, _u):
                    self.store_rgb01("runner.background", app_data)

                dpg.configure_item(bg_picker, callback=_on_bg_picker)

                dpg.add_spacer(height=6)
                # 線色
                dpg.add_text("Line Color")
                ln_picker = dpg.add_color_edit(
                    tag="runner.line_color",
                    default_value=[
                        int(round(float(lnf[0]) * 255)),
                        int(round(float(lnf[1]) * 255)),
                        int(round(float(lnf[2]) * 255)),
                    ],
                    no_label=True,
                    no_picker=False,
                    no_small_preview=False,
                    no_options=False,
                    no_alpha=True,
                    alpha_preview=getattr(dpg, "mvColorEdit_AlphaPreviewHalf", 1),
                    display_mode=getattr(dpg, "mvColorEdit_DisplayRGB", 0),
                    display_type=getattr(dpg, "mvColorEdit_DisplayInt", 0),
                    input_mode=getattr(dpg, "mvColorEdit_InputRGB", 0),
                    alpha_bar=False,
                )
                try:
                    r, g, b = float(lnf[0]), float(lnf[1]), float(lnf[2])
                    self.force_set_rgb_u8(
                        ln_picker, [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
                    )
                except Exception:
                    pass

                def _on_ln_picker(_s, app_data, _u):
                    self.store_rgb01("runner.line_color", app_data)

                dpg.configure_item(ln_picker, callback=_on_ln_picker)

                # 初期同期（store→GUI）を明示呼び出し（ロード復帰が subscribe 前に済むため）
                try:
                    self.sync_display_from_store(store)
                except Exception:
                    logger.exception("initial store->gui sync failed")

        def sync_display_from_store(self, store: ParameterStore) -> None:
            """runner.* の色を Store から GUI に同期（0..255 RGB 強制）。"""
            ids = ["runner.background", "runner.line_color"]
            self._on_store_change(ids)

            # Descriptor の登録は ParameterManager.initialize() 側で行う

        def _create_widget(self, parent: int | str, desc: ParameterDescriptor) -> int:
            """Descriptor に応じたウィジェットを生成し、値変更コールバックを設定する。

            Parameters
            ----------
            parent
                親アイテムのタグまたはID。
            desc
                パラメータ記述子。

            Returns
            -------
            int
                生成されたアイテムのID。
            """
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

        # ---- ValueType 別ウィジェット ----
        def _create_bool(self, parent: int | str, desc: ParameterDescriptor, value: Any) -> int:
            """真偽値入力（チェックボックス）を生成する。"""
            return dpg.add_checkbox(
                tag=desc.id,
                parent=parent,
                label="",
                default_value=bool(value),
                callback=self._on_widget_change,
                user_data=desc.id,
            )

        def _create_enum(self, parent: int | str, desc: ParameterDescriptor, value: Any) -> int:
            """列挙入力（ラジオまたはコンボ）を生成する。"""
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
            """文字列入力（単行/複数行）を生成する。"""
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
            """ベクトル入力（3/4成分のスライダー群）を生成する。"""
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
                parent=parent, header_row=False, policy=dpg.mvTable_SizingStretchSame
            ) as vec_table:
                try:
                    with dpg.theme() as _vec_theme:
                        with dpg.theme_component(dpg.mvAll):
                            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 1, 0)
                    dpg.bind_item_theme(vec_table, _vec_theme)
                except Exception:
                    logger.debug("vector cell padding theme not supported")
                for _ in range(dim):
                    dpg.add_table_column(width_stretch=True)
                with dpg.table_row():
                    for i, suffix in enumerate(("x", "y", "z", "w")[:dim]):
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
                            dpg.set_item_width(slider_id, -1)
            return vec_table

        def _create_int(
            self, parent: int | str, desc: ParameterDescriptor, value: Any, hint: Any
        ) -> int:
            """整数スライダーを生成する。"""
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
            dpg.set_item_width(slider_id, -1)
            return slider_id

        def _create_float(
            self, parent: int | str, desc: ParameterDescriptor, value: Any, hint: Any
        ) -> int:
            """浮動小数スライダーを生成する。"""
            slider_id = dpg.add_slider_float(
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
            dpg.set_item_width(slider_id, -1)
            return slider_id

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
                    logger.exception("vector override failed: id=%s idx=%s", parent_id, idx)
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
                            if pid in ("runner.background", "runner.line_color"):
                                from util.color import normalize_color as _norm

                                r, g, b, a = _norm(value)
                                # ウィジェットへ整数RGBで強制反映
                                set_val_i = [
                                    int(round(r * 255)),
                                    int(round(g * 255)),
                                    int(round(b * 255)),
                                ]
                                dpg.set_value(pid, set_val_i)
                            else:
                                dpg.set_value(pid, value)
                        except Exception:
                            logger.exception("set_value failed: id=%s", pid)
                        continue
                    # 親ベクトル ID の場合は子スライダへ反映
                    try:
                        desc = self._store.get_descriptor(pid)
                    except Exception:
                        logger.exception("get_descriptor failed: id=%s", pid)
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
                            logger.exception("set_value failed: id=%s", tag)
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
                logger.exception("render_dearpygui_frame failed")

        # ---- テーマ（最小: パディングのみ） ----
        def _setup_theme(self) -> None:
            """テーマを構築してバインドする（スタイル→色の順）。"""
            try:
                with dpg.theme() as theme:
                    with dpg.theme_component(dpg.mvAll):
                        # スタイル適用（config優先、無ければ既定の最小スタイル）
                        if not self._apply_styles_from_config():
                            self._apply_default_styles()
                        # 色適用
                        self._apply_colors_from_config()
                dpg.bind_theme(theme)
            except Exception:
                # 失敗しても既定スタイルで継続
                logger.exception("setup_theme failed; continue with defaults")

        def _apply_default_styles(self) -> bool:
            """既定の最小スタイル（padding ベース）を適用する。"""
            pad = int(self._layout.padding)
            try:
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, pad, pad)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, pad, max(1, pad // 2))
                dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, pad, max(1, pad // 2))
                return True
            except Exception:
                logger.debug("default styles not supported")
                return False

        def _apply_styles_from_config(self) -> bool:
            """設定由来のスタイルを適用する（存在時）。"""
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
                except Exception:
                    logger.exception("add_theme_style failed: key=%s val=%s", key, value)

            for key, var in smap.items():
                if key in self._theme.style:
                    _add_style_value(var, key, self._theme.style[key])
            return True

        def _apply_colors_from_config(self) -> None:
            """設定由来の色を適用する（存在時）。"""
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
                # アクセントはスライダーの Grab に反映
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
                except Exception:
                    logger.exception("add_theme_color failed: key=%s val=%s", key, col)

        def _to_dpg_color(self, value: Any) -> Any:
            """RGBA 値を Dear PyGui の 0..255 RGBA 配列へ正規化する。

            - 受理: Hex 文字列 / (r,g,b[,a]) 0..1 / 0..255 配列
            - 返値: [r,g,b,a] 0..255
            """
            try:
                from util.color import to_u8_rgba as _to_u8_rgba

                r, g, b, a = _to_u8_rgba(value)
                return [int(r), int(g), int(b), int(a)]
            except Exception:
                # 旧形式（配列）の最小互換
                try:
                    if isinstance(value, (list, tuple)) and len(value) >= 4:
                        vals = [
                            float(value[0]),
                            float(value[1]),
                            float(value[2]),
                            float(value[3]),
                        ]
                        if all(0.0 <= v <= 1.0 for v in vals):
                            return [int(round(v * 255)) for v in vals]
                        return [int(round(v)) for v in vals]
                except Exception:
                    return None
                return None


__all__ = ["ParameterWindow"]
