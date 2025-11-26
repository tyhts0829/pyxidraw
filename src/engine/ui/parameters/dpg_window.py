"""
どこで: `engine.ui.parameters` の Dear PyGui 実装。
何を: ParameterStore の Descriptor からパラメータウィンドウを生成し、表示/非表示/終了を扱う。
なぜ: ウィンドウ寿命管理と DPG ドライバ制御だけを担い、レイアウト/テーマは別クラスへ委譲して単純化するため。
"""

from __future__ import annotations

import logging
from threading import Thread
from typing import Any, Iterable

import dearpygui.dearpygui as dpg  # type: ignore

from .dpg_window_content import ParameterWindowContentBuilder
from .dpg_window_theme import ParameterWindowThemeManager
from .state import ParameterDescriptor, ParameterLayoutConfig, ParameterStore, ParameterThemeConfig

# タグ定数
ROOT_TAG = "__pxd_param_root__"
STAGE_TAG = "__pxd_param_stage__"

logger = logging.getLogger("engine.ui.parameters.dpg")


class ParameterWindowBase:
    """パラメータウィンドウの最小インタフェース。"""

    def set_visible(self, _visible: bool) -> None:  # pragma: no cover
        return None

    def close(self) -> None:  # pragma: no cover
        return None

    def mount(self, _descriptors: list[ParameterDescriptor]) -> None:  # pragma: no cover
        return None


class ParameterWindow(ParameterWindowBase):  # type: ignore[override]
    """Dear PyGui によるパラメータウィンドウ実装。"""

    def __init__(
        self,
        *,
        store: ParameterStore,
        layout: ParameterLayoutConfig,
        width: int = 420,
        height: int = 640,
        title: str = "Parameters",
        theme: ParameterThemeConfig | None = None,
        auto_show: bool = True,
    ) -> None:
        self._store = store
        self._layout = layout
        self._width = width
        self._height = height
        self._title = title
        self._theme_cfg = theme
        self._visible = False
        self._driver: Any | None = None
        self._closing: bool = False
        self._store_listener: Any | None = None

        self._theme_mgr = ParameterWindowThemeManager(
            layout=self._layout, theme_cfg=self._theme_cfg
        )
        self._content = ParameterWindowContentBuilder(
            store=self._store,
            layout=self._layout,
            theme_mgr=self._theme_mgr,
        )

        dpg.create_context()
        self._viewport = dpg.create_viewport(
            title=self._title,
            width=self._width,
            height=self._height,
        )
        dpg.setup_dearpygui()

        try:
            self._theme_mgr.setup_fonts()
        except Exception:
            # フォント設定失敗時も既定フォントで続行
            pass

        self._content.build_root_window(ROOT_TAG, self._title)
        self._theme_mgr.setup_theme()

        # 初期マウントと購読（登録順を維持してマウント）
        self.mount(self._store.descriptors())

        def _on_store_change_wrapper(ids: Iterable[str]) -> None:
            try:
                self._content.on_store_change(ids)
                self._content.sync_style_from_store()
            except Exception:
                logger.exception("store change handling failed")

        self._store_listener = _on_store_change_wrapper
        self._store.subscribe(self._store_listener)

        if auto_show:
            dpg.show_viewport()
            self._visible = True
            self._start_driver()

    def set_visible(self, visible: bool) -> None:
        if visible and not self._visible:
            dpg.show_viewport()
            self._visible = True
            self._start_driver()
        elif not visible and self._visible:
            dpg.hide_viewport()
            self._visible = False
            self._stop_driver()

    def close(self) -> None:
        # 閉鎖フラグを最初に立て、以降の _tick を無害化
        self._closing = True
        # 先に購読解除して、destroy 後の dpg.* 呼び出し経路を断つ
        try:
            if self._store_listener is not None:
                self._store.unsubscribe(self._store_listener)
        except Exception:
            pass

        # ドライバ停止 → ビューポート非表示 → コンテキスト破棄（順序厳守）
        self._stop_driver()
        try:
            dpg.stop_dearpygui()
        except Exception:
            pass
        try:
            if self._visible:
                dpg.hide_viewport()
        except Exception:
            pass
        try:
            dpg.destroy_context()
        except Exception:
            pass

    def mount(self, descriptors: list[ParameterDescriptor]) -> None:
        with dpg.stage(tag=STAGE_TAG):
            self._content.mount_descriptors(ROOT_TAG, descriptors)
        dpg.unstage(STAGE_TAG)

    # ---- 互換用 ----
    def _build_root_window(self) -> None:  # pragma: no cover
        """後方互換用のダミー（旧コードからの直接呼び出しのみ想定）。"""
        self._content.build_root_window(ROOT_TAG, self._title)

    # ---- internal: drivers ----
    def _tick(self, _dt: float) -> None:  # noqa: ANN001
        if self._closing:
            return
        try:
            is_run = getattr(dpg, "is_dearpygui_running", None)
            if callable(is_run) and not bool(is_run()):
                self._closing = True
                self._stop_driver()
                return
            is_vp_ok = getattr(dpg, "is_viewport_ok", None)
            if callable(is_vp_ok) and not bool(is_vp_ok()):
                self._closing = True
                self._stop_driver()
                return
        except Exception:
            # 確認に失敗した場合は続行（後段 try で保護）
            pass
        try:
            dpg.render_dearpygui_frame()
        except Exception:
            logger.exception("render_dearpygui_frame failed")

    def _start_driver(self) -> None:
        if self._driver is not None:
            return
        # Prefer pyglet clock on main thread; fallback to background thread loop
        try:
            import pyglet  # type: ignore

            interval = 1.0 / 60.0
            pyglet.clock.schedule_interval(self._tick, interval)
            self._driver = ("pyglet", self._tick)
            logger.debug("ParameterWindow: pyglet driver started")
        except ImportError:
            try:
                t = Thread(target=dpg.start_dearpygui, name="DPGLoop", daemon=True)
                t.start()
                self._driver = ("thread", t)
                logger.debug("ParameterWindow: thread driver started")
            except Exception:
                logger.exception("failed to start any driver")

    def _stop_driver(self) -> None:
        drv = self._driver
        self._driver = None
        if drv is None:
            return
        kind, handle = drv
        try:
            if kind == "pyglet":
                import pyglet  # type: ignore

                pyglet.clock.unschedule(handle)  # type: ignore[arg-type]
            elif kind == "thread":
                try:
                    dpg.stop_dearpygui()
                except Exception:
                    pass
        except Exception:
            logger.exception("failed to stop driver: %s", kind)
