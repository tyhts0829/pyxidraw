"""
どこで: `engine.ui.parameters` のウィンドウ実装。
何を: pyglet によるネイティブウィンドウ（イベント/描画/入力）と、ヘッドレス環境用スタブを切替提供。
なぜ: GUI 環境の有無に依存せず、同一 API でパラメータ編集/非編集を両立させるため。
"""

from __future__ import annotations

from typing import Iterable

try:  # Headless 環境（CI 等）では pyglet が利用できないことがある
    import pyglet
    from pyglet.window import key, mouse
except Exception:  # pragma: no cover - headless fallback
    pyglet = None
    key = None  # type: ignore[assignment]
    mouse = None  # type: ignore[assignment]

from .state import ParameterLayoutConfig, ParameterStore

if pyglet is None:  # pragma: no cover - tests/headless 用スタブ

    class _ParameterWindowStub:
        """GUI が無効な環境向けのダミーウィンドウ。"""

        def __init__(
            self,
            *,
            store: ParameterStore,
            layout: ParameterLayoutConfig,
            **_kwargs: int,
        ) -> None:
            self._store = store
            self._layout = layout

        def set_visible(self, _visible: bool) -> None:
            return None

        def close(self) -> None:
            return None

else:

    from .panel import ParameterPanel, ParameterWidget

    assert pyglet is not None
    assert key is not None
    assert mouse is not None

    _pyglet_clock = pyglet.clock
    _pyglet_window = pyglet.window.Window
    _pyglet_mouse = mouse
    _pyglet_key = key

    class _ParameterWindowImpl(_pyglet_window):
        """描画パラメータ編集用のサイドウィンドウ。"""

        def __init__(
            self,
            *,
            store: ParameterStore,
            layout: ParameterLayoutConfig,
            width: int = 420,
            height: int = 640,
        ) -> None:
            super().__init__(width=width, height=height, caption="Parameters", resizable=True)
            self._store = store
            self._layout = layout
            self._panel = ParameterPanel(store, layout)
            self._needs_refresh = True
            self._active_widget: ParameterWidget | None = None
            self._store.subscribe(self._on_store_change)
            _pyglet_clock.schedule_interval(self._tick, 1 / 30)
            self.set_location(40, 40)

        # --- 内部ヘルパ ---
        def _on_store_change(self, _ids: Iterable[str]) -> None:
            self._needs_refresh = True

        def _tick(self, _dt: float) -> None:
            if self._needs_refresh:
                self.refresh()
                self._needs_refresh = False
                self.invalid = True  # trigger redraw

        def refresh(self) -> None:
            descriptors = sorted(self._store.descriptors(), key=lambda d: d.id)
            self._panel.update_descriptors(descriptors)
            self._panel.layout(self.width, self.height)

        # --- pyglet イベント ---
        def on_draw(self) -> None:  # noqa: D401
            self.clear()
            self._panel.layout(self.width, self.height)
            self._panel.draw()

        def on_resize(self, width: int, height: int) -> None:  # noqa: D401
            super().on_resize(width, height)
            self._panel.layout(width, height)

        def on_mouse_press(
            self, x: float, y: float, button: int, modifiers: int
        ) -> None:  # noqa: D401
            widget = self._panel.hit_test(x, y)
            if widget is None:
                return
            self._active_widget = widget
            if button == _pyglet_mouse.LEFT:
                if modifiers & _pyglet_key.MOD_ACCEL:
                    widget.reset()
                else:
                    widget.begin_drag()
                    widget.drag_to(x, modifiers=modifiers)

        def on_mouse_drag(
            self,
            x: float,
            y: float,
            dx: float,
            dy: float,
            buttons: int,
            modifiers: int,
        ) -> None:  # noqa: D401
            if not (buttons & _pyglet_mouse.LEFT):
                return
            if self._active_widget is not None:
                self._active_widget.drag_to(x, modifiers=modifiers)

        def on_mouse_release(
            self, x: float, y: float, button: int, modifiers: int
        ) -> None:  # noqa: D401
            if button == _pyglet_mouse.LEFT and self._active_widget is not None:
                self._active_widget.end_drag()
                self._active_widget = None

        def on_mouse_scroll(
            self, x: float, y: float, scroll_x: float, scroll_y: float
        ) -> None:  # noqa: D401
            self._panel.scroll(-scroll_y)
            self.invalid = True

        def close(self) -> None:  # noqa: D401
            _pyglet_clock.unschedule(self._tick)
            self._store.unsubscribe(self._on_store_change)
            super().close()


if pyglet is None:
    ParameterWindow = _ParameterWindowStub
else:
    ParameterWindow = _ParameterWindowImpl

__all__ = ["ParameterWindow"]
