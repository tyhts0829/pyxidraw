from typing import Callable, Optional

import pyglet


class RenderWindow(pyglet.window.Window):
    def __init__(self, width: int, height: int, on_draw_cb: Optional[Callable[[], None]] = None):
        super().__init__(width=width, height=height, caption="PyLineSketch")
        if on_draw_cb is None:
            self._draw_callback = lambda: None
        else:
            self._draw_callback = on_draw_cb

    def set_draw_callback(self, func: Callable[[], None]) -> None:
        self._draw_callback = func

    def on_draw(self):  # Pyglet 既定のイベント名
        self.clear()
        self._draw_callback()  # Renderer.draw() を呼ぶだけ
