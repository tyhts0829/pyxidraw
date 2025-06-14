from typing import Callable

import pyglet


class RenderWindow(pyglet.window.Window):
    def __init__(self, width: int, height: int, on_draw_cb: Callable[[], None]):
        super().__init__(width=width, height=height, caption="PyLineSketch")
        self._on_draw_cb = on_draw_cb

    def on_draw(self):  # Pyglet 既定のイベント名
        self.clear()
        self._on_draw_cb()  # Renderer.draw() を呼ぶだけ
