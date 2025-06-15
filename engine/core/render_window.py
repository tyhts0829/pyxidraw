from typing import Callable

import pyglet
from pyglet.gl import Config, glClearColor


class RenderWindow(pyglet.window.Window):
    def __init__(self, width: int, height: int, *, bg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)):
        # Enable MSAA for smooth line rendering
        config = Config(double_buffer=True, sample_buffers=1, samples=4, vsync=True)
        super().__init__(width=width, height=height, caption="Pyxidraw", config=config)
        self._bg_color = bg_color
        self._draw_callbacks: list[Callable[[], None]] = []

    def add_draw_callback(self, func: Callable[[], None]) -> None:
        """
        Add a function to be called during on_draw.
        The function should take no arguments and perform rendering.
        Callbacks are called in the order added.
        """
        self._draw_callbacks.append(func)

    def on_draw(self):  # Pyglet 既定のイベント名
        r, g, b, a = self._bg_color
        glClearColor(r, g, b, a)
        self.clear()
        for cb in self._draw_callbacks:
            cb()
