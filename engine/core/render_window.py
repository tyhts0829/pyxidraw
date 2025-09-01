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
        
        # Center the window on the screen after it's fully initialized
        self._should_center = True

    def add_draw_callback(self, func: Callable[[], None]) -> None:
        """
        Add a function to be called during on_draw.
        The function should take no arguments and perform rendering.
        Callbacks are called in the order added.
        """
        self._draw_callbacks.append(func)
    
    def center_on_screen(self) -> None:
        """Center the window on the primary screen."""
        # Get the display and screen
        display = pyglet.display.get_display()
        screen = display.get_default_screen()
        
        # Calculate center position
        x = (screen.width - self.width) // 2
        y = (screen.height - self.height) // 2
        
        # Set the window location
        self.set_location(x, y)

    def on_draw(self):  # Pyglet 既定のイベント名
        # Center window on first draw if needed
        if hasattr(self, '_should_center') and self._should_center:
            self._should_center = False
            self.center_on_screen()
        
        r, g, b, a = self._bg_color
        glClearColor(r, g, b, a)
        self.clear()
        for cb in self._draw_callbacks:
            cb()
