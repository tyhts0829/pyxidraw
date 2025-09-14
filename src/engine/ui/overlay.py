from __future__ import annotations

import pyglet
from pyglet.window import Window

from ..core.tickable import Tickable
from ..monitor.sampler import MetricSampler


class OverlayHUD(Tickable):
    """MetricSampler が溜めた文字列を pyglet Label で描画する。"""

    def __init__(
        self,
        window: Window,
        sampler: MetricSampler,
        font_size: int = 8,
        color=(0, 0, 0, 155),
    ):
        self.window = window
        self.sampler = sampler
        self._labels: dict[str, pyglet.text.Label] = {}
        self._y_cursor = window.height - 10
        self._color = color
        self._font = "HackGenConsoleNF-Regular"
        self.font_size = font_size

    # -------- Tickable --------
    def tick(self, dt: float) -> None:
        # ラベル生成 & 更新
        for key, txt in self.sampler.data.items():
            if key not in self._labels:
                self._y_cursor -= 18
                self._labels[key] = pyglet.text.Label(
                    text="",
                    x=10,
                    y=self._y_cursor,
                    anchor_x="left",
                    anchor_y="top",
                    font_name=self._font,
                    font_size=self.font_size,
                    color=self._color,
                )
            self._labels[key].text = f"{key} : {txt}"

    # -------- draw --------
    def draw(self) -> None:
        for lab in self._labels.values():
            lab.draw()
