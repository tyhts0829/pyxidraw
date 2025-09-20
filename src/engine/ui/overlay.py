"""
どこで: `engine.ui` の HUD 表示モジュール。
何を: MetricSampler のキー/値ペアを pyglet の Label でオーバーレイ描画する。
なぜ: 実行時メトリクスを即座に可視化し、デバッグ/チューニングのフィードバックを高めるため。
"""

from __future__ import annotations

import time
from typing import Literal

import pyglet
from pyglet.window import Window

from ..core.tickable import Tickable
from .monitor import MetricSampler


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
        self._y_cursor = 10
        self._color = color
        self._font = "HackGenConsoleNF-Regular"
        self.font_size = font_size
        # --- messages/progress ---
        self._messages: list[tuple[str, float, Literal["info", "warn", "error"]]] = []
        self._progress: dict[str, tuple[int, int]] = {}

    # -------- Tickable --------
    def tick(self, dt: float) -> None:
        # ラベル生成 & 更新
        for key, txt in self.sampler.data.items():
            if key not in self._labels:
                self._labels[key] = pyglet.text.Label(
                    text="",
                    x=10,
                    y=self._y_cursor,
                    anchor_x="left",
                    anchor_y="bottom",
                    font_name=self._font,
                    font_size=self.font_size,
                    color=self._color,
                )
                self._y_cursor += 18
            self._labels[key].text = f"{key} : {txt}"
        # メッセージの有効期限を掃除
        now = time.monotonic()
        self._messages = [m for m in self._messages if m[1] > now]

    # -------- draw --------
    def draw(self) -> None:
        # 既存メトリクス
        for lab in self._labels.values():
            lab.draw()
        # 進捗 (%表示)
        y = self._y_cursor + 8
        for key, (done, total) in self._progress.items():
            pct = 0 if total <= 0 else int(round(100 * done / max(1, total)))
            lbl = pyglet.text.Label(
                text=f"{key} : {pct}%",
                x=10,
                y=y,
                anchor_x="left",
                anchor_y="bottom",
                font_name=self._font,
                font_size=self.font_size,
                color=self._color,
            )
            lbl.draw()
            y += 18
        # 一時メッセージ（最後に重ねて表示）
        for text, _expire, level in self._messages:
            rgba = {
                "info": (0, 0, 0, 200),
                "warn": (200, 120, 0, 230),
                "error": (200, 0, 0, 230),
            }[level]
            lbl = pyglet.text.Label(
                text=text,
                x=10,
                y=self.window.height - 20,
                anchor_x="left",
                anchor_y="top",
                font_name=self._font,
                font_size=self.font_size + 2,
                color=rgba,
            )
            lbl.draw()

    # ---- public helpers ----
    def show_message(
        self, text: str, level: Literal["info", "warn", "error"] = "info", timeout_sec: float = 3
    ) -> None:
        expire = time.monotonic() + max(0.1, float(timeout_sec))
        self._messages.append((text, expire, level))

    def set_progress(self, key: str, done: int, total: int) -> None:
        self._progress[key] = (int(done), int(total))

    def clear_progress(self, key: str) -> None:
        self._progress.pop(key, None)
