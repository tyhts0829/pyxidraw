"""
どこで: `engine.ui.hud` の HUD 表示モジュール。
何を: MetricSampler のキー/値ペアを pyglet の Label でオーバーレイ描画する。
なぜ: 実行時メトリクスを即座に可視化し、デバッグ/チューニングのフィードバックを高めるため。
"""

from __future__ import annotations

import time
from typing import Literal

import pyglet
from pyglet.window import Window

from ...core.tickable import Tickable
from .config import HUDConfig
from .sampler import MetricSampler


class OverlayHUD(Tickable):
    """MetricSampler が溜めた文字列を pyglet Label で描画する。"""

    def __init__(
        self,
        window: Window,
        sampler: MetricSampler,
        *,
        config: HUDConfig | None = None,
        font_size: int = 8,
        color=(0, 0, 0, 155),
    ):
        self.window = window
        self.sampler = sampler
        self._config = config or HUDConfig()
        self._labels: dict[str, pyglet.text.Label] = {}
        self._color = color
        self._font = "HackGenConsoleNF-Regular"
        self.font_size = font_size
        # --- messages/progress ---
        self._messages: list[tuple[str, float, Literal["info", "warn", "error"]]] = []
        self._progress: dict[str, tuple[int, int]] = {}

    # -------- Tickable --------
    def tick(self, dt: float) -> None:
        # ラベル生成 & 更新（順序は HUDConfig に従う。未知キーは末尾に追加）
        desired = list(self._config.resolved_order())
        # sampler.data に存在するが order に無いキーを後置
        for k in self.sampler.data.keys():
            if k not in desired:
                desired.append(k)

        # 再配置のため、新しいラベル辞書を構築
        new_labels: dict[str, pyglet.text.Label] = {}
        start_y = 10
        line_h = 18
        for i, key in enumerate(desired):
            if key not in self.sampler.data:
                continue
            y = start_y + i * line_h
            lab = self._labels.get(key)
            if lab is None:
                lab = pyglet.text.Label(
                    text="",
                    x=10,
                    y=y,
                    anchor_x="left",
                    anchor_y="bottom",
                    font_name=self._font,
                    font_size=self.font_size,
                    color=self._color,
                )
            else:
                # 位置を更新
                lab.y = y
            lab.text = f"{key} : {self.sampler.data.get(key, '')}"
            new_labels[key] = lab
        # 不要になったラベルは破棄（pyglet 側のリソース管理は任せる）
        self._labels = new_labels
        # メッセージの有効期限を掃除
        now = time.monotonic()
        self._messages = [m for m in self._messages if m[1] > now]

    # -------- draw --------
    def draw(self) -> None:
        # 既存メトリクス
        for lab in self._labels.values():
            lab.draw()
        # 進捗 (%表示)
        y = 10 + len(self._labels) * 18 + 8
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
