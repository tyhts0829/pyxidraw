"""
どこで: `engine.core` の描画ウィンドウ薄ラッパ。
何を: Pyglet Window（MSAA/背景クリア/中心配置）と描画コールバック登録を提供。
なぜ: レンダラ/ジオメトリ層から GUI 依存を切り離し、最小インターフェイスで統一するため。

使用例:
    win = RenderWindow(1280, 720, bg_color=(1, 1, 1, 1))

    def draw_scene():
        renderer.draw(...)

    win.add_draw_callback(draw_scene)
    pyglet.app.run()
"""

from typing import Callable

import pyglet
from pyglet.gl import Config, glClearColor


class RenderWindow(pyglet.window.Window):
    def __init__(
        self,
        width: int,
        height: int,
        *,
        bg_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    ):
        """ウィンドウを生成する。

        引数:
            width: ウィンドウ幅（ピクセル）。
            height: ウィンドウ高さ（ピクセル）。
            bg_color: 背景色 RGBA（0.0〜1.0）。
        """
        # 線描画を滑らかにするために MSAA を有効化
        config = Config(double_buffer=True, sample_buffers=1, samples=4, vsync=True)
        super().__init__(width=width, height=height, caption="Pyxidraw", config=config)
        self._bg_color = bg_color
        self._draw_callbacks: list[Callable[[], None]] = []

        # ウィンドウ初期化後、最初の描画タイミングで中央配置を行うフラグ
        self._should_center = True

    def add_draw_callback(self, func: Callable[[], None]) -> None:
        """
        `on_draw` 中に呼び出す描画関数を登録する。

        - 関数は引数を取らず、副作用で描画を行うこと。
        - 登録順に呼び出される。
        """
        self._draw_callbacks.append(func)

    def on_draw(self):  # Pyglet 既定のイベント名
        """ウィンドウ描画イベントハンドラ。登録された描画コールバックを呼び出す。"""
        # フレームカウンタは保持しない（不要な状態を削減）
        r, g, b, a = self._bg_color
        glClearColor(r, g, b, a)
        self.clear()
        for cb in self._draw_callbacks:
            cb()

    # ---- helpers ----
    def set_background_color(self, rgba: tuple[float, float, float, float]) -> None:
        """背景色 RGBA(0–1) を更新する。次フレームから反映。"""
        try:
            r, g, b, a = rgba
            self._bg_color = (float(r), float(g), float(b), float(a))
        except Exception:
            # 不正入力は無視
            pass
