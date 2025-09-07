"""
レンダリング用ウィンドウ（Pyglet ラッパー）

本モジュールは、Pyglet の `Window` を軽量にラップし、次の機能を提供する。

- アンチエイリアス（MSAA）付きのウィンドウ生成（線描画の見栄え向上）。
- 背景色の一括クリア（`glClearColor`）。
- `on_draw` フェーズで実行される描画コールバックの登録・順次実行。
- 初回描画時にプライマリスクリーン中央へ自動配置（作業開始時の視認性向上）。

設計方針:
- UI 依存の詳細（ウィンドウ生成/配置）は本クラスに集約し、レンダラや
  ジオメトリ層から切り離す（責務分離）。
- コールバックはシンプルな「引数なし・副作用で描画する関数」を想定し、
  登録順に実行するだけの最小インターフェイスとする。

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

        Args:
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

    def center_on_screen(self) -> None:
        """プライマリスクリーンの中央へウィンドウを配置する。"""
        # ディスプレイとデフォルトスクリーンを取得
        display = pyglet.display.get_display()
        screen = display.get_default_screen()

        # 中央位置を算出
        x = (screen.width - self.width) // 2
        y = (screen.height - self.height) // 2

        # ウィンドウ位置を設定
        self.set_location(x, y)

    def on_draw(self):  # Pyglet 既定のイベント名
        # 初回描画時のみウィンドウを中央へ移動
        if hasattr(self, "_should_center") and self._should_center:
            self._should_center = False
            self.center_on_screen()

        r, g, b, a = self._bg_color
        glClearColor(r, g, b, a)
        self.clear()
        for cb in self._draw_callbacks:
            cb()
