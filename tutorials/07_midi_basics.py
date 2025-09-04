#!/usr/bin/env python3
"""
チュートリアル 07: MIDI 入門（最小）

目的:
- CC 値でパラメータを直接コントロールする、いちばんシンプルな例。

割り当て:
- CC#1 → ノイズ強度（mm）
- CC#2 → 回転速度（係数）
"""

import os
import sys
import math

# リポジトリルートを import path の先頭に追加
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
try:
    while REPO_ROOT in sys.path:
        sys.path.remove(REPO_ROOT)
except ValueError:
    pass
sys.path.insert(0, REPO_ROOT)

from api import E, G, run


def draw(t, cc):
    # CC は 0..1 の正規化値。
    noise = cc.get(1, 0.0) * 0.6  # mm
    speed = 0.2 + cc.get(2, 0.5)  # 回転速度係数

    base = G.torus(major_radius=55, minor_radius=22).translate(200, 200, 0)

    pipeline = (
        E.pipeline
        .displace(amplitude_mm=noise, t_sec=t)
        .rotate(angles_rad=(0.0, t * speed * 0.8, 0.0))
        .build()
    )
    return pipeline(base)


if __name__ == "__main__":
    # MIDI を有効にして起動（依存/接続がなければ内部で安全にフォールバック）
    run(draw, canvas_size=(300, 300), use_midi=True)
