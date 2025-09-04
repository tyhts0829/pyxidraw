#!/usr/bin/env python3
"""
クイックスタート: 最小のスケッチ（関数エフェクト + パイプライン）
"""

import os
import sys

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
    base = G.sphere(subdivisions=0.4).scale(90, 90, 90).translate(150, 150, 0)
    pipeline = (
        E.pipeline
        .ripple(amplitude=0.12, frequency=0.25)
        .rotate(angles_rad=(0.0, t * 0.01, 0.0))
        .build()
    )
    return pipeline(base)


if __name__ == "__main__":
    # MIDIは不要。300mm 四方のキャンバス。
    run(draw, canvas_size=(300, 300))

