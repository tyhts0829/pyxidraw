"""
ユーザー定義 Shape を `from api import shape` で登録し、描画する最小サンプル。

使い方:
    # 仮想環境などを用意した上で
    python user_shape_demo.py

    - ウィンドウが開かない/依存が足りない場合は、`pip install -e .[dev]` を実行してください。
    - 関連依存: numpy, pyglet, moderngl など。

ポイント:
    - デコレータは API ルートから: `from api import shape`
    - 実行時は `api.run_sketch` に user_draw を渡して描画します。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))
from api import G, cc, run_sketch, shape
from engine.core.geometry import Geometry


@shape("my_star")
def my_star(*, points: int = 7, r: float = 80.0, inner: float = 0.5) -> Geometry:
    """正多角形の頂点を結んだ簡易スター（星形）。"""
    n = int(points) * 2
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False, dtype=np.float32)
    rr = np.where(np.arange(n) % 2 == 0, float(r), float(r) * float(inner))
    xy = np.c_[rr * np.cos(th), rr * np.sin(th)]
    return Geometry.from_lines([xy])


def user_draw(t: float) -> Geometry:  # noqa: D401 - 簡潔
    # 時間で半径をゆっくり変調（視覚変化）。CC#1 でも上書き可能。
    base_r = 80.0 + 10.0 * np.sin(t * 0.8)
    r = base_r + 40.0 * (cc[1] * 1.0)
    return G.my_star(points=7, r=r, inner=0.5)


if __name__ == "__main__":
    run_sketch(user_draw, canvas_size="A5", render_scale=4, fps=60)
