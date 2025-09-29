"""
ユーザー定義 Effect を `from api import effect` で登録し、パイプラインに組み込む最小サンプル。

実行:
    python effect.py

備考:
    - 依存が未導入の場合は、プロジェクトルートで以下を実行してください。
        python3.10 -m venv .venv && source .venv/bin/activate \
        && pip install -U pip && pip install -e .[dev]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))
from api import E, G, cc, effect, run_sketch
from engine.core.geometry import Geometry


@effect("my_shift")
def my_shift(g: Geometry, *, dx: float = 20.0, dy: float = 0.0) -> Geometry:
    """平行移動エフェクト（デモ用）。"""
    return g.translate(dx, dy, 0.0)


def user_draw(t: float) -> Geometry:  # noqa: D401 - 簡潔
    # ベース形状（正多角形）
    base = G.polygon(n_sides=5).scale(120, 120, 1).translate(150, 120, 0)

    # 時間で x 方向にふらつかせる（CC#1 で強さを加算）
    dx = 25.0 * float(np.sin(t * 1.4)) + 50.0 * (cc[1] * 1.0)

    pipe = E.pipeline.my_shift(dx=dx, dy=0.0).build()
    return pipe(base)


if __name__ == "__main__":
    run_sketch(user_draw, canvas_size="A5", render_scale=4, fps=60)
