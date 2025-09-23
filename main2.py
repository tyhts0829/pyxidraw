from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

# src/ レイアウトからシンプルに import できるよう、`python main.py` 実行時にパスを補助
sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))

from api import E, G, run  # type: ignore  # after sys.path tweak
from engine.core.geometry import Geometry  # type: ignore

CANVAS_SIZE = 400

pipe = E.pipeline.affine(scale=(2, 2, 2)).offset().build()


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    """デモ描画関数（MIDI 未接続でも安全に動作）。"""

    def c(i: int, default: float = 0.0) -> float:
        return float(cc.get(i, default))

    t = t * c(9, 1.0) * 10
    poly = G.polyhedron().scale(400 * c(8, 0.25)).translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    return pipe(poly)


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
        render_scale=3,
        use_midi=True,
        use_parameter_gui=True,
        workers=6,
        line_thickness=0.001,
    )
