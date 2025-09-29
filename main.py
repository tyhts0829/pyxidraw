from __future__ import annotations

import sys
from pathlib import Path

# src/ レイアウトからシンプルに import できるよう、`python main.py` 実行時にパスを補助
sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))

from api import E, G, cc, run  # type: ignore  # after sys.path tweak
from engine.core.geometry import Geometry  # type: ignore

CANVAS_SIZE = 400


def draw(t: float) -> Geometry:
    """デモ描画関数（MIDI は `api.cc` で制御）。"""
    base = G.polyhedron().scale(cc[1] * 500).translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    pipe = E.pipeline.affine().extrude().subdivide().displace().build()
    return pipe(base)


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
        render_scale=5,
        use_midi=True,
        use_parameter_gui=True,
        workers=6,
        line_thickness=0.001,
    )
