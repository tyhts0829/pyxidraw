from __future__ import annotations

import sys
from pathlib import Path

# src/ レイアウトからシンプルに import できるよう、`python main.py` 実行時にパスを補助
sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))

import numpy as np

from api import E, G, cc, run  # type: ignore  # after sys.path tweak
from engine.core.geometry import Geometry  # type: ignore

CANVAS_SIZE = 400


def draw(t: float) -> Geometry:
    """デモ描画関数（MIDI は `api.cc` で制御）。"""
    base = (
        G.polyhedron(polygon_index=int(cc[1] * 6))
        .scale(cc[2] * 500)
        .translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    )
    pipe = (
        E.pipeline.affine(angles_rad=(cc[3] * np.pi, cc[4] * np.pi, cc[5] * np.pi))
        .fill(density=cc[6] * 200)
        .subdivide()
        .displace(t_sec=t * cc[7] * 2)
        .build()
    )
    return pipe(base)


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
        render_scale=3,
        use_midi=True,
        use_parameter_gui=True,
        workers=3,
        line_thickness=0.001,
    )
