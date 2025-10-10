from __future__ import annotations

import sys
from pathlib import Path

# src/ レイアウトからシンプルに import できるよう、`python main.py` 実行時にパスを補助
sys.path.insert(0, str((Path(__file__).resolve().parent / "src")))

import os

import numpy as np

from api import E, G, cc, lfo, run  # type: ignore  # after sys.path tweak
from engine.core.geometry import Geometry  # type: ignore

PXD_FILL_DEBUG = 1
os.environ["PXD_FILL_DEBUG"] = "1"

CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float) -> Geometry:
    """デモ描画関数（MIDI は `api.cc` で制御）。"""
    base = (
        # G.polyhedron(polygon_index=int(cc[1] * 6))
        # .scale(80 + cc[2] * 150)
        # .translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
        G.text(em_size_mm=cc[1] * 200).translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    )
    pipe = (
        E.pipeline.affine(angles_rad=(cc[3] * np.pi, cc[4] * np.pi, cc[5] * np.pi))
        # .partition()
        .fill(density=(cc[6] * 200, cc[6] * 200), angle_rad=(np.pi / 3, 2 * np.pi / 3, np.pi / 2))
        .subdivide()
        .displace()
        # .displace(t_sec=osc(t * cc[7]) * 100 * cc[8])
    )
    return pipe(base)


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
        render_scale=4.5,
        use_midi=True,
        use_parameter_gui=True,
        workers=6,
        line_thickness=0.001,
    )
