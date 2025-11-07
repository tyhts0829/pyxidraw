from __future__ import annotations

import os

from api import E, G, lfo, run
from engine.core.geometry import Geometry

# PXD_DEBUG_PREFIX_CACHEをTrueに
os.environ["PXD_IBO_FREEZE_ENABLED"] = "1"
CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float) -> Geometry:
    """デモ描画関数（MIDI は `api.cc` で制御）。"""
    geo = G.polyhedron()
    # pipe = E.pipeline.affine().partition().fill().subdivide().displace()
    e_pipe1 = (
        E.pipeline.affine()
        .scale(scale=(80, 80, 80))
        .translate()
        .fill()
        .subdivide()
        .displace(t_sec=t)
        # .mirror()
    )

    return e_pipe1(geo)


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVAS_SIZE, CANVAS_SIZE),
        render_scale=3,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        # show_hud=False,
    )
