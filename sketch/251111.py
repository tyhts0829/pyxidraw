from __future__ import annotations

from api import E, G, lfo, run
from engine.core.geometry import Geometry

# PXD_DEBUG_PREFIX_CACHEをTrueに
CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float) -> Geometry:
    ring = G.polygon()
    p2 = E.pipeline.scale(scale=(80, 80, 1.0)).affine().displace().fill().rotate()
    return p2(ring)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=6,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        # show_hud=False,
    )
