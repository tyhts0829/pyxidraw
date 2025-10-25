from __future__ import annotations

from api import E, G, lfo, run
from engine.core.geometry import Geometry

CANVAS_SIZE = 400

osc = lfo(wave="sine", freq=0.1, octaves=4, persistence=0.5, lacunarity=2.0)


def draw(t: float) -> Geometry:
    """デモ描画関数（MIDI は `api.cc` で制御）。"""
    sphere = (
        G.sphere()
        .scale()
        .translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
        # G.text(em_size_mm=150).translate(CANVAS_SIZE // 2, CANVAS_SIZE // 2, 0)
    )
    # pipe = E.pipeline.affine().partition().fill().subdivide().displace()
    e_pipe1 = E.pipeline.affine().fill().subdivide().displace(t_sec=t * 1)
    text = G.text(text="PyxiDraw", font="Cappadocia")
    e_pipe2 = E.pipeline.affine().translate().fill()

    return e_pipe1(sphere) + e_pipe2(text)


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
