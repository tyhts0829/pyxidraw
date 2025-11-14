import numpy as np

from api import E, G, run

A5 = (148, 210)


def draw(t: float):
    WHITE_SPACE = 0.1
    g1 = (
        G.polygon()
        .rotate(z=-np.pi / 4, center=(0, 0, 0))
        .scale(A5[0] - WHITE_SPACE * 2, A5[1] - WHITE_SPACE * 2, 1)
        .translate(A5[0] / 2, A5[1] / 2, 0)
    )
    p1 = E.pipeline.affine()
    return p1(g1)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=6,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        show_hud=True,
    )
