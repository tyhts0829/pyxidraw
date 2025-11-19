from api import E, G, run

A5 = (148, 210)


def draw(t: float):
    g = G.line().scale(100, 100, 1).translate(A5[0] / 2, A5[1] / 2, 0)
    p = E.affine().subdivide().displace().scale(scale=(1, 1, 0)).offset().offset().offset()
    return p(g)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=10,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        show_hud=True,
    )
