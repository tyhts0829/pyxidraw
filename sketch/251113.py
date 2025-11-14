from api import E, G, run


def draw(t: float):
    g2 = G.polygon().scale(100, 100, 1)
    p2 = E.pipeline.label(uid="Sphere_Large").affine().fill()

    g3 = G.text().scale(5, 5, 1)
    p3 = E.pipeline.label(uid="Text_Large").translate().affine().fill()
    g4 = G.text().scale(1, 1, 1)
    p4 = E.pipeline.label(uid="Text_Small").translate().affine()

    g1 = G.polygon().scale(10, 10, 1)
    p1 = E.pipeline.label(uid="Sphere_Small").affine().fill()

    g5 = G.text().scale(2, 2, 1)
    p5 = E.pipeline.label(uid="Text_Mid").translate().affine().fill()

    return p1(g1), p2(g2), p3(g3), p4(g4), p5(g5)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=6,
        use_midi=True,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
        show_hud=False,
    )
