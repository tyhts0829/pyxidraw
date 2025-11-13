from api import E, G, run


def draw(t: float):
    g2 = G.polygon().scale(100, 100, 1)
    p2 = E.pipeline.label(uid="Sphere_Large").affine().fill()

    g3 = G.text().scale(5, 5, 1)
    p3 = E.pipeline.label(uid="Text_Large").translate().affine().fill()
    g4 = G.text().scale(1, 1, 1)
    p4 = E.pipeline.label(uid="Text_Small").translate().affine()

    g1 = G.polygon().scale(10, 10, 1)
    p1 = E.pipeline.label(uid="Sphere_Small").affine().fill().clip(outline=p4(g4))

    return p1(g1) + p2(g2) + p3(g3)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=6,
        use_midi=False,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
    )
