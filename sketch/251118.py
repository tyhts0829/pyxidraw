from api import E, G, run

A5 = (148, 210)


def draw(t: float):
    N = 3
    geos = []
    for i in range(N):
        g = (
            G.polygon()
            .scale(4 * (i + 1), 4 * (i + 1), 4 * (i + 1))
            .translate(A5[0] / 2, A5[1] / 2, 0)
        )
        p = E.label(uid="poly_effect").affine().fill()
        geos.append(p(g))
    return geos


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
