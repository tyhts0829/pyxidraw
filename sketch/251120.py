from api import E, G, L, run

A5 = (148, 210)


def draw(t: float):
    g1 = G.polygon().scale(40, 40, 40).translate(A5[0] / 2, A5[1] / 2, 0)
    p1 = E.label(uid="polygons").affine(rotation=(0, 0, 45.0)).repeat().repeat().fill()
    g2 = G.text()
    p2 = E.affine().fill()
    return L.layer(geometry=p1(g1)), p2(g2)


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
