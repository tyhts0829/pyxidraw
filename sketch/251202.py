from api import E, G, L, run

A5 = (148, 210)


def draw(t: float):
    g1 = G.text().label(uid="g1_shape")
    g1_p = E.affine().fill().label(uid="g1_effect")
    g1_l = L(g1_p(g1), name="g1_layer")

    g2 = G.text().label(uid="g2_shape")
    g2_p = E.affine().fill().label(uid="g2_effect")
    g2_l = L(g2_p(g2), name="g2_layer")

    g3 = G.text().label(uid="g3_shape")
    g3_p = E.affine().fill().label(uid="g3_effect")
    g3_l = L(g3_p(g3), name="g3_layer")

    title = G.text().label(uid="title_shape")
    title_p = E.affine().fill().label(uid="title_effect")
    title_l = L(title_p(title), name="title_layer")

    subtitle = G.text().label(uid="subtitle_shape")
    subtitle_p = E.affine().repeat().label(uid="subtitle_effect")
    subtitle_l = L(subtitle_p(subtitle), name="subtitle_layer")

    square1 = G.polygon(n_sides=4).label(uid="square1_shape")
    square1_p = E.affine().fill().label(uid="square1_effect")
    square1_l = L(square1_p(square1), name="square1_layer")

    square2 = G.polygon(n_sides=4).label(uid="square2_shape")
    square2_p = E.affine().fill().label(uid="square2_effect")
    square2_l = L(square2_p(square2), name="square2_layer")

    square3 = G.polygon(n_sides=4).label(uid="square3_shape")
    square3_p = E.affine().fill().label(uid="square3_effect")
    square3_l = L(square3_p(square3), name="square3_layer")

    return g1_l, g2_l, g3_l, title_l, subtitle_l, square1_l, square2_l, square3_l


if __name__ == "__main__":
    run(draw, canvas_size="A5", render_scale=10, show_hud=False)
