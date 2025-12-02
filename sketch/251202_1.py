from api import E, G, L, run


def draw(t: float):
    TITLE = "Pyxidraw"
    title = G.text(text=TITLE).label(uid="title")
    title_e = E.affine().fill(density=30).label(uid="title_effect")
    title_l = L(title_e(title), name="title")

    COPY1 = "line-based generative geometry,  python-native creative coding framework, modulate everything via effects pipeline"
    copy1 = G.text(text=COPY1).label(uid="copy1")
    copy1_e = E.affine().fill(density=20).label(uid="copy1_effect")
    copy1_l = L(copy1_e(copy1), name="copy1")

    EXPLANATION = "G.Shape()"
    explanation = G.text(text=EXPLANATION).label(uid="explanation")
    explanation_e = E.affine().fill(density=5).label(uid="explanation_effect")
    explanation_l = L(explanation_e(explanation), name="explanation")

    g1 = G.polyhedron().label(uid="g1").rotate(x=0.2, y=0.3, z=0.1)
    g1_e = E.affine().label(uid="g1_effect")
    g1_l = L(g1_e(g1), name="g1")

    g2_e = E.affine().fill().label(uid="g2_effect")
    g2_l = L(g2_e(g1), name="g2")

    g3_e = E.affine().fill(density=10).subdivide().displace().label(uid="g3_effect")
    g3_l = L(g3_e(g1), name="g3")

    h_line = G.line()
    h_line_e = E.affine().label(uid="h_line_effect")
    h_line_l = L(h_line_e(h_line), name="h_line")

    return title_l, copy1_l, g1_l, g2_l, g3_l, explanation_l, h_line_l


if __name__ == "__main__":
    run(draw, canvas_size="A4", render_scale=6.5, show_hud=False)
