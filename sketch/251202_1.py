from api import E, G, L, run


def draw(t: float):
    title = G.text()
    title_e = E.affine().fill()
    title_l = L(title_e(title))

    g1 = G.polyhedron()
    g1_e = E.affine().repeat().repeat()
    g1_l = L(g1_e(g1))

    return title_l, g1_l


if __name__ == "__main__":
    run(draw, canvas_size="A5_LANDSCAPE", render_scale=10, show_hud=False)
