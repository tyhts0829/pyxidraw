from api import E, G, L, run

A5 = (148, 210)


def draw(t: float):
    g1_clip = G.polygon().label(uid="g1_clip_shape")
    g1_clip_p = E.affine().label(uid="g1_clip_effect")

    g1 = G.polygon().label(uid="g1_shape")
    g1_p = E.affine().fill().displace().clip(outline=g1_clip_p(g1_clip)).label(uid="g1_effect")
    g1_l = L(g1_p(g1), name="g1_layer")

    return g1_l


if __name__ == "__main__":
    run(draw, canvas_size="A5", render_scale=10, show_hud=False)
