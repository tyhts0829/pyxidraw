from api import E, G, run

A5 = (148, 210)


def draw(t: float):
    g1 = G.polyhedron().scale(40, 40, 40)
    p1 = E.pipeline.affine()
    g2 = G.polyhedron().scale(30, 30, 30)
    p2 = E.pipeline.affine()
    g3 = G.polyhedron().scale(20, 20, 20)
    p3 = E.pipeline.affine().subdivide().displace()
    return p1(g1) + p2(g2) + p3(g3)


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
