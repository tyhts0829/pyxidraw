from api import E, G, run


def draw(t: float):
    g1 = G.polygon().scale(100, 100, 1)
    p1 = E.pipeline.affine().fill().dash().subdivide().displace().offset().fill()
    return p1(g1)


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
