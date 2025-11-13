from api import E, G, run


def draw(t: float):
    g = G.sphere().scale(200, 200, 200).translate(74, 110).rotate()
    p1 = E.pipeline.affine().collapse().collapse()
    p2 = E.pipeline.affine().collapse()
    return p1(g), p2(g)


if __name__ == "__main__":
    run(
        draw,
        canvas_size="A5",
        render_scale=10,
        use_midi=False,
        use_parameter_gui=True,
        workers=4,
        line_thickness=0.001,
    )
