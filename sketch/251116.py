from api import E, G, cc, run

A5 = (148, 210)


def draw(t: float):
    g1 = G.polygon().scale(40, 40, 40).translate(A5[0] / 2, A5[1] / 2, 0)
    p1 = (
        E.affine()
        .repeat()
        .dash(
            dash_length=[cc[1] * 10, cc[2] * 10, cc[3] * 10],
            gap_length=[cc[4] * 10, cc[5] * 10, cc[6] * 10],
            offset=[cc[7] * 10, cc[8] * 10, cc[9] * 10],
        )
        .offset()
        .fill()
    )
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
        show_hud=True,
    )
