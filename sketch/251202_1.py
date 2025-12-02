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

    METHOD = "G.Shape()"
    method = G.text(text=METHOD).label(uid="method")
    method_e = E.affine().fill(density=5).label(uid="method_effect")
    method_l = L(method_e(method), name="method")

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

    COPY2 = """
    This framework approaches visual design with an audio mindset.
    A minimal, line-based geometry engine keeps the representation intentionally simple, treating constraints as a source of creativity rather than a limitation. Instead of hiding structure and styling decisions inside a black-box renderer, pyxidraw keeps them close to your code: you build multi-layer sketches where each layer can carry its own color and line weight, echoing pen changes in a plotter. Effects are composed as method-chained processors, forming an effect chain that feels closer to a synth and pedalboard than a monolithic graphics API. MIDI control and LFO-driven modulation keep parameters in constant motion, making geometry something you can “play” rather than merely render. From real-time OpenGL preview to pen-plotter-ready G-code, pyxidraw offers a continuous path from experimental patch to physical output, with new Shapes and Effects defined as lightweight Python decorators. The aim is not just to produce images, but to compose line-based scores that unfold in time, on screen and on paper.
    """
    copy2 = G.text(text=COPY2).label(uid="copy2")
    copy2_e = E.affine().fill().label(uid="copy2_effect")
    copy2_l = L(copy2_e(copy2), name="copy2")

    grid_background = G.grid().label(uid="grid_background")
    grid_background_e = E.affine().label(uid="grid_background_effect")
    grid_background_l = L(grid_background_e(grid_background), name="grid_background")

    circle = G.text().label(uid="circle")
    circle_e = E.affine().fill().label(uid="circle_effect")
    circle_l = L(circle_e(circle), name="circle")

    box = G.polygon(phase=45).label(uid="box")
    box_e = E.affine().fill().repeat().rotate().label(uid="box_effect")
    box_l = L(box_e(box), name="box")

    FEATURES = "GEOMETRIES | EFFECTS | LAYERS | MIDI | LFO | G-CODE"
    features = G.text(text=FEATURES).label(uid="features")
    features_e = E.affine().fill().label(uid="features_effect")
    features_l = L(features_e(features), name="features")

    return (
        title_l,
        copy1_l,
        g1_l,
        g2_l,
        g3_l,
        method_l,
        h_line_l,
        copy2_l,
        grid_background_l,
        circle_l,
        box_l,
        features_l,
    )


if __name__ == "__main__":
    run(draw, canvas_size="A4", render_scale=6.5, show_hud=False)
