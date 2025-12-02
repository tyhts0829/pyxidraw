from api import E, G, run


def draw(t: float):
    poly = G.grid()
    effect = E.affine().subdivide().displace(t_sec=t * 0.1)
    return effect(poly)


if __name__ == "__main__":
    run(draw, canvas_size="SQUARE_300", render_scale=4)
