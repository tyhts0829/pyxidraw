from api import E, G, run

A5 = (148, 210)


def draw(t: float):
    sh = G.polygon().label(uid="sphere")
    shp = E.affine().fill().subdivide().displace()
    return shp(sh)


if __name__ == "__main__":
    run(draw, canvas_size="A5", render_scale=6)
