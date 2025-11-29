from api import G, run

A5 = (148, 210)


def draw(t: float):
    sh = G.polygon().label()
    return sh


if __name__ == "__main__":
    run(draw, canvas_size="A5")
