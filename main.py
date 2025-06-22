import math

import arc
import numpy as np

from api import effects, shapes
from api.runner import run_sketch
from util.constants import CANVAS_SIZES

cw, ch = CANVAS_SIZES["SQUARE_200"]


def draw(t, cc) -> list[np.ndarray]:
    # Demonstrate new shape and effect system

    # Use polygon shape with number of sides controlled by MIDI
    polyh = shapes.polyhedron(
        polygon_type="dodeca",
        center=(cw / 2, ch / 2, 0),
        scale=(80, 80, 80),
        # rotate=(math.sin(t / 10) * 5, math.sin(t / 10) * 5, math.sin(t / 10) * 5),
    )
    scale = (cc[1], cc[1], cc[1])  # Use MIDI CC to control scale
    rotate = (cc[2], cc[2], cc[2])  # Use MIDI CC to control rotation
    offset = (cc[3] * 50, cc[3] * 50, cc[3] * 50)  # Use MIDI CC to control offset
    polyh = effects.array(polyh, center=(cw / 2, ch / 2, 0), scale=scale, rotate=rotate, offset=offset)
    # polyとpolyhを組み合わせて描画
    ret = []
    ret.extend(polyh)
    return ret


if __name__ == "__main__":
    arc.start(midi=True)
    run_sketch(draw, canvas_size=CANVAS_SIZES["SQUARE_200"], render_scale=8, background=(1, 1, 1, 1))
    arc.stop()
