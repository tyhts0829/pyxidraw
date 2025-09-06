from __future__ import annotations

import logging
import os
from typing import Mapping

import numpy as np

from api import E, G, run
from engine.core.geometry import Geometry

CANVS_SIZE = 400


def draw(t: float, cc: Mapping[int, float]) -> Geometry:
    t = t * cc[9] * 10
    print(cc)
    sphere = G.sphere(subdivisions=cc[1], sphere_type=cc[2])
    pipe = (
        E.pipeline.rotate(angles_rad=(cc[3], cc[4], cc[5]), pivot=(CANVS_SIZE // 2, CANVS_SIZE // 2, 0))
        .displace(amplitude_mm=cc[6] * 50, spatial_freq=(cc[7] * 0.01, cc[7] * 0.01, cc[7] * 0.01), t_sec=t)
        .displace(amplitude_mm=cc[6] * 20, spatial_freq=(cc[7] * 0.05, cc[7] * 0.05, cc[7] * 0.05), t_sec=t * 2)
        .build()
    )
    return pipe(sphere.scale(400 * cc[8]).translate(CANVS_SIZE // 2, CANVS_SIZE // 2, 0))


if __name__ == "__main__":
    run(
        draw,
        canvas_size=(CANVS_SIZE, CANVS_SIZE),
        render_scale=4,
        use_midi=True,
    )
